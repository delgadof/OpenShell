// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Vendor-neutral runtime certificate provisioning through cert-manager.

use base64::{Engine as _, engine::general_purpose::STANDARD};
use kube::{
    Api, Client, ResourceExt,
    api::{ApiResource, DeleteParams, PostParams, Preconditions},
    core::{DynamicObject, GroupVersionKind},
};
use openshell_core::SandboxSessionId;
use openshell_sandbox_backend::boundary_protocol::{
    SandboxTlsMaterial, generate_sandbox_tls_material,
};
use rcgen::{CertificateParams, DnType, ExtendedKeyUsagePurpose, KeyPair, KeyUsagePurpose};
use rustls::{
    RootCertStore,
    client::{WebPkiServerVerifier, danger::ServerCertVerifier},
    pki_types::{PrivatePkcs8KeyDer, ServerName, UnixTime},
};
use serde_json::{Value, json};
use std::{
    io::Cursor,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use x509_parser::{extensions::GeneralName, parse_x509_certificate};

use crate::config::{KubernetesSandboxRuntimeConfig, RuntimeCertificateConfig};

const MAX_PEM_BYTES: usize = 64 * 1024;
pub const REQUEST_CLEANUP_TIMEOUT: Duration = Duration::from_secs(10);

/// Preserve local issuance unless the operator explicitly configures cert-manager.
/// External issuance errors never fall back to a locally generated certificate.
pub async fn provision_runtime_tls(
    client: Client,
    config: &KubernetesSandboxRuntimeConfig,
    session_id: SandboxSessionId,
) -> Result<SandboxTlsMaterial, String> {
    let Some(config) = &config.cert_manager else {
        return generate_sandbox_tls_material(session_id).map_err(|error| error.to_string());
    };
    config.validate()?;
    let trust = read_trust_bundle(&config.trust_bundle_path).await?;
    let name = format!("sandbox.{session_id}.openshell.internal");
    let key = KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)
        .map_err(|error| format!("generate runtime key: {error}"))?;
    let request = certificate_request(config, session_id, &name, &key)?;
    let resource = ApiResource::from_gvk(&GroupVersionKind::gvk(
        "cert-manager.io",
        "v1",
        "CertificateRequest",
    ));
    let api: Api<DynamicObject> = Api::namespaced_with(client, &config.namespace, &resource);
    // Keep the CSR and name for recovering ownership if POST's response is lost.
    let request_name = request.name_any();
    let mut request_uid = None;
    let result = tokio::time::timeout(Duration::from_secs(config.timeout_seconds), async {
        let created = api
            .create(&PostParams::default(), &request)
            .await
            .map_err(|error| api_error("create", &error))?;
        request_uid = Some(owned_request_uid(&created, &request)?);
        loop {
            let current = match api.get(&request_name).await {
                Ok(current) => current,
                Err(error) if retryable_api_error(&error) => {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    continue;
                }
                Err(error) => return Err(api_error("read", &error)),
            };
            if Some(owned_request_uid(&current, &request)?) != request_uid {
                return Err("runtime CertificateRequest was replaced during issuance".to_string());
            }
            if let Some(certificate) = issued_certificate(&current.data["status"])? {
                validate_issued_certificate(
                    &certificate,
                    &key,
                    &name,
                    &trust,
                    config.minimum_validity_seconds,
                )?;
                tracing::info!(session_id = %session_id, certificate_request = %request_name,
                    "validated externally issued runtime certificate");
                return Ok(SandboxTlsMaterial {
                    server_name: name,
                    trust_anchor_pem: trust,
                    certificate_chain_pem: certificate,
                    private_key_pem: key.serialize_pem(),
                });
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    })
    .await
    .map_err(|_| "runtime certificate issuance timed out".to_string())
    .and_then(|result| result);
    // Requests contain only public material. Cancellation or a process crash
    // can leave a labeled request; cleanup failures must not mask issuance errors.
    match tokio::time::timeout(
        REQUEST_CLEANUP_TIMEOUT,
        remove_request(&api, &request, request_uid),
    )
    .await
    {
        Ok(Ok(())) => {}
        _ => {
            tracing::warn!(certificate_request = %request_name, "runtime CertificateRequest cleanup failed");
        }
    }
    result
}

async fn read_trust_bundle(path: &str) -> Result<String, String> {
    let path = path.to_string();
    // Reuse the shared regular-file reader: it rejects devices/FIFOs and bounds
    // allocation to 1 MiB. Apply the smaller runtime bundle limit below.
    let pem = tokio::task::spawn_blocking(move || {
        openshell_core::driver_utils::read_upstream_proxy_ca_bundle_file(
            &path,
            "sandbox_runtime.cert_manager.trust_bundle_path",
        )
    })
    .await
    .map_err(|_| "runtime trust bundle read task failed".to_string())??;
    let mut roots = RootCertStore::empty();
    for certificate in certificate_pem(&pem)? {
        roots
            .add(certificate)
            .map_err(|_| "invalid runtime trust anchor".to_string())?;
    }
    Ok(pem)
}

/// cert-manager adds requester identity fields to spec. Compare every field we
/// submitted, while allowing those server-owned additions.
fn owned_request_uid(current: &DynamicObject, expected: &DynamicObject) -> Result<String, String> {
    let same_spec = expected.data["spec"].as_object().is_some_and(|spec| {
        spec.iter().all(|(key, value)| {
            // Go's optional bool serialization may omit isCA=false.
            (key == "isCA" && value == false && current.data["spec"].get(key).is_none())
                || current.data["spec"].get(key) == Some(value)
        })
    });
    if current.metadata.name != expected.metadata.name
        || current.metadata.namespace != expected.metadata.namespace
        || current.metadata.deletion_timestamp.is_some()
        || !same_spec
    {
        return Err("runtime CertificateRequest identity or spec changed".to_string());
    }
    current
        .uid()
        .filter(|uid| !uid.is_empty())
        .ok_or_else(|| "runtime CertificateRequest has no UID".to_string())
}

async fn remove_request(
    api: &Api<DynamicObject>,
    expected: &DynamicObject,
    uid: Option<String>,
) -> Result<(), kube::Error> {
    let name = expected.name_any();
    let uid = if let Some(uid) = uid {
        uid
    } else {
        let Some(current) = api.get_opt(&name).await? else {
            return Ok(());
        };
        let Ok(uid) = owned_request_uid(&current, expected) else {
            // A rejected create must never delete someone else's request.
            return Ok(());
        };
        uid
    };
    let params = DeleteParams::default().preconditions(Preconditions {
        uid: Some(uid),
        resource_version: None,
    });
    match api.delete(&name, &params).await {
        Ok(_) | Err(kube::Error::Api(kube::error::ErrorResponse { code: 404, .. })) => Ok(()),
        Err(error) => Err(error),
    }
}

fn retryable_api_error(error: &kube::Error) -> bool {
    match error {
        kube::Error::Api(response) => response.code == 429 || response.code >= 500,
        kube::Error::Service(_) | kube::Error::HyperError(_) => true,
        _ => false,
    }
}

fn api_error(operation: &str, error: &kube::Error) -> String {
    // Admission responses may contain operator credentials or the CSR. Retain
    // the operation and HTTP status, never the untrusted response body.
    match error {
        kube::Error::Api(response) => format!(
            "{operation} runtime CertificateRequest: Kubernetes API returned HTTP {}",
            response.code
        ),
        _ => format!("{operation} runtime CertificateRequest: Kubernetes API request failed"),
    }
}

fn certificate_request(
    config: &RuntimeCertificateConfig,
    session_id: SandboxSessionId,
    name: &str,
    key: &KeyPair,
) -> Result<DynamicObject, String> {
    let mut params =
        CertificateParams::new(vec![name.to_string()]).map_err(|error| error.to_string())?;
    params.distinguished_name.push(DnType::CommonName, name);
    params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
    params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
    let csr = params
        .serialize_request(key)
        .and_then(|request| request.pem())
        .map_err(|error| format!("build runtime CSR: {error}"))?;
    // Emit Go's canonical whole-second duration form so a typed API client
    // round trip cannot change the spec merely by normalizing its spelling.
    let seconds = config.duration_seconds;
    let duration = if seconds >= 3600 {
        format!(
            "{}h{}m{}s",
            seconds / 3600,
            seconds % 3600 / 60,
            seconds % 60
        )
    } else {
        format!("{}m{}s", seconds / 60, seconds % 60)
    };
    // A fresh suffix permits retries of one durable session without reusing keys.
    serde_json::from_value(json!({
        "apiVersion": "cert-manager.io/v1",
        "kind": "CertificateRequest",
        "metadata": {
            "name": format!("runtime-{}", SandboxSessionId::new()),
            "namespace": config.namespace,
            "labels": {
                "openshell.ai/runtime-session": session_id.to_string(),
                "app.kubernetes.io/managed-by": "openshell"
            }
        },
        "spec": {
            "request": STANDARD.encode(csr),
            "isCA": false,
            "duration": duration,
            "usages": ["digital signature", "server auth"],
            "issuerRef": {
                "name": config.issuer_ref.name,
                "kind": config.issuer_ref.kind,
                "group": config.issuer_ref.group
            }
        }
    }))
    .map_err(|error| format!("build runtime CertificateRequest: {error}"))
}

fn issued_certificate(status: &Value) -> Result<Option<String>, String> {
    let conditions = status["conditions"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or_default();
    let is_true = |kind: &str| {
        conditions
            .iter()
            .any(|c| c["type"] == kind && c["status"] == "True")
    };
    if is_true("Denied")
        || is_true("InvalidRequest")
        || conditions
            .iter()
            .any(|c| c["type"] == "Ready" && c["status"] == "False" && c["reason"] == "Failed")
    {
        return Err("runtime CertificateRequest denied, invalid, or failed".to_string());
    }
    if !is_true("Ready") {
        return Ok(None);
    }
    if !is_true("Approved") {
        return Err("runtime CertificateRequest became Ready without approval".to_string());
    }
    let encoded = status["certificate"]
        .as_str()
        .filter(|value| !value.is_empty() && value.len() <= MAX_PEM_BYTES * 2)
        .ok_or_else(|| "runtime CertificateRequest has no usable certificate".to_string())?;
    let pem = STANDARD
        .decode(encoded)
        .map_err(|_| "invalid certificate encoding".to_string())?;
    if pem.len() > MAX_PEM_BYTES {
        return Err("runtime certificate chain too large".to_string());
    }
    String::from_utf8(pem)
        .map(Some)
        .map_err(|_| "runtime certificate is not PEM text".to_string())
}

fn certificate_pem(pem: &str) -> Result<Vec<rustls::pki_types::CertificateDer<'static>>, String> {
    if pem.is_empty() || pem.len() > MAX_PEM_BYTES {
        return Err("certificate PEM is empty or too large".to_string());
    }
    let mut certificates = Vec::new();
    let mut remaining = pem.trim();
    while !remaining.is_empty() {
        // CA exports can prefix certificates with public descriptive metadata.
        // Do not let a metadata line hide another kind of PEM boundary.
        if let Some((line, rest)) = remaining.split_once('\n')
            && (line.starts_with("Subject:") || line.starts_with("Issuer:"))
            && !line.contains("-----")
        {
            remaining = rest.trim_start();
            continue;
        }
        if !remaining.starts_with("-----BEGIN CERTIFICATE-----") {
            return Err("certificate bundle must contain only PEM certificates".to_string());
        }
        let end = remaining
            .find("-----END CERTIFICATE-----")
            .ok_or_else(|| "unterminated certificate PEM".to_string())?
            + "-----END CERTIFICATE-----".len();
        let item = rustls_pemfile::read_one(&mut Cursor::new(&remaining[..end]))
            .map_err(|_| "invalid certificate PEM".to_string())?
            .ok_or_else(|| "empty certificate PEM block".to_string())?;
        match item {
            rustls_pemfile::Item::X509Certificate(certificate) => {
                let (trailing, _) = parse_x509_certificate(&certificate)
                    .map_err(|_| "invalid certificate DER".to_string())?;
                if !trailing.is_empty() {
                    return Err("unexpected data after certificate DER".to_string());
                }
                certificates.push(certificate);
            }
            _ => {
                return Err(
                    "certificate bundle must contain only certificates, never keys".to_string(),
                );
            }
        }
        remaining = remaining[end..].trim_start();
    }
    if certificates.is_empty() || certificates.len() > 16 {
        return Err("certificate bundle has invalid certificate count".to_string());
    }
    Ok(certificates)
}

fn validate_issued_certificate(
    certificate: &str,
    key: &KeyPair,
    server_name: &str,
    trust: &str,
    minimum_validity_seconds: u64,
) -> Result<(), String> {
    let chain = certificate_pem(certificate)?;
    let leaf = chain
        .first()
        .ok_or_else(|| "empty runtime certificate chain".to_string())?;
    let (_, parsed) =
        parse_x509_certificate(leaf).map_err(|_| "invalid runtime leaf certificate".to_string())?;
    if parsed.is_ca() {
        return Err("runtime leaf must not be a CA".to_string());
    }
    let san = parsed
        .subject_alternative_name()
        .map_err(|_| "invalid runtime SAN".to_string())?
        .ok_or_else(|| "runtime certificate requires a SAN".to_string())?;
    if san.value.general_names.as_slice() != [GeneralName::DNSName(server_name)] {
        return Err(
            "runtime certificate must contain only the requested session DNS SAN".to_string(),
        );
    }
    let eku = parsed
        .extended_key_usage()
        .map_err(|_| "invalid runtime EKU".to_string())?
        .ok_or_else(|| "runtime certificate requires serverAuth EKU".to_string())?;
    if !eku.value.server_auth
        || eku.value.any
        || eku.value.client_auth
        || eku.value.code_signing
        || eku.value.email_protection
        || eku.value.time_stamping
        || eku.value.ocsp_signing
        || !eku.value.other.is_empty()
    {
        return Err("runtime certificate requires only serverAuth EKU".to_string());
    }
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let certified_key = rustls::sign::CertifiedKey::from_der(
        chain.clone(),
        PrivatePkcs8KeyDer::from(key.serialize_der()).into(),
        &provider,
    )
    .map_err(|_| "runtime certificate key mismatch".to_string())?;
    certified_key
        .keys_match()
        .map_err(|_| "runtime certificate key mismatch".to_string())?;
    let mut roots = RootCertStore::empty();
    for root in certificate_pem(trust)? {
        roots
            .add(root)
            .map_err(|_| "invalid runtime trust anchor".to_string())?;
    }
    let verifier = WebPkiServerVerifier::builder_with_provider(Arc::new(roots), provider)
        .build()
        .map_err(|_| "cannot construct runtime trust verifier".to_string())?;
    let name =
        ServerName::try_from(server_name).map_err(|_| "invalid runtime server name".to_string())?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "invalid system time".to_string())?;
    // Verify the whole chain now and at the minimum required remaining lifetime.
    for time in [now, now + Duration::from_secs(minimum_validity_seconds)] {
        verifier
            .verify_server_cert(
                leaf,
                &chain[1..],
                &name,
                &[],
                UnixTime::since_unix_epoch(time),
            )
            .map_err(|error| {
                format!("runtime certificate trust, identity, or validity check failed: {error}")
            })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RuntimeCertificateIssuerRef;
    use http_body_util::BodyExt as _;
    use rcgen::{BasicConstraints, Certificate, CertificateParams, ExtendedKeyUsagePurpose, IsCa};
    use serde_json::json;
    use std::{
        convert::Infallible,
        io::{Read as _, Write as _},
        sync::Mutex,
    };

    fn assert_tls_handshake(material: &SandboxTlsMaterial) {
        let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
        let key = KeyPair::from_pem(&material.private_key_pem).unwrap();
        let server = rustls::ServerConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(
                certificate_pem(&material.certificate_chain_pem).unwrap(),
                PrivatePkcs8KeyDer::from(key.serialize_der()).into(),
            )
            .unwrap();
        let mut roots = RootCertStore::empty();
        for root in certificate_pem(&material.trust_anchor_pem).unwrap() {
            roots.add(root).unwrap();
        }
        let client = rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_root_certificates(roots)
            .with_no_client_auth();
        let mut server = rustls::ServerConnection::new(Arc::new(server)).unwrap();
        let mut client = rustls::ClientConnection::new(
            Arc::new(client),
            ServerName::try_from(material.server_name.clone()).unwrap(),
        )
        .unwrap();
        for _ in 0..16 {
            let mut data = Vec::new();
            client.write_tls(&mut data).unwrap();
            server.read_tls(&mut Cursor::new(data)).unwrap();
            server.process_new_packets().unwrap();
            let mut data = Vec::new();
            server.write_tls(&mut data).unwrap();
            client.read_tls(&mut Cursor::new(data)).unwrap();
            client.process_new_packets().unwrap();
            if !client.is_handshaking() && !server.is_handshaking() {
                client.writer().write_all(b"runtime TLS").unwrap();
                let mut data = Vec::new();
                client.write_tls(&mut data).unwrap();
                server.read_tls(&mut Cursor::new(data)).unwrap();
                server.process_new_packets().unwrap();
                let mut received = [0; 11];
                server.reader().read_exact(&mut received).unwrap();
                assert_eq!(&received, b"runtime TLS");
                return;
            }
        }
        panic!("issued material did not complete a TLS handshake");
    }

    /// Creates and removes a real `CertificateRequest` using an operator-configured
    /// cluster and issuer. Configuration contains public trust and issuer names;
    /// enrollment credentials remain with the issuer controller.
    #[tokio::test]
    #[ignore = "requires an explicitly configured Kubernetes context and cert-manager issuer"]
    async fn live_runtime_certificate_issuance() {
        let config_path = std::env::var("OPENSHELL_RUNTIME_CERT_MANAGER_TEST_CONFIG")
            .expect("set OPENSHELL_RUNTIME_CERT_MANAGER_TEST_CONFIG to a runtime config JSON file");
        let context = std::env::var("OPENSHELL_RUNTIME_CERT_MANAGER_TEST_CONTEXT")
            .expect("set OPENSHELL_RUNTIME_CERT_MANAGER_TEST_CONTEXT to a test cluster context");
        let runtime: KubernetesSandboxRuntimeConfig =
            serde_json::from_slice(&std::fs::read(config_path).unwrap()).unwrap();
        let external = runtime
            .cert_manager
            .as_ref()
            .expect("configure cert_manager");
        let client = Client::try_from(
            kube::Config::from_kubeconfig(&kube::config::KubeConfigOptions {
                context: Some(context),
                ..Default::default()
            })
            .await
            .unwrap(),
        )
        .unwrap();
        let session = SandboxSessionId::new();
        let material = provision_runtime_tls(client.clone(), &runtime, session)
            .await
            .expect("live issuance and certificate validation");
        assert_tls_handshake(&material);
        let certificates = certificate_pem(&material.certificate_chain_pem).unwrap();
        let (_, leaf) = parse_x509_certificate(&certificates[0]).unwrap();
        println!(
            "Runtime certificate and TLS handshake verified: issuer={}, serial={}, DNS={}",
            leaf.issuer(),
            leaf.raw_serial_as_string(),
            material.server_name
        );
        let resource = ApiResource::from_gvk(&GroupVersionKind::gvk(
            "cert-manager.io",
            "v1",
            "CertificateRequest",
        ));
        let api: Api<DynamicObject> = Api::namespaced_with(client, &external.namespace, &resource);
        let remaining = api
            .list(
                &kube::api::ListParams::default()
                    .labels(&format!("openshell.ai/runtime-session={session}")),
            )
            .await
            .unwrap();
        assert!(
            remaining.items.is_empty(),
            "consumed request must be removed"
        );
    }

    async fn exercise_issuer(outcome: &str) -> (Result<SandboxTlsMaterial, String>, Vec<String>) {
        let timeout_seconds = if outcome == "pending" { 1 } else { 5 };
        let (ca, ca_key) = ca();
        let mut trust = tempfile::NamedTempFile::new().unwrap();
        if outcome == "invalid-trust" {
            trust
                .write_all(b"-----BEGIN CERTIFICATE-----\nYmFk\n-----END CERTIFICATE-----\n")
                .unwrap();
        } else {
            trust.write_all(ca.pem().as_bytes()).unwrap();
        }
        let session = SandboxSessionId::new();
        let expected_name = format!("sandbox.{session}.openshell.internal");
        let calls = Arc::new(Mutex::new(Vec::new()));
        let returned = Arc::new(Mutex::new(json!(null)));
        let captured_calls = calls.clone();
        let outcome = outcome.to_string();
        let ca = Arc::new(ca);
        let ca_key = Arc::new(ca_key);
        let service = tower::service_fn(move |request: http::Request<kube::client::Body>| {
            let (ca, ca_key, expected_name, outcome, returned, calls) = (
                ca.clone(),
                ca_key.clone(),
                expected_name.clone(),
                outcome.clone(),
                returned.clone(),
                captured_calls.clone(),
            );
            async move {
                let method = request.method().to_string();
                assert_ne!(
                    outcome, "invalid-trust",
                    "invalid trust must fail before API access"
                );
                assert!(request.uri().path().starts_with(
                    "/apis/cert-manager.io/v1/namespaces/gateway-system/certificaterequests"
                ));
                calls.lock().unwrap().push(method.clone());
                let mut code = 200;
                let response = match method.as_str() {
                    "POST" => {
                        let bytes = request.into_body().collect().await.unwrap().to_bytes();
                        let mut submitted: Value = serde_json::from_slice(&bytes).unwrap();
                        submitted["metadata"]["uid"] = json!("created-request-uid");
                        assert!(!String::from_utf8_lossy(&bytes).contains("PRIVATE KEY"));
                        assert_eq!(
                            submitted["spec"]["issuerRef"],
                            json!({"name":"runtime","kind":"Issuer","group":"example.com"})
                        );
                        assert_eq!(submitted["spec"]["isCA"], false);
                        // Typed cert-manager API clients normalize Go durations
                        // and may omit optional false booleans when serializing.
                        if outcome == "normalized-spec" {
                            submitted["spec"]["duration"] = json!("2h0m0s");
                            submitted["spec"].as_object_mut().unwrap().remove("isCA");
                        }
                        assert_eq!(
                            submitted["spec"]["usages"],
                            json!(["digital signature", "server auth"])
                        );
                        let pem = String::from_utf8(
                            STANDARD
                                .decode(submitted["spec"]["request"].as_str().unwrap())
                                .unwrap(),
                        )
                        .unwrap();
                        let csr = rcgen::CertificateSigningRequestParams::from_pem(&pem).unwrap();
                        assert_eq!(
                            csr.params.subject_alt_names,
                            vec![rcgen::SanType::DnsName(expected_name.try_into().unwrap())]
                        );
                        let signed = csr.signed_by(&ca, &ca_key).unwrap();
                        submitted["status"] = match outcome.as_str() {
                            "ready" | "replaced" | "changed-spec" | "retry-get"
                            | "normalized-spec" => {
                                json!({"conditions":[{"type":"Approved","status":"True"},{"type":"Ready","status":"True"}], "certificate":STANDARD.encode(signed.pem())})
                            }
                            "denied" => json!({"conditions":[{"type":"Denied","status":"True"}]}),
                            _ => {
                                json!({"conditions":[{"type":"Ready","status":"False","reason":"Pending"}]})
                            }
                        };
                        let mut stored = submitted.clone();
                        if outcome == "replaced" {
                            stored["metadata"]["uid"] = json!("different-request-uid");
                        }
                        if outcome == "changed-spec" || outcome == "conflict" {
                            stored["spec"]["request"] = json!("unrelated-csr");
                        }
                        *returned.lock().unwrap() = stored;
                        if outcome == "unavailable" {
                            code = 503;
                            json!({"apiVersion":"v1","kind":"Status","status":"Failure","reason":"ServiceUnavailable","code":503,"message":"test unavailable"})
                        } else if outcome == "conflict" {
                            code = 409;
                            json!({"apiVersion":"v1","kind":"Status","status":"Failure","reason":"AlreadyExists","code":409,"message":"sensitive admission detail"})
                        } else {
                            code = 201;
                            submitted
                        }
                    }
                    "GET" => {
                        if outcome == "retry-get"
                            && calls
                                .lock()
                                .unwrap()
                                .iter()
                                .filter(|call| *call == "GET")
                                .count()
                                == 1
                        {
                            code = 503;
                            json!({"apiVersion":"v1","kind":"Status","status":"Failure","reason":"ServiceUnavailable","code":503,"message":"temporary outage"})
                        } else {
                            returned.lock().unwrap().clone()
                        }
                    }
                    "DELETE" => {
                        assert_ne!(outcome, "conflict", "must not delete an unrelated request");
                        let bytes = request.into_body().collect().await.unwrap().to_bytes();
                        let options: Value = serde_json::from_slice(&bytes).unwrap();
                        assert_eq!(options["preconditions"]["uid"], "created-request-uid");
                        json!({"apiVersion":"v1","kind":"Status","status":"Success","code":200})
                    }
                    _ => panic!("unexpected API method: {method}"),
                };
                Ok::<_, Infallible>(
                    http::Response::builder()
                        .status(code)
                        .header("content-type", "application/json")
                        .body(kube::client::Body::from(response.to_string().into_bytes()))
                        .unwrap(),
                )
            }
        });
        let config = KubernetesSandboxRuntimeConfig {
            cert_manager: Some(RuntimeCertificateConfig {
                namespace: "gateway-system".to_string(),
                issuer_ref: RuntimeCertificateIssuerRef {
                    name: "runtime".to_string(),
                    kind: "Issuer".to_string(),
                    group: "example.com".to_string(),
                },
                trust_bundle_path: trust.path().to_str().unwrap().to_string(),
                duration_seconds: 7200,
                minimum_validity_seconds: 1200,
                timeout_seconds,
            }),
            ..Default::default()
        };
        let result = provision_runtime_tls(Client::new(service, "default"), &config, session).await;
        let calls = calls.lock().unwrap().clone();
        (result, calls)
    }

    #[tokio::test]
    async fn provisions_a_real_csr_through_the_kubernetes_api_and_removes_the_request() {
        let (result, calls) = exercise_issuer("ready").await;
        let material = result.unwrap();
        assert!(material.certificate_chain_pem.contains("BEGIN CERTIFICATE"));
        assert!(material.private_key_pem.contains("BEGIN PRIVATE KEY"));
        assert_tls_handshake(&material);
        assert_eq!(calls, ["POST", "GET", "DELETE"]);
    }

    #[tokio::test]
    async fn denial_and_api_failure_never_fall_back_to_local_issuance() {
        for outcome in ["denied", "unavailable"] {
            let (result, calls) = exercise_issuer(outcome).await;
            assert!(result.is_err());
            assert_eq!(calls.last().unwrap(), "DELETE");
        }
    }

    #[tokio::test]
    async fn pending_issuance_times_out_and_removes_the_request() {
        let (result, calls) = exercise_issuer("pending").await;
        assert_eq!(
            result.unwrap_err(),
            "runtime certificate issuance timed out"
        );
        assert_eq!(calls.last().unwrap(), "DELETE");
    }

    #[tokio::test]
    async fn rejects_replaced_or_modified_requests() {
        for outcome in ["replaced", "changed-spec"] {
            let (result, _) = exercise_issuer(outcome).await;
            assert!(result.is_err(), "must reject {outcome}");
        }
    }

    #[tokio::test]
    async fn create_conflicts_never_delete_an_unrelated_request_or_expose_api_details() {
        let (result, calls) = exercise_issuer("conflict").await;
        let error = result.unwrap_err();
        assert!(!error.contains("sensitive admission detail"));
        assert!(!calls.iter().any(|method| method == "DELETE"));
    }

    #[tokio::test]
    async fn retries_a_transient_read_failure_within_the_issuance_deadline() {
        let (result, calls) = exercise_issuer("retry-get").await;
        result.unwrap();
        assert_eq!(calls, ["POST", "GET", "GET", "DELETE"]);
    }

    #[tokio::test]
    async fn accepts_spec_normalization_by_typed_cert_manager_clients() {
        let (result, calls) = exercise_issuer("normalized-spec").await;
        assert_tls_handshake(&result.unwrap());
        assert_eq!(calls, ["POST", "GET", "DELETE"]);
    }

    #[test]
    fn accepts_public_subject_and_issuer_metadata_around_pem_certificates() {
        let (ca, _) = ca();
        let pem = format!("Subject: CN=Test CA\nIssuer: CN=Test CA\n{}", ca.pem());
        assert_eq!(
            certificate_pem(&pem).unwrap(),
            certificate_pem(&ca.pem()).unwrap()
        );
    }

    #[tokio::test]
    async fn rejects_invalid_trust_before_contacting_the_issuer() {
        let (result, calls) = exercise_issuer("invalid-trust").await;
        assert!(result.is_err());
        assert!(calls.is_empty());
    }

    #[test]
    fn rejects_unknown_pem_blocks_and_trailing_text() {
        let (ca, _) = ca();
        for extra in [
            "-----BEGIN UNKNOWN-----\nYmFk\n-----END UNKNOWN-----\n",
            "unexpected trailing text",
        ] {
            assert!(certificate_pem(&(ca.pem() + extra)).is_err());
        }
    }

    #[tokio::test]
    async fn rejects_non_regular_and_oversized_trust_files() {
        let directory = tempfile::tempdir().unwrap();
        assert!(
            read_trust_bundle(directory.path().to_str().unwrap())
                .await
                .is_err()
        );
        #[cfg(unix)]
        assert!(
            read_trust_bundle("/dev/zero")
                .await
                .unwrap_err()
                .contains("not a regular file")
        );
        let (ca, _) = ca();
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(ca.pem().as_bytes()).unwrap();
        file.write_all(&vec![b' '; MAX_PEM_BYTES]).unwrap();
        assert!(
            read_trust_bundle(file.path().to_str().unwrap())
                .await
                .unwrap_err()
                .contains("too large")
        );
    }

    #[tokio::test]
    async fn unconfigured_driver_keeps_existing_local_issuance_without_api_calls() {
        let service = tower::service_fn(|_: http::Request<kube::client::Body>| async {
            panic!("default local issuance must not contact cert-manager");
            #[allow(unreachable_code)]
            Ok::<_, Infallible>(http::Response::new(kube::client::Body::empty()))
        });
        let session = SandboxSessionId::new();
        let material = provision_runtime_tls(
            Client::new(service, "default"),
            &KubernetesSandboxRuntimeConfig::default(),
            session,
        )
        .await
        .unwrap();
        assert_eq!(
            material.server_name,
            format!("sandbox.{session}.openshell.internal")
        );
        let key = KeyPair::from_pem(&material.private_key_pem).unwrap();
        validate_issued_certificate(
            &material.certificate_chain_pem,
            &key,
            &material.server_name,
            &material.trust_anchor_pem,
            1200,
        )
        .unwrap();
    }

    const NAME: &str = "sandbox.session-a.openshell.internal";

    fn ca() -> (Certificate, KeyPair) {
        let key = KeyPair::generate().unwrap();
        let mut params = CertificateParams::default();
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        (params.self_signed(&key).unwrap(), key)
    }

    fn leaf(params: CertificateParams, ca: &Certificate, ca_key: &KeyPair) -> (String, KeyPair) {
        let key = KeyPair::generate().unwrap();
        (params.signed_by(&key, ca, ca_key).unwrap().pem(), key)
    }

    fn server_params() -> CertificateParams {
        let mut params = CertificateParams::new(vec![NAME.to_string()]).unwrap();
        params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
        params
    }

    #[test]
    fn accepts_only_approved_ready_certificate_responses() {
        let status = json!({"conditions":[{"type":"Approved","status":"True"},
            {"type":"Ready","status":"True"}],"certificate":STANDARD.encode("public certificate")});
        assert_eq!(
            issued_certificate(&status).unwrap().as_deref(),
            Some("public certificate")
        );
        assert!(
            issued_certificate(
                &json!({"conditions":[{"type":"Ready","status":"False","reason":"Pending"}]})
            )
            .unwrap()
            .is_none()
        );
        for conditions in [
            json!([{"type":"Ready","status":"True"}]),
            json!([{"type":"Approved","status":"True"},{"type":"Denied","status":"True"},{"type":"Ready","status":"True"}]),
            json!([{"type":"Ready","status":"False","reason":"Failed"}]),
        ] {
            assert!(issued_certificate(&json!({"conditions":conditions,"certificate":STANDARD.encode("public certificate")})).is_err());
        }
    }

    #[test]
    fn accepts_matching_server_certificate_under_operator_trust() {
        let (ca, ca_key) = ca();
        let (cert, key) = leaf(server_params(), &ca, &ca_key);
        validate_issued_certificate(&cert, &key, NAME, &ca.pem(), 1200).unwrap();
    }

    #[test]
    fn rejects_wrong_key_wrong_session_and_untrusted_issuer() {
        let (ca, ca_key) = ca();
        let (other_ca, _) = self::ca();
        let (cert, key) = leaf(server_params(), &ca, &ca_key);
        assert!(
            validate_issued_certificate(
                &cert,
                &KeyPair::generate().unwrap(),
                NAME,
                &ca.pem(),
                1200
            )
            .is_err()
        );
        assert!(
            validate_issued_certificate(
                &cert,
                &key,
                "sandbox.session-b.openshell.internal",
                &ca.pem(),
                1200
            )
            .is_err()
        );
        assert!(validate_issued_certificate(&cert, &key, NAME, &other_ca.pem(), 1200).is_err());
        assert!(
            validate_issued_certificate(
                &cert,
                &key,
                NAME,
                &(ca.pem() + &ca_key.serialize_pem()),
                1200
            )
            .is_err()
        );
        assert!(
            validate_issued_certificate(
                &(cert + &key.serialize_pem()),
                &key,
                NAME,
                &ca.pem(),
                1200
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_expiry_ca_certificates_and_unrequested_identities() {
        let (ca, ca_key) = ca();
        let mut expired = server_params();
        expired.not_after = rcgen::date_time_ymd(2000, 1, 1);
        let mut extra_san = server_params();
        extra_san.subject_alt_names.push(rcgen::SanType::DnsName(
            "other.openshell.internal".try_into().unwrap(),
        ));
        let mut client = server_params();
        client.extended_key_usages = vec![ExtendedKeyUsagePurpose::ClientAuth];
        let mut unconstrained = server_params();
        unconstrained.extended_key_usages.clear();
        for params in [expired, extra_san, client, unconstrained] {
            let (cert, key) = leaf(params, &ca, &ca_key);
            assert!(validate_issued_certificate(&cert, &key, NAME, &ca.pem(), 1200).is_err());
        }
        assert!(validate_issued_certificate(&ca.pem(), &ca_key, NAME, &ca.pem(), 1200).is_err());
    }

    #[test]
    fn enforces_remaining_validity_before_releasing_the_runtime() {
        let (ca, ca_key) = ca();
        let mut params = server_params();
        params.not_after = rcgen::date_time_ymd(1970, 1, 1)
            + SystemTime::now().duration_since(UNIX_EPOCH).unwrap()
            + Duration::from_mins(10);
        let (certificate, key) = leaf(params, &ca, &ca_key);
        validate_issued_certificate(&certificate, &key, NAME, &ca.pem(), 300).unwrap();
        assert!(validate_issued_certificate(&certificate, &key, NAME, &ca.pem(), 1200).is_err());
    }
}
