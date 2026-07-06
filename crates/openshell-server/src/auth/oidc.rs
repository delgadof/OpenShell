// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! OIDC access-token authentication provider.
//!
//! Validates `authorization: Bearer <access_token>` headers either locally
//! against cached issuer JWKS keys or through the issuer's discovered
//! `UserInfo` endpoint. Produces an `Identity` that the authorization layer
//! (`authz.rs`) evaluates.
//!
//! This module owns authentication (verifying who the caller is).
//! Authorization (deciding what the caller can do) is in `authz.rs`.

use super::authenticator::Authenticator;
use super::identity::{Identity, IdentityProvider};
use super::principal::{Principal, UserPrincipal};
use async_trait::async_trait;
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use openshell_core::{OidcConfig, OidcTokenValidation};
use reqwest::Client;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tonic::Status;
use tracing::{debug, info, warn};

/// Path prefixes that bypass OIDC validation (gRPC reflection, health probes).
///
/// These are structural bypasses for gRPC infrastructure that doesn't map to a
/// single RPC method. Per-method bypasses (e.g. `Health`) are declared at the
/// handler with `#[rpc_auth(auth = "unauthenticated")]`.
const UNAUTHENTICATED_PREFIXES: &[&str] = &["/grpc.reflection.", "/grpc.health."];

/// Returns `true` if the method needs no authentication at all.
pub fn is_unauthenticated_method(path: &str) -> bool {
    super::method_authz::is_unauthenticated(path)
        || UNAUTHENTICATED_PREFIXES
            .iter()
            .any(|prefix| path.starts_with(prefix))
}

/// Cached JWKS key set fetched from the OIDC issuer.
///
/// A `refresh_mutex` ensures that only one refresh runs at a time,
/// preventing a "thundering herd" when the TTL expires or a new `kid`
/// is encountered under concurrent load.
pub struct JwksCache {
    keys: Arc<RwLock<HashMap<String, DecodingKey>>>,
    jwks_uri: String,
    ttl: Duration,
    last_refresh: Arc<RwLock<Instant>>,
    /// Serializes JWKS refresh operations so concurrent requests coalesce
    /// into a single HTTP fetch rather than stampeding the OIDC provider.
    refresh_mutex: tokio::sync::Mutex<()>,
    http: Client,
    userinfo_http: Client,
    config: OidcConfig,
    userinfo_endpoint: Option<String>,
    userinfo_cache: Arc<RwLock<HashMap<[u8; 32], CachedIdentity>>>,
}

#[derive(Debug, Clone)]
struct CachedIdentity {
    identity: Identity,
    expires_at: Instant,
}

impl std::fmt::Debug for JwksCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JwksCache")
            .field("jwks_uri", &self.jwks_uri)
            .field("ttl", &self.ttl)
            .finish()
    }
}

/// OIDC discovery document (subset of fields we need).
#[derive(Deserialize)]
struct OidcDiscovery {
    issuer: String,
    jwks_uri: String,
    #[serde(default)]
    userinfo_endpoint: Option<String>,
}

/// JWKS key set.
#[derive(Deserialize)]
struct JwkSet {
    keys: Vec<JwkKey>,
}

/// A single JWK key.
#[derive(Deserialize)]
struct JwkKey {
    kid: Option<String>,
    kty: String,
    #[serde(default)]
    n: String,
    #[serde(default)]
    e: String,
}

/// Claims extracted from a validated JWT.
#[derive(Debug, Deserialize)]
pub struct OidcClaims {
    pub sub: String,
    #[serde(default)]
    pub preferred_username: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub email: Option<String>,
    /// Roles extracted from the configurable claim path.
    #[serde(skip)]
    pub roles: Vec<String>,
    /// Raw claims for flexible role extraction.
    #[serde(flatten)]
    extra: serde_json::Value,
}

const STANDARD_OIDC_SCOPES: &[&str] = &["openid", "profile", "email", "offline_access"];

impl OidcClaims {
    /// Extract roles from the JWT claims using a dot-separated path.
    ///
    /// Supports paths like:
    /// - `realm_access.roles` (Keycloak)
    /// - `roles` (Entra ID)
    /// - `groups` (Okta)
    fn extract_roles(&mut self, roles_claim: &str) {
        let mut value = &self.extra;
        for segment in roles_claim.split('.') {
            match value.get(segment) {
                Some(v) => value = v,
                None => return,
            }
        }
        if let Some(arr) = value.as_array() {
            self.roles = arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }
    }

    /// Extract scopes from the JWT claims using a dot-separated path.
    ///
    /// Handles two formats:
    /// - Space-delimited string: `"openid sandbox:read sandbox:write"` (Keycloak, Entra)
    /// - JSON array: `["sandbox:read", "sandbox:write"]` (Okta)
    ///
    /// Filters out standard OIDC scopes (`openid`, `profile`, `email`, `offline_access`).
    fn extract_scopes(&self, scopes_claim: &str) -> Vec<String> {
        let mut value = &self.extra;
        for segment in scopes_claim.split('.') {
            match value.get(segment) {
                Some(v) => value = v,
                None => return vec![],
            }
        }

        let raw: Vec<String> = if let Some(s) = value.as_str() {
            s.split_whitespace().map(String::from).collect()
        } else if let Some(arr) = value.as_array() {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        } else {
            return vec![];
        };

        raw.into_iter()
            .filter(|s| !STANDARD_OIDC_SCOPES.contains(&s.as_str()))
            .collect()
    }
}

impl JwksCache {
    /// Create a new JWKS cache, discovering the JWKS URI and fetching the
    /// initial key set.
    pub async fn new(config: &OidcConfig) -> Result<Self, String> {
        let http = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| format!("failed to create HTTP client: {e}"))?;
        let userinfo_http = Client::builder()
            .timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| format!("failed to create UserInfo HTTP client: {e}"))?;

        // Discover JWKS URI from the OIDC discovery endpoint.
        let discovery_url = format!(
            "{}/.well-known/openid-configuration",
            config.issuer.trim_end_matches('/')
        );
        info!(url = %discovery_url, "Discovering OIDC configuration");

        let discovery: OidcDiscovery = http
            .get(&discovery_url)
            .send()
            .await
            .map_err(|e| format!("OIDC discovery request failed: {e}"))?
            .json()
            .await
            .map_err(|e| format!("OIDC discovery response parse failed: {e}"))?;

        // Validate the discovery document's issuer matches our configured issuer.
        let expected = config.issuer.trim_end_matches('/');
        let actual = discovery.issuer.trim_end_matches('/');
        if expected != actual {
            return Err(format!(
                "OIDC discovery issuer mismatch: expected '{expected}', got '{actual}'"
            ));
        }

        if config.token_validation == OidcTokenValidation::Userinfo
            && discovery.userinfo_endpoint.is_none()
        {
            return Err(
                "OIDC discovery document has no userinfo_endpoint required for userinfo token validation"
                    .to_string(),
            );
        }

        info!(jwks_uri = %discovery.jwks_uri, "OIDC JWKS URI discovered");

        let cache = Self {
            keys: Arc::new(RwLock::new(HashMap::new())),
            jwks_uri: discovery.jwks_uri,
            ttl: Duration::from_secs(config.jwks_ttl_secs),
            last_refresh: Arc::new(RwLock::new(
                Instant::now()
                    .checked_sub(Duration::from_secs(config.jwks_ttl_secs + 1))
                    .unwrap_or_else(Instant::now),
            )),
            refresh_mutex: tokio::sync::Mutex::new(()),
            http,
            userinfo_http,
            config: config.clone(),
            userinfo_endpoint: discovery.userinfo_endpoint,
            userinfo_cache: Arc::new(RwLock::new(HashMap::new())),
        };

        cache.refresh_keys().await?;
        Ok(cache)
    }

    /// Fetch the JWKS and update the cached keys.
    async fn refresh_keys(&self) -> Result<(), String> {
        debug!(uri = %self.jwks_uri, "Refreshing JWKS keys");

        let jwk_set: JwkSet = self
            .http
            .get(&self.jwks_uri)
            .send()
            .await
            .map_err(|e| format!("JWKS fetch failed: {e}"))?
            .json()
            .await
            .map_err(|e| format!("JWKS parse failed: {e}"))?;

        let mut new_keys = HashMap::new();
        for key in &jwk_set.keys {
            if key.kty != "RSA" {
                continue;
            }
            let Some(ref kid) = key.kid else {
                continue;
            };
            match DecodingKey::from_rsa_components(&key.n, &key.e) {
                Ok(dk) => {
                    new_keys.insert(kid.clone(), dk);
                }
                Err(e) => {
                    warn!(kid = %kid, error = %e, "Failed to parse JWK");
                }
            }
        }

        info!(count = new_keys.len(), "JWKS keys loaded");
        *self.keys.write().await = new_keys;
        *self.last_refresh.write().await = Instant::now();
        Ok(())
    }

    /// Refresh keys if the TTL has elapsed.
    ///
    /// Holds the refresh mutex so concurrent callers coalesce into a single
    /// HTTP fetch. The second caller will re-check the TTL after acquiring
    /// the lock and find it fresh.
    async fn refresh_if_stale(&self) -> Result<(), String> {
        let last = *self.last_refresh.read().await;
        if last.elapsed() <= self.ttl {
            return Ok(());
        }
        let _guard = self.refresh_mutex.lock().await;
        // Re-check after acquiring the lock — another task may have refreshed.
        let last = *self.last_refresh.read().await;
        if last.elapsed() <= self.ttl {
            return Ok(());
        }
        self.refresh_keys().await
    }

    /// Refresh keys unconditionally, coalescing concurrent callers.
    async fn refresh_keys_coalesced(&self) -> Result<(), String> {
        let _guard = self.refresh_mutex.lock().await;
        self.refresh_keys().await
    }

    /// Validate an access token using the configured strategy.
    pub async fn validate_access_token(&self, token: &str) -> Result<Identity, Status> {
        match self.config.token_validation {
            OidcTokenValidation::Jwt => self.validate_jwt(token).await,
            OidcTokenValidation::Userinfo => self.validate_through_userinfo(token).await,
        }
    }

    /// Validate a JWT and return an `Identity`.
    ///
    /// This is the authentication step — it verifies the caller's identity
    /// but does not check authorization (that's `authz::AuthzPolicy::check`).
    async fn validate_jwt(&self, token: &str) -> Result<Identity, Status> {
        self.refresh_if_stale().await.map_err(|e| {
            warn!(error = %e, "JWKS refresh failed");
            Status::internal("OIDC key refresh failed")
        })?;

        // Decode the header to find the key ID.
        let header = decode_header(token).map_err(|e| {
            debug!(error = %e, "Failed to decode JWT header");
            Status::unauthenticated("invalid token")
        })?;

        let kid = header.kid.ok_or_else(|| {
            debug!("JWT has no kid in header");
            Status::unauthenticated("invalid token: missing kid")
        })?;

        // Look up the key in cache.
        let keys = self.keys.read().await;
        let decoding_key = if let Some(k) = keys.get(&kid) {
            k.clone()
        } else {
            // Key not found -- try refreshing once (key rotation).
            drop(keys);
            self.refresh_keys_coalesced().await.map_err(|e| {
                warn!(error = %e, "JWKS refresh on kid miss failed");
                Status::internal("OIDC key refresh failed")
            })?;
            let keys = self.keys.read().await;
            keys.get(&kid).cloned().ok_or_else(|| {
                debug!(kid = %kid, "JWT kid not found in JWKS");
                Status::unauthenticated("invalid token: unknown signing key")
            })?
        };

        // Validate the JWT.
        let mut validation = Validation::new(Algorithm::RS256);
        validation.set_issuer(&[&self.config.issuer]);
        validation.set_audience(&[&self.config.audience]);

        let token_data = decode::<OidcClaims>(token, &decoding_key, &validation).map_err(|e| {
            debug!(error = %e, "JWT validation failed");
            Status::unauthenticated(format!("invalid token: {e}"))
        })?;

        Ok(self.identity_from_claims(token_data.claims))
    }

    fn identity_from_claims(&self, mut claims: OidcClaims) -> Identity {
        if !self.config.roles_claim.is_empty() {
            claims.extract_roles(&self.config.roles_claim);
        }

        let scopes = if self.config.scopes_claim.is_empty() {
            vec![]
        } else {
            claims.extract_scopes(&self.config.scopes_claim)
        };

        Identity {
            subject: claims.sub,
            display_name: claims.preferred_username,
            roles: claims.roles,
            scopes,
            provider: IdentityProvider::Oidc,
        }
    }

    async fn validate_through_userinfo(&self, token: &str) -> Result<Identity, Status> {
        let cache_key = token_cache_key(token);
        let now = Instant::now();
        if let Some(cached) = self
            .userinfo_cache
            .read()
            .await
            .get(&cache_key)
            .filter(|cached| cached.expires_at > now)
        {
            return Ok(cached.identity.clone());
        }

        let endpoint = self
            .userinfo_endpoint
            .as_deref()
            .ok_or_else(|| Status::internal("OIDC UserInfo endpoint is not configured"))?;
        let response = self
            .userinfo_http
            .get(endpoint)
            .bearer_auth(token)
            .header(reqwest::header::ACCEPT, "application/json, application/jwt")
            .send()
            .await
            .map_err(|error| {
                warn!(%error, "OIDC UserInfo request failed");
                Status::unavailable("OIDC UserInfo validation unavailable")
            })?;

        let status = response.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            debug!(%status, "OIDC UserInfo rejected access token");
            return Err(Status::unauthenticated("invalid token"));
        }
        if !status.is_success() {
            warn!(%status, "OIDC UserInfo returned an unexpected status");
            return Err(Status::unavailable("OIDC UserInfo validation unavailable"));
        }

        let body = response.text().await.map_err(|error| {
            warn!(%error, "failed to read OIDC UserInfo response");
            Status::unavailable("OIDC UserInfo response invalid")
        })?;
        let identity = self.identity_from_userinfo_response(&body).await?;

        let cache_ttl = Duration::from_secs(self.config.userinfo_cache_ttl_secs);
        if !cache_ttl.is_zero() {
            let expires_at = Instant::now()
                .checked_add(cache_ttl)
                .unwrap_or_else(Instant::now);
            let mut cache = self.userinfo_cache.write().await;
            cache.retain(|_, cached| cached.expires_at > now);
            cache.insert(
                cache_key,
                CachedIdentity {
                    identity: identity.clone(),
                    expires_at,
                },
            );
        }

        Ok(identity)
    }

    async fn identity_from_userinfo_response(&self, body: &str) -> Result<Identity, Status> {
        let trimmed = body.trim();
        match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(serde_json::Value::String(jwt)) => self.validate_jwt(&jwt).await,
            Ok(value @ serde_json::Value::Object(_)) => {
                let claims = serde_json::from_value::<OidcClaims>(value).map_err(|error| {
                    debug!(%error, "OIDC UserInfo JSON did not contain valid identity claims");
                    Status::unauthenticated("invalid token")
                })?;
                Ok(self.identity_from_claims(claims))
            }
            Ok(_) => {
                debug!("OIDC UserInfo response was not a JWT or JSON object");
                Err(Status::unauthenticated("invalid token"))
            }
            Err(_) if trimmed.split('.').count() == 3 => self.validate_jwt(trimmed).await,
            Err(error) => {
                debug!(%error, "OIDC UserInfo response was malformed");
                Err(Status::unauthenticated("invalid token"))
            }
        }
    }
}

fn token_cache_key(token: &str) -> [u8; 32] {
    Sha256::digest(token.as_bytes()).into()
}

/// Authenticator that validates `Authorization: Bearer <access_token>` headers
/// against the configured OIDC issuer.
///
/// Returns `Ok(None)` when no Bearer header is present, so the chain can fall
/// through to other authenticators (e.g. the gateway-minted sandbox JWT
/// authenticator).
pub struct OidcAuthenticator {
    cache: Arc<JwksCache>,
}

impl OidcAuthenticator {
    pub fn new(cache: Arc<JwksCache>) -> Self {
        Self { cache }
    }
}

#[async_trait]
impl Authenticator for OidcAuthenticator {
    async fn authenticate(
        &self,
        headers: &http::HeaderMap,
        _path: &str,
    ) -> Result<Option<Principal>, Status> {
        let Some(token) = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
        else {
            return Ok(None);
        };

        let identity = self.cache.validate_access_token(token).await?;
        Ok(Some(Principal::User(UserPrincipal { identity })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn userinfo_config(issuer: String) -> OidcConfig {
        OidcConfig {
            issuer,
            audience: "openshell-client".to_string(),
            token_validation: OidcTokenValidation::Userinfo,
            jwks_ttl_secs: 3600,
            userinfo_cache_ttl_secs: 30,
            roles_claim: String::new(),
            admin_role: String::new(),
            user_role: String::new(),
            scopes_claim: String::new(),
        }
    }

    async fn mount_discovery(mock_server: &MockServer, include_userinfo: bool) {
        let mut discovery = serde_json::json!({
            "issuer": mock_server.uri(),
            "jwks_uri": format!("{}/jwks", mock_server.uri()),
        });
        if include_userinfo {
            discovery["userinfo_endpoint"] =
                serde_json::Value::String(format!("{}/userinfo", mock_server.uri()));
        }

        Mock::given(method("GET"))
            .and(path("/.well-known/openid-configuration"))
            .respond_with(ResponseTemplate::new(200).set_body_json(discovery))
            .mount(mock_server)
            .await;

        Mock::given(method("GET"))
            .and(path("/jwks"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "keys": [] })),
            )
            .mount(mock_server)
            .await;
    }

    #[test]
    fn health_is_unauthenticated() {
        assert!(is_unauthenticated_method("/openshell.v1.OpenShell/Health"));
    }

    #[test]
    fn sandbox_operations_require_auth() {
        assert!(!is_unauthenticated_method(
            "/openshell.v1.OpenShell/CreateSandbox"
        ));
    }

    #[test]
    fn reflection_is_unauthenticated() {
        assert!(is_unauthenticated_method(
            "/grpc.reflection.v1alpha.ServerReflection/ServerReflectionInfo"
        ));
        assert!(is_unauthenticated_method(
            "/grpc.reflection.v1.ServerReflection/ServerReflectionInfo"
        ));
    }

    #[test]
    fn grpc_health_is_unauthenticated() {
        assert!(is_unauthenticated_method("/grpc.health.v1.Health/Check"));
    }

    #[test]
    fn extract_roles_keycloak_path() {
        let json = serde_json::json!({
            "sub": "user1",
            "realm_access": { "roles": ["openshell-user", "openshell-admin"] }
        });
        let mut claims: OidcClaims = serde_json::from_value(json).unwrap();
        claims.extract_roles("realm_access.roles");
        assert_eq!(claims.roles, vec!["openshell-user", "openshell-admin"]);
    }

    #[test]
    fn extract_roles_flat_path() {
        // Entra ID / Okta style: roles at top level
        let json = serde_json::json!({
            "sub": "user1",
            "roles": ["OpenShell.Admin", "OpenShell.User"]
        });
        let mut claims: OidcClaims = serde_json::from_value(json).unwrap();
        claims.extract_roles("roles");
        assert_eq!(claims.roles, vec!["OpenShell.Admin", "OpenShell.User"]);
    }

    #[test]
    fn extract_roles_groups_path() {
        // Okta style: groups claim
        let json = serde_json::json!({
            "sub": "user1",
            "groups": ["everyone", "openshell-admin"]
        });
        let mut claims: OidcClaims = serde_json::from_value(json).unwrap();
        claims.extract_roles("groups");
        assert_eq!(claims.roles, vec!["everyone", "openshell-admin"]);
    }

    #[test]
    fn extract_roles_missing_claim() {
        let json = serde_json::json!({ "sub": "user1" });
        let mut claims: OidcClaims = serde_json::from_value(json).unwrap();
        claims.extract_roles("realm_access.roles");
        assert!(claims.roles.is_empty());
    }

    #[test]
    fn extract_scopes_space_delimited() {
        let json = serde_json::json!({
            "sub": "user1",
            "scope": "openid sandbox:read sandbox:write"
        });
        let claims: OidcClaims = serde_json::from_value(json).unwrap();
        let scopes = claims.extract_scopes("scope");
        assert_eq!(scopes, vec!["sandbox:read", "sandbox:write"]);
    }

    #[test]
    fn extract_scopes_json_array() {
        let json = serde_json::json!({
            "sub": "user1",
            "scp": ["sandbox:read", "provider:read"]
        });
        let claims: OidcClaims = serde_json::from_value(json).unwrap();
        let scopes = claims.extract_scopes("scp");
        assert_eq!(scopes, vec!["sandbox:read", "provider:read"]);
    }

    #[test]
    fn extract_scopes_filters_standard_oidc_scopes() {
        let json = serde_json::json!({
            "sub": "user1",
            "scope": "openid profile email sandbox:read offline_access"
        });
        let claims: OidcClaims = serde_json::from_value(json).unwrap();
        let scopes = claims.extract_scopes("scope");
        assert_eq!(scopes, vec!["sandbox:read"]);
    }

    #[test]
    fn extract_scopes_missing_claim() {
        let json = serde_json::json!({ "sub": "user1" });
        let claims: OidcClaims = serde_json::from_value(json).unwrap();
        let scopes = claims.extract_scopes("scope");
        assert!(scopes.is_empty());
    }

    #[test]
    fn extract_scopes_openid_only_yields_empty() {
        let json = serde_json::json!({
            "sub": "user1",
            "scope": "openid"
        });
        let claims: OidcClaims = serde_json::from_value(json).unwrap();
        let scopes = claims.extract_scopes("scope");
        assert!(scopes.is_empty());
    }

    #[tokio::test]
    async fn userinfo_validates_and_caches_opaque_access_token() {
        let mock_server = MockServer::start().await;
        mount_discovery(&mock_server, true).await;
        Mock::given(method("GET"))
            .and(path("/userinfo"))
            .and(header("authorization", "Bearer opaque-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "sub": "user-123",
                "preferred_username": "fran"
            })))
            .expect(1)
            .mount(&mock_server)
            .await;

        let cache = JwksCache::new(&userinfo_config(mock_server.uri()))
            .await
            .expect("UserInfo cache should initialize");

        let first = cache
            .validate_access_token("opaque-token")
            .await
            .expect("opaque token should validate");
        let second = cache
            .validate_access_token("opaque-token")
            .await
            .expect("cached opaque token should validate");

        assert_eq!(first.subject, "user-123");
        assert_eq!(first.display_name.as_deref(), Some("fran"));
        assert_eq!(second.subject, first.subject);
        assert_eq!(first.provider, IdentityProvider::Oidc);
    }

    #[tokio::test]
    async fn userinfo_rejection_is_unauthenticated() {
        let mock_server = MockServer::start().await;
        mount_discovery(&mock_server, true).await;
        Mock::given(method("GET"))
            .and(path("/userinfo"))
            .and(header("authorization", "Bearer rejected-token"))
            .respond_with(ResponseTemplate::new(401))
            .expect(1)
            .mount(&mock_server)
            .await;

        let cache = JwksCache::new(&userinfo_config(mock_server.uri()))
            .await
            .expect("UserInfo cache should initialize");
        let status = cache
            .validate_access_token("rejected-token")
            .await
            .expect_err("rejected token must fail");

        assert_eq!(status.code(), tonic::Code::Unauthenticated);
    }

    #[tokio::test]
    async fn userinfo_redirect_is_not_followed() {
        let mock_server = MockServer::start().await;
        mount_discovery(&mock_server, true).await;
        Mock::given(method("GET"))
            .and(path("/userinfo"))
            .and(header("authorization", "Bearer redirected-token"))
            .respond_with(
                ResponseTemplate::new(302)
                    .insert_header("Location", format!("{}/redirected", mock_server.uri())),
            )
            .expect(1)
            .mount(&mock_server)
            .await;
        Mock::given(method("GET"))
            .and(path("/redirected"))
            .and(header("authorization", "Bearer redirected-token"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "sub": "user-123"
            })))
            .expect(0)
            .mount(&mock_server)
            .await;

        let cache = JwksCache::new(&userinfo_config(mock_server.uri()))
            .await
            .expect("UserInfo cache should initialize");
        let status = cache
            .validate_access_token("redirected-token")
            .await
            .expect_err("redirected UserInfo response must fail");

        assert_eq!(status.code(), tonic::Code::Unavailable);
        mock_server.verify().await;
    }

    #[tokio::test]
    async fn userinfo_mode_requires_discovered_endpoint() {
        let mock_server = MockServer::start().await;
        mount_discovery(&mock_server, false).await;

        let error = JwksCache::new(&userinfo_config(mock_server.uri()))
            .await
            .expect_err("missing UserInfo endpoint must fail startup");

        assert!(error.contains("userinfo_endpoint"));
    }
}
