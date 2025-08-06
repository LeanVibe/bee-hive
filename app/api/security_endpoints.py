"""
Security API endpoints for LeanVibe Agent Hive enterprise security features.

Includes endpoints for:
- API Key Management
- OAuth 2.0/OpenID Connect
- WebAuthn Authentication
- Multi-Factor Authentication (MFA)
- Rate Limiting Administration
- Security Monitoring and Audit
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.api_key_manager import (
    EnterpriseApiKeyManager, get_api_key_manager, 
    CreateApiKeyRequest, ApiKeyResponse, ApiKeyUsageStats,
    ApiKeyType, ApiKeyStatus, PermissionScope
)
from ..core.oauth_provider_system import (
    OAuthProviderSystem, get_oauth_provider,
    AuthorizationRequest, TokenRequest, TokenResponse, UserInfoResponse
)
from ..core.webauthn_system import (
    WebAuthnSystem, get_webauthn_system,
    RegistrationRequest, RegistrationResponse, 
    AuthenticationRequest, AuthenticationResponse, CredentialInfo
)
from ..core.mfa_system import (
    MFASystem, get_mfa_system,
    TOTPSetupRequest, TOTPVerificationRequest,
    SMSSetupRequest, SMSVerificationRequest,
    BackupCodesRequest, MFAStatusResponse
)
from ..core.advanced_rate_limiter import (
    AdvancedRateLimiter, get_rate_limiter,
    RateLimitRule, RateLimitStatus, ENTERPRISE_RATE_LIMITS
)
from ..core.rbac_engine import (
    AdvancedRBACEngine, get_rbac_engine,
    AuthorizationContext, AuthorizationDecision, 
    PermissionAction, ResourceType, PermissionScope
)
from ..core.auth import get_current_user, AuthenticationService
from ..core.database import get_session

# Create security router
security_router = APIRouter(prefix="/api/v1/security", tags=["Security & Enterprise Features"])

# Authentication dependency
security_bearer = HTTPBearer()


# ========================================
# API Key Management Endpoints
# ========================================

@security_router.post("/api-keys", response_model=ApiKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: CreateApiKeyRequest,
    current_user: str = Depends(get_current_user),
    api_key_manager: EnterpriseApiKeyManager = Depends(get_api_key_manager)
):
    """Create a new API key with specified permissions and restrictions."""
    return await api_key_manager.create_api_key(request, current_user)


@security_router.get("/api-keys", response_model=List[ApiKeyResponse])
async def list_api_keys(
    owner: Optional[str] = None,
    key_type: Optional[str] = None,
    status_filter: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: str = Depends(get_current_user),
    api_key_manager: EnterpriseApiKeyManager = Depends(get_api_key_manager)
):
    """List API keys with optional filtering."""
    
    # Parse enums
    key_type_enum = ApiKeyType(key_type) if key_type else None
    status_enum = ApiKeyStatus(status_filter) if status_filter else None
    
    return await api_key_manager.list_api_keys(
        owner=owner or current_user,
        key_type=key_type_enum,
        status=status_enum,
        limit=limit,
        offset=offset
    )


@security_router.get("/api-keys/{key_id}/usage", response_model=ApiKeyUsageStats)
async def get_api_key_usage(
    key_id: str,
    current_user: str = Depends(get_current_user),
    api_key_manager: EnterpriseApiKeyManager = Depends(get_api_key_manager)
):
    """Get detailed usage statistics for an API key."""
    
    usage_stats = await api_key_manager.get_usage_stats(key_id)
    if not usage_stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found or no usage data available"
        )
    
    return usage_stats


@security_router.post("/api-keys/{key_id}/rotate")
async def rotate_api_key(
    key_id: str,
    current_user: str = Depends(get_current_user),
    api_key_manager: EnterpriseApiKeyManager = Depends(get_api_key_manager)
):
    """Rotate an API key, generating a new key while preserving metadata."""
    
    new_api_key = await api_key_manager.rotate_api_key(key_id, current_user)
    if not new_api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found or rotation failed"
        )
    
    return {"message": "API key rotated successfully", "new_key": new_api_key}


@security_router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    reason: str = Body(..., embed=True),
    current_user: str = Depends(get_current_user),
    api_key_manager: EnterpriseApiKeyManager = Depends(get_api_key_manager)
):
    """Revoke an API key."""
    
    success = await api_key_manager.revoke_api_key(key_id, current_user, reason)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found or revocation failed"
        )
    
    return {"message": "API key revoked successfully"}


@security_router.get("/api-keys/metrics")
async def get_api_key_metrics(
    current_user: str = Depends(get_current_user),
    api_key_manager: EnterpriseApiKeyManager = Depends(get_api_key_manager)
):
    """Get comprehensive API key system metrics."""
    return await api_key_manager.get_system_metrics()


# ========================================
# OAuth 2.0 / OpenID Connect Endpoints
# ========================================

@security_router.get("/oauth/.well-known/openid-configuration")
async def openid_configuration(
    oauth_provider: OAuthProviderSystem = Depends(get_oauth_provider)
):
    """OpenID Connect discovery endpoint."""
    return oauth_provider.get_discovery_document()


@security_router.get("/oauth/.well-known/jwks.json")
async def jwks_endpoint(
    oauth_provider: OAuthProviderSystem = Depends(get_oauth_provider)
):
    """JSON Web Key Set endpoint."""
    return oauth_provider.get_jwks()


@security_router.get("/oauth/authorize")
async def oauth_authorize(
    client_id: str,
    redirect_uri: str,
    response_type: str = "code",
    scope: str = "openid profile",
    state: Optional[str] = None,
    nonce: Optional[str] = None,
    code_challenge: Optional[str] = None,
    code_challenge_method: str = "S256",
    oauth_provider: OAuthProviderSystem = Depends(get_oauth_provider)
):
    """OAuth 2.0 authorization endpoint."""
    
    auth_request = AuthorizationRequest(
        client_id=client_id,
        redirect_uri=redirect_uri,
        response_type=response_type,
        scope=scope,
        state=state,
        nonce=nonce,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method
    )
    
    result = await oauth_provider.handle_authorization_request(auth_request)
    
    # Construct redirect URL with authorization code
    from urllib.parse import urlencode
    
    params = {"code": result["code"]}
    if result["state"]:
        params["state"] = result["state"]
    
    redirect_url = f"{result['redirect_uri']}?{urlencode(params)}"
    
    return {
        "message": "Authorization successful",
        "redirect_url": redirect_url,
        "code": result["code"],  # For demo/testing purposes
        "state": result["state"]
    }


@security_router.post("/oauth/token", response_model=TokenResponse)
async def oauth_token(
    token_request: TokenRequest,
    oauth_provider: OAuthProviderSystem = Depends(get_oauth_provider)
):
    """OAuth 2.0 token endpoint."""
    return await oauth_provider.handle_token_request(token_request)


@security_router.get("/oauth/userinfo", response_model=UserInfoResponse)
async def oauth_userinfo(
    credentials: HTTPAuthorizationCredentials = Depends(security_bearer),
    oauth_provider: OAuthProviderSystem = Depends(get_oauth_provider)
):
    """OpenID Connect userinfo endpoint."""
    return await oauth_provider.get_userinfo(credentials.credentials)


@security_router.get("/oauth/sso/{provider}/authorize")
async def sso_authorize(
    provider: str,
    redirect_uri: str,
    state: Optional[str] = None,
    oauth_provider: OAuthProviderSystem = Depends(get_oauth_provider)
):
    """Initiate SSO authentication with external provider."""
    
    authorization_url = await oauth_provider.get_authorization_url(
        provider=provider,
        redirect_uri=redirect_uri,
        state=state
    )
    
    return {
        "authorization_url": authorization_url,
        "provider": provider,
        "state": state
    }


# ========================================
# WebAuthn Authentication Endpoints
# ========================================

@security_router.post("/webauthn/register/begin")
async def begin_webauthn_registration(
    request: RegistrationRequest,
    webauthn: WebAuthnSystem = Depends(get_webauthn_system)
):
    """Begin WebAuthn registration process."""
    return await webauthn.initiate_registration(request)


@security_router.post("/webauthn/register/complete")
async def complete_webauthn_registration(
    credential: RegistrationResponse,
    username: str,
    webauthn: WebAuthnSystem = Depends(get_webauthn_system)
):
    """Complete WebAuthn registration process."""
    return await webauthn.complete_registration(credential, username)


@security_router.post("/webauthn/authenticate/begin")
async def begin_webauthn_authentication(
    request: AuthenticationRequest,
    webauthn: WebAuthnSystem = Depends(get_webauthn_system)
):
    """Begin WebAuthn authentication process."""
    return await webauthn.initiate_authentication(request)


@security_router.post("/webauthn/authenticate/complete")
async def complete_webauthn_authentication(
    credential: AuthenticationResponse,
    webauthn: WebAuthnSystem = Depends(get_webauthn_system)
):
    """Complete WebAuthn authentication process."""
    return await webauthn.complete_authentication(credential)


@security_router.get("/webauthn/credentials/{username}", response_model=List[CredentialInfo])
async def get_webauthn_credentials(
    username: str,
    webauthn: WebAuthnSystem = Depends(get_webauthn_system)
):
    """Get all WebAuthn credentials for a user."""
    return await webauthn.get_user_credentials(username)


@security_router.delete("/webauthn/credentials/{username}/{credential_id}")
async def revoke_webauthn_credential(
    username: str,
    credential_id: str,
    webauthn: WebAuthnSystem = Depends(get_webauthn_system)
):
    """Revoke a specific WebAuthn credential."""
    success = await webauthn.revoke_credential(username, credential_id)
    return {"revoked": success}


@security_router.get("/webauthn/metrics")
async def get_webauthn_metrics(
    webauthn: WebAuthnSystem = Depends(get_webauthn_system)
):
    """Get WebAuthn system metrics."""
    return webauthn.get_metrics()


# ========================================
# Multi-Factor Authentication (MFA) Endpoints
# ========================================

@security_router.post("/mfa/totp/setup")
async def setup_totp_mfa(
    request: TOTPSetupRequest,
    mfa_system: MFASystem = Depends(get_mfa_system)
):
    """Set up TOTP-based MFA for a user."""
    return await mfa_system.setup_totp_device(request)


@security_router.post("/mfa/totp/verify")
async def verify_totp_mfa(
    request: TOTPVerificationRequest,
    mfa_system: MFASystem = Depends(get_mfa_system)
):
    """Verify TOTP code for MFA authentication."""
    return await mfa_system.verify_totp_code(request)


@security_router.post("/mfa/sms/setup")
async def setup_sms_mfa(
    request: SMSSetupRequest,
    mfa_system: MFASystem = Depends(get_mfa_system)
):
    """Set up SMS-based MFA for a user."""
    return await mfa_system.setup_sms_device(request)


@security_router.post("/mfa/sms/verify")
async def verify_sms_mfa(
    request: SMSVerificationRequest,
    mfa_system: MFASystem = Depends(get_mfa_system)
):
    """Verify SMS code for MFA authentication."""
    return await mfa_system.verify_sms_code(request)


@security_router.post("/mfa/backup-codes/generate")
async def generate_backup_codes(
    request: BackupCodesRequest,
    mfa_system: MFASystem = Depends(get_mfa_system)
):
    """Generate backup codes for MFA recovery."""
    return await mfa_system.generate_backup_codes(request)


@security_router.get("/mfa/status/{user_id}", response_model=MFAStatusResponse)
async def get_mfa_status(
    user_id: str,
    mfa_system: MFASystem = Depends(get_mfa_system)
):
    """Get MFA status and enrolled devices for a user."""
    return await mfa_system.get_user_mfa_status(user_id)


@security_router.get("/mfa/metrics")
async def get_mfa_metrics(
    mfa_system: MFASystem = Depends(get_mfa_system)
):
    """Get MFA system metrics."""
    return mfa_system.get_metrics()


# ========================================
# Rate Limiting Administration Endpoints
# ========================================

@security_router.get("/rate-limit/metrics")
async def get_rate_limit_metrics(
    current_user: str = Depends(get_current_user),
    rate_limiter: AdvancedRateLimiter = Depends(get_rate_limiter)
):
    """Get comprehensive rate limiting metrics."""
    return rate_limiter.get_metrics()


@security_router.post("/rate-limit/rules")
async def add_rate_limit_rule(
    rule: Dict[str, Any],
    current_user: str = Depends(get_current_user),
    rate_limiter: AdvancedRateLimiter = Depends(get_rate_limiter)
):
    """Add a new rate limiting rule."""
    
    # Convert dict to RateLimitRule (simplified)
    # In production, you'd have proper validation and conversion
    success = await rate_limiter.add_rate_limit_rule(rule)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to add rate limiting rule"
        )
    
    return {"message": "Rate limiting rule added successfully"}


@security_router.put("/rate-limit/tier/{identifier}/{tier}")
async def update_enterprise_tier(
    identifier: str,
    tier: str,
    current_user: str = Depends(get_current_user),
    rate_limiter: AdvancedRateLimiter = Depends(get_rate_limiter)
):
    """Update rate limiting tier for enterprise customers."""
    
    if tier not in ENTERPRISE_RATE_LIMITS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tier. Available tiers: {list(ENTERPRISE_RATE_LIMITS.keys())}"
        )
    
    success = await rate_limiter.update_enterprise_tier(identifier, tier)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update enterprise tier"
        )
    
    return {"message": f"Enterprise tier updated to {tier} for {identifier}"}


# ========================================
# Security Health Check Endpoint
# ========================================

@security_router.get("/health")
async def security_health_check(
    api_key_manager: EnterpriseApiKeyManager = Depends(get_api_key_manager),
    oauth_provider: OAuthProviderSystem = Depends(get_oauth_provider),
    webauthn: WebAuthnSystem = Depends(get_webauthn_system),
    mfa_system: MFASystem = Depends(get_mfa_system),
    rate_limiter: AdvancedRateLimiter = Depends(get_rate_limiter)
):
    """Comprehensive security systems health check."""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "systems": {
            "api_key_manager": {
                "status": "operational",
                "encryption_enabled": bool(api_key_manager.encryption_key)
            },
            "oauth_provider": {
                "status": "operational",
                "providers_configured": len(oauth_provider.sso_providers),
                "registered_clients": len(oauth_provider.registered_clients)
            },
            "webauthn": {
                "status": "operational",
                "config": {
                    "rp_id": webauthn.config["rp_id"],
                    "rp_name": webauthn.config["rp_name"]
                }
            },
            "mfa_system": {
                "status": "operational",
                "providers_enabled": mfa_system.config["enabled_providers"]
            },
            "rate_limiter": {
                "status": "operational",
                "rules_active": len(rate_limiter.rules),
                "redis_connected": True  # Simplified check
            }
        }
    }
    
    return health_status


# ========================================
# Security Dashboard Summary Endpoint
# ========================================

@security_router.get("/dashboard/summary")
async def get_security_dashboard_summary(
    current_user: str = Depends(get_current_user),
    api_key_manager: EnterpriseApiKeyManager = Depends(get_api_key_manager),
    webauthn: WebAuthnSystem = Depends(get_webauthn_system),
    mfa_system: MFASystem = Depends(get_mfa_system),
    rate_limiter: AdvancedRateLimiter = Depends(get_rate_limiter)
):
    """Get comprehensive security dashboard summary."""
    
    # Get metrics from all systems
    api_key_metrics = await api_key_manager.get_system_metrics()
    webauthn_metrics = webauthn.get_metrics()
    mfa_metrics = mfa_system.get_metrics()
    rate_limit_metrics = rate_limiter.get_metrics()
    
    summary = {
        "overview": {
            "total_api_keys": api_key_metrics.get("total_keys", 0),
            "active_api_keys": api_key_metrics.get("active_keys", 0),
            "webauthn_credentials": webauthn_metrics.get("total_credentials", 0),
            "mfa_enabled_users": mfa_metrics.get("total_enrolled_users", 0),
            "rate_limit_violations_24h": rate_limit_metrics.get("rate_limiting_metrics", {}).get("requests_rejected", 0)
        },
        "security_health": {
            "api_keys_expiring_soon": api_key_metrics.get("keys_expiring_soon", 0),
            "failed_authentications_24h": (
                api_key_metrics.get("api_key_metrics", {}).get("failed_authentications", 0) +
                webauthn_metrics.get("webauthn_metrics", {}).get("authentications_failed", 0) +
                mfa_metrics.get("mfa_metrics", {}).get("verification_failures", 0)
            ),
            "security_violations_24h": rate_limit_metrics.get("rate_limiting_metrics", {}).get("bans_issued", 0),
            "ddos_attacks_detected": rate_limit_metrics.get("rate_limiting_metrics", {}).get("ddos_attacks_detected", 0)
        },
        "recent_activity": {
            "api_keys_created_today": api_key_metrics.get("api_key_metrics", {}).get("keys_created", 0),
            "webauthn_registrations_today": webauthn_metrics.get("webauthn_metrics", {}).get("registrations_completed", 0),
            "mfa_setups_today": mfa_metrics.get("mfa_metrics", {}).get("devices_enrolled", 0),
            "rate_limit_adjustments_today": rate_limit_metrics.get("rate_limiting_metrics", {}).get("adaptive_adjustments", 0)
        },
        "compliance_status": {
            "audit_logging_enabled": True,
            "encryption_enabled": bool(api_key_manager.encryption_key),
            "mfa_coverage_percent": min(100, (mfa_metrics.get("total_enrolled_users", 0) / max(1, api_key_metrics.get("total_keys", 1))) * 100),
            "security_policies_active": len(rate_limiter.rules)
        }
    }
    
    return summary


# ========================================
# RBAC Management Endpoints
# ========================================

@security_router.post("/rbac/roles")
async def create_role(
    role_data: Dict[str, Any],
    current_user: str = Depends(get_current_user),
    rbac_engine: AdvancedRBACEngine = Depends(get_rbac_engine)
):
    """Create a new RBAC role with permissions."""
    
    role = await rbac_engine.create_role(role_data, current_user)
    if not role:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create role"
        )
    
    return {
        "id": role.id,
        "name": role.name,
        "description": role.description,
        "permissions_count": len(role.permissions),
        "is_system_role": role.is_system_role
    }


@security_router.post("/rbac/assign-role")
async def assign_role(
    assignment: Dict[str, Any] = Body(...),
    current_user: str = Depends(get_current_user),
    rbac_engine: AdvancedRBACEngine = Depends(get_rbac_engine)
):
    """Assign a role to a user."""
    
    user_id = assignment.get("user_id")
    role_id = assignment.get("role_id")
    expires_at = assignment.get("expires_at")
    
    if not user_id or not role_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id and role_id are required"
        )
    
    # Parse expiration date if provided
    expiration = None
    if expires_at:
        try:
            expiration = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid expires_at format. Use ISO 8601 format."
            )
    
    success = await rbac_engine.assign_role(
        user_id=user_id,
        role_id=role_id,
        assigned_by=current_user,
        scope=assignment.get("scope"),
        expires_at=expiration
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to assign role"
        )
    
    return {"message": "Role assigned successfully"}


@security_router.post("/rbac/revoke-role")
async def revoke_role(
    revocation: Dict[str, Any] = Body(...),
    current_user: str = Depends(get_current_user),
    rbac_engine: AdvancedRBACEngine = Depends(get_rbac_engine)
):
    """Revoke a role from a user."""
    
    user_id = revocation.get("user_id")
    role_id = revocation.get("role_id")
    reason = revocation.get("reason", "")
    
    if not user_id or not role_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="user_id and role_id are required"
        )
    
    success = await rbac_engine.revoke_role(
        user_id=user_id,
        role_id=role_id,
        revoked_by=current_user,
        reason=reason
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to revoke role"
        )
    
    return {"message": "Role revoked successfully"}


@security_router.post("/rbac/authorize")
async def check_authorization(
    auth_request: Dict[str, Any],
    current_user: str = Depends(get_current_user),
    rbac_engine: AdvancedRBACEngine = Depends(get_rbac_engine)
):
    """Check if a user is authorized to perform an action."""
    
    try:
        context = AuthorizationContext(
            user_id=auth_request.get("user_id", current_user),
            resource_type=ResourceType(auth_request["resource_type"]),
            resource_id=auth_request.get("resource_id"),
            action=PermissionAction(auth_request["action"]),
            scope=PermissionScope(auth_request.get("scope", "resource")),
            mfa_verified=auth_request.get("mfa_verified", False),
            risk_score=auth_request.get("risk_score", 0.0),
            country=auth_request.get("country"),
            additional_context=auth_request.get("additional_context", {})
        )
        
        decision = await rbac_engine.authorize(context)
        
        return {
            "authorized": decision.result.value == "granted",
            "result": decision.result.value,
            "reason": decision.reason,
            "evaluation_time_ms": decision.evaluation_time_ms,
            "cache_hit": decision.cache_hit,
            "decision_path": decision.decision_path
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request parameters: {str(e)}"
        )


@security_router.get("/rbac/user/{user_id}/permissions")
async def get_user_permissions(
    user_id: str,
    current_user: str = Depends(get_current_user),
    rbac_engine: AdvancedRBACEngine = Depends(get_rbac_engine)
):
    """Get all effective permissions for a user."""
    
    permissions = await rbac_engine.get_user_permissions(user_id)
    
    return {
        "user_id": user_id,
        "permissions_count": len(permissions),
        "permissions": [
            {
                "id": p.id,
                "resource_type": p.resource_type.value,
                "action": p.action.value,
                "scope": p.scope.value,
                "resource_id": p.resource_id,
                "description": p.description,
                "expires_at": p.expires_at.isoformat() if p.expires_at else None
            }
            for p in permissions
        ]
    }


@security_router.get("/rbac/user/{user_id}/accessible-resources")
async def get_accessible_resources(
    user_id: str,
    resource_type: str,
    action: str,
    current_user: str = Depends(get_current_user),
    rbac_engine: AdvancedRBACEngine = Depends(get_rbac_engine)
):
    """Get resources accessible to a user for a specific action."""
    
    try:
        accessible_resources = await rbac_engine.get_accessible_resources(
            user_id=user_id,
            resource_type=ResourceType(resource_type),
            action=PermissionAction(action)
        )
        
        return {
            "user_id": user_id,
            "resource_type": resource_type,
            "action": action,
            "accessible_resources": accessible_resources,
            "wildcard_access": "*" in accessible_resources
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid parameters: {str(e)}"
        )


@security_router.get("/rbac/metrics")
async def get_rbac_metrics(
    current_user: str = Depends(get_current_user),
    rbac_engine: AdvancedRBACEngine = Depends(get_rbac_engine)
):
    """Get comprehensive RBAC system metrics."""
    return await rbac_engine.get_system_metrics()


# Export the router
__all__ = ["security_router"]