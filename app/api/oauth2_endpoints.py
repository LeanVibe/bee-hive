"""
OAuth2 API Endpoints for LeanVibe Agent Hive 2.0

Implements OAuth2 authorization server endpoints including:
- Authorization endpoint (/oauth2/authorize)
- Token endpoint (/oauth2/token)  
- Revocation endpoint (/oauth2/revoke)
- Introspection endpoint (/oauth2/introspect)
- Client management endpoints

CRITICAL COMPONENT: Provides OAuth2 flows for enterprise SSO integration.
"""

import asyncio
import base64
import json
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import urlencode, parse_qs

import structlog
from fastapi import APIRouter, HTTPException, Request, Response, Form, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials, HTTPBearer
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel, Field, validator

from ..core.oauth2_provider import (
    get_oauth2_provider, OAuth2Provider, OAuth2GrantType, OAuth2Scope,
    OAuth2TokenRequest, OAuth2AuthorizeRequest, OAuth2TokenResponse,
    OAuth2ErrorResponse, OAuth2Client, OAuth2ClientType
)
from ..core.auth import get_current_user, User, require_permission, Permission
from ..core.enterprise_security_system import get_security_system, SecurityEvent

logger = structlog.get_logger()

# Security schemes
security = HTTPBearer()
basic_auth = HTTPBasic()

# Router
oauth2_router = APIRouter(prefix="/oauth2", tags=["OAuth2"])


class ClientRegistrationRequest(BaseModel):
    """Client registration request."""
    client_name: str = Field(..., min_length=1, max_length=100)
    client_type: str = Field(..., regex="^(confidential|public)$")
    redirect_uris: List[str] = []
    allowed_scopes: List[str] = []
    allowed_grant_types: List[str] = []
    organization_id: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    @validator('redirect_uris')
    def validate_redirect_uris(cls, v):
        for uri in v:
            if not uri.startswith(('http://', 'https://')):
                raise ValueError('Invalid redirect URI format')
        return v
    
    @validator('allowed_scopes')
    def validate_scopes(cls, v):
        valid_scopes = [scope.value for scope in OAuth2Scope]
        for scope in v:
            if scope not in valid_scopes:
                raise ValueError(f'Invalid scope: {scope}')
        return v
    
    @validator('allowed_grant_types')
    def validate_grant_types(cls, v):
        valid_grants = [grant.value for grant in OAuth2GrantType]
        for grant in v:
            if grant not in valid_grants:
                raise ValueError(f'Invalid grant type: {grant}')
        return v


class ClientResponse(BaseModel):
    """Client response model."""
    client_id: str
    client_name: str
    client_type: str
    client_secret: Optional[str] = None
    redirect_uris: List[str]
    allowed_scopes: List[str]
    allowed_grant_types: List[str]
    is_active: bool
    created_at: datetime
    organization_id: Optional[str] = None


# Authorization Server Metadata
@oauth2_router.get("/.well-known/oauth-authorization-server")
async def authorization_server_metadata():
    """OAuth2 Authorization Server Metadata (RFC 8414)."""
    provider = await get_oauth2_provider()
    return provider.get_authorization_server_metadata()


# OpenID Connect Discovery
@oauth2_router.get("/.well-known/openid_configuration")
async def openid_configuration():
    """OpenID Connect Discovery endpoint."""
    provider = await get_oauth2_provider()
    metadata = provider.get_authorization_server_metadata()
    
    # Add OpenID Connect specific endpoints
    metadata.update({
        "userinfo_endpoint": f"{metadata['issuer']}/oauth2/userinfo",
        "subject_types_supported": ["public"],
        "id_token_signing_alg_values_supported": ["RS256"],
        "response_modes_supported": ["query", "fragment"],
        "claims_supported": ["sub", "iss", "aud", "exp", "iat", "name", "email", "email_verified"]
    })
    
    return metadata


# Authorization Endpoint
@oauth2_router.get("/authorize")
async def authorize_get(
    request: Request,
    response_type: str,
    client_id: str,
    redirect_uri: str,
    scope: Optional[str] = None,
    state: Optional[str] = None,
    code_challenge: Optional[str] = None,
    code_challenge_method: Optional[str] = "S256",
    nonce: Optional[str] = None
):
    """OAuth2 Authorization endpoint (GET) - redirect to login page."""
    try:
        provider = await get_oauth2_provider()
        
        # Validate client
        client = await provider.get_client(client_id)
        if not client or not client.is_active:
            raise HTTPException(status_code=400, detail="Invalid client_id")
        
        # Validate redirect URI
        if redirect_uri not in client.redirect_uris:
            raise HTTPException(status_code=400, detail="Invalid redirect_uri")
        
        # Validate response type
        if response_type not in ["code", "token"]:
            return _redirect_with_error(redirect_uri, "unsupported_response_type", state)
        
        # Store authorization request in session or temporary storage
        auth_request_id = secrets.token_urlsafe(32)
        
        # In production, store in Redis or database
        # For now, store in memory (this would be lost on restart)
        auth_request_data = {
            "response_type": response_type,
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
            "nonce": nonce,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Redirect to login page with auth request ID
        login_url = f"/auth/login?auth_request={auth_request_id}"
        return RedirectResponse(url=login_url, status_code=302)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Authorization endpoint error", error=str(e))
        return _redirect_with_error(redirect_uri, "server_error", state)


@oauth2_router.post("/authorize")
async def authorize_post(
    request: Request,
    auth_request_id: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    """OAuth2 Authorization endpoint (POST) - process user consent."""
    try:
        provider = await get_oauth2_provider()
        
        # Retrieve authorization request (in production, from Redis/database)
        # For now, this is a simplified implementation
        
        # Mock authorization request data
        auth_request = {
            "response_type": "code",
            "client_id": "leanvibe-pwa",
            "redirect_uri": "http://localhost:3000/auth/callback",
            "scope": "openid profile email agents:read agents:write tasks:read tasks:write",
            "state": "random_state_value"
        }
        
        client = await provider.get_client(auth_request["client_id"])
        if not client:
            raise HTTPException(status_code=400, detail="Invalid client")
        
        scopes = auth_request["scope"].split() if auth_request["scope"] else []
        validated_scopes = provider.validate_scopes(client, scopes)
        
        if auth_request["response_type"] == "code":
            # Authorization Code Flow
            code = await provider.create_authorization_code(
                client_id=auth_request["client_id"],
                user_id=current_user.id,
                redirect_uri=auth_request["redirect_uri"],
                scopes=validated_scopes,
                code_challenge=auth_request.get("code_challenge"),
                code_challenge_method=auth_request.get("code_challenge_method")
            )
            
            # Redirect back to client with authorization code
            params = {"code": code}
            if auth_request.get("state"):
                params["state"] = auth_request["state"]
            
            redirect_url = f"{auth_request['redirect_uri']}?{urlencode(params)}"
            return RedirectResponse(url=redirect_url, status_code=302)
        
        else:
            # Implicit Flow (not recommended, but supported)
            tokens = await provider._create_tokens(auth_request["client_id"], current_user.id, validated_scopes)
            
            params = {
                "access_token": tokens.access_token,
                "token_type": tokens.token_type,
                "expires_in": str(tokens.expires_in),
                "scope": tokens.scope
            }
            if auth_request.get("state"):
                params["state"] = auth_request["state"]
            
            # Return as fragment
            redirect_url = f"{auth_request['redirect_uri']}#{urlencode(params)}"
            return RedirectResponse(url=redirect_url, status_code=302)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Authorization POST error", error=str(e))
        raise HTTPException(status_code=500, detail="Authorization failed")


# Token Endpoint
@oauth2_router.post("/token", response_model=OAuth2TokenResponse)
async def token_endpoint(
    request: Request,
    grant_type: str = Form(...),
    code: Optional[str] = Form(None),
    redirect_uri: Optional[str] = Form(None),
    client_id: Optional[str] = Form(None),
    client_secret: Optional[str] = Form(None),
    refresh_token: Optional[str] = Form(None),
    scope: Optional[str] = Form(None),
    code_verifier: Optional[str] = Form(None),
    credentials: HTTPBasicCredentials = Depends(basic_auth)
):
    """OAuth2 Token endpoint."""
    try:
        provider = await get_oauth2_provider()
        security_system = await get_security_system()
        
        # Extract client credentials (Basic Auth or form)
        if credentials.username:
            client_id = credentials.username
            client_secret = credentials.password
        
        if not client_id:
            raise HTTPException(status_code=400, detail="Client authentication required")
        
        # Authenticate client
        if not await provider.authenticate_client(client_id, client_secret):
            await security_system.log_security_event(
                SecurityEvent.LOGIN_FAILED,
                request=request,
                client_id=client_id,
                action="oauth2_client_auth_failed"
            )
            raise HTTPException(status_code=401, detail="Invalid client credentials")
        
        # Handle different grant types
        if grant_type == OAuth2GrantType.AUTHORIZATION_CODE.value:
            if not code or not redirect_uri:
                raise HTTPException(status_code=400, detail="Missing required parameters")
            
            return await provider.exchange_authorization_code(
                code=code,
                client_id=client_id,
                redirect_uri=redirect_uri,
                code_verifier=code_verifier
            )
        
        elif grant_type == OAuth2GrantType.CLIENT_CREDENTIALS.value:
            scopes = scope.split() if scope else []
            return await provider.client_credentials_grant(client_id, scopes)
        
        elif grant_type == OAuth2GrantType.REFRESH_TOKEN.value:
            if not refresh_token:
                raise HTTPException(status_code=400, detail="Missing refresh token")
            
            return await provider.refresh_token_grant(refresh_token, client_id)
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported grant type")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token endpoint error", error=str(e))
        raise HTTPException(status_code=500, detail="Token request failed")


# Token Revocation Endpoint
@oauth2_router.post("/revoke")
async def revoke_token(
    request: Request,
    token: str = Form(...),
    token_type_hint: Optional[str] = Form(None),
    client_id: Optional[str] = Form(None),
    client_secret: Optional[str] = Form(None),
    credentials: HTTPBasicCredentials = Depends(basic_auth)
):
    """OAuth2 Token Revocation endpoint (RFC 7009)."""
    try:
        provider = await get_oauth2_provider()
        
        # Extract client credentials
        if credentials.username:
            client_id = credentials.username
            client_secret = credentials.password
        
        if client_id:
            # Authenticate client if provided
            if not await provider.authenticate_client(client_id, client_secret):
                raise HTTPException(status_code=401, detail="Invalid client credentials")
        
        # Revoke token
        success = await provider.revoke_token(token, token_type_hint or "access_token")
        
        # Always return 200 (per RFC 7009)
        return Response(status_code=200)
        
    except HTTPException as e:
        if e.status_code == 401:
            raise
        # Return 200 even on errors (per RFC 7009)
        return Response(status_code=200)
    except Exception as e:
        logger.error("Token revocation error", error=str(e))
        # Return 200 even on errors (per RFC 7009)
        return Response(status_code=200)


# Token Introspection Endpoint
@oauth2_router.post("/introspect")
async def introspect_token(
    request: Request,
    token: str = Form(...),
    token_type_hint: Optional[str] = Form(None),
    client_id: Optional[str] = Form(None),
    client_secret: Optional[str] = Form(None),
    credentials: HTTPBasicCredentials = Depends(basic_auth)
):
    """OAuth2 Token Introspection endpoint (RFC 7662)."""
    try:
        provider = await get_oauth2_provider()
        
        # Extract client credentials
        if credentials.username:
            client_id = credentials.username
            client_secret = credentials.password
        
        if not client_id:
            raise HTTPException(status_code=401, detail="Client authentication required")
        
        # Authenticate client
        if not await provider.authenticate_client(client_id, client_secret):
            raise HTTPException(status_code=401, detail="Invalid client credentials")
        
        # Validate token
        token_info = await provider.validate_access_token(token)
        
        if not token_info:
            return {"active": False}
        
        return {
            "active": True,
            "client_id": token_info["client_id"],
            "username": token_info.get("user_id"),
            "scope": " ".join(token_info["scopes"]),
            "exp": int(token_info["expires_at"].timestamp()),
            "iat": int(token_info["created_at"].timestamp()),
            "sub": token_info.get("user_id"),
            "aud": token_info["client_id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token introspection error", error=str(e))
        raise HTTPException(status_code=500, detail="Introspection failed")


# UserInfo Endpoint (OpenID Connect)
@oauth2_router.get("/userinfo")
@oauth2_router.post("/userinfo")
async def userinfo_endpoint(request: Request, authorization: str = Depends(HTTPBearer())):
    """OpenID Connect UserInfo endpoint."""
    try:
        provider = await get_oauth2_provider()
        
        # Validate access token
        token_info = await provider.validate_access_token(authorization.credentials)
        if not token_info:
            raise HTTPException(
                status_code=401,
                detail="Invalid access token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Check if token has openid scope
        if "openid" not in token_info.get("scopes", []):
            raise HTTPException(status_code=403, detail="Insufficient scope for UserInfo")
        
        # Get user information
        from ..core.auth import get_auth_service
        auth_service = get_auth_service()
        user = auth_service.get_user_by_id(token_info["user_id"])
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Build UserInfo response based on scopes
        userinfo = {"sub": token_info["user_id"]}
        
        if "profile" in token_info.get("scopes", []):
            userinfo.update({
                "name": user.full_name,
                "preferred_username": user.email.split("@")[0],
                "updated_at": int(user.created_at.timestamp())
            })
        
        if "email" in token_info.get("scopes", []):
            userinfo.update({
                "email": user.email,
                "email_verified": True
            })
        
        return userinfo
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("UserInfo endpoint error", error=str(e))
        raise HTTPException(status_code=500, detail="UserInfo request failed")


# Client Management Endpoints

@oauth2_router.post("/clients", response_model=ClientResponse)
async def register_client(
    client_request: ClientRegistrationRequest,
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Register new OAuth2 client (admin only)."""
    try:
        provider = await get_oauth2_provider()
        
        client = await provider.register_client(
            client_request.dict(),
            created_by=current_user.id
        )
        
        return ClientResponse(
            client_id=client.client_id,
            client_name=client.client_name,
            client_type=client.client_type.value,
            client_secret=client.client_secret,
            redirect_uris=client.redirect_uris,
            allowed_scopes=[scope.value for scope in client.allowed_scopes],
            allowed_grant_types=[grant.value for grant in client.allowed_grant_types],
            is_active=client.is_active,
            created_at=datetime.utcnow(),
            organization_id=client.organization_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Client registration error", error=str(e))
        raise HTTPException(status_code=500, detail="Client registration failed")


@oauth2_router.get("/clients/{client_id}", response_model=ClientResponse)
async def get_client(
    client_id: str,
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Get OAuth2 client details (admin only)."""
    try:
        provider = await get_oauth2_provider()
        client = await provider.get_client(client_id)
        
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        return ClientResponse(
            client_id=client.client_id,
            client_name=client.client_name,
            client_type=client.client_type.value,
            client_secret=None,  # Never return client secret in GET
            redirect_uris=client.redirect_uris,
            allowed_scopes=[scope.value for scope in client.allowed_scopes],
            allowed_grant_types=[grant.value for grant in client.allowed_grant_types],
            is_active=client.is_active,
            created_at=datetime.utcnow(),
            organization_id=client.organization_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Get client error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get client")


@oauth2_router.delete("/clients/{client_id}")
async def delete_client(
    client_id: str,
    current_user: User = Depends(require_permission(Permission.MANAGE_USERS))
):
    """Delete OAuth2 client (admin only)."""
    try:
        provider = await get_oauth2_provider()
        client = await provider.get_client(client_id)
        
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")
        
        # Deactivate client (don't actually delete to maintain audit trail)
        client.is_active = False
        
        security_system = await get_security_system()
        await security_system.log_security_event(
            SecurityEvent.CONFIGURATION_CHANGED,
            user_id=current_user.id,
            client_id=client_id,
            action="oauth2_client_deleted"
        )
        
        return {"message": "Client deactivated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Delete client error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete client")


# JWKS Endpoint (for JWT verification)
@oauth2_router.get("/jwks")
async def jwks_endpoint():
    """JSON Web Key Set (JWKS) endpoint for JWT verification."""
    # In production, would return actual JWK set for JWT signature verification
    return {
        "keys": [
            {
                "kty": "RSA",
                "use": "sig",
                "kid": "1",
                "n": "example_modulus",
                "e": "AQAB"
            }
        ]
    }


# Helper Functions

def _redirect_with_error(redirect_uri: str, error: str, state: Optional[str] = None, description: Optional[str] = None):
    """Redirect with OAuth2 error."""
    params = {"error": error}
    if description:
        params["error_description"] = description
    if state:
        params["state"] = state
    
    redirect_url = f"{redirect_uri}?{urlencode(params)}"
    return RedirectResponse(url=redirect_url, status_code=302)


# Include router in main application
def get_oauth2_router():
    """Get OAuth2 router for inclusion in main app."""
    return oauth2_router