"""
OAuth2 Provider System for LeanVibe Agent Hive 2.0

Enterprise-grade OAuth2 implementation with support for multiple flows,
client management, and integration with existing authentication system.

CRITICAL COMPONENT: Enables enterprise SSO integration and secure API access.
"""

import asyncio
import base64
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from urllib.parse import urlparse, parse_qs, urlencode

import structlog
from fastapi import HTTPException, Request, Response, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, update, delete
import jwt

from .config import settings
from .database import get_session
from .auth import get_auth_service, AuthenticationService, User, UserRole, Permission
from .enterprise_security_system import get_security_system, SecurityEvent, SecurityLevel

logger = structlog.get_logger()


class OAuth2GrantType(Enum):
    """OAuth2 grant types supported by the system."""
    AUTHORIZATION_CODE = "authorization_code"
    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"
    DEVICE_CODE = "device_code"
    IMPLICIT = "implicit"  # Deprecated but supported for legacy


class OAuth2ResponseType(Enum):
    """OAuth2 response types for authorization endpoint."""
    CODE = "code"
    TOKEN = "token"
    ID_TOKEN = "id_token"


class OAuth2Scope(Enum):
    """OAuth2 scopes for access control."""
    OPENID = "openid"
    PROFILE = "profile"
    EMAIL = "email"
    AGENTS_READ = "agents:read"
    AGENTS_WRITE = "agents:write"
    TASKS_READ = "tasks:read"
    TASKS_WRITE = "tasks:write"
    ADMIN = "admin"
    ANALYTICS = "analytics"


class OAuth2ClientType(Enum):
    """OAuth2 client types."""
    CONFIDENTIAL = "confidential"  # Can maintain credential confidentiality
    PUBLIC = "public"  # Cannot maintain credential confidentiality


class TokenType(Enum):
    """Token types."""
    BEARER = "Bearer"
    MAC = "mac"


# Pydantic Models

class OAuth2Client(BaseModel):
    """OAuth2 client configuration."""
    client_id: str
    client_name: str
    client_type: OAuth2ClientType
    client_secret: Optional[str] = None
    redirect_uris: List[str] = []
    allowed_scopes: List[OAuth2Scope] = []
    allowed_grant_types: List[OAuth2GrantType] = []
    is_active: bool = True
    created_by: str
    organization_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class OAuth2AuthorizationCode(BaseModel):
    """OAuth2 authorization code."""
    code: str
    client_id: str
    user_id: str
    redirect_uri: str
    scopes: List[str]
    code_challenge: Optional[str] = None
    code_challenge_method: Optional[str] = None
    expires_at: datetime
    used: bool = False


class OAuth2AccessToken(BaseModel):
    """OAuth2 access token."""
    access_token: str
    token_type: TokenType = TokenType.BEARER
    expires_in: int
    refresh_token: Optional[str] = None
    scope: str
    client_id: str
    user_id: Optional[str] = None


class OAuth2RefreshToken(BaseModel):
    """OAuth2 refresh token."""
    refresh_token: str
    client_id: str
    user_id: str
    scopes: List[str]
    expires_at: datetime
    revoked: bool = False


class OAuth2TokenRequest(BaseModel):
    """OAuth2 token request."""
    grant_type: OAuth2GrantType
    code: Optional[str] = None
    redirect_uri: Optional[str] = None
    client_id: str
    client_secret: Optional[str] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    code_verifier: Optional[str] = None


class OAuth2AuthorizeRequest(BaseModel):
    """OAuth2 authorization request."""
    response_type: str
    client_id: str
    redirect_uri: str
    scope: Optional[str] = None
    state: Optional[str] = None
    code_challenge: Optional[str] = None
    code_challenge_method: Optional[str] = None
    nonce: Optional[str] = None


class OAuth2TokenResponse(BaseModel):
    """OAuth2 token response."""
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    scope: str
    id_token: Optional[str] = None


class OAuth2ErrorResponse(BaseModel):
    """OAuth2 error response."""
    error: str
    error_description: Optional[str] = None
    error_uri: Optional[str] = None
    state: Optional[str] = None


class OAuth2Provider:
    """
    Comprehensive OAuth2 Provider implementation.
    
    Supports multiple OAuth2 flows with enterprise security features:
    - Authorization Code Flow (with PKCE)
    - Client Credentials Flow
    - Refresh Token Flow
    - Device Code Flow
    - OpenID Connect integration
    """
    
    def __init__(self, auth_service: AuthenticationService):
        self.auth_service = auth_service
        self.security_system = None
        
        # In-memory stores (should be replaced with database models)
        self._clients: Dict[str, OAuth2Client] = {}
        self._authorization_codes: Dict[str, OAuth2AuthorizationCode] = {}
        self._access_tokens: Dict[str, Dict[str, Any]] = {}
        self._refresh_tokens: Dict[str, OAuth2RefreshToken] = {}
        
        # Configuration
        self.config = {
            "authorization_code_lifetime": 600,  # 10 minutes
            "access_token_lifetime": 3600,  # 1 hour
            "refresh_token_lifetime": 86400 * 30,  # 30 days
            "require_pkce": True,
            "require_state": True,
            "enable_openid_connect": True,
            "issuer": settings.OAUTH2_ISSUER or "https://api.leanvibe.com",
        }
        
        # Initialize default system client
        self._create_system_clients()
    
    async def initialize(self):
        """Initialize async components."""
        self.security_system = await get_security_system()
        logger.info("OAuth2 provider initialized")
    
    def _create_system_clients(self):
        """Create default system clients."""
        # PWA Client (public client with PKCE)
        pwa_client = OAuth2Client(
            client_id="leanvibe-pwa",
            client_name="LeanVibe PWA",
            client_type=OAuth2ClientType.PUBLIC,
            redirect_uris=["http://localhost:3000/auth/callback", "https://app.leanvibe.com/auth/callback"],
            allowed_scopes=[
                OAuth2Scope.OPENID, OAuth2Scope.PROFILE, OAuth2Scope.EMAIL,
                OAuth2Scope.AGENTS_READ, OAuth2Scope.AGENTS_WRITE,
                OAuth2Scope.TASKS_READ, OAuth2Scope.TASKS_WRITE
            ],
            allowed_grant_types=[OAuth2GrantType.AUTHORIZATION_CODE, OAuth2GrantType.REFRESH_TOKEN],
            created_by="system",
            metadata={"description": "Default PWA client"}
        )
        self._clients[pwa_client.client_id] = pwa_client
        
        # API Client (confidential client)
        api_client_secret = secrets.token_urlsafe(64)
        api_client = OAuth2Client(
            client_id="leanvibe-api",
            client_name="LeanVibe API Client",
            client_type=OAuth2ClientType.CONFIDENTIAL,
            client_secret=api_client_secret,
            allowed_scopes=[OAuth2Scope.AGENTS_READ, OAuth2Scope.AGENTS_WRITE, OAuth2Scope.TASKS_READ, OAuth2Scope.TASKS_WRITE],
            allowed_grant_types=[OAuth2GrantType.CLIENT_CREDENTIALS],
            created_by="system",
            metadata={"description": "Default API client for service-to-service communication"}
        )
        self._clients[api_client.client_id] = api_client
        
        logger.info("System OAuth2 clients created",
                   pwa_client_id=pwa_client.client_id,
                   api_client_id=api_client.client_id)
    
    # Client Management
    
    async def register_client(self, client_request: Dict[str, Any], created_by: str) -> OAuth2Client:
        """Register new OAuth2 client."""
        try:
            client_id = f"client_{secrets.token_urlsafe(16)}"
            client_secret = None
            
            if client_request.get("client_type") == OAuth2ClientType.CONFIDENTIAL.value:
                client_secret = secrets.token_urlsafe(64)
            
            client = OAuth2Client(
                client_id=client_id,
                client_name=client_request["client_name"],
                client_type=OAuth2ClientType(client_request["client_type"]),
                client_secret=client_secret,
                redirect_uris=client_request.get("redirect_uris", []),
                allowed_scopes=[OAuth2Scope(scope) for scope in client_request.get("allowed_scopes", [])],
                allowed_grant_types=[OAuth2GrantType(gt) for gt in client_request.get("allowed_grant_types", [])],
                created_by=created_by,
                organization_id=client_request.get("organization_id"),
                metadata=client_request.get("metadata", {})
            )
            
            self._clients[client_id] = client
            
            if self.security_system:
                await self.security_system.log_security_event(
                    SecurityEvent.CONFIGURATION_CHANGED,
                    user_id=created_by,
                    client_id=client_id,
                    action="oauth2_client_registered"
                )
            
            logger.info("OAuth2 client registered", client_id=client_id, client_name=client.client_name)
            return client
            
        except Exception as e:
            logger.error("Client registration failed", error=str(e))
            raise HTTPException(status_code=400, detail=f"Client registration failed: {str(e)}")
    
    async def get_client(self, client_id: str) -> Optional[OAuth2Client]:
        """Get OAuth2 client by ID."""
        return self._clients.get(client_id)
    
    async def authenticate_client(self, client_id: str, client_secret: Optional[str] = None) -> bool:
        """Authenticate OAuth2 client."""
        client = await self.get_client(client_id)
        if not client or not client.is_active:
            return False
        
        if client.client_type == OAuth2ClientType.CONFIDENTIAL:
            if not client_secret or client.client_secret != client_secret:
                return False
        
        return True
    
    # Authorization Code Flow
    
    async def create_authorization_code(
        self,
        client_id: str,
        user_id: str,
        redirect_uri: str,
        scopes: List[str],
        code_challenge: Optional[str] = None,
        code_challenge_method: Optional[str] = None
    ) -> str:
        """Create authorization code for OAuth2 flow."""
        try:
            code = secrets.token_urlsafe(32)
            expires_at = datetime.utcnow() + timedelta(seconds=self.config["authorization_code_lifetime"])
            
            auth_code = OAuth2AuthorizationCode(
                code=code,
                client_id=client_id,
                user_id=user_id,
                redirect_uri=redirect_uri,
                scopes=scopes,
                code_challenge=code_challenge,
                code_challenge_method=code_challenge_method,
                expires_at=expires_at
            )
            
            self._authorization_codes[code] = auth_code
            
            logger.info("Authorization code created", client_id=client_id, user_id=user_id)
            return code
            
        except Exception as e:
            logger.error("Authorization code creation failed", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to create authorization code")
    
    async def exchange_authorization_code(
        self,
        code: str,
        client_id: str,
        redirect_uri: str,
        code_verifier: Optional[str] = None
    ) -> OAuth2TokenResponse:
        """Exchange authorization code for access token."""
        try:
            auth_code = self._authorization_codes.get(code)
            if not auth_code:
                raise HTTPException(status_code=400, detail="Invalid authorization code")
            
            # Validate authorization code
            if auth_code.used:
                raise HTTPException(status_code=400, detail="Authorization code already used")
            
            if auth_code.expires_at < datetime.utcnow():
                raise HTTPException(status_code=400, detail="Authorization code expired")
            
            if auth_code.client_id != client_id:
                raise HTTPException(status_code=400, detail="Client ID mismatch")
            
            if auth_code.redirect_uri != redirect_uri:
                raise HTTPException(status_code=400, detail="Redirect URI mismatch")
            
            # Validate PKCE if required
            if auth_code.code_challenge:
                if not code_verifier:
                    raise HTTPException(status_code=400, detail="Code verifier required")
                
                if not self._verify_pkce_challenge(code_verifier, auth_code.code_challenge, auth_code.code_challenge_method):
                    raise HTTPException(status_code=400, detail="Invalid code verifier")
            
            # Mark code as used
            auth_code.used = True
            
            # Create tokens
            tokens = await self._create_tokens(client_id, auth_code.user_id, auth_code.scopes)
            
            if self.security_system:
                await self.security_system.log_security_event(
                    SecurityEvent.LOGIN_SUCCESS,
                    user_id=auth_code.user_id,
                    client_id=client_id,
                    action="oauth2_code_exchange"
                )
            
            return tokens
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Authorization code exchange failed", error=str(e))
            raise HTTPException(status_code=500, detail="Token exchange failed")
    
    # Client Credentials Flow
    
    async def client_credentials_grant(self, client_id: str, scopes: List[str]) -> OAuth2TokenResponse:
        """Handle client credentials grant."""
        try:
            client = await self.get_client(client_id)
            if not client:
                raise HTTPException(status_code=400, detail="Invalid client")
            
            if OAuth2GrantType.CLIENT_CREDENTIALS not in client.allowed_grant_types:
                raise HTTPException(status_code=400, detail="Client credentials grant not allowed")
            
            # Validate scopes
            requested_scopes = [OAuth2Scope(scope) for scope in scopes if scope]
            invalid_scopes = set(requested_scopes) - set(client.allowed_scopes)
            if invalid_scopes:
                raise HTTPException(status_code=400, detail=f"Invalid scopes: {invalid_scopes}")
            
            # Create tokens (no user_id for client credentials)
            tokens = await self._create_tokens(client_id, None, scopes)
            
            if self.security_system:
                await self.security_system.log_security_event(
                    SecurityEvent.LOGIN_SUCCESS,
                    client_id=client_id,
                    action="oauth2_client_credentials"
                )
            
            return tokens
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Client credentials grant failed", error=str(e))
            raise HTTPException(status_code=500, detail="Client credentials grant failed")
    
    # Refresh Token Flow
    
    async def refresh_token_grant(self, refresh_token_str: str, client_id: str) -> OAuth2TokenResponse:
        """Handle refresh token grant."""
        try:
            refresh_token = self._refresh_tokens.get(refresh_token_str)
            if not refresh_token:
                raise HTTPException(status_code=400, detail="Invalid refresh token")
            
            if refresh_token.revoked:
                raise HTTPException(status_code=400, detail="Refresh token revoked")
            
            if refresh_token.expires_at < datetime.utcnow():
                raise HTTPException(status_code=400, detail="Refresh token expired")
            
            if refresh_token.client_id != client_id:
                raise HTTPException(status_code=400, detail="Client ID mismatch")
            
            # Create new tokens
            tokens = await self._create_tokens(client_id, refresh_token.user_id, refresh_token.scopes)
            
            # Optionally rotate refresh token (security best practice)
            if True:  # Always rotate for security
                refresh_token.revoked = True
            
            if self.security_system:
                await self.security_system.log_security_event(
                    SecurityEvent.LOGIN_SUCCESS,
                    user_id=refresh_token.user_id,
                    client_id=client_id,
                    action="oauth2_token_refresh"
                )
            
            return tokens
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Refresh token grant failed", error=str(e))
            raise HTTPException(status_code=500, detail="Refresh token grant failed")
    
    # Token Management
    
    async def _create_tokens(self, client_id: str, user_id: Optional[str], scopes: List[str]) -> OAuth2TokenResponse:
        """Create access and refresh tokens."""
        # Create access token
        access_token = secrets.token_urlsafe(32)
        expires_in = self.config["access_token_lifetime"]
        
        # Store access token info
        self._access_tokens[access_token] = {
            "client_id": client_id,
            "user_id": user_id,
            "scopes": scopes,
            "expires_at": datetime.utcnow() + timedelta(seconds=expires_in),
            "created_at": datetime.utcnow()
        }
        
        # Create refresh token (if user is involved)
        refresh_token_str = None
        if user_id:
            refresh_token_str = secrets.token_urlsafe(32)
            refresh_token = OAuth2RefreshToken(
                refresh_token=refresh_token_str,
                client_id=client_id,
                user_id=user_id,
                scopes=scopes,
                expires_at=datetime.utcnow() + timedelta(seconds=self.config["refresh_token_lifetime"])
            )
            self._refresh_tokens[refresh_token_str] = refresh_token
        
        # Create ID token for OpenID Connect
        id_token = None
        if self.config["enable_openid_connect"] and OAuth2Scope.OPENID.value in scopes and user_id:
            id_token = await self._create_id_token(user_id, client_id, scopes)
        
        return OAuth2TokenResponse(
            access_token=access_token,
            token_type="Bearer",
            expires_in=expires_in,
            refresh_token=refresh_token_str,
            scope=" ".join(scopes),
            id_token=id_token
        )
    
    async def _create_id_token(self, user_id: str, client_id: str, scopes: List[str]) -> str:
        """Create OpenID Connect ID token."""
        user = self.auth_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=500, detail="User not found")
        
        now = datetime.utcnow()
        expires = now + timedelta(hours=1)
        
        claims = {
            "iss": self.config["issuer"],
            "sub": user_id,
            "aud": client_id,
            "exp": expires,
            "iat": now,
            "auth_time": now,
        }
        
        # Add profile claims based on scopes
        if OAuth2Scope.PROFILE.value in scopes:
            claims.update({
                "name": user.full_name,
                "preferred_username": user.email.split("@")[0],
                "updated_at": user.created_at.timestamp()
            })
        
        if OAuth2Scope.EMAIL.value in scopes:
            claims.update({
                "email": user.email,
                "email_verified": True
            })
        
        return jwt.encode(claims, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    
    async def validate_access_token(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Validate OAuth2 access token."""
        token_info = self._access_tokens.get(access_token)
        if not token_info:
            return None
        
        if token_info["expires_at"] < datetime.utcnow():
            # Token expired, remove it
            del self._access_tokens[access_token]
            return None
        
        return token_info
    
    async def revoke_token(self, token: str, token_type_hint: str = "access_token") -> bool:
        """Revoke OAuth2 token."""
        try:
            if token_type_hint == "access_token" or token in self._access_tokens:
                if token in self._access_tokens:
                    del self._access_tokens[token]
                    return True
            
            if token_type_hint == "refresh_token" or token in self._refresh_tokens:
                if token in self._refresh_tokens:
                    self._refresh_tokens[token].revoked = True
                    return True
            
            return False
            
        except Exception as e:
            logger.error("Token revocation failed", error=str(e))
            return False
    
    # PKCE Support
    
    def _verify_pkce_challenge(self, verifier: str, challenge: str, method: Optional[str] = "S256") -> bool:
        """Verify PKCE code challenge."""
        if method == "plain":
            return verifier == challenge
        elif method == "S256":
            computed_challenge = base64.urlsafe_b64encode(
                hashlib.sha256(verifier.encode()).digest()
            ).decode().rstrip("=")
            return computed_challenge == challenge
        
        return False
    
    # Scope Validation
    
    def validate_scopes(self, client: OAuth2Client, requested_scopes: List[str]) -> List[str]:
        """Validate and filter scopes for client."""
        if not requested_scopes:
            return []
        
        valid_scopes = []
        for scope in requested_scopes:
            try:
                scope_enum = OAuth2Scope(scope)
                if scope_enum in client.allowed_scopes:
                    valid_scopes.append(scope)
            except ValueError:
                continue
        
        return valid_scopes
    
    # Utility Methods
    
    def get_authorization_server_metadata(self) -> Dict[str, Any]:
        """Get OAuth2 authorization server metadata (RFC 8414)."""
        return {
            "issuer": self.config["issuer"],
            "authorization_endpoint": f"{self.config['issuer']}/oauth2/authorize",
            "token_endpoint": f"{self.config['issuer']}/oauth2/token",
            "revocation_endpoint": f"{self.config['issuer']}/oauth2/revoke",
            "introspection_endpoint": f"{self.config['issuer']}/oauth2/introspect",
            "jwks_uri": f"{self.config['issuer']}/oauth2/jwks",
            "response_types_supported": ["code", "token"],
            "grant_types_supported": [
                "authorization_code",
                "client_credentials", 
                "refresh_token"
            ],
            "token_endpoint_auth_methods_supported": [
                "client_secret_basic",
                "client_secret_post",
                "none"
            ],
            "scopes_supported": [scope.value for scope in OAuth2Scope],
            "code_challenge_methods_supported": ["S256", "plain"],
            "service_documentation": f"{self.config['issuer']}/docs"
        }


# Global OAuth2 provider instance
_oauth2_provider: Optional[OAuth2Provider] = None


async def get_oauth2_provider() -> OAuth2Provider:
    """Get or create OAuth2 provider instance."""
    global _oauth2_provider
    if _oauth2_provider is None:
        auth_service = get_auth_service()
        _oauth2_provider = OAuth2Provider(auth_service)
        await _oauth2_provider.initialize()
    return _oauth2_provider


# FastAPI Dependencies

async def get_oauth2_token_info(
    authorization: str = Depends(HTTPBearer())
) -> Dict[str, Any]:
    """Extract and validate OAuth2 token information."""
    if not authorization or not authorization.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing access token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    provider = await get_oauth2_provider()
    token_info = await provider.validate_access_token(authorization.credentials)
    
    if not token_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return token_info


def require_oauth2_scope(required_scope: str):
    """Require specific OAuth2 scope."""
    async def scope_checker(token_info: Dict[str, Any] = Depends(get_oauth2_token_info)):
        if required_scope not in token_info.get("scopes", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient scope. Required: {required_scope}"
            )
        return token_info
    
    return scope_checker


def require_oauth2_scopes(required_scopes: List[str], require_all: bool = True):
    """Require multiple OAuth2 scopes."""
    async def scopes_checker(token_info: Dict[str, Any] = Depends(get_oauth2_token_info)):
        user_scopes = set(token_info.get("scopes", []))
        required_scopes_set = set(required_scopes)
        
        if require_all:
            if not required_scopes_set.issubset(user_scopes):
                missing = required_scopes_set - user_scopes
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient scopes. Missing: {list(missing)}"
                )
        else:
            if not required_scopes_set.intersection(user_scopes):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient scopes. Required any of: {required_scopes}"
                )
        
        return token_info
    
    return scopes_checker


# Export components
__all__ = [
    "OAuth2Provider", "get_oauth2_provider", "get_oauth2_token_info",
    "require_oauth2_scope", "require_oauth2_scopes",
    "OAuth2GrantType", "OAuth2Scope", "OAuth2ClientType",
    "OAuth2TokenResponse", "OAuth2ErrorResponse"
]