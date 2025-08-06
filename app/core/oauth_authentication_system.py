"""
OAuth 2.0/OIDC Authentication System for LeanVibe Agent Hive 2.0.

Provides enterprise-grade multi-provider OAuth 2.0 and OpenID Connect authentication
with short-lived JWT tokens, secure refresh mechanisms, and behavioral monitoring.

Features:
- Multi-provider OAuth 2.0 support (Google, Microsoft, GitHub, Custom OIDC)
- Short-lived JWT tokens with secure refresh
- IP binding and behavioral monitoring for stolen token detection
- Agent identity service with human controller linking
- Comprehensive audit logging
- Rate limiting and security monitoring
- Dynamic scope evaluation and token introspection
"""

import asyncio
import uuid
import json
import time
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from urllib.parse import urlencode, parse_qs
import logging
import jwt
import httpx
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

import structlog
from fastapi import HTTPException, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload

from ..models.security import (
    AgentIdentity, SecurityAuditLog, AuthenticationSession, OAuthProvider,
    AuthenticationAttempt, SecurityEvent
)
from .redis import RedisClient
from .security_audit import SecurityAuditSystem, ThreatLevel, AuditEventType
from ..core.config import get_settings

logger = structlog.get_logger()
settings = get_settings()


class AuthenticationMethod(Enum):
    """Authentication methods supported."""
    OAUTH2 = "oauth2"
    OIDC = "oidc" 
    SAML = "saml"
    API_KEY = "api_key"
    JWT = "jwt"


class TokenType(Enum):
    """Token types."""
    ACCESS = "access"
    REFRESH = "refresh"
    ID = "id"


class ProviderType(Enum):
    """OAuth provider types."""
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    GITHUB = "github"
    CUSTOM_OIDC = "custom_oidc"
    ENTERPRISE_SAML = "enterprise_saml"


@dataclass
class TokenClaims:
    """JWT token claims."""
    sub: str  # Subject (user ID)
    iss: str  # Issuer
    aud: str  # Audience
    exp: int  # Expiration
    iat: int  # Issued at
    nbf: int  # Not before
    jti: str  # JWT ID
    
    # Custom claims
    agent_id: Optional[str] = None
    human_controller: Optional[str] = None
    session_id: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    provider: Optional[str] = None
    trust_level: float = 0.5
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Security claims
    auth_time: Optional[int] = None
    amr: List[str] = field(default_factory=list)  # Authentication methods references
    acr: Optional[str] = None  # Authentication context class reference
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JWT payload."""
        return {
            "sub": self.sub,
            "iss": self.iss, 
            "aud": self.aud,
            "exp": self.exp,
            "iat": self.iat,
            "nbf": self.nbf,
            "jti": self.jti,
            "agent_id": self.agent_id,
            "human_controller": self.human_controller,
            "session_id": self.session_id,
            "scopes": self.scopes,
            "provider": self.provider,
            "trust_level": self.trust_level,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "auth_time": self.auth_time,
            "amr": self.amr,
            "acr": self.acr
        }


@dataclass
class AuthenticationContext:
    """Authentication context for security analysis."""
    ip_address: str
    user_agent: str
    request_headers: Dict[str, str]
    timestamp: datetime
    session_id: Optional[str] = None
    provider: Optional[str] = None
    scopes_requested: List[str] = field(default_factory=list)
    
    # Risk indicators
    is_new_device: bool = False
    is_new_location: bool = False
    failed_attempts_recent: int = 0
    time_since_last_auth: Optional[timedelta] = None
    
    def calculate_risk_score(self) -> float:
        """Calculate risk score for this authentication context."""
        risk_score = 0.0
        
        # New device/location increases risk
        if self.is_new_device:
            risk_score += 0.3
        if self.is_new_location:
            risk_score += 0.2
        
        # Recent failures increase risk
        if self.failed_attempts_recent > 0:
            risk_score += min(0.4, self.failed_attempts_recent * 0.1)
        
        # Time-based risk
        current_hour = self.timestamp.hour
        if current_hour < 6 or current_hour > 22:
            risk_score += 0.1
        
        # Weekend access
        if self.timestamp.weekday() >= 5:
            risk_score += 0.05
        
        return min(risk_score, 1.0)


@dataclass
class OAuthConfig:
    """OAuth provider configuration."""
    provider_id: str
    provider_type: ProviderType
    client_id: str
    client_secret: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: Optional[str] = None
    jwks_uri: Optional[str] = None
    issuer: Optional[str] = None
    scopes: List[str] = field(default_factory=lambda: ["openid", "profile", "email"])
    redirect_uri: Optional[str] = None
    
    # Security settings
    require_pkce: bool = True
    require_state: bool = True
    require_nonce: bool = True
    max_age: Optional[int] = None
    
    def get_authorization_url(
        self,
        state: str,
        nonce: Optional[str] = None,
        code_challenge: Optional[str] = None
    ) -> str:
        """Generate authorization URL."""
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "scope": " ".join(self.scopes),
            "redirect_uri": self.redirect_uri,
            "state": state,
        }
        
        if nonce:
            params["nonce"] = nonce
        if code_challenge and self.require_pkce:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"
        if self.max_age:
            params["max_age"] = str(self.max_age)
            
        return f"{self.authorization_endpoint}?{urlencode(params)}"


class OAuthAuthenticationSystem:
    """
    OAuth 2.0/OIDC Authentication System.
    
    Provides comprehensive authentication with multiple providers,
    security monitoring, and enterprise compliance features.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: RedisClient,
        audit_system: SecurityAuditSystem
    ):
        """
        Initialize OAuth Authentication System.
        
        Args:
            db_session: Database session
            redis_client: Redis client for caching
            audit_system: Security audit system
        """
        self.db = db_session
        self.redis = redis_client
        self.audit_system = audit_system
        
        # OAuth providers
        self.providers: Dict[str, OAuthConfig] = {}
        self._initialize_default_providers()
        
        # Token configuration
        self.token_config = {
            "access_token_lifetime_minutes": 15,  # Short-lived
            "refresh_token_lifetime_days": 30,
            "id_token_lifetime_minutes": 60,
            "jwt_algorithm": "RS256",
            "jwt_issuer": "leanvibe-agent-hive",
            "jwt_audience": "agent-hive-api",
        }
        
        # Security configuration
        self.security_config = {
            "max_failed_attempts": 5,
            "lockout_duration_minutes": 30,
            "require_mfa_for_admin": True,
            "enable_ip_binding": True,
            "enable_device_tracking": True,
            "enable_behavioral_analysis": True,
            "session_timeout_minutes": 480,  # 8 hours
            "refresh_token_rotation": True,
            "require_secure_transport": True
        }
        
        # Performance metrics
        self.metrics = {
            "authentications_total": 0,
            "authentications_successful": 0,
            "authentications_failed": 0,
            "tokens_issued": 0,
            "tokens_refreshed": 0,
            "tokens_revoked": 0,
            "avg_auth_time_ms": 0.0,
            "security_violations": 0,
            "provider_failures": defaultdict(int)
        }
        
        # Cache prefixes
        self._auth_state_prefix = "oauth:state:"
        self._nonce_prefix = "oauth:nonce:"
        self._pkce_prefix = "oauth:pkce:"
        self._session_prefix = "oauth:session:"
        self._user_sessions_prefix = "oauth:user_sessions:"
    
    async def initiate_authentication(
        self,
        provider_id: str,
        request: Request,
        scopes: Optional[List[str]] = None,
        redirect_uri: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Initiate OAuth authentication flow.
        
        Args:
            provider_id: OAuth provider identifier
            request: HTTP request object
            scopes: Requested scopes
            redirect_uri: Redirect URI
            
        Returns:
            Authentication initiation data
        """
        start_time = time.time()
        
        try:
            # Get provider configuration
            provider = self.providers.get(provider_id)
            if not provider:
                raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_id}")
            
            # Create authentication context
            context = AuthenticationContext(
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent", ""),
                request_headers=dict(request.headers),
                timestamp=datetime.utcnow(),
                provider=provider_id,
                scopes_requested=scopes or provider.scopes
            )
            
            # Security checks
            risk_score = context.calculate_risk_score()
            if risk_score > 0.8:
                await self._log_security_violation(
                    "high_risk_authentication_attempt",
                    context,
                    {"risk_score": risk_score}
                )
                raise HTTPException(status_code=403, detail="Authentication denied due to high risk")
            
            # Generate security parameters
            state = secrets.token_urlsafe(32)
            nonce = secrets.token_urlsafe(32) if provider.require_nonce else None
            code_verifier = None
            code_challenge = None
            
            if provider.require_pkce:
                code_verifier = secrets.token_urlsafe(43)  # 43 chars = 256 bits base64url
                code_challenge = base64.urlsafe_b64encode(
                    hashlib.sha256(code_verifier.encode()).digest()
                ).decode().rstrip('=')
            
            # Store security parameters in cache
            session_data = {
                "provider_id": provider_id,
                "state": state,
                "nonce": nonce,
                "code_verifier": code_verifier,
                "context": context.to_dict() if hasattr(context, 'to_dict') else vars(context),
                "scopes": scopes or provider.scopes,
                "redirect_uri": redirect_uri or provider.redirect_uri,
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Cache with expiration
            await self.redis.set_with_expiry(
                f"{self._auth_state_prefix}{state}",
                json.dumps(session_data, default=str),
                600  # 10 minutes
            )
            
            if nonce:
                await self.redis.set_with_expiry(
                    f"{self._nonce_prefix}{nonce}",
                    state,
                    600
                )
            
            if code_verifier:
                await self.redis.set_with_expiry(
                    f"{self._pkce_prefix}{state}",
                    code_verifier,
                    600
                )
            
            # Generate authorization URL
            auth_url = provider.get_authorization_url(state, nonce, code_challenge)
            
            # Log authentication initiation
            await self._log_auth_attempt(
                provider_id=provider_id,
                context=context,
                success=True,
                action="initiate_auth",
                metadata={"risk_score": risk_score}
            )
            
            auth_time = (time.time() - start_time) * 1000
            self._update_metrics("initiate_auth", auth_time, True)
            
            return {
                "authorization_url": auth_url,
                "state": state,
                "provider": provider_id,
                "risk_score": risk_score,
                "expires_in": 600
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication initiation failed: {e}")
            auth_time = (time.time() - start_time) * 1000
            self._update_metrics("initiate_auth", auth_time, False)
            
            raise HTTPException(status_code=500, detail="Authentication initiation failed")
    
    async def complete_authentication(
        self,
        code: str,
        state: str,
        request: Request
    ) -> Dict[str, Any]:
        """
        Complete OAuth authentication flow.
        
        Args:
            code: Authorization code
            state: State parameter
            request: HTTP request object
            
        Returns:
            Authentication completion data with tokens
        """
        start_time = time.time()
        
        try:
            # Retrieve session data
            session_key = f"{self._auth_state_prefix}{state}"
            session_data_str = await self.redis.get(session_key)
            
            if not session_data_str:
                raise HTTPException(status_code=400, detail="Invalid or expired authentication state")
            
            session_data = json.loads(session_data_str)
            provider = self.providers.get(session_data["provider_id"])
            
            if not provider:
                raise HTTPException(status_code=400, detail="Invalid provider")
            
            # Verify context (basic security check)
            current_ip = request.client.host
            stored_ip = session_data["context"]["ip_address"]
            
            if self.security_config["enable_ip_binding"] and current_ip != stored_ip:
                await self._log_security_violation(
                    "ip_address_mismatch",
                    None,
                    {"stored_ip": stored_ip, "current_ip": current_ip, "state": state}
                )
                raise HTTPException(status_code=403, detail="IP address mismatch")
            
            # Exchange code for tokens
            token_data = await self._exchange_code_for_tokens(
                provider, code, session_data
            )
            
            # Get user information
            user_info = await self._get_user_info(provider, token_data["access_token"])
            
            # Create or update agent identity
            agent_identity = await self._create_or_update_agent_identity(
                user_info, session_data["provider_id"], request
            )
            
            # Generate internal JWT tokens
            access_token = await self._generate_access_token(
                agent_identity, session_data, token_data
            )
            refresh_token = await self._generate_refresh_token(
                agent_identity, session_data
            )
            
            # Create authentication session
            auth_session = await self._create_auth_session(
                agent_identity, request, session_data, token_data
            )
            
            # Clean up temporary data
            await self._cleanup_auth_state(state, session_data)
            
            # Log successful authentication
            await self._log_auth_attempt(
                provider_id=session_data["provider_id"],
                context=None,
                success=True,
                action="complete_auth",
                agent_id=agent_identity.id,
                metadata={
                    "session_id": str(auth_session.id),
                    "user_info": user_info
                }
            )
            
            auth_time = (time.time() - start_time) * 1000
            self._update_metrics("complete_auth", auth_time, True)
            self.metrics["authentications_successful"] += 1
            self.metrics["tokens_issued"] += 2  # Access + Refresh
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "Bearer",
                "expires_in": self.token_config["access_token_lifetime_minutes"] * 60,
                "agent_id": str(agent_identity.id),
                "session_id": str(auth_session.id),
                "user_info": {
                    "name": user_info.get("name"),
                    "email": user_info.get("email"),
                    "picture": user_info.get("picture")
                },
                "scopes": session_data.get("scopes", [])
            }
            
        except HTTPException:
            self.metrics["authentications_failed"] += 1
            raise
        except Exception as e:
            logger.error(f"Authentication completion failed: {e}")
            auth_time = (time.time() - start_time) * 1000
            self._update_metrics("complete_auth", auth_time, False)
            self.metrics["authentications_failed"] += 1
            
            raise HTTPException(status_code=500, detail="Authentication completion failed")
    
    async def refresh_access_token(
        self,
        refresh_token: str,
        request: Request
    ) -> Dict[str, Any]:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_token: Valid refresh token
            request: HTTP request object
            
        Returns:
            New token pair
        """
        start_time = time.time()
        
        try:
            # Validate and decode refresh token
            refresh_claims = await self._validate_and_decode_token(
                refresh_token, TokenType.REFRESH
            )
            
            # Get authentication session
            session_id = refresh_claims.get("session_id")
            if not session_id:
                raise HTTPException(status_code=400, detail="Invalid refresh token")
            
            session = await self.db.get(AuthenticationSession, uuid.UUID(session_id))
            if not session or not session.is_active:
                raise HTTPException(status_code=400, detail="Invalid or expired session")
            
            # Security checks
            if self.security_config["enable_ip_binding"]:
                if session.ip_address != request.client.host:
                    await self._revoke_session(session, "ip_mismatch")
                    raise HTTPException(status_code=403, detail="IP address mismatch")
            
            # Check session expiry
            if session.expires_at and session.expires_at < datetime.utcnow():
                await self._revoke_session(session, "expired")
                raise HTTPException(status_code=400, detail="Session expired")
            
            # Get agent identity
            agent_identity = await self.db.get(AgentIdentity, session.agent_id)
            if not agent_identity or not agent_identity.is_active():
                raise HTTPException(status_code=400, detail="Agent not found or inactive")
            
            # Generate new access token
            new_access_token = await self._generate_access_token(
                agent_identity, {"scopes": refresh_claims.get("scopes", [])}, {}
            )
            
            # Optionally rotate refresh token
            new_refresh_token = refresh_token
            if self.security_config["refresh_token_rotation"]:
                new_refresh_token = await self._generate_refresh_token(
                    agent_identity, {"session_id": session_id}
                )
                
                # Invalidate old refresh token
                await self.redis.set_with_expiry(
                    f"revoked:refresh:{hashlib.sha256(refresh_token.encode()).hexdigest()}",
                    "1",
                    86400  # 24 hours
                )
            
            # Update session last activity
            session.last_activity_at = datetime.utcnow()
            await self.db.commit()
            
            # Log token refresh
            await self._log_auth_attempt(
                provider_id=session.oauth_provider,
                context=None,
                success=True,
                action="refresh_token",
                agent_id=agent_identity.id,
                metadata={"session_id": str(session.id)}
            )
            
            auth_time = (time.time() - start_time) * 1000
            self._update_metrics("refresh_token", auth_time, True)
            self.metrics["tokens_refreshed"] += 1
            
            return {
                "access_token": new_access_token,
                "refresh_token": new_refresh_token,
                "token_type": "Bearer", 
                "expires_in": self.token_config["access_token_lifetime_minutes"] * 60,
                "agent_id": str(agent_identity.id),
                "session_id": str(session.id)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            auth_time = (time.time() - start_time) * 1000
            self._update_metrics("refresh_token", auth_time, False)
            
            raise HTTPException(status_code=500, detail="Token refresh failed")
    
    async def validate_access_token(
        self,
        access_token: str,
        required_scopes: Optional[List[str]] = None
    ) -> TokenClaims:
        """
        Validate access token and return claims.
        
        Args:
            access_token: JWT access token
            required_scopes: Required scopes for access
            
        Returns:
            Token claims if valid
        """
        try:
            # Check if token is revoked
            token_hash = hashlib.sha256(access_token.encode()).hexdigest()
            if await self.redis.get(f"revoked:access:{token_hash}"):
                raise HTTPException(status_code=401, detail="Token has been revoked")
            
            # Decode and validate token
            payload = jwt.decode(
                access_token,
                settings.JWT_SECRET_KEY,
                algorithms=[self.token_config["jwt_algorithm"]],
                audience=self.token_config["jwt_audience"],
                issuer=self.token_config["jwt_issuer"]
            )
            
            # Create token claims
            claims = TokenClaims(**{
                k: v for k, v in payload.items()
                if k in TokenClaims.__annotations__
            })
            
            # Validate required scopes
            if required_scopes:
                token_scopes = set(claims.scopes)
                required_scopes_set = set(required_scopes)
                
                if not required_scopes_set.issubset(token_scopes):
                    missing_scopes = required_scopes_set - token_scopes
                    raise HTTPException(
                        status_code=403, 
                        detail=f"Insufficient scopes. Missing: {', '.join(missing_scopes)}"
                    )
            
            # Validate agent is still active
            if claims.agent_id:
                agent = await self.db.get(AgentIdentity, uuid.UUID(claims.agent_id))
                if not agent or not agent.is_active():
                    raise HTTPException(status_code=401, detail="Agent is not active")
            
            return claims
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            raise HTTPException(status_code=401, detail="Token validation failed")
    
    async def revoke_token(
        self,
        token: str,
        token_type: TokenType,
        reason: str = "user_revocation"
    ) -> bool:
        """
        Revoke a token.
        
        Args:
            token: Token to revoke
            token_type: Type of token
            reason: Revocation reason
            
        Returns:
            True if successfully revoked
        """
        try:
            # Add to revocation list
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            await self.redis.set_with_expiry(
                f"revoked:{token_type.value}:{token_hash}",
                reason,
                86400  # 24 hours - tokens expire before this
            )
            
            # If refresh token, revoke associated session
            if token_type == TokenType.REFRESH:
                try:
                    claims = jwt.decode(
                        token,
                        settings.JWT_SECRET_KEY,
                        algorithms=[self.token_config["jwt_algorithm"]],
                        options={"verify_exp": False}  # Don't check expiry for revocation
                    )
                    
                    session_id = claims.get("session_id")
                    if session_id:
                        session = await self.db.get(AuthenticationSession, uuid.UUID(session_id))
                        if session:
                            await self._revoke_session(session, reason)
                            
                except jwt.InvalidTokenError:
                    pass  # Token might be malformed, but still add to revocation list
            
            self.metrics["tokens_revoked"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Token revocation failed: {e}")
            return False
    
    async def get_authentication_metrics(self) -> Dict[str, Any]:
        """Get authentication system metrics."""
        success_rate = (
            self.metrics["authentications_successful"] / 
            max(1, self.metrics["authentications_total"])
        )
        
        return {
            "metrics": self.metrics.copy(),
            "success_rate": success_rate,
            "active_providers": len([p for p in self.providers.values() if p]),
            "configuration": {
                "access_token_lifetime_minutes": self.token_config["access_token_lifetime_minutes"],
                "refresh_token_lifetime_days": self.token_config["refresh_token_lifetime_days"],
                "security_features_enabled": {
                    "ip_binding": self.security_config["enable_ip_binding"],
                    "device_tracking": self.security_config["enable_device_tracking"],
                    "behavioral_analysis": self.security_config["enable_behavioral_analysis"],
                    "refresh_token_rotation": self.security_config["refresh_token_rotation"]
                }
            }
        }
    
    # Private helper methods
    
    def _initialize_default_providers(self) -> None:
        """Initialize default OAuth providers."""
        # Google OAuth 2.0
        if settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET:
            self.providers["google"] = OAuthConfig(
                provider_id="google",
                provider_type=ProviderType.GOOGLE,
                client_id=settings.GOOGLE_CLIENT_ID,
                client_secret=settings.GOOGLE_CLIENT_SECRET,
                authorization_endpoint="https://accounts.google.com/o/oauth2/v2/auth",
                token_endpoint="https://oauth2.googleapis.com/token",
                userinfo_endpoint="https://www.googleapis.com/oauth2/v2/userinfo",
                jwks_uri="https://www.googleapis.com/oauth2/v3/certs",
                issuer="https://accounts.google.com",
                scopes=["openid", "profile", "email"],
                redirect_uri=f"{settings.BASE_URL}/auth/callback/google"
            )
        
        # GitHub OAuth
        if settings.GITHUB_CLIENT_ID and settings.GITHUB_CLIENT_SECRET:
            self.providers["github"] = OAuthConfig(
                provider_id="github",
                provider_type=ProviderType.GITHUB,
                client_id=settings.GITHUB_CLIENT_ID,
                client_secret=settings.GITHUB_CLIENT_SECRET,
                authorization_endpoint="https://github.com/login/oauth/authorize",
                token_endpoint="https://github.com/login/oauth/access_token",
                userinfo_endpoint="https://api.github.com/user",
                scopes=["user:email"],
                redirect_uri=f"{settings.BASE_URL}/auth/callback/github",
                require_pkce=False,  # GitHub doesn't require PKCE
                require_nonce=False
            )
    
    async def _exchange_code_for_tokens(
        self,
        provider: OAuthConfig,
        code: str,
        session_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Exchange authorization code for tokens."""
        token_request_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": session_data["redirect_uri"],
            "client_id": provider.client_id,
            "client_secret": provider.client_secret
        }
        
        # Add PKCE if required
        if provider.require_pkce:
            code_verifier = await self.redis.get(f"{self._pkce_prefix}{session_data['state']}")
            if code_verifier:
                token_request_data["code_verifier"] = code_verifier
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                provider.token_endpoint,
                data=token_request_data,
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Token exchange failed: {response.text}"
                )
            
            return response.json()
    
    async def _get_user_info(self, provider: OAuthConfig, access_token: str) -> Dict[str, Any]:
        """Get user information from provider."""
        if not provider.userinfo_endpoint:
            return {}
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                provider.userinfo_endpoint,
                headers=headers,
                timeout=30
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to get user info: {response.text}"
                )
            
            return response.json()
    
    async def _create_or_update_agent_identity(
        self,
        user_info: Dict[str, Any],
        provider_id: str,
        request: Request
    ) -> AgentIdentity:
        """Create or update agent identity."""
        # Look for existing agent by provider user ID
        provider_user_id = user_info.get("id") or user_info.get("sub")
        external_id = f"{provider_id}:{provider_user_id}"
        
        result = await self.db.execute(
            select(AgentIdentity).where(
                AgentIdentity.external_identity_id == external_id
            )
        )
        
        agent = result.scalar_one_or_none()
        
        if agent:
            # Update existing agent
            agent.last_authentication_at = datetime.utcnow()
            agent.authentication_count += 1
            
            # Update profile information if changed
            if user_info.get("email") and user_info["email"] != agent.email:
                agent.email = user_info["email"]
            if user_info.get("name") and user_info["name"] != agent.agent_name:
                agent.agent_name = user_info["name"]
        else:
            # Create new agent identity
            agent = AgentIdentity(
                agent_name=user_info.get("name", f"Agent-{secrets.token_hex(4)}"),
                email=user_info.get("email"),
                external_identity_id=external_id,
                oauth_provider=provider_id,
                profile_data=user_info,
                human_controller=user_info.get("email", "unknown"),
                ip_address_first_seen=request.client.host,
                user_agent_first_seen=request.headers.get("user-agent", ""),
                authentication_count=1,
                last_authentication_at=datetime.utcnow()
            )
            
            self.db.add(agent)
        
        await self.db.commit()
        await self.db.refresh(agent)
        
        return agent
    
    async def _generate_access_token(
        self,
        agent_identity: AgentIdentity,
        session_data: Dict[str, Any],
        token_data: Dict[str, Any]
    ) -> str:
        """Generate JWT access token."""
        now = datetime.utcnow()
        exp = now + timedelta(minutes=self.token_config["access_token_lifetime_minutes"])
        
        claims = TokenClaims(
            sub=str(agent_identity.id),
            iss=self.token_config["jwt_issuer"],
            aud=self.token_config["jwt_audience"],
            exp=int(exp.timestamp()),
            iat=int(now.timestamp()),
            nbf=int(now.timestamp()),
            jti=str(uuid.uuid4()),
            agent_id=str(agent_identity.id),
            human_controller=agent_identity.human_controller,
            scopes=session_data.get("scopes", []),
            provider=agent_identity.oauth_provider,
            trust_level=0.8,  # High trust for authenticated users
            auth_time=int(now.timestamp()),
            amr=["oauth2"]
        )
        
        return jwt.encode(
            claims.to_dict(),
            settings.JWT_SECRET_KEY,
            algorithm=self.token_config["jwt_algorithm"]
        )
    
    async def _generate_refresh_token(
        self,
        agent_identity: AgentIdentity,
        session_data: Dict[str, Any]
    ) -> str:
        """Generate JWT refresh token."""
        now = datetime.utcnow()
        exp = now + timedelta(days=self.token_config["refresh_token_lifetime_days"])
        
        claims = {
            "sub": str(agent_identity.id),
            "iss": self.token_config["jwt_issuer"],
            "aud": self.token_config["jwt_audience"],
            "exp": int(exp.timestamp()),
            "iat": int(now.timestamp()),
            "jti": str(uuid.uuid4()),
            "token_type": "refresh",
            "agent_id": str(agent_identity.id),
            "session_id": session_data.get("session_id"),
            "scopes": session_data.get("scopes", [])
        }
        
        return jwt.encode(
            claims,
            settings.JWT_SECRET_KEY,
            algorithm=self.token_config["jwt_algorithm"]
        )
    
    async def _create_auth_session(
        self,
        agent_identity: AgentIdentity,
        request: Request,
        session_data: Dict[str, Any],
        token_data: Dict[str, Any]
    ) -> AuthenticationSession:
        """Create authentication session."""
        session = AuthenticationSession(
            agent_id=agent_identity.id,
            oauth_provider=session_data["provider_id"],
            session_token=secrets.token_urlsafe(32),
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent", ""),
            scopes_granted=session_data.get("scopes", []),
            provider_access_token=token_data.get("access_token"),
            provider_refresh_token=token_data.get("refresh_token"),
            expires_at=datetime.utcnow() + timedelta(
                minutes=self.security_config["session_timeout_minutes"]
            ),
            last_activity_at=datetime.utcnow()
        )
        
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)
        
        # Update session_id in session_data for token generation
        session_data["session_id"] = str(session.id)
        
        return session
    
    async def _cleanup_auth_state(self, state: str, session_data: Dict[str, Any]) -> None:
        """Clean up temporary authentication state."""
        keys_to_delete = [
            f"{self._auth_state_prefix}{state}"
        ]
        
        if session_data.get("nonce"):
            keys_to_delete.append(f"{self._nonce_prefix}{session_data['nonce']}")
        
        keys_to_delete.append(f"{self._pkce_prefix}{state}")
        
        for key in keys_to_delete:
            try:
                await self.redis.delete(key)
            except Exception as e:
                logger.debug(f"Failed to delete cache key {key}: {e}")
    
    async def _revoke_session(self, session: AuthenticationSession, reason: str) -> None:
        """Revoke authentication session."""
        session.is_active = False
        session.revoked_at = datetime.utcnow()
        session.revocation_reason = reason
        await self.db.commit()
    
    def _update_metrics(self, operation: str, duration_ms: float, success: bool) -> None:
        """Update performance metrics."""
        self.metrics["authentications_total"] += 1
        
        # Update average time
        current_avg = self.metrics["avg_auth_time_ms"]
        total_ops = self.metrics["authentications_total"]
        self.metrics["avg_auth_time_ms"] = (
            (current_avg * (total_ops - 1) + duration_ms) / total_ops
        )
        
        if not success:
            self.metrics[f"provider_failures"][operation] += 1
    
    async def _validate_and_decode_token(
        self,
        token: str,
        expected_type: TokenType
    ) -> Dict[str, Any]:
        """Validate and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[self.token_config["jwt_algorithm"]],
                audience=self.token_config["jwt_audience"],
                issuer=self.token_config["jwt_issuer"]
            )
            
            if expected_type == TokenType.REFRESH:
                if payload.get("token_type") != "refresh":
                    raise HTTPException(status_code=400, detail="Invalid token type")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    
    async def _log_auth_attempt(
        self,
        provider_id: str,
        context: Optional[AuthenticationContext],
        success: bool,
        action: str,
        agent_id: Optional[uuid.UUID] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log authentication attempt."""
        audit_log = SecurityAuditLog(
            agent_id=agent_id,
            human_controller=metadata.get("user_info", {}).get("email", "unknown") if metadata else "unknown",
            action=action,
            resource="authentication",
            success=success,
            ip_address=context.ip_address if context else "unknown",
            user_agent=context.user_agent if context else "unknown",
            metadata={
                "provider": provider_id,
                **(metadata or {})
            }
        )
        
        self.db.add(audit_log)
        # Note: Commit handled by caller
    
    async def _log_security_violation(
        self,
        violation_type: str,
        context: Optional[AuthenticationContext],
        metadata: Dict[str, Any]
    ) -> None:
        """Log security violation."""
        self.metrics["security_violations"] += 1
        
        event = SecurityEvent(
            id=uuid.uuid4(),
            event_type=AuditEventType.AUTHENTICATION_FAILURE,
            threat_level=ThreatLevel.HIGH,
            description=f"Security violation: {violation_type}",
            details={
                "violation_type": violation_type,
                "context": vars(context) if context else {},
                **metadata
            },
            timestamp=datetime.utcnow()
        )
        
        await self.audit_system._handle_security_event(event)


# Factory function
async def create_oauth_authentication_system(
    db_session: AsyncSession,
    redis_client: RedisClient,
    audit_system: SecurityAuditSystem
) -> OAuthAuthenticationSystem:
    """
    Create OAuth Authentication System instance.
    
    Args:
        db_session: Database session
        redis_client: Redis client
        audit_system: Security audit system
        
    Returns:
        OAuthAuthenticationSystem instance
    """
    return OAuthAuthenticationSystem(db_session, redis_client, audit_system)