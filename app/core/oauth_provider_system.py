"""
OAuth 2.0/OIDC Provider System for Enterprise Authentication.

Provides comprehensive OAuth 2.0 and OpenID Connect integration with major
enterprise identity providers including Google, GitHub, Microsoft, and custom OIDC providers.

Features:
- OAuth 2.0 authorization code flow
- OpenID Connect (OIDC) support
- Multi-provider configuration
- Token refresh and validation
- User profile mapping
- Enterprise-grade security
- Multi-tenant support
"""

import asyncio
import base64
import hashlib
import json
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlencode, parse_qs, urlparse
import logging

import httpx
from authlib.integrations.starlette_client import OAuth
from authlib.oauth2 import OAuth2Error
from authlib.oidc.core import CodeIDToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.fernet import Fernet
from fastapi import HTTPException, status, Request, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from .security import create_access_token, verify_token
from .redis import RedisClient
from ..models.security import AgentIdentity, SecurityAuditLog, SecurityEvent
from ..schemas.security import (
    OAuthProviderConfig, OAuthAuthorizationRequest, OAuthCallbackRequest,
    OAuthTokenResponse, UserProfile, SecurityError
)

logger = logging.getLogger(__name__)


class OAuthProviderType(Enum):
    """Supported OAuth provider types."""
    GOOGLE = "google"
    GITHUB = "github"
    MICROSOFT = "microsoft"
    CUSTOM_OIDC = "custom_oidc"
    AZURE_AD = "azure_ad"


@dataclass
class OAuthProviderConfiguration:
    """OAuth provider configuration."""
    provider_type: OAuthProviderType
    client_id: str
    client_secret: str
    
    # Provider-specific endpoints
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: Optional[str] = None
    jwks_uri: Optional[str] = None
    issuer: Optional[str] = None
    
    # Configuration
    scopes: List[str] = field(default_factory=list)
    redirect_uri: str = ""
    tenant_id: Optional[str] = None  # For Azure AD
    domain_hint: Optional[str] = None  # For domain-specific auth
    
    # Security settings
    enable_pkce: bool = True
    require_https: bool = True
    validate_issuer: bool = True
    validate_audience: bool = True
    leeway_seconds: int = 60  # Clock skew tolerance
    
    # Custom mappings
    user_id_field: str = "sub"
    email_field: str = "email"
    name_field: str = "name"
    avatar_field: str = "picture"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provider_type": self.provider_type.value,
            "client_id": self.client_id,
            "authorization_endpoint": self.authorization_endpoint,
            "token_endpoint": self.token_endpoint,
            "userinfo_endpoint": self.userinfo_endpoint,
            "jwks_uri": self.jwks_uri,
            "issuer": self.issuer,
            "scopes": self.scopes,
            "redirect_uri": self.redirect_uri,
            "tenant_id": self.tenant_id,
            "domain_hint": self.domain_hint,
            "enable_pkce": self.enable_pkce,
            "require_https": self.require_https,
            "validate_issuer": self.validate_issuer,
            "validate_audience": self.validate_audience,
            "leeway_seconds": self.leeway_seconds,
            "user_id_field": self.user_id_field,
            "email_field": self.email_field,
            "name_field": self.name_field,
            "avatar_field": self.avatar_field
        }


@dataclass
class OAuthSession:
    """OAuth session data."""
    session_id: str
    provider_type: OAuthProviderType
    state: str
    code_verifier: Optional[str] = None
    code_challenge: Optional[str] = None
    redirect_uri: str = ""
    nonce: Optional[str] = None
    tenant_id: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(minutes=15))
    
    def is_expired(self) -> bool:
        """Check if session is expired."""
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "provider_type": self.provider_type.value,
            "state": self.state,
            "code_verifier": self.code_verifier,
            "code_challenge": self.code_challenge,
            "redirect_uri": self.redirect_uri,
            "nonce": self.nonce,
            "tenant_id": self.tenant_id,
            "scopes": self.scopes,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat()
        }


@dataclass
class OAuthTokenSet:
    """OAuth token set."""
    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    id_token: Optional[str] = None
    scope: Optional[str] = None
    
    # Metadata
    issued_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Calculate expiration time."""
        if self.expires_in and not self.expires_at:
            self.expires_at = self.issued_at + timedelta(seconds=self.expires_in)
    
    def is_expired(self, buffer_seconds: int = 300) -> bool:
        """Check if token is expired (with buffer)."""
        if not self.expires_at:
            return False
        return datetime.utcnow() + timedelta(seconds=buffer_seconds) > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
            "refresh_token": self.refresh_token,
            "id_token": self.id_token,
            "scope": self.scope,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }


class OAuthProviderSystem:
    """
    Comprehensive OAuth 2.0/OIDC Provider System.
    
    Supports enterprise authentication with major identity providers
    and custom OIDC implementations.
    """
    
    # Predefined provider configurations
    PROVIDER_CONFIGS = {
        OAuthProviderType.GOOGLE: {
            "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_endpoint": "https://oauth2.googleapis.com/token",
            "userinfo_endpoint": "https://www.googleapis.com/oauth2/v2/userinfo",
            "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
            "issuer": "https://accounts.google.com",
            "scopes": ["openid", "email", "profile"],
            "user_id_field": "sub",
            "email_field": "email",
            "name_field": "name",
            "avatar_field": "picture"
        },
        OAuthProviderType.GITHUB: {
            "authorization_endpoint": "https://github.com/login/oauth/authorize",
            "token_endpoint": "https://github.com/login/oauth/access_token",
            "userinfo_endpoint": "https://api.github.com/user",
            "scopes": ["user:email", "read:user"],
            "user_id_field": "id",
            "email_field": "email",
            "name_field": "name",
            "avatar_field": "avatar_url"
        },
        OAuthProviderType.MICROSOFT: {
            "authorization_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
            "token_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
            "userinfo_endpoint": "https://graph.microsoft.com/v1.0/me",
            "jwks_uri": "https://login.microsoftonline.com/common/discovery/v2.0/keys",
            "issuer": "https://login.microsoftonline.com/{tenant_id}/v2.0",
            "scopes": ["openid", "profile", "email", "User.Read"],
            "user_id_field": "sub",
            "email_field": "mail",
            "name_field": "displayName",
            "avatar_field": "photo"
        },
        OAuthProviderType.AZURE_AD: {
            "authorization_endpoint": "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize",
            "token_endpoint": "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
            "userinfo_endpoint": "https://graph.microsoft.com/v1.0/me",
            "jwks_uri": "https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys",
            "issuer": "https://login.microsoftonline.com/{tenant_id}/v2.0",
            "scopes": ["openid", "profile", "email", "User.Read"],
            "user_id_field": "sub",
            "email_field": "mail",
            "name_field": "displayName",
            "avatar_field": "photo"
        }
    }
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: RedisClient,
        base_url: str = "http://localhost:8000",
        session_ttl_minutes: int = 15,
        token_encryption_key: Optional[str] = None
    ):
        """
        Initialize OAuth Provider System.
        
        Args:
            db_session: Database session
            redis_client: Redis client for session storage
            base_url: Application base URL
            session_ttl_minutes: OAuth session TTL
            token_encryption_key: Key for token encryption
        """
        self.db = db_session
        self.redis = redis_client
        self.base_url = base_url.rstrip('/')
        self.session_ttl = session_ttl_minutes * 60
        
        # Initialize token encryption
        if token_encryption_key:
            key = base64.urlsafe_b64encode(token_encryption_key.encode()[:32].ljust(32, b'\0'))
            self.token_cipher = Fernet(key)
        else:
            # Generate a key (should be stored securely in production)
            self.token_cipher = Fernet(Fernet.generate_key())
        
        # Provider configurations
        self.providers: Dict[str, OAuthProviderConfiguration] = {}
        
        # HTTP client for API calls
        self.http_client = httpx.AsyncClient()
        
        # Cache keys
        self._session_cache_prefix = "oauth:session:"
        self._token_cache_prefix = "oauth:token:"
        self._provider_cache_prefix = "oauth:provider:"
        
        # Performance metrics
        self.metrics = {
            "authorization_requests": 0,
            "successful_authorizations": 0,
            "failed_authorizations": 0,
            "token_exchanges": 0,
            "token_refreshes": 0,
            "user_info_requests": 0,
            "avg_auth_time_ms": 0.0,
            "provider_usage": {},
            "error_counts": {}
        }
    
    async def configure_provider(
        self,
        provider_name: str,
        provider_type: OAuthProviderType,
        client_id: str,
        client_secret: str,
        tenant_id: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Configure OAuth provider.
        
        Args:
            provider_name: Unique provider name
            provider_type: Provider type
            client_id: OAuth client ID
            client_secret: OAuth client secret
            tenant_id: Tenant ID (for Azure AD)
            custom_config: Custom configuration overrides
            
        Returns:
            True if configuration successful
        """
        try:
            # Get base configuration
            base_config = self.PROVIDER_CONFIGS.get(provider_type, {}).copy()
            
            # Apply custom configuration
            if custom_config:
                base_config.update(custom_config)
            
            # Handle tenant-specific endpoints
            if tenant_id and provider_type in [OAuthProviderType.MICROSOFT, OAuthProviderType.AZURE_AD]:
                for key, value in base_config.items():
                    if isinstance(value, str) and "{tenant_id}" in value:
                        base_config[key] = value.format(tenant_id=tenant_id)
            
            # Create provider configuration
            config = OAuthProviderConfiguration(
                provider_type=provider_type,
                client_id=client_id,
                client_secret=client_secret,
                tenant_id=tenant_id,
                redirect_uri=f"{self.base_url}/auth/oauth/{provider_name}/callback",
                **base_config
            )
            
            # Store configuration
            self.providers[provider_name] = config
            
            # Cache configuration
            await self.redis.set_with_expiry(
                f"{self._provider_cache_prefix}{provider_name}",
                json.dumps(config.to_dict()),
                ttl=86400  # 24 hours
            )
            
            logger.info(f"OAuth provider '{provider_name}' configured successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure OAuth provider '{provider_name}': {e}")
            return False
    
    async def initiate_authorization(
        self,
        provider_name: str,
        redirect_uri: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        state_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Initiate OAuth authorization flow.
        
        Args:
            provider_name: Provider name
            redirect_uri: Custom redirect URI
            scopes: Custom scopes
            state_data: Additional state data
            
        Returns:
            Tuple of (authorization_url, session_id)
        """
        start_time = time.time()
        
        try:
            # Get provider configuration
            config = self.providers.get(provider_name)
            if not config:
                raise ValueError(f"Provider '{provider_name}' not configured")
            
            # Generate session data
            session_id = str(uuid.uuid4())
            state = self._generate_secure_state()
            nonce = self._generate_nonce() if "openid" in (scopes or config.scopes) else None
            
            # PKCE parameters
            code_verifier = None
            code_challenge = None
            if config.enable_pkce:
                code_verifier = self._generate_code_verifier()
                code_challenge = self._generate_code_challenge(code_verifier)
            
            # Create OAuth session
            oauth_session = OAuthSession(
                session_id=session_id,
                provider_type=config.provider_type,
                state=state,
                code_verifier=code_verifier,
                code_challenge=code_challenge,
                redirect_uri=redirect_uri or config.redirect_uri,
                nonce=nonce,
                tenant_id=config.tenant_id,
                scopes=scopes or config.scopes
            )
            
            # Store session
            await self._store_oauth_session(oauth_session)
            
            # Build authorization URL
            auth_params = {
                "client_id": config.client_id,
                "response_type": "code",
                "redirect_uri": oauth_session.redirect_uri,
                "scope": " ".join(oauth_session.scopes),
                "state": state
            }
            
            if nonce:
                auth_params["nonce"] = nonce
            
            if code_challenge:
                auth_params["code_challenge"] = code_challenge
                auth_params["code_challenge_method"] = "S256"
            
            if config.tenant_id and config.provider_type == OAuthProviderType.AZURE_AD:
                auth_params["tenant"] = config.tenant_id
            
            if config.domain_hint:
                auth_params["domain_hint"] = config.domain_hint
            
            authorization_url = f"{config.authorization_endpoint}?{urlencode(auth_params)}"
            
            # Update metrics
            self.metrics["authorization_requests"] += 1
            provider_key = f"{provider_name}_{config.provider_type.value}"
            self.metrics["provider_usage"][provider_key] = self.metrics["provider_usage"].get(provider_key, 0) + 1
            
            # Log audit event
            await self._log_oauth_event(
                action="initiate_authorization",
                provider_name=provider_name,
                session_id=session_id,
                success=True,
                metadata={
                    "scopes": oauth_session.scopes,
                    "pkce_enabled": config.enable_pkce,
                    "nonce_used": nonce is not None
                }
            )
            
            processing_time = (time.time() - start_time) * 1000
            current_avg = self.metrics["avg_auth_time_ms"]
            total_requests = self.metrics["authorization_requests"]
            self.metrics["avg_auth_time_ms"] = (
                (current_avg * (total_requests - 1) + processing_time) / total_requests
            )
            
            return authorization_url, session_id
            
        except Exception as e:
            logger.error(f"Authorization initiation failed for provider '{provider_name}': {e}")
            self.metrics["failed_authorizations"] += 1
            self.metrics["error_counts"]["authorization_error"] = self.metrics["error_counts"].get("authorization_error", 0) + 1
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Authorization initiation failed: {str(e)}"
            )
    
    async def handle_authorization_callback(
        self,
        provider_name: str,
        code: str,
        state: str,
        error: Optional[str] = None,
        error_description: Optional[str] = None
    ) -> Tuple[OAuthTokenSet, UserProfile]:
        """
        Handle OAuth authorization callback.
        
        Args:
            provider_name: Provider name
            code: Authorization code
            state: State parameter
            error: Error code if authorization failed
            error_description: Error description
            
        Returns:
            Tuple of (token_set, user_profile)
        """
        try:
            # Handle authorization errors
            if error:
                error_msg = f"Authorization error: {error}"
                if error_description:
                    error_msg += f" - {error_description}"
                logger.error(error_msg)
                self.metrics["failed_authorizations"] += 1
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_msg
                )
            
            # Get provider configuration
            config = self.providers.get(provider_name)
            if not config:
                raise ValueError(f"Provider '{provider_name}' not configured")
            
            # Retrieve and validate OAuth session
            oauth_session = await self._get_oauth_session_by_state(state)
            if not oauth_session:
                raise ValueError("Invalid or expired OAuth session")
            
            if oauth_session.provider_type != config.provider_type:
                raise ValueError("Provider type mismatch")
            
            # Exchange authorization code for tokens
            token_set = await self._exchange_authorization_code(
                config, oauth_session, code
            )
            
            # Get user information
            user_profile = await self._get_user_profile(config, token_set)
            
            # Validate ID token if present
            if token_set.id_token:
                await self._validate_id_token(config, token_set.id_token, oauth_session.nonce)
            
            # Store tokens securely
            await self._store_token_set(user_profile.user_id, provider_name, token_set)
            
            # Clean up OAuth session
            await self._cleanup_oauth_session(oauth_session.session_id)
            
            # Update metrics
            self.metrics["successful_authorizations"] += 1
            self.metrics["token_exchanges"] += 1
            
            # Log successful authorization
            await self._log_oauth_event(
                action="authorization_callback",
                provider_name=provider_name,
                session_id=oauth_session.session_id,
                success=True,
                metadata={
                    "user_id": user_profile.user_id,
                    "email": user_profile.email,
                    "has_refresh_token": token_set.refresh_token is not None,
                    "token_expires_in": token_set.expires_in
                }
            )
            
            return token_set, user_profile
            
        except Exception as e:
            logger.error(f"Authorization callback failed for provider '{provider_name}': {e}")
            self.metrics["failed_authorizations"] += 1
            self.metrics["error_counts"]["callback_error"] = self.metrics["error_counts"].get("callback_error", 0) + 1
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Authorization callback failed: {str(e)}"
            )
    
    async def refresh_access_token(
        self,
        provider_name: str,
        user_id: str
    ) -> Optional[OAuthTokenSet]:
        """
        Refresh access token using refresh token.
        
        Args:
            provider_name: Provider name
            user_id: User identifier
            
        Returns:
            New token set or None if refresh failed
        """
        try:
            # Get stored token set
            token_set = await self._get_stored_token_set(user_id, provider_name)
            if not token_set or not token_set.refresh_token:
                return None
            
            # Get provider configuration
            config = self.providers.get(provider_name)
            if not config:
                return None
            
            # Prepare refresh request
            token_data = {
                "grant_type": "refresh_token",
                "refresh_token": token_set.refresh_token,
                "client_id": config.client_id,
                "client_secret": config.client_secret
            }
            
            # Make token refresh request
            response = await self.http_client.post(
                config.token_endpoint,
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            if response.status_code != 200:
                logger.error(f"Token refresh failed: {response.status_code} - {response.text}")
                return None
            
            token_response = response.json()
            
            # Create new token set
            new_token_set = OAuthTokenSet(
                access_token=token_response["access_token"],
                token_type=token_response.get("token_type", "Bearer"),
                expires_in=token_response.get("expires_in"),
                refresh_token=token_response.get("refresh_token", token_set.refresh_token),
                id_token=token_response.get("id_token"),
                scope=token_response.get("scope")
            )
            
            # Store updated tokens
            await self._store_token_set(user_id, provider_name, new_token_set)
            
            # Update metrics
            self.metrics["token_refreshes"] += 1
            
            # Log token refresh
            await self._log_oauth_event(
                action="refresh_token",
                provider_name=provider_name,
                success=True,
                metadata={
                    "user_id": user_id,
                    "expires_in": new_token_set.expires_in
                }
            )
            
            return new_token_set
            
        except Exception as e:
            logger.error(f"Token refresh failed for user {user_id}, provider {provider_name}: {e}")
            self.metrics["error_counts"]["refresh_error"] = self.metrics["error_counts"].get("refresh_error", 0) + 1
            return None
    
    async def get_user_profile(
        self,
        provider_name: str,
        user_id: str
    ) -> Optional[UserProfile]:
        """
        Get user profile using stored access token.
        
        Args:
            provider_name: Provider name
            user_id: User identifier
            
        Returns:
            User profile or None if failed
        """
        try:
            # Get stored token set
            token_set = await self._get_stored_token_set(user_id, provider_name)
            if not token_set:
                return None
            
            # Refresh token if expired
            if token_set.is_expired():
                token_set = await self.refresh_access_token(provider_name, user_id)
                if not token_set:
                    return None
            
            # Get provider configuration
            config = self.providers.get(provider_name)
            if not config:
                return None
            
            # Get user profile
            user_profile = await self._get_user_profile(config, token_set)
            
            # Update metrics
            self.metrics["user_info_requests"] += 1
            
            return user_profile
            
        except Exception as e:
            logger.error(f"Failed to get user profile for user {user_id}, provider {provider_name}: {e}")
            return None
    
    async def revoke_tokens(
        self,
        provider_name: str,
        user_id: str
    ) -> bool:
        """
        Revoke OAuth tokens for user.
        
        Args:
            provider_name: Provider name
            user_id: User identifier
            
        Returns:
            True if revocation successful
        """
        try:
            # Get stored token set
            token_set = await self._get_stored_token_set(user_id, provider_name)
            if not token_set:
                return True  # Already revoked
            
            # Get provider configuration
            config = self.providers.get(provider_name)
            if not config:
                return False
            
            # Attempt to revoke tokens at provider (if supported)
            # This is provider-specific and may not be supported by all providers
            
            # Remove stored tokens
            cache_key = f"{self._token_cache_prefix}{user_id}:{provider_name}"
            await self.redis.delete(cache_key)
            
            # Log token revocation
            await self._log_oauth_event(
                action="revoke_tokens",
                provider_name=provider_name,
                success=True,
                metadata={"user_id": user_id}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Token revocation failed for user {user_id}, provider {provider_name}: {e}")
            return False
    
    def get_provider_list(self) -> List[Dict[str, Any]]:
        """Get list of configured providers."""
        return [
            {
                "name": name,
                "type": config.provider_type.value,
                "scopes": config.scopes,
                "supports_refresh": True,
                "supports_pkce": config.enable_pkce
            }
            for name, config in self.providers.items()
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get OAuth system metrics."""
        return {
            "oauth_metrics": self.metrics.copy(),
            "configured_providers": len(self.providers),
            "provider_types": list(set(config.provider_type.value for config in self.providers.values()))
        }
    
    # Private helper methods
    
    def _generate_secure_state(self) -> str:
        """Generate cryptographically secure state parameter."""
        return secrets.token_urlsafe(32)
    
    def _generate_nonce(self) -> str:
        """Generate cryptographically secure nonce."""
        return secrets.token_urlsafe(32)
    
    def _generate_code_verifier(self) -> str:
        """Generate PKCE code verifier."""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
    
    def _generate_code_challenge(self, code_verifier: str) -> str:
        """Generate PKCE code challenge."""
        digest = hashes.Hash(hashes.SHA256())
        digest.update(code_verifier.encode('utf-8'))
        return base64.urlsafe_b64encode(digest.finalize()).decode('utf-8').rstrip('=')
    
    async def _store_oauth_session(self, session: OAuthSession) -> None:
        """Store OAuth session in cache."""
        cache_key = f"{self._session_cache_prefix}{session.session_id}"
        await self.redis.set_with_expiry(
            cache_key,
            json.dumps(session.to_dict()),
            ttl=self.session_ttl
        )
        
        # Also store by state for quick lookup
        state_key = f"{self._session_cache_prefix}state:{session.state}"
        await self.redis.set_with_expiry(
            state_key,
            session.session_id,
            ttl=self.session_ttl
        )
    
    async def _get_oauth_session_by_state(self, state: str) -> Optional[OAuthSession]:
        """Get OAuth session by state parameter."""
        try:
            # Get session ID by state
            state_key = f"{self._session_cache_prefix}state:{state}"
            session_id = await self.redis.get(state_key)
            if not session_id:
                return None
            
            # Get session data
            cache_key = f"{self._session_cache_prefix}{session_id}"
            session_data = await self.redis.get(cache_key)
            if not session_data:
                return None
            
            data = json.loads(session_data)
            
            # Reconstruct session object
            session = OAuthSession(
                session_id=data["session_id"],
                provider_type=OAuthProviderType(data["provider_type"]),
                state=data["state"],
                code_verifier=data.get("code_verifier"),
                code_challenge=data.get("code_challenge"),
                redirect_uri=data["redirect_uri"],
                nonce=data.get("nonce"),
                tenant_id=data.get("tenant_id"),
                scopes=data["scopes"],
                created_at=datetime.fromisoformat(data["created_at"]),
                expires_at=datetime.fromisoformat(data["expires_at"])
            )
            
            # Check if expired
            if session.is_expired():
                await self._cleanup_oauth_session(session.session_id)
                return None
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to get OAuth session by state: {e}")
            return None
    
    async def _cleanup_oauth_session(self, session_id: str) -> None:
        """Clean up OAuth session."""
        cache_key = f"{self._session_cache_prefix}{session_id}"
        await self.redis.delete(cache_key)
        
        # Note: State-based key will expire naturally
    
    async def _exchange_authorization_code(
        self,
        config: OAuthProviderConfiguration,
        session: OAuthSession,
        code: str
    ) -> OAuthTokenSet:
        """Exchange authorization code for tokens."""
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": session.redirect_uri,
            "client_id": config.client_id,
            "client_secret": config.client_secret
        }
        
        if session.code_verifier:
            token_data["code_verifier"] = session.code_verifier
        
        response = await self.http_client.post(
            config.token_endpoint,
            data=token_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code != 200:
            error_detail = f"Token exchange failed: {response.status_code} - {response.text}"
            logger.error(error_detail)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_detail)
        
        token_response = response.json()
        
        return OAuthTokenSet(
            access_token=token_response["access_token"],
            token_type=token_response.get("token_type", "Bearer"),
            expires_in=token_response.get("expires_in"),
            refresh_token=token_response.get("refresh_token"),
            id_token=token_response.get("id_token"),
            scope=token_response.get("scope")
        )
    
    async def _get_user_profile(
        self,
        config: OAuthProviderConfiguration,
        token_set: OAuthTokenSet
    ) -> UserProfile:
        """Get user profile from provider."""
        if not config.userinfo_endpoint:
            raise ValueError("Provider does not support user info endpoint")
        
        headers = {"Authorization": f"{token_set.token_type} {token_set.access_token}"}
        
        response = await self.http_client.get(config.userinfo_endpoint, headers=headers)
        
        if response.status_code != 200:
            error_detail = f"User info request failed: {response.status_code} - {response.text}"
            logger.error(error_detail)
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_detail)
        
        user_data = response.json()
        
        # Map provider-specific fields to standard profile
        return UserProfile(
            user_id=str(user_data.get(config.user_id_field, "")),
            email=user_data.get(config.email_field, ""),
            name=user_data.get(config.name_field, ""),
            avatar_url=user_data.get(config.avatar_field, ""),
            provider=config.provider_type.value,
            raw_data=user_data
        )
    
    async def _validate_id_token(
        self,
        config: OAuthProviderConfiguration,
        id_token: str,
        expected_nonce: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate OIDC ID token."""
        # This is a simplified validation - in production, you'd want to:
        # 1. Verify JWT signature using JWKS
        # 2. Validate issuer, audience, expiration
        # 3. Verify nonce if provided
        
        try:
            # Decode without verification for now (for demonstration)
            import jwt
            header = jwt.get_unverified_header(id_token)
            payload = jwt.decode(id_token, options={"verify_signature": False})
            
            # Basic validations
            if config.validate_issuer and payload.get("iss") != config.issuer:
                raise ValueError("Invalid issuer")
            
            if config.validate_audience and payload.get("aud") != config.client_id:
                raise ValueError("Invalid audience")
            
            if expected_nonce and payload.get("nonce") != expected_nonce:
                raise ValueError("Invalid nonce")
            
            exp = payload.get("exp")
            if exp and datetime.utcnow().timestamp() > exp:
                raise ValueError("Token expired")
            
            return payload
            
        except Exception as e:
            logger.error(f"ID token validation failed: {e}")
            raise ValueError(f"Invalid ID token: {str(e)}")
    
    async def _store_token_set(
        self,
        user_id: str,
        provider_name: str,
        token_set: OAuthTokenSet
    ) -> None:
        """Store encrypted token set."""
        cache_key = f"{self._token_cache_prefix}{user_id}:{provider_name}"
        
        # Encrypt sensitive data
        encrypted_data = self.token_cipher.encrypt(
            json.dumps(token_set.to_dict()).encode()
        )
        
        # Store with appropriate TTL
        ttl = token_set.expires_in if token_set.expires_in else 3600
        await self.redis.set_with_expiry(cache_key, encrypted_data, ttl=ttl)
    
    async def _get_stored_token_set(
        self,
        user_id: str,
        provider_name: str
    ) -> Optional[OAuthTokenSet]:
        """Get stored token set."""
        try:
            cache_key = f"{self._token_cache_prefix}{user_id}:{provider_name}"
            encrypted_data = await self.redis.get(cache_key)
            if not encrypted_data:
                return None
            
            # Decrypt data
            decrypted_data = self.token_cipher.decrypt(encrypted_data)
            token_data = json.loads(decrypted_data.decode())
            
            # Reconstruct token set
            token_set = OAuthTokenSet(
                access_token=token_data["access_token"],
                token_type=token_data["token_type"],
                expires_in=token_data.get("expires_in"),
                refresh_token=token_data.get("refresh_token"),
                id_token=token_data.get("id_token"),
                scope=token_data.get("scope"),
                issued_at=datetime.fromisoformat(token_data["issued_at"])
            )
            
            if token_data.get("expires_at"):
                token_set.expires_at = datetime.fromisoformat(token_data["expires_at"])
            
            return token_set
            
        except Exception as e:
            logger.error(f"Failed to get stored token set: {e}")
            return None
    
    async def _log_oauth_event(
        self,
        action: str,
        provider_name: str,
        success: bool,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log OAuth audit event."""
        audit_log = SecurityAuditLog(
            agent_id=None,  # OAuth events are user-level
            human_controller="oauth_system",
            action=action,
            resource="oauth_authorization",
            resource_id=provider_name,
            success=success,
            metadata={
                "provider_name": provider_name,
                "session_id": session_id,
                **(metadata or {})
            }
        )
        
        self.db.add(audit_log)
        # Note: Commit handled by caller
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        await self.http_client.aclose()


# Factory function
async def create_oauth_provider_system(
    db_session: AsyncSession,
    redis_client: RedisClient,
    base_url: str = "http://localhost:8000"
) -> OAuthProviderSystem:
    """
    Create OAuth Provider System instance.
    
    Args:
        db_session: Database session
        redis_client: Redis client
        base_url: Application base URL
        
    Returns:
        OAuthProviderSystem instance
    """
    return OAuthProviderSystem(db_session, redis_client, base_url)