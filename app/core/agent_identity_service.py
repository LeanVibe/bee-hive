"""
Agent Identity Service for OAuth 2.0/OIDC Authentication.

Implements enterprise-grade agent authentication with JWT tokens,
human accountability, rate limiting, and secure credential management.
"""

import uuid
import hashlib
import hmac
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption
import bcrypt
import logging

from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.security import (
    AgentIdentity, AgentToken, SecurityAuditLog, SecurityEvent,
    AgentStatus, SecurityEventSeverity
)
from ..schemas.security import (
    AgentTokenRequest, TokenRefreshRequest, TokenResponse,
    JWTTokenPayload, SecurityError
)
from .redis import RedisClient

logger = logging.getLogger(__name__)


class TokenValidationError(Exception):
    """Raised when token validation fails."""
    pass


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""
    pass


class AgentIdentityService:
    """
    Agent Identity Service for OAuth 2.0/OIDC authentication.
    
    Provides:
    - Agent registration and credential management
    - JWT token generation and validation
    - Rate limiting and security controls
    - Human accountability tracking
    - Comprehensive audit logging
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: RedisClient,
        jwt_secret_key: str,
        jwt_algorithm: str = "RS256",
        issuer: str = "leanvibe-agent-hive"
    ):
        """
        Initialize Agent Identity Service.
        
        Args:
            db_session: Database session
            redis_client: Redis client for rate limiting and token storage
            jwt_secret_key: JWT signing key
            jwt_algorithm: JWT signing algorithm
            issuer: JWT issuer identifier
        """
        self.db = db_session
        self.redis = redis_client
        self.jwt_secret_key = jwt_secret_key
        self.jwt_algorithm = jwt_algorithm
        self.issuer = issuer
        
        # Security configuration
        self.config = {
            "max_token_lifetime_hours": 24,
            "default_token_lifetime_seconds": 3600,  # 1 hour
            "refresh_token_lifetime_seconds": 604800,  # 7 days
            "rate_limit_window_seconds": 60,
            "max_failed_attempts": 5,
            "token_cleanup_interval_hours": 24,
            "require_human_controller": True,
            "enable_ip_binding": True,
            "log_all_requests": True
        }
        
        # Rate limiting keys
        self._rate_limit_key_prefix = "auth:rate_limit:"
        self._failed_attempts_key_prefix = "auth:failed:"
        self._token_blacklist_key_prefix = "auth:blacklist:"
    
    async def register_agent(
        self,
        agent_name: str,
        human_controller: str,
        scopes: List[str],
        created_by: str,
        rate_limit_per_minute: int = 10,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentIdentity:
        """
        Register a new agent identity.
        
        Args:
            agent_name: Unique agent name
            human_controller: Human responsible for the agent
            scopes: List of allowed scopes
            created_by: Who created the agent
            rate_limit_per_minute: Rate limit for this agent
            metadata: Additional metadata
            
        Returns:
            Created AgentIdentity
            
        Raises:
            ValueError: If agent name already exists
        """
        try:
            # Check if agent name already exists
            existing_agent = await self.db.execute(
                select(AgentIdentity).where(AgentIdentity.agent_name == agent_name)
            )
            if existing_agent.scalar_one_or_none():
                raise ValueError(f"Agent name '{agent_name}' already exists")
            
            # Generate OAuth credentials
            oauth_client_id = self._generate_client_id()
            oauth_client_secret = self._generate_client_secret()
            client_secret_hash = self._hash_client_secret(oauth_client_secret)
            
            # Generate RSA key pair for JWT signing
            private_key, public_key = self._generate_rsa_key_pair()
            private_key_encrypted = self._encrypt_private_key(private_key)
            
            # Create agent identity
            agent_identity = AgentIdentity(
                agent_name=agent_name,
                human_controller=human_controller,
                oauth_client_id=oauth_client_id,
                oauth_client_secret_hash=client_secret_hash,
                public_key=public_key,
                private_key_encrypted=private_key_encrypted,
                scopes=scopes,
                rate_limit_per_minute=rate_limit_per_minute,
                status=AgentStatus.active.value,
                metadata=metadata or {},
                created_by=created_by
            )
            
            self.db.add(agent_identity)
            await self.db.commit()
            await self.db.refresh(agent_identity)
            
            # Log registration
            await self._log_audit_event(
                agent_id=agent_identity.id,
                human_controller=human_controller,
                action="agent_registration",
                resource="agent_identity",
                resource_id=str(agent_identity.id),
                success=True,
                metadata={
                    "agent_name": agent_name,
                    "scopes": scopes,
                    "created_by": created_by
                }
            )
            
            logger.info(f"Agent registered: {agent_name} by {created_by}")
            
            # Return client secret only once (for initial setup)
            agent_identity.oauth_client_secret_plaintext = oauth_client_secret
            
            return agent_identity
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Agent registration failed: {e}")
            raise
    
    async def authenticate_agent(
        self,
        request: AgentTokenRequest,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> TokenResponse:
        """
        Authenticate agent and issue tokens.
        
        Args:
            request: Token request
            client_ip: Client IP address
            user_agent: Client user agent
            
        Returns:
            TokenResponse with access and refresh tokens
            
        Raises:
            SecurityError: If authentication fails
            RateLimitExceededError: If rate limit exceeded
        """
        correlation_id = str(uuid.uuid4())
        
        try:
            # Check rate limiting
            await self._check_rate_limit(request.agent_id, client_ip)
            
            # Get agent identity
            agent_identity = await self._get_agent_identity(request.agent_id)
            if not agent_identity or not agent_identity.is_active():
                await self._log_failed_attempt(request.agent_id, "invalid_agent", correlation_id)
                raise SecurityError(
                    error="invalid_client",
                    error_description="Invalid agent or agent is not active",
                    correlation_id=correlation_id
                )
            
            # Verify human controller matches
            if (self.config["require_human_controller"] and 
                agent_identity.human_controller != request.human_controller):
                await self._log_failed_attempt(request.agent_id, "human_controller_mismatch", correlation_id)
                raise SecurityError(
                    error="access_denied",
                    error_description="Human controller mismatch",
                    correlation_id=correlation_id
                )
            
            # Validate requested scopes
            valid_scopes = self._validate_scopes(request.requested_scopes, agent_identity.scopes)
            
            # Check concurrent token limit
            active_token_count = await self._count_active_tokens(agent_identity.id)
            if active_token_count >= agent_identity.max_concurrent_tokens:
                await self._cleanup_expired_tokens(agent_identity.id)
                active_token_count = await self._count_active_tokens(agent_identity.id)
                
                if active_token_count >= agent_identity.max_concurrent_tokens:
                    raise SecurityError(
                        error="too_many_tokens",
                        error_description="Maximum concurrent tokens exceeded",
                        correlation_id=correlation_id
                    )
            
            # Generate tokens
            access_token_data = await self._generate_access_token(
                agent_identity, valid_scopes, client_ip, user_agent
            )
            refresh_token_data = await self._generate_refresh_token(
                agent_identity, valid_scopes, client_ip, user_agent
            )
            
            # Update agent last used
            agent_identity.last_used = datetime.utcnow()
            await self.db.commit()
            
            # Log successful authentication
            await self._log_audit_event(
                agent_id=agent_identity.id,
                human_controller=request.human_controller,
                action="authenticate",
                resource="token",
                success=True,
                request_data={
                    "scopes": valid_scopes,
                    "token_type": "access+refresh"
                },
                ip_address=client_ip,
                user_agent=user_agent,
                correlation_id=correlation_id
            )
            
            return TokenResponse(
                access_token=access_token_data["token"],
                refresh_token=refresh_token_data["token"],
                token_type="Bearer",
                expires_in=agent_identity.token_expires_in_seconds,
                scope=valid_scopes
            )
            
        except (SecurityError, RateLimitExceededError):
            raise
        except Exception as e:
            logger.error(f"Authentication error for {request.agent_id}: {e}")
            await self._log_audit_event(
                agent_id=None,
                human_controller=request.human_controller,
                action="authenticate",
                resource="token",
                success=False,
                error_message=str(e),
                correlation_id=correlation_id
            )
            raise SecurityError(
                error="server_error",
                error_description="Authentication service error",
                correlation_id=correlation_id
            )
    
    async def refresh_token(
        self,
        request: TokenRefreshRequest,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> TokenResponse:
        """
        Refresh access token using refresh token.
        
        Args:
            request: Token refresh request
            client_ip: Client IP address
            user_agent: Client user agent
            
        Returns:
            New TokenResponse
            
        Raises:
            SecurityError: If refresh fails
        """
        correlation_id = str(uuid.uuid4())
        
        try:
            # Validate refresh token
            token_data = await self._validate_refresh_token(request.refresh_token)
            agent_identity = await self._get_agent_identity(token_data["agent_id"])
            
            if not agent_identity or not agent_identity.is_active():
                raise SecurityError(
                    error="invalid_grant",
                    error_description="Invalid refresh token or agent inactive",
                    correlation_id=correlation_id
                )
            
            # Validate scopes (use existing or requested)
            requested_scopes = request.requested_scopes or token_data["scopes"]
            valid_scopes = self._validate_scopes(requested_scopes, agent_identity.scopes)
            
            # Generate new access token
            access_token_data = await self._generate_access_token(
                agent_identity, valid_scopes, client_ip, user_agent
            )
            
            # Update refresh token usage
            await self._record_token_usage(token_data["jti"])
            
            # Log token refresh
            await self._log_audit_event(
                agent_id=agent_identity.id,
                human_controller=agent_identity.human_controller,
                action="refresh_token",
                resource="token",
                success=True,
                request_data={"scopes": valid_scopes},
                ip_address=client_ip,
                user_agent=user_agent,
                correlation_id=correlation_id
            )
            
            return TokenResponse(
                access_token=access_token_data["token"],
                refresh_token=request.refresh_token,  # Keep existing refresh token
                token_type="Bearer",
                expires_in=agent_identity.token_expires_in_seconds,
                scope=valid_scopes
            )
            
        except SecurityError:
            raise
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            raise SecurityError(
                error="server_error",
                error_description="Token refresh service error",
                correlation_id=correlation_id
            )
    
    async def validate_token(
        self,
        token: str,
        required_scopes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate JWT access token.
        
        Args:
            token: JWT token to validate
            required_scopes: Required scopes for access
            
        Returns:
            Decoded token payload
            
        Raises:
            TokenValidationError: If token is invalid
        """
        try:
            # Check token blacklist
            if await self._is_token_blacklisted(token):
                raise TokenValidationError("Token is revoked")
            
            # Decode and validate JWT
            payload = jwt.decode(
                token,
                self.jwt_secret_key,
                algorithms=[self.jwt_algorithm],
                issuer=self.issuer,
                options={"require": ["exp", "iat", "sub", "jti"]}
            )
            
            # Validate token in database
            token_hash = self._hash_token(token)
            db_token = await self.db.execute(
                select(AgentToken).where(
                    and_(
                        AgentToken.token_hash == token_hash,
                        AgentToken.token_type == "access",
                        AgentToken.is_revoked == False,
                        AgentToken.expires_at > datetime.utcnow()
                    )
                )
            )
            
            token_record = db_token.scalar_one_or_none()
            if not token_record:
                raise TokenValidationError("Token not found or expired")
            
            # Check agent status
            agent_identity = await self._get_agent_identity(payload["sub"])
            if not agent_identity or not agent_identity.is_active():
                raise TokenValidationError("Agent inactive")
            
            # Validate required scopes
            if required_scopes:
                token_scopes = payload.get("scope", [])
                if not all(scope in token_scopes for scope in required_scopes):
                    raise TokenValidationError("Insufficient scopes")
            
            # Record token usage
            token_record.record_usage()
            await self.db.commit()
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise TokenValidationError("Token expired")
        except jwt.InvalidTokenError as e:
            raise TokenValidationError(f"Invalid token: {e}")
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            raise TokenValidationError("Token validation failed")
    
    async def revoke_token(
        self,
        token: str,
        revoked_by: str,
        reason: str = "manual_revocation"
    ) -> bool:
        """
        Revoke a token.
        
        Args:
            token: Token to revoke
            revoked_by: Who revoked the token
            reason: Revocation reason
            
        Returns:
            True if revoked successfully
        """
        try:
            token_hash = self._hash_token(token)
            
            # Find and revoke token
            result = await self.db.execute(
                select(AgentToken).where(AgentToken.token_hash == token_hash)
            )
            token_record = result.scalar_one_or_none()
            
            if token_record:
                token_record.revoke(reason)
                
                # Add to blacklist
                await self._blacklist_token(token)
                
                # Log revocation
                await self._log_audit_event(
                    agent_id=token_record.agent_id,
                    human_controller=revoked_by,
                    action="revoke_token",
                    resource="token",
                    resource_id=str(token_record.id),
                    success=True,
                    metadata={"reason": reason, "revoked_by": revoked_by}
                )
                
                await self.db.commit()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Token revocation error: {e}")
            await self.db.rollback()
            return False
    
    async def cleanup_expired_tokens(self) -> int:
        """
        Clean up expired tokens and audit logs.
        
        Returns:
            Number of tokens cleaned up
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            
            # Delete expired tokens
            result = await self.db.execute(
                select(AgentToken).where(
                    or_(
                        AgentToken.expires_at < cutoff_time,
                        and_(
                            AgentToken.is_revoked == True,
                            AgentToken.revoked_at < cutoff_time
                        )
                    )
                )
            )
            
            expired_tokens = result.scalars().all()
            
            for token in expired_tokens:
                await self.db.delete(token)
            
            await self.db.commit()
            
            logger.info(f"Cleaned up {len(expired_tokens)} expired tokens")
            return len(expired_tokens)
            
        except Exception as e:
            logger.error(f"Token cleanup error: {e}")
            await self.db.rollback()
            return 0
    
    # Private helper methods
    
    async def _get_agent_identity(self, agent_id: str) -> Optional[AgentIdentity]:
        """Get agent identity by ID or name."""
        try:
            # Try by UUID first
            agent_uuid = uuid.UUID(agent_id)
            result = await self.db.execute(
                select(AgentIdentity).where(AgentIdentity.id == agent_uuid)
            )
        except ValueError:
            # Try by name
            result = await self.db.execute(
                select(AgentIdentity).where(AgentIdentity.agent_name == agent_id)
            )
        
        return result.scalar_one_or_none()
    
    async def _check_rate_limit(self, agent_id: str, client_ip: Optional[str] = None) -> None:
        """Check if agent or IP is rate limited."""
        window_start = datetime.utcnow() - timedelta(seconds=self.config["rate_limit_window_seconds"])
        
        # Get agent for rate limit
        agent = await self._get_agent_identity(agent_id)
        if not agent:
            return
        
        # Check agent-specific rate limit
        rate_limit_key = f"{self._rate_limit_key_prefix}agent:{agent_id}"
        current_requests = await self.redis.get_count_in_window(
            rate_limit_key, self.config["rate_limit_window_seconds"]
        )
        
        if current_requests >= agent.rate_limit_per_minute:
            raise RateLimitExceededError(f"Rate limit exceeded for agent {agent_id}")
        
        # Increment counter
        await self.redis.increment_counter(rate_limit_key, self.config["rate_limit_window_seconds"])
    
    async def _log_failed_attempt(self, agent_id: str, reason: str, correlation_id: str) -> None:
        """Log failed authentication attempt."""
        failed_key = f"{self._failed_attempts_key_prefix}{agent_id}"
        failed_count = await self.redis.increment_counter(failed_key, 3600)  # 1 hour window
        
        # Create security event for repeated failures
        if failed_count >= self.config["max_failed_attempts"]:
            await self._create_security_event(
                event_type="repeated_auth_failures",
                severity=SecurityEventSeverity.HIGH,
                agent_id=agent_id,
                description=f"Agent {agent_id} has {failed_count} failed authentication attempts",
                details={
                    "failed_count": failed_count,
                    "reason": reason,
                    "correlation_id": correlation_id
                }
            )
    
    def _generate_client_id(self) -> str:
        """Generate OAuth client ID."""
        return f"agent_{secrets.token_hex(16)}"
    
    def _generate_client_secret(self) -> str:
        """Generate OAuth client secret."""
        return secrets.token_urlsafe(32)
    
    def _hash_client_secret(self, secret: str) -> str:
        """Hash client secret using bcrypt."""
        return bcrypt.hashpw(secret.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def _verify_client_secret(self, secret: str, hashed: str) -> bool:
        """Verify client secret against hash."""
        return bcrypt.checkpw(secret.encode('utf-8'), hashed.encode('utf-8'))
    
    def _generate_rsa_key_pair(self) -> Tuple[str, str]:
        """Generate RSA key pair for JWT signing."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        private_pem = private_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption()
        ).decode('utf-8')
        
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        
        return private_pem, public_pem
    
    def _encrypt_private_key(self, private_key: str) -> str:
        """Encrypt private key for storage (simplified - use proper encryption in production)."""
        # In production, use proper key encryption with KMS or HSM
        return private_key  # Placeholder
    
    def _validate_scopes(self, requested: List[str], allowed: List[str]) -> List[str]:
        """Validate and filter requested scopes."""
        if not requested:
            return []
        
        # Return intersection of requested and allowed scopes
        return [scope for scope in requested if scope in allowed]
    
    async def _count_active_tokens(self, agent_id: uuid.UUID) -> int:
        """Count active tokens for agent."""
        result = await self.db.execute(
            select(func.count(AgentToken.id)).where(
                and_(
                    AgentToken.agent_id == agent_id,
                    AgentToken.is_revoked == False,
                    AgentToken.expires_at > datetime.utcnow()
                )
            )
        )
        return result.scalar() or 0
    
    async def _cleanup_expired_tokens(self, agent_id: uuid.UUID) -> None:
        """Clean up expired tokens for specific agent."""
        await self.db.execute(
            select(AgentToken).where(
                and_(
                    AgentToken.agent_id == agent_id,
                    AgentToken.expires_at <= datetime.utcnow()
                )
            ).delete()
        )
        await self.db.commit()
    
    async def _generate_access_token(
        self,
        agent_identity: AgentIdentity,
        scopes: List[str],
        client_ip: Optional[str],
        user_agent: Optional[str]
    ) -> Dict[str, Any]:
        """Generate JWT access token."""
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=agent_identity.token_expires_in_seconds)
        jti = str(uuid.uuid4())
        
        payload = {
            "iss": self.issuer,
            "sub": str(agent_identity.id),
            "aud": "leanvibe-api",
            "exp": int(expires_at.timestamp()),
            "iat": int(now.timestamp()),
            "jti": jti,
            "scope": scopes,
            "human_controller": agent_identity.human_controller,
            "agent_name": agent_identity.agent_name,
            "role_ids": []  # Will be populated by authorization service
        }
        
        token = jwt.encode(payload, self.jwt_secret_key, algorithm=self.jwt_algorithm)
        token_hash = self._hash_token(token)
        
        # Store token in database
        db_token = AgentToken(
            agent_id=agent_identity.id,
            token_type="access",
            token_hash=token_hash,
            jti=jti,
            scopes=scopes,
            expires_at=expires_at,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        self.db.add(db_token)
        await self.db.commit()
        
        return {"token": token, "expires_at": expires_at, "jti": jti}
    
    async def _generate_refresh_token(
        self,
        agent_identity: AgentIdentity,
        scopes: List[str],
        client_ip: Optional[str],
        user_agent: Optional[str]
    ) -> Dict[str, Any]:
        """Generate refresh token."""
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=agent_identity.refresh_token_expires_in_seconds)
        jti = str(uuid.uuid4())
        
        payload = {
            "iss": self.issuer,
            "sub": str(agent_identity.id),
            "aud": "leanvibe-api",
            "exp": int(expires_at.timestamp()),
            "iat": int(now.timestamp()),
            "jti": jti,
            "scope": scopes,
            "token_type": "refresh"
        }
        
        token = jwt.encode(payload, self.jwt_secret_key, algorithm=self.jwt_algorithm)
        token_hash = self._hash_token(token)
        
        # Store refresh token
        db_token = AgentToken(
            agent_id=agent_identity.id,
            token_type="refresh",
            token_hash=token_hash,
            jti=jti,
            scopes=scopes,
            expires_at=expires_at,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        self.db.add(db_token)
        await self.db.commit()
        
        return {"token": token, "expires_at": expires_at, "jti": jti}
    
    async def _validate_refresh_token(self, token: str) -> Dict[str, Any]:
        """Validate refresh token."""
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret_key,
                algorithms=[self.jwt_algorithm],
                issuer=self.issuer
            )
            
            if payload.get("token_type") != "refresh":
                raise TokenValidationError("Invalid token type")
            
            # Check database
            token_hash = self._hash_token(token)
            result = await self.db.execute(
                select(AgentToken).where(
                    and_(
                        AgentToken.token_hash == token_hash,
                        AgentToken.token_type == "refresh",
                        AgentToken.is_revoked == False,
                        AgentToken.expires_at > datetime.utcnow()
                    )
                )
            )
            
            if not result.scalar_one_or_none():
                raise TokenValidationError("Refresh token not found or expired")
            
            return {
                "agent_id": payload["sub"],
                "scopes": payload.get("scope", []),
                "jti": payload["jti"]
            }
            
        except jwt.InvalidTokenError as e:
            raise TokenValidationError(f"Invalid refresh token: {e}")
    
    def _hash_token(self, token: str) -> str:
        """Hash token for storage."""
        return hashlib.sha256(token.encode('utf-8')).hexdigest()
    
    async def _record_token_usage(self, jti: str) -> None:
        """Record token usage."""
        await self.db.execute(
            select(AgentToken).where(AgentToken.jti == jti)
        )
        # Usage recording is handled in token record
    
    async def _is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted."""
        token_hash = self._hash_token(token)
        blacklist_key = f"{self._token_blacklist_key_prefix}{token_hash}"
        return await self.redis.exists(blacklist_key)
    
    async def _blacklist_token(self, token: str) -> None:
        """Add token to blacklist."""
        token_hash = self._hash_token(token)
        blacklist_key = f"{self._token_blacklist_key_prefix}{token_hash}"
        # Set expiration to token's max lifetime
        await self.redis.set_with_expiry(
            blacklist_key, "1", self.config["max_token_lifetime_hours"] * 3600
        )
    
    async def _log_audit_event(
        self,
        agent_id: Optional[uuid.UUID],
        human_controller: str,
        action: str,
        resource: str,
        success: bool,
        resource_id: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
        response_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log security audit event."""
        audit_log = SecurityAuditLog(
            agent_id=agent_id,
            human_controller=human_controller,
            action=action,
            resource=resource,
            resource_id=resource_id,
            request_data=request_data,
            response_data=response_data,
            success=success,
            error_message=error_message,
            ip_address=ip_address,
            user_agent=user_agent,
            correlation_id=correlation_id,
            metadata=metadata or {}
        )
        
        self.db.add(audit_log)
        # Note: Commit handled by caller
    
    async def _create_security_event(
        self,
        event_type: str,
        severity: SecurityEventSeverity,
        description: str,
        agent_id: Optional[str] = None,
        human_controller: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        risk_score: Optional[float] = None
    ) -> None:
        """Create security event."""
        security_event = SecurityEvent(
            event_type=event_type,
            severity=severity.value,
            agent_id=uuid.UUID(agent_id) if agent_id else None,
            human_controller=human_controller,
            description=description,
            details=details or {},
            risk_score=risk_score
        )
        
        self.db.add(security_event)
        await self.db.commit()


# Factory function
async def create_agent_identity_service(
    db_session: AsyncSession,
    redis_client: RedisClient,
    jwt_secret_key: str
) -> AgentIdentityService:
    """
    Create Agent Identity Service instance.
    
    Args:
        db_session: Database session
        redis_client: Redis client
        jwt_secret_key: JWT signing key
        
    Returns:
        AgentIdentityService instance
    """
    return AgentIdentityService(db_session, redis_client, jwt_secret_key)