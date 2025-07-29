"""
Enhanced JWT Token Management with Key Rotation for LeanVibe Agent Hive 2.0.

Implements secure JWT token handling with automatic key rotation, token blacklisting,
comprehensive validation, and advanced security features.
"""

import uuid
import time
import json
import secrets
import hashlib
import base64
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
from cryptography.hazmat.backends import default_backend

import jwt
import structlog

from .redis import RedisClient

logger = structlog.get_logger()


class KeyAlgorithm(Enum):
    """JWT signing algorithm types."""
    HS256 = "HS256"  # HMAC with SHA-256
    HS384 = "HS384"  # HMAC with SHA-384  
    HS512 = "HS512"  # HMAC with SHA-512
    RS256 = "RS256"  # RSA with SHA-256
    RS384 = "RS384"  # RSA with SHA-384
    RS512 = "RS512"  # RSA with SHA-512
    ES256 = "ES256"  # ECDSA with SHA-256
    ES384 = "ES384"  # ECDSA with SHA-384
    ES512 = "ES512"  # ECDSA with SHA-512


class TokenType(Enum):
    """JWT token types."""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    TEMPORARY = "temporary"


class KeyStatus(Enum):
    """Key status enumeration."""
    ACTIVE = "active"
    ROTATING = "rotating"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"


@dataclass
class JWTKey:
    """JWT signing/verification key."""
    key_id: str
    algorithm: KeyAlgorithm
    private_key: Optional[str]  # PEM format
    public_key: str  # PEM format
    status: KeyStatus
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    usage_count: int = 0
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes private key)."""
        return {
            "key_id": self.key_id,
            "algorithm": self.algorithm.value,
            "public_key": self.public_key,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "usage_count": self.usage_count,
            "metadata": self.metadata or {}
        }


@dataclass
class TokenValidationResult:
    """Result of token validation."""
    is_valid: bool
    payload: Optional[Dict[str, Any]]
    error: Optional[str]
    key_id: Optional[str]
    algorithm: Optional[str]
    expires_at: Optional[datetime]
    issued_at: Optional[datetime]
    token_age_seconds: float
    validation_time_ms: float
    security_warnings: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "payload": self.payload,
            "error": self.error,
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "issued_at": self.issued_at.isoformat() if self.issued_at else None,
            "token_age_seconds": self.token_age_seconds,
            "validation_time_ms": self.validation_time_ms,
            "security_warnings": self.security_warnings or []
        }


@dataclass
class TokenGenerationOptions:
    """Options for token generation."""
    token_type: TokenType = TokenType.ACCESS
    expires_in_seconds: Optional[int] = None
    audience: Optional[str] = None
    issuer: Optional[str] = None
    subject: Optional[str] = None
    key_id: Optional[str] = None
    algorithm: Optional[KeyAlgorithm] = None
    include_jti: bool = True
    include_iat: bool = True
    include_nbf: bool = False
    custom_claims: Optional[Dict[str, Any]] = None


class EnhancedJWTManager:
    """
    Enhanced JWT Manager with automatic key rotation and advanced security features.
    
    Features:
    - Automatic key rotation with configurable intervals
    - Multiple signing algorithms support (HMAC, RSA, ECDSA)
    - Token blacklisting and revocation
    - Comprehensive token validation with security warnings
    - Key lifecycle management
    - Performance-optimized verification
    - Audit logging for all token operations
    - Token replay protection
    - Advanced token claims validation
    """
    
    def __init__(
        self,
        redis_client: RedisClient,
        default_algorithm: KeyAlgorithm = KeyAlgorithm.RS256,
        default_issuer: str = "leanvibe-agent-hive",
        key_rotation_interval_hours: int = 24,
        max_active_keys: int = 3,
        enable_automatic_rotation: bool = True
    ):
        """Initialize enhanced JWT manager."""
        self.redis = redis_client
        self.default_algorithm = default_algorithm
        self.default_issuer = default_issuer
        self.key_rotation_interval = timedelta(hours=key_rotation_interval_hours)
        self.max_active_keys = max_active_keys
        self.enable_automatic_rotation = enable_automatic_rotation
        
        # Redis key prefixes
        self.keys_prefix = "jwt:keys:"
        self.blacklist_prefix = "jwt:blacklist:"
        self.metrics_prefix = "jwt:metrics:"
        self.audit_prefix = "jwt:audit:"
        
        # Active keys cache
        self._active_keys: Dict[str, JWTKey] = {}
        self._keys_last_loaded = datetime.min
        self._keys_cache_ttl = timedelta(minutes=5)
        
        # Configuration
        self.config = {
            "default_access_token_ttl_seconds": 3600,  # 1 hour
            "default_refresh_token_ttl_seconds": 604800,  # 7 days
            "default_api_key_ttl_seconds": 2592000,  # 30 days
            "max_token_age_seconds": 86400,  # 24 hours
            "enable_token_blacklisting": True,
            "enable_jti_tracking": True,
            "enable_replay_protection": True,
            "max_clock_skew_seconds": 300,  # 5 minutes
            "require_exp_claim": True,
            "require_iat_claim": True,
            "require_iss_claim": True,
            "require_aud_claim": False,
            "enable_key_usage_tracking": True,
            "key_deprecation_warning_days": 7,
            "automatic_cleanup_interval_hours": 6
        }
        
        # Performance metrics
        self.metrics = {
            "tokens_generated": 0,
            "tokens_validated": 0,
            "keys_rotated": 0,
            "tokens_blacklisted": 0,
            "validation_errors": 0,
            "algorithm_usage": {alg.value: 0 for alg in KeyAlgorithm},
            "avg_generation_time_ms": 0.0,
            "avg_validation_time_ms": 0.0,
            "security_warnings_count": 0
        }
        
        # Initialize with default key if none exist
        self._initialize_default_key_if_needed()
    
    async def generate_token(
        self,
        payload: Dict[str, Any],
        options: Optional[TokenGenerationOptions] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate JWT token with specified payload and options.
        
        Args:
            payload: Token payload/claims
            options: Token generation options
            
        Returns:
            Tuple of (token_string, token_metadata)
            
        Raises:
            ValueError: If token generation fails
        """
        start_time = time.time()
        
        try:
            # Set default options
            if options is None:
                options = TokenGenerationOptions()
            
            # Get active signing key
            signing_key = await self._get_signing_key(options.key_id, options.algorithm)
            if not signing_key:
                raise ValueError("No active signing key available")
            
            # Prepare claims
            now = datetime.utcnow()
            claims = payload.copy()
            
            # Add standard claims
            if options.include_iat or self.config["require_iat_claim"]:
                claims["iat"] = int(now.timestamp())
            
            if options.include_jti:
                claims["jti"] = str(uuid.uuid4())
            
            if options.include_nbf:
                claims["nbf"] = int(now.timestamp())
            
            # Add expiration
            if options.expires_in_seconds:
                expires_in = options.expires_in_seconds
            else:
                expires_in = self._get_default_ttl(options.token_type)
            
            claims["exp"] = int((now + timedelta(seconds=expires_in)).timestamp())
            
            # Add issuer and audience
            claims["iss"] = options.issuer or self.default_issuer
            if options.audience:
                claims["aud"] = options.audience
            
            if options.subject:
                claims["sub"] = options.subject
            
            # Add token type
            claims["token_type"] = options.token_type.value
            
            # Add custom claims
            if options.custom_claims:
                claims.update(options.custom_claims)
            
            # Add key identifier
            claims["kid"] = signing_key.key_id
            
            # Generate token
            headers = {"kid": signing_key.key_id, "alg": signing_key.algorithm.value}
            
            if signing_key.algorithm.value.startswith("HS"):
                # HMAC algorithms - use secret key
                secret_key = signing_key.private_key or signing_key.public_key
                token = jwt.encode(claims, secret_key, algorithm=signing_key.algorithm.value, headers=headers)
            else:
                # RSA/ECDSA algorithms - use private key
                if not signing_key.private_key:
                    raise ValueError(f"Private key required for algorithm {signing_key.algorithm.value}")
                
                private_key = serialization.load_pem_private_key(
                    signing_key.private_key.encode('utf-8'),
                    password=None,
                    backend=default_backend()
                )
                token = jwt.encode(claims, private_key, algorithm=signing_key.algorithm.value, headers=headers)
            
            # Update key usage
            await self._update_key_usage(signing_key.key_id)
            
            # Track JTI if enabled
            if options.include_jti and self.config["enable_jti_tracking"]:
                await self._track_jti(claims["jti"], expires_in)
            
            # Prepare metadata
            expires_at = datetime.fromtimestamp(claims["exp"])
            metadata = {
                "token_id": claims.get("jti"),
                "key_id": signing_key.key_id,
                "algorithm": signing_key.algorithm.value,
                "token_type": options.token_type.value,
                "issued_at": now.isoformat(),
                "expires_at": expires_at.isoformat(),
                "expires_in_seconds": expires_in,
                "issuer": claims["iss"],
                "audience": claims.get("aud"),
                "subject": claims.get("sub")
            }
            
            # Update metrics
            generation_time = (time.time() - start_time) * 1000
            self.metrics["tokens_generated"] += 1
            self.metrics["algorithm_usage"][signing_key.algorithm.value] += 1
            self._update_avg_metric("avg_generation_time_ms", generation_time)
            
            # Audit log
            await self._log_token_event("generate", {
                "key_id": signing_key.key_id,
                "algorithm": signing_key.algorithm.value,
                "token_type": options.token_type.value,
                "expires_in_seconds": expires_in,
                "generation_time_ms": generation_time
            })
            
            logger.info(
                "JWT token generated",
                key_id=signing_key.key_id,
                algorithm=signing_key.algorithm.value,
                token_type=options.token_type.value,
                expires_in=expires_in
            )
            
            return token, metadata
            
        except Exception as e:
            logger.error(f"Token generation failed: {e}")
            raise ValueError(f"Failed to generate token: {str(e)}")
    
    async def validate_token(
        self,
        token: str,
        expected_audience: Optional[str] = None,
        required_claims: Optional[List[str]] = None,
        verify_signature: bool = True,
        check_blacklist: bool = True
    ) -> TokenValidationResult:
        """
        Validate JWT token with comprehensive security checks.
        
        Args:
            token: JWT token string
            expected_audience: Expected audience claim
            required_claims: List of required claims
            verify_signature: Whether to verify token signature
            check_blacklist: Whether to check token blacklist
            
        Returns:
            TokenValidationResult with validation outcome
        """
        start_time = time.time()
        security_warnings = []
        
        try:
            self.metrics["tokens_validated"] += 1
            
            # Check blacklist first
            if check_blacklist and self.config["enable_token_blacklisting"]:
                if await self._is_token_blacklisted(token):
                    return TokenValidationResult(
                        is_valid=False,
                        payload=None,
                        error="Token is blacklisted",
                        key_id=None,
                        algorithm=None,
                        expires_at=None,
                        issued_at=None,
                        token_age_seconds=0,
                        validation_time_ms=(time.time() - start_time) * 1000,
                        security_warnings=["blacklisted_token"]
                    )
            
            # Decode header to get key ID and algorithm
            try:
                header = jwt.get_unverified_header(token)
                key_id = header.get("kid")
                algorithm = header.get("alg")
            except Exception as e:
                return TokenValidationResult(
                    is_valid=False,
                    payload=None,
                    error=f"Invalid token header: {str(e)}",
                    key_id=None,
                    algorithm=None,
                    expires_at=None,
                    issued_at=None,
                    token_age_seconds=0,
                    validation_time_ms=(time.time() - start_time) * 1000,
                    security_warnings=["invalid_header"]
                )
            
            # Get verification key
            verification_key = await self._get_verification_key(key_id)
            if not verification_key:
                return TokenValidationResult(
                    is_valid=False,
                    payload=None,
                    error=f"Unknown key ID: {key_id}",
                    key_id=key_id,
                    algorithm=algorithm,
                    expires_at=None,
                    issued_at=None,
                    token_age_seconds=0,
                    validation_time_ms=(time.time() - start_time) * 1000,
                    security_warnings=["unknown_key"]
                )
            
            # Check key status
            if verification_key.status == KeyStatus.REVOKED:
                security_warnings.append("revoked_key")
                return TokenValidationResult(
                    is_valid=False,
                    payload=None,
                    error="Token signed with revoked key",
                    key_id=key_id,
                    algorithm=algorithm,
                    expires_at=None,
                    issued_at=None,
                    token_age_seconds=0,
                    validation_time_ms=(time.time() - start_time) * 1000,
                    security_warnings=security_warnings
                )
            
            if verification_key.status == KeyStatus.DEPRECATED:
                security_warnings.append("deprecated_key")
            
            # Prepare verification key
            if algorithm.startswith("HS"):
                # HMAC algorithms
                verify_key = verification_key.private_key or verification_key.public_key
            else:
                # RSA/ECDSA algorithms - use public key
                verify_key = serialization.load_pem_public_key(
                    verification_key.public_key.encode('utf-8'),
                    backend=default_backend()
                )
            
            # Decode and verify token
            options = {
                "verify_signature": verify_signature,
                "verify_exp": self.config["require_exp_claim"],
                "verify_iat": self.config["require_iat_claim"],
                "verify_iss": self.config["require_iss_claim"],
                "verify_aud": self.config["require_aud_claim"] or expected_audience is not None,
                "require_exp": self.config["require_exp_claim"],
                "require_iat": self.config["require_iat_claim"],
                "require_iss": self.config["require_iss_claim"]
            }
            
            try:
                payload = jwt.decode(
                    token,
                    verify_key,
                    algorithms=[algorithm],
                    audience=expected_audience,
                    issuer=self.default_issuer,
                    options=options
                )
            except jwt.ExpiredSignatureError:
                return TokenValidationResult(
                    is_valid=False,
                    payload=None,
                    error="Token has expired",
                    key_id=key_id,
                    algorithm=algorithm,
                    expires_at=None,
                    issued_at=None,
                    token_age_seconds=0,
                    validation_time_ms=(time.time() - start_time) * 1000,
                    security_warnings=security_warnings + ["expired_token"]
                )
            except jwt.InvalidTokenError as e:
                return TokenValidationResult(
                    is_valid=False,
                    payload=None,
                    error=f"Invalid token: {str(e)}",
                    key_id=key_id,
                    algorithm=algorithm,
                    expires_at=None,
                    issued_at=None,
                    token_age_seconds=0,
                    validation_time_ms=(time.time() - start_time) * 1000,
                    security_warnings=security_warnings + ["invalid_token"]
                )
            
            # Additional security validations
            now = datetime.utcnow()
            
            # Check required claims
            if required_claims:
                missing_claims = [claim for claim in required_claims if claim not in payload]
                if missing_claims:
                    return TokenValidationResult(
                        is_valid=False,
                        payload=payload,
                        error=f"Missing required claims: {missing_claims}",
                        key_id=key_id,
                        algorithm=algorithm,
                        expires_at=None,
                        issued_at=None,
                        token_age_seconds=0,
                        validation_time_ms=(time.time() - start_time) * 1000,
                        security_warnings=security_warnings + ["missing_claims"]
                    )
            
            # Check token age
            issued_at = None
            token_age_seconds = 0
            if "iat" in payload:
                issued_at = datetime.fromtimestamp(payload["iat"])
                token_age_seconds = (now - issued_at).total_seconds()
                
                if token_age_seconds > self.config["max_token_age_seconds"]:
                    security_warnings.append("old_token")
                
                # Check for future tokens (clock skew protection)
                if issued_at > now + timedelta(seconds=self.config["max_clock_skew_seconds"]):
                    security_warnings.append("future_token")
            
            # Check expiration
            expires_at = None
            if "exp" in payload:
                expires_at = datetime.fromtimestamp(payload["exp"])
                
                # Check if token expires soon
                if expires_at < now + timedelta(minutes=5):
                    security_warnings.append("expires_soon")
            
            # Check for replay attacks (JTI tracking)
            if self.config["enable_replay_protection"] and "jti" in payload:
                if await self._is_jti_used(payload["jti"]):
                    return TokenValidationResult(
                        is_valid=False,
                        payload=payload,
                        error="Token replay detected",
                        key_id=key_id,
                        algorithm=algorithm,
                        expires_at=expires_at,
                        issued_at=issued_at,
                        token_age_seconds=token_age_seconds,
                        validation_time_ms=(time.time() - start_time) * 1000,
                        security_warnings=security_warnings + ["token_replay"]
                    )
                
                # Mark JTI as used
                await self._mark_jti_used(payload["jti"])
            
            # Update key usage
            await self._update_key_usage(key_id)
            
            # Update metrics
            validation_time = (time.time() - start_time) * 1000
            self._update_avg_metric("avg_validation_time_ms", validation_time)
            
            if security_warnings:
                self.metrics["security_warnings_count"] += len(security_warnings)
            
            # Audit log for successful validation
            await self._log_token_event("validate", {
                "key_id": key_id,
                "algorithm": algorithm,
                "token_type": payload.get("token_type"),
                "validation_time_ms": validation_time,
                "security_warnings": security_warnings
            })
            
            return TokenValidationResult(
                is_valid=True,
                payload=payload,
                error=None,
                key_id=key_id,
                algorithm=algorithm,
                expires_at=expires_at,
                issued_at=issued_at,
                token_age_seconds=token_age_seconds,
                validation_time_ms=validation_time,
                security_warnings=security_warnings
            )
            
        except Exception as e:
            self.metrics["validation_errors"] += 1
            logger.error(f"Token validation error: {e}")
            
            return TokenValidationResult(
                is_valid=False,
                payload=None,
                error=f"Validation error: {str(e)}",
                key_id=None,
                algorithm=None,
                expires_at=None,
                issued_at=None,
                token_age_seconds=0,
                validation_time_ms=(time.time() - start_time) * 1000,
                security_warnings=security_warnings + ["validation_error"]
            )
    
    async def revoke_token(self, token: str, reason: str = "manual_revocation") -> bool:
        """
        Revoke (blacklist) a token.
        
        Args:
            token: JWT token to revoke
            reason: Revocation reason
            
        Returns:
            True if token was successfully revoked
        """
        try:
            if not self.config["enable_token_blacklisting"]:
                logger.warning("Token blacklisting is disabled")
                return False
            
            # Get token hash for storage
            token_hash = self._get_token_hash(token)
            
            # Extract expiration from token
            try:
                payload = jwt.decode(token, options={"verify_signature": False})
                expires_at = payload.get("exp")
                jti = payload.get("jti")
            except:
                # If we can't decode, use long expiration
                expires_at = int((datetime.utcnow() + timedelta(days=30)).timestamp())
                jti = None
            
            # Store in blacklist
            blacklist_data = {
                "revoked_at": datetime.utcnow().isoformat(),
                "reason": reason,
                "jti": jti,
                "expires_at": expires_at
            }
            
            ttl = max(0, expires_at - int(datetime.utcnow().timestamp())) if expires_at else 2592000  # 30 days
            await self.redis.set_with_expiry(
                f"{self.blacklist_prefix}{token_hash}",
                json.dumps(blacklist_data),
                ttl
            )
            
            self.metrics["tokens_blacklisted"] += 1
            
            # Audit log
            await self._log_token_event("revoke", {
                "token_hash": token_hash[:16],  # Only log prefix for security
                "reason": reason,
                "jti": jti
            })
            
            logger.info(f"Token revoked: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Token revocation failed: {e}")
            return False
    
    async def rotate_keys(self, force: bool = False) -> Dict[str, Any]:
        """
        Rotate signing keys.
        
        Args:
            force: Force rotation even if not due
            
        Returns:
            Rotation summary
        """
        try:
            # Check if rotation is needed
            if not force and not await self._should_rotate_keys():
                return {"message": "Key rotation not needed", "rotated": False}
            
            # Generate new key
            new_key = await self._generate_new_key(self.default_algorithm)
            
            # Load current active keys
            await self._load_active_keys()
            
            # Mark old keys as deprecated if we have too many
            if len(self._active_keys) >= self.max_active_keys:
                oldest_keys = sorted(
                    self._active_keys.values(),
                    key=lambda k: k.created_at
                )[:len(self._active_keys) - self.max_active_keys + 1]
                
                for old_key in oldest_keys:
                    old_key.status = KeyStatus.DEPRECATED
                    await self._store_key(old_key)
            
            # Store new key
            await self._store_key(new_key)
            self._active_keys[new_key.key_id] = new_key
            
            self.metrics["keys_rotated"] += 1
            
            # Audit log
            await self._log_token_event("key_rotation", {
                "new_key_id": new_key.key_id,
                "algorithm": new_key.algorithm.value,
                "deprecated_keys": len(oldest_keys) if 'oldest_keys' in locals() else 0
            })
            
            logger.info(
                "JWT keys rotated",
                new_key_id=new_key.key_id,
                algorithm=new_key.algorithm.value,
                active_keys_count=len(self._active_keys)
            )
            
            return {
                "message": "Keys rotated successfully",
                "rotated": True,
                "new_key_id": new_key.key_id,
                "active_keys_count": len(self._active_keys)
            }
            
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            return {"error": str(e), "rotated": False}
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive JWT manager metrics."""
        try:
            # Get current metrics
            current_metrics = self.metrics.copy()
            
            # Add key statistics
            await self._load_active_keys()
            key_stats = {
                "active_keys_count": len([k for k in self._active_keys.values() if k.status == KeyStatus.ACTIVE]),
                "deprecated_keys_count": len([k for k in self._active_keys.values() if k.status == KeyStatus.DEPRECATED]),
                "total_keys_count": len(self._active_keys),
                "oldest_key_age_hours": 0,
                "newest_key_age_hours": 0
            }
            
            if self._active_keys:
                now = datetime.utcnow()
                oldest_key = min(self._active_keys.values(), key=lambda k: k.created_at)
                newest_key = max(self._active_keys.values(), key=lambda k: k.created_at)
                
                key_stats["oldest_key_age_hours"] = (now - oldest_key.created_at).total_seconds() / 3600
                key_stats["newest_key_age_hours"] = (now - newest_key.created_at).total_seconds() / 3600
            
            current_metrics["key_statistics"] = key_stats
            
            # Add configuration
            current_metrics["configuration"] = self.config.copy()
            
            # Calculate rates
            total_tokens = current_metrics["tokens_generated"]
            if total_tokens > 0:
                current_metrics["blacklist_rate"] = current_metrics["tokens_blacklisted"] / total_tokens
                current_metrics["validation_error_rate"] = current_metrics["validation_errors"] / total_tokens
            else:
                current_metrics["blacklist_rate"] = 0.0
                current_metrics["validation_error_rate"] = 0.0
            
            return current_metrics
            
        except Exception as e:
            logger.error(f"Failed to get JWT metrics: {e}")
            return {"error": str(e)}
    
    # Private methods
    
    async def _initialize_default_key_if_needed(self) -> None:
        """Initialize with default key if none exist."""
        try:
            await self._load_active_keys()
            
            if not self._active_keys:
                default_key = await self._generate_new_key(self.default_algorithm)
                await self._store_key(default_key)
                self._active_keys[default_key.key_id] = default_key
                
                logger.info(f"Initialized default JWT key: {default_key.key_id}")
                
        except Exception as e:
            logger.error(f"Failed to initialize default key: {e}")
    
    async def _generate_new_key(self, algorithm: KeyAlgorithm) -> JWTKey:
        """Generate new JWT signing key."""
        key_id = f"jwt_key_{secrets.token_hex(8)}"
        now = datetime.utcnow()
        
        if algorithm.value.startswith("HS"):
            # HMAC algorithms - generate secret key
            secret_key = secrets.token_urlsafe(64)  # 512 bits
            return JWTKey(
                key_id=key_id,
                algorithm=algorithm,
                private_key=secret_key,
                public_key=secret_key,  # Same for HMAC
                status=KeyStatus.ACTIVE,
                created_at=now,
                expires_at=now + self.key_rotation_interval,
                last_used_at=None,
                usage_count=0,
                metadata={"generated_by": "enhanced_jwt_manager"}
            )
        
        elif algorithm.value.startswith("RS"):
            # RSA algorithms
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            
            private_pem = private_key.private_bytes(
                encoding=Encoding.PEM,
                format=PrivateFormat.PKCS8,
                encryption_algorithm=NoEncryption()
            ).decode('utf-8')
            
            public_key = private_key.public_key()
            public_pem = public_key.public_bytes(
                encoding=Encoding.PEM,
                format=PublicFormat.SubjectPublicKeyInfo
            ).decode('utf-8')
            
            return JWTKey(
                key_id=key_id,
                algorithm=algorithm,
                private_key=private_pem,
                public_key=public_pem,
                status=KeyStatus.ACTIVE,
                created_at=now,
                expires_at=now + self.key_rotation_interval,
                last_used_at=None,
                usage_count=0,
                metadata={"generated_by": "enhanced_jwt_manager", "key_size": 2048}
            )
        
        else:
            # ECDSA algorithms not implemented yet
            raise NotImplementedError(f"Algorithm {algorithm.value} not yet supported")
    
    async def _load_active_keys(self) -> None:
        """Load active keys from Redis."""
        try:
            # Check if cache is still valid
            if (datetime.utcnow() - self._keys_last_loaded) < self._keys_cache_ttl:
                return
            
            # Scan for key patterns
            key_pattern = f"{self.keys_prefix}*"
            key_names = await self.redis.scan_pattern(key_pattern)
            
            self._active_keys.clear()
            
            for key_name in key_names:
                try:
                    key_data = await self.redis.get(key_name)
                    if key_data:
                        key_dict = json.loads(key_data.decode('utf-8'))
                        
                        # Reconstruct JWTKey object
                        jwt_key = JWTKey(
                            key_id=key_dict["key_id"],
                            algorithm=KeyAlgorithm(key_dict["algorithm"]),
                            private_key=key_dict.get("private_key"),
                            public_key=key_dict["public_key"],
                            status=KeyStatus(key_dict["status"]),
                            created_at=datetime.fromisoformat(key_dict["created_at"]),
                            expires_at=datetime.fromisoformat(key_dict["expires_at"]) if key_dict.get("expires_at") else None,
                            last_used_at=datetime.fromisoformat(key_dict["last_used_at"]) if key_dict.get("last_used_at") else None,
                            usage_count=key_dict.get("usage_count", 0),
                            metadata=key_dict.get("metadata", {})
                        )
                        
                        self._active_keys[jwt_key.key_id] = jwt_key
                        
                except Exception as e:
                    logger.debug(f"Failed to load key {key_name}: {e}")
            
            self._keys_last_loaded = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to load active keys: {e}")
    
    async def _store_key(self, key: JWTKey) -> None:
        """Store JWT key in Redis."""
        try:
            key_data = key.to_dict()
            # Include private key for storage (would use encryption in production)
            key_data["private_key"] = key.private_key
            
            await self.redis.set(
                f"{self.keys_prefix}{key.key_id}",
                json.dumps(key_data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Failed to store key {key.key_id}: {e}")
    
    async def _get_signing_key(
        self, 
        key_id: Optional[str] = None, 
        algorithm: Optional[KeyAlgorithm] = None
    ) -> Optional[JWTKey]:
        """Get signing key for token generation."""
        await self._load_active_keys()
        
        if key_id:
            # Use specific key if requested
            return self._active_keys.get(key_id)
        
        # Find active key with matching algorithm (or any active key)
        active_keys = [k for k in self._active_keys.values() if k.status == KeyStatus.ACTIVE]
        
        if algorithm:
            matching_keys = [k for k in active_keys if k.algorithm == algorithm]
            if matching_keys:
                # Return most recently created key
                return max(matching_keys, key=lambda k: k.created_at)
        
        # Return any active key
        if active_keys:
            return max(active_keys, key=lambda k: k.created_at)
        
        return None
    
    async def _get_verification_key(self, key_id: str) -> Optional[JWTKey]:
        """Get verification key for token validation."""
        await self._load_active_keys()
        
        # Return key if active or deprecated (but not revoked)
        key = self._active_keys.get(key_id)
        if key and key.status in [KeyStatus.ACTIVE, KeyStatus.DEPRECATED]:
            return key
        
        return None
    
    async def _update_key_usage(self, key_id: str) -> None:
        """Update key usage statistics."""
        if not self.config["enable_key_usage_tracking"]:
            return
        
        try:
            key = self._active_keys.get(key_id)
            if key:
                key.usage_count += 1
                key.last_used_at = datetime.utcnow()
                await self._store_key(key)
                
        except Exception as e:
            logger.debug(f"Failed to update key usage: {e}")
    
    async def _should_rotate_keys(self) -> bool:
        """Check if keys should be rotated."""
        if not self.enable_automatic_rotation:
            return False
        
        await self._load_active_keys()
        
        # Check if any active key is due for rotation
        now = datetime.utcnow()
        for key in self._active_keys.values():
            if key.status == KeyStatus.ACTIVE and key.expires_at and key.expires_at <= now:
                return True
        
        return False
    
    def _get_token_hash(self, token: str) -> str:
        """Get hash of token for blacklisting."""
        return hashlib.sha256(token.encode('utf-8')).hexdigest()
    
    async def _is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted."""
        try:
            token_hash = self._get_token_hash(token)
            blacklist_key = f"{self.blacklist_prefix}{token_hash}"
            return await self.redis.exists(blacklist_key)
        except Exception:
            return False
    
    async def _track_jti(self, jti: str, expires_in_seconds: int) -> None:
        """Track JTI for replay protection."""
        try:
            jti_key = f"jwt:jti:{jti}"
            await self.redis.set_with_expiry(jti_key, "1", expires_in_seconds)
        except Exception as e:
            logger.debug(f"Failed to track JTI: {e}")
    
    async def _is_jti_used(self, jti: str) -> bool:
        """Check if JTI has been used."""
        try:
            jti_key = f"jwt:jti:{jti}"
            return await self.redis.exists(jti_key)
        except Exception:
            return False
    
    async def _mark_jti_used(self, jti: str) -> None:
        """Mark JTI as used."""
        try:
            jti_key = f"jwt:jti:used:{jti}"
            await self.redis.set_with_expiry(jti_key, "1", 86400)  # 24 hours
        except Exception as e:
            logger.debug(f"Failed to mark JTI as used: {e}")
    
    def _get_default_ttl(self, token_type: TokenType) -> int:
        """Get default TTL for token type."""
        ttl_mapping = {
            TokenType.ACCESS: self.config["default_access_token_ttl_seconds"],
            TokenType.REFRESH: self.config["default_refresh_token_ttl_seconds"],
            TokenType.API_KEY: self.config["default_api_key_ttl_seconds"],
            TokenType.TEMPORARY: 300  # 5 minutes
        }
        
        return ttl_mapping.get(token_type, self.config["default_access_token_ttl_seconds"])
    
    def _update_avg_metric(self, metric_name: str, new_value: float) -> None:
        """Update running average metric."""
        current_avg = self.metrics[metric_name]
        if metric_name == "avg_generation_time_ms":
            total_count = self.metrics["tokens_generated"]
        else:  # validation time
            total_count = self.metrics["tokens_validated"]
        
        if total_count > 1:
            self.metrics[metric_name] = (current_avg * (total_count - 1) + new_value) / total_count
        else:
            self.metrics[metric_name] = new_value
    
    async def _log_token_event(self, event_type: str, metadata: Dict[str, Any]) -> None:
        """Log token event for audit."""
        try:
            event_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "metadata": metadata
            }
            
            event_key = f"{self.audit_prefix}{datetime.utcnow().strftime('%Y%m%d')}"
            await self.redis.lpush(event_key, json.dumps(event_data))
            
            # Set expiration for audit logs
            await self.redis.expire(event_key, 30 * 24 * 3600)  # 30 days
            
        except Exception as e:
            logger.debug(f"Failed to log token event: {e}")


# Factory function
async def create_enhanced_jwt_manager(
    redis_client: RedisClient,
    algorithm: KeyAlgorithm = KeyAlgorithm.RS256
) -> EnhancedJWTManager:
    """Create enhanced JWT manager instance."""
    return EnhancedJWTManager(redis_client, default_algorithm=algorithm)