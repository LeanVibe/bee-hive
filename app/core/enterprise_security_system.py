"""
Enterprise Security System for LeanVibe Agent Hive 2.0

Production-grade security framework with comprehensive authentication,
authorization, audit logging, and compliance capabilities.

CRITICAL COMPONENT: Provides enterprise-ready security for autonomous development platform.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import secrets
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union
import ipaddress
from pathlib import Path

import structlog
from fastapi import HTTPException, Request, Response, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
import jwt
from pydantic import BaseModel, Field, validator
import pyotp
import qrcode
import io
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .config import settings
from .redis import get_redis

logger = structlog.get_logger()


class SecurityLevel(Enum):
    """Security clearance levels for enterprise operations."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class AuthenticationMethod(Enum):
    """Supported authentication methods."""
    PASSWORD = "password"
    MFA_TOTP = "mfa_totp"
    MFA_SMS = "mfa_sms"
    BIOMETRIC = "biometric"
    SSO_OAUTH2 = "sso_oauth2"
    SSO_SAML = "sso_saml"
    API_KEY = "api_key"


class SecurityEvent(Enum):
    """Security events for audit logging."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGIN_BLOCKED = "login_blocked"
    PASSWORD_CHANGED = "password_changed"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    CONFIGURATION_CHANGED = "configuration_changed"


class ThreatLevel(Enum):
    """Threat detection levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityConfig(BaseModel):
    """Enterprise security configuration."""
    
    # Authentication
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    mfa_enabled: bool = True
    mfa_issuer: str = "LeanVibe Agent Hive"
    session_timeout_minutes: int = 480  # 8 hours
    
    # Rate Limiting
    rate_limit_requests_per_minute: int = 60
    rate_limit_burst: int = 100
    rate_limit_window_minutes: int = 15
    
    # Password Policy
    password_min_length: int = 12
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_symbols: bool = True
    password_history_count: int = 5
    password_max_age_days: int = 90
    
    # Account Security
    max_login_attempts: int = 5
    account_lockout_duration_minutes: int = 30
    require_mfa_for_admin: bool = True
    
    # API Security
    api_key_length: int = 64
    api_key_expire_days: int = 365
    require_api_key_rotation: bool = True
    
    # Audit & Compliance
    audit_log_retention_days: int = 365
    compliance_mode: str = "SOC2_TYPE2"  # SOC2_TYPE2, GDPR, HIPAA
    enable_data_encryption: bool = True
    
    # Threat Detection
    enable_threat_detection: bool = True
    suspicious_login_threshold: int = 3
    geo_blocking_enabled: bool = False
    allowed_countries: List[str] = ["US", "CA", "GB", "DE", "AU"]


class EnterpriseSecuritySystem:
    """
    Comprehensive enterprise security system.
    
    Provides production-grade security with authentication, authorization,
    audit logging, threat detection, and compliance capabilities.
    """
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.cipher_suite = self._initialize_encryption()
        self.redis = None
        self._threat_detector = ThreatDetectionEngine(self.config)
        self._audit_logger = SecurityAuditLogger(self.config)
        self._rate_limiter = EnterpriseRateLimiter(self.config)
        
        logger.info("Enterprise security system initialized", 
                   mfa_enabled=self.config.mfa_enabled,
                   compliance_mode=self.config.compliance_mode)
    
    async def initialize(self):
        """Initialize async components."""
        self.redis = get_redis()
        await self._audit_logger.initialize()
        await self._rate_limiter.initialize()
        logger.info("Security system async components initialized")
    
    def _initialize_encryption(self) -> Fernet:
        """Initialize encryption for sensitive data."""
        # Generate key from SECRET_KEY for consistent encryption
        password = settings.SECRET_KEY.encode()
        salt = b'leanvibe_agent_hive_salt_v1'  # Fixed salt for consistency
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return Fernet(key)
    
    # Authentication Methods
    def hash_password(self, password: str) -> str:
        """Hash password with bcrypt."""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def validate_password_strength(self, password: str) -> Dict[str, Any]:
        """Validate password against security policy."""
        issues = []
        
        if len(password) < self.config.password_min_length:
            issues.append(f"Password must be at least {self.config.password_min_length} characters")
        
        if self.config.password_require_uppercase and not any(c.isupper() for c in password):
            issues.append("Password must contain at least one uppercase letter")
        
        if self.config.password_require_lowercase and not any(c.islower() for c in password):
            issues.append("Password must contain at least one lowercase letter")
        
        if self.config.password_require_numbers and not any(c.isdigit() for c in password):
            issues.append("Password must contain at least one number")
        
        if self.config.password_require_symbols and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            issues.append("Password must contain at least one symbol")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "strength_score": self._calculate_password_strength(password)
        }
    
    def _calculate_password_strength(self, password: str) -> int:
        """Calculate password strength score (0-100)."""
        score = 0
        
        # Length bonus
        score += min(25, len(password) * 2)
        
        # Character variety
        if any(c.isupper() for c in password):
            score += 15
        if any(c.islower() for c in password):
            score += 15
        if any(c.isdigit() for c in password):
            score += 15
        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 20
        
        # Common patterns penalty
        if password.lower() in ['password', '123456', 'qwerty', 'admin']:
            score -= 50
        
        return max(0, min(100, score))
    
    # Multi-Factor Authentication
    def generate_mfa_secret(self) -> str:
        """Generate TOTP secret for MFA setup."""
        return pyotp.random_base32()
    
    def generate_mfa_qr_code(self, user_email: str, secret: str) -> bytes:
        """Generate QR code for MFA setup."""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name=self.config.mfa_issuer
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        return img_buffer.getvalue()
    
    def verify_mfa_token(self, secret: str, token: str, window: int = 1) -> bool:
        """Verify TOTP token for MFA."""
        try:
            totp = pyotp.TOTP(secret)
            return totp.verify(token, valid_window=window)
        except Exception as e:
            logger.error("MFA verification failed", error=str(e))
            return False
    
    # JWT Token Management
    def create_access_token(self, user_data: Dict[str, Any]) -> str:
        """Create JWT access token with enhanced security."""
        now = datetime.utcnow()
        expire = now + timedelta(minutes=self.config.jwt_access_token_expire_minutes)
        
        payload = {
            "sub": user_data["id"],
            "email": user_data["email"],
            "role": user_data["role"],
            "permissions": user_data.get("permissions", []),
            "security_level": user_data.get("security_level", SecurityLevel.INTERNAL.value),
            "mfa_verified": user_data.get("mfa_verified", False),
            "session_id": str(uuid.uuid4()),
            "exp": expire,
            "iat": now,
            "nbf": now,
            "iss": "leanvibe-agent-hive",
            "aud": "leanvibe-users",
            "type": "access_token"
        }
        
        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    
    def create_refresh_token(self, user_id: str, session_id: str) -> str:
        """Create JWT refresh token."""
        now = datetime.utcnow()
        expire = now + timedelta(days=self.config.jwt_refresh_token_expire_days)
        
        payload = {
            "sub": user_id,
            "session_id": session_id,
            "exp": expire,
            "iat": now,
            "type": "refresh_token"
        }
        
        return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token with security checks."""
        try:
            payload = jwt.decode(
                token, 
                settings.JWT_SECRET_KEY, 
                algorithms=[settings.JWT_ALGORITHM],
                options={"verify_exp": True, "verify_iat": True, "verify_nbf": True}
            )
            
            # Additional security validations
            if payload.get("iss") != "leanvibe-agent-hive":
                raise jwt.InvalidTokenError("Invalid issuer")
            
            if payload.get("aud") != "leanvibe-users":
                raise jwt.InvalidTokenError("Invalid audience")
            
            # Note: Blacklist check would need to be done in async context
            # For now, skip this check in sync verify_token method
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token", error=str(e))
            return None
    
    async def verify_token_async(self, token: str) -> Optional[Dict[str, Any]]:
        """Async version of verify_token with blacklist checking."""
        try:
            payload = jwt.decode(
                token, 
                settings.JWT_SECRET_KEY, 
                algorithms=[settings.JWT_ALGORITHM],
                options={"verify_exp": True, "verify_iat": True, "verify_nbf": True}
            )
            
            # Additional security validations
            if payload.get("iss") != "leanvibe-agent-hive":
                raise jwt.InvalidTokenError("Invalid issuer")
            
            if payload.get("aud") != "leanvibe-users":
                raise jwt.InvalidTokenError("Invalid audience")
            
            # Check if token is blacklisted
            if await self._is_token_blacklisted(payload.get("session_id")):
                raise jwt.InvalidTokenError("Token has been revoked")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid token", error=str(e))
            return None
    
    async def _is_token_blacklisted(self, session_id: str) -> bool:
        """Check if token session is blacklisted."""
        if not self.redis or not session_id:
            return False
        
        try:
            result = await self.redis.get(f"blacklist:session:{session_id}")
            return result is not None
        except Exception:
            return False
    
    async def blacklist_token(self, session_id: str):
        """Add token session to blacklist."""
        if self.redis and session_id:
            try:
                await self.redis.set(
                    f"blacklist:session:{session_id}",
                    "revoked",
                    ex=self.config.jwt_refresh_token_expire_days * 24 * 3600
                )
                logger.info("Token session blacklisted", session_id=session_id)
            except Exception as e:
                logger.error("Failed to blacklist token", error=str(e))
    
    # API Key Management
    def generate_api_key(self, prefix: str = "lv") -> tuple[str, str]:
        """Generate secure API key and hash."""
        # Generate secure random key
        key_bytes = secrets.token_bytes(self.config.api_key_length // 2)
        key = f"{prefix}_{base64.urlsafe_b64encode(key_bytes).decode().rstrip('=')}"
        
        # Create hash for storage
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        return key, key_hash
    
    def verify_api_key(self, provided_key: str, stored_hash: str) -> bool:
        """Verify API key against stored hash."""
        provided_hash = hashlib.sha256(provided_key.encode()).hexdigest()
        return hmac.compare_digest(provided_hash, stored_hash)
    
    # Data Encryption
    def encrypt_sensitive_data(self, data: Union[str, dict]) -> str:
        """Encrypt sensitive data for storage."""
        if isinstance(data, dict):
            data = json.dumps(data)
        
        encrypted_data = self.cipher_suite.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> Union[str, dict]:
        """Decrypt sensitive data from storage."""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_bytes = self.cipher_suite.decrypt(decoded_data)
            decrypted_str = decrypted_bytes.decode()
            
            # Try to parse as JSON
            try:
                return json.loads(decrypted_str)
            except json.JSONDecodeError:
                return decrypted_str
        except Exception as e:
            logger.error("Data decryption failed", error=str(e))
            raise
    
    # Security Event Logging
    async def log_security_event(self, event: SecurityEvent, user_id: str = None, 
                                request: Request = None, **kwargs):
        """Log security event for audit trail."""
        await self._audit_logger.log_event(
            event=event,
            user_id=user_id,
            request=request,
            **kwargs
        )
    
    # Threat Detection
    async def detect_threat(self, request: Request, user_id: str = None) -> Optional[Dict[str, Any]]:
        """Detect potential security threats."""
        return await self._threat_detector.analyze_request(request, user_id)
    
    # Rate Limiting
    async def check_rate_limit(self, identifier: str, action: str = "api_call") -> bool:
        """Check if request is within rate limits."""
        return await self._rate_limiter.check_limit(identifier, action)
    
    async def get_rate_limit_info(self, identifier: str, action: str = "api_call") -> Dict[str, Any]:
        """Get current rate limit status."""
        return await self._rate_limiter.get_limit_info(identifier, action)


class SecurityAuditLogger:
    """Comprehensive security audit logging system."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.redis = None
        self._redis_initialized = False
        
    async def initialize(self):
        """Initialize async components."""
        try:
            self.redis = get_redis()
            self._redis_initialized = True
        except Exception as e:
            logger.warning(f"Redis initialization failed for audit logger: {e}")
            self.redis = None
            self._redis_initialized = False
    
    async def _ensure_redis_connection(self):
        """Ensure Redis connection is available, with lazy initialization."""
        if not self._redis_initialized or self.redis is None:
            try:
                from .redis import get_redis
                self.redis = get_redis()
                self._redis_initialized = True
                # Test the connection
                await self.redis.ping()
            except Exception as e:
                logger.warning(f"Redis connection failed in audit logger: {e}")
                self.redis = None
                self._redis_initialized = False
                return False
        return True
    
    async def log_event(self, event: SecurityEvent, user_id: str = None,
                       request: Request = None, **kwargs):
        """Log security event with comprehensive context."""
        timestamp = datetime.utcnow()
        
        event_data = {
            "event_id": str(uuid.uuid4()),
            "event_type": event.value,
            "timestamp": timestamp.isoformat(),
            "user_id": user_id,
            "severity": self._get_event_severity(event),
            **kwargs
        }
        
        # Add request context if available
        if request:
            event_data.update({
                "ip_address": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent"),
                "path": str(request.url.path),
                "method": request.method,
                "host": request.headers.get("host")
            })
        
        # Store in Redis for real-time monitoring
        if await self._ensure_redis_connection():
            try:
                await self.redis.lpush("security_events", json.dumps(event_data))
                await self.redis.ltrim("security_events", 0, 9999)  # Keep last 10k events
            except Exception as e:
                logger.error("Failed to store security event in Redis", error=str(e))
        
        # Log to structured logger
        logger.info("Security event", **event_data)
    
    def _get_event_severity(self, event: SecurityEvent) -> str:
        """Determine event severity level."""
        high_severity = {
            SecurityEvent.LOGIN_BLOCKED,
            SecurityEvent.SUSPICIOUS_ACTIVITY,
            SecurityEvent.PERMISSION_DENIED
        }
        
        if event in high_severity:
            return "HIGH"
        elif event in [SecurityEvent.LOGIN_FAILED, SecurityEvent.API_KEY_REVOKED]:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        forwarded = request.headers.get("x-forwarded")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to client host
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"


class ThreatDetectionEngine:
    """Advanced threat detection and analysis system."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.redis = None
        self._redis_initialized = False
        self.suspicious_patterns = self._load_suspicious_patterns()
    
    async def initialize(self):
        """Initialize async components."""
        try:
            self.redis = get_redis()
            self._redis_initialized = True
        except Exception as e:
            logger.warning(f"Redis initialization failed for threat detector: {e}")
            self.redis = None
            self._redis_initialized = False
    
    async def _ensure_redis_connection(self):
        """Ensure Redis connection is available, with lazy initialization."""
        if not self._redis_initialized or self.redis is None:
            try:
                from .redis import get_redis
                self.redis = get_redis()
                self._redis_initialized = True
                # Test the connection
                await self.redis.ping()
            except Exception as e:
                logger.warning(f"Redis connection failed in threat detector: {e}")
                self.redis = None
                self._redis_initialized = False
                return False
        return True
    
    def _load_suspicious_patterns(self) -> Dict[str, List[str]]:
        """Load patterns for threat detection."""
        return {
            "sql_injection": [
                r"union.*select", r"drop.*table", r"exec.*xp_",
                r"';.*--", r"1'.*or.*'1'.*=.*'1"
            ],
            "xss": [
                r"<script", r"javascript:", r"onload=", r"onerror=",
                r"expression\(", r"vbscript:"
            ],
            "path_traversal": [
                r"\.\./", r"\.\.\\", r"%2e%2e%2f", r"%2e%2e%5c"
            ],
            "command_injection": [
                r";.*rm.*-rf", r"&&.*cat", r"\|.*nc", r"`.*whoami`"
            ]
        }
    
    async def analyze_request(self, request: Request, user_id: str = None) -> Optional[Dict[str, Any]]:
        """Analyze request for potential threats."""
        threats = []
        
        # Analyze URL and query parameters
        url_threats = self._analyze_url_threats(str(request.url))
        if url_threats:
            threats.extend(url_threats)
        
        # Analyze headers
        header_threats = self._analyze_header_threats(request.headers)
        if header_threats:
            threats.extend(header_threats)
        
        # Check for suspicious IP activity
        if user_id:
            ip_threats = await self._analyze_ip_behavior(
                self._get_client_ip(request), user_id
            )
            if ip_threats:
                threats.extend(ip_threats)
        
        if threats:
            threat_level = self._calculate_threat_level(threats)
            return {
                "threat_detected": True,
                "threat_level": threat_level.value,
                "threats": threats,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return None
    
    def _analyze_url_threats(self, url: str) -> List[Dict[str, Any]]:
        """Analyze URL for malicious patterns."""
        threats = []
        url_lower = url.lower()
        
        for threat_type, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if pattern in url_lower:
                    threats.append({
                        "type": threat_type,
                        "pattern": pattern,
                        "location": "url",
                        "severity": "high" if threat_type in ["sql_injection", "command_injection"] else "medium"
                    })
        
        return threats
    
    def _analyze_header_threats(self, headers) -> List[Dict[str, Any]]:
        """Analyze HTTP headers for threats."""
        threats = []
        
        # Check User-Agent for known attack tools
        user_agent = headers.get("user-agent", "").lower()
        suspicious_agents = [
            "sqlmap", "nikto", "nessus", "burpsuite", "owaspzap",
            "python-requests", "curl", "wget"
        ]
        
        for agent in suspicious_agents:
            if agent in user_agent:
                threats.append({
                    "type": "suspicious_user_agent",
                    "pattern": agent,
                    "location": "headers",
                    "severity": "medium"
                })
        
        return threats
    
    async def _analyze_ip_behavior(self, ip_address: str, user_id: str) -> List[Dict[str, Any]]:
        """Analyze IP address behavior patterns."""
        threats = []
        
        if ip_address == "unknown" or not await self._ensure_redis_connection():
            return threats
        
        try:
            # Check request rate from this IP
            current_time = time.time()
            window_start = current_time - 300  # 5 minutes
            
            request_count = await self.redis.zcount(
                f"ip_requests:{ip_address}",
                window_start,
                current_time
            )
            
            if request_count > 100:  # More than 100 requests in 5 minutes
                threats.append({
                    "type": "high_request_rate",
                    "pattern": f"{request_count} requests in 5 minutes",
                    "location": "ip_behavior",
                    "severity": "high"
                })
            
            # Track this request
            await self.redis.zadd(
                f"ip_requests:{ip_address}",
                {str(current_time): current_time}
            )
            await self.redis.expire(f"ip_requests:{ip_address}", 3600)
            
        except Exception as e:
            logger.error("IP behavior analysis failed", error=str(e))
        
        return threats
    
    def _calculate_threat_level(self, threats: List[Dict[str, Any]]) -> ThreatLevel:
        """Calculate overall threat level."""
        high_count = sum(1 for t in threats if t.get("severity") == "high")
        medium_count = sum(1 for t in threats if t.get("severity") == "medium")
        
        if high_count >= 2 or (high_count >= 1 and medium_count >= 2):
            return ThreatLevel.CRITICAL
        elif high_count >= 1 or medium_count >= 3:
            return ThreatLevel.HIGH
        elif medium_count >= 1:
            return ThreatLevel.MEDIUM
        else:
            return ThreatLevel.LOW
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"


class EnterpriseRateLimiter:
    """Advanced rate limiting system with multiple strategies."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.redis = None
        self._redis_initialized = False
        
    async def initialize(self):
        """Initialize async components."""
        try:
            self.redis = get_redis()
            self._redis_initialized = True
        except Exception as e:
            logger.warning(f"Redis initialization failed for rate limiter: {e}")
            self.redis = None
            self._redis_initialized = False
    
    async def _ensure_redis_connection(self):
        """Ensure Redis connection is available, with lazy initialization."""
        if not self._redis_initialized or self.redis is None:
            try:
                from .redis import get_redis
                self.redis = get_redis()
                self._redis_initialized = True
                # Test the connection
                await self.redis.ping()
            except Exception as e:
                logger.warning(f"Redis connection failed in rate limiter: {e}")
                self.redis = None
                self._redis_initialized = False
                return False
        return True
    
    async def check_limit(self, identifier: str, action: str = "api_call") -> bool:
        """Check if request is within rate limits."""
        # Ensure Redis connection is available
        if not await self._ensure_redis_connection():
            logger.debug("Redis not available for rate limiting, allowing request")
            return True  # Allow if Redis not available
        
        try:
            current_time = time.time()
            window_start = current_time - (self.config.rate_limit_window_minutes * 60)
            
            # Use sliding window rate limiting
            key = f"rate_limit:{action}:{identifier}"
            
            # Remove old entries
            await self.redis.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            current_count = await self.redis.zcard(key)
            
            if current_count >= self.config.rate_limit_requests_per_minute:
                logger.info(f"Rate limit exceeded for {identifier}:{action}", 
                           current_count=current_count, 
                           limit=self.config.rate_limit_requests_per_minute)
                return False
            
            # Add current request
            await self.redis.zadd(key, {str(uuid.uuid4()): current_time})
            await self.redis.expire(key, self.config.rate_limit_window_minutes * 60)
            
            return True
            
        except Exception as e:
            logger.error("Rate limit check failed", error=str(e), identifier=identifier)
            return True  # Allow on error to prevent service disruption
    
    async def get_limit_info(self, identifier: str, action: str = "api_call") -> Dict[str, Any]:
        """Get current rate limit information."""
        # Ensure Redis connection is available
        if not await self._ensure_redis_connection():
            return {
                "requests_remaining": self.config.rate_limit_requests_per_minute,
                "redis_available": False
            }
        
        try:
            current_time = time.time()
            window_start = current_time - (self.config.rate_limit_window_minutes * 60)
            
            key = f"rate_limit:{action}:{identifier}"
            
            # Clean old entries
            await self.redis.zremrangebyscore(key, 0, window_start)
            
            # Get current count
            current_count = await self.redis.zcard(key)
            
            return {
                "requests_made": current_count,
                "requests_remaining": max(0, self.config.rate_limit_requests_per_minute - current_count),
                "window_minutes": self.config.rate_limit_window_minutes,
                "reset_time": window_start + (self.config.rate_limit_window_minutes * 60),
                "redis_available": True
            }
            
        except Exception as e:
            logger.error("Rate limit info failed", error=str(e), identifier=identifier)
            return {
                "requests_remaining": 0,
                "error": str(e),
                "redis_available": False
            }


# Global security system instance
_security_system: Optional[EnterpriseSecuritySystem] = None


async def get_security_system() -> EnterpriseSecuritySystem:
    """Get or create security system instance with graceful error handling."""
    global _security_system
    if _security_system is None:
        try:
            _security_system = EnterpriseSecuritySystem()
            await _security_system.initialize()
            logger.info("Enterprise security system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize enterprise security system: {e}")
            # Create a fallback instance that works without Redis
            _security_system = EnterpriseSecuritySystem()
            # Don't call initialize to avoid Redis dependency
            logger.warning("Enterprise security system running in fallback mode (no Redis features)")
    return _security_system


# Security middleware and utilities
class SecurityMiddleware:
    """Enterprise security middleware for FastAPI."""
    
    def __init__(self):
        self.security_system = None
    
    async def __call__(self, request: Request, call_next):
        """Process request through security pipeline."""
        if self.security_system is None:
            self.security_system = await get_security_system()
        
        start_time = time.time()
        
        try:
            # Threat detection
            threat_analysis = await self.security_system.detect_threat(request)
            if threat_analysis and threat_analysis.get("threat_level") in ["HIGH", "CRITICAL"]:
                await self.security_system.log_security_event(
                    SecurityEvent.SUSPICIOUS_ACTIVITY,
                    request=request,
                    threat_analysis=threat_analysis
                )
                
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Request blocked due to security policy"
                )
            
            # Rate limiting
            client_ip = self._get_client_ip(request)
            if not await self.security_system.check_rate_limit(client_ip):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            
            # Log successful request
            process_time = time.time() - start_time
            logger.info("Request processed",
                       path=str(request.url.path),
                       method=request.method,
                       status_code=response.status_code,
                       process_time=process_time)
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Security middleware error", error=str(e))
            # Log security incident
            await self.security_system.log_security_event(
                SecurityEvent.SUSPICIOUS_ACTIVITY,
                request=request,
                error=str(e)
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal security error"
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"


# FastAPI Dependencies for Security

security_bearer = HTTPBearer()

async def get_current_user_secure(
    credentials: HTTPAuthorizationCredentials = Depends(security_bearer)
) -> Dict[str, Any]:
    """Get current authenticated user with enhanced security checks."""
    security_system = await get_security_system()
    
    token_payload = security_system.verify_token(credentials.credentials)
    if not token_payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return token_payload


def require_security_level(required_level: SecurityLevel):
    """Require minimum security clearance level."""
    def security_checker(current_user: Dict[str, Any] = Depends(get_current_user_secure)):
        user_level = SecurityLevel(current_user.get("security_level", SecurityLevel.PUBLIC.value))
        
        # Define security level hierarchy
        level_hierarchy = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.RESTRICTED: 3,
            SecurityLevel.TOP_SECRET: 4
        }
        
        if level_hierarchy[user_level] < level_hierarchy[required_level]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient security clearance. Required: {required_level.value}"
            )
        
        return current_user
    
    return security_checker


def require_mfa_verification():
    """Require MFA verification for sensitive operations."""
    def mfa_checker(current_user: Dict[str, Any] = Depends(get_current_user_secure)):
        if not current_user.get("mfa_verified", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Multi-factor authentication required"
            )
        
        return current_user
    
    return mfa_checker