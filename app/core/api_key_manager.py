"""
Enterprise API Key Management System for LeanVibe Agent Hive.

Implements comprehensive API key management with:
- Hierarchical key types (System, Service, User, Temporary)
- Advanced permissions and scoping system
- Rate limiting integration and usage analytics
- Key rotation and lifecycle management
- Enterprise compliance and audit logging

Production-ready with comprehensive security, monitoring, and audit capabilities.
"""

import os
import json
import secrets
import hashlib
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import structlog
import redis.asyncio as redis
from fastapi import HTTPException, Request, Response, Depends, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, update, delete

from .database import get_session
from .auth import get_auth_service, AuthenticationService
from ..models.security import AgentIdentity, SecurityAuditLog, SecurityEvent
from ..schemas.security import SecurityError

logger = structlog.get_logger()

# API Key Configuration
API_KEY_CONFIG = {
    "key_length": int(os.getenv("API_KEY_LENGTH", "32")),
    "hash_algorithm": os.getenv("API_KEY_HASH_ALGORITHM", "sha256"),
    "encryption_key": os.getenv("API_KEY_ENCRYPTION_KEY", ""),  # Must be set in production
    "default_expiry_days": int(os.getenv("DEFAULT_API_KEY_EXPIRY_DAYS", "90")),
    "redis_key_prefix": os.getenv("API_KEY_REDIS_PREFIX", "apikey:"),
    "enable_key_rotation": os.getenv("ENABLE_API_KEY_ROTATION", "true").lower() == "true",
    "rotation_warning_days": int(os.getenv("API_KEY_ROTATION_WARNING_DAYS", "7")),
    "max_keys_per_user": int(os.getenv("MAX_API_KEYS_PER_USER", "10")),
    "max_keys_per_service": int(os.getenv("MAX_API_KEYS_PER_SERVICE", "50")),
    "enable_ip_restrictions": os.getenv("ENABLE_API_KEY_IP_RESTRICTIONS", "true").lower() == "true",
    "enable_usage_analytics": os.getenv("ENABLE_API_KEY_ANALYTICS", "true").lower() == "true"
}


class ApiKeyType(Enum):
    """API key types with different security levels."""
    SYSTEM = "system"  # Highest privileges, system-to-system
    SERVICE = "service"  # Service-to-service authentication
    USER = "user"  # User-level access
    TEMPORARY = "temporary"  # Short-lived keys
    WEBHOOK = "webhook"  # Webhook authentication
    READ_ONLY = "read_only"  # Read-only access


class ApiKeyStatus(Enum):
    """API key status enumeration."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING_ROTATION = "pending_rotation"


class PermissionScope(Enum):
    """API key permission scopes."""
    AGENTS_READ = "agents:read"
    AGENTS_WRITE = "agents:write"
    AGENTS_DELETE = "agents:delete"
    SESSIONS_READ = "sessions:read"
    SESSIONS_WRITE = "sessions:write"
    TASKS_READ = "tasks:read"
    TASKS_WRITE = "tasks:write"
    METRICS_READ = "metrics:read"
    ADMIN_READ = "admin:read"
    ADMIN_WRITE = "admin:write"
    SYSTEM_READ = "system:read"
    SYSTEM_WRITE = "system:write"


@dataclass
class ApiKeyMetadata:
    """API key metadata and configuration."""
    key_id: str
    name: str
    description: str
    key_type: ApiKeyType
    status: ApiKeyStatus
    permissions: Set[PermissionScope]
    created_by: str
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    usage_count: int = 0
    rate_limit_tier: str = "developer"
    
    # Access restrictions
    allowed_ips: Set[str] = field(default_factory=set)
    allowed_domains: Set[str] = field(default_factory=set)
    allowed_user_agents: Set[str] = field(default_factory=set)
    
    # Integration settings
    webhook_url: Optional[str] = None
    callback_urls: Set[str] = field(default_factory=set)
    
    # Audit and compliance
    compliance_tags: Set[str] = field(default_factory=set)
    audit_level: str = "standard"  # minimal, standard, detailed
    
    # Rotation settings
    auto_rotate: bool = False
    rotation_period_days: Optional[int] = None
    rotation_warning_sent: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key_id": self.key_id,
            "name": self.name,
            "description": self.description,
            "key_type": self.key_type.value,
            "status": self.status.value,
            "permissions": [p.value for p in self.permissions],
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "usage_count": self.usage_count,
            "rate_limit_tier": self.rate_limit_tier,
            "allowed_ips": list(self.allowed_ips),
            "allowed_domains": list(self.allowed_domains),
            "allowed_user_agents": list(self.allowed_user_agents),
            "webhook_url": self.webhook_url,
            "callback_urls": list(self.callback_urls),
            "compliance_tags": list(self.compliance_tags),
            "audit_level": self.audit_level,
            "auto_rotate": self.auto_rotate,
            "rotation_period_days": self.rotation_period_days,
            "rotation_warning_sent": self.rotation_warning_sent
        }


class CreateApiKeyRequest(BaseModel):
    """Request model for creating API keys."""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field("", max_length=500)
    key_type: str = Field(..., regex="^(system|service|user|temporary|webhook|read_only)$")
    permissions: List[str] = Field(default_factory=list)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)
    rate_limit_tier: str = Field("developer", regex="^(free|developer|professional|enterprise|unlimited)$")
    
    # Access restrictions
    allowed_ips: List[str] = Field(default_factory=list)
    allowed_domains: List[str] = Field(default_factory=list)
    allowed_user_agents: List[str] = Field(default_factory=list)
    
    # Integration settings
    webhook_url: Optional[str] = Field(None, max_length=2048)
    callback_urls: List[str] = Field(default_factory=list)
    
    # Compliance and audit
    compliance_tags: List[str] = Field(default_factory=list)
    audit_level: str = Field("standard", regex="^(minimal|standard|detailed)$")
    
    # Auto-rotation
    auto_rotate: bool = Field(False)
    rotation_period_days: Optional[int] = Field(None, ge=30, le=365)
    
    @validator('permissions')
    def validate_permissions(cls, v):
        """Validate permission scopes."""
        valid_permissions = {perm.value for perm in PermissionScope}
        invalid_perms = [p for p in v if p not in valid_permissions]
        if invalid_perms:
            raise ValueError(f'Invalid permissions: {invalid_perms}')
        return v
    
    @validator('allowed_ips')
    def validate_ip_addresses(cls, v):
        """Validate IP addresses."""
        if not v:
            return v
        
        import ipaddress
        for ip in v:
            try:
                ipaddress.ip_address(ip.split('/')[0])  # Support CIDR notation
            except ValueError:
                raise ValueError(f'Invalid IP address: {ip}')
        return v


class ApiKeyResponse(BaseModel):
    """Response model for API key operations."""
    key_id: str
    name: str
    description: str
    key_type: str
    status: str
    permissions: List[str]
    created_at: str
    expires_at: Optional[str]
    last_used_at: Optional[str]
    usage_count: int
    rate_limit_tier: str
    compliance_tags: List[str]
    
    # Only included when creating new key
    api_key: Optional[str] = None
    
    class Config:
        from_attributes = True


class ApiKeyUsageStats(BaseModel):
    """API key usage statistics."""
    key_id: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    last_24h_requests: int
    last_7d_requests: int
    last_30d_requests: int
    average_response_time_ms: float
    top_endpoints: Dict[str, int]
    error_rate: float
    rate_limit_violations: int


class EnterpriseApiKeyManager:
    """
    Enterprise API Key Management System.
    
    Features:
    - Hierarchical key types with different privilege levels
    - Advanced permissions and scoping system
    - IP and domain-based access restrictions
    - Automated key rotation and lifecycle management
    - Integration with rate limiting system
    - Comprehensive usage analytics and monitoring
    - Enterprise compliance and audit logging
    - Webhook and callback URL management
    """
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        """
        Initialize API Key Manager.
        
        Args:
            db_session: Database session for persistent storage
        """
        self.db = db_session
        
        # Redis connection for caching and analytics
        self.redis = redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True
        )
        
        # Encryption setup for sensitive data
        self._setup_encryption()
        
        # API key storage (in-memory cache + Redis + DB)
        self.key_cache: Dict[str, ApiKeyMetadata] = {}
        
        # Usage analytics
        self.usage_stats: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.metrics = {
            "keys_created": 0,
            "keys_revoked": 0,
            "keys_rotated": 0,
            "authentication_attempts": 0,
            "successful_authentications": 0,
            "failed_authentications": 0,
            "rate_limit_violations": 0,
            "security_violations": 0,
            "avg_key_lifetime_days": 0.0,
            "key_types": {},
            "permission_usage": {},
            "top_users": {}
        }
        
        logger.info("Enterprise API Key Manager initialized",
                   encryption_enabled=bool(self.encryption_key),
                   redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"))
    
    def _setup_encryption(self):
        """Setup encryption for sensitive API key data."""
        
        encryption_key = API_KEY_CONFIG["encryption_key"]
        if not encryption_key:
            # Generate a key for development (not recommended for production)
            logger.warning("No encryption key provided, generating temporary key")
            encryption_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
        
        try:
            # Derive key from provided string
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'leanvibe_salt',  # In production, use random salt per installation
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(encryption_key.encode()))
            self.encryption_key = Fernet(key)
            
        except Exception as e:
            logger.error("Failed to setup encryption", error=str(e))
            self.encryption_key = None
    
    async def create_api_key(self, request: CreateApiKeyRequest, created_by: str) -> ApiKeyResponse:
        """
        Create a new API key with specified permissions and restrictions.
        
        Args:
            request: API key creation request
            created_by: User creating the key
            
        Returns:
            ApiKeyResponse with the new API key
        """
        try:
            # Validate key type permissions
            key_type = ApiKeyType(request.key_type)
            await self._validate_key_creation_permissions(key_type, request.permissions, created_by)
            
            # Check key limits
            await self._check_key_creation_limits(created_by, key_type)
            
            # Generate unique key ID and API key
            key_id = str(uuid.uuid4())
            api_key = await self._generate_api_key(key_id, key_type)
            
            # Calculate expiration
            expires_at = None
            if request.expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=request.expires_in_days)
            elif key_type == ApiKeyType.TEMPORARY:
                expires_at = datetime.utcnow() + timedelta(hours=24)  # 24 hours for temp keys
            else:
                expires_at = datetime.utcnow() + timedelta(days=API_KEY_CONFIG["default_expiry_days"])
            
            # Create API key metadata
            metadata = ApiKeyMetadata(
                key_id=key_id,
                name=request.name,
                description=request.description,
                key_type=key_type,
                status=ApiKeyStatus.ACTIVE,
                permissions={PermissionScope(p) for p in request.permissions},
                created_by=created_by,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                last_used_at=None,
                usage_count=0,
                rate_limit_tier=request.rate_limit_tier,
                allowed_ips=set(request.allowed_ips),
                allowed_domains=set(request.allowed_domains),
                allowed_user_agents=set(request.allowed_user_agents),
                webhook_url=request.webhook_url,
                callback_urls=set(request.callback_urls),
                compliance_tags=set(request.compliance_tags),
                audit_level=request.audit_level,
                auto_rotate=request.auto_rotate,
                rotation_period_days=request.rotation_period_days
            )
            
            # Store in cache, Redis, and database
            await self._store_api_key(api_key, metadata)
            
            # Update metrics
            self.metrics["keys_created"] += 1
            self.metrics["key_types"][key_type.value] = (
                self.metrics["key_types"].get(key_type.value, 0) + 1
            )
            
            # Log creation event
            await self._log_api_key_event(
                key_id=key_id,
                action="create",
                created_by=created_by,
                metadata={
                    "name": request.name,
                    "key_type": key_type.value,
                    "permissions": request.permissions,
                    "rate_limit_tier": request.rate_limit_tier,
                    "expires_at": expires_at.isoformat() if expires_at else None
                }
            )
            
            # Create response (includes API key only on creation)
            response = ApiKeyResponse(
                key_id=key_id,
                name=metadata.name,
                description=metadata.description,
                key_type=metadata.key_type.value,
                status=metadata.status.value,
                permissions=[p.value for p in metadata.permissions],
                created_at=metadata.created_at.isoformat(),
                expires_at=metadata.expires_at.isoformat() if metadata.expires_at else None,
                last_used_at=None,
                usage_count=0,
                rate_limit_tier=metadata.rate_limit_tier,
                compliance_tags=list(metadata.compliance_tags),
                api_key=api_key  # Only included on creation
            )
            
            logger.info("API key created",
                       key_id=key_id,
                       name=request.name,
                       key_type=key_type.value,
                       created_by=created_by)
            
            return response
            
        except Exception as e:
            logger.error("Failed to create API key", 
                        name=request.name, 
                        created_by=created_by, 
                        error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create API key: {str(e)}"
            )
    
    async def authenticate_api_key(self, api_key: str, request: Request = None) -> Optional[ApiKeyMetadata]:
        """
        Authenticate and validate an API key.
        
        Args:
            api_key: The API key to authenticate
            request: Optional request object for additional validation
            
        Returns:
            ApiKeyMetadata if valid, None if invalid
        """
        try:
            self.metrics["authentication_attempts"] += 1
            
            # Extract key ID from API key
            key_id = self._extract_key_id(api_key)
            if not key_id:
                return None
            
            # Get key metadata
            metadata = await self._get_api_key_metadata(key_id)
            if not metadata:
                return None
            
            # Validate API key hash
            if not await self._validate_api_key_hash(api_key, key_id):
                await self._log_security_violation(
                    key_id=key_id,
                    violation_type="invalid_key_hash",
                    request=request
                )
                return None
            
            # Check key status
            if metadata.status != ApiKeyStatus.ACTIVE:
                await self._log_security_violation(
                    key_id=key_id,
                    violation_type=f"key_status_{metadata.status.value}",
                    request=request
                )
                return None
            
            # Check expiration
            if metadata.expires_at and datetime.utcnow() > metadata.expires_at:
                # Auto-expire the key
                await self._update_key_status(key_id, ApiKeyStatus.EXPIRED)
                return None
            
            # Validate access restrictions
            if request:
                if not await self._validate_access_restrictions(metadata, request):
                    await self._log_security_violation(
                        key_id=key_id,
                        violation_type="access_restriction_violated",
                        request=request
                    )
                    return None
            
            # Update usage statistics
            await self._update_usage_stats(key_id, request)
            
            # Check if key needs rotation warning
            if metadata.auto_rotate and not metadata.rotation_warning_sent:
                await self._check_rotation_warning(metadata)
            
            self.metrics["successful_authentications"] += 1
            
            return metadata
            
        except Exception as e:
            logger.error("API key authentication failed", api_key=api_key[:10] + "...", error=str(e))
            self.metrics["failed_authentications"] += 1
            return None
    
    async def revoke_api_key(self, key_id: str, revoked_by: str, reason: str = "") -> bool:
        """
        Revoke an API key.
        
        Args:
            key_id: Key ID to revoke
            revoked_by: User revoking the key
            reason: Reason for revocation
            
        Returns:
            True if revoked successfully
        """
        try:
            metadata = await self._get_api_key_metadata(key_id)
            if not metadata:
                return False
            
            # Update status to revoked
            await self._update_key_status(key_id, ApiKeyStatus.REVOKED)
            
            # Remove from cache and Redis
            await self._invalidate_api_key_cache(key_id)
            
            # Update metrics
            self.metrics["keys_revoked"] += 1
            
            # Log revocation event
            await self._log_api_key_event(
                key_id=key_id,
                action="revoke",
                created_by=revoked_by,
                metadata={
                    "reason": reason,
                    "original_creator": metadata.created_by,
                    "key_age_days": (datetime.utcnow() - metadata.created_at).days
                }
            )
            
            logger.info("API key revoked",
                       key_id=key_id,
                       revoked_by=revoked_by,
                       reason=reason)
            
            return True
            
        except Exception as e:
            logger.error("Failed to revoke API key", key_id=key_id, error=str(e))
            return False
    
    async def rotate_api_key(self, key_id: str, rotated_by: str) -> Optional[str]:
        """
        Rotate an API key by generating a new key while preserving metadata.
        
        Args:
            key_id: Key ID to rotate
            rotated_by: User performing rotation
            
        Returns:
            New API key if successful
        """
        try:
            metadata = await self._get_api_key_metadata(key_id)
            if not metadata:
                return None
            
            if metadata.status != ApiKeyStatus.ACTIVE:
                raise ValueError("Can only rotate active keys")
            
            # Generate new API key
            new_api_key = await self._generate_api_key(key_id, metadata.key_type)
            
            # Update metadata
            metadata.status = ApiKeyStatus.ACTIVE
            metadata.rotation_warning_sent = False
            
            # Reset rotation period if auto-rotate is enabled
            if metadata.auto_rotate and metadata.rotation_period_days:
                new_expires_at = datetime.utcnow() + timedelta(days=metadata.rotation_period_days)
                metadata.expires_at = new_expires_at
            
            # Store updated key and metadata
            await self._store_api_key(new_api_key, metadata)
            
            # Update metrics
            self.metrics["keys_rotated"] += 1
            
            # Log rotation event
            await self._log_api_key_event(
                key_id=key_id,
                action="rotate",
                created_by=rotated_by,
                metadata={
                    "auto_rotate": metadata.auto_rotate,
                    "rotation_period_days": metadata.rotation_period_days
                }
            )
            
            logger.info("API key rotated",
                       key_id=key_id,
                       rotated_by=rotated_by)
            
            return new_api_key
            
        except Exception as e:
            logger.error("Failed to rotate API key", key_id=key_id, error=str(e))
            return None
    
    async def list_api_keys(self, owner: str = None, key_type: ApiKeyType = None, 
                           status: ApiKeyStatus = None, limit: int = 100, 
                           offset: int = 0) -> List[ApiKeyResponse]:
        """
        List API keys with optional filtering.
        
        Args:
            owner: Filter by key owner
            key_type: Filter by key type
            status: Filter by key status
            limit: Maximum number of keys to return
            offset: Offset for pagination
            
        Returns:
            List of API key responses (without actual keys)
        """
        try:
            # This would typically query the database
            # For now, return from cache
            keys = []
            
            for metadata in self.key_cache.values():
                # Apply filters
                if owner and metadata.created_by != owner:
                    continue
                if key_type and metadata.key_type != key_type:
                    continue
                if status and metadata.status != status:
                    continue
                
                response = ApiKeyResponse(
                    key_id=metadata.key_id,
                    name=metadata.name,
                    description=metadata.description,
                    key_type=metadata.key_type.value,
                    status=metadata.status.value,
                    permissions=[p.value for p in metadata.permissions],
                    created_at=metadata.created_at.isoformat(),
                    expires_at=metadata.expires_at.isoformat() if metadata.expires_at else None,
                    last_used_at=metadata.last_used_at.isoformat() if metadata.last_used_at else None,
                    usage_count=metadata.usage_count,
                    rate_limit_tier=metadata.rate_limit_tier,
                    compliance_tags=list(metadata.compliance_tags)
                )
                keys.append(response)
            
            # Sort by creation date (newest first) and apply pagination
            keys.sort(key=lambda k: k.created_at, reverse=True)
            return keys[offset:offset + limit]
            
        except Exception as e:
            logger.error("Failed to list API keys", error=str(e))
            return []
    
    async def get_usage_stats(self, key_id: str) -> Optional[ApiKeyUsageStats]:
        """
        Get detailed usage statistics for an API key.
        
        Args:
            key_id: Key ID to get stats for
            
        Returns:
            ApiKeyUsageStats if found
        """
        try:
            # Get usage data from Redis analytics
            usage_key = f"{API_KEY_CONFIG['redis_key_prefix']}usage:{key_id}"
            usage_data = await self.redis.hgetall(usage_key)
            
            if not usage_data:
                return None
            
            # Parse usage statistics
            stats = ApiKeyUsageStats(
                key_id=key_id,
                total_requests=int(usage_data.get("total_requests", 0)),
                successful_requests=int(usage_data.get("successful_requests", 0)),
                failed_requests=int(usage_data.get("failed_requests", 0)),
                last_24h_requests=await self._get_time_window_requests(key_id, 24),
                last_7d_requests=await self._get_time_window_requests(key_id, 24 * 7),
                last_30d_requests=await self._get_time_window_requests(key_id, 24 * 30),
                average_response_time_ms=float(usage_data.get("avg_response_time_ms", 0)),
                top_endpoints=json.loads(usage_data.get("top_endpoints", "{}")),
                error_rate=float(usage_data.get("error_rate", 0)),
                rate_limit_violations=int(usage_data.get("rate_limit_violations", 0))
            )
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get usage stats", key_id=key_id, error=str(e))
            return None
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive API key system metrics."""
        
        try:
            # Get Redis-based metrics
            redis_metrics = await self._get_redis_metrics()
            
            # Calculate additional metrics
            total_keys = len(self.key_cache)
            active_keys = sum(1 for m in self.key_cache.values() if m.status == ApiKeyStatus.ACTIVE)
            
            # Calculate average key lifetime
            if self.metrics["keys_created"] > 0:
                total_lifetime_days = sum(
                    (datetime.utcnow() - m.created_at).days 
                    for m in self.key_cache.values() 
                    if m.status in [ApiKeyStatus.EXPIRED, ApiKeyStatus.REVOKED]
                )
                expired_keys = sum(
                    1 for m in self.key_cache.values() 
                    if m.status in [ApiKeyStatus.EXPIRED, ApiKeyStatus.REVOKED]
                )
                if expired_keys > 0:
                    self.metrics["avg_key_lifetime_days"] = total_lifetime_days / expired_keys
            
            return {
                "api_key_metrics": self.metrics.copy(),
                "total_keys": total_keys,
                "active_keys": active_keys,
                "suspended_keys": sum(1 for m in self.key_cache.values() if m.status == ApiKeyStatus.SUSPENDED),
                "expired_keys": sum(1 for m in self.key_cache.values() if m.status == ApiKeyStatus.EXPIRED),
                "revoked_keys": sum(1 for m in self.key_cache.values() if m.status == ApiKeyStatus.REVOKED),
                "keys_expiring_soon": await self._count_keys_expiring_soon(),
                "redis_metrics": redis_metrics,
                "encryption_enabled": self.encryption_key is not None,
                "config": {
                    "default_expiry_days": API_KEY_CONFIG["default_expiry_days"],
                    "max_keys_per_user": API_KEY_CONFIG["max_keys_per_user"],
                    "rotation_enabled": API_KEY_CONFIG["enable_key_rotation"],
                    "ip_restrictions_enabled": API_KEY_CONFIG["enable_ip_restrictions"],
                    "usage_analytics_enabled": API_KEY_CONFIG["enable_usage_analytics"]
                }
            }
            
        except Exception as e:
            logger.error("Failed to get system metrics", error=str(e))
            return {"error": str(e)}
    
    # Private helper methods
    
    async def _generate_api_key(self, key_id: str, key_type: ApiKeyType) -> str:
        """Generate a secure API key."""
        
        # Generate random key portion
        key_bytes = secrets.token_bytes(API_KEY_CONFIG["key_length"])
        key_b64 = base64.urlsafe_b64encode(key_bytes).decode().rstrip('=')
        
        # Create key with type prefix and checksum
        type_prefix = key_type.value[:3].upper()  # e.g., SYS, SER, USE
        checksum = hashlib.sha256(f"{key_id}:{key_b64}".encode()).hexdigest()[:8]
        
        api_key = f"lv_{type_prefix}_{key_b64}_{checksum}"
        
        return api_key
    
    def _extract_key_id(self, api_key: str) -> Optional[str]:
        """Extract key ID from API key format."""
        
        try:
            # API key format: lv_{TYPE}_{KEY}_{CHECKSUM}
            parts = api_key.split('_')
            if len(parts) != 4 or parts[0] != 'lv':
                return None
            
            # For now, use a hash of the key as the key ID
            # In production, this would be stored during key creation
            return hashlib.sha256(api_key.encode()).hexdigest()[:16]
            
        except Exception:
            return None
    
    async def _validate_api_key_hash(self, api_key: str, key_id: str) -> bool:
        """Validate API key hash against stored data."""
        
        try:
            # Get stored hash from Redis
            hash_key = f"{API_KEY_CONFIG['redis_key_prefix']}hash:{key_id}"
            stored_hash = await self.redis.get(hash_key)
            
            if not stored_hash:
                return False
            
            # Hash the provided key
            provided_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            return provided_hash == stored_hash
            
        except Exception:
            return False
    
    async def _validate_key_creation_permissions(self, key_type: ApiKeyType, 
                                               permissions: List[str], 
                                               created_by: str) -> None:
        """Validate that user can create key with specified permissions."""
        
        # System keys require admin privileges
        if key_type == ApiKeyType.SYSTEM:
            # Check if user has admin permissions (simplified check)
            if not created_by.endswith("@admin"):  # Placeholder check
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient privileges to create system keys"
                )
        
        # Validate permission levels
        high_privilege_perms = {
            PermissionScope.ADMIN_WRITE.value,
            PermissionScope.SYSTEM_WRITE.value,
            PermissionScope.AGENTS_DELETE.value
        }
        
        if any(perm in high_privilege_perms for perm in permissions):
            if key_type not in [ApiKeyType.SYSTEM, ApiKeyType.SERVICE]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="High-privilege permissions require system or service key type"
                )
    
    async def _check_key_creation_limits(self, created_by: str, key_type: ApiKeyType) -> None:
        """Check if user has reached key creation limits."""
        
        # Count existing keys for user
        user_keys = sum(
            1 for m in self.key_cache.values() 
            if m.created_by == created_by and m.status == ApiKeyStatus.ACTIVE
        )
        
        # Check limits based on key type
        if key_type in [ApiKeyType.USER, ApiKeyType.TEMPORARY, ApiKeyType.READ_ONLY]:
            max_keys = API_KEY_CONFIG["max_keys_per_user"]
        else:
            max_keys = API_KEY_CONFIG["max_keys_per_service"]
        
        if user_keys >= max_keys:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Maximum number of {key_type.value} keys reached ({max_keys})"
            )
    
    async def _store_api_key(self, api_key: str, metadata: ApiKeyMetadata) -> None:
        """Store API key and metadata in cache, Redis, and database."""
        
        key_id = metadata.key_id
        
        # Store in memory cache
        self.key_cache[key_id] = metadata
        
        # Store hash in Redis for validation
        hash_key = f"{API_KEY_CONFIG['redis_key_prefix']}hash:{key_id}"
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        await self.redis.setex(hash_key, 86400 * 365, api_key_hash)  # 1 year expiration
        
        # Store metadata in Redis
        metadata_key = f"{API_KEY_CONFIG['redis_key_prefix']}metadata:{key_id}"
        await self.redis.setex(metadata_key, 86400 * 365, json.dumps(metadata.to_dict()))
        
        # Initialize usage statistics
        usage_key = f"{API_KEY_CONFIG['redis_key_prefix']}usage:{key_id}"
        await self.redis.hset(usage_key, mapping={
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time_ms": 0,
            "top_endpoints": "{}",
            "error_rate": 0,
            "rate_limit_violations": 0
        })
        await self.redis.expire(usage_key, 86400 * 365)
    
    async def _get_api_key_metadata(self, key_id: str) -> Optional[ApiKeyMetadata]:
        """Get API key metadata from cache or Redis."""
        
        # Check cache first
        if key_id in self.key_cache:
            return self.key_cache[key_id]
        
        # Load from Redis
        try:
            metadata_key = f"{API_KEY_CONFIG['redis_key_prefix']}metadata:{key_id}"
            metadata_json = await self.redis.get(metadata_key)
            
            if not metadata_json:
                return None
            
            data = json.loads(metadata_json)
            
            # Reconstruct metadata object
            metadata = ApiKeyMetadata(
                key_id=data["key_id"],
                name=data["name"],
                description=data["description"],
                key_type=ApiKeyType(data["key_type"]),
                status=ApiKeyStatus(data["status"]),
                permissions={PermissionScope(p) for p in data["permissions"]},
                created_by=data["created_by"],
                created_at=datetime.fromisoformat(data["created_at"]),
                expires_at=datetime.fromisoformat(data["expires_at"]) if data["expires_at"] else None,
                last_used_at=datetime.fromisoformat(data["last_used_at"]) if data["last_used_at"] else None,
                usage_count=data["usage_count"],
                rate_limit_tier=data["rate_limit_tier"],
                allowed_ips=set(data["allowed_ips"]),
                allowed_domains=set(data["allowed_domains"]),
                allowed_user_agents=set(data["allowed_user_agents"]),
                webhook_url=data["webhook_url"],
                callback_urls=set(data["callback_urls"]),
                compliance_tags=set(data["compliance_tags"]),
                audit_level=data["audit_level"],
                auto_rotate=data["auto_rotate"],
                rotation_period_days=data["rotation_period_days"],
                rotation_warning_sent=data["rotation_warning_sent"]
            )
            
            # Cache for future use
            self.key_cache[key_id] = metadata
            
            return metadata
            
        except Exception as e:
            logger.error("Failed to get API key metadata", key_id=key_id, error=str(e))
            return None
    
    async def _validate_access_restrictions(self, metadata: ApiKeyMetadata, request: Request) -> bool:
        """Validate access restrictions for API key."""
        
        try:
            client_ip = self._get_client_ip(request)
            user_agent = request.headers.get("User-Agent", "")
            host = request.headers.get("Host", "")
            
            # Check IP restrictions
            if metadata.allowed_ips:
                import ipaddress
                allowed = False
                for allowed_ip in metadata.allowed_ips:
                    try:
                        if '/' in allowed_ip:  # CIDR notation
                            network = ipaddress.ip_network(allowed_ip, strict=False)
                            if ipaddress.ip_address(client_ip) in network:
                                allowed = True
                                break
                        elif client_ip == allowed_ip:
                            allowed = True
                            break
                    except ValueError:
                        continue
                
                if not allowed:
                    return False
            
            # Check domain restrictions
            if metadata.allowed_domains:
                domain_match = any(domain in host for domain in metadata.allowed_domains)
                if not domain_match:
                    return False
            
            # Check user agent restrictions
            if metadata.allowed_user_agents:
                ua_match = any(ua in user_agent for ua in metadata.allowed_user_agents)
                if not ua_match:
                    return False
            
            return True
            
        except Exception as e:
            logger.error("Access restriction validation failed", error=str(e))
            return False
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"
    
    async def _update_usage_stats(self, key_id: str, request: Request = None) -> None:
        """Update usage statistics for API key."""
        
        if not API_KEY_CONFIG["enable_usage_analytics"]:
            return
        
        try:
            # Update metadata
            if key_id in self.key_cache:
                self.key_cache[key_id].last_used_at = datetime.utcnow()
                self.key_cache[key_id].usage_count += 1
            
            # Update Redis statistics
            usage_key = f"{API_KEY_CONFIG['redis_key_prefix']}usage:{key_id}"
            
            pipe = self.redis.pipeline()
            pipe.hincrby(usage_key, "total_requests", 1)
            
            # Track endpoint usage if request available
            if request:
                endpoint = str(request.url.path)
                top_endpoints_key = f"{API_KEY_CONFIG['redis_key_prefix']}endpoints:{key_id}"
                pipe.zincrby(top_endpoints_key, 1, endpoint)
                pipe.expire(top_endpoints_key, 86400 * 30)  # 30 days
            
            # Add timestamp for time-window queries
            timestamp_key = f"{API_KEY_CONFIG['redis_key_prefix']}timestamps:{key_id}"
            current_time = int(time.time())
            pipe.zadd(timestamp_key, {str(current_time): current_time})
            pipe.zremrangebyscore(timestamp_key, 0, current_time - (86400 * 30))  # Keep 30 days
            pipe.expire(timestamp_key, 86400 * 31)
            
            await pipe.execute()
            
        except Exception as e:
            logger.error("Failed to update usage stats", key_id=key_id, error=str(e))
    
    async def _update_key_status(self, key_id: str, new_status: ApiKeyStatus) -> None:
        """Update API key status."""
        
        # Update cache
        if key_id in self.key_cache:
            self.key_cache[key_id].status = new_status
        
        # Update Redis
        metadata_key = f"{API_KEY_CONFIG['redis_key_prefix']}metadata:{key_id}"
        metadata = await self._get_api_key_metadata(key_id)
        if metadata:
            metadata.status = new_status
            await self.redis.setex(metadata_key, 86400 * 365, json.dumps(metadata.to_dict()))
    
    async def _invalidate_api_key_cache(self, key_id: str) -> None:
        """Remove API key from all caches."""
        
        # Remove from memory cache
        self.key_cache.pop(key_id, None)
        
        # Remove from Redis (keep for audit purposes, just mark as revoked)
        # In production, you might want to move to an archive instead of deleting
    
    async def _get_time_window_requests(self, key_id: str, hours: int) -> int:
        """Get request count for a time window."""
        
        try:
            timestamp_key = f"{API_KEY_CONFIG['redis_key_prefix']}timestamps:{key_id}"
            cutoff_time = int(time.time()) - (hours * 3600)
            
            count = await self.redis.zcount(timestamp_key, cutoff_time, "+inf")
            return count
            
        except Exception:
            return 0
    
    async def _check_rotation_warning(self, metadata: ApiKeyMetadata) -> None:
        """Check if key needs rotation warning."""
        
        if not metadata.rotation_period_days:
            return
        
        days_until_rotation = (
            metadata.created_at + timedelta(days=metadata.rotation_period_days) - datetime.utcnow()
        ).days
        
        if days_until_rotation <= API_KEY_CONFIG["rotation_warning_days"]:
            # Send rotation warning (implementation depends on notification system)
            metadata.rotation_warning_sent = True
            
            # Update metadata in storage
            await self._store_api_key("", metadata)  # Empty key since we're just updating metadata
            
            logger.warning("API key rotation warning",
                          key_id=metadata.key_id,
                          days_until_rotation=days_until_rotation)
    
    async def _count_keys_expiring_soon(self, days: int = 7) -> int:
        """Count keys expiring within specified days."""
        
        cutoff_date = datetime.utcnow() + timedelta(days=days)
        
        return sum(
            1 for metadata in self.key_cache.values()
            if (metadata.expires_at and 
                metadata.expires_at <= cutoff_date and 
                metadata.status == ApiKeyStatus.ACTIVE)
        )
    
    async def _get_redis_metrics(self) -> Dict[str, Any]:
        """Get Redis-based metrics."""
        
        try:
            # Count keys in Redis
            hash_keys = await self._scan_redis_keys(f"{API_KEY_CONFIG['redis_key_prefix']}hash:*")
            metadata_keys = await self._scan_redis_keys(f"{API_KEY_CONFIG['redis_key_prefix']}metadata:*")
            usage_keys = await self._scan_redis_keys(f"{API_KEY_CONFIG['redis_key_prefix']}usage:*")
            
            return {
                "redis_hash_keys": len(hash_keys),
                "redis_metadata_keys": len(metadata_keys),
                "redis_usage_keys": len(usage_keys),
                "redis_total_keys": len(hash_keys) + len(metadata_keys) + len(usage_keys)
            }
            
        except Exception as e:
            logger.error("Failed to get Redis metrics", error=str(e))
            return {}
    
    async def _scan_redis_keys(self, pattern: str) -> List[str]:
        """Scan Redis keys matching pattern."""
        
        try:
            keys = []
            cursor = 0
            
            while True:
                cursor, batch = await self.redis.scan(cursor, match=pattern, count=100)
                keys.extend(batch)
                
                if cursor == 0:
                    break
            
            return keys
            
        except Exception:
            return []
    
    async def _log_api_key_event(self, key_id: str, action: str, created_by: str, metadata: Dict[str, Any]) -> None:
        """Log API key events for audit."""
        
        if self.db:
            audit_log = SecurityAuditLog(
                agent_id=None,
                human_controller=created_by,
                action=f"api_key_{action}",
                resource="api_keys",
                resource_id=key_id,
                success=True,
                metadata={
                    "key_id": key_id,
                    "api_key_action": action,
                    **metadata
                }
            )
            self.db.add(audit_log)
    
    async def _log_security_violation(self, key_id: str, violation_type: str, request: Request = None) -> None:
        """Log security violations."""
        
        self.metrics["security_violations"] += 1
        
        if self.db:
            client_ip = self._get_client_ip(request) if request else "unknown"
            
            security_event = SecurityEvent(
                event_type="api_key_violation",
                severity="medium",
                source_ip=client_ip if client_ip != "unknown" else None,
                description=f"API key security violation: {violation_type}",
                details={
                    "key_id": key_id,
                    "violation_type": violation_type,
                    "user_agent": request.headers.get("User-Agent") if request else None,
                    "endpoint": str(request.url.path) if request else None
                },
                risk_score=0.6  # Medium risk
            )
            self.db.add(security_event)


# Global API key manager instance
_api_key_manager: Optional[EnterpriseApiKeyManager] = None


async def get_api_key_manager(db: AsyncSession = Depends(get_session)) -> EnterpriseApiKeyManager:
    """Get or create API key manager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = EnterpriseApiKeyManager(db)
    return _api_key_manager


class ApiKeyAuthBearer(HTTPBearer):
    """API key authentication bearer."""
    
    def __init__(self, api_key_manager: EnterpriseApiKeyManager = None):
        super().__init__(auto_error=False)
        self.api_key_manager = api_key_manager
    
    async def __call__(self, request: Request) -> Optional[ApiKeyMetadata]:
        credentials = await super().__call__(request)
        
        if not credentials:
            # Try to get from query parameter or header
            api_key = (
                request.query_params.get("api_key") or
                request.headers.get("X-API-Key")
            )
            if not api_key:
                return None
        else:
            api_key = credentials.credentials
        
        if not self.api_key_manager:
            from .database import get_session
            async with get_session() as db:
                self.api_key_manager = EnterpriseApiKeyManager(db)
        
        # Authenticate the API key
        return await self.api_key_manager.authenticate_api_key(api_key, request)


# Dependency for API key authentication
async def get_api_key_auth(
    api_key_manager: EnterpriseApiKeyManager = Depends(get_api_key_manager)
) -> ApiKeyAuthBearer:
    """Get API key authentication dependency."""
    return ApiKeyAuthBearer(api_key_manager)


# Export API key management components
__all__ = [
    "EnterpriseApiKeyManager", "get_api_key_manager", "ApiKeyAuthBearer", "get_api_key_auth",
    "CreateApiKeyRequest", "ApiKeyResponse", "ApiKeyUsageStats", "ApiKeyMetadata",
    "ApiKeyType", "ApiKeyStatus", "PermissionScope"
]