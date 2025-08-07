"""
Enterprise Secrets Management System for LeanVibe Agent Hive 2.0

Production-grade secrets management with encryption, rotation, audit trails,
and integration with external secret stores (AWS Secrets Manager, HashiCorp Vault).

CRITICAL COMPONENT: Secure handling of API keys, passwords, and sensitive configuration.
"""

import asyncio
import base64
import json
import os
import secrets
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
import hashlib
import hmac

import structlog
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pydantic import BaseModel, Field
import aiofiles
import asyncio

from .config import settings
from .redis import get_redis

logger = structlog.get_logger()


class SecretType(Enum):
    """Types of secrets managed by the system."""
    API_KEY = "api_key"
    DATABASE_PASSWORD = "database_password"
    JWT_SECRET = "jwt_secret"
    ENCRYPTION_KEY = "encryption_key"
    WEBHOOK_SECRET = "webhook_secret"
    OAUTH_CLIENT_SECRET = "oauth_client_secret"
    CERTIFICATE = "certificate"
    PRIVATE_KEY = "private_key"
    TOKEN = "token"
    CONFIGURATION = "configuration"


class SecretStatus(Enum):
    """Secret lifecycle status."""
    ACTIVE = "active"
    PENDING_ROTATION = "pending_rotation"
    ROTATING = "rotating"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"
    EXPIRED = "expired"


class RotationPolicy(Enum):
    """Secret rotation policies."""
    MANUAL = "manual"
    TIME_BASED = "time_based"
    USAGE_BASED = "usage_based"
    ON_DEMAND = "on_demand"


class SecretMetadata(BaseModel):
    """Metadata for stored secrets."""
    secret_id: str
    name: str
    secret_type: SecretType
    status: SecretStatus = SecretStatus.ACTIVE
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    rotation_policy: RotationPolicy = RotationPolicy.MANUAL
    rotation_interval_days: Optional[int] = None
    last_rotation: Optional[datetime] = None
    usage_count: int = 0
    max_usage_count: Optional[int] = None
    tags: Dict[str, str] = {}
    owner_id: str
    access_permissions: List[str] = []
    audit_trail: List[Dict[str, Any]] = []


class SecretRequest(BaseModel):
    """Request to create or update a secret."""
    name: str = Field(..., min_length=1, max_length=100)
    secret_type: SecretType
    value: str = Field(..., min_length=1)
    expires_days: Optional[int] = Field(None, ge=1, le=3650)  # Max 10 years
    rotation_policy: RotationPolicy = RotationPolicy.MANUAL
    rotation_interval_days: Optional[int] = Field(None, ge=1, le=365)
    max_usage_count: Optional[int] = Field(None, ge=1)
    tags: Dict[str, str] = {}
    access_permissions: List[str] = []


class EnterpriseSecretsManager:
    """
    Enterprise-grade secrets management system.
    
    Provides secure storage, encryption, rotation, and audit capabilities
    for all sensitive data in the autonomous development platform.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.redis = None
        
        # Initialize encryption
        self.cipher_suite = self._initialize_encryption()
        self.key_derivation_salt = b'leanvibe_secrets_salt_v2'
        
        # Storage backends
        self.storage_backends = {}
        self._initialize_storage_backends()
        
        # Rotation scheduler
        self.rotation_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("Enterprise secrets manager initialized")
    
    async def initialize(self):
        """Initialize async components."""
        self.redis = get_redis()
        await self._start_rotation_scheduler()
        logger.info("Secrets manager async components initialized")
    
    def _initialize_encryption(self) -> Fernet:
        """Initialize encryption for secrets storage."""
        # Derive key from settings
        password = settings.SECRET_KEY.encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.key_derivation_salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return Fernet(key)
    
    def _initialize_storage_backends(self):
        """Initialize different storage backends."""
        # Local file storage (default)
        self.storage_backends['local'] = LocalFileSecretStorage()
        
        # Redis storage for temporary secrets
        self.storage_backends['redis'] = RedisSecretStorage()
        
        # Would add AWS Secrets Manager, HashiCorp Vault, etc. in production
        if self.config.get('aws_secrets_manager'):
            self.storage_backends['aws'] = AWSSecretsManagerBackend(self.config['aws_secrets_manager'])
        
        if self.config.get('vault'):
            self.storage_backends['vault'] = VaultSecretBackend(self.config['vault'])
    
    async def create_secret(self, secret_request: SecretRequest, owner_id: str) -> SecretMetadata:
        """Create a new secret with encryption and metadata."""
        
        secret_id = f"secret_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()
        
        # Create metadata
        metadata = SecretMetadata(
            secret_id=secret_id,
            name=secret_request.name,
            secret_type=secret_request.secret_type,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(days=secret_request.expires_days) if secret_request.expires_days else None,
            rotation_policy=secret_request.rotation_policy,
            rotation_interval_days=secret_request.rotation_interval_days,
            max_usage_count=secret_request.max_usage_count,
            tags=secret_request.tags,
            owner_id=owner_id,
            access_permissions=secret_request.access_permissions,
            audit_trail=[{
                "action": "created",
                "timestamp": now.isoformat(),
                "user_id": owner_id
            }]
        )
        
        # Encrypt secret value
        encrypted_value = self._encrypt_secret(secret_request.value)
        
        # Store secret and metadata
        backend = self._get_storage_backend(secret_request.secret_type)
        await backend.store_secret(secret_id, encrypted_value, metadata)
        
        # Schedule rotation if needed
        if secret_request.rotation_policy == RotationPolicy.TIME_BASED and secret_request.rotation_interval_days:
            await self._schedule_rotation(secret_id, secret_request.rotation_interval_days)
        
        logger.info("Secret created",
                   secret_id=secret_id,
                   name=secret_request.name,
                   type=secret_request.secret_type.value,
                   owner_id=owner_id)
        
        return metadata
    
    async def get_secret(self, secret_id: str, user_id: str) -> Optional[str]:
        """Retrieve and decrypt a secret value."""
        
        try:
            # Get metadata and verify permissions
            metadata = await self.get_secret_metadata(secret_id)
            if not metadata:
                logger.warning("Secret not found", secret_id=secret_id)
                return None
            
            # Check permissions
            if not self._check_access_permission(metadata, user_id):
                logger.warning("Access denied to secret",
                             secret_id=secret_id,
                             user_id=user_id)
                return None
            
            # Check if secret is expired
            if metadata.expires_at and datetime.utcnow() > metadata.expires_at:
                logger.warning("Secret expired",
                             secret_id=secret_id,
                             expired_at=metadata.expires_at)
                await self._update_secret_status(secret_id, SecretStatus.EXPIRED)
                return None
            
            # Check usage limits
            if metadata.max_usage_count and metadata.usage_count >= metadata.max_usage_count:
                logger.warning("Secret usage limit exceeded",
                             secret_id=secret_id,
                             usage_count=metadata.usage_count)
                return None
            
            # Retrieve encrypted secret
            backend = self._get_storage_backend(metadata.secret_type)
            encrypted_value = await backend.get_secret(secret_id)
            
            if not encrypted_value:
                logger.error("Secret value not found in storage", secret_id=secret_id)
                return None
            
            # Decrypt secret
            decrypted_value = self._decrypt_secret(encrypted_value)
            
            # Update usage count and audit trail
            await self._record_secret_access(secret_id, user_id)
            
            logger.info("Secret accessed",
                       secret_id=secret_id,
                       user_id=user_id,
                       usage_count=metadata.usage_count + 1)
            
            return decrypted_value
            
        except Exception as e:
            logger.error("Failed to retrieve secret",
                        secret_id=secret_id,
                        error=str(e))
            return None
    
    async def update_secret(self, secret_id: str, new_value: str, user_id: str) -> bool:
        """Update a secret's value."""
        
        try:
            metadata = await self.get_secret_metadata(secret_id)
            if not metadata:
                return False
            
            # Check permissions (owner or admin)
            if metadata.owner_id != user_id and 'admin' not in (await self._get_user_permissions(user_id)):
                logger.warning("Insufficient permissions to update secret",
                             secret_id=secret_id,
                             user_id=user_id)
                return False
            
            # Encrypt new value
            encrypted_value = self._encrypt_secret(new_value)
            
            # Update in storage
            backend = self._get_storage_backend(metadata.secret_type)
            await backend.update_secret(secret_id, encrypted_value)
            
            # Update metadata
            metadata.updated_at = datetime.utcnow()
            metadata.audit_trail.append({
                "action": "updated",
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id
            })
            
            await backend.update_metadata(secret_id, metadata)
            
            logger.info("Secret updated",
                       secret_id=secret_id,
                       user_id=user_id)
            
            return True
            
        except Exception as e:
            logger.error("Failed to update secret",
                        secret_id=secret_id,
                        error=str(e))
            return False
    
    async def delete_secret(self, secret_id: str, user_id: str) -> bool:
        """Delete a secret (soft delete with audit trail)."""
        
        try:
            metadata = await self.get_secret_metadata(secret_id)
            if not metadata:
                return False
            
            # Check permissions
            if metadata.owner_id != user_id and 'admin' not in (await self._get_user_permissions(user_id)):
                return False
            
            # Soft delete - mark as revoked
            metadata.status = SecretStatus.REVOKED
            metadata.updated_at = datetime.utcnow()
            metadata.audit_trail.append({
                "action": "deleted",
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id
            })
            
            backend = self._get_storage_backend(metadata.secret_type)
            await backend.update_metadata(secret_id, metadata)
            
            # Cancel any scheduled rotation
            if secret_id in self.rotation_tasks:
                self.rotation_tasks[secret_id].cancel()
                del self.rotation_tasks[secret_id]
            
            logger.info("Secret deleted",
                       secret_id=secret_id,
                       user_id=user_id)
            
            return True
            
        except Exception as e:
            logger.error("Failed to delete secret",
                        secret_id=secret_id,
                        error=str(e))
            return False
    
    async def rotate_secret(self, secret_id: str, rotation_callback: Optional[Callable] = None) -> bool:
        """Rotate a secret with optional callback for custom rotation logic."""
        
        try:
            metadata = await self.get_secret_metadata(secret_id)
            if not metadata:
                return False
            
            # Mark as rotating
            await self._update_secret_status(secret_id, SecretStatus.ROTATING)
            
            # Generate new secret value
            new_value = None
            if rotation_callback:
                new_value = await rotation_callback(metadata)
            else:
                new_value = self._generate_default_secret(metadata.secret_type)
            
            if not new_value:
                logger.error("Failed to generate new secret value", secret_id=secret_id)
                return False
            
            # Update with new value
            encrypted_value = self._encrypt_secret(new_value)
            backend = self._get_storage_backend(metadata.secret_type)
            await backend.update_secret(secret_id, encrypted_value)
            
            # Update metadata
            metadata.status = SecretStatus.ACTIVE
            metadata.updated_at = datetime.utcnow()
            metadata.last_rotation = datetime.utcnow()
            metadata.audit_trail.append({
                "action": "rotated",
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": "system"
            })
            
            await backend.update_metadata(secret_id, metadata)
            
            # Schedule next rotation if needed
            if metadata.rotation_policy == RotationPolicy.TIME_BASED and metadata.rotation_interval_days:
                await self._schedule_rotation(secret_id, metadata.rotation_interval_days)
            
            logger.info("Secret rotated successfully", secret_id=secret_id)
            return True
            
        except Exception as e:
            logger.error("Secret rotation failed",
                        secret_id=secret_id,
                        error=str(e))
            # Reset status on failure
            await self._update_secret_status(secret_id, SecretStatus.ACTIVE)
            return False
    
    async def list_secrets(self, owner_id: str, filters: Optional[Dict[str, Any]] = None) -> List[SecretMetadata]:
        """List secrets owned by or accessible to a user."""
        
        try:
            all_secrets = []
            
            # Query all storage backends
            for backend_name, backend in self.storage_backends.items():
                secrets = await backend.list_secrets(owner_id, filters)
                all_secrets.extend(secrets)
            
            # Filter by permissions
            accessible_secrets = []
            for secret in all_secrets:
                if self._check_access_permission(secret, owner_id):
                    accessible_secrets.append(secret)
            
            return accessible_secrets
            
        except Exception as e:
            logger.error("Failed to list secrets",
                        owner_id=owner_id,
                        error=str(e))
            return []
    
    async def get_secret_metadata(self, secret_id: str) -> Optional[SecretMetadata]:
        """Get secret metadata without the actual secret value."""
        
        try:
            # Try each storage backend
            for backend in self.storage_backends.values():
                metadata = await backend.get_metadata(secret_id)
                if metadata:
                    return metadata
            
            return None
            
        except Exception as e:
            logger.error("Failed to get secret metadata",
                        secret_id=secret_id,
                        error=str(e))
            return None
    
    # Encryption/Decryption Methods
    def _encrypt_secret(self, value: str) -> str:
        """Encrypt a secret value."""
        encrypted_bytes = self.cipher_suite.encrypt(value.encode())
        return base64.urlsafe_b64encode(encrypted_bytes).decode()
    
    def _decrypt_secret(self, encrypted_value: str) -> str:
        """Decrypt a secret value."""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
        decrypted_bytes = self.cipher_suite.decrypt(encrypted_bytes)
        return decrypted_bytes.decode()
    
    # Helper Methods
    def _get_storage_backend(self, secret_type: SecretType):
        """Get appropriate storage backend for secret type."""
        # Route different secret types to different backends
        if secret_type in [SecretType.JWT_SECRET, SecretType.WEBHOOK_SECRET]:
            return self.storage_backends.get('redis', self.storage_backends['local'])
        elif secret_type in [SecretType.CERTIFICATE, SecretType.PRIVATE_KEY]:
            return self.storage_backends.get('vault', self.storage_backends['local'])
        else:
            return self.storage_backends['local']
    
    def _check_access_permission(self, metadata: SecretMetadata, user_id: str) -> bool:
        """Check if user has access to a secret."""
        # Owner always has access
        if metadata.owner_id == user_id:
            return True
        
        # Check explicit permissions
        if user_id in metadata.access_permissions:
            return True
        
        # Check role-based permissions (would integrate with RBAC system)
        # For now, simple check
        return False
    
    async def _get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions (would integrate with auth system)."""
        # Placeholder - would integrate with actual auth system
        if user_id == "admin-001":
            return ["admin"]
        return []
    
    def _generate_default_secret(self, secret_type: SecretType) -> str:
        """Generate default secret value for rotation."""
        if secret_type == SecretType.API_KEY:
            return f"lv_{secrets.token_urlsafe(32)}"
        elif secret_type == SecretType.JWT_SECRET:
            return secrets.token_urlsafe(64)
        elif secret_type == SecretType.WEBHOOK_SECRET:
            return secrets.token_hex(32)
        elif secret_type == SecretType.DATABASE_PASSWORD:
            # Generate secure password
            chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
            return ''.join(secrets.choice(chars) for _ in range(32))
        else:
            return secrets.token_urlsafe(32)
    
    async def _update_secret_status(self, secret_id: str, status: SecretStatus):
        """Update secret status."""
        metadata = await self.get_secret_metadata(secret_id)
        if metadata:
            metadata.status = status
            metadata.updated_at = datetime.utcnow()
            backend = self._get_storage_backend(metadata.secret_type)
            await backend.update_metadata(secret_id, metadata)
    
    async def _record_secret_access(self, secret_id: str, user_id: str):
        """Record secret access for audit and usage tracking."""
        metadata = await self.get_secret_metadata(secret_id)
        if metadata:
            metadata.usage_count += 1
            metadata.audit_trail.append({
                "action": "accessed",
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id
            })
            backend = self._get_storage_backend(metadata.secret_type)
            await backend.update_metadata(secret_id, metadata)
    
    async def _schedule_rotation(self, secret_id: str, interval_days: int):
        """Schedule automatic secret rotation."""
        
        async def rotation_task():
            await asyncio.sleep(interval_days * 24 * 3600)  # Convert days to seconds
            await self.rotate_secret(secret_id)
        
        # Cancel existing task if any
        if secret_id in self.rotation_tasks:
            self.rotation_tasks[secret_id].cancel()
        
        # Schedule new task
        self.rotation_tasks[secret_id] = asyncio.create_task(rotation_task())
        logger.info("Rotation scheduled",
                   secret_id=secret_id,
                   interval_days=interval_days)
    
    async def _start_rotation_scheduler(self):
        """Start the rotation scheduler for existing secrets."""
        try:
            # Get all secrets that need rotation
            for backend in self.storage_backends.values():
                secrets = await backend.list_all_secrets()
                for secret in secrets:
                    if (secret.rotation_policy == RotationPolicy.TIME_BASED and 
                        secret.rotation_interval_days and 
                        secret.status == SecretStatus.ACTIVE):
                        
                        # Calculate next rotation time
                        if secret.last_rotation:
                            next_rotation = secret.last_rotation + timedelta(days=secret.rotation_interval_days)
                        else:
                            next_rotation = secret.created_at + timedelta(days=secret.rotation_interval_days)
                        
                        # Schedule if needed
                        if next_rotation > datetime.utcnow():
                            remaining_seconds = (next_rotation - datetime.utcnow()).total_seconds()
                            
                            async def delayed_rotation_task(sid=secret.secret_id, delay=remaining_seconds):
                                await asyncio.sleep(delay)
                                await self.rotate_secret(sid)
                            
                            self.rotation_tasks[secret.secret_id] = asyncio.create_task(
                                delayed_rotation_task()
                            )
            
            logger.info("Rotation scheduler started",
                       scheduled_rotations=len(self.rotation_tasks))
        except Exception as e:
            logger.error("Failed to start rotation scheduler", error=str(e))


# Storage Backend Implementations
class SecretStorageBackend:
    """Base class for secret storage backends."""
    
    async def store_secret(self, secret_id: str, encrypted_value: str, metadata: SecretMetadata):
        raise NotImplementedError
    
    async def get_secret(self, secret_id: str) -> Optional[str]:
        raise NotImplementedError
    
    async def update_secret(self, secret_id: str, encrypted_value: str):
        raise NotImplementedError
    
    async def get_metadata(self, secret_id: str) -> Optional[SecretMetadata]:
        raise NotImplementedError
    
    async def update_metadata(self, secret_id: str, metadata: SecretMetadata):
        raise NotImplementedError
    
    async def list_secrets(self, owner_id: str, filters: Optional[Dict[str, Any]] = None) -> List[SecretMetadata]:
        raise NotImplementedError
    
    async def list_all_secrets(self) -> List[SecretMetadata]:
        raise NotImplementedError


class LocalFileSecretStorage(SecretStorageBackend):
    """Local file system storage for secrets."""
    
    def __init__(self):
        self.storage_dir = Path(settings.WORKSPACE_DIR) / "secrets"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    async def store_secret(self, secret_id: str, encrypted_value: str, metadata: SecretMetadata):
        """Store secret in local file."""
        secret_file = self.storage_dir / f"{secret_id}.secret"
        metadata_file = self.storage_dir / f"{secret_id}.metadata.json"
        
        # Store encrypted secret
        async with aiofiles.open(secret_file, 'w') as f:
            await f.write(encrypted_value)
        
        # Store metadata
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(metadata.json())
    
    async def get_secret(self, secret_id: str) -> Optional[str]:
        """Retrieve secret from local file."""
        secret_file = self.storage_dir / f"{secret_id}.secret"
        if not secret_file.exists():
            return None
        
        async with aiofiles.open(secret_file, 'r') as f:
            return await f.read()
    
    async def update_secret(self, secret_id: str, encrypted_value: str):
        """Update secret in local file."""
        secret_file = self.storage_dir / f"{secret_id}.secret"
        async with aiofiles.open(secret_file, 'w') as f:
            await f.write(encrypted_value)
    
    async def get_metadata(self, secret_id: str) -> Optional[SecretMetadata]:
        """Get secret metadata from local file."""
        metadata_file = self.storage_dir / f"{secret_id}.metadata.json"
        if not metadata_file.exists():
            return None
        
        async with aiofiles.open(metadata_file, 'r') as f:
            content = await f.read()
            return SecretMetadata.parse_raw(content)
    
    async def update_metadata(self, secret_id: str, metadata: SecretMetadata):
        """Update secret metadata in local file."""
        metadata_file = self.storage_dir / f"{secret_id}.metadata.json"
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(metadata.json())
    
    async def list_secrets(self, owner_id: str, filters: Optional[Dict[str, Any]] = None) -> List[SecretMetadata]:
        """List secrets from local files."""
        secrets = []
        
        for metadata_file in self.storage_dir.glob("*.metadata.json"):
            try:
                async with aiofiles.open(metadata_file, 'r') as f:
                    content = await f.read()
                    metadata = SecretMetadata.parse_raw(content)
                    
                    # Filter by owner
                    if metadata.owner_id == owner_id or owner_id == "admin-001":
                        # Apply additional filters if provided
                        if self._matches_filters(metadata, filters):
                            secrets.append(metadata)
            except Exception as e:
                logger.error("Failed to load secret metadata",
                           file=str(metadata_file),
                           error=str(e))
        
        return secrets
    
    async def list_all_secrets(self) -> List[SecretMetadata]:
        """List all secrets from local files."""
        secrets = []
        
        for metadata_file in self.storage_dir.glob("*.metadata.json"):
            try:
                async with aiofiles.open(metadata_file, 'r') as f:
                    content = await f.read()
                    metadata = SecretMetadata.parse_raw(content)
                    secrets.append(metadata)
            except Exception as e:
                logger.error("Failed to load secret metadata",
                           file=str(metadata_file),
                           error=str(e))
        
        return secrets
    
    def _matches_filters(self, metadata: SecretMetadata, filters: Optional[Dict[str, Any]]) -> bool:
        """Check if metadata matches provided filters."""
        if not filters:
            return True
        
        if 'secret_type' in filters and metadata.secret_type != filters['secret_type']:
            return False
        
        if 'status' in filters and metadata.status != filters['status']:
            return False
        
        if 'tags' in filters:
            for key, value in filters['tags'].items():
                if key not in metadata.tags or metadata.tags[key] != value:
                    return False
        
        return True


class RedisSecretStorage(SecretStorageBackend):
    """Redis storage for temporary/cached secrets."""
    
    def __init__(self):
        self.redis = None
        self.key_prefix = "secrets:"
    
    async def _ensure_redis(self):
        """Ensure Redis connection is available."""
        if not self.redis:
            self.redis = get_redis()
    
    async def store_secret(self, secret_id: str, encrypted_value: str, metadata: SecretMetadata):
        """Store secret in Redis."""
        await self._ensure_redis()
        if not self.redis:
            raise Exception("Redis not available")
        
        # Store secret value
        await self.redis.set(f"{self.key_prefix}value:{secret_id}", encrypted_value)
        
        # Store metadata
        await self.redis.set(f"{self.key_prefix}meta:{secret_id}", metadata.json())
        
        # Set expiration if specified
        if metadata.expires_at:
            ttl = int((metadata.expires_at - datetime.utcnow()).total_seconds())
            if ttl > 0:
                await self.redis.expire(f"{self.key_prefix}value:{secret_id}", ttl)
                await self.redis.expire(f"{self.key_prefix}meta:{secret_id}", ttl)
    
    async def get_secret(self, secret_id: str) -> Optional[str]:
        """Get secret from Redis."""
        await self._ensure_redis()
        if not self.redis:
            return None
        
        value = await self.redis.get(f"{self.key_prefix}value:{secret_id}")
        return value.decode() if value else None
    
    async def update_secret(self, secret_id: str, encrypted_value: str):
        """Update secret in Redis."""
        await self._ensure_redis()
        if not self.redis:
            raise Exception("Redis not available")
        
        await self.redis.set(f"{self.key_prefix}value:{secret_id}", encrypted_value)
    
    async def get_metadata(self, secret_id: str) -> Optional[SecretMetadata]:
        """Get metadata from Redis."""
        await self._ensure_redis()
        if not self.redis:
            return None
        
        metadata_json = await self.redis.get(f"{self.key_prefix}meta:{secret_id}")
        if metadata_json:
            return SecretMetadata.parse_raw(metadata_json.decode())
        return None
    
    async def update_metadata(self, secret_id: str, metadata: SecretMetadata):
        """Update metadata in Redis."""
        await self._ensure_redis()
        if not self.redis:
            raise Exception("Redis not available")
        
        await self.redis.set(f"{self.key_prefix}meta:{secret_id}", metadata.json())
    
    async def list_secrets(self, owner_id: str, filters: Optional[Dict[str, Any]] = None) -> List[SecretMetadata]:
        """List secrets from Redis."""
        await self._ensure_redis()
        if not self.redis:
            return []
        
        secrets = []
        pattern = f"{self.key_prefix}meta:*"
        
        async for key in self.redis.scan_iter(match=pattern):
            try:
                metadata_json = await self.redis.get(key)
                if metadata_json:
                    metadata = SecretMetadata.parse_raw(metadata_json.decode())
                    if metadata.owner_id == owner_id:
                        secrets.append(metadata)
            except Exception as e:
                logger.error("Failed to load secret metadata from Redis",
                           key=key.decode(),
                           error=str(e))
        
        return secrets
    
    async def list_all_secrets(self) -> List[SecretMetadata]:
        """List all secrets from Redis."""
        await self._ensure_redis()
        if not self.redis:
            return []
        
        secrets = []
        pattern = f"{self.key_prefix}meta:*"
        
        async for key in self.redis.scan_iter(match=pattern):
            try:
                metadata_json = await self.redis.get(key)
                if metadata_json:
                    metadata = SecretMetadata.parse_raw(metadata_json.decode())
                    secrets.append(metadata)
            except Exception as e:
                logger.error("Failed to load secret metadata from Redis",
                           key=key.decode(),
                           error=str(e))
        
        return secrets


# Global secrets manager instance
_secrets_manager: Optional[EnterpriseSecretsManager] = None


async def get_secrets_manager() -> EnterpriseSecretsManager:
    """Get or create secrets manager instance."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = EnterpriseSecretsManager()
        await _secrets_manager.initialize()
    return _secrets_manager


# Convenience functions for common secret operations
async def store_secret(name: str, value: str, secret_type: SecretType, owner_id: str, **kwargs) -> str:
    """Convenience function to store a secret."""
    secrets_manager = await get_secrets_manager()
    
    request = SecretRequest(
        name=name,
        secret_type=secret_type,
        value=value,
        **kwargs
    )
    
    metadata = await secrets_manager.create_secret(request, owner_id)
    return metadata.secret_id


async def get_secret_value(secret_id: str, user_id: str) -> Optional[str]:
    """Convenience function to retrieve a secret."""
    secrets_manager = await get_secrets_manager()
    return await secrets_manager.get_secret(secret_id, user_id)


async def rotate_api_keys():
    """Rotate all API keys that are due for rotation."""
    secrets_manager = await get_secrets_manager()
    
    # Get all API key secrets
    all_secrets = []
    for backend in secrets_manager.storage_backends.values():
        secrets = await backend.list_all_secrets()
        all_secrets.extend(secrets)
    
    # Find API keys due for rotation
    now = datetime.utcnow()
    for secret in all_secrets:
        if (secret.secret_type == SecretType.API_KEY and 
            secret.rotation_policy == RotationPolicy.TIME_BASED and
            secret.rotation_interval_days and
            secret.last_rotation):
            
            next_rotation = secret.last_rotation + timedelta(days=secret.rotation_interval_days)
            if now >= next_rotation:
                logger.info("Rotating API key", secret_id=secret.secret_id)
                await secrets_manager.rotate_secret(secret.secret_id)