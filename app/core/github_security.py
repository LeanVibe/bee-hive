"""
GitHub Security Layer for LeanVibe Agent Hive 2.0

Comprehensive security management for GitHub integration including token management,
permission scopes, audit trails, and access control for multi-agent environments.
"""

import asyncio
import logging
import uuid
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set
from enum import Enum
import json

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
from sqlalchemy.orm import selectinload

from ..core.config import get_settings
from ..core.database import get_db_session
from ..models.agent import Agent
from ..models.github_integration import GitHubRepository


logger = logging.getLogger(__name__)
settings = get_settings()


class GitHubPermission(Enum):
    """GitHub permission types."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    ISSUES = "issues"
    PULL_REQUESTS = "pull_requests"
    METADATA = "metadata"
    CONTENTS = "contents"
    ACTIONS = "actions"
    CHECKS = "checks"
    DEPLOYMENTS = "deployments"
    PACKAGES = "packages"
    PAGES = "pages"
    SECURITY_EVENTS = "security_events"
    SINGLE_FILE = "single_file"
    VULNERABILITIES = "vulnerabilities"


class AccessLevel(Enum):
    """Access levels for GitHub resources."""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class AuditEventType(Enum):
    """Types of audit events."""
    TOKEN_CREATED = "token_created"
    TOKEN_REFRESHED = "token_refreshed"
    TOKEN_REVOKED = "token_revoked"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    ACCESS_DENIED = "access_denied"
    REPOSITORY_ACCESS = "repository_access"
    API_CALL = "api_call"
    WEBHOOK_RECEIVED = "webhook_received"
    SECURITY_VIOLATION = "security_violation"


class GitHubSecurityError(Exception):
    """Custom exception for GitHub security operations."""
    pass


class TokenManager:
    """
    Secure token management for GitHub access tokens.
    
    Handles encryption, rotation, and secure storage of GitHub tokens
    with automatic expiration and refresh capabilities.
    """
    
    def __init__(self):
        self.encryption_key = self._derive_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        
    def _derive_encryption_key(self) -> bytes:
        """Derive encryption key from application secret."""
        password = settings.SECRET_KEY.encode()
        salt = b'github_token_salt'  # In production, use a secure random salt
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return key
        
    def encrypt_token(self, token: str) -> str:
        """Encrypt GitHub token for secure storage."""
        if not token:
            raise GitHubSecurityError("Cannot encrypt empty token")
            
        try:
            encrypted_token = self.fernet.encrypt(token.encode())
            return base64.urlsafe_b64encode(encrypted_token).decode()
        except Exception as e:
            logger.error(f"Token encryption failed: {e}")
            raise GitHubSecurityError(f"Token encryption failed: {str(e)}")
            
    def decrypt_token(self, encrypted_token: str) -> str:
        """Decrypt GitHub token for use."""
        if not encrypted_token:
            raise GitHubSecurityError("Cannot decrypt empty token")
            
        try:
            decoded_token = base64.urlsafe_b64decode(encrypted_token.encode())
            decrypted_token = self.fernet.decrypt(decoded_token)
            return decrypted_token.decode()
        except Exception as e:
            logger.error(f"Token decryption failed: {e}")
            raise GitHubSecurityError(f"Token decryption failed: {str(e)}")
            
    def validate_token_format(self, token: str) -> bool:
        """Validate GitHub token format."""
        if not token:
            return False
            
        # GitHub personal access tokens start with 'ghp_' or 'github_pat_'
        # GitHub App tokens start with 'ghs_'
        valid_prefixes = ['ghp_', 'github_pat_', 'ghs_']
        
        return any(token.startswith(prefix) for prefix in valid_prefixes)
        
    def generate_webhook_secret(self) -> str:
        """Generate secure webhook secret."""
        return secrets.token_urlsafe(32)
        
    def verify_webhook_signature(self, payload: bytes, signature: str, secret: str) -> bool:
        """Verify webhook signature."""
        if not all([payload, signature, secret]):
            return False
            
        try:
            expected_signature = hmac.new(
                secret.encode('utf-8'),
                payload,
                hashlib.sha256
            ).hexdigest()
            
            # Remove 'sha256=' prefix from GitHub signature
            if signature.startswith('sha256='):
                signature = signature[7:]
                
            return hmac.compare_digest(expected_signature, signature)
        except Exception as e:
            logger.error(f"Webhook signature verification failed: {e}")
            return False


class PermissionManager:
    """
    Permission management for GitHub resources.
    
    Manages fine-grained permissions for agents accessing GitHub repositories
    with role-based access control and dynamic permission updates.
    """
    
    def __init__(self):
        self.default_permissions = {
            GitHubPermission.READ: AccessLevel.READ,
            GitHubPermission.METADATA: AccessLevel.READ,
            GitHubPermission.ISSUES: AccessLevel.WRITE,
            GitHubPermission.PULL_REQUESTS: AccessLevel.WRITE,
            GitHubPermission.CONTENTS: AccessLevel.WRITE,
        }
        
        self.permission_hierarchies = {
            AccessLevel.ADMIN: [AccessLevel.WRITE, AccessLevel.READ, AccessLevel.NONE],
            AccessLevel.WRITE: [AccessLevel.READ, AccessLevel.NONE],
            AccessLevel.READ: [AccessLevel.NONE],
            AccessLevel.NONE: []
        }
        
    def get_default_agent_permissions(self) -> Dict[str, str]:
        """Get default permissions for new agents."""
        return {perm.value: level.value for perm, level in self.default_permissions.items()}
        
    def validate_permission_request(
        self,
        requested_permissions: Dict[str, str],
        max_allowed_permissions: Dict[str, str]
    ) -> Dict[str, Any]:
        """Validate permission request against maximum allowed permissions."""
        
        validation_result = {
            "valid": True,
            "granted_permissions": {},
            "denied_permissions": {},
            "warnings": []
        }
        
        for permission, requested_level in requested_permissions.items():
            try:
                requested_access = AccessLevel(requested_level)
                max_allowed_access = AccessLevel(max_allowed_permissions.get(permission, "none"))
                
                if self._has_permission(requested_access, max_allowed_access):
                    validation_result["granted_permissions"][permission] = requested_level
                else:
                    validation_result["denied_permissions"][permission] = {
                        "requested": requested_level,
                        "max_allowed": max_allowed_access.value
                    }
                    validation_result["valid"] = False
                    
            except ValueError:
                validation_result["warnings"].append(f"Invalid permission level: {requested_level}")
                validation_result["denied_permissions"][permission] = {
                    "requested": requested_level,
                    "reason": "invalid_level"
                }
                validation_result["valid"] = False
                
        return validation_result
        
    def _has_permission(self, requested: AccessLevel, maximum: AccessLevel) -> bool:
        """Check if requested access level is within maximum allowed."""
        return requested in self.permission_hierarchies.get(maximum, []) or requested == maximum
        
    def calculate_minimum_required_permissions(self, operations: List[str]) -> Dict[str, str]:
        """Calculate minimum permissions required for specific operations."""
        
        operation_requirements = {
            "read_repository": {GitHubPermission.READ: AccessLevel.READ},
            "create_branch": {GitHubPermission.CONTENTS: AccessLevel.WRITE},
            "create_pull_request": {GitHubPermission.PULL_REQUESTS: AccessLevel.WRITE},
            "merge_pull_request": {GitHubPermission.PULL_REQUESTS: AccessLevel.WRITE},
            "create_issue": {GitHubPermission.ISSUES: AccessLevel.WRITE},
            "close_issue": {GitHubPermission.ISSUES: AccessLevel.WRITE},
            "push_commits": {GitHubPermission.CONTENTS: AccessLevel.WRITE},
            "create_webhook": {GitHubPermission.ADMIN: AccessLevel.ADMIN},
            "manage_deployments": {GitHubPermission.DEPLOYMENTS: AccessLevel.WRITE}
        }
        
        required_permissions = {}
        
        for operation in operations:
            if operation in operation_requirements:
                for permission, level in operation_requirements[operation].items():
                    current_level = required_permissions.get(permission.value, AccessLevel.NONE)
                    if self._has_permission(level, current_level) or level == current_level:
                        continue
                    required_permissions[permission.value] = level.value
                    
        return required_permissions
        
    def check_agent_permission(
        self,
        agent_permissions: Dict[str, str],
        required_permission: str,
        required_level: str
    ) -> bool:
        """Check if agent has required permission level."""
        
        try:
            agent_level = AccessLevel(agent_permissions.get(required_permission, "none"))
            required_access = AccessLevel(required_level)
            
            return self._has_permission(required_access, agent_level) or agent_level == required_access
            
        except ValueError:
            return False


class AuditLogger:
    """
    Comprehensive audit logging for GitHub security events.
    
    Tracks all security-related events with detailed context,
    correlation IDs, and automated anomaly detection.
    """
    
    def __init__(self):
        self.sensitive_fields = {'token', 'secret', 'password', 'key', 'authorization'}
        
    async def log_security_event(
        self,
        event_type: AuditEventType,
        agent_id: Optional[str] = None,
        repository_id: Optional[str] = None,
        context: Dict[str, Any] = None,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """Log security event with full context."""
        
        correlation_id = correlation_id or str(uuid.uuid4())
        context = context or {}
        
        # Sanitize sensitive information
        sanitized_context = self._sanitize_context(context)
        
        audit_entry = {
            "correlation_id": correlation_id,
            "event_type": event_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": agent_id,
            "repository_id": repository_id,
            "context": sanitized_context,
            "user_agent": user_agent,
            "ip_address": ip_address,
            "severity": self._calculate_event_severity(event_type),
            "tags": self._generate_event_tags(event_type, context)
        }
        
        # Log to application logger
        logger.info(f"Security Event: {event_type.value}", extra={
            "audit_entry": audit_entry,
            "correlation_id": correlation_id
        })
        
        # Store in database for analysis
        await self._store_audit_entry(audit_entry)
        
        # Check for anomalies
        await self._check_for_anomalies(audit_entry)
        
        return correlation_id
        
    def _sanitize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or mask sensitive information from context."""
        
        sanitized = {}
        
        for key, value in context.items():
            if any(sensitive in key.lower() for sensitive in self.sensitive_fields):
                if isinstance(value, str) and len(value) > 8:
                    sanitized[key] = f"{value[:4]}***{value[-4:]}"
                else:
                    sanitized[key] = "***"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_context(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_context(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
                
        return sanitized
        
    def _calculate_event_severity(self, event_type: AuditEventType) -> str:
        """Calculate event severity level."""
        
        severity_map = {
            AuditEventType.SECURITY_VIOLATION: "critical",
            AuditEventType.ACCESS_DENIED: "high",
            AuditEventType.TOKEN_REVOKED: "medium",
            AuditEventType.TOKEN_CREATED: "low",
            AuditEventType.API_CALL: "info",
            AuditEventType.WEBHOOK_RECEIVED: "info"
        }
        
        return severity_map.get(event_type, "medium")
        
    def _generate_event_tags(self, event_type: AuditEventType, context: Dict[str, Any]) -> List[str]:
        """Generate tags for event categorization."""
        
        tags = [event_type.value]
        
        # Add context-based tags
        if "permission" in context:
            tags.append(f"permission:{context['permission']}")
            
        if "repository" in context:
            tags.append("repository_access")
            
        if "webhook" in context:
            tags.append("webhook")
            
        if "api_endpoint" in context:
            tags.append(f"api:{context['api_endpoint']}")
            
        return tags
        
    async def _store_audit_entry(self, audit_entry: Dict[str, Any]) -> None:
        """Store audit entry in database."""
        
        try:
            # In a real implementation, this would store in a dedicated audit table
            # For now, just log the entry
            logger.debug(f"Audit entry stored: {audit_entry['correlation_id']}")
            
        except Exception as e:
            logger.error(f"Failed to store audit entry: {e}")
            
    async def _check_for_anomalies(self, audit_entry: Dict[str, Any]) -> None:
        """Check for security anomalies and alert if necessary."""
        
        try:
            # Implement anomaly detection logic
            # Examples:
            # - Multiple failed access attempts
            # - Unusual API usage patterns
            # - Unexpected permission requests
            # - Geographic anomalies
            
            event_type = AuditEventType(audit_entry["event_type"])
            
            if event_type == AuditEventType.ACCESS_DENIED:
                await self._check_access_denial_patterns(audit_entry)
            elif event_type == AuditEventType.API_CALL:
                await self._check_api_usage_patterns(audit_entry)
                
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            
    async def _check_access_denial_patterns(self, audit_entry: Dict[str, Any]) -> None:
        """Check for suspicious access denial patterns."""
        
        agent_id = audit_entry.get("agent_id")
        if not agent_id:
            return
            
        # Count recent access denials for this agent
        # This would query the audit database in a real implementation
        recent_denials = 0  # Placeholder
        
        if recent_denials > 5:  # Threshold
            await self._trigger_security_alert(
                "Multiple access denials detected",
                audit_entry,
                "potential_brute_force"
            )
            
    async def _check_api_usage_patterns(self, audit_entry: Dict[str, Any]) -> None:
        """Check for unusual API usage patterns."""
        
        # Implement rate limiting checks, unusual endpoint access, etc.
        pass
        
    async def _trigger_security_alert(
        self,
        message: str,
        audit_entry: Dict[str, Any],
        alert_type: str
    ) -> None:
        """Trigger security alert for investigation."""
        
        alert = {
            "alert_type": alert_type,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "audit_entry": audit_entry,
            "severity": "high"
        }
        
        logger.warning(f"Security Alert: {message}", extra={"alert": alert})
        
        # In production, this would:
        # - Send notifications to security team
        # - Create incident tickets
        # - Trigger automated responses


class GitHubSecurityManager:
    """
    Central security manager for GitHub integration.
    
    Orchestrates token management, permission validation, and audit logging
    for comprehensive security coverage across all GitHub operations.
    """
    
    def __init__(self):
        self.token_manager = TokenManager()
        self.permission_manager = PermissionManager()
        self.audit_logger = AuditLogger()
        
        # Security policies
        self.security_policies = {
            "max_token_age_days": 30,
            "require_webhook_signatures": True,
            "minimum_permission_level": AccessLevel.READ,
            "allowed_operations": [
                "read_repository", "create_branch", "create_pull_request",
                "merge_pull_request", "create_issue", "close_issue", "push_commits"
            ],
            "blocked_operations": ["delete_repository", "transfer_repository"],
            "rate_limits": {
                "api_calls_per_hour": 1000,
                "webhook_events_per_hour": 5000
            }
        }
        
    async def setup_agent_github_access(
        self,
        agent_id: str,
        repository_id: str,
        github_token: str,
        requested_permissions: Dict[str, str],
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Set up GitHub access for agent with security validation."""
        
        correlation_id = correlation_id or str(uuid.uuid4())
        
        try:
            # Validate token format
            if not self.token_manager.validate_token_format(github_token):
                await self.audit_logger.log_security_event(
                    AuditEventType.SECURITY_VIOLATION,
                    agent_id=agent_id,
                    context={"error": "invalid_token_format"},
                    correlation_id=correlation_id
                )
                raise GitHubSecurityError("Invalid GitHub token format")
                
            # Validate permissions
            max_allowed = self.permission_manager.get_default_agent_permissions()
            permission_validation = self.permission_manager.validate_permission_request(
                requested_permissions, max_allowed
            )
            
            if not permission_validation["valid"]:
                await self.audit_logger.log_security_event(
                    AuditEventType.ACCESS_DENIED,
                    agent_id=agent_id,
                    repository_id=repository_id,
                    context={
                        "denied_permissions": permission_validation["denied_permissions"],
                        "requested_permissions": requested_permissions
                    },
                    correlation_id=correlation_id
                )
                raise GitHubSecurityError(f"Permission validation failed: {permission_validation['denied_permissions']}")
                
            # Encrypt and store token
            encrypted_token = self.token_manager.encrypt_token(github_token)
            
            # Update repository record
            async with get_db_session() as session:
                repo_result = await session.execute(
                    select(GitHubRepository).where(GitHubRepository.id == uuid.UUID(repository_id))
                )
                repository = repo_result.scalar_one_or_none()
                
                if not repository:
                    raise GitHubSecurityError("Repository not found")
                    
                repository.access_token_hash = encrypted_token
                repository.agent_permissions = permission_validation["granted_permissions"]
                
                await session.commit()
                
            # Log successful setup
            await self.audit_logger.log_security_event(
                AuditEventType.TOKEN_CREATED,
                agent_id=agent_id,
                repository_id=repository_id,
                context={
                    "granted_permissions": permission_validation["granted_permissions"],
                    "token_created": True
                },
                correlation_id=correlation_id
            )
            
            return {
                "success": True,
                "granted_permissions": permission_validation["granted_permissions"],
                "denied_permissions": permission_validation["denied_permissions"],
                "correlation_id": correlation_id
            }
            
        except Exception as e:
            await self.audit_logger.log_security_event(
                AuditEventType.SECURITY_VIOLATION,
                agent_id=agent_id,
                repository_id=repository_id,
                context={"error": str(e), "setup_failed": True},
                correlation_id=correlation_id
            )
            raise GitHubSecurityError(f"GitHub access setup failed: {str(e)}")
            
    async def validate_agent_operation(
        self,
        agent_id: str,
        repository_id: str,
        operation: str,
        context: Dict[str, Any] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate if agent can perform specific GitHub operation."""
        
        correlation_id = correlation_id or str(uuid.uuid4())
        context = context or {}
        
        try:
            # Check if operation is allowed by policy
            if operation in self.security_policies["blocked_operations"]:
                await self.audit_logger.log_security_event(
                    AuditEventType.ACCESS_DENIED,
                    agent_id=agent_id,
                    repository_id=repository_id,
                    context={"operation": operation, "reason": "blocked_by_policy"},
                    correlation_id=correlation_id
                )
                return {"allowed": False, "reason": "Operation blocked by security policy"}
                
            if operation not in self.security_policies["allowed_operations"]:
                await self.audit_logger.log_security_event(
                    AuditEventType.ACCESS_DENIED,
                    agent_id=agent_id,
                    repository_id=repository_id,
                    context={"operation": operation, "reason": "operation_not_allowed"},
                    correlation_id=correlation_id
                )
                return {"allowed": False, "reason": "Operation not in allowed list"}
                
            # Get repository and check permissions
            async with get_db_session() as session:
                repo_result = await session.execute(
                    select(GitHubRepository).where(GitHubRepository.id == uuid.UUID(repository_id))
                )
                repository = repo_result.scalar_one_or_none()
                
                if not repository:
                    return {"allowed": False, "reason": "Repository not found"}
                    
                # Calculate required permissions for operation
                required_permissions = self.permission_manager.calculate_minimum_required_permissions([operation])
                
                # Check if agent has required permissions
                agent_permissions = repository.agent_permissions or {}
                
                for permission, required_level in required_permissions.items():
                    if not self.permission_manager.check_agent_permission(
                        agent_permissions, permission, required_level
                    ):
                        await self.audit_logger.log_security_event(
                            AuditEventType.ACCESS_DENIED,
                            agent_id=agent_id,
                            repository_id=repository_id,
                            context={
                                "operation": operation,
                                "missing_permission": permission,
                                "required_level": required_level,
                                "agent_level": agent_permissions.get(permission, "none")
                            },
                            correlation_id=correlation_id
                        )
                        return {
                            "allowed": False,
                            "reason": f"Insufficient {permission} permission (requires {required_level})"
                        }
                        
            # Log successful validation
            await self.audit_logger.log_security_event(
                AuditEventType.REPOSITORY_ACCESS,
                agent_id=agent_id,
                repository_id=repository_id,
                context={"operation": operation, "validation": "passed"},
                correlation_id=correlation_id
            )
            
            return {
                "allowed": True,
                "correlation_id": correlation_id,
                "required_permissions": required_permissions
            }
            
        except Exception as e:
            await self.audit_logger.log_security_event(
                AuditEventType.SECURITY_VIOLATION,
                agent_id=agent_id,
                repository_id=repository_id,
                context={"operation": operation, "validation_error": str(e)},
                correlation_id=correlation_id
            )
            return {"allowed": False, "reason": f"Validation error: {str(e)}"}
            
    async def get_decrypted_token(
        self,
        repository_id: str,
        agent_id: str,
        correlation_id: Optional[str] = None
    ) -> Optional[str]:
        """Get decrypted GitHub token for repository access."""
        
        correlation_id = correlation_id or str(uuid.uuid4())
        
        try:
            async with get_db_session() as session:
                repo_result = await session.execute(
                    select(GitHubRepository).where(GitHubRepository.id == uuid.UUID(repository_id))
                )
                repository = repo_result.scalar_one_or_none()
                
                if not repository or not repository.access_token_hash:
                    await self.audit_logger.log_security_event(
                        AuditEventType.ACCESS_DENIED,
                        agent_id=agent_id,
                        repository_id=repository_id,
                        context={"reason": "no_token_found"},
                        correlation_id=correlation_id
                    )
                    return None
                    
                # Decrypt token
                decrypted_token = self.token_manager.decrypt_token(repository.access_token_hash)
                
                # Log token access
                await self.audit_logger.log_security_event(
                    AuditEventType.API_CALL,
                    agent_id=agent_id,
                    repository_id=repository_id,
                    context={"action": "token_access"},
                    correlation_id=correlation_id
                )
                
                return decrypted_token
                
        except Exception as e:
            await self.audit_logger.log_security_event(
                AuditEventType.SECURITY_VIOLATION,
                agent_id=agent_id,
                repository_id=repository_id,
                context={"token_access_error": str(e)},
                correlation_id=correlation_id
            )
            logger.error(f"Token decryption failed: {e}")
            return None
            
    async def rotate_repository_token(
        self,
        repository_id: str,
        new_token: str,
        agent_id: str,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Rotate GitHub token for repository."""
        
        correlation_id = correlation_id or str(uuid.uuid4())
        
        try:
            # Validate new token
            if not self.token_manager.validate_token_format(new_token):
                raise GitHubSecurityError("Invalid token format")
                
            # Encrypt new token
            encrypted_token = self.token_manager.encrypt_token(new_token)
            
            # Update repository
            async with get_db_session() as session:
                repo_result = await session.execute(
                    select(GitHubRepository).where(GitHubRepository.id == uuid.UUID(repository_id))
                )
                repository = repo_result.scalar_one_or_none()
                
                if not repository:
                    raise GitHubSecurityError("Repository not found")
                    
                old_token_exists = bool(repository.access_token_hash)
                repository.access_token_hash = encrypted_token
                
                await session.commit()
                
            # Log token rotation
            await self.audit_logger.log_security_event(
                AuditEventType.TOKEN_REFRESHED,
                agent_id=agent_id,
                repository_id=repository_id,
                context={
                    "old_token_existed": old_token_exists,
                    "rotation_reason": "manual"
                },
                correlation_id=correlation_id
            )
            
            return {
                "success": True,
                "correlation_id": correlation_id,
                "rotated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            await self.audit_logger.log_security_event(
                AuditEventType.SECURITY_VIOLATION,
                agent_id=agent_id,
                repository_id=repository_id,
                context={"token_rotation_error": str(e)},
                correlation_id=correlation_id
            )
            raise GitHubSecurityError(f"Token rotation failed: {str(e)}")
            
    async def revoke_repository_access(
        self,
        repository_id: str,
        agent_id: str,
        reason: str,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Revoke GitHub access for repository."""
        
        correlation_id = correlation_id or str(uuid.uuid4())
        
        try:
            async with get_db_session() as session:
                repo_result = await session.execute(
                    select(GitHubRepository).where(GitHubRepository.id == uuid.UUID(repository_id))
                )
                repository = repo_result.scalar_one_or_none()
                
                if not repository:
                    raise GitHubSecurityError("Repository not found")
                    
                # Clear token and permissions
                had_token = bool(repository.access_token_hash)
                had_permissions = bool(repository.agent_permissions)
                
                repository.access_token_hash = None
                repository.agent_permissions = {}
                
                await session.commit()
                
            # Log access revocation
            await self.audit_logger.log_security_event(
                AuditEventType.TOKEN_REVOKED,
                agent_id=agent_id,
                repository_id=repository_id,
                context={
                    "reason": reason,
                    "had_token": had_token,
                    "had_permissions": had_permissions
                },
                correlation_id=correlation_id
            )
            
            return {
                "success": True,
                "correlation_id": correlation_id,
                "revoked_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            await self.audit_logger.log_security_event(
                AuditEventType.SECURITY_VIOLATION,
                agent_id=agent_id,
                repository_id=repository_id,
                context={"revocation_error": str(e)},
                correlation_id=correlation_id
            )
            raise GitHubSecurityError(f"Access revocation failed: {str(e)}")
            
    async def get_security_audit_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate security audit report."""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # In a real implementation, this would query the audit database
        # For now, return a summary structure
        
        report = {
            "period_days": days,
            "report_generated": datetime.utcnow().isoformat(),
            "summary": {
                "total_events": 0,  # Would be queried from audit database
                "security_violations": 0,
                "access_denials": 0,
                "token_operations": 0,
                "api_calls": 0
            },
            "top_agents_by_activity": [],
            "top_repositories_by_access": [],
            "security_alerts": [],
            "recommendations": []
        }
        
        # Add security recommendations based on patterns
        report["recommendations"] = [
            "Consider implementing token rotation for repositories with tokens older than 30 days",
            "Review agents with multiple access denials for potential permission adjustments",
            "Monitor for unusual API usage patterns outside normal business hours"
        ]
        
        return report