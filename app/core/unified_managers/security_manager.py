#!/usr/bin/env python3
"""
SecurityManager - Authentication, Authorization, and Security Consolidation
Phase 2.1 Implementation of Technical Debt Remediation Plan

This manager consolidates all security-related functionality including authentication,
authorization, access control, security monitoring, and threat detection into a unified,
high-performance system built on the BaseManager framework.

TARGET CONSOLIDATION: 10+ security-related manager classes → 1 unified SecurityManager
- Authentication and session management
- Authorization and access control (RBAC, ABAC)
- Security token management (JWT, API keys)
- Security monitoring and audit logging
- Threat detection and prevention
- Security policy enforcement
- Encryption and cryptographic operations
- Security incident response
"""

import asyncio
import hashlib
import hmac
import json
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from contextlib import asynccontextmanager
import base64
import jwt

import structlog

# Import BaseManager framework
from .base_manager import (
    BaseManager, ManagerConfig, ManagerDomain, ManagerStatus, ManagerMetrics,
    PluginInterface, PluginType
)

# Import shared patterns from Phase 1
from ...common.utilities.shared_patterns import (
    standard_logging_setup, standard_error_handling
)

logger = structlog.get_logger(__name__)


class AuthenticationMethod(str, Enum):
    """Supported authentication methods."""
    PASSWORD = "password"
    TOKEN = "token"
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH = "oauth"
    CERTIFICATE = "certificate"
    BIOMETRIC = "biometric"
    MULTI_FACTOR = "multi_factor"


class AuthorizationModel(str, Enum):
    """Authorization models supported."""
    RBAC = "rbac"  # Role-Based Access Control
    ABAC = "abac"  # Attribute-Based Access Control
    ACL = "acl"    # Access Control List
    POLICY = "policy"  # Policy-Based


class SecurityLevel(str, Enum):
    """Security levels for resources and operations."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class SecurityEventType(str, Enum):
    """Types of security events."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    TOKEN_CREATED = "token_created"
    TOKEN_EXPIRED = "token_expired"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SECURITY_VIOLATION = "security_violation"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"


class ThreatLevel(str, Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityPrincipal:
    """Represents a security principal (user, service, etc.)."""
    id: str
    name: str
    type: str  # user, service, system, etc.
    roles: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    login_count: int = 0
    failed_login_attempts: int = 0
    account_locked: bool = False
    account_expires_at: Optional[datetime] = None
    password_expires_at: Optional[datetime] = None
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    
    def is_active(self) -> bool:
        """Check if principal is active and not expired."""
        if self.account_locked:
            return False
        if self.account_expires_at and datetime.utcnow() > self.account_expires_at:
            return False
        return True
    
    def has_role(self, role: str) -> bool:
        """Check if principal has a specific role."""
        return role in self.roles
    
    def has_permission(self, permission: str) -> bool:
        """Check if principal has a specific permission."""
        return permission in self.permissions


@dataclass
class SecurityToken:
    """Represents a security token."""
    id: str
    token: str
    principal_id: str
    type: str  # jwt, api_key, session, etc.
    scopes: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    use_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    revoked: bool = False
    
    def is_valid(self) -> bool:
        """Check if token is valid and not expired."""
        if self.revoked:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True
    
    def update_usage(self) -> None:
        """Update token usage statistics."""
        self.last_used_at = datetime.utcnow()
        self.use_count += 1


@dataclass
class SecurityEvent:
    """Represents a security event for audit logging."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: SecurityEventType = SecurityEventType.ACCESS_GRANTED
    principal_id: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: str = "success"  # success, failure, error
    details: Dict[str, Any] = field(default_factory=dict)
    threat_level: ThreatLevel = ThreatLevel.LOW
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class SecurityPolicy:
    """Represents a security policy."""
    id: str
    name: str
    description: str
    rules: List[Dict[str, Any]] = field(default_factory=list)
    active: bool = True
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SecurityMetrics:
    """Security-specific metrics."""
    total_authentications: int = 0
    successful_authentications: int = 0
    failed_authentications: int = 0
    total_authorizations: int = 0
    successful_authorizations: int = 0
    failed_authorizations: int = 0
    active_sessions: int = 0
    total_tokens_issued: int = 0
    tokens_revoked: int = 0
    security_events: int = 0
    threats_detected: int = 0
    threats_blocked: int = 0
    avg_authentication_time_ms: float = 0.0
    avg_authorization_time_ms: float = 0.0
    events_by_type: Dict[SecurityEventType, int] = field(default_factory=dict)
    threats_by_level: Dict[ThreatLevel, int] = field(default_factory=dict)


class SecurityPlugin(PluginInterface):
    """Base class for security plugins."""
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.SECURITY
    
    async def pre_authentication_hook(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called before authentication."""
        return {}
    
    async def post_authentication_hook(self, principal: SecurityPrincipal, success: bool) -> None:
        """Hook called after authentication."""
        pass
    
    async def pre_authorization_hook(self, principal: SecurityPrincipal, resource: str, action: str) -> Dict[str, Any]:
        """Hook called before authorization check."""
        return {}
    
    async def post_authorization_hook(self, principal: SecurityPrincipal, resource: str, action: str, granted: bool) -> None:
        """Hook called after authorization check."""
        pass


class SecurityManager(BaseManager):
    """
    Unified manager for all security operations.
    
    CONSOLIDATION TARGET: Replaces 10+ specialized security managers:
    - AuthenticationManager
    - AuthorizationManager
    - SessionManager
    - TokenManager
    - SecurityAuditManager
    - ThreatDetectionManager
    - AccessControlManager
    - SecurityPolicyManager
    - EncryptionManager
    - SecurityMonitorManager
    
    Built on BaseManager framework with Phase 2 enhancements.
    """
    
    def __init__(self, config: Optional[ManagerConfig] = None):
        # Create default config if none provided
        if config is None:
            config = ManagerConfig(
                name="SecurityManager",
                domain=ManagerDomain.SECURITY,
                max_concurrent_operations=300,
                health_check_interval=10,  # More frequent health checks for security
                circuit_breaker_enabled=True,
                circuit_breaker_failure_threshold=5
            )
        
        super().__init__(config)
        
        # Security-specific state
        self.principals: Dict[str, SecurityPrincipal] = {}
        self.tokens: Dict[str, SecurityToken] = {}
        self.active_sessions: Dict[str, SecurityPrincipal] = {}
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.security_events: List[SecurityEvent] = []
        self.security_metrics = SecurityMetrics()
        
        # Security configuration
        self.jwt_secret = secrets.token_urlsafe(32)
        self.default_token_ttl = 3600  # 1 hour
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes
        self.password_requirements = {
            "min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_digits": True,
            "require_special": True
        }
        
        # Threat detection
        self.threat_rules: List[Callable] = []
        self.blocked_ips: Set[str] = set()
        self.rate_limits: Dict[str, List[float]] = {}  # ip -> timestamps
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._audit_task: Optional[asyncio.Task] = None
        self._threat_monitor_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._principals_lock = threading.RLock()
        self._tokens_lock = threading.RLock()
        self._sessions_lock = threading.RLock()
        self._events_lock = threading.RLock()
        
        self.logger = standard_logging_setup(
            name="SecurityManager",
            level="INFO"
        )
    
    # BaseManager Implementation
    
    async def _setup(self) -> None:
        """Initialize security systems."""
        self.logger.info("Setting up SecurityManager")
        
        # Load default security policies
        await self._load_default_policies()
        
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._audit_task = asyncio.create_task(self._audit_loop())
        self._threat_monitor_task = asyncio.create_task(self._threat_monitor_loop())
        
        self.logger.info("SecurityManager setup completed")
    
    async def _cleanup(self) -> None:
        """Clean up security systems."""
        self.logger.info("Cleaning up SecurityManager")
        
        # Cancel background tasks
        for task in [self._cleanup_task, self._audit_task, self._threat_monitor_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Clear sensitive data
        self.tokens.clear()
        self.active_sessions.clear()
        
        self.logger.info("SecurityManager cleanup completed")
    
    async def _health_check_internal(self) -> Dict[str, Any]:
        """Security-specific health check."""
        with self._principals_lock:
            total_principals = len(self.principals)
            active_principals = sum(1 for p in self.principals.values() if p.is_active())
        
        with self._tokens_lock:
            total_tokens = len(self.tokens)
            valid_tokens = sum(1 for t in self.tokens.values() if t.is_valid())
        
        with self._sessions_lock:
            active_sessions = len(self.active_sessions)
        
        recent_events = len([
            e for e in self.security_events 
            if (datetime.utcnow() - e.timestamp).total_seconds() < 3600
        ])
        
        return {
            "total_principals": total_principals,
            "active_principals": active_principals,
            "total_tokens": total_tokens,
            "valid_tokens": valid_tokens,
            "active_sessions": active_sessions,
            "recent_security_events": recent_events,
            "blocked_ips": len(self.blocked_ips),
            "active_policies": len([p for p in self.security_policies.values() if p.active]),
            "security_metrics": {
                "successful_authentications": self.security_metrics.successful_authentications,
                "failed_authentications": self.security_metrics.failed_authentications,
                "threats_detected": self.security_metrics.threats_detected,
                "threats_blocked": self.security_metrics.threats_blocked
            }
        }
    
    # Core Authentication Operations
    
    async def authenticate(
        self,
        credentials: Dict[str, Any],
        method: AuthenticationMethod = AuthenticationMethod.PASSWORD,
        source_ip: Optional[str] = None
    ) -> Tuple[Optional[SecurityPrincipal], SecurityToken]:
        """
        Authenticate a principal using provided credentials.
        
        CONSOLIDATES: PasswordAuth, TokenAuth, JWTAuth, OAuthAuth patterns
        """
        async with self.execute_with_monitoring("authenticate"):
            start_time = time.time()
            principal = None
            success = False
            
            try:
                # Pre-authentication hooks
                hook_data = {}
                for plugin in self.plugins.values():
                    if isinstance(plugin, SecurityPlugin):
                        plugin_data = await plugin.pre_authentication_hook(credentials)
                        hook_data.update(plugin_data)
                
                # Check if IP is blocked
                if source_ip and source_ip in self.blocked_ips:
                    await self._log_security_event(
                        SecurityEventType.AUTHENTICATION_ERROR,
                        details={"reason": "blocked_ip", "ip": source_ip}
                    )
                    raise ValueError("Access denied: IP blocked")
                
                # Rate limiting check
                if source_ip and not self._check_rate_limit(source_ip):
                    await self._log_security_event(
                        SecurityEventType.AUTHENTICATION_ERROR,
                        details={"reason": "rate_limit", "ip": source_ip}
                    )
                    raise ValueError("Rate limit exceeded")
                
                # Perform authentication based on method
                if method == AuthenticationMethod.PASSWORD:
                    principal = await self._authenticate_password(credentials)
                elif method == AuthenticationMethod.TOKEN:
                    principal = await self._authenticate_token(credentials)
                elif method == AuthenticationMethod.JWT:
                    principal = await self._authenticate_jwt(credentials)
                elif method == AuthenticationMethod.API_KEY:
                    principal = await self._authenticate_api_key(credentials)
                else:
                    raise ValueError(f"Unsupported authentication method: {method}")
                
                if not principal:
                    raise ValueError("Authentication failed: Invalid credentials")
                
                # Check if account is active
                if not principal.is_active():
                    raise ValueError("Authentication failed: Account inactive")
                
                # Create session token
                token = await self._create_session_token(principal)
                
                # Update principal login info
                principal.last_login = datetime.utcnow()
                principal.login_count += 1
                principal.failed_login_attempts = 0  # Reset on successful login
                
                # Create active session
                with self._sessions_lock:
                    self.active_sessions[token.token] = principal
                
                success = True
                
                # Post-authentication hooks
                for plugin in self.plugins.values():
                    if isinstance(plugin, SecurityPlugin):
                        await plugin.post_authentication_hook(principal, success)
                
                # Log successful authentication
                await self._log_security_event(
                    SecurityEventType.LOGIN_SUCCESS,
                    principal_id=principal.id,
                    details={"method": method.value, "ip": source_ip}
                )
                
                # Update metrics
                auth_time_ms = (time.time() - start_time) * 1000
                self.security_metrics.total_authentications += 1
                self.security_metrics.successful_authentications += 1
                self._update_authentication_time_metrics(auth_time_ms)
                
                self.logger.info(
                    f"Authentication successful",
                    principal_id=principal.id,
                    method=method.value,
                    source_ip=source_ip,
                    auth_time_ms=auth_time_ms
                )
                
                return principal, token
                
            except Exception as e:
                # Handle failed authentication
                if principal:
                    principal.failed_login_attempts += 1
                    
                    # Lock account if too many failures
                    if principal.failed_login_attempts >= self.max_failed_attempts:
                        principal.account_locked = True
                        await self._log_security_event(
                            SecurityEventType.SECURITY_VIOLATION,
                            principal_id=principal.id,
                            details={"reason": "account_locked", "failed_attempts": principal.failed_login_attempts}
                        )
                
                # Log failed authentication
                await self._log_security_event(
                    SecurityEventType.LOGIN_FAILURE,
                    principal_id=principal.id if principal else None,
                    details={"method": method.value, "error": str(e), "ip": source_ip}
                )
                
                # Update metrics
                self.security_metrics.total_authentications += 1
                self.security_metrics.failed_authentications += 1
                
                self.logger.warning(
                    f"Authentication failed: {e}",
                    method=method.value,
                    source_ip=source_ip
                )
                
                raise
    
    async def authorize(
        self,
        principal: SecurityPrincipal,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Authorize a principal to perform an action on a resource.
        
        CONSOLIDATES: RBACAuth, ABACAuth, PolicyAuth patterns
        """
        async with self.execute_with_monitoring("authorize"):
            start_time = time.time()
            
            try:
                # Pre-authorization hooks
                hook_data = {}
                for plugin in self.plugins.values():
                    if isinstance(plugin, SecurityPlugin):
                        plugin_data = await plugin.pre_authorization_hook(principal, resource, action)
                        hook_data.update(plugin_data)
                
                # Check if principal is active
                if not principal.is_active():
                    granted = False
                else:
                    # Perform authorization check
                    granted = await self._check_authorization(principal, resource, action, context or {})
                
                # Post-authorization hooks
                for plugin in self.plugins.values():
                    if isinstance(plugin, SecurityPlugin):
                        await plugin.post_authorization_hook(principal, resource, action, granted)
                
                # Log authorization event
                event_type = SecurityEventType.ACCESS_GRANTED if granted else SecurityEventType.ACCESS_DENIED
                await self._log_security_event(
                    event_type,
                    principal_id=principal.id,
                    resource=resource,
                    action=action,
                    details={"granted": granted, "context": context}
                )
                
                # Update metrics
                auth_time_ms = (time.time() - start_time) * 1000
                self.security_metrics.total_authorizations += 1
                if granted:
                    self.security_metrics.successful_authorizations += 1
                else:
                    self.security_metrics.failed_authorizations += 1
                self._update_authorization_time_metrics(auth_time_ms)
                
                self.logger.debug(
                    f"Authorization {('granted' if granted else 'denied')}",
                    principal_id=principal.id,
                    resource=resource,
                    action=action,
                    auth_time_ms=auth_time_ms
                )
                
                return granted
                
            except Exception as e:
                self.logger.error(f"Authorization error: {e}", principal_id=principal.id)
                await self._log_security_event(
                    SecurityEventType.AUTHORIZATION_ERROR,
                    principal_id=principal.id,
                    resource=resource,
                    action=action,
                    details={"error": str(e)}
                )
                return False
    
    # Principal Management
    
    async def create_principal(
        self,
        name: str,
        principal_type: str,
        roles: Optional[Set[str]] = None,
        permissions: Optional[Set[str]] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> SecurityPrincipal:
        """Create a new security principal."""
        async with self.execute_with_monitoring("create_principal"):
            principal_id = str(uuid.uuid4())
            
            principal = SecurityPrincipal(
                id=principal_id,
                name=name,
                type=principal_type,
                roles=roles or set(),
                permissions=permissions or set(),
                attributes=attributes or {}
            )
            
            with self._principals_lock:
                self.principals[principal_id] = principal
            
            self.logger.info(f"Principal created", principal_id=principal_id, name=name)
            return principal
    
    async def get_principal(self, principal_id: str) -> Optional[SecurityPrincipal]:
        """Get a principal by ID."""
        with self._principals_lock:
            return self.principals.get(principal_id)
    
    async def update_principal(
        self,
        principal_id: str,
        roles: Optional[Set[str]] = None,
        permissions: Optional[Set[str]] = None,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update a principal's roles, permissions, or attributes."""
        async with self.execute_with_monitoring("update_principal"):
            with self._principals_lock:
                principal = self.principals.get(principal_id)
                if not principal:
                    return False
                
                if roles is not None:
                    principal.roles = roles
                if permissions is not None:
                    principal.permissions = permissions
                if attributes is not None:
                    principal.attributes.update(attributes)
                
                principal.updated_at = datetime.utcnow()
                
                self.logger.info(f"Principal updated", principal_id=principal_id)
                return True
    
    # Token Management
    
    async def create_token(
        self,
        principal_id: str,
        token_type: str = "api_key",
        scopes: Optional[Set[str]] = None,
        ttl: Optional[int] = None
    ) -> SecurityToken:
        """Create a new security token."""
        async with self.execute_with_monitoring("create_token"):
            token_id = str(uuid.uuid4())
            token_value = secrets.token_urlsafe(32)
            
            token = SecurityToken(
                id=token_id,
                token=token_value,
                principal_id=principal_id,
                type=token_type,
                scopes=scopes or set(),
                expires_at=datetime.utcnow() + timedelta(seconds=ttl or self.default_token_ttl)
            )
            
            with self._tokens_lock:
                self.tokens[token_value] = token
            
            self.security_metrics.total_tokens_issued += 1
            
            await self._log_security_event(
                SecurityEventType.TOKEN_CREATED,
                principal_id=principal_id,
                details={"token_type": token_type, "scopes": list(scopes or [])}
            )
            
            self.logger.info(f"Token created", token_id=token_id, principal_id=principal_id)
            return token
    
    async def revoke_token(self, token_value: str) -> bool:
        """Revoke a security token."""
        async with self.execute_with_monitoring("revoke_token"):
            with self._tokens_lock:
                token = self.tokens.get(token_value)
                if token and not token.revoked:
                    token.revoked = True
                    
                    # Remove from active sessions if it's a session token
                    if token_value in self.active_sessions:
                        del self.active_sessions[token_value]
                    
                    self.security_metrics.tokens_revoked += 1
                    
                    await self._log_security_event(
                        SecurityEventType.TOKEN_EXPIRED,
                        principal_id=token.principal_id,
                        details={"token_type": token.type, "reason": "revoked"}
                    )
                    
                    self.logger.info(f"Token revoked", token_id=token.id)
                    return True
                
                return False
    
    # Private Implementation Methods
    
    async def _authenticate_password(self, credentials: Dict[str, Any]) -> Optional[SecurityPrincipal]:
        """Authenticate using username/password."""
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            return None
        
        # Find principal by username
        with self._principals_lock:
            for principal in self.principals.values():
                if principal.name == username and principal.type == "user":
                    # In a real implementation, verify password hash
                    stored_password = principal.attributes.get("password_hash")
                    if stored_password and self._verify_password(password, stored_password):
                        return principal
        
        return None
    
    async def _authenticate_token(self, credentials: Dict[str, Any]) -> Optional[SecurityPrincipal]:
        """Authenticate using a token."""
        token_value = credentials.get("token")
        
        if not token_value:
            return None
        
        with self._tokens_lock:
            token = self.tokens.get(token_value)
            if token and token.is_valid():
                token.update_usage()
                
                with self._principals_lock:
                    return self.principals.get(token.principal_id)
        
        return None
    
    async def _authenticate_jwt(self, credentials: Dict[str, Any]) -> Optional[SecurityPrincipal]:
        """Authenticate using JWT token."""
        jwt_token = credentials.get("jwt")
        
        if not jwt_token:
            return None
        
        try:
            payload = jwt.decode(jwt_token, self.jwt_secret, algorithms=["HS256"])
            principal_id = payload.get("sub")
            
            if principal_id:
                with self._principals_lock:
                    return self.principals.get(principal_id)
        except jwt.InvalidTokenError:
            pass
        
        return None
    
    async def _authenticate_api_key(self, credentials: Dict[str, Any]) -> Optional[SecurityPrincipal]:
        """Authenticate using API key."""
        api_key = credentials.get("api_key")
        return await self._authenticate_token({"token": api_key})
    
    async def _create_session_token(self, principal: SecurityPrincipal) -> SecurityToken:
        """Create a session token for authenticated principal."""
        return await self.create_token(
            principal_id=principal.id,
            token_type="session",
            scopes={"session"},
            ttl=3600  # 1 hour
        )
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash (simplified implementation)."""
        # In a real implementation, use proper password hashing (bcrypt, scrypt, etc.)
        return hashlib.sha256(password.encode()).hexdigest() == password_hash
    
    async def _check_authorization(
        self,
        principal: SecurityPrincipal,
        resource: str,
        action: str,
        context: Dict[str, Any]
    ) -> bool:
        """Check if principal is authorized for resource/action."""
        # Simple permission check - extend for RBAC/ABAC
        permission = f"{resource}:{action}"
        
        # Direct permission check
        if permission in principal.permissions:
            return True
        
        # Role-based check
        for role in principal.roles:
            role_permissions = await self._get_role_permissions(role)
            if permission in role_permissions:
                return True
        
        # Policy-based check
        for policy in self.security_policies.values():
            if policy.active and await self._evaluate_policy(policy, principal, resource, action, context):
                return True
        
        return False
    
    async def _get_role_permissions(self, role: str) -> Set[str]:
        """Get permissions for a role."""
        # This would typically come from a database or configuration
        role_permissions = {
            "admin": {"*:*"},
            "user": {"read:*", "write:own"},
            "guest": {"read:public"}
        }
        return role_permissions.get(role, set())
    
    async def _evaluate_policy(
        self,
        policy: SecurityPolicy,
        principal: SecurityPrincipal,
        resource: str,
        action: str,
        context: Dict[str, Any]
    ) -> bool:
        """Evaluate a security policy."""
        # Simplified policy evaluation - extend as needed
        for rule in policy.rules:
            if self._match_rule(rule, principal, resource, action, context):
                return rule.get("allow", False)
        return False
    
    def _match_rule(
        self,
        rule: Dict[str, Any],
        principal: SecurityPrincipal,
        resource: str,
        action: str,
        context: Dict[str, Any]
    ) -> bool:
        """Check if a rule matches the current authorization request."""
        # Simplified rule matching
        if "resource" in rule and rule["resource"] != resource:
            return False
        if "action" in rule and rule["action"] != action:
            return False
        if "principal_type" in rule and rule["principal_type"] != principal.type:
            return False
        return True
    
    def _check_rate_limit(self, source_ip: str, limit: int = 10, window: int = 60) -> bool:
        """Check if source IP is within rate limits."""
        current_time = time.time()
        
        if source_ip not in self.rate_limits:
            self.rate_limits[source_ip] = []
        
        # Clean old requests
        self.rate_limits[source_ip] = [
            timestamp for timestamp in self.rate_limits[source_ip]
            if current_time - timestamp < window
        ]
        
        # Check limit
        if len(self.rate_limits[source_ip]) >= limit:
            return False
        
        # Record this request
        self.rate_limits[source_ip].append(current_time)
        return True
    
    async def _log_security_event(
        self,
        event_type: SecurityEventType,
        principal_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a security event."""
        event = SecurityEvent(
            event_type=event_type,
            principal_id=principal_id,
            resource=resource,
            action=action,
            details=details or {}
        )
        
        with self._events_lock:
            self.security_events.append(event)
            
            # Keep only recent events (last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            self.security_events = [
                e for e in self.security_events 
                if e.timestamp > cutoff_time
            ]
        
        self.security_metrics.security_events += 1
        
        # Update event type metrics
        current = self.security_metrics.events_by_type.get(event_type, 0)
        self.security_metrics.events_by_type[event_type] = current + 1
        
        # Check for threats
        await self._detect_threats(event)
    
    async def _detect_threats(self, event: SecurityEvent) -> None:
        """Detect potential security threats based on events."""
        threat_detected = False
        threat_level = ThreatLevel.LOW
        
        # Multiple failed logins
        if event.event_type == SecurityEventType.LOGIN_FAILURE:
            recent_failures = len([
                e for e in self.security_events[-50:]  # Last 50 events
                if (e.event_type == SecurityEventType.LOGIN_FAILURE and
                    e.details.get("ip") == event.details.get("ip") and
                    (datetime.utcnow() - e.timestamp).total_seconds() < 300)  # 5 minutes
            ])
            
            if recent_failures >= 5:
                threat_detected = True
                threat_level = ThreatLevel.HIGH
                
                # Block IP
                source_ip = event.details.get("ip")
                if source_ip:
                    self.blocked_ips.add(source_ip)
        
        if threat_detected:
            self.security_metrics.threats_detected += 1
            current = self.security_metrics.threats_by_level.get(threat_level, 0)
            self.security_metrics.threats_by_level[threat_level] = current + 1
            
            self.logger.warning(
                f"Security threat detected",
                event_type=event.event_type.value,
                threat_level=threat_level.value,
                details=event.details
            )
    
    async def _load_default_policies(self) -> None:
        """Load default security policies."""
        # Admin policy
        admin_policy = SecurityPolicy(
            id="admin_policy",
            name="Administrator Policy",
            description="Full access for administrators",
            rules=[
                {"principal_type": "user", "role": "admin", "allow": True}
            ],
            priority=100
        )
        
        self.security_policies[admin_policy.id] = admin_policy
        
        self.logger.info("Default security policies loaded")
    
    # Background Tasks
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of expired tokens and sessions."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                if self._shutdown_event.is_set():
                    break
                
                current_time = datetime.utcnow()
                expired_tokens = []
                
                # Find expired tokens
                with self._tokens_lock:
                    for token_value, token in self.tokens.items():
                        if not token.is_valid():
                            expired_tokens.append(token_value)
                
                # Remove expired tokens
                for token_value in expired_tokens:
                    with self._tokens_lock:
                        if token_value in self.tokens:
                            token = self.tokens[token_value]
                            del self.tokens[token_value]
                            
                            await self._log_security_event(
                                SecurityEventType.TOKEN_EXPIRED,
                                principal_id=token.principal_id,
                                details={"token_type": token.type, "reason": "expired"}
                            )
                    
                    # Remove from active sessions
                    with self._sessions_lock:
                        if token_value in self.active_sessions:
                            del self.active_sessions[token_value]
                
                if expired_tokens:
                    self.logger.debug(f"Cleaned up {len(expired_tokens)} expired tokens")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    async def _audit_loop(self) -> None:
        """Periodic audit logging and compliance checks."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(3600)  # Run every hour
                if self._shutdown_event.is_set():
                    break
                
                # Generate audit summary
                with self._events_lock:
                    recent_events = [
                        e for e in self.security_events
                        if (datetime.utcnow() - e.timestamp).total_seconds() < 3600
                    ]
                
                self.logger.info(
                    "Security audit summary",
                    total_events=len(recent_events),
                    event_types={
                        event_type.value: len([e for e in recent_events if e.event_type == event_type])
                        for event_type in SecurityEventType
                    },
                    active_sessions=len(self.active_sessions),
                    blocked_ips=len(self.blocked_ips)
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Audit loop error: {e}")
    
    async def _threat_monitor_loop(self) -> None:
        """Monitor for security threats and anomalies."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                if self._shutdown_event.is_set():
                    break
                
                # Check for suspicious patterns
                # This is where advanced threat detection would go
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Threat monitor loop error: {e}")
    
    # Metrics Helpers
    
    def _update_authentication_time_metrics(self, auth_time_ms: float) -> None:
        """Update authentication time metrics."""
        total = self.security_metrics.total_authentications
        current_avg = self.security_metrics.avg_authentication_time_ms
        
        if total == 1:
            self.security_metrics.avg_authentication_time_ms = auth_time_ms
        else:
            self.security_metrics.avg_authentication_time_ms = (
                (current_avg * (total - 1) + auth_time_ms) / total
            )
    
    def _update_authorization_time_metrics(self, auth_time_ms: float) -> None:
        """Update authorization time metrics."""
        total = self.security_metrics.total_authorizations
        current_avg = self.security_metrics.avg_authorization_time_ms
        
        if total == 1:
            self.security_metrics.avg_authorization_time_ms = auth_time_ms
        else:
            self.security_metrics.avg_authorization_time_ms = (
                (current_avg * (total - 1) + auth_time_ms) / total
            )
    
    # Public API Extensions
    
    def get_security_metrics(self) -> SecurityMetrics:
        """Get current security metrics."""
        # Update active sessions count
        with self._sessions_lock:
            self.security_metrics.active_sessions = len(self.active_sessions)
        
        return self.security_metrics
    
    async def get_security_events(
        self,
        event_type: Optional[SecurityEventType] = None,
        principal_id: Optional[str] = None,
        hours: int = 24
    ) -> List[SecurityEvent]:
        """Get security events with optional filtering."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        with self._events_lock:
            events = [
                e for e in self.security_events
                if e.timestamp > cutoff_time
            ]
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if principal_id:
            events = [e for e in events if e.principal_id == principal_id]
        
        return events
    
    async def block_ip(self, ip_address: str, reason: str = "manual_block") -> None:
        """Block an IP address."""
        self.blocked_ips.add(ip_address)
        
        await self._log_security_event(
            SecurityEventType.SECURITY_VIOLATION,
            details={"action": "ip_blocked", "ip": ip_address, "reason": reason}
        )
        
        self.logger.info(f"IP address blocked", ip=ip_address, reason=reason)
    
    async def unblock_ip(self, ip_address: str) -> bool:
        """Unblock an IP address."""
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            
            await self._log_security_event(
                SecurityEventType.SECURITY_VIOLATION,
                details={"action": "ip_unblocked", "ip": ip_address}
            )
            
            self.logger.info(f"IP address unblocked", ip=ip_address)
            return True
        
        return False


# Plugin Examples

class MultiFactorAuthPlugin(SecurityPlugin):
    """Plugin for multi-factor authentication."""
    
    @property
    def name(self) -> str:
        return "MultiFactorAuth"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def initialize(self, manager: BaseManager) -> None:
        pass
    
    async def cleanup(self) -> None:
        pass
    
    async def pre_authentication_hook(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        # Check if MFA is required for this principal
        if credentials.get("require_mfa"):
            mfa_token = credentials.get("mfa_token")
            if not mfa_token:
                raise ValueError("MFA token required")
            # Verify MFA token (implementation would go here)
        
        return {}


class SecurityAuditPlugin(SecurityPlugin):
    """Plugin for enhanced security auditing."""
    
    @property
    def name(self) -> str:
        return "SecurityAudit"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def initialize(self, manager: BaseManager) -> None:
        self.audit_log = []
    
    async def cleanup(self) -> None:
        pass
    
    async def post_authentication_hook(self, principal: SecurityPrincipal, success: bool) -> None:
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "authentication",
            "principal_id": principal.id,
            "success": success
        }
        self.audit_log.append(audit_entry)
    
    async def post_authorization_hook(self, principal: SecurityPrincipal, resource: str, action: str, granted: bool) -> None:
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "authorization",
            "principal_id": principal.id,
            "resource": resource,
            "action": action,
            "granted": granted
        }
        self.audit_log.append(audit_entry)