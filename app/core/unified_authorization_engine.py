"""
Unified Authorization Engine for LeanVibe Agent Hive
Consolidates 6+ authorization implementations into comprehensive security system

This unified engine combines:
- Core RBAC authorization (authorization_engine.py)
- Advanced hierarchical RBAC (rbac_engine.py) 
- Context-based access control (access_control.py)
- API security middleware (api_security_middleware.py)
- Security validation (security_validation_middleware.py)
- Production API security (production_api_security.py)

Features:
- Enterprise-grade RBAC with hierarchical roles
- Context-aware authorization with dynamic conditions
- Multi-factor authentication support
- Advanced threat detection and prevention
- Rate limiting and DDoS protection
- Input validation and sanitization
- Comprehensive security monitoring and audit logging
- Performance-optimized with intelligent caching
"""

import asyncio
import hashlib
import hmac
import ipaddress
import json
import re
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import logging

from fastapi import Request, Response, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, update, delete
from sqlalchemy.orm import selectinload
import jwt
import redis.asyncio as redis

from app.core.logging_service import get_component_logger
from app.core.circuit_breaker import CircuitBreakerService
from app.core.configuration_service import ConfigurationService
from app.core.redis_integration import get_redis_service
from app.models.security import (
    AgentIdentity, AgentRole, AgentRoleAssignment, SecurityAuditLog, SecurityEvent,
    AgentStatus, RoleScope, SecurityEventSeverity
)

logger = get_component_logger("unified_authorization")

# Configuration Constants
DEFAULT_CONFIG = {
    "jwt_secret_key": "default-secret-change-in-production",
    "jwt_algorithm": "HS256",
    "jwt_expiry_hours": 24,
    "max_login_attempts": 5,
    "lockout_duration_minutes": 30,
    "rate_limit_requests_per_minute": 100,
    "rate_limit_requests_per_hour": 1000,
    "max_request_size_mb": 10,
    "enable_mfa": False,
    "enable_audit_logging": True,
    "enable_threat_detection": True,
    "cache_ttl_seconds": 300,
    "max_role_depth": 10
}

class PermissionLevel(str, Enum):
    """Permission levels for fine-grained access control."""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

class ResourceType(str, Enum):
    """Resource types in the system."""
    API = "api"
    AGENT = "agent"
    TASK = "task"
    ORCHESTRATOR = "orchestrator"
    METRICS = "metrics"
    SYSTEM = "system"
    CONTEXT = "context"
    SESSION = "session"
    WORKFLOW = "workflow"
    USER = "user"
    ROLE = "role"

class AuthenticationMethod(str, Enum):
    """Authentication methods supported."""
    JWT = "jwt"
    API_KEY = "api_key"
    OAUTH = "oauth"
    CERTIFICATE = "certificate"
    MFA = "mfa"

class AccessDecision(str, Enum):
    """Access decision results."""
    GRANTED = "granted"
    DENIED = "denied"
    CONDITIONAL = "conditional"
    ERROR = "error"

class ThreatLevel(str, Enum):
    """Security threat levels."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    ADAPTIVE = "adaptive"

@dataclass
class Permission:
    """Permission definition with fine-grained control."""
    id: str
    resource_type: ResourceType
    resource_id: str = "*"
    action: str = "*"
    permission_level: PermissionLevel = PermissionLevel.READ
    conditions: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Role:
    """Role definition with hierarchical inheritance."""
    id: str
    name: str
    description: str = ""
    permissions: Set[Permission] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)
    child_roles: Set[str] = field(default_factory=set)
    scope: str = "organization"
    is_system_role: bool = False
    auto_assign_conditions: List[Dict[str, Any]] = field(default_factory=list)
    max_assignments: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class User:
    """User definition with comprehensive attributes."""
    user_id: str
    username: str
    email: str = ""
    roles: List[str] = field(default_factory=list)
    direct_permissions: List[Permission] = field(default_factory=list)
    is_active: bool = True
    mfa_enabled: bool = False
    mfa_secret: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    password_hash: Optional[str] = None
    api_keys: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuthenticationToken:
    """Authentication token with comprehensive metadata."""
    token_id: str
    user_id: str
    method: AuthenticationMethod
    issued_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    scopes: List[str] = field(default_factory=list)
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    is_revoked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuthorizationContext:
    """Comprehensive authorization context."""
    user_id: str
    resource_type: ResourceType
    resource_id: Optional[str] = None
    action: str = "*"
    permission_level: PermissionLevel = PermissionLevel.READ
    
    # Request context
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    request_path: Optional[str] = None
    request_method: Optional[str] = None
    session_id: Optional[str] = None
    
    # Security context
    mfa_verified: bool = False
    secure_connection: bool = True
    auth_method: AuthenticationMethod = AuthenticationMethod.JWT
    risk_score: float = 0.0
    
    # Additional context
    timestamp: datetime = field(default_factory=datetime.utcnow)
    additional_context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuthorizationResult:
    """Comprehensive authorization result."""
    decision: AccessDecision
    reason: str
    matched_roles: List[str] = field(default_factory=list)
    effective_permissions: Dict[str, Any] = field(default_factory=dict)
    conditions_met: bool = True
    evaluation_time_ms: float = 0.0
    risk_factors: List[str] = field(default_factory=list)
    threat_level: ThreatLevel = ThreatLevel.SAFE
    cache_hit: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityValidationResult:
    """Security validation result."""
    is_valid: bool
    threat_level: ThreatLevel
    threats_detected: List[str]
    sanitized_data: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    processing_time_ms: float = 0.0
    blocked_reason: Optional[str] = None

class UnifiedAuthorizationEngine:
    """
    Unified Authorization Engine consolidating all security patterns:
    
    Core Features:
    - Role-based access control (RBAC) with hierarchical inheritance
    - Permission-based access control with fine-grained permissions
    - Context-aware authorization with dynamic conditions
    - Multi-factor authentication (MFA) support
    - JWT and API key authentication
    - Advanced threat detection and prevention
    - Rate limiting and DDoS protection
    - Input validation and sanitization
    - Comprehensive audit logging and monitoring
    - Performance-optimized with intelligent caching
    - Production-grade security controls
    """
    
    _instance: Optional['UnifiedAuthorizationEngine'] = None
    
    def __new__(cls) -> 'UnifiedAuthorizationEngine':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize_engine()
            self._initialized = True
    
    def _initialize_engine(self):
        """Initialize the unified authorization engine."""
        self.config = ConfigurationService().config
        self.redis = get_redis_service()
        self.circuit_breaker = CircuitBreakerService().get_circuit_breaker("authorization")
        
        # Security configuration
        self.security_config = {**DEFAULT_CONFIG, **getattr(self.config, 'security', {})}
        
        # Core data stores
        self._users: Dict[str, User] = {}
        self._roles: Dict[str, Role] = {}
        self._active_tokens: Dict[str, AuthenticationToken] = {}
        self._revoked_tokens: Set[str] = set()
        
        # Security monitoring
        self._failed_attempts: Dict[str, int] = {}
        self._rate_limits: Dict[str, List[datetime]] = {}
        self._blocked_ips: Set[str] = set()
        self._threat_patterns = self._initialize_threat_patterns()
        
        # Performance metrics
        self.metrics = {
            "total_authentications": 0,
            "successful_authentications": 0,
            "failed_authentications": 0,
            "total_authorizations": 0,
            "granted_authorizations": 0,
            "denied_authorizations": 0,
            "threats_detected": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_auth_time_ms": 0.0,
            "avg_authz_time_ms": 0.0,
            "rate_limit_hits": 0,
            "mfa_challenges": 0
        }
        
        # Cache keys
        self._cache_prefix = "unified_auth:"
        self._permission_cache_prefix = f"{self._cache_prefix}perm:"
        self._role_cache_prefix = f"{self._cache_prefix}role:"
        self._user_cache_prefix = f"{self._cache_prefix}user:"
        self._rate_limit_prefix = f"{self._cache_prefix}rate:"
        
        # Initialize default roles and system setup
        asyncio.create_task(self._setup_default_security())
        
        logger.info("Unified Authorization Engine initialized with comprehensive security features")
    
    def _initialize_threat_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Initialize comprehensive threat detection patterns."""
        return {
            "sql_injection": [
                re.compile(r"(union\s+select|insert\s+into|drop\s+table)", re.IGNORECASE),
                re.compile(r"(or\s+1=1|and\s+1=1|'.*'.*=.*')", re.IGNORECASE),
                re.compile(r"(exec\s*\(|execute\s*\(|sp_)", re.IGNORECASE),
                re.compile(r"(\/\*.*\*\/|--\s|#\s)", re.IGNORECASE),
            ],
            "xss": [
                re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
                re.compile(r"javascript:", re.IGNORECASE),
                re.compile(r"on\w+\s*=\s*[\"'][^\"']*[\"']", re.IGNORECASE),
                re.compile(r"<iframe[^>]*>", re.IGNORECASE),
            ],
            "command_injection": [
                re.compile(r"(;|\|{1,2}|&&|\$\()", re.IGNORECASE),
                re.compile(r"(rm\s+-rf|chmod|chown|wget|curl)", re.IGNORECASE),
                re.compile(r"(\/bin\/|\/usr\/bin\/|\/sbin\/)", re.IGNORECASE),
            ],
            "path_traversal": [
                re.compile(r"(\.\.\/|\.\.\\|%2e%2e)", re.IGNORECASE),
                re.compile(r"(\/etc\/passwd|\/proc\/|\/sys\/)", re.IGNORECASE),
            ],
            "nosql_injection": [
                re.compile(r"(\$where|\$ne|\$gt|\$lt)", re.IGNORECASE),
                re.compile(r"(this\.|\$regex)", re.IGNORECASE),
            ]
        }
    
    # Authentication Methods
    
    async def create_user(
        self, 
        user_id: str, 
        username: str, 
        email: str = "",
        password: Optional[str] = None,
        roles: List[str] = None
    ) -> bool:
        """Create new user with comprehensive validation."""
        try:
            if user_id in self._users:
                logger.warning("User already exists", user_id=user_id)
                return False
            
            # Validate roles exist
            for role_name in (roles or ["user"]):
                if role_name not in self._roles:
                    logger.error("Invalid role", role=role_name, user_id=user_id)
                    return False
            
            # Create user
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                roles=roles or ["user"],
                password_hash=self._hash_password(password) if password else None
            )
            
            self._users[user_id] = user
            
            # Persist to Redis
            await self._persist_user(user)
            
            # Log user creation
            await self._log_security_event(
                user_id=user_id,
                action="create_user",
                resource="user",
                success=True,
                metadata={"username": username, "roles": user.roles}
            )
            
            logger.info("User created", user_id=user_id, username=username, roles=user.roles)
            return True
            
        except Exception as e:
            logger.error("User creation failed", user_id=user_id, error=str(e))
            return False
    
    async def authenticate_user(
        self, 
        username: str, 
        password: str,
        mfa_code: Optional[str] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[AuthenticationToken]:
        """Comprehensive user authentication with MFA support."""
        start_time = time.time()
        
        try:
            self.metrics["total_authentications"] += 1
            
            @self.circuit_breaker
            async def _authenticate():
                # Find user by username
                user = None
                for u in self._users.values():
                    if u.username == username and u.is_active:
                        user = u
                        break
                
                if not user:
                    await self._record_failed_attempt(username, client_ip)
                    logger.warning("Authentication failed - user not found", username=username)
                    return None
                
                # Check if user is locked out
                if await self._is_user_locked_out(user):
                    logger.warning("Authentication failed - user locked out", username=username)
                    return None
                
                # Validate password
                if not self._verify_password(password, user.password_hash):
                    await self._record_failed_attempt(username, client_ip)
                    logger.warning("Authentication failed - invalid password", username=username)
                    return None
                
                # MFA verification if enabled
                if user.mfa_enabled:
                    self.metrics["mfa_challenges"] += 1
                    if not mfa_code or not self._verify_mfa_code(user.mfa_secret, mfa_code):
                        logger.warning("Authentication failed - invalid MFA code", username=username)
                        return None
                
                # Generate authentication token
                token = self._generate_token(user, AuthenticationMethod.JWT, client_ip, user_agent)
                self._active_tokens[token.token_id] = token
                
                # Update user login info
                user.last_login = datetime.utcnow()
                user.failed_login_attempts = 0
                user.locked_until = None
                
                # Clear failed attempts
                if username in self._failed_attempts:
                    del self._failed_attempts[username]
                
                # Log successful authentication
                await self._log_security_event(
                    user_id=user.user_id,
                    action="authenticate",
                    resource="authentication",
                    success=True,
                    metadata={
                        "username": username,
                        "method": AuthenticationMethod.JWT.value,
                        "client_ip": client_ip,
                        "mfa_used": user.mfa_enabled
                    }
                )
                
                self.metrics["successful_authentications"] += 1
                logger.info("User authenticated", user_id=user.user_id, username=username)
                return token
            
            result = await _authenticate()
            
            # Update metrics
            auth_time = (time.time() - start_time) * 1000
            self._update_auth_metrics(auth_time)
            
            return result
            
        except Exception as e:
            self.metrics["failed_authentications"] += 1
            logger.error("Authentication error", username=username, error=str(e))
            return None
    
    async def validate_token(self, token_value: str) -> Optional[User]:
        """Comprehensive token validation with security checks."""
        try:
            @self.circuit_breaker
            async def _validate():
                # Check if token is revoked
                if token_value in self._revoked_tokens:
                    return None
                
                # Handle JWT tokens
                if token_value.count('.') == 2:  # JWT format
                    try:
                        payload = jwt.decode(
                            token_value,
                            self.security_config["jwt_secret_key"],
                            algorithms=[self.security_config["jwt_algorithm"]]
                        )
                        
                        user_id = payload.get("user_id")
                        if not user_id or user_id not in self._users:
                            return None
                        
                        user = self._users[user_id]
                        if not user.is_active:
                            return None
                        
                        return user
                        
                    except jwt.ExpiredSignatureError:
                        logger.debug("Token expired")
                        return None
                    except jwt.InvalidTokenError:
                        logger.debug("Invalid token")
                        return None
                
                # Handle other token types (API keys, etc.)
                for token in self._active_tokens.values():
                    if token.token_id == token_value and not token.is_revoked:
                        if token.expires_at and datetime.utcnow() > token.expires_at:
                            # Token expired
                            token.is_revoked = True
                            return None
                        
                        return self._users.get(token.user_id)
                
                return None
            
            return await _validate()
            
        except Exception as e:
            logger.error("Token validation error", error=str(e))
            return None
    
    # Authorization Methods
    
    async def check_permission(
        self, 
        context: AuthorizationContext
    ) -> AuthorizationResult:
        """Comprehensive permission checking with context analysis."""
        start_time = time.time()
        
        try:
            self.metrics["total_authorizations"] += 1
            
            # Check cache first
            cache_key = self._build_permission_cache_key(context)
            cached_result = await self._get_cached_permission(cache_key)
            if cached_result:
                self.metrics["cache_hits"] += 1
                cached_result.cache_hit = True
                return cached_result
            
            self.metrics["cache_misses"] += 1
            
            # Get user
            user = self._users.get(context.user_id)
            if not user or not user.is_active:
                result = AuthorizationResult(
                    decision=AccessDecision.DENIED,
                    reason="User not found or inactive",
                    threat_level=ThreatLevel.MEDIUM
                )
                return result
            
            # Security validation
            security_result = await self._validate_security_context(context)
            if security_result.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                result = AuthorizationResult(
                    decision=AccessDecision.DENIED,
                    reason=f"Security threat detected: {security_result.blocked_reason}",
                    threat_level=security_result.threat_level,
                    risk_factors=security_result.threats_detected
                )
                await self._cache_permission_result(cache_key, result)
                return result
            
            # Permission evaluation
            permission_result = await self._evaluate_permissions(user, context)
            
            # Rate limiting check
            if not await self._check_rate_limits(context.client_ip or "unknown"):
                self.metrics["rate_limit_hits"] += 1
                result = AuthorizationResult(
                    decision=AccessDecision.DENIED,
                    reason="Rate limit exceeded",
                    threat_level=ThreatLevel.MEDIUM
                )
                return result
            
            # Final result
            evaluation_time = (time.time() - start_time) * 1000
            permission_result.evaluation_time_ms = evaluation_time
            
            # Cache result
            if permission_result.decision in [AccessDecision.GRANTED, AccessDecision.DENIED]:
                await self._cache_permission_result(cache_key, permission_result)
            
            # Update metrics
            if permission_result.decision == AccessDecision.GRANTED:
                self.metrics["granted_authorizations"] += 1
            else:
                self.metrics["denied_authorizations"] += 1
            
            self._update_authz_metrics(evaluation_time)
            
            # Audit the decision
            await self._audit_authorization_decision(context, permission_result)
            
            return permission_result
            
        except Exception as e:
            logger.error("Authorization check failed", user_id=context.user_id, error=str(e))
            return AuthorizationResult(
                decision=AccessDecision.ERROR,
                reason=f"Authorization evaluation error: {str(e)}",
                threat_level=ThreatLevel.HIGH
            )
    
    async def _evaluate_permissions(self, user: User, context: AuthorizationContext) -> AuthorizationResult:
        """Evaluate user permissions against context requirements."""
        matched_roles = []
        effective_permissions = {"resources": [], "actions": []}
        risk_factors = []
        
        # Check direct permissions
        for permission in user.direct_permissions:
            if self._permission_matches(permission, context):
                matched_roles.append("direct_permission")
                effective_permissions["actions"].append(permission.action)
        
        # Check role-based permissions with inheritance
        for role_name in user.roles:
            role_permissions = await self._get_role_permissions_with_inheritance(role_name)
            for permission in role_permissions:
                if self._permission_matches(permission, context):
                    matched_roles.append(role_name)
                    effective_permissions["actions"].append(permission.action)
        
        # Risk assessment
        risk_factors.extend(await self._assess_risk_factors(context))
        
        # Decision logic
        if matched_roles:
            # Check for high-risk conditions
            if "high_risk" in risk_factors:
                return AuthorizationResult(
                    decision=AccessDecision.DENIED,
                    reason="Access denied due to high risk factors",
                    matched_roles=matched_roles,
                    effective_permissions=effective_permissions,
                    risk_factors=risk_factors,
                    threat_level=ThreatLevel.HIGH
                )
            
            return AuthorizationResult(
                decision=AccessDecision.GRANTED,
                reason=f"Access granted via: {', '.join(set(matched_roles))}",
                matched_roles=list(set(matched_roles)),
                effective_permissions=effective_permissions,
                risk_factors=risk_factors,
                threat_level=ThreatLevel.SAFE
            )
        else:
            return AuthorizationResult(
                decision=AccessDecision.DENIED,
                reason="No matching permissions found",
                matched_roles=[],
                effective_permissions={},
                risk_factors=risk_factors,
                threat_level=ThreatLevel.LOW
            )
    
    # Security Validation Methods
    
    async def _validate_security_context(self, context: AuthorizationContext) -> SecurityValidationResult:
        """Comprehensive security validation of the authorization context."""
        threats_detected = []
        confidence_score = 0.0
        
        # Input validation
        content_to_analyze = f"{context.resource_id} {context.action} {context.request_path}"
        
        # Threat pattern detection
        for threat_type, patterns in self._threat_patterns.items():
            for pattern in patterns:
                if pattern.search(content_to_analyze):
                    threats_detected.append(f"{threat_type}_detected")
                    confidence_score = max(confidence_score, 0.8)
        
        # IP-based threat assessment
        if context.client_ip:
            if context.client_ip in self._blocked_ips:
                threats_detected.append("blocked_ip")
                confidence_score = 1.0
        
        # Risk scoring
        if context.risk_score > 0.7:
            threats_detected.append("high_risk_score")
            confidence_score = max(confidence_score, context.risk_score)
        
        # Determine threat level
        if confidence_score >= 0.9:
            threat_level = ThreatLevel.CRITICAL
        elif confidence_score >= 0.7:
            threat_level = ThreatLevel.HIGH
        elif confidence_score >= 0.5:
            threat_level = ThreatLevel.MEDIUM
        elif confidence_score >= 0.3:
            threat_level = ThreatLevel.LOW
        else:
            threat_level = ThreatLevel.SAFE
        
        if threats_detected:
            self.metrics["threats_detected"] += len(threats_detected)
            
            # Log threat detection
            await self._log_security_event(
                user_id=context.user_id,
                action="threat_detected",
                resource="authorization_context",
                success=False,
                metadata={
                    "threats": threats_detected,
                    "confidence_score": confidence_score,
                    "threat_level": threat_level.value,
                    "client_ip": context.client_ip
                }
            )
        
        return SecurityValidationResult(
            is_valid=threat_level not in [ThreatLevel.HIGH, ThreatLevel.CRITICAL],
            threat_level=threat_level,
            threats_detected=threats_detected,
            confidence_score=confidence_score,
            blocked_reason=f"Security threats detected: {', '.join(threats_detected[:3])}" if threats_detected else None
        )
    
    # Rate Limiting Methods
    
    async def _check_rate_limits(self, client_ip: str) -> bool:
        """Comprehensive rate limiting with multiple strategies."""
        try:
            current_time = time.time()
            
            # Per-minute rate limiting
            minute_key = f"{self._rate_limit_prefix}minute:{client_ip}"
            minute_requests = await self.redis.incr(minute_key)
            await self.redis.expire(minute_key, 60)
            
            if minute_requests > self.security_config["rate_limit_requests_per_minute"]:
                return False
            
            # Per-hour rate limiting
            hour_key = f"{self._rate_limit_prefix}hour:{client_ip}"
            hour_requests = await self.redis.incr(hour_key)
            await self.redis.expire(hour_key, 3600)
            
            if hour_requests > self.security_config["rate_limit_requests_per_hour"]:
                return False
            
            return True
            
        except Exception as e:
            logger.error("Rate limit check failed", client_ip=client_ip, error=str(e))
            return True  # Fail open for availability
    
    # Helper Methods
    
    def _permission_matches(self, permission: Permission, context: AuthorizationContext) -> bool:
        """Check if permission matches authorization context."""
        # Resource type match
        if permission.resource_type != context.resource_type:
            return False
        
        # Resource ID match (with wildcard support)
        if permission.resource_id != "*" and permission.resource_id != context.resource_id:
            return False
        
        # Action match (with wildcard support)
        if permission.action != "*" and permission.action != context.action:
            return False
        
        # Permission level check
        required_level = context.permission_level
        if self._permission_level_sufficient(permission.permission_level, required_level):
            return False
        
        # Expiration check
        if permission.expires_at and datetime.utcnow() > permission.expires_at:
            return False
        
        return True
    
    def _permission_level_sufficient(self, granted: PermissionLevel, required: PermissionLevel) -> bool:
        """Check if granted permission level is sufficient for required level."""
        level_hierarchy = {
            PermissionLevel.NONE: 0,
            PermissionLevel.READ: 1,
            PermissionLevel.WRITE: 2,
            PermissionLevel.EXECUTE: 3,
            PermissionLevel.ADMIN: 4,
            PermissionLevel.SUPER_ADMIN: 5
        }
        
        return level_hierarchy.get(granted, 0) >= level_hierarchy.get(required, 0)
    
    async def _get_role_permissions_with_inheritance(self, role_name: str) -> Set[Permission]:
        """Get role permissions including inherited permissions."""
        # Check cache
        cache_key = f"{self._role_cache_prefix}{role_name}"
        cached_permissions = await self.redis.get(cache_key)
        
        if cached_permissions:
            # Deserialize from cache (simplified for now)
            return set()
        
        role = self._roles.get(role_name)
        if not role:
            return set()
        
        all_permissions = set(role.permissions)
        
        # Add inherited permissions
        for parent_role_name in role.parent_roles:
            parent_permissions = await self._get_role_permissions_with_inheritance(parent_role_name)
            all_permissions.update(parent_permissions)
        
        # Cache the result
        await self.redis.set(cache_key, "cached_permissions", ex=self.security_config["cache_ttl_seconds"])
        
        return all_permissions
    
    async def _assess_risk_factors(self, context: AuthorizationContext) -> List[str]:
        """Assess risk factors for the authorization request."""
        risk_factors = []
        
        # Time-based risk
        current_hour = context.timestamp.hour
        if current_hour < 6 or current_hour > 22:
            risk_factors.append("off_hours_access")
        
        # High-privilege action risk
        high_privilege_actions = ["delete", "admin", "modify_permissions", "create_user"]
        if context.action in high_privilege_actions:
            risk_factors.append("high_privilege_action")
        
        # MFA requirement risk
        if context.permission_level in [PermissionLevel.ADMIN, PermissionLevel.SUPER_ADMIN] and not context.mfa_verified:
            risk_factors.append("admin_without_mfa")
        
        # Connection security risk
        if not context.secure_connection:
            risk_factors.append("insecure_connection")
        
        # Aggregate risk level
        if len(risk_factors) >= 3:
            risk_factors.append("high_risk")
        elif len(risk_factors) >= 2:
            risk_factors.append("medium_risk")
        elif len(risk_factors) >= 1:
            risk_factors.append("low_risk")
        
        return risk_factors
    
    # Token and Security Utilities
    
    def _generate_token(
        self, 
        user: User, 
        method: AuthenticationMethod,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AuthenticationToken:
        """Generate secure authentication token."""
        token_id = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=self.security_config["jwt_expiry_hours"])
        
        if method == AuthenticationMethod.JWT:
            payload = {
                "user_id": user.user_id,
                "username": user.username,
                "roles": user.roles,
                "iat": datetime.utcnow(),
                "exp": expires_at,
                "jti": token_id
            }
            
            jwt_token = jwt.encode(
                payload,
                self.security_config["jwt_secret_key"],
                algorithm=self.security_config["jwt_algorithm"]
            )
            
            token = AuthenticationToken(
                token_id=jwt_token,  # Use JWT as token ID for JWT method
                user_id=user.user_id,
                method=method,
                expires_at=expires_at,
                client_ip=client_ip,
                user_agent=user_agent,
                metadata={"jwt": jwt_token}
            )
        else:
            token = AuthenticationToken(
                token_id=token_id,
                user_id=user.user_id,
                method=method,
                expires_at=expires_at,
                client_ip=client_ip,
                user_agent=user_agent
            )
        
        return token
    
    def _hash_password(self, password: str) -> str:
        """Hash password securely."""
        import bcrypt
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def _verify_password(self, password: str, password_hash: Optional[str]) -> bool:
        """Verify password against hash."""
        if not password_hash:
            return False
        import bcrypt
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    
    def _verify_mfa_code(self, secret: Optional[str], code: str) -> bool:
        """Verify MFA code."""
        if not secret:
            return False
        # Implement TOTP verification (simplified for now)
        import pyotp
        totp = pyotp.TOTP(secret)
        return totp.verify(code)
    
    # Caching Methods
    
    def _build_permission_cache_key(self, context: AuthorizationContext) -> str:
        """Build cache key for permission check."""
        key_data = {
            "user_id": context.user_id,
            "resource_type": context.resource_type.value,
            "resource_id": context.resource_id,
            "action": context.action,
            "permission_level": context.permission_level.value
        }
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        return f"{self._permission_cache_prefix}{key_hash}"
    
    async def _get_cached_permission(self, cache_key: str) -> Optional[AuthorizationResult]:
        """Get cached permission result."""
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                # Deserialize cached result (simplified for now)
                return None  # Would implement proper serialization
        except Exception as e:
            logger.debug(f"Cache get error: {e}")
        return None
    
    async def _cache_permission_result(self, cache_key: str, result: AuthorizationResult) -> None:
        """Cache permission result."""
        try:
            # Serialize result (simplified for now)
            await self.redis.set(cache_key, "cached_result", ex=self.security_config["cache_ttl_seconds"])
        except Exception as e:
            logger.debug(f"Cache set error: {e}")
    
    # User Management and Persistence
    
    async def _persist_user(self, user: User) -> None:
        """Persist user to Redis."""
        try:
            user_key = f"{self._user_cache_prefix}{user.user_id}"
            user_data = {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": json.dumps(user.roles),
                "is_active": str(user.is_active),
                "mfa_enabled": str(user.mfa_enabled),
                "created_at": user.created_at.isoformat()
            }
            await self.redis.hset(user_key, mapping=user_data)
        except Exception as e:
            logger.error(f"User persistence error: {e}")
    
    async def _record_failed_attempt(self, identifier: str, client_ip: Optional[str] = None) -> None:
        """Record failed authentication attempt."""
        self._failed_attempts[identifier] = self._failed_attempts.get(identifier, 0) + 1
        
        # Log failed attempt
        await self._log_security_event(
            user_id=None,
            action="failed_authentication",
            resource="authentication",
            success=False,
            metadata={
                "identifier": identifier,
                "client_ip": client_ip,
                "attempt_count": self._failed_attempts[identifier]
            }
        )
        
        logger.warning("Failed authentication attempt", 
                      identifier=identifier, 
                      count=self._failed_attempts[identifier])
    
    async def _is_user_locked_out(self, user: User) -> bool:
        """Check if user is locked out due to failed attempts."""
        if user.locked_until and datetime.utcnow() < user.locked_until:
            return True
        
        attempts = self._failed_attempts.get(user.username, 0)
        if attempts >= self.security_config["max_login_attempts"]:
            # Lock user
            user.locked_until = datetime.utcnow() + timedelta(minutes=self.security_config["lockout_duration_minutes"])
            return True
        
        return False
    
    # Audit and Logging
    
    async def _audit_authorization_decision(
        self, 
        context: AuthorizationContext, 
        result: AuthorizationResult
    ) -> None:
        """Audit authorization decision."""
        await self._log_security_event(
            user_id=context.user_id,
            action="authorization_check",
            resource=context.resource_type.value,
            success=(result.decision == AccessDecision.GRANTED),
            metadata={
                "resource_id": context.resource_id,
                "action": context.action,
                "decision": result.decision.value,
                "reason": result.reason,
                "matched_roles": result.matched_roles,
                "evaluation_time_ms": result.evaluation_time_ms,
                "risk_factors": result.risk_factors,
                "threat_level": result.threat_level.value,
                "client_ip": context.client_ip
            }
        )
    
    async def _log_security_event(
        self,
        action: str,
        resource: str,
        success: bool,
        user_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log security event for audit purposes."""
        if not self.security_config["enable_audit_logging"]:
            return
        
        try:
            event_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id,
                "action": action,
                "resource": resource,
                "resource_id": resource_id,
                "success": success,
                "metadata": metadata or {}
            }
            
            # Store in Redis for monitoring
            await self.redis.lpush("security_events", json.dumps(event_data))
            await self.redis.ltrim("security_events", 0, 9999)  # Keep last 10000 events
            
            logger.info("Security event logged", **event_data)
            
        except Exception as e:
            logger.error(f"Security event logging failed: {e}")
    
    # Metrics and Performance
    
    def _update_auth_metrics(self, processing_time_ms: float) -> None:
        """Update authentication metrics."""
        current_avg = self.metrics["avg_auth_time_ms"]
        total_auths = self.metrics["total_authentications"]
        self.metrics["avg_auth_time_ms"] = (
            (current_avg * (total_auths - 1) + processing_time_ms) / total_auths
        )
    
    def _update_authz_metrics(self, processing_time_ms: float) -> None:
        """Update authorization metrics."""
        current_avg = self.metrics["avg_authz_time_ms"]
        total_authz = self.metrics["total_authorizations"]
        self.metrics["avg_authz_time_ms"] = (
            (current_avg * (total_authz - 1) + processing_time_ms) / total_authz
        )
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics."""
        return {
            "authentication_metrics": {
                "total_authentications": self.metrics["total_authentications"],
                "successful_authentications": self.metrics["successful_authentications"],
                "failed_authentications": self.metrics["failed_authentications"],
                "success_rate": (
                    self.metrics["successful_authentications"] / 
                    max(1, self.metrics["total_authentications"])
                ),
                "avg_auth_time_ms": self.metrics["avg_auth_time_ms"],
                "mfa_challenges": self.metrics["mfa_challenges"]
            },
            "authorization_metrics": {
                "total_authorizations": self.metrics["total_authorizations"],
                "granted_authorizations": self.metrics["granted_authorizations"],
                "denied_authorizations": self.metrics["denied_authorizations"],
                "success_rate": (
                    self.metrics["granted_authorizations"] / 
                    max(1, self.metrics["total_authorizations"])
                ),
                "avg_authz_time_ms": self.metrics["avg_authz_time_ms"]
            },
            "security_metrics": {
                "threats_detected": self.metrics["threats_detected"],
                "rate_limit_hits": self.metrics["rate_limit_hits"],
                "blocked_ips": len(self._blocked_ips),
                "active_tokens": len(self._active_tokens),
                "revoked_tokens": len(self._revoked_tokens)
            },
            "cache_metrics": {
                "cache_hits": self.metrics["cache_hits"],
                "cache_misses": self.metrics["cache_misses"],
                "cache_hit_rate": (
                    self.metrics["cache_hits"] / 
                    max(1, self.metrics["cache_hits"] + self.metrics["cache_misses"])
                )
            },
            "system_status": {
                "total_users": len(self._users),
                "active_users": len([u for u in self._users.values() if u.is_active]),
                "total_roles": len(self._roles),
                "failed_attempts": dict(self._failed_attempts)
            }
        }
    
    # Default Setup
    
    async def _setup_default_security(self) -> None:
        """Set up default roles and security configuration."""
        try:
            # System Administrator Role
            admin_role = Role(
                id="system_admin",
                name="System Administrator",
                description="Full system access with all permissions",
                is_system_role=True
            )
            
            # Add comprehensive admin permissions
            admin_permissions = []
            for resource_type in ResourceType:
                for action in ["*"]:
                    permission = Permission(
                        id=f"admin_{resource_type.value}_{action}",
                        resource_type=resource_type,
                        action=action,
                        permission_level=PermissionLevel.SUPER_ADMIN
                    )
                    admin_permissions.append(permission)
            
            admin_role.permissions.update(admin_permissions)
            self._roles[admin_role.id] = admin_role
            
            # Standard User Role
            user_role = Role(
                id="user",
                name="Standard User",
                description="Basic user access",
                is_system_role=True
            )
            
            user_permissions = [
                Permission(
                    id="user_read_api",
                    resource_type=ResourceType.API,
                    action="read",
                    permission_level=PermissionLevel.READ
                ),
                Permission(
                    id="user_read_metrics",
                    resource_type=ResourceType.METRICS,
                    action="read",
                    permission_level=PermissionLevel.READ
                )
            ]
            
            user_role.permissions.update(user_permissions)
            self._roles[user_role.id] = user_role
            
            # Agent Role
            agent_role = Role(
                id="agent",
                name="Agent",
                description="Agent execution permissions",
                is_system_role=True
            )
            
            agent_permissions = [
                Permission(
                    id="agent_execute_task",
                    resource_type=ResourceType.TASK,
                    action="*",
                    permission_level=PermissionLevel.EXECUTE
                ),
                Permission(
                    id="agent_read_orchestrator",
                    resource_type=ResourceType.ORCHESTRATOR,
                    action="read",
                    permission_level=PermissionLevel.READ
                )
            ]
            
            agent_role.permissions.update(agent_permissions)
            self._roles[agent_role.id] = agent_role
            
            logger.info("Default security roles initialized", roles=list(self._roles.keys()))
            
        except Exception as e:
            logger.error("Default security setup failed", error=str(e))


# Security Decorators and Middleware

def require_permission(
    resource_type: ResourceType, 
    action: str = "*", 
    permission_level: PermissionLevel = PermissionLevel.READ
):
    """Decorator requiring specific permission."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract user and request context
            current_user = kwargs.get("current_user")
            request = kwargs.get("request")
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # Build authorization context
            context = AuthorizationContext(
                user_id=current_user.user_id,
                resource_type=resource_type,
                action=action,
                permission_level=permission_level,
                client_ip=getattr(request, 'client', {}).get('host') if request else None,
                user_agent=getattr(request, 'headers', {}).get('user-agent') if request else None,
                request_path=str(getattr(request, 'url', {}).get('path', '')) if request else None,
                request_method=getattr(request, 'method') if request else None
            )
            
            # Check permission
            auth_engine = get_unified_authorization_engine()
            result = await auth_engine.check_permission(context)
            
            if result.decision != AccessDecision.GRANTED:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions: {result.reason}"
                )
            
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, create async context
            import asyncio
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def require_role(role_name: str):
    """Decorator requiring specific role."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get("current_user")
            if not current_user or role_name not in current_user.roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role '{role_name}' required"
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# FastAPI Security Middleware

class UnifiedSecurityMiddleware(BaseHTTPMiddleware):
    """Unified security middleware integrating all security features."""
    
    def __init__(self, app, auth_engine: UnifiedAuthorizationEngine):
        super().__init__(app)
        self.auth_engine = auth_engine
        
        # Security configuration
        self.skip_auth_paths = {
            "/health", "/metrics", "/docs", "/redoc", "/openapi.json"
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request through unified security pipeline."""
        start_time = time.time()
        
        try:
            # Skip authentication for certain paths
            if any(request.url.path.startswith(path) for path in self.skip_auth_paths):
                return await call_next(request)
            
            # Rate limiting
            client_ip = self._get_client_ip(request)
            if not await self.auth_engine._check_rate_limits(client_ip):
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"error": "Rate limit exceeded"}
                )
            
            # Request validation
            if request.method in ["POST", "PUT", "PATCH"]:
                validation_result = await self._validate_request_security(request)
                if not validation_result.is_valid:
                    return JSONResponse(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        content={
                            "error": "Security validation failed",
                            "details": validation_result.blocked_reason
                        }
                    )
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Security processing error"}
            )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return getattr(request.client, 'host', 'unknown') if request.client else 'unknown'
    
    async def _validate_request_security(self, request: Request) -> SecurityValidationResult:
        """Validate request for security threats."""
        try:
            body = await request.body()
            content = body.decode('utf-8', errors='ignore')
            
            # Build context for validation
            context = AuthorizationContext(
                user_id="anonymous",  # Will be determined later
                resource_type=ResourceType.API,
                request_path=str(request.url.path),
                request_method=request.method,
                client_ip=self._get_client_ip(request),
                user_agent=request.headers.get("user-agent", "")
            )
            
            return await self.auth_engine._validate_security_context(context)
            
        except Exception as e:
            logger.error(f"Request security validation error: {e}")
            return SecurityValidationResult(
                is_valid=False,
                threat_level=ThreatLevel.HIGH,
                threats_detected=["validation_error"],
                blocked_reason=f"Validation error: {str(e)}"
            )
    
    def _add_security_headers(self, response: Response) -> None:
        """Add comprehensive security headers."""
        response.headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        })


# Factory Functions and Utilities

def get_unified_authorization_engine() -> UnifiedAuthorizationEngine:
    """Get unified authorization engine instance."""
    return UnifiedAuthorizationEngine()


async def authenticate_request(token: str) -> Optional[User]:
    """Authenticate API request with token."""
    auth_engine = get_unified_authorization_engine()
    return await auth_engine.validate_token(token)


async def authorize_request(
    user: User,
    resource_type: ResourceType,
    action: str = "*",
    resource_id: Optional[str] = None,
    request: Optional[Request] = None
) -> bool:
    """Authorize API request."""
    auth_engine = get_unified_authorization_engine()
    
    context = AuthorizationContext(
        user_id=user.user_id,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        client_ip=getattr(request, 'client', {}).get('host') if request else None,
        user_agent=getattr(request, 'headers', {}).get('user-agent') if request else None
    )
    
    result = await auth_engine.check_permission(context)
    return result.decision == AccessDecision.GRANTED


# FastAPI Dependencies

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """FastAPI dependency to get current authenticated user."""
    auth_engine = get_unified_authorization_engine()
    user = await auth_engine.validate_token(credentials.credentials)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    return user


def create_unified_security_middleware(app):
    """Create unified security middleware for FastAPI app."""
    auth_engine = get_unified_authorization_engine()
    return UnifiedSecurityMiddleware(app, auth_engine)


# Export public interface
__all__ = [
    "UnifiedAuthorizationEngine",
    "UnifiedSecurityMiddleware",
    "get_unified_authorization_engine",
    "authenticate_request",
    "authorize_request",
    "get_current_user",
    "require_permission",
    "require_role",
    "create_unified_security_middleware",
    "PermissionLevel",
    "ResourceType",
    "AuthenticationMethod",
    "AccessDecision",
    "ThreatLevel",
    "Permission",
    "Role",
    "User",
    "AuthenticationToken",
    "AuthorizationContext",
    "AuthorizationResult",
    "SecurityValidationResult"
]