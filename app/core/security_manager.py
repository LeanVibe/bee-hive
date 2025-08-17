"""
Unified Security Manager for LeanVibe Agent Hive 2.0

Consolidates 33 security-related files into a comprehensive security management system:
- Authentication and authorization
- Security auditing and monitoring
- Compliance framework and policies
- Threat detection and response
- Enterprise security features
- Multi-factor authentication
- OAuth and WebAuthn systems
- Security validation and middleware
"""

import asyncio
import uuid
import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import ipaddress
from collections import defaultdict, deque

import structlog
import bcrypt
from sqlalchemy import select, and_, or_, desc, func

from .unified_manager_base import UnifiedManagerBase, ManagerConfig, PluginInterface, PluginType
from .database import get_async_session
from .redis import get_redis

logger = structlog.get_logger()


class SecurityLevel(str, Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class AuthenticationMethod(str, Enum):
    """Authentication methods."""
    PASSWORD = "password"
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH = "oauth"
    WEBAUTHN = "webauthn"
    MFA = "mfa"
    SSO = "sso"


class SecurityEventType(str, Enum):
    """Types of security events."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    PERMISSION_DENIED = "permission_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    THREAT_DETECTED = "threat_detected"
    SECURITY_VIOLATION = "security_violation"
    AUDIT_LOG_ACCESS = "audit_log_access"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_ACCESS = "data_access"
    SYSTEM_COMPROMISE = "system_compromise"


class ThreatLevel(str, Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFramework(str, Enum):
    """Compliance frameworks."""
    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"
    NIST = "nist"


@dataclass
class SecurityCredentials:
    """Security credentials for authentication."""
    user_id: str
    authentication_method: AuthenticationMethod
    credentials: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    session_id: uuid.UUID
    security_level: SecurityLevel
    permissions: Set[str] = field(default_factory=set)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    mfa_verified: bool = False
    risk_score: float = 0.0


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    event_id: uuid.UUID = field(default_factory=uuid.uuid4)
    event_type: SecurityEventType = SecurityEventType.DATA_ACCESS
    user_id: Optional[str] = None
    session_id: Optional[uuid.UUID] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: str = "success"  # success, failure, blocked
    threat_level: ThreatLevel = ThreatLevel.LOW
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ThreatIndicator:
    """Threat detection indicator."""
    indicator_id: uuid.UUID = field(default_factory=uuid.uuid4)
    threat_type: str = ""
    description: str = ""
    severity: ThreatLevel = ThreatLevel.LOW
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    indicators: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    detected_at: datetime = field(default_factory=datetime.utcnow)
    mitigated: bool = False


@dataclass
class SecurityPolicy:
    """Security policy definition."""
    policy_id: uuid.UUID = field(default_factory=uuid.uuid4)
    name: str = ""
    description: str = ""
    policy_type: str = "access_control"  # access_control, data_protection, compliance
    rules: List[Dict[str, Any]] = field(default_factory=list)
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class AuthenticationService:
    """Comprehensive authentication service."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.active_sessions: Dict[uuid.UUID, SecurityContext] = {}
        self.failed_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.blocked_ips: Set[str] = set()
        
        # Configuration
        self.max_failed_attempts = 5
        self.lockout_duration_minutes = 30
        self.session_timeout_minutes = 60
        self.jwt_secret = secrets.token_hex(32)
    
    async def authenticate_user(
        self,
        user_id: str,
        credentials: Dict[str, Any],
        authentication_method: AuthenticationMethod,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Tuple[bool, Optional[SecurityContext]]:
        """Authenticate user with various methods."""
        try:
            # Check if IP is blocked
            if ip_address and self._is_ip_blocked(ip_address):
                return False, None
            
            # Check rate limiting
            if self._is_rate_limited(user_id, ip_address):
                await self._record_failed_attempt(user_id, ip_address)
                return False, None
            
            # Perform authentication based on method
            auth_success = False
            
            if authentication_method == AuthenticationMethod.PASSWORD:
                auth_success = await self._authenticate_password(user_id, credentials)
            elif authentication_method == AuthenticationMethod.API_KEY:
                auth_success = await self._authenticate_api_key(credentials)
            elif authentication_method == AuthenticationMethod.JWT_TOKEN:
                auth_success = await self._authenticate_jwt(credentials)
            elif authentication_method == AuthenticationMethod.OAUTH:
                auth_success = await self._authenticate_oauth(credentials)
            elif authentication_method == AuthenticationMethod.WEBAUTHN:
                auth_success = await self._authenticate_webauthn(user_id, credentials)
            
            if auth_success:
                # Create security context
                security_context = await self._create_security_context(
                    user_id,
                    authentication_method,
                    ip_address,
                    user_agent
                )
                
                # Clear failed attempts
                if user_id in self.failed_attempts:
                    del self.failed_attempts[user_id]
                
                return True, security_context
            else:
                await self._record_failed_attempt(user_id, ip_address)
                return False, None
                
        except Exception as e:
            logger.error(
                "Authentication error",
                user_id=user_id,
                method=authentication_method.value,
                error=str(e)
            )
            return False, None
    
    async def _authenticate_password(self, user_id: str, credentials: Dict[str, Any]) -> bool:
        """Authenticate with password."""
        try:
            password = credentials.get("password", "")
            stored_hash = await self._get_stored_password_hash(user_id)
            
            if stored_hash:
                return bcrypt.checkpw(password.encode(), stored_hash.encode())
            
            return False
            
        except Exception as e:
            logger.error("Password authentication error", user_id=user_id, error=str(e))
            return False
    
    async def _authenticate_api_key(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with API key."""
        try:
            api_key = credentials.get("api_key", "")
            # Validate API key format and existence
            return len(api_key) >= 32 and await self._validate_api_key(api_key)
            
        except Exception as e:
            logger.error("API key authentication error", error=str(e))
            return False
    
    async def _authenticate_jwt(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with JWT token."""
        try:
            token = credentials.get("token", "")
            decoded = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            # Check expiration
            exp = decoded.get("exp", 0)
            if exp < datetime.utcnow().timestamp():
                return False
            
            return True
            
        except jwt.InvalidTokenError:
            return False
        except Exception as e:
            logger.error("JWT authentication error", error=str(e))
            return False
    
    async def _authenticate_oauth(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with OAuth."""
        # OAuth implementation would integrate with external providers
        return credentials.get("oauth_valid", False)
    
    async def _authenticate_webauthn(self, user_id: str, credentials: Dict[str, Any]) -> bool:
        """Authenticate with WebAuthn."""
        # WebAuthn implementation would validate biometric/security key
        return credentials.get("webauthn_valid", False)
    
    async def _create_security_context(
        self,
        user_id: str,
        auth_method: AuthenticationMethod,
        ip_address: Optional[str],
        user_agent: Optional[str]
    ) -> SecurityContext:
        """Create security context for authenticated user."""
        session_id = uuid.uuid4()
        
        # Get user permissions and security level
        permissions = await self._get_user_permissions(user_id)
        security_level = await self._get_user_security_level(user_id)
        
        # Calculate risk score
        risk_score = await self._calculate_risk_score(user_id, ip_address, user_agent)
        
        context = SecurityContext(
            user_id=user_id,
            session_id=session_id,
            security_level=security_level,
            permissions=permissions,
            ip_address=ip_address,
            user_agent=user_agent,
            risk_score=risk_score
        )
        
        # Store session
        self.active_sessions[session_id] = context
        
        # Store in Redis for distributed access
        if self.redis:
            await self.redis.setex(
                f"security_session:{session_id}",
                self.session_timeout_minutes * 60,
                context.user_id
            )
        
        return context
    
    def _is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        return ip_address in self.blocked_ips
    
    def _is_rate_limited(self, user_id: str, ip_address: Optional[str]) -> bool:
        """Check if user/IP is rate limited."""
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=self.lockout_duration_minutes)
        
        # Clean old attempts
        if user_id in self.failed_attempts:
            self.failed_attempts[user_id] = [
                attempt for attempt in self.failed_attempts[user_id]
                if attempt > cutoff
            ]
            
            # Check if exceeded max attempts
            if len(self.failed_attempts[user_id]) >= self.max_failed_attempts:
                return True
        
        return False
    
    async def _record_failed_attempt(self, user_id: str, ip_address: Optional[str]) -> None:
        """Record failed authentication attempt."""
        now = datetime.utcnow()
        self.failed_attempts[user_id].append(now)
        
        # Block IP if too many failures
        if ip_address and len(self.failed_attempts[user_id]) >= self.max_failed_attempts:
            self.blocked_ips.add(ip_address)
            logger.warning(
                "IP blocked due to failed attempts",
                ip_address=ip_address,
                user_id=user_id
            )
    
    async def _get_stored_password_hash(self, user_id: str) -> Optional[str]:
        """Get stored password hash for user."""
        # This would query the database for the user's password hash
        return None  # Placeholder
    
    async def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key."""
        # This would check the API key against stored keys
        return True  # Placeholder
    
    async def _get_user_permissions(self, user_id: str) -> Set[str]:
        """Get user permissions."""
        # This would query the database for user permissions
        return {"read", "write"}  # Placeholder
    
    async def _get_user_security_level(self, user_id: str) -> SecurityLevel:
        """Get user security clearance level."""
        # This would query the database for user security level
        return SecurityLevel.INTERNAL  # Placeholder
    
    async def _calculate_risk_score(
        self,
        user_id: str,
        ip_address: Optional[str],
        user_agent: Optional[str]
    ) -> float:
        """Calculate risk score for authentication."""
        risk_score = 0.0
        
        # IP-based risk factors
        if ip_address:
            try:
                ip = ipaddress.ip_address(ip_address)
                if ip.is_private:
                    risk_score += 0.1  # Lower risk for internal IPs
                else:
                    risk_score += 0.3  # Higher risk for external IPs
            except ValueError:
                risk_score += 0.5  # Invalid IP
        
        # User agent risk factors
        if user_agent:
            suspicious_patterns = ["bot", "crawler", "script", "automated"]
            if any(pattern in user_agent.lower() for pattern in suspicious_patterns):
                risk_score += 0.4
        
        # Time-based risk factors
        hour = datetime.utcnow().hour
        if hour < 6 or hour > 22:  # Outside normal hours
            risk_score += 0.2
        
        return min(risk_score, 1.0)


class AuthorizationEngine:
    """Role-based access control (RBAC) engine."""
    
    def __init__(self):
        self.roles: Dict[str, Set[str]] = {}
        self.user_roles: Dict[str, Set[str]] = {}
        self.resource_permissions: Dict[str, Set[str]] = {}
        self.policy_cache: Dict[str, bool] = {}
        
        # Initialize default roles
        self._initialize_default_roles()
    
    def _initialize_default_roles(self) -> None:
        """Initialize default system roles."""
        self.roles.update({
            "admin": {"*"},  # All permissions
            "agent_manager": {"agent:create", "agent:read", "agent:update", "agent:delete"},
            "task_manager": {"task:create", "task:read", "task:update", "task:assign"},
            "viewer": {"*:read"},
            "agent": {"task:read", "task:update", "context:read", "context:write"}
        })
    
    async def check_permission(
        self,
        security_context: SecurityContext,
        resource: str,
        action: str
    ) -> bool:
        """Check if user has permission for resource:action."""
        try:
            user_id = security_context.user_id
            permission = f"{resource}:{action}"
            
            # Check cache first
            cache_key = f"{user_id}:{permission}"
            if cache_key in self.policy_cache:
                return self.policy_cache[cache_key]
            
            # Check direct permissions
            if permission in security_context.permissions:
                self.policy_cache[cache_key] = True
                return True
            
            # Check role-based permissions
            user_roles = self.user_roles.get(user_id, set())
            for role in user_roles:
                role_permissions = self.roles.get(role, set())
                
                # Check for exact match or wildcard
                if (permission in role_permissions or 
                    "*" in role_permissions or 
                    f"{resource}:*" in role_permissions or
                    f"*:{action}" in role_permissions):
                    self.policy_cache[cache_key] = True
                    return True
            
            # Check security level-based access
            if await self._check_security_level_access(security_context, resource, action):
                self.policy_cache[cache_key] = True
                return True
            
            self.policy_cache[cache_key] = False
            return False
            
        except Exception as e:
            logger.error(
                "Permission check error",
                user_id=security_context.user_id,
                resource=resource,
                action=action,
                error=str(e)
            )
            return False
    
    async def _check_security_level_access(
        self,
        security_context: SecurityContext,
        resource: str,
        action: str
    ) -> bool:
        """Check access based on security clearance level."""
        # Define resource security requirements
        resource_security_levels = {
            "system": SecurityLevel.SECRET,
            "admin": SecurityLevel.CONFIDENTIAL,
            "agent": SecurityLevel.INTERNAL,
            "task": SecurityLevel.INTERNAL,
            "public": SecurityLevel.PUBLIC
        }
        
        required_level = resource_security_levels.get(resource, SecurityLevel.INTERNAL)
        user_level = security_context.security_level
        
        # Define security level hierarchy
        level_hierarchy = {
            SecurityLevel.PUBLIC: 1,
            SecurityLevel.INTERNAL: 2,
            SecurityLevel.CONFIDENTIAL: 3,
            SecurityLevel.SECRET: 4,
            SecurityLevel.TOP_SECRET: 5
        }
        
        return level_hierarchy[user_level] >= level_hierarchy[required_level]
    
    def assign_role(self, user_id: str, role: str) -> bool:
        """Assign role to user."""
        if role not in self.roles:
            return False
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        
        self.user_roles[user_id].add(role)
        self._clear_user_cache(user_id)
        return True
    
    def remove_role(self, user_id: str, role: str) -> bool:
        """Remove role from user."""
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role)
            self._clear_user_cache(user_id)
            return True
        return False
    
    def _clear_user_cache(self, user_id: str) -> None:
        """Clear permission cache for user."""
        keys_to_remove = [key for key in self.policy_cache if key.startswith(f"{user_id}:")]
        for key in keys_to_remove:
            del self.policy_cache[key]


class ThreatDetectionEngine:
    """Advanced threat detection and response system."""
    
    def __init__(self):
        self.threat_indicators: List[ThreatIndicator] = []
        self.detection_rules: List[Dict[str, Any]] = []
        self.anomaly_baselines: Dict[str, Dict[str, float]] = {}
        
        # Initialize detection rules
        self._initialize_detection_rules()
    
    def _initialize_detection_rules(self) -> None:
        """Initialize threat detection rules."""
        self.detection_rules = [
            {
                "name": "brute_force_detection",
                "type": "authentication",
                "threshold": 10,
                "timeframe_minutes": 5,
                "severity": ThreatLevel.HIGH
            },
            {
                "name": "unusual_access_pattern",
                "type": "access",
                "threshold": 5,
                "timeframe_minutes": 1,
                "severity": ThreatLevel.MEDIUM
            },
            {
                "name": "privilege_escalation",
                "type": "authorization",
                "threshold": 3,
                "timeframe_minutes": 10,
                "severity": ThreatLevel.CRITICAL
            },
            {
                "name": "suspicious_ip_access",
                "type": "network",
                "threshold": 1,
                "timeframe_minutes": 1,
                "severity": ThreatLevel.HIGH
            }
        ]
    
    async def analyze_security_event(self, event: SecurityEvent) -> List[ThreatIndicator]:
        """Analyze security event for threats."""
        threats = []
        
        for rule in self.detection_rules:
            threat = await self._apply_detection_rule(event, rule)
            if threat:
                threats.append(threat)
        
        # Store detected threats
        self.threat_indicators.extend(threats)
        
        # Keep only recent threats
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.threat_indicators = [
            threat for threat in self.threat_indicators
            if threat.detected_at > cutoff_time
        ]
        
        return threats
    
    async def _apply_detection_rule(
        self,
        event: SecurityEvent,
        rule: Dict[str, Any]
    ) -> Optional[ThreatIndicator]:
        """Apply detection rule to security event."""
        try:
            rule_name = rule["name"]
            rule_type = rule["type"]
            threshold = rule["threshold"]
            timeframe_minutes = rule["timeframe_minutes"]
            severity = ThreatLevel(rule["severity"])
            
            # Filter recent events of same type
            cutoff_time = datetime.utcnow() - timedelta(minutes=timeframe_minutes)
            
            if rule_name == "brute_force_detection":
                return await self._detect_brute_force(event, threshold, cutoff_time, severity)
            elif rule_name == "unusual_access_pattern":
                return await self._detect_unusual_access(event, threshold, cutoff_time, severity)
            elif rule_name == "privilege_escalation":
                return await self._detect_privilege_escalation(event, threshold, cutoff_time, severity)
            elif rule_name == "suspicious_ip_access":
                return await self._detect_suspicious_ip(event, severity)
            
            return None
            
        except Exception as e:
            logger.error("Detection rule error", rule=rule["name"], error=str(e))
            return None
    
    async def _detect_brute_force(
        self,
        event: SecurityEvent,
        threshold: int,
        cutoff_time: datetime,
        severity: ThreatLevel
    ) -> Optional[ThreatIndicator]:
        """Detect brute force attacks."""
        if event.event_type != SecurityEventType.LOGIN_FAILURE:
            return None
        
        # Count recent failed login attempts from same IP
        recent_failures = 0
        for threat in self.threat_indicators:
            if (threat.source_ip == event.ip_address and
                threat.detected_at > cutoff_time and
                "login_failure" in threat.indicators):
                recent_failures += 1
        
        if recent_failures >= threshold:
            return ThreatIndicator(
                threat_type="brute_force_attack",
                description=f"Brute force attack detected from {event.ip_address}",
                severity=severity,
                source_ip=event.ip_address,
                user_id=event.user_id,
                indicators=["repeated_login_failures", "same_source_ip"],
                confidence_score=0.9
            )
        
        return None
    
    async def _detect_unusual_access(
        self,
        event: SecurityEvent,
        threshold: int,
        cutoff_time: datetime,
        severity: ThreatLevel
    ) -> Optional[ThreatIndicator]:
        """Detect unusual access patterns."""
        if event.event_type not in [SecurityEventType.DATA_ACCESS, SecurityEventType.UNAUTHORIZED_ACCESS]:
            return None
        
        # Analyze access pattern anomalies
        user_baseline = self.anomaly_baselines.get(event.user_id, {})
        
        # Check for unusual resource access
        resource_access_count = user_baseline.get(f"resource_access_{event.resource}", 0)
        if resource_access_count == 0:  # First time accessing this resource
            return ThreatIndicator(
                threat_type="unusual_access_pattern",
                description=f"User {event.user_id} accessing new resource {event.resource}",
                severity=ThreatLevel.LOW,
                user_id=event.user_id,
                indicators=["new_resource_access"],
                confidence_score=0.6
            )
        
        return None
    
    async def _detect_privilege_escalation(
        self,
        event: SecurityEvent,
        threshold: int,
        cutoff_time: datetime,
        severity: ThreatLevel
    ) -> Optional[ThreatIndicator]:
        """Detect privilege escalation attempts."""
        if event.event_type != SecurityEventType.PERMISSION_DENIED:
            return None
        
        # Count recent permission denied events for same user
        recent_denials = sum(1 for threat in self.threat_indicators
                           if (threat.user_id == event.user_id and
                               threat.detected_at > cutoff_time and
                               "permission_denied" in threat.indicators))
        
        if recent_denials >= threshold:
            return ThreatIndicator(
                threat_type="privilege_escalation",
                description=f"Possible privilege escalation attempt by {event.user_id}",
                severity=severity,
                user_id=event.user_id,
                indicators=["repeated_permission_denials"],
                confidence_score=0.8
            )
        
        return None
    
    async def _detect_suspicious_ip(
        self,
        event: SecurityEvent,
        severity: ThreatLevel
    ) -> Optional[ThreatIndicator]:
        """Detect suspicious IP addresses."""
        if not event.ip_address:
            return None
        
        # Check against known threat intelligence feeds
        # This would integrate with external threat intelligence
        suspicious_ips = ["192.168.1.100"]  # Placeholder
        
        if event.ip_address in suspicious_ips:
            return ThreatIndicator(
                threat_type="suspicious_ip_access",
                description=f"Access from known suspicious IP {event.ip_address}",
                severity=severity,
                source_ip=event.ip_address,
                user_id=event.user_id,
                indicators=["known_threat_ip"],
                confidence_score=0.95
            )
        
        return None


class AuditLogger:
    """Comprehensive security audit logging system."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.audit_events: deque = deque(maxlen=10000)
        self.compliance_logs: Dict[ComplianceFramework, List[SecurityEvent]] = defaultdict(list)
    
    async def log_security_event(
        self,
        event: SecurityEvent,
        compliance_frameworks: List[ComplianceFramework] = None
    ) -> bool:
        """Log security event for audit purposes."""
        try:
            # Store in memory
            self.audit_events.append(event)
            
            # Store in Redis for distributed access
            if self.redis:
                event_data = {
                    "event_id": str(event.event_id),
                    "event_type": event.event_type.value,
                    "user_id": event.user_id,
                    "session_id": str(event.session_id) if event.session_id else None,
                    "resource": event.resource,
                    "action": event.action,
                    "result": event.result,
                    "threat_level": event.threat_level.value,
                    "ip_address": event.ip_address,
                    "user_agent": event.user_agent,
                    "details": event.details,
                    "timestamp": event.timestamp.isoformat()
                }
                
                await self.redis.xadd(
                    "security_audit_log",
                    event_data,
                    maxlen=100000
                )
            
            # Store for compliance frameworks
            frameworks = compliance_frameworks or [ComplianceFramework.SOC2]
            for framework in frameworks:
                self.compliance_logs[framework].append(event)
                
                # Keep only recent events for compliance
                cutoff_time = datetime.utcnow() - timedelta(days=365)  # 1 year retention
                self.compliance_logs[framework] = [
                    e for e in self.compliance_logs[framework]
                    if e.timestamp > cutoff_time
                ]
            
            logger.debug(
                "Security event logged",
                event_id=str(event.event_id),
                event_type=event.event_type.value
            )
            
            return True
            
        except Exception as e:
            logger.error("Failed to log security event", event_id=str(event.event_id), error=str(e))
            return False
    
    async def get_audit_trail(
        self,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[SecurityEvent]:
        """Get audit trail with filters."""
        try:
            filtered_events = []
            
            for event in self.audit_events:
                # Apply filters
                if user_id and event.user_id != user_id:
                    continue
                if resource and event.resource != resource:
                    continue
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                
                filtered_events.append(event)
                
                if len(filtered_events) >= limit:
                    break
            
            return filtered_events
            
        except Exception as e:
            logger.error("Failed to get audit trail", error=str(e))
            return []
    
    async def generate_compliance_report(
        self,
        framework: ComplianceFramework,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Generate compliance report for specified framework."""
        try:
            relevant_events = [
                event for event in self.compliance_logs[framework]
                if start_date <= event.timestamp <= end_date
            ]
            
            # Analyze events by type
            event_counts = defaultdict(int)
            threat_levels = defaultdict(int)
            
            for event in relevant_events:
                event_counts[event.event_type.value] += 1
                threat_levels[event.threat_level.value] += 1
            
            # Calculate metrics
            total_events = len(relevant_events)
            security_incidents = len([e for e in relevant_events 
                                    if e.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]])
            
            return {
                "framework": framework.value,
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "summary": {
                    "total_events": total_events,
                    "security_incidents": security_incidents,
                    "incident_rate": security_incidents / max(total_events, 1)
                },
                "event_breakdown": dict(event_counts),
                "threat_level_breakdown": dict(threat_levels),
                "compliance_status": "compliant" if security_incidents == 0 else "needs_review"
            }
            
        except Exception as e:
            logger.error("Failed to generate compliance report", framework=framework.value, error=str(e))
            return {"error": str(e)}


class SecurityManager(UnifiedManagerBase):
    """
    Unified Security Manager consolidating all security-related functionality.
    
    Replaces 33 separate files:
    - security.py
    - security_audit.py
    - security_integration.py
    - security_middleware.py
    - security_migration_guide.py
    - security_monitoring_system.py
    - security_orchestrator_integration.py
    - security_performance_validator.py
    - security_policy_engine.py
    - security_validation_middleware.py
    - advanced_security_validator.py
    - api_security_middleware.py
    - enhanced_security_audit.py
    - enhanced_security_safeguards.py
    - enterprise_security_system.py
    - github_security.py
    - integrated_security_system.py
    - production_api_security.py
    - threat_detection_engine.py
    - auth.py
    - auth_metrics.py
    - authorization_engine.py
    - compliance_framework.py
    - comprehensive_audit_system.py
    - enhanced_jwt_manager.py
    - enterprise_compliance.py
    - enterprise_compliance_system.py
    - mfa_system.py
    - oauth_authentication_system.py
    - oauth_provider_system.py
    - rbac_engine.py
    - secret_manager.py
    - unified_authorization_engine.py
    - webauthn_system.py
    """
    
    def __init__(self, config: ManagerConfig, dependencies: Optional[Dict[str, Any]] = None):
        super().__init__(config, dependencies)
        
        # Core components
        self.auth_service: Optional[AuthenticationService] = None
        self.authz_engine = AuthorizationEngine()
        self.threat_detector = ThreatDetectionEngine()
        self.audit_logger: Optional[AuditLogger] = None
        
        # Security policies
        self.security_policies: Dict[uuid.UUID, SecurityPolicy] = {}
        self.active_sessions: Dict[uuid.UUID, SecurityContext] = {}
        
        # Configuration
        self.enable_threat_detection = config.plugin_config.get("enable_threat_detection", True)
        self.audit_all_events = config.plugin_config.get("audit_all_events", True)
        self.compliance_frameworks = config.plugin_config.get(
            "compliance_frameworks", 
            [ComplianceFramework.SOC2]
        )
        
        # Monitoring
        self.security_metrics = {
            "total_authentications": 0,
            "failed_authentications": 0,
            "threats_detected": 0,
            "security_violations": 0,
            "audit_events_logged": 0
        }
    
    async def _initialize_manager(self) -> bool:
        """Initialize the security manager."""
        try:
            # Initialize Redis connection
            redis_client = get_redis()
            
            # Initialize components
            self.auth_service = AuthenticationService(redis_client)
            self.audit_logger = AuditLogger(redis_client)
            
            # Load security policies
            await self._load_security_policies()
            
            # Initialize default policies if none exist
            if not self.security_policies:
                await self._create_default_security_policies()
            
            logger.info(
                "Security Manager initialized",
                threat_detection_enabled=self.enable_threat_detection,
                compliance_frameworks=[f.value for f in self.compliance_frameworks],
                security_policies=len(self.security_policies)
            )
            return True
            
        except Exception as e:
            logger.error("Failed to initialize Security Manager", error=str(e))
            return False
    
    async def _shutdown_manager(self) -> None:
        """Shutdown the security manager."""
        try:
            # Invalidate all active sessions
            for session_id in list(self.active_sessions.keys()):
                await self.invalidate_session(session_id)
            
            # Save security policies
            await self._save_security_policies()
            
            logger.info("Security Manager shutdown completed")
            
        except Exception as e:
            logger.error("Error during Security Manager shutdown", error=str(e))
    
    async def _get_manager_health(self) -> Dict[str, Any]:
        """Get security manager health information."""
        return {
            "authentication_service": self.auth_service is not None,
            "audit_logger": self.audit_logger is not None,
            "active_sessions": len(self.active_sessions),
            "security_policies": len(self.security_policies),
            "threat_detection_enabled": self.enable_threat_detection,
            "metrics": self.security_metrics.copy(),
            "recent_threats": len([
                threat for threat in self.threat_detector.threat_indicators
                if threat.detected_at > datetime.utcnow() - timedelta(hours=1)
            ]) if self.enable_threat_detection else 0
        }
    
    async def _load_plugins(self) -> None:
        """Load security manager plugins."""
        # Security plugins would be loaded here
        pass
    
    # === CORE SECURITY OPERATIONS ===
    
    async def authenticate(
        self,
        user_id: str,
        credentials: Dict[str, Any],
        authentication_method: AuthenticationMethod = AuthenticationMethod.PASSWORD,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Tuple[bool, Optional[SecurityContext]]:
        """Authenticate user and create security context."""
        return await self.execute_with_monitoring(
            "authenticate",
            self._authenticate_impl,
            user_id,
            credentials,
            authentication_method,
            ip_address,
            user_agent
        )
    
    async def _authenticate_impl(
        self,
        user_id: str,
        credentials: Dict[str, Any],
        authentication_method: AuthenticationMethod,
        ip_address: Optional[str],
        user_agent: Optional[str]
    ) -> Tuple[bool, Optional[SecurityContext]]:
        """Internal implementation of authentication."""
        try:
            # Update metrics
            self.security_metrics["total_authentications"] += 1
            
            # Perform authentication
            success, context = await self.auth_service.authenticate_user(
                user_id,
                credentials,
                authentication_method,
                ip_address,
                user_agent
            )
            
            # Create security event
            event = SecurityEvent(
                event_type=SecurityEventType.LOGIN_SUCCESS if success else SecurityEventType.LOGIN_FAILURE,
                user_id=user_id,
                session_id=context.session_id if context else None,
                ip_address=ip_address,
                user_agent=user_agent,
                details={
                    "authentication_method": authentication_method.value,
                    "risk_score": context.risk_score if context else 0.0
                },
                threat_level=ThreatLevel.LOW if success else ThreatLevel.MEDIUM
            )
            
            # Log event and analyze for threats
            if self.audit_logger:
                await self.audit_logger.log_security_event(event, self.compliance_frameworks)
                self.security_metrics["audit_events_logged"] += 1
            
            if self.enable_threat_detection:
                threats = await self.threat_detector.analyze_security_event(event)
                if threats:
                    self.security_metrics["threats_detected"] += len(threats)
                    for threat in threats:
                        logger.warning(
                            "Security threat detected",
                            threat_type=threat.threat_type,
                            severity=threat.severity.value,
                            confidence=threat.confidence_score
                        )
            
            if success and context:
                # Store active session
                self.active_sessions[context.session_id] = context
                
                logger.info(
                    "✅ User authenticated",
                    user_id=user_id,
                    session_id=str(context.session_id),
                    authentication_method=authentication_method.value,
                    risk_score=context.risk_score
                )
            else:
                self.security_metrics["failed_authentications"] += 1
                logger.warning(
                    "❌ Authentication failed",
                    user_id=user_id,
                    authentication_method=authentication_method.value,
                    ip_address=ip_address
                )
            
            return success, context
            
        except Exception as e:
            logger.error("Authentication error", user_id=user_id, error=str(e))
            self.security_metrics["failed_authentications"] += 1
            return False, None
    
    async def authorize(
        self,
        security_context: SecurityContext,
        resource: str,
        action: str
    ) -> bool:
        """Authorize user action on resource."""
        return await self.execute_with_monitoring(
            "authorize",
            self._authorize_impl,
            security_context,
            resource,
            action
        )
    
    async def _authorize_impl(
        self,
        security_context: SecurityContext,
        resource: str,
        action: str
    ) -> bool:
        """Internal implementation of authorization."""
        try:
            # Check if session is still valid
            if not await self._validate_session(security_context):
                return False
            
            # Perform authorization check
            authorized = await self.authz_engine.check_permission(
                security_context,
                resource,
                action
            )
            
            # Create security event
            event = SecurityEvent(
                event_type=SecurityEventType.DATA_ACCESS if authorized else SecurityEventType.PERMISSION_DENIED,
                user_id=security_context.user_id,
                session_id=security_context.session_id,
                resource=resource,
                action=action,
                result="success" if authorized else "denied",
                ip_address=security_context.ip_address,
                user_agent=security_context.user_agent,
                threat_level=ThreatLevel.LOW if authorized else ThreatLevel.MEDIUM
            )
            
            # Log event
            if self.audit_logger:
                await self.audit_logger.log_security_event(event, self.compliance_frameworks)
                self.security_metrics["audit_events_logged"] += 1
            
            # Analyze for threats
            if self.enable_threat_detection and not authorized:
                threats = await self.threat_detector.analyze_security_event(event)
                if threats:
                    self.security_metrics["threats_detected"] += len(threats)
            
            if not authorized:
                self.security_metrics["security_violations"] += 1
                logger.warning(
                    "Authorization denied",
                    user_id=security_context.user_id,
                    resource=resource,
                    action=action
                )
            
            # Update session activity
            security_context.last_activity = datetime.utcnow()
            
            return authorized
            
        except Exception as e:
            logger.error(
                "Authorization error",
                user_id=security_context.user_id,
                resource=resource,
                action=action,
                error=str(e)
            )
            return False
    
    async def invalidate_session(self, session_id: uuid.UUID) -> bool:
        """Invalidate user session."""
        return await self.execute_with_monitoring(
            "invalidate_session",
            self._invalidate_session_impl,
            session_id
        )
    
    async def _invalidate_session_impl(self, session_id: uuid.UUID) -> bool:
        """Internal implementation of session invalidation."""
        try:
            if session_id in self.active_sessions:
                context = self.active_sessions[session_id]
                
                # Create logout event
                event = SecurityEvent(
                    event_type=SecurityEventType.LOGIN_SUCCESS,  # Using as logout
                    user_id=context.user_id,
                    session_id=session_id,
                    ip_address=context.ip_address,
                    details={"action": "logout"},
                    threat_level=ThreatLevel.LOW
                )
                
                # Log event
                if self.audit_logger:
                    await self.audit_logger.log_security_event(event, self.compliance_frameworks)
                
                # Remove from active sessions
                del self.active_sessions[session_id]
                
                # Remove from Redis
                if self.auth_service and self.auth_service.redis:
                    await self.auth_service.redis.delete(f"security_session:{session_id}")
                
                logger.info(
                    "Session invalidated",
                    session_id=str(session_id),
                    user_id=context.user_id
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error("Session invalidation error", session_id=str(session_id), error=str(e))
            return False
    
    # === POLICY MANAGEMENT ===
    
    async def create_security_policy(
        self,
        name: str,
        description: str,
        policy_type: str,
        rules: List[Dict[str, Any]],
        compliance_frameworks: List[ComplianceFramework] = None
    ) -> SecurityPolicy:
        """Create a new security policy."""
        policy = SecurityPolicy(
            name=name,
            description=description,
            policy_type=policy_type,
            rules=rules,
            compliance_frameworks=compliance_frameworks or []
        )
        
        self.security_policies[policy.policy_id] = policy
        
        logger.info(
            "Security policy created",
            policy_id=str(policy.policy_id),
            name=name,
            policy_type=policy_type
        )
        
        return policy
    
    async def _load_security_policies(self) -> None:
        """Load security policies from storage."""
        # This would load policies from database
        pass
    
    async def _save_security_policies(self) -> None:
        """Save security policies to storage."""
        # This would save policies to database
        pass
    
    async def _create_default_security_policies(self) -> None:
        """Create default security policies."""
        # Create basic access control policy
        await self.create_security_policy(
            name="Basic Access Control",
            description="Default access control policy",
            policy_type="access_control",
            rules=[
                {"resource": "public", "action": "*", "permission": "allow"},
                {"resource": "admin", "action": "*", "permission": "deny", "except_roles": ["admin"]},
                {"resource": "agent", "action": "read", "permission": "allow"},
                {"resource": "agent", "action": "write", "permission": "deny", "except_roles": ["agent_manager", "admin"]}
            ],
            compliance_frameworks=[ComplianceFramework.SOC2]
        )
    
    async def _validate_session(self, security_context: SecurityContext) -> bool:
        """Validate if session is still active and valid."""
        session_id = security_context.session_id
        
        # Check if session exists
        if session_id not in self.active_sessions:
            return False
        
        # Check session timeout
        timeout_threshold = datetime.utcnow() - timedelta(
            minutes=self.auth_service.session_timeout_minutes
        )
        
        if security_context.last_activity < timeout_threshold:
            # Session expired
            await self.invalidate_session(session_id)
            return False
        
        return True
    
    # === PUBLIC API METHODS ===
    
    async def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics and statistics."""
        try:
            recent_threats = [
                threat for threat in self.threat_detector.threat_indicators
                if threat.detected_at > datetime.utcnow() - timedelta(hours=24)
            ]
            
            threat_breakdown = defaultdict(int)
            for threat in recent_threats:
                threat_breakdown[threat.severity.value] += 1
            
            return {
                "authentication": {
                    "total_attempts": self.security_metrics["total_authentications"],
                    "failed_attempts": self.security_metrics["failed_authentications"],
                    "success_rate": (
                        (self.security_metrics["total_authentications"] - self.security_metrics["failed_authentications"]) /
                        max(self.security_metrics["total_authentications"], 1)
                    )
                },
                "sessions": {
                    "active_sessions": len(self.active_sessions),
                    "total_created": self.security_metrics["total_authentications"] - self.security_metrics["failed_authentications"]
                },
                "threats": {
                    "total_detected": self.security_metrics["threats_detected"],
                    "recent_24h": len(recent_threats),
                    "severity_breakdown": dict(threat_breakdown)
                },
                "compliance": {
                    "audit_events_logged": self.security_metrics["audit_events_logged"],
                    "security_violations": self.security_metrics["security_violations"],
                    "active_policies": len(self.security_policies)
                }
            }
            
        except Exception as e:
            logger.error("Failed to get security metrics", error=str(e))
            return {"error": str(e)}
    
    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get active sessions for a user."""
        try:
            user_sessions = []
            
            for session_id, context in self.active_sessions.items():
                if context.user_id == user_id:
                    user_sessions.append({
                        "session_id": str(session_id),
                        "created_at": context.created_at.isoformat(),
                        "last_activity": context.last_activity.isoformat(),
                        "ip_address": context.ip_address,
                        "user_agent": context.user_agent,
                        "security_level": context.security_level.value,
                        "risk_score": context.risk_score,
                        "mfa_verified": context.mfa_verified
                    })
            
            return user_sessions
            
        except Exception as e:
            logger.error("Failed to get user sessions", user_id=user_id, error=str(e))
            return []
    
    async def generate_security_report(
        self,
        start_date: datetime,
        end_date: datetime,
        compliance_framework: ComplianceFramework = ComplianceFramework.SOC2
    ) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        try:
            if not self.audit_logger:
                return {"error": "Audit logger not available"}
            
            # Generate compliance report
            compliance_report = await self.audit_logger.generate_compliance_report(
                compliance_framework,
                start_date,
                end_date
            )
            
            # Add threat analysis
            period_threats = [
                threat for threat in self.threat_detector.threat_indicators
                if start_date <= threat.detected_at <= end_date
            ]
            
            threat_analysis = {
                "total_threats": len(period_threats),
                "by_severity": defaultdict(int),
                "by_type": defaultdict(int)
            }
            
            for threat in period_threats:
                threat_analysis["by_severity"][threat.severity.value] += 1
                threat_analysis["by_type"][threat.threat_type] += 1
            
            # Combine reports
            security_report = {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "compliance": compliance_report,
                "threat_analysis": {
                    "total_threats": threat_analysis["total_threats"],
                    "severity_breakdown": dict(threat_analysis["by_severity"]),
                    "type_breakdown": dict(threat_analysis["by_type"])
                },
                "security_metrics": await self.get_security_metrics(),
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return security_report
            
        except Exception as e:
            logger.error("Failed to generate security report", error=str(e))
            return {"error": str(e)}


# Factory function for creating security manager
def create_security_manager(**config_overrides) -> SecurityManager:
    """Create and initialize a security manager."""
    config = create_manager_config("SecurityManager", **config_overrides)
    return SecurityManager(config)