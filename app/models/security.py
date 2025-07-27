"""
Security & Authentication Models for LeanVibe Agent Hive.

Implements OAuth 2.0/OIDC identity models, RBAC authorization,
and comprehensive audit logging for production security.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from sqlalchemy import Column, String, Text, Boolean, Integer, Float, DateTime, ForeignKey, text
from sqlalchemy.dialects.postgresql import UUID, JSON, ARRAY, INET
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from ..core.database import Base


class AgentStatus(Enum):
    """Agent identity status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    REVOKED = "revoked"


class RoleScope(Enum):
    """Role scope enumeration for RBAC."""
    GLOBAL = "global"
    SESSION = "session"
    CONTEXT = "context"
    RESOURCE = "resource"


class SecurityEventSeverity(Enum):
    """Security event severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentIdentity(Base):
    """
    Agent identity model for OAuth 2.0/OIDC authentication.
    
    Stores agent credentials, metadata, and authentication configuration
    with human accountability and rate limiting controls.
    """
    __tablename__ = 'agent_identities'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    agent_name = Column(String(255), nullable=False, index=True)
    human_controller = Column(String(255), nullable=False, index=True)
    
    # OAuth 2.0/OIDC credentials
    oauth_client_id = Column(String(255), nullable=True, unique=True)
    oauth_client_secret_hash = Column(String(255), nullable=True)  # Hashed with bcrypt
    public_key = Column(Text, nullable=True)  # RSA public key for JWT verification
    private_key_encrypted = Column(Text, nullable=True)  # Encrypted RSA private key
    
    # Authorization and rate limiting
    scopes = Column(ARRAY(String), nullable=True, server_default='{}')
    rate_limit_per_minute = Column(Integer, nullable=False, server_default='10')
    token_expires_in_seconds = Column(Integer, nullable=False, server_default='3600')  # 1 hour
    refresh_token_expires_in_seconds = Column(Integer, nullable=False, server_default='604800')  # 7 days
    max_concurrent_tokens = Column(Integer, nullable=False, server_default='5')
    allowed_redirect_uris = Column(ARRAY(String), nullable=True, server_default='{}')
    
    # Status and metadata
    status = Column(String(20), nullable=False, server_default='active', index=True)
    suspension_reason = Column(Text, nullable=True)
    agent_metadata = Column(JSON, nullable=True, server_default='{}')
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_used = Column(DateTime(timezone=True), nullable=True)
    last_token_refresh = Column(DateTime(timezone=True), nullable=True)
    
    # Audit fields
    created_by = Column(String(255), nullable=False)
    
    # Relationships
    role_assignments = relationship("AgentRoleAssignment", back_populates="agent_identity", cascade="all, delete-orphan")
    tokens = relationship("AgentToken", back_populates="agent_identity", cascade="all, delete-orphan")
    audit_logs = relationship("SecurityAuditLog", back_populates="agent_identity")
    security_events = relationship("SecurityEvent", back_populates="agent_identity")
    
    def __repr__(self):
        return f"<AgentIdentity(id={self.id}, name={self.agent_name}, status={self.status})>"
    
    def is_active(self) -> bool:
        """Check if agent identity is active."""
        return self.status == AgentStatus.ACTIVE.value
    
    def is_rate_limited(self, requests_in_window: int) -> bool:
        """Check if agent is rate limited."""
        return requests_in_window >= self.rate_limit_per_minute
    
    def get_active_scopes(self) -> List[str]:
        """Get active scopes for the agent."""
        if not self.is_active():
            return []
        return self.scopes or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "agent_name": self.agent_name,
            "human_controller": self.human_controller,
            "oauth_client_id": self.oauth_client_id,
            "scopes": self.scopes,
            "status": self.status,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "token_expires_in_seconds": self.token_expires_in_seconds,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "agent_metadata": self.agent_metadata
        }


class AgentRole(Base):
    """
    RBAC role model with fine-grained permissions.
    
    Defines roles with specific permissions, resource patterns,
    and access control configurations.
    """
    __tablename__ = 'agent_roles'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    role_name = Column(String(100), nullable=False, unique=True, index=True)
    display_name = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    
    # Role configuration
    scope = Column(String(20), nullable=False, server_default='resource', index=True)
    permissions = Column(JSON, nullable=False, server_default='{}')
    # Structure: {"resources": ["github", "files"], "actions": ["read", "write"], "conditions": {}}
    
    resource_patterns = Column(ARRAY(String), nullable=True, server_default='{}')
    # e.g., ["github/repos/org/*", "files/workspace/*"]
    
    max_access_level = Column(String(20), nullable=False, server_default='read')
    can_delegate = Column(Boolean, nullable=False, server_default=text('false'))
    auto_expire_hours = Column(Integer, nullable=True)  # Auto-expire role assignments
    is_system_role = Column(Boolean, nullable=False, server_default=text('false'))
    
    # Timestamps and audit
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by = Column(String(255), nullable=False)
    
    # Relationships
    role_assignments = relationship("AgentRoleAssignment", back_populates="role", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<AgentRole(id={self.id}, name={self.role_name}, scope={self.scope})>"
    
    def has_permission(self, resource: str, action: str) -> bool:
        """Check if role has specific permission."""
        permissions = self.permissions or {}
        
        # Check resource access
        allowed_resources = permissions.get("resources", [])
        if allowed_resources and resource not in allowed_resources:
            # Check resource patterns
            resource_match = False
            for pattern in self.resource_patterns or []:
                if self._matches_pattern(resource, pattern):
                    resource_match = True
                    break
            if not resource_match:
                return False
        
        # Check action access
        allowed_actions = permissions.get("actions", [])
        if allowed_actions and action not in allowed_actions:
            return False
        
        return True
    
    def _matches_pattern(self, resource: str, pattern: str) -> bool:
        """Check if resource matches pattern (simple wildcard support)."""
        if pattern.endswith("*"):
            return resource.startswith(pattern[:-1])
        return resource == pattern
    
    def get_effective_permissions(self) -> Dict[str, Any]:
        """Get effective permissions with resolved patterns."""
        return {
            "role_name": self.role_name,
            "scope": self.scope,
            "permissions": self.permissions,
            "resource_patterns": self.resource_patterns,
            "max_access_level": self.max_access_level,
            "can_delegate": self.can_delegate
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "role_name": self.role_name,
            "display_name": self.display_name,
            "description": self.description,
            "scope": self.scope,
            "permissions": self.permissions,
            "resource_patterns": self.resource_patterns,
            "max_access_level": self.max_access_level,
            "can_delegate": self.can_delegate,
            "auto_expire_hours": self.auto_expire_hours,
            "is_system_role": self.is_system_role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "created_by": self.created_by
        }


class AgentRoleAssignment(Base):
    """
    Agent role assignment model with temporal controls.
    
    Many-to-many relationship between agents and roles with
    time-based expiration and conditional access.
    """
    __tablename__ = 'agent_role_assignments'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agent_identities.id', ondelete='CASCADE'), 
                      nullable=False, index=True)
    role_id = Column(UUID(as_uuid=True), ForeignKey('agent_roles.id', ondelete='CASCADE'), 
                     nullable=False, index=True)
    
    # Assignment metadata
    granted_by = Column(String(255), nullable=False)
    granted_reason = Column(Text, nullable=True)
    resource_scope = Column(String(255), nullable=True)  # Specific resource or pattern
    conditions = Column(JSON, nullable=True, server_default='{}')
    # e.g., {"time_restricted": "09:00-17:00", "ip_restricted": ["10.0.0.0/8"]}
    
    # Temporal controls
    granted_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    revoked_by = Column(String(255), nullable=True)
    revoked_reason = Column(Text, nullable=True)
    
    # Status
    is_active = Column(Boolean, nullable=False, server_default=text('true'), index=True)
    agent_metadata = Column(JSON, nullable=True, server_default='{}')
    
    # Relationships
    agent_identity = relationship("AgentIdentity", back_populates="role_assignments")
    role = relationship("AgentRole", back_populates="role_assignments")
    
    def __repr__(self):
        return f"<AgentRoleAssignment(agent={self.agent_id}, role={self.role_id}, active={self.is_active})>"
    
    def is_currently_active(self) -> bool:
        """Check if role assignment is currently active."""
        if not self.is_active or self.revoked_at:
            return False
        
        now = datetime.utcnow()
        if self.expires_at and now > self.expires_at:
            return False
        
        return True
    
    def check_conditions(self, context: Dict[str, Any]) -> bool:
        """Check if current context meets assignment conditions."""
        if not self.conditions:
            return True
        
        # Check time restrictions
        time_restricted = self.conditions.get("time_restricted")
        if time_restricted:
            current_time = datetime.utcnow().strftime("%H:%M")
            start_time, end_time = time_restricted.split("-")
            if not (start_time <= current_time <= end_time):
                return False
        
        # Check IP restrictions
        ip_restricted = self.conditions.get("ip_restricted", [])
        if ip_restricted and context.get("ip_address"):
            # Simple IP checking (in production, use proper CIDR matching)
            client_ip = str(context["ip_address"])
            if not any(client_ip.startswith(allowed.split("/")[0]) for allowed in ip_restricted):
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "role_id": str(self.role_id),
            "granted_by": self.granted_by,
            "granted_reason": self.granted_reason,
            "resource_scope": self.resource_scope,
            "conditions": self.conditions,
            "granted_at": self.granted_at.isoformat() if self.granted_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "revoked_by": self.revoked_by,
            "is_active": self.is_active,
            "agent_metadata": self.agent_metadata
        }


class SecurityAuditLog(Base):
    """
    Comprehensive security audit log model.
    
    Records all agent actions with request/response data,
    timing information, and integrity signatures.
    """
    __tablename__ = 'security_audit_log'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agent_identities.id', ondelete='SET NULL'), 
                      nullable=True, index=True)
    human_controller = Column(String(255), nullable=False, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey('sessions.id', ondelete='SET NULL'), 
                        nullable=True, index=True)
    request_id = Column(String(255), nullable=True, index=True)  # For request correlation
    
    # Action details
    action = Column(String(255), nullable=False, index=True)
    resource = Column(String(255), nullable=True, index=True)
    resource_id = Column(String(255), nullable=True, index=True)
    method = Column(String(10), nullable=True)  # HTTP method
    endpoint = Column(String(255), nullable=True)
    
    # Request/response data
    request_data = Column(JSON, nullable=True)
    response_data = Column(JSON, nullable=True)
    
    # Network and client information
    ip_address = Column(INET, nullable=True, index=True)
    user_agent = Column(Text, nullable=True)
    geo_location = Column(String(100), nullable=True)  # Country/region
    
    # Outcome and performance
    success = Column(Boolean, nullable=False, index=True)
    http_status_code = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    error_code = Column(String(50), nullable=True, index=True)
    duration_ms = Column(Integer, nullable=True)
    
    # Security-specific fields
    tokens_used = Column(Integer, nullable=True)  # For rate limiting tracking
    permission_checked = Column(String(255), nullable=True)
    authorization_result = Column(String(50), nullable=True, index=True)  # granted, denied, error
    risk_score = Column(Float, nullable=True)  # Calculated risk score
    security_labels = Column(ARRAY(String), nullable=True, server_default='{}')
    # e.g., ["suspicious", "bulk_access", "privilege_escalation"]
    
    # Integrity and correlation
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    log_signature = Column(String(255), nullable=True)  # HMAC signature for integrity
    correlation_id = Column(String(255), nullable=True, index=True)  # For related events
    agent_metadata = Column(JSON, nullable=True, server_default='{}')
    
    # Relationships
    agent_identity = relationship("AgentIdentity", back_populates="audit_logs")
    
    def __repr__(self):
        return f"<SecurityAuditLog(id={self.id}, action={self.action}, success={self.success})>"
    
    def calculate_risk_indicators(self) -> Dict[str, Any]:
        """Calculate risk indicators for this log entry."""
        indicators = {
            "failed_auth": self.action.lower().find("auth") != -1 and not self.success,
            "privilege_escalation": self.action.lower().find("privilege") != -1,
            "bulk_access": self.security_labels and "bulk_access" in self.security_labels,
            "off_hours": self._is_off_hours(),
            "suspicious_user_agent": self._is_suspicious_user_agent(),
            "high_risk_action": self.action in ["delete", "modify_permissions", "create_token"]
        }
        
        # Calculate composite risk score
        risk_weights = {
            "failed_auth": 0.2,
            "privilege_escalation": 0.4,
            "bulk_access": 0.3,
            "off_hours": 0.1,
            "suspicious_user_agent": 0.1,
            "high_risk_action": 0.2
        }
        
        risk_score = sum(risk_weights.get(key, 0) for key, value in indicators.items() if value)
        indicators["composite_risk_score"] = min(1.0, risk_score)
        
        return indicators
    
    def _is_off_hours(self) -> bool:
        """Check if action occurred during off-hours (outside 9-17 UTC)."""
        if not self.timestamp:
            return False
        hour = self.timestamp.hour
        return hour < 9 or hour > 17
    
    def _is_suspicious_user_agent(self) -> bool:
        """Check for suspicious user agent patterns."""
        if not self.user_agent:
            return False
        
        suspicious_patterns = ["bot", "crawler", "scanner", "automated"]
        user_agent_lower = self.user_agent.lower()
        return any(pattern in user_agent_lower for pattern in suspicious_patterns)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "human_controller": self.human_controller,
            "session_id": str(self.session_id) if self.session_id else None,
            "request_id": self.request_id,
            "action": self.action,
            "resource": self.resource,
            "resource_id": self.resource_id,
            "method": self.method,
            "endpoint": self.endpoint,
            "ip_address": str(self.ip_address) if self.ip_address else None,
            "success": self.success,
            "http_status_code": self.http_status_code,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "duration_ms": self.duration_ms,
            "risk_score": self.risk_score,
            "security_labels": self.security_labels,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "correlation_id": self.correlation_id,
            "agent_metadata": self.agent_metadata
        }


class AgentToken(Base):
    """
    Agent token model for JWT token management.
    
    Tracks issued tokens with usage statistics and revocation support.
    """
    __tablename__ = 'agent_tokens'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agent_identities.id', ondelete='CASCADE'), 
                      nullable=False, index=True)
    
    # Token information
    token_type = Column(String(20), nullable=False, index=True)  # access, refresh
    token_hash = Column(String(255), nullable=False, unique=True, index=True)  # SHA-256 hash
    jti = Column(String(255), nullable=False, unique=True, index=True)  # JWT ID
    scopes = Column(ARRAY(String), nullable=True, server_default='{}')
    
    # Lifecycle management
    issued_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    usage_count = Column(Integer, nullable=False, server_default=text('0'))
    
    # Client information
    ip_address = Column(INET, nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Revocation
    is_revoked = Column(Boolean, nullable=False, server_default=text('false'), index=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    revoked_reason = Column(String(255), nullable=True)
    
    agent_metadata = Column(JSON, nullable=True, server_default='{}')
    
    # Relationships
    agent_identity = relationship("AgentIdentity", back_populates="tokens")
    
    def __repr__(self):
        return f"<AgentToken(id={self.id}, type={self.token_type}, revoked={self.is_revoked})>"
    
    def is_valid(self) -> bool:
        """Check if token is currently valid."""
        if self.is_revoked:
            return False
        
        now = datetime.utcnow()
        return now < self.expires_at
    
    def record_usage(self):
        """Record token usage."""
        self.last_used_at = datetime.utcnow()
        self.usage_count += 1
    
    def revoke(self, reason: str = None):
        """Revoke the token."""
        self.is_revoked = True
        self.revoked_at = datetime.utcnow()
        self.revoked_reason = reason
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "agent_id": str(self.agent_id),
            "token_type": self.token_type,
            "jti": self.jti,
            "scopes": self.scopes,
            "issued_at": self.issued_at.isoformat() if self.issued_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "usage_count": self.usage_count,
            "is_revoked": self.is_revoked,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "revoked_reason": self.revoked_reason,
            "agent_metadata": self.agent_metadata
        }


class SecurityEvent(Base):
    """
    Security event model for threat detection and monitoring.
    
    Records security incidents, suspicious activities, and system alerts.
    """
    __tablename__ = 'security_events'
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    event_type = Column(String(50), nullable=False, index=True)
    # e.g., "failed_auth", "suspicious_activity", "privilege_escalation"
    
    severity = Column(String(20), nullable=False, index=True)  # low, medium, high, critical
    agent_id = Column(UUID(as_uuid=True), ForeignKey('agent_identities.id', ondelete='SET NULL'), 
                      nullable=True, index=True)
    human_controller = Column(String(255), nullable=True, index=True)
    source_ip = Column(INET, nullable=True, index=True)
    
    # Event details
    description = Column(Text, nullable=False)
    details = Column(JSON, nullable=True, server_default='{}')
    risk_score = Column(Float, nullable=True, index=True)
    auto_detected = Column(Boolean, nullable=False, server_default=text('true'))
    
    # Resolution tracking
    is_resolved = Column(Boolean, nullable=False, server_default=text('false'), index=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolved_by = Column(String(255), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    false_positive = Column(Boolean, nullable=False, server_default=text('false'))
    
    # Correlation
    related_audit_log_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True, server_default='{}')
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    agent_metadata = Column(JSON, nullable=True, server_default='{}')
    
    # Relationships
    agent_identity = relationship("AgentIdentity", back_populates="security_events")
    
    def __repr__(self):
        return f"<SecurityEvent(id={self.id}, type={self.event_type}, severity={self.severity})>"
    
    def resolve(self, resolved_by: str, notes: str = None, false_positive: bool = False):
        """Mark security event as resolved."""
        self.is_resolved = True
        self.resolved_at = datetime.utcnow()
        self.resolved_by = resolved_by
        self.resolution_notes = notes
        self.false_positive = false_positive
    
    def calculate_priority_score(self) -> float:
        """Calculate priority score for incident response."""
        severity_weights = {
            "low": 0.1,
            "medium": 0.3,
            "high": 0.7,
            "critical": 1.0
        }
        
        base_score = severity_weights.get(self.severity, 0.5)
        
        # Adjust based on risk score
        if self.risk_score:
            base_score = (base_score + self.risk_score) / 2
        
        # Increase priority for unresolved events
        if not self.is_resolved:
            base_score *= 1.2
        
        return min(1.0, base_score)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "event_type": self.event_type,
            "severity": self.severity,
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "human_controller": self.human_controller,
            "source_ip": str(self.source_ip) if self.source_ip else None,
            "description": self.description,
            "details": self.details,
            "risk_score": self.risk_score,
            "auto_detected": self.auto_detected,
            "is_resolved": self.is_resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "resolution_notes": self.resolution_notes,
            "false_positive": self.false_positive,
            "related_audit_log_ids": [str(id) for id in (self.related_audit_log_ids or [])],
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "priority_score": self.calculate_priority_score(),
            "agent_metadata": self.agent_metadata
        }