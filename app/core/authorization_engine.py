"""
Authorization Engine with RBAC (Role-Based Access Control).

Implements fine-grained permission checking, role management,
and dynamic scope evaluation with <100ms decision latency.
"""

import uuid
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from dataclasses import dataclass
from functools import lru_cache

from sqlalchemy import select, and_, or_, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models.security import (
    AgentIdentity, AgentRole, AgentRoleAssignment, SecurityAuditLog, SecurityEvent,
    AgentStatus, RoleScope, SecurityEventSeverity
)
from ..schemas.security import PermissionCheckRequest, PermissionCheckResponse, SecurityError
from .redis import RedisClient

logger = logging.getLogger(__name__)


class AccessDecision(Enum):
    """Access decision enumeration."""
    GRANTED = "granted"
    DENIED = "denied"
    ERROR = "error"


@dataclass
class PermissionContext:
    """Context for permission evaluation."""
    agent_id: uuid.UUID
    resource: str
    action: str
    request_context: Dict[str, Any]
    timestamp: datetime
    ip_address: Optional[str] = None
    session_id: Optional[uuid.UUID] = None
    correlation_id: Optional[str] = None


@dataclass
class AuthorizationResult:
    """Result of authorization check."""
    decision: AccessDecision
    reason: str
    matched_roles: List[str]
    effective_permissions: Dict[str, Any]
    conditions_met: bool
    evaluation_time_ms: float
    risk_factors: List[str]
    
    def to_response(self) -> PermissionCheckResponse:
        """Convert to API response."""
        return PermissionCheckResponse(
            allowed=(self.decision == AccessDecision.GRANTED),
            reason=self.reason,
            effective_permissions=self.effective_permissions,
            conditions_met=self.conditions_met
        )


class AuthorizationEngine:
    """
    Authorization Engine implementing RBAC with fine-grained permissions.
    
    Features:
    - Role-based access control with resource patterns
    - Dynamic permission evaluation
    - Condition-based access (time, IP, context)
    - Performance-optimized caching
    - Comprehensive audit logging
    - Risk-based decision making
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: RedisClient,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 300,
        max_evaluation_time_ms: int = 100
    ):
        """
        Initialize Authorization Engine.
        
        Args:
            db_session: Database session
            redis_client: Redis client for caching
            enable_caching: Enable permission caching
            cache_ttl_seconds: Cache TTL in seconds
            max_evaluation_time_ms: Maximum evaluation time target
        """
        self.db = db_session
        self.redis = redis_client
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl_seconds
        self.max_evaluation_time = max_evaluation_time_ms
        
        # Configuration
        self.config = {
            "default_deny": True,
            "require_explicit_grants": True,
            "enable_condition_evaluation": True,
            "enable_risk_assessment": True,
            "max_role_depth": 5,
            "permission_cache_size": 10000,
            "audit_all_decisions": True,
            "emergency_override_enabled": False
        }
        
        # Performance tracking
        self.performance_stats = {
            "total_checks": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_evaluation_time_ms": 0.0,
            "slow_evaluations": 0
        }
        
        # Cache keys
        self._permission_cache_prefix = "authz:perm:"
        self._role_cache_prefix = "authz:role:"
        self._agent_roles_cache_prefix = "authz:agent_roles:"
    
    async def check_permission(
        self,
        agent_id: str,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AuthorizationResult:
        """
        Check if agent has permission to perform action on resource.
        
        Args:
            agent_id: Agent identifier
            resource: Resource being accessed
            action: Action being performed
            context: Additional context for evaluation
            
        Returns:
            AuthorizationResult with decision and metadata
        """
        start_time = time.time()
        correlation_id = str(uuid.uuid4())
        
        try:
            # Create permission context
            perm_context = PermissionContext(
                agent_id=uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id,
                resource=resource,
                action=action,
                request_context=context or {},
                timestamp=datetime.utcnow(),
                ip_address=context.get("ip_address") if context else None,
                session_id=uuid.UUID(context.get("session_id")) if context and context.get("session_id") else None,
                correlation_id=correlation_id
            )
            
            # Check cache first
            if self.enable_caching:
                cached_result = await self._get_cached_permission(perm_context)
                if cached_result:
                    self.performance_stats["cache_hits"] += 1
                    return cached_result
                else:
                    self.performance_stats["cache_misses"] += 1
            
            # Perform authorization check
            result = await self._evaluate_permission(perm_context)
            
            # Cache result if successful evaluation
            if self.enable_caching and result.decision != AccessDecision.ERROR:
                await self._cache_permission_result(perm_context, result)
            
            # Update performance stats
            evaluation_time = (time.time() - start_time) * 1000
            result.evaluation_time_ms = evaluation_time
            
            self.performance_stats["total_checks"] += 1
            self.performance_stats["avg_evaluation_time_ms"] = (
                (self.performance_stats["avg_evaluation_time_ms"] * (self.performance_stats["total_checks"] - 1) + evaluation_time) /
                self.performance_stats["total_checks"]
            )
            
            if evaluation_time > self.max_evaluation_time:
                self.performance_stats["slow_evaluations"] += 1
                logger.warning(f"Slow authorization evaluation: {evaluation_time:.2f}ms for {agent_id}")
            
            # Audit the decision
            if self.config["audit_all_decisions"]:
                await self._audit_authorization_decision(perm_context, result)
            
            return result
            
        except Exception as e:
            evaluation_time = (time.time() - start_time) * 1000
            logger.error(f"Authorization check failed for {agent_id}: {e}")
            
            return AuthorizationResult(
                decision=AccessDecision.ERROR,
                reason=f"Authorization evaluation error: {str(e)}",
                matched_roles=[],
                effective_permissions={},
                conditions_met=False,
                evaluation_time_ms=evaluation_time,
                risk_factors=["evaluation_error"]
            )
    
    async def assign_role(
        self,
        agent_id: str,
        role_id: str,
        granted_by: str,
        granted_reason: Optional[str] = None,
        resource_scope: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        conditions: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Assign role to agent.
        
        Args:
            agent_id: Agent identifier
            role_id: Role identifier
            granted_by: Who granted the role
            granted_reason: Reason for granting
            resource_scope: Specific resource scope
            expires_at: Expiration time
            conditions: Access conditions
            
        Returns:
            True if assignment successful
        """
        try:
            # Validate agent and role exist
            agent_uuid = uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id
            role_uuid = uuid.UUID(role_id) if isinstance(role_id, str) else role_id
            
            agent = await self.db.get(AgentIdentity, agent_uuid)
            role = await self.db.get(AgentRole, role_uuid)
            
            if not agent:
                raise ValueError(f"Agent {agent_id} not found")
            if not role:
                raise ValueError(f"Role {role_id} not found")
            if not agent.is_active():
                raise ValueError(f"Agent {agent_id} is not active")
            
            # Check for existing assignment
            existing = await self.db.execute(
                select(AgentRoleAssignment).where(
                    and_(
                        AgentRoleAssignment.agent_id == agent_uuid,
                        AgentRoleAssignment.role_id == role_uuid,
                        AgentRoleAssignment.resource_scope == resource_scope,
                        AgentRoleAssignment.is_active == True
                    )
                )
            )
            
            if existing.scalar_one_or_none():
                raise ValueError("Role assignment already exists")
            
            # Create role assignment
            assignment = AgentRoleAssignment(
                agent_id=agent_uuid,
                role_id=role_uuid,
                granted_by=granted_by,
                granted_reason=granted_reason,
                resource_scope=resource_scope,
                expires_at=expires_at,
                conditions=conditions or {}
            )
            
            self.db.add(assignment)
            await self.db.commit()
            
            # Clear cached permissions for agent
            await self._clear_agent_permission_cache(agent_uuid)
            
            # Log role assignment
            await self._log_audit_event(
                agent_id=agent_uuid,
                action="assign_role",
                resource="role_assignment",
                resource_id=str(assignment.id),
                success=True,
                metadata={
                    "role_name": role.role_name,
                    "granted_by": granted_by,
                    "resource_scope": resource_scope,
                    "expires_at": expires_at.isoformat() if expires_at else None
                }
            )
            
            logger.info(f"Role {role.role_name} assigned to agent {agent.agent_name} by {granted_by}")
            return True
            
        except Exception as e:
            logger.error(f"Role assignment failed: {e}")
            await self.db.rollback()
            return False
    
    async def revoke_role(
        self,
        agent_id: str,
        role_id: str,
        revoked_by: str,
        revoked_reason: Optional[str] = None,
        resource_scope: Optional[str] = None
    ) -> bool:
        """
        Revoke role from agent.
        
        Args:
            agent_id: Agent identifier
            role_id: Role identifier
            revoked_by: Who revoked the role
            revoked_reason: Revocation reason
            resource_scope: Specific resource scope
            
        Returns:
            True if revocation successful
        """
        try:
            agent_uuid = uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id
            role_uuid = uuid.UUID(role_id) if isinstance(role_id, str) else role_id
            
            # Find active assignment
            result = await self.db.execute(
                select(AgentRoleAssignment).where(
                    and_(
                        AgentRoleAssignment.agent_id == agent_uuid,
                        AgentRoleAssignment.role_id == role_uuid,
                        AgentRoleAssignment.resource_scope == resource_scope,
                        AgentRoleAssignment.is_active == True
                    )
                )
            )
            
            assignment = result.scalar_one_or_none()
            if not assignment:
                return False
            
            # Revoke assignment
            assignment.is_active = False
            assignment.revoked_at = datetime.utcnow()
            assignment.revoked_by = revoked_by
            assignment.revoked_reason = revoked_reason
            
            await self.db.commit()
            
            # Clear cached permissions
            await self._clear_agent_permission_cache(agent_uuid)
            
            # Log revocation
            await self._log_audit_event(
                agent_id=agent_uuid,
                action="revoke_role",
                resource="role_assignment",
                resource_id=str(assignment.id),
                success=True,
                metadata={
                    "revoked_by": revoked_by,
                    "revoked_reason": revoked_reason
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Role revocation failed: {e}")
            await self.db.rollback()
            return False
    
    async def get_agent_permissions(
        self,
        agent_id: str,
        include_expired: bool = False
    ) -> Dict[str, Any]:
        """
        Get all permissions for an agent.
        
        Args:
            agent_id: Agent identifier
            include_expired: Include expired assignments
            
        Returns:
            Agent permissions summary
        """
        try:
            agent_uuid = uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id
            
            # Get active role assignments
            query = select(AgentRoleAssignment).options(
                selectinload(AgentRoleAssignment.role)
            ).where(AgentRoleAssignment.agent_id == agent_uuid)
            
            if not include_expired:
                query = query.where(
                    and_(
                        AgentRoleAssignment.is_active == True,
                        or_(
                            AgentRoleAssignment.expires_at.is_(None),
                            AgentRoleAssignment.expires_at > datetime.utcnow()
                        )
                    )
                )
            
            result = await self.db.execute(query)
            assignments = result.scalars().all()
            
            # Build permissions summary
            roles = []
            all_permissions = {"resources": set(), "actions": set()}
            resource_patterns = set()
            
            for assignment in assignments:
                if assignment.is_currently_active():
                    role_info = {
                        "role_id": str(assignment.role_id),
                        "role_name": assignment.role.role_name,
                        "scope": assignment.role.scope,
                        "resource_scope": assignment.resource_scope,
                        "granted_by": assignment.granted_by,
                        "granted_at": assignment.granted_at,
                        "expires_at": assignment.expires_at,
                        "conditions": assignment.conditions
                    }
                    roles.append(role_info)
                    
                    # Aggregate permissions
                    role_perms = assignment.role.permissions or {}
                    all_permissions["resources"].update(role_perms.get("resources", []))
                    all_permissions["actions"].update(role_perms.get("actions", []))
                    resource_patterns.update(assignment.role.resource_patterns or [])
            
            return {
                "agent_id": agent_id,
                "roles": roles,
                "aggregated_permissions": {
                    "resources": list(all_permissions["resources"]),
                    "actions": list(all_permissions["actions"])
                },
                "resource_patterns": list(resource_patterns),
                "total_roles": len(roles),
                "evaluated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get agent permissions: {e}")
            return {"error": str(e)}
    
    async def create_role(
        self,
        role_name: str,
        permissions: Dict[str, Any],
        created_by: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        scope: RoleScope = RoleScope.RESOURCE,
        resource_patterns: Optional[List[str]] = None,
        max_access_level: str = "read",
        can_delegate: bool = False,
        auto_expire_hours: Optional[int] = None
    ) -> Optional[AgentRole]:
        """
        Create a new role.
        
        Args:
            role_name: Unique role name
            permissions: Role permissions
            created_by: Who created the role
            display_name: Display name
            description: Role description
            scope: Role scope
            resource_patterns: Resource patterns
            max_access_level: Maximum access level
            can_delegate: Can delegate permissions
            auto_expire_hours: Auto-expire hours
            
        Returns:
            Created AgentRole or None if failed
        """
        try:
            # Check if role name exists
            existing = await self.db.execute(
                select(AgentRole).where(AgentRole.role_name == role_name)
            )
            
            if existing.scalar_one_or_none():
                raise ValueError(f"Role '{role_name}' already exists")
            
            # Validate permissions structure
            if not self._validate_permissions(permissions):
                raise ValueError("Invalid permissions structure")
            
            role = AgentRole(
                role_name=role_name,
                display_name=display_name,
                description=description,
                scope=scope.value,
                permissions=permissions,
                resource_patterns=resource_patterns or [],
                max_access_level=max_access_level,
                can_delegate=can_delegate,
                auto_expire_hours=auto_expire_hours,
                created_by=created_by
            )
            
            self.db.add(role)
            await self.db.commit()
            await self.db.refresh(role)
            
            # Log role creation
            await self._log_audit_event(
                agent_id=None,
                action="create_role",
                resource="role",
                resource_id=str(role.id),
                success=True,
                metadata={
                    "role_name": role_name,
                    "created_by": created_by,
                    "permissions": permissions
                }
            )
            
            logger.info(f"Role '{role_name}' created by {created_by}")
            return role
            
        except Exception as e:
            logger.error(f"Role creation failed: {e}")
            await self.db.rollback()
            return None
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get authorization engine performance metrics."""
        return {
            "performance_stats": self.performance_stats.copy(),
            "cache_hit_rate": (
                self.performance_stats["cache_hits"] / 
                max(1, self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"])
            ),
            "avg_evaluation_time_ms": self.performance_stats["avg_evaluation_time_ms"],
            "slow_evaluation_rate": (
                self.performance_stats["slow_evaluations"] / 
                max(1, self.performance_stats["total_checks"])
            ),
            "config": self.config.copy()
        }
    
    # Private helper methods
    
    async def _evaluate_permission(self, context: PermissionContext) -> AuthorizationResult:
        """Evaluate permission for given context."""
        risk_factors = []
        
        try:
            # Get agent
            agent = await self.db.get(AgentIdentity, context.agent_id)
            if not agent:
                return AuthorizationResult(
                    decision=AccessDecision.DENIED,
                    reason="Agent not found",
                    matched_roles=[],
                    effective_permissions={},
                    conditions_met=False,
                    evaluation_time_ms=0,
                    risk_factors=["unknown_agent"]
                )
            
            if not agent.is_active():
                return AuthorizationResult(
                    decision=AccessDecision.DENIED,
                    reason="Agent is not active",
                    matched_roles=[],
                    effective_permissions={},
                    conditions_met=False,
                    evaluation_time_ms=0,
                    risk_factors=["inactive_agent"]
                )
            
            # Get active role assignments
            assignments = await self._get_agent_role_assignments(context.agent_id)
            
            if not assignments:
                return AuthorizationResult(
                    decision=AccessDecision.DENIED,
                    reason="No active role assignments",
                    matched_roles=[],
                    effective_permissions={},
                    conditions_met=False,
                    evaluation_time_ms=0,
                    risk_factors=["no_roles"]
                )
            
            # Evaluate each role assignment
            matched_roles = []
            effective_permissions = {"resources": [], "actions": []}
            all_conditions_met = True
            
            for assignment in assignments:
                role = assignment.role
                
                # Check role permissions
                if role.has_permission(context.resource, context.action):
                    # Check conditions
                    conditions_met = True
                    if self.config["enable_condition_evaluation"]:
                        conditions_met = assignment.check_conditions(context.request_context)
                        if not conditions_met:
                            all_conditions_met = False
                            risk_factors.append("condition_violation")
                    
                    if conditions_met:
                        matched_roles.append(role.role_name)
                        role_perms = role.permissions or {}
                        effective_permissions["resources"].extend(role_perms.get("resources", []))
                        effective_permissions["actions"].extend(role_perms.get("actions", []))
            
            # Decision logic
            if matched_roles:
                # Check for risk factors
                if self.config["enable_risk_assessment"]:
                    risk_factors.extend(await self._assess_risk_factors(context, agent))
                
                # High-risk requests might be denied
                if "high_risk" in risk_factors and not self.config.get("allow_high_risk", False):
                    return AuthorizationResult(
                        decision=AccessDecision.DENIED,
                        reason="Access denied due to high risk factors",
                        matched_roles=matched_roles,
                        effective_permissions=effective_permissions,
                        conditions_met=all_conditions_met,
                        evaluation_time_ms=0,
                        risk_factors=risk_factors
                    )
                
                return AuthorizationResult(
                    decision=AccessDecision.GRANTED,
                    reason=f"Access granted via roles: {', '.join(matched_roles)}",
                    matched_roles=matched_roles,
                    effective_permissions=effective_permissions,
                    conditions_met=all_conditions_met,
                    evaluation_time_ms=0,
                    risk_factors=risk_factors
                )
            else:
                return AuthorizationResult(
                    decision=AccessDecision.DENIED,
                    reason="No matching role permissions",
                    matched_roles=[],
                    effective_permissions={},
                    conditions_met=all_conditions_met,
                    evaluation_time_ms=0,
                    risk_factors=risk_factors
                )
            
        except Exception as e:
            logger.error(f"Permission evaluation error: {e}")
            return AuthorizationResult(
                decision=AccessDecision.ERROR,
                reason=f"Evaluation error: {str(e)}",
                matched_roles=[],
                effective_permissions={},
                conditions_met=False,
                evaluation_time_ms=0,
                risk_factors=["evaluation_error"]
            )
    
    async def _get_agent_role_assignments(self, agent_id: uuid.UUID) -> List[AgentRoleAssignment]:
        """Get active role assignments for agent."""
        result = await self.db.execute(
            select(AgentRoleAssignment).options(
                selectinload(AgentRoleAssignment.role)
            ).where(
                and_(
                    AgentRoleAssignment.agent_id == agent_id,
                    AgentRoleAssignment.is_active == True,
                    or_(
                        AgentRoleAssignment.expires_at.is_(None),
                        AgentRoleAssignment.expires_at > datetime.utcnow()
                    )
                )
            )
        )
        
        return result.scalars().all()
    
    async def _assess_risk_factors(
        self, 
        context: PermissionContext, 
        agent: AgentIdentity
    ) -> List[str]:
        """Assess risk factors for the request."""
        risk_factors = []
        
        # Check for off-hours access
        current_hour = context.timestamp.hour
        if current_hour < 6 or current_hour > 22:
            risk_factors.append("off_hours_access")
        
        # Check for high-privilege actions
        high_privilege_actions = ["delete", "admin", "modify_permissions", "create_user"]
        if context.action in high_privilege_actions:
            risk_factors.append("high_privilege_action")
        
        # Check recent failed attempts (simplified)
        if context.request_context.get("recent_failures", 0) > 3:
            risk_factors.append("recent_failures")
        
        # Check for bulk operations
        if context.resource.endswith("/*") or "bulk" in context.action:
            risk_factors.append("bulk_operation")
        
        # Aggregate risk level
        if len(risk_factors) >= 3:
            risk_factors.append("high_risk")
        elif len(risk_factors) >= 2:
            risk_factors.append("medium_risk")
        elif len(risk_factors) >= 1:
            risk_factors.append("low_risk")
        
        return risk_factors
    
    def _validate_permissions(self, permissions: Dict[str, Any]) -> bool:
        """Validate permissions structure."""
        required_keys = ["resources", "actions"]
        
        if not isinstance(permissions, dict):
            return False
        
        for key in required_keys:
            if key not in permissions:
                return False
            if not isinstance(permissions[key], list):
                return False
        
        return True
    
    async def _get_cached_permission(self, context: PermissionContext) -> Optional[AuthorizationResult]:
        """Get cached permission result."""
        cache_key = self._build_permission_cache_key(context)
        
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                # Deserialize cached result (simplified)
                return None  # Would implement proper serialization
        except Exception as e:
            logger.debug(f"Cache get error: {e}")
        
        return None
    
    async def _cache_permission_result(
        self, 
        context: PermissionContext, 
        result: AuthorizationResult
    ) -> None:
        """Cache permission result."""
        cache_key = self._build_permission_cache_key(context)
        
        try:
            # Serialize result (simplified)
            await self.redis.set_with_expiry(cache_key, "cached", self.cache_ttl)
        except Exception as e:
            logger.debug(f"Cache set error: {e}")
    
    def _build_permission_cache_key(self, context: PermissionContext) -> str:
        """Build cache key for permission."""
        return f"{self._permission_cache_prefix}{context.agent_id}:{context.resource}:{context.action}"
    
    async def _clear_agent_permission_cache(self, agent_id: uuid.UUID) -> None:
        """Clear all cached permissions for agent."""
        pattern = f"{self._permission_cache_prefix}{agent_id}:*"
        try:
            await self.redis.delete_pattern(pattern)
        except Exception as e:
            logger.debug(f"Cache clear error: {e}")
    
    async def _audit_authorization_decision(
        self,
        context: PermissionContext,
        result: AuthorizationResult
    ) -> None:
        """Audit authorization decision."""
        await self._log_audit_event(
            agent_id=context.agent_id,
            action="check_permission",
            resource=context.resource,
            success=(result.decision == AccessDecision.GRANTED),
            metadata={
                "action": context.action,
                "decision": result.decision.value,
                "reason": result.reason,
                "matched_roles": result.matched_roles,
                "evaluation_time_ms": result.evaluation_time_ms,
                "risk_factors": result.risk_factors,
                "correlation_id": context.correlation_id
            }
        )
    
    async def _log_audit_event(
        self,
        action: str,
        resource: str,
        success: bool,
        agent_id: Optional[uuid.UUID] = None,
        resource_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log audit event."""
        audit_log = SecurityAuditLog(
            agent_id=agent_id,
            human_controller="system",  # System-generated
            action=action,
            resource=resource,
            resource_id=resource_id,
            success=success,
            metadata=metadata or {}
        )
        
        self.db.add(audit_log)
        # Note: Commit handled by caller


# Factory function
async def create_authorization_engine(
    db_session: AsyncSession,
    redis_client: RedisClient
) -> AuthorizationEngine:
    """
    Create Authorization Engine instance.
    
    Args:
        db_session: Database session
        redis_client: Redis client
        
    Returns:
        AuthorizationEngine instance
    """
    return AuthorizationEngine(db_session, redis_client)