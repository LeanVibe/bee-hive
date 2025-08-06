"""
Advanced Role-Based Access Control (RBAC) Permission Engine for LeanVibe Agent Hive.

Implements enterprise-grade RBAC with:
- Hierarchical role system with inheritance
- Context-aware authorization with dynamic conditions
- Resource-based access control with fine-grained permissions
- Policy-based permissions with temporal and geographic constraints
- Attribute-based access control (ABAC) integration
- Real-time permission evaluation and caching

Production-ready with comprehensive audit logging, performance optimization, and compliance features.
"""

import os
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

import structlog
import redis.asyncio as redis
from fastapi import HTTPException, Request, status, Depends
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, update, delete, func

from .database import get_session
from .auth import get_auth_service, AuthenticationService
from ..models.security import AgentIdentity, AgentRole, AgentRoleAssignment, SecurityAuditLog
from ..schemas.security import SecurityError

logger = structlog.get_logger()

# RBAC Configuration
RBAC_CONFIG = {
    "enable_hierarchical_roles": os.getenv("RBAC_HIERARCHICAL_ROLES", "true").lower() == "true",
    "enable_context_aware_auth": os.getenv("RBAC_CONTEXT_AWARE", "true").lower() == "true",
    "enable_temporal_permissions": os.getenv("RBAC_TEMPORAL_PERMISSIONS", "true").lower() == "true",
    "enable_geographic_restrictions": os.getenv("RBAC_GEOGRAPHIC_RESTRICTIONS", "true").lower() == "true",
    "cache_ttl_seconds": int(os.getenv("RBAC_CACHE_TTL", "300")),  # 5 minutes
    "permission_evaluation_timeout": int(os.getenv("RBAC_EVAL_TIMEOUT", "5000")),  # 5 seconds
    "redis_key_prefix": os.getenv("RBAC_REDIS_PREFIX", "rbac:"),
    "audit_all_decisions": os.getenv("RBAC_AUDIT_ALL", "false").lower() == "true",
    "max_role_depth": int(os.getenv("RBAC_MAX_ROLE_DEPTH", "10")),
    "enable_abac_integration": os.getenv("RBAC_ABAC_INTEGRATION", "true").lower() == "true"
}


class PermissionAction(Enum):
    """Standard permission actions."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"
    APPROVE = "approve"
    REVIEW = "review"
    PUBLISH = "publish"
    CONFIGURE = "configure"


class ResourceType(Enum):
    """Resource types in the system."""
    AGENT = "agent"
    SESSION = "session"
    TASK = "task"
    WORKFLOW = "workflow"
    DATASET = "dataset"
    MODEL = "model"
    CONFIGURATION = "configuration"
    USER = "user"
    ROLE = "role"
    PERMISSION = "permission"
    AUDIT_LOG = "audit_log"
    API_KEY = "api_key"
    WEBHOOK = "webhook"
    INTEGRATION = "integration"
    SYSTEM = "system"


class PermissionScope(Enum):
    """Permission scopes for fine-grained control."""
    GLOBAL = "global"
    ORGANIZATION = "organization"
    TEAM = "team"
    PROJECT = "project"
    PERSONAL = "personal"
    RESOURCE = "resource"


class AuthorizationResult(Enum):
    """Authorization decision results."""
    GRANTED = "granted"
    DENIED = "denied"
    CONDITIONAL = "conditional"
    DEFERRED = "deferred"
    ERROR = "error"


@dataclass
class PermissionCondition:
    """Dynamic permission condition."""
    type: str  # time, location, attribute, context, policy
    operator: str  # eq, ne, gt, lt, in, contains, matches
    value: Any
    description: str = ""
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition against context."""
        try:
            if self.type == "time":
                return self._evaluate_time_condition(context)
            elif self.type == "location":
                return self._evaluate_location_condition(context)
            elif self.type == "attribute":
                return self._evaluate_attribute_condition(context)
            elif self.type == "context":
                return self._evaluate_context_condition(context)
            elif self.type == "policy":
                return self._evaluate_policy_condition(context)
            else:
                return True  # Unknown condition types default to allow
                
        except Exception as e:
            logger.error("Condition evaluation failed", condition=self.type, error=str(e))
            return False  # Fail secure
    
    def _evaluate_time_condition(self, context: Dict[str, Any]) -> bool:
        """Evaluate time-based conditions."""
        current_time = datetime.utcnow()
        
        if self.operator == "hour_range":
            start_hour, end_hour = self.value
            current_hour = current_time.hour
            if start_hour <= end_hour:
                return start_hour <= current_hour <= end_hour
            else:  # Crosses midnight
                return current_hour >= start_hour or current_hour <= end_hour
        
        elif self.operator == "day_of_week":
            allowed_days = self.value  # List of weekday numbers (0=Monday)
            return current_time.weekday() in allowed_days
        
        elif self.operator == "date_range":
            start_date, end_date = self.value
            start_dt = datetime.fromisoformat(start_date)
            end_dt = datetime.fromisoformat(end_date)
            return start_dt <= current_time <= end_dt
        
        return True
    
    def _evaluate_location_condition(self, context: Dict[str, Any]) -> bool:
        """Evaluate location-based conditions."""
        user_ip = context.get("ip_address")
        user_country = context.get("country")
        
        if self.operator == "country_in":
            return user_country in self.value
        elif self.operator == "country_not_in":
            return user_country not in self.value
        elif self.operator == "ip_range":
            # Implement IP range checking
            import ipaddress
            try:
                user_ip_obj = ipaddress.ip_address(user_ip)
                for ip_range in self.value:
                    network = ipaddress.ip_network(ip_range, strict=False)
                    if user_ip_obj in network:
                        return True
                return False
            except Exception:
                return False
        
        return True
    
    def _evaluate_attribute_condition(self, context: Dict[str, Any]) -> bool:
        """Evaluate user/resource attribute conditions."""
        attribute_value = context.get("attributes", {}).get(self.value[0])  # attribute name
        expected_value = self.value[1]  # expected value
        
        if self.operator == "eq":
            return attribute_value == expected_value
        elif self.operator == "ne":
            return attribute_value != expected_value
        elif self.operator == "in":
            return attribute_value in expected_value
        elif self.operator == "contains":
            return expected_value in str(attribute_value)
        elif self.operator == "gt":
            return float(attribute_value) > float(expected_value)
        elif self.operator == "lt":
            return float(attribute_value) < float(expected_value)
        
        return True
    
    def _evaluate_context_condition(self, context: Dict[str, Any]) -> bool:
        """Evaluate contextual conditions."""
        context_value = context.get(self.value[0])  # context key
        expected_value = self.value[1] if len(self.value) > 1 else True
        
        if self.operator == "exists":
            return context_value is not None
        elif self.operator == "eq":
            return context_value == expected_value
        elif self.operator == "matches":
            import re
            return bool(re.match(str(expected_value), str(context_value)))
        
        return True
    
    def _evaluate_policy_condition(self, context: Dict[str, Any]) -> bool:
        """Evaluate policy-based conditions."""
        # This would integrate with external policy engines
        # For now, implement basic policy rules
        policy_name = self.value
        
        # Example policies
        if policy_name == "business_hours_only":
            current_hour = datetime.utcnow().hour
            return 9 <= current_hour <= 17  # 9 AM to 5 PM UTC
        
        elif policy_name == "require_mfa":
            return context.get("mfa_verified", False)
        
        elif policy_name == "require_secure_connection":
            return context.get("secure_connection", False)
        
        return True


@dataclass
class Permission:
    """Individual permission definition."""
    id: str
    resource_type: ResourceType
    action: PermissionAction
    scope: PermissionScope
    resource_id: Optional[str] = None
    conditions: List[PermissionCondition] = field(default_factory=list)
    priority: int = 100
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    description: str = ""
    tags: Set[str] = field(default_factory=set)


@dataclass
class Role:
    """Role definition with permissions and hierarchy."""
    id: str
    name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)
    child_roles: Set[str] = field(default_factory=set)
    scope: PermissionScope = PermissionScope.ORGANIZATION
    is_system_role: bool = False
    auto_assign_conditions: List[PermissionCondition] = field(default_factory=list)
    max_assignments: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tags: Set[str] = field(default_factory=set)


@dataclass
class AuthorizationContext:
    """Context for authorization decisions."""
    user_id: str
    resource_type: ResourceType
    resource_id: Optional[str]
    action: PermissionAction
    scope: PermissionScope = PermissionScope.RESOURCE
    
    # Request context
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_path: Optional[str] = None
    request_method: Optional[str] = None
    
    # User context
    user_attributes: Dict[str, Any] = field(default_factory=dict)
    user_roles: Set[str] = field(default_factory=set)
    mfa_verified: bool = False
    session_age_minutes: int = 0
    
    # Resource context
    resource_attributes: Dict[str, Any] = field(default_factory=dict)
    resource_owner: Optional[str] = None
    resource_tags: Set[str] = field(default_factory=set)
    
    # Temporal context
    request_time: datetime = field(default_factory=datetime.utcnow)
    
    # Geographic context
    country: Optional[str] = None
    region: Optional[str] = None
    
    # Security context
    secure_connection: bool = True
    auth_method: str = "unknown"
    risk_score: float = 0.0
    
    # Additional context
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthorizationDecision:
    """Result of authorization evaluation."""
    result: AuthorizationResult
    granted_permissions: Set[Permission] = field(default_factory=set)
    denied_permissions: Set[Permission] = field(default_factory=set)
    conditions: List[PermissionCondition] = field(default_factory=list)
    evaluation_time_ms: float = 0.0
    cache_hit: bool = False
    decision_path: List[str] = field(default_factory=list)
    reason: str = ""
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedRBACEngine:
    """
    Advanced Role-Based Access Control Engine.
    
    Features:
    - Hierarchical roles with inheritance
    - Context-aware authorization with dynamic conditions
    - Resource-based permissions with fine-grained control
    - Temporal and geographic access restrictions
    - Attribute-based access control (ABAC) integration
    - High-performance caching and evaluation
    - Comprehensive audit logging and compliance
    - Policy engine integration
    """
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        """
        Initialize RBAC Engine.
        
        Args:
            db_session: Database session for persistent storage
        """
        self.db = db_session
        
        # Redis connection for caching
        self.redis = redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True
        )
        
        # In-memory caches for performance
        self.role_cache: Dict[str, Role] = {}
        self.permission_cache: Dict[str, Set[Permission]] = {}
        self.hierarchy_cache: Dict[str, Set[str]] = {}  # role_id -> all inherited role_ids
        
        # Thread pool for parallel evaluation
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Performance metrics
        self.metrics = {
            "authorization_requests": 0,
            "authorization_grants": 0,
            "authorization_denials": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_evaluation_time_ms": 0.0,
            "condition_evaluations": 0,
            "hierarchy_traversals": 0,
            "role_assignments": 0,
            "permission_grants": 0,
            "temporal_restrictions": 0,
            "geographic_restrictions": 0,
            "policy_evaluations": 0
        }
        
        # Initialize system roles
        asyncio.create_task(self._initialize_system_roles())
        
        logger.info("Advanced RBAC Engine initialized",
                   hierarchical_roles=RBAC_CONFIG["enable_hierarchical_roles"],
                   context_aware=RBAC_CONFIG["enable_context_aware_auth"],
                   cache_ttl=RBAC_CONFIG["cache_ttl_seconds"])
    
    async def authorize(self, context: AuthorizationContext) -> AuthorizationDecision:
        """
        Perform authorization check with comprehensive evaluation.
        
        Args:
            context: Authorization context with user, resource, and environmental data
            
        Returns:
            AuthorizationDecision with detailed results
        """
        start_time = time.time()
        
        try:
            self.metrics["authorization_requests"] += 1
            
            # Check cache first
            cache_key = self._generate_cache_key(context)
            cached_decision = await self._get_cached_decision(cache_key)
            
            if cached_decision:
                self.metrics["cache_hits"] += 1
                cached_decision.cache_hit = True
                return cached_decision
            
            self.metrics["cache_misses"] += 1
            
            # Perform full authorization evaluation
            decision = await self._evaluate_authorization(context)
            
            # Calculate evaluation time
            evaluation_time = (time.time() - start_time) * 1000
            decision.evaluation_time_ms = evaluation_time
            
            # Update metrics
            self._update_evaluation_metrics(decision, evaluation_time)
            
            # Cache decision if appropriate
            if decision.result in [AuthorizationResult.GRANTED, AuthorizationResult.DENIED]:
                await self._cache_decision(cache_key, decision)
            
            # Audit decision if required
            if RBAC_CONFIG["audit_all_decisions"] or decision.result == AuthorizationResult.DENIED:
                await self._audit_authorization_decision(context, decision)
            
            return decision
            
        except Exception as e:
            logger.error("Authorization evaluation failed", 
                        user_id=context.user_id,
                        resource_type=context.resource_type.value,
                        action=context.action.value,
                        error=str(e))
            
            # Return secure default (deny) with error information
            return AuthorizationDecision(
                result=AuthorizationResult.ERROR,
                evaluation_time_ms=(time.time() - start_time) * 1000,
                reason=f"Authorization evaluation error: {str(e)}"
            )
    
    async def assign_role(self, user_id: str, role_id: str, assigned_by: str,
                         scope: Optional[str] = None, 
                         conditions: Optional[List[PermissionCondition]] = None,
                         expires_at: Optional[datetime] = None) -> bool:
        """
        Assign a role to a user with optional conditions and scope.
        
        Args:
            user_id: User to assign role to
            role_id: Role to assign
            assigned_by: User making the assignment
            scope: Optional scope limitation
            conditions: Optional assignment conditions
            expires_at: Optional expiration time
            
        Returns:
            True if assignment successful
        """
        try:
            # Validate role exists
            role = await self._get_role(role_id)
            if not role:
                raise ValueError(f"Role {role_id} not found")
            
            # Check if assigner has permission to assign this role
            assigner_context = AuthorizationContext(
                user_id=assigned_by,
                resource_type=ResourceType.ROLE,
                resource_id=role_id,
                action=PermissionAction.ADMIN
            )
            
            assignment_decision = await self.authorize(assigner_context)
            if assignment_decision.result != AuthorizationResult.GRANTED:
                raise PermissionError("Insufficient permissions to assign role")
            
            # Create role assignment
            assignment = AgentRoleAssignment(
                agent_id=user_id,
                role_id=role_id,
                granted_by=assigned_by,
                granted_reason="Role assignment via RBAC engine",
                resource_scope=scope,
                conditions=json.dumps([c.__dict__ for c in (conditions or [])]),
                expires_at=expires_at,
                is_active=True
            )
            
            if self.db:
                self.db.add(assignment)
                await self.db.commit()
            
            # Clear user's permission cache
            await self._invalidate_user_cache(user_id)
            
            # Update metrics
            self.metrics["role_assignments"] += 1
            
            # Log assignment
            await self._log_role_event(
                action="assign_role",
                user_id=user_id,
                role_id=role_id,
                assigned_by=assigned_by,
                metadata={
                    "scope": scope,
                    "expires_at": expires_at.isoformat() if expires_at else None,
                    "conditions_count": len(conditions) if conditions else 0
                }
            )
            
            logger.info("Role assigned successfully",
                       user_id=user_id,
                       role_id=role_id,
                       assigned_by=assigned_by)
            
            return True
            
        except Exception as e:
            logger.error("Role assignment failed",
                        user_id=user_id,
                        role_id=role_id,
                        error=str(e))
            return False
    
    async def revoke_role(self, user_id: str, role_id: str, revoked_by: str, 
                         reason: str = "") -> bool:
        """
        Revoke a role from a user.
        
        Args:
            user_id: User to revoke role from
            role_id: Role to revoke
            revoked_by: User making the revocation
            reason: Reason for revocation
            
        Returns:
            True if revocation successful
        """
        try:
            # Check revocation permissions
            revoker_context = AuthorizationContext(
                user_id=revoked_by,
                resource_type=ResourceType.ROLE,
                resource_id=role_id,
                action=PermissionAction.ADMIN
            )
            
            revocation_decision = await self.authorize(revoker_context)
            if revocation_decision.result != AuthorizationResult.GRANTED:
                raise PermissionError("Insufficient permissions to revoke role")
            
            # Update assignment in database
            if self.db:
                update_stmt = update(AgentRoleAssignment).where(
                    and_(
                        AgentRoleAssignment.agent_id == user_id,
                        AgentRoleAssignment.role_id == role_id,
                        AgentRoleAssignment.is_active == True
                    )
                ).values(
                    is_active=False,
                    revoked_at=datetime.utcnow(),
                    revoked_by=revoked_by,
                    revoked_reason=reason
                )
                
                await self.db.execute(update_stmt)
                await self.db.commit()
            
            # Clear user's permission cache
            await self._invalidate_user_cache(user_id)
            
            # Log revocation
            await self._log_role_event(
                action="revoke_role",
                user_id=user_id,
                role_id=role_id,
                assigned_by=revoked_by,
                metadata={
                    "reason": reason
                }
            )
            
            logger.info("Role revoked successfully",
                       user_id=user_id,
                       role_id=role_id,
                       revoked_by=revoked_by)
            
            return True
            
        except Exception as e:
            logger.error("Role revocation failed",
                        user_id=user_id,
                        role_id=role_id,
                        error=str(e))
            return False
    
    async def create_role(self, role_data: Dict[str, Any], created_by: str) -> Optional[Role]:
        """
        Create a new role with permissions.
        
        Args:
            role_data: Role configuration data
            created_by: User creating the role
            
        Returns:
            Created Role object if successful
        """
        try:
            # Check role creation permissions
            creator_context = AuthorizationContext(
                user_id=created_by,
                resource_type=ResourceType.ROLE,
                action=PermissionAction.CREATE
            )
            
            creation_decision = await self.authorize(creator_context)
            if creation_decision.result != AuthorizationResult.GRANTED:
                raise PermissionError("Insufficient permissions to create role")
            
            # Create role object
            role = Role(
                id=role_data.get("id", str(uuid.uuid4())),
                name=role_data["name"],
                description=role_data.get("description", ""),
                scope=PermissionScope(role_data.get("scope", "organization")),
                is_system_role=role_data.get("is_system_role", False),
                tags=set(role_data.get("tags", []))
            )
            
            # Add permissions
            for perm_data in role_data.get("permissions", []):
                permission = Permission(
                    id=str(uuid.uuid4()),
                    resource_type=ResourceType(perm_data["resource_type"]),
                    action=PermissionAction(perm_data["action"]),
                    scope=PermissionScope(perm_data.get("scope", "resource")),
                    resource_id=perm_data.get("resource_id"),
                    conditions=[
                        PermissionCondition(**cond) 
                        for cond in perm_data.get("conditions", [])
                    ],
                    description=perm_data.get("description", ""),
                    tags=set(perm_data.get("tags", []))
                )
                role.permissions.add(permission)
            
            # Store role in database
            if self.db:
                db_role = AgentRole(
                    id=role.id,
                    role_name=role.name,
                    display_name=role.name,
                    description=role.description,
                    scope=role.scope.value,
                    permissions=json.dumps([p.__dict__ for p in role.permissions]),
                    is_system_role=role.is_system_role,
                    created_by=created_by
                )
                
                self.db.add(db_role)
                await self.db.commit()
            
            # Cache the role
            self.role_cache[role.id] = role
            
            # Log creation
            await self._log_role_event(
                action="create_role",
                role_id=role.id,
                assigned_by=created_by,
                metadata={
                    "name": role.name,
                    "permissions_count": len(role.permissions),
                    "scope": role.scope.value
                }
            )
            
            logger.info("Role created successfully",
                       role_id=role.id,
                       name=role.name,
                       created_by=created_by)
            
            return role
            
        except Exception as e:
            logger.error("Role creation failed", error=str(e))
            return None
    
    async def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """
        Get all effective permissions for a user.
        
        Args:
            user_id: User to get permissions for
            
        Returns:
            Set of all effective permissions
        """
        try:
            # Check cache first
            cache_key = f"{RBAC_CONFIG['redis_key_prefix']}user_perms:{user_id}"
            cached_perms = await self.redis.get(cache_key)
            
            if cached_perms:
                self.metrics["cache_hits"] += 1
                return self._deserialize_permissions(json.loads(cached_perms))
            
            self.metrics["cache_misses"] += 1
            
            # Get user's roles
            user_roles = await self._get_user_roles(user_id)
            
            # Collect all permissions from roles (with inheritance)
            all_permissions = set()
            
            for role_id in user_roles:
                role_permissions = await self._get_role_permissions_with_inheritance(role_id)
                all_permissions.update(role_permissions)
            
            # Cache the result
            serialized_perms = json.dumps([p.__dict__ for p in all_permissions])
            await self.redis.setex(cache_key, RBAC_CONFIG["cache_ttl_seconds"], serialized_perms)
            
            return all_permissions
            
        except Exception as e:
            logger.error("Failed to get user permissions", user_id=user_id, error=str(e))
            return set()
    
    async def check_permission(self, user_id: str, resource_type: ResourceType, 
                              action: PermissionAction, resource_id: Optional[str] = None,
                              context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Quick permission check for a specific action.
        
        Args:
            user_id: User to check permissions for
            resource_type: Type of resource
            action: Action to perform
            resource_id: Specific resource ID (optional)
            context: Additional context for evaluation
            
        Returns:
            True if permission granted
        """
        auth_context = AuthorizationContext(
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            additional_context=context or {}
        )
        
        decision = await self.authorize(auth_context)
        return decision.result == AuthorizationResult.GRANTED
    
    async def get_accessible_resources(self, user_id: str, resource_type: ResourceType,
                                     action: PermissionAction) -> List[str]:
        """
        Get list of resources the user can access for a specific action.
        
        Args:
            user_id: User to check for
            resource_type: Type of resources
            action: Action to check
            
        Returns:
            List of accessible resource IDs
        """
        try:
            accessible_resources = []
            
            # Get user's permissions
            user_permissions = await self.get_user_permissions(user_id)
            
            # Filter permissions by resource type and action
            relevant_permissions = [
                p for p in user_permissions
                if p.resource_type == resource_type and p.action == action
            ]
            
            # Collect specific resource IDs and handle wildcards
            for permission in relevant_permissions:
                if permission.resource_id:
                    if permission.resource_id == "*":
                        # Wildcard permission - would need to query all resources
                        # For now, return indicator of wildcard access
                        return ["*"]
                    else:
                        accessible_resources.append(permission.resource_id)
            
            return list(set(accessible_resources))
            
        except Exception as e:
            logger.error("Failed to get accessible resources", 
                        user_id=user_id, 
                        resource_type=resource_type.value,
                        error=str(e))
            return []
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive RBAC system metrics."""
        
        try:
            # Get role and assignment counts from database
            db_metrics = await self._get_database_metrics()
            
            # Calculate cache statistics
            cache_total = self.metrics["cache_hits"] + self.metrics["cache_misses"]
            cache_hit_rate = (self.metrics["cache_hits"] / cache_total) if cache_total > 0 else 0.0
            
            return {
                "rbac_metrics": self.metrics.copy(),
                "cache_statistics": {
                    "hit_rate": round(cache_hit_rate, 3),
                    "total_lookups": cache_total,
                    "cache_entries": len(self.role_cache) + len(self.permission_cache)
                },
                "database_metrics": db_metrics,
                "system_roles_count": sum(1 for role in self.role_cache.values() if role.is_system_role),
                "custom_roles_count": sum(1 for role in self.role_cache.values() if not role.is_system_role),
                "config": {
                    "hierarchical_roles": RBAC_CONFIG["enable_hierarchical_roles"],
                    "context_aware": RBAC_CONFIG["enable_context_aware_auth"],
                    "temporal_permissions": RBAC_CONFIG["enable_temporal_permissions"],
                    "geographic_restrictions": RBAC_CONFIG["enable_geographic_restrictions"],
                    "cache_ttl": RBAC_CONFIG["cache_ttl_seconds"],
                    "max_role_depth": RBAC_CONFIG["max_role_depth"]
                }
            }
            
        except Exception as e:
            logger.error("Failed to get system metrics", error=str(e))
            return {"error": str(e)}
    
    # Private helper methods
    
    async def _evaluate_authorization(self, context: AuthorizationContext) -> AuthorizationDecision:
        """Perform comprehensive authorization evaluation."""
        
        decision = AuthorizationDecision(result=AuthorizationResult.DENIED)
        decision.decision_path.append("Started authorization evaluation")
        
        try:
            # Get user's active roles from database
            roles = await self._get_user_roles_objects(context.user_id)
            decision.decision_path.append(f"Retrieved {len(roles)} user roles")
            
            if not roles:
                decision.reason = "User has no assigned roles"
                return decision
            
            # Check role hierarchy and inheritance
            effective_permissions = []
            role_hierarchy = {}
            
            for role in roles:
                # Add base role permissions
                role_perms = role.permissions.get("actions", [])
                effective_permissions.extend(role_perms)
                
                # Build role hierarchy for inheritance
                role_hierarchy[role.role_name] = {
                    "permissions": role_perms,
                    "max_access_level": role.max_access_level,
                    "can_delegate": role.can_delegate,
                    "resource_patterns": role.resource_patterns or []
                }
                
                # Check for inherited roles (recursive resolution)
                parent_roles = role.permissions.get("inherits", [])
                for parent_role_name in parent_roles:
                    parent_perms = await self._resolve_inherited_permissions(parent_role_name)
                    effective_permissions.extend(parent_perms)
            
            # Remove duplicates and sort by priority
            effective_permissions = list(set(effective_permissions))
            
            # Evaluate resource-based permissions
            resource_allowed = await self._check_resource_permissions(
                context, roles, effective_permissions
            )
            
            if not resource_allowed["allowed"]:
                decision.result = AuthorizationResult.DENIED
                decision.reason = f"Resource access denied: {resource_allowed['reason']}"
                decision.resource_evaluations = resource_allowed["evaluations"]
                await self._audit_authorization_decision(context, decision)
                return decision
            
            # Evaluate policy conditions
            policy_decision = await self._evaluate_policy_conditions(
                context, roles, effective_permissions
            )
            
            if not policy_decision["allowed"]:
                decision.result = AuthorizationResult.DENIED
                decision.reason = f"Policy condition failed: {policy_decision['reason']}"
                decision.policy_evaluations = policy_decision["evaluations"]
                await self._audit_authorization_decision(context, decision)
                return decision
                
            # Check access level requirements
            access_level_check = self._check_access_level(
                context, role_hierarchy, effective_permissions
            )
            
            if not access_level_check["allowed"]:
                decision.result = AuthorizationResult.DENIED
                decision.reason = f"Access level insufficient: {access_level_check['reason']}"
                await self._audit_authorization_decision(context, decision)
                return decision
                
            # Final decision - check if action is explicitly allowed
            if context.action.value in effective_permissions:
                decision.result = AuthorizationResult.GRANTED
                decision.reason = f"Action '{context.action.value}' allowed by roles: {[r.role_name for r in roles]}"
                decision.granted_permissions = set()  # Would populate with Permission objects
                decision.role_hierarchy = role_hierarchy
                self.metrics["authorization_grants"] += 1
            else:
                decision.result = AuthorizationResult.DENIED
                decision.reason = f"Action '{context.action.value}' not permitted by assigned roles"
                self.metrics["authorization_denials"] += 1
                
            # Cache the decision
            await self._cache_decision(self._generate_cache_key(context), decision)
            
            # Audit the authorization
            await self._audit_authorization_decision(context, decision)
            
            return decision
            
        except Exception as e:
            decision.result = AuthorizationResult.ERROR
            decision.reason = f"Authorization evaluation error: {str(e)}"
            decision.decision_path.append(f"Error during evaluation: {str(e)}")
            return decision
    
    def _find_matching_permissions(self, permissions: Set[Permission], 
                                  context: AuthorizationContext) -> List[Permission]:
        """Find permissions that match the authorization context."""
        
        matching = []
        
        for permission in permissions:
            # Check resource type
            if permission.resource_type != context.resource_type:
                continue
            
            # Check action
            if permission.action != context.action:
                continue
            
            # Check resource ID (if specified)
            if permission.resource_id and context.resource_id:
                if permission.resource_id != "*" and permission.resource_id != context.resource_id:
                    continue
            
            # Check scope
            if not self._check_permission_scope(permission, context):
                continue
            
            matching.append(permission)
        
        return matching
    
    def _check_permission_scope(self, permission: Permission, 
                               context: AuthorizationContext) -> bool:
        """Check if permission scope matches context."""
        
        # Global scope always matches
        if permission.scope == PermissionScope.GLOBAL:
            return True
        
        # Resource-specific scope
        if permission.scope == PermissionScope.RESOURCE:
            return True  # Handled by resource_id matching
        
        # For other scopes, would need additional logic based on your system
        # This is a simplified implementation
        return True
    
    async def _evaluate_permission_conditions(self, permission: Permission,
                                            context: AuthorizationContext) -> bool:
        """Evaluate all conditions for a permission."""
        
        if not permission.conditions:
            return True  # No conditions means always allow
        
        # Check expiration
        if permission.expires_at and datetime.utcnow() > permission.expires_at:
            return False
        
        # Evaluate dynamic conditions
        context_dict = {
            "user_id": context.user_id,
            "resource_type": context.resource_type.value,
            "resource_id": context.resource_id,
            "action": context.action.value,
            "ip_address": context.ip_address,
            "country": context.country,
            "mfa_verified": context.mfa_verified,
            "secure_connection": context.secure_connection,
            "risk_score": context.risk_score,
            "attributes": context.user_attributes,
            **context.additional_context
        }
        
        for condition in permission.conditions:
            self.metrics["condition_evaluations"] += 1
            
            if not condition.evaluate(context_dict):
                return False  # All conditions must pass
        
        return True
    
    async def _get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID from cache or database."""
        
        # Check cache first
        if role_id in self.role_cache:
            return self.role_cache[role_id]
        
        # Load from database
        if self.db:
            stmt = select(AgentRole).where(AgentRole.id == role_id)
            result = await self.db.execute(stmt)
            db_role = result.scalar_one_or_none()
            
            if db_role:
                # Convert to Role object
                role = Role(
                    id=db_role.id,
                    name=db_role.role_name,
                    description=db_role.description or "",
                    scope=PermissionScope(db_role.scope),
                    is_system_role=db_role.is_system_role
                )
                
                # Parse permissions from JSON
                if db_role.permissions:
                    permissions_data = json.loads(db_role.permissions)
                    for perm_data in permissions_data:
                        permission = Permission(**perm_data)
                        role.permissions.add(permission)
                
                # Cache the role
                self.role_cache[role_id] = role
                
                return role
        
        return None
    
    async def _get_user_roles(self, user_id: str) -> Set[str]:
        """Get all active role IDs for a user."""
        
        if not self.db:
            return set()
        
        current_time = datetime.utcnow()
        
        stmt = select(AgentRoleAssignment.role_id).where(
            and_(
                AgentRoleAssignment.agent_id == user_id,
                AgentRoleAssignment.is_active == True,
                or_(
                    AgentRoleAssignment.expires_at.is_(None),
                    AgentRoleAssignment.expires_at > current_time
                )
            )
        )
        
        result = await self.db.execute(stmt)
        role_ids = {row.role_id for row in result}
        
        return role_ids
    
    async def _get_role_permissions_with_inheritance(self, role_id: str) -> Set[Permission]:
        """Get all permissions for a role including inherited permissions."""
        
        # Check cache
        if role_id in self.permission_cache:
            return self.permission_cache[role_id]
        
        all_permissions = set()
        
        # Get direct role permissions
        role = await self._get_role(role_id)
        if role:
            all_permissions.update(role.permissions)
            
            # Get inherited permissions if hierarchical roles are enabled
            if RBAC_CONFIG["enable_hierarchical_roles"]:
                inherited_roles = await self._get_inherited_roles(role_id)
                self.metrics["hierarchy_traversals"] += len(inherited_roles)
                
                for inherited_role_id in inherited_roles:
                    inherited_role = await self._get_role(inherited_role_id)
                    if inherited_role:
                        all_permissions.update(inherited_role.permissions)
        
        # Cache the result
        self.permission_cache[role_id] = all_permissions
        
        return all_permissions
    
    async def _get_inherited_roles(self, role_id: str, visited: Optional[Set[str]] = None,
                                  depth: int = 0) -> Set[str]:
        """Get all inherited role IDs with cycle detection."""
        
        if visited is None:
            visited = set()
        
        if depth > RBAC_CONFIG["max_role_depth"]:
            logger.warning("Role inheritance depth exceeded", role_id=role_id, depth=depth)
            return set()
        
        if role_id in visited:
            logger.warning("Role inheritance cycle detected", role_id=role_id)
            return set()
        
        visited.add(role_id)
        inherited = set()
        
        role = await self._get_role(role_id)
        if role:
            for parent_role_id in role.parent_roles:
                inherited.add(parent_role_id)
                # Recursively get parent's inherited roles
                parent_inherited = await self._get_inherited_roles(
                    parent_role_id, visited.copy(), depth + 1
                )
                inherited.update(parent_inherited)
        
        return inherited
    
    async def _resolve_inherited_permissions(self, parent_role_name: str) -> List[str]:
        """Resolve permissions from inherited roles."""
        try:
            # Find role by name
            parent_role = None
            for role in self.role_cache.values():
                if role.name == parent_role_name:
                    parent_role = role
                    break
            
            if not parent_role:
                # Try to load from database
                if self.db:
                    stmt = select(AgentRole).where(AgentRole.role_name == parent_role_name)
                    result = await self.db.execute(stmt)
                    db_role = result.scalar_one_or_none()
                    if db_role:
                        parent_role = await self._get_role(db_role.id)
            
            if parent_role:
                return [p.action.value for p in parent_role.permissions]
            
            return []
            
        except Exception as e:
            logger.error("Failed to resolve inherited permissions", 
                        parent_role=parent_role_name, error=str(e))
            return []
    
    async def _check_resource_permissions(self, context: AuthorizationContext, 
                                        roles: List[AgentRole], 
                                        effective_permissions: List[str]) -> Dict[str, Any]:
        """Check resource-based permissions with pattern matching."""
        try:
            # If no specific resource patterns, allow based on basic permissions
            if not any(role.resource_patterns for role in roles):
                return {"allowed": True, "reason": "No resource restrictions", "evaluations": []}
            
            evaluations = []
            
            # Check each role's resource patterns
            for role in roles:
                if not role.resource_patterns:
                    continue
                    
                for pattern in role.resource_patterns:
                    # Simple pattern matching (could be enhanced with regex)
                    resource_path = f"{context.resource_type.value}/{context.resource_id or '*'}"
                    
                    if self._matches_pattern(resource_path, pattern):
                        evaluations.append({
                            "role": role.role_name,
                            "pattern": pattern,
                            "resource_path": resource_path,
                            "allowed": True
                        })
                        return {
                            "allowed": True, 
                            "reason": f"Resource access allowed by pattern: {pattern}",
                            "evaluations": evaluations
                        }
                    else:
                        evaluations.append({
                            "role": role.role_name,
                            "pattern": pattern,
                            "resource_path": resource_path,
                            "allowed": False
                        })
            
            return {
                "allowed": False,
                "reason": "Resource access denied by all patterns",
                "evaluations": evaluations
            }
            
        except Exception as e:
            logger.error("Resource permission check failed", error=str(e))
            return {"allowed": False, "reason": f"Error checking resource permissions: {str(e)}", "evaluations": []}
    
    def _matches_pattern(self, resource_path: str, pattern: str) -> bool:
        """Check if resource path matches pattern with wildcard support."""
        import fnmatch
        return fnmatch.fnmatch(resource_path, pattern)
    
    def _check_access_level(self, context: AuthorizationContext, 
                           role_hierarchy: Dict[str, Any],
                           effective_permissions: List[str]) -> Dict[str, Any]:
        """Check access level requirements."""
        try:
            # Define access level hierarchy
            access_levels = {
                "read": 1,
                "write": 2,
                "admin": 3,
                "super_admin": 4
            }
            
            # Determine required access level based on action
            required_level = access_levels.get("read")  # Default
            
            if context.action in [PermissionAction.CREATE, PermissionAction.UPDATE]:
                required_level = access_levels.get("write", 2)
            elif context.action in [PermissionAction.DELETE, PermissionAction.ADMIN]:
                required_level = access_levels.get("admin", 3)
            
            # Find highest access level from user's roles
            user_access_level = 0
            for role_name, role_info in role_hierarchy.items():
                role_level = access_levels.get(role_info["max_access_level"], 0)
                user_access_level = max(user_access_level, role_level)
            
            if user_access_level >= required_level:
                return {
                    "allowed": True,
                    "reason": f"Access level sufficient: {user_access_level} >= {required_level}"
                }
            else:
                return {
                    "allowed": False,
                    "reason": f"Access level insufficient: {user_access_level} < {required_level}"
                }
                
        except Exception as e:
            logger.error("Access level check failed", error=str(e))
            return {"allowed": False, "reason": f"Error checking access level: {str(e)}"}
    
    def _generate_cache_key(self, context: AuthorizationContext) -> str:
        """Generate cache key for authorization decision."""
        
        # Create a hash of the authorization context
        key_data = {
            "user_id": context.user_id,
            "resource_type": context.resource_type.value,
            "resource_id": context.resource_id,
            "action": context.action.value,
            "scope": context.scope.value,
            "mfa_verified": context.mfa_verified
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"{RBAC_CONFIG['redis_key_prefix']}auth:{key_hash}"
    
    async def _get_cached_decision(self, cache_key: str) -> Optional[AuthorizationDecision]:
        """Get cached authorization decision."""
        
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                decision_data = json.loads(cached_data)
                
                # Check if cached decision has expired
                if decision_data.get("expires_at"):
                    expires_at = datetime.fromisoformat(decision_data["expires_at"])
                    if datetime.utcnow() > expires_at:
                        await self.redis.delete(cache_key)
                        return None
                
                # Reconstruct decision object
                decision = AuthorizationDecision(
                    result=AuthorizationResult(decision_data["result"]),
                    reason=decision_data.get("reason", ""),
                    evaluation_time_ms=decision_data.get("evaluation_time_ms", 0.0),
                    expires_at=datetime.fromisoformat(decision_data["expires_at"]) if decision_data.get("expires_at") else None
                )
                
                return decision
            
        except Exception as e:
            logger.error("Failed to get cached decision", error=str(e))
        
        return None
    
    async def _cache_decision(self, cache_key: str, decision: AuthorizationDecision) -> None:
        """Cache authorization decision."""
        
        try:
            # Determine cache TTL
            ttl = RBAC_CONFIG["cache_ttl_seconds"]
            
            if decision.expires_at:
                # Use the earlier of decision expiration and default TTL
                time_until_expiry = (decision.expires_at - datetime.utcnow()).total_seconds()
                ttl = min(ttl, max(1, int(time_until_expiry)))
            
            # Serialize decision
            decision_data = {
                "result": decision.result.value,
                "reason": decision.reason,
                "evaluation_time_ms": decision.evaluation_time_ms,
                "expires_at": decision.expires_at.isoformat() if decision.expires_at else None
            }
            
            await self.redis.setex(cache_key, ttl, json.dumps(decision_data))
            
        except Exception as e:
            logger.error("Failed to cache decision", error=str(e))
    
    async def _invalidate_user_cache(self, user_id: str) -> None:
        """Invalidate all cached data for a user."""
        
        try:
            # Get all cache keys for this user
            pattern = f"{RBAC_CONFIG['redis_key_prefix']}*{user_id}*"
            keys = []
            
            cursor = 0
            while True:
                cursor, batch = await self.redis.scan(cursor, match=pattern, count=100)
                keys.extend(batch)
                if cursor == 0:
                    break
            
            if keys:
                await self.redis.delete(*keys)
                logger.debug("Invalidated cache entries", user_id=user_id, count=len(keys))
            
        except Exception as e:
            logger.error("Failed to invalidate user cache", user_id=user_id, error=str(e))
    
    def _deserialize_permissions(self, permissions_data: List[Dict]) -> Set[Permission]:
        """Deserialize permissions from JSON data."""
        
        permissions = set()
        
        for perm_data in permissions_data:
            try:
                permission = Permission(
                    id=perm_data["id"],
                    resource_type=ResourceType(perm_data["resource_type"]),
                    action=PermissionAction(perm_data["action"]),
                    scope=PermissionScope(perm_data["scope"]),
                    resource_id=perm_data.get("resource_id"),
                    conditions=[PermissionCondition(**c) for c in perm_data.get("conditions", [])],
                    priority=perm_data.get("priority", 100),
                    expires_at=datetime.fromisoformat(perm_data["expires_at"]) if perm_data.get("expires_at") else None,
                    description=perm_data.get("description", ""),
                    tags=set(perm_data.get("tags", []))
                )
                permissions.add(permission)
            except Exception as e:
                logger.error("Failed to deserialize permission", permission_data=perm_data, error=str(e))
        
        return permissions
    
    async def _get_user_roles_objects(self, user_id: str) -> List:
        """Get user's active role objects from database."""
        if not self.db:
            return []
        
        current_time = datetime.utcnow()
        
        stmt = (
            select(AgentRole)
            .join(AgentRoleAssignment, AgentRole.id == AgentRoleAssignment.role_id)
            .where(
                and_(
                    AgentRoleAssignment.agent_id == user_id,
                    AgentRoleAssignment.is_active == True,
                    or_(
                        AgentRoleAssignment.expires_at.is_(None),
                        AgentRoleAssignment.expires_at > current_time
                    )
                )
            )
        )
        
        result = await self.db.execute(stmt)
        roles = result.scalars().all()
        
        return list(roles)
    
    async def _evaluate_policy_conditions(self, context: AuthorizationContext,
                                        roles: List, 
                                        effective_permissions: List[str]) -> Dict[str, Any]:
        """Evaluate policy-based conditions."""
        try:
            evaluations = []
            
            # Check temporal restrictions
            if RBAC_CONFIG["enable_temporal_permissions"]:
                current_hour = datetime.utcnow().hour
                # Business hours policy (9 AM to 5 PM UTC)
                if not (9 <= current_hour <= 17):
                    # Allow if user has "after_hours" permission
                    if "after_hours_access" not in effective_permissions:
                        return {
                            "allowed": False,
                            "reason": "Access denied outside business hours",
                            "evaluations": [{"policy": "business_hours", "result": False}]
                        }
                
                evaluations.append({"policy": "business_hours", "result": True})
            
            # Check geographic restrictions
            if RBAC_CONFIG["enable_geographic_restrictions"] and context.country:
                # Example: Restrict access from certain countries
                restricted_countries = ["XX", "YY"]  # Example restricted countries
                if context.country in restricted_countries:
                    if "global_access" not in effective_permissions:
                        return {
                            "allowed": False,
                            "reason": f"Access denied from country: {context.country}",
                            "evaluations": evaluations + [{"policy": "geo_restriction", "result": False}]
                        }
                
                evaluations.append({"policy": "geo_restriction", "result": True})
            
            # Check MFA requirements for sensitive actions
            sensitive_actions = [PermissionAction.DELETE, PermissionAction.ADMIN, PermissionAction.CONFIGURE]
            if context.action in sensitive_actions and not context.mfa_verified:
                if "bypass_mfa" not in effective_permissions:
                    return {
                        "allowed": False,
                        "reason": "MFA required for sensitive operations",
                        "evaluations": evaluations + [{"policy": "mfa_required", "result": False}]
                    }
            
            evaluations.append({"policy": "mfa_required", "result": True})
            
            # Check risk score thresholds
            if context.risk_score > 0.8:  # High risk threshold
                if "high_risk_access" not in effective_permissions:
                    return {
                        "allowed": False,
                        "reason": f"Access denied due to high risk score: {context.risk_score}",
                        "evaluations": evaluations + [{"policy": "risk_threshold", "result": False}]
                    }
            
            evaluations.append({"policy": "risk_threshold", "result": True})
            
            return {
                "allowed": True,
                "reason": "All policy conditions passed",
                "evaluations": evaluations
            }
            
        except Exception as e:
            logger.error("Policy condition evaluation failed", error=str(e))
            return {
                "allowed": False,
                "reason": f"Policy evaluation error: {str(e)}",
                "evaluations": []
            }
    
    def _update_evaluation_metrics(self, decision: AuthorizationDecision, evaluation_time_ms: float):
        """Update performance metrics."""
        
        # Update average evaluation time
        current_avg = self.metrics["avg_evaluation_time_ms"]
        total_requests = self.metrics["authorization_requests"]
        self.metrics["avg_evaluation_time_ms"] = (
            (current_avg * (total_requests - 1) + evaluation_time_ms) / total_requests
        )
    
    async def _audit_authorization_decision(self, context: AuthorizationContext,
                                          decision: AuthorizationDecision) -> None:
        """Audit authorization decisions."""
        
        if self.db:
            audit_log = SecurityAuditLog(
                agent_id=context.user_id,
                human_controller=context.user_id,  # Simplified
                action=f"authorize_{context.action.value}",
                resource=context.resource_type.value,
                resource_id=context.resource_id or "",
                success=decision.result == AuthorizationResult.GRANTED,
                metadata={
                    "authorization_result": decision.result.value,
                    "reason": decision.reason,
                    "evaluation_time_ms": decision.evaluation_time_ms,
                    "cache_hit": decision.cache_hit,
                    "granted_permissions": len(decision.granted_permissions),
                    "denied_permissions": len(decision.denied_permissions),
                    "conditions_evaluated": len(decision.conditions),
                    "decision_path_length": len(decision.decision_path),
                    "ip_address": context.ip_address,
                    "user_agent": context.user_agent,
                    "mfa_verified": context.mfa_verified,
                    "risk_score": context.risk_score
                }
            )
            self.db.add(audit_log)
    
    async def _log_role_event(self, action: str, user_id: str = None, role_id: str = None,
                             assigned_by: str = None, metadata: Dict[str, Any] = None) -> None:
        """Log role management events."""
        
        if self.db:
            audit_log = SecurityAuditLog(
                agent_id=user_id,
                human_controller=assigned_by or user_id,
                action=f"rbac_{action}",
                resource="role",
                resource_id=role_id or "",
                success=True,
                metadata={
                    "rbac_action": action,
                    "role_id": role_id,
                    "user_id": user_id,
                    "assigned_by": assigned_by,
                    **(metadata or {})
                }
            )
            self.db.add(audit_log)
    
    async def _get_database_metrics(self) -> Dict[str, Any]:
        """Get metrics from database."""
        
        if not self.db:
            return {}
        
        try:
            # Count roles
            roles_count = await self.db.scalar(select(func.count(AgentRole.id)))
            
            # Count active assignments
            assignments_count = await self.db.scalar(
                select(func.count(AgentRoleAssignment.id)).where(
                    AgentRoleAssignment.is_active == True
                )
            )
            
            return {
                "total_roles": roles_count or 0,
                "active_assignments": assignments_count or 0
            }
            
        except Exception as e:
            logger.error("Failed to get database metrics", error=str(e))
            return {}
    
    async def _initialize_system_roles(self) -> None:
        """Initialize default system roles."""
        
        try:
            # System Administrator
            admin_role = Role(
                id="system_admin",
                name="System Administrator",
                description="Full system access with all permissions",
                scope=PermissionScope.GLOBAL,
                is_system_role=True
            )
            
            # Add comprehensive admin permissions
            admin_permissions = []
            for resource_type in ResourceType:
                for action in PermissionAction:
                    permission = Permission(
                        id=f"admin_{resource_type.value}_{action.value}",
                        resource_type=resource_type,
                        action=action,
                        scope=PermissionScope.GLOBAL,
                        description=f"Admin {action.value} on {resource_type.value}"
                    )
                    admin_permissions.append(permission)
            
            admin_role.permissions.update(admin_permissions)
            self.role_cache[admin_role.id] = admin_role
            
            # Developer Role
            developer_role = Role(
                id="developer",
                name="Developer",
                description="Standard developer access",
                scope=PermissionScope.PROJECT,
                is_system_role=True
            )
            
            # Add developer permissions
            dev_resources = [ResourceType.AGENT, ResourceType.SESSION, ResourceType.TASK, ResourceType.WORKFLOW]
            dev_actions = [PermissionAction.CREATE, PermissionAction.READ, PermissionAction.UPDATE]
            
            for resource_type in dev_resources:
                for action in dev_actions:
                    permission = Permission(
                        id=f"dev_{resource_type.value}_{action.value}",
                        resource_type=resource_type,
                        action=action,
                        scope=PermissionScope.PROJECT,
                        description=f"Developer {action.value} on {resource_type.value}"
                    )
                    developer_role.permissions.add(permission)
            
            self.role_cache[developer_role.id] = developer_role
            
            # Read-Only Role
            readonly_role = Role(
                id="readonly",
                name="Read Only",
                description="Read-only access to most resources",
                scope=PermissionScope.ORGANIZATION,
                is_system_role=True
            )
            
            # Add read permissions
            for resource_type in ResourceType:
                if resource_type not in [ResourceType.SYSTEM, ResourceType.ROLE, ResourceType.PERMISSION]:
                    permission = Permission(
                        id=f"readonly_{resource_type.value}_read",
                        resource_type=resource_type,
                        action=PermissionAction.READ,
                        scope=PermissionScope.ORGANIZATION,
                        description=f"Read {resource_type.value}"
                    )
                    readonly_role.permissions.add(permission)
            
            self.role_cache[readonly_role.id] = readonly_role
            
            logger.info("System roles initialized", count=len(self.role_cache))
            
        except Exception as e:
            logger.error("Failed to initialize system roles", error=str(e))


# Global RBAC engine instance
_rbac_engine: Optional[AdvancedRBACEngine] = None


async def get_rbac_engine(db: AsyncSession = Depends(get_session)) -> AdvancedRBACEngine:
    """Get or create RBAC engine instance."""
    global _rbac_engine
    if _rbac_engine is None:
        _rbac_engine = AdvancedRBACEngine(db)
    return _rbac_engine


# Dependency for authorization checks
async def require_permission(resource_type: ResourceType, action: PermissionAction,
                           resource_id: Optional[str] = None):
    """Create a dependency that requires specific permission."""
    
    def permission_dependency(
        request: Request,
        current_user: str = Depends(get_current_user),
        rbac_engine: AdvancedRBACEngine = Depends(get_rbac_engine)
    ):
        async def check_permission():
            context = AuthorizationContext(
                user_id=current_user,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("User-Agent"),
                request_path=str(request.url.path),
                request_method=request.method
            )
            
            decision = await rbac_engine.authorize(context)
            
            if decision.result != AuthorizationResult.GRANTED:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied: {decision.reason}"
                )
            
            return decision
        
        return check_permission()
    
    return permission_dependency


# Export RBAC components
__all__ = [
    "AdvancedRBACEngine", "get_rbac_engine", "require_permission",
    "Permission", "Role", "AuthorizationContext", "AuthorizationDecision",
    "PermissionAction", "ResourceType", "PermissionScope", "AuthorizationResult",
    "PermissionCondition"
]