"""
Security Policy Engine for LeanVibe Agent Hive 2.0.

This system provides comprehensive security policy management with role-based access control,
configurable rules, dynamic policy evaluation, and integration with the existing security infrastructure.

Features:
- Dynamic security policy creation and management
- Role-based access control with fine-grained permissions
- Context-aware policy evaluation
- Policy conflict resolution and prioritization
- Integration with authorization engine and threat detection
- Real-time policy updates and versioning
- Comprehensive policy audit and compliance tracking
"""

import asyncio
import uuid
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, Counter
import logging

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from .advanced_security_validator import CommandContext, SecurityAnalysisResult, ThreatCategory
from .threat_detection_engine import ThreatDetection, ThreatType
from .enhanced_security_safeguards import (
    ControlDecision, SecurityContext, AgentBehaviorState, SecurityRiskLevel
)
from .authorization_engine import AuthorizationEngine, AuthorizationResult, AccessDecision
from .security_audit import SecurityAuditSystem, ThreatLevel, AuditEventType, SecurityEvent
from ..models.agent import Agent
from ..models.security import AgentRole, AgentRoleAssignment

logger = structlog.get_logger()


class PolicyType(Enum):
    """Types of security policies."""
    COMMAND_EXECUTION = "COMMAND_EXECUTION"
    DATA_ACCESS = "DATA_ACCESS"
    NETWORK_ACCESS = "NETWORK_ACCESS"
    RESOURCE_USAGE = "RESOURCE_USAGE"
    TIME_BASED = "TIME_BASED"
    BEHAVIORAL = "BEHAVIORAL"
    THREAT_RESPONSE = "THREAT_RESPONSE"
    COMPLIANCE = "COMPLIANCE"


class PolicyScope(Enum):
    """Scope of policy application."""
    GLOBAL = "GLOBAL"         # Applies to all agents
    ROLE_BASED = "ROLE_BASED" # Applies to specific roles
    AGENT_SPECIFIC = "AGENT_SPECIFIC"  # Applies to specific agents
    CONTEXT_BASED = "CONTEXT_BASED"   # Applies based on context


class PolicyAction(Enum):
    """Actions that can be taken by policies."""
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    MONITOR = "MONITOR"
    QUARANTINE = "QUARANTINE"
    ESCALATE = "ESCALATE"
    LOG_ONLY = "LOG_ONLY"


class PolicyPriority(Enum):
    """Priority levels for policy evaluation."""
    CRITICAL = 1000    # Security-critical policies
    HIGH = 800        # Important security policies
    MEDIUM = 500      # Standard policies
    LOW = 200         # Advisory policies
    INFORMATIONAL = 1 # Logging/monitoring only


@dataclass
class PolicyCondition:
    """Individual condition within a security policy."""
    id: str
    name: str
    condition_type: str  # e.g., "command_pattern", "time_range", "risk_score"
    operator: str       # e.g., "matches", "greater_than", "in_range"
    value: Any          # The value to compare against
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Evaluate condition against given context.
        
        Returns:
            Tuple of (matches, confidence)
        """
        try:
            context_value = self._extract_context_value(context)
            
            if context_value is None:
                return False, 0.0
            
            matches = self._apply_operator(context_value, self.operator, self.value)
            confidence = 1.0 if matches else 0.0
            
            return matches, confidence
            
        except Exception as e:
            logger.error(f"Policy condition evaluation error: {e}")
            return False, 0.0
    
    def _extract_context_value(self, context: Dict[str, Any]) -> Any:
        """Extract relevant value from context based on condition type."""
        
        if self.condition_type == "command_pattern":
            return context.get("command", "")
        elif self.condition_type == "risk_score":
            return context.get("current_risk_score", 0.0)
        elif self.condition_type == "agent_type":
            return context.get("agent_type", "")
        elif self.condition_type == "time_range":
            timestamp = context.get("timestamp")
            if timestamp:
                return timestamp.hour if hasattr(timestamp, 'hour') else None
            return None
        elif self.condition_type == "trust_level":
            return context.get("trust_level", 0.5)
        elif self.condition_type == "threat_categories":
            return context.get("threat_categories", [])
        elif self.condition_type == "behavioral_state":
            return context.get("behavior_state", "")
        elif self.condition_type == "resource_access":
            return context.get("accessed_resources", [])
        else:
            return context.get(self.condition_type)
    
    def _apply_operator(self, context_value: Any, operator: str, expected_value: Any) -> bool:
        """Apply operator to compare context value with expected value."""
        
        try:
            if operator == "equals":
                return context_value == expected_value
            elif operator == "not_equals":
                return context_value != expected_value
            elif operator == "greater_than":
                return float(context_value) > float(expected_value)
            elif operator == "less_than":
                return float(context_value) < float(expected_value)
            elif operator == "greater_equal":
                return float(context_value) >= float(expected_value)
            elif operator == "less_equal":
                return float(context_value) <= float(expected_value)
            elif operator == "in_range":
                if isinstance(expected_value, (list, tuple)) and len(expected_value) == 2:
                    return expected_value[0] <= float(context_value) <= expected_value[1]
                return False
            elif operator == "contains":
                return str(expected_value).lower() in str(context_value).lower()
            elif operator == "matches":
                import re
                return bool(re.search(str(expected_value), str(context_value), re.IGNORECASE))
            elif operator == "in_list":
                return context_value in expected_value if isinstance(expected_value, (list, set)) else False
            elif operator == "not_in_list":
                return context_value not in expected_value if isinstance(expected_value, (list, set)) else True
            elif operator == "intersects":
                if isinstance(context_value, (list, set)) and isinstance(expected_value, (list, set)):
                    return bool(set(context_value) & set(expected_value))
                return False
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False
                
        except (ValueError, TypeError) as e:
            logger.debug(f"Operator application error: {e}")
            return False


@dataclass
class SecurityPolicy:
    """Security policy with conditions and actions."""
    id: str
    name: str
    description: str
    policy_type: PolicyType
    scope: PolicyScope
    priority: PolicyPriority
    
    # Policy logic
    conditions: List[PolicyCondition]
    actions: List[PolicyAction]
    require_all_conditions: bool = True  # AND vs OR logic
    
    # Targeting
    target_roles: List[str] = field(default_factory=list)
    target_agents: List[str] = field(default_factory=list)
    excluded_agents: List[str] = field(default_factory=list)
    
    # Metadata
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    version: int = 1
    enabled: bool = True
    
    # Compliance and audit
    compliance_tags: List[str] = field(default_factory=list)
    audit_required: bool = True
    
    # Performance and effectiveness
    evaluation_count: int = 0
    match_count: int = 0
    false_positive_count: int = 0
    
    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, List[PolicyAction], float]:
        """
        Evaluate policy against given context.
        
        Returns:
            Tuple of (matches, actions, confidence)
        """
        self.evaluation_count += 1
        
        if not self.enabled:
            return False, [], 0.0
        
        # Check targeting
        if not self._matches_targeting(context):
            return False, [], 0.0
        
        # Evaluate conditions
        condition_results = []
        for condition in self.conditions:
            matches, confidence = condition.evaluate(context)
            condition_results.append((matches, confidence))
        
        if not condition_results:
            return False, [], 0.0
        
        # Apply logic (AND vs OR)
        if self.require_all_conditions:
            # ALL conditions must match
            matches = all(result[0] for result in condition_results)
            confidence = min(result[1] for result in condition_results) if matches else 0.0
        else:
            # ANY condition can match
            matches = any(result[0] for result in condition_results)
            confidence = max(result[1] for result in condition_results) if matches else 0.0
        
        if matches:
            self.match_count += 1
            return True, self.actions.copy(), confidence
        
        return False, [], 0.0
    
    def _matches_targeting(self, context: Dict[str, Any]) -> bool:
        """Check if policy targets apply to the context."""
        
        agent_id = context.get("agent_id")
        agent_roles = context.get("agent_roles", [])
        
        # Check excluded agents
        if agent_id and str(agent_id) in self.excluded_agents:
            return False
        
        # Apply scope-based targeting
        if self.scope == PolicyScope.GLOBAL:
            return True
        elif self.scope == PolicyScope.AGENT_SPECIFIC:
            return agent_id and str(agent_id) in self.target_agents
        elif self.scope == PolicyScope.ROLE_BASED:
            return bool(set(agent_roles) & set(self.target_roles))
        elif self.scope == PolicyScope.CONTEXT_BASED:
            # Context-based policies always apply (conditions handle the logic)
            return True
        
        return False
    
    def update_effectiveness(self, was_false_positive: bool = False) -> None:
        """Update policy effectiveness metrics."""
        if was_false_positive:
            self.false_positive_count += 1
        
        self.updated_at = datetime.utcnow()
    
    def get_effectiveness_score(self) -> float:
        """Calculate policy effectiveness score."""
        if self.evaluation_count == 0:
            return 0.0
        
        match_rate = self.match_count / self.evaluation_count
        false_positive_rate = self.false_positive_count / max(self.match_count, 1)
        
        # Effectiveness = match rate * (1 - false positive rate)
        return match_rate * (1.0 - false_positive_rate)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert policy to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "policy_type": self.policy_type.value,
            "scope": self.scope.value,
            "priority": self.priority.value,
            "conditions": [
                {
                    "id": cond.id,
                    "name": cond.name,
                    "condition_type": cond.condition_type,
                    "operator": cond.operator,
                    "value": cond.value,
                    "metadata": cond.metadata
                }
                for cond in self.conditions
            ],
            "actions": [action.value for action in self.actions],
            "require_all_conditions": self.require_all_conditions,
            "target_roles": self.target_roles,
            "target_agents": self.target_agents,
            "excluded_agents": self.excluded_agents,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "enabled": self.enabled,
            "compliance_tags": self.compliance_tags,
            "audit_required": self.audit_required,
            "effectiveness_score": self.get_effectiveness_score(),
            "evaluation_count": self.evaluation_count,
            "match_count": self.match_count,
            "false_positive_count": self.false_positive_count
        }


@dataclass
class PolicyEvaluationResult:
    """Result of policy engine evaluation."""
    decision: ControlDecision
    matched_policies: List[SecurityPolicy]
    failed_policies: List[SecurityPolicy]
    confidence: float
    
    # Detailed results
    policy_actions: List[PolicyAction]
    escalation_required: bool
    audit_required: bool
    
    # Recommendations
    recommended_actions: List[str]
    monitoring_requirements: List[str]
    
    # Metadata
    evaluation_time_ms: float
    total_policies_evaluated: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "decision": self.decision.value,
            "matched_policies": [p.id for p in self.matched_policies],
            "failed_policies": [p.id for p in self.failed_policies],
            "confidence": self.confidence,
            "policy_actions": [action.value for action in self.policy_actions],
            "escalation_required": self.escalation_required,
            "audit_required": self.audit_required,
            "recommended_actions": self.recommended_actions,
            "monitoring_requirements": self.monitoring_requirements,
            "evaluation_time_ms": self.evaluation_time_ms,
            "total_policies_evaluated": self.total_policies_evaluated
        }


class PolicyConflictResolver:
    """Resolves conflicts between multiple matching policies."""
    
    def __init__(self):
        # Action precedence (higher number = higher precedence)
        self.action_precedence = {
            PolicyAction.DENY: 1000,
            PolicyAction.QUARANTINE: 900,
            PolicyAction.REQUIRE_APPROVAL: 800,
            PolicyAction.ESCALATE: 700,
            PolicyAction.MONITOR: 600,
            PolicyAction.LOG_ONLY: 500,
            PolicyAction.ALLOW: 100
        }
    
    def resolve_conflicts(
        self,
        matched_policies: List[Tuple[SecurityPolicy, List[PolicyAction], float]]
    ) -> Tuple[List[PolicyAction], ControlDecision]:
        """
        Resolve conflicts between multiple matching policies.
        
        Args:
            matched_policies: List of (policy, actions, confidence) tuples
            
        Returns:
            Tuple of (resolved_actions, control_decision)
        """
        if not matched_policies:
            return [], ControlDecision.ALLOW
        
        # Sort policies by priority
        sorted_policies = sorted(
            matched_policies,
            key=lambda x: (x[0].priority.value, x[2]),  # Priority then confidence
            reverse=True
        )
        
        # Collect all actions with their priorities
        action_priorities = []
        for policy, actions, confidence in sorted_policies:
            for action in actions:
                precedence = self.action_precedence.get(action, 0)
                action_priorities.append((action, precedence, policy.priority.value, confidence))
        
        # Sort actions by precedence
        action_priorities.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        
        # Take the highest precedence action for control decision
        if action_priorities:
            highest_action = action_priorities[0][0]
            control_decision = self._map_action_to_decision(highest_action)
        else:
            control_decision = ControlDecision.ALLOW
        
        # Collect unique actions in precedence order
        resolved_actions = []
        seen_actions = set()
        
        for action, _, _, _ in action_priorities:
            if action not in seen_actions:
                resolved_actions.append(action)
                seen_actions.add(action)
        
        return resolved_actions, control_decision
    
    def _map_action_to_decision(self, action: PolicyAction) -> ControlDecision:
        """Map policy action to control decision."""
        mapping = {
            PolicyAction.ALLOW: ControlDecision.ALLOW,
            PolicyAction.DENY: ControlDecision.DENY,
            PolicyAction.REQUIRE_APPROVAL: ControlDecision.REQUIRE_APPROVAL,
            PolicyAction.QUARANTINE: ControlDecision.QUARANTINE,
            PolicyAction.ESCALATE: ControlDecision.ESCALATE,
            PolicyAction.MONITOR: ControlDecision.ALLOW,  # Allow but monitor
            PolicyAction.LOG_ONLY: ControlDecision.ALLOW  # Allow but log
        }
        return mapping.get(action, ControlDecision.ALLOW)


class SecurityPolicyEngine:
    """
    Security Policy Engine for comprehensive policy management and evaluation.
    
    Provides dynamic security policy creation, evaluation, and management with
    integration to the existing security infrastructure.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        authorization_engine: AuthorizationEngine,
        audit_system: SecurityAuditSystem
    ):
        self.db = db_session
        self.authorization_engine = authorization_engine
        self.audit_system = audit_system
        
        # Policy management
        self.policies: Dict[str, SecurityPolicy] = {}
        self.policy_cache_ttl = timedelta(minutes=30)
        self.last_policy_reload = datetime.utcnow()
        
        # Conflict resolution
        self.conflict_resolver = PolicyConflictResolver()
        
        # Performance metrics
        self.metrics = {
            "evaluations_performed": 0,
            "policies_matched": 0,
            "conflicts_resolved": 0,
            "avg_evaluation_time_ms": 0.0,
            "policy_effectiveness_scores": {},
            "false_positive_rate": 0.0
        }
        
        # Configuration
        self.config = {
            "enable_policy_caching": True,
            "max_evaluation_time_ms": 50,
            "auto_disable_ineffective_policies": True,
            "effectiveness_threshold": 0.1,
            "enable_policy_learning": True,
            "conflict_resolution_strategy": "priority_based"
        }
        
        # Initialize default policies
        asyncio.create_task(self._initialize_default_policies())
    
    async def evaluate_policies(
        self,
        context: CommandContext,
        security_result: SecurityAnalysisResult,
        threat_detections: List[ThreatDetection] = None
    ) -> PolicyEvaluationResult:
        """
        Evaluate all applicable policies against the given context.
        
        Args:
            context: Command execution context
            security_result: Security analysis result
            threat_detections: Detected threats
            
        Returns:
            PolicyEvaluationResult with evaluation outcome
        """
        start_time = time.time()
        threat_detections = threat_detections or []
        
        try:
            # Build evaluation context
            eval_context = await self._build_evaluation_context(
                context, security_result, threat_detections
            )
            
            # Get applicable policies
            applicable_policies = await self._get_applicable_policies(eval_context)
            
            # Evaluate policies
            matched_policies = []
            failed_policies = []
            
            for policy in applicable_policies:
                try:
                    matches, actions, confidence = policy.evaluate(eval_context)
                    
                    if matches:
                        matched_policies.append((policy, actions, confidence))
                    else:
                        failed_policies.append(policy)
                        
                except Exception as e:
                    logger.error(f"Policy evaluation error for {policy.id}: {e}")
                    failed_policies.append(policy)
            
            # Resolve conflicts
            resolved_actions, control_decision = self.conflict_resolver.resolve_conflicts(
                matched_policies
            )
            
            # Build result
            result = PolicyEvaluationResult(
                decision=control_decision,
                matched_policies=[p[0] for p in matched_policies],
                failed_policies=failed_policies,
                confidence=max([p[2] for p in matched_policies], default=0.0),
                policy_actions=resolved_actions,
                escalation_required=PolicyAction.ESCALATE in resolved_actions,
                audit_required=any(p[0].audit_required for p in matched_policies),
                recommended_actions=self._generate_recommendations(resolved_actions, matched_policies),
                monitoring_requirements=self._generate_monitoring_requirements(resolved_actions),
                evaluation_time_ms=0.0,  # Will be set below
                total_policies_evaluated=len(applicable_policies)
            )
            
            # Update metrics and return
            evaluation_time = (time.time() - start_time) * 1000
            result.evaluation_time_ms = evaluation_time
            self._update_metrics(result, matched_policies)
            
            # Log policy evaluations
            await self._log_policy_evaluation(context, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Policy evaluation error: {e}")
            evaluation_time = (time.time() - start_time) * 1000
            
            return PolicyEvaluationResult(
                decision=ControlDecision.ESCALATE,
                matched_policies=[],
                failed_policies=[],
                confidence=0.0,
                policy_actions=[PolicyAction.ESCALATE],
                escalation_required=True,
                audit_required=True,
                recommended_actions=["Manual review required due to policy evaluation error"],
                monitoring_requirements=["Enhanced monitoring due to evaluation failure"],
                evaluation_time_ms=evaluation_time,
                total_policies_evaluated=0
            )
    
    async def create_policy(
        self,
        name: str,
        description: str,
        policy_type: PolicyType,
        scope: PolicyScope,
        priority: PolicyPriority,
        conditions: List[Dict[str, Any]],
        actions: List[PolicyAction],
        **kwargs
    ) -> SecurityPolicy:
        """
        Create a new security policy.
        
        Args:
            name: Policy name
            description: Policy description
            policy_type: Type of policy
            scope: Policy scope
            priority: Policy priority
            conditions: List of condition definitions
            actions: List of policy actions
            **kwargs: Additional policy parameters
            
        Returns:
            Created SecurityPolicy
        """
        try:
            policy_id = str(uuid.uuid4())
            
            # Create condition objects
            policy_conditions = []
            for i, cond_def in enumerate(conditions):
                condition = PolicyCondition(
                    id=f"{policy_id}_cond_{i}",
                    name=cond_def.get("name", f"Condition {i+1}"),
                    condition_type=cond_def["condition_type"],
                    operator=cond_def["operator"],
                    value=cond_def["value"],
                    metadata=cond_def.get("metadata", {})
                )
                policy_conditions.append(condition)
            
            # Create policy
            policy = SecurityPolicy(
                id=policy_id,
                name=name,
                description=description,
                policy_type=policy_type,
                scope=scope,
                priority=priority,
                conditions=policy_conditions,
                actions=actions,
                require_all_conditions=kwargs.get("require_all_conditions", True),
                target_roles=kwargs.get("target_roles", []),
                target_agents=kwargs.get("target_agents", []),
                excluded_agents=kwargs.get("excluded_agents", []),
                created_by=kwargs.get("created_by", "system"),
                compliance_tags=kwargs.get("compliance_tags", []),
                audit_required=kwargs.get("audit_required", True)
            )
            
            # Add to policy store
            self.policies[policy_id] = policy
            
            # Log policy creation
            logger.info(
                f"Security policy created",
                policy_id=policy_id,
                name=name,
                type=policy_type.value,
                scope=scope.value,
                priority=priority.value
            )
            
            return policy
            
        except Exception as e:
            logger.error(f"Policy creation error: {e}")
            raise
    
    async def update_policy(
        self,
        policy_id: str,
        updates: Dict[str, Any]
    ) -> Optional[SecurityPolicy]:
        """
        Update an existing security policy.
        
        Args:
            policy_id: Policy identifier
            updates: Dictionary of updates to apply
            
        Returns:
            Updated SecurityPolicy or None if not found
        """
        if policy_id not in self.policies:
            return None
        
        policy = self.policies[policy_id]
        
        try:
            # Update policy attributes
            for key, value in updates.items():
                if hasattr(policy, key):
                    if key == "conditions" and isinstance(value, list):
                        # Handle condition updates
                        policy_conditions = []
                        for i, cond_def in enumerate(value):
                            condition = PolicyCondition(
                                id=f"{policy_id}_cond_{i}",
                                name=cond_def.get("name", f"Condition {i+1}"),
                                condition_type=cond_def["condition_type"],
                                operator=cond_def["operator"],
                                value=cond_def["value"],
                                metadata=cond_def.get("metadata", {})
                            )
                            policy_conditions.append(condition)
                        policy.conditions = policy_conditions
                    elif key == "actions" and isinstance(value, list):
                        policy.actions = [PolicyAction(action) for action in value]
                    else:
                        setattr(policy, key, value)
            
            # Update metadata
            policy.version += 1
            policy.updated_at = datetime.utcnow()
            
            logger.info(f"Security policy updated", policy_id=policy_id, version=policy.version)
            return policy
            
        except Exception as e:
            logger.error(f"Policy update error: {e}")
            return None
    
    async def delete_policy(self, policy_id: str) -> bool:
        """
        Delete a security policy.
        
        Args:
            policy_id: Policy identifier
            
        Returns:
            True if deleted successfully
        """
        if policy_id in self.policies:
            policy = self.policies[policy_id]
            del self.policies[policy_id]
            
            logger.info(f"Security policy deleted", policy_id=policy_id, name=policy.name)
            return True
        
        return False
    
    async def get_policy_by_id(self, policy_id: str) -> Optional[SecurityPolicy]:
        """Get policy by ID."""
        return self.policies.get(policy_id)
    
    async def list_policies(
        self,
        policy_type: Optional[PolicyType] = None,
        scope: Optional[PolicyScope] = None,
        enabled_only: bool = True
    ) -> List[SecurityPolicy]:
        """
        List policies with optional filtering.
        
        Args:
            policy_type: Filter by policy type
            scope: Filter by policy scope
            enabled_only: Only return enabled policies
            
        Returns:
            List of matching policies
        """
        policies = list(self.policies.values())
        
        if policy_type:
            policies = [p for p in policies if p.policy_type == policy_type]
        
        if scope:
            policies = [p for p in policies if p.scope == scope]
        
        if enabled_only:
            policies = [p for p in policies if p.enabled]
        
        return policies
    
    async def _build_evaluation_context(
        self,
        context: CommandContext,
        security_result: SecurityAnalysisResult,
        threat_detections: List[ThreatDetection]
    ) -> Dict[str, Any]:
        """Build comprehensive evaluation context."""
        
        # Get agent roles
        agent_roles = []
        if context.agent_id:
            try:
                permissions = await self.authorization_engine.get_agent_permissions(str(context.agent_id))
                agent_roles = [role["role_name"] for role in permissions.get("roles", [])]
            except Exception as e:
                logger.debug(f"Failed to get agent roles: {e}")
        
        # Build context
        eval_context = {
            # Basic context
            "agent_id": context.agent_id,
            "session_id": context.session_id,
            "command": context.command,
            "timestamp": context.timestamp,
            
            # Agent context
            "agent_type": context.agent_type,
            "agent_capabilities": context.agent_capabilities,
            "trust_level": context.trust_level,
            "agent_roles": agent_roles,
            
            # Security analysis results
            "is_safe": security_result.is_safe,
            "risk_level": security_result.risk_level.value,
            "threat_categories": [cat.value for cat in security_result.threat_categories],
            "confidence_score": security_result.confidence_score,
            "matched_signatures": security_result.matched_signatures,
            "risk_factors": security_result.risk_factors,
            "behavioral_anomalies": security_result.behavioral_anomalies,
            "command_intent": security_result.command_intent,
            "potential_impact": security_result.potential_impact,
            "affected_resources": security_result.affected_resources,
            
            # Threat detections
            "threat_detections": [detection.threat_type.value for detection in threat_detections],
            "threat_count": len(threat_detections),
            "max_threat_confidence": max([d.confidence for d in threat_detections], default=0.0),
            
            # Context-specific data
            "working_directory": context.working_directory,
            "environment_vars": context.environment_vars,
            "previous_commands": context.previous_commands,
            "current_risk_score": context.current_risk_score,
            "recent_anomalies": context.recent_anomalies,
            "security_alerts_24h": context.security_alerts_24h,
            
            # Network context
            "source_ip": context.source_ip,
            "user_agent": context.user_agent,
            "geo_location": context.geo_location,
            
            # Temporal context
            "time_since_last_command": context.time_since_last_command,
            "command_frequency": context.command_frequency
        }
        
        return eval_context
    
    async def _get_applicable_policies(self, context: Dict[str, Any]) -> List[SecurityPolicy]:
        """Get policies applicable to the given context."""
        
        applicable_policies = []
        
        for policy in self.policies.values():
            if policy.enabled and policy._matches_targeting(context):
                applicable_policies.append(policy)
        
        # Sort by priority
        applicable_policies.sort(key=lambda p: p.priority.value, reverse=True)
        
        return applicable_policies
    
    def _generate_recommendations(
        self,
        actions: List[PolicyAction],
        matched_policies: List[Tuple[SecurityPolicy, List[PolicyAction], float]]
    ) -> List[str]:
        """Generate recommendations based on policy actions."""
        
        recommendations = []
        
        if PolicyAction.DENY in actions:
            recommendations.append("Block command execution immediately")
            recommendations.append("Investigate potential security threat")
        
        if PolicyAction.QUARANTINE in actions:
            recommendations.append("Quarantine agent pending investigation")
            recommendations.append("Review all recent agent activities")
        
        if PolicyAction.REQUIRE_APPROVAL in actions:
            recommendations.append("Require human approval before execution")
            recommendations.append("Provide security context to approver")
        
        if PolicyAction.ESCALATE in actions:
            recommendations.append("Escalate to security team")
            recommendations.append("Enhanced monitoring and logging")
        
        if PolicyAction.MONITOR in actions:
            recommendations.append("Enable enhanced monitoring")
            recommendations.append("Log detailed execution information")
        
        # Add policy-specific recommendations
        for policy, policy_actions, confidence in matched_policies:
            if policy.compliance_tags:
                recommendations.append(f"Ensure compliance with: {', '.join(policy.compliance_tags)}")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_monitoring_requirements(self, actions: List[PolicyAction]) -> List[str]:
        """Generate monitoring requirements based on actions."""
        
        requirements = []
        
        if PolicyAction.MONITOR in actions or PolicyAction.LOG_ONLY in actions:
            requirements.append("Command execution logging")
            requirements.append("Resource access monitoring")
        
        if PolicyAction.ESCALATE in actions:
            requirements.append("Real-time alerting")
            requirements.append("Behavioral analysis")
        
        if PolicyAction.REQUIRE_APPROVAL in actions:
            requirements.append("Approval workflow tracking")
            requirements.append("Decision audit trail")
        
        return requirements
    
    async def _initialize_default_policies(self) -> None:
        """Initialize default security policies."""
        
        try:
            # Critical command blocking policy
            await self.create_policy(
                name="Block Critical Security Threats",
                description="Block commands with critical security threats",
                policy_type=PolicyType.COMMAND_EXECUTION,
                scope=PolicyScope.GLOBAL,
                priority=PolicyPriority.CRITICAL,
                conditions=[
                    {
                        "condition_type": "risk_level",
                        "operator": "equals",
                        "value": "CRITICAL"
                    }
                ],
                actions=[PolicyAction.DENY, PolicyAction.LOG_ONLY],
                compliance_tags=["security_critical"]
            )
            
            # High-risk command approval policy
            await self.create_policy(
                name="High-Risk Command Approval",
                description="Require approval for high-risk commands",
                policy_type=PolicyType.COMMAND_EXECUTION,
                scope=PolicyScope.GLOBAL,
                priority=PolicyPriority.HIGH,
                conditions=[
                    {
                        "condition_type": "risk_level",
                        "operator": "equals",
                        "value": "HIGH"
                    },
                    {
                        "condition_type": "trust_level",
                        "operator": "less_than",
                        "value": 0.7
                    }
                ],
                actions=[PolicyAction.REQUIRE_APPROVAL, PolicyAction.MONITOR],
                require_all_conditions=False  # OR logic
            )
            
            # Privilege escalation monitoring
            await self.create_policy(
                name="Privilege Escalation Monitoring",
                description="Monitor privilege escalation attempts",
                policy_type=PolicyType.BEHAVIORAL,
                scope=PolicyScope.GLOBAL,
                priority=PolicyPriority.HIGH,
                conditions=[
                    {
                        "condition_type": "threat_categories",
                        "operator": "intersects",
                        "value": ["COMMAND_INJECTION", "PRIVILEGE_ESCALATION"]
                    }
                ],
                actions=[PolicyAction.ESCALATE, PolicyAction.MONITOR]
            )
            
            # Off-hours activity policy
            await self.create_policy(
                name="Off-Hours Activity Monitoring",
                description="Enhanced monitoring for off-hours activity",
                policy_type=PolicyType.TIME_BASED,
                scope=PolicyScope.GLOBAL,
                priority=PolicyPriority.MEDIUM,
                conditions=[
                    {
                        "condition_type": "time_range",
                        "operator": "not_in_list",
                        "value": list(range(6, 23))  # Outside 6 AM - 11 PM
                    },
                    {
                        "condition_type": "risk_level",
                        "operator": "in_list",
                        "value": ["MODERATE", "HIGH", "CRITICAL"]
                    }
                ],
                actions=[PolicyAction.MONITOR, PolicyAction.LOG_ONLY]
            )
            
            # Data exfiltration prevention
            await self.create_policy(
                name="Data Exfiltration Prevention",
                description="Prevent potential data exfiltration",
                policy_type=PolicyType.DATA_ACCESS,
                scope=PolicyScope.GLOBAL,
                priority=PolicyPriority.CRITICAL,
                conditions=[
                    {
                        "condition_type": "threat_categories",
                        "operator": "contains",
                        "value": "DATA_EXFILTRATION"
                    }
                ],
                actions=[PolicyAction.DENY, PolicyAction.ESCALATE]
            )
            
            # Behavioral anomaly escalation
            await self.create_policy(
                name="Behavioral Anomaly Escalation",
                description="Escalate significant behavioral anomalies",
                policy_type=PolicyType.BEHAVIORAL,
                scope=PolicyScope.GLOBAL,
                priority=PolicyPriority.HIGH,
                conditions=[
                    {
                        "condition_type": "threat_detections",
                        "operator": "intersects",
                        "value": ["BEHAVIORAL_ANOMALY", "ADVANCED_PERSISTENT_THREAT"]
                    },
                    {
                        "condition_type": "max_threat_confidence",
                        "operator": "greater_than",
                        "value": 0.8
                    }
                ],
                actions=[PolicyAction.ESCALATE, PolicyAction.MONITOR]
            )
            
            logger.info(f"Initialized {len(self.policies)} default security policies")
            
        except Exception as e:
            logger.error(f"Failed to initialize default policies: {e}")
    
    async def _log_policy_evaluation(
        self,
        context: CommandContext,
        result: PolicyEvaluationResult
    ) -> None:
        """Log policy evaluation for audit and analysis."""
        
        if result.matched_policies or result.decision != ControlDecision.ALLOW:
            logger.info(
                "Security policy evaluation completed",
                agent_id=str(context.agent_id),
                command_hash=hashlib.sha256(context.command.encode()).hexdigest()[:16],
                decision=result.decision.value,
                matched_policies=[p.id for p in result.matched_policies],
                confidence=result.confidence,
                evaluation_time_ms=result.evaluation_time_ms
            )
        
        # Create security events for significant policy matches
        for policy in result.matched_policies:
            if policy.priority in [PolicyPriority.CRITICAL, PolicyPriority.HIGH]:
                security_event = SecurityEvent(
                    id=uuid.uuid4(),
                    event_type=AuditEventType.POLICY_VIOLATION,
                    threat_level=ThreatLevel.HIGH if policy.priority == PolicyPriority.CRITICAL else ThreatLevel.MEDIUM,
                    agent_id=context.agent_id,
                    context_id=None,
                    session_id=context.session_id,
                    description=f"Security policy triggered: {policy.name}",
                    details={
                        "policy_id": policy.id,
                        "policy_name": policy.name,
                        "policy_type": policy.policy_type.value,
                        "actions_taken": [action.value for action in result.policy_actions],
                        "command_hash": hashlib.sha256(context.command.encode()).hexdigest()[:16]
                    },
                    timestamp=context.timestamp
                )
                
                await self.audit_system._handle_security_event(security_event)
    
    def _update_metrics(
        self,
        result: PolicyEvaluationResult,
        matched_policies: List[Tuple[SecurityPolicy, List[PolicyAction], float]]
    ) -> None:
        """Update engine performance metrics."""
        
        self.metrics["evaluations_performed"] += 1
        
        if matched_policies:
            self.metrics["policies_matched"] += len(matched_policies)
        
        if len(matched_policies) > 1:
            self.metrics["conflicts_resolved"] += 1
        
        # Update average evaluation time
        current_avg = self.metrics["avg_evaluation_time_ms"]
        total_evaluations = self.metrics["evaluations_performed"]
        self.metrics["avg_evaluation_time_ms"] = (
            (current_avg * (total_evaluations - 1) + result.evaluation_time_ms) / total_evaluations
        )
        
        # Update policy effectiveness scores
        for policy, actions, confidence in matched_policies:
            self.metrics["policy_effectiveness_scores"][policy.id] = policy.get_effectiveness_score()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive engine metrics."""
        
        return {
            "security_policy_engine": self.metrics.copy(),
            "total_policies": len(self.policies),
            "enabled_policies": len([p for p in self.policies.values() if p.enabled]),
            "policies_by_type": dict(Counter(p.policy_type.value for p in self.policies.values())),
            "policies_by_scope": dict(Counter(p.scope.value for p in self.policies.values())),
            "policies_by_priority": dict(Counter(p.priority.name for p in self.policies.values())),
            "configuration": self.config.copy()
        }
    
    def get_policy_statistics(self) -> Dict[str, Any]:
        """Get detailed policy statistics."""
        
        policies = list(self.policies.values())
        
        if not policies:
            return {"error": "No policies found"}
        
        effectiveness_scores = [p.get_effectiveness_score() for p in policies]
        evaluation_counts = [p.evaluation_count for p in policies]
        match_counts = [p.match_count for p in policies]
        
        return {
            "total_policies": len(policies),
            "effectiveness_statistics": {
                "mean": np.mean(effectiveness_scores) if effectiveness_scores else 0.0,
                "median": np.median(effectiveness_scores) if effectiveness_scores else 0.0,
                "min": min(effectiveness_scores) if effectiveness_scores else 0.0,
                "max": max(effectiveness_scores) if effectiveness_scores else 0.0
            },
            "evaluation_statistics": {
                "total_evaluations": sum(evaluation_counts),
                "total_matches": sum(match_counts),
                "average_evaluations_per_policy": np.mean(evaluation_counts) if evaluation_counts else 0.0,
                "match_rate": sum(match_counts) / max(sum(evaluation_counts), 1)
            },
            "most_effective_policies": [
                {"id": p.id, "name": p.name, "effectiveness": p.get_effectiveness_score()}
                for p in sorted(policies, key=lambda x: x.get_effectiveness_score(), reverse=True)[:5]
            ],
            "most_active_policies": [
                {"id": p.id, "name": p.name, "evaluations": p.evaluation_count, "matches": p.match_count}
                for p in sorted(policies, key=lambda x: x.evaluation_count, reverse=True)[:5]
            ]
        }


# Factory function
async def create_security_policy_engine(
    db_session: AsyncSession,
    authorization_engine: AuthorizationEngine,
    audit_system: SecurityAuditSystem
) -> SecurityPolicyEngine:
    """
    Create SecurityPolicyEngine instance.
    
    Args:
        db_session: Database session
        authorization_engine: Authorization engine
        audit_system: Security audit system
        
    Returns:
        SecurityPolicyEngine instance
    """
    return SecurityPolicyEngine(db_session, authorization_engine, audit_system)