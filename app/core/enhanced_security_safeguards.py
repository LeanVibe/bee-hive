"""
Enhanced Security Safeguards & Deterministic Control System for LeanVibe Agent Hive 2.0.

This system implements multi-layer security protection with deterministic control mechanisms
for agent behavior, ensuring predictable and secure agent operations in production environments.

Features:
- Deterministic rule-based decision engine
- Multi-layer security validation
- Risk-based access control with dynamic permissions
- Real-time agent behavior monitoring
- Automated security policy enforcement
- Incident response automation
- Comprehensive audit trails and compliance validation
"""

import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from enum import Enum, auto
from dataclasses import dataclass, field
import logging
import hashlib
from collections import defaultdict, deque

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from .security_middleware import SecurityMiddleware
from .security_audit import SecurityAuditSystem, ThreatLevel, AuditEventType, SecurityEvent
from .code_execution import SecurityAnalyzer, SecurityLevel, CodeBlock
from ..models.agent import Agent
from ..models.context import Context
# Note: SecurityPolicy and AgentSecurityProfile models would be imported here if they exist
# from ..models.security import SecurityPolicy, AgentSecurityProfile
from ..core.database import get_async_session

logger = structlog.get_logger()


class ControlDecision(Enum):
    """Deterministic control decisions."""
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"
    QUARANTINE = "QUARANTINE"
    ESCALATE = "ESCALATE"


class SecurityRiskLevel(Enum):
    """Security risk assessment levels."""
    MINIMAL = "MINIMAL"      # 0.0-0.2
    LOW = "LOW"             # 0.2-0.4  
    MODERATE = "MODERATE"   # 0.4-0.6
    HIGH = "HIGH"           # 0.6-0.8
    CRITICAL = "CRITICAL"   # 0.8-1.0


class AgentBehaviorState(Enum):
    """Agent behavior monitoring states."""
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    ANOMALOUS = "ANOMALOUS"
    COMPROMISED = "COMPROMISED"
    QUARANTINED = "QUARANTINED"


class SecurityPolicyType(Enum):
    """Types of security policies."""
    CODE_EXECUTION = "CODE_EXECUTION"
    DATA_ACCESS = "DATA_ACCESS"
    NETWORK_ACCESS = "NETWORK_ACCESS"
    FILE_OPERATIONS = "FILE_OPERATIONS"
    CROSS_AGENT_ACCESS = "CROSS_AGENT_ACCESS"
    TIME_BASED_ACCESS = "TIME_BASED_ACCESS"
    RESOURCE_USAGE = "RESOURCE_USAGE"
    BEHAVIORAL_ANALYSIS = "BEHAVIORAL_ANALYSIS"


@dataclass
class SecurityRule:
    """Individual security rule for deterministic control."""
    id: str
    name: str
    policy_type: SecurityPolicyType
    conditions: Dict[str, Any]
    decision: ControlDecision
    priority: int  # Higher = more important
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluate if this rule matches the given context.
        
        Returns:
            Tuple of (matches, reason)
        """
        try:
            for condition_key, condition_value in self.conditions.items():
                context_value = context.get(condition_key)
                
                if isinstance(condition_value, dict):
                    # Complex condition (e.g., {"operator": ">=", "value": 0.8})
                    operator = condition_value.get("operator", "==")
                    expected = condition_value.get("value")
                    
                    if not self._evaluate_condition(context_value, operator, expected):
                        return False, f"Condition {condition_key} {operator} {expected} failed"
                        
                elif isinstance(condition_value, list):
                    # Value must be in list
                    if context_value not in condition_value:
                        return False, f"Value {context_value} not in allowed list"
                        
                else:
                    # Simple equality check
                    if context_value != condition_value:
                        return False, f"Value {context_value} != {condition_value}"
            
            return True, "All conditions met"
            
        except Exception as e:
            logger.error(f"Rule evaluation error: {e}")
            return False, f"Evaluation error: {str(e)}"
    
    def _evaluate_condition(self, actual: Any, operator: str, expected: Any) -> bool:
        """Evaluate condition with operator."""
        if actual is None or expected is None:
            return False
            
        try:
            if operator == "==":
                return actual == expected
            elif operator == "!=":
                return actual != expected
            elif operator == ">":
                return float(actual) > float(expected)
            elif operator == ">=":
                return float(actual) >= float(expected)
            elif operator == "<":
                return float(actual) < float(expected)
            elif operator == "<=":
                return float(actual) <= float(expected)
            elif operator == "in":
                return actual in expected
            elif operator == "not_in":
                return actual not in expected
            elif operator == "contains":
                return expected in str(actual)
            elif operator == "matches":
                import re
                return bool(re.search(expected, str(actual)))
            else:
                return False
        except (ValueError, TypeError):
            return False


@dataclass
class AgentBehaviorProfile:
    """Agent behavior profile for monitoring."""
    agent_id: uuid.UUID
    behavior_state: AgentBehaviorState = AgentBehaviorState.NORMAL
    risk_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Behavior metrics
    action_count_24h: int = 0
    failed_attempts_24h: int = 0
    code_executions_24h: int = 0
    data_access_count_24h: int = 0
    anomaly_score: float = 0.0
    
    # Historical patterns
    typical_actions_per_hour: float = 0.0
    typical_failure_rate: float = 0.0
    behavioral_baseline: Dict[str, float] = field(default_factory=dict)
    
    # Alerts and incidents
    active_alerts: List[str] = field(default_factory=list)
    quarantine_reason: Optional[str] = None
    quarantine_until: Optional[datetime] = None


@dataclass
class SecurityContext:
    """Context for security decision making."""
    agent_id: uuid.UUID
    session_id: Optional[uuid.UUID]
    action_type: str
    resource_type: str
    resource_id: Optional[str]
    
    # Request context
    timestamp: datetime
    ip_address: Optional[str]
    user_agent: Optional[str]
    
    # Agent context
    agent_type: Optional[str]
    agent_capabilities: List[str]
    current_risk_score: float
    behavior_state: AgentBehaviorState
    
    # Resource context
    data_sensitivity: str = "normal"  # minimal, normal, sensitive, confidential
    operation_impact: str = "low"     # low, medium, high, critical
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class DeterministicControlEngine:
    """
    Deterministic control engine for security decision making.
    
    Provides rule-based, predictable security decisions with comprehensive
    audit trails and policy enforcement capabilities.
    """
    
    def __init__(self):
        self.security_rules: List[SecurityRule] = []
        self.default_decision = ControlDecision.DENY
        self.decision_cache: Dict[str, Tuple[ControlDecision, str, datetime]] = {}
        self.cache_ttl = timedelta(minutes=5)
        
        # Performance metrics
        self.metrics = {
            "decisions_made": 0,
            "allow_decisions": 0,
            "deny_decisions": 0,
            "escalation_decisions": 0,
            "cache_hits": 0,
            "rule_evaluations": 0,
            "avg_decision_time_ms": 0.0
        }
        
        # Initialize with default security rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self) -> None:
        """Initialize default security rules."""
        
        # Critical security rules (highest priority)
        self.add_rule(SecurityRule(
            id="block_dangerous_code",
            name="Block Dangerous Code Execution",
            policy_type=SecurityPolicyType.CODE_EXECUTION,
            conditions={
                "security_level": ["DANGEROUS"],
                "has_exec_statements": True
            },
            decision=ControlDecision.DENY,
            priority=1000,
            metadata={"description": "Block code with dangerous operations"}
        ))
        
        self.add_rule(SecurityRule(
            id="quarantine_compromised_agents",
            name="Quarantine Compromised Agents",
            policy_type=SecurityPolicyType.BEHAVIORAL_ANALYSIS,
            conditions={
                "behavior_state": ["COMPROMISED"],
                "risk_score": {"operator": ">=", "value": 0.9}
            },
            decision=ControlDecision.QUARANTINE,
            priority=1000,
            metadata={"description": "Quarantine agents showing signs of compromise"}
        ))
        
        # High priority rules
        self.add_rule(SecurityRule(
            id="require_approval_high_risk",
            name="Require Approval for High Risk Operations",
            policy_type=SecurityPolicyType.DATA_ACCESS,
            conditions={
                "risk_score": {"operator": ">=", "value": 0.8},
                "data_sensitivity": ["confidential", "sensitive"]
            },
            decision=ControlDecision.REQUIRE_APPROVAL,
            priority=900,
            metadata={"description": "High risk operations require human approval"}
        ))
        
        self.add_rule(SecurityRule(
            id="escalate_anomalous_behavior",
            name="Escalate Anomalous Agent Behavior",
            policy_type=SecurityPolicyType.BEHAVIORAL_ANALYSIS,
            conditions={
                "behavior_state": ["ANOMALOUS"],
                "anomaly_score": {"operator": ">=", "value": 0.7}
            },
            decision=ControlDecision.ESCALATE,
            priority=800,
            metadata={"description": "Escalate anomalous agent behavior"}
        ))
        
        # Medium priority rules
        self.add_rule(SecurityRule(
            id="limit_cross_agent_access",
            name="Limit Cross-Agent Data Access",
            policy_type=SecurityPolicyType.CROSS_AGENT_ACCESS,
            conditions={
                "action_type": ["cross_agent_access"],
                "cross_agent_ratio": {"operator": ">", "value": 0.7}
            },
            decision=ControlDecision.REQUIRE_APPROVAL,
            priority=700,
            metadata={"description": "Limit excessive cross-agent access"}
        ))
        
        self.add_rule(SecurityRule(
            id="block_off_hours_sensitive",
            name="Block Off-Hours Sensitive Access",
            policy_type=SecurityPolicyType.TIME_BASED_ACCESS,
            conditions={
                "is_off_hours": True,
                "data_sensitivity": ["confidential", "sensitive"]
            },
            decision=ControlDecision.DENY,
            priority=600,
            metadata={"description": "Block sensitive access outside business hours"}
        ))
        
        # Default allow rule (lowest priority)
        self.add_rule(SecurityRule(
            id="default_allow",
            name="Default Allow for Normal Operations",
            policy_type=SecurityPolicyType.DATA_ACCESS,
            conditions={
                "risk_score": {"operator": "<", "value": 0.5},
                "behavior_state": ["NORMAL"]
            },
            decision=ControlDecision.ALLOW,
            priority=1,
            metadata={"description": "Allow normal operations for low-risk agents"}
        ))
    
    def add_rule(self, rule: SecurityRule) -> None:
        """Add a new security rule."""
        self.security_rules.append(rule)
        # Sort by priority (descending)
        self.security_rules.sort(key=lambda r: r.priority, reverse=True)
        logger.info(f"Added security rule: {rule.name}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a security rule by ID."""
        original_count = len(self.security_rules)
        self.security_rules = [r for r in self.security_rules if r.id != rule_id]
        removed = len(self.security_rules) < original_count
        if removed:
            logger.info(f"Removed security rule: {rule_id}")
        return removed
    
    async def make_decision(self, context: SecurityContext) -> Tuple[ControlDecision, str, List[str]]:
        """
        Make a deterministic security decision based on context.
        
        Returns:
            Tuple of (decision, reason, applied_rules)
        """
        start_time = datetime.utcnow()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(context)
            cached_decision = self._get_cached_decision(cache_key)
            if cached_decision:
                self.metrics["cache_hits"] += 1
                return cached_decision[0], cached_decision[1], []
            
            # Evaluate rules in priority order
            context_dict = self._context_to_dict(context)
            applied_rules = []
            
            for rule in self.security_rules:
                if not rule.enabled:
                    continue
                
                self.metrics["rule_evaluations"] += 1
                matches, reason = rule.evaluate(context_dict)
                
                if matches:
                    applied_rules.append(rule.id)
                    decision = rule.decision
                    full_reason = f"Rule '{rule.name}': {reason}"
                    
                    # Cache the decision
                    self._cache_decision(cache_key, decision, full_reason)
                    
                    # Update metrics
                    self._update_decision_metrics(decision, start_time)
                    
                    logger.info(
                        f"Security decision made",
                        agent_id=str(context.agent_id),
                        decision=decision.value,
                        rule_applied=rule.name,
                        reason=full_reason
                    )
                    
                    return decision, full_reason, applied_rules
            
            # No rules matched, use default decision
            default_reason = f"No rules matched, using default: {self.default_decision.value}"
            self._cache_decision(cache_key, self.default_decision, default_reason)
            self._update_decision_metrics(self.default_decision, start_time)
            
            logger.warning(
                f"No security rules matched, using default decision",
                agent_id=str(context.agent_id),
                decision=self.default_decision.value
            )
            
            return self.default_decision, default_reason, []
            
        except Exception as e:
            logger.error(f"Security decision error: {e}")
            error_reason = f"Decision engine error: {str(e)}"
            self._update_decision_metrics(ControlDecision.DENY, start_time)
            return ControlDecision.DENY, error_reason, []
    
    def _context_to_dict(self, context: SecurityContext) -> Dict[str, Any]:
        """Convert SecurityContext to dictionary for rule evaluation."""
        current_hour = context.timestamp.hour
        is_off_hours = current_hour < 6 or current_hour > 22  # Outside 6 AM - 10 PM
        
        return {
            "agent_id": str(context.agent_id),
            "action_type": context.action_type,
            "resource_type": context.resource_type,
            "data_sensitivity": context.data_sensitivity,
            "operation_impact": context.operation_impact,
            "risk_score": context.current_risk_score,
            "behavior_state": context.behavior_state.value,
            "agent_type": context.agent_type,
            "is_off_hours": is_off_hours,
            "hour": current_hour,
            **context.metadata
        }
    
    def _generate_cache_key(self, context: SecurityContext) -> str:
        """Generate cache key for decision caching."""
        key_data = {
            "agent_id": str(context.agent_id),
            "action_type": context.action_type,
            "resource_type": context.resource_type,
            "data_sensitivity": context.data_sensitivity,
            "behavior_state": context.behavior_state.value,
            "risk_score_bucket": int(context.current_risk_score * 10) / 10  # Round to 0.1
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _get_cached_decision(self, cache_key: str) -> Optional[Tuple[ControlDecision, str]]:
        """Get cached decision if still valid."""
        if cache_key in self.decision_cache:
            decision, reason, cached_at = self.decision_cache[cache_key]
            if datetime.utcnow() - cached_at < self.cache_ttl:
                return decision, reason
            else:
                # Remove expired cache entry
                del self.decision_cache[cache_key]
        return None
    
    def _cache_decision(self, cache_key: str, decision: ControlDecision, reason: str) -> None:
        """Cache decision with timestamp."""
        self.decision_cache[cache_key] = (decision, reason, datetime.utcnow())
        
        # Limit cache size
        if len(self.decision_cache) > 1000:
            # Remove oldest entries
            sorted_cache = sorted(
                self.decision_cache.items(),
                key=lambda x: x[1][2]  # Sort by timestamp
            )
            # Keep newest 800 entries
            self.decision_cache = dict(sorted_cache[-800:])
    
    def _update_decision_metrics(self, decision: ControlDecision, start_time: datetime) -> None:
        """Update decision metrics."""
        self.metrics["decisions_made"] += 1
        
        if decision == ControlDecision.ALLOW:
            self.metrics["allow_decisions"] += 1
        elif decision == ControlDecision.DENY:
            self.metrics["deny_decisions"] += 1
        elif decision in [ControlDecision.ESCALATE, ControlDecision.REQUIRE_APPROVAL]:
            self.metrics["escalation_decisions"] += 1
        
        # Update average decision time
        decision_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        current_avg = self.metrics["avg_decision_time_ms"]
        total_decisions = self.metrics["decisions_made"]
        self.metrics["avg_decision_time_ms"] = (
            (current_avg * (total_decisions - 1) + decision_time) / total_decisions
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get engine performance metrics."""
        return {
            "deterministic_control_engine": self.metrics.copy(),
            "rules_count": len(self.security_rules),
            "cache_size": len(self.decision_cache),
            "cache_ttl_minutes": self.cache_ttl.total_seconds() / 60
        }


class AgentBehaviorMonitor:
    """
    Real-time agent behavior monitoring and anomaly detection.
    
    Tracks agent patterns, detects anomalies, and updates risk profiles
    for use in security decision making.
    """
    
    def __init__(self):
        self.behavior_profiles: Dict[uuid.UUID, AgentBehaviorProfile] = {}
        self.behavior_history: Dict[uuid.UUID, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Anomaly detection thresholds
        self.anomaly_thresholds = {
            "action_rate_multiplier": 3.0,    # 3x normal action rate
            "failure_rate_threshold": 0.3,    # 30% failure rate
            "off_hours_threshold": 0.2,       # 20% of actions off-hours
            "cross_agent_threshold": 0.8,     # 80% cross-agent access
            "pattern_deviation": 2.0           # 2 standard deviations
        }
        
        self.monitoring_window = timedelta(hours=24)
    
    async def update_agent_behavior(
        self,
        agent_id: uuid.UUID,
        action_type: str,
        success: bool,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentBehaviorProfile:
        """Update agent behavior profile with new action."""
        timestamp = timestamp or datetime.utcnow()
        metadata = metadata or {}
        
        # Get or create behavior profile
        if agent_id not in self.behavior_profiles:
            self.behavior_profiles[agent_id] = AgentBehaviorProfile(agent_id=agent_id)
        
        profile = self.behavior_profiles[agent_id]
        
        # Record behavior event
        behavior_event = {
            "timestamp": timestamp,
            "action_type": action_type,
            "success": success,
            "metadata": metadata
        }
        self.behavior_history[agent_id].append(behavior_event)
        
        # Update 24-hour metrics
        await self._update_24h_metrics(profile, timestamp)
        
        # Calculate anomaly score
        anomaly_score = await self._calculate_anomaly_score(profile, agent_id)
        profile.anomaly_score = anomaly_score
        
        # Update behavior state
        new_state = await self._assess_behavior_state(profile, anomaly_score)
        if new_state != profile.behavior_state:
            logger.info(
                f"Agent behavior state changed",
                agent_id=str(agent_id),
                old_state=profile.behavior_state.value,
                new_state=new_state.value,
                anomaly_score=anomaly_score
            )
            profile.behavior_state = new_state
        
        # Update risk score
        profile.risk_score = await self._calculate_risk_score(profile, anomaly_score)
        profile.last_updated = timestamp
        
        return profile
    
    async def _update_24h_metrics(self, profile: AgentBehaviorProfile, timestamp: datetime) -> None:
        """Update 24-hour rolling metrics."""
        cutoff_time = timestamp - self.monitoring_window
        recent_events = [
            event for event in self.behavior_history[profile.agent_id]
            if event["timestamp"] >= cutoff_time
        ]
        
        # Update action counts
        profile.action_count_24h = len(recent_events)
        profile.failed_attempts_24h = sum(1 for event in recent_events if not event["success"])
        profile.code_executions_24h = sum(1 for event in recent_events if event["action_type"] == "code_execution")
        profile.data_access_count_24h = sum(1 for event in recent_events if event["action_type"] == "data_access")
    
    async def _calculate_anomaly_score(self, profile: AgentBehaviorProfile, agent_id: uuid.UUID) -> float:
        """Calculate anomaly score based on behavioral deviations."""
        score = 0.0
        
        # Check action rate anomaly
        if profile.typical_actions_per_hour > 0:
            current_rate = profile.action_count_24h / 24.0
            rate_multiplier = current_rate / profile.typical_actions_per_hour
            if rate_multiplier > self.anomaly_thresholds["action_rate_multiplier"]:
                score += 0.3
        
        # Check failure rate anomaly
        if profile.action_count_24h > 0:
            current_failure_rate = profile.failed_attempts_24h / profile.action_count_24h
            if current_failure_rate > self.anomaly_thresholds["failure_rate_threshold"]:
                score += 0.4
        
        # Check for time-based anomalies
        recent_events = list(self.behavior_history[agent_id])[-100:]  # Last 100 events
        off_hours_count = sum(
            1 for event in recent_events
            if event["timestamp"].hour < 6 or event["timestamp"].hour > 22
        )
        if recent_events and off_hours_count / len(recent_events) > self.anomaly_thresholds["off_hours_threshold"]:
            score += 0.2
        
        # Check for cross-agent access anomalies
        cross_agent_count = sum(
            1 for event in recent_events
            if event["metadata"].get("cross_agent_access", False)
        )
        if recent_events and cross_agent_count / len(recent_events) > self.anomaly_thresholds["cross_agent_threshold"]:
            score += 0.3
        
        return min(1.0, score)
    
    async def _assess_behavior_state(
        self,
        profile: AgentBehaviorProfile,
        anomaly_score: float
    ) -> AgentBehaviorState:
        """Assess agent behavior state based on anomaly score and patterns."""
        
        # Check for quarantine conditions
        if profile.quarantine_until and datetime.utcnow() < profile.quarantine_until:
            return AgentBehaviorState.QUARANTINED
        
        # Assess based on anomaly score
        if anomaly_score >= 0.9:
            return AgentBehaviorState.COMPROMISED
        elif anomaly_score >= 0.7:
            return AgentBehaviorState.ANOMALOUS
        elif anomaly_score >= 0.4:
            return AgentBehaviorState.SUSPICIOUS
        else:
            return AgentBehaviorState.NORMAL
    
    async def _calculate_risk_score(self, profile: AgentBehaviorProfile, anomaly_score: float) -> float:
        """Calculate overall risk score for the agent."""
        risk_score = 0.0
        
        # Base risk from anomaly score
        risk_score += anomaly_score * 0.5
        
        # Risk from failure rate
        if profile.action_count_24h > 0:
            failure_rate = profile.failed_attempts_24h / profile.action_count_24h
            risk_score += failure_rate * 0.3
        
        # Risk from behavior state
        state_risk_map = {
            AgentBehaviorState.NORMAL: 0.0,
            AgentBehaviorState.SUSPICIOUS: 0.2,
            AgentBehaviorState.ANOMALOUS: 0.6,
            AgentBehaviorState.COMPROMISED: 0.9,
            AgentBehaviorState.QUARANTINED: 1.0
        }
        risk_score += state_risk_map.get(profile.behavior_state, 0.0) * 0.2
        
        return min(1.0, risk_score)
    
    def get_agent_profile(self, agent_id: uuid.UUID) -> Optional[AgentBehaviorProfile]:
        """Get agent behavior profile."""
        return self.behavior_profiles.get(agent_id)
    
    async def quarantine_agent(
        self,
        agent_id: uuid.UUID,
        reason: str,
        duration_hours: int = 24
    ) -> None:
        """Quarantine an agent for specified duration."""
        if agent_id not in self.behavior_profiles:
            self.behavior_profiles[agent_id] = AgentBehaviorProfile(agent_id=agent_id)
        
        profile = self.behavior_profiles[agent_id]
        profile.behavior_state = AgentBehaviorState.QUARANTINED
        profile.quarantine_reason = reason
        profile.quarantine_until = datetime.utcnow() + timedelta(hours=duration_hours)
        profile.risk_score = 1.0
        
        logger.warning(
            f"Agent quarantined",
            agent_id=str(agent_id),
            reason=reason,
            duration_hours=duration_hours,
            until=profile.quarantine_until.isoformat()
        )
    
    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """Get behavior monitoring metrics."""
        total_agents = len(self.behavior_profiles)
        state_counts = defaultdict(int)
        risk_distribution = defaultdict(int)
        
        for profile in self.behavior_profiles.values():
            state_counts[profile.behavior_state.value] += 1
            
            # Risk distribution in buckets
            risk_bucket = int(profile.risk_score * 10) / 10
            risk_distribution[f"{risk_bucket:.1f}"] += 1
        
        return {
            "total_agents_monitored": total_agents,
            "behavior_state_distribution": dict(state_counts),
            "risk_score_distribution": dict(risk_distribution),
            "anomaly_thresholds": self.anomaly_thresholds,
            "monitoring_window_hours": self.monitoring_window.total_seconds() / 3600
        }


class EnhancedSecuritySafeguards:
    """
    Enhanced Security Safeguards & Deterministic Control System.
    
    Orchestrates all security components to provide comprehensive protection
    with deterministic control and real-time monitoring capabilities.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        security_audit_system: SecurityAuditSystem,
        code_security_analyzer: SecurityAnalyzer
    ):
        self.db = db_session
        self.audit_system = security_audit_system
        self.code_analyzer = code_security_analyzer
        
        # Core components
        self.control_engine = DeterministicControlEngine()
        self.behavior_monitor = AgentBehaviorMonitor()
        
        # Performance metrics
        self.metrics = {
            "security_checks_performed": 0,
            "decisions_made": 0,
            "threats_blocked": 0,
            "agents_quarantined": 0,
            "approvals_required": 0,
            "avg_check_time_ms": 0.0
        }
        
        # Configuration
        self.config = {
            "enable_real_time_monitoring": True,
            "enable_behavioral_analysis": True,
            "enable_deterministic_control": True,
            "auto_quarantine_enabled": True,
            "require_approval_for_high_risk": True,
            "max_risk_score_for_auto_approval": 0.6
        }
    
    async def validate_agent_action(
        self,
        agent_id: uuid.UUID,
        action_type: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        session_id: Optional[uuid.UUID] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[ControlDecision, str, Dict[str, Any]]:
        """
        Validate agent action through comprehensive security checks.
        
        Returns:
            Tuple of (decision, reason, additional_data)
        """
        start_time = datetime.utcnow()
        metadata = metadata or {}
        
        try:
            # Get or update agent behavior profile
            behavior_profile = await self.behavior_monitor.update_agent_behavior(
                agent_id=agent_id,
                action_type=action_type,
                success=True,  # Will be updated after actual execution
                metadata=metadata
            )
            
            # Create security context
            security_context = SecurityContext(
                agent_id=agent_id,
                session_id=session_id,
                action_type=action_type,
                resource_type=resource_type,
                resource_id=resource_id,
                timestamp=datetime.utcnow(),
                ip_address=metadata.get("ip_address"),
                user_agent=metadata.get("user_agent"),
                agent_type=metadata.get("agent_type"),
                agent_capabilities=metadata.get("capabilities", []),
                current_risk_score=behavior_profile.risk_score,
                behavior_state=behavior_profile.behavior_state,
                data_sensitivity=metadata.get("data_sensitivity", "normal"),
                operation_impact=metadata.get("operation_impact", "low"),
                metadata=metadata
            )
            
            # Make security decision
            decision, reason, applied_rules = await self.control_engine.make_decision(security_context)
            
            # Handle decision
            additional_data = await self._handle_security_decision(
                decision, security_context, behavior_profile, applied_rules
            )
            
            # Update metrics
            self._update_metrics(decision, start_time)
            
            # Audit the decision
            await self._audit_security_decision(security_context, decision, reason)
            
            logger.info(
                f"Agent action validated",
                agent_id=str(agent_id),
                action_type=action_type,
                decision=decision.value,
                reason=reason,
                risk_score=behavior_profile.risk_score
            )
            
            return decision, reason, additional_data
            
        except Exception as e:
            logger.error(f"Security validation error: {e}")
            self._update_metrics(ControlDecision.DENY, start_time)
            return ControlDecision.DENY, f"Security validation error: {str(e)}", {}
    
    async def validate_code_execution(
        self,
        agent_id: uuid.UUID,
        code_block: CodeBlock,
        session_id: Optional[uuid.UUID] = None
    ) -> Tuple[ControlDecision, str, Dict[str, Any]]:
        """Validate code execution with enhanced security checks."""
        
        # Perform security analysis on the code
        security_result = await self.code_analyzer.analyze_security(code_block)
        
        # Determine action type based on security level
        if security_result.security_level == SecurityLevel.DANGEROUS:
            action_type = "dangerous_code_execution"
        elif security_result.security_level == SecurityLevel.RESTRICTED:
            action_type = "restricted_code_execution"
        else:
            action_type = "code_execution"
        
        # Prepare metadata
        metadata = {
            "code_language": code_block.language.value,
            "security_level": security_result.security_level.value,
            "has_file_operations": security_result.has_file_operations,
            "has_network_access": security_result.has_network_access,
            "has_system_calls": security_result.has_system_calls,
            "has_dangerous_imports": security_result.has_dangerous_imports,
            "has_exec_statements": security_result.has_exec_statements,
            "threats_detected": security_result.threats_detected,
            "safe_to_execute": security_result.safe_to_execute,
            "confidence_score": security_result.confidence_score
        }
        
        # Validate through standard process
        decision, reason, additional_data = await self.validate_agent_action(
            agent_id=agent_id,
            action_type=action_type,
            resource_type="code_execution",
            resource_id=code_block.id,
            session_id=session_id,
            metadata=metadata
        )
        
        # Add code-specific data
        additional_data.update({
            "security_analysis": security_result,
            "code_block_id": code_block.id
        })
        
        return decision, reason, additional_data
    
    async def _handle_security_decision(
        self,
        decision: ControlDecision,
        context: SecurityContext,
        behavior_profile: AgentBehaviorProfile,
        applied_rules: List[str]
    ) -> Dict[str, Any]:
        """Handle the security decision and take appropriate actions."""
        
        additional_data = {
            "applied_rules": applied_rules,
            "behavior_profile": {
                "state": behavior_profile.behavior_state.value,
                "risk_score": behavior_profile.risk_score,
                "anomaly_score": behavior_profile.anomaly_score
            }
        }
        
        if decision == ControlDecision.QUARANTINE:
            # Quarantine the agent
            await self.behavior_monitor.quarantine_agent(
                agent_id=context.agent_id,
                reason=f"Quarantined due to security decision for {context.action_type}",
                duration_hours=24
            )
            self.metrics["agents_quarantined"] += 1
            
            # Create security event
            await self.audit_system.audit_context_access(
                context_id=uuid.uuid4(),  # Placeholder
                agent_id=context.agent_id,
                session_id=context.session_id,
                access_granted=False,
                permission="quarantine",
                access_time=context.timestamp
            )
            
            additional_data["quarantine_duration_hours"] = 24
            
        elif decision == ControlDecision.REQUIRE_APPROVAL:
            self.metrics["approvals_required"] += 1
            additional_data["approval_required"] = True
            additional_data["approval_reason"] = "High risk operation requires human approval"
            
        elif decision == ControlDecision.ESCALATE:
            # Generate security alert
            security_event = SecurityEvent(
                id=uuid.uuid4(),
                event_type=AuditEventType.SUSPICIOUS_PATTERN,
                threat_level=ThreatLevel.HIGH,
                agent_id=context.agent_id,
                context_id=uuid.uuid4(),  # Placeholder
                session_id=context.session_id,
                description=f"Escalated security decision for {context.action_type}",
                details={
                    "action_type": context.action_type,
                    "resource_type": context.resource_type,
                    "risk_score": context.current_risk_score,
                    "behavior_state": context.behavior_state.value
                },
                timestamp=context.timestamp
            )
            
            await self.audit_system._handle_security_event(security_event)
            additional_data["security_event_id"] = str(security_event.id)
            
        elif decision == ControlDecision.DENY:
            self.metrics["threats_blocked"] += 1
            
        return additional_data
    
    async def _audit_security_decision(
        self,
        context: SecurityContext,
        decision: ControlDecision,
        reason: str
    ) -> None:
        """Audit the security decision for compliance and analysis."""
        
        audit_data = {
            "decision": decision.value,
            "reason": reason,
            "agent_id": str(context.agent_id),
            "action_type": context.action_type,
            "resource_type": context.resource_type,
            "risk_score": context.current_risk_score,
            "behavior_state": context.behavior_state.value,
            "timestamp": context.timestamp.isoformat()
        }
        
        # Log to audit system
        logger.info(
            "Security decision audit",
            **audit_data
        )
    
    def _update_metrics(self, decision: ControlDecision, start_time: datetime) -> None:
        """Update security safeguards metrics."""
        self.metrics["security_checks_performed"] += 1
        self.metrics["decisions_made"] += 1
        
        # Update average check time
        check_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        current_avg = self.metrics["avg_check_time_ms"]
        total_checks = self.metrics["security_checks_performed"]
        self.metrics["avg_check_time_ms"] = (
            (current_avg * (total_checks - 1) + check_time) / total_checks
        )
    
    async def get_agent_security_status(self, agent_id: uuid.UUID) -> Dict[str, Any]:
        """Get comprehensive security status for an agent."""
        behavior_profile = self.behavior_monitor.get_agent_profile(agent_id)
        
        if not behavior_profile:
            return {
                "agent_id": str(agent_id),
                "status": "unknown",
                "message": "No behavior profile found"
            }
        
        status = {
            "agent_id": str(agent_id),
            "behavior_state": behavior_profile.behavior_state.value,
            "risk_score": behavior_profile.risk_score,
            "anomaly_score": behavior_profile.anomaly_score,
            "last_updated": behavior_profile.last_updated.isoformat(),
            "metrics_24h": {
                "total_actions": behavior_profile.action_count_24h,
                "failed_attempts": behavior_profile.failed_attempts_24h,
                "code_executions": behavior_profile.code_executions_24h,
                "data_accesses": behavior_profile.data_access_count_24h
            },
            "security_status": "normal"
        }
        
        # Determine overall security status
        if behavior_profile.behavior_state == AgentBehaviorState.QUARANTINED:
            status["security_status"] = "quarantined"
            status["quarantine_reason"] = behavior_profile.quarantine_reason
            status["quarantine_until"] = behavior_profile.quarantine_until.isoformat() if behavior_profile.quarantine_until else None
        elif behavior_profile.behavior_state == AgentBehaviorState.COMPROMISED:
            status["security_status"] = "compromised"
        elif behavior_profile.behavior_state == AgentBehaviorState.ANOMALOUS:
            status["security_status"] = "anomalous"
        elif behavior_profile.behavior_state == AgentBehaviorState.SUSPICIOUS:
            status["security_status"] = "suspicious"
        
        return status
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics."""
        return {
            "enhanced_security_safeguards": self.metrics.copy(),
            "deterministic_control_engine": self.control_engine.get_metrics(),
            "behavior_monitoring": self.behavior_monitor.get_monitoring_metrics(),
            "configuration": self.config.copy()
        }


# Global enhanced security safeguards instance
_enhanced_security_safeguards: Optional[EnhancedSecuritySafeguards] = None


async def get_enhanced_security_safeguards() -> EnhancedSecuritySafeguards:
    """Get global enhanced security safeguards instance."""
    global _enhanced_security_safeguards
    
    if _enhanced_security_safeguards is None:
        # Initialize dependencies
        db_session = await get_async_session().__anext__()
        
        # This would typically be injected
        from .security_audit import SecurityAuditSystem
        from .access_control import AccessControlManager
        audit_system = SecurityAuditSystem(db_session, AccessControlManager(db_session))
        
        from .code_execution import SecurityAnalyzer
        from anthropic import AsyncAnthropic
        from .config import settings
        code_analyzer = SecurityAnalyzer(AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY))
        
        _enhanced_security_safeguards = EnhancedSecuritySafeguards(
            db_session=db_session,
            security_audit_system=audit_system,
            code_security_analyzer=code_analyzer
        )
    
    return _enhanced_security_safeguards


# Convenience functions for common operations
async def validate_agent_action(
    agent_id: uuid.UUID,
    action_type: str,
    resource_type: str,
    **kwargs
) -> Tuple[ControlDecision, str, Dict[str, Any]]:
    """Convenience function for validating agent actions."""
    safeguards = await get_enhanced_security_safeguards()
    return await safeguards.validate_agent_action(
        agent_id=agent_id,
        action_type=action_type,
        resource_type=resource_type,
        **kwargs
    )


async def validate_code_execution(
    agent_id: uuid.UUID,
    code_block: CodeBlock,
    session_id: Optional[uuid.UUID] = None
) -> Tuple[ControlDecision, str, Dict[str, Any]]:
    """Convenience function for validating code execution."""
    safeguards = await get_enhanced_security_safeguards()
    return await safeguards.validate_code_execution(
        agent_id=agent_id,
        code_block=code_block,
        session_id=session_id
    )


async def get_agent_security_status(agent_id: uuid.UUID) -> Dict[str, Any]:
    """Convenience function for getting agent security status."""
    safeguards = await get_enhanced_security_safeguards()
    return await safeguards.get_agent_security_status(agent_id)