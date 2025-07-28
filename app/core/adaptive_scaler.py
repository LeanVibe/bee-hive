"""
Adaptive Scaler for LeanVibe Agent Hive 2.0

Implements intelligent auto-scaling based on workload patterns, performance trends,
and predictive analytics for optimal resource utilization and cost efficiency.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import math
import json

import structlog
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .redis import get_message_broker, get_session_cache, AgentMessageBroker, SessionCache
from .database import get_session
from .agent_load_balancer import AgentLoadBalancer, AgentLoadState, LoadBalancingStrategy
from .capacity_manager import CapacityManager, ScalingDecision, ScalingAction, CapacityTier
from .performance_metrics_collector import PerformanceMetricsCollector
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.agent_performance import WorkloadSnapshot, AgentPerformanceHistory
from ..models.task import Task, TaskStatus, TaskPriority

logger = structlog.get_logger()


class ScalingTrigger(Enum):
    """Types of scaling triggers."""
    WORKLOAD_BASED = "workload_based"
    PERFORMANCE_BASED = "performance_based"
    PREDICTIVE = "predictive"
    COST_OPTIMIZATION = "cost_optimization"
    MANUAL = "manual"
    EMERGENCY = "emergency"


class ScalingPattern(Enum):
    """Identified scaling patterns."""
    STEADY_STATE = "steady_state"
    GRADUAL_INCREASE = "gradual_increase"
    GRADUAL_DECREASE = "gradual_decrease"
    SPIKE = "spike"
    OSCILLATING = "oscillating"
    UNPREDICTABLE = "unpredictable"


@dataclass
class ScalingRule:
    """Rule for automatic scaling decisions."""
    name: str
    trigger: ScalingTrigger
    condition: str  # Python expression to evaluate
    action: ScalingAction
    priority: int  # 1=highest, 10=lowest
    cooldown_seconds: int = 300
    max_scale_factor: float = 2.0
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
    def is_ready_to_trigger(self) -> bool:
        """Check if rule is ready to trigger (cooldown period passed)."""
        if not self.enabled:
            return False
        
        if self.last_triggered is None:
            return True
        
        return (datetime.utcnow() - self.last_triggered).seconds >= self.cooldown_seconds
    
    def evaluate_condition(self, context: Dict[str, Any]) -> bool:
        """Evaluate rule condition against current context."""
        try:
            # Create safe evaluation environment
            safe_context = {
                'avg_load': context.get('avg_load', 0),
                'max_load': context.get('max_load', 0),
                'min_load': context.get('min_load', 0),
                'total_agents': context.get('total_agents', 0),
                'active_agents': context.get('active_agents', 0),
                'overloaded_agents': context.get('overloaded_agents', 0),
                'underloaded_agents': context.get('underloaded_agents', 0),
                'avg_response_time': context.get('avg_response_time', 0),
                'error_rate': context.get('error_rate', 0),
                'pending_tasks': context.get('pending_tasks', 0),
                'predicted_load': context.get('predicted_load', 0),
                'cost_per_hour': context.get('cost_per_hour', 0),
                # Mathematical functions
                'min': min,
                'max': max,
                'abs': abs,
                'round': round
            }
            
            return bool(eval(self.condition, {"__builtins__": {}}, safe_context))
        
        except Exception as e:
            logger.error("Error evaluating scaling rule condition",
                        rule_name=self.name,
                        condition=self.condition,
                        error=str(e))
            return False


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    timestamp: datetime
    trigger: ScalingTrigger
    action: ScalingAction
    context: Dict[str, Any]
    decision: ScalingDecision
    execution_result: Optional[Dict[str, Any]] = None
    success: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "trigger": self.trigger.value,
            "action": self.action.value,
            "context": self.context,
            "success": self.success,
            "execution_result": self.execution_result
        }


class AdaptiveScaler:
    """
    Intelligent adaptive scaler for agent orchestration system.
    
    Features:
    - Rule-based auto-scaling with customizable conditions
    - Pattern recognition for proactive scaling
    - Predictive scaling based on historical trends
    - Cost-aware scaling decisions
    - Emergency scaling for system protection
    - Learning from scaling outcomes
    """
    
    def __init__(
        self,
        load_balancer: AgentLoadBalancer,
        capacity_manager: CapacityManager,
        metrics_collector: PerformanceMetricsCollector,
        redis_client=None,
        session_factory: Optional[Callable] = None
    ):
        self.load_balancer = load_balancer
        self.capacity_manager = capacity_manager
        self.metrics_collector = metrics_collector
        self.redis_client = redis_client
        self.session_factory = session_factory or get_session
        
        # Scaling state
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.scaling_history: deque = deque(maxlen=200)
        self.pattern_detection_window = deque(maxlen=288)  # 24 hours at 5-minute intervals
        
        # Auto-scaling control
        self.auto_scaling_enabled = True
        self.scaling_task: Optional[asyncio.Task] = None
        self.evaluation_interval = 60  # seconds
        
        # Performance tracking
        self.scaling_effectiveness: Dict[ScalingAction, deque] = {
            action: deque(maxlen=50) for action in ScalingAction
        }
        
        # Configuration
        self.config = {
            "min_agents": 2,
            "max_agents": 20,
            "default_evaluation_interval": 60,
            "pattern_detection_samples": 20,
            "prediction_accuracy_threshold": 0.7,
            "emergency_load_threshold": 0.95,
            "cost_optimization_enabled": True,
            "learning_enabled": True,
            "max_scaling_velocity": 0.5,  # Max 50% change per scaling event
        }
        
        # Initialize default scaling rules
        self._initialize_default_rules()
        
        logger.info("AdaptiveScaler initialized",
                   config=self.config,
                   rules_count=len(self.scaling_rules))
    
    def _initialize_default_rules(self) -> None:
        """Initialize default scaling rules."""
        
        # High load scale-up rule
        self.add_scaling_rule(ScalingRule(
            name="high_load_scale_up",
            trigger=ScalingTrigger.WORKLOAD_BASED,
            condition="avg_load > 0.75 and overloaded_agents > total_agents * 0.3",
            action=ScalingAction.SCALE_UP,
            priority=2,
            cooldown_seconds=180
        ))
        
        # Low load scale-down rule
        self.add_scaling_rule(ScalingRule(
            name="low_load_scale_down",
            trigger=ScalingTrigger.WORKLOAD_BASED,
            condition="avg_load < 0.3 and underloaded_agents > total_agents * 0.6 and total_agents > min(3, self.config['min_agents'] + 1)",
            action=ScalingAction.SCALE_DOWN,
            priority=5,
            cooldown_seconds=300
        ))
        
        # Performance-based scale-up
        self.add_scaling_rule(ScalingRule(
            name="performance_scale_up",
            trigger=ScalingTrigger.PERFORMANCE_BASED,
            condition="avg_response_time > 5000 or error_rate > 5.0",
            action=ScalingAction.SCALE_UP,
            priority=3,
            cooldown_seconds=240
        ))
        
        # Emergency scale-up
        self.add_scaling_rule(ScalingRule(
            name="emergency_scale_up",
            trigger=ScalingTrigger.EMERGENCY,
            condition="max_load > 0.95 or error_rate > 15.0",
            action=ScalingAction.SCALE_UP,
            priority=1,
            cooldown_seconds=60,
            max_scale_factor=3.0
        ))
        
        # Predictive scale-up
        self.add_scaling_rule(ScalingRule(
            name="predictive_scale_up",
            trigger=ScalingTrigger.PREDICTIVE,
            condition="predicted_load > 0.8 and total_agents < self.config['max_agents']",
            action=ScalingAction.SCALE_UP,
            priority=4,
            cooldown_seconds=600
        ))
        
        # Load balancing rule
        self.add_scaling_rule(ScalingRule(
            name="load_rebalance",
            trigger=ScalingTrigger.PERFORMANCE_BASED,
            condition="overloaded_agents > 0 and underloaded_agents > 0",
            action=ScalingAction.REBALANCE,
            priority=6,
            cooldown_seconds=120
        ))
    
    def add_scaling_rule(self, rule: ScalingRule) -> None:
        """Add a new scaling rule."""
        self.scaling_rules[rule.name] = rule
        logger.info("Scaling rule added",
                   rule_name=rule.name,
                   trigger=rule.trigger.value,
                   action=rule.action.value)
    
    def remove_scaling_rule(self, rule_name: str) -> bool:
        """Remove a scaling rule."""
        if rule_name in self.scaling_rules:
            del self.scaling_rules[rule_name]
            logger.info("Scaling rule removed", rule_name=rule_name)
            return True
        return False
    
    def enable_rule(self, rule_name: str) -> bool:
        """Enable a scaling rule."""
        if rule_name in self.scaling_rules:
            self.scaling_rules[rule_name].enabled = True
            return True
        return False
    
    def disable_rule(self, rule_name: str) -> bool:
        """Disable a scaling rule."""
        if rule_name in self.scaling_rules:
            self.scaling_rules[rule_name].enabled = False
            return True
        return False
    
    async def start_auto_scaling(self) -> None:
        """Start automatic scaling evaluation loop."""
        if self.scaling_task and not self.scaling_task.done():
            logger.warning("Auto-scaling already running")
            return
        
        self.auto_scaling_enabled = True
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        
        logger.info("Auto-scaling started", interval=self.evaluation_interval)
    
    async def stop_auto_scaling(self) -> None:
        """Stop automatic scaling."""
        self.auto_scaling_enabled = False
        
        if self.scaling_task:
            self.scaling_task.cancel()
            try:
                await self.scaling_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Auto-scaling stopped")
    
    async def _scaling_loop(self) -> None:
        """Main auto-scaling evaluation loop."""
        while self.auto_scaling_enabled:
            try:
                # Evaluate scaling needs
                scaling_decisions = await self.evaluate_scaling_needs()
                
                # Execute scaling decisions
                for decision in scaling_decisions:
                    if decision.action != ScalingAction.MAINTAIN:
                        await self._execute_scaling_decision(decision)
                
                # Update pattern detection
                await self._update_pattern_detection()
                
                # Learn from recent scaling events
                if self.config["learning_enabled"]:
                    await self._learn_from_scaling_history()
                
                await asyncio.sleep(self.evaluation_interval)
                
            except Exception as e:
                logger.error("Error in scaling loop", error=str(e))
                await asyncio.sleep(self.evaluation_interval)
    
    async def evaluate_scaling_needs(self) -> List[ScalingDecision]:
        """Evaluate current scaling needs based on rules and patterns."""
        try:
            # Get current system context
            context = await self._build_scaling_context()
            
            # Evaluate all scaling rules
            triggered_rules = []
            for rule in self.scaling_rules.values():
                if rule.is_ready_to_trigger() and rule.evaluate_condition(context):
                    triggered_rules.append(rule)
            
            if not triggered_rules:
                return [ScalingDecision(
                    action=ScalingAction.MAINTAIN,
                    target_agents=[],
                    resource_changes={},
                    reasoning="No scaling rules triggered",
                    confidence=1.0,
                    estimated_impact={}
                )]
            
            # Sort by priority and select best rule
            triggered_rules.sort(key=lambda r: (r.priority, -r.trigger_count))
            selected_rule = triggered_rules[0]
            
            # Generate scaling decision based on rule
            decision = await self._generate_scaling_decision(selected_rule, context)
            
            # Update rule state
            selected_rule.last_triggered = datetime.utcnow()
            selected_rule.trigger_count += 1
            
            logger.info("Scaling rule triggered",
                       rule_name=selected_rule.name,
                       action=decision.action.value,
                       confidence=decision.confidence)
            
            return [decision]
            
        except Exception as e:
            logger.error("Error evaluating scaling needs", error=str(e))
            return []
    
    async def _build_scaling_context(self) -> Dict[str, Any]:
        """Build context for scaling rule evaluation."""
        try:
            # Get load balancer metrics
            load_metrics = await self.load_balancer.get_load_balancing_metrics()
            
            # Get capacity metrics
            capacity_metrics = await self.capacity_manager.get_capacity_metrics()
            
            # Get performance metrics
            performance_summary = await self.metrics_collector.get_performance_summary()
            
            # Get system metrics from recent workload snapshots
            async with self.session_factory() as session:
                query = select(WorkloadSnapshot).where(
                    WorkloadSnapshot.snapshot_time >= datetime.utcnow() - timedelta(minutes=5)
                )
                result = await session.execute(query)
                recent_snapshots = result.scalars().all()
            
            # Calculate aggregated metrics
            if recent_snapshots:
                load_factors = [s.calculate_load_factor() for s in recent_snapshots]
                response_times = [s.average_response_time_ms or 0 for s in recent_snapshots if s.average_response_time_ms]
                error_rates = [s.error_rate_percent for s in recent_snapshots]
                
                avg_load = statistics.mean(load_factors)
                max_load = max(load_factors)
                min_load = min(load_factors)
                avg_response_time = statistics.mean(response_times) if response_times else 0
                error_rate = statistics.mean(error_rates) if error_rates else 0
            else:
                avg_load = max_load = min_load = avg_response_time = error_rate = 0
            
            # Get predictions from capacity manager
            capacity_predictions = capacity_metrics.get("predictions", {})
            predicted_load = 0
            if capacity_predictions:
                # Use shortest term prediction (15 minutes)
                shortest_prediction = min(capacity_predictions.keys()) if capacity_predictions else None
                if shortest_prediction:
                    predicted_load = capacity_predictions[shortest_prediction].get("predicted_load", 0)
            
            # Count agent states
            agent_summary = load_metrics.get("agent_load_summary", {})
            overloaded_agents = sum(1 for data in agent_summary.values() if not data.get("can_handle_more", True))
            underloaded_agents = sum(1 for data in agent_summary.values() if data.get("load_factor", 1) < 0.3)
            active_agents = len([data for data in agent_summary.values() if data.get("active_tasks", 0) > 0])
            
            # Estimate current costs (simplified)
            total_agents = capacity_metrics.get("total_agents", 0)
            cost_per_hour = total_agents * 10  # $10 per agent per hour (example)
            
            context = {
                "avg_load": avg_load,
                "max_load": max_load,
                "min_load": min_load,
                "total_agents": total_agents,
                "active_agents": active_agents,
                "overloaded_agents": overloaded_agents,
                "underloaded_agents": underloaded_agents,
                "avg_response_time": avg_response_time,
                "error_rate": error_rate,
                "pending_tasks": sum(data.get("pending_tasks", 0) for data in agent_summary.values()),
                "predicted_load": predicted_load,
                "cost_per_hour": cost_per_hour,
                "timestamp": datetime.utcnow()
            }
            
            return context
            
        except Exception as e:
            logger.error("Error building scaling context", error=str(e))
            return {}
    
    async def _generate_scaling_decision(
        self,
        rule: ScalingRule,
        context: Dict[str, Any]
    ) -> ScalingDecision:
        """Generate scaling decision based on triggered rule."""
        
        current_agents = context.get("total_agents", 0)
        
        if rule.action == ScalingAction.SCALE_UP:
            # Calculate scale-up amount
            if rule.trigger == ScalingTrigger.EMERGENCY:
                # Emergency scaling - be more aggressive
                scale_factor = min(rule.max_scale_factor, 2.0)
                target_agents = min(
                    self.config["max_agents"],
                    int(current_agents * scale_factor)
                )
            else:
                # Normal scale-up - conservative approach
                overloaded_count = context.get("overloaded_agents", 0)
                target_agents = min(
                    self.config["max_agents"],
                    current_agents + max(1, overloaded_count)
                )
            
            new_agents = [f"scaled_agent_{i}" for i in range(current_agents, target_agents)]
            
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                target_agents=new_agents,
                resource_changes={},
                reasoning=f"Scale up triggered by rule '{rule.name}': {len(new_agents)} agents added",
                confidence=0.8 if rule.trigger != ScalingTrigger.EMERGENCY else 0.95,
                estimated_impact={
                    "load_reduction": context.get("avg_load", 0) * 0.3,
                    "agents_added": len(new_agents),
                    "cost_increase": len(new_agents) * 10
                }
            )
        
        elif rule.action == ScalingAction.SCALE_DOWN:
            # Calculate scale-down amount
            underloaded_count = context.get("underloaded_agents", 0)
            agents_to_remove = min(
                current_agents - self.config["min_agents"],
                max(1, underloaded_count // 2)
            )
            
            if agents_to_remove <= 0:
                return ScalingDecision(
                    action=ScalingAction.MAINTAIN,
                    target_agents=[],
                    resource_changes={},
                    reasoning="Scale down not possible - at minimum capacity",
                    confidence=1.0,
                    estimated_impact={}
                )
            
            # Select agents to remove (would need integration with actual agent management)
            agents_to_remove_ids = [f"agent_to_remove_{i}" for i in range(agents_to_remove)]
            
            return ScalingDecision(
                action=ScalingAction.SCALE_DOWN,
                target_agents=agents_to_remove_ids,
                resource_changes={},
                reasoning=f"Scale down triggered by rule '{rule.name}': {agents_to_remove} agents removed",
                confidence=0.7,
                estimated_impact={
                    "load_increase": context.get("avg_load", 0) * 0.2,
                    "agents_removed": agents_to_remove,
                    "cost_savings": agents_to_remove * 10
                }
            )
        
        elif rule.action == ScalingAction.REBALANCE:
            return ScalingDecision(
                action=ScalingAction.REBALANCE,
                target_agents=[],
                resource_changes={},
                reasoning=f"Load rebalancing triggered by rule '{rule.name}'",
                confidence=0.75,
                estimated_impact={
                    "load_distribution_improvement": 0.2,
                    "performance_improvement": 0.1
                }
            )
        
        else:
            return ScalingDecision(
                action=ScalingAction.MAINTAIN,
                target_agents=[],
                resource_changes={},
                reasoning=f"Rule '{rule.name}' triggered but no action needed",
                confidence=0.5,
                estimated_impact={}
            )
    
    async def _execute_scaling_decision(self, decision: ScalingDecision) -> None:
        """Execute a scaling decision."""
        try:
            scaling_event = ScalingEvent(
                timestamp=datetime.utcnow(),
                trigger=ScalingTrigger.WORKLOAD_BASED,  # TODO: Pass actual trigger
                action=decision.action,
                context=await self._build_scaling_context(),
                decision=decision
            )
            
            # Execute through capacity manager
            execution_result = await self.capacity_manager.execute_scaling_decision(decision)
            
            scaling_event.execution_result = execution_result
            scaling_event.success = execution_result.get("success", False)
            
            # Record scaling event
            self.scaling_history.append(scaling_event)
            
            # Track effectiveness
            if decision.action in self.scaling_effectiveness:
                self.scaling_effectiveness[decision.action].append({
                    "timestamp": datetime.utcnow(),
                    "success": scaling_event.success,
                    "impact": decision.estimated_impact
                })
            
            # Store event to Redis for monitoring
            if self.redis_client:
                try:
                    event_key = "adaptive_scaler:events"
                    await self.redis_client.lpush(event_key, json.dumps(scaling_event.to_dict()))
                    await self.redis_client.ltrim(event_key, 0, 99)  # Keep last 100 events
                    await self.redis_client.expire(event_key, 86400)  # 24 hours TTL
                except Exception as e:
                    logger.error("Error storing scaling event to Redis", error=str(e))
            
            logger.info("Scaling decision executed",
                       action=decision.action.value,
                       success=scaling_event.success,
                       reasoning=decision.reasoning)
        
        except Exception as e:
            logger.error("Error executing scaling decision", error=str(e))
    
    async def _update_pattern_detection(self) -> None:
        """Update workload pattern detection."""
        try:
            context = await self._build_scaling_context()
            
            # Add current load to pattern detection window
            self.pattern_detection_window.append({
                "timestamp": datetime.utcnow(),
                "load_factor": context.get("avg_load", 0),
                "agent_count": context.get("total_agents", 0),
                "response_time": context.get("avg_response_time", 0)
            })
            
            # Detect patterns if we have sufficient data
            if len(self.pattern_detection_window) >= self.config["pattern_detection_samples"]:
                pattern = self._detect_workload_pattern()
                
                # Adjust evaluation interval based on pattern
                if pattern == ScalingPattern.SPIKE:
                    self.evaluation_interval = 30  # More frequent evaluation during spikes
                elif pattern == ScalingPattern.STEADY_STATE:
                    self.evaluation_interval = 120  # Less frequent during steady state
                else:
                    self.evaluation_interval = self.config["default_evaluation_interval"]
        
        except Exception as e:
            logger.error("Error updating pattern detection", error=str(e))
    
    def _detect_workload_pattern(self) -> ScalingPattern:
        """Detect current workload pattern from recent data."""
        if len(self.pattern_detection_window) < self.config["pattern_detection_samples"]:
            return ScalingPattern.STEADY_STATE
        
        recent_data = list(self.pattern_detection_window)[-self.config["pattern_detection_samples"]:]
        load_factors = [d["load_factor"] for d in recent_data]
        
        # Calculate trend and volatility
        mean_load = statistics.mean(load_factors)
        std_load = statistics.stdev(load_factors) if len(load_factors) > 1 else 0
        
        # Linear trend
        n = len(load_factors)
        x_mean = (n - 1) / 2
        xy_sum = sum(i * load_factors[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        trend_slope = (xy_sum - n * x_mean * mean_load) / (x2_sum - n * x_mean * x_mean) if x2_sum != n * x_mean * x_mean else 0
        
        # Pattern classification
        if std_load > 0.3:  # High volatility
            if abs(trend_slope) > 0.01:
                return ScalingPattern.SPIKE
            else:
                return ScalingPattern.OSCILLATING
        elif abs(trend_slope) > 0.005:  # Significant trend
            if trend_slope > 0:
                return ScalingPattern.GRADUAL_INCREASE
            else:
                return ScalingPattern.GRADUAL_DECREASE
        elif std_load < 0.1:  # Low volatility and trend
            return ScalingPattern.STEADY_STATE
        else:
            return ScalingPattern.UNPREDICTABLE
    
    async def _learn_from_scaling_history(self) -> None:
        """Learn from recent scaling events to improve future decisions."""
        try:
            # Analyze recent scaling events (last 10)
            recent_events = list(self.scaling_history)[-10:] if len(self.scaling_history) >= 10 else list(self.scaling_history)
            
            if len(recent_events) < 3:
                return  # Need more data for learning
            
            # Calculate success rates by action type
            action_success_rates = defaultdict(list)
            for event in recent_events:
                action_success_rates[event.action].append(1.0 if event.success else 0.0)
            
            # Adjust rule priorities based on success rates
            for rule_name, rule in self.scaling_rules.items():
                if rule.action in action_success_rates:
                    success_rate = statistics.mean(action_success_rates[rule.action])
                    
                    # Increase priority for successful actions, decrease for unsuccessful
                    if success_rate > 0.8 and rule.priority > 1:
                        rule.priority = max(1, rule.priority - 1)
                    elif success_rate < 0.5 and rule.priority < 10:
                        rule.priority = min(10, rule.priority + 1)
            
            # Adjust cooldown periods based on effectiveness
            for action, events in self.scaling_effectiveness.items():
                if len(events) >= 5:
                    recent_events_for_action = list(events)[-5:]
                    avg_success = statistics.mean([1.0 if e["success"] else 0.0 for e in recent_events_for_action])
                    
                    # Find rules with this action and adjust cooldown
                    for rule in self.scaling_rules.values():
                        if rule.action == action:
                            if avg_success > 0.8:
                                # Successful actions can have shorter cooldowns
                                rule.cooldown_seconds = max(60, int(rule.cooldown_seconds * 0.9))
                            elif avg_success < 0.5:
                                # Unsuccessful actions need longer cooldowns
                                rule.cooldown_seconds = min(600, int(rule.cooldown_seconds * 1.1))
        
        except Exception as e:
            logger.error("Error learning from scaling history", error=str(e))
    
    async def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scaling metrics and statistics."""
        try:
            # Rule statistics
            rule_stats = {}
            for rule_name, rule in self.scaling_rules.items():
                rule_stats[rule_name] = {
                    "enabled": rule.enabled,
                    "trigger_count": rule.trigger_count,
                    "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None,
                    "priority": rule.priority,
                    "cooldown_seconds": rule.cooldown_seconds,
                    "ready_to_trigger": rule.is_ready_to_trigger()
                }
            
            # Scaling history summary
            recent_events = list(self.scaling_history)[-24:] if len(self.scaling_history) >= 24 else list(self.scaling_history)
            
            action_counts = defaultdict(int)
            success_counts = defaultdict(int)
            for event in recent_events:
                action_counts[event.action.value] += 1
                if event.success:
                    success_counts[event.action.value] += 1
            
            # Pattern detection
            current_pattern = self._detect_workload_pattern() if len(self.pattern_detection_window) > 10 else ScalingPattern.STEADY_STATE
            
            # Effectiveness metrics
            effectiveness_summary = {}
            for action, events in self.scaling_effectiveness.items():
                if events:
                    recent_effectiveness = list(events)[-10:]
                    success_rate = statistics.mean([1.0 if e["success"] else 0.0 for e in recent_effectiveness])
                    effectiveness_summary[action.value] = {
                        "success_rate": success_rate,
                        "total_events": len(events),
                        "recent_events": len(recent_effectiveness)
                    }
            
            return {
                "auto_scaling_enabled": self.auto_scaling_enabled,
                "evaluation_interval": self.evaluation_interval,
                "rules": rule_stats,
                "scaling_history": {
                    "total_events": len(self.scaling_history),
                    "recent_events": len(recent_events),
                    "action_counts": dict(action_counts),
                    "success_counts": dict(success_counts)
                },
                "pattern_detection": {
                    "current_pattern": current_pattern.value,
                    "data_points": len(self.pattern_detection_window),
                    "evaluation_interval": self.evaluation_interval
                },
                "effectiveness": effectiveness_summary,
                "configuration": self.config
            }
        
        except Exception as e:
            logger.error("Error getting scaling metrics", error=str(e))
            return {"error": str(e)}
    
    async def force_scaling_evaluation(self) -> List[ScalingDecision]:
        """Force immediate scaling evaluation (for testing/manual intervention)."""
        logger.info("Forcing scaling evaluation")
        return await self.evaluate_scaling_needs()
    
    def set_evaluation_interval(self, interval_seconds: int) -> None:
        """Set scaling evaluation interval."""
        self.evaluation_interval = max(30, min(600, interval_seconds))  # Between 30s and 10min
        logger.info("Scaling evaluation interval updated", interval=self.evaluation_interval)
    
    async def get_scaling_recommendations(self) -> Dict[str, Any]:
        """Get scaling recommendations without executing them."""
        try:
            context = await self._build_scaling_context()
            scaling_decisions = await self.evaluate_scaling_needs()
            
            return {
                "current_context": context,
                "recommendations": [
                    {
                        "action": decision.action.value,
                        "reasoning": decision.reasoning,
                        "confidence": decision.confidence,
                        "estimated_impact": decision.estimated_impact,
                        "target_agents_count": len(decision.target_agents)
                    }
                    for decision in scaling_decisions
                ],
                "pattern": self._detect_workload_pattern().value,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error("Error getting scaling recommendations", error=str(e))
            return {"error": str(e)}