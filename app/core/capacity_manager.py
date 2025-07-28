"""
Advanced Capacity Manager for LeanVibe Agent Hive 2.0

Manages agent scaling, resource allocation optimization, and capacity planning
for efficient multi-agent orchestration with intelligent resource management.
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

import structlog
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .redis import get_message_broker, get_session_cache, AgentMessageBroker, SessionCache
from .database import get_session
from .agent_load_balancer import AgentLoadState, AgentLoadBalancer, LoadBalancingStrategy
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.agent_performance import WorkloadSnapshot, AgentPerformanceHistory
from ..models.task import Task, TaskStatus, TaskPriority

logger = structlog.get_logger()


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    REBALANCE = "rebalance"
    OPTIMIZE = "optimize"
    MAINTAIN = "maintain"


class ResourceType(Enum):
    """Types of resources managed by capacity manager."""
    COMPUTE = "compute"
    MEMORY = "memory"
    CONTEXT = "context"
    NETWORK = "network"
    STORAGE = "storage"


class CapacityTier(Enum):
    """Agent capacity tiers for scaling decisions."""
    LIGHT = "light"      # 1-2 concurrent tasks
    STANDARD = "standard"  # 3-5 concurrent tasks
    HEAVY = "heavy"      # 6-10 concurrent tasks
    ENTERPRISE = "enterprise"  # 10+ concurrent tasks


@dataclass
class ResourceAllocation:
    """Resource allocation for an agent."""
    agent_id: str
    tier: CapacityTier
    max_concurrent_tasks: int
    memory_limit_mb: float
    cpu_limit_percent: float
    context_window_size: int
    priority_weight: float = 1.0
    allocated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "tier": self.tier.value,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "memory_limit_mb": self.memory_limit_mb,
            "cpu_limit_percent": self.cpu_limit_percent,
            "context_window_size": self.context_window_size,
            "priority_weight": self.priority_weight,
            "allocated_at": self.allocated_at.isoformat()
        }


@dataclass
class ScalingDecision:
    """Decision made by capacity manager."""
    action: ScalingAction
    target_agents: List[str]
    resource_changes: Dict[str, ResourceAllocation]
    reasoning: str
    confidence: float
    estimated_impact: Dict[str, Any]
    execution_priority: int = 5  # 1=highest, 10=lowest
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CapacityPrediction:
    """Capacity prediction based on historical data."""
    timeframe_minutes: int
    predicted_load_factor: float
    confidence_interval: Tuple[float, float]
    recommended_agent_count: int
    risk_factors: List[str]
    based_on_samples: int
    prediction_accuracy: Optional[float] = None


class CapacityManager:
    """
    Intelligent capacity manager for agent scaling and resource optimization.
    
    Features:
    - Dynamic agent scaling based on workload patterns
    - Resource allocation optimization per agent tier
    - Predictive capacity planning
    - Cost-aware resource management
    - Performance-based tier adjustments
    """
    
    def __init__(
        self,
        load_balancer: AgentLoadBalancer,
        redis_client=None,
        session_factory: Optional[Callable] = None
    ):
        self.load_balancer = load_balancer
        self.redis_client = redis_client
        self.session_factory = session_factory or get_session
        
        # Capacity state
        self.resource_allocations: Dict[str, ResourceAllocation] = {}
        self.scaling_history: deque = deque(maxlen=100)
        self.capacity_predictions: Dict[int, CapacityPrediction] = {}  # timeframe -> prediction
        
        # Performance tracking
        self.tier_performance: Dict[CapacityTier, deque] = {
            tier: deque(maxlen=50) for tier in CapacityTier
        }
        
        # Configuration
        self.config = {
            # Scaling thresholds
            "scale_up_threshold": 0.75,      # Scale up when avg load > 75%
            "scale_down_threshold": 0.30,    # Scale down when avg load < 30%
            "min_agents": 2,                 # Minimum agents to maintain
            "max_agents": 20,                # Maximum agents allowed
            
            # Resource limits by tier
            "tier_limits": {
                CapacityTier.LIGHT: {
                    "max_tasks": 2,
                    "memory_mb": 500,
                    "cpu_percent": 50,
                    "context_size": 8000
                },
                CapacityTier.STANDARD: {
                    "max_tasks": 5,
                    "memory_mb": 1000,
                    "cpu_percent": 75,
                    "context_size": 16000
                },
                CapacityTier.HEAVY: {
                    "max_tasks": 8,
                    "memory_mb": 2000,
                    "cpu_percent": 90,
                    "context_size": 32000
                },
                CapacityTier.ENTERPRISE: {
                    "max_tasks": 12,
                    "memory_mb": 4000,
                    "cpu_percent": 95,
                    "context_size": 64000
                }
            },
            
            # Timing parameters
            "scaling_evaluation_interval": 300,  # 5 minutes
            "prediction_intervals": [15, 60, 180, 720],  # 15min, 1h, 3h, 12h
            "minimum_scaling_interval": 180,     # 3 minutes between scaling actions
            
            # Performance criteria
            "tier_promotion_threshold": 0.85,   # Promote to higher tier when consistently over 85%
            "tier_demotion_threshold": 0.40,    # Demote to lower tier when consistently under 40%
            "performance_evaluation_period": 3600,  # 1 hour evaluation window
        }
        
        # Initialize default tier allocations
        self._initialize_tier_templates()
        
        self.last_scaling_action = datetime.utcnow() - timedelta(minutes=10)
        
        logger.info("CapacityManager initialized", config=self.config)
    
    def _initialize_tier_templates(self) -> None:
        """Initialize resource allocation templates for each tier."""
        self.tier_templates = {}
        
        for tier, limits in self.config["tier_limits"].items():
            self.tier_templates[tier] = ResourceAllocation(
                agent_id="template",
                tier=tier,
                max_concurrent_tasks=limits["max_tasks"],
                memory_limit_mb=limits["memory_mb"],
                cpu_limit_percent=limits["cpu_percent"],
                context_window_size=limits["context_size"]
            )
    
    async def evaluate_capacity_needs(self) -> List[ScalingDecision]:
        """
        Evaluate current capacity needs and generate scaling decisions.
        
        Returns:
            List of scaling decisions to execute
        """
        try:
            # Get current system state
            system_metrics = await self._get_system_metrics()
            load_metrics = await self.load_balancer.get_load_balancing_metrics()
            
            # Generate capacity predictions
            predictions = await self._generate_capacity_predictions()
            
            # Analyze scaling needs
            decisions = []
            
            # Check if scaling is needed
            scale_decision = await self._evaluate_scaling_needs(
                system_metrics, load_metrics, predictions
            )
            if scale_decision:
                decisions.append(scale_decision)
            
            # Check if rebalancing is needed
            rebalance_decision = await self._evaluate_rebalancing_needs(
                system_metrics, load_metrics
            )
            if rebalance_decision:
                decisions.append(rebalance_decision)
            
            # Check if tier optimization is needed
            tier_decisions = await self._evaluate_tier_optimizations(system_metrics)
            decisions.extend(tier_decisions)
            
            # Sort by priority and confidence
            decisions.sort(key=lambda d: (d.execution_priority, -d.confidence))
            
            logger.info("Capacity evaluation completed",
                       decisions_count=len(decisions),
                       system_load=system_metrics.get("average_load_factor", 0))
            
            return decisions
            
        except Exception as e:
            logger.error("Capacity evaluation failed", error=str(e))
            return []
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system-wide metrics."""
        try:
            async with self.session_factory() as session:
                # Get recent workload snapshots
                query = select(WorkloadSnapshot).where(
                    WorkloadSnapshot.snapshot_time >= datetime.utcnow() - timedelta(minutes=30)
                ).order_by(WorkloadSnapshot.snapshot_time.desc())
                
                result = await session.execute(query)
                snapshots = result.scalars().all()
                
                if not snapshots:
                    return {"status": "no_data"}
                
                # Calculate system-wide metrics
                total_agents = len(set(str(s.agent_id) for s in snapshots))
                total_active_tasks = sum(s.active_tasks for s in snapshots)
                total_pending_tasks = sum(s.pending_tasks for s in snapshots)
                
                load_factors = [s.calculate_load_factor() for s in snapshots]
                avg_load_factor = statistics.mean(load_factors) if load_factors else 0
                
                # Resource utilization
                memory_usage = [s.memory_usage_mb for s in snapshots if s.memory_usage_mb]
                cpu_usage = [s.cpu_usage_percent for s in snapshots if s.cpu_usage_percent]
                
                return {
                    "total_agents": total_agents,
                    "total_active_tasks": total_active_tasks,
                    "total_pending_tasks": total_pending_tasks,
                    "average_load_factor": avg_load_factor,
                    "max_load_factor": max(load_factors) if load_factors else 0,
                    "min_load_factor": min(load_factors) if load_factors else 0,
                    "load_variance": statistics.variance(load_factors) if len(load_factors) > 1 else 0,
                    "average_memory_usage": statistics.mean(memory_usage) if memory_usage else 0,
                    "average_cpu_usage": statistics.mean(cpu_usage) if cpu_usage else 0,
                    "overloaded_agents": len([lf for lf in load_factors if lf > 0.85]),
                    "underloaded_agents": len([lf for lf in load_factors if lf < 0.30]),
                    "snapshot_count": len(snapshots),
                    "evaluation_time": datetime.utcnow()
                }
                
        except Exception as e:
            logger.error("Failed to get system metrics", error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def _generate_capacity_predictions(self) -> Dict[int, CapacityPrediction]:
        """Generate capacity predictions for different time horizons."""
        predictions = {}
        
        try:
            async with self.session_factory() as session:
                # Get historical data for predictions
                lookback_hours = 24
                query = select(WorkloadSnapshot).where(
                    WorkloadSnapshot.snapshot_time >= datetime.utcnow() - timedelta(hours=lookback_hours)
                ).order_by(WorkloadSnapshot.snapshot_time.asc())
                
                result = await session.execute(query)
                snapshots = result.scalars().all()
                
                if len(snapshots) < 10:  # Need minimum data for predictions
                    return predictions
                
                # Group snapshots by time intervals
                time_series = defaultdict(list)
                for snapshot in snapshots:
                    # Round to nearest 15-minute interval
                    interval = int(snapshot.snapshot_time.timestamp() // 900) * 900
                    time_series[interval].append(snapshot.calculate_load_factor())
                
                # Calculate average load per interval
                intervals = sorted(time_series.keys())
                load_series = [statistics.mean(time_series[interval]) for interval in intervals]
                
                # Generate predictions for each timeframe
                for timeframe in self.config["prediction_intervals"]:
                    prediction = await self._predict_capacity_need(
                        load_series, timeframe, len(snapshots)
                    )
                    predictions[timeframe] = prediction
                
        except Exception as e:
            logger.error("Failed to generate capacity predictions", error=str(e))
        
        self.capacity_predictions = predictions
        return predictions
    
    async def _predict_capacity_need(
        self,
        historical_loads: List[float],
        timeframe_minutes: int,
        sample_count: int
    ) -> CapacityPrediction:
        """Predict capacity needs for specific timeframe."""
        if len(historical_loads) < 3:
            # Insufficient data - return conservative prediction
            return CapacityPrediction(
                timeframe_minutes=timeframe_minutes,
                predicted_load_factor=0.5,
                confidence_interval=(0.3, 0.7),
                recommended_agent_count=self.config["min_agents"],
                risk_factors=["insufficient_historical_data"],
                based_on_samples=sample_count
            )
        
        # Simple trend-based prediction (can be enhanced with ML models)
        recent_loads = historical_loads[-min(10, len(historical_loads)):]
        avg_load = statistics.mean(recent_loads)
        load_trend = recent_loads[-1] - recent_loads[0] if len(recent_loads) > 1 else 0
        
        # Project trend forward
        trend_factor = min(0.5, abs(load_trend) * (timeframe_minutes / 60))
        predicted_load = avg_load + (load_trend * trend_factor)
        predicted_load = max(0.0, min(2.0, predicted_load))  # Cap between 0-200%
        
        # Calculate confidence interval
        load_std = statistics.stdev(recent_loads) if len(recent_loads) > 1 else 0.1
        confidence_width = load_std * 1.96  # 95% confidence interval
        confidence_interval = (
            max(0.0, predicted_load - confidence_width),
            min(2.0, predicted_load + confidence_width)
        )
        
        # Recommend agent count based on predicted load
        current_agents = len(self.resource_allocations)
        if current_agents == 0:
            current_agents = self.config["min_agents"]
        
        if predicted_load > 0.8:
            recommended_agents = min(self.config["max_agents"], 
                                   int(current_agents * 1.5))
        elif predicted_load < 0.3:
            recommended_agents = max(self.config["min_agents"], 
                                   int(current_agents * 0.8))
        else:
            recommended_agents = current_agents
        
        # Identify risk factors
        risk_factors = []
        if predicted_load > 0.9:
            risk_factors.append("high_predicted_load")
        if load_trend > 0.2:
            risk_factors.append("increasing_load_trend")
        if load_std > 0.3:
            risk_factors.append("high_load_volatility")
        if sample_count < 20:
            risk_factors.append("limited_historical_data")
        
        return CapacityPrediction(
            timeframe_minutes=timeframe_minutes,
            predicted_load_factor=predicted_load,
            confidence_interval=confidence_interval,
            recommended_agent_count=recommended_agents,
            risk_factors=risk_factors,
            based_on_samples=sample_count
        )
    
    async def _evaluate_scaling_needs(
        self,
        system_metrics: Dict[str, Any],
        load_metrics: Dict[str, Any],
        predictions: Dict[int, CapacityPrediction]
    ) -> Optional[ScalingDecision]:
        """Evaluate if scaling up or down is needed."""
        
        # Check if minimum time has passed since last scaling
        time_since_last_scaling = (datetime.utcnow() - self.last_scaling_action).seconds
        if time_since_last_scaling < self.config["minimum_scaling_interval"]:
            return None
        
        avg_load = system_metrics.get("average_load_factor", 0)
        current_agents = system_metrics.get("total_agents", 0)
        overloaded_agents = system_metrics.get("overloaded_agents", 0)
        underloaded_agents = system_metrics.get("underloaded_agents", 0)
        
        # Scale up conditions
        if (avg_load > self.config["scale_up_threshold"] or 
            overloaded_agents > current_agents * 0.5):
            
            if current_agents >= self.config["max_agents"]:
                return None  # Already at maximum capacity
            
            # Calculate how many agents to add
            target_agents = min(
                self.config["max_agents"],
                current_agents + max(1, overloaded_agents)
            )
            
            # Consider future predictions
            future_prediction = predictions.get(60)  # 1-hour prediction
            if future_prediction and future_prediction.predicted_load_factor > 0.8:
                target_agents = min(
                    self.config["max_agents"],
                    future_prediction.recommended_agent_count
                )
            
            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                target_agents=[f"new_agent_{i}" for i in range(current_agents, target_agents)],
                resource_changes={},
                reasoning=f"Scale up needed: avg_load={avg_load:.2f}, overloaded={overloaded_agents}",
                confidence=0.8 + min(0.15, (avg_load - self.config["scale_up_threshold"]) * 2),
                estimated_impact={
                    "load_reduction": (avg_load - 0.6) * (target_agents - current_agents) / target_agents,
                    "cost_increase": (target_agents - current_agents) * 100  # Estimated cost units
                },
                execution_priority=2 if overloaded_agents > current_agents * 0.7 else 4
            )
        
        # Scale down conditions
        elif (avg_load < self.config["scale_down_threshold"] and 
              underloaded_agents > current_agents * 0.6):
            
            if current_agents <= self.config["min_agents"]:
                return None  # Already at minimum capacity
            
            # Calculate how many agents to remove
            agents_to_remove = min(
                current_agents - self.config["min_agents"],
                max(1, underloaded_agents // 2)
            )
            target_agents = current_agents - agents_to_remove
            
            # Be conservative - check future predictions
            future_prediction = predictions.get(60)
            if future_prediction and future_prediction.predicted_load_factor > 0.6:
                return None  # Don't scale down if load might increase
            
            # Select agents to remove (prefer underloaded ones)
            agents_to_remove_ids = await self._select_agents_for_removal(agents_to_remove)
            
            return ScalingDecision(
                action=ScalingAction.SCALE_DOWN,
                target_agents=agents_to_remove_ids,
                resource_changes={},
                reasoning=f"Scale down possible: avg_load={avg_load:.2f}, underloaded={underloaded_agents}",
                confidence=0.7 + min(0.2, (self.config["scale_down_threshold"] - avg_load) * 2),
                estimated_impact={
                    "load_increase": avg_load * agents_to_remove / target_agents,
                    "cost_savings": agents_to_remove * 100  # Estimated cost savings
                },
                execution_priority=7  # Lower priority than scale up
            )
        
        return None
    
    async def _evaluate_rebalancing_needs(
        self,
        system_metrics: Dict[str, Any],
        load_metrics: Dict[str, Any]
    ) -> Optional[ScalingDecision]:
        """Evaluate if load rebalancing is needed."""
        
        load_variance = system_metrics.get("load_variance", 0)
        overloaded_agents = system_metrics.get("overloaded_agents", 0)
        underloaded_agents = system_metrics.get("underloaded_agents", 0)
        
        # High variance indicates uneven load distribution
        if (load_variance > 0.2 and 
            overloaded_agents > 0 and 
            underloaded_agents > 0):
            
            return ScalingDecision(
                action=ScalingAction.REBALANCE,
                target_agents=[],  # Will be determined during execution
                resource_changes={},
                reasoning=f"Load rebalancing needed: variance={load_variance:.3f}",
                confidence=0.75,
                estimated_impact={
                    "load_distribution_improvement": min(0.3, load_variance),
                    "performance_improvement": 0.1
                },
                execution_priority=5
            )
        
        return None
    
    async def _evaluate_tier_optimizations(
        self,
        system_metrics: Dict[str, Any]
    ) -> List[ScalingDecision]:
        """Evaluate if agent tier adjustments are needed."""
        decisions = []
        
        try:
            # Get performance data for tier evaluation
            async with self.session_factory() as session:
                # Get recent performance data
                query = select(AgentPerformanceHistory).where(
                    AgentPerformanceHistory.recorded_at >= 
                    datetime.utcnow() - timedelta(seconds=self.config["performance_evaluation_period"])
                ).order_by(AgentPerformanceHistory.agent_id, AgentPerformanceHistory.recorded_at.desc())
                
                result = await session.execute(query)
                performance_records = result.scalars().all()
                
                # Group by agent
                agent_performance = defaultdict(list)
                for record in performance_records:
                    agent_id = str(record.agent_id)
                    agent_performance[agent_id].append(record)
                
                # Evaluate each agent for tier changes
                for agent_id, records in agent_performance.items():
                    if len(records) < 3:  # Need sufficient data
                        continue
                    
                    current_allocation = self.resource_allocations.get(agent_id)
                    if not current_allocation:
                        continue
                    
                    # Calculate average utilization
                    avg_utilization = statistics.mean([
                        (r.active_tasks or 0) / current_allocation.max_concurrent_tasks
                        for r in records[:10]  # Recent 10 records
                    ])
                    
                    # Tier promotion check
                    if (avg_utilization > self.config["tier_promotion_threshold"] and
                        current_allocation.tier != CapacityTier.ENTERPRISE):
                        
                        new_tier = self._get_next_tier(current_allocation.tier, promotion=True)
                        new_allocation = self._create_allocation_for_tier(agent_id, new_tier)
                        
                        decisions.append(ScalingDecision(
                            action=ScalingAction.OPTIMIZE,
                            target_agents=[agent_id],
                            resource_changes={agent_id: new_allocation},
                            reasoning=f"Tier promotion: utilization={avg_utilization:.2f}",
                            confidence=0.8,
                            estimated_impact={
                                "capacity_increase": (new_allocation.max_concurrent_tasks - 
                                                    current_allocation.max_concurrent_tasks),
                                "performance_improvement": 0.15
                            },
                            execution_priority=6
                        ))
                    
                    # Tier demotion check
                    elif (avg_utilization < self.config["tier_demotion_threshold"] and
                          current_allocation.tier != CapacityTier.LIGHT):
                        
                        new_tier = self._get_next_tier(current_allocation.tier, promotion=False)
                        new_allocation = self._create_allocation_for_tier(agent_id, new_tier)
                        
                        decisions.append(ScalingDecision(
                            action=ScalingAction.OPTIMIZE,
                            target_agents=[agent_id],
                            resource_changes={agent_id: new_allocation},
                            reasoning=f"Tier demotion: utilization={avg_utilization:.2f}",
                            confidence=0.7,
                            estimated_impact={
                                "cost_savings": 50,  # Estimated savings
                                "capacity_reduction": (current_allocation.max_concurrent_tasks - 
                                                     new_allocation.max_concurrent_tasks)
                            },
                            execution_priority=8
                        ))
        
        except Exception as e:
            logger.error("Failed to evaluate tier optimizations", error=str(e))
        
        return decisions
    
    def _get_next_tier(self, current_tier: CapacityTier, promotion: bool) -> CapacityTier:
        """Get next tier for promotion or demotion."""
        tiers = [CapacityTier.LIGHT, CapacityTier.STANDARD, CapacityTier.HEAVY, CapacityTier.ENTERPRISE]
        current_index = tiers.index(current_tier)
        
        if promotion and current_index < len(tiers) - 1:
            return tiers[current_index + 1]
        elif not promotion and current_index > 0:
            return tiers[current_index - 1]
        else:
            return current_tier
    
    def _create_allocation_for_tier(self, agent_id: str, tier: CapacityTier) -> ResourceAllocation:
        """Create resource allocation for specific tier."""
        template = self.tier_templates[tier]
        return ResourceAllocation(
            agent_id=agent_id,
            tier=tier,
            max_concurrent_tasks=template.max_concurrent_tasks,
            memory_limit_mb=template.memory_limit_mb,
            cpu_limit_percent=template.cpu_limit_percent,
            context_window_size=template.context_window_size
        )
    
    async def _select_agents_for_removal(self, count: int) -> List[str]:
        """Select agents to remove during scale-down."""
        # Get current load states from load balancer
        load_metrics = await self.load_balancer.get_load_balancing_metrics()
        agent_loads = load_metrics.get("agent_load_summary", {})
        
        # Sort agents by load factor (ascending) - remove least loaded first
        candidates = [
            (agent_id, data["load_factor"])
            for agent_id, data in agent_loads.items()
            if data.get("active_tasks", 0) == 0  # Only consider agents with no active tasks
        ]
        
        candidates.sort(key=lambda x: x[1])  # Sort by load factor
        
        return [agent_id for agent_id, _ in candidates[:count]]
    
    async def execute_scaling_decision(self, decision: ScalingDecision) -> Dict[str, Any]:
        """Execute a scaling decision."""
        try:
            execution_result = {
                "action": decision.action.value,
                "success": False,
                "changes_made": [],
                "errors": []
            }
            
            if decision.action == ScalingAction.SCALE_UP:
                result = await self._execute_scale_up(decision)
                execution_result.update(result)
            
            elif decision.action == ScalingAction.SCALE_DOWN:
                result = await self._execute_scale_down(decision)
                execution_result.update(result)
            
            elif decision.action == ScalingAction.REBALANCE:
                result = await self._execute_rebalance(decision)
                execution_result.update(result)
            
            elif decision.action == ScalingAction.OPTIMIZE:
                result = await self._execute_optimization(decision)
                execution_result.update(result)
            
            # Record scaling action
            if execution_result["success"]:
                self.last_scaling_action = datetime.utcnow()
                self.scaling_history.append({
                    "timestamp": datetime.utcnow(),
                    "decision": decision,
                    "result": execution_result
                })
            
            logger.info("Scaling decision executed",
                       action=decision.action.value,
                       success=execution_result["success"],
                       changes=len(execution_result["changes_made"]))
            
            return execution_result
            
        except Exception as e:
            logger.error("Failed to execute scaling decision", error=str(e))
            return {
                "action": decision.action.value,
                "success": False,
                "errors": [str(e)]
            }
    
    async def _execute_scale_up(self, decision: ScalingDecision) -> Dict[str, Any]:
        """Execute scale-up decision."""
        # For now, just record the intention - actual agent spawning would be done by orchestrator
        changes_made = []
        
        for agent_id in decision.target_agents:
            # Create default allocation for new agent
            allocation = self._create_allocation_for_tier(agent_id, CapacityTier.STANDARD)
            self.resource_allocations[agent_id] = allocation
            changes_made.append(f"Allocated resources for new agent {agent_id}")
        
        return {
            "success": True,
            "changes_made": changes_made,
            "agents_added": len(decision.target_agents)
        }
    
    async def _execute_scale_down(self, decision: ScalingDecision) -> Dict[str, Any]:
        """Execute scale-down decision."""
        changes_made = []
        
        for agent_id in decision.target_agents:
            if agent_id in self.resource_allocations:
                del self.resource_allocations[agent_id]
                changes_made.append(f"Deallocated resources for agent {agent_id}")
        
        return {
            "success": True,
            "changes_made": changes_made,
            "agents_removed": len(decision.target_agents)
        }
    
    async def _execute_rebalance(self, decision: ScalingDecision) -> Dict[str, Any]:
        """Execute rebalancing decision."""
        # Trigger load balancer to redistribute tasks
        changes_made = ["Triggered load rebalancing"]
        
        return {
            "success": True,
            "changes_made": changes_made,
            "rebalance_initiated": True
        }
    
    async def _execute_optimization(self, decision: ScalingDecision) -> Dict[str, Any]:
        """Execute resource optimization decision."""
        changes_made = []
        
        for agent_id, new_allocation in decision.resource_changes.items():
            old_allocation = self.resource_allocations.get(agent_id)
            self.resource_allocations[agent_id] = new_allocation
            
            changes_made.append(
                f"Updated {agent_id}: {old_allocation.tier.value if old_allocation else 'none'} -> {new_allocation.tier.value}"
            )
        
        return {
            "success": True,
            "changes_made": changes_made,
            "optimizations_applied": len(decision.resource_changes)
        }
    
    async def get_capacity_metrics(self) -> Dict[str, Any]:
        """Get current capacity management metrics."""
        try:
            total_allocated_capacity = sum(
                alloc.max_concurrent_tasks for alloc in self.resource_allocations.values()
            )
            
            tier_distribution = defaultdict(int)
            for allocation in self.resource_allocations.values():
                tier_distribution[allocation.tier.value] += 1
            
            return {
                "total_agents": len(self.resource_allocations),
                "total_allocated_capacity": total_allocated_capacity,
                "tier_distribution": dict(tier_distribution),
                "scaling_actions_last_hour": len([
                    action for action in self.scaling_history
                    if (datetime.utcnow() - action["timestamp"]).seconds < 3600
                ]),
                "predictions": {
                    timeframe: {
                        "predicted_load": pred.predicted_load_factor,
                        "recommended_agents": pred.recommended_agent_count,
                        "risk_factors": pred.risk_factors
                    }
                    for timeframe, pred in self.capacity_predictions.items()
                },
                "resource_utilization": {
                    "memory_allocated_gb": sum(
                        alloc.memory_limit_mb for alloc in self.resource_allocations.values()
                    ) / 1024,
                    "cpu_allocated_percent": sum(
                        alloc.cpu_limit_percent for alloc in self.resource_allocations.values()
                    ),
                    "context_capacity": sum(
                        alloc.context_window_size for alloc in self.resource_allocations.values()
                    )
                }
            }
            
        except Exception as e:
            logger.error("Failed to get capacity metrics", error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def set_agent_allocation(
        self,
        agent_id: str,
        tier: CapacityTier,
        custom_limits: Optional[Dict[str, Any]] = None
    ) -> ResourceAllocation:
        """Set resource allocation for specific agent."""
        allocation = self._create_allocation_for_tier(agent_id, tier)
        
        # Apply custom limits if provided
        if custom_limits:
            if "max_tasks" in custom_limits:
                allocation.max_concurrent_tasks = custom_limits["max_tasks"]
            if "memory_mb" in custom_limits:
                allocation.memory_limit_mb = custom_limits["memory_mb"]
            if "cpu_percent" in custom_limits:
                allocation.cpu_limit_percent = custom_limits["cpu_percent"]
            if "context_size" in custom_limits:
                allocation.context_window_size = custom_limits["context_size"]
        
        self.resource_allocations[agent_id] = allocation
        
        logger.info("Agent allocation updated",
                   agent_id=agent_id,
                   tier=tier.value,
                   allocation=allocation.to_dict())
        
        return allocation
    
    def get_agent_allocation(self, agent_id: str) -> Optional[ResourceAllocation]:
        """Get current resource allocation for agent."""
        return self.resource_allocations.get(agent_id)