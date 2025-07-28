"""
Advanced Agent Load Balancer for LeanVibe Agent Hive 2.0

Implements intelligent load distribution with real-time capacity monitoring,
adaptive scaling, and performance-based agent selection for optimal
multi-agent orchestration at scale.
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
import heapq
import math

import structlog
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .redis import get_message_broker, get_session_cache, AgentMessageBroker, SessionCache
from .database import get_session
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.agent_performance import WorkloadSnapshot, AgentPerformanceHistory, TaskRoutingDecision
from ..models.task import Task, TaskStatus, TaskPriority

logger = structlog.get_logger()


class LoadBalancingStrategy(Enum):
    """Available load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    CAPABILITY_AWARE = "capability_aware"
    PERFORMANCE_BASED = "performance_based"
    ADAPTIVE_HYBRID = "adaptive_hybrid"


class AgentLoadMetric(Enum):
    """Metrics used for load calculation."""
    ACTIVE_TASKS = "active_tasks"
    CONTEXT_USAGE = "context_usage"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


@dataclass
class AgentLoadState:
    """Current load state of an agent."""
    agent_id: str
    active_tasks: int = 0
    pending_tasks: int = 0
    context_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    average_response_time_ms: float = 0.0
    error_rate_percent: float = 0.0
    throughput_tasks_per_hour: float = 0.0
    estimated_capacity: float = 1.0
    utilization_ratio: float = 0.0
    health_score: float = 1.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def calculate_load_factor(self) -> float:
        """Calculate composite load factor (0.0 to 1.0+)."""
        # Task-based load (40% weight)
        max_concurrent_tasks = 5  # Configurable per agent type
        task_load = min(1.0, (self.active_tasks + self.pending_tasks * 0.5) / max_concurrent_tasks)
        
        # Resource-based load (35% weight)
        context_load = self.context_usage_percent / 100.0
        memory_load = min(1.0, self.memory_usage_mb / 1000.0)  # 1GB threshold
        cpu_load = self.cpu_usage_percent / 100.0
        resource_load = max(context_load, memory_load, cpu_load)
        
        # Performance-based load (25% weight)
        # Higher response time and error rate increase load
        response_penalty = min(0.5, self.average_response_time_ms / 10000.0)  # 10s = 0.5 penalty
        error_penalty = min(0.5, self.error_rate_percent / 100.0)
        performance_load = (1.0 - self.health_score) + response_penalty + error_penalty
        
        # Weighted combination
        composite_load = (
            task_load * 0.40 +
            resource_load * 0.35 +
            performance_load * 0.25
        )
        
        return min(2.0, max(0.0, composite_load))  # Cap at 2.0 for overload detection
    
    def can_handle_task(self, task_complexity: float = 1.0) -> bool:
        """Check if agent can handle additional task."""
        current_load = self.calculate_load_factor()
        projected_load = current_load + (task_complexity * 0.2)  # Estimate task impact
        
        return projected_load < 0.85 and self.health_score > 0.7
    
    def is_overloaded(self, threshold: float = 0.85) -> bool:
        """Check if agent is overloaded."""
        return self.calculate_load_factor() > threshold
    
    def is_underloaded(self, threshold: float = 0.3) -> bool:
        """Check if agent is underloaded."""
        return self.calculate_load_factor() < threshold


@dataclass  
class LoadBalancingDecision:
    """Result of load balancing decision."""
    selected_agent_id: str
    strategy_used: LoadBalancingStrategy
    decision_time_ms: float
    agent_scores: Dict[str, float]
    load_factors: Dict[str, float]
    decision_confidence: float
    reasoning: str
    alternative_agents: List[str] = field(default_factory=list)


class AgentLoadBalancer:
    """
    Advanced load balancer for intelligent agent task distribution.
    
    Features:
    - Multiple load balancing strategies
    - Real-time capacity monitoring
    - Performance-based agent selection
    - Adaptive strategy switching
    - Sub-100ms decision times
    """
    
    def __init__(
        self,
        redis_client=None,
        session_factory: Optional[Callable] = None,
        default_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ADAPTIVE_HYBRID
    ):
        self.redis_client = redis_client
        self.session_factory = session_factory or get_session
        self.default_strategy = default_strategy
        
        # Load state tracking
        self.agent_loads: Dict[str, AgentLoadState] = {}
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Strategy state
        self.round_robin_index = 0
        self.strategy_performance: Dict[LoadBalancingStrategy, deque] = {
            strategy: deque(maxlen=50) for strategy in LoadBalancingStrategy
        }
        
        # Performance tracking
        self.decision_times: deque = deque(maxlen=1000)
        self.last_metrics_update = datetime.utcnow()
        
        # Configuration
        self.config = {
            "max_decision_time_ms": 100,
            "load_update_interval_seconds": 30,
            "overload_threshold": 0.85,
            "underload_threshold": 0.3,
            "health_score_threshold": 0.7,
            "strategy_switch_threshold": 0.2  # Performance difference to trigger switch
        }
        
        logger.info("AgentLoadBalancer initialized", 
                   strategy=default_strategy.value,
                   config=self.config)
    
    async def select_agent_for_task(
        self,
        task: Task,
        available_agents: List[str],
        strategy: Optional[LoadBalancingStrategy] = None,
        required_capabilities: Optional[List[str]] = None
    ) -> LoadBalancingDecision:
        """
        Select optimal agent for task using specified or adaptive strategy.
        
        Args:
            task: Task to be assigned
            available_agents: List of available agent IDs
            strategy: Load balancing strategy to use (optional)
            required_capabilities: Required agent capabilities (optional)
            
        Returns:
            LoadBalancingDecision with selected agent and metadata
        """
        start_time = time.time()
        
        try:
            # Update agent load states
            await self._update_agent_loads(available_agents)
            
            # Filter agents by capabilities and health
            viable_agents = await self._filter_viable_agents(
                available_agents, required_capabilities, task
            )
            
            if not viable_agents:
                raise ValueError("No viable agents available for task assignment")
            
            # Select strategy
            selected_strategy = strategy or await self._select_optimal_strategy(
                viable_agents, task
            )
            
            # Execute load balancing decision
            decision = await self._execute_load_balancing_strategy(
                selected_strategy, viable_agents, task
            )
            
            # Record decision metrics
            decision_time_ms = (time.time() - start_time) * 1000
            decision.decision_time_ms = decision_time_ms
            
            self.decision_times.append(decision_time_ms)
            self.strategy_performance[selected_strategy].append(decision_time_ms)
            
            # Log performance warning if decision took too long
            if decision_time_ms > self.config["max_decision_time_ms"]:
                logger.warning("Load balancing decision exceeded target time",
                              decision_time_ms=decision_time_ms,
                              target_ms=self.config["max_decision_time_ms"],
                              strategy=selected_strategy.value)
            
            logger.info("Agent selected for task",
                       agent_id=decision.selected_agent_id,
                       task_id=str(task.id),
                       strategy=selected_strategy.value,
                       decision_time_ms=decision_time_ms,
                       confidence=decision.decision_confidence)
            
            return decision
            
        except Exception as e:
            logger.error("Load balancing decision failed",
                        error=str(e),
                        task_id=str(task.id),
                        available_agents=len(available_agents))
            raise
    
    async def _update_agent_loads(self, agent_ids: List[str]) -> None:
        """Update load states for specified agents."""
        if (datetime.utcnow() - self.last_metrics_update).seconds < self.config["load_update_interval_seconds"]:
            return  # Skip update if too recent
        
        try:
            async with self.session_factory() as session:
                # Get latest workload snapshots
                query = select(WorkloadSnapshot).where(
                    WorkloadSnapshot.agent_id.in_(agent_ids)
                ).order_by(WorkloadSnapshot.snapshot_time.desc())
                
                result = await session.execute(query)
                snapshots = result.scalars().unique().all()
                
                # Group by agent_id and take most recent
                latest_snapshots = {}
                for snapshot in snapshots:
                    agent_id = str(snapshot.agent_id)
                    if agent_id not in latest_snapshots:
                        latest_snapshots[agent_id] = snapshot
                
                # Update load states
                for agent_id in agent_ids:
                    if agent_id in latest_snapshots:
                        snapshot = latest_snapshots[agent_id]
                        self.agent_loads[agent_id] = AgentLoadState(
                            agent_id=agent_id,
                            active_tasks=snapshot.active_tasks,
                            pending_tasks=snapshot.pending_tasks,
                            context_usage_percent=snapshot.context_usage_percent,
                            memory_usage_mb=snapshot.memory_usage_mb or 0.0,
                            cpu_usage_percent=snapshot.cpu_usage_percent or 0.0,
                            average_response_time_ms=snapshot.average_response_time_ms or 0.0,
                            error_rate_percent=snapshot.error_rate_percent,
                            throughput_tasks_per_hour=snapshot.throughput_tasks_per_hour or 0.0,
                            estimated_capacity=snapshot.estimated_capacity,
                            utilization_ratio=snapshot.utilization_ratio,
                            health_score=1.0 - (snapshot.error_rate_percent / 100.0),  # Simple health calc
                            last_updated=snapshot.snapshot_time
                        )
                    else:
                        # Initialize with default state if no snapshot available
                        if agent_id not in self.agent_loads:
                            self.agent_loads[agent_id] = AgentLoadState(agent_id=agent_id)
                
                # Record load history for trending
                for agent_id, load_state in self.agent_loads.items():
                    self.load_history[agent_id].append({
                        'timestamp': datetime.utcnow(),
                        'load_factor': load_state.calculate_load_factor(),
                        'health_score': load_state.health_score
                    })
                
                self.last_metrics_update = datetime.utcnow()
                
        except Exception as e:
            logger.error("Failed to update agent load states", error=str(e))
    
    async def _filter_viable_agents(
        self,
        available_agents: List[str],
        required_capabilities: Optional[List[str]],
        task: Task
    ) -> List[str]:
        """Filter agents based on viability for task."""
        viable_agents = []
        
        for agent_id in available_agents:
            load_state = self.agent_loads.get(agent_id)
            if not load_state:
                continue
            
            # Check basic health and capacity
            if (load_state.health_score >= self.config["health_score_threshold"] and
                load_state.can_handle_task(task.complexity_score or 1.0)):
                
                # TODO: Add capability matching logic here
                # For now, assume all agents are viable
                viable_agents.append(agent_id)
        
        return viable_agents
    
    async def _select_optimal_strategy(
        self,
        viable_agents: List[str],
        task: Task
    ) -> LoadBalancingStrategy:
        """Select optimal load balancing strategy based on current conditions."""
        if self.default_strategy != LoadBalancingStrategy.ADAPTIVE_HYBRID:
            return self.default_strategy
        
        # Analyze current system state
        total_load = sum(
            self.agent_loads[agent_id].calculate_load_factor() 
            for agent_id in viable_agents
            if agent_id in self.agent_loads
        )
        avg_load = total_load / len(viable_agents) if viable_agents else 0
        
        load_variance = statistics.variance([
            self.agent_loads[agent_id].calculate_load_factor()
            for agent_id in viable_agents
            if agent_id in self.agent_loads
        ]) if len(viable_agents) > 1 else 0
        
        # Strategy selection logic
        if avg_load > 0.7:  # High load - prioritize performance
            return LoadBalancingStrategy.PERFORMANCE_BASED
        elif load_variance > 0.2:  # Uneven load - balance it out
            return LoadBalancingStrategy.LEAST_CONNECTIONS
        elif task.priority == TaskPriority.HIGH:  # High priority - use best agent
            return LoadBalancingStrategy.CAPABILITY_AWARE
        else:  # Normal conditions - simple round robin
            return LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN
    
    async def _execute_load_balancing_strategy(
        self,
        strategy: LoadBalancingStrategy,
        viable_agents: List[str],
        task: Task
    ) -> LoadBalancingDecision:
        """Execute the specified load balancing strategy."""
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return await self._round_robin_selection(viable_agents, task)
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return await self._least_connections_selection(viable_agents, task)
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return await self._weighted_round_robin_selection(viable_agents, task)
        elif strategy == LoadBalancingStrategy.CAPABILITY_AWARE:
            return await self._capability_aware_selection(viable_agents, task)
        elif strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
            return await self._performance_based_selection(viable_agents, task)
        else:
            # Fallback to round robin
            return await self._round_robin_selection(viable_agents, task)
    
    async def _round_robin_selection(
        self,
        viable_agents: List[str],
        task: Task
    ) -> LoadBalancingDecision:
        """Simple round-robin agent selection."""
        selected_agent = viable_agents[self.round_robin_index % len(viable_agents)]
        self.round_robin_index += 1
        
        agent_scores = {agent_id: 1.0 for agent_id in viable_agents}
        load_factors = {
            agent_id: self.agent_loads.get(agent_id, AgentLoadState(agent_id)).calculate_load_factor()
            for agent_id in viable_agents
        }
        
        return LoadBalancingDecision(
            selected_agent_id=selected_agent,
            strategy_used=LoadBalancingStrategy.ROUND_ROBIN,
            decision_time_ms=0.0,  # Will be set by caller
            agent_scores=agent_scores,
            load_factors=load_factors,
            decision_confidence=0.8,
            reasoning="Round-robin selection for even distribution",
            alternative_agents=viable_agents[:3]  # Top 3 alternatives
        )
    
    async def _least_connections_selection(
        self,
        viable_agents: List[str],
        task: Task
    ) -> LoadBalancingDecision:
        """Select agent with least active connections/tasks."""
        agent_loads = []
        
        for agent_id in viable_agents:
            load_state = self.agent_loads.get(agent_id, AgentLoadState(agent_id))
            load_factor = load_state.calculate_load_factor()
            agent_loads.append((load_factor, agent_id))
        
        # Sort by load factor (ascending)
        agent_loads.sort(key=lambda x: x[0])
        selected_agent = agent_loads[0][1]
        
        agent_scores = {
            agent_id: 1.0 - load_factor for load_factor, agent_id in agent_loads
        }
        load_factors = {agent_id: load_factor for load_factor, agent_id in agent_loads}
        
        return LoadBalancingDecision(
            selected_agent_id=selected_agent,
            strategy_used=LoadBalancingStrategy.LEAST_CONNECTIONS,
            decision_time_ms=0.0,
            agent_scores=agent_scores,
            load_factors=load_factors,
            decision_confidence=0.9,
            reasoning=f"Selected agent with lowest load factor: {agent_loads[0][0]:.3f}",
            alternative_agents=[agent_id for _, agent_id in agent_loads[1:4]]
        )
    
    async def _weighted_round_robin_selection(
        self,
        viable_agents: List[str],
        task: Task
    ) -> LoadBalancingDecision:
        """Weighted round-robin based on agent capacity."""
        agent_weights = []
        
        for agent_id in viable_agents:
            load_state = self.agent_loads.get(agent_id, AgentLoadState(agent_id))
            # Weight is inverse of load factor
            weight = max(0.1, 1.0 - load_state.calculate_load_factor())
            agent_weights.append((weight, agent_id))
        
        # Select based on cumulative weights
        total_weight = sum(weight for weight, _ in agent_weights)
        if total_weight == 0:
            return await self._round_robin_selection(viable_agents, task)
        
        # Use round-robin index to distribute fairly across weighted selections
        cumulative_weight = 0
        target_weight = (self.round_robin_index * total_weight / len(viable_agents)) % total_weight
        
        selected_agent = agent_weights[0][1]  # Fallback
        for weight, agent_id in agent_weights:
            cumulative_weight += weight
            if cumulative_weight >= target_weight:
                selected_agent = agent_id
                break
        
        self.round_robin_index += 1
        
        agent_scores = {agent_id: weight / total_weight for weight, agent_id in agent_weights}
        load_factors = {
            agent_id: self.agent_loads.get(agent_id, AgentLoadState(agent_id)).calculate_load_factor()
            for agent_id in viable_agents
        }
        
        return LoadBalancingDecision(
            selected_agent_id=selected_agent,
            strategy_used=LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN,
            decision_time_ms=0.0,
            agent_scores=agent_scores,
            load_factors=load_factors,
            decision_confidence=0.85,
            reasoning=f"Weighted selection based on capacity (weight: {agent_scores[selected_agent]:.3f})",
            alternative_agents=[agent_id for _, agent_id in sorted(agent_weights, reverse=True)[1:4]]
        )
    
    async def _capability_aware_selection(
        self,
        viable_agents: List[str],
        task: Task
    ) -> LoadBalancingDecision:
        """Select agent based on capabilities and specialization."""
        # For now, use performance-based selection as proxy for capability
        # TODO: Integrate with capability matcher once implemented
        return await self._performance_based_selection(viable_agents, task)
    
    async def _performance_based_selection(
        self,
        viable_agents: List[str],
        task: Task
    ) -> LoadBalancingDecision:
        """Select agent based on historical performance metrics."""
        agent_scores = {}
        
        try:
            async with self.session_factory() as session:
                # Get recent performance history for agents
                query = select(AgentPerformanceHistory).where(
                    AgentPerformanceHistory.agent_id.in_(viable_agents),
                    AgentPerformanceHistory.recorded_at >= datetime.utcnow() - timedelta(hours=24)
                ).order_by(AgentPerformanceHistory.recorded_at.desc())
                
                result = await session.execute(query)
                performance_records = result.scalars().all()
                
                # Calculate performance scores
                agent_performance = defaultdict(list)
                for record in performance_records:
                    agent_id = str(record.agent_id)
                    if agent_id in viable_agents:
                        # Calculate composite performance score
                        efficiency = record.calculate_efficiency_score()
                        reliability = record.calculate_reliability_score()
                        composite_score = (efficiency * 0.6 + reliability * 0.4)
                        agent_performance[agent_id].append(composite_score)
                
                # Calculate average performance scores
                for agent_id in viable_agents:
                    if agent_id in agent_performance:
                        agent_scores[agent_id] = statistics.mean(agent_performance[agent_id])
                    else:
                        agent_scores[agent_id] = 0.5  # Default score for agents without history
                
                # Adjust scores based on current load
                for agent_id in viable_agents:
                    load_state = self.agent_loads.get(agent_id, AgentLoadState(agent_id))
                    load_penalty = load_state.calculate_load_factor() * 0.3
                    agent_scores[agent_id] = max(0.0, agent_scores[agent_id] - load_penalty)
        
        except Exception as e:
            logger.error("Error calculating performance scores", error=str(e))
            # Fallback to load-based selection
            for agent_id in viable_agents:
                load_state = self.agent_loads.get(agent_id, AgentLoadState(agent_id))
                agent_scores[agent_id] = 1.0 - load_state.calculate_load_factor()
        
        # Select agent with highest score
        if not agent_scores:
            return await self._round_robin_selection(viable_agents, task)
        
        selected_agent = max(agent_scores, key=agent_scores.get)
        
        load_factors = {
            agent_id: self.agent_loads.get(agent_id, AgentLoadState(agent_id)).calculate_load_factor()
            for agent_id in viable_agents
        }
        
        return LoadBalancingDecision(
            selected_agent_id=selected_agent,
            strategy_used=LoadBalancingStrategy.PERFORMANCE_BASED,
            decision_time_ms=0.0,
            agent_scores=agent_scores,
            load_factors=load_factors,
            decision_confidence=0.95,
            reasoning=f"Performance-based selection (score: {agent_scores[selected_agent]:.3f})",
            alternative_agents=sorted(agent_scores.keys(), key=agent_scores.get, reverse=True)[1:4]
        )
    
    async def get_load_balancing_metrics(self) -> Dict[str, Any]:
        """Get current load balancing performance metrics."""
        if not self.decision_times:
            return {"status": "no_data"}
        
        return {
            "decision_metrics": {
                "average_decision_time_ms": statistics.mean(self.decision_times),
                "p95_decision_time_ms": statistics.quantiles(self.decision_times, n=20)[18] if len(self.decision_times) > 20 else max(self.decision_times),
                "p99_decision_time_ms": statistics.quantiles(self.decision_times, n=100)[98] if len(self.decision_times) > 100 else max(self.decision_times),
                "total_decisions": len(self.decision_times)
            },
            "agent_load_summary": {
                agent_id: {
                    "load_factor": load_state.calculate_load_factor(),
                    "health_score": load_state.health_score,
                    "active_tasks": load_state.active_tasks,
                    "can_handle_more": load_state.can_handle_task()
                }
                for agent_id, load_state in self.agent_loads.items()
            },
            "strategy_performance": {
                strategy.value: {
                    "average_time_ms": statistics.mean(times) if times else 0,
                    "usage_count": len(times)
                }
                for strategy, times in self.strategy_performance.items()
            },
            "system_health": {
                "overloaded_agents": len([
                    agent_id for agent_id, load_state in self.agent_loads.items()
                    if load_state.is_overloaded()
                ]),
                "underloaded_agents": len([
                    agent_id for agent_id, load_state in self.agent_loads.items()
                    if load_state.is_underloaded()
                ]),
                "total_agents": len(self.agent_loads)
            }
        }
    
    async def update_agent_load_state(self, agent_id: str, **kwargs) -> None:
        """Update specific agent load state parameters."""
        if agent_id not in self.agent_loads:
            self.agent_loads[agent_id] = AgentLoadState(agent_id=agent_id)
        
        load_state = self.agent_loads[agent_id]
        
        # Update provided parameters
        for key, value in kwargs.items():
            if hasattr(load_state, key):
                setattr(load_state, key, value)
        
        load_state.last_updated = datetime.utcnow()
        
        # Record in history
        self.load_history[agent_id].append({
            'timestamp': datetime.utcnow(),
            'load_factor': load_state.calculate_load_factor(),
            'health_score': load_state.health_score
        })
        
        logger.debug("Agent load state updated",
                    agent_id=agent_id,
                    load_factor=load_state.calculate_load_factor(),
                    health_score=load_state.health_score)