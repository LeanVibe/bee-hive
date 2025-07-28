"""
Task Scheduler Service for LeanVibe Agent Hive 2.0

Provides intelligent agent-task assignment with capability matching,
load balancing, performance optimization, and comprehensive scheduling algorithms.
Integrates with TaskQueue and AgentRegistry for optimal orchestration.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import math
import random

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, func

from .database import get_session
from .agent_registry import AgentRegistry, AgentResourceUsage
from .task_queue import TaskQueue, QueuedTask, TaskAssignmentResult
from ..models.agent import Agent, AgentStatus
from ..models.task import Task, TaskStatus, TaskPriority, TaskType

logger = structlog.get_logger()


class SchedulingStrategy(Enum):
    """Task scheduling strategies."""
    ROUND_ROBIN = "round_robin"
    CAPABILITY_MATCH = "capability_match"
    LOAD_BALANCED = "load_balanced"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    HYBRID = "hybrid"


@dataclass
class AgentSuitabilityScore:
    """Agent suitability score for task assignment."""
    agent_id: uuid.UUID
    total_score: float
    capability_score: float
    load_score: float
    performance_score: float
    availability_score: float
    reasoning: List[str]


@dataclass
class SchedulingDecision:
    """Result of scheduling decision."""
    success: bool
    task_id: uuid.UUID
    assigned_agent_id: Optional[uuid.UUID]
    assignment_confidence: float
    scheduling_strategy: SchedulingStrategy
    decision_time_ms: float
    reasoning: List[str]
    error_message: Optional[str]


class TaskScheduler:
    """
    Intelligent task scheduler with multiple assignment strategies.
    
    Features:
    - Multiple scheduling algorithms
    - Capability-based matching
    - Load balancing across agents
    - Performance-optimized assignment
    - Real-time agent monitoring
    - Assignment confidence scoring
    - Fallback strategies
    """
    
    def __init__(
        self,
        agent_registry: Optional[AgentRegistry] = None,
        task_queue: Optional[TaskQueue] = None
    ):
        self.agent_registry = agent_registry
        self.task_queue = task_queue
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
        
        # Scheduling configuration
        self.default_strategy = SchedulingStrategy.HYBRID
        self.assignment_timeout_seconds = 2.0
        self.max_concurrent_assignments = 50
        self.capability_match_threshold = 0.7
        self.load_balance_weight = 0.3
        self.performance_weight = 0.4
        self.availability_weight = 0.3
        
        # Metrics and tracking
        self._assignment_metrics = {
            "total_assignments": 0,
            "successful_assignments": 0,
            "failed_assignments": 0,
            "strategy_usage": {strategy: 0 for strategy in SchedulingStrategy},
            "average_assignment_time_ms": 0.0,
            "average_confidence": 0.0
        }
        
        # Round-robin state
        self._round_robin_index = 0
        self._active_agents_cache: List[Agent] = []
        self._cache_last_updated = datetime.min
        self._cache_ttl_seconds = 30
    
    async def start(self) -> None:
        """Start the task scheduler service."""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._assignment_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._cache_refresh_loop())
        ]
        
        logger.info("TaskScheduler started")
    
    async def stop(self) -> None:
        """Stop the task scheduler and cleanup resources."""
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        logger.info("TaskScheduler stopped")
    
    async def assign_task(
        self,
        task_id: uuid.UUID,
        strategy: Optional[SchedulingStrategy] = None,
        preferred_agent_id: Optional[uuid.UUID] = None,
        timeout_seconds: Optional[float] = None
    ) -> SchedulingDecision:
        """
        Assign a task to the most suitable agent.
        
        Args:
            task_id: Task ID to assign
            strategy: Scheduling strategy to use
            preferred_agent_id: Preferred agent (if available)
            timeout_seconds: Assignment timeout
            
        Returns:
            SchedulingDecision with assignment result
        """
        assignment_start = datetime.utcnow()
        strategy = strategy or self.default_strategy
        timeout_seconds = timeout_seconds or self.assignment_timeout_seconds
        
        try:
            # Get task details
            async with get_session() as db:
                result = await db.execute(
                    select(Task).where(Task.id == task_id)
                )
                task = result.scalar_one_or_none()
                
                if not task:
                    return SchedulingDecision(
                        success=False,
                        task_id=task_id,
                        assigned_agent_id=None,
                        assignment_confidence=0.0,
                        scheduling_strategy=strategy,
                        decision_time_ms=0.0,
                        reasoning=["Task not found"],
                        error_message="Task not found"
                    )
            
            # Check if preferred agent is available
            if preferred_agent_id:
                agent = await self.agent_registry.get_agent(preferred_agent_id)
                if agent and self._is_agent_available(agent):
                    decision = await self._assign_to_agent(
                        task, agent, SchedulingStrategy.CAPABILITY_MATCH, ["Preferred agent available"]
                    )
                    if decision.success:
                        decision.decision_time_ms = (datetime.utcnow() - assignment_start).total_seconds() * 1000
                        return decision
            
            # Find best agent using selected strategy
            suitable_agents = await self._find_suitable_agents(task, strategy)
            
            if not suitable_agents:
                return SchedulingDecision(
                    success=False,
                    task_id=task_id,
                    assigned_agent_id=None,
                    assignment_confidence=0.0,
                    scheduling_strategy=strategy,
                    decision_time_ms=(datetime.utcnow() - assignment_start).total_seconds() * 1000,
                    reasoning=["No suitable agents available"],
                    error_message="No suitable agents available"
                )
            
            # Select best agent
            best_agent_score = suitable_agents[0]
            agent = await self.agent_registry.get_agent(best_agent_score.agent_id)
            
            if not agent:
                return SchedulingDecision(
                    success=False,
                    task_id=task_id,
                    assigned_agent_id=None,
                    assignment_confidence=0.0,
                    scheduling_strategy=strategy,
                    decision_time_ms=(datetime.utcnow() - assignment_start).total_seconds() * 1000,
                    reasoning=["Selected agent not available"],
                    error_message="Selected agent not available"
                )
            
            # Assign task to agent
            decision = await self._assign_to_agent(task, agent, strategy, best_agent_score.reasoning)
            decision.assignment_confidence = best_agent_score.total_score
            decision.decision_time_ms = (datetime.utcnow() - assignment_start).total_seconds() * 1000
            
            # Update metrics
            self._assignment_metrics["total_assignments"] += 1
            if decision.success:
                self._assignment_metrics["successful_assignments"] += 1
            else:
                self._assignment_metrics["failed_assignments"] += 1
            
            self._assignment_metrics["strategy_usage"][strategy] += 1
            
            # Update average assignment time
            current_avg = self._assignment_metrics["average_assignment_time_ms"]
            total_assignments = self._assignment_metrics["total_assignments"]
            new_avg = ((current_avg * (total_assignments - 1)) + decision.decision_time_ms) / total_assignments
            self._assignment_metrics["average_assignment_time_ms"] = new_avg
            
            # Update average confidence
            current_confidence_avg = self._assignment_metrics["average_confidence"]
            new_confidence_avg = ((current_confidence_avg * (total_assignments - 1)) + decision.assignment_confidence) / total_assignments
            self._assignment_metrics["average_confidence"] = new_confidence_avg
            
            return decision
            
        except Exception as e:
            logger.error("Task assignment failed", task_id=str(task_id), error=str(e))
            
            return SchedulingDecision(
                success=False,
                task_id=task_id,
                assigned_agent_id=None,
                assignment_confidence=0.0,
                scheduling_strategy=strategy,
                decision_time_ms=(datetime.utcnow() - assignment_start).total_seconds() * 1000,
                reasoning=[f"Assignment error: {str(e)}"],
                error_message=str(e)
            )
    
    async def _find_suitable_agents(
        self,
        task: Task,
        strategy: SchedulingStrategy
    ) -> List[AgentSuitabilityScore]:
        """Find and score suitable agents for a task."""
        try:
            # Get active agents
            await self._refresh_agents_cache()
            active_agents = [agent for agent in self._active_agents_cache if self._is_agent_available(agent)]
            
            if not active_agents:
                return []
            
            # Score agents based on strategy
            agent_scores = []
            
            for agent in active_agents:
                score = await self._calculate_agent_suitability(task, agent, strategy)
                agent_scores.append(score)
            
            # Filter by minimum capability match
            qualified_scores = [
                score for score in agent_scores
                if score.capability_score >= self.capability_match_threshold
            ]
            
            # Sort by total score (descending)
            qualified_scores.sort(key=lambda x: x.total_score, reverse=True)
            
            return qualified_scores
            
        except Exception as e:
            logger.error("Failed to find suitable agents", task_id=str(task.id), error=str(e))
            return []
    
    async def _calculate_agent_suitability(
        self,
        task: Task,
        agent: Agent,
        strategy: SchedulingStrategy
    ) -> AgentSuitabilityScore:
        """Calculate comprehensive suitability score for an agent."""
        reasoning = []
        
        # Calculate capability score
        capability_score = self._calculate_capability_score(task, agent)
        reasoning.append(f"Capability match: {capability_score:.2f}")
        
        # Calculate load score
        load_score = self._calculate_load_score(agent)
        reasoning.append(f"Load score: {load_score:.2f}")
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(agent)
        reasoning.append(f"Performance score: {performance_score:.2f}")
        
        # Calculate availability score
        availability_score = self._calculate_availability_score(agent)
        reasoning.append(f"Availability score: {availability_score:.2f}")
        
        # Combine scores based on strategy
        if strategy == SchedulingStrategy.CAPABILITY_MATCH:
            total_score = capability_score * 0.8 + availability_score * 0.2
        elif strategy == SchedulingStrategy.LOAD_BALANCED:
            total_score = load_score * 0.6 + capability_score * 0.3 + availability_score * 0.1
        elif strategy == SchedulingStrategy.PERFORMANCE_OPTIMIZED:
            total_score = performance_score * 0.5 + capability_score * 0.3 + availability_score * 0.2
        elif strategy == SchedulingStrategy.ROUND_ROBIN:
            total_score = availability_score * 0.7 + capability_score * 0.3
        else:  # HYBRID
            total_score = (
                capability_score * 0.35 +
                load_score * self.load_balance_weight +
                performance_score * self.performance_weight +
                availability_score * self.availability_weight
            )
        
        reasoning.append(f"Total score ({strategy.value}): {total_score:.2f}")
        
        return AgentSuitabilityScore(
            agent_id=agent.id,
            total_score=total_score,
            capability_score=capability_score,
            load_score=load_score,
            performance_score=performance_score,
            availability_score=availability_score,
            reasoning=reasoning
        )
    
    def _calculate_capability_score(self, task: Task, agent: Agent) -> float:
        """Calculate capability match score (0.0 to 1.0)."""
        if not task.required_capabilities or not agent.capabilities:
            return 0.5  # Neutral score when no capabilities specified
        
        required_caps = set(cap.lower() for cap in task.required_capabilities)
        agent_caps = []
        
        # Extract capability names from agent capabilities
        for cap in agent.capabilities:
            if isinstance(cap, dict):
                agent_caps.append(cap.get("name", "").lower())
                # Also include specialization areas
                for area in cap.get("specialization_areas", []):
                    agent_caps.append(area.lower())
            else:
                agent_caps.append(str(cap).lower())
        
        agent_caps_set = set(agent_caps)
        
        # Calculate match ratio
        matches = len(required_caps.intersection(agent_caps_set))
        total_required = len(required_caps)
        
        if total_required == 0:
            return 1.0
        
        match_ratio = matches / total_required
        
        # Bonus for additional capabilities
        bonus = min(0.2, len(agent_caps_set - required_caps) * 0.02)
        
        return min(1.0, match_ratio + bonus)
    
    def _calculate_load_score(self, agent: Agent) -> float:
        """Calculate load score (1.0 = low load, 0.0 = high load)."""
        context_usage = float(agent.context_window_usage or 0.0)
        
        # Parse resource usage
        resource_usage = agent.resource_usage or {}
        active_tasks = resource_usage.get("active_tasks_count", 0)
        cpu_percent = resource_usage.get("cpu_percent", 0.0)
        
        # Calculate combined load score
        context_score = 1.0 - context_usage
        task_score = max(0.0, 1.0 - (active_tasks / 10.0))  # Assume 10 is max reasonable tasks
        cpu_score = max(0.0, 1.0 - (cpu_percent / 100.0))
        
        return (context_score * 0.5 + task_score * 0.3 + cpu_score * 0.2)
    
    def _calculate_performance_score(self, agent: Agent) -> float:
        """Calculate performance score based on historical metrics."""
        # Health score
        health_score = agent.health_score or 0.5
        
        # Response time (lower is better)
        avg_response_time = float(agent.average_response_time or 5.0)
        response_score = max(0.0, 1.0 - (avg_response_time / 10.0))  # Normalize to 10 seconds max
        
        # Success rate
        total_completed = int(agent.total_tasks_completed or 0)
        total_failed = int(agent.total_tasks_failed or 0)
        total_tasks = total_completed + total_failed
        
        success_rate = (total_completed / total_tasks) if total_tasks > 0 else 0.5
        
        return (health_score * 0.4 + response_score * 0.3 + success_rate * 0.3)
    
    def _calculate_availability_score(self, agent: Agent) -> float:
        """Calculate availability score."""
        if not self._is_agent_available(agent):
            return 0.0
        
        # Check heartbeat freshness
        if agent.last_heartbeat:
            heartbeat_age = (datetime.utcnow() - agent.last_heartbeat).total_seconds()
            heartbeat_score = max(0.0, 1.0 - (heartbeat_age / 300.0))  # 5 minutes max
        else:
            heartbeat_score = 0.0
        
        # Check if agent is idle vs busy
        status_score = 1.0 if agent.status == AgentStatus.ACTIVE else 0.5
        
        return (heartbeat_score * 0.7 + status_score * 0.3)
    
    def _is_agent_available(self, agent: Agent) -> bool:
        """Check if agent is available for task assignment."""
        if agent.status not in [AgentStatus.ACTIVE]:
            return False
        
        # Check context window usage
        context_usage = float(agent.context_window_usage or 0.0)
        if context_usage > 0.95:
            return False
        
        # Check health score
        if agent.health_score and agent.health_score < 0.3:
            return False
        
        # Check heartbeat (must be within last 5 minutes)
        if agent.last_heartbeat:
            heartbeat_age = (datetime.utcnow() - agent.last_heartbeat).total_seconds()
            if heartbeat_age > 300:
                return False
        
        return True
    
    async def _assign_to_agent(
        self,
        task: Task,
        agent: Agent,
        strategy: SchedulingStrategy,
        reasoning: List[str]
    ) -> SchedulingDecision:
        """Assign task to specific agent."""
        try:
            async with get_session() as db:
                # Update task assignment
                await db.execute(
                    update(Task)
                    .where(Task.id == task.id)
                    .values(
                        assigned_agent_id=agent.id,
                        status=TaskStatus.ASSIGNED,
                        assigned_at=datetime.utcnow(),
                        assignment_strategy=strategy.value,
                        orchestrator_metadata={
                            **(task.orchestrator_metadata or {}),
                            "assigned_at": datetime.utcnow().isoformat(),
                            "assignment_strategy": strategy.value,
                            "assigned_agent_id": str(agent.id)
                        },
                        updated_at=datetime.utcnow()
                    )
                )
                
                # Update agent status
                await db.execute(
                    update(Agent)
                    .where(Agent.id == agent.id)
                    .values(
                        status=AgentStatus.BUSY,
                        updated_at=datetime.utcnow()
                    )
                )
                
                await db.commit()
                
                # Record assignment metrics
                await self._record_assignment_metric(db, task.id, agent.id, strategy)
            
            logger.info(
                "Task assigned successfully",
                task_id=str(task.id),
                agent_id=str(agent.id),
                strategy=strategy.value
            )
            
            return SchedulingDecision(
                success=True,
                task_id=task.id,
                assigned_agent_id=agent.id,
                assignment_confidence=0.0,  # Will be set by caller
                scheduling_strategy=strategy,
                decision_time_ms=0.0,  # Will be set by caller
                reasoning=reasoning,
                error_message=None
            )
            
        except Exception as e:
            logger.error(
                "Failed to assign task to agent",
                task_id=str(task.id),
                agent_id=str(agent.id),
                error=str(e)
            )
            
            return SchedulingDecision(
                success=False,
                task_id=task.id,
                assigned_agent_id=None,
                assignment_confidence=0.0,
                scheduling_strategy=strategy,
                decision_time_ms=0.0,
                reasoning=[f"Assignment failed: {str(e)}"],
                error_message=str(e)
            )
    
    async def _record_assignment_metric(
        self,
        db: AsyncSession,
        task_id: uuid.UUID,
        agent_id: uuid.UUID,
        strategy: SchedulingStrategy
    ) -> None:
        """Record assignment metrics."""
        try:
            await db.execute(
                """
                INSERT INTO orchestrator_metrics 
                (metric_type, metric_name, metric_value, metric_unit, agent_id, task_id, metadata, measured_at, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8)
                """,
                (
                    "task_assignment",
                    "assignment_success",
                    1.0,
                    "count",
                    agent_id,
                    task_id,
                    {"strategy": strategy.value},
                    datetime.utcnow()
                )
            )
        except Exception as e:
            logger.warning("Failed to record assignment metric", error=str(e))
    
    async def _refresh_agents_cache(self) -> None:
        """Refresh the active agents cache."""
        current_time = datetime.utcnow()
        
        if (current_time - self._cache_last_updated).total_seconds() < self._cache_ttl_seconds:
            return
        
        try:
            if self.agent_registry:
                self._active_agents_cache = await self.agent_registry.get_active_agents()
                self._cache_last_updated = current_time
                
        except Exception as e:
            logger.error("Failed to refresh agents cache", error=str(e))
    
    async def get_scheduling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scheduling statistics."""
        try:
            await self._refresh_agents_cache()
            
            active_agents_count = len([agent for agent in self._active_agents_cache if self._is_agent_available(agent)])
            
            return {
                "scheduling_metrics": self._assignment_metrics.copy(),
                "active_agents_count": active_agents_count,
                "total_agents_cached": len(self._active_agents_cache),
                "cache_last_updated": self._cache_last_updated.isoformat(),
                "scheduler_status": "running" if self._running else "stopped",
                "configuration": {
                    "default_strategy": self.default_strategy.value,
                    "assignment_timeout_seconds": self.assignment_timeout_seconds,
                    "capability_match_threshold": self.capability_match_threshold,
                    "load_balance_weight": self.load_balance_weight,
                    "performance_weight": self.performance_weight,
                    "availability_weight": self.availability_weight
                }
            }
            
        except Exception as e:
            logger.error("Failed to get scheduling statistics", error=str(e))
            return {
                "scheduling_metrics": self._assignment_metrics.copy(),
                "error": str(e)
            }
    
    async def _assignment_loop(self) -> None:
        """Background assignment loop for queued tasks."""
        while self._running:
            try:
                # This would integrate with TaskQueue to automatically assign queued tasks
                # For now, assignments are made on-demand via assign_task()
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Assignment loop error", error=str(e))
                await asyncio.sleep(5)
    
    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop."""
        while self._running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(300)  # Collect every 5 minutes
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics collection loop error", error=str(e))
                await asyncio.sleep(60)
    
    async def _collect_metrics(self) -> None:
        """Collect and update scheduling metrics."""
        try:
            # Update cache for accurate metrics
            await self._refresh_agents_cache()
            
            # Additional metrics could be collected here
            # such as agent utilization, queue depths, etc.
            
        except Exception as e:
            logger.error("Failed to collect metrics", error=str(e))
    
    async def _cache_refresh_loop(self) -> None:
        """Background cache refresh loop."""
        while self._running:
            try:
                await self._refresh_agents_cache()
                await asyncio.sleep(self._cache_ttl_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cache refresh loop error", error=str(e))
                await asyncio.sleep(10)