"""
Task Distributor for LeanVibe Agent Hive 2.0 - Phase 6.1

Intelligent task distribution and agent selection for multi-agent workflow commands.
Provides load balancing, capability matching, and real-time task assignment optimization.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import math

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from .database import get_session
from .agent_registry import AgentRegistry, AgentResourceUsage
from .redis import get_message_broker, AgentMessageBroker
from ..schemas.custom_commands import AgentRole, AgentRequirement, WorkflowStep
from ..models.agent import Agent, AgentStatus
from ..models.task import Task, TaskStatus, TaskPriority

logger = structlog.get_logger()


class DistributionStrategy(str, Enum):
    """Task distribution strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CAPABILITY_MATCH = "capability_match"
    PERFORMANCE_BASED = "performance_based"
    HYBRID = "hybrid"


class TaskUrgency(str, Enum):
    """Task urgency levels for prioritization."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AgentLoadMetrics:
    """Agent load and performance metrics."""
    agent_id: str
    current_tasks: int
    cpu_usage: float
    memory_usage: float
    context_usage: float
    health_score: float
    response_time_ms: float
    success_rate: float
    last_task_completed: Optional[datetime]
    specialization_match_score: float


@dataclass
class TaskAssignment:
    """Task assignment result."""
    task_id: str
    agent_id: str
    assignment_score: float
    estimated_completion_time: datetime
    assignment_reason: str
    backup_agents: List[str]


@dataclass
class DistributionResult:
    """Result of task distribution operation."""
    assignments: List[TaskAssignment]
    unassigned_tasks: List[str]
    distribution_time_ms: float
    strategy_used: DistributionStrategy
    optimization_metrics: Dict[str, Any]


class TaskDistributor:
    """
    Intelligent task distributor for multi-agent workflow execution.
    
    Features:
    - Multiple distribution strategies with automatic selection
    - Real-time agent load monitoring and optimization
    - Capability-based agent matching with scoring
    - Dynamic workload balancing and resource optimization
    - Failure recovery with backup agent assignment
    - Performance analytics and continuous optimization
    """
    
    def __init__(
        self,
        agent_registry: AgentRegistry,
        message_broker: Optional[AgentMessageBroker] = None,
        default_strategy: DistributionStrategy = DistributionStrategy.HYBRID
    ):
        self.agent_registry = agent_registry
        self.message_broker = message_broker
        self.default_strategy = default_strategy
        
        # Load balancing configuration
        self.max_tasks_per_agent = 10
        self.max_cpu_threshold = 80.0
        self.max_memory_threshold = 85.0
        self.max_context_threshold = 90.0
        self.min_health_score = 0.7
        
        # Performance tracking
        self.agent_performance_cache: Dict[str, AgentLoadMetrics] = {}
        self.cache_ttl_seconds = 30
        self.last_cache_update = datetime.min
        
        # Distribution statistics
        self.distribution_stats = {
            "total_distributions": 0,
            "successful_assignments": 0,
            "failed_assignments": 0,
            "average_distribution_time_ms": 0.0,
            "strategy_usage": {strategy.value: 0 for strategy in DistributionStrategy}
        }
        
        # Optimization parameters
        self.capability_weight = 0.4
        self.load_weight = 0.3
        self.performance_weight = 0.2
        self.availability_weight = 0.1
        
        logger.info(
            "TaskDistributor initialized",
            default_strategy=default_strategy.value,
            max_tasks_per_agent=self.max_tasks_per_agent
        )
    
    async def distribute_tasks(
        self,
        workflow_steps: List[WorkflowStep],
        agent_requirements: List[AgentRequirement],
        execution_context: Dict[str, Any] = None,
        strategy_override: Optional[DistributionStrategy] = None,
        urgency: TaskUrgency = TaskUrgency.NORMAL
    ) -> DistributionResult:
        """
        Distribute workflow tasks to optimal agents.
        
        Args:
            workflow_steps: List of workflow steps to distribute
            agent_requirements: Agent requirements for the workflow
            execution_context: Additional execution context
            strategy_override: Override default distribution strategy
            urgency: Task urgency level
            
        Returns:
            DistributionResult with assignment details
        """
        start_time = datetime.utcnow()
        strategy = strategy_override or self.default_strategy
        execution_context = execution_context or {}
        
        try:
            logger.info(
                "Starting task distribution",
                workflow_steps=len(workflow_steps),
                agent_requirements=len(agent_requirements),
                strategy=strategy.value,
                urgency=urgency.value
            )
            
            # Update agent performance cache
            await self._update_agent_performance_cache()
            
            # Get available agents
            available_agents = await self._get_available_agents(agent_requirements)
            
            if not available_agents:
                logger.warning("No available agents found for task distribution")
                return DistributionResult(
                    assignments=[],
                    unassigned_tasks=[step.step for step in workflow_steps],
                    distribution_time_ms=0.0,
                    strategy_used=strategy,
                    optimization_metrics={"error": "no_available_agents"}
                )
            
            # Execute distribution strategy
            assignments = await self._execute_distribution_strategy(
                workflow_steps, available_agents, strategy, urgency, execution_context
            )
            
            # Identify unassigned tasks
            assigned_task_ids = {assignment.task_id for assignment in assignments}
            unassigned_tasks = [
                step.step for step in workflow_steps 
                if step.step not in assigned_task_ids
            ]
            
            # Calculate metrics
            distribution_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            optimization_metrics = self._calculate_optimization_metrics(
                assignments, available_agents
            )
            
            # Update statistics
            self._update_distribution_stats(strategy, distribution_time_ms, len(assignments))
            
            logger.info(
                "Task distribution completed",
                assigned_tasks=len(assignments),
                unassigned_tasks=len(unassigned_tasks),
                distribution_time_ms=distribution_time_ms,
                strategy=strategy.value
            )
            
            return DistributionResult(
                assignments=assignments,
                unassigned_tasks=unassigned_tasks,
                distribution_time_ms=distribution_time_ms,
                strategy_used=strategy,
                optimization_metrics=optimization_metrics
            )
            
        except Exception as e:
            logger.error("Task distribution failed", error=str(e))
            raise
    
    async def reassign_failed_task(
        self,
        task_id: str,
        failed_agent_id: str,
        task_requirements: AgentRequirement,
        exclude_agents: List[str] = None
    ) -> Optional[TaskAssignment]:
        """
        Reassign a failed task to a different agent.
        
        Args:
            task_id: ID of the failed task
            failed_agent_id: ID of the agent that failed
            task_requirements: Requirements for the task
            exclude_agents: List of agent IDs to exclude
            
        Returns:
            New task assignment or None if no suitable agent found
        """
        try:
            exclude_agents = exclude_agents or []
            exclude_agents.append(failed_agent_id)
            
            # Get available agents excluding failed agents
            available_agents = await self._get_available_agents([task_requirements])
            suitable_agents = [
                agent for agent in available_agents 
                if str(agent.id) not in exclude_agents
            ]
            
            if not suitable_agents:
                logger.warning(
                    "No suitable agents for task reassignment",
                    task_id=task_id,
                    failed_agent_id=failed_agent_id
                )
                return None
            
            # Find best agent for reassignment
            best_agent = await self._select_best_agent_for_task(
                task_requirements, suitable_agents, TaskUrgency.HIGH
            )
            
            if not best_agent:
                return None
            
            # Create assignment
            assignment = TaskAssignment(
                task_id=task_id,
                agent_id=str(best_agent.id),
                assignment_score=0.9,  # High priority reassignment
                estimated_completion_time=datetime.utcnow() + timedelta(minutes=30),
                assignment_reason="task_reassignment_after_failure",
                backup_agents=[str(agent.id) for agent in suitable_agents[1:3]]
            )
            
            logger.info(
                "Task reassigned successfully",
                task_id=task_id,
                new_agent_id=assignment.agent_id,
                failed_agent_id=failed_agent_id
            )
            
            return assignment
            
        except Exception as e:
            logger.error(
                "Task reassignment failed",
                task_id=task_id,
                failed_agent_id=failed_agent_id,
                error=str(e)
            )
            return None
    
    async def get_agent_workload_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current workload status for all agents."""
        try:
            await self._update_agent_performance_cache()
            
            workload_status = {}
            for agent_id, metrics in self.agent_performance_cache.items():
                workload_status[agent_id] = {
                    "current_tasks": metrics.current_tasks,
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "context_usage": metrics.context_usage,
                    "health_score": metrics.health_score,
                    "availability": self._calculate_agent_availability(metrics),
                    "load_score": self._calculate_load_score(metrics),
                    "last_updated": datetime.utcnow().isoformat()
                }
            
            return workload_status
            
        except Exception as e:
            logger.error("Failed to get agent workload status", error=str(e))
            return {}
    
    async def optimize_distribution_strategy(
        self,
        historical_data: Dict[str, Any]
    ) -> DistributionStrategy:
        """
        Analyze performance and recommend optimal distribution strategy.
        
        Args:
            historical_data: Historical performance data
            
        Returns:
            Recommended distribution strategy
        """
        try:
            # Analyze strategy performance
            strategy_performance = {}
            
            for strategy in DistributionStrategy:
                strategy_data = historical_data.get(strategy.value, {})
                if strategy_data:
                    success_rate = strategy_data.get("success_rate", 0.0)
                    avg_completion_time = strategy_data.get("avg_completion_time", 0.0)
                    resource_efficiency = strategy_data.get("resource_efficiency", 0.0)
                    
                    # Calculate composite performance score
                    performance_score = (
                        success_rate * 0.4 +
                        (1.0 / max(avg_completion_time, 1.0)) * 0.3 +
                        resource_efficiency * 0.3
                    )
                    
                    strategy_performance[strategy] = performance_score
            
            # Select best performing strategy
            if strategy_performance:
                best_strategy = max(strategy_performance, key=strategy_performance.get)
                
                logger.info(
                    "Distribution strategy optimized",
                    recommended_strategy=best_strategy.value,
                    performance_scores=strategy_performance
                )
                
                return best_strategy
            
            # Default to hybrid if no historical data
            return DistributionStrategy.HYBRID
            
        except Exception as e:
            logger.error("Strategy optimization failed", error=str(e))
            return DistributionStrategy.HYBRID
    
    # Private helper methods
    
    async def _update_agent_performance_cache(self) -> None:
        """Update cached agent performance metrics."""
        try:
            if (datetime.utcnow() - self.last_cache_update).total_seconds() < self.cache_ttl_seconds:
                return
            
            # Get active agents
            active_agents = await self.agent_registry.get_active_agents()
            
            for agent in active_agents:
                # Calculate current task count
                current_tasks = await self._get_agent_current_tasks(str(agent.id))
                
                # Parse resource usage
                resource_usage = agent.resource_usage or {}
                cpu_usage = float(resource_usage.get("cpu_percent", 0.0))
                memory_usage = float(resource_usage.get("memory_mb", 0.0))
                context_usage = float(agent.context_window_usage or 0.0)
                
                # Calculate performance metrics
                response_time = float(agent.average_response_time or 100.0)
                health_score = agent.health_score or 0.0
                
                # Get specialization match score (would be calculated based on requirements)
                specialization_score = 1.0  # Default, would be calculated dynamically
                
                metrics = AgentLoadMetrics(
                    agent_id=str(agent.id),
                    current_tasks=current_tasks,
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    context_usage=context_usage,
                    health_score=health_score,
                    response_time_ms=response_time,
                    success_rate=90.0,  # Would be calculated from historical data
                    last_task_completed=agent.last_active,
                    specialization_match_score=specialization_score
                )
                
                self.agent_performance_cache[str(agent.id)] = metrics
            
            self.last_cache_update = datetime.utcnow()
            
        except Exception as e:
            logger.error("Failed to update agent performance cache", error=str(e))
    
    async def _get_agent_current_tasks(self, agent_id: str) -> int:
        """Get current number of tasks assigned to an agent."""
        try:
            async with get_session() as db:
                result = await db.execute(
                    select(func.count(Task.id))
                    .where(
                        and_(
                            Task.assigned_agent_id == agent_id,
                            Task.status.in_([TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS])
                        )
                    )
                )
                return result.scalar() or 0
        except Exception as e:
            logger.error("Failed to get agent current tasks", agent_id=agent_id, error=str(e))
            return 0
    
    async def _get_available_agents(
        self,
        agent_requirements: List[AgentRequirement]
    ) -> List[Agent]:
        """Get agents that meet the requirements and are available."""
        try:
            # Get all active agents
            active_agents = await self.agent_registry.get_active_agents()
            
            # Filter by requirements
            available_agents = []
            
            for requirement in agent_requirements:
                matching_agents = [
                    agent for agent in active_agents
                    if (
                        agent.role == requirement.role.value and
                        self._agent_meets_requirements(agent, requirement) and
                        self._is_agent_available(agent)
                    )
                ]
                available_agents.extend(matching_agents)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_agents = []
            for agent in available_agents:
                if str(agent.id) not in seen:
                    seen.add(str(agent.id))
                    unique_agents.append(agent)
            
            return unique_agents
            
        except Exception as e:
            logger.error("Failed to get available agents", error=str(e))
            return []
    
    def _agent_meets_requirements(self, agent: Agent, requirement: AgentRequirement) -> bool:
        """Check if agent meets specific requirements."""
        try:
            # Check capabilities
            agent_capabilities = {cap.get("name", "") for cap in agent.capabilities}
            required_capabilities = set(requirement.required_capabilities)
            
            if not required_capabilities.issubset(agent_capabilities):
                return False
            
            # Check specializations (if any)
            if requirement.specialization:
                agent_specializations = {cap.get("specialization", "") for cap in agent.capabilities}
                required_specializations = set(requirement.specialization)
                
                if not required_specializations.intersection(agent_specializations):
                    return False
            
            return True
            
        except Exception as e:
            logger.error("Error checking agent requirements", agent_id=str(agent.id), error=str(e))
            return False
    
    def _is_agent_available(self, agent: Agent) -> bool:
        """Check if agent is available for new tasks."""
        try:
            agent_id = str(agent.id)
            
            # Check cache for performance metrics
            if agent_id in self.agent_performance_cache:
                metrics = self.agent_performance_cache[agent_id]
                
                # Check load thresholds
                if metrics.current_tasks >= self.max_tasks_per_agent:
                    return False
                if metrics.cpu_usage >= self.max_cpu_threshold:
                    return False
                if metrics.memory_usage >= self.max_memory_threshold:
                    return False
                if metrics.context_usage >= self.max_context_threshold:
                    return False
                if metrics.health_score < self.min_health_score:
                    return False
            
            return agent.status == AgentStatus.active
            
        except Exception as e:
            logger.error("Error checking agent availability", agent_id=str(agent.id), error=str(e))
            return False
    
    async def _execute_distribution_strategy(
        self,
        workflow_steps: List[WorkflowStep],
        available_agents: List[Agent],
        strategy: DistributionStrategy,
        urgency: TaskUrgency,
        execution_context: Dict[str, Any]
    ) -> List[TaskAssignment]:
        """Execute the specified distribution strategy."""
        
        if strategy == DistributionStrategy.ROUND_ROBIN:
            return await self._round_robin_distribution(workflow_steps, available_agents, urgency)
        elif strategy == DistributionStrategy.LEAST_LOADED:
            return await self._least_loaded_distribution(workflow_steps, available_agents, urgency)
        elif strategy == DistributionStrategy.CAPABILITY_MATCH:
            return await self._capability_match_distribution(workflow_steps, available_agents, urgency)
        elif strategy == DistributionStrategy.PERFORMANCE_BASED:
            return await self._performance_based_distribution(workflow_steps, available_agents, urgency)
        elif strategy == DistributionStrategy.HYBRID:
            return await self._hybrid_distribution(workflow_steps, available_agents, urgency, execution_context)
        else:
            # Default to hybrid
            return await self._hybrid_distribution(workflow_steps, available_agents, urgency, execution_context)
    
    async def _hybrid_distribution(
        self,
        workflow_steps: List[WorkflowStep],
        available_agents: List[Agent],
        urgency: TaskUrgency,
        execution_context: Dict[str, Any]
    ) -> List[TaskAssignment]:
        """Hybrid distribution strategy combining multiple factors."""
        assignments = []
        
        for step in workflow_steps:
            # Find suitable agents for this step
            if step.agent:
                # Single agent requirement
                suitable_agents = [
                    agent for agent in available_agents
                    if agent.role == step.agent.value
                ]
            else:
                # Any available agent (or use first agent requirement)
                suitable_agents = available_agents
            
            if not suitable_agents:
                continue
            
            # Calculate composite score for each agent
            best_agent = None
            best_score = -1.0
            
            for agent in suitable_agents:
                score = self._calculate_assignment_score(agent, step, urgency)
                if score > best_score:
                    best_score = score
                    best_agent = agent
            
            if best_agent:
                # Create assignment
                assignment = TaskAssignment(
                    task_id=step.step,
                    agent_id=str(best_agent.id),
                    assignment_score=best_score,
                    estimated_completion_time=datetime.utcnow() + timedelta(
                        minutes=step.timeout_minutes or 60
                    ),
                    assignment_reason="hybrid_strategy_optimal_match",
                    backup_agents=[
                        str(agent.id) for agent in suitable_agents[:3]
                        if str(agent.id) != str(best_agent.id)
                    ]
                )
                assignments.append(assignment)
        
        return assignments
    
    async def _round_robin_distribution(
        self,
        workflow_steps: List[WorkflowStep],
        available_agents: List[Agent],
        urgency: TaskUrgency
    ) -> List[TaskAssignment]:
        """Simple round-robin distribution."""
        assignments = []
        agent_index = 0
        
        for step in workflow_steps:
            if available_agents:
                agent = available_agents[agent_index % len(available_agents)]
                
                assignment = TaskAssignment(
                    task_id=step.step,
                    agent_id=str(agent.id),
                    assignment_score=0.5,  # Neutral score for round-robin
                    estimated_completion_time=datetime.utcnow() + timedelta(
                        minutes=step.timeout_minutes or 60
                    ),
                    assignment_reason="round_robin_distribution",
                    backup_agents=[]
                )
                assignments.append(assignment)
                agent_index += 1
        
        return assignments
    
    async def _least_loaded_distribution(
        self,
        workflow_steps: List[WorkflowStep],
        available_agents: List[Agent],
        urgency: TaskUrgency
    ) -> List[TaskAssignment]:
        """Distribute to least loaded agents."""
        assignments = []
        
        for step in workflow_steps:
            # Find least loaded agent
            best_agent = None
            lowest_load = float('inf')
            
            for agent in available_agents:
                agent_id = str(agent.id)
                if agent_id in self.agent_performance_cache:
                    metrics = self.agent_performance_cache[agent_id]
                    load_score = self._calculate_load_score(metrics)
                    
                    if load_score < lowest_load:
                        lowest_load = load_score
                        best_agent = agent
            
            if best_agent:
                assignment = TaskAssignment(
                    task_id=step.step,
                    agent_id=str(best_agent.id),
                    assignment_score=1.0 - (lowest_load / 100.0),  # Invert load for score
                    estimated_completion_time=datetime.utcnow() + timedelta(
                        minutes=step.timeout_minutes or 60
                    ),
                    assignment_reason="least_loaded_agent",
                    backup_agents=[]
                )
                assignments.append(assignment)
        
        return assignments
    
    async def _capability_match_distribution(
        self,
        workflow_steps: List[WorkflowStep],
        available_agents: List[Agent],
        urgency: TaskUrgency
    ) -> List[TaskAssignment]:
        """Distribute based on capability matching."""
        assignments = []
        
        for step in workflow_steps:
            # Find agent with best capability match
            best_agent = None
            best_match_score = 0.0
            
            for agent in available_agents:
                match_score = self._calculate_capability_match_score(agent, step)
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_agent = agent
            
            if best_agent:
                assignment = TaskAssignment(
                    task_id=step.step,
                    agent_id=str(best_agent.id),
                    assignment_score=best_match_score,
                    estimated_completion_time=datetime.utcnow() + timedelta(
                        minutes=step.timeout_minutes or 60
                    ),
                    assignment_reason="capability_match",
                    backup_agents=[]
                )
                assignments.append(assignment)
        
        return assignments
    
    async def _performance_based_distribution(
        self,
        workflow_steps: List[WorkflowStep],
        available_agents: List[Agent],
        urgency: TaskUrgency
    ) -> List[TaskAssignment]:
        """Distribute based on agent performance metrics."""
        assignments = []
        
        for step in workflow_steps:
            # Find highest performing agent
            best_agent = None
            best_performance = 0.0
            
            for agent in available_agents:
                agent_id = str(agent.id)
                if agent_id in self.agent_performance_cache:
                    metrics = self.agent_performance_cache[agent_id]
                    performance_score = (
                        metrics.health_score * 0.4 +
                        (metrics.success_rate / 100.0) * 0.4 +
                        (1.0 / max(metrics.response_time_ms, 1.0)) * 0.2
                    )
                    
                    if performance_score > best_performance:
                        best_performance = performance_score
                        best_agent = agent
            
            if best_agent:
                assignment = TaskAssignment(
                    task_id=step.step,
                    agent_id=str(best_agent.id),
                    assignment_score=best_performance,
                    estimated_completion_time=datetime.utcnow() + timedelta(
                        minutes=step.timeout_minutes or 60
                    ),
                    assignment_reason="performance_based",
                    backup_agents=[]
                )
                assignments.append(assignment)
        
        return assignments
    
    def _calculate_assignment_score(
        self,
        agent: Agent,
        step: WorkflowStep,
        urgency: TaskUrgency
    ) -> float:
        """Calculate composite assignment score for agent-task pair."""
        try:
            agent_id = str(agent.id)
            
            # Capability match score
            capability_score = self._calculate_capability_match_score(agent, step)
            
            # Load score (lower is better, so invert)
            load_score = 1.0
            if agent_id in self.agent_performance_cache:
                metrics = self.agent_performance_cache[agent_id]
                load_raw = self._calculate_load_score(metrics)
                load_score = max(0.0, 1.0 - (load_raw / 100.0))
            
            # Performance score
            performance_score = 0.5
            if agent_id in self.agent_performance_cache:
                metrics = self.agent_performance_cache[agent_id]
                performance_score = (
                    metrics.health_score * 0.5 +
                    (metrics.success_rate / 100.0) * 0.3 +
                    (1.0 / max(metrics.response_time_ms, 1.0)) * 100 * 0.2
                )
            
            # Availability score
            availability_score = 1.0 if self._is_agent_available(agent) else 0.0
            
            # Urgency multiplier
            urgency_multiplier = {
                TaskUrgency.LOW: 1.0,
                TaskUrgency.NORMAL: 1.1,
                TaskUrgency.HIGH: 1.3,
                TaskUrgency.CRITICAL: 1.5
            }.get(urgency, 1.0)
            
            # Composite score
            composite_score = (
                capability_score * self.capability_weight +
                load_score * self.load_weight +
                performance_score * self.performance_weight +
                availability_score * self.availability_weight
            ) * urgency_multiplier
            
            return min(1.0, max(0.0, composite_score))
            
        except Exception as e:
            logger.error("Error calculating assignment score", error=str(e))
            return 0.0
    
    def _calculate_capability_match_score(self, agent: Agent, step: WorkflowStep) -> float:
        """Calculate how well agent capabilities match step requirements."""
        try:
            # Basic role match
            if step.agent and agent.role != step.agent.value:
                return 0.0
            
            # If no specific requirements, return base score
            if not step.agent:
                return 0.7
            
            # Perfect role match
            return 1.0
            
        except Exception as e:
            logger.error("Error calculating capability match", error=str(e))
            return 0.0
    
    def _calculate_load_score(self, metrics: AgentLoadMetrics) -> float:
        """Calculate agent load score (0-100, lower is better)."""
        try:
            # Weighted load calculation
            task_load = (metrics.current_tasks / self.max_tasks_per_agent) * 100
            cpu_load = metrics.cpu_usage
            memory_load = metrics.memory_usage
            context_load = metrics.context_usage
            
            # Composite load score
            load_score = (
                task_load * 0.4 +
                cpu_load * 0.25 +
                memory_load * 0.2 +
                context_load * 0.15
            )
            
            return min(100.0, max(0.0, load_score))
            
        except Exception as e:
            logger.error("Error calculating load score", error=str(e))
            return 100.0  # Assume high load on error
    
    def _calculate_agent_availability(self, metrics: AgentLoadMetrics) -> str:
        """Calculate agent availability status."""
        try:
            load_score = self._calculate_load_score(metrics)
            
            if load_score < 30:
                return "high"
            elif load_score < 60:
                return "medium"
            elif load_score < 85:
                return "low"
            else:
                return "overloaded"
                
        except Exception as e:
            logger.error("Error calculating availability", error=str(e))
            return "unknown"
    
    async def _select_best_agent_for_task(
        self,
        requirement: AgentRequirement,
        available_agents: List[Agent],
        urgency: TaskUrgency
    ) -> Optional[Agent]:
        """Select the best agent for a specific task requirement."""
        if not available_agents:
            return None
        
        best_agent = None
        best_score = -1.0
        
        # Create a mock workflow step for scoring
        mock_step = WorkflowStep(
            step="temp",
            agent=requirement.role,
            task="temporary task for scoring"
        )
        
        for agent in available_agents:
            score = self._calculate_assignment_score(agent, mock_step, urgency)
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent
    
    def _calculate_optimization_metrics(
        self,
        assignments: List[TaskAssignment],
        available_agents: List[Agent]
    ) -> Dict[str, Any]:
        """Calculate optimization metrics for distribution result."""
        try:
            if not assignments:
                return {"error": "no_assignments"}
            
            # Load distribution metrics
            agent_task_counts = {}
            for assignment in assignments:
                agent_id = assignment.agent_id
                agent_task_counts[agent_id] = agent_task_counts.get(agent_id, 0) + 1
            
            load_distribution = list(agent_task_counts.values())
            load_balance_score = 1.0 - (
                (max(load_distribution) - min(load_distribution)) / max(max(load_distribution), 1)
            )
            
            # Assignment quality metrics
            avg_assignment_score = sum(a.assignment_score for a in assignments) / len(assignments)
            
            # Resource utilization
            utilized_agents = len(agent_task_counts)
            utilization_rate = utilized_agents / max(len(available_agents), 1)
            
            return {
                "load_balance_score": load_balance_score,
                "average_assignment_score": avg_assignment_score,
                "agent_utilization_rate": utilization_rate,
                "tasks_per_agent": load_distribution,
                "optimization_quality": (load_balance_score + avg_assignment_score + utilization_rate) / 3
            }
            
        except Exception as e:
            logger.error("Error calculating optimization metrics", error=str(e))
            return {"error": str(e)}
    
    def _update_distribution_stats(
        self,
        strategy: DistributionStrategy,
        distribution_time_ms: float,
        assignments_count: int
    ) -> None:
        """Update distribution statistics."""
        try:
            self.distribution_stats["total_distributions"] += 1
            self.distribution_stats["successful_assignments"] += assignments_count
            self.distribution_stats["strategy_usage"][strategy.value] += 1
            
            # Update average distribution time
            current_avg = self.distribution_stats["average_distribution_time_ms"]
            total_distributions = self.distribution_stats["total_distributions"]
            
            new_avg = (
                (current_avg * (total_distributions - 1) + distribution_time_ms) / total_distributions
            )
            self.distribution_stats["average_distribution_time_ms"] = new_avg
            
        except Exception as e:
            logger.error("Error updating distribution stats", error=str(e))
    
    def get_distribution_statistics(self) -> Dict[str, Any]:
        """Get current distribution statistics."""
        return {
            **self.distribution_stats,
            "cache_stats": {
                "cached_agents": len(self.agent_performance_cache),
                "cache_last_updated": self.last_cache_update.isoformat(),
                "cache_ttl_seconds": self.cache_ttl_seconds
            },
            "configuration": {
                "max_tasks_per_agent": self.max_tasks_per_agent,
                "max_cpu_threshold": self.max_cpu_threshold,
                "max_memory_threshold": self.max_memory_threshold,
                "max_context_threshold": self.max_context_threshold,
                "min_health_score": self.min_health_score
            }
        }