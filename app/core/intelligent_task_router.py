"""
Intelligent Task Routing Engine for LeanVibe Agent Hive 2.0

Provides advanced task routing algorithms that optimize agent selection
based on capabilities, performance history, workload balancing, and
workflow dependencies for maximum system efficiency.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum

import structlog
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import Session

from .database import get_session
from .capability_matcher import CapabilityMatcher, AgentPerformanceProfile, WorkloadMetrics
from ..models.agent import Agent, AgentStatus
from ..models.task import Task, TaskStatus, TaskPriority, TaskType
from ..models.workflow import Workflow, WorkflowStatus

logger = structlog.get_logger()


class RoutingStrategy(Enum):
    """Different routing strategies for task assignment."""
    CAPABILITY_FIRST = "capability_first"
    PERFORMANCE_FIRST = "performance_first"
    LOAD_BALANCED = "load_balanced"
    PRIORITY_AWARE = "priority_aware"
    ADAPTIVE = "adaptive"


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    CONSISTENT_HASHING = "consistent_hashing"


@dataclass
class TaskRoutingContext:
    """Context information for task routing decisions."""
    task_id: str
    task_type: str
    priority: TaskPriority
    required_capabilities: List[str]
    estimated_effort: Optional[int]
    due_date: Optional[datetime]
    dependencies: List[str]
    workflow_id: Optional[str]
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentSuitabilityScore:
    """Comprehensive suitability score for an agent."""
    agent_id: str
    total_score: float
    capability_score: float
    performance_score: float
    availability_score: float
    priority_alignment_score: float
    specialization_bonus: float
    workload_penalty: float
    score_breakdown: Dict[str, float]
    confidence_level: float


@dataclass
class TaskReassignment:
    """Represents a task reassignment for load balancing."""
    task_id: str
    from_agent_id: str
    to_agent_id: str
    reason: str
    expected_improvement: float


@dataclass
class TaskExecution:
    """Represents a task execution plan with dependencies resolved."""
    task_id: str
    agent_id: str
    execution_order: int
    estimated_start_time: datetime
    dependencies_satisfied: bool
    blocking_tasks: List[str]


class IntelligentTaskRouter:
    """
    Advanced task routing engine with intelligent agent selection.
    
    Provides sophisticated algorithms for optimal task assignment considering
    agent capabilities, performance history, current workload, and workflow
    dependencies to maximize system throughput and task completion rates.
    """
    
    def __init__(self):
        self.capability_matcher = CapabilityMatcher()
        self.routing_history: Dict[str, List[str]] = {}  # task_id -> agent_id history
        self.load_balancer_state: Dict[str, Any] = {}
        self.performance_thresholds = {
            "min_success_rate": 0.7,
            "max_workload_factor": 0.85,
            "min_capability_score": 0.6,
            "response_time_sla": 500  # milliseconds
        }
    
    async def route_task(
        self,
        task: TaskRoutingContext,
        available_agents: List[str],
        strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE,
        exclude_agents: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Route a task to the most suitable agent using intelligent algorithms.
        
        Args:
            task: Task routing context with requirements
            available_agents: List of available agent IDs
            strategy: Routing strategy to use
            exclude_agents: Agents to exclude from consideration
            
        Returns:
            Selected agent ID or None if no suitable agent found
        """
        start_time = datetime.utcnow()
        
        try:
            # Filter out excluded agents
            if exclude_agents:
                candidate_agents = [aid for aid in available_agents if aid not in exclude_agents]
            else:
                candidate_agents = available_agents.copy()
            
            if not candidate_agents:
                logger.warning("No candidate agents available", task_id=task.task_id)
                return None
            
            # Get agent suitability scores
            suitability_scores = await self._calculate_agent_suitability_scores(
                task, candidate_agents, strategy
            )
            
            if not suitability_scores:
                logger.warning("No suitable agents found", task_id=task.task_id)
                return None
            
            # Select best agent based on strategy
            selected_agent = await self._select_optimal_agent(
                suitability_scores, strategy, task
            )
            
            if selected_agent:
                # Record routing decision
                await self._record_routing_decision(task.task_id, selected_agent, suitability_scores)
                
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                logger.info(
                    "Task routed successfully",
                    task_id=task.task_id,
                    selected_agent=selected_agent,
                    strategy=strategy.value,
                    response_time_ms=response_time,
                    total_candidates=len(candidate_agents)
                )
            
            return selected_agent
            
        except Exception as e:
            logger.error("Error routing task", task_id=task.task_id, error=str(e))
            return None
    
    async def calculate_agent_suitability(
        self,
        agent_id: str,
        task: TaskRoutingContext
    ) -> Optional[AgentSuitabilityScore]:
        """
        Calculate comprehensive suitability score for a specific agent-task pair.
        
        Args:
            agent_id: Agent identifier
            task: Task routing context
            
        Returns:
            Detailed suitability score or None if agent is unavailable
        """
        try:
            async with get_session() as db_session:
                # Get agent details
                agent = await db_session.get(Agent, agent_id)
                if not agent or not agent.is_available_for_task():
                    return None
                
                # Calculate capability matching score
                capability_score, capability_breakdown = await self.capability_matcher.calculate_composite_suitability_score(
                    agent_id=agent_id,
                    requirements=task.required_capabilities,
                    agent_capabilities=agent.capabilities or [],
                    task_type=task.task_type,
                    priority=task.priority,
                    consider_workload=True
                )
                
                # Get performance score
                performance_score = await self.capability_matcher.calculate_performance_score(
                    agent_id, task.task_type
                )
                
                # Get workload factor
                workload_factor = await self.capability_matcher.get_workload_factor(agent_id)
                availability_score = 1.0 - workload_factor
                
                # Calculate priority alignment
                priority_alignment = await self._calculate_priority_alignment(agent_id, task.priority)
                
                # Calculate specialization bonus
                specialization_bonus = await self._calculate_specialization_bonus(
                    agent_id, task.task_type, task.required_capabilities
                )
                
                # Apply workload penalty for overloaded agents
                workload_penalty = max(0.0, workload_factor - 0.7) * 0.5
                
                # Calculate composite score
                weights = self._get_scoring_weights(task.priority)
                total_score = (
                    capability_score * weights["capability"] +
                    performance_score * weights["performance"] +
                    availability_score * weights["availability"] +
                    priority_alignment * weights["priority"] +
                    specialization_bonus * weights["specialization"] -
                    workload_penalty * weights["workload_penalty"]
                )
                
                # Calculate confidence level
                confidence_level = self._calculate_confidence_level(
                    capability_score, performance_score, availability_score
                )
                
                score_breakdown = {
                    "capability": capability_score,
                    "performance": performance_score,
                    "availability": availability_score,
                    "priority_alignment": priority_alignment,
                    "specialization_bonus": specialization_bonus,
                    "workload_penalty": workload_penalty,
                    **capability_breakdown
                }
                
                return AgentSuitabilityScore(
                    agent_id=agent_id,
                    total_score=max(0.0, min(1.0, total_score)),
                    capability_score=capability_score,
                    performance_score=performance_score,
                    availability_score=availability_score,
                    priority_alignment_score=priority_alignment,
                    specialization_bonus=specialization_bonus,
                    workload_penalty=workload_penalty,
                    score_breakdown=score_breakdown,
                    confidence_level=confidence_level
                )
                
        except Exception as e:
            logger.error("Error calculating agent suitability", agent_id=agent_id, error=str(e))
            return None
    
    async def update_agent_performance(
        self,
        agent_id: str,
        task_result: Dict[str, Any]
    ) -> None:
        """
        Update agent performance metrics based on task completion.
        
        Args:
            agent_id: Agent identifier
            task_result: Task completion result with metrics
        """
        try:
            # Clear cached performance data for this agent
            self.capability_matcher.clear_cache(agent_id)
            
            # Extract performance metrics
            success = task_result.get("success", False)
            completion_time = task_result.get("completion_time", 0)
            task_type = task_result.get("task_type", "general")
            
            logger.info(
                "Agent performance updated",
                agent_id=agent_id,
                success=success,
                completion_time=completion_time,
                task_type=task_type
            )
            
        except Exception as e:
            logger.error("Error updating agent performance", agent_id=agent_id, error=str(e))
    
    async def rebalance_workload(
        self,
        algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.LEAST_LOADED
    ) -> List[TaskReassignment]:
        """
        Rebalance workload across agents by reassigning tasks.
        
        Args:
            algorithm: Load balancing algorithm to use
            
        Returns:
            List of recommended task reassignments
        """
        reassignments = []
        
        try:
            async with get_session() as db_session:
                # Get all active agents
                agents_query = select(Agent).where(Agent.status == AgentStatus.ACTIVE)
                agents = (await db_session.execute(agents_query)).scalars().all()
                
                if len(agents) < 2:
                    return reassignments  # Need at least 2 agents for rebalancing
                
                # Calculate workload metrics for each agent
                agent_workloads = {}
                for agent in agents:
                    workload = await self.capability_matcher.get_workload_factor(str(agent.id))
                    agent_workloads[str(agent.id)] = workload
                
                # Identify overloaded and underloaded agents
                avg_workload = sum(agent_workloads.values()) / len(agent_workloads)
                overloaded_threshold = avg_workload + 0.2
                underloaded_threshold = avg_workload - 0.2
                
                overloaded_agents = [
                    aid for aid, workload in agent_workloads.items()
                    if workload > overloaded_threshold
                ]
                underloaded_agents = [
                    aid for aid, workload in agent_workloads.items()
                    if workload < underloaded_threshold
                ]
                
                # Find reassignment opportunities
                for overloaded_agent in overloaded_agents:
                    reassignments.extend(
                        await self._find_reassignment_candidates(
                            overloaded_agent, underloaded_agents, algorithm
                        )
                    )
                
                logger.info(
                    "Workload rebalancing completed",
                    algorithm=algorithm.value,
                    reassignments_count=len(reassignments),
                    overloaded_agents=len(overloaded_agents),
                    underloaded_agents=len(underloaded_agents)
                )
                
        except Exception as e:
            logger.error("Error rebalancing workload", algorithm=algorithm.value, error=str(e))
        
        return reassignments
    
    async def resolve_task_dependencies(
        self,
        tasks: List[TaskRoutingContext]
    ) -> List[TaskExecution]:
        """
        Resolve task dependencies and create optimal execution plan.
        
        Args:
            tasks: List of tasks with dependency information
            
        Returns:
            Ordered execution plan with dependency resolution
        """
        execution_plan = []
        
        try:
            # Build dependency graph
            dependency_graph = {}
            task_lookup = {task.task_id: task for task in tasks}
            
            for task in tasks:
                dependency_graph[task.task_id] = {
                    "task": task,
                    "dependencies": task.dependencies,
                    "dependents": [],
                    "resolved": False,
                    "execution_order": -1
                }
            
            # Build reverse dependencies (dependents)
            for task_id, node in dependency_graph.items():
                for dep_id in node["dependencies"]:
                    if dep_id in dependency_graph:
                        dependency_graph[dep_id]["dependents"].append(task_id)
            
            # Topological sort with agent assignment
            execution_order = 0
            available_agents = await self._get_available_agents()
            
            while True:
                # Find tasks with no unresolved dependencies
                ready_tasks = [
                    task_id for task_id, node in dependency_graph.items()
                    if not node["resolved"] and all(
                        dependency_graph.get(dep_id, {}).get("resolved", True)
                        for dep_id in node["dependencies"]
                    )
                ]
                
                if not ready_tasks:
                    # Check for circular dependencies
                    unresolved_tasks = [
                        task_id for task_id, node in dependency_graph.items()
                        if not node["resolved"]
                    ]
                    if unresolved_tasks:
                        logger.warning("Circular dependencies detected", tasks=unresolved_tasks)
                    break
                
                # Assign agents to ready tasks
                for task_id in ready_tasks:
                    node = dependency_graph[task_id]
                    task = node["task"]
                    
                    # Route task to best available agent
                    selected_agent = await self.route_task(
                        task, available_agents, RoutingStrategy.ADAPTIVE
                    )
                    
                    if selected_agent:
                        # Calculate estimated start time
                        estimated_start = await self._calculate_estimated_start_time(
                            selected_agent, execution_order
                        )
                        
                        execution_plan.append(TaskExecution(
                            task_id=task_id,
                            agent_id=selected_agent,
                            execution_order=execution_order,
                            estimated_start_time=estimated_start,
                            dependencies_satisfied=True,
                            blocking_tasks=node["dependents"]
                        ))
                        
                        node["resolved"] = True
                        node["execution_order"] = execution_order
                        execution_order += 1
                
                if not ready_tasks:
                    break
            
            # Sort execution plan by order
            execution_plan.sort(key=lambda x: x.execution_order)
            
            logger.info(
                "Task dependencies resolved",
                total_tasks=len(tasks),
                execution_plan_length=len(execution_plan),
                max_execution_order=execution_order - 1
            )
            
        except Exception as e:
            logger.error("Error resolving task dependencies", error=str(e))
        
        return execution_plan
    
    async def _calculate_agent_suitability_scores(
        self,
        task: TaskRoutingContext,
        candidate_agents: List[str],
        strategy: RoutingStrategy
    ) -> List[AgentSuitabilityScore]:
        """Calculate suitability scores for all candidate agents."""
        scores = []
        
        for agent_id in candidate_agents:
            score = await self.calculate_agent_suitability(agent_id, task)
            if score and score.total_score >= self.performance_thresholds["min_capability_score"]:
                scores.append(score)
        
        # Filter based on strategy-specific criteria
        if strategy == RoutingStrategy.PERFORMANCE_FIRST:
            scores = [s for s in scores if s.performance_score >= self.performance_thresholds["min_success_rate"]]
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            scores = [s for s in scores if s.workload_penalty < 0.3]
        
        return sorted(scores, key=lambda x: x.total_score, reverse=True)
    
    async def _select_optimal_agent(
        self,
        suitability_scores: List[AgentSuitabilityScore],
        strategy: RoutingStrategy,
        task: TaskRoutingContext
    ) -> Optional[str]:
        """Select the optimal agent based on strategy and scores."""
        if not suitability_scores:
            return None
        
        if strategy == RoutingStrategy.CAPABILITY_FIRST:
            # Select highest capability score
            best_score = max(suitability_scores, key=lambda x: x.capability_score)
            return best_score.agent_id
        
        elif strategy == RoutingStrategy.PERFORMANCE_FIRST:
            # Select highest performance score
            best_score = max(suitability_scores, key=lambda x: x.performance_score)
            return best_score.agent_id
        
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            # Select lowest workload penalty
            best_score = min(suitability_scores, key=lambda x: x.workload_penalty)
            return best_score.agent_id
        
        elif strategy == RoutingStrategy.PRIORITY_AWARE:
            # Weight by priority alignment for high-priority tasks
            if task.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]:
                best_score = max(suitability_scores, key=lambda x: x.priority_alignment_score)
                return best_score.agent_id
            else:
                # Use total score for normal priority
                return suitability_scores[0].agent_id
        
        else:  # ADAPTIVE strategy
            # Use total composite score
            return suitability_scores[0].agent_id
    
    async def _record_routing_decision(
        self,
        task_id: str,
        selected_agent: str,
        suitability_scores: List[AgentSuitabilityScore]
    ) -> None:
        """Record routing decision for analytics and learning."""
        if task_id not in self.routing_history:
            self.routing_history[task_id] = []
        
        self.routing_history[task_id].append(selected_agent)
        
        # Log decision with scores for analysis
        logger.debug(
            "Routing decision recorded",
            task_id=task_id,
            selected_agent=selected_agent,
            total_candidates=len(suitability_scores),
            selected_score=next(
                (s.total_score for s in suitability_scores if s.agent_id == selected_agent),
                0.0
            )
        )
    
    async def _calculate_priority_alignment(self, agent_id: str, priority: TaskPriority) -> float:
        """Calculate how well an agent aligns with task priority requirements."""
        try:
            async with get_session() as db_session:
                # Get agent's historical performance for this priority
                completed_query = select(func.count(Task.id)).where(
                    and_(
                        Task.assigned_agent_id == agent_id,
                        Task.priority == priority,
                        Task.status == TaskStatus.COMPLETED
                    )
                )
                total_query = select(func.count(Task.id)).where(
                    and_(
                        Task.assigned_agent_id == agent_id,
                        Task.priority == priority,
                        Task.status.in_([TaskStatus.COMPLETED, TaskStatus.FAILED])
                    )
                )
                
                completed = (await db_session.execute(completed_query)).scalar() or 0
                total = (await db_session.execute(total_query)).scalar() or 0
                
                if total > 0:
                    return completed / total
                else:
                    # No history - return priority-based default
                    if priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]:
                        return 0.6  # Conservative for high priority
                    else:
                        return 0.8  # More optimistic for lower priority
                        
        except Exception as e:
            logger.error("Error calculating priority alignment", agent_id=agent_id, error=str(e))
            return 0.5
    
    async def _calculate_specialization_bonus(
        self,
        agent_id: str,
        task_type: str,
        required_capabilities: List[str]
    ) -> float:
        """Calculate bonus for agent specialization in task area."""
        try:
            async with get_session() as db_session:
                agent = await db_session.get(Agent, agent_id)
                if not agent or not agent.capabilities:
                    return 0.0
                
                # Check for exact specialization matches
                specialization_score = 0.0
                for capability in agent.capabilities:
                    cap_areas = capability.get("specialization_areas", [])
                    confidence = capability.get("confidence_level", 0.0)
                    
                    # Check task type alignment
                    if task_type.lower() in [area.lower() for area in cap_areas]:
                        specialization_score += confidence * 0.2
                    
                    # Check required capabilities alignment
                    for req_cap in required_capabilities:
                        if req_cap.lower() in [area.lower() for area in cap_areas]:
                            specialization_score += confidence * 0.1
                
                return min(0.3, specialization_score)  # Cap at 30% bonus
                
        except Exception as e:
            logger.error("Error calculating specialization bonus", agent_id=agent_id, error=str(e))
            return 0.0
    
    def _get_scoring_weights(self, priority: TaskPriority) -> Dict[str, float]:
        """Get scoring weights based on task priority."""
        if priority == TaskPriority.CRITICAL:
            return {
                "capability": 0.3,
                "performance": 0.4,
                "availability": 0.2,
                "priority": 0.05,
                "specialization": 0.03,
                "workload_penalty": 0.02
            }
        elif priority == TaskPriority.HIGH:
            return {
                "capability": 0.35,
                "performance": 0.3,
                "availability": 0.25,
                "priority": 0.05,
                "specialization": 0.03,
                "workload_penalty": 0.02
            }
        else:  # MEDIUM, LOW
            return {
                "capability": 0.4,
                "performance": 0.25,
                "availability": 0.25,
                "priority": 0.03,
                "specialization": 0.05,
                "workload_penalty": 0.02
            }
    
    def _calculate_confidence_level(
        self,
        capability_score: float,
        performance_score: float,
        availability_score: float
    ) -> float:
        """Calculate confidence level in the routing decision."""
        # Higher scores = higher confidence
        avg_score = (capability_score + performance_score + availability_score) / 3
        
        # Boost confidence if all scores are high
        if all(score > 0.8 for score in [capability_score, performance_score, availability_score]):
            return min(1.0, avg_score + 0.1)
        
        # Penalize if any score is very low
        if any(score < 0.3 for score in [capability_score, performance_score, availability_score]):
            return max(0.0, avg_score - 0.2)
        
        return avg_score
    
    async def _find_reassignment_candidates(
        self,
        overloaded_agent: str,
        underloaded_agents: List[str],
        algorithm: LoadBalancingAlgorithm
    ) -> List[TaskReassignment]:
        """Find task reassignment candidates for load balancing."""
        reassignments = []
        
        try:
            async with get_session() as db_session:
                # Get reassignable tasks from overloaded agent
                tasks_query = select(Task).where(
                    and_(
                        Task.assigned_agent_id == overloaded_agent,
                        Task.status.in_([TaskStatus.ASSIGNED, TaskStatus.PENDING]),
                        Task.priority.in_([TaskPriority.LOW, TaskPriority.MEDIUM])
                    )
                ).limit(3)  # Limit reassignments per cycle
                
                tasks = (await db_session.execute(tasks_query)).scalars().all()
                
                for task in tasks:
                    # Find best underloaded agent for this task
                    task_context = TaskRoutingContext(
                        task_id=str(task.id),
                        task_type=task.task_type.value if task.task_type else "general",
                        priority=task.priority,
                        required_capabilities=task.required_capabilities or [],
                        estimated_effort=task.estimated_effort,
                        due_date=task.due_date,
                        dependencies=task.dependencies or [],
                        workflow_id=None
                    )
                    
                    best_agent = await self.route_task(
                        task_context, underloaded_agents, RoutingStrategy.LOAD_BALANCED
                    )
                    
                    if best_agent and best_agent != overloaded_agent:
                        expected_improvement = await self._calculate_reassignment_benefit(
                            str(task.id), overloaded_agent, best_agent
                        )
                        
                        if expected_improvement > 0.1:  # Minimum improvement threshold
                            reassignments.append(TaskReassignment(
                                task_id=str(task.id),
                                from_agent_id=overloaded_agent,
                                to_agent_id=best_agent,
                                reason=f"Load balancing: {algorithm.value}",
                                expected_improvement=expected_improvement
                            ))
                            
        except Exception as e:
            logger.error("Error finding reassignment candidates", agent_id=overloaded_agent, error=str(e))
        
        return reassignments
    
    async def _calculate_reassignment_benefit(
        self,
        task_id: str,
        from_agent: str,
        to_agent: str
    ) -> float:
        """Calculate expected benefit of task reassignment."""
        try:
            # Get workload factors
            from_workload = await self.capability_matcher.get_workload_factor(from_agent)
            to_workload = await self.capability_matcher.get_workload_factor(to_agent)
            
            # Simple benefit calculation based on workload difference
            workload_benefit = from_workload - to_workload
            
            # Factor in capability alignment (placeholder)
            capability_benefit = 0.0  # Could be enhanced with capability matching
            
            return max(0.0, workload_benefit + capability_benefit)
            
        except Exception as e:
            logger.error("Error calculating reassignment benefit", task_id=task_id, error=str(e))
            return 0.0
    
    async def _get_available_agents(self) -> List[str]:
        """Get list of currently available agent IDs."""
        try:
            async with get_session() as db_session:
                agents_query = select(Agent.id).where(Agent.status == AgentStatus.ACTIVE)
                agents = (await db_session.execute(agents_query)).scalars().all()
                return [str(agent_id) for agent_id in agents]
                
        except Exception as e:
            logger.error("Error getting available agents", error=str(e))
            return []
    
    async def _calculate_estimated_start_time(
        self,
        agent_id: str,
        execution_order: int
    ) -> datetime:
        """Calculate estimated start time for a task."""
        base_time = datetime.utcnow()
        
        # Add buffer time based on execution order
        buffer_minutes = execution_order * 30  # 30 minutes per task in queue
        
        # Factor in agent workload
        workload_factor = await self.capability_matcher.get_workload_factor(agent_id)
        workload_delay = workload_factor * 60  # Up to 60 minutes delay for full load
        
        total_delay_minutes = buffer_minutes + workload_delay
        return base_time + timedelta(minutes=total_delay_minutes)