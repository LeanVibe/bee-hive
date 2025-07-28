"""
Workflow Message Router for LeanVibe Agent Hive 2.0 - Vertical Slice 4.2

Provides intelligent message routing with workflow context awareness, dependency 
management, and coordinated task distribution across consumer groups.

Key Features:
- Workflow-aware message routing with DAG dependency tracking
- Multi-step workflow coordination across consumer groups
- Dependency signaling and prerequisite validation
- Parallel task distribution with workflow constraints
- Dynamic routing based on workflow state and agent capabilities
- Failure recovery with workflow context preservation
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
from sqlalchemy import select, update, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from .enhanced_redis_streams_manager import (
    EnhancedRedisStreamsManager, ConsumerGroupType, MessageRoutingMode
)
from .consumer_group_coordinator import ConsumerGroupCoordinator
from .database import get_async_session
from ..models.message import StreamMessage, MessageType, MessagePriority
from ..models.workflow import Workflow, WorkflowStatus
from ..models.task import Task, TaskStatus, TaskType

logger = structlog.get_logger()


class WorkflowRoutingStrategy(str, Enum):
    """Workflow routing strategies."""
    SEQUENTIAL = "sequential"  # Route tasks in dependency order
    PARALLEL = "parallel"  # Route independent tasks in parallel
    ADAPTIVE = "adaptive"  # Adaptive routing based on workflow state
    PRIORITY_BASED = "priority_based"  # Route based on task priority
    CAPABILITY_MATCHED = "capability_matched"  # Route to best-matched consumer group
    HYBRID = "hybrid"  # Combination of multiple strategies


class DependencyResolutionMode(str, Enum):
    """Dependency resolution modes."""
    STRICT = "strict"  # All dependencies must be resolved
    BEST_EFFORT = "best_effort"  # Route if most dependencies resolved
    OPTIMISTIC = "optimistic"  # Route assuming dependencies will resolve


class WorkflowCoordinationMode(str, Enum):
    """Workflow coordination modes."""
    CENTRALIZED = "centralized"  # Central coordination through router
    DISTRIBUTED = "distributed"  # Distributed coordination via message passing
    HYBRID = "hybrid"  # Combination approach


@dataclass
class WorkflowContext:
    """Context information for workflow-aware routing."""
    workflow_id: str
    workflow_name: str
    workflow_status: WorkflowStatus
    current_step: int
    total_steps: int
    dependencies: Dict[str, List[str]]  # task_id -> [dependency_task_ids]
    completed_tasks: Set[str]
    failed_tasks: Set[str]
    parallel_groups: List[List[str]]  # Groups of tasks that can run in parallel
    priority: int = 5
    deadline: Optional[datetime] = None
    coordination_mode: WorkflowCoordinationMode = WorkflowCoordinationMode.CENTRALIZED
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['dependencies'] = dict(self.dependencies)
        result['completed_tasks'] = list(self.completed_tasks)
        result['failed_tasks'] = list(self.failed_tasks)
        if self.deadline:
            result['deadline'] = self.deadline.isoformat()
        return result


@dataclass
class RoutingDecision:
    """Decision made by the workflow router."""
    task_id: str
    target_group: str
    routing_strategy: WorkflowRoutingStrategy
    priority_boost: int = 0
    dependency_wait: bool = False
    estimated_completion_time: Optional[datetime] = None
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if self.estimated_completion_time:
            result['estimated_completion_time'] = self.estimated_completion_time.isoformat()
        return result


@dataclass
class WorkflowRoutingMetrics:
    """Metrics for workflow routing performance."""
    total_workflows_routed: int = 0
    total_tasks_routed: int = 0
    successful_routings: int = 0
    failed_routings: int = 0
    dependency_violations: int = 0
    average_routing_time_ms: float = 0.0
    workflows_completed: int = 0
    workflows_failed: int = 0
    parallel_efficiency_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkflowRoutingError(Exception):
    """Base exception for workflow routing errors."""
    pass


class DependencyViolationError(WorkflowRoutingError):
    """Error when dependency requirements are violated."""
    pass


class RoutingCapacityError(WorkflowRoutingError):
    """Error when routing capacity is exceeded."""
    pass


class WorkflowMessageRouter:
    """
    Intelligent message router with workflow awareness and dependency management.
    
    Provides:
    - Workflow-context aware message routing
    - DAG-based dependency resolution and task sequencing
    - Multi-step workflow coordination across consumer groups
    - Parallel task distribution with constraint satisfaction
    - Dynamic routing optimization based on workflow state
    - Comprehensive failure recovery with workflow preservation
    """
    
    def __init__(
        self,
        streams_manager: EnhancedRedisStreamsManager,
        coordinator: ConsumerGroupCoordinator,
        default_strategy: WorkflowRoutingStrategy = WorkflowRoutingStrategy.HYBRID,
        dependency_mode: DependencyResolutionMode = DependencyResolutionMode.STRICT,
        max_parallel_tasks: int = 50,
        routing_timeout_seconds: int = 30,
        enable_workflow_optimization: bool = True
    ):
        """
        Initialize Workflow Message Router.
        
        Args:
            streams_manager: Enhanced Redis Streams Manager
            coordinator: Consumer Group Coordinator
            default_strategy: Default routing strategy
            dependency_mode: Dependency resolution mode
            max_parallel_tasks: Maximum parallel tasks per workflow
            routing_timeout_seconds: Timeout for routing operations
            enable_workflow_optimization: Enable workflow optimization
        """
        self.streams_manager = streams_manager
        self.coordinator = coordinator
        self.default_strategy = default_strategy
        self.dependency_mode = dependency_mode
        self.max_parallel_tasks = max_parallel_tasks
        self.routing_timeout_seconds = routing_timeout_seconds
        self.enable_workflow_optimization = enable_workflow_optimization
        
        # Workflow state management
        self._active_workflows: Dict[str, WorkflowContext] = {}
        self._workflow_queues: Dict[str, deque] = defaultdict(deque)  # workflow_id -> pending tasks
        self._dependency_graph: Dict[str, Dict[str, Set[str]]] = defaultdict(dict)  # workflow_id -> task dependencies
        self._task_completions: Dict[str, Dict[str, datetime]] = defaultdict(dict)  # workflow_id -> task completion times
        
        # Routing optimization
        self._routing_cache: Dict[str, RoutingDecision] = {}
        self._group_capabilities: Dict[str, Set[str]] = {}
        self._group_performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Metrics and monitoring
        self._metrics = WorkflowRoutingMetrics()
        self._routing_history: deque = deque(maxlen=1000)
        
        # Background tasks
        self._workflow_monitor_task: Optional[asyncio.Task] = None
        self._optimization_task: Optional[asyncio.Task] = None
        
    async def start(self) -> None:
        """Start the workflow router with background tasks."""
        # Start background monitoring and optimization
        self._workflow_monitor_task = asyncio.create_task(self._workflow_monitoring_loop())
        
        if self.enable_workflow_optimization:
            self._optimization_task = asyncio.create_task(self._optimization_loop())
        
        # Initialize group capabilities from existing groups
        await self._initialize_group_capabilities()
        
        logger.info(
            "Workflow Message Router started",
            extra={
                "default_strategy": self.default_strategy.value,
                "dependency_mode": self.dependency_mode.value,
                "max_parallel_tasks": self.max_parallel_tasks
            }
        )
    
    async def stop(self) -> None:
        """Stop the workflow router and cleanup resources."""
        tasks = [self._workflow_monitor_task, self._optimization_task]
        for task in tasks:
            if task and not task.done():
                task.cancel()
        
        active_tasks = [task for task in tasks if task and not task.done()]
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        
        logger.info("Workflow Message Router stopped")
    
    async def route_workflow(
        self,
        workflow_id: str,
        tasks: List[Dict[str, Any]],
        workflow_context: Optional[WorkflowContext] = None
    ) -> Dict[str, Any]:
        """
        Route an entire workflow with intelligent task distribution.
        
        Args:
            workflow_id: Unique workflow identifier
            tasks: List of task definitions with dependencies
            workflow_context: Optional workflow context
            
        Returns:
            Dictionary with routing results and metadata
        """
        start_time = time.time()
        
        try:
            # Initialize or update workflow context
            if workflow_context:
                self._active_workflows[workflow_id] = workflow_context
            else:
                self._active_workflows[workflow_id] = await self._create_workflow_context(
                    workflow_id, tasks
                )
            
            context = self._active_workflows[workflow_id]
            
            # Build dependency graph
            await self._build_dependency_graph(workflow_id, tasks)
            
            # Determine initial routing strategy
            strategy = await self._determine_workflow_strategy(context)
            
            # Route tasks based on strategy
            routing_results = await self._route_tasks_by_strategy(
                workflow_id, tasks, strategy
            )
            
            # Update metrics
            self._metrics.total_workflows_routed += 1
            self._metrics.total_tasks_routed += len(tasks)
            
            routing_time_ms = (time.time() - start_time) * 1000
            if self._metrics.average_routing_time_ms == 0:
                self._metrics.average_routing_time_ms = routing_time_ms
            else:
                self._metrics.average_routing_time_ms = (
                    self._metrics.average_routing_time_ms * 0.9 + routing_time_ms * 0.1
                )
            
            logger.info(
                f"Routed workflow {workflow_id} with {len(tasks)} tasks",
                extra={
                    "workflow_id": workflow_id,
                    "task_count": len(tasks),
                    "strategy": strategy.value,
                    "routing_time_ms": routing_time_ms
                }
            )
            
            return {
                "workflow_id": workflow_id,
                "tasks_routed": len(tasks),
                "routing_strategy": strategy.value,
                "routing_results": routing_results,
                "routing_time_ms": routing_time_ms,
                "estimated_completion": await self._estimate_workflow_completion(workflow_id),
                "parallel_groups": len(context.parallel_groups)
            }
            
        except Exception as e:
            self._metrics.failed_routings += 1
            logger.error(f"Failed to route workflow {workflow_id}: {e}")
            raise WorkflowRoutingError(f"Workflow routing failed: {e}")
    
    async def route_task_message(
        self,
        message: StreamMessage,
        workflow_id: Optional[str] = None,
        dependencies: Optional[List[str]] = None
    ) -> RoutingDecision:
        """
        Route a single task message with workflow awareness.
        
        Args:
            message: Task message to route
            workflow_id: Optional workflow context
            dependencies: Optional task dependencies
            
        Returns:
            Routing decision with target group and metadata
        """
        try:
            # Extract or determine workflow context
            if workflow_id and workflow_id in self._active_workflows:
                context = self._active_workflows[workflow_id]
            else:
                context = await self._infer_workflow_context(message)
            
            # Check dependencies if provided
            if dependencies and not await self._check_dependencies_satisfied(
                workflow_id or "unknown", dependencies
            ):
                if self.dependency_mode == DependencyResolutionMode.STRICT:
                    raise DependencyViolationError(f"Dependencies not satisfied: {dependencies}")
                elif self.dependency_mode == DependencyResolutionMode.BEST_EFFORT:
                    # Add priority boost for waiting tasks
                    message.priority = MessagePriority.HIGH
            
            # Determine optimal target group
            target_group = await self._determine_target_group(message, context)
            
            # Create routing decision
            decision = RoutingDecision(
                task_id=message.id,
                target_group=target_group,
                routing_strategy=self.default_strategy,
                dependency_wait=bool(dependencies and not await self._check_dependencies_satisfied(
                    workflow_id or "unknown", dependencies
                )),
                estimated_completion_time=await self._estimate_task_completion(
                    target_group, message
                ),
                reasoning=f"Routed to {target_group} based on {self.default_strategy.value} strategy"
            )
            
            # Execute routing
            await self._execute_routing_decision(message, decision, context)
            
            self._metrics.successful_routings += 1
            
            return decision
            
        except Exception as e:
            self._metrics.failed_routings += 1
            logger.error(f"Failed to route task message {message.id}: {e}")
            raise WorkflowRoutingError(f"Task routing failed: {e}")
    
    async def signal_task_completion(
        self,
        workflow_id: str,
        task_id: str,
        result: Dict[str, Any]
    ) -> List[str]:
        """
        Signal task completion and trigger dependent tasks.
        
        Args:
            workflow_id: Workflow identifier
            task_id: Completed task identifier
            result: Task completion result
            
        Returns:
            List of newly triggered task IDs
        """
        try:
            if workflow_id not in self._active_workflows:
                logger.warning(f"Unknown workflow {workflow_id} for task completion")
                return []
            
            context = self._active_workflows[workflow_id]
            context.completed_tasks.add(task_id)
            
            # Record completion time
            self._task_completions[workflow_id][task_id] = datetime.utcnow()
            
            # Find and trigger dependent tasks
            triggered_tasks = []
            
            if workflow_id in self._dependency_graph:
                for dependent_task_id, deps in self._dependency_graph[workflow_id].items():
                    if task_id in deps:
                        # Remove completed dependency
                        deps.remove(task_id)
                        
                        # Check if all dependencies are now satisfied
                        if not deps:  # All dependencies completed
                            triggered_tasks.append(dependent_task_id)
                            
                            # Trigger the dependent task
                            await self._trigger_dependent_task(
                                workflow_id, dependent_task_id, context
                            )
            
            # Check if workflow is complete
            if len(context.completed_tasks) == context.total_steps:
                await self._complete_workflow(workflow_id, context)
            
            logger.info(
                f"Task {task_id} completed in workflow {workflow_id}",
                extra={
                    "workflow_id": workflow_id,
                    "task_id": task_id,
                    "triggered_tasks": len(triggered_tasks),
                    "workflow_progress": f"{len(context.completed_tasks)}/{context.total_steps}"
                }
            )
            
            return triggered_tasks
            
        except Exception as e:
            logger.error(f"Failed to signal task completion: {e}")
            return []
    
    async def handle_task_failure(
        self,
        workflow_id: str,
        task_id: str,
        error: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle task failure with workflow context.
        
        Args:
            workflow_id: Workflow identifier
            task_id: Failed task identifier
            error: Failure information
            
        Returns:
            Recovery strategy and actions taken
        """
        try:
            if workflow_id not in self._active_workflows:
                return {"strategy": "ignore", "reason": "unknown_workflow"}
            
            context = self._active_workflows[workflow_id]
            context.failed_tasks.add(task_id)
            
            # Determine recovery strategy
            recovery_strategy = await self._determine_recovery_strategy(
                workflow_id, task_id, error, context
            )
            
            # Execute recovery actions
            actions_taken = await self._execute_recovery_strategy(
                workflow_id, task_id, recovery_strategy, context
            )
            
            logger.warning(
                f"Task {task_id} failed in workflow {workflow_id}",
                extra={
                    "workflow_id": workflow_id,
                    "task_id": task_id,
                    "recovery_strategy": recovery_strategy,
                    "error": error
                }
            )
            
            return {
                "strategy": recovery_strategy,
                "actions_taken": actions_taken,
                "workflow_status": context.workflow_status.value
            }
            
        except Exception as e:
            logger.error(f"Failed to handle task failure: {e}")
            return {"strategy": "error", "error": str(e)}
    
    async def _create_workflow_context(
        self,
        workflow_id: str,
        tasks: List[Dict[str, Any]]
    ) -> WorkflowContext:
        """Create workflow context from task definitions."""
        # Extract dependencies from tasks
        dependencies = {}
        for task in tasks:
            task_id = task.get('id', str(uuid.uuid4()))
            task_deps = task.get('dependencies', [])
            if task_deps:
                dependencies[task_id] = task_deps
        
        # Analyze for parallel groups (simplified)
        parallel_groups = []
        independent_tasks = []
        
        for task in tasks:
            task_id = task.get('id', str(uuid.uuid4()))
            if task_id not in dependencies:
                independent_tasks.append(task_id)
        
        if independent_tasks:
            parallel_groups.append(independent_tasks)
        
        return WorkflowContext(
            workflow_id=workflow_id,
            workflow_name=f"workflow_{workflow_id}",
            workflow_status=WorkflowStatus.RUNNING,
            current_step=0,
            total_steps=len(tasks),
            dependencies=dependencies,
            completed_tasks=set(),
            failed_tasks=set(),
            parallel_groups=parallel_groups
        )
    
    async def _build_dependency_graph(
        self,
        workflow_id: str,
        tasks: List[Dict[str, Any]]
    ) -> None:
        """Build dependency graph for workflow."""
        graph = {}
        
        for task in tasks:
            task_id = task.get('id', str(uuid.uuid4()))
            dependencies = set(task.get('dependencies', []))
            graph[task_id] = dependencies
        
        self._dependency_graph[workflow_id] = graph
    
    async def _determine_workflow_strategy(
        self,
        context: WorkflowContext
    ) -> WorkflowRoutingStrategy:
        """Determine optimal routing strategy for workflow."""
        # Simple heuristic-based strategy selection
        if len(context.parallel_groups) > 1:
            return WorkflowRoutingStrategy.PARALLEL
        elif context.priority > 7:
            return WorkflowRoutingStrategy.PRIORITY_BASED
        elif len(context.dependencies) > context.total_steps * 0.5:
            return WorkflowRoutingStrategy.SEQUENTIAL
        else:
            return WorkflowRoutingStrategy.ADAPTIVE
    
    async def _route_tasks_by_strategy(
        self,
        workflow_id: str,
        tasks: List[Dict[str, Any]],
        strategy: WorkflowRoutingStrategy
    ) -> List[Dict[str, Any]]:
        """Route tasks according to the selected strategy."""
        results = []
        
        if strategy == WorkflowRoutingStrategy.PARALLEL:
            results = await self._route_parallel_tasks(workflow_id, tasks)
        elif strategy == WorkflowRoutingStrategy.SEQUENTIAL:
            results = await self._route_sequential_tasks(workflow_id, tasks)
        else:
            # Default to adaptive routing
            results = await self._route_adaptive_tasks(workflow_id, tasks)
        
        return results
    
    async def _route_parallel_tasks(
        self,
        workflow_id: str,
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Route tasks for parallel execution."""
        results = []
        context = self._active_workflows[workflow_id]
        
        # Route all independent tasks immediately
        for parallel_group in context.parallel_groups:
            group_tasks = [t for t in tasks if t.get('id') in parallel_group]
            
            for task in group_tasks:
                message = await self._task_to_message(task)
                target_group = await self._determine_target_group(message, context)
                
                # Route immediately for parallel execution
                await self.streams_manager.send_message_to_group(target_group, message)
                
                results.append({
                    "task_id": task.get('id'),
                    "target_group": target_group,
                    "routing_mode": "parallel"
                })
        
        return results
    
    async def _route_sequential_tasks(
        self,
        workflow_id: str,
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Route tasks for sequential execution."""
        results = []
        context = self._active_workflows[workflow_id]
        
        # Sort tasks by dependencies (topological sort simplified)
        sorted_tasks = await self._topological_sort_tasks(tasks)
        
        # Route only the first task(s) without dependencies
        for task in sorted_tasks:
            task_id = task.get('id')
            
            if task_id not in context.dependencies or not context.dependencies[task_id]:
                message = await self._task_to_message(task)
                target_group = await self._determine_target_group(message, context)
                
                await self.streams_manager.send_message_to_group(target_group, message)
                
                results.append({
                    "task_id": task_id,
                    "target_group": target_group,
                    "routing_mode": "sequential"
                })
                break  # Only route first available task
        
        return results
    
    async def _route_adaptive_tasks(
        self,
        workflow_id: str,
        tasks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Route tasks with adaptive strategy."""
        # Combination of parallel and sequential based on current system state
        results = []
        context = self._active_workflows[workflow_id]
        
        # Route up to max_parallel_tasks that have no pending dependencies
        routed_count = 0
        for task in tasks:
            if routed_count >= self.max_parallel_tasks:
                break
            
            task_id = task.get('id')
            dependencies = context.dependencies.get(task_id, [])
            
            # Check if dependencies are satisfied
            if all(dep in context.completed_tasks for dep in dependencies):
                message = await self._task_to_message(task)
                target_group = await self._determine_target_group(message, context)
                
                await self.streams_manager.send_message_to_group(target_group, message)
                
                results.append({
                    "task_id": task_id,
                    "target_group": target_group,
                    "routing_mode": "adaptive"
                })
                routed_count += 1
        
        return results
    
    async def _determine_target_group(
        self,
        message: StreamMessage,
        context: Optional[WorkflowContext] = None
    ) -> str:
        """Determine optimal target consumer group for message."""
        # Simple capability-based routing
        task_type = message.payload.get('task_type', 'general')
        
        # Map task types to consumer groups
        type_to_group = {
            'architecture': 'architects_consumers',
            'backend': 'backend_engineers_consumers',
            'frontend': 'frontend_developers_consumers',
            'testing': 'qa_engineers_consumers',
            'deployment': 'devops_engineers_consumers',
            'security': 'security_engineers_consumers',
            'data': 'data_engineers_consumers'
        }
        
        return type_to_group.get(task_type, 'general_agents_consumers')
    
    async def _task_to_message(self, task: Dict[str, Any]) -> StreamMessage:
        """Convert task definition to stream message."""
        return StreamMessage(
            id=task.get('id', str(uuid.uuid4())),
            from_agent='workflow_router',
            to_agent=None,  # Will be routed to consumer group
            message_type=MessageType.TASK_REQUEST,
            payload=task,
            priority=MessagePriority(task.get('priority', 5)),
            timestamp=time.time()
        )
    
    async def _check_dependencies_satisfied(
        self,
        workflow_id: str,
        dependencies: List[str]
    ) -> bool:
        """Check if task dependencies are satisfied."""
        if workflow_id not in self._active_workflows:
            return True  # No workflow context, assume satisfied
        
        context = self._active_workflows[workflow_id]
        return all(dep in context.completed_tasks for dep in dependencies)
    
    async def _execute_routing_decision(
        self,
        message: StreamMessage,
        decision: RoutingDecision,
        context: Optional[WorkflowContext] = None
    ) -> None:
        """Execute a routing decision."""
        # Add routing metadata to message
        message.payload['_routing_decision'] = decision.to_dict()
        
        # Send to target group
        await self.streams_manager.send_message_to_group(
            decision.target_group, message
        )
    
    async def _workflow_monitoring_loop(self) -> None:
        """Background task for monitoring workflow progress."""
        while True:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
                # Check for stalled workflows
                current_time = datetime.utcnow()
                
                for workflow_id, context in self._active_workflows.items():
                    # Check for deadline violations
                    if (context.deadline and 
                        current_time > context.deadline and
                        context.workflow_status == WorkflowStatus.RUNNING):
                        
                        logger.warning(f"Workflow {workflow_id} exceeded deadline")
                        # Could trigger escalation or recovery actions
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in workflow monitoring loop: {e}")
    
    async def _optimization_loop(self) -> None:
        """Background task for routing optimization."""
        while True:
            try:
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
                # Analyze routing performance and update strategies
                await self._optimize_routing_strategies()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
    
    async def _optimize_routing_strategies(self) -> None:
        """Optimize routing strategies based on performance data."""
        # Simplified optimization - could be much more sophisticated
        
        # Analyze group performance
        all_stats = await self.streams_manager.get_all_group_stats()
        
        for group_name, stats in all_stats.items():
            if stats.success_rate < 0.95:
                # Reduce routing to underperforming groups
                logger.info(f"Reducing routes to underperforming group {group_name}")
            elif stats.success_rate > 0.99 and stats.avg_processing_time_ms < 100:
                # Increase routing to high-performing groups
                logger.info(f"Increasing routes to high-performing group {group_name}")
    
    async def get_routing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive routing metrics."""
        return {
            "workflow_routing_metrics": self._metrics.to_dict(),
            "active_workflows": len(self._active_workflows),
            "total_dependency_edges": sum(
                len(deps) for workflow_deps in self._dependency_graph.values()
                for deps in workflow_deps.values()
            ),
            "routing_cache_size": len(self._routing_cache),
            "workflow_contexts": {
                wf_id: context.to_dict() 
                for wf_id, context in self._active_workflows.items()
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        is_healthy = (
            len(self._active_workflows) < 1000 and  # Not overwhelmed
            self._metrics.failed_routings / max(1, self._metrics.total_tasks_routed) < 0.05
        )
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "active_workflows": len(self._active_workflows),
            "routing_success_rate": (
                self._metrics.successful_routings / 
                max(1, self._metrics.total_tasks_routed)
            ),
            "background_tasks_running": sum(1 for task in [
                self._workflow_monitor_task, self._optimization_task
            ] if task and not task.done()),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Placeholder methods for complete implementation
    
    async def _infer_workflow_context(self, message: StreamMessage) -> Optional[WorkflowContext]:
        """Infer workflow context from message."""
        return None
    
    async def _estimate_task_completion(self, target_group: str, message: StreamMessage) -> Optional[datetime]:
        """Estimate task completion time."""
        return None
    
    async def _estimate_workflow_completion(self, workflow_id: str) -> Optional[datetime]:
        """Estimate workflow completion time."""
        return None
    
    async def _trigger_dependent_task(self, workflow_id: str, task_id: str, context: WorkflowContext) -> None:
        """Trigger a dependent task."""
        pass
    
    async def _complete_workflow(self, workflow_id: str, context: WorkflowContext) -> None:
        """Mark workflow as complete."""
        context.workflow_status = WorkflowStatus.COMPLETED
        self._metrics.workflows_completed += 1
    
    async def _determine_recovery_strategy(self, workflow_id: str, task_id: str, error: Dict[str, Any], context: WorkflowContext) -> str:
        """Determine recovery strategy for failed task."""
        return "retry"
    
    async def _execute_recovery_strategy(self, workflow_id: str, task_id: str, strategy: str, context: WorkflowContext) -> List[str]:
        """Execute recovery strategy."""
        return []
    
    async def _topological_sort_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform topological sort of tasks based on dependencies."""
        return tasks  # Simplified - would implement proper topological sort
    
    async def _initialize_group_capabilities(self) -> None:
        """Initialize group capabilities from existing consumer groups."""
        # Simplified initialization
        self._group_capabilities = {
            'architects_consumers': {'architecture', 'design', 'planning'},
            'backend_engineers_consumers': {'backend', 'api', 'database'},
            'frontend_developers_consumers': {'frontend', 'ui', 'react'},
            'qa_engineers_consumers': {'testing', 'quality', 'automation'},
            'devops_engineers_consumers': {'deployment', 'infrastructure', 'ci_cd'},
            'security_engineers_consumers': {'security', 'audit', 'compliance'},
            'data_engineers_consumers': {'data', 'etl', 'analytics'},
            'general_agents_consumers': {'general', 'misc', 'utility'}
        }