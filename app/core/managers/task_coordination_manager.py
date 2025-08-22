"""
Task Coordination Manager - Consolidated Task Management

Consolidates functionality from:
- SimpleOrchestrator task delegation
- TaskManager, WorkflowManager, TaskCoordinationManager
- IntelligentTaskRouter, CapabilityMatcher
- All task-related orchestrator components (20+ files)

Preserves Epic 1 performance optimizations and API v2 compatibility.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import heapq

from ..config import settings
from ..logging_service import get_component_logger
from ...models.task import Task, TaskStatus, TaskPriority
from ...models.agent import AgentStatus

logger = get_component_logger("task_coordination_manager")


class RoutingStrategy(Enum):
    """Task routing strategies."""
    ROUND_ROBIN = "round_robin"
    CAPABILITY_MATCH = "capability_match"
    LOAD_BALANCED = "load_balanced"
    PRIORITY_QUEUE = "priority_queue"


@dataclass
class TaskAssignment:
    """Consolidated task assignment representation."""
    task_id: str
    agent_id: str
    task_description: str
    task_type: str
    priority: TaskPriority
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    status: TaskStatus = TaskStatus.PENDING
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    progress_percentage: float = 0.0
    estimated_duration_minutes: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "task_description": self.task_description,
            "task_type": self.task_type,
            "priority": self.priority.value,
            "assigned_at": self.assigned_at.isoformat(),
            "status": self.status.value,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "progress_percentage": self.progress_percentage,
            "estimated_duration_minutes": self.estimated_duration_minutes
        }


@dataclass
class WorkflowDefinition:
    """Workflow definition for complex multi-step tasks."""
    workflow_id: str
    name: str
    steps: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    timeout_minutes: int = 60
    retry_policy: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class WorkflowExecution:
    """Workflow execution state tracking."""
    workflow_id: str
    execution_id: str
    status: str
    current_step: int
    step_results: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class TaskCoordinationError(Exception):
    """Task coordination management errors."""
    pass


class TaskCoordinationManager:
    """
    Consolidated Task Coordination Manager
    
    Replaces and consolidates:
    - SimpleOrchestrator task methods (delegate_task, task status)
    - TaskManager, WorkflowManager, TaskCoordinationManager
    - IntelligentTaskRouter, CapabilityMatcher
    - TaskQueue, TaskDistributor, TaskScheduler
    - All task-related manager classes (20+ files)
    
    Preserves:
    - API v2 compatibility for PWA integration
    - Epic 1 performance optimizations
    - WebSocket broadcasting integration
    - Intelligent task routing and load balancing
    """

    def __init__(self, master_orchestrator):
        """Initialize task coordination manager."""
        self.master_orchestrator = master_orchestrator
        self.tasks: Dict[str, TaskAssignment] = {}
        self.workflows: Dict[str, WorkflowExecution] = {}
        
        # Task queues by priority
        self.task_queues = {
            TaskPriority.CRITICAL: [],
            TaskPriority.HIGH: [],
            TaskPriority.MEDIUM: [],
            TaskPriority.LOW: []
        }
        
        # Routing and load balancing
        self.routing_strategy = RoutingStrategy.CAPABILITY_MATCH
        self.agent_workloads: Dict[str, int] = {}
        self.capability_map: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.delegation_count = 0
        self.completion_count = 0
        self.failure_count = 0
        self.average_completion_time_minutes = 0.0
        
        # Background processing
        self.task_processor_running = False
        self.task_processor_task: Optional[asyncio.Task] = None
        
        logger.info("Task Coordination Manager initialized")

    async def initialize(self) -> None:
        """Initialize task coordination manager."""
        try:
            # Initialize capability mapping
            await self._initialize_capability_mapping()
            
            # Load task routing configuration
            await self._load_routing_configuration()
            
            logger.info("✅ Task Coordination Manager initialized successfully")
            
        except Exception as e:
            logger.error("❌ Task Coordination Manager initialization failed", error=str(e))
            raise TaskCoordinationError(f"Initialization failed: {e}") from e

    async def start(self) -> None:
        """Start task coordination background processes."""
        if self.task_processor_running:
            return
        
        # Start task processing loop
        self.task_processor_running = True
        self.task_processor_task = asyncio.create_task(self._task_processing_loop())
        
        logger.info("🚀 Task Coordination Manager started")

    async def shutdown(self) -> None:
        """Shutdown task coordination manager."""
        logger.info("🛑 Shutting down Task Coordination Manager...")
        
        # Stop task processor
        self.task_processor_running = False
        if self.task_processor_task and not self.task_processor_task.done():
            self.task_processor_task.cancel()
            try:
                await self.task_processor_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all pending tasks
        pending_tasks = [task for task in self.tasks.values() 
                        if task.status == TaskStatus.PENDING]
        
        for task in pending_tasks:
            task.status = TaskStatus.CANCELLED
            await self._broadcast_task_update(task)
        
        logger.info("✅ Task Coordination Manager shutdown complete")

    async def delegate_task(
        self,
        task_description: str,
        task_type: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        preferred_agent_role: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Delegate task to optimal agent - compatible with SimpleOrchestrator API.
        
        Preserves API v2 compatibility and intelligent routing.
        Maintains Epic 1 performance targets.
        """
        operation_start = datetime.utcnow()
        
        try:
            # Generate task ID
            task_id = str(uuid.uuid4())
            
            # Find optimal agent for task
            agent_id = await self._find_optimal_agent(
                task_type=task_type,
                priority=priority,
                preferred_role=preferred_agent_role,
                required_capabilities=kwargs.get('required_capabilities', [])
            )
            
            if not agent_id:
                raise TaskCoordinationError("No suitable agent available")
            
            # Create task assignment
            task_assignment = TaskAssignment(
                task_id=task_id,
                agent_id=agent_id,
                task_description=task_description,
                task_type=task_type,
                priority=priority,
                estimated_duration_minutes=kwargs.get('estimated_duration_minutes')
            )
            
            # Store task
            self.tasks[task_id] = task_assignment
            self.delegation_count += 1
            
            # Update agent workload
            self.agent_workloads[agent_id] = self.agent_workloads.get(agent_id, 0) + 1
            
            # Queue task for processing
            heapq.heappush(
                self.task_queues[priority],
                (task_assignment.assigned_at.timestamp(), task_id)
            )
            
            # Persist to database
            await self._persist_task(task_assignment)
            
            # Broadcast task creation (WebSocket integration)
            task_data = {
                "task_id": task_id,
                "agent_id": agent_id,
                "task_description": task_description,
                "task_type": task_type,
                "priority": priority.value,
                "status": TaskStatus.PENDING.value,
                "created_at": task_assignment.assigned_at.isoformat(),
                "source": "task_delegation"
            }
            await self.master_orchestrator.broadcast_task_update(task_id, task_data)
            
            # Performance tracking
            duration_ms = (datetime.utcnow() - operation_start).total_seconds() * 1000
            
            logger.info("✅ Task delegated successfully",
                       task_id=task_id,
                       agent_id=agent_id,
                       task_type=task_type,
                       priority=priority.value,
                       duration_ms=duration_ms)
            
            return task_id
            
        except Exception as e:
            logger.error("❌ Task delegation failed", task_id=task_id, error=str(e))
            raise TaskCoordinationError(f"Failed to delegate task: {e}") from e

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status - compatible with SimpleOrchestrator API."""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        
        # Get additional task details from agent
        agent_details = await self.master_orchestrator.agent_lifecycle.get_agent_status(
            task.agent_id
        )
        
        return {
            "task": task.to_dict(),
            "agent_details": agent_details,
            "workflow_id": self._get_workflow_for_task(task_id),
            "dependencies": await self._get_task_dependencies(task_id)
        }

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel task - compatible with SimpleOrchestrator API."""
        try:
            if task_id not in self.tasks:
                logger.warning("Task not found for cancellation", task_id=task_id)
                return False
            
            task = self.tasks[task_id]
            
            # Only cancel if task is still pending or in progress
            if task.status not in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]:
                logger.warning("Task cannot be cancelled in current status",
                             task_id=task_id, status=task.status.value)
                return False
            
            # Update task status
            old_status = task.status
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.utcnow()
            
            # Update agent workload
            if task.agent_id in self.agent_workloads:
                self.agent_workloads[task.agent_id] = max(0, 
                    self.agent_workloads[task.agent_id] - 1)
            
            # Update database
            await self._update_task_status(task_id, TaskStatus.CANCELLED)
            
            # Broadcast cancellation
            cancel_data = {
                "task_id": task_id,
                "status": TaskStatus.CANCELLED.value,
                "previous_status": old_status.value,
                "cancelled_at": task.completed_at.isoformat(),
                "source": "task_cancellation"
            }
            await self.master_orchestrator.broadcast_task_update(task_id, cancel_data)
            
            logger.info("✅ Task cancelled successfully", task_id=task_id)
            return True
            
        except Exception as e:
            logger.error("❌ Task cancellation failed", task_id=task_id, error=str(e))
            return False

    async def complete_task(
        self,
        task_id: str,
        result: Optional[Dict[str, Any]] = None,
        success: bool = True
    ) -> bool:
        """Complete task with result."""
        try:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            
            # Update task status
            task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            task.completed_at = datetime.utcnow()
            task.result = result
            task.progress_percentage = 100.0
            
            # Update statistics
            if success:
                self.completion_count += 1
            else:
                self.failure_count += 1
            
            # Update average completion time
            duration_minutes = (task.completed_at - task.assigned_at).total_seconds() / 60
            self._update_average_completion_time(duration_minutes)
            
            # Update agent workload
            if task.agent_id in self.agent_workloads:
                self.agent_workloads[task.agent_id] = max(0,
                    self.agent_workloads[task.agent_id] - 1)
            
            # Update database
            await self._update_task_completion(task_id, task.status, result)
            
            # Broadcast completion
            completion_data = {
                "task_id": task_id,
                "status": task.status.value,
                "completed_at": task.completed_at.isoformat(),
                "result": result,
                "success": success,
                "duration_minutes": duration_minutes,
                "source": "task_completion"
            }
            await self.master_orchestrator.broadcast_task_update(task_id, completion_data)
            
            logger.info("✅ Task completed",
                       task_id=task_id,
                       success=success,
                       duration_minutes=duration_minutes)
            
            return True
            
        except Exception as e:
            logger.error("❌ Task completion failed", task_id=task_id, error=str(e))
            return False

    async def update_task_progress(
        self,
        task_id: str,
        progress_percentage: float,
        status_message: Optional[str] = None
    ) -> bool:
        """Update task progress."""
        try:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            task.progress_percentage = min(100.0, max(0.0, progress_percentage))
            task.status = TaskStatus.IN_PROGRESS
            
            # Broadcast progress update
            progress_data = {
                "task_id": task_id,
                "progress_percentage": task.progress_percentage,
                "status_message": status_message,
                "source": "task_progress"
            }
            await self.master_orchestrator.broadcast_task_update(task_id, progress_data)
            
            return True
            
        except Exception as e:
            logger.error("Task progress update failed", task_id=task_id, error=str(e))
            return False

    async def execute_workflow(self, workflow_def: WorkflowDefinition) -> str:
        """Execute multi-step workflow - UnifiedOrchestrator compatibility."""
        execution_id = str(uuid.uuid4())
        
        try:
            # Create workflow execution
            execution = WorkflowExecution(
                workflow_id=workflow_def.workflow_id,
                execution_id=execution_id,
                status="running",
                current_step=0
            )
            
            self.workflows[execution_id] = execution
            
            # Start workflow execution
            asyncio.create_task(self._execute_workflow_steps(workflow_def, execution))
            
            logger.info("Workflow execution started",
                       workflow_id=workflow_def.workflow_id,
                       execution_id=execution_id)
            
            return execution_id
            
        except Exception as e:
            logger.error("Workflow execution failed", 
                        workflow_id=workflow_def.workflow_id, error=str(e))
            raise TaskCoordinationError(f"Workflow execution failed: {e}") from e

    async def get_status(self) -> Dict[str, Any]:
        """Get task coordination manager status."""
        pending_tasks = len([t for t in self.tasks.values() 
                           if t.status == TaskStatus.PENDING])
        active_tasks = len([t for t in self.tasks.values()
                          if t.status == TaskStatus.IN_PROGRESS])
        completed_tasks = len([t for t in self.tasks.values()
                             if t.status == TaskStatus.COMPLETED])
        
        return {
            "total_tasks": len(self.tasks),
            "pending_tasks": pending_tasks,
            "active_tasks": active_tasks,
            "completed_tasks": completed_tasks,
            "delegation_count": self.delegation_count,
            "completion_count": self.completion_count,
            "failure_count": self.failure_count,
            "success_rate": (self.completion_count / max(1, self.completion_count + self.failure_count)) * 100,
            "average_completion_time_minutes": self.average_completion_time_minutes,
            "routing_strategy": self.routing_strategy.value,
            "active_workflows": len(self.workflows)
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get task metrics for system monitoring."""
        return {
            "task_count": len(self.tasks),
            "pending_tasks": len([t for t in self.tasks.values() 
                                if t.status == TaskStatus.PENDING]),
            "active_tasks": len([t for t in self.tasks.values()
                               if t.status == TaskStatus.IN_PROGRESS]),
            "delegation_count": self.delegation_count,
            "completion_count": self.completion_count,
            "failure_count": self.failure_count,
            "average_completion_time": self.average_completion_time_minutes,
            "agent_workload_distribution": dict(self.agent_workloads)
        }

    async def cleanup_expired_tasks(self) -> None:
        """Cleanup expired and old completed tasks."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            expired_tasks = [
                task_id for task_id, task in self.tasks.items()
                if task.completed_at and task.completed_at < cutoff_time
            ]
            
            for task_id in expired_tasks:
                del self.tasks[task_id]
                logger.debug("Cleaned up expired task", task_id=task_id)
            
        except Exception as e:
            logger.error("Failed to cleanup expired tasks", error=str(e))

    # ==================================================================
    # INTELLIGENT TASK ROUTING
    # ==================================================================

    async def _find_optimal_agent(
        self,
        task_type: str,
        priority: TaskPriority,
        preferred_role: Optional[str] = None,
        required_capabilities: List[str] = None
    ) -> Optional[str]:
        """Find optimal agent using intelligent routing."""
        # Get available agents
        available_agents = await self._get_available_agents()
        
        if not available_agents:
            return None
        
        # Apply routing strategy
        if self.routing_strategy == RoutingStrategy.CAPABILITY_MATCH:
            return await self._route_by_capability_match(
                available_agents, task_type, required_capabilities or []
            )
        elif self.routing_strategy == RoutingStrategy.LOAD_BALANCED:
            return await self._route_by_load_balance(available_agents)
        elif self.routing_strategy == RoutingStrategy.PRIORITY_QUEUE:
            return await self._route_by_priority(available_agents, priority)
        else:  # ROUND_ROBIN
            return await self._route_round_robin(available_agents)

    async def _get_available_agents(self) -> List[str]:
        """Get list of available agent IDs."""
        available_agents = []
        
        for agent_id, agent in self.master_orchestrator.agent_lifecycle.agents.items():
            if (agent.status == AgentStatus.ACTIVE and 
                self.agent_workloads.get(agent_id, 0) < 3):  # Max 3 concurrent tasks
                available_agents.append(agent_id)
        
        return available_agents

    async def _route_by_capability_match(
        self,
        available_agents: List[str],
        task_type: str,
        required_capabilities: List[str]
    ) -> Optional[str]:
        """Route task based on agent capabilities."""
        best_match = None
        best_score = 0
        
        for agent_id in available_agents:
            agent_capabilities = self.capability_map.get(agent_id, [])
            
            # Calculate capability match score
            score = 0
            if task_type in agent_capabilities:
                score += 10
            
            for capability in required_capabilities:
                if capability in agent_capabilities:
                    score += 5
            
            # Consider workload as tiebreaker
            workload = self.agent_workloads.get(agent_id, 0)
            score -= workload
            
            if score > best_score:
                best_score = score
                best_match = agent_id
        
        return best_match

    async def _route_by_load_balance(self, available_agents: List[str]) -> str:
        """Route to agent with lowest workload."""
        min_workload = float('inf')
        best_agent = None
        
        for agent_id in available_agents:
            workload = self.agent_workloads.get(agent_id, 0)
            if workload < min_workload:
                min_workload = workload
                best_agent = agent_id
        
        return best_agent

    async def _route_by_priority(
        self,
        available_agents: List[str],
        priority: TaskPriority
    ) -> str:
        """Route based on task priority and agent performance."""
        # For high priority tasks, prefer agents with good completion rates
        if priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
            # Simple implementation - could use actual performance metrics
            return available_agents[0] if available_agents else None
        
        return await self._route_by_load_balance(available_agents)

    async def _route_round_robin(self, available_agents: List[str]) -> str:
        """Simple round-robin routing."""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        agent_id = available_agents[self._round_robin_index % len(available_agents)]
        self._round_robin_index += 1
        
        return agent_id

    # ==================================================================
    # WORKFLOW EXECUTION
    # ==================================================================

    async def _execute_workflow_steps(
        self,
        workflow_def: WorkflowDefinition,
        execution: WorkflowExecution
    ) -> None:
        """Execute workflow steps sequentially."""
        try:
            for step_index, step in enumerate(workflow_def.steps):
                execution.current_step = step_index
                
                # Execute step
                step_result = await self._execute_workflow_step(step, execution)
                execution.step_results[f"step_{step_index}"] = step_result
                
                # Check for failure
                if not step_result.get('success', False):
                    execution.status = "failed"
                    break
            
            if execution.status == "running":
                execution.status = "completed"
                execution.completed_at = datetime.utcnow()
            
        except Exception as e:
            logger.error("Workflow execution failed",
                        workflow_id=workflow_def.workflow_id,
                        execution_id=execution.execution_id,
                        error=str(e))
            execution.status = "failed"

    async def _execute_workflow_step(
        self,
        step: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute single workflow step."""
        # Delegate step as task
        task_id = await self.delegate_task(
            task_description=step.get('description', 'Workflow step'),
            task_type=step.get('type', 'workflow_step'),
            priority=TaskPriority.MEDIUM
        )
        
        # Wait for task completion (simplified)
        timeout_minutes = step.get('timeout_minutes', 30)
        timeout_time = datetime.utcnow() + timedelta(minutes=timeout_minutes)
        
        while datetime.utcnow() < timeout_time:
            task_status = await self.get_task_status(task_id)
            if task_status and task_status['task']['status'] in ['completed', 'failed']:
                return {
                    'success': task_status['task']['status'] == 'completed',
                    'result': task_status['task']['result'],
                    'task_id': task_id
                }
            
            await asyncio.sleep(10)  # Check every 10 seconds
        
        # Timeout
        await self.cancel_task(task_id)
        return {'success': False, 'error': 'Step timeout', 'task_id': task_id}

    # ==================================================================
    # BACKGROUND PROCESSING
    # ==================================================================

    async def _task_processing_loop(self) -> None:
        """Background task processing loop."""
        while self.task_processor_running:
            try:
                # Process tasks from priority queues
                await self._process_priority_queues()
                
                # Update task statuses
                await self._update_task_statuses()
                
                await asyncio.sleep(5)  # Process every 5 seconds
                
            except Exception as e:
                logger.error("Error in task processing loop", error=str(e))
                await asyncio.sleep(10)

    async def _process_priority_queues(self) -> None:
        """Process tasks from priority queues."""
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, 
                        TaskPriority.MEDIUM, TaskPriority.LOW]:
            queue = self.task_queues[priority]
            
            while queue:
                _, task_id = heapq.heappop(queue)
                
                if task_id in self.tasks:
                    task = self.tasks[task_id]
                    if task.status == TaskStatus.PENDING:
                        # Start task processing
                        task.status = TaskStatus.IN_PROGRESS
                        await self._broadcast_task_update(task)

    async def _update_task_statuses(self) -> None:
        """Update task statuses from agents."""
        # This would integrate with agent status updates
        # Simplified implementation for now
        pass

    async def _broadcast_task_update(self, task: TaskAssignment) -> None:
        """Broadcast task update via WebSocket."""
        update_data = {
            "task_id": task.task_id,
            "status": task.status.value,
            "progress": task.progress_percentage,
            "updated_at": datetime.utcnow().isoformat(),
            "source": "task_status_update"
        }
        await self.master_orchestrator.broadcast_task_update(
            task.task_id, update_data
        )

    # ==================================================================
    # HELPER METHODS
    # ==================================================================

    async def _initialize_capability_mapping(self) -> None:
        """Initialize agent capability mapping."""
        # This would be loaded from configuration or learned from agent behavior
        self.capability_map = {
            # Will be populated dynamically as agents register
        }

    async def _load_routing_configuration(self) -> None:
        """Load task routing configuration."""
        # Load from settings or configuration file
        routing_config = getattr(settings, 'TASK_ROUTING_STRATEGY', 'capability_match')
        
        try:
            self.routing_strategy = RoutingStrategy(routing_config)
        except ValueError:
            self.routing_strategy = RoutingStrategy.CAPABILITY_MATCH
        
        logger.info("Task routing strategy loaded", strategy=self.routing_strategy.value)

    def _update_average_completion_time(self, duration_minutes: float) -> None:
        """Update average completion time with new data point."""
        if self.completion_count == 1:
            self.average_completion_time_minutes = duration_minutes
        else:
            # Weighted average
            weight = 0.1  # Give more weight to recent completions
            self.average_completion_time_minutes = (
                (1 - weight) * self.average_completion_time_minutes +
                weight * duration_minutes
            )

    def _get_workflow_for_task(self, task_id: str) -> Optional[str]:
        """Get workflow ID that contains this task."""
        for execution in self.workflows.values():
            if task_id in execution.step_results:
                return execution.workflow_id
        return None

    async def _get_task_dependencies(self, task_id: str) -> List[str]:
        """Get task dependencies."""
        # Simplified - would implement actual dependency tracking
        return []

    async def _persist_task(self, task: TaskAssignment) -> None:
        """Persist task to database."""
        try:
            db_session = await self.master_orchestrator.integration.get_database_session()
            if not db_session:
                return
            
            db_task = Task(
                id=task.task_id,
                description=task.task_description,
                task_type=task.task_type,
                priority=task.priority,
                status=task.status,
                assigned_agent_id=task.agent_id,
                created_at=task.assigned_at
            )
            
            db_session.add(db_task)
            await db_session.commit()
            
        except Exception as e:
            logger.warning("Failed to persist task to database",
                         task_id=task.task_id, error=str(e))

    async def _update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Update task status in database."""
        try:
            db_session = await self.master_orchestrator.integration.get_database_session()
            if not db_session:
                return
            
            from sqlalchemy import update
            await db_session.execute(
                update(Task)
                .where(Task.id == task_id)
                .values(status=status, updated_at=datetime.utcnow())
            )
            await db_session.commit()
            
        except Exception as e:
            logger.warning("Failed to update task status in database",
                         task_id=task_id, error=str(e))

    async def _update_task_completion(
        self,
        task_id: str,
        status: TaskStatus,
        result: Optional[Dict[str, Any]]
    ) -> None:
        """Update task completion in database."""
        try:
            db_session = await self.master_orchestrator.integration.get_database_session()
            if not db_session:
                return
            
            from sqlalchemy import update
            await db_session.execute(
                update(Task)
                .where(Task.id == task_id)
                .values(
                    status=status,
                    result=result,
                    updated_at=datetime.utcnow()
                )
            )
            await db_session.commit()
            
        except Exception as e:
            logger.warning("Failed to update task completion in database",
                         task_id=task_id, error=str(e))