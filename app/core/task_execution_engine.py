"""
Task Execution Engine for LeanVibe Agent Hive 2.0

Handles task execution workflow, state management, progress tracking,
and integration with the hook system for comprehensive observability.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass
from enum import Enum
import json
import time

import structlog
from sqlalchemy import select, update, and_
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_async_session
from .redis import get_redis, AgentMessageBroker
from .hook_lifecycle_system import HookLifecycleSystem, HookEvent, HookType
from ..models.task import Task, TaskStatus, TaskPriority, TaskType
from ..models.agent import Agent, AgentStatus

logger = structlog.get_logger()


class ExecutionPhase(str, Enum):
    """Phases of task execution."""
    PREPARATION = "preparation"
    EXECUTION = "execution"
    VALIDATION = "validation"
    COMPLETION = "completion"


class ExecutionOutcome(str, Enum):
    """Possible task execution outcomes."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


@dataclass
class TaskExecutionContext:
    """Context for task execution."""
    task_id: uuid.UUID
    agent_id: uuid.UUID
    started_at: datetime
    phase: ExecutionPhase
    metadata: Dict[str, Any]
    progress_percentage: float = 0.0
    estimated_completion: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "agent_id": str(self.agent_id),
            "started_at": self.started_at.isoformat(),
            "phase": self.phase.value,
            "progress_percentage": self.progress_percentage,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "metadata": self.metadata
        }


@dataclass
class ExecutionResult:
    """Result of task execution."""
    task_id: uuid.UUID
    outcome: ExecutionOutcome
    result_data: Dict[str, Any]
    execution_time_ms: float
    error_message: Optional[str] = None
    artifacts: List[Dict[str, Any]] = None
    performance_metrics: Dict[str, float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "outcome": self.outcome.value,
            "result_data": self.result_data,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
            "artifacts": self.artifacts or [],
            "performance_metrics": self.performance_metrics or {}
        }


class TaskExecutionEngine:
    """
    Manages task execution workflow with comprehensive state tracking.
    
    This engine orchestrates the execution of tasks by agents, provides
    real-time progress tracking, handles execution phases, and integrates
    with the hook system for observability.
    """
    
    def __init__(
        self,
        redis_client=None,
        hook_system: Optional[HookLifecycleSystem] = None,
        max_execution_time_minutes: int = 60
    ):
        self.redis = redis_client or get_redis()
        self.message_broker = AgentMessageBroker(self.redis)
        self.hook_system = hook_system
        self.max_execution_time = timedelta(minutes=max_execution_time_minutes)
        
        # Active executions tracking
        self.active_executions: Dict[uuid.UUID, TaskExecutionContext] = {}
        self.execution_callbacks: Dict[uuid.UUID, List[Callable]] = {}
        
        # Performance metrics
        self.execution_times: Dict[str, List[float]] = {}
        self.success_rates: Dict[str, float] = {}
        
        logger.info("🔧 Task Execution Engine initialized")
    
    async def start_task_execution(
        self,
        task_id: uuid.UUID,
        agent_id: uuid.UUID,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Start task execution by an agent.
        
        Args:
            task_id: ID of the task to execute
            agent_id: ID of the executing agent
            execution_context: Additional context for execution
        
        Returns:
            True if execution started successfully
        """
        start_time = datetime.utcnow()
        
        try:
            async with get_async_session() as db:
                # Get task and agent
                task_result = await db.execute(select(Task).where(Task.id == task_id))
                task = task_result.scalar_one_or_none()
                
                agent_result = await db.execute(select(Agent).where(Agent.id == agent_id))
                agent = agent_result.scalar_one_or_none()
                
                if not task or not agent:
                    logger.error("Task or agent not found", task_id=str(task_id), agent_id=str(agent_id))
                    return False
                
                if task.status != TaskStatus.ASSIGNED:
                    logger.warning("Task not in ASSIGNED status", task_id=str(task_id), status=task.status)
                    return False
                
                # Start execution
                task.start_execution()
                agent.status = AgentStatus.BUSY
                await db.commit()
                
                # Create execution context
                exec_context = TaskExecutionContext(
                    task_id=task_id,
                    agent_id=agent_id,
                    started_at=start_time,
                    phase=ExecutionPhase.PREPARATION,
                    metadata=execution_context or {},
                    estimated_completion=task.get_estimated_completion_time()
                )
                
                self.active_executions[task_id] = exec_context
                
                # Send execution start message to agent
                await self.message_broker.send_message(
                    from_agent="execution_engine",
                    to_agent=str(agent_id),
                    message_type="task_execution_start",
                    payload={
                        "task_id": str(task_id),
                        "task_title": task.title,
                        "task_description": task.description,
                        "task_type": task.task_type.value if task.task_type else "general",
                        "priority": task.priority.name.lower(),
                        "context": task.context or {},
                        "execution_context": execution_context or {},
                        "estimated_effort": task.estimated_effort,
                        "max_execution_time_minutes": self.max_execution_time.total_seconds() / 60
                    }
                )
                
                # Emit hook event for task start
                if self.hook_system:
                    hook_event = HookEvent(
                        hook_type=HookType.PRE_TOOL_USE,
                        agent_id=agent_id,
                        session_id=None,
                        timestamp=start_time,
                        payload={
                            "action": "task_execution_start",
                            "task_id": str(task_id),
                            "task_type": task.task_type.value if task.task_type else "general",
                            "tool_name": "task_executor",
                            "parameters": {
                                "task_title": task.title,
                                "priority": task.priority.name.lower(),
                                "estimated_effort": task.estimated_effort
                            }
                        }
                    )
                    await self.hook_system.process_hook_event(hook_event)
                
                # Publish execution start event
                await self._publish_execution_event(
                    "task_execution_started",
                    task_id,
                    agent_id,
                    {
                        "task_title": task.title,
                        "agent_name": agent.name,
                        "execution_context": exec_context.to_dict()
                    }
                )
                
                # Start timeout monitoring
                asyncio.create_task(self._monitor_execution_timeout(task_id))
                
                logger.info(
                    "✅ Task execution started",
                    task_id=str(task_id),
                    agent_id=str(agent_id),
                    task_title=task.title,
                    estimated_completion=exec_context.estimated_completion.isoformat() if exec_context.estimated_completion else None
                )
                
                return True
                
        except Exception as e:
            logger.error("❌ Failed to start task execution", task_id=str(task_id), error=str(e))
            return False
    
    async def update_execution_progress(
        self,
        task_id: uuid.UUID,
        phase: ExecutionPhase,
        progress_percentage: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update task execution progress.
        
        Args:
            task_id: ID of the executing task
            phase: Current execution phase
            progress_percentage: Progress as percentage (0-100)
            metadata: Additional progress metadata
        
        Returns:
            True if update successful
        """
        try:
            if task_id not in self.active_executions:
                logger.warning("No active execution found", task_id=str(task_id))
                return False
            
            exec_context = self.active_executions[task_id]
            exec_context.phase = phase
            exec_context.progress_percentage = min(100.0, max(0.0, progress_percentage))
            
            if metadata:
                exec_context.metadata.update(metadata)
            
            # Publish progress event
            await self._publish_execution_event(
                "task_execution_progress",
                task_id,
                exec_context.agent_id,
                {
                    "phase": phase.value,
                    "progress_percentage": progress_percentage,
                    "metadata": metadata or {}
                }
            )
            
            logger.debug(
                "📊 Task execution progress updated",
                task_id=str(task_id),
                phase=phase.value,
                progress=f"{progress_percentage}%"
            )
            
            return True
            
        except Exception as e:
            logger.error("❌ Failed to update execution progress", task_id=str(task_id), error=str(e))
            return False
    
    async def complete_task_execution(
        self,
        task_id: uuid.UUID,
        outcome: ExecutionOutcome,
        result_data: Dict[str, Any],
        error_message: Optional[str] = None,
        artifacts: Optional[List[Dict[str, Any]]] = None
    ) -> ExecutionResult:
        """
        Complete task execution with results.
        
        Args:
            task_id: ID of the completed task
            outcome: Execution outcome
            result_data: Task execution results
            error_message: Error message if failed
            artifacts: Generated artifacts
        
        Returns:
            ExecutionResult with completion details
        """
        completion_time = datetime.utcnow()
        
        try:
            if task_id not in self.active_executions:
                logger.warning("No active execution found", task_id=str(task_id))
                return ExecutionResult(
                    task_id=task_id,
                    outcome=ExecutionOutcome.FAILURE,
                    result_data={},
                    execution_time_ms=0,
                    error_message="No active execution found"
                )
            
            exec_context = self.active_executions[task_id]
            execution_time_ms = (completion_time - exec_context.started_at).total_seconds() * 1000
            
            async with get_async_session() as db:
                # Get task and agent
                task_result = await db.execute(select(Task).where(Task.id == task_id))
                task = task_result.scalar_one_or_none()
                
                agent_result = await db.execute(select(Agent).where(Agent.id == exec_context.agent_id))
                agent = agent_result.scalar_one_or_none()
                
                if not task or not agent:
                    logger.error("Task or agent not found during completion", task_id=str(task_id))
                    return ExecutionResult(
                        task_id=task_id,
                        outcome=ExecutionOutcome.FAILURE,
                        result_data={},
                        execution_time_ms=execution_time_ms,
                        error_message="Task or agent not found"
                    )
                
                # Update task based on outcome
                if outcome == ExecutionOutcome.SUCCESS:
                    task.complete_successfully(result_data)
                    # Update agent success metrics
                    completed_count = int(agent.total_tasks_completed or 0) + 1
                    agent.total_tasks_completed = str(completed_count)
                else:
                    task.fail_with_error(error_message or f"Task failed with outcome: {outcome.value}")
                    # Update agent failure metrics
                    failed_count = int(agent.total_tasks_failed or 0) + 1
                    agent.total_tasks_failed = str(failed_count)
                
                # Update agent status back to active
                agent.status = AgentStatus.ACTIVE
                agent.last_active = completion_time
                
                # Update average response time
                current_avg = float(agent.average_response_time or 0.0)
                total_tasks = int(agent.total_tasks_completed or 0) + int(agent.total_tasks_failed or 0)
                if total_tasks > 0:
                    new_avg = ((current_avg * (total_tasks - 1)) + execution_time_ms) / total_tasks
                    agent.average_response_time = str(new_avg)
                
                await db.commit()
                
                # Create execution result
                execution_result = ExecutionResult(
                    task_id=task_id,
                    outcome=outcome,
                    result_data=result_data,
                    execution_time_ms=execution_time_ms,
                    error_message=error_message,
                    artifacts=artifacts or [],
                    performance_metrics={
                        "execution_time_ms": execution_time_ms,
                        "progress_updates": len(exec_context.metadata.get("progress_updates", [])),
                        "memory_usage": exec_context.metadata.get("memory_usage", 0),
                        "cpu_usage": exec_context.metadata.get("cpu_usage", 0)
                    }
                )
                
                # Remove from active executions
                del self.active_executions[task_id]
                
                # Update performance metrics
                task_type = task.task_type.value if task.task_type else "general"
                if task_type not in self.execution_times:
                    self.execution_times[task_type] = []
                self.execution_times[task_type].append(execution_time_ms)
                
                # Calculate success rate
                total_executions = len(self.execution_times[task_type])
                if total_executions > 0:
                    # For simplicity, assume success rate based on outcome
                    success_count = total_executions if outcome == ExecutionOutcome.SUCCESS else total_executions - 1
                    self.success_rates[task_type] = success_count / total_executions
                
                # Emit hook event for task completion
                if self.hook_system:
                    hook_event = HookEvent(
                        hook_type=HookType.POST_TOOL_USE,
                        agent_id=exec_context.agent_id,
                        session_id=None,
                        timestamp=completion_time,
                        payload={
                            "action": "task_execution_complete",
                            "task_id": str(task_id),
                            "outcome": outcome.value,
                            "tool_name": "task_executor",
                            "execution_time_ms": execution_time_ms,
                            "success": outcome == ExecutionOutcome.SUCCESS,
                            "result": result_data
                        }
                    )
                    await self.hook_system.process_hook_event(hook_event)
                
                # Send completion message to agent
                await self.message_broker.send_message(
                    from_agent="execution_engine",
                    to_agent=str(exec_context.agent_id),
                    message_type="task_execution_complete",
                    payload={
                        "task_id": str(task_id),
                        "outcome": outcome.value,
                        "execution_time_ms": execution_time_ms,
                        "result": execution_result.to_dict()
                    }
                )
                
                # Publish completion event
                await self._publish_execution_event(
                    "task_execution_completed",
                    task_id,
                    exec_context.agent_id,
                    {
                        "outcome": outcome.value,
                        "execution_time_ms": execution_time_ms,
                        "task_title": task.title,
                        "agent_name": agent.name,
                        "success": outcome == ExecutionOutcome.SUCCESS
                    }
                )
                
                logger.info(
                    "✅ Task execution completed",
                    task_id=str(task_id),
                    agent_id=str(exec_context.agent_id),
                    outcome=outcome.value,
                    execution_time_ms=execution_time_ms,
                    success=outcome == ExecutionOutcome.SUCCESS
                )
                
                return execution_result
                
        except Exception as e:
            logger.error("❌ Failed to complete task execution", task_id=str(task_id), error=str(e))
            # Clean up active execution on error
            self.active_executions.pop(task_id, None)
            return ExecutionResult(
                task_id=task_id,
                outcome=ExecutionOutcome.FAILURE,
                result_data={},
                execution_time_ms=0,
                error_message=str(e)
            )
    
    async def cancel_task_execution(self, task_id: uuid.UUID, reason: str = "Cancelled by system") -> bool:
        """
        Cancel an active task execution.
        
        Args:
            task_id: ID of the task to cancel
            reason: Cancellation reason
        
        Returns:
            True if cancellation successful
        """
        try:
            if task_id not in self.active_executions:
                logger.warning("No active execution to cancel", task_id=str(task_id))
                return False
            
            exec_context = self.active_executions[task_id]
            
            # Send cancellation message to agent
            await self.message_broker.send_message(
                from_agent="execution_engine",
                to_agent=str(exec_context.agent_id),
                message_type="task_execution_cancel",
                payload={
                    "task_id": str(task_id),
                    "reason": reason
                }
            )
            
            # Complete with cancelled outcome
            return await self.complete_task_execution(
                task_id=task_id,
                outcome=ExecutionOutcome.CANCELLED,
                result_data={"cancellation_reason": reason},
                error_message=reason
            )
            
        except Exception as e:
            logger.error("❌ Failed to cancel task execution", task_id=str(task_id), error=str(e))
            return False
    
    async def get_execution_status(self, task_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get current execution status for a task."""
        if task_id not in self.active_executions:
            return None
        
        exec_context = self.active_executions[task_id]
        current_time = datetime.utcnow()
        elapsed_time_ms = (current_time - exec_context.started_at).total_seconds() * 1000
        
        return {
            **exec_context.to_dict(),
            "elapsed_time_ms": elapsed_time_ms,
            "is_active": True
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get execution engine performance metrics."""
        metrics = {
            "active_executions": len(self.active_executions),
            "execution_times_by_type": {},
            "success_rates_by_type": dict(self.success_rates),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Calculate statistics per task type
        for task_type, times in self.execution_times.items():
            if times:
                metrics["execution_times_by_type"][task_type] = {
                    "count": len(times),
                    "average_ms": sum(times) / len(times),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "median_ms": sorted(times)[len(times) // 2] if times else 0
                }
        
        return metrics
    
    async def _monitor_execution_timeout(self, task_id: uuid.UUID) -> None:
        """Monitor execution timeout for a task."""
        try:
            await asyncio.sleep(self.max_execution_time.total_seconds())
            
            # Check if task is still executing
            if task_id in self.active_executions:
                logger.warning("Task execution timeout", task_id=str(task_id))
                await self.complete_task_execution(
                    task_id=task_id,
                    outcome=ExecutionOutcome.TIMEOUT,
                    result_data={"timeout_minutes": self.max_execution_time.total_seconds() / 60},
                    error_message=f"Task execution timed out after {self.max_execution_time.total_seconds() / 60} minutes"
                )
                
        except asyncio.CancelledError:
            # Task completed before timeout
            pass
        except Exception as e:
            logger.error("Error in timeout monitoring", task_id=str(task_id), error=str(e))
    
    async def _publish_execution_event(
        self,
        event_type: str,
        task_id: uuid.UUID,
        agent_id: uuid.UUID,
        payload: Dict[str, Any]
    ) -> None:
        """Publish execution event to Redis streams."""
        try:
            event_data = {
                "event_type": event_type,
                "task_id": str(task_id),
                "agent_id": str(agent_id),
                "timestamp": datetime.utcnow().isoformat(),
                "payload": payload
            }
            
            # Publish to system events stream
            await self.redis.xadd(
                "system_events:task_execution",
                event_data,
                maxlen=10000
            )
            
            # Also publish to real-time pub/sub for dashboard
            await self.redis.publish(
                "realtime:task_execution",
                json.dumps(event_data)
            )
            
        except Exception as e:
            logger.error("Failed to publish execution event", event_type=event_type, error=str(e))