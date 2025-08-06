"""
AI Task Worker for LeanVibe Agent Hive 2.0

Autonomous development worker that processes tasks from the queue using AI models.
Connects the Task Queue with the AI Gateway for real autonomous development.

Based on Gemini CLI strategic analysis recommendations for asynchronous workers.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from .database import get_session
from .ai_gateway import get_ai_gateway, AIRequest, AIResponse, AIModel, TaskType as AITaskType
from .task_queue import TaskQueue
from ..models.task import Task, TaskStatus, TaskType, TaskPriority
from ..models.agent import Agent, AgentStatus

logger = structlog.get_logger()


class WorkerStatus(Enum):
    """AI Worker status values."""
    IDLE = "idle"
    PROCESSING = "processing"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class AITaskResult:
    """Result from AI task processing."""
    task_id: uuid.UUID
    success: bool
    result_data: Dict[str, Any]
    ai_response: Optional[AIResponse] = None
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0


class AITaskWorker:
    """
    Autonomous AI worker that processes development tasks using AI models.
    
    Features:
    - Task queue polling with capability matching
    - AI model integration for code generation, review, testing
    - Task lifecycle management (pending -> in_progress -> completed/failed)
    - Error handling with retry logic
    - Performance monitoring and metrics
    - Support for long-running autonomous development workflows
    """
    
    def __init__(
        self,
        worker_id: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        ai_model: AIModel = AIModel.CLAUDE_3_5_SONNET
    ):
        self.worker_id = worker_id or f"ai_worker_{uuid.uuid4().hex[:8]}"
        self.capabilities = capabilities or [
            "code_generation",
            "code_review", 
            "testing",
            "documentation",
            "architecture",
            "debugging"
        ]
        self.ai_model = ai_model
        
        self.status = WorkerStatus.IDLE
        self.task_queue: Optional[TaskQueue] = None
        self.ai_gateway = None
        self.current_task: Optional[Task] = None
        
        # Worker configuration
        self.poll_interval_seconds = 5.0
        self.max_processing_time_minutes = 30
        self.batch_size = 1  # Process one task at a time for now
        
        # Performance tracking
        self._stats = {
            "tasks_processed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "uptime_start": datetime.utcnow()
        }
        
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the AI worker."""
        if self._running:
            logger.warning("Worker already running", worker_id=self.worker_id)
            return
        
        self._running = True
        self.status = WorkerStatus.IDLE
        
        # Initialize dependencies
        self.task_queue = TaskQueue()
        await self.task_queue.start()
        
        self.ai_gateway = await get_ai_gateway()
        
        # Register worker as agent in database
        await self._register_worker()
        
        # Start worker loop
        self._worker_task = asyncio.create_task(self._worker_loop())
        
        logger.info(
            "AI Worker started",
            worker_id=self.worker_id,
            capabilities=self.capabilities,
            ai_model=self.ai_model.value
        )
    
    async def stop(self) -> None:
        """Stop the AI worker gracefully."""
        if not self._running:
            return
        
        self._running = False
        self.status = WorkerStatus.STOPPING
        
        # Cancel worker task
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        
        # Complete current task if any
        if self.current_task:
            await self._complete_current_task_gracefully()
        
        # Stop task queue
        if self.task_queue:
            await self.task_queue.stop()
        
        # Update worker status in database
        await self._unregister_worker()
        
        logger.info("AI Worker stopped", worker_id=self.worker_id)
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get worker performance statistics."""
        uptime = (datetime.utcnow() - self._stats["uptime_start"]).total_seconds()
        
        return {
            "worker_id": self.worker_id,
            "status": self.status.value,
            "capabilities": self.capabilities,
            "ai_model": self.ai_model.value,
            "current_task_id": str(self.current_task.id) if self.current_task else None,
            "uptime_seconds": uptime,
            **self._stats,
            "tasks_per_hour": (self._stats["tasks_completed"] / (uptime / 3600)) if uptime > 0 else 0.0
        }
    
    async def _worker_loop(self) -> None:
        """Main worker processing loop."""
        while self._running:
            try:
                self.status = WorkerStatus.IDLE
                
                # Poll for new tasks
                queued_task = await self.task_queue.dequeue_task(
                    agent_capabilities=self.capabilities,
                    timeout_seconds=self.poll_interval_seconds
                )
                
                if queued_task:
                    await self._process_task(queued_task.task_id)
                else:
                    # No tasks available, wait before polling again
                    await asyncio.sleep(self.poll_interval_seconds)
                
            except asyncio.CancelledError:
                logger.info("Worker loop cancelled", worker_id=self.worker_id)
                break
            except Exception as e:
                logger.error(
                    "Worker loop error",
                    worker_id=self.worker_id,
                    error=str(e)
                )
                self.status = WorkerStatus.ERROR
                await asyncio.sleep(10)  # Error backoff
    
    async def _process_task(self, task_id: uuid.UUID) -> None:
        """Process a single task using AI."""
        start_time = datetime.utcnow()
        
        try:
            self.status = WorkerStatus.PROCESSING
            
            # Load task from database
            async with get_session() as db:
                result = await db.execute(
                    select(Task).where(Task.id == task_id)
                )
                task = result.scalar_one_or_none()
                
                if not task:
                    logger.error("Task not found", task_id=str(task_id))
                    return
                
                self.current_task = task
                
                # Update task status to in_progress
                task.start_execution()
                await db.execute(
                    update(Task)
                    .where(Task.id == task_id)
                    .values(
                        status=TaskStatus.IN_PROGRESS,
                        started_at=datetime.utcnow(),
                        assigned_agent_id=self.worker_id  # Use worker_id as agent_id
                    )
                )
                await db.commit()
            
            logger.info(
                "Processing task",
                worker_id=self.worker_id,
                task_id=str(task_id),
                task_type=task.task_type.value if task.task_type else "unknown",
                title=task.title
            )
            
            # Process task with AI
            result = await self._execute_ai_task(task)
            
            # Update task with results
            async with get_session() as db:
                if result.success:
                    task.complete_successfully(result.result_data)
                    
                    await db.execute(
                        update(Task)
                        .where(Task.id == task_id)
                        .values(
                            status=TaskStatus.COMPLETED,
                            completed_at=datetime.utcnow(),
                            result=result.result_data,
                            actual_effort=int(result.processing_time_seconds / 60)
                        )
                    )
                    
                    self._stats["tasks_completed"] += 1
                    
                    logger.info(
                        "Task completed successfully",
                        worker_id=self.worker_id,
                        task_id=str(task_id),
                        processing_time=result.processing_time_seconds
                    )
                    
                else:
                    task.fail_with_error(result.error_message or "AI processing failed")
                    
                    await db.execute(
                        update(Task)
                        .where(Task.id == task_id)
                        .values(
                            status=TaskStatus.FAILED,
                            error_message=result.error_message,
                            retry_count=task.retry_count,
                            actual_effort=int(result.processing_time_seconds / 60)
                        )
                    )
                    
                    self._stats["tasks_failed"] += 1
                    
                    logger.error(
                        "Task failed",
                        worker_id=self.worker_id,
                        task_id=str(task_id),
                        error=result.error_message
                    )
                
                await db.commit()
            
            # Update performance stats
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self._stats["tasks_processed"] += 1
            self._stats["total_processing_time"] += processing_time
            self._stats["average_processing_time"] = (
                self._stats["total_processing_time"] / self._stats["tasks_processed"]
            )
            
        except Exception as e:
            logger.error(
                "Task processing error",
                worker_id=self.worker_id,
                task_id=str(task_id),
                error=str(e)
            )
            
            # Mark task as failed
            try:
                async with get_session() as db:
                    await db.execute(
                        update(Task)
                        .where(Task.id == task_id)
                        .values(
                            status=TaskStatus.FAILED,
                            error_message=f"Worker error: {str(e)}",
                            actual_effort=int((datetime.utcnow() - start_time).total_seconds() / 60)
                        )
                    )
                    await db.commit()
                    
                self._stats["tasks_failed"] += 1
            except Exception as db_error:
                logger.error("Failed to update task status", error=str(db_error))
        
        finally:
            self.current_task = None
    
    async def _execute_ai_task(self, task: Task) -> AITaskResult:
        """Execute a task using AI models."""
        start_time = datetime.utcnow()
        
        try:
            # Map task type to AI task type
            ai_task_type = self._map_task_type(task.task_type)
            
            # Create AI request
            ai_request = AIRequest(
                task_id=str(task.id),
                agent_id=self.worker_id,
                task_type=ai_task_type,
                prompt=self._build_task_prompt(task),
                model=self.ai_model,
                context=task.context or {},
                max_tokens=4096,
                temperature=0.7,
                timeout_seconds=self.max_processing_time_minutes * 60
            )
            
            # Execute AI request
            ai_response = await self.ai_gateway.generate(ai_request)
            
            # Process AI response into task result
            result_data = self._process_ai_response(task, ai_response)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return AITaskResult(
                task_id=task.id,
                success=True,
                result_data=result_data,
                ai_response=ai_response,
                processing_time_seconds=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return AITaskResult(
                task_id=task.id,
                success=False,
                result_data={},
                error_message=str(e),
                processing_time_seconds=processing_time
            )
    
    def _map_task_type(self, task_type: Optional[TaskType]) -> AITaskType:
        """Map database task type to AI task type."""
        mapping = {
            TaskType.FEATURE_DEVELOPMENT: AITaskType.CODE_GENERATION,
            TaskType.BUG_FIX: AITaskType.DEBUGGING,
            TaskType.REFACTORING: AITaskType.CODE_GENERATION,
            TaskType.TESTING: AITaskType.TESTING,
            TaskType.DOCUMENTATION: AITaskType.DOCUMENTATION,
            TaskType.ARCHITECTURE: AITaskType.ARCHITECTURE,
            TaskType.CODE_REVIEW: AITaskType.CODE_REVIEW,
            TaskType.CODE_GENERATION: AITaskType.CODE_GENERATION,
        }
        
        return mapping.get(task_type, AITaskType.CODE_GENERATION)
    
    def _build_task_prompt(self, task: Task) -> str:
        """Build AI prompt from task description and context."""
        base_prompt = f"""
Task: {task.title}

Description:
{task.description or 'No description provided'}

Task Type: {task.task_type.value if task.task_type else 'general'}
Priority: {task.priority.name}
"""
        
        # Add context if available
        if task.context:
            context_str = "\n\nContext:\n"
            for key, value in task.context.items():
                context_str += f"- {key}: {value}\n"
            base_prompt += context_str
        
        # Add requirements if available
        if task.required_capabilities:
            base_prompt += f"\n\nRequired Capabilities: {', '.join(task.required_capabilities)}"
        
        # Add specific instructions based on task type
        if task.task_type == TaskType.CODE_GENERATION:
            base_prompt += """

Please provide:
1. Complete, production-ready code
2. Proper error handling and validation
3. Comprehensive comments and documentation
4. Unit tests if applicable
5. Usage examples
"""
        elif task.task_type == TaskType.CODE_REVIEW:
            base_prompt += """

Please provide:
1. Code quality assessment
2. Security vulnerability analysis
3. Performance considerations
4. Maintainability recommendations
5. Specific improvement suggestions
"""
        elif task.task_type == TaskType.TESTING:
            base_prompt += """

Please provide:
1. Comprehensive test cases
2. Edge case coverage
3. Integration test scenarios
4. Test data setup/teardown
5. Performance test considerations
"""
        elif task.task_type == TaskType.DOCUMENTATION:
            base_prompt += """

Please provide:
1. Clear, comprehensive documentation
2. Usage examples with code
3. API reference if applicable
4. Installation/setup instructions
5. Common troubleshooting scenarios
"""
        
        return base_prompt
    
    def _process_ai_response(self, task: Task, ai_response: AIResponse) -> Dict[str, Any]:
        """Process AI response into structured task result."""
        return {
            "ai_generated_content": ai_response.content,
            "task_type": task.task_type.value if task.task_type else "unknown",
            "ai_model_used": ai_response.model.value,
            "processing_duration_seconds": ai_response.duration_seconds,
            "ai_usage_stats": ai_response.usage,
            "estimated_cost": ai_response.cost_estimate,
            "quality_metrics": {
                "content_length": len(ai_response.content),
                "response_time": ai_response.duration_seconds,
                "model_confidence": "high" if ai_response.duration_seconds < 30 else "medium"
            },
            "metadata": {
                "worker_id": self.worker_id,
                "processing_timestamp": datetime.utcnow().isoformat(),
                "ai_model": ai_response.model.value,
                "request_id": ai_response.request_id
            }
        }
    
    async def _register_worker(self) -> None:
        """Register worker as an agent in the database."""
        try:
            async with get_session() as db:
                # Check if worker already exists
                result = await db.execute(
                    select(Agent).where(Agent.id == self.worker_id)
                )
                existing_agent = result.scalar_one_or_none()
                
                if not existing_agent:
                    # Create new agent record
                    agent = Agent(
                        id=self.worker_id,
                        name=f"AI Worker {self.worker_id}",
                        agent_type="ai_worker",
                        status=AgentStatus.active,
                        capabilities=self.capabilities,
                        configuration={
                            "ai_model": self.ai_model.value,
                            "worker_type": "autonomous_development",
                            "max_processing_time_minutes": self.max_processing_time_minutes
                        }
                    )
                    db.add(agent)
                else:
                    # Update existing agent
                    await db.execute(
                        update(Agent)
                        .where(Agent.id == self.worker_id)
                        .values(
                            status=AgentStatus.active,
                            capabilities=self.capabilities,
                            last_heartbeat=datetime.utcnow()
                        )
                    )
                
                await db.commit()
                
        except Exception as e:
            logger.error("Failed to register worker", worker_id=self.worker_id, error=str(e))
    
    async def _unregister_worker(self) -> None:
        """Unregister worker from database."""
        try:
            async with get_session() as db:
                await db.execute(
                    update(Agent)
                    .where(Agent.id == self.worker_id)
                    .values(
                        status=AgentStatus.inactive,
                        last_heartbeat=datetime.utcnow()
                    )
                )
                await db.commit()
                
        except Exception as e:
            logger.error("Failed to unregister worker", worker_id=self.worker_id, error=str(e))
    
    async def _complete_current_task_gracefully(self) -> None:
        """Complete current task gracefully during shutdown."""
        if not self.current_task:
            return
        
        try:
            async with get_session() as db:
                await db.execute(
                    update(Task)
                    .where(Task.id == self.current_task.id)
                    .values(
                        status=TaskStatus.FAILED,
                        error_message="Worker stopped during processing",
                        actual_effort=int((datetime.utcnow() - (self.current_task.started_at or datetime.utcnow())).total_seconds() / 60)
                    )
                )
                await db.commit()
                
        except Exception as e:
            logger.error("Failed to complete current task gracefully", error=str(e))


# Global worker registry
_workers: Dict[str, AITaskWorker] = {}


async def create_ai_worker(
    worker_id: Optional[str] = None,
    capabilities: Optional[List[str]] = None,
    ai_model: AIModel = AIModel.CLAUDE_3_5_SONNET
) -> AITaskWorker:
    """Create and start a new AI task worker."""
    worker = AITaskWorker(
        worker_id=worker_id,
        capabilities=capabilities,
        ai_model=ai_model
    )
    
    await worker.start()
    _workers[worker.worker_id] = worker
    
    return worker


async def stop_ai_worker(worker_id: str) -> bool:
    """Stop a specific AI worker."""
    if worker_id in _workers:
        await _workers[worker_id].stop()
        del _workers[worker_id]
        return True
    return False


async def get_worker_stats() -> Dict[str, Any]:
    """Get statistics for all active workers."""
    stats = {}
    for worker_id, worker in _workers.items():
        stats[worker_id] = await worker.get_stats()
    
    return {
        "active_workers": len(_workers),
        "worker_details": stats,
        "total_tasks_processed": sum(s.get("tasks_processed", 0) for s in stats.values()),
        "total_tasks_completed": sum(s.get("tasks_completed", 0) for s in stats.values()),
        "total_tasks_failed": sum(s.get("tasks_failed", 0) for s in stats.values())
    }


async def stop_all_workers() -> None:
    """Stop all active AI workers."""
    tasks = []
    for worker in _workers.values():
        tasks.append(worker.stop())
    
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    
    _workers.clear()