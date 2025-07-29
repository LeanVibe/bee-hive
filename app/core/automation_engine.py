"""
VS 7.2: Automation Engine for Distributed Consolidation Coordination - LeanVibe Agent Hive 2.0 Phase 5.3

Advanced automation engine providing distributed task coordination with comprehensive safety controls.
Manages automated consolidation operations across multiple agents with shadow mode validation.

Features:
- Distributed task coordination with Redis-based locking
- Shadow mode execution for validation before live deployment
- Concurrency limits and circuit breaker integration
- Automated rollback triggers based on performance metrics
- Real-time safety monitoring and emergency stop capabilities
- Integration with Smart Scheduler for decision execution
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import UUID, uuid4
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, text
from sqlalchemy.orm import selectinload

from ..core.database import get_async_session
from ..core.redis import get_redis
from ..core.config import get_settings
from ..core.circuit_breaker import CircuitBreaker
from ..core.smart_scheduler import get_smart_scheduler, SchedulingDecisionResult, AutomationTier
from ..core.sleep_wake_manager import get_sleep_wake_manager
from ..models.agent import Agent
from ..models.sleep_wake import SleepState


logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution modes for automation engine."""
    SHADOW = "shadow"                # Log only, no actual execution
    LIVE = "live"                   # Full execution
    VALIDATION = "validation"       # Execute with enhanced monitoring


class AutomationStatus(Enum):
    """Status of automation operations."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"


class TaskType(Enum):
    """Types of automation tasks."""
    CONSOLIDATION = "consolidation"
    WAKE = "wake"
    HEALTH_CHECK = "health_check"
    ROLLBACK = "rollback"


class TaskPriority(Enum):
    """Task priorities for execution ordering."""
    EMERGENCY = "emergency"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


@dataclass
class AutomationTask:
    """Represents an automation task."""
    id: str
    task_type: TaskType
    priority: TaskPriority
    agent_id: UUID
    created_at: datetime
    scheduled_at: Optional[datetime] = None
    attempts: int = 0
    max_attempts: int = 3
    timeout_seconds: int = 300
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecutionResult:
    """Result of task execution."""
    task_id: str
    success: bool
    execution_time_ms: float
    error_message: Optional[str] = None
    rollback_required: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AutomationMetrics:
    """Metrics for automation engine performance."""
    total_tasks_executed: int
    successful_tasks: int
    failed_tasks: int
    avg_execution_time_ms: float
    current_concurrency: int
    queue_depth: int
    success_rate: float
    tasks_per_minute: float
    rollback_count: int
    emergency_stops: int


class AutomationEngine:
    """
    Advanced automation engine for distributed consolidation coordination.
    
    Core Features:
    - Distributed task coordination with Redis locking
    - Shadow mode validation and gradual rollout
    - Concurrency control and circuit breaker protection
    - Automatic rollback triggers and emergency stops
    - Real-time performance monitoring and alerting
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Core configuration
        self.execution_mode = ExecutionMode.SHADOW  # Start in shadow mode
        self.status = AutomationStatus.IDLE
        self.enabled = False
        
        # Concurrency and safety configuration
        self.max_concurrent_tasks = 5
        self.max_consolidations_per_minute = 10
        self.emergency_stop_error_threshold = 0.2  # 20% error rate triggers stop
        self.rollback_latency_threshold_ms = 5000  # Rollback if >5s response time
        
        # Distributed coordination
        self.coordination_lock_timeout = 300  # 5 minutes
        self.task_distribution_enabled = True
        self.leader_election_enabled = True
        
        # Internal state
        self._task_queue: deque = deque()
        self._active_tasks: Dict[str, AutomationTask] = {}
        self._execution_history: deque = deque(maxlen=1000)
        self._performance_metrics: AutomationMetrics = AutomationMetrics(
            total_tasks_executed=0,
            successful_tasks=0,
            failed_tasks=0,
            avg_execution_time_ms=0,
            current_concurrency=0,
            queue_depth=0,
            success_rate=1.0,
            tasks_per_minute=0,
            rollback_count=0,
            emergency_stops=0
        )
        
        # Coordination state
        self._is_leader = False
        self._leader_heartbeat_task = None
        self._consolidation_timestamps: deque = deque(maxlen=100)
        
        # Circuit breakers
        self._execution_circuit_breaker = CircuitBreaker(
            name="automation_execution",
            failure_threshold=5,
            timeout_seconds=300
        )
        
        self._coordination_circuit_breaker = CircuitBreaker(
            name="distributed_coordination",
            failure_threshold=3,
            timeout_seconds=180
        )
    
    async def initialize(self) -> None:
        """Initialize the automation engine."""
        try:
            logger.info("Initializing Automation Engine VS 7.2")
            
            # Load configuration
            await self._load_configuration()
            
            # Initialize distributed coordination
            if self.leader_election_enabled:
                await self._initialize_leader_election()
            
            # Start background tasks
            asyncio.create_task(self._task_processor())
            asyncio.create_task(self._performance_monitor())
            asyncio.create_task(self._safety_monitor())
            asyncio.create_task(self._cleanup_completed_tasks())
            
            self.status = AutomationStatus.IDLE
            logger.info("Automation Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Automation Engine: {e}")
            raise
    
    async def execute_scheduling_decision(
        self,
        decision: SchedulingDecisionResult,
        priority: TaskPriority = TaskPriority.NORMAL,
        execution_mode: Optional[ExecutionMode] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute a scheduling decision through the automation engine.
        
        Args:
            decision: The scheduling decision to execute
            priority: Task priority for execution ordering
            execution_mode: Override global execution mode
            
        Returns:
            (queued_successfully, task_info)
        """
        try:
            if not self.enabled and execution_mode != ExecutionMode.SHADOW:
                return False, {"error": "Automation engine disabled"}
            
            if self.status == AutomationStatus.EMERGENCY_STOP:
                return False, {"error": "Emergency stop active"}
            
            # Determine task type from decision
            task_type_map = {
                "consolidate_agent": TaskType.CONSOLIDATION,
                "wake_agent": TaskType.WAKE,
                "maintain_status": None,  # No task needed
                "defer_decision": None    # No task needed
            }
            
            task_type = task_type_map.get(decision.decision.value)
            if task_type is None:
                return True, {"action": "no_task_required", "decision": decision.decision.value}
            
            # Create automation task
            task = AutomationTask(
                id=str(uuid4()),
                task_type=task_type,
                priority=priority,
                agent_id=decision.agent_id,
                created_at=datetime.utcnow(),
                metadata={
                    "decision": asdict(decision),
                    "execution_mode": (execution_mode or self.execution_mode).value,
                    "requested_by": "smart_scheduler"
                }
            )
            
            # Queue the task
            queued = await self._queue_task(task)
            
            if queued:
                return True, {
                    "task_id": task.id,
                    "task_type": task.task_type.value,
                    "priority": task.priority.value,
                    "queue_position": len(self._task_queue)
                }
            else:
                return False, {"error": "Failed to queue task"}
            
        except Exception as e:
            logger.error(f"Error executing scheduling decision: {e}")
            return False, {"error": str(e)}
    
    async def execute_bulk_operation(
        self,
        agent_ids: List[UUID],
        operation_type: TaskType,
        coordination_strategy: str = "sequential",
        max_concurrent: int = 3
    ) -> Dict[str, Any]:
        """
        Execute bulk operations with distributed coordination.
        
        Args:
            agent_ids: List of agent IDs to operate on
            operation_type: Type of operation to perform
            coordination_strategy: How to coordinate execution
            max_concurrent: Maximum concurrent operations
            
        Returns:
            Operation summary with results
        """
        operation_id = str(uuid4())
        
        try:
            logger.info(f"Starting bulk operation {operation_id}: {operation_type.value} on {len(agent_ids)} agents")
            
            # Validate operation
            if not self._is_leader and self.leader_election_enabled:
                return {"error": "Not the coordination leader"}
            
            if self.status == AutomationStatus.EMERGENCY_STOP:
                return {"error": "Emergency stop active"}
            
            # Acquire distributed lock for bulk operation
            redis = await get_redis()
            lock_key = f"automation_bulk_lock:{operation_id}"
            
            async with self._coordination_circuit_breaker:
                lock_acquired = await redis.set(
                    lock_key, 
                    json.dumps({"operation_id": operation_id, "timestamp": datetime.utcnow().isoformat()}),
                    nx=True, 
                    ex=self.coordination_lock_timeout
                )
                
                if not lock_acquired:
                    return {"error": "Could not acquire coordination lock"}
            
            try:
                # Create tasks for all agents
                tasks = []
                for agent_id in agent_ids:
                    task = AutomationTask(
                        id=str(uuid4()),
                        task_type=operation_type,
                        priority=TaskPriority.HIGH,
                        agent_id=agent_id,
                        created_at=datetime.utcnow(),
                        metadata={
                            "bulk_operation_id": operation_id,
                            "coordination_strategy": coordination_strategy,
                            "execution_mode": self.execution_mode.value
                        }
                    )
                    tasks.append(task)
                
                # Execute based on coordination strategy
                if coordination_strategy == "sequential":
                    results = await self._execute_sequential(tasks)
                elif coordination_strategy == "parallel":
                    results = await self._execute_parallel(tasks, max_concurrent)
                elif coordination_strategy == "staged":
                    results = await self._execute_staged(tasks, max_concurrent)
                else:
                    return {"error": f"Unknown coordination strategy: {coordination_strategy}"}
                
                # Compile results
                successful = len([r for r in results if r.success])
                failed = len(results) - successful
                
                return {
                    "operation_id": operation_id,
                    "total_tasks": len(tasks),
                    "successful": successful,
                    "failed": failed,
                    "success_rate": successful / len(tasks) if tasks else 0,
                    "results": [asdict(r) for r in results],
                    "coordination_strategy": coordination_strategy,
                    "completed_at": datetime.utcnow().isoformat()
                }
                
            finally:
                # Release coordination lock
                await redis.delete(lock_key)
            
        except Exception as e:
            logger.error(f"Error in bulk operation {operation_id}: {e}")
            return {"error": str(e), "operation_id": operation_id}
    
    async def trigger_emergency_stop(self, reason: str) -> bool:
        """Trigger emergency stop of all automation."""
        try:
            logger.critical(f"EMERGENCY STOP triggered: {reason}")
            
            self.status = AutomationStatus.EMERGENCY_STOP
            self._performance_metrics.emergency_stops += 1
            
            # Cancel all active tasks
            cancelled_tasks = []
            for task_id, task in self._active_tasks.items():
                cancelled_tasks.append(task_id)
            
            self._active_tasks.clear()
            
            # Persist emergency stop state
            redis = await get_redis()
            await redis.hset(
                "automation_engine_state",
                mapping={
                    "status": self.status.value,
                    "emergency_stop_reason": reason,
                    "emergency_stop_time": datetime.utcnow().isoformat(),
                    "cancelled_tasks": json.dumps(cancelled_tasks)
                }
            )
            
            logger.info(f"Emergency stop completed, cancelled {len(cancelled_tasks)} tasks")
            return True
            
        except Exception as e:
            logger.error(f"Error triggering emergency stop: {e}")
            return False
    
    async def resume_automation(self, authorization_key: str) -> bool:
        """Resume automation after emergency stop."""
        try:
            # Validate authorization (simplified for demo)
            if authorization_key != "emergency_resume_key":
                return False
            
            logger.info("Resuming automation after emergency stop")
            
            self.status = AutomationStatus.IDLE
            
            # Clear emergency stop state
            redis = await get_redis()
            await redis.hdel("automation_engine_state", "emergency_stop_reason", "emergency_stop_time")
            await redis.hset("automation_engine_state", "status", self.status.value)
            
            return True
            
        except Exception as e:
            logger.error(f"Error resuming automation: {e}")
            return False
    
    async def get_automation_status(self) -> Dict[str, Any]:
        """Get comprehensive automation engine status."""
        try:
            # Current status
            status_info = {
                "status": self.status.value,
                "execution_mode": self.execution_mode.value,
                "enabled": self.enabled,
                "is_leader": self._is_leader
            }
            
            # Performance metrics
            performance_info = asdict(self._performance_metrics)
            
            # Queue information
            queue_info = {
                "total_queued": len(self._task_queue),
                "active_tasks": len(self._active_tasks),
                "tasks_by_priority": self._get_queue_by_priority(),
                "tasks_by_type": self._get_queue_by_type()
            }
            
            # Circuit breaker status
            circuit_breaker_info = {
                "execution_circuit_breaker": {
                    "state": self._execution_circuit_breaker.state,
                    "failure_count": self._execution_circuit_breaker.failure_count,
                    "success_count": self._execution_circuit_breaker.success_count
                },
                "coordination_circuit_breaker": {
                    "state": self._coordination_circuit_breaker.state,
                    "failure_count": self._coordination_circuit_breaker.failure_count,
                    "success_count": self._coordination_circuit_breaker.success_count
                }
            }
            
            # Safety metrics
            safety_info = {
                "recent_consolidations": len([
                    ts for ts in self._consolidation_timestamps
                    if ts > datetime.utcnow() - timedelta(minutes=5)
                ]),
                "error_rate": 1.0 - self._performance_metrics.success_rate,
                "meets_safety_threshold": (1.0 - self._performance_metrics.success_rate) < self.emergency_stop_error_threshold
            }
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "status": status_info,
                "performance": performance_info,
                "queue": queue_info,
                "circuit_breakers": circuit_breaker_info,
                "safety": safety_info
            }
            
        except Exception as e:
            logger.error(f"Error getting automation status: {e}")
            return {"error": str(e)}
    
    async def update_configuration(self, config: Dict[str, Any]) -> bool:
        """Update automation engine configuration."""
        try:
            # Validate and apply configuration changes
            if "execution_mode" in config:
                try:
                    self.execution_mode = ExecutionMode(config["execution_mode"])
                except ValueError:
                    return False
            
            if "enabled" in config:
                self.enabled = bool(config["enabled"])
            
            if "max_concurrent_tasks" in config:
                self.max_concurrent_tasks = max(1, int(config["max_concurrent_tasks"]))
            
            if "max_consolidations_per_minute" in config:
                self.max_consolidations_per_minute = max(1, int(config["max_consolidations_per_minute"]))
            
            # Persist configuration
            redis = await get_redis()
            await redis.hset(
                "automation_engine_config",
                mapping={
                    "execution_mode": self.execution_mode.value,
                    "enabled": json.dumps(self.enabled),
                    "max_concurrent_tasks": str(self.max_concurrent_tasks),
                    "max_consolidations_per_minute": str(self.max_consolidations_per_minute),
                    "updated_at": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Automation engine configuration updated: {config}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False
    
    # Internal methods
    
    async def _load_configuration(self) -> None:
        """Load configuration from persistent storage."""
        try:
            redis = await get_redis()
            config = await redis.hgetall("automation_engine_config")
            
            if config:
                self.execution_mode = ExecutionMode(config.get("execution_mode", "shadow"))
                self.enabled = json.loads(config.get("enabled", "false"))
                self.max_concurrent_tasks = int(config.get("max_concurrent_tasks", "5"))
                self.max_consolidations_per_minute = int(config.get("max_consolidations_per_minute", "10"))
            
            # Load state
            state = await redis.hgetall("automation_engine_state")
            if state:
                self.status = AutomationStatus(state.get("status", "idle"))
            
        except Exception as e:
            logger.warning(f"Could not load automation configuration, using defaults: {e}")
    
    async def _initialize_leader_election(self) -> None:
        """Initialize leader election for distributed coordination."""
        try:
            # Simple leader election using Redis
            redis = await get_redis()
            leader_key = "automation_engine_leader"
            
            # Try to become leader
            leader_acquired = await redis.set(
                leader_key,
                json.dumps({"node_id": "automation_engine", "timestamp": datetime.utcnow().isoformat()}),
                nx=True,
                ex=60  # 1 minute TTL
            )
            
            self._is_leader = leader_acquired
            
            if self._is_leader:
                logger.info("Became automation engine leader")
                # Start heartbeat
                self._leader_heartbeat_task = asyncio.create_task(self._leader_heartbeat())
            else:
                logger.info("Not the automation engine leader")
            
        except Exception as e:
            logger.error(f"Error in leader election: {e}")
            self._is_leader = False
    
    async def _leader_heartbeat(self) -> None:
        """Maintain leadership with regular heartbeats."""
        while self._is_leader:
            try:
                redis = await get_redis()
                await redis.set(
                    "automation_engine_leader",
                    json.dumps({"node_id": "automation_engine", "timestamp": datetime.utcnow().isoformat()}),
                    ex=60
                )
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in leader heartbeat: {e}")
                self._is_leader = False
                break
    
    async def _queue_task(self, task: AutomationTask) -> bool:
        """Queue a task for execution."""
        try:
            # Check queue limits
            if len(self._task_queue) > 1000:  # Prevent queue overflow
                logger.warning("Task queue full, rejecting new task")
                return False
            
            # Insert task based on priority
            if task.priority == TaskPriority.EMERGENCY:
                self._task_queue.appendleft(task)
            else:
                # Find correct position based on priority
                inserted = False
                for i, queued_task in enumerate(self._task_queue):
                    if self._priority_value(task.priority) > self._priority_value(queued_task.priority):
                        self._task_queue.insert(i, task)
                        inserted = True
                        break
                
                if not inserted:
                    self._task_queue.append(task)
            
            logger.debug(f"Queued task {task.id} with priority {task.priority.value}")
            return True
            
        except Exception as e:
            logger.error(f"Error queuing task: {e}")
            return False
    
    def _priority_value(self, priority: TaskPriority) -> int:
        """Get numeric value for priority comparison."""
        priority_values = {
            TaskPriority.EMERGENCY: 4,
            TaskPriority.HIGH: 3,
            TaskPriority.NORMAL: 2,
            TaskPriority.LOW: 1
        }
        return priority_values.get(priority, 0)
    
    async def _task_processor(self) -> None:
        """Main task processing loop."""
        while True:
            try:
                if self.status == AutomationStatus.EMERGENCY_STOP:
                    await asyncio.sleep(10)
                    continue
                
                if not self.enabled or self.status == AutomationStatus.PAUSED:
                    await asyncio.sleep(5)
                    continue
                
                # Check if we can process more tasks
                if len(self._active_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(1)
                    continue
                
                # Get next task
                if not self._task_queue:
                    await asyncio.sleep(1)
                    continue
                
                task = self._task_queue.popleft()
                
                # Execute task
                asyncio.create_task(self._execute_task(task))
                
            except Exception as e:
                logger.error(f"Error in task processor: {e}")
                await asyncio.sleep(5)
    
    async def _execute_task(self, task: AutomationTask) -> None:
        """Execute a single automation task."""
        execution_start = time.time()
        
        try:
            # Add to active tasks
            self._active_tasks[task.id] = task
            self.status = AutomationStatus.RUNNING
            
            # Determine execution mode
            execution_mode = ExecutionMode(task.metadata.get("execution_mode", self.execution_mode.value))
            
            # Execute based on task type
            if task.task_type == TaskType.CONSOLIDATION:
                success = await self._execute_consolidation_task(task, execution_mode)
            elif task.task_type == TaskType.WAKE:
                success = await self._execute_wake_task(task, execution_mode)
            elif task.task_type == TaskType.HEALTH_CHECK:
                success = await self._execute_health_check_task(task, execution_mode)
            elif task.task_type == TaskType.ROLLBACK:
                success = await self._execute_rollback_task(task, execution_mode)
            else:
                success = False
                logger.error(f"Unknown task type: {task.task_type}")
            
            execution_time_ms = (time.time() - execution_start) * 1000
            
            # Create execution result
            result = ExecutionResult(
                task_id=task.id,
                success=success,
                execution_time_ms=execution_time_ms,
                metadata={
                    "task_type": task.task_type.value,
                    "agent_id": str(task.agent_id),
                    "execution_mode": execution_mode.value
                }
            )
            
            # Record result
            self._execution_history.append(result)
            
            # Update metrics
            await self._update_performance_metrics(result)
            
            # Check for rollback conditions
            if success and execution_time_ms > self.rollback_latency_threshold_ms:
                logger.warning(f"Task {task.id} exceeded latency threshold, may trigger rollback")
            
        except Exception as e:
            logger.error(f"Error executing task {task.id}: {e}")
            
            result = ExecutionResult(
                task_id=task.id,
                success=False,
                execution_time_ms=(time.time() - execution_start) * 1000,
                error_message=str(e),
                rollback_required=True
            )
            
            self._execution_history.append(result)
            await self._update_performance_metrics(result)
            
        finally:
            # Remove from active tasks
            self._active_tasks.pop(task.id, None)
            
            # Update status
            if not self._active_tasks:
                self.status = AutomationStatus.IDLE
    
    async def _execute_consolidation_task(self, task: AutomationTask, execution_mode: ExecutionMode) -> bool:
        """Execute a consolidation task."""
        try:
            if execution_mode == ExecutionMode.SHADOW:
                logger.info(f"SHADOW MODE: Would consolidate agent {task.agent_id}")
                await asyncio.sleep(0.1)  # Simulate execution time
                return True
            
            # Check rate limits
            recent_consolidations = len([
                ts for ts in self._consolidation_timestamps
                if ts > datetime.utcnow() - timedelta(minutes=1)
            ])
            
            if recent_consolidations >= self.max_consolidations_per_minute:
                logger.warning("Consolidation rate limit exceeded")
                return False
            
            # Execute consolidation
            sleep_manager = await get_sleep_wake_manager()
            success = await sleep_manager.initiate_sleep_cycle(
                agent_id=task.agent_id,
                cycle_type="automated"
            )
            
            if success:
                self._consolidation_timestamps.append(datetime.utcnow())
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing consolidation task: {e}")
            return False
    
    async def _execute_wake_task(self, task: AutomationTask, execution_mode: ExecutionMode) -> bool:
        """Execute a wake task."""
        try:
            if execution_mode == ExecutionMode.SHADOW:
                logger.info(f"SHADOW MODE: Would wake agent {task.agent_id}")
                await asyncio.sleep(0.1)  # Simulate execution time
                return True
            
            # Execute wake
            sleep_manager = await get_sleep_wake_manager()
            success = await sleep_manager.initiate_wake_cycle(task.agent_id)
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing wake task: {e}")
            return False
    
    async def _execute_health_check_task(self, task: AutomationTask, execution_mode: ExecutionMode) -> bool:
        """Execute a health check task."""
        try:
            # Check agent health
            async with get_async_session() as session:
                agent = await session.get(Agent, task.agent_id)
                if agent:
                    # Simplified health check
                    return agent.current_sleep_state in [SleepState.AWAKE, SleepState.SLEEPING]
                return False
                
        except Exception as e:
            logger.error(f"Error executing health check task: {e}")
            return False
    
    async def _execute_rollback_task(self, task: AutomationTask, execution_mode: ExecutionMode) -> bool:
        """Execute a rollback task."""
        try:
            if execution_mode == ExecutionMode.SHADOW:
                logger.info(f"SHADOW MODE: Would rollback operations for agent {task.agent_id}")
                return True
            
            # Execute rollback (simplified)
            sleep_manager = await get_sleep_wake_manager()
            
            # If agent was consolidated, wake it up
            async with get_async_session() as session:
                agent = await session.get(Agent, task.agent_id)
                if agent and agent.current_sleep_state == SleepState.SLEEPING:
                    success = await sleep_manager.initiate_wake_cycle(task.agent_id)
                    self._performance_metrics.rollback_count += 1
                    return success
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing rollback task: {e}")
            return False
    
    async def _execute_sequential(self, tasks: List[AutomationTask]) -> List[ExecutionResult]:
        """Execute tasks sequentially."""
        results = []
        
        for task in tasks:
            # Execute task and wait for completion
            await self._execute_task(task)
            
            # Find the result in execution history
            result = None
            for r in reversed(self._execution_history):
                if r.task_id == task.id:
                    result = r
                    break
            
            if result:
                results.append(result)
            else:
                # Create a failure result if not found
                results.append(ExecutionResult(
                    task_id=task.id,
                    success=False,
                    execution_time_ms=0,
                    error_message="Result not found"
                ))
        
        return results
    
    async def _execute_parallel(self, tasks: List[AutomationTask], max_concurrent: int) -> List[ExecutionResult]:
        """Execute tasks in parallel with concurrency limit."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(task: AutomationTask) -> ExecutionResult:
            async with semaphore:
                await self._execute_task(task)
                
                # Find the result
                for r in reversed(self._execution_history):
                    if r.task_id == task.id:
                        return r
                
                return ExecutionResult(
                    task_id=task.id,
                    success=False,
                    execution_time_ms=0,
                    error_message="Result not found"
                )
        
        # Execute all tasks
        task_coroutines = [execute_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ExecutionResult(
                    task_id=tasks[i].id,
                    success=False,
                    execution_time_ms=0,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_staged(self, tasks: List[AutomationTask], stage_size: int) -> List[ExecutionResult]:
        """Execute tasks in stages."""
        all_results = []
        
        # Split tasks into stages
        for i in range(0, len(tasks), stage_size):
            stage_tasks = tasks[i:i + stage_size]
            
            # Execute stage in parallel
            stage_results = await self._execute_parallel(stage_tasks, stage_size)
            all_results.extend(stage_results)
            
            # Check for failures that might require stopping
            stage_failures = [r for r in stage_results if not r.success]
            if len(stage_failures) > len(stage_tasks) * 0.5:  # More than 50% failed
                logger.warning(f"Stage {i // stage_size + 1} had high failure rate, stopping staged execution")
                break
        
        return all_results
    
    async def _update_performance_metrics(self, result: ExecutionResult) -> None:
        """Update performance metrics based on execution result."""
        self._performance_metrics.total_tasks_executed += 1
        
        if result.success:
            self._performance_metrics.successful_tasks += 1
        else:
            self._performance_metrics.failed_tasks += 1
        
        # Update success rate
        self._performance_metrics.success_rate = (
            self._performance_metrics.successful_tasks / self._performance_metrics.total_tasks_executed
        )
        
        # Update average execution time
        recent_results = list(self._execution_history)[-100:]  # Last 100 results
        if recent_results:
            avg_time = sum(r.execution_time_ms for r in recent_results) / len(recent_results)
            self._performance_metrics.avg_execution_time_ms = avg_time
        
        # Update current metrics
        self._performance_metrics.current_concurrency = len(self._active_tasks)
        self._performance_metrics.queue_depth = len(self._task_queue)
        
        # Calculate tasks per minute
        recent_tasks = len([
            r for r in self._execution_history
            if hasattr(r, 'timestamp') and getattr(r, 'timestamp', datetime.min) > datetime.utcnow() - timedelta(minutes=1)
        ])
        self._performance_metrics.tasks_per_minute = recent_tasks
    
    async def _performance_monitor(self) -> None:
        """Monitor performance and trigger alerts."""
        while True:
            try:
                # Check error rate
                if self._performance_metrics.success_rate < (1.0 - self.emergency_stop_error_threshold):
                    logger.critical(f"High error rate detected: {1.0 - self._performance_metrics.success_rate:.2f}")
                    await self.trigger_emergency_stop("High error rate threshold exceeded")
                
                # Check circuit breaker states
                if (self._execution_circuit_breaker.state == "open" or 
                    self._coordination_circuit_breaker.state == "open"):
                    logger.warning("Circuit breakers open, reducing automation activity")
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(60)
    
    async def _safety_monitor(self) -> None:
        """Monitor safety conditions."""
        while True:
            try:
                # Check for high latency operations
                if len(self._execution_history) > 10:
                    recent_results = list(self._execution_history)[-20:]
                    high_latency_count = len([
                        r for r in recent_results 
                        if r.execution_time_ms > self.rollback_latency_threshold_ms
                    ])
                    
                    if high_latency_count > len(recent_results) * 0.3:  # More than 30%
                        logger.warning("High latency operations detected, consider rollback")
                
                await asyncio.sleep(120)  # Monitor every 2 minutes
                
            except Exception as e:
                logger.error(f"Error in safety monitor: {e}")
                await asyncio.sleep(120)
    
    async def _cleanup_completed_tasks(self) -> None:
        """Clean up old completed tasks and results."""
        while True:
            try:
                # Clean up old execution history
                if len(self._execution_history) > 1000:
                    # Keep only the most recent 500
                    recent_results = list(self._execution_history)[-500:]
                    self._execution_history.clear()
                    self._execution_history.extend(recent_results)
                
                # Clean up old consolidation timestamps
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                while self._consolidation_timestamps and self._consolidation_timestamps[0] < cutoff_time:
                    self._consolidation_timestamps.popleft()
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(300)
    
    def _get_queue_by_priority(self) -> Dict[str, int]:
        """Get queue statistics by priority."""
        priority_counts = defaultdict(int)
        for task in self._task_queue:
            priority_counts[task.priority.value] += 1
        return dict(priority_counts)
    
    def _get_queue_by_type(self) -> Dict[str, int]:
        """Get queue statistics by task type."""
        type_counts = defaultdict(int)
        for task in self._task_queue:
            type_counts[task.task_type.value] += 1
        return dict(type_counts)


# Global instance
_automation_engine: Optional[AutomationEngine] = None


async def get_automation_engine() -> AutomationEngine:
    """Get the global automation engine instance."""
    global _automation_engine
    
    if _automation_engine is None:
        _automation_engine = AutomationEngine()
        await _automation_engine.initialize()
    
    return _automation_engine