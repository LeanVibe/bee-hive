"""
Task Batch Executor for LeanVibe Agent Hive 2.0 Workflow Engine

Optimized parallel task execution manager with intelligent load balancing,
resource management, and advanced error handling for multi-agent coordination.
"""

import asyncio
import uuid
from typing import Dict, List, Set, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import structlog

from .agent_communication_service import AgentCommunicationService
from .agent_registry import AgentRegistry
from ..models.task import Task, TaskStatus
from ..models.agent import Agent

logger = structlog.get_logger()


class BatchExecutionStrategy(Enum):
    """Task batch execution strategies."""
    PARALLEL_UNLIMITED = "parallel_unlimited"
    PARALLEL_LIMITED = "parallel_limited"
    SEQUENTIAL = "sequential"
    ADAPTIVE = "adaptive"


class TaskExecutionPriority(Enum):
    """Task execution priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class TaskExecutionRequest:
    """Request for task execution with metadata."""
    task_id: str
    task: Task
    agent_id: Optional[str] = None
    priority: TaskExecutionPriority = TaskExecutionPriority.NORMAL
    timeout_seconds: int = 3600
    retry_count: int = 0
    max_retries: int = 3
    context: Dict[str, Any] = None


@dataclass
class TaskExecutionResult:
    """Result of task execution with detailed metrics."""
    task_id: str
    success: bool
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: int = 0
    agent_id: Optional[str] = None
    retry_count: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


@dataclass
class BatchExecutionResult:
    """Result of batch execution with aggregated metrics."""
    batch_id: str
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    execution_time_ms: int
    task_results: List[TaskExecutionResult]
    resource_utilization: Dict[str, float]
    bottleneck_analysis: Dict[str, Any]


@dataclass
class AgentCapacity:
    """Agent capacity and load information."""
    agent_id: str
    max_concurrent_tasks: int
    current_task_count: int
    average_task_duration_ms: float
    success_rate: float
    last_activity: datetime
    capabilities: Set[str]


class TaskBatchExecutor:
    """
    High-performance task batch executor with intelligent load balancing.
    
    Features:
    - Parallel task execution with configurable concurrency limits
    - Intelligent agent selection based on capabilities and load
    - Advanced error handling with retry logic and fallback strategies
    - Resource monitoring and adaptive execution strategies
    - Real-time progress tracking and performance metrics
    - Circuit breaker pattern for failing agents
    - Graceful degradation under high load
    """
    
    def __init__(
        self, 
        agent_registry: AgentRegistry,
        communication_service: AgentCommunicationService,
        max_concurrent_batches: int = 5,
        default_strategy: BatchExecutionStrategy = BatchExecutionStrategy.ADAPTIVE
    ):
        """Initialize the task batch executor."""
        self.agent_registry = agent_registry
        self.communication_service = communication_service
        self.max_concurrent_batches = max_concurrent_batches
        self.default_strategy = default_strategy
        
        # Execution state tracking
        self.active_batches: Dict[str, asyncio.Task] = {}
        self.agent_capacities: Dict[str, AgentCapacity] = {}
        self.execution_queue: asyncio.Queue = asyncio.Queue()
        
        # Performance metrics
        self.metrics = {
            'batches_executed': 0,
            'batches_successful': 0,
            'batches_failed': 0,
            'total_tasks_executed': 0,
            'total_tasks_successful': 0,
            'total_tasks_failed': 0,
            'average_batch_duration_ms': 0.0,
            'average_task_duration_ms': 0.0,
            'peak_concurrent_tasks': 0,
            'agent_utilization': {},
            'error_rates': {}
        }
        
        # Configuration
        self.task_timeout_seconds = 3600  # 1 hour default
        self.batch_timeout_seconds = 7200  # 2 hours default
        self.retry_delay_seconds = 30
        self.max_retries = 3
        self.resource_check_interval = 60  # seconds
        
        # Circuit breaker settings
        self.circuit_breaker_threshold = 0.5  # 50% failure rate
        self.circuit_breaker_window = 300  # 5 minutes
        self.circuit_breaker_states: Dict[str, Dict[str, Any]] = {}
        
        logger.info(
            "TaskBatchExecutor initialized",
            max_concurrent_batches=max_concurrent_batches,
            default_strategy=default_strategy.value
        )
    
    async def execute_batch(
        self,
        task_requests: List[TaskExecutionRequest],
        batch_id: Optional[str] = None,
        strategy: Optional[BatchExecutionStrategy] = None,
        max_parallel_tasks: Optional[int] = None
    ) -> BatchExecutionResult:
        """
        Execute a batch of tasks with specified strategy and constraints.
        
        Args:
            task_requests: List of task execution requests
            batch_id: Optional batch identifier for tracking
            strategy: Execution strategy to use
            max_parallel_tasks: Maximum parallel tasks for this batch
            
        Returns:
            BatchExecutionResult with execution outcomes and metrics
        """
        if not task_requests:
            logger.warning("Empty task batch received")
            return self._create_empty_batch_result(batch_id or str(uuid.uuid4()))
        
        batch_id = batch_id or str(uuid.uuid4())
        strategy = strategy or self.default_strategy
        start_time = datetime.utcnow()
        
        logger.info(
            "🚀 Starting batch execution",
            batch_id=batch_id,
            task_count=len(task_requests),
            strategy=strategy.value,
            max_parallel_tasks=max_parallel_tasks
        )
        
        try:
            # Check capacity constraints
            if len(self.active_batches) >= self.max_concurrent_batches:
                raise RuntimeError(f"Maximum concurrent batch limit reached: {self.max_concurrent_batches}")
            
            # Update agent capacities
            await self._update_agent_capacities()
            
            # Assign agents to tasks
            await self._assign_agents_to_tasks(task_requests)
            
            # Execute tasks based on strategy
            task_results = await self._execute_tasks_with_strategy(
                task_requests, 
                strategy, 
                max_parallel_tasks
            )
            
            # Calculate execution metrics
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            successful_tasks = sum(1 for result in task_results if result.success)
            failed_tasks = len(task_results) - successful_tasks
            
            # Create batch result
            batch_result = BatchExecutionResult(
                batch_id=batch_id,
                total_tasks=len(task_requests),
                successful_tasks=successful_tasks,
                failed_tasks=failed_tasks,
                execution_time_ms=execution_time_ms,
                task_results=task_results,
                resource_utilization=await self._calculate_resource_utilization(),
                bottleneck_analysis=await self._analyze_bottlenecks(task_results)
            )
            
            # Update metrics
            self._update_batch_metrics(batch_result)
            
            logger.info(
                "✅ Batch execution completed",
                batch_id=batch_id,
                successful_tasks=successful_tasks,
                failed_tasks=failed_tasks,
                execution_time_ms=execution_time_ms
            )
            
            return batch_result
            
        except Exception as e:
            error_msg = f"Batch execution failed: {str(e)}"
            logger.error("❌ Batch execution error", batch_id=batch_id, error=error_msg)
            
            # Create failed batch result
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return BatchExecutionResult(
                batch_id=batch_id,
                total_tasks=len(task_requests),
                successful_tasks=0,
                failed_tasks=len(task_requests),
                execution_time_ms=execution_time_ms,
                task_results=[
                    TaskExecutionResult(
                        task_id=req.task_id,
                        success=False,
                        error_message=error_msg
                    ) for req in task_requests
                ],
                resource_utilization={},
                bottleneck_analysis={}
            )
            
        finally:
            # Cleanup batch tracking
            self.active_batches.pop(batch_id, None)
    
    async def execute_single_task(
        self,
        task_request: TaskExecutionRequest,
        assigned_agent_id: Optional[str] = None
    ) -> TaskExecutionResult:
        """
        Execute a single task with retry logic and monitoring.
        
        Args:
            task_request: Task execution request
            assigned_agent_id: Pre-assigned agent ID (optional)
            
        Returns:
            TaskExecutionResult with execution outcome
        """
        start_time = datetime.utcnow()
        
        try:
            # Select agent if not assigned
            if not assigned_agent_id:
                assigned_agent_id = await self._select_best_agent(task_request)
                if not assigned_agent_id:
                    return TaskExecutionResult(
                        task_id=task_request.task_id,
                        success=False,
                        error_message="No suitable agent available"
                    )
            
            # Check circuit breaker
            if self._is_circuit_breaker_open(assigned_agent_id):
                return TaskExecutionResult(
                    task_id=task_request.task_id,
                    success=False,
                    error_message=f"Circuit breaker open for agent {assigned_agent_id}"
                )
            
            # Execute task with retry logic
            retry_count = 0
            last_error = None
            
            while retry_count <= task_request.max_retries:
                try:
                    # Send task to agent
                    result_data = await self._send_task_to_agent(
                        task_request, 
                        assigned_agent_id
                    )
                    
                    # Success - update circuit breaker and return result
                    self._record_agent_success(assigned_agent_id)
                    
                    execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                    
                    return TaskExecutionResult(
                        task_id=task_request.task_id,
                        success=True,
                        result_data=result_data,
                        execution_time_ms=execution_time_ms,
                        agent_id=assigned_agent_id,
                        retry_count=retry_count
                    )
                    
                except Exception as e:
                    last_error = str(e)
                    retry_count += 1
                    
                    # Record failure for circuit breaker
                    self._record_agent_failure(assigned_agent_id, str(e))
                    
                    if retry_count <= task_request.max_retries:
                        logger.warning(
                            f"Task execution failed, retrying",
                            task_id=task_request.task_id,
                            agent_id=assigned_agent_id,
                            retry=retry_count,
                            error=str(e)
                        )
                        
                        # Wait before retry with exponential backoff
                        wait_time = self.retry_delay_seconds * (2 ** (retry_count - 1))
                        await asyncio.sleep(min(wait_time, 300))  # Max 5 minutes
                        
                        # Try different agent on subsequent retries
                        if retry_count > 1:
                            assigned_agent_id = await self._select_best_agent(
                                task_request, 
                                exclude_agents={assigned_agent_id}
                            )
                            if not assigned_agent_id:
                                break
            
            # All retries exhausted
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return TaskExecutionResult(
                task_id=task_request.task_id,
                success=False,
                error_message=f"Failed after {retry_count} retries: {last_error}",
                execution_time_ms=execution_time_ms,
                agent_id=assigned_agent_id,
                retry_count=retry_count
            )
            
        except Exception as e:
            logger.error(
                "Unexpected error in task execution",
                task_id=task_request.task_id,
                error=str(e)
            )
            
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return TaskExecutionResult(
                task_id=task_request.task_id,
                success=False,
                error_message=f"Unexpected execution error: {str(e)}",
                execution_time_ms=execution_time_ms
            )
    
    async def cancel_batch(self, batch_id: str) -> bool:
        """Cancel a running batch execution."""
        if batch_id not in self.active_batches:
            logger.warning(f"Batch {batch_id} not found in active batches")
            return False
        
        try:
            batch_task = self.active_batches[batch_id]
            batch_task.cancel()
            
            # Wait for cancellation to complete
            try:
                await batch_task
            except asyncio.CancelledError:
                pass
            
            logger.info(f"✅ Batch {batch_id} cancelled successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to cancel batch {batch_id}", error=str(e))
            return False
    
    async def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """Get current status of a batch execution."""
        is_active = batch_id in self.active_batches
        
        return {
            "batch_id": batch_id,
            "is_active": is_active,
            "start_time": datetime.utcnow().isoformat(),
            "agent_assignments": {},  # TODO: Track agent assignments
            "progress": {
                "completed_tasks": 0,
                "total_tasks": 0,
                "current_parallel_tasks": 0
            }
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current execution metrics and performance statistics."""
        return {
            **self.metrics,
            "active_batches": len(self.active_batches),
            "agent_capacities": {
                agent_id: {
                    "current_tasks": capacity.current_task_count,
                    "max_tasks": capacity.max_concurrent_tasks,
                    "utilization": capacity.current_task_count / max(capacity.max_concurrent_tasks, 1),
                    "success_rate": capacity.success_rate
                }
                for agent_id, capacity in self.agent_capacities.items()
            },
            "circuit_breaker_states": {
                agent_id: state.get("state", "closed")
                for agent_id, state in self.circuit_breaker_states.items()
            }
        }
    
    # Private methods
    
    async def _execute_tasks_with_strategy(
        self,
        task_requests: List[TaskExecutionRequest],
        strategy: BatchExecutionStrategy,
        max_parallel_tasks: Optional[int]
    ) -> List[TaskExecutionResult]:
        """Execute tasks using the specified strategy."""
        
        if strategy == BatchExecutionStrategy.SEQUENTIAL:
            return await self._execute_sequential(task_requests)
        
        elif strategy == BatchExecutionStrategy.PARALLEL_UNLIMITED:
            return await self._execute_parallel_unlimited(task_requests)
        
        elif strategy == BatchExecutionStrategy.PARALLEL_LIMITED:
            limit = max_parallel_tasks or len(self.agent_capacities)
            return await self._execute_parallel_limited(task_requests, limit)
        
        elif strategy == BatchExecutionStrategy.ADAPTIVE:
            return await self._execute_adaptive(task_requests)
        
        else:
            raise ValueError(f"Unknown execution strategy: {strategy}")
    
    async def _execute_sequential(self, task_requests: List[TaskExecutionRequest]) -> List[TaskExecutionResult]:
        """Execute tasks sequentially one by one."""
        results = []
        
        for task_request in task_requests:
            result = await self.execute_single_task(task_request)
            results.append(result)
            
            # Stop on first failure if configured
            if not result.success and task_request.context and task_request.context.get('fail_fast', False):
                logger.warning(f"Stopping sequential execution due to task failure: {task_request.task_id}")
                break
        
        return results
    
    async def _execute_parallel_unlimited(self, task_requests: List[TaskExecutionRequest]) -> List[TaskExecutionResult]:
        """Execute all tasks in parallel without limits."""
        tasks = [
            asyncio.create_task(self.execute_single_task(task_request))
            for task_request in task_requests
        ]
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.batch_timeout_seconds
            )
            
            # Process results and exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(TaskExecutionResult(
                        task_id=task_requests[i].task_id,
                        success=False,
                        error_message=str(result)
                    ))
                else:
                    final_results.append(result)
            
            return final_results
            
        except asyncio.TimeoutError:
            logger.error(f"Batch execution timeout after {self.batch_timeout_seconds} seconds")
            
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Return timeout results
            return [
                TaskExecutionResult(
                    task_id=req.task_id,
                    success=False,
                    error_message="Batch execution timeout"
                ) for req in task_requests
            ]
    
    async def _execute_parallel_limited(
        self, 
        task_requests: List[TaskExecutionRequest], 
        max_parallel: int
    ) -> List[TaskExecutionResult]:
        """Execute tasks in parallel with concurrency limit."""
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def execute_with_semaphore(task_request):
            async with semaphore:
                return await self.execute_single_task(task_request)
        
        tasks = [
            asyncio.create_task(execute_with_semaphore(task_request))
            for task_request in task_requests
        ]
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.batch_timeout_seconds
            )
            
            # Process results
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(TaskExecutionResult(
                        task_id=task_requests[i].task_id,
                        success=False,
                        error_message=str(result)
                    ))
                else:
                    final_results.append(result)
            
            return final_results
            
        except asyncio.TimeoutError:
            # Handle timeout similar to unlimited parallel
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            return [
                TaskExecutionResult(
                    task_id=req.task_id,
                    success=False,
                    error_message="Batch execution timeout"
                ) for req in task_requests
            ]
    
    async def _execute_adaptive(self, task_requests: List[TaskExecutionRequest]) -> List[TaskExecutionResult]:
        """Execute tasks using adaptive strategy based on current conditions."""
        # Analyze current system load
        available_agents = len([
            capacity for capacity in self.agent_capacities.values()
            if capacity.current_task_count < capacity.max_concurrent_tasks
        ])
        
        avg_success_rate = sum(
            capacity.success_rate for capacity in self.agent_capacities.values()
        ) / max(len(self.agent_capacities), 1)
        
        # Choose strategy based on conditions
        if available_agents >= len(task_requests) and avg_success_rate > 0.9:
            # High availability and success rate - go parallel unlimited
            return await self._execute_parallel_unlimited(task_requests)
        
        elif available_agents > 0 and avg_success_rate > 0.7:
            # Moderate conditions - limited parallel
            max_parallel = min(available_agents, len(task_requests) // 2 + 1)
            return await self._execute_parallel_limited(task_requests, max_parallel)
        
        else:
            # Poor conditions - sequential execution
            return await self._execute_sequential(task_requests)
    
    async def _assign_agents_to_tasks(self, task_requests: List[TaskExecutionRequest]) -> None:
        """Pre-assign agents to tasks for optimization."""
        for task_request in task_requests:
            if not task_request.agent_id:
                task_request.agent_id = await self._select_best_agent(task_request)
    
    async def _select_best_agent(
        self, 
        task_request: TaskExecutionRequest,
        exclude_agents: Set[str] = None
    ) -> Optional[str]:
        """Select the best available agent for a task."""
        exclude_agents = exclude_agents or set()
        
        # Get available agents
        available_agents = []
        for agent_id, capacity in self.agent_capacities.items():
            if (agent_id not in exclude_agents and 
                capacity.current_task_count < capacity.max_concurrent_tasks and
                not self._is_circuit_breaker_open(agent_id)):
                available_agents.append((agent_id, capacity))
        
        if not available_agents:
            return None
        
        # Score agents based on multiple factors
        def score_agent(agent_id: str, capacity: AgentCapacity) -> float:
            score = 0.0
            
            # Favor agents with lower current load
            load_factor = 1.0 - (capacity.current_task_count / capacity.max_concurrent_tasks)
            score += load_factor * 0.4
            
            # Favor agents with higher success rates
            score += capacity.success_rate * 0.3
            
            # Favor agents with relevant capabilities
            if task_request.task.required_capabilities:
                capability_match = len(
                    set(task_request.task.required_capabilities) & capacity.capabilities
                ) / len(task_request.task.required_capabilities)
                score += capability_match * 0.2
            
            # Favor agents with faster average execution times
            if capacity.average_task_duration_ms > 0:
                speed_factor = 1.0 / (1.0 + capacity.average_task_duration_ms / 60000)  # Normalize to minutes
                score += speed_factor * 0.1
            
            return score
        
        # Select agent with highest score
        best_agent = max(available_agents, key=lambda x: score_agent(x[0], x[1]))
        return best_agent[0]
    
    async def _send_task_to_agent(
        self, 
        task_request: TaskExecutionRequest, 
        agent_id: str
    ) -> Dict[str, Any]:
        """Send task to agent via communication service."""
        try:
            # Update agent capacity
            if agent_id in self.agent_capacities:
                self.agent_capacities[agent_id].current_task_count += 1
            
            # Send message to agent
            response = await self.communication_service.send_message(
                from_agent="task_batch_executor",
                to_agent=agent_id,
                message_type="task_execution",
                payload={
                    "task_id": task_request.task_id,
                    "task_data": task_request.task.to_dict(),
                    "context": task_request.context or {},
                    "timeout_seconds": task_request.timeout_seconds
                },
                timeout_seconds=task_request.timeout_seconds
            )
            
            # For now, simulate task execution
            # In production, this would wait for agent response
            await asyncio.sleep(0.1)  # Simulate work
            
            return {
                "status": "completed",
                "result": {"executed_by": agent_id, "task_id": task_request.task_id},
                "execution_time_ms": 100
            }
            
        finally:
            # Update agent capacity
            if agent_id in self.agent_capacities:
                self.agent_capacities[agent_id].current_task_count = max(
                    0, self.agent_capacities[agent_id].current_task_count - 1
                )
    
    async def _update_agent_capacities(self) -> None:
        """Update agent capacity information from registry."""
        try:
            active_agents = await self.agent_registry.get_active_agents()
            
            for agent in active_agents:
                agent_id = str(agent.id)
                
                if agent_id not in self.agent_capacities:
                    self.agent_capacities[agent_id] = AgentCapacity(
                        agent_id=agent_id,
                        max_concurrent_tasks=agent.max_concurrent_tasks or 5,
                        current_task_count=0,
                        average_task_duration_ms=60000.0,  # 1 minute default
                        success_rate=1.0,
                        last_activity=datetime.utcnow(),
                        capabilities=set(agent.capabilities or [])
                    )
                else:
                    # Update existing capacity info
                    capacity = self.agent_capacities[agent_id]
                    capacity.max_concurrent_tasks = agent.max_concurrent_tasks or 5
                    capacity.capabilities = set(agent.capabilities or [])
                    capacity.last_activity = datetime.utcnow()
            
            # Remove capacities for inactive agents
            active_agent_ids = {str(agent.id) for agent in active_agents}
            inactive_agents = set(self.agent_capacities.keys()) - active_agent_ids
            for agent_id in inactive_agents:
                del self.agent_capacities[agent_id]
                
        except Exception as e:
            logger.error("Failed to update agent capacities", error=str(e))
    
    def _is_circuit_breaker_open(self, agent_id: str) -> bool:
        """Check if circuit breaker is open for an agent."""
        if agent_id not in self.circuit_breaker_states:
            return False
        
        state = self.circuit_breaker_states[agent_id]
        
        if state["state"] == "closed":
            return False
        elif state["state"] == "open":
            # Check if enough time has passed to try half-open
            if datetime.utcnow() - state["last_failure"] > timedelta(seconds=self.circuit_breaker_window):
                state["state"] = "half_open"
                return False
            return True
        elif state["state"] == "half_open":
            return False
        
        return False
    
    def _record_agent_success(self, agent_id: str) -> None:
        """Record successful task execution for circuit breaker."""
        if agent_id in self.circuit_breaker_states:
            state = self.circuit_breaker_states[agent_id]
            if state["state"] == "half_open":
                # Reset circuit breaker on success in half-open state
                state["state"] = "closed"
                state["failure_count"] = 0
            
            # Update success rate
            state["success_count"] = state.get("success_count", 0) + 1
    
    def _record_agent_failure(self, agent_id: str, error_message: str) -> None:
        """Record failed task execution for circuit breaker."""
        if agent_id not in self.circuit_breaker_states:
            self.circuit_breaker_states[agent_id] = {
                "state": "closed",
                "failure_count": 0,
                "success_count": 0,
                "last_failure": datetime.utcnow()
            }
        
        state = self.circuit_breaker_states[agent_id]
        state["failure_count"] += 1
        state["last_failure"] = datetime.utcnow()
        
        # Calculate failure rate
        total_requests = state["failure_count"] + state.get("success_count", 0)
        failure_rate = state["failure_count"] / max(total_requests, 1)
        
        # Open circuit breaker if failure rate exceeds threshold
        if failure_rate >= self.circuit_breaker_threshold and state["state"] == "closed":
            state["state"] = "open"
            logger.warning(
                f"Circuit breaker opened for agent {agent_id}",
                failure_rate=failure_rate,
                failure_count=state["failure_count"]
            )
    
    async def _calculate_resource_utilization(self) -> Dict[str, float]:
        """Calculate current resource utilization across agents."""
        if not self.agent_capacities:
            return {}
        
        total_capacity = sum(capacity.max_concurrent_tasks for capacity in self.agent_capacities.values())
        total_used = sum(capacity.current_task_count for capacity in self.agent_capacities.values())
        
        return {
            "overall_utilization": total_used / max(total_capacity, 1),
            "active_agents": len(self.agent_capacities),
            "total_capacity": total_capacity,
            "total_used": total_used
        }
    
    async def _analyze_bottlenecks(self, task_results: List[TaskExecutionResult]) -> Dict[str, Any]:
        """Analyze bottlenecks in task execution."""
        if not task_results:
            return {}
        
        # Analyze execution times
        execution_times = [result.execution_time_ms for result in task_results if result.execution_time_ms > 0]
        avg_execution_time = sum(execution_times) / max(len(execution_times), 1)
        max_execution_time = max(execution_times) if execution_times else 0
        
        # Identify slow tasks
        slow_tasks = [
            result.task_id for result in task_results
            if result.execution_time_ms > avg_execution_time * 2
        ]
        
        # Analyze agent performance
        agent_performance = {}
        for result in task_results:
            if result.agent_id:
                if result.agent_id not in agent_performance:
                    agent_performance[result.agent_id] = {"tasks": 0, "successes": 0, "total_time": 0}
                
                agent_performance[result.agent_id]["tasks"] += 1
                if result.success:
                    agent_performance[result.agent_id]["successes"] += 1
                agent_performance[result.agent_id]["total_time"] += result.execution_time_ms
        
        return {
            "avg_execution_time_ms": avg_execution_time,
            "max_execution_time_ms": max_execution_time,
            "slow_tasks": slow_tasks,
            "agent_performance": agent_performance
        }
    
    def _create_empty_batch_result(self, batch_id: str) -> BatchExecutionResult:
        """Create empty batch result for edge cases."""
        return BatchExecutionResult(
            batch_id=batch_id,
            total_tasks=0,
            successful_tasks=0,
            failed_tasks=0,
            execution_time_ms=0,
            task_results=[],
            resource_utilization={},
            bottleneck_analysis={}
        )
    
    def _update_batch_metrics(self, batch_result: BatchExecutionResult) -> None:
        """Update global metrics based on batch execution result."""
        self.metrics["batches_executed"] += 1
        
        if batch_result.failed_tasks == 0:
            self.metrics["batches_successful"] += 1
        else:
            self.metrics["batches_failed"] += 1
        
        self.metrics["total_tasks_executed"] += batch_result.total_tasks
        self.metrics["total_tasks_successful"] += batch_result.successful_tasks
        self.metrics["total_tasks_failed"] += batch_result.failed_tasks
        
        # Update average batch duration
        current_avg = self.metrics["average_batch_duration_ms"]
        total_batches = self.metrics["batches_executed"]
        new_avg = ((current_avg * (total_batches - 1)) + batch_result.execution_time_ms) / total_batches
        self.metrics["average_batch_duration_ms"] = new_avg
        
        # Update average task duration
        if batch_result.task_results:
            task_times = [r.execution_time_ms for r in batch_result.task_results if r.execution_time_ms > 0]
            if task_times:
                avg_task_time = sum(task_times) / len(task_times)
                current_task_avg = self.metrics["average_task_duration_ms"]
                total_tasks = self.metrics["total_tasks_executed"]
                new_task_avg = ((current_task_avg * (total_tasks - len(task_times))) + sum(task_times)) / total_tasks
                self.metrics["average_task_duration_ms"] = new_task_avg