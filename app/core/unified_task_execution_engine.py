"""
Unified Task Execution Engine for LeanVibe Agent Hive 2.0
Consolidates 8+ task management implementations into high-performance execution system
Integrates seamlessly with unified production orchestrator for optimal multi-agent coordination.

Epic 1, Phase 2 Week 3 - Task System Consolidation
"""

from typing import Optional, Dict, Any, List, Set, Callable, Union, AsyncGenerator, Tuple
import asyncio
import uuid
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
import heapq
import time
import math

# Core imports with fallbacks
try:
    from .logging_service import get_component_logger
except (ImportError, NameError, AttributeError):
    import logging
    def get_component_logger(name):
        return logging.getLogger(name)

try:
    from .configuration_service import ConfigurationService
    CONFIGURATION_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    ConfigurationService = None
    CONFIGURATION_AVAILABLE = False

try:
    from .messaging_service import get_messaging_service, Message, MessageType
    MESSAGING_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    get_messaging_service = None
    MESSAGING_AVAILABLE = False

try:
    from .redis_integration import get_redis_service
    REDIS_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    get_redis_service = None
    REDIS_AVAILABLE = False

# Optional imports
try:
    from .circuit_breaker import CircuitBreakerService
    CIRCUIT_BREAKER_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    CircuitBreakerService = None
    CIRCUIT_BREAKER_AVAILABLE = False

try:
    from .database import get_async_session
    DATABASE_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    get_async_session = None
    DATABASE_AVAILABLE = False

try:
    from ..models.task import Task, TaskStatus, TaskPriority, TaskType
    from ..models.agent import Agent, AgentStatus
    MODELS_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    MODELS_AVAILABLE = False

logger = get_component_logger("unified_task_execution_engine")

class TaskExecutionStatus(str, Enum):
    """Unified task execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    TIMEOUT = "timeout"

class TaskExecutionType(str, Enum):
    """Unified task execution types"""
    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    RECURRING = "recurring"
    BATCH = "batch"
    PRIORITY = "priority"
    BACKGROUND = "background"
    WORKFLOW = "workflow"

class ExecutionMode(str, Enum):
    """Task execution modes"""
    ASYNC = "async"
    SYNC = "sync"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    BATCH = "batch"
    STREAMING = "streaming"

class SchedulingStrategy(str, Enum):
    """Unified scheduling strategies"""
    ROUND_ROBIN = "round_robin"
    CAPABILITY_MATCH = "capability_match"
    LOAD_BALANCED = "load_balanced"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    HYBRID = "hybrid"
    INTELLIGENT = "intelligent"
    ADAPTIVE = "adaptive"

@dataclass
class TaskDependency:
    """Task dependency definition"""
    dependency_id: str
    dependency_type: str = "completion"  # completion, data, resource
    required: bool = True
    timeout: Optional[timedelta] = None
    condition: Optional[str] = None

@dataclass
class TaskExecutionRequest:
    """Unified task execution request"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    function_name: str = ""
    function_args: List[Any] = field(default_factory=list)
    function_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Execution configuration
    task_type: TaskExecutionType = TaskExecutionType.IMMEDIATE
    execution_mode: ExecutionMode = ExecutionMode.ASYNC
    priority: int = 5  # 1-10 scale
    
    # Scheduling
    scheduled_at: Optional[datetime] = None
    execute_after: Optional[datetime] = None
    recurring_interval: Optional[timedelta] = None
    max_executions: Optional[int] = None
    
    # Dependencies and constraints
    dependencies: List[TaskDependency] = field(default_factory=list)
    required_capabilities: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    timeout: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    
    # Retry configuration
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    retry_backoff_max: timedelta = field(default_factory=lambda: timedelta(minutes=10))
    
    # Agent assignment preferences
    preferred_agent_id: Optional[str] = None
    agent_requirements: Dict[str, Any] = field(default_factory=dict)
    scheduling_strategy: SchedulingStrategy = SchedulingStrategy.HYBRID
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskExecutionContext:
    """Active task execution context"""
    task_id: str
    request: TaskExecutionRequest
    status: TaskExecutionStatus = TaskExecutionStatus.PENDING
    assigned_agent_id: Optional[str] = None
    
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Progress tracking
    progress_percentage: float = 0.0
    current_phase: str = "initialization"
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    
    # Results and errors
    result: Any = None
    error: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    
    # Performance metrics
    execution_time_ms: Optional[float] = None
    queue_wait_time_ms: Optional[float] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Workflow context
    workflow_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    child_task_ids: List[str] = field(default_factory=list)

class TaskQueue:
    """High-performance unified task queue"""
    
    def __init__(self, name: str, max_size: int = 10000):
        self.name = name
        self.max_size = max_size
        self._priority_heap: List[Tuple[float, str, TaskExecutionRequest]] = []
        self._task_lookup: Dict[str, TaskExecutionRequest] = {}
        self._size = 0
        self._lock = asyncio.Lock()
        
        # Queue metrics
        self._enqueue_count = 0
        self._dequeue_count = 0
        self._total_wait_time = 0.0
    
    async def enqueue(self, request: TaskExecutionRequest, priority: Optional[int] = None) -> bool:
        """Add task to priority queue"""
        async with self._lock:
            if self._size >= self.max_size:
                logger.warning("Task queue full", queue=self.name, size=self._size)
                return False
            
            # Calculate priority score (lower = higher priority)
            priority_score = priority or request.priority
            if request.task_type == TaskExecutionType.PRIORITY:
                priority_score -= 5  # Boost priority tasks
            
            # Add scheduled delay to priority for future tasks
            if request.scheduled_at:
                delay_minutes = max(0, (request.scheduled_at - datetime.utcnow()).total_seconds() / 60)
                priority_score += delay_minutes / 60  # Reduce priority by hours of delay
            
            # Use negative priority for max-heap behavior with heapq (min-heap)
            heapq.heappush(
                self._priority_heap,
                (-priority_score, request.task_id, request)
            )
            
            self._task_lookup[request.task_id] = request
            self._size += 1
            self._enqueue_count += 1
            
            logger.debug("Task enqueued", 
                        task_id=request.task_id, 
                        priority=priority_score, 
                        queue=self.name)
            return True
    
    async def dequeue(self) -> Optional[TaskExecutionRequest]:
        """Get highest priority ready task"""
        async with self._lock:
            current_time = datetime.utcnow()
            
            # Look for ready tasks in priority order
            while self._priority_heap:
                neg_priority, task_id, request = self._priority_heap[0]
                
                # Check if task is ready to execute
                if request.scheduled_at and request.scheduled_at > current_time:
                    # Task not ready yet
                    break
                
                # Remove task from heap
                heapq.heappop(self._priority_heap)
                del self._task_lookup[task_id]
                self._size -= 1
                self._dequeue_count += 1
                
                return request
            
            return None
    
    async def remove_task(self, task_id: str) -> bool:
        """Remove specific task from queue"""
        async with self._lock:
            if task_id not in self._task_lookup:
                return False
            
            # Mark for removal (will be filtered out during dequeue)
            del self._task_lookup[task_id]
            self._size -= 1
            
            # Rebuild heap without the removed task
            new_heap = [
                (neg_p, tid, req) for neg_p, tid, req in self._priority_heap
                if tid != task_id
            ]
            self._priority_heap = new_heap
            heapq.heapify(self._priority_heap)
            
            return True
    
    def size(self) -> int:
        """Get current queue size"""
        return self._size
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get queue performance metrics"""
        return {
            "size": self._size,
            "enqueue_count": self._enqueue_count,
            "dequeue_count": self._dequeue_count,
            "throughput": self._dequeue_count / max(1, time.time() - getattr(self, '_start_time', time.time())),
            "average_wait_time_ms": self._total_wait_time / max(1, self._dequeue_count)
        }

class AgentMatcher:
    """Intelligent agent matching and selection"""
    
    def __init__(self):
        self._agent_cache = {}
        self._cache_ttl = 60  # seconds
        self._last_cache_update = 0
    
    async def find_best_agent(self, 
                            request: TaskExecutionRequest,
                            available_agents: List[Agent],
                            strategy: SchedulingStrategy = SchedulingStrategy.HYBRID) -> Optional[Agent]:
        """Find best agent for task execution"""
        
        if not available_agents:
            return None
        
        # Filter by capabilities
        capable_agents = self._filter_by_capabilities(available_agents, request.required_capabilities)
        if not capable_agents:
            logger.warning("No agents with required capabilities", 
                         capabilities=request.required_capabilities)
            return None
        
        # Apply scheduling strategy
        if strategy == SchedulingStrategy.CAPABILITY_MATCH:
            return self._select_by_capability_match(capable_agents, request)
        elif strategy == SchedulingStrategy.LOAD_BALANCED:
            return self._select_by_load_balance(capable_agents)
        elif strategy == SchedulingStrategy.PERFORMANCE_OPTIMIZED:
            return self._select_by_performance(capable_agents)
        elif strategy == SchedulingStrategy.ROUND_ROBIN:
            return self._select_round_robin(capable_agents)
        else:  # HYBRID or INTELLIGENT
            return self._select_hybrid(capable_agents, request)
    
    def _filter_by_capabilities(self, agents: List[Agent], required_caps: List[str]) -> List[Agent]:
        """Filter agents by required capabilities"""
        if not required_caps:
            return agents
        
        capable_agents = []
        for agent in agents:
            if not agent.capabilities:
                continue
                
            agent_caps = set()
            for cap in agent.capabilities:
                if isinstance(cap, dict):
                    agent_caps.add(cap.get("name", "").lower())
                    agent_caps.update(area.lower() for area in cap.get("specialization_areas", []))
                else:
                    agent_caps.add(str(cap).lower())
            
            required_caps_set = set(cap.lower() for cap in required_caps)
            if required_caps_set.issubset(agent_caps):
                capable_agents.append(agent)
        
        return capable_agents
    
    def _select_by_capability_match(self, agents: List[Agent], request: TaskExecutionRequest) -> Agent:
        """Select agent with best capability match"""
        # For now, return first capable agent
        # TODO: Implement sophisticated capability scoring
        return agents[0]
    
    def _select_by_load_balance(self, agents: List[Agent]) -> Agent:
        """Select least loaded agent"""
        best_agent = None
        lowest_load = float('inf')
        
        for agent in agents:
            # Calculate load score (lower is better)
            context_usage = float(agent.context_window_usage or 0.0)
            resource_usage = agent.resource_usage or {}
            active_tasks = resource_usage.get("active_tasks_count", 0)
            
            load_score = context_usage + (active_tasks * 0.1)
            
            if load_score < lowest_load:
                lowest_load = load_score
                best_agent = agent
        
        return best_agent or agents[0]
    
    def _select_by_performance(self, agents: List[Agent]) -> Agent:
        """Select highest performing agent"""
        best_agent = None
        best_score = 0.0
        
        for agent in agents:
            # Calculate performance score
            health_score = agent.health_score or 0.5
            avg_response_time = float(agent.average_response_time or 5.0)
            response_score = max(0.0, 1.0 - (avg_response_time / 10.0))
            
            performance_score = (health_score * 0.6) + (response_score * 0.4)
            
            if performance_score > best_score:
                best_score = performance_score
                best_agent = agent
        
        return best_agent or agents[0]
    
    def _select_round_robin(self, agents: List[Agent]) -> Agent:
        """Simple round-robin selection"""
        # TODO: Implement proper round-robin state tracking
        return agents[0]
    
    def _select_hybrid(self, agents: List[Agent], request: TaskExecutionRequest) -> Agent:
        """Hybrid selection combining multiple factors"""
        best_agent = None
        best_score = 0.0
        
        for agent in agents:
            # Combine capability, load, and performance scores
            capability_score = 0.8  # TODO: Implement capability scoring
            
            # Load score (inverted - lower load = higher score)
            context_usage = float(agent.context_window_usage or 0.0)
            load_score = 1.0 - context_usage
            
            # Performance score
            health_score = agent.health_score or 0.5
            
            # Weighted hybrid score
            hybrid_score = (capability_score * 0.4) + (load_score * 0.3) + (health_score * 0.3)
            
            if hybrid_score > best_score:
                best_score = hybrid_score
                best_agent = agent
        
        return best_agent or agents[0]

class UnifiedTaskExecutionEngine:
    """
    Unified task execution engine consolidating all task management patterns:
    - High-performance task queuing with priority handling
    - Intelligent task scheduling and routing
    - Batch processing and parallel execution
    - Dependency resolution and constraint handling
    - Performance monitoring and adaptive optimization
    - Integration with production orchestrator
    """
    
    _instance: Optional['UnifiedTaskExecutionEngine'] = None
    
    def __new__(cls) -> 'UnifiedTaskExecutionEngine':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize_engine()
            self._initialized = True
    
    def _initialize_engine(self):
        """Initialize unified task execution engine"""
        # Initialize configuration if available
        if CONFIGURATION_AVAILABLE and ConfigurationService:
            self.config = ConfigurationService().config
        else:
            # Create minimal config object
            class MinimalConfig:
                class Performance:
                    max_workers = 10
                performance = Performance()
            self.config = MinimalConfig()
        
        # Initialize messaging if available
        if MESSAGING_AVAILABLE and get_messaging_service:
            self.messaging = get_messaging_service()
        else:
            self.messaging = None
        
        # Initialize Redis if available
        if REDIS_AVAILABLE and get_redis_service:
            self.redis = get_redis_service()
        else:
            self.redis = None
        
        # Initialize circuit breaker if available
        if CIRCUIT_BREAKER_AVAILABLE and CircuitBreakerService:
            self.circuit_breaker = CircuitBreakerService().get_circuit_breaker("task_execution")
        else:
            self.circuit_breaker = None
        
        # Task queues by type
        self._immediate_queue = TaskQueue("immediate", max_size=5000)
        self._scheduled_queue = TaskQueue("scheduled", max_size=10000)
        self._batch_queue = TaskQueue("batch", max_size=2000)
        self._priority_queue = TaskQueue("priority", max_size=1000)
        self._workflow_queue = TaskQueue("workflow", max_size=3000)
        
        # Execution tracking
        self._active_executions: Dict[str, TaskExecutionContext] = {}
        self._completed_tasks: deque = deque(maxlen=10000)
        self._failed_tasks: deque = deque(maxlen=1000)
        
        # Worker management
        max_workers = getattr(self.config.performance, 'max_workers', 10)
        self._worker_pool = ThreadPoolExecutor(max_workers=max_workers)
        self._execution_semaphore = asyncio.Semaphore(max_workers * 2)
        
        # Components
        self._agent_matcher = AgentMatcher()
        
        # Scheduling and execution
        self._scheduler_running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self._execution_stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time_ms": 0.0,
            "average_execution_time_ms": 0.0,
            "queue_sizes": {},
            "active_executions": 0,
            "throughput_per_minute": 0.0
        }
        
        # Task function registry
        self._task_registry: Dict[str, Callable] = {}
        
        # Dependency tracking
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._dependents_graph: Dict[str, Set[str]] = defaultdict(set)
        
        logger.info("Unified task execution engine initialized", 
                   max_workers=max_workers)
    
    async def start_engine(self):
        """Start unified task execution engine"""
        if not self._scheduler_running:
            self._scheduler_task = asyncio.create_task(self._execution_scheduler())
            self._scheduler_running = True
            
            # Start background services
            asyncio.create_task(self._metrics_collector())
            asyncio.create_task(self._dependency_resolver())
            asyncio.create_task(self._queue_monitor())
        
        logger.info("Unified task execution engine started")
    
    async def stop_engine(self):
        """Stop task execution engine gracefully"""
        self._scheduler_running = False
        
        if self._scheduler_task and not self._scheduler_task.done():
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Wait for executing tasks to complete
        timeout = 30  # seconds
        start_time = time.time()
        while self._active_executions and (time.time() - start_time) < timeout:
            logger.info("Waiting for tasks to complete", count=len(self._active_executions))
            await asyncio.sleep(1)
        
        # Force cleanup remaining tasks
        for task_id, context in list(self._active_executions.items()):
            context.status = TaskExecutionStatus.CANCELLED
            context.error = "Engine shutdown"
            self._active_executions.pop(task_id, None)
        
        self._worker_pool.shutdown(wait=True)
        logger.info("Unified task execution engine stopped")
    
    def register_task_function(self, name: str, function: Callable):
        """Register function for task execution"""
        self._task_registry[name] = function
        logger.debug("Task function registered", name=name)
    
    async def submit_task(self, request: TaskExecutionRequest) -> str:
        """Submit task for execution"""
        
        # Validate request
        if not request.function_name:
            raise ValueError("Function name is required")
        
        if request.function_name not in self._task_registry:
            raise ValueError(f"Function not registered: {request.function_name}")
        
        # Update statistics
        self._execution_stats["tasks_submitted"] += 1
        
        # Create execution context
        context = TaskExecutionContext(
            task_id=request.task_id,
            request=request,
            status=TaskExecutionStatus.QUEUED
        )
        
        # Handle dependencies
        if request.dependencies:
            await self._register_dependencies(request.task_id, request.dependencies)
            context.status = TaskExecutionStatus.PENDING
        
        # Queue task based on type
        queue_success = False
        if request.task_type == TaskExecutionType.IMMEDIATE:
            queue_success = await self._immediate_queue.enqueue(request, request.priority)
        elif request.task_type == TaskExecutionType.SCHEDULED:
            queue_success = await self._scheduled_queue.enqueue(request, request.priority)
        elif request.task_type == TaskExecutionType.BATCH:
            queue_success = await self._batch_queue.enqueue(request, request.priority)
        elif request.task_type == TaskExecutionType.PRIORITY:
            queue_success = await self._priority_queue.enqueue(request, 1)  # Highest priority
        elif request.task_type == TaskExecutionType.WORKFLOW:
            queue_success = await self._workflow_queue.enqueue(request, request.priority)
        else:
            queue_success = await self._immediate_queue.enqueue(request, request.priority)
        
        if not queue_success:
            raise RuntimeError(f"Failed to queue task: {request.task_id}")
        
        # Store execution context
        self._active_executions[request.task_id] = context
        
        # Persist task for fault tolerance
        await self._persist_task_context(context)
        
        logger.info("Task submitted", 
                   task_id=request.task_id, 
                   function=request.function_name, 
                   type=request.task_type.value,
                   priority=request.priority)
        
        return request.task_id
    
    async def _execution_scheduler(self):
        """Main execution scheduler loop"""
        while self._scheduler_running:
            try:
                # Process priority queue first
                await self._process_queue(self._priority_queue, "priority")
                
                # Process immediate tasks
                await self._process_queue(self._immediate_queue, "immediate")
                
                # Process workflow tasks
                await self._process_queue(self._workflow_queue, "workflow")
                
                # Process scheduled tasks
                await self._process_queue(self._scheduled_queue, "scheduled")
                
                # Process batch tasks
                await self._process_queue(self._batch_queue, "batch")
                
                # Brief pause to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error("Scheduler error", error=str(e))
                await asyncio.sleep(1)
    
    async def _process_queue(self, queue: TaskQueue, queue_name: str):
        """Process tasks from specific queue"""
        max_batch_size = 10  # Process up to 10 tasks per cycle
        processed = 0
        
        while processed < max_batch_size:
            request = await queue.dequeue()
            if not request:
                break
            
            # Check if task has unfulfilled dependencies
            if request.task_id in self._dependency_graph:
                dependencies = self._dependency_graph[request.task_id]
                unfulfilled = await self._check_unfulfilled_dependencies(dependencies)
                if unfulfilled:
                    # Requeue for later processing
                    await queue.enqueue(request, request.priority + 1)
                    continue
            
            # Try to execute task
            success = await self._try_execute_task(request)
            if not success:
                # Requeue with lower priority if execution failed to start
                await queue.enqueue(request, request.priority + 2)
            
            processed += 1
    
    async def _try_execute_task(self, request: TaskExecutionRequest) -> bool:
        """Attempt to execute task"""
        try:
            # Get available agents if needed
            if request.required_capabilities or request.preferred_agent_id:
                agent = await self._select_agent(request)
                if not agent:
                    logger.debug("No suitable agent available", task_id=request.task_id)
                    return False
                
                # Update context with assigned agent
                if request.task_id in self._active_executions:
                    self._active_executions[request.task_id].assigned_agent_id = str(agent.id)
            
            # Start task execution
            asyncio.create_task(self._execute_task(request))
            return True
            
        except Exception as e:
            logger.error("Failed to start task execution", 
                        task_id=request.task_id, error=str(e))
            return False
    
    async def _execute_task(self, request: TaskExecutionRequest):
        """Execute individual task with comprehensive error handling"""
        task_id = request.task_id
        context = self._active_executions.get(task_id)
        
        if not context:
            logger.error("Task context not found", task_id=task_id)
            return
        
        async with self._execution_semaphore:
            start_time = datetime.utcnow()
            context.status = TaskExecutionStatus.EXECUTING
            context.started_at = start_time
            context.current_phase = "execution"
            
            try:
                async def _execute_with_circuit_breaker():
                    # Apply circuit breaker if available
                    if self.circuit_breaker:
                        @self.circuit_breaker
                        async def _protected_execution():
                            return await _do_execution()
                        return await _protected_execution()
                    else:
                        return await _do_execution()
                
                async def _do_execution():
                    function = self._task_registry[request.function_name]
                    
                    if request.execution_mode == ExecutionMode.ASYNC:
                        if asyncio.iscoroutinefunction(function):
                            result = await function(*request.function_args, **request.function_kwargs)
                        else:
                            # Run sync function in thread pool
                            result = await asyncio.get_event_loop().run_in_executor(
                                self._worker_pool, 
                                lambda: function(*request.function_args, **request.function_kwargs)
                            )
                    else:
                        # Synchronous execution
                        result = function(*request.function_args, **request.function_kwargs)
                    
                    return result
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    _execute_with_circuit_breaker(),
                    timeout=request.timeout.total_seconds()
                )
                
                # Task completed successfully
                context.result = result
                context.status = TaskExecutionStatus.COMPLETED
                context.completed_at = datetime.utcnow()
                context.execution_time_ms = (context.completed_at - start_time).total_seconds() * 1000
                context.progress_percentage = 100.0
                
                # Update statistics
                self._execution_stats["tasks_completed"] += 1
                self._execution_stats["total_execution_time_ms"] += context.execution_time_ms
                
                # Resolve dependent tasks
                await self._resolve_task_dependencies(task_id)
                
                logger.info("Task completed successfully", 
                           task_id=task_id, 
                           execution_time_ms=context.execution_time_ms)
                
            except asyncio.TimeoutError:
                context.status = TaskExecutionStatus.TIMEOUT
                context.error = f"Task timed out after {request.timeout.total_seconds()} seconds"
                context.completed_at = datetime.utcnow()
                await self._handle_task_failure(context, "timeout")
                
            except Exception as e:
                context.status = TaskExecutionStatus.FAILED
                context.error = str(e)
                context.error_details = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": str(e)  # TODO: Add proper traceback handling
                }
                context.completed_at = datetime.utcnow()
                
                # Handle retry logic
                await self._handle_task_failure(context, "error")
                
            finally:
                # Clean up active execution
                if task_id in self._active_executions:
                    if context.status in [TaskExecutionStatus.COMPLETED, TaskExecutionStatus.FAILED, TaskExecutionStatus.TIMEOUT]:
                        # Move to completed/failed collections
                        if context.status == TaskExecutionStatus.COMPLETED:
                            self._completed_tasks.append(context)
                        else:
                            self._failed_tasks.append(context)
                            self._execution_stats["tasks_failed"] += 1
                        
                        # Remove from active executions
                        del self._active_executions[task_id]
                        
                        # Clean up dependencies
                        self._dependency_graph.pop(task_id, None)
                        
                        # Clean up persistence
                        await self._cleanup_persisted_task(task_id)
    
    async def _handle_task_failure(self, context: TaskExecutionContext, failure_type: str):
        """Handle task failure with retry logic"""
        request = context.request
        
        if context.retry_count < request.max_retries:
            context.retry_count += 1
            context.status = TaskExecutionStatus.RETRYING
            
            # Calculate exponential backoff delay
            delay_seconds = min(
                request.retry_backoff_base ** context.retry_count,
                request.retry_backoff_max.total_seconds()
            )
            
            # Schedule retry
            retry_request = TaskExecutionRequest(
                task_id=f"{request.task_id}_retry_{context.retry_count}",
                function_name=request.function_name,
                function_args=request.function_args,
                function_kwargs=request.function_kwargs,
                task_type=request.task_type,
                execution_mode=request.execution_mode,
                priority=max(1, request.priority - 1),  # Increase priority for retries
                scheduled_at=datetime.utcnow() + timedelta(seconds=delay_seconds),
                required_capabilities=request.required_capabilities,
                resource_requirements=request.resource_requirements,
                timeout=request.timeout,
                max_retries=request.max_retries - context.retry_count,
                metadata={**request.metadata, "original_task_id": request.task_id, "retry_attempt": context.retry_count}
            )
            
            await self.submit_task(retry_request)
            
            logger.warning("Task retry scheduled", 
                          task_id=request.task_id, 
                          retry_count=context.retry_count, 
                          delay_seconds=delay_seconds,
                          failure_type=failure_type)
        else:
            logger.error("Task failed after max retries", 
                        task_id=request.task_id, 
                        retries=context.retry_count,
                        failure_type=failure_type)
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive task status"""
        # Check active executions
        if task_id in self._active_executions:
            context = self._active_executions[task_id]
            return {
                "task_id": task_id,
                "status": context.status.value,
                "progress_percentage": context.progress_percentage,
                "current_phase": context.current_phase,
                "started_at": context.started_at.isoformat() if context.started_at else None,
                "assigned_agent_id": context.assigned_agent_id,
                "retry_count": context.retry_count,
                "execution_time_ms": (datetime.utcnow() - context.started_at).total_seconds() * 1000 if context.started_at else 0,
                "metadata": context.request.metadata
            }
        
        # Check completed tasks
        for context in self._completed_tasks:
            if context.task_id == task_id:
                return {
                    "task_id": task_id,
                    "status": context.status.value,
                    "result": context.result,
                    "completed_at": context.completed_at.isoformat() if context.completed_at else None,
                    "execution_time_ms": context.execution_time_ms,
                    "assigned_agent_id": context.assigned_agent_id
                }
        
        # Check failed tasks
        for context in self._failed_tasks:
            if context.task_id == task_id:
                return {
                    "task_id": task_id,
                    "status": context.status.value,
                    "error": context.error,
                    "error_details": context.error_details,
                    "failed_at": context.completed_at.isoformat() if context.completed_at else None,
                    "retry_count": context.retry_count,
                    "assigned_agent_id": context.assigned_agent_id
                }
        
        return None
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get comprehensive execution engine statistics"""
        # Update queue sizes
        self._execution_stats["queue_sizes"] = {
            "immediate": self._immediate_queue.size(),
            "scheduled": self._scheduled_queue.size(),
            "batch": self._batch_queue.size(),
            "priority": self._priority_queue.size(),
            "workflow": self._workflow_queue.size()
        }
        
        self._execution_stats["active_executions"] = len(self._active_executions)
        
        # Calculate average execution time
        if self._execution_stats["tasks_completed"] > 0:
            self._execution_stats["average_execution_time_ms"] = (
                self._execution_stats["total_execution_time_ms"] / 
                self._execution_stats["tasks_completed"]
            )
        
        # Calculate throughput
        # TODO: Implement proper throughput calculation based on time windows
        
        return {
            **self._execution_stats,
            "timestamp": datetime.utcnow().isoformat(),
            "engine_status": "running" if self._scheduler_running else "stopped",
            "queue_metrics": {
                "immediate": self._immediate_queue.get_metrics(),
                "scheduled": self._scheduled_queue.get_metrics(),
                "batch": self._batch_queue.get_metrics(),
                "priority": self._priority_queue.get_metrics(),
                "workflow": self._workflow_queue.get_metrics()
            }
        }
    
    # Helper methods for dependency management, agent selection, etc.
    async def _register_dependencies(self, task_id: str, dependencies: List[TaskDependency]):
        """Register task dependencies"""
        for dep in dependencies:
            self._dependency_graph[task_id].add(dep.dependency_id)
            self._dependents_graph[dep.dependency_id].add(task_id)
    
    async def _check_unfulfilled_dependencies(self, dependencies: Set[str]) -> List[str]:
        """Check which dependencies are still unfulfilled"""
        unfulfilled = []
        for dep_id in dependencies:
            # Check if dependency task is completed
            status = await self.get_task_status(dep_id)
            if not status or status["status"] != TaskExecutionStatus.COMPLETED.value:
                unfulfilled.append(dep_id)
        return unfulfilled
    
    async def _resolve_task_dependencies(self, completed_task_id: str):
        """Resolve dependencies when a task completes"""
        if completed_task_id in self._dependents_graph:
            dependent_tasks = self._dependents_graph[completed_task_id]
            for dependent_id in dependent_tasks:
                if dependent_id in self._dependency_graph:
                    self._dependency_graph[dependent_id].discard(completed_task_id)
    
    async def _select_agent(self, request: TaskExecutionRequest) -> Optional[Agent]:
        """Select best agent for task execution"""
        try:
            # TODO: Integrate with agent registry to get available agents
            # For now, return None to indicate no agent selection required
            return None
        except Exception as e:
            logger.error("Agent selection failed", task_id=request.task_id, error=str(e))
            return None
    
    async def _persist_task_context(self, context: TaskExecutionContext):
        """Persist task context for fault tolerance"""
        try:
            if self.redis:
                # Store in Redis for quick recovery
                await self.redis.setex(
                    f"task_context:{context.task_id}",
                    3600,  # 1 hour TTL
                    json.dumps({
                        "task_id": context.task_id,
                        "status": context.status.value,
                        "created_at": context.request.created_at.isoformat(),
                        "function_name": context.request.function_name,
                        "metadata": context.request.metadata
                    })
                )
            else:
                # No persistence available, log warning
                logger.debug("Redis not available, skipping task context persistence")
        except Exception as e:
            logger.warning("Failed to persist task context", task_id=context.task_id, error=str(e))
    
    async def _cleanup_persisted_task(self, task_id: str):
        """Clean up persisted task data"""
        try:
            if self.redis:
                await self.redis.delete(f"task_context:{task_id}")
        except Exception as e:
            logger.warning("Failed to cleanup persisted task", task_id=task_id, error=str(e))
    
    async def _metrics_collector(self):
        """Background metrics collection"""
        while self._scheduler_running:
            try:
                # Collect and update metrics
                # TODO: Implement comprehensive metrics collection
                await asyncio.sleep(60)  # Collect every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
                await asyncio.sleep(10)
    
    async def _dependency_resolver(self):
        """Background dependency resolution"""
        while self._scheduler_running:
            try:
                # Check for tasks with resolved dependencies
                # TODO: Implement proactive dependency resolution
                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Dependency resolver error", error=str(e))
                await asyncio.sleep(5)
    
    async def _queue_monitor(self):
        """Background queue monitoring and optimization"""
        while self._scheduler_running:
            try:
                # Monitor queue health and performance
                # TODO: Implement queue optimization logic
                await asyncio.sleep(30)  # Monitor every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Queue monitor error", error=str(e))
                await asyncio.sleep(10)

# Convenience functions
def get_unified_task_execution_engine() -> UnifiedTaskExecutionEngine:
    """Get unified task execution engine instance"""
    return UnifiedTaskExecutionEngine()

async def execute_task(function_name: str, 
                      *args, 
                      priority: int = 5,
                      execution_mode: ExecutionMode = ExecutionMode.ASYNC,
                      **kwargs) -> str:
    """Quick task execution"""
    engine = get_unified_task_execution_engine()
    request = TaskExecutionRequest(
        function_name=function_name,
        function_args=list(args),
        function_kwargs=kwargs,
        priority=priority,
        execution_mode=execution_mode,
        task_type=TaskExecutionType.IMMEDIATE
    )
    return await engine.submit_task(request)

async def schedule_task(function_name: str,
                       scheduled_at: datetime,
                       *args,
                       priority: int = 5,
                       **kwargs) -> str:
    """Schedule task for future execution"""
    engine = get_unified_task_execution_engine()
    request = TaskExecutionRequest(
        function_name=function_name,
        function_args=list(args),
        function_kwargs=kwargs,
        priority=priority,
        task_type=TaskExecutionType.SCHEDULED,
        scheduled_at=scheduled_at
    )
    return await engine.submit_task(request)

async def execute_batch_tasks(tasks: List[Tuple[str, List[Any], Dict[str, Any]]], 
                             priority: int = 5) -> List[str]:
    """Execute multiple tasks in batch"""
    engine = get_unified_task_execution_engine()
    task_ids = []
    
    for function_name, args, kwargs in tasks:
        request = TaskExecutionRequest(
            function_name=function_name,
            function_args=args,
            function_kwargs=kwargs,
            priority=priority,
            task_type=TaskExecutionType.BATCH
        )
        task_id = await engine.submit_task(request)
        task_ids.append(task_id)
    
    return task_ids