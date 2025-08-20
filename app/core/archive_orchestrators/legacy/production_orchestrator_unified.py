"""
Unified Production Orchestrator for LeanVibe Agent Hive
Consolidates 6+ orchestrator implementations into high-performance, reliable engine

Epic 1, Phase 2 Week 3: Critical Orchestrator Consolidation
This module consolidates the following orchestrator implementations:
1. orchestrator.py - Core orchestrator (782 lines)
2. production_orchestrator.py - Production-specific orchestrator
3. unified_production_orchestrator.py - Enhanced production orchestrator
4. automated_orchestrator.py - Automation-focused orchestrator  
5. performance_orchestrator.py - Performance-optimized orchestrator
6. high_concurrency_orchestrator.py - High-throughput orchestrator

Key Features:
- Enterprise-grade agent lifecycle management (spawn, monitor, terminate)
- High-performance task routing and load balancing (< 50ms assignment)
- Intelligent capability matching and resource optimization
- Fault tolerance with automatic recovery and circuit breaker patterns
- Performance monitoring and adaptive optimization
- Multi-agent coordination and collaboration
- Auto-scaling for 50+ concurrent agents

Performance Targets:
- Agent Registration: <100ms per agent
- Task Assignment: <50ms for intelligent routing
- Concurrent Agents: 50+ simultaneous agents with optimal resource usage
- Memory Efficiency: <50MB base overhead, <5KB per agent
- System Uptime: 99.9% availability with graceful degradation
"""

from typing import Optional, Dict, Any, List, Set, Callable, Union, Tuple, Protocol
import asyncio
import uuid
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from dataclasses import dataclass, field
import weakref
from collections import defaultdict, deque
import time
import statistics
import heapq
from concurrent.futures import ThreadPoolExecutor
import threading
import psutil
import json

from app.core.logging_service import get_component_logger
from app.core.configuration_service import ConfigurationService
from app.core.messaging_service import get_messaging_service, Message, MessageType, MessagePriority
from app.core.redis_integration import get_redis_service, redis_session

# Optional circuit breaker import for resilience
try:
    from app.core.circuit_breaker import CircuitBreakerService
    CIRCUIT_BREAKER_AVAILABLE = True
except (ImportError, NameError, AttributeError, Exception):
    CIRCUIT_BREAKER_AVAILABLE = False
    CircuitBreakerService = None

logger = get_component_logger("production_orchestrator")


class AgentState(str, Enum):
    """Unified agent states across all orchestrator implementations"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    IDLE = "idle"
    SLEEPING = "sleeping"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    TERMINATING = "terminating"
    TERMINATED = "terminated"


class OrchestrationStrategy(str, Enum):
    """Task routing strategies for optimal agent utilization"""
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    CAPABILITY_BASED = "capability_based"
    PRIORITY_BASED = "priority_based"
    INTELLIGENT = "intelligent"
    PERFORMANCE_OPTIMIZED = "performance_optimized"


class TaskPriority(IntEnum):
    """Task priority levels for intelligent scheduling"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class ResourceType(str, Enum):
    """System resource types for monitoring and optimization"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE_CONNECTIONS = "database_connections"
    REDIS_CONNECTIONS = "redis_connections"


@dataclass
class AgentCapability:
    """Agent capability definition with skill assessment"""
    capability_type: str
    skill_level: int  # 1-10 scale
    max_concurrent_tasks: int = 1
    estimated_processing_time: float = 60.0  # seconds
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    specialization_areas: List[str] = field(default_factory=list)
    confidence_level: float = 1.0  # 0.0-1.0


@dataclass
class RegisteredAgent:
    """Registered agent in the unified orchestrator"""
    agent_id: str
    agent_type: str
    capabilities: List[AgentCapability]
    state: AgentState = AgentState.INITIALIZING
    current_tasks: Set[str] = field(default_factory=set)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    spawn_time: datetime = field(default_factory=datetime.utcnow)
    total_tasks_completed: int = 0
    average_task_time: float = 0.0
    error_count: int = 0
    context_window_usage: float = 0.0  # 0.0-1.0
    max_context_window: int = 100000  # tokens
    anthropic_client: Optional[Any] = None
    tmux_session: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "capabilities": [
                {
                    "capability_type": cap.capability_type,
                    "skill_level": cap.skill_level,
                    "max_concurrent_tasks": cap.max_concurrent_tasks,
                    "estimated_processing_time": cap.estimated_processing_time,
                    "resource_requirements": cap.resource_requirements,
                    "specialization_areas": cap.specialization_areas,
                    "confidence_level": cap.confidence_level
                } for cap in self.capabilities
            ],
            "state": self.state.value,
            "current_tasks": list(self.current_tasks),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "performance_metrics": self.performance_metrics,
            "spawn_time": self.spawn_time.isoformat(),
            "total_tasks_completed": self.total_tasks_completed,
            "average_task_time": self.average_task_time,
            "error_count": self.error_count,
            "context_window_usage": self.context_window_usage,
            "max_context_window": self.max_context_window,
            "tmux_session": self.tmux_session
        }


@dataclass
class OrchestrationTask:
    """Task to be orchestrated to agents with comprehensive metadata"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = "generic"
    priority: TaskPriority = TaskPriority.NORMAL
    required_capabilities: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    assigned_agent: Optional[str] = None
    status: str = "pending"
    retry_count: int = 0
    max_retries: int = 3
    estimated_duration: float = 60.0  # seconds
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "priority": self.priority,
            "required_capabilities": self.required_capabilities,
            "payload": self.payload,
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "assigned_agent": self.assigned_agent,
            "status": self.status,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "estimated_duration": self.estimated_duration,
            "resource_requirements": self.resource_requirements
        }


@dataclass
class OrchestrationMetrics:
    """Comprehensive orchestration performance metrics"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tasks_processed: int = 0
    tasks_failed: int = 0
    agents_spawned: int = 0
    agents_terminated: int = 0
    average_task_time: float = 0.0
    current_load: float = 0.0
    active_agents: int = 0
    busy_agents: int = 0
    pending_tasks: int = 0
    active_tasks: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    redis_connections: int = 0
    database_connections: int = 0
    average_agent_registration_time: float = 0.0
    average_task_assignment_time: float = 0.0
    system_health_score: float = 1.0  # 0.0-1.0


class UnifiedProductionOrchestrator:
    """
    Unified production orchestrator consolidating all orchestration patterns:
    
    Core Responsibilities:
    - Agent lifecycle management (spawn, monitor, terminate)
    - High-performance task routing and load balancing  
    - Intelligent capability matching and resource optimization
    - Fault tolerance with automatic recovery and failover
    - Performance monitoring and adaptive optimization
    - Multi-agent coordination and collaboration
    - Auto-scaling for 50+ concurrent agents
    
    Performance Requirements:
    - Agent Registration: <100ms per agent
    - Task Assignment: <50ms for intelligent routing
    - Memory Efficiency: <50MB base + <5KB per agent
    - 99.9% uptime with graceful degradation
    """
    
    _instance: Optional['UnifiedProductionOrchestrator'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'UnifiedProductionOrchestrator':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not getattr(self, '_initialized', False):
            self._initialize_orchestrator()
            self._initialized = True
    
    def _initialize_orchestrator(self):
        """Initialize orchestrator with all required services"""
        try:
            # Core service initialization
            self.config = ConfigurationService().config
            self.messaging = get_messaging_service()
            self.redis = get_redis_service()
            
            # Circuit breaker for resilience
            if CIRCUIT_BREAKER_AVAILABLE:
                self.circuit_breaker = CircuitBreakerService().get_circuit_breaker("orchestrator")
            else:
                self.circuit_breaker = None
                logger.warning("Circuit breaker not available, operating without resilience patterns")
            
            # Agent management
            self._agents: Dict[str, RegisteredAgent] = {}
            self._agent_capabilities: Dict[str, List[str]] = defaultdict(list)
            self._agent_performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
            
            # Task management
            self._pending_tasks: deque = deque()
            self._active_tasks: Dict[str, OrchestrationTask] = {}
            self._completed_tasks: deque = deque(maxlen=1000)
            self._task_assignment_times: deque = deque(maxlen=100)
            
            # Performance tracking
            self._orchestration_metrics = OrchestrationMetrics()
            self._system_metrics: Dict[str, float] = {}
            self._agent_registration_times: deque = deque(maxlen=100)
            
            # Orchestration configuration
            self._max_agents = getattr(self.config.performance, 'max_workers', 50) * 2
            self._task_timeout = timedelta(minutes=30)
            self._heartbeat_interval = 30  # seconds
            self._strategy = OrchestrationStrategy.INTELLIGENT
            self._auto_scaling_enabled = True
            self._resource_monitoring_enabled = True
            
            # Background task management
            self._orchestration_task: Optional[asyncio.Task] = None
            self._monitoring_task: Optional[asyncio.Task] = None
            self._health_check_task: Optional[asyncio.Task] = None
            self._metrics_collection_task: Optional[asyncio.Task] = None
            self._running = False
            
            # Thread pool for CPU-intensive operations
            self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="orchestrator")
            
            logger.info("Unified production orchestrator initialized", 
                       max_agents=self._max_agents, 
                       strategy=self._strategy.value,
                       auto_scaling_enabled=self._auto_scaling_enabled)
            
        except Exception as e:
            logger.error("Failed to initialize orchestrator", error=str(e))
            raise
    
    async def start_orchestrator(self):
        """Start orchestrator background processes"""
        if self._running:
            logger.warning("Orchestrator already running")
            return
        
        self._running = True
        
        try:
            # Start core orchestration processes
            if self._orchestration_task is None or self._orchestration_task.done():
                self._orchestration_task = asyncio.create_task(self._orchestration_loop())
            
            if self._monitoring_task is None or self._monitoring_task.done():
                self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            if self._health_check_task is None or self._health_check_task.done():
                self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            if self._metrics_collection_task is None or self._metrics_collection_task.done():
                self._metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())
            
            logger.info("Unified production orchestrator started successfully")
            
        except Exception as e:
            logger.error("Failed to start orchestrator", error=str(e))
            self._running = False
            raise
    
    async def stop_orchestrator(self):
        """Gracefully stop orchestrator"""
        if not self._running:
            return
        
        logger.info("Stopping orchestrator gracefully...")
        self._running = False
        
        try:
            # Cancel background tasks
            tasks_to_cancel = [
                self._orchestration_task,
                self._monitoring_task,
                self._health_check_task,
                self._metrics_collection_task
            ]
            
            for task in tasks_to_cancel:
                if task and not task.done():
                    task.cancel()
            
            # Wait for tasks to complete with timeout
            await asyncio.gather(*[task for task in tasks_to_cancel if task], return_exceptions=True)
            
            # Gracefully terminate all agents
            await self._terminate_all_agents()
            
            # Shutdown thread pool
            self._thread_pool.shutdown(wait=True)
            
            logger.info("Unified production orchestrator stopped successfully")
            
        except Exception as e:
            logger.error("Error during orchestrator shutdown", error=str(e))
    
    async def _orchestration_loop(self):
        """Main orchestration loop for task processing"""
        logger.info("Starting orchestration loop")
        
        while self._running:
            try:
                start_time = time.time()
                
                # Process pending tasks
                await self._process_pending_tasks()
                
                # Check for task timeouts
                await self._check_task_timeouts()
                
                # Optimize agent allocation
                if self._auto_scaling_enabled:
                    await self._optimize_agent_allocation()
                
                # Update performance metrics
                await self._update_orchestration_metrics()
                
                # Sleep to prevent busy waiting (aim for ~100Hz processing)
                processing_time = time.time() - start_time
                sleep_time = max(0.01, 0.01 - processing_time)  # 10ms target cycle
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                logger.info("Orchestration loop cancelled")
                break
            except Exception as e:
                logger.error("Orchestration loop error", error=str(e))
                await asyncio.sleep(5)  # Back off on error
    
    async def _monitoring_loop(self):
        """Agent monitoring and health check loop"""
        logger.info("Starting monitoring loop")
        
        while self._running:
            try:
                start_time = time.time()
                
                # Check agent health
                await self._check_agent_health()
                
                # Update performance metrics for agents
                await self._update_agent_performance_metrics()
                
                # Cleanup stale agents
                await self._cleanup_stale_agents()
                
                # Check system resources
                if self._resource_monitoring_enabled:
                    await self._monitor_system_resources()
                
                # Sleep until next monitoring cycle
                processing_time = time.time() - start_time
                sleep_time = max(1, self._heartbeat_interval - processing_time)
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                logger.error("Monitoring loop error", error=str(e))
                await asyncio.sleep(10)  # Back off on error
    
    async def _health_check_loop(self):
        """System health monitoring loop"""
        logger.info("Starting health check loop")
        
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(60)  # Health checks every minute
                
            except asyncio.CancelledError:
                logger.info("Health check loop cancelled")
                break
            except Exception as e:
                logger.error("Health check loop error", error=str(e))
                await asyncio.sleep(30)
    
    async def _metrics_collection_loop(self):
        """Metrics collection and reporting loop"""
        logger.info("Starting metrics collection loop")
        
        while self._running:
            try:
                await self._collect_and_publish_metrics()
                await asyncio.sleep(30)  # Metrics collection every 30 seconds
                
            except asyncio.CancelledError:
                logger.info("Metrics collection loop cancelled")
                break
            except Exception as e:
                logger.error("Metrics collection loop error", error=str(e))
                await asyncio.sleep(60)
    
    # Agent Management Methods
    async def register_agent(self, 
                           agent_id: str, 
                           agent_type: str,
                           capabilities: List[AgentCapability],
                           anthropic_client: Optional[Any] = None,
                           tmux_session: Optional[str] = None) -> bool:
        """Register new agent with orchestrator"""
        start_time = time.time()
        
        try:
            # Circuit breaker pattern for resilience
            if self.circuit_breaker:
                return await self._register_agent_with_circuit_breaker(
                    agent_id, agent_type, capabilities, anthropic_client, tmux_session, start_time
                )
            else:
                return await self._register_agent_direct(
                    agent_id, agent_type, capabilities, anthropic_client, tmux_session, start_time
                )
                
        except Exception as e:
            logger.error("Agent registration failed", agent_id=agent_id, error=str(e))
            return False
    
    async def _register_agent_with_circuit_breaker(self, agent_id: str, agent_type: str, 
                                                 capabilities: List[AgentCapability],
                                                 anthropic_client: Optional[Any],
                                                 tmux_session: Optional[str],
                                                 start_time: float) -> bool:
        """Register agent with circuit breaker protection"""
        @self.circuit_breaker
        async def _register():
            return await self._register_agent_direct(
                agent_id, agent_type, capabilities, anthropic_client, tmux_session, start_time
            )
        
        return await _register()
    
    async def _register_agent_direct(self, agent_id: str, agent_type: str,
                                   capabilities: List[AgentCapability],
                                   anthropic_client: Optional[Any],
                                   tmux_session: Optional[str],
                                   start_time: float) -> bool:
        """Direct agent registration implementation"""
        # Check if agent already exists
        if agent_id in self._agents:
            logger.warning("Agent already registered", agent_id=agent_id)
            return False
        
        # Create agent instance
        agent = RegisteredAgent(
            agent_id=agent_id,
            agent_type=agent_type,
            capabilities=capabilities,
            state=AgentState.READY,
            anthropic_client=anthropic_client,
            tmux_session=tmux_session
        )
        
        # Register agent
        self._agents[agent_id] = agent
        
        # Index capabilities for fast lookup
        for capability in capabilities:
            self._agent_capabilities[capability.capability_type].append(agent_id)
        
        # Persist agent registration in Redis with TTL
        try:
            async with redis_session(self.redis) as redis:
                agent_data = agent.to_dict()
                await redis.setex(
                    f"agent:{agent_id}",
                    3600,  # 1 hour TTL
                    json.dumps(agent_data)
                )
        except Exception as e:
            logger.warning("Failed to persist agent registration in Redis", agent_id=agent_id, error=str(e))
        
        # Update metrics
        registration_time = time.time() - start_time
        self._agent_registration_times.append(registration_time)
        self._orchestration_metrics.agents_spawned += 1
        
        logger.info("Agent registered successfully", 
                   agent_id=agent_id, 
                   agent_type=agent_type,
                   capabilities=[cap.capability_type for cap in capabilities],
                   registration_time_ms=registration_time * 1000)
        
        return True
    
    async def deregister_agent(self, agent_id: str) -> bool:
        """Deregister agent from orchestrator"""
        try:
            if agent_id not in self._agents:
                logger.warning("Agent not found for deregistration", agent_id=agent_id)
                return False
            
            agent = self._agents[agent_id]
            
            # Cancel all active tasks for this agent
            for task_id in list(agent.current_tasks):
                await self._reassign_task(task_id)
            
            # Remove from capability index
            for capability in agent.capabilities:
                if agent_id in self._agent_capabilities[capability.capability_type]:
                    self._agent_capabilities[capability.capability_type].remove(agent_id)
            
            # Remove agent
            del self._agents[agent_id]
            
            # Remove from Redis
            try:
                async with redis_session(self.redis) as redis:
                    await redis.delete(f"agent:{agent_id}")
            except Exception as e:
                logger.warning("Failed to remove agent from Redis", agent_id=agent_id, error=str(e))
            
            # Update metrics
            self._orchestration_metrics.agents_terminated += 1
            
            logger.info("Agent deregistered successfully", agent_id=agent_id)
            return True
            
        except Exception as e:
            logger.error("Agent deregistration failed", agent_id=agent_id, error=str(e))
            return False
    
    # Task Management Methods
    async def submit_task(self, task: OrchestrationTask) -> bool:
        """Submit task for orchestration"""
        try:
            # Validate task
            if not self._validate_task(task):
                logger.error("Task validation failed", task_id=task.task_id)
                return False
            
            # Add to pending queue with priority
            if task.priority >= TaskPriority.HIGH:
                # High priority tasks go to front
                self._pending_tasks.appendleft(task)
            else:
                # Normal priority tasks go to back
                self._pending_tasks.append(task)
            
            # Persist task in Redis for fault tolerance
            try:
                async with redis_session(self.redis) as redis:
                    task_data = task.to_dict()
                    await redis.setex(
                        f"task:{task.task_id}",
                        7200,  # 2 hours TTL
                        json.dumps(task_data)
                    )
            except Exception as e:
                logger.warning("Failed to persist task in Redis", task_id=task.task_id, error=str(e))
            
            logger.debug("Task submitted successfully", 
                        task_id=task.task_id, 
                        task_type=task.task_type, 
                        priority=task.priority,
                        queue_size=len(self._pending_tasks))
            return True
            
        except Exception as e:
            logger.error("Task submission failed", task_id=task.task_id, error=str(e))
            return False
    
    def _validate_task(self, task: OrchestrationTask) -> bool:
        """Validate task before submission"""
        if not task.task_id:
            return False
        if not task.task_type:
            return False
        if not isinstance(task.required_capabilities, list):
            return False
        if task.priority not in TaskPriority:
            return False
        return True
    
    async def _process_pending_tasks(self):
        """Process pending tasks and assign to agents"""
        processed_count = 0
        max_batch_size = 10  # Process up to 10 tasks per cycle
        
        while self._pending_tasks and processed_count < max_batch_size:
            task = self._pending_tasks.popleft()
            processed_count += 1
            
            start_time = time.time()
            
            # Find best agent for task
            best_agent = await self._find_best_agent_for_task(task)
            
            if best_agent:
                success = await self._assign_task_to_agent(task, best_agent.agent_id)
                if success:
                    assignment_time = time.time() - start_time
                    self._task_assignment_times.append(assignment_time)
                    logger.debug("Task assigned successfully", 
                               task_id=task.task_id, 
                               agent_id=best_agent.agent_id,
                               assignment_time_ms=assignment_time * 1000)
                else:
                    # Failed to assign, requeue
                    self._pending_tasks.appendleft(task)
            else:
                # No suitable agent available, handle based on retry count
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    self._pending_tasks.append(task)  # Requeue at end for retry
                    logger.debug("Task requeued, no suitable agent", 
                               task_id=task.task_id, 
                               retry_count=task.retry_count)
                else:
                    # Task failed after max retries
                    task.status = "failed"
                    self._orchestration_metrics.tasks_failed += 1
                    logger.error("Task failed after max retries", task_id=task.task_id)
    
    async def _find_best_agent_for_task(self, task: OrchestrationTask) -> Optional[RegisteredAgent]:
        """Find best agent for task using intelligent routing"""
        suitable_agents = []
        
        # Find agents with required capabilities
        for required_capability in task.required_capabilities:
            agent_ids = self._agent_capabilities.get(required_capability, [])
            for agent_id in agent_ids:
                if agent_id in self._agents:
                    agent = self._agents[agent_id]
                    if self._is_agent_suitable_for_task(agent, task):
                        suitable_agents.append(agent)
        
        if not suitable_agents:
            return None
        
        # Remove duplicates
        suitable_agents = list({agent.agent_id: agent for agent in suitable_agents}.values())
        
        # Apply routing strategy
        return await self._select_agent_by_strategy(suitable_agents, task)
    
    def _is_agent_suitable_for_task(self, agent: RegisteredAgent, task: OrchestrationTask) -> bool:
        """Check if agent is suitable for task"""
        # Check agent state
        if agent.state not in (AgentState.READY, AgentState.IDLE):
            return False
        
        # Check task capacity
        max_tasks = max((cap.max_concurrent_tasks for cap in agent.capabilities), default=1)
        if len(agent.current_tasks) >= max_tasks:
            return False
        
        # Check context window usage for high priority tasks
        if task.priority >= TaskPriority.HIGH and agent.context_window_usage > 0.8:
            return False
        
        # Check error rate
        if agent.total_tasks_completed > 0:
            error_rate = agent.error_count / agent.total_tasks_completed
            if error_rate > 0.1:  # 10% error rate threshold
                return False
        
        return True
    
    async def _select_agent_by_strategy(self, suitable_agents: List[RegisteredAgent], 
                                      task: OrchestrationTask) -> Optional[RegisteredAgent]:
        """Select agent based on orchestration strategy"""
        if not suitable_agents:
            return None
        
        if self._strategy == OrchestrationStrategy.ROUND_ROBIN:
            # Simple round-robin selection
            return suitable_agents[len(self._active_tasks) % len(suitable_agents)]
        
        elif self._strategy == OrchestrationStrategy.LOAD_BALANCED:
            # Select agent with least current tasks
            return min(suitable_agents, key=lambda a: len(a.current_tasks))
        
        elif self._strategy == OrchestrationStrategy.CAPABILITY_BASED:
            # Select agent with highest skill level for required capabilities
            def capability_score(agent):
                total_score = 0
                for required_cap in task.required_capabilities:
                    for cap in agent.capabilities:
                        if cap.capability_type == required_cap:
                            total_score += cap.skill_level * cap.confidence_level
                return total_score
            
            return max(suitable_agents, key=capability_score)
        
        elif self._strategy == OrchestrationStrategy.PRIORITY_BASED:
            # For high priority tasks, prefer agents with better performance
            if task.priority >= TaskPriority.HIGH:
                return min(suitable_agents, key=lambda a: a.average_task_time or float('inf'))
            else:
                return min(suitable_agents, key=lambda a: len(a.current_tasks))
        
        elif self._strategy == OrchestrationStrategy.PERFORMANCE_OPTIMIZED:
            # Select based on comprehensive performance metrics
            def performance_score(agent):
                score = 0
                
                # Performance factor (lower average time is better)
                if agent.average_task_time > 0:
                    score += 100 / agent.average_task_time  # Inverse relationship
                
                # Error rate factor (lower is better)
                if agent.total_tasks_completed > 0:
                    error_rate = agent.error_count / agent.total_tasks_completed
                    score += 100 * (1 - error_rate)
                
                # Capability match factor
                for required_cap in task.required_capabilities:
                    for cap in agent.capabilities:
                        if cap.capability_type == required_cap:
                            score += cap.skill_level * cap.confidence_level
                
                return score
            
            return max(suitable_agents, key=performance_score)
        
        else:  # INTELLIGENT strategy (default)
            # Comprehensive intelligent scoring
            def intelligent_score(agent):
                score = 0
                
                # Load factor (lower is better)
                max_tasks = max((cap.max_concurrent_tasks for cap in agent.capabilities), default=1)
                load_factor = len(agent.current_tasks) / max_tasks
                score -= load_factor * 20
                
                # Performance factor
                if agent.average_task_time > 0:
                    score += 50 / agent.average_task_time
                
                # Capability match factor
                for required_cap in task.required_capabilities:
                    for cap in agent.capabilities:
                        if cap.capability_type == required_cap:
                            score += cap.skill_level * cap.confidence_level * 2
                
                # Error rate factor
                if agent.total_tasks_completed > 0:
                    error_rate = agent.error_count / agent.total_tasks_completed
                    score += 30 * (1 - error_rate)
                
                # Context window usage factor (lower is better)
                score -= agent.context_window_usage * 10
                
                # Recency factor (recently active agents get slight preference)
                time_since_heartbeat = (datetime.utcnow() - agent.last_heartbeat).total_seconds()
                score += max(0, 10 - time_since_heartbeat / 60)  # Decay over minutes
                
                return score
            
            return max(suitable_agents, key=intelligent_score)
    
    async def _assign_task_to_agent(self, task: OrchestrationTask, agent_id: str) -> bool:
        """Assign task to specific agent"""
        try:
            if agent_id not in self._agents:
                logger.error("Agent not found for task assignment", agent_id=agent_id)
                return False
            
            agent = self._agents[agent_id]
            
            # Update task assignment
            task.assigned_agent = agent_id
            task.status = "assigned"
            
            # Update agent state
            agent.current_tasks.add(task.task_id)
            agent.state = AgentState.BUSY
            
            # Track active task
            self._active_tasks[task.task_id] = task
            
            # Send task to agent via messaging
            task_message = Message(
                type=MessageType.TASK_REQUEST,
                sender="production_orchestrator",
                recipient=agent_id,
                priority=MessagePriority.HIGH if task.priority >= TaskPriority.HIGH else MessagePriority.NORMAL,
                payload={
                    "command": "execute_task",
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "task_data": task.payload,
                    "priority": task.priority,
                    "deadline": task.deadline.isoformat() if task.deadline else None,
                    "estimated_duration": task.estimated_duration,
                    "resource_requirements": task.resource_requirements
                }
            )
            
            success = await self.messaging.send_message(task_message)
            if not success:
                # Failed to send message, rollback assignment
                agent.current_tasks.discard(task.task_id)
                if not agent.current_tasks:
                    agent.state = AgentState.IDLE
                del self._active_tasks[task.task_id]
                task.assigned_agent = None
                task.status = "pending"
                return False
            
            # Update metrics
            self._orchestration_metrics.tasks_processed += 1
            
            logger.info("Task assigned to agent successfully", 
                       task_id=task.task_id, 
                       agent_id=agent_id, 
                       task_type=task.task_type)
            
            return True
            
        except Exception as e:
            logger.error("Task assignment failed", 
                        task_id=task.task_id, 
                        agent_id=agent_id, 
                        error=str(e))
            return False
    
    async def _reassign_task(self, task_id: str):
        """Reassign task to different agent"""
        if task_id in self._active_tasks:
            task = self._active_tasks[task_id]
            
            # Remove from current agent
            if task.assigned_agent and task.assigned_agent in self._agents:
                agent = self._agents[task.assigned_agent]
                agent.current_tasks.discard(task_id)
                if not agent.current_tasks:
                    agent.state = AgentState.IDLE
            
            # Reset task for reassignment
            task.assigned_agent = None
            task.status = "pending"
            task.retry_count += 1
            
            # Remove from active tasks
            del self._active_tasks[task_id]
            
            # Requeue for assignment
            self._pending_tasks.appendleft(task)
            
            logger.info("Task reassigned", task_id=task_id)
    
    async def _check_task_timeouts(self):
        """Check for and handle task timeouts"""
        current_time = datetime.utcnow()
        timed_out_tasks = []
        
        for task_id, task in self._active_tasks.items():
            # Check deadline timeout
            if task.deadline and current_time > task.deadline:
                timed_out_tasks.append(task_id)
            # Check general timeout
            elif (current_time - task.created_at) > self._task_timeout:
                timed_out_tasks.append(task_id)
        
        for task_id in timed_out_tasks:
            logger.warning("Task timeout detected", task_id=task_id)
            await self._handle_task_timeout(task_id)
    
    async def _handle_task_timeout(self, task_id: str):
        """Handle task timeout"""
        if task_id not in self._active_tasks:
            return
        
        task = self._active_tasks[task_id]
        
        # Update agent error count
        if task.assigned_agent and task.assigned_agent in self._agents:
            agent = self._agents[task.assigned_agent]
            agent.error_count += 1
            agent.current_tasks.discard(task_id)
            if not agent.current_tasks:
                agent.state = AgentState.IDLE
        
        # Try to reassign if retries available
        if task.retry_count < task.max_retries:
            await self._reassign_task(task_id)
        else:
            # Mark as failed
            task.status = "failed"
            del self._active_tasks[task_id]
            self._orchestration_metrics.tasks_failed += 1
            logger.error("Task failed due to timeout", task_id=task_id)
    
    # Performance and Monitoring Methods
    async def _check_agent_health(self):
        """Check health of all registered agents"""
        current_time = datetime.utcnow()
        stale_threshold = timedelta(seconds=self._heartbeat_interval * 2)
        
        for agent_id, agent in list(self._agents.items()):
            time_since_heartbeat = current_time - agent.last_heartbeat
            
            if time_since_heartbeat > stale_threshold:
                logger.warning("Agent heartbeat timeout", 
                             agent_id=agent_id, 
                             time_since_heartbeat=time_since_heartbeat.total_seconds())
                
                # Mark agent as having error
                agent.state = AgentState.ERROR
                
                # Reassign tasks if agent is unresponsive for too long
                if time_since_heartbeat > timedelta(minutes=5):
                    for task_id in list(agent.current_tasks):
                        await self._reassign_task(task_id)
    
    async def _update_agent_performance_metrics(self):
        """Update performance metrics for all agents"""
        for agent_id, agent in self._agents.items():
            # Calculate performance metrics
            if agent.total_tasks_completed > 0:
                # Update average task time from history
                history = self._agent_performance_history[agent_id]
                if history:
                    agent.average_task_time = statistics.mean(history)
    
    async def _cleanup_stale_agents(self):
        """Remove stale agents that haven't been active"""
        current_time = datetime.utcnow()
        stale_threshold = timedelta(hours=1)  # Remove agents inactive for 1 hour
        
        agents_to_remove = []
        
        for agent_id, agent in self._agents.items():
            time_since_heartbeat = current_time - agent.last_heartbeat
            
            if (time_since_heartbeat > stale_threshold and 
                agent.state in (AgentState.ERROR, AgentState.TERMINATED) and
                not agent.current_tasks):
                agents_to_remove.append(agent_id)
        
        for agent_id in agents_to_remove:
            await self.deregister_agent(agent_id)
            logger.info("Removed stale agent", agent_id=agent_id)
    
    async def _monitor_system_resources(self):
        """Monitor system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._system_metrics["cpu_usage"] = cpu_percent
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self._system_metrics["memory_usage"] = memory_percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self._system_metrics["disk_usage"] = disk_percent
            
            # Update orchestration metrics
            self._orchestration_metrics.cpu_usage = cpu_percent
            self._orchestration_metrics.memory_usage = memory_percent
            
            # Log warnings for high resource usage
            if cpu_percent > 80:
                logger.warning("High CPU usage detected", cpu_percent=cpu_percent)
            if memory_percent > 80:
                logger.warning("High memory usage detected", memory_percent=memory_percent)
            if disk_percent > 80:
                logger.warning("High disk usage detected", disk_percent=disk_percent)
                
        except Exception as e:
            logger.error("Failed to monitor system resources", error=str(e))
    
    async def _optimize_agent_allocation(self):
        """Optimize agent allocation based on current load"""
        if not self._auto_scaling_enabled:
            return
        
        try:
            # Calculate current load metrics
            pending_tasks = len(self._pending_tasks)
            active_agents = len([a for a in self._agents.values() if a.state != AgentState.TERMINATED])
            busy_agents = len([a for a in self._agents.values() if a.state == AgentState.BUSY])
            
            # Calculate load ratio
            load_ratio = busy_agents / max(active_agents, 1)
            
            # Auto-scaling decisions
            if pending_tasks > 10 and load_ratio > 0.8 and active_agents < self._max_agents:
                logger.info("High load detected, considering scale-up", 
                          pending_tasks=pending_tasks, 
                          load_ratio=load_ratio,
                          active_agents=active_agents)
                # Note: Actual agent spawning would be handled by external agent manager
                
            elif pending_tasks == 0 and load_ratio < 0.2 and active_agents > 5:
                logger.info("Low load detected, considering scale-down",
                          pending_tasks=pending_tasks,
                          load_ratio=load_ratio,
                          active_agents=active_agents)
                # Note: Actual agent termination would be handled by external agent manager
            
            # Update load metrics
            self._orchestration_metrics.current_load = load_ratio
            
        except Exception as e:
            logger.error("Failed to optimize agent allocation", error=str(e))
    
    async def _update_orchestration_metrics(self):
        """Update comprehensive orchestration metrics"""
        try:
            current_time = datetime.utcnow()
            
            # Update basic counts
            self._orchestration_metrics.timestamp = current_time
            self._orchestration_metrics.active_agents = len([a for a in self._agents.values() 
                                                           if a.state != AgentState.TERMINATED])
            self._orchestration_metrics.busy_agents = len([a for a in self._agents.values() 
                                                         if a.state == AgentState.BUSY])
            self._orchestration_metrics.pending_tasks = len(self._pending_tasks)
            self._orchestration_metrics.active_tasks = len(self._active_tasks)
            
            # Calculate average registration time
            if self._agent_registration_times:
                self._orchestration_metrics.average_agent_registration_time = statistics.mean(self._agent_registration_times)
            
            # Calculate average task assignment time
            if self._task_assignment_times:
                self._orchestration_metrics.average_task_assignment_time = statistics.mean(self._task_assignment_times)
            
            # Calculate system health score
            health_factors = []
            
            # Resource health (CPU, memory)
            if "cpu_usage" in self._system_metrics:
                cpu_health = max(0, (100 - self._system_metrics["cpu_usage"]) / 100)
                health_factors.append(cpu_health)
            
            if "memory_usage" in self._system_metrics:
                memory_health = max(0, (100 - self._system_metrics["memory_usage"]) / 100)
                health_factors.append(memory_health)
            
            # Agent health
            if self._orchestration_metrics.active_agents > 0:
                error_agents = len([a for a in self._agents.values() if a.state == AgentState.ERROR])
                agent_health = max(0, 1 - (error_agents / self._orchestration_metrics.active_agents))
                health_factors.append(agent_health)
            
            # Task success rate
            total_tasks = self._orchestration_metrics.tasks_processed + self._orchestration_metrics.tasks_failed
            if total_tasks > 0:
                task_success_rate = self._orchestration_metrics.tasks_processed / total_tasks
                health_factors.append(task_success_rate)
            
            # Calculate overall health score
            if health_factors:
                self._orchestration_metrics.system_health_score = statistics.mean(health_factors)
            
        except Exception as e:
            logger.error("Failed to update orchestration metrics", error=str(e))
    
    async def _perform_health_checks(self):
        """Perform comprehensive system health checks"""
        try:
            health_status = {
                "timestamp": datetime.utcnow().isoformat(),
                "orchestrator_running": self._running,
                "total_agents": len(self._agents),
                "active_agents": len([a for a in self._agents.values() if a.state != AgentState.TERMINATED]),
                "pending_tasks": len(self._pending_tasks),
                "active_tasks": len(self._active_tasks),
                "system_health_score": self._orchestration_metrics.system_health_score,
                "redis_connected": False,
                "messaging_healthy": False
            }
            
            # Check Redis connectivity
            try:
                async with redis_session(self.redis) as redis:
                    await redis.ping()
                    health_status["redis_connected"] = True
            except Exception:
                health_status["redis_connected"] = False
            
            # Check messaging service health
            try:
                health_status["messaging_healthy"] = await self.messaging.health_check()
            except Exception:
                health_status["messaging_healthy"] = False
            
            # Store health status in Redis
            try:
                async with redis_session(self.redis) as redis:
                    await redis.setex(
                        "orchestrator:health_status",
                        300,  # 5 minutes TTL
                        json.dumps(health_status)
                    )
            except Exception as e:
                logger.warning("Failed to store health status", error=str(e))
            
            # Log health warnings
            if not health_status["redis_connected"]:
                logger.warning("Redis connectivity check failed")
            if not health_status["messaging_healthy"]:
                logger.warning("Messaging service health check failed")
            if health_status["system_health_score"] < 0.7:
                logger.warning("Low system health score", score=health_status["system_health_score"])
                
        except Exception as e:
            logger.error("Health check failed", error=str(e))
    
    async def _collect_and_publish_metrics(self):
        """Collect and publish orchestration metrics"""
        try:
            metrics = self.get_orchestration_metrics()
            
            # Store metrics in Redis for dashboard consumption
            async with redis_session(self.redis) as redis:
                await redis.setex(
                    "orchestrator:metrics",
                    300,  # 5 minutes TTL
                    json.dumps(metrics)
                )
            
            # Log key metrics periodically
            logger.info("Orchestration metrics updated",
                       active_agents=metrics["active_agents"],
                       pending_tasks=metrics["pending_tasks"],
                       tasks_processed=metrics["tasks_processed"],
                       system_health_score=metrics["system_health_score"])
                       
        except Exception as e:
            logger.error("Failed to collect and publish metrics", error=str(e))
    
    async def _terminate_all_agents(self):
        """Gracefully terminate all agents"""
        logger.info("Terminating all agents...")
        
        # Send shutdown messages to all active agents
        shutdown_tasks = []
        for agent_id, agent in self._agents.items():
            if agent.state not in (AgentState.TERMINATED, AgentState.TERMINATING):
                agent.state = AgentState.TERMINATING
                
                # Send shutdown message
                shutdown_message = Message(
                    type=MessageType.COMMAND,
                    sender="production_orchestrator",
                    recipient=agent_id,
                    priority=MessagePriority.HIGH,
                    payload={"command": "shutdown", "reason": "orchestrator_shutdown"}
                )
                
                shutdown_tasks.append(self.messaging.send_message(shutdown_message))
        
        # Wait for shutdown messages to be sent
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        # Wait a bit for agents to process shutdown
        await asyncio.sleep(5)
        
        # Force cleanup of remaining agents
        for agent_id in list(self._agents.keys()):
            await self.deregister_agent(agent_id)
        
        logger.info("All agents terminated")
    
    # Public API Methods
    def get_orchestration_metrics(self) -> Dict[str, Any]:
        """Get comprehensive orchestration metrics"""
        return {
            "timestamp": self._orchestration_metrics.timestamp.isoformat(),
            "tasks_processed": self._orchestration_metrics.tasks_processed,
            "tasks_failed": self._orchestration_metrics.tasks_failed,
            "agents_spawned": self._orchestration_metrics.agents_spawned,
            "agents_terminated": self._orchestration_metrics.agents_terminated,
            "average_task_time": self._orchestration_metrics.average_task_time,
            "current_load": self._orchestration_metrics.current_load,
            "active_agents": self._orchestration_metrics.active_agents,
            "busy_agents": self._orchestration_metrics.busy_agents,
            "pending_tasks": self._orchestration_metrics.pending_tasks,
            "active_tasks": self._orchestration_metrics.active_tasks,
            "cpu_usage": self._orchestration_metrics.cpu_usage,
            "memory_usage": self._orchestration_metrics.memory_usage,
            "average_agent_registration_time": self._orchestration_metrics.average_agent_registration_time,
            "average_task_assignment_time": self._orchestration_metrics.average_task_assignment_time,
            "system_health_score": self._orchestration_metrics.system_health_score,
            "orchestration_strategy": self._strategy.value,
            "max_agents": self._max_agents,
            "auto_scaling_enabled": self._auto_scaling_enabled,
            "total_agents": len(self._agents),
            "system_metrics": self._system_metrics.copy()
        }
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific agent"""
        if agent_id not in self._agents:
            return None
        
        agent = self._agents[agent_id]
        return agent.to_dict()
    
    def get_all_agents_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents"""
        return {agent_id: agent.to_dict() for agent_id, agent in self._agents.items()}
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific task"""
        # Check active tasks
        if task_id in self._active_tasks:
            return self._active_tasks[task_id].to_dict()
        
        # Check pending tasks
        for task in self._pending_tasks:
            if task.task_id == task_id:
                return task.to_dict()
        
        # Check completed tasks
        for task in self._completed_tasks:
            if task.task_id == task_id:
                return task.to_dict()
        
        return None
    
    async def update_agent_heartbeat(self, agent_id: str, context_usage: Optional[float] = None):
        """Update agent heartbeat and context usage"""
        if agent_id in self._agents:
            agent = self._agents[agent_id]
            agent.last_heartbeat = datetime.utcnow()
            
            if context_usage is not None:
                agent.context_window_usage = min(1.0, max(0.0, context_usage))
            
            # Update state if agent was in error
            if agent.state == AgentState.ERROR:
                agent.state = AgentState.IDLE if not agent.current_tasks else AgentState.BUSY
            
            logger.debug("Agent heartbeat updated", agent_id=agent_id, context_usage=context_usage)
    
    async def complete_task(self, task_id: str, result: Dict[str, Any], agent_id: str):
        """Mark task as completed with result"""
        if task_id not in self._active_tasks:
            logger.warning("Task not found for completion", task_id=task_id)
            return
        
        task = self._active_tasks[task_id]
        
        # Update task
        task.status = "completed"
        completion_time = datetime.utcnow()
        task_duration = (completion_time - task.created_at).total_seconds()
        
        # Update agent
        if agent_id in self._agents:
            agent = self._agents[agent_id]
            agent.current_tasks.discard(task_id)
            agent.total_tasks_completed += 1
            
            # Update performance history
            self._agent_performance_history[agent_id].append(task_duration)
            
            # Update agent state
            agent.state = AgentState.IDLE if not agent.current_tasks else AgentState.BUSY
        
        # Move to completed tasks
        self._completed_tasks.append(task)
        del self._active_tasks[task_id]
        
        # Execute callback if provided
        if task.callback:
            try:
                await task.callback(task_id, result)
            except Exception as e:
                logger.error("Task completion callback failed", task_id=task_id, error=str(e))
        
        logger.info("Task completed successfully", 
                   task_id=task_id, 
                   agent_id=agent_id,
                   duration_seconds=task_duration)
    
    async def fail_task(self, task_id: str, error: str, agent_id: str):
        """Mark task as failed with error"""
        if task_id not in self._active_tasks:
            logger.warning("Task not found for failure", task_id=task_id)
            return
        
        task = self._active_tasks[task_id]
        
        # Update task
        task.status = "failed"
        
        # Update agent
        if agent_id in self._agents:
            agent = self._agents[agent_id]
            agent.current_tasks.discard(task_id)
            agent.error_count += 1
            
            # Update agent state
            agent.state = AgentState.IDLE if not agent.current_tasks else AgentState.BUSY
        
        # Move to completed tasks
        self._completed_tasks.append(task)
        del self._active_tasks[task_id]
        
        # Update metrics
        self._orchestration_metrics.tasks_failed += 1
        
        logger.error("Task failed", task_id=task_id, agent_id=agent_id, error=error)
    
    def set_orchestration_strategy(self, strategy: OrchestrationStrategy):
        """Set orchestration strategy"""
        self._strategy = strategy
        logger.info("Orchestration strategy updated", strategy=strategy.value)
    
    def set_auto_scaling(self, enabled: bool):
        """Enable or disable auto-scaling"""
        self._auto_scaling_enabled = enabled
        logger.info("Auto-scaling setting updated", enabled=enabled)


# Convenience functions
def get_production_orchestrator() -> UnifiedProductionOrchestrator:
    """Get unified production orchestrator instance"""
    return UnifiedProductionOrchestrator()


async def submit_orchestration_task(task_type: str, 
                                  required_capabilities: List[str],
                                  payload: Dict[str, Any],
                                  priority: TaskPriority = TaskPriority.NORMAL,
                                  deadline: Optional[datetime] = None,
                                  callback: Optional[Callable] = None) -> str:
    """Submit task for orchestration"""
    orchestrator = get_production_orchestrator()
    
    task = OrchestrationTask(
        task_type=task_type,
        required_capabilities=required_capabilities,
        payload=payload,
        priority=priority,
        deadline=deadline,
        callback=callback
    )
    
    success = await orchestrator.submit_task(task)
    return task.task_id if success else ""


async def register_orchestration_agent(agent_id: str,
                                     agent_type: str, 
                                     capabilities: List[str],
                                     skill_levels: Optional[List[int]] = None) -> bool:
    """Register agent with orchestrator"""
    orchestrator = get_production_orchestrator()
    
    # Convert capabilities to AgentCapability objects
    agent_capabilities = []
    for i, capability_type in enumerate(capabilities):
        skill_level = skill_levels[i] if skill_levels and i < len(skill_levels) else 5
        agent_capabilities.append(AgentCapability(
            capability_type=capability_type,
            skill_level=skill_level
        ))
    
    return await orchestrator.register_agent(agent_id, agent_type, agent_capabilities)


# Export key classes and functions
__all__ = [
    'UnifiedProductionOrchestrator',
    'AgentState',
    'OrchestrationStrategy', 
    'TaskPriority',
    'AgentCapability',
    'RegisteredAgent',
    'OrchestrationTask',
    'OrchestrationMetrics',
    'get_production_orchestrator',
    'submit_orchestration_task',
    'register_orchestration_agent'
]