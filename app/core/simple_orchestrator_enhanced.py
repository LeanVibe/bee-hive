"""
Enhanced Simple Orchestrator for LeanVibe Agent Hive 2.0

A production-ready orchestrator that provides core agent management functionality
with comprehensive error handling, database integration, and monitoring.
Designed for <100ms response times with fault tolerance and observability.

This implementation replaces the existing simple orchestrator with:
- Comprehensive error handling with custom exceptions
- Proper async/await patterns for database operations
- Configuration support for different deployment modes
- Comprehensive logging using structlog
- Metrics and performance monitoring
- Advanced task assignment logic (round-robin or availability-based)
- Compatibility with existing API endpoints
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Protocol, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from contextlib import asynccontextmanager
import statistics
from collections import defaultdict, deque

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_, func
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError
from anthropic import AsyncAnthropic

from .config import settings
from .database import get_session, DatabaseHealthCheck
from .logging_service import get_component_logger
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.task import Task, TaskStatus, TaskPriority, TaskType

logger = get_component_logger("simple_orchestrator_enhanced")


# Enums and Data Structures
class AgentRole(Enum):
    """Core agent roles for the orchestrator."""
    BACKEND_DEVELOPER = "backend_developer"
    FRONTEND_DEVELOPER = "frontend_developer"
    DEVOPS_ENGINEER = "devops_engineer"
    QA_ENGINEER = "qa_engineer"
    ARCHITECT = "architect"
    DATA_ENGINEER = "data_engineer"


class TaskAssignmentStrategy(Enum):
    """Task assignment strategies."""
    ROUND_ROBIN = "round_robin"
    AVAILABILITY_BASED = "availability_based"
    CAPABILITY_MATCH = "capability_match"
    PERFORMANCE_BASED = "performance_based"


class OrchestratorMode(Enum):
    """Orchestrator deployment modes."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


# Custom Exceptions for comprehensive error handling
class SimpleOrchestratorError(Exception):
    """Base exception for orchestrator errors."""
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "ORCHESTRATOR_ERROR"
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class AgentNotFoundError(SimpleOrchestratorError):
    """Raised when agent is not found."""
    def __init__(self, agent_id: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"Agent {agent_id} not found", "AGENT_NOT_FOUND", details)
        self.agent_id = agent_id


class TaskDelegationError(SimpleOrchestratorError):
    """Raised when task delegation fails."""
    def __init__(self, message: str, task_details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "TASK_DELEGATION_ERROR", task_details)


class DatabaseOperationError(SimpleOrchestratorError):
    """Raised when database operations fail."""
    def __init__(self, operation: str, error: Exception, details: Optional[Dict[str, Any]] = None):
        super().__init__(f"Database operation '{operation}' failed: {str(error)}", "DATABASE_ERROR", details)
        self.operation = operation
        self.original_error = error


class ConfigurationError(SimpleOrchestratorError):
    """Raised when configuration is invalid."""
    def __init__(self, config_key: str, issue: str):
        super().__init__(f"Configuration error for '{config_key}': {issue}", "CONFIG_ERROR")
        self.config_key = config_key


class ResourceLimitError(SimpleOrchestratorError):
    """Raised when resource limits are exceeded."""
    def __init__(self, resource: str, limit: int, current: int):
        super().__init__(f"{resource} limit exceeded: {current}/{limit}", "RESOURCE_LIMIT_ERROR")
        self.resource = resource
        self.limit = limit
        self.current = current


# Enhanced Data Classes
@dataclass
class AgentInstance:
    """Enhanced agent instance representation with metrics."""
    id: str
    role: AgentRole
    status: AgentStatus
    current_task_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: Optional[datetime] = None
    capabilities: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    load_score: float = 0.0  # Current load (0.0 to 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "role": self.role.value,
            "status": self.status.value,
            "current_task_id": self.current_task_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "capabilities": self.capabilities,
            "performance_metrics": self.performance_metrics,
            "config": self.config,
            "load_score": self.load_score
        }
    
    def is_healthy(self) -> bool:
        """Check if agent is healthy based on heartbeat."""
        if not self.last_heartbeat:
            return True  # Newly created agents are considered healthy
        
        heartbeat_timeout = timedelta(minutes=5)  # 5 minute timeout
        return datetime.utcnow() - self.last_heartbeat < heartbeat_timeout
    
    def update_heartbeat(self) -> None:
        """Update agent heartbeat and last activity."""
        now = datetime.utcnow()
        self.last_heartbeat = now
        self.last_activity = now
    
    def is_available_for_task(self) -> bool:
        """Check if agent is available to take new tasks."""
        return (
            self.status == AgentStatus.active and
            self.current_task_id is None and
            self.is_healthy() and
            self.load_score < 0.8  # Not overloaded
        )
    
    def calculate_task_suitability(self, required_capabilities: List[str], task_type: str) -> float:
        """Calculate suitability score for a task (0.0 to 1.0)."""
        if not self.is_available_for_task():
            return 0.0
        
        base_score = 0.5  # Base score for available agents
        
        # Capability matching
        if required_capabilities:
            capability_matches = sum(1 for cap in required_capabilities if cap in self.capabilities)
            capability_score = capability_matches / len(required_capabilities)
            base_score += capability_score * 0.3
        
        # Role matching
        role_task_mapping = {
            AgentRole.BACKEND_DEVELOPER: ["backend", "api", "database", "server"],
            AgentRole.FRONTEND_DEVELOPER: ["frontend", "ui", "react", "javascript"],
            AgentRole.DEVOPS_ENGINEER: ["deployment", "infrastructure", "docker", "kubernetes"],
            AgentRole.QA_ENGINEER: ["testing", "quality", "automation"],
            AgentRole.ARCHITECT: ["architecture", "design", "planning"],
            AgentRole.DATA_ENGINEER: ["data", "etl", "analytics"]
        }
        
        role_keywords = role_task_mapping.get(self.role, [])
        if any(keyword in task_type.lower() for keyword in role_keywords):
            base_score += 0.2
        
        # Load balancing - prefer less loaded agents
        load_penalty = self.load_score * 0.1
        base_score -= load_penalty
        
        return min(1.0, max(0.0, base_score))


@dataclass
class TaskAssignment:
    """Enhanced task assignment representation."""
    task_id: str
    agent_id: str
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    estimated_duration: Optional[int] = None  # minutes
    actual_duration: Optional[int] = None  # minutes
    suitability_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "assigned_at": self.assigned_at.isoformat(),
            "status": self.status.value,
            "priority": self.priority.value,
            "estimated_duration": self.estimated_duration,
            "actual_duration": self.actual_duration,
            "suitability_score": self.suitability_score
        }


@dataclass
class PerformanceMetrics:
    """Performance tracking for the orchestrator."""
    operation_count: int = 0
    success_count: int = 0
    error_count: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_reset: datetime = field(default_factory=datetime.utcnow)
    
    def record_operation(self, success: bool, response_time_ms: float) -> None:
        """Record an operation result."""
        self.operation_count += 1
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        self.response_times.append(response_time_ms)
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.operation_count == 0:
            return 1.0
        return self.success_count / self.operation_count
    
    def get_average_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    def get_p95_response_time(self) -> float:
        """Calculate 95th percentile response time."""
        if len(self.response_times) < 20:
            return self.get_average_response_time()
        return statistics.quantiles(self.response_times, n=20)[18]  # 95th percentile


# Configuration and dependency injection protocols
class DatabaseDependency(Protocol):
    """Protocol for database dependency injection."""
    async def get_session(self) -> AsyncSession: ...


class CacheDependency(Protocol):
    """Protocol for cache dependency injection."""
    async def get(self, key: str) -> Optional[Any]: ...
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool: ...


class MetricsDependency(Protocol):
    """Protocol for metrics dependency injection."""
    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> None: ...
    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None: ...


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    mode: OrchestratorMode = OrchestratorMode.DEVELOPMENT
    max_concurrent_agents: int = 10
    default_task_assignment_strategy: TaskAssignmentStrategy = TaskAssignmentStrategy.AVAILABILITY_BASED
    enable_performance_monitoring: bool = True
    enable_database_persistence: bool = True
    enable_caching: bool = True
    heartbeat_interval_seconds: int = 30
    task_timeout_minutes: int = 60
    agent_spawn_timeout_seconds: int = 30
    database_retry_attempts: int = 3
    database_retry_delay_seconds: float = 1.0
    
    @classmethod
    def from_settings(cls) -> 'OrchestratorConfig':
        """Create config from application settings."""
        mode_map = {
            "development": OrchestratorMode.DEVELOPMENT,
            "staging": OrchestratorMode.STAGING,
            "production": OrchestratorMode.PRODUCTION,
            "test": OrchestratorMode.TEST
        }
        
        return cls(
            mode=mode_map.get(settings.ENVIRONMENT, OrchestratorMode.DEVELOPMENT),
            max_concurrent_agents=getattr(settings, 'MAX_CONCURRENT_AGENTS', 10),
            enable_performance_monitoring=getattr(settings, 'PROMETHEUS_METRICS_ENABLED', True),
            enable_database_persistence=not settings.DEBUG,  # Disable in debug mode for faster iteration
            enable_caching=True,
            heartbeat_interval_seconds=getattr(settings, 'AGENT_HEARTBEAT_INTERVAL', 30),
            task_timeout_minutes=getattr(settings, 'AGENT_TASK_TIMEOUT_MINUTES', 60)
        )


class EnhancedSimpleOrchestrator:
    """
    Enhanced simple orchestrator with production-ready features.
    
    Provides:
    - Agent lifecycle management (spawn, shutdown, monitoring)
    - Advanced task delegation with multiple strategies
    - Comprehensive error handling and recovery
    - Performance monitoring and metrics
    - Database persistence with retry logic
    - Caching for fast operations
    - Configuration support for different environments
    
    Design goals:
    - <100ms response times for core operations
    - Fault tolerance and graceful degradation
    - Comprehensive observability
    - Easy testing and maintenance
    """
    
    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        db_session_factory: Optional[DatabaseDependency] = None,
        cache: Optional[CacheDependency] = None,
        metrics: Optional[MetricsDependency] = None,
        anthropic_client: Optional[AsyncAnthropic] = None
    ):
        """Initialize orchestrator with dependency injection."""
        self.config = config or OrchestratorConfig.from_settings()
        self._db_session_factory = db_session_factory
        self._cache = cache
        self._metrics = metrics
        self._anthropic_client = anthropic_client or AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY
        ) if hasattr(settings, 'ANTHROPIC_API_KEY') else None
        
        # In-memory state
        self._agents: Dict[str, AgentInstance] = {}
        self._task_assignments: Dict[str, TaskAssignment] = {}
        self._round_robin_index = 0
        
        # Performance tracking
        self._metrics_tracker = PerformanceMetrics()
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        logger.info(
            "Enhanced SimpleOrchestrator initialized",
            mode=self.config.mode.value,
            max_agents=self.config.max_concurrent_agents,
            assignment_strategy=self.config.default_task_assignment_strategy.value
        )
    
    async def start(self) -> None:
        """Start the orchestrator and background tasks."""
        try:
            # Validate configuration
            await self._validate_configuration()
            
            # Start background tasks
            if self.config.enable_performance_monitoring:
                self._background_tasks.append(
                    asyncio.create_task(self._heartbeat_monitor())
                )
                self._background_tasks.append(
                    asyncio.create_task(self._performance_monitor())
                )
            
            # Load existing agents from database if enabled
            if self.config.enable_database_persistence:
                await self._load_agents_from_database()
            
            logger.info("Enhanced SimpleOrchestrator started successfully")
            
        except Exception as e:
            logger.error("Failed to start orchestrator", error=str(e))
            raise SimpleOrchestratorError(f"Failed to start orchestrator: {e}") from e
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator."""
        try:
            logger.info("Shutting down Enhanced SimpleOrchestrator")
            
            # Signal shutdown to background tasks
            self._shutdown_event.set()
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Wait for tasks to complete with timeout
            if self._background_tasks:
                await asyncio.wait_for(
                    asyncio.gather(*self._background_tasks, return_exceptions=True),
                    timeout=10.0
                )
            
            # Shutdown all agents gracefully
            agent_ids = list(self._agents.keys())
            for agent_id in agent_ids:
                try:
                    await self.shutdown_agent(agent_id, graceful=True)
                except Exception as e:
                    logger.warning("Error shutting down agent", agent_id=agent_id, error=str(e))
            
            logger.info("Enhanced SimpleOrchestrator shutdown complete")
            
        except Exception as e:
            logger.error("Error during orchestrator shutdown", error=str(e))
            raise
    
    @asynccontextmanager
    async def _database_operation(self, operation_name: str):
        """Context manager for database operations with retry logic."""
        if not self.config.enable_database_persistence:
            yield None
            return
        
        for attempt in range(self.config.database_retry_attempts):
            try:
                async with get_session() as session:
                    yield session
                    return
            except (SQLAlchemyError, OperationalError) as e:
                logger.warning(
                    "Database operation failed, retrying",
                    operation=operation_name,
                    attempt=attempt + 1,
                    max_attempts=self.config.database_retry_attempts,
                    error=str(e)
                )
                
                if attempt == self.config.database_retry_attempts - 1:
                    raise DatabaseOperationError(operation_name, e)
                
                await asyncio.sleep(self.config.database_retry_delay_seconds * (2 ** attempt))
    
    def _record_operation(self, operation_name: str, success: bool, response_time_ms: float) -> None:
        """Record operation metrics."""
        self._metrics_tracker.record_operation(success, response_time_ms)
        
        if self._metrics:
            tags = {"operation": operation_name}
            self._metrics.increment_counter("orchestrator_operations_total", tags)
            if success:
                self._metrics.increment_counter("orchestrator_operations_success_total", tags)
            else:
                self._metrics.increment_counter("orchestrator_operations_error_total", tags)
            self._metrics.record_histogram("orchestrator_operation_duration_ms", response_time_ms, tags)
    
    async def spawn_agent(
        self,
        role: AgentRole,
        agent_id: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Spawn a new agent instance with enhanced error handling.
        
        Args:
            role: The role for the new agent
            agent_id: Optional specific ID, otherwise generated
            capabilities: List of agent capabilities
            config: Agent-specific configuration
            
        Returns:
            The agent ID
            
        Raises:
            SimpleOrchestratorError: If spawning fails
            ResourceLimitError: If agent limit exceeded
        """
        start_time = time.time()
        
        try:
            # Generate ID if not provided
            if agent_id is None:
                agent_id = str(uuid.uuid4())
            
            # Validate agent doesn't exist
            if agent_id in self._agents:
                raise SimpleOrchestratorError(f"Agent {agent_id} already exists")
            
            # Check agent limit
            active_agents = len([a for a in self._agents.values() 
                               if a.status == AgentStatus.active])
            if active_agents >= self.config.max_concurrent_agents:
                raise ResourceLimitError("agents", self.config.max_concurrent_agents, active_agents)
            
            # Create agent instance
            agent = AgentInstance(
                id=agent_id,
                role=role,
                status=AgentStatus.active,
                capabilities=capabilities or [],
                config=config or {}
            )
            
            # Store in memory registry
            self._agents[agent_id] = agent
            
            # Persist to database if enabled
            async with self._database_operation("spawn_agent") as session:
                if session:
                    await self._persist_agent_to_db(session, agent)
            
            # Cache for fast access
            if self._cache and self.config.enable_caching:
                await self._cache.set(f"agent:{agent_id}", agent.to_dict(), ttl=3600)
            
            response_time_ms = (time.time() - start_time) * 1000
            self._record_operation("spawn_agent", True, response_time_ms)
            
            logger.info(
                "Agent spawned successfully",
                agent_id=agent_id,
                role=role.value,
                capabilities_count=len(agent.capabilities),
                duration_ms=response_time_ms
            )
            
            return agent_id
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._record_operation("spawn_agent", False, response_time_ms)
            
            logger.error(
                "Failed to spawn agent",
                agent_id=agent_id,
                role=role.value if role else None,
                error=str(e),
                duration_ms=response_time_ms
            )
            
            if isinstance(e, (SimpleOrchestratorError, ResourceLimitError)):
                raise
            raise SimpleOrchestratorError(f"Failed to spawn agent: {e}") from e
    
    async def shutdown_agent(self, agent_id: str, graceful: bool = True) -> bool:
        """
        Shutdown a specific agent instance with enhanced handling.
        
        Args:
            agent_id: ID of agent to shutdown
            graceful: Whether to wait for current task completion
            
        Returns:
            True if shutdown successful, False if agent not found
            
        Raises:
            SimpleOrchestratorError: If shutdown fails
        """
        start_time = time.time()
        
        try:
            # Check if agent exists
            if agent_id not in self._agents:
                logger.warning("Agent not found for shutdown", agent_id=agent_id)
                return False
            
            agent = self._agents[agent_id]
            
            # Handle graceful shutdown
            if graceful and agent.current_task_id:
                logger.info(
                    "Graceful shutdown - waiting for task completion",
                    agent_id=agent_id,
                    task_id=agent.current_task_id
                )
                # Wait for current task with timeout
                timeout = self.config.task_timeout_minutes * 60
                await asyncio.wait_for(
                    self._wait_for_task_completion(agent.current_task_id),
                    timeout=timeout
                )
            
            # Update agent status
            agent.status = AgentStatus.inactive
            agent.last_activity = datetime.utcnow()
            
            # Update database if enabled
            async with self._database_operation("shutdown_agent") as session:
                if session:
                    await self._update_agent_status_in_db(session, agent_id, AgentStatus.inactive)
            
            # Remove from active registry
            del self._agents[agent_id]
            
            # Remove from cache
            if self._cache and self.config.enable_caching:
                await self._cache.set(f"agent:{agent_id}", None, ttl=1)
            
            response_time_ms = (time.time() - start_time) * 1000
            self._record_operation("shutdown_agent", True, response_time_ms)
            
            logger.info(
                "Agent shutdown successful",
                agent_id=agent_id,
                graceful=graceful,
                duration_ms=response_time_ms
            )
            
            return True
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._record_operation("shutdown_agent", False, response_time_ms)
            
            logger.error(
                "Failed to shutdown agent",
                agent_id=agent_id,
                error=str(e),
                duration_ms=response_time_ms
            )
            
            if isinstance(e, SimpleOrchestratorError):
                raise
            raise SimpleOrchestratorError(f"Failed to shutdown agent: {e}") from e
    
    async def delegate_task(
        self,
        task_description: str,
        task_type: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        required_capabilities: Optional[List[str]] = None,
        preferred_agent_role: Optional[AgentRole] = None,
        assignment_strategy: Optional[TaskAssignmentStrategy] = None,
        estimated_duration: Optional[int] = None
    ) -> str:
        """
        Delegate a task to the most suitable available agent using advanced assignment logic.
        
        Args:
            task_description: Description of the task
            task_type: Type/category of the task
            priority: Task priority level
            required_capabilities: List of required capabilities
            preferred_agent_role: Preferred agent role for the task
            assignment_strategy: Task assignment strategy to use
            estimated_duration: Estimated task duration in minutes
            
        Returns:
            Task ID
            
        Raises:
            TaskDelegationError: If no suitable agent available
        """
        start_time = time.time()
        
        try:
            # Generate task ID
            task_id = str(uuid.uuid4())
            
            # Use default strategy if not specified
            strategy = assignment_strategy or self.config.default_task_assignment_strategy
            
            # Find suitable agent using specified strategy
            suitable_agent = await self._find_suitable_agent(
                strategy=strategy,
                required_capabilities=required_capabilities or [],
                task_type=task_type,
                preferred_role=preferred_agent_role,
                priority=priority
            )
            
            if not suitable_agent:
                raise TaskDelegationError(
                    "No suitable agent available",
                    {
                        "task_type": task_type,
                        "required_capabilities": required_capabilities,
                        "preferred_role": preferred_agent_role.value if preferred_agent_role else None,
                        "strategy": strategy.value,
                        "available_agents": len([a for a in self._agents.values() if a.is_available_for_task()])
                    }
                )
            
            # Calculate suitability score
            suitability_score = suitable_agent.calculate_task_suitability(
                required_capabilities or [], task_type
            )
            
            # Create task assignment
            assignment = TaskAssignment(
                task_id=task_id,
                agent_id=suitable_agent.id,
                status=TaskStatus.ASSIGNED,
                priority=priority,
                estimated_duration=estimated_duration,
                suitability_score=suitability_score
            )
            
            # Update agent with current task
            suitable_agent.current_task_id = task_id
            suitable_agent.last_activity = datetime.utcnow()
            suitable_agent.load_score = min(1.0, suitable_agent.load_score + 0.2)  # Increase load
            
            # Store assignment
            self._task_assignments[task_id] = assignment
            
            # Persist to database if enabled
            async with self._database_operation("delegate_task") as session:
                if session:
                    await self._persist_task_to_db(
                        session, task_id, task_description, task_type, 
                        priority, suitable_agent.id, required_capabilities, estimated_duration
                    )
            
            response_time_ms = (time.time() - start_time) * 1000
            self._record_operation("delegate_task", True, response_time_ms)
            
            logger.info(
                "Task delegated successfully",
                task_id=task_id,
                agent_id=suitable_agent.id,
                agent_role=suitable_agent.role.value,
                task_type=task_type,
                priority=priority.value,
                strategy=strategy.value,
                suitability_score=suitability_score,
                duration_ms=response_time_ms
            )
            
            return task_id
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._record_operation("delegate_task", False, response_time_ms)
            
            logger.error(
                "Failed to delegate task",
                task_description=task_description[:100],
                task_type=task_type,
                error=str(e),
                duration_ms=response_time_ms
            )
            
            if isinstance(e, TaskDelegationError):
                raise
            raise TaskDelegationError(f"Failed to delegate task: {e}", {"task_type": task_type})
    
    async def complete_task(self, task_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """
        Mark a task as completed and update agent availability.
        
        Args:
            task_id: ID of the task to complete
            result: Optional task result data
            
        Returns:
            True if task was completed successfully
            
        Raises:
            SimpleOrchestratorError: If task completion fails
        """
        start_time = time.time()
        
        try:
            # Find task assignment
            if task_id not in self._task_assignments:
                logger.warning("Task not found for completion", task_id=task_id)
                return False
            
            assignment = self._task_assignments[task_id]
            agent_id = assignment.agent_id
            
            # Update assignment
            assignment.status = TaskStatus.COMPLETED
            if assignment.assigned_at:
                assignment.actual_duration = int(
                    (datetime.utcnow() - assignment.assigned_at).total_seconds() / 60
                )
            
            # Update agent
            if agent_id in self._agents:
                agent = self._agents[agent_id]
                agent.current_task_id = None
                agent.last_activity = datetime.utcnow()
                agent.load_score = max(0.0, agent.load_score - 0.2)  # Decrease load
                
                # Update performance metrics
                if assignment.actual_duration:
                    agent.performance_metrics.setdefault("completed_tasks", 0)
                    agent.performance_metrics["completed_tasks"] += 1
                    agent.performance_metrics.setdefault("total_duration", 0)
                    agent.performance_metrics["total_duration"] += assignment.actual_duration
            
            # Update database if enabled
            async with self._database_operation("complete_task") as session:
                if session:
                    await self._update_task_status_in_db(session, task_id, TaskStatus.COMPLETED, result)
            
            response_time_ms = (time.time() - start_time) * 1000
            self._record_operation("complete_task", True, response_time_ms)
            
            logger.info(
                "Task completed successfully",
                task_id=task_id,
                agent_id=agent_id,
                actual_duration=assignment.actual_duration,
                duration_ms=response_time_ms
            )
            
            return True
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._record_operation("complete_task", False, response_time_ms)
            
            logger.error("Failed to complete task", task_id=task_id, error=str(e))
            
            if isinstance(e, SimpleOrchestratorError):
                raise
            raise SimpleOrchestratorError(f"Failed to complete task: {e}") from e
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status for monitoring.
        
        Returns:
            Dictionary with detailed system status information
        """
        start_time = time.time()
        
        try:
            # Collect agent statuses
            agent_statuses = {
                agent_id: agent.to_dict() 
                for agent_id, agent in self._agents.items()
            }
            
            # Count agents by status and role
            status_counts = defaultdict(int)
            role_counts = defaultdict(int)
            healthy_agents = 0
            
            for agent in self._agents.values():
                status_counts[agent.status.value] += 1
                role_counts[agent.role.value] += 1
                if agent.is_healthy():
                    healthy_agents += 1
            
            # Task assignment summary
            assignment_count = len(self._task_assignments)
            task_status_counts = defaultdict(int)
            for assignment in self._task_assignments.values():
                task_status_counts[assignment.status.value] += 1
            
            # Performance metrics
            now = datetime.utcnow()
            uptime_seconds = (now - self._metrics_tracker.last_reset).total_seconds()
            
            # Database health check
            db_healthy = True
            if self.config.enable_database_persistence:
                try:
                    db_healthy = await DatabaseHealthCheck.check_connection()
                except Exception:
                    db_healthy = False
            
            status = {
                "timestamp": now.isoformat(),
                "mode": self.config.mode.value,
                "uptime_seconds": uptime_seconds,
                "agents": {
                    "total": len(self._agents),
                    "healthy": healthy_agents,
                    "by_status": dict(status_counts),
                    "by_role": dict(role_counts),
                    "available": len([a for a in self._agents.values() if a.is_available_for_task()]),
                    "details": agent_statuses
                },
                "tasks": {
                    "active_assignments": assignment_count,
                    "by_status": dict(task_status_counts)
                },
                "performance": {
                    "operations_total": self._metrics_tracker.operation_count,
                    "success_rate": self._metrics_tracker.get_success_rate(),
                    "avg_response_time_ms": self._metrics_tracker.get_average_response_time(),
                    "p95_response_time_ms": self._metrics_tracker.get_p95_response_time(),
                    "errors_total": self._metrics_tracker.error_count
                },
                "health": {
                    "overall": "healthy" if healthy_agents > 0 and db_healthy else "degraded",
                    "database": "healthy" if db_healthy else "unhealthy",
                    "agents": "healthy" if healthy_agents > 0 else "no_agents"
                },
                "configuration": {
                    "max_concurrent_agents": self.config.max_concurrent_agents,
                    "assignment_strategy": self.config.default_task_assignment_strategy.value,
                    "database_persistence": self.config.enable_database_persistence,
                    "performance_monitoring": self.config.enable_performance_monitoring,
                    "caching": self.config.enable_caching
                }
            }
            
            response_time_ms = (time.time() - start_time) * 1000
            
            logger.debug(
                "System status retrieved",
                agent_count=len(self._agents),
                task_count=assignment_count,
                duration_ms=response_time_ms
            )
            
            return status
            
        except Exception as e:
            logger.error("Failed to get system status", error=str(e))
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "health": {"overall": "error"}
            }
    
    # Private helper methods
    
    async def _find_suitable_agent(
        self,
        strategy: TaskAssignmentStrategy,
        required_capabilities: List[str],
        task_type: str,
        preferred_role: Optional[AgentRole] = None,
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> Optional[AgentInstance]:
        """Find the most suitable available agent using specified strategy."""
        available_agents = [
            agent for agent in self._agents.values()
            if agent.is_available_for_task()
        ]
        
        if not available_agents:
            return None
        
        if strategy == TaskAssignmentStrategy.ROUND_ROBIN:
            return self._round_robin_assignment(available_agents)
        
        elif strategy == TaskAssignmentStrategy.AVAILABILITY_BASED:
            return self._availability_based_assignment(available_agents)
        
        elif strategy == TaskAssignmentStrategy.CAPABILITY_MATCH:
            return self._capability_based_assignment(
                available_agents, required_capabilities, task_type, preferred_role
            )
        
        elif strategy == TaskAssignmentStrategy.PERFORMANCE_BASED:
            return self._performance_based_assignment(available_agents, required_capabilities, task_type)
        
        # Fallback to first available
        return available_agents[0]
    
    def _round_robin_assignment(self, available_agents: List[AgentInstance]) -> AgentInstance:
        """Assign task using round-robin strategy."""
        if not available_agents:
            return None
        
        agent = available_agents[self._round_robin_index % len(available_agents)]
        self._round_robin_index += 1
        return agent
    
    def _availability_based_assignment(self, available_agents: List[AgentInstance]) -> AgentInstance:
        """Assign task to agent with lowest load."""
        return min(available_agents, key=lambda a: a.load_score)
    
    def _capability_based_assignment(
        self,
        available_agents: List[AgentInstance],
        required_capabilities: List[str],
        task_type: str,
        preferred_role: Optional[AgentRole]
    ) -> AgentInstance:
        """Assign task based on capability matching."""
        # Score agents by suitability
        scored_agents = [
            (agent, agent.calculate_task_suitability(required_capabilities, task_type))
            for agent in available_agents
        ]
        
        # Prefer agents with matching role
        if preferred_role:
            role_matches = [(a, s) for a, s in scored_agents if a.role == preferred_role]
            if role_matches:
                scored_agents = role_matches
        
        # Return agent with highest score
        return max(scored_agents, key=lambda x: x[1])[0]
    
    def _performance_based_assignment(
        self,
        available_agents: List[AgentInstance],
        required_capabilities: List[str],
        task_type: str
    ) -> AgentInstance:
        """Assign task based on agent performance history."""
        # Score agents by performance and suitability
        scored_agents = []
        
        for agent in available_agents:
            suitability_score = agent.calculate_task_suitability(required_capabilities, task_type)
            
            # Performance score based on completion rate and average duration
            performance_score = 0.5  # Base score
            if agent.performance_metrics.get("completed_tasks", 0) > 0:
                # Prefer agents with good completion history
                performance_score += 0.3
                
                # Consider average task duration
                avg_duration = (
                    agent.performance_metrics.get("total_duration", 0) /
                    agent.performance_metrics.get("completed_tasks", 1)
                )
                if avg_duration < 60:  # Less than 1 hour average
                    performance_score += 0.2
            
            total_score = (suitability_score * 0.7) + (performance_score * 0.3)
            scored_agents.append((agent, total_score))
        
        return max(scored_agents, key=lambda x: x[1])[0]
    
    async def _wait_for_task_completion(self, task_id: str) -> None:
        """Wait for a task to complete with polling."""
        while task_id in self._task_assignments:
            assignment = self._task_assignments[task_id]
            if assignment.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                break
            await asyncio.sleep(1)
    
    async def _persist_agent_to_db(self, session: AsyncSession, agent: AgentInstance) -> None:
        """Persist agent to database."""
        try:
            db_agent = Agent(
                id=uuid.UUID(agent.id),
                name=f"{agent.role.value}_agent",
                role=agent.role.value,
                type=AgentType.CLAUDE,
                status=agent.status,
                capabilities=agent.capabilities,
                config=agent.config,
                created_at=agent.created_at,
                last_active=agent.last_activity
            )
            session.add(db_agent)
            await session.flush()
            
        except Exception as e:
            logger.warning("Failed to persist agent to database", 
                         agent_id=agent.id, error=str(e))
            raise
    
    async def _update_agent_status_in_db(
        self, session: AsyncSession, agent_id: str, status: AgentStatus
    ) -> None:
        """Update agent status in database."""
        try:
            stmt = (
                update(Agent)
                .where(Agent.id == uuid.UUID(agent_id))
                .values(status=status, updated_at=datetime.utcnow())
            )
            await session.execute(stmt)
            await session.flush()
            
        except Exception as e:
            logger.warning("Failed to update agent status in database",
                         agent_id=agent_id, error=str(e))
            raise
    
    async def _persist_task_to_db(
        self,
        session: AsyncSession,
        task_id: str,
        description: str,
        task_type: str,
        priority: TaskPriority,
        agent_id: str,
        required_capabilities: List[str],
        estimated_duration: Optional[int]
    ) -> None:
        """Persist task to database."""
        try:
            # Map task_type string to TaskType enum
            task_type_enum = None
            for tt in TaskType:
                if tt.value == task_type:
                    task_type_enum = tt
                    break
            
            db_task = Task(
                id=uuid.UUID(task_id),
                title=f"Task: {task_type}",
                description=description,
                task_type=task_type_enum,
                priority=priority,
                status=TaskStatus.ASSIGNED,
                assigned_agent_id=uuid.UUID(agent_id),
                required_capabilities=required_capabilities,
                estimated_effort=estimated_duration,
                context={"task_type": task_type},
                created_at=datetime.utcnow(),
                assigned_at=datetime.utcnow()
            )
            session.add(db_task)
            await session.flush()
            
        except Exception as e:
            logger.warning("Failed to persist task to database",
                         task_id=task_id, error=str(e))
            raise
    
    async def _update_task_status_in_db(
        self,
        session: AsyncSession,
        task_id: str,
        status: TaskStatus,
        result: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update task status in database."""
        try:
            update_values = {
                "status": status,
                "updated_at": datetime.utcnow()
            }
            
            if status == TaskStatus.COMPLETED:
                update_values["completed_at"] = datetime.utcnow()
                if result:
                    update_values["result"] = result
            
            stmt = (
                update(Task)
                .where(Task.id == uuid.UUID(task_id))
                .values(**update_values)
            )
            await session.execute(stmt)
            await session.flush()
            
        except Exception as e:
            logger.warning("Failed to update task status in database",
                         task_id=task_id, error=str(e))
            raise
    
    async def _load_agents_from_database(self) -> None:
        """Load existing agents from database on startup."""
        try:
            async with self._database_operation("load_agents") as session:
                if not session:
                    return
                
                stmt = select(Agent).where(Agent.status == AgentStatus.active)
                result = await session.execute(stmt)
                db_agents = result.scalars().all()
                
                for db_agent in db_agents:
                    # Convert database agent to AgentInstance
                    try:
                        role = AgentRole(db_agent.role)
                        agent = AgentInstance(
                            id=str(db_agent.id),
                            role=role,
                            status=db_agent.status,
                            created_at=db_agent.created_at or datetime.utcnow(),
                            last_activity=db_agent.last_active or datetime.utcnow(),
                            capabilities=db_agent.capabilities or [],
                            config=db_agent.config or {}
                        )
                        self._agents[agent.id] = agent
                        
                    except ValueError as e:
                        logger.warning("Invalid agent role in database", 
                                     agent_id=str(db_agent.id), role=db_agent.role, error=str(e))
                
                logger.info(f"Loaded {len(self._agents)} agents from database")
                
        except Exception as e:
            logger.warning("Failed to load agents from database", error=str(e))
    
    async def _validate_configuration(self) -> None:
        """Validate orchestrator configuration."""
        if self.config.max_concurrent_agents <= 0:
            raise ConfigurationError("max_concurrent_agents", "Must be greater than 0")
        
        if self.config.heartbeat_interval_seconds <= 0:
            raise ConfigurationError("heartbeat_interval_seconds", "Must be greater than 0")
        
        if self.config.task_timeout_minutes <= 0:
            raise ConfigurationError("task_timeout_minutes", "Must be greater than 0")
        
        # Validate database connection if persistence is enabled
        if self.config.enable_database_persistence:
            try:
                healthy = await DatabaseHealthCheck.check_connection()
                if not healthy:
                    logger.warning("Database health check failed, disabling persistence")
                    self.config.enable_database_persistence = False
            except Exception as e:
                logger.warning("Database validation failed, disabling persistence", error=str(e))
                self.config.enable_database_persistence = False
    
    async def _heartbeat_monitor(self) -> None:
        """Background task to monitor agent heartbeats."""
        logger.info("Starting heartbeat monitor")
        
        while not self._shutdown_event.is_set():
            try:
                now = datetime.utcnow()
                unhealthy_agents = []
                
                for agent_id, agent in self._agents.items():
                    if not agent.is_healthy():
                        unhealthy_agents.append(agent_id)
                        logger.warning("Unhealthy agent detected", agent_id=agent_id)
                
                # Remove unhealthy agents
                for agent_id in unhealthy_agents:
                    try:
                        await self.shutdown_agent(agent_id, graceful=False)
                    except Exception as e:
                        logger.error("Failed to shutdown unhealthy agent", 
                                   agent_id=agent_id, error=str(e))
                
                await asyncio.sleep(self.config.heartbeat_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in heartbeat monitor", error=str(e))
                await asyncio.sleep(5)
        
        logger.info("Heartbeat monitor stopped")
    
    async def _performance_monitor(self) -> None:
        """Background task to monitor and log performance metrics."""
        logger.info("Starting performance monitor")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Report every minute
                
                metrics = {
                    "operations_per_minute": self._metrics_tracker.operation_count,
                    "success_rate": self._metrics_tracker.get_success_rate(),
                    "avg_response_time_ms": self._metrics_tracker.get_average_response_time(),
                    "active_agents": len([a for a in self._agents.values() if a.status == AgentStatus.active]),
                    "active_tasks": len(self._task_assignments)
                }
                
                logger.info("Performance metrics", **metrics)
                
                # Reset counters for next period
                self._metrics_tracker.operation_count = 0
                self._metrics_tracker.success_count = 0
                self._metrics_tracker.error_count = 0
                self._metrics_tracker.last_reset = datetime.utcnow()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in performance monitor", error=str(e))
        
        logger.info("Performance monitor stopped")


# Factory functions and global instance management
def create_enhanced_orchestrator(
    config: Optional[OrchestratorConfig] = None,
    db_session_factory: Optional[DatabaseDependency] = None,
    cache: Optional[CacheDependency] = None,
    metrics: Optional[MetricsDependency] = None,
    anthropic_client: Optional[AsyncAnthropic] = None
) -> EnhancedSimpleOrchestrator:
    """
    Factory function to create EnhancedSimpleOrchestrator with proper dependencies.
    
    This makes it easy to inject dependencies for testing and different environments.
    """
    return EnhancedSimpleOrchestrator(
        config=config,
        db_session_factory=db_session_factory,
        cache=cache,
        metrics=metrics,
        anthropic_client=anthropic_client
    )


# Global instance for API usage (can be overridden in tests)
_global_enhanced_orchestrator: Optional[EnhancedSimpleOrchestrator] = None


async def get_enhanced_orchestrator() -> EnhancedSimpleOrchestrator:
    """Get the global enhanced orchestrator instance."""
    global _global_enhanced_orchestrator
    if _global_enhanced_orchestrator is None:
        _global_enhanced_orchestrator = create_enhanced_orchestrator()
        await _global_enhanced_orchestrator.start()
    return _global_enhanced_orchestrator


def set_enhanced_orchestrator(orchestrator: EnhancedSimpleOrchestrator) -> None:
    """Set the global enhanced orchestrator instance (useful for testing)."""
    global _global_enhanced_orchestrator
    _global_enhanced_orchestrator = orchestrator


# Backward compatibility - make this the new default SimpleOrchestrator
SimpleOrchestratorEnhanced = EnhancedSimpleOrchestrator
create_simple_orchestrator_enhanced = create_enhanced_orchestrator
get_simple_orchestrator_enhanced = get_enhanced_orchestrator
set_simple_orchestrator_enhanced = set_enhanced_orchestrator