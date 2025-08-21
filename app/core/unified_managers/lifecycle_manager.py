#!/usr/bin/env python3
"""
LifecycleManager - Agent and Resource Lifecycle Consolidation
Phase 2.1 Implementation of Technical Debt Remediation Plan

This manager consolidates all agent lifecycle, resource management, and state
transitions into a unified, high-performance system built on the BaseManager framework.

TARGET CONSOLIDATION: 12+ manager classes → 1 unified LifecycleManager
- Agent lifecycle management (spawn, monitor, terminate)
- Resource allocation and cleanup
- State transitions and recovery
- Health monitoring and auto-healing
- Process management and oversight
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from contextlib import asynccontextmanager

import structlog

# Import BaseManager framework
from .base_manager import (
    BaseManager, ManagerConfig, ManagerDomain, ManagerStatus, ManagerMetrics,
    PluginInterface, PluginType
)

# Import shared patterns from Phase 1
from ...common.utilities.shared_patterns import (
    standard_logging_setup, standard_error_handling
)

logger = structlog.get_logger(__name__)


class LifecycleState(str, Enum):
    """Standardized lifecycle states for agents and resources."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    ERROR = "error"


class ResourceType(str, Enum):
    """Types of resources managed by LifecycleManager."""
    AGENT = "agent"
    CONNECTION = "connection"
    MEMORY_POOL = "memory_pool"
    THREAD_POOL = "thread_pool"
    FILE_HANDLE = "file_handle"
    DATABASE_SESSION = "database_session"
    CACHE = "cache"
    TEMPORARY_FILE = "temporary_file"
    NETWORK_SOCKET = "network_socket"


@dataclass
class LifecycleEntity:
    """Represents any managed entity (agent, resource, etc.)."""
    id: str
    name: str
    type: ResourceType
    state: LifecycleState
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    health_score: float = 100.0
    last_activity: Optional[datetime] = None
    restart_count: int = 0
    max_restarts: int = 3
    timeout_seconds: int = 300
    cleanup_callbacks: List[Callable] = field(default_factory=list)
    
    def is_healthy(self) -> bool:
        """Check if entity is in a healthy state."""
        return (
            self.state in [LifecycleState.ACTIVE, LifecycleState.IDLE] and
            self.health_score >= 70.0 and
            self.restart_count < self.max_restarts
        )
    
    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
        self.updated_at = self.last_activity


@dataclass
class LifecycleMetrics:
    """Lifecycle-specific metrics."""
    total_entities: int = 0
    active_entities: int = 0
    entities_by_type: Dict[ResourceType, int] = field(default_factory=dict)
    entities_by_state: Dict[LifecycleState, int] = field(default_factory=dict)
    total_spawns: int = 0
    total_terminations: int = 0
    total_restarts: int = 0
    failed_spawns: int = 0
    failed_terminations: int = 0
    avg_entity_lifetime_seconds: float = 0.0
    avg_spawn_time_ms: float = 0.0
    avg_termination_time_ms: float = 0.0


class LifecyclePlugin(PluginInterface):
    """Base class for lifecycle plugins."""
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.WORKFLOW
    
    async def pre_spawn_hook(self, entity_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called before entity spawn."""
        return {}
    
    async def post_spawn_hook(self, entity: LifecycleEntity) -> None:
        """Hook called after entity spawn."""
        pass
    
    async def pre_terminate_hook(self, entity: LifecycleEntity) -> None:
        """Hook called before entity termination."""
        pass
    
    async def post_terminate_hook(self, entity_id: str, final_state: LifecycleState) -> None:
        """Hook called after entity termination."""
        pass


class LifecycleManager(BaseManager):
    """
    Unified manager for all agent and resource lifecycle operations.
    
    CONSOLIDATION TARGET: Replaces 12+ specialized lifecycle/resource managers:
    - AgentLifecycleManager
    - ResourceManager  
    - ProcessManager
    - StateManager
    - HealthMonitor
    - CleanupManager
    - RestartManager
    - DependencyManager
    - TimeoutManager
    - RecoveryManager
    - MemoryManager
    - ConnectionPoolManager
    
    Built on BaseManager framework with Phase 2 enhancements.
    """
    
    def __init__(self, config: Optional[ManagerConfig] = None):
        # Create default config if none provided
        if config is None:
            config = ManagerConfig(
                name="LifecycleManager",
                domain=ManagerDomain.LIFECYCLE,
                max_concurrent_operations=200,
                health_check_interval=15,
                circuit_breaker_enabled=True
            )
        
        super().__init__(config)
        
        # Lifecycle-specific state
        self.entities: Dict[str, LifecycleEntity] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.lifecycle_metrics = LifecycleMetrics()
        
        # Monitoring and maintenance
        self._monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._entities_lock = threading.RLock()
        self._spawn_semaphore = asyncio.Semaphore(self.config.max_concurrent_operations)
        
        self.logger = standard_logging_setup(
            name="LifecycleManager",
            level="INFO"
        )
    
    # BaseManager Implementation
    
    async def _setup(self) -> None:
        """Initialize lifecycle management systems."""
        self.logger.info("Setting up LifecycleManager")
        
        # Start background tasks
        self._monitor_task = asyncio.create_task(self._entity_monitor_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        
        self.logger.info("LifecycleManager setup completed")
    
    async def _cleanup(self) -> None:
        """Clean up lifecycle management systems."""
        self.logger.info("Cleaning up LifecycleManager")
        
        # Cancel background tasks
        for task in [self._monitor_task, self._cleanup_task, self._health_monitor_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Terminate all entities
        await self._terminate_all_entities()
        
        self.logger.info("LifecycleManager cleanup completed")
    
    async def _health_check_internal(self) -> Dict[str, Any]:
        """Lifecycle-specific health check."""
        with self._entities_lock:
            total_entities = len(self.entities)
            healthy_entities = sum(1 for e in self.entities.values() if e.is_healthy())
            degraded_entities = sum(
                1 for e in self.entities.values() 
                if e.state == LifecycleState.DEGRADED
            )
            error_entities = sum(
                1 for e in self.entities.values() 
                if e.state == LifecycleState.ERROR
            )
        
        return {
            "total_entities": total_entities,
            "healthy_entities": healthy_entities,
            "degraded_entities": degraded_entities,
            "error_entities": error_entities,
            "health_percentage": (healthy_entities / max(total_entities, 1)) * 100,
            "lifecycle_metrics": {
                "total_spawns": self.lifecycle_metrics.total_spawns,
                "total_terminations": self.lifecycle_metrics.total_terminations,
                "failed_spawns": self.lifecycle_metrics.failed_spawns,
                "avg_spawn_time_ms": self.lifecycle_metrics.avg_spawn_time_ms
            }
        }
    
    # Core Lifecycle Operations
    
    async def spawn_entity(
        self,
        name: str,
        entity_type: ResourceType,
        config: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Set[str]] = None
    ) -> str:
        """
        Spawn a new managed entity with full lifecycle support.
        
        CONSOLIDATES: AgentSpawner, ResourceAllocator, ProcessCreator patterns
        """
        async with self._spawn_semaphore:
            start_time = time.time()
            entity_id = str(uuid.uuid4())
            
            try:
                async with self.execute_with_monitoring("spawn_entity"):
                    # Pre-spawn hooks
                    hook_data = {}
                    for plugin in self.plugins.values():
                        if isinstance(plugin, LifecyclePlugin):
                            plugin_data = await plugin.pre_spawn_hook(config or {})
                            hook_data.update(plugin_data)
                    
                    # Create entity
                    entity = LifecycleEntity(
                        id=entity_id,
                        name=name,
                        type=entity_type,
                        state=LifecycleState.INITIALIZING,
                        dependencies=dependencies or set(),
                        metadata=config or {}
                    )
                    
                    # Check dependencies
                    if dependencies:
                        await self._validate_dependencies(dependencies)
                    
                    # Register entity
                    with self._entities_lock:
                        self.entities[entity_id] = entity
                        self._update_dependency_graph(entity_id, dependencies or set())
                    
                    # Entity-specific initialization
                    await self._initialize_entity(entity)
                    
                    # Update state to active
                    entity.state = LifecycleState.ACTIVE
                    entity.update_activity()
                    
                    # Post-spawn hooks
                    for plugin in self.plugins.values():
                        if isinstance(plugin, LifecyclePlugin):
                            await plugin.post_spawn_hook(entity)
                    
                    # Update metrics
                    spawn_time_ms = (time.time() - start_time) * 1000
                    self.lifecycle_metrics.total_spawns += 1
                    self._update_spawn_time_metrics(spawn_time_ms)
                    self._update_entity_metrics()
                    
                    self.logger.info(
                        f"Entity spawned successfully",
                        entity_id=entity_id,
                        name=name,
                        type=entity_type.value,
                        spawn_time_ms=spawn_time_ms
                    )
                    
                    return entity_id
                    
            except Exception as e:
                self.lifecycle_metrics.failed_spawns += 1
                self.logger.error(f"Failed to spawn entity: {e}", entity_id=entity_id)
                
                # Cleanup failed entity
                with self._entities_lock:
                    if entity_id in self.entities:
                        del self.entities[entity_id]
                
                raise
    
    async def terminate_entity(self, entity_id: str, force: bool = False) -> bool:
        """
        Terminate a managed entity with proper cleanup.
        
        CONSOLIDATES: AgentTerminator, ResourceCleaner, ProcessKiller patterns
        """
        start_time = time.time()
        
        async with self.execute_with_monitoring("terminate_entity"):
            with self._entities_lock:
                entity = self.entities.get(entity_id)
                if not entity:
                    self.logger.warning(f"Entity not found for termination", entity_id=entity_id)
                    return False
            
            try:
                # Pre-terminate hooks
                for plugin in self.plugins.values():
                    if isinstance(plugin, LifecyclePlugin):
                        await plugin.pre_terminate_hook(entity)
                
                # Update state
                entity.state = LifecycleState.TERMINATING
                entity.updated_at = datetime.utcnow()
                
                # Check dependencies (don't terminate if other entities depend on this)
                if not force and self._has_dependents(entity_id):
                    raise ValueError(f"Cannot terminate entity {entity_id} - has dependents")
                
                # Entity-specific cleanup
                await self._cleanup_entity(entity)
                
                # Remove from tracking
                with self._entities_lock:
                    if entity_id in self.entities:
                        del self.entities[entity_id]
                    self._remove_from_dependency_graph(entity_id)
                
                # Post-terminate hooks
                final_state = LifecycleState.TERMINATED
                for plugin in self.plugins.values():
                    if isinstance(plugin, LifecyclePlugin):
                        await plugin.post_terminate_hook(entity_id, final_state)
                
                # Update metrics
                termination_time_ms = (time.time() - start_time) * 1000
                self.lifecycle_metrics.total_terminations += 1
                self._update_termination_time_metrics(termination_time_ms)
                self._update_entity_metrics()
                
                self.logger.info(
                    f"Entity terminated successfully",
                    entity_id=entity_id,
                    name=entity.name,
                    type=entity.type.value,
                    termination_time_ms=termination_time_ms
                )
                
                return True
                
            except Exception as e:
                self.lifecycle_metrics.failed_terminations += 1
                self.logger.error(f"Failed to terminate entity: {e}", entity_id=entity_id)
                
                # Update state to error
                entity.state = LifecycleState.ERROR
                entity.updated_at = datetime.utcnow()
                
                return False
    
    async def restart_entity(self, entity_id: str) -> bool:
        """
        Restart an entity with state preservation.
        
        CONSOLIDATES: AgentRestarter, ResourceRecycler patterns
        """
        async with self.execute_with_monitoring("restart_entity"):
            with self._entities_lock:
                entity = self.entities.get(entity_id)
                if not entity:
                    return False
                
                if entity.restart_count >= entity.max_restarts:
                    self.logger.error(
                        f"Entity exceeded maximum restarts",
                        entity_id=entity_id,
                        restart_count=entity.restart_count,
                        max_restarts=entity.max_restarts
                    )
                    return False
            
            try:
                # Preserve configuration
                config = entity.metadata.copy()
                dependencies = entity.dependencies.copy()
                name = entity.name
                entity_type = entity.type
                
                # Terminate current instance (force=True for restart)
                await self.terminate_entity(entity_id, force=True)
                
                # Spawn new instance with same ID and incremented restart count
                new_entity = LifecycleEntity(
                    id=entity_id,
                    name=name,
                    type=entity_type,
                    state=LifecycleState.INITIALIZING,
                    dependencies=dependencies,
                    metadata=config,
                    restart_count=entity.restart_count + 1
                )
                
                # Register new instance
                with self._entities_lock:
                    self.entities[entity_id] = new_entity
                    self._update_dependency_graph(entity_id, dependencies)
                
                # Initialize and activate
                await self._initialize_entity(new_entity)
                new_entity.state = LifecycleState.ACTIVE
                new_entity.update_activity()
                
                self.lifecycle_metrics.total_restarts += 1
                
                self.logger.info(
                    f"Entity restarted successfully",
                    entity_id=entity_id,
                    restart_count=new_entity.restart_count
                )
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to restart entity: {e}", entity_id=entity_id)
                return False
    
    # Entity Management
    
    def get_entity(self, entity_id: str) -> Optional[LifecycleEntity]:
        """Get entity by ID."""
        with self._entities_lock:
            return self.entities.get(entity_id)
    
    def list_entities(
        self,
        entity_type: Optional[ResourceType] = None,
        state: Optional[LifecycleState] = None
    ) -> List[LifecycleEntity]:
        """List entities with optional filtering."""
        with self._entities_lock:
            entities = list(self.entities.values())
        
        if entity_type:
            entities = [e for e in entities if e.type == entity_type]
        
        if state:
            entities = [e for e in entities if e.state == state]
        
        return entities
    
    def get_entity_count(self, entity_type: Optional[ResourceType] = None) -> int:
        """Get count of entities, optionally filtered by type."""
        with self._entities_lock:
            if entity_type:
                return sum(1 for e in self.entities.values() if e.type == entity_type)
            return len(self.entities)
    
    async def update_entity_state(self, entity_id: str, new_state: LifecycleState) -> bool:
        """Update entity state with validation."""
        async with self.execute_with_monitoring("update_entity_state"):
            with self._entities_lock:
                entity = self.entities.get(entity_id)
                if not entity:
                    return False
                
                old_state = entity.state
                entity.state = new_state
                entity.updated_at = datetime.utcnow()
                
                if new_state in [LifecycleState.ACTIVE, LifecycleState.BUSY]:
                    entity.update_activity()
                
                self.logger.debug(
                    f"Entity state updated",
                    entity_id=entity_id,
                    old_state=old_state.value,
                    new_state=new_state.value
                )
                
                return True
    
    # Private Implementation Methods
    
    async def _initialize_entity(self, entity: LifecycleEntity) -> None:
        """Entity-specific initialization based on type."""
        try:
            # Call entity-specific initialization
            if entity.type == ResourceType.AGENT:
                await self._initialize_agent(entity)
            elif entity.type == ResourceType.CONNECTION:
                await self._initialize_connection(entity)
            elif entity.type == ResourceType.THREAD_POOL:
                await self._initialize_thread_pool(entity)
            # Add other types as needed
            
            self.logger.debug(f"Entity initialized", entity_id=entity.id, type=entity.type.value)
            
        except Exception as e:
            self.logger.error(f"Entity initialization failed: {e}", entity_id=entity.id)
            raise
    
    async def _cleanup_entity(self, entity: LifecycleEntity) -> None:
        """Entity-specific cleanup based on type."""
        try:
            # Run cleanup callbacks first
            for callback in entity.cleanup_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    self.logger.warning(f"Cleanup callback failed: {e}")
            
            # Entity-specific cleanup
            if entity.type == ResourceType.AGENT:
                await self._cleanup_agent(entity)
            elif entity.type == ResourceType.CONNECTION:
                await self._cleanup_connection(entity)
            elif entity.type == ResourceType.THREAD_POOL:
                await self._cleanup_thread_pool(entity)
            # Add other types as needed
            
            self.logger.debug(f"Entity cleanup completed", entity_id=entity.id)
            
        except Exception as e:
            self.logger.error(f"Entity cleanup failed: {e}", entity_id=entity.id)
            # Don't re-raise - we want cleanup to continue
    
    # Entity Type Specific Methods (extensible)
    
    async def _initialize_agent(self, entity: LifecycleEntity) -> None:
        """Initialize agent-specific resources."""
        # Agent-specific initialization logic
        pass
    
    async def _cleanup_agent(self, entity: LifecycleEntity) -> None:
        """Cleanup agent-specific resources."""
        # Agent-specific cleanup logic
        pass
    
    async def _initialize_connection(self, entity: LifecycleEntity) -> None:
        """Initialize connection-specific resources."""
        # Connection-specific initialization logic
        pass
    
    async def _cleanup_connection(self, entity: LifecycleEntity) -> None:
        """Cleanup connection-specific resources."""
        # Connection-specific cleanup logic
        pass
    
    async def _initialize_thread_pool(self, entity: LifecycleEntity) -> None:
        """Initialize thread pool resources."""
        # Thread pool initialization logic
        pass
    
    async def _cleanup_thread_pool(self, entity: LifecycleEntity) -> None:
        """Cleanup thread pool resources."""
        # Thread pool cleanup logic
        pass
    
    # Background Tasks
    
    async def _entity_monitor_loop(self) -> None:
        """Monitor entities for health and lifecycle transitions."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.config.health_check_interval)
                if self._shutdown_event.is_set():
                    break
                
                current_time = datetime.utcnow()
                entities_to_check = []
                
                with self._entities_lock:
                    entities_to_check = list(self.entities.values())
                
                for entity in entities_to_check:
                    # Check for timeouts
                    if entity.last_activity:
                        time_since_activity = (current_time - entity.last_activity).total_seconds()
                        if time_since_activity > entity.timeout_seconds:
                            self.logger.warning(
                                f"Entity timeout detected",
                                entity_id=entity.id,
                                time_since_activity=time_since_activity,
                                timeout_seconds=entity.timeout_seconds
                            )
                            
                            # Attempt restart if possible
                            if entity.restart_count < entity.max_restarts:
                                asyncio.create_task(self.restart_entity(entity.id))
                            else:
                                await self.update_entity_state(entity.id, LifecycleState.ERROR)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Entity monitor loop error: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of terminated entities and resources."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Run every minute
                if self._shutdown_event.is_set():
                    break
                
                # Cleanup terminated entities that might still be in memory
                current_time = datetime.utcnow()
                entities_to_remove = []
                
                with self._entities_lock:
                    for entity_id, entity in self.entities.items():
                        if (entity.state == LifecycleState.TERMINATED and
                            (current_time - entity.updated_at).total_seconds() > 300):  # 5 minutes
                            entities_to_remove.append(entity_id)
                    
                    for entity_id in entities_to_remove:
                        del self.entities[entity_id]
                        self._remove_from_dependency_graph(entity_id)
                
                if entities_to_remove:
                    self.logger.debug(f"Cleaned up {len(entities_to_remove)} terminated entities")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    async def _health_monitor_loop(self) -> None:
        """Monitor entity health and trigger recovery actions."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                if self._shutdown_event.is_set():
                    break
                
                entities_to_check = []
                with self._entities_lock:
                    entities_to_check = [
                        e for e in self.entities.values()
                        if e.state in [LifecycleState.ACTIVE, LifecycleState.DEGRADED]
                    ]
                
                for entity in entities_to_check:
                    # Check health score
                    if entity.health_score < 50.0:
                        if entity.state != LifecycleState.DEGRADED:
                            await self.update_entity_state(entity.id, LifecycleState.DEGRADED)
                            self.logger.warning(
                                f"Entity health degraded",
                                entity_id=entity.id,
                                health_score=entity.health_score
                            )
                    elif entity.health_score >= 70.0 and entity.state == LifecycleState.DEGRADED:
                        await self.update_entity_state(entity.id, LifecycleState.ACTIVE)
                        self.logger.info(
                            f"Entity health recovered",
                            entity_id=entity.id,
                            health_score=entity.health_score
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitor loop error: {e}")
    
    # Dependency Management
    
    async def _validate_dependencies(self, dependencies: Set[str]) -> None:
        """Validate that all dependencies exist and are healthy."""
        with self._entities_lock:
            for dep_id in dependencies:
                if dep_id not in self.entities:
                    raise ValueError(f"Dependency {dep_id} not found")
                
                dep_entity = self.entities[dep_id]
                if not dep_entity.is_healthy():
                    raise ValueError(f"Dependency {dep_id} is not healthy")
    
    def _update_dependency_graph(self, entity_id: str, dependencies: Set[str]) -> None:
        """Update the dependency graph for an entity."""
        self.dependency_graph[entity_id] = dependencies.copy()
    
    def _remove_from_dependency_graph(self, entity_id: str) -> None:
        """Remove entity from dependency graph."""
        if entity_id in self.dependency_graph:
            del self.dependency_graph[entity_id]
        
        # Remove this entity from other entities' dependencies
        for deps in self.dependency_graph.values():
            deps.discard(entity_id)
    
    def _has_dependents(self, entity_id: str) -> bool:
        """Check if any other entities depend on this entity."""
        for deps in self.dependency_graph.values():
            if entity_id in deps:
                return True
        return False
    
    # Metrics Helpers
    
    def _update_entity_metrics(self) -> None:
        """Update entity count metrics."""
        with self._entities_lock:
            self.lifecycle_metrics.total_entities = len(self.entities)
            self.lifecycle_metrics.active_entities = sum(
                1 for e in self.entities.values() if e.state == LifecycleState.ACTIVE
            )
            
            # Update by type
            self.lifecycle_metrics.entities_by_type.clear()
            for entity in self.entities.values():
                current = self.lifecycle_metrics.entities_by_type.get(entity.type, 0)
                self.lifecycle_metrics.entities_by_type[entity.type] = current + 1
            
            # Update by state
            self.lifecycle_metrics.entities_by_state.clear()
            for entity in self.entities.values():
                current = self.lifecycle_metrics.entities_by_state.get(entity.state, 0)
                self.lifecycle_metrics.entities_by_state[entity.state] = current + 1
    
    def _update_spawn_time_metrics(self, spawn_time_ms: float) -> None:
        """Update spawn time metrics."""
        current_avg = self.lifecycle_metrics.avg_spawn_time_ms
        total_spawns = self.lifecycle_metrics.total_spawns
        
        if total_spawns == 1:
            self.lifecycle_metrics.avg_spawn_time_ms = spawn_time_ms
        else:
            # Running average
            self.lifecycle_metrics.avg_spawn_time_ms = (
                (current_avg * (total_spawns - 1) + spawn_time_ms) / total_spawns
            )
    
    def _update_termination_time_metrics(self, termination_time_ms: float) -> None:
        """Update termination time metrics."""
        current_avg = self.lifecycle_metrics.avg_termination_time_ms
        total_terminations = self.lifecycle_metrics.total_terminations
        
        if total_terminations == 1:
            self.lifecycle_metrics.avg_termination_time_ms = termination_time_ms
        else:
            # Running average
            self.lifecycle_metrics.avg_termination_time_ms = (
                (current_avg * (total_terminations - 1) + termination_time_ms) / total_terminations
            )
    
    async def _terminate_all_entities(self) -> None:
        """Terminate all entities during shutdown."""
        entity_ids = []
        with self._entities_lock:
            entity_ids = list(self.entities.keys())
        
        self.logger.info(f"Terminating {len(entity_ids)} entities during shutdown")
        
        # Terminate in parallel with limited concurrency
        termination_tasks = []
        semaphore = asyncio.Semaphore(10)  # Limit concurrent terminations
        
        async def terminate_with_semaphore(entity_id: str):
            async with semaphore:
                try:
                    await self.terminate_entity(entity_id, force=True)
                except Exception as e:
                    self.logger.warning(f"Failed to terminate entity during shutdown: {e}")
        
        for entity_id in entity_ids:
            task = asyncio.create_task(terminate_with_semaphore(entity_id))
            termination_tasks.append(task)
        
        # Wait for all terminations to complete (or timeout after 30 seconds)
        try:
            await asyncio.wait_for(
                asyncio.gather(*termination_tasks, return_exceptions=True),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            self.logger.warning("Entity termination timeout during shutdown")
    
    # Public API Extensions
    
    def get_lifecycle_metrics(self) -> LifecycleMetrics:
        """Get current lifecycle metrics."""
        self._update_entity_metrics()
        return self.lifecycle_metrics
    
    async def add_cleanup_callback(self, entity_id: str, callback: Callable) -> bool:
        """Add a cleanup callback for an entity."""
        with self._entities_lock:
            entity = self.entities.get(entity_id)
            if entity:
                entity.cleanup_callbacks.append(callback)
                return True
            return False
    
    async def update_entity_health_score(self, entity_id: str, health_score: float) -> bool:
        """Update entity health score."""
        with self._entities_lock:
            entity = self.entities.get(entity_id)
            if entity:
                entity.health_score = max(0.0, min(100.0, health_score))
                entity.updated_at = datetime.utcnow()
                return True
            return False


# Plugin Examples

class PerformanceMonitoringPlugin(LifecyclePlugin):
    """Plugin for monitoring entity performance."""
    
    @property
    def name(self) -> str:
        return "PerformanceMonitoring"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def initialize(self, manager: BaseManager) -> None:
        self.performance_data: Dict[str, List[float]] = {}
    
    async def cleanup(self) -> None:
        self.performance_data.clear()
    
    async def post_spawn_hook(self, entity: LifecycleEntity) -> None:
        self.performance_data[entity.id] = []
    
    async def post_terminate_hook(self, entity_id: str, final_state: LifecycleState) -> None:
        if entity_id in self.performance_data:
            del self.performance_data[entity_id]


class ResourceQuotaPlugin(LifecyclePlugin):
    """Plugin for enforcing resource quotas."""
    
    def __init__(self, max_entities_per_type: Dict[ResourceType, int]):
        self.max_entities_per_type = max_entities_per_type
    
    @property
    def name(self) -> str:
        return "ResourceQuota"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def initialize(self, manager: BaseManager) -> None:
        self.manager = manager
    
    async def cleanup(self) -> None:
        pass
    
    async def pre_spawn_hook(self, entity_spec: Dict[str, Any]) -> Dict[str, Any]:
        entity_type = ResourceType(entity_spec.get('type'))
        max_allowed = self.max_entities_per_type.get(entity_type)
        
        if max_allowed is not None:
            current_count = self.manager.get_entity_count(entity_type)
            if current_count >= max_allowed:
                raise ValueError(f"Resource quota exceeded for {entity_type.value}")
        
        return {}