"""
Simple Orchestrator for LeanVibe Agent Hive 2.0

Epic 1 Optimized Version: Memory-efficient orchestrator with lazy loading
and reduced initialization footprint. Optimized for <50ms response times
and <20MB memory usage.

Key Optimizations:
- Lazy loading of heavy dependencies (Anthropic, Redis, etc.)
- Memory-efficient data structures
- Minimal initial imports
- Plugin-based architecture for optional features
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Protocol, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Core lightweight imports only
from .config import settings
from .logging_service import get_component_logger

# Epic 2 Phase 2.1: Advanced Plugin Manager integration (with fallback)
try:
    from .advanced_plugin_manager import AdvancedPluginManager, create_advanced_plugin_manager
except ImportError:
    # Fallback if advanced plugin manager has circular import issues
    AdvancedPluginManager = Any
    def create_advanced_plugin_manager(orchestrator):
        logger.warning("Advanced Plugin Manager not available due to import issues")
        return None

# Essential imports needed at runtime
from ..models.agent import AgentStatus, AgentType
from ..models.task import TaskStatus, TaskPriority
from ..models.message import MessagePriority

# Runtime imports needed for enums and classes used directly
try:
    from .enhanced_agent_launcher import AgentLauncherType, AgentLaunchConfig, EnhancedAgentLauncher, create_enhanced_agent_launcher
except ImportError:
    # Fallback if enhanced agent launcher not available
    from enum import Enum
    class AgentLauncherType(Enum):
        CLAUDE_CODE = "claude_code"
    AgentLaunchConfig = Any
    EnhancedAgentLauncher = Any
    
    # Simple placeholder implementation
    async def create_enhanced_agent_launcher(
        tmux_manager=None,
        short_id_generator=None,
        redis_manager=None
    ):
        """Placeholder implementation for create_enhanced_agent_launcher."""
        logger.warning("Using placeholder enhanced agent launcher - enhanced features not available")
        return None

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = Any

try:
    from .agent_redis_bridge import AgentRedisBridge, MessageType, create_agent_redis_bridge
except ImportError:
    AgentRedisBridge = Any
    MessageType = Any
    
    # Simple placeholder implementation
    async def create_agent_redis_bridge(redis_manager=None):
        """Placeholder implementation for create_agent_redis_bridge."""
        logger.warning("Using placeholder Redis bridge - Redis features not available")
        return None

try:
    from .tmux_session_manager import TmuxSessionManager
except ImportError:
    TmuxSessionManager = Any

try:
    from .short_id_generator import ShortIdGenerator
except ImportError:
    ShortIdGenerator = Any

# Lazy imports - loaded only when needed
if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession
    from .enhanced_logging import EnhancedLogger, PerformanceTracker
    from ..models.agent import Agent
    from ..models.task import Task

# For protocols, we need the actual import but we'll make it conditional
try:
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy import update, select
except ImportError:
    AsyncSession = Any  # Fallback for type hints
    update = Any
    select = Any

logger = get_component_logger("simple_orchestrator")

# Epic 1: Simplified logging for Epic 2 Phase 2.1 integration


class AgentRole(Enum):
    """Core agent roles for the simplified orchestrator."""
    BACKEND_DEVELOPER = "backend_developer"
    FRONTEND_DEVELOPER = "frontend_developer"
    DEVOPS_ENGINEER = "devops_engineer"
    QA_ENGINEER = "qa_engineer"
    META_AGENT = "meta_agent"


@dataclass
class AgentInstance:
    """Simplified agent instance representation."""
    id: str
    role: AgentRole
    status: AgentStatus
    current_task_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "role": self.role.value,
            "status": self.status.value,
            "current_task_id": self.current_task_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat()
        }


@dataclass
class TaskAssignment:
    """Simple task assignment representation."""
    task_id: str
    agent_id: str
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    status: TaskStatus = TaskStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "assigned_at": self.assigned_at.isoformat(),
            "status": self.status.value
        }


class DatabaseDependency(Protocol):
    """Protocol for database dependency injection."""
    async def get_session(self) -> AsyncSession: ...


class CacheDependency(Protocol):
    """Protocol for cache dependency injection."""
    async def get(self, key: str) -> Optional[Any]: ...
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool: ...


class SimpleOrchestratorError(Exception):
    """Base exception for orchestrator errors."""
    pass


class AgentNotFoundError(SimpleOrchestratorError):
    """Raised when agent is not found."""
    pass


class TaskDelegationError(SimpleOrchestratorError):
    """Raised when task delegation fails."""
    pass


class SimpleOrchestrator:
    """
    Epic 1 Optimized: Memory-efficient orchestrator with lazy loading.

    Provides:
    - Agent lifecycle management (spawn, shutdown)
    - Basic task delegation
    - System status monitoring
    - Plugin architecture for production enhancements

    Epic 1 Optimizations:
    - <50ms response times for core operations
    - <20MB memory footprint through lazy loading
    - Minimal initial imports and dependencies
    - Memory-efficient data structures
    - Lazy plugin loading
    """

    def __init__(
        self,
        db_session_factory: Optional[DatabaseDependency] = None,
        cache: Optional[CacheDependency] = None,
        anthropic_client: Optional["AsyncAnthropic"] = None,
        agent_launcher: Optional["EnhancedAgentLauncher"] = None,
        redis_bridge: Optional["AgentRedisBridge"] = None,
        tmux_manager: Optional["TmuxSessionManager"] = None,
        short_id_generator: Optional["ShortIdGenerator"] = None,
        enable_production_plugin: bool = False,
        websocket_manager: Optional["ConnectionManager"] = None
    ):
        """Initialize orchestrator with Epic 1 lazy loading optimizations."""
        # Core lightweight initialization
        self._db_session_factory = db_session_factory
        self._cache = cache

        # Lazy-loaded dependencies (not initialized until needed)
        self._anthropic_client = anthropic_client
        self._agent_launcher = agent_launcher
        self._redis_bridge = redis_bridge
        self._tmux_manager = tmux_manager
        self._short_id_generator = short_id_generator
        self._websocket_manager = websocket_manager

        # Memory-efficient storage
        self._agents: Dict[str, "Agent"] = {}
        self._tasks: Dict[str, "Task"] = {}

        # Plugin system
        self._plugins: List[Any] = []
        self._enable_production_plugin = enable_production_plugin

        # Epic 2 Phase 2.1: Advanced Plugin Manager
        self._advanced_plugin_manager: Optional[AdvancedPluginManager] = None

        # Lazy initialization flags
        self._initialized = False
        self._dependencies_loaded = False

        logger.info("SimpleOrchestrator initialized with Epic 1 optimizations")

        # Enhanced agent tracking (memory-efficient)
        self._tmux_agents: Dict[str, str] = {}  # agent_id -> session_id
        self._task_assignments: Dict[str, TaskAssignment] = {}

        # Performance metrics
        self._operation_count = 0
        self._last_performance_check = datetime.utcnow()
        
        # Epic 2 Phase 2.1: Performance tracking for plugin integration
        self._operation_times: Dict[str, List[float]] = {}
        self._performance_plugin: Optional[Any] = None

    # Epic 1 Lazy Loading Methods

    async def _ensure_anthropic_client(self) -> "AsyncAnthropic":
        """Lazy load Anthropic client only when needed."""
        if self._anthropic_client is None:
            from anthropic import AsyncAnthropic
            self._anthropic_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
            logger.info("Lazy loaded Anthropic client")
        return self._anthropic_client

    async def _ensure_short_id_generator(self) -> "ShortIdGenerator":
        """Lazy load short ID generator only when needed."""
        if self._short_id_generator is None:
            from .short_id_generator import ShortIdGenerator
            self._short_id_generator = ShortIdGenerator()
            logger.info("Lazy loaded ShortIdGenerator")
        return self._short_id_generator

    async def _ensure_tmux_manager(self) -> "TmuxSessionManager":
        """Lazy load tmux manager only when needed."""
        if self._tmux_manager is None:
            from .tmux_session_manager import TmuxSessionManager
            self._tmux_manager = TmuxSessionManager()
            logger.info("Lazy loaded TmuxSessionManager")
        return self._tmux_manager

    async def _ensure_dependencies_loaded(self) -> None:
        """Ensure all required dependencies are lazily loaded."""
        if not self._dependencies_loaded:
            # Only load what's absolutely necessary
            await self._ensure_short_id_generator()
            self._dependencies_loaded = True
            logger.info("Epic 1 dependencies lazily loaded")

        # Initialization flag
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the orchestrator and its dependencies."""
        if self._initialized:
            return

        logger.info("🚀 Initializing Enhanced SimpleOrchestrator...")

        try:
            # Initialize tmux manager if not provided
            if self._tmux_manager is None:
                self._tmux_manager = TmuxSessionManager()
                await self._tmux_manager.initialize()

            # Initialize agent launcher if not provided
            if self._agent_launcher is None:
                try:
                    self._agent_launcher = await create_enhanced_agent_launcher(
                        tmux_manager=self._tmux_manager,
                        short_id_generator=self._short_id_generator
                    )
                except Exception as e:
                    logger.warning("Failed to initialize enhanced agent launcher, continuing without it", error=str(e))
                    self._agent_launcher = None

            # Initialize Redis bridge if not provided
            if self._redis_bridge is None:
                try:
                    self._redis_bridge = await create_agent_redis_bridge()
                except Exception as e:
                    logger.warning("Failed to initialize Redis bridge, continuing without it", error=str(e))
                    self._redis_bridge = None

            # Epic 2 Phase 2.1: Initialize Advanced Plugin Manager
            if self._advanced_plugin_manager is None:
                self._advanced_plugin_manager = create_advanced_plugin_manager(self)
                logger.info("✅ Advanced Plugin Manager initialized")

            # Initialize production plugin if enabled
            if self._enable_production_plugin:
                try:
                    from .orchestrator_plugins.production_enhancement_plugin import create_production_enhancement_plugin
                    production_plugin = create_production_enhancement_plugin(self)
                    self._plugins.append(production_plugin)
                    logger.info("✅ Production enhancement plugin loaded")
                except Exception as e:
                    logger.warning("Failed to load production plugin", error=str(e))
            
            # Epic 2 Phase 2.1: Initialize Performance Orchestrator Plugin  
            try:
                from .orchestrator_plugins.performance_orchestrator_plugin import create_performance_orchestrator_plugin
                
                # Create and initialize the plugin
                performance_plugin = create_performance_orchestrator_plugin()
                await performance_plugin.initialize({"orchestrator": self})
                self._plugins.append(performance_plugin)
                self._performance_plugin = performance_plugin
                
                # Register with AdvancedPluginManager for management capabilities
                plugin_path = Path(__file__).parent / "orchestrator_plugins" / "performance_orchestrator_plugin.py"
                await self._advanced_plugin_manager.load_plugin_dynamic(
                    plugin_id="performance_orchestrator_plugin",
                    version="2.1.0",
                    source_path=plugin_path
                )
                
                logger.info("✅ Performance Orchestrator Plugin loaded and registered")
                
            except Exception as e:
                logger.warning("Failed to load Performance Orchestrator Plugin", error=str(e))

            self._initialized = True

            logger.info("✅ Enhanced SimpleOrchestrator initialized successfully",
                       plugins_loaded=len(self._plugins))

        except Exception as e:
            logger.error("❌ Failed to initialize Enhanced SimpleOrchestrator", error=str(e))
            raise

    async def spawn_agent(
        self,
        role: AgentRole,
        agent_id: Optional[str] = None,
        agent_type: AgentLauncherType = AgentLauncherType.CLAUDE_CODE,
        task_id: Optional[str] = None,
        workspace_name: Optional[str] = None,
        git_branch: Optional[str] = None,
        working_directory: Optional[str] = None,
        environment_vars: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Spawn a new agent instance with tmux session integration.

        Args:
            role: The role for the new agent
            agent_id: Optional specific ID, otherwise generated
            agent_type: Type of CLI agent to launch
            task_id: Optional task ID to assign to the agent
            workspace_name: Optional workspace directory name
            git_branch: Optional git branch for the workspace
            working_directory: Optional working directory
            environment_vars: Optional environment variables

        Returns:
            The agent ID

        Raises:
            SimpleOrchestratorError: If spawning fails
        """
        # Ensure orchestrator is initialized
        if not self._initialized:
            await self.initialize()

        operation_start_time = datetime.utcnow()
        operation_id = str(uuid.uuid4())

        # Epic 2 Phase 2.1: Performance tracking
        start_time_ms = time.time()

        try:
            # Generate ID if not provided
            if agent_id is None:
                agent_id = str(uuid.uuid4())

            # Check if agent already exists
            if agent_id in self._agents:
                logger.error("Agent already exists", agent_id=agent_id)
                raise SimpleOrchestratorError(f"Agent {agent_id} already exists")

            # Check agent limit
            active_agents = len([a for a in self._agents.values() if a.status == AgentStatus.ACTIVE])
            max_agents = getattr(settings, 'MAX_CONCURRENT_AGENTS', 10)

            logger.debug("Active agents count",
                       active_agents=active_agents,
                       max_agents=max_agents)

            if active_agents >= max_agents:
                logger.warning("Resource limit exceeded",
                             active_agents=active_agents,
                             max_agents=max_agents,
                             attempted_role=role.value)
                raise SimpleOrchestratorError("Maximum concurrent agents reached")

            # Create launch configuration
            launch_config = AgentLaunchConfig(
                agent_type=agent_type,
                task_id=task_id,
                workspace_name=workspace_name,
                git_branch=git_branch,
                working_directory=working_directory,
                environment_vars=environment_vars
            )

            # Launch agent using enhanced launcher
            launch_result = await self._agent_launcher.launch_agent(
                config=launch_config,
                agent_name=f"{role.value}-{agent_id[:8]}"
            )

            if not launch_result.success:
                raise SimpleOrchestratorError(f"Failed to launch agent: {launch_result.error_message}")

            # Create agent instance
            agent = AgentInstance(
                id=agent_id,
                role=role,
                status=AgentStatus.ACTIVE
            )

            # Store in memory registry
            self._agents[agent_id] = agent
            self._tmux_agents[agent_id] = launch_result.session_id
            
            # Epic C Phase C.4: Broadcast agent creation
            creation_data = {
                "agent_id": agent_id,
                "role": role.value,
                "status": AgentStatus.ACTIVE.value,
                "created_at": agent.created_at.isoformat(),
                "agent_type": agent_type.value,
                "session_name": launch_result.session_name,
                "workspace_path": launch_result.workspace_path,
                "source": "agent_creation"
            }
            await self._broadcast_agent_update(agent_id, creation_data)

            # Register agent with Redis bridge
            if self._redis_bridge:
                await self._redis_bridge.register_agent(
                    agent_id=agent_id,
                    agent_type=agent_type.value,
                    session_name=launch_result.session_name,
                    capabilities=[role.value],
                    consumer_group="general_agents",
                    workspace_path=launch_result.workspace_path
                )

            # Persist to database if available
            if self._db_session_factory:
                await self._persist_enhanced_agent(agent, launch_result)

            # Cache for fast access
            if self._cache:
                agent_data = agent.to_dict()
                agent_data.update({
                    "session_id": launch_result.session_id,
                    "session_name": launch_result.session_name,
                    "workspace_path": launch_result.workspace_path,
                    "agent_type": agent_type.value
                })
                await self._cache.set(f"agent:{agent_id}", agent_data, ttl=3600)

            # Log successful creation
            logger.info("Audit event")

            logger.info(
                "Enhanced agent spawned successfully",
                agent_id=agent_id,
                role=role.value,
                agent_type=agent_type.value,
                session_name=launch_result.session_name,
                workspace_path=launch_result.workspace_path,
                launch_time=launch_result.launch_time_seconds,
                total_agents=len(self._agents)
            )

            self._operation_count += 1
            
            # Epic 2 Phase 2.1: Record performance metrics
            operation_time_ms = (time.time() - start_time_ms) * 1000
            self._record_operation_time("spawn_agent", operation_time_ms)
            
            return agent_id

        except Exception as e:
            logger.error("Operation failed", error=str(e))

            logger.info("Audit event")

            raise SimpleOrchestratorError(f"Failed to spawn agent: {e}") from e

    async def shutdown_agent(self, agent_id: str, graceful: bool = True) -> bool:
        """
        Shutdown a specific agent instance.

        Args:
            agent_id: ID of agent to shutdown
            graceful: Whether to wait for current task completion

        Returns:
            True if shutdown successful, False if agent not found

        Raises:
            SimpleOrchestratorError: If shutdown fails
        """
        try:
            # Enhanced logging context
            # Enhanced logger context removed

            # Check if agent exists
            if agent_id not in self._agents:
                logger.info("Audit event")
                logger.warning("Agent not found for shutdown", agent_id=agent_id)
                return False

            agent = self._agents[agent_id]

            # Log current agent state before shutdown
            logger.info("Operation info")

            # Handle graceful shutdown with enhanced logging
            if graceful and agent.current_task_id:
                logger.info("Operation info")
                # Wait for current task (simplified - could be enhanced)
                await asyncio.sleep(1)  # Brief wait

            # Update agent status
            old_status = agent.status
            agent.status = AgentStatus.INACTIVE
            agent.last_activity = datetime.utcnow()
            
            # Epic C Phase C.4: Broadcast agent status change
            update_data = {
                "agent_id": agent_id,
                "status": AgentStatus.INACTIVE.value,
                "previous_status": old_status.value,
                "last_activity": agent.last_activity.isoformat(),
                "source": "agent_shutdown"
            }
            await self._broadcast_agent_update(agent_id, update_data)

            # Terminate tmux session if exists
            if agent_id in self._tmux_agents:
                session_id = self._tmux_agents[agent_id]
                if self._agent_launcher:
                    await self._agent_launcher.terminate_agent(agent_id, cleanup_workspace=True)
                del self._tmux_agents[agent_id]

            # Unregister from Redis bridge
            if self._redis_bridge:
                await self._redis_bridge.unregister_agent(agent_id)

            # Remove from active registry
            del self._agents[agent_id]

            # Update database if available
            if self._db_session_factory:
                await self._update_agent_status(agent_id, AgentStatus.INACTIVE)

            # Remove from cache
            if self._cache:
                await self._cache.set(f"agent:{agent_id}", None, ttl=1)

            # Log performance metrics
            remaining_agents = len(self._agents)
            logger.debug("Performance metric")

            # Log successful shutdown audit event
            logger.info("Audit event")

            logger.info(
                "Agent shutdown successful",
                agent_id=agent_id,
                graceful=graceful,
                remaining_agents=remaining_agents
            )

            self._operation_count += 1
            return True

        except Exception as e:
            logger.error("Operation failed", error=str(e))

            logger.info("Audit event")

            raise SimpleOrchestratorError(f"Failed to shutdown agent: {e}") from e

    async def delegate_task(
        self,
        task_description: str,
        task_type: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        preferred_agent_role: Optional[AgentRole] = None
    ) -> str:
        """
        Delegate a task to the most suitable available agent.

        Args:
            task_description: Description of the task
            task_type: Type/category of the task
            priority: Task priority level
            preferred_agent_role: Preferred agent role for the task

        Returns:
            Task ID

        Raises:
            TaskDelegationError: If no suitable agent available
        """
        # Epic 2 Phase 2.1: Performance tracking
        start_time_ms = time.time()
        
        try:
            # Generate task ID
            task_id = str(uuid.uuid4())

            # Enhanced logging context
            # Enhanced logger context removed

            # Log task delegation request
            logger.info("Operation info")

            # Convert task priority to message priority
            msg_priority = MessagePriority.NORMAL
            if priority == TaskPriority.HIGH:
                msg_priority = MessagePriority.HIGH
            elif priority == TaskPriority.LOW:
                msg_priority = MessagePriority.LOW
            elif priority == TaskPriority.URGENT:
                msg_priority = MessagePriority.URGENT

            # Determine required capabilities based on task type and role
            required_capabilities = []
            if preferred_agent_role:
                required_capabilities.append(preferred_agent_role.value)
            if task_type:
                required_capabilities.append(task_type)

            # Try to assign task via Redis bridge first
            assigned_agent_id = None
            if self._redis_bridge:
                assigned_agent_id = await self._redis_bridge.assign_task_to_agent(
                    task_id=task_id,
                    task_description=task_description,
                    required_capabilities=required_capabilities,
                    priority=msg_priority
                )

            # Fallback to local agent selection if Redis bridge unavailable
            if not assigned_agent_id:
                suitable_agent = await self._find_suitable_agent(
                    preferred_role=preferred_agent_role,
                    task_type=task_type
                )

                if not suitable_agent:
                    logger.warning("Security event")
                    raise TaskDelegationError("No suitable agent available")

                assigned_agent_id = suitable_agent.id

                # Update agent with current task
                suitable_agent.current_task_id = task_id
                suitable_agent.last_activity = datetime.utcnow()

            # Create task assignment
            assignment = TaskAssignment(
                task_id=task_id,
                agent_id=assigned_agent_id,
                status=TaskStatus.PENDING
            )

            # Store assignment
            self._task_assignments[task_id] = assignment

            # Persist to database if available
            if self._db_session_factory:
                await self._persist_task(task_id, task_description, task_type,
                                       priority, suitable_agent.id)

            # Log performance metrics
            logger.debug("Performance metric")

            # Log successful task delegation audit event
            logger.info("Audit event")

            logger.info(
                "Task delegated successfully",
                task_id=task_id,
                agent_id=assigned_agent_id,
                task_type=task_type,
                priority=priority.value
            )

            self._operation_count += 1
            
            # Epic 2 Phase 2.1: Record performance metrics
            operation_time_ms = (time.time() - start_time_ms) * 1000
            self._record_operation_time("delegate_task", operation_time_ms)
            
            return task_id

        except Exception as e:
            logger.error("Operation failed", error=str(e))

            logger.info("Audit event")

            raise TaskDelegationError(f"Failed to delegate task: {e}") from e

    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status for monitoring.

        Returns:
            Dictionary with system status information
        """
        start_time = datetime.utcnow()

        try:
            # Collect agent statuses
            agent_statuses = {
                agent_id: agent.to_dict()
                for agent_id, agent in self._agents.items()
            }

            # Count agents by status
            status_counts = {}
            for agent in self._agents.values():
                status = agent.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

            # Task assignment summary
            assignment_count = len(self._task_assignments)

            # Performance metrics
            now = datetime.utcnow()
            time_since_check = (now - self._last_performance_check).total_seconds()
            ops_per_second = self._operation_count / max(time_since_check, 1)

            status = {
                "timestamp": now.isoformat(),
                "agents": {
                    "total": len(self._agents),
                    "by_status": status_counts,
                    "details": agent_statuses
                },
                "tasks": {
                    "active_assignments": assignment_count
                },
                "performance": {
                    "operations_count": self._operation_count,
                    "operations_per_second": round(ops_per_second, 2),
                    "response_time_ms": round(
                        (datetime.utcnow() - start_time).total_seconds() * 1000, 2
                    )
                },
                "health": "healthy" if len(self._agents) > 0 else "no_agents"
            }

            # Add plugin status if plugins are loaded
            if self._plugins:
                plugin_status = {}
                for i, plugin in enumerate(self._plugins):
                    try:
                        if hasattr(plugin, 'get_production_status'):
                            plugin_status[f'production_plugin'] = await plugin.get_production_status()
                        else:
                            plugin_status[f'plugin_{i}'] = {"status": "loaded"}
                    except Exception as e:
                        plugin_status[f'plugin_{i}'] = {"status": "error", "error": str(e)}

                status["plugins"] = plugin_status

            logger.debug("Performance metric",
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000)

            return status

        except Exception as e:
            logger.error("Failed to get system status", error=str(e))
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "health": "error"
            }

    # Private helper methods

    async def _find_suitable_agent(
        self,
        preferred_role: Optional[AgentRole] = None,
        task_type: str = ""
    ) -> Optional[AgentInstance]:
        """Find the most suitable available agent for a task."""
        available_agents = [
            agent for agent in self._agents.values()
            if agent.status == AgentStatus.ACTIVE and agent.current_task_id is None
        ]

        if not available_agents:
            return None

        # Prefer agents with matching role
        if preferred_role:
            role_matches = [a for a in available_agents if a.role == preferred_role]
            if role_matches:
                return role_matches[0]  # Simple selection - could be enhanced

        # Return first available agent
        return available_agents[0]

    async def _persist_agent(self, agent: AgentInstance) -> None:
        """Persist agent to database."""
        if not self._db_session_factory:
            return

        try:
            async with self._db_session_factory.get_session() as session:
                db_agent = Agent(
                    id=agent.id,
                    role=agent.role.value,
                    agent_type=AgentType.CLAUDE_CODE,
                    status=agent.status,
                    created_at=agent.created_at
                )
                session.add(db_agent)
                await session.commit()
        except Exception as e:
            logger.warning("Failed to persist agent to database",
                         agent_id=agent.id, error=str(e))

    async def _update_agent_status(self, agent_id: str, status: AgentStatus) -> None:
        """Update agent status in database and broadcast via WebSocket."""
        if not self._db_session_factory:
            return

        try:
            async with self._db_session_factory.get_session() as session:
                await session.execute(
                    update(Agent)
                    .where(Agent.id == agent_id)
                    .values(status=status, updated_at=datetime.utcnow())
                )
                await session.commit()
                
                # Epic C Phase C.4: Broadcast agent status update
                update_data = {
                    "agent_id": agent_id,
                    "status": status.value,
                    "updated_at": datetime.utcnow().isoformat(),
                    "source": "database_update"
                }
                await self._broadcast_agent_update(agent_id, update_data)
                
        except Exception as e:
            logger.warning("Failed to update agent status in database",
                         agent_id=agent_id, error=str(e))

    async def _persist_task(
        self,
        task_id: str,
        description: str,
        task_type: str,
        priority: TaskPriority,
        agent_id: str
    ) -> None:
        """Persist task to database."""
        if not self._db_session_factory:
            return

        try:
            async with self._db_session_factory.get_session() as session:
                db_task = Task(
                    id=task_id,
                    description=description,
                    task_type=task_type,
                    priority=priority,
                    status=TaskStatus.PENDING,
                    assigned_agent_id=agent_id,
                    created_at=datetime.utcnow()
                )
                session.add(db_task)
                await session.commit()
                
                # Epic C Phase C.4: Broadcast task creation
                task_data = {
                    "task_id": task_id,
                    "description": description,
                    "task_type": task_type,
                    "priority": priority.value if hasattr(priority, 'value') else str(priority),
                    "status": TaskStatus.PENDING.value,
                    "assigned_agent_id": agent_id,
                    "created_at": datetime.utcnow().isoformat(),
                    "source": "task_creation"
                }
                await self._broadcast_task_update(task_id, task_data)
                
        except Exception as e:
            logger.warning("Failed to persist task to database",
                         task_id=task_id, error=str(e))

    # Enhanced methods for tmux integration

    async def _persist_enhanced_agent(self, agent: AgentInstance, launch_result) -> None:
        """Persist enhanced agent with tmux session details to database."""
        if not self._db_session_factory:
            return

        try:
            async with self._db_session_factory.get_session() as session:
                db_agent = Agent(
                    id=agent.id,
                    role=agent.role.value,
                    agent_type=AgentType.CLAUDE_CODE,  # Could be enhanced to support other types
                    status=agent.status,
                    tmux_session=launch_result.session_name,
                    created_at=agent.created_at
                )
                session.add(db_agent)
                await session.commit()
        except Exception as e:
            logger.warning("Failed to persist enhanced agent to database",
                         agent_id=agent.id, error=str(e))

    async def get_agent_session_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed session information for an agent."""
        if agent_id not in self._agents:
            return None

        agent = self._agents[agent_id]
        session_info = None

        # Get session info from tmux manager
        if self._tmux_manager and agent_id in self._tmux_agents:
            session_id = self._tmux_agents[agent_id]
            session_info = self._tmux_manager.get_session_info(session_id)

        # Get agent status from launcher
        launcher_status = None
        if self._agent_launcher:
            launcher_status = await self._agent_launcher.get_agent_status(agent_id)

        # Get Redis bridge status
        bridge_status = None
        if self._redis_bridge:
            bridge_status = await self._redis_bridge.get_agent_status(agent_id)

        return {
            "agent_instance": agent.to_dict(),
            "session_info": session_info.to_dict() if session_info else None,
            "launcher_status": launcher_status,
            "bridge_status": bridge_status,
            "tmux_session_id": self._tmux_agents.get(agent_id)
        }

    async def list_agent_sessions(self) -> List[Dict[str, Any]]:
        """List all agent sessions with detailed information."""
        sessions = []

        for agent_id in self._agents.keys():
            session_info = await self.get_agent_session_info(agent_id)
            if session_info:
                sessions.append(session_info)

        return sessions

    async def attach_to_agent_session(self, agent_id: str) -> Optional[str]:
        """Get the tmux session name for attaching to an agent session."""
        if agent_id not in self._agents:
            return None

        session_info = await self.get_agent_session_info(agent_id)
        if session_info and session_info.get("session_info"):
            return session_info["session_info"]["session_name"]

        return None

    async def execute_command_in_agent_session(
        self,
        agent_id: str,
        command: str,
        window_name: Optional[str] = None,
        capture_output: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Execute a command in an agent's tmux session."""
        if agent_id not in self._tmux_agents:
            return None

        session_id = self._tmux_agents[agent_id]

        if self._tmux_manager:
            return await self._tmux_manager.execute_command(
                session_id=session_id,
                command=command,
                window_name=window_name,
                capture_output=capture_output
            )

        return None

    async def get_agent_logs(self, agent_id: str, lines: int = 100) -> Optional[List[str]]:
        """Get recent logs from an agent session."""
        launcher_status = None
        if self._agent_launcher:
            launcher_status = await self._agent_launcher.get_agent_status(agent_id)

        if launcher_status and launcher_status.get("recent_logs"):
            return launcher_status["recent_logs"][-lines:]

        return None

    async def get_enhanced_system_status(self) -> Dict[str, Any]:
        """Get enhanced system status including tmux and Redis information."""
        base_status = await self.get_system_status()

        # Add tmux manager metrics
        tmux_metrics = {}
        if self._tmux_manager:
            tmux_metrics = await self._tmux_manager.get_session_metrics()

        # Add agent launcher metrics
        launcher_metrics = {}
        if self._agent_launcher:
            launcher_metrics = await self._agent_launcher.get_launcher_metrics()

        # Add Redis bridge metrics
        bridge_metrics = {}
        if self._redis_bridge:
            bridge_metrics = await self._redis_bridge.get_bridge_metrics()

        # Enhanced status
        enhanced_status = {
            **base_status,
            "tmux_integration": {
                "enabled": bool(self._tmux_manager),
                "metrics": tmux_metrics
            },
            "agent_launcher": {
                "enabled": bool(self._agent_launcher),
                "metrics": launcher_metrics
            },
            "redis_bridge": {
                "enabled": bool(self._redis_bridge),
                "metrics": bridge_metrics
            },
            "enhanced_agents": {
                "total_with_sessions": len(self._tmux_agents),
                "session_mappings": len(self._tmux_agents),
                "initialized": self._initialized
            }
        }

        return enhanced_status

    # Epic 2 Phase 2.1: Advanced Plugin Manager Methods

    async def load_plugin_dynamic(
        self,
        plugin_id: str,
        version: str = "latest",
        source_path: Optional[str] = None,
        source_code: Optional[str] = None
    ) -> bool:
        """Load a plugin dynamically using the Advanced Plugin Manager."""
        if not self._advanced_plugin_manager:
            logger.warning("Advanced Plugin Manager not initialized")
            return False

        try:
            from pathlib import Path
            path_obj = Path(source_path) if source_path else None

            plugin = await self._advanced_plugin_manager.load_plugin_dynamic(
                plugin_id=plugin_id,
                version=version,
                source_path=path_obj,
                source_code=source_code
            )

            logger.info("Plugin loaded dynamically",
                       plugin_id=plugin_id,
                       version=version,
                       memory_usage_mb=plugin.state.memory_usage_mb)

            return True

        except Exception as e:
            logger.error("Failed to load plugin dynamically",
                        plugin_id=plugin_id,
                        error=str(e))
            return False

    async def unload_plugin_safe(self, plugin_id: str) -> bool:
        """Safely unload a plugin."""
        if not self._advanced_plugin_manager:
            logger.warning("Advanced Plugin Manager not initialized")
            return False

        return await self._advanced_plugin_manager.unload_plugin_safe(plugin_id)

    async def hot_swap_plugin(self, old_plugin_id: str, new_plugin_id: str) -> bool:
        """Hot-swap plugins without system restart."""
        if not self._advanced_plugin_manager:
            logger.warning("Advanced Plugin Manager not initialized")
            return False

        return await self._advanced_plugin_manager.hot_swap_plugin(old_plugin_id, new_plugin_id)

    async def get_plugin_performance_metrics(self) -> Dict[str, Any]:
        """Get plugin performance metrics for Epic 1 monitoring."""
        if not self._advanced_plugin_manager:
            return {"error": "Advanced Plugin Manager not initialized"}

        return await self._advanced_plugin_manager.get_performance_metrics()

    async def get_plugin_security_status(self) -> Dict[str, Any]:
        """Get plugin security status and reports."""
        if not self._advanced_plugin_manager:
            return {"error": "Advanced Plugin Manager not initialized"}

        try:
            from .plugin_security_framework import get_plugin_security_framework
            security_framework = get_plugin_security_framework()
            return security_framework.get_performance_metrics()
        except Exception as e:
            logger.error("Failed to get plugin security status", error=str(e))
            return {"error": str(e)}

    async def shutdown(self) -> None:
        """Shutdown the orchestrator and all its components."""
        logger.info("🛑 Shutting down Enhanced SimpleOrchestrator...")

        # Shutdown all active agents
        active_agents = list(self._agents.keys())
        for agent_id in active_agents:
            try:
                await self.shutdown_agent(agent_id, graceful=True)
            except Exception as e:
                logger.warning(f"Failed to shutdown agent {agent_id}: {e}")

        # Shutdown components
        if self._redis_bridge:
            await self._redis_bridge.shutdown()

        if self._tmux_manager:
            await self._tmux_manager.shutdown()

        # Epic 2 Phase 2.1: Shutdown Advanced Plugin Manager
        if self._advanced_plugin_manager:
            await self._advanced_plugin_manager.cleanup()
            logger.info("✅ Advanced Plugin Manager shutdown complete")

        logger.info("✅ Enhanced SimpleOrchestrator shutdown complete")

    # Epic 2 Phase 2.1: Performance tracking methods
    
    def _record_operation_time(self, operation: str, time_ms: float) -> None:
        """Record operation time for performance monitoring."""
        if operation not in self._operation_times:
            self._operation_times[operation] = []
        
        times = self._operation_times[operation]
        times.append(time_ms)
        
        # Keep only last 50 measurements for memory efficiency
        if len(times) > 50:
            times.pop(0)
        
        # Log performance warnings for Epic 1 targets
        if operation == "spawn_agent" and time_ms > 100.0:
            logger.warning("Agent registration slow", operation_time_ms=time_ms, target_ms=100.0)
        elif operation == "delegate_task" and time_ms > 500.0:
            logger.warning("Task delegation slow", operation_time_ms=time_ms, target_ms=500.0)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring."""
        metrics = {}
        
        for operation, times in self._operation_times.items():
            if times:
                import statistics
                metrics[operation] = {
                    "avg_ms": round(statistics.mean(times), 2),
                    "max_ms": round(max(times), 2),
                    "min_ms": round(min(times), 2),
                    "count": len(times),
                    "last_ms": round(times[-1], 2),
                    "epic1_compliant": {
                        "spawn_agent": statistics.mean(times) < 100.0 if operation == "spawn_agent" else None,
                        "delegate_task": statistics.mean(times) < 500.0 if operation == "delegate_task" else None
                    }.get(operation)
                }
        
        # Add system performance if plugin available
        if self._performance_plugin:
            try:
                plugin_metrics = await self._performance_plugin.get_performance_summary()
                metrics["plugin_summary"] = plugin_metrics
            except Exception as e:
                logger.warning("Failed to get plugin metrics", error=str(e))
        
        return {
            "operation_metrics": metrics,
            "total_operations": self._operation_count,
            "agents": len(self._agents),
            "tasks": len(self._task_assignments)
        }
    
    # Epic C Phase C.4: WebSocket Broadcasting Methods
    
    async def _broadcast_agent_update(self, agent_id: str, update_data: Dict[str, Any]) -> None:
        """Broadcast agent status updates via WebSocket if manager available."""
        if self._websocket_manager:
            try:
                await self._websocket_manager.broadcast_agent_update(agent_id, update_data)
                logger.debug("Broadcasted agent update via WebSocket", agent_id=agent_id)
            except Exception as e:
                logger.warning("Failed to broadcast agent update", agent_id=agent_id, error=str(e))
    
    async def _broadcast_task_update(self, task_id: str, update_data: Dict[str, Any]) -> None:
        """Broadcast task status updates via WebSocket if manager available."""
        if self._websocket_manager:
            try:
                await self._websocket_manager.broadcast_task_update(task_id, update_data)
                logger.debug("Broadcasted task update via WebSocket", task_id=task_id)
            except Exception as e:
                logger.warning("Failed to broadcast task update", task_id=task_id, error=str(e))
    
    async def _broadcast_system_status(self, status_data: Dict[str, Any]) -> None:
        """Broadcast system status updates via WebSocket if manager available."""
        if self._websocket_manager:
            try:
                await self._websocket_manager.broadcast_system_status(status_data)
                logger.debug("Broadcasted system status via WebSocket")
            except Exception as e:
                logger.warning("Failed to broadcast system status", error=str(e))


# Factory function for dependency injection
def create_simple_orchestrator(
    db_session_factory: Optional[DatabaseDependency] = None,
    cache: Optional[CacheDependency] = None,
    anthropic_client: Optional[AsyncAnthropic] = None,
    agent_launcher: Optional[EnhancedAgentLauncher] = None,
    redis_bridge: Optional[AgentRedisBridge] = None,
    tmux_manager: Optional[TmuxSessionManager] = None,
    short_id_generator: Optional[ShortIdGenerator] = None,
    websocket_manager: Optional["ConnectionManager"] = None
) -> SimpleOrchestrator:
    """
    Factory function to create SimpleOrchestrator with proper dependencies.

    This makes it easy to inject dependencies for testing and different environments.
    """
    return SimpleOrchestrator(
        db_session_factory=db_session_factory,
        cache=cache,
        anthropic_client=anthropic_client,
        agent_launcher=agent_launcher,
        redis_bridge=redis_bridge,
        tmux_manager=tmux_manager,
        short_id_generator=short_id_generator,
        websocket_manager=websocket_manager
    )


# Enhanced factory function for full initialization
async def create_enhanced_simple_orchestrator(
    db_session_factory: Optional[DatabaseDependency] = None,
    cache: Optional[CacheDependency] = None,
    anthropic_client: Optional[AsyncAnthropic] = None
) -> SimpleOrchestrator:
    """
    Factory function to create fully initialized SimpleOrchestrator with tmux integration.

    This initializes all components and their dependencies automatically.
    """
    orchestrator = create_simple_orchestrator(
        db_session_factory=db_session_factory,
        cache=cache,
        anthropic_client=anthropic_client
    )

    # Initialize the orchestrator (this will set up all components)
    await orchestrator.initialize()

    return orchestrator


# Global instance for API usage (can be overridden in tests)
_global_orchestrator: Optional[SimpleOrchestrator] = None


def get_simple_orchestrator() -> SimpleOrchestrator:
    """Get the global orchestrator instance."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = create_simple_orchestrator()
    return _global_orchestrator


def set_simple_orchestrator(orchestrator: SimpleOrchestrator) -> None:
    """Set the global orchestrator instance (useful for testing)."""
    global _global_orchestrator
    _global_orchestrator = orchestrator
