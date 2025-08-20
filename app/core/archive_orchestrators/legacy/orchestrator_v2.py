"""
OrchestratorV2 - Unified Orchestrator with Plugin Architecture
LeanVibe Agent Hive 2.0 - Phase 0 POC Week 2

This module implements the unified orchestrator that consolidates 35+ legacy
orchestrator implementations into a single, plugin-based architecture.

Design validated by Gemini CLI expert analysis with implementations for:
- Plugin dependency management with topological sorting
- Granular hook system with before/after events
- Plugin state sandboxing and performance monitoring
- Migration adapters for legacy compatibility
- Circuit breaker protection and error isolation

CONSOLIDATION TARGET: 35+ orchestrators → 1 core + ~6 plugins
PERFORMANCE REQUIREMENTS: <100ms agent spawn, <500ms task delegation, 50+ concurrent agents
MIGRATION STRATEGY: Strangler Fig pattern with gradual rollout
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Union, Tuple

import structlog

from .unified_communication_protocol import (
    UnifiedCommunicationManager, 
    StandardUniversalMessage, 
    MessageType, 
    MessagePriority,
    get_communication_manager
)
from .circuit_breaker import UnifiedCircuitBreaker, CircuitBreakerConfig

logger = structlog.get_logger("orchestrator_v2")

# ================================================================================
# Core Data Models - Unified Across All Orchestrator Types
# ================================================================================

class AgentStatus(str, Enum):
    """Standardized agent status across all orchestrator types."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    SLEEPING = "sleeping"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"

class AgentRole(str, Enum):
    """Standard agent roles consolidated from legacy implementations."""
    CLAUDE_CODE = "claude_code"
    GEMINI_CLI = "gemini_cli" 
    GITHUB_COPILOT = "github_copilot"
    CURSOR = "cursor"
    OPENCODE = "opencode"
    WORKFLOW_COORDINATOR = "workflow_coordinator"
    TASK_EXECUTOR = "task_executor"
    CONTEXT_MANAGER = "context_manager"
    PERFORMANCE_MONITOR = "performance_monitor"

class TaskExecutionState(str, Enum):
    """Task execution states across all orchestrator types."""
    PENDING = "pending"
    ASSIGNED = "assigned" 
    RUNNING = "running"
    WAITING_FOR_INPUT = "waiting_for_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class AgentInstance:
    """Unified agent instance data structure."""
    id: str
    role: AgentRole
    status: AgentStatus
    capabilities: List[str] = field(default_factory=list)
    current_task_id: Optional[str] = None
    context_window_usage: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "role": self.role.value,
            "status": self.status.value,
            "capabilities": self.capabilities,
            "current_task_id": self.current_task_id,
            "context_window_usage": self.context_window_usage,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "performance_metrics": self.performance_metrics,
            "metadata": self.metadata
        }

@dataclass
class Task:
    """Unified task definition across all orchestrator types."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = "generic"
    priority: MessagePriority = MessagePriority.NORMAL
    description: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskExecution:
    """Task execution tracking across all orchestrator types."""
    task_id: str
    agent_id: str
    state: TaskExecutionState
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percentage: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

# ================================================================================
# Plugin System - Expert-Validated Architecture
# ================================================================================

class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass

class CircularDependencyError(PluginError):
    """Raised when a circular dependency is detected among plugins."""
    pass

class MissingDependencyError(PluginError):
    """Raised when a plugin's dependency is not found."""
    pass

class PluginTimeoutError(PluginError):
    """Raised when a plugin hook exceeds timeout."""
    pass

class OrchestratorPlugin(ABC):
    """
    Base plugin interface for specialized orchestrator behaviors.
    
    Gemini CLI recommendation: Plugins declare dependencies and receive
    sandboxed state management with performance monitoring.
    """
    
    plugin_name: str = "BasePlugin"
    dependencies: List[str] = []
    hook_timeout_ms: int = 100  # Per Gemini recommendation: performance budget
    
    def __init__(self, state_manager: 'PluginStateManager', performance_monitor: 'PluginPerformanceMonitor'):
        self.state = state_manager.get_plugin_state(self.plugin_name)
        self.performance_monitor = performance_monitor
        self._orchestrator: Optional['OrchestratorV2'] = None
    
    async def initialize(self, orchestrator: 'OrchestratorV2') -> None:
        """Initialize plugin with orchestrator reference."""
        self._orchestrator = orchestrator
        await self._register_hooks()
        logger.info("Plugin initialized", plugin=self.plugin_name)
    
    async def _register_hooks(self):
        """Register plugin hooks with the orchestrator."""
        # Gemini recommendation: granular before/after hooks
        hook_methods = [
            ("before_agent_spawn", getattr(self, "before_agent_spawn", None)),
            ("after_agent_spawn", getattr(self, "after_agent_spawn", None)),
            ("before_task_delegate", getattr(self, "before_task_delegate", None)),
            ("after_task_delegate", getattr(self, "after_task_delegate", None)),
            ("before_task_complete", getattr(self, "before_task_complete", None)),
            ("after_task_complete", getattr(self, "after_task_complete", None)),
            ("on_agent_error", getattr(self, "on_agent_error", None)),
            ("on_performance_metric", getattr(self, "on_performance_metric", None)),
            ("on_health_check", getattr(self, "on_health_check", None))
        ]
        
        for hook_name, method in hook_methods:
            if method and callable(method):
                self._orchestrator.hook_manager.register_hook(hook_name, self, method)
    
    # Hook methods (optional implementations)
    async def before_agent_spawn(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Called before agent spawn. Can modify agent_config."""
        return agent_config
    
    async def after_agent_spawn(self, agent: AgentInstance) -> None:
        """Called after agent spawn."""
        pass
    
    async def before_task_delegate(self, task: Task) -> Task:
        """Called before task delegation. Can modify task."""
        return task
    
    async def after_task_delegate(self, task: Task, agent_id: str) -> None:
        """Called after task delegation."""
        pass
    
    async def before_task_complete(self, execution: TaskExecution) -> None:
        """Called before task completion."""
        pass
    
    async def after_task_complete(self, execution: TaskExecution) -> None:
        """Called after task completion."""
        pass
    
    async def on_agent_error(self, agent_id: str, error: Exception) -> None:
        """Called when agent encounters error."""
        pass
    
    async def on_performance_metric(self, metric_name: str, value: float, metadata: Dict[str, Any]) -> None:
        """Called when performance metric is recorded."""
        pass
    
    async def on_health_check(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """Called during health check. Can add plugin-specific health info."""
        return health_data

# ================================================================================
# Plugin State Management - Gemini CLI Recommended Sandboxing
# ================================================================================

class PluginStateManager:
    """
    Manages plugin state with sandboxing to prevent state corruption.
    Gemini CLI recommendation: Namespaced state dictionaries per plugin.
    """
    
    def __init__(self):
        self._plugin_states = defaultdict(dict)
        self._state_lock = asyncio.Lock()
    
    def get_plugin_state(self, plugin_name: str) -> Dict[str, Any]:
        """Get sandboxed state dictionary for plugin."""
        return self._plugin_states[plugin_name]
    
    async def snapshot_state(self) -> Dict[str, Any]:
        """Get complete state snapshot for debugging/recovery."""
        async with self._state_lock:
            return {
                plugin_name: state.copy() 
                for plugin_name, state in self._plugin_states.items()
            }
    
    async def restore_state(self, snapshot: Dict[str, Any]) -> None:
        """Restore state from snapshot."""
        async with self._state_lock:
            self._plugin_states.clear()
            for plugin_name, state in snapshot.items():
                self._plugin_states[plugin_name] = state.copy()
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get state summary for monitoring."""
        return {
            "total_plugins": len(self._plugin_states),
            "total_state_items": sum(len(state) for state in self._plugin_states.values()),
            "plugin_names": list(self._plugin_states.keys())
        }

# ================================================================================
# Performance Monitoring - Gemini CLI Recommended Implementation
# ================================================================================

class PluginPerformanceMonitor:
    """
    Plugin performance monitoring with timeouts and budgets.
    Gemini CLI recommendation: Performance budgets with timeout enforcement.
    """
    
    def __init__(self, default_timeout_ms: int = 100):
        self.default_timeout = default_timeout_ms / 1000.0
        self.metrics = defaultdict(list)
        self.violation_counts = defaultdict(int)
        self.disabled_plugins = set()
        
    @asynccontextmanager
    async def time_hook(self, plugin_name: str, hook_name: str, timeout_ms: Optional[int] = None):
        """Time plugin hook execution with timeout enforcement."""
        timeout = (timeout_ms or (self.default_timeout * 1000)) / 1000.0
        start_time = time.perf_counter()
        
        try:
            # Gemini CLI recommendation: Use asyncio.wait_for to prevent blocking
            async with asyncio.timeout(timeout):
                yield
        except (asyncio.TimeoutError, TimeoutError):
            duration = time.perf_counter() - start_time
            self.violation_counts[plugin_name] += 1
            logger.warning("Plugin hook timeout", 
                         plugin=plugin_name, 
                         hook=hook_name, 
                         duration_ms=duration*1000,
                         timeout_ms=timeout*1000)
            
            # Disable plugin after repeated violations
            if self.violation_counts[plugin_name] > 5:
                self.disabled_plugins.add(plugin_name)
                logger.error("Plugin disabled due to repeated timeouts", plugin=plugin_name)
            
            raise PluginTimeoutError(f"Plugin {plugin_name} hook {hook_name} exceeded timeout")
        finally:
            duration = time.perf_counter() - start_time
            self.metrics[f"{plugin_name}.{hook_name}"].append(duration)
            
            # Log performance warnings
            if duration > timeout * 0.8:  # Warning at 80% of timeout
                logger.warning("Plugin hook approaching timeout", 
                             plugin=plugin_name, 
                             hook=hook_name, 
                             duration_ms=duration*1000,
                             timeout_ms=timeout*1000)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring."""
        return {
            "total_hooks_executed": sum(len(timings) for timings in self.metrics.values()),
            "average_hook_times": {
                hook: sum(timings)/len(timings)*1000 for hook, timings in self.metrics.items() if timings
            },
            "violation_counts": dict(self.violation_counts),
            "disabled_plugins": list(self.disabled_plugins)
        }
    
    def is_plugin_disabled(self, plugin_name: str) -> bool:
        """Check if plugin is disabled due to performance violations."""
        return plugin_name in self.disabled_plugins

# ================================================================================
# Hook Management System - Gemini CLI Enhanced Design
# ================================================================================

class HookManager:
    """
    Manages plugin hooks with selective subscription and error isolation.
    Gemini CLI recommendation: Map events to subscribed plugins only.
    """
    
    def __init__(self, performance_monitor: PluginPerformanceMonitor):
        self.hooks = defaultdict(list)  # hook_name -> [(plugin, method)]
        self.performance_monitor = performance_monitor
        
    def register_hook(self, hook_name: str, plugin: OrchestratorPlugin, method: Callable):
        """Register plugin method for specific hook."""
        if not self.performance_monitor.is_plugin_disabled(plugin.plugin_name):
            self.hooks[hook_name].append((plugin, method))
            logger.debug("Hook registered", hook=hook_name, plugin=plugin.plugin_name)
    
    async def trigger_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """
        Trigger all registered methods for hook with error isolation.
        Gemini CLI recommendation: Don't let one plugin failure stop others.
        """
        if hook_name not in self.hooks:
            return []
        
        results = []
        for plugin, method in self.hooks[hook_name]:
            if self.performance_monitor.is_plugin_disabled(plugin.plugin_name):
                continue
                
            try:
                async with self.performance_monitor.time_hook(
                    plugin.plugin_name, 
                    hook_name, 
                    plugin.hook_timeout_ms
                ):
                    result = await method(*args, **kwargs)
                    results.append(result)
            except PluginTimeoutError:
                # Already logged in performance monitor
                continue
            except Exception as e:
                logger.error("Plugin hook error", 
                           plugin=plugin.plugin_name, 
                           hook=hook_name, 
                           error=str(e))
                # Don't re-raise - isolate plugin failures
                continue
        
        return results

# ================================================================================
# Plugin Dependency Management - Gemini CLI Topological Sort
# ================================================================================

class PluginManager:
    """
    Manages plugin lifecycle with dependency resolution.
    Gemini CLI recommendation: Topological sort for dependency ordering.
    """
    
    def __init__(self, state_manager: PluginStateManager, performance_monitor: PluginPerformanceMonitor):
        self.plugins = {}  # plugin_name -> {"class": class, "instance": instance}
        self.state_manager = state_manager
        self.performance_monitor = performance_monitor
        self.sorted_plugins = []
    
    def register_plugin(self, plugin_class: type):
        """Register a plugin class for loading."""
        self.plugins[plugin_class.plugin_name] = {
            "class": plugin_class,
            "instance": None
        }
        logger.debug("Plugin registered", plugin=plugin_class.plugin_name)
    
    async def load_plugins(self, orchestrator: 'OrchestratorV2') -> None:
        """Load and initialize plugins in dependency order."""
        try:
            # Gemini CLI recommendation: Topological sort for dependencies
            self.sorted_plugins = self._topological_sort()
            logger.info("Plugin load order determined", 
                       order=[p[0] for p in self.sorted_plugins])
        except (CircularDependencyError, MissingDependencyError) as e:
            logger.error("Plugin dependency error", error=str(e))
            raise
        
        # Initialize plugins in dependency order
        for plugin_name, plugin_data in self.sorted_plugins:
            try:
                plugin_class = plugin_data["class"]
                instance = plugin_class(self.state_manager, self.performance_monitor)
                await instance.initialize(orchestrator)
                self.plugins[plugin_name]["instance"] = instance
                logger.info("Plugin initialized", plugin=plugin_name)
            except Exception as e:
                logger.error("Plugin initialization failed", plugin=plugin_name, error=str(e))
                # Don't stop other plugins from loading
                continue
    
    def _topological_sort(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Perform topological sort of plugins based on dependencies."""
        # Build dependency graph
        graph = {
            name: set(data["class"].dependencies) 
            for name, data in self.plugins.items()
        }
        
        # Calculate in-degrees
        in_degree = {name: 0 for name in graph}
        for name in graph:
            for dep in graph[name]:
                if dep not in in_degree:
                    raise MissingDependencyError(
                        f"Plugin '{name}' has missing dependency: '{dep}'"
                    )
                in_degree[dep] += 1
        
        # Kahn's algorithm for topological sort
        queue = [name for name in in_degree if in_degree[name] == 0]
        sorted_list = []
        
        while queue:
            name = queue.pop(0)
            sorted_list.append((name, self.plugins[name]))
            
            for dep_name in in_degree:
                if name in graph[dep_name]:
                    in_degree[dep_name] -= 1
                    if in_degree[dep_name] == 0:
                        queue.append(dep_name)
        
        # Check for circular dependencies
        if len(sorted_list) != len(self.plugins):
            raise CircularDependencyError(
                "Circular dependency detected among plugins"
            )
        
        return sorted_list
    
    def get_plugin(self, plugin_name: str) -> Optional[OrchestratorPlugin]:
        """Get plugin instance by name."""
        return self.plugins.get(plugin_name, {}).get("instance")
    
    def get_all_plugins(self) -> Dict[str, OrchestratorPlugin]:
        """Get all plugin instances."""
        return {
            name: data.get("instance") 
            for name, data in self.plugins.items() 
            if data.get("instance") is not None
        }

# ================================================================================
# OrchestratorV2 - Core Kernel Implementation
# ================================================================================

@dataclass
class OrchestratorConfig:
    """Configuration for OrchestratorV2."""
    max_concurrent_agents: int = 50
    agent_spawn_timeout_ms: int = 100
    task_delegation_timeout_ms: int = 500
    health_check_interval_seconds: int = 30
    performance_monitoring_enabled: bool = True
    plugin_performance_budget_ms: int = 100
    circuit_breaker_enabled: bool = True

class OrchestratorV2:
    """
    Unified orchestrator kernel consolidating 35+ legacy implementations.
    
    Core responsibilities (20% of functionality):
    - Agent lifecycle management
    - Task delegation and routing  
    - Communication coordination
    - Health monitoring
    - Performance metrics
    
    Specialized behaviors (80% of functionality) handled by plugins:
    - Production SLA monitoring
    - Performance optimization
    - Development debugging
    - Automation and self-healing
    - Security and compliance
    """
    
    def __init__(self, config: OrchestratorConfig, plugins: List[type]):
        self.config = config
        
        # Core state management
        self.active_agents: Dict[str, AgentInstance] = {}
        self.task_executions: Dict[str, TaskExecution] = {}
        self.agent_task_assignments: Dict[str, Set[str]] = defaultdict(set)
        
        # Component initialization
        self.state_manager = PluginStateManager()
        self.performance_monitor = PluginPerformanceMonitor(config.plugin_performance_budget_ms)
        self.hook_manager = HookManager(self.performance_monitor)
        self.plugin_manager = PluginManager(self.state_manager, self.performance_monitor)
        
        # Communication and circuit breaker
        self.communication_manager: Optional[UnifiedCommunicationManager] = None
        self.circuit_breaker = UnifiedCircuitBreaker(
            CircuitBreakerConfig.for_orchestrator("orchestrator_v2")
        ) if config.circuit_breaker_enabled else None
        
        # Register plugins
        for plugin_class in plugins:
            self.plugin_manager.register_plugin(plugin_class)
        
        # Runtime state
        self._running = False
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> None:
        """Initialize orchestrator and all components."""
        logger.info("Initializing OrchestratorV2", 
                   max_agents=self.config.max_concurrent_agents,
                   plugins_count=len(self.plugin_manager.plugins))
        
        # Initialize communication
        self.communication_manager = await get_communication_manager()
        
        # Load plugins in dependency order
        await self.plugin_manager.load_plugins(self)
        
        # Start background tasks
        self._running = True
        if self.config.health_check_interval_seconds > 0:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("OrchestratorV2 initialized successfully")
    
    async def spawn_agent(self, role: AgentRole, capabilities: List[str] = None) -> str:
        """
        Spawn new agent with plugin hooks and circuit breaker protection.
        Target: <100ms per agent spawn.
        """
        start_time = time.perf_counter()
        
        if len(self.active_agents) >= self.config.max_concurrent_agents:
            raise ValueError(f"Maximum concurrent agents ({self.config.max_concurrent_agents}) exceeded")
        
        agent_id = str(uuid.uuid4())
        agent_config = {
            "id": agent_id,
            "role": role,
            "capabilities": capabilities or []
        }
        
        try:
            # Circuit breaker protection
            if self.circuit_breaker:
                async with self.circuit_breaker:
                    return await self._spawn_agent_impl(role, agent_id, agent_config, start_time)
            else:
                return await self._spawn_agent_impl(role, agent_id, agent_config, start_time)
                
        except Exception as e:
            logger.error("Agent spawn failed", agent_id=agent_id, role=role.value, error=str(e))
            # Cleanup on failure
            if agent_id in self.active_agents:
                del self.active_agents[agent_id]
            raise
    
    async def _spawn_agent_impl(self, role: AgentRole, agent_id: str, agent_config: Dict[str, Any], start_time: float) -> str:
        """Implementation of agent spawning."""
        # Trigger before_agent_spawn hooks
        hook_results = await self.hook_manager.trigger_hook("before_agent_spawn", agent_config)
        
        # Apply plugin modifications to config
        for result in hook_results:
            if isinstance(result, dict):
                agent_config.update(result)
        
        # Create agent instance
        agent = AgentInstance(
            id=agent_id,
            role=role,
            status=AgentStatus.INITIALIZING,
            capabilities=agent_config["capabilities"]
        )
        
        # Register agent
        self.active_agents[agent_id] = agent
        
        # Update status to active
        agent.status = AgentStatus.ACTIVE
        
        # Trigger after_agent_spawn hooks
        await self.hook_manager.trigger_hook("after_agent_spawn", agent)
        
        # Record performance metric
        spawn_time = (time.perf_counter() - start_time) * 1000
        await self._record_performance_metric("agent_spawn_time_ms", spawn_time, {
            "agent_id": agent_id,
            "role": role.value
        })
        
        logger.info("Agent spawned", 
                  agent_id=agent_id, 
                  role=role.value, 
                  spawn_time_ms=spawn_time)
        
        return agent_id
    
    async def delegate_task(self, task: Task, agent_id: Optional[str] = None) -> str:
        """
        Delegate task to agent with plugin hooks and routing.
        Target: <500ms for complex routing decisions.
        """
        start_time = time.perf_counter()
        
        try:
            if self.circuit_breaker:
                async with self.circuit_breaker:
                    return await self._delegate_task_impl(task, agent_id, start_time)
            else:
                return await self._delegate_task_impl(task, agent_id, start_time)
                
        except Exception as e:
            logger.error("Task delegation failed", task_id=task.id, error=str(e))
            # Cleanup on failure
            if task.id in self.task_executions:
                del self.task_executions[task.id]
            if agent_id and task.id in self.agent_task_assignments[agent_id]:
                self.agent_task_assignments[agent_id].remove(task.id)
            raise
    
    async def _delegate_task_impl(self, task: Task, agent_id: Optional[str], start_time: float) -> str:
        """Implementation of task delegation."""
        # Trigger before_task_delegate hooks (allow task modification)
        hook_results = await self.hook_manager.trigger_hook("before_task_delegate", task)
        
        # Apply plugin modifications to task
        for result in hook_results:
            if isinstance(result, Task):
                task = result
        
        # Agent selection logic
        if agent_id is None:
            agent_id = await self._select_optimal_agent(task)
        elif agent_id not in self.active_agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Create task execution
        execution = TaskExecution(
            task_id=task.id,
            agent_id=agent_id,
            state=TaskExecutionState.ASSIGNED,
            started_at=datetime.utcnow()
        )
        
        self.task_executions[task.id] = execution
        self.agent_task_assignments[agent_id].add(task.id)
        
        # Update agent status
        agent = self.active_agents[agent_id]
        agent.current_task_id = task.id
        agent.status = AgentStatus.BUSY
        
        # Send task to agent via communication manager
        message_success = await self._send_task_to_agent(task, agent_id)
        if not message_success:
            raise RuntimeError(f"Failed to send task {task.id} to agent {agent_id}")
        
        # Update execution state
        execution.state = TaskExecutionState.RUNNING
        
        # Trigger after_task_delegate hooks
        await self.hook_manager.trigger_hook("after_task_delegate", task, agent_id)
        
        # Record performance metric
        delegation_time = (time.perf_counter() - start_time) * 1000
        await self._record_performance_metric("task_delegation_time_ms", delegation_time, {
            "task_id": task.id,
            "agent_id": agent_id,
            "task_type": task.type
        })
        
        logger.info("Task delegated", 
                  task_id=task.id, 
                  agent_id=agent_id, 
                  delegation_time_ms=delegation_time)
        
        return task.id
    
    async def _select_optimal_agent(self, task: Task) -> str:
        """Select optimal agent for task based on capabilities and load."""
        available_agents = [
            agent for agent in self.active_agents.values() 
            if agent.status in [AgentStatus.ACTIVE, AgentStatus.IDLE]
        ]
        
        if not available_agents:
            raise RuntimeError("No available agents for task delegation")
        
        # Simple capability matching and load balancing
        scored_agents = []
        for agent in available_agents:
            # Calculate capability match score
            capability_score = 0
            for requirement in task.requirements:
                if requirement in agent.capabilities:
                    capability_score += 1
            
            # Calculate load score (lower is better)
            current_tasks = len(self.agent_task_assignments[agent.id])
            load_score = 1.0 / (current_tasks + 1)
            
            # Combined score
            total_score = capability_score + load_score
            scored_agents.append((agent.id, total_score))
        
        # Select agent with highest score
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        return scored_agents[0][0]
    
    async def _send_task_to_agent(self, task: Task, agent_id: str) -> bool:
        """Send task to agent via communication manager."""
        if not self.communication_manager:
            logger.error("Communication manager not initialized")
            return False
        
        try:
            message = StandardUniversalMessage(
                from_agent="orchestrator_v2",
                to_agent=agent_id,
                message_type=MessageType.TASK_REQUEST,
                priority=task.priority,
                payload={
                    "task": {
                        "id": task.id,
                        "type": task.type,
                        "description": task.description,
                        "payload": task.payload,
                        "timeout_seconds": task.timeout_seconds
                    }
                }
            )
            
            return await self.communication_manager.send_message(message)
            
        except Exception as e:
            logger.error("Failed to send task message", 
                        task_id=task.id, agent_id=agent_id, error=str(e))
            return False
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status with plugin contributions."""
        base_health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "active_agents": len(self.active_agents),
            "running_tasks": len([
                exec for exec in self.task_executions.values() 
                if exec.state == TaskExecutionState.RUNNING
            ]),
            "performance": self.performance_monitor.get_performance_summary(),
            "circuit_breaker": {
                "state": self.circuit_breaker.state.value if self.circuit_breaker else "disabled",
                "failure_count": getattr(self.circuit_breaker, 'failure_count', 0) if self.circuit_breaker else 0
            }
        }
        
        # Allow plugins to contribute to health status
        plugin_health_data = await self.hook_manager.trigger_hook("on_health_check", base_health)
        
        # Merge plugin contributions
        for plugin_data in plugin_health_data:
            if isinstance(plugin_data, dict):
                base_health.update(plugin_data)
        
        return base_health
    
    async def _record_performance_metric(self, metric_name: str, value: float, metadata: Dict[str, Any]):
        """Record performance metric and notify plugins."""
        await self.hook_manager.trigger_hook("on_performance_metric", metric_name, value, metadata)
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while self._running:
            try:
                health_status = await self.get_health_status()
                logger.debug("Health check completed", 
                           active_agents=health_status["active_agents"],
                           running_tasks=health_status["running_tasks"])
            except Exception as e:
                logger.error("Health check failed", error=str(e))
            
            await asyncio.sleep(self.config.health_check_interval_seconds)
    
    async def shutdown(self, graceful: bool = True) -> None:
        """Shutdown orchestrator and all components."""
        logger.info("Shutting down OrchestratorV2", graceful=graceful)
        
        self._running = False
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if graceful:
            # Wait for running tasks to complete (with timeout)
            running_tasks = [
                exec for exec in self.task_executions.values() 
                if exec.state == TaskExecutionState.RUNNING
            ]
            
            if running_tasks:
                logger.info("Waiting for running tasks to complete", count=len(running_tasks))
                # TODO: Implement graceful task completion wait
        
        # Shutdown agents
        for agent_id in list(self.active_agents.keys()):
            try:
                await self.shutdown_agent(agent_id, graceful=graceful)
            except Exception as e:
                logger.error("Error shutting down agent", agent_id=agent_id, error=str(e))
        
        logger.info("OrchestratorV2 shutdown complete")
    
    async def shutdown_agent(self, agent_id: str, graceful: bool = True) -> bool:
        """Shutdown specific agent."""
        if agent_id not in self.active_agents:
            return False
        
        agent = self.active_agents[agent_id]
        agent.status = AgentStatus.SHUTTING_DOWN
        
        if graceful and agent.current_task_id:
            # Wait for current task to complete
            # TODO: Implement graceful task completion
            pass
        
        # Cleanup agent state
        del self.active_agents[agent_id]
        if agent_id in self.agent_task_assignments:
            del self.agent_task_assignments[agent_id]
        
        logger.info("Agent shutdown", agent_id=agent_id, graceful=graceful)
        return True