"""
Core Plugin Development Interfaces for LeanVibe SDK.

Provides base classes and interfaces that all plugins must implement.
Designed for developer ease-of-use while maintaining performance.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable, Protocol
from dataclasses import dataclass
from enum import Enum

from .models import TaskResult, CoordinationResult, PluginConfig, PluginContext, PluginEvent
from .exceptions import PluginSDKError, PluginExecutionError


class PluginType(Enum):
    """Types of plugins supported by the SDK."""
    WORKFLOW = "workflow"
    MONITORING = "monitoring" 
    SECURITY = "security"
    INTEGRATION = "integration"
    PRODUCTIVITY = "productivity"
    COMMUNICATION = "communication"
    ANALYTICS = "analytics"
    AUTOMATION = "automation"


class PluginStatus(Enum):
    """Plugin execution status."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentInterface(Protocol):
    """
    Interface for interacting with agents in the system.
    Provides a simplified view of agent capabilities for plugin developers.
    """
    
    @property
    def agent_id(self) -> str:
        """Unique identifier for the agent."""
        ...
    
    @property
    def capabilities(self) -> List[str]:
        """List of agent capabilities."""
        ...
    
    @property
    def status(self) -> str:
        """Current agent status."""
        ...
    
    async def execute_task(self, task: Dict[str, Any]) -> TaskResult:
        """Execute a task using this agent."""
        ...
    
    async def get_context(self) -> Dict[str, Any]:
        """Get current agent context."""
        ...
    
    async def send_message(self, message: str, target: Optional[str] = None) -> bool:
        """Send a message to another agent or broadcast."""
        ...


class TaskInterface(Protocol):
    """
    Interface for task operations within plugins.
    Simplifies task management and execution for plugin developers.
    """
    
    @property
    def task_id(self) -> str:
        """Unique identifier for the task."""
        ...
    
    @property
    def task_type(self) -> str:
        """Type of the task."""
        ...
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """Task parameters and configuration."""
        ...
    
    @property
    def priority(self) -> int:
        """Task priority level."""
        ...
    
    async def update_status(self, status: str, progress: float = 0.0) -> None:
        """Update task execution status."""
        ...
    
    async def add_result(self, key: str, value: Any) -> None:
        """Add a result to the task."""
        ...
    
    async def get_dependency_results(self) -> Dict[str, Any]:
        """Get results from dependent tasks."""
        ...


class OrchestratorInterface(Protocol):
    """
    Interface for interacting with the orchestrator.
    Provides high-level orchestration capabilities to plugins.
    """
    
    async def get_agents(self, filters: Optional[Dict[str, Any]] = None) -> List[AgentInterface]:
        """Get available agents with optional filtering."""
        ...
    
    async def create_task(self, task_type: str, parameters: Dict[str, Any]) -> TaskInterface:
        """Create a new task in the system."""
        ...
    
    async def schedule_task(self, task: TaskInterface, delay_seconds: int = 0) -> str:
        """Schedule a task for execution."""
        ...
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        ...
    
    async def broadcast_event(self, event: PluginEvent) -> None:
        """Broadcast an event to all interested components."""
        ...


class MonitoringInterface(Protocol):
    """
    Interface for monitoring and observability within plugins.
    Provides easy access to metrics, logging, and alerting.
    """
    
    async def log_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Log a custom metric."""
        ...
    
    async def log_event(self, event: PluginEvent) -> None:
        """Log a plugin event."""
        ...
    
    async def create_alert(self, message: str, severity: str = "info") -> None:
        """Create an alert/notification."""
        ...
    
    async def get_performance_data(self, plugin_id: str) -> Dict[str, Any]:
        """Get performance data for a plugin."""
        ...


class CoordinationInterface(Protocol):
    """
    Interface for multi-agent coordination within plugins.
    Simplifies complex coordination patterns for plugin developers.
    """
    
    async def coordinate_agents(self, agents: List[AgentInterface], strategy: str = "parallel") -> CoordinationResult:
        """Coordinate multiple agents with a specified strategy."""
        ...
    
    async def create_agent_group(self, agent_ids: List[str], group_name: str) -> str:
        """Create a coordination group of agents."""
        ...
    
    async def sync_agents(self, group_id: str, timeout_seconds: int = 30) -> bool:
        """Synchronize agents in a coordination group."""
        ...
    
    async def distribute_work(self, work_items: List[Dict[str, Any]], agent_group: str) -> List[TaskResult]:
        """Distribute work items across a group of agents."""
        ...


class SecurityInterface(Protocol):
    """
    Interface for security operations within plugins.
    Provides secure access to sensitive operations.
    """
    
    async def validate_permissions(self, operation: str, context: Dict[str, Any]) -> bool:
        """Validate permissions for a specific operation."""
        ...
    
    async def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        ...
    
    async def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt previously encrypted data."""
        ...
    
    async def audit_log(self, action: str, details: Dict[str, Any]) -> None:
        """Log security-relevant actions."""
        ...


class PluginBase(ABC):
    """
    Base class for all LeanVibe plugins.
    
    Provides a comprehensive foundation for plugin development with:
    - Lifecycle management
    - Error handling
    - Performance monitoring
    - Integration with core interfaces
    
    Epic 1 Optimizations:
    - Minimal memory footprint
    - Fast initialization (<10ms)
    - Efficient event handling
    """
    
    def __init__(self, config: PluginConfig):
        """
        Initialize the plugin.
        
        Args:
            config: Plugin configuration and metadata
        """
        self.config = config
        self.plugin_id = config.plugin_id or f"plugin_{uuid.uuid4().hex[:8]}"
        self.status = PluginStatus.INITIALIZED
        self.created_at = datetime.utcnow()
        self.last_executed = None
        
        # Performance tracking
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._error_count = 0
        
        # Context and state
        self._context: Optional[PluginContext] = None
        self._orchestrator: Optional[OrchestratorInterface] = None
        self._monitoring: Optional[MonitoringInterface] = None
        self._coordination: Optional[CoordinationInterface] = None
        self._security: Optional[SecurityInterface] = None
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}
    
    @property
    def name(self) -> str:
        """Plugin name."""
        return self.config.name
    
    @property
    def version(self) -> str:
        """Plugin version."""
        return self.config.version
    
    @property
    def plugin_type(self) -> PluginType:
        """Plugin type."""
        return self.config.plugin_type
    
    @property
    def is_running(self) -> bool:
        """Check if plugin is currently running."""
        return self.status == PluginStatus.RUNNING
    
    @property
    def performance_metrics(self) -> Dict[str, Any]:
        """Get plugin performance metrics."""
        avg_execution_time = (
            self._total_execution_time / self._execution_count 
            if self._execution_count > 0 else 0
        )
        
        return {
            "execution_count": self._execution_count,
            "total_execution_time_ms": round(self._total_execution_time, 2),
            "average_execution_time_ms": round(avg_execution_time, 2),
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._execution_count, 1),
            "uptime_seconds": (datetime.utcnow() - self.created_at).total_seconds(),
            "last_executed": self.last_executed.isoformat() if self.last_executed else None
        }
    
    async def initialize(self, orchestrator: OrchestratorInterface) -> bool:
        """
        Initialize the plugin with orchestrator interfaces.
        
        Args:
            orchestrator: Main orchestrator interface
            
        Returns:
            bool: True if initialization successful
        """
        try:
            self._orchestrator = orchestrator
            
            # Initialize context
            self._context = PluginContext(
                plugin_id=self.plugin_id,
                plugin_name=self.name,
                orchestrator=orchestrator
            )
            
            # Call plugin-specific initialization
            await self._on_initialize()
            
            self.status = PluginStatus.INITIALIZED
            return True
            
        except Exception as e:
            self.status = PluginStatus.FAILED
            self._error_count += 1
            raise PluginExecutionError(f"Plugin initialization failed: {str(e)}")
    
    async def execute(self, task: TaskInterface) -> TaskResult:
        """
        Execute the plugin with a given task.
        
        Args:
            task: Task to execute
            
        Returns:
            TaskResult: Execution result
        """
        start_time = datetime.utcnow()
        self.status = PluginStatus.RUNNING
        self.last_executed = start_time
        
        try:
            # Pre-execution hooks
            await self._before_execute(task)
            
            # Main execution
            result = await self.handle_task(task)
            
            # Post-execution hooks
            await self._after_execute(task, result)
            
            # Update metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._execution_count += 1
            self._total_execution_time += execution_time
            
            self.status = PluginStatus.COMPLETED
            return result
            
        except Exception as e:
            self.status = PluginStatus.FAILED
            self._error_count += 1
            
            error_result = TaskResult(
                success=False,
                error=str(e),
                plugin_id=self.plugin_id,
                task_id=task.task_id
            )
            
            await self._on_error(task, e)
            return error_result
    
    async def coordinate_agents(self, agents: List[AgentInterface]) -> CoordinationResult:
        """
        Coordinate multiple agents (default implementation).
        
        Args:
            agents: List of agents to coordinate
            
        Returns:
            CoordinationResult: Coordination result
        """
        if not self._coordination:
            raise PluginSDKError("Coordination interface not available")
        
        return await self._coordination.coordinate_agents(agents)
    
    async def cleanup(self) -> bool:
        """
        Cleanup plugin resources.
        
        Returns:
            bool: True if cleanup successful
        """
        try:
            await self._on_cleanup()
            self.status = PluginStatus.COMPLETED
            return True
        except Exception as e:
            self._error_count += 1
            raise PluginExecutionError(f"Plugin cleanup failed: {str(e)}")
    
    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """
        Register an event handler for specific event types.
        
        Args:
            event_type: Type of event to handle
            handler: Handler function
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    async def emit_event(self, event: PluginEvent) -> None:
        """
        Emit an event to registered handlers.
        
        Args:
            event: Event to emit
        """
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                # Log error but don't fail the event emission
                if self._monitoring:
                    await self._monitoring.log_event(PluginEvent(
                        event_type="error",
                        data={"error": str(e), "handler": str(handler)},
                        plugin_id=self.plugin_id
                    ))
    
    # Abstract methods that plugins must implement
    
    @abstractmethod
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """
        Handle a task execution (must be implemented by plugins).
        
        Args:
            task: Task to handle
            
        Returns:
            TaskResult: Task execution result
        """
        pass
    
    # Optional lifecycle hooks
    
    async def _on_initialize(self) -> None:
        """Called during plugin initialization (override in subclass)."""
        pass
    
    async def _before_execute(self, task: TaskInterface) -> None:
        """Called before task execution (override in subclass)."""
        pass
    
    async def _after_execute(self, task: TaskInterface, result: TaskResult) -> None:
        """Called after task execution (override in subclass)."""
        pass
    
    async def _on_error(self, task: TaskInterface, error: Exception) -> None:
        """Called when an error occurs (override in subclass)."""
        pass
    
    async def _on_cleanup(self) -> None:
        """Called during cleanup (override in subclass)."""
        pass
    
    # Utility methods for plugin developers
    
    async def log_info(self, message: str, **kwargs) -> None:
        """Log an info message."""
        if self._monitoring:
            event = PluginEvent(
                event_type="info",
                data={"message": message, **kwargs},
                plugin_id=self.plugin_id
            )
            await self._monitoring.log_event(event)
    
    async def log_error(self, message: str, **kwargs) -> None:
        """Log an error message."""
        if self._monitoring:
            event = PluginEvent(
                event_type="error", 
                data={"message": message, **kwargs},
                plugin_id=self.plugin_id
            )
            await self._monitoring.log_event(event)
    
    async def create_alert(self, message: str, severity: str = "info") -> None:
        """Create an alert."""
        if self._monitoring:
            await self._monitoring.create_alert(message, severity)
    
    async def get_available_agents(self, capabilities: Optional[List[str]] = None) -> List[AgentInterface]:
        """Get available agents with optional capability filtering."""
        if not self._orchestrator:
            return []
        
        filters = {"capabilities": capabilities} if capabilities else None
        return await self._orchestrator.get_agents(filters)
    
    async def create_subtask(self, task_type: str, parameters: Dict[str, Any]) -> TaskInterface:
        """Create a subtask."""
        if not self._orchestrator:
            raise PluginSDKError("Orchestrator interface not available")
        
        return await self._orchestrator.create_task(task_type, parameters)


# Convenience base classes for specific plugin types

class WorkflowPlugin(PluginBase):
    """Base class for workflow plugins."""
    
    def __init__(self, config: PluginConfig):
        if not config.plugin_type:
            config.plugin_type = PluginType.WORKFLOW
        super().__init__(config)


class MonitoringPlugin(PluginBase):
    """Base class for monitoring plugins."""
    
    def __init__(self, config: PluginConfig):
        if not config.plugin_type:
            config.plugin_type = PluginType.MONITORING
        super().__init__(config)


class SecurityPlugin(PluginBase):
    """Base class for security plugins."""
    
    def __init__(self, config: PluginConfig):
        if not config.plugin_type:
            config.plugin_type = PluginType.SECURITY
        super().__init__(config)


class IntegrationPlugin(PluginBase):
    """Base class for integration plugins."""
    
    def __init__(self, config: PluginConfig):
        if not config.plugin_type:
            config.plugin_type = PluginType.INTEGRATION
        super().__init__(config)