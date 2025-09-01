"""
Common Orchestrator Interfaces and Patterns
Epic 1 Phase 1.1 - Interface extraction for ConsolidatedProductionOrchestrator

This module defines the common interfaces, patterns, and protocols found across
all orchestrator implementations in the system. These serve as the foundation
for the ConsolidatedProductionOrchestrator implementation.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, List, Optional, Any, Union, Callable, AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid


# Core Enums (extracted from existing implementations)

class OrchestratorMode(str, Enum):
    """Orchestrator operational modes."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    HIGH_AVAILABILITY = "high_availability"


class HealthStatus(str, Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    NO_AGENTS = "no_agents"


class ScalingAction(str, Enum):
    """Auto-scaling action types."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    EMERGENCY_SCALE = "emergency_scale"


# Core Data Models

@dataclass
class OrchestratorConfig:
    """Unified configuration for all orchestrator implementations."""
    # Basic configuration
    max_agents: int = 50
    task_timeout: int = 300
    mode: OrchestratorMode = OrchestratorMode.PRODUCTION
    
    # Plugin configuration
    plugin_dir: str = "app/core/orchestrator_plugins"
    enable_plugins: bool = True
    
    # Feature flags
    enable_advanced_features: bool = True
    enable_redis_bridge: bool = True
    enable_tmux_integration: bool = True
    enable_monitoring: bool = True
    enable_auto_scaling: bool = False
    
    # Performance configuration
    response_time_target_ms: int = 50
    memory_limit_mb: int = 100
    
    # Production features
    enable_sla_monitoring: bool = False
    enable_anomaly_detection: bool = False
    enable_security_monitoring: bool = False
    enable_disaster_recovery: bool = False


@dataclass
class AgentSpec:
    """Standardized agent specification across orchestrators."""
    role: str
    agent_type: str = "claude_code"
    workspace_name: Optional[str] = None
    git_branch: Optional[str] = None
    environment_vars: Dict[str, str] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpec:
    """Standardized task specification across orchestrators."""
    description: str
    task_type: str = "general"
    priority: str = "medium"
    preferred_agent_role: Optional[str] = None
    estimated_duration_seconds: Optional[int] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStatus:
    """Standardized agent status information."""
    id: str
    role: str
    status: str
    created_at: datetime
    last_activity: Optional[datetime]
    current_task_id: Optional[str]
    health: str
    session_info: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Standardized task result information."""
    id: str
    description: str
    task_type: str
    priority: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime]
    assigned_agent_id: Optional[str]
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_info: Optional[Dict[str, Any]] = None


@dataclass
class SystemHealth:
    """Comprehensive system health information."""
    status: HealthStatus
    timestamp: datetime
    version: str
    uptime_seconds: float
    orchestrator_type: str
    
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)


# Core Protocols

class BaseOrchestratorProtocol(Protocol):
    """Core protocol that all orchestrators must implement."""
    
    async def initialize(self) -> None:
        """Initialize the orchestrator and its components."""
        ...
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator."""
        ...
    
    async def health_check(self) -> SystemHealth:
        """Get comprehensive system health information."""
        ...


class AgentManagementProtocol(Protocol):
    """Protocol for agent lifecycle management."""
    
    async def register_agent(self, agent_spec: AgentSpec) -> str:
        """Register a new agent and return its ID."""
        ...
    
    async def shutdown_agent(self, agent_id: str, graceful: bool = True) -> bool:
        """Shutdown a specific agent."""
        ...
    
    async def get_agent_status(self, agent_id: str) -> AgentStatus:
        """Get detailed status of a specific agent."""
        ...
    
    async def list_agents(self) -> List[AgentStatus]:
        """List all registered agents with their status."""
        ...


class TaskOrchestrationProtocol(Protocol):
    """Protocol for task orchestration capabilities."""
    
    async def delegate_task(self, task: TaskSpec) -> TaskResult:
        """Delegate a task to an appropriate agent."""
        ...
    
    async def get_task_status(self, task_id: str) -> TaskResult:
        """Get status of a specific task."""
        ...
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        ...
    
    async def list_tasks(self, filter_criteria: Optional[Dict[str, Any]] = None) -> List[TaskResult]:
        """List tasks with optional filtering."""
        ...


class WorkflowOrchestrationProtocol(Protocol):
    """Protocol for advanced workflow orchestration."""
    
    async def execute_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Execute a complex workflow and return workflow ID."""
        ...
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a running workflow."""
        ...
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        ...


class MonitoringProtocol(Protocol):
    """Protocol for monitoring and observability."""
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        ...
    
    async def get_agent_metrics(self, agent_id: str) -> Dict[str, Any]:
        """Get metrics for a specific agent."""
        ...
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics and benchmarks."""
        ...


class ScalingProtocol(Protocol):
    """Protocol for auto-scaling capabilities."""
    
    async def scale_agents(self, target_count: int, agent_role: Optional[str] = None) -> Dict[str, Any]:
        """Scale agent pool to target count."""
        ...
    
    async def auto_scale_check(self) -> ScalingAction:
        """Check if auto-scaling action is needed."""
        ...
    
    async def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get metrics relevant to scaling decisions."""
        ...


class PluginProtocol(Protocol):
    """Protocol for plugin system integration."""
    
    async def load_plugin(self, plugin_name: str, plugin_config: Optional[Dict[str, Any]] = None) -> bool:
        """Load and initialize a plugin."""
        ...
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin."""
        ...
    
    async def list_plugins(self) -> List[Dict[str, Any]]:
        """List all loaded plugins with their status."""
        ...
    
    async def plugin_health_check(self, plugin_name: str) -> Dict[str, Any]:
        """Check health of a specific plugin."""
        ...


# Comprehensive Orchestrator Interface

class ConsolidatedOrchestratorProtocol(
    BaseOrchestratorProtocol,
    AgentManagementProtocol,
    TaskOrchestrationProtocol,
    WorkflowOrchestrationProtocol,
    MonitoringProtocol,
    ScalingProtocol,
    PluginProtocol,
    Protocol
):
    """
    Comprehensive protocol combining all orchestrator capabilities.
    This is the interface that ConsolidatedProductionOrchestrator will implement.
    """
    
    # Additional consolidated methods
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status (alias for health_check)."""
        ...
    
    async def handle_emergency(self, emergency_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emergency situations with appropriate response."""
        ...
    
    async def backup_state(self) -> str:
        """Create backup of orchestrator state and return backup ID."""
        ...
    
    async def restore_state(self, backup_id: str) -> bool:
        """Restore orchestrator state from backup."""
        ...


# Abstract Base Classes

class BaseOrchestrator(ABC):
    """
    Abstract base class providing common orchestrator functionality.
    Contains shared implementation patterns found across orchestrators.
    """
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._initialized = False
        self._start_time = datetime.utcnow()
        self._operations_count = 0
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the orchestrator - must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the orchestrator - must be implemented by subclasses."""
        pass
    
    def _get_uptime_seconds(self) -> float:
        """Calculate orchestrator uptime."""
        return (datetime.utcnow() - self._start_time).total_seconds()
    
    def _increment_operations(self) -> None:
        """Increment operations counter."""
        self._operations_count += 1
    
    def _generate_id(self) -> str:
        """Generate unique ID."""
        return str(uuid.uuid4())


# Event System

@dataclass
class OrchestratorEvent:
    """Event emitted by orchestrator for monitoring/plugins."""
    event_type: str
    timestamp: datetime
    source_component: str
    data: Dict[str, Any]
    severity: str = "info"


class EventHandler(Protocol):
    """Protocol for orchestrator event handlers."""
    
    async def handle_event(self, event: OrchestratorEvent) -> None:
        """Handle an orchestrator event."""
        ...


# Plugin System Base

class OrchestratorPlugin(ABC):
    """
    Base class for orchestrator plugins.
    Extracted from existing plugin implementations.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self._initialized = False
    
    @abstractmethod
    async def initialize(self, orchestrator: ConsolidatedOrchestratorProtocol) -> None:
        """Initialize the plugin with orchestrator reference."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin health."""
        pass
    
    async def handle_event(self, event: OrchestratorEvent) -> None:
        """Handle orchestrator events (optional)."""
        pass


# Factory and Builder Patterns

class OrchestratorFactory:
    """Factory for creating orchestrator instances."""
    
    @staticmethod
    def create_orchestrator(
        orchestrator_type: str = "consolidated",
        config: Optional[OrchestratorConfig] = None
    ) -> ConsolidatedOrchestratorProtocol:
        """Create orchestrator instance based on type."""
        # Implementation will be in the actual orchestrator file
        pass
    
    @staticmethod
    def create_config(
        mode: OrchestratorMode = OrchestratorMode.PRODUCTION,
        **kwargs
    ) -> OrchestratorConfig:
        """Create orchestrator configuration."""
        return OrchestratorConfig(mode=mode, **kwargs)


# Migration and Compatibility

@dataclass
class MigrationResult:
    """Result of orchestrator migration."""
    success: bool
    migrated_agents: int
    migrated_tasks: int
    errors: List[str]
    warnings: List[str]
    duration_seconds: float


class MigrationProtocol(Protocol):
    """Protocol for orchestrator migration capabilities."""
    
    async def migrate_from(
        self, 
        source_orchestrator: Any, 
        migration_options: Optional[Dict[str, Any]] = None
    ) -> MigrationResult:
        """Migrate data from another orchestrator."""
        ...
    
    async def export_state(self) -> Dict[str, Any]:
        """Export orchestrator state for migration."""
        ...
    
    async def import_state(self, state: Dict[str, Any]) -> bool:
        """Import orchestrator state from export."""
        ...