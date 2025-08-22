"""
Data Models for LeanVibe Plugin SDK.

Provides comprehensive data structures for plugin development
with type safety and serialization support.
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import json


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CoordinationStrategy(Enum):
    """Agent coordination strategies."""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    PIPELINE = "pipeline"
    CONSENSUS = "consensus"
    LEADER_FOLLOWER = "leader_follower"


class EventSeverity(Enum):
    """Event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PluginConfig:
    """
    Plugin configuration and metadata.
    
    Contains all the information needed to configure and initialize a plugin.
    """
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    plugin_id: Optional[str] = None
    plugin_type: Optional['PluginType'] = None
    
    # Configuration parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    
    # Resource limits
    max_memory_mb: int = 100
    max_execution_time_seconds: int = 300
    max_concurrent_tasks: int = 10
    
    # Security and permissions
    required_permissions: List[str] = field(default_factory=list)
    security_level: str = "standard"
    sandbox_enabled: bool = True
    
    # Epic 1: Performance settings
    performance_optimized: bool = True
    lazy_loading: bool = True
    cache_enabled: bool = True
    
    def __post_init__(self):
        if not self.plugin_id:
            self.plugin_id = f"{self.name}_{uuid.uuid4().hex[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.plugin_type:
            data['plugin_type'] = self.plugin_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginConfig':
        """Create from dictionary."""
        # Handle plugin_type enum
        if 'plugin_type' in data and isinstance(data['plugin_type'], str):
            from .interfaces import PluginType
            data['plugin_type'] = PluginType(data['plugin_type'])
        return cls(**data)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        if not self.name or not self.name.strip():
            errors.append("Plugin name is required")
        
        if not self.version:
            errors.append("Plugin version is required")
        
        if self.max_memory_mb <= 0:
            errors.append("Max memory must be positive")
        
        if self.max_execution_time_seconds <= 0:
            errors.append("Max execution time must be positive")
        
        if self.max_concurrent_tasks <= 0:
            errors.append("Max concurrent tasks must be positive")
        
        return errors


@dataclass
class PluginContext:
    """
    Runtime context for plugin execution.
    
    Provides access to system interfaces and runtime information.
    """
    plugin_id: str
    plugin_name: str
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: datetime = field(default_factory=datetime.utcnow)
    
    # Interface references
    orchestrator: Optional[Any] = None
    monitoring: Optional[Any] = None
    coordination: Optional[Any] = None
    security: Optional[Any] = None
    
    # Runtime data
    variables: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    events: List['PluginEvent'] = field(default_factory=list)
    
    # Performance tracking
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    execution_count: int = 0
    
    def set_variable(self, key: str, value: Any) -> None:
        """Set a context variable."""
        self.variables[key] = value
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.variables.get(key, default)
    
    def record_metric(self, name: str, value: float) -> None:
        """Record a performance metric."""
        self.metrics[name] = value
    
    def add_event(self, event: 'PluginEvent') -> None:
        """Add an event to the context."""
        self.events.append(event)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding interface references)."""
        return {
            "plugin_id": self.plugin_id,
            "plugin_name": self.plugin_name,
            "execution_id": self.execution_id,
            "started_at": self.started_at.isoformat(),
            "variables": self.variables,
            "metrics": self.metrics,
            "events": [event.to_dict() for event in self.events],
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "execution_count": self.execution_count
        }


@dataclass
class TaskResult:
    """
    Result of task execution.
    
    Contains execution outcome, data, and performance metrics.
    """
    success: bool
    plugin_id: str
    task_id: str
    execution_time_ms: float = 0.0
    
    # Result data
    data: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    
    # Error information
    error: Optional[str] = None
    error_code: Optional[str] = None
    error_details: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    memory_used_mb: float = 0.0
    cpu_time_ms: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def set_result(self, key: str, value: Any) -> None:
        """Set a result value."""
        self.data[key] = value
    
    def get_result(self, key: str, default: Any = None) -> Any:
        """Get a result value."""
        return self.data.get(key, default)
    
    def add_artifact(self, artifact_path: str) -> None:
        """Add an artifact path."""
        self.artifacts.append(artifact_path)
    
    def set_error(self, error: str, error_code: str = None, details: Dict[str, Any] = None) -> None:
        """Set error information."""
        self.success = False
        self.error = error
        self.error_code = error_code
        self.error_details = details or {}
        self.completed_at = datetime.utcnow()
    
    def complete(self) -> None:
        """Mark the task as completed."""
        self.completed_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "plugin_id": self.plugin_id,
            "task_id": self.task_id,
            "execution_time_ms": self.execution_time_ms,
            "data": self.data,
            "artifacts": self.artifacts,
            "error": self.error,
            "error_code": self.error_code,
            "error_details": self.error_details,
            "memory_used_mb": self.memory_used_mb,
            "cpu_time_ms": self.cpu_time_ms,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskResult':
        """Create from dictionary."""
        # Convert datetime strings back
        if 'created_at' in data:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'completed_at' in data and data['completed_at']:
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)


@dataclass
class CoordinationResult:
    """
    Result of agent coordination.
    
    Contains coordination outcome and individual agent results.
    """
    success: bool
    coordination_id: str
    strategy: CoordinationStrategy
    total_agents: int
    
    # Agent results
    agent_results: Dict[str, TaskResult] = field(default_factory=dict)
    failed_agents: List[str] = field(default_factory=list)
    
    # Timing information
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    total_time_ms: float = 0.0
    
    # Coordination metrics
    synchronization_points: int = 0
    coordination_overhead_ms: float = 0.0
    
    # Error information
    error: Optional[str] = None
    
    def add_agent_result(self, agent_id: str, result: TaskResult) -> None:
        """Add result for a specific agent."""
        self.agent_results[agent_id] = result
        if not result.success:
            self.failed_agents.append(agent_id)
    
    def get_successful_agents(self) -> List[str]:
        """Get list of agents that completed successfully."""
        return [
            agent_id for agent_id, result in self.agent_results.items() 
            if result.success
        ]
    
    def get_success_rate(self) -> float:
        """Get coordination success rate."""
        if not self.agent_results:
            return 0.0
        successful = len(self.get_successful_agents())
        return successful / len(self.agent_results)
    
    def complete(self, success: bool = None) -> None:
        """Mark coordination as completed."""
        self.completed_at = datetime.utcnow()
        self.total_time_ms = (self.completed_at - self.started_at).total_seconds() * 1000
        
        if success is not None:
            self.success = success
        else:
            # Determine success based on results
            self.success = len(self.failed_agents) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "coordination_id": self.coordination_id,
            "strategy": self.strategy.value,
            "total_agents": self.total_agents,
            "agent_results": {k: v.to_dict() for k, v in self.agent_results.items()},
            "failed_agents": self.failed_agents,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_time_ms": self.total_time_ms,
            "synchronization_points": self.synchronization_points,
            "coordination_overhead_ms": self.coordination_overhead_ms,
            "error": self.error
        }


@dataclass
class PluginEvent:
    """
    Plugin event for monitoring and coordination.
    
    Represents significant occurrences within plugin execution.
    """
    event_type: str
    plugin_id: str
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    severity: EventSeverity = EventSeverity.INFO
    
    # Context information
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    execution_id: Optional[str] = None
    
    # Tags for filtering and search
    tags: Dict[str, str] = field(default_factory=dict)
    
    def add_data(self, key: str, value: Any) -> None:
        """Add data to the event."""
        self.data[key] = value
    
    def add_tag(self, key: str, value: str) -> None:
        """Add a tag to the event."""
        self.tags[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type,
            "plugin_id": self.plugin_id,
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "severity": self.severity.value,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "execution_id": self.execution_id,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginEvent':
        """Create from dictionary."""
        # Convert timestamp back
        if 'timestamp' in data:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        # Convert severity enum
        if 'severity' in data and isinstance(data['severity'], str):
            data['severity'] = EventSeverity(data['severity'])
        
        return cls(**data)


@dataclass
class PluginError:
    """
    Plugin error information.
    
    Provides detailed error context for debugging and monitoring.
    """
    error_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    plugin_id: str = ""
    error_type: str = ""
    message: str = ""
    
    # Error context
    task_id: Optional[str] = None
    execution_id: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Error details
    details: Dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    
    # Recovery information
    recoverable: bool = True
    recovery_suggestions: List[str] = field(default_factory=list)
    
    def add_detail(self, key: str, value: Any) -> None:
        """Add error detail."""
        self.details[key] = value
    
    def add_recovery_suggestion(self, suggestion: str) -> None:
        """Add recovery suggestion."""
        self.recovery_suggestions.append(suggestion)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_id": self.error_id,
            "plugin_id": self.plugin_id,
            "error_type": self.error_type,
            "message": self.message,
            "task_id": self.task_id,
            "execution_id": self.execution_id,
            "stack_trace": self.stack_trace,
            "details": self.details,
            "occurred_at": self.occurred_at.isoformat(),
            "recoverable": self.recoverable,
            "recovery_suggestions": self.recovery_suggestions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginError':
        """Create from dictionary."""
        if 'occurred_at' in data:
            data['occurred_at'] = datetime.fromisoformat(data['occurred_at'])
        return cls(**data)


# Utility functions for model operations

def serialize_model(model: Union[PluginConfig, PluginContext, TaskResult, CoordinationResult, PluginEvent, PluginError]) -> str:
    """Serialize a model to JSON string."""
    return json.dumps(model.to_dict(), indent=2)


def deserialize_model(json_str: str, model_type: str) -> Union[PluginConfig, PluginContext, TaskResult, CoordinationResult, PluginEvent, PluginError]:
    """Deserialize a model from JSON string."""
    data = json.loads(json_str)
    
    model_classes = {
        "PluginConfig": PluginConfig,
        "PluginContext": PluginContext,
        "TaskResult": TaskResult,
        "CoordinationResult": CoordinationResult,
        "PluginEvent": PluginEvent,
        "PluginError": PluginError
    }
    
    model_class = model_classes.get(model_type)
    if not model_class:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_class.from_dict(data)


def create_success_result(plugin_id: str, task_id: str, data: Dict[str, Any] = None) -> TaskResult:
    """Create a successful task result."""
    result = TaskResult(
        success=True,
        plugin_id=plugin_id,
        task_id=task_id,
        data=data or {}
    )
    result.complete()
    return result


def create_error_result(plugin_id: str, task_id: str, error: str, error_code: str = None) -> TaskResult:
    """Create an error task result."""
    result = TaskResult(
        success=False,
        plugin_id=plugin_id,
        task_id=task_id
    )
    result.set_error(error, error_code)
    return result


def create_plugin_event(event_type: str, plugin_id: str, data: Dict[str, Any] = None, severity: EventSeverity = EventSeverity.INFO) -> PluginEvent:
    """Create a plugin event."""
    return PluginEvent(
        event_type=event_type,
        plugin_id=plugin_id,
        data=data or {},
        severity=severity
    )