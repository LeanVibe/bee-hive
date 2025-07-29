"""
Observability schemas for LeanVibe Agent Hive 2.0.

Comprehensive Pydantic schemas for event processing, hook management, and monitoring.
Implements the complete observability event schema contract for multi-agent coordination.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Literal
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict, validator


class HookEventCreate(BaseModel):
    """Schema for creating hook events."""
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }
    )
    
    session_id: uuid.UUID = Field(..., description="Session identifier")
    agent_id: uuid.UUID = Field(..., description="Agent identifier")
    event_type: str = Field(..., description="Type of event")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event payload data")
    correlation_id: Optional[str] = Field(None, description="Optional correlation ID")
    severity: Optional[str] = Field("info", description="Event severity level")
    tags: Optional[Dict[str, str]] = Field(None, description="Optional metadata tags")


class HookEventResponse(BaseModel):
    """Schema for hook event response."""
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    event_id: str = Field(..., description="Generated event ID")
    status: str = Field(..., description="Processing status")
    timestamp: datetime = Field(..., description="Processing timestamp")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class EventAnalyticsRequest(BaseModel):
    """Schema for event analytics request."""
    
    session_id: Optional[uuid.UUID] = None
    agent_id: Optional[uuid.UUID] = None
    time_range_hours: int = Field(default=1, ge=1, le=168)
    event_types: Optional[List[str]] = None
    include_trends: bool = True
    include_patterns: bool = True


class EventAnalyticsResponse(BaseModel):
    """Schema for event analytics response."""
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }
    )
    
    summary: Dict[str, Any]
    event_distribution: Dict[str, Any]
    performance_trends: Dict[str, Any]
    error_patterns: Dict[str, Any]
    agent_activity: Optional[Dict[str, Any]] = None
    recommendations: List[str]
    generated_at: datetime


class WebSocketSubscription(BaseModel):
    """Schema for WebSocket subscription preferences."""
    
    event_types: List[str] = Field(default_factory=list)
    agent_filters: List[str] = Field(default_factory=list)
    session_filters: List[str] = Field(default_factory=list)
    severity_filters: List[str] = Field(
        default=["info", "warning", "error", "critical"]
    )


class HealthCheckResponse(BaseModel):
    """Schema for health check response."""
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        }
    )
    
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    websocket_connections: Dict[str, Any] = Field(..., description="WebSocket statistics")
    system_info: Dict[str, Any] = Field(..., description="System information")
    health_checks: Dict[str, str] = Field(..., description="Individual health checks")


class TestEventRequest(BaseModel):
    """Schema for test event triggering."""
    
    event_type: str = Field(..., description="Event type to trigger")
    agent_id: Optional[str] = Field(None, description="Optional agent ID")
    session_id: Optional[str] = Field(None, description="Optional session ID")
    payload: Optional[Dict[str, Any]] = Field(None, description="Optional event payload")


class TestEventResponse(BaseModel):
    """Schema for test event response."""
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        }
    )
    
    success: bool = Field(..., description="Whether test event was triggered")
    event_id: str = Field(..., description="Generated event ID")
    event_type: str = Field(..., description="Event type that was triggered")
    agent_id: str = Field(..., description="Agent ID used")
    session_id: str = Field(..., description="Session ID used")
    timestamp: datetime = Field(..., description="Trigger timestamp")


# ==========================================
# COMPREHENSIVE OBSERVABILITY EVENT SCHEMAS
# ==========================================

class EventCategory(str, Enum):
    """Event categories for comprehensive observability."""
    WORKFLOW = "workflow"
    AGENT = "agent"
    TOOL = "tool"
    MEMORY = "memory"
    COMMUNICATION = "communication"
    RECOVERY = "recovery"
    SYSTEM = "system"


class WorkflowEventType(str, Enum):
    """Workflow-related event types."""
    WORKFLOW_STARTED = "WorkflowStarted"
    WORKFLOW_ENDED = "WorkflowEnded"
    WORKFLOW_PAUSED = "WorkflowPaused"
    WORKFLOW_RESUMED = "WorkflowResumed"
    NODE_EXECUTING = "NodeExecuting"
    NODE_COMPLETED = "NodeCompleted"
    NODE_FAILED = "NodeFailed"


class AgentEventType(str, Enum):
    """Agent-related event types."""
    AGENT_STATE_CHANGED = "AgentStateChanged"
    AGENT_CAPABILITY_UTILIZED = "AgentCapabilityUtilized"
    AGENT_STARTED = "AgentStarted"
    AGENT_STOPPED = "AgentStopped"
    AGENT_PAUSED = "AgentPaused"
    AGENT_RESUMED = "AgentResumed"


class ToolEventType(str, Enum):
    """Tool-related event types."""
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    TOOL_REGISTERED = "ToolRegistered"
    TOOL_UNREGISTERED = "ToolUnregistered"
    TOOL_UPDATED = "ToolUpdated"


class MemoryEventType(str, Enum):
    """Memory-related event types."""
    SEMANTIC_QUERY = "SemanticQuery"
    SEMANTIC_UPDATE = "SemanticUpdate"
    MEMORY_CONSOLIDATION = "MemoryConsolidation"


class CommunicationEventType(str, Enum):
    """Communication-related event types."""
    MESSAGE_PUBLISHED = "MessagePublished"
    MESSAGE_RECEIVED = "MessageReceived"
    BROADCAST_EVENT = "BroadcastEvent"


class RecoveryEventType(str, Enum):
    """Recovery-related event types."""
    FAILURE_DETECTED = "FailureDetected"
    RECOVERY_INITIATED = "RecoveryInitiated"
    RECOVERY_COMPLETED = "RecoveryCompleted"


class SystemEventType(str, Enum):
    """System-related event types."""
    SYSTEM_HEALTH_CHECK = "SystemHealthCheck"
    CONFIGURATION_CHANGE = "ConfigurationChange"


class PerformanceMetrics(BaseModel):
    """Performance metrics for events."""
    
    execution_time_ms: Optional[float] = Field(None, ge=0, description="Execution time in milliseconds")
    memory_usage_mb: Optional[float] = Field(None, ge=0, description="Memory usage in megabytes")
    cpu_usage_percent: Optional[float] = Field(None, ge=0, le=100, description="CPU usage percentage")


class EventMetadata(BaseModel):
    """Event metadata for tracing and correlation."""
    
    schema_version: str = Field(default="1.0.0", description="Event schema version")
    correlation_id: Optional[uuid.UUID] = Field(None, description="Request correlation identifier")
    source_service: Optional[str] = Field(None, description="Source service name")
    trace_id: Optional[str] = Field(None, description="Distributed tracing identifier")
    span_id: Optional[str] = Field(None, description="Tracing span identifier")
    sampling_probability: Optional[float] = Field(None, ge=0, le=1, description="Event sampling probability")


class BaseObservabilityEvent(BaseModel):
    """Base observability event with comprehensive fields."""
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v)
        },
        use_enum_values=True
    )
    
    event_id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique event identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="ISO8601 event timestamp")
    event_type: str = Field(..., description="Specific event type from predefined enums")
    event_category: EventCategory = Field(..., description="High-level event category")
    
    # Optional identifiers
    workflow_id: Optional[uuid.UUID] = Field(None, description="Optional workflow identifier")
    agent_id: Optional[uuid.UUID] = Field(None, description="Optional agent identifier")
    session_id: Optional[uuid.UUID] = Field(None, description="Optional session identifier")
    context_id: Optional[uuid.UUID] = Field(None, description="Optional context identifier")
    
    # Semantic embedding for context-aware analysis
    semantic_embedding: Optional[List[float]] = Field(None, description="Optional 1536-dimensional semantic embedding vector")
    
    # Event-specific payload
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event-specific data payload")
    
    # Performance metrics
    performance_metrics: Optional[PerformanceMetrics] = Field(None, description="Performance metrics")
    
    # Metadata for tracing and correlation
    metadata: EventMetadata = Field(default_factory=EventMetadata, description="Event metadata")
    
    @validator('semantic_embedding')
    def validate_embedding_dimensions(cls, v):
        """Validate semantic embedding has correct dimensions."""
        if v is not None and len(v) != 1536:
            raise ValueError("Semantic embedding must be exactly 1536 dimensions")
        return v


# ==========================================
# WORKFLOW EVENT SCHEMAS
# ==========================================

class WorkflowStartedEvent(BaseObservabilityEvent):
    """Workflow started event schema."""
    
    event_type: Literal[WorkflowEventType.WORKFLOW_STARTED] = WorkflowEventType.WORKFLOW_STARTED
    event_category: Literal[EventCategory.WORKFLOW] = EventCategory.WORKFLOW
    workflow_id: uuid.UUID = Field(..., description="Workflow identifier")
    
    # Workflow-specific payload validation
    workflow_name: str = Field(..., description="Workflow name")
    workflow_definition: Dict[str, Any] = Field(..., description="Workflow definition")
    initial_context: Optional[Dict[str, Any]] = Field(None, description="Initial workflow context")
    estimated_duration_ms: Optional[float] = Field(None, description="Estimated duration in milliseconds")
    priority: Optional[str] = Field(None, description="Workflow priority")
    initiating_agent: Optional[uuid.UUID] = Field(None, description="Agent that initiated the workflow")


class WorkflowEndedEvent(BaseObservabilityEvent):
    """Workflow ended event schema."""
    
    event_type: Literal[WorkflowEventType.WORKFLOW_ENDED] = WorkflowEventType.WORKFLOW_ENDED
    event_category: Literal[EventCategory.WORKFLOW] = EventCategory.WORKFLOW
    workflow_id: uuid.UUID = Field(..., description="Workflow identifier")
    
    # Workflow completion data
    status: Literal["completed", "failed", "cancelled"] = Field(..., description="Workflow completion status")
    completion_reason: str = Field(..., description="Reason for completion")
    final_result: Optional[Dict[str, Any]] = Field(None, description="Final workflow result")
    total_tasks_executed: Optional[int] = Field(None, ge=0, description="Total tasks executed")
    failed_tasks: Optional[int] = Field(None, ge=0, description="Number of failed tasks")
    actual_duration_ms: Optional[float] = Field(None, description="Actual duration in milliseconds")


class NodeExecutingEvent(BaseObservabilityEvent):
    """Task node executing event schema."""
    
    event_type: Literal[WorkflowEventType.NODE_EXECUTING] = WorkflowEventType.NODE_EXECUTING
    event_category: Literal[EventCategory.WORKFLOW] = EventCategory.WORKFLOW
    workflow_id: uuid.UUID = Field(..., description="Workflow identifier")
    
    # Node execution data
    node_id: str = Field(..., description="Task node identifier")
    node_type: str = Field(..., description="Type of task node")
    node_name: Optional[str] = Field(None, description="Human-readable node name")
    input_data: Optional[Dict[str, Any]] = Field(None, description="Node input data")
    dependencies_satisfied: Optional[List[str]] = Field(None, description="Satisfied dependencies")
    assigned_agent: Optional[uuid.UUID] = Field(None, description="Agent assigned to execute node")
    estimated_execution_time_ms: Optional[float] = Field(None, description="Estimated execution time")


class NodeCompletedEvent(BaseObservabilityEvent):
    """Task node completed event schema."""
    
    event_type: Literal[WorkflowEventType.NODE_COMPLETED] = WorkflowEventType.NODE_COMPLETED
    event_category: Literal[EventCategory.WORKFLOW] = EventCategory.WORKFLOW
    workflow_id: uuid.UUID = Field(..., description="Workflow identifier")
    
    # Node completion data
    node_id: str = Field(..., description="Task node identifier")
    success: bool = Field(..., description="Whether node execution succeeded")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Node output data")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error details if failed")
    retry_count: Optional[int] = Field(None, ge=0, description="Number of retries attempted")
    downstream_nodes: Optional[List[str]] = Field(None, description="Downstream nodes to execute")


# ==========================================
# AGENT EVENT SCHEMAS
# ==========================================

class AgentStateChangedEvent(BaseObservabilityEvent):
    """Agent state changed event schema."""
    
    event_type: Literal[AgentEventType.AGENT_STATE_CHANGED] = AgentEventType.AGENT_STATE_CHANGED
    event_category: Literal[EventCategory.AGENT] = EventCategory.AGENT
    agent_id: uuid.UUID = Field(..., description="Agent identifier")
    
    # State change data
    previous_state: str = Field(..., description="Previous agent state")
    new_state: str = Field(..., description="New agent state")
    state_transition_reason: str = Field(..., description="Reason for state transition")
    capabilities: Optional[List[str]] = Field(None, description="Agent capabilities")
    resource_allocation: Optional[Dict[str, Any]] = Field(None, description="Resource allocation")
    persona_data: Optional[Dict[str, Any]] = Field(None, description="Agent persona data")


class AgentCapabilityUtilizedEvent(BaseObservabilityEvent):
    """Agent capability utilized event schema."""
    
    event_type: Literal[AgentEventType.AGENT_CAPABILITY_UTILIZED] = AgentEventType.AGENT_CAPABILITY_UTILIZED
    event_category: Literal[EventCategory.AGENT] = EventCategory.AGENT
    agent_id: uuid.UUID = Field(..., description="Agent identifier")
    
    # Capability utilization data
    capability_name: str = Field(..., description="Name of utilized capability")
    utilization_context: str = Field(..., description="Context of capability utilization")
    input_parameters: Optional[Dict[str, Any]] = Field(None, description="Input parameters")
    capability_result: Optional[Dict[str, Any]] = Field(None, description="Capability execution result")
    efficiency_score: Optional[float] = Field(None, ge=0, le=1, description="Efficiency score")


# ==========================================
# TOOL EVENT SCHEMAS
# ==========================================

class PreToolUseEvent(BaseObservabilityEvent):
    """Pre-tool use event schema."""
    
    event_type: Literal[ToolEventType.PRE_TOOL_USE] = ToolEventType.PRE_TOOL_USE
    event_category: Literal[EventCategory.TOOL] = EventCategory.TOOL
    agent_id: uuid.UUID = Field(..., description="Agent identifier")
    
    # Tool execution data
    tool_name: str = Field(..., description="Name of the tool")
    parameters: Dict[str, Any] = Field(..., description="Tool parameters")
    tool_version: Optional[str] = Field(None, description="Tool version")
    expected_output_type: Optional[str] = Field(None, description="Expected output type")
    timeout_ms: Optional[int] = Field(None, description="Timeout in milliseconds")
    retry_policy: Optional[Dict[str, Any]] = Field(None, description="Retry policy")


class PostToolUseEvent(BaseObservabilityEvent):
    """Post-tool use event schema."""
    
    event_type: Literal[ToolEventType.POST_TOOL_USE] = ToolEventType.POST_TOOL_USE
    event_category: Literal[EventCategory.TOOL] = EventCategory.TOOL
    agent_id: uuid.UUID = Field(..., description="Agent identifier")
    
    # Tool execution result data
    tool_name: str = Field(..., description="Name of the tool")
    success: bool = Field(..., description="Whether tool execution succeeded")
    result: Optional[Any] = Field(None, description="Tool execution result")
    error: Optional[str] = Field(None, description="Error message if failed")
    error_type: Optional[str] = Field(None, description="Error type classification")
    retry_count: Optional[int] = Field(None, ge=0, description="Number of retries attempted")
    result_truncated: Optional[bool] = Field(False, description="Whether result was truncated")
    full_result_size: Optional[int] = Field(None, description="Size of full result before truncation")


# ==========================================
# MEMORY EVENT SCHEMAS
# ==========================================

class SemanticQueryEvent(BaseObservabilityEvent):
    """Semantic query event schema."""
    
    event_type: Literal[MemoryEventType.SEMANTIC_QUERY] = MemoryEventType.SEMANTIC_QUERY
    event_category: Literal[EventCategory.MEMORY] = EventCategory.MEMORY
    
    # Query data
    query_text: str = Field(..., description="Query text")
    query_embedding: List[float] = Field(..., description="Query embedding vector")
    similarity_threshold: Optional[float] = Field(None, description="Similarity threshold")
    max_results: Optional[int] = Field(None, description="Maximum number of results")
    filter_criteria: Optional[Dict[str, Any]] = Field(None, description="Filter criteria")
    results_count: Optional[int] = Field(None, description="Number of results returned")
    search_strategy: Optional[str] = Field(None, description="Search strategy used")


class SemanticUpdateEvent(BaseObservabilityEvent):
    """Semantic update event schema."""
    
    event_type: Literal[MemoryEventType.SEMANTIC_UPDATE] = MemoryEventType.SEMANTIC_UPDATE
    event_category: Literal[EventCategory.MEMORY] = EventCategory.MEMORY
    
    # Update operation data
    operation_type: Literal["insert", "update", "delete"] = Field(..., description="Type of operation")
    content: Dict[str, Any] = Field(..., description="Content being updated")
    content_embedding: Optional[List[float]] = Field(None, description="Content embedding vector")
    content_id: Optional[str] = Field(None, description="Content identifier")
    content_type: Optional[str] = Field(None, description="Content type")
    content_metadata: Optional[Dict[str, Any]] = Field(None, description="Content metadata")
    affected_records: Optional[int] = Field(None, description="Number of affected records")


# ==========================================
# COMMUNICATION EVENT SCHEMAS
# ==========================================

class MessagePublishedEvent(BaseObservabilityEvent):
    """Message published event schema."""
    
    event_type: Literal[CommunicationEventType.MESSAGE_PUBLISHED] = CommunicationEventType.MESSAGE_PUBLISHED
    event_category: Literal[EventCategory.COMMUNICATION] = EventCategory.COMMUNICATION
    
    # Message data
    message_id: uuid.UUID = Field(..., description="Message identifier")
    from_agent: str = Field(..., description="Sender agent")
    to_agent: str = Field(..., description="Recipient agent")
    message_type: str = Field(..., description="Message type")
    message_content: Dict[str, Any] = Field(..., description="Message content")
    priority: Optional[str] = Field(None, description="Message priority")
    delivery_method: Optional[str] = Field(None, description="Delivery method")
    expected_response: Optional[bool] = Field(False, description="Whether response is expected")


class MessageReceivedEvent(BaseObservabilityEvent):
    """Message received event schema."""
    
    event_type: Literal[CommunicationEventType.MESSAGE_RECEIVED] = CommunicationEventType.MESSAGE_RECEIVED
    event_category: Literal[EventCategory.COMMUNICATION] = EventCategory.COMMUNICATION
    
    # Message receipt data
    message_id: uuid.UUID = Field(..., description="Message identifier")
    from_agent: str = Field(..., description="Sender agent")
    processing_status: Literal["accepted", "rejected", "deferred"] = Field(..., description="Processing status")
    processing_reason: Optional[str] = Field(None, description="Reason for processing status")
    response_generated: Optional[bool] = Field(False, description="Whether response was generated")
    delivery_latency_ms: Optional[float] = Field(None, description="Delivery latency in milliseconds")


# ==========================================
# RECOVERY EVENT SCHEMAS
# ==========================================

class FailureDetectedEvent(BaseObservabilityEvent):
    """Failure detected event schema."""
    
    event_type: Literal[RecoveryEventType.FAILURE_DETECTED] = RecoveryEventType.FAILURE_DETECTED
    event_category: Literal[EventCategory.RECOVERY] = EventCategory.RECOVERY
    
    # Failure data
    failure_type: str = Field(..., description="Type of failure")
    failure_description: str = Field(..., description="Description of failure")
    affected_component: str = Field(..., description="Affected system component")
    severity: Literal["low", "medium", "high", "critical"] = Field(..., description="Failure severity")
    error_details: Dict[str, Any] = Field(..., description="Detailed error information")
    detection_method: Optional[str] = Field(None, description="How failure was detected")
    impact_assessment: Optional[Dict[str, Any]] = Field(None, description="Impact assessment")


class RecoveryInitiatedEvent(BaseObservabilityEvent):
    """Recovery initiated event schema."""
    
    event_type: Literal[RecoveryEventType.RECOVERY_INITIATED] = RecoveryEventType.RECOVERY_INITIATED
    event_category: Literal[EventCategory.RECOVERY] = EventCategory.RECOVERY
    
    # Recovery data
    recovery_strategy: str = Field(..., description="Recovery strategy being used")
    trigger_failure: str = Field(..., description="Failure that triggered recovery")
    recovery_steps: List[str] = Field(..., description="Steps in recovery process")
    estimated_recovery_time_ms: Optional[float] = Field(None, description="Estimated recovery time")
    backup_systems_activated: Optional[List[str]] = Field(None, description="Activated backup systems")
    rollback_checkpoint: Optional[str] = Field(None, description="Rollback checkpoint if used")


# ==========================================
# SYSTEM EVENT SCHEMAS
# ==========================================

class SystemHealthCheckEvent(BaseObservabilityEvent):
    """System health check event schema."""
    
    event_type: Literal[SystemEventType.SYSTEM_HEALTH_CHECK] = SystemEventType.SYSTEM_HEALTH_CHECK
    event_category: Literal[EventCategory.SYSTEM] = EventCategory.SYSTEM
    
    # Health check data
    health_status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Overall health status")
    check_type: str = Field(..., description="Type of health check")
    component_statuses: Dict[str, str] = Field(..., description="Individual component statuses")
    performance_indicators: Dict[str, Any] = Field(..., description="Performance indicators")
    alerts_triggered: Optional[List[str]] = Field(None, description="Alerts triggered")
    recommended_actions: Optional[List[str]] = Field(None, description="Recommended actions")


# ==========================================
# EVENT UNION TYPE FOR COMPREHENSIVE TYPING
# ==========================================

ObservabilityEvent = Union[
    WorkflowStartedEvent,
    WorkflowEndedEvent,
    NodeExecutingEvent,
    NodeCompletedEvent,
    AgentStateChangedEvent,
    AgentCapabilityUtilizedEvent,
    PreToolUseEvent,
    PostToolUseEvent,
    SemanticQueryEvent,
    SemanticUpdateEvent,
    MessagePublishedEvent,
    MessageReceivedEvent,
    FailureDetectedEvent,
    RecoveryInitiatedEvent,
    SystemHealthCheckEvent,
]