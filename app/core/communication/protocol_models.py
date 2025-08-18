"""
Data Models for Multi-CLI Communication Protocol

This module defines the core data structures for communication between
heterogeneous CLI agents with different protocols and message formats.
"""

import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pydantic import BaseModel, Field

from ..agents.universal_agent_interface import AgentType

# ================================================================================
# Enums and Constants
# ================================================================================

class MessageType(str, Enum):
    """Types of messages in the communication protocol."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    STATUS_UPDATE = "status_update"
    CONTEXT_HANDOFF = "context_handoff"
    CAPABILITY_QUERY = "capability_query"
    HEALTH_CHECK = "health_check"
    ERROR_REPORT = "error_report"
    COORDINATION_REQUEST = "coordination_request"
    WORKFLOW_CONTROL = "workflow_control"

class MessagePriority(str, Enum):
    """Message priority levels."""
    CRITICAL = "critical"    # System-critical messages
    HIGH = "high"           # High-priority task messages  
    NORMAL = "normal"       # Standard messages
    LOW = "low"            # Background messages
    BULK = "bulk"          # Bulk data transfer

class DeliveryMode(str, Enum):
    """Message delivery modes."""
    FIRE_AND_FORGET = "fire_and_forget"    # No acknowledgment required
    AT_LEAST_ONCE = "at_least_once"        # Retry until acknowledged
    EXACTLY_ONCE = "exactly_once"          # Guaranteed single delivery
    ORDERED = "ordered"                     # Maintain message order

class CLIProtocol(str, Enum):
    """Supported CLI protocols."""
    CLAUDE_CODE = "claude_code"             # Claude Code CLI protocol
    CURSOR = "cursor"                       # Cursor CLI protocol  
    GEMINI_CLI = "gemini_cli"              # Gemini CLI protocol
    GITHUB_COPILOT = "github_copilot"      # GitHub Copilot CLI protocol
    OPENCODE = "opencode"                   # OpenCode CLI protocol
    UNIVERSAL = "universal"                 # Universal agent protocol

class HandoffStatus(str, Enum):
    """Status of agent handoff process."""
    INITIATED = "initiated"
    CONTEXT_PACKAGED = "context_packaged"
    AGENT_SELECTED = "agent_selected"
    CONTEXT_TRANSFERRED = "context_transferred"
    HANDOFF_COMPLETED = "handoff_completed"
    HANDOFF_FAILED = "handoff_failed"

# ================================================================================
# Core Message Models
# ================================================================================

@dataclass
class UniversalMessage:
    """Universal message format for inter-agent communication."""
    
    # Message identification
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    
    # Message routing
    source_agent_id: str = ""
    source_agent_type: AgentType = AgentType.CLAUDE_CODE
    target_agent_id: Optional[str] = None
    target_agent_type: Optional[AgentType] = None
    
    # Message content
    message_type: MessageType = MessageType.TASK_REQUEST
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Delivery configuration
    priority: MessagePriority = MessagePriority.NORMAL
    delivery_mode: DeliveryMode = DeliveryMode.AT_LEAST_ONCE
    ttl_seconds: int = 3600  # Time to live
    retry_count: int = 0
    max_retries: int = 3
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    
    # Context preservation
    execution_context: Optional[Dict[str, Any]] = None
    previous_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Set expiration time based on TTL."""
        if self.expires_at is None and self.ttl_seconds > 0:
            self.expires_at = self.created_at + timedelta(seconds=self.ttl_seconds)

@dataclass
class CLIMessage:
    """CLI-specific message format for individual CLI tools."""
    
    # Message identification  
    cli_message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    universal_message_id: str = ""
    
    # CLI-specific routing
    cli_protocol: CLIProtocol = CLIProtocol.CLAUDE_CODE
    cli_command: str = ""
    cli_args: List[str] = field(default_factory=list)
    cli_options: Dict[str, str] = field(default_factory=dict)
    
    # CLI-specific content
    input_data: Dict[str, Any] = field(default_factory=dict)
    input_files: List[str] = field(default_factory=list)
    working_directory: Optional[str] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    # Execution configuration
    timeout_seconds: int = 300
    require_human_approval: bool = False
    isolation_required: bool = True
    
    # Output expectations
    expected_output_format: str = "json"
    capture_stdout: bool = True
    capture_stderr: bool = True
    capture_files: bool = True
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

# ================================================================================
# Context and Handoff Models
# ================================================================================

@dataclass
class ContextPackage:
    """Package of context information for agent handoffs."""
    
    # Package identification
    package_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_agent_id: str = ""
    target_agent_id: str = ""
    
    # Context data
    execution_context: Dict[str, Any] = field(default_factory=dict)
    task_history: List[Dict[str, Any]] = field(default_factory=list)
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    
    # State information
    current_state: Dict[str, Any] = field(default_factory=dict)
    variable_bindings: Dict[str, Any] = field(default_factory=dict)
    workflow_position: Optional[str] = None
    
    # Handoff metadata
    handoff_reason: str = ""
    required_capabilities: List[str] = field(default_factory=list)
    context_format_version: str = "1.0"
    compression_used: bool = False
    
    # Quality and validation
    context_integrity_hash: Optional[str] = None
    validation_status: str = "pending"  # pending, valid, invalid
    package_size_bytes: int = 0
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

@dataclass
class HandoffRequest:
    """Request for agent handoff with context transfer."""
    
    # Request identification
    handoff_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_agent_id: str = ""
    target_agent_type: Optional[AgentType] = None
    target_agent_id: Optional[str] = None
    
    # Handoff configuration
    handoff_reason: str = ""
    urgency_level: MessagePriority = MessagePriority.NORMAL
    require_capability_match: bool = True
    preserve_isolation: bool = True
    
    # Context and requirements
    context_package: Optional[ContextPackage] = None
    required_capabilities: List[str] = field(default_factory=list)
    preferred_agents: List[str] = field(default_factory=list)
    excluded_agents: List[str] = field(default_factory=list)
    
    # Execution preferences
    max_handoff_time_seconds: int = 60
    fallback_strategy: str = "retry_with_different_agent"
    notification_required: bool = True
    
    # Status tracking
    status: HandoffStatus = HandoffStatus.INITIATED
    attempted_agents: List[str] = field(default_factory=list)
    error_history: List[str] = field(default_factory=list)
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

# ================================================================================
# Protocol Configuration Models
# ================================================================================

@dataclass
class ProtocolConfig:
    """Configuration for CLI protocol communication."""
    
    # Protocol identification
    protocol_name: CLIProtocol = CLIProtocol.UNIVERSAL
    protocol_version: str = "1.0"
    
    # Communication settings
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    # Message formatting
    message_format: str = "json"  # json, yaml, xml, binary
    compression_enabled: bool = False
    encryption_enabled: bool = False
    
    # Protocol-specific settings
    cli_command_prefix: str = ""
    default_working_directory: Optional[str] = None
    default_environment: Dict[str, str] = field(default_factory=dict)
    
    # Performance settings
    batch_size: int = 10
    connection_pool_size: int = 5
    keep_alive: bool = True
    
    # Capability mappings
    capability_mappings: Dict[str, str] = field(default_factory=dict)
    command_mappings: Dict[str, str] = field(default_factory=dict)
    
    # Security settings
    allowed_commands: List[str] = field(default_factory=list)
    blocked_commands: List[str] = field(default_factory=list)
    require_authentication: bool = True

@dataclass
class MessageRoute:
    """Routing information for message delivery."""
    
    # Route identification
    route_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_protocol: CLIProtocol = CLIProtocol.UNIVERSAL
    target_protocol: CLIProtocol = CLIProtocol.CLAUDE_CODE
    
    # Routing configuration
    route_pattern: str = ""  # Pattern for matching messages
    priority: int = 100  # Lower numbers = higher priority
    load_balancing: bool = False
    
    # Translation settings
    requires_translation: bool = True
    translation_rules: Dict[str, Any] = field(default_factory=dict)
    context_preservation: bool = True
    
    # Quality and performance
    success_rate: float = 0.0
    average_latency_ms: float = 0.0
    total_messages_routed: int = 0
    last_used: Optional[datetime] = None
    
    # Conditions and constraints
    conditions: Dict[str, Any] = field(default_factory=dict)
    time_constraints: Optional[Dict[str, Any]] = None
    resource_constraints: Optional[Dict[str, Any]] = None

# ================================================================================
# Communication Bridge Models
# ================================================================================

@dataclass
class BridgeConnection:
    """Connection information for communication bridges."""
    
    # Connection identification
    connection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    connection_name: str = ""
    
    # Connection details
    protocol: CLIProtocol = CLIProtocol.UNIVERSAL
    endpoint: str = ""
    connection_type: str = "websocket"  # websocket, redis, http, tcp
    
    # Authentication
    auth_method: str = "none"  # none, api_key, oauth, certificate
    credentials: Dict[str, str] = field(default_factory=dict)
    
    # Connection state
    is_connected: bool = False
    last_heartbeat: Optional[datetime] = None
    connection_quality: float = 1.0  # 0.0 to 1.0
    
    # Performance metrics
    messages_sent: int = 0
    messages_received: int = 0
    average_response_time_ms: float = 0.0
    error_count: int = 0
    
    # Configuration
    auto_reconnect: bool = True
    heartbeat_interval_seconds: int = 30
    max_message_size_bytes: int = 1024 * 1024  # 1MB
    
    # Timing
    created_at: datetime = field(default_factory=datetime.utcnow)
    connected_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None

# ================================================================================
# Error and Quality Models
# ================================================================================

@dataclass
class CommunicationError:
    """Error information for communication failures."""
    
    # Error identification
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_id: Optional[str] = None
    connection_id: Optional[str] = None
    
    # Error details
    error_type: str = ""
    error_code: Optional[str] = None
    error_message: str = ""
    stack_trace: Optional[str] = None
    
    # Context
    occurred_at: datetime = field(default_factory=datetime.utcnow)
    operation: str = ""
    source_protocol: Optional[CLIProtocol] = None
    target_protocol: Optional[CLIProtocol] = None
    
    # Recovery information
    is_recoverable: bool = True
    retry_recommended: bool = True
    alternative_routes: List[str] = field(default_factory=list)
    
    # Impact assessment
    affects_handoff: bool = False
    blocks_workflow: bool = False
    requires_manual_intervention: bool = False

@dataclass
class QualityMetrics:
    """Quality metrics for communication performance."""
    
    # Metrics identification
    metrics_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    time_window_start: datetime = field(default_factory=datetime.utcnow)
    time_window_end: datetime = field(default_factory=datetime.utcnow)
    
    # Performance metrics
    total_messages: int = 0
    successful_messages: int = 0
    failed_messages: int = 0
    average_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    
    # Quality indicators
    success_rate: float = 0.0
    message_loss_rate: float = 0.0
    duplicate_rate: float = 0.0
    out_of_order_rate: float = 0.0
    
    # Protocol-specific metrics
    protocol_metrics: Dict[CLIProtocol, Dict[str, float]] = field(default_factory=dict)
    
    # Trend indicators
    performance_trend: str = "stable"  # improving, stable, degrading
    quality_score: float = 0.0  # 0.0 to 1.0
    
    # Recommendations
    optimization_suggestions: List[str] = field(default_factory=list)