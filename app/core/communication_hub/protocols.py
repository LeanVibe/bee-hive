"""
Unified Message Protocols and Schemas for CommunicationHub

This module defines the standardized message formats, types, and protocols
that unify all communication patterns across the bee-hive system.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from typing import Dict, List, Optional, Any, Union


class MessageType(str, Enum):
    """Standardized message types for all system communication."""
    
    # Task and workflow messages
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_ASSIGNMENT = "task_assignment"
    TASK_COMPLETION = "task_completion"
    TASK_FAILURE = "task_failure"
    TASK_CANCELLATION = "task_cancellation"
    
    # Agent coordination messages
    AGENT_HEARTBEAT = "agent_heartbeat"
    AGENT_REGISTRATION = "agent_registration"
    AGENT_DEREGISTRATION = "agent_deregistration"
    AGENT_STATUS_UPDATE = "agent_status_update"
    COORDINATION_REQUEST = "coordination_request"
    COORDINATION_RESPONSE = "coordination_response"
    SYNC_POINT = "sync_point"
    
    # Event and notification messages
    EVENT_PUBLISH = "event_publish"
    EVENT_SUBSCRIBE = "event_subscribe"
    EVENT_UNSUBSCRIBE = "event_unsubscribe"
    SYSTEM_NOTIFICATION = "system_notification"
    ERROR_NOTIFICATION = "error_notification"
    
    # Real-time communication
    REALTIME_UPDATE = "realtime_update"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    
    # System management
    HEALTH_CHECK = "health_check"
    CONFIG_UPDATE = "config_update"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_RESTART = "system_restart"
    
    # Hook and lifecycle messages
    HOOK_PRE_TOOL_USE = "hook_pre_tool_use"
    HOOK_POST_TOOL_USE = "hook_post_tool_use"
    HOOK_ERROR = "hook_error"
    LIFECYCLE_START = "lifecycle_start"
    LIFECYCLE_STOP = "lifecycle_stop"
    
    # Knowledge and context sharing
    KNOWLEDGE_SHARE = "knowledge_share"
    CONTEXT_UPDATE = "context_update"
    MEMORY_SYNC = "memory_sync"
    
    # General purpose
    REQUEST = "request"
    RESPONSE = "response"
    COMMAND = "command"
    ACK = "acknowledgment"
    NACK = "negative_acknowledgment"


class Priority(IntEnum):
    """Message priority levels (lower number = higher priority)."""
    CRITICAL = 1    # System-critical messages
    URGENT = 2      # High-priority operations
    HIGH = 3        # Important but not urgent
    MEDIUM = 4      # Normal priority (default)
    LOW = 5         # Background operations


class DeliveryGuarantee(str, Enum):
    """Message delivery guarantee levels."""
    BEST_EFFORT = "best_effort"        # Fire and forget
    AT_LEAST_ONCE = "at_least_once"    # May be delivered multiple times
    EXACTLY_ONCE = "exactly_once"      # Delivered exactly once
    ORDERED = "ordered"                # Maintains order within stream


class ProtocolType(str, Enum):
    """Supported communication protocols."""
    REDIS_PUBSUB = "redis_pubsub"
    REDIS_STREAMS = "redis_streams"
    WEBSOCKET = "websocket"
    HTTP = "http"
    HTTPS = "https"
    TCP = "tcp"
    UDP = "udp"


class MessageStatus(str, Enum):
    """Message processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class UnifiedMessage:
    """
    Unified message format for all system communication.
    
    This message format consolidates all communication patterns and provides
    a standard interface for routing, processing, and monitoring messages.
    """
    
    # Core identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Routing information
    source: str = ""                    # Source component/agent ID
    destination: str = ""               # Target component/agent ID
    routing_key: Optional[str] = None   # Topic/routing key for pattern matching
    
    # Message classification
    message_type: MessageType = MessageType.REQUEST
    priority: Priority = Priority.MEDIUM
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.BEST_EFFORT
    
    # Message lifecycle
    ttl: Optional[int] = None           # Time-to-live in milliseconds
    expires_at: Optional[datetime] = None
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Content
    headers: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Delivery tracking
    delivery_attempts: int = 0
    max_retries: int = 3
    status: MessageStatus = MessageStatus.PENDING
    
    # Protocol-specific data
    protocol_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Set expiration if TTL is specified
        if self.ttl and not self.expires_at:
            self.expires_at = self.timestamp + timedelta(milliseconds=self.ttl)
        
        # Generate conversation ID if not provided
        if not self.conversation_id and self.correlation_id:
            self.conversation_id = self.correlation_id
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        return self.expires_at is not None and datetime.utcnow() > self.expires_at
    
    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return (
            self.delivery_attempts < self.max_retries and
            not self.is_expired() and
            self.status in (MessageStatus.PENDING, MessageStatus.FAILED)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "destination": self.destination,
            "routing_key": self.routing_key,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "delivery_guarantee": self.delivery_guarantee.value,
            "ttl": self.ttl,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "conversation_id": self.conversation_id,
            "headers": self.headers,
            "payload": self.payload,
            "delivery_attempts": self.delivery_attempts,
            "max_retries": self.max_retries,
            "status": self.status.value,
            "protocol_data": self.protocol_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedMessage":
        """Create message from dictionary."""
        message = cls(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.utcnow(),
            source=data.get("source", ""),
            destination=data.get("destination", ""),
            routing_key=data.get("routing_key"),
            message_type=MessageType(data.get("message_type", MessageType.REQUEST.value)),
            priority=Priority(data.get("priority", Priority.MEDIUM.value)),
            delivery_guarantee=DeliveryGuarantee(data.get("delivery_guarantee", DeliveryGuarantee.BEST_EFFORT.value)),
            ttl=data.get("ttl"),
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            conversation_id=data.get("conversation_id"),
            headers=data.get("headers", {}),
            payload=data.get("payload", {}),
            delivery_attempts=data.get("delivery_attempts", 0),
            max_retries=data.get("max_retries", 3),
            status=MessageStatus(data.get("status", MessageStatus.PENDING.value)),
            protocol_data=data.get("protocol_data", {})
        )
        
        # Handle expires_at
        if "expires_at" in data and data["expires_at"]:
            message.expires_at = datetime.fromisoformat(data["expires_at"])
        
        return message


@dataclass
class MessageRoute:
    """Message routing configuration."""
    pattern: str                        # Routing pattern (glob or regex)
    destination: str                    # Target destination
    protocol: ProtocolType              # Protocol to use
    priority: Priority = Priority.MEDIUM
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubscriptionConfig:
    """Subscription configuration for message handlers."""
    pattern: str                        # Message pattern to match
    handler_id: str                     # Unique handler identifier
    message_types: List[MessageType] = field(default_factory=list)
    priority_filter: Optional[Priority] = None
    source_filter: Optional[str] = None
    auto_ack: bool = True
    batch_size: int = 1
    timeout_ms: int = 5000


@dataclass
class ConnectionConfig:
    """Configuration for protocol connections."""
    protocol: ProtocolType
    host: str = "localhost"
    port: int = 6379
    connection_params: Dict[str, Any] = field(default_factory=dict)
    pool_size: int = 10
    timeout_ms: int = 30000
    retry_attempts: int = 3
    health_check_interval: int = 30


@dataclass
class UnifiedEvent:
    """
    Unified event format for event bus communication.
    
    Events are special messages for pub/sub patterns and system notifications.
    """
    
    # Core identification
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Event details
    event_type: str = ""                # Event type/name
    source: str = ""                    # Event source
    topic: str = ""                     # Event topic for routing
    
    # Event data
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Event properties
    priority: Priority = Priority.MEDIUM
    ttl: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    
    def to_message(self) -> UnifiedMessage:
        """Convert event to unified message."""
        return UnifiedMessage(
            id=self.id,
            timestamp=self.timestamp,
            source=self.source,
            routing_key=self.topic,
            message_type=MessageType.EVENT_PUBLISH,
            priority=self.priority,
            ttl=self.ttl,
            headers=self.metadata,
            payload={
                "event_type": self.event_type,
                "topic": self.topic,
                "data": self.data,
                "tags": self.tags
            }
        )
    
    @classmethod
    def from_message(cls, message: UnifiedMessage) -> "UnifiedEvent":
        """Create event from unified message."""
        payload = message.payload
        return cls(
            id=message.id,
            timestamp=message.timestamp,
            event_type=payload.get("event_type", ""),
            source=message.source,
            topic=payload.get("topic", message.routing_key or ""),
            data=payload.get("data", {}),
            metadata=message.headers,
            priority=message.priority,
            ttl=message.ttl,
            tags=payload.get("tags", [])
        )


@dataclass
class MessageResult:
    """Result of message send operation."""
    success: bool
    message_id: str
    protocol_used: Optional[ProtocolType] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SubscriptionResult:
    """Result of subscription operation."""
    success: bool
    subscription_id: str
    pattern: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventResult:
    """Result of event publication."""
    success: bool
    event_id: str
    subscribers_notified: int = 0
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Message type hierarchies for pattern matching
TASK_MESSAGES = {
    MessageType.TASK_REQUEST,
    MessageType.TASK_RESPONSE,
    MessageType.TASK_ASSIGNMENT,
    MessageType.TASK_COMPLETION,
    MessageType.TASK_FAILURE,
    MessageType.TASK_CANCELLATION
}

AGENT_MESSAGES = {
    MessageType.AGENT_HEARTBEAT,
    MessageType.AGENT_REGISTRATION,
    MessageType.AGENT_DEREGISTRATION,
    MessageType.AGENT_STATUS_UPDATE,
    MessageType.COORDINATION_REQUEST,
    MessageType.COORDINATION_RESPONSE,
    MessageType.SYNC_POINT
}

EVENT_MESSAGES = {
    MessageType.EVENT_PUBLISH,
    MessageType.EVENT_SUBSCRIBE,
    MessageType.EVENT_UNSUBSCRIBE,
    MessageType.SYSTEM_NOTIFICATION,
    MessageType.ERROR_NOTIFICATION
}

REALTIME_MESSAGES = {
    MessageType.REALTIME_UPDATE,
    MessageType.BROADCAST,
    MessageType.MULTICAST
}

SYSTEM_MESSAGES = {
    MessageType.HEALTH_CHECK,
    MessageType.CONFIG_UPDATE,
    MessageType.SYSTEM_SHUTDOWN,
    MessageType.SYSTEM_RESTART
}

HOOK_MESSAGES = {
    MessageType.HOOK_PRE_TOOL_USE,
    MessageType.HOOK_POST_TOOL_USE,
    MessageType.HOOK_ERROR,
    MessageType.LIFECYCLE_START,
    MessageType.LIFECYCLE_STOP
}

KNOWLEDGE_MESSAGES = {
    MessageType.KNOWLEDGE_SHARE,
    MessageType.CONTEXT_UPDATE,
    MessageType.MEMORY_SYNC
}

# Priority-based message categories
CRITICAL_MESSAGES = {
    MessageType.SYSTEM_SHUTDOWN,
    MessageType.SYSTEM_RESTART,
    MessageType.ERROR_NOTIFICATION
}

HIGH_PRIORITY_MESSAGES = {
    MessageType.HEALTH_CHECK,
    MessageType.AGENT_HEARTBEAT,
    MessageType.COORDINATION_REQUEST,
    MessageType.TASK_FAILURE
}

# Delivery guarantee defaults by message type
DELIVERY_GUARANTEE_DEFAULTS = {
    MessageType.SYSTEM_SHUTDOWN: DeliveryGuarantee.EXACTLY_ONCE,
    MessageType.TASK_ASSIGNMENT: DeliveryGuarantee.AT_LEAST_ONCE,
    MessageType.TASK_COMPLETION: DeliveryGuarantee.AT_LEAST_ONCE,
    MessageType.AGENT_HEARTBEAT: DeliveryGuarantee.BEST_EFFORT,
    MessageType.REALTIME_UPDATE: DeliveryGuarantee.BEST_EFFORT,
    MessageType.BROADCAST: DeliveryGuarantee.BEST_EFFORT
}

# TTL defaults by message type (in milliseconds)
TTL_DEFAULTS = {
    MessageType.AGENT_HEARTBEAT: 30000,      # 30 seconds
    MessageType.REALTIME_UPDATE: 10000,      # 10 seconds
    MessageType.HEALTH_CHECK: 60000,         # 1 minute
    MessageType.TASK_REQUEST: 300000,        # 5 minutes
    MessageType.SYSTEM_NOTIFICATION: 600000, # 10 minutes
}


def create_message(
    source: str,
    destination: str,
    message_type: MessageType,
    payload: Dict[str, Any],
    **kwargs
) -> UnifiedMessage:
    """
    Convenience function to create a standardized message.
    
    Args:
        source: Source component/agent ID
        destination: Target component/agent ID
        message_type: Type of message
        payload: Message payload data
        **kwargs: Additional message properties
    
    Returns:
        UnifiedMessage with defaults applied based on message type
    """
    # Apply message type defaults
    defaults = {
        "priority": Priority.HIGH if message_type in HIGH_PRIORITY_MESSAGES else Priority.MEDIUM,
        "delivery_guarantee": DELIVERY_GUARANTEE_DEFAULTS.get(message_type, DeliveryGuarantee.BEST_EFFORT),
        "ttl": TTL_DEFAULTS.get(message_type)
    }
    
    # Override defaults with provided kwargs
    defaults.update(kwargs)
    
    return UnifiedMessage(
        source=source,
        destination=destination,
        message_type=message_type,
        payload=payload,
        **defaults
    )


def create_event(
    source: str,
    event_type: str,
    topic: str,
    data: Dict[str, Any],
    **kwargs
) -> UnifiedEvent:
    """
    Convenience function to create a standardized event.
    
    Args:
        source: Event source
        event_type: Type of event
        topic: Event topic for routing
        data: Event data
        **kwargs: Additional event properties
    
    Returns:
        UnifiedEvent with appropriate defaults
    """
    return UnifiedEvent(
        source=source,
        event_type=event_type,
        topic=topic,
        data=data,
        **kwargs
    )