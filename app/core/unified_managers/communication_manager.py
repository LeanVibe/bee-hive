#!/usr/bin/env python3
"""
CommunicationManager - Messaging and Event Consolidation
Phase 2.1 Implementation of Technical Debt Remediation Plan

This manager consolidates all messaging, event handling, and communication patterns
into a unified, high-performance system built on the BaseManager framework.

TARGET CONSOLIDATION: 15+ communication-related manager classes → 1 unified CommunicationManager
- Message routing and delivery
- Event bus and pub/sub patterns
- WebSocket and real-time communication
- Queue management and processing
- Notification systems
- Inter-agent communication
- Protocol abstraction
- Message persistence and reliability
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable, Protocol as TypingProtocol
from dataclasses import dataclass, field
from enum import Enum
import threading
from contextlib import asynccontextmanager
from collections import defaultdict, deque

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


class MessagePriority(str, Enum):
    """Message priority levels for routing and processing."""
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    BACKGROUND = "background"


class MessageType(str, Enum):
    """Types of messages handled by the communication system."""
    COMMAND = "command"
    QUERY = "query"
    EVENT = "event"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    RESPONSE = "response"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class DeliveryMode(str, Enum):
    """Message delivery modes."""
    FIRE_AND_FORGET = "fire_and_forget"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"


class CommunicationProtocol(str, Enum):
    """Supported communication protocols."""
    IN_MEMORY = "in_memory"
    WEBSOCKET = "websocket"
    HTTP = "http"
    TCP = "tcp"
    UDP = "udp"
    MESSAGE_QUEUE = "message_queue"
    EVENT_BUS = "event_bus"


@dataclass
class Message:
    """Unified message structure for all communication."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.COMMAND
    priority: MessagePriority = MessagePriority.NORMAL
    sender_id: str = ""
    recipient_id: Optional[str] = None
    topic: Optional[str] = None
    content: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    delivery_mode: DeliveryMode = DeliveryMode.FIRE_AND_FORGET
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    protocol: CommunicationProtocol = CommunicationProtocol.IN_MEMORY
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.value,
            "priority": self.priority.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "topic": self.topic,
            "content": self.content,
            "headers": self.headers,
            "timestamp": self.timestamp.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "delivery_mode": self.delivery_mode.value,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "protocol": self.protocol.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        message = cls()
        message.id = data.get("id", message.id)
        message.type = MessageType(data.get("type", MessageType.COMMAND.value))
        message.priority = MessagePriority(data.get("priority", MessagePriority.NORMAL.value))
        message.sender_id = data.get("sender_id", "")
        message.recipient_id = data.get("recipient_id")
        message.topic = data.get("topic")
        message.content = data.get("content", {})
        message.headers = data.get("headers", {})
        
        if "timestamp" in data:
            message.timestamp = datetime.fromisoformat(data["timestamp"])
        
        if data.get("expires_at"):
            message.expires_at = datetime.fromisoformat(data["expires_at"])
        
        message.delivery_mode = DeliveryMode(data.get("delivery_mode", DeliveryMode.FIRE_AND_FORGET.value))
        message.correlation_id = data.get("correlation_id")
        message.reply_to = data.get("reply_to")
        message.retry_count = data.get("retry_count", 0)
        message.max_retries = data.get("max_retries", 3)
        message.protocol = CommunicationProtocol(data.get("protocol", CommunicationProtocol.IN_MEMORY.value))
        
        return message


@dataclass
class MessageRoute:
    """Message routing configuration."""
    pattern: str  # Topic pattern or recipient pattern
    handler: Callable[[Message], Any]
    priority: int = 0
    conditions: Dict[str, Any] = field(default_factory=dict)
    rate_limit: Optional[int] = None  # Messages per second
    timeout: float = 30.0  # Handler timeout in seconds


@dataclass
class CommunicationMetrics:
    """Communication-specific metrics."""
    messages_sent: int = 0
    messages_received: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    messages_expired: int = 0
    messages_retried: int = 0
    avg_processing_time_ms: float = 0.0
    avg_delivery_time_ms: float = 0.0
    active_connections: int = 0
    total_connections: int = 0
    failed_connections: int = 0
    messages_by_type: Dict[MessageType, int] = field(default_factory=dict)
    messages_by_priority: Dict[MessagePriority, int] = field(default_factory=dict)
    messages_by_protocol: Dict[CommunicationProtocol, int] = field(default_factory=dict)


class MessageHandler(TypingProtocol):
    """Protocol for message handlers."""
    
    async def handle(self, message: Message) -> Any:
        """Handle a message."""
        ...
    
    def can_handle(self, message: Message) -> bool:
        """Check if this handler can process the message."""
        ...


class CommunicationPlugin(PluginInterface):
    """Base class for communication plugins."""
    
    @property
    def plugin_type(self) -> PluginType:
        return PluginType.COMMUNICATION
    
    async def pre_send_hook(self, message: Message) -> Message:
        """Hook called before sending a message."""
        return message
    
    async def post_send_hook(self, message: Message, result: Any) -> None:
        """Hook called after sending a message."""
        pass
    
    async def pre_receive_hook(self, message: Message) -> Message:
        """Hook called before processing a received message."""
        return message
    
    async def post_receive_hook(self, message: Message, result: Any) -> None:
        """Hook called after processing a received message."""
        pass


class CommunicationManager(BaseManager):
    """
    Unified manager for all messaging and communication operations.
    
    CONSOLIDATION TARGET: Replaces 15+ specialized communication managers:
    - MessageRouter
    - EventBus
    - WebSocketManager
    - QueueManager
    - NotificationManager
    - BroadcastManager
    - PubSubManager
    - MessagePersistence
    - ConnectionManager
    - ProtocolManager
    - ReliabilityManager
    - RateLimitManager
    - MessageSerializer
    - DeliveryManager
    - CommunicationMonitor
    
    Built on BaseManager framework with Phase 2 enhancements.
    """
    
    def __init__(self, config: Optional[ManagerConfig] = None):
        # Create default config if none provided
        if config is None:
            config = ManagerConfig(
                name="CommunicationManager",
                domain=ManagerDomain.COMMUNICATION,
                max_concurrent_operations=500,
                health_check_interval=20,
                circuit_breaker_enabled=True,
                circuit_breaker_failure_threshold=10
            )
        
        super().__init__(config)
        
        # Communication-specific state
        self.routes: List[MessageRoute] = []
        self.handlers: Dict[str, MessageHandler] = {}
        self.subscribers: Dict[str, Set[Callable]] = defaultdict(set)  # topic -> subscribers
        self.connections: Dict[str, Any] = {}  # connection_id -> connection object
        self.message_queues: Dict[MessagePriority, asyncio.Queue] = {}
        self.pending_messages: Dict[str, Message] = {}  # message_id -> message
        self.communication_metrics = CommunicationMetrics()
        
        # Rate limiting
        self.rate_limiters: Dict[str, deque] = defaultdict(deque)  # handler_id -> request timestamps
        
        # Background processing
        self._processor_tasks: List[asyncio.Task] = []
        self._connection_monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._routes_lock = threading.RLock()
        self._handlers_lock = threading.RLock()
        self._connections_lock = threading.RLock()
        
        # Initialize priority queues
        for priority in MessagePriority:
            self.message_queues[priority] = asyncio.Queue(maxsize=1000)
        
        self.logger = standard_logging_setup(
            name="CommunicationManager",
            level="INFO"
        )
    
    # BaseManager Implementation
    
    async def _setup(self) -> None:
        """Initialize communication systems."""
        self.logger.info("Setting up CommunicationManager")
        
        # Start message processors for each priority level
        for priority in MessagePriority:
            processor_task = asyncio.create_task(
                self._message_processor_loop(priority)
            )
            self._processor_tasks.append(processor_task)
        
        # Start background tasks
        self._connection_monitor_task = asyncio.create_task(self._connection_monitor_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("CommunicationManager setup completed")
    
    async def _cleanup(self) -> None:
        """Clean up communication systems."""
        self.logger.info("Cleaning up CommunicationManager")
        
        # Cancel processor tasks
        for task in self._processor_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Cancel background tasks
        for task in [self._connection_monitor_task, self._cleanup_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close all connections
        await self._close_all_connections()
        
        # Clear queues
        for queue in self.message_queues.values():
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        
        self.logger.info("CommunicationManager cleanup completed")
    
    async def _health_check_internal(self) -> Dict[str, Any]:
        """Communication-specific health check."""
        # Check queue sizes
        queue_sizes = {}
        total_queued = 0
        for priority, queue in self.message_queues.items():
            size = queue.qsize()
            queue_sizes[priority.value] = size
            total_queued += size
        
        # Check connection health
        with self._connections_lock:
            active_connections = len(self.connections)
        
        return {
            "total_queued_messages": total_queued,
            "queue_sizes": queue_sizes,
            "active_connections": active_connections,
            "registered_handlers": len(self.handlers),
            "registered_routes": len(self.routes),
            "total_subscribers": sum(len(subs) for subs in self.subscribers.values()),
            "communication_metrics": {
                "messages_sent": self.communication_metrics.messages_sent,
                "messages_received": self.communication_metrics.messages_received,
                "messages_processed": self.communication_metrics.messages_processed,
                "messages_failed": self.communication_metrics.messages_failed,
                "avg_processing_time_ms": self.communication_metrics.avg_processing_time_ms
            }
        }
    
    # Core Communication Operations
    
    async def send_message(
        self,
        message: Message,
        wait_for_response: bool = False,
        timeout: float = 30.0
    ) -> Optional[Any]:
        """
        Send a message through the communication system.
        
        CONSOLIDATES: MessageSender, EventPublisher, NotificationSender patterns
        """
        async with self.execute_with_monitoring("send_message"):
            start_time = time.time()
            
            try:
                # Pre-send hooks
                for plugin in self.plugins.values():
                    if isinstance(plugin, CommunicationPlugin):
                        message = await plugin.pre_send_hook(message)
                
                # Validate message
                if message.is_expired():
                    raise ValueError(f"Message {message.id} has expired")
                
                # Set sender info if not provided
                if not message.sender_id:
                    message.sender_id = f"communication_manager_{self.config.name}"
                
                # Add to pending if expecting response
                if wait_for_response:
                    self.pending_messages[message.id] = message
                
                # Route message based on delivery mode
                if message.delivery_mode == DeliveryMode.BROADCAST:
                    result = await self._broadcast_message(message)
                elif message.delivery_mode == DeliveryMode.MULTICAST:
                    result = await self._multicast_message(message)
                else:
                    result = await self._route_message(message)
                
                # Post-send hooks
                for plugin in self.plugins.values():
                    if isinstance(plugin, CommunicationPlugin):
                        await plugin.post_send_hook(message, result)
                
                # Update metrics
                send_time_ms = (time.time() - start_time) * 1000
                self.communication_metrics.messages_sent += 1
                self._update_delivery_time_metrics(send_time_ms)
                self._update_message_type_metrics(message.type)
                
                # Wait for response if requested
                if wait_for_response:
                    try:
                        response = await asyncio.wait_for(
                            self._wait_for_response(message.id),
                            timeout=timeout
                        )
                        return response
                    finally:
                        # Cleanup pending message
                        self.pending_messages.pop(message.id, None)
                
                return result
                
            except Exception as e:
                self.communication_metrics.messages_failed += 1
                self.logger.error(f"Failed to send message: {e}", message_id=message.id)
                
                # Cleanup pending message
                self.pending_messages.pop(message.id, None)
                
                # Retry logic
                if message.retry_count < message.max_retries:
                    message.retry_count += 1
                    self.communication_metrics.messages_retried += 1
                    
                    # Queue for retry with backoff
                    await asyncio.sleep(min(2 ** message.retry_count, 30))  # Exponential backoff
                    return await self.send_message(message, wait_for_response, timeout)
                
                raise
    
    async def register_handler(
        self,
        pattern: str,
        handler: Union[MessageHandler, Callable[[Message], Any]],
        priority: int = 0,
        rate_limit: Optional[int] = None,
        conditions: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a message handler.
        
        CONSOLIDATES: HandlerRegistry, RouteManager patterns
        """
        async with self.execute_with_monitoring("register_handler"):
            handler_id = str(uuid.uuid4())
            
            with self._routes_lock:
                route = MessageRoute(
                    pattern=pattern,
                    handler=handler,
                    priority=priority,
                    rate_limit=rate_limit,
                    conditions=conditions or {}
                )
                self.routes.append(route)
                self.routes.sort(key=lambda r: r.priority, reverse=True)  # Higher priority first
            
            self.logger.info(
                f"Handler registered",
                handler_id=handler_id,
                pattern=pattern,
                priority=priority
            )
            
            return handler_id
    
    async def subscribe(self, topic: str, callback: Callable[[Message], Any]) -> str:
        """
        Subscribe to a topic for pub/sub messaging.
        
        CONSOLIDATES: PubSubManager, EventBus subscription patterns
        """
        async with self.execute_with_monitoring("subscribe"):
            subscription_id = str(uuid.uuid4())
            
            self.subscribers[topic].add(callback)
            
            self.logger.info(f"Subscribed to topic", topic=topic, subscription_id=subscription_id)
            
            return subscription_id
    
    async def unsubscribe(self, topic: str, callback: Callable[[Message], Any]) -> bool:
        """Unsubscribe from a topic."""
        async with self.execute_with_monitoring("unsubscribe"):
            if topic in self.subscribers:
                self.subscribers[topic].discard(callback)
                
                # Remove topic if no subscribers left
                if not self.subscribers[topic]:
                    del self.subscribers[topic]
                
                self.logger.info(f"Unsubscribed from topic", topic=topic)
                return True
            
            return False
    
    async def publish(self, topic: str, content: Dict[str, Any], **kwargs) -> None:
        """
        Publish a message to a topic.
        
        CONSOLIDATES: EventPublisher, TopicManager patterns
        """
        message = Message(
            type=MessageType.EVENT,
            topic=topic,
            content=content,
            delivery_mode=DeliveryMode.BROADCAST,
            **kwargs
        )
        
        await self.send_message(message)
    
    async def create_connection(
        self,
        connection_id: str,
        protocol: CommunicationProtocol,
        endpoint: str,
        options: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a new communication connection.
        
        CONSOLIDATES: ConnectionManager, ProtocolManager patterns
        """
        async with self.execute_with_monitoring("create_connection"):
            try:
                connection = await self._establish_connection(protocol, endpoint, options or {})
                
                with self._connections_lock:
                    self.connections[connection_id] = connection
                
                self.communication_metrics.active_connections += 1
                self.communication_metrics.total_connections += 1
                
                self.logger.info(
                    f"Connection established",
                    connection_id=connection_id,
                    protocol=protocol.value,
                    endpoint=endpoint
                )
                
                return True
                
            except Exception as e:
                self.communication_metrics.failed_connections += 1
                self.logger.error(f"Failed to create connection: {e}", connection_id=connection_id)
                return False
    
    # Private Implementation Methods
    
    async def _route_message(self, message: Message) -> Any:
        """Route message to appropriate handlers."""
        matched_routes = []
        
        with self._routes_lock:
            for route in self.routes:
                if self._matches_route(message, route):
                    matched_routes.append(route)
        
        if not matched_routes:
            raise ValueError(f"No route found for message {message.id}")
        
        # Process routes in priority order
        results = []
        for route in matched_routes:
            try:
                # Check rate limiting
                if route.rate_limit and not self._check_rate_limit(route, message):
                    self.logger.warning(f"Rate limit exceeded for route", pattern=route.pattern)
                    continue
                
                # Execute handler
                result = await self._execute_handler(route.handler, message, route.timeout)
                results.append(result)
                
                # For single delivery modes, stop after first successful handler
                if message.delivery_mode in [DeliveryMode.FIRE_AND_FORGET, DeliveryMode.AT_LEAST_ONCE]:
                    break
                
            except Exception as e:
                self.logger.error(f"Handler failed: {e}", pattern=route.pattern, message_id=message.id)
                if message.delivery_mode == DeliveryMode.EXACTLY_ONCE:
                    raise
        
        return results[0] if results else None
    
    async def _broadcast_message(self, message: Message) -> List[Any]:
        """Broadcast message to all matching handlers and subscribers."""
        results = []
        
        # Route to handlers
        try:
            handler_result = await self._route_message(message)
            if handler_result is not None:
                results.append(handler_result)
        except ValueError:
            pass  # No handlers is OK for broadcast
        
        # Notify subscribers
        if message.topic and message.topic in self.subscribers:
            subscriber_tasks = []
            for callback in self.subscribers[message.topic]:
                task = asyncio.create_task(self._execute_subscriber(callback, message))
                subscriber_tasks.append(task)
            
            if subscriber_tasks:
                subscriber_results = await asyncio.gather(*subscriber_tasks, return_exceptions=True)
                results.extend([r for r in subscriber_results if not isinstance(r, Exception)])
        
        return results
    
    async def _multicast_message(self, message: Message) -> List[Any]:
        """Multicast message to specific recipients."""
        # For multicast, we would typically have a list of recipients
        # This is a simplified version - can be extended based on needs
        return await self._broadcast_message(message)
    
    def _matches_route(self, message: Message, route: MessageRoute) -> bool:
        """Check if message matches a route."""
        # Pattern matching (simplified - can be extended with regex, etc.)
        pattern_match = (
            route.pattern == "*" or  # Wildcard
            (message.topic and route.pattern == message.topic) or  # Topic match
            (message.recipient_id and route.pattern == message.recipient_id) or  # Recipient match
            route.pattern in str(message.content)  # Content match (simple)
        )
        
        if not pattern_match:
            return False
        
        # Check conditions
        for key, expected_value in route.conditions.items():
            if key == "message_type":
                if message.type.value != expected_value:
                    return False
            elif key == "priority":
                if message.priority.value != expected_value:
                    return False
            elif key == "sender_id":
                if message.sender_id != expected_value:
                    return False
            # Add more condition types as needed
        
        return True
    
    def _check_rate_limit(self, route: MessageRoute, message: Message) -> bool:
        """Check if route is within rate limits."""
        if not route.rate_limit:
            return True
        
        current_time = time.time()
        rate_key = f"{route.pattern}:{message.sender_id}"
        
        # Clean old requests
        request_times = self.rate_limiters[rate_key]
        while request_times and request_times[0] < current_time - 1.0:  # 1 second window
            request_times.popleft()
        
        # Check limit
        if len(request_times) >= route.rate_limit:
            return False
        
        # Record this request
        request_times.append(current_time)
        return True
    
    async def _execute_handler(self, handler: Callable, message: Message, timeout: float) -> Any:
        """Execute a message handler with timeout."""
        start_time = time.time()
        
        try:
            # Execute with timeout
            if asyncio.iscoroutinefunction(handler):
                result = await asyncio.wait_for(handler(message), timeout=timeout)
            else:
                result = handler(message)
            
            # Update metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self.communication_metrics.messages_processed += 1
            self._update_processing_time_metrics(processing_time_ms)
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(f"Handler timeout", timeout=timeout)
            raise
        except Exception as e:
            self.communication_metrics.messages_failed += 1
            raise
    
    async def _execute_subscriber(self, callback: Callable, message: Message) -> Any:
        """Execute a subscriber callback."""
        try:
            if asyncio.iscoroutinefunction(callback):
                return await callback(message)
            else:
                return callback(message)
        except Exception as e:
            self.logger.warning(f"Subscriber callback failed: {e}")
            return None
    
    async def _wait_for_response(self, message_id: str) -> Any:
        """Wait for a response to a message."""
        # This is a simplified implementation
        # In a real system, you might use a response queue or callback mechanism
        timeout = 30.0
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check for response (this would be implemented based on your response mechanism)
            await asyncio.sleep(0.1)
        
        raise asyncio.TimeoutError(f"No response received for message {message_id}")
    
    # Background Processing
    
    async def _message_processor_loop(self, priority: MessagePriority) -> None:
        """Process messages for a specific priority level."""
        queue = self.message_queues[priority]
        
        while not self._shutdown_event.is_set():
            try:
                # Get message from queue with timeout
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                if self._shutdown_event.is_set():
                    break
                
                # Process message
                try:
                    await self._route_message(message)
                except Exception as e:
                    self.logger.error(f"Message processing failed: {e}", message_id=message.id)
                finally:
                    queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Message processor loop error: {e}")
    
    async def _connection_monitor_loop(self) -> None:
        """Monitor connection health."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                if self._shutdown_event.is_set():
                    break
                
                # Check connection health
                connections_to_check = []
                with self._connections_lock:
                    connections_to_check = list(self.connections.items())
                
                for conn_id, connection in connections_to_check:
                    try:
                        # Check if connection is still healthy
                        if hasattr(connection, 'is_healthy') and not await connection.is_healthy():
                            self.logger.warning(f"Unhealthy connection detected", connection_id=conn_id)
                            # Attempt to reconnect or remove
                            await self._handle_unhealthy_connection(conn_id, connection)
                    except Exception as e:
                        self.logger.error(f"Connection health check failed: {e}", connection_id=conn_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Connection monitor loop error: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of expired messages and stale data."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Run every minute
                if self._shutdown_event.is_set():
                    break
                
                current_time = datetime.utcnow()
                
                # Clean up expired pending messages
                expired_message_ids = []
                for message_id, message in self.pending_messages.items():
                    if message.is_expired():
                        expired_message_ids.append(message_id)
                
                for message_id in expired_message_ids:
                    del self.pending_messages[message_id]
                    self.communication_metrics.messages_expired += 1
                
                if expired_message_ids:
                    self.logger.debug(f"Cleaned up {len(expired_message_ids)} expired messages")
                
                # Clean up old rate limit data
                cutoff_time = time.time() - 3600  # 1 hour ago
                for rate_key, request_times in self.rate_limiters.items():
                    while request_times and request_times[0] < cutoff_time:
                        request_times.popleft()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
    
    # Connection Management
    
    async def _establish_connection(
        self,
        protocol: CommunicationProtocol,
        endpoint: str,
        options: Dict[str, Any]
    ) -> Any:
        """Establish connection based on protocol."""
        if protocol == CommunicationProtocol.IN_MEMORY:
            return {"type": "in_memory", "endpoint": endpoint}
        elif protocol == CommunicationProtocol.WEBSOCKET:
            # WebSocket connection logic would go here
            return {"type": "websocket", "endpoint": endpoint, "options": options}
        elif protocol == CommunicationProtocol.HTTP:
            # HTTP client setup would go here
            return {"type": "http", "endpoint": endpoint, "options": options}
        # Add other protocols as needed
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
    
    async def _handle_unhealthy_connection(self, connection_id: str, connection: Any) -> None:
        """Handle unhealthy connections."""
        try:
            # Attempt to close gracefully
            if hasattr(connection, 'close'):
                await connection.close()
        except Exception:
            pass
        
        # Remove from active connections
        with self._connections_lock:
            if connection_id in self.connections:
                del self.connections[connection_id]
                self.communication_metrics.active_connections -= 1
    
    async def _close_all_connections(self) -> None:
        """Close all connections during shutdown."""
        connections_to_close = []
        with self._connections_lock:
            connections_to_close = list(self.connections.items())
            self.connections.clear()
        
        for conn_id, connection in connections_to_close:
            try:
                if hasattr(connection, 'close'):
                    if asyncio.iscoroutinefunction(connection.close):
                        await connection.close()
                    else:
                        connection.close()
            except Exception as e:
                self.logger.warning(f"Failed to close connection: {e}", connection_id=conn_id)
        
        self.communication_metrics.active_connections = 0
    
    # Metrics Helpers
    
    def _update_processing_time_metrics(self, processing_time_ms: float) -> None:
        """Update processing time metrics."""
        processed = self.communication_metrics.messages_processed
        current_avg = self.communication_metrics.avg_processing_time_ms
        
        if processed == 1:
            self.communication_metrics.avg_processing_time_ms = processing_time_ms
        else:
            self.communication_metrics.avg_processing_time_ms = (
                (current_avg * (processed - 1) + processing_time_ms) / processed
            )
    
    def _update_delivery_time_metrics(self, delivery_time_ms: float) -> None:
        """Update delivery time metrics."""
        sent = self.communication_metrics.messages_sent
        current_avg = self.communication_metrics.avg_delivery_time_ms
        
        if sent == 1:
            self.communication_metrics.avg_delivery_time_ms = delivery_time_ms
        else:
            self.communication_metrics.avg_delivery_time_ms = (
                (current_avg * (sent - 1) + delivery_time_ms) / sent
            )
    
    def _update_message_type_metrics(self, message_type: MessageType) -> None:
        """Update message type metrics."""
        current = self.communication_metrics.messages_by_type.get(message_type, 0)
        self.communication_metrics.messages_by_type[message_type] = current + 1
    
    # Public API Extensions
    
    def get_communication_metrics(self) -> CommunicationMetrics:
        """Get current communication metrics."""
        return self.communication_metrics
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get status of all message queues."""
        queue_status = {}
        for priority, queue in self.message_queues.items():
            queue_status[priority.value] = {
                "size": queue.qsize(),
                "maxsize": queue.maxsize,
                "full": queue.full()
            }
        return queue_status
    
    async def clear_queue(self, priority: MessagePriority) -> int:
        """Clear messages from a specific priority queue."""
        queue = self.message_queues[priority]
        cleared_count = 0
        
        while not queue.empty():
            try:
                queue.get_nowait()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break
        
        return cleared_count


# Plugin Examples

class MessageLoggingPlugin(CommunicationPlugin):
    """Plugin for logging all messages."""
    
    @property
    def name(self) -> str:
        return "MessageLogging"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def initialize(self, manager: BaseManager) -> None:
        self.logger = structlog.get_logger(f"{self.name}")
    
    async def cleanup(self) -> None:
        pass
    
    async def pre_send_hook(self, message: Message) -> Message:
        self.logger.info(
            "Message sending",
            message_id=message.id,
            type=message.type.value,
            sender=message.sender_id,
            recipient=message.recipient_id
        )
        return message
    
    async def post_receive_hook(self, message: Message, result: Any) -> None:
        self.logger.info(
            "Message processed",
            message_id=message.id,
            success=result is not None
        )


class MessageEncryptionPlugin(CommunicationPlugin):
    """Plugin for message encryption."""
    
    @property
    def name(self) -> str:
        return "MessageEncryption"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def initialize(self, manager: BaseManager) -> None:
        # Initialize encryption keys and algorithms
        pass
    
    async def cleanup(self) -> None:
        pass
    
    async def pre_send_hook(self, message: Message) -> Message:
        # Encrypt message content
        # This is a placeholder - implement actual encryption
        if message.headers.get("encrypt") == "true":
            # encrypted_content = encrypt(message.content)
            # message.content = {"encrypted": encrypted_content}
            message.headers["encrypted"] = "true"
        return message
    
    async def pre_receive_hook(self, message: Message) -> Message:
        # Decrypt message content
        if message.headers.get("encrypted") == "true":
            # decrypted_content = decrypt(message.content["encrypted"])
            # message.content = decrypted_content
            pass
        return message