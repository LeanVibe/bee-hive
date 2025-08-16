"""
Unified Messaging Service for LeanVibe Agent Hive
Consolidates multiple messaging implementations into reliable communication infrastructure

Epic 1, Phase 1 Week 2: Messaging Service Infrastructure Consolidation
- Consolidates 5 separate messaging implementations into a single service
- Supports all communication patterns: agent-to-agent, events, broadcasts, priority queuing
- Integrates with circuit breaker, logging, and configuration services
- Provides foundation for multi-agent coordination
"""

from typing import Optional, Dict, Any, List, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import asyncio
import json
import uuid
import time
import hmac
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict, deque
import weakref

# Core infrastructure imports
from app.core.logging_service import get_component_logger
from app.core.configuration_service import ConfigurationService

# Optional circuit breaker import
try:
    from app.core.circuit_breaker import CircuitBreakerService
    CIRCUIT_BREAKER_AVAILABLE = True
except (ImportError, NameError, AttributeError, Exception):
    CIRCUIT_BREAKER_AVAILABLE = False
    CircuitBreakerService = None

# Redis and existing messaging imports
import redis.asyncio as redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError, ResponseError

logger = get_component_logger("messaging")


class MessageType(str, Enum):
    """Unified message types supporting all communication patterns"""
    # Task and coordination messages
    REQUEST = "request"
    RESPONSE = "response"
    TASK_REQUEST = "task_request"
    TASK_RESULT = "task_result"
    TASK_ASSIGNMENT = "task_assignment"
    TASK_COMPLETION = "task_completion"
    TASK_FAILURE = "task_failure"
    
    # System and lifecycle messages
    EVENT = "event"
    BROADCAST = "broadcast"
    COMMAND = "command"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_REQUEST = "heartbeat_request"
    HEARTBEAT_RESPONSE = "heartbeat_response"
    
    # Agent lifecycle
    AGENT_REGISTERED = "agent_registered"
    AGENT_DEREGISTERED = "agent_deregistered"
    STATUS_UPDATE = "status_update"
    
    # System management
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_UPDATE = "config_update"
    ERROR = "error"
    
    # Hook messages
    HOOK_PRE_TOOL_USE = "hook_pre_tool_use"
    HOOK_POST_TOOL_USE = "hook_post_tool_use"
    HOOK_ERROR = "hook_error"


class MessagePriority(IntEnum):
    """Priority levels for message processing (lower number = higher priority)"""
    CRITICAL = 1
    URGENT = 2
    HIGH = 3
    NORMAL = 4
    LOW = 5


class RoutingStrategy(str, Enum):
    """Message routing strategies"""
    DIRECT = "direct"           # Point-to-point delivery
    ROUND_ROBIN = "round_robin" # Load balanced delivery
    BROADCAST = "broadcast"     # Send to all subscribers
    TOPIC = "topic"            # Topic-based routing
    CONSUMER_GROUP = "consumer_group"  # Consumer group routing


class MessageStatus(str, Enum):
    """Message processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"
    DEAD_LETTER = "dead_letter"


@dataclass
class Message:
    """Unified message format consolidating all messaging patterns"""
    # Core message fields
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.REQUEST
    priority: MessagePriority = MessagePriority.NORMAL
    sender: str = "unknown"
    recipient: Optional[str] = None
    topic: Optional[str] = None
    
    # Message content
    payload: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    
    # Timing and lifecycle
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[int] = None  # TTL in seconds
    expires_at: Optional[datetime] = None
    
    # Routing and correlation
    routing_strategy: RoutingStrategy = RoutingStrategy.DIRECT
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    
    # Processing tracking
    retry_count: int = 0
    max_retries: int = 3
    status: MessageStatus = MessageStatus.PENDING
    
    # Security
    signature: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_redis_dict(self) -> Dict[str, str]:
        """Convert message to Redis-compatible dictionary"""
        data = {
            "id": self.id,
            "type": self.type.value,
            "priority": str(self.priority.value),
            "sender": self.sender,
            "recipient": self.recipient or "",
            "topic": self.topic or "",
            "payload": json.dumps(self.payload),
            "headers": json.dumps(self.headers),
            "timestamp": str(self.timestamp),
            "routing_strategy": self.routing_strategy.value,
            "correlation_id": self.correlation_id or "",
            "reply_to": self.reply_to or "",
            "retry_count": str(self.retry_count),
            "max_retries": str(self.max_retries),
            "status": self.status.value,
            "signature": self.signature or "",
            "metadata": json.dumps(self.metadata),
            "created_at": self.created_at.isoformat()
        }
        
        if self.ttl:
            data["ttl"] = str(self.ttl)
        if self.expires_at:
            data["expires_at"] = self.expires_at.isoformat()
            
        return data
    
    @classmethod
    def from_redis_dict(cls, data: Dict[str, str]) -> "Message":
        """Create message from Redis dictionary"""
        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"])
        
        return cls(
            id=data["id"],
            type=MessageType(data["type"]),
            priority=MessagePriority(int(data["priority"])),
            sender=data["sender"],
            recipient=data["recipient"] if data["recipient"] else None,
            topic=data["topic"] if data["topic"] else None,
            payload=json.loads(data["payload"]),
            headers=json.loads(data.get("headers", "{}")),
            timestamp=float(data["timestamp"]),
            ttl=int(data["ttl"]) if data.get("ttl") else None,
            expires_at=expires_at,
            routing_strategy=RoutingStrategy(data.get("routing_strategy", "direct")),
            correlation_id=data.get("correlation_id") or None,
            reply_to=data.get("reply_to") or None,
            retry_count=int(data.get("retry_count", 0)),
            max_retries=int(data.get("max_retries", 3)),
            status=MessageStatus(data.get("status", "pending")),
            signature=data.get("signature") or None,
            metadata=json.loads(data.get("metadata", "{}")),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat()))
        )
    
    def sign(self, secret_key: str) -> None:
        """Sign message with HMAC for authenticity"""
        payload_str = json.dumps(self.payload, sort_keys=True)
        message_data = f"{self.id}{self.sender}{self.recipient or ''}{self.type.value}{payload_str}{self.timestamp}"
        
        self.signature = hmac.new(
            secret_key.encode(),
            message_data.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def verify_signature(self, secret_key: str) -> bool:
        """Verify message signature"""
        if not self.signature:
            return False
            
        original_signature = self.signature
        self.signature = None
        
        try:
            self.sign(secret_key)
            is_valid = hmac.compare_digest(original_signature, self.signature)
            return is_valid
        finally:
            self.signature = original_signature
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        if self.ttl:
            return time.time() > (self.timestamp + self.ttl)
        return False
    
    def can_retry(self) -> bool:
        """Check if message can be retried"""
        return self.retry_count < self.max_retries and not self.is_expired()
    
    def get_stream_name(self) -> str:
        """Get Redis stream name for this message"""
        if self.recipient:
            return f"agent_messages:{self.recipient}"
        elif self.topic:
            return f"topic_messages:{self.topic}"
        else:
            return "agent_messages:broadcast"
    
    def get_channel_name(self) -> str:
        """Get Redis pub/sub channel name for this message"""
        if self.recipient:
            return f"agent:{self.recipient}"
        elif self.topic:
            return f"topic:{self.topic}"
        else:
            return "broadcast"


class MessageHandler:
    """Base class for message handlers"""
    
    def __init__(self, handler_id: str, pattern: str = "*", message_types: List[MessageType] = None):
        self.handler_id = handler_id
        self.pattern = pattern
        self.message_types = message_types or []
        self.message_count = 0
        self.last_activity = datetime.utcnow()
        self.processing_times = deque(maxlen=1000)
    
    async def handle_message(self, message: Message) -> Optional[Message]:
        """Handle incoming message and optionally return response"""
        start_time = time.time()
        self.message_count += 1
        self.last_activity = datetime.utcnow()
        
        try:
            # Check if handler supports this message type
            if self.message_types and message.type not in self.message_types:
                logger.debug(f"Handler {self.handler_id} skipping message type {message.type}")
                return None
            
            result = await self._process_message(message)
            
            # Record processing time
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Handler {self.handler_id} failed to process message {message.id}", error=str(e))
            raise
    
    async def _process_message(self, message: Message) -> Optional[Message]:
        """Override this method to implement message processing"""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        avg_processing_time = 0
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        
        return {
            "handler_id": self.handler_id,
            "pattern": self.pattern,
            "message_types": [mt.value for mt in self.message_types],
            "message_count": self.message_count,
            "last_activity": self.last_activity.isoformat(),
            "avg_processing_time_ms": avg_processing_time,
            "recent_processing_times": list(self.processing_times)[-10:]  # Last 10 times
        }


class AgentCommunicationHandler(MessageHandler):
    """Handler for agent-to-agent communication"""
    
    def __init__(self, handler_id: str = "agent_communication"):
        super().__init__(
            handler_id=handler_id, 
            pattern="agent.*",
            message_types=[
                MessageType.REQUEST, MessageType.RESPONSE, MessageType.COMMAND,
                MessageType.TASK_REQUEST, MessageType.TASK_RESULT, MessageType.HEARTBEAT
            ]
        )
    
    async def _process_message(self, message: Message) -> Optional[Message]:
        """Process agent communication messages"""
        logger.debug(f"Processing agent message {message.id} of type {message.type}")
        
        if message.type == MessageType.REQUEST:
            return await self._handle_agent_request(message)
        elif message.type == MessageType.COMMAND:
            return await self._handle_agent_command(message)
        elif message.type == MessageType.HEARTBEAT_REQUEST:
            return await self._handle_heartbeat_request(message)
        elif message.type in [MessageType.TASK_REQUEST, MessageType.TASK_RESULT]:
            await self._handle_task_message(message)
        
        return None
    
    async def _handle_agent_request(self, message: Message) -> Message:
        """Handle agent requests"""
        response_payload = {
            "status": "processed",
            "request_id": message.id,
            "timestamp": datetime.utcnow().isoformat(),
            "processed_by": "agent_communication_handler"
        }
        
        return Message(
            type=MessageType.RESPONSE,
            sender="agent_communication_handler",
            recipient=message.sender,
            payload=response_payload,
            correlation_id=message.id,
            priority=message.priority
        )
    
    async def _handle_agent_command(self, message: Message) -> None:
        """Handle agent commands"""
        logger.info(f"Executing command from {message.sender}", 
                   command=message.payload.get("command"))
    
    async def _handle_heartbeat_request(self, message: Message) -> Message:
        """Handle heartbeat requests"""
        return Message(
            type=MessageType.HEARTBEAT_RESPONSE,
            sender="system",
            recipient=message.sender,
            payload={
                "status": "alive",
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": message.payload.get("request_id")
            },
            correlation_id=message.id,
            priority=MessagePriority.HIGH
        )
    
    async def _handle_task_message(self, message: Message) -> None:
        """Handle task-related messages"""
        logger.info(f"Task message received from {message.sender}",
                   task_id=message.payload.get("task_id"),
                   message_type=message.type.value)


class EventHandler(MessageHandler):
    """Handler for system events and broadcasts"""
    
    def __init__(self, handler_id: str = "event_handler"):
        super().__init__(
            handler_id=handler_id,
            pattern="event.*",
            message_types=[
                MessageType.EVENT, MessageType.BROADCAST, MessageType.SYSTEM_SHUTDOWN,
                MessageType.CONFIG_UPDATE, MessageType.STATUS_UPDATE
            ]
        )
    
    async def _process_message(self, message: Message) -> Optional[Message]:
        """Process event messages"""
        logger.info(f"Processing event {message.type} from {message.sender}",
                   topic=message.topic, event_data=message.payload)
        
        if message.type == MessageType.SYSTEM_SHUTDOWN:
            await self._handle_shutdown_event(message)
        elif message.type == MessageType.CONFIG_UPDATE:
            await self._handle_config_update(message)
        
        return None
    
    async def _handle_shutdown_event(self, message: Message) -> None:
        """Handle system shutdown events"""
        logger.warning("System shutdown event received",
                      reason=message.payload.get("reason"),
                      grace_period=message.payload.get("grace_period_seconds"))
    
    async def _handle_config_update(self, message: Message) -> None:
        """Handle configuration updates"""
        logger.info("Configuration update received",
                   config_key=message.payload.get("config_key"),
                   new_value=message.payload.get("new_value"))


@dataclass
class MessagingMetrics:
    """Comprehensive messaging service metrics"""
    messages_sent: int = 0
    messages_received: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    messages_expired: int = 0
    messages_dead_letter: int = 0
    
    # Performance metrics
    avg_processing_time_ms: float = 0.0
    p95_processing_time_ms: float = 0.0
    p99_processing_time_ms: float = 0.0
    throughput_msg_per_sec: float = 0.0
    
    # Queue metrics
    queue_depth: int = 0
    backlog_age_seconds: float = 0.0
    
    # Handler metrics
    active_handlers: int = 0
    handler_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Connection metrics
    redis_connected: bool = False
    redis_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "messages": {
                "sent": self.messages_sent,
                "received": self.messages_received,
                "processed": self.messages_processed,
                "failed": self.messages_failed,
                "expired": self.messages_expired,
                "dead_letter": self.messages_dead_letter
            },
            "performance": {
                "avg_processing_time_ms": self.avg_processing_time_ms,
                "p95_processing_time_ms": self.p95_processing_time_ms,
                "p99_processing_time_ms": self.p99_processing_time_ms,
                "throughput_msg_per_sec": self.throughput_msg_per_sec
            },
            "queue": {
                "depth": self.queue_depth,
                "backlog_age_seconds": self.backlog_age_seconds
            },
            "handlers": {
                "active_count": self.active_handlers,
                "stats": self.handler_stats
            },
            "connection": {
                "redis_connected": self.redis_connected,
                "redis_latency_ms": self.redis_latency_ms
            }
        }


class MessagingService:
    """
    Unified messaging service supporting all communication patterns:
    - Agent-to-agent communication (consolidated from agent_communication_service.py)
    - Event publishing and subscription (consolidated from communication.py)
    - Request-response patterns (consolidated from message_processor.py)
    - Broadcast messaging (consolidated from agent_messaging_service.py)
    - Topic-based routing with consumer groups
    - Priority message handling with circuit breaker protection
    """
    
    _instance: Optional['MessagingService'] = None
    
    def __new__(cls) -> 'MessagingService':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize_service()
            self._initialized = True
    
    def _initialize_service(self):
        """Initialize messaging service components"""
        try:
            config_service = ConfigurationService()
            self.config = getattr(config_service.config, "messaging", {})
            if not isinstance(self.config, dict):
                self.config = {}
        except Exception:
            self.config = {}
        
        # Initialize circuit breaker if available
        if CIRCUIT_BREAKER_AVAILABLE and CircuitBreakerService:
            try:
                self.circuit_breaker = CircuitBreakerService().get_circuit_breaker("messaging")
            except Exception:
                self.circuit_breaker = None
        else:
            self.circuit_breaker = None
        
        # Redis configuration
        self.redis_url = self.config.get("redis_url", "redis://localhost:6379")
        self.secret_key = self.config.get("secret_key", "default-secret")
        self.max_retries = self.config.get("max_retries", 3)
        self.message_ttl = self.config.get("message_ttl_seconds", 3600)
        
        # Connection management
        self._redis: Optional[redis.Redis] = None
        self._connection_pool = None
        self._connected = False
        
        # Message handling infrastructure
        self._handlers: Dict[str, MessageHandler] = {}
        self._subscribers: Dict[str, List[str]] = {}  # topic -> handler_ids
        self._consumer_groups: Dict[str, Set[str]] = {}  # group_name -> consumer_ids
        
        # Message processing
        self._message_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=10000)
        self._dead_letter_queue: List[Message] = []
        self._processing_tasks: Set[asyncio.Task] = set()
        
        # Request-response handling
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._request_timeout = self.config.get("request_timeout", 30.0)
        
        # Performance tracking
        self._metrics = MessagingMetrics()
        self._processing_times: deque = deque(maxlen=1000)
        self._start_time = time.time()
        
        # Service state
        self._running = False
        self._processor_tasks: List[asyncio.Task] = []
        self._subscription_tasks: Dict[str, asyncio.Task] = {}
        
        # Initialize default handlers
        self._initialize_default_handlers()
        
        logger.info("Messaging service initialized",
                   redis_url=self.redis_url,
                   max_retries=self.max_retries,
                   message_ttl=self.message_ttl)
    
    def _initialize_default_handlers(self):
        """Initialize default message handlers"""
        # Register agent communication handler
        agent_handler = AgentCommunicationHandler()
        self.register_handler(agent_handler)
        
        # Register event handler
        event_handler = EventHandler()
        self.register_handler(event_handler)
        
        logger.info("Default message handlers registered")
    
    async def connect(self) -> None:
        """Connect to Redis with resilience"""
        try:
            self._connection_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True,
                retry_on_error=[RedisConnectionError],
                socket_keepalive=True,
                health_check_interval=30
            )
            
            self._redis = redis.Redis(connection_pool=self._connection_pool)
            
            # Test connection
            start_time = time.time()
            await self._redis.ping()
            self._metrics.redis_latency_ms = (time.time() - start_time) * 1000
            
            self._connected = True
            self._metrics.redis_connected = True
            
            logger.info("Connected to Redis for unified messaging service",
                       latency_ms=self._metrics.redis_latency_ms)
            
        except RedisError as e:
            logger.error("Failed to connect to Redis", error=str(e))
            self._connected = False
            self._metrics.redis_connected = False
            raise ConnectionError(f"Redis connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from Redis and cleanup resources"""
        await self.stop_service()
        
        if self._redis:
            await self._redis.close()
        
        if self._connection_pool:
            await self._connection_pool.disconnect()
        
        self._connected = False
        self._metrics.redis_connected = False
        
        logger.info("Disconnected from unified messaging service")
    
    async def start_service(self) -> None:
        """Start message processing loops"""
        if self._running:
            logger.warning("Messaging service is already running")
            return
        
        if not self._connected:
            await self.connect()
        
        self._running = True
        
        # Start message processor tasks
        for i in range(self.config.get("processor_count", 3)):
            task = asyncio.create_task(self._message_processing_loop(f"processor-{i}"))
            self._processor_tasks.append(task)
        
        # Start subscription management
        asyncio.create_task(self._subscription_management_loop())
        
        logger.info("Messaging service started",
                   processor_count=len(self._processor_tasks))
    
    async def stop_service(self) -> None:
        """Stop message processing and cleanup"""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all processor tasks
        for task in self._processor_tasks:
            if not task.done():
                task.cancel()
        
        # Cancel subscription tasks
        for task in self._subscription_tasks.values():
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        all_tasks = self._processor_tasks + list(self._subscription_tasks.values())
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)
        
        logger.info("Messaging service stopped")
    
    async def _message_processing_loop(self, processor_name: str) -> None:
        """Main message processing loop"""
        logger.info(f"Started message processor: {processor_name}")
        
        while self._running:
            try:
                # Process messages with optional circuit breaker protection
                async def process_next_message():
                    # Get message from priority queue (blocks with timeout)
                    try:
                        priority, sequence, message = await asyncio.wait_for(
                            self._message_queue.get(), timeout=1.0
                        )
                        await self._route_message(message)
                        self._metrics.messages_processed += 1
                    except asyncio.TimeoutError:
                        # No message available, continue loop
                        pass
                
                # Use circuit breaker if available
                if self.circuit_breaker:
                    await self.circuit_breaker(process_next_message)()
                else:
                    await process_next_message()
                
            except Exception as e:
                logger.error(f"Error in message processor {processor_name}", error=str(e))
                self._metrics.messages_failed += 1
                await asyncio.sleep(1)  # Brief delay on error
    
    async def _route_message(self, message: Message) -> None:
        """Route message to appropriate handlers"""
        start_time = time.time()
        
        try:
            # Check if message is expired
            if message.is_expired():
                logger.warning(f"Message {message.id} expired during processing")
                self._metrics.messages_expired += 1
                return
            
            # Find appropriate handlers
            handlers = self._find_handlers_for_message(message)
            
            if not handlers:
                logger.warning(f"No handlers found for message {message.id} type {message.type}")
                return
            
            # Process message with each handler
            for handler in handlers:
                try:
                    response = await handler.handle_message(message)
                    if response:
                        # Queue response message
                        await self.send_message(response)
                        
                except Exception as e:
                    logger.error(f"Handler {handler.handler_id} failed for message {message.id}", 
                               error=str(e))
            
            # Track processing time
            processing_time = (time.time() - start_time) * 1000
            self._processing_times.append(processing_time)
            
            logger.debug(f"Message {message.id} processed by {len(handlers)} handlers",
                        processing_time_ms=processing_time)
            
        except Exception as e:
            logger.error(f"Failed to route message {message.id}", error=str(e))
            self._metrics.messages_failed += 1
    
    def _find_handlers_for_message(self, message: Message) -> List[MessageHandler]:
        """Find handlers that should process this message"""
        handlers = []
        
        for handler in self._handlers.values():
            # Check if handler supports this message type
            if handler.message_types and message.type not in handler.message_types:
                continue
            
            # Check pattern matching (simplified)
            if handler.pattern == "*" or handler.pattern in str(message.type):
                handlers.append(handler)
        
        return handlers
    
    async def _subscription_management_loop(self) -> None:
        """Manage Redis subscriptions for real-time messaging"""
        if not self._redis:
            return
        
        pubsub = self._redis.pubsub()
        
        try:
            # Subscribe to broadcast channel
            await pubsub.subscribe("broadcast")
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        msg = Message.from_redis_dict(data)
                        await self.enqueue_message(msg)
                    except Exception as e:
                        logger.error("Failed to process subscription message", error=str(e))
                        
        except Exception as e:
            logger.error("Error in subscription management loop", error=str(e))
        finally:
            await pubsub.close()
    
    async def send_message(self, message: Message) -> bool:
        """Send message through the messaging system"""
        if not self._connected:
            logger.error("Cannot send message: not connected to Redis")
            return False
        
        try:
            # Sign message if secret key available
            if self.secret_key:
                message.sign(self.secret_key)
            
            # Determine routing strategy
            if message.routing_strategy == RoutingStrategy.DIRECT:
                return await self._send_direct_message(message)
            elif message.routing_strategy == RoutingStrategy.BROADCAST:
                return await self._send_broadcast_message(message)
            elif message.routing_strategy == RoutingStrategy.TOPIC:
                return await self._send_topic_message(message)
            else:
                # Default: queue message for processing
                return await self.enqueue_message(message)
                
        except Exception as e:
            logger.error(f"Failed to send message {message.id}", error=str(e))
            self._metrics.messages_failed += 1
            return False
    
    async def _send_direct_message(self, message: Message) -> bool:
        """Send direct message via Redis Streams"""
        try:
            stream_name = message.get_stream_name()
            redis_data = message.to_redis_dict()
            
            message_id = await self._redis.xadd(
                stream_name,
                redis_data,
                maxlen=10000,
                approximate=True
            )
            
            self._metrics.messages_sent += 1
            logger.debug(f"Direct message sent to {stream_name}: {message_id}")
            return True
            
        except RedisError as e:
            logger.error(f"Failed to send direct message", error=str(e))
            return False
    
    async def _send_broadcast_message(self, message: Message) -> bool:
        """Send broadcast message via Redis Pub/Sub"""
        try:
            channel = message.get_channel_name()
            serialized_message = json.dumps(message.to_redis_dict())
            
            subscriber_count = await self._redis.publish(channel, serialized_message)
            
            self._metrics.messages_sent += 1
            logger.debug(f"Broadcast message sent to {channel}: {subscriber_count} subscribers")
            return True
            
        except RedisError as e:
            logger.error(f"Failed to send broadcast message", error=str(e))
            return False
    
    async def _send_topic_message(self, message: Message) -> bool:
        """Send topic message to subscribers"""
        try:
            # Send to both stream and pub/sub for reliability and real-time delivery
            stream_success = await self._send_direct_message(message)
            pubsub_success = await self._send_broadcast_message(message)
            
            return stream_success or pubsub_success
            
        except Exception as e:
            logger.error(f"Failed to send topic message", error=str(e))
            return False
    
    async def enqueue_message(self, message: Message) -> bool:
        """Add message to processing queue with priority"""
        try:
            # Prepare priority queue entry (priority, sequence, message)
            priority = message.priority.value
            sequence = int(time.time() * 1000000)  # Microsecond precision for ordering
            
            await self._message_queue.put((priority, sequence, message))
            self._metrics.messages_received += 1
            
            logger.debug(f"Message {message.id} queued with priority {message.priority.name}")
            return True
            
        except asyncio.QueueFull:
            logger.warning(f"Message queue full, dropping message {message.id}")
            self._metrics.messages_failed += 1
            return False
        except Exception as e:
            logger.error(f"Failed to queue message {message.id}", error=str(e))
            self._metrics.messages_failed += 1
            return False
    
    async def send_request(self, 
                          recipient: str, 
                          payload: Dict[str, Any],
                          message_type: MessageType = MessageType.REQUEST,
                          timeout: float = None) -> Optional[Message]:
        """Send request and wait for response"""
        timeout = timeout or self._request_timeout
        
        request = Message(
            type=message_type,
            sender="messaging_service",
            recipient=recipient,
            payload=payload,
            routing_strategy=RoutingStrategy.DIRECT
        )
        
        # Set up response waiting mechanism
        response_future = asyncio.Future()
        self._pending_requests[request.id] = response_future
        
        try:
            success = await self.send_message(request)
            if not success:
                return None
            
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout for {request.id} to {recipient}")
            return None
        finally:
            self._pending_requests.pop(request.id, None)
    
    def register_handler(self, handler: MessageHandler) -> None:
        """Register message handler"""
        self._handlers[handler.handler_id] = handler
        self._metrics.active_handlers = len(self._handlers)
        
        logger.info(f"Handler registered: {handler.handler_id}",
                   pattern=handler.pattern,
                   message_types=[mt.value for mt in handler.message_types])
    
    def unregister_handler(self, handler_id: str) -> None:
        """Unregister message handler"""
        if handler_id in self._handlers:
            del self._handlers[handler_id]
            self._metrics.active_handlers = len(self._handlers)
            logger.info(f"Handler unregistered: {handler_id}")
    
    async def subscribe_to_topic(self, topic: str, handler_id: str) -> None:
        """Subscribe handler to topic"""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        
        if handler_id not in self._subscribers[topic]:
            self._subscribers[topic].append(handler_id)
            
            # Subscribe to Redis channel for real-time messages
            if self._redis:
                channel = f"topic:{topic}"
                pubsub = self._redis.pubsub()
                await pubsub.subscribe(channel)
                
                # Start subscription task
                task = asyncio.create_task(self._topic_subscription_loop(topic, pubsub))
                self._subscription_tasks[f"topic:{topic}"] = task
        
        logger.info(f"Handler {handler_id} subscribed to topic {topic}")
    
    async def _topic_subscription_loop(self, topic: str, pubsub) -> None:
        """Handle topic subscription messages"""
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        msg = Message.from_redis_dict(data)
                        await self.enqueue_message(msg)
                    except Exception as e:
                        logger.error(f"Failed to process topic {topic} message", error=str(e))
        finally:
            await pubsub.close()
    
    async def publish_event(self, topic: str, payload: Dict[str, Any], sender: str = "system") -> bool:
        """Publish event to topic subscribers"""
        event = Message(
            type=MessageType.EVENT,
            sender=sender,
            topic=topic,
            payload=payload,
            routing_strategy=RoutingStrategy.TOPIC
        )
        
        return await self.send_message(event)
    
    async def broadcast_message(self, payload: Dict[str, Any], 
                               message_type: MessageType = MessageType.BROADCAST,
                               sender: str = "system") -> bool:
        """Broadcast message to all agents"""
        broadcast = Message(
            type=message_type,
            sender=sender,
            payload=payload,
            routing_strategy=RoutingStrategy.BROADCAST
        )
        
        return await self.send_message(broadcast)
    
    def get_service_metrics(self) -> MessagingMetrics:
        """Get comprehensive messaging service metrics"""
        # Update metrics
        self._update_performance_metrics()
        self._update_handler_metrics()
        self._update_queue_metrics()
        
        return self._metrics
    
    def _update_performance_metrics(self) -> None:
        """Update performance metrics"""
        if self._processing_times:
            times = list(self._processing_times)
            times.sort()
            n = len(times)
            
            self._metrics.avg_processing_time_ms = sum(times) / n
            self._metrics.p95_processing_time_ms = times[int(n * 0.95)] if n > 0 else 0
            self._metrics.p99_processing_time_ms = times[int(n * 0.99)] if n > 0 else 0
        
        # Calculate throughput
        elapsed_time = time.time() - self._start_time
        if elapsed_time > 0:
            self._metrics.throughput_msg_per_sec = self._metrics.messages_processed / elapsed_time
    
    def _update_handler_metrics(self) -> None:
        """Update handler metrics"""
        self._metrics.handler_stats = {}
        for handler_id, handler in self._handlers.items():
            self._metrics.handler_stats[handler_id] = handler.get_stats()
    
    def _update_queue_metrics(self) -> None:
        """Update queue metrics"""
        self._metrics.queue_depth = self._message_queue.qsize()
        # Simplified backlog age calculation
        if self._metrics.queue_depth > 0:
            self._metrics.backlog_age_seconds = time.time() - self._start_time
        else:
            self._metrics.backlog_age_seconds = 0.0
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status"""
        try:
            if not self._connected or not self._redis:
                return {"status": "unhealthy", "error": "Not connected to Redis"}
            
            # Test Redis connection
            start_time = time.time()
            await self._redis.ping()
            ping_latency = (time.time() - start_time) * 1000
            
            # Get metrics
            metrics = self.get_service_metrics()
            
            # Determine health status
            is_healthy = (
                self._connected and
                ping_latency < 1000 and  # < 1 second
                self._running and
                len(self._handlers) > 0
            )
            
            return {
                "status": "healthy" if is_healthy else "degraded",
                "connected": self._connected,
                "running": self._running,
                "ping_latency_ms": ping_latency,
                "handlers_active": len(self._handlers),
                "queue_depth": metrics.queue_depth,
                "messages_processed": metrics.messages_processed,
                "messages_failed": metrics.messages_failed,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Convenience functions for common messaging patterns
def get_messaging_service() -> MessagingService:
    """Get messaging service instance (singleton)"""
    return MessagingService()


async def send_agent_message(agent_id: str, command: str, payload: Dict[str, Any]) -> bool:
    """Send message to specific agent"""
    messaging = get_messaging_service()
    message = Message(
        type=MessageType.COMMAND,
        sender="system",
        recipient=agent_id,
        payload={"command": command, **payload},
        routing_strategy=RoutingStrategy.DIRECT
    )
    return await messaging.send_message(message)


async def send_task_assignment(agent_id: str, task_id: str, task_data: Dict[str, Any]) -> bool:
    """Send task assignment to agent"""
    messaging = get_messaging_service()
    message = Message(
        type=MessageType.TASK_ASSIGNMENT,
        sender="orchestrator",
        recipient=agent_id,
        payload={
            "task_id": task_id,
            "task_data": task_data,
            "assigned_at": datetime.utcnow().isoformat()
        },
        priority=MessagePriority.HIGH,
        routing_strategy=RoutingStrategy.DIRECT
    )
    return await messaging.send_message(message)


async def send_heartbeat_request(agent_id: str) -> Optional[Message]:
    """Send heartbeat request and wait for response"""
    messaging = get_messaging_service()
    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": str(uuid.uuid4())
    }
    return await messaging.send_request(
        recipient=agent_id,
        payload=payload,
        message_type=MessageType.HEARTBEAT_REQUEST,
        timeout=30.0
    )


async def publish_system_event(event_type: str, data: Dict[str, Any]) -> bool:
    """Publish system-wide event"""
    messaging = get_messaging_service()
    return await messaging.publish_event(f"system.{event_type}", data)


async def broadcast_system_shutdown(reason: str = "System maintenance") -> bool:
    """Broadcast system shutdown message"""
    messaging = get_messaging_service()
    return await messaging.broadcast_message(
        payload={
            "reason": reason,
            "shutdown_time": datetime.utcnow().isoformat(),
            "grace_period_seconds": 300
        },
        message_type=MessageType.SYSTEM_SHUTDOWN
    )