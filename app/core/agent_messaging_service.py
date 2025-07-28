"""
Agent Messaging Service for LeanVibe Agent Hive 2.0

Enhanced messaging layer for agent lifecycle events, task assignments,
and real-time coordination. Builds upon Redis Streams for reliable
message delivery and event replay capabilities.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

import structlog
import redis.asyncio as redis
from redis.exceptions import ConnectionError, ResponseError

from .redis import get_redis, AgentMessageBroker, RedisStreamMessage
from .config import settings

logger = structlog.get_logger()


class MessagePriority(int, Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


class MessageType(str, Enum):
    """Types of agent messages."""
    # Lifecycle messages
    AGENT_REGISTERED = "agent_registered"
    AGENT_DEREGISTERED = "agent_deregistered"
    HEARTBEAT_REQUEST = "heartbeat_request"
    HEARTBEAT_RESPONSE = "heartbeat_response"
    
    # Task messages
    TASK_ASSIGNMENT = "task_assignment"
    TASK_ACCEPTANCE = "task_acceptance"
    TASK_REJECTION = "task_rejection"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETION = "task_completion"
    TASK_FAILURE = "task_failure"
    TASK_CANCELLATION = "task_cancellation"
    
    # System messages
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_UPDATE = "config_update"
    STATUS_UPDATE = "status_update"
    
    # Hook messages
    HOOK_PRE_TOOL_USE = "hook_pre_tool_use"
    HOOK_POST_TOOL_USE = "hook_post_tool_use"
    HOOK_ERROR = "hook_error"


@dataclass
class AgentMessage:
    """Enhanced agent message with lifecycle support."""
    message_id: str
    message_type: MessageType
    from_agent: str
    to_agent: str
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "payload": json.dumps(self.payload),
            "priority": self.priority.value,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "retry_count": str(self.retry_count),
            "max_retries": str(self.max_retries),
            "created_at": self.created_at.isoformat(),
            "metadata": json.dumps(self.metadata)
        }
    
    @classmethod
    def from_redis_message(cls, redis_msg: RedisStreamMessage) -> "AgentMessage":
        """Create AgentMessage from Redis stream message."""
        try:
            payload = json.loads(redis_msg.fields.get("payload", "{}"))
            metadata = json.loads(redis_msg.fields.get("metadata", "{}"))
            
            expires_at = None
            if redis_msg.fields.get("expires_at"):
                expires_at = datetime.fromisoformat(redis_msg.fields["expires_at"])
            
            return cls(
                message_id=redis_msg.fields["message_id"],
                message_type=MessageType(redis_msg.fields["message_type"]),
                from_agent=redis_msg.fields["from_agent"],
                to_agent=redis_msg.fields["to_agent"],
                payload=payload,
                priority=MessagePriority(int(redis_msg.fields.get("priority", 5))),
                correlation_id=redis_msg.fields.get("correlation_id"),
                reply_to=redis_msg.fields.get("reply_to"),
                expires_at=expires_at,
                retry_count=int(redis_msg.fields.get("retry_count", 0)),
                max_retries=int(redis_msg.fields.get("max_retries", 3)),
                created_at=datetime.fromisoformat(redis_msg.fields["created_at"]),
                metadata=metadata
            )
        except Exception as e:
            logger.error("Failed to parse Redis message", error=str(e))
            raise ValueError(f"Invalid Redis message format: {e}")
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        return self.expires_at is not None and datetime.utcnow() > self.expires_at
    
    def can_retry(self) -> bool:
        """Check if message can be retried."""
        return self.retry_count < self.max_retries and not self.is_expired()


class AgentMessagingService:
    """
    Enhanced messaging service for agent lifecycle coordination.
    
    Provides reliable message delivery, priority queuing, message replay,
    and real-time event streaming for agent coordination.
    """
    
    def __init__(self, redis_client=None):
        self.redis = redis_client or get_redis()
        self.message_broker = AgentMessageBroker(self.redis)
        
        # Message handlers registry
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        
        # Active subscriptions and consumers
        self.active_subscriptions: Set[str] = set()
        self.consumer_tasks: Dict[str, asyncio.Task] = {}
        
        # Performance metrics
        self.message_counts: Dict[str, int] = {}
        self.processing_times: Dict[str, List[float]] = {}
        self.failed_messages: List[Dict[str, Any]] = []
        
        logger.info("📡 Agent Messaging Service initialized")
    
    async def send_lifecycle_message(
        self,
        message_type: MessageType,
        from_agent: str,
        to_agent: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        correlation_id: Optional[str] = None,
        expires_in_seconds: Optional[int] = None
    ) -> str:
        """
        Send a lifecycle message between agents.
        
        Args:
            message_type: Type of message to send
            from_agent: Sender agent ID
            to_agent: Recipient agent ID or 'broadcast'
            payload: Message payload
            priority: Message priority
            correlation_id: Optional correlation ID for tracking
            expires_in_seconds: Message expiration time
        
        Returns:
            Message ID for tracking
        """
        message_id = str(uuid.uuid4())
        
        expires_at = None
        if expires_in_seconds:
            expires_at = datetime.utcnow() + timedelta(seconds=expires_in_seconds)
        
        message = AgentMessage(
            message_id=message_id,
            message_type=message_type,
            from_agent=from_agent,
            to_agent=to_agent,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id or str(uuid.uuid4()),
            expires_at=expires_at
        )
        
        try:
            # Determine stream name based on recipient
            if to_agent == "broadcast":
                stream_name = "agent_messages:broadcast"
            else:
                stream_name = f"agent_messages:{to_agent}"
            
            # Add to Redis stream with priority-based scoring
            stream_id = await self.redis.xadd(
                stream_name,
                message.to_dict(),
                maxlen=getattr(settings, 'REDIS_STREAM_MAX_LEN', 10000)
            )
            
            # Also add to priority queue for urgent messages
            if priority.value >= MessagePriority.HIGH.value:
                priority_key = f"priority_messages:{to_agent}"
                await self.redis.zadd(
                    priority_key,
                    {message_id: time.time() + priority.value}
                )
                await self.redis.expire(priority_key, 3600)  # 1 hour TTL
            
            # Publish to real-time pub/sub for immediate delivery
            pubsub_channel = f"realtime:{to_agent}" if to_agent != "broadcast" else "realtime:broadcast"
            await self.redis.publish(
                pubsub_channel,
                json.dumps({
                    "message_id": message_id,
                    "message_type": message_type.value,
                    "priority": priority.value,
                    "from_agent": from_agent,
                    "timestamp": datetime.utcnow().isoformat()
                })
            )
            
            # Update metrics
            self.message_counts[message_type.value] = self.message_counts.get(message_type.value, 0) + 1
            
            logger.info(
                "📤 Lifecycle message sent",
                message_id=message_id,
                message_type=message_type.value,
                from_agent=from_agent,
                to_agent=to_agent,
                priority=priority.name,
                stream_id=stream_id
            )
            
            return message_id
            
        except Exception as e:
            logger.error(
                "❌ Failed to send lifecycle message",
                message_type=message_type.value,
                from_agent=from_agent,
                to_agent=to_agent,
                error=str(e)
            )
            raise
    
    async def register_message_handler(
        self,
        message_type: MessageType,
        handler: Callable[[AgentMessage], Any]
    ) -> None:
        """Register a message handler for a specific message type."""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        
        self.message_handlers[message_type].append(handler)
        logger.info(f"📋 Registered handler for {message_type.value}")
    
    async def start_agent_consumer(
        self,
        agent_id: str,
        consumer_name: Optional[str] = None,
        batch_size: int = 10
    ) -> None:
        """
        Start consuming messages for a specific agent.
        
        Args:
            agent_id: ID of the agent to consume messages for
            consumer_name: Name of the consumer (defaults to agent_id)
            batch_size: Number of messages to process in each batch
        """
        consumer_name = consumer_name or f"consumer_{agent_id}"
        
        if agent_id in self.consumer_tasks:
            logger.warning(f"Consumer already running for agent {agent_id}")
            return
        
        # Create consumer task
        consumer_task = asyncio.create_task(
            self._consume_agent_messages(agent_id, consumer_name, batch_size)
        )
        self.consumer_tasks[agent_id] = consumer_task
        
        logger.info(f"🔄 Started message consumer for agent {agent_id}")
    
    async def stop_agent_consumer(self, agent_id: str) -> None:
        """Stop the message consumer for an agent."""
        if agent_id not in self.consumer_tasks:
            logger.warning(f"No consumer running for agent {agent_id}")
            return
        
        # Cancel consumer task
        self.consumer_tasks[agent_id].cancel()
        try:
            await self.consumer_tasks[agent_id]
        except asyncio.CancelledError:
            pass
        
        del self.consumer_tasks[agent_id]
        logger.info(f"⏹️ Stopped message consumer for agent {agent_id}")
    
    async def send_heartbeat_request(self, target_agent: str) -> str:
        """Send heartbeat request to an agent."""
        return await self.send_lifecycle_message(
            message_type=MessageType.HEARTBEAT_REQUEST,
            from_agent="orchestrator",
            to_agent=target_agent,
            payload={
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": str(uuid.uuid4())
            },
            priority=MessagePriority.HIGH,
            expires_in_seconds=30
        )
    
    async def send_task_assignment(
        self,
        agent_id: str,
        task_id: str,
        task_data: Dict[str, Any]
    ) -> str:
        """Send task assignment to an agent."""
        return await self.send_lifecycle_message(
            message_type=MessageType.TASK_ASSIGNMENT,
            from_agent="orchestrator",
            to_agent=agent_id,
            payload={
                "task_id": task_id,
                "task_data": task_data,
                "assigned_at": datetime.utcnow().isoformat()
            },
            priority=MessagePriority.HIGH
        )
    
    async def send_system_shutdown(self, reason: str = "System maintenance") -> List[str]:
        """Send shutdown message to all agents."""
        return [await self.send_lifecycle_message(
            message_type=MessageType.SYSTEM_SHUTDOWN,
            from_agent="orchestrator",
            to_agent="broadcast",
            payload={
                "reason": reason,
                "shutdown_time": datetime.utcnow().isoformat(),
                "grace_period_seconds": 300
            },
            priority=MessagePriority.CRITICAL
        )]
    
    async def get_agent_message_history(
        self,
        agent_id: str,
        message_type: Optional[MessageType] = None,
        limit: int = 100
    ) -> List[AgentMessage]:
        """
        Get message history for an agent.
        
        Args:
            agent_id: Agent ID to get history for
            message_type: Optional message type filter
            limit: Maximum number of messages to return
        
        Returns:
            List of AgentMessage objects
        """
        try:
            stream_name = f"agent_messages:{agent_id}"
            
            # Read messages from stream
            messages = await self.redis.xrevrange(
                stream_name,
                count=limit
            )
            
            parsed_messages = []
            for msg_id, fields in messages:
                try:
                    # Convert bytes to strings
                    str_fields = {}
                    for k, v in fields.items():
                        key = k.decode() if isinstance(k, bytes) else k
                        value = v.decode() if isinstance(v, bytes) else v
                        str_fields[key] = value
                    
                    redis_msg = RedisStreamMessage(msg_id.decode(), str_fields)
                    agent_msg = AgentMessage.from_redis_message(redis_msg)
                    
                    # Apply message type filter
                    if message_type is None or agent_msg.message_type == message_type:
                        parsed_messages.append(agent_msg)
                        
                except Exception as e:
                    logger.error(f"Failed to parse message {msg_id}", error=str(e))
                    continue
            
            return parsed_messages
            
        except Exception as e:
            logger.error(f"Failed to get message history for {agent_id}", error=str(e))
            return []
    
    async def get_messaging_metrics(self) -> Dict[str, Any]:
        """Get messaging service performance metrics."""
        try:
            # Get stream lengths
            stream_info = {}
            streams = await self.redis.keys("agent_messages:*")
            
            for stream in streams:
                stream_name = stream.decode() if isinstance(stream, bytes) else stream
                try:
                    info = await self.redis.xinfo_stream(stream_name)
                    stream_info[stream_name] = {
                        "length": info.get(b"length", 0),
                        "first_entry_id": info.get(b"first-entry", [b"0-0"])[0].decode(),
                        "last_entry_id": info.get(b"last-entry", [b"0-0"])[0].decode()
                    }
                except Exception:
                    # Stream might be empty or not exist
                    stream_info[stream_name] = {"length": 0}
            
            # Calculate average processing times
            avg_processing_times = {}
            for msg_type, times in self.processing_times.items():
                if times:
                    avg_processing_times[msg_type] = {
                        "count": len(times),
                        "average_ms": sum(times) / len(times),
                        "min_ms": min(times),
                        "max_ms": max(times)
                    }
            
            return {
                "active_consumers": len(self.consumer_tasks),
                "message_counts_by_type": dict(self.message_counts),
                "stream_info": stream_info,
                "average_processing_times": avg_processing_times,
                "failed_messages_count": len(self.failed_messages),
                "active_subscriptions": len(self.active_subscriptions),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get messaging metrics", error=str(e))
            return {"error": str(e)}
    
    async def _consume_agent_messages(
        self,
        agent_id: str,
        consumer_name: str,
        batch_size: int
    ) -> None:
        """
        Internal method to consume messages for an agent.
        """
        stream_name = f"agent_messages:{agent_id}"
        group_name = f"group_{agent_id}"
        
        # Ensure consumer group exists
        try:
            await self.redis.xgroup_create(
                stream_name,
                group_name,
                id='0',
                mkstream=True
            )
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                logger.error(f"Failed to create consumer group", error=str(e))
                return
        
        logger.info(f"📡 Starting message consumption for agent {agent_id}")
        
        while True:
            try:
                # Read messages from stream
                messages = await self.redis.xreadgroup(
                    group_name,
                    consumer_name,
                    {stream_name: '>'},
                    count=batch_size,
                    block=1000  # Block for 1 second
                )
                
                # Process messages
                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        start_time = time.time()
                        
                        try:
                            # Convert to AgentMessage
                            str_fields = {}
                            for k, v in fields.items():
                                key = k.decode() if isinstance(k, bytes) else k
                                value = v.decode() if isinstance(v, bytes) else v
                                str_fields[key] = value
                            
                            redis_msg = RedisStreamMessage(msg_id.decode(), str_fields)
                            agent_msg = AgentMessage.from_redis_message(redis_msg)
                            
                            # Check if message is expired
                            if agent_msg.is_expired():
                                logger.warning(f"Discarding expired message {agent_msg.message_id}")
                                await self.redis.xack(stream_name, group_name, msg_id)
                                continue
                            
                            # Process message with registered handlers
                            await self._process_message(agent_msg)
                            
                            # Acknowledge message
                            await self.redis.xack(stream_name, group_name, msg_id)
                            
                            # Record processing time
                            processing_time = (time.time() - start_time) * 1000
                            msg_type = agent_msg.message_type.value
                            if msg_type not in self.processing_times:
                                self.processing_times[msg_type] = []
                            self.processing_times[msg_type].append(processing_time)
                            
                        except Exception as e:
                            logger.error(
                                f"Failed to process message {msg_id}",
                                agent_id=agent_id,
                                error=str(e)
                            )
                            # Add to failed messages for debugging
                            self.failed_messages.append({
                                "message_id": msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id),
                                "agent_id": agent_id,
                                "error": str(e),
                                "timestamp": datetime.utcnow().isoformat()
                            })
                            
                            # Still acknowledge to prevent reprocessing
                            await self.redis.xack(stream_name, group_name, msg_id)
                
            except asyncio.CancelledError:
                logger.info(f"Message consumer cancelled for agent {agent_id}")
                break
            except Exception as e:
                logger.error(f"Error in message consumer for {agent_id}", error=str(e))
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _process_message(self, message: AgentMessage) -> None:
        """Process a message using registered handlers."""
        handlers = self.message_handlers.get(message.message_type, [])
        
        if not handlers:
            logger.debug(f"No handlers registered for message type {message.message_type.value}")
            return
        
        # Execute all handlers
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(
                    f"Handler failed for message {message.message_id}",
                    handler=handler.__name__,
                    error=str(e)
                )