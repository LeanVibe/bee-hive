"""
Unified Communication Manager for LeanVibe Agent Hive 2.0

Consolidates 19 communication-related files into a comprehensive communication system:
- Messaging service and coordination
- Redis pub/sub and streams management
- WebSocket communication
- Real-time coordination and synchronization
- Inter-agent messaging
- Communication analytics and monitoring
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

import structlog
import redis.asyncio as aioredis
from redis.asyncio import Redis

from .unified_manager_base import UnifiedManagerBase, ManagerConfig, PluginInterface, PluginType
from .redis import get_redis

logger = structlog.get_logger()


class MessageType(str, Enum):
    """Types of messages in the system."""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_COMPLETION = "task_completion"
    AGENT_HEARTBEAT = "agent_heartbeat"
    COORDINATION_REQUEST = "coordination_request"
    COORDINATION_RESPONSE = "coordination_response"
    KNOWLEDGE_SHARE = "knowledge_share"
    SYSTEM_NOTIFICATION = "system_notification"
    REALTIME_UPDATE = "realtime_update"
    ERROR_NOTIFICATION = "error_notification"
    HEALTH_CHECK = "health_check"


class MessagePriority(int, Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    URGENT = 10


class DeliveryGuarantee(str, Enum):
    """Message delivery guarantees."""
    BEST_EFFORT = "best_effort"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"


class CoordinationState(str, Enum):
    """Real-time coordination states."""
    ACTIVE = "active"
    PAUSED = "paused"
    SYNCHRONIZED = "synchronized"
    CONFLICT = "conflict"
    TERMINATED = "terminated"


@dataclass
class Message:
    """Standard message format for inter-agent communication."""
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    from_agent: str = ""
    to_agent: str = ""
    message_type: MessageType = MessageType.SYSTEM_NOTIFICATION
    priority: MessagePriority = MessagePriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.BEST_EFFORT
    retry_count: int = 0
    max_retries: int = 3
    expires_at: Optional[datetime] = None
    correlation_id: Optional[uuid.UUID] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": str(self.id),
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "payload": self.payload,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "delivery_guarantee": self.delivery_guarantee.value,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            id=uuid.UUID(data["id"]),
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            message_type=MessageType(data["message_type"]),
            priority=MessagePriority(data["priority"]),
            payload=data["payload"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            delivery_guarantee=DeliveryGuarantee(data.get("delivery_guarantee", "best_effort")),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            correlation_id=uuid.UUID(data["correlation_id"]) if data.get("correlation_id") else None
        )


@dataclass
class CoordinationSession:
    """Real-time coordination session between agents."""
    session_id: uuid.UUID = field(default_factory=uuid.uuid4)
    participants: Set[str] = field(default_factory=set)
    state: CoordinationState = CoordinationState.ACTIVE
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)
    sync_points: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CommunicationMetrics:
    """Communication performance metrics."""
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_messages_failed: int = 0
    average_delivery_time_ms: float = 0.0
    active_connections: int = 0
    active_coordination_sessions: int = 0
    redis_operations_per_second: float = 0.0
    websocket_connections: int = 0
    message_queue_depth: int = 0


class MessageBroker:
    """Advanced message broker with Redis Streams and Pub/Sub."""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.subscription_tasks: Dict[str, asyncio.Task] = {}
        self.delivery_confirmations: Dict[uuid.UUID, bool] = {}
        self.retry_queue: deque = deque()
        
    async def send_message(
        self,
        message: Message,
        use_streams: bool = True
    ) -> bool:
        """Send message via Redis Streams or Pub/Sub."""
        try:
            message_data = message.to_dict()
            
            if use_streams:
                # Use Redis Streams for reliable delivery
                stream_key = f"agent_messages:{message.to_agent}"
                message_id = await self.redis.xadd(
                    stream_key,
                    message_data,
                    maxlen=10000
                )
                
                # Store delivery confirmation expectation
                if message.delivery_guarantee != DeliveryGuarantee.BEST_EFFORT:
                    self.delivery_confirmations[message.id] = False
                
                logger.debug(
                    "Message sent via stream",
                    message_id=str(message.id),
                    stream_key=stream_key,
                    redis_message_id=message_id
                )
                
            else:
                # Use Redis Pub/Sub for real-time delivery
                channel = f"agent_channel:{message.to_agent}"
                await self.redis.publish(channel, json.dumps(message_data))
                
                logger.debug(
                    "Message sent via pub/sub",
                    message_id=str(message.id),
                    channel=channel
                )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to send message",
                message_id=str(message.id),
                error=str(e)
            )
            
            # Add to retry queue if applicable
            if message.retry_count < message.max_retries:
                message.retry_count += 1
                self.retry_queue.append(message)
            
            return False
    
    async def receive_messages(
        self,
        agent_id: str,
        consumer_group: str = "default",
        consumer_name: Optional[str] = None
    ) -> List[Message]:
        """Receive messages for an agent from Redis Streams."""
        try:
            if consumer_name is None:
                consumer_name = f"consumer_{uuid.uuid4().hex[:8]}"
            
            stream_key = f"agent_messages:{agent_id}"
            
            # Create consumer group if it doesn't exist
            try:
                await self.redis.xgroup_create(stream_key, consumer_group, id="0", mkstream=True)
            except Exception:
                pass  # Group already exists
            
            # Read messages
            messages = await self.redis.xreadgroup(
                consumer_group,
                consumer_name,
                streams={stream_key: ">"},
                count=10,
                block=1000
            )
            
            parsed_messages = []
            for stream, message_list in messages:
                for message_id, fields in message_list:
                    try:
                        message_data = {k.decode() if isinstance(k, bytes) else k: 
                                      v.decode() if isinstance(v, bytes) else v 
                                      for k, v in fields.items()}
                        
                        # Parse JSON fields
                        for field in ["payload", "metadata"]:
                            if field in message_data:
                                message_data[field] = json.loads(message_data[field])
                        
                        message = Message.from_dict(message_data)
                        parsed_messages.append(message)
                        
                        # Acknowledge message
                        await self.redis.xack(stream_key, consumer_group, message_id)
                        
                    except Exception as e:
                        logger.error(
                            "Failed to parse message",
                            message_id=message_id,
                            error=str(e)
                        )
            
            return parsed_messages
            
        except Exception as e:
            logger.error(
                "Failed to receive messages",
                agent_id=agent_id,
                error=str(e)
            )
            return []
    
    async def subscribe_to_channel(
        self,
        channel: str,
        handler: Callable[[Dict[str, Any]], None]
    ) -> str:
        """Subscribe to a Redis Pub/Sub channel."""
        try:
            subscription_id = str(uuid.uuid4())
            
            async def subscription_loop():
                pubsub = self.redis.pubsub()
                await pubsub.subscribe(channel)
                
                try:
                    async for message in pubsub.listen():
                        if message["type"] == "message":
                            try:
                                data = json.loads(message["data"])
                                await handler(data)
                            except Exception as e:
                                logger.error(
                                    "Error in message handler",
                                    channel=channel,
                                    error=str(e)
                                )
                finally:
                    await pubsub.unsubscribe(channel)
                    await pubsub.close()
            
            task = asyncio.create_task(subscription_loop())
            self.subscription_tasks[subscription_id] = task
            
            logger.info(
                "Subscribed to channel",
                channel=channel,
                subscription_id=subscription_id
            )
            
            return subscription_id
            
        except Exception as e:
            logger.error(
                "Failed to subscribe to channel",
                channel=channel,
                error=str(e)
            )
            return ""
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from a channel."""
        if subscription_id in self.subscription_tasks:
            task = self.subscription_tasks[subscription_id]
            task.cancel()
            del self.subscription_tasks[subscription_id]
            
            logger.info("Unsubscribed", subscription_id=subscription_id)
            return True
        
        return False
    
    async def process_retry_queue(self) -> None:
        """Process messages in retry queue."""
        while self.retry_queue:
            message = self.retry_queue.popleft()
            
            # Check if message has expired
            if message.expires_at and datetime.utcnow() > message.expires_at:
                logger.warning(
                    "Message expired, dropping",
                    message_id=str(message.id)
                )
                continue
            
            # Retry sending
            success = await self.send_message(message)
            if not success:
                logger.warning(
                    "Message retry failed",
                    message_id=str(message.id),
                    retry_count=message.retry_count
                )


class CoordinationEngine:
    """Real-time coordination engine for multi-agent synchronization."""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.active_sessions: Dict[uuid.UUID, CoordinationSession] = {}
        self.sync_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
    async def create_coordination_session(
        self,
        participants: List[str],
        context: Dict[str, Any] = None
    ) -> CoordinationSession:
        """Create a new coordination session."""
        session = CoordinationSession(
            participants=set(participants),
            context=context or {}
        )
        
        self.active_sessions[session.session_id] = session
        
        # Publish session creation to participants
        for participant in participants:
            await self.redis.publish(
                f"coordination:{participant}",
                json.dumps({
                    "event": "session_created",
                    "session_id": str(session.session_id),
                    "participants": list(participants),
                    "context": context
                })
            )
        
        logger.info(
            "Coordination session created",
            session_id=str(session.session_id),
            participants=participants
        )
        
        return session
    
    async def sync_point(
        self,
        session_id: uuid.UUID,
        agent_id: str,
        sync_data: Dict[str, Any]
    ) -> bool:
        """Synchronize agents at a specific point."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # Add sync point
        sync_point = {
            "agent_id": agent_id,
            "timestamp": datetime.utcnow(),
            "data": sync_data
        }
        session.sync_points.append(sync_point)
        session.last_activity = datetime.utcnow()
        
        # Check if all participants have reached sync point
        recent_sync_agents = set()
        cutoff_time = datetime.utcnow() - timedelta(seconds=30)  # 30 second window
        
        for sp in session.sync_points:
            if sp["timestamp"] > cutoff_time:
                recent_sync_agents.add(sp["agent_id"])
        
        if recent_sync_agents == session.participants:
            # All agents synchronized
            session.state = CoordinationState.SYNCHRONIZED
            
            # Notify all participants
            for participant in session.participants:
                await self.redis.publish(
                    f"coordination:{participant}",
                    json.dumps({
                        "event": "synchronized",
                        "session_id": str(session_id),
                        "sync_data": [sp["data"] for sp in session.sync_points if sp["timestamp"] > cutoff_time]
                    })
                )
            
            logger.info(
                "Coordination session synchronized",
                session_id=str(session_id),
                participants=list(session.participants)
            )
            
            return True
        
        return False
    
    async def detect_conflicts(self, session_id: uuid.UUID) -> List[Dict[str, Any]]:
        """Detect conflicts in coordination session."""
        if session_id not in self.active_sessions:
            return []
        
        session = self.active_sessions[session_id]
        conflicts = []
        
        # Analyze recent sync points for conflicts
        recent_sync_points = [
            sp for sp in session.sync_points
            if sp["timestamp"] > datetime.utcnow() - timedelta(minutes=5)
        ]
        
        # Simple conflict detection based on conflicting actions
        agent_actions = defaultdict(list)
        for sp in recent_sync_points:
            agent_id = sp["agent_id"]
            action = sp["data"].get("action")
            if action:
                agent_actions[action].append(agent_id)
        
        # Find conflicting actions
        for action, agents in agent_actions.items():
            if len(agents) > 1 and action in ["write_file", "modify_database", "exclusive_resource"]:
                conflicts.append({
                    "type": "resource_conflict",
                    "action": action,
                    "conflicting_agents": agents,
                    "detected_at": datetime.utcnow()
                })
        
        if conflicts:
            session.state = CoordinationState.CONFLICT
            
        return conflicts
    
    async def resolve_conflict(
        self,
        session_id: uuid.UUID,
        resolution_strategy: str = "priority_based"
    ) -> bool:
        """Resolve conflicts in coordination session."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        conflicts = await self.detect_conflicts(session_id)
        
        if not conflicts:
            return True
        
        for conflict in conflicts:
            if resolution_strategy == "priority_based":
                # Resolve based on agent priority (first agent wins)
                winning_agent = conflict["conflicting_agents"][0]
                losing_agents = conflict["conflicting_agents"][1:]
                
                # Notify agents of resolution
                for agent in losing_agents:
                    await self.redis.publish(
                        f"coordination:{agent}",
                        json.dumps({
                            "event": "conflict_resolved",
                            "session_id": str(session_id),
                            "action": conflict["action"],
                            "result": "deferred",
                            "winning_agent": winning_agent
                        })
                    )
                
                await self.redis.publish(
                    f"coordination:{winning_agent}",
                    json.dumps({
                        "event": "conflict_resolved",
                        "session_id": str(session_id),
                        "action": conflict["action"],
                        "result": "proceed"
                    })
                )
        
        session.state = CoordinationState.ACTIVE
        
        logger.info(
            "Conflicts resolved",
            session_id=str(session_id),
            conflicts_count=len(conflicts)
        )
        
        return True
    
    async def terminate_session(self, session_id: uuid.UUID) -> bool:
        """Terminate coordination session."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        session.state = CoordinationState.TERMINATED
        
        # Notify all participants
        for participant in session.participants:
            await self.redis.publish(
                f"coordination:{participant}",
                json.dumps({
                    "event": "session_terminated",
                    "session_id": str(session_id)
                })
            )
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        logger.info(
            "Coordination session terminated",
            session_id=str(session_id)
        )
        
        return True


class CommunicationAnalytics:
    """Communication analytics and monitoring."""
    
    def __init__(self):
        self.metrics = CommunicationMetrics()
        self.message_history: deque = deque(maxlen=10000)
        self.delivery_times: deque = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
    def record_message_sent(self, message: Message) -> None:
        """Record message sent event."""
        self.metrics.total_messages_sent += 1
        self.message_history.append({
            "event": "sent",
            "message_id": str(message.id),
            "timestamp": datetime.utcnow(),
            "message_type": message.message_type.value,
            "priority": message.priority.value
        })
    
    def record_message_received(self, message: Message) -> None:
        """Record message received event."""
        self.metrics.total_messages_received += 1
        self.message_history.append({
            "event": "received",
            "message_id": str(message.id),
            "timestamp": datetime.utcnow(),
            "message_type": message.message_type.value
        })
    
    def record_message_failed(self, message: Message, error: str) -> None:
        """Record message failure."""
        self.metrics.total_messages_failed += 1
        self.error_counts[error] += 1
        self.message_history.append({
            "event": "failed",
            "message_id": str(message.id),
            "timestamp": datetime.utcnow(),
            "error": error
        })
    
    def record_delivery_time(self, delivery_time_ms: float) -> None:
        """Record message delivery time."""
        self.delivery_times.append(delivery_time_ms)
        
        # Update average
        if self.delivery_times:
            self.metrics.average_delivery_time_ms = sum(self.delivery_times) / len(self.delivery_times)
    
    def get_communication_stats(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get communication statistics."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        recent_events = [
            event for event in self.message_history
            if event["timestamp"] > cutoff_time
        ]
        
        sent_count = len([e for e in recent_events if e["event"] == "sent"])
        received_count = len([e for e in recent_events if e["event"] == "received"])
        failed_count = len([e for e in recent_events if e["event"] == "failed"])
        
        message_types = defaultdict(int)
        for event in recent_events:
            if "message_type" in event:
                message_types[event["message_type"]] += 1
        
        return {
            "time_window_minutes": time_window_minutes,
            "messages_sent": sent_count,
            "messages_received": received_count,
            "messages_failed": failed_count,
            "success_rate": (sent_count - failed_count) / max(sent_count, 1),
            "average_delivery_time_ms": self.metrics.average_delivery_time_ms,
            "message_types": dict(message_types),
            "top_errors": dict(sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        }


class CommunicationManager(UnifiedManagerBase):
    """
    Unified Communication Manager consolidating all communication-related functionality.
    
    Replaces 19 separate files:
    - communication.py
    - communication_analyzer.py
    - messaging_service.py
    - messaging_migration.py
    - redis_pubsub_manager.py
    - enhanced_redis_streams_manager.py
    - redis_integration.py
    - optimized_redis.py
    - coordination_dashboard.py
    - enhanced_coordination_bridge.py
    - enhanced_coordination_commands.py
    - enhanced_coordination_database_integration.py
    - global_coordination_integration.py
    - realtime_coordination_sync.py
    - observability_streams.py
    - team_coordination_error_handler.py
    - team_coordination_metrics.py
    - team_coordination_redis.py
    """
    
    def __init__(self, config: ManagerConfig, dependencies: Optional[Dict[str, Any]] = None):
        super().__init__(config, dependencies)
        
        # Core components
        self.redis_client: Optional[Redis] = None
        self.message_broker: Optional[MessageBroker] = None
        self.coordination_engine: Optional[CoordinationEngine] = None
        self.analytics = CommunicationAnalytics()
        
        # State tracking
        self.active_agents: Set[str] = set()
        self.websocket_connections: Dict[str, Any] = {}
        self.processing_tasks: List[asyncio.Task] = []
        
        # Configuration
        self.max_message_size = config.plugin_config.get("max_message_size", 1024 * 1024)  # 1MB
        self.message_retention_hours = config.plugin_config.get("message_retention_hours", 24)
        self.enable_coordination = config.plugin_config.get("enable_coordination", True)
    
    async def _initialize_manager(self) -> bool:
        """Initialize the communication manager."""
        try:
            # Initialize Redis connection
            self.redis_client = get_redis()
            if not self.redis_client:
                logger.error("Failed to get Redis client")
                return False
            
            # Initialize components
            self.message_broker = MessageBroker(self.redis_client)
            
            if self.enable_coordination:
                self.coordination_engine = CoordinationEngine(self.redis_client)
            
            # Start background processing tasks
            self.processing_tasks.extend([
                asyncio.create_task(self._message_retry_processor()),
                asyncio.create_task(self._cleanup_processor()),
                asyncio.create_task(self._metrics_processor())
            ])
            
            logger.info(
                "Communication Manager initialized",
                coordination_enabled=self.enable_coordination,
                max_message_size=self.max_message_size
            )
            return True
            
        except Exception as e:
            logger.error("Failed to initialize Communication Manager", error=str(e))
            return False
    
    async def _shutdown_manager(self) -> None:
        """Shutdown the communication manager."""
        try:
            # Cancel processing tasks
            for task in self.processing_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*self.processing_tasks, return_exceptions=True)
            
            # Close message broker subscriptions
            if self.message_broker:
                for subscription_id in list(self.message_broker.subscription_tasks.keys()):
                    await self.message_broker.unsubscribe(subscription_id)
            
            # Terminate active coordination sessions
            if self.coordination_engine:
                for session_id in list(self.coordination_engine.active_sessions.keys()):
                    await self.coordination_engine.terminate_session(session_id)
            
            logger.info("Communication Manager shutdown completed")
            
        except Exception as e:
            logger.error("Error during Communication Manager shutdown", error=str(e))
    
    async def _get_manager_health(self) -> Dict[str, Any]:
        """Get communication manager health information."""
        health_info = {
            "redis_connected": self.redis_client is not None,
            "active_agents": len(self.active_agents),
            "websocket_connections": len(self.websocket_connections),
            "processing_tasks": len([t for t in self.processing_tasks if not t.done()]),
            "metrics": {
                "total_messages_sent": self.analytics.metrics.total_messages_sent,
                "total_messages_received": self.analytics.metrics.total_messages_received,
                "total_messages_failed": self.analytics.metrics.total_messages_failed,
                "average_delivery_time_ms": self.analytics.metrics.average_delivery_time_ms
            }
        }
        
        if self.coordination_engine:
            health_info["coordination"] = {
                "active_sessions": len(self.coordination_engine.active_sessions),
                "enabled": self.enable_coordination
            }
        
        if self.message_broker:
            health_info["message_broker"] = {
                "retry_queue_size": len(self.message_broker.retry_queue),
                "active_subscriptions": len(self.message_broker.subscription_tasks)
            }
        
        return health_info
    
    async def _load_plugins(self) -> None:
        """Load communication manager plugins."""
        # Communication plugins would be loaded here
        pass
    
    # === BACKGROUND PROCESSING ===
    
    async def _message_retry_processor(self) -> None:
        """Background processor for message retries."""
        while True:
            try:
                await asyncio.sleep(10)  # Process every 10 seconds
                
                if self.message_broker:
                    await self.message_broker.process_retry_queue()
                
            except Exception as e:
                logger.error("Error in message retry processor", error=str(e))
                await asyncio.sleep(30)
    
    async def _cleanup_processor(self) -> None:
        """Background processor for cleanup operations."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old messages and coordination sessions
                await self._cleanup_old_data()
                
            except Exception as e:
                logger.error("Error in cleanup processor", error=str(e))
                await asyncio.sleep(1800)  # Wait longer on error
    
    async def _metrics_processor(self) -> None:
        """Background processor for metrics collection."""
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                # Update connection metrics
                if self.redis_client:
                    self.analytics.metrics.active_connections = 1  # Redis connection
                
                self.analytics.metrics.websocket_connections = len(self.websocket_connections)
                
                if self.coordination_engine:
                    self.analytics.metrics.active_coordination_sessions = len(
                        self.coordination_engine.active_sessions
                    )
                
            except Exception as e:
                logger.error("Error in metrics processor", error=str(e))
                await asyncio.sleep(120)
    
    # === CORE MESSAGING OPERATIONS ===
    
    async def send_message(
        self,
        from_agent: str,
        to_agent: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.BEST_EFFORT,
        correlation_id: Optional[uuid.UUID] = None
    ) -> bool:
        """Send a message between agents."""
        return await self.execute_with_monitoring(
            "send_message",
            self._send_message_impl,
            from_agent,
            to_agent,
            message_type,
            payload,
            priority,
            delivery_guarantee,
            correlation_id
        )
    
    async def _send_message_impl(
        self,
        from_agent: str,
        to_agent: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: MessagePriority,
        delivery_guarantee: DeliveryGuarantee,
        correlation_id: Optional[uuid.UUID]
    ) -> bool:
        """Internal implementation of message sending."""
        try:
            # Validate message size
            message_size = len(json.dumps(payload))
            if message_size > self.max_message_size:
                logger.warning(
                    "Message size exceeds limit",
                    from_agent=from_agent,
                    to_agent=to_agent,
                    size=message_size,
                    limit=self.max_message_size
                )
                return False
            
            # Create message
            message = Message(
                from_agent=from_agent,
                to_agent=to_agent,
                message_type=message_type,
                priority=priority,
                payload=payload,
                delivery_guarantee=delivery_guarantee,
                correlation_id=correlation_id
            )
            
            # Set expiration for high priority messages
            if priority == MessagePriority.URGENT:
                message.expires_at = datetime.utcnow() + timedelta(minutes=5)
            
            # Send via message broker
            success = await self.message_broker.send_message(message)
            
            # Record analytics
            if success:
                self.analytics.record_message_sent(message)
                self.active_agents.add(from_agent)
                self.active_agents.add(to_agent)
            else:
                self.analytics.record_message_failed(message, "broker_send_failed")
            
            return success
            
        except Exception as e:
            logger.error(
                "Failed to send message",
                from_agent=from_agent,
                to_agent=to_agent,
                error=str(e)
            )
            return False
    
    async def receive_messages(
        self,
        agent_id: str,
        consumer_group: str = "default",
        max_messages: int = 10
    ) -> List[Message]:
        """Receive messages for an agent."""
        return await self.execute_with_monitoring(
            "receive_messages",
            self._receive_messages_impl,
            agent_id,
            consumer_group,
            max_messages
        )
    
    async def _receive_messages_impl(
        self,
        agent_id: str,
        consumer_group: str,
        max_messages: int
    ) -> List[Message]:
        """Internal implementation of message receiving."""
        try:
            messages = await self.message_broker.receive_messages(
                agent_id,
                consumer_group
            )
            
            # Record analytics
            for message in messages:
                self.analytics.record_message_received(message)
            
            # Update active agents
            self.active_agents.add(agent_id)
            
            return messages[:max_messages]
            
        except Exception as e:
            logger.error(
                "Failed to receive messages",
                agent_id=agent_id,
                error=str(e)
            )
            return []
    
    async def broadcast_message(
        self,
        from_agent: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        target_agents: Optional[List[str]] = None
    ) -> int:
        """Broadcast message to multiple agents."""
        return await self.execute_with_monitoring(
            "broadcast_message",
            self._broadcast_message_impl,
            from_agent,
            message_type,
            payload,
            target_agents
        )
    
    async def _broadcast_message_impl(
        self,
        from_agent: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        target_agents: Optional[List[str]]
    ) -> int:
        """Internal implementation of message broadcasting."""
        try:
            targets = target_agents if target_agents else list(self.active_agents)
            successful_sends = 0
            
            # Send to all targets
            tasks = []
            for target in targets:
                if target != from_agent:  # Don't send to self
                    task = self._send_message_impl(
                        from_agent,
                        target,
                        message_type,
                        payload,
                        MessagePriority.NORMAL,
                        DeliveryGuarantee.BEST_EFFORT,
                        None
                    )
                    tasks.append(task)
            
            # Wait for all sends to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful_sends = sum(1 for result in results if result is True)
            
            logger.info(
                "Broadcast completed",
                from_agent=from_agent,
                message_type=message_type.value,
                targets=len(targets),
                successful=successful_sends
            )
            
            return successful_sends
            
        except Exception as e:
            logger.error(
                "Failed to broadcast message",
                from_agent=from_agent,
                error=str(e)
            )
            return 0
    
    # === COORDINATION OPERATIONS ===
    
    async def create_coordination_session(
        self,
        participants: List[str],
        context: Dict[str, Any] = None
    ) -> Optional[uuid.UUID]:
        """Create a coordination session between agents."""
        if not self.coordination_engine:
            return None
        
        return await self.execute_with_monitoring(
            "create_coordination_session",
            self._create_coordination_session_impl,
            participants,
            context
        )
    
    async def _create_coordination_session_impl(
        self,
        participants: List[str],
        context: Dict[str, Any]
    ) -> Optional[uuid.UUID]:
        """Internal implementation of coordination session creation."""
        try:
            session = await self.coordination_engine.create_coordination_session(
                participants,
                context
            )
            
            logger.info(
                "Coordination session created",
                session_id=str(session.session_id),
                participants=participants
            )
            
            return session.session_id
            
        except Exception as e:
            logger.error(
                "Failed to create coordination session",
                participants=participants,
                error=str(e)
            )
            return None
    
    async def sync_agents(
        self,
        session_id: uuid.UUID,
        agent_id: str,
        sync_data: Dict[str, Any]
    ) -> bool:
        """Synchronize agent at coordination point."""
        if not self.coordination_engine:
            return False
        
        return await self.execute_with_monitoring(
            "sync_agents",
            self.coordination_engine.sync_point,
            session_id,
            agent_id,
            sync_data
        )
    
    # === WEBSOCKET SUPPORT ===
    
    async def register_websocket_connection(
        self,
        connection_id: str,
        agent_id: str,
        websocket_handler: Any
    ) -> bool:
        """Register a WebSocket connection for real-time communication."""
        try:
            self.websocket_connections[connection_id] = {
                "agent_id": agent_id,
                "handler": websocket_handler,
                "connected_at": datetime.utcnow(),
                "last_activity": datetime.utcnow()
            }
            
            # Subscribe to agent's real-time channel
            channel = f"realtime:{agent_id}"
            
            async def websocket_message_handler(data: Dict[str, Any]):
                try:
                    if connection_id in self.websocket_connections:
                        await websocket_handler.send_text(json.dumps(data))
                        self.websocket_connections[connection_id]["last_activity"] = datetime.utcnow()
                except Exception as e:
                    logger.error(
                        "WebSocket send failed",
                        connection_id=connection_id,
                        error=str(e)
                    )
            
            subscription_id = await self.message_broker.subscribe_to_channel(
                channel,
                websocket_message_handler
            )
            
            self.websocket_connections[connection_id]["subscription_id"] = subscription_id
            
            logger.info(
                "WebSocket connection registered",
                connection_id=connection_id,
                agent_id=agent_id
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to register WebSocket connection",
                connection_id=connection_id,
                error=str(e)
            )
            return False
    
    async def unregister_websocket_connection(self, connection_id: str) -> bool:
        """Unregister a WebSocket connection."""
        try:
            if connection_id in self.websocket_connections:
                connection = self.websocket_connections[connection_id]
                
                # Unsubscribe from channel
                if "subscription_id" in connection:
                    await self.message_broker.unsubscribe(connection["subscription_id"])
                
                del self.websocket_connections[connection_id]
                
                logger.info(
                    "WebSocket connection unregistered",
                    connection_id=connection_id
                )
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(
                "Failed to unregister WebSocket connection",
                connection_id=connection_id,
                error=str(e)
            )
            return False
    
    async def send_realtime_update(
        self,
        agent_id: str,
        update_data: Dict[str, Any]
    ) -> bool:
        """Send real-time update to agent's WebSocket connections."""
        try:
            channel = f"realtime:{agent_id}"
            await self.redis_client.publish(channel, json.dumps({
                "type": "realtime_update",
                "data": update_data,
                "timestamp": datetime.utcnow().isoformat()
            }))
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to send real-time update",
                agent_id=agent_id,
                error=str(e)
            )
            return False
    
    # === UTILITY METHODS ===
    
    async def _cleanup_old_data(self) -> None:
        """Clean up old messages and sessions."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=self.message_retention_hours)
            
            # Clean up old message streams
            for agent_id in self.active_agents:
                stream_key = f"agent_messages:{agent_id}"
                try:
                    # Trim stream to remove old messages
                    await self.redis_client.xtrim(stream_key, maxlen=1000)
                except Exception:
                    pass
            
            # Clean up old coordination sessions
            if self.coordination_engine:
                for session_id, session in list(self.coordination_engine.active_sessions.items()):
                    if session.last_activity < cutoff_time:
                        await self.coordination_engine.terminate_session(session_id)
            
            logger.debug("Cleanup completed", cutoff_time=cutoff_time.isoformat())
            
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))
    
    # === PUBLIC API METHODS ===
    
    async def get_communication_stats(self, time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get communication statistics."""
        try:
            stats = self.analytics.get_communication_stats(time_window_minutes)
            
            # Add current state information
            stats.update({
                "current_state": {
                    "active_agents": len(self.active_agents),
                    "websocket_connections": len(self.websocket_connections),
                    "coordination_sessions": len(self.coordination_engine.active_sessions) if self.coordination_engine else 0,
                    "message_broker_retry_queue": len(self.message_broker.retry_queue) if self.message_broker else 0
                }
            })
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get communication stats", error=str(e))
            return {"error": str(e)}
    
    async def get_agent_message_history(
        self,
        agent_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get message history for an agent."""
        try:
            # Filter message history for specific agent
            agent_messages = [
                event for event in self.analytics.message_history
                if (event.get("from_agent") == agent_id or 
                    event.get("to_agent") == agent_id)
            ]
            
            # Sort by timestamp and limit
            agent_messages.sort(key=lambda x: x["timestamp"], reverse=True)
            return agent_messages[:limit]
            
        except Exception as e:
            logger.error(
                "Failed to get agent message history",
                agent_id=agent_id,
                error=str(e)
            )
            return []


# Factory function for creating communication manager
def create_communication_manager(**config_overrides) -> CommunicationManager:
    """Create and initialize a communication manager."""
    config = create_manager_config("CommunicationManager", **config_overrides)
    return CommunicationManager(config)