"""
Messaging SDK for LeanVibe Agent Hive 2.0.

Provides high-level Python SDK for reliable inter-agent communication
with automatic retry, error handling, and performance optimization.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from ..core.communication import MessageBroker, SimplePubSub, CommunicationError
from ..core.performance_optimizations import (
    HighPerformanceMessageBroker, 
    BatchConfig, 
    CompressionConfig, 
    ConnectionConfig
)
from ..core.stream_monitor import StreamMonitor
from ..core.backpressure_manager import BackPressureManager, BackPressureConfig
from ..models.message import (
    StreamMessage,
    MessageType,
    MessagePriority,
    MessageDeliveryReport
)

logger = logging.getLogger(__name__)


class MessagingClient:
    """
    High-level messaging client for agent communication.
    
    Provides simple interface for sending and receiving messages
    with automatic connection management, retry logic, and monitoring.
    """
    
    def __init__(
        self,
        agent_id: str,
        redis_url: str = None,
        secret_key: str = None,
        auto_retry: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_high_performance: bool = False,
        enable_monitoring: bool = False,
        enable_backpressure: bool = False
    ):
        self.agent_id = agent_id
        self.auto_retry = auto_retry
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_high_performance = enable_high_performance
        self.enable_monitoring = enable_monitoring
        self.enable_backpressure = enable_backpressure
        
        # Communication components
        if enable_high_performance:
            # Use high-performance broker for production workloads
            batch_config = BatchConfig(
                max_batch_size=50,
                max_batch_wait_ms=25,
                adaptive_batching=True
            )
            compression_config = CompressionConfig(
                min_payload_size=500,
                adaptive_compression=True
            )
            connection_config = ConnectionConfig(
                pool_size=20,
                adaptive_scaling=True
            )
            
            self._hp_broker = HighPerformanceMessageBroker(
                redis_url=redis_url,
                batch_config=batch_config,
                compression_config=compression_config,
                connection_config=connection_config
            )
            self._broker = None  # Use high-performance broker instead
        else:
            self._broker = MessageBroker(
                redis_url=redis_url,
                secret_key=secret_key
            )
            self._hp_broker = None
        
        self._pubsub = SimplePubSub(redis_url=redis_url)
        
        # Enhanced components
        self._monitor: Optional[StreamMonitor] = None
        self._backpressure_manager: Optional[BackPressureManager] = None
        
        # Connection state
        self._connected = False
        self._message_handlers: Dict[str, Callable] = {}
        self._consumer_tasks: List[asyncio.Task] = []
        
        # Performance tracking
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "retries": 0
        }
    
    async def connect(self) -> None:
        """Connect to messaging infrastructure."""
        if self._connected:
            return
        
        try:
            # Connect communication components
            if self._hp_broker:
                await self._hp_broker.start()
            else:
                await self._broker.connect()
            
            await self._pubsub.connect()
            
            # Initialize enhanced components if enabled
            if self.enable_monitoring:
                from redis.asyncio import Redis
                redis_client = Redis.from_url(self._pubsub.redis_url)
                self._monitor = StreamMonitor(redis_client, enable_prometheus=True)
                await self._monitor.start()
                logger.info("Stream monitoring enabled")
            
            if self.enable_backpressure:
                from redis.asyncio import Redis
                redis_client = Redis.from_url(self._pubsub.redis_url)
                bp_config = BackPressureConfig(
                    monitoring_interval_seconds=5,
                    throttling_enabled=True
                )
                self._backpressure_manager = BackPressureManager(redis_client, bp_config)
                await self._backpressure_manager.start()
                logger.info("Back-pressure management enabled")
            
            self._connected = True
            
            logger.info(
                f"Messaging client connected for agent {self.agent_id}",
                high_performance=self.enable_high_performance,
                monitoring=self.enable_monitoring,
                backpressure=self.enable_backpressure
            )
            
        except Exception as e:
            logger.error(f"Failed to connect messaging client: {e}")
            raise CommunicationError(f"Connection failed: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from messaging infrastructure."""
        if not self._connected:
            return
        
        # Stop all consumer tasks
        for task in self._consumer_tasks:
            task.cancel()
        
        if self._consumer_tasks:
            await asyncio.gather(*self._consumer_tasks, return_exceptions=True)
        
        # Disconnect enhanced components
        if self._monitor:
            await self._monitor.stop()
        
        if self._backpressure_manager:
            await self._backpressure_manager.stop()
        
        # Disconnect communication components
        if self._hp_broker:
            await self._hp_broker.stop()
        else:
            await self._broker.disconnect()
            
        await self._pubsub.disconnect()
        
        self._connected = False
        logger.info(f"Messaging client disconnected for agent {self.agent_id}")
    
    @asynccontextmanager
    async def session(self):
        """Context manager for messaging session."""
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()
    
    async def send_message(
        self,
        to_agent: str,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        ttl: Optional[int] = None,
        correlation_id: Optional[str] = None,
        wait_for_ack: bool = False
    ) -> str:
        """
        Send a message to another agent.
        
        Args:
            to_agent: Target agent ID
            message_type: Type of message
            payload: Message payload
            priority: Message priority
            ttl: Time to live in seconds
            correlation_id: For request/response correlation
            wait_for_ack: Whether to wait for acknowledgment
            
        Returns:
            Message ID
        """
        if not self._connected:
            raise CommunicationError("Not connected")
        
        message = StreamMessage(
            from_agent=self.agent_id,
            to_agent=to_agent,
            message_type=message_type,
            payload=payload,
            priority=priority,
            ttl=ttl,
            correlation_id=correlation_id
        )
        
        # Send with retry logic
        for attempt in range(self.max_retries + 1):
            try:
                # Use appropriate broker
                if self._hp_broker:
                    message_id = await self._hp_broker.send_message(message)
                else:
                    message_id = await self._broker.send_message(message)
                
                self._stats["messages_sent"] += 1
                
                logger.debug(f"Message sent: {message_id} to {to_agent}")
                return message_id
                
            except CommunicationError as e:
                self._stats["errors"] += 1
                
                if attempt < self.max_retries and self.auto_retry:
                    self._stats["retries"] += 1
                    logger.warning(f"Message send failed (attempt {attempt + 1}), retrying: {e}")
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    logger.error(f"Message send failed after {attempt + 1} attempts: {e}")
                    raise
    
    async def broadcast(
        self,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> str:
        """
        Broadcast a message to all agents.
        
        Args:
            message_type: Type of message
            payload: Message payload  
            priority: Message priority
            
        Returns:
            Message ID
        """
        return await self.send_message(
            to_agent=None,  # Broadcast
            message_type=message_type,
            payload=payload,
            priority=priority
        )
    
    async def publish_notification(
        self,
        channel: str,
        data: Dict[str, Any]
    ) -> int:
        """
        Publish a fire-and-forget notification.
        
        Args:
            channel: Pub/Sub channel
            data: Notification data
            
        Returns:
            Number of subscribers notified
        """
        if not self._connected:
            raise CommunicationError("Not connected")
        
        try:
            return await self._pubsub.publish(channel, data)
            
        except Exception as e:
            logger.error(f"Notification publish failed: {e}")
            raise CommunicationError(f"Publish failed: {e}")
    
    async def start_consuming(
        self,
        message_types: Optional[List[MessageType]] = None,
        handler: Optional[Callable[[StreamMessage], bool]] = None
    ) -> None:
        """
        Start consuming messages for this agent.
        
        Args:
            message_types: Types of messages to consume
            handler: Message handler function
        """
        if not self._connected:
            raise CommunicationError("Not connected")
        
        if not handler:
            handler = self._default_message_handler
        
        # Start consuming from agent's personal stream
        personal_stream = f"agent_messages:{self.agent_id}"
        group_name = f"consumer_{self.agent_id}"
        
        await self._broker.consume_messages(
            stream_name=personal_stream,
            group_name=group_name,
            consumer_name=self.agent_id,
            handler=handler
        )
        
        # Also consume from broadcast stream
        broadcast_stream = "agent_messages:broadcast"
        broadcast_group = "broadcast_consumers"
        
        await self._broker.consume_messages(
            stream_name=broadcast_stream,
            group_name=broadcast_group,
            consumer_name=self.agent_id,
            handler=handler
        )
        
        logger.info(f"Started consuming messages for agent {self.agent_id}")
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable[[StreamMessage], bool]
    ) -> None:
        """
        Register a handler for specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Handler function (returns True if successful)
        """
        self._message_handlers[message_type.value] = handler
        logger.debug(f"Registered handler for {message_type.value}")
    
    async def _default_message_handler(self, message: StreamMessage) -> bool:
        """Default message handler with type-based routing."""
        try:
            # Check for specific handler
            handler = self._message_handlers.get(message.message_type.value)
            
            if handler:
                # Call specific handler
                result = handler(message)
                if asyncio.iscoroutine(result):
                    result = await result
                    
                self._stats["messages_received"] += 1
                return bool(result)
            else:
                # Log unhandled message
                logger.warning(f"No handler for message type {message.message_type.value}")
                self._stats["messages_received"] += 1
                return True  # Acknowledge anyway
                
        except Exception as e:
            logger.error(f"Error in message handler: {e}")
            self._stats["errors"] += 1
            return False
    
    async def subscribe_to_notifications(
        self,
        channel: str,
        handler: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """
        Subscribe to pub/sub notifications.
        
        Args:
            channel: Channel to subscribe to
            handler: Notification handler
        """
        if not self._connected:
            raise CommunicationError("Not connected")
        
        await self._pubsub.subscribe(channel, handler)
        logger.info(f"Subscribed to notifications on {channel}")
    
    async def send_task_request(
        self,
        to_agent: str,
        task_type: str,
        task_payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        timeout: Optional[int] = None
    ) -> str:
        """
        Send a task request to another agent.
        
        Args:
            to_agent: Target agent ID
            task_type: Type of task
            task_payload: Task parameters
            priority: Request priority
            timeout: Timeout in seconds
            
        Returns:
            Correlation ID for tracking response
        """
        correlation_id = f"task_{self.agent_id}_{datetime.utcnow().timestamp()}"
        
        payload = {
            "task_type": task_type,
            "task_payload": task_payload,
            "requester": self.agent_id,
            "timeout": timeout
        }
        
        await self.send_message(
            to_agent=to_agent,
            message_type=MessageType.TASK_REQUEST,
            payload=payload,
            priority=priority,
            correlation_id=correlation_id,
            ttl=timeout
        )
        
        return correlation_id
    
    async def send_task_result(
        self,
        to_agent: str,
        correlation_id: str,
        result: Dict[str, Any],
        success: bool = True,
        error_message: Optional[str] = None
    ) -> str:
        """
        Send a task result back to requesting agent.
        
        Args:
            to_agent: Requesting agent ID
            correlation_id: Original request correlation ID
            result: Task result data
            success: Whether task succeeded
            error_message: Error details if failed
            
        Returns:
            Message ID
        """
        payload = {
            "result": result,
            "success": success,
            "error_message": error_message,
            "responder": self.agent_id
        }
        
        return await self.send_message(
            to_agent=to_agent,
            message_type=MessageType.TASK_RESULT,
            payload=payload,
            correlation_id=correlation_id
        )
    
    async def wait_for_response(
        self,
        correlation_id: str,
        timeout: float = 30.0
    ) -> Optional[StreamMessage]:
        """
        Wait for a response message with specific correlation ID.
        
        Args:
            correlation_id: Correlation ID to wait for
            timeout: Timeout in seconds
            
        Returns:
            Response message or None if timeout
        """
        # This would require a more sophisticated implementation
        # with response tracking and async events
        # Simplified for this demo
        
        logger.warning("wait_for_response not fully implemented")
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get messaging client statistics."""
        return {
            "agent_id": self.agent_id,
            "connected": self._connected,
            "stats": self._stats.copy(),
            "active_handlers": len(self._message_handlers),
            "consumer_tasks": len(self._consumer_tasks)
        }
    
    async def get_performance_report(self) -> MessageDeliveryReport:
        """Get detailed performance report."""
        if not self._connected:
            raise CommunicationError("Not connected")
        
        if self._hp_broker:
            # Convert high-performance metrics to delivery report format
            hp_metrics = self._hp_broker.get_performance_metrics()
            return MessageDeliveryReport(
                total_sent=hp_metrics.get("messages_sent", 0),
                total_acknowledged=0,  # Not tracked in HP broker
                total_failed=0,
                success_rate=1.0,  # Simplified
                average_latency_ms=hp_metrics.get("avg_latency_ms", 0.0),
                p95_latency_ms=hp_metrics.get("p95_latency_ms", 0.0),
                p99_latency_ms=hp_metrics.get("p99_latency_ms", 0.0),
                throughput_msg_per_sec=hp_metrics.get("avg_throughput_msg_per_sec", 0.0),
                error_rate=0.0
            )
        else:
            return await self._broker.get_delivery_report()
    
    async def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics from all enabled components."""
        if not self._connected:
            raise CommunicationError("Not connected")
        
        metrics = {
            "client_stats": self.get_stats(),
            "timestamp": time.time()
        }
        
        # Add broker metrics
        if self._hp_broker:
            metrics["high_performance_broker"] = self._hp_broker.get_performance_metrics()
        elif self._broker:
            metrics["standard_broker"] = (await self._broker.get_comprehensive_metrics())
        
        # Add monitoring metrics
        if self._monitor:
            metrics["stream_monitoring"] = {
                "system_status": self._monitor.get_system_status(),
                "stream_metrics": self._monitor.get_metrics()
            }
        
        # Add back-pressure metrics
        if self._backpressure_manager:
            metrics["backpressure"] = {
                "stream_metrics": self._backpressure_manager.get_stream_metrics(),
                "consumer_metrics": self._backpressure_manager.get_consumer_metrics()
            }
        
        return metrics
    
    async def get_stream_health(self, stream_name: Optional[str] = None) -> Dict[str, Any]:
        """Get stream health information."""
        if not self._monitor:
            return {"error": "Monitoring not enabled"}
        
        return self._monitor.get_metrics(stream_name)
    
    async def get_backpressure_status(self) -> Dict[str, Any]:
        """Get back-pressure status for all streams."""
        if not self._backpressure_manager:
            return {"error": "Back-pressure management not enabled"}
        
        return {
            "stream_metrics": self._backpressure_manager.get_stream_metrics(),
            "throttle_factors": {
                stream: self._backpressure_manager.get_throttle_factor(stream)
                for stream in self._backpressure_manager.get_stream_metrics().keys()
            }
        }
    
    async def force_batch_flush(self) -> int:
        """Force flush pending message batches (high-performance mode only)."""
        if not self._hp_broker:
            return 0
        
        return await self._hp_broker.force_flush()
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics (monitoring mode only)."""
        if not self._monitor:
            return "# Monitoring not enabled\n"
        
        return self._monitor.get_prometheus_metrics()


# Convenience functions for common patterns
async def send_quick_message(
    from_agent: str,
    to_agent: str,
    message_type: MessageType,
    payload: Dict[str, Any],
    redis_url: str = None
) -> str:
    """
    Send a single message without maintaining connection.
    
    Useful for simple one-off messages.
    """
    async with MessagingClient(from_agent, redis_url=redis_url).session() as client:
        return await client.send_message(to_agent, message_type, payload)


async def broadcast_quick_message(
    from_agent: str,
    message_type: MessageType,
    payload: Dict[str, Any],
    redis_url: str = None
) -> str:
    """
    Broadcast a single message without maintaining connection.
    """
    async with MessagingClient(from_agent, redis_url=redis_url).session() as client:
        return await client.broadcast(message_type, payload)


class TaskRequestContext:
    """
    Context manager for task request/response pattern.
    
    Simplifies the common pattern of sending a task request
    and waiting for the response.
    """
    
    def __init__(
        self,
        client: MessagingClient,
        to_agent: str,
        task_type: str,
        task_payload: Dict[str, Any],
        timeout: float = 30.0
    ):
        self.client = client
        self.to_agent = to_agent
        self.task_type = task_type
        self.task_payload = task_payload
        self.timeout = timeout
        self.correlation_id: Optional[str] = None
    
    async def __aenter__(self) -> str:
        """Send task request and return correlation ID."""
        self.correlation_id = await self.client.send_task_request(
            to_agent=self.to_agent,
            task_type=self.task_type,
            task_payload=self.task_payload,
            timeout=int(self.timeout)
        )
        return self.correlation_id
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Clean up any resources."""
        pass
    
    async def wait_for_result(self) -> Optional[StreamMessage]:
        """Wait for task result."""
        if not self.correlation_id:
            raise ValueError("No active request")
        
        return await self.client.wait_for_response(
            self.correlation_id,
            self.timeout
        )