"""
Communication API endpoints for Redis Streams message passing.

Provides REST API for sending messages, consuming from streams,
monitoring performance, and managing consumer groups.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.messaging_service import (
    get_messaging_service, MessagingService, Message, 
    MessageType as UnifiedMessageType, 
    MessagePriority as UnifiedMessagePriority, 
    RoutingStrategy
)
from ...core.messaging_migration import LegacyMessageAdapter, mark_migration_complete
# Legacy imports - DEPRECATED, use messaging_service instead
# from ...core.communication import MessageBroker, SimplePubSub, CommunicationError
# from ...core.agent_communication_service import AgentCommunicationService, AgentMessage
# from ...core.redis_pubsub_manager import RedisPubSubManager, StreamStats
# from ...core.message_processor import MessageProcessor, ProcessingMetrics
from ...core.database import get_async_session
from ...models.message import (
    StreamMessage,
    MessageType as LegacyMessageType,
    MessagePriority as LegacyMessagePriority,
    MessageAudit,
    StreamInfo,
    MessageDeliveryReport
)
from ...models.agent import Agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/communication", tags=["communication"])


# Request/Response Models
class SendMessageRequest(BaseModel):
    """Request model for sending messages."""
    
    from_agent: str = Field(..., min_length=1, max_length=255)
    to_agent: Optional[str] = Field(None, max_length=255)  # None for broadcast
    message_type: UnifiedMessageType
    payload: Dict[str, Any] = Field(default_factory=dict)
    priority: UnifiedMessagePriority = UnifiedMessagePriority.NORMAL
    ttl: Optional[int] = Field(None, gt=0, description="TTL in seconds")
    correlation_id: Optional[str] = Field(None, description="For request/response correlation")
    routing_strategy: RoutingStrategy = RoutingStrategy.DIRECT


class SendMessageResponse(BaseModel):
    """Response model for sent messages."""
    
    message_id: str
    stream_name: str
    status: str = "sent"
    timestamp: float


class ConsumeMessagesRequest(BaseModel):
    """Request model for consuming messages."""
    
    stream_name: str = Field(..., min_length=1)
    group_name: str = Field(..., min_length=1)
    consumer_name: str = Field(..., min_length=1)
    count: int = Field(10, ge=1, le=100)
    block_ms: int = Field(1000, ge=0, le=60000)


class BroadcastRequest(BaseModel):
    """Request model for broadcast messages."""
    
    from_agent: str = Field(..., min_length=1, max_length=255)
    message_type: MessageType
    payload: Dict[str, Any] = Field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL


class StreamStatusResponse(BaseModel):
    """Stream status information."""
    
    stream_name: str
    length: int
    consumer_groups: int
    pending_messages: int
    last_activity: Optional[datetime]


class ReplayMessagesRequest(BaseModel):
    """Request model for message replay."""
    
    stream_name: str = Field(..., min_length=1)
    from_time: datetime
    to_time: Optional[datetime] = None
    target_stream: Optional[str] = None
    message_types: Optional[List[MessageType]] = None


# Dependency to get message broker
async def get_message_broker(
    db: AsyncSession = Depends(get_async_session)
) -> MessageBroker:
    """Get configured message broker instance."""
    broker = MessageBroker(db_session=db)
    await broker.connect()
    
    try:
        yield broker
    finally:
        await broker.disconnect()


# Dependency to get pub/sub
async def get_pubsub() -> SimplePubSub:
    """Get configured pub/sub instance."""
    pubsub = SimplePubSub()
    await pubsub.connect()
    
    try:
        yield pubsub
    finally:
        await pubsub.disconnect()


# Dependency to get enhanced communication service
async def get_communication_service() -> AgentCommunicationService:
    """Get configured agent communication service."""
    service = AgentCommunicationService(enable_streams=True)
    await service.connect()
    
    try:
        yield service
    finally:
        await service.disconnect()


# Additional Request/Response Models for Enhanced Features
class DurableMessageRequest(BaseModel):
    """Request model for durable message sending."""
    
    from_agent: str = Field(..., min_length=1, max_length=255)
    to_agent: Optional[str] = Field(None, max_length=255)
    message_type: MessageType
    payload: Dict[str, Any] = Field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    ttl: Optional[int] = Field(None, gt=0)
    correlation_id: Optional[str] = None
    acknowledgment_required: bool = False


class StreamStatsResponse(BaseModel):
    """Response model for stream statistics."""
    
    name: str
    length: int
    consumer_groups: int
    pending_messages: int
    first_entry_id: Optional[str]
    last_entry_id: Optional[str]
    
    
class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics."""
    
    messages_sent: int
    messages_received: int
    messages_failed: int
    delivery_rate: float
    error_rate: float
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_msg_per_sec: float
    
    
class ReplayRequest(BaseModel):
    """Request model for message replay."""
    
    stream_name: str = Field(..., min_length=1)
    start_id: str = Field("0-0", description="Starting message ID")
    end_id: str = Field("+", description="Ending message ID")
    count: int = Field(100, ge=1, le=1000)


async def get_messaging_service_dependency():
    """Dependency to get unified messaging service"""
    messaging = get_messaging_service()
    if not messaging._connected:
        await messaging.connect()
        await messaging.start_service()
    return messaging

@router.post("/send", response_model=SendMessageResponse, status_code=status.HTTP_201_CREATED)
async def send_message(
    request: SendMessageRequest,
    messaging: MessagingService = Depends(get_messaging_service_dependency)
) -> SendMessageResponse:
    """
    Send a message via unified messaging service with reliability guarantees.
    
    The message will be delivered with at-least-once semantics,
    circuit breaker protection, and comprehensive monitoring.
    """
    try:
        # Create unified message
        message = Message(
            type=request.message_type,
            sender=request.from_agent,
            recipient=request.to_agent,
            payload=request.payload,
            priority=request.priority,
            ttl=request.ttl,
            correlation_id=request.correlation_id,
            routing_strategy=request.routing_strategy
        )
        
        # Send message via unified messaging service
        success = await messaging.send_message(message)
        stream_name = message.get_stream_name()
        
        if success:
            logger.info(f"Message sent: {message.id} to {stream_name}")
            
            return SendMessageResponse(
                message_id=message.id,
                stream_name=stream_name,
                timestamp=message.timestamp
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Message delivery failed"
            )
        
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Message delivery failed: {str(e)}"
        )


@router.post("/broadcast", response_model=SendMessageResponse)
async def broadcast_message(
    request: BroadcastRequest,
    broker: MessageBroker = Depends(get_message_broker)
) -> SendMessageResponse:
    """
    Broadcast a message to all agents via the broadcast stream.
    
    Broadcast messages are delivered to all consumers in the broadcast
    consumer group and are useful for system-wide notifications.
    """
    try:
        # Create broadcast message (to_agent=None)
        message = StreamMessage(
            from_agent=request.from_agent,
            to_agent=None,  # Broadcast
            message_type=request.message_type,
            payload=request.payload,
            priority=request.priority
        )
        
        # Send to broadcast stream
        message_id = await broker.send_message(message)
        stream_name = message.get_stream_name()
        
        logger.info(f"Broadcast message sent: {message_id}")
        
        return SendMessageResponse(
            message_id=message_id,
            stream_name=stream_name,
            timestamp=message.timestamp
        )
        
    except CommunicationError as e:
        logger.error(f"Failed to broadcast message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Broadcast failed: {str(e)}"
        )


@router.post("/pubsub/publish")
async def publish_notification(
    channel: str = Query(..., min_length=1),
    message: Dict[str, Any] = {},
    pubsub: SimplePubSub = Depends(get_pubsub)
) -> Dict[str, Any]:
    """
    Publish a fire-and-forget notification via Redis Pub/Sub.
    
    Pub/Sub is used for urgent notifications that don't require
    delivery guarantees. Messages are not persisted.
    """
    try:
        subscriber_count = await pubsub.publish(channel, message)
        
        return {
            "channel": channel,
            "subscribers_notified": subscriber_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except CommunicationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Publish failed: {str(e)}"
        )


@router.get("/streams/{stream_name}/info", response_model=StreamInfo)
async def get_stream_info(
    stream_name: str,
    broker: MessageBroker = Depends(get_message_broker)
) -> StreamInfo:
    """
    Get detailed information about a Redis stream.
    
    Includes stream length, consumer groups, pending messages,
    and performance metrics for monitoring and debugging.
    """
    try:
        info = await broker.get_stream_info(stream_name)
        return info
        
    except CommunicationError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stream not found or inaccessible: {str(e)}"
        )


@router.get("/streams", response_model=List[StreamStatusResponse])
async def list_streams(
    pattern: str = Query("agent_messages:*", description="Stream name pattern"),
    broker: MessageBroker = Depends(get_message_broker)
) -> List[StreamStatusResponse]:
    """
    List all active streams matching the pattern.
    
    Provides overview of stream activity for monitoring
    and capacity planning.
    """
    try:
        # This would require Redis SCAN functionality
        # Simplified implementation for now
        
        # Get common streams
        common_streams = [
            "agent_messages:broadcast",
            "agent_messages:task_requests",
            "agent_messages:task_results",
            "agent_messages:coordination"
        ]
        
        streams = []
        for stream_name in common_streams:
            try:
                info = await broker.get_stream_info(stream_name)
                
                # Calculate stats
                total_pending = sum(group.pending for group in info.groups)
                
                streams.append(StreamStatusResponse(
                    stream_name=stream_name,
                    length=info.length,
                    consumer_groups=len(info.groups),
                    pending_messages=total_pending,
                    last_activity=datetime.utcnow()  # Simplified
                ))
                
            except CommunicationError:
                # Stream doesn't exist or is inaccessible
                continue
        
        return streams
        
    except Exception as e:
        logger.error(f"Failed to list streams: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve stream list"
        )


@router.get("/performance/report", response_model=MessageDeliveryReport)
async def get_performance_report(
    broker: MessageBroker = Depends(get_message_broker)
) -> MessageDeliveryReport:
    """
    Get comprehensive message delivery performance report.
    
    Includes success rates, latency metrics, throughput,
    and error rates for system monitoring.
    """
    try:
        report = await broker.get_delivery_report()
        return report
        
    except CommunicationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate report: {str(e)}"
        )


@router.post("/replay")
async def replay_messages(
    request: ReplayMessagesRequest,
    broker: MessageBroker = Depends(get_message_broker),
    db: AsyncSession = Depends(get_async_session)
) -> Dict[str, Any]:
    """
    Replay messages from audit log to a target stream.
    
    Useful for debugging, disaster recovery, and testing.
    Messages are filtered by time range and type.
    """
    try:
        # Query audit log for messages in time range
        from sqlalchemy import select
        
        query = select(MessageAudit).where(
            MessageAudit.stream_name == request.stream_name,
            MessageAudit.sent_at >= request.from_time
        )
        
        if request.to_time:
            query = query.where(MessageAudit.sent_at <= request.to_time)
            
        if request.message_types:
            query = query.where(MessageAudit.message_type.in_(request.message_types))
        
        result = await db.execute(query)
        audit_records = result.scalars().all()
        
        # Replay messages
        target_stream = request.target_stream or f"{request.stream_name}:replay"
        replayed_count = 0
        
        for audit in audit_records:
            try:
                # Reconstruct message
                message = StreamMessage(
                    from_agent=str(audit.from_agent_id),
                    to_agent=str(audit.to_agent_id) if audit.to_agent_id else None,
                    message_type=audit.message_type,
                    payload=audit.payload,
                    priority=audit.priority,
                    correlation_id=audit.correlation_id
                )
                
                # Send to target stream (override stream name)
                original_get_stream = message.get_stream_name
                message.get_stream_name = lambda: target_stream
                
                await broker.send_message(message)
                replayed_count += 1
                
                # Restore original method
                message.get_stream_name = original_get_stream
                
            except Exception as e:
                logger.error(f"Failed to replay message {audit.message_id}: {e}")
                continue
        
        return {
            "replayed_count": replayed_count,
            "target_stream": target_stream,
            "time_range": {
                "from": request.from_time.isoformat(),
                "to": request.to_time.isoformat() if request.to_time else None
            }
        }
        
    except Exception as e:
        logger.error(f"Replay failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Message replay failed: {str(e)}"
        )


@router.get("/health")
async def health_check(
    broker: MessageBroker = Depends(get_message_broker)
) -> Dict[str, Any]:
    """
    Health check for communication system.
    
    Verifies Redis connectivity and basic functionality.
    """
    try:
        # Test Redis connection
        if not broker._redis:
            raise CommunicationError("Not connected to Redis")
            
        await broker._redis.ping()
        
        # Get basic metrics
        report = await broker.get_delivery_report()
        
        return {
            "status": "healthy",
            "redis_connected": True,
            "messages_sent": report.total_sent,
            "success_rate": report.success_rate,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Communication system unhealthy: {str(e)}"
        )


@router.websocket("/streams/{stream_name}/subscribe")
async def subscribe_to_stream(
    websocket,
    stream_name: str,
    group_name: str = Query("websocket_consumers"),
    consumer_name: str = Query(None)
):
    """
    WebSocket endpoint for real-time stream subscription.
    
    Allows clients to receive messages in real-time via WebSocket
    connection with automatic consumer group management.
    """
    if not consumer_name:
        consumer_name = f"websocket_{id(websocket)}"
    
    await websocket.accept()
    
    try:
        broker = MessageBroker()
        await broker.connect()
        
        # Message handler for WebSocket
        async def handle_message(message: StreamMessage) -> bool:
            try:
                await websocket.send_json({
                    "id": message.id,
                    "from_agent": message.from_agent,
                    "to_agent": message.to_agent,
                    "message_type": message.message_type.value,
                    "payload": message.payload,
                    "timestamp": message.timestamp
                })
                return True
                
            except Exception as e:
                logger.error(f"WebSocket send failed: {e}")
                return False
        
        # Start consuming
        await broker.consume_messages(
            stream_name=stream_name,
            group_name=group_name,
            consumer_name=consumer_name,
            handler=handle_message
        )
        
        # Keep connection alive
        while True:
            try:
                await websocket.receive_text()
            except:
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        
    finally:
        if 'broker' in locals():
            await broker.disconnect()


# Enhanced API Endpoints for Redis Streams and Pub/Sub

@router.post("/durable/send", response_model=SendMessageResponse, status_code=status.HTTP_201_CREATED)
async def send_durable_message(
    request: DurableMessageRequest,
    service: AgentCommunicationService = Depends(get_communication_service)
) -> SendMessageResponse:
    """
    Send a durable message via Redis Streams with guaranteed delivery.
    
    Provides at-least-once delivery semantics, consumer group support,
    and automatic retry with dead letter queue functionality.
    """
    try:
        # Create agent message
        message = AgentMessage(
            id=str(uuid.uuid4()),
            from_agent=request.from_agent,
            to_agent=request.to_agent,
            type=request.message_type,
            payload=request.payload,
            timestamp=time.time(),
            priority=request.priority,
            ttl=request.ttl,
            correlation_id=request.correlation_id,
            acknowledgment_required=request.acknowledgment_required
        )
        
        # Send via streams
        stream_message_id = await service.send_durable_message(message)
        
        if not stream_message_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send durable message"
            )
        
        logger.info(f"Durable message sent: {stream_message_id}")
        
        return SendMessageResponse(
            message_id=stream_message_id,
            stream_name=message.get_channel_name(),
            timestamp=message.timestamp
        )
        
    except Exception as e:
        logger.error(f"Failed to send durable message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Durable message delivery failed: {str(e)}"
        )


@router.post("/pubsub/send", response_model=SendMessageResponse)
async def send_pubsub_message(
    request: DurableMessageRequest,
    service: AgentCommunicationService = Depends(get_communication_service)
) -> SendMessageResponse:
    """
    Send a message via Redis Pub/Sub for fast, fire-and-forget delivery.
    
    Provides low-latency delivery without persistence guarantees.
    Suitable for notifications and real-time updates.
    """
    try:
        # Create agent message
        message = AgentMessage(
            id=str(uuid.uuid4()),
            from_agent=request.from_agent,
            to_agent=request.to_agent,
            type=request.message_type,
            payload=request.payload,
            timestamp=time.time(),
            priority=request.priority,
            ttl=request.ttl,
            correlation_id=request.correlation_id,
            acknowledgment_required=request.acknowledgment_required
        )
        
        # Send via pub/sub
        success = await service.send_message(message)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send pub/sub message"
            )
        
        logger.info(f"Pub/Sub message sent: {message.id}")
        
        return SendMessageResponse(
            message_id=message.id,
            stream_name=message.get_channel_name(),
            timestamp=message.timestamp
        )
        
    except Exception as e:
        logger.error(f"Failed to send pub/sub message: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pub/Sub message delivery failed: {str(e)}"
        )


@router.get("/streams/{stream_name}/stats", response_model=StreamStatsResponse)
async def get_enhanced_stream_stats(
    stream_name: str,
    service: AgentCommunicationService = Depends(get_communication_service)
) -> StreamStatsResponse:
    """
    Get comprehensive statistics for a Redis Stream.
    
    Includes consumer group information, pending messages,
    and performance metrics for monitoring and debugging.
    """
    try:
        stats = await service.get_stream_stats(stream_name)
        
        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stream {stream_name} not found or streams not enabled"
            )
        
        # Calculate totals
        total_consumer_groups = len(stats.groups)
        total_pending = sum(group.pending_count for group in stats.groups)
        
        return StreamStatsResponse(
            name=stats.name,
            length=stats.length,
            consumer_groups=total_consumer_groups,
            pending_messages=total_pending,
            first_entry_id=stats.first_entry_id,
            last_entry_id=stats.last_entry_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get stream stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve stream statistics: {str(e)}"
        )


@router.get("/metrics/performance", response_model=PerformanceMetricsResponse)
async def get_enhanced_performance_metrics(
    service: AgentCommunicationService = Depends(get_communication_service)
) -> PerformanceMetricsResponse:
    """
    Get comprehensive performance metrics for the communication system.
    
    Includes throughput, latency percentiles, success rates,
    and error rates for system monitoring and optimization.
    """
    try:
        comprehensive_metrics = await service.get_comprehensive_metrics()
        
        # Extract relevant metrics
        msg_metrics = comprehensive_metrics.get("message_metrics", {})
        streams_metrics = comprehensive_metrics.get("streams_metrics", {})
        
        return PerformanceMetricsResponse(
            messages_sent=msg_metrics.get("total_sent", 0) + streams_metrics.get("messages_sent", 0),
            messages_received=msg_metrics.get("total_acknowledged", 0) + streams_metrics.get("messages_received", 0),
            messages_failed=msg_metrics.get("total_failed", 0) + streams_metrics.get("messages_failed", 0),
            delivery_rate=msg_metrics.get("delivery_rate", 0.0),
            error_rate=msg_metrics.get("error_rate", 0.0),
            average_latency_ms=msg_metrics.get("average_latency_ms", 0.0),
            p95_latency_ms=msg_metrics.get("p95_latency_ms", 0.0),
            p99_latency_ms=msg_metrics.get("p99_latency_ms", 0.0),
            throughput_msg_per_sec=msg_metrics.get("throughput_msg_per_sec", 0.0)
        )
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve performance metrics: {str(e)}"
        )


@router.post("/streams/{stream_name}/replay")
async def replay_stream_messages(
    stream_name: str,
    request: ReplayRequest,
    service: AgentCommunicationService = Depends(get_communication_service)
) -> Dict[str, Any]:
    """
    Replay messages from a Redis Stream for debugging or recovery.
    
    Retrieves messages from the specified range and returns them
    for analysis or re-processing.
    """
    try:
        messages = await service.replay_messages(
            stream_name=request.stream_name or stream_name,
            start_id=request.start_id,
            end_id=request.end_id,
            count=request.count
        )
        
        return {
            "stream_name": stream_name,
            "messages_found": len(messages),
            "start_id": request.start_id,
            "end_id": request.end_id,
            "messages": messages
        }
        
    except Exception as e:
        logger.error(f"Failed to replay messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Message replay failed: {str(e)}"
        )


@router.get("/streams/{stream_name}/dlq")
async def get_dead_letter_queue(
    stream_name: str,
    count: int = Query(100, ge=1, le=1000),
    service: AgentCommunicationService = Depends(get_communication_service)
) -> Dict[str, Any]:
    """
    Get messages from the dead letter queue for a stream.
    
    Retrieves messages that failed processing after maximum
    retry attempts for manual intervention or analysis.
    """
    try:
        dlq_messages = await service.get_dead_letter_messages(
            original_stream=stream_name,
            count=count
        )
        
        return {
            "original_stream": stream_name,
            "dlq_stream": f"{stream_name}:dlq",
            "messages_found": len(dlq_messages),
            "messages": dlq_messages
        }
        
    except Exception as e:
        logger.error(f"Failed to get DLQ messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dead letter messages: {str(e)}"
        )


@router.get("/health/comprehensive")
async def comprehensive_health_check(
    service: AgentCommunicationService = Depends(get_communication_service)
) -> Dict[str, Any]:
    """
    Comprehensive health check for the enhanced communication system.
    
    Checks Redis connectivity, stream functionality, pub/sub status,
    and overall system health with detailed diagnostics.
    """
    try:
        health_status = await service.health_check()
        comprehensive_metrics = await service.get_comprehensive_metrics()
        
        # Determine overall health
        is_healthy = (
            health_status.get("status") == "healthy" and
            comprehensive_metrics.get("connection_status", {}).get("connected", False)
        )
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "redis_connection": health_status.get("connected", False),
                "pub_sub": health_status.get("status") == "healthy",
                "streams": comprehensive_metrics.get("streams_metrics", {}).get("active_consumers", 0) >= 0,
                "circuit_breaker": not comprehensive_metrics.get("streams_metrics", {}).get("circuit_breaker_status", {}).get("open", True)
            },
            "metrics": {
                "ping_latency_ms": health_status.get("ping_latency_ms", 0),
                "delivery_rate": health_status.get("delivery_rate", 0),
                "error_rate": health_status.get("error_rate", 0),
                "active_subscriptions": comprehensive_metrics.get("subscription_status", {}).get("active_subscriptions", 0)
            },
            "detailed_status": comprehensive_metrics
        }
        
    except Exception as e:
        logger.error(f"Comprehensive health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
            "components": {
                "redis_connection": False,
                "pub_sub": False,
                "streams": False,
                "circuit_breaker": False
            }
        }