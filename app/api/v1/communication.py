"""
Communication API endpoints for Redis Streams message passing.

Provides REST API for sending messages, consuming from streams,
monitoring performance, and managing consumer groups.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.communication import MessageBroker, SimplePubSub, CommunicationError
from ...core.database import get_async_session
from ...models.message import (
    StreamMessage,
    MessageType,
    MessagePriority,
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
    message_type: MessageType
    payload: Dict[str, Any] = Field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    ttl: Optional[int] = Field(None, gt=0, description="TTL in seconds")
    correlation_id: Optional[str] = Field(None, description="For request/response correlation")


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


@router.post("/send", response_model=SendMessageResponse, status_code=status.HTTP_201_CREATED)
async def send_message(
    request: SendMessageRequest,
    broker: MessageBroker = Depends(get_message_broker)
) -> SendMessageResponse:
    """
    Send a message via Redis Streams with reliability guarantees.
    
    The message will be delivered with at-least-once semantics,
    signed for authenticity, and audited for compliance.
    """
    try:
        # Create stream message
        message = StreamMessage(
            from_agent=request.from_agent,
            to_agent=request.to_agent,
            message_type=request.message_type,
            payload=request.payload,
            priority=request.priority,
            ttl=request.ttl,
            correlation_id=request.correlation_id
        )
        
        # Send message
        message_id = await broker.send_message(message)
        stream_name = message.get_stream_name()
        
        logger.info(f"Message sent: {message_id} to {stream_name}")
        
        return SendMessageResponse(
            message_id=message_id,
            stream_name=stream_name,
            timestamp=message.timestamp
        )
        
    except CommunicationError as e:
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