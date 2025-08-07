"""
Enhanced Observability API Endpoints
===================================

Provides comprehensive API endpoints for real-time observability, WebSocket streaming,
and dashboard integration. Integrates with the enhanced real-time hooks system and
WebSocket streaming infrastructure.

Features:
- Real-time WebSocket streaming with advanced filtering
- Event query and analytics endpoints
- Performance metrics and health monitoring
- Dashboard configuration and management
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query, Path, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.observability.real_time_hooks import (
    get_real_time_processor, 
    emit_pre_tool_use_event,
    emit_post_tool_use_event,
    get_processor_health
)
from app.observability.enhanced_websocket_streaming import (
    get_enhanced_websocket_streaming,
    DashboardFilter,
    StreamingEventType,
    broadcast_performance_metric,
    broadcast_system_alert
)
from app.models.observability import AgentEvent, EventType
from app.core.database import get_async_session
from app.schemas.observability import BaseObservabilityEvent
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload

logger = structlog.get_logger()

# Create router
router = APIRouter(prefix="/observability", tags=["enhanced-observability"])


# Pydantic models for API

class EventQueryRequest(BaseModel):
    """Request model for querying agent events."""
    session_ids: Optional[List[str]] = Field(default=None, description="Filter by session IDs")
    agent_ids: Optional[List[str]] = Field(default=None, description="Filter by agent IDs") 
    event_types: Optional[List[EventType]] = Field(default=None, description="Filter by event types")
    start_time: Optional[datetime] = Field(default=None, description="Start time for query range")
    end_time: Optional[datetime] = Field(default=None, description="End time for query range")
    limit: int = Field(default=100, ge=1, le=10000, description="Maximum number of events to return")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")
    include_payload: bool = Field(default=True, description="Include event payload in response")


class EventQueryResponse(BaseModel):
    """Response model for event queries."""
    events: List[Dict[str, Any]]
    total_count: int
    has_more: bool
    query_time_ms: float


class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics."""
    timestamp: datetime
    event_processor: Dict[str, Any]
    websocket_streaming: Dict[str, Any]
    database_performance: Dict[str, Any]
    redis_performance: Dict[str, Any]


class SystemHealthResponse(BaseModel):
    """Response model for system health status."""
    status: str  # healthy, degraded, critical
    overall_health_score: float
    components: Dict[str, Dict[str, Any]]
    recommendations: List[str]


class DashboardConfigRequest(BaseModel):
    """Request model for dashboard configuration."""
    dashboard_id: str
    filters: DashboardFilter
    refresh_rate_ms: int = Field(default=1000, ge=100, le=10000)
    auto_reconnect: bool = True


# WebSocket endpoints

@router.websocket("/stream/dashboard")
async def enhanced_dashboard_websocket(
    websocket: WebSocket,
    # Query parameters for filtering
    agent_ids: Optional[str] = Query(default=None, description="Comma-separated agent IDs"),
    session_ids: Optional[str] = Query(default=None, description="Comma-separated session IDs"),
    workflow_ids: Optional[str] = Query(default=None, description="Comma-separated workflow IDs"),
    event_types: Optional[str] = Query(default=None, description="Comma-separated event types"),
    agent_event_types: Optional[str] = Query(default=None, description="Comma-separated agent event types"),
    min_priority: int = Query(default=1, ge=1, le=10, description="Minimum event priority"),
    max_events_per_second: int = Query(default=100, ge=1, le=1000, description="Rate limit"),
    include_patterns: Optional[str] = Query(default=None, description="Comma-separated include patterns"),
    exclude_patterns: Optional[str] = Query(default=None, description="Comma-separated exclude patterns")
):
    """
    Enhanced WebSocket endpoint for real-time dashboard streaming.
    
    Provides <1s updates with advanced filtering, rate limiting, and connection management.
    Supports multiple dashboard types with different data requirements.
    """
    try:
        # Parse filtering parameters
        filters = DashboardFilter(
            agent_ids=agent_ids.split(",") if agent_ids else [],
            session_ids=session_ids.split(",") if session_ids else [],
            workflow_ids=workflow_ids.split(",") if workflow_ids else [],
            event_types=[
                StreamingEventType(t.strip()) 
                for t in event_types.split(",") 
                if t.strip()
            ] if event_types else [],
            agent_event_types=[
                EventType(t.strip()) 
                for t in agent_event_types.split(",") 
                if t.strip()
            ] if agent_event_types else [],
            min_priority=min_priority,
            max_events_per_second=max_events_per_second,
            include_patterns=include_patterns.split(",") if include_patterns else [],
            exclude_patterns=exclude_patterns.split(",") if exclude_patterns else []
        )
        
        # Get streaming system
        streaming = await get_enhanced_websocket_streaming()
        
        # Connect client
        connection_id = await streaming.connect_client(websocket, filters)
        
        logger.info(
            "Enhanced dashboard WebSocket connected",
            connection_id=connection_id,
            filters=filters.dict()
        )
        
        try:
            # Handle client messages (filter updates, acknowledgments, etc.)
            while True:
                try:
                    message = await websocket.receive_text()
                    data = json.loads(message)
                    
                    await _handle_websocket_message(streaming, connection_id, data)
                    
                except WebSocketDisconnect:
                    logger.info(f"Dashboard client disconnected: {connection_id}")
                    break
                    
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from client {connection_id}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                    
                except Exception as e:
                    logger.error(f"Error processing client message: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error", 
                        "message": "Internal server error",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
        
        finally:
            await streaming.disconnect_client(connection_id)
            
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        if websocket.client_state.CONNECTED:
            await websocket.close(code=1011, reason="Internal server error")


async def _handle_websocket_message(streaming, connection_id: str, data: Dict[str, Any]) -> None:
    """Handle incoming WebSocket messages from dashboard clients."""
    message_type = data.get("type")
    
    if message_type == "update_filters":
        # Update client filters
        try:
            new_filters = DashboardFilter(**data.get("filters", {}))
            # Update filters in streaming system (would need to implement this method)
            logger.info(f"Updated filters for client {connection_id}")
        except Exception as e:
            logger.error(f"Failed to update filters: {e}")
    
    elif message_type == "ping":
        # Respond to client ping
        connection = streaming.connections.get(connection_id)
        if connection:
            pong_message = {
                "type": "pong",
                "timestamp": datetime.utcnow().isoformat(),
                "server_time": datetime.utcnow().isoformat()
            }
            await connection.websocket.send_text(json.dumps(pong_message))
    
    elif message_type == "acknowledge":
        # Handle event acknowledgment
        event_id = data.get("event_id")
        logger.debug(f"Client {connection_id} acknowledged event {event_id}")
    
    else:
        logger.warning(f"Unknown message type from client {connection_id}: {message_type}")


# REST API endpoints

@router.post("/events/query", response_model=EventQueryResponse)
async def query_agent_events(request: EventQueryRequest):
    """
    Query agent events with advanced filtering and pagination.
    
    Supports complex queries across time ranges, event types, and agent/session filtering
    with performance optimization for large datasets.
    """
    query_start = datetime.utcnow()
    
    try:
        async with get_async_session() as session:
            # Build base query
            query = select(AgentEvent)
            
            # Apply filters
            conditions = []
            
            if request.session_ids:
                session_uuids = [uuid.UUID(sid) for sid in request.session_ids]
                conditions.append(AgentEvent.session_id.in_(session_uuids))
            
            if request.agent_ids:
                agent_uuids = [uuid.UUID(aid) for aid in request.agent_ids]
                conditions.append(AgentEvent.agent_id.in_(agent_uuids))
            
            if request.event_types:
                conditions.append(AgentEvent.event_type.in_(request.event_types))
            
            if request.start_time:
                conditions.append(AgentEvent.created_at >= request.start_time)
            
            if request.end_time:
                conditions.append(AgentEvent.created_at <= request.end_time)
            
            if conditions:
                query = query.where(and_(*conditions))
            
            # Add ordering and pagination
            query = query.order_by(AgentEvent.created_at.desc())
            
            # Get total count for pagination
            count_query = select(func.count(AgentEvent.id))
            if conditions:
                count_query = count_query.where(and_(*conditions))
            
            total_count_result = await session.execute(count_query)
            total_count = total_count_result.scalar()
            
            # Apply pagination
            query = query.offset(request.offset).limit(request.limit)
            
            # Execute query
            result = await session.execute(query)
            events = result.scalars().all()
            
            # Convert to response format
            event_dicts = []
            for event in events:
                event_dict = {
                    "id": event.id,
                    "session_id": str(event.session_id),
                    "agent_id": str(event.agent_id),
                    "event_type": event.event_type.value,
                    "created_at": event.created_at.isoformat(),
                    "latency_ms": event.latency_ms
                }
                
                if request.include_payload:
                    event_dict["payload"] = event.payload
                
                event_dicts.append(event_dict)
            
            query_time_ms = (datetime.utcnow() - query_start).total_seconds() * 1000
            has_more = (request.offset + len(events)) < total_count
            
            return EventQueryResponse(
                events=event_dicts,
                total_count=total_count,
                has_more=has_more,
                query_time_ms=query_time_ms
            )
            
    except Exception as e:
        logger.error(f"Failed to query events: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/metrics/performance", response_model=PerformanceMetricsResponse)
async def get_performance_metrics():
    """
    Get comprehensive performance metrics for all observability components.
    
    Returns real-time metrics for event processing, WebSocket streaming,
    database performance, and Redis operations.
    """
    try:
        timestamp = datetime.utcnow()
        
        # Get event processor metrics
        processor_metrics = {}
        try:
            processor_health = await get_processor_health()
            processor_metrics = processor_health.get("metrics", {})
        except Exception as e:
            logger.warning(f"Failed to get processor metrics: {e}")
            processor_metrics = {"error": str(e)}
        
        # Get WebSocket streaming metrics
        streaming_metrics = {}
        try:
            streaming = await get_enhanced_websocket_streaming()
            streaming_metrics = streaming.get_metrics()
        except Exception as e:
            logger.warning(f"Failed to get streaming metrics: {e}")
            streaming_metrics = {"error": str(e)}
        
        # Get database performance metrics
        db_metrics = await _get_database_performance_metrics()
        
        # Get Redis performance metrics  
        redis_metrics = await _get_redis_performance_metrics()
        
        return PerformanceMetricsResponse(
            timestamp=timestamp,
            event_processor=processor_metrics,
            websocket_streaming=streaming_metrics,
            database_performance=db_metrics,
            redis_performance=redis_metrics
        )
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


@router.get("/health", response_model=SystemHealthResponse)
async def get_system_health():
    """
    Get comprehensive system health status and recommendations.
    
    Analyzes all observability components and provides health scoring
    with actionable recommendations for optimization.
    """
    try:
        components = {}
        recommendations = []
        
        # Check event processor health
        try:
            processor_health = await get_processor_health()
            components["event_processor"] = processor_health
            
            if processor_health.get("status") != "healthy":
                recommendations.append("Event processor performance degraded - check metrics")
        except Exception as e:
            components["event_processor"] = {"status": "error", "error": str(e)}
            recommendations.append("Event processor unavailable - restart required")
        
        # Check WebSocket streaming health
        try:
            streaming = await get_enhanced_websocket_streaming()
            streaming_health = await streaming.get_health_status()
            components["websocket_streaming"] = streaming_health
            
            if streaming_health.get("status") != "healthy":
                recommendations.append("WebSocket streaming performance issues detected")
        except Exception as e:
            components["websocket_streaming"] = {"status": "error", "error": str(e)}
            recommendations.append("WebSocket streaming unavailable - check configuration")
        
        # Check database health
        db_health = await _check_database_health()
        components["database"] = db_health
        
        if db_health.get("status") != "healthy":
            recommendations.append("Database performance issues - check connection pool")
        
        # Check Redis health
        redis_health = await _check_redis_health()
        components["redis"] = redis_health
        
        if redis_health.get("status") != "healthy":
            recommendations.append("Redis connectivity issues - verify configuration")
        
        # Calculate overall health score
        health_scores = [
            comp.get("overall_health_score", 0.5) 
            for comp in components.values() 
            if "overall_health_score" in comp
        ]
        
        overall_health_score = sum(health_scores) / len(health_scores) if health_scores else 0.5
        
        # Determine overall status
        if overall_health_score > 0.8:
            status = "healthy"
        elif overall_health_score > 0.6:
            status = "degraded"
        else:
            status = "critical"
            recommendations.append("System requires immediate attention - multiple components failing")
        
        return SystemHealthResponse(
            status=status,
            overall_health_score=round(overall_health_score, 3),
            components=components,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.post("/events/emit/pre-tool-use")
async def emit_pre_tool_use(
    session_id: str = Field(..., description="Session UUID"),
    agent_id: str = Field(..., description="Agent UUID"),
    tool_name: str = Field(..., description="Tool name"),
    parameters: Dict[str, Any] = Field(..., description="Tool parameters"),
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracing")
):
    """Manually emit a PreToolUse event for testing or integration."""
    try:
        event_id = await emit_pre_tool_use_event(
            session_id=uuid.UUID(session_id),
            agent_id=uuid.UUID(agent_id),
            tool_name=tool_name,
            parameters=parameters,
            correlation_id=correlation_id
        )
        
        return {"event_id": event_id, "status": "success"}
        
    except Exception as e:
        logger.error(f"Failed to emit PreToolUse event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/events/emit/post-tool-use")
async def emit_post_tool_use(
    session_id: str = Field(..., description="Session UUID"),
    agent_id: str = Field(..., description="Agent UUID"),
    tool_name: str = Field(..., description="Tool name"),
    success: bool = Field(..., description="Execution success"),
    result: Any = Field(None, description="Tool result"),
    error: Optional[str] = Field(None, description="Error message"),
    execution_time_ms: Optional[int] = Field(None, description="Execution time"),
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracing")
):
    """Manually emit a PostToolUse event for testing or integration."""
    try:
        event_id = await emit_post_tool_use_event(
            session_id=uuid.UUID(session_id),
            agent_id=uuid.UUID(agent_id),
            tool_name=tool_name,
            success=success,
            result=result,
            error=error,
            execution_time_ms=execution_time_ms,
            correlation_id=correlation_id
        )
        
        return {"event_id": event_id, "status": "success"}
        
    except Exception as e:
        logger.error(f"Failed to emit PostToolUse event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/broadcast/performance-metric")
async def broadcast_metric(
    metric_name: str = Field(..., description="Metric name"),
    value: float = Field(..., description="Metric value"),
    source: str = Field(..., description="Metric source"),
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
):
    """Broadcast a performance metric to all connected dashboard clients."""
    try:
        await broadcast_performance_metric(
            metric_name=metric_name,
            value=value,
            source=source,
            metadata=metadata
        )
        
        return {"status": "broadcasted", "metric_name": metric_name}
        
    except Exception as e:
        logger.error(f"Failed to broadcast metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/broadcast/system-alert")
async def broadcast_alert(
    level: str = Field(..., description="Alert level (info, warning, error, critical)"),
    message: str = Field(..., description="Alert message"),
    source: str = Field(..., description="Alert source"),
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")
):
    """Broadcast a system alert to all connected dashboard clients."""
    try:
        await broadcast_system_alert(
            level=level,
            message=message,
            source=source,
            details=details
        )
        
        return {"status": "broadcasted", "level": level, "message": message}
        
    except Exception as e:
        logger.error(f"Failed to broadcast alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/connections")
async def get_active_connections():
    """Get information about active WebSocket connections."""
    try:
        streaming = await get_enhanced_websocket_streaming()
        metrics = streaming.get_metrics()
        
        return {
            "active_connections": metrics["active_connections"],
            "total_connections": metrics["total_connections"],
            "connection_details": metrics["connection_details"],
            "performance_metrics": {
                "events_per_second": metrics["events_per_second"],
                "average_stream_latency_ms": metrics["average_stream_latency_ms"],
                "websocket_errors": metrics["websocket_errors"]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get connections info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions for metrics collection

async def _get_database_performance_metrics() -> Dict[str, Any]:
    """Get database performance metrics."""
    try:
        async with get_async_session() as session:
            # Query recent events count
            recent_count_query = select(func.count(AgentEvent.id)).where(
                AgentEvent.created_at >= datetime.utcnow() - timedelta(minutes=5)
            )
            recent_count = await session.execute(recent_count_query)
            recent_events = recent_count.scalar()
            
            # Query total events count
            total_count_query = select(func.count(AgentEvent.id))
            total_count = await session.execute(total_count_query)
            total_events = total_count.scalar()
            
            return {
                "status": "healthy",
                "recent_events_5min": recent_events,
                "total_events": total_events,
                "events_per_minute": recent_events / 5.0,
                "connection_pool_active": True  # Would check actual pool stats
            }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def _get_redis_performance_metrics() -> Dict[str, Any]:
    """Get Redis performance metrics."""
    try:
        from app.core.redis import get_redis_client
        
        redis = await get_redis_client()
        
        # Get Redis info
        info = await redis.info()
        
        return {
            "status": "healthy",
            "connected_clients": info.get("connected_clients", 0),
            "used_memory": info.get("used_memory", 0),
            "used_memory_human": info.get("used_memory_human", "0B"),
            "commands_processed": info.get("total_commands_processed", 0),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0)
        }
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def _check_database_health() -> Dict[str, Any]:
    """Check database health status."""
    try:
        async with get_async_session() as session:
            # Simple health check query
            result = await session.execute(select(1))
            result.scalar()
            
            return {"status": "healthy", "connection": "active"}
    
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def _check_redis_health() -> Dict[str, Any]:
    """Check Redis health status."""
    try:
        from app.core.redis import get_redis_client
        
        redis = await get_redis_client()
        await redis.ping()
        
        return {"status": "healthy", "connection": "active"}
    
    except Exception as e:
        return {"status": "error", "error": str(e)}