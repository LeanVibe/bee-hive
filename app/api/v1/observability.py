"""
Observability API routes for LeanVibe Agent Hive 2.0

Provides REST endpoints for event ingestion, querying, metrics exposition,
and health monitoring of the observability system.
"""

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session_dependency
from app.core.event_processor import get_event_processor
from app.core.redis import get_redis
from app.models.observability import AgentEvent, EventType
from app.observability.hooks import get_hook_interceptor

logger = structlog.get_logger()

router = APIRouter(prefix="/observability", tags=["observability"])


# === Request/Response Models ===

class EventRequest(BaseModel):
    """Request model for posting events."""
    
    event_type: str = Field(..., description="Event type (PreToolUse, PostToolUse, etc.)")
    agent_id: str = Field(..., description="Agent UUID")
    session_id: str = Field(..., description="Session UUID")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event payload")
    timestamp: Optional[str] = Field(None, description="Event timestamp (ISO format)")
    
    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, v):
        """Validate event type is supported."""
        valid_types = [et.value for et in EventType]
        if v not in valid_types:
            raise ValueError(f"Invalid event type. Must be one of: {valid_types}")
        return v
    
    @field_validator("agent_id", "session_id")
    @classmethod
    def validate_uuid(cls, v):
        """Validate UUID format."""
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError("Invalid UUID format")
        return v


class EventResponse(BaseModel):
    """Response model for event operations."""
    
    status: str = Field(..., description="Operation status")
    event_id: Optional[str] = Field(None, description="Event ID if created")
    message: Optional[str] = Field(None, description="Additional message")


class EventQueryResponse(BaseModel):
    """Response model for event queries."""
    
    events: List[Dict[str, Any]] = Field(..., description="List of events")
    pagination: Dict[str, Any] = Field(..., description="Pagination information")
    total_count: Optional[int] = Field(None, description="Total event count")


class HealthResponse(BaseModel):
    """Response model for health checks."""
    
    status: str = Field(..., description="Overall health status")
    components: Dict[str, Any] = Field(..., description="Component health status")
    timestamp: str = Field(..., description="Health check timestamp")


# === API Endpoints ===

@router.post("/event", response_model=EventResponse, status_code=201)
async def create_event(
    event_request: EventRequest,
    request: Request,
    session: AsyncSession = Depends(get_session_dependency)
) -> EventResponse:
    """
    Create a new observability event.
    
    Accepts events from agents and processes them through the observability pipeline.
    Events are streamed to Redis and persisted to PostgreSQL for analysis.
    """
    try:
        # Get event processor
        processor = get_event_processor()
        if not processor:
            raise HTTPException(
                status_code=503,
                detail="Event processor not available"
            )
        
        # Parse UUIDs
        session_id = uuid.UUID(event_request.session_id)
        agent_id = uuid.UUID(event_request.agent_id)
        event_type = EventType(event_request.event_type)
        
        # Add request context to payload
        payload = event_request.payload.copy()
        payload.setdefault("source", "api")
        payload.setdefault("user_agent", request.headers.get("user-agent", "unknown"))
        
        if hasattr(request.state, "correlation_id"):
            payload.setdefault("correlation_id", request.state.correlation_id)
        
        # Process event
        stream_id = await processor.process_event(
            session_id=session_id,
            agent_id=agent_id,
            event_type=event_type,
            payload=payload
        )
        
        logger.info(
            "📝 Event created via API",
            event_type=event_request.event_type,
            session_id=event_request.session_id,
            agent_id=event_request.agent_id,
            stream_id=stream_id
        )
        
        return EventResponse(
            status="queued",
            event_id=stream_id,
            message="Event successfully queued for processing"
        )
        
    except ValueError as e:
        logger.warning("❌ Invalid event request", error=str(e))
        raise HTTPException(status_code=422, detail=str(e))
    
    except Exception as e:
        logger.error("❌ Failed to create event", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to process event"
        )


@router.get("/events", response_model=EventQueryResponse)
async def get_events(
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    from_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    to_time: Optional[str] = Query(None, description="End time (ISO format)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of events"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    session: AsyncSession = Depends(get_session_dependency)
) -> EventQueryResponse:
    """
    Query observability events with filtering and pagination.
    
    Supports filtering by session, agent, event type, and time range.
    Returns paginated results with total count for efficient client-side pagination.
    """
    try:
        # Build query conditions
        conditions = []
        
        if session_id:
            try:
                session_uuid = uuid.UUID(session_id)
                conditions.append(AgentEvent.session_id == session_uuid)
            except ValueError:
                raise HTTPException(status_code=422, detail="Invalid session_id UUID format")
        
        if agent_id:
            try:
                agent_uuid = uuid.UUID(agent_id)
                conditions.append(AgentEvent.agent_id == agent_uuid)
            except ValueError:
                raise HTTPException(status_code=422, detail="Invalid agent_id UUID format")
        
        if event_type:
            try:
                event_type_enum = EventType(event_type)
                conditions.append(AgentEvent.event_type == event_type_enum)
            except ValueError:
                raise HTTPException(status_code=422, detail=f"Invalid event_type: {event_type}")
        
        if from_time:
            try:
                from_dt = datetime.fromisoformat(from_time.replace('Z', '+00:00'))
                conditions.append(AgentEvent.created_at >= from_dt)
            except ValueError:
                raise HTTPException(status_code=422, detail="Invalid from_time format")
        
        if to_time:
            try:
                to_dt = datetime.fromisoformat(to_time.replace('Z', '+00:00'))
                conditions.append(AgentEvent.created_at <= to_dt)
            except ValueError:
                raise HTTPException(status_code=422, detail="Invalid to_time format")
        
        # Build base query
        base_query = select(AgentEvent)
        if conditions:
            base_query = base_query.where(and_(*conditions))
        
        # Get total count
        count_query = select(func.count(AgentEvent.id))
        if conditions:
            count_query = count_query.where(and_(*conditions))
        
        total_result = await session.execute(count_query)
        total_count = total_result.scalar() or 0
        
        # Get paginated events
        query = (
            base_query
            .order_by(desc(AgentEvent.created_at))
            .limit(limit)
            .offset(offset)
        )
        
        result = await session.execute(query)
        events = result.scalars().all()
        
        # Convert to response format
        event_data = []
        for event in events:
            event_dict = {
                "id": str(event.id),
                "session_id": str(event.session_id),
                "agent_id": str(event.agent_id),
                "event_type": event.event_type.value,
                "payload": event.payload,
                "latency_ms": event.latency_ms,
                "created_at": event.created_at.isoformat()
            }
            event_data.append(event_dict)
        
        # Pagination info
        pagination = {
            "limit": limit,
            "offset": offset,
            "total": total_count,
            "has_next": offset + limit < total_count,
            "has_prev": offset > 0
        }
        
        logger.debug(
            "📊 Events queried",
            total_count=total_count,
            returned=len(event_data),
            filters={
                "session_id": session_id,
                "agent_id": agent_id,
                "event_type": event_type,
                "from_time": from_time,
                "to_time": to_time
            }
        )
        
        return EventQueryResponse(
            events=event_data,
            pagination=pagination,
            total_count=total_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Failed to query events", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to query events"
        )


@router.get("/metrics", response_class=Response)
async def get_metrics(
    session: AsyncSession = Depends(get_session_dependency)
) -> Response:
    """
    Get comprehensive observability metrics in Prometheus format.
    
    Exposes detailed metrics for monitoring agent performance, event processing,
    system health, and business metrics. Compatible with Prometheus scraping.
    """
    try:
        from app.observability.prometheus_exporter import get_metrics_exporter
        
        # Get the metrics exporter
        exporter = get_metrics_exporter()
        
        # Collect latest metrics from all sources
        await exporter.collect_all_metrics()
        
        # Query additional database metrics for context
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        # Active sessions from database
        active_sessions_query = (
            select(func.count(func.distinct(AgentEvent.session_id)))
            .where(AgentEvent.created_at >= hour_ago)
        )
        active_sessions_result = await session.execute(active_sessions_query)
        active_sessions = active_sessions_result.scalar() or 0
        exporter.active_sessions_total.set(active_sessions)
        
        # Event type distribution
        event_type_query = (
            select(AgentEvent.event_type, func.count(AgentEvent.id))
            .where(AgentEvent.created_at >= hour_ago)
            .group_by(AgentEvent.event_type)
        )
        event_type_result = await session.execute(event_type_query)
        
        for event_type, count in event_type_result.fetchall():
            exporter.events_processed_total.labels(
                event_type=event_type.value, 
                status='success'
            ).inc(count)
        
        # Tool success rate calculation
        tool_success_query = (
            select(
                func.count(AgentEvent.id).label('total'),
                func.sum(
                    func.case(
                        (func.json_extract_path_text(AgentEvent.payload, 'success') == 'true', 1),
                        else_=0
                    )
                ).label('successful')
            )
            .where(
                and_(
                    AgentEvent.event_type == EventType.POST_TOOL_USE,
                    AgentEvent.created_at >= hour_ago
                )
            )
        )
        tool_result = await session.execute(tool_success_query)
        tool_stats = tool_result.fetchone()
        
        if tool_stats and tool_stats.total > 0:
            success_rate = (tool_stats.successful or 0) / tool_stats.total
            exporter.tool_success_rate.labels(tool_name='all').set(success_rate)
        
        # Average event processing latency
        latency_query = (
            select(func.avg(AgentEvent.latency_ms))
            .where(
                and_(
                    AgentEvent.latency_ms.isnot(None),
                    AgentEvent.created_at >= hour_ago
                )
            )
        )
        latency_result = await session.execute(latency_query)
        avg_latency_ms = latency_result.scalar() or 0
        
        if avg_latency_ms > 0:
            exporter.event_processing_duration_seconds.labels(
                event_type='all'
            ).observe(avg_latency_ms / 1000.0)
        
        # WebSocket connection metrics
        from app.api.v1.websocket import connection_manager
        stats = connection_manager.get_connection_stats()
        
        exporter.set_websocket_connections('observability', stats['observability_connections'])
        for agent_id, count in stats['agent_connections'].items():
            exporter.set_websocket_connections('agent', count)
        
        # Generate and return Prometheus metrics
        response = exporter.generate_metrics_response()
        
        logger.debug("📊 Comprehensive metrics exported", 
                    active_sessions=active_sessions,
                    total_connections=stats['total_connections'])
        
        return response
        
    except Exception as e:
        logger.error("❌ Failed to generate comprehensive metrics", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate metrics"
        )


@router.get("/health", response_model=HealthResponse)
async def get_health(
    session: AsyncSession = Depends(get_session_dependency)
) -> HealthResponse:
    """
    Get observability system health status.
    
    Checks the health of all observability components including Redis,
    database, event processor, and hook interceptor.
    """
    try:
        components = {}
        overall_status = "healthy"
        
        # Check database health
        try:
            await session.execute(select(1))
            components["database"] = {"status": "healthy", "message": "Database connection OK"}
        except Exception as e:
            components["database"] = {"status": "unhealthy", "error": str(e)}
            overall_status = "unhealthy"
        
        # Check Redis health
        try:
            redis_client = get_redis()
            await redis_client.ping()
            components["redis"] = {"status": "healthy", "message": "Redis connection OK"}
        except Exception as e:
            components["redis"] = {"status": "unhealthy", "error": str(e)}
            overall_status = "unhealthy"
        
        # Check event processor health
        processor = get_event_processor()
        if processor:
            processor_health = await processor.health_check()
            components["event_processor"] = {
                "status": processor_health["status"],
                "is_running": processor_health["is_running"],
                "events_processed": processor_health["events_processed"],
                "events_failed": processor_health["events_failed"],
                "processing_rate": processor_health["processing_rate_per_second"]
            }
            
            if processor_health["status"] != "healthy":
                overall_status = "degraded" if overall_status == "healthy" else "unhealthy"
        else:
            components["event_processor"] = {"status": "unavailable", "message": "Not initialized"}
            overall_status = "degraded" if overall_status == "healthy" else "unhealthy"
        
        # Check hook interceptor health
        interceptor = get_hook_interceptor()
        if interceptor:
            components["hook_interceptor"] = {
                "status": "healthy" if interceptor.is_enabled else "disabled",
                "enabled": interceptor.is_enabled
            }
        else:
            components["hook_interceptor"] = {"status": "unavailable", "message": "Not initialized"}
        
        logger.debug("🏥 Health check completed", status=overall_status)
        
        return HealthResponse(
            status=overall_status,
            components=components,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error("❌ Health check failed", error=str(e), exc_info=True)
        return HealthResponse(
            status="unhealthy",
            components={"error": str(e)},
            timestamp=datetime.utcnow().isoformat()
        )


@router.get("/stream-info")
async def get_stream_info() -> Dict[str, Any]:
    """
    Get Redis Stream information for debugging and monitoring.
    
    Returns stream statistics, consumer group info, and processing metrics.
    """
    try:
        processor = get_event_processor()
        if not processor:
            raise HTTPException(
                status_code=503,
                detail="Event processor not available"
            )
        
        # Get stream info
        stream_info = await processor.get_stream_info()
        consumer_info = await processor.get_consumer_group_info()
        
        return {
            "stream": stream_info,
            "consumer_groups": consumer_info,
            "processor_status": {
                "is_running": processor.is_running,
                "events_processed": processor.events_processed,
                "events_failed": processor.events_failed,
                "last_processed": processor.last_processed_time.isoformat() if processor.last_processed_time else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Failed to get stream info", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to get stream information"
        )