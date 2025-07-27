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
from app.core.hook_processor import get_hook_event_processor
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


class HookEventRequest(BaseModel):
    """Request model for Claude Code hook events."""
    
    session_id: str = Field(..., description="Session UUID")
    agent_id: str = Field(..., description="Agent UUID")
    event_type: str = Field(..., description="Hook event type")
    tool_name: Optional[str] = Field(None, description="Tool name for tool events")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Tool parameters for PreToolUse")
    result: Optional[Any] = Field(None, description="Tool result for PostToolUse")
    success: Optional[bool] = Field(None, description="Tool success status for PostToolUse")
    error: Optional[str] = Field(None, description="Error message for error events")
    error_type: Optional[str] = Field(None, description="Error type classification")
    stack_trace: Optional[str] = Field(None, description="Stack trace for error events")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for event matching")
    execution_time_ms: Optional[int] = Field(None, description="Tool execution time in milliseconds")
    timestamp: Optional[str] = Field(None, description="Event timestamp (ISO format)")
    
    @field_validator("session_id", "agent_id")
    @classmethod
    def validate_uuid(cls, v):
        """Validate UUID format."""
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError("Invalid UUID format")
        return v


class HookEventResponse(BaseModel):
    """Response model for hook event operations."""
    
    status: str = Field(..., description="Processing status")
    event_id: Optional[str] = Field(None, description="Event ID if processed")
    processing_time_ms: Optional[float] = Field(None, description="Processing time in milliseconds")
    redacted: Optional[bool] = Field(None, description="Whether data was redacted")
    performance_warnings: Optional[List[str]] = Field(None, description="Performance warnings")


class PerformanceMetricsResponse(BaseModel):
    """Response model for real-time performance metrics."""
    
    timestamp: str = Field(..., description="Metrics timestamp")
    performance: Dict[str, Any] = Field(..., description="Performance metrics")
    health: str = Field(..., description="Health status")
    degradation: Optional[Dict[str, Any]] = Field(None, description="Performance degradation info")
    stream_info: Optional[Dict[str, Any]] = Field(None, description="Redis stream information")


# === API Endpoints ===

@router.post("/hook-events", response_model=HookEventResponse, status_code=201)
async def process_hook_event(
    hook_request: HookEventRequest,
    request: Request
) -> HookEventResponse:
    """
    Process Claude Code hook events with automatic PII redaction and performance monitoring.
    
    This endpoint receives events from Claude Code hooks and processes them through
    the enhanced hook event processor with security filtering, PII redaction,
    and real-time streaming capabilities.
    """
    import time
    start_time = time.time()
    
    try:
        # Get hook event processor
        processor = get_hook_event_processor()
        if not processor:
            raise HTTPException(
                status_code=503,
                detail="Hook event processor not available"
            )
        
        # Route to appropriate processing method based on event type
        event_id = None
        performance_warnings = []
        
        if hook_request.event_type == "PRE_TOOL_USE":
            event_data = {
                "session_id": hook_request.session_id,
                "agent_id": hook_request.agent_id,
                "tool_name": hook_request.tool_name,
                "parameters": hook_request.parameters or {},
                "correlation_id": hook_request.correlation_id,
                "timestamp": hook_request.timestamp
            }
            event_id = await processor.process_pre_tool_use(event_data)
            
        elif hook_request.event_type == "POST_TOOL_USE":
            event_data = {
                "session_id": hook_request.session_id,
                "agent_id": hook_request.agent_id,
                "tool_name": hook_request.tool_name,
                "result": hook_request.result,
                "success": hook_request.success,
                "error": hook_request.error,
                "correlation_id": hook_request.correlation_id,
                "execution_time_ms": hook_request.execution_time_ms,
                "timestamp": hook_request.timestamp
            }
            event_id = await processor.process_post_tool_use(event_data)
            
        elif hook_request.event_type == "ERROR":
            event_data = {
                "session_id": hook_request.session_id,
                "agent_id": hook_request.agent_id,
                "error_type": hook_request.error_type,
                "error_message": hook_request.error,
                "stack_trace": hook_request.stack_trace,
                "context": hook_request.context or {},
                "correlation_id": hook_request.correlation_id,
                "timestamp": hook_request.timestamp
            }
            event_id = await processor.process_error_event(event_data)
            
        elif hook_request.event_type in ["AGENT_START", "AGENT_STOP"]:
            from app.models.observability import EventType
            event_type = EventType(hook_request.event_type)
            
            event_data = {
                "session_id": hook_request.session_id,
                "agent_id": hook_request.agent_id,
                "context": hook_request.context or {},
                "timestamp": hook_request.timestamp
            }
            event_id = await processor.process_agent_lifecycle_event(event_data, event_type)
            
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported event type: {hook_request.event_type}"
            )
        
        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Check for performance warnings
        if processing_time_ms > 150:  # PRD requirement
            performance_warnings.append(f"Processing time ({processing_time_ms:.1f}ms) exceeded 150ms threshold")
        
        logger.info(
            "🔗 Hook event processed",
            event_type=hook_request.event_type,
            event_id=event_id,
            processing_time_ms=processing_time_ms,
            session_id=hook_request.session_id
        )
        
        return HookEventResponse(
            status="processed" if event_id else "failed",
            event_id=event_id,
            processing_time_ms=processing_time_ms,
            redacted=True,  # PII redaction is always enabled
            performance_warnings=performance_warnings if performance_warnings else None
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("❌ Invalid hook event request", error=str(e))
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        logger.error(
            "❌ Failed to process hook event", 
            error=str(e), 
            event_type=hook_request.event_type,
            processing_time_ms=processing_time_ms,
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to process hook event"
        )


@router.get("/hook-performance", response_model=PerformanceMetricsResponse)
async def get_hook_performance() -> PerformanceMetricsResponse:
    """
    Get real-time performance metrics for hook event processing.
    
    Returns comprehensive performance data including processing times,
    throughput, error rates, and system health indicators.
    """
    try:
        processor = get_hook_event_processor()
        if not processor:
            raise HTTPException(
                status_code=503,
                detail="Hook event processor not available"
            )
        
        # Get real-time metrics
        metrics = await processor.get_real_time_metrics()
        
        logger.debug("📊 Hook performance metrics retrieved")
        
        return PerformanceMetricsResponse(**metrics)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Failed to get hook performance metrics", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve performance metrics"
        )


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
        
        # Check hook event processor health
        hook_processor = get_hook_event_processor()
        if hook_processor:
            hook_health = await hook_processor.health_check()
            components["hook_event_processor"] = hook_health
            
            if hook_health["status"] != "healthy":
                overall_status = "degraded" if overall_status == "healthy" else "unhealthy"
        else:
            components["hook_event_processor"] = {"status": "unavailable", "message": "Not initialized"}
            overall_status = "degraded" if overall_status == "healthy" else "unhealthy"
        
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