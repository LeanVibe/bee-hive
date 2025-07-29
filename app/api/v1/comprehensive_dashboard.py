"""
Comprehensive Dashboard API Endpoints for LeanVibe Agent Hive 2.0

Provides RESTful API endpoints and WebSocket connections for comprehensive
dashboard integration with real-time monitoring capabilities.

Features:
- Multi-agent workflow progress tracking endpoints
- Quality gates visualization and results API
- Extended thinking sessions monitoring API
- Hook execution performance metrics API
- Agent performance aggregation endpoints
- Real-time WebSocket streaming endpoints
- Mobile-optimized data formatting
- Comprehensive error handling and validation
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query, Path, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from starlette.status import (
    HTTP_200_OK, HTTP_201_CREATED, HTTP_400_BAD_REQUEST, 
    HTTP_404_NOT_FOUND, HTTP_500_INTERNAL_SERVER_ERROR
)

# Import core integration components
from ...core.comprehensive_dashboard_integration import (
    comprehensive_dashboard_integration, IntegrationEventType,
    QualityGateStatus, ThinkingSessionPhase
)
from ...core.realtime_dashboard_streaming import (
    realtime_dashboard_streaming, CompressionType, StreamPriority
)

# Import existing models and schemas
from ...models.user import User
from ...schemas.base import BaseResponse
from ...core.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/comprehensive-dashboard", tags=["Comprehensive Dashboard"])


# Request/Response Models

class WorkflowTrackingRequest(BaseModel):
    """Request model for workflow tracking initialization."""
    workflow_name: str = Field(..., description="Human-readable workflow name")
    total_steps: int = Field(..., ge=1, le=1000, description="Total number of workflow steps")
    active_agents: List[str] = Field(..., description="List of agent IDs participating in workflow")
    current_phase: str = Field(default="initialization", description="Current workflow phase")
    estimated_duration_minutes: Optional[int] = Field(None, ge=1, description="Estimated duration in minutes")


class WorkflowProgressUpdate(BaseModel):
    """Request model for workflow progress updates."""
    completed_steps: Optional[int] = Field(None, ge=0, description="Number of completed steps")
    current_phase: Optional[str] = Field(None, description="Current workflow phase")
    active_agents: Optional[List[str]] = Field(None, description="Updated list of active agents")
    increment_errors: bool = Field(False, description="Whether to increment error count")
    additional_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class QualityGateRequest(BaseModel):
    """Request model for quality gate result recording."""
    gate_name: str = Field(..., description="Human-readable gate name")
    status: str = Field(..., description="Gate execution status")
    execution_time_ms: int = Field(..., ge=0, description="Execution time in milliseconds")
    success_criteria: Dict[str, Any] = Field(..., description="Success criteria for the gate")
    actual_results: Dict[str, Any] = Field(..., description="Actual results from gate execution")
    validation_errors: Optional[List[str]] = Field(None, description="Validation error messages")
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="Performance metrics")
    recommendations: Optional[List[str]] = Field(None, description="Improvement recommendations")
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = [status.value for status in QualityGateStatus]
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of: {', '.join(valid_statuses)}")
        return v


class ThinkingSessionRequest(BaseModel):
    """Request model for thinking session updates."""
    session_name: str = Field(..., description="Human-readable session name")
    phase: str = Field(..., description="Current thinking session phase")
    participating_agents: List[str] = Field(..., description="Agents participating in session")
    insights_generated: int = Field(..., ge=0, description="Number of insights generated")
    consensus_level: float = Field(..., ge=0.0, le=1.0, description="Consensus level (0-1)")
    collaboration_quality: float = Field(..., ge=0.0, le=1.0, description="Collaboration quality (0-1)")
    current_focus: str = Field(..., description="Current focus area")
    key_insights: Optional[List[str]] = Field(None, description="Key insights generated")
    disagreements: Optional[List[str]] = Field(None, description="Areas of disagreement")
    next_steps: Optional[List[str]] = Field(None, description="Planned next steps")
    
    @validator('phase')
    def validate_phase(cls, v):
        valid_phases = [phase.value for phase in ThinkingSessionPhase]
        if v not in valid_phases:
            raise ValueError(f"Phase must be one of: {', '.join(valid_phases)}")
        return v


class StreamConfigRequest(BaseModel):
    """Request model for WebSocket stream configuration."""
    event_types: Optional[List[str]] = Field(None, description="Event types to subscribe to")
    agent_ids: Optional[List[str]] = Field(None, description="Agent IDs to filter by")
    session_ids: Optional[List[str]] = Field(None, description="Session IDs to filter by")
    priority_threshold: str = Field(default="low", description="Minimum priority level")
    max_events_per_second: int = Field(default=10, ge=1, le=100, description="Rate limit")
    batch_size: int = Field(default=5, ge=1, le=50, description="Batch size for events")
    compression: str = Field(default="smart", description="Compression type")
    mobile_optimized: bool = Field(False, description="Enable mobile optimizations")
    
    @validator('priority_threshold')
    def validate_priority(cls, v):
        valid_priorities = [p.value for p in StreamPriority]
        if v not in valid_priorities:
            raise ValueError(f"Priority must be one of: {', '.join(valid_priorities)}")
        return v
    
    @validator('compression')
    def validate_compression(cls, v):
        valid_compressions = [c.value for c in CompressionType]
        if v not in valid_compressions:
            raise ValueError(f"Compression must be one of: {', '.join(valid_compressions)}")
        return v


# Response Models

class WorkflowProgressResponse(BaseModel):
    """Response model for workflow progress data."""
    workflow_id: str
    workflow_name: str
    total_steps: int
    completed_steps: int
    completion_percentage: float
    active_agents: List[str]
    current_phase: str
    start_time: str
    estimated_completion: Optional[str]
    error_count: int
    success_rate: float
    is_completed: bool
    duration_seconds: Optional[float]


class QualityGateResponse(BaseModel):
    """Response model for quality gate data."""
    gate_id: str
    gate_name: str
    status: str
    execution_time_ms: int
    success_criteria: Dict[str, Any]
    actual_results: Dict[str, Any]
    validation_errors: List[str]
    performance_metrics: Dict[str, float]
    recommendations: List[str]
    passed: bool
    timestamp: str


class SystemOverviewResponse(BaseModel):
    """Response model for system overview data."""
    active_workflows: int
    completed_workflows: int
    active_thinking_sessions: int
    active_agents: int
    average_agent_performance: float
    agent_performance_std: float
    quality_gates_summary: Dict[str, Any]
    hook_performance_summary: Dict[str, Any]
    system_health: Dict[str, Any]
    timestamp: str


# Multi-Agent Workflow Endpoints

@router.post("/workflows/track", response_model=Dict[str, Any], status_code=HTTP_201_CREATED)
async def start_workflow_tracking(
    workflow_id: str = Query(..., description="Unique workflow identifier"),
    request: WorkflowTrackingRequest = None,
    current_user: User = Depends(get_current_user)
):
    """
    Initialize tracking for a multi-agent workflow.
    
    Starts comprehensive monitoring of workflow progress, agent coordination,
    and performance metrics for the specified workflow.
    """
    try:
        logger.info(
            "Starting workflow tracking",
            workflow_id=workflow_id,
            user_id=current_user.id,
            workflow_name=request.workflow_name
        )
        
        # Start workflow tracking
        progress = await comprehensive_dashboard_integration.track_workflow_progress(
            workflow_id=workflow_id,
            workflow_name=request.workflow_name,
            total_steps=request.total_steps,
            active_agents=request.active_agents,
            current_phase=request.current_phase
        )
        
        return {
            "success": True,
            "message": "Workflow tracking started",
            "workflow_id": workflow_id,
            "progress": progress.to_dict(),
            "tracking_started_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Error starting workflow tracking",
            workflow_id=workflow_id,
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start workflow tracking: {str(e)}"
        )


@router.put("/workflows/{workflow_id}/progress", response_model=Dict[str, Any])
async def update_workflow_progress(
    workflow_id: str = Path(..., description="Workflow identifier"),
    request: WorkflowProgressUpdate = None,
    current_user: User = Depends(get_current_user)
):
    """
    Update progress for a tracked workflow.
    
    Updates workflow completion status, phase, active agents, and other
    progress indicators with real-time dashboard notifications.
    """
    try:
        logger.info(
            "Updating workflow progress",
            workflow_id=workflow_id,
            user_id=current_user.id,
            completed_steps=request.completed_steps
        )
        
        # Update workflow progress
        progress = await comprehensive_dashboard_integration.update_workflow_progress(
            workflow_id=workflow_id,
            completed_steps=request.completed_steps,
            current_phase=request.current_phase,
            active_agents=request.active_agents,
            increment_errors=request.increment_errors
        )
        
        if not progress:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Workflow {workflow_id} not found"
            )
        
        return {
            "success": True,
            "message": "Workflow progress updated",
            "workflow_id": workflow_id,
            "progress": progress.to_dict(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error updating workflow progress",
            workflow_id=workflow_id,
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update workflow progress: {str(e)}"
        )


@router.post("/workflows/{workflow_id}/complete", response_model=Dict[str, Any])
async def complete_workflow(
    workflow_id: str = Path(..., description="Workflow identifier"),
    success: bool = Query(True, description="Whether workflow completed successfully"),
    current_user: User = Depends(get_current_user)
):
    """
    Mark a workflow as completed.
    
    Finalizes workflow tracking and generates completion statistics
    and performance reports.
    """
    try:
        logger.info(
            "Completing workflow",
            workflow_id=workflow_id,
            user_id=current_user.id,
            success=success
        )
        
        # Complete workflow tracking
        await comprehensive_dashboard_integration.complete_workflow(
            workflow_id=workflow_id,
            success=success
        )
        
        return {
            "success": True,
            "message": "Workflow completed",
            "workflow_id": workflow_id,
            "completed_successfully": success,
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Error completing workflow",
            workflow_id=workflow_id,
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to complete workflow: {str(e)}"
        )


@router.get("/workflows", response_model=Dict[str, Any])
async def get_workflows(
    include_completed: bool = Query(False, description="Include completed workflows"),
    workflow_ids: Optional[str] = Query(None, description="Comma-separated workflow IDs"),
    current_user: User = Depends(get_current_user)
):
    """
    Get workflow progress data.
    
    Returns comprehensive workflow tracking data with progress indicators,
    performance metrics, and completion statistics.
    """
    try:
        logger.info(
            "Getting workflows data",
            user_id=current_user.id,
            include_completed=include_completed
        )
        
        # Parse workflow IDs filter
        workflow_id_list = None
        if workflow_ids:
            workflow_id_list = [wid.strip() for wid in workflow_ids.split(",")]
        
        # Get workflow data
        data = await comprehensive_dashboard_integration.get_workflow_progress_data(
            workflow_ids=workflow_id_list,
            include_completed=include_completed
        )
        
        return {
            "success": True,
            "data": data,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Error getting workflows data",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflows data: {str(e)}"
        )


# Quality Gates Endpoints

@router.post("/quality-gates/{gate_id}/result", response_model=Dict[str, Any], status_code=HTTP_201_CREATED)
async def record_quality_gate_result(
    gate_id: str = Path(..., description="Quality gate identifier"),
    request: QualityGateRequest = None,
    current_user: User = Depends(get_current_user)
):
    """
    Record the result of a quality gate execution.
    
    Stores quality gate results with performance metrics, validation errors,
    and improvement recommendations for dashboard visualization.
    """
    try:
        logger.info(
            "Recording quality gate result",
            gate_id=gate_id,
            user_id=current_user.id,
            status=request.status
        )
        
        # Record quality gate result
        result = await comprehensive_dashboard_integration.record_quality_gate_result(
            gate_id=gate_id,
            gate_name=request.gate_name,
            status=QualityGateStatus(request.status),
            execution_time_ms=request.execution_time_ms,
            success_criteria=request.success_criteria,
            actual_results=request.actual_results,
            validation_errors=request.validation_errors or [],
            performance_metrics=request.performance_metrics or {},
            recommendations=request.recommendations or []
        )
        
        return {
            "success": True,
            "message": "Quality gate result recorded",
            "gate_id": gate_id,
            "result": result.to_dict(),
            "recorded_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Error recording quality gate result",
            gate_id=gate_id,
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record quality gate result: {str(e)}"
        )


@router.get("/quality-gates", response_model=Dict[str, Any])
async def get_quality_gates(
    gate_ids: Optional[str] = Query(None, description="Comma-separated gate IDs"),
    time_window_hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    current_user: User = Depends(get_current_user)
):
    """
    Get quality gates data and results.
    
    Returns comprehensive quality gate execution results, trends,
    and performance statistics within the specified time window.
    """
    try:
        logger.info(
            "Getting quality gates data",
            user_id=current_user.id,
            time_window_hours=time_window_hours
        )
        
        # Parse gate IDs filter
        gate_id_list = None
        if gate_ids:
            gate_id_list = [gid.strip() for gid in gate_ids.split(",")]
        
        # Get quality gates data
        data = await comprehensive_dashboard_integration.get_quality_gates_data(
            gate_ids=gate_id_list,
            time_window_hours=time_window_hours
        )
        
        return {
            "success": True,
            "data": data,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Error getting quality gates data",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get quality gates data: {str(e)}"
        )


@router.get("/quality-gates/summary", response_model=Dict[str, Any])
async def get_quality_gates_summary(
    time_window_hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    current_user: User = Depends(get_current_user)
):
    """
    Get quality gates summary statistics.
    
    Returns aggregated quality gate performance metrics, success rates,
    and trend analysis for dashboard overview displays.
    """
    try:
        logger.info(
            "Getting quality gates summary",
            user_id=current_user.id,
            time_window_hours=time_window_hours
        )
        
        # Get summary data
        summary = await comprehensive_dashboard_integration.get_quality_gate_summary(
            time_window_hours=time_window_hours
        )
        
        return {
            "success": True,
            "summary": summary,
            "time_window_hours": time_window_hours,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Error getting quality gates summary",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get quality gates summary: {str(e)}"
        )


# Thinking Sessions Endpoints

@router.put("/thinking-sessions/{session_id}", response_model=Dict[str, Any])
async def update_thinking_session(
    session_id: str = Path(..., description="Thinking session identifier"),
    request: ThinkingSessionRequest = None,
    current_user: User = Depends(get_current_user)
):
    """
    Update status of an extended thinking session.
    
    Updates thinking session progress, collaboration metrics, insights,
    and consensus levels with real-time dashboard notifications.
    """
    try:
        logger.info(
            "Updating thinking session",
            session_id=session_id,
            user_id=current_user.id,
            phase=request.phase
        )
        
        # Update thinking session
        update = await comprehensive_dashboard_integration.update_thinking_session(
            session_id=session_id,
            session_name=request.session_name,
            phase=ThinkingSessionPhase(request.phase),
            participating_agents=request.participating_agents,
            insights_generated=request.insights_generated,
            consensus_level=request.consensus_level,
            collaboration_quality=request.collaboration_quality,
            current_focus=request.current_focus,
            key_insights=request.key_insights,
            disagreements=request.disagreements,
            next_steps=request.next_steps
        )
        
        return {
            "success": True,
            "message": "Thinking session updated",
            "session_id": session_id,
            "update": update.to_dict(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Error updating thinking session",
            session_id=session_id,
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update thinking session: {str(e)}"
        )


@router.get("/thinking-sessions", response_model=Dict[str, Any])
async def get_thinking_sessions(
    current_user: User = Depends(get_current_user)
):
    """
    Get thinking sessions data.
    
    Returns comprehensive thinking session data including collaboration
    metrics, insights generated, and consensus tracking.
    """
    try:
        logger.info(
            "Getting thinking sessions data",
            user_id=current_user.id
        )
        
        # Get thinking sessions data
        data = await comprehensive_dashboard_integration.get_thinking_sessions_data()
        
        return {
            "success": True,
            "data": data,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Error getting thinking sessions data",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get thinking sessions data: {str(e)}"
        )


# Agent Performance Endpoints

@router.get("/agents/performance", response_model=Dict[str, Any])
async def get_agent_performance(
    agent_ids: Optional[str] = Query(None, description="Comma-separated agent IDs"),
    current_user: User = Depends(get_current_user)
):
    """
    Get agent performance data.
    
    Returns comprehensive agent performance metrics including task completion
    rates, response times, collaboration effectiveness, and quality scores.
    """
    try:
        logger.info(
            "Getting agent performance data",
            user_id=current_user.id
        )
        
        # Parse agent IDs filter
        agent_id_list = None
        if agent_ids:
            agent_id_list = [aid.strip() for aid in agent_ids.split(",")]
        
        # Get agent performance data
        data = await comprehensive_dashboard_integration.get_agent_performance_data(
            agent_ids=agent_id_list
        )
        
        return {
            "success": True,
            "data": data,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Error getting agent performance data",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent performance data: {str(e)}"
        )


@router.get("/hooks/performance", response_model=Dict[str, Any])
async def get_hook_performance(
    time_window_minutes: int = Query(30, ge=1, le=1440, description="Time window in minutes"),
    current_user: User = Depends(get_current_user)
):
    """
    Get hook execution performance data.
    
    Returns comprehensive hook performance metrics including execution times,
    success rates, memory usage, and performance trends.
    """
    try:
        logger.info(
            "Getting hook performance data",
            user_id=current_user.id,
            time_window_minutes=time_window_minutes
        )
        
        # Get hook performance summary
        summary = await comprehensive_dashboard_integration.get_hook_performance_summary(
            time_window_minutes=time_window_minutes
        )
        
        return {
            "success": True,
            "summary": summary,
            "time_window_minutes": time_window_minutes,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Error getting hook performance data",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get hook performance data: {str(e)}"
        )


# System Overview Endpoints

@router.get("/overview", response_model=SystemOverviewResponse)
async def get_system_overview(
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive system performance overview.
    
    Returns high-level system health metrics, performance indicators,
    and aggregate statistics for dashboard overview displays.
    """
    try:
        logger.info(
            "Getting system overview",
            user_id=current_user.id
        )
        
        # Get system overview
        overview = await comprehensive_dashboard_integration.get_system_performance_overview()
        
        return SystemOverviewResponse(**overview)
        
    except Exception as e:
        logger.error(
            "Error getting system overview",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system overview: {str(e)}"
        )


# Real-time Streaming Endpoints

@router.websocket("/stream")
async def dashboard_stream(
    websocket: WebSocket,
    user_id: Optional[str] = Query(None, description="User ID for connection tracking")
):
    """
    WebSocket endpoint for real-time dashboard updates.
    
    Provides real-time streaming of workflow progress, quality gate results,
    thinking session updates, and performance metrics with intelligent
    batching and compression.
    """
    stream_id = None
    
    try:
        # Register stream with default configuration
        stream_id = await realtime_dashboard_streaming.register_stream(
            websocket=websocket,
            user_id=user_id
        )
        
        logger.info(
            "Dashboard stream connected",
            stream_id=stream_id,
            user_id=user_id
        )
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "configure_stream":
                    config = message.get("config", {})
                    await _handle_stream_configuration(stream_id, config)
                
                elif message.get("type") == "request_data":
                    data_type = message.get("data_type")
                    await _handle_data_request(stream_id, data_type, websocket)
                
                elif message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(
                    "Error handling WebSocket message",
                    stream_id=stream_id,
                    error=str(e)
                )
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Internal server error"
                }))
                
    except WebSocketDisconnect:
        logger.info("Dashboard stream disconnected", stream_id=stream_id)
    except Exception as e:
        logger.error(
            "Dashboard stream error",
            stream_id=stream_id,
            error=str(e)
        )
    finally:
        if stream_id:
            await realtime_dashboard_streaming.unregister_stream(stream_id)


@router.post("/stream/{stream_id}/configure", response_model=Dict[str, Any])
async def configure_stream(
    stream_id: str = Path(..., description="Stream identifier"),
    request: StreamConfigRequest = None,
    current_user: User = Depends(get_current_user)
):
    """
    Configure an existing dashboard stream.
    
    Updates stream filters, performance settings, and subscription
    preferences for real-time dashboard updates.
    """
    try:
        logger.info(
            "Configuring dashboard stream",
            stream_id=stream_id,
            user_id=current_user.id
        )
        
        # Prepare filters
        filters = {
            'event_types': request.event_types or [],
            'agent_ids': request.agent_ids or [],
            'session_ids': request.session_ids or [],
            'priority_threshold': request.priority_threshold
        }
        
        # Update stream filters
        success = await realtime_dashboard_streaming.update_stream_filters(
            stream_id=stream_id,
            filters=filters
        )
        
        if not success:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Stream {stream_id} not found"
            )
        
        return {
            "success": True,
            "message": "Stream configured successfully",
            "stream_id": stream_id,
            "filters": filters,
            "configured_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error configuring stream",
            stream_id=stream_id,
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to configure stream: {str(e)}"
        )


@router.get("/streams/statistics", response_model=Dict[str, Any])
async def get_stream_statistics(
    current_user: User = Depends(get_current_user)
):
    """
    Get streaming system statistics.
    
    Returns comprehensive statistics about active streams, performance
    metrics, and system resource utilization.
    """
    try:
        logger.info(
            "Getting stream statistics",
            user_id=current_user.id
        )
        
        # Get streaming statistics
        stats = await realtime_dashboard_streaming.get_stream_statistics()
        
        return {
            "success": True,
            "statistics": stats,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Error getting stream statistics",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stream statistics: {str(e)}"
        )


# Helper Functions

async def _handle_stream_configuration(stream_id: str, config: Dict[str, Any]) -> None:
    """Handle stream configuration updates."""
    try:
        filters = {
            'event_types': config.get('event_types', []),
            'agent_ids': config.get('agent_ids', []),
            'session_ids': config.get('session_ids', []),
            'priority_threshold': config.get('priority_threshold', 'low')
        }
        
        await realtime_dashboard_streaming.update_stream_filters(
            stream_id=stream_id,
            filters=filters
        )
        
    except Exception as e:
        logger.error(
            "Error handling stream configuration",
            stream_id=stream_id,
            error=str(e)
        )


async def _handle_data_request(
    stream_id: str,
    data_type: str,
    websocket: WebSocket
) -> None:
    """Handle specific data requests from clients."""
    try:
        if data_type == "workflows":
            data = await comprehensive_dashboard_integration.get_workflow_progress_data()
        elif data_type == "quality_gates":
            data = await comprehensive_dashboard_integration.get_quality_gates_data()
        elif data_type == "thinking_sessions":
            data = await comprehensive_dashboard_integration.get_thinking_sessions_data()
        elif data_type == "agent_performance":
            data = await comprehensive_dashboard_integration.get_agent_performance_data()
        elif data_type == "system_overview":
            data = await comprehensive_dashboard_integration.get_system_performance_overview()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Send data directly to client
        await realtime_dashboard_streaming.send_direct_message(
            stream_id=stream_id,
            event_type=f"data_response_{data_type}",
            data=data,
            priority=StreamPriority.HIGH
        )
        
    except Exception as e:
        logger.error(
            "Error handling data request",
            stream_id=stream_id,
            data_type=data_type,
            error=str(e)
        )