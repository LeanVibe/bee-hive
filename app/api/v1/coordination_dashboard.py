"""
Coordination Dashboard API Endpoints for LeanVibe Agent Hive 2.0

Provides RESTful API and WebSocket endpoints for real-time agent coordination
monitoring with visual graph representation and session management.

Features:
- WebSocket streaming for real-time graph updates
- REST API for graph data and session transcripts
- Advanced event filtering and session management
- Integration with enhanced lifecycle hooks system
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query, Path, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from starlette.status import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from ...core.coordination_dashboard import (
    coordination_dashboard,
    EventFilter,
    AgentCommunicationEvent,
    SessionColorManager
)
from ...core.dependencies import get_current_user
from ...models.user import User
from ...schemas.base import BaseResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/coordination", tags=["Coordination Dashboard"])


# Request/Response Models
class GraphDataRequest(BaseModel):
    """Request model for graph data."""
    session_id: str = Field(default="all", description="Session ID to filter by ('all' for all sessions)")
    event_filter: Optional[EventFilter] = Field(None, description="Advanced event filtering")
    include_inactive: bool = Field(False, description="Include inactive nodes")


class TranscriptRequest(BaseModel):
    """Request model for session transcripts."""
    session_id: str = Field(..., description="Session ID to get transcript for")
    agent_filter: Optional[List[str]] = Field(None, description="Filter by specific agent IDs")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum number of events")
    start_time: Optional[str] = Field(None, description="Start time filter (ISO string)")
    end_time: Optional[str] = Field(None, description="End time filter (ISO string)")


class GraphDataResponse(BaseModel):
    """Response model for graph data."""
    success: bool = Field(..., description="Whether request was successful")
    nodes: List[Dict[str, Any]] = Field(..., description="Graph nodes")
    edges: List[Dict[str, Any]] = Field(..., description="Graph edges")
    stats: Dict[str, Any] = Field(..., description="Graph statistics")
    session_colors: Dict[str, str] = Field(..., description="Session color mapping")
    timestamp: str = Field(..., description="Response timestamp")


class TranscriptResponse(BaseModel):
    """Response model for session transcripts."""
    success: bool = Field(..., description="Whether request was successful")
    session_id: str = Field(..., description="Session ID")
    events: List[Dict[str, Any]] = Field(..., description="Communication events")
    total_events: int = Field(..., description="Total number of events in session")
    agent_summary: Dict[str, Any] = Field(..., description="Agent activity summary")
    timestamp: str = Field(..., description="Response timestamp")


class SessionManagementResponse(BaseModel):
    """Response model for session management operations."""
    success: bool = Field(..., description="Whether operation was successful")
    active_sessions: List[str] = Field(..., description="List of active session IDs")
    session_stats: Dict[str, Dict[str, Any]] = Field(..., description="Statistics per session")
    total_connections: int = Field(..., description="Total WebSocket connections")


# WebSocket Endpoints
@router.websocket("/ws/{session_id}")
async def coordination_websocket(
    websocket: WebSocket,
    session_id: str = Path(..., description="Session ID to monitor ('all' for all sessions)"),
    event_types: Optional[str] = Query(None, description="Comma-separated event types to filter"),
    agent_types: Optional[str] = Query(None, description="Comma-separated agent types to filter"),
    include_system: bool = Query(True, description="Include system events")
):
    """
    WebSocket endpoint for real-time coordination dashboard updates.
    
    Streams real-time graph updates, agent state changes, and coordination events
    filtered by session, event types, and other criteria.
    """
    connection_id = None
    
    try:
        # Parse query parameters into event filter
        event_filter = EventFilter()
        
        if session_id != "all":
            event_filter.session_ids = [session_id]
        
        if event_types:
            event_filter.event_types = [t.strip() for t in event_types.split(",")]
        
        if agent_types:
            event_filter.agent_types = [t.strip() for t in agent_types.split(",")]
        
        event_filter.include_system_events = include_system
        
        # Register WebSocket connection
        connection_id = await coordination_dashboard.register_websocket(
            websocket, session_id, event_filter
        )
        
        logger.info(
            "Coordination WebSocket connected",
            connection_id=connection_id,
            session_id=session_id,
            filters=event_filter.dict()
        )
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for client messages (for bidirectional communication)
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle client requests
                if message.get("type") == "request_update":
                    await _handle_client_update_request(websocket, message, session_id)
                elif message.get("type") == "filter_update":
                    await _handle_filter_update(connection_id, message)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                # Send error message for invalid JSON
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(
                    "Error handling WebSocket message",
                    connection_id=connection_id,
                    error=str(e)
                )
                await websocket.send_text(json.dumps({
                    "type": "error", 
                    "message": "Internal server error"
                }))
                
    except WebSocketDisconnect:
        logger.info("Coordination WebSocket disconnected", connection_id=connection_id)
    except Exception as e:
        logger.error(
            "Coordination WebSocket error",
            connection_id=connection_id,
            error=str(e)
        )
    finally:
        if connection_id:
            await coordination_dashboard.unregister_websocket(connection_id)


async def _handle_client_update_request(
    websocket: WebSocket, 
    message: Dict[str, Any], 
    session_id: str
) -> None:
    """Handle client requests for specific updates."""
    try:
        request_type = message.get("request")
        
        if request_type == "graph_data":
            graph_data = await coordination_dashboard.get_graph_data(session_id)
            await websocket.send_text(json.dumps({
                "type": "graph_data_response",
                "data": graph_data
            }))
        elif request_type == "session_stats":
            # Get session statistics
            stats = await _get_session_statistics(session_id)
            await websocket.send_text(json.dumps({
                "type": "session_stats_response",
                "data": stats
            }))
            
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Failed to handle request: {str(e)}"
        }))


async def _handle_filter_update(connection_id: str, message: Dict[str, Any]) -> None:
    """Handle filter updates from client."""
    try:
        filter_data = message.get("filter", {})
        new_filter = EventFilter(**filter_data)
        
        # Update filter for this connection
        coordination_dashboard.event_filters[connection_id] = new_filter
        
        logger.info(
            "WebSocket filter updated",
            connection_id=connection_id,
            new_filter=new_filter.dict()
        )
        
    except ValidationError as e:
        logger.error(
            "Invalid filter update",
            connection_id=connection_id,
            error=str(e)
        )


# REST API Endpoints
@router.get("/graph-data", response_model=GraphDataResponse, status_code=HTTP_200_OK)
async def get_graph_data(
    session_id: str = Query(default="all", description="Session ID to filter by"),
    event_types: Optional[str] = Query(None, description="Comma-separated event types"),
    agent_types: Optional[str] = Query(None, description="Comma-separated agent types"),
    include_inactive: bool = Query(False, description="Include inactive nodes"),
    current_user: User = Depends(get_current_user)
):
    """
    Get current graph data for coordination dashboard.
    
    Returns nodes, edges, and statistics for the agent coordination graph
    with optional filtering by session, event types, and agent types.
    """
    try:
        logger.info(
            "Graph data requested",
            user_id=current_user.id,
            session_id=session_id
        )
        
        # Build event filter from query parameters
        event_filter = EventFilter()
        
        if session_id != "all":
            event_filter.session_ids = [session_id]
        
        if event_types:
            event_filter.event_types = [t.strip() for t in event_types.split(",")]
        
        if agent_types:
            event_filter.agent_types = [t.strip() for t in agent_types.split(",")]
        
        # Get graph data
        graph_data = await coordination_dashboard.get_graph_data(session_id, event_filter)
        
        return GraphDataResponse(
            success=True,
            nodes=graph_data["nodes"],
            edges=graph_data["edges"],
            stats=graph_data["stats"],
            session_colors=graph_data["session_colors"],
            timestamp=graph_data["timestamp"]
        )
        
    except Exception as e:
        logger.error(
            "Error getting graph data",
            user_id=current_user.id,
            session_id=session_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to get graph data: {str(e)}"
        )


@router.get("/transcript/{session_id}", response_model=TranscriptResponse, status_code=HTTP_200_OK)
async def get_session_transcript(
    session_id: str = Path(..., description="Session ID to get transcript for"),
    agent_filter: Optional[str] = Query(None, description="Comma-separated agent IDs"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum events"),
    start_time: Optional[str] = Query(None, description="Start time filter (ISO)"),
    end_time: Optional[str] = Query(None, description="End time filter (ISO)"),
    current_user: User = Depends(get_current_user)
):
    """
    Get chronological communication transcript for a session.
    
    Returns agent-to-agent communications, tool calls, and context sharing
    events in chronological order with optional filtering.
    """
    try:
        logger.info(
            "Session transcript requested",
            user_id=current_user.id,
            session_id=session_id,
            limit=limit
        )
        
        # Parse agent filter
        agent_list = None
        if agent_filter:
            agent_list = [a.strip() for a in agent_filter.split(",")]
        
        # Get transcript events
        events = await coordination_dashboard.get_session_transcript(
            session_id=session_id,
            agent_filter=agent_list,
            limit=limit
        )
        
        # Calculate agent activity summary
        agent_summary = await _calculate_agent_summary(events)
        
        # Format events for response
        formatted_events = [
            {
                "id": event.id,
                "timestamp": event.timestamp.isoformat(),
                "source_agent": event.source_agent_id,
                "target_agent": event.target_agent_id,
                "message_type": event.message_type,
                "content": event.content,
                "context_shared": event.context_shared,
                "tool_calls": event.tool_calls
            }
            for event in events
        ]
        
        return TranscriptResponse(
            success=True,
            session_id=session_id,
            events=formatted_events,
            total_events=len(formatted_events),
            agent_summary=agent_summary,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(
            "Error getting session transcript",
            user_id=current_user.id,
            session_id=session_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to get session transcript: {str(e)}"
        )


@router.get("/sessions", response_model=SessionManagementResponse, status_code=HTTP_200_OK)
async def get_session_management(
    current_user: User = Depends(get_current_user)
):
    """
    Get session management information.
    
    Returns active sessions, statistics, and WebSocket connection counts
    for coordination dashboard monitoring.
    """
    try:
        logger.info("Session management data requested", user_id=current_user.id)
        
        # Get active sessions from graph nodes
        active_sessions = set()
        session_stats = {}
        
        for node in coordination_dashboard.nodes.values():
            session_id = node.metadata.get("session_id", "default")
            active_sessions.add(session_id)
            
            if session_id not in session_stats:
                session_stats[session_id] = {
                    "agents": 0,
                    "tools": 0,
                    "contexts": 0,
                    "active_agents": 0,
                    "last_activity": None
                }
            
            stats = session_stats[session_id]
            
            if node.type.value == "agent":
                stats["agents"] += 1
                if node.status.value == "active":
                    stats["active_agents"] += 1
            elif node.type.value == "tool":
                stats["tools"] += 1
            elif node.type.value == "context":
                stats["contexts"] += 1
            
            # Update last activity
            if (not stats["last_activity"] or 
                node.last_updated.isoformat() > stats["last_activity"]):
                stats["last_activity"] = node.last_updated.isoformat()
        
        return SessionManagementResponse(
            success=True,
            active_sessions=list(active_sessions),
            session_stats=session_stats,
            total_connections=len(coordination_dashboard.active_websockets)
        )
        
    except Exception as e:
        logger.error(
            "Error getting session management data",
            user_id=current_user.id,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to get session data: {str(e)}"
        )


@router.post("/test-event", status_code=HTTP_200_OK)
async def trigger_test_event(
    session_id: str = Query(..., description="Session ID for test event"),
    event_type: str = Query(default="agent_activation", description="Type of test event"),
    current_user: User = Depends(get_current_user)
):
    """
    Trigger a test event for dashboard development and testing.
    
    Creates synthetic coordination events for testing the dashboard
    visualization and real-time update functionality.
    """
    try:
        logger.info(
            "Test event triggered",
            user_id=current_user.id,
            session_id=session_id,
            event_type=event_type
        )
        
        # Create synthetic lifecycle event
        from ...core.enhanced_lifecycle_hooks import LifecycleEventData, EnhancedEventType
        from uuid import uuid4
        
        # Map event type strings to EnhancedEventType enum
        event_type_map = {
            "agent_activation": EnhancedEventType.AGENT_LIFECYCLE_START,
            "agent_sleep": EnhancedEventType.AGENT_LIFECYCLE_PAUSE,
            "tool_use": EnhancedEventType.PRE_TOOL_USE,
            "context_creation": EnhancedEventType.TASK_ASSIGNMENT,
            "error": EnhancedEventType.ERROR_PATTERN_DETECTED
        }
        
        if event_type not in event_type_map:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"Unknown event type: {event_type}"
            )
        
        test_event = LifecycleEventData(
            session_id=session_id,
            agent_id=f"test_agent_{uuid4().hex[:8]}",
            event_type=event_type_map[event_type],
            timestamp=datetime.utcnow().isoformat(),
            payload={
                "agent_type": "test",
                "tool_id": "test_tool" if event_type == "tool_use" else None,
                "context_id": f"test_context_{uuid4().hex[:8]}" if event_type == "context_creation" else None,
                "error": "Test error message" if event_type == "error" else None,
                "test_event": True
            }
        )
        
        # Process the test event
        await coordination_dashboard.process_lifecycle_event(test_event)
        
        return {
            "success": True,
            "message": f"Test {event_type} event created",
            "agent_id": test_event.agent_id,
            "session_id": session_id,
            "timestamp": test_event.timestamp
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Error creating test event",
            user_id=current_user.id,
            session_id=session_id,
            event_type=event_type,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to create test event: {str(e)}"
        )


@router.delete("/cleanup", status_code=HTTP_200_OK)
async def cleanup_dashboard_data(
    max_age_hours: int = Query(default=24, ge=1, le=168, description="Maximum age in hours"),
    current_user: User = Depends(get_current_user)
):
    """
    Clean up old dashboard data.
    
    Removes old nodes, edges, and communication events based on age
    to maintain optimal performance and memory usage.
    """
    try:
        logger.info(
            "Dashboard cleanup requested",
            user_id=current_user.id,
            max_age_hours=max_age_hours
        )
        
        # Get counts before cleanup
        nodes_before = len(coordination_dashboard.nodes)
        edges_before = len(coordination_dashboard.edges)
        events_before = len(coordination_dashboard.communication_history)
        
        # Perform cleanup
        await coordination_dashboard.cleanup_old_data(max_age_hours)
        
        # Get counts after cleanup
        nodes_after = len(coordination_dashboard.nodes)
        edges_after = len(coordination_dashboard.edges)
        events_after = len(coordination_dashboard.communication_history)
        
        return {
            "success": True,
            "message": "Dashboard data cleanup completed",
            "cleanup_stats": {
                "max_age_hours": max_age_hours,
                "nodes_removed": nodes_before - nodes_after,
                "edges_removed": edges_before - edges_after,
                "events_removed": events_before - events_after,
                "nodes_remaining": nodes_after,
                "edges_remaining": edges_after,
                "events_remaining": events_after
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(
            "Error during dashboard cleanup",
            user_id=current_user.id,
            max_age_hours=max_age_hours,
            error=str(e)
        )
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Cleanup failed: {str(e)}"
        )


# Helper Functions
async def _get_session_statistics(session_id: str) -> Dict[str, Any]:
    """Get detailed statistics for a session."""
    stats = {
        "session_id": session_id,
        "agents": {"total": 0, "active": 0, "sleeping": 0, "error": 0},
        "tools": {"total": 0, "usage_count": 0},
        "contexts": {"total": 0, "shared_count": 0},
        "communication": {"total_events": 0, "last_activity": None},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Analyze nodes
    for node in coordination_dashboard.nodes.values():
        node_session = node.metadata.get("session_id", "default")
        if session_id != "all" and node_session != session_id:
            continue
        
        if node.type.value == "agent":
            stats["agents"]["total"] += 1
            if node.status.value == "active":
                stats["agents"]["active"] += 1
            elif node.status.value == "sleeping":
                stats["agents"]["sleeping"] += 1
            elif node.status.value == "error":
                stats["agents"]["error"] += 1
        elif node.type.value == "tool":
            stats["tools"]["total"] += 1
            stats["tools"]["usage_count"] += node.metadata.get("usage_count", 0)
        elif node.type.value == "context":
            stats["contexts"]["total"] += 1
            stats["contexts"]["shared_count"] += len(node.metadata.get("sharing_agents", []))
    
    # Analyze communication events
    for event in coordination_dashboard.communication_history:
        if session_id != "all" and event.session_id != session_id:
            continue
        
        stats["communication"]["total_events"] += 1
        if (not stats["communication"]["last_activity"] or 
            event.timestamp.isoformat() > stats["communication"]["last_activity"]):
            stats["communication"]["last_activity"] = event.timestamp.isoformat()
    
    return stats


async def _calculate_agent_summary(events: List[AgentCommunicationEvent]) -> Dict[str, Any]:
    """Calculate agent activity summary from communication events."""
    agent_activity = {}
    
    for event in events:
        source_id = event.source_agent_id
        
        if source_id not in agent_activity:
            agent_activity[source_id] = {
                "total_messages": 0,
                "tool_calls": 0,
                "contexts_shared": 0,
                "first_activity": event.timestamp.isoformat(),
                "last_activity": event.timestamp.isoformat()
            }
        
        activity = agent_activity[source_id]
        activity["total_messages"] += 1
        activity["tool_calls"] += len(event.tool_calls)
        
        if event.context_shared:
            activity["contexts_shared"] += 1
        
        # Update activity timestamps
        if event.timestamp.isoformat() < activity["first_activity"]:
            activity["first_activity"] = event.timestamp.isoformat()
        if event.timestamp.isoformat() > activity["last_activity"]:
            activity["last_activity"] = event.timestamp.isoformat()
    
    return {
        "total_agents": len(agent_activity),
        "agent_details": agent_activity,
        "most_active_agent": max(
            agent_activity.items(),
            key=lambda x: x[1]["total_messages"],
            default=(None, {"total_messages": 0})
        )[0] if agent_activity else None
    }