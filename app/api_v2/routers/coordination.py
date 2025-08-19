"""
Coordination API - Consolidated multi-agent coordination endpoints

Consolidates coordination_endpoints.py, v1/coordination_monitoring.py,
v1/multi_agent_coordination.py, and v1/global_coordination.py
into a unified RESTful resource for multi-agent coordination.

Performance target: <100ms P95 response time
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

import structlog
from fastapi import APIRouter, Request, HTTPException, Query, BackgroundTasks
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_
from sqlalchemy.orm import selectinload

from ...core.database import get_session_dependency
from ...core.coordination import coordination_engine
from ...models.coordination import EnhancedCoordinationEvent, CoordinationEventType, CoordinationPattern
from ...models.agent import Agent, AgentStatus
from ...schemas.team_coordination import (
    CoordinationMode,
    CoordinationSessionRequest,
    CoordinationSessionResponse
)
from ..middleware import (
    get_current_user_from_request
)

logger = structlog.get_logger()
router = APIRouter()

# Coordination engine dependency
async def get_coordination_engine() -> coordination_engine:
    """Get coordination engine instance."""
    return coordination_engine()

@router.post("/sessions", response_model=CoordinationSessionResponse, status_code=201)
async def create_coordination_session(
    request: Request,
    session_data: CoordinationSessionCreateRequest,
    db: AsyncSession = Depends(get_session_dependency),
    coordination_engine: coordination_engine = Depends(get_coordination_engine)
) -> CoordinationSessionResponse:
    """
    Create a new multi-agent coordination session.
    
    Performance target: <100ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Validate participating agents
        if session_data.participating_agents:
            agent_query = select(Agent).where(
                Agent.id.in_(session_data.participating_agents)
            )
            agent_result = await db.execute(agent_query)
            existing_agents = agent_result.scalars().all()
            
            if len(existing_agents) != len(session_data.participating_agents):
                raise HTTPException(
                    status_code=400,
                    detail="One or more participating agents not found"
                )
            
            # Check if all agents are active
            inactive_agents = [a for a in existing_agents if a.status != AgentStatus.ACTIVE]
            if inactive_agents:
                raise HTTPException(
                    status_code=400,
                    detail=f"Agents must be active: {[a.id for a in inactive_agents]}"
                )
        
        # Create coordination session
        session = CoordinationSession(
            id=str(uuid.uuid4()),
            name=session_data.name,
            description=session_data.description,
            type=session_data.type,
            status=SessionStatus.CREATED,
            participating_agents=session_data.participating_agents or [],
            coordination_rules=session_data.coordination_rules or {},
            configuration=session_data.configuration or {},
            metadata={
                "created_by": current_user.id,
                "created_at": datetime.utcnow().isoformat(),
                "version": "2.0"
            }
        )
        
        db.add(session)
        await db.commit()
        await db.refresh(session)
        
        # Initialize session in coordination engine
        await coordination_engine.create_session(
            session_id=session.id,
            session_type=session.type,
            participating_agents=session.participating_agents,
            coordination_rules=session.coordination_rules
        )
        
        logger.info(
            "coordination_session_created",
            session_id=session.id,
            session_name=session.name,
            session_type=session.type.value,
            participating_agents=session.participating_agents,
            created_by=current_user.id
        )
        
        return CoordinationSessionResponse.from_orm(session)
        
    except Exception as e:
        await db.rollback()
        logger.error("coordination_session_creation_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create coordination session: {str(e)}"
        )

@router.get("/sessions", response_model=CoordinationSessionListResponse)
async def list_coordination_sessions(
    request: Request,
    skip: int = Query(0, ge=0, description="Number of sessions to skip"),
    limit: int = Query(50, ge=1, le=1000, description="Number of sessions to return"),
    status: Optional[SessionStatus] = Query(None, description="Filter by session status"),
    type: Optional[CoordinationType] = Query(None, description="Filter by coordination type"),
    agent_id: Optional[str] = Query(None, description="Filter by participating agent"),
    db: AsyncSession = Depends(get_session_dependency)
) -> CoordinationSessionListResponse:
    """
    List coordination sessions with optional filtering.
    
    Performance target: <100ms
    """
    try:
        # Build query with filters
        query = select(CoordinationSession)
        
        filters = []
        if status:
            filters.append(CoordinationSession.status == status)
        if type:
            filters.append(CoordinationSession.type == type)
        if agent_id:
            # Filter sessions that include the specified agent
            filters.append(CoordinationSession.participating_agents.contains([agent_id]))
            
        if filters:
            query = query.where(and_(*filters))
            
        # Apply pagination
        query = query.offset(skip).limit(limit)
        
        # Execute query
        result = await db.execute(query)
        sessions = result.scalars().all()
        
        # Get total count for pagination
        count_query = select(CoordinationSession)
        if filters:
            count_query = count_query.where(and_(*filters))
            
        total_result = await db.execute(count_query)
        total = len(total_result.scalars().all())
        
        return CoordinationSessionListResponse(
            sessions=[CoordinationSessionResponse.from_orm(session) for session in sessions],
            total=total,
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        logger.error("coordination_session_list_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list coordination sessions: {str(e)}"
        )

@router.get("/sessions/{session_id}", response_model=CoordinationSessionResponse)
async def get_coordination_session(
    session_id: str,
    db: AsyncSession = Depends(get_session_dependency)
) -> CoordinationSessionResponse:
    """
    Get details of a specific coordination session.
    
    Performance target: <100ms
    """
    try:
        # Query session
        query = select(CoordinationSession).where(CoordinationSession.id == session_id)
        result = await db.execute(query)
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Coordination session {session_id} not found"
            )
            
        return CoordinationSessionResponse.from_orm(session)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("coordination_session_get_failed", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get coordination session: {str(e)}"
        )

@router.post("/sessions/{session_id}/start")
async def start_coordination_session(
    request: Request,
    session_id: str,
    db: AsyncSession = Depends(get_session_dependency),
    coordination_engine: coordination_engine = Depends(get_coordination_engine)
):
    """
    Start a coordination session.
    
    Performance target: <100ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Get session
        query = select(CoordinationSession).where(CoordinationSession.id == session_id)
        result = await db.execute(query)
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Coordination session {session_id} not found"
            )
        
        if session.status != SessionStatus.CREATED:
            raise HTTPException(
                status_code=400,
                detail=f"Session must be in CREATED status to start (current: {session.status.value})"
            )
        
        # Start session in coordination engine
        await coordination_engine.start_session(session_id)
        
        # Update session status
        await db.execute(
            update(CoordinationSession)
            .where(CoordinationSession.id == session_id)
            .values(
                status=SessionStatus.ACTIVE,
                started_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                updated_by=current_user.id
            )
        )
        await db.commit()
        
        logger.info(
            "coordination_session_started",
            session_id=session_id,
            started_by=current_user.id
        )
        
        return {
            "session_id": session_id,
            "status": "started",
            "message": "Coordination session started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("coordination_session_start_failed", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start coordination session: {str(e)}"
        )

@router.post("/sessions/{session_id}/stop")
async def stop_coordination_session(
    request: Request,
    session_id: str,
    db: AsyncSession = Depends(get_session_dependency),
    coordination_engine: coordination_engine = Depends(get_coordination_engine)
):
    """
    Stop a coordination session.
    
    Performance target: <100ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Get session
        query = select(CoordinationSession).where(CoordinationSession.id == session_id)
        result = await db.execute(query)
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Coordination session {session_id} not found"
            )
        
        if session.status != SessionStatus.ACTIVE:
            raise HTTPException(
                status_code=400,
                detail=f"Session must be active to stop (current: {session.status.value})"
            )
        
        # Stop session in coordination engine
        await coordination_engine.stop_session(session_id)
        
        # Update session status
        await db.execute(
            update(CoordinationSession)
            .where(CoordinationSession.id == session_id)
            .values(
                status=SessionStatus.COMPLETED,
                ended_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                updated_by=current_user.id
            )
        )
        await db.commit()
        
        logger.info(
            "coordination_session_stopped",
            session_id=session_id,
            stopped_by=current_user.id
        )
        
        return {
            "session_id": session_id,
            "status": "stopped",
            "message": "Coordination session stopped"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("coordination_session_stop_failed", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop coordination session: {str(e)}"
        )

@router.post("/sessions/{session_id}/events")
async def send_coordination_event(
    request: Request,
    session_id: str,
    event_data: CoordinationEventRequest,
    coordination_engine: coordination_engine = Depends(get_coordination_engine),
    db: AsyncSession = Depends(get_session_dependency)
):
    """
    Send an event to a coordination session.
    
    Performance target: <100ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Verify session exists and is active
        query = select(CoordinationSession).where(CoordinationSession.id == session_id)
        result = await db.execute(query)
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Coordination session {session_id} not found"
            )
        
        if session.status != SessionStatus.ACTIVE:
            raise HTTPException(
                status_code=400,
                detail=f"Session must be active to send events (current: {session.status.value})"
            )
        
        # Send event through coordination engine
        event_id = await coordination_engine.send_event(
            session_id=session_id,
            event_type=event_data.event_type,
            payload=event_data.payload,
            source_agent_id=event_data.source_agent_id,
            target_agent_ids=event_data.target_agent_ids
        )
        
        logger.info(
            "coordination_event_sent",
            session_id=session_id,
            event_id=event_id,
            event_type=event_data.event_type,
            source_agent=event_data.source_agent_id,
            sent_by=current_user.id
        )
        
        return {
            "event_id": event_id,
            "session_id": session_id,
            "status": "sent",
            "message": "Coordination event sent successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("coordination_event_send_failed", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send coordination event: {str(e)}"
        )

@router.get("/sessions/{session_id}/events")
async def list_coordination_events(
    session_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    source_agent_id: Optional[str] = Query(None, description="Filter by source agent"),
    coordination_engine: coordination_engine = Depends(get_coordination_engine),
    db: AsyncSession = Depends(get_session_dependency)
):
    """
    List events from a coordination session.
    
    Performance target: <100ms
    """
    try:
        # Verify session exists
        query = select(CoordinationSession).where(CoordinationSession.id == session_id)
        result = await db.execute(query)
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Coordination session {session_id} not found"
            )
        
        # Get events from coordination engine
        events = await coordination_engine.get_session_events(
            session_id=session_id,
            skip=skip,
            limit=limit,
            event_type=event_type,
            source_agent_id=source_agent_id
        )
        
        return {
            "events": events,
            "session_id": session_id,
            "skip": skip,
            "limit": limit
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("coordination_events_list_failed", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list coordination events: {str(e)}"
        )

@router.get("/sessions/{session_id}/agents")
async def list_session_agents(
    session_id: str,
    db: AsyncSession = Depends(get_session_dependency)
):
    """
    List agents participating in a coordination session.
    
    Performance target: <100ms
    """
    try:
        # Get session
        query = select(CoordinationSession).where(CoordinationSession.id == session_id)
        result = await db.execute(query)
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Coordination session {session_id} not found"
            )
        
        # Get agent details
        if session.participating_agents:
            agent_query = select(Agent).where(
                Agent.id.in_(session.participating_agents)
            )
            agent_result = await db.execute(agent_query)
            agents = agent_result.scalars().all()
            
            return {
                "agents": [
                    {
                        "id": agent.id,
                        "name": agent.name,
                        "type": agent.type.value,
                        "status": agent.status.value,
                        "capabilities": agent.capabilities
                    }
                    for agent in agents
                ],
                "session_id": session_id
            }
        else:
            return {
                "agents": [],
                "session_id": session_id
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("coordination_session_agents_failed", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list session agents: {str(e)}"
        )

@router.get("/stats/overview", response_model=CoordinationStatsResponse)
async def get_coordination_stats(
    db: AsyncSession = Depends(get_session_dependency)
) -> CoordinationStatsResponse:
    """
    Get system-wide coordination statistics.
    
    Performance target: <100ms
    """
    try:
        # Get all coordination sessions
        query = select(CoordinationSession)
        result = await db.execute(query)
        sessions = result.scalars().all()
        
        # Calculate statistics
        total_sessions = len(sessions)
        
        status_counts = {}
        for status in SessionStatus:
            status_counts[status.value] = len([s for s in sessions if s.status == status])
        
        type_counts = {}
        for coord_type in CoordinationType:
            type_counts[coord_type.value] = len([s for s in sessions if s.type == coord_type])
        
        # Calculate active sessions
        active_sessions = status_counts.get(SessionStatus.ACTIVE.value, 0)
        
        # Calculate average session duration (placeholder)
        average_duration_hours = 0  # Would calculate from session start/end times
        
        return CoordinationStatsResponse(
            total_sessions=total_sessions,
            active_sessions=active_sessions,
            status_breakdown=status_counts,
            type_breakdown=type_counts,
            average_session_duration_hours=average_duration_hours
        )
        
    except Exception as e:
        logger.error("coordination_stats_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get coordination stats: {str(e)}"
        )