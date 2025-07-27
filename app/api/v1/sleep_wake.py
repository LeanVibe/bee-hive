"""
Sleep-Wake Manager API endpoints.

Provides REST API for sleep-wake management with:
- Agent sleep/wake control with validation
- Sleep window CRUD operations
- Checkpoint management and inspection
- Recovery operations and status monitoring
- Health monitoring and status endpoints
- Performance metrics and analytics
"""

import logging
from datetime import datetime, time, date
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, desc

from ...core.database import get_async_session
from ...core.sleep_scheduler import get_sleep_scheduler
from ...core.checkpoint_manager import get_checkpoint_manager
from ...core.recovery_manager import get_recovery_manager
from ...core.consolidation_engine import get_consolidation_engine
from ...models.sleep_wake import (
    SleepWindow, Checkpoint, SleepWakeCycle, ConsolidationJob,
    SleepWakeAnalytics, SleepState, CheckpointType, ConsolidationStatus
)
from ...models.agent import Agent


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sleep-wake", tags=["sleep-wake"])


# Pydantic models for request/response validation
class SleepWindowCreate(BaseModel):
    agent_id: Optional[UUID] = Field(None, description="Agent ID for agent-specific window, null for system-wide")
    start_time: time = Field(..., description="Sleep start time")
    end_time: time = Field(..., description="Sleep end time")
    timezone: str = Field("UTC", description="Timezone for sleep window")
    active: bool = Field(True, description="Whether the sleep window is active")
    days_of_week: List[int] = Field([1, 2, 3, 4, 5, 6, 7], description="Days of week (1=Monday, 7=Sunday)")
    priority: int = Field(0, description="Priority level (higher values override lower)")
    
    @validator('days_of_week')
    def validate_days_of_week(cls, v):
        if not all(1 <= day <= 7 for day in v):
            raise ValueError('Days of week must be integers between 1 and 7')
        return v
    
    @validator('timezone')
    def validate_timezone(cls, v):
        try:
            import pytz
            pytz.timezone(v)
        except pytz.exceptions.UnknownTimeZoneError:
            raise ValueError(f'Unknown timezone: {v}')
        return v


class SleepWindowUpdate(BaseModel):
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    timezone: Optional[str] = None
    active: Optional[bool] = None
    days_of_week: Optional[List[int]] = None
    priority: Optional[int] = None
    
    @validator('days_of_week')
    def validate_days_of_week(cls, v):
        if v is not None and not all(1 <= day <= 7 for day in v):
            raise ValueError('Days of week must be integers between 1 and 7')
        return v


class SleepRequest(BaseModel):
    duration_minutes: Optional[int] = Field(None, description="Sleep duration in minutes, defaults to next wake window")
    cycle_type: str = Field("manual", description="Type of sleep cycle")


class RecoveryRequest(BaseModel):
    target_checkpoint_id: Optional[UUID] = Field(None, description="Specific checkpoint to restore")
    recovery_type: str = Field("manual", description="Type of recovery operation")


class SleepWindowResponse(BaseModel):
    id: int
    agent_id: Optional[UUID]
    start_time: str
    end_time: str
    timezone: str
    active: bool
    days_of_week: List[int]
    priority: int
    created_at: str
    updated_at: str


class CheckpointResponse(BaseModel):
    id: UUID
    agent_id: Optional[UUID]
    checkpoint_type: str
    path: str
    sha256: str
    size_bytes: int
    size_mb: float
    is_valid: bool
    validation_errors: List[str]
    metadata: Dict[str, Any]
    compression_ratio: Optional[float]
    creation_time_ms: Optional[float]
    validation_time_ms: Optional[float]
    created_at: str
    expires_at: Optional[str]


class SleepCycleResponse(BaseModel):
    id: UUID
    agent_id: UUID
    cycle_type: str
    sleep_state: str
    sleep_time: Optional[str]
    wake_time: Optional[str]
    expected_wake_time: Optional[str]
    consolidation_summary: Optional[str]
    token_reduction_achieved: Optional[float]
    consolidation_time_ms: Optional[float]
    recovery_time_ms: Optional[float]
    created_at: str
    updated_at: str


class HealthStatusResponse(BaseModel):
    healthy: bool
    timestamp: str
    agent_id: Optional[UUID]
    checks: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


# Sleep/Wake Control Endpoints
@router.post("/agents/{agent_id}/sleep")
async def force_agent_sleep(
    agent_id: UUID = Path(..., description="Agent ID to put to sleep"),
    sleep_request: SleepRequest = Body(...),
    session: AsyncSession = Depends(get_async_session)
) -> JSONResponse:
    """Force an agent into immediate sleep state."""
    try:
        # Validate agent exists
        agent = await session.get(Agent, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Check current state
        if agent.current_sleep_state != SleepState.AWAKE:
            raise HTTPException(
                status_code=400,
                detail=f"Agent is already in state {agent.current_sleep_state.value}"
            )
        
        # Initiate sleep
        scheduler = await get_sleep_scheduler()
        success = await scheduler.force_sleep(agent_id, sleep_request.duration_minutes)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to initiate sleep")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Agent {agent_id} sleep initiated",
                "agent_id": str(agent_id),
                "cycle_type": sleep_request.cycle_type,
                "duration_minutes": sleep_request.duration_minutes
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forcing agent sleep: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{agent_id}/wake")
async def force_agent_wake(
    agent_id: UUID = Path(..., description="Agent ID to wake up"),
    session: AsyncSession = Depends(get_async_session)
) -> JSONResponse:
    """Force an agent to wake up immediately."""
    try:
        # Validate agent exists
        agent = await session.get(Agent, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Check current state
        if agent.current_sleep_state == SleepState.AWAKE:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": f"Agent {agent_id} is already awake",
                    "agent_id": str(agent_id)
                }
            )
        
        # Initiate wake
        scheduler = await get_sleep_scheduler()
        success = await scheduler.force_wake(agent_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to wake agent")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Agent {agent_id} wake initiated",
                "agent_id": str(agent_id)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error forcing agent wake: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/status")
async def get_agent_sleep_status(
    agent_id: UUID = Path(..., description="Agent ID"),
    session: AsyncSession = Depends(get_async_session)
) -> Dict[str, Any]:
    """Get current sleep status for an agent."""
    try:
        agent = await session.get(Agent, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Get current cycle if active
        current_cycle = None
        if agent.current_cycle_id:
            current_cycle = await session.get(SleepWakeCycle, agent.current_cycle_id)
        
        # Get next scheduled sleep time
        scheduler = await get_sleep_scheduler()
        next_sleep_time = await scheduler.get_next_sleep_time(agent_id)
        
        return {
            "agent_id": str(agent_id),
            "current_state": agent.current_sleep_state.value,
            "last_sleep_time": agent.last_sleep_time.isoformat() if agent.last_sleep_time else None,
            "last_wake_time": agent.last_wake_time.isoformat() if agent.last_wake_time else None,
            "current_cycle_id": str(agent.current_cycle_id) if agent.current_cycle_id else None,
            "current_cycle": current_cycle.to_dict() if current_cycle else None,
            "next_scheduled_sleep": next_sleep_time.isoformat() if next_sleep_time else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent sleep status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Sleep Window Management Endpoints
@router.get("/sleep-windows")
async def list_sleep_windows(
    agent_id: Optional[UUID] = Query(None, description="Filter by agent ID"),
    active_only: bool = Query(True, description="Return only active windows"),
    session: AsyncSession = Depends(get_async_session)
) -> List[SleepWindowResponse]:
    """List sleep windows with optional filtering."""
    try:
        query = select(SleepWindow)
        
        if agent_id is not None:
            query = query.where(SleepWindow.agent_id == agent_id)
        
        if active_only:
            query = query.where(SleepWindow.active == True)
        
        query = query.order_by(desc(SleepWindow.priority), SleepWindow.created_at)
        
        result = await session.execute(query)
        windows = result.scalars().all()
        
        return [
            SleepWindowResponse(
                id=window.id,
                agent_id=window.agent_id,
                start_time=window.start_time.isoformat(),
                end_time=window.end_time.isoformat(),
                timezone=window.timezone,
                active=window.active,
                days_of_week=window.days_of_week,
                priority=window.priority,
                created_at=window.created_at.isoformat(),
                updated_at=window.updated_at.isoformat()
            )
            for window in windows
        ]
        
    except Exception as e:
        logger.error(f"Error listing sleep windows: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sleep-windows")
async def create_sleep_window(
    window_data: SleepWindowCreate,
    session: AsyncSession = Depends(get_async_session)
) -> SleepWindowResponse:
    """Create a new sleep window."""
    try:
        # Validate agent exists if specified
        if window_data.agent_id:
            agent = await session.get(Agent, window_data.agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent {window_data.agent_id} not found")
        
        # Create sleep window
        sleep_window = SleepWindow(
            agent_id=window_data.agent_id,
            start_time=window_data.start_time,
            end_time=window_data.end_time,
            timezone=window_data.timezone,
            active=window_data.active,
            days_of_week=window_data.days_of_week,
            priority=window_data.priority
        )
        
        session.add(sleep_window)
        await session.commit()
        await session.refresh(sleep_window)
        
        # Update scheduler
        scheduler = await get_sleep_scheduler()
        await scheduler.add_sleep_window(sleep_window)
        
        return SleepWindowResponse(
            id=sleep_window.id,
            agent_id=sleep_window.agent_id,
            start_time=sleep_window.start_time.isoformat(),
            end_time=sleep_window.end_time.isoformat(),
            timezone=sleep_window.timezone,
            active=sleep_window.active,
            days_of_week=sleep_window.days_of_week,
            priority=sleep_window.priority,
            created_at=sleep_window.created_at.isoformat(),
            updated_at=sleep_window.updated_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating sleep window: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/sleep-windows/{window_id}")
async def update_sleep_window(
    window_id: int = Path(..., description="Sleep window ID"),
    window_data: SleepWindowUpdate = Body(...),
    session: AsyncSession = Depends(get_async_session)
) -> SleepWindowResponse:
    """Update an existing sleep window."""
    try:
        sleep_window = await session.get(SleepWindow, window_id)
        if not sleep_window:
            raise HTTPException(status_code=404, detail=f"Sleep window {window_id} not found")
        
        # Apply updates
        update_dict = window_data.dict(exclude_unset=True)
        
        scheduler = await get_sleep_scheduler()
        success = await scheduler.update_sleep_window(window_id, update_dict)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update sleep window")
        
        # Refresh from database
        await session.refresh(sleep_window)
        
        return SleepWindowResponse(
            id=sleep_window.id,
            agent_id=sleep_window.agent_id,
            start_time=sleep_window.start_time.isoformat(),
            end_time=sleep_window.end_time.isoformat(),
            timezone=sleep_window.timezone,
            active=sleep_window.active,
            days_of_week=sleep_window.days_of_week,
            priority=sleep_window.priority,
            created_at=sleep_window.created_at.isoformat(),
            updated_at=sleep_window.updated_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating sleep window {window_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sleep-windows/{window_id}")
async def delete_sleep_window(
    window_id: int = Path(..., description="Sleep window ID"),
    session: AsyncSession = Depends(get_async_session)
) -> JSONResponse:
    """Delete a sleep window."""
    try:
        sleep_window = await session.get(SleepWindow, window_id)
        if not sleep_window:
            raise HTTPException(status_code=404, detail=f"Sleep window {window_id} not found")
        
        scheduler = await get_sleep_scheduler()
        success = await scheduler.remove_sleep_window(window_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete sleep window")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": f"Sleep window {window_id} deleted",
                "window_id": window_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting sleep window {window_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Checkpoint Management Endpoints
@router.get("/checkpoints")
async def list_checkpoints(
    agent_id: Optional[UUID] = Query(None, description="Filter by agent ID"),
    checkpoint_type: Optional[CheckpointType] = Query(None, description="Filter by checkpoint type"),
    valid_only: bool = Query(True, description="Return only valid checkpoints"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of checkpoints"),
    session: AsyncSession = Depends(get_async_session)
) -> List[CheckpointResponse]:
    """List checkpoints with optional filtering."""
    try:
        query = select(Checkpoint)
        
        if agent_id is not None:
            query = query.where(Checkpoint.agent_id == agent_id)
        
        if checkpoint_type is not None:
            query = query.where(Checkpoint.checkpoint_type == checkpoint_type)
        
        if valid_only:
            query = query.where(Checkpoint.is_valid == True)
        
        query = query.order_by(desc(Checkpoint.created_at)).limit(limit)
        
        result = await session.execute(query)
        checkpoints = result.scalars().all()
        
        return [
            CheckpointResponse(
                id=cp.id,
                agent_id=cp.agent_id,
                checkpoint_type=cp.checkpoint_type.value,
                path=cp.path,
                sha256=cp.sha256,
                size_bytes=cp.size_bytes,
                size_mb=cp.size_mb,
                is_valid=cp.is_valid,
                validation_errors=cp.validation_errors or [],
                metadata=cp.metadata or {},
                compression_ratio=cp.compression_ratio,
                creation_time_ms=cp.creation_time_ms,
                validation_time_ms=cp.validation_time_ms,
                created_at=cp.created_at.isoformat(),
                expires_at=cp.expires_at.isoformat() if cp.expires_at else None
            )
            for cp in checkpoints
        ]
        
    except Exception as e:
        logger.error(f"Error listing checkpoints: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/checkpoints/{checkpoint_id}")
async def get_checkpoint(
    checkpoint_id: UUID = Path(..., description="Checkpoint ID"),
    session: AsyncSession = Depends(get_async_session)
) -> CheckpointResponse:
    """Get detailed information about a specific checkpoint."""
    try:
        checkpoint = await session.get(Checkpoint, checkpoint_id)
        if not checkpoint:
            raise HTTPException(status_code=404, detail=f"Checkpoint {checkpoint_id} not found")
        
        return CheckpointResponse(
            id=checkpoint.id,
            agent_id=checkpoint.agent_id,
            checkpoint_type=checkpoint.checkpoint_type.value,
            path=checkpoint.path,
            sha256=checkpoint.sha256,
            size_bytes=checkpoint.size_bytes,
            size_mb=checkpoint.size_mb,
            is_valid=checkpoint.is_valid,
            validation_errors=checkpoint.validation_errors or [],
            metadata=checkpoint.checkpoint_metadata or {},
            compression_ratio=checkpoint.compression_ratio,
            creation_time_ms=checkpoint.creation_time_ms,
            validation_time_ms=checkpoint.validation_time_ms,
            created_at=checkpoint.created_at.isoformat(),
            expires_at=checkpoint.expires_at.isoformat() if checkpoint.expires_at else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting checkpoint {checkpoint_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/checkpoints")
async def create_checkpoint(
    agent_id: Optional[UUID] = Query(None, description="Agent ID for agent-specific checkpoint"),
    checkpoint_type: CheckpointType = Query(CheckpointType.MANUAL, description="Checkpoint type"),
    metadata: Dict[str, Any] = Body(default_factory=dict, description="Additional metadata")
) -> CheckpointResponse:
    """Create a new checkpoint."""
    try:
        checkpoint_manager = get_checkpoint_manager()
        checkpoint = await checkpoint_manager.create_checkpoint(
            agent_id=agent_id,
            checkpoint_type=checkpoint_type,
            metadata=metadata
        )
        
        if not checkpoint:
            raise HTTPException(status_code=500, detail="Failed to create checkpoint")
        
        return CheckpointResponse(
            id=checkpoint.id,
            agent_id=checkpoint.agent_id,
            checkpoint_type=checkpoint.checkpoint_type.value,
            path=checkpoint.path,
            sha256=checkpoint.sha256,
            size_bytes=checkpoint.size_bytes,
            size_mb=checkpoint.size_mb,
            is_valid=checkpoint.is_valid,
            validation_errors=checkpoint.validation_errors or [],
            metadata=checkpoint.checkpoint_metadata or {},
            compression_ratio=checkpoint.compression_ratio,
            creation_time_ms=checkpoint.creation_time_ms,
            validation_time_ms=checkpoint.validation_time_ms,
            created_at=checkpoint.created_at.isoformat(),
            expires_at=checkpoint.expires_at.isoformat() if checkpoint.expires_at else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating checkpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/checkpoints/{checkpoint_id}/validate")
async def validate_checkpoint(
    checkpoint_id: UUID = Path(..., description="Checkpoint ID to validate")
) -> Dict[str, Any]:
    """Validate a checkpoint's integrity."""
    try:
        checkpoint_manager = get_checkpoint_manager()
        is_valid, validation_errors = await checkpoint_manager.validate_checkpoint(checkpoint_id)
        
        return {
            "checkpoint_id": str(checkpoint_id),
            "is_valid": is_valid,
            "validation_errors": validation_errors,
            "validated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error validating checkpoint {checkpoint_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Recovery Endpoints
@router.post("/agents/{agent_id}/recover")
async def recover_agent(
    agent_id: UUID = Path(..., description="Agent ID to recover"),
    recovery_request: RecoveryRequest = Body(...)
) -> Dict[str, Any]:
    """Initiate recovery for an agent."""
    try:
        recovery_manager = get_recovery_manager()
        success, checkpoint = await recovery_manager.initiate_recovery(
            agent_id=agent_id,
            target_checkpoint_id=recovery_request.target_checkpoint_id,
            recovery_type=recovery_request.recovery_type
        )
        
        return {
            "status": "success" if success else "failed",
            "agent_id": str(agent_id),
            "recovery_type": recovery_request.recovery_type,
            "restored_checkpoint_id": str(checkpoint.id) if checkpoint else None,
            "recovered_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error recovering agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{agent_id}/emergency-recover")
async def emergency_recover_agent(
    agent_id: UUID = Path(..., description="Agent ID for emergency recovery")
) -> Dict[str, Any]:
    """Perform emergency recovery for an agent."""
    try:
        recovery_manager = get_recovery_manager()
        success = await recovery_manager.emergency_recovery(agent_id)
        
        return {
            "status": "success" if success else "failed",
            "agent_id": str(agent_id),
            "recovery_type": "emergency",
            "recovered_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in emergency recovery for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/recovery-readiness")
async def check_recovery_readiness(
    agent_id: UUID = Path(..., description="Agent ID to check")
) -> Dict[str, Any]:
    """Check if an agent is ready for recovery operations."""
    try:
        recovery_manager = get_recovery_manager()
        readiness = await recovery_manager.validate_recovery_readiness(agent_id)
        
        return readiness
        
    except Exception as e:
        logger.error(f"Error checking recovery readiness for agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health and Status Endpoints
@router.get("/health")
async def health_check(
    agent_id: Optional[UUID] = Query(None, description="Agent ID for agent-specific health check")
) -> HealthStatusResponse:
    """Perform health check for sleep-wake system."""
    try:
        recovery_manager = get_recovery_manager()
        readiness = await recovery_manager.validate_recovery_readiness(agent_id)
        
        return HealthStatusResponse(
            healthy=readiness["ready"],
            timestamp=datetime.utcnow().isoformat(),
            agent_id=agent_id,
            checks=readiness["checks"],
            errors=readiness["errors"],
            warnings=readiness["warnings"]
        )
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics")
async def get_sleep_wake_analytics(
    agent_id: Optional[UUID] = Query(None, description="Filter by agent ID"),
    start_date: Optional[date] = Query(None, description="Start date for analytics"),
    end_date: Optional[date] = Query(None, description="End date for analytics"),
    session: AsyncSession = Depends(get_async_session)
) -> List[Dict[str, Any]]:
    """Get sleep-wake analytics data."""
    try:
        query = select(SleepWakeAnalytics)
        
        if agent_id is not None:
            query = query.where(SleepWakeAnalytics.agent_id == agent_id)
        
        if start_date:
            query = query.where(SleepWakeAnalytics.date >= start_date)
        
        if end_date:
            query = query.where(SleepWakeAnalytics.date <= end_date)
        
        query = query.order_by(desc(SleepWakeAnalytics.date))
        
        result = await session.execute(query)
        analytics = result.scalars().all()
        
        return [analytics_record.to_dict() for analytics_record in analytics]
        
    except Exception as e:
        logger.error(f"Error getting sleep-wake analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/consolidation/{cycle_id}/status")
async def get_consolidation_status(
    cycle_id: UUID = Path(..., description="Sleep-wake cycle ID")
) -> Dict[str, Any]:
    """Get consolidation status for a sleep-wake cycle."""
    try:
        consolidation_engine = get_consolidation_engine()
        status = await consolidation_engine.get_consolidation_status(cycle_id)
        
        if not status:
            raise HTTPException(status_code=404, detail=f"Consolidation cycle {cycle_id} not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting consolidation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cleanup endpoint
@router.post("/cleanup")
async def cleanup_old_data() -> Dict[str, Any]:
    """Clean up old checkpoints and data."""
    try:
        checkpoint_manager = get_checkpoint_manager()
        cleaned_count = await checkpoint_manager.cleanup_old_checkpoints()
        
        return {
            "status": "success",
            "checkpoints_cleaned": cleaned_count,
            "cleaned_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up old data: {e}")
        raise HTTPException(status_code=500, detail=str(e))