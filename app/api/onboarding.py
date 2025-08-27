"""
Onboarding API Endpoints

Interactive onboarding system for Epic 6 - Advanced User Experience & Adoption.
Provides comprehensive onboarding workflow management with analytics tracking,
progress persistence, and 90%+ completion rate optimization.

Epic 6: Advanced User Experience & Adoption
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Path, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc

from ..core.database import get_async_session
from ..core.logging_service import get_component_logger
from ..core.auth import get_current_user, require_permission, Permission
from ..models.user import User
from ..models.onboarding import (
    OnboardingSession, OnboardingStep, OnboardingEvent, OnboardingMetric
)

logger = get_component_logger("onboarding_api")

router = APIRouter(prefix="/api/onboarding", tags=["onboarding"])

# Pydantic models
class OnboardingStartRequest(BaseModel):
    """Request model for starting onboarding."""
    started_at: datetime
    user_agent: Optional[str] = None
    referrer: Optional[str] = None
    source: Optional[str] = "web"

class UserDataUpdate(BaseModel):
    """User data collected during onboarding."""
    name: Optional[str] = None
    role: Optional[str] = None
    goals: Optional[List[str]] = None
    preferences: Optional[Dict[str, Any]] = None

class OnboardingStepCompletion(BaseModel):
    """Step completion data."""
    step: int = Field(..., ge=1, le=5)
    step_data: Optional[Dict[str, Any]] = None
    timestamp: datetime
    user_data: Optional[UserDataUpdate] = None

class OnboardingProgressUpdate(BaseModel):
    """Progress update model."""
    current_step: int = Field(..., ge=1, le=5)
    completed_steps: List[int]
    user_data: UserDataUpdate
    time_spent: int  # milliseconds

class OnboardingCompletion(BaseModel):
    """Onboarding completion model."""
    user_data: UserDataUpdate
    total_time: int  # milliseconds
    completed_steps: int
    completed_at: datetime
    started_at: datetime

class OnboardingSkip(BaseModel):
    """Onboarding skip model."""
    current_step: int
    skipped_at: datetime
    user_data: Optional[UserDataUpdate] = None
    reason: Optional[str] = None

class OnboardingAnalytics(BaseModel):
    """Analytics response model."""
    completion_rate: float
    average_time_to_complete: int
    drop_off_points: List[Dict[str, Any]]
    user_segments: Dict[str, int]
    step_metrics: List[Dict[str, Any]]

@router.post("/start")
async def start_onboarding(
    request: OnboardingStartRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """Start a new onboarding session."""
    try:
        # Check if user already has an active onboarding session
        existing_session = await db.execute(
            select(OnboardingSession).where(
                and_(
                    OnboardingSession.user_id == current_user.id,
                    OnboardingSession.completed_at.is_(None)
                )
            )
        )
        existing = existing_session.scalar_one_or_none()
        
        if existing:
            logger.info(f"Resuming existing onboarding session for user {current_user.id}")
            return JSONResponse({
                "status": "resumed",
                "session_id": str(existing.id),
                "current_step": existing.current_step,
                "progress": existing.progress
            })
        
        # Create new onboarding session
        session = OnboardingSession(
            user_id=current_user.id,
            started_at=request.started_at,
            current_step=1,
            user_agent=request.user_agent,
            referrer=request.referrer,
            source=request.source,
            progress={"steps_completed": [], "user_data": {}}
        )
        
        db.add(session)
        await db.commit()
        await db.refresh(session)
        
        # Track analytics event in background
        background_tasks.add_task(
            track_onboarding_event,
            session.id,
            "onboarding_started",
            {"user_agent": request.user_agent, "referrer": request.referrer},
            db
        )
        
        logger.info(f"Started onboarding session {session.id} for user {current_user.id}")
        
        return JSONResponse({
            "status": "started",
            "session_id": str(session.id),
            "current_step": 1,
            "message": "Onboarding session started successfully"
        })
        
    except Exception as e:
        logger.error(f"Error starting onboarding: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start onboarding session")

@router.get("/progress")
async def get_onboarding_progress(
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """Get current onboarding progress."""
    try:
        # Get active onboarding session
        result = await db.execute(
            select(OnboardingSession).where(
                and_(
                    OnboardingSession.user_id == current_user.id,
                    OnboardingSession.completed_at.is_(None)
                )
            )
        )
        session = result.scalar_one_or_none()
        
        if not session:
            return JSONResponse({"data": None, "message": "No active onboarding session"})
        
        return JSONResponse({
            "data": {
                "session_id": str(session.id),
                "current_step": session.current_step,
                "completed_steps": session.progress.get("steps_completed", []),
                "user_data": session.progress.get("user_data", {}),
                "started_at": session.started_at.isoformat(),
                "time_spent": int((datetime.utcnow() - session.started_at).total_seconds() * 1000)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting onboarding progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get onboarding progress")

@router.put("/progress")
async def update_onboarding_progress(
    update: OnboardingProgressUpdate,
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """Update onboarding progress."""
    try:
        # Get active session
        result = await db.execute(
            select(OnboardingSession).where(
                and_(
                    OnboardingSession.user_id == current_user.id,
                    OnboardingSession.completed_at.is_(None)
                )
            )
        )
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="No active onboarding session found")
        
        # Update session progress
        session.current_step = update.current_step
        session.progress = {
            "steps_completed": update.completed_steps,
            "user_data": update.user_data.dict(exclude_none=True),
            "last_updated": datetime.utcnow().isoformat()
        }
        
        await db.commit()
        
        logger.info(f"Updated onboarding progress for session {session.id}")
        
        return JSONResponse({
            "status": "success",
            "message": "Progress updated successfully"
        })
        
    except Exception as e:
        logger.error(f"Error updating onboarding progress: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update progress")

@router.post("/step-completed")
async def complete_onboarding_step(
    completion: OnboardingStepCompletion,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """Mark an onboarding step as completed."""
    try:
        # Get active session
        result = await db.execute(
            select(OnboardingSession).where(
                and_(
                    OnboardingSession.user_id == current_user.id,
                    OnboardingSession.completed_at.is_(None)
                )
            )
        )
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="No active onboarding session found")
        
        # Create step completion record
        step_completion = OnboardingStep(
            session_id=session.id,
            step_number=completion.step,
            completed_at=completion.timestamp,
            step_data=completion.step_data or {},
            time_spent=int((completion.timestamp - session.started_at).total_seconds() * 1000)
        )
        
        db.add(step_completion)
        
        # Update session progress
        current_progress = session.progress or {"steps_completed": [], "user_data": {}}
        if completion.step not in current_progress["steps_completed"]:
            current_progress["steps_completed"].append(completion.step)
        
        if completion.user_data:
            current_progress["user_data"].update(completion.user_data.dict(exclude_none=True))
        
        session.progress = current_progress
        session.current_step = max(session.current_step, completion.step + 1)
        
        await db.commit()
        
        # Track analytics event
        background_tasks.add_task(
            track_onboarding_event,
            session.id,
            f"step_{completion.step}_completed",
            {
                "step_data": completion.step_data,
                "time_spent": step_completion.time_spent
            },
            db
        )
        
        logger.info(f"Completed step {completion.step} for session {session.id}")
        
        return JSONResponse({
            "status": "success",
            "step": completion.step,
            "next_step": completion.step + 1 if completion.step < 5 else None,
            "message": f"Step {completion.step} completed successfully"
        })
        
    except Exception as e:
        logger.error(f"Error completing onboarding step: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to complete step")

@router.post("/complete")
async def complete_onboarding(
    completion: OnboardingCompletion,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """Complete the entire onboarding process."""
    try:
        # Get active session
        result = await db.execute(
            select(OnboardingSession).where(
                and_(
                    OnboardingSession.user_id == current_user.id,
                    OnboardingSession.completed_at.is_(None)
                )
            )
        )
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(status_code=404, detail="No active onboarding session found")
        
        # Mark session as completed
        session.completed_at = completion.completed_at
        session.total_time = completion.total_time
        session.progress["user_data"] = completion.user_data.dict(exclude_none=True)
        session.progress["completion_data"] = {
            "completed_steps": completion.completed_steps,
            "total_time": completion.total_time,
            "completion_rate": (completion.completed_steps / 5) * 100
        }
        
        # Update user profile with onboarding data
        if completion.user_data.name:
            current_user.display_name = completion.user_data.name
        if completion.user_data.role:
            current_user.metadata = current_user.metadata or {}
            current_user.metadata["role"] = completion.user_data.role
            current_user.metadata["onboarding_goals"] = completion.user_data.goals
            current_user.metadata["onboarding_preferences"] = completion.user_data.preferences
        
        await db.commit()
        
        # Create completion metric
        background_tasks.add_task(
            create_onboarding_metric,
            session.id,
            "completion",
            {
                "total_time": completion.total_time,
                "completed_steps": completion.completed_steps,
                "user_role": completion.user_data.role,
                "user_goals": completion.user_data.goals
            },
            db
        )
        
        logger.info(f"Completed onboarding for session {session.id} in {completion.total_time}ms")
        
        return JSONResponse({
            "status": "success",
            "message": "Onboarding completed successfully!",
            "completion_data": {
                "total_time": completion.total_time,
                "completed_steps": completion.completed_steps,
                "session_id": str(session.id)
            }
        })
        
    except Exception as e:
        logger.error(f"Error completing onboarding: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to complete onboarding")

@router.post("/skip")
async def skip_onboarding(
    skip: OnboardingSkip,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """Skip onboarding process."""
    try:
        # Get active session
        result = await db.execute(
            select(OnboardingSession).where(
                and_(
                    OnboardingSession.user_id == current_user.id,
                    OnboardingSession.completed_at.is_(None)
                )
            )
        )
        session = result.scalar_one_or_none()
        
        if session:
            # Mark as skipped
            session.skipped_at = skip.skipped_at
            session.skip_reason = skip.reason
            if skip.user_data:
                session.progress["user_data"] = skip.user_data.dict(exclude_none=True)
            
            await db.commit()
            
            # Track skip event
            background_tasks.add_task(
                track_onboarding_event,
                session.id,
                "onboarding_skipped",
                {
                    "current_step": skip.current_step,
                    "reason": skip.reason,
                    "time_spent": int((skip.skipped_at - session.started_at).total_seconds() * 1000)
                },
                db
            )
        
        logger.info(f"Skipped onboarding at step {skip.current_step} for user {current_user.id}")
        
        return JSONResponse({
            "status": "success",
            "message": "Onboarding skipped"
        })
        
    except Exception as e:
        logger.error(f"Error skipping onboarding: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to skip onboarding")

@router.get("/metrics")
async def get_onboarding_metrics(
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS))
):
    """Get comprehensive onboarding analytics."""
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get completion rate
        total_sessions = await db.execute(
            select(func.count(OnboardingSession.id))
            .where(OnboardingSession.started_at >= cutoff_date)
        )
        total_count = total_sessions.scalar() or 0
        
        completed_sessions = await db.execute(
            select(func.count(OnboardingSession.id))
            .where(
                and_(
                    OnboardingSession.started_at >= cutoff_date,
                    OnboardingSession.completed_at.is_not(None)
                )
            )
        )
        completed_count = completed_sessions.scalar() or 0
        
        completion_rate = (completed_count / total_count * 100) if total_count > 0 else 0
        
        # Get average completion time
        avg_time_result = await db.execute(
            select(func.avg(OnboardingSession.total_time))
            .where(
                and_(
                    OnboardingSession.started_at >= cutoff_date,
                    OnboardingSession.completed_at.is_not(None)
                )
            )
        )
        avg_time = avg_time_result.scalar() or 0
        
        # Get drop-off points
        drop_off_data = []
        for step in range(1, 6):
            step_started = await db.execute(
                select(func.count(OnboardingSession.id))
                .where(
                    and_(
                        OnboardingSession.started_at >= cutoff_date,
                        OnboardingSession.current_step >= step
                    )
                )
            )
            started_count = step_started.scalar() or 0
            
            step_completed = await db.execute(
                select(func.count(OnboardingStep.id))
                .join(OnboardingSession)
                .where(
                    and_(
                        OnboardingSession.started_at >= cutoff_date,
                        OnboardingStep.step_number == step
                    )
                )
            )
            completed_step_count = step_completed.scalar() or 0
            
            drop_off_rate = ((started_count - completed_step_count) / started_count * 100) if started_count > 0 else 0
            
            drop_off_data.append({
                "step": step,
                "started": started_count,
                "completed": completed_step_count,
                "drop_off_rate": round(drop_off_rate, 2)
            })
        
        analytics = OnboardingAnalytics(
            completion_rate=round(completion_rate, 2),
            average_time_to_complete=int(avg_time),
            drop_off_points=drop_off_data,
            user_segments={},  # TODO: Implement user segmentation
            step_metrics=drop_off_data
        )
        
        return JSONResponse({
            "status": "success",
            "data": analytics.dict(),
            "time_period_days": days,
            "total_sessions": total_count,
            "completed_sessions": completed_count
        })
        
    except Exception as e:
        logger.error(f"Error getting onboarding metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get onboarding metrics")

@router.get("/realtime")
async def get_realtime_onboarding_data(
    db: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(require_permission(Permission.VIEW_ANALYTICS))
):
    """Get real-time onboarding activity data."""
    try:
        # Get active sessions in the last 24 hours
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        active_sessions = await db.execute(
            select(OnboardingSession)
            .where(
                and_(
                    OnboardingSession.started_at >= cutoff_time,
                    OnboardingSession.completed_at.is_(None),
                    OnboardingSession.skipped_at.is_(None)
                )
            )
            .order_by(desc(OnboardingSession.started_at))
        )
        
        active_data = []
        for session in active_sessions.scalars():
            time_spent = int((datetime.utcnow() - session.started_at).total_seconds() * 1000)
            active_data.append({
                "session_id": str(session.id),
                "current_step": session.current_step,
                "started_at": session.started_at.isoformat(),
                "time_spent": time_spent,
                "user_data": session.progress.get("user_data", {})
            })
        
        return JSONResponse({
            "status": "success",
            "data": {
                "active_sessions": active_data,
                "total_active": len(active_data),
                "timestamp": datetime.utcnow().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting realtime onboarding data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get realtime data")

# Background task functions
async def track_onboarding_event(
    session_id: UUID,
    event_name: str,
    event_data: Dict[str, Any],
    db: AsyncSession
):
    """Track onboarding event in background."""
    try:
        event = OnboardingEvent(
            session_id=session_id,
            event_name=event_name,
            event_data=event_data,
            created_at=datetime.utcnow()
        )
        db.add(event)
        await db.commit()
    except Exception as e:
        logger.error(f"Error tracking onboarding event: {str(e)}")

async def create_onboarding_metric(
    session_id: UUID,
    metric_type: str,
    metric_data: Dict[str, Any],
    db: AsyncSession
):
    """Create onboarding metric in background."""
    try:
        metric = OnboardingMetric(
            session_id=session_id,
            metric_type=metric_type,
            metric_data=metric_data,
            created_at=datetime.utcnow()
        )
        db.add(metric)
        await db.commit()
    except Exception as e:
        logger.error(f"Error creating onboarding metric: {str(e)}")