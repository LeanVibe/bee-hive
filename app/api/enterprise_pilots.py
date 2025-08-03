"""
Enterprise Pilot Management API Endpoints

FastAPI router for comprehensive enterprise pilot operations including
pilot creation, status management, ROI tracking, and success metrics.

CRITICAL COMPONENT: Provides REST API access to all enterprise pilot functionality.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, desc, func
from pydantic import BaseModel, Field, validator

from ..core.database import get_session
from ..core.database_models import (
    EnterprisePilot, ROIMetrics, ExecutiveEngagement, DemoSession, 
    DevelopmentTask, PilotTier, PilotStatus, EngagementType
)
from ..core.ai_model_integration import get_ai_model_service, execute_development_task

logger = structlog.get_logger()

router = APIRouter(prefix="/api/v1/pilots", tags=["Enterprise Pilots"])


# Pydantic Models for Request/Response
class ContactInfo(BaseModel):
    """Contact information model."""
    name: str
    email: str
    title: str
    phone: Optional[str] = None


class PilotCreateRequest(BaseModel):
    """Request model for creating enterprise pilot."""
    company_name: str = Field(..., min_length=1, max_length=255)
    company_tier: PilotTier
    industry: str = Field(..., min_length=1, max_length=100)
    annual_revenue: Optional[float] = Field(None, ge=0)
    employee_count: Optional[int] = Field(None, ge=1)
    
    # Contacts
    primary_contact: ContactInfo
    technical_contacts: List[ContactInfo] = []
    executive_contacts: List[ContactInfo] = []
    
    # Configuration
    use_cases: List[str] = []
    compliance_requirements: List[str] = []
    integration_requirements: List[str] = []
    success_criteria: Dict[str, float] = {}
    
    # Timeline
    pilot_duration_weeks: int = Field(4, ge=1, le=12)
    pilot_start_date: Optional[datetime] = None


class PilotResponse(BaseModel):
    """Response model for pilot information."""
    id: str
    pilot_id: str
    company_name: str
    company_tier: PilotTier
    industry: str
    current_status: PilotStatus
    success_score: float
    stakeholder_satisfaction: float
    technical_success_rate: float
    created_at: datetime
    updated_at: datetime


class ROIMetricsRequest(BaseModel):
    """Request model for ROI metrics."""
    baseline_velocity: float = Field(0.0, ge=0)
    baseline_quality_score: float = Field(0.0, ge=0, le=100)
    baseline_development_cost: float = Field(0.0, ge=0)
    
    current_velocity: float = Field(0.0, ge=0)
    current_quality_score: float = Field(0.0, ge=0, le=100)
    current_development_cost: float = Field(0.0, ge=0)
    
    measurement_type: str = "automated"
    validation_status: str = "pending"


class ExecutiveEngagementRequest(BaseModel):
    """Request model for executive engagement."""
    executive_name: str = Field(..., min_length=1)
    executive_title: str = Field(..., min_length=1)
    executive_email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    executive_role: str = Field(..., min_length=1)
    engagement_type: EngagementType
    scheduled_time: datetime
    duration_minutes: int = Field(30, ge=15, le=180)
    agenda_items: List[str] = []


class DevelopmentTaskRequest(BaseModel):
    """Request model for development task."""
    task_type: str = Field(..., min_length=1)
    task_description: str = Field(..., min_length=10)
    task_complexity: str = Field("medium", pattern=r'^(simple|medium|complex|critical)$')
    task_priority: str = Field("medium", pattern=r'^(low|medium|high|critical)$')
    requirements: Dict[str, Any] = {}
    context_data: Dict[str, Any] = {}
    estimated_hours: float = Field(0.0, ge=0)


# API Endpoints
@router.post("/", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_enterprise_pilot(
    pilot_request: PilotCreateRequest,
    session: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    """Create new enterprise pilot program."""
    
    try:
        # Generate unique pilot ID
        pilot_id = f"pilot_{str(uuid.uuid4())[:8]}"
        
        # Create pilot record
        pilot = EnterprisePilot(
            pilot_id=pilot_id,
            company_name=pilot_request.company_name,
            company_tier=pilot_request.company_tier,
            industry=pilot_request.industry,
            annual_revenue=pilot_request.annual_revenue,
            employee_count=pilot_request.employee_count,
            primary_contact=pilot_request.primary_contact.dict(),
            technical_contacts=[contact.dict() for contact in pilot_request.technical_contacts],
            executive_contacts=[contact.dict() for contact in pilot_request.executive_contacts],
            use_cases=pilot_request.use_cases,
            compliance_requirements=pilot_request.compliance_requirements,
            integration_requirements=pilot_request.integration_requirements,
            success_criteria=pilot_request.success_criteria,
            pilot_duration_weeks=pilot_request.pilot_duration_weeks,
            pilot_start_date=pilot_request.pilot_start_date or datetime.utcnow(),
            pilot_end_date=(
                pilot_request.pilot_start_date or datetime.utcnow()
            ) + timedelta(weeks=pilot_request.pilot_duration_weeks),
            current_status=PilotStatus.QUEUED
        )
        
        session.add(pilot)
        await session.commit()
        await session.refresh(pilot)
        
        logger.info(
            "Enterprise pilot created successfully",
            pilot_id=pilot.pilot_id,
            company=pilot.company_name,
            tier=pilot.company_tier.value
        )
        
        return {
            "success": True,
            "pilot_id": pilot.pilot_id,
            "message": f"Enterprise pilot created for {pilot.company_name}",
            "pilot": {
                "id": str(pilot.id),
                "pilot_id": pilot.pilot_id,
                "company_name": pilot.company_name,
                "company_tier": pilot.company_tier.value,
                "status": pilot.current_status.value,
                "pilot_start_date": pilot.pilot_start_date.isoformat(),
                "pilot_end_date": pilot.pilot_end_date.isoformat()
            }
        }
        
    except Exception as e:
        logger.error("Failed to create enterprise pilot", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create pilot: {str(e)}"
        )


@router.get("/", response_model=Dict[str, Any])
async def list_enterprise_pilots(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    company_tier: Optional[PilotTier] = None,
    status: Optional[PilotStatus] = None,
    industry: Optional[str] = None,
    session: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    """List enterprise pilots with filtering and pagination."""
    
    try:
        # Build query with filters
        query = select(EnterprisePilot)
        
        if company_tier:
            query = query.where(EnterprisePilot.company_tier == company_tier)
        if status:
            query = query.where(EnterprisePilot.current_status == status)
        if industry:
            query = query.where(EnterprisePilot.industry == industry)
        
        # Add pagination and ordering
        query = query.order_by(desc(EnterprisePilot.created_at)).offset(skip).limit(limit)
        
        # Execute query
        result = await session.execute(query)
        pilots = result.scalars().all()
        
        # Get total count
        count_query = select(func.count(EnterprisePilot.id))
        if company_tier:
            count_query = count_query.where(EnterprisePilot.company_tier == company_tier)
        if status:
            count_query = count_query.where(EnterprisePilot.current_status == status)
        if industry:
            count_query = count_query.where(EnterprisePilot.industry == industry)
        
        total_result = await session.execute(count_query)
        total_count = total_result.scalar()
        
        pilots_data = []
        for pilot in pilots:
            pilots_data.append({
                "id": str(pilot.id),
                "pilot_id": pilot.pilot_id,
                "company_name": pilot.company_name,
                "company_tier": pilot.company_tier.value,
                "industry": pilot.industry,
                "current_status": pilot.current_status.value,
                "success_score": float(pilot.success_score) if pilot.success_score else 0.0,
                "stakeholder_satisfaction": float(pilot.stakeholder_satisfaction) if pilot.stakeholder_satisfaction else 0.0,
                "technical_success_rate": float(pilot.technical_success_rate) if pilot.technical_success_rate else 0.0,
                "created_at": pilot.created_at.isoformat(),
                "updated_at": pilot.updated_at.isoformat()
            })
        
        return {
            "success": True,
            "pilots": pilots_data,
            "pagination": {
                "skip": skip,
                "limit": limit,
                "total": total_count,
                "has_more": skip + limit < total_count
            }
        }
        
    except Exception as e:
        logger.error("Failed to list enterprise pilots", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list pilots: {str(e)}"
        )


@router.get("/{pilot_id}", response_model=Dict[str, Any])
async def get_enterprise_pilot(
    pilot_id: str = Path(..., description="Pilot ID"),
    session: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    """Get detailed information about specific enterprise pilot."""
    
    try:
        # Query pilot with related data
        query = select(EnterprisePilot).where(EnterprisePilot.pilot_id == pilot_id)
        result = await session.execute(query)
        pilot = result.scalar_one_or_none()
        
        if not pilot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pilot not found: {pilot_id}"
            )
        
        # Get related ROI metrics
        roi_query = select(ROIMetrics).where(ROIMetrics.pilot_id == pilot.id).order_by(desc(ROIMetrics.created_at))
        roi_result = await session.execute(roi_query)
        roi_metrics = roi_result.scalars().all()
        
        # Get executive engagements
        engagement_query = select(ExecutiveEngagement).where(ExecutiveEngagement.pilot_id == pilot.id)
        engagement_result = await session.execute(engagement_query)
        engagements = engagement_result.scalars().all()
        
        # Get development tasks
        task_query = select(DevelopmentTask).where(DevelopmentTask.pilot_id == pilot.id)
        task_result = await session.execute(task_query)
        tasks = task_result.scalars().all()
        
        pilot_data = {
            "id": str(pilot.id),
            "pilot_id": pilot.pilot_id,
            "company_name": pilot.company_name,
            "company_tier": pilot.company_tier.value,
            "industry": pilot.industry,
            "annual_revenue": float(pilot.annual_revenue) if pilot.annual_revenue else None,
            "employee_count": pilot.employee_count,
            "primary_contact": pilot.primary_contact,
            "technical_contacts": pilot.technical_contacts,
            "executive_contacts": pilot.executive_contacts,
            "use_cases": pilot.use_cases,
            "compliance_requirements": pilot.compliance_requirements,
            "integration_requirements": pilot.integration_requirements,
            "success_criteria": pilot.success_criteria,
            "pilot_start_date": pilot.pilot_start_date.isoformat() if pilot.pilot_start_date else None,
            "pilot_end_date": pilot.pilot_end_date.isoformat() if pilot.pilot_end_date else None,
            "pilot_duration_weeks": pilot.pilot_duration_weeks,
            "current_status": pilot.current_status.value,
            "success_score": float(pilot.success_score) if pilot.success_score else 0.0,
            "stakeholder_satisfaction": float(pilot.stakeholder_satisfaction) if pilot.stakeholder_satisfaction else 0.0,
            "technical_success_rate": float(pilot.technical_success_rate) if pilot.technical_success_rate else 0.0,
            "created_at": pilot.created_at.isoformat(),
            "updated_at": pilot.updated_at.isoformat(),
            "roi_metrics_count": len(roi_metrics),
            "executive_engagements_count": len(engagements),
            "development_tasks_count": len(tasks)
        }
        
        return {
            "success": True,
            "pilot": pilot_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get enterprise pilot", pilot_id=pilot_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get pilot: {str(e)}"
        )


@router.put("/{pilot_id}/status", response_model=Dict[str, Any])
async def update_pilot_status(
    pilot_id: str = Path(..., description="Pilot ID"),
    new_status: PilotStatus = Query(..., description="New pilot status"),
    success_score: Optional[float] = Query(None, ge=0, le=100),
    stakeholder_satisfaction: Optional[float] = Query(None, ge=0, le=100),
    technical_success_rate: Optional[float] = Query(None, ge=0, le=100),
    session: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    """Update pilot status and success metrics."""
    
    try:
        # Find pilot
        query = select(EnterprisePilot).where(EnterprisePilot.pilot_id == pilot_id)
        result = await session.execute(query)
        pilot = result.scalar_one_or_none()
        
        if not pilot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pilot not found: {pilot_id}"
            )
        
        # Update status and metrics
        pilot.current_status = new_status
        if success_score is not None:
            pilot.success_score = success_score
        if stakeholder_satisfaction is not None:
            pilot.stakeholder_satisfaction = stakeholder_satisfaction
        if technical_success_rate is not None:
            pilot.technical_success_rate = technical_success_rate
        
        pilot.updated_at = datetime.utcnow()
        
        await session.commit()
        
        logger.info(
            "Pilot status updated",
            pilot_id=pilot_id,
            new_status=new_status.value,
            success_score=success_score
        )
        
        return {
            "success": True,
            "message": f"Pilot status updated to {new_status.value}",
            "pilot_id": pilot_id,
            "current_status": new_status.value,
            "success_metrics": {
                "success_score": float(pilot.success_score) if pilot.success_score else 0.0,
                "stakeholder_satisfaction": float(pilot.stakeholder_satisfaction) if pilot.stakeholder_satisfaction else 0.0,
                "technical_success_rate": float(pilot.technical_success_rate) if pilot.technical_success_rate else 0.0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update pilot status", pilot_id=pilot_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update pilot status: {str(e)}"
        )


@router.post("/{pilot_id}/roi-metrics", response_model=Dict[str, Any])
async def record_roi_metrics(
    pilot_id: str = Path(..., description="Pilot ID"),
    metrics_request: ROIMetricsRequest = ...,
    session: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    """Record ROI metrics for enterprise pilot."""
    
    try:
        # Find pilot
        query = select(EnterprisePilot).where(EnterprisePilot.pilot_id == pilot_id)
        result = await session.execute(query)
        pilot = result.scalar_one_or_none()
        
        if not pilot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pilot not found: {pilot_id}"
            )
        
        # Calculate improvement metrics
        velocity_improvement_factor = (
            metrics_request.current_velocity / metrics_request.baseline_velocity
            if metrics_request.baseline_velocity > 0 else 0.0
        )
        
        quality_improvement_percentage = (
            ((metrics_request.current_quality_score - metrics_request.baseline_quality_score) / 
             metrics_request.baseline_quality_score * 100)
            if metrics_request.baseline_quality_score > 0 else 0.0
        )
        
        cost_savings_percentage = (
            ((metrics_request.baseline_development_cost - metrics_request.current_development_cost) / 
             metrics_request.baseline_development_cost * 100)
            if metrics_request.baseline_development_cost > 0 else 0.0
        )
        
        # Estimate time saved and cost savings
        time_saved_hours = max(0, metrics_request.baseline_development_cost - metrics_request.current_development_cost) * 0.1  # Rough estimate
        cost_savings = max(0, metrics_request.baseline_development_cost - metrics_request.current_development_cost)
        
        # Calculate ROI percentage
        roi_percentage = (cost_savings / max(1, metrics_request.baseline_development_cost)) * 100
        
        # Create ROI metrics record
        roi_metrics = ROIMetrics(
            pilot_id=pilot.id,
            baseline_velocity=metrics_request.baseline_velocity,
            baseline_quality_score=metrics_request.baseline_quality_score,
            baseline_development_cost=metrics_request.baseline_development_cost,
            current_velocity=metrics_request.current_velocity,
            current_quality_score=metrics_request.current_quality_score,
            current_development_cost=metrics_request.current_development_cost,
            velocity_improvement_factor=velocity_improvement_factor,
            quality_improvement_percentage=quality_improvement_percentage,
            cost_savings_percentage=cost_savings_percentage,
            total_time_saved_hours=time_saved_hours,
            total_cost_savings=cost_savings,
            roi_percentage=roi_percentage,
            measurement_type=metrics_request.measurement_type,
            validation_status=metrics_request.validation_status
        )
        
        session.add(roi_metrics)
        await session.commit()
        await session.refresh(roi_metrics)
        
        logger.info(
            "ROI metrics recorded",
            pilot_id=pilot_id,
            velocity_improvement=f"{velocity_improvement_factor:.1f}x",
            roi_percentage=f"{roi_percentage:.1f}%"
        )
        
        return {
            "success": True,
            "message": "ROI metrics recorded successfully",
            "roi_metrics": {
                "id": str(roi_metrics.id),
                "velocity_improvement_factor": float(velocity_improvement_factor),
                "quality_improvement_percentage": float(quality_improvement_percentage),
                "cost_savings_percentage": float(cost_savings_percentage),
                "total_time_saved_hours": float(time_saved_hours),
                "total_cost_savings": float(cost_savings),
                "roi_percentage": float(roi_percentage),
                "measurement_date": roi_metrics.measurement_date.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to record ROI metrics", pilot_id=pilot_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record ROI metrics: {str(e)}"
        )


@router.post("/{pilot_id}/development-tasks", response_model=Dict[str, Any])
async def create_development_task(
    pilot_id: str = Path(..., description="Pilot ID"),
    task_request: DevelopmentTaskRequest = ...,
    session: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    """Create autonomous development task for pilot."""
    
    try:
        # Find pilot
        query = select(EnterprisePilot).where(EnterprisePilot.pilot_id == pilot_id)
        result = await session.execute(query)
        pilot = result.scalar_one_or_none()
        
        if not pilot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pilot not found: {pilot_id}"
            )
        
        # Generate task ID
        task_id = f"task_{str(uuid.uuid4())[:8]}"
        
        # Create development task
        task = DevelopmentTask(
            task_id=task_id,
            pilot_id=pilot.id,
            task_type=task_request.task_type,
            task_description=task_request.task_description,
            task_complexity=task_request.task_complexity,
            task_priority=task_request.task_priority,
            requirements=task_request.requirements,
            context_data=task_request.context_data,
            estimated_hours=task_request.estimated_hours,
            assigned_at=datetime.utcnow()
        )
        
        session.add(task)
        await session.commit()
        await session.refresh(task)
        
        # Execute task with AI model integration
        try:
            ai_response = await execute_development_task(
                prompt=f"""
                Please implement the following development task:
                
                Task Type: {task_request.task_type}
                Description: {task_request.task_description}
                Complexity: {task_request.task_complexity}
                Priority: {task_request.task_priority}
                
                Requirements: {task_request.requirements}
                Context: {task_request.context_data}
                
                Please provide:
                1. Implementation approach
                2. Code structure and architecture
                3. Testing strategy
                4. Quality assurance recommendations
                """,
                pilot_id=pilot_id,
                context=task_request.context_data
            )
            
            if ai_response.success:
                # Update task with AI results
                task.status = "completed" if ai_response.confidence_score > 0.8 else "in_progress"
                task.output_artifacts = [{"ai_response": ai_response.content}]
                task.quality_score = ai_response.confidence_score * 100
                task.completed_at = datetime.utcnow() if task.status == "completed" else None
                task.actual_hours = ai_response.response_time_ms / 3600000  # Convert ms to hours
                
                await session.commit()
        
        except Exception as ai_error:
            logger.warning("AI task execution failed", task_id=task_id, error=str(ai_error))
            # Task still created, just not executed by AI
        
        logger.info(
            "Development task created",
            pilot_id=pilot_id,
            task_id=task_id,
            task_type=task_request.task_type
        )
        
        return {
            "success": True,
            "message": "Development task created successfully",
            "task": {
                "id": str(task.id),
                "task_id": task.task_id,
                "task_type": task.task_type,
                "task_description": task.task_description,
                "status": task.status.value if hasattr(task.status, 'value') else task.status,
                "quality_score": float(task.quality_score) if task.quality_score else 0.0,
                "created_at": task.created_at.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create development task", pilot_id=pilot_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create development task: {str(e)}"
        )


@router.get("/{pilot_id}/analytics", response_model=Dict[str, Any])
async def get_pilot_analytics(
    pilot_id: str = Path(..., description="Pilot ID"),
    session: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    """Get comprehensive analytics for enterprise pilot."""
    
    try:
        # Find pilot
        query = select(EnterprisePilot).where(EnterprisePilot.pilot_id == pilot_id)
        result = await session.execute(query)
        pilot = result.scalar_one_or_none()
        
        if not pilot:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pilot not found: {pilot_id}"
            )
        
        # Get latest ROI metrics
        roi_query = select(ROIMetrics).where(ROIMetrics.pilot_id == pilot.id).order_by(desc(ROIMetrics.created_at)).limit(1)
        roi_result = await session.execute(roi_query)
        latest_roi = roi_result.scalar_one_or_none()
        
        # Get task statistics
        task_stats_query = select(
            func.count(DevelopmentTask.id).label('total_tasks'),
            func.avg(DevelopmentTask.quality_score).label('avg_quality'),
            func.sum(DevelopmentTask.actual_hours).label('total_hours'),
            func.avg(DevelopmentTask.velocity_improvement_factor).label('avg_velocity_improvement')
        ).where(DevelopmentTask.pilot_id == pilot.id)
        
        task_stats_result = await session.execute(task_stats_query)
        task_stats = task_stats_result.first()
        
        # Get engagement statistics
        engagement_stats_query = select(
            func.count(ExecutiveEngagement.id).label('total_engagements'),
            func.avg(ExecutiveEngagement.satisfaction_score).label('avg_satisfaction')
        ).where(ExecutiveEngagement.pilot_id == pilot.id)
        
        engagement_stats_result = await session.execute(engagement_stats_query)
        engagement_stats = engagement_stats_result.first()
        
        analytics = {
            "pilot_overview": {
                "pilot_id": pilot_id,
                "company_name": pilot.company_name,
                "company_tier": pilot.company_tier.value,
                "current_status": pilot.current_status.value,
                "days_active": (datetime.utcnow() - pilot.created_at).days,
                "success_score": float(pilot.success_score) if pilot.success_score else 0.0
            },
            "roi_metrics": {
                "velocity_improvement": float(latest_roi.velocity_improvement_factor) if latest_roi else 0.0,
                "cost_savings_percentage": float(latest_roi.cost_savings_percentage) if latest_roi else 0.0,
                "roi_percentage": float(latest_roi.roi_percentage) if latest_roi else 0.0,
                "total_cost_savings": float(latest_roi.total_cost_savings) if latest_roi else 0.0
            },
            "development_performance": {
                "total_tasks": int(task_stats.total_tasks) if task_stats.total_tasks else 0,
                "average_quality_score": float(task_stats.avg_quality) if task_stats.avg_quality else 0.0,
                "total_development_hours": float(task_stats.total_hours) if task_stats.total_hours else 0.0,
                "average_velocity_improvement": float(task_stats.avg_velocity_improvement) if task_stats.avg_velocity_improvement else 0.0
            },
            "executive_engagement": {
                "total_engagements": int(engagement_stats.total_engagements) if engagement_stats.total_engagements else 0,
                "average_satisfaction": float(engagement_stats.avg_satisfaction) if engagement_stats.avg_satisfaction else 0.0,
                "stakeholder_satisfaction": float(pilot.stakeholder_satisfaction) if pilot.stakeholder_satisfaction else 0.0
            }
        }
        
        return {
            "success": True,
            "analytics": analytics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get pilot analytics", pilot_id=pilot_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}"
        )