"""
Comprehensive Customer Success API
Complete API integration for customer onboarding and success framework.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
import json
import logging

from fastapi import APIRouter, HTTPException, Depends, Query, Path, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.core.redis import get_redis_client
from app.core.security import SecurityManager, get_current_user

# Import our new customer success components
from app.core.customer_onboarding_engine import (
    get_qualification_engine, 
    get_feasibility_analyzer,
    LeadQualificationEngine,
    ProjectFeasibilityAnalyzer
)
from app.core.weekly_milestone_framework import (
    get_milestone_framework,
    WeeklyMilestoneFramework
)
from app.core.autonomous_project_execution import (
    get_project_executor,
    AutonomousProjectExecutor
)
from app.core.customer_expansion_engine import (
    get_expansion_engine,
    CustomerExpansionEngine
)
from app.services.customer_success_service import get_success_service, CustomerSuccessService


# Pydantic models for request/response
class LeadQualificationRequest(BaseModel):
    """Request model for lead qualification."""
    organization_name: str = Field(..., description="Organization name")
    contact_information: Dict[str, str] = Field(..., description="Contact information")
    project_requirements: Dict[str, Any] = Field(..., description="Project requirements")
    technical_readiness: Dict[str, Any] = Field(..., description="Technical readiness assessment")
    business_readiness: Dict[str, Any] = Field(..., description="Business readiness assessment")
    financial_readiness: Dict[str, Any] = Field(..., description="Financial readiness assessment")
    organizational_readiness: Dict[str, Any] = Field(..., description="Organizational readiness assessment")
    
    class Config:
        schema_extra = {
            "example": {
                "organization_name": "TechCorp Inc.",
                "contact_information": {
                    "primary_contact": "John Smith",
                    "email": "john.smith@techcorp.com",
                    "phone": "+1-555-0123"
                },
                "project_requirements": {
                    "service_type": "mvp_development",
                    "timeline_weeks": 8,
                    "complexity": "medium",
                    "requirements": [
                        "User authentication system",
                        "Product catalog with search",
                        "Shopping cart functionality"
                    ],
                    "budget_usd": 150000
                },
                "technical_readiness": {
                    "has_existing_codebase": False,
                    "development_team_size": 2,
                    "documentation_quality": "partial",
                    "preferred_tech_stack": ["React", "Node.js"]
                },
                "business_readiness": {
                    "requirements_clarity": "mostly_clear",
                    "timeline_urgency": "moderate",
                    "stakeholder_alignment": "good",
                    "has_success_criteria": True
                },
                "financial_readiness": {
                    "budget_status": "approved",
                    "decision_authority": "full",
                    "contract_timeline_weeks": 3
                },
                "organizational_readiness": {
                    "change_readiness": "open",
                    "team_availability": "partial",
                    "communication_structure": "good"
                }
            }
        }


class MilestoneValidationRequest(BaseModel):
    """Request model for milestone validation."""
    milestone_id: str = Field(..., description="Milestone ID to validate")
    validation_data: Dict[str, Any] = Field(..., description="Validation evidence and data")
    
    class Config:
        schema_extra = {
            "example": {
                "milestone_id": "milestone_proj_12345_week_1",
                "validation_data": {
                    "criterion_requirements_complete": {
                        "stakeholder_approval": {
                            "total_stakeholders": 3,
                            "approved_count": 3,
                            "feedback": ["Excellent analysis", "Very thorough", "Ready to proceed"]
                        },
                        "deliverable_url": "https://docs.company.com/requirements-spec"
                    },
                    "criterion_architecture_design": {
                        "technical_review": {
                            "review_score": 9.2,
                            "reviewer_feedback": "Solid architecture design"
                        },
                        "deliverable_url": "https://docs.company.com/architecture"
                    }
                }
            }
        }


class ProgressUpdateRequest(BaseModel):
    """Request model for project progress updates."""
    project_id: str = Field(..., description="Project ID")
    agent_id: str = Field(..., description="Agent reporting the progress")
    task_id: str = Field(..., description="Task being updated")
    current_progress: float = Field(..., ge=0.0, le=100.0, description="Current progress percentage")
    work_completed: str = Field(..., description="Description of work completed")
    time_spent_hours: float = Field(..., ge=0.0, description="Time spent on this update")
    blockers: List[str] = Field(default=[], description="Any blockers encountered")
    quality_indicators: Dict[str, float] = Field(default={}, description="Quality metrics")
    next_steps: List[str] = Field(default=[], description="Planned next steps")
    
    class Config:
        schema_extra = {
            "example": {
                "project_id": "proj_customer_20250801_123456",
                "agent_id": "agent_proj_123_requirements_analyst_1",
                "task_id": "task_proj_123_w1_1",
                "current_progress": 75.0,
                "work_completed": "Requirements analysis 75% complete. Documented 15 of 20 requirements.",
                "time_spent_hours": 8.0,
                "blockers": [],
                "quality_indicators": {
                    "documentation_completeness": 75.0,
                    "stakeholder_approval": 90.0
                },
                "next_steps": [
                    "Complete remaining 5 requirements",
                    "Schedule stakeholder review session"
                ]
            }
        }


class CustomerExpansionRequest(BaseModel):
    """Request model for customer expansion analysis."""
    customer_id: str = Field(..., description="Customer ID")
    customer_data: Dict[str, Any] = Field(..., description="Comprehensive customer data for analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "customer_techcorp",
                "customer_data": {
                    "customer_name": "TechCorp Inc.",
                    "current_services": ["mvp_development"],
                    "satisfaction_metrics": {
                        "overall_satisfaction": 8.7,
                        "nps_score": 65,
                        "support_satisfaction": 8.2
                    },
                    "engagement_metrics": {
                        "platform_usage_percentage": 78.0,
                        "feature_adoption_percentage": 65.0
                    },
                    "project_success_metrics": {
                        "delivery_success_rate": 95.0,
                        "timeline_adherence": 88.0
                    },
                    "lifetime_value": 150000
                }
            }
        }


# Response Models
class QualificationResponse(BaseModel):
    """Lead qualification response model."""
    status: str
    lead_id: str
    qualification_score: float
    qualification_level: str
    disqualifiers: List[str]
    next_steps: List[str]
    feasibility_assessment: Optional[Dict[str, Any]] = None


class MilestoneResponse(BaseModel):
    """Milestone validation response model."""
    status: str
    validation_id: str
    milestone_id: str
    overall_score: float
    milestone_status: str
    criteria_results: List[Dict[str, Any]]
    recommendations: List[str]
    next_actions: List[str]
    escalation_required: bool


class ProjectStatusResponse(BaseModel):
    """Project status response model."""
    status: str
    project_id: str
    project_name: str
    current_status: str
    current_phase: str
    overall_progress: float
    timeline: Dict[str, Any]
    team_status: Dict[str, Any]
    task_status: Dict[str, Any]
    quality_metrics: Dict[str, float]
    active_tasks: List[Dict[str, Any]]
    upcoming_milestones: List[Dict[str, Any]]


class ExpansionDashboardResponse(BaseModel):
    """Customer expansion dashboard response model."""
    status: str
    customer_id: str
    customer_name: str
    health_summary: Dict[str, Any]
    expansion_readiness: str
    expansion_opportunities: List[Dict[str, Any]]
    retention_actions: List[Dict[str, Any]]
    key_metrics: Dict[str, float]
    next_touchpoint: str


# API Router
router = APIRouter(prefix="/api/v1/customer-success", tags=["Customer Success"])

logger = logging.getLogger(__name__)


# Lead Qualification and Onboarding Endpoints
@router.post("/lead-qualification", response_model=QualificationResponse)
async def qualify_lead(
    request: LeadQualificationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Qualify a lead with comprehensive assessment and feasibility analysis."""
    
    try:
        logger.info(f"Qualifying lead for organization: {request.organization_name}")
        
        # Get qualification engine
        qualification_engine = await get_qualification_engine()
        
        # Prepare lead data
        lead_data = {
            "organization_name": request.organization_name,
            "contact_information": request.contact_information,
            "project_requirements": request.project_requirements,
            "technical_readiness": request.technical_readiness,
            "business_readiness": request.business_readiness,
            "financial_readiness": request.financial_readiness,
            "organizational_readiness": request.organizational_readiness
        }
        
        # Qualify lead
        lead_profile = await qualification_engine.qualify_lead(lead_data)
        
        # If qualified, perform feasibility analysis
        feasibility_assessment = None
        if lead_profile.qualification_score >= 60:
            feasibility_analyzer = await get_feasibility_analyzer()
            feasibility_result = await feasibility_analyzer.analyze_project_feasibility(
                request.project_requirements,
                lead_profile
            )
            
            feasibility_assessment = {
                "assessment_id": feasibility_result.assessment_id,
                "overall_feasibility": feasibility_result.overall_feasibility,
                "feasibility_level": feasibility_result.feasibility_level.value,
                "success_probability": feasibility_result.success_probability,
                "technical_feasibility": feasibility_result.technical_feasibility,
                "timeline_feasibility": feasibility_result.timeline_feasibility,
                "resource_feasibility": feasibility_result.resource_feasibility,
                "risk_factors": feasibility_result.risk_factors,
                "recommended_approach": feasibility_result.recommended_approach
            }
        
        return QualificationResponse(
            status="success",
            lead_id=lead_profile.lead_id,
            qualification_score=lead_profile.qualification_score,
            qualification_level=lead_profile.qualification_level.value,
            disqualifiers=lead_profile.disqualifiers,
            next_steps=lead_profile.next_steps,
            feasibility_assessment=feasibility_assessment
        )
        
    except Exception as e:
        logger.error(f"Failed to qualify lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/project-initiation/{lead_id}")
async def initiate_project_from_lead(
    lead_id: str = Path(..., description="Lead ID from qualification"),
    guarantee_config: Dict[str, Any] = {},
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Initiate autonomous project execution from qualified lead."""
    
    try:
        logger.info(f"Initiating project from lead: {lead_id}")
        
        # Get project executor
        project_executor = await get_project_executor()
        
        # Load lead profile (would implement lead profile retrieval)
        # For now, using sample project config
        project_config = {
            "customer_id": current_user["tenant_id"],
            "service_type": "mvp_development",
            "project_name": f"Project from Lead {lead_id}",
            "timeline_weeks": 4,
            "budget_usd": 150000
        }
        
        # Initiate project execution
        execution_result = await project_executor.initiate_project_execution(
            project_config, guarantee_config
        )
        
        return execution_result
        
    except Exception as e:
        logger.error(f"Failed to initiate project from lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Milestone Management Endpoints
@router.post("/milestones/create-plan/{guarantee_id}")
async def create_milestone_plan(
    guarantee_id: str = Path(..., description="Success guarantee ID"),
    service_type: str = Query(..., description="Service type"),
    custom_criteria: Optional[Dict[str, Any]] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create comprehensive 4-week milestone plan for a service."""
    
    try:
        logger.info(f"Creating milestone plan for guarantee: {guarantee_id}")
        
        # Get milestone framework
        milestone_framework = await get_milestone_framework()
        
        # Create milestone plan
        milestone_plan = await milestone_framework.create_milestone_plan(
            guarantee_id, service_type, custom_criteria
        )
        
        # Format response
        plan_summary = {}
        for week_key, milestone in milestone_plan.items():
            plan_summary[week_key] = {
                "milestone_id": milestone.milestone_id,
                "title": milestone.title,
                "description": milestone.description,
                "success_criteria_count": len(milestone.success_criteria),
                "minimum_success_threshold": milestone.minimum_success_threshold,
                "escalation_threshold": milestone.escalation_threshold
            }
        
        return {
            "status": "created",
            "guarantee_id": guarantee_id,
            "service_type": service_type,
            "milestone_plan": plan_summary,
            "total_weeks": len(milestone_plan),
            "total_criteria": sum(len(m.success_criteria) for m in milestone_plan.values())
        }
        
    except Exception as e:
        logger.error(f"Failed to create milestone plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/milestones/validate", response_model=MilestoneResponse)
async def validate_milestone(
    request: MilestoneValidationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Validate weekly milestone completion with comprehensive evidence."""
    
    try:
        logger.info(f"Validating milestone: {request.milestone_id}")
        
        # Get milestone framework
        milestone_framework = await get_milestone_framework()
        
        # Validate milestone
        validation_result = await milestone_framework.validate_weekly_milestone(
            request.milestone_id,
            request.validation_data
        )
        
        return MilestoneResponse(
            status="validated",
            validation_id=validation_result.validation_id,
            milestone_id=validation_result.milestone_id,
            overall_score=validation_result.overall_score,
            milestone_status=validation_result.status.value,
            criteria_results=validation_result.criteria_results,
            recommendations=validation_result.recommendations,
            next_actions=validation_result.next_actions,
            escalation_required=validation_result.escalation_required
        )
        
    except Exception as e:
        logger.error(f"Failed to validate milestone: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/milestones/{milestone_id}/status")
async def get_milestone_status(
    milestone_id: str = Path(..., description="Milestone ID"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get current status of a milestone."""
    
    try:
        # Get milestone framework
        milestone_framework = await get_milestone_framework()
        
        # Get milestone status
        status = await milestone_framework.get_milestone_status(milestone_id)
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get milestone status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Project Execution and Tracking Endpoints
@router.post("/projects/progress-update")
async def update_project_progress(
    request: ProgressUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Process real-time progress update from agents."""
    
    try:
        logger.info(f"Processing progress update for project: {request.project_id}")
        
        # Get project executor
        project_executor = await get_project_executor()
        
        # Prepare progress data
        progress_data = {
            "agent_id": request.agent_id,
            "task_id": request.task_id,
            "current_progress": request.current_progress,
            "work_completed": request.work_completed,
            "time_spent_hours": request.time_spent_hours,
            "blockers": request.blockers,
            "quality_indicators": request.quality_indicators,
            "next_steps": request.next_steps
        }
        
        # Process progress update
        update_result = await project_executor.process_progress_update(
            request.project_id, progress_data
        )
        
        return update_result
        
    except Exception as e:
        logger.error(f"Failed to process progress update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/status", response_model=ProjectStatusResponse)
async def get_project_status(
    project_id: str = Path(..., description="Project ID"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get comprehensive real-time project status."""
    
    try:
        # Get project executor
        project_executor = await get_project_executor()
        
        # Get project status
        status = await project_executor.get_project_status(project_id)
        
        if status["status"] != "success":
            raise HTTPException(status_code=404, detail=status["message"])
        
        return ProjectStatusResponse(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get project status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Customer Expansion and Retention Endpoints
@router.post("/expansion/analyze")
async def analyze_customer_expansion(
    request: CustomerExpansionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Analyze customer for expansion opportunities and create expansion profile."""
    
    try:
        logger.info(f"Analyzing customer expansion: {request.customer_id}")
        
        # Get expansion engine
        expansion_engine = await get_expansion_engine()
        
        # Create expansion profile
        expansion_profile = await expansion_engine.create_expansion_profile(
            request.customer_id, request.customer_data
        )
        
        return {
            "status": "analyzed",
            "customer_id": expansion_profile.customer_id,
            "health_score": expansion_profile.health_score.overall_score,
            "health_status": expansion_profile.health_score.health_status.value,
            "expansion_readiness": expansion_profile.expansion_readiness.value,
            "expansion_opportunities_count": len(expansion_profile.expansion_opportunities),
            "retention_actions_count": len(expansion_profile.retention_actions),
            "expansion_potential_value": float(expansion_profile.expansion_potential_value),
            "churn_risk_probability": expansion_profile.churn_risk_probability,
            "next_touchpoint": expansion_profile.next_touchpoint.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze customer expansion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/expansion/{customer_id}/dashboard", response_model=ExpansionDashboardResponse)
async def get_expansion_dashboard(
    customer_id: str = Path(..., description="Customer ID"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get comprehensive expansion dashboard for a customer."""
    
    try:
        # Get expansion engine
        expansion_engine = await get_expansion_engine()
        
        # Get expansion dashboard
        dashboard = await expansion_engine.get_expansion_dashboard(customer_id)
        
        if dashboard["status"] != "success":
            raise HTTPException(status_code=404, detail=dashboard["message"])
        
        return ExpansionDashboardResponse(**dashboard)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get expansion dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retention/{customer_id}/execute-actions")
async def execute_retention_actions(
    customer_id: str = Path(..., description="Customer ID"),
    action_ids: List[str] = [],
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Execute specific retention actions for a customer."""
    
    try:
        logger.info(f"Executing retention actions for customer: {customer_id}")
        
        # Get expansion engine
        expansion_engine = await get_expansion_engine()
        
        # Execute retention actions
        execution_result = await expansion_engine.execute_retention_actions(
            customer_id, action_ids
        )
        
        return execution_result
        
    except Exception as e:
        logger.error(f"Failed to execute retention actions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Analytics and Reporting Endpoints
@router.get("/analytics/customer-health-trends")
async def get_customer_health_trends(
    customer_ids: List[str] = Query(..., description="Customer IDs to analyze"),
    period_days: int = Query(90, ge=30, le=365, description="Analysis period in days"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get customer health trends analysis."""
    
    try:
        # Implementation would analyze health trends
        # For now, return sample data structure
        
        return {
            "status": "success",
            "analysis_period_days": period_days,
            "customers_analyzed": len(customer_ids),
            "health_trends": [
                {
                    "customer_id": customer_id,
                    "current_health_score": 85.0,  # Would be calculated
                    "trend_direction": "improving",
                    "health_change_30_days": 5.0,
                    "risk_level": "low"
                }
                for customer_id in customer_ids
            ],
            "aggregate_metrics": {
                "average_health_score": 82.5,
                "customers_improving": 8,
                "customers_declining": 2,
                "high_risk_customers": 1
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get customer health trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/success-guarantee-metrics")
async def get_success_guarantee_metrics(
    period_days: int = Query(30, ge=7, le=365, description="Analysis period"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get success guarantee performance metrics."""
    
    try:
        # Implementation would calculate actual metrics
        # For now, return sample data structure
        
        return {
            "status": "success",
            "period_days": period_days,
            "guarantee_metrics": {
                "total_guarantees": 45,
                "active_guarantees": 23,
                "successful_guarantees": 20,
                "failed_guarantees": 2,
                "success_rate": 90.9,
                "average_success_score": 87.3,
                "guarantee_claim_rate": 4.4
            },
            "milestone_metrics": {
                "total_milestones": 180,
                "milestones_passed": 162,
                "milestones_at_risk": 15,
                "milestones_failed": 3,
                "milestone_success_rate": 90.0
            },
            "quality_metrics": {
                "average_delivery_score": 92.1,
                "average_customer_satisfaction": 8.6,
                "timeline_adherence_rate": 88.9
            },
            "financial_metrics": {
                "total_guarantee_value": 6750000,
                "payouts_made": 300000,
                "payout_rate": 4.4
            },
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get success guarantee metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Webhook and Integration Endpoints
@router.post("/webhooks/project-milestone-completed")
async def handle_milestone_completion_webhook(
    webhook_data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """Handle milestone completion webhook from project execution system."""
    
    try:
        logger.info("Processing milestone completion webhook")
        
        # Add background task to process webhook
        background_tasks.add_task(
            process_milestone_completion_webhook,
            webhook_data
        )
        
        return {"status": "accepted", "message": "Webhook processing started"}
        
    except Exception as e:
        logger.error(f"Failed to handle milestone webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_milestone_completion_webhook(webhook_data: Dict[str, Any]):
    """Background task to process milestone completion webhook."""
    
    try:
        # Extract webhook data
        milestone_id = webhook_data.get("milestone_id")
        project_id = webhook_data.get("project_id")
        completion_status = webhook_data.get("status")
        
        if completion_status == "completed":
            # Trigger customer success actions
            success_service = await get_success_service()
            
            # Update success guarantee metrics
            # Implementation would update guarantee status
            
            # Trigger stakeholder notifications
            # Implementation would send notifications
            
        logger.info(f"Processed milestone completion webhook for: {milestone_id}")
        
    except Exception as e:
        logger.error(f"Failed to process milestone webhook: {e}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check endpoint for customer success API."""
    
    try:
        # Check component health
        redis_client = await get_redis_client()
        redis_health = await redis_client.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "lead_qualification": "available",
                "milestone_framework": "available",
                "project_execution": "available",
                "customer_expansion": "available",
                "redis_connection": "healthy" if redis_health else "unhealthy"
            },
            "api_version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }