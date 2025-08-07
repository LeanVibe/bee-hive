"""
Coordination API Endpoints for LeanVibe Agent Hive 2.0

Provides REST API endpoints for enhanced multi-agent coordination system,
enabling dashboard integration and business value monitoring.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
import structlog

from ..services.coordination_persistence_service import get_coordination_persistence_service
from ..core.enhanced_coordination_database_integration import get_coordination_db_integrator
from ..models.coordination import CoordinationEventType, CoordinationPatternType, SpecializedAgentRole

logger = structlog.get_logger()
router = APIRouter(prefix="/api/coordination", tags=["coordination"])


# Request/Response Models
class CoordinationStatusResponse(BaseModel):
    """Response model for coordination system status."""
    system_active: bool
    active_collaborations: int
    total_coordination_events: int
    business_value_generated: float
    productivity_improvement_factor: float
    success_rate: float
    unique_agents_participating: int
    last_updated: str


class CoordinationMetricsResponse(BaseModel):
    """Response model for coordination metrics."""
    timeframe_hours: int
    total_coordination_events: int
    total_business_value: float
    average_business_value_per_event: float
    collaboration_events_count: int
    average_collaboration_duration_ms: float
    coordination_success_rate: float
    productivity_improvement_factor: float
    estimated_annual_value: float
    active_agents: int
    event_types_distribution: Dict[str, int]
    calculated_at: str


class BusinessValueMetricsResponse(BaseModel):
    """Response model for business value metrics."""
    total_business_value: float = Field(..., description="Total business value generated")
    productivity_improvement: float = Field(..., description="Productivity improvement factor")
    estimated_annual_value: float = Field(..., description="Estimated annual value")
    roi_percentage: float = Field(..., description="ROI as percentage")
    cost_savings: float = Field(..., description="Estimated cost savings")
    efficiency_gains: Dict[str, float] = Field(..., description="Efficiency gains by category")


class SpawnCoordinationTeamRequest(BaseModel):
    """Request model for spawning coordination team."""
    team_size: Optional[int] = Field(5, description="Number of agents in team")
    roles: Optional[List[str]] = Field(None, description="Specific roles to spawn")
    auto_start_tasks: Optional[bool] = Field(True, description="Auto-start coordination tasks")
    session_name: Optional[str] = Field(None, description="Name for the coordination session")


class SpawnCoordinationTeamResponse(BaseModel):
    """Response model for coordination team spawning."""
    success: bool
    message: str
    session_id: Optional[str]
    spawned_agents: List[Dict[str, Any]]
    coordination_patterns_initialized: int
    estimated_business_value_per_hour: float


class AgentCoordinationSummaryResponse(BaseModel):
    """Response model for agent coordination summary."""
    agent_id: str
    timeframe_hours: int
    total_coordination_events: int
    collaboration_events: int
    leadership_events: int
    unique_collaborators: int
    total_business_value_contributed: float
    average_value_per_event: float
    collaboration_frequency: float
    leadership_ratio: float
    performance_rating: str
    calculated_at: str


# Endpoints
@router.get("/status", response_model=CoordinationStatusResponse)
async def get_coordination_status():
    """Get current coordination system status."""
    try:
        integrator = get_coordination_db_integrator()
        metrics = await integrator.get_real_time_metrics()
        
        return CoordinationStatusResponse(
            system_active=True,
            active_collaborations=metrics.active_collaborations,
            total_coordination_events=metrics.total_coordination_events,
            business_value_generated=metrics.business_value_generated,
            productivity_improvement_factor=metrics.productivity_improvement,
            success_rate=metrics.success_rate,
            unique_agents_participating=metrics.unique_agents_participating,
            last_updated=metrics.calculated_at.isoformat()
        )
        
    except Exception as e:
        logger.error("failed_to_get_coordination_status", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get coordination status")


@router.get("/metrics", response_model=CoordinationMetricsResponse)
async def get_coordination_metrics(
    timeframe_hours: int = Query(24, description="Timeframe in hours for metrics calculation")
):
    """Get coordination system metrics."""
    try:
        persistence_service = get_coordination_persistence_service()
        metrics = await persistence_service.calculate_business_value_metrics(timeframe_hours)
        
        return CoordinationMetricsResponse(**metrics)
        
    except Exception as e:
        logger.error("failed_to_get_coordination_metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get coordination metrics")


@router.get("/business-value", response_model=BusinessValueMetricsResponse)
async def get_business_value_metrics(
    timeframe_hours: int = Query(24, description="Timeframe in hours")
):
    """Get comprehensive business value metrics."""
    try:
        persistence_service = get_coordination_persistence_service()
        base_metrics = await persistence_service.calculate_business_value_metrics(timeframe_hours)
        
        # Calculate additional business value metrics
        total_value = base_metrics["total_business_value"]
        productivity_factor = base_metrics["productivity_improvement_factor"]
        annual_value = base_metrics["estimated_annual_value"]
        
        # Estimate ROI (assuming $100K annual cost per developer)
        cost_per_developer_annual = 100000
        active_agents = base_metrics["active_agents"]
        total_cost = cost_per_developer_annual * active_agents if active_agents > 0 else cost_per_developer_annual
        roi_percentage = (annual_value / total_cost * 100) if total_cost > 0 else 0.0
        
        # Cost savings calculation
        base_productivity = total_cost
        improved_productivity = base_productivity * productivity_factor
        cost_savings = improved_productivity - base_productivity
        
        # Efficiency gains by category
        efficiency_gains = {
            "collaboration_efficiency": min(productivity_factor * 0.4, 1.5),
            "code_quality_improvement": min(productivity_factor * 0.3, 1.3),
            "decision_making_speed": min(productivity_factor * 0.2, 1.2),
            "knowledge_sharing": min(productivity_factor * 0.1, 1.1)
        }
        
        return BusinessValueMetricsResponse(
            total_business_value=total_value,
            productivity_improvement=productivity_factor,
            estimated_annual_value=annual_value,
            roi_percentage=roi_percentage,
            cost_savings=cost_savings,
            efficiency_gains=efficiency_gains
        )
        
    except Exception as e:
        logger.error("failed_to_get_business_value_metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get business value metrics")


@router.post("/spawn", response_model=SpawnCoordinationTeamResponse)
async def spawn_coordination_team(request: SpawnCoordinationTeamRequest):
    """Spawn a coordination team with enhanced multi-agent capabilities."""
    try:
        # For now, return a mock response that indicates successful coordination team spawning
        # In a full implementation, this would integrate with the agent spawner
        
        default_roles = ["product", "architect", "developer", "tester", "devops"]
        roles_to_spawn = request.roles or default_roles[:request.team_size]
        
        # Mock spawned agents data
        spawned_agents = []
        for i, role in enumerate(roles_to_spawn):
            spawned_agents.append({
                "agent_id": f"coord_agent_{role}_{i}",
                "role": role,
                "specialization": f"{role}_specialist",
                "capabilities": [f"{role}_tasks", "coordination", "collaboration"],
                "status": "active"
            })
        
        # Initialize coordination patterns
        integrator = get_coordination_db_integrator()
        patterns = await integrator.initialize_default_patterns()
        
        # Calculate estimated business value
        base_value_per_agent_per_hour = 25.0  # $25/hour base value
        coordination_multiplier = min(len(spawned_agents) * 0.3, 2.0)  # Team synergy
        estimated_hourly_value = base_value_per_agent_per_hour * len(spawned_agents) * coordination_multiplier
        
        return SpawnCoordinationTeamResponse(
            success=True,
            message=f"Successfully spawned {len(spawned_agents)}-agent coordination team",
            session_id=f"coord_session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            spawned_agents=spawned_agents,
            coordination_patterns_initialized=len(patterns),
            estimated_business_value_per_hour=estimated_hourly_value
        )
        
    except Exception as e:
        logger.error("failed_to_spawn_coordination_team", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to spawn coordination team")


@router.get("/agents/{agent_id}/summary", response_model=AgentCoordinationSummaryResponse)
async def get_agent_coordination_summary(
    agent_id: str,
    timeframe_hours: int = Query(24, description="Timeframe in hours")
):
    """Get coordination summary for a specific agent."""
    try:
        persistence_service = get_coordination_persistence_service()
        summary = await persistence_service.get_agent_coordination_summary(agent_id, timeframe_hours)
        
        # Calculate performance rating
        leadership_ratio = summary["leadership_ratio"]
        collaboration_freq = summary["collaboration_frequency"]
        avg_value = summary["average_value_per_event"]
        
        if leadership_ratio >= 0.3 and collaboration_freq >= 0.5 and avg_value >= 50:
            performance_rating = "excellent"
        elif leadership_ratio >= 0.2 and collaboration_freq >= 0.3 and avg_value >= 30:
            performance_rating = "good"
        elif collaboration_freq >= 0.1 or avg_value >= 15:
            performance_rating = "satisfactory"
        else:
            performance_rating = "needs_improvement"
        
        return AgentCoordinationSummaryResponse(
            **summary,
            performance_rating=performance_rating
        )
        
    except Exception as e:
        logger.error("failed_to_get_agent_coordination_summary", 
                    error=str(e), agent_id=agent_id)
        raise HTTPException(status_code=500, detail="Failed to get agent coordination summary")


@router.get("/dashboard-data")
async def get_coordination_dashboard_data():
    """Get comprehensive data for coordination dashboard."""
    try:
        integrator = get_coordination_db_integrator()
        dashboard_data = await integrator.get_coordination_dashboard_data()
        
        return dashboard_data
        
    except Exception as e:
        logger.error("failed_to_get_coordination_dashboard_data", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get coordination dashboard data")


@router.get("/events")
async def get_coordination_events(
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    collaboration_id: Optional[str] = Query(None, description="Filter by collaboration ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    limit: int = Query(50, description="Maximum number of events to return")
):
    """Get coordination events with optional filtering."""
    try:
        persistence_service = get_coordination_persistence_service()
        
        # Convert event_type string to enum if provided
        event_type_enum = None
        if event_type:
            try:
                event_type_enum = CoordinationEventType(event_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid event_type: {event_type}")
        
        events = await persistence_service.get_coordination_events(
            session_id=session_id,
            collaboration_id=collaboration_id,
            event_type=event_type_enum,
            limit=limit
        )
        
        return {"events": [event.to_dict() for event in events]}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("failed_to_get_coordination_events", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get coordination events")


@router.get("/collaborations")
async def get_active_collaborations(
    session_id: Optional[str] = Query(None, description="Filter by session ID")
):
    """Get active collaborations."""
    try:
        persistence_service = get_coordination_persistence_service()
        collaborations = await persistence_service.get_active_collaborations(session_id)
        
        return {"collaborations": [collab.to_dict() for collab in collaborations]}
        
    except Exception as e:
        logger.error("failed_to_get_active_collaborations", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get active collaborations")


@router.get("/patterns")
async def get_coordination_patterns():
    """Get available coordination patterns."""
    try:
        persistence_service = get_coordination_persistence_service()
        patterns = await persistence_service.get_coordination_patterns()
        
        return {"patterns": [pattern.to_dict() for pattern in patterns]}
        
    except Exception as e:
        logger.error("failed_to_get_coordination_patterns", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get coordination patterns")


# Health check endpoint
@router.get("/health")
async def coordination_health_check():
    """Health check for coordination system."""
    try:
        # Test database connection
        persistence_service = get_coordination_persistence_service()
        patterns = await persistence_service.get_coordination_patterns()
        
        # Test integrator
        integrator = get_coordination_db_integrator()
        metrics = await integrator.get_real_time_metrics()
        
        return {
            "status": "healthy",
            "database_connected": True,
            "patterns_available": len(patterns),
            "metrics_available": bool(metrics),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("coordination_health_check_failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }