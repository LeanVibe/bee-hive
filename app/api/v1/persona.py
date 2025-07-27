"""
Agent Persona System API Endpoints for LeanVibe Agent Hive 2.0

Provides RESTful API endpoints for persona management, assignment,
performance analytics, and recommendation services.

Features:
- Persona definition management (CRUD operations)
- Dynamic persona assignment to agents
- Performance analytics and insights
- Persona recommendation engine
- Capability tracking and optimization
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Path, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from starlette.status import HTTP_200_OK, HTTP_201_CREATED, HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from ...core.agent_persona_system import (
    get_agent_persona_system,
    assign_optimal_persona,
    get_agent_persona,
    update_agent_persona_performance,
    PersonaType,
    PersonaAdaptationMode,
    PersonaCapabilityLevel,
    PersonaDefinition,
    PersonaAssignment
)
from ...models.task import Task, TaskType
from ...models.agent import Agent
from ...core.dependencies import get_current_user
from ...models.user import User
from ...schemas.base import BaseResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/persona", tags=["Agent Persona System"])


# Request/Response Models

class PersonaCapabilityRequest(BaseModel):
    """Request model for persona capability."""
    name: str = Field(..., description="Capability name")
    level: PersonaCapabilityLevel = Field(..., description="Capability level")
    proficiency_score: float = Field(..., ge=0.0, le=1.0, description="Proficiency score (0.0-1.0)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")


class PersonaDefinitionRequest(BaseModel):
    """Request model for creating/updating persona definitions."""
    id: str = Field(..., description="Unique persona ID")
    name: str = Field(..., description="Human-readable persona name")
    description: str = Field(..., description="Detailed persona description")
    persona_type: PersonaType = Field(..., description="Persona type classification")
    adaptation_mode: PersonaAdaptationMode = Field(default=PersonaAdaptationMode.ADAPTIVE, description="Adaptation behavior")
    
    # Capabilities and preferences
    capabilities: Dict[str, PersonaCapabilityRequest] = Field(..., description="Persona capabilities")
    preferred_task_types: List[TaskType] = Field(..., description="Preferred task types")
    expertise_domains: List[str] = Field(..., description="Areas of expertise")
    
    # Behavioral characteristics
    communication_style: Dict[str, Any] = Field(default_factory=dict, description="Communication preferences")
    decision_making_style: Dict[str, Any] = Field(default_factory=dict, description="Decision making approach")
    problem_solving_approach: Dict[str, Any] = Field(default_factory=dict, description="Problem solving style")
    
    # Collaboration preferences
    min_team_size: int = Field(default=1, ge=1, description="Minimum preferred team size")
    max_team_size: int = Field(default=8, ge=1, description="Maximum preferred team size")
    collaboration_patterns: List[str] = Field(default_factory=list, description="Preferred collaboration patterns")
    mentoring_capability: bool = Field(default=False, description="Can mentor other agents")
    
    # Performance characteristics
    typical_response_time: float = Field(default=90.0, gt=0, description="Typical response time in seconds")
    accuracy_vs_speed_preference: float = Field(default=0.65, ge=0.0, le=1.0, description="Accuracy vs speed preference")
    risk_tolerance: float = Field(default=0.5, ge=0.0, le=1.0, description="Risk tolerance level")


class PersonaAssignmentRequest(BaseModel):
    """Request model for persona assignment."""
    agent_id: UUID = Field(..., description="Target agent ID")
    task_id: Optional[UUID] = Field(None, description="Optional task for context")
    preferred_persona_id: Optional[str] = Field(None, description="Specific persona to assign")
    context: Dict[str, Any] = Field(default_factory=dict, description="Assignment context")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")


class PersonaPerformanceUpdateRequest(BaseModel):
    """Request model for updating persona performance."""
    agent_id: UUID = Field(..., description="Agent ID")
    task_id: UUID = Field(..., description="Task ID")
    success: bool = Field(..., description="Task success status")
    completion_time: float = Field(..., gt=0, description="Task completion time in seconds")
    complexity: float = Field(default=0.5, ge=0.0, le=1.0, description="Task complexity score")


class PersonaAnalyticsRequest(BaseModel):
    """Request model for persona analytics."""
    persona_id: Optional[str] = Field(None, description="Specific persona ID (None for all)")
    time_range_hours: int = Field(default=24, ge=1, le=168, description="Time range in hours")
    include_capability_trends: bool = Field(default=True, description="Include capability trend analysis")
    include_recommendations: bool = Field(default=True, description="Include optimization recommendations")


class PersonaResponse(BaseModel):
    """Response model for persona operations."""
    success: bool = Field(..., description="Operation success status")
    persona: Optional[Dict[str, Any]] = Field(None, description="Persona data")
    message: str = Field(..., description="Response message")
    timestamp: str = Field(..., description="Response timestamp")


class PersonaListResponse(BaseModel):
    """Response model for persona listing."""
    success: bool = Field(..., description="Operation success status")
    personas: List[Dict[str, Any]] = Field(..., description="List of personas")
    total_count: int = Field(..., description="Total persona count")
    filters_applied: Dict[str, Any] = Field(..., description="Applied filters")
    timestamp: str = Field(..., description="Response timestamp")


class PersonaAssignmentResponse(BaseModel):
    """Response model for persona assignment."""
    success: bool = Field(..., description="Assignment success status")
    assignment: Dict[str, Any] = Field(..., description="Assignment details")
    persona: Dict[str, Any] = Field(..., description="Assigned persona")
    recommendation_reasoning: Dict[str, Any] = Field(..., description="Assignment reasoning")
    timestamp: str = Field(..., description="Response timestamp")


class PersonaAnalyticsResponse(BaseModel):
    """Response model for persona analytics."""
    success: bool = Field(..., description="Analytics generation success")
    analytics: Dict[str, Any] = Field(..., description="Comprehensive analytics data")
    insights: List[str] = Field(..., description="Key insights and findings")
    recommendations: List[str] = Field(..., description="Optimization recommendations")
    timestamp: str = Field(..., description="Response timestamp")


# API Endpoints

@router.post("/definitions", response_model=PersonaResponse, status_code=HTTP_201_CREATED)
async def create_persona_definition(
    request: PersonaDefinitionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Create a new persona definition.
    
    Creates a new persona with specified capabilities, preferences,
    and behavioral characteristics for agent specialization.
    """
    try:
        logger.info(
            "Creating persona definition",
            user_id=current_user.id,
            persona_id=request.id,
            persona_type=request.persona_type.value
        )
        
        persona_system = get_agent_persona_system()
        
        # Convert request to PersonaDefinition
        from ...core.agent_persona_system import PersonaCapability
        
        capabilities = {}
        for cap_name, cap_req in request.capabilities.items():
            capabilities[cap_name] = PersonaCapability(
                name=cap_req.name,
                level=cap_req.level,
                proficiency_score=cap_req.proficiency_score,
                confidence=cap_req.confidence
            )
        
        persona_def = PersonaDefinition(
            id=request.id,
            name=request.name,
            description=request.description,
            persona_type=request.persona_type,
            adaptation_mode=request.adaptation_mode,
            capabilities=capabilities,
            preferred_task_types=request.preferred_task_types,
            expertise_domains=request.expertise_domains,
            communication_style=request.communication_style,
            decision_making_style=request.decision_making_style,
            problem_solving_approach=request.problem_solving_approach,
            preferred_team_size=(request.min_team_size, request.max_team_size),
            collaboration_patterns=request.collaboration_patterns,
            mentoring_capability=request.mentoring_capability,
            typical_response_time=request.typical_response_time,
            accuracy_vs_speed_preference=request.accuracy_vs_speed_preference,
            risk_tolerance=request.risk_tolerance,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        
        success = await persona_system.register_persona(persona_def)
        
        if success:
            return PersonaResponse(
                success=True,
                persona=persona_def.__dict__,
                message="Persona definition created successfully",
                timestamp=datetime.utcnow().isoformat()
            )
        else:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Failed to create persona definition"
            )
            
    except ValidationError as e:
        logger.error(f"Validation error creating persona: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Invalid persona definition: {e}"
        )
    except Exception as e:
        logger.error(f"Error creating persona definition: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to create persona: {str(e)}"
        )


@router.get("/definitions", response_model=PersonaListResponse, status_code=HTTP_200_OK)
async def list_persona_definitions(
    task_type: Optional[TaskType] = Query(None, description="Filter by task type"),
    required_capabilities: Optional[str] = Query(None, description="Comma-separated required capabilities"),
    persona_type: Optional[PersonaType] = Query(None, description="Filter by persona type"),
    active_only: bool = Query(True, description="Show only active personas"),
    current_user: User = Depends(get_current_user)
):
    """
    List available persona definitions with optional filtering.
    
    Returns all registered personas that match the specified criteria,
    useful for persona selection and system overview.
    """
    try:
        logger.info(
            "Listing persona definitions",
            user_id=current_user.id,
            task_type=task_type.value if task_type else None,
            persona_type=persona_type.value if persona_type else None
        )
        
        persona_system = get_agent_persona_system()
        
        # Parse required capabilities
        capabilities_list = []
        if required_capabilities:
            capabilities_list = [cap.strip() for cap in required_capabilities.split(",")]
        
        # Get filtered personas
        personas = await persona_system.list_available_personas(
            task_type=task_type,
            required_capabilities=capabilities_list if capabilities_list else None
        )
        
        # Apply additional filters
        if persona_type:
            personas = [p for p in personas if p.persona_type == persona_type]
        
        if active_only:
            personas = [p for p in personas if p.active]
        
        # Convert to dict format
        persona_dicts = []
        for persona in personas:
            persona_dict = persona.__dict__.copy()
            # Convert capabilities to serializable format
            if 'capabilities' in persona_dict:
                persona_dict['capabilities'] = {
                    name: {
                        'name': cap.name,
                        'level': cap.level.value,
                        'proficiency_score': cap.proficiency_score,
                        'confidence': cap.confidence,
                        'usage_count': cap.usage_count,
                        'success_rate': cap.success_rate
                    }
                    for name, cap in persona_dict['capabilities'].items()
                }
            persona_dicts.append(persona_dict)
        
        filters_applied = {
            "task_type": task_type.value if task_type else None,
            "required_capabilities": capabilities_list,
            "persona_type": persona_type.value if persona_type else None,
            "active_only": active_only
        }
        
        return PersonaListResponse(
            success=True,
            personas=persona_dicts,
            total_count=len(persona_dicts),
            filters_applied=filters_applied,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error listing persona definitions: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to list personas: {str(e)}"
        )


@router.get("/definitions/{persona_id}", response_model=PersonaResponse, status_code=HTTP_200_OK)
async def get_persona_definition(
    persona_id: str = Path(..., description="Persona ID to retrieve"),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed information about a specific persona definition.
    
    Returns complete persona specification including capabilities,
    performance metrics, and current assignments.
    """
    try:
        logger.info(
            "Getting persona definition",
            user_id=current_user.id,
            persona_id=persona_id
        )
        
        persona_system = get_agent_persona_system()
        persona = await persona_system.get_persona(persona_id)
        
        if not persona:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Persona {persona_id} not found"
            )
        
        # Convert to serializable format
        persona_dict = persona.__dict__.copy()
        if 'capabilities' in persona_dict:
            persona_dict['capabilities'] = {
                name: {
                    'name': cap.name,
                    'level': cap.level.value,
                    'proficiency_score': cap.proficiency_score,
                    'confidence': cap.confidence,
                    'usage_count': cap.usage_count,
                    'success_rate': cap.success_rate,
                    'last_used': cap.last_used.isoformat() if cap.last_used else None
                }
                for name, cap in persona_dict['capabilities'].items()
            }
        
        return PersonaResponse(
            success=True,
            persona=persona_dict,
            message="Persona definition retrieved successfully",
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting persona definition: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to get persona: {str(e)}"
        )


@router.post("/assignments", response_model=PersonaAssignmentResponse, status_code=HTTP_201_CREATED)
async def assign_persona_to_agent(
    request: PersonaAssignmentRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Assign optimal persona to agent based on task and context.
    
    Uses the recommendation engine to select the best persona for
    the given agent, task, and context, or assigns a specific persona.
    """
    try:
        logger.info(
            "Assigning persona to agent",
            user_id=current_user.id,
            agent_id=str(request.agent_id),
            preferred_persona=request.preferred_persona_id,
            task_id=str(request.task_id) if request.task_id else None
        )
        
        persona_system = get_agent_persona_system()
        
        # Get task if provided
        task = None
        if request.task_id:
            # In a real implementation, you'd fetch the task from database
            # For now, we'll create a mock task object
            task = type('Task', (), {
                'id': request.task_id,
                'task_type': TaskType.CODE_GENERATION,  # Default
                'metadata': request.context
            })()
        
        # Assign persona
        assignment = await persona_system.assign_persona_to_agent(
            agent_id=request.agent_id,
            task=task,
            context=request.context,
            preferred_persona_id=request.preferred_persona_id,
            session_id=request.session_id
        )
        
        # Get assigned persona details
        persona = await persona_system.get_persona(assignment.persona_id)
        
        # Convert to serializable format
        assignment_dict = {
            'agent_id': str(assignment.agent_id),
            'persona_id': assignment.persona_id,
            'session_id': assignment.session_id,
            'assigned_at': assignment.assigned_at.isoformat(),
            'assignment_reason': assignment.assignment_reason,
            'confidence_score': assignment.confidence_score,
            'tasks_completed': assignment.tasks_completed,
            'success_rate': assignment.success_rate,
            'active_adaptations': assignment.active_adaptations
        }
        
        persona_dict = persona.__dict__.copy() if persona else {}
        
        return PersonaAssignmentResponse(
            success=True,
            assignment=assignment_dict,
            persona=persona_dict,
            recommendation_reasoning={'confidence': assignment.confidence_score},
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error assigning persona to agent: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to assign persona: {str(e)}"
        )


@router.get("/assignments/{agent_id}", response_model=PersonaAssignmentResponse, status_code=HTTP_200_OK)
async def get_agent_persona_assignment(
    agent_id: UUID = Path(..., description="Agent ID to get assignment for"),
    current_user: User = Depends(get_current_user)
):
    """
    Get current persona assignment for a specific agent.
    
    Returns the agent's current persona assignment with performance
    metrics and adaptation details.
    """
    try:
        logger.info(
            "Getting agent persona assignment",
            user_id=current_user.id,
            agent_id=str(agent_id)
        )
        
        persona_system = get_agent_persona_system()
        assignment = await persona_system.get_agent_current_persona(agent_id)
        
        if not assignment:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"No persona assignment found for agent {agent_id}"
            )
        
        # Get persona details
        persona = await persona_system.get_persona(assignment.persona_id)
        
        # Convert to serializable format
        assignment_dict = {
            'agent_id': str(assignment.agent_id),
            'persona_id': assignment.persona_id,
            'session_id': assignment.session_id,
            'assigned_at': assignment.assigned_at.isoformat(),
            'assignment_reason': assignment.assignment_reason,
            'confidence_score': assignment.confidence_score,
            'tasks_completed': assignment.tasks_completed,
            'success_rate': assignment.success_rate,
            'active_adaptations': assignment.active_adaptations
        }
        
        persona_dict = persona.__dict__.copy() if persona else {}
        
        return PersonaAssignmentResponse(
            success=True,
            assignment=assignment_dict,
            persona=persona_dict,
            recommendation_reasoning={'existing_assignment': True},
            timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent persona assignment: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to get assignment: {str(e)}"
        )


@router.put("/performance", status_code=HTTP_200_OK)
async def update_persona_performance(
    request: PersonaPerformanceUpdateRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Update persona performance metrics based on task completion.
    
    Records task success/failure and updates capability proficiency
    scores for continuous persona optimization.
    """
    try:
        logger.info(
            "Updating persona performance",
            user_id=current_user.id,
            agent_id=str(request.agent_id),
            task_id=str(request.task_id),
            success=request.success
        )
        
        # Create mock task for update
        task = type('Task', (), {
            'id': request.task_id,
            'metadata': {'required_capabilities': []}  # Would be populated from real task
        })()
        
        await update_agent_persona_performance(
            agent_id=request.agent_id,
            task=task,
            success=request.success,
            completion_time=request.completion_time,
            complexity=request.complexity
        )
        
        return {
            "success": True,
            "message": "Persona performance updated successfully",
            "agent_id": str(request.agent_id),
            "task_id": str(request.task_id),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating persona performance: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to update performance: {str(e)}"
        )


@router.get("/analytics", response_model=PersonaAnalyticsResponse, status_code=HTTP_200_OK)
async def get_persona_analytics(
    persona_id: Optional[str] = Query(None, description="Specific persona ID"),
    time_range_hours: int = Query(default=24, ge=1, le=168, description="Time range in hours"),
    include_trends: bool = Query(default=True, description="Include capability trends"),
    include_recommendations: bool = Query(default=True, description="Include recommendations"),
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive persona analytics and insights.
    
    Returns performance metrics, capability trends, assignment patterns,
    and optimization recommendations for personas.
    """
    try:
        logger.info(
            "Getting persona analytics",
            user_id=current_user.id,
            persona_id=persona_id,
            time_range_hours=time_range_hours
        )
        
        persona_system = get_agent_persona_system()
        analytics = await persona_system.get_persona_analytics(
            persona_id=persona_id,
            time_range_hours=time_range_hours
        )
        
        # Extract insights and recommendations
        insights = analytics.get("recommendations", [])
        recommendations = analytics.get("optimization_suggestions", [])
        
        return PersonaAnalyticsResponse(
            success=True,
            analytics=analytics,
            insights=insights,
            recommendations=recommendations,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error getting persona analytics: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to get analytics: {str(e)}"
        )


@router.delete("/assignments/{agent_id}", status_code=HTTP_200_OK)
async def remove_persona_assignment(
    agent_id: UUID = Path(..., description="Agent ID to remove assignment from"),
    current_user: User = Depends(get_current_user)
):
    """
    Remove persona assignment from agent.
    
    Clears the current persona assignment, returning the agent
    to unspecialized state for new assignment.
    """
    try:
        logger.info(
            "Removing persona assignment",
            user_id=current_user.id,
            agent_id=str(agent_id)
        )
        
        persona_system = get_agent_persona_system()
        success = await persona_system.remove_persona_assignment(agent_id)
        
        if success:
            return {
                "success": True,
                "message": "Persona assignment removed successfully",
                "agent_id": str(agent_id),
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"No persona assignment found for agent {agent_id}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing persona assignment: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to remove assignment: {str(e)}"
        )


@router.post("/recommendations/{agent_id}", status_code=HTTP_200_OK)
async def get_persona_recommendations(
    agent_id: UUID = Path(..., description="Agent ID to get recommendations for"),
    task_type: Optional[TaskType] = Query(None, description="Task type for context"),
    context: Dict[str, Any] = {},
    current_user: User = Depends(get_current_user)
):
    """
    Get persona recommendations for agent and task context.
    
    Returns ranked list of suitable personas for the given agent
    and task context with confidence scores and reasoning.
    """
    try:
        logger.info(
            "Getting persona recommendations",
            user_id=current_user.id,
            agent_id=str(agent_id),
            task_type=task_type.value if task_type else None
        )
        
        persona_system = get_agent_persona_system()
        
        # Get available personas
        personas = await persona_system.list_available_personas(task_type=task_type)
        
        # Create mock task for recommendation
        if task_type:
            task = type('Task', (), {
                'id': None,
                'task_type': task_type,
                'metadata': context
            })()
            
            # Get recommendation
            recommended_persona, confidence, reasoning = await persona_system.recommendation_engine.recommend_persona(
                task, agent_id, context, personas
            )
            
            recommendations = [
                {
                    "persona_id": recommended_persona.id,
                    "persona_name": recommended_persona.name,
                    "confidence_score": confidence,
                    "reasoning": reasoning
                }
            ]
        else:
            # Return all personas with basic scoring
            recommendations = [
                {
                    "persona_id": persona.id,
                    "persona_name": persona.name,
                    "confidence_score": 0.5,  # Neutral score without task context
                    "reasoning": {"general_suitability": True}
                }
                for persona in personas
            ]
        
        return {
            "success": True,
            "recommendations": recommendations,
            "agent_id": str(agent_id),
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting persona recommendations: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to get recommendations: {str(e)}"
        )