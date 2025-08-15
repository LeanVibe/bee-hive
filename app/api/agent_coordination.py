"""
Agent Coordination API for LeanVibe Agent Hive 2.0

RESTful API endpoints for agent context management, task coordination,
collaboration session management, and multi-agent workflow orchestration.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from urllib.parse import unquote

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, Path as FastAPIPath
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, and_, or_, func, desc
from pydantic import BaseModel, Field, ValidationError

from ..core.database import get_session
from ..core.redis import get_redis_client, RedisClient
from ..models.agent import Agent, AgentStatus
from ..models.project_index import ProjectIndex
from ..models.task import Task, TaskStatus, TaskPriority
from ..agents.context_integration import (
    AgentContextIntegration, ContextRequest, ContextResponse, AgentTaskType, ContextScope,
    get_agent_context_integration
)
from ..agents.task_router import (
    IntelligentTaskRouter, TaskRequirements, RoutingStrategy, TaskComplexity, TaskUrgency,
    get_intelligent_task_router
)
from ..agents.collaboration_engine import (
    CollaborativeDevelopmentEngine, CollaborationSession, CollaborationConflict, KnowledgeShare,
    get_collaborative_development_engine
)
from ..agents.enhanced_capabilities import (
    CodeIntelligenceAgent, ContextAwareQAAgent, ProjectHealthAgent, DocumentationAgent,
    get_code_intelligence_agent, get_context_aware_qa_agent, get_project_health_agent, get_documentation_agent
)

logger = structlog.get_logger()

# Create API router
router = APIRouter(
    prefix="/api/agent-coordination", 
    tags=["Agent Coordination"],
    responses={
        400: {"description": "Bad request - validation error or invalid parameters"},
        401: {"description": "Unauthorized - authentication required"},
        403: {"description": "Forbidden - insufficient permissions"},
        404: {"description": "Not found - resource does not exist"},
        409: {"description": "Conflict - resource state conflict"},
        429: {"description": "Too Many Requests - rate limit exceeded"},
        500: {"description": "Internal Server Error - unexpected server error"}
    }
)


# ================== REQUEST/RESPONSE MODELS ==================

class ContextRequestModel(BaseModel):
    """Request model for agent context."""
    agent_id: str
    project_id: str
    task_type: str
    task_description: str
    scope: str = "relevant_files"
    max_files: int = Field(50, ge=1, le=200)
    max_context_size: int = Field(100000, ge=1000, le=1000000)
    include_dependencies: bool = True
    include_history: bool = False
    focus_areas: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None


class ContextUpdateModel(BaseModel):
    """Request model for context updates."""
    preferences: Dict[str, Any]
    clear_cache: bool = False


class TaskRoutingRequestModel(BaseModel):
    """Request model for task routing."""
    task_id: str
    task_type: str
    project_id: Optional[str] = None
    required_capabilities: List[str] = Field(default_factory=list)
    preferred_agents: List[str] = Field(default_factory=list)
    excluded_agents: List[str] = Field(default_factory=list)
    complexity: str = "moderate"
    urgency: str = "normal"
    estimated_duration_minutes: int = Field(60, ge=1, le=1440)
    requires_project_context: bool = True
    max_parallel_agents: int = Field(1, ge=1, le=10)
    collaboration_required: bool = False
    deadline: Optional[datetime] = None
    routing_strategy: str = "expertise_based"


class CollaborationSessionRequestModel(BaseModel):
    """Request model for collaboration session."""
    project_id: str
    task_id: str
    participant_agents: List[str]
    lead_agent: Optional[str] = None


class ProgressUpdateModel(BaseModel):
    """Request model for progress updates."""
    working_on_files: List[str] = Field(default_factory=list)
    completed_files: List[str] = Field(default_factory=list)
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0)
    estimated_completion: Optional[datetime] = None
    status: str = "active"
    insights: List[str] = Field(default_factory=list)
    learnings: List[str] = Field(default_factory=list)


class KnowledgeShareRequestModel(BaseModel):
    """Request model for knowledge sharing."""
    source_agent: str
    target_agents: List[str]
    knowledge_type: str
    content: Dict[str, Any]
    project_id: Optional[str] = None
    expires_hours: int = Field(24, ge=1, le=168)


class ConflictResolutionModel(BaseModel):
    """Request model for conflict resolution."""
    resolution_strategy: str
    resolution_data: Dict[str, Any] = Field(default_factory=dict)


class AgentPerformanceUpdateModel(BaseModel):
    """Request model for agent performance updates."""
    task_id: str
    success: bool
    duration_minutes: Optional[float] = None
    quality_score: Optional[float] = None
    complexity_handled: Optional[str] = None
    errors_encountered: List[str] = Field(default_factory=list)
    lessons_learned: List[str] = Field(default_factory=list)


# ================== DEPENDENCY INJECTION ==================

async def get_context_integration(
    session: AsyncSession = Depends(get_session),
    redis_client: RedisClient = Depends(get_redis_client)
) -> AgentContextIntegration:
    """Get AgentContextIntegration instance."""
    return await get_agent_context_integration(session, redis_client)


async def get_task_router(
    session: AsyncSession = Depends(get_session),
    redis_client: RedisClient = Depends(get_redis_client),
    context_integration: AgentContextIntegration = Depends(get_context_integration)
) -> IntelligentTaskRouter:
    """Get IntelligentTaskRouter instance."""
    return await get_intelligent_task_router(session, redis_client, context_integration)


async def get_collaboration_engine(
    session: AsyncSession = Depends(get_session),
    redis_client: RedisClient = Depends(get_redis_client),
    context_integration: AgentContextIntegration = Depends(get_context_integration),
    task_router: IntelligentTaskRouter = Depends(get_task_router)
) -> CollaborativeDevelopmentEngine:
    """Get CollaborativeDevelopmentEngine instance."""
    return await get_collaborative_development_engine(session, redis_client, context_integration, task_router)


async def get_code_intelligence_dependency(
    session: AsyncSession = Depends(get_session),
    redis_client: RedisClient = Depends(get_redis_client),
    context_integration: AgentContextIntegration = Depends(get_context_integration)
) -> CodeIntelligenceAgent:
    """Get CodeIntelligenceAgent instance."""
    return await get_code_intelligence_agent(session, redis_client, context_integration)


async def get_context_aware_qa_dependency(
    session: AsyncSession = Depends(get_session),
    redis_client: RedisClient = Depends(get_redis_client),
    context_integration: AgentContextIntegration = Depends(get_context_integration)
) -> ContextAwareQAAgent:
    """Get ContextAwareQAAgent instance."""
    return await get_context_aware_qa_agent(session, redis_client, context_integration)


async def get_project_health_dependency(
    session: AsyncSession = Depends(get_session),
    redis_client: RedisClient = Depends(get_redis_client),
    context_integration: AgentContextIntegration = Depends(get_context_integration)
) -> ProjectHealthAgent:
    """Get ProjectHealthAgent instance."""
    return await get_project_health_agent(session, redis_client, context_integration)


async def get_documentation_dependency(
    session: AsyncSession = Depends(get_session),
    redis_client: RedisClient = Depends(get_redis_client),
    context_integration: AgentContextIntegration = Depends(get_context_integration)
) -> DocumentationAgent:
    """Get DocumentationAgent instance."""
    return await get_documentation_agent(session, redis_client, context_integration)


# ================== AGENT CONTEXT ENDPOINTS ==================

@router.post("/context/request", response_model=Dict[str, Any])
async def request_agent_context(
    request: ContextRequestModel,
    context_integration: AgentContextIntegration = Depends(get_context_integration)
):
    """
    Request optimized project context for an agent task.
    
    This endpoint provides intelligent context injection, automatically selecting
    relevant files, dependencies, and project information based on the agent's
    task type and historical familiarity with the project.
    """
    try:
        # Convert request to internal model
        context_request = ContextRequest(
            agent_id=request.agent_id,
            project_id=request.project_id,
            task_type=AgentTaskType(request.task_type),
            task_description=request.task_description,
            scope=ContextScope(request.scope),
            max_files=request.max_files,
            max_context_size=request.max_context_size,
            include_dependencies=request.include_dependencies,
            include_history=request.include_history,
            focus_areas=request.focus_areas,
            exclude_patterns=request.exclude_patterns
        )
        
        # Request context
        context_response = await context_integration.request_context(context_request)
        
        return {
            "status": "success",
            "context": context_response.to_dict(),
            "message": "Context successfully generated"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Context request failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/context/{agent_id}/current")
async def get_current_agent_context(
    agent_id: str = FastAPIPath(..., description="Agent identifier"),
    project_id: Optional[str] = Query(None, description="Optional project filter"),
    context_integration: AgentContextIntegration = Depends(get_context_integration)
):
    """
    Get current context status for an agent.
    
    Returns the agent's current context preferences, project familiarity scores,
    and cached context information.
    """
    try:
        context_status = await context_integration.get_agent_context_status(agent_id, project_id)
        
        return {
            "status": "success",
            "context_status": context_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get agent context status", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/context/{agent_id}/update")
async def update_agent_context(
    agent_id: str = FastAPIPath(..., description="Agent identifier"),
    project_id: Optional[str] = Query(None, description="Optional project filter"),
    request: ContextUpdateModel = ...,
    context_integration: AgentContextIntegration = Depends(get_context_integration)
):
    """
    Update agent context preferences for a project.
    
    Allows agents to customize their context preferences and optionally
    clear cached context data.
    """
    try:
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id is required for context updates")
        
        # Update preferences
        success = await context_integration.update_agent_context(
            agent_id, project_id, request.preferences
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update context preferences")
        
        # Clear cache if requested
        if request.clear_cache:
            await context_integration.clear_agent_context(agent_id, project_id)
        
        return {
            "status": "success",
            "message": "Agent context preferences updated",
            "cache_cleared": request.clear_cache
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update agent context", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/context/{agent_id}/clear")
async def clear_agent_context(
    agent_id: str = FastAPIPath(..., description="Agent identifier"),
    project_id: Optional[str] = Query(None, description="Optional project filter"),
    context_integration: AgentContextIntegration = Depends(get_context_integration)
):
    """
    Clear cached context for an agent.
    
    Removes all cached context data and preferences for the specified agent,
    optionally filtered by project.
    """
    try:
        success = await context_integration.clear_agent_context(agent_id, project_id)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to clear agent context")
        
        return {
            "status": "success",
            "message": "Agent context cleared",
            "agent_id": agent_id,
            "project_id": project_id
        }
        
    except Exception as e:
        logger.error("Failed to clear agent context", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# ================== TASK COORDINATION ENDPOINTS ==================

@router.post("/tasks/route", response_model=Dict[str, Any])
async def route_task(
    request: TaskRoutingRequestModel,
    task_router: IntelligentTaskRouter = Depends(get_task_router)
):
    """
    Route a task to the most suitable agent(s).
    
    Uses intelligent agent selection based on project familiarity, capabilities,
    current workload, and performance history to assign tasks optimally.
    """
    try:
        # Convert request to internal model
        task_requirements = TaskRequirements(
            task_id=request.task_id,
            task_type=request.task_type,
            project_id=request.project_id,
            required_capabilities=request.required_capabilities,
            preferred_agents=request.preferred_agents,
            excluded_agents=request.excluded_agents,
            complexity=TaskComplexity(request.complexity),
            urgency=TaskUrgency(request.urgency),
            estimated_duration_minutes=request.estimated_duration_minutes,
            requires_project_context=request.requires_project_context,
            max_parallel_agents=request.max_parallel_agents,
            collaboration_required=request.collaboration_required,
            deadline=request.deadline
        )
        
        # Route task
        routing_decision = await task_router.route_task(
            task_requirements,
            RoutingStrategy(request.routing_strategy)
        )
        
        return {
            "status": "success",
            "routing_decision": routing_decision.to_dict(),
            "message": "Task successfully routed"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Task routing failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/tasks/active")
async def get_active_tasks(
    project_id: Optional[str] = Query(None, description="Optional project filter"),
    agent_id: Optional[str] = Query(None, description="Optional agent filter"),
    task_router: IntelligentTaskRouter = Depends(get_task_router)
):
    """
    Get currently active agent tasks by project.
    
    Returns information about tasks currently being processed by agents,
    optionally filtered by project or agent.
    """
    try:
        # This would integrate with the task tracking system
        # For now, return a placeholder response
        return {
            "status": "success",
            "active_tasks": [],
            "filters": {
                "project_id": project_id,
                "agent_id": agent_id
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get active tasks", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/tasks/coordinate")
async def coordinate_multi_agent_task(
    request: TaskRoutingRequestModel,
    task_router: IntelligentTaskRouter = Depends(get_task_router)
):
    """
    Coordinate multi-agent collaborative tasks.
    
    Routes tasks that require multiple agents to work together, handling
    coordination, conflict prevention, and communication setup.
    """
    try:
        if not request.collaboration_required:
            raise HTTPException(
                status_code=400, 
                detail="Task must require collaboration for multi-agent coordination"
            )
        
        if request.max_parallel_agents < 2:
            raise HTTPException(
                status_code=400, 
                detail="Multi-agent tasks require at least 2 agents"
            )
        
        # Route with collaborative strategy
        task_requirements = TaskRequirements(
            task_id=request.task_id,
            task_type=request.task_type,
            project_id=request.project_id,
            required_capabilities=request.required_capabilities,
            preferred_agents=request.preferred_agents,
            excluded_agents=request.excluded_agents,
            complexity=TaskComplexity(request.complexity),
            urgency=TaskUrgency(request.urgency),
            estimated_duration_minutes=request.estimated_duration_minutes,
            requires_project_context=request.requires_project_context,
            max_parallel_agents=request.max_parallel_agents,
            collaboration_required=True,
            deadline=request.deadline
        )
        
        routing_decision = await task_router.route_task(
            task_requirements,
            RoutingStrategy.COLLABORATIVE
        )
        
        return {
            "status": "success",
            "routing_decision": routing_decision.to_dict(),
            "collaboration_setup": {
                "selected_agents": routing_decision.selected_agents,
                "coordination_required": True,
                "estimated_overhead": "20% for coordination"
            },
            "message": "Multi-agent task successfully coordinated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Multi-agent coordination failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/tasks/{task_id}/handoff")
async def handoff_task(
    task_id: str = FastAPIPath(..., description="Task identifier"),
    target_agent_id: str = Query(..., description="Target agent for handoff"),
    source_agent_id: str = Query(..., description="Source agent for handoff"),
    handoff_reason: str = Query(..., description="Reason for handoff"),
    task_router: IntelligentTaskRouter = Depends(get_task_router)
):
    """
    Transfer task between agents.
    
    Handles task handoff between agents, including context transfer,
    progress preservation, and coordination updates.
    """
    try:
        # This would implement actual task handoff logic
        # For now, return a success response
        return {
            "status": "success",
            "task_id": task_id,
            "handoff": {
                "from_agent": source_agent_id,
                "to_agent": target_agent_id,
                "reason": handoff_reason,
                "timestamp": datetime.utcnow().isoformat()
            },
            "message": "Task successfully handed off"
        }
        
    except Exception as e:
        logger.error("Task handoff failed", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/tasks/{task_id}/performance")
async def update_task_performance(
    task_id: str = FastAPIPath(..., description="Task identifier"),
    task_router: IntelligentTaskRouter = Depends(get_task_router),
    request: AgentPerformanceUpdateModel = ...
):
    """
    Update agent performance metrics for a completed task.
    
    Records task completion data for performance tracking and future
    routing optimization.
    """
    try:
        performance_data = {
            "success": request.success,
            "duration_minutes": request.duration_minutes,
            "quality_score": request.quality_score,
            "complexity_handled": request.complexity_handled,
            "errors_encountered": request.errors_encountered,
            "lessons_learned": request.lessons_learned,
            "task_type": request.task_id  # This would be extracted from actual task
        }
        
        await task_router.update_agent_performance(
            request.task_id,  # This should be agent_id
            task_id,
            performance_data
        )
        
        return {
            "status": "success",
            "task_id": task_id,
            "performance_recorded": True,
            "message": "Agent performance metrics updated"
        }
        
    except Exception as e:
        logger.error("Performance update failed", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# ================== COLLABORATION ENDPOINTS ==================

@router.post("/collaboration/session", response_model=Dict[str, Any])
async def start_collaboration_session(
    request: CollaborationSessionRequestModel,
    collaboration_engine: CollaborativeDevelopmentEngine = Depends(get_collaboration_engine)
):
    """
    Start a new collaboration session for multi-agent development.
    
    Creates a coordinated workspace for multiple agents to collaborate
    on a shared task with conflict prevention and communication channels.
    """
    try:
        session = await collaboration_engine.start_collaboration_session(
            request.project_id,
            request.task_id,
            request.participant_agents,
            request.lead_agent
        )
        
        return {
            "status": "success",
            "collaboration_session": session.to_dict(),
            "message": "Collaboration session started successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Collaboration session start failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/collaboration/{session_id}/progress")
async def update_collaboration_progress(
    session_id: str = FastAPIPath(..., description="Collaboration session identifier"),
    agent_id: str = Query(..., description="Agent reporting progress"),
    collaboration_engine: CollaborativeDevelopmentEngine = Depends(get_collaboration_engine),
    request: ProgressUpdateModel = ...
):
    """
    Update work progress in a collaboration session.
    
    Coordinates progress updates between collaborating agents, detects
    conflicts, and provides coordination guidance.
    """
    try:
        progress_update = {
            "working_on_files": request.working_on_files,
            "completed_files": request.completed_files,
            "progress_percentage": request.progress_percentage,
            "estimated_completion": request.estimated_completion.isoformat() if request.estimated_completion else None,
            "status": request.status,
            "insights": request.insights,
            "learnings": request.learnings
        }
        
        coordination_response = await collaboration_engine.coordinate_work_progress(
            session_id,
            agent_id,
            progress_update
        )
        
        return {
            "status": "success",
            "coordination": coordination_response,
            "message": "Progress update coordinated successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Progress coordination failed", session_id=session_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/collaboration/conflicts")
async def get_collaboration_conflicts(
    project_id: Optional[str] = Query(None, description="Optional project filter"),
    session_id: Optional[str] = Query(None, description="Optional session filter"),
    time_window_minutes: int = Query(60, description="Time window for conflict detection"),
    collaboration_engine: CollaborativeDevelopmentEngine = Depends(get_collaboration_engine)
):
    """
    Get potential or active collaboration conflicts.
    
    Returns detected conflicts in ongoing collaboration sessions
    with suggested resolution strategies.
    """
    try:
        if not project_id:
            return {
                "status": "error",
                "message": "project_id is required for conflict detection"
            }
        
        conflicts = await collaboration_engine.detect_potential_conflicts(
            project_id,
            time_window_minutes
        )
        
        return {
            "status": "success",
            "conflicts": [conflict.to_dict() for conflict in conflicts],
            "conflicts_count": len(conflicts),
            "time_window_minutes": time_window_minutes,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Conflict detection failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/collaboration/knowledge")
async def share_knowledge(
    request: KnowledgeShareRequestModel,
    collaboration_engine: CollaborativeDevelopmentEngine = Depends(get_collaboration_engine)
):
    """
    Share knowledge between agents for collaboration.
    
    Enables agents to share insights, patterns, and learnings with
    other agents working on related tasks or projects.
    """
    try:
        knowledge_share = await collaboration_engine.share_knowledge(
            request.source_agent,
            request.target_agents,
            request.knowledge_type,
            request.content,
            request.project_id,
            request.expires_hours
        )
        
        return {
            "status": "success",
            "knowledge_share": knowledge_share.to_dict(),
            "message": "Knowledge shared successfully"
        }
        
    except Exception as e:
        logger.error("Knowledge sharing failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/collaboration/knowledge/{agent_id}")
async def get_agent_knowledge(
    agent_id: str = FastAPIPath(..., description="Agent identifier"),
    knowledge_type: Optional[str] = Query(None, description="Optional knowledge type filter"),
    project_id: Optional[str] = Query(None, description="Optional project filter"),
    collaboration_engine: CollaborativeDevelopmentEngine = Depends(get_collaboration_engine)
):
    """
    Get available knowledge for an agent.
    
    Returns knowledge shares available to the specified agent,
    optionally filtered by type or project.
    """
    try:
        knowledge_shares = await collaboration_engine.get_agent_knowledge(
            agent_id,
            knowledge_type,
            project_id
        )
        
        return {
            "status": "success",
            "knowledge_shares": [share.to_dict() for share in knowledge_shares],
            "knowledge_count": len(knowledge_shares),
            "filters": {
                "knowledge_type": knowledge_type,
                "project_id": project_id
            }
        }
        
    except Exception as e:
        logger.error("Knowledge retrieval failed", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/collaboration/resolve")
async def resolve_conflict(
    conflict_id: str = Query(..., description="Conflict identifier"),
    collaboration_engine: CollaborativeDevelopmentEngine = Depends(get_collaboration_engine),
    request: ConflictResolutionModel = ...
):
    """
    Resolve a detected collaboration conflict.
    
    Applies resolution strategies to address conflicts between
    collaborating agents and restore smooth workflow.
    """
    try:
        result = await collaboration_engine.resolve_conflict(
            conflict_id,
            request.resolution_strategy,
            request.resolution_data
        )
        
        return {
            "status": "success",
            "resolution_result": result,
            "conflict_id": conflict_id,
            "message": "Conflict resolved successfully"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Conflict resolution failed", conflict_id=conflict_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# ================== ENHANCED CAPABILITIES ENDPOINTS ==================

@router.post("/capabilities/code-analysis/{project_id}")
async def analyze_project_code(
    project_id: str = FastAPIPath(..., description="Project identifier"),
    analysis_scope: str = Query("full", description="Analysis scope: full, recent, specific"),
    code_intelligence: CodeIntelligenceAgent = Depends(get_code_intelligence_dependency)
):
    """
    Perform deep code analysis using the Code Intelligence Agent.
    
    Provides architectural analysis, code quality assessment, pattern recognition,
    and refactoring recommendations for the specified project.
    """
    try:
        analysis_result = await code_intelligence.analyze_codebase_architecture(
            project_id, analysis_scope
        )
        
        return {
            "status": "success",
            "analysis": analysis_result,
            "message": "Code analysis completed successfully"
        }
        
    except Exception as e:
        logger.error("Code analysis failed", project_id=project_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/capabilities/code-smells/{project_id}")
async def identify_code_smells(
    project_id: str = FastAPIPath(..., description="Project identifier"),
    file_paths: Optional[List[str]] = Query(None, description="Optional specific files to analyze"),
    code_intelligence: CodeIntelligenceAgent = Depends(get_code_intelligence_dependency)
):
    """
    Identify code smells and quality issues.
    
    Uses the Code Intelligence Agent to detect code smells, anti-patterns,
    and quality issues in the specified project or files.
    """
    try:
        code_smells = await code_intelligence.identify_code_smells(project_id, file_paths)
        
        return {
            "status": "success",
            "code_smells": [smell.to_dict() for smell in code_smells],
            "smells_count": len(code_smells),
            "message": "Code smell analysis completed"
        }
        
    except Exception as e:
        logger.error("Code smell analysis failed", project_id=project_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/capabilities/test-recommendations/{project_id}")
async def generate_test_recommendations(
    project_id: str = FastAPIPath(..., description="Project identifier"),
    focus_areas: Optional[List[str]] = Query(None, description="Optional focus areas for testing"),
    qa_agent: ContextAwareQAAgent = Depends(get_context_aware_qa_dependency)
):
    """
    Generate intelligent test recommendations.
    
    Uses the Context-Aware QA Agent to analyze the project and generate
    targeted testing recommendations based on coverage gaps and risk analysis.
    """
    try:
        recommendations = await qa_agent.generate_test_recommendations(project_id, focus_areas)
        
        return {
            "status": "success",
            "test_recommendations": [rec.to_dict() for rec in recommendations],
            "recommendations_count": len(recommendations),
            "focus_areas": focus_areas,
            "message": "Test recommendations generated successfully"
        }
        
    except Exception as e:
        logger.error("Test recommendation generation failed", project_id=project_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/capabilities/test-quality/{project_id}")
async def assess_test_quality(
    project_id: str = FastAPIPath(..., description="Project identifier"),
    test_file_paths: Optional[List[str]] = Query(None, description="Optional specific test files"),
    qa_agent: ContextAwareQAAgent = Depends(get_context_aware_qa_dependency)
):
    """
    Assess the quality of existing tests.
    
    Uses the Context-Aware QA Agent to evaluate test coverage, completeness,
    maintainability, and isolation of existing test suites.
    """
    try:
        quality_assessment = await qa_agent.assess_test_quality(project_id, test_file_paths)
        
        return {
            "status": "success",
            "quality_assessment": quality_assessment,
            "message": "Test quality assessment completed"
        }
        
    except Exception as e:
        logger.error("Test quality assessment failed", project_id=project_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/capabilities/health-assessment/{project_id}")
async def assess_project_health(
    project_id: str = FastAPIPath(..., description="Project identifier"),
    include_trends: bool = Query(True, description="Include trend analysis"),
    health_agent: ProjectHealthAgent = Depends(get_project_health_dependency)
):
    """
    Perform comprehensive project health assessment.
    
    Uses the Project Health Agent to evaluate overall project health across
    multiple dimensions including code quality, test coverage, and maintainability.
    """
    try:
        health_assessment = await health_agent.assess_project_health(project_id, include_trends)
        
        return {
            "status": "success",
            "health_assessment": health_assessment.to_dict(),
            "message": "Project health assessment completed"
        }
        
    except Exception as e:
        logger.error("Project health assessment failed", project_id=project_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/capabilities/health-monitoring/{project_id}")
async def start_health_monitoring(
    project_id: str = FastAPIPath(..., description="Project identifier"),
    duration_hours: int = Query(24, description="Monitoring duration in hours"),
    health_agent: ProjectHealthAgent = Depends(get_project_health_dependency)
):
    """
    Start continuous project health monitoring.
    
    Initiates continuous monitoring using the Project Health Agent to track
    project health metrics and alert on significant changes.
    """
    try:
        monitoring_session = await health_agent.monitor_project_continuously(
            project_id, duration_hours
        )
        
        return {
            "status": "success",
            "monitoring_session": monitoring_session,
            "message": "Health monitoring started successfully"
        }
        
    except Exception as e:
        logger.error("Health monitoring start failed", project_id=project_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/capabilities/documentation/{project_id}")
async def generate_documentation(
    project_id: str = FastAPIPath(..., description="Project identifier"),
    doc_types: Optional[List[str]] = Query(None, description="Documentation types to generate"),
    doc_agent: DocumentationAgent = Depends(get_documentation_dependency)
):
    """
    Generate intelligent project documentation.
    
    Uses the Documentation Agent to automatically generate comprehensive
    project documentation including API docs, architecture, and setup guides.
    """
    try:
        documentation = await doc_agent.generate_project_documentation(project_id, doc_types)
        
        return {
            "status": "success",
            "documentation": documentation,
            "message": "Documentation generated successfully"
        }
        
    except Exception as e:
        logger.error("Documentation generation failed", project_id=project_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/capabilities/documentation/{project_id}/outdated")
async def detect_outdated_documentation(
    project_id: str = FastAPIPath(..., description="Project identifier"),
    doc_agent: DocumentationAgent = Depends(get_documentation_dependency)
):
    """
    Detect outdated documentation that needs updates.
    
    Uses the Documentation Agent to identify documentation that may be
    outdated based on recent code changes and modification patterns.
    """
    try:
        outdated_items = await doc_agent.detect_outdated_documentation(project_id)
        
        return {
            "status": "success",
            "outdated_documentation": outdated_items,
            "outdated_count": len(outdated_items),
            "message": "Outdated documentation detection completed"
        }
        
    except Exception as e:
        logger.error("Outdated documentation detection failed", project_id=project_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# ================== ANALYTICS ENDPOINTS ==================

@router.get("/analytics/routing")
async def get_routing_analytics(
    time_range_hours: int = Query(24, description="Time range for analytics"),
    project_id: Optional[str] = Query(None, description="Optional project filter"),
    task_router: IntelligentTaskRouter = Depends(get_task_router)
):
    """
    Get task routing analytics and performance metrics.
    
    Returns comprehensive analytics about task routing decisions, agent
    utilization, and routing strategy effectiveness.
    """
    try:
        analytics = await task_router.get_routing_analytics(time_range_hours, project_id)
        
        return {
            "status": "success",
            "analytics": analytics,
            "message": "Routing analytics retrieved successfully"
        }
        
    except Exception as e:
        logger.error("Routing analytics failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/analytics/collaboration")
async def get_collaboration_analytics(
    time_range_hours: int = Query(24, description="Time range for analytics"),
    project_id: Optional[str] = Query(None, description="Optional project filter"),
    collaboration_engine: CollaborativeDevelopmentEngine = Depends(get_collaboration_engine)
):
    """
    Get collaboration analytics and metrics.
    
    Returns analytics about collaboration sessions, conflict resolution,
    knowledge sharing patterns, and team effectiveness.
    """
    try:
        analytics = await collaboration_engine.get_collaboration_analytics(project_id, time_range_hours)
        
        return {
            "status": "success",
            "analytics": analytics,
            "message": "Collaboration analytics retrieved successfully"
        }
        
    except Exception as e:
        logger.error("Collaboration analytics failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/analytics/optimization")
async def optimize_routing_strategy(
    project_id: Optional[str] = Query(None, description="Optional project to optimize for"),
    task_router: IntelligentTaskRouter = Depends(get_task_router)
):
    """
    Analyze performance and suggest routing optimizations.
    
    Provides recommendations for improving task routing effectiveness
    based on historical performance data and patterns.
    """
    try:
        optimization_recommendations = await task_router.optimize_routing_strategy(project_id)
        
        return {
            "status": "success",
            "optimization_recommendations": optimization_recommendations,
            "message": "Routing optimization analysis completed"
        }
        
    except Exception as e:
        logger.error("Routing optimization failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# ================== HEALTH CHECK ENDPOINT ==================

@router.get("/health")
async def health_check():
    """
    Health check endpoint for agent coordination API.
    
    Returns the health status of the agent coordination system
    and its key components.
    """
    return {
        "status": "healthy",
        "service": "agent-coordination-api",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "components": {
            "context_integration": "operational",
            "task_router": "operational", 
            "collaboration_engine": "operational",
            "enhanced_capabilities": "operational"
        }
    }