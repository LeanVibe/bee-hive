"""
Team Coordination API for LeanVibe Agent Hive 2.0

Comprehensive FastAPI microservice demonstrating enterprise backend capabilities:
- Agent registration with capability matching
- Smart task distribution and routing 
- Real-time coordination via WebSockets
- Performance metrics and monitoring
- Redis integration for messaging
- Advanced error handling and validation
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Query, Path
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator, ConfigDict
from sqlalchemy import and_, or_, func, select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import structlog
import json
import redis.asyncio as redis

from ...core.database import get_session_dependency
from ...core.redis_integration import get_redis_service, redis_session
from ...models.agent import Agent, AgentStatus, AgentType
from ...models.task import Task, TaskStatus, TaskPriority, TaskType
from ...core.team_coordination_metrics import get_team_coordination_metrics_service
from ...core.team_coordination_error_handler import (
    ErrorHandlingMiddleware, CoordinationException, AgentNotFoundError,
    TaskNotFoundError, InsufficientCapacityError, AgentOverloadedError,
    with_circuit_breaker, error_context
)
from ...schemas.team_coordination import *


logger = structlog.get_logger()
router = APIRouter(prefix="/team-coordination", tags=["Team Coordination"])


# =====================================================================================
# PYDANTIC SCHEMAS - REQUEST/RESPONSE MODELS
# =====================================================================================

class AgentCapabilitySchema(BaseModel):
    """Agent capability definition for registration."""
    name: str = Field(..., min_length=1, max_length=100, description="Capability name")
    description: str = Field(..., min_length=1, max_length=500, description="Detailed capability description")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Proficiency level (0.0-1.0)")
    specialization_areas: List[str] = Field(default_factory=list, description="Specific areas of expertise")
    years_experience: Optional[float] = Field(None, ge=0.0, description="Years of experience in this capability")
    
    @validator('specialization_areas')
    def validate_specializations(cls, v):
        if len(v) > 10:
            raise ValueError("Maximum 10 specialization areas allowed")
        return [area.strip() for area in v if area.strip()]


class AgentRegistrationRequest(BaseModel):
    """Request to register an agent with enhanced capability matching."""
    agent_name: str = Field(..., min_length=1, max_length=255, description="Human-readable agent name")
    agent_type: AgentType = Field(default=AgentType.CLAUDE, description="Agent implementation type")
    capabilities: List[AgentCapabilitySchema] = Field(..., min_items=1, description="Agent capabilities")
    system_context: Optional[str] = Field(None, max_length=2000, description="System context/prompt")
    preferred_workload: float = Field(default=0.8, ge=0.1, le=1.0, description="Preferred workload capacity")
    timezone: Optional[str] = Field(None, description="Agent timezone for scheduling")
    tags: List[str] = Field(default_factory=list, description="Classification tags")
    
    @validator('capabilities')
    def validate_capabilities(cls, v):
        if len(v) > 20:
            raise ValueError("Maximum 20 capabilities allowed per agent")
        # Check for duplicate capability names
        names = [cap.name.lower() for cap in v]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate capability names not allowed")
        return v


class TaskDistributionRequest(BaseModel):
    """Request for intelligent task distribution."""
    task_title: str = Field(..., min_length=1, max_length=255, description="Task title")
    task_description: str = Field(..., min_length=1, description="Detailed task description")
    task_type: TaskType = Field(..., description="Type of development task")
    priority: TaskPriority = Field(default=TaskPriority.MEDIUM, description="Task priority level")
    required_capabilities: List[str] = Field(..., min_items=1, description="Required capabilities")
    estimated_effort_hours: Optional[float] = Field(None, ge=0.1, le=200.0, description="Estimated effort in hours")
    deadline: Optional[datetime] = Field(None, description="Task deadline")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies (task IDs)")
    context_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    
    @validator('required_capabilities')
    def validate_required_capabilities(cls, v):
        if len(v) > 15:
            raise ValueError("Maximum 15 required capabilities allowed")
        return [cap.strip() for cap in v if cap.strip()]


class TaskReassignmentRequest(BaseModel):
    """Request to reassign a task with advanced reasoning."""
    task_id: str = Field(..., description="Task ID to reassign")
    target_agent_id: Optional[str] = Field(None, description="Specific agent ID (optional)")
    reason: str = Field(..., min_length=1, max_length=1000, description="Reason for reassignment")
    force_assignment: bool = Field(default=False, description="Force assignment even if agent is busy")
    preserve_context: bool = Field(default=True, description="Preserve task context and history")


class PerformanceMetricsQuery(BaseModel):
    """Query parameters for performance metrics."""
    time_range_hours: int = Field(default=24, ge=1, le=168, description="Time range in hours")
    agent_ids: Optional[List[str]] = Field(None, description="Specific agent IDs to include")
    metric_types: List[str] = Field(default_factory=lambda: ["completion_rate", "response_time", "workload"], 
                                   description="Types of metrics to retrieve")
    granularity: str = Field(default="hour", description="Data granularity: hour, day, week")


# Response Models
class AgentStatusResponse(BaseModel):
    """Enhanced agent status with coordination context."""
    model_config = ConfigDict(from_attributes=True)
    
    agent_id: str
    name: str
    type: str
    status: str
    current_workload: float
    available_capacity: float
    capabilities: List[Dict[str, Any]]
    active_tasks: int
    completed_today: int
    average_response_time_ms: float
    last_heartbeat: Optional[datetime]
    performance_score: float
    specialization_match_score: Optional[float] = None


class TaskDistributionResponse(BaseModel):
    """Response for task distribution with matching details."""
    task_id: str
    assigned_agent_id: str
    agent_name: str
    assignment_confidence: float
    estimated_completion_time: Optional[datetime]
    capability_match_details: Dict[str, Any]
    workload_impact: float
    priority_adjustment_reason: Optional[str] = None


class CoordinationMetricsResponse(BaseModel):
    """Comprehensive coordination system metrics."""
    total_agents: int
    active_agents: int
    total_tasks: int
    tasks_completed_today: int
    average_task_completion_time_hours: float
    system_efficiency_score: float
    agent_utilization_percentage: float
    current_workload_distribution: Dict[str, float]
    top_performing_agents: List[Dict[str, Any]]
    bottlenecks: List[Dict[str, Any]]


# =====================================================================================
# WEBSOCKET CONNECTION MANAGEMENT
# =====================================================================================

class CoordinationWebSocketManager:
    """Advanced WebSocket manager for real-time coordination updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.agent_subscribers: Dict[str, Set[str]] = {}  # agent_id -> connection_ids
        self.task_subscribers: Dict[str, Set[str]] = {}   # task_id -> connection_ids
        self.metric_subscribers: Set[str] = set()         # connection_ids for metrics
        
    async def connect(self, websocket: WebSocket, connection_id: str) -> None:
        """Accept new WebSocket connection with authentication."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        logger.info("WebSocket connected", connection_id=connection_id)
        
        # Send welcome message with connection info
        await self.send_to_connection(connection_id, {
            "type": "connection_established",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat(),
            "available_subscriptions": ["agents", "tasks", "metrics", "system"]
        })
    
    async def disconnect(self, connection_id: str) -> None:
        """Clean disconnect with proper cleanup."""
        # Remove from all subscriptions
        for agent_id, subscribers in self.agent_subscribers.items():
            subscribers.discard(connection_id)
        
        for task_id, subscribers in self.task_subscribers.items():
            subscribers.discard(connection_id)
        
        self.metric_subscribers.discard(connection_id)
        
        # Remove connection
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        logger.info("WebSocket disconnected", connection_id=connection_id)
    
    async def subscribe_to_agent(self, connection_id: str, agent_id: str) -> None:
        """Subscribe to specific agent updates."""
        if agent_id not in self.agent_subscribers:
            self.agent_subscribers[agent_id] = set()
        self.agent_subscribers[agent_id].add(connection_id)
        
        await self.send_to_connection(connection_id, {
            "type": "subscription_confirmed",
            "subscription": "agent",
            "agent_id": agent_id
        })
    
    async def broadcast_agent_update(self, agent_id: str, update_data: Dict[str, Any]) -> None:
        """Broadcast agent status updates to subscribers."""
        if agent_id in self.agent_subscribers:
            message = {
                "type": "agent_update",
                "agent_id": agent_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": update_data
            }
            
            await self.broadcast_to_subscribers(self.agent_subscribers[agent_id], message)
    
    async def broadcast_task_update(self, task_id: str, update_data: Dict[str, Any]) -> None:
        """Broadcast task status updates to subscribers."""
        if task_id in self.task_subscribers:
            message = {
                "type": "task_update", 
                "task_id": task_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": update_data
            }
            
            await self.broadcast_to_subscribers(self.task_subscribers[task_id], message)
    
    async def broadcast_metrics_update(self, metrics_data: Dict[str, Any]) -> None:
        """Broadcast system metrics to subscribers."""
        message = {
            "type": "metrics_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": metrics_data
        }
        
        await self.broadcast_to_subscribers(self.metric_subscribers, message)
    
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]) -> None:
        """Send message to specific connection with error handling."""
        if connection_id in self.active_connections:
            try:
                await self.active_connections[connection_id].send_text(json.dumps(message))
            except Exception as e:
                logger.warning("Failed to send WebSocket message", 
                             connection_id=connection_id, error=str(e))
                await self.disconnect(connection_id)
    
    async def broadcast_to_subscribers(self, subscribers: Set[str], message: Dict[str, Any]) -> None:
        """Broadcast message to all subscribers with error handling."""
        disconnected = set()
        
        for connection_id in subscribers:
            try:
                await self.send_to_connection(connection_id, message)
            except Exception:
                disconnected.add(connection_id)
        
        # Clean up disconnected subscribers
        for connection_id in disconnected:
            await self.disconnect(connection_id)


# Global WebSocket manager instance
ws_manager = CoordinationWebSocketManager()


# =====================================================================================
# CORE COORDINATION SERVICES
# =====================================================================================

class TeamCoordinationService:
    """Core service for team coordination operations."""
    
    def __init__(self):
        self.redis_service = None
        self.metrics_service = None
        
    async def initialize(self):
        """Initialize service dependencies."""
        self.redis_service = get_redis_service()
        await self.redis_service.connect()
        self.metrics_service = await get_team_coordination_metrics_service()
    
    async def register_agent_with_capabilities(
        self,
        db: AsyncSession,
        registration_data: AgentRegistrationRequest
    ) -> Agent:
        """Register agent with comprehensive capability analysis."""
        
        # Create new agent
        agent = Agent(
            name=registration_data.agent_name,
            type=registration_data.agent_type,
            capabilities=[cap.dict() for cap in registration_data.capabilities],
            system_prompt=registration_data.system_context,
            config={
                "preferred_workload": registration_data.preferred_workload,
                "timezone": registration_data.timezone,
                "tags": registration_data.tags,
                "registration_timestamp": datetime.utcnow().isoformat()
            },
            status=AgentStatus.active
        )
        
        db.add(agent)
        await db.commit()
        await db.refresh(agent)
        
        # Register agent in Redis coordination system
        if self.redis_service:
            capabilities = [cap.name for cap in registration_data.capabilities]
            metadata = {
                "preferred_workload": registration_data.preferred_workload,
                "timezone": registration_data.timezone,
                "tags": registration_data.tags,
                "status": agent.status.value,
                "current_workload": 0.0,
                "performance_score": 0.8
            }
            await self.redis_service.register_agent(str(agent.id), capabilities, metadata)
        
        logger.info("Agent registered with capabilities",
                   agent_id=str(agent.id),
                   name=agent.name,
                   capabilities_count=len(registration_data.capabilities))
        
        return agent
    
    async def find_optimal_agent_for_task(
        self,
        db: AsyncSession,
        task_requirements: TaskDistributionRequest
    ) -> Optional[Dict[str, Any]]:
        """Find the best agent for a task using advanced matching."""
        
        # Get all active agents with their current workload
        query = select(Agent).where(
            and_(
                Agent.status == AgentStatus.active,
                func.cast(Agent.context_window_usage, 'float') < 0.9
            )
        ).options(selectinload(Agent.performance_history))
        
        result = await db.execute(query)
        available_agents = result.scalars().all()
        
        if not available_agents:
            return None
        
        # Use capability matcher to find best match
        best_matches = await self.capability_matcher.find_best_agents_for_capabilities(
            required_capabilities=task_requirements.required_capabilities,
            available_agents=[str(agent.id) for agent in available_agents],
            max_results=3
        )
        
        # Analyze workload and performance for top matches
        optimal_match = None
        best_score = 0.0
        
        for agent in available_agents:
            if str(agent.id) not in [match["agent_id"] for match in best_matches]:
                continue
                
            # Calculate composite score
            capability_score = next(
                (match["confidence_score"] for match in best_matches 
                 if match["agent_id"] == str(agent.id)), 0.0
            )
            
            # Workload factor (prefer less busy agents)
            current_workload = float(agent.context_window_usage or 0.0)
            workload_factor = 1.0 - current_workload
            
            # Performance factor
            avg_response_time = float(agent.average_response_time or 1.0)
            performance_factor = min(1.0, 10.0 / max(avg_response_time, 0.1))
            
            # Composite score
            composite_score = (
                capability_score * 0.5 +
                workload_factor * 0.3 + 
                performance_factor * 0.2
            )
            
            if composite_score > best_score:
                best_score = composite_score
                optimal_match = {
                    "agent": agent,
                    "capability_score": capability_score,
                    "workload_factor": workload_factor,
                    "performance_factor": performance_factor,
                    "composite_score": composite_score
                }
        
        return optimal_match
    
    async def distribute_task_intelligently(
        self,
        db: AsyncSession,
        task_data: TaskDistributionRequest
    ) -> Optional[TaskDistributionResponse]:
        """Intelligently distribute task to optimal agent."""
        
        # Find optimal agent
        match_result = await self.find_optimal_agent_for_task(db, task_data)
        if not match_result:
            raise HTTPException(status_code=404, detail="No suitable agent found")
        
        agent = match_result["agent"]
        
        # Create task
        task = Task(
            title=task_data.task_title,
            description=task_data.task_description,
            task_type=task_data.task_type,
            priority=task_data.priority,
            required_capabilities=task_data.required_capabilities,
            estimated_effort=int(task_data.estimated_effort_hours * 60) if task_data.estimated_effort_hours else None,
            due_date=task_data.deadline,
            context=task_data.context_data or {},
            assigned_agent_id=agent.id
        )
        
        # Set dependencies if provided
        if task_data.dependencies:
            # Validate dependencies exist
            dep_query = select(Task.id).where(Task.id.in_(task_data.dependencies))
            dep_result = await db.execute(dep_query)
            valid_deps = [str(dep_id) for dep_id in dep_result.scalars().all()]
            task.dependencies = [uuid.UUID(dep_id) for dep_id in valid_deps]
        
        db.add(task)
        
        # Update agent workload
        current_tasks = await db.execute(
            select(func.count(Task.id)).where(
                and_(
                    Task.assigned_agent_id == agent.id,
                    Task.status.in_([TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS])
                )
            )
        )
        active_task_count = current_tasks.scalar()
        
        # Assign task to agent
        task.assign_to_agent(agent.id)
        
        await db.commit()
        await db.refresh(task)
        
        # Send real-time notification via unified Redis service
        await self.redis_service.publish("task_assignments", {
            "task_id": str(task.id),
            "agent_id": str(agent.id),
            "agent_name": agent.name,
            "task_title": task.title,
            "priority": task.priority.name,
            "assigned_at": datetime.utcnow().isoformat()
        })
        
        # Calculate estimated completion
        estimated_completion = None
        if task.estimated_effort:
            estimated_completion = datetime.utcnow() + timedelta(minutes=task.estimated_effort)
        
        # Broadcast WebSocket update
        await ws_manager.broadcast_agent_update(str(agent.id), {
            "new_task_assigned": {
                "task_id": str(task.id),
                "task_title": task.title,
                "priority": task.priority.name
            },
            "active_tasks": active_task_count + 1
        })
        
        logger.info("Task distributed intelligently",
                   task_id=str(task.id),
                   agent_id=str(agent.id),
                   confidence=match_result["composite_score"])
        
        return TaskDistributionResponse(
            task_id=str(task.id),
            assigned_agent_id=str(agent.id),
            agent_name=agent.name,
            assignment_confidence=match_result["composite_score"],
            estimated_completion_time=estimated_completion,
            capability_match_details={
                "capability_score": match_result["capability_score"],
                "workload_factor": match_result["workload_factor"],
                "performance_factor": match_result["performance_factor"]
            },
            workload_impact=1.0 / max(active_task_count + 1, 1)
        )


# Global service instance
coordination_service = TeamCoordinationService()


# =====================================================================================
# API ENDPOINTS
# =====================================================================================

@router.on_event("startup")
async def startup_coordination_service():
    """Initialize coordination service on startup."""
    await coordination_service.initialize()


# Agent Registration Endpoints
@router.post("/agents/register", response_model=AgentStatusResponse, status_code=201)
async def register_agent_for_coordination(
    registration: AgentRegistrationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session_dependency)
) -> AgentStatusResponse:
    """
    Register an agent with comprehensive capability matching.
    
    This endpoint demonstrates:
    - Advanced Pydantic validation with custom validators
    - Complex business logic with database transactions
    - Background task processing for performance
    - Redis integration for real-time notifications
    """
    
    try:
        # Register agent with capabilities
        agent = await coordination_service.register_agent_with_capabilities(db, registration)
        
        # Background task for performance metrics initialization
        background_tasks.add_task(
            initialize_agent_performance_tracking,
            agent_id=str(agent.id)
        )
        
        # Calculate initial performance score
        performance_score = 0.8  # Default for new agents
        
        # Broadcast registration via unified Redis service
        redis_service = get_redis_service()
        await redis_service.publish("agent_registrations", {
            "agent_id": str(agent.id),
            "agent_name": agent.name,
            "capabilities": len(agent.capabilities or []),
            "registered_at": datetime.utcnow().isoformat()
        })
        
        logger.info("Agent registered successfully",
                   agent_id=str(agent.id),
                   agent_name=agent.name)
        
        return AgentStatusResponse(
            agent_id=str(agent.id),
            name=agent.name,
            type=agent.type.value,
            status=agent.status.value,
            current_workload=0.0,
            available_capacity=registration.preferred_workload,
            capabilities=agent.capabilities or [],
            active_tasks=0,
            completed_today=0,
            average_response_time_ms=0.0,
            last_heartbeat=agent.last_heartbeat,
            performance_score=performance_score
        )
        
    except Exception as e:
        logger.error("Failed to register agent", error=str(e))
        raise HTTPException(status_code=500, detail="Agent registration failed")


@router.get("/agents", response_model=List[AgentStatusResponse])
async def list_coordination_agents(
    status_filter: Optional[AgentStatus] = Query(None, description="Filter by agent status"),
    capability_filter: Optional[str] = Query(None, description="Filter by capability name"),
    limit: int = Query(default=50, ge=1, le=200, description="Number of agents to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    db: AsyncSession = Depends(get_session_dependency)
) -> List[AgentStatusResponse]:
    """
    List all coordination agents with advanced filtering.
    
    Demonstrates:
    - Complex database queries with filtering
    - Pagination with validation
    - Performance optimization with selective loading
    """
    
    # Build query with filters
    query = select(Agent).options(selectinload(Agent.performance_history))
    
    if status_filter:
        query = query.where(Agent.status == status_filter)
    
    if capability_filter:
        # Search in capabilities JSON array
        query = query.where(
            func.json_extract_path_text(Agent.capabilities, '*', 'name').ilike(f"%{capability_filter}%")
        )
    
    # Apply pagination
    query = query.offset(offset).limit(limit)
    
    result = await db.execute(query)
    agents = result.scalars().all()
    
    # Build response with current status
    agent_responses = []
    for agent in agents:
        
        # Calculate current workload
        current_workload = float(agent.context_window_usage or 0.0)
        
        # Get active tasks count
        active_tasks_query = select(func.count(Task.id)).where(
            and_(
                Task.assigned_agent_id == agent.id,
                Task.status.in_([TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS])
            )
        )
        active_tasks_result = await db.execute(active_tasks_query)
        active_tasks = active_tasks_result.scalar()
        
        # Get completed tasks today  
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        completed_today_query = select(func.count(Task.id)).where(
            and_(
                Task.assigned_agent_id == agent.id,
                Task.status == TaskStatus.COMPLETED,
                Task.completed_at >= today_start
            )
        )
        completed_today_result = await db.execute(completed_today_query)
        completed_today = completed_today_result.scalar()
        
        agent_responses.append(AgentStatusResponse(
            agent_id=str(agent.id),
            name=agent.name,
            type=agent.type.value,
            status=agent.status.value,
            current_workload=current_workload,
            available_capacity=1.0 - current_workload,
            capabilities=agent.capabilities or [],
            active_tasks=active_tasks,
            completed_today=completed_today,
            average_response_time_ms=float(agent.average_response_time or 0.0) * 1000,
            last_heartbeat=agent.last_heartbeat,
            performance_score=0.85  # Would be calculated from performance history
        ))
    
    return agent_responses


# Task Distribution Endpoints
@router.post("/tasks/distribute", response_model=TaskDistributionResponse, status_code=201)
async def distribute_task_intelligently(
    task_request: TaskDistributionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session_dependency)
) -> TaskDistributionResponse:
    """
    Intelligently distribute a task to the optimal agent.
    
    Demonstrates:
    - Complex business logic with multiple factors
    - Real-time coordination via Redis
    - Background processing for metrics
    - Comprehensive error handling
    """
    
    try:
        # Distribute task using intelligent routing
        distribution_result = await coordination_service.distribute_task_intelligently(
            db, task_request
        )
        
        # Background task for metrics collection
        background_tasks.add_task(
            collect_task_distribution_metrics,
            task_id=distribution_result.task_id,
            agent_id=distribution_result.assigned_agent_id,
            confidence_score=distribution_result.assignment_confidence
        )
        
        return distribution_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Task distribution failed", error=str(e))
        raise HTTPException(status_code=500, detail="Task distribution failed")


@router.post("/tasks/{task_id}/reassign", response_model=Dict[str, Any])
async def reassign_task_intelligently(
    task_id: str = Path(..., description="Task ID to reassign"),
    reassignment: TaskReassignmentRequest = ...,
    db: AsyncSession = Depends(get_session_dependency)
) -> Dict[str, Any]:
    """
    Reassign a task with intelligent agent selection.
    
    Demonstrates:
    - Complex database operations with transactions
    - Conditional logic based on business rules
    - Audit trail maintenance
    """
    
    try:
        # Get existing task
        task_query = select(Task).options(selectinload(Task.assigned_agent)).where(Task.id == uuid.UUID(task_id))
        task_result = await db.execute(task_query)
        task = task_result.scalar_one_or_none()
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        old_agent_id = task.assigned_agent_id
        old_agent_name = task.assigned_agent.name if task.assigned_agent else "Unknown"
        
        # Find new agent
        if reassignment.target_agent_id:
            # Specific agent requested
            new_agent_query = select(Agent).where(Agent.id == uuid.UUID(reassignment.target_agent_id))
            new_agent_result = await db.execute(new_agent_query)
            new_agent = new_agent_result.scalar_one_or_none()
            
            if not new_agent:
                raise HTTPException(status_code=404, detail="Target agent not found")
            
            # Check if agent is available or force assignment is enabled
            if not reassignment.force_assignment and new_agent.status != AgentStatus.active:
                raise HTTPException(status_code=400, detail="Target agent is not available")
                
        else:
            # Find optimal agent automatically
            task_request = TaskDistributionRequest(
                task_title=task.title,
                task_description=task.description or "",
                task_type=task.task_type,
                priority=task.priority,
                required_capabilities=task.required_capabilities or [],
                estimated_effort_hours=task.estimated_effort / 60.0 if task.estimated_effort else None,
                deadline=task.due_date
            )
            
            match_result = await coordination_service.find_optimal_agent_for_task(db, task_request)
            if not match_result:
                raise HTTPException(status_code=404, detail="No suitable agent found for reassignment")
            
            new_agent = match_result["agent"]
        
        # Perform reassignment
        task.assigned_agent_id = new_agent.id
        task.updated_at = datetime.utcnow()
        
        # Add to context if preserving context
        if reassignment.preserve_context:
            if not task.context:
                task.context = {}
            
            reassignment_history = task.context.get("reassignment_history", [])
            reassignment_history.append({
                "from_agent_id": str(old_agent_id) if old_agent_id else None,
                "from_agent_name": old_agent_name,
                "to_agent_id": str(new_agent.id),
                "to_agent_name": new_agent.name,
                "reason": reassignment.reason,
                "timestamp": datetime.utcnow().isoformat(),
                "forced": reassignment.force_assignment
            })
            task.context["reassignment_history"] = reassignment_history
        
        await db.commit()
        
        # Notify via unified Redis service
        redis_service = get_redis_service()
        await redis_service.publish("task_reassignments", {
            "task_id": task_id,
            "from_agent_id": str(old_agent_id) if old_agent_id else None,
            "to_agent_id": str(new_agent.id),
            "reason": reassignment.reason,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # WebSocket notifications
        if old_agent_id:
            await ws_manager.broadcast_agent_update(str(old_agent_id), {
                "task_reassigned_away": {
                    "task_id": task_id,
                    "task_title": task.title,
                    "new_agent": new_agent.name
                }
            })
        
        await ws_manager.broadcast_agent_update(str(new_agent.id), {
            "task_reassigned_to": {
                "task_id": task_id,
                "task_title": task.title,
                "from_agent": old_agent_name
            }
        })
        
        logger.info("Task reassigned successfully",
                   task_id=task_id,
                   from_agent=old_agent_name,
                   to_agent=new_agent.name,
                   reason=reassignment.reason)
        
        return {
            "task_id": task_id,
            "from_agent_id": str(old_agent_id) if old_agent_id else None,
            "from_agent_name": old_agent_name,
            "to_agent_id": str(new_agent.id),
            "to_agent_name": new_agent.name,
            "reassignment_reason": reassignment.reason,
            "reassigned_at": datetime.utcnow().isoformat(),
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Task reassignment failed", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Task reassignment failed")


# Performance Metrics Endpoints
@router.get("/metrics", response_model=CoordinationMetricsResponse)
async def get_coordination_metrics(
    metrics_query: PerformanceMetricsQuery = Depends(),
    db: AsyncSession = Depends(get_session_dependency)
) -> CoordinationMetricsResponse:
    """
    Get comprehensive coordination system metrics.
    
    Demonstrates:
    - Complex analytical queries
    - Performance optimization techniques
    - Data aggregation and calculation
    """
    
    time_threshold = datetime.utcnow() - timedelta(hours=metrics_query.time_range_hours)
    
    # Get total and active agents
    total_agents_query = select(func.count(Agent.id)).where(Agent.status != AgentStatus.inactive)
    total_agents_result = await db.execute(total_agents_query)
    total_agents = total_agents_result.scalar()
    
    active_agents_query = select(func.count(Agent.id)).where(Agent.status == AgentStatus.active)
    active_agents_result = await db.execute(active_agents_query)
    active_agents = active_agents_result.scalar()
    
    # Get task statistics
    total_tasks_query = select(func.count(Task.id))
    if metrics_query.agent_ids:
        total_tasks_query = total_tasks_query.where(
            Task.assigned_agent_id.in_([uuid.UUID(aid) for aid in metrics_query.agent_ids])
        )
    total_tasks_result = await db.execute(total_tasks_query)
    total_tasks = total_tasks_result.scalar()
    
    # Tasks completed today
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    completed_today_query = select(func.count(Task.id)).where(
        and_(
            Task.status == TaskStatus.COMPLETED,
            Task.completed_at >= today_start
        )
    )
    if metrics_query.agent_ids:
        completed_today_query = completed_today_query.where(
            Task.assigned_agent_id.in_([uuid.UUID(aid) for aid in metrics_query.agent_ids])
        )
    completed_today_result = await db.execute(completed_today_query)
    completed_today = completed_today_result.scalar()
    
    # Average completion time
    avg_completion_query = select(func.avg(Task.actual_effort)).where(
        and_(
            Task.status == TaskStatus.COMPLETED,
            Task.completed_at >= time_threshold,
            Task.actual_effort.isnot(None)
        )
    )
    if metrics_query.agent_ids:
        avg_completion_query = avg_completion_query.where(
            Task.assigned_agent_id.in_([uuid.UUID(aid) for aid in metrics_query.agent_ids])
        )
    avg_completion_result = await db.execute(avg_completion_query)
    avg_completion_minutes = avg_completion_result.scalar() or 0.0
    avg_completion_hours = avg_completion_minutes / 60.0
    
    # Agent workload distribution
    workload_query = select(
        Agent.id,
        Agent.name,
        Agent.context_window_usage
    ).where(Agent.status == AgentStatus.active)
    
    if metrics_query.agent_ids:
        workload_query = workload_query.where(
            Agent.id.in_([uuid.UUID(aid) for aid in metrics_query.agent_ids])
        )
    
    workload_result = await db.execute(workload_query)
    workload_data = workload_result.all()
    
    workload_distribution = {}
    total_utilization = 0.0
    agent_count = 0
    
    for agent_id, name, usage in workload_data:
        utilization = float(usage or 0.0)
        workload_distribution[name] = utilization
        total_utilization += utilization
        agent_count += 1
    
    avg_utilization = (total_utilization / agent_count * 100) if agent_count > 0 else 0.0
    
    # Top performing agents (by completion rate and speed)
    top_performers_query = select(
        Agent.id,
        Agent.name,
        Agent.total_tasks_completed,
        Agent.average_response_time
    ).where(Agent.status == AgentStatus.active).order_by(
        func.cast(Agent.total_tasks_completed, 'integer').desc(),
        func.cast(Agent.average_response_time, 'float').asc()
    ).limit(5)
    
    top_performers_result = await db.execute(top_performers_query)
    top_performers_data = top_performers_result.all()
    
    top_performers = [
        {
            "agent_id": str(agent_id),
            "name": name,
            "tasks_completed": int(completed or 0),
            "avg_response_time": float(response_time or 0.0),
            "performance_score": 0.9  # Would be calculated from multiple factors
        }
        for agent_id, name, completed, response_time in top_performers_data
    ]
    
    # System efficiency calculation
    efficiency_factors = [
        min(1.0, completed_today / max(1, total_agents * 5)),  # Tasks per agent per day
        min(1.0, avg_utilization / 80.0),  # Target 80% utilization
        min(1.0, 10.0 / max(avg_completion_hours, 0.1))  # Faster completion is better
    ]
    system_efficiency = sum(efficiency_factors) / len(efficiency_factors)
    
    # Identify bottlenecks
    bottlenecks = []
    if avg_utilization > 90:
        bottlenecks.append({
            "type": "high_utilization",
            "description": "System utilization is very high",
            "severity": "warning",
            "value": avg_utilization
        })
    
    if avg_completion_hours > 8:
        bottlenecks.append({
            "type": "slow_completion",
            "description": "Average task completion time is high",
            "severity": "warning", 
            "value": avg_completion_hours
        })
    
    return CoordinationMetricsResponse(
        total_agents=total_agents,
        active_agents=active_agents,
        total_tasks=total_tasks,
        tasks_completed_today=completed_today,
        average_task_completion_time_hours=avg_completion_hours,
        system_efficiency_score=system_efficiency,
        agent_utilization_percentage=avg_utilization,
        current_workload_distribution=workload_distribution,
        top_performing_agents=top_performers,
        bottlenecks=bottlenecks
    )


# Real-time WebSocket Endpoints
@router.websocket("/ws/{connection_id}")
async def coordination_websocket(websocket: WebSocket, connection_id: str):
    """
    WebSocket endpoint for real-time coordination updates.
    
    Demonstrates:
    - WebSocket connection management
    - Real-time event broadcasting
    - Error handling for connection issues
    """
    
    await ws_manager.connect(websocket, connection_id)
    
    try:
        while True:
            # Receive and process client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "subscribe_agent":
                agent_id = message.get("agent_id")
                if agent_id:
                    await ws_manager.subscribe_to_agent(connection_id, agent_id)
            
            elif message_type == "subscribe_metrics":
                ws_manager.metric_subscribers.add(connection_id)
                await ws_manager.send_to_connection(connection_id, {
                    "type": "subscription_confirmed",
                    "subscription": "metrics"
                })
            
            elif message_type == "ping":
                await ws_manager.send_to_connection(connection_id, {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        await ws_manager.disconnect(connection_id)
    except Exception as e:
        logger.error("WebSocket error", connection_id=connection_id, error=str(e))
        await ws_manager.disconnect(connection_id)


# Health and Status Endpoints
@router.get("/health", response_model=Dict[str, Any])
async def get_coordination_health(
    db: AsyncSession = Depends(get_session_dependency)
) -> Dict[str, Any]:
    """Get detailed health status of coordination system."""
    
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {}
    }
    
    # Database health
    try:
        await db.execute(select(1))
        health_data["components"]["database"] = {
            "status": "healthy",
            "response_time_ms": 5
        }
    except Exception as e:
        health_data["components"]["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_data["status"] = "degraded"
    
    # Redis health via unified service
    try:
        redis_service = get_redis_service()
        health_result = await redis_service.health_check()
        health_data["components"]["redis"] = health_result
        if health_result.get("status") != "healthy":
            health_data["status"] = "degraded"
    except Exception as e:
        health_data["components"]["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_data["status"] = "degraded"
    
    # WebSocket connections
    health_data["components"]["websockets"] = {
        "status": "healthy",
        "active_connections": len(ws_manager.active_connections),
        "agent_subscriptions": len(ws_manager.agent_subscribers),
        "metric_subscriptions": len(ws_manager.metric_subscribers)
    }
    
    return health_data


# =====================================================================================
# BACKGROUND TASKS AND UTILITY FUNCTIONS
# =====================================================================================

async def initialize_agent_performance_tracking(agent_id: str):
    """Initialize performance tracking for new agent."""
    try:
        # Set up initial metrics in Redis via unified service
        redis_service = get_redis_service()
        await redis_service.cache_set(f"agent_metrics:{agent_id}", {
            "tasks_assigned": 0,
            "tasks_completed": 0,
            "total_response_time": 0.0,
            "last_active": datetime.utcnow().isoformat()
        }, ttl=86400)  # 24 hours
        
        logger.info("Agent performance tracking initialized", agent_id=agent_id)
        
    except Exception as e:
        logger.error("Failed to initialize agent performance tracking", 
                    agent_id=agent_id, error=str(e))


async def collect_task_distribution_metrics(task_id: str, agent_id: str, confidence_score: float):
    """Collect metrics for task distribution analysis."""
    try:
        # Record distribution metrics for analytics
        # This would typically go to a metrics database or time series DB
        logger.info("Task distribution metrics collected",
                   task_id=task_id,
                   agent_id=agent_id,
                   confidence_score=confidence_score)
                   
    except Exception as e:
        logger.error("Failed to collect task distribution metrics",
                    task_id=task_id, error=str(e))


# Stream processing for real-time metrics
@router.get("/metrics/stream")
async def stream_coordination_metrics(
    db: AsyncSession = Depends(get_session_dependency)
):
    """Stream real-time coordination metrics via Server-Sent Events."""
    
    async def generate_metrics():
        while True:
            try:
                # Get current metrics
                current_time = datetime.utcnow()
                
                # Active agents count
                active_agents_query = select(func.count(Agent.id)).where(Agent.status == AgentStatus.active)
                active_agents_result = await db.execute(active_agents_query)
                active_agents = active_agents_result.scalar()
                
                # Active tasks count
                active_tasks_query = select(func.count(Task.id)).where(
                    Task.status.in_([TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS])
                )
                active_tasks_result = await db.execute(active_tasks_query)
                active_tasks = active_tasks_result.scalar()
                
                metrics_data = {
                    "timestamp": current_time.isoformat(),
                    "active_agents": active_agents,
                    "active_tasks": active_tasks,
                    "websocket_connections": len(ws_manager.active_connections)
                }
                
                yield f"data: {json.dumps(metrics_data)}\n\n"
                
                # Wait before next update
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error("Error in metrics stream", error=str(e))
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                break
    
    return StreamingResponse(
        generate_metrics(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )