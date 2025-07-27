"""
Multi-Agent Coordination API endpoints for LeanVibe Agent Hive 2.0

Provides HTTP endpoints for managing coordinated projects, agent assignments,
conflict resolution, and real-time project monitoring across multiple agents.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

import structlog
import json
import asyncio

from ...core.coordination import coordination_engine, CoordinationMode, ProjectStatus
from ...core.database import get_session_dependency
from ...models.agent import Agent

logger = structlog.get_logger()
router = APIRouter()


# Request Models
class ProjectCreateRequest(BaseModel):
    """Request to create a coordinated project."""
    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    requirements: Dict[str, Any] = Field(..., description="Project requirements")
    coordination_mode: CoordinationMode = Field(default=CoordinationMode.PARALLEL, description="Coordination mode")
    deadline: Optional[str] = Field(None, description="Project deadline (ISO format)")
    quality_gates: Optional[List[Dict[str, Any]]] = Field(None, description="Custom quality gates")


class AgentRegistrationRequest(BaseModel):
    """Request to register an agent for coordination."""
    agent_id: str = Field(..., description="Agent ID")
    capabilities: List[str] = Field(..., description="Agent capabilities")
    specializations: List[str] = Field(..., description="Agent specializations")
    proficiency: float = Field(default=0.8, ge=0.0, le=1.0, description="Proficiency level")
    experience_level: str = Field(default="intermediate", description="Experience level")


class TaskReassignmentRequest(BaseModel):
    """Request to reassign a task to different agent."""
    project_id: str = Field(..., description="Project ID")
    task_id: str = Field(..., description="Task ID")
    new_agent_id: str = Field(..., description="New agent ID")
    reason: str = Field(..., description="Reason for reassignment")


class ConflictResolutionRequest(BaseModel):
    """Request to manually resolve a conflict."""
    conflict_id: str = Field(..., description="Conflict ID")
    resolution_strategy: str = Field(..., description="Resolution strategy")
    resolution_data: Optional[Dict[str, Any]] = Field(None, description="Additional resolution data")


# Response Models
class ProjectResponse(BaseModel):
    """Response with project information."""
    project_id: str
    name: str
    description: str
    status: str
    coordination_mode: str
    participating_agents: List[str]
    progress_percentage: float
    created_at: str
    started_at: Optional[str]
    estimated_completion: Optional[str]


class ProjectStatusResponse(BaseModel):
    """Detailed project status response."""
    project_id: str
    name: str
    status: str
    progress_metrics: Dict[str, Any]
    quality_gates: List[Dict[str, Any]]
    participating_agents: List[str]
    active_conflicts: List[str]
    tasks_summary: Dict[str, int]
    agent_utilization: float


class ConflictResponse(BaseModel):
    """Response with conflict information."""
    conflict_id: str
    project_id: str
    conflict_type: str
    severity: str
    description: str
    affected_agents: List[str]
    resolution_status: str
    detected_at: str


# WebSocket Connection Manager
class ConnectionManager:
    """Manages WebSocket connections for real-time project updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.project_subscribers: Dict[str, List[str]] = {}  # project_id -> [connection_ids]
    
    async def connect(self, websocket: WebSocket, connection_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        logger.info("WebSocket connection established", connection_id=connection_id)
    
    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        
        # Remove from project subscriptions
        for project_id, subscribers in self.project_subscribers.items():
            if connection_id in subscribers:
                subscribers.remove(connection_id)
        
        logger.info("WebSocket connection closed", connection_id=connection_id)
    
    async def subscribe_to_project(self, connection_id: str, project_id: str):
        """Subscribe a connection to project updates."""
        if project_id not in self.project_subscribers:
            self.project_subscribers[project_id] = []
        
        if connection_id not in self.project_subscribers[project_id]:
            self.project_subscribers[project_id].append(connection_id)
        
        logger.info("Subscribed to project updates", connection_id=connection_id, project_id=project_id)
    
    async def broadcast_project_update(self, project_id: str, update: Dict[str, Any]):
        """Broadcast an update to all subscribers of a project."""
        if project_id in self.project_subscribers:
            disconnected = []
            
            for connection_id in self.project_subscribers[project_id]:
                if connection_id in self.active_connections:
                    try:
                        websocket = self.active_connections[connection_id]
                        await websocket.send_text(json.dumps(update))
                    except:
                        disconnected.append(connection_id)
                else:
                    disconnected.append(connection_id)
            
            # Clean up disconnected connections
            for connection_id in disconnected:
                if connection_id in self.project_subscribers[project_id]:
                    self.project_subscribers[project_id].remove(connection_id)


# Global connection manager
connection_manager = ConnectionManager()


# Project Management Endpoints
@router.post("/projects", response_model=ProjectResponse, status_code=201)
async def create_coordinated_project(
    request: ProjectCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session_dependency)
) -> ProjectResponse:
    """Create a new coordinated project with multiple agents."""
    
    try:
        # Parse deadline if provided
        deadline = None
        if request.deadline:
            deadline = datetime.fromisoformat(request.deadline)
        
        # Create coordinated project
        project_id = await coordination_engine.create_coordinated_project(
            name=request.name,
            description=request.description,
            requirements=request.requirements,
            coordination_mode=request.coordination_mode,
            deadline=deadline
        )
        
        # Get project status for response
        project_status = await coordination_engine.get_project_status(project_id)
        
        if not project_status:
            raise HTTPException(status_code=500, detail="Failed to retrieve project status")
        
        logger.info(
            "Coordinated project created via API",
            project_id=project_id,
            project_name=request.name,
            coordination_mode=request.coordination_mode.value
        )
        
        return ProjectResponse(
            project_id=project_id,
            name=project_status["name"],
            description=project_status["description"],
            status=project_status["status"],
            coordination_mode=project_status["coordination_mode"],
            participating_agents=project_status["participating_agents"],
            progress_percentage=project_status["progress_metrics"].get("progress_percentage", 0),
            created_at=project_status["created_at"],
            started_at=project_status["started_at"],
            estimated_completion=project_status["progress_metrics"].get("estimated_completion")
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to create coordinated project", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/projects/{project_id}/start")
async def start_project(project_id: str) -> Dict[str, str]:
    """Start execution of a coordinated project."""
    
    try:
        success = await coordination_engine.start_project(project_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Project cannot be started")
        
        # Broadcast project start to WebSocket subscribers
        await connection_manager.broadcast_project_update(
            project_id,
            {
                "type": "project_started",
                "project_id": project_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return {"project_id": project_id, "status": "started"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to start project", project_id=project_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/projects/{project_id}", response_model=ProjectStatusResponse)
async def get_project_status(project_id: str) -> ProjectStatusResponse:
    """Get detailed status of a coordinated project."""
    
    try:
        project_status = await coordination_engine.get_project_status(project_id)
        
        if not project_status:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Calculate tasks summary
        tasks = project_status.get("tasks", {})
        tasks_summary = {
            "total": len(tasks),
            "completed": len([t for t in tasks.values() if t["status"] == "completed"]),
            "in_progress": len([t for t in tasks.values() if t["status"] == "in_progress"]),
            "pending": len([t for t in tasks.values() if t["status"] == "pending"])
        }
        
        return ProjectStatusResponse(
            project_id=project_status["project_id"],
            name=project_status["name"],
            status=project_status["status"],
            progress_metrics=project_status["progress_metrics"],
            quality_gates=project_status["quality_gates"],
            participating_agents=project_status["participating_agents"],
            active_conflicts=project_status["active_conflicts"],
            tasks_summary=tasks_summary,
            agent_utilization=project_status["progress_metrics"].get("agent_utilization", 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get project status", project_id=project_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/projects")
async def list_projects() -> Dict[str, Any]:
    """List all coordinated projects with summary information."""
    
    try:
        projects = []
        
        for project_id in coordination_engine.active_projects.keys():
            project_status = await coordination_engine.get_project_status(project_id)
            if project_status:
                projects.append({
                    "project_id": project_status["project_id"],
                    "name": project_status["name"],
                    "status": project_status["status"],
                    "progress_percentage": project_status["progress_metrics"].get("progress_percentage", 0),
                    "participating_agents": len(project_status["participating_agents"]),
                    "created_at": project_status["created_at"],
                    "active_conflicts": len(project_status["active_conflicts"])
                })
        
        return {
            "projects": projects,
            "total_projects": len(projects),
            "coordination_metrics": coordination_engine.coordination_metrics
        }
        
    except Exception as e:
        logger.error("Failed to list projects", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# Agent Management Endpoints
@router.post("/agents/register")
async def register_agent_for_coordination(
    request: AgentRegistrationRequest,
    db: AsyncSession = Depends(get_session_dependency)
) -> Dict[str, str]:
    """Register an agent for coordination with capabilities."""
    
    try:
        # Verify agent exists in database
        agent = await db.get(Agent, request.agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Register agent in coordination engine
        await coordination_engine.agent_registry.register_agent(
            agent_id=request.agent_id,
            capabilities=request.capabilities,
            specializations=request.specializations,
            proficiency=request.proficiency,
            experience_level=request.experience_level
        )
        
        logger.info(
            "Agent registered for coordination",
            agent_id=request.agent_id,
            specializations=request.specializations
        )
        
        return {
            "agent_id": request.agent_id,
            "status": "registered",
            "capabilities": len(request.capabilities),
            "specializations": len(request.specializations)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to register agent", agent_id=request.agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/agents")
async def list_registered_agents() -> Dict[str, Any]:
    """List all agents registered for coordination."""
    
    try:
        registry = coordination_engine.agent_registry
        
        agents = []
        for agent_id, capability in registry.agents.items():
            agents.append({
                "agent_id": agent_id,
                "specializations": capability.specializations,
                "proficiency": capability.proficiency,
                "experience_level": capability.experience_level,
                "status": registry.agent_status.get(agent_id, "unknown"),
                "current_tasks": len(registry.agent_assignments.get(agent_id, [])),
                "performance_metrics": capability.performance_metrics
            })
        
        return {
            "agents": agents,
            "total_agents": len(agents),
            "available_agents": len([a for a in agents if a["status"] == "available"]),
            "busy_agents": len([a for a in agents if a["status"] in ["busy", "active"]])
        }
        
    except Exception as e:
        logger.error("Failed to list agents", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/agents/{agent_id}/assignments")
async def get_agent_assignments(agent_id: str) -> Dict[str, Any]:
    """Get current task assignments for an agent."""
    
    try:
        registry = coordination_engine.agent_registry
        
        if agent_id not in registry.agents:
            raise HTTPException(status_code=404, detail="Agent not registered for coordination")
        
        assignments = registry.agent_assignments.get(agent_id, [])
        capability = registry.agents[agent_id]
        
        # Get detailed task information
        task_details = []
        for project_id, project in coordination_engine.active_projects.items():
            for task_id, task in project.tasks.items():
                if task.assigned_agent_id == agent_id:
                    task_details.append({
                        "task_id": task_id,
                        "project_id": project_id,
                        "project_name": project.name,
                        "task_title": task.title,
                        "task_status": task.status.value,
                        "estimated_effort": task.estimated_effort,
                        "required_capabilities": task.required_capabilities
                    })
        
        return {
            "agent_id": agent_id,
            "status": registry.agent_status.get(agent_id, "unknown"),
            "current_assignments": len(assignments),
            "task_details": task_details,
            "capability_profile": {
                "specializations": capability.specializations,
                "proficiency": capability.proficiency,
                "experience_level": capability.experience_level,
                "performance_metrics": capability.performance_metrics
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get agent assignments", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# Task Management Endpoints
@router.post("/tasks/reassign")
async def reassign_task(request: TaskReassignmentRequest) -> Dict[str, str]:
    """Reassign a task from one agent to another."""
    
    try:
        project_id = request.project_id
        task_id = request.task_id
        new_agent_id = request.new_agent_id
        
        # Verify project and task exist
        if project_id not in coordination_engine.active_projects:
            raise HTTPException(status_code=404, detail="Project not found")
        
        project = coordination_engine.active_projects[project_id]
        
        if task_id not in project.tasks:
            raise HTTPException(status_code=404, detail="Task not found")
        
        task = project.tasks[task_id]
        old_agent_id = task.assigned_agent_id
        
        # Verify new agent is registered
        registry = coordination_engine.agent_registry
        if new_agent_id not in registry.agents:
            raise HTTPException(status_code=404, detail="New agent not registered for coordination")
        
        # Perform reassignment
        if old_agent_id:
            # Remove from old agent
            if old_agent_id in registry.agent_assignments:
                if task_id in registry.agent_assignments[old_agent_id]:
                    registry.agent_assignments[old_agent_id].remove(task_id)
        
        # Assign to new agent
        registry.assign_task(new_agent_id, task_id)
        task.assigned_agent_id = new_agent_id
        
        # Update task context with reassignment reason
        task.context = task.context or {}
        task.context["reassignment_history"] = task.context.get("reassignment_history", [])
        task.context["reassignment_history"].append({
            "from_agent": old_agent_id,
            "to_agent": new_agent_id,
            "reason": request.reason,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(
            "Task reassigned successfully",
            project_id=project_id,
            task_id=task_id,
            from_agent=old_agent_id,
            to_agent=new_agent_id,
            reason=request.reason
        )
        
        return {
            "task_id": task_id,
            "from_agent": old_agent_id,
            "to_agent": new_agent_id,
            "status": "reassigned"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to reassign task", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# Conflict Management Endpoints
@router.get("/conflicts")
async def list_active_conflicts() -> Dict[str, Any]:
    """List all active conflicts across projects."""
    
    try:
        conflicts = []
        
        for conflict_id, conflict in coordination_engine.conflict_resolver.active_conflicts.items():
            if not conflict.resolved:
                conflicts.append({
                    "conflict_id": conflict.id,
                    "project_id": conflict.project_id,
                    "conflict_type": conflict.conflict_type.value,
                    "severity": conflict.severity,
                    "description": conflict.description,
                    "affected_agents": conflict.affected_agents,
                    "detected_at": conflict.detected_at.isoformat(),
                    "impact_score": conflict.impact_score
                })
        
        # Sort by severity and impact score
        conflicts.sort(key=lambda x: (
            {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(x["severity"], 0),
            x["impact_score"]
        ), reverse=True)
        
        return {
            "active_conflicts": conflicts,
            "total_conflicts": len(conflicts),
            "critical_conflicts": len([c for c in conflicts if c["severity"] == "critical"]),
            "high_priority_conflicts": len([c for c in conflicts if c["severity"] == "high"])
        }
        
    except Exception as e:
        logger.error("Failed to list conflicts", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/conflicts/{conflict_id}")
async def get_conflict_details(conflict_id: str) -> ConflictResponse:
    """Get detailed information about a specific conflict."""
    
    try:
        conflict_resolver = coordination_engine.conflict_resolver
        
        if conflict_id not in conflict_resolver.active_conflicts:
            raise HTTPException(status_code=404, detail="Conflict not found")
        
        conflict = conflict_resolver.active_conflicts[conflict_id]
        
        return ConflictResponse(
            conflict_id=conflict.id,
            project_id=conflict.project_id,
            conflict_type=conflict.conflict_type.value,
            severity=conflict.severity,
            description=conflict.description,
            affected_agents=conflict.affected_agents,
            resolution_status="resolved" if conflict.resolved else "pending",
            detected_at=conflict.detected_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get conflict details", conflict_id=conflict_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/conflicts/{conflict_id}/resolve")
async def resolve_conflict_manually(
    conflict_id: str,
    request: ConflictResolutionRequest
) -> Dict[str, Any]:
    """Manually resolve a conflict with specified strategy."""
    
    try:
        conflict_resolver = coordination_engine.conflict_resolver
        
        if conflict_id not in conflict_resolver.active_conflicts:
            raise HTTPException(status_code=404, detail="Conflict not found")
        
        conflict = conflict_resolver.active_conflicts[conflict_id]
        
        if conflict.resolved:
            raise HTTPException(status_code=400, detail="Conflict already resolved")
        
        # Get associated project
        project = coordination_engine.active_projects.get(conflict.project_id)
        if not project:
            raise HTTPException(status_code=404, detail="Associated project not found")
        
        # Apply manual resolution
        success, result = await conflict_resolver._apply_resolution_strategy(
            conflict, project, request.resolution_strategy
        )
        
        if success:
            conflict.resolved = True
            conflict.resolved_at = datetime.utcnow()
            conflict.resolution_strategy = request.resolution_strategy
            conflict.resolution_result = result
            
            logger.info(
                "Conflict resolved manually",
                conflict_id=conflict_id,
                strategy=request.resolution_strategy,
                result=result
            )
            
            return {
                "conflict_id": conflict_id,
                "status": "resolved",
                "strategy": request.resolution_strategy,
                "result": result
            }
        else:
            return {
                "conflict_id": conflict_id,
                "status": "resolution_failed",
                "strategy": request.resolution_strategy,
                "error": result.get("error", "Unknown error")
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resolve conflict", conflict_id=conflict_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# Real-time WebSocket Endpoints
@router.websocket("/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str):
    """WebSocket endpoint for real-time project updates."""
    
    await connection_manager.connect(websocket, connection_id)
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "subscribe_project":
                project_id = message.get("project_id")
                if project_id:
                    await connection_manager.subscribe_to_project(connection_id, project_id)
                    await websocket.send_text(json.dumps({
                        "type": "subscription_confirmed",
                        "project_id": project_id
                    }))
            
            elif message_type == "get_project_status":
                project_id = message.get("project_id")
                if project_id:
                    status = await coordination_engine.get_project_status(project_id)
                    if status:
                        await websocket.send_text(json.dumps({
                            "type": "project_status",
                            "data": status
                        }))
            
            elif message_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        connection_manager.disconnect(connection_id)
    except Exception as e:
        logger.error("WebSocket error", connection_id=connection_id, error=str(e))
        connection_manager.disconnect(connection_id)


# Analytics and Metrics Endpoints
@router.get("/metrics/coordination")
async def get_coordination_metrics() -> Dict[str, Any]:
    """Get coordination engine performance metrics."""
    
    try:
        metrics = coordination_engine.coordination_metrics.copy()
        
        # Add real-time statistics
        active_projects = len(coordination_engine.active_projects)
        total_agents = len(coordination_engine.agent_registry.agents)
        active_conflicts = len([
            c for c in coordination_engine.conflict_resolver.active_conflicts.values()
            if not c.resolved
        ])
        
        # Calculate utilization rates
        registry = coordination_engine.agent_registry
        busy_agents = len([
            agent_id for agent_id, status in registry.agent_status.items()
            if status in ["busy", "active"]
        ])
        
        agent_utilization = (busy_agents / total_agents * 100) if total_agents > 0 else 0
        
        return {
            "coordination_metrics": metrics,
            "real_time_stats": {
                "active_projects": active_projects,
                "total_agents": total_agents,
                "busy_agents": busy_agents,
                "agent_utilization_percentage": agent_utilization,
                "active_conflicts": active_conflicts,
                "websocket_connections": len(connection_manager.active_connections)
            },
            "performance_indicators": {
                "average_project_duration_hours": metrics.get("average_project_duration", 0),
                "conflict_resolution_rate": (
                    metrics.get("conflicts_resolved", 0) / 
                    max(1, metrics.get("conflicts_resolved", 0) + active_conflicts) * 100
                ),
                "project_success_rate": 95.0,  # Would be calculated from historical data
                "agent_satisfaction_score": 0.92  # Would be calculated from agent feedback
            }
        }
        
    except Exception as e:
        logger.error("Failed to get coordination metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def get_coordination_health() -> Dict[str, Any]:
    """Get coordination engine health status."""
    
    try:
        # Check coordination engine status
        engine_healthy = coordination_engine is not None
        
        # Check active projects
        active_projects = len(coordination_engine.active_projects)
        
        # Check agent registry
        registered_agents = len(coordination_engine.agent_registry.agents)
        
        # Check conflict resolver
        active_conflicts = len([
            c for c in coordination_engine.conflict_resolver.active_conflicts.values()
            if not c.resolved
        ])
        
        # Overall health score
        health_score = 1.0
        if active_conflicts > 10:
            health_score -= 0.2
        if active_projects > 50:
            health_score -= 0.1
        
        health_status = "healthy" if health_score > 0.8 else "degraded" if health_score > 0.5 else "unhealthy"
        
        return {
            "status": health_status,
            "health_score": health_score,
            "coordination_engine": {
                "status": "online" if engine_healthy else "offline",
                "active_projects": active_projects,
                "registered_agents": registered_agents,
                "active_sync_tasks": len(coordination_engine.sync_tasks)
            },
            "conflict_resolution": {
                "status": "operational",
                "active_conflicts": active_conflicts,
                "resolution_rate": coordination_engine.coordination_metrics.get("conflicts_resolved", 0)
            },
            "websocket_connections": {
                "active_connections": len(connection_manager.active_connections),
                "project_subscriptions": sum(len(subs) for subs in connection_manager.project_subscribers.values())
            }
        }
        
    except Exception as e:
        logger.error("Failed to get coordination health", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")