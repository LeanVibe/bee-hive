"""
Agent management API endpoints for LeanVibe Agent Hive 2.0

Provides CRUD operations for managing AI agents in the multi-agent system.
Supports agent spawning, monitoring, configuration, and lifecycle management.
Enhanced with Vertical Slice 1.1 capabilities.
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import selectinload

import structlog

from ...core.database import get_session_dependency
from ...core.orchestrator import AgentOrchestrator, AgentRole, AgentCapability
from ...core.vertical_slice_orchestrator import VerticalSliceOrchestrator
from ...core.agent_lifecycle_manager import AgentLifecycleManager
from ...core.task_execution_engine import TaskExecutionEngine
from ...models.agent import Agent, AgentStatus, AgentType
from ...models.task import Task, TaskStatus, TaskType, TaskPriority
from ...schemas.agent import (
    AgentCreate, AgentUpdate, AgentResponse, AgentListResponse,
    AgentCapabilityCreate, AgentStatsResponse
)

logger = structlog.get_logger()
router = APIRouter()

# Global vertical slice orchestrator (would be dependency-injected in production)
_vertical_slice_orchestrator: Optional[VerticalSliceOrchestrator] = None

def get_vertical_slice_orchestrator() -> VerticalSliceOrchestrator:
    """Get the vertical slice orchestrator instance."""
    global _vertical_slice_orchestrator
    if _vertical_slice_orchestrator is None:
        _vertical_slice_orchestrator = VerticalSliceOrchestrator()
    return _vertical_slice_orchestrator


@router.post("/", response_model=AgentResponse, status_code=201)
async def create_agent(
    agent_data: AgentCreate,
    db: AsyncSession = Depends(get_session_dependency)
) -> AgentResponse:
    """Create a new AI agent."""
    
    try:
        # Create agent in database
        agent = Agent(
            name=agent_data.name,
            type=agent_data.type,
            role=agent_data.role,
            capabilities=agent_data.capabilities or [],
            system_prompt=agent_data.system_prompt,
            config=agent_data.config or {}
        )
        
        db.add(agent)
        await db.commit()
        await db.refresh(agent)
        
        logger.info(
            "Agent created",
            agent_id=str(agent.id),
            name=agent.name,
            role=agent.role
        )
        
        return AgentResponse.from_orm(agent)
        
    except Exception as e:
        logger.error("Failed to create agent", error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create agent")


@router.get("/", response_model=AgentListResponse)
async def list_agents(
    status: Optional[AgentStatus] = Query(None, description="Filter by agent status"),
    role: Optional[str] = Query(None, description="Filter by agent role"),
    limit: int = Query(50, ge=1, le=100, description="Number of agents to return"),
    offset: int = Query(0, ge=0, description="Number of agents to skip"),
    db: AsyncSession = Depends(get_session_dependency)
) -> AgentListResponse:
    """List all agents with optional filtering."""
    
    try:
        # Build query with filters
        query = select(Agent)
        
        if status:
            query = query.where(Agent.status == status)
        if role:
            query = query.where(Agent.role == role)
        
        query = query.offset(offset).limit(limit).order_by(Agent.created_at.desc())
        
        result = await db.execute(query)
        agents = result.scalars().all()
        
        # Get total count for pagination
        count_query = select(Agent)
        if status:
            count_query = count_query.where(Agent.status == status)
        if role:
            count_query = count_query.where(Agent.role == role)
        
        count_result = await db.execute(count_query)
        total = len(count_result.scalars().all())
        
        return AgentListResponse(
            agents=[AgentResponse.from_orm(agent) for agent in agents],
            total=total,
            offset=offset,
            limit=limit
        )
        
    except Exception as e:
        logger.error("Failed to list agents", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve agents")


@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> AgentResponse:
    """Get a specific agent by ID."""
    
    try:
        result = await db.execute(
            select(Agent).where(Agent.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return AgentResponse.from_orm(agent)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get agent", agent_id=str(agent_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve agent")


@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: uuid.UUID,
    agent_data: AgentUpdate,
    db: AsyncSession = Depends(get_session_dependency)
) -> AgentResponse:
    """Update an existing agent."""
    
    try:
        # Check if agent exists
        result = await db.execute(
            select(Agent).where(Agent.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Update agent fields
        update_data = agent_data.dict(exclude_unset=True)
        update_data["updated_at"] = datetime.utcnow()
        
        await db.execute(
            update(Agent).where(Agent.id == agent_id).values(**update_data)
        )
        await db.commit()
        
        # Fetch updated agent
        result = await db.execute(
            select(Agent).where(Agent.id == agent_id)
        )
        updated_agent = result.scalar_one()
        
        logger.info(
            "Agent updated",
            agent_id=str(agent_id),
            updated_fields=list(update_data.keys())
        )
        
        return AgentResponse.from_orm(updated_agent)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update agent", agent_id=str(agent_id), error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update agent")


@router.delete("/{agent_id}", status_code=204)
async def delete_agent(
    agent_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> None:
    """Deactivate an agent (soft delete)."""
    
    try:
        # Check if agent exists
        result = await db.execute(
            select(Agent).where(Agent.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Soft delete by setting status to inactive
        await db.execute(
            update(Agent)
            .where(Agent.id == agent_id)
            .values(status=AgentStatus.INACTIVE, updated_at=datetime.utcnow())
        )
        await db.commit()
        
        logger.info("Agent deactivated", agent_id=str(agent_id))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete agent", agent_id=str(agent_id), error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete agent")


@router.post("/{agent_id}/heartbeat", status_code=200)
async def agent_heartbeat(
    agent_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> dict:
    """Update agent heartbeat timestamp."""
    
    try:
        result = await db.execute(
            select(Agent).where(Agent.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Update heartbeat
        await db.execute(
            update(Agent)
            .where(Agent.id == agent_id)
            .values(
                last_heartbeat=datetime.utcnow(),
                last_active=datetime.utcnow() if agent.status == AgentStatus.ACTIVE else agent.last_active
            )
        )
        await db.commit()
        
        return {"status": "heartbeat_updated", "timestamp": datetime.utcnow().isoformat()}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update heartbeat", agent_id=str(agent_id), error=str(e))
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update heartbeat")


# === VERTICAL SLICE 1.1 LIFECYCLE ENDPOINTS ===

@router.post("/lifecycle/register", status_code=201)
async def register_agent_lifecycle(
    agent_data: Dict[str, Any],
    orchestrator: VerticalSliceOrchestrator = Depends(get_vertical_slice_orchestrator)
) -> Dict[str, Any]:
    """
    Register an agent using the enhanced lifecycle manager.
    
    This endpoint demonstrates the complete agent registration flow
    with persona assignment and capability matching.
    """
    try:
        if not orchestrator.is_running:
            await orchestrator.start_system()
        
        registration_result = await orchestrator.lifecycle_manager.register_agent(
            name=agent_data.get("name"),
            agent_type=AgentType(agent_data.get("type", "claude")),
            role=agent_data.get("role"),
            capabilities=agent_data.get("capabilities", []),
            system_prompt=agent_data.get("system_prompt"),
            config=agent_data.get("config", {}),
            tmux_session=agent_data.get("tmux_session")
        )
        
        if registration_result.success:
            return {
                "success": True,
                "agent_id": str(registration_result.agent_id),
                "capabilities_assigned": registration_result.capabilities_assigned,
                "persona_assigned": registration_result.persona_assigned,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Registration failed: {registration_result.error_message}"
            )
            
    except Exception as e:
        logger.error("Agent lifecycle registration failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.delete("/lifecycle/{agent_id}/deregister", status_code=200)
async def deregister_agent_lifecycle(
    agent_id: uuid.UUID,
    orchestrator: VerticalSliceOrchestrator = Depends(get_vertical_slice_orchestrator)
) -> Dict[str, Any]:
    """
    Deregister an agent using the enhanced lifecycle manager.
    
    This endpoint demonstrates proper agent shutdown with task cleanup
    and resource deallocation.
    """
    try:
        success = await orchestrator.lifecycle_manager.deregister_agent(agent_id)
        
        if success:
            return {
                "success": True,
                "agent_id": str(agent_id),
                "message": "Agent deregistered successfully",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="Agent not found or deregistration failed"
            )
            
    except Exception as e:
        logger.error("Agent lifecycle deregistration failed", agent_id=str(agent_id), error=str(e))
        raise HTTPException(status_code=500, detail=f"Deregistration failed: {str(e)}")


@router.post("/lifecycle/tasks/{task_id}/assign", status_code=200)
async def assign_task_lifecycle(
    task_id: uuid.UUID,
    assignment_data: Optional[Dict[str, Any]] = None,
    orchestrator: VerticalSliceOrchestrator = Depends(get_vertical_slice_orchestrator)
) -> Dict[str, Any]:
    """
    Assign a task using the intelligent task assignment system.
    
    This endpoint demonstrates persona-based task routing with
    performance tracking and capability matching.
    """
    try:
        assignment_data = assignment_data or {}
        preferred_agent_id = None
        
        if assignment_data.get("preferred_agent_id"):
            preferred_agent_id = uuid.UUID(assignment_data["preferred_agent_id"])
        
        assignment_result = await orchestrator.lifecycle_manager.assign_task_to_agent(
            task_id=task_id,
            preferred_agent_id=preferred_agent_id,
            max_assignment_time_ms=assignment_data.get("max_assignment_time_ms", 500.0)
        )
        
        if assignment_result.success:
            return {
                "success": True,
                "task_id": str(assignment_result.task_id),
                "agent_id": str(assignment_result.agent_id),
                "assignment_time": assignment_result.assignment_time.isoformat(),
                "confidence_score": assignment_result.confidence_score,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Task assignment failed: {assignment_result.error_message}"
            )
            
    except Exception as e:
        logger.error("Task lifecycle assignment failed", task_id=str(task_id), error=str(e))
        raise HTTPException(status_code=500, detail=f"Assignment failed: {str(e)}")


@router.post("/lifecycle/tasks/{task_id}/complete", status_code=200)
async def complete_task_lifecycle(
    task_id: uuid.UUID,
    completion_data: Dict[str, Any],
    orchestrator: VerticalSliceOrchestrator = Depends(get_vertical_slice_orchestrator)
) -> Dict[str, Any]:
    """
    Complete a task using the lifecycle manager.
    
    This endpoint demonstrates task completion tracking with
    performance metrics and result storage.
    """
    try:
        agent_id = uuid.UUID(completion_data["agent_id"])
        result = completion_data.get("result", {})
        success = completion_data.get("success", True)
        
        completion_success = await orchestrator.lifecycle_manager.complete_task(
            task_id=task_id,
            agent_id=agent_id,
            result=result,
            success=success
        )
        
        if completion_success:
            return {
                "success": True,
                "task_id": str(task_id),
                "agent_id": str(agent_id),
                "completion_success": success,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=400,
                detail="Task completion failed"
            )
            
    except Exception as e:
        logger.error("Task lifecycle completion failed", task_id=str(task_id), error=str(e))
        raise HTTPException(status_code=500, detail=f"Completion failed: {str(e)}")


@router.get("/lifecycle/{agent_id}/status", status_code=200)
async def get_agent_lifecycle_status(
    agent_id: uuid.UUID,
    orchestrator: VerticalSliceOrchestrator = Depends(get_vertical_slice_orchestrator)
) -> Dict[str, Any]:
    """
    Get comprehensive agent status from the lifecycle manager.
    
    This endpoint provides detailed agent metrics, current tasks,
    and performance statistics.
    """
    try:
        status = await orchestrator.lifecycle_manager.get_agent_status(agent_id)
        
        if status:
            return {
                "success": True,
                "agent_status": status,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="Agent not found"
            )
            
    except Exception as e:
        logger.error("Failed to get agent lifecycle status", agent_id=str(agent_id), error=str(e))
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


@router.get("/lifecycle/system/metrics", status_code=200)
async def get_lifecycle_system_metrics(
    orchestrator: VerticalSliceOrchestrator = Depends(get_vertical_slice_orchestrator)
) -> Dict[str, Any]:
    """
    Get comprehensive system metrics from the lifecycle system.
    
    This endpoint provides system-wide performance metrics,
    agent statistics, and operational insights.
    """
    try:
        metrics = await orchestrator.get_comprehensive_status()
        return {
            "success": True,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get lifecycle system metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


@router.post("/lifecycle/demo/complete-flow", status_code=200)
async def demonstrate_complete_lifecycle(
    orchestrator: VerticalSliceOrchestrator = Depends(get_vertical_slice_orchestrator)
) -> Dict[str, Any]:
    """
    Demonstrate the complete agent lifecycle flow.
    
    This endpoint runs the full vertical slice demonstration including:
    - Agent registration with personas
    - Task assignment and execution
    - Hook system integration
    - Performance metrics collection
    """
    try:
        if not orchestrator.is_running:
            await orchestrator.start_system()
        
        demo_results = await orchestrator.demonstrate_complete_lifecycle()
        
        return {
            "success": demo_results["success"],
            "demonstration": demo_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Complete lifecycle demonstration failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Demonstration failed: {str(e)}")


@router.post("/lifecycle/system/start", status_code=200)
async def start_lifecycle_system(
    orchestrator: VerticalSliceOrchestrator = Depends(get_vertical_slice_orchestrator)
) -> Dict[str, Any]:
    """Start the vertical slice lifecycle system."""
    try:
        success = await orchestrator.start_system()
        
        return {
            "success": success,
            "message": "Lifecycle system started" if success else "Failed to start system",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to start lifecycle system", error=str(e))
        raise HTTPException(status_code=500, detail=f"System start failed: {str(e)}")


@router.post("/lifecycle/system/stop", status_code=200)
async def stop_lifecycle_system(
    orchestrator: VerticalSliceOrchestrator = Depends(get_vertical_slice_orchestrator)
) -> Dict[str, Any]:
    """Stop the vertical slice lifecycle system gracefully."""
    try:
        success = await orchestrator.stop_system()
        
        return {
            "success": success,
            "message": "Lifecycle system stopped" if success else "Failed to stop system",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to stop lifecycle system", error=str(e))
        raise HTTPException(status_code=500, detail=f"System stop failed: {str(e)}")


@router.get("/{agent_id}/stats", response_model=AgentStatsResponse)
async def get_agent_stats(
    agent_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> AgentStatsResponse:
    """Get detailed statistics for an agent."""
    
    try:
        result = await db.execute(
            select(Agent).where(Agent.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Calculate additional stats
        uptime_hours = 0.0
        if agent.created_at:
            uptime_delta = datetime.utcnow() - agent.created_at
            uptime_hours = uptime_delta.total_seconds() / 3600
        
        success_rate = 0.0
        total_completed = int(agent.total_tasks_completed or 0)
        total_failed = int(agent.total_tasks_failed or 0)
        total_tasks = total_completed + total_failed
        
        if total_tasks > 0:
            success_rate = (total_completed / total_tasks) * 100
        
        return AgentStatsResponse(
            agent_id=agent.id,
            total_tasks_completed=total_completed,
            total_tasks_failed=total_failed,
            success_rate=success_rate,
            average_response_time=float(agent.average_response_time or 0.0),
            context_window_usage=float(agent.context_window_usage or 0.0),
            uptime_hours=uptime_hours,
            last_active=agent.last_active,
            capabilities_count=len(agent.capabilities or [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get agent stats", agent_id=str(agent_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve agent stats")


# Orchestrator-specific endpoints

@router.post("/spawn", response_model=AgentResponse, status_code=201)
async def spawn_agent(
    role: str,
    agent_name: Optional[str] = None,
    capabilities: Optional[List[dict]] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_session_dependency)
) -> AgentResponse:
    """Spawn a new agent instance via orchestrator."""
    
    try:
        # Validate agent role
        try:
            agent_role = AgentRole(role)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid agent role: {role}. Valid roles: {[r.value for r in AgentRole]}"
            )
        
        # Get orchestrator from app state
        from ...main import app
        if not hasattr(app.state, 'orchestrator'):
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        orchestrator: AgentOrchestrator = app.state.orchestrator
        
        # Convert capabilities if provided
        agent_capabilities = []
        if capabilities:
            for cap in capabilities:
                agent_capabilities.append(AgentCapability(
                    name=cap.get("name", ""),
                    description=cap.get("description", ""),
                    confidence_level=cap.get("confidence_level", 0.5),
                    specialization_areas=cap.get("specialization_areas", [])
                ))
        
        # Spawn agent via orchestrator
        agent_id = await orchestrator.spawn_agent(
            role=agent_role,
            agent_id=agent_name,
            capabilities=agent_capabilities
        )
        
        # Return agent data
        result = await db.execute(
            select(Agent).where(Agent.id == agent_id)
        )
        agent = result.scalar_one()
        
        logger.info(
            "Agent spawned via orchestrator",
            agent_id=agent_id,
            role=role
        )
        
        return AgentResponse.from_orm(agent)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to spawn agent", role=role, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to spawn agent")


@router.post("/{agent_id}/terminate", status_code=200)
async def terminate_agent(
    agent_id: uuid.UUID,
    graceful: bool = True,
    db: AsyncSession = Depends(get_session_dependency)
) -> dict:
    """Terminate an agent instance via orchestrator."""
    
    try:
        # Check if agent exists
        result = await db.execute(
            select(Agent).where(Agent.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get orchestrator from app state
        from ...main import app
        if not hasattr(app.state, 'orchestrator'):
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        orchestrator: AgentOrchestrator = app.state.orchestrator
        
        # Terminate agent via orchestrator
        success = await orchestrator.shutdown_agent(str(agent_id), graceful=graceful)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to terminate agent")
        
        logger.info(
            "Agent terminated via orchestrator",
            agent_id=str(agent_id),
            graceful=graceful
        )
        
        return {
            "message": "Agent terminated successfully",
            "agent_id": str(agent_id),
            "graceful": graceful
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to terminate agent", agent_id=str(agent_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to terminate agent")


@router.post("/{agent_id}/restart", response_model=AgentResponse)
async def restart_agent(
    agent_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> AgentResponse:
    """Restart an agent instance via orchestrator."""
    
    try:
        # Check if agent exists
        result = await db.execute(
            select(Agent).where(Agent.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get orchestrator from app state
        from ...main import app
        if not hasattr(app.state, 'orchestrator'):
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        orchestrator: AgentOrchestrator = app.state.orchestrator
        
        # Terminate and respawn agent
        await orchestrator.shutdown_agent(str(agent_id), graceful=True)
        
        # Wait briefly for cleanup
        import asyncio
        await asyncio.sleep(1)
        
        # Convert agent role back to enum
        agent_role = AgentRole(agent.role) if agent.role else AgentRole.STRATEGIC_PARTNER
        
        # Convert capabilities
        agent_capabilities = []
        if agent.capabilities:
            for cap in agent.capabilities:
                agent_capabilities.append(AgentCapability(
                    name=cap.get("name", ""),
                    description=cap.get("description", ""),
                    confidence_level=cap.get("confidence_level", 0.5),
                    specialization_areas=cap.get("specialization_areas", [])
                ))
        
        # Respawn with same ID and configuration
        new_agent_id = await orchestrator.spawn_agent(
            role=agent_role,
            agent_id=str(agent_id),
            capabilities=agent_capabilities
        )
        
        # Fetch updated agent
        result = await db.execute(
            select(Agent).where(Agent.id == new_agent_id)
        )
        restarted_agent = result.scalar_one()
        
        logger.info("Agent restarted via orchestrator", agent_id=str(agent_id))
        
        return AgentResponse.from_orm(restarted_agent)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to restart agent", agent_id=str(agent_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to restart agent")


@router.get("/system/status")
async def get_system_status() -> dict:
    """Get comprehensive system status from orchestrator."""
    
    try:
        # Get orchestrator from app state
        from ...main import app
        if not hasattr(app.state, 'orchestrator'):
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        orchestrator: AgentOrchestrator = app.state.orchestrator
        
        # Get system status
        system_status = await orchestrator.get_system_status()
        
        return system_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get system status", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get system status")


@router.post("/{agent_id}/sleep-cycle", status_code=200)
async def initiate_sleep_cycle(
    agent_id: uuid.UUID,
    db: AsyncSession = Depends(get_session_dependency)
) -> dict:
    """Initiate a sleep-wake cycle for an agent."""
    
    try:
        # Check if agent exists
        result = await db.execute(
            select(Agent).where(Agent.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get orchestrator from app state
        from ...main import app
        if not hasattr(app.state, 'orchestrator'):
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        orchestrator: AgentOrchestrator = app.state.orchestrator
        
        # Initiate sleep cycle
        success = await orchestrator.initiate_sleep_cycle(str(agent_id))
        
        if not success:
            return {
                "message": "Sleep cycle not initiated - agent context usage below threshold",
                "agent_id": str(agent_id),
                "initiated": False
            }
        
        logger.info("Sleep cycle initiated for agent", agent_id=str(agent_id))
        
        return {
            "message": "Sleep cycle initiated successfully",
            "agent_id": str(agent_id),
            "initiated": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to initiate sleep cycle", agent_id=str(agent_id), error=str(e))
        raise HTTPException(status_code=500, detail="Failed to initiate sleep cycle")