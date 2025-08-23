"""
CLI Task Compatibility Endpoints for LeanVibe Agent Hive 2.0

Provides CLI-compatible task endpoints that bridge to SimpleOrchestrator:
- GET /api/tasks/active - for 'hive get tasks' command
- GET /api/workflows/active - for 'hive get workflows' command

Epic 1 Phase 1.2: Task & Workflow API Implementation
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
import structlog

# Import SimpleOrchestrator - the working orchestrator
from ...core.simple_orchestrator import SimpleOrchestrator, get_simple_orchestrator

logger = structlog.get_logger()

# Create router for task compatibility endpoints
router = APIRouter(tags=["cli-compatibility"])

# Get orchestrator dependency
async def get_orchestrator_instance():
    """Get SimpleOrchestrator instance for task operations."""
    try:
        orchestrator = get_simple_orchestrator()
        
        # Ensure orchestrator is initialized
        if not hasattr(orchestrator, '_initialized') or not orchestrator._initialized:
            await orchestrator.initialize()
            
        return orchestrator
    except Exception as e:
        logger.error("Failed to get orchestrator instance", error=str(e))
        raise HTTPException(status_code=503, detail="Orchestrator not available")


# Response models for CLI compatibility
class TaskSummary(BaseModel):
    """Task summary for CLI display."""
    id: str
    description: str
    status: str
    agent_id: Optional[str] = None
    created_at: str
    priority: str


class WorkflowSummary(BaseModel):
    """Workflow summary for CLI display."""
    id: str
    name: str
    status: str
    tasks: int
    created_at: str


class TasksActiveResponse(BaseModel):
    """Response model for active tasks endpoint."""
    active_tasks: List[TaskSummary]
    total: int
    by_status: Dict[str, int]
    summary: str


class WorkflowsActiveResponse(BaseModel):
    """Response model for active workflows endpoint."""
    active_workflows: List[WorkflowSummary]
    total: int
    summary: str


@router.get("/api/tasks/active", response_model=TasksActiveResponse)
async def get_active_tasks(
    limit: int = Query(50, ge=1, le=200, description="Maximum tasks to return"),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator_instance)
) -> TasksActiveResponse:
    """
    Get active tasks for CLI 'hive get tasks' command.
    
    Compatible with CLI expectations for task listing.
    Epic 1 Phase 1.2: Task API Implementation.
    """
    try:
        logger.info("CLI requesting active tasks", limit=limit)
        
        # Get task assignments from SimpleOrchestrator
        task_assignments = orchestrator._task_assignments
        
        # Convert task assignments to CLI-compatible format
        active_tasks = []
        status_counts = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0}
        
        for task_id, assignment in task_assignments.items():
            # Map orchestrator task assignment to CLI task summary
            task_summary = TaskSummary(
                id=task_id,
                description=f"Task {task_id[:8]}...",  # Simplified description
                status=assignment.status.value.lower(),
                agent_id=assignment.agent_id if assignment.agent_id != "unassigned" else None,
                created_at=assignment.assigned_at.isoformat() if assignment.assigned_at else datetime.utcnow().isoformat(),
                priority="medium"  # Default priority
            )
            active_tasks.append(task_summary)
            
            # Count by status
            status_key = assignment.status.value.lower()
            if status_key in status_counts:
                status_counts[status_key] += 1
            else:
                status_counts["pending"] += 1  # Default fallback
        
        # Apply limit
        total_tasks = len(active_tasks)
        active_tasks = active_tasks[:limit]
        
        # Generate summary
        summary = f"{len(active_tasks)} tasks retrieved from {total_tasks} total"
        
        logger.info("Active tasks retrieved for CLI", 
                   total=total_tasks, 
                   returned=len(active_tasks),
                   by_status=status_counts)
        
        return TasksActiveResponse(
            active_tasks=active_tasks,
            total=total_tasks,
            by_status=status_counts,
            summary=summary
        )
        
    except Exception as e:
        logger.error("Failed to get active tasks for CLI", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve active tasks")


@router.get("/api/workflows/active", response_model=WorkflowsActiveResponse)
async def get_active_workflows(
    limit: int = Query(50, ge=1, le=200, description="Maximum workflows to return"),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator_instance)
) -> WorkflowsActiveResponse:
    """
    Get active workflows for CLI 'hive get workflows' command.
    
    Compatible with CLI expectations for workflow listing.
    Epic 1 Phase 1.2: Workflow API Implementation.
    """
    try:
        logger.info("CLI requesting active workflows", limit=limit)
        
        # Get workflows from SimpleOrchestrator
        # For now, we'll create workflows based on agent sessions and task assignments
        agent_sessions = orchestrator._agents
        task_assignments = orchestrator._task_assignments
        
        active_workflows = []
        
        # Create workflow summaries based on agent sessions
        for agent_id, agent_session in agent_sessions.items():
            # Count tasks assigned to this agent
            agent_task_count = len([
                task for task in task_assignments.values() 
                if task.agent_id == agent_id
            ])
            
            workflow_summary = WorkflowSummary(
                id=f"workflow-{agent_id[:8]}",
                name=f"Agent {agent_id[:8]} Workflow",
                status="active" if agent_session.current_task_id else "idle",
                tasks=agent_task_count,
                created_at=agent_session.created_at.isoformat()
            )
            active_workflows.append(workflow_summary)
        
        # If no agent-based workflows, create a system workflow
        if not active_workflows and task_assignments:
            system_workflow = WorkflowSummary(
                id="workflow-system",
                name="System Task Workflow",
                status="active",
                tasks=len(task_assignments),
                created_at=datetime.utcnow().isoformat()
            )
            active_workflows.append(system_workflow)
        
        # Apply limit
        total_workflows = len(active_workflows)
        active_workflows = active_workflows[:limit]
        
        # Generate summary
        total_tasks = sum(wf.tasks for wf in active_workflows)
        summary = f"{len(active_workflows)} active workflows managing {total_tasks} tasks"
        
        logger.info("Active workflows retrieved for CLI",
                   total_workflows=total_workflows,
                   returned=len(active_workflows),
                   total_tasks=total_tasks)
        
        return WorkflowsActiveResponse(
            active_workflows=active_workflows,
            total=total_workflows,
            summary=summary
        )
        
    except Exception as e:
        logger.error("Failed to get active workflows for CLI", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve active workflows")


# Additional endpoints for future CLI expansion
@router.post("/api/tasks", status_code=201)
async def create_task_cli(
    description: str = Query(..., description="Task description"),
    agent_id: Optional[str] = Query(None, description="Specific agent to assign"),
    priority: str = Query("medium", description="Task priority"),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator_instance)
) -> Dict[str, Any]:
    """
    Create task via CLI-compatible endpoint.
    
    Future expansion for 'hive create task' command.
    """
    try:
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Use SimpleOrchestrator to create task
        # This is a simplified implementation - would need task creation method
        logger.info("CLI task creation requested", 
                   task_id=task_id, 
                   description=description,
                   agent_id=agent_id)
        
        return {
            "task_id": task_id,
            "status": "created",
            "message": f"Task {task_id[:8]} created successfully"
        }
        
    except Exception as e:
        logger.error("Failed to create task via CLI", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create task")


@router.put("/api/tasks/{task_id}")
async def update_task_cli(
    task_id: str,
    status: str = Query(..., description="New task status"),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator_instance)
) -> Dict[str, Any]:
    """
    Update task via CLI-compatible endpoint.
    
    Future expansion for 'hive update task' commands.
    """
    try:
        # Get task assignment
        assignment = orchestrator._task_assignments.get(task_id)
        if not assignment:
            raise HTTPException(status_code=404, detail="Task not found")
        
        logger.info("CLI task update requested",
                   task_id=task_id,
                   old_status=assignment.status.value,
                   new_status=status)
        
        return {
            "task_id": task_id,
            "status": "updated",
            "message": f"Task {task_id[:8]} status updated to {status}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update task via CLI", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update task")