"""
Workspace management API endpoints.

Provides HTTP endpoints for managing agent workspaces, including creation,
monitoring, command execution, and resource management.
"""

import uuid
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

import structlog

from ...core.workspace_manager import workspace_manager, WorkspaceConfig
from ...core.database import get_session_dependency
from ...models.agent import Agent

logger = structlog.get_logger()
router = APIRouter()


class WorkspaceCreateRequest(BaseModel):
    """Request to create a new workspace."""
    agent_id: str = Field(..., description="Agent ID to create workspace for")
    project_name: Optional[str] = Field(None, description="Project name")
    max_memory_mb: int = Field(default=2048, ge=512, le=8192, description="Maximum memory in MB")
    max_cpu_percent: float = Field(default=50.0, ge=10.0, le=100.0, description="Maximum CPU percentage")
    python_version: str = Field(default="3.11", description="Python version to use")
    additional_packages: Optional[list] = Field(default=[], description="Additional packages to install")


class CommandExecuteRequest(BaseModel):
    """Request to execute a command in workspace."""
    command: str = Field(..., description="Command to execute")
    window: str = Field(default="code", description="tmux window to execute in")
    capture_output: bool = Field(default=True, description="Whether to capture command output")
    timeout_seconds: int = Field(default=300, ge=1, le=3600, description="Command timeout")


class CommandExecuteResponse(BaseModel):
    """Response from command execution."""
    success: bool
    output: str
    error: str
    execution_time_ms: int


@router.post("/create", status_code=201)
async def create_workspace(
    request: WorkspaceCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session_dependency)
) -> Dict[str, Any]:
    """Create a new workspace for an agent."""
    
    try:
        # Verify agent exists
        agent = await db.get(Agent, request.agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Check if workspace already exists
        existing_workspace = await workspace_manager.get_workspace(request.agent_id)
        if existing_workspace:
            raise HTTPException(status_code=409, detail="Workspace already exists for agent")
        
        # Create workspace config overrides
        config_overrides = {
            "max_memory_mb": request.max_memory_mb,
            "max_cpu_percent": request.max_cpu_percent,
            "python_version": request.python_version,
            "additional_packages": request.additional_packages or []
        }
        
        # Create workspace asynchronously
        workspace = await workspace_manager.create_workspace(
            agent_id=request.agent_id,
            project_name=request.project_name,
            config_overrides=config_overrides
        )
        
        if not workspace:
            raise HTTPException(status_code=500, detail="Failed to create workspace")
        
        logger.info(
            "Workspace created via API",
            agent_id=request.agent_id,
            workspace_name=workspace.config.workspace_name
        )
        
        return {
            "agent_id": request.agent_id,
            "workspace_name": workspace.config.workspace_name,
            "project_path": str(workspace.config.project_path),
            "status": workspace.status.value,
            "tmux_session": workspace.tmux_session.name if workspace.tmux_session else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create workspace via API", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{agent_id}")
async def get_workspace_info(agent_id: str) -> Dict[str, Any]:
    """Get information about an agent's workspace."""
    
    try:
        workspace = await workspace_manager.get_workspace(agent_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        metrics = await workspace.get_metrics()
        
        return {
            "agent_id": agent_id,
            "workspace_name": workspace.config.workspace_name,
            "project_path": str(workspace.config.project_path),
            "status": workspace.status.value,
            "created_at": workspace.created_at.isoformat(),
            "last_activity": workspace.last_activity.isoformat(),
            "metrics": {
                "memory_mb": metrics.total_memory_mb,
                "cpu_percent": metrics.total_cpu_percent,
                "disk_usage_mb": metrics.disk_usage_mb,
                "total_processes": metrics.total_processes,
                "commands_executed": metrics.commands_executed,
                "uptime_seconds": metrics.uptime_seconds
            },
            "processes": [
                {
                    "pid": proc.pid,
                    "name": proc.name,
                    "type": proc.process_type.value,
                    "cpu_percent": proc.cpu_percent,
                    "memory_mb": proc.memory_mb,
                    "started_at": proc.started_at.isoformat()
                }
                for proc in metrics.active_processes
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get workspace info", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{agent_id}/execute")
async def execute_command(
    agent_id: str,
    request: CommandExecuteRequest
) -> CommandExecuteResponse:
    """Execute a command in an agent's workspace."""
    
    try:
        workspace = await workspace_manager.get_workspace(agent_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        import time
        start_time = time.time()
        
        success, output, error = await workspace.execute_command(
            command=request.command,
            window=request.window,
            capture_output=request.capture_output,
            timeout_seconds=request.timeout_seconds
        )
        
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            "Command executed via API",
            agent_id=agent_id,
            command=request.command[:100],  # Truncate long commands
            success=success,
            execution_time_ms=execution_time_ms
        )
        
        return CommandExecuteResponse(
            success=success,
            output=output,
            error=error,
            execution_time_ms=execution_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to execute command", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{agent_id}/server/start")
async def start_development_server(
    agent_id: str,
    command: str,
    port: int,
    process_name: str = "dev_server"
) -> Dict[str, Any]:
    """Start a development server in the workspace."""
    
    try:
        workspace = await workspace_manager.get_workspace(agent_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        success = await workspace.start_development_server(
            command=command,
            port=port,
            process_name=process_name
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to start development server")
        
        return {
            "agent_id": agent_id,
            "process_name": process_name,
            "command": command,
            "port": port,
            "status": "started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to start development server", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{agent_id}/tests/run")
async def run_tests(
    agent_id: str,
    test_command: str = "python -m pytest",
    test_path: str = "tests/"
) -> Dict[str, Any]:
    """Run tests in the workspace."""
    
    try:
        workspace = await workspace_manager.get_workspace(agent_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        success, output = await workspace.run_tests(
            test_command=test_command,
            test_path=test_path
        )
        
        return {
            "agent_id": agent_id,
            "success": success,
            "output": output,
            "test_command": f"{test_command} {test_path}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to run tests", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{agent_id}/suspend")
async def suspend_workspace(agent_id: str) -> Dict[str, str]:
    """Suspend a workspace to save resources."""
    
    try:
        workspace = await workspace_manager.get_workspace(agent_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        success = await workspace.suspend()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to suspend workspace")
        
        return {"agent_id": agent_id, "status": "suspended"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to suspend workspace", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/{agent_id}/resume")
async def resume_workspace(agent_id: str) -> Dict[str, str]:
    """Resume a suspended workspace."""
    
    try:
        workspace = await workspace_manager.get_workspace(agent_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        success = await workspace.resume()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to resume workspace")
        
        return {"agent_id": agent_id, "status": "active"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resume workspace", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{agent_id}")
async def terminate_workspace(agent_id: str) -> Dict[str, str]:
    """Terminate a workspace and clean up resources."""
    
    try:
        workspace = await workspace_manager.get_workspace(agent_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        success = await workspace_manager.terminate_workspace(agent_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to terminate workspace")
        
        return {"agent_id": agent_id, "status": "terminated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to terminate workspace", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/")
async def list_workspaces() -> Dict[str, Any]:
    """List all workspaces with system metrics."""
    
    try:
        system_metrics = await workspace_manager.get_system_metrics()
        
        return {
            "system_metrics": {
                "total_workspaces": system_metrics["total_workspaces"],
                "active_workspaces": system_metrics["active_workspaces"],
                "total_memory_mb": system_metrics["total_memory_mb"],
                "total_cpu_percent": system_metrics["total_cpu_percent"],
                "memory_utilization": system_metrics["memory_utilization"],
                "cpu_utilization": system_metrics["cpu_utilization"]
            },
            "workspaces": system_metrics["workspaces"]
        }
        
    except Exception as e:
        logger.error("Failed to list workspaces", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")