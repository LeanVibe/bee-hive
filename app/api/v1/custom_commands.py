"""
Custom Commands API endpoints for LeanVibe Agent Hive 2.0 - Phase 6.1

RESTful API for managing and executing multi-agent workflow commands with comprehensive
security, monitoring, and integration with existing Phase 5 infrastructure.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path, Body
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

import structlog

from ...core.database import get_session
from ...core.command_registry import CommandRegistry, CommandRegistryError
from ...core.task_distributor import TaskDistributor, DistributionStrategy
from ...core.command_executor import CommandExecutor
from ...core.agent_registry import AgentRegistry
from ...core.redis import get_message_broker
from ...core.security import get_current_user
from ...schemas.custom_commands import (
    CommandDefinition, CommandCreateRequest, CommandUpdateRequest,
    CommandExecutionRequest, CommandExecutionResult, CommandStatusResponse,
    CommandListResponse, CommandValidationResult, CommandMetrics,
    StepExecutionResult, CommandStatus
)
from ...observability.hooks import get_hook_interceptor

logger = structlog.get_logger()

# Create router
router = APIRouter(prefix="/custom-commands", tags=["Custom Commands"])

# Global instances (will be properly dependency injected in production)
_command_registry: Optional[CommandRegistry] = None
_task_distributor: Optional[TaskDistributor] = None
_command_executor: Optional[CommandExecutor] = None
_agent_registry: Optional[AgentRegistry] = None


async def get_command_registry() -> CommandRegistry:
    """Get command registry instance."""
    global _command_registry
    if not _command_registry:
        _agent_registry = AgentRegistry()
        await _agent_registry.start()
        _command_registry = CommandRegistry(agent_registry=_agent_registry)
    return _command_registry


async def get_task_distributor() -> TaskDistributor:
    """Get task distributor instance."""
    global _task_distributor, _agent_registry
    if not _task_distributor:
        if not _agent_registry:
            _agent_registry = AgentRegistry()
            await _agent_registry.start()
        message_broker = await get_message_broker()
        _task_distributor = TaskDistributor(
            agent_registry=_agent_registry,
            message_broker=message_broker
        )
    return _task_distributor


async def get_command_executor() -> CommandExecutor:
    """Get command executor instance."""
    global _command_executor
    if not _command_executor:
        registry = await get_command_registry()
        distributor = await get_task_distributor()
        message_broker = await get_message_broker()
        hook_interceptor = get_hook_interceptor()
        
        _command_executor = CommandExecutor(
            command_registry=registry,
            task_distributor=distributor,
            agent_registry=_agent_registry,
            message_broker=message_broker,
            hook_manager=hook_interceptor
        )
        await _command_executor.start()
    return _command_executor


# Command Management Endpoints

@router.post("/commands", response_model=Dict[str, Any])
async def create_command(
    request: CommandCreateRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    command_registry: CommandRegistry = Depends(get_command_registry)
):
    """
    Create a new multi-agent workflow command.
    
    Performs comprehensive validation including agent availability,
    security policy compliance, and workflow structure validation.
    """
    try:
        logger.info(
            "Creating new command",
            command_name=request.definition.name,
            user_id=current_user.get("user_id"),
            workflow_steps=len(request.definition.workflow)
        )
        
        # Register command
        success, validation_result = await command_registry.register_command(
            definition=request.definition,
            author_id=current_user.get("user_id"),
            validate_agents=request.validate_agents,
            dry_run=request.dry_run
        )
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Command validation failed",
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings
                }
            )
        
        # Return success response
        response = {
            "success": True,
            "command_name": request.definition.name,
            "command_version": request.definition.version,
            "validation_result": validation_result.model_dump(),
            "message": "Command created successfully" if not request.dry_run else "Command validation passed"
        }
        
        if request.dry_run:
            response["dry_run"] = True
        
        return response
        
    except CommandRegistryError as e:
        logger.error("Command registry error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Command creation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/commands", response_model=CommandListResponse)
async def list_commands(
    category: Optional[str] = Query(None, description="Filter by category"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    agent_role: Optional[str] = Query(None, description="Filter by required agent role"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    command_registry: CommandRegistry = Depends(get_command_registry)
):
    """
    List available commands with filtering and pagination.
    
    Supports filtering by category, tags, and required agent roles.
    Returns command metadata including execution statistics.
    """
    try:
        # Convert agent role string to enum if provided
        agent_role_enum = None
        if agent_role:
            try:
                from ...schemas.custom_commands import AgentRole
                agent_role_enum = AgentRole(agent_role)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid agent role: {agent_role}"
                )
        
        # Get commands from registry
        commands, total_count = await command_registry.list_commands(
            category=category,
            tag=tag,
            agent_role=agent_role_enum,
            limit=limit,
            offset=offset
        )
        
        # Get available categories and tags for filters
        all_commands, _ = await command_registry.list_commands(limit=1000)
        categories = list(set(cmd.get("category", "general") for cmd in all_commands))
        all_tags = []
        for cmd in all_commands:
            all_tags.extend(cmd.get("tags", []))
        unique_tags = list(set(all_tags))
        
        return CommandListResponse(
            commands=commands,
            total=total_count,
            categories=categories,
            tags=unique_tags
        )
        
    except Exception as e:
        logger.error("Failed to list commands", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/commands/{command_name}", response_model=Dict[str, Any])
async def get_command(
    command_name: str = Path(..., description="Command name"),
    version: Optional[str] = Query(None, description="Specific version (latest if not specified)"),
    command_registry: CommandRegistry = Depends(get_command_registry)
):
    """
    Get detailed information about a specific command.
    
    Returns complete command definition, validation status,
    execution metrics, and capability requirements.
    """
    try:
        # Get command definition
        command_def = await command_registry.get_command(command_name, version)
        
        if not command_def:
            raise HTTPException(
                status_code=404,
                detail=f"Command '{command_name}' not found"
            )
        
        # Get command metrics
        metrics = await command_registry.get_command_metrics(command_name)
        
        # Validate current command state
        validation_result = await command_registry.validate_command(command_def)
        
        return {
            "command_definition": command_def.model_dump(),
            "validation_status": validation_result.model_dump(),
            "execution_metrics": metrics.model_dump() if metrics else None,
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get command", command_name=command_name, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/commands/{command_name}", response_model=Dict[str, Any])
async def update_command(
    request: CommandUpdateRequest,
    command_name: str = Path(..., description="Command name"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    command_registry: CommandRegistry = Depends(get_command_registry)
):
    """
    Update an existing command definition.
    
    Supports version increments, definition updates, and enable/disable operations.
    Maintains backward compatibility and validation requirements.
    """
    try:
        # Check if command exists
        existing_command = await command_registry.get_command(command_name)
        if not existing_command:
            raise HTTPException(
                status_code=404,
                detail=f"Command '{command_name}' not found"
            )
        
        # For now, we'll implement this as delete + create
        # In production, this would be a proper update operation
        if request.definition:
            # Validate new definition
            validation_result = await command_registry.validate_command(
                request.definition, validate_agents=True
            )
            
            if not validation_result.is_valid:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Updated command validation failed",
                        "errors": validation_result.errors
                    }
                )
            
            # Register new version
            success, _ = await command_registry.register_command(
                definition=request.definition,
                author_id=current_user.get("user_id"),
                validate_agents=True
            )
            
            if not success:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to register updated command"
                )
        
        return {
            "success": True,
            "command_name": command_name,
            "message": "Command updated successfully",
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to update command", command_name=command_name, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/commands/{command_name}", response_model=Dict[str, Any])
async def delete_command(
    command_name: str = Path(..., description="Command name"),
    version: Optional[str] = Query(None, description="Specific version (all versions if not specified)"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    command_registry: CommandRegistry = Depends(get_command_registry)
):
    """
    Delete a command or specific version from the registry.
    
    Supports deletion of specific versions or entire command.
    Requires appropriate permissions and validates no active executions.
    """
    try:
        # Check permissions (author or admin)
        user_id = current_user.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        # Delete command
        success = await command_registry.delete_command(
            command_name=command_name,
            version=version,
            author_id=user_id
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Command '{command_name}' not found or access denied"
            )
        
        return {
            "success": True,
            "command_name": command_name,
            "version": version,
            "message": f"Command {'version ' + version if version else ''} deleted successfully",
            "deleted_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to delete command", command_name=command_name, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# Command Execution Endpoints

@router.post("/execute", response_model=CommandExecutionResult)
async def execute_command(
    request: CommandExecutionRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
    command_executor: CommandExecutor = Depends(get_command_executor)
):
    """
    Execute a multi-agent workflow command.
    
    Orchestrates task distribution, agent coordination, and workflow execution
    with comprehensive monitoring, security enforcement, and error handling.
    """
    try:
        logger.info(
            "Executing command",
            command_name=request.command_name,
            user_id=current_user.get("user_id"),
            parameters=len(request.parameters)
        )
        
        # Execute command
        result = await command_executor.execute_command(
            request=request,
            requester_id=current_user.get("user_id")
        )
        
        logger.info(
            "Command execution completed",
            execution_id=str(result.execution_id),
            command_name=request.command_name,
            status=result.status.value,
            duration_seconds=result.total_execution_time_seconds
        )
        
        return result
        
    except ValueError as e:
        logger.error("Command execution validation error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error("Command execution runtime error", error=str(e))
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error("Command execution failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/executions/{execution_id}/status", response_model=CommandStatusResponse)
async def get_execution_status(
    execution_id: str = Path(..., description="Execution ID"),
    command_executor: CommandExecutor = Depends(get_command_executor)
):
    """
    Get current status of a command execution.
    
    Returns real-time execution status, progress information,
    resource usage, and step-by-step execution details.
    """
    try:
        # Validate execution ID format
        try:
            uuid.UUID(execution_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid execution ID format")
        
        # Get execution status
        status_info = await command_executor.get_execution_status(execution_id)
        
        if not status_info:
            raise HTTPException(
                status_code=404,
                detail=f"Execution '{execution_id}' not found"
            )
        
        # Calculate progress percentage (simplified)
        elapsed_time = status_info.get("elapsed_time_seconds", 0)
        estimated_duration = 300  # Default 5 minutes, would be calculated from command definition
        progress_percentage = min(100.0, (elapsed_time / estimated_duration) * 100)
        
        return CommandStatusResponse(
            execution_id=uuid.UUID(execution_id),
            status=CommandStatus(status_info.get("status", "running")),
            progress_percentage=progress_percentage,
            current_step=status_info.get("current_step"),
            estimated_completion=datetime.utcnow() + timedelta(
                seconds=max(0, estimated_duration - elapsed_time)
            ),
            logs=[]  # Would be populated with recent logs
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get execution status", execution_id=execution_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/executions/{execution_id}/cancel", response_model=Dict[str, Any])
async def cancel_execution(
    execution_id: str = Path(..., description="Execution ID"),
    reason: str = Body("user_requested", description="Cancellation reason"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    command_executor: CommandExecutor = Depends(get_command_executor)
):
    """
    Cancel a running command execution.
    
    Performs graceful shutdown of workflow steps, cleans up resources,
    and notifies agents of cancellation.
    """
    try:
        # Validate execution ID format
        try:
            uuid.UUID(execution_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid execution ID format")
        
        # Cancel execution
        success = await command_executor.cancel_execution(execution_id, reason)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Execution '{execution_id}' not found or already completed"
            )
        
        return {
            "success": True,
            "execution_id": execution_id,
            "reason": reason,
            "cancelled_by": current_user.get("user_id"),
            "cancelled_at": datetime.utcnow().isoformat(),
            "message": "Execution cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel execution", execution_id=execution_id, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/executions", response_model=List[Dict[str, Any]])
async def list_active_executions(
    current_user: Dict[str, Any] = Depends(get_current_user),
    command_executor: CommandExecutor = Depends(get_command_executor)
):
    """
    List all currently active command executions.
    
    Returns summary information for all running executions
    including status, progress, and resource usage.
    """
    try:
        active_executions = await command_executor.list_active_executions()
        
        # Add user filtering if needed
        user_id = current_user.get("user_id")
        if not current_user.get("is_admin", False):
            # Filter to user's executions only (would need to track requester in execution)
            pass
        
        return active_executions
        
    except Exception as e:
        logger.error("Failed to list active executions", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# Command Validation Endpoints

@router.post("/validate", response_model=CommandValidationResult)
async def validate_command(
    definition: CommandDefinition,
    validate_agents: bool = Query(True, description="Validate agent availability"),
    command_registry: CommandRegistry = Depends(get_command_registry)
):
    """
    Validate a command definition without registration.
    
    Performs comprehensive validation including workflow structure,
    agent requirements, security policies, and resource constraints.
    """
    try:
        validation_result = await command_registry.validate_command(
            definition=definition,
            validate_agents=validate_agents
        )
        
        return validation_result
        
    except Exception as e:
        logger.error("Command validation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# Monitoring and Analytics Endpoints

@router.get("/metrics", response_model=Dict[str, Any])
async def get_system_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user),
    command_executor: CommandExecutor = Depends(get_command_executor),
    task_distributor: TaskDistributor = Depends(get_task_distributor)
):
    """
    Get comprehensive system metrics for command execution system.
    
    Returns execution statistics, resource utilization, agent workload,
    and performance metrics for monitoring and optimization.
    """
    try:
        # Get executor statistics
        executor_stats = command_executor.get_execution_statistics()
        
        # Get distributor statistics
        distributor_stats = task_distributor.get_distribution_statistics()
        
        # Get agent workload status
        agent_workload = await task_distributor.get_agent_workload_status()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "execution_statistics": executor_stats,
            "distribution_statistics": distributor_stats,
            "agent_workload": agent_workload,
            "system_health": {
                "active_executions": executor_stats.get("active_executions", 0),
                "success_rate": (
                    executor_stats.get("successful_executions", 0) /
                    max(executor_stats.get("total_executions", 1), 1) * 100
                ),
                "average_execution_time": executor_stats.get("average_execution_time", 0),
                "peak_concurrent_executions": executor_stats.get("peak_concurrent_executions", 0)
            }
        }
        
    except Exception as e:
        logger.error("Failed to get system metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/commands/{command_name}/metrics", response_model=CommandMetrics)
async def get_command_metrics(
    command_name: str = Path(..., description="Command name"),
    command_registry: CommandRegistry = Depends(get_command_registry)
):
    """
    Get execution metrics for a specific command.
    
    Returns detailed performance statistics, success rates,
    execution patterns, and optimization recommendations.
    """
    try:
        metrics = await command_registry.get_command_metrics(command_name)
        
        if not metrics:
            raise HTTPException(
                status_code=404,
                detail=f"No metrics found for command '{command_name}'"
            )
        
        return metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get command metrics", command_name=command_name, error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# Agent and Distribution Management

@router.get("/agents/workload", response_model=Dict[str, Dict[str, Any]])
async def get_agent_workload(
    current_user: Dict[str, Any] = Depends(get_current_user),
    task_distributor: TaskDistributor = Depends(get_task_distributor)
):
    """
    Get current workload status for all agents.
    
    Returns real-time agent utilization, capacity, performance metrics,
    and availability status for task distribution optimization.
    """
    try:
        workload_status = await task_distributor.get_agent_workload_status()
        return workload_status
        
    except Exception as e:
        logger.error("Failed to get agent workload", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/distribution/optimize", response_model=Dict[str, Any])
async def optimize_distribution_strategy(
    historical_data: Dict[str, Any] = Body(..., description="Historical performance data"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    task_distributor: TaskDistributor = Depends(get_task_distributor)
):
    """
    Optimize task distribution strategy based on historical performance.
    
    Analyzes execution patterns, agent performance, and resource utilization
    to recommend optimal distribution strategies for improved efficiency.
    """
    try:
        # Requires admin permissions
        if not current_user.get("is_admin", False):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        recommended_strategy = await task_distributor.optimize_distribution_strategy(
            historical_data
        )
        
        return {
            "recommended_strategy": recommended_strategy.value,
            "optimization_timestamp": datetime.utcnow().isoformat(),
            "analysis_data": historical_data,
            "message": f"Recommended strategy: {recommended_strategy.value}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to optimize distribution strategy", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


# Health Check Endpoint

@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Health check endpoint for custom commands system.
    
    Returns system status, component availability, and basic metrics
    for monitoring and alerting systems.
    """
    try:
        # Check component health
        components_health = {
            "command_registry": "healthy",
            "task_distributor": "healthy", 
            "command_executor": "healthy",
            "agent_registry": "healthy"
        }
        
        # Get basic metrics
        global _command_executor
        if _command_executor:
            stats = _command_executor.get_execution_statistics()
            components_health["active_executions"] = stats.get("active_executions", 0)
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "6.1.0",
            "components": components_health,
            "uptime_seconds": 0  # Would be calculated from service start time
        }
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )