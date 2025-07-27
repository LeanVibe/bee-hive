"""
Enhanced Tools API endpoints for dynamic tool discovery and execution.

Provides RESTful API for:
- Dynamic tool discovery with filtering and search
- Secure tool execution with validation and monitoring
- Tool usage analytics and health monitoring
- Plugin management and tool registration
- Integration with existing security and agent systems
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query, Path, Body
from pydantic import BaseModel, Field, validator
from starlette.status import HTTP_200_OK, HTTP_201_CREATED, HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from ...core.enhanced_tool_registry import (
    EnhancedToolRegistry,
    ToolDefinition,
    ToolCategory,
    ToolAccessLevel,
    ToolExecutionResult,
    ToolUsageMetrics,
    ToolHealthStatus,
    get_enhanced_tool_registry,
    discover_available_tools,
    execute_tool_by_id,
    get_agent_tool_recommendations
)
from ...core.dependencies import get_current_user
from ...schemas.base import BaseResponse
from ...models.user import User


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/enhanced-tools", tags=["Enhanced Tools"])


# Request/Response Models
class ToolDiscoveryRequest(BaseModel):
    """Request model for tool discovery."""
    agent_id: UUID = Field(..., description="Agent ID requesting tool discovery")
    category: Optional[ToolCategory] = Field(None, description="Filter by tool category")
    access_level: Optional[ToolAccessLevel] = Field(None, description="Filter by access level")
    search_query: Optional[str] = Field(None, description="Search in tool name/description")
    include_health_status: bool = Field(False, description="Include health status in response")


class ToolExecutionRequest(BaseModel):
    """Request model for tool execution."""
    tool_id: str = Field(..., description="Tool ID to execute")
    agent_id: UUID = Field(..., description="Agent ID requesting execution")
    input_data: Dict[str, Any] = Field(..., description="Input parameters for tool")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional execution context")
    
    @validator('input_data')
    def validate_input_data(cls, v):
        if not isinstance(v, dict):
            raise ValueError('Input data must be a dictionary')
        return v


class ToolRegistrationRequest(BaseModel):
    """Request model for tool registration."""
    tool_definition: Dict[str, Any] = Field(..., description="Tool definition data")
    override_existing: bool = Field(False, description="Override existing tool with same ID")


class ToolHealthCheckRequest(BaseModel):
    """Request model for tool health checks."""
    tool_ids: Optional[List[str]] = Field(None, description="Specific tool IDs to check (all if empty)")


class ToolDiscoveryResponse(BaseModel):
    """Response model for tool discovery."""
    success: bool = Field(..., description="Whether discovery was successful")
    tools: List[Dict[str, Any]] = Field(..., description="Discovered tools")
    total_count: int = Field(..., description="Total number of tools found")
    agent_id: str = Field(..., description="Agent ID that requested discovery")
    filters_applied: Dict[str, Any] = Field(..., description="Filters that were applied")


class ToolExecutionResponse(BaseModel):
    """Response model for tool execution."""
    success: bool = Field(..., description="Whether execution was successful")
    tool_id: str = Field(..., description="Tool that was executed")
    agent_id: str = Field(..., description="Agent that requested execution")
    output: Any = Field(..., description="Tool execution output")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional execution metadata")


class ToolHealthResponse(BaseModel):
    """Response model for tool health status."""
    tool_id: str = Field(..., description="Tool ID")
    is_healthy: bool = Field(..., description="Whether tool is healthy")
    health_score: float = Field(..., description="Health score (0.0-1.0)")
    last_check: datetime = Field(..., description="Last health check timestamp")
    issues: List[str] = Field(..., description="Health issues found")
    dependencies_status: Dict[str, bool] = Field(..., description="Dependency health status")


class ToolMetricsResponse(BaseModel):
    """Response model for tool metrics."""
    tool_id: str = Field(..., description="Tool ID")
    total_executions: int = Field(..., description="Total number of executions")
    successful_executions: int = Field(..., description="Number of successful executions")
    failed_executions: int = Field(..., description="Number of failed executions")
    success_rate: float = Field(..., description="Success rate (0.0-1.0)")
    average_execution_time_ms: float = Field(..., description="Average execution time")
    last_used_at: Optional[datetime] = Field(None, description="Last usage timestamp")
    unique_agents_count: int = Field(..., description="Number of unique agents using tool")
    error_patterns: Dict[str, int] = Field(..., description="Common error patterns")


@router.post("/discover", response_model=ToolDiscoveryResponse, status_code=HTTP_200_OK)
async def discover_tools(
    request: ToolDiscoveryRequest,
    current_user: User = Depends(get_current_user),
    registry: EnhancedToolRegistry = Depends(get_enhanced_tool_registry)
):
    """
    Discover available tools based on filters and search criteria.
    
    Returns a list of tools available to the specified agent with optional
    filtering by category, access level, and search query.
    """
    try:
        logger.info(f"Tool discovery requested by user {current_user.id} for agent {request.agent_id}")
        
        # Discover tools using registry
        discovered_tools = registry.discover_tools(
            agent_id=request.agent_id,
            category=request.category,
            access_level=request.access_level,
            search_query=request.search_query
        )
        
        # Convert tools to dict format
        tools_data = []
        for tool in discovered_tools:
            tool_data = {
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "category": tool.category.value,
                "access_level": tool.access_level.value,
                "usage_examples": tool.usage_examples,
                "when_to_use": tool.when_to_use,
                "limitations": tool.limitations,
                "timeout_seconds": tool.timeout_seconds,
                "version": tool.version,
                "tags": tool.tags
            }
            
            # Include health status if requested
            if request.include_health_status:
                health_status = await registry.health_check_tool(tool.id)
                tool_data["health_status"] = {
                    "is_healthy": health_status.is_healthy,
                    "health_score": health_status.health_score,
                    "issues": health_status.issues
                }
            
            tools_data.append(tool_data)
        
        # Prepare filters applied summary
        filters_applied = {
            "category": request.category.value if request.category else None,
            "access_level": request.access_level.value if request.access_level else None,
            "search_query": request.search_query,
            "include_health_status": request.include_health_status
        }
        
        return ToolDiscoveryResponse(
            success=True,
            tools=tools_data,
            total_count=len(tools_data),
            agent_id=str(request.agent_id),
            filters_applied=filters_applied
        )
        
    except Exception as e:
        logger.error(f"Error during tool discovery: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Tool discovery failed: {str(e)}"
        )


@router.post("/execute", response_model=ToolExecutionResponse, status_code=HTTP_200_OK)
async def execute_tool(
    request: ToolExecutionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    registry: EnhancedToolRegistry = Depends(get_enhanced_tool_registry)
):
    """
    Execute a tool with specified input parameters.
    
    Performs security validation, input validation, and executes the tool
    with comprehensive monitoring and error handling.
    """
    try:
        logger.info(f"Tool execution requested: {request.tool_id} by agent {request.agent_id}")
        
        # Execute tool using registry
        result = await registry.execute_tool(
            tool_id=request.tool_id,
            agent_id=request.agent_id,
            input_data=request.input_data,
            context=request.context
        )
        
        # Log execution for monitoring
        background_tasks.add_task(
            _log_tool_execution,
            tool_id=request.tool_id,
            agent_id=request.agent_id,
            user_id=current_user.id,
            success=result.success,
            execution_time_ms=result.execution_time_ms
        )
        
        return ToolExecutionResponse(
            success=result.success,
            tool_id=request.tool_id,
            agent_id=str(request.agent_id),
            output=result.output,
            error=result.error,
            execution_time_ms=result.execution_time_ms,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error(f"Error executing tool {request.tool_id}: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Tool execution failed: {str(e)}"
        )


@router.get("/recommendations/{agent_id}")
async def get_tool_recommendations(
    agent_id: UUID = Path(..., description="Agent ID to get recommendations for"),
    current_task: Optional[str] = Query(None, description="Current task description for context"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of recommendations"),
    current_user: User = Depends(get_current_user)
):
    """
    Get tool recommendations for an agent based on current task and usage patterns.
    
    Returns a prioritized list of tools that would be most useful for the agent's
    current context and historical usage patterns.
    """
    try:
        logger.info(f"Tool recommendations requested for agent {agent_id}")
        
        # Get recommendations using convenience function
        recommendations = await get_agent_tool_recommendations(
            agent_id=agent_id,
            current_task=current_task
        )
        
        # Limit results
        recommendations = recommendations[:limit]
        
        # Format recommendations
        recommendations_data = []
        for tool in recommendations:
            recommendations_data.append({
                "id": tool.id,
                "name": tool.name,
                "description": tool.description,
                "category": tool.category.value,
                "when_to_use": tool.when_to_use,
                "usage_examples": tool.usage_examples[:2],  # First 2 examples
                "confidence_score": 0.8  # Would be calculated based on context matching
            })
        
        return {
            "success": True,
            "agent_id": str(agent_id),
            "current_task": current_task,
            "recommendations": recommendations_data,
            "total_count": len(recommendations_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting tool recommendations for agent {agent_id}: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to get tool recommendations: {str(e)}"
        )


@router.get("/health")
async def get_tools_health_status(
    tool_ids: Optional[str] = Query(None, description="Comma-separated tool IDs (all if empty)"),
    current_user: User = Depends(get_current_user),
    registry: EnhancedToolRegistry = Depends(get_enhanced_tool_registry)
):
    """
    Get health status for tools.
    
    Returns health status for specified tools or all tools if no IDs provided.
    """
    try:
        if tool_ids:
            tool_id_list = [tid.strip() for tid in tool_ids.split(",")]
            health_results = {}
            for tool_id in tool_id_list:
                health_results[tool_id] = await registry.health_check_tool(tool_id)
        else:
            health_results = await registry.health_check_all_tools()
        
        # Format health status
        health_data = {}
        for tool_id, health_status in health_results.items():
            health_data[tool_id] = {
                "is_healthy": health_status.is_healthy,
                "health_score": health_status.health_score,
                "last_check": health_status.last_health_check,
                "issues": health_status.issues,
                "dependencies_status": health_status.dependencies_status
            }
        
        # Calculate overall health summary
        total_tools = len(health_data)
        healthy_tools = sum(1 for h in health_data.values() if h["is_healthy"])
        overall_health_score = sum(h["health_score"] for h in health_data.values()) / max(1, total_tools)
        
        return {
            "success": True,
            "health_summary": {
                "total_tools": total_tools,
                "healthy_tools": healthy_tools,
                "unhealthy_tools": total_tools - healthy_tools,
                "overall_health_score": overall_health_score,
                "health_percentage": (healthy_tools / max(1, total_tools)) * 100
            },
            "tool_health_details": health_data,
            "check_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting tools health status: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/metrics")
async def get_tools_metrics(
    tool_id: Optional[str] = Query(None, description="Specific tool ID (all tools if empty)"),
    agent_id: Optional[UUID] = Query(None, description="Filter metrics by agent"),
    include_trends: bool = Query(False, description="Include performance trend data"),
    current_user: User = Depends(get_current_user),
    registry: EnhancedToolRegistry = Depends(get_enhanced_tool_registry)
):
    """
    Get usage metrics for tools.
    
    Returns comprehensive usage statistics and performance metrics.
    """
    try:
        if tool_id:
            # Get metrics for specific tool
            metrics = registry.get_tool_metrics(tool_id)
            if not metrics:
                raise HTTPException(
                    status_code=HTTP_404_NOT_FOUND,
                    detail=f"Tool {tool_id} not found"
                )
            
            metrics_data = {
                tool_id: _format_tool_metrics(metrics, include_trends)
            }
        else:
            # Get metrics for all tools
            all_metrics = registry.get_all_metrics()
            metrics_data = {
                tid: _format_tool_metrics(metrics, include_trends)
                for tid, metrics in all_metrics.items()
            }
        
        # Filter by agent if specified
        if agent_id:
            agent_str = str(agent_id)
            metrics_data = {
                tid: metrics for tid, metrics in metrics_data.items()
                if agent_str in registry.get_tool_metrics(tid).agents_using_tool
            }
        
        # Calculate summary statistics
        if metrics_data:
            total_executions = sum(m["total_executions"] for m in metrics_data.values())
            total_successful = sum(m["successful_executions"] for m in metrics_data.values())
            overall_success_rate = total_successful / max(1, total_executions)
            avg_execution_time = sum(m["average_execution_time_ms"] for m in metrics_data.values()) / len(metrics_data)
        else:
            total_executions = 0
            overall_success_rate = 0.0
            avg_execution_time = 0.0
        
        return {
            "success": True,
            "metrics_summary": {
                "total_tools": len(metrics_data),
                "total_executions": total_executions,
                "overall_success_rate": overall_success_rate,
                "average_execution_time_ms": avg_execution_time
            },
            "tool_metrics": metrics_data,
            "filters_applied": {
                "tool_id": tool_id,
                "agent_id": str(agent_id) if agent_id else None,
                "include_trends": include_trends
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tools metrics: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Metrics retrieval failed: {str(e)}"
        )


@router.get("/agent/{agent_id}/usage")
async def get_agent_tool_usage(
    agent_id: UUID = Path(..., description="Agent ID to get usage for"),
    current_user: User = Depends(get_current_user),
    registry: EnhancedToolRegistry = Depends(get_enhanced_tool_registry)
):
    """
    Get tool usage summary for a specific agent.
    
    Returns tools used by the agent with usage statistics and patterns.
    """
    try:
        usage_data = registry.get_agent_tool_usage(agent_id)
        
        # Enhance with additional statistics
        enhanced_usage = {}
        for tool_id, tool_data in usage_data.items():
            tool_metrics = registry.get_tool_metrics(tool_id)
            enhanced_usage[tool_id] = {
                **tool_data,
                "total_executions": tool_metrics.total_executions if tool_metrics else 0,
                "last_execution_time_ms": (
                    tool_metrics.average_execution_time_ms if tool_metrics else 0
                )
            }
        
        # Calculate agent-specific statistics
        total_tools_used = len(enhanced_usage)
        most_used_tool = max(
            enhanced_usage.items(),
            key=lambda x: x[1]["total_executions"],
            default=(None, {"total_executions": 0})
        )
        
        return {
            "success": True,
            "agent_id": str(agent_id),
            "usage_summary": {
                "total_tools_used": total_tools_used,
                "most_used_tool": {
                    "tool_id": most_used_tool[0],
                    "tool_name": most_used_tool[1].get("tool_name"),
                    "executions": most_used_tool[1]["total_executions"]
                } if most_used_tool[0] else None,
                "average_success_rate": (
                    sum(tool["success_rate"] for tool in enhanced_usage.values()) / 
                    max(1, len(enhanced_usage))
                )
            },
            "tool_usage_details": enhanced_usage
        }
        
    except Exception as e:
        logger.error(f"Error getting agent tool usage for {agent_id}: {e}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Failed to get agent tool usage: {str(e)}"
        )


@router.get("/categories")
async def get_tool_categories(
    current_user: User = Depends(get_current_user)
):
    """
    Get available tool categories with descriptions.
    
    Returns all available tool categories that can be used for filtering.
    """
    categories = {
        category.value: {
            "name": category.name,
            "value": category.value,
            "description": _get_category_description(category)
        }
        for category in ToolCategory
    }
    
    return {
        "success": True,
        "categories": categories,
        "total_categories": len(categories)
    }


@router.get("/access-levels")
async def get_access_levels(
    current_user: User = Depends(get_current_user)
):
    """
    Get available tool access levels with descriptions.
    
    Returns all available access levels for tools.
    """
    access_levels = {
        level.value: {
            "name": level.name,
            "value": level.value,
            "description": _get_access_level_description(level)
        }
        for level in ToolAccessLevel
    }
    
    return {
        "success": True,
        "access_levels": access_levels,
        "total_levels": len(access_levels)
    }


# Helper functions
async def _log_tool_execution(
    tool_id: str,
    agent_id: UUID,
    user_id: int,
    success: bool,
    execution_time_ms: int
):
    """Log tool execution for monitoring and analytics."""
    try:
        logger.info(
            f"Tool execution logged: {tool_id} by agent {agent_id} "
            f"(user {user_id}) - Success: {success}, Time: {execution_time_ms}ms"
        )
        # In production, this would write to a monitoring system
    except Exception as e:
        logger.error(f"Error logging tool execution: {e}")


def _format_tool_metrics(metrics: ToolUsageMetrics, include_trends: bool = False) -> Dict[str, Any]:
    """Format tool metrics for API response."""
    formatted = {
        "total_executions": metrics.total_executions,
        "successful_executions": metrics.successful_executions,
        "failed_executions": metrics.failed_executions,
        "success_rate": (
            metrics.successful_executions / max(1, metrics.total_executions)
        ),
        "average_execution_time_ms": metrics.average_execution_time_ms,
        "last_used_at": metrics.last_used_at.isoformat() if metrics.last_used_at else None,
        "unique_agents_count": len(metrics.agents_using_tool),
        "error_patterns": dict(metrics.error_patterns)
    }
    
    if include_trends:
        formatted["performance_trends"] = metrics.performance_trends[-20:]  # Last 20 data points
    
    return formatted


def _get_category_description(category: ToolCategory) -> str:
    """Get description for tool category."""
    descriptions = {
        ToolCategory.VERSION_CONTROL: "Tools for version control operations like Git and GitHub",
        ToolCategory.DEPLOYMENT: "Tools for application deployment and container management",
        ToolCategory.TESTING: "Tools for running tests and quality assurance",
        ToolCategory.MONITORING: "Tools for system monitoring and observability",
        ToolCategory.COMMUNICATION: "Tools for notifications and team communication",
        ToolCategory.DATA_PROCESSING: "Tools for data manipulation and analysis",
        ToolCategory.SECURITY: "Tools for security scanning and vulnerability assessment",
        ToolCategory.DEVELOPMENT: "General development tools and utilities",
        ToolCategory.INFRASTRUCTURE: "Infrastructure management and provisioning tools",
        ToolCategory.UTILITY: "General utility tools and helpers"
    }
    return descriptions.get(category, "Category description not available")


def _get_access_level_description(access_level: ToolAccessLevel) -> str:
    """Get description for access level."""
    descriptions = {
        ToolAccessLevel.PUBLIC: "Available to all agents without restrictions",
        ToolAccessLevel.RESTRICTED: "Requires special permissions or approval",
        ToolAccessLevel.ADMIN_ONLY: "Only available to administrative agents",
        ToolAccessLevel.QUARANTINED: "Temporarily disabled due to issues"
    }
    return descriptions.get(access_level, "Access level description not available")