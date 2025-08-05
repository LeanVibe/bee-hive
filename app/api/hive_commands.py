"""
Hive Slash Commands API for LeanVibe Agent Hive 2.0

This API provides endpoints for executing custom slash commands with the 
`hive:` prefix, enabling Claude Code-style custom commands specifically
for autonomous development orchestration.
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
import structlog

from ..core.hive_slash_commands import execute_hive_command, get_hive_command_registry

logger = structlog.get_logger()
router = APIRouter()


class HiveCommandRequest(BaseModel):
    """Request to execute a hive slash command."""
    command: str = Field(..., description="The hive command to execute (e.g., '/hive:start')")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context for command execution")


class HiveCommandResponse(BaseModel):
    """Response from hive command execution."""
    success: bool
    command: str
    result: Dict[str, Any]
    execution_time_ms: Optional[float] = None


@router.post("/execute", response_model=HiveCommandResponse)
async def execute_command(request: HiveCommandRequest):
    """
    Execute a hive slash command.
    
    This endpoint allows execution of custom slash commands with the `hive:` prefix
    for autonomous development platform control.
    
    Examples:
    - `/hive:start` - Start multi-agent platform
    - `/hive:spawn backend_developer` - Spawn specific agent
    - `/hive:status --detailed` - Get detailed platform status
    - `/hive:develop "Build authentication API"` - Start autonomous development
    """
    try:
        import time
        start_time = time.time()
        
        logger.info("🎯 Executing hive command", command=request.command)
        
        # Execute the command
        result = await execute_hive_command(request.command, request.context)
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        logger.info("✅ Hive command completed", 
                   command=request.command, 
                   success=result.get("success", False),
                   execution_time_ms=execution_time)
        
        return HiveCommandResponse(
            success=result.get("success", False),
            command=request.command,
            result=result,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        logger.error("❌ Hive command execution failed", 
                    command=request.command, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Command execution failed: {str(e)}"
        )


@router.get("/list")
async def list_commands():
    """
    List all available hive slash commands.
    
    Returns a registry of all available commands with their descriptions
    and usage information.
    """
    try:
        registry = get_hive_command_registry()
        commands_info = {}
        
        for name, command in registry.commands.items():
            commands_info[name] = {
                "name": command.name,
                "description": command.description,
                "usage": command.usage,
                "full_command": f"/hive:{command.name}"
            }
        
        return {
            "success": True,
            "total_commands": len(commands_info),
            "commands": commands_info,
            "usage_examples": [
                "/hive:start --team-size=5",
                "/hive:spawn architect --capabilities=system_design,security",
                "/hive:status --detailed",
                "/hive:productivity --developer --mobile",
                "/hive:develop \"Build user authentication with JWT\"",
                "/hive:oversight --mobile-info",
                "/hive:stop --agents-only"
            ]
        }
        
    except Exception as e:
        logger.error("Failed to list commands", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list commands: {str(e)}"
        )


@router.get("/help/{command_name}")
async def get_command_help(command_name: str):
    """
    Get detailed help for a specific hive command.
    
    Provides usage information, examples, and parameter details
    for the specified command.
    """
    try:
        registry = get_hive_command_registry()
        command = registry.get_command(command_name)
        
        if not command:
            raise HTTPException(
                status_code=404,
                detail=f"Command '{command_name}' not found"
            )
        
        # Generate examples based on command type
        examples = []
        if command_name == "start":
            examples = [
                "/hive:start",
                "/hive:start --quick",
                "/hive:start --team-size=7"
            ]
        elif command_name == "spawn":
            examples = [
                "/hive:spawn product_manager",
                "/hive:spawn backend_developer --capabilities=api_development,database_design",
                "/hive:spawn architect"
            ]
        elif command_name == "develop":
            examples = [
                "/hive:develop \"Build authentication API\"",
                "/hive:develop \"Create user management system\" --dashboard",
                "/hive:develop \"Build REST API with tests\" --timeout=600"
            ]
        elif command_name == "status":
            examples = [
                "/hive:status",
                "/hive:status --detailed",
                "/hive:status --agents-only"
            ]
        elif command_name == "productivity":
            examples = [
                "/hive:productivity",
                "/hive:productivity --developer --mobile",
                "/hive:productivity --insights --workflow=development"
            ]
        elif command_name == "oversight":
            examples = [
                "/hive:oversight",
                "/hive:oversight --mobile-info"
            ]
        elif command_name == "stop":
            examples = [
                "/hive:stop",
                "/hive:stop --agents-only",
                "/hive:stop --force"
            ]
        
        return {
            "success": True,
            "command": {
                "name": command.name,
                "full_command": f"/hive:{command.name}",
                "description": command.description,
                "usage": command.usage,
                "examples": examples,
                "created_at": command.created_at.isoformat() if hasattr(command, 'created_at') else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get command help", command=command_name, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get help for command: {str(e)}"
        )


@router.post("/quick/{command_name}")
async def quick_execute_command(command_name: str, args: Optional[str] = None):
    """
    Quick execution endpoint for hive commands without full request body.
    
    Convenient way to execute commands with simple string arguments.
    
    Examples:
    - POST /api/hive/quick/start
    - POST /api/hive/quick/spawn?args=backend_developer
    - POST /api/hive/quick/status?args=--detailed
    """
    try:
        # Construct command string
        command_text = f"/hive:{command_name}"
        if args:
            command_text += f" {args}"
        
        logger.info("🚀 Quick executing hive command", command=command_text)
        
        # Execute the command
        result = await execute_hive_command(command_text)
        
        return {
            "success": result.get("success", False),
            "command": command_text,
            "result": result
        }
        
    except Exception as e:
        logger.error("Quick command execution failed", 
                    command_name=command_name, args=args, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Quick execution failed: {str(e)}"
        )


@router.get("/status")
async def get_command_system_status():
    """
    Get status of the hive command system.
    
    Returns information about the command registry and system readiness.
    """
    try:
        registry = get_hive_command_registry()
        
        return {
            "success": True,
            "system_ready": True,
            "total_commands": len(registry.commands),
            "available_commands": list(registry.commands.keys()),
            "command_prefix": "hive:",
            "api_endpoints": {
                "execute": "/api/hive/execute",
                "list": "/api/hive/list", 
                "help": "/api/hive/help/{command_name}",
                "quick": "/api/hive/quick/{command_name}"
            }
        }
        
    except Exception as e:
        logger.error("Failed to get command system status", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system status: {str(e)}"
        )