"""
Agent Activation API for LeanVibe Agent Hive 2.0

This API endpoint activates the multi-agent system by spawning real agents
and making them operational. It bridges the infrastructure with actual
autonomous development capabilities.
"""

import asyncio
from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import structlog

from ..core.agent_manager import (
    AgentManager,
    AgentSpec
)
from ..core.simple_orchestrator import SimpleOrchestrator, AgentRole, get_simple_orchestrator

logger = structlog.get_logger()
router = APIRouter()


class AgentActivationRequest(BaseModel):
    """Request to activate agents."""
    team_size: int = 5
    roles: List[str] = None
    auto_start_tasks: bool = True


class AgentActivationResponse(BaseModel):
    """Response from agent activation."""
    success: bool
    message: str
    active_agents: Dict[str, Any]
    team_composition: Dict[str, str]


@router.post("/activate", response_model=AgentActivationResponse)
async def activate_agent_system(
    request: AgentActivationRequest,
    background_tasks: BackgroundTasks
):
    """
    Activate the multi-agent system by spawning development team.
    
    This endpoint transforms the system from "infrastructure only" to 
    "operational autonomous development platform" with active agents.
    """
    try:
        logger.info("🚀 Activating agent system", 
                   team_size=request.team_size,
                   auto_start=request.auto_start_tasks)
        
        # Use SimpleOrchestrator to spawn development team
        orchestrator = get_simple_orchestrator()
        team_composition = await spawn_development_team_simple(orchestrator)
        
        # Get current agent status from SimpleOrchestrator
        orchestrator_status = await orchestrator.get_system_status()
        active_agents = orchestrator_status.get("agents", {}).get("details", {})
        
        # Start background task monitoring if requested
        if request.auto_start_tasks:
            background_tasks.add_task(_start_demo_tasks)
        
        logger.info("✅ Agent system activated successfully", 
                   active_agents=len(active_agents))
        
        return AgentActivationResponse(
            success=True,
            message=f"Successfully activated {len(active_agents)} agents",
            active_agents=active_agents,
            team_composition=team_composition
        )
        
    except Exception as e:
        logger.error("❌ Failed to activate agent system", error=str(e))
        raise HTTPException(
            status_code=500, 
            detail=f"Agent activation failed: {str(e)}"
        )


@router.get("/status")
async def get_agent_system_status():
    """Get current status of the agent system using SimpleOrchestrator."""
    try:
        # Get SimpleOrchestrator status
        orchestrator = get_simple_orchestrator()
        system_status = await orchestrator.get_system_status()
        
        agent_count = system_status.get("agents", {}).get("total", 0)
        agents_detail = system_status.get("agents", {}).get("details", {})
        
        # Also check legacy spawner agents for backward compatibility
        spawner_count = 0
        spawner_agents = {}
        try:
            manager = await get_agent_manager()
            spawner_agents = await get_active_agents_status()
            spawner_count = len(spawner_agents) if spawner_agents else 0
        except Exception as e:
            logger.debug(f"Could not get spawner agents: {e}")
        
        total_agents = agent_count + spawner_count
        
        return {
            "active": total_agents > 0,
            "agent_count": total_agents,
            "simple_orchestrator_agents": agent_count,
            "spawner_agents": spawner_count,
            "agents": agents_detail,
            "spawner_agents_detail": spawner_agents,
            "system_ready": total_agents >= 2,  # Minimum viable team
            "orchestrator_type": "SimpleOrchestrator",
            "orchestrator_health": system_status.get("health", "unknown"),
            "performance": system_status.get("performance", {})
        }
        
    except Exception as e:
        logger.error("Failed to get agent system status", error=str(e))
        return {
            "active": False,
            "agent_count": 0,
            "simple_orchestrator_agents": 0,
            "spawner_agents": 0,
            "agents": {},
            "system_ready": False,
            "orchestrator_type": "SimpleOrchestrator",
            "error": str(e)
        }


@router.post("/spawn/{role}")
async def spawn_specific_agent(role: str):
    """Spawn a specific agent with the given role using SimpleOrchestrator."""
    try:
        # Validate role
        try:
            agent_role = AgentRole(role.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role: {role}. Valid roles: {[r.value for r in AgentRole]}"
            )
        
        # Use SimpleOrchestrator to spawn agent
        orchestrator = get_simple_orchestrator()
        agent_id = await orchestrator.spawn_agent(role=agent_role)
        
        return {
            "success": True,
            "agent_id": agent_id,
            "role": agent_role.value,
            "message": f"Successfully spawned {agent_role.value} agent using SimpleOrchestrator"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to spawn agent", role=role, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to spawn {role} agent: {str(e)}"
        )


@router.delete("/deactivate")
async def deactivate_agent_system():
    """Deactivate all agents and stop the agent system."""
    try:
        # Use SimpleOrchestrator to shutdown all agents
        orchestrator = get_simple_orchestrator()
        status = await orchestrator.get_system_status()
        agent_ids = list(status.get("agents", {}).get("details", {}).keys())
        
        shutdown_count = 0
        for agent_id in agent_ids:
            try:
                await orchestrator.shutdown_agent(agent_id, graceful=True)
                shutdown_count += 1
            except Exception as e:
                logger.warning(f"Failed to shutdown agent {agent_id}: {e}")
        
        # Also try to stop legacy agent manager for backward compatibility
        try:
            manager = await get_agent_manager()
            await manager.stop()
        except Exception as e:
            logger.debug(f"Legacy agent manager stop failed: {e}")
        
        return {
            "success": True,
            "message": f"Agent system deactivated successfully - shutdown {shutdown_count} agents",
            "agents_shutdown": shutdown_count
        }
        
    except Exception as e:
        logger.error("Failed to deactivate agent system", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Deactivation failed: {str(e)}"
        )


@router.get("/capabilities")
async def get_agent_capabilities():
    """Get capabilities of all active agents."""
    try:
        active_agents = await get_active_agents_status()
        
        capabilities_summary = {}
        for agent_id, status in active_agents.items():
            role = status.get("role", "unknown")
            capabilities = status.get("capabilities", [])
            
            if role not in capabilities_summary:
                capabilities_summary[role] = {
                    "count": 0,
                    "capabilities": set()
                }
            
            capabilities_summary[role]["count"] += 1
            capabilities_summary[role]["capabilities"].update(capabilities)
        
        # Convert sets to lists for JSON serialization
        for role_info in capabilities_summary.values():
            role_info["capabilities"] = list(role_info["capabilities"])
        
        return {
            "total_agents": len(active_agents),
            "roles": capabilities_summary,
            "system_capabilities": _get_system_wide_capabilities(capabilities_summary)
        }
        
    except Exception as e:
        logger.error("Failed to get agent capabilities", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get capabilities: {str(e)}"
        )


async def _start_demo_tasks():
    """Background task to start demo development tasks."""
    try:
        await asyncio.sleep(5)  # Wait for agents to fully initialize
        
        logger.info("🎬 Starting demo development tasks")
        
        manager = await get_agent_manager()
        active_agents = manager.get_active_agents()
        
        # Create a simple demo task for each agent type
        demo_tasks = [
            ("Requirements analysis for authentication API", "product_manager"),
            ("Architecture design for user management system", "architect"), 
            ("API endpoint implementation planning", "backend_developer"),
            ("Test strategy for authentication flows", "qa_engineer"),
            ("Deployment pipeline design", "devops_engineer")
        ]
        
        # Assign demo tasks to appropriate agents
        for task_description, preferred_role in demo_tasks:
            # Find agent with matching role
            target_agent = None
            for agent in active_agents.values():
                if preferred_role in agent.role.value:
                    target_agent = agent
                    break
            
            if target_agent:
                # Create a demo task ID
                task_id = f"demo_{preferred_role}_{int(asyncio.get_event_loop().time())}"
                
                # Assign the task
                await manager.assign_task(target_agent.id, task_id)
                
                logger.info("📋 Demo task assigned", 
                           agent_id=target_agent.id, 
                           task=task_description)
                
                # Simulate task completion after some time
                asyncio.create_task(_complete_demo_task(
                    manager, target_agent.id, task_id, task_description
                ))
        
    except Exception as e:
        logger.error("Failed to start demo tasks", error=str(e))


async def _complete_demo_task(manager, agent_id: str, task_id: str, description: str):
    """Complete a demo task after a delay."""
    try:
        # Wait 30-60 seconds to simulate real work
        await asyncio.sleep(45)
        
        # Mark task as completed
        result = {
            "task_description": description,
            "completion_time": 45,
            "status": "completed",
            "output": f"Successfully completed: {description}",
            "demo": True
        }
        
        await manager.complete_task(agent_id, task_id, result)
        
        logger.info("✅ Demo task completed", 
                   agent_id=agent_id, 
                   task_id=task_id)
        
    except Exception as e:
        logger.error("Failed to complete demo task", 
                    agent_id=agent_id, task_id=task_id, error=str(e))


def _get_system_wide_capabilities(capabilities_summary: Dict[str, Any]) -> List[str]:
    """Get overall system capabilities."""
    all_capabilities = set()
    
    for role_info in capabilities_summary.values():
        all_capabilities.update(role_info["capabilities"])
    
    return list(all_capabilities)


async def spawn_development_team_simple(orchestrator: SimpleOrchestrator) -> Dict[str, str]:
    """Spawn a development team using SimpleOrchestrator."""
    team_composition = {}
    
    # Define the core development team
    core_roles = [
        AgentRole.BACKEND_DEVELOPER,
        AgentRole.FRONTEND_DEVELOPER,
        AgentRole.QA_ENGINEER,
        AgentRole.DEVOPS_ENGINEER
    ]
    
    for role in core_roles:
        try:
            agent_id = await orchestrator.spawn_agent(role=role)
            team_composition[role.value] = agent_id
            logger.info(f"Spawned {role.value}: {agent_id}")
        except Exception as e:
            logger.warning(f"Failed to spawn {role.value}: {e}")
            team_composition[role.value] = f"failed: {str(e)}"
    
    return team_composition