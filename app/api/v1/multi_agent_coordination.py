"""
Multi-Agent Coordination API endpoints for LeanVibe Agent Hive 2.0 Phase 1

Provides REST API endpoints for multi-agent workflow coordination,
agent management, and real-time monitoring of coordinated activities.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import asyncio

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import structlog

from ...core.orchestrator import AgentOrchestrator
from ...core.redis import get_message_broker
from ...core.database import get_session

logger = structlog.get_logger()

router = APIRouter(prefix="/multi-agent", tags=["multi-agent-coordination"])

# Global orchestrator instance (in production, this would be dependency injected)
_orchestrator: Optional[AgentOrchestrator] = None


async def get_orchestrator() -> AgentOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
        await _orchestrator.initialize()
    return _orchestrator


# Pydantic models for API requests/responses

class WorkflowTaskSpec(BaseModel):
    """Specification for a task in a multi-agent workflow."""
    id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Human-readable task name")
    description: str = Field(default="", description="Task description")
    type: str = Field(default="general", description="Task type")
    priority: str = Field(default="medium", description="Task priority")
    estimated_effort: int = Field(default=30, description="Estimated effort in minutes")
    required_capabilities: List[str] = Field(default=[], description="Required agent capabilities")
    dependencies: List[str] = Field(default=[], description="Task dependencies")
    context: Dict[str, Any] = Field(default={}, description="Additional task context")


class WorkflowSpec(BaseModel):
    """Specification for a multi-agent workflow."""
    id: Optional[str] = Field(None, description="Workflow ID (auto-generated if not provided)")
    name: str = Field(..., description="Workflow name")
    description: str = Field(default="", description="Workflow description")
    tasks: List[WorkflowTaskSpec] = Field(..., description="List of tasks in the workflow")
    coordination_strategy: str = Field(default="parallel", description="Coordination strategy")
    metadata: Dict[str, Any] = Field(default={}, description="Additional workflow metadata")


class WorkflowExecutionRequest(BaseModel):
    """Request to execute a multi-agent workflow."""
    workflow: WorkflowSpec
    async_execution: bool = Field(default=True, description="Execute workflow asynchronously")


class AgentRegistrationRequest(BaseModel):
    """Request to register an agent for coordination."""
    agent_id: str = Field(..., description="Agent identifier")
    role: str = Field(..., description="Agent role")
    capabilities: List[str] = Field(..., description="Agent capabilities")


class CoordinationStatusResponse(BaseModel):
    """Response with coordination system status."""
    status: str
    active_workflows: int
    registered_agents: int
    coordination_enabled: bool
    message: str


class WorkflowExecutionResponse(BaseModel):
    """Response for workflow execution request."""
    workflow_id: str
    status: str
    execution_mode: str
    agents_assigned: int
    tasks_count: int
    estimated_duration: int
    message: str


# API Endpoints

@router.post("/workflows/execute", response_model=WorkflowExecutionResponse)
async def execute_multi_agent_workflow(
    request: WorkflowExecutionRequest,
    background_tasks: BackgroundTasks,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """
    Execute a multi-agent workflow with intelligent task distribution.
    
    This endpoint demonstrates the core Phase 1 functionality:
    - Intelligent task decomposition
    - Agent capability matching and assignment
    - Coordinated execution via Redis Streams
    - Real-time monitoring and synchronization
    """
    try:
        workflow_spec = request.workflow.dict()
        
        # Generate workflow ID if not provided
        if not workflow_spec.get('id'):
            workflow_spec['id'] = str(uuid.uuid4())
        
        logger.info("🚀 API: Starting multi-agent workflow execution",
                   workflow_id=workflow_spec['id'],
                   tasks_count=len(workflow_spec['tasks']),
                   strategy=workflow_spec['coordination_strategy'])
        
        if request.async_execution:
            # Execute workflow in background
            background_tasks.add_task(
                orchestrator.execute_multi_agent_workflow,
                workflow_spec,
                request.workflow.coordination_strategy
            )
            
            execution_mode = "async"
            message = "Workflow execution started in background"
        else:
            # Execute workflow synchronously (for demonstration)
            result = await orchestrator.execute_multi_agent_workflow(
                workflow_spec,
                request.workflow.coordination_strategy
            )
            execution_mode = "sync"
            message = f"Workflow completed: {result.get('status', 'unknown')}"
        
        # Count available agents for assignment
        available_agents = len([
            agent for agent in orchestrator.agents.values()
            if agent.status.value == 'active'
        ])
        
        return WorkflowExecutionResponse(
            workflow_id=workflow_spec['id'],
            status="executing" if request.async_execution else "completed",
            execution_mode=execution_mode,
            agents_assigned=min(available_agents, len(workflow_spec['tasks'])),
            tasks_count=len(workflow_spec['tasks']),
            estimated_duration=sum(task.estimated_effort for task in request.workflow.tasks),
            message=message
        )
        
    except Exception as e:
        logger.error("❌ API: Multi-agent workflow execution failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")


@router.get("/coordination/status", response_model=CoordinationStatusResponse)
async def get_coordination_status(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """
    Get the current status of the multi-agent coordination system.
    
    Returns information about active workflows, registered agents,
    and overall system health.
    """
    try:
        active_workflows = len(orchestrator.active_workflows)
        registered_agents = len(orchestrator.agents)
        coordination_enabled = orchestrator.coordination_enabled
        
        status = "healthy" if coordination_enabled else "disabled"
        
        return CoordinationStatusResponse(
            status=status,
            active_workflows=active_workflows,
            registered_agents=registered_agents,
            coordination_enabled=coordination_enabled,
            message=f"Coordination system {status} with {registered_agents} agents and {active_workflows} active workflows"
        )
        
    except Exception as e:
        logger.error("❌ API: Failed to get coordination status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.post("/agents/register")
async def register_agent_for_coordination(
    request: AgentRegistrationRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """
    Register an agent for multi-agent coordination.
    
    This enables an agent to participate in coordinated workflows
    and receive task assignments based on its capabilities.
    """
    try:
        message_broker = await get_message_broker()
        
        success = await message_broker.register_agent(
            agent_id=request.agent_id,
            capabilities=request.capabilities,
            role=request.role
        )
        
        if success:
            logger.info("🤖 API: Agent registered for coordination",
                       agent_id=request.agent_id,
                       role=request.role,
                       capabilities=len(request.capabilities))
            
            return {
                "status": "success",
                "agent_id": request.agent_id,
                "message": "Agent successfully registered for coordination"
            }
        else:
            raise HTTPException(status_code=400, detail="Agent registration failed")
            
    except Exception as e:
        logger.error("❌ API: Agent registration failed", 
                    agent_id=request.agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.get("/workflows/{workflow_id}/status")
async def get_workflow_status(
    workflow_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """
    Get the status of a specific multi-agent workflow.
    
    Returns detailed information about workflow execution progress,
    agent assignments, and task completion status.
    """
    try:
        workflow_data = orchestrator.active_workflows.get(workflow_id)
        
        if not workflow_data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return {
            "workflow_id": workflow_id,
            "status": workflow_data.get('status', 'unknown'),
            "strategy": workflow_data.get('strategy', 'unknown'),
            "tasks_count": len(workflow_data.get('tasks', [])),
            "agents_assigned": len(workflow_data.get('agent_assignments', {})),
            "started_at": workflow_data.get('started_at'),
            "agent_assignments": workflow_data.get('agent_assignments', {}),
            "message": "Workflow status retrieved successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ API: Failed to get workflow status", 
                    workflow_id=workflow_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get workflow status: {str(e)}")


@router.get("/agents")
async def list_coordinated_agents(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """
    List all agents participating in multi-agent coordination.
    
    Returns information about registered agents, their capabilities,
    current status, and task assignments.
    """
    try:
        agents_list = []
        
        for agent_id, agent in orchestrator.agents.items():
            agent_info = {
                "agent_id": agent_id,
                "role": agent.role.value,
                "status": agent.status.value,
                "capabilities": [cap.name for cap in agent.capabilities],
                "current_task": agent.current_task,
                "context_usage": agent.context_window_usage,
                "last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None
            }
            agents_list.append(agent_info)
        
        return {
            "agents": agents_list,
            "total_count": len(agents_list),
            "active_count": len([a for a in agents_list if a["status"] == "active"]),
            "message": f"Retrieved {len(agents_list)} coordinated agents"
        }
        
    except Exception as e:
        logger.error("❌ API: Failed to list agents", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")


@router.post("/coordination/demo")
async def run_coordination_demo():
    """
    Run a demonstration of multi-agent coordination capabilities.
    
    This endpoint executes a predefined workflow that showcases
    the Phase 1 multi-agent coordination features.
    """
    try:
        from phase_1_multi_agent_demo import Phase1MultiAgentDemo
        
        demo = Phase1MultiAgentDemo()
        results = await demo.run_demonstration()
        
        return {
            "demonstration_id": results['demonstration_id'],
            "phase": results['phase'],
            "duration_seconds": results['duration_seconds'],
            "overall_status": results['overall_status'],
            "tests_completed": len(results['tests']),
            "test_results": {
                test_name: test_result['status'] 
                for test_name, test_result in results['tests'].items()
            },
            "message": "Phase 1 coordination demonstration completed successfully"
        }
        
    except Exception as e:
        logger.error("❌ API: Coordination demo failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Demo failed: {str(e)}")


# WebSocket endpoint for real-time coordination monitoring
@router.websocket("/coordination/ws")
async def coordination_websocket(websocket):
    """
    WebSocket endpoint for real-time multi-agent coordination monitoring.
    
    Streams live updates about workflow execution, agent status changes,
    and coordination events to connected clients.
    """
    await websocket.accept()
    
    try:
        orchestrator = await get_orchestrator()
        
        while True:
            # Send periodic status updates
            status_update = {
                "timestamp": datetime.utcnow().isoformat(),
                "active_workflows": len(orchestrator.active_workflows),
                "registered_agents": len(orchestrator.agents),
                "coordination_enabled": orchestrator.coordination_enabled,
                "system_metrics": orchestrator.metrics
            }
            
            await websocket.send_json(status_update)
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except Exception as e:
        logger.error("❌ WebSocket connection error", error=str(e))
        await websocket.close()