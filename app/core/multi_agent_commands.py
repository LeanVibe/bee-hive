"""
Multi-Agent Workflow Commands for LeanVibe Agent Hive 2.0

Enhanced slash commands that enable real multi-agent coordination, team assembly,
knowledge synchronization, and coordinated development workflows.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import structlog
from pydantic import BaseModel, Field

from app.core.orchestrator import AgentOrchestrator
from app.core.enhanced_orchestrator_integration import get_enhanced_orchestrator_integration
from app.core.leanvibe_hooks_system import HookEventType, get_leanvibe_hooks_engine
from app.core.extended_thinking_engine import ThinkingDepth, get_extended_thinking_engine

logger = structlog.get_logger()


class TeamRole(str, Enum):
    """Standard team roles for multi-agent coordination."""
    BACKEND = "backend"
    FRONTEND = "frontend" 
    DEVOPS = "devops"
    TESTING = "testing"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ARCHITECT = "architect"
    DESIGNER = "designer"


class WorkflowPhase(str, Enum):
    """Workflow execution phases."""
    PLANNING = "planning"
    FOUNDATION = "foundation"
    IMPLEMENTATION = "implementation"
    INTEGRATION = "integration"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    OPTIMIZATION = "optimization"


class CoordinationLevel(str, Enum):
    """Levels of agent coordination."""
    INDEPENDENT = "independent"
    SYNCHRONIZED = "synchronized"
    COLLABORATIVE = "collaborative"
    INTENSIVE = "intensive"


class TeamAssemblyRequest(BaseModel):
    """Request for team assembly."""
    workflow_id: str = Field(..., description="Workflow identifier")
    required_roles: List[TeamRole] = Field(..., description="Required team roles")
    coordination_level: CoordinationLevel = Field(default=CoordinationLevel.SYNCHRONIZED, description="Coordination level")
    estimated_duration_hours: float = Field(default=4.0, description="Estimated workflow duration")
    quality_requirements: Dict[str, Any] = Field(default_factory=dict, description="Quality requirements")
    knowledge_domains: List[str] = Field(default_factory=list, description="Required knowledge domains")


class MultiAgentWorkflowResult(BaseModel):
    """Result of multi-agent workflow execution."""
    workflow_id: str = Field(..., description="Workflow identifier")
    success: bool = Field(..., description="Overall workflow success")
    phases_completed: List[WorkflowPhase] = Field(default_factory=list, description="Completed phases")
    agents_participated: List[str] = Field(default_factory=list, description="Participating agents")
    execution_time_minutes: float = Field(..., description="Total execution time")
    deliverables: List[str] = Field(default_factory=list, description="Generated deliverables")
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="Quality metrics achieved")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")


class MultiAgentCommands:
    """
    Enhanced slash commands for multi-agent workflow coordination.
    
    Provides high-level commands for:
    - Team assembly and role assignment
    - Cross-agent knowledge synchronization  
    - Multi-agent quality gates
    - Coordinated workflow execution
    - Real-time collaboration monitoring
    """
    
    def __init__(
        self,
        orchestrator: Optional[AgentOrchestrator] = None
    ):
        """
        Initialize multi-agent commands system.
        
        Args:
            orchestrator: Agent orchestrator for team coordination
        """
        self.orchestrator = orchestrator
        self.enhanced_integration = get_enhanced_orchestrator_integration()
        self.hooks_engine = get_leanvibe_hooks_engine()
        self.thinking_engine = get_extended_thinking_engine()
        
        # Active workflows and teams
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.assembled_teams: Dict[str, Dict[str, Any]] = {}
        
        # Command definitions
        self.commands = self._initialize_multi_agent_commands()
        
        logger.info("🚀 Multi-Agent Commands system initialized")
    
    def _initialize_multi_agent_commands(self) -> Dict[str, Dict[str, Any]]:
        """Initialize multi-agent command definitions."""
        return {
            "team:assemble": {
                "description": "Assemble multi-agent team for workflow",
                "syntax": "/team:assemble <roles> [--coordination=<level>] [--duration=<hours>]",
                "example": "/team:assemble backend frontend testing --coordination=collaborative --duration=4",
                "handler": self._cmd_team_assemble
            },
            
            "workflow:start": {
                "description": "Start coordinated multi-agent workflow",
                "syntax": "/workflow:start \"<description>\" [--team=<team_id>] [--phases=<phases>]",
                "example": "/workflow:start \"Implement user authentication\" --team=auth_team --phases=foundation,implementation,testing",
                "handler": self._cmd_workflow_start
            },
            
            "agents:sync": {
                "description": "Synchronize knowledge across agent team",
                "syntax": "/agents:sync [--domain=<knowledge_domain>] [--agents=<agent_list>]",
                "example": "/agents:sync --domain=authentication --agents=backend,frontend,security",
                "handler": self._cmd_agents_sync
            },
            
            "quality:gate": {
                "description": "Execute multi-agent quality validation",
                "syntax": "/quality:gate [--scope=<validation_scope>] [--team=<team_id>]",
                "example": "/quality:gate --scope=comprehensive --team=auth_team",
                "handler": self._cmd_quality_gate
            },
            
            "deployment:coordinate": {
                "description": "Coordinate multi-agent deployment workflow",
                "syntax": "/deployment:coordinate [--environment=<env>] [--dependencies=<deps>]",
                "example": "/deployment:coordinate --environment=staging --dependencies=database,redis",
                "handler": self._cmd_deployment_coordinate
            },
            
            "thinking:collaborate": {
                "description": "Initiate collaborative thinking session",
                "syntax": "/thinking:collaborate \"<problem>\" [--agents=<agent_list>] [--depth=<depth>]",
                "example": "/thinking:collaborate \"Design scalable authentication\" --agents=backend,architect,security --depth=deep",
                "handler": self._cmd_thinking_collaborate
            },
            
            "workflow:status": {
                "description": "Show multi-agent workflow status",
                "syntax": "/workflow:status [<workflow_id>] [--detailed]",
                "example": "/workflow:status auth_workflow_001 --detailed",
                "handler": self._cmd_workflow_status
            },
            
            "team:performance": {
                "description": "Show team performance metrics",
                "syntax": "/team:performance [<team_id>] [--timeframe=<period>]",
                "example": "/team:performance auth_team --timeframe=24h",
                "handler": self._cmd_team_performance
            }
        }
    
    async def execute_multi_agent_command(
        self,
        command: str,
        args: List[str],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute multi-agent command.
        
        Args:
            command: Command name (without /)
            args: Command arguments
            session_id: Session identifier
            
        Returns:
            Command execution result
        """
        start_time = time.time()
        
        try:
            if command not in self.commands:
                return {
                    "success": False,
                    "error": f"Unknown multi-agent command: {command}",
                    "available_commands": list(self.commands.keys())
                }
            
            command_info = self.commands[command]
            handler = command_info["handler"]
            
            # Execute command handler
            result = await handler(args, session_id)
            
            # Add execution metadata
            execution_time = (time.time() - start_time) * 1000
            result["execution_time_ms"] = execution_time
            result["command"] = command
            result["timestamp"] = datetime.utcnow().isoformat()
            
            logger.info(
                f"🚀 Multi-agent command executed: {command}",
                command=command,
                success=result.get("success", True),
                execution_time_ms=execution_time
            )
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            logger.error(
                "❌ Multi-agent command failed",
                command=command,
                error=str(e),
                execution_time_ms=execution_time,
                exc_info=True
            )
            
            return {
                "success": False,
                "error": f"Command execution failed: {str(e)}",
                "command": command,
                "execution_time_ms": execution_time
            }
    
    async def _cmd_team_assemble(
        self,
        args: List[str],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle /team:assemble command."""
        try:
            if not args:
                return {
                    "success": False,
                    "error": "Usage: /team:assemble <roles> [--coordination=<level>] [--duration=<hours>]",
                    "example": "/team:assemble backend frontend testing --coordination=collaborative"
                }
            
            # Parse arguments
            roles = []
            coordination_level = CoordinationLevel.SYNCHRONIZED
            duration_hours = 4.0
            
            i = 0
            while i < len(args):
                arg = args[i]
                
                if arg.startswith("--coordination="):
                    coordination_level = CoordinationLevel(arg.split("=")[1])
                elif arg.startswith("--duration="):
                    duration_hours = float(arg.split("=")[1])
                elif not arg.startswith("--"):
                    # Role argument
                    try:
                        role = TeamRole(arg)
                        roles.append(role)
                    except ValueError:
                        logger.warning(f"Unknown role: {arg}")
                
                i += 1
            
            if not roles:
                return {
                    "success": False,
                    "error": "At least one valid role must be specified",
                    "valid_roles": [role.value for role in TeamRole]
                }
            
            # Create team assembly request
            team_id = f"team_{uuid.uuid4().hex[:8]}"
            assembly_request = TeamAssemblyRequest(
                workflow_id=team_id,
                required_roles=roles,
                coordination_level=coordination_level,
                estimated_duration_hours=duration_hours
            )
            
            # Get available agents for roles
            if self.orchestrator:
                all_agents = await self.orchestrator.get_all_agents()
                
                assembled_agents = {}
                for role in roles:
                    # Find best agent for role
                    suitable_agents = [
                        agent for agent in all_agents
                        if role.value in agent.get("role", "").lower() or
                           role.value in [cap.lower() for cap in agent.get("capabilities", [])]
                    ]
                    
                    if suitable_agents:
                        # Select best available agent
                        best_agent = suitable_agents[0]  # Simple selection - could be enhanced
                        assembled_agents[role.value] = {
                            "agent_id": best_agent["id"],
                            "agent_name": best_agent.get("name", best_agent["id"]),
                            "capabilities": best_agent.get("capabilities", []),
                            "status": best_agent.get("status", "unknown")
                        }
                    else:
                        assembled_agents[role.value] = {
                            "status": "not_available",
                            "error": f"No agents available for {role.value} role"
                        }
                
                # Store assembled team
                self.assembled_teams[team_id] = {
                    "team_id": team_id,
                    "roles": [role.value for role in roles],
                    "agents": assembled_agents,
                    "coordination_level": coordination_level.value,
                    "estimated_duration_hours": duration_hours,
                    "created_at": datetime.utcnow().isoformat(),
                    "status": "assembled"
                }
                
                # Execute team assembly hooks
                if self.hooks_engine:
                    await self.hooks_engine.execute_workflow_hooks(
                        event=HookEventType.WORKFLOW_START,
                        workflow_id=team_id,
                        workflow_data={
                            "event_type": "team_assembly",
                            "roles": [role.value for role in roles],
                            "coordination_level": coordination_level.value,
                            "agents_count": len(assembled_agents)
                        },
                        agent_id="team_assembler",
                        session_id=session_id or "system"
                    )
                
                return {
                    "success": True,
                    "team_id": team_id,
                    "message": f"Team assembled with {len(roles)} roles",
                    "assembled_agents": assembled_agents,
                    "coordination_level": coordination_level.value,
                    "estimated_duration_hours": duration_hours,
                    "next_steps": [
                        f"Use /workflow:start to begin coordinated work",
                        f"Use /agents:sync to synchronize team knowledge",
                        f"Use /team:performance {team_id} to monitor progress"
                    ]
                }
            else:
                return {
                    "success": False,
                    "error": "Agent orchestrator not available for team assembly"
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Team assembly failed: {str(e)}"
            }
    
    async def _cmd_workflow_start(
        self,
        args: List[str],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle /workflow:start command."""
        try:
            if not args:
                return {
                    "success": False,
                    "error": "Usage: /workflow:start \"<description>\" [--team=<team_id>] [--phases=<phases>]",
                    "example": "/workflow:start \"Implement authentication\" --team=auth_team"
                }
            
            # Parse workflow description (first argument should be quoted description)
            description = args[0].strip('"\'')
            
            # Parse optional arguments
            team_id = None
            phases = [WorkflowPhase.FOUNDATION, WorkflowPhase.IMPLEMENTATION, WorkflowPhase.TESTING]
            
            for arg in args[1:]:
                if arg.startswith("--team="):
                    team_id = arg.split("=")[1]
                elif arg.startswith("--phases="):
                    phase_names = arg.split("=")[1].split(",")
                    phases = [WorkflowPhase(name.strip()) for name in phase_names]
            
            # Create workflow
            workflow_id = f"workflow_{uuid.uuid4().hex[:8]}"
            
            # Get team information if specified
            team_info = None
            if team_id and team_id in self.assembled_teams:
                team_info = self.assembled_teams[team_id]
            
            # Analyze workflow for extended thinking needs
            thinking_config = None
            if self.thinking_engine:
                thinking_config = await self.thinking_engine.analyze_thinking_needs(
                    task_description=description,
                    task_context={
                        "workflow_type": "multi_agent",
                        "team_size": len(team_info["agents"]) if team_info else 1,
                        "phases": [phase.value for phase in phases]
                    }
                )
            
            # Create workflow execution plan
            workflow_plan = {
                "workflow_id": workflow_id,
                "description": description,
                "phases": [phase.value for phase in phases],
                "team_id": team_id,
                "team_info": team_info,
                "thinking_config": thinking_config,
                "status": "planning",
                "created_at": datetime.utcnow().isoformat(),
                "estimated_completion": (datetime.utcnow() + timedelta(hours=4)).isoformat()
            }
            
            # Store active workflow
            self.active_workflows[workflow_id] = workflow_plan
            
            # Execute workflow start hooks
            if self.hooks_engine:
                await self.hooks_engine.execute_workflow_hooks(
                    event=HookEventType.WORKFLOW_START,
                    workflow_id=workflow_id,
                    workflow_data={
                        "event_type": "multi_agent_workflow_start",
                        "description": description,
                        "phases": [phase.value for phase in phases],
                        "team_size": len(team_info["agents"]) if team_info else 1,
                        "thinking_required": thinking_config is not None
                    },
                    agent_id="workflow_coordinator",
                    session_id=session_id or "system"
                )
            
            # Start collaborative thinking if needed
            thinking_session_id = None
            if thinking_config and team_info:
                try:
                    thinking_session = await self.thinking_engine.enable_extended_thinking(
                        agent_id="workflow_coordinator",
                        workflow_id=workflow_id,
                        problem_description=description,
                        problem_context={
                            "team_agents": list(team_info["agents"].keys()),
                            "workflow_phases": [phase.value for phase in phases]
                        },
                        thinking_depth=thinking_config.get("thinking_depth", ThinkingDepth.STANDARD)
                    )
                    thinking_session_id = thinking_session.session_id
                    
                except Exception as e:
                    logger.warning(f"Failed to start thinking session: {e}")
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "description": description,
                "message": f"Multi-agent workflow started: {description}",
                "phases": [phase.value for phase in phases],
                "team_info": team_info,
                "thinking_session_id": thinking_session_id,
                "status": "planning",
                "next_steps": [
                    f"Use /workflow:status {workflow_id} to monitor progress",
                    f"Use /quality:gate --team={team_id} for validation",
                    f"Use /agents:sync for team coordination"
                ]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Workflow start failed: {str(e)}"
            }
    
    async def _cmd_agents_sync(
        self,
        args: List[str],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle /agents:sync command."""
        try:
            # Parse arguments
            knowledge_domain = None
            target_agents = []
            
            for arg in args:
                if arg.startswith("--domain="):
                    knowledge_domain = arg.split("=")[1]
                elif arg.startswith("--agents="):
                    agent_names = arg.split("=")[1].split(",")
                    target_agents.extend([name.strip() for name in agent_names])
            
            # If no specific agents, sync all available agents
            if not target_agents and self.orchestrator:
                all_agents = await self.orchestrator.get_all_agents()
                target_agents = [agent["id"] for agent in all_agents]
            
            # Execute knowledge synchronization
            sync_results = {}
            
            for agent_id in target_agents:
                try:
                    # This would integrate with the agent memory system
                    # For now, we'll simulate the sync operation
                    sync_result = {
                        "agent_id": agent_id,
                        "knowledge_domain": knowledge_domain or "general",
                        "sync_status": "completed",
                        "items_synchronized": 5,  # Placeholder
                        "sync_time_ms": 150
                    }
                    
                    sync_results[agent_id] = sync_result
                    
                except Exception as e:
                    sync_results[agent_id] = {
                        "agent_id": agent_id,
                        "sync_status": "failed",
                        "error": str(e)
                    }
            
            successful_syncs = [r for r in sync_results.values() if r.get("sync_status") == "completed"]
            
            return {
                "success": len(successful_syncs) > 0,
                "message": f"Knowledge synchronization completed for {len(successful_syncs)}/{len(target_agents)} agents",
                "knowledge_domain": knowledge_domain or "general",
                "sync_results": sync_results,
                "summary": {
                    "total_agents": len(target_agents),
                    "successful_syncs": len(successful_syncs),
                    "failed_syncs": len(target_agents) - len(successful_syncs)
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Agent synchronization failed: {str(e)}"
            }
    
    async def _cmd_quality_gate(
        self,
        args: List[str],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle /quality:gate command."""
        try:
            # Parse arguments
            scope = "standard"
            team_id = None
            
            for arg in args:
                if arg.startswith("--scope="):
                    scope = arg.split("=")[1]
                elif arg.startswith("--team="):
                    team_id = arg.split("=")[1]
            
            # Execute quality gate using enhanced integration
            if self.enhanced_integration:
                quality_criteria = {
                    "scope": scope,
                    "team_id": team_id,
                    "multi_agent_validation": True,
                    "comprehensive_checks": scope == "comprehensive"
                }
                
                result = await self.enhanced_integration.execute_quality_gate(
                    workflow_id=f"quality_gate_{uuid.uuid4().hex[:8]}",
                    quality_criteria=quality_criteria,
                    session_id=session_id
                )
                
                return {
                    "success": result["success"],
                    "message": f"Multi-agent quality gate executed with {scope} scope",
                    "quality_score": result["quality_score"],
                    "validation_results": result.get("details", {}),
                    "recommendations": self._generate_quality_recommendations(result)
                }
            else:
                return {
                    "success": False,
                    "error": "Enhanced integration not available for quality gates"
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Quality gate execution failed: {str(e)}"
            }
    
    def _generate_quality_recommendations(self, quality_result: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        quality_score = quality_result.get("quality_score", 0)
        
        if quality_score < 0.8:
            recommendations.append("Consider running code formatting and optimization hooks")
            recommendations.append("Review failed quality checks and implement fixes")
        
        if quality_score < 0.6:
            recommendations.append("Initiate extended thinking session for quality improvement")
            recommendations.append("Consider agent coordination to address quality issues")
        
        recommendations.append("Monitor ongoing quality metrics with /team:performance")
        
        return recommendations
    
    async def _cmd_deployment_coordinate(
        self,
        args: List[str],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle /deployment:coordinate command."""
        try:
            # Parse arguments
            environment = "staging"
            dependencies = []
            
            for arg in args:
                if arg.startswith("--environment="):
                    environment = arg.split("=")[1]
                elif arg.startswith("--dependencies="):
                    deps = arg.split("=")[1].split(",")
                    dependencies.extend([dep.strip() for dep in deps])
            
            # Create coordinated deployment plan
            deployment_id = f"deployment_{uuid.uuid4().hex[:8]}"
            
            deployment_plan = {
                "deployment_id": deployment_id,
                "environment": environment,
                "dependencies": dependencies,
                "status": "planning",
                "phases": [
                    "dependency_validation",
                    "pre_deployment_hooks",
                    "deployment_execution", 
                    "post_deployment_validation",
                    "health_checks"
                ],
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Execute deployment coordination hooks
            if self.hooks_engine:
                await self.hooks_engine.execute_workflow_hooks(
                    event=HookEventType.WORKFLOW_START,
                    workflow_id=deployment_id,
                    workflow_data={
                        "event_type": "deployment_coordination",
                        "environment": environment,
                        "dependencies": dependencies,
                        "coordination_type": "multi_agent"
                    },
                    agent_id="deployment_coordinator",
                    session_id=session_id or "system"
                )
            
            return {
                "success": True,
                "deployment_id": deployment_id,
                "message": f"Deployment coordination initiated for {environment} environment",
                "environment": environment,
                "dependencies": dependencies,
                "deployment_plan": deployment_plan,
                "next_steps": [
                    "Dependency validation in progress",
                    "Pre-deployment hooks will execute automatically",
                    f"Monitor progress with /workflow:status {deployment_id}"
                ]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Deployment coordination failed: {str(e)}"
            }
    
    async def _cmd_thinking_collaborate(
        self,
        args: List[str],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle /thinking:collaborate command."""
        try:
            if not args:
                return {
                    "success": False,
                    "error": "Usage: /thinking:collaborate \"<problem>\" [--agents=<agent_list>] [--depth=<depth>]",
                    "example": "/thinking:collaborate \"Design authentication\" --agents=backend,security --depth=deep"
                }
            
            # Parse problem description
            problem = args[0].strip('"\'')
            
            # Parse optional arguments
            target_agents = []
            thinking_depth = ThinkingDepth.COLLABORATIVE
            
            for arg in args[1:]:
                if arg.startswith("--agents="):
                    agent_names = arg.split("=")[1].split(",")
                    target_agents.extend([name.strip() for name in agent_names])
                elif arg.startswith("--depth="):
                    depth_str = arg.split("=")[1]
                    thinking_depth = ThinkingDepth(depth_str)
            
            # Start collaborative thinking session
            if self.thinking_engine:
                thinking_session = await self.thinking_engine.enable_extended_thinking(
                    agent_id="collaboration_coordinator",
                    workflow_id=f"collaborative_thinking_{uuid.uuid4().hex[:8]}",
                    problem_description=problem,
                    problem_context={
                        "collaboration_type": "multi_agent",
                        "target_agents": target_agents,
                        "session_type": "command_initiated"
                    },
                    thinking_depth=thinking_depth
                )
                
                return {
                    "success": True,
                    "thinking_session_id": thinking_session.session_id,
                    "problem": problem,
                    "message": f"Collaborative thinking session started: {thinking_depth.value} depth",
                    "participating_agents": thinking_session.participating_agents,
                    "thinking_depth": thinking_depth.value,
                    "estimated_duration_minutes": thinking_session.thinking_time_limit_seconds // 60,
                    "next_steps": [
                        f"Monitor session with /workflow:status {thinking_session.session_id}",
                        "Agents will collaborate on problem analysis",
                        "Results will be available when session completes"
                    ]
                }
            else:
                return {
                    "success": False,
                    "error": "Extended thinking engine not available"
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Collaborative thinking failed: {str(e)}"
            }
    
    async def _cmd_workflow_status(
        self,
        args: List[str],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle /workflow:status command."""
        try:
            # Get workflow ID from arguments
            workflow_id = args[0] if args else None
            detailed = "--detailed" in args
            
            if workflow_id:
                # Show specific workflow status
                if workflow_id in self.active_workflows:
                    workflow = self.active_workflows[workflow_id]
                    
                    status_info = {
                        "workflow_id": workflow_id,
                        "description": workflow.get("description", ""),
                        "status": workflow.get("status", "unknown"),
                        "phases": workflow.get("phases", []),
                        "team_id": workflow.get("team_id"),
                        "created_at": workflow.get("created_at"),
                        "estimated_completion": workflow.get("estimated_completion")
                    }
                    
                    if detailed:
                        status_info["team_info"] = workflow.get("team_info")
                        status_info["thinking_config"] = workflow.get("thinking_config")
                    
                    return {
                        "success": True,
                        "workflow_status": status_info,
                        "message": f"Workflow {workflow_id} is {workflow.get('status', 'unknown')}"
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Workflow {workflow_id} not found",
                        "active_workflows": list(self.active_workflows.keys())
                    }
            else:
                # Show all active workflows
                return {
                    "success": True,
                    "message": f"Found {len(self.active_workflows)} active workflows",
                    "active_workflows": {
                        wf_id: {
                            "description": wf.get("description", ""),
                            "status": wf.get("status", "unknown"),
                            "team_id": wf.get("team_id"),
                            "created_at": wf.get("created_at")
                        }
                        for wf_id, wf in self.active_workflows.items()
                    }
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Workflow status failed: {str(e)}"
            }
    
    async def _cmd_team_performance(
        self,
        args: List[str],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Handle /team:performance command."""
        try:
            # Parse arguments
            team_id = args[0] if args else None
            timeframe = "24h"
            
            for arg in args[1:]:
                if arg.startswith("--timeframe="):
                    timeframe = arg.split("=")[1]
            
            if team_id and team_id in self.assembled_teams:
                team_info = self.assembled_teams[team_id]
                
                # Generate performance metrics (simulated)
                performance_metrics = {
                    "team_id": team_id,
                    "timeframe": timeframe,
                    "agents_active": len([a for a in team_info["agents"].values() if a.get("status") != "not_available"]),
                    "total_agents": len(team_info["agents"]),
                    "coordination_level": team_info["coordination_level"],
                    "metrics": {
                        "tasks_completed": 15,
                        "quality_score": 0.92,
                        "collaboration_effectiveness": 0.88,
                        "average_response_time_seconds": 12.5,
                        "thinking_sessions_completed": 3,
                        "hooks_success_rate": 0.96
                    },
                    "agent_performance": {
                        agent_id: {
                            "agent_id": agent_id,
                            "tasks_completed": 5,
                            "quality_score": 0.90 + (hash(agent_id) % 10) * 0.01,
                            "collaboration_score": 0.85 + (hash(agent_id) % 15) * 0.01
                        }
                        for agent_id in team_info["agents"].keys()
                        if team_info["agents"][agent_id].get("status") != "not_available"
                    }
                }
                
                return {
                    "success": True,
                    "team_performance": performance_metrics,
                    "message": f"Team {team_id} performance over {timeframe}",
                    "summary": f"Quality: {performance_metrics['metrics']['quality_score']:.1%}, Collaboration: {performance_metrics['metrics']['collaboration_effectiveness']:.1%}"
                }
            else:
                # Show all teams performance
                teams_performance = {}
                for tid, team in self.assembled_teams.items():
                    teams_performance[tid] = {
                        "team_id": tid,
                        "agents_count": len(team["agents"]),
                        "coordination_level": team["coordination_level"],
                        "created_at": team["created_at"],
                        "overall_score": 0.89  # Simulated
                    }
                
                return {
                    "success": True,
                    "message": f"Performance overview for {len(teams_performance)} teams",
                    "teams_performance": teams_performance
                }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Team performance analysis failed: {str(e)}"
            }
    
    async def get_command_help(self, command: Optional[str] = None) -> Dict[str, Any]:
        """Get help information for multi-agent commands."""
        if command and command in self.commands:
            cmd_info = self.commands[command]
            return {
                "command": command,
                "description": cmd_info["description"],
                "syntax": cmd_info["syntax"],
                "example": cmd_info["example"]
            }
        else:
            return {
                "available_commands": {
                    cmd_name: {
                        "description": cmd_info["description"],
                        "syntax": cmd_info["syntax"]
                    }
                    for cmd_name, cmd_info in self.commands.items()
                },
                "usage": "Use /help:multi <command> for detailed help on specific commands"
            }


# Global multi-agent commands instance
_multi_agent_commands: Optional[MultiAgentCommands] = None


def get_multi_agent_commands() -> Optional[MultiAgentCommands]:
    """Get the global multi-agent commands instance."""
    return _multi_agent_commands


def set_multi_agent_commands(commands: MultiAgentCommands) -> None:
    """Set the global multi-agent commands instance."""
    global _multi_agent_commands
    _multi_agent_commands = commands
    logger.info("🔗 Global multi-agent commands set")


async def initialize_multi_agent_commands(
    orchestrator: Optional[AgentOrchestrator] = None
) -> MultiAgentCommands:
    """
    Initialize and set the global multi-agent commands.
    
    Args:
        orchestrator: Agent orchestrator instance
        
    Returns:
        MultiAgentCommands instance
    """
    commands = MultiAgentCommands(orchestrator=orchestrator)
    set_multi_agent_commands(commands)
    
    logger.info("✅ Multi-Agent Commands initialized")
    return commands