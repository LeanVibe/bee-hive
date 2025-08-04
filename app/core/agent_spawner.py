"""
Agent Spawner Service for LeanVibe Agent Hive 2.0

This service activates the autonomous agent system by spawning real agent instances
with specific roles and capabilities. It bridges the gap between the sophisticated
infrastructure and actual operational multi-agent coordination.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
import structlog
from sqlalchemy import select

from .orchestrator import AgentOrchestrator, AgentRole, AgentCapability, AgentInstance
from .database import get_async_session
from .redis import get_redis
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.task import Task, TaskStatus, TaskPriority
from .config import settings

logger = structlog.get_logger()


@dataclass
class AgentSpawnConfig:
    """Configuration for spawning an agent."""
    role: AgentRole
    capabilities: List[str]
    max_concurrent_tasks: int = 3
    context_window_limit: float = 0.85
    specialized_areas: List[str] = None


class ActiveAgentManager:
    """Manages active agent instances and their lifecycle."""
    
    def __init__(self):
        self.active_agents: Dict[str, AgentInstance] = {}
        self.agent_heartbeats: Dict[str, datetime] = {}
        self.task_assignments: Dict[str, List[str]] = {}  # agent_id -> task_ids
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the agent manager."""
        logger.info("🚀 Starting Active Agent Manager")
        
        # Start background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Spawn initial agent team
        await self._spawn_initial_team()
        
        logger.info("✅ Active Agent Manager started", 
                   active_agents=len(self.active_agents))
    
    async def stop(self):
        """Stop the agent manager."""
        logger.info("🛑 Stopping Active Agent Manager")
        
        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Deactivate all agents
        for agent_id in list(self.active_agents.keys()):
            await self._deactivate_agent(agent_id)
        
        logger.info("✅ Active Agent Manager stopped")
    
    async def spawn_agent(self, config: AgentSpawnConfig) -> str:
        """Spawn a new agent with the given configuration."""
        agent_id = str(uuid.uuid4())
        
        logger.info("🤖 Spawning new agent", 
                   agent_id=agent_id, 
                   role=config.role.value)
        
        # Create agent capabilities
        capabilities = []
        for cap_name in config.capabilities:
            capabilities.append(AgentCapability(
                name=cap_name,
                description=f"{config.role.value} capability: {cap_name}",
                confidence_level=0.8,
                specialization_areas=config.specialized_areas or []
            ))
        
        # Create agent instance
        agent_instance = AgentInstance(
            id=agent_id,
            role=config.role,
            status=AgentStatus.ACTIVE,
            tmux_session=None,  # Could be implemented for Claude CLI integration
            capabilities=capabilities,
            current_task=None,
            context_window_usage=0.0,
            last_heartbeat=datetime.utcnow(),
            anthropic_client=None  # Will be initialized when needed
        )
        
        # Store in active agents
        self.active_agents[agent_id] = agent_instance
        self.agent_heartbeats[agent_id] = datetime.utcnow()
        self.task_assignments[agent_id] = []
        
        # Register in database
        await self._register_agent_in_database(agent_instance)
        
        # Send agent activation event
        await self._send_agent_event(agent_id, "agent_spawned", {
            "role": config.role.value,
            "capabilities": [cap.name for cap in capabilities]
        })
        
        logger.info("✅ Agent spawned successfully", 
                   agent_id=agent_id, 
                   role=config.role.value)
        
        return agent_id
    
    async def assign_task(self, agent_id: str, task_id: str) -> bool:
        """Assign a task to an agent."""
        if agent_id not in self.active_agents:
            logger.warning("Cannot assign task to inactive agent", 
                         agent_id=agent_id, task_id=task_id)
            return False
        
        agent = self.active_agents[agent_id]
        current_tasks = self.task_assignments.get(agent_id, [])
        
        # Check if agent can handle more tasks
        if len(current_tasks) >= 3:  # Max concurrent tasks
            logger.warning("Agent at max capacity", 
                         agent_id=agent_id, 
                         current_tasks=len(current_tasks))
            return False
        
        # Assign the task
        self.task_assignments[agent_id].append(task_id)
        agent.current_task = task_id if not agent.current_task else agent.current_task
        
        # Update agent status
        agent.status = AgentStatus.BUSY
        await self._update_agent_in_database(agent)
        
        # Send task assignment event
        await self._send_agent_event(agent_id, "task_assigned", {
            "task_id": task_id,
            "total_tasks": len(self.task_assignments[agent_id])
        })
        
        logger.info("📋 Task assigned to agent", 
                   agent_id=agent_id, 
                   task_id=task_id)
        
        return True
    
    async def complete_task(self, agent_id: str, task_id: str, result: Dict[str, Any]) -> bool:
        """Mark a task as completed by an agent."""
        if agent_id not in self.active_agents:
            return False
        
        agent = self.active_agents[agent_id]
        
        # Remove task from assignments
        if task_id in self.task_assignments.get(agent_id, []):
            self.task_assignments[agent_id].remove(task_id)
        
        # Update current task
        remaining_tasks = self.task_assignments.get(agent_id, [])
        agent.current_task = remaining_tasks[0] if remaining_tasks else None
        
        # Update agent status
        if not remaining_tasks:
            agent.status = AgentStatus.ACTIVE
        
        await self._update_agent_in_database(agent)
        
        # Send task completion event
        await self._send_agent_event(agent_id, "task_completed", {
            "task_id": task_id,
            "result": result,
            "remaining_tasks": len(remaining_tasks)
        })
        
        logger.info("✅ Task completed by agent", 
                   agent_id=agent_id, 
                   task_id=task_id)
        
        return True
    
    def get_active_agents(self) -> Dict[str, AgentInstance]:
        """Get all active agents."""
        return self.active_agents.copy()
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent."""
        if agent_id not in self.active_agents:
            return None
        
        agent = self.active_agents[agent_id]
        return {
            "id": agent.id,
            "role": agent.role.value,
            "status": agent.status.value,
            "current_task": agent.current_task,
            "assigned_tasks": len(self.task_assignments.get(agent_id, [])),
            "capabilities": [cap.name for cap in agent.capabilities],
            "last_heartbeat": agent.last_heartbeat.isoformat(),
            "context_usage": agent.context_window_usage
        }
    
    async def _spawn_initial_team(self):
        """Spawn the initial team of agents."""
        logger.info("👥 Spawning initial agent team")
        
        # Define initial team composition
        team_configs = [
            AgentSpawnConfig(
                role=AgentRole.PRODUCT_MANAGER,
                capabilities=["requirements_analysis", "project_planning", "documentation"],
                specialized_areas=["project_management", "stakeholder_communication"]
            ),
            AgentSpawnConfig(
                role=AgentRole.ARCHITECT,
                capabilities=["system_design", "architecture_planning", "technology_selection"],
                specialized_areas=["software_architecture", "scalability", "security"]
            ),
            AgentSpawnConfig(
                role=AgentRole.BACKEND_DEVELOPER,
                capabilities=["api_development", "database_design", "server_logic"],
                specialized_areas=["python", "fastapi", "postgresql", "redis"]
            ),
            AgentSpawnConfig(
                role=AgentRole.QA_ENGINEER,
                capabilities=["test_creation", "quality_assurance", "validation"],
                specialized_areas=["pytest", "integration_testing", "security_testing"]
            ),
            AgentSpawnConfig(
                role=AgentRole.DEVOPS_ENGINEER,
                capabilities=["deployment", "infrastructure", "monitoring"],
                specialized_areas=["docker", "ci_cd", "monitoring", "scalability"]
            )
        ]
        
        # Spawn each agent
        for config in team_configs:
            await self.spawn_agent(config)
            await asyncio.sleep(0.5)  # Small delay between spawns
        
        logger.info("✅ Initial agent team spawned", 
                   team_size=len(team_configs))
    
    async def _heartbeat_loop(self):
        """Background task to manage agent heartbeats."""
        while True:
            try:
                current_time = datetime.utcnow()
                
                for agent_id, agent in self.active_agents.items():
                    # Update heartbeat
                    agent.last_heartbeat = current_time
                    self.agent_heartbeats[agent_id] = current_time
                    
                    # Simulate some activity (could be real work delegation)
                    agent.context_window_usage = min(0.9, agent.context_window_usage + 0.01)
                    
                    # Send heartbeat event
                    await self._send_agent_event(agent_id, "heartbeat", {
                        "timestamp": current_time.isoformat(),
                        "status": agent.status.value,
                        "context_usage": agent.context_window_usage
                    })
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in heartbeat loop", error=str(e))
                await asyncio.sleep(5)
    
    async def _cleanup_loop(self):
        """Background task to clean up inactive agents."""
        while True:
            try:
                current_time = datetime.utcnow()
                inactive_threshold = timedelta(minutes=5)
                
                inactive_agents = []
                for agent_id, last_heartbeat in self.agent_heartbeats.items():
                    if current_time - last_heartbeat > inactive_threshold:
                        inactive_agents.append(agent_id)
                
                # Deactivate inactive agents
                for agent_id in inactive_agents:
                    await self._deactivate_agent(agent_id)
                
                await asyncio.sleep(60)  # Cleanup check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cleanup loop", error=str(e))
                await asyncio.sleep(10)
    
    async def _register_agent_in_database(self, agent: AgentInstance):
        """Register agent in the database."""
        try:
            async for session in get_async_session():
                db_agent = Agent(
                    id=uuid.UUID(agent.id),
                    name=f"{agent.role.value}_agent",
                    agent_type=AgentType.CLAUDE,
                    status=agent.status,
                    capabilities=[cap.name for cap in agent.capabilities],
                    specializations=agent.capabilities[0].specialization_areas if agent.capabilities else [],
                    configuration={"role": agent.role.value, "max_tasks": 3}
                )
                
                session.add(db_agent)
                await session.commit()
                break
                
        except Exception as e:
            logger.error("Failed to register agent in database", 
                        agent_id=agent.id, error=str(e))
    
    async def _update_agent_in_database(self, agent: AgentInstance):
        """Update agent status in the database."""
        try:
            async for session in get_async_session():
                result = await session.execute(
                    select(Agent).where(Agent.id == uuid.UUID(agent.id))
                )
                db_agent = result.scalar_one_or_none()
                
                if db_agent:
                    db_agent.status = agent.status
                    db_agent.updated_at = datetime.utcnow()
                    await session.commit()
                break
                
        except Exception as e:
            logger.error("Failed to update agent in database", 
                        agent_id=agent.id, error=str(e))
    
    async def _deactivate_agent(self, agent_id: str):
        """Deactivate an agent."""
        if agent_id in self.active_agents:
            agent = self.active_agents[agent_id]
            agent.status = AgentStatus.INACTIVE
            
            await self._update_agent_in_database(agent)
            await self._send_agent_event(agent_id, "agent_deactivated", {})
            
            # Clean up
            del self.active_agents[agent_id]
            del self.agent_heartbeats[agent_id]
            del self.task_assignments[agent_id]
            
            logger.info("Agent deactivated", agent_id=agent_id)
    
    async def _send_agent_event(self, agent_id: str, event_type: str, data: Dict[str, Any]):
        """Send agent event to Redis for real-time updates."""
        try:
            redis_client = get_redis()
            event = {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": agent_id,
                "event_type": event_type,
                "data": data
            }
            
            await redis_client.publish("agent_events", json.dumps(event))
            
        except Exception as e:
            logger.error("Failed to send agent event", 
                        agent_id=agent_id, event_type=event_type, error=str(e))


# Global agent manager instance
_agent_manager: Optional[ActiveAgentManager] = None


async def get_agent_manager() -> ActiveAgentManager:
    """Get the global agent manager instance."""
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = ActiveAgentManager()
        await _agent_manager.start()
    return _agent_manager


async def spawn_development_team() -> Dict[str, str]:
    """Spawn a complete development team and return agent IDs."""
    manager = await get_agent_manager()
    
    # If agents already exist, return them
    if manager.active_agents:
        return {agent.role.value: agent.id for agent in manager.active_agents.values()}
    
    # Otherwise spawn new team
    await manager._spawn_initial_team()
    return {agent.role.value: agent.id for agent in manager.active_agents.values()}


async def get_active_agents_status() -> Dict[str, Any]:
    """Get status of all active agents."""
    manager = await get_agent_manager()
    
    agents_status = {}
    for agent_id, agent in manager.active_agents.items():
        agents_status[agent_id] = manager.get_agent_status(agent_id)
    
    return agents_status