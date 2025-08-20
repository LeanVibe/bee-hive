"""
Simple Orchestrator Adapter for LeanVibe Agent Hive 2.0

Provides basic compatibility layer for existing code during the orchestrator consolidation.
This is a temporary adapter to ensure system continues working while migration is completed.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from enum import Enum

from .unified_orchestrator import (
    UnifiedOrchestrator, 
    get_orchestrator,
    OrchestratorConfig,
    OrchestratorMode,
    AgentRole as UnifiedAgentRole
)
from ..models.agent import AgentStatus, AgentType
from ..models.task import Task, TaskStatus, TaskPriority
from .logging_service import get_component_logger

logger = get_component_logger("orchestrator_adapter")


class AgentRole(Enum):
    """Legacy agent role compatibility"""
    STRATEGIC_PARTNER = "strategic_partner"
    PRODUCT_MANAGER = "product_manager" 
    ARCHITECT = "architect"
    BACKEND_DEVELOPER = "backend_developer"
    FRONTEND_DEVELOPER = "frontend_developer"
    DEVOPS_ENGINEER = "devops_engineer"
    QA_ENGINEER = "qa_engineer"
    META_AGENT = "meta_agent"


class AgentInstance:
    """Legacy AgentInstance compatibility wrapper"""
    def __init__(self, id: str, role: AgentRole, status: AgentStatus,
                 tmux_session: Optional[str] = None,
                 capabilities: Optional[List[Any]] = None,
                 current_task: Optional[str] = None,
                 context_window_usage: float = 0.0,
                 last_heartbeat: Optional[datetime] = None,
                 anthropic_client: Optional[Any] = None):
        self.id = id
        self.role = role
        self.status = status
        self.tmux_session = tmux_session
        self.capabilities = capabilities or []
        self.current_task = current_task
        self.context_window_usage = context_window_usage
        self.last_heartbeat = last_heartbeat or datetime.utcnow()
        self.anthropic_client = anthropic_client
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility"""
        return {
            'id': self.id,
            'role': self.role.value,
            'status': self.status.value,
            'tmux_session': self.tmux_session,
            'capabilities': self.capabilities,
            'current_task': self.current_task,
            'context_window_usage': self.context_window_usage,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'anthropic_client': None  # Don't serialize the client
        }


class AgentOrchestrator:
    """
    Simple compatibility wrapper for legacy AgentOrchestrator.
    
    This provides basic functionality to keep existing code working
    while the full migration to unified orchestrator is completed.
    """
    
    def __init__(self):
        self._unified_orchestrator: Optional[UnifiedOrchestrator] = None
        self._legacy_agents: Dict[str, AgentInstance] = {}
        self._initialized = False
        
        logger.info("Simple AgentOrchestrator adapter initialized")
    
    async def _ensure_initialized(self):
        """Ensure the unified orchestrator is initialized"""
        if not self._initialized:
            config = OrchestratorConfig(
                mode=OrchestratorMode.PRODUCTION,
                max_agents=50,
                auto_scaling_enabled=True
            )
            self._unified_orchestrator = await get_orchestrator(config)
            self._initialized = True
    
    async def start(self):
        """Start the orchestrator"""
        await self._ensure_initialized()
        logger.info("Legacy AgentOrchestrator started")
    
    async def stop(self):
        """Stop the orchestrator"""
        if self._unified_orchestrator:
            await self._unified_orchestrator.cleanup()
        logger.info("Legacy AgentOrchestrator stopped")
    
    def _map_legacy_role_to_unified(self, role: AgentRole) -> UnifiedAgentRole:
        """Map legacy role to unified role"""
        role_mapping = {
            AgentRole.STRATEGIC_PARTNER: UnifiedAgentRole.COORDINATOR,
            AgentRole.PRODUCT_MANAGER: UnifiedAgentRole.COORDINATOR,
            AgentRole.ARCHITECT: UnifiedAgentRole.SPECIALIST,
            AgentRole.BACKEND_DEVELOPER: UnifiedAgentRole.SPECIALIST,
            AgentRole.FRONTEND_DEVELOPER: UnifiedAgentRole.SPECIALIST,
            AgentRole.DEVOPS_ENGINEER: UnifiedAgentRole.SPECIALIST,
            AgentRole.QA_ENGINEER: UnifiedAgentRole.WORKER,
            AgentRole.META_AGENT: UnifiedAgentRole.COORDINATOR
        }
        return role_mapping.get(role, UnifiedAgentRole.WORKER)
    
    async def spawn_agent(self, role: AgentRole, capabilities: Optional[List[Any]] = None,
                         tmux_session: Optional[str] = None,
                         anthropic_client: Optional[Any] = None) -> str:
        """Spawn new agent (legacy compatibility)"""
        await self._ensure_initialized()
        
        try:
            # Map to unified orchestrator
            unified_role = self._map_legacy_role_to_unified(role)
            capabilities_set = set(capabilities) if capabilities else set()
            
            agent_id = await self._unified_orchestrator.spawn_agent(
                agent_type=AgentType.ANTHROPIC_CLAUDE,
                role=unified_role,
                capabilities=capabilities_set
            )
            
            # Create legacy wrapper
            legacy_agent = AgentInstance(
                id=agent_id,
                role=role,
                status=AgentStatus.ACTIVE,
                tmux_session=tmux_session,
                capabilities=capabilities,
                anthropic_client=anthropic_client
            )
            
            self._legacy_agents[agent_id] = legacy_agent
            
            logger.info(f"Spawned legacy agent {agent_id} with role {role.value}")
            return agent_id
            
        except Exception as e:
            logger.error(f"Failed to spawn agent: {e}")
            raise
    
    async def delegate_task(self, task: Any, agent_id: Optional[str] = None,
                           priority: Optional[TaskPriority] = None) -> str:
        """Delegate task to agent (legacy compatibility)"""
        await self._ensure_initialized()
        
        try:
            # Convert to Task object if needed
            if not isinstance(task, Task):
                # Create a basic Task wrapper
                task_obj = Task(
                    id=f"task_{int(datetime.utcnow().timestamp())}",
                    task_type="legacy_task",
                    description=str(task),
                    priority=priority or TaskPriority.NORMAL,
                    status=TaskStatus.PENDING
                )
            else:
                task_obj = task
            
            # Delegate through unified orchestrator
            task_id = await self._unified_orchestrator.delegate_task(
                task=task_obj,
                preferred_agent_id=agent_id
            )
            
            # Update legacy agent if specified
            if agent_id and agent_id in self._legacy_agents:
                self._legacy_agents[agent_id].current_task = task_id
                self._legacy_agents[agent_id].status = AgentStatus.BUSY
            
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to delegate task: {e}")
            raise
    
    def get_agent(self, agent_id: str) -> Optional[AgentInstance]:
        """Get agent by ID (legacy compatibility)"""
        return self._legacy_agents.get(agent_id)
    
    def get_all_agents(self) -> List[AgentInstance]:
        """Get all agents (legacy compatibility)"""
        return list(self._legacy_agents.values())
    
    def get_active_agents(self) -> List[AgentInstance]:
        """Get active agents (legacy compatibility)"""
        return [agent for agent in self._legacy_agents.values() 
                if agent.status in [AgentStatus.ACTIVE, AgentStatus.BUSY]]
    
    async def remove_agent(self, agent_id: str) -> bool:
        """Remove agent (legacy compatibility)"""
        await self._ensure_initialized()
        
        try:
            # Remove from unified orchestrator
            await self._unified_orchestrator._remove_agent(agent_id)
            
            # Remove from legacy tracking
            if agent_id in self._legacy_agents:
                del self._legacy_agents[agent_id]
                
            logger.info(f"Removed agent {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove agent {agent_id}: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Get health status (legacy compatibility)"""
        await self._ensure_initialized()
        
        if self._unified_orchestrator:
            status = await self._unified_orchestrator.get_status()
            
            # Add legacy-specific information
            status["legacy_agents"] = {
                "total": len(self._legacy_agents),
                "active": len(self.get_active_agents())
            }
            
            return status
        else:
            return {
                "status": "not_initialized",
                "legacy_agents": {"total": 0, "active": 0}
            }
    
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get agent status (legacy compatibility)"""
        agent = self.get_agent(agent_id)
        if agent:
            return agent.to_dict()
        else:
            return {"error": f"Agent {agent_id} not found"}
    
    async def update_agent_status(self, agent_id: str, status: AgentStatus):
        """Update agent status (legacy compatibility)"""
        if agent_id in self._legacy_agents:
            self._legacy_agents[agent_id].status = status
            self._legacy_agents[agent_id].last_heartbeat = datetime.utcnow()
            
    async def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics (legacy compatibility)"""
        await self._ensure_initialized()
        
        if self._unified_orchestrator:
            return self._unified_orchestrator.metrics.copy()
        else:
            return {
                "tasks_completed": 0,
                "tasks_failed": 0,
                "agents_spawned": len(self._legacy_agents),
                "agents_terminated": 0
            }


# Global instance for compatibility
_global_orchestrator: Optional[AgentOrchestrator] = None


def get_agent_orchestrator() -> AgentOrchestrator:
    """Get global orchestrator instance (legacy compatibility)"""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = AgentOrchestrator()
    return _global_orchestrator


async def initialize_orchestrator():
    """Initialize global orchestrator (legacy compatibility)"""
    orchestrator = get_agent_orchestrator()
    await orchestrator.start()
    return orchestrator


async def shutdown_orchestrator():
    """Shutdown global orchestrator (legacy compatibility)"""
    global _global_orchestrator
    if _global_orchestrator:
        await _global_orchestrator.stop()
        _global_orchestrator = None