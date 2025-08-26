"""
Unified Orchestrator for LeanVibe Agent Hive 2.0

Epic 5 Phase 1 - Critical Import Resolution: Unified orchestrator combining
simple_orchestrator capabilities with plugin architecture to resolve import
chain failures and make the system operational.

This file provides the missing orchestrator.py that main.py expects,
while leveraging the existing sophisticated SimpleOrchestrator implementation.
"""

import asyncio
import uuid
import time
from datetime import datetime
from typing import Protocol, Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

# Core imports
from .config import settings
from .logging_service import get_component_logger

# Import the existing sophisticated orchestrator
from .simple_orchestrator import (
    SimpleOrchestrator, 
    create_simple_orchestrator,
    create_enhanced_simple_orchestrator,
    AgentRole,
    AgentStatus,
    TaskPriority,
    SimpleOrchestratorError,
    AgentNotFoundError,
    TaskDelegationError,
    AgentInstance  # Add this for compatibility
)

logger = get_component_logger("orchestrator")


@dataclass
class AgentCapability:
    """Agent capability definition for compatibility with agent_spawner."""
    name: str
    description: str
    confidence_level: float = 0.8
    specialization_areas: List[str] = None

    def __post_init__(self):
        if self.specialization_areas is None:
            self.specialization_areas = []


class OrchestratorProtocol(Protocol):
    """Standard interface all orchestrators must implement."""
    
    async def register_agent(self, agent_spec: dict) -> str:
        """Register a new agent with the orchestrator."""
        ...
    
    async def delegate_task(self, task: dict) -> dict:
        """Delegate a task to an appropriate agent."""
        ...
    
    async def get_agent_status(self, agent_id: str) -> dict:
        """Get status of specific agent."""
        ...
    
    async def list_agents(self) -> List[dict]:
        """List all registered agents."""
        ...
    
    async def health_check(self) -> dict:
        """System health check."""
        ...


@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator behavior."""
    max_agents: int = 50
    task_timeout: int = 300
    plugin_dir: str = "app/core/orchestrator_plugins"
    enable_plugins: bool = True
    use_simple_orchestrator: bool = True
    enable_advanced_features: bool = True


class Orchestrator(OrchestratorProtocol):
    """
    Unified orchestrator combining simple_orchestrator capabilities with plugin architecture.
    
    This orchestrator acts as a facade/adapter over SimpleOrchestrator, providing the
    interface that main.py expects while leveraging all the sophisticated features
    already built in the system.
    
    Key Features:
    - Agent lifecycle management (spawn, shutdown, monitor)
    - Task delegation with intelligent routing
    - Plugin system for extensibility
    - Real-time status monitoring
    - Performance tracking and optimization
    - WebSocket broadcasting for real-time updates
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """Initialize the unified orchestrator."""
        self.config = config or OrchestratorConfig()
        self._simple_orchestrator: Optional[SimpleOrchestrator] = None
        self._initialized = False
        self._start_time = datetime.utcnow()
        
        # Performance tracking
        self._operations_count = 0
        self._last_performance_check = datetime.utcnow()
        
        logger.info("Unified Orchestrator initialized", 
                   max_agents=self.config.max_agents,
                   plugins_enabled=self.config.enable_plugins)
    
    async def initialize(self):
        """Initialize orchestrator and underlying components."""
        if self._initialized:
            return
        
        logger.info("🚀 Initializing Unified Orchestrator...")
        
        try:
            # Create and initialize the sophisticated SimpleOrchestrator
            if self.config.enable_advanced_features:
                # Use the enhanced version with tmux, Redis, and all features
                self._simple_orchestrator = await create_enhanced_simple_orchestrator()
                logger.info("✅ Enhanced SimpleOrchestrator initialized")
            else:
                # Use the basic version for minimal setups
                self._simple_orchestrator = create_simple_orchestrator()
                await self._simple_orchestrator.initialize()
                logger.info("✅ Basic SimpleOrchestrator initialized")
            
            self._initialized = True
            logger.info("✅ Unified Orchestrator initialization complete")
            
        except Exception as e:
            logger.error("❌ Failed to initialize Unified Orchestrator", error=str(e))
            raise
    
    async def register_agent(self, agent_spec: dict) -> str:
        """
        Register a new agent with the orchestrator.
        
        Args:
            agent_spec: Dictionary with agent configuration:
                - role: Agent role (backend_developer, frontend_developer, etc.)
                - agent_type: Optional agent type (defaults to claude_code)
                - workspace_name: Optional workspace name
                - git_branch: Optional git branch
                - environment_vars: Optional environment variables
        
        Returns:
            Agent ID
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Extract agent configuration
            role_str = agent_spec.get("role", "backend_developer")
            agent_type_str = agent_spec.get("agent_type", "claude_code")
            workspace_name = agent_spec.get("workspace_name")
            git_branch = agent_spec.get("git_branch")
            environment_vars = agent_spec.get("environment_vars", {})
            
            # Convert string role to AgentRole enum
            try:
                role = AgentRole(role_str)
            except ValueError:
                logger.warning("Invalid agent role, defaulting to backend_developer", 
                             role=role_str)
                role = AgentRole.BACKEND_DEVELOPER
            
            # Convert agent type
            from .simple_orchestrator import AgentLauncherType
            try:
                agent_type = AgentLauncherType(agent_type_str)
            except ValueError:
                logger.warning("Invalid agent type, defaulting to claude_code",
                             agent_type=agent_type_str)
                agent_type = AgentLauncherType.CLAUDE_CODE
            
            # Spawn agent using SimpleOrchestrator
            agent_id = await self._simple_orchestrator.spawn_agent(
                role=role,
                agent_type=agent_type,
                workspace_name=workspace_name,
                git_branch=git_branch,
                environment_vars=environment_vars
            )
            
            self._operations_count += 1
            
            # Record performance
            operation_time_ms = (time.time() - start_time) * 1000
            
            logger.info("Agent registered successfully",
                       agent_id=agent_id,
                       role=role_str,
                       agent_type=agent_type_str,
                       operation_time_ms=round(operation_time_ms, 2))
            
            return agent_id
            
        except Exception as e:
            logger.error("Failed to register agent", 
                        error=str(e), 
                        agent_spec=agent_spec)
            raise
    
    async def delegate_task(self, task: dict) -> dict:
        """
        Delegate a task to an appropriate agent.
        
        Args:
            task: Dictionary with task information:
                - description: Task description
                - task_type: Type of task
                - priority: Task priority (high, medium, low, urgent)
                - preferred_agent_role: Optional preferred agent role
        
        Returns:
            Dictionary with task delegation result
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Extract task information
            description = task.get("description", "")
            task_type = task.get("task_type", "general")
            priority_str = task.get("priority", "medium")
            preferred_role_str = task.get("preferred_agent_role")
            
            # Convert priority to TaskPriority enum
            try:
                priority = TaskPriority(priority_str.upper())
            except ValueError:
                logger.warning("Invalid task priority, defaulting to medium",
                             priority=priority_str)
                priority = TaskPriority.MEDIUM
            
            # Convert preferred role
            preferred_role = None
            if preferred_role_str:
                try:
                    preferred_role = AgentRole(preferred_role_str)
                except ValueError:
                    logger.warning("Invalid preferred agent role",
                                 preferred_role=preferred_role_str)
            
            # Delegate task using SimpleOrchestrator
            task_id = await self._simple_orchestrator.delegate_task(
                task_description=description,
                task_type=task_type,
                priority=priority,
                preferred_agent_role=preferred_role
            )
            
            self._operations_count += 1
            
            # Build response
            result = {
                "id": task_id,
                "description": description,
                "task_type": task_type,
                "priority": priority_str,
                "status": "assigned",
                "created_at": datetime.utcnow().isoformat(),
                "assigned_agent_role": preferred_role_str if preferred_role else None
            }
            
            # Record performance
            operation_time_ms = (time.time() - start_time) * 1000
            
            logger.info("Task delegated successfully",
                       task_id=task_id,
                       task_type=task_type,
                       priority=priority_str,
                       operation_time_ms=round(operation_time_ms, 2))
            
            return result
            
        except Exception as e:
            logger.error("Failed to delegate task",
                        error=str(e),
                        task=task)
            raise
    
    async def get_agent_status(self, agent_id: str) -> dict:
        """Get status of specific agent."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get detailed agent session info from SimpleOrchestrator
            session_info = await self._simple_orchestrator.get_agent_session_info(agent_id)
            
            if not session_info:
                raise AgentNotFoundError(f"Agent {agent_id} not found")
            
            # Format response
            agent_instance = session_info.get("agent_instance", {})
            launcher_status = session_info.get("launcher_status", {})
            
            status = {
                "id": agent_id,
                "role": agent_instance.get("role"),
                "status": agent_instance.get("status"),
                "created_at": agent_instance.get("created_at"),
                "last_activity": agent_instance.get("last_activity"),
                "current_task_id": agent_instance.get("current_task_id"),
                "session_info": {
                    "session_name": session_info.get("session_info", {}).get("session_name") if session_info.get("session_info") else None,
                    "workspace_path": launcher_status.get("workspace_path"),
                    "tmux_session_id": session_info.get("tmux_session_id")
                },
                "performance": launcher_status.get("performance", {}),
                "health": "healthy" if agent_instance.get("status") == "active" else "inactive"
            }
            
            return status
            
        except Exception as e:
            logger.error("Failed to get agent status",
                        agent_id=agent_id,
                        error=str(e))
            raise
    
    async def list_agents(self) -> List[dict]:
        """List all registered agents."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get all agent sessions from SimpleOrchestrator
            sessions = await self._simple_orchestrator.list_agent_sessions()
            
            agents = []
            for session in sessions:
                agent_instance = session.get("agent_instance", {})
                launcher_status = session.get("launcher_status", {})
                
                agent_info = {
                    "id": agent_instance.get("id"),
                    "role": agent_instance.get("role"),
                    "status": agent_instance.get("status"),
                    "created_at": agent_instance.get("created_at"),
                    "last_activity": agent_instance.get("last_activity"),
                    "current_task_id": agent_instance.get("current_task_id"),
                    "session_name": session.get("session_info", {}).get("session_name") if session.get("session_info") else None,
                    "workspace_path": launcher_status.get("workspace_path"),
                    "health": "healthy" if agent_instance.get("status") == "active" else "inactive"
                }
                agents.append(agent_info)
            
            return agents
            
        except Exception as e:
            logger.error("Failed to list agents", error=str(e))
            raise
    
    async def health_check(self) -> dict:
        """System health check."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Get enhanced system status from SimpleOrchestrator
            system_status = await self._simple_orchestrator.get_enhanced_system_status()
            
            # Calculate uptime
            uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()
            
            # Format health check response
            health = {
                "status": "healthy",
                "version": "2.0.0",
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": round(uptime_seconds, 2),
                "orchestrator_type": "UnifiedOrchestrator",
                "components": {
                    "simple_orchestrator": {
                        "status": "healthy",
                        "initialized": self._simple_orchestrator._initialized if self._simple_orchestrator else False,
                        "agents_count": system_status.get("agents", {}).get("total", 0),
                        "active_agents": system_status.get("agents", {}).get("by_status", {}).get("active", 0),
                        "tasks_count": system_status.get("tasks", {}).get("active_assignments", 0)
                    },
                    "tmux_integration": system_status.get("tmux_integration", {}),
                    "agent_launcher": system_status.get("agent_launcher", {}),
                    "redis_bridge": system_status.get("redis_bridge", {}),
                    "plugins": system_status.get("plugins", {})
                },
                "performance": {
                    "operations_count": self._operations_count,
                    "response_time_ms": system_status.get("performance", {}).get("response_time_ms", 0),
                    "operations_per_second": system_status.get("performance", {}).get("operations_per_second", 0),
                    "enhanced_agents": system_status.get("enhanced_agents", {}).get("total_with_sessions", 0)
                },
                "config": {
                    "max_agents": self.config.max_agents,
                    "plugins_enabled": self.config.enable_plugins,
                    "advanced_features": self.config.enable_advanced_features
                }
            }
            
            # Determine overall health status
            agents_count = system_status.get("agents", {}).get("total", 0)
            if agents_count == 0:
                health["status"] = "no_agents"
            
            # Check component health
            components_healthy = True
            for component_name, component_status in health["components"].items():
                if isinstance(component_status, dict) and component_status.get("status") == "unhealthy":
                    components_healthy = False
                    break
            
            if not components_healthy:
                health["status"] = "degraded"
            
            return health
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "version": "2.0.0", 
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "orchestrator_type": "UnifiedOrchestrator"
            }
    
    async def shutdown_agent(self, agent_id: str, graceful: bool = True) -> bool:
        """Shutdown a specific agent."""
        if not self._initialized:
            await self.initialize()
        
        try:
            return await self._simple_orchestrator.shutdown_agent(agent_id, graceful)
        except Exception as e:
            logger.error("Failed to shutdown agent",
                        agent_id=agent_id,
                        error=str(e))
            raise
    
    async def get_system_status(self) -> dict:
        """Get comprehensive system status."""
        return await self.health_check()
    
    async def shutdown(self) -> None:
        """Shutdown the orchestrator and all components."""
        logger.info("🛑 Shutting down Unified Orchestrator...")
        
        try:
            if self._simple_orchestrator:
                await self._simple_orchestrator.shutdown()
                logger.info("✅ SimpleOrchestrator shutdown complete")
            
            self._initialized = False
            logger.info("✅ Unified Orchestrator shutdown complete")
            
        except Exception as e:
            logger.error("Error during orchestrator shutdown", error=str(e))


# Global orchestrator instance
_orchestrator_instance: Optional[Orchestrator] = None


async def get_orchestrator(config: Optional[OrchestratorConfig] = None) -> Orchestrator:
    """Get global orchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator(config)
        await _orchestrator_instance.initialize()
    return _orchestrator_instance


def get_orchestrator_sync(config: Optional[OrchestratorConfig] = None) -> Orchestrator:
    """Get orchestrator instance for sync contexts (initialization deferred)."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator(config)
    return _orchestrator_instance


def set_orchestrator(orchestrator: Orchestrator) -> None:
    """Set global orchestrator instance (useful for testing)."""
    global _orchestrator_instance
    _orchestrator_instance = orchestrator


# Compatibility aliases for existing code
AgentOrchestrator = Orchestrator  # Legacy alias
create_orchestrator = get_orchestrator_sync  # Factory alias


# Export key classes and functions
__all__ = [
    "Orchestrator", 
    "OrchestratorProtocol",
    "OrchestratorConfig", 
    "get_orchestrator",
    "get_orchestrator_sync",
    "set_orchestrator",
    "AgentOrchestrator",  # Legacy alias
    "create_orchestrator",  # Legacy alias
    "AgentRole",
    "AgentStatus", 
    "TaskPriority",
    "AgentCapability",  # Add missing capability class
    "AgentInstance",    # Add missing instance class
    "SimpleOrchestratorError",
    "AgentNotFoundError",
    "TaskDelegationError"
]