"""
Simple Orchestrator for LeanVibe Agent Hive 2.0

A clean, minimal orchestrator interface that provides the core 20% functionality
for agent management and task delegation. Designed for <100ms response times
and easy testing/maintenance.

Follows YAGNI principles - only implements what's needed now.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Protocol
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from anthropic import AsyncAnthropic

from .config import settings
from .database import get_session
from .logging_service import get_component_logger
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.task import Task, TaskStatus, TaskPriority

logger = get_component_logger("simple_orchestrator")


class AgentRole(Enum):
    """Core agent roles for the simplified orchestrator."""
    BACKEND_DEVELOPER = "backend_developer"
    FRONTEND_DEVELOPER = "frontend_developer"
    DEVOPS_ENGINEER = "devops_engineer"
    QA_ENGINEER = "qa_engineer"


@dataclass
class AgentInstance:
    """Simplified agent instance representation."""
    id: str
    role: AgentRole
    status: AgentStatus
    current_task_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "role": self.role.value,
            "status": self.status.value,
            "current_task_id": self.current_task_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat()
        }


@dataclass
class TaskAssignment:
    """Simple task assignment representation."""
    task_id: str
    agent_id: str
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    status: TaskStatus = TaskStatus.PENDING
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "assigned_at": self.assigned_at.isoformat(),
            "status": self.status.value
        }


class DatabaseDependency(Protocol):
    """Protocol for database dependency injection."""
    async def get_session(self) -> AsyncSession: ...


class CacheDependency(Protocol):
    """Protocol for cache dependency injection."""
    async def get(self, key: str) -> Optional[Any]: ...
    async def set(self, key: str, value: Any, ttl: int = 300) -> bool: ...


class SimpleOrchestratorError(Exception):
    """Base exception for orchestrator errors."""
    pass


class AgentNotFoundError(SimpleOrchestratorError):
    """Raised when agent is not found."""
    pass


class TaskDelegationError(SimpleOrchestratorError):
    """Raised when task delegation fails."""
    pass


class SimpleOrchestrator:
    """
    Simple, fast orchestrator focused on core agent management operations.
    
    Provides:
    - Agent lifecycle management (spawn, shutdown)
    - Basic task delegation
    - System status monitoring
    
    Design goals:
    - <100ms response times for core operations
    - Easy to test and maintain
    - Clear separation of concerns
    - Minimal dependencies
    """
    
    def __init__(
        self,
        db_session_factory: Optional[DatabaseDependency] = None,
        cache: Optional[CacheDependency] = None,
        anthropic_client: Optional[AsyncAnthropic] = None
    ):
        """Initialize orchestrator with dependency injection."""
        self._db_session_factory = db_session_factory
        self._cache = cache
        self._anthropic_client = anthropic_client or AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY
        )
        
        # In-memory agent registry for fast access
        self._agents: Dict[str, AgentInstance] = {}
        self._task_assignments: Dict[str, TaskAssignment] = {}
        
        # Performance metrics
        self._operation_count = 0
        self._last_performance_check = datetime.utcnow()
    
    async def spawn_agent(
        self,
        role: AgentRole,
        agent_id: Optional[str] = None
    ) -> str:
        """
        Spawn a new agent instance.
        
        Args:
            role: The role for the new agent
            agent_id: Optional specific ID, otherwise generated
            
        Returns:
            The agent ID
            
        Raises:
            SimpleOrchestratorError: If spawning fails
        """
        start_time = datetime.utcnow()
        
        try:
            # Generate ID if not provided
            if agent_id is None:
                agent_id = str(uuid.uuid4())
            
            # Check if agent already exists
            if agent_id in self._agents:
                raise SimpleOrchestratorError(f"Agent {agent_id} already exists")
            
            # Check agent limit
            active_agents = len([a for a in self._agents.values() 
                               if a.status == AgentStatus.ACTIVE])
            if active_agents >= getattr(settings, 'MAX_CONCURRENT_AGENTS', 10):
                raise SimpleOrchestratorError("Maximum concurrent agents reached")
            
            # Create agent instance
            agent = AgentInstance(
                id=agent_id,
                role=role,
                status=AgentStatus.ACTIVE
            )
            
            # Store in memory registry
            self._agents[agent_id] = agent
            
            # Persist to database if available
            if self._db_session_factory:
                await self._persist_agent(agent)
            
            # Cache for fast access
            if self._cache:
                await self._cache.set(f"agent:{agent_id}", agent.to_dict(), ttl=3600)
            
            logger.info(
                "Agent spawned successfully",
                agent_id=agent_id,
                role=role.value,
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
            self._operation_count += 1
            return agent_id
            
        except Exception as e:
            logger.error(
                "Failed to spawn agent",
                agent_id=agent_id,
                role=role.value if role else None,
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
            raise SimpleOrchestratorError(f"Failed to spawn agent: {e}") from e
    
    async def shutdown_agent(self, agent_id: str, graceful: bool = True) -> bool:
        """
        Shutdown a specific agent instance.
        
        Args:
            agent_id: ID of agent to shutdown
            graceful: Whether to wait for current task completion
            
        Returns:
            True if shutdown successful, False if agent not found
            
        Raises:
            SimpleOrchestratorError: If shutdown fails
        """
        start_time = datetime.utcnow()
        
        try:
            # Check if agent exists
            if agent_id not in self._agents:
                logger.warning("Agent not found for shutdown", agent_id=agent_id)
                return False
            
            agent = self._agents[agent_id]
            
            # Handle graceful shutdown
            if graceful and agent.current_task_id:
                logger.info(
                    "Graceful shutdown - waiting for task completion",
                    agent_id=agent_id,
                    task_id=agent.current_task_id
                )
                # Wait for current task (simplified - could be enhanced)
                await asyncio.sleep(1)  # Brief wait
            
            # Update agent status
            agent.status = AgentStatus.INACTIVE
            agent.last_activity = datetime.utcnow()
            
            # Remove from active registry
            del self._agents[agent_id]
            
            # Update database if available
            if self._db_session_factory:
                await self._update_agent_status(agent_id, AgentStatus.INACTIVE)
            
            # Remove from cache
            if self._cache:
                await self._cache.set(f"agent:{agent_id}", None, ttl=1)
            
            logger.info(
                "Agent shutdown successful",
                agent_id=agent_id,
                graceful=graceful,
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
            self._operation_count += 1
            return True
            
        except Exception as e:
            logger.error(
                "Failed to shutdown agent",
                agent_id=agent_id,
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
            raise SimpleOrchestratorError(f"Failed to shutdown agent: {e}") from e
    
    async def delegate_task(
        self,
        task_description: str,
        task_type: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        preferred_agent_role: Optional[AgentRole] = None
    ) -> str:
        """
        Delegate a task to the most suitable available agent.
        
        Args:
            task_description: Description of the task
            task_type: Type/category of the task
            priority: Task priority level
            preferred_agent_role: Preferred agent role for the task
            
        Returns:
            Task ID
            
        Raises:
            TaskDelegationError: If no suitable agent available
        """
        start_time = datetime.utcnow()
        
        try:
            # Generate task ID
            task_id = str(uuid.uuid4())
            
            # Find suitable agent
            suitable_agent = await self._find_suitable_agent(
                preferred_role=preferred_agent_role,
                task_type=task_type
            )
            
            if not suitable_agent:
                raise TaskDelegationError("No suitable agent available")
            
            # Create task assignment
            assignment = TaskAssignment(
                task_id=task_id,
                agent_id=suitable_agent.id,
                status=TaskStatus.PENDING
            )
            
            # Update agent with current task
            suitable_agent.current_task_id = task_id
            suitable_agent.last_activity = datetime.utcnow()
            
            # Store assignment
            self._task_assignments[task_id] = assignment
            
            # Persist to database if available
            if self._db_session_factory:
                await self._persist_task(task_id, task_description, task_type, 
                                       priority, suitable_agent.id)
            
            logger.info(
                "Task delegated successfully",
                task_id=task_id,
                agent_id=suitable_agent.id,
                task_type=task_type,
                priority=priority.value,
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
            self._operation_count += 1
            return task_id
            
        except Exception as e:
            logger.error(
                "Failed to delegate task",
                task_description=task_description[:100],
                task_type=task_type,
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
            raise TaskDelegationError(f"Failed to delegate task: {e}") from e
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status for monitoring.
        
        Returns:
            Dictionary with system status information
        """
        start_time = datetime.utcnow()
        
        try:
            # Collect agent statuses
            agent_statuses = {
                agent_id: agent.to_dict() 
                for agent_id, agent in self._agents.items()
            }
            
            # Count agents by status
            status_counts = {}
            for agent in self._agents.values():
                status = agent.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Task assignment summary
            assignment_count = len(self._task_assignments)
            
            # Performance metrics
            now = datetime.utcnow()
            time_since_check = (now - self._last_performance_check).total_seconds()
            ops_per_second = self._operation_count / max(time_since_check, 1)
            
            status = {
                "timestamp": now.isoformat(),
                "agents": {
                    "total": len(self._agents),
                    "by_status": status_counts,
                    "details": agent_statuses
                },
                "tasks": {
                    "active_assignments": assignment_count
                },
                "performance": {
                    "operations_count": self._operation_count,
                    "operations_per_second": round(ops_per_second, 2),
                    "response_time_ms": round(
                        (datetime.utcnow() - start_time).total_seconds() * 1000, 2
                    )
                },
                "health": "healthy" if len(self._agents) > 0 else "no_agents"
            }
            
            logger.debug(
                "System status retrieved",
                agent_count=len(self._agents),
                task_count=assignment_count,
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
            
            return status
            
        except Exception as e:
            logger.error("Failed to get system status", error=str(e))
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "health": "error"
            }
    
    # Private helper methods
    
    async def _find_suitable_agent(
        self,
        preferred_role: Optional[AgentRole] = None,
        task_type: str = ""
    ) -> Optional[AgentInstance]:
        """Find the most suitable available agent for a task."""
        available_agents = [
            agent for agent in self._agents.values()
            if agent.status == AgentStatus.ACTIVE and agent.current_task_id is None
        ]
        
        if not available_agents:
            return None
        
        # Prefer agents with matching role
        if preferred_role:
            role_matches = [a for a in available_agents if a.role == preferred_role]
            if role_matches:
                return role_matches[0]  # Simple selection - could be enhanced
        
        # Return first available agent
        return available_agents[0]
    
    async def _persist_agent(self, agent: AgentInstance) -> None:
        """Persist agent to database."""
        if not self._db_session_factory:
            return
            
        try:
            async with get_session() as session:
                db_agent = Agent(
                    id=agent.id,
                    role=agent.role.value,
                    agent_type=AgentType.CLAUDE_CODE,
                    status=agent.status,
                    created_at=agent.created_at
                )
                session.add(db_agent)
                await session.commit()
        except Exception as e:
            logger.warning("Failed to persist agent to database", 
                         agent_id=agent.id, error=str(e))
    
    async def _update_agent_status(self, agent_id: str, status: AgentStatus) -> None:
        """Update agent status in database."""
        if not self._db_session_factory:
            return
            
        try:
            async with get_session() as session:
                await session.execute(
                    update(Agent)
                    .where(Agent.id == agent_id)
                    .values(status=status, updated_at=datetime.utcnow())
                )
                await session.commit()
        except Exception as e:
            logger.warning("Failed to update agent status in database",
                         agent_id=agent_id, error=str(e))
    
    async def _persist_task(
        self,
        task_id: str,
        description: str,
        task_type: str,
        priority: TaskPriority,
        agent_id: str
    ) -> None:
        """Persist task to database."""
        if not self._db_session_factory:
            return
            
        try:
            async with get_session() as session:
                db_task = Task(
                    id=task_id,
                    description=description,
                    task_type=task_type,
                    priority=priority,
                    status=TaskStatus.PENDING,
                    assigned_agent_id=agent_id,
                    created_at=datetime.utcnow()
                )
                session.add(db_task)
                await session.commit()
        except Exception as e:
            logger.warning("Failed to persist task to database",
                         task_id=task_id, error=str(e))


# Factory function for dependency injection
def create_simple_orchestrator(
    db_session_factory: Optional[DatabaseDependency] = None,
    cache: Optional[CacheDependency] = None,
    anthropic_client: Optional[AsyncAnthropic] = None
) -> SimpleOrchestrator:
    """
    Factory function to create SimpleOrchestrator with proper dependencies.
    
    This makes it easy to inject dependencies for testing and different environments.
    """
    return SimpleOrchestrator(
        db_session_factory=db_session_factory,
        cache=cache,
        anthropic_client=anthropic_client
    )


# Global instance for API usage (can be overridden in tests)
_global_orchestrator: Optional[SimpleOrchestrator] = None


def get_simple_orchestrator() -> SimpleOrchestrator:
    """Get the global orchestrator instance."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = create_simple_orchestrator()
    return _global_orchestrator


def set_simple_orchestrator(orchestrator: SimpleOrchestrator) -> None:
    """Set the global orchestrator instance (useful for testing)."""
    global _global_orchestrator
    _global_orchestrator = orchestrator