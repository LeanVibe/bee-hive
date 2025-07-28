"""
Agent Lifecycle Manager for LeanVibe Agent Hive 2.0

Implements comprehensive agent lifecycle management including registration,
deregistration, task assignment, completion tracking, and health monitoring.
Integrates with persona system for intelligent task routing.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import json

import structlog
from sqlalchemy import select, update, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import cast
from sqlalchemy.types import Float

from .database import get_async_session
from .redis import get_redis, AgentMessageBroker
from .agent_persona_system import AgentPersonaSystem, PersonaAssignment
from .hook_lifecycle_system import HookLifecycleSystem, HookEvent, HookType
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.task import Task, TaskStatus, TaskPriority, TaskType
from ..models.persona import PersonaAssignmentModel

logger = structlog.get_logger()


class LifecycleEventType(str, Enum):
    """Types of agent lifecycle events."""
    AGENT_REGISTERED = "agent_registered"
    AGENT_DEREGISTERED = "agent_deregistered"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    AGENT_HEARTBEAT = "agent_heartbeat"
    AGENT_STATUS_CHANGED = "agent_status_changed"


@dataclass
class TaskAssignmentResult:
    """Result of task assignment operation."""
    success: bool
    agent_id: Optional[uuid.UUID] = None
    task_id: Optional[uuid.UUID] = None
    assignment_time: Optional[datetime] = None
    confidence_score: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class AgentRegistrationResult:
    """Result of agent registration operation."""
    success: bool
    agent_id: Optional[uuid.UUID] = None
    capabilities_assigned: List[str] = None
    persona_assigned: Optional[str] = None
    error_message: Optional[str] = None


class AgentLifecycleManager:
    """
    Manages the complete lifecycle of agents in the multi-agent system.
    
    This class coordinates between the agent persona system, task routing,
    Redis messaging, and database persistence to provide a unified
    agent lifecycle management interface.
    """
    
    def __init__(
        self,
        redis_client=None,
        persona_system: Optional[AgentPersonaSystem] = None,
        hook_system: Optional[HookLifecycleSystem] = None
    ):
        self.redis = redis_client or get_redis()
        self.message_broker = AgentMessageBroker(self.redis)
        self.persona_system = persona_system
        self.hook_system = hook_system
        
        # Performance tracking
        self.assignment_times: Dict[str, float] = {}
        self.active_agents: Set[uuid.UUID] = set()
        self.task_assignments: Dict[uuid.UUID, uuid.UUID] = {}  # task_id -> agent_id
        
        logger.info("🚀 Agent Lifecycle Manager initialized")
    
    async def register_agent(
        self,
        name: str,
        agent_type: AgentType = AgentType.CLAUDE,
        role: Optional[str] = None,
        capabilities: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tmux_session: Optional[str] = None
    ) -> AgentRegistrationResult:
        """
        Register a new agent in the system with persona-based capabilities.
        
        Args:
            name: Agent name
            agent_type: Type of agent (Claude, GPT, etc.)
            role: Agent role in the system
            capabilities: List of agent capabilities
            system_prompt: System prompt for the agent
            config: Agent configuration
            tmux_session: Associated tmux session
        
        Returns:
            AgentRegistrationResult with registration details
        """
        start_time = datetime.utcnow()
        
        try:
            async with get_async_session() as db:
                # Create agent in database
                agent = Agent(
                    name=name,
                    type=agent_type,
                    role=role,
                    capabilities=capabilities or [],
                    system_prompt=system_prompt,
                    config=config or {},
                    tmux_session=tmux_session,
                    status=AgentStatus.INITIALIZING
                )
                
                db.add(agent)
                await db.commit()
                await db.refresh(agent)
                
                # Update agent status to active
                agent.status = AgentStatus.ACTIVE
                agent.last_heartbeat = datetime.utcnow()
                agent.last_active = datetime.utcnow()
                await db.commit()
                
                # Add to active agents set
                self.active_agents.add(agent.id)
                
                # Assign persona if persona system is available
                persona_assigned = None
                if self.persona_system:
                    try:
                        persona_assignment = await self.persona_system.assign_optimal_persona(
                            agent_id=agent.id,
                            context={"role": role, "capabilities": capabilities}
                        )
                        if persona_assignment:
                            persona_assigned = persona_assignment.persona_type
                    except Exception as e:
                        logger.warning("Failed to assign persona", agent_id=str(agent.id), error=str(e))
                
                # Send lifecycle event via Redis
                await self._publish_lifecycle_event(
                    LifecycleEventType.AGENT_REGISTERED,
                    agent.id,
                    {
                        "name": name,
                        "type": agent_type.value,
                        "role": role,
                        "capabilities_count": len(capabilities or []),
                        "persona_assigned": persona_assigned,
                        "registration_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
                    }
                )
                
                # Emit hook event
                if self.hook_system:
                    hook_event = HookEvent(
                        hook_type=HookType.AGENT_START,
                        agent_id=agent.id,
                        session_id=None,
                        timestamp=datetime.utcnow(),
                        payload={
                            "action": "register",
                            "agent_name": name,
                            "agent_type": agent_type.value,
                            "capabilities": capabilities
                        }
                    )
                    await self.hook_system.process_hook_event(hook_event)
                
                logger.info(
                    "✅ Agent registered successfully",
                    agent_id=str(agent.id),
                    name=name,
                    role=role,
                    persona_assigned=persona_assigned,
                    registration_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
                )
                
                return AgentRegistrationResult(
                    success=True,
                    agent_id=agent.id,
                    capabilities_assigned=[cap.get("name") for cap in capabilities or []],
                    persona_assigned=persona_assigned
                )
                
        except Exception as e:
            logger.error("❌ Agent registration failed", name=name, error=str(e))
            return AgentRegistrationResult(
                success=False,
                error_message=str(e)
            )
    
    async def deregister_agent(self, agent_id: uuid.UUID) -> bool:
        """
        Deregister an agent from the system.
        
        Args:
            agent_id: ID of the agent to deregister
        
        Returns:
            True if successful, False otherwise
        """
        try:
            async with get_async_session() as db:
                # Get agent
                result = await db.execute(select(Agent).where(Agent.id == agent_id))
                agent = result.scalar_one_or_none()
                
                if not agent:
                    logger.warning("Agent not found for deregistration", agent_id=str(agent_id))
                    return False
                
                # Update agent status
                agent.status = AgentStatus.SHUTTING_DOWN
                await db.commit()
                
                # Remove from active agents
                self.active_agents.discard(agent_id)
                
                # Cancel any assigned tasks
                await self._cancel_agent_tasks(db, agent_id)
                
                # Send lifecycle event
                await self._publish_lifecycle_event(
                    LifecycleEventType.AGENT_DEREGISTERED,
                    agent_id,
                    {
                        "agent_name": agent.name,
                        "final_status": agent.status.value,
                        "total_tasks_completed": int(agent.total_tasks_completed or 0)
                    }
                )
                
                # Emit hook event
                if self.hook_system:
                    hook_event = HookEvent(
                        hook_type=HookType.AGENT_STOP,
                        agent_id=agent_id,
                        session_id=None,
                        timestamp=datetime.utcnow(),
                        payload={
                            "action": "deregister",
                            "agent_name": agent.name,
                            "reason": "normal_shutdown"
                        }
                    )
                    await self.hook_system.process_hook_event(hook_event)
                
                # Mark as inactive in database
                agent.status = AgentStatus.INACTIVE
                await db.commit()
                
                logger.info("✅ Agent deregistered", agent_id=str(agent_id), name=agent.name)
                return True
                
        except Exception as e:
            logger.error("❌ Agent deregistration failed", agent_id=str(agent_id), error=str(e))
            return False
    
    async def assign_task_to_agent(
        self,
        task_id: uuid.UUID,
        preferred_agent_id: Optional[uuid.UUID] = None,
        max_assignment_time_ms: float = 500.0
    ) -> TaskAssignmentResult:
        """
        Assign a task to the most suitable available agent.
        
        Args:
            task_id: ID of the task to assign
            preferred_agent_id: Preferred agent ID (optional)
            max_assignment_time_ms: Maximum assignment time in milliseconds
        
        Returns:
            TaskAssignmentResult with assignment details
        """
        start_time = datetime.utcnow()
        
        try:
            async with get_async_session() as db:
                # Get task details
                task_result = await db.execute(
                    select(Task).where(Task.id == task_id)
                )
                task = task_result.scalar_one_or_none()
                
                if not task:
                    return TaskAssignmentResult(
                        success=False,
                        error_message="Task not found"
                    )
                
                if task.status != TaskStatus.PENDING:
                    return TaskAssignmentResult(
                        success=False,
                        error_message=f"Task is not in PENDING status: {task.status}"
                    )
                
                # Find suitable agent
                suitable_agent = await self._find_suitable_agent(
                    db, task, preferred_agent_id
                )
                
                if not suitable_agent:
                    return TaskAssignmentResult(
                        success=False,
                        error_message="No suitable agent available"
                    )
                
                # Calculate confidence score
                confidence_score = suitable_agent.calculate_task_suitability(
                    task.task_type.value if task.task_type else "general",
                    task.required_capabilities or []
                )
                
                # Assign task to agent
                task.assign_to_agent(suitable_agent.id)
                suitable_agent.status = AgentStatus.BUSY
                await db.commit()
                
                # Track assignment
                self.task_assignments[task_id] = suitable_agent.id
                assignment_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.assignment_times[str(task_id)] = assignment_time_ms
                
                # Send task assignment message to agent
                await self.message_broker.send_message(
                    from_agent="orchestrator",
                    to_agent=str(suitable_agent.id),
                    message_type="task_assignment",
                    payload={
                        "task_id": str(task_id),
                        "task_title": task.title,
                        "task_type": task.task_type.value if task.task_type else "general",
                        "priority": task.priority.name.lower(),
                        "required_capabilities": task.required_capabilities or [],
                        "context": task.context or {},
                        "estimated_effort": task.estimated_effort
                    }
                )
                
                # Send lifecycle event
                await self._publish_lifecycle_event(
                    LifecycleEventType.TASK_ASSIGNED,
                    suitable_agent.id,
                    {
                        "task_id": str(task_id),
                        "task_title": task.title,
                        "task_type": task.task_type.value if task.task_type else "general",
                        "priority": task.priority.name.lower(),
                        "confidence_score": confidence_score,
                        "assignment_time_ms": assignment_time_ms
                    }
                )
                
                logger.info(
                    "✅ Task assigned successfully",
                    task_id=str(task_id),
                    agent_id=str(suitable_agent.id),
                    agent_name=suitable_agent.name,
                    confidence_score=confidence_score,
                    assignment_time_ms=assignment_time_ms
                )
                
                return TaskAssignmentResult(
                    success=True,
                    agent_id=suitable_agent.id,
                    task_id=task_id,
                    assignment_time=datetime.utcnow(),
                    confidence_score=confidence_score
                )
                
        except Exception as e:
            assignment_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.error(
                "❌ Task assignment failed",
                task_id=str(task_id),
                assignment_time_ms=assignment_time_ms,
                error=str(e)
            )
            return TaskAssignmentResult(
                success=False,
                error_message=str(e)
            )
    
    async def complete_task(
        self,
        task_id: uuid.UUID,
        agent_id: uuid.UUID,
        result: Dict[str, Any],
        success: bool = True
    ) -> bool:
        """
        Mark a task as completed and update agent status.
        
        Args:
            task_id: ID of the completed task
            agent_id: ID of the agent that completed the task
            result: Task execution result
            success: Whether the task completed successfully
        
        Returns:
            True if successful, False otherwise
        """
        try:
            async with get_async_session() as db:
                # Get task and agent
                task_result = await db.execute(select(Task).where(Task.id == task_id))
                task = task_result.scalar_one_or_none()
                
                agent_result = await db.execute(select(Agent).where(Agent.id == agent_id))
                agent = agent_result.scalar_one_or_none()
                
                if not task or not agent:
                    logger.warning("Task or agent not found", task_id=str(task_id), agent_id=str(agent_id))
                    return False
                
                if success:
                    # Mark task as completed
                    task.complete_successfully(result)
                    
                    # Update agent statistics
                    completed_count = int(agent.total_tasks_completed or 0) + 1
                    agent.total_tasks_completed = str(completed_count)
                    
                    event_type = LifecycleEventType.TASK_COMPLETED
                else:
                    # Mark task as failed
                    error_msg = result.get("error", "Task execution failed")
                    task.fail_with_error(error_msg)
                    
                    # Update agent failure statistics
                    failed_count = int(agent.total_tasks_failed or 0) + 1
                    agent.total_tasks_failed = str(failed_count)
                    
                    event_type = LifecycleEventType.TASK_FAILED
                
                # Update agent status back to active
                agent.status = AgentStatus.ACTIVE
                agent.last_active = datetime.utcnow()
                
                await db.commit()
                
                # Remove from tracking
                self.task_assignments.pop(task_id, None)
                
                # Calculate execution time if available
                execution_time_ms = None
                if task.started_at:
                    execution_time_ms = (datetime.utcnow() - task.started_at).total_seconds() * 1000
                
                # Send lifecycle event
                await self._publish_lifecycle_event(
                    event_type,
                    agent_id,
                    {
                        "task_id": str(task_id),
                        "task_title": task.title,
                        "success": success,
                        "execution_time_ms": execution_time_ms,
                        "result": result
                    }
                )
                
                logger.info(
                    "✅ Task completion processed",
                    task_id=str(task_id),
                    agent_id=str(agent_id),
                    success=success,
                    execution_time_ms=execution_time_ms
                )
                
                return True
                
        except Exception as e:
            logger.error(
                "❌ Task completion failed",
                task_id=str(task_id),
                agent_id=str(agent_id),
                error=str(e)
            )
            return False
    
    async def process_agent_heartbeat(self, agent_id: uuid.UUID, status_data: Dict[str, Any]) -> bool:
        """
        Process agent heartbeat and update status.
        
        Args:
            agent_id: ID of the agent
            status_data: Status information from the agent
        
        Returns:
            True if successful, False otherwise
        """
        try:
            async with get_async_session() as db:
                # Update agent heartbeat
                await db.execute(
                    update(Agent)
                    .where(Agent.id == agent_id)
                    .values(
                        last_heartbeat=datetime.utcnow(),
                        context_window_usage=str(status_data.get("context_usage", 0.0)),
                        average_response_time=str(status_data.get("avg_response_time", 0.0))
                    )
                )
                await db.commit()
                
                # Send heartbeat event
                await self._publish_lifecycle_event(
                    LifecycleEventType.AGENT_HEARTBEAT,
                    agent_id,
                    status_data
                )
                
                return True
                
        except Exception as e:
            logger.error("❌ Heartbeat processing failed", agent_id=str(agent_id), error=str(e))
            return False
    
    async def get_agent_status(self, agent_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get current agent status and metrics."""
        try:
            async with get_async_session() as db:
                result = await db.execute(
                    select(Agent).where(Agent.id == agent_id)
                )
                agent = result.scalar_one_or_none()
                
                if not agent:
                    return None
                
                # Get current task if any
                current_task = None
                if agent_id in [aid for aid in self.task_assignments.values()]:
                    task_id = next(tid for tid, aid in self.task_assignments.items() if aid == agent_id)
                    task_result = await db.execute(select(Task).where(Task.id == task_id))
                    task = task_result.scalar_one_or_none()
                    if task:
                        current_task = {
                            "id": str(task.id),
                            "title": task.title,
                            "status": task.status.value,
                            "started_at": task.started_at.isoformat() if task.started_at else None
                        }
                
                return {
                    "agent_id": str(agent.id),
                    "name": agent.name,
                    "status": agent.status.value,
                    "role": agent.role,
                    "capabilities": agent.capabilities,
                    "current_task": current_task,
                    "total_tasks_completed": int(agent.total_tasks_completed or 0),
                    "total_tasks_failed": int(agent.total_tasks_failed or 0),
                    "context_window_usage": float(agent.context_window_usage or 0.0),
                    "last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None,
                    "last_active": agent.last_active.isoformat() if agent.last_active else None
                }
                
        except Exception as e:
            logger.error("❌ Failed to get agent status", agent_id=str(agent_id), error=str(e))
            return None
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics for monitoring."""
        try:
            async with get_async_session() as db:
                # Get agent counts by status
                agent_counts = await db.execute(
                    select(Agent.status, func.count(Agent.id))
                    .group_by(Agent.status)
                )
                
                status_counts = {}
                for status, count in agent_counts:
                    status_counts[status.value] = count
                
                # Get task counts by status
                task_counts = await db.execute(
                    select(Task.status, func.count(Task.id))
                    .group_by(Task.status)
                )
                
                task_status_counts = {}
                for status, count in task_counts:
                    task_status_counts[status.value] = count
                
                # Calculate average assignment time
                avg_assignment_time = 0.0
                if self.assignment_times:
                    avg_assignment_time = sum(self.assignment_times.values()) / len(self.assignment_times)
                
                return {
                    "active_agents": len(self.active_agents),
                    "agent_status_counts": status_counts,
                    "task_status_counts": task_status_counts,
                    "active_task_assignments": len(self.task_assignments),
                    "average_assignment_time_ms": avg_assignment_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error("❌ Failed to get system metrics", error=str(e))
            return {"error": str(e)}
    
    async def _find_suitable_agent(
        self,
        db: AsyncSession,
        task: Task,
        preferred_agent_id: Optional[uuid.UUID] = None
    ) -> Optional[Agent]:
        """Find the most suitable agent for a task."""
        # If preferred agent is specified and available, use it
        if preferred_agent_id:
            result = await db.execute(
                select(Agent).where(
                    and_(
                        Agent.id == preferred_agent_id,
                        Agent.status == AgentStatus.ACTIVE,
                        or_(
                            Agent.context_window_usage.is_(None),
                            cast(Agent.context_window_usage, Float) < 0.8
                        )
                    )
                )
            )
            preferred_agent = result.scalar_one_or_none()
            if preferred_agent:
                return preferred_agent
        
        # Find available agents
        result = await db.execute(
            select(Agent).where(
                and_(
                    Agent.status == AgentStatus.ACTIVE,
                    or_(
                        Agent.context_window_usage.is_(None),
                        cast(Agent.context_window_usage, Float) < 0.8
                    )
                )
            )
        )
        available_agents = result.scalars().all()
        
        if not available_agents:
            return None
        
        # Score agents based on suitability
        best_agent = None
        best_score = 0.0
        
        for agent in available_agents:
            score = agent.calculate_task_suitability(
                task.task_type.value if task.task_type else "general",
                task.required_capabilities or []
            )
            
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent or available_agents[0]  # Fallback to first available
    
    async def _cancel_agent_tasks(self, db: AsyncSession, agent_id: uuid.UUID) -> None:
        """Cancel all tasks assigned to an agent."""
        # Get assigned tasks
        result = await db.execute(
            select(Task).where(
                and_(
                    Task.assigned_agent_id == agent_id,
                    Task.status.in_([TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS])
                )
            )
        )
        tasks = result.scalars().all()
        
        # Cancel tasks
        for task in tasks:
            task.status = TaskStatus.CANCELLED
            task.error_message = "Agent shutdown"
            self.task_assignments.pop(task.id, None)
        
        await db.commit()
    
    async def _publish_lifecycle_event(
        self,
        event_type: LifecycleEventType,
        agent_id: uuid.UUID,
        payload: Dict[str, Any]
    ) -> None:
        """Publish lifecycle event to Redis streams."""
        try:
            event_data = {
                "event_type": event_type.value,
                "agent_id": str(agent_id),
                "timestamp": datetime.utcnow().isoformat(),
                "payload": payload
            }
            
            # Publish to system events stream
            await self.redis.xadd(
                "system_events:agent_lifecycle",
                event_data,
                maxlen=10000
            )
            
            # Also publish to real-time pub/sub for dashboard
            await self.redis.publish(
                "realtime:agent_lifecycle",
                json.dumps(event_data)
            )
            
        except Exception as e:
            logger.error("Failed to publish lifecycle event", event_type=event_type, error=str(e))