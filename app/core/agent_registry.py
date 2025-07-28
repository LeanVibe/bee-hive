"""
Agent Registry Service for LeanVibe Agent Hive 2.0

Provides enhanced CRUD operations and lifecycle management for AI agents
in the orchestration system. Handles agent registration, deregistration,
status updates, and resource tracking with comprehensive error handling.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.orm import selectinload

from .database import get_session
from .redis import get_message_broker, AgentMessageBroker
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.task import Task, TaskStatus

logger = structlog.get_logger()


class LifecycleState(Enum):
    """Enhanced agent lifecycle states for orchestrator core."""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    SLEEPING = "sleeping"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"


@dataclass
class AgentRegistrationResult:
    """Result of agent registration operation."""
    success: bool
    agent_id: Optional[uuid.UUID]
    error_message: Optional[str]
    registration_time: Optional[datetime]
    capabilities_assigned: List[str]
    health_score: float


@dataclass
class AgentResourceUsage:
    """Resource usage metrics for an agent."""
    memory_mb: float
    cpu_percent: float
    context_window_usage: float
    active_tasks_count: int
    last_updated: datetime


class AgentRegistry:
    """
    Enhanced agent registry service for orchestrator core.
    
    Provides comprehensive agent management with lifecycle tracking,
    resource monitoring, and performance metrics collection.
    """
    
    def __init__(self, message_broker: Optional[AgentMessageBroker] = None):
        self.message_broker = message_broker
        self._health_check_interval = 30  # seconds
        self._cleanup_interval = 300  # 5 minutes
        self._running = False
        self._background_tasks: List[asyncio.Task] = []
    
    async def start(self) -> None:
        """Start the agent registry background services."""
        if self._running:
            return
        
        self._running = True
        
        if not self.message_broker:
            self.message_broker = await get_message_broker()
        
        # Start background tasks
        self._background_tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._cleanup_loop())
        ]
        
        logger.info("AgentRegistry started")
    
    async def stop(self) -> None:
        """Stop the agent registry and cleanup resources."""
        self._running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        logger.info("AgentRegistry stopped")
    
    async def register_agent(
        self,
        name: str,
        agent_type: AgentType,
        role: Optional[str] = None,
        capabilities: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tmux_session: Optional[str] = None
    ) -> AgentRegistrationResult:
        """
        Register a new agent with enhanced lifecycle tracking.
        
        Args:
            name: Agent name
            agent_type: Type of agent (Claude, GPT, etc.)
            role: Agent role/specialization
            capabilities: List of agent capabilities
            system_prompt: Agent system prompt
            config: Agent configuration
            tmux_session: Associated tmux session
            
        Returns:
            AgentRegistrationResult with registration details
        """
        registration_start = datetime.utcnow()
        
        try:
            async with get_session() as db:
                # Create agent with enhanced fields
                agent = Agent(
                    name=name,
                    type=agent_type,
                    role=role,
                    capabilities=capabilities or [],
                    system_prompt=system_prompt,
                    config=config or {},
                    tmux_session=tmux_session,
                    status=AgentStatus.INITIALIZING,
                    lifecycle_state=LifecycleState.INITIALIZING.value,
                    spawn_time=registration_start,
                    health_score=1.0,
                    agent_version="3.1.0",
                    orchestrator_metadata={
                        "registration_time": registration_start.isoformat(),
                        "orchestrator_version": "3.1.0",
                        "registration_source": "agent_registry"
                    },
                    resource_usage={
                        "memory_mb": 0.0,
                        "cpu_percent": 0.0,
                        "context_window_usage": 0.0,
                        "active_tasks_count": 0,
                        "last_updated": registration_start.isoformat()
                    }
                )
                
                db.add(agent)
                await db.commit()
                await db.refresh(agent)
                
                # Record registration metrics
                await self._record_metric(
                    db,
                    "agent_registration",
                    "registration_time_ms",
                    (datetime.utcnow() - registration_start).total_seconds() * 1000,
                    "ms",
                    agent_id=agent.id
                )
                
                # Send registration event
                if self.message_broker:
                    await self.message_broker.publish_agent_event(
                        str(agent.id),
                        "agent_registered",
                        {
                            "agent_id": str(agent.id),
                            "name": name,
                            "type": agent_type.value,
                            "role": role,
                            "capabilities_count": len(capabilities or []),
                            "registration_time": registration_start.isoformat()
                        }
                    )
                
                # Update to active state
                await self._update_agent_lifecycle_state(
                    db, agent.id, LifecycleState.ACTIVE, AgentStatus.ACTIVE
                )
                
                logger.info(
                    "Agent registered successfully",
                    agent_id=str(agent.id),
                    name=name,
                    type=agent_type.value,
                    registration_time_ms=(datetime.utcnow() - registration_start).total_seconds() * 1000
                )
                
                return AgentRegistrationResult(
                    success=True,
                    agent_id=agent.id,
                    error_message=None,
                    registration_time=registration_start,
                    capabilities_assigned=[cap.get("name", "") for cap in (capabilities or [])],
                    health_score=1.0
                )
                
        except Exception as e:
            logger.error(
                "Agent registration failed",
                name=name,
                type=agent_type.value,
                error=str(e)
            )
            
            return AgentRegistrationResult(
                success=False,
                agent_id=None,
                error_message=f"Registration failed: {str(e)}",
                registration_time=registration_start,
                capabilities_assigned=[],
                health_score=0.0
            )
    
    async def deregister_agent(self, agent_id: uuid.UUID, graceful: bool = True) -> bool:
        """
        Deregister an agent with proper cleanup.
        
        Args:
            agent_id: Agent ID to deregister
            graceful: Whether to perform graceful shutdown
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with get_session() as db:
                # Get agent
                result = await db.execute(
                    select(Agent).where(Agent.id == agent_id)
                )
                agent = result.scalar_one_or_none()
                
                if not agent:
                    logger.warning("Agent not found for deregistration", agent_id=str(agent_id))
                    return False
                
                # Update lifecycle state to shutting down
                await self._update_agent_lifecycle_state(
                    db, agent_id, LifecycleState.SHUTTING_DOWN, AgentStatus.SHUTTING_DOWN
                )
                
                if graceful:
                    # Cancel any pending tasks assigned to this agent
                    await db.execute(
                        update(Task)
                        .where(
                            and_(
                                Task.assigned_agent_id == agent_id,
                                Task.status.in_([TaskStatus.PENDING, TaskStatus.ASSIGNED])
                            )
                        )
                        .values(
                            status=TaskStatus.CANCELLED,
                            error_message="Agent deregistered",
                            updated_at=datetime.utcnow()
                        )
                    )
                
                # Update final state
                await db.execute(
                    update(Agent)
                    .where(Agent.id == agent_id)
                    .values(
                        lifecycle_state=LifecycleState.TERMINATED.value,
                        status=AgentStatus.INACTIVE,
                        termination_time=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                )
                
                await db.commit()
                
                # Send deregistration event
                if self.message_broker:
                    await self.message_broker.publish_agent_event(
                        str(agent_id),
                        "agent_deregistered",
                        {
                            "agent_id": str(agent_id),
                            "graceful": graceful,
                            "termination_time": datetime.utcnow().isoformat()
                        }
                    )
                
                logger.info(
                    "Agent deregistered successfully",
                    agent_id=str(agent_id),
                    graceful=graceful
                )
                
                return True
                
        except Exception as e:
            logger.error(
                "Agent deregistration failed",
                agent_id=str(agent_id),
                error=str(e)
            )
            return False
    
    async def get_agent(self, agent_id: uuid.UUID) -> Optional[Agent]:
        """Get agent by ID with full details."""
        try:
            async with get_session() as db:
                result = await db.execute(
                    select(Agent).where(Agent.id == agent_id)
                )
                return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Failed to get agent", agent_id=str(agent_id), error=str(e))
            return None
    
    async def list_agents(
        self,
        status: Optional[AgentStatus] = None,
        lifecycle_state: Optional[LifecycleState] = None,
        role: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Agent]:
        """List agents with filtering options."""
        try:
            async with get_session() as db:
                query = select(Agent)
                
                if status:
                    query = query.where(Agent.status == status)
                if lifecycle_state:
                    query = query.where(Agent.lifecycle_state == lifecycle_state.value)
                if role:
                    query = query.where(Agent.role == role)
                
                query = query.offset(offset).limit(limit).order_by(Agent.created_at.desc())
                
                result = await db.execute(query)
                return result.scalars().all()
        except Exception as e:
            logger.error("Failed to list agents", error=str(e))
            return []
    
    async def get_active_agents(self) -> List[Agent]:
        """Get all active agents."""
        return await self.list_agents(
            lifecycle_state=LifecycleState.ACTIVE,
            status=AgentStatus.ACTIVE
        )
    
    async def update_agent_heartbeat(self, agent_id: uuid.UUID) -> bool:
        """Update agent heartbeat and health metrics."""
        try:
            async with get_session() as db:
                # Update heartbeat
                result = await db.execute(
                    update(Agent)
                    .where(Agent.id == agent_id)
                    .values(
                        last_heartbeat=datetime.utcnow(),
                        last_active=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    .returning(Agent.id)
                )
                
                if result.rowcount == 0:
                    return False
                
                await db.commit()
                
                # Record health check
                await self._record_health_check(
                    db, agent_id, "heartbeat", "healthy", {}, None
                )
                
                return True
        except Exception as e:
            logger.error("Failed to update agent heartbeat", agent_id=str(agent_id), error=str(e))
            return False
    
    async def update_agent_resource_usage(
        self,
        agent_id: uuid.UUID,
        resource_usage: AgentResourceUsage
    ) -> bool:
        """Update agent resource usage metrics."""
        try:
            async with get_session() as db:
                # Update resource usage
                await db.execute(
                    update(Agent)
                    .where(Agent.id == agent_id)
                    .values(
                        resource_usage={
                            "memory_mb": resource_usage.memory_mb,
                            "cpu_percent": resource_usage.cpu_percent,
                            "context_window_usage": resource_usage.context_window_usage,
                            "active_tasks_count": resource_usage.active_tasks_count,
                            "last_updated": resource_usage.last_updated.isoformat()
                        },
                        context_window_usage=str(resource_usage.context_window_usage),
                        updated_at=datetime.utcnow()
                    )
                )
                
                await db.commit()
                
                # Record resource metrics
                await self._record_metric(
                    db, "resource_usage", "memory_mb", resource_usage.memory_mb, "mb", agent_id
                )
                await self._record_metric(
                    db, "resource_usage", "cpu_percent", resource_usage.cpu_percent, "percent", agent_id
                )
                await self._record_metric(
                    db, "resource_usage", "context_window_usage", resource_usage.context_window_usage, "percent", agent_id
                )
                
                return True
        except Exception as e:
            logger.error("Failed to update agent resource usage", agent_id=str(agent_id), error=str(e))
            return False
    
    async def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics."""
        try:
            async with get_session() as db:
                # Basic counts
                total_count = await db.execute(select(func.count(Agent.id)))
                active_count = await db.execute(
                    select(func.count(Agent.id)).where(Agent.status == AgentStatus.ACTIVE)
                )
                
                # Health distribution
                healthy_count = await db.execute(
                    select(func.count(Agent.id)).where(Agent.health_score >= 0.8)
                )
                
                # Average response time
                avg_response_time = await db.execute(
                    select(func.avg(func.cast(Agent.average_response_time, func.Float())))
                    .where(Agent.average_response_time.isnot(None))
                )
                
                return {
                    "total_agents": total_count.scalar() or 0,
                    "active_agents": active_count.scalar() or 0,
                    "healthy_agents": healthy_count.scalar() or 0,
                    "average_response_time": avg_response_time.scalar() or 0.0,
                    "registry_status": "running" if self._running else "stopped",
                    "last_updated": datetime.utcnow().isoformat()
                }
        except Exception as e:
            logger.error("Failed to get agent statistics", error=str(e))
            return {
                "total_agents": 0,
                "active_agents": 0,
                "healthy_agents": 0,
                "average_response_time": 0.0,
                "registry_status": "error",
                "last_updated": datetime.utcnow().isoformat()
            }
    
    async def _update_agent_lifecycle_state(
        self,
        db: AsyncSession,
        agent_id: uuid.UUID,
        lifecycle_state: LifecycleState,
        status: AgentStatus
    ) -> None:
        """Update agent lifecycle state and status."""
        await db.execute(
            update(Agent)
            .where(Agent.id == agent_id)
            .values(
                lifecycle_state=lifecycle_state.value,
                status=status,
                updated_at=datetime.utcnow()
            )
        )
        await db.commit()
    
    async def _record_metric(
        self,
        db: AsyncSession,
        metric_type: str,
        metric_name: str,
        metric_value: float,
        metric_unit: str,
        agent_id: Optional[uuid.UUID] = None,
        task_id: Optional[uuid.UUID] = None
    ) -> None:
        """Record a performance metric."""
        try:
            await db.execute(
                """
                INSERT INTO orchestrator_metrics 
                (metric_type, metric_name, metric_value, metric_unit, agent_id, task_id, metadata, measured_at, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $8)
                """,
                (
                    metric_type,
                    metric_name,
                    metric_value,
                    metric_unit,
                    agent_id,
                    task_id,
                    {},
                    datetime.utcnow()
                )
            )
        except Exception as e:
            logger.warning("Failed to record metric", error=str(e))
    
    async def _record_health_check(
        self,
        db: AsyncSession,
        agent_id: uuid.UUID,
        check_type: str,
        check_result: str,
        check_data: Dict[str, Any],
        error_message: Optional[str]
    ) -> None:
        """Record a health check result."""
        try:
            await db.execute(
                """
                INSERT INTO orchestrator_health_checks 
                (agent_id, check_type, check_result, check_data, error_message, checked_at, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $6)
                """,
                (
                    agent_id,
                    check_type,
                    check_result,
                    check_data,
                    error_message,
                    datetime.utcnow()
                )
            )
        except Exception as e:
            logger.warning("Failed to record health check", error=str(e))
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                await self._run_health_checks()
                await asyncio.sleep(self._health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health check loop error", error=str(e))
                await asyncio.sleep(5)
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._running:
            try:
                await self._cleanup_stale_agents()
                await asyncio.sleep(self._cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup loop error", error=str(e))
                await asyncio.sleep(30)
    
    async def _run_health_checks(self) -> None:
        """Run health checks on all active agents."""
        try:
            active_agents = await self.get_active_agents()
            
            for agent in active_agents:
                await self._check_agent_health(agent)
                
        except Exception as e:
            logger.error("Failed to run health checks", error=str(e))
    
    async def _check_agent_health(self, agent: Agent) -> None:
        """Check health of a specific agent."""
        try:
            current_time = datetime.utcnow()
            health_score = 1.0
            
            # Check heartbeat freshness
            if agent.last_heartbeat:
                heartbeat_age = (current_time - agent.last_heartbeat).total_seconds()
                if heartbeat_age > 300:  # 5 minutes
                    health_score -= 0.5
                elif heartbeat_age > 120:  # 2 minutes
                    health_score -= 0.2
            else:
                health_score -= 0.3
            
            # Check context window usage
            context_usage = float(agent.context_window_usage or 0.0)
            if context_usage > 0.9:
                health_score -= 0.3
            elif context_usage > 0.8:
                health_score -= 0.1
            
            # Update health score
            async with get_session() as db:
                await db.execute(
                    update(Agent)
                    .where(Agent.id == agent.id)
                    .values(health_score=max(0.0, health_score))
                )
                await db.commit()
                
                # Record health check
                await self._record_health_check(
                    db,
                    agent.id,
                    "comprehensive",
                    "healthy" if health_score >= 0.7 else "warning" if health_score >= 0.4 else "critical",
                    {
                        "health_score": health_score,
                        "heartbeat_age_seconds": heartbeat_age if agent.last_heartbeat else None,
                        "context_usage": context_usage
                    },
                    None
                )
                
        except Exception as e:
            logger.error("Failed to check agent health", agent_id=str(agent.id), error=str(e))
    
    async def _cleanup_stale_agents(self) -> None:
        """Clean up stale agent records and sessions."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            async with get_session() as db:
                # Mark stale agents as terminated
                await db.execute(
                    update(Agent)
                    .where(
                        and_(
                            Agent.last_heartbeat < cutoff_time,
                            Agent.lifecycle_state.in_([LifecycleState.ACTIVE.value, LifecycleState.BUSY.value])
                        )
                    )
                    .values(
                        lifecycle_state=LifecycleState.TERMINATED.value,
                        status=AgentStatus.ERROR,
                        termination_time=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                )
                
                # Clean up old health checks (keep 7 days)
                health_cutoff = datetime.utcnow() - timedelta(days=7)
                await db.execute(
                    delete(db.exec_statement(
                        "DELETE FROM orchestrator_health_checks WHERE checked_at < $1",
                        (health_cutoff,)
                    ))
                )
                
                # Clean up old metrics (keep 30 days)
                metrics_cutoff = datetime.utcnow() - timedelta(days=30)
                await db.execute(
                    delete(db.exec_statement(
                        "DELETE FROM orchestrator_metrics WHERE measured_at < $1",
                        (metrics_cutoff,)
                    ))
                )
                
                await db.commit()
                
        except Exception as e:
            logger.error("Failed to cleanup stale agents", error=str(e))