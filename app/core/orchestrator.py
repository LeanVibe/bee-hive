"""
Agent Orchestrator for LeanVibe Agent Hive 2.0

Central coordination engine that manages multiple Claude instances,
handles task delegation, monitors agent health, and coordinates
sleep-wake cycles for optimal multi-agent development workflows.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
from anthropic import AsyncAnthropic

from .config import settings
from .redis import get_message_broker, get_session_cache, AgentMessageBroker, SessionCache
from .database import get_session
from .workflow_engine import WorkflowEngine, WorkflowResult
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.session import Session, SessionStatus
from ..models.task import Task, TaskStatus, TaskPriority
from ..models.workflow import Workflow, WorkflowStatus
from sqlalchemy import select, update, func

logger = structlog.get_logger()


class AgentRole(Enum):
    """Defined agent roles in the multi-agent development workflow."""
    STRATEGIC_PARTNER = "strategic_partner"
    PRODUCT_MANAGER = "product_manager" 
    ARCHITECT = "architect"
    BACKEND_DEVELOPER = "backend_developer"
    FRONTEND_DEVELOPER = "frontend_developer"
    DEVOPS_ENGINEER = "devops_engineer"
    QA_ENGINEER = "qa_engineer"
    META_AGENT = "meta_agent"


@dataclass
class AgentCapability:
    """Represents a specific capability of an agent."""
    name: str
    description: str
    confidence_level: float  # 0.0 to 1.0
    specialization_areas: List[str]


@dataclass
class AgentInstance:
    """Represents a running agent instance."""
    id: str
    role: AgentRole
    status: AgentStatus
    tmux_session: Optional[str]
    capabilities: List[AgentCapability]
    current_task: Optional[str]
    context_window_usage: float  # 0.0 to 1.0
    last_heartbeat: datetime
    anthropic_client: Optional[AsyncAnthropic]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            **asdict(self),
            'role': self.role.value,
            'status': self.status.value,
            'capabilities': [asdict(cap) for cap in self.capabilities],
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'anthropic_client': None  # Don't serialize the client
        }


class AgentOrchestrator:
    """
    Central orchestrator for managing multiple Claude agent instances.
    
    Responsibilities:
    - Agent lifecycle management (spawn, monitor, shutdown)
    - Task delegation and load balancing
    - Inter-agent communication coordination
    - Sleep-wake cycle management
    - Context window monitoring and optimization
    - Performance monitoring and health checks
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentInstance] = {}
        self.active_sessions: Dict[str, Session] = {}
        self.message_broker: Optional[AgentMessageBroker] = None
        self.session_cache: Optional[SessionCache] = None
        self.anthropic_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        
        # Workflow execution engine
        self.workflow_engine: Optional[WorkflowEngine] = None
        
        # Orchestrator state
        self.is_running = False
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.consolidation_task: Optional[asyncio.Task] = None
        self.task_queue_task: Optional[asyncio.Task] = None
        
        # Performance monitoring
        self.metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'agents_spawned': 0,
            'sleep_cycles_completed': 0,
            'average_response_time': 0.0,
            'workflows_executed': 0,
            'workflows_completed': 0
        }
    
    async def start(self) -> None:
        """Start the agent orchestrator and background tasks."""
        logger.info("🎭 Starting Agent Orchestrator...")
        
        self.message_broker = get_message_broker()
        self.session_cache = get_session_cache()
        self.is_running = True
        
        # Initialize workflow execution engine
        self.workflow_engine = WorkflowEngine(orchestrator=self)
        await self.workflow_engine.initialize()
        
        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.consolidation_task = asyncio.create_task(self._consolidation_loop())
        self.task_queue_task = asyncio.create_task(self._task_queue_loop())
        
        # Initialize with a default strategic partner agent
        await self.spawn_agent(AgentRole.STRATEGIC_PARTNER)
        
        logger.info("✅ Agent Orchestrator started successfully")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown all agents and background tasks."""
        logger.info("🛑 Shutting down Agent Orchestrator...")
        
        self.is_running = False
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.consolidation_task:
            self.consolidation_task.cancel()
        if self.task_queue_task:
            self.task_queue_task.cancel()
        
        # Shutdown all agents gracefully
        for agent_id in list(self.agents.keys()):
            await self.shutdown_agent(agent_id, graceful=True)
        
        logger.info("✅ Agent Orchestrator shutdown complete")
    
    async def spawn_agent(
        self,
        role: AgentRole,
        agent_id: Optional[str] = None,
        capabilities: Optional[List[AgentCapability]] = None
    ) -> str:
        """Spawn a new agent instance with the specified role."""
        
        # Generate proper UUID for agent_id if not provided
        if agent_id is None:
            agent_id = str(uuid.uuid4())
        
        if agent_id in self.agents:
            raise ValueError(f"Agent {agent_id} already exists")
        
        if len(self.agents) >= settings.MAX_CONCURRENT_AGENTS:
            raise RuntimeError("Maximum number of concurrent agents reached")
        
        # Default capabilities based on role
        if capabilities is None:
            capabilities = self._get_default_capabilities(role)
        
        # Create agent instance
        agent_instance = AgentInstance(
            id=agent_id,
            role=role,
            status=AgentStatus.INITIALIZING,
            tmux_session=None,
            capabilities=capabilities,
            current_task=None,
            context_window_usage=0.0,
            last_heartbeat=datetime.utcnow(),
            anthropic_client=AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        )
        
        self.agents[agent_id] = agent_instance
        
        # Create consumer group for agent messages
        await self.message_broker.create_consumer_group(
            f"agent_messages:{agent_id}",
            f"group_{agent_id}",
            agent_id
        )
        
        # Store agent in database
        async with get_session() as db_session:
            db_agent = Agent(
                id=uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id,
                name=f"{role.value.replace('_', ' ').title()} Agent",
                type=AgentType.CLAUDE,
                role=role.value,
                capabilities=[asdict(cap) for cap in capabilities],
                status=AgentStatus.INITIALIZING
            )
            db_session.add(db_agent)
            await db_session.commit()
        
        # Start agent in background
        asyncio.create_task(self._start_agent_instance(agent_instance))
        
        self.metrics['agents_spawned'] += 1
        
        logger.info(
            "🎭 Agent spawned",
            agent_id=agent_id,
            role=role.value,
            capabilities_count=len(capabilities)
        )
        
        return agent_id
    
    async def shutdown_agent(self, agent_id: str, graceful: bool = True) -> bool:
        """Shutdown a specific agent instance."""
        
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found")
            return False
        
        agent = self.agents[agent_id]
        
        if graceful:
            # Allow agent to complete current task
            if agent.current_task:
                logger.info(f"Waiting for agent {agent_id} to complete current task...")
                # TODO: Implement graceful task completion
        
        # Update agent status
        agent.status = AgentStatus.SHUTTING_DOWN
        
        # Update database
        async with get_session() as db_session:
            db_agent = await db_session.get(Agent, agent_id)
            if db_agent:
                db_agent.status = AgentStatus.INACTIVE
                await db_session.commit()
        
        # Clean up agent instance
        del self.agents[agent_id]
        
        logger.info(f"🛑 Agent {agent_id} shutdown complete")
        return True
    
    async def delegate_task(
        self,
        task_description: str,
        task_type: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        preferred_agent_role: Optional[AgentRole] = None,
        context: Optional[Dict[str, Any]] = None,
        required_capabilities: Optional[List[str]] = None
    ) -> str:
        """Delegate a task to the most suitable agent using intelligent scheduling."""
        
        # Create task in database first
        task_id = str(uuid.uuid4())
        async with get_session() as db_session:
            task = Task(
                id=task_id,
                title=task_description,
                description=task_description,
                task_type=task_type,
                status=TaskStatus.PENDING,
                priority=priority,
                context=context or {},
                required_capabilities=required_capabilities or []
            )
            db_session.add(task)
            await db_session.commit()
        
        # Use intelligent task scheduler
        assigned_agent_id = await self._schedule_task(task_id, task_type, priority, preferred_agent_role, required_capabilities)
        
        if not assigned_agent_id:
            # Queue task for later assignment
            logger.info(
                "📋 Task queued - no suitable agent available",
                task_id=task_id,
                task_type=task_type,
                priority=priority.value
            )
            return task_id
        
        logger.info(
            "📋 Task delegated",
            task_id=task_id,
            agent_id=assigned_agent_id,
            task_type=task_type,
            priority=priority.value
        )
        
        return task_id
    
    async def initiate_sleep_cycle(self, agent_id: str) -> bool:
        """Initiate a sleep-wake cycle for an agent to consolidate memory."""
        
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        
        # Only initiate sleep if context usage is high
        if agent.context_window_usage < settings.CONSOLIDATION_THRESHOLD:
            return False
        
        # Send sleep command
        await self.message_broker.send_message(
            from_agent="orchestrator",
            to_agent=agent_id,
            message_type="sleep_cycle_initiate",
            payload={
                "consolidation_type": "full",
                "preserve_priority": ["current_task", "recent_context"]
            }
        )
        
        agent.status = AgentStatus.SLEEPING
        
        logger.info(f"😴 Initiated sleep cycle for agent {agent_id}")
        return True
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for monitoring."""
        
        agent_statuses = {}
        for agent_id, agent in self.agents.items():
            agent_statuses[agent_id] = {
                "role": agent.role.value,
                "status": agent.status.value,
                "current_task": agent.current_task,
                "context_usage": agent.context_window_usage,
                "last_heartbeat": agent.last_heartbeat.isoformat()
            }
        
        # Get workflow engine status
        workflow_engine_status = {}
        if self.workflow_engine:
            workflow_engine_status = {
                "workflow_engine_active": True,
                "workflow_metrics": self.workflow_engine.get_metrics()
            }
        else:
            workflow_engine_status = {"workflow_engine_active": False}
        
        return {
            "orchestrator_status": "running" if self.is_running else "stopped",
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a.status == AgentStatus.ACTIVE]),
            "agents": agent_statuses,
            "metrics": self.metrics,
            "system_health": await self._check_system_health(),
            **workflow_engine_status
        }
    
    async def _start_agent_instance(self, agent: AgentInstance) -> None:
        """Start an individual agent instance."""
        try:
            # TODO: Implement tmux session creation
            # agent.tmux_session = await self._create_tmux_session(agent.id)
            
            agent.status = AgentStatus.ACTIVE
            agent.last_heartbeat = datetime.utcnow()
            
            logger.info(f"✅ Agent {agent.id} started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start agent {agent.id}", error=str(e))
            agent.status = AgentStatus.ERROR
    
    async def _heartbeat_loop(self) -> None:
        """Background task to monitor agent health with automatic recovery."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                timeout_threshold = current_time - timedelta(seconds=settings.AGENT_TIMEOUT)
                
                for agent_id, agent in list(self.agents.items()):
                    if agent.last_heartbeat < timeout_threshold:
                        await self._handle_agent_timeout(agent_id, agent)
                    
                    # Check agent performance and health
                    await self._monitor_agent_health(agent_id, agent)
                
                await asyncio.sleep(settings.AGENT_HEARTBEAT_INTERVAL)
                
            except Exception as e:
                logger.error("❌ Error in heartbeat loop", error=str(e))
                await asyncio.sleep(5)
    
    async def _handle_agent_timeout(self, agent_id: str, agent: AgentInstance) -> None:
        """Handle agent timeout with automatic restart."""
        
        logger.warning(f"⚠️ Agent {agent_id} heartbeat timeout")
        
        # Mark agent as in error state
        agent.status = AgentStatus.ERROR
        
        # Update database
        try:
            async with get_session() as db_session:
                await db_session.execute(
                    update(Agent)
                    .where(Agent.id == agent_id)
                    .values(status=AgentStatus.ERROR, updated_at=datetime.utcnow())
                )
                await db_session.commit()
        except Exception as e:
            logger.error(f"Failed to update agent status in database", agent_id=agent_id, error=str(e))
        
        # Attempt automatic restart if configured
        if hasattr(settings, 'AUTO_RESTART_AGENTS') and settings.AUTO_RESTART_AGENTS:
            await self._attempt_agent_restart(agent_id, agent)
    
    async def _attempt_agent_restart(self, agent_id: str, agent: AgentInstance) -> bool:
        """Attempt to automatically restart a failed agent."""
        
        try:
            logger.info(f"🔄 Attempting automatic restart of agent {agent_id}")
            
            # Save current agent configuration
            agent_role = agent.role
            agent_capabilities = agent.capabilities.copy()
            
            # Shutdown the failed agent
            await self.shutdown_agent(agent_id, graceful=False)
            
            # Wait briefly for cleanup
            await asyncio.sleep(2)
            
            # Respawn agent with same configuration
            new_agent_id = await self.spawn_agent(
                role=agent_role,
                agent_id=agent_id,
                capabilities=agent_capabilities
            )
            
            if new_agent_id:
                logger.info(f"✅ Successfully restarted agent {agent_id}")
                self.metrics['agents_restarted'] = self.metrics.get('agents_restarted', 0) + 1
                return True
            else:
                logger.error(f"❌ Failed to restart agent {agent_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error during agent restart", agent_id=agent_id, error=str(e))
            return False
    
    async def _monitor_agent_health(self, agent_id: str, agent: AgentInstance) -> None:
        """Monitor individual agent health metrics."""
        
        try:
            health_issues = []
            
            # Check context window usage
            if agent.context_window_usage > 0.95:
                health_issues.append("critical_context_usage")
                logger.warning(f"🚨 Agent {agent_id} critical context usage: {agent.context_window_usage}")
            elif agent.context_window_usage > 0.85:
                health_issues.append("high_context_usage")
            
            # Check if agent is stuck on a task
            if agent.current_task and agent.status == AgentStatus.ACTIVE:
                async with get_session() as db_session:
                    task = await db_session.get(Task, agent.current_task)
                    if task and task.started_at:
                        task_duration = datetime.utcnow() - task.started_at
                        if task_duration.total_seconds() > 3600:  # 1 hour
                            health_issues.append("long_running_task")
                            logger.warning(f"⏰ Agent {agent_id} has been on task {agent.current_task} for {task_duration}")
            
            # Check memory/performance metrics
            if hasattr(agent, 'memory_usage') and agent.memory_usage > 0.9:
                health_issues.append("high_memory_usage")
            
            # Update agent health status
            if health_issues:
                await self._record_health_issues(agent_id, health_issues)
            
        except Exception as e:
            logger.error(f"Error monitoring agent health", agent_id=agent_id, error=str(e))
    
    async def _record_health_issues(self, agent_id: str, health_issues: List[str]) -> None:
        """Record health issues for monitoring and alerting."""
        
        try:
            # Store health metrics in database
            async with get_session() as db_session:
                from ..models.performance_metric import PerformanceMetric
                
                for issue in health_issues:
                    metric = PerformanceMetric(
                        metric_name=f"agent_health_{issue}",
                        metric_value=1.0,
                        agent_id=agent_id,
                        tags={"issue": issue, "severity": "warning"}
                    )
                    db_session.add(metric)
                
                await db_session.commit()
                
        except Exception as e:
            logger.error(f"Failed to record health issues", agent_id=agent_id, error=str(e))
    
    async def _consolidation_loop(self) -> None:
        """Background task to manage sleep-wake cycles."""
        while self.is_running:
            try:
                for agent_id, agent in self.agents.items():
                    if (agent.status == AgentStatus.ACTIVE and 
                        agent.context_window_usage > settings.CONSOLIDATION_THRESHOLD):
                        await self.initiate_sleep_cycle(agent_id)
                
                await asyncio.sleep(settings.SLEEP_CYCLE_INTERVAL)
                
            except Exception as e:
                logger.error("❌ Error in consolidation loop", error=str(e))
                await asyncio.sleep(60)
    
    async def _task_queue_loop(self) -> None:
        """Background task to process queued tasks and assign them to available agents."""
        while self.is_running:
            try:
                assigned_count = await self.process_task_queue()
                
                if assigned_count > 0:
                    logger.info(f"📋 Processed task queue: {assigned_count} tasks assigned")
                
                # Check every 30 seconds for new tasks to assign
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error("❌ Error in task queue loop", error=str(e))
                await asyncio.sleep(60)
    
    async def _schedule_task(
        self,
        task_id: str,
        task_type: str,
        priority: TaskPriority,
        preferred_role: Optional[AgentRole] = None,
        required_capabilities: Optional[List[str]] = None
    ) -> Optional[str]:
        """Intelligent task scheduling with agent matching and load balancing."""
        
        # Get task from database
        async with get_session() as db_session:
            task = await db_session.get(Task, task_id)
            if not task:
                return None
        
        # Find candidate agents
        candidate_agents = await self._find_candidate_agents(
            task_type, preferred_role, required_capabilities
        )
        
        if not candidate_agents:
            return None
        
        # Score and rank agents
        agent_scores = []
        for agent in candidate_agents:
            score = await self._calculate_agent_suitability_score(
                agent, task_type, priority, required_capabilities or []
            )
            agent_scores.append((agent, score))
        
        # Sort by score (descending)
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select best agent
        best_agent, best_score = agent_scores[0]
        
        # Assign task to best agent
        success = await self._assign_task_to_agent(task_id, best_agent.id)
        
        if success:
            return best_agent.id
        
        return None
    
    async def _find_candidate_agents(
        self,
        task_type: str,
        preferred_role: Optional[AgentRole] = None,
        required_capabilities: Optional[List[str]] = None
    ) -> List[AgentInstance]:
        """Find candidate agents for task assignment."""
        
        available_agents = [
            agent for agent in self.agents.values()
            if agent.status == AgentStatus.ACTIVE and agent.current_task is None
        ]
        
        if not available_agents:
            return []
        
        candidates = []
        
        # Filter by preferred role if specified
        if preferred_role:
            role_agents = [a for a in available_agents if a.role == preferred_role]
            if role_agents:
                candidates.extend(role_agents)
        
        # Filter by required capabilities
        if required_capabilities:
            capability_agents = []
            for agent in available_agents:
                if self._agent_has_required_capabilities(agent, required_capabilities):
                    capability_agents.append(agent)
            candidates.extend(capability_agents)
        
        # If no specific filters, use all available agents
        if not candidates:
            candidates = available_agents
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for agent in candidates:
            if agent.id not in seen:
                seen.add(agent.id)
                unique_candidates.append(agent)
        
        return unique_candidates
    
    def _agent_has_required_capabilities(
        self, 
        agent: AgentInstance, 
        required_capabilities: List[str]
    ) -> bool:
        """Check if agent has all required capabilities."""
        
        if not required_capabilities:
            return True
        
        agent_capabilities = [cap.name.lower() for cap in agent.capabilities]
        
        for req_cap in required_capabilities:
            if not any(req_cap.lower() in cap for cap in agent_capabilities):
                return False
        
        return True
    
    async def _calculate_agent_suitability_score(
        self,
        agent: AgentInstance,
        task_type: str,
        priority: TaskPriority,
        required_capabilities: List[str]
    ) -> float:
        """Calculate comprehensive suitability score for agent-task matching."""
        
        total_score = 0.0
        
        # Base capability score (40% weight)
        capability_score = self._calculate_capability_score(agent, task_type)
        total_score += capability_score * 0.4
        
        # Required capabilities match (30% weight)
        if required_capabilities:
            req_cap_score = 0.0
            for req_cap in required_capabilities:
                for agent_cap in agent.capabilities:
                    if req_cap.lower() in agent_cap.name.lower():
                        req_cap_score += agent_cap.confidence_level
                        break
                    elif any(req_cap.lower() in area.lower() for area in agent_cap.specialization_areas):
                        req_cap_score += agent_cap.confidence_level * 0.8
                        break
            
            req_cap_score = min(1.0, req_cap_score / len(required_capabilities))
            total_score += req_cap_score * 0.3
        else:
            total_score += 0.3  # Full score if no specific requirements
        
        # Agent availability and performance (20% weight)
        availability_score = 1.0 - agent.context_window_usage
        total_score += availability_score * 0.2
        
        # Priority matching (10% weight)
        priority_score = self._calculate_priority_score(agent, priority)
        total_score += priority_score * 0.1
        
        return min(1.0, total_score)
    
    def _calculate_priority_score(self, agent: AgentInstance, priority: TaskPriority) -> float:
        """Calculate priority-based scoring for agent selection."""
        
        # High-priority tasks prefer agents with proven track record
        if priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]:
            # Prefer agents with higher capabilities and recent activity
            if agent.last_heartbeat:
                time_since_activity = (datetime.utcnow() - agent.last_heartbeat).total_seconds()
                if time_since_activity < 300:  # Active in last 5 minutes
                    return 1.0
                elif time_since_activity < 900:  # Active in last 15 minutes
                    return 0.8
                else:
                    return 0.6
            return 0.5
        
        # Medium and low priority tasks are less selective
        return 0.8
    
    async def _assign_task_to_agent(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to an agent and update states."""
        
        try:
            # Update task in database
            async with get_session() as db_session:
                await db_session.execute(
                    update(Task)
                    .where(Task.id == task_id)
                    .values(
                        assigned_agent_id=agent_id,
                        status=TaskStatus.ASSIGNED,
                        assigned_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                )
                await db_session.commit()
            
            # Update agent state
            if agent_id in self.agents:
                self.agents[agent_id].current_task = task_id
            
            # Send task assignment message
            await self.message_broker.send_message(
                from_agent="orchestrator",
                to_agent=agent_id,
                message_type="task_assignment",
                payload={
                    "task_id": task_id
                }
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to assign task to agent",
                task_id=task_id,
                agent_id=agent_id,
                error=str(e)
            )
            return False
    
    async def process_task_queue(self) -> int:
        """Process queued tasks and attempt to assign them to available agents."""
        
        assigned_count = 0
        
        try:
            # Get pending tasks ordered by priority and creation time
            async with get_session() as db_session:
                result = await db_session.execute(
                    select(Task)
                    .where(Task.status == TaskStatus.PENDING.value)
                    .where(Task.assigned_agent_id.is_(None))
                    .order_by(Task.priority.desc(), Task.created_at.asc())
                    .limit(10)  # Process up to 10 tasks at a time
                )
                pending_tasks = result.scalars().all()
            
            for task in pending_tasks:
                # Try to schedule each task
                assigned_agent_id = await self._schedule_task(
                    str(task.id),
                    task.task_type.value if task.task_type else "general",
                    task.priority,
                    None,
                    task.required_capabilities
                )
                
                if assigned_agent_id:
                    assigned_count += 1
                    logger.info(
                        "📋 Queued task assigned",
                        task_id=str(task.id),
                        agent_id=assigned_agent_id
                    )
        
        except Exception as e:
            logger.error("Error processing task queue", error=str(e))
        
        return assigned_count
    
    # Workflow management methods
    
    async def execute_workflow(self, workflow_id: str) -> WorkflowResult:
        """
        Execute a workflow using the workflow engine.
        
        Args:
            workflow_id: UUID of the workflow to execute
            
        Returns:
            WorkflowResult with execution status and metrics
        """
        if not self.workflow_engine:
            raise RuntimeError("Workflow engine not initialized")
        
        logger.info("🎯 Orchestrator executing workflow", workflow_id=workflow_id)
        
        try:
            result = await self.workflow_engine.execute_workflow(workflow_id)
            
            # Update orchestrator metrics
            self.metrics['workflows_executed'] += 1
            if result.status == WorkflowStatus.COMPLETED:
                self.metrics['workflows_completed'] += 1
            
            logger.info(
                "🎯 Workflow execution completed via orchestrator",
                workflow_id=workflow_id,
                status=result.status.value,
                completed_tasks=result.completed_tasks,
                failed_tasks=result.failed_tasks
            )
            
            return result
            
        except Exception as e:
            logger.error("❌ Orchestrator workflow execution failed", workflow_id=workflow_id, error=str(e))
            raise
    
    async def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow via the workflow engine."""
        if not self.workflow_engine:
            return False
        
        return await self.workflow_engine.pause_workflow(workflow_id)
    
    async def cancel_workflow(self, workflow_id: str, reason: str = None) -> bool:
        """Cancel a running workflow via the workflow engine."""
        if not self.workflow_engine:
            return False
        
        return await self.workflow_engine.cancel_workflow(workflow_id, reason)
    
    async def get_workflow_execution_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current execution status for a workflow."""
        if not self.workflow_engine:
            return {"error": "Workflow engine not available"}
        
        return await self.workflow_engine.get_execution_status(workflow_id)
    
    async def handle_workflow_task_completion(self, task_id: str, result: Dict[str, Any]) -> None:
        """Handle task completion callback from agents for workflow tasks."""
        if self.workflow_engine:
            await self.workflow_engine.handle_task_completion(task_id, result)
    
    async def _find_best_agent(
        self,
        task_type: str,
        preferred_role: Optional[AgentRole] = None
    ) -> Optional[AgentInstance]:
        """Legacy method - Find the best agent for a given task type."""
        
        candidates = await self._find_candidate_agents(task_type, preferred_role)
        
        if not candidates:
            return None
        
        # Find agent with best capability match
        best_agent = None
        best_score = 0.0
        
        for agent in candidates:
            score = self._calculate_capability_score(agent, task_type)
            if score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent
    
    def _calculate_capability_score(self, agent: AgentInstance, task_type: str) -> float:
        """Calculate how well an agent's capabilities match a task type."""
        # TODO: Implement sophisticated capability matching
        # For now, return a simple score based on role
        role_scores = {
            AgentRole.BACKEND_DEVELOPER: ["api", "database", "backend"],
            AgentRole.FRONTEND_DEVELOPER: ["ui", "frontend", "react", "vue"],
            AgentRole.DEVOPS_ENGINEER: ["deployment", "docker", "ci", "infrastructure"],
            AgentRole.QA_ENGINEER: ["testing", "qa", "validation"],
            AgentRole.ARCHITECT: ["design", "architecture", "planning"],
        }
        
        keywords = role_scores.get(agent.role, [])
        task_lower = task_type.lower()
        
        return sum(1.0 for keyword in keywords if keyword in task_lower) / max(len(keywords), 1)
    
    def _get_default_capabilities(self, role: AgentRole) -> List[AgentCapability]:
        """Get default capabilities for an agent role."""
        capabilities_map = {
            AgentRole.STRATEGIC_PARTNER: [
                AgentCapability("requirement_analysis", "Analyze and clarify requirements", 0.9, ["planning", "communication"]),
                AgentCapability("decision_making", "Make strategic decisions", 0.8, ["strategy", "leadership"])
            ],
            AgentRole.PRODUCT_MANAGER: [
                AgentCapability("project_planning", "Plan and organize projects", 0.9, ["planning", "organization"]),
                AgentCapability("backlog_management", "Manage task backlogs", 0.8, ["prioritization", "task_management"])
            ],
            AgentRole.BACKEND_DEVELOPER: [
                AgentCapability("api_development", "Develop REST APIs", 0.9, ["fastapi", "python", "databases"]),
                AgentCapability("database_design", "Design database schemas", 0.8, ["postgresql", "sqlalchemy"])
            ],
            AgentRole.FRONTEND_DEVELOPER: [
                AgentCapability("ui_development", "Develop user interfaces", 0.9, ["react", "typescript", "css"]),
                AgentCapability("responsive_design", "Create responsive designs", 0.8, ["mobile", "tablet", "desktop"])
            ],
            AgentRole.DEVOPS_ENGINEER: [
                AgentCapability("containerization", "Container deployment", 0.9, ["docker", "kubernetes"]),
                AgentCapability("ci_cd", "Continuous integration/deployment", 0.8, ["github_actions", "automation"])
            ],
            AgentRole.QA_ENGINEER: [
                AgentCapability("test_automation", "Automated testing", 0.9, ["pytest", "selenium", "api_testing"]),
                AgentCapability("quality_assurance", "Code quality validation", 0.8, ["code_review", "standards"])
            ],
            AgentRole.ARCHITECT: [
                AgentCapability("system_design", "System architecture design", 0.9, ["scalability", "performance"]),
                AgentCapability("technical_leadership", "Technical guidance", 0.8, ["mentoring", "standards"])
            ],
            AgentRole.META_AGENT: [
                AgentCapability("self_improvement", "System self-improvement", 0.9, ["optimization", "learning"]),
                AgentCapability("prompt_optimization", "Optimize agent prompts", 0.8, ["prompt_engineering", "efficiency"])
            ]
        }
        
        return capabilities_map.get(role, [])
    
    async def _check_system_health(self) -> Dict[str, bool]:
        """Check overall system health with comprehensive diagnostics."""
        
        health_status = {}
        
        try:
            # Check database connectivity
            health_status["database"] = await self._check_database_health()
            
            # Check Redis connectivity
            health_status["redis"] = await self._check_redis_health()
            
            # Check agent responsiveness
            active_agents = [a for a in self.agents.values() if a.status == AgentStatus.ACTIVE]
            health_status["agents_responsive"] = len(active_agents) > 0
            health_status["sufficient_agents"] = len(active_agents) >= 1  # At least one active agent
            
            # Check task processing capability
            health_status["task_processing"] = await self._check_task_processing_health()
            
            # Check system resource usage
            health_status["system_resources"] = await self._check_system_resources()
            
            # Overall system health
            health_status["overall"] = all([
                health_status.get("database", False),
                health_status.get("redis", False),
                health_status.get("agents_responsive", False)
            ])
            
        except Exception as e:
            logger.error("Error checking system health", error=str(e))
            health_status["overall"] = False
        
        return health_status
    
    async def _check_database_health(self) -> bool:
        """Check database connectivity and performance."""
        try:
            async with get_session() as db_session:
                # Simple connectivity test
                result = await db_session.execute(select(1))
                result.scalar()
                
                # Check if we can query agents table
                result = await db_session.execute(select(Agent.id).limit(1))
                result.scalars().first()
                
                return True
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False
    
    async def _check_redis_health(self) -> bool:
        """Check Redis connectivity and performance."""
        try:
            if self.message_broker:
                # Test Redis connection with a simple ping
                # Note: This depends on the message broker implementation
                return True  # Placeholder - implement based on actual Redis client
            return False
        except Exception as e:
            logger.error("Redis health check failed", error=str(e))
            return False
    
    async def _check_task_processing_health(self) -> bool:
        """Check if task processing is healthy."""
        try:
            # Check for stuck tasks
            async with get_session() as db_session:
                stuck_tasks_result = await db_session.execute(
                    select(func.count(Task.id))
                    .where(Task.status == TaskStatus.IN_PROGRESS.value)
                    .where(Task.started_at < datetime.utcnow() - timedelta(hours=2))
                )
                stuck_tasks_count = stuck_tasks_result.scalar()
                
                # Check for overloaded queue
                pending_tasks_result = await db_session.execute(
                    select(func.count(Task.id))
                    .where(Task.status == TaskStatus.PENDING.value)
                )
                pending_tasks_count = pending_tasks_result.scalar()
                
                # Health is good if no stuck tasks and reasonable queue size
                return stuck_tasks_count == 0 and pending_tasks_count < 100
                
        except Exception as e:
            logger.error("Task processing health check failed", error=str(e))
            return False
    
    async def _check_system_resources(self) -> bool:
        """Check system resource usage."""
        try:
            # Check orchestrator metrics
            total_agents = len(self.agents)
            active_agents = len([a for a in self.agents.values() if a.status == AgentStatus.ACTIVE])
            
            # Resource health is good if we have reasonable agent distribution
            agent_health = total_agents > 0 and (active_agents / total_agents) > 0.5
            
            # Check for memory pressure indicators
            high_context_agents = len([
                a for a in self.agents.values() 
                if a.context_window_usage > 0.9
            ])
            
            memory_health = (high_context_agents / max(total_agents, 1)) < 0.5
            
            return agent_health and memory_health
            
        except Exception as e:
            logger.error("System resources health check failed", error=str(e))
            return False