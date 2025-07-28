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
from collections import defaultdict, deque
import time
import random
import heapq
import threading

import structlog
from anthropic import AsyncAnthropic

from .config import settings
from .redis import get_message_broker, get_session_cache, AgentMessageBroker, SessionCache
from .database import get_session
from .workflow_engine import WorkflowEngine, WorkflowResult, TaskExecutionState
from .intelligent_task_router import IntelligentTaskRouter, TaskRoutingContext, RoutingStrategy
from .capability_matcher import CapabilityMatcher
from .agent_persona_system import AgentPersonaSystem, PersonaAssignment, get_agent_persona_system
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.session import Session, SessionStatus
from ..models.task import Task, TaskStatus, TaskPriority
from ..models.workflow import Workflow, WorkflowStatus
from ..models.agent_performance import AgentPerformanceHistory, TaskRoutingDecision, WorkloadSnapshot
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
        
        # Intelligent task routing system
        self.intelligent_router: Optional[IntelligentTaskRouter] = None
        self.capability_matcher: Optional[CapabilityMatcher] = None
        
        # Agent persona system for role-based assignment
        self.persona_system: Optional[AgentPersonaSystem] = None
        
        # Enhanced task queuing system
        self.task_queues = {
            'high_priority': [],  # Priority queue using heapq
            'medium_priority': deque(),  # FIFO queue
            'low_priority': deque(),  # FIFO queue
            'workflow_tasks': {},  # Workflow-specific queues
            'retry_queue': []  # Priority queue for retry tasks
        }
        self.queue_lock = threading.RLock()
        self.queue_processing_active = False
        
        # Orchestrator state
        self.is_running = False
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.consolidation_task: Optional[asyncio.Task] = None
        self.task_queue_task: Optional[asyncio.Task] = None
        self.workload_monitoring_task: Optional[asyncio.Task] = None
        
        # Performance monitoring
        self.metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'agents_spawned': 0,
            'sleep_cycles_completed': 0,
            'average_response_time': 0.0,
            'workflows_executed': 0,
            'workflows_completed': 0,
            'routing_decisions': 0,
            'routing_accuracy': 0.0,
            'load_balancing_actions': 0,
            'circuit_breaker_trips': 0,
            'retry_attempts': 0,
            'automatic_recovery_actions': 0
        }
        
        # Circuit breaker and error handling
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.error_thresholds = {
            'agent_failure_rate': 0.3,      # 30% failure rate trips circuit breaker
            'consecutive_failures': 5,       # 5 consecutive failures trips breaker
            'recovery_time_seconds': 60,     # 1 minute before attempting recovery
            'max_retry_attempts': 3,         # Maximum retry attempts per task
            'exponential_backoff_base': 2    # Base for exponential backoff
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
        
        # Initialize intelligent routing system
        self.intelligent_router = IntelligentTaskRouter()
        self.capability_matcher = CapabilityMatcher()
        
        # Initialize persona system
        self.persona_system = get_agent_persona_system()
        await self.persona_system.initialize_default_personas()
        
        # Start background tasks
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self.consolidation_task = asyncio.create_task(self._consolidation_loop())
        self.task_queue_task = asyncio.create_task(self._enhanced_task_queue_loop())
        self.workload_monitoring_task = asyncio.create_task(self._workload_monitoring_loop())
        
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
        if self.workload_monitoring_task:
            self.workload_monitoring_task.cancel()
        
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
        required_capabilities: Optional[List[str]] = None,
        estimated_effort: Optional[int] = None,
        due_date: Optional[datetime] = None,
        dependencies: Optional[List[str]] = None,
        routing_strategy: RoutingStrategy = RoutingStrategy.ADAPTIVE,
        preferred_persona_id: Optional[str] = None,
        workflow_id: Optional[str] = None
    ) -> str:
        """Delegate a task to the most suitable agent using intelligent routing."""
        
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
                required_capabilities=required_capabilities or [],
                estimated_effort=estimated_effort,
                due_date=due_date,
                dependencies=dependencies or [],
                workflow_id=workflow_id
            )
            db_session.add(task)
            await db_session.commit()
        
        # Use intelligent task router
        available_agents = await self._get_available_agent_ids()
        
        # Filter by preferred role if specified
        if preferred_agent_role:
            role_filtered_agents = []
            for agent_id in available_agents:
                if agent_id in self.agents and self.agents[agent_id].role == preferred_agent_role:
                    role_filtered_agents.append(agent_id)
            available_agents = role_filtered_agents if role_filtered_agents else available_agents
        
        # Create routing context
        routing_context = TaskRoutingContext(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            required_capabilities=required_capabilities or [],
            estimated_effort=estimated_effort,
            due_date=due_date,
            dependencies=dependencies or [],
            workflow_id=None,
            context=context or {}
        )
        
        # Route task using intelligent algorithm
        assigned_agent_id = await self.intelligent_router.route_task(
            task=routing_context,
            available_agents=available_agents,
            strategy=routing_strategy
        )
        
        if not assigned_agent_id:
            # Use enhanced queuing system for later assignment
            queued = await self.enqueue_task(
                task_id=task_id,
                priority=priority,
                workflow_id=workflow_id,
                estimated_effort=estimated_effort
            )
            
            if queued:
                logger.info(
                    "📋 Task queued using enhanced system",
                    task_id=task_id,
                    task_type=task_type,
                    priority=priority.value,
                    available_agents=len(available_agents),
                    workflow_id=workflow_id
                )
            else:
                logger.error(
                    "❌ Failed to queue task",
                    task_id=task_id,
                    task_type=task_type
                )
            
            return task_id
        
        # Assign task to selected agent with persona integration
        success = await self._assign_task_to_agent_with_persona(
            task_id, assigned_agent_id, task, preferred_persona_id, context
        )
        
        if success:
            # Record routing decision for analytics
            await self._record_routing_analytics(
                task_id, assigned_agent_id, routing_context, routing_strategy
            )
            
            self.metrics['routing_decisions'] += 1
            
            logger.info(
                "📋 Task delegated intelligently",
                task_id=task_id,
                agent_id=assigned_agent_id,
                task_type=task_type,
                priority=priority.value,
                strategy=routing_strategy.value
            )
        else:
            logger.error(
                "❌ Failed to assign task after routing",
                task_id=task_id,
                agent_id=assigned_agent_id
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
        """Handle agent timeout with circuit breaker and enhanced recovery."""
        
        logger.warning(f"⚠️ Agent {agent_id} heartbeat timeout")
        
        # Update circuit breaker state
        await self._update_circuit_breaker(agent_id, success=False, error_type='timeout')
        
        # Check if circuit breaker should trip
        if await self._should_trip_circuit_breaker(agent_id):
            await self._trip_circuit_breaker(agent_id, 'agent_timeout')
            return
        
        # Mark agent as in error state
        agent.status = AgentStatus.ERROR
        
        # Update database with error details
        try:
            async with get_session() as db_session:
                await db_session.execute(
                    update(Agent)
                    .where(Agent.id == agent_id)
                    .values(
                        status=AgentStatus.ERROR, 
                        updated_at=datetime.utcnow(),
                        error_count=Agent.error_count + 1,
                        last_error='heartbeat_timeout'
                    )
                )
                await db_session.commit()
        except Exception as e:
            logger.error(f"Failed to update agent status in database", agent_id=agent_id, error=str(e))
        
        # Attempt automatic restart with circuit breaker protection
        if hasattr(settings, 'AUTO_RESTART_AGENTS') and settings.AUTO_RESTART_AGENTS:
            await self._attempt_agent_restart_with_protection(agent_id, agent)
    
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
        """Legacy task queue loop - use _enhanced_task_queue_loop instead."""
        return await self._enhanced_task_queue_loop()
    
    async def _enhanced_task_queue_loop(self) -> None:
        """Enhanced background task processing with priority queuing and workflow coordination."""
        self.queue_processing_active = True
        
        while self.is_running:
            try:
                # Process tasks from priority queues
                assigned_count = await self._process_priority_queues()
                
                # Process workflow-specific tasks
                workflow_assigned = await self._process_workflow_queues()
                
                # Process retry queue
                retry_assigned = await self._process_retry_queue()
                
                total_assigned = assigned_count + workflow_assigned + retry_assigned
                
                if total_assigned > 0:
                    logger.info(
                        f"📋 Enhanced queue processing completed",
                        regular_tasks=assigned_count,
                        workflow_tasks=workflow_assigned,
                        retry_tasks=retry_assigned,
                        total=total_assigned
                    )
                
                # Adaptive sleep based on queue activity
                if total_assigned > 0:
                    await asyncio.sleep(10)  # Short sleep if tasks were processed
                else:
                    await asyncio.sleep(30)  # Longer sleep if no activity
                
            except Exception as e:
                logger.error("❌ Error in enhanced task queue loop", error=str(e))
                await asyncio.sleep(60)
        
        self.queue_processing_active = False
    
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
        """Legacy task assignment method - use _assign_task_to_agent_with_persona instead."""
        return await self._assign_task_to_agent_with_persona(task_id, agent_id, None, None, None)
    
    async def _assign_task_to_agent_with_persona(
        self, 
        task_id: str, 
        agent_id: str, 
        task: Optional[Task] = None,
        preferred_persona_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Assign a task to an agent with persona integration and enhanced coordination."""
        
        try:
            # Get task from database if not provided
            if not task:
                async with get_session() as db_session:
                    task = await db_session.get(Task, task_id)
                    if not task:
                        logger.error("Task not found for assignment", task_id=task_id)
                        return False
            
            # Assign optimal persona to agent for this task
            persona_assignment = None
            if self.persona_system:
                try:
                    persona_assignment = await self.persona_system.assign_persona_to_agent(
                        agent_id=uuid.UUID(agent_id),
                        task=task,
                        context=context or {},
                        preferred_persona_id=preferred_persona_id
                    )
                    
                    logger.info(
                        "Persona assigned for task",
                        task_id=task_id,
                        agent_id=agent_id,
                        persona_id=persona_assignment.persona_id,
                        confidence=persona_assignment.confidence_score
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to assign persona, proceeding without",
                        task_id=task_id,
                        agent_id=agent_id,
                        error=str(e)
                    )
            
            # Update task in database with enhanced metadata
            async with get_session() as db_session:
                update_values = {
                    'assigned_agent_id': agent_id,
                    'status': TaskStatus.ASSIGNED,
                    'assigned_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow()
                }
                
                # Add persona information to task metadata
                if persona_assignment:
                    task_metadata = task.metadata or {}
                    task_metadata.update({
                        'assigned_persona_id': persona_assignment.persona_id,
                        'persona_confidence': persona_assignment.confidence_score,
                        'persona_adaptations': persona_assignment.active_adaptations
                    })
                    update_values['metadata'] = task_metadata
                
                await db_session.execute(
                    update(Task)
                    .where(Task.id == task_id)
                    .values(**update_values)
                )
                await db_session.commit()
            
            # Update agent state
            if agent_id in self.agents:
                self.agents[agent_id].current_task = task_id
            
            # Prepare enhanced task assignment payload
            assignment_payload = {
                "task_id": task_id,
                "task_metadata": task.metadata or {},
                "priority": task.priority.value,
                "estimated_effort": task.estimated_effort,
                "context": context or {}
            }
            
            # Add persona information to payload
            if persona_assignment:
                assignment_payload.update({
                    "persona_id": persona_assignment.persona_id,
                    "persona_adaptations": persona_assignment.active_adaptations,
                    "assignment_reason": persona_assignment.assignment_reason
                })
            
            # Send enhanced task assignment message
            await self.message_broker.send_message(
                from_agent="orchestrator",
                to_agent=agent_id,
                message_type="enhanced_task_assignment",
                payload=assignment_payload
            )
            
            logger.info(
                "Enhanced task assignment completed",
                task_id=task_id,
                agent_id=agent_id,
                has_persona=persona_assignment is not None
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to assign task to agent with persona",
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
    
    async def execute_workflow(self, workflow_id: str, context: Optional[Dict[str, Any]] = None) -> WorkflowResult:
        """
        Execute a workflow using the enhanced workflow engine with persona coordination.
        
        Args:
            workflow_id: UUID of the workflow to execute
            context: Additional context for workflow execution
            
        Returns:
            WorkflowResult with execution status and metrics
        """
        if not self.workflow_engine:
            raise RuntimeError("Workflow engine not initialized")
        
        logger.info("🎯 Orchestrator executing enhanced workflow", workflow_id=workflow_id)
        
        try:
            # Pre-execution planning with persona optimization
            await self._prepare_workflow_execution(workflow_id, context or {})
            
            # Execute workflow with enhanced coordination
            result = await self.workflow_engine.execute_workflow(workflow_id)
            
            # Post-execution persona performance updates
            await self._update_workflow_persona_performance(workflow_id, result)
            
            # Update orchestrator metrics
            self.metrics['workflows_executed'] += 1
            if result.status == WorkflowStatus.COMPLETED:
                self.metrics['workflows_completed'] += 1
            
            logger.info(
                "🎯 Enhanced workflow execution completed",
                workflow_id=workflow_id,
                status=result.status.value,
                completed_tasks=result.completed_tasks,
                failed_tasks=result.failed_tasks
            )
            
            return result
            
        except Exception as e:
            logger.error("❌ Enhanced workflow execution failed", workflow_id=workflow_id, error=str(e))
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
        """Handle task completion callback from agents for workflow tasks with persona updates."""
        if self.workflow_engine:
            await self.workflow_engine.handle_task_completion(task_id, result)
        
        # Update persona performance metrics
        if self.persona_system:
            await self._handle_task_completion_persona_update(task_id, result)
    
    async def _prepare_workflow_execution(self, workflow_id: str, context: Dict[str, Any]) -> None:
        """Prepare workflow execution with persona optimization and resource allocation."""
        try:
            # Load workflow details
            async with get_session() as db_session:
                from ..models.workflow import Workflow
                workflow = await db_session.get(Workflow, workflow_id)
                if not workflow:
                    raise ValueError(f"Workflow {workflow_id} not found")
            
            # Analyze workflow tasks for optimal persona assignments
            if workflow.task_ids and self.persona_system:
                await self._optimize_workflow_persona_assignments(workflow, context)
            
            # Pre-allocate agents based on workflow requirements
            await self._pre_allocate_workflow_agents(workflow, context)
            
            logger.info(
                "Workflow execution prepared",
                workflow_id=workflow_id,
                task_count=len(workflow.task_ids) if workflow.task_ids else 0
            )
            
        except Exception as e:
            logger.error("Failed to prepare workflow execution", workflow_id=workflow_id, error=str(e))
            # Continue execution even if preparation fails
    
    async def _optimize_workflow_persona_assignments(
        self, 
        workflow, 
        context: Dict[str, Any]
    ) -> None:
        """Optimize persona assignments for workflow tasks based on dependencies and requirements."""
        try:
            if not workflow.task_ids:
                return
            
            # Load all workflow tasks
            async with get_session() as db_session:
                tasks_result = await db_session.execute(
                    select(Task).where(Task.id.in_(workflow.task_ids))
                )
                tasks = tasks_result.scalars().all()
            
            # Group tasks by type and requirements for optimal persona allocation
            task_groups = defaultdict(list)
            for task in tasks:
                # Group by task type and required capabilities
                key = (task.task_type, tuple(sorted(task.required_capabilities or [])))
                task_groups[key].append(task)
            
            # Pre-select personas for each task group
            persona_recommendations = {}
            for (task_type, capabilities), group_tasks in task_groups.items():
                # Get available personas suitable for this task type
                suitable_personas = await self.persona_system.list_available_personas(
                    task_type=task_type,
                    required_capabilities=list(capabilities)
                )
                
                if suitable_personas:
                    # Select best persona for this group
                    best_persona = suitable_personas[0]  # Could be enhanced with scoring
                    for task in group_tasks:
                        persona_recommendations[str(task.id)] = best_persona.id
            
            # Store recommendations in workflow context
            context['persona_recommendations'] = persona_recommendations
            
            logger.info(
                "Workflow persona assignments optimized",
                workflow_id=str(workflow.id),
                task_groups=len(task_groups),
                recommendations=len(persona_recommendations)
            )
            
        except Exception as e:
            logger.error("Failed to optimize workflow persona assignments", error=str(e))
    
    async def _pre_allocate_workflow_agents(self, workflow, context: Dict[str, Any]) -> None:
        """Pre-allocate agents for workflow execution based on estimated requirements."""
        try:
            if not workflow.task_ids:
                return
            
            # Calculate required agent capacity
            estimated_parallel_tasks = min(len(workflow.task_ids), 3)  # Max 3 parallel by default
            available_agents = len([a for a in self.agents.values() if a.status == AgentStatus.ACTIVE])
            
            # Spawn additional agents if needed
            agents_needed = max(0, estimated_parallel_tasks - available_agents)
            if agents_needed > 0:
                logger.info(
                    "Pre-allocating additional agents for workflow",
                    workflow_id=str(workflow.id),
                    agents_needed=agents_needed
                )
                
                # Spawn diverse agent types based on workflow task types
                task_types = set()
                async with get_session() as db_session:
                    tasks_result = await db_session.execute(
                        select(Task.task_type).where(Task.id.in_(workflow.task_ids))
                    )
                    task_types = {t[0] for t in tasks_result.fetchall()}
                
                # Map task types to agent roles
                role_mapping = {
                    'code_generation': AgentRole.BACKEND_DEVELOPER,
                    'testing': AgentRole.QA_ENGINEER,
                    'deployment': AgentRole.DEVOPS_ENGINEER,
                    'code_review': AgentRole.ARCHITECT
                }
                
                roles_needed = [role_mapping.get(tt, AgentRole.BACKEND_DEVELOPER) for tt in task_types]
                
                # Spawn agents with needed roles
                for i in range(min(agents_needed, len(roles_needed))):
                    try:
                        await self.spawn_agent(roles_needed[i])
                    except Exception as e:
                        logger.warning(f"Failed to spawn additional agent: {e}")
            
        except Exception as e:
            logger.error("Failed to pre-allocate workflow agents", error=str(e))
    
    async def _update_workflow_persona_performance(
        self, 
        workflow_id: str, 
        result: WorkflowResult
    ) -> None:
        """Update persona performance metrics based on workflow execution results."""
        try:
            if not self.persona_system or not result.task_results:
                return
            
            # Update performance for each task result
            for task_result in result.task_results:
                if task_result.agent_id:
                    try:
                        # Get task details
                        async with get_session() as db_session:
                            task = await db_session.get(Task, task_result.task_id)
                            if task:
                                success = task_result.status == TaskExecutionState.COMPLETED
                                completion_time = task_result.execution_time or 0.0
                                
                                # Update persona performance
                                await self.persona_system.update_persona_performance(
                                    agent_id=uuid.UUID(task_result.agent_id),
                                    task=task,
                                    success=success,
                                    completion_time=completion_time
                                )
                    except Exception as e:
                        logger.warning(
                            "Failed to update persona performance for task",
                            task_id=task_result.task_id,
                            error=str(e)
                        )
            
            logger.info(
                "Workflow persona performance updated",
                workflow_id=workflow_id,
                updated_tasks=len(result.task_results)
            )
            
        except Exception as e:
            logger.error("Failed to update workflow persona performance", error=str(e))
    
    async def _handle_task_completion_persona_update(
        self, 
        task_id: str, 
        result: Dict[str, Any]
    ) -> None:
        """Handle persona performance updates when individual tasks complete."""
        try:
            # Extract agent ID from result or find it in active agents
            agent_id = result.get('agent_id')
            if not agent_id:
                # Find agent assigned to this task
                for aid, agent in self.agents.items():
                    if agent.current_task == task_id:
                        agent_id = aid
                        break
            
            if not agent_id:
                return
            
            # Get task details
            async with get_session() as db_session:
                task = await db_session.get(Task, task_id)
                if not task:
                    return
            
            # Update persona performance
            success = result.get('success', True)
            completion_time = result.get('completion_time', 0.0)
            complexity = result.get('complexity', 0.5)
            
            await self.persona_system.update_persona_performance(
                agent_id=uuid.UUID(agent_id),
                task=task,
                success=success,
                completion_time=completion_time,
                complexity=complexity
            )
            
            logger.debug(
                "Persona performance updated for task completion",
                task_id=task_id,
                agent_id=agent_id,
                success=success
            )
            
        except Exception as e:
            logger.error(
                "Failed to handle task completion persona update",
                task_id=task_id,
                error=str(e)
            )
    
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
    
    # Enhanced intelligent routing methods
    
    async def rebalance_agent_workloads(self, force_rebalance: bool = False) -> Dict[str, Any]:
        """Enhanced workload rebalancing with persona-aware optimization."""
        if not self.intelligent_router:
            return {"error": "Intelligent router not initialized"}
        
        try:
            # Perform workload analysis
            workload_analysis = await self._analyze_agent_workloads()
            
            # Skip rebalancing if loads are already well-balanced (unless forced)
            if not force_rebalance and workload_analysis['balance_score'] > 0.8:
                logger.info("Workloads already well-balanced, skipping rebalancing")
                return {
                    "skipped": True,
                    "reason": "workloads_balanced",
                    "balance_score": workload_analysis['balance_score']
                }
            
            # Perform intelligent rebalancing with persona consideration
            reassignments = await self._intelligent_workload_rebalancing(workload_analysis)
            
            # Execute approved reassignments
            executed_reassignments = []
            for reassignment in reassignments:
                if reassignment.expected_improvement > 0.15:  # Adjusted threshold
                    success = await self._execute_task_reassignment_with_persona(reassignment)
                    if success:
                        executed_reassignments.append(reassignment)
                        self.metrics['load_balancing_actions'] += 1
            
            # Update workload balance metrics
            post_balance_analysis = await self._analyze_agent_workloads()
            improvement = post_balance_analysis['balance_score'] - workload_analysis['balance_score']
            
            logger.info(
                "Enhanced workload rebalancing completed",
                total_reassignments=len(reassignments),
                executed_reassignments=len(executed_reassignments),
                balance_improvement=improvement
            )
            
            return {
                "total_recommendations": len(reassignments),
                "executed_reassignments": len(executed_reassignments),
                "balance_improvement": improvement,
                "pre_balance_score": workload_analysis['balance_score'],
                "post_balance_score": post_balance_analysis['balance_score'],
                "reassignments": [self._reassignment_to_dict(r) for r in executed_reassignments]
            }
            
        except Exception as e:
            logger.error("Error in enhanced workload rebalancing", error=str(e))
            return {"error": str(e)}
    
    async def get_routing_analytics(self) -> Dict[str, Any]:
        """Get intelligent routing analytics and performance metrics."""
        try:
            async with get_session() as db_session:
                # Calculate routing accuracy
                total_decisions_query = select(func.count(TaskRoutingDecision.id))
                successful_decisions_query = select(func.count(TaskRoutingDecision.id)).where(
                    TaskRoutingDecision.task_success == True
                )
                
                total_decisions = (await db_session.execute(total_decisions_query)).scalar() or 0
                successful_decisions = (await db_session.execute(successful_decisions_query)).scalar() or 0
                
                routing_accuracy = successful_decisions / total_decisions if total_decisions > 0 else 0.0
                self.metrics['routing_accuracy'] = routing_accuracy
                
                # Get recent performance trends
                recent_date = datetime.utcnow() - timedelta(days=7)
                recent_performance_query = select(
                    TaskRoutingDecision.routing_strategy,
                    func.avg(TaskRoutingDecision.final_score),
                    func.count(TaskRoutingDecision.id)
                ).where(
                    TaskRoutingDecision.decided_at >= recent_date
                ).group_by(TaskRoutingDecision.routing_strategy)
                
                strategy_performance = {}
                result = await db_session.execute(recent_performance_query)
                for row in result:
                    strategy, avg_score, count = row
                    strategy_performance[strategy] = {
                        "average_score": float(avg_score or 0.0),
                        "decision_count": count
                    }
                
                return {
                    "routing_accuracy": routing_accuracy,
                    "total_routing_decisions": self.metrics['routing_decisions'],
                    "load_balancing_actions": self.metrics['load_balancing_actions'],
                    "strategy_performance": strategy_performance,
                    "agent_utilization": await self._calculate_agent_utilization_stats()
                }
                
        except Exception as e:
            logger.error("Error getting routing analytics", error=str(e))
            return {"error": str(e)}
    
    async def update_task_completion_metrics(
        self,
        task_id: str,
        agent_id: str,
        success: bool,
        completion_time: Optional[float] = None
    ) -> None:
        """Update performance metrics when a task is completed."""
        try:
            # Update intelligent router performance data
            if self.intelligent_router:
                task_result = {
                    "success": success,
                    "completion_time": completion_time,
                    "task_id": task_id
                }
                await self.intelligent_router.update_agent_performance(agent_id, task_result)
            
            # Record performance history
            async with get_session() as db_session:
                # Get task details
                task = await db_session.get(Task, task_id)
                if not task:
                    return
                
                # Create performance history record
                performance_record = AgentPerformanceHistory(
                    agent_id=agent_id,
                    task_id=task_id,
                    task_type=task.task_type.value if task.task_type else "general",
                    success=success,
                    completion_time_minutes=completion_time,
                    estimated_time_minutes=task.estimated_effort,
                    time_variance_ratio=completion_time / task.estimated_effort if task.estimated_effort and completion_time else None,
                    retry_count=task.retry_count,
                    priority_level=task.priority.value,
                    required_capabilities=task.required_capabilities,
                    started_at=task.started_at,
                    completed_at=task.completed_at or datetime.utcnow()
                )
                
                db_session.add(performance_record)
                
                # Update routing decision outcome
                routing_decision_query = select(TaskRoutingDecision).where(
                    TaskRoutingDecision.task_id == task_id
                )
                routing_decision = (await db_session.execute(routing_decision_query)).scalar_one_or_none()
                
                if routing_decision:
                    routing_decision.task_completed = True
                    routing_decision.task_success = success
                    routing_decision.actual_completion_time = completion_time
                    routing_decision.outcome_recorded_at = datetime.utcnow()
                    routing_decision.outcome_score = 1.0 if success else 0.0
                
                await db_session.commit()
                
                # Update orchestrator metrics
                if success:
                    self.metrics['tasks_completed'] += 1
                else:
                    self.metrics['tasks_failed'] += 1
                
                logger.debug(
                    "Task completion metrics updated",
                    task_id=task_id,
                    agent_id=agent_id,
                    success=success,
                    completion_time=completion_time
                )
                
        except Exception as e:
            logger.error("Error updating task completion metrics", task_id=task_id, error=str(e))
    
    async def _get_available_agent_ids(self) -> List[str]:
        """Get list of available agent IDs for task assignment."""
        available_agents = []
        for agent_id, agent_instance in self.agents.items():
            if (agent_instance.status == AgentStatus.ACTIVE and 
                agent_instance.current_task is None and
                agent_instance.context_window_usage < 0.9):
                available_agents.append(agent_id)
        return available_agents
    
    async def _record_routing_analytics(
        self,
        task_id: str,
        selected_agent_id: str,
        routing_context: TaskRoutingContext,
        strategy: RoutingStrategy
    ) -> None:
        """Record routing decision for analytics and learning."""
        try:
            async with get_session() as db_session:
                # Get suitability scores for analytics
                available_agents = await self._get_available_agent_ids()
                agent_scores = {}
                
                if self.intelligent_router:
                    for agent_id in available_agents[:5]:  # Limit to top 5 for analytics
                        score = await self.intelligent_router.calculate_agent_suitability(
                            agent_id, routing_context
                        )
                        if score:
                            agent_scores[agent_id] = {
                                "total_score": score.total_score,
                                "capability_score": score.capability_score,
                                "performance_score": score.performance_score,
                                "availability_score": score.availability_score
                            }
                
                # Create routing decision record
                routing_decision = TaskRoutingDecision(
                    task_id=task_id,
                    selected_agent_id=selected_agent_id,
                    routing_strategy=strategy.value,
                    candidate_agents=available_agents,
                    agent_scores=agent_scores,
                    final_score=agent_scores.get(selected_agent_id, {}).get("total_score", 0.0),
                    confidence_level=0.8,  # Placeholder - could be enhanced
                    selection_criteria={
                        "required_capabilities": routing_context.required_capabilities,
                        "priority": routing_context.priority.value,
                        "task_type": routing_context.task_type
                    }
                )
                
                db_session.add(routing_decision)
                await db_session.commit()
                
        except Exception as e:
            logger.error("Error recording routing analytics", task_id=task_id, error=str(e))
    
    async def _execute_task_reassignment(self, reassignment) -> bool:
        """Legacy task reassignment - use _execute_task_reassignment_with_persona instead."""
        return await self._execute_task_reassignment_with_persona(reassignment)
    
    async def _execute_task_reassignment_with_persona(self, reassignment) -> bool:
        """Execute a task reassignment with persona coordination."""
        try:
            # Get task details for persona coordination
            async with get_session() as db_session:
                task = await db_session.get(Task, reassignment.task_id)
                if not task:
                    logger.error("Task not found for reassignment", task_id=reassignment.task_id)
                    return False
                
                # Handle persona reassignment
                if self.persona_system:
                    # Remove persona from source agent
                    await self.persona_system.remove_persona_assignment(
                        uuid.UUID(reassignment.from_agent_id)
                    )
                    
                    # Assign optimal persona to target agent
                    await self.persona_system.assign_persona_to_agent(
                        agent_id=uuid.UUID(reassignment.to_agent_id),
                        task=task,
                        context={'reassignment': True, 'reason': reassignment.reason}
                    )
                
                # Update task assignment
                await db_session.execute(
                    update(Task)
                    .where(Task.id == reassignment.task_id)
                    .values(
                        assigned_agent_id=reassignment.to_agent_id,
                        updated_at=datetime.utcnow()
                    )
                )
                
                # Update agent states
                if reassignment.from_agent_id in self.agents:
                    from_agent = self.agents[reassignment.from_agent_id]
                    if from_agent.current_task == reassignment.task_id:
                        from_agent.current_task = None
                
                if reassignment.to_agent_id in self.agents:
                    to_agent = self.agents[reassignment.to_agent_id]
                    to_agent.current_task = reassignment.task_id
                
                await db_session.commit()
                
                # Send enhanced reassignment messages
                await self.message_broker.send_message(
                    from_agent="orchestrator",
                    to_agent=reassignment.from_agent_id,
                    message_type="enhanced_task_reassignment_remove",
                    payload={
                        "task_id": reassignment.task_id, 
                        "reason": reassignment.reason,
                        "expected_improvement": reassignment.expected_improvement
                    }
                )
                
                await self.message_broker.send_message(
                    from_agent="orchestrator",
                    to_agent=reassignment.to_agent_id,
                    message_type="enhanced_task_assignment",
                    payload={
                        "task_id": reassignment.task_id,
                        "reassignment": True,
                        "from_agent": reassignment.from_agent_id,
                        "task_metadata": task.metadata or {},
                        "priority": task.priority.value
                    }
                )
                
                logger.info(
                    "Enhanced task reassignment completed",
                    task_id=reassignment.task_id,
                    from_agent=reassignment.from_agent_id,
                    to_agent=reassignment.to_agent_id,
                    reason=reassignment.reason,
                    expected_improvement=reassignment.expected_improvement
                )
                
                return True
                
        except Exception as e:
            logger.error(
                "Error executing enhanced task reassignment", 
                reassignment=self._reassignment_to_dict(reassignment), 
                error=str(e)
            )
            return False
    
    async def _calculate_agent_utilization_stats(self) -> Dict[str, Any]:
        """Calculate agent utilization statistics."""
        try:
            utilization_stats = {}
            
            for agent_id, agent_instance in self.agents.items():
                if self.capability_matcher:
                    workload_factor = await self.capability_matcher.get_workload_factor(agent_id)
                    utilization_stats[agent_id] = {
                        "status": agent_instance.status.value,
                        "workload_factor": workload_factor,
                        "context_usage": agent_instance.context_window_usage,
                        "current_task": agent_instance.current_task,
                        "role": agent_instance.role.value
                    }
            
            return utilization_stats
            
        except Exception as e:
            logger.error("Error calculating agent utilization stats", error=str(e))
            return {}
    
    async def _workload_monitoring_loop(self) -> None:
        """Background task to monitor and optimize agent workloads."""
        while self.is_running:
            try:
                # Take workload snapshots
                await self._take_workload_snapshots()
                
                # Perform periodic rebalancing if needed
                if self.metrics['routing_decisions'] % 10 == 0:  # Every 10 routing decisions
                    await self.rebalance_agent_workloads()
                
                # Sleep for monitoring interval
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error("Error in workload monitoring loop", error=str(e))
                await asyncio.sleep(60)
    
    async def _take_workload_snapshots(self) -> None:
        """Take snapshots of current agent workloads."""
        try:
            async with get_session() as db_session:
                for agent_id, agent_instance in self.agents.items():
                    if self.capability_matcher:
                        # Get current workload metrics
                        workload_metrics = await self.capability_matcher._get_agent_workload_metrics(agent_id)
                        
                        if workload_metrics:
                            snapshot = WorkloadSnapshot(
                                agent_id=agent_id,
                                active_tasks=workload_metrics.active_tasks,
                                pending_tasks=workload_metrics.pending_tasks,
                                context_usage_percent=workload_metrics.context_usage * 100,
                                estimated_capacity=1.0,  # Could be dynamic
                                utilization_ratio=await self.capability_matcher.get_workload_factor(agent_id),
                                priority_distribution={
                                    str(k.value): v for k, v in workload_metrics.priority_distribution.items()
                                },
                                task_type_distribution=workload_metrics.task_type_distribution
                            )
                            
                            db_session.add(snapshot)
                
                await db_session.commit()
                
        except Exception as e:
            logger.error("Error taking workload snapshots", error=str(e))
    
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
    
    # Enhanced intelligent task delegation methods
    
    async def _analyze_agent_workloads(self) -> Dict[str, Any]:
        """Analyze current agent workloads for optimization opportunities."""
        try:
            workload_data = {}
            total_workload = 0.0
            agent_count = 0
            
            for agent_id, agent_instance in self.agents.items():
                if agent_instance.status == AgentStatus.ACTIVE:
                    if self.capability_matcher:
                        workload_factor = await self.capability_matcher.get_workload_factor(agent_id)
                    else:
                        # Fallback calculation
                        workload_factor = 0.5 if agent_instance.current_task else 0.0
                    
                    workload_data[agent_id] = {
                        'workload_factor': workload_factor,
                        'context_usage': agent_instance.context_window_usage,
                        'current_task': agent_instance.current_task,
                        'role': agent_instance.role.value,
                        'last_heartbeat': agent_instance.last_heartbeat
                    }
                    
                    total_workload += workload_factor
                    agent_count += 1
            
            # Calculate balance metrics
            if agent_count == 0:
                balance_score = 1.0  # No agents to balance
                avg_workload = 0.0
                workload_variance = 0.0
            else:
                avg_workload = total_workload / agent_count
                workload_variance = sum(
                    (data['workload_factor'] - avg_workload) ** 2 
                    for data in workload_data.values()
                ) / agent_count
                
                # Balance score: higher is better (0.0 to 1.0)
                # Lower variance = better balance
                balance_score = max(0.0, 1.0 - (workload_variance * 2))
            
            return {
                'agent_workloads': workload_data,
                'total_workload': total_workload,
                'average_workload': avg_workload,
                'workload_variance': workload_variance,
                'balance_score': balance_score,
                'active_agent_count': agent_count,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Error analyzing agent workloads", error=str(e))
            return {
                'error': str(e),
                'balance_score': 0.0,
                'agent_workloads': {},
                'active_agent_count': 0
            }
    
    async def _intelligent_workload_rebalancing(
        self, 
        workload_analysis: Dict[str, Any]
    ) -> List:
        """Perform intelligent workload rebalancing with persona consideration."""
        try:
            reassignments = []
            agent_workloads = workload_analysis.get('agent_workloads', {})
            avg_workload = workload_analysis.get('average_workload', 0.0)
            
            if not agent_workloads or avg_workload == 0.0:
                return reassignments
            
            # Identify overloaded and underloaded agents
            overloaded_threshold = avg_workload + 0.3
            underloaded_threshold = avg_workload - 0.3
            
            overloaded_agents = [
                (agent_id, data) for agent_id, data in agent_workloads.items()
                if data['workload_factor'] > overloaded_threshold
            ]
            
            underloaded_agents = [
                (agent_id, data) for agent_id, data in agent_workloads.items()
                if data['workload_factor'] < underloaded_threshold
            ]
            
            # Use intelligent router for reassignment recommendations
            if self.intelligent_router and overloaded_agents and underloaded_agents:
                router_reassignments = await self.intelligent_router.rebalance_workload()
                
                # Enhance with persona-aware filtering
                for reassignment in router_reassignments:
                    # Check if persona compatibility improves with reassignment
                    persona_improvement = await self._calculate_persona_reassignment_benefit(
                        reassignment.task_id,
                        reassignment.from_agent_id,
                        reassignment.to_agent_id
                    )
                    
                    # Adjust expected improvement based on persona compatibility
                    total_improvement = reassignment.expected_improvement + persona_improvement
                    
                    if total_improvement > 0.1:  # Minimum benefit threshold
                        # Create enhanced reassignment with persona consideration
                        enhanced_reassignment = type('EnhancedReassignment', (), {
                            'task_id': reassignment.task_id,
                            'from_agent_id': reassignment.from_agent_id,
                            'to_agent_id': reassignment.to_agent_id,
                            'reason': f"{reassignment.reason} + persona_optimization",
                            'expected_improvement': total_improvement,
                            'persona_improvement': persona_improvement,
                            'workload_improvement': reassignment.expected_improvement
                        })()
                        
                        reassignments.append(enhanced_reassignment)
            
            logger.info(
                "Intelligent workload rebalancing analysis completed",
                overloaded_agents=len(overloaded_agents),
                underloaded_agents=len(underloaded_agents),
                potential_reassignments=len(reassignments)
            )
            
            return reassignments
            
        except Exception as e:
            logger.error("Error in intelligent workload rebalancing", error=str(e))
            return []
    
    async def _calculate_persona_reassignment_benefit(
        self,
        task_id: str,
        from_agent_id: str,
        to_agent_id: str
    ) -> float:
        """Calculate the persona-based benefit of reassigning a task."""
        try:
            if not self.persona_system:
                return 0.0
            
            # Get task details
            async with get_session() as db_session:
                task = await db_session.get(Task, task_id)
                if not task:
                    return 0.0
            
            # Get current persona assignments
            from_agent_persona = await self.persona_system.get_agent_current_persona(
                uuid.UUID(from_agent_id)
            )
            to_agent_persona = await self.persona_system.get_agent_current_persona(
                uuid.UUID(to_agent_id)
            )
            
            # Calculate persona affinity scores
            from_affinity = 0.5  # Default neutral
            to_affinity = 0.5    # Default neutral
            
            if from_agent_persona:
                from_persona = await self.persona_system.get_persona(from_agent_persona.persona_id)
                if from_persona:
                    from_affinity = from_persona.get_task_affinity(task.task_type)
            
            if to_agent_persona:
                to_persona = await self.persona_system.get_persona(to_agent_persona.persona_id)
                if to_persona:
                    to_affinity = to_persona.get_task_affinity(task.task_type)
            
            # Calculate improvement (positive if target agent is better suited)
            persona_improvement = to_affinity - from_affinity
            
            # Scale improvement to reasonable range (0.0 to 0.5)
            scaled_improvement = max(-0.25, min(0.5, persona_improvement * 0.5))
            
            return scaled_improvement
            
        except Exception as e:
            logger.error(
                "Error calculating persona reassignment benefit",
                task_id=task_id,
                error=str(e)
            )
            return 0.0
    
    def _reassignment_to_dict(self, reassignment) -> Dict[str, Any]:
        """Convert reassignment object to dictionary for serialization."""
        return {
            'task_id': reassignment.task_id,
            'from_agent_id': reassignment.from_agent_id,
            'to_agent_id': reassignment.to_agent_id,
            'reason': reassignment.reason,
            'expected_improvement': reassignment.expected_improvement,
            'persona_improvement': getattr(reassignment, 'persona_improvement', 0.0),
            'workload_improvement': getattr(reassignment, 'workload_improvement', 0.0)
        }
    
    # Circuit breaker and error handling methods
    
    async def _update_circuit_breaker(
        self, 
        agent_id: str, 
        success: bool, 
        error_type: str = None
    ) -> None:
        """Update circuit breaker state for an agent."""
        if agent_id not in self.circuit_breakers:
            self.circuit_breakers[agent_id] = {
                'state': 'closed',  # closed, open, half_open
                'failure_count': 0,
                'consecutive_failures': 0,
                'last_failure_time': None,
                'total_requests': 0,
                'successful_requests': 0,
                'trip_time': None,
                'last_error_type': None
            }
        
        breaker = self.circuit_breakers[agent_id]
        breaker['total_requests'] += 1
        
        if success:
            breaker['successful_requests'] += 1
            breaker['consecutive_failures'] = 0
            
            # Potentially close circuit breaker if in half-open state
            if breaker['state'] == 'half_open':
                breaker['state'] = 'closed'
                breaker['failure_count'] = 0
                logger.info(f"Circuit breaker closed for agent {agent_id}")
        else:
            breaker['failure_count'] += 1
            breaker['consecutive_failures'] += 1
            breaker['last_failure_time'] = time.time()
            breaker['last_error_type'] = error_type
    
    async def _should_trip_circuit_breaker(self, agent_id: str) -> bool:
        """Determine if circuit breaker should trip for an agent."""
        if agent_id not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[agent_id]
        total_requests = breaker['total_requests']
        
        # Don't trip if not enough requests
        if total_requests < 10:
            return False
        
        # Trip on consecutive failures
        if breaker['consecutive_failures'] >= self.error_thresholds['consecutive_failures']:
            return True
        
        # Trip on failure rate
        failure_rate = breaker['failure_count'] / total_requests
        if failure_rate >= self.error_thresholds['agent_failure_rate']:
            return True
        
        return False
    
    async def _trip_circuit_breaker(self, agent_id: str, reason: str) -> None:
        """Trip circuit breaker for an agent."""
        if agent_id not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[agent_id]
        breaker['state'] = 'open'
        breaker['trip_time'] = time.time()
        
        self.metrics['circuit_breaker_trips'] += 1
        
        logger.warning(
            f"🚫 Circuit breaker tripped for agent {agent_id}",
            reason=reason,
            failure_count=breaker['failure_count'],
            consecutive_failures=breaker['consecutive_failures']
        )
        
        # Remove agent from available pool temporarily
        if agent_id in self.agents:
            self.agents[agent_id].status = AgentStatus.ERROR  # Use existing status
        
        # Schedule recovery attempt
        asyncio.create_task(self._schedule_circuit_breaker_recovery(agent_id))
    
    async def _schedule_circuit_breaker_recovery(self, agent_id: str) -> None:
        """Schedule circuit breaker recovery attempt."""
        try:
            recovery_time = self.error_thresholds['recovery_time_seconds']
            await asyncio.sleep(recovery_time)
            
            if agent_id in self.circuit_breakers:
                breaker = self.circuit_breakers[agent_id]
                if breaker['state'] == 'open':
                    breaker['state'] = 'half_open'
                    
                    logger.info(f"🔄 Circuit breaker half-open for agent {agent_id}")
                    
                    # Try a test request
                    await self._test_circuit_breaker_recovery(agent_id)
                    
        except Exception as e:
            logger.error(f"Error in circuit breaker recovery scheduling", agent_id=agent_id, error=str(e))
    
    async def _test_circuit_breaker_recovery(self, agent_id: str) -> None:
        """Test if agent is ready for circuit breaker recovery."""
        try:
            if agent_id not in self.agents:
                return
            
            agent = self.agents[agent_id]
            
            # Send health check message
            test_message = {
                "type": "health_check",
                "timestamp": datetime.utcnow().isoformat(),
                "circuit_breaker_test": True
            }
            
            try:
                await self.message_broker.send_message(
                    from_agent="orchestrator",
                    to_agent=agent_id,
                    message_type="health_check",
                    payload=test_message
                )
                
                # Wait for response (simplified - in production would use proper response handling)
                await asyncio.sleep(5)
                
                # Check if agent responded (simplified check)
                current_time = datetime.utcnow()
                if (current_time - agent.last_heartbeat).total_seconds() < 30:
                    # Agent is responsive
                    await self._update_circuit_breaker(agent_id, success=True)
                    agent.status = AgentStatus.ACTIVE
                    
                    logger.info(f"✅ Circuit breaker recovery successful for agent {agent_id}")
                    self.metrics['automatic_recovery_actions'] += 1
                else:
                    # Agent still not responding
                    await self._update_circuit_breaker(agent_id, success=False, error_type='recovery_test_failed')
                    
            except Exception as e:
                await self._update_circuit_breaker(agent_id, success=False, error_type='recovery_test_error')
                logger.warning(f"Circuit breaker recovery test failed", agent_id=agent_id, error=str(e))
                
        except Exception as e:
            logger.error(f"Error in circuit breaker recovery test", agent_id=agent_id, error=str(e))
    
    async def _attempt_agent_restart_with_protection(self, agent_id: str, agent: AgentInstance) -> bool:
        """Attempt to restart agent with circuit breaker protection."""
        try:
            # Check circuit breaker state
            if agent_id in self.circuit_breakers:
                breaker = self.circuit_breakers[agent_id]
                if breaker['state'] == 'open':
                    logger.info(f"Agent restart blocked by circuit breaker", agent_id=agent_id)
                    return False
            
            # Use exponential backoff for restart attempts
            restart_attempt = breaker.get('restart_attempts', 0) if agent_id in self.circuit_breakers else 0
            if restart_attempt > 0:
                backoff_time = self.error_thresholds['exponential_backoff_base'] ** restart_attempt
                backoff_time = min(backoff_time, 300)  # Cap at 5 minutes
                
                logger.info(f"Waiting {backoff_time}s before restart attempt", agent_id=agent_id)
                await asyncio.sleep(backoff_time)
            
            # Attempt restart with enhanced logging
            logger.info(f"🔄 Attempting protected restart of agent {agent_id}")
            
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
                # Update circuit breaker with successful restart
                await self._update_circuit_breaker(agent_id, success=True)
                
                # Reset restart attempts
                if agent_id in self.circuit_breakers:
                    self.circuit_breakers[agent_id]['restart_attempts'] = 0
                
                logger.info(f"✅ Protected agent restart successful", agent_id=agent_id)
                self.metrics['agents_restarted'] = self.metrics.get('agents_restarted', 0) + 1
                self.metrics['automatic_recovery_actions'] += 1
                return True
            else:
                # Update circuit breaker with failed restart
                await self._update_circuit_breaker(agent_id, success=False, error_type='restart_failed')
                
                # Increment restart attempts
                if agent_id in self.circuit_breakers:
                    self.circuit_breakers[agent_id]['restart_attempts'] = restart_attempt + 1
                
                logger.error(f"❌ Protected agent restart failed", agent_id=agent_id)
                return False
                
        except Exception as e:
            await self._update_circuit_breaker(agent_id, success=False, error_type='restart_exception')
            logger.error(f"❌ Error during protected agent restart", agent_id=agent_id, error=str(e))
            return False
    
    async def retry_with_exponential_backoff(
        self,
        operation: Callable,
        *args,
        max_retries: int = None,
        operation_name: str = "operation",
        **kwargs
    ) -> Any:
        """Execute operation with exponential backoff retry logic."""
        max_retries = max_retries or self.error_thresholds['max_retry_attempts']
        base_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                result = await operation(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(
                        f"✅ {operation_name} succeeded after {attempt} retries"
                    )
                
                return result
                
            except Exception as e:
                self.metrics['retry_attempts'] += 1
                
                if attempt == max_retries:
                    logger.error(
                        f"❌ {operation_name} failed after {max_retries} retries",
                        error=str(e)
                    )
                    raise
                
                # Calculate delay with jitter
                delay = base_delay * (self.error_thresholds['exponential_backoff_base'] ** attempt)
                jitter = random.uniform(0.1, 0.3) * delay
                total_delay = delay + jitter
                
                logger.warning(
                    f"⚠️ {operation_name} failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {total_delay:.2f}s",
                    error=str(e)
                )
                
                await asyncio.sleep(total_delay)
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status for all agents."""
        status = {
            'circuit_breakers': {},
            'summary': {
                'total_breakers': len(self.circuit_breakers),
                'open_breakers': 0,
                'half_open_breakers': 0,
                'closed_breakers': 0,
                'total_trips': self.metrics['circuit_breaker_trips'],
                'total_recoveries': self.metrics['automatic_recovery_actions']
            }
        }
        
        for agent_id, breaker in self.circuit_breakers.items():
            status['circuit_breakers'][agent_id] = {
                'state': breaker['state'],
                'failure_count': breaker['failure_count'],
                'consecutive_failures': breaker['consecutive_failures'],
                'success_rate': breaker['successful_requests'] / max(breaker['total_requests'], 1),
                'last_error_type': breaker['last_error_type'],
                'trip_time': breaker['trip_time']
            }
            
            # Update summary counts
            if breaker['state'] == 'open':
                status['summary']['open_breakers'] += 1
            elif breaker['state'] == 'half_open':
                status['summary']['half_open_breakers'] += 1
            else:
                status['summary']['closed_breakers'] += 1
        
        return status
    
    # Enhanced async task delegation and queuing methods
    
    async def enqueue_task(
        self,
        task_id: str,
        priority: TaskPriority,
        workflow_id: Optional[str] = None,
        retry_count: int = 0,
        estimated_effort: Optional[int] = None
    ) -> bool:
        """Add task to appropriate priority queue for processing."""
        try:
            with self.queue_lock:
                task_entry = {
                    'task_id': task_id,
                    'priority': priority,
                    'workflow_id': workflow_id,
                    'retry_count': retry_count,
                    'estimated_effort': estimated_effort or 60,
                    'queued_at': time.time(),
                    'queue_priority': self._calculate_queue_priority(priority, retry_count, estimated_effort)
                }
                
                if retry_count > 0:
                    # Add to retry queue with priority based on retry count (lower is higher priority)
                    heapq.heappush(self.task_queues['retry_queue'], (retry_count, time.time(), task_entry))
                    logger.info(f"Task queued for retry", task_id=task_id, retry_count=retry_count)
                    
                elif workflow_id:
                    # Add to workflow-specific queue
                    if workflow_id not in self.task_queues['workflow_tasks']:
                        self.task_queues['workflow_tasks'][workflow_id] = deque()
                    
                    self.task_queues['workflow_tasks'][workflow_id].append(task_entry)
                    logger.info(f"Task queued for workflow", task_id=task_id, workflow_id=workflow_id)
                    
                elif priority == TaskPriority.HIGH or priority == TaskPriority.CRITICAL:
                    # Add to high priority queue (using heapq for priority ordering)
                    heapq.heappush(
                        self.task_queues['high_priority'], 
                        (task_entry['queue_priority'], time.time(), task_entry)
                    )
                    logger.info(f"Task queued with high priority", task_id=task_id, priority=priority.value)
                    
                elif priority == TaskPriority.MEDIUM:
                    self.task_queues['medium_priority'].append(task_entry)
                    logger.info(f"Task queued with medium priority", task_id=task_id)
                    
                else:  # LOW priority
                    self.task_queues['low_priority'].append(task_entry)
                    logger.info(f"Task queued with low priority", task_id=task_id)
                
                return True
                
        except Exception as e:
            logger.error("Failed to enqueue task", task_id=task_id, error=str(e))
            return False
    
    def _calculate_queue_priority(self, priority: TaskPriority, retry_count: int, estimated_effort: Optional[int]) -> float:
        """Calculate numeric priority for queue ordering (lower = higher priority)."""
        base_priority = {
            TaskPriority.CRITICAL: 1.0,
            TaskPriority.HIGH: 2.0,
            TaskPriority.MEDIUM: 3.0,
            TaskPriority.LOW: 4.0
        }.get(priority, 3.0)
        
        # Boost priority for retries (but not too much)
        retry_boost = max(0, 1.0 - (retry_count * 0.2))
        
        # Slightly prioritize shorter tasks
        effort_factor = min(1.2, (estimated_effort or 60) / 60.0) if estimated_effort else 1.0
        
        return base_priority - retry_boost + (effort_factor * 0.1)
    
    async def _process_priority_queues(self) -> int:
        """Process tasks from priority queues in order."""
        assigned_count = 0
        
        try:
            # Process high priority queue first
            while self.task_queues['high_priority']:
                available_agents = await self._get_available_agent_ids()
                if not available_agents:
                    break
                
                with self.queue_lock:
                    if self.task_queues['high_priority']:
                        _, _, task_entry = heapq.heappop(self.task_queues['high_priority'])
                        
                        if await self._assign_queued_task(task_entry, available_agents):
                            assigned_count += 1
                        else:
                            # Re-queue if assignment failed
                            await self._requeue_failed_task(task_entry)
            
            # Process medium priority queue
            while self.task_queues['medium_priority'] and assigned_count < 3:  # Limit batch size
                available_agents = await self._get_available_agent_ids()
                if not available_agents:
                    break
                
                with self.queue_lock:
                    if self.task_queues['medium_priority']:
                        task_entry = self.task_queues['medium_priority'].popleft()
                        
                        if await self._assign_queued_task(task_entry, available_agents):
                            assigned_count += 1
                        else:
                            await self._requeue_failed_task(task_entry)
            
            # Process low priority queue (limited)
            low_priority_processed = 0
            while (self.task_queues['low_priority'] and 
                   assigned_count < 2 and 
                   low_priority_processed < 1):  # Very limited low priority processing
                
                available_agents = await self._get_available_agent_ids()
                if not available_agents:
                    break
                
                with self.queue_lock:
                    if self.task_queues['low_priority']:
                        task_entry = self.task_queues['low_priority'].popleft()
                        
                        if await self._assign_queued_task(task_entry, available_agents):
                            assigned_count += 1
                            low_priority_processed += 1
                        else:
                            await self._requeue_failed_task(task_entry)
            
        except Exception as e:
            logger.error("Error processing priority queues", error=str(e))
        
        return assigned_count
    
    async def _process_workflow_queues(self) -> int:
        """Process workflow-specific task queues with dependency awareness."""
        assigned_count = 0
        
        try:
            with self.queue_lock:
                workflow_ids = list(self.task_queues['workflow_tasks'].keys())
            
            for workflow_id in workflow_ids:
                # Check if workflow is ready for task processing
                if await self._is_workflow_ready_for_tasks(workflow_id):
                    available_agents = await self._get_available_agent_ids()
                    if not available_agents:
                        break
                    
                    with self.queue_lock:
                        workflow_queue = self.task_queues['workflow_tasks'].get(workflow_id)
                        if workflow_queue:
                            task_entry = workflow_queue.popleft()
                            
                            if await self._assign_queued_task(task_entry, available_agents):
                                assigned_count += 1
                            else:
                                await self._requeue_failed_task(task_entry)
                            
                            # Clean up empty queues
                            if not workflow_queue:
                                del self.task_queues['workflow_tasks'][workflow_id]
                
        except Exception as e:
            logger.error("Error processing workflow queues", error=str(e))
        
        return assigned_count
    
    async def _process_retry_queue(self) -> int:
        """Process retry queue with exponential backoff consideration."""
        assigned_count = 0
        
        try:
            current_time = time.time()
            
            while self.task_queues['retry_queue'] and assigned_count < 2:
                with self.queue_lock:
                    if not self.task_queues['retry_queue']:
                        break
                    
                    # Peek at the next retry task
                    retry_count, queued_time, task_entry = self.task_queues['retry_queue'][0]
                    
                    # Check if enough time has passed for retry (exponential backoff)
                    backoff_time = (2 ** retry_count) * 30  # 30s, 60s, 120s, etc.
                    if current_time - queued_time < backoff_time:
                        break  # Not ready for retry yet
                    
                    # Remove from queue
                    heapq.heappop(self.task_queues['retry_queue'])
                
                available_agents = await self._get_available_agent_ids()
                if not available_agents:
                    # Re-queue if no agents available
                    with self.queue_lock:
                        heapq.heappush(
                            self.task_queues['retry_queue'], 
                            (retry_count, current_time, task_entry)
                        )
                    break
                
                if await self._assign_queued_task(task_entry, available_agents):
                    assigned_count += 1
                    logger.info(
                        f"Retry task assigned successfully",
                        task_id=task_entry['task_id'],
                        retry_count=retry_count
                    )
                else:
                    # Increment retry count and re-queue
                    task_entry['retry_count'] = retry_count + 1
                    if task_entry['retry_count'] <= self.error_thresholds['max_retry_attempts']:
                        with self.queue_lock:
                            heapq.heappush(
                                self.task_queues['retry_queue'], 
                                (task_entry['retry_count'], current_time, task_entry)
                            )
                    else:
                        logger.error(
                            f"Task exceeded maximum retry attempts",
                            task_id=task_entry['task_id'],
                            max_retries=self.error_thresholds['max_retry_attempts']
                        )
        
        except Exception as e:
            logger.error("Error processing retry queue", error=str(e))
        
        return assigned_count
    
    async def _assign_queued_task(self, task_entry: Dict[str, Any], available_agents: List[str]) -> bool:
        """Assign a queued task to an available agent."""
        try:
            task_id = task_entry['task_id']
            
            # Get task details from database
            async with get_session() as db_session:
                task = await db_session.get(Task, task_id)
                if not task:
                    logger.error(f"Queued task not found in database", task_id=task_id)
                    return False
                
                # Check if task is still in a state that can be assigned
                if task.status not in [TaskStatus.PENDING, TaskStatus.ASSIGNED]:
                    logger.info(f"Task no longer assignable", task_id=task_id, status=task.status.value)
                    return True  # Consider it successful to remove from queue
            
            # Use intelligent routing to select best agent
            if self.intelligent_router:
                routing_context = TaskRoutingContext(
                    task_id=task_id,
                    task_type=task.task_type.value if task.task_type else "general",
                    priority=task.priority,
                    required_capabilities=task.required_capabilities or [],
                    estimated_effort=task_entry.get('estimated_effort'),
                    due_date=task.due_date,
                    dependencies=task.dependencies or [],
                    workflow_id=task_entry.get('workflow_id'),
                    context=task.context or {}
                )
                
                selected_agent = await self.intelligent_router.route_task(
                    routing_context, available_agents, RoutingStrategy.ADAPTIVE
                )
                
                if selected_agent:
                    # Assign task with persona integration
                    success = await self._assign_task_to_agent_with_persona(
                        task_id, selected_agent, task, None, task.context or {}
                    )
                    
                    if success:
                        logger.info(
                            f"Queued task assigned successfully",
                            task_id=task_id,
                            agent_id=selected_agent,
                            queue_wait_time=time.time() - task_entry['queued_at']
                        )
                        return True
            
            return False
            
        except Exception as e:
            logger.error(
                f"Failed to assign queued task",
                task_id=task_entry.get('task_id', 'unknown'),
                error=str(e)
            )
            return False
    
    async def _requeue_failed_task(self, task_entry: Dict[str, Any]) -> None:
        """Re-queue a task that failed to be assigned."""
        try:
            task_entry['retry_count'] = task_entry.get('retry_count', 0) + 1
            
            if task_entry['retry_count'] <= self.error_thresholds['max_retry_attempts']:
                await self.enqueue_task(
                    task_entry['task_id'],
                    task_entry['priority'],
                    task_entry.get('workflow_id'),
                    task_entry['retry_count'],
                    task_entry.get('estimated_effort')
                )
            else:
                logger.error(
                    f"Task assignment failed permanently",
                    task_id=task_entry['task_id'],
                    retry_count=task_entry['retry_count']
                )
        except Exception as e:
            logger.error("Failed to re-queue task", error=str(e))
    
    async def _is_workflow_ready_for_tasks(self, workflow_id: str) -> bool:
        """Check if a workflow is ready to process queued tasks."""
        try:
            # Simple check - could be enhanced with dependency analysis
            async with get_session() as db_session:
                from ..models.workflow import Workflow
                workflow = await db_session.get(Workflow, workflow_id)
                
                if workflow:
                    return workflow.status in [WorkflowStatus.RUNNING, WorkflowStatus.READY]
                
                return False
                
        except Exception as e:
            logger.error(f"Error checking workflow readiness", workflow_id=workflow_id, error=str(e))
            return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current status of all task queues."""
        with self.queue_lock:
            return {
                'queue_processing_active': self.queue_processing_active,
                'high_priority_count': len(self.task_queues['high_priority']),
                'medium_priority_count': len(self.task_queues['medium_priority']),
                'low_priority_count': len(self.task_queues['low_priority']),
                'retry_queue_count': len(self.task_queues['retry_queue']),
                'workflow_queues': {
                    wf_id: len(queue) for wf_id, queue in self.task_queues['workflow_tasks'].items()
                },
                'total_queued_tasks': (
                    len(self.task_queues['high_priority']) +
                    len(self.task_queues['medium_priority']) +
                    len(self.task_queues['low_priority']) +
                    len(self.task_queues['retry_queue']) +
                    sum(len(q) for q in self.task_queues['workflow_tasks'].values())
                )
            }