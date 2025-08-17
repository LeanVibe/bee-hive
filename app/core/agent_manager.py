"""
Unified Agent Manager for LeanVibe Agent Hive 2.0

Consolidates 22 agent-related files into a single, comprehensive agent management system:
- Agent lifecycle management
- Agent spawning and registry
- Agent communication and messaging
- Agent persona system
- Agent load balancing
- Agent knowledge management
- Multi-agent coordination
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass
from enum import Enum

import structlog
from sqlalchemy import select, update, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import cast
from sqlalchemy.types import Float

from .unified_manager_base import UnifiedManagerBase, ManagerConfig, PluginInterface, PluginType
from .database import get_async_session
from .redis import get_redis, AgentMessageBroker
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.task import Task, TaskStatus, TaskPriority, TaskType
from ..models.persona import PersonaAssignmentModel

logger = structlog.get_logger()


class AgentCapabilityType(str, Enum):
    """Types of agent capabilities."""
    CODE_EXECUTION = "code_execution"
    FILE_OPERATIONS = "file_operations"
    WEB_BROWSING = "web_browsing"
    API_INTEGRATION = "api_integration"
    DATA_ANALYSIS = "data_analysis"
    CONTENT_GENERATION = "content_generation"
    COORDINATION = "coordination"
    MONITORING = "monitoring"


class PersonaType(str, Enum):
    """Agent persona types."""
    DEVELOPER = "developer"
    ARCHITECT = "architect"
    QA_ENGINEER = "qa_engineer"
    DEVOPS = "devops"
    PROJECT_MANAGER = "project_manager"
    DATA_SCIENTIST = "data_scientist"
    SECURITY_ANALYST = "security_analyst"
    META_AGENT = "meta_agent"


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
    PERSONA_ASSIGNED = "persona_assigned"
    CAPABILITY_UPDATED = "capability_updated"


@dataclass
class AgentSpec:
    """Specification for creating an agent."""
    name: str
    agent_type: AgentType = AgentType.CLAUDE
    role: Optional[str] = None
    capabilities: List[AgentCapabilityType] = None
    persona_type: Optional[PersonaType] = None
    system_prompt: Optional[str] = None
    config: Dict[str, Any] = None
    tmux_session: Optional[str] = None
    resource_limits: Optional[Dict[str, Any]] = None


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


@dataclass
class CoordinationRequest:
    """Request for multi-agent coordination."""
    requester_agent_id: uuid.UUID
    task_description: str
    required_capabilities: List[AgentCapabilityType]
    coordination_type: str = "collaborative"  # collaborative, sequential, competitive
    max_agents: int = 5
    deadline: Optional[datetime] = None


@dataclass
class CoordinationResult:
    """Result of coordination request."""
    success: bool
    coordination_id: uuid.UUID
    participating_agents: List[uuid.UUID]
    coordination_plan: Dict[str, Any]
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None


class AgentPersonaSystem:
    """Manages agent personas and intelligent task routing."""
    
    def __init__(self):
        self.persona_capabilities = {
            PersonaType.DEVELOPER: [
                AgentCapabilityType.CODE_EXECUTION,
                AgentCapabilityType.FILE_OPERATIONS,
                AgentCapabilityType.API_INTEGRATION
            ],
            PersonaType.ARCHITECT: [
                AgentCapabilityType.CODE_EXECUTION,
                AgentCapabilityType.DATA_ANALYSIS,
                AgentCapabilityType.COORDINATION
            ],
            PersonaType.QA_ENGINEER: [
                AgentCapabilityType.CODE_EXECUTION,
                AgentCapabilityType.FILE_OPERATIONS,
                AgentCapabilityType.MONITORING
            ],
            PersonaType.DEVOPS: [
                AgentCapabilityType.CODE_EXECUTION,
                AgentCapabilityType.MONITORING,
                AgentCapabilityType.API_INTEGRATION
            ],
            PersonaType.DATA_SCIENTIST: [
                AgentCapabilityType.DATA_ANALYSIS,
                AgentCapabilityType.CONTENT_GENERATION,
                AgentCapabilityType.FILE_OPERATIONS
            ],
            PersonaType.SECURITY_ANALYST: [
                AgentCapabilityType.CODE_EXECUTION,
                AgentCapabilityType.MONITORING,
                AgentCapabilityType.DATA_ANALYSIS
            ],
            PersonaType.META_AGENT: [
                AgentCapabilityType.COORDINATION,
                AgentCapabilityType.MONITORING,
                AgentCapabilityType.API_INTEGRATION
            ]
        }
    
    async def assign_optimal_persona(
        self, 
        agent_id: uuid.UUID, 
        context: Dict[str, Any]
    ) -> Optional[PersonaType]:
        """Assign the optimal persona based on context."""
        role = context.get("role", "").lower()
        capabilities = context.get("capabilities", [])
        
        # Direct role mapping
        role_mapping = {
            "developer": PersonaType.DEVELOPER,
            "architect": PersonaType.ARCHITECT,
            "qa": PersonaType.QA_ENGINEER,
            "devops": PersonaType.DEVOPS,
            "data_scientist": PersonaType.DATA_SCIENTIST,
            "security": PersonaType.SECURITY_ANALYST,
            "meta": PersonaType.META_AGENT
        }
        
        for role_key, persona in role_mapping.items():
            if role_key in role:
                return persona
        
        # Capability-based assignment
        capability_scores = {}
        for persona, persona_caps in self.persona_capabilities.items():
            score = len(set(capabilities) & set(persona_caps))
            capability_scores[persona] = score
        
        if capability_scores:
            best_persona = max(capability_scores, key=capability_scores.get)
            if capability_scores[best_persona] > 0:
                return best_persona
        
        # Default to developer
        return PersonaType.DEVELOPER


class AgentLoadBalancer:
    """Intelligent load balancing for agent assignment."""
    
    def __init__(self):
        self.agent_loads: Dict[uuid.UUID, float] = {}
        self.agent_specializations: Dict[uuid.UUID, List[AgentCapabilityType]] = {}
    
    def update_agent_load(self, agent_id: uuid.UUID, load: float) -> None:
        """Update agent load metric."""
        self.agent_loads[agent_id] = load
    
    def get_optimal_agent(
        self, 
        available_agents: List[Agent], 
        required_capabilities: List[AgentCapabilityType]
    ) -> Optional[Agent]:
        """Get the optimal agent for task assignment."""
        if not available_agents:
            return None
        
        scored_agents = []
        
        for agent in available_agents:
            # Capability match score
            agent_caps = agent.capabilities or []
            cap_names = [cap.get("name") for cap in agent_caps if isinstance(cap, dict)]
            capability_score = len(set(required_capabilities) & set(cap_names)) / max(len(required_capabilities), 1)
            
            # Load score (inverse of current load)
            current_load = self.agent_loads.get(agent.id, 0.0)
            load_score = 1.0 - min(current_load, 1.0)
            
            # Combined score
            total_score = (capability_score * 0.7) + (load_score * 0.3)
            scored_agents.append((agent, total_score))
        
        # Sort by score and return best
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        return scored_agents[0][0]


class AgentKnowledgeManager:
    """Manages cross-agent knowledge sharing and learning."""
    
    def __init__(self):
        self.knowledge_base: Dict[str, Any] = {}
        self.agent_experiences: Dict[uuid.UUID, List[Dict[str, Any]]] = {}
    
    async def record_agent_experience(
        self, 
        agent_id: uuid.UUID, 
        task_type: str, 
        outcome: Dict[str, Any]
    ) -> None:
        """Record an agent's task experience for future learning."""
        if agent_id not in self.agent_experiences:
            self.agent_experiences[agent_id] = []
        
        experience = {
            "timestamp": datetime.utcnow(),
            "task_type": task_type,
            "outcome": outcome,
            "success": outcome.get("success", False)
        }
        
        self.agent_experiences[agent_id].append(experience)
        
        # Keep only recent experiences (last 100)
        if len(self.agent_experiences[agent_id]) > 100:
            self.agent_experiences[agent_id] = self.agent_experiences[agent_id][-100:]
    
    async def get_relevant_knowledge(
        self, 
        agent_id: uuid.UUID, 
        task_type: str
    ) -> Dict[str, Any]:
        """Get relevant knowledge for an agent based on task type."""
        relevant_knowledge = {
            "similar_experiences": [],
            "best_practices": [],
            "common_pitfalls": []
        }
        
        # Find similar experiences from other agents
        for other_agent_id, experiences in self.agent_experiences.items():
            if other_agent_id == agent_id:
                continue
                
            for exp in experiences:
                if exp["task_type"] == task_type and exp["success"]:
                    relevant_knowledge["similar_experiences"].append({
                        "agent_id": str(other_agent_id),
                        "outcome": exp["outcome"],
                        "timestamp": exp["timestamp"]
                    })
        
        return relevant_knowledge


class AgentManager(UnifiedManagerBase):
    """
    Unified Agent Manager consolidating all agent-related functionality.
    
    Replaces 22 separate files:
    - agent_lifecycle_manager.py
    - agent_spawner.py
    - agent_registry.py
    - agent_communication_service.py
    - agent_messaging_service.py
    - agent_persona_system.py
    - agent_load_balancer.py
    - agent_knowledge_manager.py
    - cross_agent_knowledge_manager.py
    - enhanced_multi_agent_coordination.py
    - multi_agent_commands.py
    - agent_identity_service.py
    - agent_workflow_tracker.py
    - ai_architect_agent.py
    - ai_task_worker.py
    - code_intelligence_agent.py
    - enhanced_agent_implementations.py
    - real_agent_implementations.py
    - real_multiagent_workflow.py
    - self_optimization_agent.py
    - cli_agent_orchestrator.py
    - agent_lifecycle_hooks.py
    """
    
    def __init__(self, config: ManagerConfig, dependencies: Optional[Dict[str, Any]] = None):
        super().__init__(config, dependencies)
        
        # Core components
        self.redis = None
        self.message_broker = None
        self.persona_system = AgentPersonaSystem()
        self.load_balancer = AgentLoadBalancer()
        self.knowledge_manager = AgentKnowledgeManager()
        
        # State tracking
        self.active_agents: Set[uuid.UUID] = set()
        self.task_assignments: Dict[uuid.UUID, uuid.UUID] = {}  # task_id -> agent_id
        self.coordination_sessions: Dict[uuid.UUID, CoordinationRequest] = {}
        self.assignment_times: Dict[str, float] = {}
        
        # Performance metrics
        self.total_agents_spawned = 0
        self.total_tasks_assigned = 0
        self.total_coordinations = 0
    
    async def _initialize_manager(self) -> bool:
        """Initialize the agent manager."""
        try:
            # Initialize Redis connection
            self.redis = get_redis()
            self.message_broker = AgentMessageBroker(self.redis)
            
            # Load existing active agents from database
            await self._load_active_agents()
            
            logger.info(
                "Agent Manager initialized",
                active_agents=len(self.active_agents),
                plugins=len(self.plugins)
            )
            return True
            
        except Exception as e:
            logger.error("Failed to initialize Agent Manager", error=str(e))
            return False
    
    async def _shutdown_manager(self) -> None:
        """Shutdown the agent manager."""
        try:
            # Gracefully shutdown all active agents
            for agent_id in list(self.active_agents):
                await self.deregister_agent(agent_id)
            
            logger.info("Agent Manager shutdown completed")
            
        except Exception as e:
            logger.error("Error during Agent Manager shutdown", error=str(e))
    
    async def _get_manager_health(self) -> Dict[str, Any]:
        """Get agent manager health information."""
        return {
            "active_agents": len(self.active_agents),
            "total_agents_spawned": self.total_agents_spawned,
            "total_tasks_assigned": self.total_tasks_assigned,
            "total_coordinations": self.total_coordinations,
            "coordination_sessions": len(self.coordination_sessions),
            "message_broker_connected": self.message_broker is not None
        }
    
    async def _load_plugins(self) -> None:
        """Load agent manager plugins."""
        # Load performance monitoring plugin
        if "performance" in self.config.plugin_config:
            # Plugin loading will be implemented when needed
            pass
    
    async def _load_active_agents(self) -> None:
        """Load active agents from database."""
        try:
            async with get_async_session() as db:
                result = await db.execute(
                    select(Agent.id).where(Agent.status == AgentStatus.active)
                )
                agent_ids = result.scalars().all()
                self.active_agents = set(agent_ids)
                
        except Exception as e:
            logger.error("Failed to load active agents", error=str(e))
    
    # === CORE AGENT LIFECYCLE OPERATIONS ===
    
    async def spawn_agent(self, spec: AgentSpec) -> AgentRegistrationResult:
        """
        Spawn a new agent with comprehensive lifecycle management.
        
        Consolidates functionality from:
        - agent_spawner.py
        - agent_registry.py
        - agent_lifecycle_manager.py
        """
        return await self.execute_with_monitoring(
            "spawn_agent",
            self._spawn_agent_impl,
            spec
        )
    
    async def _spawn_agent_impl(self, spec: AgentSpec) -> AgentRegistrationResult:
        """Internal implementation of agent spawning."""
        start_time = datetime.utcnow()
        
        try:
            async with get_async_session() as db:
                # Create agent in database
                agent = Agent(
                    name=spec.name,
                    type=spec.agent_type,
                    role=spec.role,
                    capabilities=[{"name": cap.value} for cap in (spec.capabilities or [])],
                    system_prompt=spec.system_prompt,
                    config=spec.config or {},
                    tmux_session=spec.tmux_session,
                    status=AgentStatus.INITIALIZING
                )
                
                db.add(agent)
                await db.commit()
                await db.refresh(agent)
                
                # Assign persona
                persona_assigned = None
                if spec.persona_type:
                    persona_assigned = spec.persona_type.value
                else:
                    optimal_persona = await self.persona_system.assign_optimal_persona(
                        agent.id,
                        {"role": spec.role, "capabilities": spec.capabilities}
                    )
                    if optimal_persona:
                        persona_assigned = optimal_persona.value
                
                # Update agent status to active
                agent.status = AgentStatus.active
                agent.last_heartbeat = datetime.utcnow()
                agent.last_active = datetime.utcnow()
                await db.commit()
                
                # Add to active agents
                self.active_agents.add(agent.id)
                self.total_agents_spawned += 1
                
                # Initialize load balancer entry
                self.load_balancer.update_agent_load(agent.id, 0.0)
                
                # Publish lifecycle event
                await self._publish_lifecycle_event(
                    LifecycleEventType.AGENT_REGISTERED,
                    agent.id,
                    {
                        "name": spec.name,
                        "type": spec.agent_type.value,
                        "role": spec.role,
                        "capabilities": [cap.value for cap in (spec.capabilities or [])],
                        "persona_assigned": persona_assigned,
                        "registration_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
                    }
                )
                
                logger.info(
                    "✅ Agent spawned successfully",
                    agent_id=str(agent.id),
                    name=spec.name,
                    persona=persona_assigned,
                    capabilities=len(spec.capabilities or [])
                )
                
                return AgentRegistrationResult(
                    success=True,
                    agent_id=agent.id,
                    capabilities_assigned=[cap.value for cap in (spec.capabilities or [])],
                    persona_assigned=persona_assigned
                )
                
        except Exception as e:
            logger.error("❌ Agent spawn failed", name=spec.name, error=str(e))
            return AgentRegistrationResult(
                success=False,
                error_message=str(e)
            )
    
    async def deregister_agent(self, agent_id: uuid.UUID) -> bool:
        """Deregister an agent from the system."""
        return await self.execute_with_monitoring(
            "deregister_agent",
            self._deregister_agent_impl,
            agent_id
        )
    
    async def _deregister_agent_impl(self, agent_id: uuid.UUID) -> bool:
        """Internal implementation of agent deregistration."""
        try:
            async with get_async_session() as db:
                # Get agent
                result = await db.execute(select(Agent).where(Agent.id == agent_id))
                agent = result.scalar_one_or_none()
                
                if not agent:
                    logger.warning("Agent not found for deregistration", agent_id=str(agent_id))
                    return False
                
                # Cancel assigned tasks
                await self._cancel_agent_tasks(db, agent_id)
                
                # Update agent status
                agent.status = AgentStatus.inactive
                await db.commit()
                
                # Remove from tracking
                self.active_agents.discard(agent_id)
                self.load_balancer.agent_loads.pop(agent_id, None)
                
                # Publish lifecycle event
                await self._publish_lifecycle_event(
                    LifecycleEventType.AGENT_DEREGISTERED,
                    agent_id,
                    {"agent_name": agent.name}
                )
                
                logger.info("✅ Agent deregistered", agent_id=str(agent_id), name=agent.name)
                return True
                
        except Exception as e:
            logger.error("❌ Agent deregistration failed", agent_id=str(agent_id), error=str(e))
            return False
    
    # === TASK ASSIGNMENT AND COORDINATION ===
    
    async def assign_task(
        self,
        task_id: uuid.UUID,
        preferred_agent_id: Optional[uuid.UUID] = None,
        required_capabilities: List[AgentCapabilityType] = None
    ) -> TaskAssignmentResult:
        """
        Assign a task to the most suitable agent using intelligent routing.
        
        Consolidates functionality from:
        - intelligent_task_router.py
        - enhanced_intelligent_task_router.py
        - agent_load_balancer.py
        """
        return await self.execute_with_monitoring(
            "assign_task",
            self._assign_task_impl,
            task_id,
            preferred_agent_id,
            required_capabilities or []
        )
    
    async def _assign_task_impl(
        self,
        task_id: uuid.UUID,
        preferred_agent_id: Optional[uuid.UUID],
        required_capabilities: List[AgentCapabilityType]
    ) -> TaskAssignmentResult:
        """Internal implementation of task assignment."""
        start_time = datetime.utcnow()
        
        try:
            async with get_async_session() as db:
                # Get task details
                task_result = await db.execute(select(Task).where(Task.id == task_id))
                task = task_result.scalar_one_or_none()
                
                if not task or task.status != TaskStatus.PENDING:
                    return TaskAssignmentResult(
                        success=False,
                        error_message="Task not found or not in PENDING status"
                    )
                
                # Find suitable agent
                suitable_agent = await self._find_suitable_agent(
                    db, task, preferred_agent_id, required_capabilities
                )
                
                if not suitable_agent:
                    return TaskAssignmentResult(
                        success=False,
                        error_message="No suitable agent available"
                    )
                
                # Assign task
                task.assign_to_agent(suitable_agent.id)
                suitable_agent.status = AgentStatus.busy
                await db.commit()
                
                # Update tracking
                self.task_assignments[task_id] = suitable_agent.id
                self.total_tasks_assigned += 1
                assignment_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.assignment_times[str(task_id)] = assignment_time_ms
                
                # Update load balancer
                self.load_balancer.update_agent_load(suitable_agent.id, 1.0)
                
                # Send task to agent via message broker
                await self.message_broker.send_message(
                    from_agent="orchestrator",
                    to_agent=str(suitable_agent.id),
                    message_type="task_assignment",
                    payload={
                        "task_id": str(task_id),
                        "task_title": task.title,
                        "task_type": task.task_type.value if task.task_type else "general",
                        "required_capabilities": [cap.value for cap in required_capabilities],
                        "context": task.context or {}
                    }
                )
                
                # Get relevant knowledge for the agent
                knowledge = await self.knowledge_manager.get_relevant_knowledge(
                    suitable_agent.id,
                    task.task_type.value if task.task_type else "general"
                )
                
                if knowledge["similar_experiences"]:
                    await self.message_broker.send_message(
                        from_agent="knowledge_manager",
                        to_agent=str(suitable_agent.id),
                        message_type="knowledge_share",
                        payload=knowledge
                    )
                
                # Publish lifecycle event
                await self._publish_lifecycle_event(
                    LifecycleEventType.TASK_ASSIGNED,
                    suitable_agent.id,
                    {
                        "task_id": str(task_id),
                        "assignment_time_ms": assignment_time_ms
                    }
                )
                
                logger.info(
                    "✅ Task assigned successfully",
                    task_id=str(task_id),
                    agent_id=str(suitable_agent.id),
                    assignment_time_ms=assignment_time_ms
                )
                
                return TaskAssignmentResult(
                    success=True,
                    agent_id=suitable_agent.id,
                    task_id=task_id,
                    assignment_time=datetime.utcnow()
                )
                
        except Exception as e:
            logger.error("❌ Task assignment failed", task_id=str(task_id), error=str(e))
            return TaskAssignmentResult(success=False, error_message=str(e))
    
    async def coordinate_agents(self, request: CoordinationRequest) -> CoordinationResult:
        """
        Coordinate multiple agents for complex tasks.
        
        Consolidates functionality from:
        - enhanced_multi_agent_coordination.py
        - multi_agent_commands.py
        - real_multiagent_workflow.py
        """
        return await self.execute_with_monitoring(
            "coordinate_agents",
            self._coordinate_agents_impl,
            request
        )
    
    async def _coordinate_agents_impl(self, request: CoordinationRequest) -> CoordinationResult:
        """Internal implementation of agent coordination."""
        coordination_id = uuid.uuid4()
        
        try:
            async with get_async_session() as db:
                # Find suitable agents for coordination
                available_agents = await self._get_available_agents_for_coordination(
                    db, request.required_capabilities, request.max_agents
                )
                
                if len(available_agents) < 2:
                    return CoordinationResult(
                        success=False,
                        coordination_id=coordination_id,
                        participating_agents=[],
                        coordination_plan={},
                        error_message="Insufficient agents available for coordination"
                    )
                
                # Create coordination plan
                coordination_plan = {
                    "type": request.coordination_type,
                    "task_description": request.task_description,
                    "agent_roles": {},
                    "workflow": [],
                    "communication_channels": []
                }
                
                participating_agent_ids = []
                
                # Assign roles based on capabilities and personas
                for i, agent in enumerate(available_agents):
                    agent_id = agent.id
                    participating_agent_ids.append(agent_id)
                    
                    # Determine role based on agent capabilities
                    role = self._determine_coordination_role(agent, request.required_capabilities)
                    coordination_plan["agent_roles"][str(agent_id)] = role
                    
                    # Update agent status
                    agent.status = AgentStatus.busy
                
                await db.commit()
                
                # Store coordination session
                self.coordination_sessions[coordination_id] = request
                self.total_coordinations += 1
                
                # Set up communication channels
                for agent_id in participating_agent_ids:
                    await self.message_broker.send_message(
                        from_agent="coordinator",
                        to_agent=str(agent_id),
                        message_type="coordination_invite",
                        payload={
                            "coordination_id": str(coordination_id),
                            "task_description": request.task_description,
                            "role": coordination_plan["agent_roles"][str(agent_id)],
                            "other_agents": [str(aid) for aid in participating_agent_ids if aid != agent_id]
                        }
                    )
                
                logger.info(
                    "✅ Agent coordination established",
                    coordination_id=str(coordination_id),
                    participating_agents=len(participating_agent_ids),
                    coordination_type=request.coordination_type
                )
                
                return CoordinationResult(
                    success=True,
                    coordination_id=coordination_id,
                    participating_agents=participating_agent_ids,
                    coordination_plan=coordination_plan
                )
                
        except Exception as e:
            logger.error("❌ Agent coordination failed", error=str(e))
            return CoordinationResult(
                success=False,
                coordination_id=coordination_id,
                participating_agents=[],
                coordination_plan={},
                error_message=str(e)
            )
    
    # === AGENT MONITORING AND HEALTH ===
    
    async def process_agent_heartbeat(
        self, 
        agent_id: uuid.UUID, 
        status_data: Dict[str, Any]
    ) -> bool:
        """Process agent heartbeat and update status."""
        return await self.execute_with_monitoring(
            "process_heartbeat",
            self._process_heartbeat_impl,
            agent_id,
            status_data
        )
    
    async def _process_heartbeat_impl(
        self, 
        agent_id: uuid.UUID, 
        status_data: Dict[str, Any]
    ) -> bool:
        """Internal implementation of heartbeat processing."""
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
                
                # Update load balancer
                context_usage = status_data.get("context_usage", 0.0)
                self.load_balancer.update_agent_load(agent_id, context_usage)
                
                # Publish heartbeat event
                await self._publish_lifecycle_event(
                    LifecycleEventType.AGENT_HEARTBEAT,
                    agent_id,
                    status_data
                )
                
                return True
                
        except Exception as e:
            logger.error("❌ Heartbeat processing failed", agent_id=str(agent_id), error=str(e))
            return False
    
    async def complete_task(
        self,
        task_id: uuid.UUID,
        agent_id: uuid.UUID,
        result: Dict[str, Any],
        success: bool = True
    ) -> bool:
        """Mark a task as completed and update agent status."""
        return await self.execute_with_monitoring(
            "complete_task",
            self._complete_task_impl,
            task_id,
            agent_id,
            result,
            success
        )
    
    async def _complete_task_impl(
        self,
        task_id: uuid.UUID,
        agent_id: uuid.UUID,
        result: Dict[str, Any],
        success: bool
    ) -> bool:
        """Internal implementation of task completion."""
        try:
            async with get_async_session() as db:
                # Get task and agent
                task_result = await db.execute(select(Task).where(Task.id == task_id))
                task = task_result.scalar_one_or_none()
                
                agent_result = await db.execute(select(Agent).where(Agent.id == agent_id))
                agent = agent_result.scalar_one_or_none()
                
                if not task or not agent:
                    return False
                
                # Update task status
                if success:
                    task.complete_successfully(result)
                    agent.total_tasks_completed = str(int(agent.total_tasks_completed or 0) + 1)
                    event_type = LifecycleEventType.TASK_COMPLETED
                else:
                    task.fail_with_error(result.get("error", "Task failed"))
                    agent.total_tasks_failed = str(int(agent.total_tasks_failed or 0) + 1)
                    event_type = LifecycleEventType.TASK_FAILED
                
                # Update agent status
                agent.status = AgentStatus.active
                agent.last_active = datetime.utcnow()
                await db.commit()
                
                # Update tracking
                self.task_assignments.pop(task_id, None)
                self.load_balancer.update_agent_load(agent_id, 0.0)
                
                # Record experience for knowledge management
                await self.knowledge_manager.record_agent_experience(
                    agent_id,
                    task.task_type.value if task.task_type else "general",
                    {"success": success, "result": result}
                )
                
                # Publish lifecycle event
                await self._publish_lifecycle_event(event_type, agent_id, {"task_id": str(task_id)})
                
                return True
                
        except Exception as e:
            logger.error("❌ Task completion failed", task_id=str(task_id), error=str(e))
            return False
    
    # === UTILITY METHODS ===
    
    async def _find_suitable_agent(
        self,
        db: AsyncSession,
        task: Task,
        preferred_agent_id: Optional[uuid.UUID],
        required_capabilities: List[AgentCapabilityType]
    ) -> Optional[Agent]:
        """Find the most suitable agent for a task."""
        # If preferred agent is specified and available, use it
        if preferred_agent_id and preferred_agent_id in self.active_agents:
            result = await db.execute(
                select(Agent).where(
                    and_(
                        Agent.id == preferred_agent_id,
                        Agent.status == AgentStatus.active
                    )
                )
            )
            preferred_agent = result.scalar_one_or_none()
            if preferred_agent:
                return preferred_agent
        
        # Get available agents
        result = await db.execute(
            select(Agent).where(
                and_(
                    Agent.status == AgentStatus.active,
                    Agent.id.in_(self.active_agents)
                )
            )
        )
        available_agents = result.scalars().all()
        
        if not available_agents:
            return None
        
        # Use load balancer to find optimal agent
        return self.load_balancer.get_optimal_agent(available_agents, required_capabilities)
    
    async def _get_available_agents_for_coordination(
        self,
        db: AsyncSession,
        required_capabilities: List[AgentCapabilityType],
        max_agents: int
    ) -> List[Agent]:
        """Get available agents for coordination."""
        result = await db.execute(
            select(Agent).where(
                and_(
                    Agent.status == AgentStatus.active,
                    Agent.id.in_(self.active_agents)
                )
            ).limit(max_agents)
        )
        return result.scalars().all()
    
    def _determine_coordination_role(
        self, 
        agent: Agent, 
        required_capabilities: List[AgentCapabilityType]
    ) -> str:
        """Determine an agent's role in coordination based on capabilities."""
        agent_caps = [cap.get("name") for cap in (agent.capabilities or []) if isinstance(cap, dict)]
        
        if AgentCapabilityType.COORDINATION.value in agent_caps:
            return "coordinator"
        elif AgentCapabilityType.CODE_EXECUTION.value in agent_caps:
            return "executor"
        elif AgentCapabilityType.MONITORING.value in agent_caps:
            return "monitor"
        else:
            return "participant"
    
    async def _cancel_agent_tasks(self, db: AsyncSession, agent_id: uuid.UUID) -> None:
        """Cancel all tasks assigned to an agent."""
        result = await db.execute(
            select(Task).where(
                and_(
                    Task.assigned_agent_id == agent_id,
                    Task.status.in_([TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS])
                )
            )
        )
        tasks = result.scalars().all()
        
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
            if not self.redis:
                return
                
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
            
        except Exception as e:
            logger.error("Failed to publish lifecycle event", event_type=event_type.value, error=str(e))
    
    # === PUBLIC API METHODS ===
    
    async def get_agent_status(self, agent_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get current agent status and metrics."""
        try:
            async with get_async_session() as db:
                result = await db.execute(select(Agent).where(Agent.id == agent_id))
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
                            "status": task.status.value
                        }
                
                return {
                    "agent_id": str(agent.id),
                    "name": agent.name,
                    "status": agent.status.value,
                    "role": agent.role,
                    "capabilities": agent.capabilities,
                    "current_task": current_task,
                    "load": self.load_balancer.agent_loads.get(agent_id, 0.0),
                    "total_tasks_completed": int(agent.total_tasks_completed or 0),
                    "last_heartbeat": agent.last_heartbeat.isoformat() if agent.last_heartbeat else None
                }
                
        except Exception as e:
            logger.error("Failed to get agent status", agent_id=str(agent_id), error=str(e))
            return None
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide agent metrics."""
        try:
            async with get_async_session() as db:
                # Get agent counts by status
                agent_counts = await db.execute(
                    select(Agent.status, func.count(Agent.id)).group_by(Agent.status)
                )
                
                status_counts = {}
                for status, count in agent_counts:
                    status_counts[status.value] = count
                
                return {
                    "active_agents": len(self.active_agents),
                    "agent_status_counts": status_counts,
                    "total_agents_spawned": self.total_agents_spawned,
                    "total_tasks_assigned": self.total_tasks_assigned,
                    "total_coordinations": self.total_coordinations,
                    "active_task_assignments": len(self.task_assignments),
                    "coordination_sessions": len(self.coordination_sessions),
                    "average_assignment_time_ms": sum(self.assignment_times.values()) / max(len(self.assignment_times), 1),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            logger.error("Failed to get system metrics", error=str(e))
            return {"error": str(e)}


# Factory function for creating agent manager
def create_agent_manager(
    redis_client=None,
    **config_overrides
) -> AgentManager:
    """Create and initialize an agent manager."""
    config = create_manager_config("AgentManager", **config_overrides)
    dependencies = {"redis": redis_client} if redis_client else {}
    return AgentManager(config, dependencies)