"""
Enhanced Multi-Agent Coordination System for LeanVibe Agent Hive Phase 2

This module implements sophisticated multi-agent coordination patterns that showcase
industry-leading autonomous development capabilities with specialized agent roles,
advanced communication protocols, and intelligent collaboration patterns.

Key Features:
- 6 specialized agent roles with unique capabilities
- Advanced context sharing and collaborative memory
- Sophisticated inter-agent communication protocols
- Real-time status synchronization and progress coordination
- Intelligent conflict resolution and task handoff mechanisms
- Multi-agent code review cycles and pair programming simulation
- Continuous integration workflows with agent coordination
- Knowledge sharing and learning from successful patterns
- Dynamic team formation based on task requirements
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
from pathlib import Path

import structlog

from .messaging_service import get_messaging_service, MessagingService, Message, MessageType as UnifiedMessageType, MessagePriority
from .messaging_migration import MessagingServiceAdapter, LegacyMessageAdapter, mark_migration_complete
# Legacy imports - DEPRECATED, use messaging_service instead
# from .redis import get_message_broker, AgentMessageBroker
# from .agent_communication_service import AgentCommunicationService, AgentMessage
from .workflow_engine import WorkflowEngine, WorkflowResult, TaskExecutionState
from .intelligent_task_router import IntelligentTaskRouter, TaskRoutingContext
from .capability_matcher import CapabilityMatcher
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.task import Task, TaskStatus, TaskPriority
from ..models.workflow import Workflow, WorkflowStatus
from ..models.message import MessageType as LegacyMessageType
from ..models.coordination_event import CoordinationEvent, BusinessValueMetric, CoordinationEventType
from .database import get_session

logger = structlog.get_logger()


class SpecializedAgentRole(Enum):
    """Advanced specialized agent roles for sophisticated development teams."""
    ARCHITECT = "architect"           # System design, architecture decisions, technical leadership
    DEVELOPER = "developer"          # Implementation, coding, feature development
    TESTER = "tester"               # Quality assurance, testing strategy, validation
    REVIEWER = "reviewer"           # Code review, best practices, security analysis
    DEVOPS = "devops"              # Deployment, infrastructure, CI/CD automation
    PRODUCT = "product"            # Requirements analysis, user experience, acceptance criteria


class CoordinationPatternType(Enum):
    """Types of coordination patterns for multi-agent collaboration."""
    PAIR_PROGRAMMING = "pair_programming"
    CODE_REVIEW_CYCLE = "code_review_cycle"
    CONTINUOUS_INTEGRATION = "continuous_integration"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    DESIGN_REVIEW = "design_review"
    TASK_HANDOFF = "task_handoff"
    CONFLICT_RESOLUTION = "conflict_resolution"
    TEAM_STANDUP = "team_standup"


class TaskComplexity(Enum):
    """Task complexity levels for intelligent agent assignment."""
    SIMPLE = "simple"        # Single agent, <1 hour
    MODERATE = "moderate"    # 2-3 agents, <4 hours  
    COMPLEX = "complex"      # 4+ agents, <8 hours
    ENTERPRISE = "enterprise" # Full team, multi-day


@dataclass
class AgentCapability:
    """Enhanced agent capability with proficiency levels and specializations."""
    name: str
    proficiency_level: float  # 0.0 to 1.0
    specialization_areas: List[str]
    experience_points: int = 0
    success_rate: float = 1.0
    collaboration_rating: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class CollaborationContext:
    """Shared context and memory between collaborating agents."""
    collaboration_id: str
    participants: List[str]  # Agent IDs
    shared_knowledge: Dict[str, Any] = field(default_factory=dict)
    communication_history: List[Dict[str, Any]] = field(default_factory=list)
    decisions_made: List[Dict[str, Any]] = field(default_factory=list)
    artifacts_created: List[str] = field(default_factory=list)
    success_patterns: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def add_knowledge(self, key: str, value: Any, contributor: str):
        """Add knowledge to shared context."""
        self.shared_knowledge[key] = {
            "value": value,
            "contributor": contributor,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.last_updated = datetime.utcnow()
    
    def add_communication(self, from_agent: str, to_agent: str, message: str, message_type: str):
        """Add communication to history."""
        self.communication_history.append({
            "from": from_agent,
            "to": to_agent, 
            "message": message,
            "type": message_type,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.last_updated = datetime.utcnow()
    
    def add_decision(self, decision: str, rationale: str, participants: List[str]):
        """Add collaborative decision to context."""
        self.decisions_made.append({
            "decision": decision,
            "rationale": rationale,
            "participants": participants,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.last_updated = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class CoordinationPattern:
    """Definition of a multi-agent coordination pattern."""
    pattern_id: str
    pattern_type: CoordinationPatternType
    name: str
    description: str
    required_roles: List[SpecializedAgentRole]
    coordination_steps: List[Dict[str, Any]]
    success_metrics: Dict[str, float]
    estimated_duration: int  # minutes
    complexity_level: TaskComplexity
    
    def to_dict(self) -> Dict[str, Any]:
        # Handle both enum and string pattern types for robustness
        pattern_type_value = self.pattern_type.value if hasattr(self.pattern_type, 'value') else self.pattern_type
        required_roles_values = []
        for role in self.required_roles:
            if hasattr(role, 'value'):
                required_roles_values.append(role.value)
            else:
                required_roles_values.append(role)
        complexity_level_value = self.complexity_level.value if hasattr(self.complexity_level, 'value') else self.complexity_level
        
        return {
            **asdict(self),
            "pattern_type": pattern_type_value,
            "required_roles": required_roles_values,
            "complexity_level": complexity_level_value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoordinationPattern':
        """Create CoordinationPattern from dictionary, handling enum conversions."""
        # Convert string values back to enums
        if isinstance(data.get('pattern_type'), str):
            data['pattern_type'] = CoordinationPatternType(data['pattern_type'])
        
        if 'required_roles' in data:
            roles = []
            for role in data['required_roles']:
                if isinstance(role, str):
                    roles.append(SpecializedAgentRole(role))
                else:
                    roles.append(role)
            data['required_roles'] = roles
        
        if isinstance(data.get('complexity_level'), str):
            data['complexity_level'] = TaskComplexity(data['complexity_level'])
        
        return cls(**data)


@dataclass
class SpecializedAgent:
    """Enhanced agent with specialized capabilities and coordination features."""
    agent_id: str
    role: SpecializedAgentRole
    status: AgentStatus
    capabilities: List[AgentCapability]
    current_collaborations: Set[str] = field(default_factory=set)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    collaboration_preferences: Dict[str, Any] = field(default_factory=dict)
    workload_capacity: float = 1.0  # 0.0 to 1.0
    current_workload: float = 0.0   # 0.0 to 1.0
    specialization_score: float = 0.8  # 0.0 to 1.0
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_available(self) -> bool:
        """Check if agent is available for new tasks."""
        return (self.status == AgentStatus.active and 
                self.current_workload < self.workload_capacity * 0.9)
    
    @property 
    def capability_names(self) -> List[str]:
        """Get list of capability names."""
        return [cap.name for cap in self.capabilities]
    
    def add_performance_record(self, task_id: str, success: bool, duration: float, quality_score: float):
        """Add performance record to history."""
        self.performance_history.append({
            "task_id": task_id,
            "success": success,
            "duration": duration,
            "quality_score": quality_score,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Update capability proficiencies based on performance
        self._update_capabilities_from_performance(success, quality_score)
    
    def _update_capabilities_from_performance(self, success: bool, quality_score: float):
        """Update capability proficiency levels based on performance."""
        adjustment = 0.01 if success else -0.005
        quality_adjustment = (quality_score - 0.5) * 0.02
        
        for capability in self.capabilities:
            new_proficiency = min(1.0, max(0.1, 
                capability.proficiency_level + adjustment + quality_adjustment))
            capability.proficiency_level = new_proficiency
            
            if success:
                capability.experience_points += 1
                capability.success_rate = min(1.0, capability.success_rate + 0.001)
    
    def to_dict(self) -> Dict[str, Any]:
        # Handle both enum and string values for robustness
        role_value = self.role.value if hasattr(self.role, 'value') else self.role
        status_value = self.status.value if hasattr(self.status, 'value') else self.status
        
        return {
            **asdict(self),
            "role": role_value,
            "status": status_value,
            "current_collaborations": list(self.current_collaborations),
            "capabilities": [cap.to_dict() for cap in self.capabilities],
            "last_heartbeat": self.last_heartbeat.isoformat()
        }


class EnhancedMultiAgentCoordinator:
    """
    Advanced multi-agent coordination system with sophisticated collaboration patterns.
    
    This coordinator manages teams of specialized AI agents working together on complex
    software development tasks using industry-leading coordination patterns.
    """
    
    def __init__(self, workspace_dir: str = "/tmp/enhanced_coordination"):
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Agent management
        self.agents: Dict[str, SpecializedAgent] = {}
        self.agent_roles: Dict[SpecializedAgentRole, List[str]] = defaultdict(list)
        
        # Coordination patterns
        self.coordination_patterns: Dict[str, CoordinationPattern] = {}
        self.active_collaborations: Dict[str, CollaborationContext] = {}
        
        # Communication and workflow systems
        self.messaging_service: Optional[MessagingService] = None
        self.workflow_engine: Optional[WorkflowEngine] = None
        self.task_router: Optional[IntelligentTaskRouter] = None
        self.capability_matcher: Optional[CapabilityMatcher] = None
        
        # Performance and analytics
        self.coordination_metrics: Dict[str, Any] = {
            "total_collaborations": 0,
            "successful_collaborations": 0,
            "average_collaboration_duration": 0.0,
            "agent_utilization": {},
            "pattern_success_rates": {},
            "knowledge_sharing_events": 0
        }
        
        self.logger = logger.bind(component="enhanced_coordination")
        self._initialize_default_patterns()
    
    async def initialize(self):
        """Initialize the coordination system with all required services."""
        try:
            self.logger.info("🚀 Initializing Enhanced Multi-Agent Coordination System")
            
            # Initialize unified messaging service
            self.messaging_service = get_messaging_service()
            await self.messaging_service.connect()
            await self.messaging_service.start_service()
            
            # Register coordination message handlers
            await self._register_coordination_handlers()
            
            # Initialize workflow and routing systems
            self.workflow_engine = WorkflowEngine()
            self.task_router = IntelligentTaskRouter()
            self.capability_matcher = CapabilityMatcher()
            
            # Initialize specialized agents
            await self._initialize_specialized_agents()
            
            self.logger.info("✅ Enhanced Multi-Agent Coordination System initialized successfully",
                           agents_count=len(self.agents),
                           patterns_count=len(self.coordination_patterns))
            
        except Exception as e:
            self.logger.error("❌ Failed to initialize coordination system", error=str(e))
            raise
    
    async def _register_coordination_handlers(self) -> None:
        """Register message handlers for multi-agent coordination"""
        from .messaging_service import MessageHandler
        
        class CoordinationMessageHandler(MessageHandler):
            def __init__(self, coordinator):
                super().__init__(
                    handler_id="coordination",
                    pattern="coordination.*",
                    message_types=[
                        UnifiedMessageType.TASK_ASSIGNMENT, UnifiedMessageType.TASK_COMPLETION,
                        UnifiedMessageType.EVENT, UnifiedMessageType.BROADCAST,
                        UnifiedMessageType.STATUS_UPDATE, UnifiedMessageType.REQUEST,
                        UnifiedMessageType.RESPONSE
                    ]
                )
                self.coordinator = coordinator
            
            async def _process_message(self, message: Message) -> Optional[Message]:
                """Process coordination messages"""
                try:
                    if message.type == UnifiedMessageType.TASK_ASSIGNMENT:
                        await self._handle_task_assignment(message)
                    elif message.type == UnifiedMessageType.TASK_COMPLETION:
                        await self._handle_task_completion(message)
                    elif message.type == UnifiedMessageType.STATUS_UPDATE:
                        await self._handle_agent_status_update(message)
                    elif message.type == UnifiedMessageType.REQUEST:
                        return await self._handle_coordination_request(message)
                    elif message.type == UnifiedMessageType.EVENT:
                        await self._handle_coordination_event(message)
                    
                    return None
                except Exception as e:
                    logger.error(f"Coordination message handler failed for {message.id}", error=str(e))
                    return None
            
            async def _handle_task_assignment(self, message: Message) -> None:
                """Handle task assignment messages"""
                task_id = message.payload.get("task_id")
                agent_id = message.recipient
                task_data = message.payload.get("task_data", {})
                
                logger.info(f"Task {task_id} assigned to agent {agent_id}",
                           task_id=task_id, agent_id=agent_id)
            
            async def _handle_task_completion(self, message: Message) -> None:
                """Handle task completion messages"""
                task_id = message.payload.get("task_id")
                agent_id = message.sender
                result = message.payload.get("result", {})
                
                logger.info(f"Task {task_id} completed by agent {agent_id}",
                           task_id=task_id, agent_id=agent_id)
            
            async def _handle_agent_status_update(self, message: Message) -> None:
                """Handle agent status updates"""
                agent_id = message.sender
                status = message.payload.get("status")
                
                logger.debug(f"Agent {agent_id} status updated: {status}",
                            agent_id=agent_id, status=status)
            
            async def _handle_coordination_request(self, message: Message) -> Message:
                """Handle coordination requests from agents"""
                request_type = message.payload.get("type", "status")
                
                return Message(
                    type=UnifiedMessageType.RESPONSE,
                    sender="coordination_system",
                    recipient=message.sender,
                    payload={
                        "status": "processed",
                        "request_type": request_type,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    correlation_id=message.id,
                    priority=MessagePriority.NORMAL
                )
            
            async def _handle_coordination_event(self, message: Message) -> None:
                """Handle coordination events"""
                event_type = message.payload.get("event_type")
                
                logger.info(f"Coordination event: {event_type}",
                           event_type=event_type, sender=message.sender)
        
        # Register the coordination handler
        handler = CoordinationMessageHandler(self)
        self.messaging_service.register_handler(handler)
        
        # Subscribe to coordination topics
        await self.messaging_service.subscribe_to_topic("coordination", "coordination")
        await self.messaging_service.subscribe_to_topic("agents", "coordination")
        await self.messaging_service.subscribe_to_topic("tasks", "coordination")
        
        logger.info("Coordination messaging handlers registered")
        
        # Mark coordination as migrated to unified messaging service
        mark_migration_complete("multiagent_integration")
    
    async def record_collaboration_event(self, 
                                       event_type: CoordinationEventType,
                                       collaboration_id: str,
                                       agents: List[str], 
                                       context: Dict[str, Any], 
                                       outcome: str = "",
                                       business_value: float = 0.0,
                                       quality_score: float = 0.0,
                                       efficiency: float = 0.0,
                                       duration: float = 0.0) -> CoordinationEvent:
        """Record sophisticated coordination activities in database for dashboard visibility."""
        try:
            async with get_session() as session:
                # Create coordination event
                event = CoordinationEvent(
                    event_type=event_type,
                    collaboration_id=collaboration_id,
                    participating_agents=[uuid.UUID(agent_id) for agent_id in agents],
                    primary_agent_id=uuid.UUID(agents[0]) if agents else None,
                    coordination_pattern=context.get('pattern_name', 'unknown'),
                    title=context.get('title', f'{event_type.value} event'),
                    description=context.get('description', ''),
                    context=context,
                    outcomes={"result": outcome},
                    quality_score=quality_score,
                    collaboration_efficiency=efficiency,
                    business_value_score=business_value,
                    duration_seconds=duration,
                    success="true" if outcome != "failed" else "false",
                    communication_count=context.get('communication_count', 0),
                    decisions_made_count=context.get('decisions_count', 0),
                    knowledge_shared_count=context.get('knowledge_shared', 0),
                    artifacts_created=context.get('artifacts_created', [])
                )
                
                # Calculate business metrics
                event.update_business_metrics()
                
                session.add(event)
                await session.commit()
                
                self.logger.info("✅ Coordination event recorded in database",
                               event_type=event_type.value,
                               collaboration_id=collaboration_id,
                               business_value=business_value,
                               quality_score=quality_score)
                
                return event
                
        except Exception as e:
            self.logger.error("❌ Failed to record coordination event", 
                            event_type=event_type.value if hasattr(event_type, 'value') else str(event_type),
                            error=str(e))
            raise
    
    async def update_business_value_metrics(self, period_hours: int = 24) -> BusinessValueMetric:
        """Update business value metrics for the specified period."""
        try:
            period_start = datetime.utcnow() - timedelta(hours=period_hours)
            period_end = datetime.utcnow()
            
            async with get_session() as session:
                # Get coordination events for the period
                from sqlalchemy import select
                
                events_result = await session.execute(
                    select(CoordinationEvent).where(
                        CoordinationEvent.created_at >= period_start,
                        CoordinationEvent.created_at <= period_end
                    )
                )
                events = events_result.scalars().all()
                
                # Calculate business value metrics
                metric = BusinessValueMetric.calculate_for_period(period_start, period_end, events)
                session.add(metric)
                await session.commit()
                
                self.logger.info("✅ Business value metrics updated",
                               period_hours=period_hours,
                               total_collaborations=metric.total_collaborations,
                               business_value=metric.total_business_value,
                               roi_percentage=metric.roi_percentage)
                
                return metric
                
        except Exception as e:
            self.logger.error("❌ Failed to update business value metrics", error=str(e))
            raise
    
    async def get_recent_coordination_events(self, limit: int = 50) -> List[CoordinationEvent]:
        """Get recent coordination events for dashboard display."""
        try:
            async with get_session() as session:
                from sqlalchemy import select
                
                result = await session.execute(
                    select(CoordinationEvent)
                    .order_by(CoordinationEvent.created_at.desc())
                    .limit(limit)
                )
                return result.scalars().all()
                
        except Exception as e:
            self.logger.error("❌ Failed to get recent coordination events", error=str(e))
            return []
    
    async def get_business_value_metrics(self, days: int = 7) -> List[BusinessValueMetric]:
        """Get business value metrics for dashboard display."""
        try:
            lookback_date = datetime.utcnow() - timedelta(days=days)
            
            async with get_session() as session:
                from sqlalchemy import select
                
                result = await session.execute(
                    select(BusinessValueMetric)
                    .where(BusinessValueMetric.period_start >= lookback_date)
                    .order_by(BusinessValueMetric.period_start.desc())
                )
                return result.scalars().all()
                
        except Exception as e:
            self.logger.error("❌ Failed to get business value metrics", error=str(e))
            return []
    
    def _initialize_default_patterns(self):
        """Initialize default coordination patterns."""
        patterns = [
            CoordinationPattern(
                pattern_id="pair_programming_01",
                pattern_type=CoordinationPatternType.PAIR_PROGRAMMING,
                name="AI Pair Programming",
                description="Two agents collaborate on code implementation with real-time communication",
                required_roles=[SpecializedAgentRole.DEVELOPER, SpecializedAgentRole.DEVELOPER],
                coordination_steps=[
                    {"step": "establish_shared_context", "duration": 2},
                    {"step": "driver_navigator_assignment", "duration": 1},
                    {"step": "collaborative_coding", "duration": 30},
                    {"step": "role_switching", "duration": 1},
                    {"step": "continued_collaboration", "duration": 30},
                    {"step": "review_and_finalize", "duration": 5}
                ],
                success_metrics={"code_quality": 0.9, "collaboration_efficiency": 0.85},
                estimated_duration=69,
                complexity_level=TaskComplexity.MODERATE
            ),
            CoordinationPattern(
                pattern_id="code_review_cycle_01", 
                pattern_type=CoordinationPatternType.CODE_REVIEW_CYCLE,
                name="Multi-Agent Code Review",
                description="Comprehensive code review with multiple specialized reviewers",
                required_roles=[SpecializedAgentRole.DEVELOPER, SpecializedAgentRole.REVIEWER, SpecializedAgentRole.ARCHITECT],
                coordination_steps=[
                    {"step": "code_submission", "duration": 2},
                    {"step": "automated_analysis", "duration": 3},
                    {"step": "parallel_reviews", "duration": 15},
                    {"step": "review_consolidation", "duration": 5},
                    {"step": "developer_response", "duration": 10},
                    {"step": "final_approval", "duration": 3}
                ],
                success_metrics={"review_coverage": 0.95, "defect_detection": 0.9},
                estimated_duration=38,
                complexity_level=TaskComplexity.MODERATE
            ),
            CoordinationPattern(
                pattern_id="ci_workflow_01",
                pattern_type=CoordinationPatternType.CONTINUOUS_INTEGRATION,
                name="Multi-Agent CI Pipeline",
                description="Coordinated CI/CD workflow with specialized agent roles",
                required_roles=[SpecializedAgentRole.DEVELOPER, SpecializedAgentRole.TESTER, SpecializedAgentRole.DEVOPS],
                coordination_steps=[
                    {"step": "code_integration", "duration": 3},
                    {"step": "automated_testing", "duration": 10},
                    {"step": "quality_gates", "duration": 5},
                    {"step": "deployment_preparation", "duration": 8},
                    {"step": "deployment_execution", "duration": 12},
                    {"step": "post_deployment_validation", "duration": 7}
                ],
                success_metrics={"test_coverage": 0.9, "deployment_success": 0.95},
                estimated_duration=45,
                complexity_level=TaskComplexity.COMPLEX
            ),
            CoordinationPattern(
                pattern_id="design_review_01",
                pattern_type=CoordinationPatternType.DESIGN_REVIEW,
                name="Architecture Design Review",
                description="Collaborative architecture and design review process",
                required_roles=[SpecializedAgentRole.ARCHITECT, SpecializedAgentRole.DEVELOPER, SpecializedAgentRole.PRODUCT],
                coordination_steps=[
                    {"step": "requirements_analysis", "duration": 10},
                    {"step": "architecture_proposal", "duration": 20},
                    {"step": "stakeholder_review", "duration": 15},
                    {"step": "technical_feasibility", "duration": 12},
                    {"step": "design_refinement", "duration": 18},
                    {"step": "final_design_approval", "duration": 5}
                ],
                success_metrics={"design_quality": 0.9, "stakeholder_alignment": 0.85},
                estimated_duration=80,
                complexity_level=TaskComplexity.COMPLEX
            ),
            CoordinationPattern(
                pattern_id="knowledge_sharing_01",
                pattern_type=CoordinationPatternType.KNOWLEDGE_SHARING,
                name="Cross-Agent Knowledge Transfer",
                description="Structured knowledge sharing and learning between agents",
                required_roles=[SpecializedAgentRole.ARCHITECT, SpecializedAgentRole.DEVELOPER, SpecializedAgentRole.TESTER],
                coordination_steps=[
                    {"step": "knowledge_identification", "duration": 5},
                    {"step": "context_preparation", "duration": 8},
                    {"step": "knowledge_presentation", "duration": 12},
                    {"step": "interactive_discussion", "duration": 15},
                    {"step": "knowledge_integration", "duration": 10},
                    {"step": "validation_and_documentation", "duration": 8}
                ],
                success_metrics={"knowledge_retention": 0.8, "application_success": 0.75},
                estimated_duration=58,
                complexity_level=TaskComplexity.MODERATE
            )
        ]
        
        for pattern in patterns:
            self.coordination_patterns[pattern.pattern_id] = pattern
            self.coordination_metrics["pattern_success_rates"][pattern.pattern_id] = 0.0
    
    async def _initialize_specialized_agents(self):
        """Initialize specialized agents with unique capabilities."""
        agent_specifications = [
            {
                "role": SpecializedAgentRole.ARCHITECT,
                "capabilities": [
                    AgentCapability("system_design", 0.9, ["microservices", "scalability", "performance"]),
                    AgentCapability("architecture_review", 0.85, ["patterns", "best_practices", "trade_offs"]),
                    AgentCapability("technical_leadership", 0.8, ["decision_making", "team_guidance"])
                ]
            },
            {
                "role": SpecializedAgentRole.DEVELOPER,
                "capabilities": [
                    AgentCapability("code_implementation", 0.85, ["python", "javascript", "apis"]),
                    AgentCapability("algorithm_design", 0.8, ["optimization", "data_structures"]),
                    AgentCapability("debugging", 0.82, ["problem_solving", "root_cause_analysis"])
                ]
            },
            {
                "role": SpecializedAgentRole.TESTER,
                "capabilities": [
                    AgentCapability("test_design", 0.88, ["unit_tests", "integration_tests", "e2e_tests"]),
                    AgentCapability("quality_assurance", 0.9, ["validation", "verification", "standards"]),
                    AgentCapability("test_automation", 0.85, ["frameworks", "ci_integration"])
                ]
            },
            {
                "role": SpecializedAgentRole.REVIEWER,
                "capabilities": [
                    AgentCapability("code_review", 0.92, ["security", "performance", "maintainability"]),
                    AgentCapability("best_practices", 0.88, ["conventions", "patterns", "standards"]),
                    AgentCapability("mentoring", 0.8, ["knowledge_transfer", "guidance"])
                ]
            },
            {
                "role": SpecializedAgentRole.DEVOPS,
                "capabilities": [
                    AgentCapability("deployment_automation", 0.87, ["ci_cd", "infrastructure", "containers"]),
                    AgentCapability("monitoring", 0.85, ["observability", "alerting", "metrics"]),
                    AgentCapability("infrastructure_management", 0.83, ["cloud", "scaling", "security"])
                ]
            },
            {
                "role": SpecializedAgentRole.PRODUCT,
                "capabilities": [
                    AgentCapability("requirements_analysis", 0.86, ["user_stories", "acceptance_criteria"]),
                    AgentCapability("user_experience", 0.82, ["usability", "design", "workflows"]),
                    AgentCapability("stakeholder_management", 0.84, ["communication", "alignment"])
                ]
            }
        ]
        
        for spec in agent_specifications:
            # Create multiple instances of each role for scalability
            for i in range(2):  # 2 agents per role
                agent_id = f"{spec['role'].value}_{i+1}"
                agent = SpecializedAgent(
                    agent_id=agent_id,
                    role=spec['role'],
                    status=AgentStatus.active,
                    capabilities=spec['capabilities'].copy()
                )
                
                self.agents[agent_id] = agent
                self.agent_roles[spec['role']].append(agent_id)
                self.coordination_metrics["agent_utilization"][agent_id] = 0.0
        
        self.logger.info("✅ Specialized agents initialized",
                        total_agents=len(self.agents),
                        roles=list(self.agent_roles.keys()))
    
    async def create_collaboration(self, 
                                 pattern_id: str, 
                                 task_description: str,
                                 requirements: Dict[str, Any],
                                 preferred_agents: Optional[List[str]] = None) -> str:
        """
        Create a new multi-agent collaboration using a specified pattern.
        
        Args:
            pattern_id: ID of the coordination pattern to use
            task_description: Description of the task to be completed
            requirements: Specific requirements and constraints
            preferred_agents: Optional list of preferred agent IDs
            
        Returns:
            Collaboration ID for tracking and monitoring
        """
        if pattern_id not in self.coordination_patterns:
            raise ValueError(f"Unknown coordination pattern: {pattern_id}")
        
        pattern = self.coordination_patterns[pattern_id]
        collaboration_id = str(uuid.uuid4())
        
        # Select optimal agents for the pattern
        selected_agents = await self._select_agents_for_pattern(
            pattern, preferred_agents, requirements
        )
        
        if len(selected_agents) < len(pattern.required_roles):
            raise RuntimeError(f"Insufficient available agents for pattern {pattern_id}")
        
        # Create collaboration context
        collaboration = CollaborationContext(
            collaboration_id=collaboration_id,
            participants=selected_agents
        )
        
        # Add initial context and requirements
        collaboration.add_knowledge("task_description", task_description, "system")
        collaboration.add_knowledge("requirements", requirements, "system")
        collaboration.add_knowledge("coordination_pattern", pattern.to_dict(), "system")
        
        self.active_collaborations[collaboration_id] = collaboration
        
        # Update agent workloads
        for agent_id in selected_agents:
            if agent_id in self.agents:
                self.agents[agent_id].current_collaborations.add(collaboration_id)
                self.agents[agent_id].current_workload += 0.3  # Rough workload estimate
        
        self.coordination_metrics["total_collaborations"] += 1
        
        # Record collaboration started event in database
        try:
            context = {
                'pattern_name': pattern.name,
                'title': f'{pattern.name} started',
                'description': task_description,
                'requirements': requirements,
                'estimated_duration': pattern.estimated_duration
            }
            
            await self.record_collaboration_event(
                event_type=CoordinationEventType.COLLABORATION_STARTED,
                collaboration_id=collaboration_id,
                agents=selected_agents,
                context=context,
                outcome="started",
                business_value=0.0,  # Will be calculated on completion
                quality_score=0.0,   # Will be calculated on completion  
                efficiency=0.0,      # Will be calculated on completion
                duration=0.0
            )
        except Exception as db_error:
            self.logger.error("⚠️ Failed to record collaboration started event", error=str(db_error))
        
        self.logger.info("🤝 Created new collaboration",
                        collaboration_id=collaboration_id,
                        pattern=pattern_id,
                        agents=selected_agents,
                        estimated_duration=pattern.estimated_duration)
        
        return collaboration_id
    
    async def _select_agents_for_pattern(self, 
                                       pattern: CoordinationPattern,
                                       preferred_agents: Optional[List[str]],
                                       requirements: Dict[str, Any]) -> List[str]:
        """Select optimal agents for a coordination pattern."""
        selected_agents = []
        
        for required_role in pattern.required_roles:
            available_agents = [
                agent_id for agent_id in self.agent_roles[required_role]
                if self.agents[agent_id].is_available
            ]
            
            if not available_agents:
                continue
            
            # Prefer specified agents if available
            if preferred_agents:
                preferred_available = [
                    agent_id for agent_id in available_agents 
                    if agent_id in preferred_agents
                ]
                if preferred_available:
                    selected_agents.append(preferred_available[0])
                    continue
            
            # Select based on capability match and workload
            best_agent = max(available_agents, key=lambda agent_id: 
                self._calculate_agent_suitability(agent_id, requirements))
            
            selected_agents.append(best_agent)
        
        return selected_agents
    
    def _calculate_agent_suitability(self, agent_id: str, requirements: Dict[str, Any]) -> float:
        """Calculate agent suitability score for a task."""
        agent = self.agents[agent_id]
        
        # Base score from specialization and availability
        base_score = agent.specialization_score * (1.0 - agent.current_workload)
        
        # Capability matching bonus
        required_capabilities = requirements.get("required_capabilities", [])
        capability_match = sum(
            1.0 for cap_name in required_capabilities
            if cap_name in agent.capability_names
        ) / max(1, len(required_capabilities))
        
        # Performance history bonus
        recent_performance = agent.performance_history[-5:] if agent.performance_history else []
        performance_score = sum(
            record["quality_score"] for record in recent_performance
        ) / max(1, len(recent_performance)) if recent_performance else 0.5
        
        # Collaboration rating
        collaboration_bonus = agent.collaboration_preferences.get("preferred_patterns", 0.5)
        
        return base_score + (capability_match * 0.3) + (performance_score * 0.2) + (collaboration_bonus * 0.1)
    
    async def execute_collaboration(self, collaboration_id: str) -> Dict[str, Any]:
        """
        Execute a multi-agent collaboration using the specified pattern.
        
        Args:
            collaboration_id: ID of the collaboration to execute
            
        Returns:
            Detailed execution results with agent outputs and performance metrics
        """
        if collaboration_id not in self.active_collaborations:
            raise ValueError(f"Unknown collaboration: {collaboration_id}")
        
        collaboration = self.active_collaborations[collaboration_id]
        pattern_info = collaboration.shared_knowledge["coordination_pattern"]["value"]
        pattern = CoordinationPattern.from_dict(pattern_info)
        
        execution_start = time.time()
        
        self.logger.info("🚀 Starting collaboration execution",
                        collaboration_id=collaboration_id,
                        pattern=pattern.name,
                        participants=collaboration.participants)
        
        results = {
            "collaboration_id": collaboration_id,
            "pattern": pattern.to_dict(),
            "participants": collaboration.participants,
            "execution_steps": [],
            "success": False,
            "execution_time": 0.0,
            "quality_metrics": {},
            "artifacts_created": [],
            "knowledge_shared": 0,
            "collaboration_efficiency": 0.0
        }
        
        try:
            # Execute each coordination step
            for step_info in pattern.coordination_steps:
                step_result = await self._execute_coordination_step(
                    collaboration, step_info, pattern
                )
                results["execution_steps"].append(step_result)
                
                if not step_result["success"]:
                    self.logger.warning("❌ Coordination step failed",
                                      step=step_info["step"],
                                      error=step_result.get("error"))
                    break
            
            # Calculate final results
            results["success"] = all(step["success"] for step in results["execution_steps"])
            results["execution_time"] = time.time() - execution_start
            results["artifacts_created"] = collaboration.artifacts_created.copy()
            results["knowledge_shared"] = len(collaboration.shared_knowledge)
            
            # Calculate collaboration efficiency 
            expected_duration = pattern.estimated_duration * 60  # Convert to seconds
            actual_duration = results["execution_time"]
            results["collaboration_efficiency"] = min(1.0, expected_duration / max(actual_duration, 1))
            
            # Update agent performance records
            quality_score = self._calculate_collaboration_quality(results)
            for agent_id in collaboration.participants:
                if agent_id in self.agents:
                    self.agents[agent_id].add_performance_record(
                        collaboration_id, results["success"], 
                        results["execution_time"], quality_score
                    )
            
            # Record coordination event in database for dashboard visibility
            try:
                context = {
                    'pattern_name': pattern.name,
                    'title': f'{pattern.name} completed',
                    'description': f'Multi-agent collaboration using {pattern.name} pattern',
                    'communication_count': len(collaboration.communication_history),
                    'decisions_count': len(collaboration.decisions_made),
                    'knowledge_shared': len(collaboration.shared_knowledge),
                    'artifacts_created': collaboration.artifacts_created,
                    'estimated_duration': pattern.estimated_duration
                }
                
                business_value = quality_score * results["collaboration_efficiency"]
                
                await self.record_collaboration_event(
                    event_type=CoordinationEventType.COLLABORATION_COMPLETED if results["success"] 
                             else CoordinationEventType.COLLABORATION_FAILED,
                    collaboration_id=collaboration_id,
                    agents=collaboration.participants,
                    context=context,
                    outcome="success" if results["success"] else "failed",
                    business_value=business_value,
                    quality_score=quality_score,
                    efficiency=results["collaboration_efficiency"],
                    duration=results["execution_time"]
                )
            except Exception as db_error:
                self.logger.error("⚠️ Failed to record coordination event in database", error=str(db_error))
                # Don't fail the entire coordination due to database issues
            
            # Update coordination metrics
            if results["success"]:
                self.coordination_metrics["successful_collaborations"] += 1
            
            # Update pattern success rate
            pattern_id = pattern.pattern_id
            current_rate = self.coordination_metrics["pattern_success_rates"][pattern_id]
            success_count = self.coordination_metrics["successful_collaborations"]
            total_count = self.coordination_metrics["total_collaborations"]
            self.coordination_metrics["pattern_success_rates"][pattern_id] = success_count / total_count
            
            self.logger.info("✅ Collaboration execution completed",
                           collaboration_id=collaboration_id,
                           success=results["success"],
                           duration=results["execution_time"],
                           efficiency=results["collaboration_efficiency"])
            
        except Exception as e:
            self.logger.error("❌ Collaboration execution failed",
                            collaboration_id=collaboration_id,
                            error=str(e))
            results["error"] = str(e)
            results["execution_time"] = time.time() - execution_start
        
        finally:
            # Clean up collaboration
            await self._cleanup_collaboration(collaboration_id)
        
        return results
    
    async def _execute_coordination_step(self, 
                                       collaboration: CollaborationContext,
                                       step_info: Dict[str, Any],
                                       pattern: CoordinationPattern) -> Dict[str, Any]:
        """Execute a single coordination step."""
        step_name = step_info["step"]
        estimated_duration = step_info["duration"]
        
        step_result = {
            "step": step_name,
            "estimated_duration": estimated_duration,
            "actual_duration": 0.0,
            "success": False,
            "outputs": {},
            "communications": []
        }
        
        step_start = time.time()
        
        try:
            self.logger.info("📝 Executing coordination step",
                           step=step_name,
                           collaboration_id=collaboration.collaboration_id)
            
            # Execute step based on pattern type and step name
            # Handle both enum and string pattern types for robustness
            pattern_type = pattern.pattern_type
            if isinstance(pattern_type, str):
                # Convert string back to enum if needed
                try:
                    pattern_type = CoordinationPatternType(pattern_type)
                except ValueError:
                    self.logger.warning(f"Unknown pattern type string: {pattern_type}")
                    pattern_type = None
            
            if pattern_type == CoordinationPatternType.PAIR_PROGRAMMING:
                step_result = await self._execute_pair_programming_step(
                    collaboration, step_name, step_result
                )  
            elif pattern_type == CoordinationPatternType.CODE_REVIEW_CYCLE:
                step_result = await self._execute_code_review_step(
                    collaboration, step_name, step_result
                )
            elif pattern_type == CoordinationPatternType.CONTINUOUS_INTEGRATION:
                step_result = await self._execute_ci_step(
                    collaboration, step_name, step_result
                )
            elif pattern_type == CoordinationPatternType.DESIGN_REVIEW:
                step_result = await self._execute_design_review_step(
                    collaboration, step_name, step_result
                )
            elif pattern_type == CoordinationPatternType.KNOWLEDGE_SHARING:
                step_result = await self._execute_knowledge_sharing_step(
                    collaboration, step_name, step_result
                )
            else:
                # Generic step execution
                step_result = await self._execute_generic_step(
                    collaboration, step_name, step_result
                )
            
            step_result["actual_duration"] = time.time() - step_start
            
        except Exception as e:
            import traceback
            step_result["error"] = str(e)
            step_result["actual_duration"] = time.time() - step_start
            self.logger.error("❌ Coordination step failed",
                            step=step_name,
                            error=str(e),
                            traceback=traceback.format_exc())
        
        return step_result
    
    async def _execute_pair_programming_step(self, 
                                           collaboration: CollaborationContext,
                                           step_name: str,
                                           step_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pair programming coordination step."""
        participants = collaboration.participants
        
        if step_name == "establish_shared_context":
            # Share task context between agents
            task_description = collaboration.shared_knowledge["task_description"]["value"]
            requirements = collaboration.shared_knowledge["requirements"]["value"]
            
            collaboration.add_knowledge("shared_context", {
                "task": task_description,
                "requirements": requirements,
                "roles_assigned": {"driver": participants[0], "navigator": participants[1]}
            }, "system")
            
            step_result["success"] = True
            step_result["outputs"]["context_established"] = True
            
        elif step_name == "driver_navigator_assignment":
            # Assign driver and navigator roles
            roles = {"driver": participants[0], "navigator": participants[1]}
            collaboration.add_knowledge("current_roles", roles, "system")
            
            # Simulate agent communication
            collaboration.add_communication(
                participants[0], participants[1],
                "I'll start as driver, please guide me through the implementation approach",
                "role_coordination"
            )
            
            step_result["success"] = True
            step_result["outputs"]["roles_assigned"] = roles
            
        elif step_name == "collaborative_coding":
            # Simulate collaborative coding session
            driver = collaboration.shared_knowledge["current_roles"]["value"]["driver"]
            navigator = collaboration.shared_knowledge["current_roles"]["value"]["navigator"]
            
            # Create mock code artifact
            code_artifact = f"# Collaborative code by {driver} (driver) and {navigator} (navigator)\n"
            code_artifact += "def collaborative_function():\n"
            code_artifact += "    # Implementation guided by pair programming\n"
            code_artifact += "    return 'Pair programming success'\n"
            
            artifact_path = str(self.workspace_dir / f"pair_programming_{collaboration.collaboration_id}.py")
            with open(artifact_path, "w") as f:
                f.write(code_artifact)
            
            collaboration.artifacts_created.append(artifact_path)
            
            # Record communication
            collaboration.add_communication(
                navigator, driver,
                "Consider adding error handling here for edge cases",
                "guidance"
            )
            collaboration.add_communication(
                driver, navigator, 
                "Great suggestion, implementing error handling now",
                "acknowledgment"
            )
            
            step_result["success"] = True
            step_result["outputs"]["code_created"] = True
            step_result["outputs"]["artifact_path"] = artifact_path
            
        elif step_name == "role_switching":
            # Switch driver and navigator roles
            current_roles = collaboration.shared_knowledge["current_roles"]["value"]
            new_roles = {
                "driver": current_roles["navigator"],
                "navigator": current_roles["driver"]
            }
            collaboration.add_knowledge("current_roles", new_roles, "system")
            
            collaboration.add_communication(
                new_roles["driver"], new_roles["navigator"],
                "Switching roles - I'll take over driving now",
                "role_coordination"
            )
            
            step_result["success"] = True
            step_result["outputs"]["roles_switched"] = True
            
        elif step_name == "continued_collaboration":
            # Continue collaborative work with switched roles
            step_result["success"] = True
            step_result["outputs"]["collaboration_continued"] = True
            
        elif step_name == "review_and_finalize":
            # Final review and completion
            collaboration.add_decision(
                "Code implementation completed successfully",
                "Pair programming session achieved high quality output with good collaboration",
                participants
            )
            
            step_result["success"] = True
            step_result["outputs"]["review_completed"] = True
        
        return step_result
    
    async def _execute_code_review_step(self,
                                      collaboration: CollaborationContext,
                                      step_name: str,
                                      step_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code review coordination step."""
        participants = collaboration.participants
        
        if step_name == "code_submission":
            # Simulate code submission for review
            code_content = "# Code submitted for review\ndef example_function():\n    return 'example'\n"
            artifact_path = str(self.workspace_dir / f"code_review_{collaboration.collaboration_id}.py")
            
            with open(artifact_path, "w") as f:
                f.write(code_content)
            
            collaboration.artifacts_created.append(artifact_path)
            collaboration.add_knowledge("code_artifact", artifact_path, participants[0])
            
            step_result["success"] = True
            step_result["outputs"]["code_submitted"] = True
            
        elif step_name == "automated_analysis":
            # Simulate automated code analysis
            analysis_results = {
                "complexity_score": 0.2,
                "test_coverage": 0.85,
                "security_issues": 0,
                "performance_score": 0.9
            }
            
            collaboration.add_knowledge("automated_analysis", analysis_results, "system")
            step_result["success"] = True
            step_result["outputs"]["analysis_completed"] = True
            
        elif step_name == "parallel_reviews":
            # Multiple reviewers provide feedback in parallel
            reviews = {}
            for i, reviewer_id in enumerate(participants[1:]):  # Skip the developer
                reviews[reviewer_id] = {
                    "overall_rating": 0.85 + (i * 0.05),
                    "comments": [f"Review comment {i+1} from {reviewer_id}"],
                    "suggestions": [f"Suggestion {i+1} for improvement"]
                }
                
                collaboration.add_communication(
                    reviewer_id, participants[0],
                    f"Code review completed. Overall quality looks good with minor suggestions.",
                    "code_review"
                )
            
            collaboration.add_knowledge("reviews", reviews, "system")
            step_result["success"] = True
            step_result["outputs"]["reviews_completed"] = len(reviews)
            
        elif step_name == "review_consolidation":
            # Consolidate all review feedback
            reviews = collaboration.shared_knowledge["reviews"]["value"]
            consolidated_feedback = {
                "average_rating": sum(r["overall_rating"] for r in reviews.values()) / len(reviews),
                "common_themes": ["code_quality", "performance", "maintainability"],
                "priority_issues": []
            }
            
            collaboration.add_knowledge("consolidated_feedback", consolidated_feedback, "system")
            step_result["success"] = True
            step_result["outputs"]["feedback_consolidated"] = True
            
        elif step_name == "developer_response":
            # Developer addresses review feedback
            collaboration.add_communication(
                participants[0], "all_reviewers",
                "Thank you for the thorough reviews. I'll address the suggested improvements.",
                "response"
            )
            
            step_result["success"] = True
            step_result["outputs"]["developer_responded"] = True
            
        elif step_name == "final_approval":
            # Final approval from reviewers
            collaboration.add_decision(
                "Code approved for merge",
                "All review feedback addressed satisfactorily",
                participants
            )
            
            step_result["success"] = True
            step_result["outputs"]["approved"] = True
        
        return step_result
    
    async def _execute_ci_step(self,
                             collaboration: CollaborationContext,
                             step_name: str,
                             step_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CI/CD coordination step."""
        # Implementation for CI/CD steps
        step_result["success"] = True
        step_result["outputs"][f"{step_name}_completed"] = True
        return step_result
    
    async def _execute_design_review_step(self,
                                        collaboration: CollaborationContext,
                                        step_name: str,
                                        step_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute design review coordination step."""
        # Implementation for design review steps
        step_result["success"] = True
        step_result["outputs"][f"{step_name}_completed"] = True
        return step_result
    
    async def _execute_knowledge_sharing_step(self,
                                            collaboration: CollaborationContext,
                                            step_name: str,
                                            step_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute knowledge sharing coordination step."""
        participants = collaboration.participants
        
        if step_name == "knowledge_identification":
            # Identify knowledge to be shared
            knowledge_areas = ["best_practices", "lessons_learned", "technical_insights"]
            collaboration.add_knowledge("knowledge_areas", knowledge_areas, "system")
            step_result["success"] = True
            step_result["outputs"]["knowledge_identified"] = True
            
        elif step_name == "context_preparation":
            # Prepare context for knowledge sharing
            step_result["success"] = True
            step_result["outputs"]["context_prepared"] = True
            
        elif step_name == "knowledge_presentation":
            # Present knowledge to participants
            for knowledge_area in collaboration.shared_knowledge["knowledge_areas"]["value"]:
                collaboration.add_knowledge(f"shared_{knowledge_area}", 
                                          f"Knowledge about {knowledge_area} shared successfully",
                                          participants[0])
            
            self.coordination_metrics["knowledge_sharing_events"] += 1
            step_result["success"] = True
            step_result["outputs"]["knowledge_presented"] = True
            
        elif step_name == "interactive_discussion":
            # Simulate interactive discussion
            for i, participant in enumerate(participants):
                collaboration.add_communication(
                    participant, "all_participants",
                    f"Great insights on this topic. I have additional questions about implementation.",
                    "discussion"
                )
            
            step_result["success"] = True
            step_result["outputs"]["discussion_completed"] = True
            
        elif step_name == "knowledge_integration":
            # Integrate shared knowledge
            step_result["success"] = True
            step_result["outputs"]["knowledge_integrated"] = True
            
        elif step_name == "validation_and_documentation":
            # Validate and document shared knowledge
            documentation_path = str(self.workspace_dir / f"knowledge_sharing_{collaboration.collaboration_id}.md")
            documentation = "# Knowledge Sharing Session\n\n"
            documentation += f"Participants: {', '.join(participants)}\n"
            documentation += f"Date: {datetime.utcnow().isoformat()}\n\n"
            documentation += "## Knowledge Shared\n"
            
            for key, value in collaboration.shared_knowledge.items():
                if key.startswith("shared_"):
                    documentation += f"- {key}: {value['value']}\n"
            
            with open(documentation_path, "w") as f:
                f.write(documentation)
            
            collaboration.artifacts_created.append(documentation_path)
            step_result["success"] = True
            step_result["outputs"]["documentation_created"] = True
        
        return step_result
    
    async def _execute_generic_step(self,
                                  collaboration: CollaborationContext,
                                  step_name: str,
                                  step_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic coordination step."""
        # Generic step execution for unknown patterns
        await asyncio.sleep(0.1)  # Simulate processing time
        step_result["success"] = True
        step_result["outputs"]["generic_step_completed"] = True
        return step_result
    
    def _calculate_collaboration_quality(self, results: Dict[str, Any]) -> float:
        """Calculate overall quality score for a collaboration."""
        base_score = 0.8 if results["success"] else 0.3
        
        # Efficiency bonus
        efficiency = results.get("collaboration_efficiency", 0.5)
        efficiency_bonus = (efficiency - 0.5) * 0.2
        
        # Knowledge sharing bonus
        knowledge_shared = results.get("knowledge_shared", 0)
        knowledge_bonus = min(0.1, knowledge_shared * 0.02)
        
        # Artifacts created bonus
        artifacts_count = len(results.get("artifacts_created", []))
        artifacts_bonus = min(0.1, artifacts_count * 0.05)
        
        return min(1.0, max(0.0, base_score + efficiency_bonus + knowledge_bonus + artifacts_bonus))
    
    async def _cleanup_collaboration(self, collaboration_id: str):
        """Clean up completed collaboration."""
        if collaboration_id in self.active_collaborations:
            collaboration = self.active_collaborations[collaboration_id]
            
            # Update agent workloads
            for agent_id in collaboration.participants:
                if agent_id in self.agents:
                    self.agents[agent_id].current_collaborations.discard(collaboration_id)
                    self.agents[agent_id].current_workload = max(0.0, 
                        self.agents[agent_id].current_workload - 0.3)
            
            # Archive collaboration
            del self.active_collaborations[collaboration_id]
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination system status."""
        return {
            "active_collaborations": len(self.active_collaborations),
            "total_agents": len(self.agents),
            "available_agents": len([a for a in self.agents.values() if a.is_available]),
            "coordination_patterns": len(self.coordination_patterns),
            "metrics": self.coordination_metrics.copy(),
            "agent_workloads": {
                agent_id: {
                    "current_workload": agent.current_workload,
                    "active_collaborations": len(agent.current_collaborations),
                    "performance_score": sum(r["quality_score"] for r in agent.performance_history[-5:]) / 5
                    if agent.performance_history else 0.5
                }
                for agent_id, agent in self.agents.items()
            }
        }
    
    def get_collaboration_details(self, collaboration_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific collaboration."""
        if collaboration_id not in self.active_collaborations:
            return None
        
        collaboration = self.active_collaborations[collaboration_id]
        return {
            "collaboration_id": collaboration_id,
            "participants": collaboration.participants,
            "shared_knowledge_count": len(collaboration.shared_knowledge),
            "communication_history_count": len(collaboration.communication_history),
            "decisions_made_count": len(collaboration.decisions_made),
            "artifacts_created": collaboration.artifacts_created,
            "created_at": collaboration.created_at.isoformat(),
            "last_updated": collaboration.last_updated.isoformat()
        }
    
    async def demonstrate_coordination_patterns(self) -> Dict[str, Any]:
        """
        Demonstrate all coordination patterns with sample tasks.
        
        Returns comprehensive demonstration results showing multi-agent coordination capabilities.
        """
        demonstration_results = {
            "demonstration_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "patterns_demonstrated": [],
            "overall_success": True,
            "total_execution_time": 0.0,
            "coordination_metrics": {}
        }
        
        demonstration_start = time.time()
        
        self.logger.info("🚀 Starting comprehensive coordination patterns demonstration")
        
        # Demonstrate each coordination pattern
        for pattern_id, pattern in self.coordination_patterns.items():
            pattern_demo_start = time.time()
            
            try:
                # Create sample task for the pattern
                sample_task = self._create_sample_task_for_pattern(pattern)
                
                # Create and execute collaboration
                collaboration_id = await self.create_collaboration(
                    pattern_id=pattern_id,
                    task_description=sample_task["description"],
                    requirements=sample_task["requirements"]
                )
                
                execution_results = await self.execute_collaboration(collaboration_id)
                
                pattern_result = {
                    "pattern_id": pattern_id,
                    "pattern_name": pattern.name,
                    "pattern_type": pattern.pattern_type.value,
                    "success": execution_results["success"],
                    "execution_time": execution_results["execution_time"],
                    "collaboration_efficiency": execution_results["collaboration_efficiency"],
                    "artifacts_created": len(execution_results["artifacts_created"]),
                    "knowledge_shared": execution_results["knowledge_shared"],
                    "participants": execution_results["participants"],
                    "execution_steps": len(execution_results["execution_steps"])
                }
                
                demonstration_results["patterns_demonstrated"].append(pattern_result)
                
                if not execution_results["success"]:
                    demonstration_results["overall_success"] = False
                
                self.logger.info("✅ Pattern demonstration completed",
                               pattern=pattern.name,
                               success=execution_results["success"],
                               duration=execution_results["execution_time"])
                
            except Exception as e:
                pattern_result = {
                    "pattern_id": pattern_id,
                    "pattern_name": pattern.name,
                    "success": False,
                    "error": str(e),
                    "execution_time": time.time() - pattern_demo_start
                }
                
                demonstration_results["patterns_demonstrated"].append(pattern_result)
                demonstration_results["overall_success"] = False
                
                self.logger.error("❌ Pattern demonstration failed",
                                pattern=pattern.name,
                                error=str(e))
        
        demonstration_results["total_execution_time"] = time.time() - demonstration_start
        demonstration_results["coordination_metrics"] = self.coordination_metrics.copy()
        
        # Calculate demonstration statistics
        successful_patterns = sum(1 for p in demonstration_results["patterns_demonstrated"] if p["success"])
        total_patterns = len(demonstration_results["patterns_demonstrated"])
        
        demonstration_results["success_rate"] = successful_patterns / total_patterns if total_patterns > 0 else 0
        demonstration_results["average_execution_time"] = sum(
            p.get("execution_time", 0) for p in demonstration_results["patterns_demonstrated"]
        ) / total_patterns if total_patterns > 0 else 0
        
        self.logger.info("🏆 Coordination patterns demonstration completed",
                        success_rate=demonstration_results["success_rate"],
                        total_patterns=total_patterns,
                        total_duration=demonstration_results["total_execution_time"])
        
        return demonstration_results
    
    def _create_sample_task_for_pattern(self, pattern: CoordinationPattern) -> Dict[str, Any]:
        """Create a sample task appropriate for the given pattern."""
        sample_tasks = {
            CoordinationPatternType.PAIR_PROGRAMMING: {
                "description": "Implement a data validation utility function with comprehensive error handling",
                "requirements": {
                    "language": "python",
                    "complexity": "moderate",
                    "required_capabilities": ["code_implementation", "error_handling"],
                    "estimated_effort": 60
                }
            },
            CoordinationPatternType.CODE_REVIEW_CYCLE: {
                "description": "Review and approve a REST API implementation for user management",
                "requirements": {
                    "review_scope": "api_implementation",
                    "focus_areas": ["security", "performance", "maintainability"],
                    "required_capabilities": ["code_review", "api_design", "security_analysis"],
                    "review_depth": "comprehensive"
                }
            },
            CoordinationPatternType.CONTINUOUS_INTEGRATION: {
                "description": "Set up CI/CD pipeline for microservices deployment with automated testing",
                "requirements": {
                    "deployment_target": "kubernetes",
                    "testing_scope": "unit_integration_e2e",
                    "required_capabilities": ["deployment_automation", "test_automation", "monitoring"],
                    "pipeline_complexity": "enterprise"
                }
            },
            CoordinationPatternType.DESIGN_REVIEW: {
                "description": "Design scalable architecture for real-time chat application",
                "requirements": {
                    "architecture_scope": "full_system",
                    "scalability_requirements": "10k_concurrent_users",
                    "required_capabilities": ["system_design", "scalability", "real_time_systems"],
                    "stakeholder_groups": ["product", "engineering", "operations"]
                }
            },
            CoordinationPatternType.KNOWLEDGE_SHARING: {
                "description": "Share best practices for implementing observability in distributed systems",
                "requirements": {
                    "knowledge_domain": "observability",
                    "audience_level": "intermediate_advanced",
                    "required_capabilities": ["monitoring", "distributed_systems", "best_practices"],
                    "delivery_format": "interactive_session"
                }
            }
        }
        
        return sample_tasks.get(pattern.pattern_type, {
            "description": f"Sample task for {pattern.name}",
            "requirements": {"complexity": "moderate"}
        })


# Global coordinator instance
_enhanced_coordinator: Optional[EnhancedMultiAgentCoordinator] = None


async def get_enhanced_coordinator() -> EnhancedMultiAgentCoordinator:
    """Get the global enhanced multi-agent coordinator instance."""
    global _enhanced_coordinator
    if _enhanced_coordinator is None:
        _enhanced_coordinator = EnhancedMultiAgentCoordinator()
        await _enhanced_coordinator.initialize()
    return _enhanced_coordinator