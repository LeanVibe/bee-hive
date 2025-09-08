"""
Dynamic Agent Collaboration System - Epic 2 Phase 2 Implementation

Advanced multi-agent coordination with team formation, collaborative execution,
and consensus mechanisms for 60% more complex task completion.

Building on Epic 2 Phase 1 context intelligence foundations:
- Dynamic team formation based on task requirements and agent capabilities  
- Real-time collaborative task execution with coordination protocols
- Consensus mechanisms for complex decision-making scenarios
- Expertise routing and specialization systems
- Performance monitoring and team optimization
- Integration with Phase 1 context engine and semantic memory

Key Performance Targets:
- 60% improvement in complex task success rate
- <2s team formation time for optimal agent selection
- 70%+ resource utilization across collaborative teams
- <5s consensus decision speed
- Real-time team performance monitoring and optimization
"""

import asyncio
import uuid
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging

from .intelligent_orchestrator import (
    IntelligentOrchestrator, IntelligentTaskRequest, IntelligentTaskResult,
    AgentPerformanceProfile, get_intelligent_orchestrator
)
from .context_engine import (
    AdvancedContextEngine, TaskRoutingRecommendation, TaskRoutingStrategy,
    ContextMatch, get_context_engine
)
from .semantic_memory import (
    SemanticMemorySystem, SemanticMatch, SemanticSearchMode,
    get_semantic_memory
)
from ..models.context import Context, ContextType
from ..core.orchestrator import Orchestrator, AgentRole, TaskPriority
from ..core.logging_service import get_component_logger


logger = get_component_logger("agent_collaboration")


class AgentCapability(Enum):
    """Agent capabilities for team formation."""
    FRONTEND_DEVELOPMENT = "frontend_development"
    BACKEND_DEVELOPMENT = "backend_development"
    DEVOPS_INFRASTRUCTURE = "devops_infrastructure"
    DATA_SCIENCE = "data_science"
    TESTING_QA = "testing_qa"
    PROJECT_MANAGEMENT = "project_management"
    SECURITY_ANALYSIS = "security_analysis"
    DATABASE_DESIGN = "database_design"
    API_INTEGRATION = "api_integration"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    DOCUMENTATION = "documentation"
    CODE_REVIEW = "code_review"


class TaskComplexityLevel(Enum):
    """Task complexity levels for team formation."""
    SIMPLE = "simple"           # Single agent, <2 hours
    MODERATE = "moderate"       # 2-3 agents, <8 hours  
    COMPLEX = "complex"         # 3-5 agents, 1-3 days
    ADVANCED = "advanced"       # 5-8 agents, 1-2 weeks
    ENTERPRISE = "enterprise"   # 8+ agents, 2+ weeks


class CollaborationPattern(Enum):
    """Collaboration patterns for different task types."""
    SEQUENTIAL = "sequential"       # Tasks executed in sequence
    PARALLEL = "parallel"          # Tasks executed in parallel
    HIERARCHICAL = "hierarchical"  # Lead agent with supporting agents
    PEER_TO_PEER = "peer_to_peer"  # Equal collaboration
    SPECIALIST_TEAMS = "specialist_teams"  # Specialized sub-teams
    HYBRID = "hybrid"              # Dynamic combination


class ConsensusType(Enum):
    """Types of consensus mechanisms."""
    MAJORITY_VOTE = "majority_vote"         # Simple majority decision
    WEIGHTED_EXPERTISE = "weighted_expertise"  # Expertise-weighted voting
    UNANIMOUS = "unanimous"                 # All agents must agree
    LEAD_DECISION = "lead_decision"         # Lead agent decides
    HYBRID_CONSENSUS = "hybrid_consensus"   # Context-dependent method


@dataclass
class AgentExpertise:
    """Agent expertise profile for team formation."""
    agent_id: uuid.UUID
    agent_role: AgentRole
    capabilities: Set[AgentCapability]
    expertise_scores: Dict[AgentCapability, float]  # 0.0-1.0
    collaboration_history: Dict[uuid.UUID, float]  # Agent ID -> success rate
    availability_score: float  # Current availability 0.0-1.0
    workload_capacity: float  # Current capacity 0.0-1.0
    preferred_team_size: int
    communication_style: str
    last_updated: datetime


@dataclass
class ComplexTask:
    """Complex task requiring multi-agent collaboration."""
    task_id: uuid.UUID
    title: str
    description: str
    task_type: str
    complexity_level: TaskComplexityLevel
    required_capabilities: Set[AgentCapability]
    estimated_duration: timedelta
    priority: TaskPriority
    dependencies: List[uuid.UUID] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    quality_requirements: Dict[str, Any] = field(default_factory=dict)
    context_requirements: List[str] = field(default_factory=list)
    collaboration_pattern: Optional[CollaborationPattern] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SubTask:
    """Individual sub-task within a complex task."""
    subtask_id: uuid.UUID
    parent_task_id: uuid.UUID
    title: str
    description: str
    required_capability: AgentCapability
    estimated_duration: timedelta
    dependencies: List[uuid.UUID] = field(default_factory=list)
    assigned_agent_id: Optional[uuid.UUID] = None
    status: str = "pending"
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentTeam:
    """Dynamic agent team for collaborative task execution."""
    team_id: uuid.UUID
    task_id: uuid.UUID
    team_name: str
    lead_agent_id: uuid.UUID
    agent_members: List[uuid.UUID]
    agent_roles: Dict[uuid.UUID, AgentRole]
    agent_capabilities: Dict[uuid.UUID, Set[AgentCapability]]
    collaboration_pattern: CollaborationPattern
    communication_channels: Dict[str, str]
    team_formation_confidence: float
    expected_performance_metrics: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "forming"


@dataclass
class CollaborativeDecision:
    """Decision requiring consensus from team members."""
    decision_id: uuid.UUID
    task_id: uuid.UUID
    team_id: uuid.UUID
    decision_topic: str
    decision_context: str
    options: List[Dict[str, Any]]
    consensus_type: ConsensusType
    required_participants: List[uuid.UUID]
    votes: Dict[uuid.UUID, Dict[str, Any]] = field(default_factory=dict)
    final_decision: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    decided_at: Optional[datetime] = None


@dataclass
class TeamPerformanceMetrics:
    """Real-time team performance metrics."""
    team_id: uuid.UUID
    task_id: uuid.UUID
    collaboration_effectiveness: float
    communication_quality: float
    resource_utilization: float
    progress_velocity: float
    consensus_efficiency: float
    knowledge_sharing_score: float
    overall_performance: float
    bottlenecks: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExecutionPlan:
    """Execution plan for parallel task processing."""
    plan_id: uuid.UUID
    task_id: uuid.UUID
    team_id: uuid.UUID
    subtasks: List[SubTask]
    dependency_graph: Dict[uuid.UUID, List[uuid.UUID]]
    execution_phases: List[List[uuid.UUID]]  # Parallel execution batches
    estimated_total_duration: timedelta
    resource_allocation: Dict[uuid.UUID, Dict[str, Any]]
    risk_assessment: Dict[str, float]
    optimization_strategy: str
    created_at: datetime = field(default_factory=datetime.utcnow)


class DynamicAgentCollaboration:
    """
    Dynamic Agent Collaboration System for Epic 2 Phase 2.
    
    Enables multi-agent teams to collaboratively execute complex tasks
    with dynamic team formation, real-time coordination, and consensus mechanisms.
    
    Key Capabilities:
    - Dynamic team formation based on task requirements and agent capabilities
    - Real-time collaborative task execution with coordination protocols
    - Intelligent consensus mechanisms for complex decision-making
    - Expertise routing and agent specialization systems
    - Performance monitoring and team optimization
    - Integration with Phase 1 context intelligence
    """
    
    def __init__(self, intelligent_orchestrator: Optional[IntelligentOrchestrator] = None):
        """Initialize the Dynamic Agent Collaboration system."""
        self.intelligent_orchestrator = intelligent_orchestrator
        self.context_engine: Optional[AdvancedContextEngine] = None
        self.semantic_memory: Optional[SemanticMemorySystem] = None
        
        # Agent expertise tracking
        self.agent_expertise: Dict[uuid.UUID, AgentExpertise] = {}
        
        # Active teams and tasks
        self.active_teams: Dict[uuid.UUID, AgentTeam] = {}
        self.complex_tasks: Dict[uuid.UUID, ComplexTask] = {}
        self.execution_plans: Dict[uuid.UUID, ExecutionPlan] = {}
        self.active_decisions: Dict[uuid.UUID, CollaborativeDecision] = {}
        
        # Performance tracking
        self.team_performance: Dict[uuid.UUID, TeamPerformanceMetrics] = {}
        self.collaboration_history: Dict[uuid.UUID, List[Dict[str, Any]]] = defaultdict(list)
        
        # System metrics
        self._system_metrics = {
            'total_collaborative_tasks': 0,
            'avg_team_formation_time_ms': 0.0,
            'avg_consensus_time_ms': 0.0,
            'collaboration_success_rate': 0.0,
            'resource_utilization_rate': 0.0,
            'team_performance_improvement': 0.0
        }
        
        logger.info("Dynamic Agent Collaboration system initialized")
    
    async def initialize(self) -> None:
        """Initialize the collaboration system with dependencies."""
        try:
            # Initialize intelligent orchestrator if not provided
            if not self.intelligent_orchestrator:
                self.intelligent_orchestrator = await get_intelligent_orchestrator()
            
            # Initialize Phase 1 components
            self.context_engine = await get_context_engine()
            self.semantic_memory = await get_semantic_memory()
            
            # Initialize agent expertise profiles
            await self._initialize_agent_expertise()
            
            logger.info("✅ Dynamic Agent Collaboration initialization complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Dynamic Agent Collaboration: {e}")
            raise
    
    async def form_optimal_team(
        self,
        task: ComplexTask,
        available_agents: List[Dict[str, Any]]
    ) -> AgentTeam:
        """
        Form optimal team for complex task execution.
        
        Args:
            task: Complex task requiring team collaboration
            available_agents: List of available agents
            
        Returns:
            Optimally formed agent team
        """
        start_time = time.perf_counter()
        
        try:
            if not self.context_engine:
                await self.initialize()
            
            logger.info(f"🔧 Forming optimal team for task: {task.title}")
            
            # Step 1: Analyze task requirements and determine team size
            optimal_team_size = await self._determine_optimal_team_size(task)
            
            # Step 2: Filter agents by capability requirements
            capable_agents = await self._filter_agents_by_capabilities(
                available_agents, task.required_capabilities
            )
            
            if len(capable_agents) < optimal_team_size:
                logger.warning(
                    f"Insufficient capable agents: need {optimal_team_size}, "
                    f"have {len(capable_agents)}"
                )
                optimal_team_size = len(capable_agents)
            
            # Step 3: Score agents for team composition
            agent_scores = await self._score_agents_for_team(task, capable_agents)
            
            # Step 4: Select optimal team combination
            selected_agents = await self._select_optimal_team_combination(
                task, agent_scores, optimal_team_size
            )
            
            # Step 5: Determine team structure and roles
            team_structure = await self._determine_team_structure(task, selected_agents)
            
            # Step 6: Select team leader
            lead_agent_id = await self._select_team_leader(task, selected_agents)
            
            # Step 7: Determine collaboration pattern
            collaboration_pattern = self._determine_collaboration_pattern(task)
            
            # Step 8: Create team
            team = AgentTeam(
                team_id=uuid.uuid4(),
                task_id=task.task_id,
                team_name=f"Team-{task.title[:20]}",
                lead_agent_id=lead_agent_id,
                agent_members=[agent['id'] for agent in selected_agents],
                agent_roles={
                    uuid.UUID(agent['id']): AgentRole(agent['role'])
                    for agent in selected_agents
                },
                agent_capabilities={
                    uuid.UUID(agent['id']): self.agent_expertise[uuid.UUID(agent['id'])].capabilities
                    for agent in selected_agents if uuid.UUID(agent['id']) in self.agent_expertise
                },
                collaboration_pattern=collaboration_pattern,
                communication_channels={
                    'primary': f"team-{task.task_id}",
                    'urgent': f"urgent-{task.task_id}",
                    'coordination': f"coord-{task.task_id}"
                },
                team_formation_confidence=await self._calculate_team_confidence(
                    task, selected_agents
                )
            )
            
            # Step 9: Store team and update metrics
            self.active_teams[team.team_id] = team
            
            formation_time = (time.perf_counter() - start_time) * 1000
            self._update_team_formation_metrics(formation_time)
            
            logger.info(
                f"✅ Optimal team formed: {team.team_id} with {len(selected_agents)} agents "
                f"(confidence: {team.team_formation_confidence:.2f}, time: {formation_time:.1f}ms)"
            )
            
            return team
            
        except Exception as e:
            logger.error(f"Failed to form optimal team: {e}")
            raise
    
    async def coordinate_collaborative_execution(
        self,
        team: AgentTeam,
        task: ComplexTask
    ) -> Dict[str, Any]:
        """
        Coordinate collaborative task execution across team members.
        
        Args:
            team: Agent team for task execution
            task: Complex task to execute collaboratively
            
        Returns:
            Collaborative execution result with performance metrics
        """
        try:
            logger.info(f"🎯 Starting collaborative execution: {task.title}")
            
            # Step 1: Create execution plan with task decomposition
            execution_plan = await self._create_execution_plan(team, task)
            self.execution_plans[task.task_id] = execution_plan
            
            # Step 2: Initialize team performance monitoring
            await self._initialize_team_performance_monitoring(team, task)
            
            # Step 3: Coordinate task execution across phases
            execution_results = []
            
            for phase_index, phase_subtasks in enumerate(execution_plan.execution_phases):
                logger.info(f"🔄 Executing phase {phase_index + 1}/{len(execution_plan.execution_phases)}")
                
                # Execute subtasks in parallel within phase
                phase_results = await self._execute_parallel_subtasks(
                    team, phase_subtasks, execution_plan
                )
                execution_results.extend(phase_results)
                
                # Update team performance metrics
                await self._update_team_performance_metrics(team, task, phase_results)
                
                # Check for bottlenecks and optimization opportunities
                await self._check_execution_bottlenecks(team, task, execution_plan)
            
            # Step 4: Aggregate final results
            final_result = await self._aggregate_collaborative_results(
                team, task, execution_results
            )
            
            # Step 5: Update system metrics
            self._update_collaboration_metrics(team, task, final_result)
            
            logger.info(f"✅ Collaborative execution completed: {task.title}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Collaborative execution failed: {e}")
            raise
    
    async def implement_consensus_mechanism(
        self,
        team: AgentTeam,
        decisions: List[CollaborativeDecision]
    ) -> Dict[uuid.UUID, Dict[str, Any]]:
        """
        Implement consensus mechanism for collaborative decision-making.
        
        Args:
            team: Agent team making decisions
            decisions: List of decisions requiring consensus
            
        Returns:
            Dictionary of decision results with confidence scores
        """
        try:
            logger.info(f"🗳️ Implementing consensus for {len(decisions)} decisions")
            
            consensus_results = {}
            
            for decision in decisions:
                start_time = time.perf_counter()
                
                # Step 1: Gather votes from team members
                votes = await self._gather_team_votes(team, decision)
                decision.votes = votes
                
                # Step 2: Apply consensus mechanism
                final_decision = await self._apply_consensus_mechanism(decision, votes)
                decision.final_decision = final_decision
                decision.decided_at = datetime.utcnow()
                
                # Step 3: Calculate confidence score
                confidence = await self._calculate_decision_confidence(decision, votes)
                decision.confidence_score = confidence
                
                # Step 4: Store decision result
                consensus_results[decision.decision_id] = {
                    'decision': final_decision,
                    'confidence': confidence,
                    'consensus_time_ms': (time.perf_counter() - start_time) * 1000,
                    'participation_rate': len(votes) / len(decision.required_participants),
                    'consensus_type': decision.consensus_type.value
                }
                
                # Update active decisions
                self.active_decisions[decision.decision_id] = decision
                
                logger.info(
                    f"✅ Consensus reached: {decision.decision_topic} "
                    f"(confidence: {confidence:.2f})"
                )
            
            # Update consensus metrics
            avg_consensus_time = sum(
                result['consensus_time_ms'] for result in consensus_results.values()
            ) / len(consensus_results)
            self._update_consensus_metrics(avg_consensus_time)
            
            return consensus_results
            
        except Exception as e:
            logger.error(f"Consensus mechanism failed: {e}")
            raise
    
    async def route_by_expertise(
        self,
        subtask: SubTask,
        team: AgentTeam
    ) -> uuid.UUID:
        """
        Route subtask to optimal team member based on expertise.
        
        Args:
            subtask: Subtask to route
            team: Agent team
            
        Returns:
            Agent ID of optimal team member
        """
        try:
            # Step 1: Score team members for subtask
            expertise_scores = {}
            
            for agent_id in team.agent_members:
                if agent_id in self.agent_expertise:
                    expertise = self.agent_expertise[agent_id]
                    
                    # Calculate expertise score for required capability
                    capability_score = expertise.expertise_scores.get(
                        subtask.required_capability, 0.0
                    )
                    
                    # Factor in availability and workload
                    availability_factor = expertise.availability_score
                    workload_factor = expertise.workload_capacity
                    
                    # Calculate total score
                    total_score = (
                        capability_score * 0.6 +
                        availability_factor * 0.25 +
                        workload_factor * 0.15
                    )
                    
                    expertise_scores[agent_id] = total_score
            
            # Step 2: Select agent with highest expertise score
            if not expertise_scores:
                # Fallback to team leader
                return team.lead_agent_id
            
            optimal_agent = max(expertise_scores.items(), key=lambda x: x[1])
            
            logger.info(
                f"🎯 Subtask routed by expertise: {subtask.title} → "
                f"agent {optimal_agent[0]} (score: {optimal_agent[1]:.2f})"
            )
            
            return optimal_agent[0]
            
        except Exception as e:
            logger.error(f"Expertise routing failed: {e}")
            # Fallback to team leader
            return team.lead_agent_id
    
    async def monitor_team_performance(
        self,
        team: AgentTeam
    ) -> TeamPerformanceMetrics:
        """
        Monitor real-time team performance with optimization insights.
        
        Args:
            team: Agent team to monitor
            
        Returns:
            Real-time team performance metrics
        """
        try:
            if team.team_id not in self.team_performance:
                await self._initialize_team_performance_monitoring(
                    team, self.complex_tasks.get(team.task_id)
                )
            
            # Get current performance metrics
            performance = self.team_performance[team.team_id]
            
            # Update real-time metrics
            await self._update_real_time_performance_metrics(team, performance)
            
            # Identify bottlenecks and improvement opportunities
            await self._identify_performance_bottlenecks(team, performance)
            
            # Generate improvement suggestions
            await self._generate_performance_improvements(team, performance)
            
            performance.last_updated = datetime.utcnow()
            
            logger.info(
                f"📊 Team performance monitored: {team.team_name} "
                f"(overall: {performance.overall_performance:.2f})"
            )
            
            return performance
            
        except Exception as e:
            logger.error(f"Team performance monitoring failed: {e}")
            raise
    
    # Private helper methods for internal functionality
    
    async def _initialize_agent_expertise(self) -> None:
        """Initialize agent expertise profiles."""
        try:
            if not self.intelligent_orchestrator:
                return
            
            agents = await self.intelligent_orchestrator.base_orchestrator.list_agents()
            
            for agent in agents:
                agent_id = uuid.UUID(agent['id'])
                agent_role = AgentRole(agent['role'])
                
                # Map agent role to capabilities
                capabilities = self._map_role_to_capabilities(agent_role)
                
                # Initialize expertise scores
                expertise_scores = {cap: 0.7 for cap in capabilities}  # Start with moderate scores
                
                # Create expertise profile
                expertise = AgentExpertise(
                    agent_id=agent_id,
                    agent_role=agent_role,
                    capabilities=capabilities,
                    expertise_scores=expertise_scores,
                    collaboration_history={},
                    availability_score=0.8,  # Start optimistic
                    workload_capacity=0.9,   # Start with high capacity
                    preferred_team_size=4,
                    communication_style="professional",
                    last_updated=datetime.utcnow()
                )
                
                self.agent_expertise[agent_id] = expertise
            
            logger.info(f"Initialized expertise profiles for {len(self.agent_expertise)} agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize agent expertise: {e}")
    
    def _map_role_to_capabilities(self, role: AgentRole) -> Set[AgentCapability]:
        """Map agent role to capabilities."""
        role_mapping = {
            AgentRole.FRONTEND_DEVELOPER: {
                AgentCapability.FRONTEND_DEVELOPMENT,
                AgentCapability.API_INTEGRATION,
                AgentCapability.TESTING_QA,
                AgentCapability.CODE_REVIEW
            },
            AgentRole.BACKEND_DEVELOPER: {
                AgentCapability.BACKEND_DEVELOPMENT,
                AgentCapability.API_INTEGRATION,
                AgentCapability.DATABASE_DESIGN,
                AgentCapability.PERFORMANCE_OPTIMIZATION,
                AgentCapability.CODE_REVIEW
            },
            AgentRole.DEVOPS_ENGINEER: {
                AgentCapability.DEVOPS_INFRASTRUCTURE,
                AgentCapability.SECURITY_ANALYSIS,
                AgentCapability.PERFORMANCE_OPTIMIZATION,
                AgentCapability.TESTING_QA
            },
            AgentRole.QA_ENGINEER: {
                AgentCapability.TESTING_QA,
                AgentCapability.CODE_REVIEW,
                AgentCapability.PERFORMANCE_OPTIMIZATION
            },
            AgentRole.META_AGENT: {
                AgentCapability.PROJECT_MANAGEMENT,
                AgentCapability.DOCUMENTATION,
                AgentCapability.CODE_REVIEW,
                AgentCapability.PERFORMANCE_OPTIMIZATION
            }
        }
        
        return role_mapping.get(role, {AgentCapability.CODE_REVIEW})
    
    async def _determine_optimal_team_size(self, task: ComplexTask) -> int:
        """Determine optimal team size based on task complexity."""
        complexity_mapping = {
            TaskComplexityLevel.SIMPLE: 1,
            TaskComplexityLevel.MODERATE: 3,
            TaskComplexityLevel.COMPLEX: 5,
            TaskComplexityLevel.ADVANCED: 7,
            TaskComplexityLevel.ENTERPRISE: 10
        }
        
        base_size = complexity_mapping.get(task.complexity_level, 3)
        
        # Adjust based on required capabilities
        capability_factor = min(len(task.required_capabilities), 8)
        
        # Adjust based on estimated duration
        duration_hours = task.estimated_duration.total_seconds() / 3600
        duration_factor = 1 + (duration_hours - 8) / 40  # Scale with duration
        duration_factor = max(0.5, min(2.0, duration_factor))
        
        optimal_size = int(base_size * duration_factor)
        optimal_size = max(1, min(optimal_size, capability_factor))
        
        return optimal_size
    
    async def _filter_agents_by_capabilities(
        self,
        available_agents: List[Dict[str, Any]],
        required_capabilities: Set[AgentCapability]
    ) -> List[Dict[str, Any]]:
        """Filter agents by required capabilities."""
        capable_agents = []
        
        for agent in available_agents:
            agent_id = uuid.UUID(agent['id'])
            
            if agent_id in self.agent_expertise:
                agent_caps = self.agent_expertise[agent_id].capabilities
                
                # Check if agent has any required capabilities
                if agent_caps.intersection(required_capabilities):
                    capable_agents.append(agent)
        
        return capable_agents
    
    async def _score_agents_for_team(
        self,
        task: ComplexTask,
        capable_agents: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Score agents for team composition."""
        agent_scores = []
        
        for agent in capable_agents:
            agent_id = uuid.UUID(agent['id'])
            
            if agent_id not in self.agent_expertise:
                continue
                
            expertise = self.agent_expertise[agent_id]
            
            # Calculate capability match score
            capability_scores = [
                expertise.expertise_scores.get(cap, 0.0)
                for cap in task.required_capabilities
                if cap in expertise.capabilities
            ]
            capability_score = sum(capability_scores) / max(len(capability_scores), 1)
            
            # Factor in availability and workload
            availability_score = expertise.availability_score
            workload_score = expertise.workload_capacity
            
            # Factor in collaboration history
            collaboration_score = sum(expertise.collaboration_history.values()) / max(
                len(expertise.collaboration_history), 1
            ) if expertise.collaboration_history else 0.5
            
            # Calculate total score
            total_score = (
                capability_score * 0.4 +
                availability_score * 0.25 +
                workload_score * 0.2 +
                collaboration_score * 0.15
            )
            
            agent_scores.append((agent, total_score))
        
        # Sort by score descending
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        return agent_scores
    
    async def _select_optimal_team_combination(
        self,
        task: ComplexTask,
        agent_scores: List[Tuple[Dict[str, Any], float]],
        team_size: int
    ) -> List[Dict[str, Any]]:
        """Select optimal team combination using capability coverage."""
        selected_agents = []
        covered_capabilities = set()
        
        # First pass: Select agents that cover required capabilities
        for agent, score in agent_scores:
            if len(selected_agents) >= team_size:
                break
                
            agent_id = uuid.UUID(agent['id'])
            if agent_id in self.agent_expertise:
                agent_caps = self.agent_expertise[agent_id].capabilities
                new_coverage = agent_caps - covered_capabilities
                
                # Select if agent adds new capability coverage or has high score
                if new_coverage or (score > 0.8 and len(selected_agents) < team_size):
                    selected_agents.append(agent)
                    covered_capabilities.update(agent_caps)
        
        # Second pass: Fill remaining slots with highest-scoring agents
        if len(selected_agents) < team_size:
            remaining_agents = [
                agent for agent, _ in agent_scores
                if agent not in selected_agents
            ]
            
            needed_count = team_size - len(selected_agents)
            selected_agents.extend(remaining_agents[:needed_count])
        
        return selected_agents
    
    async def _determine_team_structure(
        self,
        task: ComplexTask,
        selected_agents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Determine optimal team structure and roles."""
        # Simplified team structure based on agent roles
        team_structure = {
            'hierarchy': 'flat',  # Default to flat structure
            'communication_pattern': 'hub-and-spoke',  # Lead-centric
            'decision_authority': 'consensus',
            'specialization_level': 'moderate'
        }
        
        # Adjust based on team size and complexity
        if len(selected_agents) > 6:
            team_structure['hierarchy'] = 'hierarchical'
            team_structure['communication_pattern'] = 'mixed'
        
        if task.complexity_level in [TaskComplexityLevel.ADVANCED, TaskComplexityLevel.ENTERPRISE]:
            team_structure['specialization_level'] = 'high'
            team_structure['decision_authority'] = 'lead_with_input'
        
        return team_structure
    
    async def _select_team_leader(
        self,
        task: ComplexTask,
        selected_agents: List[Dict[str, Any]]
    ) -> uuid.UUID:
        """Select optimal team leader based on task and agent characteristics."""
        leadership_scores = {}
        
        for agent in selected_agents:
            agent_id = uuid.UUID(agent['id'])
            agent_role = AgentRole(agent['role'])
            
            # Base leadership score by role
            role_leadership = {
                AgentRole.META_AGENT: 1.0,
                AgentRole.BACKEND_DEVELOPER: 0.8,
                AgentRole.DEVOPS_ENGINEER: 0.7,
                AgentRole.FRONTEND_DEVELOPER: 0.6,
                AgentRole.QA_ENGINEER: 0.6
            }.get(agent_role, 0.5)
            
            # Factor in expertise
            if agent_id in self.agent_expertise:
                expertise = self.agent_expertise[agent_id]
                collaboration_score = sum(expertise.collaboration_history.values()) / max(
                    len(expertise.collaboration_history), 1
                ) if expertise.collaboration_history else 0.5
                
                total_score = role_leadership * 0.6 + collaboration_score * 0.4
                leadership_scores[agent_id] = total_score
        
        # Select agent with highest leadership score
        if leadership_scores:
            return max(leadership_scores.items(), key=lambda x: x[1])[0]
        else:
            # Fallback to first agent
            return uuid.UUID(selected_agents[0]['id'])
    
    def _determine_collaboration_pattern(self, task: ComplexTask) -> CollaborationPattern:
        """Determine optimal collaboration pattern for task."""
        complexity_patterns = {
            TaskComplexityLevel.SIMPLE: CollaborationPattern.SEQUENTIAL,
            TaskComplexityLevel.MODERATE: CollaborationPattern.PARALLEL,
            TaskComplexityLevel.COMPLEX: CollaborationPattern.HIERARCHICAL,
            TaskComplexityLevel.ADVANCED: CollaborationPattern.SPECIALIST_TEAMS,
            TaskComplexityLevel.ENTERPRISE: CollaborationPattern.HYBRID
        }
        
        return complexity_patterns.get(task.complexity_level, CollaborationPattern.PEER_TO_PEER)
    
    async def _calculate_team_confidence(
        self,
        task: ComplexTask,
        selected_agents: List[Dict[str, Any]]
    ) -> float:
        """Calculate team formation confidence score."""
        # Factor 1: Capability coverage
        required_caps = task.required_capabilities
        covered_caps = set()
        
        for agent in selected_agents:
            agent_id = uuid.UUID(agent['id'])
            if agent_id in self.agent_expertise:
                covered_caps.update(self.agent_expertise[agent_id].capabilities)
        
        coverage_score = len(covered_caps.intersection(required_caps)) / max(len(required_caps), 1)
        
        # Factor 2: Average agent expertise
        expertise_scores = []
        for agent in selected_agents:
            agent_id = uuid.UUID(agent['id'])
            if agent_id in self.agent_expertise:
                agent_expertise = self.agent_expertise[agent_id]
                relevant_scores = [
                    agent_expertise.expertise_scores.get(cap, 0.0)
                    for cap in required_caps
                    if cap in agent_expertise.capabilities
                ]
                if relevant_scores:
                    expertise_scores.append(sum(relevant_scores) / len(relevant_scores))
        
        avg_expertise = sum(expertise_scores) / max(len(expertise_scores), 1)
        
        # Factor 3: Team size appropriateness
        optimal_size = await self._determine_optimal_team_size(task)
        size_score = 1.0 - abs(len(selected_agents) - optimal_size) / max(optimal_size, 1)
        size_score = max(0.0, size_score)
        
        # Calculate overall confidence
        confidence = (
            coverage_score * 0.4 +
            avg_expertise * 0.4 +
            size_score * 0.2
        )
        
        return min(1.0, confidence)
    
    def _update_team_formation_metrics(self, formation_time_ms: float) -> None:
        """Update team formation performance metrics."""
        current_avg = self._system_metrics['avg_team_formation_time_ms']
        total_tasks = self._system_metrics['total_collaborative_tasks']
        
        if total_tasks == 0:
            new_avg = formation_time_ms
        else:
            new_avg = ((current_avg * total_tasks) + formation_time_ms) / (total_tasks + 1)
        
        self._system_metrics['avg_team_formation_time_ms'] = new_avg
        self._system_metrics['total_collaborative_tasks'] += 1
    
    async def _create_execution_plan(
        self,
        team: AgentTeam,
        task: ComplexTask
    ) -> ExecutionPlan:
        """Create detailed execution plan with task decomposition."""
        # This would implement intelligent task decomposition
        # For now, creating a simplified plan
        
        subtasks = []
        
        # Create subtasks based on required capabilities
        for i, capability in enumerate(task.required_capabilities):
            subtask = SubTask(
                subtask_id=uuid.uuid4(),
                parent_task_id=task.task_id,
                title=f"{capability.value.replace('_', ' ').title()} Task",
                description=f"Handle {capability.value} aspects of {task.title}",
                required_capability=capability,
                estimated_duration=task.estimated_duration // len(task.required_capabilities)
            )
            subtasks.append(subtask)
        
        # Simple dependency graph (sequential for now)
        dependency_graph = {}
        for i, subtask in enumerate(subtasks):
            if i > 0:
                dependency_graph[subtask.subtask_id] = [subtasks[i-1].subtask_id]
            else:
                dependency_graph[subtask.subtask_id] = []
        
        # Create execution phases (simplified)
        execution_phases = [[subtask.subtask_id] for subtask in subtasks]
        
        plan = ExecutionPlan(
            plan_id=uuid.uuid4(),
            task_id=task.task_id,
            team_id=team.team_id,
            subtasks=subtasks,
            dependency_graph=dependency_graph,
            execution_phases=execution_phases,
            estimated_total_duration=task.estimated_duration,
            resource_allocation={},
            risk_assessment={'complexity': 0.5, 'dependency': 0.3},
            optimization_strategy='sequential_with_monitoring'
        )
        
        return plan
    
    async def _initialize_team_performance_monitoring(
        self,
        team: AgentTeam,
        task: Optional[ComplexTask]
    ) -> None:
        """Initialize team performance monitoring."""
        metrics = TeamPerformanceMetrics(
            team_id=team.team_id,
            task_id=team.task_id,
            collaboration_effectiveness=0.8,  # Start optimistic
            communication_quality=0.8,
            resource_utilization=0.7,
            progress_velocity=0.6,
            consensus_efficiency=0.8,
            knowledge_sharing_score=0.7,
            overall_performance=0.75
        )
        
        self.team_performance[team.team_id] = metrics
    
    async def _execute_parallel_subtasks(
        self,
        team: AgentTeam,
        subtask_ids: List[uuid.UUID],
        execution_plan: ExecutionPlan
    ) -> List[Dict[str, Any]]:
        """Execute subtasks in parallel (simplified implementation)."""
        results = []
        
        for subtask_id in subtask_ids:
            # Find subtask
            subtask = None
            for st in execution_plan.subtasks:
                if st.subtask_id == subtask_id:
                    subtask = st
                    break
            
            if subtask:
                # Route to optimal agent
                assigned_agent = await self.route_by_expertise(subtask, team)
                subtask.assigned_agent_id = assigned_agent
                
                # Simulate execution (in real implementation, would delegate to agent)
                result = {
                    'subtask_id': subtask_id,
                    'assigned_agent': assigned_agent,
                    'status': 'completed',
                    'execution_time_ms': 1000,  # Simulated
                    'quality_score': 0.85,
                    'success': True
                }
                results.append(result)
        
        return results
    
    async def _update_team_performance_metrics(
        self,
        team: AgentTeam,
        task: ComplexTask,
        phase_results: List[Dict[str, Any]]
    ) -> None:
        """Update team performance metrics based on execution results."""
        if team.team_id not in self.team_performance:
            return
        
        metrics = self.team_performance[team.team_id]
        
        # Calculate success rate for this phase
        successful_results = [r for r in phase_results if r.get('success', False)]
        phase_success_rate = len(successful_results) / max(len(phase_results), 1)
        
        # Update collaboration effectiveness
        metrics.collaboration_effectiveness = (
            metrics.collaboration_effectiveness * 0.7 + phase_success_rate * 0.3
        )
        
        # Update overall performance
        metrics.overall_performance = (
            metrics.collaboration_effectiveness * 0.3 +
            metrics.communication_quality * 0.2 +
            metrics.resource_utilization * 0.2 +
            metrics.progress_velocity * 0.2 +
            metrics.knowledge_sharing_score * 0.1
        )
        
        metrics.last_updated = datetime.utcnow()
    
    async def _check_execution_bottlenecks(
        self,
        team: AgentTeam,
        task: ComplexTask,
        execution_plan: ExecutionPlan
    ) -> None:
        """Check for execution bottlenecks and optimization opportunities."""
        # Simplified bottleneck detection
        # In real implementation would analyze agent workload, communication delays, etc.
        pass
    
    async def _aggregate_collaborative_results(
        self,
        team: AgentTeam,
        task: ComplexTask,
        execution_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate results from collaborative task execution."""
        successful_results = [r for r in execution_results if r.get('success', False)]
        success_rate = len(successful_results) / max(len(execution_results), 1)
        
        avg_quality = sum(r.get('quality_score', 0.0) for r in successful_results) / max(
            len(successful_results), 1
        )
        
        total_execution_time = sum(r.get('execution_time_ms', 0) for r in execution_results)
        
        return {
            'task_id': task.task_id,
            'team_id': team.team_id,
            'success_rate': success_rate,
            'average_quality_score': avg_quality,
            'total_execution_time_ms': total_execution_time,
            'subtasks_completed': len(successful_results),
            'collaboration_effectiveness': self.team_performance.get(
                team.team_id, TeamPerformanceMetrics(
                    team.team_id, task.task_id, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                )
            ).collaboration_effectiveness,
            'completed_at': datetime.utcnow()
        }
    
    def _update_collaboration_metrics(
        self,
        team: AgentTeam,
        task: ComplexTask,
        final_result: Dict[str, Any]
    ) -> None:
        """Update system collaboration metrics."""
        success_rate = final_result.get('success_rate', 0.0)
        
        # Update collaboration success rate
        current_rate = self._system_metrics['collaboration_success_rate']
        total_tasks = self._system_metrics['total_collaborative_tasks']
        
        new_rate = ((current_rate * (total_tasks - 1)) + success_rate) / total_tasks
        self._system_metrics['collaboration_success_rate'] = new_rate
        
        # Update resource utilization
        team_metrics = self.team_performance.get(team.team_id)
        if team_metrics:
            current_util = self._system_metrics['resource_utilization_rate']
            new_util = ((current_util * (total_tasks - 1)) + team_metrics.resource_utilization) / total_tasks
            self._system_metrics['resource_utilization_rate'] = new_util
    
    async def _gather_team_votes(
        self,
        team: AgentTeam,
        decision: CollaborativeDecision
    ) -> Dict[uuid.UUID, Dict[str, Any]]:
        """Gather votes from team members for consensus decision."""
        # Simplified vote gathering - in real implementation would query agents
        votes = {}
        
        for agent_id in decision.required_participants:
            if agent_id in team.agent_members:
                # Simulate vote based on agent expertise
                vote_option = decision.options[0] if decision.options else {'choice': 'approve'}
                confidence = 0.8  # Simulated confidence
                
                votes[agent_id] = {
                    'option': vote_option,
                    'confidence': confidence,
                    'reasoning': f"Agent {agent_id} expertise-based decision",
                    'timestamp': datetime.utcnow()
                }
        
        return votes
    
    async def _apply_consensus_mechanism(
        self,
        decision: CollaborativeDecision,
        votes: Dict[uuid.UUID, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply consensus mechanism to reach team decision."""
        if not votes:
            return decision.options[0] if decision.options else {'choice': 'default'}
        
        if decision.consensus_type == ConsensusType.MAJORITY_VOTE:
            # Simple majority voting
            option_counts = defaultdict(int)
            for vote in votes.values():
                option_key = str(vote['option'])
                option_counts[option_key] += 1
            
            winner = max(option_counts.items(), key=lambda x: x[1])
            return eval(winner[0]) if winner[0].startswith('{') else {'choice': winner[0]}
        
        elif decision.consensus_type == ConsensusType.WEIGHTED_EXPERTISE:
            # Expertise-weighted voting
            option_weights = defaultdict(float)
            for agent_id, vote in votes.items():
                weight = 1.0  # Default weight
                if agent_id in self.agent_expertise:
                    # Use agent expertise as weight
                    expertise = self.agent_expertise[agent_id]
                    weight = sum(expertise.expertise_scores.values()) / max(
                        len(expertise.expertise_scores), 1
                    )
                
                option_key = str(vote['option'])
                option_weights[option_key] += weight * vote.get('confidence', 1.0)
            
            winner = max(option_weights.items(), key=lambda x: x[1])
            return eval(winner[0]) if winner[0].startswith('{') else {'choice': winner[0]}
        
        else:
            # Default: return first option with highest confidence
            best_vote = max(votes.values(), key=lambda v: v.get('confidence', 0.0))
            return best_vote['option']
    
    async def _calculate_decision_confidence(
        self,
        decision: CollaborativeDecision,
        votes: Dict[uuid.UUID, Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for consensus decision."""
        if not votes:
            return 0.1
        
        # Calculate average confidence of votes supporting final decision
        final_decision_str = str(decision.final_decision)
        supporting_votes = [
            vote for vote in votes.values()
            if str(vote['option']) == final_decision_str
        ]
        
        if not supporting_votes:
            return 0.3
        
        avg_confidence = sum(vote.get('confidence', 0.0) for vote in supporting_votes) / len(supporting_votes)
        
        # Factor in participation rate
        participation_rate = len(votes) / len(decision.required_participants)
        
        # Factor in consensus strength (how many agreed)
        consensus_strength = len(supporting_votes) / len(votes)
        
        final_confidence = (
            avg_confidence * 0.5 +
            participation_rate * 0.3 +
            consensus_strength * 0.2
        )
        
        return min(1.0, final_confidence)
    
    def _update_consensus_metrics(self, avg_consensus_time_ms: float) -> None:
        """Update consensus mechanism performance metrics."""
        current_avg = self._system_metrics['avg_consensus_time_ms']
        total_decisions = max(self._system_metrics.get('total_decisions', 0), 1)
        
        new_avg = ((current_avg * (total_decisions - 1)) + avg_consensus_time_ms) / total_decisions
        self._system_metrics['avg_consensus_time_ms'] = new_avg
        
        if 'total_decisions' not in self._system_metrics:
            self._system_metrics['total_decisions'] = 0
        self._system_metrics['total_decisions'] += 1
    
    async def _update_real_time_performance_metrics(
        self,
        team: AgentTeam,
        performance: TeamPerformanceMetrics
    ) -> None:
        """Update real-time performance metrics."""
        # Simplified real-time updates
        # In real implementation would query actual agent status, communication metrics, etc.
        
        # Simulate some performance drift
        import random
        performance.communication_quality *= (0.95 + random.random() * 0.1)
        performance.resource_utilization *= (0.95 + random.random() * 0.1)
        performance.progress_velocity *= (0.95 + random.random() * 0.1)
        
        # Ensure values stay in valid range
        performance.communication_quality = max(0.1, min(1.0, performance.communication_quality))
        performance.resource_utilization = max(0.1, min(1.0, performance.resource_utilization))
        performance.progress_velocity = max(0.1, min(1.0, performance.progress_velocity))
        
        # Recalculate overall performance
        performance.overall_performance = (
            performance.collaboration_effectiveness * 0.25 +
            performance.communication_quality * 0.25 +
            performance.resource_utilization * 0.2 +
            performance.progress_velocity * 0.2 +
            performance.knowledge_sharing_score * 0.1
        )
    
    async def _identify_performance_bottlenecks(
        self,
        team: AgentTeam,
        performance: TeamPerformanceMetrics
    ) -> None:
        """Identify performance bottlenecks and issues."""
        bottlenecks = []
        
        if performance.communication_quality < 0.6:
            bottlenecks.append("Poor communication quality affecting coordination")
        
        if performance.resource_utilization < 0.5:
            bottlenecks.append("Low resource utilization - agents may be underutilized")
        
        if performance.progress_velocity < 0.5:
            bottlenecks.append("Slow progress velocity - consider task rebalancing")
        
        if performance.consensus_efficiency < 0.6:
            bottlenecks.append("Slow consensus decision-making - consider streamlining")
        
        performance.bottlenecks = bottlenecks
    
    async def _generate_performance_improvements(
        self,
        team: AgentTeam,
        performance: TeamPerformanceMetrics
    ) -> None:
        """Generate performance improvement suggestions."""
        suggestions = []
        
        if performance.communication_quality < 0.7:
            suggestions.append("Establish more frequent check-in meetings")
            suggestions.append("Implement structured communication protocols")
        
        if performance.resource_utilization < 0.7:
            suggestions.append("Redistribute workload to optimize agent capacity")
            suggestions.append("Consider adding more parallel task execution")
        
        if performance.knowledge_sharing_score < 0.7:
            suggestions.append("Implement knowledge sharing sessions")
            suggestions.append("Create shared documentation practices")
        
        if len(team.agent_members) > 6:
            suggestions.append("Consider breaking into smaller specialist sub-teams")
        
        performance.improvement_suggestions = suggestions
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for collaboration system."""
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {
                'intelligent_orchestrator': None,
                'context_engine': None,
                'semantic_memory': None
            },
            'system_metrics': self._system_metrics,
            'active_teams_count': len(self.active_teams),
            'agent_expertise_profiles': len(self.agent_expertise)
        }
        
        try:
            # Check component health
            if self.intelligent_orchestrator:
                health_status['components']['intelligent_orchestrator'] = \
                    await self.intelligent_orchestrator.health_check()
            
            if self.context_engine:
                health_status['components']['context_engine'] = \
                    await self.context_engine.health_check()
            
            if self.semantic_memory:
                health_status['components']['semantic_memory'] = \
                    await self.semantic_memory.health_check()
            
            # Determine overall status
            component_statuses = [
                comp.get('status', 'unknown') if isinstance(comp, dict) else 'unknown'
                for comp in health_status['components'].values()
                if comp is not None
            ]
            
            if all(status == 'healthy' for status in component_statuses):
                health_status['status'] = 'healthy'
            elif any(status == 'healthy' for status in component_statuses):
                health_status['status'] = 'degraded'
            else:
                health_status['status'] = 'unhealthy'
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
        
        return health_status
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics for collaboration system."""
        return {
            'system_metrics': self._system_metrics,
            'active_teams': {
                str(team_id): {
                    'team_name': team.team_name,
                    'member_count': len(team.agent_members),
                    'collaboration_pattern': team.collaboration_pattern.value,
                    'formation_confidence': team.team_formation_confidence,
                    'status': team.status
                }
                for team_id, team in self.active_teams.items()
            },
            'team_performance': {
                str(team_id): {
                    'overall_performance': metrics.overall_performance,
                    'collaboration_effectiveness': metrics.collaboration_effectiveness,
                    'resource_utilization': metrics.resource_utilization,
                    'bottlenecks_count': len(metrics.bottlenecks),
                    'improvement_suggestions_count': len(metrics.improvement_suggestions)
                }
                for team_id, metrics in self.team_performance.items()
            },
            'agent_expertise_summary': {
                'total_agents': len(self.agent_expertise),
                'avg_availability': sum(
                    exp.availability_score for exp in self.agent_expertise.values()
                ) / max(len(self.agent_expertise), 1),
                'avg_workload_capacity': sum(
                    exp.workload_capacity for exp in self.agent_expertise.values()
                ) / max(len(self.agent_expertise), 1)
            }
        }


# Global instance management
_dynamic_agent_collaboration: Optional[DynamicAgentCollaboration] = None


async def get_dynamic_agent_collaboration() -> DynamicAgentCollaboration:
    """Get singleton dynamic agent collaboration instance."""
    global _dynamic_agent_collaboration
    
    if _dynamic_agent_collaboration is None:
        _dynamic_agent_collaboration = DynamicAgentCollaboration()
        await _dynamic_agent_collaboration.initialize()
    
    return _dynamic_agent_collaboration


async def cleanup_dynamic_agent_collaboration() -> None:
    """Cleanup dynamic agent collaboration resources."""
    global _dynamic_agent_collaboration
    
    if _dynamic_agent_collaboration:
        # Cleanup would be implemented here
        _dynamic_agent_collaboration = None