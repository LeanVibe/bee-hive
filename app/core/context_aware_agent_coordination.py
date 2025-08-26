"""
Context-Aware Agent Coordination System - Epic 4 Integration with Epic 1

Advanced agent coordination system that leverages context intelligence for optimal
multi-agent collaboration, decision making, and task orchestration in LeanVibe Agent Hive 2.0.
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import uuid

import structlog
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sklearn.metrics.pairwise import cosine_similarity

# Epic 1 Integration - Use available orchestration modules
# from .production_orchestrator import ProductionOrchestrator
# from .advanced_orchestration_engine import AdvancedOrchestrationEngine
from .agent_manager import AgentManager

# Epic 4 Context Engine Components
# from .unified_context_engine import UnifiedContextEngine, get_unified_context_engine
from .context_reasoning_engine import (
    ContextReasoningEngine, get_context_reasoning_engine, 
    ReasoningType, ReasoningInsight
)
from .intelligent_context_persistence import get_intelligent_context_persistence

# Core imports
from .database import get_async_session
from .redis import get_redis_client
from ..models.agent import Agent, AgentType, AgentStatus
from ..models.context import Context, ContextType

logger = structlog.get_logger()


class CoordinationStrategy(Enum):
    """Agent coordination strategies."""
    HIERARCHICAL = "hierarchical"       # Top-down coordination
    PEER_TO_PEER = "peer_to_peer"      # Collaborative coordination
    CONSENSUS_DRIVEN = "consensus_driven" # Consensus-based decisions
    CONTEXT_OPTIMIZED = "context_optimized" # Context-aware optimization
    ADAPTIVE = "adaptive"               # Adaptive based on situation


class CollaborationMode(Enum):
    """Collaboration modes between agents."""
    INDEPENDENT = "independent"         # Agents work independently
    COOPERATIVE = "cooperative"         # Agents share resources/context
    COLLABORATIVE = "collaborative"     # Agents work together on tasks
    COMPETITIVE = "competitive"         # Agents compete for resources
    MENTORING = "mentoring"            # Experienced agents guide others


@dataclass
class CoordinationContext:
    """Context information for agent coordination."""
    coordination_id: str
    participating_agents: List[str]
    coordination_strategy: CoordinationStrategy
    collaboration_mode: CollaborationMode
    shared_context: Dict[str, Any]
    coordination_goals: List[str]
    success_metrics: Dict[str, Any]
    resource_constraints: Dict[str, Any]
    timeline_constraints: Dict[str, datetime]
    priority_level: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentCapabilityProfile:
    """Profile of agent capabilities and context preferences."""
    agent_id: str
    capabilities: List[str]
    specializations: List[str]
    performance_metrics: Dict[str, float]
    context_preferences: Dict[str, float]
    collaboration_history: Dict[str, Any]
    current_workload: float
    availability_schedule: Dict[str, Any]
    context_sharing_policies: Dict[str, bool]
    learning_preferences: Dict[str, Any]


@dataclass
class CoordinationDecision:
    """Decision made by coordination system."""
    decision_id: str
    decision_type: str
    affected_agents: List[str]
    decision_rationale: str
    confidence_score: float
    expected_outcomes: List[str]
    implementation_steps: List[Dict[str, Any]]
    success_probability: float
    risk_factors: List[str]
    monitoring_metrics: List[str]
    rollback_plan: Optional[Dict[str, Any]] = None


class ContextAwareAgentCoordination:
    """
    Context-Aware Agent Coordination System integrating Epic 1 and Epic 4.
    
    Features:
    - Context-driven agent selection and task assignment
    - Intelligent collaboration orchestration
    - Dynamic workload balancing based on context analysis
    - Cross-agent knowledge sharing optimization
    - Performance-based coordination strategy adaptation
    - Real-time coordination decision support
    """
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        self.db_session = db_session
        self.redis_client = get_redis_client()
        self.logger = logger.bind(component="context_aware_coordination")
        
        # Component integrations (lazy-loaded)
        self._orchestrator = None
        self._agent_manager = None
        self._context_engine = None
        self._reasoning_engine = None
        self._persistence_system = None
        
        # Coordination state
        self.active_coordinations: Dict[str, CoordinationContext] = {}
        self.agent_profiles: Dict[str, AgentCapabilityProfile] = {}
        self.coordination_history: deque = deque(maxlen=1000)
        self.decision_history: deque = deque(maxlen=500)
        
        # Performance tracking
        self.coordination_metrics = defaultdict(lambda: {
            "success_rate": 0.0,
            "average_completion_time": 0.0,
            "agent_satisfaction": 0.0,
            "context_utilization": 0.0
        })
        
        # Configuration
        self.max_coordination_size = 10  # Max agents in single coordination
        self.context_sharing_threshold = 0.7
        self.decision_confidence_threshold = 0.6
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._is_running = False
        
        self.logger.info("🤝 Context-Aware Agent Coordination initialized")
    
    async def initialize(self) -> None:
        """Initialize the coordination system and Epic integrations."""
        if self._is_running:
            return
        
        try:
            self.logger.info("🚀 Initializing Context-Aware Agent Coordination...")
            
            # Initialize Epic 1 integrations
            # self._orchestrator = ProductionOrchestrator()
            self._agent_manager = AgentManager()
            
            # Initialize Epic 4 integrations
            # self._context_engine = await get_unified_context_engine(self.db_session)
            self._reasoning_engine = get_context_reasoning_engine(self.db_session)
            self._persistence_system = await get_intelligent_context_persistence(self.db_session)
            
            # Load agent profiles
            await self._load_agent_profiles()
            
            # Start background coordination tasks
            await self._start_background_tasks()
            
            self._is_running = True
            self.logger.info("✅ Context-Aware Agent Coordination initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize coordination system: {e}")
            raise
    
    async def coordinate_agents_for_task(
        self,
        task_description: str,
        required_capabilities: List[str],
        coordination_strategy: Optional[CoordinationStrategy] = None,
        max_agents: int = 5,
        priority_level: float = 0.5
    ) -> CoordinationContext:
        """
        Coordinate agents for a specific task using context intelligence.
        
        Args:
            task_description: Description of the task
            required_capabilities: Required agent capabilities
            coordination_strategy: Preferred coordination strategy
            max_agents: Maximum number of agents to coordinate
            priority_level: Priority level (0.0-1.0)
            
        Returns:
            CoordinationContext for the task
        """
        start_time = time.time()
        
        try:
            coordination_id = str(uuid.uuid4())
            self.logger.info(f"🎯 Coordinating agents for task: {coordination_id}")
            
            # Analyze task context using reasoning engine
            task_analysis = await self._reasoning_engine.analyze_decision_context(
                context_data={"content": task_description, "capabilities": required_capabilities},
                decision_scope="task_coordination",
                complexity="moderate"
            )
            
            # Select optimal agents based on context analysis
            selected_agents = await self._select_optimal_agents(
                required_capabilities=required_capabilities,
                task_analysis=task_analysis,
                max_agents=max_agents,
                priority_level=priority_level
            )
            
            # Determine coordination strategy
            if coordination_strategy is None:
                coordination_strategy = await self._determine_optimal_coordination_strategy(
                    selected_agents, task_analysis, required_capabilities
                )
            
            # Determine collaboration mode
            collaboration_mode = await self._determine_collaboration_mode(
                selected_agents, task_analysis
            )
            
            # Extract shared context for coordination
            shared_context = await self._extract_coordination_context(
                selected_agents, task_description, task_analysis
            )
            
            # Generate coordination goals
            coordination_goals = await self._generate_coordination_goals(
                task_description, required_capabilities, task_analysis
            )
            
            # Define success metrics
            success_metrics = await self._define_coordination_success_metrics(
                coordination_goals, selected_agents, task_analysis
            )
            
            # Create coordination context
            coordination_context = CoordinationContext(
                coordination_id=coordination_id,
                participating_agents=[agent.id for agent in selected_agents],
                coordination_strategy=coordination_strategy,
                collaboration_mode=collaboration_mode,
                shared_context=shared_context,
                coordination_goals=coordination_goals,
                success_metrics=success_metrics,
                resource_constraints=task_analysis.resource_requirements,
                timeline_constraints={
                    "estimated_completion": datetime.utcnow() + timedelta(
                        hours=float(task_analysis.timeline_estimate.get("hours", 2))
                    )
                },
                priority_level=priority_level
            )
            
            # Store active coordination
            self.active_coordinations[coordination_id] = coordination_context
            
            # Initialize Epic 1 orchestration
            await self._initialize_orchestration(coordination_context, selected_agents)
            
            # Share context among selected agents
            await self._share_context_among_agents(coordination_context, selected_agents)
            
            # Start coordination monitoring
            await self._start_coordination_monitoring(coordination_context)
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                f"✅ Agent coordination established: {len(selected_agents)} agents, "
                f"{coordination_strategy.value} strategy in {processing_time:.2f}s"
            )
            
            return coordination_context
            
        except Exception as e:
            self.logger.error(f"❌ Agent coordination failed: {e}")
            raise
    
    async def make_coordination_decision(
        self,
        coordination_id: str,
        decision_context: Dict[str, Any],
        decision_type: str = "resource_allocation"
    ) -> CoordinationDecision:
        """
        Make intelligent coordination decisions using context analysis.
        
        Args:
            coordination_id: Active coordination ID
            decision_context: Context for the decision
            decision_type: Type of decision to make
            
        Returns:
            CoordinationDecision with recommended action
        """
        start_time = time.time()
        
        try:
            coordination_context = self.active_coordinations.get(coordination_id)
            if not coordination_context:
                raise ValueError(f"No active coordination found: {coordination_id}")
            
            self.logger.info(f"🧠 Making coordination decision: {decision_type}")
            
            # Get reasoning insight for decision
            reasoning_insight = await self._reasoning_engine.provide_reasoning_support(
                context=decision_context,
                reasoning_type=ReasoningType.DECISION_SUPPORT
            )
            
            # Analyze current coordination state
            coordination_state = await self._analyze_coordination_state(coordination_context)
            
            # Consider agent capabilities and current workloads
            agent_analysis = await self._analyze_participating_agents(coordination_context)
            
            # Generate decision options
            decision_options = await self._generate_decision_options(
                decision_type, coordination_context, reasoning_insight, agent_analysis
            )
            
            # Evaluate options using context intelligence
            option_evaluations = await self._evaluate_decision_options(
                decision_options, coordination_context, reasoning_insight
            )
            
            # Select best option
            best_option = max(option_evaluations, key=lambda x: x["score"])
            
            # Create coordination decision
            decision = CoordinationDecision(
                decision_id=str(uuid.uuid4()),
                decision_type=decision_type,
                affected_agents=coordination_context.participating_agents,
                decision_rationale=best_option["rationale"],
                confidence_score=reasoning_insight.confidence_score,
                expected_outcomes=reasoning_insight.recommendations,
                implementation_steps=best_option["implementation_steps"],
                success_probability=best_option["success_probability"],
                risk_factors=reasoning_insight.potential_risks,
                monitoring_metrics=best_option["monitoring_metrics"]
            )
            
            # Store decision in history
            self.decision_history.append(decision)
            
            # Implement decision if confidence is high enough
            if decision.confidence_score >= self.decision_confidence_threshold:
                await self._implement_coordination_decision(decision, coordination_context)
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                f"✅ Coordination decision made: {decision_type} with "
                f"{decision.confidence_score:.1%} confidence in {processing_time:.2f}s"
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"❌ Coordination decision failed: {e}")
            raise
    
    async def optimize_agent_workloads(
        self,
        coordination_id: Optional[str] = None,
        optimization_goals: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Optimize agent workloads using context intelligence.
        
        Args:
            coordination_id: Specific coordination to optimize (all if None)
            optimization_goals: Specific optimization objectives
            
        Returns:
            Optimization results
        """
        start_time = time.time()
        
        try:
            self.logger.info("⚡ Optimizing agent workloads")
            
            # Determine scope of optimization
            if coordination_id:
                coordinations = [self.active_coordinations[coordination_id]]
            else:
                coordinations = list(self.active_coordinations.values())
            
            optimization_results = {
                "coordinations_optimized": 0,
                "agents_rebalanced": 0,
                "performance_improvement": 0.0,
                "context_utilization_improvement": 0.0,
                "optimizations_applied": []
            }
            
            # Analyze current workload distribution
            current_workloads = await self._analyze_current_workloads(coordinations)
            
            # Identify optimization opportunities
            optimization_opportunities = await self._identify_workload_optimization_opportunities(
                coordinations, current_workloads, optimization_goals
            )
            
            # Apply optimizations
            for opportunity in optimization_opportunities:
                result = await self._apply_workload_optimization(opportunity, coordinations)
                
                optimization_results["coordinations_optimized"] += result["coordinations_affected"]
                optimization_results["agents_rebalanced"] += result["agents_rebalanced"]
                optimization_results["optimizations_applied"].append(result["optimization_type"])
            
            # Calculate performance improvement
            new_workloads = await self._analyze_current_workloads(coordinations)
            optimization_results["performance_improvement"] = await self._calculate_performance_improvement(
                current_workloads, new_workloads
            )
            
            processing_time = time.time() - start_time
            
            self.logger.info(
                f"✅ Workload optimization complete: {optimization_results['agents_rebalanced']} "
                f"agents rebalanced, {optimization_results['performance_improvement']:.1%} improvement"
            )
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"❌ Workload optimization failed: {e}")
            raise
    
    async def get_coordination_analytics(
        self,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        coordination_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive coordination analytics.
        
        Args:
            time_range: Time range for analytics
            coordination_id: Specific coordination to analyze
            
        Returns:
            Coordination analytics data
        """
        try:
            self.logger.info("📊 Generating coordination analytics")
            
            # Default to last 24 hours
            if not time_range:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(days=1)
                time_range = (start_time, end_time)
            
            analytics = {
                "time_range": {
                    "start": time_range[0].isoformat(),
                    "end": time_range[1].isoformat()
                },
                "coordination_overview": {},
                "agent_performance": {},
                "context_utilization": {},
                "decision_analytics": {},
                "optimization_insights": {},
                "recommendations": []
            }
            
            # Coordination overview
            analytics["coordination_overview"] = {
                "active_coordinations": len(self.active_coordinations),
                "total_agents_coordinated": len(set().union(*[
                    coord.participating_agents for coord in self.active_coordinations.values()
                ])),
                "coordination_success_rate": await self._calculate_coordination_success_rate(),
                "average_coordination_duration": await self._calculate_average_coordination_duration()
            }
            
            # Agent performance analysis
            analytics["agent_performance"] = await self._analyze_agent_performance(time_range)
            
            # Context utilization analysis
            analytics["context_utilization"] = await self._analyze_context_utilization(time_range)
            
            # Decision analytics
            analytics["decision_analytics"] = await self._analyze_coordination_decisions(time_range)
            
            # Generate optimization insights
            analytics["optimization_insights"] = await self._generate_coordination_optimization_insights()
            
            # Generate recommendations
            analytics["recommendations"] = await self._generate_coordination_recommendations(analytics)
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate coordination analytics: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the coordination system."""
        if not self._is_running:
            return
        
        self.logger.info("🔄 Shutting down Context-Aware Agent Coordination...")
        
        self._is_running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Gracefully complete active coordinations
        for coordination_id in list(self.active_coordinations.keys()):
            await self._gracefully_complete_coordination(coordination_id)
        
        self.logger.info("✅ Context-Aware Agent Coordination shutdown complete")
    
    # Private helper methods
    
    async def _select_optimal_agents(
        self,
        required_capabilities: List[str],
        task_analysis: Any,
        max_agents: int,
        priority_level: float
    ) -> List[Agent]:
        """Select optimal agents for coordination based on context analysis."""
        if not self.db_session:
            return []
        
        # Get available agents
        result = await self.db_session.execute(
            select(Agent).where(
                and_(
                    Agent.is_active == True,
                    Agent.status == AgentStatus.IDLE
                )
            )
        )
        available_agents = list(result.scalars().all())
        
        # Score agents based on capability match and context fit
        agent_scores = []
        for agent in available_agents:
            score = await self._calculate_agent_suitability_score(
                agent, required_capabilities, task_analysis, priority_level
            )
            agent_scores.append((agent, score))
        
        # Sort by score and select top agents
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        selected_agents = [agent for agent, score in agent_scores[:max_agents]]
        
        return selected_agents
    
    async def _determine_optimal_coordination_strategy(
        self, agents: List[Agent], task_analysis: Any, capabilities: List[str]
    ) -> CoordinationStrategy:
        """Determine optimal coordination strategy based on context."""
        # Simple heuristic-based selection
        if len(agents) <= 2:
            return CoordinationStrategy.PEER_TO_PEER
        elif "complex" in str(task_analysis.decision_context).lower():
            return CoordinationStrategy.CONSENSUS_DRIVEN
        else:
            return CoordinationStrategy.CONTEXT_OPTIMIZED
    
    async def _start_background_tasks(self) -> None:
        """Start background coordination tasks."""
        # Coordination monitoring task
        task = asyncio.create_task(self._background_coordination_monitoring())
        self._background_tasks.add(task)
        
        # Performance optimization task
        task = asyncio.create_task(self._background_performance_optimization())
        self._background_tasks.add(task)
        
        # Agent profile updates task
        task = asyncio.create_task(self._background_profile_updates())
        self._background_tasks.add(task)
        
        self.logger.info("🔄 Background coordination tasks started")
    
    # Placeholder implementations for complex methods
    
    async def _load_agent_profiles(self) -> None:
        """Load agent capability profiles."""
        pass  # Implementation would load from database/cache
    
    async def _calculate_agent_suitability_score(
        self, agent: Agent, capabilities: List[str], task_analysis: Any, priority: float
    ) -> float:
        """Calculate how suitable an agent is for the task."""
        return 0.5  # Placeholder score
    
    async def _extract_coordination_context(
        self, agents: List[Agent], task_description: str, task_analysis: Any
    ) -> Dict[str, Any]:
        """Extract shared context for coordination."""
        return {
            "task_summary": task_description,
            "analysis_insights": task_analysis.recommendations,
            "agent_count": len(agents)
        }
    
    async def _background_coordination_monitoring(self) -> None:
        """Background task for monitoring active coordinations."""
        while self._is_running:
            try:
                for coordination_id in list(self.active_coordinations.keys()):
                    await self._monitor_coordination_health(coordination_id)
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Coordination monitoring error: {e}")
                await asyncio.sleep(60)


# Global coordination system instance
_coordination_system: Optional[ContextAwareAgentCoordination] = None


async def get_context_aware_coordination(
    db_session: Optional[AsyncSession] = None
) -> ContextAwareAgentCoordination:
    """
    Get or create the global context-aware agent coordination instance.
    
    Args:
        db_session: Optional database session
        
    Returns:
        ContextAwareAgentCoordination instance
    """
    global _coordination_system
    
    if _coordination_system is None:
        _coordination_system = ContextAwareAgentCoordination(db_session)
        await _coordination_system.initialize()
    
    return _coordination_system


async def shutdown_context_aware_coordination():
    """Shutdown the global context-aware agent coordination system."""
    global _coordination_system
    
    if _coordination_system:
        await _coordination_system.shutdown()
        _coordination_system = None