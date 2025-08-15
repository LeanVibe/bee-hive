"""
Intelligent Task Router for LeanVibe Agent Hive 2.0

Advanced task routing system that provides project-aware agent selection,
workload balancing, specialization matching, and intelligent coordination
for optimal multi-agent development workflows.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import structlog
import math

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from ..core.database import get_session
from ..core.redis import get_redis_client, RedisClient
from ..models.agent import Agent, AgentStatus
from ..models.project_index import ProjectIndex
from ..models.task import Task, TaskStatus, TaskPriority
from .context_integration import AgentContextIntegration, AgentProjectHistory

logger = structlog.get_logger()


class RoutingStrategy(Enum):
    """Task routing strategies."""
    EXPERTISE_BASED = "expertise_based"
    LOAD_BALANCED = "load_balanced"
    PROJECT_FAMILIARITY = "project_familiarity"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    COLLABORATIVE = "collaborative"
    SPECIALIZATION = "specialization"


class TaskComplexity(Enum):
    """Task complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


class TaskUrgency(Enum):
    """Task urgency levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TaskRequirements:
    """Requirements for a task routing request."""
    task_id: str
    task_type: str
    project_id: Optional[str] = None
    required_capabilities: List[str] = None
    preferred_agents: List[str] = None
    excluded_agents: List[str] = None
    complexity: TaskComplexity = TaskComplexity.MODERATE
    urgency: TaskUrgency = TaskUrgency.NORMAL
    estimated_duration_minutes: int = 60
    requires_project_context: bool = True
    max_parallel_agents: int = 1
    collaboration_required: bool = False
    deadline: Optional[datetime] = None
    
    def __post_init__(self):
        if self.required_capabilities is None:
            self.required_capabilities = []
        if self.preferred_agents is None:
            self.preferred_agents = []
        if self.excluded_agents is None:
            self.excluded_agents = []


@dataclass
class AgentScore:
    """Scoring for agent suitability for a task."""
    agent_id: str
    total_score: float
    capability_score: float
    availability_score: float
    familiarity_score: float
    performance_score: float
    workload_score: float
    collaboration_score: float
    explanation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoutingDecision:
    """Result of task routing decision."""
    task_id: str
    selected_agents: List[str]
    routing_strategy: RoutingStrategy
    agent_scores: List[AgentScore]
    decision_confidence: float
    estimated_completion_time: datetime
    routing_metadata: Dict[str, Any]
    alternatives: List[List[str]]  # Alternative agent combinations
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "selected_agents": self.selected_agents,
            "routing_strategy": self.routing_strategy.value,
            "agent_scores": [score.to_dict() for score in self.agent_scores],
            "decision_confidence": self.decision_confidence,
            "estimated_completion_time": self.estimated_completion_time.isoformat(),
            "routing_metadata": self.routing_metadata,
            "alternatives": self.alternatives
        }


class WorkloadMetrics(NamedTuple):
    """Agent workload metrics."""
    active_tasks: int
    queued_tasks: int
    avg_task_duration: float
    context_usage: float
    stress_level: float


class IntelligentTaskRouter:
    """
    Advanced task routing system for intelligent agent selection and coordination.
    
    Provides project-aware routing, workload balancing, specialization matching,
    and collaboration coordination for optimal multi-agent development workflows.
    """
    
    def __init__(
        self,
        session: AsyncSession,
        redis_client: RedisClient,
        context_integration: AgentContextIntegration
    ):
        self.session = session
        self.redis = redis_client
        self.context_integration = context_integration
        
        # Routing configuration
        self.max_routing_time_seconds = 5.0
        self.score_threshold = 0.3  # Minimum score for agent selection
        self.collaboration_threshold = 0.7  # Score threshold for multi-agent tasks
        
        # Performance tracking
        self.routing_history_ttl = 86400 * 7  # 7 days
        self.metrics_cache_ttl = 300  # 5 minutes
        
        # Scoring weights for different factors
        self.scoring_weights = {
            "capability": 0.25,
            "availability": 0.20,
            "familiarity": 0.20,
            "performance": 0.15,
            "workload": 0.15,
            "collaboration": 0.05
        }
    
    async def route_task(
        self,
        requirements: TaskRequirements,
        strategy: RoutingStrategy = RoutingStrategy.EXPERTISE_BASED
    ) -> RoutingDecision:
        """
        Route a task to the most suitable agent(s).
        
        Args:
            requirements: Task requirements and constraints
            strategy: Routing strategy to use
            
        Returns:
            RoutingDecision with selected agents and metadata
        """
        start_time = datetime.utcnow()
        
        logger.info(
            "Starting task routing",
            task_id=requirements.task_id,
            strategy=strategy.value,
            project_id=requirements.project_id,
            complexity=requirements.complexity.value
        )
        
        try:
            # Get available agents
            available_agents = await self._get_available_agents(requirements)
            
            if not available_agents:
                raise ValueError("No available agents found for task routing")
            
            # Score agents based on requirements and strategy
            agent_scores = await self._score_agents(requirements, available_agents, strategy)
            
            # Select best agent(s) based on scores
            selected_agents = await self._select_agents(requirements, agent_scores, strategy)
            
            # Calculate decision confidence
            confidence = self._calculate_decision_confidence(agent_scores, selected_agents)
            
            # Estimate completion time
            completion_time = await self._estimate_completion_time(
                requirements, selected_agents
            )
            
            # Generate alternatives
            alternatives = self._generate_alternatives(agent_scores, selected_agents)
            
            # Create routing decision
            decision = RoutingDecision(
                task_id=requirements.task_id,
                selected_agents=selected_agents,
                routing_strategy=strategy,
                agent_scores=agent_scores,
                decision_confidence=confidence,
                estimated_completion_time=completion_time,
                routing_metadata={
                    "routing_time_ms": int((datetime.utcnow() - start_time).total_seconds() * 1000),
                    "available_agents_count": len(available_agents),
                    "scored_agents_count": len(agent_scores),
                    "strategy_used": strategy.value,
                    "complexity": requirements.complexity.value,
                    "urgency": requirements.urgency.value,
                    "collaboration_required": requirements.collaboration_required
                },
                alternatives=alternatives
            )
            
            # Store routing decision for analytics
            await self._store_routing_decision(decision, requirements)
            
            # Update agent workload tracking
            await self._update_agent_workloads(selected_agents, requirements)
            
            logger.info(
                "Task routing completed",
                task_id=requirements.task_id,
                selected_agents=selected_agents,
                confidence=confidence,
                routing_time_ms=decision.routing_metadata["routing_time_ms"]
            )
            
            return decision
            
        except Exception as e:
            logger.error(
                "Task routing failed",
                task_id=requirements.task_id,
                error=str(e),
                strategy=strategy.value
            )
            raise
    
    async def update_agent_performance(
        self,
        agent_id: str,
        task_id: str,
        performance_data: Dict[str, Any]
    ) -> None:
        """
        Update agent performance metrics based on task completion.
        
        Args:
            agent_id: Agent identifier
            task_id: Completed task identifier
            performance_data: Performance metrics and outcomes
        """
        try:
            # Store performance data
            performance_key = f"agent_performance:{agent_id}"
            await self.redis.hset(
                performance_key,
                mapping={
                    f"task_{task_id}": json.dumps({
                        **performance_data,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                }
            )
            await self.redis.expire(performance_key, self.routing_history_ttl)
            
            # Update aggregated metrics
            await self._update_agent_metrics(agent_id, performance_data)
            
            logger.info(
                "Agent performance updated",
                agent_id=agent_id,
                task_id=task_id,
                success=performance_data.get("success", False)
            )
            
        except Exception as e:
            logger.error(
                "Failed to update agent performance",
                agent_id=agent_id,
                task_id=task_id,
                error=str(e)
            )
    
    async def get_routing_analytics(
        self,
        time_range_hours: int = 24,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get routing analytics and performance metrics.
        
        Args:
            time_range_hours: Time range for analytics
            project_id: Optional project filter
            
        Returns:
            Analytics data
        """
        try:
            # Get routing decisions from the time range
            decisions = await self._get_routing_decisions(time_range_hours, project_id)
            
            if not decisions:
                return {"error": "No routing decisions found in time range"}
            
            # Calculate analytics
            analytics = {
                "time_range_hours": time_range_hours,
                "total_routings": len(decisions),
                "average_confidence": sum(d["decision_confidence"] for d in decisions) / len(decisions),
                "strategy_distribution": self._calculate_strategy_distribution(decisions),
                "agent_utilization": await self._calculate_agent_utilization(decisions),
                "routing_performance": self._calculate_routing_performance(decisions),
                "complexity_breakdown": self._calculate_complexity_breakdown(decisions),
                "collaboration_stats": self._calculate_collaboration_stats(decisions)
            }
            
            if project_id:
                analytics["project_id"] = project_id
                analytics["project_specific_metrics"] = await self._get_project_metrics(project_id)
            
            return analytics
            
        except Exception as e:
            logger.error(
                "Failed to get routing analytics",
                time_range_hours=time_range_hours,
                project_id=project_id,
                error=str(e)
            )
            return {"error": str(e)}
    
    async def optimize_routing_strategy(
        self,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze routing performance and suggest optimizations.
        
        Args:
            project_id: Optional project to optimize for
            
        Returns:
            Optimization recommendations
        """
        try:
            # Get historical performance data
            analytics = await self.get_routing_analytics(168, project_id)  # 7 days
            
            if "error" in analytics:
                return analytics
            
            # Analyze performance patterns
            recommendations = {
                "current_performance": {
                    "average_confidence": analytics["average_confidence"],
                    "total_routings": analytics["total_routings"],
                    "agent_utilization": analytics["agent_utilization"]
                },
                "recommendations": []
            }
            
            # Check for low confidence patterns
            if analytics["average_confidence"] < 0.6:
                recommendations["recommendations"].append({
                    "type": "low_confidence",
                    "description": "Average routing confidence is low",
                    "suggestion": "Consider expanding agent capabilities or adjusting scoring weights",
                    "priority": "high"
                })
            
            # Check for uneven agent utilization
            utilization = analytics["agent_utilization"]
            if utilization:
                max_util = max(utilization.values())
                min_util = min(utilization.values())
                if max_util - min_util > 0.5:
                    recommendations["recommendations"].append({
                        "type": "uneven_utilization",
                        "description": "Agent workload is unevenly distributed",
                        "suggestion": "Adjust load balancing weights or agent capabilities",
                        "priority": "medium"
                    })
            
            # Check collaboration efficiency
            collab_stats = analytics.get("collaboration_stats", {})
            if collab_stats.get("multi_agent_tasks", 0) > 0:
                success_rate = collab_stats.get("collaboration_success_rate", 0)
                if success_rate < 0.8:
                    recommendations["recommendations"].append({
                        "type": "collaboration_efficiency",
                        "description": "Multi-agent collaboration success rate is low",
                        "suggestion": "Review agent compatibility and communication patterns",
                        "priority": "medium"
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(
                "Failed to optimize routing strategy",
                project_id=project_id,
                error=str(e)
            )
            return {"error": str(e)}
    
    # ================== PRIVATE METHODS ==================
    
    async def _get_available_agents(
        self,
        requirements: TaskRequirements
    ) -> List[Agent]:
        """Get agents available for task assignment."""
        # Base query for active agents
        stmt = select(Agent).where(
            and_(
                Agent.status == AgentStatus.active,
                Agent.id.notin_(requirements.excluded_agents) if requirements.excluded_agents else True
            )
        )
        
        # Prefer specified agents if any
        if requirements.preferred_agents:
            preferred_stmt = stmt.where(Agent.id.in_(requirements.preferred_agents))
            result = await self.session.execute(preferred_stmt)
            preferred = result.scalars().all()
            
            if preferred:
                return preferred
        
        # Get all available agents
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def _score_agents(
        self,
        requirements: TaskRequirements,
        agents: List[Agent],
        strategy: RoutingStrategy
    ) -> List[AgentScore]:
        """Score agents based on task requirements and routing strategy."""
        scores = []
        
        for agent in agents:
            # Calculate individual scores
            capability_score = await self._calculate_capability_score(agent, requirements)
            availability_score = await self._calculate_availability_score(agent, requirements)
            familiarity_score = await self._calculate_familiarity_score(agent, requirements)
            performance_score = await self._calculate_performance_score(agent, requirements)
            workload_score = await self._calculate_workload_score(agent, requirements)
            collaboration_score = await self._calculate_collaboration_score(agent, requirements)
            
            # Apply strategy-specific weighting
            weights = self._get_strategy_weights(strategy)
            
            total_score = (
                capability_score * weights["capability"] +
                availability_score * weights["availability"] +
                familiarity_score * weights["familiarity"] +
                performance_score * weights["performance"] +
                workload_score * weights["workload"] +
                collaboration_score * weights["collaboration"]
            )
            
            # Generate explanation
            explanation = self._generate_score_explanation(
                capability_score, availability_score, familiarity_score,
                performance_score, workload_score, collaboration_score, strategy
            )
            
            score = AgentScore(
                agent_id=str(agent.id),
                total_score=total_score,
                capability_score=capability_score,
                availability_score=availability_score,
                familiarity_score=familiarity_score,
                performance_score=performance_score,
                workload_score=workload_score,
                collaboration_score=collaboration_score,
                explanation=explanation
            )
            
            scores.append(score)
        
        # Sort by total score descending
        scores.sort(key=lambda x: x.total_score, reverse=True)
        
        return scores
    
    async def _calculate_capability_score(
        self,
        agent: Agent,
        requirements: TaskRequirements
    ) -> float:
        """Calculate capability match score for agent."""
        if not agent.capabilities or not requirements.required_capabilities:
            return 0.5  # Neutral score if no capability data
        
        total_match = 0.0
        for req_cap in requirements.required_capabilities:
            best_match = 0.0
            
            for agent_cap in agent.capabilities:
                cap_name = agent_cap.get("name", "").lower()
                confidence = agent_cap.get("confidence_level", 0.0)
                areas = agent_cap.get("specialization_areas", [])
                
                # Direct capability match
                if req_cap.lower() in cap_name:
                    best_match = max(best_match, confidence)
                
                # Specialization area match
                for area in areas:
                    if req_cap.lower() in area.lower():
                        best_match = max(best_match, confidence * 0.8)
            
            total_match += best_match
        
        return min(total_match / len(requirements.required_capabilities), 1.0)
    
    async def _calculate_availability_score(
        self,
        agent: Agent,
        requirements: TaskRequirements
    ) -> float:
        """Calculate agent availability score."""
        if not agent.is_available_for_task():
            return 0.0
        
        # Check context window usage
        context_usage = float(agent.context_window_usage or 0.0)
        availability = 1.0 - context_usage
        
        # Apply urgency weighting
        if requirements.urgency == TaskUrgency.CRITICAL:
            # Critical tasks get higher availability score
            return min(availability * 1.2, 1.0)
        elif requirements.urgency == TaskUrgency.LOW:
            # Low priority tasks can use busier agents
            return availability * 0.8
        
        return availability
    
    async def _calculate_familiarity_score(
        self,
        agent: Agent,
        requirements: TaskRequirements
    ) -> float:
        """Calculate project familiarity score."""
        if not requirements.project_id:
            return 0.5  # Neutral if no project context
        
        # Get agent's project history
        history = await self.context_integration._get_agent_project_history(
            str(agent.id), requirements.project_id
        )
        
        return self.context_integration._calculate_familiarity_score(history)
    
    async def _calculate_performance_score(
        self,
        agent: Agent,
        requirements: TaskRequirements
    ) -> float:
        """Calculate agent performance score."""
        try:
            # Get recent performance metrics
            performance_key = f"agent_performance:{agent.id}"
            performance_data = await self.redis.hgetall(performance_key)
            
            if not performance_data:
                return 0.5  # Neutral score for new agents
            
            # Calculate success rate from recent tasks
            recent_tasks = []
            cutoff = datetime.utcnow() - timedelta(days=7)
            
            for task_key, task_data in performance_data.items():
                if task_key.startswith("task_"):
                    try:
                        data = json.loads(task_data)
                        task_time = datetime.fromisoformat(data["timestamp"])
                        if task_time >= cutoff:
                            recent_tasks.append(data)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
            
            if not recent_tasks:
                return 0.5
            
            # Calculate weighted performance score
            success_count = sum(1 for task in recent_tasks if task.get("success", False))
            success_rate = success_count / len(recent_tasks)
            
            # Consider task complexity and agent performance on similar tasks
            complexity_bonus = 0.0
            if requirements.task_type:
                similar_tasks = [
                    task for task in recent_tasks
                    if task.get("task_type") == requirements.task_type
                ]
                if similar_tasks:
                    similar_success = sum(1 for task in similar_tasks if task.get("success", False))
                    complexity_bonus = (similar_success / len(similar_tasks)) * 0.2
            
            return min(success_rate + complexity_bonus, 1.0)
            
        except Exception as e:
            logger.warning(
                "Failed to calculate performance score",
                agent_id=str(agent.id),
                error=str(e)
            )
            return 0.5
    
    async def _calculate_workload_score(
        self,
        agent: Agent,
        requirements: TaskRequirements
    ) -> float:
        """Calculate workload balance score."""
        try:
            workload = await self._get_agent_workload(str(agent.id))
            
            # Calculate workload pressure
            active_pressure = workload.active_tasks / 5.0  # Assume max 5 concurrent tasks
            queue_pressure = workload.queued_tasks / 10.0  # Assume max 10 queued tasks
            context_pressure = workload.context_usage
            
            total_pressure = (active_pressure + queue_pressure + context_pressure) / 3.0
            
            # Invert pressure to get score (lower pressure = higher score)
            workload_score = max(0.0, 1.0 - total_pressure)
            
            # Apply urgency weighting
            if requirements.urgency == TaskUrgency.CRITICAL:
                return workload_score  # Don't penalize busy agents for critical tasks
            
            return workload_score
            
        except Exception as e:
            logger.warning(
                "Failed to calculate workload score",
                agent_id=str(agent.id),
                error=str(e)
            )
            return 0.5
    
    async def _calculate_collaboration_score(
        self,
        agent: Agent,
        requirements: TaskRequirements
    ) -> float:
        """Calculate collaboration compatibility score."""
        if not requirements.collaboration_required:
            return 1.0  # Max score if collaboration not needed
        
        try:
            # Get agent's collaboration history
            collab_key = f"agent_collaboration:{agent.id}"
            collab_data = await self.redis.hgetall(collab_key)
            
            if not collab_data:
                return 0.5  # Neutral for agents without collaboration history
            
            # Calculate collaboration success rate
            total_collabs = int(collab_data.get("total_collaborations", 0))
            successful_collabs = int(collab_data.get("successful_collaborations", 0))
            
            if total_collabs == 0:
                return 0.5
            
            success_rate = successful_collabs / total_collabs
            
            # Consider compatibility with preferred agents
            if requirements.preferred_agents:
                compatibility_bonus = 0.0
                for preferred_agent in requirements.preferred_agents:
                    if preferred_agent != str(agent.id):
                        compat_score = float(collab_data.get(f"compatibility_{preferred_agent}", 0.5))
                        compatibility_bonus += compat_score
                
                if len(requirements.preferred_agents) > 1:
                    compatibility_bonus /= (len(requirements.preferred_agents) - 1)
                    success_rate = (success_rate + compatibility_bonus) / 2.0
            
            return min(success_rate, 1.0)
            
        except Exception as e:
            logger.warning(
                "Failed to calculate collaboration score",
                agent_id=str(agent.id),
                error=str(e)
            )
            return 0.5
    
    def _get_strategy_weights(self, strategy: RoutingStrategy) -> Dict[str, float]:
        """Get scoring weights for a specific routing strategy."""
        if strategy == RoutingStrategy.EXPERTISE_BASED:
            return {
                "capability": 0.40,
                "availability": 0.15,
                "familiarity": 0.15,
                "performance": 0.20,
                "workload": 0.05,
                "collaboration": 0.05
            }
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            return {
                "capability": 0.20,
                "availability": 0.25,
                "familiarity": 0.10,
                "performance": 0.15,
                "workload": 0.25,
                "collaboration": 0.05
            }
        elif strategy == RoutingStrategy.PROJECT_FAMILIARITY:
            return {
                "capability": 0.20,
                "availability": 0.15,
                "familiarity": 0.35,
                "performance": 0.20,
                "workload": 0.05,
                "collaboration": 0.05
            }
        elif strategy == RoutingStrategy.PERFORMANCE_OPTIMIZED:
            return {
                "capability": 0.25,
                "availability": 0.15,
                "familiarity": 0.15,
                "performance": 0.35,
                "workload": 0.05,
                "collaboration": 0.05
            }
        elif strategy == RoutingStrategy.COLLABORATIVE:
            return {
                "capability": 0.20,
                "availability": 0.15,
                "familiarity": 0.15,
                "performance": 0.15,
                "workload": 0.05,
                "collaboration": 0.30
            }
        else:
            # Default weights
            return self.scoring_weights
    
    def _generate_score_explanation(
        self,
        capability: float,
        availability: float,
        familiarity: float,
        performance: float,
        workload: float,
        collaboration: float,
        strategy: RoutingStrategy
    ) -> str:
        """Generate human-readable explanation for agent score."""
        explanations = []
        
        if capability >= 0.8:
            explanations.append("excellent capability match")
        elif capability >= 0.6:
            explanations.append("good capability match")
        elif capability >= 0.4:
            explanations.append("moderate capability match")
        else:
            explanations.append("limited capability match")
        
        if availability >= 0.8:
            explanations.append("highly available")
        elif availability >= 0.6:
            explanations.append("available")
        else:
            explanations.append("busy")
        
        if familiarity >= 0.7:
            explanations.append("familiar with project")
        elif familiarity >= 0.4:
            explanations.append("some project experience")
        else:
            explanations.append("new to project")
        
        if performance >= 0.8:
            explanations.append("strong performance history")
        elif performance >= 0.6:
            explanations.append("good performance")
        else:
            explanations.append("developing performance")
        
        return f"Agent selected based on {strategy.value} strategy: {', '.join(explanations)}"
    
    async def _select_agents(
        self,
        requirements: TaskRequirements,
        agent_scores: List[AgentScore],
        strategy: RoutingStrategy
    ) -> List[str]:
        """Select best agents based on scores and requirements."""
        # Filter agents above threshold
        qualified_agents = [
            score for score in agent_scores
            if score.total_score >= self.score_threshold
        ]
        
        if not qualified_agents:
            # If no agents meet threshold, take the best one anyway
            qualified_agents = agent_scores[:1] if agent_scores else []
        
        if requirements.max_parallel_agents == 1:
            # Single agent selection
            return [qualified_agents[0].agent_id] if qualified_agents else []
        
        # Multi-agent selection
        selected = []
        
        if requirements.collaboration_required:
            # Select agents with good collaboration scores
            collaborative_agents = [
                score for score in qualified_agents
                if score.collaboration_score >= self.collaboration_threshold
            ]
            
            if collaborative_agents:
                selected = [agent.agent_id for agent in collaborative_agents[:requirements.max_parallel_agents]]
            else:
                # Fall back to top agents even if collaboration scores are lower
                selected = [agent.agent_id for agent in qualified_agents[:requirements.max_parallel_agents]]
        else:
            # Select top scoring agents
            selected = [agent.agent_id for agent in qualified_agents[:requirements.max_parallel_agents]]
        
        return selected
    
    def _calculate_decision_confidence(
        self,
        agent_scores: List[AgentScore],
        selected_agents: List[str]
    ) -> float:
        """Calculate confidence in the routing decision."""
        if not agent_scores or not selected_agents:
            return 0.0
        
        # Get scores of selected agents
        selected_scores = [
            score.total_score for score in agent_scores
            if score.agent_id in selected_agents
        ]
        
        if not selected_scores:
            return 0.0
        
        # Base confidence is the average score of selected agents
        avg_selected_score = sum(selected_scores) / len(selected_scores)
        
        # Boost confidence if there's a clear winner
        if len(agent_scores) > 1:
            top_score = agent_scores[0].total_score
            second_score = agent_scores[1].total_score if len(agent_scores) > 1 else 0.0
            
            score_gap = top_score - second_score
            gap_bonus = min(score_gap, 0.3)  # Max 0.3 bonus for clear separation
            
            return min(avg_selected_score + gap_bonus, 1.0)
        
        return avg_selected_score
    
    async def _estimate_completion_time(
        self,
        requirements: TaskRequirements,
        selected_agents: List[str]
    ) -> datetime:
        """Estimate task completion time based on agents and requirements."""
        base_duration = requirements.estimated_duration_minutes
        
        # Adjust for complexity
        complexity_multipliers = {
            TaskComplexity.SIMPLE: 0.8,
            TaskComplexity.MODERATE: 1.0,
            TaskComplexity.COMPLEX: 1.5,
            TaskComplexity.EXPERT: 2.0
        }
        
        duration = base_duration * complexity_multipliers[requirements.complexity]
        
        # Adjust for multi-agent coordination overhead
        if len(selected_agents) > 1:
            coordination_overhead = 1.2 + (len(selected_agents) - 1) * 0.1
            duration *= coordination_overhead
        
        # Adjust based on agent workload
        total_workload = 0.0
        for agent_id in selected_agents:
            workload = await self._get_agent_workload(agent_id)
            total_workload += workload.stress_level
        
        if selected_agents:
            avg_workload = total_workload / len(selected_agents)
            workload_multiplier = 1.0 + (avg_workload * 0.5)  # Up to 50% longer if overloaded
            duration *= workload_multiplier
        
        return datetime.utcnow() + timedelta(minutes=duration)
    
    def _generate_alternatives(
        self,
        agent_scores: List[AgentScore],
        selected_agents: List[str]
    ) -> List[List[str]]:
        """Generate alternative agent combinations."""
        alternatives = []
        
        # Get non-selected agents above threshold
        alternative_agents = [
            score.agent_id for score in agent_scores
            if score.agent_id not in selected_agents and score.total_score >= self.score_threshold
        ]
        
        # Generate single agent alternatives
        for agent_id in alternative_agents[:3]:  # Top 3 alternatives
            alternatives.append([agent_id])
        
        # Generate multi-agent alternatives if original selection was multi-agent
        if len(selected_agents) > 1 and len(alternative_agents) >= len(selected_agents):
            # Combination of top alternative agents
            alternatives.append(alternative_agents[:len(selected_agents)])
        
        return alternatives
    
    async def _get_agent_workload(self, agent_id: str) -> WorkloadMetrics:
        """Get current workload metrics for an agent."""
        try:
            workload_key = f"agent_workload:{agent_id}"
            workload_data = await self.redis.hgetall(workload_key)
            
            return WorkloadMetrics(
                active_tasks=int(workload_data.get("active_tasks", 0)),
                queued_tasks=int(workload_data.get("queued_tasks", 0)),
                avg_task_duration=float(workload_data.get("avg_task_duration", 60.0)),
                context_usage=float(workload_data.get("context_usage", 0.0)),
                stress_level=float(workload_data.get("stress_level", 0.0))
            )
            
        except Exception as e:
            logger.warning(
                "Failed to get agent workload",
                agent_id=agent_id,
                error=str(e)
            )
            return WorkloadMetrics(0, 0, 60.0, 0.0, 0.0)
    
    async def _store_routing_decision(
        self,
        decision: RoutingDecision,
        requirements: TaskRequirements
    ) -> None:
        """Store routing decision for analytics."""
        try:
            decision_key = f"routing_decision:{decision.task_id}"
            decision_data = {
                **decision.to_dict(),
                "requirements": {
                    "task_type": requirements.task_type,
                    "project_id": requirements.project_id,
                    "complexity": requirements.complexity.value,
                    "urgency": requirements.urgency.value,
                    "collaboration_required": requirements.collaboration_required
                }
            }
            
            await self.redis.setex(
                decision_key,
                self.routing_history_ttl,
                json.dumps(decision_data)
            )
            
        except Exception as e:
            logger.warning(
                "Failed to store routing decision",
                task_id=decision.task_id,
                error=str(e)
            )
    
    async def _update_agent_workloads(
        self,
        agent_ids: List[str],
        requirements: TaskRequirements
    ) -> None:
        """Update workload tracking for selected agents."""
        try:
            for agent_id in agent_ids:
                workload_key = f"agent_workload:{agent_id}"
                
                # Increment active tasks
                await self.redis.hincrby(workload_key, "active_tasks", 1)
                
                # Update estimated completion time
                await self.redis.hset(
                    workload_key,
                    "last_assignment",
                    datetime.utcnow().isoformat()
                )
                
                await self.redis.expire(workload_key, self.metrics_cache_ttl)
                
        except Exception as e:
            logger.warning(
                "Failed to update agent workloads",
                agent_ids=agent_ids,
                error=str(e)
            )
    
    async def _update_agent_metrics(
        self,
        agent_id: str,
        performance_data: Dict[str, Any]
    ) -> None:
        """Update aggregated agent metrics."""
        try:
            metrics_key = f"agent_metrics:{agent_id}"
            
            # Update success rate
            if "success" in performance_data:
                if performance_data["success"]:
                    await self.redis.hincrby(metrics_key, "successful_tasks", 1)
                await self.redis.hincrby(metrics_key, "total_tasks", 1)
            
            # Update average duration
            if "duration_minutes" in performance_data:
                duration = performance_data["duration_minutes"]
                current_avg = float(await self.redis.hget(metrics_key, "avg_duration") or 0.0)
                total_tasks = int(await self.redis.hget(metrics_key, "total_tasks") or 1)
                
                new_avg = ((current_avg * (total_tasks - 1)) + duration) / total_tasks
                await self.redis.hset(metrics_key, "avg_duration", str(new_avg))
            
            await self.redis.expire(metrics_key, self.routing_history_ttl)
            
        except Exception as e:
            logger.warning(
                "Failed to update agent metrics",
                agent_id=agent_id,
                error=str(e)
            )
    
    async def _get_routing_decisions(
        self,
        time_range_hours: int,
        project_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get routing decisions from time range."""
        try:
            # Get all routing decision keys
            pattern = "routing_decision:*"
            keys = await self.redis.keys(pattern)
            
            decisions = []
            cutoff = datetime.utcnow() - timedelta(hours=time_range_hours)
            
            for key in keys:
                try:
                    data = await self.redis.get(key)
                    if data:
                        decision = json.loads(data)
                        
                        # Check time range
                        created_at = datetime.fromisoformat(decision["routing_metadata"]["timestamp"])
                        if created_at >= cutoff:
                            # Filter by project if specified
                            if project_id is None or decision.get("requirements", {}).get("project_id") == project_id:
                                decisions.append(decision)
                
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
            
            return decisions
            
        except Exception as e:
            logger.warning(
                "Failed to get routing decisions",
                time_range_hours=time_range_hours,
                error=str(e)
            )
            return []
    
    def _calculate_strategy_distribution(self, decisions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of routing strategies used."""
        distribution = {}
        for decision in decisions:
            strategy = decision.get("routing_strategy", "unknown")
            distribution[strategy] = distribution.get(strategy, 0) + 1
        return distribution
    
    async def _calculate_agent_utilization(self, decisions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate agent utilization from routing decisions."""
        agent_assignments = {}
        
        for decision in decisions:
            for agent_id in decision.get("selected_agents", []):
                agent_assignments[agent_id] = agent_assignments.get(agent_id, 0) + 1
        
        # Convert to utilization percentages
        total_assignments = sum(agent_assignments.values())
        if total_assignments == 0:
            return {}
        
        utilization = {}
        for agent_id, count in agent_assignments.items():
            utilization[agent_id] = count / total_assignments
        
        return utilization
    
    def _calculate_routing_performance(self, decisions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate routing performance metrics."""
        if not decisions:
            return {}
        
        # Average confidence
        avg_confidence = sum(d.get("decision_confidence", 0) for d in decisions) / len(decisions)
        
        # Average routing time
        routing_times = [
            d.get("routing_metadata", {}).get("routing_time_ms", 0)
            for d in decisions
        ]
        avg_routing_time = sum(routing_times) / len(routing_times) if routing_times else 0
        
        return {
            "average_confidence": avg_confidence,
            "average_routing_time_ms": avg_routing_time,
            "total_decisions": len(decisions)
        }
    
    def _calculate_complexity_breakdown(self, decisions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate breakdown by task complexity."""
        breakdown = {}
        for decision in decisions:
            complexity = decision.get("requirements", {}).get("complexity", "unknown")
            breakdown[complexity] = breakdown.get(complexity, 0) + 1
        return breakdown
    
    def _calculate_collaboration_stats(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate collaboration statistics."""
        multi_agent_tasks = sum(
            1 for d in decisions
            if len(d.get("selected_agents", [])) > 1
        )
        
        collaboration_required = sum(
            1 for d in decisions
            if d.get("requirements", {}).get("collaboration_required", False)
        )
        
        return {
            "multi_agent_tasks": multi_agent_tasks,
            "collaboration_required_tasks": collaboration_required,
            "collaboration_rate": multi_agent_tasks / len(decisions) if decisions else 0
        }
    
    async def _get_project_metrics(self, project_id: str) -> Dict[str, Any]:
        """Get project-specific routing metrics."""
        # This would integrate with project analytics
        # For now, return basic structure
        return {
            "project_id": project_id,
            "agent_familiarity_scores": {},
            "common_task_types": [],
            "preferred_strategies": []
        }


# Factory function for dependency injection
async def get_intelligent_task_router(
    session: AsyncSession = None,
    redis_client: RedisClient = None,
    context_integration: AgentContextIntegration = None
) -> IntelligentTaskRouter:
    """Factory function to create IntelligentTaskRouter instance."""
    if session is None:
        session = await get_session()
    if redis_client is None:
        redis_client = await get_redis_client()
    if context_integration is None:
        from .context_integration import get_agent_context_integration
        context_integration = await get_agent_context_integration(session, redis_client)
    
    return IntelligentTaskRouter(session, redis_client, context_integration)