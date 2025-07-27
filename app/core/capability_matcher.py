"""
Capability-Based Agent Matching Engine for LeanVibe Agent Hive 2.0

Provides intelligent matching algorithms for agent capabilities, performance
scoring, and load balancing to enable optimal task assignment decisions.
"""

import asyncio
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import structlog
from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session

from .database import get_session
from ..models.agent import Agent, AgentStatus
from ..models.task import Task, TaskStatus, TaskPriority, TaskType
from ..models.performance_metric import PerformanceMetric

logger = structlog.get_logger()


class MatchingAlgorithm(Enum):
    """Different matching algorithms for capability analysis."""
    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    SEMANTIC_MATCH = "semantic_match"
    WEIGHTED_MATCH = "weighted_match"


@dataclass
class CapabilityScore:
    """Represents a capability matching score with detailed breakdown."""
    capability_name: str
    base_score: float
    confidence_multiplier: float
    specialization_bonus: float
    experience_factor: float
    final_score: float
    match_type: MatchingAlgorithm


@dataclass
class AgentPerformanceProfile:
    """Comprehensive performance profile for an agent."""
    agent_id: str
    total_tasks_completed: int
    total_tasks_failed: int
    success_rate: float
    average_completion_time: float
    recent_performance_trend: float
    workload_capacity: float
    current_workload: float
    specialization_scores: Dict[str, float]
    reliability_score: float
    efficiency_score: float


@dataclass
class WorkloadMetrics:
    """Current workload metrics for an agent."""
    active_tasks: int
    pending_tasks: int
    context_usage: float
    estimated_availability: float
    priority_distribution: Dict[TaskPriority, int]
    task_type_distribution: Dict[str, int]


class CapabilityMatcher:
    """
    Advanced capability matching engine for intelligent task routing.
    
    Provides sophisticated algorithms for matching agent capabilities
    to task requirements, including performance-based scoring and
    workload-aware load balancing.
    """
    
    def __init__(self):
        self.performance_cache: Dict[str, AgentPerformanceProfile] = {}
        self.workload_cache: Dict[str, WorkloadMetrics] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_last_updated: Dict[str, datetime] = {}
    
    async def match_capabilities(
        self,
        requirements: List[str],
        agent_capabilities: Dict[str, Any],
        algorithm: MatchingAlgorithm = MatchingAlgorithm.WEIGHTED_MATCH
    ) -> float:
        """
        Match agent capabilities against task requirements.
        
        Args:
            requirements: List of required capability names
            agent_capabilities: Agent's capability configuration
            algorithm: Matching algorithm to use
            
        Returns:
            Match score between 0.0 and 1.0
        """
        if not requirements or not agent_capabilities:
            return 0.0
        
        try:
            if algorithm == MatchingAlgorithm.EXACT_MATCH:
                return await self._exact_match_capabilities(requirements, agent_capabilities)
            elif algorithm == MatchingAlgorithm.FUZZY_MATCH:
                return await self._fuzzy_match_capabilities(requirements, agent_capabilities)
            elif algorithm == MatchingAlgorithm.SEMANTIC_MATCH:
                return await self._semantic_match_capabilities(requirements, agent_capabilities)
            elif algorithm == MatchingAlgorithm.WEIGHTED_MATCH:
                return await self._weighted_match_capabilities(requirements, agent_capabilities)
            else:
                return await self._weighted_match_capabilities(requirements, agent_capabilities)
                
        except Exception as e:
            logger.error("Error matching capabilities", error=str(e), algorithm=algorithm.value)
            return 0.0
    
    async def calculate_performance_score(
        self, 
        agent_id: str, 
        task_type: str,
        include_historical: bool = True
    ) -> float:
        """
        Calculate comprehensive performance score for an agent for specific task type.
        
        Args:
            agent_id: Agent identifier
            task_type: Type of task being evaluated
            include_historical: Whether to include historical performance data
            
        Returns:
            Performance score between 0.0 and 1.0
        """
        try:
            performance_profile = await self._get_agent_performance_profile(agent_id)
            
            if not performance_profile:
                return 0.5  # Neutral score for new agents
            
            # Base performance metrics (40% weight)
            base_score = performance_profile.success_rate * 0.4
            
            # Task type specialization (30% weight)
            specialization_score = performance_profile.specialization_scores.get(task_type, 0.5) * 0.3
            
            # Recent performance trend (20% weight)
            trend_score = max(0.0, min(1.0, performance_profile.recent_performance_trend)) * 0.2
            
            # Efficiency score (10% weight)
            efficiency_score = performance_profile.efficiency_score * 0.1
            
            total_score = base_score + specialization_score + trend_score + efficiency_score
            
            logger.debug(
                "Performance score calculated",
                agent_id=agent_id,
                task_type=task_type,
                base_score=base_score,
                specialization_score=specialization_score,
                trend_score=trend_score,
                efficiency_score=efficiency_score,
                total_score=total_score
            )
            
            return min(1.0, total_score)
            
        except Exception as e:
            logger.error("Error calculating performance score", agent_id=agent_id, error=str(e))
            return 0.5
    
    async def get_workload_factor(self, agent_id: str) -> float:
        """
        Calculate current workload factor for load balancing.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Workload factor where 0.0 = no load, 1.0 = fully loaded
        """
        try:
            workload_metrics = await self._get_agent_workload_metrics(agent_id)
            
            if not workload_metrics:
                return 0.0  # No current workload
            
            # Context usage factor (40% weight)
            context_factor = workload_metrics.context_usage * 0.4
            
            # Active tasks factor (30% weight)
            max_concurrent_tasks = 3  # Configurable limit
            task_factor = min(1.0, workload_metrics.active_tasks / max_concurrent_tasks) * 0.3
            
            # Queue length factor (20% weight)
            max_queue_length = 5  # Configurable limit
            queue_factor = min(1.0, workload_metrics.pending_tasks / max_queue_length) * 0.2
            
            # Priority task distribution factor (10% weight)
            high_priority_tasks = workload_metrics.priority_distribution.get(TaskPriority.HIGH, 0)
            critical_tasks = workload_metrics.priority_distribution.get(TaskPriority.CRITICAL, 0)
            priority_factor = min(1.0, (high_priority_tasks + critical_tasks * 2) / 5) * 0.1
            
            total_workload = context_factor + task_factor + queue_factor + priority_factor
            
            logger.debug(
                "Workload factor calculated",
                agent_id=agent_id,
                context_factor=context_factor,
                task_factor=task_factor,
                queue_factor=queue_factor,
                priority_factor=priority_factor,
                total_workload=total_workload
            )
            
            return min(1.0, total_workload)
            
        except Exception as e:
            logger.error("Error calculating workload factor", agent_id=agent_id, error=str(e))
            return 1.0  # Conservative approach - assume fully loaded if error
    
    async def calculate_composite_suitability_score(
        self,
        agent_id: str,
        requirements: List[str],
        agent_capabilities: Dict[str, Any],
        task_type: str,
        priority: TaskPriority,
        consider_workload: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate comprehensive suitability score combining all factors.
        
        Args:
            agent_id: Agent identifier
            requirements: Required capabilities
            agent_capabilities: Agent's capabilities
            task_type: Task type
            priority: Task priority
            consider_workload: Whether to factor in current workload
            
        Returns:
            Tuple of (final_score, score_breakdown)
        """
        try:
            score_breakdown = {}
            
            # Capability matching (40% weight)
            capability_score = await self.match_capabilities(requirements, agent_capabilities)
            score_breakdown["capability_match"] = capability_score * 0.4
            
            # Performance history (30% weight)
            performance_score = await self.calculate_performance_score(agent_id, task_type)
            score_breakdown["performance"] = performance_score * 0.3
            
            # Workload consideration (20% weight)
            if consider_workload:
                workload_factor = await self.get_workload_factor(agent_id)
                # Lower workload = higher availability score
                availability_score = 1.0 - workload_factor
                score_breakdown["availability"] = availability_score * 0.2
            else:
                score_breakdown["availability"] = 0.2  # Full availability assumed
            
            # Priority alignment (10% weight)
            priority_score = await self._calculate_priority_alignment_score(agent_id, priority)
            score_breakdown["priority_alignment"] = priority_score * 0.1
            
            # Calculate final composite score
            final_score = sum(score_breakdown.values())
            
            logger.debug(
                "Composite suitability score calculated",
                agent_id=agent_id,
                task_type=task_type,
                final_score=final_score,
                breakdown=score_breakdown
            )
            
            return final_score, score_breakdown
            
        except Exception as e:
            logger.error("Error calculating composite suitability score", agent_id=agent_id, error=str(e))
            return 0.0, {"error": 1.0}
    
    async def _exact_match_capabilities(self, requirements: List[str], agent_capabilities: Dict) -> float:
        """Exact string matching for capabilities."""
        if not isinstance(agent_capabilities, list):
            return 0.0
        
        agent_capability_names = [cap.get("name", "").lower() for cap in agent_capabilities]
        matches = sum(1 for req in requirements if req.lower() in agent_capability_names)
        return matches / len(requirements) if requirements else 0.0
    
    async def _fuzzy_match_capabilities(self, requirements: List[str], agent_capabilities: Dict) -> float:
        """Fuzzy matching with partial string matching."""
        if not isinstance(agent_capabilities, list):
            return 0.0
        
        total_score = 0.0
        
        for requirement in requirements:
            req_lower = requirement.lower()
            best_match_score = 0.0
            
            for cap in agent_capabilities:
                cap_name = cap.get("name", "").lower()
                cap_areas = cap.get("specialization_areas", [])
                confidence = cap.get("confidence_level", 0.0)
                
                # Direct name match
                if req_lower in cap_name or cap_name in req_lower:
                    score = confidence * 1.0
                    best_match_score = max(best_match_score, score)
                
                # Specialization area match
                for area in cap_areas:
                    if req_lower in area.lower() or area.lower() in req_lower:
                        score = confidence * 0.8
                        best_match_score = max(best_match_score, score)
            
            total_score += best_match_score
        
        return total_score / len(requirements) if requirements else 0.0
    
    async def _semantic_match_capabilities(self, requirements: List[str], agent_capabilities: Dict) -> float:
        """Semantic matching using keyword relationships (placeholder for future enhancement)."""
        # For now, use fuzzy matching as baseline
        return await self._fuzzy_match_capabilities(requirements, agent_capabilities)
    
    async def _weighted_match_capabilities(self, requirements: List[str], agent_capabilities: Dict) -> float:
        """Advanced weighted matching considering confidence and specialization."""
        if not isinstance(agent_capabilities, list):
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for requirement in requirements:
            req_lower = requirement.lower()
            req_weight = 1.0  # Can be enhanced to have different weights per requirement
            best_match_score = 0.0
            
            for cap in agent_capabilities:
                cap_name = cap.get("name", "").lower()
                cap_areas = cap.get("specialization_areas", [])
                confidence = cap.get("confidence_level", 0.0)
                
                # Calculate match strength
                match_strength = 0.0
                
                # Exact name match
                if req_lower == cap_name:
                    match_strength = 1.0
                elif req_lower in cap_name:
                    match_strength = 0.8
                elif cap_name in req_lower:
                    match_strength = 0.7
                
                # Check specialization areas
                for area in cap_areas:
                    area_lower = area.lower()
                    if req_lower == area_lower:
                        match_strength = max(match_strength, 0.9)
                    elif req_lower in area_lower or area_lower in req_lower:
                        match_strength = max(match_strength, 0.6)
                
                # Apply confidence weighting
                final_score = match_strength * confidence
                best_match_score = max(best_match_score, final_score)
            
            total_weighted_score += best_match_score * req_weight
            total_weight += req_weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    async def _get_agent_performance_profile(self, agent_id: str) -> Optional[AgentPerformanceProfile]:
        """Retrieve or calculate agent performance profile with caching."""
        # Check cache first
        if agent_id in self.performance_cache:
            last_updated = self.cache_last_updated.get(agent_id)
            if last_updated and (datetime.utcnow() - last_updated).total_seconds() < self.cache_ttl:
                return self.performance_cache[agent_id]
        
        try:
            async with get_session() as db_session:
                # Get agent basic info
                agent = await db_session.get(Agent, agent_id)
                if not agent:
                    return None
                
                # Calculate task completion statistics
                completed_tasks_query = select(func.count(Task.id)).where(
                    and_(Task.assigned_agent_id == agent_id, Task.status == TaskStatus.COMPLETED)
                )
                failed_tasks_query = select(func.count(Task.id)).where(
                    and_(Task.assigned_agent_id == agent_id, Task.status == TaskStatus.FAILED)
                )
                
                completed_count = (await db_session.execute(completed_tasks_query)).scalar() or 0
                failed_count = (await db_session.execute(failed_tasks_query)).scalar() or 0
                total_tasks = completed_count + failed_count
                
                success_rate = completed_count / total_tasks if total_tasks > 0 else 0.5
                
                # Calculate average completion time
                avg_completion_query = select(func.avg(Task.actual_effort)).where(
                    and_(
                        Task.assigned_agent_id == agent_id,
                        Task.status == TaskStatus.COMPLETED,
                        Task.actual_effort.isnot(None)
                    )
                )
                avg_completion_time = (await db_session.execute(avg_completion_query)).scalar() or 0.0
                
                # Calculate recent performance trend (last 30 days)
                recent_date = datetime.utcnow() - timedelta(days=30)
                recent_success_query = select(func.count(Task.id)).where(
                    and_(
                        Task.assigned_agent_id == agent_id,
                        Task.status == TaskStatus.COMPLETED,
                        Task.completed_at >= recent_date
                    )
                )
                recent_total_query = select(func.count(Task.id)).where(
                    and_(
                        Task.assigned_agent_id == agent_id,
                        Task.status.in_([TaskStatus.COMPLETED, TaskStatus.FAILED]),
                        Task.completed_at >= recent_date
                    )
                )
                
                recent_success = (await db_session.execute(recent_success_query)).scalar() or 0
                recent_total = (await db_session.execute(recent_total_query)).scalar() or 0
                recent_trend = recent_success / recent_total if recent_total > 0 else success_rate
                
                # Calculate specialization scores by task type
                specialization_scores = await self._calculate_specialization_scores(db_session, agent_id)
                
                # Calculate reliability and efficiency scores
                reliability_score = self._calculate_reliability_score(success_rate, recent_trend)
                efficiency_score = self._calculate_efficiency_score(avg_completion_time)
                
                profile = AgentPerformanceProfile(
                    agent_id=agent_id,
                    total_tasks_completed=completed_count,
                    total_tasks_failed=failed_count,
                    success_rate=success_rate,
                    average_completion_time=avg_completion_time,
                    recent_performance_trend=recent_trend,
                    workload_capacity=1.0,  # Default capacity
                    current_workload=0.0,   # Will be calculated separately
                    specialization_scores=specialization_scores,
                    reliability_score=reliability_score,
                    efficiency_score=efficiency_score
                )
                
                # Cache the profile
                self.performance_cache[agent_id] = profile
                self.cache_last_updated[agent_id] = datetime.utcnow()
                
                return profile
                
        except Exception as e:
            logger.error("Error getting agent performance profile", agent_id=agent_id, error=str(e))
            return None
    
    async def _get_agent_workload_metrics(self, agent_id: str) -> Optional[WorkloadMetrics]:
        """Calculate current workload metrics for an agent."""
        try:
            async with get_session() as db_session:
                # Count active tasks
                active_tasks_query = select(func.count(Task.id)).where(
                    and_(Task.assigned_agent_id == agent_id, Task.status == TaskStatus.IN_PROGRESS)
                )
                active_tasks = (await db_session.execute(active_tasks_query)).scalar() or 0
                
                # Count pending tasks
                pending_tasks_query = select(func.count(Task.id)).where(
                    and_(Task.assigned_agent_id == agent_id, Task.status == TaskStatus.ASSIGNED)
                )
                pending_tasks = (await db_session.execute(pending_tasks_query)).scalar() or 0
                
                # Get agent context usage
                agent = await db_session.get(Agent, agent_id)
                context_usage = float(agent.context_window_usage or 0.0) if agent else 0.0
                
                # Calculate priority distribution
                priority_dist = {}
                for priority in TaskPriority:
                    count_query = select(func.count(Task.id)).where(
                        and_(
                            Task.assigned_agent_id == agent_id,
                            Task.status.in_([TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]),
                            Task.priority == priority
                        )
                    )
                    count = (await db_session.execute(count_query)).scalar() or 0
                    priority_dist[priority] = count
                
                # Calculate task type distribution
                type_dist = {}
                for task_type in TaskType:
                    count_query = select(func.count(Task.id)).where(
                        and_(
                            Task.assigned_agent_id == agent_id,
                            Task.status.in_([TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]),
                            Task.task_type == task_type
                        )
                    )
                    count = (await db_session.execute(count_query)).scalar() or 0
                    type_dist[task_type.value] = count
                
                # Calculate estimated availability
                max_capacity = 5  # Configurable max concurrent tasks
                current_load = active_tasks + pending_tasks
                estimated_availability = max(0.0, 1.0 - (current_load / max_capacity))
                
                metrics = WorkloadMetrics(
                    active_tasks=active_tasks,
                    pending_tasks=pending_tasks,
                    context_usage=context_usage,
                    estimated_availability=estimated_availability,
                    priority_distribution=priority_dist,
                    task_type_distribution=type_dist
                )
                
                # Cache the metrics
                self.workload_cache[agent_id] = metrics
                self.cache_last_updated[f"workload_{agent_id}"] = datetime.utcnow()
                
                return metrics
                
        except Exception as e:
            logger.error("Error getting agent workload metrics", agent_id=agent_id, error=str(e))
            return None
    
    async def _calculate_specialization_scores(self, db_session: Session, agent_id: str) -> Dict[str, float]:
        """Calculate specialization scores for different task types."""
        specialization_scores = {}
        
        try:
            for task_type in TaskType:
                # Get completion rate for this task type
                completed_query = select(func.count(Task.id)).where(
                    and_(
                        Task.assigned_agent_id == agent_id,
                        Task.task_type == task_type,
                        Task.status == TaskStatus.COMPLETED
                    )
                )
                total_query = select(func.count(Task.id)).where(
                    and_(
                        Task.assigned_agent_id == agent_id,
                        Task.task_type == task_type,
                        Task.status.in_([TaskStatus.COMPLETED, TaskStatus.FAILED])
                    )
                )
                
                completed = (await db_session.execute(completed_query)).scalar() or 0
                total = (await db_session.execute(total_query)).scalar() or 0
                
                if total > 0:
                    base_score = completed / total
                    # Boost score based on experience (number of tasks)
                    experience_boost = min(0.2, total * 0.02)  # Up to 20% boost
                    specialization_scores[task_type.value] = min(1.0, base_score + experience_boost)
                else:
                    specialization_scores[task_type.value] = 0.5  # Neutral score for no experience
                    
        except Exception as e:
            logger.error("Error calculating specialization scores", agent_id=agent_id, error=str(e))
        
        return specialization_scores
    
    def _calculate_reliability_score(self, success_rate: float, recent_trend: float) -> float:
        """Calculate reliability score based on success rate and recent performance."""
        # Weight current success rate 60%, recent trend 40%
        return (success_rate * 0.6) + (recent_trend * 0.4)
    
    def _calculate_efficiency_score(self, avg_completion_time: float) -> float:
        """Calculate efficiency score based on completion time."""
        if avg_completion_time <= 0:
            return 0.5  # Neutral score for no data
        
        # Efficiency is inversely related to completion time
        # Assume 60 minutes is baseline (score 0.5), with diminishing returns
        baseline_time = 60.0
        if avg_completion_time <= baseline_time:
            # Linear improvement below baseline
            return 0.5 + (0.5 * (baseline_time - avg_completion_time) / baseline_time)
        else:
            # Exponential decay above baseline
            return 0.5 * math.exp(-(avg_completion_time - baseline_time) / baseline_time)
    
    async def _calculate_priority_alignment_score(self, agent_id: str, priority: TaskPriority) -> float:
        """Calculate how well an agent handles tasks of specific priority."""
        try:
            async with get_session() as db_session:
                # Get success rate for this priority level
                completed_query = select(func.count(Task.id)).where(
                    and_(
                        Task.assigned_agent_id == agent_id,
                        Task.priority == priority,
                        Task.status == TaskStatus.COMPLETED
                    )
                )
                total_query = select(func.count(Task.id)).where(
                    and_(
                        Task.assigned_agent_id == agent_id,
                        Task.priority == priority,
                        Task.status.in_([TaskStatus.COMPLETED, TaskStatus.FAILED])
                    )
                )
                
                completed = (await db_session.execute(completed_query)).scalar() or 0
                total = (await db_session.execute(total_query)).scalar() or 0
                
                if total > 0:
                    return completed / total
                else:
                    # No history for this priority - return neutral score
                    return 0.7
                    
        except Exception as e:
            logger.error("Error calculating priority alignment score", agent_id=agent_id, error=str(e))
            return 0.5
    
    def clear_cache(self, agent_id: Optional[str] = None) -> None:
        """Clear performance and workload caches."""
        if agent_id:
            self.performance_cache.pop(agent_id, None)
            self.workload_cache.pop(agent_id, None)
            self.cache_last_updated.pop(agent_id, None)
            self.cache_last_updated.pop(f"workload_{agent_id}", None)
        else:
            self.performance_cache.clear()
            self.workload_cache.clear()
            self.cache_last_updated.clear()
        
        logger.info("Cache cleared", agent_id=agent_id or "all")