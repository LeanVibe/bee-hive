"""
Intelligent Sleep Manager - Advanced activity pattern analysis and schedule optimization.

Provides sophisticated agent behavior analysis and intelligent sleep management:
- Advanced activity pattern recognition and prediction
- Machine learning-based schedule optimization
- Adaptive sleep window adjustment based on performance
- Cross-agent coordination for system-wide efficiency
- Predictive sleep scheduling based on workload forecasting
"""

import asyncio
import json
import logging
import statistics
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from uuid import UUID
from collections import defaultdict, deque
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc
from sqlalchemy.orm import selectinload

from ..models.sleep_wake import SleepWakeCycle, SleepState, SleepWindow
from ..models.agent import Agent, AgentStatus
from ..models.task import Task, TaskStatus
from ..core.database import get_async_session
from ..core.sleep_scheduler import SleepScheduler, ActivityProfile, ScheduleOptimization, SleepMetrics
from ..core.context_consolidator import get_context_consolidator, ConsolidationResult
from ..core.config import get_settings
from ..core.redis import get_redis


logger = logging.getLogger(__name__)


@dataclass
class ActivityProfile:
    """Enhanced activity profile with predictive capabilities."""
    agent_id: UUID
    hourly_activity: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    daily_patterns: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    task_duration_patterns: deque = field(default_factory=lambda: deque(maxlen=200))
    workload_trends: List[float] = field(default_factory=list)
    efficiency_scores: Dict[str, float] = field(default_factory=dict)
    sleep_quality_metrics: Dict[str, float] = field(default_factory=dict)
    peak_performance_hours: List[int] = field(default_factory=list)
    optimal_sleep_duration: float = 4.0
    predicted_workload: Dict[str, float] = field(default_factory=dict)
    adaptation_score: float = 0.0
    
    def calculate_workload_trend(self, hours_back: int = 24) -> float:
        """Calculate workload trend over specified hours."""
        if len(self.workload_trends) < 2:
            return 0.0
        
        recent_trends = self.workload_trends[-min(hours_back, len(self.workload_trends)):]
        if len(recent_trends) < 2:
            return 0.0
        
        # Calculate trend using linear regression
        x = np.arange(len(recent_trends))
        y = np.array(recent_trends)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)
        
        return 0.0
    
    def predict_next_activity_window(self) -> Tuple[int, int, float]:
        """Predict next high-activity window (start_hour, end_hour, confidence)."""
        if not self.peak_performance_hours:
            return (9, 17, 0.5)  # Default business hours
        
        # Find most common peak performance period
        peak_counts = defaultdict(int)
        for hour in self.peak_performance_hours:
            # Group into 4-hour windows
            window_start = (hour // 4) * 4
            peak_counts[window_start] += 1
        
        if peak_counts:
            most_common_start = max(peak_counts.keys(), key=lambda k: peak_counts[k])
            confidence = peak_counts[most_common_start] / max(1, len(self.peak_performance_hours))
            return (most_common_start, (most_common_start + 4) % 24, confidence)
        
        return (9, 17, 0.3)


@dataclass
class ScheduleRecommendation:
    """Advanced schedule recommendation with predictive insights."""
    agent_id: UUID
    recommended_sleep_start: int
    recommended_sleep_end: int
    confidence_score: float
    predicted_token_reduction: float
    expected_performance_gain: float
    adaptation_rationale: List[str]
    coordination_requirements: Dict[str, Any]
    fallback_windows: List[Tuple[int, int]]
    risk_factors: List[str]
    implementation_priority: str  # high, medium, low


@dataclass
class SleepResult:
    """Comprehensive sleep execution result."""
    agent_id: UUID
    sleep_initiated: bool
    sleep_start_time: Optional[datetime]
    expected_wake_time: Optional[datetime]
    consolidation_scheduled: bool
    efficiency_prediction: float
    coordination_status: Dict[str, bool]
    warnings: List[str]
    errors: List[str]


class IntelligentSleepManager:
    """
    Advanced sleep management with machine learning and coordination.
    
    Features:
    - Predictive activity pattern analysis
    - Machine learning-based schedule optimization
    - Cross-agent coordination for system efficiency
    - Adaptive sleep duration based on workload
    - Real-time performance monitoring and adjustment
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.sleep_scheduler = None  # Will be initialized lazily
        self.context_consolidator = get_context_consolidator()
        
        # Enhanced analytics
        self._enhanced_profiles: Dict[UUID, ActivityProfile] = {}
        self._coordination_graph: Dict[UUID, Set[UUID]] = {}  # Agent dependencies
        self._system_efficiency_history: deque = deque(maxlen=168)  # Week of hourly data
        self._optimization_history: Dict[UUID, List[ScheduleRecommendation]] = defaultdict(list)
        
        # Machine learning parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.8
        self.min_data_points_for_prediction = 20
        self.coordination_weight = 0.3
        
        # Performance thresholds
        self.performance_improvement_threshold = 0.15  # 15% improvement needed
        self.system_efficiency_target = 0.85
        self.cross_agent_coordination_enabled = True
        
        # Advanced settings
        self.enable_predictive_scheduling = True
        self.enable_adaptive_duration = True
        self.enable_workload_forecasting = True
        self.max_schedule_adjustments_per_day = 3
    
    async def analyze_activity_patterns(self, agent_id: UUID) -> ActivityProfile:
        """
        Perform advanced activity pattern analysis with predictive modeling.
        
        Args:
            agent_id: Agent ID to analyze
            
        Returns:
            Enhanced ActivityProfile with predictions
        """
        try:
            logger.info(f"Analyzing activity patterns for agent {agent_id}")
            
            # Get or create enhanced profile
            profile = await self._get_or_create_enhanced_profile(agent_id)
            
            # Update with recent data
            await self._update_profile_with_recent_data(agent_id, profile)
            
            # Perform predictive analysis
            await self._analyze_workload_trends(agent_id, profile)
            await self._calculate_efficiency_metrics(agent_id, profile)
            await self._predict_future_patterns(agent_id, profile)
            
            # Update coordination graph
            await self._update_coordination_graph(agent_id)
            
            # Calculate adaptation score
            profile.adaptation_score = await self._calculate_adaptation_score(agent_id, profile)
            
            self._enhanced_profiles[agent_id] = profile
            
            logger.info(f"Activity analysis completed for agent {agent_id} (adaptation score: {profile.adaptation_score:.2f})")
            return profile
            
        except Exception as e:
            logger.error(f"Error analyzing activity patterns for agent {agent_id}: {e}")
            raise
    
    async def recommend_sleep_schedule(self, agent_id: UUID) -> ScheduleRecommendation:
        """
        Generate intelligent sleep schedule recommendation with coordination.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Advanced schedule recommendation
        """
        try:
            logger.info(f"Generating sleep schedule recommendation for agent {agent_id}")
            
            # Ensure we have current analysis
            profile = await self.analyze_activity_patterns(agent_id)
            
            # Basic schedule optimization
            next_activity_start, next_activity_end, confidence = profile.predict_next_activity_window()
            
            # Calculate optimal sleep window avoiding peak activity
            optimal_sleep_start = (next_activity_end + 1) % 24
            optimal_sleep_end = (optimal_sleep_start + int(profile.optimal_sleep_duration)) % 24
            
            # Adjust for coordination requirements
            if self.cross_agent_coordination_enabled:
                optimal_sleep_start, optimal_sleep_end = await self._coordinate_with_other_agents(
                    agent_id, optimal_sleep_start, optimal_sleep_end
                )
            
            # Calculate predictions
            predicted_token_reduction = await self._predict_token_reduction(agent_id, profile)
            expected_performance_gain = await self._predict_performance_gain(agent_id, profile)
            
            # Generate rationale
            rationale = [
                f"Predicted low activity between {optimal_sleep_start}:00-{optimal_sleep_end}:00",
                f"Expected {predicted_token_reduction:.1%} token reduction",
                f"Anticipated {expected_performance_gain:.1%} performance improvement"
            ]
            
            # Add workload trend insights
            trend = profile.calculate_workload_trend()
            if trend > 0.1:
                rationale.append("Increasing workload trend detected - prioritizing efficiency")
            elif trend < -0.1:
                rationale.append("Decreasing workload trend - opportunity for longer consolidation")
            
            # Generate fallback windows
            fallback_windows = await self._generate_fallback_windows(agent_id, profile)
            
            # Assess risk factors
            risk_factors = await self._assess_schedule_risks(agent_id, optimal_sleep_start, optimal_sleep_end)
            
            # Determine implementation priority
            priority = self._determine_implementation_priority(profile, predicted_token_reduction, risk_factors)
            
            recommendation = ScheduleRecommendation(
                agent_id=agent_id,
                recommended_sleep_start=optimal_sleep_start,
                recommended_sleep_end=optimal_sleep_end,
                confidence_score=confidence,
                predicted_token_reduction=predicted_token_reduction,
                expected_performance_gain=expected_performance_gain,
                adaptation_rationale=rationale,
                coordination_requirements=await self._get_coordination_requirements(agent_id),
                fallback_windows=fallback_windows,
                risk_factors=risk_factors,
                implementation_priority=priority
            )
            
            # Store recommendation
            self._optimization_history[agent_id].append(recommendation)
            
            logger.info(
                f"Sleep schedule recommendation generated for agent {agent_id}: "
                f"{optimal_sleep_start}:00-{optimal_sleep_end}:00 (confidence: {confidence:.2f})"
            )
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating sleep schedule recommendation for agent {agent_id}: {e}")
            raise
    
    async def execute_automated_sleep(self, agent_id: UUID) -> SleepResult:
        """
        Execute intelligent automated sleep with full coordination.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Comprehensive sleep execution result
        """
        try:
            logger.info(f"Executing automated sleep for agent {agent_id}")
            
            result = SleepResult(
                agent_id=agent_id,
                sleep_initiated=False,
                sleep_start_time=None,
                expected_wake_time=None,
                consolidation_scheduled=False,
                efficiency_prediction=0.0,
                coordination_status={},
                warnings=[],
                errors=[]
            )
            
            # Get recommendation
            recommendation = await self.recommend_sleep_schedule(agent_id)
            
            # Validate pre-conditions
            validation_result = await self._validate_sleep_preconditions(agent_id, recommendation)
            if not validation_result["valid"]:
                result.errors.extend(validation_result["errors"])
                return result
            
            # Initialize sleep scheduler if needed
            if not self.sleep_scheduler:
                from ..core.sleep_scheduler import get_sleep_scheduler
                self.sleep_scheduler = await get_sleep_scheduler()
            
            # Execute coordination steps
            coordination_success = await self._execute_coordination_steps(agent_id, recommendation)
            result.coordination_status = coordination_success
            
            if not all(coordination_success.values()):
                result.warnings.append("Some coordination steps failed")
            
            # Calculate expected wake time
            sleep_duration_hours = (recommendation.recommended_sleep_end - recommendation.recommended_sleep_start) % 24
            expected_wake_time = datetime.utcnow() + timedelta(hours=sleep_duration_hours)
            
            # Initiate sleep
            sleep_success = await self.sleep_scheduler.force_sleep(
                agent_id, 
                duration_minutes=int(sleep_duration_hours * 60)
            )
            
            if sleep_success:
                result.sleep_initiated = True
                result.sleep_start_time = datetime.utcnow()
                result.expected_wake_time = expected_wake_time
                result.efficiency_prediction = recommendation.predicted_token_reduction
                
                # Schedule context consolidation
                try:
                    consolidation_result = await self.context_consolidator.consolidate_during_sleep(agent_id)
                    result.consolidation_scheduled = True
                    
                    logger.info(
                        f"Context consolidation completed for agent {agent_id}: "
                        f"{consolidation_result.tokens_saved} tokens saved"
                    )
                    
                except Exception as e:
                    result.warnings.append(f"Context consolidation failed: {e}")
                
                # Update system efficiency tracking
                await self._update_system_efficiency_tracking(agent_id, recommendation)
                
                logger.info(f"Automated sleep executed successfully for agent {agent_id}")
                
            else:
                result.errors.append("Failed to initiate sleep cycle")
                logger.error(f"Failed to initiate sleep for agent {agent_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing automated sleep for agent {agent_id}: {e}")
            result.errors.append(str(e))
            return result
    
    async def get_system_intelligence_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system intelligence and optimization metrics."""
        try:
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "agents_analyzed": len(self._enhanced_profiles),
                "system_efficiency": await self._calculate_current_system_efficiency(),
                "coordination_graph_size": sum(len(deps) for deps in self._coordination_graph.values()),
                "predictive_accuracy": await self._calculate_predictive_accuracy(),
                "adaptation_scores": {},
                "optimization_performance": {},
                "coordination_effectiveness": await self._calculate_coordination_effectiveness()
            }
            
            # Agent-specific metrics
            for agent_id, profile in self._enhanced_profiles.items():
                metrics["adaptation_scores"][str(agent_id)] = profile.adaptation_score
                
                # Optimization performance
                recent_optimizations = self._optimization_history[agent_id][-5:]  # Last 5
                if recent_optimizations:
                    avg_confidence = statistics.mean([opt.confidence_score for opt in recent_optimizations])
                    avg_token_reduction = statistics.mean([opt.predicted_token_reduction for opt in recent_optimizations])
                    
                    metrics["optimization_performance"][str(agent_id)] = {
                        "average_confidence": avg_confidence,
                        "average_predicted_reduction": avg_token_reduction,
                        "total_optimizations": len(self._optimization_history[agent_id])
                    }
            
            # System-wide predictions
            metrics["system_predictions"] = {
                "next_hour_efficiency": await self._predict_next_hour_efficiency(),
                "optimal_system_sleep_window": await self._calculate_optimal_system_sleep_window(),
                "coordination_opportunities": await self._identify_coordination_opportunities()
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting system intelligence metrics: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    
    async def _get_or_create_enhanced_profile(self, agent_id: UUID) -> ActivityProfile:
        """Get or create enhanced activity profile."""
        if agent_id not in self._enhanced_profiles:
            profile = ActivityProfile(agent_id=agent_id)
            self._enhanced_profiles[agent_id] = profile
        return self._enhanced_profiles[agent_id]
    
    async def _update_profile_with_recent_data(self, agent_id: UUID, profile: ActivityProfile) -> None:
        """Update profile with recent activity data."""
        try:
            async with get_async_session() as session:
                # Get recent tasks (last 7 days)
                cutoff_date = datetime.utcnow() - timedelta(days=7)
                
                tasks_result = await session.execute(
                    select(Task).where(
                        and_(
                            Task.agent_id == agent_id,
                            Task.created_at >= cutoff_date
                        )
                    ).order_by(desc(Task.created_at))
                )
                tasks = tasks_result.scalars().all()
                
                # Update hourly activity
                for task in tasks:
                    if task.created_at:
                        hour = task.created_at.hour
                        day = task.created_at.strftime("%A")  # Monday, Tuesday, etc.
                        
                        profile.hourly_activity[hour] += 1
                        profile.daily_patterns[day].append(hour)
                        
                        # Track task duration
                        if task.completed_at and task.created_at:
                            duration = (task.completed_at - task.created_at).total_seconds() / 60
                            profile.task_duration_patterns.append(duration)
                
                # Calculate workload for trend analysis
                hourly_workload = defaultdict(int)
                for task in tasks:
                    if task.created_at:
                        hour_key = task.created_at.strftime("%Y-%m-%d-%H")
                        hourly_workload[hour_key] += 1
                
                # Update workload trends (last 24 hours)
                recent_hours = [
                    (datetime.utcnow() - timedelta(hours=i)).strftime("%Y-%m-%d-%H")
                    for i in range(24)
                ]
                
                profile.workload_trends = [hourly_workload.get(hour, 0) for hour in recent_hours]
                
        except Exception as e:
            logger.error(f"Error updating profile data for agent {agent_id}: {e}")
    
    async def _analyze_workload_trends(self, agent_id: UUID, profile: ActivityProfile) -> None:
        """Analyze workload trends and patterns."""
        try:
            if len(profile.workload_trends) < self.min_data_points_for_prediction:
                return
            
            # Calculate trend statistics
            trend_slope = profile.calculate_workload_trend()
            
            # Identify peak performance hours
            if profile.hourly_activity:
                avg_activity = statistics.mean(profile.hourly_activity.values())
                profile.peak_performance_hours = [
                    hour for hour, activity in profile.hourly_activity.items()
                    if activity > avg_activity * 1.2
                ]
            
            # Predict optimal sleep duration based on workload
            if trend_slope > 0.2:  # Increasing workload
                profile.optimal_sleep_duration = min(6.0, profile.optimal_sleep_duration + 0.5)
            elif trend_slope < -0.2:  # Decreasing workload
                profile.optimal_sleep_duration = max(3.0, profile.optimal_sleep_duration - 0.5)
            
        except Exception as e:
            logger.error(f"Error analyzing workload trends for agent {agent_id}: {e}")
    
    async def _calculate_efficiency_metrics(self, agent_id: UUID, profile: ActivityProfile) -> None:
        """Calculate agent efficiency metrics."""
        try:
            async with get_async_session() as session:
                # Get recent sleep cycles
                cutoff_date = datetime.utcnow() - timedelta(days=14)
                
                cycles_result = await session.execute(
                    select(SleepWakeCycle).where(
                        and_(
                            SleepWakeCycle.agent_id == agent_id,
                            SleepWakeCycle.created_at >= cutoff_date,
                            SleepWakeCycle.sleep_state == SleepState.AWAKE
                        )
                    )
                )
                cycles = cycles_result.scalars().all()
                
                if cycles:
                    # Calculate sleep quality metrics
                    token_reductions = [c.token_reduction_achieved for c in cycles if c.token_reduction_achieved]
                    recovery_times = [c.recovery_time_ms for c in cycles if c.recovery_time_ms]
                    
                    if token_reductions:
                        profile.sleep_quality_metrics["average_token_reduction"] = statistics.mean(token_reductions)
                        profile.sleep_quality_metrics["token_reduction_variance"] = statistics.variance(token_reductions) if len(token_reductions) > 1 else 0
                    
                    if recovery_times:
                        profile.sleep_quality_metrics["average_recovery_time"] = statistics.mean(recovery_times)
                    
                    # Calculate efficiency score
                    efficiency_scores = []
                    for cycle in cycles:
                        if cycle.token_reduction_achieved and cycle.sleep_time and cycle.wake_time:
                            duration_hours = (cycle.wake_time - cycle.sleep_time).total_seconds() / 3600
                            efficiency = cycle.token_reduction_achieved / max(0.1, duration_hours)
                            efficiency_scores.append(min(1.0, efficiency))
                    
                    if efficiency_scores:
                        profile.efficiency_scores["sleep_efficiency"] = statistics.mean(efficiency_scores)
                
        except Exception as e:
            logger.error(f"Error calculating efficiency metrics for agent {agent_id}: {e}")
    
    async def _predict_future_patterns(self, agent_id: UUID, profile: ActivityProfile) -> None:
        """Predict future activity patterns using historical data."""
        try:
            # Predict next 24 hours workload
            if len(profile.workload_trends) >= self.min_data_points_for_prediction:
                # Simple linear prediction for next 24 hours
                trend = profile.calculate_workload_trend()
                last_value = profile.workload_trends[-1] if profile.workload_trends else 0
                
                for i in range(24):
                    predicted_value = max(0, last_value + (trend * i))
                    profile.predicted_workload[f"hour_{i}"] = predicted_value
            
            # Predict peak activity windows
            if profile.peak_performance_hours:
                # Find most common peak patterns
                peak_patterns = defaultdict(int)
                for hour in profile.peak_performance_hours:
                    pattern_key = f"{hour//4*4}-{(hour//4*4+4)%24}"  # 4-hour windows
                    peak_patterns[pattern_key] += 1
                
                # Store most likely peak pattern
                if peak_patterns:
                    most_likely_pattern = max(peak_patterns.keys(), key=lambda k: peak_patterns[k])
                    profile.predicted_workload["most_likely_peak_window"] = most_likely_pattern
        
        except Exception as e:
            logger.error(f"Error predicting future patterns for agent {agent_id}: {e}")
    
    async def _calculate_adaptation_score(self, agent_id: UUID, profile: ActivityProfile) -> float:
        """Calculate how well the agent adapts to optimizations."""
        try:
            recent_optimizations = self._optimization_history[agent_id][-5:]  # Last 5
            
            if len(recent_optimizations) < 2:
                return 0.5  # Default score
            
            # Calculate improvement in efficiency over time
            efficiency_improvements = []
            for i in range(1, len(recent_optimizations)):
                prev_reduction = recent_optimizations[i-1].predicted_token_reduction
                curr_reduction = recent_optimizations[i].predicted_token_reduction
                
                if prev_reduction > 0:
                    improvement = (curr_reduction - prev_reduction) / prev_reduction
                    efficiency_improvements.append(improvement)
            
            if efficiency_improvements:
                avg_improvement = statistics.mean(efficiency_improvements)
                # Normalize to 0-1 scale
                adaptation_score = min(1.0, max(0.0, (avg_improvement + 1) / 2))
                return adaptation_score
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating adaptation score for agent {agent_id}: {e}")
            return 0.5
    
    async def _coordinate_with_other_agents(self, agent_id: UUID, sleep_start: int, sleep_end: int) -> Tuple[int, int]:
        """Coordinate sleep schedule with other agents to maximize system efficiency."""
        try:
            if not self.cross_agent_coordination_enabled:
                return sleep_start, sleep_end
            
            # Get dependent agents
            dependent_agents = self._coordination_graph.get(agent_id, set())
            
            if not dependent_agents:
                return sleep_start, sleep_end
            
            # Check if any dependent agents are already scheduled to sleep
            conflicts = []
            async with get_async_session() as session:
                for dep_agent_id in dependent_agents:
                    # Check existing sleep windows
                    windows_result = await session.execute(
                        select(SleepWindow).where(
                            and_(
                                SleepWindow.agent_id == dep_agent_id,
                                SleepWindow.active == True
                            )
                        )
                    )
                    windows = windows_result.scalars().all()
                    
                    for window in windows:
                        window_start = window.start_time.hour
                        window_end = window.end_time.hour
                        
                        # Check for overlap
                        if self._time_windows_overlap(sleep_start, sleep_end, window_start, window_end):
                            conflicts.append((dep_agent_id, window_start, window_end))
            
            # Adjust schedule to avoid conflicts
            if conflicts:
                # Find alternative window
                for offset in [1, 2, -1, -2]:  # Try 1-2 hours earlier or later
                    new_start = (sleep_start + offset) % 24
                    new_end = (sleep_end + offset) % 24
                    
                    has_conflict = False
                    for _, conf_start, conf_end in conflicts:
                        if self._time_windows_overlap(new_start, new_end, conf_start, conf_end):
                            has_conflict = True
                            break
                    
                    if not has_conflict:
                        logger.info(f"Adjusted sleep schedule for agent {agent_id} to avoid conflicts: {new_start}:00-{new_end}:00")
                        return new_start, new_end
            
            return sleep_start, sleep_end
            
        except Exception as e:
            logger.error(f"Error coordinating sleep schedule for agent {agent_id}: {e}")
            return sleep_start, sleep_end
    
    def _time_windows_overlap(self, start1: int, end1: int, start2: int, end2: int) -> bool:
        """Check if two time windows overlap (handling day wraparound)."""
        # Normalize windows
        if end1 < start1:  # Overnight window
            return (start2 >= start1) or (start2 <= end1) or (end2 >= start1) or (end2 <= end1)
        if end2 < start2:  # Overnight window
            return (start1 >= start2) or (start1 <= end2) or (end1 >= start2) or (end1 <= end2)
        
        # Normal windows
        return not (end1 <= start2 or end2 <= start1)
    
    async def _predict_token_reduction(self, agent_id: UUID, profile: ActivityProfile) -> float:
        """Predict token reduction for the recommended schedule."""
        try:
            # Base prediction on historical performance
            base_reduction = profile.sleep_quality_metrics.get("average_token_reduction", 0.55)
            
            # Adjust based on workload trend
            trend = profile.calculate_workload_trend()
            if trend > 0.1:  # Increasing workload
                return min(0.8, base_reduction * 1.1)
            elif trend < -0.1:  # Decreasing workload
                return max(0.3, base_reduction * 0.9)
            
            return base_reduction
            
        except Exception as e:
            logger.error(f"Error predicting token reduction for agent {agent_id}: {e}")
            return 0.55  # Default target
    
    async def _predict_performance_gain(self, agent_id: UUID, profile: ActivityProfile) -> float:
        """Predict performance gain from the optimization."""
        try:
            # Base gain on efficiency scores
            current_efficiency = profile.efficiency_scores.get("sleep_efficiency", 0.5)
            adaptation_factor = profile.adaptation_score
            
            # Higher adaptation score means better performance gains
            predicted_gain = current_efficiency * adaptation_factor * 0.3  # Up to 30% gain
            
            return min(0.5, predicted_gain)  # Cap at 50% gain
            
        except Exception as e:
            logger.error(f"Error predicting performance gain for agent {agent_id}: {e}")
            return 0.15  # Default 15% gain
    
    async def _generate_fallback_windows(self, agent_id: UUID, profile: ActivityProfile) -> List[Tuple[int, int]]:
        """Generate fallback sleep windows."""
        try:
            fallbacks = []
            
            # Generate alternative windows based on low-activity periods
            if profile.hourly_activity:
                avg_activity = statistics.mean(profile.hourly_activity.values())
                low_activity_hours = [
                    hour for hour, activity in profile.hourly_activity.items()
                    if activity < avg_activity * 0.7
                ]
                
                # Create 4-hour windows from low-activity periods
                for start_hour in low_activity_hours:
                    end_hour = (start_hour + 4) % 24
                    fallbacks.append((start_hour, end_hour))
            
            # Add default fallback windows
            default_fallbacks = [(2, 6), (22, 2), (14, 18)]  # Common low-activity periods
            fallbacks.extend(default_fallbacks)
            
            # Remove duplicates and limit to 3 fallbacks
            unique_fallbacks = list(set(fallbacks))[:3]
            
            return unique_fallbacks
            
        except Exception as e:
            logger.error(f"Error generating fallback windows for agent {agent_id}: {e}")
            return [(2, 6), (22, 2)]  # Default fallbacks
    
    async def _assess_schedule_risks(self, agent_id: UUID, sleep_start: int, sleep_end: int) -> List[str]:
        """Assess risks associated with the proposed schedule."""
        risks = []
        
        try:
            profile = self._enhanced_profiles.get(agent_id)
            if not profile:
                risks.append("Insufficient historical data for risk assessment")
                return risks
            
            # Check if sleep window conflicts with peak activity
            if profile.peak_performance_hours:
                sleep_hours = set(range(sleep_start, sleep_end)) if sleep_end > sleep_start else set(range(sleep_start, 24)).union(range(0, sleep_end))
                peak_hours = set(profile.peak_performance_hours)
                
                if sleep_hours.intersection(peak_hours):
                    risks.append("Sleep window overlaps with peak performance hours")
            
            # Check workload trend
            trend = profile.calculate_workload_trend()
            if trend > 0.3:
                risks.append("Rapidly increasing workload may require shorter sleep duration")
            
            # Check coordination risks
            dependent_agents = self._coordination_graph.get(agent_id, set())
            if len(dependent_agents) > 3:
                risks.append("High number of dependent agents may complicate coordination")
            
            # Check adaptation score
            if profile.adaptation_score < 0.4:
                risks.append("Low adaptation score indicates potential optimization challenges")
            
        except Exception as e:
            logger.error(f"Error assessing schedule risks for agent {agent_id}: {e}")
            risks.append(f"Risk assessment error: {e}")
        
        return risks
    
    def _determine_implementation_priority(self, profile: ActivityProfile, predicted_reduction: float, risks: List[str]) -> str:
        """Determine implementation priority for the recommendation."""
        try:
            # High priority conditions
            if predicted_reduction > 0.6 and profile.adaptation_score > 0.7 and len(risks) <= 1:
                return "high"
            
            # Low priority conditions
            if predicted_reduction < 0.3 or profile.adaptation_score < 0.3 or len(risks) > 3:
                return "low"
            
            # Default to medium priority
            return "medium"
            
        except Exception as e:
            logger.error(f"Error determining implementation priority: {e}")
            return "medium"
    
    async def _get_coordination_requirements(self, agent_id: UUID) -> Dict[str, Any]:
        """Get coordination requirements for the agent."""
        try:
            dependent_agents = self._coordination_graph.get(agent_id, set())
            
            return {
                "dependent_agents": [str(aid) for aid in dependent_agents],
                "coordination_required": len(dependent_agents) > 0,
                "coordination_complexity": "high" if len(dependent_agents) > 2 else "low" if len(dependent_agents) == 0 else "medium"
            }
            
        except Exception as e:
            logger.error(f"Error getting coordination requirements for agent {agent_id}: {e}")
            return {}
    
    async def _validate_sleep_preconditions(self, agent_id: UUID, recommendation: ScheduleRecommendation) -> Dict[str, Any]:
        """Validate preconditions for sleep execution."""
        validation = {"valid": True, "errors": [], "warnings": []}
        
        try:
            async with get_async_session() as session:
                # Check agent status
                agent = await session.get(Agent, agent_id)
                if not agent:
                    validation["valid"] = False
                    validation["errors"].append("Agent not found")
                    return validation
                
                if agent.current_sleep_state != SleepState.AWAKE:
                    validation["valid"] = False
                    validation["errors"].append(f"Agent is not awake (current state: {agent.current_sleep_state})")
                    return validation
                
                if agent.status != AgentStatus.active:
                    validation["valid"] = False
                    validation["errors"].append(f"Agent is not active (status: {agent.status})")
                    return validation
                
                # Check for recent sleep cycles
                if agent.last_sleep_time:
                    time_since_sleep = datetime.utcnow() - agent.last_sleep_time
                    if time_since_sleep < timedelta(hours=2):
                        validation["valid"] = False
                        validation["errors"].append("Agent slept recently, minimum interval not met")
                        return validation
                
                # Check confidence threshold
                if recommendation.confidence_score < 0.5:
                    validation["warnings"].append("Low confidence score for recommendation")
                
                # Check risk factors
                if len(recommendation.risk_factors) > 2:
                    validation["warnings"].append("Multiple risk factors identified")
            
        except Exception as e:
            logger.error(f"Error validating sleep preconditions for agent {agent_id}: {e}")
            validation["valid"] = False
            validation["errors"].append(f"Validation error: {e}")
        
        return validation
    
    async def _execute_coordination_steps(self, agent_id: UUID, recommendation: ScheduleRecommendation) -> Dict[str, bool]:
        """Execute coordination steps with dependent agents."""
        coordination_status = {}
        
        try:
            dependent_agents = self._coordination_graph.get(agent_id, set())
            
            for dep_agent_id in dependent_agents:
                try:
                    # Notify dependent agent of upcoming sleep
                    redis_client = get_redis()
                    notification = {
                        "type": "coordinator_sleep_notification",
                        "coordinator_agent_id": str(agent_id),
                        "sleep_start_time": recommendation.recommended_sleep_start,
                        "sleep_end_time": recommendation.recommended_sleep_end,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    
                    await redis_client.lpush(
                        f"agent:{dep_agent_id}:notifications",
                        json.dumps(notification)
                    )
                    
                    coordination_status[str(dep_agent_id)] = True
                    
                except Exception as e:
                    logger.error(f"Error coordinating with agent {dep_agent_id}: {e}")
                    coordination_status[str(dep_agent_id)] = False
            
        except Exception as e:
            logger.error(f"Error executing coordination steps for agent {agent_id}: {e}")
        
        return coordination_status
    
    async def _update_system_efficiency_tracking(self, agent_id: UUID, recommendation: ScheduleRecommendation) -> None:
        """Update system-wide efficiency tracking."""
        try:
            current_hour = datetime.utcnow().hour
            efficiency_score = recommendation.confidence_score * recommendation.predicted_token_reduction
            
            self._system_efficiency_history.append({
                "hour": current_hour,
                "agent_id": str(agent_id),
                "efficiency_score": efficiency_score,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error updating system efficiency tracking: {e}")
    
    async def _calculate_current_system_efficiency(self) -> float:
        """Calculate current system efficiency."""
        try:
            if not self._system_efficiency_history:
                return 0.5  # Default
            
            recent_scores = [entry["efficiency_score"] for entry in list(self._system_efficiency_history)[-24:]]  # Last 24 hours
            return statistics.mean(recent_scores)
            
        except Exception as e:
            logger.error(f"Error calculating current system efficiency: {e}")
            return 0.5
    
    async def _calculate_predictive_accuracy(self) -> float:
        """Calculate accuracy of predictions vs actual results."""
        try:
            # This would require comparing predictions with actual results
            # For now, return a placeholder based on adaptation scores
            if self._enhanced_profiles:
                adaptation_scores = [profile.adaptation_score for profile in self._enhanced_profiles.values()]
                return statistics.mean(adaptation_scores)
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating predictive accuracy: {e}")
            return 0.5
    
    async def _calculate_coordination_effectiveness(self) -> float:
        """Calculate effectiveness of cross-agent coordination."""
        try:
            if not self._coordination_graph:
                return 1.0  # No coordination needed
            
            total_connections = sum(len(deps) for deps in self._coordination_graph.values())
            # Placeholder calculation - would need actual coordination success data
            return min(1.0, 0.8 + (total_connections * 0.05))
            
        except Exception as e:
            logger.error(f"Error calculating coordination effectiveness: {e}")
            return 0.8
    
    async def _predict_next_hour_efficiency(self) -> float:
        """Predict system efficiency for the next hour."""
        try:
            current_efficiency = await self._calculate_current_system_efficiency()
            
            # Simple prediction based on current trend
            if len(self._system_efficiency_history) >= 3:
                recent_scores = [entry["efficiency_score"] for entry in list(self._system_efficiency_history)[-3:]]
                trend = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
                predicted_efficiency = current_efficiency + trend
                return max(0.0, min(1.0, predicted_efficiency))
            
            return current_efficiency
            
        except Exception as e:
            logger.error(f"Error predicting next hour efficiency: {e}")
            return 0.5
    
    async def _calculate_optimal_system_sleep_window(self) -> Tuple[int, int]:
        """Calculate optimal sleep window for the entire system."""
        try:
            # Aggregate all agent activity patterns
            system_activity = defaultdict(int)
            
            for profile in self._enhanced_profiles.values():
                for hour, activity in profile.hourly_activity.items():
                    system_activity[hour] += activity
            
            if system_activity:
                # Find 4-hour window with lowest system activity
                min_activity = float('inf')
                optimal_start = 2
                
                for start_hour in range(24):
                    window_activity = sum(
                        system_activity[(start_hour + i) % 24]
                        for i in range(4)
                    )
                    if window_activity < min_activity:
                        min_activity = window_activity
                        optimal_start = start_hour
                
                return (optimal_start, (optimal_start + 4) % 24)
            
            return (2, 6)  # Default system sleep window
            
        except Exception as e:
            logger.error(f"Error calculating optimal system sleep window: {e}")
            return (2, 6)
    
    async def _identify_coordination_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for improved coordination."""
        try:
            opportunities = []
            
            # Find agents with overlapping peak hours that aren't coordinated
            for agent_id1, profile1 in self._enhanced_profiles.items():
                for agent_id2, profile2 in self._enhanced_profiles.items():
                    if agent_id1 >= agent_id2:  # Avoid duplicates
                        continue
                    
                    # Check if they have overlapping peak hours
                    peaks1 = set(profile1.peak_performance_hours)
                    peaks2 = set(profile2.peak_performance_hours)
                    overlap = peaks1.intersection(peaks2)
                    
                    if overlap and agent_id1 not in self._coordination_graph.get(agent_id2, set()):
                        opportunities.append({
                            "type": "peak_overlap_coordination",
                            "agents": [str(agent_id1), str(agent_id2)],
                            "overlapping_hours": list(overlap),
                            "potential_benefit": len(overlap) * 0.1  # Estimate
                        })
            
            return opportunities[:5]  # Limit to top 5 opportunities
            
        except Exception as e:
            logger.error(f"Error identifying coordination opportunities: {e}")
            return []
    
    async def _update_coordination_graph(self, agent_id: UUID) -> None:
        """Update coordination graph based on agent interactions."""
        try:
            # This would analyze agent interactions and dependencies
            # For now, use a simple heuristic based on task patterns
            
            async with get_async_session() as session:
                # Find agents that frequently work on similar tasks
                cutoff_date = datetime.utcnow() - timedelta(days=7)
                
                tasks_result = await session.execute(
                    select(Task).where(
                        and_(
                            Task.agent_id == agent_id,
                            Task.created_at >= cutoff_date
                        )
                    )
                )
                agent_tasks = tasks_result.scalars().all()
                
                # Simple coordination based on task timing patterns
                # In a real implementation, this would be much more sophisticated
                if agent_id not in self._coordination_graph:
                    self._coordination_graph[agent_id] = set()
                
                # For demo purposes, add some example dependencies
                if len(agent_tasks) > 10:  # High activity agents might coordinate
                    other_agents = await session.execute(select(Agent.id).where(Agent.id != agent_id))
                    other_agent_ids = [row[0] for row in other_agents.fetchall()]
                    
                    if other_agent_ids:
                        # Add up to 2 dependencies for demo
                        dependencies = other_agent_ids[:2]
                        self._coordination_graph[agent_id].update(dependencies)
            
        except Exception as e:
            logger.error(f"Error updating coordination graph for agent {agent_id}: {e}")


# Global intelligent sleep manager instance
_intelligent_sleep_manager_instance: Optional[IntelligentSleepManager] = None


def get_intelligent_sleep_manager() -> IntelligentSleepManager:
    """Get the global intelligent sleep manager instance."""
    global _intelligent_sleep_manager_instance
    if _intelligent_sleep_manager_instance is None:
        _intelligent_sleep_manager_instance = IntelligentSleepManager()
    return _intelligent_sleep_manager_instance


# Utility functions for intelligent sleep management
async def analyze_system_sleep_patterns() -> Dict[str, Any]:
    """Analyze system-wide sleep patterns and efficiency."""
    manager = get_intelligent_sleep_manager()
    return await manager.get_system_intelligence_metrics()


async def recommend_agent_optimization(agent_id: UUID) -> Dict[str, Any]:
    """Get comprehensive optimization recommendation for an agent."""
    manager = get_intelligent_sleep_manager()
    
    # Analyze patterns
    profile = await manager.analyze_activity_patterns(agent_id)
    
    # Get recommendation
    recommendation = await manager.recommend_sleep_schedule(agent_id)
    
    return {
        "agent_id": str(agent_id),
        "activity_profile": {
            "adaptation_score": profile.adaptation_score,
            "optimal_sleep_duration": profile.optimal_sleep_duration,
            "peak_hours": profile.peak_performance_hours,
            "workload_trend": profile.calculate_workload_trend()
        },
        "recommendation": {
            "sleep_window": f"{recommendation.recommended_sleep_start}:00-{recommendation.recommended_sleep_end}:00",
            "confidence": recommendation.confidence_score,
            "predicted_token_reduction": recommendation.predicted_token_reduction,
            "expected_performance_gain": recommendation.expected_performance_gain,
            "implementation_priority": recommendation.implementation_priority,
            "risk_factors": recommendation.risk_factors
        },
        "analysis_timestamp": datetime.utcnow().isoformat()
    }

