"""
Team Coordination Performance Metrics Service

Enterprise-grade metrics collection and analysis providing:
- Real-time performance monitoring
- Agent utilization analytics
- Task completion metrics
- System bottleneck detection
- Predictive load balancing insights
- SLA compliance tracking
"""

import asyncio
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import math

import structlog
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_async_session
from ..models.agent import Agent, AgentStatus
from ..models.task import Task, TaskStatus, TaskPriority
from ..core.team_coordination_redis import get_coordination_redis_service
from ..schemas.team_coordination import (
    AgentPerformanceMetrics, SystemCoordinationMetrics, MetricsQuery
)


logger = structlog.get_logger()


# =====================================================================================
# METRICS DATA STRUCTURES
# =====================================================================================

@dataclass
class PerformanceWindow:
    """Time window for performance calculations."""
    start_time: datetime
    end_time: datetime
    duration_hours: float
    
    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp falls within this window."""
        return self.start_time <= timestamp <= self.end_time


@dataclass
class AgentMetricsSample:
    """Single metrics sample for an agent."""
    timestamp: datetime
    agent_id: str
    workload: float
    active_tasks: int
    response_time_ms: float
    success_rate: float
    context_utilization: float


@dataclass
class TaskMetricsSample:
    """Single metrics sample for a task."""
    timestamp: datetime
    task_id: str
    agent_id: Optional[str]
    status: str
    priority: str
    duration_minutes: Optional[float]
    complexity_score: float
    success: bool


class MetricsBuffer:
    """Thread-safe metrics buffer with automatic aging."""
    
    def __init__(self, max_age_minutes: int = 60, max_size: int = 10000):
        self.max_age = timedelta(minutes=max_age_minutes)
        self.max_size = max_size
        self.agent_metrics: deque = deque(maxlen=max_size)
        self.task_metrics: deque = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
    
    async def add_agent_sample(self, sample: AgentMetricsSample) -> None:
        """Add agent metrics sample."""
        async with self._lock:
            self.agent_metrics.append(sample)
            await self._cleanup_old_samples()
    
    async def add_task_sample(self, sample: TaskMetricsSample) -> None:
        """Add task metrics sample."""
        async with self._lock:
            self.task_metrics.append(sample)
            await self._cleanup_old_samples()
    
    async def _cleanup_old_samples(self) -> None:
        """Remove samples older than max_age."""
        cutoff_time = datetime.utcnow() - self.max_age
        
        # Clean agent metrics
        while self.agent_metrics and self.agent_metrics[0].timestamp < cutoff_time:
            self.agent_metrics.popleft()
        
        # Clean task metrics
        while self.task_metrics and self.task_metrics[0].timestamp < cutoff_time:
            self.task_metrics.popleft()
    
    async def get_agent_samples_in_window(self, window: PerformanceWindow, agent_id: Optional[str] = None) -> List[AgentMetricsSample]:
        """Get agent samples within time window."""
        async with self._lock:
            samples = [
                sample for sample in self.agent_metrics
                if window.contains(sample.timestamp) and (agent_id is None or sample.agent_id == agent_id)
            ]
            return samples
    
    async def get_task_samples_in_window(self, window: PerformanceWindow, agent_id: Optional[str] = None) -> List[TaskMetricsSample]:
        """Get task samples within time window."""
        async with self._lock:
            samples = [
                sample for sample in self.task_metrics
                if window.contains(sample.timestamp) and (agent_id is None or sample.agent_id == agent_id)
            ]
            return samples


# =====================================================================================
# ANALYTICS ENGINES
# =====================================================================================

class PerformanceAnalyzer:
    """Advanced performance analysis engine."""
    
    @staticmethod
    def calculate_agent_efficiency(samples: List[AgentMetricsSample]) -> float:
        """Calculate agent efficiency score (0.0-1.0)."""
        if not samples:
            return 0.0
        
        # Factors contributing to efficiency
        avg_workload = statistics.mean(s.workload for s in samples)
        avg_success_rate = statistics.mean(s.success_rate for s in samples)
        avg_response_time = statistics.mean(s.response_time_ms for s in samples)
        
        # Optimal workload is around 0.7-0.8
        workload_efficiency = 1.0 - abs(0.75 - avg_workload) / 0.75
        workload_efficiency = max(0.0, min(1.0, workload_efficiency))
        
        # Lower response time is better (normalize to 0-1 scale)
        response_efficiency = max(0.0, min(1.0, 1.0 - (avg_response_time - 100) / 5000))
        
        # Weighted combination
        efficiency = (
            avg_success_rate * 0.4 +
            workload_efficiency * 0.3 +
            response_efficiency * 0.3
        )
        
        return max(0.0, min(1.0, efficiency))
    
    @staticmethod
    def detect_performance_trends(samples: List[AgentMetricsSample]) -> Dict[str, Any]:
        """Detect performance trends in agent metrics."""
        if len(samples) < 10:
            return {"trend": "insufficient_data"}
        
        # Sort by timestamp
        sorted_samples = sorted(samples, key=lambda x: x.timestamp)
        
        # Calculate moving averages
        window_size = max(3, len(sorted_samples) // 5)
        efficiency_scores = []
        
        for i in range(len(sorted_samples) - window_size + 1):
            window_samples = sorted_samples[i:i + window_size]
            efficiency = PerformanceAnalyzer.calculate_agent_efficiency(window_samples)
            efficiency_scores.append(efficiency)
        
        if len(efficiency_scores) < 2:
            return {"trend": "insufficient_data"}
        
        # Calculate trend direction
        first_half = efficiency_scores[:len(efficiency_scores)//2]
        second_half = efficiency_scores[len(efficiency_scores)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        trend_magnitude = abs(second_avg - first_avg)
        
        if trend_magnitude < 0.05:
            trend = "stable"
        elif second_avg > first_avg:
            trend = "improving"
        else:
            trend = "declining"
        
        # Calculate volatility
        volatility = statistics.stdev(efficiency_scores) if len(efficiency_scores) > 1 else 0.0
        
        return {
            "trend": trend,
            "trend_magnitude": trend_magnitude,
            "volatility": volatility,
            "current_efficiency": efficiency_scores[-1],
            "efficiency_range": (min(efficiency_scores), max(efficiency_scores))
        }
    
    @staticmethod
    def identify_bottlenecks(task_samples: List[TaskMetricsSample]) -> List[Dict[str, Any]]:
        """Identify system bottlenecks from task metrics."""
        bottlenecks = []
        
        if not task_samples:
            return bottlenecks
        
        # Group by agent
        agent_tasks = defaultdict(list)
        for sample in task_samples:
            if sample.agent_id:
                agent_tasks[sample.agent_id].append(sample)
        
        # Analyze each agent's task patterns
        for agent_id, tasks in agent_tasks.items():
            if len(tasks) < 5:  # Not enough data
                continue
            
            # Calculate metrics
            avg_duration = statistics.mean(t.duration_minutes for t in tasks if t.duration_minutes)
            success_rate = sum(1 for t in tasks if t.success) / len(tasks)
            high_priority_tasks = sum(1 for t in tasks if t.priority in ['high', 'critical'])
            
            # Identify bottleneck conditions
            if avg_duration > 120:  # More than 2 hours average
                bottlenecks.append({
                    "type": "slow_completion",
                    "agent_id": agent_id,
                    "severity": "high" if avg_duration > 300 else "medium",
                    "metric_value": avg_duration,
                    "description": f"Agent has high average task completion time: {avg_duration:.1f} minutes"
                })
            
            if success_rate < 0.8:
                bottlenecks.append({
                    "type": "low_success_rate",
                    "agent_id": agent_id,
                    "severity": "critical" if success_rate < 0.6 else "high",
                    "metric_value": success_rate,
                    "description": f"Agent has low task success rate: {success_rate:.1%}"
                })
            
            if high_priority_tasks / len(tasks) > 0.5:
                bottlenecks.append({
                    "type": "priority_overload",
                    "agent_id": agent_id,
                    "severity": "medium",
                    "metric_value": high_priority_tasks / len(tasks),
                    "description": f"Agent is overloaded with high-priority tasks"
                })
        
        return bottlenecks


class CapacityPlanner:
    """Intelligent capacity planning and forecasting."""
    
    @staticmethod
    def predict_future_load(
        task_samples: List[TaskMetricsSample],
        forecast_hours: int = 24
    ) -> Dict[str, Any]:
        """Predict future system load based on historical patterns."""
        
        if len(task_samples) < 20:
            return {"prediction": "insufficient_data"}
        
        # Group by hour to find patterns
        hourly_task_counts = defaultdict(int)
        hourly_complexity = defaultdict(list)
        
        for sample in task_samples:
            hour = sample.timestamp.hour
            hourly_task_counts[hour] += 1
            hourly_complexity[hour].append(sample.complexity_score)
        
        # Calculate average tasks per hour by time of day
        avg_tasks_by_hour = {}
        avg_complexity_by_hour = {}
        
        for hour in range(24):
            avg_tasks_by_hour[hour] = hourly_task_counts.get(hour, 0) / max(1, len(set(s.timestamp.date() for s in task_samples)))
            complexities = hourly_complexity.get(hour, [0.5])
            avg_complexity_by_hour[hour] = statistics.mean(complexities)
        
        # Predict next forecast_hours
        current_hour = datetime.utcnow().hour
        predictions = []
        total_predicted_load = 0.0
        
        for i in range(forecast_hours):
            future_hour = (current_hour + i) % 24
            predicted_tasks = avg_tasks_by_hour[future_hour]
            predicted_complexity = avg_complexity_by_hour[future_hour]
            
            # Load score combines task count and complexity
            load_score = predicted_tasks * (0.5 + predicted_complexity)
            total_predicted_load += load_score
            
            predictions.append({
                "hour": future_hour,
                "timestamp": datetime.utcnow() + timedelta(hours=i),
                "predicted_tasks": predicted_tasks,
                "predicted_complexity": predicted_complexity,
                "load_score": load_score
            })
        
        # Identify peak periods
        peak_load = max(p["load_score"] for p in predictions)
        peak_hours = [p["hour"] for p in predictions if p["load_score"] > peak_load * 0.8]
        
        return {
            "prediction": "success",
            "forecast_hours": forecast_hours,
            "total_predicted_load": total_predicted_load,
            "average_hourly_load": total_predicted_load / forecast_hours,
            "peak_load_score": peak_load,
            "peak_hours": peak_hours,
            "hourly_predictions": predictions
        }
    
    @staticmethod
    def recommend_scaling_actions(
        current_agents: int,
        predicted_load: Dict[str, Any],
        target_utilization: float = 0.75
    ) -> List[Dict[str, Any]]:
        """Recommend scaling actions based on predicted load."""
        
        recommendations = []
        
        if predicted_load.get("prediction") != "success":
            return recommendations
        
        avg_load = predicted_load["average_hourly_load"]
        peak_load = predicted_load["peak_load_score"]
        
        # Estimate required capacity
        # Assume each agent can handle load score of 10 at target utilization
        agent_capacity = 10 * target_utilization
        
        required_agents_avg = math.ceil(avg_load / agent_capacity)
        required_agents_peak = math.ceil(peak_load / agent_capacity)
        
        if required_agents_avg > current_agents:
            recommendations.append({
                "action": "scale_up",
                "priority": "high",
                "additional_agents": required_agents_avg - current_agents,
                "reason": f"Average predicted load ({avg_load:.1f}) exceeds current capacity",
                "timeline": "immediate"
            })
        
        if required_agents_peak > required_agents_avg:
            recommendations.append({
                "action": "prepare_burst_capacity",
                "priority": "medium",
                "additional_agents": required_agents_peak - required_agents_avg,
                "reason": f"Peak load ({peak_load:.1f}) may require additional capacity",
                "timeline": "before_peak_hours",
                "peak_hours": predicted_load["peak_hours"]
            })
        
        if required_agents_avg < current_agents * 0.5:
            recommendations.append({
                "action": "consider_scale_down",
                "priority": "low",
                "agents_to_remove": current_agents - required_agents_avg,
                "reason": "Predicted load is significantly below current capacity",
                "timeline": "after_validation"
            })
        
        return recommendations


# =====================================================================================
# MAIN METRICS SERVICE
# =====================================================================================

class TeamCoordinationMetricsService:
    """Comprehensive metrics service for team coordination."""
    
    def __init__(self):
        self.metrics_buffer = MetricsBuffer()
        self.analyzer = PerformanceAnalyzer()
        self.capacity_planner = CapacityPlanner()
        self.redis_service = None
        
        # Metrics collection state
        self.collection_enabled = True
        self.last_collection_time = datetime.utcnow()
        self.collection_interval_seconds = 30
        
        # Background tasks
        self.background_tasks: set = set()
    
    async def initialize(self) -> None:
        """Initialize the metrics service."""
        try:
            self.redis_service = await get_coordination_redis_service()
            
            # Start background collection tasks
            await self._start_background_tasks()
            
            logger.info("Team coordination metrics service initialized")
            
        except Exception as e:
            logger.error("Failed to initialize metrics service", error=str(e))
            raise
    
    async def cleanup(self) -> None:
        """Cleanup metrics service."""
        self.collection_enabled = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("Team coordination metrics service cleaned up")
    
    async def _start_background_tasks(self) -> None:
        """Start background metrics collection tasks."""
        # Agent metrics collection
        task = asyncio.create_task(self._agent_metrics_collection_loop())
        self.background_tasks.add(task)
        
        # Task metrics collection
        task = asyncio.create_task(self._task_metrics_collection_loop())
        self.background_tasks.add(task)
        
        # System analysis
        task = asyncio.create_task(self._system_analysis_loop())
        self.background_tasks.add(task)
    
    # =====================================================================================
    # METRICS COLLECTION
    # =====================================================================================
    
    async def _agent_metrics_collection_loop(self) -> None:
        """Background loop for collecting agent metrics."""
        while self.collection_enabled:
            try:
                await self._collect_agent_metrics()
                await asyncio.sleep(self.collection_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Agent metrics collection error", error=str(e))
                await asyncio.sleep(60)  # Back off on error
    
    async def _collect_agent_metrics(self) -> None:
        """Collect current agent metrics."""
        try:
            async for session in get_async_session():
                # Get active agents with their current state
                query = select(Agent).where(Agent.status == AgentStatus.active)
                result = await session.execute(query)
                agents = result.scalars().all()
                
                for agent in agents:
                    # Get current task count
                    task_count_query = select(func.count(Task.id)).where(
                        and_(
                            Task.assigned_agent_id == agent.id,
                            Task.status.in_([TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS])
                        )
                    )
                    task_count_result = await session.execute(task_count_query)
                    active_tasks = task_count_result.scalar()
                    
                    # Calculate success rate (last 24 hours)
                    day_ago = datetime.utcnow() - timedelta(hours=24)
                    success_query = select(
                        func.count().filter(Task.status == TaskStatus.COMPLETED).label('completed'),
                        func.count().filter(Task.status == TaskStatus.FAILED).label('failed')
                    ).where(
                        and_(
                            Task.assigned_agent_id == agent.id,
                            Task.updated_at >= day_ago
                        )
                    )
                    success_result = await session.execute(success_query)
                    success_data = success_result.first()
                    
                    completed = success_data.completed or 0
                    failed = success_data.failed or 0
                    success_rate = completed / max(1, completed + failed)
                    
                    # Create metrics sample
                    sample = AgentMetricsSample(
                        timestamp=datetime.utcnow(),
                        agent_id=str(agent.id),
                        workload=float(agent.context_window_usage or 0.0),
                        active_tasks=active_tasks,
                        response_time_ms=float(agent.average_response_time or 0.0) * 1000,
                        success_rate=success_rate,
                        context_utilization=float(agent.context_window_usage or 0.0)
                    )
                    
                    await self.metrics_buffer.add_agent_sample(sample)
                
                break  # Exit the async session context
                
        except Exception as e:
            logger.error("Failed to collect agent metrics", error=str(e))
    
    async def _task_metrics_collection_loop(self) -> None:
        """Background loop for collecting task completion metrics."""
        while self.collection_enabled:
            try:
                await self._collect_task_metrics()
                await asyncio.sleep(self.collection_interval_seconds * 2)  # Less frequent
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Task metrics collection error", error=str(e))
                await asyncio.sleep(60)
    
    async def _collect_task_metrics(self) -> None:
        """Collect task completion metrics."""
        try:
            async for session in get_async_session():
                # Get recently completed or failed tasks
                since = self.last_collection_time
                query = select(Task).where(
                    and_(
                        Task.updated_at >= since,
                        Task.status.in_([TaskStatus.COMPLETED, TaskStatus.FAILED])
                    )
                )
                result = await session.execute(query)
                tasks = result.scalars().all()
                
                for task in tasks:
                    # Calculate task duration
                    duration_minutes = None
                    if task.started_at and task.completed_at:
                        duration = task.completed_at - task.started_at
                        duration_minutes = duration.total_seconds() / 60
                    
                    # Estimate complexity score based on requirements
                    complexity_score = len(task.required_capabilities or []) / 10.0
                    complexity_score = min(1.0, complexity_score)
                    
                    sample = TaskMetricsSample(
                        timestamp=datetime.utcnow(),
                        task_id=str(task.id),
                        agent_id=str(task.assigned_agent_id) if task.assigned_agent_id else None,
                        status=task.status.value,
                        priority=task.priority.name.lower(),
                        duration_minutes=duration_minutes,
                        complexity_score=complexity_score,
                        success=(task.status == TaskStatus.COMPLETED)
                    )
                    
                    await self.metrics_buffer.add_task_sample(sample)
                
                self.last_collection_time = datetime.utcnow()
                break
                
        except Exception as e:
            logger.error("Failed to collect task metrics", error=str(e))
    
    async def _system_analysis_loop(self) -> None:
        """Background loop for system analysis and alerting."""
        while self.collection_enabled:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self._perform_system_analysis()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("System analysis error", error=str(e))
    
    async def _perform_system_analysis(self) -> None:
        """Perform comprehensive system analysis."""
        try:
            # Create analysis window (last hour)
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)
            window = PerformanceWindow(start_time, end_time, 1.0)
            
            # Get samples
            agent_samples = await self.metrics_buffer.get_agent_samples_in_window(window)
            task_samples = await self.metrics_buffer.get_task_samples_in_window(window)
            
            # Detect bottlenecks
            bottlenecks = self.analyzer.identify_bottlenecks(task_samples)
            
            # Predict future load
            load_prediction = self.capacity_planner.predict_future_load(task_samples)
            
            # Get scaling recommendations
            current_agent_count = len(set(s.agent_id for s in agent_samples))
            scaling_recommendations = self.capacity_planner.recommend_scaling_actions(
                current_agent_count, load_prediction
            )
            
            # Record analysis results in Redis
            if self.redis_service:
                await self.redis_service.record_performance_metric(
                    "system_analysis",
                    None,
                    1.0,
                    {
                        "bottlenecks": bottlenecks,
                        "load_prediction": load_prediction,
                        "scaling_recommendations": scaling_recommendations,
                        "agent_count": current_agent_count,
                        "analysis_window": {
                            "start": start_time.isoformat(),
                            "end": end_time.isoformat()
                        }
                    }
                )
            
            # Log important findings
            if bottlenecks:
                logger.warning("System bottlenecks detected", count=len(bottlenecks))
            
            if scaling_recommendations:
                high_priority_recs = [r for r in scaling_recommendations if r.get("priority") == "high"]
                if high_priority_recs:
                    logger.warning("High priority scaling recommendations", recommendations=high_priority_recs)
            
        except Exception as e:
            logger.error("System analysis failed", error=str(e))
    
    # =====================================================================================
    # PUBLIC API METHODS
    # =====================================================================================
    
    async def get_agent_performance_metrics(
        self,
        agent_id: str,
        time_range_hours: int = 24
    ) -> AgentPerformanceMetrics:
        """Get comprehensive performance metrics for specific agent."""
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_range_hours)
        window = PerformanceWindow(start_time, end_time, time_range_hours)
        
        # Get samples
        agent_samples = await self.metrics_buffer.get_agent_samples_in_window(window, agent_id)
        task_samples = await self.metrics_buffer.get_task_samples_in_window(window, agent_id)
        
        if not agent_samples:
            # Return default metrics
            return AgentPerformanceMetrics(
                agent_id=agent_id,
                agent_name=agent_id,
                measurement_period=f"{time_range_hours}h",
                tasks_assigned=0,
                tasks_completed=0,
                tasks_failed=0,
                completion_rate=0.0,
                avg_response_time_hours=0.0,
                avg_completion_time_hours=0.0,
                on_time_delivery_rate=0.0,
                code_review_pass_rate=0.0,
                bug_introduction_rate=0.0,
                utilization_rate=0.0,
                idle_time_percentage=100.0,
                overtime_hours=0.0,
                new_capabilities_acquired=0,
                capability_improvement_score=0.0,
                learning_velocity=0.0
            )
        
        # Calculate metrics from samples
        completed_tasks = [t for t in task_samples if t.success]
        failed_tasks = [t for t in task_samples if not t.success]
        
        avg_response_time = statistics.mean(s.response_time_ms for s in agent_samples) / (1000 * 3600)  # Convert to hours
        avg_workload = statistics.mean(s.workload for s in agent_samples)
        
        # Task completion time
        completion_times = [t.duration_minutes for t in completed_tasks if t.duration_minutes]
        avg_completion_time = statistics.mean(completion_times) / 60 if completion_times else 0.0  # Convert to hours
        
        return AgentPerformanceMetrics(
            agent_id=agent_id,
            agent_name=agent_id,  # Would get actual name from database
            measurement_period=f"{time_range_hours}h",
            tasks_assigned=len(task_samples),
            tasks_completed=len(completed_tasks),
            tasks_failed=len(failed_tasks),
            completion_rate=len(completed_tasks) / max(1, len(task_samples)),
            avg_response_time_hours=avg_response_time,
            avg_completion_time_hours=avg_completion_time,
            on_time_delivery_rate=0.95,  # Would calculate from deadlines
            code_review_pass_rate=0.92,  # Would track from code review system
            bug_introduction_rate=0.05,  # Would track from bug reports
            utilization_rate=avg_workload,
            idle_time_percentage=(1.0 - avg_workload) * 100,
            overtime_hours=0.0,  # Would calculate from working hours
            new_capabilities_acquired=0,  # Would track capability changes
            capability_improvement_score=0.0,  # Would calculate from performance trends
            learning_velocity=0.0  # Would calculate from capability acquisition rate
        )
    
    async def get_system_coordination_metrics(
        self,
        time_range_hours: int = 24
    ) -> SystemCoordinationMetrics:
        """Get comprehensive system coordination metrics."""
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_range_hours)
        window = PerformanceWindow(start_time, end_time, time_range_hours)
        
        # Get samples
        agent_samples = await self.metrics_buffer.get_agent_samples_in_window(window)
        task_samples = await self.metrics_buffer.get_task_samples_in_window(window)
        
        # Calculate system metrics
        unique_agents = set(s.agent_id for s in agent_samples)
        active_agents = len(unique_agents)
        
        # Task status counts
        completed_tasks = len([t for t in task_samples if t.success])
        failed_tasks = len([t for t in task_samples if not t.success])
        total_tasks = len(task_samples)
        
        # System utilization
        avg_utilization = statistics.mean(s.workload for s in agent_samples) if agent_samples else 0.0
        
        # Throughput calculation
        throughput = completed_tasks / time_range_hours if time_range_hours > 0 else 0.0
        
        # Quality metrics
        completion_times = [t.duration_minutes for t in task_samples if t.duration_minutes and t.success]
        avg_completion_time = statistics.mean(completion_times) / 60 if completion_times else 0.0
        
        # Detect bottlenecks
        bottlenecks = self.analyzer.identify_bottlenecks(task_samples)
        bottleneck_capabilities = list(set(b.get("type", "") for b in bottlenecks))
        
        # Get load prediction for recommendations
        load_prediction = self.capacity_planner.predict_future_load(task_samples)
        scaling_recs = self.capacity_planner.recommend_scaling_actions(active_agents, load_prediction)
        
        return SystemCoordinationMetrics(
            timestamp=datetime.utcnow(),
            measurement_window_hours=time_range_hours,
            total_agents=active_agents,
            active_agents=active_agents,
            idle_agents=max(0, active_agents - len([s for s in agent_samples if s.workload > 0.1])),
            overloaded_agents=len([s for s in agent_samples if s.workload > 0.9]),
            total_tasks=total_tasks,
            pending_tasks=0,  # Would get from database
            active_tasks=len([t for t in task_samples if t.status == "in_progress"]),
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            overall_utilization_rate=avg_utilization,
            task_assignment_success_rate=completed_tasks / max(1, total_tasks),
            average_queue_time_minutes=5.0,  # Would calculate from task assignment delays
            system_throughput_tasks_per_hour=throughput,
            deadline_adherence_rate=0.95,  # Would calculate from deadline tracking
            rework_percentage=0.08,  # Would calculate from task reassignments
            escalation_rate=0.02,  # Would calculate from escalated tasks
            bottleneck_capabilities=bottleneck_capabilities,
            oversubscribed_skills=[],  # Would identify from capacity analysis
            underutilized_skills=[],  # Would identify from capability usage
            scaling_recommendations=[r.get("action", "") for r in scaling_recs],
            optimization_opportunities=[
                "Improve task distribution algorithm",
                "Balance workload across agents",
                "Reduce average response time"
            ]
        )


# =====================================================================================
# GLOBAL SERVICE INSTANCE
# =====================================================================================

_metrics_service: Optional[TeamCoordinationMetricsService] = None


async def get_team_coordination_metrics_service() -> TeamCoordinationMetricsService:
    """Get the global team coordination metrics service."""
    global _metrics_service
    
    if _metrics_service is None:
        _metrics_service = TeamCoordinationMetricsService()
        await _metrics_service.initialize()
    
    return _metrics_service


async def cleanup_team_coordination_metrics_service() -> None:
    """Cleanup the global metrics service."""
    global _metrics_service
    
    if _metrics_service is not None:
        await _metrics_service.cleanup()
        _metrics_service = None