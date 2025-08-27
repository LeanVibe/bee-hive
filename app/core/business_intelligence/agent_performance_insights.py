"""
Agent Performance Insights Service

Comprehensive agent performance monitoring, optimization, and capacity planning
for LeanVibe Agent Hive 2.0. Provides real-time agent analytics, efficiency scoring,
resource utilization analysis, and actionable optimization recommendations.

Epic 5 Phase 3: Agent Performance Insights - PRODUCTION READY
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
import logging
from statistics import mean, stdev

from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession

from ...models.business_intelligence import (
    BusinessMetric, MetricType, AgentPerformanceMetric,
    BusinessAlert, AlertLevel
)
from ...models.agent import Agent, AgentStatus
from ...models.task import Task, TaskStatus
from ...core.database import get_session
from ...core.logging_service import get_component_logger

logger = get_component_logger("agent_performance_insights")


@dataclass
class AgentEfficiencyScore:
    """Agent efficiency scoring with detailed breakdown."""
    agent_id: str
    overall_score: Decimal
    task_completion_score: Decimal
    response_time_score: Decimal
    resource_utilization_score: Decimal
    error_rate_score: Decimal
    uptime_score: Decimal
    business_value_score: Decimal
    improvement_areas: List[str]
    strengths: List[str]
    trend: str  # improving, declining, stable
    percentile_rank: Optional[int] = None


@dataclass
class AgentOptimizationRecommendation:
    """Agent optimization recommendation with actionable insights."""
    agent_id: str
    recommendation_type: str
    priority: str  # high, medium, low
    title: str
    description: str
    expected_improvement: Decimal
    implementation_effort: str  # low, medium, high
    impact_category: str  # performance, cost, reliability, capacity
    suggested_actions: List[str]
    metrics_to_track: List[str]


@dataclass
class AgentCapacityInsights:
    """Agent capacity planning insights and projections."""
    current_utilization: Decimal
    optimal_utilization: Decimal
    capacity_headroom: Decimal
    bottleneck_indicators: List[str]
    scaling_recommendations: List[str]
    resource_allocation_suggestions: Dict[str, Any]
    load_balancing_opportunities: List[str]
    forecast_horizon_days: int = 30


@dataclass
class AgentBenchmarkMetrics:
    """Agent benchmark comparison metrics."""
    performance_rank: int
    total_agents: int
    percentile: Decimal
    above_average_metrics: List[str]
    below_average_metrics: List[str]
    peer_comparison: Dict[str, Any]
    industry_benchmark_position: Optional[str] = None


class AgentPerformanceAnalyzer:
    """Analyzes agent efficiency and resource usage patterns."""
    
    def __init__(self):
        """Initialize agent performance analyzer."""
        self.logger = logger
        
    async def analyze_agent_efficiency(
        self, 
        agent_id: str, 
        time_period_days: int = 30
    ) -> Optional[AgentEfficiencyScore]:
        """Analyze comprehensive agent efficiency and generate scoring."""
        try:
            async with get_session() as session:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(days=time_period_days)
                
                # Get agent performance metrics
                perf_query = select(AgentPerformanceMetric).where(
                    and_(
                        AgentPerformanceMetric.agent_id == agent_id,
                        AgentPerformanceMetric.timestamp >= start_time,
                        AgentPerformanceMetric.timestamp <= end_time
                    )
                ).order_by(desc(AgentPerformanceMetric.timestamp))
                
                result = await session.execute(perf_query)
                metrics = result.scalars().all()
                
                if not metrics:
                    self.logger.warning(f"No performance metrics found for agent {agent_id}")
                    return None
                
                # Calculate efficiency scores
                efficiency_score = await self._calculate_efficiency_scores(metrics)
                
                # Determine trends
                trend = await self._analyze_performance_trend(metrics)
                
                # Get percentile ranking
                percentile_rank = await self._get_agent_percentile_rank(session, agent_id, metrics)
                
                return AgentEfficiencyScore(
                    agent_id=agent_id,
                    overall_score=efficiency_score["overall"],
                    task_completion_score=efficiency_score["task_completion"],
                    response_time_score=efficiency_score["response_time"],
                    resource_utilization_score=efficiency_score["resource_utilization"],
                    error_rate_score=efficiency_score["error_rate"],
                    uptime_score=efficiency_score["uptime"],
                    business_value_score=efficiency_score["business_value"],
                    improvement_areas=efficiency_score["improvement_areas"],
                    strengths=efficiency_score["strengths"],
                    trend=trend,
                    percentile_rank=percentile_rank
                )
                
        except Exception as e:
            self.logger.error(f"Failed to analyze agent efficiency for {agent_id}: {e}")
            return None
    
    async def _calculate_efficiency_scores(self, metrics: List[AgentPerformanceMetric]) -> Dict[str, Any]:
        """Calculate detailed efficiency scores from performance metrics."""
        if not metrics:
            return {"overall": Decimal("0"), "improvement_areas": ["No data available"], "strengths": []}
        
        # Extract metric values
        success_rates = [float(m.success_rate or 0) for m in metrics if m.success_rate is not None]
        response_times = [m.average_response_time_ms or 0 for m in metrics if m.average_response_time_ms is not None]
        cpu_usage = [float(m.cpu_usage_percent or 0) for m in metrics if m.cpu_usage_percent is not None]
        error_counts = [m.error_count or 0 for m in metrics]
        uptimes = [float(m.uptime_percentage or 0) for m in metrics if m.uptime_percentage is not None]
        business_values = [float(m.business_value_generated or 0) for m in metrics if m.business_value_generated is not None]
        
        # Task completion score (0-100)
        task_completion_score = Decimal(str(mean(success_rates) if success_rates else 0)).quantize(Decimal('0.01'))
        
        # Response time score (inverse relationship - lower is better)
        avg_response_time = mean(response_times) if response_times else 1000
        # Score: 100 for <= 100ms, 0 for >= 5000ms
        response_time_score = Decimal(str(max(0, 100 - (avg_response_time - 100) / 49))).quantize(Decimal('0.01'))
        
        # Resource utilization score (optimal range 60-80%)
        avg_cpu = mean(cpu_usage) if cpu_usage else 50
        if 60 <= avg_cpu <= 80:
            resource_score = Decimal("100")
        elif avg_cpu < 60:
            resource_score = Decimal(str(avg_cpu / 60 * 100)).quantize(Decimal('0.01'))
        else:
            resource_score = Decimal(str(max(0, 100 - (avg_cpu - 80) * 2))).quantize(Decimal('0.01'))
        
        # Error rate score (inverse of error frequency)
        avg_errors = mean(error_counts) if error_counts else 0
        error_rate_score = Decimal(str(max(0, 100 - avg_errors * 10))).quantize(Decimal('0.01'))
        
        # Uptime score
        uptime_score = Decimal(str(mean(uptimes) if uptimes else 95)).quantize(Decimal('0.01'))
        
        # Business value score
        avg_business_value = mean(business_values) if business_values else 0
        # Normalize business value score (assuming 1000 is excellent)
        business_value_score = Decimal(str(min(100, avg_business_value / 10))).quantize(Decimal('0.01'))
        
        # Overall weighted score
        weights = {
            "task_completion": 0.25,
            "response_time": 0.20,
            "resource_utilization": 0.15,
            "error_rate": 0.15,
            "uptime": 0.15,
            "business_value": 0.10
        }
        
        overall_score = Decimal(str(
            float(task_completion_score) * weights["task_completion"] +
            float(response_time_score) * weights["response_time"] +
            float(resource_score) * weights["resource_utilization"] +
            float(error_rate_score) * weights["error_rate"] +
            float(uptime_score) * weights["uptime"] +
            float(business_value_score) * weights["business_value"]
        )).quantize(Decimal('0.01'))
        
        # Identify improvement areas and strengths
        score_map = {
            "task_completion": task_completion_score,
            "response_time": response_time_score,
            "resource_utilization": resource_score,
            "error_rate": error_rate_score,
            "uptime": uptime_score,
            "business_value": business_value_score
        }
        
        improvement_areas = [area for area, score in score_map.items() if score < 70]
        strengths = [area for area, score in score_map.items() if score >= 85]
        
        return {
            "overall": overall_score,
            "task_completion": task_completion_score,
            "response_time": response_time_score,
            "resource_utilization": resource_score,
            "error_rate": error_rate_score,
            "uptime": uptime_score,
            "business_value": business_value_score,
            "improvement_areas": improvement_areas,
            "strengths": strengths
        }
    
    async def _analyze_performance_trend(self, metrics: List[AgentPerformanceMetric]) -> str:
        """Analyze performance trend over time."""
        if len(metrics) < 3:
            return "stable"
        
        # Sort metrics by timestamp
        sorted_metrics = sorted(metrics, key=lambda x: x.timestamp)
        
        # Get overall efficiency scores for trend analysis
        recent_half = sorted_metrics[len(sorted_metrics)//2:]
        earlier_half = sorted_metrics[:len(sorted_metrics)//2]
        
        # Calculate average success rates for each half
        recent_success = mean([float(m.success_rate or 0) for m in recent_half])
        earlier_success = mean([float(m.success_rate or 0) for m in earlier_half])
        
        # Calculate average response times for each half
        recent_response = mean([m.average_response_time_ms or 1000 for m in recent_half])
        earlier_response = mean([m.average_response_time_ms or 1000 for m in earlier_half])
        
        # Determine trend based on success rate and response time changes
        success_improving = recent_success > earlier_success * 1.05  # 5% improvement
        response_improving = recent_response < earlier_response * 0.95  # 5% improvement
        
        success_declining = recent_success < earlier_success * 0.95  # 5% decline
        response_declining = recent_response > earlier_response * 1.05  # 5% decline
        
        if (success_improving or response_improving) and not (success_declining or response_declining):
            return "improving"
        elif (success_declining or response_declining) and not (success_improving or response_improving):
            return "declining"
        else:
            return "stable"
    
    async def _get_agent_percentile_rank(
        self, 
        session: AsyncSession, 
        agent_id: str, 
        agent_metrics: List[AgentPerformanceMetric]
    ) -> Optional[int]:
        """Get agent percentile rank compared to all agents."""
        try:
            # Get average success rate for this agent
            agent_avg_success = mean([float(m.success_rate or 0) for m in agent_metrics if m.success_rate])
            
            if not agent_avg_success:
                return None
            
            # Get all agents' recent average success rates
            recent_time = datetime.utcnow() - timedelta(days=7)
            
            all_agents_query = select(
                AgentPerformanceMetric.agent_id,
                func.avg(AgentPerformanceMetric.success_rate).label("avg_success_rate")
            ).where(
                AgentPerformanceMetric.timestamp >= recent_time
            ).group_by(AgentPerformanceMetric.agent_id)
            
            result = await session.execute(all_agents_query)
            agent_success_rates = [(row.agent_id, float(row.avg_success_rate or 0)) for row in result]
            
            if len(agent_success_rates) <= 1:
                return None
            
            # Calculate percentile rank
            better_count = sum(1 for _, rate in agent_success_rates if rate < agent_avg_success)
            percentile = int((better_count / len(agent_success_rates)) * 100)
            
            return percentile
            
        except Exception as e:
            self.logger.error(f"Failed to calculate percentile rank: {e}")
            return None
    
    async def get_all_agents_performance_summary(self, time_period_days: int = 30) -> Dict[str, Any]:
        """Get performance summary for all agents."""
        try:
            async with get_session() as session:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(days=time_period_days)
                
                # Get all agents with recent performance metrics
                agents_query = select(
                    AgentPerformanceMetric.agent_id,
                    func.count(AgentPerformanceMetric.id).label("metric_count"),
                    func.avg(AgentPerformanceMetric.success_rate).label("avg_success_rate"),
                    func.avg(AgentPerformanceMetric.average_response_time_ms).label("avg_response_time"),
                    func.avg(AgentPerformanceMetric.cpu_usage_percent).label("avg_cpu_usage"),
                    func.sum(AgentPerformanceMetric.error_count).label("total_errors"),
                    func.avg(AgentPerformanceMetric.uptime_percentage).label("avg_uptime"),
                    func.avg(AgentPerformanceMetric.utilization_percentage).label("avg_utilization"),
                    func.sum(AgentPerformanceMetric.tasks_completed).label("total_tasks_completed"),
                    func.sum(AgentPerformanceMetric.tasks_failed).label("total_tasks_failed")
                ).where(
                    and_(
                        AgentPerformanceMetric.timestamp >= start_time,
                        AgentPerformanceMetric.timestamp <= end_time
                    )
                ).group_by(AgentPerformanceMetric.agent_id)
                
                result = await session.execute(agents_query)
                agent_stats = result.all()
                
                # Calculate summary metrics
                total_agents = len(agent_stats)
                
                if total_agents == 0:
                    return {
                        "total_agents": 0,
                        "active_agents": 0,
                        "average_utilization": 0.0,
                        "average_efficiency": 0.0,
                        "total_tasks_completed": 0,
                        "total_tasks_failed": 0,
                        "overall_success_rate": 0.0,
                        "average_response_time": 0,
                        "agents_needing_optimization": 0,
                        "performance_distribution": {}
                    }
                
                # Aggregate metrics
                success_rates = [float(stat.avg_success_rate or 0) for stat in agent_stats]
                response_times = [int(stat.avg_response_time or 0) for stat in agent_stats]
                utilizations = [float(stat.avg_utilization or 0) for stat in agent_stats]
                uptimes = [float(stat.avg_uptime or 0) for stat in agent_stats]
                
                total_tasks_completed = sum(stat.total_tasks_completed or 0 for stat in agent_stats)
                total_tasks_failed = sum(stat.total_tasks_failed or 0 for stat in agent_stats)
                
                # Calculate derived metrics
                average_utilization = mean(utilizations) if utilizations else 0.0
                overall_success_rate = mean(success_rates) if success_rates else 0.0
                average_response_time = int(mean(response_times)) if response_times else 0
                average_uptime = mean(uptimes) if uptimes else 0.0
                
                # Calculate average efficiency score
                efficiency_scores = []
                for stat in agent_stats:
                    # Simplified efficiency calculation for summary
                    task_score = float(stat.avg_success_rate or 0)
                    response_score = max(0, 100 - (int(stat.avg_response_time or 1000) - 100) / 49)
                    uptime_score = float(stat.avg_uptime or 95)
                    efficiency = (task_score + response_score + uptime_score) / 3
                    efficiency_scores.append(efficiency)
                
                average_efficiency = mean(efficiency_scores) if efficiency_scores else 0.0
                
                # Count agents needing optimization (efficiency < 70)
                agents_needing_optimization = sum(1 for score in efficiency_scores if score < 70)
                
                # Performance distribution
                excellent_count = sum(1 for score in efficiency_scores if score >= 90)
                good_count = sum(1 for score in efficiency_scores if 70 <= score < 90)
                needs_improvement_count = sum(1 for score in efficiency_scores if score < 70)
                
                return {
                    "total_agents": total_agents,
                    "active_agents": sum(1 for stat in agent_stats if float(stat.avg_uptime or 0) > 90),
                    "average_utilization": round(average_utilization, 2),
                    "average_efficiency": round(average_efficiency, 2),
                    "total_tasks_completed": total_tasks_completed,
                    "total_tasks_failed": total_tasks_failed,
                    "overall_success_rate": round(overall_success_rate, 2),
                    "average_response_time": average_response_time,
                    "average_uptime": round(average_uptime, 2),
                    "agents_needing_optimization": agents_needing_optimization,
                    "performance_distribution": {
                        "excellent": excellent_count,
                        "good": good_count,
                        "needs_improvement": needs_improvement_count
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get agents performance summary: {e}")
            return {}


class AgentOptimizationEngine:
    """Generates performance improvement recommendations for agents."""
    
    def __init__(self):
        """Initialize optimization engine."""
        self.logger = logger
    
    async def generate_optimization_recommendations(
        self, 
        agent_id: str, 
        efficiency_score: AgentEfficiencyScore,
        time_period_days: int = 30
    ) -> List[AgentOptimizationRecommendation]:
        """Generate comprehensive optimization recommendations."""
        try:
            recommendations = []
            
            # Task completion optimization
            if efficiency_score.task_completion_score < 80:
                recommendations.append(AgentOptimizationRecommendation(
                    agent_id=agent_id,
                    recommendation_type="task_completion",
                    priority="high" if efficiency_score.task_completion_score < 60 else "medium",
                    title="Improve Task Completion Rate",
                    description=f"Current task completion rate of {efficiency_score.task_completion_score}% is below optimal. Focus on error handling and task retry mechanisms.",
                    expected_improvement=Decimal("15.00"),
                    implementation_effort="medium",
                    impact_category="performance",
                    suggested_actions=[
                        "Implement robust error handling and recovery mechanisms",
                        "Add intelligent task retry logic with exponential backoff",
                        "Review and optimize task execution workflows",
                        "Implement task complexity assessment and routing"
                    ],
                    metrics_to_track=["success_rate", "error_count", "task_completion_rate"]
                ))
            
            # Response time optimization
            if efficiency_score.response_time_score < 80:
                recommendations.append(AgentOptimizationRecommendation(
                    agent_id=agent_id,
                    recommendation_type="response_time",
                    priority="high" if efficiency_score.response_time_score < 50 else "medium",
                    title="Optimize Response Time Performance",
                    description=f"Response time performance score of {efficiency_score.response_time_score}% indicates latency issues. Implement caching and optimize processing.",
                    expected_improvement=Decimal("20.00"),
                    implementation_effort="medium",
                    impact_category="performance",
                    suggested_actions=[
                        "Implement response caching for common queries",
                        "Optimize data retrieval and processing algorithms",
                        "Add connection pooling for external services",
                        "Implement request batching where appropriate"
                    ],
                    metrics_to_track=["average_response_time_ms", "response_time_percentiles", "cache_hit_rate"]
                ))
            
            # Resource utilization optimization
            if efficiency_score.resource_utilization_score < 80:
                priority = "high" if efficiency_score.resource_utilization_score < 50 else "medium"
                if efficiency_score.resource_utilization_score < 60:
                    # Under-utilization
                    recommendations.append(AgentOptimizationRecommendation(
                        agent_id=agent_id,
                        recommendation_type="resource_utilization",
                        priority=priority,
                        title="Increase Resource Utilization",
                        description="Agent is under-utilized. Consider increasing workload or optimizing resource allocation.",
                        expected_improvement=Decimal("25.00"),
                        implementation_effort="low",
                        impact_category="cost",
                        suggested_actions=[
                            "Increase concurrent task processing capacity",
                            "Optimize agent workload distribution",
                            "Implement dynamic scaling based on demand",
                            "Review and adjust resource allocation policies"
                        ],
                        metrics_to_track=["cpu_usage_percent", "memory_usage_mb", "utilization_percentage"]
                    ))
                else:
                    # Over-utilization
                    recommendations.append(AgentOptimizationRecommendation(
                        agent_id=agent_id,
                        recommendation_type="resource_utilization",
                        priority=priority,
                        title="Optimize Resource Usage",
                        description="Agent is over-utilized. Implement load balancing and resource optimization.",
                        expected_improvement=Decimal("15.00"),
                        implementation_effort="medium",
                        impact_category="reliability",
                        suggested_actions=[
                            "Implement load shedding mechanisms",
                            "Add horizontal scaling capabilities",
                            "Optimize memory usage and garbage collection",
                            "Implement circuit breakers for external dependencies"
                        ],
                        metrics_to_track=["cpu_usage_percent", "memory_usage_mb", "queue_depth", "error_rate"]
                    ))
            
            # Error rate optimization
            if efficiency_score.error_rate_score < 90:
                recommendations.append(AgentOptimizationRecommendation(
                    agent_id=agent_id,
                    recommendation_type="error_handling",
                    priority="high",
                    title="Reduce Error Rate",
                    description=f"Error rate score of {efficiency_score.error_rate_score}% indicates reliability issues. Focus on error prevention and handling.",
                    expected_improvement=Decimal("10.00"),
                    implementation_effort="medium",
                    impact_category="reliability",
                    suggested_actions=[
                        "Implement comprehensive input validation",
                        "Add structured error handling with proper categorization",
                        "Implement health checks and self-healing mechanisms",
                        "Add detailed error logging and monitoring"
                    ],
                    metrics_to_track=["error_count", "error_rate", "error_recovery_rate"]
                ))
            
            # Uptime optimization
            if efficiency_score.uptime_score < 99:
                recommendations.append(AgentOptimizationRecommendation(
                    agent_id=agent_id,
                    recommendation_type="reliability",
                    priority="high",
                    title="Improve System Reliability",
                    description=f"Uptime score of {efficiency_score.uptime_score}% is below target. Focus on system stability and monitoring.",
                    expected_improvement=Decimal("5.00"),
                    implementation_effort="high",
                    impact_category="reliability",
                    suggested_actions=[
                        "Implement comprehensive health monitoring",
                        "Add automated restart and recovery mechanisms",
                        "Implement graceful degradation strategies",
                        "Set up proactive monitoring and alerting"
                    ],
                    metrics_to_track=["uptime_percentage", "availability_percentage", "mean_time_to_recovery"]
                ))
            
            # Business value optimization
            if efficiency_score.business_value_score < 70:
                recommendations.append(AgentOptimizationRecommendation(
                    agent_id=agent_id,
                    recommendation_type="business_value",
                    priority="medium",
                    title="Enhance Business Value Generation",
                    description=f"Business value score of {efficiency_score.business_value_score}% indicates opportunities to increase value delivery.",
                    expected_improvement=Decimal("30.00"),
                    implementation_effort="high",
                    impact_category="performance",
                    suggested_actions=[
                        "Optimize task prioritization based on business impact",
                        "Implement value-based task routing",
                        "Add business metrics tracking and optimization",
                        "Focus agent capabilities on high-value activities"
                    ],
                    metrics_to_track=["business_value_generated", "value_per_task", "roi_metrics"]
                ))
            
            # Sort recommendations by priority and expected improvement
            priority_order = {"high": 3, "medium": 2, "low": 1}
            recommendations.sort(
                key=lambda r: (priority_order.get(r.priority, 0), float(r.expected_improvement)),
                reverse=True
            )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization recommendations: {e}")
            return []


class AgentCapacityPlanner:
    """Provides resource allocation and scaling insights for agents."""
    
    def __init__(self):
        """Initialize capacity planner."""
        self.logger = logger
    
    async def analyze_capacity_requirements(
        self, 
        time_period_days: int = 30,
        forecast_days: int = 30
    ) -> Dict[str, Any]:
        """Analyze capacity requirements and provide scaling insights."""
        try:
            async with get_session() as session:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(days=time_period_days)
                
                # Get capacity metrics
                capacity_query = select(
                    AgentPerformanceMetric.agent_id,
                    func.avg(AgentPerformanceMetric.utilization_percentage).label("avg_utilization"),
                    func.avg(AgentPerformanceMetric.queue_depth).label("avg_queue_depth"),
                    func.avg(AgentPerformanceMetric.throughput_tasks_per_hour).label("avg_throughput"),
                    func.max(AgentPerformanceMetric.queue_depth).label("max_queue_depth"),
                    func.avg(AgentPerformanceMetric.cpu_usage_percent).label("avg_cpu"),
                    func.avg(AgentPerformanceMetric.memory_usage_mb).label("avg_memory")
                ).where(
                    and_(
                        AgentPerformanceMetric.timestamp >= start_time,
                        AgentPerformanceMetric.timestamp <= end_time
                    )
                ).group_by(AgentPerformanceMetric.agent_id)
                
                result = await session.execute(capacity_query)
                capacity_data = result.all()
                
                if not capacity_data:
                    return {"message": "No capacity data available"}
                
                # Analyze overall capacity
                total_agents = len(capacity_data)
                avg_utilization = mean([float(d.avg_utilization or 0) for d in capacity_data])
                avg_queue_depth = mean([float(d.avg_queue_depth or 0) for d in capacity_data])
                total_throughput = sum([float(d.avg_throughput or 0) for d in capacity_data])
                
                # Identify capacity bottlenecks
                bottlenecks = []
                high_utilization_agents = sum(1 for d in capacity_data if float(d.avg_utilization or 0) > 80)
                high_queue_agents = sum(1 for d in capacity_data if float(d.avg_queue_depth or 0) > 10)
                
                if high_utilization_agents > total_agents * 0.3:
                    bottlenecks.append("High agent utilization across 30%+ of fleet")
                if high_queue_agents > total_agents * 0.2:
                    bottlenecks.append("Queue depth issues affecting 20%+ of agents")
                if avg_utilization > 85:
                    bottlenecks.append("Overall system utilization exceeds recommended 85% threshold")
                
                # Generate scaling recommendations
                scaling_recommendations = []
                
                if avg_utilization > 80:
                    additional_agents_needed = max(1, int(total_agents * 0.2))  # 20% increase
                    scaling_recommendations.append(
                        f"Add {additional_agents_needed} agents to reduce utilization from {avg_utilization:.1f}% to target 70%"
                    )
                elif avg_utilization < 50:
                    agents_to_optimize = max(1, int(total_agents * 0.1))  # 10% reduction
                    scaling_recommendations.append(
                        f"Consider optimizing or consolidating {agents_to_optimize} agents due to low utilization ({avg_utilization:.1f}%)"
                    )
                
                if avg_queue_depth > 5:
                    scaling_recommendations.append("Implement load balancing to distribute queue load more evenly")
                
                # Resource allocation suggestions
                resource_suggestions = {
                    "cpu_optimization": [],
                    "memory_optimization": [],
                    "load_balancing": []
                }
                
                high_cpu_agents = [d for d in capacity_data if float(d.avg_cpu or 0) > 80]
                high_memory_agents = [d for d in capacity_data if float(d.avg_memory or 0) > 2000]  # 2GB
                
                if high_cpu_agents:
                    resource_suggestions["cpu_optimization"].append(
                        f"{len(high_cpu_agents)} agents require CPU optimization (>80% usage)"
                    )
                
                if high_memory_agents:
                    resource_suggestions["memory_optimization"].append(
                        f"{len(high_memory_agents)} agents require memory optimization (>2GB usage)"
                    )
                
                # Load balancing opportunities
                utilization_variance = stdev([float(d.avg_utilization or 0) for d in capacity_data]) if len(capacity_data) > 1 else 0
                load_balancing_opportunities = []
                
                if utilization_variance > 20:
                    load_balancing_opportunities.append("High utilization variance indicates load balancing opportunities")
                
                uneven_queues = max([float(d.avg_queue_depth or 0) for d in capacity_data]) - min([float(d.avg_queue_depth or 0) for d in capacity_data])
                if uneven_queues > 10:
                    load_balancing_opportunities.append("Uneven queue distribution suggests need for better task routing")
                
                return {
                    "analysis_period_days": time_period_days,
                    "forecast_horizon_days": forecast_days,
                    "current_capacity": {
                        "total_agents": total_agents,
                        "average_utilization": round(avg_utilization, 2),
                        "average_queue_depth": round(avg_queue_depth, 2),
                        "total_throughput_per_hour": round(total_throughput, 2),
                        "capacity_headroom": round(100 - avg_utilization, 2)
                    },
                    "bottlenecks": bottlenecks,
                    "scaling_recommendations": scaling_recommendations,
                    "resource_allocation": resource_suggestions,
                    "load_balancing_opportunities": load_balancing_opportunities,
                    "capacity_forecast": {
                        "projected_utilization_increase": round(avg_utilization * 0.1, 2),  # Simplified projection
                        "recommended_additional_agents": max(0, int((avg_utilization - 70) / 10)) if avg_utilization > 70 else 0,
                        "cost_optimization_potential": "Medium" if avg_utilization < 60 else "Low"
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Failed to analyze capacity requirements: {e}")
            return {"error": str(e)}


class AgentBenchmarkTracker:
    """Tracks comparative performance analysis across agents."""
    
    def __init__(self):
        """Initialize benchmark tracker."""
        self.logger = logger
    
    async def compare_agent_performance(
        self, 
        agent_id: str, 
        time_period_days: int = 30
    ) -> Optional[AgentBenchmarkMetrics]:
        """Compare agent performance against peer agents."""
        try:
            async with get_session() as session:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(days=time_period_days)
                
                # Get performance metrics for all agents
                all_agents_query = select(
                    AgentPerformanceMetric.agent_id,
                    func.avg(AgentPerformanceMetric.success_rate).label("avg_success_rate"),
                    func.avg(AgentPerformanceMetric.average_response_time_ms).label("avg_response_time"),
                    func.avg(AgentPerformanceMetric.cpu_usage_percent).label("avg_cpu"),
                    func.avg(AgentPerformanceMetric.uptime_percentage).label("avg_uptime"),
                    func.sum(AgentPerformanceMetric.tasks_completed).label("total_tasks"),
                    func.avg(AgentPerformanceMetric.business_value_generated).label("avg_business_value")
                ).where(
                    and_(
                        AgentPerformanceMetric.timestamp >= start_time,
                        AgentPerformanceMetric.timestamp <= end_time
                    )
                ).group_by(AgentPerformanceMetric.agent_id)
                
                result = await session.execute(all_agents_query)
                all_agent_stats = {row.agent_id: row for row in result}
                
                if agent_id not in all_agent_stats:
                    self.logger.warning(f"No performance data found for agent {agent_id}")
                    return None
                
                target_stats = all_agent_stats[agent_id]
                peer_stats = [stats for aid, stats in all_agent_stats.items() if aid != agent_id]
                
                if not peer_stats:
                    return None
                
                # Calculate benchmark metrics
                metrics = {}
                peer_metrics = {}
                
                # Success rate comparison
                target_success = float(target_stats.avg_success_rate or 0)
                peer_success_rates = [float(stats.avg_success_rate or 0) for stats in peer_stats]
                peer_avg_success = mean(peer_success_rates)
                metrics["success_rate"] = target_success
                peer_metrics["success_rate"] = peer_avg_success
                
                # Response time comparison
                target_response = int(target_stats.avg_response_time or 0)
                peer_response_times = [int(stats.avg_response_time or 0) for stats in peer_stats]
                peer_avg_response = mean(peer_response_times)
                metrics["response_time"] = target_response
                peer_metrics["response_time"] = peer_avg_response
                
                # Task completion comparison
                target_tasks = int(target_stats.total_tasks or 0)
                peer_task_counts = [int(stats.total_tasks or 0) for stats in peer_stats]
                peer_avg_tasks = mean(peer_task_counts)
                metrics["task_completion"] = target_tasks
                peer_metrics["task_completion"] = peer_avg_tasks
                
                # Business value comparison
                target_value = float(target_stats.avg_business_value or 0)
                peer_values = [float(stats.avg_business_value or 0) for stats in peer_stats if stats.avg_business_value]
                peer_avg_value = mean(peer_values) if peer_values else 0
                metrics["business_value"] = target_value
                peer_metrics["business_value"] = peer_avg_value
                
                # Calculate overall performance score and ranking
                def calculate_performance_score(stats):
                    success_score = float(stats.avg_success_rate or 0)
                    response_score = max(0, 100 - (int(stats.avg_response_time or 1000) - 100) / 49)
                    uptime_score = float(stats.avg_uptime or 95)
                    task_score = min(100, (int(stats.total_tasks or 0) / 100) * 100)  # Normalize to 0-100
                    return (success_score + response_score + uptime_score + task_score) / 4
                
                agent_score = calculate_performance_score(target_stats)
                peer_scores = [calculate_performance_score(stats) for stats in peer_stats]
                
                # Calculate rank and percentile
                better_peers = sum(1 for score in peer_scores if score < agent_score)
                total_agents = len(all_agent_stats)
                performance_rank = better_peers + 1
                percentile = Decimal(str((better_peers / (total_agents - 1)) * 100)).quantize(Decimal('0.01')) if total_agents > 1 else Decimal('50.00')
                
                # Identify above/below average metrics
                above_average = []
                below_average = []
                
                if target_success > peer_avg_success * 1.05:
                    above_average.append("success_rate")
                elif target_success < peer_avg_success * 0.95:
                    below_average.append("success_rate")
                
                if target_response < peer_avg_response * 0.95:  # Lower is better for response time
                    above_average.append("response_time")
                elif target_response > peer_avg_response * 1.05:
                    below_average.append("response_time")
                
                if target_tasks > peer_avg_tasks * 1.1:
                    above_average.append("task_completion")
                elif target_tasks < peer_avg_tasks * 0.9:
                    below_average.append("task_completion")
                
                if target_value > peer_avg_value * 1.1:
                    above_average.append("business_value")
                elif target_value < peer_avg_value * 0.9:
                    below_average.append("business_value")
                
                return AgentBenchmarkMetrics(
                    performance_rank=performance_rank,
                    total_agents=total_agents,
                    percentile=percentile,
                    above_average_metrics=above_average,
                    below_average_metrics=below_average,
                    peer_comparison={
                        "agent_metrics": metrics,
                        "peer_averages": peer_metrics,
                        "relative_performance": {
                            "success_rate_ratio": round(target_success / peer_avg_success, 2) if peer_avg_success > 0 else 0,
                            "response_time_ratio": round(peer_avg_response / target_response, 2) if target_response > 0 else 0,
                            "task_completion_ratio": round(target_tasks / peer_avg_tasks, 2) if peer_avg_tasks > 0 else 0,
                            "business_value_ratio": round(target_value / peer_avg_value, 2) if peer_avg_value > 0 else 0
                        }
                    }
                )
                
        except Exception as e:
            self.logger.error(f"Failed to compare agent performance: {e}")
            return None
    
    async def get_performance_leaderboard(
        self, 
        time_period_days: int = 30, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top performing agents leaderboard."""
        try:
            async with get_session() as session:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(days=time_period_days)
                
                # Get agent performance metrics with rankings
                leaderboard_query = select(
                    AgentPerformanceMetric.agent_id,
                    func.avg(AgentPerformanceMetric.success_rate).label("avg_success_rate"),
                    func.avg(AgentPerformanceMetric.average_response_time_ms).label("avg_response_time"),
                    func.avg(AgentPerformanceMetric.uptime_percentage).label("avg_uptime"),
                    func.sum(AgentPerformanceMetric.tasks_completed).label("total_tasks_completed"),
                    func.sum(AgentPerformanceMetric.tasks_failed).label("total_tasks_failed"),
                    func.avg(AgentPerformanceMetric.business_value_generated).label("avg_business_value")
                ).where(
                    and_(
                        AgentPerformanceMetric.timestamp >= start_time,
                        AgentPerformanceMetric.timestamp <= end_time
                    )
                ).group_by(AgentPerformanceMetric.agent_id)
                
                result = await session.execute(leaderboard_query)
                agent_stats = result.all()
                
                # Calculate performance scores and rank
                leaderboard_data = []
                for stats in agent_stats:
                    success_score = float(stats.avg_success_rate or 0)
                    response_score = max(0, 100 - (int(stats.avg_response_time or 1000) - 100) / 49)
                    uptime_score = float(stats.avg_uptime or 95)
                    
                    # Calculate completion rate
                    total_tasks = (stats.total_tasks_completed or 0) + (stats.total_tasks_failed or 0)
                    completion_rate = (stats.total_tasks_completed or 0) / max(total_tasks, 1) * 100
                    
                    # Overall performance score
                    performance_score = (success_score + response_score + uptime_score + completion_rate) / 4
                    
                    leaderboard_data.append({
                        "agent_id": str(stats.agent_id),
                        "performance_score": round(performance_score, 2),
                        "success_rate": round(success_score, 2),
                        "average_response_time_ms": int(stats.avg_response_time or 0),
                        "uptime_percentage": round(float(stats.avg_uptime or 0), 2),
                        "tasks_completed": int(stats.total_tasks_completed or 0),
                        "tasks_failed": int(stats.total_tasks_failed or 0),
                        "completion_rate": round(completion_rate, 2),
                        "business_value_generated": round(float(stats.avg_business_value or 0), 2)
                    })
                
                # Sort by performance score
                leaderboard_data.sort(key=lambda x: x["performance_score"], reverse=True)
                
                # Add rankings
                for i, agent_data in enumerate(leaderboard_data[:limit]):
                    agent_data["rank"] = i + 1
                
                return leaderboard_data[:limit]
                
        except Exception as e:
            self.logger.error(f"Failed to get performance leaderboard: {e}")
            return []


# Global instances
_agent_performance_analyzer: Optional[AgentPerformanceAnalyzer] = None
_agent_optimization_engine: Optional[AgentOptimizationEngine] = None
_agent_capacity_planner: Optional[AgentCapacityPlanner] = None
_agent_benchmark_tracker: Optional[AgentBenchmarkTracker] = None

async def get_agent_performance_analyzer() -> AgentPerformanceAnalyzer:
    """Get or create agent performance analyzer instance."""
    global _agent_performance_analyzer
    if _agent_performance_analyzer is None:
        _agent_performance_analyzer = AgentPerformanceAnalyzer()
    return _agent_performance_analyzer

async def get_agent_optimization_engine() -> AgentOptimizationEngine:
    """Get or create agent optimization engine instance."""
    global _agent_optimization_engine
    if _agent_optimization_engine is None:
        _agent_optimization_engine = AgentOptimizationEngine()
    return _agent_optimization_engine

async def get_agent_capacity_planner() -> AgentCapacityPlanner:
    """Get or create agent capacity planner instance."""
    global _agent_capacity_planner
    if _agent_capacity_planner is None:
        _agent_capacity_planner = AgentCapacityPlanner()
    return _agent_capacity_planner

async def get_agent_benchmark_tracker() -> AgentBenchmarkTracker:
    """Get or create agent benchmark tracker instance."""
    global _agent_benchmark_tracker
    if _agent_benchmark_tracker is None:
        _agent_benchmark_tracker = AgentBenchmarkTracker()
    return _agent_benchmark_tracker