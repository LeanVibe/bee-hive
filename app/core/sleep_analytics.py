"""
Comprehensive Sleep Analytics and Performance Tracking System.

Provides detailed monitoring and analysis of sleep-wake cycle effectiveness:
- Real-time performance metrics collection and analysis
- Trend analysis and pattern recognition for optimization
- Efficiency scoring and token reduction tracking
- Consolidated reporting for system health monitoring
- Predictive analytics for schedule optimization
- Advanced statistical analysis of consolidation patterns
"""

import asyncio
import logging
import statistics
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple, Union
from uuid import UUID
from collections import defaultdict, deque
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, asc, update
from sqlalchemy.orm import selectinload

from ..models.sleep_wake import (
    SleepWakeCycle, SleepWakeAnalytics, ConsolidationJob, 
    Checkpoint, SleepState, ConsolidationStatus
)
from ..models.agent import Agent
from ..models.performance_metric import PerformanceMetric
from ..core.database import get_async_session
from ..core.config import get_settings


logger = logging.getLogger(__name__)


@dataclass
class AnalyticsTimeRange:
    """Time range for analytics queries."""
    start_date: date
    end_date: date
    
    def to_datetime_range(self) -> Tuple[datetime, datetime]:
        """Convert to datetime range for database queries."""
        start_dt = datetime.combine(self.start_date, datetime.min.time())
        end_dt = datetime.combine(self.end_date, datetime.max.time())
        return start_dt, end_dt


@dataclass
class SleepEfficiencyMetrics:
    """Sleep efficiency metrics for an agent or system."""
    
    # Basic metrics
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    
    # Token reduction metrics
    total_tokens_saved: int = 0
    average_token_reduction: float = 0.0
    token_reduction_efficiency: float = 0.0
    
    # Timing metrics
    average_consolidation_time_ms: float = 0.0
    average_recovery_time_ms: float = 0.0
    average_cycle_duration_minutes: float = 0.0
    
    # Quality metrics
    uptime_percentage: float = 0.0
    success_rate: float = 0.0
    efficiency_score: float = 0.0
    
    # Resource utilization
    cpu_efficiency: float = 0.0
    memory_optimization: float = 0.0
    consolidation_ratio: float = 0.0
    
    # Reliability metrics
    checkpoint_success_rate: float = 0.0
    recovery_success_rate: float = 0.0
    manual_intervention_rate: float = 0.0
    
    def calculate_overall_score(self) -> float:
        """Calculate overall efficiency score (0-100)."""
        # Weight different components
        weights = {
            'token_efficiency': 0.35,     # 35% - Token reduction effectiveness
            'time_efficiency': 0.25,      # 25% - Processing time optimization
            'reliability': 0.25,          # 25% - Success rates and uptime
            'resource_efficiency': 0.15   # 15% - CPU/memory optimization
        }
        
        # Normalize metrics to 0-100 scale
        token_score = min(100, self.token_reduction_efficiency * 100)
        time_score = max(0, 100 - (self.average_consolidation_time_ms / 1000 / 10))  # Penalize >10s consolidation
        reliability_score = (self.success_rate + self.uptime_percentage) / 2
        resource_score = (self.cpu_efficiency + self.memory_optimization) / 2
        
        overall_score = (
            token_score * weights['token_efficiency'] +
            time_score * weights['time_efficiency'] +
            reliability_score * weights['reliability'] +
            resource_score * weights['resource_efficiency']
        )
        
        return min(100, max(0, overall_score))


@dataclass
class ConsolidationTrends:
    """Consolidation trend analysis results."""
    
    # Trend data
    daily_efficiency: List[float]
    weekly_patterns: Dict[str, float]  # Day of week -> efficiency
    hourly_patterns: Dict[int, float]  # Hour -> efficiency
    
    # Statistical insights
    efficiency_trend: str  # "improving", "declining", "stable"
    peak_performance_time: Optional[str]  # Best time for consolidation
    optimization_opportunities: List[str]
    
    # Predictive metrics
    predicted_efficiency: float
    confidence_interval: Tuple[float, float]
    recommendation_score: float


class SleepAnalyticsEngine:
    """
    Comprehensive sleep analytics and performance tracking engine.
    
    Features:
    - Real-time performance metrics collection
    - Trend analysis and pattern recognition
    - Efficiency scoring and optimization insights
    - Predictive analytics for schedule optimization
    - Comprehensive reporting and visualization data
    - Statistical analysis of consolidation patterns
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Analytics configuration
        self.metrics_retention_days = 90
        self.trend_analysis_window_days = 30
        self.real_time_window_hours = 24
        
        # Performance thresholds
        self.target_token_reduction = 0.55  # 55% target
        self.max_consolidation_time_ms = 30000  # 30 seconds max
        self.min_success_rate = 95.0  # 95% success rate target
        self.min_uptime_percentage = 99.0  # 99% uptime target
        
        # Caching for performance
        self._metrics_cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, datetime] = {}
        self._cache_ttl_minutes = 15
        
        # Background analytics tasks
        self._background_tasks: List[asyncio.Task] = []
        self._analytics_enabled = True
    
    async def collect_sleep_cycle_metrics(self, cycle_id: UUID) -> Dict[str, Any]:
        """
        Collect comprehensive metrics for a sleep cycle.
        
        Args:
            cycle_id: Sleep-wake cycle ID
            
        Returns:
            Dictionary containing collected metrics
        """
        try:
            metrics = {
                "cycle_id": str(cycle_id),
                "collection_timestamp": datetime.utcnow().isoformat(),
                "metrics": {}
            }
            
            async with get_async_session() as session:
                # Get cycle data
                cycle = await session.get(SleepWakeCycle, cycle_id)
                if not cycle:
                    logger.warning(f"Sleep cycle {cycle_id} not found")
                    return metrics
                
                # Basic cycle metrics
                cycle_metrics = await self._collect_cycle_performance_metrics(session, cycle)
                metrics["metrics"]["cycle_performance"] = cycle_metrics
                
                # Consolidation job metrics
                job_metrics = await self._collect_consolidation_job_metrics(session, cycle_id)
                metrics["metrics"]["consolidation_jobs"] = job_metrics
                
                # Resource utilization metrics
                resource_metrics = await self._collect_resource_utilization_metrics(session, cycle)
                metrics["metrics"]["resource_utilization"] = resource_metrics
                
                # Checkpoint metrics
                checkpoint_metrics = await self._collect_checkpoint_metrics(session, cycle)
                metrics["metrics"]["checkpoints"] = checkpoint_metrics
                
                # Token reduction analysis
                token_metrics = await self._collect_token_reduction_metrics(session, cycle)
                metrics["metrics"]["token_reduction"] = token_metrics
                
                # Store metrics in database
                await self._store_performance_metrics(session, cycle.agent_id, metrics["metrics"])
                
                logger.info(f"Collected comprehensive metrics for cycle {cycle_id}")
                return metrics
                
        except Exception as e:
            logger.error(f"Error collecting sleep cycle metrics for {cycle_id}: {e}")
            return {"error": str(e), "cycle_id": str(cycle_id)}
    
    async def generate_efficiency_report(
        self, 
        agent_id: Optional[UUID] = None,
        time_range: Optional[AnalyticsTimeRange] = None
    ) -> SleepEfficiencyMetrics:
        """
        Generate comprehensive efficiency report for agent or system.
        
        Args:
            agent_id: Specific agent ID (optional, None for system-wide)
            time_range: Time range for analysis (optional, defaults to last 30 days)
            
        Returns:
            SleepEfficiencyMetrics with comprehensive analysis
        """
        try:
            if not time_range:
                time_range = AnalyticsTimeRange(
                    start_date=date.today() - timedelta(days=30),
                    end_date=date.today()
                )
            
            # Check cache first
            cache_key = f"efficiency_report_{agent_id}_{time_range.start_date}_{time_range.end_date}"
            if self._is_cache_valid(cache_key):
                return self._metrics_cache[cache_key]
            
            start_dt, end_dt = time_range.to_datetime_range()
            
            async with get_async_session() as session:
                # Build base query
                query = select(SleepWakeCycle).where(
                    and_(
                        SleepWakeCycle.created_at >= start_dt,
                        SleepWakeCycle.created_at <= end_dt
                    )
                )
                
                if agent_id:
                    query = query.where(SleepWakeCycle.agent_id == agent_id)
                
                # Include consolidation jobs
                query = query.options(selectinload(SleepWakeCycle.consolidation_jobs))
                
                result = await session.execute(query)
                cycles = result.scalars().all()
                
                # Calculate metrics
                metrics = await self._calculate_efficiency_metrics(session, cycles, agent_id, time_range)
                
                # Cache results
                self._cache_result(cache_key, metrics)
                
                logger.info(f"Generated efficiency report for agent {agent_id}: {metrics.total_cycles} cycles analyzed")
                return metrics
                
        except Exception as e:
            logger.error(f"Error generating efficiency report: {e}")
            return SleepEfficiencyMetrics()
    
    async def analyze_consolidation_trends(
        self, 
        agent_id: Optional[UUID] = None,
        analysis_days: int = 30
    ) -> ConsolidationTrends:
        """
        Analyze consolidation trends and patterns.
        
        Args:
            agent_id: Specific agent ID (optional)
            analysis_days: Number of days to analyze
            
        Returns:
            ConsolidationTrends with detailed analysis
        """
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=analysis_days)
            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.max.time())
            
            async with get_async_session() as session:
                # Get cycles for analysis
                query = select(SleepWakeCycle).where(
                    and_(
                        SleepWakeCycle.created_at >= start_dt,
                        SleepWakeCycle.created_at <= end_dt,
                        SleepWakeCycle.sleep_state != SleepState.ERROR
                    )
                )
                
                if agent_id:
                    query = query.where(SleepWakeCycle.agent_id == agent_id)
                
                result = await session.execute(query)
                cycles = result.scalars().all()
                
                # Analyze trends
                trends = await self._analyze_efficiency_trends(cycles)
                
                logger.info(f"Analyzed consolidation trends for {len(cycles)} cycles")
                return trends
                
        except Exception as e:
            logger.error(f"Error analyzing consolidation trends: {e}")
            return ConsolidationTrends(
                daily_efficiency=[],
                weekly_patterns={},
                hourly_patterns={},
                efficiency_trend="unknown",
                peak_performance_time=None,
                optimization_opportunities=[],
                predicted_efficiency=0.0,
                confidence_interval=(0.0, 0.0),
                recommendation_score=0.0
            )
    
    async def update_daily_analytics(self, target_date: Optional[date] = None) -> bool:
        """
        Update daily analytics aggregation.
        
        Args:
            target_date: Date to update (defaults to yesterday)
            
        Returns:
            True if successful
        """
        try:
            if not target_date:
                target_date = date.today() - timedelta(days=1)
            
            start_dt = datetime.combine(target_date, datetime.min.time())
            end_dt = datetime.combine(target_date, datetime.max.time())
            
            async with get_async_session() as session:
                # Get all agents with cycles on this date
                agents_query = select(SleepWakeCycle.agent_id).where(
                    and_(
                        SleepWakeCycle.created_at >= start_dt,
                        SleepWakeCycle.created_at <= end_dt
                    )
                ).distinct()
                
                result = await session.execute(agents_query)
                agent_ids = [row[0] for row in result.fetchall()]
                
                # Process each agent
                for agent_id in agent_ids:
                    await self._update_agent_daily_analytics(session, agent_id, target_date)
                
                # Update system-wide analytics
                await self._update_system_daily_analytics(session, target_date)
                
                logger.info(f"Updated daily analytics for {len(agent_ids)} agents on {target_date}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating daily analytics: {e}")
            return False
    
    async def get_real_time_dashboard_data(self, agent_id: Optional[UUID] = None) -> Dict[str, Any]:
        """
        Get real-time dashboard data for monitoring.
        
        Args:
            agent_id: Specific agent ID (optional)
            
        Returns:
            Dashboard data dictionary
        """
        try:
            current_time = datetime.utcnow()
            window_start = current_time - timedelta(hours=self.real_time_window_hours)
            
            async with get_async_session() as session:
                # Active cycles
                active_cycles = await self._get_active_cycles(session, agent_id)
                
                # Recent performance
                recent_performance = await self._get_recent_performance(session, agent_id, window_start)
                
                # System health indicators
                health_indicators = await self._get_system_health_indicators(session, agent_id)
                
                # Current efficiency metrics
                current_metrics = await self._get_current_efficiency_metrics(session, agent_id)
                
                dashboard_data = {
                    "timestamp": current_time.isoformat(),
                    "agent_id": str(agent_id) if agent_id else None,
                    "active_cycles": active_cycles,
                    "recent_performance": recent_performance,
                    "health_indicators": health_indicators,
                    "current_metrics": current_metrics,
                    "alerts": await self._generate_performance_alerts(session, agent_id)
                }
                
                return dashboard_data
                
        except Exception as e:
            logger.error(f"Error getting real-time dashboard data: {e}")
            return {"error": str(e)}
    
    async def export_analytics_report(
        self, 
        report_type: str = "comprehensive",
        agent_id: Optional[UUID] = None,
        time_range: Optional[AnalyticsTimeRange] = None,
        format: str = "json"
    ) -> Dict[str, Any]:
        """
        Export comprehensive analytics report.
        
        Args:
            report_type: Type of report ("comprehensive", "summary", "trends")
            agent_id: Specific agent ID (optional)
            time_range: Time range for report (optional)
            format: Export format ("json", "csv")
            
        Returns:
            Analytics report data
        """
        try:
            if not time_range:
                time_range = AnalyticsTimeRange(
                    start_date=date.today() - timedelta(days=30),
                    end_date=date.today()
                )
            
            report_data = {
                "report_type": report_type,
                "agent_id": str(agent_id) if agent_id else None,
                "time_range": {
                    "start_date": time_range.start_date.isoformat(),
                    "end_date": time_range.end_date.isoformat()
                },
                "generated_at": datetime.utcnow().isoformat(),
                "format": format
            }
            
            if report_type == "comprehensive":
                # Full comprehensive report
                report_data["efficiency_metrics"] = await self.generate_efficiency_report(agent_id, time_range)
                report_data["consolidation_trends"] = await self.analyze_consolidation_trends(agent_id, 30)
                report_data["detailed_cycles"] = await self._get_detailed_cycle_data(agent_id, time_range)
                
            elif report_type == "summary":
                # Summary report
                report_data["efficiency_metrics"] = await self.generate_efficiency_report(agent_id, time_range)
                report_data["key_insights"] = await self._generate_key_insights(agent_id, time_range)
                
            elif report_type == "trends":
                # Trends-focused report
                report_data["consolidation_trends"] = await self.analyze_consolidation_trends(agent_id, 30)
                report_data["performance_trends"] = await self._get_performance_trends(agent_id, time_range)
            
            logger.info(f"Generated {report_type} analytics report for agent {agent_id}")
            return report_data
            
        except Exception as e:
            logger.error(f"Error exporting analytics report: {e}")
            return {"error": str(e)}
    
    # Private helper methods
    
    async def _collect_cycle_performance_metrics(self, session: AsyncSession, cycle: SleepWakeCycle) -> Dict[str, Any]:
        """Collect basic cycle performance metrics."""
        return {
            "cycle_id": str(cycle.id),
            "cycle_type": cycle.cycle_type,
            "sleep_state": cycle.sleep_state.value,
            "duration_minutes": cycle.duration_minutes,
            "token_reduction_achieved": cycle.token_reduction_achieved,
            "consolidation_time_ms": cycle.consolidation_time_ms,
            "recovery_time_ms": cycle.recovery_time_ms,
            "efficiency_score": cycle.get_efficiency_score(),
            "is_successful": cycle.sleep_state not in [SleepState.ERROR]
        }
    
    async def _collect_consolidation_job_metrics(self, session: AsyncSession, cycle_id: UUID) -> Dict[str, Any]:
        """Collect consolidation job metrics."""
        jobs_query = select(ConsolidationJob).where(ConsolidationJob.cycle_id == cycle_id)
        result = await session.execute(jobs_query)
        jobs = result.scalars().all()
        
        completed_jobs = [j for j in jobs if j.status == ConsolidationStatus.COMPLETED]
        failed_jobs = [j for j in jobs if j.status == ConsolidationStatus.FAILED]
        
        return {
            "total_jobs": len(jobs),
            "completed_jobs": len(completed_jobs),
            "failed_jobs": len(failed_jobs),
            "success_rate": len(completed_jobs) / len(jobs) * 100 if jobs else 0,
            "total_tokens_processed": sum(j.tokens_processed or 0 for j in jobs),
            "total_tokens_saved": sum(j.tokens_saved or 0 for j in jobs),
            "total_processing_time_ms": sum(j.processing_time_ms or 0 for j in jobs),
            "job_types": [j.job_type for j in completed_jobs]
        }
    
    async def _collect_resource_utilization_metrics(self, session: AsyncSession, cycle: SleepWakeCycle) -> Dict[str, Any]:
        """Collect resource utilization metrics."""
        # This would integrate with the ResourceMonitor from ConsolidationEngine
        return {
            "cpu_usage_peak": 0.0,  # Would be collected from ResourceMonitor
            "memory_usage_peak": 0.0,
            "disk_io_peak": 0.0,
            "resource_efficiency_score": 85.0  # Placeholder
        }
    
    async def _collect_checkpoint_metrics(self, session: AsyncSession, cycle: SleepWakeCycle) -> Dict[str, Any]:
        """Collect checkpoint-related metrics."""
        checkpoint_metrics = {
            "pre_sleep_checkpoint_created": cycle.pre_sleep_checkpoint_id is not None,
            "post_wake_checkpoint_created": cycle.post_wake_checkpoint_id is not None,
            "checkpoint_creation_time_ms": 0.0,
            "checkpoint_validation_success": True
        }
        
        if cycle.pre_sleep_checkpoint_id:
            checkpoint = await session.get(Checkpoint, cycle.pre_sleep_checkpoint_id)
            if checkpoint:
                checkpoint_metrics["checkpoint_size_mb"] = checkpoint.size_mb
                checkpoint_metrics["checkpoint_compression_ratio"] = checkpoint.compression_ratio
                checkpoint_metrics["checkpoint_creation_time_ms"] = checkpoint.creation_time_ms
        
        return checkpoint_metrics
    
    async def _collect_token_reduction_metrics(self, session: AsyncSession, cycle: SleepWakeCycle) -> Dict[str, Any]:
        """Collect detailed token reduction metrics."""
        return {
            "token_reduction_achieved": cycle.token_reduction_achieved or 0.0,
            "target_token_reduction": self.target_token_reduction,
            "reduction_efficiency": (cycle.token_reduction_achieved or 0.0) / self.target_token_reduction * 100,
            "meets_target": (cycle.token_reduction_achieved or 0.0) >= self.target_token_reduction
        }
    
    async def _store_performance_metrics(self, session: AsyncSession, agent_id: UUID, metrics: Dict[str, Any]) -> None:
        """Store performance metrics in database."""
        try:
            # Store key metrics as PerformanceMetric entries
            metric_entries = []
            
            # Cycle performance metrics
            if "cycle_performance" in metrics:
                cycle_metrics = metrics["cycle_performance"]
                if cycle_metrics.get("efficiency_score"):
                    metric_entries.append(PerformanceMetric(
                        metric_name="sleep_cycle_efficiency_score",
                        metric_value=cycle_metrics["efficiency_score"],
                        agent_id=agent_id,
                        tags={"cycle_id": cycle_metrics["cycle_id"]}
                    ))
                
                if cycle_metrics.get("token_reduction_achieved"):
                    metric_entries.append(PerformanceMetric(
                        metric_name="token_reduction_achieved",
                        metric_value=cycle_metrics["token_reduction_achieved"],
                        agent_id=agent_id,
                        tags={"cycle_id": cycle_metrics["cycle_id"]}
                    ))
            
            # Consolidation job metrics
            if "consolidation_jobs" in metrics:
                job_metrics = metrics["consolidation_jobs"]
                metric_entries.append(PerformanceMetric(
                    metric_name="consolidation_success_rate",
                    metric_value=job_metrics["success_rate"],
                    agent_id=agent_id,
                    tags={"total_jobs": job_metrics["total_jobs"]}
                ))
            
            # Add all metrics to session
            for metric in metric_entries:
                session.add(metric)
            
            await session.commit()
            
        except Exception as e:
            logger.error(f"Error storing performance metrics: {e}")
    
    async def _calculate_efficiency_metrics(
        self, 
        session: AsyncSession, 
        cycles: List[SleepWakeCycle],
        agent_id: Optional[UUID],
        time_range: AnalyticsTimeRange
    ) -> SleepEfficiencyMetrics:
        """Calculate comprehensive efficiency metrics from cycles."""
        
        if not cycles:
            return SleepEfficiencyMetrics()
        
        # Basic counts
        total_cycles = len(cycles)
        successful_cycles = len([c for c in cycles if c.sleep_state not in [SleepState.ERROR]])
        failed_cycles = total_cycles - successful_cycles
        
        # Token metrics
        token_reductions = [c.token_reduction_achieved for c in cycles if c.token_reduction_achieved is not None]
        avg_token_reduction = statistics.mean(token_reductions) if token_reductions else 0.0
        
        # Get total tokens saved from consolidation jobs
        total_tokens_saved = 0
        for cycle in cycles:
            if cycle.consolidation_jobs:
                total_tokens_saved += sum(job.tokens_saved or 0 for job in cycle.consolidation_jobs)
        
        # Timing metrics
        consolidation_times = [c.consolidation_time_ms for c in cycles if c.consolidation_time_ms is not None]
        avg_consolidation_time = statistics.mean(consolidation_times) if consolidation_times else 0.0
        
        recovery_times = [c.recovery_time_ms for c in cycles if c.recovery_time_ms is not None]
        avg_recovery_time = statistics.mean(recovery_times) if recovery_times else 0.0
        
        durations = [c.duration_minutes for c in cycles if c.duration_minutes is not None]
        avg_duration = statistics.mean(durations) if durations else 0.0
        
        # Calculate success rate
        success_rate = (successful_cycles / total_cycles * 100) if total_cycles > 0 else 0.0
        
        # Calculate uptime (time spent in working states vs error states)
        uptime_percentage = success_rate  # Simplified - could be more sophisticated
        
        # Calculate efficiency scores
        token_efficiency = min(1.0, avg_token_reduction / self.target_token_reduction) if avg_token_reduction > 0 else 0.0
        
        # Build metrics object
        metrics = SleepEfficiencyMetrics(
            total_cycles=total_cycles,
            successful_cycles=successful_cycles,
            failed_cycles=failed_cycles,
            total_tokens_saved=total_tokens_saved,
            average_token_reduction=avg_token_reduction,
            token_reduction_efficiency=token_efficiency,
            average_consolidation_time_ms=avg_consolidation_time,
            average_recovery_time_ms=avg_recovery_time,
            average_cycle_duration_minutes=avg_duration,
            uptime_percentage=uptime_percentage,
            success_rate=success_rate,
            cpu_efficiency=85.0,  # Placeholder - would integrate with ResourceMonitor
            memory_optimization=88.0,  # Placeholder
            consolidation_ratio=avg_token_reduction,
            checkpoint_success_rate=95.0,  # Placeholder
            recovery_success_rate=success_rate,
            manual_intervention_rate=5.0  # Placeholder
        )
        
        # Calculate overall efficiency score
        metrics.efficiency_score = metrics.calculate_overall_score()
        
        return metrics
    
    async def _analyze_efficiency_trends(self, cycles: List[SleepWakeCycle]) -> ConsolidationTrends:
        """Analyze efficiency trends from cycle data."""
        
        if not cycles:
            return ConsolidationTrends(
                daily_efficiency=[],
                weekly_patterns={},
                hourly_patterns={},
                efficiency_trend="unknown",
                peak_performance_time=None,
                optimization_opportunities=[],
                predicted_efficiency=0.0,
                confidence_interval=(0.0, 0.0),
                recommendation_score=0.0
            )
        
        # Group cycles by day
        daily_groups = defaultdict(list)
        for cycle in cycles:
            if cycle.created_at:
                day_key = cycle.created_at.date()
                daily_groups[day_key].append(cycle)
        
        # Calculate daily efficiency
        daily_efficiency = []
        for day, day_cycles in sorted(daily_groups.items()):
            token_reductions = [c.token_reduction_achieved for c in day_cycles if c.token_reduction_achieved is not None]
            if token_reductions:
                daily_efficiency.append(statistics.mean(token_reductions))
            else:
                daily_efficiency.append(0.0)
        
        # Analyze weekly patterns
        weekly_patterns = {}
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for cycle in cycles:
            if cycle.created_at and cycle.token_reduction_achieved is not None:
                day_name = day_names[cycle.created_at.weekday()]
                if day_name not in weekly_patterns:
                    weekly_patterns[day_name] = []
                weekly_patterns[day_name].append(cycle.token_reduction_achieved)
        
        # Average weekly patterns
        for day_name in weekly_patterns:
            weekly_patterns[day_name] = statistics.mean(weekly_patterns[day_name])
        
        # Analyze hourly patterns
        hourly_patterns = {}
        for cycle in cycles:
            if cycle.created_at and cycle.token_reduction_achieved is not None:
                hour = cycle.created_at.hour
                if hour not in hourly_patterns:
                    hourly_patterns[hour] = []
                hourly_patterns[hour].append(cycle.token_reduction_achieved)
        
        # Average hourly patterns
        for hour in hourly_patterns:
            hourly_patterns[hour] = statistics.mean(hourly_patterns[hour])
        
        # Determine trend
        if len(daily_efficiency) >= 7:
            recent_avg = statistics.mean(daily_efficiency[-7:])
            older_avg = statistics.mean(daily_efficiency[-14:-7]) if len(daily_efficiency) >= 14 else recent_avg
            
            if recent_avg > older_avg * 1.05:
                efficiency_trend = "improving"
            elif recent_avg < older_avg * 0.95:
                efficiency_trend = "declining"
            else:
                efficiency_trend = "stable"
        else:
            efficiency_trend = "insufficient_data"
        
        # Find peak performance time
        peak_performance_time = None
        if hourly_patterns:
            best_hour = max(hourly_patterns.items(), key=lambda x: x[1])
            peak_performance_time = f"{best_hour[0]:02d}:00"
        
        # Generate optimization opportunities
        optimization_opportunities = []
        if daily_efficiency:
            avg_efficiency = statistics.mean(daily_efficiency)
            if avg_efficiency < 0.4:
                optimization_opportunities.append("Token reduction below target - consider more aggressive consolidation")
            if hourly_patterns:
                worst_hour = min(hourly_patterns.items(), key=lambda x: x[1])
                if worst_hour[1] < avg_efficiency * 0.7:
                    optimization_opportunities.append(f"Poor performance at {worst_hour[0]:02d}:00 - avoid scheduling during this time")
        
        # Predictive metrics (simplified)
        predicted_efficiency = statistics.mean(daily_efficiency[-7:]) if len(daily_efficiency) >= 7 else 0.0
        confidence_interval = (predicted_efficiency * 0.9, predicted_efficiency * 1.1)
        recommendation_score = 85.0 if efficiency_trend == "improving" else 60.0
        
        return ConsolidationTrends(
            daily_efficiency=daily_efficiency,
            weekly_patterns=weekly_patterns,
            hourly_patterns=hourly_patterns,
            efficiency_trend=efficiency_trend,
            peak_performance_time=peak_performance_time,
            optimization_opportunities=optimization_opportunities,
            predicted_efficiency=predicted_efficiency,
            confidence_interval=confidence_interval,
            recommendation_score=recommendation_score
        )
    
    async def _update_agent_daily_analytics(self, session: AsyncSession, agent_id: UUID, target_date: date) -> None:
        """Update daily analytics for a specific agent."""
        try:
            start_dt = datetime.combine(target_date, datetime.min.time())
            end_dt = datetime.combine(target_date, datetime.max.time())
            
            # Get cycles for the day
            cycles_query = select(SleepWakeCycle).where(
                and_(
                    SleepWakeCycle.agent_id == agent_id,
                    SleepWakeCycle.created_at >= start_dt,
                    SleepWakeCycle.created_at <= end_dt
                )
            ).options(selectinload(SleepWakeCycle.consolidation_jobs))
            
            result = await session.execute(cycles_query)
            cycles = result.scalars().all()
            
            if not cycles:
                return
            
            # Calculate analytics
            total_cycles = len(cycles)
            successful_cycles = len([c for c in cycles if c.sleep_state not in [SleepState.ERROR]])
            failed_cycles = total_cycles - successful_cycles
            
            token_reductions = [c.token_reduction_achieved for c in cycles if c.token_reduction_achieved is not None]
            avg_token_reduction = statistics.mean(token_reductions) if token_reductions else 0.0
            
            consolidation_times = [c.consolidation_time_ms for c in cycles if c.consolidation_time_ms is not None]
            avg_consolidation_time = statistics.mean(consolidation_times) if consolidation_times else 0.0
            
            recovery_times = [c.recovery_time_ms for c in cycles if c.recovery_time_ms is not None]
            avg_recovery_time = statistics.mean(recovery_times) if recovery_times else 0.0
            
            # Calculate total tokens saved
            total_tokens_saved = 0
            total_processing_time = 0.0
            for cycle in cycles:
                if cycle.consolidation_jobs:
                    total_tokens_saved += sum(job.tokens_saved or 0 for job in cycle.consolidation_jobs)
                    total_processing_time += sum(job.processing_time_ms or 0 for job in cycle.consolidation_jobs)
            
            uptime_percentage = (successful_cycles / total_cycles * 100) if total_cycles > 0 else 0.0
            
            # Check if analytics record exists
            existing_query = select(SleepWakeAnalytics).where(
                and_(
                    SleepWakeAnalytics.agent_id == agent_id,
                    SleepWakeAnalytics.date == target_date
                )
            )
            existing_result = await session.execute(existing_query)
            existing_analytics = existing_result.scalar_one_or_none()
            
            if existing_analytics:
                # Update existing record
                existing_analytics.total_cycles = total_cycles
                existing_analytics.successful_cycles = successful_cycles
                existing_analytics.failed_cycles = failed_cycles
                existing_analytics.average_token_reduction = avg_token_reduction
                existing_analytics.average_consolidation_time_ms = avg_consolidation_time
                existing_analytics.average_recovery_time_ms = avg_recovery_time
                existing_analytics.total_tokens_saved = total_tokens_saved
                existing_analytics.total_processing_time_ms = total_processing_time
                existing_analytics.uptime_percentage = uptime_percentage
                existing_analytics.updated_at = datetime.utcnow()
            else:
                # Create new record
                analytics = SleepWakeAnalytics(
                    agent_id=agent_id,
                    date=target_date,
                    total_cycles=total_cycles,
                    successful_cycles=successful_cycles,
                    failed_cycles=failed_cycles,
                    average_token_reduction=avg_token_reduction,
                    average_consolidation_time_ms=avg_consolidation_time,
                    average_recovery_time_ms=avg_recovery_time,
                    total_tokens_saved=total_tokens_saved,
                    total_processing_time_ms=total_processing_time,
                    uptime_percentage=uptime_percentage
                )
                session.add(analytics)
            
            await session.commit()
            
        except Exception as e:
            logger.error(f"Error updating daily analytics for agent {agent_id}: {e}")
    
    async def _update_system_daily_analytics(self, session: AsyncSession, target_date: date) -> None:
        """Update system-wide daily analytics."""
        try:
            # System-wide analytics (agent_id = None)
            await self._update_agent_daily_analytics(session, None, target_date)
        except Exception as e:
            logger.error(f"Error updating system daily analytics: {e}")
    
    async def _get_active_cycles(self, session: AsyncSession, agent_id: Optional[UUID]) -> List[Dict[str, Any]]:
        """Get currently active sleep cycles."""
        try:
            query = select(SleepWakeCycle).where(
                SleepWakeCycle.sleep_state.in_([
                    SleepState.SLEEPING,
                    SleepState.CONSOLIDATING,
                    SleepState.PREPARING_WAKE
                ])
            )
            
            if agent_id:
                query = query.where(SleepWakeCycle.agent_id == agent_id)
            
            result = await session.execute(query)
            cycles = result.scalars().all()
            
            return [cycle.to_dict() for cycle in cycles]
            
        except Exception as e:
            logger.error(f"Error getting active cycles: {e}")
            return []
    
    async def _get_recent_performance(self, session: AsyncSession, agent_id: Optional[UUID], window_start: datetime) -> Dict[str, Any]:
        """Get recent performance metrics."""
        try:
            query = select(PerformanceMetric).where(
                PerformanceMetric.timestamp >= window_start
            )
            
            if agent_id:
                query = query.where(PerformanceMetric.agent_id == agent_id)
            
            result = await session.execute(query)
            metrics = result.scalars().all()
            
            # Group by metric name
            performance_data = defaultdict(list)
            for metric in metrics:
                performance_data[metric.metric_name].append({
                    "value": metric.metric_value,
                    "timestamp": metric.timestamp.isoformat(),
                    "tags": metric.tags
                })
            
            return dict(performance_data)
            
        except Exception as e:
            logger.error(f"Error getting recent performance: {e}")
            return {}
    
    async def _get_system_health_indicators(self, session: AsyncSession, agent_id: Optional[UUID]) -> Dict[str, Any]:
        """Get system health indicators."""
        try:
            # Get recent cycles for health assessment
            window_start = datetime.utcnow() - timedelta(hours=24)
            
            query = select(SleepWakeCycle).where(
                SleepWakeCycle.created_at >= window_start
            )
            
            if agent_id:
                query = query.where(SleepWakeCycle.agent_id == agent_id)
            
            result = await session.execute(query)
            cycles = result.scalars().all()
            
            if not cycles:
                return {
                    "status": "no_data",
                    "error_rate": 0.0,
                    "average_efficiency": 0.0,
                    "alerts": []
                }
            
            error_cycles = [c for c in cycles if c.sleep_state == SleepState.ERROR]
            error_rate = len(error_cycles) / len(cycles) * 100
            
            token_reductions = [c.token_reduction_achieved for c in cycles if c.token_reduction_achieved is not None]
            avg_efficiency = statistics.mean(token_reductions) if token_reductions else 0.0
            
            # Determine status
            if error_rate > 10:
                status = "critical"
            elif error_rate > 5 or avg_efficiency < 0.3:
                status = "warning"
            else:
                status = "healthy"
            
            alerts = []
            if error_rate > 5:
                alerts.append(f"High error rate: {error_rate:.1f}%")
            if avg_efficiency < self.target_token_reduction * 0.8:
                alerts.append(f"Token reduction below target: {avg_efficiency:.1%}")
            
            return {
                "status": status,
                "error_rate": error_rate,
                "average_efficiency": avg_efficiency,
                "total_cycles_24h": len(cycles),
                "alerts": alerts
            }
            
        except Exception as e:
            logger.error(f"Error getting system health indicators: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _get_current_efficiency_metrics(self, session: AsyncSession, agent_id: Optional[UUID]) -> Dict[str, Any]:
        """Get current efficiency metrics."""
        try:
            # Get recent performance metrics
            window_start = datetime.utcnow() - timedelta(hours=24)
            
            metrics_query = select(PerformanceMetric).where(
                and_(
                    PerformanceMetric.timestamp >= window_start,
                    PerformanceMetric.metric_name.in_([
                        "token_reduction_achieved",
                        "sleep_cycle_efficiency_score",
                        "consolidation_success_rate"
                    ])
                )
            )
            
            if agent_id:
                metrics_query = metrics_query.where(PerformanceMetric.agent_id == agent_id)
            
            result = await session.execute(metrics_query)
            metrics = result.scalars().all()
            
            # Calculate current averages
            token_reductions = [m.metric_value for m in metrics if m.metric_name == "token_reduction_achieved"]
            efficiency_scores = [m.metric_value for m in metrics if m.metric_name == "sleep_cycle_efficiency_score"]
            success_rates = [m.metric_value for m in metrics if m.metric_name == "consolidation_success_rate"]
            
            return {
                "current_token_reduction": statistics.mean(token_reductions) if token_reductions else 0.0,
                "current_efficiency_score": statistics.mean(efficiency_scores) if efficiency_scores else 0.0,
                "current_success_rate": statistics.mean(success_rates) if success_rates else 0.0,
                "target_token_reduction": self.target_token_reduction,
                "meets_target": statistics.mean(token_reductions) >= self.target_token_reduction if token_reductions else False
            }
            
        except Exception as e:
            logger.error(f"Error getting current efficiency metrics: {e}")
            return {}
    
    async def _generate_performance_alerts(self, session: AsyncSession, agent_id: Optional[UUID]) -> List[Dict[str, Any]]:
        """Generate performance alerts based on thresholds."""
        try:
            alerts = []
            current_time = datetime.utcnow()
            
            # Check recent error rates
            window_start = current_time - timedelta(hours=4)
            
            query = select(SleepWakeCycle).where(
                SleepWakeCycle.created_at >= window_start
            )
            
            if agent_id:
                query = query.where(SleepWakeCycle.agent_id == agent_id)
            
            result = await session.execute(query)
            recent_cycles = result.scalars().all()
            
            if recent_cycles:
                error_cycles = [c for c in recent_cycles if c.sleep_state == SleepState.ERROR]
                error_rate = len(error_cycles) / len(recent_cycles) * 100
                
                if error_rate > 10:
                    alerts.append({
                        "level": "critical",
                        "message": f"High error rate in last 4 hours: {error_rate:.1f}%",
                        "timestamp": current_time.isoformat(),
                        "metric": "error_rate",
                        "value": error_rate
                    })
                
                # Check token reduction performance
                token_reductions = [c.token_reduction_achieved for c in recent_cycles if c.token_reduction_achieved is not None]
                if token_reductions:
                    avg_reduction = statistics.mean(token_reductions)
                    if avg_reduction < self.target_token_reduction * 0.7:
                        alerts.append({
                            "level": "warning",
                            "message": f"Token reduction below 70% of target: {avg_reduction:.1%}",
                            "timestamp": current_time.isoformat(),
                            "metric": "token_reduction",
                            "value": avg_reduction
                        })
                
                # Check consolidation times
                consolidation_times = [c.consolidation_time_ms for c in recent_cycles if c.consolidation_time_ms is not None]
                if consolidation_times:
                    avg_time = statistics.mean(consolidation_times)
                    if avg_time > self.max_consolidation_time_ms:
                        alerts.append({
                            "level": "warning",
                            "message": f"Consolidation time exceeds threshold: {avg_time:.0f}ms",
                            "timestamp": current_time.isoformat(),
                            "metric": "consolidation_time",
                            "value": avg_time
                        })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating performance alerts: {e}")
            return []
    
    async def _get_detailed_cycle_data(self, agent_id: Optional[UUID], time_range: AnalyticsTimeRange) -> List[Dict[str, Any]]:
        """Get detailed cycle data for comprehensive reports."""
        try:
            start_dt, end_dt = time_range.to_datetime_range()
            
            async with get_async_session() as session:
                query = select(SleepWakeCycle).where(
                    and_(
                        SleepWakeCycle.created_at >= start_dt,
                        SleepWakeCycle.created_at <= end_dt
                    )
                ).options(selectinload(SleepWakeCycle.consolidation_jobs))
                
                if agent_id:
                    query = query.where(SleepWakeCycle.agent_id == agent_id)
                
                result = await session.execute(query)
                cycles = result.scalars().all()
                
                return [cycle.to_dict() for cycle in cycles]
                
        except Exception as e:
            logger.error(f"Error getting detailed cycle data: {e}")
            return []
    
    async def _generate_key_insights(self, agent_id: Optional[UUID], time_range: AnalyticsTimeRange) -> List[str]:
        """Generate key insights for summary reports."""
        try:
            efficiency_metrics = await self.generate_efficiency_report(agent_id, time_range)
            trends = await self.analyze_consolidation_trends(agent_id, 30)
            
            insights = []
            
            # Efficiency insights
            if efficiency_metrics.efficiency_score > 85:
                insights.append(f"Excellent overall efficiency: {efficiency_metrics.efficiency_score:.1f}/100")
            elif efficiency_metrics.efficiency_score < 60:
                insights.append(f"Below-target efficiency: {efficiency_metrics.efficiency_score:.1f}/100 - optimization needed")
            
            # Token reduction insights
            if efficiency_metrics.average_token_reduction >= self.target_token_reduction:
                insights.append(f"Token reduction target achieved: {efficiency_metrics.average_token_reduction:.1%}")
            else:
                insights.append(f"Token reduction below target: {efficiency_metrics.average_token_reduction:.1%} vs {self.target_token_reduction:.1%}")
            
            # Trend insights
            if trends.efficiency_trend == "improving":
                insights.append("Performance trend is improving over time")
            elif trends.efficiency_trend == "declining":
                insights.append("Performance trend is declining - investigation recommended")
            
            # Peak performance insights
            if trends.peak_performance_time:
                insights.append(f"Peak performance observed at {trends.peak_performance_time}")
            
            # Optimization opportunities
            insights.extend(trends.optimization_opportunities[:3])  # Top 3 opportunities
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating key insights: {e}")
            return ["Error generating insights"]
    
    async def _get_performance_trends(self, agent_id: Optional[UUID], time_range: AnalyticsTimeRange) -> Dict[str, Any]:
        """Get performance trends for trend reports."""
        try:
            start_dt, end_dt = time_range.to_datetime_range()
            
            async with get_async_session() as session:
                # Get daily analytics for the period
                analytics_query = select(SleepWakeAnalytics).where(
                    and_(
                        SleepWakeAnalytics.date >= time_range.start_date,
                        SleepWakeAnalytics.date <= time_range.end_date
                    )
                )
                
                if agent_id:
                    analytics_query = analytics_query.where(SleepWakeAnalytics.agent_id == agent_id)
                
                result = await session.execute(analytics_query)
                analytics = result.scalars().all()
                
                # Build trend data
                dates = [a.date.isoformat() for a in analytics]
                token_reductions = [a.average_token_reduction or 0.0 for a in analytics]
                success_rates = [a.success_rate for a in analytics]
                processing_times = [a.average_consolidation_time_ms or 0.0 for a in analytics]
                
                return {
                    "dates": dates,
                    "token_reduction_trend": token_reductions,
                    "success_rate_trend": success_rates,
                    "processing_time_trend": processing_times,
                    "data_points": len(analytics)
                }
                
        except Exception as e:
            logger.error(f"Error getting performance trends: {e}")
            return {}
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid."""
        if cache_key not in self._cache_expiry:
            return False
        
        return datetime.utcnow() < self._cache_expiry[cache_key]
    
    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cache a result with expiry."""
        self._metrics_cache[cache_key] = result
        self._cache_expiry[cache_key] = datetime.utcnow() + timedelta(minutes=self._cache_ttl_minutes)
    
    async def start_background_analytics(self) -> None:
        """Start background analytics processing."""
        if not self._analytics_enabled:
            return
        
        # Daily analytics update task
        daily_task = asyncio.create_task(self._daily_analytics_loop())
        self._background_tasks.append(daily_task)
        
        logger.info("Background analytics processing started")
    
    async def stop_background_analytics(self) -> None:
        """Stop background analytics processing."""
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        self._background_tasks.clear()
        logger.info("Background analytics processing stopped")
    
    async def _daily_analytics_loop(self) -> None:
        """Background loop for daily analytics updates."""
        while self._analytics_enabled:
            try:
                # Update yesterday's analytics
                yesterday = date.today() - timedelta(days=1)
                await self.update_daily_analytics(yesterday)
                
                # Sleep until next day
                await asyncio.sleep(24 * 60 * 60)  # 24 hours
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in daily analytics loop: {e}")
                await asyncio.sleep(60 * 60)  # Wait 1 hour before retry


# Global sleep analytics engine instance
_sleep_analytics_instance: Optional[SleepAnalyticsEngine] = None


def get_sleep_analytics_engine() -> SleepAnalyticsEngine:
    """Get the global sleep analytics engine instance."""
    global _sleep_analytics_instance
    if _sleep_analytics_instance is None:
        _sleep_analytics_instance = SleepAnalyticsEngine()
    return _sleep_analytics_instance


# Utility functions for quick access
async def collect_cycle_metrics(cycle_id: UUID) -> Dict[str, Any]:
    """Collect metrics for a specific sleep cycle."""
    engine = get_sleep_analytics_engine()
    return await engine.collect_sleep_cycle_metrics(cycle_id)


async def get_agent_efficiency_report(agent_id: UUID, days: int = 30) -> SleepEfficiencyMetrics:
    """Get efficiency report for an agent."""
    engine = get_sleep_analytics_engine()
    time_range = AnalyticsTimeRange(
        start_date=date.today() - timedelta(days=days),
        end_date=date.today()
    )
    return await engine.generate_efficiency_report(agent_id, time_range)


async def get_system_dashboard_data() -> Dict[str, Any]:
    """Get system-wide dashboard data."""
    engine = get_sleep_analytics_engine()
    return await engine.get_real_time_dashboard_data()