"""
Executive Dashboard Service

Real-time business KPIs and executive-level insights for data-driven decision making.
Provides comprehensive business metrics including revenue, growth, system performance,
and strategic indicators.

Epic 5: Business Intelligence & Analytics Engine
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal
from dataclasses import dataclass
import logging

from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ...models.business_intelligence import (
    BusinessMetric, MetricType, UserSession, AgentPerformanceMetric,
    BusinessAlert, AlertLevel
)
from ...models.agent import Agent, AgentStatus
from ...models.user import User
from ...models.task import Task, TaskStatus
from ...core.database import get_session
from ...core.logging_service import get_component_logger

logger = get_component_logger("executive_dashboard")


@dataclass
class BusinessMetrics:
    """Executive-level business metrics."""
    # Core business KPIs
    revenue_growth: Optional[Decimal] = None
    user_acquisition_rate: Optional[Decimal] = None
    system_uptime: Optional[Decimal] = None
    agent_utilization: Optional[Decimal] = None
    customer_satisfaction: Optional[Decimal] = None
    
    # Performance indicators
    total_active_users: int = 0
    total_agents: int = 0
    active_agents: int = 0
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    # Growth metrics
    daily_active_users: int = 0
    weekly_active_users: int = 0
    monthly_active_users: int = 0
    new_users_today: int = 0
    user_retention_rate: Optional[Decimal] = None
    
    # System performance
    average_response_time_ms: Optional[int] = None
    success_rate: Optional[Decimal] = None
    error_rate: Optional[Decimal] = None
    throughput: Optional[Decimal] = None
    
    # Business health
    conversion_rate: Optional[Decimal] = None
    engagement_score: Optional[Decimal] = None
    efficiency_score: Optional[Decimal] = None
    
    # Timestamp
    timestamp: datetime = datetime.utcnow()


class ExecutiveDashboard:
    """Executive Dashboard service for real-time business intelligence."""
    
    def __init__(self):
        """Initialize executive dashboard service."""
        self.logger = logger
        
    async def get_current_metrics(self) -> BusinessMetrics:
        """Get real-time executive business metrics."""
        try:
            async with get_session() as session:
                # Parallel data collection for performance
                metrics_data = await asyncio.gather(
                    self._get_user_metrics(session),
                    self._get_agent_metrics(session), 
                    self._get_task_metrics(session),
                    self._get_system_performance_metrics(session),
                    self._get_growth_metrics(session),
                    return_exceptions=True
                )
                
                # Unpack results
                user_metrics = metrics_data[0] if not isinstance(metrics_data[0], Exception) else {}
                agent_metrics = metrics_data[1] if not isinstance(metrics_data[1], Exception) else {}
                task_metrics = metrics_data[2] if not isinstance(metrics_data[2], Exception) else {}
                system_metrics = metrics_data[3] if not isinstance(metrics_data[3], Exception) else {}
                growth_metrics = metrics_data[4] if not isinstance(metrics_data[4], Exception) else {}
                
                # Combine into BusinessMetrics
                business_metrics = BusinessMetrics(
                    # Core KPIs
                    revenue_growth=await self._calculate_revenue_growth(session),
                    user_acquisition_rate=growth_metrics.get("acquisition_rate"),
                    system_uptime=system_metrics.get("uptime_percentage"),
                    agent_utilization=agent_metrics.get("utilization_rate"),
                    customer_satisfaction=await self._get_satisfaction_score(session),
                    
                    # User metrics
                    total_active_users=user_metrics.get("total_active", 0),
                    daily_active_users=user_metrics.get("daily_active", 0),
                    weekly_active_users=user_metrics.get("weekly_active", 0),
                    monthly_active_users=user_metrics.get("monthly_active", 0),
                    new_users_today=user_metrics.get("new_today", 0),
                    user_retention_rate=user_metrics.get("retention_rate"),
                    
                    # Agent metrics
                    total_agents=agent_metrics.get("total", 0),
                    active_agents=agent_metrics.get("active", 0),
                    
                    # Task metrics
                    total_tasks=task_metrics.get("total", 0),
                    completed_tasks=task_metrics.get("completed", 0),
                    failed_tasks=task_metrics.get("failed", 0),
                    
                    # System performance
                    average_response_time_ms=system_metrics.get("avg_response_time"),
                    success_rate=task_metrics.get("success_rate"),
                    error_rate=system_metrics.get("error_rate"),
                    throughput=system_metrics.get("throughput"),
                    
                    # Business health
                    conversion_rate=await self._calculate_conversion_rate(session),
                    engagement_score=user_metrics.get("engagement_score"),
                    efficiency_score=system_metrics.get("efficiency_score"),
                    
                    timestamp=datetime.utcnow()
                )
                
                # Store metrics for historical tracking
                await self._store_metrics_snapshot(session, business_metrics)
                
                return business_metrics
                
        except Exception as e:
            self.logger.error(f"Failed to get current metrics: {e}")
            raise
    
    async def _get_user_metrics(self, session: AsyncSession) -> Dict[str, Any]:
        """Get user-related metrics."""
        try:
            now = datetime.utcnow()
            today = now.date()
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)
            
            # User count queries
            total_users_query = select(func.count(User.id))
            daily_active_query = select(func.count(func.distinct(UserSession.user_id))).where(
                func.date(UserSession.session_start) == today
            )
            weekly_active_query = select(func.count(func.distinct(UserSession.user_id))).where(
                UserSession.session_start >= week_ago
            )
            monthly_active_query = select(func.count(func.distinct(UserSession.user_id))).where(
                UserSession.session_start >= month_ago
            )
            new_users_today_query = select(func.count(User.id)).where(
                func.date(User.created_at) == today
            )
            
            # Execute queries
            results = await asyncio.gather(
                session.execute(total_users_query),
                session.execute(daily_active_query),
                session.execute(weekly_active_query), 
                session.execute(monthly_active_query),
                session.execute(new_users_today_query)
            )
            
            total_active = results[0].scalar() or 0
            daily_active = results[1].scalar() or 0
            weekly_active = results[2].scalar() or 0
            monthly_active = results[3].scalar() or 0
            new_today = results[4].scalar() or 0
            
            # Calculate retention rate (simplified)
            retention_rate = None
            if total_active > 0 and weekly_active > 0:
                retention_rate = Decimal(str(weekly_active / total_active * 100)).quantize(Decimal('0.01'))
            
            # Calculate engagement score
            engagement_score = None
            if total_active > 0:
                avg_session_query = select(func.avg(UserSession.duration_seconds)).where(
                    UserSession.session_start >= week_ago
                )
                avg_duration_result = await session.execute(avg_session_query)
                avg_duration = avg_duration_result.scalar() or 0
                
                # Engagement score based on session duration (normalized to 0-100)
                if avg_duration > 0:
                    # Assume 1 hour (3600s) is perfect engagement (100 points)
                    engagement_score = Decimal(str(min(avg_duration / 3600 * 100, 100))).quantize(Decimal('0.01'))
            
            return {
                "total_active": total_active,
                "daily_active": daily_active, 
                "weekly_active": weekly_active,
                "monthly_active": monthly_active,
                "new_today": new_today,
                "retention_rate": retention_rate,
                "engagement_score": engagement_score
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get user metrics: {e}")
            return {}
    
    async def _get_agent_metrics(self, session: AsyncSession) -> Dict[str, Any]:
        """Get agent-related metrics."""
        try:
            # Agent count queries
            total_agents_query = select(func.count(Agent.id))
            active_agents_query = select(func.count(Agent.id)).where(
                Agent.status == AgentStatus.active
            )
            
            results = await asyncio.gather(
                session.execute(total_agents_query),
                session.execute(active_agents_query)
            )
            
            total_agents = results[0].scalar() or 0
            active_agents = results[1].scalar() or 0
            
            # Calculate utilization rate
            utilization_rate = None
            if total_agents > 0:
                utilization_rate = Decimal(str(active_agents / total_agents * 100)).quantize(Decimal('0.01'))
            
            return {
                "total": total_agents,
                "active": active_agents,
                "utilization_rate": utilization_rate
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get agent metrics: {e}")
            return {}
    
    async def _get_task_metrics(self, session: AsyncSession) -> Dict[str, Any]:
        """Get task-related metrics."""
        try:
            # Task count queries
            total_tasks_query = select(func.count(Task.id))
            completed_tasks_query = select(func.count(Task.id)).where(
                Task.status == TaskStatus.completed
            )
            failed_tasks_query = select(func.count(Task.id)).where(
                Task.status == TaskStatus.failed
            )
            
            results = await asyncio.gather(
                session.execute(total_tasks_query),
                session.execute(completed_tasks_query),
                session.execute(failed_tasks_query)
            )
            
            total_tasks = results[0].scalar() or 0
            completed_tasks = results[1].scalar() or 0
            failed_tasks = results[2].scalar() or 0
            
            # Calculate success rate
            success_rate = None
            if total_tasks > 0:
                success_rate = Decimal(str(completed_tasks / total_tasks * 100)).quantize(Decimal('0.01'))
            
            return {
                "total": total_tasks,
                "completed": completed_tasks,
                "failed": failed_tasks,
                "success_rate": success_rate
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get task metrics: {e}")
            return {}
    
    async def _get_system_performance_metrics(self, session: AsyncSession) -> Dict[str, Any]:
        """Get system performance metrics."""
        try:
            now = datetime.utcnow()
            hour_ago = now - timedelta(hours=1)
            
            # Get recent performance metrics
            perf_query = select(
                func.avg(AgentPerformanceMetric.average_response_time_ms).label("avg_response"),
                func.avg(AgentPerformanceMetric.cpu_usage_percent).label("avg_cpu"),
                func.avg(AgentPerformanceMetric.uptime_percentage).label("avg_uptime"),
                func.sum(AgentPerformanceMetric.error_count).label("total_errors"),
                func.count(AgentPerformanceMetric.id).label("measurement_count")
            ).where(
                AgentPerformanceMetric.timestamp >= hour_ago
            )
            
            result = await session.execute(perf_query)
            row = result.first()
            
            if row:
                avg_response_time = int(row.avg_response) if row.avg_response else None
                uptime_percentage = Decimal(str(row.avg_uptime)).quantize(Decimal('0.01')) if row.avg_uptime else None
                total_errors = row.total_errors or 0
                measurement_count = row.measurement_count or 0
                
                # Calculate error rate
                error_rate = None
                if measurement_count > 0:
                    error_rate = Decimal(str(total_errors / measurement_count * 100)).quantize(Decimal('0.01'))
                
                # Calculate throughput (requests per second)
                throughput = None
                if measurement_count > 0:
                    # Simplified throughput calculation
                    throughput = Decimal(str(measurement_count / 3600)).quantize(Decimal('0.01'))  # Per hour to per second
                
                # Calculate efficiency score
                efficiency_score = None
                if uptime_percentage and avg_response_time:
                    # Efficiency based on uptime and response time (normalized to 0-100)
                    response_score = max(0, 100 - (avg_response_time / 1000) * 10)  # Penalize slow responses
                    efficiency_score = Decimal(str((float(uptime_percentage) + response_score) / 2)).quantize(Decimal('0.01'))
                
                return {
                    "avg_response_time": avg_response_time,
                    "uptime_percentage": uptime_percentage,
                    "error_rate": error_rate,
                    "throughput": throughput,
                    "efficiency_score": efficiency_score
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Failed to get system performance metrics: {e}")
            return {}
    
    async def _get_growth_metrics(self, session: AsyncSession) -> Dict[str, Any]:
        """Get growth-related metrics."""
        try:
            now = datetime.utcnow()
            yesterday = now - timedelta(days=1)
            week_ago = now - timedelta(days=7)
            
            # User acquisition rate (new users today vs yesterday)
            today_users_query = select(func.count(User.id)).where(
                func.date(User.created_at) == now.date()
            )
            yesterday_users_query = select(func.count(User.id)).where(
                func.date(User.created_at) == yesterday.date()
            )
            
            results = await asyncio.gather(
                session.execute(today_users_query),
                session.execute(yesterday_users_query)
            )
            
            today_users = results[0].scalar() or 0
            yesterday_users = results[1].scalar() or 1  # Avoid division by zero
            
            # Calculate acquisition rate
            acquisition_rate = None
            if yesterday_users > 0:
                growth = ((today_users - yesterday_users) / yesterday_users) * 100
                acquisition_rate = Decimal(str(growth)).quantize(Decimal('0.01'))
            
            return {
                "acquisition_rate": acquisition_rate
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get growth metrics: {e}")
            return {}
    
    async def _calculate_revenue_growth(self, session: AsyncSession) -> Optional[Decimal]:
        """Calculate revenue growth (placeholder - would integrate with billing system)."""
        try:
            # This is a placeholder - in a real system, this would integrate with billing/payment systems
            # For now, we'll calculate based on user growth as a proxy
            
            now = datetime.utcnow()
            month_ago = now - timedelta(days=30)
            two_months_ago = now - timedelta(days=60)
            
            # Count active users this month vs last month as revenue proxy
            this_month_query = select(func.count(func.distinct(UserSession.user_id))).where(
                UserSession.session_start >= month_ago
            )
            last_month_query = select(func.count(func.distinct(UserSession.user_id))).where(
                and_(
                    UserSession.session_start >= two_months_ago,
                    UserSession.session_start < month_ago
                )
            )
            
            results = await asyncio.gather(
                session.execute(this_month_query),
                session.execute(last_month_query)
            )
            
            this_month_users = results[0].scalar() or 0
            last_month_users = results[1].scalar() or 1  # Avoid division by zero
            
            # Calculate growth percentage
            if last_month_users > 0:
                growth = ((this_month_users - last_month_users) / last_month_users) * 100
                return Decimal(str(growth)).quantize(Decimal('0.01'))
            
            return Decimal('0.00')
            
        except Exception as e:
            self.logger.error(f"Failed to calculate revenue growth: {e}")
            return None
    
    async def _get_satisfaction_score(self, session: AsyncSession) -> Optional[Decimal]:
        """Get customer satisfaction score from recent sessions."""
        try:
            week_ago = datetime.utcnow() - timedelta(days=7)
            
            satisfaction_query = select(func.avg(UserSession.satisfaction_score)).where(
                and_(
                    UserSession.satisfaction_score.isnot(None),
                    UserSession.session_start >= week_ago
                )
            )
            
            result = await session.execute(satisfaction_query)
            avg_satisfaction = result.scalar()
            
            if avg_satisfaction:
                return Decimal(str(avg_satisfaction)).quantize(Decimal('0.01'))
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get satisfaction score: {e}")
            return None
    
    async def _calculate_conversion_rate(self, session: AsyncSession) -> Optional[Decimal]:
        """Calculate conversion rate based on user journey events."""
        try:
            week_ago = datetime.utcnow() - timedelta(days=7)
            
            # Count total sessions and conversion sessions
            total_sessions_query = select(func.count(UserSession.id)).where(
                UserSession.session_start >= week_ago
            )
            conversion_sessions_query = select(func.count(UserSession.id)).where(
                and_(
                    UserSession.session_start >= week_ago,
                    UserSession.conversion_events.isnot(None)
                )
            )
            
            results = await asyncio.gather(
                session.execute(total_sessions_query),
                session.execute(conversion_sessions_query)
            )
            
            total_sessions = results[0].scalar() or 0
            conversion_sessions = results[1].scalar() or 0
            
            if total_sessions > 0:
                rate = (conversion_sessions / total_sessions) * 100
                return Decimal(str(rate)).quantize(Decimal('0.01'))
            
            return Decimal('0.00')
            
        except Exception as e:
            self.logger.error(f"Failed to calculate conversion rate: {e}")
            return None
    
    async def _store_metrics_snapshot(self, session: AsyncSession, metrics: BusinessMetrics) -> None:
        """Store business metrics snapshot for historical tracking."""
        try:
            # Store key metrics as individual records for time-series analysis
            metric_records = []
            
            # Revenue growth
            if metrics.revenue_growth is not None:
                metric_records.append(BusinessMetric(
                    metric_name="revenue_growth",
                    metric_type=MetricType.REVENUE,
                    metric_value=metrics.revenue_growth,
                    timestamp=metrics.timestamp,
                    source_system="executive_dashboard"
                ))
            
            # User metrics
            if metrics.user_acquisition_rate is not None:
                metric_records.append(BusinessMetric(
                    metric_name="user_acquisition_rate",
                    metric_type=MetricType.USER_ACQUISITION,
                    metric_value=metrics.user_acquisition_rate,
                    timestamp=metrics.timestamp,
                    source_system="executive_dashboard"
                ))
            
            # System metrics
            if metrics.system_uptime is not None:
                metric_records.append(BusinessMetric(
                    metric_name="system_uptime",
                    metric_type=MetricType.SYSTEM_PERFORMANCE,
                    metric_value=metrics.system_uptime,
                    timestamp=metrics.timestamp,
                    source_system="executive_dashboard"
                ))
            
            # Agent utilization
            if metrics.agent_utilization is not None:
                metric_records.append(BusinessMetric(
                    metric_name="agent_utilization",
                    metric_type=MetricType.AGENT_UTILIZATION,
                    metric_value=metrics.agent_utilization,
                    timestamp=metrics.timestamp,
                    source_system="executive_dashboard"
                ))
            
            # Customer satisfaction
            if metrics.customer_satisfaction is not None:
                metric_records.append(BusinessMetric(
                    metric_name="customer_satisfaction",
                    metric_type=MetricType.CUSTOMER_SATISFACTION,
                    metric_value=metrics.customer_satisfaction,
                    timestamp=metrics.timestamp,
                    source_system="executive_dashboard"
                ))
            
            # Add records to session
            for record in metric_records:
                session.add(record)
            
            await session.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to store metrics snapshot: {e}")
            await session.rollback()
    
    async def get_alerts(self, level: Optional[AlertLevel] = None) -> List[Dict[str, Any]]:
        """Get business alerts with optional filtering by level."""
        try:
            async with get_session() as session:
                query = select(BusinessAlert).where(BusinessAlert.is_active == True)
                
                if level:
                    query = query.where(BusinessAlert.alert_level == level)
                
                query = query.order_by(desc(BusinessAlert.triggered_at)).limit(50)
                
                result = await session.execute(query)
                alerts = result.scalars().all()
                
                return [
                    {
                        "id": str(alert.id),
                        "type": alert.alert_type,
                        "name": alert.alert_name,
                        "level": alert.alert_level.value,
                        "message": alert.message,
                        "triggered_at": alert.triggered_at.isoformat(),
                        "metric_name": alert.metric_name,
                        "current_value": float(alert.current_value) if alert.current_value else None,
                        "threshold_value": float(alert.threshold_value) if alert.threshold_value else None,
                        "suggested_actions": alert.suggested_actions or []
                    }
                    for alert in alerts
                ]
                
        except Exception as e:
            self.logger.error(f"Failed to get alerts: {e}")
            return []
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data for executive view."""
        try:
            # Get current metrics and alerts in parallel
            current_metrics, alerts = await asyncio.gather(
                self.get_current_metrics(),
                self.get_alerts()
            )
            
            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": {
                    "revenue_growth": float(current_metrics.revenue_growth) if current_metrics.revenue_growth else None,
                    "user_acquisition_rate": float(current_metrics.user_acquisition_rate) if current_metrics.user_acquisition_rate else None,
                    "system_uptime": float(current_metrics.system_uptime) if current_metrics.system_uptime else None,
                    "agent_utilization": float(current_metrics.agent_utilization) if current_metrics.agent_utilization else None,
                    "customer_satisfaction": float(current_metrics.customer_satisfaction) if current_metrics.customer_satisfaction else None,
                    "total_active_users": current_metrics.total_active_users,
                    "total_agents": current_metrics.total_agents,
                    "active_agents": current_metrics.active_agents,
                    "success_rate": float(current_metrics.success_rate) if current_metrics.success_rate else None,
                    "conversion_rate": float(current_metrics.conversion_rate) if current_metrics.conversion_rate else None,
                    "efficiency_score": float(current_metrics.efficiency_score) if current_metrics.efficiency_score else None
                },
                "alerts": alerts,
                "health_status": "healthy" if len([a for a in alerts if a["level"] == "critical"]) == 0 else "critical"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard data: {e}")
            raise


# Global instance
_executive_dashboard_instance: Optional[ExecutiveDashboard] = None

async def get_executive_dashboard() -> ExecutiveDashboard:
    """Get or create executive dashboard instance."""
    global _executive_dashboard_instance
    if _executive_dashboard_instance is None:
        _executive_dashboard_instance = ExecutiveDashboard()
    return _executive_dashboard_instance