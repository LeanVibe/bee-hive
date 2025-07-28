"""
Health Monitor for LeanVibe Agent Hive 2.0

Advanced health monitoring system with agent health checks, performance degradation
detection, predictive health analytics, and automated recovery mechanisms.
"""

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import json

import structlog
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .redis import get_message_broker, get_session_cache, AgentMessageBroker, SessionCache
from .database import get_session
from .performance_metrics_collector import PerformanceMetricsCollector, MetricType
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.agent_performance import WorkloadSnapshot, AgentPerformanceHistory
from ..models.task import Task, TaskStatus, TaskPriority

logger = structlog.get_logger()


class HealthStatus(Enum):
    """Agent health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    RECOVERING = "recovering"
    UNKNOWN = "unknown"


class HealthCheckType(Enum):
    """Types of health checks."""
    HEARTBEAT = "heartbeat"
    PERFORMANCE = "performance"
    RESOURCE_USAGE = "resource_usage"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    TASK_COMPLETION = "task_completion"
    MEMORY_LEAK = "memory_leak"
    CONNECTIVITY = "connectivity"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    check_type: HealthCheckType
    check_function: str  # Method name to call
    interval_seconds: int = 60
    timeout_seconds: int = 30
    failure_threshold: int = 3  # Number of consecutive failures before marking unhealthy
    recovery_threshold: int = 2  # Number of consecutive successes to mark healthy
    enabled: bool = True
    priority: int = 5  # 1=highest, 10=lowest
    
    # State tracking
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check_time: Optional[datetime] = None
    last_check_result: Optional[bool] = None
    last_error: Optional[str] = None
    
    def is_due_for_check(self) -> bool:
        """Check if health check is due to run."""
        if not self.enabled:
            return False
        
        if self.last_check_time is None:
            return True
        
        return (datetime.utcnow() - self.last_check_time).seconds >= self.interval_seconds
    
    def record_success(self) -> None:
        """Record successful health check."""
        self.consecutive_failures = 0
        self.consecutive_successes += 1
        self.last_check_result = True
        self.last_error = None
        self.last_check_time = datetime.utcnow()
    
    def record_failure(self, error_message: str = "") -> None:
        """Record failed health check."""
        self.consecutive_successes = 0
        self.consecutive_failures += 1
        self.last_check_result = False
        self.last_error = error_message
        self.last_check_time = datetime.utcnow()
    
    def is_healthy(self) -> bool:
        """Check if health check is currently healthy."""
        return self.consecutive_failures < self.failure_threshold
    
    def is_recovering(self) -> bool:
        """Check if health check is in recovery state."""
        return (self.consecutive_failures >= self.failure_threshold and 
                0 < self.consecutive_successes < self.recovery_threshold)


@dataclass
class HealthAlert:
    """Health alert message."""
    id: str
    agent_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "alert_type": self.alert_type,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class AgentHealthProfile:
    """Comprehensive health profile for an agent."""
    agent_id: str
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    health_score: float = 1.0  # 0.0 to 1.0
    checks: Dict[str, HealthCheck] = field(default_factory=dict)
    alerts: List[HealthAlert] = field(default_factory=list)
    degradation_trend: Optional[str] = None  # "improving", "stable", "degrading"
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    # Performance tracking
    response_time_history: deque = field(default_factory=lambda: deque(maxlen=50))
    error_rate_history: deque = field(default_factory=lambda: deque(maxlen=50))
    resource_usage_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
    def update_health_score(self) -> None:
        """Calculate and update overall health score."""
        if not self.checks:
            self.health_score = 0.5  # Unknown state
            return
        
        # Weight health checks by priority
        total_weight = 0
        weighted_score = 0
        
        for check in self.checks.values():
            if not check.enabled:
                continue
            
            weight = (11 - check.priority) / 10.0  # Higher priority = higher weight
            check_score = 1.0 if check.is_healthy() else 0.0
            
            # Partial credit for recovering checks
            if check.is_recovering():
                check_score = 0.5
            
            weighted_score += check_score * weight
            total_weight += weight
        
        if total_weight > 0:
            self.health_score = weighted_score / total_weight
        else:
            self.health_score = 0.5
        
        # Update overall status based on score
        if self.health_score >= 0.9:
            self.overall_status = HealthStatus.HEALTHY
        elif self.health_score >= 0.7:
            self.overall_status = HealthStatus.DEGRADED
        elif self.health_score >= 0.3:
            self.overall_status = HealthStatus.CRITICAL
        else:
            self.overall_status = HealthStatus.FAILED
        
        # Check for recovery status
        recovering_checks = [c for c in self.checks.values() if c.is_recovering()]
        if recovering_checks and self.overall_status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]:
            self.overall_status = HealthStatus.RECOVERING
        
        self.last_updated = datetime.utcnow()
    
    def add_alert(self, alert: HealthAlert) -> None:
        """Add a new health alert."""
        self.alerts.append(alert)
        # Keep only recent alerts (last 100)
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def get_active_alerts(self) -> List[HealthAlert]:
        """Get list of unresolved alerts."""
        return [alert for alert in self.alerts if not alert.resolved]


class HealthMonitor:
    """
    Advanced health monitoring system for agent orchestration.
    
    Features:
    - Comprehensive health checks with configurable thresholds
    - Performance degradation detection with trend analysis
    - Predictive health analytics based on historical patterns
    - Automated alert generation and escalation
    - Health-based recovery recommendations
    - Real-time health dashboards and metrics
    """
    
    def __init__(
        self,
        metrics_collector: PerformanceMetricsCollector,
        redis_client=None,
        session_factory: Optional[Callable] = None
    ):
        self.metrics_collector = metrics_collector
        self.redis_client = redis_client
        self.session_factory = session_factory or get_session
        
        # Health monitoring state
        self.agent_profiles: Dict[str, AgentHealthProfile] = {}
        self.global_alerts: List[HealthAlert] = []
        
        # Monitoring control
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.check_interval = 30  # seconds
        
        # Performance tracking
        self.health_check_latencies: deque = deque(maxlen=100)
        self.alert_history: deque = deque(maxlen=500)
        
        # Configuration
        self.config = {
            "default_check_interval": 60,
            "heartbeat_timeout": 30,
            "performance_degradation_threshold": 0.3,
            "error_rate_threshold": 5.0,  # percent
            "response_time_threshold": 5000,  # milliseconds
            "memory_growth_threshold": 1.5,  # multiplier
            "alert_cooldown_seconds": 300,
            "max_alerts_per_agent": 50,
            "health_score_window": 10,  # number of checks to consider
            "predictive_window_hours": 4,
            "auto_recovery_enabled": True
        }
        
        # Initialize default health checks
        self._initialize_default_health_checks()
        
        logger.info("HealthMonitor initialized", config=self.config)
    
    def _initialize_default_health_checks(self) -> None:
        """Initialize default health check templates."""
        self.default_checks = {
            "heartbeat": HealthCheck(
                name="heartbeat",
                check_type=HealthCheckType.HEARTBEAT,
                check_function="check_agent_heartbeat",
                interval_seconds=30,
                timeout_seconds=10,
                failure_threshold=3,
                priority=1
            ),
            "performance": HealthCheck(
                name="performance",
                check_type=HealthCheckType.PERFORMANCE,
                check_function="check_agent_performance",
                interval_seconds=60,
                timeout_seconds=30,
                failure_threshold=2,
                priority=2
            ),
            "resource_usage": HealthCheck(
                name="resource_usage",
                check_type=HealthCheckType.RESOURCE_USAGE,
                check_function="check_resource_usage",
                interval_seconds=90,
                timeout_seconds=15,
                failure_threshold=3,
                priority=3
            ),
            "error_rate": HealthCheck(
                name="error_rate",
                check_type=HealthCheckType.ERROR_RATE,
                check_function="check_error_rate",
                interval_seconds=120,
                timeout_seconds=20,
                failure_threshold=2,
                priority=2
            ),
            "response_time": HealthCheck(
                name="response_time",
                check_type=HealthCheckType.RESPONSE_TIME,
                check_function="check_response_time",
                interval_seconds=90,
                timeout_seconds=20,
                failure_threshold=3,
                priority=3
            ),
            "memory_leak": HealthCheck(
                name="memory_leak",
                check_type=HealthCheckType.MEMORY_LEAK,
                check_function="check_memory_leak",
                interval_seconds=300,
                timeout_seconds=30,
                failure_threshold=2,
                priority=4
            )
        }
    
    async def start_monitoring(self) -> None:
        """Start health monitoring for all agents."""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Health monitoring started", interval=self.check_interval)
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main health monitoring loop."""
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                # Discover agents to monitor
                agent_ids = await self._discover_agents()
                
                # Initialize profiles for new agents
                for agent_id in agent_ids:
                    if agent_id not in self.agent_profiles:
                        await self._initialize_agent_profile(agent_id)
                
                # Run health checks for all agents
                await self._run_health_checks()
                
                # Update health scores and status
                await self._update_health_status()
                
                # Generate alerts for unhealthy agents
                await self._generate_health_alerts()
                
                # Store health metrics
                await self._store_health_metrics()
                
                # Detect degradation trends
                await self._detect_degradation_trends()
                
                # Record monitoring performance
                monitoring_time = time.time() - start_time
                self.health_check_latencies.append(monitoring_time)
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error("Error in health monitoring loop", error=str(e))
                await asyncio.sleep(self.check_interval)
    
    async def _discover_agents(self) -> List[str]:
        """Discover active agents to monitor."""
        try:
            async with self.session_factory() as session:
                # Get agents that have recent activity
                query = select(Agent.id).where(
                    Agent.status.in_([AgentStatus.ACTIVE, AgentStatus.BUSY, AgentStatus.IDLE])
                )
                result = await session.execute(query)
                agents = result.scalars().all()
                
                return [str(agent_id) for agent_id in agents]
        
        except Exception as e:
            logger.error("Error discovering agents", error=str(e))
            return []
    
    async def _initialize_agent_profile(self, agent_id: str) -> None:
        """Initialize health profile for new agent."""
        profile = AgentHealthProfile(agent_id=agent_id)
        
        # Add default health checks
        for check_name, check_template in self.default_checks.items():
            # Create a copy of the template for this agent
            profile.checks[check_name] = HealthCheck(
                name=check_template.name,
                check_type=check_template.check_type,
                check_function=check_template.check_function,
                interval_seconds=check_template.interval_seconds,
                timeout_seconds=check_template.timeout_seconds,
                failure_threshold=check_template.failure_threshold,
                recovery_threshold=check_template.recovery_threshold,
                enabled=check_template.enabled,
                priority=check_template.priority
            )
        
        self.agent_profiles[agent_id] = profile
        
        logger.info("Agent health profile initialized",
                   agent_id=agent_id,
                   checks_count=len(profile.checks))
    
    async def _run_health_checks(self) -> None:
        """Run health checks for all monitored agents."""
        check_tasks = []
        
        for agent_id, profile in self.agent_profiles.items():
            for check_name, health_check in profile.checks.items():
                if health_check.is_due_for_check():
                    task = asyncio.create_task(
                        self._execute_health_check(agent_id, health_check)
                    )
                    check_tasks.append(task)
        
        if check_tasks:
            await asyncio.gather(*check_tasks, return_exceptions=True)
    
    async def _execute_health_check(self, agent_id: str, health_check: HealthCheck) -> None:
        """Execute a single health check."""
        try:
            start_time = time.time()
            
            # Get the check function
            if hasattr(self, health_check.check_function):
                check_function = getattr(self, health_check.check_function)
                
                # Run the check with timeout
                try:
                    result = await asyncio.wait_for(
                        check_function(agent_id),
                        timeout=health_check.timeout_seconds
                    )
                    
                    if result:
                        health_check.record_success()
                    else:
                        health_check.record_failure("Health check returned False")
                
                except asyncio.TimeoutError:
                    health_check.record_failure(f"Health check timed out after {health_check.timeout_seconds}s")
                
                except Exception as e:
                    health_check.record_failure(f"Health check error: {str(e)}")
            
            else:
                health_check.record_failure(f"Unknown check function: {health_check.check_function}")
            
            check_time = time.time() - start_time
            
            logger.debug("Health check executed",
                        agent_id=agent_id,
                        check_name=health_check.name,
                        result=health_check.last_check_result,
                        duration_ms=check_time * 1000)
        
        except Exception as e:
            logger.error("Error executing health check",
                        agent_id=agent_id,
                        check_name=health_check.name,
                        error=str(e))
    
    async def check_agent_heartbeat(self, agent_id: str) -> bool:
        """Check if agent is responding to heartbeat."""
        try:
            async with self.session_factory() as session:
                # Check if agent has recent activity
                query = select(WorkloadSnapshot).where(
                    WorkloadSnapshot.agent_id == uuid.UUID(agent_id),
                    WorkloadSnapshot.snapshot_time >= datetime.utcnow() - timedelta(minutes=5)
                ).limit(1)
                
                result = await session.execute(query)
                snapshot = result.scalar_one_or_none()
                
                return snapshot is not None
        
        except Exception as e:
            logger.error("Error checking agent heartbeat", agent_id=agent_id, error=str(e))
            return False
    
    async def check_agent_performance(self, agent_id: str) -> bool:
        """Check agent performance metrics."""
        try:
            # Get performance summary from metrics collector
            performance_data = await self.metrics_collector.get_performance_summary(agent_id)
            
            if "error" in performance_data:
                return False
            
            health_score = performance_data.get("health_score", 0.5)
            return health_score > 0.7  # Consider healthy if score > 70%
        
        except Exception as e:
            logger.error("Error checking agent performance", agent_id=agent_id, error=str(e))
            return False
    
    async def check_resource_usage(self, agent_id: str) -> bool:
        """Check agent resource usage."""
        try:
            async with self.session_factory() as session:
                # Get recent resource usage
                query = select(WorkloadSnapshot).where(
                    WorkloadSnapshot.agent_id == uuid.UUID(agent_id),
                    WorkloadSnapshot.snapshot_time >= datetime.utcnow() - timedelta(minutes=10)
                ).order_by(WorkloadSnapshot.snapshot_time.desc()).limit(1)
                
                result = await session.execute(query)
                snapshot = result.scalar_one_or_none()
                
                if not snapshot:
                    return False
                
                # Check if resource usage is within acceptable limits
                memory_ok = (snapshot.memory_usage_mb or 0) < 2000  # 2GB limit
                cpu_ok = (snapshot.cpu_usage_percent or 0) < 90     # 90% CPU limit
                context_ok = snapshot.context_usage_percent < 95    # 95% context limit
                
                return memory_ok and cpu_ok and context_ok
        
        except Exception as e:
            logger.error("Error checking resource usage", agent_id=agent_id, error=str(e))
            return False
    
    async def check_error_rate(self, agent_id: str) -> bool:
        """Check agent error rate."""
        try:
            async with self.session_factory() as session:
                # Get recent error rate
                query = select(WorkloadSnapshot).where(
                    WorkloadSnapshot.agent_id == uuid.UUID(agent_id),
                    WorkloadSnapshot.snapshot_time >= datetime.utcnow() - timedelta(minutes=30)
                ).order_by(WorkloadSnapshot.snapshot_time.desc()).limit(5)
                
                result = await session.execute(query)
                snapshots = result.scalars().all()
                
                if not snapshots:
                    return True  # No data, assume healthy
                
                # Calculate average error rate
                error_rates = [s.error_rate_percent for s in snapshots]
                avg_error_rate = statistics.mean(error_rates)
                
                return avg_error_rate < self.config["error_rate_threshold"]
        
        except Exception as e:
            logger.error("Error checking error rate", agent_id=agent_id, error=str(e))
            return False
    
    async def check_response_time(self, agent_id: str) -> bool:
        """Check agent response time."""
        try:
            async with self.session_factory() as session:
                # Get recent response times
                query = select(WorkloadSnapshot).where(
                    WorkloadSnapshot.agent_id == uuid.UUID(agent_id),
                    WorkloadSnapshot.snapshot_time >= datetime.utcnow() - timedelta(minutes=15),
                    WorkloadSnapshot.average_response_time_ms.isnot(None)
                ).order_by(WorkloadSnapshot.snapshot_time.desc()).limit(5)
                
                result = await session.execute(query)
                snapshots = result.scalars().all()
                
                if not snapshots:
                    return True  # No data, assume healthy
                
                # Calculate average response time
                response_times = [s.average_response_time_ms for s in snapshots if s.average_response_time_ms]
                if not response_times:
                    return True
                
                avg_response_time = statistics.mean(response_times)
                
                return avg_response_time < self.config["response_time_threshold"]
        
        except Exception as e:
            logger.error("Error checking response time", agent_id=agent_id, error=str(e))
            return False
    
    async def check_memory_leak(self, agent_id: str) -> bool:
        """Check for memory leaks in agent."""
        try:
            async with self.session_factory() as session:
                # Get memory usage trend over time
                query = select(WorkloadSnapshot).where(
                    WorkloadSnapshot.agent_id == uuid.UUID(agent_id),
                    WorkloadSnapshot.snapshot_time >= datetime.utcnow() - timedelta(hours=2),
                    WorkloadSnapshot.memory_usage_mb.isnot(None)
                ).order_by(WorkloadSnapshot.snapshot_time.asc())
                
                result = await session.execute(query)
                snapshots = result.scalars().all()
                
                if len(snapshots) < 10:  # Need sufficient data
                    return True
                
                # Check if memory usage is consistently growing
                memory_values = [s.memory_usage_mb for s in snapshots]
                first_half = memory_values[:len(memory_values)//2]
                second_half = memory_values[len(memory_values)//2:]
                
                avg_first = statistics.mean(first_half)
                avg_second = statistics.mean(second_half)
                
                # Flag as potential leak if memory grew by more than threshold
                growth_ratio = avg_second / avg_first if avg_first > 0 else 1.0
                
                return growth_ratio < self.config["memory_growth_threshold"]
        
        except Exception as e:
            logger.error("Error checking memory leak", agent_id=agent_id, error=str(e))
            return False
    
    async def _update_health_status(self) -> None:
        """Update health status for all agent profiles."""
        for profile in self.agent_profiles.values():
            profile.update_health_score()
    
    async def _generate_health_alerts(self) -> None:
        """Generate alerts for unhealthy agents."""
        for agent_id, profile in self.agent_profiles.items():
            # Check for new alerts based on health status
            if profile.overall_status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
                await self._create_health_alert(
                    agent_id=agent_id,
                    alert_type="agent_unhealthy",
                    severity=AlertSeverity.CRITICAL if profile.overall_status == HealthStatus.FAILED else AlertSeverity.ERROR,
                    message=f"Agent health status: {profile.overall_status.value}",
                    details={
                        "health_score": profile.health_score,
                        "failed_checks": [
                            check.name for check in profile.checks.values()
                            if not check.is_healthy()
                        ]
                    }
                )
            
            # Check individual health check failures
            for check_name, check in profile.checks.items():
                if not check.is_healthy() and check.consecutive_failures == check.failure_threshold:
                    # New failure threshold reached
                    await self._create_health_alert(
                        agent_id=agent_id,
                        alert_type=f"health_check_failed",
                        severity=AlertSeverity.WARNING if check.priority > 3 else AlertSeverity.ERROR,
                        message=f"Health check '{check_name}' failed {check.failure_threshold} times",
                        details={
                            "check_type": check.check_type.value,
                            "consecutive_failures": check.consecutive_failures,
                            "last_error": check.last_error
                        }
                    )
    
    async def _create_health_alert(
        self,
        agent_id: str,
        alert_type: str,
        severity: AlertSeverity,
        message: str,
        details: Dict[str, Any]
    ) -> None:
        """Create a new health alert."""
        try:
            # Check for duplicate recent alerts
            recent_alerts = [
                alert for alert in self.agent_profiles[agent_id].get_active_alerts()
                if (alert.alert_type == alert_type and 
                    (datetime.utcnow() - alert.timestamp).seconds < self.config["alert_cooldown_seconds"])
            ]
            
            if recent_alerts:
                return  # Skip duplicate alert
            
            alert = HealthAlert(
                id=str(uuid.uuid4()),
                agent_id=agent_id,
                alert_type=alert_type,
                severity=severity,
                message=message,
                details=details,
                timestamp=datetime.utcnow()
            )
            
            # Add to agent profile
            self.agent_profiles[agent_id].add_alert(alert)
            
            # Add to global alerts
            self.global_alerts.append(alert)
            self.alert_history.append(alert)
            
            # Store alert to Redis for real-time notifications
            if self.redis_client:
                try:
                    alert_key = "health_monitor:alerts"
                    await self.redis_client.lpush(alert_key, json.dumps(alert.to_dict()))
                    await self.redis_client.ltrim(alert_key, 0, 999)  # Keep last 1000 alerts
                    await self.redis_client.expire(alert_key, 86400)  # 24 hours TTL
                except Exception as e:
                    logger.error("Error storing alert to Redis", error=str(e))
            
            logger.warning("Health alert generated",
                          agent_id=agent_id,
                          alert_type=alert_type,
                          severity=severity.value,
                          message=message)
        
        except Exception as e:
            logger.error("Error creating health alert", error=str(e))
    
    async def _store_health_metrics(self) -> None:
        """Store health metrics to metrics collector."""
        try:
            # System-wide health metrics
            total_agents = len(self.agent_profiles)
            healthy_agents = len([p for p in self.agent_profiles.values() if p.overall_status == HealthStatus.HEALTHY])
            degraded_agents = len([p for p in self.agent_profiles.values() if p.overall_status == HealthStatus.DEGRADED])
            critical_agents = len([p for p in self.agent_profiles.values() if p.overall_status == HealthStatus.CRITICAL])
            failed_agents = len([p for p in self.agent_profiles.values() if p.overall_status == HealthStatus.FAILED])
            
            await self.metrics_collector.record_custom_metric(
                "system", "health.total_agents", total_agents, MetricType.GAUGE
            )
            await self.metrics_collector.record_custom_metric(
                "system", "health.healthy_agents", healthy_agents, MetricType.GAUGE
            )
            await self.metrics_collector.record_custom_metric(
                "system", "health.degraded_agents", degraded_agents, MetricType.GAUGE
            )
            await self.metrics_collector.record_custom_metric(
                "system", "health.critical_agents", critical_agents, MetricType.GAUGE
            )
            await self.metrics_collector.record_custom_metric(
                "system", "health.failed_agents", failed_agents, MetricType.GAUGE
            )
            
            # Average health score
            if self.agent_profiles:
                avg_health_score = statistics.mean([p.health_score for p in self.agent_profiles.values()])
                await self.metrics_collector.record_custom_metric(
                    "system", "health.avg_health_score", avg_health_score, MetricType.GAUGE
                )
            
            # Individual agent health scores
            for agent_id, profile in self.agent_profiles.items():
                await self.metrics_collector.record_custom_metric(
                    agent_id, "health_score", profile.health_score, MetricType.GAUGE, "agent"
                )
        
        except Exception as e:
            logger.error("Error storing health metrics", error=str(e))
    
    async def _detect_degradation_trends(self) -> None:
        """Detect health degradation trends."""
        try:
            for agent_id, profile in self.agent_profiles.items():
                # Analyze health score trend
                if len(profile.response_time_history) >= 5:
                    recent_scores = list(profile.response_time_history)[-5:]
                    if len(recent_scores) > 1:
                        # Simple trend detection
                        trend_slope = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
                        
                        if trend_slope > 0.1:
                            profile.degradation_trend = "degrading"
                        elif trend_slope < -0.1:
                            profile.degradation_trend = "improving"
                        else:
                            profile.degradation_trend = "stable"
                        
                        # Generate predictive alert for degrading trend
                        if (profile.degradation_trend == "degrading" and 
                            profile.overall_status == HealthStatus.HEALTHY):
                            
                            await self._create_health_alert(
                                agent_id=agent_id,
                                alert_type="degradation_trend",
                                severity=AlertSeverity.WARNING,
                                message="Agent showing degradation trend",
                                details={
                                    "trend": profile.degradation_trend,
                                    "current_health_score": profile.health_score
                                }
                            )
        
        except Exception as e:
            logger.error("Error detecting degradation trends", error=str(e))
    
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        try:
            # System-wide statistics
            total_agents = len(self.agent_profiles)
            status_counts = defaultdict(int)
            
            for profile in self.agent_profiles.values():
                status_counts[profile.overall_status.value] += 1
            
            # Active alerts
            active_alerts = []
            for profile in self.agent_profiles.values():
                active_alerts.extend(profile.get_active_alerts())
            
            # Alert severity distribution
            alert_severity_counts = defaultdict(int)
            for alert in active_alerts:
                alert_severity_counts[alert.severity.value] += 1
            
            # Health check statistics
            check_stats = defaultdict(lambda: {"total": 0, "healthy": 0, "failed": 0})
            for profile in self.agent_profiles.values():
                for check in profile.checks.values():
                    check_stats[check.check_type.value]["total"] += 1
                    if check.is_healthy():
                        check_stats[check.check_type.value]["healthy"] += 1
                    else:
                        check_stats[check.check_type.value]["failed"] += 1
            
            # Performance metrics
            if self.health_check_latencies:
                avg_check_latency = statistics.mean(self.health_check_latencies)
                max_check_latency = max(self.health_check_latencies)
            else:
                avg_check_latency = max_check_latency = 0
            
            return {
                "system_health": {
                    "total_agents": total_agents,
                    "status_distribution": dict(status_counts),
                    "overall_health_percentage": (
                        status_counts.get("healthy", 0) / total_agents * 100 
                        if total_agents > 0 else 0
                    )
                },
                "alerts": {
                    "active_alerts": len(active_alerts),
                    "severity_distribution": dict(alert_severity_counts),
                    "recent_alerts": [alert.to_dict() for alert in active_alerts[:10]]
                },
                "health_checks": {
                    "check_statistics": dict(check_stats),
                    "monitoring_performance": {
                        "avg_check_latency_ms": avg_check_latency * 1000,
                        "max_check_latency_ms": max_check_latency * 1000,
                        "checks_per_cycle": sum(
                            len(profile.checks) for profile in self.agent_profiles.values()
                        )
                    }
                },
                "trends": {
                    "degrading_agents": len([
                        p for p in self.agent_profiles.values() 
                        if p.degradation_trend == "degrading"
                    ]),
                    "improving_agents": len([
                        p for p in self.agent_profiles.values() 
                        if p.degradation_trend == "improving"
                    ])
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error("Error getting health summary", error=str(e))
            return {"error": str(e)}
    
    async def get_agent_health_details(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed health information for specific agent."""
        try:
            if agent_id not in self.agent_profiles:
                return {"error": f"Agent {agent_id} not found"}
            
            profile = self.agent_profiles[agent_id]
            
            # Health check details
            check_details = {}
            for check_name, check in profile.checks.items():
                check_details[check_name] = {
                    "type": check.check_type.value,
                    "enabled": check.enabled,
                    "healthy": check.is_healthy(),
                    "consecutive_failures": check.consecutive_failures,
                    "consecutive_successes": check.consecutive_successes,
                    "last_check_time": check.last_check_time.isoformat() if check.last_check_time else None,
                    "last_check_result": check.last_check_result,
                    "last_error": check.last_error,
                    "priority": check.priority
                }
            
            # Recent alerts
            recent_alerts = [alert.to_dict() for alert in profile.alerts[-10:]]
            active_alerts = [alert.to_dict() for alert in profile.get_active_alerts()]
            
            return {
                "agent_id": agent_id,
                "overall_status": profile.overall_status.value,
                "health_score": profile.health_score,
                "degradation_trend": profile.degradation_trend,
                "last_updated": profile.last_updated.isoformat(),
                "health_checks": check_details,
                "alerts": {
                    "active": active_alerts,
                    "recent": recent_alerts,
                    "total_count": len(profile.alerts)
                },
                "recommendations": await self._get_health_recommendations(agent_id)
            }
        
        except Exception as e:
            logger.error("Error getting agent health details", agent_id=agent_id, error=str(e))
            return {"error": str(e)}
    
    async def _get_health_recommendations(self, agent_id: str) -> List[str]:
        """Get health improvement recommendations for agent."""
        recommendations = []
        
        try:
            if agent_id not in self.agent_profiles:
                return recommendations
            
            profile = self.agent_profiles[agent_id]
            
            # Check for specific issues and recommend actions
            for check_name, check in profile.checks.items():
                if not check.is_healthy():
                    if check.check_type == HealthCheckType.RESOURCE_USAGE:
                        recommendations.append("Consider reducing resource usage or scaling up capacity")
                    elif check.check_type == HealthCheckType.ERROR_RATE:
                        recommendations.append("Investigate and fix sources of errors")
                    elif check.check_type == HealthCheckType.RESPONSE_TIME:
                        recommendations.append("Optimize task processing or increase processing capacity")
                    elif check.check_type == HealthCheckType.MEMORY_LEAK:
                        recommendations.append("Investigate potential memory leaks and restart agent if necessary")
                    elif check.check_type == HealthCheckType.HEARTBEAT:
                        recommendations.append("Check agent connectivity and restart if unresponsive")
            
            # General health recommendations
            if profile.health_score < 0.5:
                recommendations.append("Consider restarting the agent to restore health")
            elif profile.health_score < 0.7:
                recommendations.append("Monitor closely and consider preventive maintenance")
            
            if profile.degradation_trend == "degrading":
                recommendations.append("Proactively investigate degradation causes before issues worsen")
        
        except Exception as e:
            logger.error("Error getting health recommendations", agent_id=agent_id, error=str(e))
        
        return recommendations
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved."""
        try:
            for profile in self.agent_profiles.values():
                for alert in profile.alerts:
                    if alert.id == alert_id and not alert.resolved:
                        alert.resolved = True
                        alert.resolved_at = datetime.utcnow()
                        
                        logger.info("Health alert resolved",
                                   alert_id=alert_id,
                                   agent_id=alert.agent_id)
                        return True
            
            return False
        
        except Exception as e:
            logger.error("Error resolving alert", alert_id=alert_id, error=str(e))
            return False
    
    def add_custom_health_check(
        self,
        agent_id: str,
        check_name: str,
        check_type: HealthCheckType,
        check_function: str,
        **kwargs
    ) -> bool:
        """Add custom health check for specific agent."""
        try:
            if agent_id not in self.agent_profiles:
                return False
            
            health_check = HealthCheck(
                name=check_name,
                check_type=check_type,
                check_function=check_function,
                **kwargs
            )
            
            self.agent_profiles[agent_id].checks[check_name] = health_check
            
            logger.info("Custom health check added",
                       agent_id=agent_id,
                       check_name=check_name,
                       check_type=check_type.value)
            
            return True
        
        except Exception as e:
            logger.error("Error adding custom health check", error=str(e))
            return False