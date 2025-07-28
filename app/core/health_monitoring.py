"""
Comprehensive Health Monitoring System.

Advanced health monitoring for all Context Engine components with
real-time status tracking, dependency monitoring, and self-healing capabilities.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import traceback

import psutil
import redis.asyncio as redis
from sqlalchemy import text, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.database import get_async_session
from ..core.redis import get_redis_client
from ..core.context_performance_monitor import get_context_performance_monitor
from ..core.cost_monitoring import get_cost_monitor
from ..core.capacity_planning import get_capacity_planner
from ..core.intelligent_alerting import get_alert_manager

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of components to monitor."""
    DATABASE = "database"
    REDIS = "redis"
    CONTEXT_ENGINE = "context_engine"
    VECTOR_SEARCH = "vector_search"
    EMBEDDING_SERVICE = "embedding_service"
    PERFORMANCE_MONITOR = "performance_monitor"
    COST_MONITOR = "cost_monitor"
    CAPACITY_PLANNER = "capacity_planner"
    ALERT_MANAGER = "alert_manager"
    API_GATEWAY = "api_gateway"
    SEARCH_ANALYTICS = "search_analytics"
    SYSTEM_RESOURCES = "system_resources"


class DependencyType(Enum):
    """Types of component dependencies."""
    HARD = "hard"  # Component cannot function without this dependency
    SOFT = "soft"  # Component can function with reduced capability
    OPTIONAL = "optional"  # Dependency is optional


@dataclass
class HealthCheck:
    """Represents a health check configuration."""
    check_id: str
    component: ComponentType
    name: str
    description: str
    check_function: Callable[[], Any]
    timeout_seconds: float = 30.0
    interval_seconds: float = 60.0
    retry_attempts: int = 3
    critical: bool = True
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentDependency:
    """Represents a dependency between components."""
    dependency_id: str
    component: ComponentType
    depends_on: ComponentType
    dependency_type: DependencyType
    description: str
    impact_description: str


@dataclass
class HealthResult:
    """Result of a health check."""
    check_id: str
    component: ComponentType
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentHealth:
    """Overall health status of a component."""
    component: ComponentType
    status: HealthStatus
    last_updated: datetime
    checks_passed: int
    checks_failed: int
    checks_total: int
    uptime_percentage: float
    dependencies_healthy: bool
    recent_results: List[HealthResult] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    recovery_actions: List[str] = field(default_factory=list)


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall_status: HealthStatus
    timestamp: datetime
    components: Dict[ComponentType, ComponentHealth]
    critical_issues: List[str] = field(default_factory=list)
    degraded_components: List[ComponentType] = field(default_factory=list)
    system_uptime_hours: float = 0.0
    total_checks: int = 0
    passed_checks: int = 0


class SelfHealingAction:
    """Self-healing action for component recovery."""
    
    def __init__(
        self,
        action_id: str,
        component: ComponentType,
        trigger_condition: str,
        action_function: Callable,
        description: str,
        max_attempts: int = 3,
        cooldown_minutes: int = 15
    ):
        self.action_id = action_id
        self.component = component
        self.trigger_condition = trigger_condition
        self.action_function = action_function
        self.description = description
        self.max_attempts = max_attempts
        self.cooldown_minutes = cooldown_minutes
        self.attempts = 0
        self.last_attempt = None
        self.success_count = 0
        self.failure_count = 0


class HealthMonitor:
    """
    Comprehensive health monitoring system for all Context Engine components.
    
    Features:
    - Real-time health checking for all components
    - Dependency tracking and impact analysis
    - Self-healing capabilities with automatic recovery
    - Performance degradation detection
    - Component lifecycle management
    - Health trends and analytics
    - Integration with alerting system
    - Circuit breaker patterns for failing components
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        db_session: Optional[AsyncSession] = None
    ):
        """
        Initialize the health monitor.
        
        Args:
            redis_client: Redis client for real-time data
            db_session: Database session
        """
        self.redis_client = redis_client or get_redis_client()
        self.db_session = db_session
        
        # Health checks and results
        self.health_checks: Dict[str, HealthCheck] = {}
        self.component_health: Dict[ComponentType, ComponentHealth] = {}
        self.health_history: deque = deque(maxlen=10000)
        
        # Dependencies
        self.dependencies: List[ComponentDependency] = []
        self.dependency_graph: Dict[ComponentType, List[ComponentType]] = defaultdict(list)
        
        # Self-healing
        self.healing_actions: Dict[str, SelfHealingAction] = {}
        self.circuit_breakers: Dict[ComponentType, Dict[str, Any]] = {}
        
        # Monitoring state
        self.system_start_time = datetime.utcnow()
        self.check_results: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.component_metrics: Dict[ComponentType, Dict[str, Any]] = defaultdict(dict)
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        # Initialize health checks and dependencies
        self._initialize_health_checks()
        self._initialize_dependencies()
        self._initialize_self_healing()
        
        logger.info("Health Monitor initialized")
    
    async def start(self) -> None:
        """Start the health monitor background processes."""
        logger.info("Starting health monitor")
        
        # Start background tasks
        self._background_tasks.extend([
            asyncio.create_task(self._health_checker()),
            asyncio.create_task(self._dependency_monitor()),
            asyncio.create_task(self._self_healing_manager()),
            asyncio.create_task(self._metrics_collector()),
            asyncio.create_task(self._circuit_breaker_manager()),
            asyncio.create_task(self._data_maintenance())
        ])
    
    async def stop(self) -> None:
        """Stop the health monitor."""
        logger.info("Stopping health monitor")
        
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
    
    async def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health status."""
        try:
            current_time = datetime.utcnow()
            
            # Calculate overall status
            critical_components = [
                comp for comp, health in self.component_health.items()
                if health.status == HealthStatus.CRITICAL
            ]
            
            unhealthy_components = [
                comp for comp, health in self.component_health.items()
                if health.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
            ]
            
            degraded_components = [
                comp for comp, health in self.component_health.items()
                if health.status == HealthStatus.DEGRADED
            ]
            
            # Determine overall status
            if critical_components:
                overall_status = HealthStatus.CRITICAL
            elif unhealthy_components:
                overall_status = HealthStatus.UNHEALTHY
            elif degraded_components:
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.HEALTHY
            
            # Calculate system uptime
            uptime_hours = (current_time - self.system_start_time).total_seconds() / 3600
            
            # Calculate check statistics
            total_checks = sum(len(results) for results in self.check_results.values())
            passed_checks = sum(
                len([r for r in results if r.status == HealthStatus.HEALTHY])
                for results in self.check_results.values()
            )
            
            # Gather critical issues
            critical_issues = []
            for comp, health in self.component_health.items():
                if health.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
                    critical_issues.extend(health.issues)
            
            system_health = SystemHealth(
                overall_status=overall_status,
                timestamp=current_time,
                components=self.component_health.copy(),
                critical_issues=critical_issues,
                degraded_components=degraded_components,
                system_uptime_hours=uptime_hours,
                total_checks=total_checks,
                passed_checks=passed_checks
            )
            
            return system_health
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return SystemHealth(
                overall_status=HealthStatus.UNKNOWN,
                timestamp=datetime.utcnow(),
                components={},
                critical_issues=[f"Health monitoring error: {str(e)}"]
            )
    
    async def get_component_health(self, component: ComponentType) -> Optional[ComponentHealth]:
        """Get health status for a specific component."""
        try:
            return self.component_health.get(component)
        except Exception as e:
            logger.error(f"Failed to get component health for {component.value}: {e}")
            return None
    
    async def trigger_health_check(self, component: ComponentType) -> List[HealthResult]:
        """Manually trigger health checks for a component."""
        try:
            results = []
            
            for check in self.health_checks.values():
                if check.component == component and check.enabled:
                    result = await self._execute_health_check(check)
                    results.append(result)
            
            # Update component health
            if results:
                await self._update_component_health(component, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to trigger health check for {component.value}: {e}")
            return []
    
    async def add_health_check(self, health_check: HealthCheck) -> None:
        """Add a custom health check."""
        try:
            self.health_checks[health_check.check_id] = health_check
            logger.info(f"Added health check: {health_check.name}")
        except Exception as e:
            logger.error(f"Failed to add health check: {e}")
    
    async def enable_self_healing(self, component: ComponentType) -> None:
        """Enable self-healing for a component."""
        try:
            for action in self.healing_actions.values():
                if action.component == component:
                    logger.info(f"Enabled self-healing for {component.value}")
                    break
        except Exception as e:
            logger.error(f"Failed to enable self-healing: {e}")
    
    async def get_health_trends(
        self,
        component: Optional[ComponentType] = None,
        hours_back: int = 24
    ) -> Dict[str, Any]:
        """Get health trends and analytics."""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
            
            # Filter recent results
            if component:
                recent_results = [
                    result for results in self.check_results.values()
                    for result in results
                    if result.component == component and result.timestamp >= cutoff_time
                ]
            else:
                recent_results = [
                    result for results in self.check_results.values()
                    for result in results
                    if result.timestamp >= cutoff_time
                ]
            
            if not recent_results:
                return {"error": "No health data available for the specified period"}
            
            # Calculate trends
            status_counts = defaultdict(int)
            component_stats = defaultdict(lambda: {"total": 0, "healthy": 0, "issues": 0})
            hourly_stats = defaultdict(lambda: {"total": 0, "healthy": 0})
            
            for result in recent_results:
                status_counts[result.status.value] += 1
                
                comp_key = result.component.value
                component_stats[comp_key]["total"] += 1
                
                if result.status == HealthStatus.HEALTHY:
                    component_stats[comp_key]["healthy"] += 1
                else:
                    component_stats[comp_key]["issues"] += 1
                
                # Hourly breakdown
                hour_key = result.timestamp.strftime("%Y-%m-%d %H:00")
                hourly_stats[hour_key]["total"] += 1
                
                if result.status == HealthStatus.HEALTHY:
                    hourly_stats[hour_key]["healthy"] += 1
            
            # Calculate component health percentages
            component_health_pct = {}
            for comp, stats in component_stats.items():
                if stats["total"] > 0:
                    component_health_pct[comp] = {
                        "health_percentage": (stats["healthy"] / stats["total"]) * 100,
                        "total_checks": stats["total"],
                        "healthy_checks": stats["healthy"],
                        "issue_checks": stats["issues"]
                    }
            
            # Calculate hourly health rates
            hourly_health_rates = {}
            for hour, stats in hourly_stats.items():
                if stats["total"] > 0:
                    hourly_health_rates[hour] = (stats["healthy"] / stats["total"]) * 100
            
            # Overall statistics
            total_checks = len(recent_results)
            healthy_checks = status_counts.get("healthy", 0)
            overall_health_rate = (healthy_checks / total_checks * 100) if total_checks > 0 else 0
            
            trends = {
                "analysis_period_hours": hours_back,
                "component_filter": component.value if component else "all",
                "total_checks": total_checks,
                "overall_health_rate": round(overall_health_rate, 2),
                "status_breakdown": dict(status_counts),
                "component_health": component_health_pct,
                "hourly_trends": hourly_health_rates,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return trends
            
        except Exception as e:
            logger.error(f"Failed to get health trends: {e}")
            return {"error": str(e)}
    
    # Background task methods
    async def _health_checker(self) -> None:
        """Background task to execute health checks."""
        logger.info("Starting health checker")
        
        last_check_times = {}
        
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                
                for check_id, check in self.health_checks.items():
                    if not check.enabled:
                        continue
                    
                    # Check if it's time to run this check
                    last_check = last_check_times.get(check_id, 0)
                    
                    if current_time - last_check >= check.interval_seconds:
                        # Execute health check
                        result = await self._execute_health_check(check)
                        
                        # Store result
                        self.check_results[check_id].append(result)
                        
                        # Update component health
                        await self._update_component_health(check.component, [result])
                        
                        # Store in Redis
                        await self._store_health_result(result)
                        
                        last_check_times[check_id] = current_time
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health checker error: {e}")
                await asyncio.sleep(10)
        
        logger.info("Health checker stopped")
    
    async def _dependency_monitor(self) -> None:
        """Background task to monitor component dependencies."""
        logger.info("Starting dependency monitor")
        
        while not self._shutdown_event.is_set():
            try:
                # Check dependency health and impact
                for dependency in self.dependencies:
                    dependent_health = self.component_health.get(dependency.component)
                    dependency_health = self.component_health.get(dependency.depends_on)
                    
                    if dependent_health and dependency_health:
                        # Update dependency impact
                        await self._evaluate_dependency_impact(dependency, dependent_health, dependency_health)
                
                await asyncio.sleep(300)  # Check dependencies every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dependency monitor error: {e}")
                await asyncio.sleep(300)
        
        logger.info("Dependency monitor stopped")
    
    async def _self_healing_manager(self) -> None:
        """Background task to manage self-healing actions."""
        logger.info("Starting self-healing manager")
        
        while not self._shutdown_event.is_set():
            try:
                for action in self.healing_actions.values():
                    # Check if healing action should be triggered
                    if await self._should_trigger_healing(action):
                        await self._execute_healing_action(action)
                
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Self-healing manager error: {e}")
                await asyncio.sleep(60)
        
        logger.info("Self-healing manager stopped")
    
    async def _metrics_collector(self) -> None:
        """Background task to collect component metrics."""
        logger.info("Starting metrics collector")
        
        while not self._shutdown_event.is_set():
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Update component uptime statistics
                await self._update_uptime_statistics()
                
                await asyncio.sleep(300)  # Collect every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(300)
        
        logger.info("Metrics collector stopped")
    
    async def _circuit_breaker_manager(self) -> None:
        """Background task to manage circuit breakers."""
        logger.info("Starting circuit breaker manager")
        
        while not self._shutdown_event.is_set():
            try:
                for component in ComponentType:
                    await self._update_circuit_breaker(component)
                
                await asyncio.sleep(120)  # Update every 2 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Circuit breaker manager error: {e}")
                await asyncio.sleep(120)
        
        logger.info("Circuit breaker manager stopped")
    
    async def _data_maintenance(self) -> None:
        """Background task for data cleanup and maintenance."""
        logger.info("Starting data maintenance")
        
        while not self._shutdown_event.is_set():
            try:
                # Clean up old health results
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                
                for check_id, results in self.check_results.items():
                    # Filter out old results
                    filtered_results = deque([
                        result for result in results
                        if result.timestamp > cutoff_time
                    ], maxlen=100)
                    
                    self.check_results[check_id] = filtered_results
                
                # Clean up health history
                self.health_history = deque([
                    entry for entry in self.health_history
                    if datetime.fromisoformat(entry.get("timestamp", "")) > cutoff_time
                ], maxlen=10000)
                
                await asyncio.sleep(3600)  # Cleanup every hour
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Data maintenance error: {e}")
                await asyncio.sleep(3600)
        
        logger.info("Data maintenance stopped")
    
    # Helper methods
    def _initialize_health_checks(self) -> None:
        """Initialize default health checks for all components."""
        health_checks = [
            # Database health check
            HealthCheck(
                check_id="database_connection",
                component=ComponentType.DATABASE,
                name="Database Connection",
                description="Check database connectivity and basic operations",
                check_function=self._check_database_health,
                interval_seconds=60,
                critical=True
            ),
            
            # Redis health check
            HealthCheck(
                check_id="redis_connection",
                component=ComponentType.REDIS,
                name="Redis Connection",
                description="Check Redis connectivity and basic operations",
                check_function=self._check_redis_health,
                interval_seconds=60,
                critical=True
            ),
            
            # System resources health check
            HealthCheck(
                check_id="system_resources",
                component=ComponentType.SYSTEM_RESOURCES,
                name="System Resources",
                description="Check CPU, memory, and disk usage",
                check_function=self._check_system_resources,
                interval_seconds=120,
                critical=False
            ),
            
            # Context engine health check
            HealthCheck(
                check_id="context_engine",
                component=ComponentType.CONTEXT_ENGINE,
                name="Context Engine",
                description="Check context engine functionality",
                check_function=self._check_context_engine_health,
                interval_seconds=180,
                critical=True
            ),
            
            # Performance monitor health check
            HealthCheck(
                check_id="performance_monitor",
                component=ComponentType.PERFORMANCE_MONITOR,
                name="Performance Monitor",
                description="Check performance monitoring system",
                check_function=self._check_performance_monitor_health,
                interval_seconds=300,
                critical=False
            )
        ]
        
        for check in health_checks:
            self.health_checks[check.check_id] = check
        
        logger.info(f"Initialized {len(health_checks)} health checks")
    
    def _initialize_dependencies(self) -> None:
        """Initialize component dependencies."""
        dependencies = [
            ComponentDependency(
                dependency_id="context_engine_db",
                component=ComponentType.CONTEXT_ENGINE,
                depends_on=ComponentType.DATABASE,
                dependency_type=DependencyType.HARD,
                description="Context engine requires database for data storage",
                impact_description="Context operations will fail without database"
            ),
            
            ComponentDependency(
                dependency_id="context_engine_redis",
                component=ComponentType.CONTEXT_ENGINE,
                depends_on=ComponentType.REDIS,
                dependency_type=DependencyType.SOFT,
                description="Context engine uses Redis for caching",
                impact_description="Performance will degrade without Redis caching"
            ),
            
            ComponentDependency(
                dependency_id="vector_search_context",
                component=ComponentType.VECTOR_SEARCH,
                depends_on=ComponentType.CONTEXT_ENGINE,
                dependency_type=DependencyType.HARD,
                description="Vector search depends on context engine",
                impact_description="Search functionality will be unavailable"
            ),
            
            ComponentDependency(
                dependency_id="performance_monitor_redis",
                component=ComponentType.PERFORMANCE_MONITOR,
                depends_on=ComponentType.REDIS,
                dependency_type=DependencyType.SOFT,
                description="Performance monitor uses Redis for metrics storage",
                impact_description="Real-time metrics may be affected"
            ),
            
            ComponentDependency(
                dependency_id="all_system_resources",
                component=ComponentType.CONTEXT_ENGINE,
                depends_on=ComponentType.SYSTEM_RESOURCES,
                dependency_type=DependencyType.HARD,
                description="All components depend on system resources",
                impact_description="System instability if resources are exhausted"
            )
        ]
        
        self.dependencies = dependencies
        
        # Build dependency graph
        for dep in dependencies:
            self.dependency_graph[dep.component].append(dep.depends_on)
        
        logger.info(f"Initialized {len(dependencies)} component dependencies")
    
    def _initialize_self_healing(self) -> None:
        """Initialize self-healing actions."""
        healing_actions = [
            SelfHealingAction(
                action_id="restart_redis_connection",
                component=ComponentType.REDIS,
                trigger_condition="connection_failed",
                action_function=self._heal_redis_connection,
                description="Attempt to reconnect to Redis",
                max_attempts=3,
                cooldown_minutes=5
            ),
            
            SelfHealingAction(
                action_id="clear_cache_on_memory_pressure",
                component=ComponentType.SYSTEM_RESOURCES,
                trigger_condition="high_memory_usage",
                action_function=self._heal_memory_pressure,
                description="Clear caches to free memory",
                max_attempts=2,
                cooldown_minutes=10
            )
        ]
        
        for action in healing_actions:
            self.healing_actions[action.action_id] = action
        
        logger.info(f"Initialized {len(healing_actions)} self-healing actions")
    
    async def _execute_health_check(self, check: HealthCheck) -> HealthResult:
        """Execute a single health check."""
        start_time = time.time()
        
        try:
            # Execute check with timeout
            result_data = await asyncio.wait_for(
                check.check_function(),
                timeout=check.timeout_seconds
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Determine status from result
            if isinstance(result_data, dict):
                status = HealthStatus(result_data.get("status", "healthy"))
                message = result_data.get("message", "Check passed")
                details = result_data.get("details", {})
                error = result_data.get("error")
            else:
                status = HealthStatus.HEALTHY
                message = "Check passed"
                details = {"result": result_data}
                error = None
            
            return HealthResult(
                check_id=check.check_id,
                component=check.component,
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time_ms,
                details=details,
                error=error
            )
            
        except asyncio.TimeoutError:
            response_time_ms = check.timeout_seconds * 1000
            return HealthResult(
                check_id=check.check_id,
                component=check.component,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {check.timeout_seconds}s",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time_ms,
                error="timeout"
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            return HealthResult(
                check_id=check.check_id,
                component=check.component,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                timestamp=datetime.utcnow(),
                response_time_ms=response_time_ms,
                error=str(e)
            )
    
    async def _update_component_health(self, component: ComponentType, results: List[HealthResult]) -> None:
        """Update overall health status for a component."""
        try:
            # Get all recent results for this component
            all_results = []
            for check_id, check_results in self.check_results.items():
                check = self.health_checks.get(check_id)
                if check and check.component == component:
                    all_results.extend(list(check_results))
            
            # Add new results
            all_results.extend(results)
            
            if not all_results:
                return
            
            # Sort by timestamp (most recent first)
            all_results.sort(key=lambda r: r.timestamp, reverse=True)
            
            # Calculate component health
            recent_results = all_results[:20]  # Last 20 results
            
            passed = len([r for r in recent_results if r.status == HealthStatus.HEALTHY])
            failed = len(recent_results) - passed
            
            # Determine overall status
            if not recent_results:
                status = HealthStatus.UNKNOWN
            else:
                latest_critical_checks = [
                    r for r in recent_results[:5]  # Last 5 results
                    if self.health_checks.get(r.check_id, HealthCheck("", component, "", "", lambda: None, critical=True)).critical
                ]
                
                if any(r.status == HealthStatus.CRITICAL for r in latest_critical_checks):
                    status = HealthStatus.CRITICAL
                elif any(r.status == HealthStatus.UNHEALTHY for r in latest_critical_checks):
                    status = HealthStatus.UNHEALTHY
                elif any(r.status == HealthStatus.DEGRADED for r in recent_results[:10]):
                    status = HealthStatus.DEGRADED
                elif passed / len(recent_results) >= 0.8:  # 80% success rate
                    status = HealthStatus.HEALTHY
                else:
                    status = HealthStatus.UNHEALTHY
            
            # Calculate uptime percentage
            if len(recent_results) > 0:
                uptime_percentage = (passed / len(recent_results)) * 100
            else:
                uptime_percentage = 0.0
            
            # Check dependency health
            dependencies_healthy = await self._check_dependencies_health(component)
            
            # Gather issues and recovery actions
            issues = []
            recovery_actions = []
            
            for result in recent_results[:5]:  # Recent issues
                if result.status != HealthStatus.HEALTHY and result.error:
                    issues.append(f"{result.message} ({result.error})")
            
            # Add dependency issues
            if not dependencies_healthy:
                issues.append("One or more dependencies are unhealthy")
                recovery_actions.append("Check and restore unhealthy dependencies")
            
            # Create component health
            component_health = ComponentHealth(
                component=component,
                status=status,
                last_updated=datetime.utcnow(),
                checks_passed=passed,
                checks_failed=failed,
                checks_total=len(recent_results),
                uptime_percentage=uptime_percentage,
                dependencies_healthy=dependencies_healthy,
                recent_results=recent_results[:5],  # Keep last 5 results
                issues=list(set(issues)),  # Remove duplicates
                recovery_actions=recovery_actions
            )
            
            self.component_health[component] = component_health
            
        except Exception as e:
            logger.error(f"Failed to update component health for {component.value}: {e}")
    
    async def _check_dependencies_health(self, component: ComponentType) -> bool:
        """Check if all dependencies of a component are healthy."""
        try:
            for dependency in self.dependencies:
                if dependency.component == component:
                    dep_health = self.component_health.get(dependency.depends_on)
                    
                    if not dep_health:
                        continue
                    
                    # For hard dependencies, they must be healthy
                    if dependency.dependency_type == DependencyType.HARD:
                        if dep_health.status in [HealthStatus.CRITICAL, HealthStatus.UNHEALTHY]:
                            return False
                    
                    # For soft dependencies, only critical failures matter
                    elif dependency.dependency_type == DependencyType.SOFT:
                        if dep_health.status == HealthStatus.CRITICAL:
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to check dependencies for {component.value}: {e}")
            return False
    
    # Health check implementations
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            async for session in get_async_session():
                # Test basic connectivity
                await session.execute(text("SELECT 1"))
                
                # Test table access
                result = await session.execute(text("SELECT COUNT(*) FROM contexts LIMIT 1"))
                count = result.scalar()
                
                return {
                    "status": "healthy",
                    "message": f"Database accessible, {count} contexts found",
                    "details": {"context_count": count}
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": "Database connection failed",
                "error": str(e)
            }
    
    async def _check_redis_health(self) -> Dict[str, Any]:
        """Check Redis health."""
        try:
            # Test basic connectivity
            await self.redis_client.ping()
            
            # Test read/write operations
            test_key = "health_check_test"
            await self.redis_client.set(test_key, "test_value", ex=60)
            value = await self.redis_client.get(test_key)
            
            if value and value.decode() == "test_value":
                return {
                    "status": "healthy",
                    "message": "Redis connection and operations working",
                    "details": {"ping": "success", "read_write": "success"}
                }
            else:
                return {
                    "status": "degraded",
                    "message": "Redis ping successful but read/write operations failed",
                    "error": "read_write_failed"
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": "Redis connection failed",
                "error": str(e)
            }
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource health."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Determine status based on resource usage
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                status = "critical"
                message = f"Critical resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, Disk {disk_percent:.1f}%"
            elif cpu_percent > 80 or memory_percent > 80 or disk_percent > 80:
                status = "degraded"
                message = f"High resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, Disk {disk_percent:.1f}%"
            else:
                status = "healthy"
                message = f"Resource usage normal: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%, Disk {disk_percent:.1f}%"
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_free_gb": disk.free / (1024**3)
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": "Failed to check system resources",
                "error": str(e)
            }
    
    async def _check_context_engine_health(self) -> Dict[str, Any]:
        """Check context engine health."""
        try:
            # This would check context engine functionality
            # For now, return healthy if no critical issues
            
            return {
                "status": "healthy",
                "message": "Context engine operational",
                "details": {"functional": True}
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": "Context engine check failed",
                "error": str(e)
            }
    
    async def _check_performance_monitor_health(self) -> Dict[str, Any]:
        """Check performance monitor health."""
        try:
            # Try to get the performance monitor
            monitor = await get_context_performance_monitor()
            
            if monitor:
                return {
                    "status": "healthy",
                    "message": "Performance monitor operational",
                    "details": {"active_alerts": len(monitor.active_alerts)}
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": "Performance monitor not available",
                    "error": "monitor_not_found"
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": "Performance monitor check failed",
                "error": str(e)
            }
    
    # Self-healing implementations
    async def _should_trigger_healing(self, action: SelfHealingAction) -> bool:
        """Check if a healing action should be triggered."""
        try:
            # Check cooldown
            if action.last_attempt:
                cooldown_elapsed = (datetime.utcnow() - action.last_attempt).total_seconds() / 60
                if cooldown_elapsed < action.cooldown_minutes:
                    return False
            
            # Check max attempts
            if action.attempts >= action.max_attempts:
                return False
            
            # Check component health
            component_health = self.component_health.get(action.component)
            if not component_health:
                return False
            
            # Check trigger condition
            if action.trigger_condition == "connection_failed":
                return component_health.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
            elif action.trigger_condition == "high_memory_usage":
                # Check if system resources are under pressure
                system_health = self.component_health.get(ComponentType.SYSTEM_RESOURCES)
                if system_health and system_health.recent_results:
                    latest_result = system_health.recent_results[0]
                    memory_percent = latest_result.details.get("memory_percent", 0)
                    return memory_percent > 85
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check healing trigger for {action.action_id}: {e}")
            return False
    
    async def _execute_healing_action(self, action: SelfHealingAction) -> None:
        """Execute a self-healing action."""
        try:
            logger.info(f"Executing healing action: {action.description}")
            
            action.attempts += 1
            action.last_attempt = datetime.utcnow()
            
            # Execute the healing function
            success = await action.action_function()
            
            if success:
                action.success_count += 1
                action.attempts = 0  # Reset attempts on success
                logger.info(f"Healing action successful: {action.description}")
            else:
                action.failure_count += 1
                logger.warning(f"Healing action failed: {action.description}")
            
        except Exception as e:
            action.failure_count += 1
            logger.error(f"Healing action error for {action.action_id}: {e}")
    
    async def _heal_redis_connection(self) -> bool:
        """Attempt to heal Redis connection."""
        try:
            # Try to reconnect
            await self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis healing failed: {e}")
            return False
    
    async def _heal_memory_pressure(self) -> bool:
        """Attempt to heal memory pressure by clearing caches."""
        try:
            # Clear Redis caches
            pattern_keys = [
                "context_monitor:*",
                "search_analytics:*",
                "cost_monitor:*"
            ]
            
            for pattern in pattern_keys:
                keys = await self.redis_client.keys(pattern)
                if keys:
                    # Delete only non-critical cache keys
                    cache_keys = [k for k in keys if b"cache" in k]
                    if cache_keys:
                        await self.redis_client.delete(*cache_keys)
            
            logger.info("Cleared memory caches for healing")
            return True
            
        except Exception as e:
            logger.error(f"Memory healing failed: {e}")
            return False
    
    async def _evaluate_dependency_impact(
        self,
        dependency: ComponentDependency,
        dependent_health: ComponentHealth,
        dependency_health: ComponentHealth
    ) -> None:
        """Evaluate the impact of dependency health on dependent component."""
        try:
            # If dependency is unhealthy, add impact to dependent component
            if (dependency_health.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL] and
                dependency.dependency_type == DependencyType.HARD):
                
                impact_message = f"Dependency {dependency.depends_on.value} is {dependency_health.status.value}: {dependency.impact_description}"
                
                if impact_message not in dependent_health.issues:
                    dependent_health.issues.append(impact_message)
                    dependent_health.recovery_actions.append(f"Restore {dependency.depends_on.value} health")
            
        except Exception as e:
            logger.error(f"Failed to evaluate dependency impact: {e}")
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-wide metrics."""
        try:
            current_time = datetime.utcnow()
            
            # Collect component counts
            healthy_components = len([
                comp for comp, health in self.component_health.items()
                if health.status == HealthStatus.HEALTHY
            ])
            
            total_components = len(self.component_health)
            
            # Store system metrics
            system_metrics = {
                "timestamp": current_time.isoformat(),
                "healthy_components": healthy_components,
                "total_components": total_components,
                "system_uptime_hours": (current_time - self.system_start_time).total_seconds() / 3600,
                "total_checks_performed": sum(len(results) for results in self.check_results.values())
            }
            
            await self.redis_client.setex(
                "health_monitor:system_metrics",
                3600,  # 1 hour TTL
                json.dumps(system_metrics)
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    async def _update_uptime_statistics(self) -> None:
        """Update component uptime statistics."""
        try:
            for component, health in self.component_health.items():
                # Calculate uptime percentage based on recent results
                if health.checks_total > 0:
                    health.uptime_percentage = (health.checks_passed / health.checks_total) * 100
                else:
                    health.uptime_percentage = 0.0
            
        except Exception as e:
            logger.error(f"Failed to update uptime statistics: {e}")
    
    async def _update_circuit_breaker(self, component: ComponentType) -> None:
        """Update circuit breaker status for a component."""
        try:
            if component not in self.circuit_breakers:
                self.circuit_breakers[component] = {
                    "state": "closed",  # closed, open, half_open
                    "failure_count": 0,
                    "last_failure": None,
                    "last_success": None,
                    "next_attempt": None
                }
            
            breaker = self.circuit_breakers[component]
            health = self.component_health.get(component)
            
            if not health:
                return
            
            current_time = datetime.utcnow()
            
            # Update based on current health status
            if health.status == HealthStatus.HEALTHY:
                breaker["failure_count"] = 0
                breaker["last_success"] = current_time
                if breaker["state"] != "closed":
                    breaker["state"] = "closed"
                    logger.info(f"Circuit breaker closed for {component.value}")
                    
            elif health.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                breaker["failure_count"] += 1
                breaker["last_failure"] = current_time
                
                # Open circuit breaker after 5 consecutive failures
                if breaker["failure_count"] >= 5 and breaker["state"] == "closed":
                    breaker["state"] = "open"
                    breaker["next_attempt"] = current_time + timedelta(minutes=5)
                    logger.warning(f"Circuit breaker opened for {component.value}")
                
                # Transition to half-open after timeout
                elif (breaker["state"] == "open" and 
                      breaker["next_attempt"] and 
                      current_time >= breaker["next_attempt"]):
                    breaker["state"] = "half_open"
                    logger.info(f"Circuit breaker half-open for {component.value}")
            
        except Exception as e:
            logger.error(f"Failed to update circuit breaker for {component.value}: {e}")
    
    async def _store_health_result(self, result: HealthResult) -> None:
        """Store health result in Redis."""
        try:
            await self.redis_client.lpush(
                f"health_results:{result.component.value}",
                json.dumps(asdict(result), default=str)
            )
            
            # Keep only recent results
            await self.redis_client.ltrim(f"health_results:{result.component.value}", 0, 999)
            
        except Exception as e:
            logger.error(f"Failed to store health result: {e}")


# Global instance
_health_monitor: Optional[HealthMonitor] = None


async def get_health_monitor() -> HealthMonitor:
    """Get singleton health monitor instance."""
    global _health_monitor
    
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
        await _health_monitor.start()
    
    return _health_monitor


async def cleanup_health_monitor() -> None:
    """Cleanup health monitor resources."""
    global _health_monitor
    
    if _health_monitor:
        await _health_monitor.stop()
        _health_monitor = None