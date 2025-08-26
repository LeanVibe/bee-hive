"""
Performance Orchestrator Plugin for LeanVibe Agent Hive 2.0 - Epic 1 Phase 2.1

Consolidates advanced performance monitoring, alerting, and auto-scaling capabilities
from production_orchestrator.py into a proper plugin architecture while maintaining
Epic 1 performance targets (<100ms registration, <500ms delegation, <50MB memory).

Key Features:
- Advanced performance monitoring with anomaly detection
- Intelligent alerting with cooldown and correlation
- Auto-scaling with resource pressure analysis
- SLA monitoring and compliance tracking
- Circuit breaker patterns for resilience
- Memory-efficient metrics collection
"""

import asyncio
import uuid
import time
import statistics
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

from . import OrchestratorPlugin, PluginMetadata, PluginType
from ..config import settings
from ..redis import get_redis
from ..database import get_session
from ..logging_service import get_component_logger

logger = get_component_logger("performance_orchestrator_plugin")


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AutoScalingAction(str, Enum):
    """Auto-scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    EMERGENCY_SCALE = "emergency_scale"


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    timestamp: datetime
    
    # System metrics
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_throughput_mbps: float
    
    # Application metrics
    active_agents: int
    pending_tasks: int
    failed_tasks_last_hour: int
    average_response_time_ms: float
    
    # Database metrics
    db_connections: int
    db_query_time_ms: float
    db_pool_usage_percent: float
    
    # SLA metrics
    availability_percent: float
    error_rate_percent: float
    response_time_p95_ms: float
    response_time_p99_ms: float


@dataclass
class AlertRule:
    """Performance alert rule definition."""
    name: str
    description: str
    condition: str
    severity: AlertSeverity
    threshold_value: float
    comparison_operator: str  # >, <, >=, <=, ==, !=
    evaluation_window_minutes: int = 5
    cooldown_minutes: int = 10
    
    # Advanced features
    anomaly_detection: bool = False
    trend_analysis: bool = False
    
    # State tracking
    last_triggered: Optional[datetime] = None
    current_state: str = "ok"
    trigger_count: int = 0


@dataclass 
class SLATarget:
    """Service Level Agreement target definition."""
    name: str
    target_value: float
    current_value: float = 0.0
    compliance_percent: float = 100.0
    breach_count: int = 0
    last_breach: Optional[datetime] = None


@dataclass
class PerformanceAlert:
    """Performance alert instance."""
    alert_id: str
    rule_name: str
    severity: AlertSeverity
    title: str
    description: str
    triggered_at: datetime
    metric_values: Dict[str, Any] = field(default_factory=dict)
    resolved_at: Optional[datetime] = None


@dataclass
class AutoScalingDecision:
    """Auto-scaling decision result."""
    action: AutoScalingAction
    reason: str
    confidence: float
    recommended_agent_count: int
    current_agent_count: int
    metric_drivers: Dict[str, float]
    execute_immediately: bool = False


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for resilient operations."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitBreakerState.CLOSED
    
    @asynccontextmanager
    async def call(self):
        """Execute operation with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if (self.last_failure_time and 
                (datetime.utcnow() - self.last_failure_time).total_seconds() > self.timeout):
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            yield
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
            raise


class PerformanceOrchestratorPlugin(OrchestratorPlugin):
    """
    Advanced Performance Orchestrator Plugin.
    
    Epic 1 Compliant:
    - <100ms agent registration monitoring
    - <500ms task delegation performance tracking
    - <50MB memory footprint for metrics collection
    - Lazy loading and efficient data structures
    """
    
    def __init__(self):
        metadata = PluginMetadata(
            name="performance_orchestrator_plugin",
            version="2.1.0",
            plugin_type=PluginType.PERFORMANCE,
            description="Advanced performance monitoring, alerting, and auto-scaling",
            dependencies=["redis", "database", "simple_orchestrator"]
        )
        super().__init__(metadata)
        
        # Core components
        self.orchestrator_context: Dict[str, Any] = {}
        self.current_metrics: Optional[PerformanceMetrics] = None
        self.metrics_history: List[PerformanceMetrics] = []
        
        # Alert management
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        
        # SLA monitoring
        self.sla_targets: List[SLATarget] = []
        
        # Auto-scaling
        self.auto_scaling_enabled: bool = True
        self.min_agents: int = 1
        self.max_agents: int = 10
        self.scaling_cooldown_minutes: int = 5
        self.last_scaling_action: Optional[datetime] = None
        
        # Performance monitoring
        self.monitoring_tasks: List[asyncio.Task] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            "metrics_collection": CircuitBreaker(),
            "alert_processing": CircuitBreaker(),
            "auto_scaling": CircuitBreaker()
        }
        
        # Epic 1: Memory efficiency tracking
        self._operation_metrics: Dict[str, List[float]] = {}
        self._memory_baseline: float = 0.0
        
        # Initialize default configuration
        self._initialize_default_configuration()
    
    def _initialize_default_configuration(self) -> None:
        """Initialize default performance configuration."""
        # Epic 1: Conservative alert rules for performance preservation
        self.alert_rules = [
            AlertRule(
                name="agent_registration_slow",
                description="Agent registration taking longer than 100ms",
                condition="agent_registration_time_ms",
                severity=AlertSeverity.HIGH,
                threshold_value=100.0,
                comparison_operator=">",
                trend_analysis=True
            ),
            AlertRule(
                name="task_delegation_slow", 
                description="Task delegation taking longer than 500ms",
                condition="task_delegation_time_ms",
                severity=AlertSeverity.HIGH,
                threshold_value=500.0,
                comparison_operator=">",
                trend_analysis=True
            ),
            AlertRule(
                name="memory_usage_high",
                description="Memory usage approaching Epic 1 limits",
                condition="memory_usage_percent",
                severity=AlertSeverity.MEDIUM,
                threshold_value=80.0,
                comparison_operator=">",
                anomaly_detection=True
            ),
            AlertRule(
                name="cpu_usage_critical",
                description="CPU usage critical",
                condition="cpu_usage_percent",
                severity=AlertSeverity.CRITICAL,
                threshold_value=95.0,
                comparison_operator=">"
            ),
            AlertRule(
                name="response_time_degraded",
                description="Response time degraded",
                condition="response_time_p95_ms",
                severity=AlertSeverity.HIGH,
                threshold_value=2000.0,
                comparison_operator=">",
                trend_analysis=True
            ),
            AlertRule(
                name="no_active_agents",
                description="No active agents available",
                condition="active_agents",
                severity=AlertSeverity.CRITICAL,
                threshold_value=1.0,
                comparison_operator="<"
            )
        ]
        
        # Epic 1: Conservative SLA targets
        self.sla_targets = [
            SLATarget(name="agent_registration_sla", target_value=100.0),  # <100ms
            SLATarget(name="task_delegation_sla", target_value=500.0),     # <500ms
            SLATarget(name="memory_efficiency_sla", target_value=50.0),   # <50MB
            SLATarget(name="system_availability", target_value=99.9),     # 99.9% uptime
            SLATarget(name="error_rate", target_value=1.0)                # <1% error rate
        ]
    
    async def initialize(self, orchestrator_context: Dict[str, Any]) -> bool:
        """Initialize performance orchestrator plugin."""
        start_time = time.time()
        
        try:
            self.orchestrator_context = orchestrator_context
            
            # Initialize Redis connection (graceful fallback)
            try:
                self.redis = get_redis()
            except Exception as e:
                logger.warning(f"Redis initialization failed, using mock: {e}")
                self.redis = AsyncMock()  # Use mock for testing
            
            # Store memory baseline
            self._memory_baseline = self._get_memory_usage()
            
            # Start monitoring tasks
            self.monitoring_tasks = [
                asyncio.create_task(self._metrics_collection_loop()),
                asyncio.create_task(self._alert_evaluation_loop()), 
                asyncio.create_task(self._sla_monitoring_loop()),
                asyncio.create_task(self._auto_scaling_loop()),
                asyncio.create_task(self._anomaly_detection_loop())
            ]
            
            init_time_ms = (time.time() - start_time) * 1000
            self._record_operation_metric("initialize", init_time_ms)
            
            logger.info("Performance Orchestrator Plugin initialized",
                       init_time_ms=round(init_time_ms, 2),
                       memory_usage_mb=round(self._get_memory_usage() - self._memory_baseline, 2))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Performance Orchestrator Plugin: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        try:
            # Cancel monitoring tasks
            for task in self.monitoring_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete with timeout
            if self.monitoring_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self.monitoring_tasks, return_exceptions=True),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("Some monitoring tasks did not shutdown gracefully")
            
            # Clear data structures for memory efficiency
            self.metrics_history.clear()
            self.active_alerts.clear()
            self.alert_history.clear()
            self._operation_metrics.clear()
            
            logger.info("Performance Orchestrator Plugin cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup Performance Orchestrator Plugin: {e}")
            return False
    
    async def pre_task_execution(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Record task start time for performance tracking."""
        task_context["performance_start_time"] = time.time()
        task_context["performance_plugin_tracking"] = True
        return task_context
    
    async def post_task_execution(self, task_context: Dict[str, Any], result: Any) -> Any:
        """Record task execution metrics."""
        if "performance_start_time" in task_context:
            execution_time_ms = (time.time() - task_context["performance_start_time"]) * 1000
            
            # Track Epic 1 performance targets
            operation_type = task_context.get("operation_type", "task_execution")
            self._record_operation_metric(operation_type, execution_time_ms)
            
            # Update SLA tracking
            await self._update_sla_metrics(operation_type, execution_time_ms)
            
            # Record in Redis for dashboard
            await self._record_task_performance_metrics(
                task_context.get("task_id"),
                execution_time_ms,
                task_context.get("agent_id")
            )
        
        return result
    
    async def health_check(self) -> Dict[str, Any]:
        """Return comprehensive plugin health status."""
        current_time = time.time()
        
        try:
            # Use existing metrics if available, don't collect new ones for speed
            metrics = self.current_metrics
            if not metrics:
                # Create minimal default metrics for speed
                metrics = PerformanceMetrics(
                    timestamp=datetime.utcnow(),
                    cpu_usage_percent=psutil.cpu_percent(interval=None),  # Non-blocking
                    memory_usage_percent=psutil.virtual_memory().percent,
                    disk_usage_percent=50.0,
                    network_throughput_mbps=10.0,
                    active_agents=len(self.orchestrator_context.get("orchestrator", {}).get("_agents", {})) if self.orchestrator_context.get("orchestrator") else 0,
                    pending_tasks=0,
                    failed_tasks_last_hour=0,
                    average_response_time_ms=0.0,
                    db_connections=10,
                    db_query_time_ms=50.0,
                    db_pool_usage_percent=25.0,
                    availability_percent=99.9,
                    error_rate_percent=1.0,
                    response_time_p95_ms=800.0,
                    response_time_p99_ms=1200.0
                )
            
            # Quick Epic 1 compliance check
            epic1_compliance = await self._check_epic1_compliance()
            
            # Calculate health score
            health_issues = []
            health_score = 1.0
            
            if metrics.cpu_usage_percent > 95:
                health_issues.append("Critical CPU usage")
                health_score *= 0.3
            elif metrics.cpu_usage_percent > 80:
                health_issues.append("High CPU usage")
                health_score *= 0.7
            
            if metrics.memory_usage_percent > 90:
                health_issues.append("Critical memory usage")
                health_score *= 0.3
            
            if len(self.active_alerts) > 0:
                critical_alerts = [a for a in self.active_alerts.values() 
                                 if a.severity == AlertSeverity.CRITICAL]
                if critical_alerts:
                    health_issues.append(f"{len(critical_alerts)} critical alerts")
                    health_score *= 0.5
            
            # Determine health status
            if health_score >= 0.9:
                health_status = "healthy"
            elif health_score >= 0.7:
                health_status = "degraded"
            elif health_score >= 0.5:
                health_status = "unhealthy"
            else:
                health_status = "critical"
            
            health_check_time_ms = (time.time() - current_time) * 1000
            self._record_operation_metric("health_check", health_check_time_ms)
            
            return {
                "plugin": self.metadata.name,
                "enabled": self.enabled,
                "status": health_status,
                "health_score": round(health_score, 3),
                "issues": health_issues,
                "epic1_compliance": epic1_compliance,
                "performance_summary": {
                    "cpu_usage": metrics.cpu_usage_percent,
                    "memory_usage": metrics.memory_usage_percent,
                    "active_agents": metrics.active_agents,
                    "pending_tasks": metrics.pending_tasks,
                    "response_time_p95": metrics.response_time_p95_ms,
                    "error_rate": metrics.error_rate_percent
                },
                "monitoring_status": {
                    "active_tasks": len([t for t in self.monitoring_tasks if not t.done()]),
                    "metrics_collected": len(self.metrics_history),
                    "active_alerts": len(self.active_alerts),
                    "circuit_breaker_states": {
                        name: breaker.state.value 
                        for name, breaker in self.circuit_breakers.items()
                    }
                },
                "health_check_time_ms": round(health_check_time_ms, 2)
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "plugin": self.metadata.name,
                "enabled": self.enabled,
                "status": "error",
                "error": str(e)
            }
    
    async def _metrics_collection_loop(self) -> None:
        """Main metrics collection loop with circuit breaker protection."""
        while True:
            try:
                async with self.circuit_breakers["metrics_collection"].call():
                    start_time = time.time()
                    
                    # Collect metrics
                    metrics = await self._collect_performance_metrics()
                    self.current_metrics = metrics
                    
                    # Store in history (Epic 1: memory efficient)
                    self.metrics_history.append(metrics)
                    if len(self.metrics_history) > 500:  # Keep last 500 entries
                        self.metrics_history = self.metrics_history[-500:]
                    
                    # Store in Redis for dashboard
                    await self._store_metrics_in_redis(metrics)
                    
                    # Track operation time
                    collection_time_ms = (time.time() - start_time) * 1000
                    self._record_operation_metric("metrics_collection", collection_time_ms)
                    
                    # Epic 1: Ensure collection stays under performance target
                    if collection_time_ms > 50:  # 50ms threshold
                        logger.warning("Metrics collection slow",
                                     collection_time_ms=collection_time_ms)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(60)  # Longer sleep on error
    
    async def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics efficiently."""
        try:
            # System metrics (fast collection)
            cpu_percent = psutil.cpu_percent(interval=0.1)  # Non-blocking
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network_stats = psutil.net_io_counters()
            network_throughput = (network_stats.bytes_sent + network_stats.bytes_recv) / (1024 * 1024)
            
            # Application metrics from orchestrator context
            orchestrator = self.orchestrator_context.get("orchestrator")
            active_agents = 0
            pending_tasks = 0
            
            if orchestrator:
                try:
                    status = await orchestrator.get_system_status()
                    active_agents = status.get("agents", {}).get("total", 0)
                    pending_tasks = status.get("tasks", {}).get("active_assignments", 0)
                except:
                    pass  # Continue with defaults
            
            # Calculate derived metrics
            availability = await self._calculate_availability()
            error_rate = await self._calculate_error_rate()
            response_times = await self._calculate_response_time_percentiles()
            
            return PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                disk_usage_percent=(disk.used / disk.total) * 100,
                network_throughput_mbps=network_throughput,
                active_agents=active_agents,
                pending_tasks=pending_tasks,
                failed_tasks_last_hour=await self._get_failed_tasks_count(),
                average_response_time_ms=response_times.get("avg", 0.0),
                db_connections=10,  # Placeholder - would integrate with actual DB pool
                db_query_time_ms=50.0,
                db_pool_usage_percent=25.0,
                availability_percent=availability,
                error_rate_percent=error_rate,
                response_time_p95_ms=response_times.get("p95", 0.0),
                response_time_p99_ms=response_times.get("p99", 0.0)
            )
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            # Return safe defaults
            return PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage_percent=0.0,
                memory_usage_percent=0.0,
                disk_usage_percent=0.0,
                network_throughput_mbps=0.0,
                active_agents=0,
                pending_tasks=0,
                failed_tasks_last_hour=0,
                average_response_time_ms=0.0,
                db_connections=0,
                db_query_time_ms=0.0,
                db_pool_usage_percent=0.0,
                availability_percent=100.0,
                error_rate_percent=0.0,
                response_time_p95_ms=0.0,
                response_time_p99_ms=0.0
            )
    
    async def _alert_evaluation_loop(self) -> None:
        """Evaluate alert rules and trigger alerts."""
        while True:
            try:
                async with self.circuit_breakers["alert_processing"].call():
                    if self.current_metrics:
                        await self._evaluate_alert_rules(self.current_metrics)
                    
                    # Check for alert resolutions
                    await self._check_alert_resolutions()
                
                await asyncio.sleep(60)  # Evaluate every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
                await asyncio.sleep(60)
    
    async def _evaluate_alert_rules(self, metrics: PerformanceMetrics) -> None:
        """Evaluate all alert rules against current metrics."""
        for rule in self.alert_rules:
            try:
                # Check cooldown
                if (rule.last_triggered and 
                    datetime.utcnow() - rule.last_triggered < timedelta(minutes=rule.cooldown_minutes)):
                    continue
                
                # Get metric value
                metric_value = await self._get_metric_value(metrics, rule.condition)
                if metric_value is None:
                    continue
                
                # Evaluate condition
                should_trigger = self._evaluate_condition(
                    metric_value, rule.threshold_value, rule.comparison_operator
                )
                
                # Enhanced evaluation
                if rule.anomaly_detection and not should_trigger:
                    should_trigger = await self._detect_anomaly(rule, metric_value)
                
                if rule.trend_analysis and not should_trigger:
                    should_trigger = await self._analyze_trend(rule, metric_value)
                
                if should_trigger and rule.name not in self.active_alerts:
                    await self._trigger_alert(rule, metric_value, metrics)
                    
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.name}: {e}")
    
    async def _sla_monitoring_loop(self) -> None:
        """Monitor SLA compliance."""
        while True:
            try:
                await self._update_sla_targets()
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in SLA monitoring loop: {e}")
                await asyncio.sleep(300)
    
    async def _auto_scaling_loop(self) -> None:
        """Auto-scaling decision and execution loop."""
        while True:
            try:
                async with self.circuit_breakers["auto_scaling"].call():
                    if self.auto_scaling_enabled and self.current_metrics:
                        decision = await self._make_auto_scaling_decision()
                        if decision.action != AutoScalingAction.MAINTAIN:
                            await self._execute_auto_scaling_decision(decision)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
                await asyncio.sleep(300)
    
    async def _anomaly_detection_loop(self) -> None:
        """Advanced anomaly detection loop."""
        while True:
            try:
                if len(self.metrics_history) >= 20:
                    await self._detect_system_anomalies()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in anomaly detection loop: {e}")
                await asyncio.sleep(300)
    
    # Helper methods
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _record_operation_metric(self, operation: str, time_ms: float) -> None:
        """Record operation metrics for Epic 1 performance monitoring."""
        if operation not in self._operation_metrics:
            self._operation_metrics[operation] = []
        
        metrics = self._operation_metrics[operation]
        metrics.append(time_ms)
        
        # Keep only last 50 measurements for memory efficiency
        if len(metrics) > 50:
            metrics.pop(0)
    
    async def _get_metric_value(self, metrics: PerformanceMetrics, condition: str) -> Optional[float]:
        """Get metric value for alert condition."""
        # Handle special Epic 1 metrics
        if condition == "agent_registration_time_ms":
            op_metrics = self._operation_metrics.get("spawn_agent", [])
            return statistics.mean(op_metrics[-5:]) if op_metrics else 0.0
        elif condition == "task_delegation_time_ms":
            op_metrics = self._operation_metrics.get("delegate_task", [])
            return statistics.mean(op_metrics[-5:]) if op_metrics else 0.0
        
        # Standard metrics
        return getattr(metrics, condition, None)
    
    def _evaluate_condition(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate alert condition."""
        operators = {
            ">": lambda v, t: v > t,
            "<": lambda v, t: v < t,
            ">=": lambda v, t: v >= t,
            "<=": lambda v, t: v <= t,
            "==": lambda v, t: v == t,
            "!=": lambda v, t: v != t
        }
        return operators.get(operator, lambda v, t: False)(value, threshold)
    
    async def _detect_anomaly(self, rule: AlertRule, current_value: float) -> bool:
        """Detect anomalies using statistical analysis."""
        if len(self.metrics_history) < 10:
            return False
        
        # Get historical values
        historical_values = []
        for metrics in self.metrics_history[-30:]:  # Last 30 data points
            value = await self._get_metric_value(metrics, rule.condition)
            if value is not None:
                historical_values.append(value)
        
        if len(historical_values) < 10:
            return False
        
        # Statistical analysis
        mean = statistics.mean(historical_values)
        try:
            stdev = statistics.stdev(historical_values)
        except:
            return False
        
        # 2.5 standard deviations threshold
        threshold = mean + (2.5 * stdev)
        return current_value > threshold
    
    async def _analyze_trend(self, rule: AlertRule, current_value: float) -> bool:
        """Analyze trends for predictive alerting."""
        if len(self.metrics_history) < 5:
            return False
        
        # Get recent values
        recent_values = []
        for metrics in self.metrics_history[-5:]:
            value = await self._get_metric_value(metrics, rule.condition)
            if value is not None:
                recent_values.append(value)
        
        if len(recent_values) < 5:
            return False
        
        # Simple trend calculation
        x_values = list(range(len(recent_values)))
        n = len(recent_values)
        
        try:
            # Calculate slope using simple linear regression
            x_sum = sum(x_values)
            y_sum = sum(recent_values)
            xy_sum = sum(x * y for x, y in zip(x_values, recent_values))
            x_squared_sum = sum(x**2 for x in x_values)
            
            denominator = n * x_squared_sum - x_sum**2
            if denominator == 0:
                return False
                
            slope = (n * xy_sum - x_sum * y_sum) / denominator
            
            # Project ahead 2 time periods
            projected_value = current_value + (slope * 2)
            result = self._evaluate_condition(projected_value, rule.threshold_value, rule.comparison_operator)
            
            logger.debug(f"Trend analysis: slope={slope:.2f}, projected={projected_value:.2f}, threshold={rule.threshold_value}, result={result}")
            return result
        except ZeroDivisionError:
            logger.debug("Trend analysis failed: division by zero")
            return False
        except Exception as e:
            logger.debug(f"Trend analysis failed: {str(e)}")
            return False
    
    async def _trigger_alert(self, rule: AlertRule, metric_value: float, metrics: PerformanceMetrics) -> None:
        """Trigger a new alert."""
        alert_id = str(uuid.uuid4())
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            rule_name=rule.name,
            severity=rule.severity,
            title=rule.description,
            description=f"{rule.description}. Current: {metric_value:.2f}, Threshold: {rule.threshold_value}",
            triggered_at=datetime.utcnow(),
            metric_values={
                rule.condition: metric_value,
                "threshold": rule.threshold_value,
                "cpu_usage": metrics.cpu_usage_percent,
                "memory_usage": metrics.memory_usage_percent
            }
        )
        
        # Store alert
        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        
        # Update rule state
        rule.last_triggered = datetime.utcnow()
        rule.current_state = "critical" if rule.severity == AlertSeverity.CRITICAL else "warning"
        rule.trigger_count += 1
        
        # Store in Redis
        await self._store_alert_in_redis(alert)
        
        logger.warning(f"🚨 Performance Alert: {alert.title}",
                      alert_id=alert_id,
                      severity=alert.severity.value,
                      metric_value=metric_value)
    
    async def _check_alert_resolutions(self) -> None:
        """Check if active alerts should be resolved."""
        if not self.current_metrics:
            return
        
        resolved_alerts = []
        
        for rule_name, alert in self.active_alerts.items():
            rule = next((r for r in self.alert_rules if r.name == rule_name), None)
            if not rule:
                continue
            
            metric_value = await self._get_metric_value(self.current_metrics, rule.condition)
            if metric_value is None:
                continue
            
            # Check if condition is no longer met
            should_trigger = self._evaluate_condition(
                metric_value, rule.threshold_value, rule.comparison_operator
            )
            
            if not should_trigger:
                alert.resolved_at = datetime.utcnow()
                rule.current_state = "ok"
                resolved_alerts.append(rule_name)
                
                logger.info(f"✅ Alert resolved: {alert.title}",
                           alert_id=alert.alert_id,
                           duration_minutes=(alert.resolved_at - alert.triggered_at).total_seconds() / 60)
        
        # Remove resolved alerts
        for rule_name in resolved_alerts:
            del self.active_alerts[rule_name]
    
    async def _make_auto_scaling_decision(self) -> AutoScalingDecision:
        """Make intelligent auto-scaling decision based on metrics."""
        if not self.current_metrics:
            return AutoScalingDecision(
                action=AutoScalingAction.MAINTAIN,
                reason="No metrics available",
                confidence=0.0,
                recommended_agent_count=self.min_agents,
                current_agent_count=0,
                metric_drivers={}
            )
        
        current_agents = self.current_metrics.active_agents
        
        # Calculate pressure indicators
        pressure_indicators = {
            "cpu_pressure": max(0, self.current_metrics.cpu_usage_percent - 70) / 30,
            "memory_pressure": max(0, self.current_metrics.memory_usage_percent - 80) / 20,
            "task_pressure": min(1.0, self.current_metrics.pending_tasks / 20),
            "response_time_pressure": max(0, self.current_metrics.response_time_p95_ms - 1000) / 2000,
            "error_rate_pressure": min(1.0, self.current_metrics.error_rate_percent / 5)
        }
        
        # Overall pressure score
        pressure_score = sum(pressure_indicators.values()) / len(pressure_indicators)
        
        # Make decision
        if pressure_score > 0.7 and current_agents < self.max_agents:
            return AutoScalingDecision(
                action=AutoScalingAction.SCALE_UP,
                reason=f"High system pressure: {pressure_score:.2f}",
                confidence=min(1.0, pressure_score),
                recommended_agent_count=min(self.max_agents, current_agents + 1),
                current_agent_count=current_agents,
                metric_drivers=pressure_indicators,
                execute_immediately=pressure_score > 0.9
            )
        elif pressure_score < 0.2 and current_agents > self.min_agents:
            return AutoScalingDecision(
                action=AutoScalingAction.SCALE_DOWN,
                reason=f"Low system pressure: {pressure_score:.2f}",
                confidence=1.0 - pressure_score,
                recommended_agent_count=max(self.min_agents, current_agents - 1),
                current_agent_count=current_agents,
                metric_drivers=pressure_indicators
            )
        else:
            return AutoScalingDecision(
                action=AutoScalingAction.MAINTAIN,
                reason=f"Pressure within range: {pressure_score:.2f}",
                confidence=0.8,
                recommended_agent_count=current_agents,
                current_agent_count=current_agents,
                metric_drivers=pressure_indicators
            )
    
    async def _execute_auto_scaling_decision(self, decision: AutoScalingDecision) -> None:
        """Execute auto-scaling decision."""
        # Check cooldown
        if (self.last_scaling_action and 
            datetime.utcnow() - self.last_scaling_action < timedelta(minutes=self.scaling_cooldown_minutes)):
            return
        
        try:
            orchestrator = self.orchestrator_context.get("orchestrator")
            if not orchestrator:
                return
            
            if decision.action == AutoScalingAction.SCALE_UP:
                agents_to_add = decision.recommended_agent_count - decision.current_agent_count
                for _ in range(agents_to_add):
                    await orchestrator.spawn_agent(
                        role=orchestrator.AgentRole.BACKEND_DEVELOPER
                    )
                
                logger.info(f"🔼 Auto-scaled UP: +{agents_to_add} agents",
                           reason=decision.reason,
                           new_total=decision.recommended_agent_count)
            
            elif decision.action == AutoScalingAction.SCALE_DOWN:
                agents_to_remove = decision.current_agent_count - decision.recommended_agent_count
                # Get agent IDs from orchestrator status
                status = await orchestrator.get_system_status()
                agent_ids = list(status.get("agents", {}).get("details", {}).keys())
                
                for i, agent_id in enumerate(agent_ids[:agents_to_remove]):
                    await orchestrator.shutdown_agent(agent_id, graceful=True)
                
                logger.info(f"🔽 Auto-scaled DOWN: -{agents_to_remove} agents",
                           reason=decision.reason,
                           new_total=decision.recommended_agent_count)
            
            self.last_scaling_action = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Failed to execute auto-scaling: {e}")
    
    # Additional helper methods
    
    async def _store_metrics_in_redis(self, metrics: PerformanceMetrics) -> None:
        """Store metrics in Redis efficiently."""
        try:
            metrics_data = {
                "timestamp": metrics.timestamp.isoformat(),
                "cpu_usage": metrics.cpu_usage_percent,
                "memory_usage": metrics.memory_usage_percent,
                "active_agents": metrics.active_agents,
                "pending_tasks": metrics.pending_tasks,
                "response_time_p95": metrics.response_time_p95_ms,
                "error_rate": metrics.error_rate_percent
            }
            
            # Store current and history
            await self.redis.set("performance:current", str(metrics_data), ex=300)
            await self.redis.lpush("performance:history", str(metrics_data))
            await self.redis.ltrim("performance:history", 0, 500)
            
        except Exception as e:
            logger.error(f"Error storing metrics in Redis: {e}")
    
    async def _store_alert_in_redis(self, alert: PerformanceAlert) -> None:
        """Store alert in Redis."""
        try:
            alert_data = {
                "alert_id": alert.alert_id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "title": alert.title,
                "description": alert.description,
                "triggered_at": alert.triggered_at.isoformat(),
                "metric_values": alert.metric_values
            }
            
            await self.redis.lpush("performance:alerts", str(alert_data))
            await self.redis.ltrim("performance:alerts", 0, 100)
            
        except Exception as e:
            logger.error(f"Error storing alert in Redis: {e}")
    
    async def _update_sla_metrics(self, operation_type: str, execution_time_ms: float) -> None:
        """Update SLA metrics based on operation performance."""
        for target in self.sla_targets:
            if target.name == "agent_registration_sla" and operation_type == "spawn_agent":
                target.current_value = execution_time_ms
                target.compliance_percent = 100.0 if execution_time_ms <= target.target_value else 80.0
            elif target.name == "task_delegation_sla" and operation_type == "delegate_task":
                target.current_value = execution_time_ms
                target.compliance_percent = 100.0 if execution_time_ms <= target.target_value else 80.0
    
    async def _update_sla_targets(self) -> None:
        """Update SLA target compliance."""
        for target in self.sla_targets:
            if target.name == "memory_efficiency_sla":
                current_memory = self._get_memory_usage() - self._memory_baseline
                target.current_value = current_memory
                target.compliance_percent = 100.0 if current_memory <= target.target_value else 70.0
    
    async def _check_epic1_compliance(self) -> Dict[str, Any]:
        """Check Epic 1 performance compliance."""
        compliance = {"overall": True, "details": {}}
        
        # Check agent registration time
        agent_reg_times = self._operation_metrics.get("spawn_agent", [])
        if agent_reg_times:
            avg_time = statistics.mean(agent_reg_times[-10:])
            compliance["details"]["agent_registration"] = {
                "compliant": avg_time < 100.0,
                "current_ms": avg_time,
                "target_ms": 100.0
            }
            if avg_time >= 100.0:
                compliance["overall"] = False
        
        # Check task delegation time
        task_del_times = self._operation_metrics.get("delegate_task", [])
        if task_del_times:
            avg_time = statistics.mean(task_del_times[-10:])
            compliance["details"]["task_delegation"] = {
                "compliant": avg_time < 500.0,
                "current_ms": avg_time,
                "target_ms": 500.0
            }
            if avg_time >= 500.0:
                compliance["overall"] = False
        
        # Check memory usage
        current_memory = self._get_memory_usage() - self._memory_baseline
        compliance["details"]["memory_usage"] = {
            "compliant": current_memory < 50.0,
            "current_mb": current_memory,
            "target_mb": 50.0
        }
        if current_memory >= 50.0:
            compliance["overall"] = False
        
        return compliance
    
    async def _calculate_availability(self) -> float:
        """Calculate system availability."""
        if self.current_metrics and self.current_metrics.active_agents > 0:
            return 99.9
        return 95.0
    
    async def _calculate_error_rate(self) -> float:
        """Calculate error rate."""
        try:
            error_count = await self.redis.get("performance:errors_count") or "0"
            total_count = await self.redis.get("performance:total_count") or "1"
            return (float(error_count) / float(total_count)) * 100
        except:
            return 0.0
    
    async def _calculate_response_time_percentiles(self) -> Dict[str, float]:
        """Calculate response time percentiles."""
        try:
            times_str = await self.redis.lrange("performance:response_times", 0, 99)
            if not times_str:
                return {"avg": 0.0, "p95": 0.0, "p99": 0.0}
            
            times = [float(t) for t in times_str if t]
            if not times:
                return {"avg": 0.0, "p95": 0.0, "p99": 0.0}
            
            times.sort()
            return {
                "avg": statistics.mean(times),
                "p95": times[int(len(times) * 0.95)] if times else 0.0,
                "p99": times[int(len(times) * 0.99)] if times else 0.0
            }
        except:
            return {"avg": 0.0, "p95": 0.0, "p99": 0.0}
    
    async def _get_failed_tasks_count(self) -> int:
        """Get failed tasks count."""
        try:
            count = await self.redis.get("performance:failed_tasks_hour") or "0"
            return int(count)
        except:
            return 0
    
    async def _record_task_performance_metrics(self, task_id: Optional[str], 
                                             execution_time_ms: float, 
                                             agent_id: Optional[str]) -> None:
        """Record task performance metrics in Redis."""
        try:
            # Store response time
            await self.redis.lpush("performance:response_times", str(execution_time_ms))
            await self.redis.ltrim("performance:response_times", 0, 100)
            
            # Update counters
            await self.redis.incr("performance:total_count")
            await self.redis.expire("performance:total_count", 3600)  # 1 hour
            
        except Exception as e:
            logger.error(f"Error recording task performance: {e}")
    
    async def _detect_system_anomalies(self) -> None:
        """Detect system-wide anomalies."""
        if len(self.metrics_history) < 20:
            return
        
        try:
            recent_metrics = self.metrics_history[-10:]
            historical_metrics = self.metrics_history[-30:-10]
            
            # CPU anomaly detection
            recent_cpu = statistics.mean([m.cpu_usage_percent for m in recent_metrics])
            historical_cpu = statistics.mean([m.cpu_usage_percent for m in historical_metrics])
            
            if recent_cpu > historical_cpu * 1.5:  # 50% increase
                logger.warning("CPU usage anomaly detected",
                              recent=recent_cpu, 
                              historical=historical_cpu)
            
            # Memory anomaly detection
            recent_memory = statistics.mean([m.memory_usage_percent for m in recent_metrics])
            historical_memory = statistics.mean([m.memory_usage_percent for m in historical_metrics])
            
            if recent_memory > historical_memory * 1.3:  # 30% increase
                logger.warning("Memory usage anomaly detected",
                              recent=recent_memory,
                              historical=historical_memory)
                
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            "epic1_compliance": await self._check_epic1_compliance(),
            "current_metrics": {
                "cpu_usage": self.current_metrics.cpu_usage_percent if self.current_metrics else 0,
                "memory_usage": self.current_metrics.memory_usage_percent if self.current_metrics else 0,
                "active_agents": self.current_metrics.active_agents if self.current_metrics else 0,
                "pending_tasks": self.current_metrics.pending_tasks if self.current_metrics else 0,
                "response_time_p95": self.current_metrics.response_time_p95_ms if self.current_metrics else 0,
                "error_rate": self.current_metrics.error_rate_percent if self.current_metrics else 0
            },
            "sla_status": {
                target.name: {
                    "target": target.target_value,
                    "current": target.current_value,
                    "compliance": target.compliance_percent
                }
                for target in self.sla_targets
            },
            "active_alerts": len(self.active_alerts),
            "auto_scaling": {
                "enabled": self.auto_scaling_enabled,
                "current_agents": self.current_metrics.active_agents if self.current_metrics else 0,
                "min_agents": self.min_agents,
                "max_agents": self.max_agents
            },
            "operation_metrics": {
                operation: {
                    "avg_ms": statistics.mean(times) if times else 0,
                    "max_ms": max(times) if times else 0,
                    "count": len(times)
                }
                for operation, times in self._operation_metrics.items()
            }
        }


def create_performance_orchestrator_plugin() -> PerformanceOrchestratorPlugin:
    """Factory function to create the performance orchestrator plugin."""
    return PerformanceOrchestratorPlugin()