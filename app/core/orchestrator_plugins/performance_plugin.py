"""
Performance Plugin for Orchestrator

Consolidates functionality from:
- performance_orchestrator.py
- performance_orchestrator_integration.py
- performance_orchestrator_plugin.py
- performance_monitoring.py
- performance_monitoring_dashboard.py
"""

import asyncio
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from . import OrchestratorPlugin, PluginMetadata, PluginType
from ..config import settings
from ..redis import get_redis
from ..database import get_session
from ..logging_service import get_component_logger

logger = get_component_logger("performance_plugin")


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    agent_count: int
    active_tasks: int
    response_time: float
    throughput: float
    error_rate: float


@dataclass
class PerformanceThresholds:
    """Performance monitoring thresholds."""
    max_cpu_usage: float = 80.0
    max_memory_usage: float = 80.0
    max_response_time: float = 1000.0  # milliseconds
    min_throughput: float = 10.0  # requests per second
    max_error_rate: float = 5.0  # percentage


class PerformancePlugin(OrchestratorPlugin):
    """Plugin for performance monitoring and optimization."""
    
    def __init__(self):
        metadata = PluginMetadata(
            name="performance_plugin",
            version="1.0.0",
            plugin_type=PluginType.PERFORMANCE,
            description="Performance monitoring, optimization, and resource management",
            dependencies=["redis", "database"]
        )
        super().__init__(metadata)
        
        self.thresholds = PerformanceThresholds()
        self.metrics_history: List[PerformanceMetrics] = []
        self.alert_cooldown: Dict[str, datetime] = {}
        self._monitoring_task: Optional[asyncio.Task] = None
        
    async def initialize(self, orchestrator_context: Dict[str, Any]) -> bool:
        """Initialize performance monitoring."""
        try:
            self.redis = await get_redis()
            logger.info("Performance plugin initialized successfully")
            
            # Start background monitoring
            self._monitoring_task = asyncio.create_task(self._monitor_performance())
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize performance plugin: {e}")
            return False
            
    async def cleanup(self) -> bool:
        """Cleanup performance monitoring resources."""
        try:
            if self._monitoring_task:
                self._monitoring_task.cancel()
                
            logger.info("Performance plugin cleaned up successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup performance plugin: {e}")
            return False
            
    async def pre_task_execution(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Record task start time for performance tracking."""
        task_context["performance_start_time"] = time.time()
        return task_context
        
    async def post_task_execution(self, task_context: Dict[str, Any], result: Any) -> Any:
        """Record task execution metrics."""
        if "performance_start_time" in task_context:
            execution_time = time.time() - task_context["performance_start_time"]
            await self._record_task_performance(
                task_context.get("task_id"),
                execution_time,
                task_context.get("agent_id")
            )
        return result
        
    async def health_check(self) -> Dict[str, Any]:
        """Return performance plugin health status."""
        current_metrics = await self._collect_current_metrics()
        
        health_status = "healthy"
        issues = []
        
        if current_metrics.cpu_usage > self.thresholds.max_cpu_usage:
            health_status = "warning"
            issues.append(f"High CPU usage: {current_metrics.cpu_usage:.1f}%")
            
        if current_metrics.memory_usage > self.thresholds.max_memory_usage:
            health_status = "warning"
            issues.append(f"High memory usage: {current_metrics.memory_usage:.1f}%")
            
        if current_metrics.response_time > self.thresholds.max_response_time:
            health_status = "critical"
            issues.append(f"High response time: {current_metrics.response_time:.1f}ms")
            
        return {
            "plugin": self.metadata.name,
            "enabled": self.enabled,
            "status": health_status,
            "issues": issues,
            "metrics": {
                "cpu_usage": current_metrics.cpu_usage,
                "memory_usage": current_metrics.memory_usage,
                "response_time": current_metrics.response_time,
                "agent_count": current_metrics.agent_count,
                "active_tasks": current_metrics.active_tasks
            }
        }
        
    async def _monitor_performance(self):
        """Background task for continuous performance monitoring."""
        while True:
            try:
                metrics = await self._collect_current_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                    
                # Store in Redis for dashboard
                await self._store_metrics_in_redis(metrics)
                
                # Check thresholds and trigger alerts
                await self._check_performance_thresholds(metrics)
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(30)
                
    async def _collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Application metrics from Redis
            agent_count = await self._get_agent_count()
            active_tasks = await self._get_active_task_count()
            
            # Performance metrics from recent history
            response_time = await self._calculate_avg_response_time()
            throughput = await self._calculate_throughput()
            error_rate = await self._calculate_error_rate()
            
            return PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                agent_count=agent_count,
                active_tasks=active_tasks,
                response_time=response_time,
                throughput=throughput,
                error_rate=error_rate
            )
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
            return PerformanceMetrics(
                timestamp=datetime.utcnow(),
                cpu_usage=0.0,
                memory_usage=0.0,
                agent_count=0,
                active_tasks=0,
                response_time=0.0,
                throughput=0.0,
                error_rate=0.0
            )
            
    async def _store_metrics_in_redis(self, metrics: PerformanceMetrics):
        """Store performance metrics in Redis for dashboard access."""
        try:
            metrics_data = {
                "timestamp": metrics.timestamp.isoformat(),
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "agent_count": metrics.agent_count,
                "active_tasks": metrics.active_tasks,
                "response_time": metrics.response_time,
                "throughput": metrics.throughput,
                "error_rate": metrics.error_rate
            }
            
            # Store current metrics
            await self.redis.set("performance:current", str(metrics_data))
            
            # Store in time series
            await self.redis.lpush("performance:history", str(metrics_data))
            await self.redis.ltrim("performance:history", 0, 1000)  # Keep last 1000 entries
            
        except Exception as e:
            logger.error(f"Error storing metrics in Redis: {e}")
            
    async def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """Check performance thresholds and trigger alerts."""
        alerts = []
        
        if metrics.cpu_usage > self.thresholds.max_cpu_usage:
            alerts.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
            
        if metrics.memory_usage > self.thresholds.max_memory_usage:
            alerts.append(f"High memory usage: {metrics.memory_usage:.1f}%")
            
        if metrics.response_time > self.thresholds.max_response_time:
            alerts.append(f"High response time: {metrics.response_time:.1f}ms")
            
        if metrics.throughput < self.thresholds.min_throughput:
            alerts.append(f"Low throughput: {metrics.throughput:.1f} req/s")
            
        if metrics.error_rate > self.thresholds.max_error_rate:
            alerts.append(f"High error rate: {metrics.error_rate:.1f}%")
            
        for alert in alerts:
            await self._trigger_alert(alert, metrics)
            
    async def _trigger_alert(self, alert_message: str, metrics: PerformanceMetrics):
        """Trigger performance alert with cooldown."""
        alert_key = alert_message.split(":")[0]  # Use first part as key
        
        # Check cooldown (5 minutes)
        if alert_key in self.alert_cooldown:
            time_since_last = datetime.utcnow() - self.alert_cooldown[alert_key]
            if time_since_last < timedelta(minutes=5):
                return
                
        # Store alert in Redis
        alert_data = {
            "message": alert_message,
            "timestamp": metrics.timestamp.isoformat(),
            "severity": "warning",
            "metrics": {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "response_time": metrics.response_time
            }
        }
        
        await self.redis.lpush("performance:alerts", str(alert_data))
        await self.redis.ltrim("performance:alerts", 0, 100)  # Keep last 100 alerts
        
        self.alert_cooldown[alert_key] = datetime.utcnow()
        logger.warning(f"Performance alert: {alert_message}")
        
    async def _get_agent_count(self) -> int:
        """Get current number of active agents."""
        try:
            agent_keys = await self.redis.keys("agent:*:status")
            return len(agent_keys)
        except Exception:
            return 0
            
    async def _get_active_task_count(self) -> int:
        """Get current number of active tasks."""
        try:
            task_keys = await self.redis.keys("task:*:active")
            return len(task_keys)
        except Exception:
            return 0
            
    async def _calculate_avg_response_time(self) -> float:
        """Calculate average response time from recent metrics."""
        if len(self.metrics_history) < 2:
            return 0.0
            
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        response_times = [m.response_time for m in recent_metrics if m.response_time > 0]
        
        return sum(response_times) / len(response_times) if response_times else 0.0
        
    async def _calculate_throughput(self) -> float:
        """Calculate throughput from recent task completions."""
        try:
            # Get task completion count from last minute
            completed_tasks = await self.redis.get("performance:tasks_completed_last_minute")
            return float(completed_tasks) if completed_tasks else 0.0
        except Exception:
            return 0.0
            
    async def _calculate_error_rate(self) -> float:
        """Calculate error rate from recent tasks."""
        try:
            # Get error count from last minute
            error_count = await self.redis.get("performance:errors_last_minute")
            total_count = await self.redis.get("performance:tasks_total_last_minute")
            
            if not error_count or not total_count:
                return 0.0
                
            error_count = float(error_count)
            total_count = float(total_count)
            
            return (error_count / total_count) * 100.0 if total_count > 0 else 0.0
        except Exception:
            return 0.0
            
    async def _record_task_performance(self, task_id: str, execution_time: float, agent_id: str):
        """Record task performance metrics."""
        try:
            # Update response time metrics
            await self.redis.lpush("performance:response_times", str(execution_time * 1000))  # Convert to ms
            await self.redis.ltrim("performance:response_times", 0, 100)
            
            # Update task completion counter
            await self.redis.incr("performance:tasks_completed_last_minute")
            await self.redis.expire("performance:tasks_completed_last_minute", 60)
            
            # Update total task counter
            await self.redis.incr("performance:tasks_total_last_minute")
            await self.redis.expire("performance:tasks_total_last_minute", 60)
            
        except Exception as e:
            logger.error(f"Error recording task performance: {e}")
            
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        current_metrics = await self._collect_current_metrics()
        
        # Calculate trends from history
        cpu_trend = self._calculate_trend("cpu_usage")
        memory_trend = self._calculate_trend("memory_usage")
        response_time_trend = self._calculate_trend("response_time")
        
        return {
            "current": {
                "cpu_usage": current_metrics.cpu_usage,
                "memory_usage": current_metrics.memory_usage,
                "agent_count": current_metrics.agent_count,
                "active_tasks": current_metrics.active_tasks,
                "response_time": current_metrics.response_time,
                "throughput": current_metrics.throughput,
                "error_rate": current_metrics.error_rate
            },
            "trends": {
                "cpu_usage": cpu_trend,
                "memory_usage": memory_trend,
                "response_time": response_time_trend
            },
            "thresholds": {
                "max_cpu_usage": self.thresholds.max_cpu_usage,
                "max_memory_usage": self.thresholds.max_memory_usage,
                "max_response_time": self.thresholds.max_response_time,
                "min_throughput": self.thresholds.min_throughput,
                "max_error_rate": self.thresholds.max_error_rate
            }
        }
        
    def _calculate_trend(self, metric_name: str) -> str:
        """Calculate trend for a specific metric."""
        if len(self.metrics_history) < 5:
            return "stable"
            
        recent_values = [getattr(m, metric_name) for m in self.metrics_history[-5:]]
        
        # Simple trend calculation
        first_half = sum(recent_values[:len(recent_values)//2]) / (len(recent_values)//2)
        second_half = sum(recent_values[len(recent_values)//2:]) / (len(recent_values) - len(recent_values)//2)
        
        diff_percent = ((second_half - first_half) / first_half) * 100 if first_half > 0 else 0
        
        if diff_percent > 10:
            return "increasing"
        elif diff_percent < -10:
            return "decreasing"
        else:
            return "stable"