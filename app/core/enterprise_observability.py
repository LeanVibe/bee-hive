"""
Enterprise Observability Infrastructure for LeanVibe Agent Hive 2.0

Provides comprehensive monitoring, metrics collection, and ROI tracking
for enterprise autonomous development operations. Integrates with Prometheus
and Grafana for real-time dashboards and business value demonstration.
"""

import asyncio
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict, deque
import uuid
import json

import structlog
from prometheus_client import Counter, Histogram, Gauge, Info, Summary, start_http_server

logger = structlog.get_logger()


class MetricCategory(str, Enum):
    """Categories of enterprise metrics."""
    AUTONOMOUS_DEVELOPMENT = "autonomous_development"
    CLI_AGENT_PERFORMANCE = "cli_agent_performance"
    SECURITY_OPERATIONS = "security_operations"
    SYSTEM_PERFORMANCE = "system_performance"
    BUSINESS_VALUE = "business_value"
    USER_EXPERIENCE = "user_experience"


class MetricType(str, Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class DevelopmentMetrics:
    """Metrics for autonomous development operations."""
    task_id: str
    agent_type: str
    generation_time_seconds: float
    execution_time_seconds: float
    total_time_seconds: float
    code_length: int
    success: bool
    security_level: str
    quality_score: float
    created_at: datetime

    def to_prometheus_labels(self) -> Dict[str, str]:
        """Convert to Prometheus label format."""
        return {
            'agent_type': self.agent_type,
            'success': str(self.success).lower(),
            'security_level': self.security_level
        }


@dataclass
class ROIMetrics:
    """Business value and ROI tracking metrics."""
    development_velocity_improvement: float  # Multiplier (e.g., 5.0 = 5x faster)
    cost_savings_per_hour: float  # USD saved per hour
    developer_hours_saved: float  # Hours saved per autonomous task
    quality_improvement_score: float  # 0.0 to 1.0
    error_reduction_percentage: float  # Percentage reduction in errors
    time_to_market_improvement: float  # Days saved in development cycles


class EnterpriseObservability:
    """
    Enterprise observability and metrics collection system.
    
    Provides comprehensive monitoring for:
    - Autonomous development performance
    - CLI agent effectiveness
    - Security operations
    - Business value and ROI
    - System health and performance
    """

    def __init__(self, metrics_port: int = 8001):
        self.metrics_port = metrics_port
        self.logger = structlog.get_logger().bind(component="enterprise_observability")
        
        # Metrics storage
        self.development_history: deque = deque(maxlen=10000)
        self.roi_calculator = ROICalculator()
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        self.alert_manager = AlertManager()

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics collectors."""
        
        # Autonomous Development Metrics
        self.autonomous_dev_duration = Histogram(
            'autonomous_development_duration_seconds',
            'Time spent on autonomous development tasks',
            ['agent_type', 'task_type', 'success'],
            buckets=[1, 5, 10, 30, 60, 120, 300]
        )
        
        self.autonomous_dev_tasks_total = Counter(
            'autonomous_development_tasks_total',
            'Total number of autonomous development tasks',
            ['agent_type', 'success', 'security_level']
        )
        
        self.code_generation_time = Histogram(
            'code_generation_time_seconds',
            'Time spent generating code',
            ['agent_type'],
            buckets=[1, 5, 10, 30, 60, 120]
        )
        
        self.code_execution_time = Histogram(
            'code_execution_time_seconds',
            'Time spent executing code in sandbox',
            ['language', 'success'],
            buckets=[0.1, 0.5, 1, 5, 10, 30]
        )
        
        # CLI Agent Performance Metrics
        self.cli_agent_availability = Gauge(
            'cli_agent_availability',
            'Availability status of CLI agents',
            ['agent_type', 'version']
        )
        
        self.cli_agent_response_time = Histogram(
            'cli_agent_response_time_seconds',
            'Response time of CLI agents',
            ['agent_type'],
            buckets=[1, 5, 10, 30, 60, 120]
        )
        
        self.cli_agent_success_rate = Gauge(
            'cli_agent_success_rate',
            'Success rate of CLI agents (0-1)',
            ['agent_type']
        )
        
        # Security Metrics
        self.security_violations_total = Counter(
            'security_violations_total',
            'Total security violations detected',
            ['violation_type', 'severity']
        )
        
        self.sandbox_executions_total = Counter(
            'sandbox_executions_total',
            'Total sandbox executions',
            ['language', 'success', 'security_level']
        )
        
        # System Performance Metrics
        self.active_tasks_gauge = Gauge(
            'active_autonomous_tasks',
            'Number of currently active autonomous development tasks'
        )
        
        self.system_health_score = Gauge(
            'system_health_score',
            'Overall system health score (0-1)',
            ['component']
        )
        
        # Business Value Metrics
        self.development_velocity_multiplier = Gauge(
            'development_velocity_multiplier',
            'Development velocity improvement multiplier',
            ['measurement_period']
        )
        
        self.cost_savings_hourly = Gauge(
            'cost_savings_usd_per_hour',
            'Cost savings in USD per hour'
        )
        
        self.developer_hours_saved = Counter(
            'developer_hours_saved_total',
            'Total developer hours saved through automation'
        )
        
        # Quality Metrics
        self.code_quality_score = Histogram(
            'code_quality_score',
            'Quality score of generated code (0-1)',
            ['agent_type'],
            buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
        )
        
        self.test_success_rate = Gauge(
            'test_success_rate',
            'Success rate of generated tests (0-1)',
            ['agent_type']
        )

    async def start_metrics_server(self):
        """Start Prometheus metrics HTTP server."""
        try:
            start_http_server(self.metrics_port)
            self.logger.info(
                "Prometheus metrics server started",
                port=self.metrics_port,
                endpoint=f"http://localhost:{self.metrics_port}/metrics"
            )
        except Exception as e:
            self.logger.error("Failed to start metrics server", error=str(e))
            raise

    async def record_autonomous_development(self, metrics: DevelopmentMetrics):
        """Record metrics for an autonomous development task."""
        labels = metrics.to_prometheus_labels()
        
        # Record duration metrics
        self.autonomous_dev_duration.labels(
            agent_type=metrics.agent_type,
            task_type="full_cycle",
            success=str(metrics.success).lower()
        ).observe(metrics.total_time_seconds)
        
        self.code_generation_time.labels(
            agent_type=metrics.agent_type
        ).observe(metrics.generation_time_seconds)
        
        # Record counters
        self.autonomous_dev_tasks_total.labels(**labels).inc()
        
        # Record quality metrics
        self.code_quality_score.labels(
            agent_type=metrics.agent_type
        ).observe(metrics.quality_score)
        
        # Store for ROI calculation
        self.development_history.append(metrics)
        
        # Update ROI metrics
        await self._update_roi_metrics()
        
        self.logger.info(
            "Autonomous development metrics recorded",
            task_id=metrics.task_id,
            agent_type=metrics.agent_type,
            total_time=metrics.total_time_seconds,
            success=metrics.success
        )

    async def record_cli_agent_performance(
        self, 
        agent_type: str, 
        response_time: float, 
        success: bool,
        version: str = "unknown"
    ):
        """Record CLI agent performance metrics."""
        
        # Update availability
        self.cli_agent_availability.labels(
            agent_type=agent_type,
            version=version
        ).set(1.0 if success else 0.0)
        
        # Record response time
        self.cli_agent_response_time.labels(
            agent_type=agent_type
        ).observe(response_time)
        
        # Update success rate (would need historical tracking for accuracy)
        # For now, using a simple metric
        current_rate = 1.0 if success else 0.0
        self.cli_agent_success_rate.labels(
            agent_type=agent_type
        ).set(current_rate)

    async def record_security_event(
        self, 
        violation_type: str, 
        severity: str = "medium"
    ):
        """Record security violation or event."""
        self.security_violations_total.labels(
            violation_type=violation_type,
            severity=severity
        ).inc()

    async def record_sandbox_execution(
        self, 
        language: str, 
        success: bool, 
        security_level: str,
        execution_time: float
    ):
        """Record sandbox execution metrics."""
        self.sandbox_executions_total.labels(
            language=language,
            success=str(success).lower(),
            security_level=security_level
        ).inc()
        
        self.code_execution_time.labels(
            language=language,
            success=str(success).lower()
        ).observe(execution_time)

    async def update_system_health(self, component: str, health_score: float):
        """Update system health metrics."""
        self.system_health_score.labels(component=component).set(health_score)

    async def update_active_tasks(self, count: int):
        """Update active tasks gauge."""
        self.active_tasks_gauge.set(count)

    async def _update_roi_metrics(self):
        """Calculate and update ROI metrics based on historical data."""
        if len(self.development_history) < 10:
            return  # Need sufficient data for meaningful ROI calculation
        
        roi_metrics = self.roi_calculator.calculate_roi(list(self.development_history))
        
        # Update Prometheus metrics
        self.development_velocity_multiplier.labels(
            measurement_period="last_100_tasks"
        ).set(roi_metrics.development_velocity_improvement)
        
        self.cost_savings_hourly.set(roi_metrics.cost_savings_per_hour)
        
        # Increment saved hours counter
        if roi_metrics.developer_hours_saved > 0:
            self.developer_hours_saved.inc(roi_metrics.developer_hours_saved)

    async def get_enterprise_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive data for enterprise dashboards."""
        recent_tasks = list(self.development_history)[-100:]  # Last 100 tasks
        
        if not recent_tasks:
            return {"status": "insufficient_data", "message": "Collecting metrics..."}
        
        # Calculate summary statistics
        total_tasks = len(recent_tasks)
        successful_tasks = len([t for t in recent_tasks if t.success])
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0
        
        avg_generation_time = sum(t.generation_time_seconds for t in recent_tasks) / total_tasks
        avg_execution_time = sum(t.execution_time_seconds for t in recent_tasks) / total_tasks
        avg_total_time = sum(t.total_time_seconds for t in recent_tasks) / total_tasks
        
        # Agent performance breakdown
        agent_stats = defaultdict(lambda: {"count": 0, "success": 0, "avg_time": 0})
        for task in recent_tasks:
            agent_stats[task.agent_type]["count"] += 1
            if task.success:
                agent_stats[task.agent_type]["success"] += 1
            agent_stats[task.agent_type]["avg_time"] += task.total_time_seconds
        
        # Calculate averages
        for agent, stats in agent_stats.items():
            if stats["count"] > 0:
                stats["success_rate"] = stats["success"] / stats["count"]
                stats["avg_time"] = stats["avg_time"] / stats["count"]
        
        # ROI calculation
        roi_metrics = self.roi_calculator.calculate_roi(recent_tasks)
        
        return {
            "summary": {
                "total_tasks": total_tasks,
                "success_rate": success_rate,
                "avg_generation_time": avg_generation_time,
                "avg_execution_time": avg_execution_time,
                "avg_total_time": avg_total_time
            },
            "agent_performance": dict(agent_stats),
            "roi_metrics": asdict(roi_metrics),
            "security": {
                "high_security_rate": len([t for t in recent_tasks if t.security_level == "high"]) / total_tasks,
                "avg_quality_score": sum(t.quality_score for t in recent_tasks) / total_tasks
            },
            "timestamp": datetime.utcnow().isoformat()
        }


class ROICalculator:
    """Calculate business value and ROI metrics."""
    
    def __init__(self):
        # Industry benchmarks for manual development
        self.manual_development_time_hours = {
            "simple_function": 0.5,      # 30 minutes for simple function
            "moderate_function": 2.0,     # 2 hours for moderate complexity
            "complex_feature": 8.0        # 8 hours for complex feature
        }
        
        self.developer_hourly_rate = 75.0  # USD per hour (average)
        
    def calculate_roi(self, development_history: List[DevelopmentMetrics]) -> ROIMetrics:
        """Calculate ROI metrics from development history."""
        if not development_history:
            return ROIMetrics(0, 0, 0, 0, 0, 0)
        
        successful_tasks = [t for t in development_history if t.success]
        
        if not successful_tasks:
            return ROIMetrics(0, 0, 0, 0, 0, 0)
        
        # Calculate development velocity improvement
        avg_autonomous_time = sum(t.total_time_seconds for t in successful_tasks) / len(successful_tasks)
        avg_autonomous_hours = avg_autonomous_time / 3600
        
        # Estimate manual development time (assume moderate complexity average)
        estimated_manual_hours = 2.0  # 2 hours average for tasks we're automating
        
        velocity_improvement = estimated_manual_hours / avg_autonomous_hours if avg_autonomous_hours > 0 else 1.0
        
        # Calculate cost savings
        hours_saved_per_task = max(0, estimated_manual_hours - avg_autonomous_hours)
        cost_savings_per_hour = hours_saved_per_task * self.developer_hourly_rate
        
        # Calculate quality improvements
        avg_quality_score = sum(t.quality_score for t in successful_tasks) / len(successful_tasks)
        
        # Estimate error reduction (higher quality = fewer errors)
        error_reduction_percentage = avg_quality_score * 30  # Assume up to 30% reduction
        
        # Time to market (assume 1 day saved per 10 automated tasks)
        time_to_market_days = len(successful_tasks) * 0.1
        
        return ROIMetrics(
            development_velocity_improvement=velocity_improvement,
            cost_savings_per_hour=cost_savings_per_hour,
            developer_hours_saved=hours_saved_per_task,
            quality_improvement_score=avg_quality_score,
            error_reduction_percentage=error_reduction_percentage,
            time_to_market_improvement=time_to_market_days
        )


class PerformanceTracker:
    """Track system performance metrics."""
    
    def __init__(self):
        self.performance_history = deque(maxlen=1000)
        
    def record_performance(self, component: str, metric: str, value: float):
        """Record a performance metric."""
        self.performance_history.append({
            "component": component,
            "metric": metric,
            "value": value,
            "timestamp": datetime.utcnow()
        })


class AlertManager:
    """Manage enterprise alerts and notifications."""
    
    def __init__(self):
        self.alert_thresholds = {
            "success_rate_minimum": 0.8,      # 80% success rate
            "response_time_maximum": 60.0,    # 60 seconds max
            "security_violations_max": 5,     # 5 violations per hour
            "system_health_minimum": 0.9      # 90% system health
        }
        
    async def check_thresholds(self, metrics_data: Dict[str, Any]):
        """Check if any metrics exceed alert thresholds."""
        alerts = []
        
        summary = metrics_data.get("summary", {})
        
        # Check success rate
        if summary.get("success_rate", 1.0) < self.alert_thresholds["success_rate_minimum"]:
            alerts.append({
                "type": "success_rate_low",
                "severity": "warning",
                "message": f"Success rate {summary['success_rate']:.1%} below threshold",
                "threshold": self.alert_thresholds["success_rate_minimum"]
            })
        
        # Check response time
        if summary.get("avg_total_time", 0) > self.alert_thresholds["response_time_maximum"]:
            alerts.append({
                "type": "response_time_high",
                "severity": "warning",
                "message": f"Average response time {summary['avg_total_time']:.1f}s above threshold",
                "threshold": self.alert_thresholds["response_time_maximum"]
            })
        
        return alerts


# Factory function for easy instantiation
async def create_enterprise_observability(metrics_port: int = 8001) -> EnterpriseObservability:
    """Create and initialize enterprise observability system."""
    observability = EnterpriseObservability(metrics_port)
    await observability.start_metrics_server()
    return observability


# Example usage and testing
if __name__ == "__main__":
    async def test_observability():
        """Test enterprise observability system."""
        observability = await create_enterprise_observability()
        
        # Simulate some development metrics
        test_metrics = DevelopmentMetrics(
            task_id="test_001",
            agent_type="claude_code",
            generation_time_seconds=15.5,
            execution_time_seconds=0.3,
            total_time_seconds=15.8,
            code_length=1250,
            success=True,
            security_level="high",
            quality_score=0.85,
            created_at=datetime.utcnow()
        )
        
        await observability.record_autonomous_development(test_metrics)
        
        # Get dashboard data
        dashboard_data = await observability.get_enterprise_dashboard_data()
        
        print("Enterprise Observability Test:")
        print(f"Dashboard Data: {json.dumps(dashboard_data, indent=2)}")
        print(f"Metrics server running on http://localhost:8001/metrics")

    # Run test if executed directly
    asyncio.run(test_observability())