#!/usr/bin/env python3
"""
Performance Optimization Framework for LeanVibe Agent Hive 2.0

Epic 1: Performance Excellence & Optimization Foundation
Building on the successful 90%+ consolidation achievement to implement
enterprise-grade performance optimization with predictive monitoring.

This framework provides:
- Real-time performance monitoring and alerting
- Predictive optimization using ML-based analysis
- Automated performance regression detection
- Memory and resource optimization utilities
- Concurrent scaling optimization
"""

import asyncio
import time
import psutil
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import threading
import weakref

import structlog
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger(__name__)


class PerformanceMetricType(str, Enum):
    """Types of performance metrics tracked."""
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage"
    CPU_UTILIZATION = "cpu_utilization"
    CONCURRENT_AGENTS = "concurrent_agents"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"


class OptimizationPriority(str, Enum):
    """Priority levels for optimization actions."""
    CRITICAL = "critical"      # Immediate action required
    HIGH = "high"             # Action required within 1 hour
    MEDIUM = "medium"         # Action required within 24 hours
    LOW = "low"              # Optimization opportunity
    MONITOR = "monitor"       # Continue monitoring


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    timestamp: datetime
    metric_type: PerformanceMetricType
    value: float
    context: Dict[str, Any] = field(default_factory=dict)
    component: Optional[str] = None
    agent_id: Optional[str] = None


@dataclass
class PerformanceTarget:
    """Performance target definition."""
    metric_type: PerformanceMetricType
    target_value: float
    warning_threshold: float
    critical_threshold: float
    unit: str
    description: str


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""
    priority: OptimizationPriority
    component: str
    issue_description: str
    recommended_action: str
    expected_improvement: str
    implementation_complexity: str
    estimated_effort_hours: int
    impact_metrics: List[PerformanceMetricType]


class PerformanceOptimizationFramework:
    """
    Comprehensive performance optimization framework for Epic 1.
    
    Provides real-time monitoring, predictive optimization, and
    automated performance improvement recommendations.
    """
    
    def __init__(self):
        self.metrics_history: Dict[PerformanceMetricType, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.performance_targets = self._initialize_performance_targets()
        self.optimization_history: List[OptimizationRecommendation] = []
        
        # ML models for predictive optimization
        self.trend_models: Dict[PerformanceMetricType, LinearRegression] = {}
        self.scalers: Dict[PerformanceMetricType, StandardScaler] = {}
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_interval = 5.0  # seconds
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # Component registry for targeted optimization
        self.registered_components: Dict[str, weakref.ref] = {}
        
        logger.info("Performance Optimization Framework initialized for Epic 1")
    
    def _initialize_performance_targets(self) -> Dict[PerformanceMetricType, PerformanceTarget]:
        """Initialize Epic 1 performance targets."""
        return {
            PerformanceMetricType.RESPONSE_TIME: PerformanceTarget(
                metric_type=PerformanceMetricType.RESPONSE_TIME,
                target_value=50.0,      # Epic 1 target: <50ms
                warning_threshold=75.0,
                critical_threshold=100.0,
                unit="milliseconds",
                description="API response time 95th percentile"
            ),
            PerformanceMetricType.MEMORY_USAGE: PerformanceTarget(
                metric_type=PerformanceMetricType.MEMORY_USAGE,
                target_value=80.0,      # Epic 1 target: <80MB
                warning_threshold=90.0,
                critical_threshold=100.0,
                unit="megabytes",
                description="System memory usage"
            ),
            PerformanceMetricType.CONCURRENT_AGENTS: PerformanceTarget(
                metric_type=PerformanceMetricType.CONCURRENT_AGENTS,
                target_value=200.0,     # Epic 1 target: 200+ agents
                warning_threshold=150.0,
                critical_threshold=100.0,
                unit="agents",
                description="Maximum concurrent agents without degradation"
            ),
            PerformanceMetricType.THROUGHPUT: PerformanceTarget(
                metric_type=PerformanceMetricType.THROUGHPUT,
                target_value=100.0,     # Epic 1 target: 100 requests/second
                warning_threshold=75.0,
                critical_threshold=50.0,
                unit="requests/second",
                description="System throughput capacity"
            ),
            PerformanceMetricType.ERROR_RATE: PerformanceTarget(
                metric_type=PerformanceMetricType.ERROR_RATE,
                target_value=0.1,       # Epic 1 target: <0.1% error rate
                warning_threshold=0.5,
                critical_threshold=1.0,
                unit="percentage",
                description="System error rate"
            )
        }
    
    async def record_metric(self, metric: PerformanceMetric) -> None:
        """Record a performance metric for analysis."""
        self.metrics_history[metric.metric_type].append(metric)
        
        # Check for immediate performance issues
        await self._check_performance_thresholds(metric)
        
        # Update ML models if enough data is available
        if len(self.metrics_history[metric.metric_type]) >= 10:
            await self._update_trend_model(metric.metric_type)
    
    async def _check_performance_thresholds(self, metric: PerformanceMetric) -> None:
        """Check if metric exceeds performance thresholds."""
        target = self.performance_targets.get(metric.metric_type)
        if not target:
            return
        
        if metric.value >= target.critical_threshold:
            recommendation = OptimizationRecommendation(
                priority=OptimizationPriority.CRITICAL,
                component=metric.component or "system",
                issue_description=f"{metric.metric_type.value} exceeded critical threshold: {metric.value} {target.unit}",
                recommended_action=await self._generate_optimization_action(metric),
                expected_improvement=f"Reduce {metric.metric_type.value} to <{target.target_value} {target.unit}",
                implementation_complexity="Medium",
                estimated_effort_hours=2,
                impact_metrics=[metric.metric_type]
            )
            await self._execute_optimization_recommendation(recommendation)
        
        elif metric.value >= target.warning_threshold:
            logger.warning(
                f"Performance warning: {metric.metric_type.value} approaching threshold",
                current_value=metric.value,
                warning_threshold=target.warning_threshold,
                target=target.target_value,
                unit=target.unit
            )
    
    async def _generate_optimization_action(self, metric: PerformanceMetric) -> str:
        """Generate specific optimization action based on metric type."""
        action_map = {
            PerformanceMetricType.RESPONSE_TIME: "Optimize database queries, enable caching, review algorithmic complexity",
            PerformanceMetricType.MEMORY_USAGE: "Implement memory pooling, optimize data structures, enable garbage collection tuning",
            PerformanceMetricType.CPU_UTILIZATION: "Optimize computational algorithms, implement async processing, enable connection pooling",
            PerformanceMetricType.CONCURRENT_AGENTS: "Scale horizontally, optimize resource allocation, implement load balancing",
            PerformanceMetricType.THROUGHPUT: "Optimize request handling, implement request batching, scale processing capacity",
            PerformanceMetricType.ERROR_RATE: "Review error handling, implement circuit breakers, enhance fault tolerance"
        }
        return action_map.get(metric.metric_type, "Review system performance and optimize bottlenecks")
    
    async def _update_trend_model(self, metric_type: PerformanceMetricType) -> None:
        """Update ML trend model for predictive optimization."""
        try:
            metrics = list(self.metrics_history[metric_type])
            if len(metrics) < 10:
                return
            
            # Prepare data for ML model
            timestamps = np.array([(m.timestamp - metrics[0].timestamp).total_seconds() for m in metrics])
            values = np.array([m.value for m in metrics])
            
            # Scale features
            if metric_type not in self.scalers:
                self.scalers[metric_type] = StandardScaler()
            
            X = timestamps.reshape(-1, 1)
            X_scaled = self.scalers[metric_type].fit_transform(X)
            
            # Train/update model
            if metric_type not in self.trend_models:
                self.trend_models[metric_type] = LinearRegression()
            
            self.trend_models[metric_type].fit(X_scaled, values)
            
            # Generate predictive recommendations
            await self._generate_predictive_recommendations(metric_type)
            
        except Exception as e:
            logger.warning(f"Failed to update trend model for {metric_type}: {e}")
    
    async def _generate_predictive_recommendations(self, metric_type: PerformanceMetricType) -> None:
        """Generate predictive optimization recommendations."""
        try:
            model = self.trend_models.get(metric_type)
            scaler = self.scalers.get(metric_type)
            target = self.performance_targets.get(metric_type)
            
            if not all([model, scaler, target]):
                return
            
            # Predict values for next hour
            current_time = datetime.utcnow()
            future_time = current_time + timedelta(hours=1)
            
            # Get baseline timestamp from first metric
            metrics = list(self.metrics_history[metric_type])
            baseline_time = metrics[0].timestamp
            
            future_seconds = (future_time - baseline_time).total_seconds()
            future_X = np.array([[future_seconds]])
            future_X_scaled = scaler.transform(future_X)
            
            predicted_value = model.predict(future_X_scaled)[0]
            
            # Check if prediction exceeds thresholds
            if predicted_value >= target.warning_threshold:
                recommendation = OptimizationRecommendation(
                    priority=OptimizationPriority.MEDIUM,
                    component="system",
                    issue_description=f"Predicted {metric_type.value} will reach {predicted_value:.2f} {target.unit} within 1 hour",
                    recommended_action=f"Proactive optimization needed: {await self._generate_optimization_action(PerformanceMetric(current_time, metric_type, predicted_value))}",
                    expected_improvement=f"Prevent threshold breach, maintain <{target.target_value} {target.unit}",
                    implementation_complexity="Medium",
                    estimated_effort_hours=1,
                    impact_metrics=[metric_type]
                )
                await self._queue_optimization_recommendation(recommendation)
                
        except Exception as e:
            logger.warning(f"Failed to generate predictive recommendations for {metric_type}: {e}")
    
    async def _execute_optimization_recommendation(self, recommendation: OptimizationRecommendation) -> None:
        """Execute immediate optimization recommendation."""
        logger.error(
            "CRITICAL PERFORMANCE ISSUE - Immediate optimization required",
            priority=recommendation.priority.value,
            component=recommendation.component,
            issue=recommendation.issue_description,
            action=recommendation.recommended_action
        )
        
        # Add to optimization history
        self.optimization_history.append(recommendation)
        
        # In a production system, this would trigger automated optimization
        # For now, we log the critical issue for immediate attention
    
    async def _queue_optimization_recommendation(self, recommendation: OptimizationRecommendation) -> None:
        """Queue optimization recommendation for future execution."""
        logger.info(
            "Predictive optimization recommendation generated",
            priority=recommendation.priority.value,
            component=recommendation.component,
            issue=recommendation.issue_description,
            expected_improvement=recommendation.expected_improvement
        )
        
        self.optimization_history.append(recommendation)
    
    async def start_monitoring(self) -> None:
        """Start real-time performance monitoring."""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started for Epic 1 optimization")
    
    async def stop_monitoring(self) -> None:
        """Stop real-time performance monitoring."""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for collecting system metrics."""
        while self.monitoring_active:
            try:
                await asyncio.sleep(self.monitoring_interval)
                
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Collect application metrics
                await self._collect_application_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-level performance metrics."""
        try:
            process = psutil.Process()
            
            # Memory usage
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            await self.record_metric(PerformanceMetric(
                timestamp=datetime.utcnow(),
                metric_type=PerformanceMetricType.MEMORY_USAGE,
                value=memory_mb,
                context={"process_id": process.pid},
                component="system"
            ))
            
            # CPU utilization
            cpu_percent = process.cpu_percent()
            
            await self.record_metric(PerformanceMetric(
                timestamp=datetime.utcnow(),
                metric_type=PerformanceMetricType.CPU_UTILIZATION,
                value=cpu_percent,
                context={"process_id": process.pid},
                component="system"
            ))
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
    
    async def _collect_application_metrics(self) -> None:
        """Collect application-level performance metrics."""
        try:
            # This would integrate with the SimpleOrchestrator and unified managers
            # to collect application-specific metrics like response times,
            # agent counts, and throughput metrics
            
            # For now, we'll implement basic placeholder metrics
            # In Epic 1 implementation, this would be fully integrated
            
            current_time = datetime.utcnow()
            
            # Simulate response time metric (would be real in implementation)
            await self.record_metric(PerformanceMetric(
                timestamp=current_time,
                metric_type=PerformanceMetricType.RESPONSE_TIME,
                value=25.0 + (5.0 * (time.time() % 10)),  # Simulated variance
                context={"endpoint": "api_health_check"},
                component="api"
            ))
            
        except Exception as e:
            logger.warning(f"Failed to collect application metrics: {e}")
    
    def register_component(self, name: str, component: Any) -> None:
        """Register a component for targeted performance monitoring."""
        self.registered_components[name] = weakref.ref(component)
        logger.info(f"Registered component for performance monitoring: {name}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {}
        
        for metric_type, target in self.performance_targets.items():
            metrics = list(self.metrics_history[metric_type])
            if not metrics:
                continue
            
            values = [m.value for m in metrics[-10:]]  # Last 10 measurements
            
            summary[metric_type.value] = {
                "current_value": values[-1] if values else None,
                "average": statistics.mean(values) if values else None,
                "target": target.target_value,
                "unit": target.unit,
                "status": "OK" if (values[-1] if values else float('inf')) <= target.target_value else "WARNING",
                "trend": "stable"  # Would calculate actual trend from ML model
            }
        
        summary["optimization_recommendations"] = len([
            r for r in self.optimization_history 
            if r.priority in [OptimizationPriority.CRITICAL, OptimizationPriority.HIGH]
        ])
        
        return summary
    
    async def optimize_performance(self, component: Optional[str] = None) -> List[OptimizationRecommendation]:
        """Generate comprehensive performance optimization recommendations."""
        recommendations = []
        
        # Analyze current metrics against targets
        for metric_type, target in self.performance_targets.items():
            metrics = list(self.metrics_history[metric_type])
            if not metrics:
                continue
            
            recent_values = [m.value for m in metrics[-5:]]
            avg_value = statistics.mean(recent_values)
            
            if avg_value > target.target_value:
                priority = OptimizationPriority.HIGH if avg_value > target.warning_threshold else OptimizationPriority.MEDIUM
                
                recommendation = OptimizationRecommendation(
                    priority=priority,
                    component=component or "system",
                    issue_description=f"{metric_type.value} averaging {avg_value:.2f} {target.unit}, target is {target.target_value} {target.unit}",
                    recommended_action=await self._generate_optimization_action(PerformanceMetric(
                        datetime.utcnow(), metric_type, avg_value
                    )),
                    expected_improvement=f"Improve {metric_type.value} by {((avg_value - target.target_value) / avg_value * 100):.1f}%",
                    implementation_complexity="Medium",
                    estimated_effort_hours=3,
                    impact_metrics=[metric_type]
                )
                recommendations.append(recommendation)
        
        return recommendations


# Global instance for Epic 1 optimization
_performance_optimization_framework: Optional[PerformanceOptimizationFramework] = None


def get_performance_optimization_framework() -> PerformanceOptimizationFramework:
    """Get or create the global performance optimization framework."""
    global _performance_optimization_framework
    
    if _performance_optimization_framework is None:
        _performance_optimization_framework = PerformanceOptimizationFramework()
    
    return _performance_optimization_framework


async def start_epic1_performance_monitoring() -> None:
    """Start Epic 1 performance monitoring."""
    framework = get_performance_optimization_framework()
    await framework.start_monitoring()
    logger.info("Epic 1 Performance Excellence monitoring activated")


async def stop_epic1_performance_monitoring() -> None:
    """Stop Epic 1 performance monitoring."""
    framework = get_performance_optimization_framework()
    await framework.stop_monitoring()


# Utility functions for integration with consolidated architecture

async def measure_response_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Measure response time of a function call."""
    start_time = time.time()
    try:
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        return result, response_time
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        logger.error(f"Function call failed after {response_time:.2f}ms: {e}")
        raise


def performance_monitor(component_name: str = "unknown"):
    """Decorator for automatic performance monitoring."""
    def decorator(func: Callable):
        async def async_wrapper(*args, **kwargs):
            result, response_time = await measure_response_time(func, *args, **kwargs)
            
            framework = get_performance_optimization_framework()
            await framework.record_metric(PerformanceMetric(
                timestamp=datetime.utcnow(),
                metric_type=PerformanceMetricType.RESPONSE_TIME,
                value=response_time,
                context={"function": func.__name__},
                component=component_name
            ))
            
            return result
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                response_time = (time.time() - start_time) * 1000
                
                # For sync functions, we can't await the metric recording
                # In a production system, this would use a background task
                logger.info(f"Performance: {func.__name__} completed in {response_time:.2f}ms")
                
                return result
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                logger.error(f"Performance: {func.__name__} failed after {response_time:.2f}ms: {e}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


if __name__ == "__main__":
    # Example usage and testing
    async def example_epic1_usage():
        """Example of how to use the Performance Optimization Framework."""
        framework = get_performance_optimization_framework()
        
        # Start monitoring
        await framework.start_monitoring()
        
        # Simulate some metrics
        await framework.record_metric(PerformanceMetric(
            timestamp=datetime.utcnow(),
            metric_type=PerformanceMetricType.RESPONSE_TIME,
            value=45.0,
            component="api",
            context={"endpoint": "/health"}
        ))
        
        # Wait a bit for monitoring
        await asyncio.sleep(10)
        
        # Get performance summary
        summary = framework.get_performance_summary()
        print("Performance Summary:", summary)
        
        # Generate optimization recommendations
        recommendations = await framework.optimize_performance()
        for rec in recommendations:
            print(f"Recommendation: {rec.priority.value} - {rec.issue_description}")
        
        # Stop monitoring
        await framework.stop_monitoring()
    
    # Run example
    asyncio.run(example_epic1_usage())