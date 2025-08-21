"""
Performance Regression Detection System

Provides continuous performance monitoring and regression detection during
component consolidation to ensure consolidation doesn't degrade system performance.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import statistics

import psutil


class RegressionSeverity(Enum):
    """Performance regression severity levels."""
    NONE = "none"
    MINOR = "minor"        # <10% degradation
    MODERATE = "moderate"  # 10-25% degradation  
    SEVERE = "severe"      # 25-50% degradation
    CRITICAL = "critical"  # >50% degradation


class PerformanceMetric(Enum):
    """Types of performance metrics to monitor."""
    RESPONSE_TIME = "response_time_ms"
    THROUGHPUT = "throughput_ops_per_sec"
    MEMORY_USAGE = "memory_mb"
    CPU_USAGE = "cpu_usage_percent"
    ERROR_RATE = "error_rate_percent"
    QUEUE_DEPTH = "queue_depth"
    CONNECTION_POOL = "connection_pool_utilization"


@dataclass
class PerformanceSnapshot:
    """Single performance measurement snapshot."""
    timestamp: datetime
    component: str
    operation: str
    metrics: Dict[PerformanceMetric, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""
    component: str
    operation: str
    baseline_date: datetime
    sample_count: int
    metrics: Dict[PerformanceMetric, Dict[str, float]] = field(default_factory=dict)  # mean, std, p95, etc.


@dataclass
class RegressionDetection:
    """Performance regression detection result."""
    component: str
    operation: str
    detected_at: datetime
    severity: RegressionSeverity
    degraded_metrics: Dict[PerformanceMetric, Dict[str, float]] = field(default_factory=dict)
    baseline_metrics: Dict[PerformanceMetric, Dict[str, float]] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class PerformanceRegressionDetector:
    """
    Continuous performance monitoring and regression detection system.
    
    Monitors component performance during consolidation and detects regressions
    by comparing current performance against established baselines.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.cwd() / ".performance_data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Performance snapshots storage
        self.snapshots: List[PerformanceSnapshot] = []
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.regressions: List[RegressionDetection] = []
        
        # Regression detection thresholds
        self.regression_thresholds = {
            RegressionSeverity.MINOR: {
                PerformanceMetric.RESPONSE_TIME: 1.10,  # 10% slower
                PerformanceMetric.THROUGHPUT: 0.90,     # 10% fewer ops/sec
                PerformanceMetric.MEMORY_USAGE: 1.15,   # 15% more memory
                PerformanceMetric.CPU_USAGE: 1.20,      # 20% more CPU
                PerformanceMetric.ERROR_RATE: 1.05,     # 5% more errors
            },
            RegressionSeverity.MODERATE: {
                PerformanceMetric.RESPONSE_TIME: 1.25,  # 25% slower
                PerformanceMetric.THROUGHPUT: 0.75,     # 25% fewer ops/sec
                PerformanceMetric.MEMORY_USAGE: 1.30,   # 30% more memory
                PerformanceMetric.CPU_USAGE: 1.35,      # 35% more CPU
                PerformanceMetric.ERROR_RATE: 1.15,     # 15% more errors
            },
            RegressionSeverity.SEVERE: {
                PerformanceMetric.RESPONSE_TIME: 1.50,  # 50% slower
                PerformanceMetric.THROUGHPUT: 0.50,     # 50% fewer ops/sec
                PerformanceMetric.MEMORY_USAGE: 1.50,   # 50% more memory
                PerformanceMetric.CPU_USAGE: 1.60,      # 60% more CPU
                PerformanceMetric.ERROR_RATE: 1.30,     # 30% more errors
            },
            RegressionSeverity.CRITICAL: {
                PerformanceMetric.RESPONSE_TIME: 2.0,   # 100% slower
                PerformanceMetric.THROUGHPUT: 0.25,     # 75% fewer ops/sec
                PerformanceMetric.MEMORY_USAGE: 2.0,    # 100% more memory
                PerformanceMetric.CPU_USAGE: 2.5,       # 150% more CPU
                PerformanceMetric.ERROR_RATE: 2.0,      # 100% more errors
            }
        }
        
        # Load existing data
        self._load_historical_data()
    
    async def capture_performance_snapshot(
        self, 
        component: str, 
        operation: str,
        custom_metrics: Optional[Dict[PerformanceMetric, float]] = None
    ) -> PerformanceSnapshot:
        """
        Capture a performance snapshot for a component operation.
        
        Args:
            component: Component name (e.g., "redis_broker", "security_system")
            operation: Operation name (e.g., "send_message", "authenticate")
            custom_metrics: Additional custom metrics to include
            
        Returns:
            PerformanceSnapshot with captured metrics
        """
        snapshot = PerformanceSnapshot(
            timestamp=datetime.utcnow(),
            component=component,
            operation=operation
        )
        
        # Capture system-level metrics
        process = psutil.Process()
        
        # Memory usage
        memory_info = process.memory_info()
        snapshot.metrics[PerformanceMetric.MEMORY_USAGE] = memory_info.rss / (1024 * 1024)  # MB
        
        # CPU usage (average over short interval)
        cpu_percent = process.cpu_percent(interval=0.1)
        snapshot.metrics[PerformanceMetric.CPU_USAGE] = cpu_percent
        
        # Add custom metrics if provided
        if custom_metrics:
            snapshot.metrics.update(custom_metrics)
        
        # Store snapshot
        self.snapshots.append(snapshot)
        
        # Check for regressions if we have a baseline
        baseline_key = f"{component}:{operation}"
        if baseline_key in self.baselines:
            regression = await self._detect_regression(snapshot, self.baselines[baseline_key])
            if regression.severity != RegressionSeverity.NONE:
                self.regressions.append(regression)
        
        return snapshot
    
    async def establish_performance_baseline(
        self, 
        component: str, 
        operation: str,
        measurement_function: Callable[[], Dict[PerformanceMetric, float]],
        sample_count: int = 50
    ) -> PerformanceBaseline:
        """
        Establish a performance baseline by running multiple measurements.
        
        Args:
            component: Component name
            operation: Operation name  
            measurement_function: Async function that returns performance metrics
            sample_count: Number of samples to collect for baseline
            
        Returns:
            PerformanceBaseline with statistical metrics
        """
        print(f"📊 Establishing performance baseline for {component}:{operation}")
        print(f"   Collecting {sample_count} samples...")
        
        samples: Dict[PerformanceMetric, List[float]] = {}
        
        for i in range(sample_count):
            if i % 10 == 0:
                print(f"   Sample {i+1}/{sample_count}")
            
            # Execute measurement function
            if asyncio.iscoroutinefunction(measurement_function):
                metrics = await measurement_function()
            else:
                metrics = measurement_function()
            
            # Collect metrics
            for metric, value in metrics.items():
                if metric not in samples:
                    samples[metric] = []
                samples[metric].append(value)
            
            # Small delay between samples
            await asyncio.sleep(0.01)
        
        # Calculate statistical metrics for baseline
        baseline = PerformanceBaseline(
            component=component,
            operation=operation,
            baseline_date=datetime.utcnow(),
            sample_count=sample_count
        )
        
        for metric, values in samples.items():
            if values:
                baseline.metrics[metric] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'min': min(values),
                    'max': max(values),
                    'p95': self._percentile(values, 95),
                    'p99': self._percentile(values, 99)
                }
        
        # Store baseline
        baseline_key = f"{component}:{operation}"
        self.baselines[baseline_key] = baseline
        
        print(f"✅ Baseline established for {component}:{operation}")
        for metric, stats in baseline.metrics.items():
            print(f"   {metric.value}: mean={stats['mean']:.2f}, p95={stats['p95']:.2f}")
        
        return baseline
    
    async def _detect_regression(
        self, 
        snapshot: PerformanceSnapshot, 
        baseline: PerformanceBaseline
    ) -> RegressionDetection:
        """Detect performance regression by comparing snapshot to baseline."""
        detection = RegressionDetection(
            component=snapshot.component,
            operation=snapshot.operation,
            detected_at=snapshot.timestamp,
            severity=RegressionSeverity.NONE
        )
        
        worst_severity = RegressionSeverity.NONE
        
        # Compare each metric against baseline
        for metric, current_value in snapshot.metrics.items():
            if metric not in baseline.metrics:
                continue
            
            baseline_mean = baseline.metrics[metric]['mean']
            baseline_p95 = baseline.metrics[metric]['p95']
            
            # Determine regression severity for this metric
            metric_severity = self._calculate_metric_severity(metric, current_value, baseline_mean)
            
            if metric_severity != RegressionSeverity.NONE:
                detection.degraded_metrics[metric] = {
                    'current': current_value,
                    'baseline_mean': baseline_mean,
                    'baseline_p95': baseline_p95,
                    'degradation_percent': ((current_value - baseline_mean) / baseline_mean) * 100
                }
                
                detection.baseline_metrics[metric] = baseline.metrics[metric]
                
                # Track worst severity
                if metric_severity.value > worst_severity.value:
                    worst_severity = metric_severity
        
        detection.severity = worst_severity
        
        # Generate recommendations based on detected regressions
        if detection.severity != RegressionSeverity.NONE:
            detection.recommendations = self._generate_recommendations(detection)
        
        return detection
    
    def _calculate_metric_severity(
        self, 
        metric: PerformanceMetric, 
        current_value: float, 
        baseline_value: float
    ) -> RegressionSeverity:
        """Calculate regression severity for a specific metric."""
        if baseline_value == 0:
            return RegressionSeverity.NONE
        
        # Check each severity level from most severe to least
        for severity in [RegressionSeverity.CRITICAL, RegressionSeverity.SEVERE, 
                        RegressionSeverity.MODERATE, RegressionSeverity.MINOR]:
            if metric not in self.regression_thresholds[severity]:
                continue
            
            threshold = self.regression_thresholds[severity][metric]
            
            if metric == PerformanceMetric.THROUGHPUT:
                # For throughput, lower is worse
                if current_value <= (baseline_value * threshold):
                    return severity
            else:
                # For other metrics, higher is worse
                if current_value >= (baseline_value * threshold):
                    return severity
        
        return RegressionSeverity.NONE
    
    def _generate_recommendations(self, detection: RegressionDetection) -> List[str]:
        """Generate recommendations based on detected regressions."""
        recommendations = []
        
        for metric in detection.degraded_metrics:
            if metric == PerformanceMetric.MEMORY_USAGE:
                recommendations.append("Consider memory optimization: check for memory leaks, optimize data structures")
            elif metric == PerformanceMetric.RESPONSE_TIME:
                recommendations.append("Optimize response time: review algorithmic complexity, add caching")
            elif metric == PerformanceMetric.THROUGHPUT:
                recommendations.append("Improve throughput: optimize concurrent processing, reduce blocking operations")
            elif metric == PerformanceMetric.CPU_USAGE:
                recommendations.append("Reduce CPU usage: profile hot paths, optimize computations")
            elif metric == PerformanceMetric.ERROR_RATE:
                recommendations.append("Address error rate increase: review error handling, validate inputs")
        
        if detection.severity in [RegressionSeverity.SEVERE, RegressionSeverity.CRITICAL]:
            recommendations.append("URGENT: Consider rolling back consolidation due to severe performance regression")
        elif detection.severity == RegressionSeverity.MODERATE:
            recommendations.append("WARNING: Performance regression detected, investigate before proceeding")
        
        return recommendations
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def get_performance_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the specified time window."""
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        recent_snapshots = [
            s for s in self.snapshots 
            if s.timestamp >= cutoff_time
        ]
        
        recent_regressions = [
            r for r in self.regressions 
            if r.detected_at >= cutoff_time
        ]
        
        # Group by component
        component_stats = {}
        for snapshot in recent_snapshots:
            if snapshot.component not in component_stats:
                component_stats[snapshot.component] = {
                    'snapshot_count': 0,
                    'operations': set(),
                    'avg_metrics': {}
                }
            
            stats = component_stats[snapshot.component]
            stats['snapshot_count'] += 1
            stats['operations'].add(snapshot.operation)
            
            # Aggregate metrics
            for metric, value in snapshot.metrics.items():
                metric_name = metric.value
                if metric_name not in stats['avg_metrics']:
                    stats['avg_metrics'][metric_name] = []
                stats['avg_metrics'][metric_name].append(value)
        
        # Calculate averages
        for component, stats in component_stats.items():
            for metric_name, values in stats['avg_metrics'].items():
                stats['avg_metrics'][metric_name] = {
                    'mean': statistics.mean(values),
                    'samples': len(values)
                }
            stats['operations'] = list(stats['operations'])
        
        return {
            'time_window_hours': time_window_hours,
            'total_snapshots': len(recent_snapshots),
            'total_regressions': len(recent_regressions),
            'components_monitored': len(component_stats),
            'component_stats': component_stats,
            'regression_summary': {
                'by_severity': {
                    severity.value: len([r for r in recent_regressions if r.severity == severity])
                    for severity in RegressionSeverity
                }
            },
            'baselines_established': len(self.baselines)
        }
    
    def _load_historical_data(self):
        """Load historical performance data from disk."""
        baselines_file = self.data_dir / "baselines.json"
        if baselines_file.exists():
            try:
                with open(baselines_file, 'r') as f:
                    data = json.load(f)
                    # Convert back to PerformanceBaseline objects
                    for key, baseline_data in data.items():
                        baseline = PerformanceBaseline(**baseline_data)
                        baseline.baseline_date = datetime.fromisoformat(baseline.baseline_date)
                        # Convert metric enums back
                        baseline.metrics = {
                            PerformanceMetric(k): v for k, v in baseline.metrics.items()
                        }
                        self.baselines[key] = baseline
            except Exception as e:
                print(f"Warning: Could not load baseline data: {e}")
    
    def save_performance_data(self):
        """Save performance data to disk."""
        # Save baselines
        baselines_file = self.data_dir / "baselines.json"
        baselines_data = {}
        for key, baseline in self.baselines.items():
            baseline_dict = asdict(baseline)
            baseline_dict['baseline_date'] = baseline.baseline_date.isoformat()
            baseline_dict['metrics'] = {
                metric.value: stats for metric, stats in baseline.metrics.items()
            }
            baselines_data[key] = baseline_dict
        
        with open(baselines_file, 'w') as f:
            json.dump(baselines_data, f, indent=2)
        
        print(f"💾 Performance data saved to {self.data_dir}")


# Example usage and testing
async def example_redis_performance_measurement() -> Dict[PerformanceMetric, float]:
    """Example performance measurement function for Redis operations."""
    start_time = time.time()
    
    # Simulate Redis operation
    await asyncio.sleep(0.001)  # 1ms operation
    
    response_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return {
        PerformanceMetric.RESPONSE_TIME: response_time,
        PerformanceMetric.THROUGHPUT: 1000.0,  # ops per second
        PerformanceMetric.ERROR_RATE: 0.0  # no errors
    }


async def example_security_performance_measurement() -> Dict[PerformanceMetric, float]:
    """Example performance measurement function for security operations."""
    start_time = time.time()
    
    # Simulate security validation
    await asyncio.sleep(0.005)  # 5ms operation
    
    response_time = (time.time() - start_time) * 1000
    
    return {
        PerformanceMetric.RESPONSE_TIME: response_time,
        PerformanceMetric.THROUGHPUT: 200.0,
        PerformanceMetric.ERROR_RATE: 0.1  # 0.1% error rate
    }


async def main():
    """Demonstration of performance regression detection system."""
    print("📊 Testing Framework Agent - Performance Regression Detection")
    print("=" * 65)
    
    detector = PerformanceRegressionDetector()
    
    # Establish baselines for key components
    print("\n🎯 Establishing performance baselines...")
    
    redis_baseline = await detector.establish_performance_baseline(
        "redis_broker", 
        "send_message", 
        example_redis_performance_measurement,
        sample_count=25
    )
    
    security_baseline = await detector.establish_performance_baseline(
        "security_system",
        "rate_limit_check",
        example_security_performance_measurement, 
        sample_count=25
    )
    
    # Simulate some normal operations
    print("\n📈 Capturing performance snapshots...")
    for i in range(5):
        await detector.capture_performance_snapshot(
            "redis_broker",
            "send_message", 
            await example_redis_performance_measurement()
        )
        
        await detector.capture_performance_snapshot(
            "security_system",
            "rate_limit_check",
            await example_security_performance_measurement()
        )
        
        await asyncio.sleep(0.01)
    
    # Simulate a performance regression
    print("\n⚠️ Simulating performance regression...")
    regression_metrics = await example_redis_performance_measurement()
    regression_metrics[PerformanceMetric.RESPONSE_TIME] *= 2.5  # 150% slower
    
    await detector.capture_performance_snapshot(
        "redis_broker",
        "send_message",
        regression_metrics
    )
    
    # Generate performance summary
    summary = detector.get_performance_summary()
    print(f"\n📊 Performance Summary:")
    print(f"   Total snapshots: {summary['total_snapshots']}")
    print(f"   Components monitored: {summary['components_monitored']}")
    print(f"   Regressions detected: {summary['total_regressions']}")
    print(f"   Baselines established: {summary['baselines_established']}")
    
    if detector.regressions:
        print(f"\n🚨 Detected Regressions:")
        for regression in detector.regressions:
            print(f"   {regression.component}:{regression.operation}")
            print(f"   Severity: {regression.severity.value}")
            if regression.recommendations:
                print(f"   Recommendations: {regression.recommendations[0]}")
    
    # Save data
    detector.save_performance_data()
    
    print("\n✅ Performance regression detection system ready")


if __name__ == "__main__":
    asyncio.run(main())