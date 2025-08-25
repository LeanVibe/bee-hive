"""
Comprehensive Performance Monitoring and Benchmarking System

Advanced performance monitoring system for CLI operations targeting <500ms response times.
Provides real-time metrics, benchmarking, and optimization recommendations.

Features:
- Real-time performance tracking
- Comprehensive benchmarking suite
- Performance regression detection
- Optimization recommendations
- Historical performance data
- CLI-specific metrics
"""

import time
import asyncio
import statistics
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import threading
from contextlib import contextmanager
import psutil
import sys
import subprocess


class PerformanceLevel(Enum):
    """Performance rating levels."""
    EXCELLENT = "excellent"      # <100ms
    GOOD = "good"               # 100-300ms
    ACCEPTABLE = "acceptable"   # 300-500ms
    POOR = "poor"              # 500-1000ms
    UNACCEPTABLE = "unacceptable"  # >1000ms


@dataclass
class PerformanceMetric:
    """Individual performance measurement."""
    name: str
    value: float
    unit: str
    timestamp: float
    target: Optional[float] = None
    level: Optional[PerformanceLevel] = None
    
    def __post_init__(self):
        """Calculate performance level based on value."""
        if self.unit == "ms" and self.level is None:
            if self.value < 100:
                self.level = PerformanceLevel.EXCELLENT
            elif self.value < 300:
                self.level = PerformanceLevel.GOOD
            elif self.value < 500:
                self.level = PerformanceLevel.ACCEPTABLE
            elif self.value < 1000:
                self.level = PerformanceLevel.POOR
            else:
                self.level = PerformanceLevel.UNACCEPTABLE
    
    @property
    def is_within_target(self) -> bool:
        """Check if metric is within target."""
        if self.target is None:
            return True
        return self.value <= self.target
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp,
            'target': self.target,
            'level': self.level.value if self.level else None,
            'within_target': self.is_within_target
        }


@dataclass
class BenchmarkResult:
    """Benchmark execution result."""
    command: str
    execution_times: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    iterations: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.iterations == 0:
            return 0.0
        return (self.iterations - len(self.errors)) / self.iterations
    
    @property
    def average_time_ms(self) -> float:
        """Calculate average execution time in milliseconds."""
        if not self.execution_times:
            return 0.0
        return statistics.mean(self.execution_times) * 1000
    
    @property
    def median_time_ms(self) -> float:
        """Calculate median execution time in milliseconds."""
        if not self.execution_times:
            return 0.0
        return statistics.median(self.execution_times) * 1000
    
    @property
    def p95_time_ms(self) -> float:
        """Calculate 95th percentile execution time."""
        if not self.execution_times:
            return 0.0
        sorted_times = sorted(self.execution_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)] * 1000
    
    @property
    def performance_level(self) -> PerformanceLevel:
        """Get performance level based on average time."""
        avg_ms = self.average_time_ms
        if avg_ms < 100:
            return PerformanceLevel.EXCELLENT
        elif avg_ms < 300:
            return PerformanceLevel.GOOD
        elif avg_ms < 500:
            return PerformanceLevel.ACCEPTABLE
        elif avg_ms < 1000:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.UNACCEPTABLE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'command': self.command,
            'iterations': self.iterations,
            'success_rate': self.success_rate,
            'average_time_ms': self.average_time_ms,
            'median_time_ms': self.median_time_ms,
            'p95_time_ms': self.p95_time_ms,
            'performance_level': self.performance_level.value,
            'errors': len(self.errors),
            'timestamp': self.timestamp
        }


class PerformanceTracker:
    """
    Real-time performance tracking system.
    
    Collects and analyzes performance metrics for CLI operations,
    providing insights and optimization recommendations.
    """
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.benchmarks: List[BenchmarkResult] = []
        self._lock = threading.Lock()
        self._start_time = time.time()
        
        # Performance targets
        self.targets = {
            'cli_command_ms': 500,
            'config_load_ms': 20,
            'orchestrator_init_ms': 50,
            'database_query_ms': 50,
            'redis_command_ms': 10
        }
    
    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "ms",
        target: Optional[float] = None
    ) -> PerformanceMetric:
        """Record a performance metric."""
        # Use predefined target if not specified
        if target is None:
            target = self.targets.get(f"{name}_{unit}")
        
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            target=target
        )
        
        with self._lock:
            self.metrics.append(metric)
        
        return metric
    
    def get_metrics(
        self,
        name_filter: Optional[str] = None,
        last_n: Optional[int] = None
    ) -> List[PerformanceMetric]:
        """Get performance metrics with optional filtering."""
        with self._lock:
            filtered_metrics = self.metrics
            
            if name_filter:
                filtered_metrics = [m for m in filtered_metrics if name_filter in m.name]
            
            if last_n:
                filtered_metrics = filtered_metrics[-last_n:]
            
            return filtered_metrics.copy()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            if not self.metrics:
                return {'status': 'no_data', 'metrics_count': 0}
            
            # Calculate aggregate statistics
            recent_metrics = self.metrics[-50:]  # Last 50 metrics
            
            # Group metrics by name
            metric_groups = {}
            for metric in recent_metrics:
                if metric.name not in metric_groups:
                    metric_groups[metric.name] = []
                metric_groups[metric.name].append(metric.value)
            
            # Calculate statistics for each group
            summary = {
                'total_metrics': len(self.metrics),
                'recent_metrics': len(recent_metrics),
                'uptime_seconds': time.time() - self._start_time,
                'metric_groups': {}
            }
            
            for name, values in metric_groups.items():
                if values:
                    target = self.targets.get(f"{name}_ms")
                    avg_value = statistics.mean(values)
                    
                    summary['metric_groups'][name] = {
                        'count': len(values),
                        'average': avg_value,
                        'median': statistics.median(values),
                        'min': min(values),
                        'max': max(values),
                        'target': target,
                        'within_target': avg_value <= target if target else None,
                        'performance_level': self._get_performance_level(avg_value).value
                    }
            
            return summary
    
    def _get_performance_level(self, value_ms: float) -> PerformanceLevel:
        """Get performance level for a value in milliseconds."""
        if value_ms < 100:
            return PerformanceLevel.EXCELLENT
        elif value_ms < 300:
            return PerformanceLevel.GOOD
        elif value_ms < 500:
            return PerformanceLevel.ACCEPTABLE
        elif value_ms < 1000:
            return PerformanceLevel.POOR
        else:
            return PerformanceLevel.UNACCEPTABLE
    
    def detect_performance_regressions(self) -> List[Dict[str, Any]]:
        """Detect performance regressions in recent metrics."""
        regressions = []
        
        with self._lock:
            # Group metrics by name
            metric_groups = {}
            for metric in self.metrics[-100:]:  # Check last 100 metrics
                if metric.name not in metric_groups:
                    metric_groups[metric.name] = []
                metric_groups[metric.name].append(metric)
            
            for name, metrics_list in metric_groups.items():
                if len(metrics_list) < 10:  # Need enough data
                    continue
                
                # Compare recent vs historical performance
                recent = [m.value for m in metrics_list[-5:]]  # Last 5
                historical = [m.value for m in metrics_list[:-5]]  # All but last 5
                
                if recent and historical:
                    recent_avg = statistics.mean(recent)
                    historical_avg = statistics.mean(historical)
                    
                    # Detect significant regression (>20% slower)
                    if recent_avg > historical_avg * 1.2:
                        regressions.append({
                            'metric_name': name,
                            'recent_avg_ms': recent_avg,
                            'historical_avg_ms': historical_avg,
                            'regression_percent': ((recent_avg - historical_avg) / historical_avg) * 100,
                            'severity': 'high' if recent_avg > historical_avg * 1.5 else 'medium'
                        })
        
        return regressions
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []
        summary = self.get_performance_summary()
        
        for name, stats in summary.get('metric_groups', {}).items():
            avg_value = stats['average']
            target = stats['target']
            level = stats['performance_level']
            
            if level in [PerformanceLevel.POOR.value, PerformanceLevel.UNACCEPTABLE.value]:
                recommendations.append({
                    'type': 'performance_issue',
                    'metric': name,
                    'current_value': avg_value,
                    'target_value': target,
                    'severity': 'high' if level == PerformanceLevel.UNACCEPTABLE.value else 'medium',
                    'recommendation': self._get_metric_recommendation(name, avg_value)
                })
        
        # Check for regressions
        regressions = self.detect_performance_regressions()
        for regression in regressions:
            recommendations.append({
                'type': 'performance_regression',
                'metric': regression['metric_name'],
                'regression_percent': regression['regression_percent'],
                'severity': regression['severity'],
                'recommendation': f"Performance has regressed by {regression['regression_percent']:.1f}%. Investigate recent changes."
            })
        
        return recommendations
    
    def _get_metric_recommendation(self, metric_name: str, value: float) -> str:
        """Get specific recommendation for a metric."""
        if 'config_load' in metric_name:
            return "Consider using lightweight configuration loader or increasing cache TTL"
        elif 'orchestrator_init' in metric_name:
            return "Enable orchestrator caching or reduce initialization dependencies"
        elif 'database_query' in metric_name:
            return "Optimize database queries, add indexes, or enable connection pooling"
        elif 'redis_command' in metric_name:
            return "Enable Redis pipelining or increase connection pool size"
        elif 'cli_command' in metric_name:
            return "Enable CLI caching, optimize imports, or reduce initialization overhead"
        else:
            return f"Optimize {metric_name} - current value {value}ms exceeds target"


@contextmanager
def performance_measurement(name: str, tracker: Optional[PerformanceTracker] = None):
    """Context manager for measuring performance."""
    start_time = time.time()
    
    try:
        yield
    finally:
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        
        if tracker:
            tracker.record_metric(name, execution_time_ms)
        
        # Log performance based on level
        if execution_time_ms > 500:
            print(f"🐌 {name}: {execution_time_ms:.1f}ms (SLOW)")
        elif execution_time_ms < 100:
            print(f"⚡ {name}: {execution_time_ms:.1f}ms")


class CLIBenchmarkSuite:
    """
    Comprehensive CLI benchmarking suite.
    
    Runs performance benchmarks for all CLI commands and generates
    detailed performance reports.
    """
    
    def __init__(self, cli_module_path: str = "python3 -m app.hive_cli"):
        self.cli_module_path = cli_module_path
        self.tracker = PerformanceTracker()
        self.results: List[BenchmarkResult] = []
    
    def benchmark_command(
        self,
        command: str,
        iterations: int = 5,
        timeout: float = 10.0
    ) -> BenchmarkResult:
        """Benchmark a specific CLI command."""
        result = BenchmarkResult(command=command, iterations=iterations)
        
        print(f"🔄 Benchmarking: {command} ({iterations} iterations)")
        
        for i in range(iterations):
            try:
                start_time = time.time()
                
                # Execute CLI command
                process = subprocess.run(
                    f"{self.cli_module_path} {command}".split(),
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                if process.returncode == 0:
                    result.execution_times.append(execution_time)
                else:
                    result.errors.append(f"Command failed with return code {process.returncode}")
                
            except subprocess.TimeoutExpired:
                result.errors.append(f"Command timed out after {timeout}s")
            except Exception as e:
                result.errors.append(f"Execution error: {str(e)}")
        
        # Record metrics
        if result.execution_times:
            avg_time_ms = result.average_time_ms
            self.tracker.record_metric(f"cli_{command.replace(' ', '_')}", avg_time_ms)
        
        self.results.append(result)
        return result
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive benchmark of all CLI commands."""
        commands = [
            "status",
            "agent list",
            "agent ls", 
            "version",
            "doctor",
            "metrics"
        ]
        
        print("🚀 Starting comprehensive CLI benchmark...")
        start_time = time.time()
        
        benchmark_results = {}
        for command in commands:
            result = self.benchmark_command(command, iterations=3)
            benchmark_results[command] = result.to_dict()
        
        total_time = time.time() - start_time
        
        # Generate summary
        successful_commands = [r for r in self.results if r.success_rate > 0.8]
        fast_commands = [r for r in self.results if r.average_time_ms < 500]
        
        summary = {
            'benchmark_duration_seconds': total_time,
            'total_commands': len(commands),
            'successful_commands': len(successful_commands),
            'fast_commands_under_500ms': len(fast_commands),
            'success_rate': len(successful_commands) / len(commands),
            'performance_rate': len(fast_commands) / len(commands),
            'average_command_time_ms': statistics.mean([r.average_time_ms for r in self.results if r.execution_times]),
            'results': benchmark_results,
            'performance_summary': self.tracker.get_performance_summary(),
            'recommendations': self.tracker.get_optimization_recommendations()
        }
        
        print(f"✅ Benchmark completed in {total_time:.2f}s")
        print(f"📊 Success rate: {summary['success_rate']:.1%}")
        print(f"⚡ Performance rate: {summary['performance_rate']:.1%}")
        
        return summary
    
    def save_benchmark_report(self, filename: Optional[str] = None) -> str:
        """Save benchmark report to file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"cli_benchmark_report_{timestamp}.json"
        
        report = self.run_comprehensive_benchmark()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Benchmark report saved to {filename}")
        return filename


# Global performance tracker
_global_tracker: Optional[PerformanceTracker] = None
_tracker_lock = threading.Lock()


def get_performance_tracker() -> PerformanceTracker:
    """Get global performance tracker."""
    global _global_tracker
    
    if _global_tracker is None:
        with _tracker_lock:
            if _global_tracker is None:
                _global_tracker = PerformanceTracker()
    
    return _global_tracker


def record_cli_performance(name: str, value_ms: float):
    """Record CLI performance metric."""
    tracker = get_performance_tracker()
    tracker.record_metric(name, value_ms)


def get_cli_performance_summary() -> Dict[str, Any]:
    """Get CLI performance summary."""
    tracker = get_performance_tracker()
    return tracker.get_performance_summary()


def get_performance_recommendations() -> List[Dict[str, Any]]:
    """Get performance optimization recommendations."""
    tracker = get_performance_tracker()
    return tracker.get_optimization_recommendations()


# CLI command for running benchmarks
def run_cli_benchmark(save_report: bool = True) -> Dict[str, Any]:
    """Run CLI benchmark suite."""
    suite = CLIBenchmarkSuite()
    
    if save_report:
        report_file = suite.save_benchmark_report()
        return {'report_file': report_file, 'summary': suite.run_comprehensive_benchmark()}
    else:
        return suite.run_comprehensive_benchmark()


# Performance monitoring decorators for CLI functions
def monitor_cli_performance(command_name: str):
    """Decorator to monitor CLI command performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = get_performance_tracker()
            
            with performance_measurement(f"cli_{command_name}", tracker):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


async def monitor_async_cli_performance(command_name: str, func, *args, **kwargs):
    """Monitor async CLI command performance."""
    tracker = get_performance_tracker()
    
    start_time = time.time()
    try:
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        execution_time_ms = (end_time - start_time) * 1000
        tracker.record_metric(f"async_cli_{command_name}", execution_time_ms)
        
        return result
    except Exception as e:
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        tracker.record_metric(f"async_cli_{command_name}_error", execution_time_ms)
        raise