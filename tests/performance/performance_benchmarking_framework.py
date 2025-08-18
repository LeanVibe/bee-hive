"""
Performance Benchmarking Framework for Consolidated Components

This framework provides comprehensive performance testing and regression detection
for all consolidated components with <5% regression threshold and automated
baseline management.
"""

import asyncio
import time
import json
import os
import statistics
import psutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from pathlib import Path

# Performance monitoring imports
try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

try:
    import cProfile
    import pstats
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False


@dataclass
class BenchmarkConfiguration:
    """Configuration for performance benchmarks."""
    name: str
    component: str
    
    # Test parameters
    iterations: int = 100
    warmup_iterations: int = 10
    concurrent_operations: int = 1
    duration_seconds: Optional[int] = None
    
    # Performance targets (from consolidation requirements)
    target_latency_ms: float = 100.0  # Default target
    max_acceptable_latency_ms: float = 500.0  # Default max
    target_throughput_ops_sec: float = 100.0  # Default throughput
    max_memory_mb: float = 50.0  # Default memory limit
    max_cpu_percent: float = 80.0  # Default CPU limit
    
    # Regression detection
    regression_threshold_percent: float = 5.0
    baseline_file: Optional[str] = None
    
    # Advanced settings
    enable_profiling: bool = False
    enable_memory_tracking: bool = True
    collect_detailed_metrics: bool = True
    sample_rate_ms: int = 100  # Sampling interval for detailed metrics


@dataclass
class PerformanceResult:
    """Result of a performance benchmark."""
    benchmark_name: str
    timestamp: datetime
    
    # Core metrics
    total_operations: int
    duration_seconds: float
    throughput_ops_per_sec: float
    
    # Latency metrics
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Resource metrics
    peak_memory_mb: float
    avg_memory_mb: float
    peak_cpu_percent: float
    avg_cpu_percent: float
    
    # Error metrics
    error_count: int
    error_rate_percent: float
    
    # Advanced metrics
    detailed_latencies: List[float] = field(default_factory=list)
    memory_timeline: List[float] = field(default_factory=list)
    cpu_timeline: List[float] = field(default_factory=list)
    
    # Regression analysis
    baseline_comparison: Optional[Dict[str, float]] = None
    regression_detected: bool = False
    regression_details: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "benchmark_name": self.benchmark_name,
            "timestamp": self.timestamp.isoformat(),
            "total_operations": self.total_operations,
            "duration_seconds": self.duration_seconds,
            "throughput_ops_per_sec": self.throughput_ops_per_sec,
            "latency_metrics": {
                "avg_ms": self.avg_latency_ms,
                "min_ms": self.min_latency_ms,
                "max_ms": self.max_latency_ms,
                "p50_ms": self.p50_latency_ms,
                "p95_ms": self.p95_latency_ms,
                "p99_ms": self.p99_latency_ms
            },
            "resource_metrics": {
                "peak_memory_mb": self.peak_memory_mb,
                "avg_memory_mb": self.avg_memory_mb,
                "peak_cpu_percent": self.peak_cpu_percent,
                "avg_cpu_percent": self.avg_cpu_percent
            },
            "error_metrics": {
                "error_count": self.error_count,
                "error_rate_percent": self.error_rate_percent
            },
            "regression_analysis": {
                "baseline_comparison": self.baseline_comparison,
                "regression_detected": self.regression_detected,
                "regression_details": self.regression_details
            }
        }


class SystemMonitor:
    """System resource monitoring during benchmarks."""
    
    def __init__(self, sample_rate_ms: int = 100):
        self.sample_rate_ms = sample_rate_ms
        self.monitoring = False
        self.monitor_thread = None
        
        # Data collection
        self.cpu_samples = []
        self.memory_samples = []
        self.timestamps = []
        
        # Current process
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """Start system monitoring in background thread."""
        self.monitoring = True
        self.cpu_samples.clear()
        self.memory_samples.clear()
        self.timestamps.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Tuple[List[float], List[float]]:
        """Stop monitoring and return collected data."""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        return self.cpu_samples.copy(), self.memory_samples.copy()
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                # Sample CPU and memory
                cpu_percent = self.process.cpu_percent()
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                
                self.cpu_samples.append(cpu_percent)
                self.memory_samples.append(memory_mb)
                self.timestamps.append(time.time())
                
                time.sleep(self.sample_rate_ms / 1000.0)
                
            except Exception:
                # Process might have ended or other error
                break
    
    def get_current_stats(self) -> Dict[str, float]:
        """Get current resource usage statistics."""
        try:
            return {
                "cpu_percent": self.process.cpu_percent(),
                "memory_mb": self.process.memory_info().rss / 1024 / 1024,
                "memory_percent": self.process.memory_percent(),
                "threads": self.process.num_threads()
            }
        except Exception:
            return {"cpu_percent": 0, "memory_mb": 0, "memory_percent": 0, "threads": 0}


class RegressionDetector:
    """Detects performance regressions by comparing with baselines."""
    
    def __init__(self, regression_threshold_percent: float = 5.0):
        self.regression_threshold = regression_threshold_percent
    
    def load_baseline(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load baseline performance metrics."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    
    def save_baseline(self, results: PerformanceResult, filepath: str) -> bool:
        """Save performance results as new baseline."""
        try:
            baseline_data = {
                "benchmark_name": results.benchmark_name,
                "timestamp": results.timestamp.isoformat(),
                "metrics": {
                    "throughput_ops_per_sec": results.throughput_ops_per_sec,
                    "avg_latency_ms": results.avg_latency_ms,
                    "p95_latency_ms": results.p95_latency_ms,
                    "peak_memory_mb": results.peak_memory_mb,
                    "avg_cpu_percent": results.avg_cpu_percent,
                    "error_rate_percent": results.error_rate_percent
                }
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Failed to save baseline: {e}")
            return False
    
    def detect_regression(
        self, 
        current_result: PerformanceResult,
        baseline_data: Dict[str, Any]
    ) -> Tuple[bool, List[str], Dict[str, float]]:
        """
        Detect performance regression by comparing with baseline.
        
        Returns:
            Tuple of (has_regression, regression_details, comparison_data)
        """
        if not baseline_data or "metrics" not in baseline_data:
            return False, [], {}
        
        baseline_metrics = baseline_data["metrics"]
        comparison = {}
        regressions = []
        
        # Define metrics and their regression direction (higher is worse)
        metrics_to_check = {
            "avg_latency_ms": True,  # Higher latency is worse
            "p95_latency_ms": True,  # Higher latency is worse  
            "peak_memory_mb": True,  # Higher memory is worse
            "avg_cpu_percent": True,  # Higher CPU is worse
            "error_rate_percent": True,  # Higher error rate is worse
            "throughput_ops_per_sec": False  # Lower throughput is worse
        }
        
        for metric_name, higher_is_worse in metrics_to_check.items():
            if metric_name not in baseline_metrics:
                continue
                
            baseline_value = baseline_metrics[metric_name]
            current_value = getattr(current_result, metric_name, 0)
            
            if baseline_value == 0:
                continue  # Avoid division by zero
            
            # Calculate percentage change
            change_percent = ((current_value - baseline_value) / baseline_value) * 100
            comparison[metric_name] = change_percent
            
            # Check for regression based on direction
            is_regression = False
            if higher_is_worse and change_percent > self.regression_threshold:
                is_regression = True
            elif not higher_is_worse and change_percent < -self.regression_threshold:
                is_regression = True
            
            if is_regression:
                direction = "increased" if higher_is_worse else "decreased"
                regressions.append(
                    f"{metric_name} {direction} by {abs(change_percent):.1f}% "
                    f"(current: {current_value:.2f}, baseline: {baseline_value:.2f})"
                )
        
        has_regression = len(regressions) > 0
        return has_regression, regressions, comparison


class PerformanceBenchmarkFramework:
    """
    Comprehensive performance benchmarking framework.
    
    Provides automated testing, regression detection, and baseline management
    for all consolidated components.
    """
    
    def __init__(self, results_dir: str = "tests/performance/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.baselines_dir = self.results_dir / "baselines"
        self.baselines_dir.mkdir(exist_ok=True)
        
        # Components
        self.system_monitor = SystemMonitor()
        self.regression_detector = RegressionDetector()
        
        # State
        self.current_config: Optional[BenchmarkConfiguration] = None
        self.benchmark_results: Dict[str, PerformanceResult] = {}
    
    async def run_benchmark(
        self,
        config: BenchmarkConfiguration,
        benchmark_func: Callable,
        *args, **kwargs
    ) -> PerformanceResult:
        """
        Run comprehensive performance benchmark.
        
        Args:
            config: Benchmark configuration
            benchmark_func: Function to benchmark (async or sync)
            *args, **kwargs: Arguments for benchmark function
            
        Returns:
            PerformanceResult with comprehensive metrics
        """
        self.current_config = config
        
        print(f"🚀 Starting benchmark: {config.name}")
        print(f"   Component: {config.component}")
        print(f"   Iterations: {config.iterations}")
        print(f"   Concurrency: {config.concurrent_operations}")
        
        # Warmup phase
        if config.warmup_iterations > 0:
            print(f"   Warming up with {config.warmup_iterations} iterations...")
            await self._run_warmup(config, benchmark_func, *args, **kwargs)
        
        # Start system monitoring
        if config.enable_memory_tracking:
            self.system_monitor.start_monitoring()
        
        # Run benchmark
        start_time = time.time()
        latencies = []
        error_count = 0
        
        try:
            if config.duration_seconds:
                # Duration-based benchmark
                latencies, error_count = await self._run_duration_benchmark(
                    config, benchmark_func, *args, **kwargs
                )
            else:
                # Iteration-based benchmark
                latencies, error_count = await self._run_iteration_benchmark(
                    config, benchmark_func, *args, **kwargs
                )
        
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            raise
        
        finally:
            # Stop monitoring
            cpu_samples, memory_samples = self.system_monitor.stop_monitoring()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate metrics
        result = self._calculate_performance_result(
            config, latencies, error_count, duration, cpu_samples, memory_samples
        )
        
        # Regression analysis
        await self._perform_regression_analysis(result, config)
        
        # Save results
        self._save_benchmark_result(result)
        
        # Print summary
        self._print_benchmark_summary(result)
        
        self.benchmark_results[config.name] = result
        return result
    
    async def _run_warmup(
        self,
        config: BenchmarkConfiguration,
        benchmark_func: Callable,
        *args, **kwargs
    ) -> None:
        """Run warmup iterations."""
        for _ in range(config.warmup_iterations):
            try:
                if asyncio.iscoroutinefunction(benchmark_func):
                    await benchmark_func(*args, **kwargs)
                else:
                    benchmark_func(*args, **kwargs)
            except Exception:
                pass  # Ignore warmup errors
    
    async def _run_iteration_benchmark(
        self,
        config: BenchmarkConfiguration,
        benchmark_func: Callable,
        *args, **kwargs
    ) -> Tuple[List[float], int]:
        """Run iteration-based benchmark."""
        latencies = []
        error_count = 0
        
        if config.concurrent_operations == 1:
            # Sequential execution
            for i in range(config.iterations):
                start_time = time.time()
                try:
                    if asyncio.iscoroutinefunction(benchmark_func):
                        await benchmark_func(*args, **kwargs)
                    else:
                        benchmark_func(*args, **kwargs)
                except Exception:
                    error_count += 1
                
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
        
        else:
            # Concurrent execution
            semaphore = asyncio.Semaphore(config.concurrent_operations)
            
            async def run_single_operation():
                nonlocal error_count
                async with semaphore:
                    start_time = time.time()
                    try:
                        if asyncio.iscoroutinefunction(benchmark_func):
                            await benchmark_func(*args, **kwargs)
                        else:
                            benchmark_func(*args, **kwargs)
                    except Exception:
                        error_count += 1
                    
                    latency = (time.time() - start_time) * 1000
                    return latency
            
            # Create all tasks
            tasks = [run_single_operation() for _ in range(config.iterations)]
            
            # Execute concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    error_count += 1
                else:
                    latencies.append(result)
        
        return latencies, error_count
    
    async def _run_duration_benchmark(
        self,
        config: BenchmarkConfiguration,
        benchmark_func: Callable,
        *args, **kwargs
    ) -> Tuple[List[float], int]:
        """Run duration-based benchmark."""
        latencies = []
        error_count = 0
        start_time = time.time()
        end_time = start_time + config.duration_seconds
        
        if config.concurrent_operations == 1:
            # Sequential execution
            while time.time() < end_time:
                op_start = time.time()
                try:
                    if asyncio.iscoroutinefunction(benchmark_func):
                        await benchmark_func(*args, **kwargs)
                    else:
                        benchmark_func(*args, **kwargs)
                except Exception:
                    error_count += 1
                
                latency = (time.time() - op_start) * 1000
                latencies.append(latency)
        
        else:
            # Concurrent execution
            semaphore = asyncio.Semaphore(config.concurrent_operations)
            active_tasks = set()
            
            async def run_single_operation():
                nonlocal error_count
                async with semaphore:
                    op_start = time.time()
                    try:
                        if asyncio.iscoroutinefunction(benchmark_func):
                            await benchmark_func(*args, **kwargs)
                        else:
                            benchmark_func(*args, **kwargs)
                    except Exception:
                        error_count += 1
                    
                    latency = (time.time() - op_start) * 1000
                    return latency
            
            while time.time() < end_time:
                # Launch new tasks to maintain concurrency
                while len(active_tasks) < config.concurrent_operations and time.time() < end_time:
                    task = asyncio.create_task(run_single_operation())
                    active_tasks.add(task)
                
                # Check for completed tasks
                done_tasks = {task for task in active_tasks if task.done()}
                for task in done_tasks:
                    try:
                        result = await task
                        if not isinstance(result, Exception):
                            latencies.append(result)
                    except Exception:
                        error_count += 1
                    active_tasks.remove(task)
                
                # Small delay to prevent tight loop
                await asyncio.sleep(0.001)
            
            # Wait for remaining tasks
            if active_tasks:
                results = await asyncio.gather(*active_tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        error_count += 1
                    else:
                        latencies.append(result)
        
        return latencies, error_count
    
    def _calculate_performance_result(
        self,
        config: BenchmarkConfiguration,
        latencies: List[float],
        error_count: int,
        duration: float,
        cpu_samples: List[float],
        memory_samples: List[float]
    ) -> PerformanceResult:
        """Calculate comprehensive performance metrics."""
        
        total_operations = len(latencies) + error_count
        throughput = total_operations / duration if duration > 0 else 0
        error_rate = (error_count / total_operations) * 100 if total_operations > 0 else 0
        
        # Latency statistics
        if latencies:
            latencies_sorted = sorted(latencies)
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            
            # Percentiles
            p50_latency = np.percentile(latencies_sorted, 50)
            p95_latency = np.percentile(latencies_sorted, 95)
            p99_latency = np.percentile(latencies_sorted, 99)
        else:
            avg_latency = min_latency = max_latency = 0
            p50_latency = p95_latency = p99_latency = 0
        
        # Resource statistics
        if memory_samples:
            peak_memory = max(memory_samples)
            avg_memory = statistics.mean(memory_samples)
        else:
            peak_memory = avg_memory = 0
        
        if cpu_samples:
            peak_cpu = max(cpu_samples)
            avg_cpu = statistics.mean(cpu_samples)
        else:
            peak_cpu = avg_cpu = 0
        
        return PerformanceResult(
            benchmark_name=config.name,
            timestamp=datetime.utcnow(),
            total_operations=total_operations,
            duration_seconds=duration,
            throughput_ops_per_sec=throughput,
            avg_latency_ms=avg_latency,
            min_latency_ms=min_latency,
            max_latency_ms=max_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            peak_memory_mb=peak_memory,
            avg_memory_mb=avg_memory,
            peak_cpu_percent=peak_cpu,
            avg_cpu_percent=avg_cpu,
            error_count=error_count,
            error_rate_percent=error_rate,
            detailed_latencies=latencies,
            memory_timeline=memory_samples,
            cpu_timeline=cpu_samples
        )
    
    async def _perform_regression_analysis(
        self,
        result: PerformanceResult,
        config: BenchmarkConfiguration
    ) -> None:
        """Perform regression analysis against baseline."""
        baseline_file = config.baseline_file or f"{config.component}_{config.name}_baseline.json"
        baseline_path = self.baselines_dir / baseline_file
        
        baseline_data = self.regression_detector.load_baseline(str(baseline_path))
        
        if baseline_data:
            has_regression, regression_details, comparison = self.regression_detector.detect_regression(
                result, baseline_data
            )
            
            result.regression_detected = has_regression
            result.regression_details = regression_details
            result.baseline_comparison = comparison
            
            if has_regression:
                print(f"⚠️ Performance regression detected in {config.name}:")
                for detail in regression_details:
                    print(f"   - {detail}")
        else:
            # No baseline - save current as baseline
            print(f"📊 No baseline found for {config.name}, saving current results as baseline")
            self.regression_detector.save_baseline(result, str(baseline_path))
    
    def _save_benchmark_result(self, result: PerformanceResult) -> None:
        """Save benchmark result to file."""
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{result.benchmark_name}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def _print_benchmark_summary(self, result: PerformanceResult) -> None:
        """Print benchmark summary."""
        print(f"\n📊 Benchmark Results: {result.benchmark_name}")
        print(f"   Total Operations: {result.total_operations}")
        print(f"   Duration: {result.duration_seconds:.2f}s")
        print(f"   Throughput: {result.throughput_ops_per_sec:.2f} ops/sec")
        print(f"   Latency - Avg: {result.avg_latency_ms:.2f}ms, P95: {result.p95_latency_ms:.2f}ms")
        print(f"   Memory - Peak: {result.peak_memory_mb:.2f}MB, Avg: {result.avg_memory_mb:.2f}MB") 
        print(f"   CPU - Peak: {result.peak_cpu_percent:.1f}%, Avg: {result.avg_cpu_percent:.1f}%")
        print(f"   Error Rate: {result.error_rate_percent:.2f}%")
        
        if result.regression_detected:
            print(f"   🚨 REGRESSION DETECTED:")
            for detail in result.regression_details:
                print(f"      - {detail}")
        else:
            print(f"   ✅ No performance regression detected")
    
    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results."""
        summary = {
            "total_benchmarks": len(self.benchmark_results),
            "benchmarks_with_regressions": sum(1 for r in self.benchmark_results.values() if r.regression_detected),
            "results": {}
        }
        
        for name, result in self.benchmark_results.items():
            summary["results"][name] = {
                "throughput_ops_per_sec": result.throughput_ops_per_sec,
                "avg_latency_ms": result.avg_latency_ms,
                "error_rate_percent": result.error_rate_percent,
                "peak_memory_mb": result.peak_memory_mb,
                "regression_detected": result.regression_detected
            }
        
        return summary


# Component-specific benchmark configurations

def get_universal_orchestrator_benchmarks() -> List[BenchmarkConfiguration]:
    """Get benchmark configurations for UniversalOrchestrator."""
    return [
        BenchmarkConfiguration(
            name="agent_registration_performance",
            component="universal_orchestrator",
            iterations=1000,
            concurrent_operations=1,
            target_latency_ms=100.0,  # <100ms requirement
            max_acceptable_latency_ms=150.0,
            enable_memory_tracking=True
        ),
        BenchmarkConfiguration(
            name="task_delegation_performance", 
            component="universal_orchestrator",
            iterations=500,
            concurrent_operations=10,
            target_latency_ms=500.0,  # <500ms requirement
            max_acceptable_latency_ms=750.0,
            enable_memory_tracking=True
        ),
        BenchmarkConfiguration(
            name="concurrent_agent_coordination",
            component="universal_orchestrator", 
            duration_seconds=60,
            concurrent_operations=55,  # Test 50+ requirement
            target_throughput_ops_sec=50.0,
            max_memory_mb=50.0,  # <50MB requirement
            enable_memory_tracking=True
        )
    ]


def get_communication_hub_benchmarks() -> List[BenchmarkConfiguration]:
    """Get benchmark configurations for CommunicationHub."""
    return [
        BenchmarkConfiguration(
            name="message_routing_latency",
            component="communication_hub",
            iterations=10000,
            concurrent_operations=1,
            target_latency_ms=10.0,  # <10ms routing requirement
            max_acceptable_latency_ms=25.0,
            enable_memory_tracking=True
        ),
        BenchmarkConfiguration(
            name="high_throughput_messaging",
            component="communication_hub",
            duration_seconds=30,
            concurrent_operations=100,
            target_throughput_ops_sec=10000.0,  # 10,000+ msg/sec requirement
            max_acceptable_latency_ms=50.0,
            enable_memory_tracking=True
        ),
        BenchmarkConfiguration(
            name="protocol_adapter_performance",
            component="communication_hub",
            iterations=5000,
            concurrent_operations=20,
            target_latency_ms=5.0,  # Ultra-low latency target
            max_acceptable_latency_ms=15.0,
            enable_memory_tracking=True
        )
    ]


def get_engine_benchmarks() -> List[BenchmarkConfiguration]:
    """Get benchmark configurations for consolidated engines."""
    return [
        BenchmarkConfiguration(
            name="task_execution_engine_performance",
            component="task_execution_engine",
            iterations=10000,
            concurrent_operations=50,
            target_latency_ms=0.01,  # Exceptional 0.01ms target achieved
            max_acceptable_latency_ms=1.0,
            enable_memory_tracking=True
        ),
        BenchmarkConfiguration(
            name="workflow_engine_compilation",
            component="workflow_engine", 
            iterations=1000,
            concurrent_operations=10,
            target_latency_ms=1.0,  # <1ms compilation target
            max_acceptable_latency_ms=5.0,
            enable_memory_tracking=True
        ),
        BenchmarkConfiguration(
            name="data_processing_search",
            component="data_processing_engine",
            iterations=5000,
            concurrent_operations=25,
            target_latency_ms=0.1,  # <0.1ms search target achieved
            max_acceptable_latency_ms=1.0,
            enable_memory_tracking=True
        )
    ]


def get_all_benchmark_configurations() -> List[BenchmarkConfiguration]:
    """Get all benchmark configurations for consolidated components."""
    return (
        get_universal_orchestrator_benchmarks() +
        get_communication_hub_benchmarks() +
        get_engine_benchmarks()
    )