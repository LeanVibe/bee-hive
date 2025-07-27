"""
Performance Monitor

Comprehensive performance monitoring system for measuring the impact of code
modifications. Collects metrics before and after modifications to validate
improvement claims and detect performance regressions.
"""

import asyncio
import json
import os
import psutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import structlog

logger = structlog.get_logger()


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""
    
    name: str
    value: float
    unit: str
    category: str  # 'execution', 'memory', 'disk', 'network', 'custom'
    measurement_time: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # 0.0 to 1.0


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    
    benchmark_id: str
    name: str
    baseline_metrics: List[PerformanceMetric] = field(default_factory=list)
    modified_metrics: List[PerformanceMetric] = field(default_factory=list)
    iterations: int = 0
    duration_seconds: float = 0.0
    
    # Statistical analysis
    improvement_percentage: Optional[float] = None
    statistical_significance: Optional[bool] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    
    # Metadata
    environment: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    @property
    def has_improvement(self) -> bool:
        """Check if there's a performance improvement."""
        return self.improvement_percentage is not None and self.improvement_percentage > 0
    
    @property
    def has_regression(self) -> bool:
        """Check if there's a performance regression."""
        return self.improvement_percentage is not None and self.improvement_percentage < -5  # 5% threshold


@dataclass
class PerformanceReport:
    """Comprehensive performance analysis report."""
    
    session_id: str
    modification_id: str
    benchmarks: List[BenchmarkResult] = field(default_factory=list)
    overall_improvement: Optional[float] = None
    
    # Summary statistics
    total_benchmarks: int = 0
    improved_benchmarks: int = 0
    regressed_benchmarks: int = 0
    neutral_benchmarks: int = 0
    
    # System info
    system_info: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def improvement_ratio(self) -> float:
        """Ratio of improved to total benchmarks."""
        return self.improved_benchmarks / max(1, self.total_benchmarks)
    
    @property
    def regression_ratio(self) -> float:
        """Ratio of regressed to total benchmarks."""
        return self.regressed_benchmarks / max(1, self.total_benchmarks)


class PerformanceMonitor:
    """Monitors and measures performance impact of code modifications."""
    
    def __init__(self):
        self.active_benchmarks: Dict[str, BenchmarkResult] = {}
        self.benchmark_templates = self._load_benchmark_templates()
        
    def start_benchmark(
        self,
        name: str,
        baseline_code: str,
        modified_code: str,
        test_cases: Optional[List[Dict[str, Any]]] = None,
        iterations: int = 10,
        warmup_iterations: int = 3,
        timeout_seconds: int = 300
    ) -> str:
        """Start a performance benchmark."""
        
        benchmark_id = str(uuid4())
        
        benchmark = BenchmarkResult(
            benchmark_id=benchmark_id,
            name=name,
            iterations=iterations,
            environment=self._collect_environment_info()
        )
        
        self.active_benchmarks[benchmark_id] = benchmark
        
        logger.info(
            "Starting performance benchmark",
            benchmark_id=benchmark_id,
            name=name,
            iterations=iterations
        )
        
        return benchmark_id
    
    async def run_benchmark(
        self,
        benchmark_id: str,
        baseline_code: str,
        modified_code: str,
        test_cases: Optional[List[Dict[str, Any]]] = None,
        iterations: int = 10,
        warmup_iterations: int = 3
    ) -> BenchmarkResult:
        """Run complete performance benchmark."""
        
        benchmark = self.active_benchmarks.get(benchmark_id)
        if not benchmark:
            raise ValueError(f"Benchmark {benchmark_id} not found")
        
        try:
            start_time = time.time()
            
            # Collect system baseline
            await self._collect_system_baseline()
            
            # Run baseline measurements
            logger.info("Running baseline measurements", benchmark_id=benchmark_id)
            benchmark.baseline_metrics = await self._run_code_benchmark(
                baseline_code, test_cases, iterations, warmup_iterations, "baseline"
            )
            
            # Brief pause between runs
            await asyncio.sleep(2)
            
            # Run modified code measurements
            logger.info("Running modified code measurements", benchmark_id=benchmark_id)
            benchmark.modified_metrics = await self._run_code_benchmark(
                modified_code, test_cases, iterations, warmup_iterations, "modified"
            )
            
            # Calculate performance impact
            await self._analyze_benchmark_results(benchmark)
            
            benchmark.duration_seconds = time.time() - start_time
            benchmark.completed_at = datetime.utcnow()
            
            logger.info(
                "Benchmark completed",
                benchmark_id=benchmark_id,
                duration=benchmark.duration_seconds,
                improvement=benchmark.improvement_percentage
            )
            
            return benchmark
            
        except Exception as e:
            logger.error("Benchmark failed", benchmark_id=benchmark_id, error=str(e))
            raise
        finally:
            if benchmark_id in self.active_benchmarks:
                del self.active_benchmarks[benchmark_id]
    
    async def run_micro_benchmark(
        self,
        function_name: str,
        baseline_code: str,
        modified_code: str,
        setup_code: str = "",
        iterations: int = 100000
    ) -> BenchmarkResult:
        """Run micro-benchmark for small code changes."""
        
        benchmark_id = str(uuid4())
        
        # Create micro-benchmark script
        benchmark_script = f"""
import time
import gc
import statistics

# Setup code
{setup_code}

def baseline_function():
{self._indent_code(baseline_code)}

def modified_function():
{self._indent_code(modified_code)}

def run_micro_benchmark(func, iterations):
    # Warmup
    for _ in range(min(1000, iterations // 10)):
        func()
    
    # Clear garbage
    gc.collect()
    
    # Measure
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)
    
    return {{
        'mean_time': statistics.mean(times),
        'median_time': statistics.median(times),
        'min_time': min(times),
        'max_time': max(times),
        'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
        'iterations': iterations
    }}

if __name__ == "__main__":
    print("BASELINE_START")
    baseline_results = run_micro_benchmark(baseline_function, {iterations})
    print(json.dumps(baseline_results))
    print("BASELINE_END")
    
    print("MODIFIED_START")
    modified_results = run_micro_benchmark(modified_function, {iterations})
    print(json.dumps(modified_results))
    print("MODIFIED_END")
"""
        
        # Execute benchmark
        result = await self._execute_benchmark_script(benchmark_script)
        
        # Parse results
        baseline_data = self._extract_json_section(result, "BASELINE_START", "BASELINE_END")
        modified_data = self._extract_json_section(result, "MODIFIED_START", "MODIFIED_END")
        
        # Create benchmark result
        benchmark = BenchmarkResult(
            benchmark_id=benchmark_id,
            name=f"micro_{function_name}",
            iterations=iterations,
            environment=self._collect_environment_info()
        )
        
        # Convert to metrics
        if baseline_data:
            benchmark.baseline_metrics = self._convert_to_metrics(baseline_data, "baseline")
        
        if modified_data:
            benchmark.modified_metrics = self._convert_to_metrics(modified_data, "modified")
        
        # Analyze results
        await self._analyze_benchmark_results(benchmark)
        
        return benchmark
    
    async def run_memory_benchmark(
        self,
        baseline_code: str,
        modified_code: str,
        data_sizes: List[int] = None,
        iterations: int = 10
    ) -> BenchmarkResult:
        """Run memory usage benchmark."""
        
        data_sizes = data_sizes or [100, 1000, 10000, 100000]
        benchmark_id = str(uuid4())
        
        benchmark_script = f"""
import psutil
import gc
import json
import os

def baseline_function(data_size):
{self._indent_code(baseline_code)}

def modified_function(data_size):
{self._indent_code(modified_code)}

def measure_memory_usage(func, data_size, iterations):
    process = psutil.Process(os.getpid())
    
    measurements = []
    
    for i in range(iterations):
        # Clear memory
        gc.collect()
        
        # Baseline memory
        baseline_memory = process.memory_info().rss
        
        # Run function
        result = func(data_size)
        
        # Peak memory
        peak_memory = process.memory_info().rss
        
        # Memory delta
        memory_delta = peak_memory - baseline_memory
        
        measurements.append({{
            'iteration': i,
            'baseline_memory_mb': baseline_memory / 1024 / 1024,
            'peak_memory_mb': peak_memory / 1024 / 1024,
            'memory_delta_mb': memory_delta / 1024 / 1024,
            'data_size': data_size
        }})
        
        # Cleanup
        del result
        gc.collect()
    
    return measurements

if __name__ == "__main__":
    data_sizes = {data_sizes}
    
    print("BASELINE_MEMORY_START")
    baseline_results = {{}}
    for size in data_sizes:
        baseline_results[str(size)] = measure_memory_usage(baseline_function, size, {iterations})
    print(json.dumps(baseline_results))
    print("BASELINE_MEMORY_END")
    
    print("MODIFIED_MEMORY_START")
    modified_results = {{}}
    for size in data_sizes:
        modified_results[str(size)] = measure_memory_usage(modified_function, size, {iterations})
    print(json.dumps(modified_results))
    print("MODIFIED_MEMORY_END")
"""
        
        # Execute benchmark
        result = await self._execute_benchmark_script(benchmark_script)
        
        # Parse and create benchmark result
        benchmark = BenchmarkResult(
            benchmark_id=benchmark_id,
            name="memory_usage",
            iterations=iterations * len(data_sizes),
            environment=self._collect_environment_info()
        )
        
        # Parse memory results
        baseline_data = self._extract_json_section(result, "BASELINE_MEMORY_START", "BASELINE_MEMORY_END")
        modified_data = self._extract_json_section(result, "MODIFIED_MEMORY_START", "MODIFIED_MEMORY_END")
        
        if baseline_data and modified_data:
            benchmark.baseline_metrics = self._convert_memory_to_metrics(baseline_data, "baseline")
            benchmark.modified_metrics = self._convert_memory_to_metrics(modified_data, "modified")
        
        await self._analyze_benchmark_results(benchmark)
        
        return benchmark
    
    async def generate_performance_report(
        self,
        session_id: str,
        modification_id: str,
        benchmarks: List[BenchmarkResult]
    ) -> PerformanceReport:
        """Generate comprehensive performance report."""
        
        report = PerformanceReport(
            session_id=session_id,
            modification_id=modification_id,
            benchmarks=benchmarks,
            system_info=self._collect_environment_info()
        )
        
        # Calculate summary statistics
        report.total_benchmarks = len(benchmarks)
        
        for benchmark in benchmarks:
            if benchmark.has_improvement:
                report.improved_benchmarks += 1
            elif benchmark.has_regression:
                report.regressed_benchmarks += 1
            else:
                report.neutral_benchmarks += 1
        
        # Calculate overall improvement
        improvements = [
            b.improvement_percentage for b in benchmarks 
            if b.improvement_percentage is not None
        ]
        
        if improvements:
            report.overall_improvement = mean(improvements)
        
        logger.info(
            "Performance report generated",
            session_id=session_id,
            total_benchmarks=report.total_benchmarks,
            improved=report.improved_benchmarks,
            regressed=report.regressed_benchmarks,
            overall_improvement=report.overall_improvement
        )
        
        return report
    
    async def _run_code_benchmark(
        self,
        code: str,
        test_cases: Optional[List[Dict[str, Any]]],
        iterations: int,
        warmup_iterations: int,
        label: str
    ) -> List[PerformanceMetric]:
        """Run benchmark for a piece of code."""
        
        metrics = []
        
        # Create benchmark script
        benchmark_script = f"""
import time
import psutil
import gc
import json
import os

def target_function(*args, **kwargs):
{self._indent_code(code)}

def run_benchmark(iterations, warmup):
    process = psutil.Process(os.getpid())
    
    # Test cases
    test_cases = {json.dumps(test_cases or [{}])}
    
    results = []
    
    for test_case in test_cases:
        # Warmup
        for _ in range(warmup):
            target_function(**test_case)
        
        # Clear garbage
        gc.collect()
        
        # Measure iterations
        times = []
        memory_usage = []
        
        for i in range(iterations):
            # Memory before
            mem_before = process.memory_info().rss
            
            # Time execution
            start_time = time.perf_counter()
            result = target_function(**test_case)
            end_time = time.perf_counter()
            
            # Memory after
            mem_after = process.memory_info().rss
            
            times.append(end_time - start_time)
            memory_usage.append(mem_after - mem_before)
            
            # Cleanup
            del result
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        avg_memory = sum(memory_usage) / len(memory_usage) / 1024 / 1024  # MB
        
        results.append({{
            'test_case': test_case,
            'avg_execution_time_ms': avg_time * 1000,
            'min_execution_time_ms': min_time * 1000,
            'max_execution_time_ms': max_time * 1000,
            'avg_memory_delta_mb': avg_memory,
            'iterations': iterations
        }})
    
    return results

if __name__ == "__main__":
    print("BENCHMARK_START")
    results = run_benchmark({iterations}, {warmup_iterations})
    print(json.dumps(results))
    print("BENCHMARK_END")
"""
        
        # Execute benchmark
        result = await self._execute_benchmark_script(benchmark_script)
        
        # Parse results
        benchmark_data = self._extract_json_section(result, "BENCHMARK_START", "BENCHMARK_END")
        
        if benchmark_data:
            for i, test_result in enumerate(benchmark_data):
                # Create metrics for each measurement
                metrics.extend([
                    PerformanceMetric(
                        name=f"execution_time_ms_{label}",
                        value=test_result["avg_execution_time_ms"],
                        unit="ms",
                        category="execution",
                        context={"test_case": i, "label": label}
                    ),
                    PerformanceMetric(
                        name=f"memory_delta_mb_{label}",
                        value=test_result["avg_memory_delta_mb"],
                        unit="MB",
                        category="memory",
                        context={"test_case": i, "label": label}
                    )
                ])
        
        return metrics
    
    async def _execute_benchmark_script(self, script: str) -> str:
        """Execute benchmark script and return output."""
        
        # Write script to temporary file
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script)
            script_path = f.name
        
        try:
            # Execute script
            process = await asyncio.create_subprocess_exec(
                'python', script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise RuntimeError(f"Benchmark script failed: {stderr.decode()}")
            
            return stdout.decode()
            
        finally:
            # Cleanup
            os.unlink(script_path)
    
    def _extract_json_section(self, output: str, start_marker: str, end_marker: str) -> Optional[Any]:
        """Extract JSON data between markers."""
        
        try:
            start_pos = output.find(start_marker)
            end_pos = output.find(end_marker)
            
            if start_pos == -1 or end_pos == -1:
                return None
            
            json_str = output[start_pos + len(start_marker):end_pos].strip()
            return json.loads(json_str)
            
        except Exception as e:
            logger.warning("Failed to extract JSON section", error=str(e))
            return None
    
    def _indent_code(self, code: str, spaces: int = 4) -> str:
        """Indent code for embedding in functions."""
        
        lines = code.strip().split('\n')
        return '\n'.join(' ' * spaces + line for line in lines)
    
    def _convert_to_metrics(self, data: Dict[str, Any], label: str) -> List[PerformanceMetric]:
        """Convert benchmark data to metrics."""
        
        metrics = []
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                # Determine unit based on key name
                unit = "ms" if "time" in key else "count" if "iterations" in key else "value"
                category = "execution" if "time" in key else "performance"
                
                metrics.append(PerformanceMetric(
                    name=f"{key}_{label}",
                    value=float(value),
                    unit=unit,
                    category=category,
                    context={"label": label}
                ))
        
        return metrics
    
    def _convert_memory_to_metrics(self, data: Dict[str, List], label: str) -> List[PerformanceMetric]:
        """Convert memory benchmark data to metrics."""
        
        metrics = []
        
        for data_size, measurements in data.items():
            # Calculate averages
            avg_memory = mean([m["memory_delta_mb"] for m in measurements])
            peak_memory = max([m["peak_memory_mb"] for m in measurements])
            
            metrics.extend([
                PerformanceMetric(
                    name=f"avg_memory_delta_{label}",
                    value=avg_memory,
                    unit="MB",
                    category="memory",
                    context={"label": label, "data_size": int(data_size)}
                ),
                PerformanceMetric(
                    name=f"peak_memory_{label}",
                    value=peak_memory,
                    unit="MB",
                    category="memory",
                    context={"label": label, "data_size": int(data_size)}
                )
            ])
        
        return metrics
    
    async def _analyze_benchmark_results(self, benchmark: BenchmarkResult) -> None:
        """Analyze benchmark results and calculate improvements."""
        
        if not benchmark.baseline_metrics or not benchmark.modified_metrics:
            return
        
        # Group metrics by name (without label suffix)
        baseline_by_name = {}
        modified_by_name = {}
        
        for metric in benchmark.baseline_metrics:
            name = metric.name.replace("_baseline", "")
            baseline_by_name[name] = metric
        
        for metric in benchmark.modified_metrics:
            name = metric.name.replace("_modified", "")
            modified_by_name[name] = metric
        
        # Calculate improvements for matching metrics
        improvements = []
        
        for name in baseline_by_name:
            if name in modified_by_name:
                baseline_value = baseline_by_name[name].value
                modified_value = modified_by_name[name].value
                
                if baseline_value > 0:
                    # For timing metrics, lower is better (negative improvement is good)
                    # For memory, lower is usually better too
                    improvement = ((baseline_value - modified_value) / baseline_value) * 100
                    improvements.append(improvement)
        
        # Calculate overall improvement
        if improvements:
            benchmark.improvement_percentage = mean(improvements)
            
            # Statistical significance (simplified)
            if len(improvements) > 1:
                std_dev = stdev(improvements)
                benchmark.statistical_significance = abs(benchmark.improvement_percentage) > std_dev
            else:
                benchmark.statistical_significance = abs(benchmark.improvement_percentage) > 5  # 5% threshold
    
    def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect system environment information."""
        
        try:
            return {
                "python_version": os.sys.version,
                "cpu_count": psutil.cpu_count(),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                "platform": os.sys.platform,
                "architecture": os.uname().machine if hasattr(os, 'uname') else "unknown",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.warning("Failed to collect environment info", error=str(e))
            return {"error": str(e)}
    
    async def _collect_system_baseline(self) -> None:
        """Collect system baseline metrics."""
        
        # Wait for system to stabilize
        await asyncio.sleep(1)
        
        # Could collect CPU usage, memory pressure, etc.
        # For now, just ensure system is ready
        pass
    
    def _load_benchmark_templates(self) -> Dict[str, str]:
        """Load benchmark templates for common patterns."""
        
        return {
            "function_execution": """
def benchmark_function({args}):
{code}
    return result
""",
            "memory_allocation": """
def benchmark_memory({size_param}):
{code}
    return data
""",
            "io_operation": """
def benchmark_io({io_params}):
{code}
    return result
"""
        }


# Export main classes
__all__ = [
    "PerformanceMonitor",
    "PerformanceMetric",
    "BenchmarkResult", 
    "PerformanceReport"
]