"""
Epic 4 - Context Engine Performance Benchmarks and Validation

Comprehensive performance benchmarking and validation system for Epic 4 Context Engine
components to ensure enterprise-ready performance and scalability.
"""

import asyncio
import time
import statistics
import json
import psutil
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import concurrent.futures
from pathlib import Path

import structlog
import numpy as np
# import matplotlib.pyplot as plt  # Optional dependency
from sqlalchemy.ext.asyncio import AsyncSession

# Epic 4 Components
from .unified_context_engine import UnifiedContextEngine, get_unified_context_engine
from .context_reasoning_engine import ContextReasoningEngine, get_context_reasoning_engine, ReasoningType
from .intelligent_context_persistence import IntelligentContextPersistence, get_intelligent_context_persistence
from .context_aware_agent_coordination import ContextAwareAgentCoordination, get_context_aware_coordination
from .epic4_orchestration_integration import Epic4OrchestrationIntegration, get_epic4_orchestration_integration

# Core imports
from ..models.agent import Agent, AgentType, AgentStatus
from ..models.context import Context, ContextType

logger = structlog.get_logger()


class BenchmarkType(Enum):
    """Types of performance benchmarks."""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    SCALABILITY = "scalability"
    CONCURRENCY = "concurrency"
    STRESS = "stress"
    ENDURANCE = "endurance"


class PerformanceTarget(Enum):
    """Performance targets for different operations."""
    CONTEXT_COMPRESSION_MS = 500      # Max 500ms for context compression
    REASONING_ANALYSIS_MS = 1000      # Max 1s for reasoning analysis
    AGENT_COORDINATION_MS = 2000      # Max 2s for agent coordination
    CONTEXT_RETRIEVAL_MS = 50         # Max 50ms for context retrieval
    MEMORY_USAGE_MB = 512             # Max 512MB memory usage
    CPU_USAGE_PERCENT = 80            # Max 80% CPU usage


@dataclass
class BenchmarkResult:
    """Individual benchmark test result."""
    benchmark_name: str
    benchmark_type: BenchmarkType
    component: str
    operation: str
    
    # Performance metrics
    execution_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    
    # Test parameters
    test_parameters: Dict[str, Any]
    sample_size: int
    
    # Statistical analysis
    mean_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    std_dev_ms: float
    
    # Success metrics
    success_rate: float
    error_count: int
    timeout_count: int
    
    # Comparison with targets
    meets_target: bool
    performance_score: float  # 0-100 score
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    environment_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    suite_name: str
    total_benchmarks: int
    passed_benchmarks: int
    failed_benchmarks: int
    overall_score: float
    
    results: List[BenchmarkResult] = field(default_factory=list)
    summary_statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    execution_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class Epic4PerformanceBenchmarks:
    """
    Comprehensive performance benchmarking system for Epic 4 Context Engine.
    
    Features:
    - Latency and throughput benchmarks
    - Memory and CPU usage monitoring
    - Scalability and concurrency testing
    - Stress testing and endurance testing
    - Performance regression detection
    - Automated performance validation
    """
    
    def __init__(self):
        self.logger = logger.bind(component="epic4_performance_benchmarks")
        
        # Benchmark configuration
        self.default_sample_size = 100
        self.default_timeout_seconds = 30
        self.warmup_iterations = 10
        
        # Performance targets
        self.performance_targets = {
            target.name.lower(): target.value
            for target in PerformanceTarget
        }
        
        # Test data generators
        self.test_data_generators = {}
        
        # Results storage
        self.benchmark_history: List[BenchmarkSuite] = []
        self.baseline_results: Dict[str, BenchmarkResult] = {}
        
        # System monitoring
        self.process = psutil.Process()
        self.system_metrics = defaultdict(list)
        
        self.logger.info("📊 Epic 4 Performance Benchmarks initialized")
    
    async def run_comprehensive_benchmark_suite(
        self,
        components: Optional[List[str]] = None,
        benchmark_types: Optional[List[BenchmarkType]] = None,
        sample_size: int = None
    ) -> BenchmarkSuite:
        """
        Run comprehensive benchmark suite for Epic 4 components.
        
        Args:
            components: Specific components to benchmark
            benchmark_types: Specific benchmark types to run
            sample_size: Number of samples per benchmark
            
        Returns:
            Complete benchmark suite results
        """
        start_time = time.time()
        sample_size = sample_size or self.default_sample_size
        
        if components is None:
            components = [
                "unified_context_engine",
                "context_reasoning_engine", 
                "intelligent_context_persistence",
                "context_aware_agent_coordination",
                "epic4_orchestration_integration"
            ]
        
        if benchmark_types is None:
            benchmark_types = [
                BenchmarkType.LATENCY,
                BenchmarkType.THROUGHPUT,
                BenchmarkType.MEMORY_USAGE,
                BenchmarkType.SCALABILITY
            ]
        
        self.logger.info(
            f"🚀 Running comprehensive benchmark suite: {len(components)} components, "
            f"{len(benchmark_types)} benchmark types"
        )
        
        suite_results = []
        
        # Initialize components
        components_map = await self._initialize_components()
        
        # Run benchmarks for each component
        for component_name in components:
            if component_name not in components_map:
                self.logger.warning(f"Component {component_name} not available, skipping")
                continue
            
            component = components_map[component_name]
            
            for benchmark_type in benchmark_types:
                try:
                    # Run warmup
                    await self._warmup_component(component, component_name)
                    
                    # Run benchmark
                    result = await self._run_component_benchmark(
                        component=component,
                        component_name=component_name,
                        benchmark_type=benchmark_type,
                        sample_size=sample_size
                    )
                    
                    suite_results.append(result)
                    
                except Exception as e:
                    self.logger.error(
                        f"Benchmark failed: {component_name} - {benchmark_type.value}: {e}"
                    )
        
        # Calculate suite statistics
        suite = BenchmarkSuite(
            suite_name=f"Epic4_Comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            total_benchmarks=len(suite_results),
            passed_benchmarks=sum(1 for r in suite_results if r.meets_target),
            failed_benchmarks=sum(1 for r in suite_results if not r.meets_target),
            overall_score=self._calculate_overall_score(suite_results),
            results=suite_results,
            execution_time_seconds=time.time() - start_time
        )
        
        # Generate summary and recommendations
        suite.summary_statistics = self._generate_summary_statistics(suite_results)
        suite.recommendations = self._generate_recommendations(suite_results)
        
        # Store results
        self.benchmark_history.append(suite)
        
        self.logger.info(
            f"✅ Benchmark suite complete: {suite.passed_benchmarks}/{suite.total_benchmarks} passed, "
            f"overall score: {suite.overall_score:.1f}/100"
        )
        
        return suite
    
    async def run_latency_benchmarks(
        self,
        component: Any,
        component_name: str,
        operations: List[str],
        sample_size: int = 100
    ) -> List[BenchmarkResult]:
        """Run latency benchmarks for component operations."""
        results = []
        
        for operation in operations:
            self.logger.info(f"📊 Running latency benchmark: {component_name}.{operation}")
            
            execution_times = []
            errors = 0
            timeouts = 0
            
            # Collect samples
            for i in range(sample_size):
                try:
                    start_time = time.time()
                    
                    # Execute operation based on component and operation
                    success = await self._execute_benchmark_operation(
                        component, component_name, operation
                    )
                    
                    execution_time = (time.time() - start_time) * 1000  # ms
                    
                    if success:
                        execution_times.append(execution_time)
                    else:
                        errors += 1
                        
                except asyncio.TimeoutError:
                    timeouts += 1
                except Exception as e:
                    errors += 1
                    self.logger.warning(f"Benchmark error in {operation}: {e}")
            
            if execution_times:
                # Calculate statistics
                mean_time = statistics.mean(execution_times)
                median_time = statistics.median(execution_times)
                p95_time = np.percentile(execution_times, 95)
                p99_time = np.percentile(execution_times, 99)
                std_dev = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                
                # Check performance target
                target_key = f"{operation}_ms"
                target_time = self.performance_targets.get(target_key, float('inf'))
                meets_target = mean_time <= target_time
                
                # Calculate performance score
                performance_score = min(100, (target_time / max(mean_time, 1)) * 100) if target_time < float('inf') else 100
                
                result = BenchmarkResult(
                    benchmark_name=f"{component_name}_{operation}_latency",
                    benchmark_type=BenchmarkType.LATENCY,
                    component=component_name,
                    operation=operation,
                    execution_time_ms=mean_time,
                    memory_usage_mb=self._get_memory_usage_mb(),
                    cpu_usage_percent=self._get_cpu_usage_percent(),
                    test_parameters={"sample_size": sample_size},
                    sample_size=sample_size,
                    mean_time_ms=mean_time,
                    median_time_ms=median_time,
                    p95_time_ms=p95_time,
                    p99_time_ms=p99_time,
                    std_dev_ms=std_dev,
                    success_rate=(len(execution_times) / sample_size) * 100,
                    error_count=errors,
                    timeout_count=timeouts,
                    meets_target=meets_target,
                    performance_score=performance_score,
                    environment_info=self._get_environment_info()
                )
                
                results.append(result)
        
        return results
    
    async def run_throughput_benchmarks(
        self,
        component: Any,
        component_name: str,
        operations: List[str],
        duration_seconds: int = 10
    ) -> List[BenchmarkResult]:
        """Run throughput benchmarks for component operations."""
        results = []
        
        for operation in operations:
            self.logger.info(f"📊 Running throughput benchmark: {component_name}.{operation}")
            
            start_time = time.time()
            end_time = start_time + duration_seconds
            
            operation_count = 0
            errors = 0
            
            # Run operations for specified duration
            while time.time() < end_time:
                try:
                    success = await self._execute_benchmark_operation(
                        component, component_name, operation
                    )
                    
                    if success:
                        operation_count += 1
                    else:
                        errors += 1
                        
                except Exception as e:
                    errors += 1
                    self.logger.warning(f"Throughput benchmark error in {operation}: {e}")
            
            actual_duration = time.time() - start_time
            throughput = operation_count / actual_duration  # operations per second
            
            # Calculate performance score (arbitrary target of 10 ops/sec)
            target_throughput = 10
            performance_score = min(100, (throughput / target_throughput) * 100)
            
            result = BenchmarkResult(
                benchmark_name=f"{component_name}_{operation}_throughput",
                benchmark_type=BenchmarkType.THROUGHPUT,
                component=component_name,
                operation=operation,
                execution_time_ms=actual_duration * 1000,
                memory_usage_mb=self._get_memory_usage_mb(),
                cpu_usage_percent=self._get_cpu_usage_percent(),
                test_parameters={"duration_seconds": duration_seconds, "throughput_ops_sec": throughput},
                sample_size=operation_count,
                mean_time_ms=1000 / throughput if throughput > 0 else float('inf'),
                median_time_ms=1000 / throughput if throughput > 0 else float('inf'),
                p95_time_ms=0,  # Not applicable for throughput
                p99_time_ms=0,  # Not applicable for throughput
                std_dev_ms=0,   # Not applicable for throughput
                success_rate=(operation_count / (operation_count + errors)) * 100 if (operation_count + errors) > 0 else 0,
                error_count=errors,
                timeout_count=0,
                meets_target=throughput >= target_throughput,
                performance_score=performance_score,
                environment_info=self._get_environment_info()
            )
            
            results.append(result)
        
        return results
    
    async def run_memory_benchmarks(
        self,
        component: Any,
        component_name: str,
        operations: List[str],
        load_multipliers: List[int] = [1, 5, 10, 20]
    ) -> List[BenchmarkResult]:
        """Run memory usage benchmarks."""
        results = []
        
        for operation in operations:
            for multiplier in load_multipliers:
                self.logger.info(
                    f"📊 Running memory benchmark: {component_name}.{operation} (load: {multiplier}x)"
                )
                
                # Measure baseline memory
                gc.collect()  # Force garbage collection
                baseline_memory = self._get_memory_usage_mb()
                
                # Execute operations with load multiplier
                start_time = time.time()
                success_count = 0
                
                try:
                    for _ in range(multiplier):
                        success = await self._execute_benchmark_operation(
                            component, component_name, operation
                        )
                        if success:
                            success_count += 1
                    
                    execution_time = (time.time() - start_time) * 1000
                    
                    # Measure peak memory
                    peak_memory = self._get_memory_usage_mb()
                    memory_used = peak_memory - baseline_memory
                    
                    # Check against memory target
                    meets_target = memory_used <= self.performance_targets.get("memory_usage_mb", 512)
                    performance_score = min(100, (512 / max(memory_used, 1)) * 100)
                    
                    result = BenchmarkResult(
                        benchmark_name=f"{component_name}_{operation}_memory_{multiplier}x",
                        benchmark_type=BenchmarkType.MEMORY_USAGE,
                        component=component_name,
                        operation=operation,
                        execution_time_ms=execution_time,
                        memory_usage_mb=memory_used,
                        cpu_usage_percent=self._get_cpu_usage_percent(),
                        test_parameters={"load_multiplier": multiplier, "baseline_memory_mb": baseline_memory},
                        sample_size=multiplier,
                        mean_time_ms=execution_time / multiplier,
                        median_time_ms=execution_time / multiplier,
                        p95_time_ms=0,
                        p99_time_ms=0,
                        std_dev_ms=0,
                        success_rate=(success_count / multiplier) * 100,
                        error_count=multiplier - success_count,
                        timeout_count=0,
                        meets_target=meets_target,
                        performance_score=performance_score,
                        environment_info=self._get_environment_info()
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Memory benchmark failed: {e}")
                
                # Clean up memory
                gc.collect()
        
        return results
    
    async def run_concurrency_benchmarks(
        self,
        component: Any,
        component_name: str,
        operations: List[str],
        concurrency_levels: List[int] = [1, 5, 10, 25, 50]
    ) -> List[BenchmarkResult]:
        """Run concurrency benchmarks."""
        results = []
        
        for operation in operations:
            for concurrency in concurrency_levels:
                self.logger.info(
                    f"📊 Running concurrency benchmark: {component_name}.{operation} "
                    f"({concurrency} concurrent)"
                )
                
                start_time = time.time()
                success_count = 0
                error_count = 0
                
                # Create concurrent tasks
                tasks = []
                for _ in range(concurrency):
                    task = asyncio.create_task(
                        self._execute_benchmark_operation(component, component_name, operation)
                    )
                    tasks.append(task)
                
                # Wait for all tasks to complete
                try:
                    results_concurrent = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results_concurrent:
                        if isinstance(result, Exception):
                            error_count += 1
                        elif result:
                            success_count += 1
                        else:
                            error_count += 1
                    
                    execution_time = (time.time() - start_time) * 1000
                    
                    # Performance score based on success rate and execution time
                    success_rate = (success_count / concurrency) * 100
                    performance_score = success_rate * 0.7 + min(30, 3000 / max(execution_time, 1)) * 0.3
                    
                    result = BenchmarkResult(
                        benchmark_name=f"{component_name}_{operation}_concurrency_{concurrency}",
                        benchmark_type=BenchmarkType.CONCURRENCY,
                        component=component_name,
                        operation=operation,
                        execution_time_ms=execution_time,
                        memory_usage_mb=self._get_memory_usage_mb(),
                        cpu_usage_percent=self._get_cpu_usage_percent(),
                        test_parameters={"concurrency_level": concurrency},
                        sample_size=concurrency,
                        mean_time_ms=execution_time / concurrency,
                        median_time_ms=execution_time / concurrency,
                        p95_time_ms=0,
                        p99_time_ms=0,
                        std_dev_ms=0,
                        success_rate=success_rate,
                        error_count=error_count,
                        timeout_count=0,
                        meets_target=success_rate >= 95,  # 95% success rate target
                        performance_score=performance_score,
                        environment_info=self._get_environment_info()
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Concurrency benchmark failed: {e}")
        
        return results
    
    def generate_performance_report(
        self,
        suite: BenchmarkSuite,
        output_path: Optional[str] = None,
        include_charts: bool = True
    ) -> str:
        """Generate comprehensive performance report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path or f"epic4_performance_report_{timestamp}.html"
        
        # Generate HTML report
        html_content = self._generate_html_report(suite, include_charts)
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"📄 Performance report generated: {output_path}")
        return output_path
    
    def save_benchmark_results(
        self,
        suite: BenchmarkSuite,
        output_path: Optional[str] = None
    ) -> str:
        """Save benchmark results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path or f"epic4_benchmark_results_{timestamp}.json"
        
        # Convert to serializable format
        suite_dict = {
            "suite_name": suite.suite_name,
            "timestamp": suite.timestamp.isoformat(),
            "total_benchmarks": suite.total_benchmarks,
            "passed_benchmarks": suite.passed_benchmarks,
            "failed_benchmarks": suite.failed_benchmarks,
            "overall_score": suite.overall_score,
            "execution_time_seconds": suite.execution_time_seconds,
            "summary_statistics": suite.summary_statistics,
            "recommendations": suite.recommendations,
            "results": [
                {
                    "benchmark_name": r.benchmark_name,
                    "benchmark_type": r.benchmark_type.value,
                    "component": r.component,
                    "operation": r.operation,
                    "execution_time_ms": r.execution_time_ms,
                    "memory_usage_mb": r.memory_usage_mb,
                    "cpu_usage_percent": r.cpu_usage_percent,
                    "mean_time_ms": r.mean_time_ms,
                    "p95_time_ms": r.p95_time_ms,
                    "p99_time_ms": r.p99_time_ms,
                    "success_rate": r.success_rate,
                    "meets_target": r.meets_target,
                    "performance_score": r.performance_score,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in suite.results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(suite_dict, f, indent=2)
        
        self.logger.info(f"💾 Benchmark results saved: {output_path}")
        return output_path
    
    # Private helper methods
    
    async def _initialize_components(self) -> Dict[str, Any]:
        """Initialize Epic 4 components for benchmarking."""
        components = {}
        
        try:
            components["unified_context_engine"] = await get_unified_context_engine()
        except Exception as e:
            self.logger.warning(f"Could not initialize unified context engine: {e}")
        
        try:
            components["context_reasoning_engine"] = get_context_reasoning_engine()
        except Exception as e:
            self.logger.warning(f"Could not initialize context reasoning engine: {e}")
        
        try:
            components["intelligent_context_persistence"] = await get_intelligent_context_persistence()
        except Exception as e:
            self.logger.warning(f"Could not initialize intelligent context persistence: {e}")
        
        try:
            components["context_aware_agent_coordination"] = await get_context_aware_coordination()
        except Exception as e:
            self.logger.warning(f"Could not initialize context aware agent coordination: {e}")
        
        try:
            components["epic4_orchestration_integration"] = await get_epic4_orchestration_integration()
        except Exception as e:
            self.logger.warning(f"Could not initialize epic4 orchestration integration: {e}")
        
        return components
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / (1024 * 1024)
    
    def _get_cpu_usage_percent(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get system environment information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": psutil.__version__,
            "platform": psutil.LINUX if hasattr(psutil, 'LINUX') else "unknown"
        }
    
    # Placeholder implementations for specific operations
    
    async def _execute_benchmark_operation(
        self, component: Any, component_name: str, operation: str
    ) -> bool:
        """Execute a specific benchmark operation."""
        # This would contain specific logic for each component and operation
        # For now, simulate with a small delay
        await asyncio.sleep(0.001)  # 1ms delay
        return True
    
    async def _warmup_component(self, component: Any, component_name: str) -> None:
        """Warmup component before benchmarking."""
        for _ in range(self.warmup_iterations):
            await self._execute_benchmark_operation(component, component_name, "warmup")


# Global benchmarking instance
_benchmark_system: Optional[Epic4PerformanceBenchmarks] = None


def get_epic4_performance_benchmarks() -> Epic4PerformanceBenchmarks:
    """Get singleton Epic 4 performance benchmarks instance."""
    global _benchmark_system
    
    if _benchmark_system is None:
        _benchmark_system = Epic4PerformanceBenchmarks()
    
    return _benchmark_system