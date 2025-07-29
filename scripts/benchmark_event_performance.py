"""
Event Performance Benchmarking Suite for LeanVibe Agent Hive 2.0

Comprehensive performance validation ensuring <5ms hook overhead for observability events.
Tests serialization, deserialization, and event processing performance across different scenarios.
"""

import asyncio
import time
import statistics
import uuid
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import argparse
import json
from pathlib import Path

import structlog
import psutil

from app.schemas.observability import (
    PreToolUseEvent,
    PostToolUseEvent,
    WorkflowStartedEvent,
    WorkflowEndedEvent,
    NodeExecutingEvent,
    NodeCompletedEvent,
    AgentStateChangedEvent,
    SemanticQueryEvent,
    SemanticUpdateEvent,
    MessagePublishedEvent,
    SystemHealthCheckEvent,
    PerformanceMetrics,
    EventMetadata,
)
from app.core.event_serialization import (
    EventSerializer,
    SerializationFormat,
    get_high_performance_serializer,
    get_compressed_serializer,
    get_json_serializer,
)
from mock_servers.observability_events_mock import (
    MockEventGenerator,
    WorkflowScenario,
    generate_sample_events,
)

logger = structlog.get_logger()


@dataclass
class BenchmarkResult:
    """Individual benchmark result."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    std_dev_ms: float
    throughput_ops_sec: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    passed: bool = True
    error_message: Optional[str] = None


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""
    suite_name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    suite_start_time: datetime = field(default_factory=datetime.utcnow)
    suite_end_time: Optional[datetime] = None
    
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result to the suite."""
        self.results.append(result)
    
    def finalize(self):
        """Finalize the benchmark suite."""
        self.suite_end_time = datetime.utcnow()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of benchmark results."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        duration = (self.suite_end_time - self.suite_start_time).total_seconds() if self.suite_end_time else 0
        
        return {
            "suite_name": self.suite_name,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            "suite_duration_seconds": duration,
            "results": [
                {
                    "name": r.name,
                    "avg_time_ms": r.avg_time_ms,
                    "p95_time_ms": r.p95_time_ms,
                    "throughput_ops_sec": r.throughput_ops_sec,
                    "passed": r.passed,
                    "error": r.error_message
                } for r in self.results
            ]
        }


class EventPerformanceBenchmark:
    """Main performance benchmarking class."""
    
    def __init__(self):
        """Initialize the benchmark suite."""
        self.mock_generator = MockEventGenerator()
        self.high_perf_serializer = get_high_performance_serializer()
        self.compressed_serializer = get_compressed_serializer()
        self.json_serializer = get_json_serializer()
        
        # Performance thresholds
        self.SERIALIZATION_THRESHOLD_MS = 5.0
        self.DESERIALIZATION_THRESHOLD_MS = 5.0
        self.TOTAL_PROCESSING_THRESHOLD_MS = 10.0
        self.MIN_THROUGHPUT_OPS_SEC = 200.0
        
        logger.info("Event performance benchmark initialized")
    
    def _measure_time_ms(self, func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure execution time of a function in milliseconds."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, (end_time - start_time) * 1000
    
    def _get_percentiles(self, times: List[float]) -> Dict[str, float]:
        """Calculate percentiles from timing data."""
        if not times:
            return {"p50": 0, "p95": 0, "p99": 0}
        
        sorted_times = sorted(times)
        n = len(sorted_times)
        
        return {
            "p50": sorted_times[int(0.50 * n)],
            "p95": sorted_times[int(0.95 * n)],
            "p99": sorted_times[int(0.99 * n)]
        }
    
    def _get_system_metrics(self) -> Tuple[float, float]:
        """Get current system memory and CPU usage."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        cpu_percent = process.cpu_percent()
        return memory_mb, cpu_percent
    
    def benchmark_serialization(self, events: List[Dict[str, Any]], serializer: EventSerializer, iterations: int = 1000) -> BenchmarkResult:
        """Benchmark event serialization performance."""
        benchmark_name = f"Serialization ({serializer.format.value})"
        
        try:
            times = []
            serialized_sizes = []
            
            # Warmup
            for _ in range(10):
                event = events[0]
                serializer.serialize_event(event)
            
            # Measure memory before benchmark
            memory_before, _ = self._get_system_metrics()
            
            # Actual benchmark
            for i in range(iterations):
                event = events[i % len(events)]
                
                _, time_ms = self._measure_time_ms(serializer.serialize_event, event)
                times.append(time_ms)
                
                # Track serialized size for first few iterations
                if i < 10:
                    serialized_data, metadata = serializer.serialize_event(event)
                    serialized_sizes.append(metadata['serialized_size_bytes'])
            
            # Measure memory after benchmark
            memory_after, cpu_percent = self._get_system_metrics()
            
            # Calculate statistics
            total_time_ms = sum(times)
            avg_time_ms = statistics.mean(times)
            min_time_ms = min(times)
            max_time_ms = max(times)
            std_dev_ms = statistics.stdev(times) if len(times) > 1 else 0
            percentiles = self._get_percentiles(times)
            throughput_ops_sec = (iterations * 1000) / total_time_ms if total_time_ms > 0 else 0
            
            # Validate performance requirements
            passed = (
                avg_time_ms < self.SERIALIZATION_THRESHOLD_MS and
                percentiles["p95"] < self.SERIALIZATION_THRESHOLD_MS * 2 and
                throughput_ops_sec > self.MIN_THROUGHPUT_OPS_SEC
            )
            
            error_message = None
            if not passed:
                error_reasons = []
                if avg_time_ms >= self.SERIALIZATION_THRESHOLD_MS:
                    error_reasons.append(f"Average time {avg_time_ms:.2f}ms >= {self.SERIALIZATION_THRESHOLD_MS}ms")
                if percentiles["p95"] >= self.SERIALIZATION_THRESHOLD_MS * 2:
                    error_reasons.append(f"P95 time {percentiles['p95']:.2f}ms >= {self.SERIALIZATION_THRESHOLD_MS * 2}ms")
                if throughput_ops_sec <= self.MIN_THROUGHPUT_OPS_SEC:
                    error_reasons.append(f"Throughput {throughput_ops_sec:.1f} ops/sec <= {self.MIN_THROUGHPUT_OPS_SEC}")
                error_message = "; ".join(error_reasons)
            
            return BenchmarkResult(
                name=benchmark_name,
                iterations=iterations,
                total_time_ms=total_time_ms,
                avg_time_ms=avg_time_ms,
                min_time_ms=min_time_ms,
                max_time_ms=max_time_ms,
                p50_time_ms=percentiles["p50"],
                p95_time_ms=percentiles["p95"],
                p99_time_ms=percentiles["p99"],
                std_dev_ms=std_dev_ms,
                throughput_ops_sec=throughput_ops_sec,
                memory_usage_mb=memory_after - memory_before,
                cpu_usage_percent=cpu_percent,
                passed=passed,
                error_message=error_message
            )
            
        except Exception as e:
            logger.error(f"Serialization benchmark failed: {e}", exc_info=True)
            return BenchmarkResult(
                name=benchmark_name,
                iterations=0,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                p50_time_ms=0,
                p95_time_ms=0,
                p99_time_ms=0,
                std_dev_ms=0,
                throughput_ops_sec=0,
                passed=False,
                error_message=str(e)
            )
    
    def benchmark_deserialization(self, events: List[Dict[str, Any]], serializer: EventSerializer, iterations: int = 1000) -> BenchmarkResult:
        """Benchmark event deserialization performance."""
        benchmark_name = f"Deserialization ({serializer.format.value})"
        
        try:
            # Pre-serialize events
            serialized_events = []
            for event in events:
                serialized_data, metadata = serializer.serialize_event(event)
                serialized_events.append((serialized_data, metadata))
            
            times = []
            
            # Warmup
            for _ in range(10):
                data, metadata = serialized_events[0]
                serializer.deserialize_event(data, metadata)
            
            # Measure memory before benchmark
            memory_before, _ = self._get_system_metrics()
            
            # Actual benchmark
            for i in range(iterations):
                data, metadata = serialized_events[i % len(serialized_events)]
                
                _, time_ms = self._measure_time_ms(serializer.deserialize_event, data, metadata)
                times.append(time_ms)
            
            # Measure memory after benchmark
            memory_after, cpu_percent = self._get_system_metrics()
            
            # Calculate statistics
            total_time_ms = sum(times)
            avg_time_ms = statistics.mean(times)
            min_time_ms = min(times)
            max_time_ms = max(times)
            std_dev_ms = statistics.stdev(times) if len(times) > 1 else 0
            percentiles = self._get_percentiles(times)
            throughput_ops_sec = (iterations * 1000) / total_time_ms if total_time_ms > 0 else 0
            
            # Validate performance requirements
            passed = (
                avg_time_ms < self.DESERIALIZATION_THRESHOLD_MS and
                percentiles["p95"] < self.DESERIALIZATION_THRESHOLD_MS * 2 and
                throughput_ops_sec > self.MIN_THROUGHPUT_OPS_SEC
            )
            
            error_message = None
            if not passed:
                error_reasons = []
                if avg_time_ms >= self.DESERIALIZATION_THRESHOLD_MS:
                    error_reasons.append(f"Average time {avg_time_ms:.2f}ms >= {self.DESERIALIZATION_THRESHOLD_MS}ms")
                if percentiles["p95"] >= self.DESERIALIZATION_THRESHOLD_MS * 2:
                    error_reasons.append(f"P95 time {percentiles['p95']:.2f}ms >= {self.DESERIALIZATION_THRESHOLD_MS * 2}ms")
                if throughput_ops_sec <= self.MIN_THROUGHPUT_OPS_SEC:
                    error_reasons.append(f"Throughput {throughput_ops_sec:.1f} ops/sec <= {self.MIN_THROUGHPUT_OPS_SEC}")
                error_message = "; ".join(error_reasons)
            
            return BenchmarkResult(
                name=benchmark_name,
                iterations=iterations,
                total_time_ms=total_time_ms,
                avg_time_ms=avg_time_ms,
                min_time_ms=min_time_ms,
                max_time_ms=max_time_ms,
                p50_time_ms=percentiles["p50"],
                p95_time_ms=percentiles["p95"],
                p99_time_ms=percentiles["p99"],
                std_dev_ms=std_dev_ms,
                throughput_ops_sec=throughput_ops_sec,
                memory_usage_mb=memory_after - memory_before,
                cpu_usage_percent=cpu_percent,
                passed=passed,
                error_message=error_message
            )
            
        except Exception as e:
            logger.error(f"Deserialization benchmark failed: {e}", exc_info=True)
            return BenchmarkResult(
                name=benchmark_name,
                iterations=0,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                p50_time_ms=0,
                p95_time_ms=0,
                p99_time_ms=0,
                std_dev_ms=0,
                throughput_ops_sec=0,
                passed=False,
                error_message=str(e)
            )
    
    def benchmark_roundtrip(self, events: List[Dict[str, Any]], serializer: EventSerializer, iterations: int = 1000) -> BenchmarkResult:
        """Benchmark complete serialize-deserialize roundtrip performance."""
        benchmark_name = f"Roundtrip ({serializer.format.value})"
        
        try:
            times = []
            
            # Warmup
            for _ in range(10):
                event = events[0]
                serialized_data, metadata = serializer.serialize_event(event)
                serializer.deserialize_event(serialized_data, metadata)
            
            # Measure memory before benchmark
            memory_before, _ = self._get_system_metrics()
            
            # Actual benchmark
            for i in range(iterations):
                event = events[i % len(events)]
                
                def roundtrip():
                    serialized_data, metadata = serializer.serialize_event(event)
                    return serializer.deserialize_event(serialized_data, metadata)
                
                _, time_ms = self._measure_time_ms(roundtrip)
                times.append(time_ms)
            
            # Measure memory after benchmark
            memory_after, cpu_percent = self._get_system_metrics()
            
            # Calculate statistics
            total_time_ms = sum(times)
            avg_time_ms = statistics.mean(times)
            min_time_ms = min(times)
            max_time_ms = max(times)
            std_dev_ms = statistics.stdev(times) if len(times) > 1 else 0
            percentiles = self._get_percentiles(times)
            throughput_ops_sec = (iterations * 1000) / total_time_ms if total_time_ms > 0 else 0
            
            # Validate performance requirements
            passed = (
                avg_time_ms < self.TOTAL_PROCESSING_THRESHOLD_MS and
                percentiles["p95"] < self.TOTAL_PROCESSING_THRESHOLD_MS * 2 and
                throughput_ops_sec > self.MIN_THROUGHPUT_OPS_SEC / 2  # Lower threshold for roundtrip
            )
            
            error_message = None
            if not passed:
                error_reasons = []
                if avg_time_ms >= self.TOTAL_PROCESSING_THRESHOLD_MS:
                    error_reasons.append(f"Average time {avg_time_ms:.2f}ms >= {self.TOTAL_PROCESSING_THRESHOLD_MS}ms")
                if percentiles["p95"] >= self.TOTAL_PROCESSING_THRESHOLD_MS * 2:
                    error_reasons.append(f"P95 time {percentiles['p95']:.2f}ms >= {self.TOTAL_PROCESSING_THRESHOLD_MS * 2}ms")
                if throughput_ops_sec <= self.MIN_THROUGHPUT_OPS_SEC / 2:
                    error_reasons.append(f"Throughput {throughput_ops_sec:.1f} ops/sec <= {self.MIN_THROUGHPUT_OPS_SEC / 2}")
                error_message = "; ".join(error_reasons)
            
            return BenchmarkResult(
                name=benchmark_name,
                iterations=iterations,
                total_time_ms=total_time_ms,
                avg_time_ms=avg_time_ms,
                min_time_ms=min_time_ms,
                max_time_ms=max_time_ms,
                p50_time_ms=percentiles["p50"],
                p95_time_ms=percentiles["p95"],
                p99_time_ms=percentiles["p99"],
                std_dev_ms=std_dev_ms,
                throughput_ops_sec=throughput_ops_sec,
                memory_usage_mb=memory_after - memory_before,
                cpu_usage_percent=cpu_percent,
                passed=passed,
                error_message=error_message
            )
            
        except Exception as e:
            logger.error(f"Roundtrip benchmark failed: {e}", exc_info=True)
            return BenchmarkResult(
                name=benchmark_name,
                iterations=0,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                p50_time_ms=0,
                p95_time_ms=0,
                p99_time_ms=0,
                std_dev_ms=0,
                throughput_ops_sec=0,
                passed=False,
                error_message=str(e)
            )
    
    def benchmark_batch_processing(self, events: List[Dict[str, Any]], batch_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark batch processing performance."""
        results = []
        
        for batch_size in batch_sizes:
            benchmark_name = f"Batch Processing (size={batch_size})"
            
            try:
                # Create batches
                batches = []
                for i in range(0, len(events), batch_size):
                    batch = events[i:i + batch_size]
                    if len(batch) == batch_size:  # Only use full batches
                        batches.append(batch)
                
                if not batches:
                    continue
                
                times = []
                iterations = min(100, len(batches))  # Limit iterations for large batches
                
                # Warmup
                self.high_perf_serializer.serialize_batch(batches[0])
                
                # Measure memory before benchmark
                memory_before, _ = self._get_system_metrics()
                
                # Actual benchmark
                for i in range(iterations):
                    batch = batches[i % len(batches)]
                    
                    _, time_ms = self._measure_time_ms(self.high_perf_serializer.serialize_batch, batch)
                    times.append(time_ms)
                
                # Measure memory after benchmark
                memory_after, cpu_percent = self._get_system_metrics()
                
                # Calculate statistics
                total_time_ms = sum(times)
                avg_time_ms = statistics.mean(times)
                min_time_ms = min(times)
                max_time_ms = max(times)
                std_dev_ms = statistics.stdev(times) if len(times) > 1 else 0
                percentiles = self._get_percentiles(times)
                
                # Calculate throughput as events per second
                total_events_processed = iterations * batch_size
                throughput_ops_sec = (total_events_processed * 1000) / total_time_ms if total_time_ms > 0 else 0
                
                # Batch processing should be more efficient
                avg_time_per_event = avg_time_ms / batch_size
                passed = (
                    avg_time_per_event < self.SERIALIZATION_THRESHOLD_MS and
                    throughput_ops_sec > self.MIN_THROUGHPUT_OPS_SEC * 2  # Higher throughput expected
                )
                
                error_message = None
                if not passed:
                    error_reasons = []
                    if avg_time_per_event >= self.SERIALIZATION_THRESHOLD_MS:
                        error_reasons.append(f"Average time per event {avg_time_per_event:.2f}ms >= {self.SERIALIZATION_THRESHOLD_MS}ms")
                    if throughput_ops_sec <= self.MIN_THROUGHPUT_OPS_SEC * 2:
                        error_reasons.append(f"Throughput {throughput_ops_sec:.1f} ops/sec <= {self.MIN_THROUGHPUT_OPS_SEC * 2}")
                    error_message = "; ".join(error_reasons)
                
                results.append(BenchmarkResult(
                    name=benchmark_name,
                    iterations=iterations,
                    total_time_ms=total_time_ms,
                    avg_time_ms=avg_time_ms,
                    min_time_ms=min_time_ms,
                    max_time_ms=max_time_ms,
                    p50_time_ms=percentiles["p50"],
                    p95_time_ms=percentiles["p95"],
                    p99_time_ms=percentiles["p99"],
                    std_dev_ms=std_dev_ms,
                    throughput_ops_sec=throughput_ops_sec,
                    memory_usage_mb=memory_after - memory_before,
                    cpu_usage_percent=cpu_percent,
                    passed=passed,
                    error_message=error_message
                ))
                
            except Exception as e:
                logger.error(f"Batch processing benchmark failed for size {batch_size}: {e}", exc_info=True)
                results.append(BenchmarkResult(
                    name=benchmark_name,
                    iterations=0,
                    total_time_ms=0,
                    avg_time_ms=0,
                    min_time_ms=0,
                    max_time_ms=0,
                    p50_time_ms=0,
                    p95_time_ms=0,
                    p99_time_ms=0,
                    std_dev_ms=0,
                    throughput_ops_sec=0,
                    passed=False,
                    error_message=str(e)
                ))
        
        return results
    
    def benchmark_event_types(self, iterations: int = 1000) -> List[BenchmarkResult]:
        """Benchmark different event types for performance variations."""
        results = []
        
        # Different event types with varying complexity
        event_types = [
            ("PreToolUse", lambda: PreToolUseEvent(
                agent_id=uuid.uuid4(),
                session_id=uuid.uuid4(),
                tool_name="Read",
                parameters={"file_path": "/test/file.py"},
                performance_metrics=PerformanceMetrics(execution_time_ms=1.0)
            )),
            ("PostToolUse", lambda: PostToolUseEvent(
                agent_id=uuid.uuid4(),
                session_id=uuid.uuid4(),
                tool_name="Read",
                success=True,
                result="File content here...",
                performance_metrics=PerformanceMetrics(execution_time_ms=2.0)
            )),
            ("WorkflowStarted", lambda: WorkflowStartedEvent(
                workflow_id=uuid.uuid4(),
                session_id=uuid.uuid4(),
                workflow_name="Test Workflow",
                workflow_definition={"tasks": [], "dependencies": {}},
                performance_metrics=PerformanceMetrics(execution_time_ms=3.0)
            )),
            ("SemanticQuery", lambda: SemanticQueryEvent(
                session_id=uuid.uuid4(),
                query_text="Test query",
                query_embedding=[0.1] * 1536,  # Large embedding
                performance_metrics=PerformanceMetrics(execution_time_ms=50.0)
            )),
            ("SystemHealthCheck", lambda: SystemHealthCheckEvent(
                session_id=uuid.uuid4(),
                health_status="healthy",
                check_type="periodic",
                component_statuses={"redis": "healthy", "postgres": "healthy"},
                performance_indicators={"cpu": 25.0, "memory": 60.0},
                performance_metrics=PerformanceMetrics(execution_time_ms=10.0)
            ))
        ]
        
        for event_name, event_factory in event_types:
            benchmark_name = f"Event Type: {event_name}"
            
            try:
                times = []
                
                # Generate events
                events = [event_factory() for _ in range(iterations)]
                
                # Warmup
                for _ in range(10):
                    event = events[0]
                    self.high_perf_serializer.serialize_event(event)
                
                # Measure memory before benchmark
                memory_before, _ = self._get_system_metrics()
                
                # Actual benchmark - serialize + deserialize
                for i in range(iterations):
                    event = events[i]
                    
                    def process_event():
                        serialized_data, metadata = self.high_perf_serializer.serialize_event(event)
                        return self.high_perf_serializer.deserialize_event(serialized_data, metadata)
                    
                    _, time_ms = self._measure_time_ms(process_event)
                    times.append(time_ms)
                
                # Measure memory after benchmark
                memory_after, cpu_percent = self._get_system_metrics()
                
                # Calculate statistics
                total_time_ms = sum(times)
                avg_time_ms = statistics.mean(times)
                min_time_ms = min(times)
                max_time_ms = max(times)
                std_dev_ms = statistics.stdev(times) if len(times) > 1 else 0
                percentiles = self._get_percentiles(times)
                throughput_ops_sec = (iterations * 1000) / total_time_ms if total_time_ms > 0 else 0
                
                # Validate performance requirements
                passed = (
                    avg_time_ms < self.TOTAL_PROCESSING_THRESHOLD_MS and
                    percentiles["p95"] < self.TOTAL_PROCESSING_THRESHOLD_MS * 2
                )
                
                error_message = None
                if not passed:
                    error_reasons = []
                    if avg_time_ms >= self.TOTAL_PROCESSING_THRESHOLD_MS:
                        error_reasons.append(f"Average time {avg_time_ms:.2f}ms >= {self.TOTAL_PROCESSING_THRESHOLD_MS}ms")
                    if percentiles["p95"] >= self.TOTAL_PROCESSING_THRESHOLD_MS * 2:
                        error_reasons.append(f"P95 time {percentiles['p95']:.2f}ms >= {self.TOTAL_PROCESSING_THRESHOLD_MS * 2}ms")
                    error_message = "; ".join(error_reasons)
                
                results.append(BenchmarkResult(
                    name=benchmark_name,
                    iterations=iterations,
                    total_time_ms=total_time_ms,
                    avg_time_ms=avg_time_ms,
                    min_time_ms=min_time_ms,
                    max_time_ms=max_time_ms,
                    p50_time_ms=percentiles["p50"],
                    p95_time_ms=percentiles["p95"],
                    p99_time_ms=percentiles["p99"],
                    std_dev_ms=std_dev_ms,
                    throughput_ops_sec=throughput_ops_sec,
                    memory_usage_mb=memory_after - memory_before,
                    cpu_usage_percent=cpu_percent,
                    passed=passed,
                    error_message=error_message
                ))
                
            except Exception as e:
                logger.error(f"Event type benchmark failed for {event_name}: {e}", exc_info=True)
                results.append(BenchmarkResult(
                    name=benchmark_name,
                    iterations=0,
                    total_time_ms=0,
                    avg_time_ms=0,
                    min_time_ms=0,
                    max_time_ms=0,
                    p50_time_ms=0,
                    p95_time_ms=0,
                    p99_time_ms=0,
                    std_dev_ms=0,
                    throughput_ops_sec=0,
                    passed=False,
                    error_message=str(e)
                ))
        
        return results
    
    def run_comprehensive_benchmark(self, iterations: int = 1000) -> BenchmarkSuite:
        """Run comprehensive performance benchmark suite."""
        suite = BenchmarkSuite("Comprehensive Event Performance Benchmark")
        
        logger.info(f"Starting comprehensive benchmark with {iterations} iterations")
        
        # Generate test events
        sample_events = generate_sample_events(count=100)
        
        # Test different serialization formats
        serializers = [
            ("High Performance", self.high_perf_serializer),
            ("Compressed", self.compressed_serializer),
            ("JSON", self.json_serializer)
        ]
        
        for serializer_name, serializer in serializers:
            logger.info(f"Benchmarking {serializer_name} serializer")
            
            # Serialization benchmark
            result = self.benchmark_serialization(sample_events, serializer, iterations)
            suite.add_result(result)
            
            # Deserialization benchmark
            result = self.benchmark_deserialization(sample_events, serializer, iterations)
            suite.add_result(result)
            
            # Roundtrip benchmark
            result = self.benchmark_roundtrip(sample_events, serializer, iterations)
            suite.add_result(result)
        
        # Batch processing benchmarks
        logger.info("Benchmarking batch processing")
        batch_results = self.benchmark_batch_processing(sample_events, [1, 5, 10, 25, 50])
        for result in batch_results:
            suite.add_result(result)
        
        # Event type benchmarks
        logger.info("Benchmarking different event types")
        event_type_results = self.benchmark_event_types(iterations // 2)  # Reduce iterations for complex events
        for result in event_type_results:
            suite.add_result(result)
        
        suite.finalize()
        
        logger.info("Comprehensive benchmark completed")
        return suite
    
    def print_results(self, suite: BenchmarkSuite):
        """Print benchmark results in a formatted table."""
        print("\n" + "=" * 120)
        print(f"BENCHMARK RESULTS: {suite.suite_name}")
        print("=" * 120)
        
        # Header
        print(f"{'Test Name':<40} {'Iterations':<10} {'Avg (ms)':<10} {'P95 (ms)':<10} {'P99 (ms)':<10} {'Ops/Sec':<12} {'Status':<8}")
        print("-" * 120)
        
        # Results
        for result in suite.results:
            status = "PASS" if result.passed else "FAIL"
            status_color = "\033[92m" if result.passed else "\033[91m"  # Green for pass, red for fail
            reset_color = "\033[0m"
            
            print(f"{result.name:<40} {result.iterations:<10} {result.avg_time_ms:<10.2f} {result.p95_time_ms:<10.2f} "
                  f"{result.p99_time_ms:<10.2f} {result.throughput_ops_sec:<12.1f} {status_color}{status:<8}{reset_color}")
            
            if not result.passed and result.error_message:
                print(f"    Error: {result.error_message}")
        
        # Summary
        summary = suite.get_summary()
        print("\n" + "=" * 120)
        print("SUMMARY")
        print("-" * 120)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Suite Duration: {summary['suite_duration_seconds']:.2f} seconds")
        print("=" * 120)
        
        # Performance thresholds
        print("\nPERFORMANCE THRESHOLDS:")
        print(f"- Serialization: < {self.SERIALIZATION_THRESHOLD_MS} ms")
        print(f"- Deserialization: < {self.DESERIALIZATION_THRESHOLD_MS} ms")
        print(f"- Total Processing: < {self.TOTAL_PROCESSING_THRESHOLD_MS} ms")
        print(f"- Minimum Throughput: > {self.MIN_THROUGHPUT_OPS_SEC} ops/sec")
    
    def save_results(self, suite: BenchmarkSuite, output_file: str):
        """Save benchmark results to JSON file."""
        summary = suite.get_summary()
        
        # Add detailed results
        detailed_results = []
        for result in suite.results:
            detailed_results.append({
                "name": result.name,
                "iterations": result.iterations,
                "total_time_ms": result.total_time_ms,
                "avg_time_ms": result.avg_time_ms,
                "min_time_ms": result.min_time_ms,
                "max_time_ms": result.max_time_ms,
                "p50_time_ms": result.p50_time_ms,
                "p95_time_ms": result.p95_time_ms,
                "p99_time_ms": result.p99_time_ms,
                "std_dev_ms": result.std_dev_ms,
                "throughput_ops_sec": result.throughput_ops_sec,
                "memory_usage_mb": result.memory_usage_mb,
                "cpu_usage_percent": result.cpu_usage_percent,
                "passed": result.passed,
                "error_message": result.error_message
            })
        
        summary["detailed_results"] = detailed_results
        summary["performance_thresholds"] = {
            "serialization_threshold_ms": self.SERIALIZATION_THRESHOLD_MS,
            "deserialization_threshold_ms": self.DESERIALIZATION_THRESHOLD_MS,
            "total_processing_threshold_ms": self.TOTAL_PROCESSING_THRESHOLD_MS,
            "min_throughput_ops_sec": self.MIN_THROUGHPUT_OPS_SEC
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {output_file}")


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="Event Performance Benchmark Suite")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations per test")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file for results")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    if not args.quiet:
        # Configure logging
        import logging
        logging.basicConfig(level=logging.INFO)
    
    # Run benchmark
    benchmark = EventPerformanceBenchmark()
    suite = benchmark.run_comprehensive_benchmark(iterations=args.iterations)
    
    # Print results
    if not args.quiet:
        benchmark.print_results(suite)
    
    # Save results
    benchmark.save_results(suite, args.output)
    
    # Exit with appropriate code
    summary = suite.get_summary()
    exit_code = 0 if summary['failed_tests'] == 0 else 1
    
    if not args.quiet:
        if exit_code == 0:
            print("\n✅ All performance benchmarks PASSED")
        else:
            print(f"\n❌ {summary['failed_tests']} performance benchmarks FAILED")
    
    exit(exit_code)


if __name__ == "__main__":
    main()