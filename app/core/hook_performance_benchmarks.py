"""
Performance Benchmarks for Hook Lifecycle System.

This module provides comprehensive performance testing and benchmarking
for the Hook Lifecycle System, validating the <50ms processing requirement
and identifying performance bottlenecks.

Features:
- Comprehensive performance testing suite
- Load testing with varying hook volumes
- Security validation performance benchmarks
- Event aggregation performance analysis
- WebSocket streaming performance tests
- Redis integration performance validation
- Detailed performance reporting and analysis
"""

import asyncio
import time
import uuid
import statistics
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import random

import structlog

from .hook_lifecycle_system import (
    HookLifecycleSystem,
    HookType,
    HookEvent,
    SecurityValidator,
    EventAggregator,
    WebSocketStreamer,
    get_hook_lifecycle_system
)

logger = structlog.get_logger()


@dataclass
class PerformanceTest:
    """Configuration for a performance test."""
    name: str
    description: str
    test_function: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_max_time_ms: float = 50.0
    iterations: int = 1000
    concurrent_requests: int = 10


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    test_name: str
    iterations: int
    total_time_seconds: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    success_rate: float
    throughput_per_second: float
    meets_sla: bool  # Does it meet <50ms SLA
    errors: List[str] = field(default_factory=list)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


class HookPerformanceBenchmarks:
    """
    Comprehensive performance benchmarking suite for Hook Lifecycle System.
    
    Provides detailed performance analysis and validation against
    performance requirements (<50ms processing time).
    """
    
    def __init__(self):
        self.hook_system: Optional[HookLifecycleSystem] = None
        self.results: List[BenchmarkResult] = []
        
        # Test configuration
        self.performance_tests = [
            PerformanceTest(
                name="basic_hook_processing",
                description="Basic hook processing performance",
                test_function="test_basic_hook_processing",
                expected_max_time_ms=20.0,
                iterations=5000,
                concurrent_requests=1
            ),
            PerformanceTest(
                name="concurrent_hook_processing",
                description="Concurrent hook processing performance",
                test_function="test_concurrent_hook_processing",
                expected_max_time_ms=50.0,
                iterations=1000,
                concurrent_requests=20
            ),
            PerformanceTest(
                name="security_validation_performance",
                description="Security validator performance",
                test_function="test_security_validation_performance",
                expected_max_time_ms=10.0,
                iterations=10000,
                concurrent_requests=1
            ),
            PerformanceTest(
                name="event_aggregation_performance",
                description="Event aggregation performance",
                test_function="test_event_aggregation_performance",
                expected_max_time_ms=30.0,
                iterations=2000,
                concurrent_requests=5
            ),
            PerformanceTest(
                name="websocket_streaming_performance",
                description="WebSocket streaming performance",
                test_function="test_websocket_streaming_performance",
                expected_max_time_ms=25.0,
                iterations=1000,
                concurrent_requests=10
            ),
            PerformanceTest(
                name="redis_integration_performance",
                description="Redis integration performance",
                test_function="test_redis_integration_performance",
                expected_max_time_ms=40.0,
                iterations=1000,
                concurrent_requests=5
            ),
            PerformanceTest(
                name="high_volume_load_test",
                description="High volume load testing",
                test_function="test_high_volume_load",
                expected_max_time_ms=50.0,
                iterations=10000,
                concurrent_requests=50
            ),
            PerformanceTest(
                name="mixed_workload_performance",
                description="Mixed workload performance test",
                test_function="test_mixed_workload_performance",
                expected_max_time_ms=45.0,
                iterations=2000,
                concurrent_requests=15
            )
        ]
        
        # Performance metrics
        self.metrics = {
            "tests_executed": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "total_benchmark_time": 0.0,
            "sla_violations": 0,
            "performance_degradations": []
        }
    
    async def initialize(self) -> None:
        """Initialize the performance benchmark suite."""
        try:
            # Initialize hook lifecycle system
            self.hook_system = await get_hook_lifecycle_system()
            
            logger.info("🚀 Hook Performance Benchmarks initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance benchmarks: {e}")
            raise
    
    async def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all performance benchmarks and return results."""
        benchmark_start_time = time.time()
        
        logger.info(f"🏁 Starting performance benchmark suite with {len(self.performance_tests)} tests")
        
        for test in self.performance_tests:
            try:
                logger.info(f"▶️ Running benchmark: {test.name}")
                result = await self._run_benchmark(test)
                self.results.append(result)
                
                # Log result summary
                logger.info(
                    f"✅ Benchmark completed: {test.name}",
                    avg_time_ms=result.avg_time_ms,
                    meets_sla=result.meets_sla,
                    success_rate=result.success_rate,
                    throughput_per_second=result.throughput_per_second
                )
                
                self.metrics["tests_executed"] += 1
                if result.meets_sla and result.success_rate > 0.95:
                    self.metrics["tests_passed"] += 1
                else:
                    self.metrics["tests_failed"] += 1
                    if not result.meets_sla:
                        self.metrics["sla_violations"] += 1
                
            except Exception as e:
                logger.error(f"❌ Benchmark failed: {test.name}, error: {e}")
                self.metrics["tests_failed"] += 1
        
        self.metrics["total_benchmark_time"] = time.time() - benchmark_start_time
        
        # Generate summary report
        await self._generate_performance_report()
        
        return self.results
    
    async def _run_benchmark(self, test: PerformanceTest) -> BenchmarkResult:
        """Run a single performance benchmark."""
        test_method = getattr(self, test.test_function)
        
        # Warm-up run
        await self._warmup_test(test_method, test.parameters)
        
        # Collect timing data
        times: List[float] = []
        errors: List[str] = []
        start_time = time.time()
        
        if test.concurrent_requests > 1:
            # Concurrent execution
            semaphore = asyncio.Semaphore(test.concurrent_requests)
            
            async def run_single_test():
                async with semaphore:
                    try:
                        single_start = time.time()
                        await test_method(**test.parameters)
                        execution_time = (time.time() - single_start) * 1000
                        times.append(execution_time)
                    except Exception as e:
                        errors.append(str(e))
            
            # Execute concurrent tests
            tasks = [run_single_test() for _ in range(test.iterations)]
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Sequential execution
            for _ in range(test.iterations):
                try:
                    single_start = time.time()
                    await test_method(**test.parameters)
                    execution_time = (time.time() - single_start) * 1000
                    times.append(execution_time)
                except Exception as e:
                    errors.append(str(e))
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        if times:
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            median_time = statistics.median(times)
            p95_time = self._percentile(times, 95)
            p99_time = self._percentile(times, 99)
        else:
            avg_time = min_time = max_time = median_time = p95_time = p99_time = 0.0
        
        success_rate = len(times) / test.iterations if test.iterations > 0 else 0.0
        throughput = len(times) / total_time if total_time > 0 else 0.0
        meets_sla = avg_time <= test.expected_max_time_ms and p95_time <= test.expected_max_time_ms * 1.5
        
        return BenchmarkResult(
            test_name=test.name,
            iterations=test.iterations,
            total_time_seconds=total_time,
            avg_time_ms=avg_time,
            min_time_ms=min_time,
            max_time_ms=max_time,
            median_time_ms=median_time,
            p95_time_ms=p95_time,
            p99_time_ms=p99_time,
            success_rate=success_rate,
            throughput_per_second=throughput,
            meets_sla=meets_sla,
            errors=errors[:10]  # Keep first 10 errors
        )
    
    async def _warmup_test(self, test_method, parameters: Dict[str, Any]) -> None:
        """Perform warmup runs for more accurate benchmarking."""
        for _ in range(10):  # 10 warmup iterations
            try:
                await test_method(**parameters)
            except Exception:
                pass  # Ignore warmup errors
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        lower_index = int(index)
        upper_index = min(lower_index + 1, len(sorted_data) - 1)
        weight = index - lower_index
        return sorted_data[lower_index] * (1 - weight) + sorted_data[upper_index] * weight
    
    # Performance test methods
    
    async def test_basic_hook_processing(self) -> None:
        """Test basic hook processing performance."""
        if not self.hook_system:
            raise RuntimeError("Hook system not initialized")
        
        agent_id = uuid.uuid4()
        session_id = uuid.uuid4()
        
        result = await self.hook_system.process_hook(
            hook_type=HookType.PRE_TOOL_USE,
            agent_id=agent_id,
            session_id=session_id,
            payload={
                "tool_name": "test_tool",
                "parameters": {"param1": "value1", "param2": 123}
            },
            priority=5
        )
        
        if not result.success:
            raise RuntimeError(f"Hook processing failed: {result.error}")
    
    async def test_concurrent_hook_processing(self) -> None:
        """Test concurrent hook processing performance."""
        if not self.hook_system:
            raise RuntimeError("Hook system not initialized")
        
        agent_id = uuid.uuid4()
        session_id = uuid.uuid4()
        
        # Process multiple hook types concurrently
        tasks = [
            self.hook_system.process_pre_tool_use(
                agent_id=agent_id,
                session_id=session_id,
                tool_name="concurrent_tool",
                parameters={"iteration": i}
            )
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        for result in results:
            if not result.success:
                raise RuntimeError(f"Concurrent hook processing failed: {result.error}")
    
    async def test_security_validation_performance(self) -> None:
        """Test security validation performance."""
        if not self.hook_system:
            raise RuntimeError("Hook system not initialized")
        
        validator = self.hook_system.security_validator
        
        # Test various command types
        commands = [
            "ls -la /home/user",
            "python script.py",
            "git commit -m 'test'",
            "docker run nginx",
            "npm install package"
        ]
        
        command = random.choice(commands)
        is_safe, risk_level, reason = await validator.validate_command(
            command=command,
            context={"test": True}
        )
        
        # Result validation not required for performance test
    
    async def test_event_aggregation_performance(self) -> None:
        """Test event aggregation performance."""
        if not self.hook_system:
            raise RuntimeError("Hook system not initialized")
        
        aggregator = self.hook_system.event_aggregator
        
        # Create test event
        event = HookEvent(
            hook_type=HookType.POST_TOOL_USE,
            agent_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            timestamp=datetime.utcnow(),
            payload={
                "tool_name": "aggregation_test",
                "success": True,
                "result": "test_result"
            },
            priority=5
        )
        
        await aggregator.add_event(event)
    
    async def test_websocket_streaming_performance(self) -> None:
        """Test WebSocket streaming performance."""
        if not self.hook_system:
            raise RuntimeError("Hook system not initialized")
        
        streamer = self.hook_system.websocket_streamer
        
        # Create test event
        event = HookEvent(
            hook_type=HookType.NOTIFICATION,
            agent_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            timestamp=datetime.utcnow(),
            payload={
                "level": "info",
                "message": "WebSocket streaming test",
                "details": {"test": True}
            },
            priority=5
        )
        
        # Broadcast event (will be no-op if no clients connected)
        await streamer.broadcast_event(event)
    
    async def test_redis_integration_performance(self) -> None:
        """Test Redis integration performance."""
        if not self.hook_system:
            raise RuntimeError("Hook system not initialized")
        
        agent_id = uuid.uuid4()
        session_id = uuid.uuid4()
        
        # Process hook with Redis streaming enabled
        result = await self.hook_system.process_hook(
            hook_type=HookType.PRE_TOOL_USE,
            agent_id=agent_id,
            session_id=session_id,
            payload={
                "tool_name": "redis_test",
                "parameters": {"redis_enabled": True}
            },
            priority=5
        )
        
        if not result.success:
            raise RuntimeError(f"Redis integration test failed: {result.error}")
    
    async def test_high_volume_load(self) -> None:
        """Test high volume load performance."""
        # This is called for each iteration, so just process one hook
        await self.test_basic_hook_processing()
    
    async def test_mixed_workload_performance(self) -> None:
        """Test mixed workload performance."""
        if not self.hook_system:
            raise RuntimeError("Hook system not initialized")
        
        agent_id = uuid.uuid4()
        session_id = uuid.uuid4()
        
        # Randomly select operation type
        operations = [
            ("pre_tool_use", lambda: self.hook_system.process_pre_tool_use(
                agent_id=agent_id,
                session_id=session_id,
                tool_name="mixed_test",
                parameters={"type": "pre"}
            )),
            ("post_tool_use", lambda: self.hook_system.process_post_tool_use(
                agent_id=agent_id,
                session_id=session_id,
                tool_name="mixed_test",
                success=True,
                result="mixed_result"
            )),
            ("notification", lambda: self.hook_system.process_notification(
                agent_id=agent_id,
                session_id=session_id,
                level="info",
                message="Mixed workload test"
            )),
            ("stop", lambda: self.hook_system.process_stop(
                agent_id=agent_id,
                session_id=session_id,
                reason="Mixed workload test stop"
            ))
        ]
        
        operation_name, operation_func = random.choice(operations)
        result = await operation_func()
        
        if not result.success:
            raise RuntimeError(f"Mixed workload test failed ({operation_name}): {result.error}")
    
    async def _generate_performance_report(self) -> None:
        """Generate comprehensive performance report."""
        if not self.results:
            logger.warning("No benchmark results available for report generation")
            return
        
        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.meets_sla and r.success_rate > 0.95)
        sla_violations = sum(1 for r in self.results if not r.meets_sla)
        
        avg_performance = statistics.mean([r.avg_time_ms for r in self.results])
        p95_performance = statistics.mean([r.p95_time_ms for r in self.results])
        overall_throughput = sum([r.throughput_per_second for r in self.results])
        
        # Generate detailed report
        report = {
            "benchmark_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "sla_violations": sla_violations,
                "overall_success_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
                "benchmark_duration_seconds": self.metrics["total_benchmark_time"]
            },
            "performance_summary": {
                "average_processing_time_ms": round(avg_performance, 2),
                "p95_processing_time_ms": round(p95_performance, 2),
                "meets_50ms_sla": avg_performance <= 50.0 and p95_performance <= 75.0,
                "total_throughput_per_second": round(overall_throughput, 2)
            },
            "detailed_results": [
                {
                    "test_name": r.test_name,
                    "avg_time_ms": round(r.avg_time_ms, 2),
                    "p95_time_ms": round(r.p95_time_ms, 2),
                    "p99_time_ms": round(r.p99_time_ms, 2),
                    "success_rate": round(r.success_rate, 4),
                    "throughput_per_second": round(r.throughput_per_second, 2),
                    "meets_sla": r.meets_sla,
                    "iterations": r.iterations,
                    "error_count": len(r.errors)
                }
                for r in self.results
            ],
            "sla_analysis": {
                "target_avg_time_ms": 50.0,
                "target_p95_time_ms": 75.0,
                "violations": [
                    {
                        "test_name": r.test_name,
                        "avg_time_ms": round(r.avg_time_ms, 2),
                        "p95_time_ms": round(r.p95_time_ms, 2),
                        "violation_type": "avg" if r.avg_time_ms > 50.0 else "p95"
                    }
                    for r in self.results if not r.meets_sla
                ]
            },
            "recommendations": self._generate_performance_recommendations()
        }
        
        # Log summary
        logger.info(
            "📊 Performance Benchmark Report Generated",
            total_tests=total_tests,
            passed_tests=passed_tests,
            avg_performance_ms=round(avg_performance, 2),
            meets_sla=report["performance_summary"]["meets_50ms_sla"],
            total_throughput=round(overall_throughput, 2)
        )
        
        # Save detailed report
        with open(f"/tmp/hook_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze results for recommendations
        slow_tests = [r for r in self.results if r.avg_time_ms > 40.0]
        high_error_tests = [r for r in self.results if r.success_rate < 0.95]
        low_throughput_tests = [r for r in self.results if r.throughput_per_second < 100.0]
        
        if slow_tests:
            recommendations.append(
                f"Optimize performance for slow tests: {', '.join([t.test_name for t in slow_tests])}"
            )
        
        if high_error_tests:
            recommendations.append(
                f"Investigate error rates for: {', '.join([t.test_name for t in high_error_tests])}"
            )
        
        if low_throughput_tests:
            recommendations.append(
                f"Improve throughput for: {', '.join([t.test_name for t in low_throughput_tests])}"
            )
        
        # General recommendations
        avg_performance = statistics.mean([r.avg_time_ms for r in self.results])
        if avg_performance > 30.0:
            recommendations.append("Consider caching optimizations to improve overall performance")
        
        if any(r.p99_time_ms > 100.0 for r in self.results):
            recommendations.append("Investigate performance outliers causing high P99 latency")
        
        overall_success_rate = statistics.mean([r.success_rate for r in self.results])
        if overall_success_rate < 0.98:
            recommendations.append("Improve error handling and system reliability")
        
        return recommendations
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get benchmark metrics."""
        return {
            "performance_benchmarks": self.metrics.copy(),
            "results_summary": {
                "total_results": len(self.results),
                "sla_compliant": sum(1 for r in self.results if r.meets_sla),
                "avg_processing_time_ms": statistics.mean([r.avg_time_ms for r in self.results]) if self.results else 0.0,
                "total_throughput": sum([r.throughput_per_second for r in self.results])
            }
        }


# Global benchmark instance
_performance_benchmarks: Optional[HookPerformanceBenchmarks] = None


def get_hook_performance_benchmarks() -> HookPerformanceBenchmarks:
    """Get global hook performance benchmarks instance."""
    global _performance_benchmarks
    
    if _performance_benchmarks is None:
        _performance_benchmarks = HookPerformanceBenchmarks()
    
    return _performance_benchmarks


async def run_performance_validation() -> bool:
    """
    Run performance validation and return True if all SLAs are met.
    
    This is the main entry point for validating the <50ms processing requirement.
    """
    benchmarks = get_hook_performance_benchmarks()
    await benchmarks.initialize()
    
    results = await benchmarks.run_all_benchmarks()
    
    # Check if performance requirements are met
    avg_performance = statistics.mean([r.avg_time_ms for r in results])
    p95_performance = statistics.mean([r.p95_time_ms for r in results])
    success_rate = statistics.mean([r.success_rate for r in results])
    
    meets_requirements = (
        avg_performance <= 50.0 and
        p95_performance <= 75.0 and
        success_rate >= 0.95
    )
    
    logger.info(
        "🎯 Performance Validation Complete",
        avg_time_ms=round(avg_performance, 2),
        p95_time_ms=round(p95_performance, 2),
        success_rate=round(success_rate, 4),
        meets_requirements=meets_requirements
    )
    
    return meets_requirements


# Example usage and testing
async def main():
    """Example usage of the performance benchmarks."""
    try:
        # Run performance validation
        meets_requirements = await run_performance_validation()
        
        if meets_requirements:
            print("✅ All performance requirements met!")
        else:
            print("❌ Performance requirements not met")
        
        # Get detailed metrics
        benchmarks = get_hook_performance_benchmarks()
        metrics = benchmarks.get_metrics()
        print(f"📊 Benchmark metrics: {json.dumps(metrics, indent=2)}")
        
    except Exception as e:
        logger.error(f"Performance benchmark error: {e}")


if __name__ == "__main__":
    asyncio.run(main())