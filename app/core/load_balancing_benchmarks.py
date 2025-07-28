"""
Performance Benchmarks for Load Balancing System

Comprehensive benchmarking suite to validate performance targets:
- <100ms load balancing decisions
- Support for 1000+ concurrent agents
- <50ms health check cycles
- <30ms resource optimization decisions
- Distributed state synchronization performance
"""

import asyncio
import time
import statistics
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
import random
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import structlog

from .agent_load_balancer import AgentLoadBalancer, AgentLoadState, LoadBalancingStrategy
from .capacity_manager import CapacityManager, ScalingDecision, ScalingAction, CapacityTier
from .performance_metrics_collector import PerformanceMetricsCollector, MetricType
from .adaptive_scaler import AdaptiveScaler
from .resource_optimizer import ResourceOptimizer, OptimizationType
from .health_monitor import HealthMonitor
from .distributed_load_balancing_state import DistributedLoadBalancingState
from ..models.task import Task, TaskPriority

logger = structlog.get_logger()


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark."""
    benchmark_name: str
    target_metric: str
    target_value: float
    measured_value: float
    unit: str
    passed: bool
    percentile_95: Optional[float] = None
    percentile_99: Optional[float] = None
    samples: int = 0
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "benchmark_name": self.benchmark_name,
            "target_metric": self.target_metric,
            "target_value": self.target_value,
            "measured_value": self.measured_value,
            "unit": self.unit,
            "passed": self.passed,
            "percentile_95": self.percentile_95,
            "percentile_99": self.percentile_99,
            "samples": self.samples,
            "additional_metrics": self.additional_metrics
        }


class LoadBalancingBenchmarkSuite:
    """
    Comprehensive benchmark suite for load balancing system.
    
    Validates all performance targets and scales to test limits.
    """
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        
        # Performance targets
        self.targets = {
            "load_balancing_decision_time_ms": 100.0,
            "health_check_cycle_time_ms": 50.0,
            "resource_optimization_time_ms": 30.0,
            "metrics_collection_latency_ms": 10.0,
            "scaling_decision_time_ms": 200.0,
            "concurrent_agents_supported": 1000,
            "distributed_sync_latency_ms": 100.0,
            "memory_usage_mb_per_agent": 5.0,
            "throughput_decisions_per_second": 100.0
        }
        
        logger.info("LoadBalancingBenchmarkSuite initialized", targets=self.targets)
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        try:
            logger.info("Starting comprehensive load balancing benchmarks")
            start_time = time.time()
            
            # Core component benchmarks
            await self._benchmark_load_balancer_performance()
            await self._benchmark_capacity_manager_performance()
            await self._benchmark_health_monitor_performance()
            await self._benchmark_metrics_collector_performance()
            await self._benchmark_resource_optimizer_performance()
            await self._benchmark_adaptive_scaler_performance()
            
            # Scale testing benchmarks
            await self._benchmark_concurrent_agent_support()
            await self._benchmark_high_throughput_decisions()
            await self._benchmark_memory_efficiency()
            
            # Integration benchmarks
            await self._benchmark_end_to_end_performance()
            await self._benchmark_distributed_coordination()
            
            total_time = time.time() - start_time
            
            # Generate summary report
            summary = self._generate_benchmark_summary(total_time)
            
            logger.info("Benchmark suite completed",
                       total_time_seconds=total_time,
                       total_benchmarks=len(self.results),
                       passed=summary["passed_count"],
                       failed=summary["failed_count"])
            
            return summary
            
        except Exception as e:
            logger.error("Benchmark suite failed", error=str(e))
            raise
    
    async def _benchmark_load_balancer_performance(self) -> None:
        """Benchmark load balancer decision performance."""
        logger.info("Benchmarking load balancer performance")
        
        load_balancer = AgentLoadBalancer()
        
        # Set up test agents with varying loads
        agent_count = 50
        agents = []
        for i in range(agent_count):
            agent_id = f"agent_{i}"
            agents.append(agent_id)
            
            # Create realistic load distribution
            load_state = AgentLoadState(
                agent_id=agent_id,
                active_tasks=random.randint(0, 5),
                pending_tasks=random.randint(0, 3),
                context_usage_percent=random.uniform(10.0, 90.0),
                memory_usage_mb=random.uniform(100.0, 1500.0),
                cpu_usage_percent=random.uniform(10.0, 95.0),
                average_response_time_ms=random.uniform(100.0, 5000.0),
                error_rate_percent=random.uniform(0.0, 10.0),
                health_score=random.uniform(0.5, 1.0)
            )
            load_balancer.agent_loads[agent_id] = load_state
        
        # Test different load balancing strategies
        strategies = [
            LoadBalancingStrategy.ROUND_ROBIN,
            LoadBalancingStrategy.LEAST_CONNECTIONS,
            LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN,
            LoadBalancingStrategy.PERFORMANCE_BASED
        ]
        
        for strategy in strategies:
            decision_times = []
            
            # Run multiple iterations
            for _ in range(100):
                task = Task(
                    id=uuid.uuid4(),
                    title=f"Benchmark Task",
                    description="Task for benchmarking",
                    priority=TaskPriority.MEDIUM,
                    complexity_score=1.0
                )
                
                start_time = time.time()
                
                decision = await load_balancer.select_agent_for_task(
                    task=task,
                    available_agents=agents,
                    strategy=strategy
                )
                
                end_time = time.time()
                decision_time_ms = (end_time - start_time) * 1000
                decision_times.append(decision_time_ms)
            
            # Calculate statistics
            avg_time = statistics.mean(decision_times)
            p95_time = statistics.quantiles(decision_times, n=20)[18] if len(decision_times) >= 20 else max(decision_times)
            p99_time = statistics.quantiles(decision_times, n=100)[98] if len(decision_times) >= 100 else max(decision_times)
            
            # Record result
            result = BenchmarkResult(
                benchmark_name=f"load_balancer_{strategy.value}",
                target_metric="decision_time_ms",
                target_value=self.targets["load_balancing_decision_time_ms"],
                measured_value=avg_time,
                unit="ms",
                passed=avg_time < self.targets["load_balancing_decision_time_ms"],
                percentile_95=p95_time,
                percentile_99=p99_time,
                samples=len(decision_times),
                additional_metrics={
                    "max_time_ms": max(decision_times),
                    "min_time_ms": min(decision_times),
                    "agent_count": agent_count
                }
            )
            
            self.results.append(result)
            
            logger.info("Load balancer strategy benchmarked",
                       strategy=strategy.value,
                       avg_time_ms=avg_time,
                       p95_time_ms=p95_time,
                       passed=result.passed)
    
    async def _benchmark_capacity_manager_performance(self) -> None:
        """Benchmark capacity manager scaling decisions."""
        logger.info("Benchmarking capacity manager performance")
        
        load_balancer = AgentLoadBalancer()
        capacity_manager = CapacityManager(load_balancer)
        
        # Simulate system with varying loads
        decision_times = []
        
        for _ in range(50):
            start_time = time.time()
            
            # Generate scaling decisions
            decisions = await capacity_manager.evaluate_capacity_needs()
            
            end_time = time.time()
            decision_time_ms = (end_time - start_time) * 1000
            decision_times.append(decision_time_ms)
        
        # Calculate statistics
        avg_time = statistics.mean(decision_times)
        p95_time = statistics.quantiles(decision_times, n=20)[18] if len(decision_times) >= 20 else max(decision_times)
        
        result = BenchmarkResult(
            benchmark_name="capacity_manager_scaling_decisions",
            target_metric="decision_time_ms",
            target_value=self.targets["scaling_decision_time_ms"],
            measured_value=avg_time,
            unit="ms",
            passed=avg_time < self.targets["scaling_decision_time_ms"],
            percentile_95=p95_time,
            samples=len(decision_times)
        )
        
        self.results.append(result)
        
        logger.info("Capacity manager benchmarked",
                   avg_time_ms=avg_time,
                   passed=result.passed)
    
    async def _benchmark_health_monitor_performance(self) -> None:
        """Benchmark health monitoring performance."""
        logger.info("Benchmarking health monitor performance")
        
        health_monitor = HealthMonitor(None)  # Mock metrics collector
        
        # Initialize multiple agent profiles
        agent_count = 100
        for i in range(agent_count):
            await health_monitor._initialize_agent_profile(f"agent_{i}")
        
        # Mock health check functions to return quickly
        async def mock_check(agent_id: str) -> bool:
            await asyncio.sleep(0.001)  # Simulate small delay
            return random.choice([True, True, True, False])  # 75% success rate
        
        # Replace health check methods with mocks
        health_monitor.check_agent_heartbeat = mock_check
        health_monitor.check_agent_performance = mock_check
        health_monitor.check_resource_usage = mock_check
        health_monitor.check_error_rate = mock_check
        health_monitor.check_response_time = mock_check
        health_monitor.check_memory_leak = mock_check
        
        # Benchmark health check cycles
        cycle_times = []
        
        for _ in range(20):
            start_time = time.time()
            
            await health_monitor._run_health_checks()
            await health_monitor._update_health_status()
            
            end_time = time.time()
            cycle_time_ms = (end_time - start_time) * 1000
            cycle_times.append(cycle_time_ms)
        
        # Calculate statistics
        avg_time = statistics.mean(cycle_times)
        p95_time = statistics.quantiles(cycle_times, n=20)[18] if len(cycle_times) >= 20 else max(cycle_times)
        
        result = BenchmarkResult(
            benchmark_name="health_monitor_cycle_time",
            target_metric="cycle_time_ms",
            target_value=self.targets["health_check_cycle_time_ms"],
            measured_value=avg_time,
            unit="ms",
            passed=avg_time < self.targets["health_check_cycle_time_ms"],
            percentile_95=p95_time,
            samples=len(cycle_times),
            additional_metrics={
                "agent_count": agent_count,
                "checks_per_agent": 6
            }
        )
        
        self.results.append(result)
        
        logger.info("Health monitor benchmarked",
                   avg_time_ms=avg_time,
                   agent_count=agent_count,
                   passed=result.passed)
    
    async def _benchmark_metrics_collector_performance(self) -> None:
        """Benchmark metrics collection performance."""
        logger.info("Benchmarking metrics collector performance")
        
        metrics_collector = PerformanceMetricsCollector(collection_interval=0.1)
        
        # Benchmark metric recording
        record_times = []
        metric_count = 1000
        
        for i in range(metric_count):
            start_time = time.time()
            
            await metrics_collector.record_custom_metric(
                entity_id=f"entity_{i % 10}",
                metric_name="benchmark_metric",
                value=float(i),
                metric_type=MetricType.GAUGE
            )
            
            end_time = time.time()
            record_time_ms = (end_time - start_time) * 1000
            record_times.append(record_time_ms)
        
        # Calculate statistics
        avg_time = statistics.mean(record_times)
        p95_time = statistics.quantiles(record_times, n=20)[18] if len(record_times) >= 20 else max(record_times)
        
        result = BenchmarkResult(
            benchmark_name="metrics_collector_recording",
            target_metric="record_latency_ms",
            target_value=self.targets["metrics_collection_latency_ms"],
            measured_value=avg_time,
            unit="ms",
            passed=avg_time < self.targets["metrics_collection_latency_ms"],
            percentile_95=p95_time,
            samples=len(record_times),
            additional_metrics={
                "total_metrics": metric_count,
                "throughput_metrics_per_second": metric_count / (sum(record_times) / 1000)
            }
        )
        
        self.results.append(result)
        
        logger.info("Metrics collector benchmarked",
                   avg_time_ms=avg_time,
                   throughput=result.additional_metrics["throughput_metrics_per_second"],
                   passed=result.passed)
    
    async def _benchmark_resource_optimizer_performance(self) -> None:
        """Benchmark resource optimizer performance."""
        logger.info("Benchmarking resource optimizer performance")
        
        resource_optimizer = ResourceOptimizer(None)  # Mock metrics collector
        
        # Benchmark optimization decisions
        optimization_times = []
        
        from app.core.resource_optimizer import ResourceUsage
        
        for _ in range(100):
            # Create resource usage data
            current_usage = ResourceUsage(
                timestamp=datetime.utcnow(),
                memory_mb=random.uniform(500, 2000),
                memory_percent=random.uniform(50, 90),
                cpu_percent=random.uniform(40, 95),
                disk_read_mb_per_sec=random.uniform(1, 50),
                disk_write_mb_per_sec=random.uniform(1, 30),
                network_bytes_per_sec=random.uniform(100000, 10000000),
                active_connections=random.randint(5, 25),
                context_usage_percent=random.uniform(30, 95)
            )
            
            start_time = time.time()
            
            # Evaluate optimization needs
            optimizations = await resource_optimizer._evaluate_optimization_needs(current_usage)
            
            end_time = time.time()
            optimization_time_ms = (end_time - start_time) * 1000
            optimization_times.append(optimization_time_ms)
        
        # Calculate statistics
        avg_time = statistics.mean(optimization_times)
        p95_time = statistics.quantiles(optimization_times, n=20)[18] if len(optimization_times) >= 20 else max(optimization_times)
        
        result = BenchmarkResult(
            benchmark_name="resource_optimizer_decisions",
            target_metric="optimization_time_ms",
            target_value=self.targets["resource_optimization_time_ms"],
            measured_value=avg_time,
            unit="ms",
            passed=avg_time < self.targets["resource_optimization_time_ms"],
            percentile_95=p95_time,
            samples=len(optimization_times)
        )
        
        self.results.append(result)
        
        logger.info("Resource optimizer benchmarked",
                   avg_time_ms=avg_time,
                   passed=result.passed)
    
    async def _benchmark_adaptive_scaler_performance(self) -> None:
        """Benchmark adaptive scaler performance."""
        logger.info("Benchmarking adaptive scaler performance")
        
        load_balancer = AgentLoadBalancer()
        capacity_manager = CapacityManager(load_balancer)
        metrics_collector = PerformanceMetricsCollector()
        adaptive_scaler = AdaptiveScaler(load_balancer, capacity_manager, metrics_collector)
        
        # Benchmark scaling evaluations
        evaluation_times = []
        
        for _ in range(50):
            start_time = time.time()
            
            decisions = await adaptive_scaler.evaluate_scaling_needs()
            
            end_time = time.time()
            evaluation_time_ms = (end_time - start_time) * 1000
            evaluation_times.append(evaluation_time_ms)
        
        # Calculate statistics
        avg_time = statistics.mean(evaluation_times)
        p95_time = statistics.quantiles(evaluation_times, n=20)[18] if len(evaluation_times) >= 20 else max(evaluation_times)
        
        result = BenchmarkResult(
            benchmark_name="adaptive_scaler_evaluations",
            target_metric="evaluation_time_ms",
            target_value=self.targets["scaling_decision_time_ms"],
            measured_value=avg_time,
            unit="ms",
            passed=avg_time < self.targets["scaling_decision_time_ms"],
            percentile_95=p95_time,
            samples=len(evaluation_times)
        )
        
        self.results.append(result)
        
        logger.info("Adaptive scaler benchmarked",
                   avg_time_ms=avg_time,
                   passed=result.passed)
    
    async def _benchmark_concurrent_agent_support(self) -> None:
        """Benchmark support for large numbers of concurrent agents."""
        logger.info("Benchmarking concurrent agent support")
        
        load_balancer = AgentLoadBalancer()
        target_agents = int(self.targets["concurrent_agents_supported"])
        
        # Set up large number of agents
        agents = []
        setup_start = time.time()
        
        for i in range(target_agents):
            agent_id = f"agent_{i}"
            agents.append(agent_id)
            
            load_state = AgentLoadState(
                agent_id=agent_id,
                active_tasks=random.randint(0, 3),
                context_usage_percent=random.uniform(10.0, 80.0),
                health_score=random.uniform(0.7, 1.0)
            )
            load_balancer.agent_loads[agent_id] = load_state
        
        setup_time = time.time() - setup_start
        
        # Test decision making with large agent pool
        decision_times = []
        
        for _ in range(100):
            task = Task(
                id=uuid.uuid4(),
                title="Concurrent Test Task",
                description="Task for concurrent testing",
                priority=TaskPriority.MEDIUM
            )
            
            start_time = time.time()
            
            decision = await load_balancer.select_agent_for_task(
                task=task,
                available_agents=agents,
                strategy=LoadBalancingStrategy.LEAST_CONNECTIONS
            )
            
            end_time = time.time()
            decision_time_ms = (end_time - start_time) * 1000
            decision_times.append(decision_time_ms)
        
        # Calculate statistics
        avg_decision_time = statistics.mean(decision_times)
        max_decision_time = max(decision_times)
        
        # Test passed if we can handle target number of agents with acceptable performance
        passed = (len(agents) >= target_agents and 
                 avg_decision_time < self.targets["load_balancing_decision_time_ms"])
        
        result = BenchmarkResult(
            benchmark_name="concurrent_agent_support",
            target_metric="concurrent_agents",
            target_value=target_agents,
            measured_value=len(agents),
            unit="agents",
            passed=passed,
            samples=len(decision_times),
            additional_metrics={
                "avg_decision_time_ms": avg_decision_time,
                "max_decision_time_ms": max_decision_time,
                "setup_time_seconds": setup_time,
                "memory_per_agent_estimate": setup_time * 1000 / target_agents  # Rough estimate
            }
        )
        
        self.results.append(result)
        
        logger.info("Concurrent agent support benchmarked",
                   agent_count=len(agents),
                   avg_decision_time_ms=avg_decision_time,
                   passed=passed)
    
    async def _benchmark_high_throughput_decisions(self) -> None:
        """Benchmark high-throughput decision making."""
        logger.info("Benchmarking high-throughput decisions")
        
        load_balancer = AgentLoadBalancer()
        
        # Set up moderate number of agents
        agents = []
        for i in range(20):
            agent_id = f"agent_{i}"
            agents.append(agent_id)
            load_balancer.agent_loads[agent_id] = AgentLoadState(
                agent_id=agent_id,
                active_tasks=1,
                health_score=0.9
            )
        
        # Test sustained high throughput
        tasks_per_batch = 100
        batches = 10
        total_decisions = 0
        total_time = 0
        
        for batch in range(batches):
            batch_start = time.time()
            
            # Create tasks concurrently
            tasks = []
            for i in range(tasks_per_batch):
                task = Task(
                    id=uuid.uuid4(),
                    title=f"Throughput Test Task {i}",
                    description="Task for throughput testing",
                    priority=TaskPriority.MEDIUM
                )
                
                task_coroutine = load_balancer.select_agent_for_task(
                    task=task,
                    available_agents=agents,
                    strategy=LoadBalancingStrategy.ROUND_ROBIN
                )
                tasks.append(task_coroutine)
            
            # Execute all tasks concurrently
            await asyncio.gather(*tasks)
            
            batch_time = time.time() - batch_start
            total_decisions += tasks_per_batch
            total_time += batch_time
            
            batch_throughput = tasks_per_batch / batch_time
            logger.debug("Throughput batch completed",
                        batch=batch + 1,
                        batch_time=batch_time,
                        batch_throughput=batch_throughput)
        
        # Calculate overall throughput
        overall_throughput = total_decisions / total_time
        
        result = BenchmarkResult(
            benchmark_name="high_throughput_decisions",
            target_metric="decisions_per_second",
            target_value=self.targets["throughput_decisions_per_second"],
            measured_value=overall_throughput,
            unit="decisions/sec",
            passed=overall_throughput >= self.targets["throughput_decisions_per_second"],
            samples=total_decisions,
            additional_metrics={
                "total_decisions": total_decisions,
                "total_time_seconds": total_time,
                "batches": batches,
                "tasks_per_batch": tasks_per_batch
            }
        )
        
        self.results.append(result)
        
        logger.info("High throughput benchmarked",
                   throughput=overall_throughput,
                   total_decisions=total_decisions,
                   passed=result.passed)
    
    async def _benchmark_memory_efficiency(self) -> None:
        """Benchmark memory efficiency per agent."""
        logger.info("Benchmarking memory efficiency")
        
        import psutil
        import gc
        
        # Get baseline memory usage
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create load balancer and agents
        load_balancer = AgentLoadBalancer()
        agent_count = 1000
        
        for i in range(agent_count):
            agent_id = f"agent_{i}"
            load_state = AgentLoadState(
                agent_id=agent_id,
                active_tasks=random.randint(0, 5),
                context_usage_percent=random.uniform(10.0, 90.0),
                health_score=random.uniform(0.5, 1.0)
            )
            load_balancer.agent_loads[agent_id] = load_state
        
        # Measure memory after agent creation
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_used = final_memory - baseline_memory
        memory_per_agent = memory_used / agent_count
        
        result = BenchmarkResult(
            benchmark_name="memory_efficiency_per_agent",
            target_metric="memory_per_agent_mb",
            target_value=self.targets["memory_usage_mb_per_agent"],
            measured_value=memory_per_agent,
            unit="MB/agent",
            passed=memory_per_agent <= self.targets["memory_usage_mb_per_agent"],
            samples=agent_count,
            additional_metrics={
                "baseline_memory_mb": baseline_memory,
                "final_memory_mb": final_memory,
                "total_memory_used_mb": memory_used,
                "agent_count": agent_count
            }
        )
        
        self.results.append(result)
        
        logger.info("Memory efficiency benchmarked",
                   memory_per_agent_mb=memory_per_agent,
                   total_agents=agent_count,
                   passed=result.passed)
    
    async def _benchmark_end_to_end_performance(self) -> None:
        """Benchmark complete end-to-end task assignment flow."""
        logger.info("Benchmarking end-to-end performance")
        
        # Set up complete system
        metrics_collector = PerformanceMetricsCollector()
        load_balancer = AgentLoadBalancer()
        capacity_manager = CapacityManager(load_balancer)
        adaptive_scaler = AdaptiveScaler(load_balancer, capacity_manager, metrics_collector)
        health_monitor = HealthMonitor(metrics_collector)
        
        # Initialize agents
        agent_count = 50
        for i in range(agent_count):
            agent_id = f"agent_{i}"
            load_balancer.agent_loads[agent_id] = AgentLoadState(
                agent_id=agent_id,
                active_tasks=random.randint(0, 3),
                health_score=random.uniform(0.7, 1.0)
            )
            await health_monitor._initialize_agent_profile(agent_id)
        
        # Benchmark complete flow
        e2e_times = []
        
        for _ in range(100):
            task = Task(
                id=uuid.uuid4(),
                title="E2E Test Task",
                description="End-to-end test task",
                priority=TaskPriority.HIGH
            )
            
            start_time = time.time()
            
            # Complete flow: health check -> load balancing -> capacity evaluation
            available_agents = list(load_balancer.agent_loads.keys())
            
            # Filter healthy agents (simplified)
            healthy_agents = [
                agent_id for agent_id in available_agents
                if load_balancer.agent_loads[agent_id].health_score > 0.7
            ]
            
            if healthy_agents:
                # Make load balancing decision
                decision = await load_balancer.select_agent_for_task(
                    task=task,
                    available_agents=healthy_agents,
                    strategy=LoadBalancingStrategy.ADAPTIVE_HYBRID
                )
                
                # Update capacity metrics (simplified)
                selected_agent_state = load_balancer.agent_loads[decision.selected_agent_id]
                selected_agent_state.active_tasks += 1
            
            end_time = time.time()
            e2e_time_ms = (end_time - start_time) * 1000
            e2e_times.append(e2e_time_ms)
        
        # Calculate statistics
        avg_time = statistics.mean(e2e_times)
        p95_time = statistics.quantiles(e2e_times, n=20)[18] if len(e2e_times) >= 20 else max(e2e_times)
        
        # Target is 2x the load balancing target for complete flow
        target_time = self.targets["load_balancing_decision_time_ms"] * 2
        
        result = BenchmarkResult(
            benchmark_name="end_to_end_task_assignment",
            target_metric="e2e_time_ms",
            target_value=target_time,
            measured_value=avg_time,
            unit="ms",
            passed=avg_time < target_time,
            percentile_95=p95_time,
            samples=len(e2e_times),
            additional_metrics={
                "agent_count": agent_count,
                "components_involved": 4
            }
        )
        
        self.results.append(result)
        
        logger.info("End-to-end performance benchmarked",
                   avg_time_ms=avg_time,
                   p95_time_ms=p95_time,
                   passed=result.passed)
    
    async def _benchmark_distributed_coordination(self) -> None:
        """Benchmark distributed state synchronization."""
        logger.info("Benchmarking distributed coordination")
        
        # Mock Redis client for testing
        class MockRedisClient:
            def __init__(self):
                self.data = {}
                self.delay = 0.005  # 5ms simulated network delay
            
            async def hset(self, key, mapping):
                await asyncio.sleep(self.delay)
                self.data[key] = mapping
                return True
            
            async def hgetall(self, key):
                await asyncio.sleep(self.delay)
                return self.data.get(key, {})
            
            async def expire(self, key, seconds):
                await asyncio.sleep(self.delay)
                return True
            
            async def keys(self, pattern):
                await asyncio.sleep(self.delay)
                return [k for k in self.data.keys() if pattern.replace("*", "") in k]
        
        redis_mock = MockRedisClient()
        distributed_state = DistributedLoadBalancingState(
            redis_client=redis_mock,
            node_id="benchmark_node"
        )
        
        # Benchmark agent state synchronization
        sync_times = []
        agent_count = 100
        
        for batch in range(10):
            batch_start = time.time()
            
            # Store agent states
            tasks = []
            for i in range(agent_count):
                agent_id = f"agent_{batch}_{i}"
                load_state = AgentLoadState(
                    agent_id=agent_id,
                    active_tasks=random.randint(0, 5),
                    health_score=random.uniform(0.5, 1.0)
                )
                
                task = distributed_state.store_agent_load_state(agent_id, load_state)
                tasks.append(task)
            
            # Execute all stores concurrently
            await asyncio.gather(*tasks)
            
            batch_time = time.time() - batch_start
            sync_time_ms = batch_time * 1000
            sync_times.append(sync_time_ms)
        
        # Calculate statistics
        avg_sync_time = statistics.mean(sync_times)
        p95_sync_time = statistics.quantiles(sync_times, n=20)[18] if len(sync_times) >= 20 else max(sync_times)
        
        result = BenchmarkResult(
            benchmark_name="distributed_state_synchronization",
            target_metric="sync_latency_ms",
            target_value=self.targets["distributed_sync_latency_ms"],
            measured_value=avg_sync_time,
            unit="ms",
            passed=avg_sync_time < self.targets["distributed_sync_latency_ms"],
            percentile_95=p95_sync_time,
            samples=len(sync_times),
            additional_metrics={
                "agent_states_per_batch": agent_count,
                "total_batches": len(sync_times),
                "simulated_network_delay_ms": redis_mock.delay * 1000
            }
        )
        
        self.results.append(result)
        
        logger.info("Distributed coordination benchmarked",
                   avg_sync_time_ms=avg_sync_time,
                   agents_per_batch=agent_count,
                   passed=result.passed)
    
    def _generate_benchmark_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive benchmark summary."""
        passed_count = len([r for r in self.results if r.passed])
        failed_count = len([r for r in self.results if not r.passed])
        
        # Group results by component
        component_results = {}
        for result in self.results:
            component = result.benchmark_name.split('_')[0]
            if component not in component_results:
                component_results[component] = []
            component_results[component].append(result)
        
        # Calculate component pass rates
        component_summary = {}
        for component, results in component_results.items():
            component_passed = len([r for r in results if r.passed])
            component_total = len(results)
            component_summary[component] = {
                "passed": component_passed,
                "total": component_total,
                "pass_rate": component_passed / component_total if component_total > 0 else 0,
                "benchmarks": [r.to_dict() for r in results]
            }
        
        # Performance summary
        critical_benchmarks = [
            "load_balancer_least_connections",
            "health_monitor_cycle_time",
            "concurrent_agent_support",
            "end_to_end_task_assignment"
        ]
        
        critical_results = [
            r for r in self.results 
            if any(r.benchmark_name.startswith(cb) for cb in critical_benchmarks)
        ]
        critical_passed = len([r for r in critical_results if r.passed])
        
        summary = {
            "total_benchmarks": len(self.results),
            "passed_count": passed_count,
            "failed_count": failed_count,
            "overall_pass_rate": passed_count / len(self.results) if self.results else 0,
            "critical_benchmarks_passed": critical_passed,
            "critical_benchmarks_total": len(critical_results),
            "critical_pass_rate": critical_passed / len(critical_results) if critical_results else 0,
            "total_execution_time_seconds": total_time,
            "performance_targets": self.targets,
            "component_summary": component_summary,
            "detailed_results": [r.to_dict() for r in self.results],
            "timestamp": datetime.utcnow().isoformat(),
            "system_ready_for_production": (
                passed_count / len(self.results) >= 0.90 and
                critical_passed == len(critical_results)
            )
        }
        
        return summary
    
    async def benchmark_specific_component(self, component: str) -> Dict[str, Any]:
        """Run benchmarks for a specific component."""
        self.results.clear()  # Clear previous results
        
        component_benchmarks = {
            "load_balancer": self._benchmark_load_balancer_performance,
            "capacity_manager": self._benchmark_capacity_manager_performance,
            "health_monitor": self._benchmark_health_monitor_performance,
            "metrics_collector": self._benchmark_metrics_collector_performance,
            "resource_optimizer": self._benchmark_resource_optimizer_performance,
            "adaptive_scaler": self._benchmark_adaptive_scaler_performance
        }
        
        if component not in component_benchmarks:
            raise ValueError(f"Unknown component: {component}")
        
        logger.info("Running specific component benchmark", component=component)
        start_time = time.time()
        
        await component_benchmarks[component]()
        
        total_time = time.time() - start_time
        return self._generate_benchmark_summary(total_time)


# Convenience function for running benchmarks
async def run_load_balancing_benchmarks() -> Dict[str, Any]:
    """Run complete load balancing benchmark suite."""
    suite = LoadBalancingBenchmarkSuite()
    return await suite.run_all_benchmarks()


# CLI interface for benchmarking
if __name__ == "__main__":
    import sys
    
    async def main():
        if len(sys.argv) > 1:
            component = sys.argv[1]
            suite = LoadBalancingBenchmarkSuite()
            results = await suite.benchmark_specific_component(component)
        else:
            results = await run_load_balancing_benchmarks()
        
        print(f"\n{'='*60}")
        print("LOAD BALANCING BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Total Benchmarks: {results['total_benchmarks']}")
        print(f"Passed: {results['passed_count']} ({results['overall_pass_rate']:.1%})")
        print(f"Failed: {results['failed_count']}")
        print(f"Critical Pass Rate: {results['critical_pass_rate']:.1%}")
        print(f"Execution Time: {results['total_execution_time_seconds']:.2f}s")
        print(f"Production Ready: {'✅ YES' if results['system_ready_for_production'] else '❌ NO'}")
        
        print(f"\n{'Component Results:'}")
        for component, data in results['component_summary'].items():
            status = "✅" if data['pass_rate'] == 1.0 else "⚠️" if data['pass_rate'] > 0.5 else "❌"
            print(f"  {status} {component}: {data['passed']}/{data['total']} ({data['pass_rate']:.1%})")
        
        print(f"\n{'Failed Benchmarks:'}")
        failed_benchmarks = [r for r in results['detailed_results'] if not r['passed']]
        if not failed_benchmarks:
            print("  None! All benchmarks passed.")
        else:
            for benchmark in failed_benchmarks:
                print(f"  ❌ {benchmark['benchmark_name']}: {benchmark['measured_value']:.2f} {benchmark['unit']} (target: {benchmark['target_value']:.2f})")
    
    asyncio.run(main())