"""
Test Execution Optimization Framework

Implements strategies to reduce test execution time from 45min to <15min
and reduce flaky test rate from 3% to <2%.
"""

import asyncio
import pytest
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any, Callable
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock, patch
import statistics

from app.core.unified_production_orchestrator import (
    UnifiedProductionOrchestrator,
    OrchestratorConfig
)


@dataclass
class TestExecutionMetrics:
    """Metrics for test execution optimization."""
    test_name: str
    execution_time_seconds: float
    setup_time_seconds: float
    teardown_time_seconds: float
    cpu_usage_percent: float
    memory_usage_mb: float
    success: bool
    flaky_score: float = 0.0  # 0.0 = stable, 1.0 = always flaky


class TestOptimizationFramework:
    """Framework for optimizing test execution."""
    
    def __init__(self):
        self.metrics: List[TestExecutionMetrics] = []
        self.optimization_strategies = {
            'parallel_execution': True,
            'shared_fixtures': True,
            'mock_optimization': True,
            'resource_pooling': True,
            'smart_ordering': True
        }
    
    def record_test_execution(self, metrics: TestExecutionMetrics):
        """Record test execution metrics."""
        self.metrics.append(metrics)
    
    def analyze_execution_patterns(self) -> Dict[str, Any]:
        """Analyze test execution patterns for optimization."""
        if not self.metrics:
            return {}
        
        total_time = sum(m.execution_time_seconds for m in self.metrics)
        avg_time = statistics.mean(m.execution_time_seconds for m in self.metrics)
        max_time = max(m.execution_time_seconds for m in self.metrics)
        min_time = min(m.execution_time_seconds for m in self.metrics)
        
        setup_time = sum(m.setup_time_seconds for m in self.metrics)
        teardown_time = sum(m.teardown_time_seconds for m in self.metrics)
        
        success_rate = sum(1 for m in self.metrics if m.success) / len(self.metrics)
        avg_flaky_score = statistics.mean(m.flaky_score for m in self.metrics)
        
        # Identify slow tests (top 20%)
        sorted_by_time = sorted(self.metrics, key=lambda m: m.execution_time_seconds, reverse=True)
        slow_test_count = max(1, len(sorted_by_time) // 5)
        slow_tests = sorted_by_time[:slow_test_count]
        
        return {
            'total_execution_time_seconds': total_time,
            'average_execution_time_seconds': avg_time,
            'max_execution_time_seconds': max_time,
            'min_execution_time_seconds': min_time,
            'setup_overhead_seconds': setup_time,
            'teardown_overhead_seconds': teardown_time,
            'success_rate': success_rate,
            'average_flaky_score': avg_flaky_score,
            'slow_tests': [
                {
                    'name': test.test_name,
                    'time': test.execution_time_seconds,
                    'flaky_score': test.flaky_score
                } for test in slow_tests
            ],
            'optimization_potential_seconds': self._calculate_optimization_potential()
        }
    
    def _calculate_optimization_potential(self) -> float:
        """Calculate potential time savings from optimization."""
        if not self.metrics:
            return 0.0
        
        # Calculate potential savings from parallelization
        total_time = sum(m.execution_time_seconds for m in self.metrics)
        parallel_potential = total_time * 0.6  # Assume 60% can be parallelized
        
        # Calculate setup/teardown optimization potential
        total_setup = sum(m.setup_time_seconds for m in self.metrics)
        total_teardown = sum(m.teardown_time_seconds for m in self.metrics)
        fixture_optimization = (total_setup + total_teardown) * 0.8  # 80% reduction with shared fixtures
        
        return parallel_potential + fixture_optimization
    
    def generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        analysis = self.analyze_execution_patterns()
        recommendations = []
        
        if analysis.get('total_execution_time_seconds', 0) > 900:  # > 15 minutes
            recommendations.append("Implement parallel test execution with pytest-xdist")
        
        if analysis.get('setup_overhead_seconds', 0) > 60:  # > 1 minute
            recommendations.append("Optimize fixture setup with shared session fixtures")
        
        if analysis.get('average_flaky_score', 0) > 0.02:  # > 2% flaky rate
            recommendations.append("Implement deterministic test isolation and retry policies")
        
        slow_tests = analysis.get('slow_tests', [])
        if len(slow_tests) > 0:
            slowest = slow_tests[0]
            if slowest['time'] > 30:  # > 30 seconds
                recommendations.append(f"Optimize slow test: {slowest['name']} ({slowest['time']:.1f}s)")
        
        return recommendations


class OptimizedTestRunner:
    """Optimized test runner with parallel execution and smart caching."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.shared_fixtures = {}
        self.test_cache = {}
        
    async def run_tests_parallel(self, test_functions: List[Callable]) -> List[TestExecutionMetrics]:
        """Run tests in parallel with optimized resource sharing."""
        # Group tests by fixture requirements
        test_groups = self._group_tests_by_fixtures(test_functions)
        
        # Run each group in parallel
        all_metrics = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for group_name, tests in test_groups.items():
                # Create shared fixture for the group
                shared_fixture = await self._create_shared_fixture(group_name)
                
                # Submit tests in the group
                for test_func in tests:
                    future = executor.submit(self._run_single_test, test_func, shared_fixture)
                    futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    metrics = future.result(timeout=30)  # 30 second timeout per test
                    all_metrics.append(metrics)
                except Exception as e:
                    # Record failed test
                    metrics = TestExecutionMetrics(
                        test_name="unknown_test",
                        execution_time_seconds=30.0,
                        setup_time_seconds=0.0,
                        teardown_time_seconds=0.0,
                        cpu_usage_percent=0.0,
                        memory_usage_mb=0.0,
                        success=False,
                        flaky_score=1.0
                    )
                    all_metrics.append(metrics)
        
        return all_metrics
    
    def _group_tests_by_fixtures(self, test_functions: List[Callable]) -> Dict[str, List[Callable]]:
        """Group tests by their fixture requirements."""
        groups = {
            'orchestrator_tests': [],
            'integration_tests': [],
            'unit_tests': [],
            'contract_tests': []
        }
        
        for test_func in test_functions:
            test_name = getattr(test_func, '__name__', 'unknown')
            
            if 'orchestrator' in test_name.lower():
                groups['orchestrator_tests'].append(test_func)
            elif 'integration' in test_name.lower():
                groups['integration_tests'].append(test_func)
            elif 'contract' in test_name.lower():
                groups['contract_tests'].append(test_func)
            else:
                groups['unit_tests'].append(test_func)
        
        return {k: v for k, v in groups.items() if v}  # Remove empty groups
    
    async def _create_shared_fixture(self, group_name: str) -> Any:
        """Create shared fixture for test group."""
        if group_name in self.shared_fixtures:
            return self.shared_fixtures[group_name]
        
        if group_name == 'orchestrator_tests':
            # Create shared orchestrator
            config = OrchestratorConfig(
                max_concurrent_agents=5,
                registration_target_ms=25.0,
                delegation_target_ms=100.0
            )
            
            with patch('app.core.unified_production_orchestrator.get_redis') as mock_redis, \
                 patch('app.core.unified_production_orchestrator.get_session') as mock_db:
                
                mock_redis.return_value = AsyncMock()
                mock_db.return_value.__aenter__ = AsyncMock()
                mock_db.return_value.__aexit__ = AsyncMock()
                
                orchestrator = UnifiedProductionOrchestrator(config)
                await orchestrator.start()
                
                self.shared_fixtures[group_name] = orchestrator
                return orchestrator
        
        elif group_name in ['integration_tests', 'contract_tests']:
            # Create lightweight shared mocks
            shared_mocks = {
                'redis': AsyncMock(),
                'database': AsyncMock(),
                'message_broker': AsyncMock()
            }
            self.shared_fixtures[group_name] = shared_mocks
            return shared_mocks
        
        else:
            # Unit tests don't need shared fixtures
            self.shared_fixtures[group_name] = None
            return None
    
    def _run_single_test(self, test_func: Callable, shared_fixture: Any) -> TestExecutionMetrics:
        """Run a single test with metrics collection."""
        import psutil
        
        test_name = getattr(test_func, '__name__', 'unknown')
        process = psutil.Process()
        
        # Setup phase
        setup_start = time.time()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        setup_end = time.time()
        setup_time = setup_end - setup_start
        
        # Execution phase
        exec_start = time.time()
        success = False
        
        try:
            # Run the test function
            if asyncio.iscoroutinefunction(test_func):
                # For async tests
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(test_func(shared_fixture))
                    success = True
                finally:
                    loop.close()
            else:
                # For sync tests
                test_func(shared_fixture)
                success = True
                
        except Exception as e:
            success = False
        
        exec_end = time.time()
        execution_time = exec_end - exec_start
        
        # Teardown phase
        teardown_start = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        teardown_end = time.time()
        teardown_time = teardown_end - teardown_start
        
        # Calculate metrics
        cpu_usage = process.cpu_percent()
        memory_usage = final_memory - initial_memory
        
        # Calculate flaky score (simplified - would need historical data in real implementation)
        flaky_score = 0.0 if success else 0.1
        
        return TestExecutionMetrics(
            test_name=test_name,
            execution_time_seconds=execution_time,
            setup_time_seconds=setup_time,
            teardown_time_seconds=teardown_time,
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory_usage,
            success=success,
            flaky_score=flaky_score
        )


class FlakyTestDetector:
    """Detector for flaky tests and stability improvements."""
    
    def __init__(self):
        self.test_history: Dict[str, List[bool]] = {}  # test_name -> [success_results]
        self.stability_threshold = 0.98  # 98% success rate required
    
    def record_test_result(self, test_name: str, success: bool):
        """Record test result for flaky detection."""
        if test_name not in self.test_history:
            self.test_history[test_name] = []
        
        self.test_history[test_name].append(success)
        
        # Keep only last 100 runs
        if len(self.test_history[test_name]) > 100:
            self.test_history[test_name] = self.test_history[test_name][-100:]
    
    def calculate_stability_score(self, test_name: str) -> float:
        """Calculate stability score for a test."""
        if test_name not in self.test_history:
            return 1.0  # New test assumed stable
        
        results = self.test_history[test_name]
        if not results:
            return 1.0
        
        return sum(results) / len(results)
    
    def identify_flaky_tests(self) -> List[Dict[str, Any]]:
        """Identify tests that are below stability threshold."""
        flaky_tests = []
        
        for test_name, results in self.test_history.items():
            if len(results) >= 5:  # Need at least 5 runs to assess
                stability = self.calculate_stability_score(test_name)
                if stability < self.stability_threshold:
                    flaky_tests.append({
                        'test_name': test_name,
                        'stability_score': stability,
                        'total_runs': len(results),
                        'failures': len(results) - sum(results),
                        'recommended_action': self._get_stabilization_recommendation(stability)
                    })
        
        return sorted(flaky_tests, key=lambda x: x['stability_score'])
    
    def _get_stabilization_recommendation(self, stability_score: float) -> str:
        """Get recommendation for stabilizing flaky test."""
        if stability_score < 0.7:
            return "Critical: Rewrite test with better isolation and deterministic setup"
        elif stability_score < 0.9:
            return "High: Add retry logic and improve test cleanup"
        else:
            return "Medium: Review timing dependencies and async operations"


# Optimization Strategies
class TestOptimizationStrategies:
    """Collection of test optimization strategies."""
    
    @staticmethod
    def implement_smart_test_ordering(test_metrics: List[TestExecutionMetrics]) -> List[str]:
        """Order tests to minimize overall execution time."""
        # Sort by execution time (fastest first for quick feedback)
        sorted_tests = sorted(test_metrics, key=lambda x: x.execution_time_seconds)
        
        # But prioritize stable tests over flaky ones
        stable_tests = [t for t in sorted_tests if t.flaky_score < 0.02]
        flaky_tests = [t for t in sorted_tests if t.flaky_score >= 0.02]
        
        # Return test names in optimal order
        return [t.test_name for t in stable_tests] + [t.test_name for t in flaky_tests]
    
    @staticmethod
    def calculate_optimal_parallelism(test_metrics: List[TestExecutionMetrics]) -> Dict[str, int]:
        """Calculate optimal parallelism settings."""
        total_tests = len(test_metrics)
        total_time = sum(m.execution_time_seconds for m in test_metrics)
        avg_time = total_time / total_tests if total_tests > 0 else 0
        
        # Calculate based on system resources
        cpu_count = multiprocessing.cpu_count()
        
        if avg_time < 1.0:  # Fast tests
            workers = min(cpu_count * 2, total_tests)
        elif avg_time < 5.0:  # Medium tests
            workers = min(cpu_count, total_tests)
        else:  # Slow tests
            workers = min(cpu_count // 2, total_tests)
        
        return {
            'recommended_workers': max(1, workers),
            'estimated_speedup': min(workers, total_tests),
            'estimated_time_seconds': total_time / min(workers, total_tests)
        }
    
    @staticmethod
    def identify_fixture_optimization_opportunities(test_metrics: List[TestExecutionMetrics]) -> List[str]:
        """Identify opportunities for fixture optimization."""
        opportunities = []
        
        total_setup_time = sum(m.setup_time_seconds for m in test_metrics)
        total_teardown_time = sum(m.teardown_time_seconds for m in test_metrics)
        
        if total_setup_time > 30:  # > 30 seconds total setup
            opportunities.append(f"High setup overhead ({total_setup_time:.1f}s) - implement session-scoped fixtures")
        
        if total_teardown_time > 15:  # > 15 seconds total teardown
            opportunities.append(f"High teardown overhead ({total_teardown_time:.1f}s) - optimize cleanup procedures")
        
        # Look for tests with similar setup patterns
        high_setup_tests = [m for m in test_metrics if m.setup_time_seconds > 2.0]
        if len(high_setup_tests) > 3:
            opportunities.append(f"{len(high_setup_tests)} tests with slow setup - consider shared fixtures")
        
        return opportunities


# Test the optimization framework
@pytest.fixture
def optimization_framework():
    """Optimization framework for testing."""
    return TestOptimizationFramework()


@pytest.fixture
def optimized_runner():
    """Optimized test runner for testing."""
    return OptimizedTestRunner(max_workers=4)


@pytest.fixture
def flaky_detector():
    """Flaky test detector for testing."""
    return FlakyTestDetector()


class TestOptimizationFrameworkValidation:
    """Tests for the optimization framework itself."""
    
    def test_metrics_collection(self, optimization_framework):
        """Test metrics collection functionality."""
        # Create sample metrics
        metrics = TestExecutionMetrics(
            test_name="test_sample",
            execution_time_seconds=1.5,
            setup_time_seconds=0.2,
            teardown_time_seconds=0.1,
            cpu_usage_percent=25.0,
            memory_usage_mb=50.0,
            success=True,
            flaky_score=0.0
        )
        
        optimization_framework.record_test_execution(metrics)
        
        analysis = optimization_framework.analyze_execution_patterns()
        assert analysis['total_execution_time_seconds'] == 1.5
        assert analysis['success_rate'] == 1.0
        assert analysis['average_flaky_score'] == 0.0
    
    def test_flaky_test_detection(self, flaky_detector):
        """Test flaky test detection."""
        # Record a stable test
        for _ in range(10):
            flaky_detector.record_test_result("stable_test", True)
        
        # Record a flaky test
        results = [True, False, True, True, False, True, False, True, True, False]
        for result in results:
            flaky_detector.record_test_result("flaky_test", result)
        
        flaky_tests = flaky_detector.identify_flaky_tests()
        
        assert len(flaky_tests) == 1
        assert flaky_tests[0]['test_name'] == "flaky_test"
        assert flaky_tests[0]['stability_score'] == 0.7  # 7/10 success
    
    def test_optimization_recommendations(self, optimization_framework):
        """Test optimization recommendation generation."""
        # Create metrics that should trigger recommendations
        slow_metrics = [
            TestExecutionMetrics(
                test_name=f"slow_test_{i}",
                execution_time_seconds=30.0,  # Slow test
                setup_time_seconds=5.0,  # High setup
                teardown_time_seconds=2.0,
                cpu_usage_percent=50.0,
                memory_usage_mb=100.0,
                success=True,
                flaky_score=0.0
            ) for i in range(20)  # 20 tests = 600s total (> 15 min)
        ]
        
        for metrics in slow_metrics:
            optimization_framework.record_test_execution(metrics)
        
        recommendations = optimization_framework.generate_optimization_recommendations()
        
        assert any("parallel test execution" in rec.lower() for rec in recommendations)
        assert any("fixture setup" in rec.lower() for rec in recommendations)
    
    async def test_parallel_execution_simulation(self, optimized_runner):
        """Test parallel execution simulation."""
        # Create mock test functions
        async def mock_test_1(fixture):
            await asyncio.sleep(0.1)
            return True
        
        async def mock_test_2(fixture):
            await asyncio.sleep(0.1)
            return True
        
        def mock_test_3(fixture):
            time.sleep(0.1)
            return True
        
        test_functions = [mock_test_1, mock_test_2, mock_test_3]
        
        start_time = time.time()
        metrics = await optimized_runner.run_tests_parallel(test_functions)
        total_time = time.time() - start_time
        
        # Should complete faster than sequential execution
        assert total_time < 0.3  # 3 * 0.1s = 0.3s sequential
        assert len(metrics) == 3
        assert all(m.success for m in metrics)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])