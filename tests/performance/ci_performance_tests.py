"""
Continuous Integration Performance Tests for Semantic Memory System.

This module provides lightweight performance tests designed to run in CI/CD
pipelines to catch performance regressions early in the development cycle.

Features:
- Fast-running performance smoke tests (<5 minutes total)
- Critical performance target validation
- Regression detection against baselines
- CI/CD pipeline integration
- Performance gate enforcement
- Automated alert generation for failures
- Performance trend tracking across builds
"""

import asyncio
import logging
import time
import json
import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import subprocess

import pytest

from .semantic_memory_benchmarks import (
    SemanticMemoryBenchmarks, PerformanceTarget, TestDataSize, BenchmarkResult
)
from .regression_detector import (
    PerformanceRegressionDetector, RegressionSeverity, DetectionMethod
)

logger = logging.getLogger(__name__)


class CIPerformanceConfig:
    """Configuration for CI performance tests."""
    
    # Fast test parameters (reduced from full benchmarks)
    SEARCH_QUERIES = 20
    CONCURRENT_SEARCHES = 3
    INGESTION_DOCS = 100
    INGESTION_BATCH_SIZE = 10
    KNOWLEDGE_QUERIES = 10
    
    # CI-specific performance targets (slightly relaxed)
    MAX_SEARCH_LATENCY_MS = 250.0  # 25% higher than production target
    MIN_INGESTION_THROUGHPUT = 400.0  # 20% lower than production target
    MAX_COMPRESSION_TIME_MS = 600.0  # 20% higher than production target
    MAX_KNOWLEDGE_LATENCY_MS = 250.0  # 25% higher than production target
    
    # Regression detection thresholds
    REGRESSION_THRESHOLD_PERCENT = 20.0  # More lenient for CI
    
    # Test timeouts
    INDIVIDUAL_TEST_TIMEOUT = 60  # 1 minute per test
    TOTAL_SUITE_TIMEOUT = 300  # 5 minutes total


class CIPerformanceGate:
    """Performance gate for CI/CD pipeline."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.regression_alerts = []
        self.gate_status = "PENDING"
        self.gate_message = ""
    
    def add_result(self, result: BenchmarkResult):
        """Add benchmark result to gate."""
        self.results.append(result)
    
    def evaluate_gate(self) -> Tuple[bool, str]:
        """Evaluate if performance gate should pass."""
        if not self.results:
            return False, "No performance results available"
        
        # Check critical targets
        critical_failures = []
        
        for result in self.results:
            if not result.passed:
                failure_msg = (f"{result.name}: {result.measured_value:.2f} "
                             f"{result.target_operator} {result.target_value:.2f} "
                             f"(margin: {result.margin:.2f})")
                critical_failures.append(failure_msg)
        
        # Determine gate status
        if not critical_failures:
            self.gate_status = "PASS"
            self.gate_message = f"All {len(self.results)} performance targets met"
            return True, self.gate_message
        
        elif len(critical_failures) == 1 and len(self.results) > 3:
            # Allow one failure if we have multiple tests
            self.gate_status = "PASS_WITH_WARNING"
            self.gate_message = f"1 performance target missed: {critical_failures[0]}"
            return True, self.gate_message
        
        else:
            self.gate_status = "FAIL"
            self.gate_message = f"{len(critical_failures)} performance targets failed"
            return False, self.gate_message
    
    def generate_ci_report(self) -> Dict[str, Any]:
        """Generate CI-friendly performance report."""
        passed_tests = sum(1 for r in self.results if r.passed)
        
        return {
            'gate_status': self.gate_status,
            'gate_message': self.gate_message,
            'total_tests': len(self.results),
            'passed_tests': passed_tests,
            'failed_tests': len(self.results) - passed_tests,
            'pass_rate': passed_tests / len(self.results) if self.results else 0,
            'test_results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'measured_value': r.measured_value,
                    'target_value': r.target_value,
                    'margin': r.margin,
                    'execution_time_ms': r.execution_time_ms
                }
                for r in self.results
            ],
            'regression_alerts': len(self.regression_alerts),
            'timestamp': datetime.utcnow().isoformat()
        }


class TestCIPerformance:
    """CI Performance test suite - designed for fast execution."""
    
    @pytest.fixture(scope="class")
    async def ci_benchmarks(self):
        """Initialize lightweight benchmark framework for CI."""
        benchmark_system = SemanticMemoryBenchmarks()
        await benchmark_system.initialize()
        
        # Setup minimal test data for CI
        await benchmark_system.setup_test_data(TestDataSize.SMALL, agent_count=2)
        
        yield benchmark_system
        await benchmark_system.cleanup()
    
    @pytest.fixture(scope="class")
    def performance_gate(self):
        """Performance gate for CI pipeline."""
        return CIPerformanceGate()
    
    @pytest.fixture(scope="class")
    def regression_detector(self):
        """Regression detector for CI."""
        # Use temporary database for CI
        db_path = tempfile.mktemp(suffix=".db")
        detector = PerformanceRegressionDetector(db_path)
        
        yield detector
        
        # Cleanup
        try:
            os.unlink(db_path)
        except FileNotFoundError:
            pass
    
    @pytest.mark.asyncio
    @pytest.mark.ci_performance
    @pytest.mark.timeout(CIPerformanceConfig.INDIVIDUAL_TEST_TIMEOUT)
    async def test_ci_search_latency_smoke(
        self,
        ci_benchmarks: SemanticMemoryBenchmarks,
        performance_gate: CIPerformanceGate,
        regression_detector: PerformanceRegressionDetector
    ):
        """CI smoke test for search latency."""
        logger.info("🔍 CI: Testing search latency (smoke test)")
        
        # Run lightweight search benchmark
        result = await ci_benchmarks.benchmark_search_latency(
            query_count=CIPerformanceConfig.SEARCH_QUERIES,
            concurrent_queries=CIPerformanceConfig.CONCURRENT_SEARCHES
        )
        
        # Override target for CI (more lenient)
        result.target_value = CIPerformanceConfig.MAX_SEARCH_LATENCY_MS
        result.passed = result.measured_value <= result.target_value
        result.margin = result.target_value - result.measured_value
        
        # Add to performance gate
        performance_gate.add_result(result)
        
        # Check for regression
        alert = regression_detector.detect_regression(
            "search_latency_p95_ms",
            result.measured_value,
            DetectionMethod.THRESHOLD_BASED
        )
        
        if alert and alert.severity in [RegressionSeverity.MAJOR, RegressionSeverity.CRITICAL]:
            performance_gate.regression_alerts.append(alert)
            pytest.fail(f"Critical performance regression detected: {alert.degradation_percent:.1f}% degradation")
        
        # Assert CI target
        assert result.passed, (
            f"CI search latency {result.measured_value:.2f}ms exceeds target "
            f"{result.target_value:.2f}ms"
        )
        
        logger.info(f"✅ CI search latency: {result.measured_value:.2f}ms (target: {result.target_value:.2f}ms)")
    
    @pytest.mark.asyncio
    @pytest.mark.ci_performance
    @pytest.mark.timeout(CIPerformanceConfig.INDIVIDUAL_TEST_TIMEOUT)
    async def test_ci_ingestion_throughput_smoke(
        self,
        ci_benchmarks: SemanticMemoryBenchmarks,
        performance_gate: CIPerformanceGate,
        regression_detector: PerformanceRegressionDetector
    ):
        """CI smoke test for ingestion throughput."""
        logger.info("📥 CI: Testing ingestion throughput (smoke test)")
        
        # Run lightweight ingestion benchmark
        result = await ci_benchmarks.benchmark_ingestion_throughput(
            document_count=CIPerformanceConfig.INGESTION_DOCS,
            batch_size=CIPerformanceConfig.INGESTION_BATCH_SIZE,
            concurrent_batches=2
        )
        
        # Override target for CI
        result.target_value = CIPerformanceConfig.MIN_INGESTION_THROUGHPUT
        result.target_operator = ">="
        result.passed = result.measured_value >= result.target_value
        result.margin = result.measured_value - result.target_value
        
        # Add to performance gate
        performance_gate.add_result(result)
        
        # Check for regression
        alert = regression_detector.detect_regression(
            "ingestion_throughput_docs_per_sec",
            result.measured_value,
            DetectionMethod.THRESHOLD_BASED
        )
        
        if alert and alert.severity in [RegressionSeverity.MAJOR, RegressionSeverity.CRITICAL]:
            performance_gate.regression_alerts.append(alert)
            pytest.fail(f"Critical throughput regression detected: {alert.degradation_percent:.1f}% degradation")
        
        # Assert CI target
        assert result.passed, (
            f"CI ingestion throughput {result.measured_value:.1f} docs/sec below target "
            f"{result.target_value:.1f} docs/sec"
        )
        
        logger.info(f"✅ CI ingestion throughput: {result.measured_value:.1f} docs/sec (target: {result.target_value:.1f})")
    
    @pytest.mark.asyncio
    @pytest.mark.ci_performance
    @pytest.mark.timeout(CIPerformanceConfig.INDIVIDUAL_TEST_TIMEOUT)
    async def test_ci_context_compression_smoke(
        self,
        ci_benchmarks: SemanticMemoryBenchmarks,
        performance_gate: CIPerformanceGate,
        regression_detector: PerformanceRegressionDetector
    ):
        """CI smoke test for context compression."""
        logger.info("🗜️ CI: Testing context compression (smoke test)")
        
        # Run lightweight compression benchmark
        results = await ci_benchmarks.benchmark_context_compression(
            context_sizes=[50, 100],  # Smaller sizes for CI
            compression_methods=[CompressionMethod.SEMANTIC_CLUSTERING]  # Single method
        )
        
        # Find compression time result
        time_result = next((r for r in results if "compression_time" in r.name), None)
        if time_result:
            # Override target for CI
            time_result.target_value = CIPerformanceConfig.MAX_COMPRESSION_TIME_MS
            time_result.passed = time_result.measured_value <= time_result.target_value
            time_result.margin = time_result.target_value - time_result.measured_value
            
            # Add to performance gate
            performance_gate.add_result(time_result)
            
            # Check for regression
            alert = regression_detector.detect_regression(
                "context_compression_time_ms",
                time_result.measured_value,
                DetectionMethod.THRESHOLD_BASED
            )
            
            if alert and alert.severity in [RegressionSeverity.MAJOR, RegressionSeverity.CRITICAL]:
                performance_gate.regression_alerts.append(alert)
                pytest.fail(f"Critical compression regression detected: {alert.degradation_percent:.1f}% degradation")
            
            # Assert CI target
            assert time_result.passed, (
                f"CI compression time {time_result.measured_value:.2f}ms exceeds target "
                f"{time_result.target_value:.2f}ms"
            )
            
            logger.info(f"✅ CI compression time: {time_result.measured_value:.2f}ms")
        
        # Check compression ratio
        ratio_result = next((r for r in results if "compression_ratio" in r.name), None)
        if ratio_result:
            # Ensure minimum compression is achieved
            min_compression = 0.5  # 50% minimum for CI
            ratio_result.target_value = min_compression
            ratio_result.passed = ratio_result.measured_value >= min_compression
            
            assert ratio_result.passed, (
                f"CI compression ratio {ratio_result.measured_value:.1%} below minimum {min_compression:.1%}"
            )
            
            logger.info(f"✅ CI compression ratio: {ratio_result.measured_value:.1%}")
    
    @pytest.mark.asyncio
    @pytest.mark.ci_performance
    @pytest.mark.timeout(CIPerformanceConfig.INDIVIDUAL_TEST_TIMEOUT)
    async def test_ci_knowledge_sharing_smoke(
        self,
        ci_benchmarks: SemanticMemoryBenchmarks,
        performance_gate: CIPerformanceGate,
        regression_detector: PerformanceRegressionDetector
    ):
        """CI smoke test for knowledge sharing."""
        logger.info("🤝 CI: Testing knowledge sharing (smoke test)")
        
        # Run lightweight knowledge sharing benchmark
        result = await ci_benchmarks.benchmark_cross_agent_knowledge_sharing(
            agent_count=2,  # Minimal agents for CI
            knowledge_queries=CIPerformanceConfig.KNOWLEDGE_QUERIES
        )
        
        # Override target for CI
        result.target_value = CIPerformanceConfig.MAX_KNOWLEDGE_LATENCY_MS
        result.passed = result.measured_value <= result.target_value
        result.margin = result.target_value - result.measured_value
        
        # Add to performance gate
        performance_gate.add_result(result)
        
        # Check for regression
        alert = regression_detector.detect_regression(
            "knowledge_sharing_latency_p95_ms",
            result.measured_value,
            DetectionMethod.THRESHOLD_BASED
        )
        
        if alert and alert.severity in [RegressionSeverity.MAJOR, RegressionSeverity.CRITICAL]:
            performance_gate.regression_alerts.append(alert)
            pytest.fail(f"Critical knowledge sharing regression detected: {alert.degradation_percent:.1f}% degradation")
        
        # Assert CI target
        assert result.passed, (
            f"CI knowledge sharing latency {result.measured_value:.2f}ms exceeds target "
            f"{result.target_value:.2f}ms"
        )
        
        logger.info(f"✅ CI knowledge sharing: {result.measured_value:.2f}ms")
    
    @pytest.mark.asyncio
    @pytest.mark.ci_performance
    @pytest.mark.timeout(CIPerformanceConfig.TOTAL_SUITE_TIMEOUT)
    async def test_ci_performance_gate_evaluation(
        self,
        performance_gate: CIPerformanceGate
    ):
        """Evaluate CI performance gate and generate report."""
        logger.info("🚪 CI: Evaluating performance gate")
        
        # Evaluate performance gate
        gate_passed, gate_message = performance_gate.evaluate_gate()
        
        # Generate CI report
        report = performance_gate.generate_ci_report()
        
        # Save report for CI system
        report_file = os.environ.get('CI_PERFORMANCE_REPORT_FILE', 'ci_performance_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Log summary
        logger.info(f"📊 CI Performance Gate: {performance_gate.gate_status}")
        logger.info(f"   Message: {gate_message}")
        logger.info(f"   Tests: {report['passed_tests']}/{report['total_tests']} passed")
        logger.info(f"   Pass Rate: {report['pass_rate']:.1%}")
        
        if performance_gate.regression_alerts:
            logger.warning(f"   Regression Alerts: {len(performance_gate.regression_alerts)}")
        
        # Write gate status for CI system
        gate_status_file = os.environ.get('CI_GATE_STATUS_FILE', 'performance_gate_status.txt')
        with open(gate_status_file, 'w') as f:
            f.write(f"{performance_gate.gate_status}\n{gate_message}\n")
        
        # Set exit code based on gate status
        if gate_passed:
            logger.info("✅ CI Performance Gate: PASSED")
        else:
            logger.error("❌ CI Performance Gate: FAILED")
            pytest.fail(f"Performance gate failed: {gate_message}")
        
        return report


class TestCIRegressionDetection:
    """CI-specific regression detection tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.ci_performance
    async def test_baseline_update_ci(self):
        """Test baseline update functionality in CI environment."""
        logger.info("📊 CI: Testing baseline update")
        
        # Create temporary detector
        db_path = tempfile.mktemp(suffix=".db")
        detector = PerformanceRegressionDetector(db_path)
        
        try:
            # Simulate historical performance data
            metric_name = "test_metric_ci"
            historical_values = [100.0 + i + (i % 5) for i in range(30)]  # Some variation
            
            # Update baseline
            baseline = detector.update_baseline(
                metric_name,
                historical_values=historical_values
            )
            
            # Validate baseline
            assert baseline.baseline_value > 0
            assert baseline.baseline_std > 0
            assert baseline.baseline_samples == len(historical_values)
            assert len(baseline.confidence_interval) == 2
            
            logger.info(f"✅ Baseline updated: {baseline.baseline_value:.2f} ± {baseline.baseline_std:.2f}")
            
        finally:
            # Cleanup
            try:
                os.unlink(db_path)
            except FileNotFoundError:
                pass
    
    @pytest.mark.asyncio
    @pytest.mark.ci_performance
    async def test_regression_detection_ci(self):
        """Test regression detection in CI environment."""
        logger.info("🔍 CI: Testing regression detection")
        
        # Create temporary detector
        db_path = tempfile.mktemp(suffix=".db")
        detector = PerformanceRegressionDetector(db_path)
        
        try:
            metric_name = "test_regression_ci"
            
            # Create baseline
            baseline_values = [100.0 for _ in range(20)]
            detector.update_baseline(metric_name, historical_values=baseline_values)
            
            # Test no regression
            alert = detector.detect_regression(metric_name, 105.0)  # 5% increase
            assert alert is None or alert.severity == RegressionSeverity.NONE
            
            # Test minor regression
            alert = detector.detect_regression(metric_name, 115.0)  # 15% increase
            if alert:
                assert alert.severity in [RegressionSeverity.MINOR, RegressionSeverity.MODERATE]
            
            # Test major regression
            alert = detector.detect_regression(metric_name, 150.0)  # 50% increase
            assert alert is not None
            assert alert.severity in [RegressionSeverity.MAJOR, RegressionSeverity.CRITICAL]
            assert alert.degradation_percent >= 40.0
            
            logger.info(f"✅ Regression detection working: {alert.severity.value} detected")
            
        finally:
            # Cleanup
            try:
                os.unlink(db_path)
            except FileNotFoundError:
                pass


def setup_ci_environment():
    """Setup CI environment for performance testing."""
    # Set environment variables for CI
    os.environ.setdefault('CI_PERFORMANCE_MODE', 'true')
    os.environ.setdefault('PERFORMANCE_TEST_TIMEOUT', str(CIPerformanceConfig.TOTAL_SUITE_TIMEOUT))
    
    # Create output directories
    os.makedirs('ci_reports', exist_ok=True)
    os.makedirs('performance_data', exist_ok=True)


def generate_ci_summary_report(test_session) -> Dict[str, Any]:
    """Generate summary report for CI system."""
    # Extract performance test results
    performance_tests = []
    
    for item in test_session.items:
        if item.get_closest_marker('ci_performance'):
            test_result = {
                'test_name': item.name,
                'status': 'unknown',  # Would be set by pytest hooks
                'duration': 0.0
            }
            performance_tests.append(test_result)
    
    return {
        'ci_summary': {
            'total_performance_tests': len(performance_tests),
            'timestamp': datetime.utcnow().isoformat(),
            'environment': 'ci'
        },
        'performance_tests': performance_tests
    }


# Pytest configuration for CI performance tests
def pytest_configure(config):
    """Configure pytest for CI performance testing."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "ci_performance: mark test as CI performance test"
    )
    config.addinivalue_line(
        "markers", "timeout(seconds): set test timeout"
    )
    
    # Setup CI environment
    setup_ci_environment()


def pytest_collection_modifyitems(config, items):
    """Modify test collection for CI."""
    # Add timeout to all CI performance tests
    for item in items:
        if item.get_closest_marker('ci_performance'):
            # Ensure timeout marker is present
            if not item.get_closest_marker('timeout'):
                item.add_marker(pytest.mark.timeout(CIPerformanceConfig.INDIVIDUAL_TEST_TIMEOUT))


# CI Integration helpers
class CIIntegration:
    """Integration with CI/CD systems."""
    
    @staticmethod
    def get_git_info() -> Dict[str, str]:
        """Get git information for performance tracking."""
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
            branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
            
            return {
                'commit': commit,
                'branch': branch,
                'timestamp': datetime.utcnow().isoformat()
            }
        except subprocess.CalledProcessError:
            return {
                'commit': 'unknown',
                'branch': 'unknown',
                'timestamp': datetime.utcnow().isoformat()
            }
    
    @staticmethod
    def set_ci_status(status: str, description: str):
        """Set CI status (GitHub Actions, etc.)."""
        # GitHub Actions
        if 'GITHUB_ACTIONS' in os.environ:
            print(f"::notice::{description}")
            if status == 'failure':
                print(f"::error::Performance gate failed: {description}")
        
        # Generic CI output
        print(f"CI_PERFORMANCE_STATUS={status}")
        print(f"CI_PERFORMANCE_DESCRIPTION={description}")
    
    @staticmethod
    def upload_performance_artifacts():
        """Upload performance artifacts for CI."""
        artifacts = [
            'ci_performance_report.json',
            'performance_gate_status.txt',
            'performance_history.db'
        ]
        
        for artifact in artifacts:
            if os.path.exists(artifact):
                # In real CI, this would upload to artifact storage
                logger.info(f"📄 Performance artifact available: {artifact}")


# Test markers for CI performance testing
pytestmark = [
    pytest.mark.ci_performance,
    pytest.mark.asyncio
]