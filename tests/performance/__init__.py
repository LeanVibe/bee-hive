"""
Performance Testing Suite for Semantic Memory System.

This package provides comprehensive performance testing, benchmarking, and
monitoring capabilities for the LeanVibe Agent Hive semantic memory system.

Components:
- semantic_memory_benchmarks: Core benchmarking framework
- load_testing_suite: Locust-based load testing
- test_semantic_memory_performance: Integration test suite
- regression_detector: Performance regression detection
- ci_performance_tests: CI/CD performance gates

Performance Targets (Phase 3):
- P95 Search Latency: <200ms for 1M vector index
- Ingestion Throughput: >500 documents/sec sustained
- Context Compression: 60-80% reduction with <500ms processing
- Cross-Agent Knowledge: <200ms for knowledge transfer operations
- Workflow Overhead: <10ms additional DAG orchestration impact
"""

from .semantic_memory_benchmarks import (
    SemanticMemoryBenchmarks,
    PerformanceTarget,
    TestDataSize,
    BenchmarkResult,
    PerformanceReport,
    RealisticDataGenerator,
    ResourceMonitor
)

from .load_testing_suite import (
    LoadTestRunner,
    LoadTestConfig,
    LoadTestScenario,
    LoadPattern,
    LoadTestResult,
    UserBehavior
)

from .regression_detector import (
    PerformanceRegressionDetector,
    RegressionSeverity,
    DetectionMethod,
    PerformanceBaseline,
    RegressionAlert,
    TrendAnalysis
)

__all__ = [
    # Core benchmarking
    'SemanticMemoryBenchmarks',
    'PerformanceTarget',
    'TestDataSize',
    'BenchmarkResult',
    'PerformanceReport',
    'RealisticDataGenerator',
    'ResourceMonitor',
    
    # Load testing
    'LoadTestRunner',
    'LoadTestConfig',
    'LoadTestScenario', 
    'LoadPattern',
    'LoadTestResult',
    'UserBehavior',
    
    # Regression detection
    'PerformanceRegressionDetector',
    'RegressionSeverity',
    'DetectionMethod',
    'PerformanceBaseline',
    'RegressionAlert',
    'TrendAnalysis'
]

__version__ = '1.0.0'