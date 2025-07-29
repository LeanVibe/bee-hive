"""
Comprehensive Performance Test Suite for Semantic Memory System.

This module provides integration tests that validate all Phase 3 performance
targets using the benchmarking framework and load testing suite.

Validates Performance Targets:
- P95 Search Latency: <200ms for 1M vector index
- Ingestion Throughput: >500 documents/sec sustained
- Context Compression: 60-80% reduction with <500ms processing  
- Cross-Agent Knowledge: <200ms for knowledge transfer operations
- Workflow Overhead: <10ms additional DAG orchestration impact

Test Categories:
- Unit Performance Tests: Individual component benchmarks
- Integration Performance Tests: End-to-end workflow performance
- Load Testing: Realistic production load simulation
- Scalability Testing: Performance across different data sizes
- Regression Testing: Performance degradation detection
- Stress Testing: System limits and failure modes
"""

import asyncio
import logging
import time
import json
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from unittest.mock import patch, AsyncMock

import pytest
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.semantic_memory_service import SemanticMemoryService, get_semantic_memory_service
from app.core.pgvector_manager import PGVectorManager, get_pgvector_manager
from app.core.semantic_embedding_service import SemanticEmbeddingService, get_embedding_service
from app.schemas.semantic_memory import (
    DocumentIngestRequest, BatchIngestRequest, SemanticSearchRequest,
    ContextCompressionRequest, SimilaritySearchRequest, ContextualizationRequest,
    DocumentMetadata, ProcessingOptions, BatchOptions, SearchFilters,
    CompressionMethod, ContextualizationMethod, TimeRange, KnowledgeType
)

from .semantic_memory_benchmarks import (
    SemanticMemoryBenchmarks, PerformanceTarget, TestDataSize, BenchmarkResult,
    PerformanceReport, RealisticDataGenerator, ResourceMonitor
)
from .load_testing_suite import (
    LoadTestRunner, LoadTestConfig, LoadTestScenario, LoadPattern,
    LoadTestResult, UserBehavior
)

logger = logging.getLogger(__name__)


class TestSemanticMemoryPerformance:
    """Main performance test suite for semantic memory system."""
    
    @pytest.fixture(scope="class")
    async def benchmarks(self):
        """Initialize benchmark framework."""
        benchmark_system = SemanticMemoryBenchmarks()
        await benchmark_system.initialize()
        yield benchmark_system
        await benchmark_system.cleanup()
    
    @pytest.fixture(scope="class") 
    async def load_tester(self):
        """Initialize load testing framework."""
        return LoadTestRunner()
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_search_latency_benchmark(self, benchmarks: SemanticMemoryBenchmarks):
        """Test semantic search latency meets P95 target."""
        logger.info("🔍 Testing search latency performance")
        
        # Setup test data
        await benchmarks.setup_test_data(TestDataSize.LARGE, agent_count=5)
        
        # Run search latency benchmark
        result = await benchmarks.benchmark_search_latency(
            query_count=200,
            concurrent_queries=20
        )
        
        # Validate performance target
        assert result.passed, (
            f"Search latency P95 {result.measured_value:.2f}ms exceeds target "
            f"{result.target_value:.2f}ms (margin: {result.margin:.2f}ms)"
        )
        
        # Additional performance checks
        assert result.metadata['error_rate_percent'] < 5.0, \
            f"Search error rate {result.metadata['error_rate_percent']:.1f}% too high"
        
        assert result.metadata['queries_per_second'] > 20.0, \
            f"Search throughput {result.metadata['queries_per_second']:.1f} QPS too low"
        
        logger.info(f"✅ Search latency test passed: P95={result.measured_value:.2f}ms, "
                   f"QPS={result.metadata['queries_per_second']:.1f}")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_ingestion_throughput_benchmark(self, benchmarks: SemanticMemoryBenchmarks):
        """Test document ingestion throughput meets target."""
        logger.info("📥 Testing ingestion throughput performance")
        
        # Setup minimal test data (we'll be adding more)
        await benchmarks.setup_test_data(TestDataSize.SMALL, agent_count=3)
        
        # Run ingestion throughput benchmark
        result = await benchmarks.benchmark_ingestion_throughput(
            document_count=2000,
            batch_size=100,
            concurrent_batches=10
        )
        
        # Validate performance target
        assert result.passed, (
            f"Ingestion throughput {result.measured_value:.1f} docs/sec below target "
            f"{result.target_value:.1f} docs/sec (margin: {result.margin:.1f})"
        )
        
        # Additional performance checks
        assert result.metadata['error_rate_percent'] < 5.0, \
            f"Ingestion error rate {result.metadata['error_rate_percent']:.1f}% too high"
        
        logger.info(f"✅ Ingestion throughput test passed: "
                   f"{result.measured_value:.1f} docs/sec")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_context_compression_performance(self, benchmarks: SemanticMemoryBenchmarks):
        """Test context compression performance and quality."""
        logger.info("🗜️ Testing context compression performance")
        
        # Setup test data
        await benchmarks.setup_test_data(TestDataSize.MEDIUM, agent_count=3)
        
        # Run context compression benchmarks
        results = await benchmarks.benchmark_context_compression(
            context_sizes=[100, 500, 1000, 2000],
            compression_methods=[
                CompressionMethod.SEMANTIC_CLUSTERING,
                CompressionMethod.IMPORTANCE_FILTERING,
                CompressionMethod.TEMPORAL_DECAY,
                CompressionMethod.HYBRID
            ]
        )
        
        # Validate compression time targets
        time_results = [r for r in results if "compression_time" in r.name]
        for result in time_results:
            assert result.passed, (
                f"Compression time {result.measured_value:.2f}ms exceeds target "
                f"{result.target_value:.2f}ms for {result.metadata['compression_method']}"
            )
        
        # Validate compression ratio targets
        ratio_results = [r for r in results if "compression_ratio" in r.name]
        for result in ratio_results:
            assert result.passed, (
                f"Compression ratio {result.measured_value:.1%} below target "
                f"{result.target_value:.1%} for {result.metadata['compression_method']}"
            )
            
            # Ensure semantic preservation is maintained
            preservation = result.metadata.get('avg_semantic_preservation', 0)
            assert preservation >= 0.8, \
                f"Semantic preservation {preservation:.1%} too low for {result.metadata['compression_method']}"
        
        logger.info(f"✅ Context compression tests passed for {len(results)} configurations")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_cross_agent_knowledge_sharing_performance(self, benchmarks: SemanticMemoryBenchmarks):
        """Test cross-agent knowledge sharing latency."""
        logger.info("🤝 Testing cross-agent knowledge sharing performance")
        
        # Setup test data across multiple agents
        await benchmarks.setup_test_data(TestDataSize.MEDIUM, agent_count=10)
        
        # Run knowledge sharing benchmark
        result = await benchmarks.benchmark_cross_agent_knowledge_sharing(
            agent_count=10,
            knowledge_queries=100
        )
        
        # Validate performance target
        assert result.passed, (
            f"Knowledge sharing latency P95 {result.measured_value:.2f}ms exceeds target "
            f"{result.target_value:.2f}ms"
        )
        
        # Additional performance checks
        assert result.metadata['error_rate_percent'] < 5.0, \
            f"Knowledge sharing error rate {result.metadata['error_rate_percent']:.1f}% too high"
        
        assert result.metadata['queries_per_second'] > 5.0, \
            f"Knowledge sharing throughput {result.metadata['queries_per_second']:.1f} QPS too low"
        
        logger.info(f"✅ Knowledge sharing test passed: P95={result.measured_value:.2f}ms")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_comprehensive_performance_suite(self, benchmarks: SemanticMemoryBenchmarks):
        """Run comprehensive performance benchmark suite."""
        logger.info("🚀 Running comprehensive performance benchmark suite")
        
        # Run comprehensive benchmark suite
        report = await benchmarks.run_comprehensive_benchmark_suite(
            test_data_size=TestDataSize.LARGE,
            include_load_tests=False  # Load tests run separately
        )
        
        # Validate overall performance
        assert report.overall_score >= 0.8, \
            f"Overall performance score {report.overall_score:.1%} below 80% threshold"
        
        # Validate critical benchmarks
        critical_benchmarks = [
            "search_latency_p95_ms",
            "ingestion_throughput_docs_per_sec"
        ]
        
        for benchmark in report.benchmarks:
            if benchmark.name in critical_benchmarks:
                assert benchmark.passed, \
                    f"Critical benchmark {benchmark.name} failed: {benchmark.measured_value}"
        
        # Log comprehensive results
        logger.info(f"📊 Performance suite results:")
        logger.info(f"   Overall Score: {report.overall_score:.1%}")
        logger.info(f"   Benchmarks Passed: {sum(1 for b in report.benchmarks if b.passed)}/{len(report.benchmarks)}")
        
        for benchmark in report.benchmarks:
            status = "✅ PASS" if benchmark.passed else "❌ FAIL"
            logger.info(f"   {status} {benchmark.name}: {benchmark.measured_value:.2f}")
        
        # Save detailed report
        report_path = Path(tempfile.gettempdir()) / f"semantic_memory_performance_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump({
                'timestamp': report.timestamp.isoformat(),
                'test_suite': report.test_suite,
                'overall_score': report.overall_score,
                'benchmarks': [
                    {
                        'name': b.name,
                        'target_value': b.target_value,
                        'measured_value': b.measured_value,
                        'passed': b.passed,
                        'margin': b.margin,
                        'metadata': b.metadata
                    }
                    for b in report.benchmarks
                ],
                'system_info': report.system_info,
                'recommendations': report.recommendations
            }, f, indent=2)
        
        logger.info(f"📄 Detailed report saved to: {report_path}")
        
        return report
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_load_testing_suite(self, load_tester: LoadTestRunner):
        """Run comprehensive load testing suite."""
        logger.info("🚀 Running load testing suite")
        
        # Run comprehensive load tests
        results = await load_tester.run_comprehensive_load_test_suite()
        
        # Validate load test results
        assert len(results) >= 3, f"Expected at least 3 load test scenarios, got {len(results)}"
        
        # Validate performance under load
        for result in results:
            # Check latency under load
            if result.concurrent_users <= 10:
                # Normal load should meet strict targets
                assert result.p95_latency_ms <= PerformanceTarget.P95_SEARCH_LATENCY_MS.value, \
                    f"P95 latency {result.p95_latency_ms:.1f}ms exceeds target under {result.concurrent_users} users"
            
            elif result.concurrent_users <= 50:
                # High load should meet relaxed targets (2x normal)
                assert result.p95_latency_ms <= PerformanceTarget.P95_SEARCH_LATENCY_MS.value * 2, \
                    f"P95 latency {result.p95_latency_ms:.1f}ms too high under {result.concurrent_users} users"
            
            # Error rate should stay reasonable
            assert result.error_rate_percent <= 10.0, \
                f"Error rate {result.error_rate_percent:.1f}% too high under {result.concurrent_users} users"
        
        # Generate load test report
        report = load_tester.generate_load_test_report(results)
        
        # Log summary
        logger.info("📊 Load testing results:")
        for result in results:
            logger.info(f"   {result.scenario_name}: {result.operations_per_second:.1f} ops/sec, "
                       f"P95={result.p95_latency_ms:.1f}ms, errors={result.error_rate_percent:.1f}%")
        
        # Save load test report
        report_path = Path(tempfile.gettempdir()) / f"semantic_memory_load_test_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Load test report saved to: {report_path}")
        
        return results
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_scalability_benchmarks(self, benchmarks: SemanticMemoryBenchmarks):
        """Test performance scalability across different data sizes."""
        logger.info("📈 Testing scalability across data sizes")
        
        scalability_results = {}
        
        # Test different data sizes
        test_sizes = [TestDataSize.SMALL, TestDataSize.MEDIUM, TestDataSize.LARGE]
        
        for size in test_sizes:
            logger.info(f"Testing scalability with {size.value} dataset")
            
            # Setup test data for this size
            setup_info = await benchmarks.setup_test_data(size, agent_count=5)
            
            # Run search latency benchmark
            search_result = await benchmarks.benchmark_search_latency(
                query_count=50,
                concurrent_queries=5
            )
            
            # Run ingestion throughput benchmark
            ingestion_result = await benchmarks.benchmark_ingestion_throughput(
                document_count=min(500, setup_info['document_count'] // 2),
                batch_size=25,
                concurrent_batches=3
            )
            
            scalability_results[size.value] = {
                'data_size': setup_info['document_count'],
                'search_p95_latency_ms': search_result.measured_value,
                'ingestion_throughput_docs_sec': ingestion_result.measured_value,
                'search_qps': search_result.metadata['queries_per_second'],
                'search_error_rate': search_result.metadata['error_rate_percent'],
                'ingestion_error_rate': ingestion_result.metadata['error_rate_percent']
            }
            
            # Cleanup before next test
            await benchmarks.cleanup()
            await benchmarks.initialize()
        
        # Analyze scalability characteristics
        sizes = [scalability_results[size.value]['data_size'] for size in test_sizes]
        latencies = [scalability_results[size.value]['search_p95_latency_ms'] for size in test_sizes]
        throughputs = [scalability_results[size.value]['ingestion_throughput_docs_sec'] for size in test_sizes]
        
        # Check that latency doesn't grow too quickly with data size
        if len(latencies) >= 2:
            latency_growth = latencies[-1] / latencies[0]
            data_growth = sizes[-1] / sizes[0]
            latency_scaling_factor = latency_growth / data_growth
            
            # Latency should grow sub-linearly with data size
            assert latency_scaling_factor < 1.5, \
                f"Search latency scaling factor {latency_scaling_factor:.2f} indicates poor scalability"
            
            logger.info(f"✅ Latency scaling factor: {latency_scaling_factor:.2f} (sub-linear)")
        
        # Check that throughput doesn't degrade significantly
        if len(throughputs) >= 2:
            throughput_retention = throughputs[-1] / throughputs[0]
            
            # Throughput should retain at least 70% of baseline
            assert throughput_retention >= 0.7, \
                f"Ingestion throughput retention {throughput_retention:.1%} indicates scalability issues"
            
            logger.info(f"✅ Throughput retention: {throughput_retention:.1%}")
        
        logger.info(f"📊 Scalability test results:")
        for size in test_sizes:
            result = scalability_results[size.value]
            logger.info(f"   {size.value}: {result['data_size']} docs, "
                       f"search P95={result['search_p95_latency_ms']:.1f}ms, "
                       f"ingestion={result['ingestion_throughput_docs_sec']:.1f} docs/sec")
        
        return scalability_results
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_resource_utilization_benchmarks(self, benchmarks: SemanticMemoryBenchmarks):
        """Test resource utilization under performance load."""
        logger.info("💾 Testing resource utilization under load")
        
        # Setup large test dataset
        await benchmarks.setup_test_data(TestDataSize.LARGE, agent_count=5)
        
        # Start resource monitoring
        monitor = ResourceMonitor()
        await monitor.start_monitoring()
        
        try:
            # Run intensive operations
            tasks = []
            
            # Concurrent search operations
            for _ in range(10):
                tasks.append(benchmarks.benchmark_search_latency(
                    query_count=20,
                    concurrent_queries=5
                ))
            
            # Concurrent ingestion operations
            for _ in range(5):
                tasks.append(benchmarks.benchmark_ingestion_throughput(
                    document_count=200,
                    batch_size=20,
                    concurrent_batches=2
                ))
            
            # Concurrent compression operations
            for _ in range(3):
                tasks.append(benchmarks.benchmark_context_compression(
                    context_sizes=[100, 500],
                    compression_methods=[CompressionMethod.SEMANTIC_CLUSTERING]
                ))
            
            # Execute all operations concurrently
            await asyncio.gather(*tasks)
            
        finally:
            # Stop monitoring and get stats
            await monitor.stop_monitoring()
            resource_stats = monitor.get_stats()
        
        # Validate resource usage targets
        max_memory_mb = resource_stats.get('max_memory_mb', 0)
        max_cpu_percent = resource_stats.get('max_cpu_percent', 0)
        
        assert max_memory_mb <= PerformanceTarget.MEMORY_USAGE_MB.value, \
            f"Memory usage {max_memory_mb:.1f}MB exceeds target {PerformanceTarget.MEMORY_USAGE_MB.value}MB"
        
        assert max_cpu_percent <= PerformanceTarget.CPU_UTILIZATION_PCT.value, \
            f"CPU utilization {max_cpu_percent:.1f}% exceeds target {PerformanceTarget.CPU_UTILIZATION_PCT.value}%"
        
        logger.info(f"✅ Resource utilization within targets: "
                   f"Memory={max_memory_mb:.1f}MB, CPU={max_cpu_percent:.1f}%")
        
        return resource_stats
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_performance_regression_detection(self, benchmarks: SemanticMemoryBenchmarks):
        """Test performance regression detection capability."""
        logger.info("🔍 Testing performance regression detection")
        
        # Setup test data
        await benchmarks.setup_test_data(TestDataSize.MEDIUM, agent_count=3)
        
        # Run baseline performance test
        baseline_result = await benchmarks.benchmark_search_latency(
            query_count=50,
            concurrent_queries=5
        )
        
        # Simulate performance regression by adding artificial delay
        original_search = benchmarks.semantic_service.semantic_search
        
        async def slow_search(request):
            # Add 100ms artificial delay
            await asyncio.sleep(0.1)
            return await original_search(request)
        
        # Monkey patch to simulate regression
        with patch.object(benchmarks.semantic_service, 'semantic_search', slow_search):
            regression_result = await benchmarks.benchmark_search_latency(
                query_count=50,
                concurrent_queries=5
            )
        
        # Detect regression
        latency_increase = regression_result.measured_value - baseline_result.measured_value
        regression_factor = latency_increase / baseline_result.measured_value
        
        # Should detect significant regression
        assert regression_factor > 0.5, \
            f"Failed to detect artificial regression: factor={regression_factor:.2f}"
        
        # Should flag as regression
        regression_threshold = 0.2  # 20% increase threshold
        is_regression = regression_factor > regression_threshold
        
        assert is_regression, \
            f"Regression detection failed: {regression_factor:.1%} increase not flagged"
        
        logger.info(f"✅ Regression detection working: {regression_factor:.1%} increase detected")
        
        return {
            'baseline_latency_ms': baseline_result.measured_value,
            'regression_latency_ms': regression_result.measured_value,
            'regression_factor': regression_factor,
            'regression_detected': is_regression
        }
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_endurance_performance(self, benchmarks: SemanticMemoryBenchmarks):
        """Test performance stability over extended duration."""
        logger.info("⏱️ Testing endurance performance (extended duration)")
        
        # Setup test data
        await benchmarks.setup_test_data(TestDataSize.MEDIUM, agent_count=3)
        
        # Start resource monitoring
        monitor = ResourceMonitor()
        await monitor.start_monitoring()
        
        # Track performance over time
        performance_samples = []
        test_duration = 300  # 5 minutes
        sample_interval = 30  # 30 seconds
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < test_duration:
                # Run performance sample
                sample_start = time.time()
                
                result = await benchmarks.benchmark_search_latency(
                    query_count=20,
                    concurrent_queries=3
                )
                
                sample_time = time.time()
                
                performance_samples.append({
                    'timestamp': sample_time,
                    'elapsed_seconds': sample_time - start_time,
                    'p95_latency_ms': result.measured_value,
                    'qps': result.metadata['queries_per_second'],
                    'error_rate': result.metadata['error_rate_percent']
                })
                
                # Wait for next sample
                await asyncio.sleep(sample_interval)
                
        finally:
            await monitor.stop_monitoring()
            resource_stats = monitor.get_stats()
        
        # Analyze performance stability
        latencies = [sample['p95_latency_ms'] for sample in performance_samples]
        
        if len(latencies) >= 3:
            # Check for performance degradation over time
            early_latency = np.mean(latencies[:len(latencies)//3])
            late_latency = np.mean(latencies[-len(latencies)//3:])
            
            degradation_factor = (late_latency - early_latency) / early_latency
            
            # Performance should be stable (< 20% degradation)
            assert degradation_factor < 0.2, \
                f"Performance degraded {degradation_factor:.1%} over time"
            
            # Check performance variance
            latency_std = np.std(latencies)
            latency_cv = latency_std / np.mean(latencies)  # Coefficient of variation
            
            # Performance should be consistent (CV < 0.3)
            assert latency_cv < 0.3, \
                f"Performance too variable: CV={latency_cv:.2f}"
            
            logger.info(f"✅ Endurance test passed: degradation={degradation_factor:.1%}, "
                       f"variance CV={latency_cv:.2f}")
        
        return {
            'duration_seconds': test_duration,
            'samples_collected': len(performance_samples),
            'performance_samples': performance_samples,
            'resource_stats': resource_stats,
            'degradation_factor': degradation_factor if len(latencies) >= 3 else 0,
            'performance_cv': latency_cv if len(latencies) >= 3 else 0
        }


class TestSemanticMemoryIntegrationPerformance:
    """Integration performance tests with other system components."""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_workflow_integration_performance(self):
        """Test performance impact of semantic memory in workflow context."""
        logger.info("⚙️ Testing workflow integration performance")
        
        # This test would integrate with the workflow engine
        # For now, we'll create a simplified test that measures overhead
        
        benchmarks = SemanticMemoryBenchmarks()
        await benchmarks.initialize()
        
        try:
            # Setup test data
            await benchmarks.setup_test_data(TestDataSize.SMALL, agent_count=2)
            
            # Measure baseline semantic operations
            baseline_start = time.perf_counter()
            
            search_result = await benchmarks.benchmark_search_latency(
                query_count=20,
                concurrent_queries=2
            )
            
            baseline_time = time.perf_counter() - baseline_start
            
            # Simulate workflow overhead by adding minimal DAG processing
            workflow_start = time.perf_counter()
            
            # Add simulated workflow overhead (should be minimal)
            workflow_overhead_ms = 5.0  # 5ms simulated overhead
            await asyncio.sleep(workflow_overhead_ms / 1000)
            
            # Run same operations with workflow context
            workflow_result = await benchmarks.benchmark_search_latency(
                query_count=20,
                concurrent_queries=2
            )
            
            workflow_time = time.perf_counter() - workflow_start
            
            # Calculate actual overhead
            actual_overhead_ms = (workflow_time - baseline_time) * 1000
            
            # Validate workflow overhead target
            assert actual_overhead_ms <= PerformanceTarget.WORKFLOW_OVERHEAD_MS.value, \
                f"Workflow overhead {actual_overhead_ms:.2f}ms exceeds target " \
                f"{PerformanceTarget.WORKFLOW_OVERHEAD_MS.value}ms"
            
            logger.info(f"✅ Workflow integration test passed: overhead={actual_overhead_ms:.2f}ms")
            
            return {
                'baseline_time_ms': baseline_time * 1000,
                'workflow_time_ms': workflow_time * 1000,
                'overhead_ms': actual_overhead_ms,
                'overhead_target_ms': PerformanceTarget.WORKFLOW_OVERHEAD_MS.value
            }
            
        finally:
            await benchmarks.cleanup()
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_multi_agent_performance(self):
        """Test performance with multiple agents operating concurrently."""
        logger.info("👥 Testing concurrent multi-agent performance")
        
        benchmarks = SemanticMemoryBenchmarks()
        await benchmarks.initialize()
        
        try:
            # Setup test data for multiple agents
            await benchmarks.setup_test_data(TestDataSize.MEDIUM, agent_count=10)
            
            # Test concurrent operations across multiple agents
            agent_ids = benchmarks.test_agent_ids[:5]  # Use 5 agents
            
            async def agent_workload(agent_id: str) -> Dict[str, float]:
                """Simulate typical agent workload."""
                start_time = time.perf_counter()
                
                # Mix of operations for this agent
                operations = []
                
                # Search operations
                for _ in range(5):
                    operations.append(benchmarks.benchmark_search_latency(
                        query_count=10,
                        concurrent_queries=2
                    ))
                
                # Knowledge retrieval
                operations.append(benchmarks.benchmark_cross_agent_knowledge_sharing(
                    agent_count=1,
                    knowledge_queries=10
                ))
                
                # Wait for all operations
                results = await asyncio.gather(*operations)
                
                total_time = time.perf_counter() - start_time
                
                # Calculate average latency across operations
                avg_latency = np.mean([r.measured_value for r in results])
                
                return {
                    'agent_id': agent_id,
                    'total_time_seconds': total_time,
                    'avg_latency_ms': avg_latency,
                    'operations_completed': len(results)
                }
            
            # Run concurrent agent workloads
            start_time = time.perf_counter()
            
            agent_tasks = [agent_workload(agent_id) for agent_id in agent_ids]
            agent_results = await asyncio.gather(*agent_tasks)
            
            total_time = time.perf_counter() - start_time
            
            # Analyze multi-agent performance
            avg_agent_latency = np.mean([r['avg_latency_ms'] for r in agent_results])
            max_agent_latency = max([r['avg_latency_ms'] for r in agent_results])
            
            # Performance should scale reasonably with concurrent agents
            assert max_agent_latency <= PerformanceTarget.P95_SEARCH_LATENCY_MS.value * 2, \
                f"Multi-agent max latency {max_agent_latency:.2f}ms too high"
            
            # Concurrent execution should be efficient
            total_operations = sum(r['operations_completed'] for r in agent_results)
            operations_per_second = total_operations / total_time
            
            assert operations_per_second >= 5.0, \
                f"Multi-agent throughput {operations_per_second:.1f} ops/sec too low"
            
            logger.info(f"✅ Multi-agent test passed: {len(agent_ids)} agents, "
                       f"avg_latency={avg_agent_latency:.2f}ms, "
                       f"throughput={operations_per_second:.1f} ops/sec")
            
            return {
                'agent_count': len(agent_ids),
                'total_time_seconds': total_time,
                'avg_latency_ms': avg_agent_latency,
                'max_latency_ms': max_agent_latency,
                'operations_per_second': operations_per_second,
                'agent_results': agent_results
            }
            
        finally:
            await benchmarks.cleanup()


# Test markers for performance testing
pytestmark = [
    pytest.mark.performance,
    pytest.mark.asyncio
]