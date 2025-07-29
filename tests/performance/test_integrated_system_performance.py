"""
Integrated System Performance Test Suite for LeanVibe Agent Hive 2.0.

Enterprise-grade test suite for validating complete system performance including:
- Multi-agent scalability (50+ concurrent agents)
- Security system performance (<100ms authorization)
- Database operations with pgvector semantic search
- Redis Streams message handling (>10k msgs/sec)
- GitHub integration with rate limiting
- End-to-end workflow performance validation

Performance Requirements:
- Authentication: <50ms P95
- Authorization: <100ms P95
- Context Search: <200ms P95
- Multi-agent: 50+ concurrent agents
- Redis Throughput: >10k msgs/sec
- Memory Efficiency: <2GB for 50 agents
"""

import asyncio
import pytest
import time
import json
import statistics
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.integrated_system_performance_validator import (
    IntegratedSystemPerformanceValidator,
    PerformanceTestType,
    TestSeverity,
    SystemPerformanceReport,
    run_integrated_performance_validation,
    quick_production_readiness_check
)
from app.models.agent import Agent, AgentStatus
from app.models.task import Task, TaskStatus, TaskType
from app.models.context import Context, ContextType
from tests.utils.database_test_utils import DatabaseTestUtils


logger = structlog.get_logger()


@pytest.mark.performance
@pytest.mark.asyncio
class TestIntegratedSystemPerformance:
    """Test suite for integrated system performance validation."""
    
    async def test_authentication_flow_performance(self):
        """Test authentication system performance meets <50ms P95 target."""
        validator = IntegratedSystemPerformanceValidator()
        await validator.initialize_system_components()
        
        # Test authentication performance
        auth_results = await validator._test_authentication_performance(iterations=50)
        
        assert len(auth_results) == 1
        result = auth_results[0]
        
        # Validate performance targets
        assert result.test_type == PerformanceTestType.AUTHENTICATION_FLOW
        assert result.target.target_value == 50.0  # 50ms target
        assert result.measured_value <= 50.0, f"Authentication P95 latency {result.measured_value}ms exceeds 50ms target"
        assert result.success_rate >= 0.95, f"Authentication success rate {result.success_rate:.2%} below 95%"
        assert result.error_count == 0, f"Authentication had {result.error_count} errors"
        
        # Additional performance characteristics
        assert result.additional_metrics["avg_latency_ms"] < result.measured_value
        assert result.additional_metrics["min_latency_ms"] > 0
        assert result.additional_metrics["std_dev_ms"] >= 0
        
        logger.info(f"✅ Authentication performance: {result.measured_value:.1f}ms P95, {result.success_rate:.1%} success rate")
    
    async def test_authorization_decisions_performance(self):
        """Test authorization system performance meets <100ms P95 target.""" 
        validator = IntegratedSystemPerformanceValidator()
        await validator.initialize_system_components()
        
        # Test authorization performance with various scenarios
        authz_results = await validator._test_authorization_performance(iterations=40)
        
        assert len(authz_results) == 1
        result = authz_results[0]
        
        # Validate performance targets
        assert result.test_type == PerformanceTestType.AUTHORIZATION_DECISIONS
        assert result.target.target_value == 100.0  # 100ms target
        assert result.measured_value <= 100.0, f"Authorization P95 latency {result.measured_value}ms exceeds 100ms target"
        assert result.success_rate >= 0.98, f"Authorization success rate {result.success_rate:.2%} below 98%"
        
        # Validate authorization scenario coverage
        assert result.additional_metrics["authorization_scenarios_tested"] >= 4
        assert result.additional_metrics["fastest_decision_ms"] > 0
        assert result.additional_metrics["slowest_decision_ms"] <= 100.0
        
        logger.info(f"✅ Authorization performance: {result.measured_value:.1f}ms P95, {result.additional_metrics['authorization_scenarios_tested']} scenarios tested")
    
    async def test_github_operations_performance(self):
        """Test GitHub integration performance with rate limiting."""
        validator = IntegratedSystemPerformanceValidator()
        await validator.initialize_system_components()
        
        # Test GitHub operations performance
        github_results = await validator._test_github_operations_performance(iterations=30)
        
        assert len(github_results) == 1
        result = github_results[0]
        
        # Validate performance targets
        assert result.test_type == PerformanceTestType.GITHUB_OPERATIONS
        assert result.target.target_value == 500.0  # 500ms target
        assert result.measured_value <= 500.0, f"GitHub P95 latency {result.measured_value}ms exceeds 500ms target"
        assert result.success_rate >= 0.90, f"GitHub success rate {result.success_rate:.2%} below 90%"
        
        # Validate rate limiting handling
        rate_limit_percentage = result.additional_metrics["rate_limit_percentage"]
        assert rate_limit_percentage <= 10.0, f"Rate limiting {rate_limit_percentage:.1f}% too high"
        
        # Validate operation coverage
        assert result.additional_metrics["operations_tested"] >= 5
        
        logger.info(f"✅ GitHub performance: {result.measured_value:.1f}ms P95, {rate_limit_percentage:.1f}% rate limited")
    
    async def test_context_semantic_search_performance(self):
        """Test pgvector semantic search performance meets <200ms P95 target."""
        validator = IntegratedSystemPerformanceValidator()
        await validator.initialize_system_components()
        
        # Test context search performance
        context_results = await validator._test_context_search_performance(iterations=25)
        
        assert len(context_results) == 1
        result = context_results[0]
        
        # Validate performance targets
        assert result.test_type == PerformanceTestType.CONTEXT_SEMANTIC_SEARCH
        assert result.target.target_value == 200.0  # 200ms target
        assert result.measured_value <= 200.0, f"Context search P95 latency {result.measured_value}ms exceeds 200ms target"
        assert result.success_rate >= 0.95, f"Context search success rate {result.success_rate:.2%} below 95%"
        
        # Validate search quality metrics
        assert result.additional_metrics["test_contexts_created"] == 100
        assert result.additional_metrics["avg_results_per_query"] >= 3.0
        assert result.additional_metrics["search_accuracy_score"] >= 0.80
        assert result.additional_metrics["index_performance"] == "optimized"
        
        logger.info(f"✅ Context search performance: {result.measured_value:.1f}ms P95, {result.additional_metrics['search_accuracy_score']:.2f} accuracy")
    
    @pytest.mark.parametrize("concurrent_agents", [10, 25, 50, 75])
    async def test_multi_agent_coordination_scalability(self, concurrent_agents):
        """Test multi-agent coordination scalability at different levels."""
        validator = IntegratedSystemPerformanceValidator()
        await validator.initialize_system_components()
        
        # Test multi-agent coordination
        coordination_results = await validator._test_multi_agent_coordination(concurrent_agents)
        
        assert len(coordination_results) == 1
        result = coordination_results[0]
        
        # Validate coordination performance
        assert result.test_type == PerformanceTestType.MULTI_AGENT_COORDINATION
        assert result.additional_metrics["concurrent_agents_attempted"] == concurrent_agents
        assert result.success_rate >= 0.80, f"Multi-agent success rate {result.success_rate:.2%} below 80% for {concurrent_agents} agents"
        
        # Validate successful agent count meets minimum requirements
        successful_agents = result.additional_metrics["successful_agents"]
        min_expected = max(1, int(concurrent_agents * 0.8))  # At least 80% success
        assert successful_agents >= min_expected, f"Only {successful_agents}/{concurrent_agents} agents successful"
        
        # Validate memory efficiency per agent
        memory_per_agent = result.additional_metrics["memory_per_agent_mb"]
        assert memory_per_agent <= 40.0, f"Memory per agent {memory_per_agent:.1f}MB too high"
        
        # Validate coordination latency
        avg_coordination_latency = result.additional_metrics["avg_coordination_latency_ms"]
        assert avg_coordination_latency <= 1000.0, f"Coordination latency {avg_coordination_latency:.1f}ms too high"
        
        logger.info(f"✅ Multi-agent coordination: {successful_agents}/{concurrent_agents} agents, {memory_per_agent:.1f}MB/agent")
    
    async def test_redis_message_throughput_performance(self):
        """Test Redis Streams message throughput meets >10k msgs/sec target."""
        validator = IntegratedSystemPerformanceValidator()
        await validator.initialize_system_components()
        
        # Test Redis message throughput
        redis_results = await validator._test_redis_message_throughput()
        
        assert len(redis_results) == 1
        result = redis_results[0]
        
        # Validate throughput performance
        assert result.test_type == PerformanceTestType.REDIS_MESSAGE_THROUGHPUT
        assert result.target.target_value == 10000.0  # 10k msgs/sec target
        assert result.measured_value >= 10000.0, f"Redis throughput {result.measured_value:.0f} msgs/sec below 10k target"
        assert result.success_rate >= 0.95, f"Redis message success rate {result.success_rate:.2%} below 95%"
        
        # Validate message performance characteristics
        messages_sent = result.additional_metrics["messages_sent"]
        test_duration = result.additional_metrics["test_duration_seconds"]
        avg_latency = result.additional_metrics["avg_send_latency_ms"]
        
        assert messages_sent > 50000, f"Only {messages_sent} messages sent in test"
        assert test_duration <= 15.0, f"Test took {test_duration:.1f}s, should be ~10s"
        assert avg_latency <= 1.0, f"Average send latency {avg_latency:.2f}ms too high"
        
        logger.info(f"✅ Redis throughput: {result.measured_value:.0f} msgs/sec, {avg_latency:.2f}ms avg latency")
    
    async def test_end_to_end_workflow_performance(self):
        """Test complete end-to-end workflow performance."""
        validator = IntegratedSystemPerformanceValidator()
        await validator.initialize_system_components()
        
        # Test end-to-end workflows
        workflow_results = await validator._test_end_to_end_workflows(iterations=15)
        
        assert len(workflow_results) == 1
        result = workflow_results[0]
        
        # Validate workflow performance
        assert result.test_type == PerformanceTestType.END_TO_END_WORKFLOW
        assert result.target.target_value == 5000.0  # 5s target
        assert result.measured_value <= 5000.0, f"Workflow P95 latency {result.measured_value:.0f}ms exceeds 5s target"
        assert result.success_rate >= 0.90, f"Workflow success rate {result.success_rate:.2%} below 90%"
        
        # Validate workflow completeness
        assert result.additional_metrics["workflow_steps_completed"] == 5
        assert result.additional_metrics["avg_workflow_latency_ms"] <= result.measured_value
        
        fastest_workflow = result.additional_metrics["fastest_workflow_ms"]
        slowest_workflow = result.additional_metrics["slowest_workflow_ms"]
        assert fastest_workflow > 0
        assert slowest_workflow <= 5000.0
        
        logger.info(f"✅ End-to-end workflow: {result.measured_value:.0f}ms P95, {result.success_rate:.1%} success rate")
    
    async def test_database_operations_performance(self):
        """Test database operations performance under load."""
        validator = IntegratedSystemPerformanceValidator()
        await validator.initialize_system_components()
        
        # Test database operations
        db_results = await validator._test_database_operations_performance(iterations=20)
        
        assert len(db_results) == 1
        result = db_results[0]
        
        # Validate database performance
        assert result.test_type == PerformanceTestType.DATABASE_OPERATIONS
        assert result.measured_value <= 100.0, f"Database P95 latency {result.measured_value:.1f}ms exceeds 100ms target"
        assert result.success_rate == 1.0, f"Database operations had {result.error_count} errors"
        
        # Validate operation coverage and performance
        assert result.additional_metrics["operations_tested"] == 4  # INSERT, SELECT, UPDATE, DELETE
        assert result.additional_metrics["total_operations"] == 80  # 20 iterations * 4 operations
        
        # Validate individual operation performance
        assert result.additional_metrics["select_avg_ms"] <= 50.0
        assert result.additional_metrics["insert_avg_ms"] <= 80.0
        assert result.additional_metrics["update_avg_ms"] <= 60.0
        assert result.additional_metrics["delete_avg_ms"] <= 40.0
        
        logger.info(f"✅ Database performance: {result.measured_value:.1f}ms P95, all operations within targets")
    
    @pytest.mark.parametrize("concurrent_level", [10, 25, 50])
    async def test_system_scalability_under_load(self, concurrent_level):
        """Test system scalability at different concurrent load levels."""
        validator = IntegratedSystemPerformanceValidator()
        await validator.initialize_system_components()
        
        # Test system scalability
        scalability_results = await validator._test_system_scalability([concurrent_level])
        
        assert len(scalability_results) == 1
        result = scalability_results[0]
        
        # Validate scalability performance
        assert result.test_type == PerformanceTestType.SYSTEM_SCALABILITY
        assert result.target.target_value == 0.8  # 80% efficiency target
        
        efficiency_ratio = result.measured_value
        success_rate = result.success_rate
        
        # Efficiency should remain reasonable under load
        min_efficiency = 0.6 if concurrent_level >= 50 else 0.7
        assert efficiency_ratio >= min_efficiency, f"Efficiency {efficiency_ratio:.2f} too low at {concurrent_level} concurrent"
        assert success_rate >= 0.80, f"Success rate {success_rate:.2%} too low at {concurrent_level} concurrent"
        
        # Validate resource utilization
        resource_utilization = result.additional_metrics["resource_utilization"]
        throughput_degradation = result.additional_metrics["throughput_degradation"]
        
        assert resource_utilization <= 50.0, f"Resource utilization {resource_utilization:.1f}MB per operation too high"
        assert throughput_degradation <= 0.30, f"Throughput degradation {throughput_degradation:.1%} too high"
        
        logger.info(f"✅ Scalability at {concurrent_level} concurrent: {efficiency_ratio:.2f} efficiency, {success_rate:.1%} success")
    
    async def test_memory_efficiency_validation(self):
        """Test system memory efficiency meets <2GB target for 50 agents."""
        validator = IntegratedSystemPerformanceValidator()
        await validator.initialize_system_components()
        
        # Test memory efficiency
        memory_results = await validator._test_memory_efficiency()
        
        assert len(memory_results) == 1
        result = memory_results[0]
        
        # Validate memory efficiency
        assert result.test_type == PerformanceTestType.MEMORY_EFFICIENCY
        assert result.target.target_value == 2048.0  # 2GB target
        assert result.measured_value <= 2048.0, f"Peak memory {result.measured_value:.0f}MB exceeds 2GB target"
        assert result.success_rate == 1.0, "Memory efficiency test should not have errors"
        
        # Validate memory characteristics
        initial_memory = result.additional_metrics["initial_memory_mb"]
        peak_memory = result.additional_metrics["peak_memory_mb"]
        memory_growth = result.additional_metrics["memory_growth_mb"]
        efficiency_score = result.additional_metrics["memory_efficiency_score"]
        
        assert memory_growth <= 500.0, f"Memory growth {memory_growth:.0f}MB too high"
        assert efficiency_score >= 0.5, f"Memory efficiency score {efficiency_score:.2f} too low"
        assert peak_memory > initial_memory, "Peak memory should be higher than initial"
        
        logger.info(f"✅ Memory efficiency: {peak_memory:.0f}MB peak, {memory_growth:.0f}MB growth, {efficiency_score:.2f} efficiency")
    
    async def test_comprehensive_performance_validation(self):
        """Test complete comprehensive performance validation suite."""
        # Run comprehensive validation with realistic parameters
        report = await run_integrated_performance_validation(
            test_iterations=5,  # Reduced for test performance
            concurrent_levels=[1, 10, 25],  # Subset of levels
            include_scalability=True
        )
        
        # Validate report structure
        assert isinstance(report, SystemPerformanceReport)
        assert report.validation_id is not None
        assert len(report.test_results) >= 8  # Should have results from all major test types
        
        # Validate overall performance score
        assert 0 <= report.overall_score <= 100
        assert isinstance(report.critical_failures, list)
        assert isinstance(report.performance_warnings, list)
        assert isinstance(report.optimization_recommendations, list)
        
        # Validate production readiness assessment
        readiness = report.production_readiness
        assert readiness["status"] in ["PRODUCTION_READY", "MOSTLY_READY", "NEEDS_OPTIMIZATION", "NOT_READY"]
        assert 0 <= readiness["overall_readiness_score"] <= 1.0
        assert readiness["critical_tests_total"] >= 4  # Should have critical tests
        assert readiness["deployment_recommendation"] is not None
        
        # Validate system metrics
        assert "initial" in report.system_metrics
        assert "final" in report.system_metrics
        assert "resource_delta" in report.system_metrics
        
        # Validate execution summary
        summary = report.execution_summary
        assert summary["total_tests"] >= 8
        assert summary["total_tests"] == summary["passed_tests"] + summary["failed_tests"]
        assert 0 <= summary["overall_score"] <= 100
        assert summary["test_duration_seconds"] > 0
        
        logger.info(f"✅ Comprehensive validation: {report.overall_score:.1f}% score, {readiness['status']} readiness")
        
        # Log critical failures for debugging
        if report.critical_failures:
            logger.warning(f"Critical failures detected: {report.critical_failures}")
        
        # Should have minimal critical failures for a well-performing system
        assert len(report.critical_failures) <= 2, f"Too many critical failures: {report.critical_failures}"
    
    async def test_quick_production_readiness_check(self):
        """Test quick production readiness check functionality."""
        readiness_result = await quick_production_readiness_check()
        
        # Validate readiness result structure
        assert isinstance(readiness_result, dict)
        assert "production_ready" in readiness_result
        assert "readiness_status" in readiness_result
        assert "overall_score" in readiness_result
        assert "critical_failures" in readiness_result
        assert "recommendations" in readiness_result
        
        # Validate data types
        assert isinstance(readiness_result["production_ready"], bool)
        assert readiness_result["readiness_status"] in ["PRODUCTION_READY", "MOSTLY_READY", "NEEDS_OPTIMIZATION", "NOT_READY"]
        assert 0 <= readiness_result["overall_score"] <= 100
        assert isinstance(readiness_result["critical_failures"], list)
        assert isinstance(readiness_result["recommendations"], list)
        assert len(readiness_result["recommendations"]) <= 5  # Top 5 recommendations
        
        logger.info(f"✅ Quick readiness check: {readiness_result['readiness_status']}, {readiness_result['overall_score']:.1f}% score")
    
    @pytest.mark.slow
    async def test_sustained_load_performance(self):
        """Test system performance under sustained load (longer test)."""
        validator = IntegratedSystemPerformanceValidator()
        await validator.initialize_system_components()
        
        # Run sustained load test
        start_time = time.time()
        sustained_duration = 30  # 30 second sustained test
        
        performance_samples = []
        memory_samples = []
        
        while time.time() - start_time < sustained_duration:
            # Simulate sustained operations
            sample_start = time.perf_counter()
            
            # Run a mix of operations
            await validator._simulate_agent_workflow("sustained_test_agent")
            await asyncio.sleep(0.01)  # Small delay between operations
            
            sample_end = time.perf_counter()
            operation_time_ms = (sample_end - sample_start) * 1000
            performance_samples.append(operation_time_ms)
            
            # Sample memory usage
            current_memory = validator.process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
            # Brief pause between samples
            await asyncio.sleep(0.05)
        
        # Analyze sustained performance
        avg_operation_time = statistics.mean(performance_samples)
        p95_operation_time = sorted(performance_samples)[int(len(performance_samples) * 0.95)]
        
        avg_memory = statistics.mean(memory_samples)
        peak_memory = max(memory_samples)
        memory_growth = peak_memory - memory_samples[0]
        
        # Validate sustained performance
        assert avg_operation_time <= 500.0, f"Average operation time {avg_operation_time:.1f}ms too high under sustained load"
        assert p95_operation_time <= 1000.0, f"P95 operation time {p95_operation_time:.1f}ms too high under sustained load"
        assert memory_growth <= 200.0, f"Memory growth {memory_growth:.1f}MB too high during sustained load"
        assert peak_memory <= 1000.0, f"Peak memory {peak_memory:.1f}MB too high during sustained load"
        
        # Validate performance stability (low variance)
        operation_std_dev = statistics.stdev(performance_samples)
        memory_std_dev = statistics.stdev(memory_samples)
        
        assert operation_std_dev <= 200.0, f"Operation time variance {operation_std_dev:.1f}ms too high"
        assert memory_std_dev <= 50.0, f"Memory usage variance {memory_std_dev:.1f}MB too high"
        
        logger.info(f"✅ Sustained load: {avg_operation_time:.1f}ms avg, {peak_memory:.0f}MB peak, {len(performance_samples)} operations")
    
    async def test_performance_regression_detection(self):
        """Test performance regression detection capabilities."""
        validator = IntegratedSystemPerformanceValidator()
        await validator.initialize_system_components()
        
        # Run baseline performance test
        baseline_results = await validator._test_authentication_performance(iterations=20)
        baseline_result = baseline_results[0]
        baseline_latency = baseline_result.measured_value
        
        # Simulate performance regression by introducing delays
        with patch.object(asyncio, 'sleep', new_callable=AsyncMock) as mock_sleep:
            # Mock increased sleep times to simulate regression
            async def slow_sleep(delay):
                await asyncio.sleep(delay * 2)  # Double the delay
            
            mock_sleep.side_effect = slow_sleep
            
            # Run performance test with regression
            regression_results = await validator._test_authentication_performance(iterations=20)
            regression_result = regression_results[0]
            regression_latency = regression_result.measured_value
        
        # Validate regression detection
        performance_degradation = (regression_latency - baseline_latency) / baseline_latency
        
        # Should detect significant performance regression
        assert performance_degradation > 0.50, f"Should detect >50% performance regression, got {performance_degradation:.1%}"
        assert regression_result.meets_target != baseline_result.meets_target or regression_latency > baseline_latency * 1.5
        
        logger.info(f"✅ Regression detection: {performance_degradation:.1%} degradation detected")
    
    async def test_concurrent_test_execution(self):
        """Test that multiple performance tests can run concurrently without interference."""
        validator = IntegratedSystemPerformanceValidator()
        await validator.initialize_system_components()
        
        # Run multiple test types concurrently
        test_tasks = [
            validator._test_authentication_performance(iterations=10),
            validator._test_authorization_performance(iterations=10),
            validator._test_context_search_performance(iterations=10),
            validator._test_multi_agent_coordination(concurrent_agents=15)
        ]
        
        # Execute all tests concurrently
        concurrent_results = await asyncio.gather(*test_tasks, return_exceptions=True)
        
        # Validate all tests completed successfully
        for i, result in enumerate(concurrent_results):
            assert not isinstance(result, Exception), f"Test {i} failed with: {result}"
            assert isinstance(result, list), f"Test {i} should return list of results"
            assert len(result) >= 1, f"Test {i} should have at least one result"
        
        # Validate test results are reasonable despite concurrent execution
        auth_result = concurrent_results[0][0]
        authz_result = concurrent_results[1][0]
        context_result = concurrent_results[2][0]
        coordination_result = concurrent_results[3][0]
        
        # Performance should still be within reasonable bounds
        assert auth_result.measured_value <= 100.0, "Authentication performance degraded under concurrent execution"
        assert authz_result.measured_value <= 200.0, "Authorization performance degraded under concurrent execution"
        assert context_result.measured_value <= 400.0, "Context search performance degraded under concurrent execution"
        assert coordination_result.success_rate >= 0.70, "Multi-agent coordination degraded under concurrent execution"
        
        logger.info("✅ Concurrent test execution: All tests completed successfully without interference")


@pytest.mark.performance
class TestPerformanceTargetValidation:
    """Test suite for validating specific performance targets."""
    
    def test_performance_target_definitions(self):
        """Test that performance targets are properly defined."""
        validator = IntegratedSystemPerformanceValidator()
        
        # Validate all critical targets are defined
        critical_targets = [t for t in validator.performance_targets if t.severity == TestSeverity.CRITICAL]
        
        # Should have critical targets for core systems
        critical_target_names = [t.name for t in critical_targets]
        expected_critical_targets = [
            "authentication_latency",
            "authorization_decision_latency", 
            "context_semantic_search_latency",
            "multi_agent_coordination_capacity"
        ]
        
        for expected_target in expected_critical_targets:
            assert expected_target in critical_target_names, f"Missing critical target: {expected_target}"
        
        # Validate target values are reasonable
        auth_target = next(t for t in validator.performance_targets if t.name == "authentication_latency")
        assert auth_target.target_value == 50.0
        assert auth_target.unit == "ms"
        
        authz_target = next(t for t in validator.performance_targets if t.name == "authorization_decision_latency")
        assert authz_target.target_value == 100.0
        assert authz_target.unit == "ms"
        
        context_target = next(t for t in validator.performance_targets if t.name == "context_semantic_search_latency")
        assert context_target.target_value == 200.0
        assert context_target.unit == "ms"
        
        agents_target = next(t for t in validator.performance_targets if t.name == "multi_agent_coordination_capacity")
        assert agents_target.target_value == 50.0
        assert agents_target.unit == "agents"
    
    def test_performance_validation_methods(self):
        """Test that validation methods are appropriate for each target."""
        validator = IntegratedSystemPerformanceValidator()
        
        for target in validator.performance_targets:
            # Validate validation method is appropriate
            valid_methods = ["p95_latency", "average", "throughput", "peak_usage", "efficiency_ratio"]
            assert target.validation_method in valid_methods, f"Invalid validation method: {target.validation_method}"
            
            # Validate severity levels
            assert target.severity in [TestSeverity.CRITICAL, TestSeverity.HIGH, TestSeverity.MEDIUM, TestSeverity.LOW]
            
            # Validate target values are positive
            assert target.target_value > 0, f"Target value must be positive: {target.name} = {target.target_value}"
            
            # Validate units are specified
            assert target.unit is not None and len(target.unit) > 0, f"Unit not specified for: {target.name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])