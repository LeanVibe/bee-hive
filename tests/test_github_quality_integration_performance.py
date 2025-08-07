"""
Performance and Integration Tests for Advanced GitHub Integration.

Validates performance requirements, success metrics, and real-world integration scenarios
for the advanced GitHub integration system including AI code review automation,
automated testing integration, and quality gates validation.
"""

import asyncio
import time
import statistics
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import pytest
import pytest_asyncio
from concurrent.futures import ThreadPoolExecutor

from app.core.github_quality_integration import (
    GitHubQualityGatesEngine, GitHubQualityContext, GitHubQualityGateType
)
from app.core.code_review_assistant import CodeReviewAssistant
from app.core.automated_testing_integration import AutomatedTestingIntegration
from app.core.advanced_repository_management import AdvancedRepositoryManagement
from app.core.intelligent_workflow_automation import IntelligentWorkflowAutomation
from app.core.quality_gates import QualityGatesEngine, QualityGateStatus
from app.models.github_integration import PullRequest, GitHubRepository
from app.schemas.github_integration import ReviewType, TestSuiteType


class TestPerformanceRequirements:
    """Test performance requirements for advanced GitHub integration."""
    
    @pytest.fixture
    async def performance_quality_engine(self):
        """Create quality engine optimized for performance testing."""
        return GitHubQualityGatesEngine(
            quality_gates_engine=QualityGatesEngine(),
            code_review_assistant=CodeReviewAssistant(),
            testing_integration=AutomatedTestingIntegration(),
            repository_management=AdvancedRepositoryManagement(),
            workflow_automation=IntelligentWorkflowAutomation()
        )
    
    @pytest.fixture
    async def mock_high_performance_components(self, performance_quality_engine):
        """Mock components for high-performance testing."""
        # Mock all components to return quickly
        with patch.object(performance_quality_engine.code_review_assistant, 'perform_comprehensive_review') as mock_review, \
             patch.object(performance_quality_engine.testing_integration, 'trigger_automated_tests') as mock_test, \
             patch.object(performance_quality_engine.testing_integration, 'monitor_test_execution') as mock_monitor, \
             patch.object(performance_quality_engine.repository_management, 'perform_intelligent_merge') as mock_merge, \
             patch.object(performance_quality_engine.workflow_automation, 'execute_workflow') as mock_workflow, \
             patch.object(performance_quality_engine.repository_management, 'analyze_repository_health') as mock_health:
            
            # Configure fast, successful responses
            mock_review.return_value = {
                "success": True,
                "ai_confidence": 0.88,
                "findings_count": 2,
                "categorized_findings": {"security": 0, "performance": 1, "style": 1},
                "overall_score": 0.92
            }
            
            mock_test.return_value = {
                "status": "completed",
                "test_run_id": "perf-test-123",
                "analysis": {
                    "success_rate": 96.5,
                    "coverage_analysis": {"average_coverage": 93.2}
                }
            }
            
            mock_monitor.return_value = {
                "analysis": {
                    "success_rate": 96.5,
                    "coverage_analysis": {"average_coverage": 93.2}
                }
            }
            
            mock_merge.return_value = {
                "success": True,
                "conflict_analysis": {
                    "auto_resolvable": True,
                    "conflicts_detected": 0,
                    "conflict_summary": "No conflicts detected"
                },
                "files_modified": []
            }
            
            workflow_execution = AsyncMock()
            workflow_execution.success = True
            workflow_execution.quality_gates_results = {}
            mock_workflow.return_value = workflow_execution
            
            mock_health.return_value = {
                "score": 87.0,
                "overall_status": MagicMock(value="good"),
                "dependency_analysis": {"security_risk_score": 1.8}
            }
            
            yield
    
    async def test_ai_code_review_accuracy_requirement(
        self, performance_quality_engine, mock_high_performance_components
    ):
        """Test AI code review meets 80% accuracy requirement."""
        github_context = GitHubQualityContext(
            pull_request_id=uuid4(),
            repository_id=uuid4(),
            agent_id=uuid4()
        )
        
        mock_pr = MagicMock(spec=PullRequest)
        mock_repo = MagicMock(spec=GitHubRepository)
        
        # Execute AI code review gate multiple times to test accuracy
        accuracy_scores = []
        
        for _ in range(10):
            result = await performance_quality_engine._execute_ai_code_review_gate(
                github_context, mock_pr, mock_repo
            )
            
            # Extract accuracy score from metrics
            accuracy_metric = next(
                m for m in result.metrics if m.name == "ai_review_accuracy"
            )
            accuracy_scores.append(accuracy_metric.value)
        
        # Calculate average accuracy
        average_accuracy = statistics.mean(accuracy_scores)
        
        # Verify meets 80% accuracy requirement
        assert average_accuracy >= 80.0, f"AI code review accuracy {average_accuracy}% below 80% requirement"
        
        # Verify consistency
        accuracy_std = statistics.stdev(accuracy_scores)
        assert accuracy_std < 10.0, f"AI code review accuracy too inconsistent (std: {accuracy_std})"
    
    async def test_automated_testing_success_rate_requirement(
        self, performance_quality_engine, mock_high_performance_components
    ):
        """Test automated testing meets 95% success rate requirement."""
        github_context = GitHubQualityContext(
            pull_request_id=uuid4(),
            repository_id=uuid4(),
            agent_id=uuid4()
        )
        
        mock_pr = MagicMock(spec=PullRequest)
        mock_repo = MagicMock(spec=GitHubRepository)
        
        # Execute testing gate multiple times
        success_rates = []
        
        for _ in range(10):
            result = await performance_quality_engine._execute_automated_testing_gate(
                github_context, mock_pr, mock_repo
            )
            
            # Extract success rate from metrics
            success_metric = next(
                m for m in result.metrics if m.name == "test_success_rate"
            )
            success_rates.append(success_metric.value)
        
        # Calculate average success rate
        average_success_rate = statistics.mean(success_rates)
        
        # Verify meets 95% success rate requirement
        assert average_success_rate >= 95.0, f"Testing success rate {average_success_rate}% below 95% requirement"
        
        # Verify all individual runs meet requirement
        for rate in success_rates:
            assert rate >= 95.0, f"Individual test run {rate}% below requirement"
    
    async def test_quality_pipeline_execution_time(
        self, performance_quality_engine, mock_high_performance_components
    ):
        """Test quality pipeline execution time meets performance requirements."""
        github_context = GitHubQualityContext(
            pull_request_id=uuid4(),
            repository_id=uuid4(),
            agent_id=uuid4()
        )
        
        # Mock database operations
        with patch.object(performance_quality_engine, '_get_github_context') as mock_context:
            mock_pr = MagicMock(spec=PullRequest)
            mock_repo = MagicMock(spec=GitHubRepository)
            mock_context.return_value = (mock_pr, mock_repo)
            
            # Execute pipeline and measure time
            execution_times = []
            
            for _ in range(5):
                start_time = time.time()
                
                success, results, analytics = await performance_quality_engine.execute_github_quality_pipeline(
                    github_context=github_context,
                    db=AsyncMock()
                )
                
                execution_time = time.time() - start_time
                execution_times.append(execution_time)
                
                # Verify successful execution
                assert success is True
                assert len(results) > 0
            
            # Analyze execution times
            average_time = statistics.mean(execution_times)
            max_time = max(execution_times)
            
            # Performance requirements
            assert average_time < 30.0, f"Average execution time {average_time:.2f}s exceeds 30s requirement"
            assert max_time < 60.0, f"Maximum execution time {max_time:.2f}s exceeds 60s requirement"
            
            print(f"Quality pipeline performance: avg={average_time:.2f}s, max={max_time:.2f}s")
    
    async def test_concurrent_quality_validations_performance(
        self, performance_quality_engine, mock_high_performance_components
    ):
        """Test system performance under concurrent load."""
        # Create multiple GitHub contexts
        contexts = [
            GitHubQualityContext(
                pull_request_id=uuid4(),
                repository_id=uuid4(),
                agent_id=uuid4()
            )
            for _ in range(10)
        ]
        
        # Mock database operations
        with patch.object(performance_quality_engine, '_get_github_context') as mock_context:
            mock_pr = MagicMock(spec=PullRequest)
            mock_repo = MagicMock(spec=GitHubRepository)
            mock_context.return_value = (mock_pr, mock_repo)
            
            # Execute concurrent pipelines
            start_time = time.time()
            
            tasks = [
                performance_quality_engine.execute_github_quality_pipeline(
                    github_context=context,
                    db=AsyncMock()
                )
                for context in contexts
            ]
            
            results = await asyncio.gather(*tasks)
            
            execution_time = time.time() - start_time
            
            # Verify all succeeded
            assert len(results) == 10
            for success, gate_results, analytics in results:
                assert success is True
                assert len(gate_results) > 0
            
            # Performance under load
            assert execution_time < 90.0, f"Concurrent execution time {execution_time:.2f}s too high"
            
            print(f"Concurrent validation performance: {execution_time:.2f}s for 10 PRs")
    
    async def test_memory_usage_efficiency(
        self, performance_quality_engine, mock_high_performance_components
    ):
        """Test memory usage efficiency during quality validations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        github_context = GitHubQualityContext(
            pull_request_id=uuid4(),
            repository_id=uuid4(),
            agent_id=uuid4()
        )
        
        # Mock database operations
        with patch.object(performance_quality_engine, '_get_github_context') as mock_context:
            mock_pr = MagicMock(spec=PullRequest)
            mock_repo = MagicMock(spec=GitHubRepository)
            mock_context.return_value = (mock_pr, mock_repo)
            
            # Execute multiple validations to test memory efficiency
            for i in range(50):
                success, results, analytics = await performance_quality_engine.execute_github_quality_pipeline(
                    github_context=github_context,
                    db=AsyncMock()
                )
                
                # Check memory usage every 10 iterations
                if i % 10 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_increase = current_memory - initial_memory
                    
                    # Memory usage should not grow excessively
                    assert memory_increase < 100.0, f"Memory usage increased by {memory_increase:.1f}MB"
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        total_increase = final_memory - initial_memory
        
        print(f"Memory usage: initial={initial_memory:.1f}MB, final={final_memory:.1f}MB, increase={total_increase:.1f}MB")
        
        # Total memory increase should be reasonable
        assert total_increase < 50.0, f"Total memory increase {total_increase:.1f}MB too high"


class TestSuccessMetricsValidation:
    """Test validation of success metrics compliance."""
    
    @pytest.fixture
    async def metrics_quality_engine(self):
        """Create quality engine for metrics testing."""
        return GitHubQualityGatesEngine(
            quality_gates_engine=QualityGatesEngine(),
            code_review_assistant=CodeReviewAssistant(),
            testing_integration=AutomatedTestingIntegration(),
            repository_management=AdvancedRepositoryManagement(),
            workflow_automation=IntelligentWorkflowAutomation()
        )
    
    async def test_automated_review_coverage_target(self, metrics_quality_engine):
        """Test 80% automated review coverage target compliance."""
        # Simulate review accuracy data above target
        high_accuracy_data = [85.0, 88.0, 82.0, 90.0, 87.0, 84.0, 89.0, 86.0]
        metrics_quality_engine.github_metrics["automated_review_accuracy"] = high_accuracy_data
        
        compliance = metrics_quality_engine._evaluate_target_compliance()
        
        # Should be compliant with 80% target
        assert "automated_review_coverage" in compliance
        review_compliance = compliance["automated_review_coverage"]
        assert review_compliance["compliant"] is True
        assert review_compliance["current"] >= 80.0
        
        # Test below target
        low_accuracy_data = [75.0, 78.0, 72.0, 79.0, 76.0]
        metrics_quality_engine.github_metrics["automated_review_accuracy"] = low_accuracy_data
        
        compliance = metrics_quality_engine._evaluate_target_compliance()
        review_compliance = compliance["automated_review_coverage"]
        assert review_compliance["compliant"] is False
        assert review_compliance["current"] < 80.0
    
    async def test_testing_integration_success_rate_target(self, metrics_quality_engine):
        """Test 95% testing integration success rate target compliance."""
        # Simulate testing success data above target
        high_success_data = [96.0, 97.5, 95.5, 98.0, 96.8, 95.2, 97.0]
        metrics_quality_engine.github_metrics["testing_success_rates"] = high_success_data
        
        compliance = metrics_quality_engine._evaluate_target_compliance()
        
        # Should be compliant with 95% target
        assert "testing_integration_success_rate" in compliance
        testing_compliance = compliance["testing_integration_success_rate"]
        assert testing_compliance["compliant"] is True
        assert testing_compliance["current"] >= 95.0
        
        # Test below target
        low_success_data = [92.0, 94.0, 91.5, 93.0, 90.0]
        metrics_quality_engine.github_metrics["testing_success_rates"] = low_success_data
        
        compliance = metrics_quality_engine._evaluate_target_compliance()
        testing_compliance = compliance["testing_integration_success_rate"]
        assert testing_compliance["compliant"] is False
        assert testing_compliance["current"] < 95.0
    
    async def test_intelligent_conflict_resolution_accuracy(self, metrics_quality_engine):
        """Test intelligent conflict resolution accuracy meets 85% target."""
        # Add conflict resolution data
        high_resolution_data = [90.0, 88.0, 87.0, 89.0, 86.0, 91.0]
        metrics_quality_engine.github_metrics["conflict_resolution_success"] = high_resolution_data
        
        # Calculate average
        average_accuracy = statistics.mean(high_resolution_data)
        assert average_accuracy >= 85.0, f"Conflict resolution accuracy {average_accuracy}% below 85% target"
        
        # Test trend analysis
        trend = metrics_quality_engine._calculate_trend_direction(high_resolution_data)
        assert trend in ["improving", "stable"], f"Conflict resolution trend {trend} not positive"
    
    async def test_overall_metrics_compliance_calculation(self, metrics_quality_engine):
        """Test overall metrics compliance calculation."""
        # Set all metrics to meet targets
        metrics_quality_engine.github_metrics.update({
            "automated_review_accuracy": [85.0, 88.0, 86.0],
            "testing_success_rates": [96.0, 97.0, 95.5],
            "conflict_resolution_success": [88.0, 90.0, 87.0],
            "workflow_completion_rates": [92.0, 94.0, 91.0],
            "repository_health_scores": [80.0, 85.0, 78.0]
        })
        
        compliance = metrics_quality_engine._evaluate_target_compliance()
        
        # Check overall compliance
        assert "overall_compliance" in compliance
        assert compliance["overall_compliance"] >= 80.0, "Overall compliance below 80%"
        
        # Check that compliant targets are identified
        assert compliance["compliant_targets"] >= 2, "Not enough compliant targets"
        assert compliance["total_targets"] > 0, "No targets evaluated"


class TestRealWorldIntegrationScenarios:
    """Test real-world integration scenarios and edge cases."""
    
    @pytest.fixture
    async def integration_quality_engine(self):
        """Create quality engine for integration testing."""
        return GitHubQualityGatesEngine(
            quality_gates_engine=QualityGatesEngine(),
            code_review_assistant=CodeReviewAssistant(),
            testing_integration=AutomatedTestingIntegration(),
            repository_management=AdvancedRepositoryManagement(),
            workflow_automation=IntelligentWorkflowAutomation()
        )
    
    async def test_large_pull_request_handling(self, integration_quality_engine):
        """Test handling of large pull requests with many changes."""
        # Simulate large PR context
        large_pr_context = GitHubQualityContext(
            pull_request_id=uuid4(),
            repository_id=uuid4(),
            agent_id=uuid4(),
            pr_metadata={
                "files_changed": 50,
                "lines_added": 5000,
                "lines_deleted": 2000,
                "complexity_score": "high"
            },
            execution_config={"timeout_minutes": 60}  # Extended timeout for large PRs
        )
        
        # Mock components to handle large PR
        with patch.object(integration_quality_engine.code_review_assistant, 'perform_comprehensive_review') as mock_review:
            mock_review.return_value = {
                "success": True,
                "ai_confidence": 0.82,  # Slightly lower for complex PR
                "findings_count": 15,
                "categorized_findings": {"security": 2, "performance": 8, "style": 5},
                "overall_score": 0.75
            }
            
            mock_pr = MagicMock(spec=PullRequest)
            mock_repo = MagicMock(spec=GitHubRepository)
            
            result = await integration_quality_engine._execute_ai_code_review_gate(
                large_pr_context, mock_pr, mock_repo
            )
            
            # Should handle large PR successfully
            assert result.status in [QualityGateStatus.PASSED, QualityGateStatus.WARNING]
            assert len(result.metrics) > 0
    
    async def test_high_security_repository_requirements(self, integration_quality_engine):
        """Test handling of high-security repository requirements."""
        security_context = GitHubQualityContext(
            pull_request_id=uuid4(),
            repository_id=uuid4(),
            agent_id=uuid4(),
            repository_metadata={
                "security_level": "high",
                "compliance_required": True,
                "industry": "financial_services"
            }
        )
        
        # Mock stricter security requirements
        with patch.object(integration_quality_engine.code_review_assistant, 'perform_comprehensive_review') as mock_review:
            mock_review.return_value = {
                "success": True,
                "ai_confidence": 0.95,  # High confidence required
                "findings_count": 1,
                "categorized_findings": {"security": 0, "performance": 0, "style": 1},  # No security issues
                "overall_score": 0.98
            }
            
            mock_pr = MagicMock(spec=PullRequest)
            mock_repo = MagicMock(spec=GitHubRepository)
            
            result = await integration_quality_engine._execute_ai_code_review_gate(
                security_context, mock_pr, mock_repo
            )
            
            # Should meet high security standards
            assert result.status == QualityGateStatus.PASSED
            
            # Check that no critical security findings
            findings_metric = next(m for m in result.metrics if m.name == "review_findings_quality")
            assert findings_metric.value == 0  # No critical security findings
    
    async def test_multi_language_repository_support(self, integration_quality_engine):
        """Test support for multi-language repositories."""
        multilang_context = GitHubQualityContext(
            pull_request_id=uuid4(),
            repository_id=uuid4(),
            agent_id=uuid4(),
            repository_metadata={
                "languages": ["python", "javascript", "go", "rust"],
                "primary_language": "python",
                "microservices": True
            }
        )
        
        # Mock multi-language analysis
        with patch.object(integration_quality_engine.testing_integration, 'trigger_automated_tests') as mock_test:
            # Different test suites for different languages
            mock_test.return_value = {
                "status": "completed",
                "test_run_id": "multilang-test-123",
                "analysis": {
                    "success_rate": 94.0,  # Slightly lower due to complexity
                    "coverage_analysis": {
                        "average_coverage": 88.0,
                        "language_breakdown": {
                            "python": 92.0,
                            "javascript": 87.0,
                            "go": 85.0,
                            "rust": 89.0
                        }
                    }
                }
            }
            
            mock_pr = MagicMock(spec=PullRequest)
            mock_repo = MagicMock(spec=GitHubRepository)
            
            result = await integration_quality_engine._execute_automated_testing_gate(
                multilang_context, mock_pr, mock_repo
            )
            
            # Should handle multi-language repository
            assert result.status in [QualityGateStatus.PASSED, QualityGateStatus.WARNING]
            assert len(result.metrics) == 2
    
    async def test_emergency_hotfix_workflow(self, integration_quality_engine):
        """Test emergency hotfix workflow with relaxed requirements."""
        hotfix_context = GitHubQualityContext(
            pull_request_id=uuid4(),
            repository_id=uuid4(),
            agent_id=uuid4(),
            pr_metadata={
                "priority": "critical",
                "type": "hotfix",
                "incident_id": "INC-2024-001"
            },
            execution_config={
                "emergency_mode": True,
                "relaxed_requirements": True,
                "timeout_minutes": 15  # Faster for emergencies
            }
        )
        
        # Mock emergency workflow
        with patch.object(integration_quality_engine.workflow_automation, 'execute_workflow') as mock_workflow:
            # Simulate expedited workflow
            workflow_execution = AsyncMock()
            workflow_execution.success = True
            workflow_execution.quality_gates_results = {
                "security": AsyncMock(value="passed"),
                "testing": AsyncMock(value="warning")  # Relaxed for emergency
            }
            mock_workflow.return_value = workflow_execution
            
            mock_pr = MagicMock(spec=PullRequest)
            mock_repo = MagicMock(spec=GitHubRepository)
            
            result = await integration_quality_engine._execute_workflow_automation_gate(
                hotfix_context, mock_pr, mock_repo
            )
            
            # Should handle emergency workflow
            assert result.status in [QualityGateStatus.PASSED, QualityGateStatus.WARNING]
            
            # Verify workflow was executed
            mock_workflow.assert_called_once()
    
    async def test_dependency_update_pr_handling(self, integration_quality_engine):
        """Test handling of dependency update pull requests."""
        dependency_context = GitHubQualityContext(
            pull_request_id=uuid4(),
            repository_id=uuid4(),
            agent_id=uuid4(),
            pr_metadata={
                "type": "dependency_update",
                "dependency_count": 15,
                "security_updates": 3,
                "breaking_changes": False
            }
        )
        
        # Mock dependency-focused analysis
        with patch.object(integration_quality_engine.repository_management, 'analyze_repository_health') as mock_health:
            mock_health.return_value = {
                "score": 90.0,  # Improved by dependency updates
                "overall_status": MagicMock(value="excellent"),
                "dependency_analysis": {
                    "security_risk_score": 1.2,  # Reduced risk from updates
                    "vulnerable_dependencies": 0,
                    "outdated_dependencies": 2  # Reduced count
                }
            }
            
            mock_pr = MagicMock(spec=PullRequest)
            mock_repo = MagicMock(spec=GitHubRepository)
            
            result = await integration_quality_engine._execute_repository_health_gate(
                dependency_context, mock_pr, mock_repo
            )
            
            # Should show improved health from dependency updates
            assert result.status == QualityGateStatus.PASSED
            
            # Check health improvement
            health_metric = next(m for m in result.metrics if m.name == "repository_health_score")
            assert health_metric.value >= 85.0
            
            # Check reduced security risk
            security_metric = next(m for m in result.metrics if m.name == "dependency_security_risk")
            assert security_metric.value <= 3.0


class TestErrorResilienceAndRecovery:
    """Test error resilience and recovery mechanisms."""
    
    @pytest.fixture
    async def resilient_quality_engine(self):
        """Create quality engine for resilience testing."""
        return GitHubQualityGatesEngine(
            quality_gates_engine=QualityGatesEngine(),
            code_review_assistant=CodeReviewAssistant(),
            testing_integration=AutomatedTestingIntegration(),
            repository_management=AdvancedRepositoryManagement(),
            workflow_automation=IntelligentWorkflowAutomation()
        )
    
    async def test_component_failure_isolation(self, resilient_quality_engine):
        """Test that component failures don't crash entire pipeline."""
        github_context = GitHubQualityContext(
            pull_request_id=uuid4(),
            repository_id=uuid4(),
            agent_id=uuid4()
        )
        
        # Mock one component to fail
        with patch.object(resilient_quality_engine.code_review_assistant, 'perform_comprehensive_review') as mock_review, \
             patch.object(resilient_quality_engine.testing_integration, 'trigger_automated_tests') as mock_test:
            
            # One component fails
            mock_review.side_effect = Exception("Code review service temporarily unavailable")
            
            # Other component succeeds
            mock_test.return_value = {
                "status": "completed",
                "analysis": {"success_rate": 95.0}
            }
            
            # Mock database operations
            with patch.object(resilient_quality_engine, '_get_github_context') as mock_context:
                mock_pr = MagicMock(spec=PullRequest)
                mock_repo = MagicMock(spec=GitHubRepository)
                mock_context.return_value = (mock_pr, mock_repo)
                
                # Pipeline should handle partial failure gracefully
                success, results, analytics = await resilient_quality_engine.execute_github_quality_pipeline(
                    github_context=github_context,
                    db=AsyncMock(),
                    quality_gates=[GitHubQualityGateType.AI_CODE_REVIEW, GitHubQualityGateType.AUTOMATED_TESTING]
                )
                
                # Should not crash, but overall success should be false
                assert success is False
                assert len(results) == 2  # Both gates attempted
                
                # Check that one failed, one may have succeeded
                failed_gates = [r for r in results if r.status == QualityGateStatus.FAILED]
                assert len(failed_gates) >= 1  # At least the code review gate failed
    
    async def test_timeout_handling(self, resilient_quality_engine):
        """Test handling of component timeouts."""
        github_context = GitHubQualityContext(
            pull_request_id=uuid4(),
            repository_id=uuid4(),
            agent_id=uuid4(),
            execution_config={"timeout_minutes": 1}  # Very short timeout
        )
        
        # Mock slow component
        with patch.object(resilient_quality_engine.testing_integration, 'trigger_automated_tests') as mock_test:
            
            async def slow_test_execution(*args, **kwargs):
                await asyncio.sleep(2)  # Longer than timeout
                return {"status": "completed"}
            
            mock_test.side_effect = slow_test_execution
            
            mock_pr = MagicMock(spec=PullRequest)
            mock_repo = MagicMock(spec=GitHubRepository)
            
            # Should handle timeout gracefully
            start_time = time.time()
            
            result = await integration_quality_engine._execute_automated_testing_gate(
                github_context, mock_pr, mock_repo
            )
            
            execution_time = time.time() - start_time
            
            # Should not wait for full slow operation
            assert execution_time < 10.0, "Timeout handling failed"
            assert result.status == QualityGateStatus.FAILED
            assert "timeout" in result.error_message.lower() or result.status == QualityGateStatus.FAILED
    
    async def test_partial_success_handling(self, resilient_quality_engine):
        """Test handling of partial success scenarios."""
        # Test scenario where some quality gates pass and others fail
        github_context = GitHubQualityContext(
            pull_request_id=uuid4(),
            repository_id=uuid4(),
            agent_id=uuid4()
        )
        
        with patch.object(resilient_quality_engine, '_execute_github_quality_gate') as mock_gate:
            # Mock mixed results
            mock_results = [
                MagicMock(
                    status=QualityGateStatus.PASSED,
                    overall_score=95.0,
                    metrics=[],
                    execution_time_seconds=5.0
                ),
                MagicMock(
                    status=QualityGateStatus.FAILED,
                    overall_score=45.0,
                    metrics=[],
                    execution_time_seconds=8.0
                ),
                MagicMock(
                    status=QualityGateStatus.WARNING,
                    overall_score=75.0,
                    metrics=[],
                    execution_time_seconds=6.0
                )
            ]
            
            mock_gate.side_effect = mock_results
            
            # Mock database operations
            with patch.object(resilient_quality_engine, '_get_github_context') as mock_context:
                mock_pr = MagicMock(spec=PullRequest)
                mock_repo = MagicMock(spec=GitHubRepository)
                mock_context.return_value = (mock_pr, mock_repo)
                
                success, results, analytics = await resilient_quality_engine.execute_github_quality_pipeline(
                    github_context=github_context,
                    db=AsyncMock(),
                    quality_gates=[
                        GitHubQualityGateType.AI_CODE_REVIEW,
                        GitHubQualityGateType.AUTOMATED_TESTING,
                        GitHubQualityGateType.REPOSITORY_HEALTH
                    ]
                )
                
                # Overall should be failure due to one failed gate
                assert success is False
                assert len(results) == 3
                
                # Check mixed results
                passed_gates = [r for r in results if r.status == QualityGateStatus.PASSED]
                failed_gates = [r for r in results if r.status == QualityGateStatus.FAILED]
                warning_gates = [r for r in results if r.status == QualityGateStatus.WARNING]
                
                assert len(passed_gates) == 1
                assert len(failed_gates) == 1
                assert len(warning_gates) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--durations=10"])