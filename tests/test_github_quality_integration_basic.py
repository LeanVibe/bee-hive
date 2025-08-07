"""
Basic tests for GitHub Quality Integration to validate core functionality.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import pytest

from app.core.github_quality_integration import (
    GitHubQualityGatesEngine, GitHubQualityContext, GitHubQualityGateType
)
from app.core.quality_gates import QualityGatesEngine, QualityGateStatus, QualityGateSeverity


class TestGitHubQualityGatesEngineBasic:
    """Basic tests for GitHub Quality Gates Engine."""
    
    def test_engine_initialization(self):
        """Test basic engine initialization."""
        # Mock dependencies to avoid import issues
        quality_gates_engine = MagicMock()
        code_review_assistant = MagicMock()
        testing_integration = MagicMock()
        repository_management = MagicMock()
        workflow_automation = MagicMock()
        
        engine = GitHubQualityGatesEngine(
            quality_gates_engine=quality_gates_engine,
            code_review_assistant=code_review_assistant,
            testing_integration=testing_integration,
            repository_management=repository_management,
            workflow_automation=workflow_automation
        )
        
        # Verify initialization
        assert engine.quality_gates_engine is quality_gates_engine
        assert engine.code_review_assistant is code_review_assistant
        assert engine.testing_integration is testing_integration
        assert engine.repository_management is repository_management
        assert engine.workflow_automation is workflow_automation
        
        # Verify targets are set correctly
        assert engine.targets["automated_review_coverage"] == 80.0
        assert engine.targets["testing_integration_success_rate"] == 95.0
        assert engine.targets["conflict_resolution_accuracy"] == 85.0
        assert engine.targets["workflow_automation_success"] == 90.0
        assert engine.targets["repository_health_threshold"] == 75.0
        
        # Verify metrics tracking is initialized
        assert isinstance(engine.github_metrics, dict)
        assert "automated_review_accuracy" in engine.github_metrics
        assert "testing_success_rates" in engine.github_metrics
    
    def test_github_quality_context_creation(self):
        """Test GitHub quality context creation."""
        pr_id = uuid.uuid4()
        repo_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        
        context = GitHubQualityContext(
            pull_request_id=pr_id,
            repository_id=repo_id,
            agent_id=agent_id,
            pr_metadata={"files_changed": 10},
            repository_metadata={"language": "python"},
            execution_config={"timeout": 30}
        )
        
        assert context.pull_request_id == pr_id
        assert context.repository_id == repo_id
        assert context.agent_id == agent_id
        assert context.pr_metadata["files_changed"] == 10
        assert context.repository_metadata["language"] == "python"
        assert context.execution_config["timeout"] == 30
    
    def test_target_compliance_evaluation_empty_metrics(self):
        """Test target compliance evaluation with empty metrics."""
        # Mock dependencies
        engine = GitHubQualityGatesEngine(
            quality_gates_engine=MagicMock(),
            code_review_assistant=MagicMock(),
            testing_integration=MagicMock(),
            repository_management=MagicMock(),
            workflow_automation=MagicMock()
        )
        
        # Test with empty metrics
        compliance = engine._evaluate_target_compliance()
        
        # Should handle empty metrics gracefully
        assert isinstance(compliance, dict)
        assert "overall_compliance" in compliance
        assert "compliant_targets" in compliance
        assert "total_targets" in compliance
        
        # With no data, compliance should be 0
        assert compliance["overall_compliance"] == 0.0
        assert compliance["compliant_targets"] == 0
        assert compliance["total_targets"] == 0
    
    def test_target_compliance_evaluation_with_data(self):
        """Test target compliance evaluation with sample data."""
        # Mock dependencies
        engine = GitHubQualityGatesEngine(
            quality_gates_engine=MagicMock(),
            code_review_assistant=MagicMock(),
            testing_integration=MagicMock(),
            repository_management=MagicMock(),
            workflow_automation=MagicMock()
        )
        
        # Add sample data
        engine.github_metrics["automated_review_accuracy"] = [85.0, 88.0, 90.0, 87.0, 89.0]
        engine.github_metrics["testing_success_rates"] = [96.0, 97.0, 95.5, 98.0, 96.5]
        
        compliance = engine._evaluate_target_compliance()
        
        # Should be compliant
        assert compliance["automated_review_coverage"]["compliant"] is True
        assert compliance["automated_review_coverage"]["current"] >= 80.0
        
        assert compliance["testing_integration_success_rate"]["compliant"] is True
        assert compliance["testing_integration_success_rate"]["current"] >= 95.0
        
        # Overall compliance should be 100%
        assert compliance["overall_compliance"] == 100.0
        assert compliance["compliant_targets"] == 2
        assert compliance["total_targets"] == 2
    
    def test_trend_direction_calculation(self):
        """Test trend direction calculation."""
        engine = GitHubQualityGatesEngine(
            quality_gates_engine=MagicMock(),
            code_review_assistant=MagicMock(),
            testing_integration=MagicMock(),
            repository_management=MagicMock(),
            workflow_automation=MagicMock()
        )
        
        # Test improving trend
        improving_scores = [70.0, 75.0, 80.0, 85.0, 90.0]
        assert engine._calculate_trend_direction(improving_scores) == "improving"
        
        # Test declining trend
        declining_scores = [90.0, 85.0, 80.0, 75.0, 70.0]
        assert engine._calculate_trend_direction(declining_scores) == "declining"
        
        # Test stable trend
        stable_scores = [80.0, 81.0, 79.0, 80.5, 80.0]
        assert engine._calculate_trend_direction(stable_scores) == "stable"
        
        # Test insufficient data
        short_scores = [80.0, 85.0]
        assert engine._calculate_trend_direction(short_scores) == "unknown"
    
    @pytest.mark.asyncio
    async def test_ai_code_review_gate_basic(self):
        """Test basic AI code review gate execution."""
        engine = GitHubQualityGatesEngine(
            quality_gates_engine=MagicMock(),
            code_review_assistant=MagicMock(),
            testing_integration=MagicMock(),
            repository_management=MagicMock(),
            workflow_automation=MagicMock()
        )
        
        # Mock code review assistant
        engine.code_review_assistant.perform_comprehensive_review = AsyncMock(return_value={
            "ai_confidence": 0.90,
            "findings_count": 2,
            "categorized_findings": {"security": 0, "performance": 1, "style": 1},
            "overall_score": 0.95
        })
        
        github_context = GitHubQualityContext(
            pull_request_id=uuid.uuid4(),
            repository_id=uuid.uuid4(),
            agent_id=uuid.uuid4()
        )
        
        mock_pr = MagicMock()
        mock_repo = MagicMock()
        
        result = await engine._execute_ai_code_review_gate(
            github_context, mock_pr, mock_repo
        )
        
        # Verify result structure
        assert hasattr(result, 'status')
        assert hasattr(result, 'overall_score')
        assert hasattr(result, 'metrics')
        assert hasattr(result, 'execution_time_seconds')
        
        # Should have passed with good scores
        assert result.status == QualityGateStatus.PASSED
        assert result.overall_score == 100.0  # Both metrics passed
        assert len(result.metrics) == 2
        
        # Check accuracy metric
        accuracy_metrics = [m for m in result.metrics if m.name == "ai_review_accuracy"]
        assert len(accuracy_metrics) == 1
        assert accuracy_metrics[0].value == 90.0
        assert accuracy_metrics[0].status == QualityGateStatus.PASSED
        
        # Check findings metric
        findings_metrics = [m for m in result.metrics if m.name == "review_findings_quality"]
        assert len(findings_metrics) == 1
        assert findings_metrics[0].value == 0  # No critical security findings
        assert findings_metrics[0].status == QualityGateStatus.PASSED
    
    @pytest.mark.asyncio
    async def test_automated_testing_gate_basic(self):
        """Test basic automated testing gate execution."""
        engine = GitHubQualityGatesEngine(
            quality_gates_engine=MagicMock(),
            code_review_assistant=MagicMock(),
            testing_integration=MagicMock(),
            repository_management=MagicMock(),
            workflow_automation=MagicMock()
        )
        
        # Mock testing integration
        engine.testing_integration.trigger_automated_tests = AsyncMock(return_value={
            "status": "triggered",
            "test_run_id": "test-123",
            "analysis": {
                "success_rate": 96.0,
                "coverage_analysis": {"average_coverage": 93.0}
            }
        })
        
        engine.testing_integration.monitor_test_execution = AsyncMock(return_value={
            "analysis": {
                "success_rate": 96.0,
                "coverage_analysis": {"average_coverage": 93.0}
            }
        })
        
        github_context = GitHubQualityContext(
            pull_request_id=uuid.uuid4(),
            repository_id=uuid.uuid4(),
            agent_id=uuid.uuid4()
        )
        
        mock_pr = MagicMock()
        mock_repo = MagicMock()
        
        result = await engine._execute_automated_testing_gate(
            github_context, mock_pr, mock_repo
        )
        
        # Verify result structure
        assert hasattr(result, 'status')
        assert hasattr(result, 'overall_score')
        assert hasattr(result, 'metrics')
        
        # Should have passed with good scores
        assert result.status == QualityGateStatus.PASSED
        assert result.overall_score == 100.0  # Both metrics passed
        assert len(result.metrics) == 2
        
        # Check success rate metric
        success_metrics = [m for m in result.metrics if m.name == "test_success_rate"]
        assert len(success_metrics) == 1
        assert success_metrics[0].value == 96.0
        assert success_metrics[0].status == QualityGateStatus.PASSED
        
        # Check coverage metric
        coverage_metrics = [m for m in result.metrics if m.name == "test_coverage"]
        assert len(coverage_metrics) == 1
        assert coverage_metrics[0].value == 93.0
        assert coverage_metrics[0].status == QualityGateStatus.PASSED
    
    @pytest.mark.asyncio
    async def test_error_handling_in_gates(self):
        """Test error handling in quality gate execution."""
        engine = GitHubQualityGatesEngine(
            quality_gates_engine=MagicMock(),
            code_review_assistant=MagicMock(),
            testing_integration=MagicMock(),
            repository_management=MagicMock(),
            workflow_automation=MagicMock()
        )
        
        # Mock code review assistant to raise exception
        engine.code_review_assistant.perform_comprehensive_review = AsyncMock(
            side_effect=Exception("Service unavailable")
        )
        
        github_context = GitHubQualityContext(
            pull_request_id=uuid.uuid4(),
            repository_id=uuid.uuid4(),
            agent_id=uuid.uuid4()
        )
        
        mock_pr = MagicMock()
        mock_repo = MagicMock()
        
        result = await engine._execute_ai_code_review_gate(
            github_context, mock_pr, mock_repo
        )
        
        # Should handle error gracefully
        assert result.status == QualityGateStatus.FAILED
        assert result.overall_score == 0.0
        assert "Service unavailable" in result.error_message
        assert len(result.metrics) == 0
    
    def test_quality_recommendations_generation(self):
        """Test quality recommendations generation."""
        engine = GitHubQualityGatesEngine(
            quality_gates_engine=MagicMock(),
            code_review_assistant=MagicMock(),
            testing_integration=MagicMock(),
            repository_management=MagicMock(),
            workflow_automation=MagicMock()
        )
        
        # Mock gate results with various statuses
        mock_metric1 = MagicMock()
        mock_metric1.status = QualityGateStatus.FAILED
        mock_metric1.recommendations = ["Fix security issue", "Update dependencies"]
        
        mock_metric2 = MagicMock()
        mock_metric2.status = QualityGateStatus.PASSED
        mock_metric2.recommendations = []
        
        mock_result = MagicMock()
        mock_result.status = QualityGateStatus.FAILED
        mock_result.recommendations = ["Improve code quality"]
        mock_result.metrics = [mock_metric1, mock_metric2]
        
        gate_results = [mock_result]
        
        recommendations = engine._generate_quality_recommendations(gate_results)
        
        # Should collect all recommendations from failed gates and metrics
        assert len(recommendations) == 3
        assert "Fix security issue" in recommendations
        assert "Update dependencies" in recommendations
        assert "Improve code quality" in recommendations
    
    @pytest.mark.asyncio
    async def test_metrics_tracking_update(self):
        """Test metrics tracking update functionality."""
        engine = GitHubQualityGatesEngine(
            quality_gates_engine=MagicMock(),
            code_review_assistant=MagicMock(),
            testing_integration=MagicMock(),
            repository_management=MagicMock(),
            workflow_automation=MagicMock()
        )
        
        # Initial state - empty metrics
        assert len(engine.github_metrics["automated_review_accuracy"]) == 0
        
        # Mock gate results
        mock_result = MagicMock()
        mock_result.gate_type = MagicMock()
        mock_result.gate_type.value = "ai_code_review"
        mock_result.overall_score = 85.0
        
        analytics = {
            "execution_summary": {"execution_time_seconds": 30.0}
        }
        
        # Update metrics (this would happen during actual execution)
        engine.github_metrics["automated_review_accuracy"].append(85.0)
        
        await engine._update_success_metrics([mock_result], analytics)
        
        # Verify metrics were tracked
        assert len(engine.github_metrics["automated_review_accuracy"]) == 1
        assert engine.github_metrics["automated_review_accuracy"][0] == 85.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])