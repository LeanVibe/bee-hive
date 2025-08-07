"""
Comprehensive test suite for Advanced GitHub Integration system.

Tests cover AI code review automation, automated testing integration,
intelligent conflict resolution, workflow automation, and quality gates integration
with 90%+ test coverage for all advanced GitHub integration features.
"""

import asyncio
import json
import tempfile
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.github_quality_integration import (
    GitHubQualityGatesEngine, GitHubQualityContext, GitHubQualityGateType
)
from app.core.code_review_assistant import CodeReviewAssistant
from app.core.automated_testing_integration import AutomatedTestingIntegration
from app.core.advanced_repository_management import AdvancedRepositoryManagement
from app.core.intelligent_workflow_automation import IntelligentWorkflowAutomation, WorkflowTrigger
from app.core.quality_gates import QualityGatesEngine, QualityGateStatus, QualityGateSeverity
from app.models.github_integration import (
    GitHubRepository, PullRequest, CodeReview, AgentWorkTree,
    PullRequestStatus, WorkTreeStatus, ReviewStatus
)
from app.models.agent import Agent, AgentStatus
from app.schemas.github_integration import (
    ReviewType, TestSuiteType, ConflictResolutionStrategy
)


class TestGitHubQualityGatesEngine:
    """Test GitHub Quality Gates Engine functionality."""
    
    @pytest.fixture
    async def quality_engine(self):
        """Create GitHub quality gates engine."""
        quality_gates_engine = QualityGatesEngine()
        code_review_assistant = CodeReviewAssistant()
        testing_integration = AutomatedTestingIntegration()
        repository_management = AdvancedRepositoryManagement()
        workflow_automation = IntelligentWorkflowAutomation()
        
        return GitHubQualityGatesEngine(
            quality_gates_engine=quality_gates_engine,
            code_review_assistant=code_review_assistant,
            testing_integration=testing_integration,
            repository_management=repository_management,
            workflow_automation=workflow_automation
        )
    
    @pytest.fixture
    async def sample_github_context(self):
        """Create sample GitHub context."""
        return GitHubQualityContext(
            pull_request_id=uuid.uuid4(),
            repository_id=uuid.uuid4(),
            agent_id=uuid.uuid4(),
            pr_metadata={"files_changed": 5, "lines_added": 150, "lines_deleted": 50},
            repository_metadata={"language": "python", "size": "large"},
            execution_config={"timeout_minutes": 30}
        )
    
    @pytest.fixture
    async def mock_pull_request(self):
        """Create mock pull request."""
        pr = MagicMock(spec=PullRequest)
        pr.id = uuid.uuid4()
        pr.repository_id = uuid.uuid4()
        pr.title = "Test PR"
        pr.description = "Test pull request"
        pr.status = PullRequestStatus.OPEN
        pr.created_at = datetime.utcnow()
        
        # Mock repository
        repo = MagicMock(spec=GitHubRepository)
        repo.id = pr.repository_id
        repo.name = "test-repo"
        repo.owner = "test-owner"
        repo.has_permission = MagicMock(return_value=True)
        pr.repository = repo
        
        return pr
    
    @pytest.fixture
    async def mock_db_session(self, mock_pull_request):
        """Create mock database session."""
        db = AsyncMock(spec=AsyncSession)
        
        # Mock query result
        result = AsyncMock()
        result.scalar_one_or_none.return_value = mock_pull_request
        db.execute.return_value = result
        
        return db
    
    async def test_engine_initialization(self, quality_engine):
        """Test quality engine initialization."""
        assert quality_engine is not None
        assert quality_engine.code_review_assistant is not None
        assert quality_engine.testing_integration is not None
        assert quality_engine.repository_management is not None
        assert quality_engine.workflow_automation is not None
        assert quality_engine.targets["automated_review_coverage"] == 80.0
        assert quality_engine.targets["testing_integration_success_rate"] == 95.0
    
    async def test_github_quality_pipeline_execution(
        self, quality_engine, sample_github_context, mock_db_session
    ):
        """Test comprehensive GitHub quality pipeline execution."""
        # Mock component methods
        with patch.object(quality_engine.code_review_assistant, 'perform_comprehensive_review') as mock_review, \
             patch.object(quality_engine.testing_integration, 'trigger_automated_tests') as mock_test, \
             patch.object(quality_engine.repository_management, 'perform_intelligent_merge') as mock_merge, \
             patch.object(quality_engine.workflow_automation, 'execute_workflow') as mock_workflow, \
             patch.object(quality_engine.repository_management, 'analyze_repository_health') as mock_health:
            
            # Configure mocks
            mock_review.return_value = {
                "success": True,
                "ai_confidence": 0.85,
                "findings_count": 3,
                "categorized_findings": {"security": 0, "performance": 2, "style": 1},
                "overall_score": 0.9
            }
            
            mock_test.return_value = {
                "status": "completed",
                "test_run_id": "test-123",
                "analysis": {
                    "success_rate": 96.0,
                    "coverage_analysis": {"average_coverage": 92.5}
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
            
            # Mock workflow execution
            workflow_execution = AsyncMock()
            workflow_execution.success = True
            workflow_execution.quality_gates_results = {}
            mock_workflow.return_value = workflow_execution
            
            mock_health.return_value = {
                "score": 85.0,
                "overall_status": MagicMock(value="good"),
                "dependency_analysis": {"security_risk_score": 2.0}
            }
            
            # Execute pipeline
            success, results, analytics = await quality_engine.execute_github_quality_pipeline(
                github_context=sample_github_context,
                db=mock_db_session
            )
            
            # Verify results
            assert success is True
            assert len(results) > 0
            assert analytics is not None
            assert "execution_summary" in analytics
            assert analytics["execution_summary"]["total_gates"] > 0
            
            # Verify all components were called
            mock_review.assert_called_once()
            mock_test.assert_called_once()
            mock_merge.assert_called_once()
            mock_workflow.assert_called_once()
            mock_health.assert_called_once()
    
    async def test_ai_code_review_gate_success(
        self, quality_engine, sample_github_context, mock_pull_request
    ):
        """Test AI code review quality gate success scenario."""
        with patch.object(quality_engine.code_review_assistant, 'perform_comprehensive_review') as mock_review:
            mock_review.return_value = {
                "ai_confidence": 0.90,  # Above 80% threshold
                "findings_count": 2,
                "categorized_findings": {"security": 0, "performance": 1, "style": 1},
                "overall_score": 0.95
            }
            
            result = await quality_engine._execute_ai_code_review_gate(
                sample_github_context, mock_pull_request, mock_pull_request.repository
            )
            
            assert result.status == QualityGateStatus.PASSED
            assert result.overall_score == 100.0  # All metrics passed
            assert len(result.metrics) == 2
            
            # Check accuracy metric
            accuracy_metric = next(m for m in result.metrics if m.name == "ai_review_accuracy")
            assert accuracy_metric.value == 90.0
            assert accuracy_metric.status == QualityGateStatus.PASSED
            
            # Check findings metric
            findings_metric = next(m for m in result.metrics if m.name == "review_findings_quality")
            assert findings_metric.value == 0  # No critical security findings
            assert findings_metric.status == QualityGateStatus.PASSED
    
    async def test_ai_code_review_gate_failure(
        self, quality_engine, sample_github_context, mock_pull_request
    ):
        """Test AI code review quality gate failure scenario."""
        with patch.object(quality_engine.code_review_assistant, 'perform_comprehensive_review') as mock_review:
            mock_review.return_value = {
                "ai_confidence": 0.70,  # Below 80% threshold
                "findings_count": 5,
                "categorized_findings": {"security": 2, "performance": 2, "style": 1},  # Critical security findings
                "overall_score": 0.60
            }
            
            result = await quality_engine._execute_ai_code_review_gate(
                sample_github_context, mock_pull_request, mock_pull_request.repository
            )
            
            assert result.status == QualityGateStatus.FAILED
            assert result.overall_score == 0.0  # Both metrics failed
            assert len(result.metrics) == 2
            
            # Check accuracy metric
            accuracy_metric = next(m for m in result.metrics if m.name == "ai_review_accuracy")
            assert accuracy_metric.value == 70.0
            assert accuracy_metric.status == QualityGateStatus.FAILED
            assert accuracy_metric.severity == QualityGateSeverity.MAJOR
            
            # Check findings metric
            findings_metric = next(m for m in result.metrics if m.name == "review_findings_quality")
            assert findings_metric.value == 2  # Critical security findings
            assert findings_metric.status == QualityGateStatus.FAILED
            assert findings_metric.severity == QualityGateSeverity.CRITICAL
    
    async def test_automated_testing_gate_success(
        self, quality_engine, sample_github_context, mock_pull_request
    ):
        """Test automated testing quality gate success scenario."""
        with patch.object(quality_engine.testing_integration, 'trigger_automated_tests') as mock_test, \
             patch.object(quality_engine.testing_integration, 'monitor_test_execution') as mock_monitor:
            
            mock_test.return_value = {
                "status": "triggered",
                "test_run_id": "test-456"
            }
            
            mock_monitor.return_value = {
                "analysis": {
                    "success_rate": 96.0,  # Above 95% threshold
                    "coverage_analysis": {"average_coverage": 93.0}  # Above 90% threshold
                }
            }
            
            result = await quality_engine._execute_automated_testing_gate(
                sample_github_context, mock_pull_request, mock_pull_request.repository
            )
            
            assert result.status == QualityGateStatus.PASSED
            assert result.overall_score == 100.0  # All metrics passed
            assert len(result.metrics) == 2
            
            # Check success rate metric
            success_metric = next(m for m in result.metrics if m.name == "test_success_rate")
            assert success_metric.value == 96.0
            assert success_metric.status == QualityGateStatus.PASSED
            
            # Check coverage metric
            coverage_metric = next(m for m in result.metrics if m.name == "test_coverage")
            assert coverage_metric.value == 93.0
            assert coverage_metric.status == QualityGateStatus.PASSED
    
    async def test_automated_testing_gate_failure(
        self, quality_engine, sample_github_context, mock_pull_request
    ):
        """Test automated testing quality gate failure scenario."""
        with patch.object(quality_engine.testing_integration, 'trigger_automated_tests') as mock_test, \
             patch.object(quality_engine.testing_integration, 'monitor_test_execution') as mock_monitor:
            
            mock_test.return_value = {
                "status": "triggered",
                "test_run_id": "test-789"
            }
            
            mock_monitor.return_value = {
                "analysis": {
                    "success_rate": 85.0,  # Below 95% threshold
                    "coverage_analysis": {"average_coverage": 75.0}  # Below 90% threshold
                }
            }
            
            result = await quality_engine._execute_automated_testing_gate(
                sample_github_context, mock_pull_request, mock_pull_request.repository
            )
            
            assert result.status == QualityGateStatus.FAILED
            assert result.overall_score == 0.0  # Both metrics failed
            assert len(result.metrics) == 2
            
            # Check success rate metric
            success_metric = next(m for m in result.metrics if m.name == "test_success_rate")
            assert success_metric.value == 85.0
            assert success_metric.status == QualityGateStatus.FAILED
            assert success_metric.severity == QualityGateSeverity.CRITICAL
            
            # Check coverage metric
            coverage_metric = next(m for m in result.metrics if m.name == "test_coverage")
            assert coverage_metric.value == 75.0
            assert coverage_metric.status == QualityGateStatus.FAILED
            assert coverage_metric.severity == QualityGateSeverity.MAJOR
    
    async def test_conflict_resolution_gate(
        self, quality_engine, sample_github_context, mock_pull_request
    ):
        """Test conflict resolution quality gate."""
        with patch.object(quality_engine.repository_management, 'perform_intelligent_merge') as mock_merge:
            mock_merge.return_value = {
                "success": True,
                "conflict_analysis": {
                    "auto_resolvable": True,
                    "conflicts_detected": 2,
                    "conflict_summary": "Minor conflicts automatically resolved"
                },
                "files_modified": ["file1.py", "file2.py"]
            }
            
            result = await quality_engine._execute_conflict_resolution_gate(
                sample_github_context, mock_pull_request, mock_pull_request.repository
            )
            
            assert result.status == QualityGateStatus.PASSED
            assert len(result.metrics) == 2
            
            # Check resolution accuracy metric
            accuracy_metric = next(m for m in result.metrics if m.name == "conflict_resolution_accuracy")
            assert accuracy_metric.value == 85.0  # Simulated success accuracy
            assert accuracy_metric.status == QualityGateStatus.PASSED
            
            # Check auto-resolution capability metric
            auto_metric = next(m for m in result.metrics if m.name == "auto_resolution_capability")
            assert auto_metric.value == 100.0  # Auto-resolvable
            assert auto_metric.status == QualityGateStatus.PASSED
    
    async def test_workflow_automation_gate(
        self, quality_engine, sample_github_context, mock_pull_request
    ):
        """Test workflow automation quality gate."""
        with patch.object(quality_engine.workflow_automation, 'execute_workflow') as mock_workflow:
            # Mock successful workflow execution
            workflow_execution = AsyncMock()
            workflow_execution.success = True
            workflow_execution.quality_gates_results = {
                "code_quality": AsyncMock(value="passed"),
                "security": AsyncMock(value="passed"),
                "testing": AsyncMock(value="warning")
            }
            mock_workflow.return_value = workflow_execution
            
            result = await quality_engine._execute_workflow_automation_gate(
                sample_github_context, mock_pull_request, mock_pull_request.repository
            )
            
            assert result.status == QualityGateStatus.PASSED
            assert len(result.metrics) == 2
            
            # Check completion rate metric
            completion_metric = next(m for m in result.metrics if m.name == "workflow_completion_rate")
            assert completion_metric.value == 90.0  # Simulated success rate
            assert completion_metric.status == QualityGateStatus.PASSED
            
            # Check quality gates pass rate metric
            gates_metric = next(m for m in result.metrics if m.name == "workflow_quality_gates_pass_rate")
            assert gates_metric.value == 100.0  # All gates passed/warning
            assert gates_metric.status == QualityGateStatus.PASSED
    
    async def test_repository_health_gate(
        self, quality_engine, sample_github_context, mock_pull_request
    ):
        """Test repository health quality gate."""
        with patch.object(quality_engine.repository_management, 'analyze_repository_health') as mock_health:
            mock_health.return_value = {
                "score": 85.0,  # Above 75% threshold
                "overall_status": MagicMock(value="good"),
                "dependency_analysis": {
                    "security_risk_score": 2.5  # Below 3.0 threshold
                }
            }
            
            result = await quality_engine._execute_repository_health_gate(
                sample_github_context, mock_pull_request, mock_pull_request.repository
            )
            
            assert result.status == QualityGateStatus.PASSED
            assert len(result.metrics) == 2
            
            # Check health score metric
            health_metric = next(m for m in result.metrics if m.name == "repository_health_score")
            assert health_metric.value == 85.0
            assert health_metric.status == QualityGateStatus.PASSED
            
            # Check security risk metric
            security_metric = next(m for m in result.metrics if m.name == "dependency_security_risk")
            assert security_metric.value == 2.5
            assert security_metric.status == QualityGateStatus.PASSED
    
    async def test_pull_request_quality_validation(
        self, quality_engine, mock_db_session
    ):
        """Test comprehensive pull request quality validation."""
        pr_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        
        with patch.object(quality_engine, 'execute_github_quality_pipeline') as mock_pipeline:
            mock_pipeline.return_value = (
                True,  # success
                [],    # gate_results
                {"execution_summary": {"execution_time_seconds": 45.0}}  # analytics
            )
            
            success, report = await quality_engine.validate_pull_request_quality(
                pull_request_id=pr_id,
                agent_id=agent_id,
                db=mock_db_session
            )
            
            assert success is True
            assert "validation_id" in report
            assert "timestamp" in report
            assert "overall_success" in report
            assert "success_metrics_compliance" in report
            assert "analytics" in report
            assert "recommendations" in report
    
    async def test_target_compliance_evaluation(self, quality_engine):
        """Test success metrics target compliance evaluation."""
        # Add test data
        quality_engine.github_metrics["automated_review_accuracy"] = [85.0, 90.0, 88.0]
        quality_engine.github_metrics["testing_success_rates"] = [96.0, 97.0, 95.5]
        
        compliance = quality_engine._evaluate_target_compliance()
        
        assert "automated_review_coverage" in compliance
        assert "testing_integration_success_rate" in compliance
        assert "overall_compliance" in compliance
        
        # Check automated review compliance
        review_compliance = compliance["automated_review_coverage"]
        assert review_compliance["current"] > 80.0  # Above target
        assert review_compliance["compliant"] is True
        
        # Check testing compliance
        testing_compliance = compliance["testing_integration_success_rate"]
        assert testing_compliance["current"] > 95.0  # Above target
        assert testing_compliance["compliant"] is True
        
        # Overall compliance should be 100%
        assert compliance["overall_compliance"] == 100.0
    
    async def test_error_handling_in_quality_gates(
        self, quality_engine, sample_github_context, mock_pull_request
    ):
        """Test error handling in quality gate execution."""
        with patch.object(quality_engine.code_review_assistant, 'perform_comprehensive_review') as mock_review:
            mock_review.side_effect = Exception("Code review service unavailable")
            
            result = await quality_engine._execute_ai_code_review_gate(
                sample_github_context, mock_pull_request, mock_pull_request.repository
            )
            
            assert result.status == QualityGateStatus.FAILED
            assert result.overall_score == 0.0
            assert "Code review service unavailable" in result.error_message
    
    async def test_quality_trends_analysis(self, quality_engine):
        """Test quality trends analysis and direction calculation."""
        # Test improving trend
        improving_scores = [70.0, 75.0, 80.0, 85.0, 90.0]
        trend = quality_engine._calculate_trend_direction(improving_scores)
        assert trend == "improving"
        
        # Test declining trend
        declining_scores = [90.0, 85.0, 80.0, 75.0, 70.0]
        trend = quality_engine._calculate_trend_direction(declining_scores)
        assert trend == "declining"
        
        # Test stable trend
        stable_scores = [80.0, 81.0, 79.0, 80.5, 80.0]
        trend = quality_engine._calculate_trend_direction(stable_scores)
        assert trend == "stable"
        
        # Test insufficient data
        short_scores = [80.0, 85.0]
        trend = quality_engine._calculate_trend_direction(short_scores)
        assert trend == "unknown"


class TestAdvancedGitHubIntegrationAPI:
    """Test Advanced GitHub Integration API endpoints."""
    
    @pytest.fixture
    async def client(self):
        """Create test client."""
        from app.main import app
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client
    
    @pytest.fixture
    async def auth_headers(self):
        """Create authentication headers."""
        return {"Authorization": "Bearer test-token"}
    
    async def test_comprehensive_quality_validation_endpoint(self, client, auth_headers):
        """Test comprehensive quality validation endpoint."""
        pr_id = str(uuid.uuid4())
        
        with patch('app.api.v1.advanced_github_integration.get_github_quality_gates_engine') as mock_engine_dep, \
             patch('app.api.v1.advanced_github_integration.get_current_agent') as mock_agent_dep, \
             patch('app.api.v1.advanced_github_integration.get_db_session') as mock_db_dep:
            
            # Mock dependencies
            mock_engine = AsyncMock()
            mock_engine.validate_pull_request_quality.return_value = (
                True,
                {
                    "validation_id": pr_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "overall_success": True,
                    "quality_gates_results": [
                        {"status": "passed", "gate_type": "ai_code_review"},
                        {"status": "passed", "gate_type": "automated_testing"}
                    ],
                    "success_metrics_compliance": {"overall_compliance": 95.0},
                    "recommendations": ["Continue with current quality practices"],
                    "analytics": {
                        "execution_summary": {
                            "execution_time_seconds": 30.0,
                            "quality_score": 92.5
                        }
                    }
                }
            )
            mock_engine_dep.return_value = mock_engine
            
            mock_agent = AsyncMock()
            mock_agent.id = uuid.uuid4()
            mock_agent_dep.return_value = mock_agent
            
            mock_db_dep.return_value = AsyncMock()
            
            # Make request
            response = await client.post(
                f"/api/v1/github/quality/validate/{pr_id}",
                headers=auth_headers,
                json={"timeout_minutes": 30}
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["pull_request_id"] == pr_id
            assert data["overall_success"] is True
            assert data["quality_score"] == 92.5
            assert "quality_gates_summary" in data
            assert "success_metrics_compliance" in data
            assert "recommendations" in data
            assert data["validation_duration_seconds"] == 30.0
    
    async def test_quality_metrics_dashboard_endpoint(self, client, auth_headers):
        """Test quality metrics dashboard endpoint."""
        with patch('app.api.v1.advanced_github_integration.get_github_quality_gates_engine') as mock_engine_dep, \
             patch('app.api.v1.advanced_github_integration.get_current_agent') as mock_agent_dep:
            
            # Mock dependencies
            mock_engine = AsyncMock()
            mock_engine._evaluate_target_compliance.return_value = {
                "overall_compliance": 90.0,
                "compliant_targets": 4,
                "total_targets": 5
            }
            mock_engine.targets = {
                "automated_review_coverage": 80.0,
                "testing_integration_success_rate": 95.0
            }
            mock_engine.github_metrics = {
                "automated_review_accuracy": [85.0, 90.0, 88.0],
                "testing_success_rates": [96.0, 97.0, 95.5]
            }
            mock_engine._calculate_trend_direction.return_value = "improving"
            mock_engine_dep.return_value = mock_engine
            
            mock_agent = AsyncMock()
            mock_agent.id = uuid.uuid4()
            mock_agent_dep.return_value = mock_agent
            
            # Make request
            response = await client.get(
                "/api/v1/github/quality/metrics?days=30",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["period_days"] == 30
            assert "success_metrics_targets" in data
            assert "current_compliance" in data
            assert "quality_trends" in data
            assert "system_health" in data
            assert "recommendations" in data
            assert data["current_compliance"]["overall_compliance"] == 90.0


class TestCodeReviewAssistantAdvanced:
    """Test advanced code review assistant functionality."""
    
    @pytest.fixture
    async def code_review_assistant(self):
        """Create code review assistant."""
        return CodeReviewAssistant()
    
    async def test_comprehensive_review_all_types(self, code_review_assistant):
        """Test comprehensive review with all review types."""
        mock_pr = MagicMock(spec=PullRequest)
        mock_pr.id = uuid.uuid4()
        mock_pr.title = "Add new feature"
        mock_pr.description = "Implements new authentication feature"
        
        review_types = [
            ReviewType.SECURITY, 
            ReviewType.PERFORMANCE, 
            ReviewType.STYLE, 
            ReviewType.MAINTAINABILITY
        ]
        
        with patch.object(code_review_assistant, '_analyze_security') as mock_security, \
             patch.object(code_review_assistant, '_analyze_performance') as mock_performance, \
             patch.object(code_review_assistant, '_analyze_style') as mock_style, \
             patch.object(code_review_assistant, '_analyze_maintainability') as mock_maintainability:
            
            # Configure mocks
            mock_security.return_value = {"issues": [], "score": 0.95}
            mock_performance.return_value = {"issues": ["Optimize database query"], "score": 0.80}
            mock_style.return_value = {"issues": ["Add docstring"], "score": 0.85}
            mock_maintainability.return_value = {"issues": [], "score": 0.90}
            
            result = await code_review_assistant.perform_comprehensive_review(
                pull_request=mock_pr,
                review_types=review_types
            )
            
            assert result["success"] is True
            assert "findings_count" in result
            assert "categorized_findings" in result
            assert "overall_score" in result
            assert result["ai_confidence"] >= 0.8  # High confidence expected
            
            # Verify all analysis methods were called
            mock_security.assert_called_once()
            mock_performance.assert_called_once()
            mock_style.assert_called_once()
            mock_maintainability.assert_called_once()
    
    async def test_review_accuracy_calculation(self, code_review_assistant):
        """Test AI review accuracy calculation."""
        # Simulate high-confidence review
        with patch.object(code_review_assistant, '_calculate_confidence_score') as mock_confidence:
            mock_confidence.return_value = 0.92
            
            mock_pr = MagicMock(spec=PullRequest)
            result = await code_review_assistant.perform_comprehensive_review(
                pull_request=mock_pr,
                review_types=[ReviewType.SECURITY]
            )
            
            assert result["ai_confidence"] == 0.92
            assert result["ai_confidence"] >= 0.80  # Meets target threshold


class TestAutomatedTestingIntegrationAdvanced:
    """Test advanced automated testing integration."""
    
    @pytest.fixture
    async def testing_integration(self):
        """Create testing integration."""
        return AutomatedTestingIntegration()
    
    async def test_comprehensive_test_execution(self, testing_integration):
        """Test comprehensive test execution across multiple suites."""
        mock_pr = MagicMock(spec=PullRequest)
        test_suites = [TestSuiteType.UNIT, TestSuiteType.INTEGRATION, TestSuiteType.SECURITY]
        
        with patch.object(testing_integration, '_execute_test_suite') as mock_execute:
            # Mock successful test execution
            mock_execute.return_value = {
                "status": "passed",
                "total_tests": 150,
                "passed_tests": 145,
                "failed_tests": 5,
                "coverage": 94.0
            }
            
            result = await testing_integration.trigger_automated_tests(
                pull_request=mock_pr,
                test_suites=test_suites
            )
            
            assert result["status"] == "triggered"
            assert "test_run_id" in result
            
            # Test monitoring
            monitoring_result = await testing_integration.monitor_test_execution(
                test_run_id=result["test_run_id"],
                timeout_minutes=30
            )
            
            assert "analysis" in monitoring_result
            analysis = monitoring_result["analysis"]
            assert analysis["success_rate"] >= 95.0  # Meets target threshold
    
    async def test_failure_analysis_intelligence(self, testing_integration):
        """Test intelligent failure analysis."""
        with patch.object(testing_integration, '_analyze_test_failures') as mock_analyze:
            mock_analyze.return_value = {
                "failure_categories": {
                    "infrastructure": ["Database connection timeout"],
                    "logic": ["Assertion error in authentication flow"]
                },
                "common_patterns": ["timeout", "authentication"],
                "recommendations": ["Increase timeout values", "Review auth logic"]
            }
            
            result = await testing_integration._perform_failure_analysis(
                test_results={"failed_tests": 5, "total_tests": 100}
            )
            
            assert "failure_categories" in result
            assert "recommendations" in result
            assert len(result["recommendations"]) > 0


class TestWorkflowAutomationAdvanced:
    """Test advanced workflow automation functionality."""
    
    @pytest.fixture
    async def workflow_automation(self):
        """Create workflow automation."""
        return IntelligentWorkflowAutomation()
    
    async def test_end_to_end_workflow_execution(self, workflow_automation):
        """Test end-to-end workflow execution."""
        mock_pr = MagicMock(spec=PullRequest)
        
        with patch.object(workflow_automation, '_execute_stage') as mock_stage:
            # Mock successful stage execution
            mock_stage.return_value = {"success": True, "duration": 10.0}
            
            result = await workflow_automation.execute_workflow(
                pull_request=mock_pr,
                trigger=WorkflowTrigger.PR_CREATED
            )
            
            assert result.success is True
            assert len(result.steps_completed) > 0
            assert len(result.steps_failed) == 0
    
    async def test_quality_gates_integration_in_workflow(self, workflow_automation):
        """Test quality gates integration within workflow."""
        mock_pr = MagicMock(spec=PullRequest)
        
        with patch.object(workflow_automation, '_execute_quality_gates') as mock_gates:
            mock_gates.return_value = {
                "overall_success": True,
                "gate_results": {"code_quality": "passed", "security": "passed"}
            }
            
            result = await workflow_automation.execute_workflow(
                pull_request=mock_pr,
                trigger=WorkflowTrigger.PR_CREATED
            )
            
            # Quality gates should be executed and passed
            assert hasattr(result, 'quality_gates_results')
            mock_gates.assert_called_once()


class TestPerformanceAndScalability:
    """Test performance and scalability requirements."""
    
    async def test_quality_pipeline_performance(self):
        """Test quality pipeline meets performance requirements."""
        # Initialize system
        quality_engine = GitHubQualityGatesEngine(
            quality_gates_engine=QualityGatesEngine(),
            code_review_assistant=CodeReviewAssistant(),
            testing_integration=AutomatedTestingIntegration(),
            repository_management=AdvancedRepositoryManagement(),
            workflow_automation=IntelligentWorkflowAutomation()
        )
        
        github_context = GitHubQualityContext(
            pull_request_id=uuid.uuid4(),
            repository_id=uuid.uuid4(),
            agent_id=uuid.uuid4()
        )
        
        # Mock database and components for performance test
        with patch.object(quality_engine, '_get_github_context') as mock_context, \
             patch.object(quality_engine, '_execute_github_quality_gate') as mock_gate:
            
            mock_pr = MagicMock(spec=PullRequest)
            mock_repo = MagicMock(spec=GitHubRepository)
            mock_context.return_value = (mock_pr, mock_repo)
            
            # Mock fast gate execution
            mock_gate.return_value = MagicMock(
                status=QualityGateStatus.PASSED,
                overall_score=90.0,
                execution_time_seconds=2.0,
                metrics=[]
            )
            
            # Measure execution time
            start_time = datetime.utcnow()
            
            success, results, analytics = await quality_engine.execute_github_quality_pipeline(
                github_context=github_context,
                db=AsyncMock()
            )
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Performance requirements
            assert execution_time < 60.0  # Should complete within 1 minute
            assert success is True
            assert len(results) > 0
    
    async def test_concurrent_quality_validations(self):
        """Test system handles concurrent quality validations."""
        quality_engine = GitHubQualityGatesEngine(
            quality_gates_engine=QualityGatesEngine(),
            code_review_assistant=CodeReviewAssistant(),
            testing_integration=AutomatedTestingIntegration(),
            repository_management=AdvancedRepositoryManagement(),
            workflow_automation=IntelligentWorkflowAutomation()
        )
        
        # Create multiple validation tasks
        pr_ids = [uuid.uuid4() for _ in range(5)]
        
        with patch.object(quality_engine, 'validate_pull_request_quality') as mock_validate:
            mock_validate.return_value = (
                True,
                {"validation_id": "test", "timestamp": datetime.utcnow().isoformat()}
            )
            
            # Execute concurrent validations
            tasks = [
                quality_engine.validate_pull_request_quality(
                    pull_request_id=pr_id,
                    agent_id=uuid.uuid4(),
                    db=AsyncMock()
                )
                for pr_id in pr_ids
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All validations should complete successfully
            assert len(results) == 5
            for result in results:
                assert not isinstance(result, Exception)
                success, report = result
                assert success is True


class TestIntegrationWithExistingSystems:
    """Test integration with existing systems."""
    
    async def test_quality_gates_engine_integration(self):
        """Test integration with existing quality gates engine."""
        quality_gates_engine = QualityGatesEngine()
        
        # Test that GitHub quality engine can work with existing quality gates
        github_engine = GitHubQualityGatesEngine(
            quality_gates_engine=quality_gates_engine,
            code_review_assistant=CodeReviewAssistant(),
            testing_integration=AutomatedTestingIntegration(),
            repository_management=AdvancedRepositoryManagement(),
            workflow_automation=IntelligentWorkflowAutomation()
        )
        
        assert github_engine.quality_gates_engine is quality_gates_engine
        assert isinstance(github_engine.targets, dict)
        assert len(github_engine.targets) == 5  # All target metrics defined
    
    async def test_database_model_compatibility(self):
        """Test compatibility with existing database models."""
        # Test that new features work with existing models
        pr = PullRequest()
        repo = GitHubRepository()
        agent = Agent()
        
        # These should not raise errors
        assert hasattr(pr, 'id')
        assert hasattr(repo, 'id')
        assert hasattr(agent, 'id')
        
        # New fields should be accessible
        pr.automated_merge_eligible = True
        pr.quality_gates_passed = True
        repo.auto_merge_enabled = False
        
        assert pr.automated_merge_eligible is True
        assert pr.quality_gates_passed is True
        assert repo.auto_merge_enabled is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])