#!/usr/bin/env python3
"""
Test suite for Technical Debt API endpoints.

Tests all API endpoints with comprehensive validation of request/response models,
error handling, and integration with technical debt analysis components.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession

# Add project root to path for testing
import sys
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.api.technical_debt import router as technical_debt_router
from app.models.project_index import ProjectIndex, FileEntry, ProjectStatus
from app.project_index.debt_analyzer import DebtAnalysisResult, DebtItem, DebtCategory, DebtSeverity, DebtStatus
from app.project_index.debt_remediation_engine import RemediationPlan, RemediationRecommendation, RemediationStrategy, RemediationPriority, RemediationImpact
from app.project_index.historical_analyzer import DebtEvolutionResult, DebtEvolutionPoint, DebtTrendAnalysis, DebtHotspot


# Create FastAPI app for testing
app = FastAPI()
app.include_router(technical_debt_router)

# Test client
client = TestClient(app)


# ================== TEST FIXTURES ==================

def create_mock_project() -> ProjectIndex:
    """Create mock project for testing."""
    project = Mock(spec=ProjectIndex)
    project.id = uuid.uuid4()
    project.name = "Test Project"
    project.root_path = "/test/project"
    project.status = ProjectStatus.ACTIVE
    
    # Create mock file entries
    file_entries = []
    for i in range(3):
        file_entry = Mock(spec=FileEntry)
        file_entry.id = i + 1
        file_entry.file_path = f"/test/project/file_{i}.py"
        file_entry.file_name = f"file_{i}.py"
        file_entry.is_binary = False
        file_entry.language = "python"
        file_entry.line_count = 100 + (i * 50)
        file_entries.append(file_entry)
    
    project.file_entries = file_entries
    project.dependency_relationships = []
    
    return project


def create_mock_debt_analysis_result() -> DebtAnalysisResult:
    """Create mock debt analysis result."""
    debt_items = [
        DebtItem(
            id="debt_1",
            project_id="test-project",
            file_id="file_1",
            debt_type="complexity",
            category=DebtCategory.COMPLEXITY,
            severity=DebtSeverity.HIGH,
            status=DebtStatus.ACTIVE,
            description="High cyclomatic complexity in function",
            location={"file_path": "/test/project/file_0.py", "line_number": 25},
            debt_score=0.8,
            confidence_score=0.9,
            remediation_suggestion="Consider extracting methods to reduce complexity",
            estimated_effort_hours=4
        ),
        DebtItem(
            id="debt_2",
            project_id="test-project",
            file_id="file_2",
            debt_type="duplication",
            category=DebtCategory.CODE_DUPLICATION,
            severity=DebtSeverity.MEDIUM,
            status=DebtStatus.ACTIVE,
            description="Duplicated code detected",
            location={"file_path": "/test/project/file_1.py", "line_number": 45},
            debt_score=0.5,
            confidence_score=0.8,
            remediation_suggestion="Extract common functionality to reduce duplication",
            estimated_effort_hours=2
        )
    ]
    
    return DebtAnalysisResult(
        project_id="test-project",
        total_debt_score=1.3,
        debt_items=debt_items,
        category_scores={"complexity": 0.8, "code_duplication": 0.5},
        file_count=3,
        lines_of_code=250,
        analysis_duration=2.5,
        recommendations=[
            "Focus on reducing complexity in file_0.py",
            "Address code duplication in file_1.py"
        ]
    )


def create_mock_remediation_plan() -> RemediationPlan:
    """Create mock remediation plan."""
    recommendations = [
        RemediationRecommendation(
            id="rec_1",
            strategy=RemediationStrategy.EXTRACT_METHOD,
            priority=RemediationPriority.HIGH,
            impact=RemediationImpact.SIGNIFICANT,
            title="Extract complex method",
            description="Break down complex function into smaller methods",
            rationale="High complexity affects maintainability",
            file_path="/test/project/file_0.py",
            line_ranges=[(25, 45)],
            affected_functions=["complex_function"],
            affected_classes=[],
            debt_reduction_score=0.7,
            implementation_effort=0.5,
            risk_level=0.3,
            cost_benefit_ratio=2.3,
            suggested_approach="Extract logical blocks into separate methods",
            code_examples=["# Example refactored code"],
            related_patterns=["Extract Method"],
            dependencies=[],
            debt_categories=[DebtCategory.COMPLEXITY],
            historical_context={}
        )
    ]
    
    return RemediationPlan(
        project_id="test-project",
        scope="project",
        target_path="/test/project",
        recommendations=recommendations,
        execution_phases=[["rec_1"]],
        total_debt_reduction=0.7,
        total_effort_estimate=0.5,
        total_risk_score=0.3,
        estimated_duration_days=1,
        immediate_actions=[],
        quick_wins=["rec_1"],
        long_term_goals=[],
        plan_rationale="Focus on high-impact complexity reduction",
        success_criteria=["Reduce complexity by 0.7 points"],
        potential_blockers=["Limited development time"]
    )


def create_mock_historical_analysis() -> DebtEvolutionResult:
    """Create mock historical analysis result."""
    timeline = [
        DebtEvolutionPoint(
            commit_hash="abc123",
            date=datetime.utcnow() - timedelta(days=30),
            total_debt_score=0.5,
            category_scores={"complexity": 0.3, "duplication": 0.2},
            files_analyzed=3,
            lines_of_code=200,
            debt_items_count=2,
            debt_delta=0.0,
            commit_message="Initial commit",
            author="developer"
        ),
        DebtEvolutionPoint(
            commit_hash="def456",
            date=datetime.utcnow() - timedelta(days=15),
            total_debt_score=0.8,
            category_scores={"complexity": 0.5, "duplication": 0.3},
            files_analyzed=3,
            lines_of_code=250,
            debt_items_count=3,
            debt_delta=0.3,
            commit_message="Added features",
            author="developer"
        )
    ]
    
    trend_analysis = DebtTrendAnalysis(
        trend_direction="increasing",
        trend_strength=0.8,
        velocity=0.02,
        acceleration=0.001,
        projected_debt_30_days=1.0,
        projected_debt_90_days=1.5,
        confidence_level=0.85,
        seasonal_patterns=[],
        anomaly_periods=[],
        risk_level="medium"
    )
    
    hotspots = [
        DebtHotspot(
            file_path="/test/project/file_0.py",
            debt_score=0.8,
            debt_velocity=0.05,
            stability_risk=0.6,
            contributor_count=2,
            priority="high",
            categories_affected=["complexity"],
            recommendations=["Extract methods to reduce complexity"]
        )
    ]
    
    return DebtEvolutionResult(
        project_id="test-project",
        evolution_timeline=timeline,
        trend_analysis=trend_analysis,
        debt_hotspots=hotspots,
        category_trends={
            "complexity": DebtTrendAnalysis(
                trend_direction="increasing",
                trend_strength=0.7,
                velocity=0.03,
                acceleration=0.002,
                projected_debt_30_days=0.6,
                projected_debt_90_days=0.9,
                confidence_level=0.8,
                seasonal_patterns=[],
                anomaly_periods=[],
                risk_level="medium"
            )
        },
        recommendations=["Focus on complexity reduction in file_0.py"]
    )


# ================== TEST CASES ==================

class TestTechnicalDebtAPI:
    """Test suite for technical debt API endpoints."""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment."""
        self.project = create_mock_project()
        self.project_id = str(self.project.id)
    
    @patch('app.api.technical_debt.get_project_or_404')
    @patch('app.api.technical_debt.get_technical_debt_analyzer')
    @patch('app.api.technical_debt.get_advanced_debt_detector')
    @patch('app.api.technical_debt.get_session')
    def test_analyze_technical_debt_success(
        self, mock_get_session, mock_get_advanced_detector, 
        mock_get_debt_analyzer, mock_get_project
    ):
        """Test successful technical debt analysis."""
        # Setup mocks
        mock_get_project.return_value = self.project
        
        mock_debt_analyzer = Mock()
        mock_debt_analyzer.analyze_project_debt = AsyncMock(
            return_value=create_mock_debt_analysis_result()
        )
        mock_get_debt_analyzer.return_value = mock_debt_analyzer
        
        mock_advanced_detector = Mock()
        mock_advanced_detector.analyze_advanced_debt_patterns = AsyncMock(return_value=[])
        mock_get_advanced_detector.return_value = mock_advanced_detector
        
        mock_session = Mock(spec=AsyncSession)
        mock_get_session.return_value = mock_session
        
        # Make request
        response = client.post(
            f"/api/technical-debt/{self.project_id}/analyze",
            json={
                "include_advanced_patterns": True,
                "include_historical_analysis": False,
                "analysis_depth": "standard"
            }
        )
        
        # Validate response
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "Technical debt analysis completed successfully" in data["message"]
        assert "data" in data
        
        analysis_data = data["data"]
        assert analysis_data["project_id"] == self.project_id
        assert "analysis_id" in analysis_data
        assert analysis_data["total_debt_score"] == 1.3
        assert len(analysis_data["debt_items"]) == 2
        assert "category_breakdown" in analysis_data
        assert "severity_breakdown" in analysis_data
        assert analysis_data["file_count"] == 3
        assert analysis_data["lines_of_code"] == 250
    
    @patch('app.api.technical_debt.get_project_or_404')
    def test_analyze_technical_debt_project_not_found(self, mock_get_project):
        """Test technical debt analysis with non-existent project."""
        from fastapi import HTTPException
        mock_get_project.side_effect = HTTPException(status_code=404, detail="Project not found")
        
        response = client.post(
            f"/api/technical-debt/{self.project_id}/analyze",
            json={
                "include_advanced_patterns": False,
                "analysis_depth": "quick"
            }
        )
        
        assert response.status_code == 404
        assert "Project not found" in response.json()["detail"]
    
    def test_analyze_technical_debt_invalid_request(self):
        """Test technical debt analysis with invalid request data."""
        response = client.post(
            f"/api/technical-debt/{self.project_id}/analyze",
            json={
                "include_advanced_patterns": "invalid",  # Should be boolean
                "analysis_depth": "invalid_depth"  # Invalid enum value
            }
        )
        
        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data
    
    @patch('app.api.technical_debt.get_project_or_404')
    @patch('app.api.technical_debt.get_historical_analyzer')
    def test_get_debt_history_success(self, mock_get_historical_analyzer, mock_get_project):
        """Test successful debt history retrieval."""
        # Setup mocks
        mock_get_project.return_value = self.project
        
        mock_historical_analyzer = Mock()
        mock_historical_analyzer.analyze_debt_evolution = AsyncMock(
            return_value=create_mock_historical_analysis()
        )
        mock_get_historical_analyzer.return_value = mock_historical_analyzer
        
        # Make request
        response = client.get(
            f"/api/technical-debt/{self.project_id}/history?lookback_days=90&sample_frequency_days=7"
        )
        
        # Validate response
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "Historical debt analysis completed successfully" in data["message"]
        
        history_data = data["data"]
        assert history_data["project_id"] == self.project_id
        assert history_data["lookback_days"] == 90
        assert len(history_data["evolution_timeline"]) == 2
        assert "trend_analysis" in history_data
        assert "debt_hotspots" in history_data
        assert len(history_data["debt_hotspots"]) == 1
    
    def test_get_debt_history_invalid_parameters(self):
        """Test debt history with invalid parameters."""
        response = client.get(
            f"/api/technical-debt/{self.project_id}/history?lookback_days=500&sample_frequency_days=50"
        )
        
        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data
    
    @patch('app.api.technical_debt.get_project_or_404')
    @patch('app.api.technical_debt.get_debt_remediation_engine')
    def test_generate_remediation_plan_success(self, mock_get_remediation_engine, mock_get_project):
        """Test successful remediation plan generation."""
        # Setup mocks
        mock_get_project.return_value = self.project
        
        mock_remediation_engine = Mock()
        mock_remediation_engine.generate_remediation_plan = AsyncMock(
            return_value=create_mock_remediation_plan()
        )
        mock_get_remediation_engine.return_value = mock_remediation_engine
        
        # Make request
        response = client.post(
            f"/api/technical-debt/{self.project_id}/remediation-plan?scope=project"
        )
        
        # Validate response
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "Remediation plan generated successfully" in data["message"]
        
        plan_data = data["data"]
        assert plan_data["project_id"] == self.project_id
        assert "plan_id" in plan_data
        assert plan_data["scope"] == "project"
        assert plan_data["recommendations_count"] == 1
        assert plan_data["total_debt_reduction"] == 0.7
        assert len(plan_data["quick_wins"]) == 1
    
    def test_generate_remediation_plan_invalid_scope(self):
        """Test remediation plan generation with invalid scope."""
        response = client.post(
            f"/api/technical-debt/{self.project_id}/remediation-plan?scope=invalid_scope"
        )
        
        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data
    
    @patch('app.api.technical_debt.get_project_or_404')
    @patch('app.api.technical_debt.get_debt_remediation_engine')
    def test_get_file_recommendations_success(self, mock_get_remediation_engine, mock_get_project):
        """Test successful file-specific recommendations."""
        # Setup mocks
        mock_get_project.return_value = self.project
        
        mock_recommendations = [
            RemediationRecommendation(
                id="file_rec_1",
                strategy=RemediationStrategy.EXTRACT_METHOD,
                priority=RemediationPriority.MEDIUM,
                impact=RemediationImpact.MODERATE,
                title="Extract method",
                description="Extract complex logic into separate method",
                rationale="Improves readability and maintainability",
                file_path="/test/project/file_0.py",
                line_ranges=[(10, 20)],
                affected_functions=["target_function"],
                affected_classes=[],
                debt_reduction_score=0.4,
                implementation_effort=0.3,
                risk_level=0.2,
                cost_benefit_ratio=1.8,
                suggested_approach="Extract logical block into separate method",
                code_examples=["def extracted_method(): pass"],
                related_patterns=["Extract Method"],
                dependencies=[],
                debt_categories=[DebtCategory.COMPLEXITY],
                historical_context={}
            )
        ]
        
        mock_remediation_engine = Mock()
        mock_remediation_engine.get_file_specific_recommendations = AsyncMock(
            return_value=mock_recommendations
        )
        mock_get_remediation_engine.return_value = mock_remediation_engine
        
        # Make request with URL-encoded file path
        file_path = "test%2Fproject%2Ffile_0.py"
        response = client.get(
            f"/api/technical-debt/{self.project_id}/recommendations/{file_path}"
        )
        
        # Validate response
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "recommendations for test/project/file_0.py" in data["message"]
        
        response_data = data["data"]
        assert response_data["file_path"] == "test/project/file_0.py"
        assert len(response_data["recommendations"]) == 1
        
        recommendation = response_data["recommendations"][0]
        assert recommendation["id"] == "file_rec_1"
        assert recommendation["strategy"] == "extract_method"
        assert recommendation["priority"] == "medium"
        assert recommendation["title"] == "Extract method"
    
    @patch('app.api.technical_debt.get_project_or_404')
    @patch('app.api.technical_debt.get_debt_monitor_integration')
    def test_monitoring_status_success(self, mock_get_monitor_integration, mock_get_project):
        """Test successful monitoring status retrieval."""
        # Setup mocks
        mock_get_project.return_value = self.project
        
        mock_status = {
            'enabled': True,
            'active_since': datetime.utcnow().isoformat(),
            'monitored_projects_count': 1,
            'total_files_monitored': 3,
            'total_debt_events': 5,
            'configuration': {'debt_change_threshold': 0.1},
            'projects': {self.project_id: {'name': 'Test Project'}}
        }
        
        mock_monitor_integration = Mock()
        mock_monitor_integration.get_monitoring_status = AsyncMock(return_value=mock_status)
        mock_get_monitor_integration.return_value = mock_monitor_integration
        
        # Make request
        response = client.get(f"/api/technical-debt/{self.project_id}/monitoring/status")
        
        # Validate response
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "Monitoring status retrieved successfully" in data["message"]
        
        status_data = data["data"]
        assert status_data["enabled"] == True
        assert status_data["monitored_projects_count"] == 1
        assert status_data["total_files_monitored"] == 3
        assert status_data["total_debt_events"] == 5
    
    @patch('app.api.technical_debt.get_project_or_404')
    @patch('app.api.technical_debt.get_debt_monitor_integration')
    def test_start_monitoring_success(self, mock_get_monitor_integration, mock_get_project):
        """Test successful monitoring start."""
        # Setup mocks
        mock_get_project.return_value = self.project
        
        mock_monitor_integration = Mock()
        mock_monitor_integration.initialize_components = AsyncMock()
        mock_monitor_integration.start_monitoring_project = AsyncMock()
        mock_get_monitor_integration.return_value = mock_monitor_integration
        
        # Make request
        response = client.post(f"/api/technical-debt/{self.project_id}/monitoring/start")
        
        # Validate response
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert f"Debt monitoring started for project {self.project.name}" in data["message"]
        
        monitoring_data = data["data"]
        assert monitoring_data["project_id"] == self.project_id
        assert monitoring_data["files_monitored"] == len(self.project.file_entries)
        assert monitoring_data["monitoring_active"] == True
        
        # Verify mock calls
        mock_monitor_integration.initialize_components.assert_called_once()
        mock_monitor_integration.start_monitoring_project.assert_called_once_with(self.project)
    
    @patch('app.api.technical_debt.get_project_or_404')
    @patch('app.api.technical_debt.get_debt_monitor_integration')
    def test_stop_monitoring_success(self, mock_get_monitor_integration, mock_get_project):
        """Test successful monitoring stop."""
        # Setup mocks
        mock_get_project.return_value = self.project
        
        mock_monitor_integration = Mock()
        mock_monitor_integration.stop_monitoring_project = AsyncMock()
        mock_get_monitor_integration.return_value = mock_monitor_integration
        
        # Make request
        response = client.post(f"/api/technical-debt/{self.project_id}/monitoring/stop")
        
        # Validate response
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert f"Debt monitoring stopped for project {self.project.name}" in data["message"]
        
        monitoring_data = data["data"]
        assert monitoring_data["project_id"] == self.project_id
        assert monitoring_data["monitoring_active"] == False
        
        # Verify mock calls
        mock_monitor_integration.stop_monitoring_project.assert_called_once_with(self.project_id)
    
    @patch('app.api.technical_debt.get_project_or_404')
    @patch('app.api.technical_debt.get_debt_monitor_integration')
    def test_force_debt_analysis_success(self, mock_get_monitor_integration, mock_get_project):
        """Test successful forced debt analysis."""
        # Setup mocks
        mock_get_project.return_value = self.project
        
        mock_result = {
            'project_id': self.project_id,
            'files_analyzed': 3,
            'total_debt_score': 1.2,
            'debt_items_found': 4,
            'analysis_duration': 1.5,
            'category_breakdown': {'complexity': 0.8, 'duplication': 0.4}
        }
        
        mock_monitor_integration = Mock()
        mock_monitor_integration.force_debt_analysis = AsyncMock(return_value=mock_result)
        mock_get_monitor_integration.return_value = mock_monitor_integration
        
        # Make request
        response = client.post(
            f"/api/technical-debt/{self.project_id}/analyze/force?file_paths=/test/project/file_0.py&file_paths=/test/project/file_1.py"
        )
        
        # Validate response
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "Forced debt analysis completed successfully" in data["message"]
        assert data["data"] == mock_result
        
        # Verify mock calls
        mock_monitor_integration.force_debt_analysis.assert_called_once()
    
    def test_health_check(self):
        """Test API health check endpoint."""
        response = client.get("/api/technical-debt/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "Technical Debt API is healthy" in data["message"]
        
        health_data = data["data"]
        assert health_data["status"] == "healthy"
        assert health_data["version"] == "1.0.0"
        assert "timestamp" in health_data


# ================== INTEGRATION TESTS ==================

class TestTechnicalDebtAPIIntegration:
    """Integration tests for technical debt API."""
    
    def test_api_error_handling(self):
        """Test API error handling for various scenarios."""
        # Test with invalid UUID
        response = client.post("/api/technical-debt/invalid-uuid/analyze", json={})
        assert response.status_code == 422
        
        # Test with missing request body
        valid_uuid = str(uuid.uuid4())
        response = client.post(f"/api/technical-debt/{valid_uuid}/analyze")
        assert response.status_code == 422
    
    def test_request_validation(self):
        """Test request validation for different endpoints."""
        valid_uuid = str(uuid.uuid4())
        
        # Test analysis request validation
        invalid_analysis_request = {
            "include_advanced_patterns": "not_boolean",
            "analysis_depth": "invalid_depth",
            "file_patterns": "not_list"
        }
        response = client.post(
            f"/api/technical-debt/{valid_uuid}/analyze",
            json=invalid_analysis_request
        )
        assert response.status_code == 422
    
    def test_response_format_consistency(self):
        """Test that all endpoints return consistent response format."""
        # Health check should return standard format
        response = client.get("/api/technical-debt/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "success" in data
        assert "message" in data
        assert "data" in data
        assert "timestamp" in data


# Run the tests
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])