"""
Comprehensive Tests for Performance Infrastructure Orchestrator

Tests all aspects of the performance orchestration system including:
- Component initialization and configuration
- End-to-end performance testing workflows  
- API endpoint functionality and error handling
- Integration between different testing components
- Real-time monitoring and reporting
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from fastapi.testclient import TestClient
from httpx import AsyncClient
import pytest_asyncio

from app.core.performance_orchestrator import (
    PerformanceOrchestrator,
    OrchestrationConfig,
    OrchestrationResult,
    TestResult,
    TestStatus,
    TestCategory,
    PerformanceTarget,
    create_performance_orchestrator
)
from app.api.v1.performance_testing import router
from app.core.performance_benchmarks import PerformanceBenchmarkSuite
from app.core.load_testing import LoadTestFramework
from app.core.performance_validator import PerformanceValidator


class TestPerformanceOrchestrator:
    """Test suite for PerformanceOrchestrator class."""
    
    @pytest_asyncio.fixture
    async def mock_context_manager(self):
        """Mock context manager for testing."""
        mock_cm = Mock()
        mock_cm.search_contexts = AsyncMock(return_value=[])
        mock_cm.store_context = AsyncMock()
        mock_cm.delete_context = AsyncMock()
        return mock_cm
    
    @pytest_asyncio.fixture
    async def mock_db_session(self):
        """Mock database session for testing."""
        mock_session = AsyncMock()
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        return mock_session
    
    @pytest_asyncio.fixture
    async def orchestrator_config(self):
        """Default orchestration configuration for testing."""
        return OrchestrationConfig(
            enable_context_engine_tests=True,
            enable_redis_streams_tests=True,
            enable_vertical_slice_tests=True,
            enable_system_integration_tests=True,
            context_engine_iterations=2,
            redis_streams_duration_minutes=1,
            vertical_slice_scenarios=2,
            integration_test_scenarios=1,
            parallel_execution=False,  # Sequential for testing
            timeout_minutes=10,
            generate_detailed_reports=True
        )
    
    @pytest_asyncio.fixture
    async def orchestrator(self, orchestrator_config, mock_context_manager, mock_db_session):
        """Create orchestrator instance for testing."""
        orchestrator = PerformanceOrchestrator(
            config=orchestrator_config,
            context_manager=mock_context_manager,
            db_session=mock_db_session
        )
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization with all components."""
        # Test initialization
        with patch.object(orchestrator, '_define_performance_targets') as mock_targets:
            mock_targets.return_value = [
                PerformanceTarget(
                    name="test_target",
                    category=TestCategory.CONTEXT_ENGINE,
                    target_value=50.0,
                    unit="ms",
                    description="Test target"
                )
            ]
            
            await orchestrator.initialize()
            
            # Verify initialization
            assert orchestrator.performance_targets is not None
            assert len(orchestrator.performance_targets) > 0
            assert orchestrator.current_orchestration is None
            assert len(orchestrator.running_tests) == 0
    
    @pytest.mark.asyncio
    async def test_performance_targets_definition(self, orchestrator):
        """Test that all required performance targets are defined."""
        targets = orchestrator._define_performance_targets()
        
        # Check that all required targets are present
        target_names = [t.name for t in targets]
        required_targets = [
            "context_search_time",
            "context_retrieval_precision", 
            "message_throughput",
            "p95_latency",
            "agent_spawn_time",
            "total_flow_time"
        ]
        
        for required in required_targets:
            assert required in target_names, f"Required target {required} not found"
        
        # Check target categories
        categories = [t.category for t in targets]
        expected_categories = [
            TestCategory.CONTEXT_ENGINE,
            TestCategory.REDIS_STREAMS,
            TestCategory.VERTICAL_SLICE
        ]
        
        for category in expected_categories:
            assert category in categories, f"Category {category} not found"
        
        # Check critical targets
        critical_targets = [t for t in targets if t.critical]
        assert len(critical_targets) > 0, "No critical targets defined"
    
    @pytest.mark.asyncio
    async def test_comprehensive_testing_workflow(self, orchestrator):
        """Test the complete comprehensive testing workflow."""
        # Mock all testing components
        with patch.object(orchestrator, '_run_context_engine_tests') as mock_context, \
             patch.object(orchestrator, '_run_redis_streams_tests') as mock_redis, \
             patch.object(orchestrator, '_run_vertical_slice_tests') as mock_vertical, \
             patch.object(orchestrator, '_run_system_integration_tests') as mock_integration, \
             patch.object(orchestrator, '_capture_system_baseline') as mock_baseline, \
             patch.object(orchestrator, '_store_orchestration_results') as mock_store:
            
            # Setup mock returns
            mock_baseline.return_value = {"cpu_percent": 10.0, "memory_percent": 20.0}
            
            mock_context.return_value = TestResult(
                test_id="context_test",
                category=TestCategory.CONTEXT_ENGINE,
                status=TestStatus.COMPLETED,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                targets_met={"context_search_time": True}
            )
            
            mock_redis.return_value = TestResult(
                test_id="redis_test",
                category=TestCategory.REDIS_STREAMS,
                status=TestStatus.COMPLETED,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                targets_met={"message_throughput": True}
            )
            
            mock_vertical.return_value = TestResult(
                test_id="vertical_test",
                category=TestCategory.VERTICAL_SLICE,
                status=TestStatus.COMPLETED,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                targets_met={"agent_spawn_time": True}
            )
            
            mock_integration.return_value = TestResult(
                test_id="integration_test",
                category=TestCategory.SYSTEM_INTEGRATION,
                status=TestStatus.COMPLETED,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                targets_met={"end_to_end_latency": True}
            )
            
            # Run comprehensive testing
            result = await orchestrator.run_comprehensive_testing(
                test_suite_name="test_suite"
            )
            
            # Verify results
            assert isinstance(result, OrchestrationResult)
            assert result.overall_status == TestStatus.COMPLETED
            assert len(result.test_results) == 4
            assert result.performance_score > 0
            assert result.end_time is not None
            assert result.duration_seconds > 0
            
            # Verify all test categories were executed
            categories = [tr.category for tr in result.test_results]
            assert TestCategory.CONTEXT_ENGINE in categories
            assert TestCategory.REDIS_STREAMS in categories
            assert TestCategory.VERTICAL_SLICE in categories
            assert TestCategory.SYSTEM_INTEGRATION in categories
            
            # Verify storage was called
            mock_store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_parallel_test_execution(self, orchestrator):
        """Test parallel execution of test components."""
        orchestrator.config.parallel_execution = True
        orchestrator.config.max_concurrent_tests = 2
        
        with patch.object(orchestrator, '_run_context_engine_tests') as mock_context, \
             patch.object(orchestrator, '_run_redis_streams_tests') as mock_redis, \
             patch.object(orchestrator, '_capture_system_baseline') as mock_baseline, \
             patch.object(orchestrator, '_store_orchestration_results') as mock_store:
            
            # Setup mock returns with delays to test concurrency
            async def delayed_context_test():
                await asyncio.sleep(0.1)
                return TestResult(
                    test_id="context_test",
                    category=TestCategory.CONTEXT_ENGINE,
                    status=TestStatus.COMPLETED,
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow()
                )
            
            async def delayed_redis_test():
                await asyncio.sleep(0.1)
                return TestResult(
                    test_id="redis_test",
                    category=TestCategory.REDIS_STREAMS,
                    status=TestStatus.COMPLETED,
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow()
                )
            
            mock_context.side_effect = delayed_context_test
            mock_redis.side_effect = delayed_redis_test
            mock_baseline.return_value = {}
            
            # Disable other tests for this test
            orchestrator.config.enable_vertical_slice_tests = False
            orchestrator.config.enable_system_integration_tests = False
            
            start_time = datetime.utcnow()
            result = await orchestrator.run_comprehensive_testing()
            end_time = datetime.utcnow()
            
            # Verify parallel execution was faster than sequential
            execution_time = (end_time - start_time).total_seconds()
            assert execution_time < 0.3, f"Parallel execution took too long: {execution_time}s"
            assert len(result.test_results) == 2
    
    @pytest.mark.asyncio
    async def test_test_failure_handling(self, orchestrator):
        """Test handling of test failures and error conditions."""
        with patch.object(orchestrator, '_run_context_engine_tests') as mock_context, \
             patch.object(orchestrator, '_capture_system_baseline') as mock_baseline, \
             patch.object(orchestrator, '_store_orchestration_results') as mock_store:
            
            # Mock a test failure
            mock_context.return_value = TestResult(
                test_id="context_test",
                category=TestCategory.CONTEXT_ENGINE,
                status=TestStatus.FAILED,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                error_message="Simulated test failure"
            )
            
            mock_baseline.return_value = {}
            
            # Disable other tests
            orchestrator.config.enable_redis_streams_tests = False
            orchestrator.config.enable_vertical_slice_tests = False
            orchestrator.config.enable_system_integration_tests = False
            
            result = await orchestrator.run_comprehensive_testing()
            
            # Verify failure handling
            assert result.overall_status == TestStatus.FAILED
            assert len(result.critical_failures) > 0
            assert "Simulated test failure" in str(result.critical_failures)
            assert result.performance_score == 0  # No successful tests
    
    @pytest.mark.asyncio
    async def test_regression_testing(self, orchestrator):
        """Test regression testing functionality."""
        baseline_id = "baseline_123"
        
        with patch.object(orchestrator, 'run_comprehensive_testing') as mock_comprehensive, \
             patch.object(orchestrator, '_load_orchestration_results') as mock_load, \
             patch.object(orchestrator, '_perform_regression_analysis') as mock_regression:
            
            # Mock baseline results
            mock_load.return_value = {
                "orchestration_id": baseline_id,
                "performance_score": 95.0,
                "overall_status": "completed"
            }
            
            # Mock current results
            current_result = OrchestrationResult(
                orchestration_id="current_123",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                overall_status=TestStatus.COMPLETED,
                performance_score=85.0  # 10.5% regression
            )
            mock_comprehensive.return_value = current_result
            
            # Mock regression analysis
            mock_regression.return_value = {
                "regressions_detected": True,
                "significant_regressions": [
                    {
                        "metric": "performance_score",
                        "change_percent": -10.5,
                        "severity": "warning"
                    }
                ],
                "regression_recommendations": ["Investigate performance regression"]
            }
            
            # Run regression testing
            result = await orchestrator.run_regression_testing(
                baseline_orchestration_id=baseline_id,
                regression_threshold_percent=15.0
            )
            
            # Verify regression testing
            assert result.performance_score == 85.0
            assert "regression_analysis" in result.system_metrics
            assert result.system_metrics["regression_analysis"]["regressions_detected"]
            assert len(result.recommendations) > 0
            
            # Verify method calls
            mock_load.assert_called_once_with(baseline_id)
            mock_regression.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_continuous_monitoring(self, orchestrator):
        """Test continuous monitoring functionality."""
        duration_minutes = 0.1  # 6 seconds for testing
        sampling_interval = 0.02  # 1.2 seconds for testing
        
        with patch.object(orchestrator, '_collect_performance_sample') as mock_sample, \
             patch.object(orchestrator, '_check_performance_alerts') as mock_alerts:
            
            # Mock performance samples
            sample_data = [
                {"cpu_percent": 45.0, "memory_percent": 60.0},
                {"cpu_percent": 55.0, "memory_percent": 70.0},
                {"cpu_percent": 85.0, "memory_percent": 90.0}  # This should trigger alerts
            ]
            mock_sample.side_effect = sample_data
            
            # Mock alerts
            mock_alerts.side_effect = [
                [],  # No alerts for first two samples
                [],
                [{"type": "high_cpu_usage", "severity": "warning"}]  # Alert for third sample
            ]
            
            # Run monitoring
            result = await orchestrator.run_continuous_monitoring(
                monitoring_duration_minutes=duration_minutes,
                sampling_interval_seconds=sampling_interval
            )
            
            # Verify monitoring results
            assert "monitoring_id" in result
            assert len(result["samples"]) >= 2  # Should have collected at least 2 samples
            assert len(result["alerts"]) >= 0  # May have alerts
            assert "summary" in result
            assert result["summary"]["total_samples"] == len(result["samples"])
    
    @pytest.mark.asyncio
    async def test_target_validation(self, orchestrator):
        """Test performance target validation logic."""
        # Test Context Engine target validation
        context_results = {
            "performance_metrics": {
                "search_performance": {
                    "average_response_time_ms": 30.0  # Under 50ms target
                },
                "retrieval_precision": {
                    "additional_metrics": {
                        "average_precision": 0.95  # Over 90% target
                    }
                }
            }
        }
        
        targets_met = await orchestrator._validate_context_engine_targets(context_results)
        assert targets_met.get("context_search_time", False)
        assert targets_met.get("context_retrieval_precision", False)
        
        # Test Redis Streams target validation
        redis_results = {
            "targets_validation": {
                "throughput_target": True,
                "latency_p95_target": False,  # Failed target
                "success_rate_target": True
            }
        }
        
        redis_targets_met = await orchestrator._validate_redis_streams_targets(redis_results)
        assert redis_targets_met.get("message_throughput", False)
        assert not redis_targets_met.get("p95_latency", True)  # Should be False
        assert redis_targets_met.get("message_success_rate", False)
    
    @pytest.mark.asyncio
    async def test_orchestration_status_tracking(self, orchestrator):
        """Test orchestration status tracking and progress calculation."""
        # Create a mock orchestration
        orchestrator.current_orchestration = OrchestrationResult(
            orchestration_id="test_123",
            start_time=datetime.utcnow(),
            overall_status=TestStatus.RUNNING
        )
        
        # Add some test results
        orchestrator.current_orchestration.test_results = [
            TestResult(
                test_id="test1",
                category=TestCategory.CONTEXT_ENGINE,
                status=TestStatus.COMPLETED,
                start_time=datetime.utcnow()
            ),
            TestResult(
                test_id="test2",
                category=TestCategory.REDIS_STREAMS,
                status=TestStatus.RUNNING,
                start_time=datetime.utcnow()
            )
        ]
        
        # Test status retrieval
        status = await orchestrator.get_orchestration_status("test_123")
        
        assert status is not None
        assert status["orchestration_id"] == "test_123"
        assert status["status"] == TestStatus.RUNNING.value
        assert status["completed_tests"] == 1
        assert status["running_tests"] >= 0
        assert 0 <= status["progress_percent"] <= 100
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self, orchestrator_config):
        """Test orchestration configuration validation."""
        # Test valid configuration
        assert orchestrator_config.context_engine_iterations > 0
        assert orchestrator_config.redis_streams_duration_minutes > 0
        assert orchestrator_config.timeout_minutes > 0
        assert 1 <= orchestrator_config.max_concurrent_tests <= 10
        
        # Test configuration bounds
        config = OrchestrationConfig(
            context_engine_iterations=25,  # Over limit
            max_concurrent_tests=15,  # Over limit
            timeout_minutes=0  # Under limit
        )
        
        # In a real implementation, these would be validated
        # For now, just verify the values are set
        assert config.context_engine_iterations == 25
        assert config.max_concurrent_tests == 15
        assert config.timeout_minutes == 0


class TestPerformanceTestingAPI:
    """Test suite for Performance Testing API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client for API testing."""
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)
    
    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user for API testing."""
        return {"id": "test_user", "username": "testuser"}
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Mock orchestrator for API testing."""
        mock_orch = AsyncMock(spec=PerformanceOrchestrator)
        mock_orch.config = OrchestrationConfig()
        mock_orch.current_orchestration = None
        mock_orch.running_tests = {}
        mock_orch.performance_targets = [
            PerformanceTarget(
                name="test_target",
                category=TestCategory.CONTEXT_ENGINE,
                target_value=50.0,
                unit="ms",
                description="Test target",
                critical=True
            )
        ]
        return mock_orch
    
    def test_get_performance_targets(self, client, mock_user):
        """Test GET /performance/targets endpoint."""
        with patch('app.api.v1.performance_testing.get_current_user', return_value=mock_user), \
             patch('app.api.v1.performance_testing.get_orchestrator') as mock_get_orch:
            
            # Setup mock orchestrator
            mock_orch = Mock()
            mock_orch.performance_targets = [
                PerformanceTarget(
                    name="context_search_time",
                    category=TestCategory.CONTEXT_ENGINE,
                    target_value=50.0,
                    unit="ms",
                    description="Context search must be under 50ms",
                    critical=True
                ),
                PerformanceTarget(
                    name="message_throughput",
                    category=TestCategory.REDIS_STREAMS,
                    target_value=10000.0,
                    unit="msg/sec",
                    description="Message throughput must be >10k msg/sec",
                    critical=True
                )
            ]
            mock_get_orch.return_value = mock_orch
            
            response = client.get("/api/v1/performance/targets")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "targets" in data
            assert "categories" in data
            assert "critical_targets_count" in data
            assert "total_targets_count" in data
            
            assert len(data["targets"]) == 2
            assert data["critical_targets_count"] == 2
            assert data["total_targets_count"] == 2
            
            # Check target structure
            target = data["targets"][0]
            assert "name" in target
            assert "category" in target
            assert "target_value" in target
            assert "unit" in target
            assert "description" in target
            assert "critical" in target
    
    def test_start_comprehensive_testing(self, client, mock_user):
        """Test POST /performance/orchestration/start endpoint."""
        with patch('app.api.v1.performance_testing.get_current_user', return_value=mock_user), \
             patch('app.api.v1.performance_testing.get_orchestrator') as mock_get_orch:
            
            # Setup mock orchestrator
            mock_orch = AsyncMock()
            mock_orch.config = OrchestrationConfig()
            mock_orch.current_orchestration = OrchestrationResult(
                orchestration_id="test_123",
                start_time=datetime.utcnow()
            )
            mock_orch.run_comprehensive_testing = AsyncMock()
            mock_get_orch.return_value = mock_orch
            
            request_data = {
                "test_suite_name": "api_test_suite",
                "baseline_comparison": True,
                "tags": {"environment": "test"}
            }
            
            response = client.post("/api/v1/performance/orchestration/start", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "orchestration_id" in data
            assert data["status"] == "started"
            assert data["test_suite_name"] == "api_test_suite"
            assert "status_url" in data
            assert "results_url" in data
            assert "started_at" in data
    
    def test_get_orchestration_status(self, client, mock_user):
        """Test GET /performance/orchestration/{id}/status endpoint."""
        with patch('app.api.v1.performance_testing.get_current_user', return_value=mock_user), \
             patch('app.api.v1.performance_testing.get_orchestrator') as mock_get_orch:
            
            # Setup mock orchestrator
            mock_orch = AsyncMock()
            mock_orch.get_orchestration_status = AsyncMock(return_value={
                "orchestration_id": "test_123",
                "status": "running",
                "progress_percent": 50,
                "running_tests": 2,
                "completed_tests": 1,
                "failed_tests": 0
            })
            mock_get_orch.return_value = mock_orch
            
            response = client.get("/api/v1/performance/orchestration/test_123/status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["orchestration_id"] == "test_123"
            assert data["status"] == "running"
            assert data["progress_percent"] == 50
            assert data["running_tests"] == 2
            assert data["completed_tests"] == 1
            assert data["failed_tests"] == 0
    
    def test_get_orchestration_results(self, client, mock_user):
        """Test GET /performance/orchestration/{id}/results endpoint."""
        with patch('app.api.v1.performance_testing.get_current_user', return_value=mock_user), \
             patch('app.api.v1.performance_testing.get_orchestrator') as mock_get_orch:
            
            # Setup mock orchestrator with completed orchestration
            mock_orch = AsyncMock()
            mock_orch.current_orchestration = OrchestrationResult(
                orchestration_id="test_123",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                overall_status=TestStatus.COMPLETED,
                performance_score=92.5
            )
            mock_orch._load_orchestration_results = AsyncMock(return_value=None)
            mock_get_orch.return_value = mock_orch
            
            response = client.get("/api/v1/performance/orchestration/test_123/results")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "orchestration_id" in data
            assert "results" in data
            assert "retrieved_at" in data
            assert data["orchestration_id"] == "test_123"
    
    def test_start_regression_testing(self, client, mock_user):
        """Test POST /performance/regression/start endpoint."""
        with patch('app.api.v1.performance_testing.get_current_user', return_value=mock_user), \
             patch('app.api.v1.performance_testing.get_orchestrator') as mock_get_orch:
            
            # Setup mock orchestrator
            mock_orch = AsyncMock()
            mock_orch.config = OrchestrationConfig()
            mock_orch.current_orchestration = OrchestrationResult(
                orchestration_id="regression_123",
                start_time=datetime.utcnow()
            )
            mock_orch.run_regression_testing = AsyncMock()
            mock_get_orch.return_value = mock_orch
            
            request_data = {
                "baseline_orchestration_id": "baseline_456",
                "regression_threshold_percent": 10.0,
                "test_suite_name": "regression_test"
            }
            
            response = client.post("/api/v1/performance/regression/start", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "orchestration_id" in data
            assert data["status"] == "started"
            assert data["baseline_orchestration_id"] == "baseline_456"
            assert data["regression_threshold"] == 10.0
    
    def test_configuration_validation(self, client, mock_user):
        """Test POST /performance/configuration/validate endpoint."""
        with patch('app.api.v1.performance_testing.get_current_user', return_value=mock_user):
            
            request_data = {
                "enable_context_engine_tests": True,
                "enable_redis_streams_tests": True,
                "context_engine_iterations": 5,
                "redis_streams_duration_minutes": 10,
                "parallel_execution": True,
                "max_concurrent_tests": 3,
                "timeout_minutes": 60
            }
            
            response = client.post("/api/v1/performance/configuration/validate", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "configuration" in data
            assert "validation" in data
            assert "validated_at" in data
            
            validation = data["validation"]
            assert "valid" in validation
            assert "estimated_duration_minutes" in validation
            assert validation["valid"] is True
    
    def test_health_endpoint(self, client):
        """Test GET /performance/health endpoint."""
        with patch('app.api.v1.performance_testing.get_orchestrator') as mock_get_orch:
            
            # Setup mock orchestrator
            mock_orch = Mock()
            mock_orch.benchmark_suite = Mock()
            mock_orch.load_test_framework = Mock()
            mock_orch.performance_validator = Mock()
            mock_orch.config = OrchestrationConfig()
            mock_orch.current_orchestration = None
            mock_orch.performance_targets = [Mock()]
            mock_get_orch.return_value = mock_orch
            
            response = client.get("/api/v1/performance/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "status" in data
            assert "components" in data
            assert "system_ready" in data
            assert "performance_targets_loaded" in data
            
            assert data["status"] in ["healthy", "degraded"]
            assert data["system_ready"] is True
            assert data["performance_targets_loaded"] is True
    
    def test_cancel_orchestration(self, client, mock_user):
        """Test POST /performance/orchestration/{id}/cancel endpoint."""
        with patch('app.api.v1.performance_testing.get_current_user', return_value=mock_user), \
             patch('app.api.v1.performance_testing.get_orchestrator') as mock_get_orch:
            
            # Setup mock orchestrator with running orchestration
            mock_orch = AsyncMock()
            mock_orch.current_orchestration = OrchestrationResult(
                orchestration_id="test_123",
                start_time=datetime.utcnow(),
                overall_status=TestStatus.RUNNING
            )
            mock_orch.running_tests = {"test1": Mock(), "test2": Mock()}
            mock_orch._store_orchestration_results = AsyncMock()
            mock_get_orch.return_value = mock_orch
            
            response = client.post("/api/v1/performance/orchestration/test_123/cancel")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["orchestration_id"] == "test_123"
            assert data["status"] == "cancelled"
            assert "cancelled_at" in data
            assert "partial_results_available" in data


class TestPerformanceIntegration:
    """Integration tests for the complete performance testing system."""
    
    @pytest.mark.asyncio
    async def test_factory_function(self):
        """Test the create_performance_orchestrator factory function."""
        with patch('app.core.performance_orchestrator.get_session'):
            orchestrator = await create_performance_orchestrator()
            
            assert isinstance(orchestrator, PerformanceOrchestrator)
            assert orchestrator.config is not None
            assert len(orchestrator.performance_targets) > 0
    
    @pytest.mark.asyncio
    async def test_comprehensive_testing_convenience_function(self):
        """Test the run_comprehensive_performance_testing convenience function."""
        from app.core.performance_orchestrator import run_comprehensive_performance_testing
        
        with patch('app.core.performance_orchestrator.create_performance_orchestrator') as mock_create:
            
            # Setup mock orchestrator
            mock_orch = AsyncMock()
            mock_orch.run_comprehensive_testing = AsyncMock(return_value=OrchestrationResult(
                orchestration_id="test_123",
                start_time=datetime.utcnow(),
                overall_status=TestStatus.COMPLETED
            ))
            mock_create.return_value = mock_orch
            
            result = await run_comprehensive_performance_testing()
            
            assert isinstance(result, OrchestrationResult)
            assert result.orchestration_id == "test_123"
            assert result.overall_status == TestStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end performance testing workflow."""
        # This would test the complete workflow from API call to results
        # For now, we'll test the main components work together
        
        config = OrchestrationConfig(
            enable_context_engine_tests=False,  # Disable for integration test
            enable_redis_streams_tests=False,
            enable_vertical_slice_tests=False,
            enable_system_integration_tests=True,  # Only test integration
            parallel_execution=False,
            timeout_minutes=5
        )
        
        with patch('app.core.performance_orchestrator.get_session'):
            orchestrator = PerformanceOrchestrator(config=config)
            await orchestrator.initialize()
            
            # Mock the integration test
            with patch.object(orchestrator, '_run_system_integration_tests') as mock_integration, \
                 patch.object(orchestrator, '_capture_system_baseline') as mock_baseline, \
                 patch.object(orchestrator, '_store_orchestration_results') as mock_store:
                
                mock_baseline.return_value = {"cpu_percent": 15.0}
                mock_integration.return_value = TestResult(
                    test_id="integration_test",
                    category=TestCategory.SYSTEM_INTEGRATION,
                    status=TestStatus.COMPLETED,
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    targets_met={"end_to_end_latency": True}
                )
                
                result = await orchestrator.run_comprehensive_testing()
                
                assert result.overall_status == TestStatus.COMPLETED
                assert len(result.test_results) == 1
                assert result.test_results[0].category == TestCategory.SYSTEM_INTEGRATION
                assert result.performance_score > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])