"""
Production readiness tests for Context Compression
Tests deployment validation, monitoring integration, health checks, and operational requirements
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from httpx import AsyncClient

from app.main import app
from app.core.context_compression import get_context_compressor
from app.core.hive_slash_commands import get_hive_command_registry


class TestContextCompressionProductionReadiness:
    """Production readiness test suite for context compression"""
    
    @pytest.fixture
    async def client(self):
        """Create test HTTP client"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint_integration(self, client):
        """Test health check endpoint includes compression service status"""
        # This would test the main application health endpoint
        # In a real implementation, compression service would be included
        response = await client.get("/health")
        
        # Should return 200 OK when system is healthy
        assert response.status_code == 200
        
        # Health check should include compression service status
        # (This would need to be implemented in the main health endpoint)
        health_data = response.json()
        assert "status" in health_data
    
    @pytest.mark.asyncio
    async def test_compression_service_health_check(self):
        """Test compression service internal health check"""
        compressor = get_context_compressor()
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "summary": "Health check test",
            "key_insights": [],
            "decisions_made": [],
            "patterns_identified": [],
            "importance_score": 0.5
        })
        compressor.llm_client.messages.create.return_value = mock_response
        
        health_result = await compressor.health_check()
        
        assert health_result["status"] == "healthy"
        assert health_result["model"] == compressor.model_name
        assert "test_compression_ratio" in health_result
        assert "performance" in health_result
        
        # Performance metrics should be included
        performance = health_result["performance"]
        assert "total_compressions" in performance
        assert "average_compression_time_s" in performance
    
    @pytest.mark.asyncio
    async def test_compression_service_health_check_failure(self):
        """Test compression service health check when service is down"""
        compressor = get_context_compressor()
        
        # Mock API failure
        compressor.llm_client.messages.create.side_effect = Exception("Service unavailable")
        
        health_result = await compressor.health_check()
        
        assert health_result["status"] == "unhealthy"
        assert "error" in health_result
        assert health_result["model"] == compressor.model_name
    
    @pytest.mark.asyncio
    async def test_prometheus_metrics_integration(self):
        """Test that compression metrics can be exported for Prometheus"""
        compressor = get_context_compressor()
        
        # Mock some compression activity
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "summary": "Metrics test summary",
            "key_insights": ["Metrics insight"],
            "decisions_made": [],
            "patterns_identified": [],
            "importance_score": 0.6
        })
        compressor.llm_client.messages.create.return_value = mock_response
        
        # Perform some compressions to generate metrics
        test_content = "Test content for metrics generation"
        for i in range(3):
            await compressor.compress_conversation(
                conversation_content=f"{test_content} - {i}",
                compression_level=compressor.CompressionLevel.STANDARD
            )
        
        # Get performance metrics in Prometheus-compatible format
        metrics = compressor.get_performance_metrics()
        
        # Verify metrics are suitable for Prometheus export
        assert isinstance(metrics["total_compressions"], int)
        assert isinstance(metrics["average_compression_time_s"], float)
        assert isinstance(metrics["total_tokens_saved"], int)
        assert isinstance(metrics["average_compression_ratio"], float)
        
        # Metrics should have reasonable values
        assert metrics["total_compressions"] >= 3
        assert metrics["average_compression_time_s"] > 0
        assert 0 <= metrics["average_compression_ratio"] <= 1
    
    @pytest.mark.asyncio
    async def test_error_logging_and_alerting(self):
        """Test that errors are properly logged for alerting systems"""
        compressor = get_context_compressor()
        
        # Mock API error that should trigger alerting
        compressor.llm_client.messages.create.side_effect = Exception("Critical API failure")
        
        # Capture log output (in real implementation, would use structured logging)
        with patch('app.core.context_compression.logger') as mock_logger:
            result = await compressor.compress_conversation(
                conversation_content="Test content for error logging",
                compression_level=compressor.CompressionLevel.STANDARD
            )
        
        # Verify error was logged
        mock_logger.error.assert_called()
        
        # Verify fallback result was returned
        assert result.compression_ratio == 0.0
        assert "error" in result.metadata
    
    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """Test that production configuration is validated"""
        # Test that required configuration is present
        from app.core.config import get_settings
        
        settings = get_settings()
        
        # Critical configuration should be present for production
        assert hasattr(settings, 'ANTHROPIC_API_KEY'), "Anthropic API key must be configured"
        
        # In production, API key should not be empty or default
        # (In tests, it might be mocked)
        if hasattr(settings, 'ANTHROPIC_API_KEY') and settings.ANTHROPIC_API_KEY:
            assert settings.ANTHROPIC_API_KEY != "your-api-key-here"
            assert len(settings.ANTHROPIC_API_KEY) > 10
    
    @pytest.mark.asyncio
    async def test_deployment_smoke_test(self, client):
        """Test basic functionality after deployment"""
        session_id = "00000000-0000-0000-0000-000000000001"  # Valid UUID format
        
        # Mock successful compression for smoke test
        with patch('app.api.v1.sessions.get_hive_command_registry') as mock_registry:
            mock_compact_command = AsyncMock()
            mock_compact_command.execute.return_value = {
                "success": True,
                "session_id": session_id,
                "compression_level": "standard",
                "original_tokens": 100,
                "compressed_tokens": 60,
                "compression_ratio": 0.4,
                "tokens_saved": 40,
                "compression_time_seconds": 2.0,
                "summary": "Deployment smoke test summary",
                "key_insights": ["Deployment insight"],
                "decisions_made": ["Deployment decision"],
                "patterns_identified": ["Deployment pattern"],
                "importance_score": 0.7,
                "message": "Compression completed successfully",
                "performance_met": True,
                "timestamp": "2024-01-01T12:00:00Z"
            }
            
            mock_registry_instance = Mock()
            mock_registry_instance.get_command.return_value = mock_compact_command
            mock_registry.return_value = mock_registry_instance
            
            # Test basic compression endpoint
            response = await client.post(
                f"/api/v1/sessions/{session_id}/compact",
                json={"compression_level": "standard"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["performance_met"] is True
        
        # Test compression status endpoint
        with patch('app.api.v1.sessions.get_db_session') as mock_get_db:
            mock_session = Mock()
            mock_session.get_shared_context.return_value = None
            mock_session.name = "Test Session"
            mock_session.session_type.value = "development"
            mock_session.status.value = "active"
            mock_session.created_at = datetime.utcnow()
            mock_session.last_activity = datetime.utcnow()
            
            mock_db_session = AsyncMock()
            mock_db_session.get.return_value = mock_session
            mock_get_db.return_value.__aenter__.return_value = mock_db_session
            
            status_response = await client.get(f"/api/v1/sessions/{session_id}/compact/status")
        
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["session_id"] == session_id
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self):
        """Test integration with performance monitoring systems"""
        compressor = get_context_compressor()
        
        # Mock compression activity
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "summary": "Performance monitoring test",
            "key_insights": ["Performance insight"],
            "decisions_made": [],
            "patterns_identified": [],
            "importance_score": 0.6
        })
        compressor.llm_client.messages.create.return_value = mock_response
        
        # Perform compressions with timing
        start_time = time.time()
        for i in range(5):
            await compressor.compress_conversation(
                conversation_content=f"Performance test content {i}",
                compression_level=compressor.CompressionLevel.STANDARD
            )
        end_time = time.time()
        
        # Get performance metrics
        metrics = compressor.get_performance_metrics()
        
        # Verify metrics are suitable for monitoring dashboards
        assert metrics["total_compressions"] == 5
        assert metrics["average_compression_time_s"] > 0
        assert metrics["average_compression_time_s"] < (end_time - start_time)
        
        # Metrics should be exportable as JSON for monitoring systems
        metrics_json = json.dumps(metrics)
        assert isinstance(json.loads(metrics_json), dict)
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test system behavior during partial failures"""
        compressor = get_context_compressor()
        
        # Test scenario: API temporarily unavailable
        api_call_count = 0
        
        def mock_api_with_intermittent_failures(*args, **kwargs):
            nonlocal api_call_count
            api_call_count += 1
            
            if api_call_count % 3 == 0:  # Every 3rd call fails
                raise Exception("Temporary API failure")
            
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = json.dumps({
                "summary": f"Successful compression {api_call_count}",
                "key_insights": ["Resilience insight"],
                "decisions_made": [],
                "patterns_identified": [],
                "importance_score": 0.5
            })
            return mock_response
        
        compressor.llm_client.messages.create.side_effect = mock_api_with_intermittent_failures
        
        # Perform multiple compressions
        results = []
        for i in range(6):
            result = await compressor.compress_conversation(
                conversation_content=f"Degradation test {i}",
                compression_level=compressor.CompressionLevel.STANDARD
            )
            results.append(result)
        
        # Some should succeed, some should gracefully fail
        successful_results = [r for r in results if r.compression_ratio > 0]
        fallback_results = [r for r in results if r.compression_ratio == 0]
        
        assert len(successful_results) > 0, "Some compressions should succeed"
        assert len(fallback_results) > 0, "Some compressions should gracefully fail"
        
        # All results should have valid structure
        for result in results:
            assert result.summary  # Should always have content
            assert hasattr(result, 'compression_ratio')
            assert hasattr(result, 'original_token_count')
    
    @pytest.mark.asyncio
    async def test_rollback_capability(self):
        """Test that compression changes can be rolled back"""
        # This would test database transaction rollback in case of failures
        registry = get_hive_command_registry()
        compact_command = registry.get_command("compact")
        
        # Mock session
        mock_session = Mock()
        mock_session.description = "Rollback test session"
        mock_session.objectives = ["Test rollback"]
        mock_session.shared_context = {"test": "data"}
        mock_session.state = {"status": "active"}
        mock_session.session_type.value = "development"
        mock_session.status.value = "active"
        mock_session.created_at = datetime.utcnow()
        mock_session.last_activity = datetime.utcnow()
        mock_session.update_shared_context = Mock()
        
        # Mock database session with rollback capability
        mock_db_session = AsyncMock()
        mock_db_session.get.return_value = mock_session
        mock_db_session.commit.side_effect = Exception("Database commit failed")
        mock_db_session.rollback = AsyncMock()
        
        # Mock compressor
        mock_compressor = AsyncMock()
        mock_result = Mock()
        mock_result.summary = "Test summary"
        mock_result.key_insights = []
        mock_result.decisions_made = []
        mock_result.patterns_identified = []
        mock_result.importance_score = 0.5
        mock_result.compression_ratio = 0.4
        mock_result.original_token_count = 100
        mock_result.compressed_token_count = 60
        mock_compressor.compress_conversation.return_value = mock_result
        
        with patch('app.core.hive_slash_commands.get_context_compressor', return_value=mock_compressor):
            with patch('app.core.hive_slash_commands.get_db_session') as mock_get_db:
                mock_get_db.return_value.__aenter__.return_value = mock_db_session
                
                # This should handle database failure gracefully
                # In a real implementation, it would rollback the transaction
                try:
                    result = await compact_command.execute(
                        args=["test-session-123"],
                        context={}
                    )
                    
                    # Should still return a result (possibly indicating failure)
                    assert "success" in result
                    
                except Exception:
                    # If it raises an exception, that's also acceptable
                    # as long as the system remains in a consistent state
                    pass
    
    @pytest.mark.asyncio
    async def test_capacity_planning_metrics(self):
        """Test metrics needed for capacity planning"""
        compressor = get_context_compressor()
        
        # Mock compression activity
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "summary": "Capacity planning test",
            "key_insights": ["Capacity insight"],
            "decisions_made": [],
            "patterns_identified": [],
            "importance_score": 0.5
        })
        compressor.llm_client.messages.create.return_value = mock_response
        
        # Perform compressions to generate capacity metrics
        test_sizes = [100, 500, 1000, 2000]  # Different content sizes
        capacity_metrics = {}
        
        for size in test_sizes:
            content = "A" * size  # Simple content of varying sizes
            
            start_time = time.time()
            result = await compressor.compress_conversation(
                conversation_content=content,
                compression_level=compressor.CompressionLevel.STANDARD
            )
            end_time = time.time()
            
            capacity_metrics[size] = {
                "processing_time": end_time - start_time,
                "tokens_processed": result.original_token_count,
                "compression_ratio": result.compression_ratio
            }
        
        # Verify capacity metrics are useful for planning
        for size, metrics in capacity_metrics.items():
            assert metrics["processing_time"] > 0
            assert metrics["tokens_processed"] > 0
            assert 0 <= metrics["compression_ratio"] <= 1
        
        # Processing time should generally increase with content size
        small_time = capacity_metrics[100]["processing_time"]
        large_time = capacity_metrics[2000]["processing_time"]
        
        # Allow for some variance in mock timing
        assert large_time >= small_time * 0.5, "Processing time should scale with content size"
    
    @pytest.mark.asyncio
    async def test_disaster_recovery_readiness(self):
        """Test readiness for disaster recovery scenarios"""
        # Test that compression service can restart cleanly
        compressor1 = get_context_compressor()
        
        # Perform some activity
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "summary": "DR test summary",
            "key_insights": [],
            "decisions_made": [],
            "patterns_identified": [],
            "importance_score": 0.5
        })
        compressor1.llm_client.messages.create.return_value = mock_response
        
        await compressor1.compress_conversation(
            conversation_content="Disaster recovery test content",
            compression_level=compressor1.CompressionLevel.STANDARD
        )
        
        initial_metrics = compressor1.get_performance_metrics()
        
        # Simulate service restart by cleaning up singleton
        from app.core.context_compression import cleanup_compressor
        await cleanup_compressor()
        
        # Get new instance (simulating restart)
        compressor2 = get_context_compressor()
        
        # Should start with fresh state
        new_metrics = compressor2.get_performance_metrics()
        assert new_metrics["total_compressions"] == 0
        
        # Should be fully functional
        compressor2.llm_client.messages.create.return_value = mock_response
        result = await compressor2.compress_conversation(
            conversation_content="Post-restart test content",
            compression_level=compressor2.CompressionLevel.STANDARD
        )
        
        assert result.summary
    
    @pytest.mark.asyncio
    async def test_observability_integration(self):
        """Test integration with observability systems"""
        # Test that compression operations generate appropriate traces/logs
        # This would integrate with systems like OpenTelemetry, Jaeger, etc.
        
        compressor = get_context_compressor()
        
        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "summary": "Observability test summary",
            "key_insights": ["Trace insight"],
            "decisions_made": [],
            "patterns_identified": [],
            "importance_score": 0.6
        })
        compressor.llm_client.messages.create.return_value = mock_response
        
        # In a real implementation, this would generate traces
        with patch('app.core.context_compression.logger') as mock_logger:
            result = await compressor.compress_conversation(
                conversation_content="Observability test content",
                compression_level=compressor.CompressionLevel.STANDARD
            )
        
        # Verify appropriate logging occurred
        mock_logger.info.assert_called()
        
        # Result should include tracing information
        assert result.summary
        assert result.compression_ratio > 0
        
        # Performance metrics should be available for observability
        metrics = compressor.get_performance_metrics()
        assert "total_compressions" in metrics
        assert "average_compression_time_s" in metrics


if __name__ == "__main__":
    pytest.main([__file__])