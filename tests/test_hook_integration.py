"""
Comprehensive tests for hook integration and event processing.

Tests the entire hook integration pipeline including:
- Hook event processor with PII redaction
- Real-time streaming capabilities 
- Performance monitoring
- API endpoints
- WebSocket integration
- Security filtering
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from redis.asyncio import Redis

from app.core.hook_processor import HookEventProcessor, PIIRedactor, PerformanceMonitor
from app.main import app
from app.models.observability import EventType


class TestPIIRedactor:
    """Test PII redaction functionality."""
    
    @pytest.fixture
    def redactor(self):
        return PIIRedactor()
    
    def test_redact_sensitive_fields(self, redactor):
        """Test redaction of sensitive field names."""
        data = {
            "password": "secret123",
            "api_key": "abc123def456",
            "email": "user@example.com",
            "normal_field": "safe_value"
        }
        
        result = redactor.redact_data(data)
        
        assert result["password"] == "[REDACTED]"
        assert result["api_key"] == "[REDACTED]"
        assert result["email"] == "[PII_REDACTED]"
        assert result["normal_field"] == "safe_value"
    
    def test_redact_pii_patterns(self, redactor):
        """Test pattern-based PII redaction."""
        text = """
        Contact user@example.com or call 555-123-4567.
        SSN: 123-45-6789
        Credit card: 4532015112830366
        """
        
        result = redactor.redact_data(text)
        
        assert "user@example.com" not in result
        assert "555-123-4567" not in result  
        assert "123-45-6789" not in result
        assert "4532015112830366" not in result
        assert "[EMAIL_REDACTED]" in result
        assert "[PHONE_REDACTED]" in result
        assert "[SSN_REDACTED]" in result
        assert "[CREDIT_CARD_REDACTED]" in result
    
    def test_redact_file_paths(self, redactor):
        """Test file path redaction."""
        data = {
            "error": "File not found: /Users/john/secret/file.txt",
            "stack_trace": "Error in /home/jane/project/app.py:123"
        }
        
        result = redactor.redact_data(data)
        
        assert "/Users/john" not in result["error"]
        assert "/home/jane" not in result["stack_trace"]
        # The actual implementation uses [FILE_PATH_REDACTED] pattern
        assert "FILE_PATH_REDACTED" in result["error"] or "/Users/[USER]" in result["error"]
        assert "FILE_PATH_REDACTED" in result["stack_trace"] or "/home/[USER]" in result["stack_trace"]
    
    def test_redact_nested_structures(self, redactor):
        """Test redaction in nested data structures."""
        data = {
            "config": {
                "database_url": "postgres://user:pass@localhost/db",
                "settings": {
                    "admin_email": "admin@company.com",
                    "debug": True
                }
            },
            "users": [
                {"name": "John", "email": "john@example.com"},
                {"name": "Jane", "phone": "555-987-6543"}
            ]
        }
        
        result = redactor.redact_data(data)
        
        assert result["config"]["database_url"] == "[REDACTED]"
        assert result["config"]["settings"]["admin_email"] == "[PII_REDACTED]"
        assert result["config"]["settings"]["debug"] is True
        assert "[PII_REDACTED]" in str(result["users"])
    
    def test_truncate_large_strings(self, redactor):
        """Test truncation of very large strings."""
        large_string = "x" * 60000
        
        result = redactor.redact_data(large_string)
        
        assert len(result) < 60000
        assert "[TRUNCATED]" in result
    
    def test_limit_list_size(self, redactor):
        """Test list size limiting."""
        large_list = [f"item_{i}" for i in range(150)]
        
        result = redactor.redact_data(large_list)
        
        assert len(result) <= 101  # 100 items + truncation message
        assert any("[TRUNCATED:" in str(item) for item in result)


class TestPerformanceMonitor:
    """Test performance monitoring functionality."""
    
    @pytest.fixture
    def monitor(self):
        return PerformanceMonitor()
    
    def test_record_processing_metrics(self, monitor):
        """Test recording of processing metrics."""
        monitor.record_event_processing(150.5, 25.0, 100.0)
        monitor.record_event_processing(75.2, 15.5, 50.0)
        
        summary = monitor.get_performance_summary()
        
        assert summary["events_processed"] == 2
        assert summary["avg_processing_time_ms"] == pytest.approx(112.85, abs=0.01)
        assert summary["max_processing_time_ms"] == 150.5
        assert summary["redaction_time_ms"] == 40.5
        assert summary["api_time_ms"] == 150.0
    
    def test_record_failures(self, monitor):
        """Test failure recording."""
        monitor.record_event_processing(100.0)
        monitor.record_event_failure()
        monitor.record_event_failure()
        
        summary = monitor.get_performance_summary()
        
        assert summary["events_processed"] == 1
        assert summary["events_failed"] == 2
        assert summary["success_rate"] == pytest.approx(33.33, abs=0.01)
    
    def test_performance_degradation_detection(self, monitor):
        """Test performance degradation detection."""
        # Record slow processing times
        for _ in range(5):
            monitor.record_event_processing(200.0)  # Above 150ms threshold
        
        degradation = monitor.is_performance_degraded()
        
        assert degradation["is_degraded"] is True
        assert len(degradation["issues"]) > 0
        assert "150ms threshold" in degradation["issues"][0]
    
    def test_percentile_calculations(self, monitor):
        """Test percentile calculations."""
        # Record processing times: 50, 100, 150, 200, 250ms
        times = [50, 100, 150, 200, 250]
        for time_ms in times:
            monitor.record_event_processing(time_ms)
        
        summary = monitor.get_performance_summary()
        percentiles = summary["processing_time_percentiles"]
        
        assert percentiles["p50"] == 150  # Median
        assert percentiles["p95"] == 250  # 95th percentile
        assert percentiles["p99"] == 250  # 99th percentile


class TestHookEventProcessor:
    """Test the main hook event processor."""
    
    @pytest.fixture
    def redis_mock(self):
        redis = AsyncMock(spec=Redis)
        redis.ping.return_value = True
        redis.xadd.return_value = "test-stream-id"
        return redis
    
    @pytest.fixture
    def event_processor_mock(self):
        processor = AsyncMock()
        processor.process_event.return_value = "test-event-id"
        processor.health_check.return_value = {"status": "healthy"}
        return processor
    
    @pytest.fixture
    def hook_processor(self, redis_mock, event_processor_mock):
        return HookEventProcessor(
            redis_client=redis_mock,
            event_processor=event_processor_mock,
            enable_pii_redaction=True,
            enable_performance_monitoring=True
        )
    
    @pytest.mark.asyncio
    async def test_process_pre_tool_use(self, hook_processor):
        """Test PreToolUse event processing."""
        event_data = {
            "session_id": str(uuid.uuid4()),
            "agent_id": str(uuid.uuid4()),
            "tool_name": "read",
            "parameters": {
                "file_path": "/Users/test/secret.txt",
                "password": "secret123"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        event_id = await hook_processor.process_pre_tool_use(event_data)
        
        assert event_id == "test-event-id"
        
        # Verify event processor was called with redacted data
        call_args = hook_processor.event_processor.process_event.call_args
        payload = call_args[1]["payload"]
        
        assert payload["parameters"]["password"] == "[REDACTED]"
        assert "/Users/[USER]" in payload["parameters"]["file_path"]
        assert payload["context"]["redacted"] is True
    
    @pytest.mark.asyncio
    async def test_process_post_tool_use(self, hook_processor):
        """Test PostToolUse event processing."""
        event_data = {
            "session_id": str(uuid.uuid4()),
            "agent_id": str(uuid.uuid4()),
            "tool_name": "bash",
            "success": True,
            "result": "Command completed successfully",
            "execution_time_ms": 2500,  # Slow execution
            "timestamp": datetime.utcnow().isoformat()
        }
        
        event_id = await hook_processor.process_post_tool_use(event_data)
        
        assert event_id == "test-event-id"
        
        # Verify performance analysis
        call_args = hook_processor.event_processor.process_event.call_args
        payload = call_args[1]["payload"]
        
        assert payload["performance"]["performance_score"] == "slow"
        assert len(payload["performance"]["warnings"]) > 0
    
    @pytest.mark.asyncio
    async def test_process_error_event(self, hook_processor):
        """Test error event processing."""
        event_data = {
            "session_id": str(uuid.uuid4()),
            "agent_id": str(uuid.uuid4()),
            "error_type": "FileNotFoundError",
            "error_message": "File not found: /Users/test/secret.txt with password secret123",
            "stack_trace": "Traceback...",
            "context": {"user_data": "sensitive_info"},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        event_id = await hook_processor.process_error_event(event_data)
        
        assert event_id == "test-event-id"
        
        # Verify PII redaction
        call_args = hook_processor.event_processor.process_event.call_args
        payload = call_args[1]["payload"]
        
        assert "/Users/[USER]" in payload["error_message"]
        assert "secret123" not in payload["error_message"]
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, hook_processor):
        """Test performance monitoring integration."""
        # Process multiple events to build metrics
        for i in range(10):
            await hook_processor.process_pre_tool_use({
                "session_id": str(uuid.uuid4()),
                "agent_id": str(uuid.uuid4()),
                "tool_name": "test",
                "parameters": {},
                "timestamp": datetime.utcnow().isoformat()
            })
        
        metrics = await hook_processor.get_real_time_metrics()
        
        assert metrics["performance"]["events_processed"] == 10
        assert "health" in metrics
        assert "timestamp" in metrics
    
    @pytest.mark.asyncio
    async def test_real_time_streaming(self, hook_processor, redis_mock):
        """Test real-time event streaming."""
        event_data = {
            "session_id": str(uuid.uuid4()),
            "agent_id": str(uuid.uuid4()),
            "tool_name": "test",
            "parameters": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await hook_processor.process_pre_tool_use(event_data)
        
        # Verify Redis streaming calls
        assert redis_mock.xadd.call_count >= 2  # observability_stream + websocket_stream
        
        # Check stream data structure
        stream_calls = redis_mock.xadd.call_args_list
        observability_call = stream_calls[0]
        
        assert "observability_events" in observability_call[0]
        stream_data = observability_call[0][1]
        assert stream_data["event_type"] == "PRE_TOOL_USE"
    
    @pytest.mark.asyncio
    async def test_health_check(self, hook_processor, redis_mock, event_processor_mock):
        """Test health check functionality."""
        health = await hook_processor.health_check()
        
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert "components" in health
        assert "redis" in health["components"]
        assert "event_processor" in health["components"]
        assert "performance" in health["components"]


class TestHookAPIEndpoints:
    """Test hook-related API endpoints."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def mock_hook_processor(self):
        with patch("app.api.v1.observability.get_hook_event_processor") as mock:
            processor = AsyncMock()
            processor.process_pre_tool_use.return_value = "test-event-id"
            processor.process_post_tool_use.return_value = "test-event-id"
            processor.process_error_event.return_value = "test-event-id"
            processor.process_agent_lifecycle_event.return_value = "test-event-id"
            processor.get_real_time_metrics.return_value = {
                "timestamp": datetime.utcnow().isoformat(),
                "performance": {"events_processed": 100},
                "health": "healthy"
            }
            mock.return_value = processor
            yield processor
    
    def test_hook_events_pre_tool_use(self, client, mock_hook_processor):
        """Test PreToolUse hook event endpoint."""
        event_data = {
            "session_id": str(uuid.uuid4()),
            "agent_id": str(uuid.uuid4()),
            "event_type": "PRE_TOOL_USE",
            "tool_name": "read",
            "parameters": {"file_path": "test.txt"},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        response = client.post("/api/v1/observability/hook-events", json=event_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "processed"
        assert data["event_id"] == "test-event-id"
        assert data["redacted"] is True
    
    def test_hook_events_post_tool_use(self, client, mock_hook_processor):
        """Test PostToolUse hook event endpoint."""
        event_data = {
            "session_id": str(uuid.uuid4()),
            "agent_id": str(uuid.uuid4()),
            "event_type": "POST_TOOL_USE",
            "tool_name": "bash",
            "success": True,
            "result": "Output",
            "execution_time_ms": 100,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        response = client.post("/api/v1/observability/hook-events", json=event_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "processed"
        assert data["processing_time_ms"] is not None
    
    def test_hook_events_error(self, client, mock_hook_processor):
        """Test error hook event endpoint."""
        event_data = {
            "session_id": str(uuid.uuid4()),
            "agent_id": str(uuid.uuid4()),
            "event_type": "ERROR",
            "error_type": "ValueError",
            "error": "Invalid input",
            "stack_trace": "Traceback...",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        response = client.post("/api/v1/observability/hook-events", json=event_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "processed"
    
    def test_hook_events_agent_lifecycle(self, client, mock_hook_processor):
        """Test agent lifecycle event endpoint."""
        event_data = {
            "session_id": str(uuid.uuid4()),
            "agent_id": str(uuid.uuid4()),
            "event_type": "AGENT_START",
            "context": {"agent_name": "test_agent"},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        response = client.post("/api/v1/observability/hook-events", json=event_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "processed"
    
    def test_hook_performance_metrics(self, client, mock_hook_processor):
        """Test hook performance metrics endpoint."""
        response = client.get("/api/v1/observability/hook-performance")
        
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "performance" in data
        assert "health" in data
    
    def test_hook_events_validation_errors(self, client, mock_hook_processor):
        """Test validation errors for hook events."""
        # Invalid UUID
        response = client.post("/api/v1/observability/hook-events", json={
            "session_id": "invalid-uuid",
            "agent_id": str(uuid.uuid4()),
            "event_type": "PRE_TOOL_USE"
        })
        assert response.status_code == 422
        
        # Unsupported event type
        response = client.post("/api/v1/observability/hook-events", json={
            "session_id": str(uuid.uuid4()),
            "agent_id": str(uuid.uuid4()),
            "event_type": "INVALID_TYPE"
        })
        assert response.status_code == 422
    
    def test_performance_warning_threshold(self, client):
        """Test performance warning when processing exceeds threshold."""
        with patch("app.api.v1.observability.get_hook_event_processor") as mock:
            processor = AsyncMock()
            
            # Simulate slow processing
            async def slow_process(*args, **kwargs):
                await asyncio.sleep(0.2)  # 200ms - above 150ms threshold
                return "test-event-id"
            
            processor.process_pre_tool_use.side_effect = slow_process
            mock.return_value = processor
            
            event_data = {
                "session_id": str(uuid.uuid4()),
                "agent_id": str(uuid.uuid4()),
                "event_type": "PRE_TOOL_USE",
                "tool_name": "test"
            }
            
            response = client.post("/api/v1/observability/hook-events", json=event_data)
            
            assert response.status_code == 201
            data = response.json()
            assert data["performance_warnings"] is not None
            assert len(data["performance_warnings"]) > 0


class TestWebSocketIntegration:
    """Test WebSocket integration for real-time events."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, client):
        """Test WebSocket connection for observability events."""
        with client.websocket_connect("/api/v1/websocket/observability") as websocket:
            # Should receive connection acknowledgment
            data = websocket.receive_json()
            assert data["type"] == "connection"
            assert data["status"] == "connected"
    
    @pytest.mark.asyncio 
    async def test_websocket_event_streaming(self, client):
        """Test event streaming over WebSocket."""
        with patch("app.api.v1.websocket.get_redis") as mock_redis:
            redis_mock = AsyncMock()
            redis_mock.xreadgroup.return_value = [
                ("observability_events", [
                    ("1234-0", {
                        b"event_type": b"PRE_TOOL_USE",
                        b"session_id": str(uuid.uuid4()).encode(),
                        b"agent_id": str(uuid.uuid4()).encode(),
                        b"tool_name": b"test",
                        b"timestamp": datetime.utcnow().isoformat().encode()
                    })
                ])
            ]
            mock_redis.return_value = redis_mock
            
            with client.websocket_connect("/api/v1/websocket/observability") as websocket:
                # Should receive connection message
                data = websocket.receive_json()
                assert data["type"] == "connection"


class TestIntegrationEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_complete_hook_workflow(self, client):
        """Test complete workflow from hook event to dashboard."""
        session_id = str(uuid.uuid4())
        agent_id = str(uuid.uuid4())
        
        with patch("app.api.v1.observability.get_hook_event_processor") as mock_processor_getter:
            # Mock the hook processor
            processor = AsyncMock()
            processor.process_pre_tool_use.return_value = "event-1"
            processor.process_post_tool_use.return_value = "event-2"
            processor.get_real_time_metrics.return_value = {
                "timestamp": datetime.utcnow().isoformat(),
                "performance": {
                    "events_processed": 2,
                    "avg_processing_time_ms": 50,
                    "success_rate": 100
                },
                "health": "healthy"
            }
            mock_processor_getter.return_value = processor
            
            # Step 1: Send PreToolUse event
            pre_event = {
                "session_id": session_id,
                "agent_id": agent_id,
                "event_type": "PRE_TOOL_USE",
                "tool_name": "read",
                "parameters": {"file_path": "test.txt"}
            }
            
            response = client.post("/api/v1/observability/hook-events", json=pre_event)
            assert response.status_code == 201
            assert response.json()["event_id"] == "event-1"
            
            # Step 2: Send PostToolUse event
            post_event = {
                "session_id": session_id,
                "agent_id": agent_id,
                "event_type": "POST_TOOL_USE",
                "tool_name": "read",
                "success": True,
                "result": "File contents",
                "execution_time_ms": 50
            }
            
            response = client.post("/api/v1/observability/hook-events", json=post_event)
            assert response.status_code == 201
            assert response.json()["event_id"] == "event-2"
            
            # Step 3: Check performance metrics
            response = client.get("/api/v1/observability/hook-performance")
            assert response.status_code == 200
            
            metrics = response.json()
            assert metrics["performance"]["events_processed"] == 2
            assert metrics["health"] == "healthy"
    
    def test_health_check_with_hook_processor(self, client):
        """Test health check includes hook processor status."""
        with patch("app.api.v1.observability.get_hook_event_processor") as mock:
            processor = AsyncMock()
            processor.health_check.return_value = {
                "status": "healthy",
                "components": {
                    "redis": {"status": "healthy"},
                    "event_processor": {"status": "healthy"},
                    "performance": {"status": "healthy"}
                }
            }
            mock.return_value = processor
            
            response = client.get("/api/v1/observability/health")
            assert response.status_code == 200
            
            health = response.json()
            assert "hook_event_processor" in health["components"]
            assert health["components"]["hook_event_processor"]["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])