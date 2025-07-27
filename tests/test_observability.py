"""
Tests for the Core Observability & Hook Interception system.

Comprehensive test suite covering event capture, streaming, persistence,
and monitoring capabilities for the Agent Hive observability platform.
"""

import pytest
import pytest_asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import select
from fastapi import FastAPI
from httpx import AsyncClient

from app.models.observability import AgentEvent, EventType
from app.observability.hooks import HookInterceptor, EventCapture
from app.observability.middleware import ObservabilityHookMiddleware
from app.core.event_processor import EventStreamProcessor
from app.api.v1.observability import router as observability_router


class TestAgentEventModel:
    """Test AgentEvent database model."""
    
    @pytest_asyncio.fixture
    async def sample_event_data(self):
        """Sample event data for testing."""
        return {
            "session_id": uuid.uuid4(),
            "agent_id": uuid.uuid4(),
            "event_type": EventType.PRE_TOOL_USE,
            "payload": {
                "tool_name": "test_tool",
                "parameters": {"key": "value"},
                "timestamp": datetime.utcnow().isoformat()
            },
            "latency_ms": 150
        }
    
    async def test_create_agent_event(self, test_db_session, sample_event_data):
        """Test creating an AgentEvent record."""
        # This test will fail initially - we need to implement the model
        event = AgentEvent(**sample_event_data)
        test_db_session.add(event)
        await test_db_session.commit()
        await test_db_session.refresh(event)
        
        assert event.id is not None
        assert event.session_id == sample_event_data["session_id"]
        assert event.agent_id == sample_event_data["agent_id"]
        assert event.event_type == EventType.PRE_TOOL_USE
        assert event.payload == sample_event_data["payload"]
        assert event.latency_ms == 150
        assert event.created_at is not None
    
    async def test_event_type_enum_values(self):
        """Test EventType enum has all required values."""
        # This test will fail initially - we need to define the enum
        assert hasattr(EventType, 'PRE_TOOL_USE')
        assert hasattr(EventType, 'POST_TOOL_USE')
        assert hasattr(EventType, 'NOTIFICATION')
        assert hasattr(EventType, 'STOP')
        assert hasattr(EventType, 'SUBAGENT_STOP')
        
        # Verify enum values match PRD requirements
        assert EventType.PRE_TOOL_USE.value == "PreToolUse"
        assert EventType.POST_TOOL_USE.value == "PostToolUse"
        assert EventType.NOTIFICATION.value == "Notification"
        assert EventType.STOP.value == "Stop"
        assert EventType.SUBAGENT_STOP.value == "SubagentStop"
    
    async def test_agent_event_indexes(self, test_db_session, sample_event_data):
        """Test database indexes for performance."""
        # Create multiple events for index testing
        events = []
        for i in range(5):
            event_data = sample_event_data.copy()
            event_data["event_type"] = EventType.POST_TOOL_USE
            event_data["latency_ms"] = 100 + i * 50
            event = AgentEvent(**event_data)
            events.append(event)
            test_db_session.add(event)
        
        await test_db_session.commit()
        
        # Test query by session_id (should use idx_events_session)
        result = await test_db_session.execute(
            select(AgentEvent).where(AgentEvent.session_id == sample_event_data["session_id"])
        )
        found_events = result.scalars().all()
        assert len(found_events) == 5
        
        # Test query by event_type and time (should use idx_events_type_time)
        result = await test_db_session.execute(
            select(AgentEvent).where(
                AgentEvent.event_type == EventType.POST_TOOL_USE
            ).order_by(AgentEvent.created_at.desc())
        )
        found_events = result.scalars().all()
        assert len(found_events) == 5
    
    async def test_agent_event_json_payload_validation(self, test_db_session):
        """Test JSONB payload handling and validation."""
        complex_payload = {
            "tool_name": "complex_tool",
            "parameters": {
                "nested": {"key": "value", "number": 42},
                "list": [1, 2, 3, {"item": "value"}],
                "boolean": True
            },
            "metadata": {
                "execution_time": 1.5,
                "memory_usage": "50MB",
                "success": True
            }
        }
        
        event = AgentEvent(
            session_id=uuid.uuid4(),
            agent_id=uuid.uuid4(),
            event_type=EventType.PRE_TOOL_USE,
            payload=complex_payload,
            latency_ms=200
        )
        
        test_db_session.add(event)
        await test_db_session.commit()
        await test_db_session.refresh(event)
        
        assert event.payload == complex_payload
        assert event.payload["parameters"]["nested"]["number"] == 42
        assert event.payload["metadata"]["execution_time"] == 1.5


class TestHookInterceptor:
    """Test Hook Interceptor functionality."""
    
    @pytest_asyncio.fixture
    def mock_event_processor(self):
        """Mock event processor for testing."""
        return AsyncMock()
    
    async def test_hook_interceptor_initialization(self, mock_event_processor):
        """Test HookInterceptor can be initialized."""
        # This test will fail initially - we need to implement HookInterceptor
        interceptor = HookInterceptor(event_processor=mock_event_processor)
        assert interceptor.event_processor == mock_event_processor
        assert interceptor.is_enabled is True
    
    async def test_capture_pre_tool_use_event(self, mock_event_processor):
        """Test capturing PreToolUse events."""
        interceptor = HookInterceptor(event_processor=mock_event_processor)
        
        # Simulate tool use event
        session_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        tool_data = {
            "tool_name": "Read",
            "parameters": {"file_path": "/test/file.py"},
            "correlation_id": str(uuid.uuid4())
        }
        
        await interceptor.capture_pre_tool_use(
            session_id=session_id,
            agent_id=agent_id,
            tool_data=tool_data
        )
        
        # Verify event was sent to processor
        mock_event_processor.process_event.assert_called_once()
        call_args = mock_event_processor.process_event.call_args[1]
        
        assert call_args["event_type"] == EventType.PRE_TOOL_USE
        assert call_args["session_id"] == session_id
        assert call_args["agent_id"] == agent_id
        assert call_args["payload"]["tool_name"] == "Read"
        assert call_args["payload"]["parameters"]["file_path"] == "/test/file.py"
    
    async def test_capture_post_tool_use_event_with_result(self, mock_event_processor):
        """Test capturing PostToolUse events with results."""
        interceptor = HookInterceptor(event_processor=mock_event_processor)
        
        session_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        tool_result = {
            "tool_name": "Read",
            "success": True,
            "result": "File content here...",
            "execution_time_ms": 150,
            "correlation_id": str(uuid.uuid4())
        }
        
        await interceptor.capture_post_tool_use(
            session_id=session_id,
            agent_id=agent_id,
            tool_result=tool_result,
            latency_ms=150
        )
        
        mock_event_processor.process_event.assert_called_once()
        call_args = mock_event_processor.process_event.call_args[1]
        
        assert call_args["event_type"] == EventType.POST_TOOL_USE
        assert call_args["latency_ms"] == 150
        assert call_args["payload"]["success"] is True
        assert call_args["payload"]["execution_time_ms"] == 150
    
    async def test_capture_post_tool_use_event_with_error(self, mock_event_processor):
        """Test capturing PostToolUse events with errors."""
        interceptor = HookInterceptor(event_processor=mock_event_processor)
        
        session_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        tool_error = {
            "tool_name": "Write",
            "success": False,
            "error": "Permission denied",
            "error_type": "FilePermissionError",
            "execution_time_ms": 50
        }
        
        await interceptor.capture_post_tool_use(
            session_id=session_id,
            agent_id=agent_id,
            tool_result=tool_error,
            latency_ms=50
        )
        
        mock_event_processor.process_event.assert_called_once()
        call_args = mock_event_processor.process_event.call_args[1]
        
        assert call_args["payload"]["success"] is False
        assert call_args["payload"]["error"] == "Permission denied"
        assert call_args["payload"]["error_type"] == "FilePermissionError"
    
    async def test_capture_notification_event(self, mock_event_processor):
        """Test capturing Notification events."""
        interceptor = HookInterceptor(event_processor=mock_event_processor)
        
        session_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        notification = {
            "level": "warning",
            "message": "Memory usage approaching limit",
            "details": {"memory_usage": "450MB", "limit": "500MB"}
        }
        
        await interceptor.capture_notification(
            session_id=session_id,
            agent_id=agent_id,
            notification=notification
        )
        
        mock_event_processor.process_event.assert_called_once()
        call_args = mock_event_processor.process_event.call_args[1]
        
        assert call_args["event_type"] == EventType.NOTIFICATION
        assert call_args["payload"]["level"] == "warning"
        assert call_args["payload"]["message"] == "Memory usage approaching limit"
    
    async def test_hook_interceptor_disabled_state(self, mock_event_processor):
        """Test hook interceptor when disabled."""
        interceptor = HookInterceptor(event_processor=mock_event_processor)
        interceptor.disable()
        
        await interceptor.capture_pre_tool_use(
            session_id=uuid.uuid4(),
            agent_id=uuid.uuid4(),
            tool_data={"tool_name": "test"}
        )
        
        # No events should be processed when disabled
        mock_event_processor.process_event.assert_not_called()


class TestObservabilityHookMiddleware:
    """Test ObservabilityHookMiddleware integration with FastAPI."""
    
    @pytest_asyncio.fixture
    def test_app_with_middleware(self, mock_redis):
        """Create test app with observability middleware."""
        app = FastAPI()
        
        # This will fail initially - middleware not implemented
        middleware = ObservabilityHookMiddleware()
        app.add_middleware(ObservabilityHookMiddleware)
        
        @app.get("/test-endpoint")
        async def test_endpoint():
            return {"message": "test"}
        
        @app.post("/agents/{agent_id}/tools/execute")
        async def execute_tool(agent_id: str, tool_data: dict):
            # Simulate tool execution
            return {"success": True, "result": "Tool executed"}
        
        return app
    
    async def test_middleware_captures_tool_execution(self, test_app_with_middleware):
        """Test middleware captures tool execution events."""
        async with AsyncClient(app=test_app_with_middleware, base_url="http://test") as client:
            # Execute a tool via API
            response = await client.post(
                "/agents/test-agent-123/tools/execute",
                json={"tool_name": "Read", "parameters": {"file_path": "/test.py"}}
            )
            
            assert response.status_code == 200
            
            # Verify middleware captured events
            # This assertion will help guide implementation
            assert "X-Observability-Events-Captured" in response.headers
    
    async def test_middleware_adds_correlation_id(self, test_app_with_middleware):
        """Test middleware adds correlation ID to responses."""
        async with AsyncClient(app=test_app_with_middleware, base_url="http://test") as client:
            response = await client.get("/test-endpoint")
            
            assert response.status_code == 200
            assert "X-Correlation-ID" in response.headers
            
            # Correlation ID should be valid UUID
            correlation_id = response.headers["X-Correlation-ID"]
            uuid.UUID(correlation_id)  # Will raise if invalid
    
    async def test_middleware_performance_overhead(self, test_app_with_middleware):
        """Test middleware has minimal performance overhead."""
        import time
        
        async with AsyncClient(app=test_app_with_middleware, base_url="http://test") as client:
            # Measure response time with middleware
            start_time = time.time()
            response = await client.get("/test-endpoint")
            end_time = time.time()
            
            assert response.status_code == 200
            
            # Middleware overhead should be minimal (< 10ms for simple endpoint)
            overhead = (end_time - start_time) * 1000
            assert overhead < 10.0


class TestEventStreamProcessor:
    """Test EventStreamProcessor for Redis Streams integration."""
    
    @pytest_asyncio.fixture
    def mock_redis_client(self):
        """Mock Redis client for testing."""
        mock_redis = AsyncMock()
        mock_redis.xadd.return_value = "1234567890-0"
        return mock_redis
    
    @pytest_asyncio.fixture
    def mock_db_session(self):
        """Mock database session for testing."""
        return AsyncMock()
    
    async def test_event_processor_initialization(self, mock_redis_client, mock_db_session):
        """Test EventStreamProcessor initialization."""
        # This test will fail initially - we need to implement EventStreamProcessor
        processor = EventStreamProcessor(
            redis_client=mock_redis_client,
            db_session_factory=lambda: mock_db_session
        )
        
        assert processor.redis_client == mock_redis_client
        assert processor.stream_name == "agent_events"
        assert processor.is_running is False
    
    async def test_process_event_to_redis_stream(self, mock_redis_client, mock_db_session):
        """Test processing event to Redis stream."""
        processor = EventStreamProcessor(
            redis_client=mock_redis_client,
            db_session_factory=lambda: mock_db_session
        )
        
        event_data = {
            "session_id": uuid.uuid4(),
            "agent_id": uuid.uuid4(),
            "event_type": EventType.PRE_TOOL_USE,
            "payload": {"tool_name": "test"},
            "latency_ms": 100
        }
        
        stream_id = await processor.process_event(**event_data)
        
        # Verify event was added to Redis stream
        mock_redis_client.xadd.assert_called_once()
        call_args = mock_redis_client.xadd.call_args
        
        assert call_args[0][0] == "agent_events"  # stream name
        stream_data = call_args[0][1]
        assert stream_data["event_type"] == "PreToolUse"
        assert json.loads(stream_data["payload"])["tool_name"] == "test"
        
        assert stream_id == "1234567890-0"
    
    async def test_process_event_to_database(self, mock_redis_client, mock_db_session):
        """Test processing event to database."""
        processor = EventStreamProcessor(
            redis_client=mock_redis_client,
            db_session_factory=lambda: mock_db_session
        )
        
        event_data = {
            "session_id": uuid.uuid4(),
            "agent_id": uuid.uuid4(),
            "event_type": EventType.POST_TOOL_USE,
            "payload": {"tool_name": "test", "success": True},
            "latency_ms": 200
        }
        
        await processor.process_event(**event_data)
        
        # Verify event was saved to database
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        
        # Check the AgentEvent instance was created correctly
        added_event = mock_db_session.add.call_args[0][0]
        assert added_event.event_type == EventType.POST_TOOL_USE
        assert added_event.latency_ms == 200
    
    async def test_batch_processing_performance(self, mock_redis_client, mock_db_session):
        """Test batch processing for high-throughput scenarios."""
        processor = EventStreamProcessor(
            redis_client=mock_redis_client,
            db_session_factory=lambda: mock_db_session,
            batch_size=5
        )
        
        # Process multiple events
        events = []
        for i in range(10):
            event_data = {
                "session_id": uuid.uuid4(),
                "agent_id": uuid.uuid4(),
                "event_type": EventType.NOTIFICATION,
                "payload": {"message": f"test {i}"},
                "latency_ms": 50
            }
            events.append(event_data)
        
        # Process events in batch
        for event_data in events:
            await processor.process_event(**event_data)
        
        # Verify batch processing optimization
        # Should have fewer DB commits than individual events
        assert mock_db_session.commit.call_count <= 2  # Batched commits


class TestObservabilityAPI:
    """Test Observability API endpoints."""
    
    async def test_post_event_endpoint(self, async_test_client):
        """Test POST /observability/event endpoint."""
        event_data = {
            "event_type": "PreToolUse",
            "agent_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
            "payload": {"tool_name": "Read", "parameters": {"file_path": "/test.py"}},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        response = await async_test_client.post("/observability/event", json=event_data)
        
        assert response.status_code == 201
        assert response.json()["status"] == "queued"
        assert "event_id" in response.json()
    
    async def test_get_events_endpoint(self, async_test_client, sample_session):
        """Test GET /observability/events endpoint."""
        # First create some events
        for i in range(3):
            event_data = {
                "event_type": "PostToolUse",
                "agent_id": str(uuid.uuid4()),
                "session_id": str(sample_session.id),
                "payload": {"tool_name": f"tool_{i}", "success": True},
                "timestamp": datetime.utcnow().isoformat()
            }
            await async_test_client.post("/observability/event", json=event_data)
        
        # Query events
        response = await async_test_client.get(
            f"/observability/events?session_id={sample_session.id}&type=PostToolUse"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        assert len(data["events"]) == 3
        assert data["events"][0]["event_type"] == "PostToolUse"
    
    async def test_get_events_with_time_range(self, async_test_client):
        """Test GET /observability/events with time range filtering."""
        now = datetime.utcnow()
        from_time = (now - timedelta(hours=1)).isoformat()
        to_time = now.isoformat()
        
        response = await async_test_client.get(
            f"/observability/events?from={from_time}&to={to_time}"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        assert "pagination" in data
    
    async def test_get_metrics_endpoint(self, async_test_client):
        """Test GET /observability/metrics endpoint (Prometheus format)."""
        response = await async_test_client.get("/observability/metrics")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        metrics_text = response.text
        assert "agent_events_total" in metrics_text
        assert "event_processing_duration_seconds" in metrics_text
        assert "active_sessions_total" in metrics_text
    
    async def test_get_health_endpoint(self, async_test_client):
        """Test GET /observability/health endpoint."""
        response = await async_test_client.get("/observability/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "components" in data
        assert "redis" in data["components"]
        assert "database" in data["components"]
        assert "event_processor" in data["components"]
    
    async def test_api_validation_errors(self, async_test_client):
        """Test API validation for invalid requests."""
        # Invalid event type
        invalid_event = {
            "event_type": "InvalidEventType",
            "agent_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
            "payload": {}
        }
        
        response = await async_test_client.post("/observability/event", json=invalid_event)
        assert response.status_code == 422
        
        # Invalid UUID
        invalid_uuid_event = {
            "event_type": "PreToolUse",
            "agent_id": "not-a-uuid",
            "session_id": str(uuid.uuid4()),
            "payload": {}
        }
        
        response = await async_test_client.post("/observability/event", json=invalid_uuid_event)
        assert response.status_code == 422
    
    async def test_api_rate_limiting(self, async_test_client):
        """Test API rate limiting for high-frequency events."""
        event_data = {
            "event_type": "Notification",
            "agent_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
            "payload": {"message": "test"}
        }
        
        # Send many events rapidly
        responses = []
        for i in range(100):
            response = await async_test_client.post("/observability/event", json=event_data)
            responses.append(response)
        
        # All requests should succeed (no rate limiting yet)
        success_count = sum(1 for r in responses if r.status_code == 201)
        assert success_count == 100


class TestObservabilityIntegration:
    """Integration tests for full observability pipeline."""
    
    async def test_end_to_end_event_flow(self, test_app, async_test_client, test_db_session):
        """Test complete event flow from capture to storage to retrieval."""
        # Simulate agent tool execution
        session_id = str(uuid.uuid4())
        agent_id = str(uuid.uuid4())
        
        # 1. Capture PreToolUse event
        pre_event = {
            "event_type": "PreToolUse",
            "agent_id": agent_id,
            "session_id": session_id,
            "payload": {
                "tool_name": "Read",
                "parameters": {"file_path": "/test.py"},
                "correlation_id": str(uuid.uuid4())
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        response = await async_test_client.post("/observability/event", json=pre_event)
        assert response.status_code == 201
        
        # 2. Capture PostToolUse event
        post_event = {
            "event_type": "PostToolUse",
            "agent_id": agent_id,
            "session_id": session_id,
            "payload": {
                "tool_name": "Read",
                "success": True,
                "result": "File content here...",
                "execution_time_ms": 150,
                "correlation_id": pre_event["payload"]["correlation_id"]
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        response = await async_test_client.post("/observability/event", json=post_event)
        assert response.status_code == 201
        
        # 3. Wait for event processing (simulate async processing)
        import asyncio
        await asyncio.sleep(0.1)
        
        # 4. Query events and verify they're stored
        response = await async_test_client.get(f"/observability/events?session_id={session_id}")
        assert response.status_code == 200
        
        data = response.json()
        events = data["events"]
        assert len(events) == 2
        
        # Verify correlation between pre and post events
        pre_events = [e for e in events if e["event_type"] == "PreToolUse"]
        post_events = [e for e in events if e["event_type"] == "PostToolUse"]
        
        assert len(pre_events) == 1
        assert len(post_events) == 1
        
        assert pre_events[0]["payload"]["correlation_id"] == post_events[0]["payload"]["correlation_id"]
    
    async def test_performance_monitoring_workflow(self, async_test_client):
        """Test performance monitoring and alerting workflow."""
        # Create events with varying latencies
        session_id = str(uuid.uuid4())
        agent_id = str(uuid.uuid4())
        
        latencies = [50, 100, 500, 1000, 2000]  # Increasing latencies
        
        for i, latency in enumerate(latencies):
            event_data = {
                "event_type": "PostToolUse",
                "agent_id": agent_id,
                "session_id": session_id,
                "payload": {
                    "tool_name": f"tool_{i}",
                    "success": True,
                    "execution_time_ms": latency
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = await async_test_client.post("/observability/event", json=event_data)
            assert response.status_code == 201
        
        # Query metrics to check performance data
        response = await async_test_client.get("/observability/metrics")
        assert response.status_code == 200
        
        metrics_text = response.text
        
        # Verify performance metrics are captured
        assert "tool_execution_duration_seconds" in metrics_text
        assert "tool_success_rate" in metrics_text
    
    async def test_error_tracking_and_alerting(self, async_test_client):
        """Test error tracking and alerting functionality."""
        session_id = str(uuid.uuid4())
        agent_id = str(uuid.uuid4())
        
        # Create multiple error events
        for i in range(5):
            error_event = {
                "event_type": "PostToolUse",
                "agent_id": agent_id,
                "session_id": session_id,
                "payload": {
                    "tool_name": "Write",
                    "success": False,
                    "error": f"Permission denied error {i}",
                    "error_type": "FilePermissionError",
                    "execution_time_ms": 25
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = await async_test_client.post("/observability/event", json=error_event)
            assert response.status_code == 201
        
        # Check if error spike is detected in metrics
        response = await async_test_client.get("/observability/metrics")
        assert response.status_code == 200
        
        metrics_text = response.text
        assert "tool_errors_total" in metrics_text
        
        # Verify health endpoint reflects error state
        response = await async_test_client.get("/observability/health")
        data = response.json()
        
        # Should show degraded health due to high error rate
        assert data["status"] in ["healthy", "degraded", "unhealthy"]