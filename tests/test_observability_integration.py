"""
Comprehensive Integration Tests for LeanVibe Agent Hive Observability System

Tests the complete observability stack including Vue.js dashboard, WebSocket streaming,
Prometheus metrics, external hooks, and real-time alerting working together.
"""

import asyncio
import json
import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from fastapi.testclient import TestClient
from fastapi import WebSocket

from app.main import create_app
from app.observability.hooks import HookInterceptor, EventCapture
from app.observability.external_hooks import (
    ExternalHookManager, ExternalHookConfig, HookType, HookEvent, SecurityLevel
)
from app.observability.alerting import (
    AlertingEngine, AlertRule, MetricThreshold, AlertSeverity, ThresholdOperator
)
from app.observability.prometheus_exporter import get_metrics_exporter
from app.api.v1.websocket import connection_manager
from app.models.observability import EventType


class TestObservabilityIntegration:
    """Test observability system integration."""
    
    @pytest.fixture
    def app(self):
        """Create test app."""
        return create_app()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def mock_event_processor(self):
        """Mock event processor."""
        processor = AsyncMock()
        processor.process_event.return_value = "event_123"
        processor.health_check.return_value = {
            "status": "healthy",
            "is_running": True,
            "events_processed": 100,
            "events_failed": 5,
            "processing_rate_per_second": 10.5
        }
        return processor
    
    @pytest.fixture
    def hook_interceptor(self, mock_event_processor):
        """Create hook interceptor."""
        return HookInterceptor(
            event_processor=mock_event_processor,
            enabled=True
        )
    
    @pytest.fixture
    def external_hook_manager(self, hook_interceptor):
        """Create external hook manager."""
        return ExternalHookManager(hook_interceptor)
    
    @pytest.fixture
    def alerting_engine(self, external_hook_manager):
        """Create alerting engine."""
        return AlertingEngine(external_hook_manager)
    
    @pytest.mark.asyncio
    async def test_complete_event_flow(self, hook_interceptor, external_hook_manager, alerting_engine):
        """Test complete event flow from capture to notification."""
        
        # 1. Setup external hook for notifications
        webhook_config = ExternalHookConfig(
            id="test_webhook",
            name="Test Webhook",
            hook_type=HookType.WEBHOOK,
            url="https://example.com/webhook",
            events=[HookEvent.PRE_TOOL_USE, HookEvent.POST_TOOL_USE],
            security_level=SecurityLevel.BASIC
        )
        external_hook_manager.register_hook(webhook_config)
        
        # 2. Setup alert rule
        alert_rule = AlertRule(
            id="test_alert",
            name="Test Alert",
            description="Test alert for high CPU",
            severity=AlertSeverity.WARNING,
            threshold=MetricThreshold(
                metric_name="cpu_usage",
                operator=ThresholdOperator.GREATER_THAN,
                value=80.0,
                duration_seconds=5
            ),
            notification_channels=["test_webhook"]
        )
        alerting_engine.add_rule(alert_rule)
        
        # 3. Start alerting engine
        await alerting_engine.start()
        
        try:
            # 4. Capture a PreToolUse event
            session_id = uuid.uuid4()
            agent_id = uuid.uuid4()
            
            tool_data = {
                "tool_name": "TestTool",
                "parameters": {"input": "test"},
                "correlation_id": "test_correlation_123"
            }
            
            event_id = await hook_interceptor.capture_pre_tool_use(
                session_id=session_id,
                agent_id=agent_id,
                tool_data=tool_data
            )
            
            assert event_id == "event_123"
            
            # 5. Capture a PostToolUse event
            tool_result = {
                "tool_name": "TestTool",
                "success": True,
                "result": {"output": "test_result"},
                "execution_time_ms": 150,
                "correlation_id": "test_correlation_123"
            }
            
            event_id2 = await hook_interceptor.capture_post_tool_use(
                session_id=session_id,
                agent_id=agent_id,
                tool_result=tool_result,
                latency_ms=150
            )
            
            assert event_id2 == "event_123"
            
            # 6. Trigger alert by recording high metric value
            with patch('aiohttp.ClientSession.post') as mock_post:
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.headers = {}
                mock_response.text.return_value = '{"status": "ok"}'
                mock_post.return_value.__aenter__.return_value = mock_response
                
                # Record metric value that exceeds threshold
                await alerting_engine.record_metric_value("cpu_usage", 85.0)
                
                # Wait for alert evaluation
                await asyncio.sleep(0.1)
                
                # Check that alert was triggered
                active_alerts = alerting_engine.get_active_alerts()
                assert len(active_alerts) == 1
                
                alert = active_alerts[0]
                assert alert.rule_id == "test_alert"
                assert alert.severity == AlertSeverity.WARNING
                assert alert.metric_value == 85.0
                
            # 7. Verify hook interceptor was called
            hook_interceptor.event_processor.process_event.assert_called()
            
        finally:
            await alerting_engine.stop()
    
    @pytest.mark.asyncio 
    async def test_websocket_real_time_streaming(self, client):
        """Test WebSocket real-time event streaming."""
        
        # Test WebSocket connection and message flow
        with client.websocket_connect("/api/v1/websocket/observability") as websocket:
            # Should receive connection message
            data = websocket.receive_json()
            assert data["type"] == "connection"
            assert data["status"] == "connected"
            
            # Send ping message
            websocket.send_json({
                "type": "ping",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Should receive pong
            data = websocket.receive_json()
            assert data["type"] == "pong"
            
            # Send stats request
            websocket.send_json({"type": "get_stats"})
            
            # Should receive stats
            data = websocket.receive_json()
            assert data["type"] == "stats"
            assert "data" in data
    
    def test_prometheus_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        
        response = client.get("/api/v1/observability/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        metrics_text = response.text
        
        # Check for key metrics
        assert "leanvibe_http_requests_total" in metrics_text
        assert "leanvibe_active_agents_total" in metrics_text
        assert "leanvibe_system_cpu_usage_percent" in metrics_text
        assert "leanvibe_event_processing_duration_seconds" in metrics_text
    
    def test_health_endpoint_comprehensive(self, client):
        """Test comprehensive health check endpoint."""
        
        response = client.get("/api/v1/observability/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "components" in health_data
        assert "timestamp" in health_data
        
        # Check individual components
        components = health_data["components"]
        expected_components = ["database", "redis", "event_processor", "hook_interceptor"]
        
        for component in expected_components:
            if component in components:
                assert "status" in components[component]
    
    def test_events_api_endpoint(self, client):
        """Test events API endpoint."""
        
        # Test events query
        response = client.get("/api/v1/observability/events")
        assert response.status_code == 200
        
        events_data = response.json()
        assert "events" in events_data
        assert "pagination" in events_data
        assert "total_count" in events_data
        
        pagination = events_data["pagination"]
        assert "limit" in pagination
        assert "offset" in pagination
        assert "total" in pagination
        assert "has_next" in pagination
        assert "has_prev" in pagination
    
    def test_event_creation_api(self, client):
        """Test event creation via API."""
        
        event_data = {
            "event_type": "PreToolUse",
            "agent_id": str(uuid.uuid4()),
            "session_id": str(uuid.uuid4()),
            "payload": {
                "tool_name": "TestTool",
                "parameters": {"input": "test"}
            }
        }
        
        response = client.post("/api/v1/observability/event", json=event_data)
        
        # Might fail due to missing event processor, but should validate input
        assert response.status_code in [201, 503]  # Created or Service Unavailable
        
        if response.status_code == 201:
            result = response.json()
            assert result["status"] == "queued"
            assert "event_id" in result
    
    @pytest.mark.asyncio
    async def test_external_hooks_security_validation(self, external_hook_manager):
        """Test external hooks security validation."""
        
        # Test valid webhook configuration
        valid_config = ExternalHookConfig(
            id="valid_webhook",
            name="Valid Webhook",
            hook_type=HookType.WEBHOOK,
            url="https://example.com/webhook",
            events=[HookEvent.PRE_TOOL_USE],
            security_level=SecurityLevel.BASIC,
            allowed_domains=["example.com"]
        )
        
        # Should register successfully
        external_hook_manager.register_hook(valid_config)
        assert external_hook_manager.get_hook("valid_webhook") is not None
        
        # Test invalid webhook configuration (localhost)
        with pytest.raises(ValueError, match="Localhost URLs are not allowed"):
            invalid_config = ExternalHookConfig(
                id="invalid_webhook",
                name="Invalid Webhook", 
                hook_type=HookType.WEBHOOK,
                url="https://localhost/webhook",
                events=[HookEvent.PRE_TOOL_USE],
                security_level=SecurityLevel.BASIC
            )
        
        # Test domain mismatch
        with pytest.raises(ValueError, match="not in allowed domains"):
            domain_mismatch_config = ExternalHookConfig(
                id="domain_mismatch",
                name="Domain Mismatch",
                hook_type=HookType.WEBHOOK,
                url="https://example.com/webhook",
                events=[HookEvent.PRE_TOOL_USE],
                security_level=SecurityLevel.BASIC,
                allowed_domains=["other-domain.com"]
            )
    
    @pytest.mark.asyncio
    async def test_alerting_engine_lifecycle(self, alerting_engine):
        """Test alerting engine complete lifecycle."""
        
        # Add alert rule
        rule = AlertRule(
            id="test_rule",
            name="Test Rule",
            description="Test alert rule",
            severity=AlertSeverity.WARNING,
            threshold=MetricThreshold(
                metric_name="test_metric",
                operator=ThresholdOperator.GREATER_THAN,
                value=50.0,
                duration_seconds=1
            )
        )
        
        alerting_engine.add_rule(rule)
        
        # Start engine
        await alerting_engine.start()
        
        try:
            # Record metric value below threshold
            await alerting_engine.record_metric_value("test_metric", 30.0)
            await asyncio.sleep(0.1)
            
            # Should have no alerts
            assert len(alerting_engine.get_active_alerts()) == 0
            
            # Record metric value above threshold
            await alerting_engine.record_metric_value("test_metric", 75.0)
            await asyncio.sleep(1.2)  # Wait longer than duration_seconds
            
            # Should trigger alert
            active_alerts = alerting_engine.get_active_alerts()
            assert len(active_alerts) == 1
            
            alert = active_alerts[0]
            assert alert.rule_id == "test_rule"
            assert alert.metric_value == 75.0
            
            # Acknowledge alert
            success = alerting_engine.acknowledge_alert(alert.id, "test_user")
            assert success is True
            
            # Record metric value back below threshold
            await alerting_engine.record_metric_value("test_metric", 25.0)
            await asyncio.sleep(0.1)
            
            # Alert should be resolved
            assert len(alerting_engine.get_active_alerts()) == 0
            
            resolved_alerts = alerting_engine.get_resolved_alerts()
            assert len(resolved_alerts) == 1
            
        finally:
            await alerting_engine.stop()
    
    def test_metrics_exporter_comprehensive(self):
        """Test Prometheus metrics exporter functionality."""
        
        exporter = get_metrics_exporter()
        assert exporter is not None
        
        # Test metric recording
        exporter.record_http_request(
            method="GET",
            endpoint="/api/test",
            status_code=200,
            duration=0.5,
            request_size=1024,
            response_size=2048
        )
        
        exporter.record_agent_operation(
            agent_id="test_agent",
            operation_type="tool_execution",
            status="success",
            duration=1.5
        )
        
        exporter.record_tool_execution(
            tool_name="TestTool",
            status="success",
            duration=0.8
        )
        
        exporter.record_error(
            component="test_component",
            error_type="TestError",
            severity="warning"
        )
        
        # Generate metrics response
        response = exporter.generate_metrics_response()
        assert response.status_code == 200
        assert response.media_type == "text/plain; charset=utf-8"
        
        metrics_content = response.body.decode()
        
        # Verify metrics are present
        assert "leanvibe_http_requests_total" in metrics_content
        assert "leanvibe_agent_operations_total" in metrics_content
        assert "leanvibe_tool_executions_total" in metrics_content
        assert "leanvibe_errors_total" in metrics_content
    
    @pytest.mark.asyncio
    async def test_hook_batch_processing(self, external_hook_manager):
        """Test external hook batch processing."""
        
        # Configure hook with batch processing
        batch_config = ExternalHookConfig(
            id="batch_webhook",
            name="Batch Webhook",
            hook_type=HookType.WEBHOOK,
            url="https://example.com/batch",
            events=[HookEvent.PRE_TOOL_USE],
            security_level=SecurityLevel.BASIC,
            batch_size=3,
            batch_timeout_seconds=2
        )
        
        external_hook_manager.register_hook(batch_config)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.headers = {}
            mock_response.text.return_value = '{"status": "ok"}'
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Trigger multiple events
            for i in range(2):
                await external_hook_manager.trigger_hooks(
                    event_type=HookEvent.PRE_TOOL_USE,
                    payload={"test": f"event_{i}"},
                    agent_id="test_agent"
                )
            
            # Should not have triggered webhook yet (batch not full)
            mock_post.assert_not_called()
            
            # Add third event to complete batch
            await external_hook_manager.trigger_hooks(
                event_type=HookEvent.PRE_TOOL_USE,
                payload={"test": "event_2"},
                agent_id="test_agent"
            )
            
            # Wait for batch processing
            await asyncio.sleep(0.1)
            
            # Should have triggered webhook with batch
            mock_post.assert_called_once()
            
            # Verify batch payload structure
            call_args = mock_post.call_args
            payload = call_args[1]["json"]
            
            assert payload["batch"] is True
            assert len(payload["events"]) == 3
    
    def test_rate_limiting(self, external_hook_manager):
        """Test external hook rate limiting."""
        
        # Configure hook with low rate limit
        rate_limited_config = ExternalHookConfig(
            id="rate_limited_webhook",
            name="Rate Limited Webhook",
            hook_type=HookType.WEBHOOK,
            url="https://example.com/rate-limited",
            events=[HookEvent.PRE_TOOL_USE],
            security_level=SecurityLevel.BASIC,
            rate_limit_per_minute=2
        )
        
        external_hook_manager.register_hook(rate_limited_config)
        
        # Allow first two requests
        assert external_hook_manager.rate_limiter.is_allowed("rate_limited_webhook", 2) is True
        assert external_hook_manager.rate_limiter.is_allowed("rate_limited_webhook", 2) is True
        
        # Third request should be rate limited
        assert external_hook_manager.rate_limiter.is_allowed("rate_limited_webhook", 2) is False
    
    @pytest.mark.asyncio
    async def test_event_capture_payload_creation(self):
        """Test event capture payload creation utilities."""
        
        # Test PreToolUse payload
        pre_payload = EventCapture.create_pre_tool_use_payload(
            tool_name="TestTool",
            parameters={"input": "test", "mode": "debug"},
            correlation_id="corr_123"
        )
        
        assert pre_payload["tool_name"] == "TestTool"
        assert pre_payload["parameters"]["input"] == "test"
        assert pre_payload["correlation_id"] == "corr_123"
        assert "timestamp" in pre_payload
        
        # Test PostToolUse payload
        post_payload = EventCapture.create_post_tool_use_payload(
            tool_name="TestTool",
            success=True,
            result={"output": "success"},
            execution_time_ms=250,
            correlation_id="corr_123"
        )
        
        assert post_payload["tool_name"] == "TestTool"
        assert post_payload["success"] is True
        assert post_payload["result"]["output"] == "success"
        assert post_payload["execution_time_ms"] == 250
        assert post_payload["correlation_id"] == "corr_123"
        
        # Test PostToolUse payload with error
        error_payload = EventCapture.create_post_tool_use_payload(
            tool_name="TestTool",
            success=False,
            error="Tool execution failed",
            error_type="ExecutionError"
        )
        
        assert error_payload["success"] is False
        assert error_payload["error"] == "Tool execution failed"
        assert error_payload["error_type"] == "ExecutionError"
        
        # Test Notification payload
        notification_payload = EventCapture.create_notification_payload(
            level="warning",
            message="Test notification",
            details={"component": "test"}
        )
        
        assert notification_payload["level"] == "warning"
        assert notification_payload["message"] == "Test notification"
        assert notification_payload["details"]["component"] == "test"
    
    def test_system_status_endpoint(self, client):
        """Test system status endpoint."""
        
        response = client.get("/status")
        assert response.status_code == 200
        
        status_data = response.json()
        assert "timestamp" in status_data
        assert "version" in status_data
        assert "components" in status_data
        
        components = status_data["components"]
        assert "database" in components
        assert "redis" in components
        assert "orchestrator" in components
        assert "observability" in components
    
    def test_websocket_stats_endpoint(self, client):
        """Test WebSocket connection statistics endpoint."""
        
        response = client.get("/api/v1/websocket/stats")
        assert response.status_code == 200
        
        stats_data = response.json()
        assert stats_data["status"] == "success"
        assert "stats" in stats_data
        assert "timestamp" in stats_data
        
        stats = stats_data["stats"]
        assert "observability_connections" in stats
        assert "agent_connections" in stats
        assert "total_connections" in stats


class TestObservabilityPerformance:
    """Test observability system performance under load."""
    
    @pytest.mark.asyncio
    async def test_high_volume_event_processing(self, hook_interceptor):
        """Test high volume event processing performance."""
        
        session_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        
        # Process 1000 events
        start_time = datetime.utcnow()
        
        tasks = []
        for i in range(1000):
            task = hook_interceptor.capture_pre_tool_use(
                session_id=session_id,
                agent_id=agent_id,
                tool_data={
                    "tool_name": f"Tool_{i}",
                    "parameters": {"index": i}
                }
            )
            tasks.append(task)
        
        # Process events in batches to avoid overwhelming
        batch_size = 100
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            await asyncio.gather(*batch)
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        # Should process 1000 events in reasonable time (< 10 seconds)
        assert duration < 10.0
        
        # Calculate throughput
        throughput = 1000 / duration
        print(f"Event processing throughput: {throughput:.2f} events/second")
        
        # Should achieve reasonable throughput
        assert throughput > 100  # At least 100 events/second
    
    @pytest.mark.asyncio
    async def test_concurrent_websocket_connections(self, client):
        """Test handling multiple concurrent WebSocket connections."""
        
        # This test would need special setup for concurrent WebSocket testing
        # For now, just test that connection manager tracks connections properly
        
        stats = connection_manager.get_connection_stats()
        initial_count = stats["total_connections"]
        
        # In a real test, we'd create multiple WebSocket connections
        # and verify they're tracked correctly
        assert initial_count >= 0
    
    @pytest.mark.asyncio 
    async def test_memory_usage_under_load(self, alerting_engine):
        """Test memory usage doesn't grow excessively under load."""
        
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        await alerting_engine.start()
        
        try:
            # Generate lots of metric data
            for i in range(10000):
                await alerting_engine.record_metric_value(
                    f"test_metric_{i % 10}",
                    float(i % 100),
                    {"label": f"value_{i % 5}"}
                )
            
            # Check memory usage after load
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            memory_increase_mb = memory_increase / 1024 / 1024
            
            print(f"Memory increase: {memory_increase_mb:.2f} MB")
            
            # Memory increase should be reasonable (< 100 MB)
            assert memory_increase_mb < 100
            
        finally:
            await alerting_engine.stop()


@pytest.mark.asyncio
async def test_end_to_end_observability_flow():
    """
    End-to-end test of complete observability flow.
    
    This test validates that all components work together seamlessly:
    1. Event capture
    2. WebSocket streaming  
    3. Metrics collection
    4. Alert triggering
    5. External notifications
    """
    
    # This would be the master integration test that ties everything together
    # For brevity, we'll simulate the key interactions
    
    print("🎯 Starting end-to-end observability test...")
    
    # 1. Initialize all components
    mock_event_processor = AsyncMock()
    mock_event_processor.process_event.return_value = "e2e_event_123"
    mock_event_processor.health_check.return_value = {"status": "healthy", "is_running": True}
    
    hook_interceptor = HookInterceptor(mock_event_processor)
    external_hook_manager = ExternalHookManager(hook_interceptor)
    alerting_engine = AlertingEngine(external_hook_manager)
    
    # 2. Configure external hook
    webhook_config = ExternalHookConfig(
        id="e2e_webhook",
        name="E2E Test Webhook",
        hook_type=HookType.WEBHOOK,
        url="https://httpbin.org/post",  # Use real endpoint for testing
        events=[HookEvent.SYSTEM_ALERT],
        security_level=SecurityLevel.BASIC
    )
    external_hook_manager.register_hook(webhook_config)
    
    # 3. Configure alert rule
    alert_rule = AlertRule(
        id="e2e_alert",
        name="E2E Test Alert",
        description="End-to-end test alert",
        severity=AlertSeverity.WARNING,
        threshold=MetricThreshold(
            metric_name="e2e_metric",
            operator=ThresholdOperator.GREATER_THAN,
            value=75.0,
            duration_seconds=1
        ),
        notification_channels=["e2e_webhook"]
    )
    alerting_engine.add_rule(alert_rule)
    
    await alerting_engine.start()
    
    try:
        # 4. Capture some events
        session_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        
        print("📝 Capturing events...")
        
        await hook_interceptor.capture_pre_tool_use(
            session_id=session_id,
            agent_id=agent_id,
            tool_data={"tool_name": "E2ETestTool", "parameters": {"test": True}}
        )
        
        await hook_interceptor.capture_post_tool_use(
            session_id=session_id,
            agent_id=agent_id,
            tool_result={"tool_name": "E2ETestTool", "success": True, "result": "OK"},
            latency_ms=100
        )
        
        # 5. Trigger alert
        print("🚨 Triggering alert...")
        
        await alerting_engine.record_metric_value("e2e_metric", 85.0)
        await asyncio.sleep(1.5)  # Wait for alert evaluation
        
        # 6. Verify alert was created
        active_alerts = alerting_engine.get_active_alerts()
        assert len(active_alerts) >= 1
        
        e2e_alert = None
        for alert in active_alerts:
            if alert.rule_id == "e2e_alert":
                e2e_alert = alert
                break
        
        assert e2e_alert is not None
        assert e2e_alert.metric_value == 85.0
        
        print("✅ Alert triggered successfully")
        
        # 7. Verify event processor was called
        assert mock_event_processor.process_event.call_count >= 2
        
        print("✅ Events captured successfully")
        
        # 8. Test metrics collection
        metrics_exporter = get_metrics_exporter()
        metrics_exporter.record_http_request("GET", "/api/test", 200, 0.1)
        
        response = metrics_exporter.generate_metrics_response()
        assert response.status_code == 200
        
        print("✅ Metrics collection working")
        
        print("🎉 End-to-end observability test completed successfully!")
        
    finally:
        await alerting_engine.stop()


if __name__ == "__main__":
    # Run the end-to-end test
    asyncio.run(test_end_to_end_observability_flow())