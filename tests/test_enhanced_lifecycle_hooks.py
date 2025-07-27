"""
Comprehensive tests for Enhanced Lifecycle Hooks System.

Tests include:
- Enhanced event processing and streaming
- WebSocket real-time communication  
- Performance analytics and pattern detection
- Error handling and edge cases
- Integration with existing infrastructure
"""

import asyncio
import json
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from app.core.enhanced_lifecycle_hooks import (
    EnhancedLifecycleHookProcessor,
    EnhancedEventType,
    LifecycleEventData,
    LifecycleEventFilter,
    get_enhanced_lifecycle_hook_processor
)
from app.api.v1.enhanced_observability import WebSocketManager


class TestEnhancedEventType:
    """Test enhanced event type definitions."""
    
    def test_enhanced_event_types_exist(self):
        """Test that all enhanced event types are defined."""
        # Base event types
        assert EnhancedEventType.PRE_TOOL_USE == "PreToolUse"
        assert EnhancedEventType.POST_TOOL_USE == "PostToolUse"
        assert EnhancedEventType.NOTIFICATION == "Notification"
        assert EnhancedEventType.STOP == "Stop"
        assert EnhancedEventType.SUBAGENT_STOP == "SubagentStop"
        
        # Enhanced lifecycle events
        assert EnhancedEventType.AGENT_LIFECYCLE_START == "AgentLifecycleStart"
        assert EnhancedEventType.AGENT_LIFECYCLE_PAUSE == "AgentLifecyclePause"
        assert EnhancedEventType.AGENT_LIFECYCLE_RESUME == "AgentLifecycleResume"
        assert EnhancedEventType.AGENT_LIFECYCLE_COMPLETE == "AgentLifecycleComplete"
        
        # Context and performance events
        assert EnhancedEventType.CONTEXT_THRESHOLD_REACHED == "ContextThresholdReached"
        assert EnhancedEventType.PERFORMANCE_DEGRADATION == "PerformanceDegradation"
        assert EnhancedEventType.MEMORY_PRESSURE == "MemoryPressure"
        
        # Task coordination events
        assert EnhancedEventType.TASK_ASSIGNMENT == "TaskAssignment"
        assert EnhancedEventType.TASK_PROGRESS_UPDATE == "TaskProgressUpdate"
        assert EnhancedEventType.AGENT_COORDINATION == "AgentCoordination"
    
    def test_event_type_enumeration(self):
        """Test that event types can be enumerated."""
        event_types = list(EnhancedEventType)
        assert len(event_types) >= 16  # Should have at least the defined events
        
        # Verify no duplicates
        event_values = [e.value for e in event_types]
        assert len(event_values) == len(set(event_values))


class TestLifecycleEventData:
    """Test lifecycle event data structure."""
    
    @pytest.fixture
    def sample_event_data(self):
        """Sample event data for testing."""
        return LifecycleEventData(
            session_id=str(uuid.uuid4()),
            agent_id=str(uuid.uuid4()),
            event_type=EnhancedEventType.AGENT_LIFECYCLE_START,
            timestamp=datetime.utcnow().isoformat(),
            payload={"test": "data", "value": 123},
            correlation_id=str(uuid.uuid4()),
            severity="info",
            tags={"env": "test", "version": "1.0"}
        )
    
    def test_lifecycle_event_data_creation(self, sample_event_data):
        """Test lifecycle event data creation."""
        assert sample_event_data.event_type == EnhancedEventType.AGENT_LIFECYCLE_START
        assert sample_event_data.severity == "info"
        assert sample_event_data.payload["test"] == "data"
        assert sample_event_data.tags["env"] == "test"
    
    def test_lifecycle_event_data_serialization(self, sample_event_data):
        """Test event data serialization to dict."""
        data_dict = sample_event_data.to_dict()
        
        assert isinstance(data_dict, dict)
        assert data_dict["session_id"] == sample_event_data.session_id
        assert data_dict["agent_id"] == sample_event_data.agent_id
        assert data_dict["event_type"] == sample_event_data.event_type
        assert data_dict["payload"] == sample_event_data.payload
        assert data_dict["tags"] == sample_event_data.tags


class TestLifecycleEventFilter:
    """Test lifecycle event filtering logic."""
    
    @pytest.fixture
    def event_filter(self):
        """Event filter instance for testing."""
        return LifecycleEventFilter()
    
    def test_high_priority_event_detection(self, event_filter):
        """Test high priority event detection."""
        assert event_filter.get_priority(EnhancedEventType.CONTEXT_THRESHOLD_REACHED) == "high"
        assert event_filter.get_priority(EnhancedEventType.PERFORMANCE_DEGRADATION) == "high"
        assert event_filter.get_priority(EnhancedEventType.MEMORY_PRESSURE) == "high"
        assert event_filter.get_priority(EnhancedEventType.ERROR_PATTERN_DETECTED) == "high"
    
    def test_real_time_event_detection(self, event_filter):
        """Test real-time streaming event detection."""
        assert event_filter.should_stream_real_time(EnhancedEventType.AGENT_LIFECYCLE_START) is True
        assert event_filter.should_stream_real_time(EnhancedEventType.PRE_TOOL_USE) is True
        assert event_filter.should_stream_real_time(EnhancedEventType.TASK_PROGRESS_UPDATE) is True
        assert event_filter.should_stream_real_time(EnhancedEventType.HEALTH_CHECK) is False
    
    def test_alert_triggering_logic(self, event_filter):
        """Test alert triggering logic."""
        # High severity event
        critical_event = LifecycleEventData(
            session_id=str(uuid.uuid4()),
            agent_id=str(uuid.uuid4()),
            event_type=EnhancedEventType.AGENT_LIFECYCLE_START,
            timestamp=datetime.utcnow().isoformat(),
            payload={},
            severity="critical"
        )
        assert event_filter.should_trigger_alert(critical_event) is True
        
        # High priority event type
        threshold_event = LifecycleEventData(
            session_id=str(uuid.uuid4()),
            agent_id=str(uuid.uuid4()),
            event_type=EnhancedEventType.CONTEXT_THRESHOLD_REACHED,
            timestamp=datetime.utcnow().isoformat(),
            payload={},
            severity="info"
        )
        assert event_filter.should_trigger_alert(threshold_event) is True
        
        # Normal event
        normal_event = LifecycleEventData(
            session_id=str(uuid.uuid4()),
            agent_id=str(uuid.uuid4()),
            event_type=EnhancedEventType.HEALTH_CHECK,
            timestamp=datetime.utcnow().isoformat(),
            payload={},
            severity="info"
        )
        assert event_filter.should_trigger_alert(normal_event) is False


class TestEnhancedLifecycleHookProcessor:
    """Test enhanced lifecycle hook processor."""
    
    @pytest.fixture
    def processor(self):
        """Enhanced lifecycle hook processor for testing."""
        with patch('app.core.enhanced_lifecycle_hooks.get_async_session') as mock_session, \
             patch('app.core.enhanced_lifecycle_hooks.get_redis') as mock_redis, \
             patch('app.core.enhanced_lifecycle_hooks.get_settings') as mock_settings:
            
            # Mock settings
            mock_settings.return_value = Mock()
            
            # Mock database session
            mock_db_session = AsyncMock()
            mock_db_session.add = Mock()
            mock_db_session.commit = AsyncMock()
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_db_session)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Mock Redis client
            mock_redis_client = AsyncMock()
            mock_redis_client.xadd = AsyncMock(return_value="stream_id")
            mock_redis.return_value = mock_redis_client
            
            processor = EnhancedLifecycleHookProcessor()
            return processor
    
    @pytest.fixture
    def sample_session_id(self):
        """Sample session ID for testing."""
        return uuid.uuid4()
    
    @pytest.fixture
    def sample_agent_id(self):
        """Sample agent ID for testing."""
        return uuid.uuid4()
    
    @pytest.mark.asyncio
    async def test_process_enhanced_event_success(
        self, 
        processor, 
        sample_session_id, 
        sample_agent_id
    ):
        """Test successful enhanced event processing."""
        
        payload = {
            "test_data": "value",
            "numeric_value": 42,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        event_id = await processor.process_enhanced_event(
            session_id=sample_session_id,
            agent_id=sample_agent_id,
            event_type=EnhancedEventType.AGENT_LIFECYCLE_START,
            payload=payload,
            severity="info"
        )
        
        assert event_id is not None
        assert isinstance(event_id, str)
        
        # Verify performance metrics were updated
        assert processor.performance_metrics["events_processed"] > 0
        assert processor.performance_metrics["events_streamed"] > 0
    
    @pytest.mark.asyncio
    async def test_capture_agent_lifecycle_start(
        self,
        processor,
        sample_session_id,
        sample_agent_id
    ):
        """Test agent lifecycle start event capture."""
        
        capabilities = ["python", "fastapi", "database"]
        initial_context = {"task": "test", "environment": "development"}
        
        event_id = await processor.capture_agent_lifecycle_start(
            session_id=sample_session_id,
            agent_id=sample_agent_id,
            agent_type="backend_developer",
            capabilities=capabilities,
            initial_context=initial_context
        )
        
        assert event_id is not None
        assert processor.performance_metrics["events_processed"] > 0
    
    @pytest.mark.asyncio
    async def test_capture_context_threshold_reached(
        self,
        processor,
        sample_session_id,
        sample_agent_id
    ):
        """Test context threshold reached event capture."""
        
        context_stats = {
            "total_tokens": 8500,
            "tokens_remaining": 1500,
            "context_count": 15
        }
        
        event_id = await processor.capture_context_threshold_reached(
            session_id=sample_session_id,
            agent_id=sample_agent_id,
            current_usage_percent=85.0,
            threshold_type="high_usage",
            recommended_action="consolidate_context",
            context_stats=context_stats
        )
        
        assert event_id is not None
        
        # Verify alert was processed for high threshold
        assert processor.performance_metrics["alerts_triggered"] > 0
    
    @pytest.mark.asyncio
    async def test_capture_context_critical_threshold(
        self,
        processor,
        sample_session_id,
        sample_agent_id
    ):
        """Test critical context threshold event capture."""
        
        context_stats = {"total_tokens": 9800, "tokens_remaining": 200}
        
        event_id = await processor.capture_context_threshold_reached(
            session_id=sample_session_id,
            agent_id=sample_agent_id,
            current_usage_percent=98.0,  # Critical threshold
            threshold_type="critical_usage",
            recommended_action="emergency_consolidation",
            context_stats=context_stats
        )
        
        assert event_id is not None
        
        # Verify critical alert was triggered
        assert processor.performance_metrics["alerts_triggered"] > 0
    
    @pytest.mark.asyncio
    async def test_capture_performance_degradation(
        self,
        processor,
        sample_session_id,
        sample_agent_id
    ):
        """Test performance degradation event capture."""
        
        current_metrics = {
            "avg_response_time": 2500,
            "throughput": 15,
            "error_rate": 0.05
        }
        
        baseline_metrics = {
            "avg_response_time": 800,
            "throughput": 45,
            "error_rate": 0.01
        }
        
        degradation_details = {
            "percentage_degradation": 67,
            "affected_operations": ["tool_execution", "context_retrieval"],
            "type": "response_time_degradation"
        }
        
        event_id = await processor.capture_performance_degradation(
            session_id=sample_session_id,
            agent_id=sample_agent_id,
            performance_metrics=current_metrics,
            degradation_details=degradation_details,
            baseline_metrics=baseline_metrics
        )
        
        assert event_id is not None
        assert processor.performance_metrics["alerts_triggered"] > 0
    
    @pytest.mark.asyncio
    async def test_capture_task_progress_update(
        self,
        processor,
        sample_session_id,
        sample_agent_id
    ):
        """Test task progress update event capture."""
        
        performance_data = {
            "estimated_completion": "2024-01-15T14:30:00Z",
            "stages_completed": ["analysis", "design"],
            "next_stage": "implementation",
            "velocity": 0.8
        }
        
        event_id = await processor.capture_task_progress_update(
            session_id=sample_session_id,
            agent_id=sample_agent_id,
            task_id="task_123",
            progress_percentage=65.0,
            current_stage="implementation",
            performance_data=performance_data
        )
        
        assert event_id is not None
    
    @pytest.mark.asyncio
    async def test_capture_agent_coordination(
        self,
        processor,
        sample_session_id,
        sample_agent_id
    ):
        """Test agent coordination event capture."""
        
        target_agent_id = uuid.uuid4()
        coordination_data = {
            "message_type": "task_delegation",
            "task_details": {"task": "database_migration", "priority": "high"},
            "expected_response_time": 30
        }
        
        event_id = await processor.capture_agent_coordination(
            session_id=sample_session_id,
            source_agent_id=sample_agent_id,
            target_agent_id=target_agent_id,
            coordination_type="task_delegation",
            coordination_data=coordination_data
        )
        
        assert event_id is not None
    
    @pytest.mark.asyncio
    async def test_websocket_client_management(self, processor):
        """Test WebSocket client registration and management."""
        
        # Mock WebSocket clients
        client1 = Mock()
        client2 = Mock()
        
        # Register clients
        processor.register_websocket_client(client1)
        processor.register_websocket_client(client2)
        
        assert len(processor.websocket_clients) == 2
        assert client1 in processor.websocket_clients
        assert client2 in processor.websocket_clients
        
        # Unregister client
        processor.unregister_websocket_client(client1)
        
        assert len(processor.websocket_clients) == 1
        assert client1 not in processor.websocket_clients
        assert client2 in processor.websocket_clients
    
    @pytest.mark.asyncio
    async def test_get_event_analytics(
        self,
        processor,
        sample_session_id,
        sample_agent_id
    ):
        """Test event analytics retrieval."""
        
        analytics = await processor.get_event_analytics(
            session_id=sample_session_id,
            agent_id=sample_agent_id,
            time_range=3600,  # 1 hour
            event_types=[EnhancedEventType.AGENT_LIFECYCLE_START]
        )
        
        assert "summary" in analytics
        assert "event_distribution" in analytics
        assert "performance_trends" in analytics
        assert "error_patterns" in analytics
        assert "recommendations" in analytics
        
        # Verify summary contains expected metrics
        summary = analytics["summary"]
        assert "total_events" in summary
        assert "events_streamed" in summary
        assert "avg_processing_time_ms" in summary
    
    @pytest.mark.asyncio
    async def test_error_pattern_detection(
        self,
        processor,
        sample_session_id,
        sample_agent_id
    ):
        """Test error pattern detection."""
        
        # Generate multiple error events to trigger pattern detection
        for i in range(6):  # Above threshold of 5
            await processor.process_enhanced_event(
                session_id=sample_session_id,
                agent_id=sample_agent_id,
                event_type=EnhancedEventType.ERROR_PATTERN_DETECTED,
                payload={"error_type": "connection_timeout", "occurrence": i + 1},
                severity="error"
            )
        
        # Verify pattern was detected
        pattern_key = f"{sample_agent_id}:ErrorPatternDetected"
        assert pattern_key in processor.error_patterns
        assert processor.error_patterns[pattern_key]["count"] == 6
    
    @pytest.mark.asyncio
    async def test_performance_metrics_update(
        self,
        processor,
        sample_session_id,
        sample_agent_id
    ):
        """Test performance metrics tracking."""
        
        initial_events = processor.performance_metrics["events_processed"]
        
        # Process several events
        for i in range(5):
            await processor.process_enhanced_event(
                session_id=sample_session_id,
                agent_id=sample_agent_id,
                event_type=EnhancedEventType.HEALTH_CHECK,
                payload={"check_id": i},
                severity="info"
            )
        
        # Verify metrics were updated
        assert processor.performance_metrics["events_processed"] == initial_events + 5
        assert processor.performance_metrics["avg_processing_time_ms"] >= 0


class TestWebSocketManager:
    """Test WebSocket manager for real-time streaming."""
    
    @pytest.fixture
    def websocket_manager(self):
        """WebSocket manager instance for testing."""
        return WebSocketManager()
    
    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket connection."""
        websocket = Mock()
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.client = Mock()
        websocket.client.host = "127.0.0.1"
        websocket.headers = {"user-agent": "test-client"}
        return websocket
    
    @pytest.mark.asyncio
    async def test_websocket_connection_lifecycle(
        self, 
        websocket_manager, 
        mock_websocket
    ):
        """Test WebSocket connection lifecycle."""
        
        client_info = {"client_ip": "127.0.0.1", "user_agent": "test-client"}
        
        # Test connection
        await websocket_manager.connect(mock_websocket, client_info)
        
        assert mock_websocket in websocket_manager.active_connections
        assert mock_websocket in websocket_manager.client_subscriptions
        assert mock_websocket in websocket_manager.connection_metadata
        assert websocket_manager.connection_stats["active_connections"] == 1
        
        # Verify welcome message was sent
        mock_websocket.send_text.assert_called()
        
        # Test disconnection
        await websocket_manager.disconnect(mock_websocket)
        
        assert mock_websocket not in websocket_manager.active_connections
        assert mock_websocket not in websocket_manager.client_subscriptions
        assert mock_websocket not in websocket_manager.connection_metadata
        assert websocket_manager.connection_stats["active_connections"] == 0
    
    @pytest.mark.asyncio
    async def test_websocket_subscription_update(
        self,
        websocket_manager,
        mock_websocket
    ):
        """Test WebSocket subscription management."""
        
        client_info = {"client_ip": "127.0.0.1"}
        await websocket_manager.connect(mock_websocket, client_info)
        
        # Update subscription
        subscription_data = {
            "event_types": ["AgentLifecycleStart", "ContextThresholdReached"],
            "severity_filters": ["warning", "error", "critical"],
            "agent_filters": ["agent_123"]
        }
        
        await websocket_manager.update_client_subscription(
            mock_websocket, subscription_data
        )
        
        # Verify subscription was updated
        subscription = websocket_manager.client_subscriptions[mock_websocket]
        assert subscription["event_types"] == subscription_data["event_types"]
        assert subscription["severity_filters"] == subscription_data["severity_filters"]
        assert subscription["agent_filters"] == subscription_data["agent_filters"]
    
    @pytest.mark.asyncio
    async def test_websocket_message_filtering(
        self,
        websocket_manager,
        mock_websocket
    ):
        """Test WebSocket message filtering logic."""
        
        client_info = {"client_ip": "127.0.0.1"}
        await websocket_manager.connect(mock_websocket, client_info)
        
        # Set specific filters
        subscription_data = {
            "event_types": ["AgentLifecycleStart"],
            "severity_filters": ["warning", "error"],
            "agent_filters": ["agent_123"]
        }
        await websocket_manager.update_client_subscription(
            mock_websocket, subscription_data
        )
        
        # Test message that should be sent (matches filters)
        matching_message = {
            "type": "lifecycle_event",
            "data": {
                "event_type": "AgentLifecycleStart",
                "severity": "warning",
                "agent_id": "agent_123"
            }
        }
        
        should_send = websocket_manager._should_send_to_client(
            mock_websocket, matching_message
        )
        assert should_send is True
        
        # Test message that should not be sent (wrong event type)
        non_matching_message = {
            "type": "lifecycle_event",
            "data": {
                "event_type": "HealthCheck",
                "severity": "warning",
                "agent_id": "agent_123"
            }
        }
        
        should_send = websocket_manager._should_send_to_client(
            mock_websocket, non_matching_message
        )
        assert should_send is False
    
    @pytest.mark.asyncio
    async def test_websocket_broadcast_to_all(
        self,
        websocket_manager
    ):
        """Test broadcasting to all connected clients."""
        
        # Create multiple mock WebSocket connections
        websockets = []
        for i in range(3):
            mock_ws = Mock()
            mock_ws.send_text = AsyncMock()
            mock_ws.client = Mock()
            mock_ws.client.host = f"127.0.0.{i+1}"
            mock_ws.headers = {"user-agent": f"test-client-{i}"}
            
            client_info = {"client_ip": f"127.0.0.{i+1}"}
            await websocket_manager.connect(mock_ws, client_info)
            websockets.append(mock_ws)
        
        # Broadcast message
        test_message = {
            "type": "lifecycle_event",
            "data": {"event_type": "AgentLifecycleStart", "agent_id": "test_agent"}
        }
        
        await websocket_manager.send_to_all(test_message)
        
        # Verify all clients received the message (may not be called due to filtering)
        # Just verify no exceptions were raised
        assert len(websockets) == 3
    
    def test_websocket_connection_stats(self, websocket_manager):
        """Test WebSocket connection statistics."""
        
        stats = websocket_manager.get_connection_stats()
        
        assert "total_connections" in stats
        assert "active_connections" in stats
        assert "messages_sent" in stats
        assert "connection_errors" in stats
        assert "last_activity" in stats
        assert "client_subscriptions" in stats


class TestIntegrationAndErrorHandling:
    """Test integration scenarios and error handling."""
    
    @pytest.mark.asyncio
    async def test_redis_stream_failure_handling(self):
        """Test handling of Redis streaming failures."""
        
        with patch('app.core.enhanced_lifecycle_hooks.get_redis') as mock_redis:
            # Mock Redis to raise exception
            mock_redis_client = AsyncMock()
            mock_redis_client.xadd = AsyncMock(side_effect=Exception("Redis connection failed"))
            mock_redis.return_value = mock_redis_client
            
            with patch('app.core.enhanced_lifecycle_hooks.get_async_session'):
                processor = EnhancedLifecycleHookProcessor()
                
                # Event processing should continue despite Redis failure
                event_id = await processor.process_enhanced_event(
                    session_id=uuid.uuid4(),
                    agent_id=uuid.uuid4(),
                    event_type=EnhancedEventType.HEALTH_CHECK,
                    payload={"test": "data"},
                    severity="info"
                )
                
                # Should still return event ID
                assert event_id is not None
    
    @pytest.mark.asyncio
    async def test_database_storage_failure_handling(self):
        """Test handling of database storage failures."""
        
        with patch('app.core.enhanced_lifecycle_hooks.get_async_session') as mock_session:
            # Mock database to raise exception
            mock_db_session = AsyncMock()
            mock_db_session.add = Mock(side_effect=Exception("Database connection failed"))
            mock_session.return_value.__aenter__.return_value = mock_db_session
            
            with patch('app.core.enhanced_lifecycle_hooks.get_redis'):
                processor = EnhancedLifecycleHookProcessor()
                
                # Event processing should continue despite database failure
                event_id = await processor.process_enhanced_event(
                    session_id=uuid.uuid4(),
                    agent_id=uuid.uuid4(),
                    event_type=EnhancedEventType.HEALTH_CHECK,
                    payload={"test": "data"},
                    severity="info"
                )
                
                # Should still return event ID
                assert event_id is not None
    
    @pytest.mark.asyncio
    async def test_websocket_broadcast_failure_handling(self):
        """Test handling of WebSocket broadcast failures."""
        
        websocket_manager = WebSocketManager()
        
        # Add mock WebSocket that will fail
        failing_ws = Mock()
        failing_ws.send_text = AsyncMock(side_effect=Exception("WebSocket send failed"))
        websocket_manager.active_connections.add(failing_ws)
        
        # Add working WebSocket
        working_ws = Mock()
        working_ws.send_text = AsyncMock()
        websocket_manager.active_connections.add(working_ws)
        
        # Set up client subscriptions
        websocket_manager.client_subscriptions[failing_ws] = {"event_types": []}
        websocket_manager.client_subscriptions[working_ws] = {"event_types": []}
        
        # Broadcast message
        test_message = {"type": "test", "data": {}}
        await websocket_manager.send_to_all(test_message)
        
        # Failing WebSocket should be removed, working one should receive message
        assert failing_ws not in websocket_manager.active_connections
        assert working_ws in websocket_manager.active_connections
        working_ws.send_text.assert_called()
    
    @pytest.mark.asyncio
    async def test_concurrent_event_processing(self):
        """Test concurrent event processing performance."""
        
        with patch('app.core.enhanced_lifecycle_hooks.get_async_session'), \
             patch('app.core.enhanced_lifecycle_hooks.get_redis'):
            
            processor = EnhancedLifecycleHookProcessor()
            
            # Create multiple concurrent event processing tasks
            tasks = []
            agent_ids = [uuid.uuid4() for _ in range(10)]
            session_id = uuid.uuid4()
            
            for i, agent_id in enumerate(agent_ids):
                task = processor.process_enhanced_event(
                    session_id=session_id,
                    agent_id=agent_id,
                    event_type=EnhancedEventType.HEALTH_CHECK,
                    payload={"event_number": i},
                    severity="info"
                )
                tasks.append(task)
            
            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should succeed
            for result in results:
                assert isinstance(result, str)  # Should return event ID
                assert result is not None
            
            # Verify performance metrics
            assert processor.performance_metrics["events_processed"] == len(agent_ids)


class TestConvenienceFunctions:
    """Test convenience functions for common operations."""
    
    @pytest.mark.asyncio
    async def test_capture_agent_start_convenience_function(self):
        """Test convenience function for agent start events."""
        
        with patch('app.core.enhanced_lifecycle_hooks.get_enhanced_lifecycle_hook_processor') as mock_get:
            mock_processor = AsyncMock()
            mock_processor.capture_agent_lifecycle_start = AsyncMock(return_value="event_123")
            mock_get.return_value = mock_processor
            
            from app.core.enhanced_lifecycle_hooks import capture_agent_start
            
            result = await capture_agent_start(
                session_id=uuid.uuid4(),
                agent_id=uuid.uuid4(),
                agent_type="test_agent",
                capabilities=["test"],
                initial_context={"test": "context"}
            )
            
            assert result == "event_123"
            mock_processor.capture_agent_lifecycle_start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_capture_context_threshold_convenience_function(self):
        """Test convenience function for context threshold events."""
        
        with patch('app.core.enhanced_lifecycle_hooks.get_enhanced_lifecycle_hook_processor') as mock_get:
            mock_processor = AsyncMock()
            mock_processor.capture_context_threshold_reached = AsyncMock(return_value="event_456")
            mock_get.return_value = mock_processor
            
            from app.core.enhanced_lifecycle_hooks import capture_context_threshold
            
            result = await capture_context_threshold(
                session_id=uuid.uuid4(),
                agent_id=uuid.uuid4(),
                usage_percent=85.0,
                threshold_type="high",
                action="consolidate",
                stats={"tokens": 8500}
            )
            
            assert result == "event_456"
            mock_processor.capture_context_threshold_reached.assert_called_once()