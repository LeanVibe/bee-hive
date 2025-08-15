"""
Comprehensive WebSocket Integration Tests for Project Index Events

Tests all aspects of the WebSocket integration including event publishing,
filtering, batching, performance optimization, and client management.
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch

import pytest_asyncio
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket

from app.project_index.websocket_events import (
    ProjectIndexEventPublisher, ProjectIndexEventType, ProjectIndexWebSocketEvent,
    ProjectIndexUpdateData, AnalysisProgressData, DependencyChangeData, ContextOptimizedData
)
from app.project_index.event_filters import EventFilter, UserPreferences, FilterRule, FilterCriteria
from app.project_index.event_history import EventHistoryManager, ReplayRequest
from app.project_index.websocket_performance import WebSocketPerformanceManager, EventPriority
from app.project_index.websocket_integration import ProjectIndexWebSocketManager


@pytest.fixture
async def mock_redis_client():
    """Mock Redis client for testing."""
    mock_redis = AsyncMock()
    mock_redis.publish = AsyncMock()
    mock_redis.subscribe = AsyncMock()
    mock_redis.get = AsyncMock()
    mock_redis.setex = AsyncMock()
    mock_redis.keys = AsyncMock(return_value=[])
    mock_redis.pubsub = AsyncMock()
    return mock_redis


@pytest.fixture
async def event_publisher(mock_redis_client):
    """Create event publisher for testing."""
    publisher = ProjectIndexEventPublisher(mock_redis_client)
    return publisher


@pytest.fixture
async def event_filter():
    """Create event filter for testing."""
    return EventFilter()


@pytest.fixture
async def history_manager(mock_redis_client):
    """Create event history manager for testing."""
    return EventHistoryManager(mock_redis_client)


@pytest.fixture
async def performance_manager():
    """Create performance manager for testing."""
    return WebSocketPerformanceManager()


@pytest.fixture
async def websocket_manager(mock_redis_client):
    """Create WebSocket manager for testing."""
    manager = ProjectIndexWebSocketManager(mock_redis_client)
    await manager.initialize()
    return manager


@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection."""
    websocket = Mock(spec=WebSocket)
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.receive_text = AsyncMock()
    websocket.close = AsyncMock()
    return websocket


class TestProjectIndexEventPublisher:
    """Test event publisher functionality."""
    
    @pytest.mark.asyncio
    async def test_publish_project_updated_event(self, event_publisher):
        """Test publishing project updated events."""
        project_id = uuid.uuid4()
        
        update_data = ProjectIndexUpdateData(
            project_id=project_id,
            project_name="test-project",
            files_analyzed=100,
            files_updated=50,
            dependencies_updated=25,
            analysis_duration_seconds=30.5,
            status="completed",
            statistics={
                "total_files": 100,
                "languages_detected": ["python", "javascript"],
                "dependency_count": 25,
                "complexity_score": 0.7
            },
            error_count=0,
            warnings=[]
        )
        
        subscribers_notified = await event_publisher.publish_project_updated(project_id, update_data)
        
        # Should return 0 subscribers for test (no actual subscribers)
        assert subscribers_notified == 0
        
        # Verify metrics updated
        metrics = event_publisher.get_metrics()
        assert metrics["events_published"] == 1
    
    @pytest.mark.asyncio
    async def test_publish_analysis_progress_event(self, event_publisher):
        """Test publishing analysis progress events."""
        session_id = uuid.uuid4()
        project_id = uuid.uuid4()
        
        progress_data = AnalysisProgressData(
            session_id=session_id,
            project_id=project_id,
            analysis_type="full",
            progress_percentage=75,
            files_processed=75,
            total_files=100,
            current_file="main.py",
            estimated_completion=datetime.utcnow() + timedelta(minutes=5),
            processing_rate=2.5,
            performance_metrics={
                "memory_usage_mb": 150.0,
                "cpu_usage_percent": 45.0,
                "parallel_tasks": 4
            },
            errors_encountered=1,
            last_error="File not found: missing.py"
        )
        
        subscribers_notified = await event_publisher.publish_analysis_progress(session_id, progress_data)
        
        assert subscribers_notified == 0
        
        # Verify metrics
        metrics = event_publisher.get_metrics()
        assert metrics["events_published"] == 1
    
    @pytest.mark.asyncio
    async def test_publish_dependency_changed_event(self, event_publisher):
        """Test publishing dependency changed events."""
        project_id = uuid.uuid4()
        
        dependency_data = DependencyChangeData(
            project_id=project_id,
            file_path="/src/utils.py",
            change_type="modified",
            dependency_details={
                "target_file": "/src/core.py",
                "relationship_type": "import",
                "line_number": 15,
                "is_circular": False
            },
            impact_analysis={
                "affected_files": ["/src/main.py", "/src/app.py"],
                "potential_issues": ["breaking_change"],
                "recommendations": ["update_imports"]
            },
            file_metadata={
                "language": "python",
                "file_size": 2048,
                "last_modified": datetime.utcnow().isoformat()
            }
        )
        
        subscribers_notified = await event_publisher.publish_dependency_changed(project_id, dependency_data)
        
        assert subscribers_notified == 0
        
        # Verify metrics
        metrics = event_publisher.get_metrics()
        assert metrics["events_published"] == 1
    
    @pytest.mark.asyncio
    async def test_publish_context_optimized_event(self, event_publisher):
        """Test publishing context optimized events."""
        context_id = uuid.uuid4()
        project_id = uuid.uuid4()
        
        context_data = ContextOptimizedData(
            context_id=context_id,
            project_id=project_id,
            task_description="Optimize context for bug analysis",
            task_type="bug_analysis",
            optimization_results={
                "selected_files": 25,
                "total_tokens": 15000,
                "relevance_scores": {
                    "high": 10,
                    "medium": 15,
                    "low": 5
                },
                "confidence_score": 0.92,
                "processing_time_ms": 250
            },
            recommendations={
                "architectural_patterns": ["mvc", "observer"],
                "potential_challenges": ["tight_coupling"],
                "suggested_approach": "Focus on controller and model layers"
            },
            performance_metrics={
                "cache_hit_rate": 0.88,
                "ml_analysis_time_ms": 180,
                "context_assembly_time_ms": 70
            }
        )
        
        subscribers_notified = await event_publisher.publish_context_optimized(context_id, context_data)
        
        assert subscribers_notified == 0
        
        # Verify metrics
        metrics = event_publisher.get_metrics()
        assert metrics["events_published"] == 1
    
    @pytest.mark.asyncio
    async def test_subscription_management(self, event_publisher):
        """Test subscription management functionality."""
        client_id = "test-client-123"
        project_id = uuid.uuid4()
        
        # Subscribe to project
        await event_publisher.subscribe_to_project(client_id, project_id)
        
        # Subscribe to event types
        await event_publisher.subscribe_to_events(client_id, ["project_index_updated", "analysis_progress"])
        
        # Verify subscriptions
        subscribers = event_publisher.subscriptions.get_subscribers_for_event(
            ProjectIndexEventType.PROJECT_INDEX_UPDATED,
            str(project_id)
        )
        
        assert client_id in subscribers
        
        # Unsubscribe
        await event_publisher.unsubscribe_client(client_id)
        
        # Verify unsubscribed
        subscribers = event_publisher.subscriptions.get_subscribers_for_event(
            ProjectIndexEventType.PROJECT_INDEX_UPDATED,
            str(project_id)
        )
        
        assert client_id not in subscribers


class TestEventFilter:
    """Test event filtering functionality."""
    
    @pytest.mark.asyncio
    async def test_relevance_scoring(self, event_filter):
        """Test relevance score calculation."""
        event = ProjectIndexWebSocketEvent(
            type=ProjectIndexEventType.PROJECT_INDEX_UPDATED,
            data={
                "project_id": str(uuid.uuid4()),
                "files_analyzed": 100,
                "dependencies_updated": 25,
                "error_count": 0
            }
        )
        
        # Test with no user preferences
        relevance_score = event_filter.relevance_scorer.calculate_relevance(event)
        assert 0.0 <= relevance_score <= 1.0
        assert relevance_score > 0.5  # Should be high for project updates
        
        # Test with user preferences
        user_prefs = UserPreferences(
            user_id="test-user",
            preferred_languages=["python"],
            high_impact_only=True
        )
        
        relevance_score_with_prefs = event_filter.relevance_scorer.calculate_relevance(event, user_prefs)
        assert 0.0 <= relevance_score_with_prefs <= 1.0
    
    @pytest.mark.asyncio
    async def test_event_filtering(self, event_filter):
        """Test event filtering logic."""
        user_id = "test-user"
        
        # Set user preferences
        user_prefs = UserPreferences(
            user_id=user_id,
            preferred_languages=["python", "javascript"],
            ignored_file_patterns=["*.test.js"],
            min_progress_updates=25,
            high_impact_only=False,
            notification_frequency="normal"
        )
        
        event_filter.set_user_preferences(user_id, user_prefs)
        
        # Test high relevance event (should pass)
        high_relevance_event = ProjectIndexWebSocketEvent(
            type=ProjectIndexEventType.PROJECT_INDEX_UPDATED,
            data={
                "project_id": str(uuid.uuid4()),
                "files_analyzed": 100,
                "error_count": 0
            }
        )
        
        should_deliver = await event_filter.should_deliver_event(high_relevance_event, user_id)
        assert should_deliver is True
        
        # Test progress event with low progress (should be filtered)
        low_progress_event = ProjectIndexWebSocketEvent(
            type=ProjectIndexEventType.ANALYSIS_PROGRESS,
            data={
                "project_id": str(uuid.uuid4()),
                "progress_percentage": 15  # Below 25% threshold
            }
        )
        
        should_deliver = await event_filter.should_deliver_event(low_progress_event, user_id)
        assert should_deliver is False
        
        # Test progress event with milestone progress (should pass)
        milestone_progress_event = ProjectIndexWebSocketEvent(
            type=ProjectIndexEventType.ANALYSIS_PROGRESS,
            data={
                "project_id": str(uuid.uuid4()),
                "progress_percentage": 50  # Milestone progress
            }
        )
        
        should_deliver = await event_filter.should_deliver_event(milestone_progress_event, user_id)
        assert should_deliver is True
    
    @pytest.mark.asyncio
    async def test_global_filters(self, event_filter):
        """Test global filter rules."""
        # Add global filter to only allow critical events
        critical_filter = FilterRule(
            criteria=FilterCriteria.EVENT_TYPE,
            operator="in",
            value=["project_index_updated", "dependency_changed"],
            weight=1.0,
            enabled=True
        )
        
        event_filter.add_global_filter(critical_filter)
        
        # Test allowed event type
        allowed_event = ProjectIndexWebSocketEvent(
            type=ProjectIndexEventType.PROJECT_INDEX_UPDATED,
            data={"project_id": str(uuid.uuid4())}
        )
        
        should_deliver = await event_filter.should_deliver_event(allowed_event, "test-user")
        assert should_deliver is True
        
        # Test filtered event type
        filtered_event = ProjectIndexWebSocketEvent(
            type=ProjectIndexEventType.ANALYSIS_PROGRESS,
            data={"project_id": str(uuid.uuid4()), "progress_percentage": 50}
        )
        
        should_deliver = await event_filter.should_deliver_event(filtered_event, "test-user")
        assert should_deliver is False


class TestEventHistory:
    """Test event history functionality."""
    
    @pytest.mark.asyncio
    async def test_event_storage_and_retrieval(self, history_manager):
        """Test storing and retrieving events."""
        project_id = str(uuid.uuid4())
        
        # Create test event
        event = ProjectIndexWebSocketEvent(
            type=ProjectIndexEventType.PROJECT_INDEX_UPDATED,
            data={
                "project_id": project_id,
                "files_analyzed": 50
            }
        )
        
        # Store event
        event_id = await history_manager.store_event(event, project_id)
        assert event_id is not None
        
        # Retrieve events
        events = await history_manager.get_events(project_id, limit=10)
        assert len(events) == 1
        assert events[0].event_id == event_id
        assert events[0].event_type == "project_index_updated"
    
    @pytest.mark.asyncio
    async def test_event_replay(self, history_manager):
        """Test event replay functionality."""
        project_id = str(uuid.uuid4())
        client_id = "test-client"
        
        # Store multiple events
        events_to_store = []
        for i in range(5):
            event = ProjectIndexWebSocketEvent(
                type=ProjectIndexEventType.ANALYSIS_PROGRESS,
                data={
                    "project_id": project_id,
                    "progress_percentage": i * 20
                }
            )
            events_to_store.append(event)
            await history_manager.store_event(event, project_id)
        
        # Create replay request
        replay_request = ReplayRequest(
            client_id=client_id,
            project_id=project_id,
            max_events=3,
            include_delivered=False
        )
        
        # Get replay events
        replay_events = await history_manager.replay_events(replay_request)
        
        assert len(replay_events) <= 3
        assert all("replay" in event for event in replay_events)
    
    @pytest.mark.asyncio
    async def test_event_cleanup(self, history_manager):
        """Test event cleanup functionality."""
        cleanup_stats = await history_manager.cleanup_expired_events()
        
        assert isinstance(cleanup_stats, dict)
        assert "memory_cleaned" in cleanup_stats
        assert "redis_cleaned" in cleanup_stats
        assert "database_cleaned" in cleanup_stats


class TestWebSocketPerformance:
    """Test WebSocket performance optimization."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, performance_manager):
        """Test rate limiting functionality."""
        client_id = "test-client"
        
        # Add connection
        performance_manager.add_connection(client_id)
        
        # Test rate limiting
        within_limit = performance_manager.rate_limiter.check_rate_limit(client_id, 1)
        assert within_limit is True
        
        # Exhaust rate limit
        for _ in range(100):  # Default max tokens
            performance_manager.rate_limiter.check_rate_limit(client_id, 1)
        
        # Should be rate limited now
        rate_limited = performance_manager.rate_limiter.check_rate_limit(client_id, 1)
        assert rate_limited is False
    
    @pytest.mark.asyncio
    async def test_event_compression(self, performance_manager):
        """Test event compression."""
        large_event_data = {
            "type": "project_index_updated",
            "data": {
                "project_id": str(uuid.uuid4()),
                "large_data": "x" * 2000  # Large payload
            }
        }
        
        compressed_data, compression_type, compression_ratio = (
            performance_manager.compressor.compress_event(large_event_data)
        )
        
        assert compression_ratio < 1.0  # Should be compressed
        assert len(compressed_data) < len(json.dumps(large_event_data).encode())
    
    @pytest.mark.asyncio
    async def test_connection_health_tracking(self, performance_manager):
        """Test connection health tracking."""
        client_id = "test-client"
        
        # Add connection
        performance_manager.add_connection(client_id)
        
        # Record successful message
        performance_manager.connection_pool.record_message_sent(
            client_id, 1024, 50.0, True
        )
        
        # Check health
        health = performance_manager.connection_pool.get_connection_health(client_id)
        assert 0.0 <= health <= 1.0
        
        # Record failed message
        performance_manager.connection_pool.record_message_sent(
            client_id, 0, 0, False
        )
        
        # Health should decrease
        new_health = performance_manager.connection_pool.get_connection_health(client_id)
        assert new_health < health


class TestWebSocketIntegration:
    """Test complete WebSocket integration."""
    
    @pytest.mark.asyncio
    async def test_websocket_connection_lifecycle(self, websocket_manager, mock_websocket):
        """Test complete WebSocket connection lifecycle."""
        user_id = "test-user"
        
        # Mock WebSocket message handling
        messages_received = []
        
        async def mock_receive_text():
            # Simulate client subscription message
            return json.dumps({
                "action": "subscribe",
                "event_types": ["project_index_updated", "analysis_progress"],
                "project_id": str(uuid.uuid4())
            })
        
        mock_websocket.receive_text.side_effect = [
            await mock_receive_text(),
            # Simulate WebSocket disconnect
            Exception("WebSocket disconnected")
        ]
        
        # Handle connection (should not raise exception)
        try:
            await websocket_manager.handle_connection(mock_websocket, user_id)
        except Exception:
            pass  # Expected due to mock disconnect
        
        # Verify connection was handled
        assert mock_websocket.accept.called
        assert mock_websocket.send_text.called
    
    @pytest.mark.asyncio
    async def test_message_handling(self, websocket_manager):
        """Test message handling functionality."""
        connection_id = "test-connection"
        user_id = "test-user"
        
        # Set up connection metadata
        websocket_manager.connection_users[connection_id] = user_id
        websocket_manager.connection_metadata[connection_id] = {
            "subscriptions": set(),
            "user_id": user_id
        }
        
        # Test subscription message
        subscription_message = json.dumps({
            "action": "subscribe",
            "event_types": ["project_index_updated"],
            "project_id": str(uuid.uuid4())
        })
        
        # Should not raise exception
        await websocket_manager._handle_message(connection_id, subscription_message)
        
        # Test preferences message
        preferences_message = json.dumps({
            "action": "set_preferences",
            "preferences": {
                "preferred_languages": ["python"],
                "high_impact_only": True
            }
        })
        
        await websocket_manager._handle_message(connection_id, preferences_message)
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, websocket_manager):
        """Test metrics collection."""
        metrics = websocket_manager.get_metrics()
        
        assert "websocket_manager" in metrics
        assert "performance" in metrics
        assert "event_filter" in metrics
        assert "active_connections" in metrics
        assert "connection_details" in metrics
        
        # Verify metric structure
        assert isinstance(metrics["active_connections"], int)
        assert isinstance(metrics["connection_details"], dict)


class TestEndToEndIntegration:
    """Test end-to-end WebSocket event flow."""
    
    @pytest.mark.asyncio
    async def test_complete_event_flow(self, event_publisher, event_filter, history_manager):
        """Test complete event flow from publication to delivery."""
        project_id = uuid.uuid4()
        client_id = "test-client"
        user_id = "test-user"
        
        # Set up client subscription
        await event_publisher.subscribe_to_project(client_id, project_id)
        await event_publisher.subscribe_to_events(client_id, ["project_index_updated"])
        
        # Set up user preferences
        user_prefs = UserPreferences(
            user_id=user_id,
            preferred_languages=["python"],
            high_impact_only=False
        )
        event_filter.set_user_preferences(user_id, user_prefs)
        
        # Create and publish event
        update_data = ProjectIndexUpdateData(
            project_id=project_id,
            project_name="test-project",
            files_analyzed=50,
            files_updated=25,
            dependencies_updated=10,
            analysis_duration_seconds=15.0,
            status="completed",
            statistics={
                "total_files": 50,
                "languages_detected": ["python"],
                "dependency_count": 10
            }
        )
        
        # Publish event
        subscribers_notified = await event_publisher.publish_project_updated(project_id, update_data)
        
        # In test environment, no actual subscribers
        assert subscribers_notified == 0
        
        # But verify event would be processed correctly
        test_event = ProjectIndexWebSocketEvent(
            type=ProjectIndexEventType.PROJECT_INDEX_UPDATED,
            data={
                "project_id": str(project_id),
                "project_name": "test-project",
                "files_analyzed": 50
            }
        )
        
        # Test filtering
        should_deliver = await event_filter.should_deliver_event(test_event, user_id)
        assert should_deliver is True
        
        # Test history storage
        event_id = await history_manager.store_event(test_event, str(project_id))
        assert event_id is not None
        
        # Test retrieval
        events = await history_manager.get_events(str(project_id), limit=1)
        assert len(events) == 1
        assert events[0].event_type == "project_index_updated"
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, performance_manager):
        """Test performance under simulated load."""
        # Simulate multiple connections
        connection_ids = [f"client-{i}" for i in range(50)]
        
        for client_id in connection_ids:
            performance_manager.add_connection(client_id)
        
        # Simulate event processing
        test_event = ProjectIndexWebSocketEvent(
            type=ProjectIndexEventType.ANALYSIS_PROGRESS,
            data={
                "project_id": str(uuid.uuid4()),
                "progress_percentage": 50
            }
        )
        
        target_clients = set(connection_ids[:25])  # Half the clients
        
        # Process event (simulated)
        results = await performance_manager.process_event(
            test_event,
            target_clients,
            EventPriority.NORMAL
        )
        
        # Verify results structure
        assert "events_sent" in results
        assert "events_failed" in results
        assert "compression_ratio" in results
        assert "rate_limited_clients" in results
        
        # Get performance summary
        summary = performance_manager.get_performance_summary()
        assert "global_metrics" in summary
        assert "connection_pool" in summary
        assert "priority_queue" in summary


# Performance and load testing
class TestPerformanceAndLoad:
    """Test performance and load handling."""
    
    @pytest.mark.asyncio
    async def test_concurrent_event_publishing(self, event_publisher):
        """Test concurrent event publishing."""
        project_id = uuid.uuid4()
        
        # Create multiple events to publish concurrently
        events = []
        for i in range(10):
            update_data = ProjectIndexUpdateData(
                project_id=project_id,
                project_name=f"project-{i}",
                files_analyzed=i * 10,
                files_updated=i * 5,
                dependencies_updated=i * 2,
                analysis_duration_seconds=float(i),
                status="completed",
                statistics={"total_files": i * 10}
            )
            events.append(event_publisher.publish_project_updated(project_id, update_data))
        
        # Publish all events concurrently
        results = await asyncio.gather(*events)
        
        # All should complete successfully
        assert len(results) == 10
        assert all(isinstance(r, int) for r in results)
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, history_manager):
        """Test memory usage with large event history."""
        project_id = str(uuid.uuid4())
        
        # Store many events
        for i in range(100):
            event = ProjectIndexWebSocketEvent(
                type=ProjectIndexEventType.ANALYSIS_PROGRESS,
                data={
                    "project_id": project_id,
                    "progress_percentage": i,
                    "files_processed": i * 10
                }
            )
            await history_manager.store_event(event, project_id)
        
        # Get project summary
        summary = await history_manager.get_project_event_summary(project_id)
        
        assert summary["total_events"] == 100
        assert "storage_layers" in summary
        assert summary["project_id"] == project_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])