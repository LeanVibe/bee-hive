"""
Comprehensive tests for Coordination Dashboard System.

Tests include:
- Agent graph node and edge management
- Real-time event processing and WebSocket streaming
- Session-based filtering and color management
- Communication transcript analysis
- Performance and scalability validation
- Integration with enhanced lifecycle hooks
"""

import asyncio
import json
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from app.core.coordination_dashboard import (
    CoordinationDashboard,
    AgentGraphNode,
    AgentGraphEdge,
    AgentNodeType,
    AgentNodeStatus,
    AgentEdgeType,
    EventFilter,
    AgentCommunicationEvent,
    SessionColorManager,
    coordination_dashboard
)
from app.core.enhanced_lifecycle_hooks import LifecycleEventData, EnhancedEventType


class TestSessionColorManager:
    """Test session color management functionality."""
    
    def test_color_manager_initialization(self):
        """Test color manager initializes with predefined palette."""
        manager = SessionColorManager()
        
        assert len(manager.color_palette) >= 10
        assert all(color.startswith("#") for color in manager.color_palette)
        assert len(manager.session_colors) == 0
        assert manager.color_index == 0
    
    def test_consistent_session_colors(self):
        """Test that same session ID always gets same color."""
        manager = SessionColorManager()
        session_id = "test_session_123"
        
        color1 = manager.get_session_color(session_id)
        color2 = manager.get_session_color(session_id)
        color3 = manager.get_session_color(session_id)
        
        assert color1 == color2 == color3
        assert color1 in manager.color_palette
        assert session_id in manager.session_colors
    
    def test_different_sessions_different_colors(self):
        """Test that different sessions get different colors (usually)."""
        manager = SessionColorManager()
        
        colors = []
        for i in range(5):
            session_id = f"session_{i}"
            color = manager.get_session_color(session_id)
            colors.append(color)
        
        # Should get at least some different colors
        assert len(set(colors)) > 1
    
    def test_event_color_mapping(self):
        """Test event color mapping for sessions."""
        manager = SessionColorManager()
        session_id = "test_session"
        
        color_map = manager.get_event_color_map(session_id)
        
        assert isinstance(color_map, dict)
        assert "tool_execution" in color_map
        assert "context_creation" in color_map
        assert "agent_communication" in color_map
        assert "error_event" in color_map
        assert "success_event" in color_map
        
        # Error and success should have fixed colors
        assert color_map["error_event"] == "#FF4444"
        assert color_map["success_event"] == "#44FF44"


class TestAgentGraphStructures:
    """Test agent graph node and edge structures."""
    
    def test_agent_graph_node_creation(self):
        """Test agent graph node creation with all fields."""
        node = AgentGraphNode(
            id="agent_001",
            label="Test Agent",
            type=AgentNodeType.AGENT,
            status=AgentNodeStatus.ACTIVE,
            position={"x": 100.0, "y": 200.0},
            metadata={
                "agent_id": "test_agent",
                "session_id": "session_1",
                "agent_type": "test"
            }
        )
        
        assert node.id == "agent_001"
        assert node.label == "Test Agent"
        assert node.type == AgentNodeType.AGENT
        assert node.status == AgentNodeStatus.ACTIVE
        assert node.position == {"x": 100.0, "y": 200.0}
        assert node.size == 1.0  # Default value
        assert node.color == "#4A90E2"  # Default value
        assert node.created_at is not None
        assert node.last_updated is not None
    
    def test_agent_graph_node_to_dict(self):
        """Test node serialization to dictionary."""
        node = AgentGraphNode(
            id="agent_001",
            label="Test Agent",
            type=AgentNodeType.AGENT,
            status=AgentNodeStatus.ACTIVE,
            position={"x": 100.0, "y": 200.0},
            metadata={"test": "data"}
        )
        
        node_dict = node.to_dict()
        
        assert node_dict["id"] == "agent_001"
        assert node_dict["type"] == "agent"  # Enum value
        assert node_dict["status"] == "active"  # Enum value
        assert node_dict["position"] == {"x": 100.0, "y": 200.0}
        assert "created_at" in node_dict
        assert "last_updated" in node_dict
    
    def test_agent_graph_edge_creation(self):
        """Test agent graph edge creation."""
        edge = AgentGraphEdge(
            id="edge_001",
            source="agent_001",
            target="tool_001",
            type=AgentEdgeType.TOOL_CALL,
            weight=2.5,
            timestamp=datetime.utcnow(),
            metadata={"tool_call_count": 3}
        )
        
        assert edge.id == "edge_001"
        assert edge.source == "agent_001"
        assert edge.target == "tool_001"
        assert edge.type == AgentEdgeType.TOOL_CALL
        assert edge.weight == 2.5
        assert edge.width == 1.0  # Default value
        assert edge.style == "solid"  # Default value
    
    def test_agent_graph_edge_to_dict(self):
        """Test edge serialization to dictionary."""
        timestamp = datetime.utcnow()
        edge = AgentGraphEdge(
            id="edge_001",
            source="agent_001",
            target="tool_001",
            type=AgentEdgeType.TOOL_CALL,
            weight=2.5,
            timestamp=timestamp,
            metadata={"test": "data"}
        )
        
        edge_dict = edge.to_dict()
        
        assert edge_dict["id"] == "edge_001"
        assert edge_dict["type"] == "tool_call"  # Enum value
        assert edge_dict["timestamp"] == timestamp.isoformat()
        assert edge_dict["weight"] == 2.5


class TestEventFilter:
    """Test event filtering functionality."""
    
    def test_event_filter_defaults(self):
        """Test event filter default values."""
        filter_obj = EventFilter()
        
        assert filter_obj.session_ids == []
        assert filter_obj.agent_types == []
        assert filter_obj.event_types == []
        assert filter_obj.node_types == []
        assert filter_obj.time_range is None
        assert filter_obj.severity_levels == []
        assert filter_obj.include_system_events is True
        assert filter_obj.max_events == 1000
    
    def test_event_filter_custom_values(self):
        """Test event filter with custom values."""
        filter_obj = EventFilter(
            session_ids=["session_1", "session_2"],
            agent_types=["type_a", "type_b"],
            event_types=["tool_call", "context_creation"],
            max_events=500,
            include_system_events=False
        )
        
        assert filter_obj.session_ids == ["session_1", "session_2"]
        assert filter_obj.agent_types == ["type_a", "type_b"]
        assert filter_obj.event_types == ["tool_call", "context_creation"]
        assert filter_obj.max_events == 500
        assert filter_obj.include_system_events is False


class TestAgentCommunicationEvent:
    """Test agent communication event model."""
    
    def test_communication_event_creation(self):
        """Test communication event creation."""
        event = AgentCommunicationEvent(
            session_id="session_1",
            source_agent_id="agent_001",
            target_agent_id="agent_002",
            message_type="direct_message",
            content={"message": "Hello", "data": 123}
        )
        
        assert event.session_id == "session_1"
        assert event.source_agent_id == "agent_001"
        assert event.target_agent_id == "agent_002"
        assert event.message_type == "direct_message"
        assert event.content == {"message": "Hello", "data": 123}
        assert event.context_shared is False
        assert event.tool_calls == []
        assert isinstance(event.timestamp, datetime)
    
    def test_communication_event_with_tool_calls(self):
        """Test communication event with tool calls."""
        tool_calls = [
            {"tool_id": "git_clone", "status": "success"},
            {"tool_id": "docker_build", "status": "pending"}
        ]
        
        event = AgentCommunicationEvent(
            session_id="session_1",
            source_agent_id="agent_001",
            message_type="tool_coordination",
            content={"coordination": "multi_tool"},
            context_shared=True,
            tool_calls=tool_calls
        )
        
        assert event.tool_calls == tool_calls
        assert event.context_shared is True


class TestCoordinationDashboard:
    """Test coordination dashboard core functionality."""
    
    @pytest.fixture
    def dashboard(self):
        """Create fresh dashboard for each test."""
        return CoordinationDashboard()
    
    @pytest.fixture
    def sample_lifecycle_event(self):
        """Sample lifecycle event for testing."""
        return LifecycleEventData(
            session_id="test_session",
            agent_id="test_agent_001",
            event_type=EnhancedEventType.AGENT_LIFECYCLE_START,
            timestamp=datetime.utcnow().isoformat(),
            payload={
                "agent_type": "autonomous",
                "current_task": "test_task"
            }
        )
    
    def test_dashboard_initialization(self, dashboard):
        """Test dashboard initializes correctly."""
        assert len(dashboard.nodes) == 0
        assert len(dashboard.edges) == 0
        assert len(dashboard.active_websockets) == 0
        assert isinstance(dashboard.session_color_manager, SessionColorManager)
        assert len(dashboard.event_filters) == 0
        assert len(dashboard.communication_history) == 0
        
        # Check performance limits
        assert dashboard.max_nodes == 500
        assert dashboard.max_edges == 2000
        assert dashboard.max_history == 10000
    
    @pytest.mark.asyncio
    async def test_websocket_registration(self, dashboard):
        """Test WebSocket registration and unregistration."""
        mock_websocket = Mock()
        mock_websocket.accept = AsyncMock()
        session_id = "test_session"
        
        # Mock the initial graph state sending
        with patch.object(dashboard, '_send_initial_graph_state') as mock_send:
            connection_id = await dashboard.register_websocket(
                mock_websocket, session_id
            )
        
        assert connection_id in dashboard.active_websockets
        assert dashboard.active_websockets[connection_id] == mock_websocket
        assert len(dashboard.active_websockets) == 1
        mock_websocket.accept.assert_called_once()
        mock_send.assert_called_once()
        
        # Test unregistration
        await dashboard.unregister_websocket(connection_id)
        assert connection_id not in dashboard.active_websockets
        assert len(dashboard.active_websockets) == 0
    
    @pytest.mark.asyncio
    async def test_agent_activation_processing(self, dashboard, sample_lifecycle_event):
        """Test processing agent activation events."""
        await dashboard.process_lifecycle_event(sample_lifecycle_event)
        
        # Check that agent node was created
        agent_node_id = f"agent_{sample_lifecycle_event.metadata['agent_id']}"
        assert agent_node_id in dashboard.nodes
        
        node = dashboard.nodes[agent_node_id]
        assert node.type == AgentNodeType.AGENT
        assert node.status == AgentNodeStatus.ACTIVE
        assert node.metadata["agent_id"] == sample_lifecycle_event.metadata["agent_id"]
        assert node.metadata["session_id"] == sample_lifecycle_event.metadata["session_id"]
        
        # Check communication event was recorded
        assert len(dashboard.communication_history) == 1
        comm_event = dashboard.communication_history[0]
        assert comm_event.source_agent_id == sample_lifecycle_event.metadata["agent_id"]
        assert comm_event.session_id == sample_lifecycle_event.metadata["session_id"]
    
    @pytest.mark.asyncio
    async def test_agent_sleep_processing(self, dashboard):
        """Test processing agent sleep events."""
        # First create an active agent
        agent_id = "test_agent_001"
        session_id = "test_session"
        
        activation_event = LifecycleEventData(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EnhancedEventType.AGENT_LIFECYCLE_START,
            timestamp=datetime.utcnow().isoformat(),
            payload={}
        )
        
        await dashboard.process_lifecycle_event(activation_event)
        
        # Now send sleep event
        sleep_event = LifecycleEventData(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EnhancedEventType.AGENT_LIFECYCLE_PAUSE,
            timestamp=datetime.utcnow().isoformat(),
            payload={
                "sleep_reason": "context_threshold",
                "context_usage": 85
            }
        )
        
        await dashboard.process_lifecycle_event(sleep_event)
        
        # Check agent status changed to sleeping
        agent_node_id = f"agent_{agent_id}"
        node = dashboard.nodes[agent_node_id]
        assert node.status == AgentNodeStatus.SLEEPING
        assert node.color == "#888888"  # Gray for sleeping
        assert node.metadata["sleep_reason"] == "context_threshold"
    
    @pytest.mark.asyncio
    async def test_tool_interaction_processing(self, dashboard):
        """Test processing tool interaction events."""
        agent_id = "test_agent_001"
        tool_id = "git_clone"
        session_id = "test_session"
        
        # Create agent first
        activation_event = LifecycleEventData(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EnhancedEventType.AGENT_LIFECYCLE_START,
            timestamp=datetime.utcnow().isoformat(),
            payload={}
        )
        
        await dashboard.process_lifecycle_event(activation_event)
        
        # Process tool interaction
        tool_event = LifecycleEventData(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EnhancedEventType.PRE_TOOL_USE,
            timestamp=datetime.utcnow().isoformat(),
            payload={
                "tool_id": tool_id,
                "tool_type": "version_control"
            }
        )
        
        await dashboard.process_lifecycle_event(tool_event)
        
        # Check tool node was created
        tool_node_id = f"tool_{tool_id}"
        assert tool_node_id in dashboard.nodes
        
        tool_node = dashboard.nodes[tool_node_id]
        assert tool_node.type == AgentNodeType.TOOL
        assert tool_node.color == "#FFA500"  # Orange for tools
        assert tool_node.shape == "square"
        assert tool_node.metadata["tool_id"] == tool_id
        
        # Check edge was created between agent and tool
        edge_id = f"agent_{agent_id}_{tool_node_id}"
        assert edge_id in dashboard.edges
        
        edge = dashboard.edges[edge_id]
        assert edge.type == AgentEdgeType.TOOL_CALL
        assert edge.source == f"agent_{agent_id}"
        assert edge.target == tool_node_id
        assert edge.weight == 1.0
    
    @pytest.mark.asyncio
    async def test_context_creation_processing(self, dashboard):
        """Test processing context creation events."""
        agent_id = "test_agent_001"
        context_id = "test_context_001"
        session_id = "test_session"
        
        # Create agent first
        activation_event = LifecycleEventData(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EnhancedEventType.AGENT_LIFECYCLE_START,
            timestamp=datetime.utcnow().isoformat(),
            payload={}
        )
        
        await dashboard.process_lifecycle_event(activation_event)
        
        # Process context creation
        context_event = LifecycleEventData(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EnhancedEventType.TASK_ASSIGNMENT,
            timestamp=datetime.utcnow().isoformat(),
            payload={
                "context_id": context_id,
                "context_type": "task"
            }
        )
        
        await dashboard.process_lifecycle_event(context_event)
        
        # Check context node was created
        context_node_id = f"context_{context_id}"
        assert context_node_id in dashboard.nodes
        
        context_node = dashboard.nodes[context_node_id]
        assert context_node.type == AgentNodeType.CONTEXT
        assert context_node.color == "#8A2BE2"  # Blue Violet for context
        assert context_node.shape == "diamond"
        assert context_node.metadata["context_id"] == context_id
        assert agent_id in context_node.metadata["sharing_agents"]
        
        # Check edge was created
        edge_id = f"agent_{agent_id}_{context_node_id}"
        assert edge_id in dashboard.edges
        
        edge = dashboard.edges[edge_id]
        assert edge.type == AgentEdgeType.CONTEXT_SHARE
        assert edge.style == "dashed"
    
    @pytest.mark.asyncio
    async def test_error_event_processing(self, dashboard):
        """Test processing error events."""
        agent_id = "test_agent_001"
        session_id = "test_session"
        
        # Create agent first
        activation_event = LifecycleEventData(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EnhancedEventType.AGENT_LIFECYCLE_START,
            timestamp=datetime.utcnow().isoformat(),
            payload={}
        )
        
        await dashboard.process_lifecycle_event(activation_event)
        
        # Process error event
        error_event = LifecycleEventData(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EnhancedEventType.ERROR_PATTERN_DETECTED,
            timestamp=datetime.utcnow().isoformat(),
            payload={
                "error": "Test error message"
            }
        )
        
        await dashboard.process_lifecycle_event(error_event)
        
        # Check agent status changed to error
        agent_node_id = f"agent_{agent_id}"
        node = dashboard.nodes[agent_node_id]
        assert node.status == AgentNodeStatus.ERROR
        assert node.color == "#FF4444"  # Red for errors
        assert node.metadata["error_message"] == "Test error message"
    
    @pytest.mark.asyncio
    async def test_graph_data_retrieval(self, dashboard):
        """Test getting graph data with filtering."""
        # Create test data
        agent_id = "test_agent_001"
        session_id = "test_session"
        
        activation_event = LifecycleEventData(
            session_id=session_id,
            agent_id=agent_id,
            event_type=EnhancedEventType.AGENT_LIFECYCLE_START,
            timestamp=datetime.utcnow().isoformat(),
            payload={}
        )
        
        await dashboard.process_lifecycle_event(activation_event)
        
        # Get all graph data
        graph_data = await dashboard.get_graph_data()
        
        assert "nodes" in graph_data
        assert "edges" in graph_data
        assert "stats" in graph_data
        assert "session_colors" in graph_data
        assert "timestamp" in graph_data
        
        assert len(graph_data["nodes"]) == 1  # One agent node
        assert len(graph_data["edges"]) == 0  # No edges yet
        
        stats = graph_data["stats"]
        assert stats["total_nodes"] == 1
        assert stats["total_edges"] == 0
        assert stats["active_agents"] == 1
        assert stats["tools_used"] == 0
        assert stats["contexts_shared"] == 0
    
    @pytest.mark.asyncio
    async def test_session_filtering(self, dashboard):
        """Test session-based filtering."""
        # Create agents in different sessions
        sessions = ["session_1", "session_2"]
        agent_ids = ["agent_001", "agent_002"]
        
        for i, (session, agent) in enumerate(zip(sessions, agent_ids)):
            event = LifecycleEventData(
                session_id=session,
                agent_id=agent,
                event_type=EnhancedEventType.AGENT_LIFECYCLE_START,
                timestamp=datetime.utcnow().isoformat(),
                payload={}
            )
            await dashboard.process_lifecycle_event(event)
        
        # Test filtering by specific session
        graph_data_session1 = await dashboard.get_graph_data("session_1")
        assert len(graph_data_session1["nodes"]) == 1
        assert graph_data_session1["nodes"][0]["metadata"]["session_id"] == "session_1"
        
        # Test getting all sessions
        graph_data_all = await dashboard.get_graph_data("all")
        assert len(graph_data_all["nodes"]) == 2
    
    @pytest.mark.asyncio
    async def test_session_transcript(self, dashboard):
        """Test session transcript generation."""
        session_id = "test_session"
        agent_id = "test_agent_001"
        
        # Add some communication events
        events = []
        for i in range(5):
            event = AgentCommunicationEvent(
                session_id=session_id,
                source_agent_id=agent_id,
                message_type=f"message_type_{i}",
                content={"message": f"Test message {i}"}
            )
            events.append(event)
        
        dashboard.communication_history.extend(events)
        
        # Get transcript
        transcript = await dashboard.get_session_transcript(session_id, limit=10)
        
        assert len(transcript) == 5
        assert all(event.session_id == session_id for event in transcript)
        assert all(event.source_agent_id == agent_id for event in transcript)
        
        # Test with agent filter
        transcript_filtered = await dashboard.get_session_transcript(
            session_id, agent_filter=[agent_id], limit=10
        )
        assert len(transcript_filtered) == 5
        
        # Test with non-matching agent filter
        transcript_empty = await dashboard.get_session_transcript(
            session_id, agent_filter=["non_existent_agent"], limit=10
        )
        assert len(transcript_empty) == 0
    
    @pytest.mark.asyncio
    async def test_websocket_broadcast(self, dashboard):
        """Test WebSocket broadcast functionality."""
        # Register mock WebSockets
        mock_ws1 = Mock()
        mock_ws1.send_text = AsyncMock()
        mock_ws2 = Mock()
        mock_ws2.send_text = AsyncMock()
        
        with patch.object(dashboard, '_send_initial_graph_state'):
            connection_id1 = await dashboard.register_websocket(mock_ws1, "session_1")
            connection_id2 = await dashboard.register_websocket(mock_ws2, "session_2")
        
        # Process an event that should trigger broadcast
        event = LifecycleEventData(
            session_id="session_1",
            agent_id="test_agent",
            event_type=EnhancedEventType.AGENT_LIFECYCLE_START,
            timestamp=datetime.utcnow().isoformat(),
            payload={}
        )
        
        await dashboard.process_lifecycle_event(event)
        
        # Check that both WebSockets received updates
        assert mock_ws1.send_text.called
        assert mock_ws2.send_text.called
        
        # Check message format
        call_args = mock_ws1.send_text.call_args[0][0]
        message = json.loads(call_args)
        assert message["type"] == "graph_update"
        assert "data" in message
        assert "event" in message["data"]
    
    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, dashboard):
        """Test cleanup of old data."""
        # Create old data
        old_time = datetime.utcnow() - timedelta(hours=25)  # 25 hours old
        
        # Add old node
        old_node = AgentGraphNode(
            id="old_agent",
            label="Old Agent",
            type=AgentNodeType.AGENT,
            status=AgentNodeStatus.ACTIVE,
            position={"x": 0, "y": 0},
            metadata={"session_id": "old_session"},
            created_at=old_time,
            last_updated=old_time
        )
        dashboard.nodes["old_agent"] = old_node
        
        # Add recent node
        recent_node = AgentGraphNode(
            id="recent_agent",
            label="Recent Agent",
            type=AgentNodeType.AGENT,
            status=AgentNodeStatus.ACTIVE,
            position={"x": 0, "y": 0},
            metadata={"session_id": "recent_session"}
        )
        dashboard.nodes["recent_agent"] = recent_node
        
        # Add old communication event
        old_event = AgentCommunicationEvent(
            session_id="old_session",
            source_agent_id="old_agent",
            message_type="test",
            content={},
            timestamp=old_time
        )
        dashboard.communication_history.append(old_event)
        
        # Add recent communication event
        recent_event = AgentCommunicationEvent(
            session_id="recent_session",
            source_agent_id="recent_agent",
            message_type="test",
            content={}
        )
        dashboard.communication_history.append(recent_event)
        
        # Perform cleanup (24 hour threshold)
        await dashboard.cleanup_old_data(max_age_hours=24)
        
        # Check that old data was removed
        assert "old_agent" not in dashboard.nodes
        assert "recent_agent" in dashboard.nodes
        assert len(dashboard.communication_history) == 1
        assert dashboard.communication_history[0].session_id == "recent_session"


class TestIntegrationAndPerformance:
    """Test integration scenarios and performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_high_volume_event_processing(self):
        """Test processing many events quickly."""
        dashboard = CoordinationDashboard()
        
        # Generate many events
        events = []
        for i in range(100):
            event = LifecycleEventData(
                session_id=f"session_{i % 10}",  # 10 sessions
                agent_id=f"agent_{i:03d}",
                event_type=EnhancedEventType.AGENT_LIFECYCLE_START,
                timestamp=datetime.utcnow().isoformat(),
                payload={
                    "agent_type": "test"
                }
            )
            events.append(event)
        
        # Process events
        start_time = datetime.utcnow()
        
        for event in events:
            await dashboard.process_lifecycle_event(event)
        
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        # Should process 100 events quickly
        assert processing_time < 5.0  # Less than 5 seconds
        assert len(dashboard.nodes) == 100
        assert len(dashboard.communication_history) == 100
    
    @pytest.mark.asyncio
    async def test_concurrent_websocket_connections(self):
        """Test multiple concurrent WebSocket connections."""
        dashboard = CoordinationDashboard()
        
        # Create multiple mock WebSockets
        websockets = []
        connection_ids = []
        
        for i in range(10):
            mock_ws = Mock()
            mock_ws.accept = AsyncMock()
            websockets.append(mock_ws)
            
            with patch.object(dashboard, '_send_initial_graph_state'):
                connection_id = await dashboard.register_websocket(
                    mock_ws, f"session_{i}"
                )
                connection_ids.append(connection_id)
        
        assert len(dashboard.active_websockets) == 10
        assert len(connection_ids) == 10
        assert len(set(connection_ids)) == 10  # All unique
        
        # Process event and check all get notified
        event = LifecycleEventData(
            session_id="session_0",
            agent_id="test_agent",
            event_type=EnhancedEventType.AGENT_LIFECYCLE_START,
            timestamp=datetime.utcnow().isoformat(),
            payload={}
        )
        
        with patch.object(dashboard, '_broadcast_graph_update') as mock_broadcast:
            await dashboard.process_lifecycle_event(event)
            mock_broadcast.assert_called_once_with(event)
        
        # Clean up connections
        for connection_id in connection_ids:
            await dashboard.unregister_websocket(connection_id)
        
        assert len(dashboard.active_websockets) == 0
    
    @pytest.mark.asyncio
    async def test_memory_bounded_storage(self):
        """Test that storage is bounded to prevent memory leaks."""
        dashboard = CoordinationDashboard()
        
        # Fill up communication history beyond limit
        for i in range(dashboard.max_history + 500):
            event = AgentCommunicationEvent(
                session_id="test_session",
                source_agent_id=f"agent_{i}",
                message_type="test",
                content={"index": i}
            )
            await dashboard._add_communication_event(event)
        
        # Should be trimmed to half of max_history
        expected_size = dashboard.max_history // 2
        assert len(dashboard.communication_history) == expected_size
        
        # Should keep the most recent events
        last_event = dashboard.communication_history[-1]
        assert last_event.content["index"] >= dashboard.max_history  # Recent event
    
    def test_position_calculation_distribution(self):
        """Test node position calculation creates good distribution."""
        dashboard = CoordinationDashboard()
        
        # Calculate positions for many nodes of same type
        positions = []
        for i in range(20):
            pos = asyncio.run(dashboard._calculate_node_position(
                f"agent_{i}", AgentNodeType.AGENT
            ))
            positions.append(pos)
        
        # Check that positions are distributed (not all the same)
        x_coords = [pos["x"] for pos in positions]
        y_coords = [pos["y"] for pos in positions]
        
        assert len(set(x_coords)) > 10  # Good distribution
        assert len(set(y_coords)) > 10  # Good distribution
        
        # Check positions are within reasonable bounds
        for pos in positions:
            assert -500 <= pos["x"] <= 500
            assert -500 <= pos["y"] <= 500