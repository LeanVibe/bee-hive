"""
Comprehensive Test Suite for Transcript Analysis System

Tests for chat transcript analysis, conversation debugging, search capabilities,
and dashboard integration with comprehensive coverage and edge cases.
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from app.core.chat_transcript_manager import (
    ChatTranscriptManager, ConversationEvent, ConversationEventType,
    ConversationMetrics, SearchFilter
)
from app.core.communication_analyzer import (
    CommunicationAnalyzer, AnalysisType, AlertSeverity, CommunicationInsight
)
from app.core.conversation_search_engine import (
    ConversationSearchEngine, SearchQuery, SearchType, SearchResults
)
from app.core.transcript_streaming import (
    TranscriptStreamingManager, StreamingFilter, FilterMode
)
from app.core.conversation_debugging_tools import (
    ConversationDebugger, ReplayMode, DebugLevel, AnalysisScope
)
from app.core.dashboard_integration import DashboardIntegrationManager


class TestChatTranscriptManager:
    """Test suite for ChatTranscriptManager."""
    
    @pytest.fixture
    async def transcript_manager(self):
        """Create transcript manager for testing."""
        db_session = AsyncMock()
        embedding_service = AsyncMock()
        embedding_service.get_embedding.return_value = [0.1] * 1536
        
        manager = ChatTranscriptManager(db_session, embedding_service)
        return manager
    
    @pytest.fixture
    def sample_conversation_event(self):
        """Create sample conversation event for testing."""
        return ConversationEvent(
            id=str(uuid.uuid4()),
            session_id="test_session_123",
            timestamp=datetime.utcnow(),
            event_type=ConversationEventType.MESSAGE_SENT,
            source_agent_id="agent_1",
            target_agent_id="agent_2",
            message_content="Test message content",
            metadata={"test": True},
            response_time_ms=1500.0,
            context_references=["context_1"],
            tool_calls=[{"name": "test_tool", "args": {}}]
        )
    
    @pytest.mark.asyncio
    async def test_track_conversation_event(self, transcript_manager, sample_conversation_event):
        """Test tracking a conversation event."""
        # Mock database operations
        transcript_manager._persist_conversation_event = AsyncMock()
        transcript_manager._update_conversation_thread = AsyncMock()
        transcript_manager._detect_conversation_patterns = AsyncMock()
        
        # Track event
        result = await transcript_manager.track_conversation_event(
            session_id=sample_conversation_event.session_id,
            event_type=sample_conversation_event.event_type,
            source_agent_id=sample_conversation_event.source_agent_id,
            target_agent_id=sample_conversation_event.target_agent_id,
            message_content=sample_conversation_event.message_content,
            metadata=sample_conversation_event.metadata,
            response_time_ms=sample_conversation_event.response_time_ms
        )
        
        # Verify event was created and processed
        assert result.session_id == sample_conversation_event.session_id
        assert result.event_type == sample_conversation_event.event_type
        assert result.source_agent_id == sample_conversation_event.source_agent_id
        assert result.target_agent_id == sample_conversation_event.target_agent_id
        assert result.message_content == sample_conversation_event.message_content
        
        # Verify processing methods were called
        transcript_manager._persist_conversation_event.assert_called_once()
        transcript_manager._update_conversation_thread.assert_called_once()
        transcript_manager._detect_conversation_patterns.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_conversation_transcript_empty(self, transcript_manager):
        """Test getting transcript when no events exist."""
        # Mock empty database result
        with patch('app.core.chat_transcript_manager.select') as mock_select:
            mock_result = AsyncMock()
            mock_result.scalars.return_value.all.return_value = []
            transcript_manager.db_session.execute.return_value = mock_result
            
            # Get transcript
            events = await transcript_manager.get_conversation_transcript("empty_session")
            
            # Verify empty result
            assert events == []
    
    @pytest.mark.asyncio 
    async def test_get_conversation_analytics(self, transcript_manager):
        """Test conversation analytics calculation."""
        # Mock transcript data
        mock_events = [
            ConversationEvent(
                id=str(uuid.uuid4()),
                session_id="test_session",
                timestamp=datetime.utcnow() - timedelta(hours=1),
                event_type=ConversationEventType.MESSAGE_SENT,
                source_agent_id="agent_1",
                target_agent_id="agent_2",
                message_content="Test message 1",
                metadata={},
                response_time_ms=1000.0
            ),
            ConversationEvent(
                id=str(uuid.uuid4()),
                session_id="test_session",
                timestamp=datetime.utcnow(),
                event_type=ConversationEventType.ERROR_OCCURRED,
                source_agent_id="agent_2",
                target_agent_id=None,
                message_content="Error occurred",
                metadata={},
                response_time_ms=5000.0
            )
        ]
        
        transcript_manager.get_conversation_transcript = AsyncMock(return_value=mock_events)
        transcript_manager._calculate_conversation_metrics = AsyncMock()
        transcript_manager._analyze_conversation_patterns = AsyncMock(return_value=[])
        transcript_manager._analyze_performance_metrics = AsyncMock(return_value={})
        transcript_manager._analyze_agent_participation = AsyncMock(return_value={})
        
        # Get analytics
        analytics = await transcript_manager.get_conversation_analytics("test_session")
        
        # Verify analytics structure
        assert "session_id" in analytics
        assert "time_window_hours" in analytics
        assert "total_events" in analytics
        assert "metrics" in analytics
        assert "patterns" in analytics
        assert "performance" in analytics
        assert "participation" in analytics
        assert analytics["total_events"] == 2
    
    @pytest.mark.asyncio
    async def test_replay_conversation(self, transcript_manager):
        """Test conversation replay functionality."""
        # Mock transcript data
        mock_events = [
            ConversationEvent(
                id=str(uuid.uuid4()),
                session_id="original_session",
                timestamp=datetime.utcnow() - timedelta(seconds=10),
                event_type=ConversationEventType.MESSAGE_SENT,
                source_agent_id="agent_1",
                target_agent_id="agent_2",
                message_content="First message",
                metadata={}
            ),
            ConversationEvent(
                id=str(uuid.uuid4()),
                session_id="original_session",
                timestamp=datetime.utcnow(),
                event_type=ConversationEventType.MESSAGE_RECEIVED,
                source_agent_id="agent_2",
                target_agent_id="agent_1",
                message_content="Response message",
                metadata={}
            )
        ]
        
        transcript_manager.get_conversation_transcript = AsyncMock(return_value=mock_events)
        transcript_manager._persist_conversation_event = AsyncMock()
        
        # Execute replay
        result = await transcript_manager.replay_conversation(
            session_id="original_session",
            target_session_id="replay_session",
            speed_multiplier=10.0  # Fast replay for testing
        )
        
        # Verify replay result
        assert result["status"] == "completed"
        assert result["target_session_id"] == "replay_session"
        assert result["events_replayed"] == 2
        assert result["original_session_id"] == "original_session"
        
        # Verify replay events were persisted
        assert transcript_manager._persist_conversation_event.call_count == 2


class TestCommunicationAnalyzer:
    """Test suite for CommunicationAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create communication analyzer for testing."""
        return CommunicationAnalyzer()
    
    @pytest.fixture
    def sample_events(self):
        """Create sample conversation events for analysis."""
        base_time = datetime.utcnow()
        return [
            ConversationEvent(
                id=str(uuid.uuid4()),
                session_id="test_session",
                timestamp=base_time,
                event_type=ConversationEventType.MESSAGE_SENT,
                source_agent_id="agent_1",
                target_agent_id="agent_2",
                message_content="Normal message",
                metadata={},
                response_time_ms=1000.0
            ),
            ConversationEvent(
                id=str(uuid.uuid4()),
                session_id="test_session",
                timestamp=base_time + timedelta(seconds=5),
                event_type=ConversationEventType.ERROR_OCCURRED,
                source_agent_id="agent_2",
                target_agent_id=None,
                message_content="Error: Connection timeout",
                metadata={},
                response_time_ms=8000.0
            ),
            ConversationEvent(
                id=str(uuid.uuid4()),
                session_id="test_session",
                timestamp=base_time + timedelta(seconds=10),
                event_type=ConversationEventType.TOOL_INVOCATION,
                source_agent_id="agent_1",
                target_agent_id=None,
                message_content="Tool call executed",
                metadata={},
                tool_calls=[{"name": "test_tool"}]
            )
        ]
    
    @pytest.mark.asyncio
    async def test_analyze_conversation_events_all_types(self, analyzer, sample_events):
        """Test comprehensive analysis of conversation events."""
        # Mock internal methods
        analyzer._update_agent_profiles = AsyncMock()
        analyzer._update_communication_flows = AsyncMock()
        analyzer._detect_communication_patterns = AsyncMock(return_value=[])
        analyzer._analyze_performance = AsyncMock(return_value=[])
        analyzer._analyze_agent_behavior = AsyncMock(return_value=[])
        analyzer._detect_bottlenecks = AsyncMock(return_value=[])
        analyzer._analyze_errors = AsyncMock(return_value=[
            CommunicationInsight(
                id="error_1",
                analysis_type=AnalysisType.ERROR_ANALYSIS,
                severity=AlertSeverity.ERROR,
                title="Connection Timeout Detected",
                description="Agent experienced connection timeout",
                affected_agents=["agent_2"],
                recommendations=["Check network connectivity"],
                metrics={"error_count": 1},
                timestamp=datetime.utcnow(),
                session_id="test_session"
            )
        ])
        analyzer._optimize_communication_flow = AsyncMock(return_value=[])
        analyzer._detect_anomalies = AsyncMock(return_value=[])
        
        # Analyze events
        insights = await analyzer.analyze_conversation_events(sample_events)
        
        # Verify analysis was performed
        assert len(insights) == 1
        assert insights[0].analysis_type == AnalysisType.ERROR_ANALYSIS
        assert insights[0].severity == AlertSeverity.ERROR
        assert "agent_2" in insights[0].affected_agents
        
        # Verify all analysis methods were called
        analyzer._update_agent_profiles.assert_called_once()
        analyzer._update_communication_flows.assert_called_once()
        analyzer._analyze_errors.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_detect_real_time_issues(self, analyzer, sample_events):
        """Test real-time issue detection."""
        # Add more error events to trigger alerts
        error_time = datetime.utcnow()
        error_events = [
            ConversationEvent(
                id=str(uuid.uuid4()),
                session_id="test_session",
                timestamp=error_time,
                event_type=ConversationEventType.ERROR_OCCURRED,
                source_agent_id="agent_1",
                target_agent_id=None,
                message_content="Error 1",
                metadata={}
            ),
            ConversationEvent(
                id=str(uuid.uuid4()),
                session_id="test_session",
                timestamp=error_time + timedelta(seconds=1),
                event_type=ConversationEventType.ERROR_OCCURRED,
                source_agent_id="agent_1",
                target_agent_id=None,
                message_content="Error 2",
                metadata={}
            ),
            ConversationEvent(
                id=str(uuid.uuid4()),
                session_id="test_session",
                timestamp=error_time + timedelta(seconds=2),
                event_type=ConversationEventType.MESSAGE_SENT,
                source_agent_id="agent_2",
                target_agent_id="agent_1",
                message_content="Normal message",
                metadata={}
            )
        ]
        
        # Detect issues
        issues = await analyzer.detect_real_time_issues(error_events, time_window_minutes=5)
        
        # Verify error spike was detected
        assert len(issues) > 0
        error_spike_alert = next((issue for issue in issues if "Error Spike" in issue.title), None)
        assert error_spike_alert is not None
        assert error_spike_alert.severity == AlertSeverity.CRITICAL
    
    @pytest.mark.asyncio
    async def test_generate_optimization_report(self, analyzer):
        """Test optimization report generation."""
        # Mock agent profiles and flows
        analyzer.agent_profiles = {
            "agent_1": Mock(
                session_id="test_session",
                message_frequency=10.0,
                average_response_time=2000.0,
                error_rate=0.1
            ),
            "agent_2": Mock(
                session_id="test_session", 
                message_frequency=5.0,
                average_response_time=1000.0,
                error_rate=0.02
            )
        }
        
        analyzer.communication_flows = {
            ("agent_1", "agent_2"): Mock(
                source_agent="agent_1",
                target_agent="agent_2",
                message_count=100,
                flow_efficiency=0.5,  # Low efficiency to trigger bottleneck
                average_response_time=3000.0
            )
        }
        
        # Generate report
        report = await analyzer.generate_optimization_report("test_session", 24)
        
        # Verify report structure
        assert "session_id" in report
        assert "summary" in report
        assert "optimization_opportunities" in report
        assert "implementation_roadmap" in report
        
        # Verify bottleneck was detected
        opportunities = report["optimization_opportunities"]
        bottleneck_opp = next((opp for opp in opportunities if opp["type"] == "bottleneck_resolution"), None)
        assert bottleneck_opp is not None
        assert bottleneck_opp["priority"] == "high"


class TestConversationSearchEngine:
    """Test suite for ConversationSearchEngine."""
    
    @pytest.fixture
    async def search_engine(self):
        """Create search engine for testing."""
        db_session = AsyncMock()
        embedding_service = AsyncMock()
        context_engine = AsyncMock()
        
        engine = ConversationSearchEngine(db_session, embedding_service, context_engine)
        return engine
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, search_engine):
        """Test semantic search functionality."""
        # Mock embedding service
        search_engine.embedding_service.get_embedding = AsyncMock(return_value=[0.1] * 1536)
        
        # Mock database query result
        mock_conversation = Mock()
        mock_conversation.id = uuid.uuid4()
        mock_conversation.session_id = uuid.uuid4()
        mock_conversation.from_agent_id = uuid.uuid4()
        mock_conversation.to_agent_id = uuid.uuid4()
        mock_conversation.message_type = Mock()
        mock_conversation.content = "Test conversation content"
        mock_conversation.conversation_metadata = {}
        mock_conversation.context_refs = []
        mock_conversation.embedding = [0.1] * 1536
        mock_conversation.created_at = datetime.utcnow()
        
        mock_result = AsyncMock()
        mock_result.fetchall.return_value = [(mock_conversation, 0.85)]
        search_engine.db_session.execute.return_value = mock_result
        
        # Mock conversation to event conversion
        search_engine._conversation_to_event = AsyncMock(
            return_value=ConversationEvent(
                id=str(mock_conversation.id),
                session_id=str(mock_conversation.session_id),
                timestamp=mock_conversation.created_at,
                event_type=ConversationEventType.MESSAGE_SENT,
                source_agent_id=str(mock_conversation.from_agent_id),
                target_agent_id=str(mock_conversation.to_agent_id),
                message_content=mock_conversation.content,
                metadata={}
            )
        )
        search_engine._highlight_content = AsyncMock(return_value="<mark>Test</mark> conversation content")
        search_engine._apply_base_filters = AsyncMock(side_effect=lambda q, f: q)
        
        # Create search query
        query = SearchQuery(
            query_text="test conversation",
            search_type=SearchType.SEMANTIC,
            semantic_threshold=0.7,
            limit=10
        )
        
        # Execute search
        results = await search_engine._semantic_search(query)
        
        # Verify results
        assert len(results.results) == 1
        assert results.results[0].relevance_score == 0.85
        assert results.results[0].search_type == SearchType.SEMANTIC
        assert "Semantic similarity" in results.results[0].match_reasons[0]
    
    @pytest.mark.asyncio
    async def test_keyword_search(self, search_engine):
        """Test keyword search functionality."""
        # Mock query processor
        search_engine.query_processor.parse_query = Mock(return_value={
            'terms': ['test', 'keyword'],
            'phrases': [],
            'operators': [],
            'filters': {}
        })
        
        # Mock database result
        mock_conversation = Mock()
        mock_conversation.id = uuid.uuid4()
        mock_conversation.session_id = uuid.uuid4()
        mock_conversation.from_agent_id = uuid.uuid4()
        mock_conversation.to_agent_id = None
        mock_conversation.message_type = Mock()
        mock_conversation.content = "This is a test message with keyword"
        mock_conversation.conversation_metadata = {}
        mock_conversation.context_refs = []
        mock_conversation.created_at = datetime.utcnow()
        
        mock_result = AsyncMock()
        mock_result.scalars.return_value.all.return_value = [mock_conversation]
        search_engine.db_session.execute.return_value = mock_result
        
        # Mock helper methods
        search_engine._conversation_to_event = AsyncMock(
            return_value=ConversationEvent(
                id=str(mock_conversation.id),
                session_id=str(mock_conversation.session_id),
                timestamp=mock_conversation.created_at,
                event_type=ConversationEventType.MESSAGE_SENT,
                source_agent_id=str(mock_conversation.from_agent_id),
                target_agent_id=None,
                message_content=mock_conversation.content,
                metadata={}
            )
        )
        search_engine._highlight_content = AsyncMock(return_value="Highlighted content")
        search_engine._apply_base_filters = AsyncMock(side_effect=lambda q, f: q)
        
        # Create search query
        query = SearchQuery(
            query_text="test keyword",
            search_type=SearchType.KEYWORD,
            limit=10
        )
        
        # Execute search
        results = await search_engine._keyword_search(query)
        
        # Verify results
        assert len(results.results) == 1
        assert results.results[0].search_type == SearchType.KEYWORD
        assert len(results.results[0].match_reasons) > 0
    
    @pytest.mark.asyncio
    async def test_suggest_queries(self, search_engine):
        """Test query suggestion functionality."""
        # Mock popular queries
        search_engine.search_metrics['popular_queries'] = {
            'error analysis': 10,
            'performance issues': 8,
            'agent communication': 5
        }
        
        # Test partial query matching
        suggestions = await search_engine.suggest_queries("error", limit=5)
        
        # Verify suggestions
        assert len(suggestions) > 0
        assert any('error' in suggestion['query'].lower() for suggestion in suggestions)
        
        # Test empty query (should return popular queries)
        popular_suggestions = await search_engine.suggest_queries("", limit=3)
        assert len(popular_suggestions) <= 3
        assert all(suggestion['type'] == 'popular' for suggestion in popular_suggestions)


class TestTranscriptStreamingManager:
    """Test suite for TranscriptStreamingManager."""
    
    @pytest.fixture
    def streaming_manager(self):
        """Create streaming manager for testing."""
        transcript_manager = AsyncMock()
        analyzer = AsyncMock()
        return TranscriptStreamingManager(transcript_manager, analyzer)
    
    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket for testing."""
        websocket = AsyncMock()
        websocket.accept = AsyncMock()
        websocket.send_text = AsyncMock()
        websocket.receive_text = AsyncMock()
        return websocket
    
    @pytest.mark.asyncio
    async def test_register_streaming_session(self, streaming_manager, mock_websocket):
        """Test WebSocket session registration."""
        from app.core.transcript_streaming import StreamingFilter, FilterMode, StreamEventType
        from app.core.chat_transcript_manager import ConversationEventType
        from app.core.communication_analyzer import AlertSeverity
        
        # Create streaming filter
        streaming_filter = StreamingFilter(
            session_ids=["test_session"],
            agent_ids=["agent_1"],
            event_types=[ConversationEventType.MESSAGE_SENT],
            stream_events=[StreamEventType.CONVERSATION_EVENT],
            keywords=["test"],
            min_severity=AlertSeverity.INFO,
            real_time_only=True,
            include_patterns=True,
            include_performance=True,
            filter_mode=FilterMode.INCLUSIVE
        )
        
        # Register session
        session_id = await streaming_manager.register_streaming_session(
            websocket=mock_websocket,
            session_filter=streaming_filter
        )
        
        # Verify session was registered
        assert session_id in streaming_manager.active_sessions
        assert streaming_manager.active_sessions[session_id].websocket == mock_websocket
        assert streaming_manager.active_sessions[session_id].filter_config == streaming_filter
        
        # Verify WebSocket was accepted
        mock_websocket.accept.assert_called_once()
        
        # Verify welcome message was sent
        mock_websocket.send_text.assert_called()
    
    @pytest.mark.asyncio
    async def test_stream_conversation_event(self, streaming_manager, mock_websocket):
        """Test streaming conversation events to subscribed sessions."""
        from app.core.transcript_streaming import StreamingFilter, FilterMode, StreamEventType
        from app.core.chat_transcript_manager import ConversationEventType
        from app.core.communication_analyzer import AlertSeverity
        
        # Register session with matching filter
        streaming_filter = StreamingFilter(
            session_ids=["test_session"],
            agent_ids=[],
            event_types=[ConversationEventType.MESSAGE_SENT],
            stream_events=[StreamEventType.CONVERSATION_EVENT],
            keywords=[],
            min_severity=AlertSeverity.INFO,
            real_time_only=True,
            include_patterns=False,
            include_performance=False,
            filter_mode=FilterMode.INCLUSIVE
        )
        
        session_id = await streaming_manager.register_streaming_session(
            websocket=mock_websocket,
            session_filter=streaming_filter
        )
        
        # Create conversation event
        event = ConversationEvent(
            id=str(uuid.uuid4()),
            session_id="test_session",
            timestamp=datetime.utcnow(),
            event_type=ConversationEventType.MESSAGE_SENT,
            source_agent_id="agent_1",
            target_agent_id="agent_2",
            message_content="Test message",
            metadata={}
        )
        
        # Stream event
        await streaming_manager.stream_conversation_event(event, real_time=True)
        
        # Verify event was sent to WebSocket (more than just welcome message)
        assert mock_websocket.send_text.call_count > 1
    
    @pytest.mark.asyncio
    async def test_unregister_streaming_session(self, streaming_manager, mock_websocket):
        """Test WebSocket session cleanup."""
        from app.core.transcript_streaming import StreamingFilter, FilterMode, StreamEventType
        from app.core.communication_analyzer import AlertSeverity
        
        # Register session
        streaming_filter = StreamingFilter(
            session_ids=[],
            agent_ids=[],
            event_types=[],
            stream_events=[StreamEventType.CONVERSATION_EVENT],
            keywords=[],
            min_severity=AlertSeverity.INFO,
            real_time_only=True,
            include_patterns=True,
            include_performance=True,
            filter_mode=FilterMode.INCLUSIVE
        )
        
        session_id = await streaming_manager.register_streaming_session(
            websocket=mock_websocket,
            session_filter=streaming_filter
        )
        
        # Verify session exists
        assert session_id in streaming_manager.active_sessions
        
        # Unregister session
        await streaming_manager.unregister_streaming_session(session_id)
        
        # Verify cleanup
        assert session_id not in streaming_manager.active_sessions
        assert session_id not in streaming_manager.session_filters


class TestConversationDebugger:
    """Test suite for ConversationDebugger."""
    
    @pytest.fixture
    def debugger(self):
        """Create conversation debugger for testing."""
        transcript_manager = AsyncMock()
        analyzer = AsyncMock()
        streaming_manager = AsyncMock()
        return ConversationDebugger(transcript_manager, analyzer, streaming_manager)
    
    @pytest.fixture
    def sample_debug_events(self):
        """Create sample events for debugging."""
        base_time = datetime.utcnow()
        return [
            ConversationEvent(
                id=str(uuid.uuid4()),
                session_id="debug_session",
                timestamp=base_time,
                event_type=ConversationEventType.MESSAGE_SENT,
                source_agent_id="agent_1",
                target_agent_id="agent_2",
                message_content="Debug message 1",
                metadata={}
            ),
            ConversationEvent(
                id=str(uuid.uuid4()),
                session_id="debug_session",
                timestamp=base_time + timedelta(seconds=5),
                event_type=ConversationEventType.ERROR_OCCURRED,
                source_agent_id="agent_2",
                target_agent_id=None,
                message_content="Debug error occurred",
                metadata={}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_create_debug_session(self, debugger, sample_debug_events):
        """Test debug session creation."""
        # Mock transcript manager
        debugger.transcript_manager.get_conversation_transcript = AsyncMock(
            return_value=sample_debug_events
        )
        debugger._log_debug = AsyncMock()
        debugger._perform_initial_analysis = AsyncMock()
        
        # Create debug session
        debug_session_id = await debugger.create_debug_session(
            target_session_id="debug_session",
            replay_mode=ReplayMode.STEP_BY_STEP,
            debug_level=DebugLevel.INFO,
            analysis_scope=AnalysisScope.SESSION_WIDE,
            auto_analyze=True
        )
        
        # Verify session was created
        assert debug_session_id in debugger.active_sessions
        assert debug_session_id in debugger.session_events
        assert len(debugger.session_events[debug_session_id]) == 2
        
        # Verify session properties
        session = debugger.active_sessions[debug_session_id]
        assert session.target_conversation_session == "debug_session"
        assert session.replay_mode == ReplayMode.STEP_BY_STEP
        assert session.debug_level == DebugLevel.INFO
        assert session.total_events == 2
        assert session.current_event_index == 0
        
        # Verify initial analysis was performed
        debugger._perform_initial_analysis.assert_called_once_with(debug_session_id)
    
    @pytest.mark.asyncio
    async def test_add_breakpoint(self, debugger, sample_debug_events):
        """Test adding debug breakpoints."""
        # Create debug session
        debugger.transcript_manager.get_conversation_transcript = AsyncMock(
            return_value=sample_debug_events
        )
        debugger._log_debug = AsyncMock()
        debugger._perform_initial_analysis = AsyncMock()
        
        debug_session_id = await debugger.create_debug_session("debug_session")
        
        # Add breakpoint
        breakpoint_id = await debugger.add_breakpoint(
            debug_session_id=debug_session_id,
            condition_type="event_type",
            condition_value=ConversationEventType.ERROR_OCCURRED,
            action="pause"
        )
        
        # Verify breakpoint was added
        session = debugger.active_sessions[debug_session_id]
        assert len(session.breakpoints) == 1
        assert session.breakpoints[0].breakpoint_id == breakpoint_id
        assert session.breakpoints[0].condition_type == "event_type"
        assert session.breakpoints[0].condition_value == ConversationEventType.ERROR_OCCURRED
    
    @pytest.mark.asyncio
    async def test_step_debug_session(self, debugger, sample_debug_events):
        """Test stepping through debug session."""
        # Create debug session
        debugger.transcript_manager.get_conversation_transcript = AsyncMock(
            return_value=sample_debug_events
        )
        debugger._log_debug = AsyncMock()
        debugger._perform_initial_analysis = AsyncMock()
        debugger._check_breakpoints = AsyncMock(return_value=None)
        debugger._analyze_debug_event = AsyncMock(return_value={"analysis": "test"})
        
        debug_session_id = await debugger.create_debug_session("debug_session")
        
        # Step through one event
        result = await debugger.step_debug_session(debug_session_id, steps=1)
        
        # Verify step result
        assert result["debug_session_id"] == debug_session_id
        assert result["steps_executed"] == 1
        assert len(result["step_results"]) == 1
        assert result["step_results"][0]["step_index"] == 0
        
        # Verify session state was updated
        session = debugger.active_sessions[debug_session_id]
        assert session.current_event_index == 1
    
    @pytest.mark.asyncio
    async def test_analyze_error_patterns(self, debugger, sample_debug_events):
        """Test error pattern analysis in debug session."""
        # Create debug session with error events
        error_events = [
            ConversationEvent(
                id=str(uuid.uuid4()),
                session_id="debug_session",
                timestamp=datetime.utcnow(),
                event_type=ConversationEventType.ERROR_OCCURRED,
                source_agent_id="agent_1",
                target_agent_id=None,
                message_content="Connection timeout error",
                metadata={"error_type": "timeout"}
            )
        ]
        
        debugger.transcript_manager.get_conversation_transcript = AsyncMock(
            return_value=error_events
        )
        debugger._log_debug = AsyncMock()
        debugger._perform_initial_analysis = AsyncMock()
        debugger._find_error_cascade = AsyncMock(return_value=[])
        debugger._determine_root_cause = AsyncMock(return_value="Connection timeout")
        debugger._generate_resolution_steps = AsyncMock(return_value=["Check network"])
        debugger._find_similar_errors = AsyncMock(return_value=[])
        debugger._generate_prevention_recommendations = AsyncMock(return_value=["Add retry logic"])
        
        debug_session_id = await debugger.create_debug_session("debug_session")
        
        # Analyze error patterns
        error_analyses = await debugger.analyze_error_patterns(debug_session_id)
        
        # Verify analysis results
        assert len(error_analyses) == 1
        assert error_analyses[0].root_cause == "Connection timeout"
        assert error_analyses[0].affected_agents == ["agent_1"]
        assert "Check network" in error_analyses[0].resolution_steps
        assert "Add retry logic" in error_analyses[0].prevention_recommendations


class TestDashboardIntegration:
    """Test suite for DashboardIntegrationManager."""
    
    @pytest.fixture
    def integration_manager(self):
        """Create dashboard integration manager for testing."""
        transcript_manager = AsyncMock()
        analyzer = AsyncMock()
        streaming_manager = AsyncMock()
        debugger = AsyncMock()
        coordination_dashboard = AsyncMock()
        
        return DashboardIntegrationManager(
            transcript_manager, analyzer, streaming_manager, 
            debugger, coordination_dashboard
        )
    
    @pytest.mark.asyncio
    async def test_visualize_conversation_thread(self, integration_manager):
        """Test conversation thread visualization."""
        # Mock conversation events
        sample_events = [
            ConversationEvent(
                id=str(uuid.uuid4()),
                session_id="viz_session",
                timestamp=datetime.utcnow(),
                event_type=ConversationEventType.MESSAGE_SENT,
                source_agent_id="agent_1",
                target_agent_id="agent_2",
                message_content="Visualization test message",
                metadata={},
                response_time_ms=1500.0,
                tool_calls=[],
                context_references=[]
            )
        ]
        
        # Mock dependencies
        integration_manager.transcript_manager.get_conversation_transcript = AsyncMock(
            return_value=sample_events
        )
        integration_manager.analyzer.analyze_conversation_events = AsyncMock(return_value=[])
        integration_manager._build_timeline_data = AsyncMock(return_value=[])
        integration_manager._build_pattern_highlights = AsyncMock(return_value=[])
        integration_manager._calculate_performance_indicators = AsyncMock(return_value={})
        integration_manager._send_dashboard_update = AsyncMock()
        
        # Create visualization
        visualization = await integration_manager.visualize_conversation_thread(
            session_id="viz_session",
            thread_participants=["agent_1", "agent_2"],
            highlight_patterns=True
        )
        
        # Verify visualization was created
        assert visualization.participating_agents == ["agent_1", "agent_2"]
        assert len(visualization.message_flow) == 1
        assert visualization.message_flow[0]["source_agent"] == "agent_1"
        assert visualization.message_flow[0]["target_agent"] == "agent_2"
        
        # Verify dashboard update was sent
        integration_manager._send_dashboard_update.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_integration_lifecycle(self, integration_manager):
        """Test integration manager lifecycle."""
        # Start integration
        await integration_manager.start_integration()
        assert integration_manager.integration_active == True
        assert len(integration_manager.background_tasks) > 0
        
        # Stop integration
        await integration_manager.stop_integration()
        assert integration_manager.integration_active == False
        assert len(integration_manager.background_tasks) == 0


class TestIntegrationScenarios:
    """Test integration scenarios across multiple components."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_conversation_analysis(self):
        """Test complete conversation analysis workflow."""
        # This would test the complete workflow from event tracking
        # through analysis, search, and dashboard integration
        
        # Mock all components
        db_session = AsyncMock()
        embedding_service = AsyncMock()
        embedding_service.get_embedding.return_value = [0.1] * 1536
        
        # Create transcript manager
        transcript_manager = ChatTranscriptManager(db_session, embedding_service)
        transcript_manager._persist_conversation_event = AsyncMock()
        transcript_manager._update_conversation_thread = AsyncMock()
        transcript_manager._detect_conversation_patterns = AsyncMock()
        
        # Track conversation event
        event = await transcript_manager.track_conversation_event(
            session_id="e2e_session",
            event_type=ConversationEventType.MESSAGE_SENT,
            source_agent_id="agent_1",
            target_agent_id="agent_2",
            message_content="End-to-end test message",
            metadata={"test": "e2e"},
            response_time_ms=1000.0
        )
        
        # Verify event was processed
        assert event.session_id == "e2e_session"
        assert event.source_agent_id == "agent_1"
        assert event.target_agent_id == "agent_2"
        
        # Verify processing pipeline was triggered
        transcript_manager._persist_conversation_event.assert_called_once() 
        transcript_manager._update_conversation_thread.assert_called_once()
        transcript_manager._detect_conversation_patterns.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_analysis_to_dashboard_flow(self):
        """Test error analysis flowing to dashboard visualization."""
        # This would test error detection, analysis, and dashboard integration
        
        # Create analyzer
        analyzer = CommunicationAnalyzer()
        analyzer._update_agent_profiles = AsyncMock()
        analyzer._update_communication_flows = AsyncMock()
        
        # Create error events
        error_events = [
            ConversationEvent(
                id=str(uuid.uuid4()),
                session_id="error_session",
                timestamp=datetime.utcnow(),
                event_type=ConversationEventType.ERROR_OCCURRED,
                source_agent_id="agent_1",
                target_agent_id=None,
                message_content="Critical system error",
                metadata={}
            )
        ]
        
        # Mock error analysis methods
        analyzer._detect_communication_patterns = AsyncMock(return_value=[])
        analyzer._analyze_performance = AsyncMock(return_value=[])
        analyzer._analyze_agent_behavior = AsyncMock(return_value=[])
        analyzer._detect_bottlenecks = AsyncMock(return_value=[])
        analyzer._optimize_communication_flow = AsyncMock(return_value=[])
        analyzer._detect_anomalies = AsyncMock(return_value=[])
        analyzer._analyze_errors = AsyncMock(return_value=[
            CommunicationInsight(
                id="error_insight",
                analysis_type=AnalysisType.ERROR_ANALYSIS,
                severity=AlertSeverity.CRITICAL,
                title="Critical System Error",
                description="System experienced critical error",
                affected_agents=["agent_1"],
                recommendations=["Investigate system logs"],
                metrics={"error_count": 1},
                timestamp=datetime.utcnow(),
                session_id="error_session"
            )
        ])
        
        # Analyze events
        insights = await analyzer.analyze_conversation_events(
            error_events, [AnalysisType.ERROR_ANALYSIS]
        )
        
        # Verify critical error was detected
        assert len(insights) == 1
        assert insights[0].severity == AlertSeverity.CRITICAL
        assert "agent_1" in insights[0].affected_agents
        
        # This would continue to test dashboard integration...


# Fixtures for test database and services
@pytest.fixture
async def test_db_session():
    """Create test database session."""
    # This would create a test database session
    # For now, return a mock
    return AsyncMock()


@pytest.fixture
def test_embedding_service():
    """Create test embedding service."""
    service = AsyncMock()
    service.get_embedding.return_value = [0.1] * 1536
    return service


# Performance and load testing
class TestPerformance:
    """Performance tests for transcript analysis system."""
    
    @pytest.mark.asyncio
    async def test_high_volume_event_processing(self):
        """Test processing large volumes of conversation events."""
        # Create transcript manager
        db_session = AsyncMock()
        embedding_service = AsyncMock()
        embedding_service.get_embedding.return_value = [0.1] * 1536
        
        transcript_manager = ChatTranscriptManager(db_session, embedding_service)
        transcript_manager._persist_conversation_event = AsyncMock()
        transcript_manager._update_conversation_thread = AsyncMock()
        transcript_manager._detect_conversation_patterns = AsyncMock()
        
        # Process many events
        start_time = datetime.utcnow()
        event_count = 100
        
        tasks = []
        for i in range(event_count):
            task = transcript_manager.track_conversation_event(
                session_id="perf_session",
                event_type=ConversationEventType.MESSAGE_SENT,
                source_agent_id=f"agent_{i % 5}",  # 5 agents
                target_agent_id=f"agent_{(i + 1) % 5}",
                message_content=f"Performance test message {i}",
                metadata={"sequence": i}
            )
            tasks.append(task)
        
        # Execute all tasks
        results = await asyncio.gather(*tasks)
        
        # Verify performance
        duration = (datetime.utcnow() - start_time).total_seconds()
        events_per_second = event_count / duration
        
        # Verify all events were processed
        assert len(results) == event_count
        assert all(result.message_content.startswith("Performance test message") for result in results)
        
        # Log performance metrics
        print(f"Processed {event_count} events in {duration:.2f}s ({events_per_second:.1f} events/sec)")
        
        # Performance assertion (adjust based on requirements)
        assert events_per_second > 10, f"Performance too slow: {events_per_second} events/sec"
    
    @pytest.mark.asyncio
    async def test_concurrent_search_performance(self):
        """Test concurrent search operations."""
        # This would test multiple concurrent searches
        # to verify the system handles load appropriately
        pass


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])