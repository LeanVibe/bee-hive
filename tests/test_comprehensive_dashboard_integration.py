"""
Comprehensive Test Suite for Dashboard Integration System

Tests all aspects of the comprehensive dashboard integration including:
- Multi-agent workflow tracking
- Quality gates visualization
- Extended thinking sessions monitoring  
- Hook execution performance tracking
- Agent performance metrics aggregation
- Real-time WebSocket streaming
- API endpoints functionality
- Error handling and edge cases
"""

import asyncio
import json
import pytest
import websockets
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch
import uuid

from fastapi.testclient import TestClient
from fastapi import WebSocket

# Import components under test
from app.core.comprehensive_dashboard_integration import (
    ComprehensiveDashboardIntegration, IntegrationEventType,
    QualityGateStatus, ThinkingSessionPhase, WorkflowProgress,
    QualityGateResult, ThinkingSessionUpdate, HookPerformanceMetric,
    AgentPerformanceMetrics
)
from app.core.realtime_dashboard_streaming import (
    RealtimeDashboardStreaming, StreamPriority, CompressionType,
    StreamSubscription, StreamEvent
)
from app.api.v1.comprehensive_dashboard import router
from app.main import app


@pytest.fixture
def dashboard_integration():
    """Create dashboard integration instance for testing."""
    integration = ComprehensiveDashboardIntegration()
    return integration


@pytest.fixture
def streaming_system():
    """Create streaming system instance for testing."""
    streaming = RealtimeDashboardStreaming()
    return streaming


@pytest.fixture
def test_client():
    """Create test client for API testing."""
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def sample_workflow_data():
    """Sample workflow data for testing."""
    return {
        'workflow_id': 'test_workflow_001',
        'workflow_name': 'Test Multi-Agent Workflow',
        'total_steps': 10,
        'active_agents': ['agent_001', 'agent_002', 'agent_003'],
        'current_phase': 'initialization'
    }


@pytest.fixture
def sample_quality_gate_data():
    """Sample quality gate data for testing."""
    return {
        'gate_id': 'security_gate_001',
        'gate_name': 'Security Validation Gate',
        'status': QualityGateStatus.PASSED,
        'execution_time_ms': 1500,
        'success_criteria': {
            'security_score': 95,
            'vulnerability_count': 0,
            'compliance_check': True
        },
        'actual_results': {
            'security_score': 98,
            'vulnerability_count': 0,
            'compliance_check': True
        },
        'performance_metrics': {
            'scan_duration_ms': 1200,
            'files_analyzed': 150,
            'rules_executed': 45
        }
    }


@pytest.fixture
def sample_thinking_session_data():
    """Sample thinking session data for testing."""
    return {
        'session_id': 'thinking_session_001',
        'session_name': 'Problem Analysis Session',
        'phase': ThinkingSessionPhase.SOLUTION_EXPLORATION,
        'participating_agents': ['analyst_001', 'architect_001', 'reviewer_001'],
        'insights_generated': 8,
        'consensus_level': 0.85,
        'collaboration_quality': 0.92,
        'current_focus': 'Performance optimization strategies',
        'key_insights': [
            'Caching layer needed for database queries',
            'Async processing can improve response times',
            'Load balancing required for high availability'
        ]
    }


class TestComprehensiveDashboardIntegration:
    """Test comprehensive dashboard integration functionality."""
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, dashboard_integration):
        """Test dashboard integration system initialization."""
        # Test initial state
        assert not dashboard_integration.is_running
        assert len(dashboard_integration.workflow_progress) == 0
        assert len(dashboard_integration.quality_gate_results) == 0
        assert len(dashboard_integration.thinking_sessions) == 0
        
        # Test start
        await dashboard_integration.start()
        assert dashboard_integration.is_running
        assert len(dashboard_integration.background_tasks) > 0
        
        # Test stop
        await dashboard_integration.stop()
        assert not dashboard_integration.is_running
        assert len(dashboard_integration.background_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_workflow_progress_tracking(self, dashboard_integration, sample_workflow_data):
        """Test multi-agent workflow progress tracking."""
        await dashboard_integration.start()
        
        try:
            # Test workflow initialization
            progress = await dashboard_integration.track_workflow_progress(
                workflow_id=sample_workflow_data['workflow_id'],
                workflow_name=sample_workflow_data['workflow_name'],
                total_steps=sample_workflow_data['total_steps'],
                active_agents=sample_workflow_data['active_agents'],
                current_phase=sample_workflow_data['current_phase']
            )
            
            assert progress.workflow_id == sample_workflow_data['workflow_id']
            assert progress.workflow_name == sample_workflow_data['workflow_name']
            assert progress.total_steps == sample_workflow_data['total_steps']
            assert progress.completed_steps == 0
            assert progress.completion_percentage == 0.0
            assert not progress.is_completed
            
            # Test progress update
            updated_progress = await dashboard_integration.update_workflow_progress(
                workflow_id=sample_workflow_data['workflow_id'],
                completed_steps=5,
                current_phase='execution'
            )
            
            assert updated_progress.completed_steps == 5
            assert updated_progress.completion_percentage == 50.0
            assert updated_progress.current_phase == 'execution'
            assert not updated_progress.is_completed
            
            # Test completion
            await dashboard_integration.complete_workflow(
                workflow_id=sample_workflow_data['workflow_id'],
                success=True
            )
            
            final_progress = dashboard_integration.workflow_progress[sample_workflow_data['workflow_id']]
            assert final_progress.is_completed
            assert final_progress.success_rate > 0
            
        finally:
            await dashboard_integration.stop()
    
    @pytest.mark.asyncio
    async def test_quality_gate_results(self, dashboard_integration, sample_quality_gate_data):
        """Test quality gate result recording and analysis."""
        await dashboard_integration.start()
        
        try:
            # Record quality gate result
            result = await dashboard_integration.record_quality_gate_result(
                gate_id=sample_quality_gate_data['gate_id'],
                gate_name=sample_quality_gate_data['gate_name'],
                status=sample_quality_gate_data['status'],
                execution_time_ms=sample_quality_gate_data['execution_time_ms'],
                success_criteria=sample_quality_gate_data['success_criteria'],
                actual_results=sample_quality_gate_data['actual_results'],
                performance_metrics=sample_quality_gate_data['performance_metrics']
            )
            
            assert result.gate_id == sample_quality_gate_data['gate_id']
            assert result.gate_name == sample_quality_gate_data['gate_name']
            assert result.status == sample_quality_gate_data['status']
            assert result.passed == True
            assert result.execution_time_ms == sample_quality_gate_data['execution_time_ms']
            
            # Test summary generation
            summary = await dashboard_integration.get_quality_gate_summary(time_window_hours=1)
            
            assert summary['total_executions'] == 1
            assert summary['passed'] == 1
            assert summary['failed'] == 0
            assert summary['success_rate'] == 1.0
            
        finally:
            await dashboard_integration.stop()
    
    @pytest.mark.asyncio
    async def test_thinking_session_updates(self, dashboard_integration, sample_thinking_session_data):
        """Test extended thinking session monitoring."""
        await dashboard_integration.start()
        
        try:
            # Update thinking session
            update = await dashboard_integration.update_thinking_session(
                session_id=sample_thinking_session_data['session_id'],
                session_name=sample_thinking_session_data['session_name'],
                phase=sample_thinking_session_data['phase'],
                participating_agents=sample_thinking_session_data['participating_agents'],
                insights_generated=sample_thinking_session_data['insights_generated'],
                consensus_level=sample_thinking_session_data['consensus_level'],
                collaboration_quality=sample_thinking_session_data['collaboration_quality'],
                current_focus=sample_thinking_session_data['current_focus'],
                key_insights=sample_thinking_session_data['key_insights']
            )
            
            assert update.session_id == sample_thinking_session_data['session_id']
            assert update.session_name == sample_thinking_session_data['session_name']
            assert update.phase == sample_thinking_session_data['phase']
            assert update.insights_generated == sample_thinking_session_data['insights_generated']
            assert update.consensus_level == sample_thinking_session_data['consensus_level']
            assert update.collaboration_quality == sample_thinking_session_data['collaboration_quality']
            
            # Verify session is stored
            stored_session = dashboard_integration.thinking_sessions[sample_thinking_session_data['session_id']]
            assert stored_session.session_id == sample_thinking_session_data['session_id']
            
        finally:
            await dashboard_integration.stop()
    
    @pytest.mark.asyncio
    async def test_hook_performance_tracking(self, dashboard_integration):
        """Test hook execution performance tracking."""
        await dashboard_integration.start()
        
        try:
            # Record hook performance
            metric = await dashboard_integration.record_hook_performance(
                hook_id='pre_tool_use_001',
                hook_type='pre_tool_use',
                execution_time_ms=250,
                memory_usage_mb=15.5,
                success=True,
                agent_id='agent_001',
                session_id='session_001',
                payload_size_bytes=1024
            )
            
            assert metric.hook_id == 'pre_tool_use_001'
            assert metric.hook_type == 'pre_tool_use'
            assert metric.execution_time_ms == 250
            assert metric.memory_usage_mb == 15.5
            assert metric.success == True
            
            # Record multiple metrics for summary testing
            for i in range(5):
                await dashboard_integration.record_hook_performance(
                    hook_id=f'test_hook_{i}',
                    hook_type='test_hook',
                    execution_time_ms=100 + i * 10,
                    memory_usage_mb=10.0 + i,
                    success=i % 2 == 0,  # Alternate success/failure
                    agent_id='agent_001',
                    session_id='session_001'
                )
            
            # Test performance summary
            summary = await dashboard_integration.get_hook_performance_summary(time_window_minutes=5)
            
            assert summary['total_executions'] > 0
            assert summary['successful_executions'] > 0
            assert summary['success_rate'] > 0
            assert 'hook_type_details' in summary
            
        finally:
            await dashboard_integration.stop()
    
    @pytest.mark.asyncio
    async def test_agent_performance_metrics(self, dashboard_integration):
        """Test agent performance metrics aggregation."""
        await dashboard_integration.start()
        
        try:
            # Update agent performance
            metrics = await dashboard_integration.update_agent_performance(
                agent_id='agent_001',
                session_id='session_001',
                metrics={
                    'task_completion_rate': 0.95,
                    'average_response_time_ms': 1200.0,
                    'error_rate': 0.02,
                    'tool_usage_efficiency': 0.88,
                    'context_sharing_effectiveness': 0.92,
                    'output_quality_score': 0.89
                }
            )
            
            assert metrics.agent_id == 'agent_001'
            assert metrics.task_completion_rate == 0.95
            assert metrics.average_response_time_ms == 1200.0
            assert metrics.error_rate == 0.02
            
            # Test overall score calculation
            overall_score = metrics.calculate_overall_score()
            assert 0 <= overall_score <= 100
            assert overall_score > 80  # Should be high with good metrics
            
            # Test system overview
            overview = await dashboard_integration.get_system_performance_overview()
            
            assert 'active_agents' in overview
            assert 'average_agent_performance' in overview
            assert overview['active_agents'] >= 1
            
        finally:
            await dashboard_integration.stop()
    
    @pytest.mark.asyncio
    async def test_data_access_methods(self, dashboard_integration, sample_workflow_data):
        """Test API data access methods."""
        await dashboard_integration.start()
        
        try:
            # Set up test data
            await dashboard_integration.track_workflow_progress(
                workflow_id=sample_workflow_data['workflow_id'],
                workflow_name=sample_workflow_data['workflow_name'],
                total_steps=sample_workflow_data['total_steps'],
                active_agents=sample_workflow_data['active_agents']
            )
            
            # Test workflow data access
            workflow_data = await dashboard_integration.get_workflow_progress_data(
                workflow_ids=[sample_workflow_data['workflow_id']],
                include_completed=False
            )
            
            assert 'workflows' in workflow_data
            assert 'total_count' in workflow_data
            assert workflow_data['total_count'] >= 1
            
            # Test quality gates data access
            quality_data = await dashboard_integration.get_quality_gates_data(
                time_window_hours=24
            )
            
            assert 'gates' in quality_data
            assert 'summary' in quality_data
            assert 'timestamp' in quality_data
            
            # Test thinking sessions data access
            sessions_data = await dashboard_integration.get_thinking_sessions_data()
            
            assert 'sessions' in sessions_data
            assert 'active_sessions' in sessions_data
            assert 'timestamp' in sessions_data
            
            # Test agent performance data access
            performance_data = await dashboard_integration.get_agent_performance_data()
            
            assert 'agents' in performance_data
            assert 'system_overview' in performance_data
            assert 'timestamp' in performance_data
            
        finally:
            await dashboard_integration.stop()


class TestRealtimeDashboardStreaming:
    """Test real-time dashboard streaming functionality."""
    
    @pytest.mark.asyncio
    async def test_streaming_system_initialization(self, streaming_system):
        """Test streaming system initialization."""
        # Test initial state
        assert not streaming_system.is_running
        assert len(streaming_system.subscriptions) == 0
        assert len(streaming_system.active_streams) == 0
        
        # Mock Redis for testing
        with patch('app.core.realtime_dashboard_streaming.get_message_broker') as mock_redis:
            mock_redis.return_value = AsyncMock()
            
            await streaming_system.start()
            assert streaming_system.is_running
            assert len(streaming_system.background_tasks) > 0
            
            await streaming_system.stop()
            assert not streaming_system.is_running
    
    @pytest.mark.asyncio
    async def test_stream_registration(self, streaming_system):
        """Test WebSocket stream registration."""
        # Mock WebSocket
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        
        with patch('app.core.realtime_dashboard_streaming.get_message_broker') as mock_redis:
            mock_redis.return_value = AsyncMock()
            
            await streaming_system.start()
            
            try:
                # Test stream registration
                stream_id = await streaming_system.register_stream(
                    websocket=mock_websocket,
                    user_id='test_user',
                    filters={
                        'event_types': ['workflow_progress_update'],
                        'agent_ids': ['agent_001'],
                        'priority_threshold': 'medium'
                    }
                )
                
                assert stream_id is not None
                assert stream_id in streaming_system.subscriptions
                
                subscription = streaming_system.subscriptions[stream_id]
                assert subscription.user_id == 'test_user'
                assert 'workflow_progress_update' in subscription.event_types
                assert 'agent_001' in subscription.agent_ids
                
                # Test stream unregistration
                await streaming_system.unregister_stream(stream_id)
                assert stream_id not in streaming_system.subscriptions
                
            finally:
                await streaming_system.stop()
    
    @pytest.mark.asyncio
    async def test_event_broadcasting(self, streaming_system):
        """Test event broadcasting to streams."""
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        
        with patch('app.core.realtime_dashboard_streaming.get_message_broker') as mock_redis:
            mock_redis.return_value = AsyncMock()
            
            await streaming_system.start()
            
            try:
                # Register stream
                stream_id = await streaming_system.register_stream(
                    websocket=mock_websocket,
                    user_id='test_user'
                )
                
                # Broadcast event
                event_data = {
                    'workflow_id': 'test_workflow',
                    'progress': {'completed_steps': 5, 'total_steps': 10}
                }
                
                broadcast_count = await streaming_system.broadcast_event(
                    event_type='workflow_progress_update',
                    data=event_data,
                    priority=StreamPriority.MEDIUM
                )
                
                assert broadcast_count > 0
                
                # Allow some time for background processing
                await asyncio.sleep(0.2)
                
                # Verify message was sent (should be batched)
                # Note: In real implementation, this would be tested with actual message content
                
            finally:
                await streaming_system.stop()
    
    @pytest.mark.asyncio
    async def test_stream_filtering(self, streaming_system):
        """Test stream event filtering."""
        # Create subscription with specific filters
        subscription = StreamSubscription(
            stream_id='test_stream',
            websocket=AsyncMock(),
            event_types={'workflow_progress_update'},
            agent_ids={'agent_001'},
            priority_threshold=StreamPriority.MEDIUM
        )
        
        # Test event that should pass filters
        passing_event = {
            'event_type': 'workflow_progress_update',
            'priority': 'high',
            'data': {'agent_id': 'agent_001'}
        }
        
        assert subscription.should_receive_event(passing_event)
        
        # Test event that should be filtered out by event type
        filtered_event_1 = {
            'event_type': 'quality_gate_result',
            'priority': 'high',
            'data': {'agent_id': 'agent_001'}
        }
        
        assert not subscription.should_receive_event(filtered_event_1)
        
        # Test event that should be filtered out by agent ID
        filtered_event_2 = {
            'event_type': 'workflow_progress_update',
            'priority': 'high',
            'data': {'agent_id': 'agent_002'}
        }
        
        assert not subscription.should_receive_event(filtered_event_2)
        
        # Test event that should be filtered out by priority
        filtered_event_3 = {
            'event_type': 'workflow_progress_update',
            'priority': 'low',
            'data': {'agent_id': 'agent_001'}
        }
        
        assert not subscription.should_receive_event(filtered_event_3)
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, streaming_system):
        """Test stream rate limiting."""
        subscription = StreamSubscription(
            stream_id='test_stream',
            websocket=AsyncMock(),
            max_events_per_second=2
        )
        
        # Simulate rapid events
        import time
        current_time = time.time()
        
        # Add events up to limit
        subscription.event_timestamps.extend([current_time, current_time])
        
        # Should be rate limited
        assert subscription.is_rate_limited()
        
        # Clear timestamps and test again
        subscription.event_timestamps.clear()
        assert not subscription.is_rate_limited()
    
    def test_stream_event_serialization(self):
        """Test stream event serialization and compression."""
        event = StreamEvent(
            event_id='test_event_001',
            event_type='workflow_progress_update',
            priority=StreamPriority.HIGH,
            data={'test': 'data', 'large_field': 'x' * 200},
            source_component='test'
        )
        
        # Test normal serialization
        normal_dict = event.to_dict(compress=False)
        assert normal_dict['event_id'] == 'test_event_001'
        assert normal_dict['event_type'] == 'workflow_progress_update'
        assert normal_dict['priority'] == 'high'
        assert len(normal_dict['data']['large_field']) == 200
        
        # Test compressed serialization
        compressed_dict = event.to_dict(compress=True)
        assert compressed_dict['event_id'] == 'test_event_001'
        # Large field should be truncated for mobile
        assert len(compressed_dict['data']['large_field']) <= 103  # 100 + '...'
    
    @pytest.mark.asyncio
    async def test_statistics_collection(self, streaming_system):
        """Test streaming statistics collection."""
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        
        with patch('app.core.realtime_dashboard_streaming.get_message_broker') as mock_redis:
            mock_redis.return_value = AsyncMock()
            
            await streaming_system.start()
            
            try:
                # Register some streams
                stream_ids = []
                for i in range(3):
                    stream_id = await streaming_system.register_stream(
                        websocket=mock_websocket,
                        user_id=f'user_{i}'
                    )
                    stream_ids.append(stream_id)
                
                # Get statistics
                stats = await streaming_system.get_stream_statistics()
                
                assert 'total_connections' in stats
                assert 'metrics' in stats
                assert 'performance' in stats
                assert 'subscription_details' in stats
                
                assert stats['total_connections'] == 3
                assert len(stats['subscription_details']) == 3
                
            finally:
                await streaming_system.stop()


class TestDashboardAPI:
    """Test comprehensive dashboard API endpoints."""
    
    def test_workflow_tracking_endpoints(self, test_client):
        """Test workflow tracking API endpoints."""
        # Mock authentication
        with patch('app.api.v1.comprehensive_dashboard.get_current_user') as mock_auth:
            mock_auth.return_value = Mock(id='test_user')
            
            # Mock comprehensive dashboard integration
            with patch('app.api.v1.comprehensive_dashboard.comprehensive_dashboard_integration') as mock_integration:
                mock_progress = Mock()
                mock_progress.to_dict.return_value = {
                    'workflow_id': 'test_workflow',
                    'completion_percentage': 0.0,
                    'is_completed': False
                }
                mock_integration.track_workflow_progress.return_value = mock_progress
                
                # Test workflow tracking start
                response = test_client.post(
                    "/comprehensive-dashboard/workflows/track?workflow_id=test_workflow",
                    json={
                        "workflow_name": "Test Workflow",
                        "total_steps": 10,
                        "active_agents": ["agent_001", "agent_002"],
                        "current_phase": "initialization"
                    }
                )
                
                assert response.status_code == 201
                data = response.json()
                assert data['success'] == True
                assert data['workflow_id'] == 'test_workflow'
    
    def test_quality_gates_endpoints(self, test_client):
        """Test quality gates API endpoints."""
        with patch('app.api.v1.comprehensive_dashboard.get_current_user') as mock_auth:
            mock_auth.return_value = Mock(id='test_user')
            
            with patch('app.api.v1.comprehensive_dashboard.comprehensive_dashboard_integration') as mock_integration:
                mock_result = Mock()
                mock_result.to_dict.return_value = {
                    'gate_id': 'test_gate',
                    'status': 'passed',
                    'passed': True
                }
                mock_integration.record_quality_gate_result.return_value = mock_result
                
                # Test quality gate result recording
                response = test_client.post(
                    "/comprehensive-dashboard/quality-gates/test_gate/result",
                    json={
                        "gate_name": "Test Gate",
                        "status": "passed",
                        "execution_time_ms": 1000,
                        "success_criteria": {"score": 90},
                        "actual_results": {"score": 95}
                    }
                )
                
                assert response.status_code == 201
                data = response.json()
                assert data['success'] == True
                assert data['gate_id'] == 'test_gate'
    
    def test_thinking_sessions_endpoints(self, test_client):
        """Test thinking sessions API endpoints."""
        with patch('app.api.v1.comprehensive_dashboard.get_current_user') as mock_auth:
            mock_auth.return_value = Mock(id='test_user')
            
            with patch('app.api.v1.comprehensive_dashboard.comprehensive_dashboard_integration') as mock_integration:
                mock_session = Mock()
                mock_session.to_dict.return_value = {
                    'session_id': 'test_session',
                    'phase': 'solution_exploration',
                    'consensus_level': 0.85
                }
                mock_integration.update_thinking_session.return_value = mock_session
                
                # Test thinking session update
                response = test_client.put(
                    "/comprehensive-dashboard/thinking-sessions/test_session",
                    json={
                        "session_name": "Test Session",
                        "phase": "solution_exploration",
                        "participating_agents": ["agent_001", "agent_002"],
                        "insights_generated": 5,
                        "consensus_level": 0.85,
                        "collaboration_quality": 0.90,
                        "current_focus": "Performance optimization"
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data['success'] == True
                assert data['session_id'] == 'test_session'
    
    def test_system_overview_endpoint(self, test_client):
        """Test system overview API endpoint."""
        with patch('app.api.v1.comprehensive_dashboard.get_current_user') as mock_auth:
            mock_auth.return_value = Mock(id='test_user')
            
            with patch('app.api.v1.comprehensive_dashboard.comprehensive_dashboard_integration') as mock_integration:
                mock_overview = {
                    'active_workflows': 5,
                    'completed_workflows': 10,
                    'active_thinking_sessions': 2,
                    'active_agents': 8,
                    'average_agent_performance': 85.5,
                    'agent_performance_std': 5.2,
                    'quality_gates_summary': {'total_executions': 20},
                    'hook_performance_summary': {'success_rate': 0.95},
                    'system_health': {'status': 'healthy'},
                    'timestamp': '2024-01-01T00:00:00'
                }
                mock_integration.get_system_performance_overview.return_value = mock_overview
                
                # Test system overview
                response = test_client.get("/comprehensive-dashboard/overview")
                
                assert response.status_code == 200
                data = response.json()
                assert data['active_workflows'] == 5
                assert data['active_agents'] == 8
                assert data['system_health']['status'] == 'healthy'
    
    def test_stream_configuration_endpoint(self, test_client):
        """Test stream configuration API endpoint."""
        with patch('app.api.v1.comprehensive_dashboard.get_current_user') as mock_auth:
            mock_auth.return_value = Mock(id='test_user')
            
            with patch('app.api.v1.comprehensive_dashboard.realtime_dashboard_streaming') as mock_streaming:
                mock_streaming.update_stream_filters.return_value = True
                
                # Test stream configuration
                response = test_client.post(
                    "/comprehensive-dashboard/stream/test_stream_id/configure",
                    json={
                        "event_types": ["workflow_progress_update"],
                        "agent_ids": ["agent_001"],
                        "priority_threshold": "medium",
                        "max_events_per_second": 5,
                        "batch_size": 3,
                        "compression": "smart",
                        "mobile_optimized": True
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data['success'] == True
                assert data['stream_id'] == 'test_stream_id'
    
    def test_error_handling(self, test_client):
        """Test API error handling."""
        with patch('app.api.v1.comprehensive_dashboard.get_current_user') as mock_auth:
            mock_auth.return_value = Mock(id='test_user')
            
            with patch('app.api.v1.comprehensive_dashboard.comprehensive_dashboard_integration') as mock_integration:
                # Simulate error
                mock_integration.track_workflow_progress.side_effect = Exception("Test error")
                
                # Test error response
                response = test_client.post(
                    "/comprehensive-dashboard/workflows/track?workflow_id=test_workflow",
                    json={
                        "workflow_name": "Test Workflow",
                        "total_steps": 10,
                        "active_agents": ["agent_001"],
                        "current_phase": "initialization"
                    }
                )
                
                assert response.status_code == 500
                data = response.json()
                assert "Failed to start workflow tracking" in data['detail']


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_lifecycle(self, dashboard_integration):
        """Test complete workflow lifecycle with all monitoring features."""
        await dashboard_integration.start()
        
        try:
            workflow_id = 'integration_test_workflow'
            
            # Start workflow tracking
            progress = await dashboard_integration.track_workflow_progress(
                workflow_id=workflow_id,
                workflow_name='Integration Test Workflow',
                total_steps=5,
                active_agents=['agent_001', 'agent_002'],
                current_phase='initialization'
            )
            
            # Simulate workflow execution with quality gates
            for step in range(1, 6):
                # Update workflow progress
                await dashboard_integration.update_workflow_progress(
                    workflow_id=workflow_id,
                    completed_steps=step,
                    current_phase=f'step_{step}'
                )
                
                # Record quality gate for each step
                await dashboard_integration.record_quality_gate_result(
                    gate_id=f'gate_{step}',
                    gate_name=f'Quality Gate {step}',
                    status=QualityGateStatus.PASSED,
                    execution_time_ms=500 + step * 100,
                    success_criteria={'quality_score': 90},
                    actual_results={'quality_score': 95}
                )
                
                # Record hook performance
                await dashboard_integration.record_hook_performance(
                    hook_id=f'hook_{step}',
                    hook_type='step_execution',
                    execution_time_ms=100 + step * 20,
                    memory_usage_mb=10.0 + step,
                    success=True,
                    agent_id='agent_001',
                    session_id='session_001'
                )
            
            # Complete workflow
            await dashboard_integration.complete_workflow(workflow_id, success=True)
            
            # Verify final state
            final_progress = dashboard_integration.workflow_progress[workflow_id]
            assert final_progress.is_completed
            assert final_progress.success_rate > 0
            
            # Verify quality gate summary
            summary = await dashboard_integration.get_quality_gate_summary()
            assert summary['total_executions'] >= 5
            assert summary['success_rate'] == 1.0
            
            # Verify hook performance summary
            hook_summary = await dashboard_integration.get_hook_performance_summary()
            assert hook_summary['total_executions'] >= 5
            assert hook_summary['success_rate'] == 1.0
            
        finally:
            await dashboard_integration.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, dashboard_integration):
        """Test concurrent dashboard operations."""
        await dashboard_integration.start()
        
        try:
            # Create multiple concurrent operations
            tasks = []
            
            # Start multiple workflows
            for i in range(3):
                task = dashboard_integration.track_workflow_progress(
                    workflow_id=f'concurrent_workflow_{i}',
                    workflow_name=f'Concurrent Workflow {i}',
                    total_steps=3,
                    active_agents=[f'agent_{i}']
                )
                tasks.append(task)
            
            # Wait for all workflows to start
            results = await asyncio.gather(*tasks)
            assert len(results) == 3
            
            # Update all workflows concurrently
            update_tasks = []
            for i in range(3):
                task = dashboard_integration.update_workflow_progress(
                    workflow_id=f'concurrent_workflow_{i}',
                    completed_steps=2
                )
                update_tasks.append(task)
            
            await asyncio.gather(*update_tasks)
            
            # Verify all workflows were updated
            for i in range(3):
                workflow = dashboard_integration.workflow_progress[f'concurrent_workflow_{i}']
                assert workflow.completed_steps == 2
            
        finally:
            await dashboard_integration.stop()
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, dashboard_integration):
        """Test error handling and recovery scenarios."""
        await dashboard_integration.start()
        
        try:
            # Test workflow with errors
            workflow_id = 'error_test_workflow'
            
            await dashboard_integration.track_workflow_progress(
                workflow_id=workflow_id,
                workflow_name='Error Test Workflow',
                total_steps=3,
                active_agents=['agent_001']
            )
            
            # Simulate errors
            for _ in range(2):
                await dashboard_integration.update_workflow_progress(
                    workflow_id=workflow_id,
                    completed_steps=1,
                    increment_errors=True
                )
            
            # Complete with errors
            await dashboard_integration.complete_workflow(workflow_id, success=False)
            
            # Verify error tracking
            final_progress = dashboard_integration.workflow_progress[workflow_id]
            assert final_progress.error_count == 2
            assert final_progress.success_rate < 1.0
            
            # Test non-existent workflow update
            result = await dashboard_integration.update_workflow_progress(
                workflow_id='non_existent_workflow',
                completed_steps=1
            )
            assert result is None
            
        finally:
            await dashboard_integration.stop()


if __name__ == "__main__":
    # Run specific test for debugging
    pytest.main([__file__ + "::TestComprehensiveDashboardIntegration::test_workflow_progress_tracking", "-v"])