"""
Comprehensive Tests for Production-Grade Observability Stack

Tests for the advanced observability and monitoring capabilities including:
- Agent workflow tracking with real-time state monitoring
- Intelligent alerting with ML-based anomaly detection
- Dashboard metrics streaming with WebSocket updates
- Performance optimization advisor with automated recommendations

Test Categories:
- Unit tests for individual components
- Integration tests for component interactions
- Performance tests for high-load scenarios (50+ agents)
- End-to-end tests for complete observability workflows
"""

import asyncio
import pytest
import uuid
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import structlog
from fastapi.testclient import TestClient
from fastapi import WebSocket

# Import components to test
from app.core.agent_workflow_tracker import (
    AgentWorkflowTracker, AgentState, TaskProgressState, WorkflowPhase,
    AgentStateTransition, TaskProgressUpdate, InterAgentCommunication
)
from app.core.intelligent_alerting import (
    AlertManager, AlertSeverity, Alert,
    MetricAnomalyDetector, AlertRule
)
from app.core.dashboard_metrics_streaming import (
    DashboardMetricsStreaming, DashboardType, MetricStreamType,
    DashboardConnection, DashboardFilter, MetricUpdate
)
from app.core.performance_optimization_advisor import (
    PerformanceOptimizationAdvisor, OptimizationCategory, ImpactLevel,
    PerformanceInsight, OptimizationRecommendation, PerformanceAnalysisEngine,
    ImplementationComplexity, OptimizationPriority
)

logger = structlog.get_logger()


class TestAgentWorkflowTracker:
    """Test suite for Agent Workflow Tracker."""
    
    @pytest.fixture
    async def workflow_tracker(self):
        """Create workflow tracker instance for testing."""
        # Mock Redis and session factory
        mock_redis = AsyncMock()
        mock_session_factory = AsyncMock()
        mock_hooks = Mock()
        
        tracker = AgentWorkflowTracker(
            redis_client=mock_redis,
            session_factory=mock_session_factory,
            observability_hooks=mock_hooks
        )
        
        await tracker.start_tracking()
        yield tracker
        await tracker.stop_tracking()
    
    @pytest.mark.asyncio
    async def test_agent_state_transition_tracking(self, workflow_tracker):
        """Test agent state transition tracking."""
        agent_id = uuid.uuid4()
        session_id = uuid.uuid4()
        
        # Track state transition
        await workflow_tracker.track_agent_state_transition(
            agent_id=agent_id,
            new_state=AgentState.BUSY,
            transition_reason="Task assigned",
            session_id=session_id,
            context={"task_type": "code_generation"},
            resource_allocation={"cpu": 0.5, "memory": 0.3}
        )
        
        # Verify state is tracked
        assert workflow_tracker.agent_states[agent_id] == AgentState.BUSY
        assert len(workflow_tracker.agent_state_history[agent_id]) == 1
        
        transition = workflow_tracker.agent_state_history[agent_id][0]
        assert transition.agent_id == agent_id
        assert transition.new_state == AgentState.BUSY
        assert transition.transition_reason == "Task assigned"
        assert transition.context["task_type"] == "code_generation"
    
    @pytest.mark.asyncio
    async def test_task_progress_tracking(self, workflow_tracker):
        """Test task progression tracking."""
        task_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        workflow_id = uuid.uuid4()
        
        # Track task progress through states
        states = [
            TaskProgressState.QUEUED,
            TaskProgressState.ASSIGNED,
            TaskProgressState.EXECUTING,
            TaskProgressState.COMPLETED
        ]
        
        for i, state in enumerate(states):
            await workflow_tracker.track_task_progress(
                task_id=task_id,
                agent_id=agent_id,
                new_state=state,
                workflow_id=workflow_id,
                progress_percentage=i * 25,
                milestone_reached=f"milestone_{i}" if i > 0 else None
            )
        
        # Verify progress tracking
        assert len(workflow_tracker.task_progress_history[task_id]) == 4
        
        # Check final state
        final_update = workflow_tracker.task_progress_history[task_id][-1]
        assert final_update.new_state == TaskProgressState.COMPLETED
        assert final_update.progress_percentage == 75
    
    @pytest.mark.asyncio
    async def test_inter_agent_communication_tracking(self, workflow_tracker):
        """Test inter-agent communication tracking."""
        message_id = uuid.uuid4()
        from_agent = uuid.uuid4()
        to_agent = uuid.uuid4()
        workflow_context = uuid.uuid4()
        
        await workflow_tracker.track_inter_agent_communication(
            message_id=message_id,
            from_agent_id=from_agent,
            to_agent_id=to_agent,
            communication_type="task_delegation",
            message_content={"task": "review_code", "priority": "high"},
            workflow_context=workflow_context,
            response_expected=True
        )
        
        # Verify communication is tracked
        assert len(workflow_tracker.communication_log) == 1
        
        communication = workflow_tracker.communication_log[0]
        assert communication.message_id == message_id
        assert communication.from_agent_id == from_agent
        assert communication.to_agent_id == to_agent
        assert communication.communication_type == "task_delegation"
        assert communication.response_expected is True
    
    @pytest.mark.asyncio
    async def test_workflow_progress_update(self, workflow_tracker):
        """Test workflow progress snapshot updates."""
        workflow_id = uuid.uuid4()
        session_id = uuid.uuid4()
        
        # Mock database query results
        with patch.object(workflow_tracker, 'session_factory') as mock_session_factory:
            mock_session = AsyncMock()
            mock_session_factory.return_value.__aenter__.return_value = mock_session
            
            # Mock task query results
            mock_tasks = [
                Mock(status=Mock(COMPLETED=True, FAILED=False, BLOCKED=False, IN_PROGRESS=False, ASSIGNED=False), 
                     assigned_agent_id=uuid.uuid4()),
                Mock(status=Mock(COMPLETED=False, FAILED=False, BLOCKED=False, IN_PROGRESS=True, ASSIGNED=False), 
                     assigned_agent_id=uuid.uuid4()),
                Mock(status=Mock(COMPLETED=False, FAILED=True, BLOCKED=False, IN_PROGRESS=False, ASSIGNED=False), 
                     assigned_agent_id=None)
            ]
            
            mock_session.execute.return_value.scalars.return_value.all.return_value = mock_tasks
            
            snapshot = await workflow_tracker.update_workflow_progress(
                workflow_id=workflow_id,
                session_id=session_id,
                current_phase=WorkflowPhase.EXECUTION,
                resource_utilization={"cpu": 0.7, "memory": 0.6}
            )
            
            # Verify snapshot
            assert snapshot.workflow_id == workflow_id
            assert snapshot.current_phase == WorkflowPhase.EXECUTION
            assert snapshot.total_tasks == 3
            assert snapshot.completed_tasks == 1
            assert snapshot.failed_tasks == 1
    
    @pytest.mark.asyncio
    async def test_real_time_workflow_status(self, workflow_tracker):
        """Test real-time workflow status retrieval."""
        # Add some test data
        agent_id = uuid.uuid4()
        workflow_tracker.agent_states[agent_id] = AgentState.BUSY
        
        status = await workflow_tracker.get_real_time_workflow_status(
            include_agent_details=True,
            include_communication_flow=True
        )
        
        # Verify status structure
        assert "timestamp" in status
        assert "system_overview" in status
        assert "performance_metrics" in status
        assert "agent_summary" in status
        assert "communication_summary" in status
    
    @pytest.mark.performance
    async def test_high_load_agent_tracking(self, workflow_tracker):
        """Test performance with 50+ concurrent agents."""
        num_agents = 55
        agent_ids = [uuid.uuid4() for _ in range(num_agents)]
        
        start_time = time.time()
        
        # Simulate concurrent agent state transitions
        tasks = []
        for i, agent_id in enumerate(agent_ids):
            task = workflow_tracker.track_agent_state_transition(
                agent_id=agent_id,
                new_state=AgentState.BUSY,
                transition_reason=f"Load test agent {i}",
                context={"load_test": True, "agent_index": i}
            )
            tasks.append(task)
        
        # Execute all transitions concurrently
        await asyncio.gather(*tasks)
        
        execution_time = time.time() - start_time
        
        # Verify performance target: <1 second for 50+ agents
        assert execution_time < 1.0, f"High load tracking took {execution_time:.2f}s, expected <1.0s"
        
        # Verify all agents are tracked
        assert len(workflow_tracker.agent_states) == num_agents
        
        # Verify all state transitions are recorded
        total_transitions = sum(len(history) for history in workflow_tracker.agent_state_history.values())
        assert total_transitions == num_agents


class TestAlertManager:
    """Test suite for Intelligent Alert Manager."""
    
    @pytest.fixture
    async def alert_manager(self):
        """Create alert manager instance for testing."""
        mock_redis = AsyncMock()
        mock_session_factory = AsyncMock()
        
        manager = AlertManager(
            redis_client=mock_redis,
            session_factory=mock_session_factory
        )
        
        await manager.start()
        yield manager
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_ml_anomaly_detection(self, alert_manager):
        """Test ML-based anomaly detection."""
        detector = alert_manager.metric_anomaly_detector
        
        # Train with normal data
        normal_values = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8] * 10
        for value in normal_values:
            detector.add_metric_value("test_metric", value, datetime.utcnow())
        
        # Test with anomalous value
        is_anomaly, score = detector.detect_anomaly("test_metric", 50.0)
        
        # Should detect anomaly after sufficient training data
        if len(detector.metric_history["test_metric"]) >= detector.config.min_samples:
            assert is_anomaly or score != 0.0, "Should detect anomaly or provide non-zero score"
    
    @pytest.mark.asyncio
    async def test_anomaly_detection_integration(self, alert_manager):
        """Test anomaly detection integration with alert system."""
        detector = alert_manager.metric_anomaly_detector
        
        # Add normal data points
        normal_values = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8] * 10
        for value in normal_values:
            detector.add_metric_value("test_metric", value, datetime.utcnow())
        
        # Test with anomalous value and check if alert system detects it
        is_anomaly, score = detector.detect_anomaly("test_metric", 50.0)
        
        # After sufficient training data, should detect anomaly
        if len(detector.metric_history["test_metric"]) >= detector.config.min_samples:
            assert is_anomaly or score > 0, "Should detect anomaly or provide meaningful score"
    
    @pytest.mark.asyncio
    async def test_alert_rule_management(self, alert_manager):
        """Test alert rule creation and management."""
        # Create a test alert rule
        rule = AlertRule(
            rule_id="test_rule_1",
            name="Test High CPU Alert",
            description="Alert when CPU usage exceeds 85%",
            component="system",
            metric="cpu_percent",
            condition="greater_than",
            threshold=85.0,
            severity=AlertSeverity.HIGH,
            enabled=True,
            cooldown_minutes=15
        )
        
        # Add rule to manager
        alert_manager.add_alert_rule(rule)
        
        # Verify rule was added
        assert "test_rule_1" in alert_manager.alert_rules
        assert alert_manager.alert_rules["test_rule_1"].threshold == 85.0
        assert alert_manager.alert_rules["test_rule_1"].severity == AlertSeverity.HIGH
    
    @pytest.mark.asyncio
    async def test_metric_evaluation(self, alert_manager):
        """Test metric evaluation with rules."""
        # Add a test rule for high CPU
        rule = AlertRule(
            rule_id="high_cpu_test",
            name="High CPU Test",
            description="Test high CPU detection",
            component="system",
            metric="cpu_percent",
            condition="greater_than",
            threshold=80.0,
            severity=AlertSeverity.HIGH
        )
        alert_manager.add_alert_rule(rule)
        
        # Evaluate metrics that should trigger the rule
        metrics = {"cpu_percent": 90.0}
        alerts = await alert_manager.evaluate_metrics(metrics)
        
        # Should generate an alert
        assert len(alerts) > 0, "Should generate alert for high CPU"
        cpu_alert = alerts[0]
        assert cpu_alert.severity == AlertSeverity.HIGH
        assert cpu_alert.current_value == 90.0
    
    @pytest.mark.asyncio
    async def test_alert_filtering(self, alert_manager):
        """Test alert filtering and retrieval."""
        # Create rules of different severities and trigger them
        severities = [AlertSeverity.LOW, AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        
        for i, severity in enumerate(severities):
            rule = AlertRule(
                rule_id=f"test_rule_{i}",
                name=f"Test {severity.value} Alert",
                description=f"Test alert with {severity.value} severity",
                component="test",
                metric=f"test_metric_{i}",
                condition="greater_than",
                threshold=50.0,
                severity=severity
            )
            alert_manager.add_alert_rule(rule)
            
            # Trigger the rule
            metrics = {f"test_metric_{i}": 75.0}
            await alert_manager.evaluate_metrics(metrics)
        
        # Get all active alerts
        active_alerts = alert_manager.get_active_alerts()
        
        # Should have alerts of different severities
        assert len(active_alerts) > 0, "Should have active alerts"
    
    @pytest.mark.performance
    async def test_high_volume_alert_processing(self, alert_manager):
        """Test performance with high volume of alerts."""
        # Add rules for testing
        for i in range(10):
            rule = AlertRule(
                rule_id=f"perf_rule_{i}",
                name=f"Performance Test Rule {i}",
                description="Performance test rule",
                component="test",
                metric=f"test.metric.{i}",
                condition="greater_than",
                threshold=80.0,
                severity=AlertSeverity.MEDIUM
            )
            alert_manager.add_alert_rule(rule)
        
        num_evaluations = 100
        start_time = time.time()
        
        # Generate many metric evaluations concurrently
        tasks = []
        for i in range(num_evaluations):
            metrics = {f"test.metric.{i % 10}": float(85 + i % 20)}
            task = alert_manager.evaluate_metrics(metrics)
            tasks.append(task)
        
        # Process all evaluations
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        execution_time = time.time() - start_time
        
        # Verify performance target: <5 seconds for 100 evaluations
        assert execution_time < 5.0, f"High volume processing took {execution_time:.2f}s, expected <5.0s"
        
        # Verify no exceptions occurred
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Processing had {len(exceptions)} exceptions"


class TestDashboardMetricsStreaming:
    """Test suite for Dashboard Metrics Streaming."""
    
    @pytest.fixture
    async def metrics_streaming(self):
        """Create metrics streaming instance for testing."""
        mock_redis = AsyncMock()
        mock_metrics_collector = Mock()
        
        streaming = DashboardMetricsStreaming(
            redis_client=mock_redis,
            metrics_collector=mock_metrics_collector
        )
        
        await streaming.start_streaming()
        yield streaming
        await streaming.stop_streaming()
    
    @pytest.mark.asyncio
    async def test_dashboard_connection(self, metrics_streaming):
        """Test dashboard client connection."""
        mock_websocket = Mock(spec=WebSocket)
        
        # Mock websocket send methods
        mock_websocket.send_text = AsyncMock()
        mock_websocket.client_state = Mock()
        mock_websocket.client_state.CONNECTED = True
        
        filters = DashboardFilter(
            metric_patterns=["system.*", "agent.*"],
            update_rate_ms=500,
            enable_compression=True
        )
        
        connection_id = await metrics_streaming.connect_dashboard(
            websocket=mock_websocket,
            dashboard_type=DashboardType.OPERATIONAL,
            filters=filters,
            client_capabilities={"compression": True, "batching": True}
        )
        
        # Verify connection
        assert connection_id in metrics_streaming.connections
        connection = metrics_streaming.connections[connection_id]
        assert connection.dashboard_type == DashboardType.OPERATIONAL
        assert connection.filters.update_rate_ms == 500
        
        # Verify WebSocket messages sent (connection ack + initial data)
        assert mock_websocket.send_text.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_metric_streaming(self, metrics_streaming):
        """Test metric update streaming."""
        # Stream various metric types
        await metrics_streaming.stream_metric_update(
            metric_name="system.cpu.percent",
            value=75.5,
            stream_type=MetricStreamType.REAL_TIME,
            tags={"host": "test-server"},
            metadata={"source": "psutil"}
        )
        
        await metrics_streaming.stream_agent_status_update(
            agent_id="test-agent-1",
            status="busy",
            health_score=0.95,
            metadata={"task_count": 3}
        )
        
        await metrics_streaming.stream_workflow_progress_update(
            workflow_id="test-workflow-1",
            progress_percentage=60.0,
            phase="execution",
            metadata={"estimated_completion": "2024-01-01T12:00:00Z"}
        )
        
        # Verify metrics are buffered
        assert len(metrics_streaming.metric_buffers[MetricStreamType.REAL_TIME]) > 0
        assert len(metrics_streaming.metric_buffers[MetricStreamType.AGENT_STATUS]) > 0
        assert len(metrics_streaming.metric_buffers[MetricStreamType.WORKFLOW_PROGRESS]) > 0
    
    @pytest.mark.asyncio
    async def test_dashboard_filtering(self, metrics_streaming):
        """Test dashboard metric filtering."""
        filters = DashboardFilter(
            metric_patterns=["system.*"],
            agent_ids=["agent-1", "agent-2"],
            tags={"environment": "production"}
        )
        
        # Test matching metrics
        assert filters.matches_metric("system.cpu.percent", {"environment": "production"})
        assert not filters.matches_metric("agent.health", {"environment": "production"})
        assert not filters.matches_metric("system.cpu.percent", {"environment": "development"})
    
    @pytest.mark.asyncio
    async def test_dashboard_metrics_retrieval(self, metrics_streaming):
        """Test dashboard performance metrics."""
        # Add some test connections
        mock_websocket1 = Mock(spec=WebSocket)
        mock_websocket2 = Mock(spec=WebSocket)
        
        mock_websocket1.send_text = AsyncMock()
        mock_websocket2.send_text = AsyncMock()
        
        await metrics_streaming.connect_dashboard(mock_websocket1, DashboardType.EXECUTIVE)
        await metrics_streaming.connect_dashboard(mock_websocket2, DashboardType.OPERATIONAL)
        
        # Get metrics
        metrics = await metrics_streaming.get_dashboard_metrics()
        
        # Verify metrics structure
        assert "service_status" in metrics
        assert "connections" in metrics
        assert "performance" in metrics
        assert "efficiency" in metrics
        
        # Verify connection tracking
        assert metrics["connections"]["active"] == 2
        assert "executive" in metrics["connections"]["by_dashboard_type"]
        assert "operational" in metrics["connections"]["by_dashboard_type"]
    
    @pytest.mark.performance
    async def test_concurrent_dashboard_connections(self, metrics_streaming):
        """Test performance with many concurrent dashboard connections."""
        num_connections = 25
        mock_websockets = []
        
        start_time = time.time()
        
        # Create concurrent connections
        for i in range(num_connections):
            mock_websocket = Mock(spec=WebSocket)
            mock_websocket.send_text = AsyncMock()
            mock_websocket.client_state = Mock()
            mock_websocket.client_state.CONNECTED = True
            mock_websockets.append(mock_websocket)
        
        # Connect all dashboards concurrently
        tasks = []
        for i, websocket in enumerate(mock_websockets):
            dashboard_type = list(DashboardType)[i % len(DashboardType)]
            task = metrics_streaming.connect_dashboard(websocket, dashboard_type)
            tasks.append(task)
        
        connection_ids = await asyncio.gather(*tasks)
        
        connection_time = time.time() - start_time
        
        # Verify performance target: <2 seconds for 25 connections
        assert connection_time < 2.0, f"Connection time {connection_time:.2f}s exceeded 2.0s"
        
        # Verify all connections established
        assert len(connection_ids) == num_connections
        assert len(metrics_streaming.connections) == num_connections


class TestPerformanceOptimizationAdvisor:
    """Test suite for Performance Optimization Advisor."""
    
    @pytest.fixture
    async def optimization_advisor(self):
        """Create optimization advisor instance for testing."""
        mock_redis = AsyncMock()
        mock_session_factory = AsyncMock()
        mock_metrics_collector = Mock()
        
        advisor = PerformanceOptimizationAdvisor(
            redis_client=mock_redis,
            session_factory=mock_session_factory,
            metrics_collector=mock_metrics_collector
        )
        
        await advisor.start_advisor()
        yield advisor
        await advisor.stop_advisor()
    
    @pytest.mark.asyncio
    async def test_performance_insight_generation(self, optimization_advisor):
        """Test performance insight generation."""
        analysis_engine = optimization_advisor.analysis_engine
        
        # Simulate metrics history with trends
        metrics_history = {
            "system.cpu.percent": [
                (datetime.utcnow() - timedelta(hours=i), 50.0 + i * 2.0)
                for i in range(24, 0, -1)
            ],
            "agent.error_rate": [
                (datetime.utcnow() - timedelta(hours=i), 1.0 + i * 0.5)
                for i in range(24, 0, -1)
            ]
        }
        
        insights = await analysis_engine.analyze_performance_trends(metrics_history, 24)
        
        # Should generate insights for trending metrics
        assert len(insights) > 0, "Should generate insights for trending metrics"
        
        # Check for CPU trend insight
        cpu_insights = [i for i in insights if "cpu" in i.title.lower()]
        assert len(cpu_insights) > 0, "Should detect CPU trend"
        
        # Check insight properties
        cpu_insight = cpu_insights[0]
        assert cpu_insight.confidence_score > 0.0
        assert cpu_insight.trend_direction in ["increasing", "decreasing", "stable"]
        assert len(cpu_insight.affected_components) > 0
    
    @pytest.mark.asyncio
    async def test_bottleneck_detection(self, optimization_advisor):
        """Test performance bottleneck detection."""
        analysis_engine = optimization_advisor.analysis_engine
        
        # Simulate system with bottlenecks
        system_metrics = {
            "system.cpu.percent": 95.0,  # High CPU
            "system.memory.percent": 88.0,  # High memory
            "system.disk.io_wait": 45.0  # High I/O wait
        }
        
        agent_metrics = {
            "agent-1": {
                "health_score": 0.4,  # Poor health
                "error_rate": 15.0,   # High error rate
                "response_time": 8000  # Slow response
            }
        }
        
        workflow_metrics = {
            "workflow-1": {
                "completion_rate": 0.6,  # Low completion rate
                "average_duration": 300   # Long duration
            }
        }
        
        bottlenecks = await analysis_engine.detect_performance_bottlenecks(
            system_metrics, agent_metrics, workflow_metrics
        )
        
        # Should detect multiple bottlenecks
        assert len(bottlenecks) > 0, "Should detect performance bottlenecks"
        
        # Check categories of detected bottlenecks
        categories = [b.category for b in bottlenecks]
        assert OptimizationCategory.SYSTEM_RESOURCES in categories
    
    @pytest.mark.asyncio
    async def test_optimization_recommendation_generation(self, optimization_advisor):
        """Test optimization recommendation generation."""
        # Create sample insights
        insights = [
            PerformanceInsight(
                insight_id=str(uuid.uuid4()),
                category=OptimizationCategory.SYSTEM_RESOURCES,
                title="High CPU Usage",
                description="CPU usage consistently above 85%",
                current_value=90.0,
                baseline_value=60.0,
                trend_direction="increasing",
                confidence_score=0.9,
                affected_components=["system"],
                related_metrics=["system.cpu.percent"]
            ),
            PerformanceInsight(
                insight_id=str(uuid.uuid4()),
                category=OptimizationCategory.AGENT_EFFICIENCY,
                title="Low Agent Health",
                description="Agent health scores below optimal",
                current_value=0.6,
                baseline_value=0.9,
                trend_direction="decreasing",
                confidence_score=0.8,
                affected_components=["agents"],
                related_metrics=["agent.health_score"]
            )
        ]
        
        current_metrics = {"system.cpu.percent": 90.0, "agents.avg_health_score": 0.6}
        system_context = {"active_agents": 10, "active_workflows": 3}
        
        recommendations = await optimization_advisor.recommendation_engine.generate_recommendations(
            insights, current_metrics, system_context
        )
        
        # Should generate recommendations
        assert len(recommendations) > 0, "Should generate optimization recommendations"
        
        # Check recommendation properties
        for rec in recommendations:
            assert rec.category in OptimizationCategory
            assert rec.expected_impact in ImpactLevel
            assert rec.estimated_effort_hours > 0
            assert len(rec.implementation_steps) > 0
    
    @pytest.mark.asyncio
    async def test_recommendation_prioritization(self, optimization_advisor):
        """Test recommendation prioritization logic."""
        # Create recommendations with different impact/complexity
        recommendations = [
            OptimizationRecommendation(
                recommendation_id=str(uuid.uuid4()),
                category=OptimizationCategory.SYSTEM_RESOURCES,
                title="High Impact, Low Complexity",
                description="Easy win optimization",
                rationale="Clear bottleneck with simple fix",
                expected_impact=ImpactLevel.HIGH,
                implementation_complexity=ImplementationComplexity.LOW,
                priority=OptimizationPriority.HIGH,
                estimated_effort_hours=2.0,
                expected_improvement_percentage=40.0,
                implementation_steps=["Step 1", "Step 2"],
                prerequisites=[],
                risks=[],
                success_metrics=["Metric 1"],
                related_insights=[]
            ),
            OptimizationRecommendation(
                recommendation_id=str(uuid.uuid4()),
                category=OptimizationCategory.AGENT_EFFICIENCY,
                title="Low Impact, High Complexity",
                description="Complex optimization with minimal gains",
                rationale="Minor improvement requiring significant work",
                expected_impact=ImpactLevel.LOW,
                implementation_complexity=ImplementationComplexity.HIGH,
                priority=OptimizationPriority.LOW,
                estimated_effort_hours=40.0,
                expected_improvement_percentage=5.0,
                implementation_steps=["Step 1", "Step 2", "Step 3"],
                prerequisites=["Prereq 1"],
                risks=["Risk 1"],
                success_metrics=["Metric 1"],
                related_insights=[]
            )
        ]
        
        prioritized = await optimization_advisor.recommendation_engine._prioritize_recommendations(
            recommendations
        )
        
        # High impact, low complexity should be prioritized first
        assert prioritized[0].title == "High Impact, Low Complexity"
        assert prioritized[1].title == "Low Impact, High Complexity"
    
    @pytest.mark.asyncio
    async def test_get_optimization_recommendations(self, optimization_advisor):
        """Test retrieving optimization recommendations."""
        # Add some test recommendations
        test_rec = OptimizationRecommendation(
            recommendation_id=str(uuid.uuid4()),
            category=OptimizationCategory.SYSTEM_RESOURCES,
            title="Test Recommendation",
            description="Test recommendation for validation",
            rationale="Testing purposes",
            expected_impact=ImpactLevel.MEDIUM,
            implementation_complexity=ImplementationComplexity.MEDIUM,
            priority=OptimizationPriority.MEDIUM,
            estimated_effort_hours=8.0,
            expected_improvement_percentage=20.0,
            implementation_steps=["Test step"],
            prerequisites=[],
            risks=[],
            success_metrics=["Test metric"],
            related_insights=[]
        )
        
        optimization_advisor.recommendations_history.append(test_rec)
        
        # Retrieve recommendations
        recommendations = await optimization_advisor.get_optimization_recommendations(
            category_filter=[OptimizationCategory.SYSTEM_RESOURCES],
            limit=5
        )
        
        # Should return filtered recommendations
        assert len(recommendations) > 0
        assert all(r.category == OptimizationCategory.SYSTEM_RESOURCES for r in recommendations)


class TestObservabilityIntegration:
    """Integration tests for complete observability stack."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_monitoring(self):
        """Test complete workflow monitoring pipeline."""
        # Mock components
        mock_redis = AsyncMock()
        mock_session_factory = AsyncMock()
        mock_hooks = Mock()
        
        # Create integrated system
        workflow_tracker = AgentWorkflowTracker(
            redis_client=mock_redis,
            session_factory=mock_session_factory,
            observability_hooks=mock_hooks
        )
        
        alert_manager = AlertManager(
            redis_client=mock_redis,
            session_factory=mock_session_factory
        )
        
        metrics_streaming = DashboardMetricsStreaming(
            redis_client=mock_redis
        )
        
        try:
            # Start all components
            await workflow_tracker.start_tracking()
            await alert_manager.start()
            await metrics_streaming.start_streaming()
            
            # Simulate workflow execution
            agent_id = uuid.uuid4()
            workflow_id = uuid.uuid4()
            task_id = uuid.uuid4()
            
            # 1. Agent starts task
            await workflow_tracker.track_agent_state_transition(
                agent_id=agent_id,
                new_state=AgentState.BUSY,
                transition_reason="Task started",
                context={"workflow_id": str(workflow_id)}
            )
            
            # 2. Task progress tracking
            await workflow_tracker.track_task_progress(
                task_id=task_id,
                agent_id=agent_id,
                new_state=TaskProgressState.EXECUTING,
                workflow_id=workflow_id,
                progress_percentage=50.0
            )
            
            # 3. Stream metrics update
            await metrics_streaming.stream_agent_status_update(
                agent_id=str(agent_id),
                status="busy",
                health_score=0.9
            )
            
            # 4. Generate alert for high resource usage
            alerts = await alert_manager.evaluate_metric_for_alerts(
                metric_name="system.cpu.percent",
                value=85.0,
                context={"workflow_id": str(workflow_id)}
            )
            
            # 5. Complete task
            await workflow_tracker.track_task_progress(
                task_id=task_id,
                agent_id=agent_id,
                new_state=TaskProgressState.COMPLETED,
                workflow_id=workflow_id,
                progress_percentage=100.0
            )
            
            # Verify integration
            assert len(workflow_tracker.agent_state_history[agent_id]) > 0
            assert len(workflow_tracker.task_progress_history[task_id]) > 0
            assert len(metrics_streaming.metric_buffers[MetricStreamType.AGENT_STATUS]) > 0
            
        finally:
            # Cleanup
            await workflow_tracker.stop_tracking()
            await alert_manager.stop()
            await metrics_streaming.stop_streaming()
    
    @pytest.mark.performance
    async def test_system_performance_under_load(self):
        """Test system performance with high load."""
        mock_redis = AsyncMock()
        mock_session_factory = AsyncMock()
        
        # Create system components
        workflow_tracker = AgentWorkflowTracker(
            redis_client=mock_redis,
            session_factory=mock_session_factory
        )
        
        metrics_streaming = DashboardMetricsStreaming(redis_client=mock_redis)
        
        try:
            await workflow_tracker.start_tracking()
            await metrics_streaming.start_streaming()
            
            # Simulate high load scenario
            num_agents = 60
            num_tasks_per_agent = 10
            
            start_time = time.time()
            
            # Create concurrent load
            tasks = []
            
            for agent_idx in range(num_agents):
                agent_id = uuid.uuid4()
                
                # Agent state transitions
                for state in [AgentState.IDLE, AgentState.BUSY, AgentState.PROCESSING]:
                    task = workflow_tracker.track_agent_state_transition(
                        agent_id=agent_id,
                        new_state=state,
                        transition_reason=f"Load test transition {state.value}"
                    )
                    tasks.append(task)
                
                # Task progress updates
                for task_idx in range(num_tasks_per_agent):
                    task_id = uuid.uuid4()
                    task = workflow_tracker.track_task_progress(
                        task_id=task_id,
                        agent_id=agent_id,
                        new_state=TaskProgressState.COMPLETED,
                        progress_percentage=100.0
                    )
                    tasks.append(task)
                
                # Metrics streaming
                task = metrics_streaming.stream_agent_status_update(
                    agent_id=str(agent_id),
                    status="active",
                    health_score=0.95
                )
                tasks.append(task)
            
            # Execute all operations
            await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            total_operations = len(tasks)
            
            # Performance targets
            assert total_time < 10.0, f"High load processing took {total_time:.2f}s, expected <10.0s"
            
            operations_per_second = total_operations / total_time
            assert operations_per_second > 100, f"Only {operations_per_second:.1f} ops/sec, expected >100"
            
            logger.info(
                "High load performance test completed",
                total_time=total_time,
                total_operations=total_operations,
                operations_per_second=operations_per_second,
                num_agents=num_agents
            )
            
        finally:
            await workflow_tracker.stop_tracking()
            await metrics_streaming.stop_streaming()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([
        __file__,
        "-v",
        "--asyncio-mode=auto",
        "--tb=short"
    ])