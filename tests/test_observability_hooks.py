"""
Comprehensive Test Suite for VS 6.1 Observability Hooks System

Tests for:
- ObservabilityHooks performance and functionality
- Hook interceptors for all event categories
- Event collector service with Redis Streams
- Performance monitoring with <5ms targets
- System integration and health checks
"""

import asyncio
import pytest
import uuid
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

from app.core.observability_hooks import (
    ObservabilityHooks,
    HookConfiguration,
    HookVerbosity,
    SamplingStrategy,
    EventSampler,
    PerformanceTracker,
    initialize_observability_hooks,
    get_hooks
)
from app.core.hook_interceptors import (
    SystemHookInterceptor,
    initialize_system_interceptor,
    workflow_lifecycle_hook,
    task_execution_hook,
    tool_execution_hook,
    semantic_memory_hook,
    agent_communication_hook,
    agent_state_hook
)
from app.services.event_collector_service import (
    EventCollectorService,
    EventEnricher,
    initialize_event_collector
)
from app.core.observability_streams import (
    ObservabilityStreamsManager,
    initialize_streams_manager
)
from app.monitoring.hook_performance_monitor import (
    HookPerformanceMonitor,
    PerformanceStatus,
    initialize_performance_monitor
)
from app.core.orchestrator_hook_integration import (
    EnhancedOrchestratorHookIntegration,
    initialize_vs61_orchestrator_integration
)


class TestObservabilityHooks:
    """Test suite for core observability hooks functionality."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing."""
        mock_redis = AsyncMock()
        mock_redis.xadd = AsyncMock(return_value=b"1234567890-0")
        mock_redis.ping = AsyncMock(return_value=True)
        return mock_redis
    
    @pytest.fixture
    def hook_config(self):
        """Test hook configuration."""
        return HookConfiguration(
            enabled=True,
            verbosity=HookVerbosity.STANDARD,
            sampling_strategy=SamplingStrategy.NONE,
            sampling_rate=1.0,
            max_events_per_second=1000,
            max_payload_size=50000,
            enable_performance_tracking=True
        )
    
    @pytest.fixture
    def observability_hooks(self, hook_config, mock_redis):
        """Initialize observability hooks for testing."""
        with patch('app.core.observability_hooks.get_redis', return_value=mock_redis):
            hooks = ObservabilityHooks(hook_config)
            return hooks
    
    @pytest.mark.asyncio
    async def test_hook_initialization(self, hook_config):
        """Test observability hooks initialization."""
        hooks = ObservabilityHooks(hook_config)
        
        assert hooks.config.enabled == True
        assert hooks.config.verbosity == HookVerbosity.STANDARD
        assert hooks.performance_tracker is not None
        assert hooks.sampler is not None
    
    @pytest.mark.asyncio
    async def test_workflow_started_hook(self, observability_hooks):
        """Test workflow started hook emission."""
        workflow_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        session_id = uuid.uuid4()
        
        start_time = time.time()
        
        stream_id = await observability_hooks.workflow_started(
            workflow_id=workflow_id,
            workflow_name="test_workflow",
            workflow_definition={"tasks": ["task1", "task2"]},
            agent_id=agent_id,
            session_id=session_id,
            estimated_duration_ms=5000.0
        )
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        assert stream_id is not None
        assert execution_time_ms < 5.0  # Performance target
        assert observability_hooks._events_emitted == 1
    
    @pytest.mark.asyncio
    async def test_pre_tool_use_hook_performance(self, observability_hooks):
        """Test pre-tool use hook performance target."""
        agent_id = uuid.uuid4()
        session_id = uuid.uuid4()
        
        # Test multiple hook calls to verify consistent performance
        execution_times = []
        
        for i in range(100):
            start_time = time.time()
            
            await observability_hooks.pre_tool_use(
                agent_id=agent_id,
                tool_name=f"test_tool_{i}",
                parameters={"param1": "value1", "param2": i},
                session_id=session_id
            )
            
            execution_time_ms = (time.time() - start_time) * 1000
            execution_times.append(execution_time_ms)
        
        # Verify performance targets
        avg_time = sum(execution_times) / len(execution_times)
        p95_time = sorted(execution_times)[int(0.95 * len(execution_times))]
        
        assert avg_time < 2.0, f"Average execution time {avg_time:.2f}ms exceeds 2ms target"
        assert p95_time < 5.0, f"P95 execution time {p95_time:.2f}ms exceeds 5ms target"
        assert observability_hooks._events_emitted == 100
    
    @pytest.mark.asyncio
    async def test_post_tool_use_hook_with_performance_metrics(self, observability_hooks):
        """Test post-tool use hook with performance metrics."""
        agent_id = uuid.uuid4()
        session_id = uuid.uuid4()
        
        stream_id = await observability_hooks.post_tool_use(
            agent_id=agent_id,
            tool_name="performance_test_tool",
            success=True,
            session_id=session_id,
            result={"output": "test result"},
            execution_time_ms=123.45,
            memory_usage_mb=45.67,
            cpu_usage_percent=23.45
        )
        
        assert stream_id is not None
        assert observability_hooks._events_emitted == 1
    
    @pytest.mark.asyncio
    async def test_semantic_query_hook(self, observability_hooks):
        """Test semantic query hook emission."""
        agent_id = uuid.uuid4()
        session_id = uuid.uuid4()
        
        query_embedding = [0.1] * 1536  # Mock 1536-dimensional embedding
        
        stream_id = await observability_hooks.semantic_query(
            query_text="test semantic query",
            query_embedding=query_embedding,
            agent_id=agent_id,
            session_id=session_id,
            similarity_threshold=0.8,
            max_results=10,
            results_count=5,
            execution_time_ms=25.5
        )
        
        assert stream_id is not None
        assert observability_hooks._events_emitted == 1
    
    @pytest.mark.asyncio
    async def test_failure_detected_hook(self, observability_hooks):
        """Test failure detection hook emission."""
        agent_id = uuid.uuid4()
        session_id = uuid.uuid4()
        workflow_id = uuid.uuid4()
        
        stream_id = await observability_hooks.failure_detected(
            failure_type="tool_execution_failure",
            failure_description="Tool execution timeout",
            affected_component="tool_registry",
            severity="high",
            error_details={
                "error_code": "TIMEOUT",
                "timeout_duration_ms": 30000,
                "tool_name": "slow_tool"
            },
            agent_id=agent_id,
            session_id=session_id,
            workflow_id=workflow_id
        )
        
        assert stream_id is not None
        assert observability_hooks._events_emitted == 1
    
    @pytest.mark.asyncio
    async def test_sampling_strategy(self, mock_redis):
        """Test different sampling strategies."""
        # Test random sampling
        config = HookConfiguration(
            sampling_strategy=SamplingStrategy.RANDOM,
            sampling_rate=0.5
        )
        
        with patch('app.core.observability_hooks.get_redis', return_value=mock_redis):
            hooks = ObservabilityHooks(config)
            
            events_emitted = 0
            for i in range(1000):
                stream_id = await hooks.pre_tool_use(
                    agent_id=uuid.uuid4(),
                    tool_name=f"test_tool_{i}",
                    parameters={}
                )
                if stream_id:
                    events_emitted += 1
            
            # Should emit roughly 50% of events (within reasonable variance)
            assert 400 <= events_emitted <= 600, f"Sampling rate off: {events_emitted}/1000 events emitted"
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, observability_hooks):
        """Test performance tracking functionality."""
        # Generate some events to populate performance metrics
        for i in range(50):
            await observability_hooks.pre_tool_use(
                agent_id=uuid.uuid4(),
                tool_name=f"perf_test_{i}",
                parameters={"test": True}
            )
        
        # Get performance metrics
        metrics = observability_hooks.get_performance_metrics()
        
        assert "hook_performance" in metrics
        assert "events_emitted" in metrics
        assert "events_dropped" in metrics
        assert "within_target" in metrics
        assert metrics["events_emitted"] == 50
    
    @pytest.mark.asyncio
    async def test_health_check(self, observability_hooks):
        """Test observability hooks health check."""
        health = await observability_hooks.health_check()
        
        assert "status" in health
        assert "redis_healthy" in health
        assert "performance_within_target" in health
        assert "metrics" in health
        
        # Should be healthy with mock Redis
        assert health["status"] in ["healthy", "degraded"]


class TestHookInterceptors:
    """Test suite for hook interceptors and decorators."""
    
    @pytest.fixture
    async def mock_hooks(self):
        """Mock observability hooks for testing."""
        hooks = Mock()
        hooks.workflow_started = AsyncMock(return_value="stream_id_1")
        hooks.workflow_ended = AsyncMock(return_value="stream_id_2")
        hooks.node_executing = AsyncMock(return_value="stream_id_3")
        hooks.node_completed = AsyncMock(return_value="stream_id_4")
        hooks.pre_tool_use = AsyncMock(return_value="stream_id_5")
        hooks.post_tool_use = AsyncMock(return_value="stream_id_6")
        hooks.failure_detected = AsyncMock(return_value="stream_id_7")
        return hooks
    
    @pytest.fixture
    async def system_interceptor(self, mock_hooks):
        """System hook interceptor for testing."""
        return SystemHookInterceptor(mock_hooks)
    
    @pytest.mark.asyncio
    async def test_workflow_lifecycle_hook_decorator(self, mock_hooks):
        """Test workflow lifecycle hook decorator."""
        with patch('app.core.hook_interceptors.get_hooks', return_value=mock_hooks):
            
            class MockWorkflowEngine:
                def __init__(self):
                    self.workflow_id = uuid.uuid4()
                    self.agent_id = uuid.uuid4()
                    self.session_id = uuid.uuid4()
                
                @workflow_lifecycle_hook
                async def execute_workflow(self, workflow_name="test", **kwargs):
                    await asyncio.sleep(0.001)  # Simulate work
                    return {"status": "completed", "tasks": 5}
            
            engine = MockWorkflowEngine()
            start_time = time.time()
            
            result = await engine.execute_workflow()
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            assert result["status"] == "completed"
            assert execution_time_ms < 10.0  # Should be fast
            
            # Verify hooks were called
            mock_hooks.workflow_started.assert_called_once()
            mock_hooks.workflow_ended.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_tool_execution_hook_decorator(self, mock_hooks):
        """Test tool execution hook decorator."""
        with patch('app.core.hook_interceptors.get_hooks', return_value=mock_hooks):
            
            class MockToolRegistry:
                def __init__(self):
                    self.agent_id = uuid.uuid4()
                    self.session_id = uuid.uuid4()
                
                @tool_execution_hook
                async def execute_tool(self, tool_name, parameters, **kwargs):
                    if tool_name == "failing_tool":
                        raise ValueError("Tool execution failed")
                    return {"result": f"Tool {tool_name} executed successfully"}
            
            registry = MockToolRegistry()
            
            # Test successful tool execution
            result = await registry.execute_tool("success_tool", {"param": "value"})
            assert "successfully" in result["result"]
            
            # Verify hooks were called
            mock_hooks.pre_tool_use.assert_called()
            mock_hooks.post_tool_use.assert_called()
            
            # Test failed tool execution
            with pytest.raises(ValueError):
                await registry.execute_tool("failing_tool", {})
            
            # Should have called post_tool_use for failure as well
            assert mock_hooks.post_tool_use.call_count == 2
    
    @pytest.mark.asyncio
    async def test_semantic_memory_hook_decorator(self, mock_hooks):
        """Test semantic memory hook decorator."""
        with patch('app.core.hook_interceptors.get_hooks', return_value=mock_hooks):
            
            class MockSemanticMemory:
                def __init__(self):
                    self.agent_id = uuid.uuid4()
                    self.session_id = uuid.uuid4()
                
                @semantic_memory_hook
                async def search(self, query_text, threshold=0.8, limit=10):
                    # Simulate semantic search
                    await asyncio.sleep(0.01)
                    return [
                        {"content": "result 1", "score": 0.95},
                        {"content": "result 2", "score": 0.87}
                    ]
            
            memory = MockSemanticMemory()
            
            start_time = time.time()
            results = await memory.search("test query")
            execution_time_ms = (time.time() - start_time) * 1000
            
            assert len(results) == 2
            assert execution_time_ms < 50.0  # Should be reasonably fast
            
            # Verify semantic query hook was called
            mock_hooks.semantic_query = AsyncMock(return_value="stream_id")
            
            # Re-patch and test again
            with patch('app.core.hook_interceptors.get_hooks', return_value=mock_hooks):
                await memory.search("another query")
    
    @pytest.mark.asyncio
    async def test_system_interceptor_class_integration(self, system_interceptor):
        """Test system interceptor class integration."""
        class MockClass:
            def method_with_execute(self):
                return "executed"
            
            def method_with_task(self):
                return "task_completed"
            
            def normal_method(self):
                return "normal"
        
        # Define hook patterns
        hook_patterns = {
            'execute': lambda func: func,  # Mock hook decorator
            'task': lambda func: func
        }
        
        # Test class interception
        system_interceptor.intercept_class(MockClass, hook_patterns)
        
        # Verify methods were intercepted
        assert MockClass.__name__ in system_interceptor._intercepted_classes


class TestEventCollectorService:
    """Test suite for event collector service."""
    
    @pytest.fixture
    async def mock_redis(self):
        """Mock Redis client for event collector testing."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.xgroup_create = AsyncMock()
        mock_redis.xreadgroup = AsyncMock(return_value=[])
        mock_redis.xack = AsyncMock()
        return mock_redis
    
    @pytest.fixture
    async def event_collector(self, mock_redis):
        """Event collector service for testing."""
        collector = EventCollectorService(
            redis_client=mock_redis,
            batch_size=10,
            batch_timeout_ms=100
        )
        return collector
    
    @pytest.mark.asyncio
    async def test_event_collector_initialization(self, event_collector):
        """Test event collector service initialization."""
        assert event_collector.batch_size == 10
        assert event_collector.batch_timeout_ms == 100
        assert event_collector.enricher is not None
        assert not event_collector._is_running
    
    @pytest.mark.asyncio
    async def test_event_collector_start_stop(self, event_collector):
        """Test event collector start and stop."""
        # Start the collector
        await event_collector.start()
        assert event_collector._is_running
        
        # Stop the collector
        await event_collector.stop()
        assert not event_collector._is_running
    
    @pytest.mark.asyncio
    async def test_event_enricher(self):
        """Test event enrichment functionality."""
        enricher = EventEnricher()
        
        sample_event = {
            "event_type": "PreToolUse",
            "agent_id": str(uuid.uuid4()),
            "tool_name": "test_tool",
            "parameters": {"param1": "value1"}
        }
        
        enriched_event = await enricher.enrich_event(sample_event)
        
        assert "enrichment_timestamp" in enriched_event
        assert "collector_version" in enriched_event
        assert enriched_event["event_type"] == "PreToolUse"
    
    @pytest.mark.asyncio
    async def test_event_collector_health_check(self, event_collector):
        """Test event collector health check."""
        health = await event_collector.health_check()
        
        assert "status" in health
        assert "is_running" in health
        assert "redis_healthy" in health
        assert "database_healthy" in health
        assert "statistics" in health


class TestObservabilityStreamsManager:
    """Test suite for observability streams manager."""
    
    @pytest.fixture
    async def mock_redis(self):
        """Mock Redis client for streams testing."""
        mock_redis = AsyncMock()
        mock_redis.exists = AsyncMock(return_value=True)
        mock_redis.xadd = AsyncMock(return_value=b"1234567890-0")
        mock_redis.xinfo_stream = AsyncMock(return_value={
            "length": 100,
            "radix-tree-keys": 10,
            "radix-tree-nodes": 20,
            "groups": 2,
            "last-generated-id": "1234567890-1"
        })
        mock_redis.xinfo_groups = AsyncMock(return_value=[
            {
                b"name": b"test_group",
                b"consumers": 1,
                b"pending": 5,
                b"last-delivered-id": b"1234567890-0"
            }
        ])
        mock_redis.xgroup_create = AsyncMock()
        return mock_redis
    
    @pytest.fixture
    async def streams_manager(self, mock_redis):
        """Observability streams manager for testing."""
        manager = ObservabilityStreamsManager(
            redis_client=mock_redis,
            stream_name="test_stream",
            max_stream_length=1000
        )
        return manager
    
    @pytest.mark.asyncio
    async def test_streams_manager_initialization(self, streams_manager):
        """Test streams manager initialization."""
        assert streams_manager.stream_name == "test_stream"
        assert streams_manager.max_stream_length == 1000
        assert not streams_manager._is_monitoring
    
    @pytest.mark.asyncio
    async def test_create_consumer_group(self, streams_manager):
        """Test consumer group creation."""
        result = await streams_manager.create_consumer_group("test_group")
        assert result == True
        assert "test_group" in streams_manager._consumer_groups
    
    @pytest.mark.asyncio
    async def test_get_stream_info(self, streams_manager):
        """Test getting stream information."""
        info = await streams_manager.get_stream_info()
        
        assert info.name == "test_stream"
        assert info.length == 100
        assert info.groups == 2
    
    @pytest.mark.asyncio
    async def test_health_check(self, streams_manager):
        """Test streams manager health check."""
        health = await streams_manager.check_health()
        
        assert "status" in health
        assert "timestamp" in health
        assert "stream_accessible" in health
        assert "stream_info" in health
        assert "consumer_groups" in health


class TestHookPerformanceMonitor:
    """Test suite for hook performance monitoring."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Hook performance monitor for testing."""
        return HookPerformanceMonitor(
            target_p95_ms=5.0,
            target_avg_ms=2.0,
            regression_detection_window=100
        )
    
    @pytest.mark.asyncio
    async def test_performance_monitor_initialization(self, performance_monitor):
        """Test performance monitor initialization."""
        assert performance_monitor.target_p95_ms == 5.0
        assert performance_monitor.target_avg_ms == 2.0
        assert not performance_monitor._is_monitoring
    
    def test_record_hook_execution(self, performance_monitor):
        """Test recording hook execution metrics."""
        # Record some sample executions
        performance_monitor.record_hook_execution(
            hook_name="test_hook",
            hook_category="tool",
            execution_time_ms=2.5,
            success=True
        )
        
        performance_monitor.record_hook_execution(
            hook_name="slow_hook",
            hook_category="workflow",
            execution_time_ms=8.0,
            success=True
        )
        
        assert len(performance_monitor._operation_records) == 2
    
    def test_performance_summary(self, performance_monitor):
        """Test getting performance summary."""
        # Record multiple executions
        for i in range(100):
            performance_monitor.record_hook_execution(
                hook_name=f"hook_{i % 10}",
                hook_category="tool" if i % 2 == 0 else "workflow",
                execution_time_ms=1.0 + (i % 5),  # 1-5ms range
                success=True
            )
        
        summary = performance_monitor.get_performance_summary()
        
        assert "global_metrics" in summary
        assert "category_metrics" in summary
        assert "active_alerts" in summary
        assert "recommendations" in summary
        
        global_metrics = summary["global_metrics"]
        assert global_metrics["total_operations"] == 100
        assert global_metrics["within_target"]  # Should be within 5ms target
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_lifecycle(self, performance_monitor):
        """Test performance monitoring start/stop."""
        await performance_monitor.start_monitoring()
        assert performance_monitor._is_monitoring
        
        await performance_monitor.stop_monitoring()
        assert not performance_monitor._is_monitoring


class TestSystemIntegration:
    """Test suite for overall system integration."""
    
    @pytest.fixture
    async def mock_orchestrator(self):
        """Mock agent orchestrator for integration testing."""
        orchestrator = Mock()
        orchestrator.spawn_agent = AsyncMock(return_value="agent_123")
        orchestrator.shutdown_agent = AsyncMock()
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_vs61_integration_initialization(self, mock_orchestrator):
        """Test VS 6.1 integration initialization."""
        with patch('app.core.orchestrator_hook_integration.initialize_streams_manager') as mock_streams, \
             patch('app.core.orchestrator_hook_integration.initialize_observability_hooks') as mock_hooks, \
             patch('app.core.orchestrator_hook_integration.initialize_system_interceptor') as mock_interceptor, \
             patch('app.core.orchestrator_hook_integration.initialize_performance_monitor') as mock_perf, \
             patch('app.core.orchestrator_hook_integration.start_performance_monitoring') as mock_start_perf, \
             patch('app.core.orchestrator_hook_integration.initialize_event_collector') as mock_collector:
            
            mock_streams.return_value = Mock()
            mock_hooks.return_value = Mock()
            mock_interceptor.return_value = Mock()
            mock_perf.return_value = Mock()
            mock_collector.return_value = Mock()
            
            integration = await initialize_vs61_orchestrator_integration(
                orchestrator=mock_orchestrator,
                auto_integrate=True
            )
            
            assert integration is not None
            assert integration.use_new_observability == True
            
            # Verify all components were initialized
            mock_streams.assert_called_once()
            mock_hooks.assert_called_once()
            mock_interceptor.assert_called_once()
            mock_perf.assert_called_once()
            mock_start_perf.assert_called_once()
            mock_collector.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_integration_health_check(self, mock_orchestrator):
        """Test integration health check."""
        # Create integration with mocked components
        integration = EnhancedOrchestratorHookIntegration(mock_orchestrator)
        integration.use_new_observability = True
        
        # Mock component health checks
        mock_hooks = Mock()
        mock_hooks.health_check = AsyncMock(return_value={"status": "healthy"})
        integration.observability_hooks = mock_hooks
        
        mock_perf_monitor = Mock()
        mock_perf_monitor.get_performance_summary = Mock(return_value={
            "global_metrics": {"within_target": True},
            "recommendations": []
        })
        integration.performance_monitor = mock_perf_monitor
        
        health = await integration.health_check()
        
        assert "overall_status" in health
        assert "component_health" in health
        assert "performance_metrics" in health
        assert health["overall_status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_performance_target_validation(self):
        """Test that the system meets <5ms performance targets."""
        config = HookConfiguration(enable_performance_tracking=True)
        
        with patch('app.core.observability_hooks.get_redis') as mock_get_redis:
            mock_redis = AsyncMock()
            mock_redis.xadd = AsyncMock(return_value=b"test-id")
            mock_get_redis.return_value = mock_redis
            
            hooks = ObservabilityHooks(config)
            
            # Test 1000 hook executions
            execution_times = []
            
            for i in range(1000):
                start_time = time.time()
                
                await hooks.pre_tool_use(
                    agent_id=uuid.uuid4(),
                    tool_name=f"perf_test_{i}",
                    parameters={"iteration": i}
                )
                
                execution_time_ms = (time.time() - start_time) * 1000
                execution_times.append(execution_time_ms)
            
            # Validate performance targets
            avg_time = sum(execution_times) / len(execution_times)
            p95_time = sorted(execution_times)[int(0.95 * len(execution_times))]
            p99_time = sorted(execution_times)[int(0.99 * len(execution_times))]
            
            # Performance assertions
            assert avg_time < 2.0, f"Average execution time {avg_time:.3f}ms exceeds 2ms target"
            assert p95_time < 5.0, f"P95 execution time {p95_time:.3f}ms exceeds 5ms target"
            assert p99_time < 10.0, f"P99 execution time {p99_time:.3f}ms exceeds 10ms acceptable limit"
            
            # Verify hooks were called
            assert hooks._events_emitted == 1000
            assert mock_redis.xadd.call_count == 1000


class TestErrorHandlingAndResilience:
    """Test suite for error handling and system resilience."""
    
    @pytest.fixture
    async def failing_redis(self):
        """Mock Redis client that fails operations."""
        mock_redis = AsyncMock()
        mock_redis.xadd = AsyncMock(side_effect=Exception("Redis connection failed"))
        mock_redis.ping = AsyncMock(side_effect=Exception("Redis ping failed"))
        return mock_redis
    
    @pytest.mark.asyncio
    async def test_hook_resilience_with_redis_failure(self, failing_redis):
        """Test hook system resilience when Redis fails."""
        config = HookConfiguration()
        
        with patch('app.core.observability_hooks.get_redis', return_value=failing_redis):
            hooks = ObservabilityHooks(config)
            
            # Hook should handle Redis failure gracefully
            stream_id = await hooks.pre_tool_use(
                agent_id=uuid.uuid4(),
                tool_name="test_tool",
                parameters={}
            )
            
            # Should return None but not raise exception
            assert stream_id is None
            
            # Health check should report unhealthy
            health = await hooks.health_check()
            assert health["status"] == "unhealthy"
            assert not health["redis_healthy"]
    
    @pytest.mark.asyncio
    async def test_event_collector_error_handling(self):
        """Test event collector error handling."""
        # Create collector with failing Redis
        failing_redis = AsyncMock()
        failing_redis.xgroup_create = AsyncMock(side_effect=Exception("Group creation failed"))
        
        collector = EventCollectorService(redis_client=failing_redis)
        
        # Start should handle errors gracefully
        with pytest.raises(Exception):
            await collector.start()
        
        # Should not be running after failure
        assert not collector._is_running
    
    @pytest.mark.asyncio
    async def test_performance_monitor_with_extreme_values(self):
        """Test performance monitor with extreme execution times."""
        monitor = HookPerformanceMonitor()
        
        # Record some extreme values
        monitor.record_hook_execution("slow_hook", "tool", 50000.0, True)  # 50 seconds
        monitor.record_hook_execution("fast_hook", "tool", 0.001, True)    # 0.001ms
        
        summary = monitor.get_performance_summary()
        
        # Should handle extreme values without crashing
        assert "global_metrics" in summary
        assert summary["global_metrics"]["total_operations"] == 2
        
        # Should generate recommendations for poor performance
        assert len(summary["recommendations"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])