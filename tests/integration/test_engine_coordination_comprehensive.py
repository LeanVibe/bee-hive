"""
Epic 2 Phase 2: Comprehensive EngineCoordinationLayer Tests

Tests for the EngineCoordinationLayer that consolidates 26+ engines:
- Workflow engine integration and performance (<2s compilation)
- Task execution engine performance (<100ms assignment)
- Communication engine functionality
- Engine request routing and coordination
- Performance validation and benchmarks
- Error handling and resilience

Isolated testing approach without complex ML/analytics dependencies.
"""

import pytest
import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Mock complex dependencies
@pytest.fixture(autouse=True)
def mock_complex_dependencies():
    """Mock complex dependencies that cause import issues."""
    modules_to_mock = {
        'sklearn': Mock(),
        'sklearn.base': Mock(),
        'sklearn.preprocessing': Mock(),
        'scipy': Mock(),
        'scipy.stats': Mock(),
        'numpy': Mock(),
        'pandas': Mock(),
        'structlog': Mock()
    }
    
    with patch.dict('sys.modules', modules_to_mock):
        # Mock structlog specifically for the engine module
        mock_structlog = Mock()
        mock_logger = Mock()
        mock_logger.info = Mock()
        mock_logger.error = Mock()
        mock_logger.warning = Mock()
        mock_logger.debug = Mock()
        mock_structlog.get_logger.return_value = mock_logger
        
        with patch('app.core.engines.consolidated_engine.structlog', mock_structlog):
            yield


@pytest.fixture
def engine_config():
    """Comprehensive engine configuration for testing."""
    return {
        'test_mode': True,
        'performance_targets': {
            'workflow_compilation_ms': 2000,
            'task_assignment_ms': 100,
            'communication_delivery_ms': 50
        },
        'engine_settings': {
            'workflow_engine': {
                'max_concurrent_workflows': 50,
                'timeout_seconds': 30
            },
            'task_execution_engine': {
                'max_concurrent_tasks': 100,
                'thread_pool_size': 10
            },
            'communication_engine': {
                'max_subscribers_per_type': 1000,
                'message_queue_size': 500
            }
        },
        'integration_settings': {
            'enable_metrics': True,
            'enable_health_checks': True,
            'enable_performance_tracking': True
        }
    }


@pytest.fixture
async def engine_coordinator(engine_config):
    """Create EngineCoordinationLayer for testing."""
    from app.core.engines.consolidated_engine import EngineCoordinationLayer
    
    coordinator = EngineCoordinationLayer(engine_config)
    await coordinator.initialize()
    return coordinator


class TestEngineCoordinationLayerCore:
    """Test core EngineCoordinationLayer functionality."""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self, engine_coordinator):
        """Test proper engine initialization."""
        # Verify all engines are initialized
        assert engine_coordinator.workflow_engine is not None
        assert engine_coordinator.task_engine is not None
        assert engine_coordinator.communication_engine is not None
        
        # Verify engine registry
        assert len(engine_coordinator._engines) == 3
        assert 'workflow' in [engine_type.value for engine_type in engine_coordinator._engines.keys()]
        assert 'task_execution' in [engine_type.value for engine_type in engine_coordinator._engines.keys()]
        assert 'communication' in [engine_type.value for engine_type in engine_coordinator._engines.keys()]
    
    @pytest.mark.asyncio
    async def test_health_check_comprehensive(self, engine_coordinator):
        """Test comprehensive health check functionality."""
        health = await engine_coordinator.health_check()
        
        # Verify health check structure
        required_engines = ['workflow_engine', 'task_execution_engine', 'communication_engine']
        for engine in required_engines:
            assert engine in health
            assert 'status' in health[engine]
            assert 'uptime_seconds' in health[engine]
            assert 'processed_count' in health[engine]
            assert 'error_count' in health[engine]
            assert 'average_processing_time_ms' in health[engine]
        
        assert 'overall_health' in health
        assert health['overall_health'] == 'healthy'
    
    @pytest.mark.asyncio
    async def test_status_reporting(self, engine_coordinator):
        """Test comprehensive status reporting."""
        status = await engine_coordinator.get_status()
        
        # Verify status structure
        required_keys = [
            'workflow_engine', 'task_execution_engine', 'communication_engine',
            'total_workflows_processed', 'total_tasks_processed', 'total_messages_processed',
            'active_workflows', 'active_tasks'
        ]
        
        for key in required_keys:
            assert key in status
        
        # Verify engine statuses
        for engine_key in ['workflow_engine', 'task_execution_engine', 'communication_engine']:
            assert 'status' in status[engine_key]
            assert status[engine_key]['status'] == 'running'
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, engine_coordinator):
        """Test performance metrics collection."""
        metrics = await engine_coordinator.get_performance_metrics()
        
        # Verify metrics structure
        engine_types = ['workflow_engine', 'task_execution_engine', 'communication_engine']
        
        for engine_type in engine_types:
            assert engine_type in metrics
            engine_metrics = metrics[engine_type]
            
            assert 'total_processed' in engine_metrics
            assert 'success_rate_percent' in engine_metrics
            
            if engine_type == 'communication_engine':
                assert 'average_delivery_time_ms' in engine_metrics
            else:
                assert 'average_execution_time_ms' in engine_metrics


class TestWorkflowEngineIntegration:
    """Test consolidated workflow engine integration."""
    
    @pytest.mark.asyncio
    async def test_workflow_execution_performance(self, engine_coordinator):
        """Test workflow execution meets performance targets (<2s)."""
        from app.core.engines.consolidated_engine import WorkflowRequest
        
        # Create comprehensive workflow definition
        workflow_definition = {
            'name': 'performance_test_workflow',
            'version': '1.0',
            'steps': [
                {
                    'name': 'initialization',
                    'type': 'setup',
                    'config': {'timeout': 5}
                },
                {
                    'name': 'data_analysis',
                    'type': 'analysis',
                    'config': {'complexity': 'high'}
                },
                {
                    'name': 'coordination',
                    'type': 'coordination',
                    'config': {'agents': 5}
                },
                {
                    'name': 'communication',
                    'type': 'communication',
                    'config': {'broadcast': True}
                },
                {
                    'name': 'finalization',
                    'type': 'cleanup',
                    'config': {'persist_results': True}
                }
            ],
            'dependencies': {
                'data_analysis': ['initialization'],
                'coordination': ['data_analysis'],
                'communication': ['coordination'],
                'finalization': ['communication']
            }
        }
        
        workflow_request = WorkflowRequest(
            workflow_definition=workflow_definition,
            context={
                'user_id': 'test_user',
                'session_id': 'test_session',
                'priority': 'high',
                'tracking_enabled': True
            },
            execution_mode='async',
            priority=8
        )
        
        # Execute workflow and measure time
        start_time = time.time()
        result = await engine_coordinator.workflow_engine.execute_workflow(workflow_request)
        execution_time = (time.time() - start_time) * 1000
        
        # Validate result
        assert result is not None
        assert result.success
        assert result.workflow_id == workflow_request.workflow_id
        assert result.execution_time_ms > 0
        
        # Performance validation (<2s target)
        assert execution_time < 2000, f"Workflow execution took {execution_time}ms, should be <2000ms"
        assert result.execution_time_ms < 2000, f"Tracked execution time {result.execution_time_ms}ms exceeded target"
        
        # Validate performance tracking
        performance = engine_coordinator.workflow_engine._performance_tracker.get_performance_summary()
        if performance.get('status') != 'no_data':
            assert 'average_execution_time_ms' in performance
            assert 'success_rate' in performance
            assert performance['target_met'], "Workflow performance target should be met"
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, engine_coordinator):
        """Test concurrent workflow execution capabilities."""
        from app.core.engines.consolidated_engine import WorkflowRequest
        
        # Create multiple workflow requests
        workflow_requests = []
        for i in range(10):
            workflow_request = WorkflowRequest(
                workflow_definition={
                    'name': f'concurrent_workflow_{i}',
                    'steps': [
                        {'name': f'step_1_{i}', 'type': 'analysis'},
                        {'name': f'step_2_{i}', 'type': 'coordination'},
                        {'name': f'step_3_{i}', 'type': 'communication'}
                    ]
                },
                context={'concurrent_batch': True, 'workflow_index': i}
            )
            workflow_requests.append(workflow_request)
        
        # Execute workflows concurrently
        start_time = time.time()
        tasks = [
            engine_coordinator.workflow_engine.execute_workflow(request) 
            for request in workflow_requests
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = (time.time() - start_time) * 1000
        
        # Validate results
        successful_results = [
            r for r in results 
            if not isinstance(r, Exception) and hasattr(r, 'success') and r.success
        ]
        
        assert len(successful_results) >= 8, f"Expected at least 8 successful workflows, got {len(successful_results)}"
        
        # Performance validation for concurrent execution
        avg_time_per_workflow = total_time / len(workflow_requests)
        assert avg_time_per_workflow < 1000, f"Average time per concurrent workflow: {avg_time_per_workflow}ms"
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, engine_coordinator):
        """Test workflow engine error handling."""
        from app.core.engines.consolidated_engine import WorkflowRequest
        
        # Test with invalid workflow definition
        invalid_workflow = WorkflowRequest(
            workflow_definition={
                'invalid_structure': True,
                'missing_required_fields': None
            },
            context={'test_error_handling': True}
        )
        
        # Should handle error gracefully
        result = await engine_coordinator.workflow_engine.execute_workflow(invalid_workflow)
        
        assert result is not None
        assert result.workflow_id == invalid_workflow.workflow_id
        # Error handling may vary - either success with fallback or proper error tracking


class TestTaskExecutionEngineIntegration:
    """Test consolidated task execution engine integration."""
    
    @pytest.mark.asyncio
    async def test_task_execution_performance(self, engine_coordinator):
        """Test task execution meets performance targets (<100ms)."""
        from app.core.engines.consolidated_engine import TaskExecutionRequest
        
        task_types = ['general', 'workflow', 'communication', 'analysis', 'coordination']
        
        for task_type in task_types:
            task_request = TaskExecutionRequest(
                task_type=task_type,
                payload={
                    'data': f'test_data_for_{task_type}',
                    'complexity': 'medium',
                    'priority': 'high'
                },
                agent_id=f'test_agent_{task_type}',
                priority=7,
                timeout_seconds=30
            )
            
            # Execute task and measure time
            start_time = time.time()
            result = await engine_coordinator.task_engine.execute_task(task_request)
            execution_time = (time.time() - start_time) * 1000
            
            # Validate result
            assert result is not None
            assert result.success
            assert result.task_id == task_request.task_id
            assert result.execution_time_ms > 0
            
            # Performance validation (<100ms target)
            assert execution_time < 100, f"Task {task_type} execution took {execution_time}ms, should be <100ms"
            assert result.execution_time_ms < 100, f"Tracked execution time {result.execution_time_ms}ms exceeded target"
        
        # Validate overall performance tracking
        performance = engine_coordinator.task_engine._performance_tracker.get_performance_summary()
        if performance.get('status') != 'no_data':
            assert performance['target_met'], "Task execution performance target should be met"
    
    @pytest.mark.asyncio
    async def test_high_throughput_task_execution(self, engine_coordinator):
        """Test high throughput task execution."""
        from app.core.engines.consolidated_engine import TaskExecutionRequest
        
        # Create batch of diverse tasks
        task_requests = []
        task_types = ['general', 'workflow', 'communication', 'analysis', 'coordination']
        
        for i in range(50):
            task_type = task_types[i % len(task_types)]
            task_request = TaskExecutionRequest(
                task_type=task_type,
                payload={
                    'batch_id': 'throughput_test',
                    'task_index': i,
                    'data': f'batch_data_{i}'
                },
                agent_id=f'agent_{i % 5}',  # Distribute across 5 agents
                priority=5
            )
            task_requests.append(task_request)
        
        # Execute tasks with high throughput
        start_time = time.time()
        tasks = [
            engine_coordinator.task_engine.execute_task(request)
            for request in task_requests
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = (time.time() - start_time) * 1000
        
        # Validate throughput
        successful_results = [
            r for r in results 
            if not isinstance(r, Exception) and hasattr(r, 'success') and r.success
        ]
        
        assert len(successful_results) >= 45, f"Expected at least 45 successful tasks, got {len(successful_results)}"
        
        # Performance validation
        throughput = len(successful_results) / (total_time / 1000)  # tasks per second
        assert throughput > 100, f"Throughput: {throughput} tasks/sec, should be >100 tasks/sec"
    
    @pytest.mark.asyncio
    async def test_task_type_handler_coverage(self, engine_coordinator):
        """Test all task type handlers are properly registered."""
        # Verify task handlers are registered
        task_registry = engine_coordinator.task_engine._task_registry
        
        expected_handlers = ['general', 'workflow', 'communication', 'analysis', 'coordination']
        
        for handler_type in expected_handlers:
            assert handler_type in task_registry, f"Missing task handler for {handler_type}"
            assert callable(task_registry[handler_type]), f"Task handler for {handler_type} is not callable"
    
    @pytest.mark.asyncio
    async def test_task_execution_error_recovery(self, engine_coordinator):
        """Test task execution error recovery."""
        from app.core.engines.consolidated_engine import TaskExecutionRequest
        
        # Test with invalid task type
        invalid_task = TaskExecutionRequest(
            task_type='nonexistent_type',
            payload={'cause_error': True}
        )
        
        result = await engine_coordinator.task_engine.execute_task(invalid_task)
        
        # Should handle error gracefully by falling back to general handler
        assert result is not None
        assert result.task_id == invalid_task.task_id


class TestCommunicationEngineIntegration:
    """Test consolidated communication engine integration."""
    
    @pytest.mark.asyncio
    async def test_direct_message_delivery(self, engine_coordinator):
        """Test direct message delivery functionality."""
        from app.core.engines.consolidated_engine import CommunicationRequest
        
        # Test direct message
        direct_message = CommunicationRequest(
            sender_id='test_sender',
            recipient_id='test_recipient',
            message_type='direct_communication',
            payload={
                'message': 'Direct message test',
                'priority': 'high',
                'timestamp': datetime.utcnow().isoformat()
            },
            broadcast=False
        )
        
        result = await engine_coordinator.communication_engine.send_message(direct_message)
        
        # Validate result
        assert result is not None
        assert result.success
        assert result.message_id == direct_message.message_id
        assert result.recipients_reached == 1  # Direct message
        assert result.execution_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_broadcast_message_delivery(self, engine_coordinator):
        """Test broadcast message delivery functionality."""
        from app.core.engines.consolidated_engine import CommunicationRequest
        
        # First subscribe some handlers to the message type
        test_handlers = []
        for i in range(5):
            handler = AsyncMock()
            test_handlers.append(handler)
            engine_coordinator.communication_engine.subscribe('broadcast_test', handler)
        
        # Test broadcast message
        broadcast_message = CommunicationRequest(
            sender_id='system',
            message_type='broadcast_test',
            payload={
                'announcement': 'System-wide notification',
                'level': 'info',
                'timestamp': datetime.utcnow().isoformat()
            },
            broadcast=True
        )
        
        result = await engine_coordinator.communication_engine.send_message(broadcast_message)
        
        # Validate result
        assert result is not None
        assert result.success
        assert result.message_id == broadcast_message.message_id
        assert result.recipients_reached == len(test_handlers)
        assert result.execution_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_communication_performance(self, engine_coordinator):
        """Test communication performance benchmarks."""
        from app.core.engines.consolidated_engine import CommunicationRequest
        
        # Test multiple message types with performance measurement
        message_types = ['notification', 'alert', 'coordination', 'status', 'data']
        
        for message_type in message_types:
            message = CommunicationRequest(
                sender_id='performance_tester',
                recipient_id='test_recipient',
                message_type=message_type,
                payload={'test_data': f'performance_test_{message_type}'}
            )
            
            start_time = time.time()
            result = await engine_coordinator.communication_engine.send_message(message)
            execution_time = (time.time() - start_time) * 1000
            
            # Validate performance (<50ms target for communication)
            assert execution_time < 50, f"Communication {message_type} took {execution_time}ms, should be <50ms"
            assert result.success
    
    @pytest.mark.asyncio
    async def test_subscription_management(self, engine_coordinator):
        """Test subscription management functionality."""
        # Test subscribing to different message types
        message_types = ['alerts', 'notifications', 'system_events', 'user_actions']
        handlers = {}
        
        for msg_type in message_types:
            handler = AsyncMock()
            handlers[msg_type] = handler
            engine_coordinator.communication_engine.subscribe(msg_type, handler)
        
        # Verify subscriptions
        comm_engine = engine_coordinator.communication_engine
        for msg_type in message_types:
            assert msg_type in comm_engine._subscribers
            assert handlers[msg_type] in comm_engine._subscribers[msg_type]


class TestEngineRequestRouting:
    """Test unified engine request routing system."""
    
    @pytest.mark.asyncio
    async def test_workflow_request_routing(self, engine_coordinator):
        """Test routing workflow requests through coordination layer."""
        from app.core.engines.consolidated_engine import EngineRequest, EngineType
        
        workflow_request = EngineRequest(
            engine_type=EngineType.WORKFLOW,
            operation='execute_workflow',
            payload={
                'workflow_definition': {
                    'name': 'routing_test_workflow',
                    'steps': [
                        {'name': 'step1', 'type': 'analysis'},
                        {'name': 'step2', 'type': 'coordination'}
                    ]
                },
                'context': {
                    'routing_test': True,
                    'priority': 'high'
                }
            },
            priority=8
        )
        
        response = await engine_coordinator.execute_request(workflow_request)
        
        # Validate routing and execution
        assert response is not None
        assert response.success
        assert response.request_id == workflow_request.request_id
        assert response.engine_type == EngineType.WORKFLOW
        assert response.execution_time_ms > 0
        assert 'workflow_id' in response.result
    
    @pytest.mark.asyncio
    async def test_task_request_routing(self, engine_coordinator):
        """Test routing task execution requests through coordination layer."""
        from app.core.engines.consolidated_engine import EngineRequest, EngineType
        
        task_request = EngineRequest(
            engine_type=EngineType.TASK_EXECUTION,
            operation='execute_task',
            payload={
                'task_type': 'coordination',
                'payload': {
                    'coordination_data': 'test_data',
                    'agent_count': 5
                },
                'agent_id': 'routing_test_agent'
            },
            priority=7
        )
        
        response = await engine_coordinator.execute_request(task_request)
        
        # Validate routing and execution
        assert response is not None
        assert response.success
        assert response.request_id == task_request.request_id
        assert response.engine_type == EngineType.TASK_EXECUTION
        assert response.execution_time_ms > 0
        assert 'task_id' in response.result
    
    @pytest.mark.asyncio
    async def test_communication_request_routing(self, engine_coordinator):
        """Test routing communication requests through coordination layer."""
        from app.core.engines.consolidated_engine import EngineRequest, EngineType
        
        communication_request = EngineRequest(
            engine_type=EngineType.COMMUNICATION,
            operation='send_message',
            payload={
                'sender_id': 'routing_tester',
                'recipient_id': 'test_recipient',
                'message_type': 'routing_test',
                'payload': {
                    'test_message': 'Request routing validation',
                    'routing_enabled': True
                },
                'broadcast': False
            },
            priority=6
        )
        
        response = await engine_coordinator.execute_request(communication_request)
        
        # Validate routing and execution
        assert response is not None
        assert response.success
        assert response.request_id == communication_request.request_id
        assert response.engine_type == EngineType.COMMUNICATION
        assert response.execution_time_ms > 0
        assert 'message_id' in response.result
    
    @pytest.mark.asyncio
    async def test_invalid_engine_type_handling(self, engine_coordinator):
        """Test handling of invalid engine type requests."""
        from app.core.engines.consolidated_engine import EngineRequest
        
        # Create request with invalid engine type (using string instead of enum)
        invalid_request = EngineRequest(
            engine_type='invalid_engine_type',  # This will fail validation
            operation='invalid_operation',
            payload={'invalid': 'data'},
            priority=5
        )
        
        # This should raise an error during request creation due to Pydantic validation
        # Test the coordination layer's error handling for unknown engine types
        try:
            response = await engine_coordinator.execute_request(invalid_request)
            # If we get here, the request was processed but should have failed
            assert not response.success
            assert response.error is not None
        except Exception as e:
            # Expected: validation error or similar
            assert 'engine_type' in str(e).lower() or 'invalid' in str(e).lower()


class TestPerformanceBenchmarks:
    """Test comprehensive performance benchmarks."""
    
    @pytest.mark.asyncio
    async def test_mixed_workload_performance(self, engine_coordinator):
        """Test performance under mixed workload conditions."""
        from app.core.engines.consolidated_engine import (
            WorkflowRequest, TaskExecutionRequest, CommunicationRequest
        )
        
        # Create mixed workload
        workflows = []
        tasks = []
        messages = []
        
        # Create 10 workflows
        for i in range(10):
            workflow = WorkflowRequest(
                workflow_definition={
                    'name': f'mixed_workflow_{i}',
                    'steps': [{'name': f'step_{j}', 'type': 'analysis'} for j in range(3)]
                },
                context={'workload_test': True, 'batch': i}
            )
            workflows.append(workflow)
        
        # Create 30 tasks
        for i in range(30):
            task = TaskExecutionRequest(
                task_type='general',
                payload={'workload_test': True, 'task_index': i}
            )
            tasks.append(task)
        
        # Create 20 messages
        for i in range(20):
            message = CommunicationRequest(
                sender_id='workload_tester',
                message_type='workload_test',
                payload={'message_index': i}
            )
            messages.append(message)
        
        # Execute mixed workload concurrently
        start_time = time.time()
        
        workflow_tasks = [engine_coordinator.workflow_engine.execute_workflow(w) for w in workflows]
        task_tasks = [engine_coordinator.task_engine.execute_task(t) for t in tasks]
        message_tasks = [engine_coordinator.communication_engine.send_message(m) for m in messages]
        
        all_tasks = workflow_tasks + task_tasks + message_tasks
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        total_time = (time.time() - start_time) * 1000
        
        # Analyze results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        success_rate = len(successful_results) / len(results)
        
        assert success_rate > 0.9, f"Success rate {success_rate:.2%} should be >90%"
        
        # Performance validation for mixed workload
        avg_time_per_operation = total_time / len(results)
        assert avg_time_per_operation < 500, f"Average time per operation: {avg_time_per_operation}ms"
        
        throughput = len(successful_results) / (total_time / 1000)  # operations per second
        assert throughput > 50, f"Mixed workload throughput: {throughput} ops/sec"
    
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, engine_coordinator):
        """Test performance under sustained load conditions."""
        from app.core.engines.consolidated_engine import TaskExecutionRequest
        
        # Execute tasks in batches to simulate sustained load
        batch_size = 20
        num_batches = 5
        batch_results = []
        
        for batch_num in range(num_batches):
            batch_tasks = []
            for task_num in range(batch_size):
                task = TaskExecutionRequest(
                    task_type='general',
                    payload={
                        'sustained_load_test': True,
                        'batch': batch_num,
                        'task': task_num
                    }
                )
                batch_tasks.append(task)
            
            # Execute batch
            start_time = time.time()
            batch_task_coroutines = [
                engine_coordinator.task_engine.execute_task(task) 
                for task in batch_tasks
            ]
            batch_task_results = await asyncio.gather(*batch_task_coroutines, return_exceptions=True)
            batch_time = (time.time() - start_time) * 1000
            
            successful_batch_results = [
                r for r in batch_task_results 
                if not isinstance(r, Exception) and hasattr(r, 'success') and r.success
            ]
            
            batch_info = {
                'batch_num': batch_num,
                'success_count': len(successful_batch_results),
                'total_count': len(batch_task_results),
                'batch_time_ms': batch_time,
                'throughput': len(successful_batch_results) / (batch_time / 1000)
            }
            batch_results.append(batch_info)
            
            # Small delay between batches to simulate sustained load
            await asyncio.sleep(0.1)
        
        # Validate sustained performance
        for batch_info in batch_results:
            success_rate = batch_info['success_count'] / batch_info['total_count']
            assert success_rate > 0.9, f"Batch {batch_info['batch_num']} success rate: {success_rate:.2%}"
            assert batch_info['throughput'] > 50, f"Batch {batch_info['batch_num']} throughput: {batch_info['throughput']}"
        
        # Verify performance consistency across batches
        throughputs = [b['throughput'] for b in batch_results]
        avg_throughput = sum(throughputs) / len(throughputs)
        throughput_variance = sum((t - avg_throughput) ** 2 for t in throughputs) / len(throughputs)
        throughput_std_dev = throughput_variance ** 0.5
        
        # Standard deviation should be relatively low for consistent performance
        coefficient_of_variation = throughput_std_dev / avg_throughput
        assert coefficient_of_variation < 0.3, f"Performance too variable: CoV = {coefficient_of_variation:.2%}"


class TestErrorHandlingAndResilience:
    """Test error handling and system resilience."""
    
    @pytest.mark.asyncio
    async def test_workflow_engine_error_resilience(self, engine_coordinator):
        """Test workflow engine resilience to errors."""
        from app.core.engines.consolidated_engine import WorkflowRequest
        
        # Test multiple error scenarios
        error_scenarios = [
            {'name': 'empty_workflow', 'definition': {}},
            {'name': 'null_steps', 'definition': {'steps': None}},
            {'name': 'invalid_step_type', 'definition': {'steps': [{'type': 'invalid_type'}]}},
            {'name': 'circular_dependency', 'definition': {
                'steps': [{'name': 'step1'}, {'name': 'step2'}],
                'dependencies': {'step1': ['step2'], 'step2': ['step1']}
            }}
        ]
        
        for scenario in error_scenarios:
            workflow_request = WorkflowRequest(
                workflow_definition=scenario['definition'],
                context={'error_test': scenario['name']}
            )
            
            # Should handle error gracefully without crashing
            result = await engine_coordinator.workflow_engine.execute_workflow(workflow_request)
            assert result is not None
            assert result.workflow_id == workflow_request.workflow_id
    
    @pytest.mark.asyncio
    async def test_task_engine_error_resilience(self, engine_coordinator):
        """Test task execution engine resilience to errors."""
        from app.core.engines.consolidated_engine import TaskExecutionRequest
        
        # Test error scenarios
        error_tasks = [
            TaskExecutionRequest(task_type='nonexistent_type', payload={'error': 'invalid_type'}),
            TaskExecutionRequest(task_type='general', payload=None),  # Null payload
            TaskExecutionRequest(task_type='general', payload={'extremely_large_data': 'x' * 10000})
        ]
        
        for task in error_tasks:
            # Should handle errors gracefully
            result = await engine_coordinator.task_engine.execute_task(task)
            assert result is not None
            assert result.task_id == task.task_id
    
    @pytest.mark.asyncio
    async def test_communication_engine_error_resilience(self, engine_coordinator):
        """Test communication engine resilience to errors."""
        from app.core.engines.consolidated_engine import CommunicationRequest
        
        # Test error scenarios
        error_messages = [
            CommunicationRequest(sender_id='', recipient_id='', message_type=''),  # Empty fields
            CommunicationRequest(
                sender_id='test', 
                message_type='failing_type',
                payload={'extremely_large_payload': 'x' * 10000}
            ),
            CommunicationRequest(sender_id='test', message_type='test', payload=None)  # Null payload
        ]
        
        for message in error_messages:
            # Should handle errors gracefully
            result = await engine_coordinator.communication_engine.send_message(message)
            assert result is not None
            assert result.message_id == message.message_id
    
    @pytest.mark.asyncio
    async def test_coordination_layer_error_recovery(self, engine_coordinator):
        """Test coordination layer error recovery capabilities."""
        from app.core.engines.consolidated_engine import EngineRequest, EngineType
        
        # Test request with invalid payload that should be handled gracefully
        invalid_request = EngineRequest(
            engine_type=EngineType.WORKFLOW,
            operation='execute_workflow',
            payload={'invalid': 'structure', 'missing_required_fields': True},
            priority=5
        )
        
        # Should handle error gracefully at coordination level
        response = await engine_coordinator.execute_request(invalid_request)
        assert response is not None
        assert response.request_id == invalid_request.request_id
        assert response.engine_type == EngineType.WORKFLOW


class TestSystemIntegration:
    """Test system integration and end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_with_tasks_and_communication(self, engine_coordinator):
        """Test complete end-to-end workflow involving all engines."""
        from app.core.engines.consolidated_engine import EngineRequest, EngineType
        
        # Step 1: Execute a workflow that generates tasks
        workflow_request = EngineRequest(
            engine_type=EngineType.WORKFLOW,
            operation='execute_workflow',
            payload={
                'workflow_definition': {
                    'name': 'integration_test_workflow',
                    'steps': [
                        {'name': 'analysis_step', 'type': 'analysis', 'generates_tasks': True},
                        {'name': 'coordination_step', 'type': 'coordination', 'requires_communication': True}
                    ]
                },
                'context': {'integration_test': True}
            }
        )
        
        workflow_response = await engine_coordinator.execute_request(workflow_request)
        assert workflow_response.success
        workflow_id = workflow_response.result['workflow_id']
        
        # Step 2: Execute tasks generated by the workflow
        task_request = EngineRequest(
            engine_type=EngineType.TASK_EXECUTION,
            operation='execute_task',
            payload={
                'task_type': 'analysis',
                'payload': {
                    'workflow_id': workflow_id,
                    'analysis_type': 'integration_test'
                }
            }
        )
        
        task_response = await engine_coordinator.execute_request(task_request)
        assert task_response.success
        task_id = task_response.result['task_id']
        
        # Step 3: Send notifications about workflow and task completion
        notification_request = EngineRequest(
            engine_type=EngineType.COMMUNICATION,
            operation='send_message',
            payload={
                'sender_id': 'integration_tester',
                'message_type': 'workflow_completion',
                'payload': {
                    'workflow_id': workflow_id,
                    'task_id': task_id,
                    'status': 'completed',
                    'timestamp': datetime.utcnow().isoformat()
                },
                'broadcast': True
            }
        )
        
        communication_response = await engine_coordinator.execute_request(notification_request)
        assert communication_response.success
        
        # Validate end-to-end integration
        system_status = await engine_coordinator.get_system_status()
        assert system_status['engines_active'] == 3
        
        # All engines should show some activity from the integration test
        performance_metrics = await engine_coordinator.get_performance_metrics()
        for engine_type in ['workflow_engine', 'task_execution_engine', 'communication_engine']:
            assert engine_type in performance_metrics
    
    @pytest.mark.asyncio
    async def test_system_status_after_operations(self, engine_coordinator):
        """Test system status reporting after various operations."""
        from app.core.engines.consolidated_engine import (
            WorkflowRequest, TaskExecutionRequest, CommunicationRequest
        )
        
        # Execute various operations to generate metrics
        operations = []
        
        # Add workflow execution
        workflow = WorkflowRequest(
            workflow_definition={'name': 'status_test', 'steps': []},
            context={'status_test': True}
        )
        operations.append(engine_coordinator.workflow_engine.execute_workflow(workflow))
        
        # Add task executions
        for i in range(5):
            task = TaskExecutionRequest(task_type='general', payload={'index': i})
            operations.append(engine_coordinator.task_engine.execute_task(task))
        
        # Add communications
        for i in range(3):
            message = CommunicationRequest(
                sender_id='status_tester',
                message_type='status_test',
                payload={'index': i}
            )
            operations.append(engine_coordinator.communication_engine.send_message(message))
        
        # Execute all operations
        await asyncio.gather(*operations, return_exceptions=True)
        
        # Get comprehensive system status
        system_status = await engine_coordinator.get_system_status()
        
        # Validate status completeness
        required_keys = [
            'engines_active', 'orchestrator_integrated', 'managers_integrated',
            'workflow_performance', 'task_performance', 'communication_status'
        ]
        
        for key in required_keys:
            assert key in system_status
        
        assert system_status['engines_active'] == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '--durations=10'])