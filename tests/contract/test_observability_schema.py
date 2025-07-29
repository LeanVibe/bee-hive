"""
Comprehensive Event Schema Validation Tests for LeanVibe Agent Hive 2.0

Tests the observability event schema contract enforcement to ensure all events
conform to the defined schema specification for multi-agent coordination.
"""

import json
import uuid
import pytest
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch

import jsonschema
from pydantic import ValidationError

from app.schemas.observability import (
    BaseObservabilityEvent,
    WorkflowStartedEvent,
    WorkflowEndedEvent,
    NodeExecutingEvent,
    NodeCompletedEvent,
    AgentStateChangedEvent,
    AgentCapabilityUtilizedEvent,
    PreToolUseEvent,
    PostToolUseEvent,
    SemanticQueryEvent,
    SemanticUpdateEvent,
    MessagePublishedEvent,
    MessageReceivedEvent,
    FailureDetectedEvent,
    RecoveryInitiatedEvent,
    SystemHealthCheckEvent,
    EventCategory,
    WorkflowEventType,
    AgentEventType,
    ToolEventType,
    MemoryEventType,
    CommunicationEventType,
    RecoveryEventType,
    SystemEventType,
    PerformanceMetrics,
    EventMetadata,
    ObservabilityEvent,
)
from app.core.event_serialization import (
    EventSerializer,
    SerializationFormat,
    serialize_for_stream,
    serialize_for_storage,
    deserialize_from_stream,
    deserialize_from_storage,
)
from mock_servers.observability_events_mock import (
    MockEventGenerator,
    WorkflowScenario,
    generate_sample_events,
)


class TestEventSchemaContract:
    """Test suite for event schema contract validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.schema_path = "/Users/bogdan/work/leanvibe-dev/bee-hive/schemas/observability_events.json"
        
        # Load JSON schema
        with open(self.schema_path, 'r') as f:
            self.json_schema = json.load(f)
        
        self.mock_generator = MockEventGenerator()
        
        # Common test data
        self.test_workflow_id = uuid.uuid4()
        self.test_agent_id = uuid.uuid4()
        self.test_session_id = uuid.uuid4()
        self.test_context_id = uuid.uuid4()
        self.test_message_id = uuid.uuid4()
    
    def test_json_schema_loading(self):
        """Test that JSON schema loads correctly."""
        assert self.json_schema is not None
        assert self.json_schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert self.json_schema["title"] == "LeanVibe Agent Hive 2.0 Observability Event Schema"
        assert "definitions" in self.json_schema
        assert "anyOf" in self.json_schema
    
    def test_base_event_structure(self):
        """Test base event structure validation."""
        # Valid base event
        base_event = BaseObservabilityEvent(
            event_type="TestEvent",
            event_category=EventCategory.SYSTEM,
            performance_metrics=PerformanceMetrics(
                execution_time_ms=1.5,
                memory_usage_mb=10.0,
                cpu_usage_percent=5.0
            )
        )
        
        assert base_event.event_id is not None
        assert base_event.timestamp is not None
        assert base_event.event_type == "TestEvent"
        assert base_event.event_category == EventCategory.SYSTEM
        assert base_event.metadata.schema_version == "1.0.0"
    
    def test_semantic_embedding_validation(self):
        """Test semantic embedding dimension validation."""
        # Valid embedding (1536 dimensions)
        valid_embedding = [0.1] * 1536
        base_event = BaseObservabilityEvent(
            event_type="TestEvent",
            event_category=EventCategory.MEMORY,
            semantic_embedding=valid_embedding
        )
        assert len(base_event.semantic_embedding) == 1536
        
        # Invalid embedding (wrong dimensions)
        with pytest.raises(ValidationError) as exc_info:
            BaseObservabilityEvent(
                event_type="TestEvent",
                event_category=EventCategory.MEMORY,
                semantic_embedding=[0.1] * 512  # Wrong dimension count
            )
        assert "exactly 1536 dimensions" in str(exc_info.value)
    
    def test_performance_metrics_validation(self):
        """Test performance metrics validation."""
        # Valid metrics
        valid_metrics = PerformanceMetrics(
            execution_time_ms=1.5,
            memory_usage_mb=10.0,
            cpu_usage_percent=25.0
        )
        assert valid_metrics.execution_time_ms == 1.5
        assert valid_metrics.memory_usage_mb == 10.0
        assert valid_metrics.cpu_usage_percent == 25.0
        
        # Invalid metrics (negative values)
        with pytest.raises(ValidationError):
            PerformanceMetrics(execution_time_ms=-1.0)
        
        with pytest.raises(ValidationError):
            PerformanceMetrics(memory_usage_mb=-5.0)
        
        with pytest.raises(ValidationError):
            PerformanceMetrics(cpu_usage_percent=150.0)  # > 100%


class TestWorkflowEventValidation:
    """Test workflow event schema validation."""
    
    def setup_method(self):
        self.test_workflow_id = uuid.uuid4()
        self.test_session_id = uuid.uuid4()
    
    def test_workflow_started_event(self):
        """Test workflow started event validation."""
        event = WorkflowStartedEvent(
            workflow_id=self.test_workflow_id,
            session_id=self.test_session_id,
            workflow_name="Test Workflow",
            workflow_definition={"tasks": [], "dependencies": {}},
            initial_context={"scenario": "test"},
            estimated_duration_ms=60000.0,
            priority="high"
        )
        
        assert event.event_type == WorkflowEventType.WORKFLOW_STARTED
        assert event.event_category == EventCategory.WORKFLOW
        assert event.workflow_id == self.test_workflow_id
        assert event.workflow_name == "Test Workflow"
        
        # Validate against JSON schema
        event_dict = event.dict()
        jsonschema.validate(event_dict, self._get_workflow_started_schema())
    
    def test_workflow_ended_event(self):
        """Test workflow ended event validation."""
        event = WorkflowEndedEvent(
            workflow_id=self.test_workflow_id,
            session_id=self.test_session_id,
            status="completed",
            completion_reason="All tasks completed successfully",
            final_result={"success": True},
            total_tasks_executed=5,
            failed_tasks=0,
            actual_duration_ms=45000.0
        )
        
        assert event.event_type == WorkflowEventType.WORKFLOW_ENDED
        assert event.event_category == EventCategory.WORKFLOW
        assert event.status == "completed"
        assert event.total_tasks_executed == 5
        
        # Test invalid status
        with pytest.raises(ValidationError):
            WorkflowEndedEvent(
                workflow_id=self.test_workflow_id,
                session_id=self.test_session_id,
                status="invalid_status",  # Not in allowed values
                completion_reason="Test"
            )
    
    def test_node_executing_event(self):
        """Test node executing event validation."""
        event = NodeExecutingEvent(
            workflow_id=self.test_workflow_id,
            session_id=self.test_session_id,
            node_id="task_1",
            node_type="analysis",
            node_name="Analyze Requirements",
            input_data={"requirement": "Feature X"},
            assigned_agent=uuid.uuid4(),
            estimated_execution_time_ms=30000.0
        )
        
        assert event.event_type == WorkflowEventType.NODE_EXECUTING
        assert event.node_id == "task_1"
        assert event.node_type == "analysis"
    
    def test_node_completed_event(self):
        """Test node completed event validation."""
        event = NodeCompletedEvent(
            workflow_id=self.test_workflow_id,
            session_id=self.test_session_id,
            node_id="task_1",
            success=True,
            output_data={"result": "Analysis complete"},
            retry_count=0,
            downstream_nodes=["task_2", "task_3"]
        )
        
        assert event.event_type == WorkflowEventType.NODE_COMPLETED
        assert event.success is True
        assert event.retry_count == 0
    
    def _get_workflow_started_schema(self) -> Dict[str, Any]:
        """Get workflow started event schema subset."""
        return {
            "type": "object",
            "required": ["event_type", "event_category", "workflow_id"],
            "properties": {
                "event_type": {"const": "WorkflowStarted"},
                "event_category": {"const": "workflow"},
                "workflow_id": {"type": "string", "format": "uuid"}
            }
        }


class TestAgentEventValidation:
    """Test agent event schema validation."""
    
    def setup_method(self):
        self.test_agent_id = uuid.uuid4()
        self.test_session_id = uuid.uuid4()
    
    def test_agent_state_changed_event(self):
        """Test agent state changed event validation."""
        event = AgentStateChangedEvent(
            agent_id=self.test_agent_id,
            session_id=self.test_session_id,
            previous_state="idle",
            new_state="active",
            state_transition_reason="Assigned to workflow",
            capabilities=["coding", "analysis"],
            resource_allocation={"cpu_percent": 50.0, "memory_mb": 256.0},
            persona_data={"persona": "senior_developer"}
        )
        
        assert event.event_type == AgentEventType.AGENT_STATE_CHANGED
        assert event.event_category == EventCategory.AGENT
        assert event.previous_state == "idle"
        assert event.new_state == "active"
        assert len(event.capabilities) == 2
    
    def test_agent_capability_utilized_event(self):
        """Test agent capability utilized event validation."""
        event = AgentCapabilityUtilizedEvent(
            agent_id=self.test_agent_id,
            session_id=self.test_session_id,
            capability_name="code_analysis",
            utilization_context="Analyzing Python code for performance",
            input_parameters={"file_path": "/src/main.py"},
            capability_result={"issues_found": 3},
            efficiency_score=0.85
        )
        
        assert event.event_type == AgentEventType.AGENT_CAPABILITY_UTILIZED
        assert event.capability_name == "code_analysis"
        assert event.efficiency_score == 0.85
        
        # Test invalid efficiency score
        with pytest.raises(ValidationError):
            AgentCapabilityUtilizedEvent(
                agent_id=self.test_agent_id,
                session_id=self.test_session_id,
                capability_name="test",
                utilization_context="test",
                efficiency_score=1.5  # > 1.0
            )


class TestToolEventValidation:
    """Test tool event schema validation."""
    
    def setup_method(self):
        self.test_agent_id = uuid.uuid4()
        self.test_session_id = uuid.uuid4()
    
    def test_pre_tool_use_event(self):
        """Test pre-tool use event validation."""
        event = PreToolUseEvent(
            agent_id=self.test_agent_id,
            session_id=self.test_session_id,
            tool_name="Read",
            parameters={"file_path": "/src/main.py"},
            tool_version="1.0.0",
            expected_output_type="text",
            timeout_ms=30000
        )
        
        assert event.event_type == ToolEventType.PRE_TOOL_USE
        assert event.event_category == EventCategory.TOOL
        assert event.tool_name == "Read"
        assert event.parameters["file_path"] == "/src/main.py"
    
    def test_post_tool_use_event(self):
        """Test post-tool use event validation."""
        event = PostToolUseEvent(
            agent_id=self.test_agent_id,
            session_id=self.test_session_id,
            tool_name="Read",
            success=True,
            result="File content here...",
            retry_count=0,
            result_truncated=False
        )
        
        assert event.event_type == ToolEventType.POST_TOOL_USE
        assert event.success is True
        assert event.retry_count == 0
        
        # Test failure case
        error_event = PostToolUseEvent(
            agent_id=self.test_agent_id,
            session_id=self.test_session_id,
            tool_name="Read",
            success=False,
            error="File not found",
            error_type="file_not_found",
            retry_count=2
        )
        
        assert error_event.success is False
        assert error_event.error == "File not found"
        assert error_event.retry_count == 2


class TestMemoryEventValidation:
    """Test memory event schema validation."""
    
    def setup_method(self):
        self.test_agent_id = uuid.uuid4()
        self.test_session_id = uuid.uuid4()
        self.test_context_id = uuid.uuid4()
    
    def test_semantic_query_event(self):
        """Test semantic query event validation."""
        query_embedding = [0.1] * 1536  # Valid 1536-dimensional vector
        
        event = SemanticQueryEvent(
            agent_id=self.test_agent_id,
            session_id=self.test_session_id,
            context_id=self.test_context_id,
            query_text="How to implement authentication?",
            query_embedding=query_embedding,
            similarity_threshold=0.7,
            max_results=10,
            results_count=5,
            search_strategy="semantic_similarity"
        )
        
        assert event.event_type == MemoryEventType.SEMANTIC_QUERY
        assert event.event_category == EventCategory.MEMORY
        assert event.query_text == "How to implement authentication?"
        assert len(event.query_embedding) == 1536
        assert event.similarity_threshold == 0.7
    
    def test_semantic_update_event(self):
        """Test semantic update event validation."""
        content_embedding = [0.2] * 1536
        
        event = SemanticUpdateEvent(
            agent_id=self.test_agent_id,
            session_id=self.test_session_id,
            context_id=self.test_context_id,
            operation_type="insert",
            content={"type": "knowledge", "data": "Authentication best practices"},
            content_embedding=content_embedding,
            content_id="knowledge_123",
            content_type="documentation",
            affected_records=1
        )
        
        assert event.event_type == MemoryEventType.SEMANTIC_UPDATE
        assert event.operation_type == "insert"
        assert event.content["type"] == "knowledge"
        assert len(event.content_embedding) == 1536
        
        # Test invalid operation type
        with pytest.raises(ValidationError):
            SemanticUpdateEvent(
                agent_id=self.test_agent_id,
                session_id=self.test_session_id,
                context_id=self.test_context_id,
                operation_type="invalid_op",  # Not in allowed values
                content={}
            )


class TestCommunicationEventValidation:
    """Test communication event schema validation."""
    
    def setup_method(self):
        self.test_session_id = uuid.uuid4()
        self.test_message_id = uuid.uuid4()
    
    def test_message_published_event(self):
        """Test message published event validation."""
        event = MessagePublishedEvent(
            session_id=self.test_session_id,
            message_id=self.test_message_id,
            from_agent="agent_1",
            to_agent="agent_2",
            message_type="coordination",
            message_content={"action": "status_update", "data": "Task complete"},
            priority="high",
            delivery_method="redis_stream",
            expected_response=True
        )
        
        assert event.event_type == CommunicationEventType.MESSAGE_PUBLISHED
        assert event.event_category == EventCategory.COMMUNICATION
        assert event.from_agent == "agent_1"
        assert event.to_agent == "agent_2"
        assert event.expected_response is True
    
    def test_message_received_event(self):
        """Test message received event validation."""
        event = MessageReceivedEvent(
            session_id=self.test_session_id,
            message_id=self.test_message_id,
            from_agent="agent_1",
            processing_status="accepted",
            processing_reason="Message processed successfully",
            response_generated=True,
            delivery_latency_ms=25.5
        )
        
        assert event.event_type == CommunicationEventType.MESSAGE_RECEIVED
        assert event.processing_status == "accepted"
        assert event.delivery_latency_ms == 25.5
        
        # Test invalid processing status
        with pytest.raises(ValidationError):
            MessageReceivedEvent(
                session_id=self.test_session_id,
                message_id=self.test_message_id,
                from_agent="agent_1",
                processing_status="invalid_status"  # Not in allowed values
            )


class TestRecoveryEventValidation:
    """Test recovery event schema validation."""
    
    def setup_method(self):
        self.test_agent_id = uuid.uuid4()
        self.test_session_id = uuid.uuid4()
    
    def test_failure_detected_event(self):
        """Test failure detected event validation."""
        event = FailureDetectedEvent(
            agent_id=self.test_agent_id,
            session_id=self.test_session_id,
            failure_type="timeout",
            failure_description="Tool execution timeout",
            affected_component="Read tool",
            severity="high",
            error_details={"timeout_ms": 30000, "tool": "Read"},
            detection_method="automated_monitoring",
            impact_assessment={"affected_workflows": 1}
        )
        
        assert event.event_type == RecoveryEventType.FAILURE_DETECTED
        assert event.event_category == EventCategory.RECOVERY
        assert event.failure_type == "timeout"
        assert event.severity == "high"
        
        # Test invalid severity
        with pytest.raises(ValidationError):
            FailureDetectedEvent(
                agent_id=self.test_agent_id,
                session_id=self.test_session_id,
                failure_type="test",
                failure_description="test",
                affected_component="test",
                severity="invalid",  # Not in allowed values
                error_details={}
            )
    
    def test_recovery_initiated_event(self):
        """Test recovery initiated event validation."""
        event = RecoveryInitiatedEvent(
            agent_id=self.test_agent_id,
            session_id=self.test_session_id,
            recovery_strategy="retry",
            trigger_failure="Tool execution timeout",
            recovery_steps=["retry_tool", "fallback_strategy", "escalate"],
            estimated_recovery_time_ms=15000.0,
            backup_systems_activated=["backup_processor"],
            rollback_checkpoint="checkpoint_1"
        )
        
        assert event.event_type == RecoveryEventType.RECOVERY_INITIATED
        assert event.recovery_strategy == "retry"
        assert len(event.recovery_steps) == 3
        assert event.estimated_recovery_time_ms == 15000.0


class TestSystemEventValidation:
    """Test system event schema validation."""
    
    def setup_method(self):
        self.test_session_id = uuid.uuid4()
    
    def test_system_health_check_event(self):
        """Test system health check event validation."""
        event = SystemHealthCheckEvent(
            session_id=self.test_session_id,
            health_status="healthy",
            check_type="periodic_health_check",
            component_statuses={
                "redis": "healthy",
                "postgres": "healthy",
                "agents": "degraded"
            },
            performance_indicators={
                "memory_usage_percent": 65.0,
                "cpu_usage_percent": 45.0,
                "disk_usage_percent": 30.0
            },
            alerts_triggered=["high_memory_usage"],
            recommended_actions=["monitor_memory", "scale_up"]
        )
        
        assert event.event_type == SystemEventType.SYSTEM_HEALTH_CHECK
        assert event.event_category == EventCategory.SYSTEM
        assert event.health_status == "healthy"
        assert len(event.component_statuses) == 3
        assert len(event.alerts_triggered) == 1
        
        # Test invalid health status
        with pytest.raises(ValidationError):
            SystemHealthCheckEvent(
                session_id=self.test_session_id,
                health_status="invalid_status",  # Not in allowed values
                check_type="test",
                component_statuses={},
                performance_indicators={}
            )


class TestEventSerialization:
    """Test event serialization contract compliance."""
    
    def setup_method(self):
        self.serializer = EventSerializer(format=SerializationFormat.MSGPACK)
        self.json_serializer = EventSerializer(format=SerializationFormat.JSON)
    
    def test_event_serialization_roundtrip(self):
        """Test event serialization and deserialization roundtrip."""
        # Create test event
        event = PreToolUseEvent(
            agent_id=uuid.uuid4(),
            session_id=uuid.uuid4(),
            tool_name="Read",
            parameters={"file_path": "/test/file.py"},
            performance_metrics=PerformanceMetrics(execution_time_ms=2.5)
        )
        
        # Serialize
        serialized_data, metadata = self.serializer.serialize_event(event)
        
        # Deserialize
        deserialized_event = self.serializer.deserialize_event(serialized_data, metadata)
        
        # Validate roundtrip
        assert deserialized_event['event_type'] == event.event_type
        assert deserialized_event['tool_name'] == event.tool_name
        assert deserialized_event['parameters'] == event.parameters
    
    def test_batch_serialization(self):
        """Test batch event serialization."""
        events = [
            PreToolUseEvent(
                agent_id=uuid.uuid4(),
                session_id=uuid.uuid4(),
                tool_name="Read",
                parameters={"file_path": f"/test/file_{i}.py"}
            ) for i in range(10)
        ]
        
        # Serialize batch
        serialized_data, metadata = self.serializer.serialize_batch(events)
        
        assert metadata['batch_serialization'] is True
        assert metadata['event_count'] == 10
        assert metadata['batch_size_bytes'] > 0
        assert metadata['avg_time_per_event_ms'] > 0
    
    def test_serialization_performance(self):
        """Test serialization performance requirements."""
        # Generate sample events
        sample_events = []
        for i in range(100):
            event = PostToolUseEvent(
                agent_id=uuid.uuid4(),
                session_id=uuid.uuid4(),
                tool_name="Bash",
                success=True,
                result=f"Command output {i}",
                performance_metrics=PerformanceMetrics(execution_time_ms=random.uniform(1.0, 5.0))
            )
            sample_events.append(event)
        
        # Benchmark performance
        performance_results = self.serializer.benchmark_performance(sample_events, iterations=100)
        
        # Validate performance requirements
        assert performance_results['avg_serialize_ms'] < 5.0, "Serialization must be < 5ms"
        assert performance_results['avg_deserialize_ms'] < 5.0, "Deserialization must be < 5ms"
        assert performance_results['p95_serialize_ms'] < 10.0, "95th percentile serialization must be < 10ms"
        assert performance_results['events_per_second'] > 100, "Must process > 100 events per second"
    
    def test_compression_effectiveness(self):
        """Test compression effectiveness for storage."""
        compressed_serializer = EventSerializer(
            format=SerializationFormat.MSGPACK_COMPRESSED,
            enable_compression=True
        )
        
        # Create event with large payload
        large_event = SemanticQueryEvent(
            session_id=uuid.uuid4(),
            query_text="Large query text " * 100,  # Repeat to create size
            query_embedding=[0.1] * 1536,
            filter_criteria={"large_filter": "x" * 1000},
            performance_metrics=PerformanceMetrics(execution_time_ms=100.0)
        )
        
        # Compare compressed vs uncompressed
        uncompressed_data, _ = self.serializer.serialize_event(large_event)
        compressed_data, _ = compressed_serializer.serialize_event(large_event)
        
        compression_ratio = len(compressed_data) / len(uncompressed_data)
        
        # Should achieve reasonable compression
        assert compression_ratio < 0.8, f"Compression ratio {compression_ratio} should be < 0.8"


class TestMockEventGeneration:
    """Test mock event generation contract compliance."""
    
    def setup_method(self):
        self.generator = MockEventGenerator()
    
    def test_workflow_scenario_generation(self):
        """Test complete workflow scenario generation."""
        events = list(self.generator.generate_workflow_scenario(WorkflowScenario.CODE_REVIEW))
        
        # Should generate multiple events
        assert len(events) > 10, "Should generate substantial number of events"
        
        # Should have workflow start and end
        event_types = [event['event_type'] for event in events]
        assert 'WorkflowStarted' in event_types
        assert 'WorkflowEnded' in event_types
        
        # Should have tool usage events
        assert 'PreToolUse' in event_types
        assert 'PostToolUse' in event_types
        
        # Should have agent events
        assert 'AgentStateChanged' in event_types
    
    def test_event_schema_compliance(self):
        """Test that generated events comply with schema."""
        events = generate_sample_events(count=50)
        
        for event in events:
            # Basic structure validation
            assert 'event_id' in event
            assert 'timestamp' in event
            assert 'event_type' in event
            assert 'event_category' in event
            
            # Category validation
            assert event['event_category'] in [cat.value for cat in EventCategory]
            
            # Performance metrics validation
            if 'performance_metrics' in event and event['performance_metrics']:
                metrics = event['performance_metrics']
                if 'execution_time_ms' in metrics and metrics['execution_time_ms'] is not None:
                    assert metrics['execution_time_ms'] >= 0
                if 'memory_usage_mb' in metrics and metrics['memory_usage_mb'] is not None:
                    assert metrics['memory_usage_mb'] >= 0
                if 'cpu_usage_percent' in metrics and metrics['cpu_usage_percent'] is not None:
                    assert 0 <= metrics['cpu_usage_percent'] <= 100
    
    def test_realistic_event_patterns(self):
        """Test that generated events follow realistic patterns."""
        events = list(self.generator.generate_workflow_scenario(WorkflowScenario.FEATURE_DEVELOPMENT))
        
        # Extract workflow events
        workflow_events = [e for e in events if e['event_category'] == 'workflow']
        
        # Should have workflow start before workflow end
        start_found = False
        for event in workflow_events:
            if event['event_type'] == 'WorkflowStarted':
                start_found = True
            elif event['event_type'] == 'WorkflowEnded':
                assert start_found, "Workflow should start before ending"
        
        # Tool events should have matching pre/post pairs
        tool_events = [e for e in events if e['event_category'] == 'tool']
        pre_tool_events = [e for e in tool_events if e['event_type'] == 'PreToolUse']
        post_tool_events = [e for e in tool_events if e['event_type'] == 'PostToolUse']
        
        # Should have reasonable ratio of pre/post events
        ratio = len(post_tool_events) / len(pre_tool_events) if pre_tool_events else 0
        assert 0.8 <= ratio <= 1.2, f"Pre/Post tool event ratio {ratio} should be close to 1"


class TestIntegrationValidation:
    """Test integration with existing infrastructure."""
    
    @pytest.mark.asyncio
    async def test_redis_stream_integration(self):
        """Test integration with Redis streams."""
        # This would require Redis running - mock for now
        with patch('app.core.redis.get_redis') as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client
            
            # Test serialization for stream
            event = PreToolUseEvent(
                agent_id=uuid.uuid4(),
                session_id=uuid.uuid4(),
                tool_name="Read",
                parameters={"file_path": "/test.py"}
            )
            
            serialized_data, metadata = serialize_for_stream(event)
            
            # Should be MessagePack format
            assert metadata['serialization_format'] == 'msgpack'
            assert not metadata['compressed']
            assert metadata['serialized_size_bytes'] > 0
            assert metadata['serialization_time_ms'] < 5.0
    
    def test_semantic_memory_integration(self):
        """Test semantic memory event integration."""
        # Test semantic events have proper embedding structure
        event = SemanticQueryEvent(
            session_id=uuid.uuid4(),
            query_text="Test query",
            query_embedding=[0.1] * 1536,
            similarity_threshold=0.7
        )
        
        # Should be compatible with existing embedding service
        assert len(event.query_embedding) == 1536
        assert all(isinstance(x, (int, float)) for x in event.query_embedding)
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration."""
        # Test that all events include performance metrics
        events = generate_sample_events(count=20)
        
        for event in events:
            if 'performance_metrics' in event and event['performance_metrics']:
                metrics = event['performance_metrics']
                
                # Should have at least execution time
                assert 'execution_time_ms' in metrics
                
                # Values should be realistic for monitoring
                if metrics.get('execution_time_ms'):
                    assert metrics['execution_time_ms'] < 10000  # < 10 seconds


# Performance benchmark test
def test_performance_benchmarks():
    """Test that serialization meets performance requirements."""
    import time
    
    # Generate test events
    events = generate_sample_events(count=1000)
    
    # Test high-performance serialization
    serializer = EventSerializer(format=SerializationFormat.MSGPACK)
    
    start_time = time.time()
    
    for event in events:
        serialized_data, metadata = serializer.serialize_event(event)
        # Validate serialization time
        assert metadata['serialization_time_ms'] < 5.0, "Individual serialization must be < 5ms"
    
    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / len(events)
    
    # Overall performance validation
    assert avg_time_ms < 2.0, f"Average serialization time {avg_time_ms}ms must be < 2ms"
    
    events_per_second = len(events) / (total_time_ms / 1000)
    assert events_per_second > 500, f"Must process > 500 events/sec, got {events_per_second}"


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "test_performance_benchmarks or test_event_schema_compliance"
    ])