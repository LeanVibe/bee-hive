# LeanVibe Agent Hive 2.0 - Observability Event Schema Contract

## Overview

This document defines the comprehensive observability event schema contract for LeanVibe Agent Hive 2.0, enabling coordinated development of four parallel subagents: Observability-Agent, Dashboard-Agent, Analytics-Agent, and Performance-Agent.

The event schema provides a standardized contract for capturing all critical multi-agent system events with semantic intelligence, performance metrics, and high-performance serialization support.

## Schema Architecture

### Base Event Structure

All observability events inherit from the `BaseObservabilityEvent` schema:

```json
{
  "event_id": "uuid",
  "timestamp": "ISO8601_datetime",
  "event_type": "enum_from_specific_categories",
  "event_category": "workflow|agent|tool|memory|communication|recovery|system",
  "workflow_id": "uuid_optional",
  "agent_id": "uuid_optional",
  "session_id": "uuid_optional",
  "context_id": "uuid_optional",
  "semantic_embedding": "1536d_vector_optional",
  "payload": "event_specific_data",
  "performance_metrics": {
    "execution_time_ms": "float",
    "memory_usage_mb": "float",
    "cpu_usage_percent": "float"
  },
  "metadata": {
    "schema_version": "string",
    "correlation_id": "uuid",
    "source_service": "string",
    "trace_id": "string",
    "span_id": "string",
    "sampling_probability": "float"
  }
}
```

### Event Categories

The schema defines seven primary event categories, each containing specific event types:

#### 1. Workflow Events (`event_category: "workflow"`)

Critical for tracking multi-agent workflow execution and DAG progression:

- **WorkflowStarted**: Workflow initialization with definition and context
- **WorkflowEnded**: Workflow completion with results and metrics
- **WorkflowPaused**: Workflow suspension with checkpoint data
- **WorkflowResumed**: Workflow continuation from checkpoint
- **NodeExecuting**: Task node execution start with dependencies
- **NodeCompleted**: Task node completion with output data
- **NodeFailed**: Task node failure with error details and recovery info

#### 2. Agent Events (`event_category: "agent"`)

Essential for agent lifecycle and capability tracking:

- **AgentStateChanged**: Agent state transitions with resource allocation
- **AgentCapabilityUtilized**: Capability usage with efficiency metrics
- **AgentStarted**: Agent initialization with persona data
- **AgentStopped**: Agent termination with cleanup info
- **AgentPaused**: Agent suspension
- **AgentResumed**: Agent continuation

#### 3. Tool Events (`event_category: "tool"`)

Most critical for debugging tool execution (PreToolUse/PostToolUse):

- **PreToolUse**: Tool execution start with parameters and timeout
- **PostToolUse**: Tool execution completion with results and errors
- **ToolRegistered**: Tool registration in the system
- **ToolUnregistered**: Tool removal from the system
- **ToolUpdated**: Tool metadata or capability updates

#### 4. Memory Events (`event_category: "memory"`)

Semantic intelligence operations for context-aware analysis:

- **SemanticQuery**: Vector similarity search operations
- **SemanticUpdate**: Memory content insertion/update/deletion
- **MemoryConsolidation**: Knowledge consolidation and optimization

#### 5. Communication Events (`event_category: "communication"`)

Inter-agent messaging and coordination:

- **MessagePublished**: Message broadcast to agents
- **MessageReceived**: Message processing and acknowledgment
- **BroadcastEvent**: System-wide announcements

#### 6. Recovery Events (`event_category: "recovery"`)

System resilience and failure handling:

- **FailureDetected**: Error detection with impact assessment
- **RecoveryInitiated**: Recovery strategy activation
- **RecoveryCompleted**: Recovery completion with lessons learned

#### 7. System Events (`event_category: "system"`)

System health and configuration monitoring:

- **SystemHealthCheck**: Periodic health validation
- **ConfigurationChange**: System configuration updates

## Event Schema Implementation

### Pydantic Models

The schema is implemented using Pydantic v2 models with strict validation:

```python
from app.schemas.observability import (
    BaseObservabilityEvent,
    WorkflowStartedEvent,
    PreToolUseEvent,
    PostToolUseEvent,
    SemanticQueryEvent,
    # ... other event types
)

# Example: Creating a PreToolUse event
event = PreToolUseEvent(
    agent_id=uuid.uuid4(),
    session_id=uuid.uuid4(),
    tool_name="Read",
    parameters={"file_path": "/src/main.py"},
    performance_metrics=PerformanceMetrics(
        execution_time_ms=2.5,
        memory_usage_mb=15.0,
        cpu_usage_percent=12.0
    )
)
```

### JSON Schema Validation

The complete JSON schema is available at `/schemas/observability_events.json` for contract validation:

```python
import jsonschema
import json

# Load schema
with open('schemas/observability_events.json', 'r') as f:
    schema = json.load(f)

# Validate event
jsonschema.validate(event.dict(), schema)
```

## High-Performance Serialization

### MessagePack Support

Binary serialization with <5ms overhead requirement:

```python
from app.core.event_serialization import (
    serialize_for_stream,
    serialize_for_storage,
    deserialize_from_stream,
    EventSerializer,
    SerializationFormat
)

# High-performance streaming
serialized_data, metadata = serialize_for_stream(event)

# Compressed storage
serialized_data, metadata = serialize_for_storage(event)

# Custom serializer
serializer = EventSerializer(
    format=SerializationFormat.MSGPACK,
    enable_compression=False,
    max_payload_size=50000
)
```

### Performance Benchmarks

Serialization performance requirements:

- **Individual Event**: < 5ms serialization + deserialization
- **Batch Processing**: > 200 events/second throughput
- **Memory Usage**: < 100MB peak during processing
- **P95 Latency**: < 10ms for 95th percentile operations

## Mock Event Generation

### Realistic Workflow Simulation

The mock event generator provides realistic multi-agent scenarios:

```python
from mock_servers.observability_events_mock import (
    MockEventGenerator,
    WorkflowScenario,
    generate_sample_events
)

# Generate complete workflow
generator = MockEventGenerator()
events = list(generator.generate_workflow_scenario(
    WorkflowScenario.CODE_REVIEW
))

# Generate sample events for testing
sample_events = generate_sample_events(count=100)
```

### Supported Scenarios

- **CODE_REVIEW**: Pull request analysis with security scanning
- **FEATURE_DEVELOPMENT**: Requirements to deployment pipeline
- **BUG_INVESTIGATION**: Debugging with root cause analysis
- **PERFORMANCE_OPTIMIZATION**: Profiling and optimization workflow
- **SECURITY_AUDIT**: Comprehensive security assessment
- **DOCUMENTATION_UPDATE**: Knowledge management workflow

## Integration Patterns for Subagents

### 1. Observability-Agent Integration

**Hook Implementation Pattern:**

```python
from app.observability.hooks import HookInterceptor, EventProcessor

class ObservabilityEventProcessor(EventProcessor):
    async def process_event(self, session_id, agent_id, event_type, payload, latency_ms=None):
        # Create typed event from payload
        event = self._create_typed_event(event_type, payload)
        
        # Serialize for stream
        serialized_data, metadata = serialize_for_stream(event)
        
        # Send to Redis stream
        stream_id = await self.redis.xadd(
            "observability_events",
            {"event_data": serialized_data, "metadata": str(metadata)}
        )
        
        return stream_id

# Initialize hook interceptor
processor = ObservabilityEventProcessor()
interceptor = HookInterceptor(processor)
```

**Event Collection Pattern:**

```python
# Capture PreToolUse
await interceptor.capture_pre_tool_use(
    session_id=session_id,
    agent_id=agent_id,
    tool_data={
        "tool_name": tool_name,
        "parameters": parameters,
        "correlation_id": correlation_id
    }
)

# Capture PostToolUse
await interceptor.capture_post_tool_use(
    session_id=session_id,
    agent_id=agent_id,
    tool_result={
        "tool_name": tool_name,
        "success": success,
        "result": result,
        "execution_time_ms": execution_time
    },
    latency_ms=latency_ms
)
```

### 2. Dashboard-Agent Integration

**Real-time Event Consumption:**

```python
import asyncio
from app.core.redis import get_redis

class DashboardEventConsumer:
    async def consume_events(self):
        redis_client = get_redis()
        
        # Consumer group for dashboard
        await redis_client.xgroup_create(
            "observability_events",
            "dashboard_group",
            id='0',
            mkstream=True
        )
        
        while True:
            # Read events
            messages = await redis_client.xreadgroup(
                "dashboard_group",
                "dashboard_consumer",
                {"observability_events": ">"},
                count=10,
                block=1000
            )
            
            for stream, msgs in messages:
                for msg_id, fields in msgs:
                    await self._process_dashboard_event(msg_id, fields)
                    
                    # Acknowledge
                    await redis_client.xack(
                        "observability_events",
                        "dashboard_group",
                        msg_id
                    )
```

**Event Type Filtering:**

```python
class DashboardEventFilter:
    def __init__(self):
        self.critical_events = {
            "WorkflowStarted", "WorkflowEnded",
            "PreToolUse", "PostToolUse",
            "FailureDetected", "RecoveryInitiated"
        }
    
    def should_display(self, event: dict) -> bool:
        return (
            event["event_type"] in self.critical_events or
            event.get("performance_metrics", {}).get("execution_time_ms", 0) > 1000 or
            not event.get("success", True)  # Failed operations
        )
```

### 3. Analytics-Agent Integration

**Historical Data Analysis:**

```python
from app.core.database import get_db_session
from app.models.observability import AgentEvent

class AnalyticsEventProcessor:
    async def analyze_performance_trends(self, hours: int = 24):
        async with get_db_session() as session:
            # Query tool performance trends
            tool_events = await session.execute(
                select(AgentEvent)
                .where(
                    AgentEvent.event_type.in_(["PreToolUse", "PostToolUse"]),
                    AgentEvent.created_at > datetime.utcnow() - timedelta(hours=hours)
                )
                .order_by(AgentEvent.created_at)
            )
            
            # Analyze patterns
            return self._analyze_tool_patterns(tool_events.scalars().all())
    
    def _analyze_tool_patterns(self, events):
        # Group by tool name
        tool_metrics = defaultdict(list)
        
        for event in events:
            if event.event_type == "PostToolUse":
                tool_name = event.payload.get("tool_name")
                execution_time = event.payload.get("execution_time_ms")
                
                if tool_name and execution_time:
                    tool_metrics[tool_name].append(execution_time)
        
        # Calculate statistics
        return {
            tool: {
                "avg_time_ms": statistics.mean(times),
                "p95_time_ms": sorted(times)[int(0.95 * len(times))],
                "total_executions": len(times)
            }
            for tool, times in tool_metrics.items()
        }
```

**Semantic Analysis Integration:**

```python
class SemanticAnalytics:
    async def analyze_query_patterns(self):
        # Find semantic queries
        semantic_events = await self._get_semantic_events()
        
        # Cluster similar queries
        embeddings = [event.semantic_embedding for event in semantic_events 
                     if event.semantic_embedding]
        
        # Use existing semantic memory service
        clusters = await self.semantic_service.cluster_embeddings(embeddings)
        
        return self._generate_insights(clusters)
```

### 4. Performance-Agent Integration

**Performance Monitoring:**

```python
class PerformanceMonitor:
    def __init__(self):
        self.thresholds = {
            "tool_execution_ms": 5000,
            "workflow_duration_ms": 300000,
            "memory_usage_mb": 1000,
            "error_rate_percent": 5.0
        }
    
    async def monitor_performance_violations(self):
        events = await self._get_recent_events(minutes=5)
        
        violations = []
        for event in events:
            metrics = event.get("performance_metrics", {})
            
            # Check thresholds
            if metrics.get("execution_time_ms", 0) > self.thresholds["tool_execution_ms"]:
                violations.append({
                    "type": "slow_execution",
                    "event_id": event["event_id"],
                    "value": metrics["execution_time_ms"],
                    "threshold": self.thresholds["tool_execution_ms"]
                })
        
        return violations
```

**Optimization Triggers:**

```python
class OptimizationTriggers:
    async def should_trigger_optimization(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        return {
            "memory_optimization": metrics.get("avg_memory_mb", 0) > 800,
            "performance_tuning": metrics.get("p95_execution_ms", 0) > 2000,
            "cache_warming": metrics.get("cache_hit_rate", 1.0) < 0.8,
            "load_balancing": metrics.get("agent_utilization", 0) > 0.9
        }
```

## Redis Streams Integration

### Stream Configuration

```python
# Event streams setup
OBSERVABILITY_STREAM = "observability_events"
CONSUMER_GROUPS = {
    "dashboard": "dashboard_group",
    "analytics": "analytics_group", 
    "performance": "performance_group",
    "alerting": "alerting_group"
}

# Stream creation with consumer groups
async def setup_observability_streams():
    redis_client = get_redis()
    
    for purpose, group_name in CONSUMER_GROUPS.items():
        try:
            await redis_client.xgroup_create(
                OBSERVABILITY_STREAM,
                group_name,
                id='0',
                mkstream=True
            )
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                raise
```

### Event Publishing Pattern

```python
async def publish_observability_event(event: BaseObservabilityEvent):
    # Serialize event
    serialized_data, metadata = serialize_for_stream(event)
    
    # Add to stream with automatic ID
    stream_id = await redis_client.xadd(
        OBSERVABILITY_STREAM,
        {
            "event_data": serialized_data,
            "event_type": event.event_type,
            "event_category": event.event_category,
            "agent_id": str(event.agent_id) if event.agent_id else "",
            "session_id": str(event.session_id) if event.session_id else "",
            "timestamp": event.timestamp.isoformat(),
            "metadata": json.dumps(metadata)
        },
        maxlen=100000  # Retain last 100k events
    )
    
    return stream_id
```

## Semantic Memory Integration

### Embedding Generation

Events with semantic content automatically generate embeddings:

```python
from app.core.embedding_service import get_embedding_service

class SemanticEventEnricher:
    async def enrich_event(self, event: BaseObservabilityEvent):
        # Generate embedding for semantic content
        if hasattr(event, 'query_text'):
            embedding = await self.embedding_service.generate_embedding(
                event.query_text
            )
            event.semantic_embedding = embedding
        
        # Enrich with contextual information
        if event.session_id:
            context = await self._get_session_context(event.session_id)
            event.payload.update({"session_context": context})
        
        return event
```

### Context-Aware Analysis

```python
class ContextualAnalyzer:
    async def analyze_with_context(self, event: BaseObservabilityEvent):
        if not event.semantic_embedding:
            return {}
        
        # Find similar events
        similar_events = await self.semantic_service.similarity_search(
            query_embedding=event.semantic_embedding,
            threshold=0.7,
            limit=10
        )
        
        # Analyze patterns
        patterns = self._extract_patterns(similar_events)
        
        return {
            "similar_events_count": len(similar_events),
            "common_patterns": patterns,
            "recommendations": self._generate_recommendations(patterns)
        }
```

## Testing and Validation

### Contract Validation Tests

```python
# Run comprehensive schema validation
pytest tests/contract/test_observability_schema.py -v

# Performance benchmark validation
python scripts/benchmark_event_performance.py --iterations 1000

# Mock event generation test
pytest tests/contract/test_observability_schema.py::TestMockEventGeneration -v
```

### Integration Tests

```python
# Redis streams integration
pytest tests/integration/test_redis_observability.py

# Semantic memory integration  
pytest tests/integration/test_semantic_observability.py

# End-to-end workflow
pytest tests/integration/test_observability_e2e.py
```

## Performance Requirements Summary

| Metric | Requirement | Measurement |
|--------|-------------|-------------|
| Serialization | < 5ms per event | Individual event processing |
| Deserialization | < 5ms per event | Individual event processing |
| Throughput | > 200 events/sec | Sustained processing rate |
| Memory Usage | < 100MB peak | During batch processing |
| P95 Latency | < 10ms | 95th percentile operations |
| Hook Overhead | < 5ms total | PreToolUse + PostToolUse |

## Error Handling and Resilience

### Event Processing Failures

```python
class ResilientEventProcessor:
    async def process_with_retry(self, event, max_retries=3):
        for attempt in range(max_retries + 1):
            try:
                return await self._process_event(event)
            except Exception as e:
                if attempt == max_retries:
                    # Send to dead letter queue
                    await self._send_to_dlq(event, str(e))
                    raise
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
```

### Schema Evolution

```python
class SchemaEvolutionHandler:
    def __init__(self):
        self.schema_versions = {
            "1.0.0": self._handle_v1_0_0,
            "1.1.0": self._handle_v1_1_0,
        }
    
    async def handle_event(self, event_data: dict):
        version = event_data.get("metadata", {}).get("schema_version", "1.0.0")
        handler = self.schema_versions.get(version, self._handle_unknown)
        
        return await handler(event_data)
```

## Migration and Deployment

### Backward Compatibility

- Schema versioning ensures backward compatibility
- New event types are additive only
- Required fields cannot be removed
- Field type changes require new schema version

### Deployment Strategy

1. **Phase 1**: Deploy event schema and serialization (completed)
2. **Phase 2**: Deploy mock event service for development
3. **Phase 3**: Parallel subagent development using contract
4. **Phase 4**: Integration testing and validation
5. **Phase 5**: Production deployment with monitoring

### Monitoring and Alerting

```python
# Event processing health metrics
PROMETHEUS_METRICS = {
    "events_processed_total": Counter("events_processed_total", ["event_type"]),
    "processing_duration_seconds": Histogram("processing_duration_seconds"),
    "processing_errors_total": Counter("processing_errors_total", ["error_type"]),
    "schema_validation_failures": Counter("schema_validation_failures", ["event_type"])
}
```

## Conclusion

This observability event schema contract provides a comprehensive foundation for coordinated subagent development in Phase 4. The schema ensures:

✅ **Complete Coverage**: All critical multi-agent system events captured  
✅ **High Performance**: <5ms processing overhead with MessagePack serialization  
✅ **Semantic Intelligence**: 1536-dimensional embeddings for context-aware analysis  
✅ **Production Ready**: Comprehensive validation, testing, and mock data generation  
✅ **Integration Patterns**: Clear guidance for all four subagents  
✅ **Extensible Design**: Schema versioning and backward compatibility  

The contract is now frozen and ready for parallel subagent development, enabling the Observability-Agent, Dashboard-Agent, Analytics-Agent, and Performance-Agent teams to work simultaneously while ensuring seamless integration.