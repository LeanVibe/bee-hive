# Vertical Slice 4.2: Redis Streams with Consumer Groups - Migration Guide

## Overview

Vertical Slice 4.2 upgrades the LeanVibe Agent Hive communication system from basic Redis Pub/Sub (VS 4.1) to enterprise-grade Redis Streams with Consumer Groups. This enhancement provides:

- **Reliable Message Delivery** with at-least-once guarantees
- **Load Balancing** across multiple consumers per agent type
- **Automatic Failure Recovery** with message claiming
- **Workflow-Aware Routing** with dependency management
- **Dead Letter Queue** handling for poison messages
- **Auto-Scaling** based on consumer group lag monitoring

## Performance Improvements

| Metric | VS 4.1 (Pub/Sub) | VS 4.2 (Consumer Groups) | Improvement |
|--------|-------------------|---------------------------|-------------|
| Message Throughput | ~1k msg/sec | ≥10k msg/sec | 10x |
| Delivery Guarantee | Fire-and-forget | At-least-once | Reliability |
| Failure Recovery | Manual | <30 seconds auto | Automated |
| Consumer Scaling | Manual | Auto-scaling | Dynamic |
| Message Persistence | Optional | 24h retention | Built-in |

## Architecture Changes

### VS 4.1 Architecture (Pub/Sub)
```
Orchestrator → Redis Pub/Sub → Individual Agents
                 ↓
            Message Loss Risk
```

### VS 4.2 Architecture (Consumer Groups)
```
Orchestrator → Workflow Router → Consumer Groups → Load Balanced Agents
                 ↓                     ↓
            Dependency Mgmt      Automatic Scaling
                 ↓                     ↓
           Dead Letter Queue    Failure Recovery
```

## Migration Steps

### Step 1: Update Dependencies

Ensure your project includes the new VS 4.2 components:

```python
from app.core.enhanced_redis_streams_manager import (
    EnhancedRedisStreamsManager, ConsumerGroupConfig, ConsumerGroupType
)
from app.core.consumer_group_coordinator import ConsumerGroupCoordinator
from app.core.workflow_message_router import WorkflowMessageRouter
from app.core.dead_letter_queue_handler import DeadLetterQueueHandler
```

### Step 2: Update Communication Service Configuration

**Before (VS 4.1):**
```python
communication_service = AgentCommunicationService(
    redis_url="redis://localhost:6379",
    enable_streams=True,
    enable_persistence=True
)
```

**After (VS 4.2):**
```python
communication_service = AgentCommunicationService(
    redis_url="redis://localhost:6379",
    enable_streams=True,  # Keep for backward compatibility
    enable_consumer_groups=True,  # NEW: Enable consumer groups
    enable_workflow_routing=True,  # NEW: Enable workflow routing
    enable_persistence=True
)
```

### Step 3: Create Consumer Groups

Define consumer groups for different agent types:

```python
# Create consumer groups for different agent specializations
await communication_service.create_consumer_group(
    group_name="backend_engineers_consumers",
    stream_name="agent_messages:backend",
    agent_type=ConsumerGroupType.BACKEND_ENGINEERS,
    routing_mode=MessageRoutingMode.LOAD_BALANCED,
    max_consumers=15,
    min_consumers=2,
    auto_scale_enabled=True
)

await communication_service.create_consumer_group(
    group_name="frontend_developers_consumers", 
    stream_name="agent_messages:frontend",
    agent_type=ConsumerGroupType.FRONTEND_DEVELOPERS,
    routing_mode=MessageRoutingMode.CAPABILITY_MATCHED,
    max_consumers=12,
    min_consumers=1,
    auto_scale_enabled=True
)
```

### Step 4: Update Agent Registration

**Before (VS 4.1):**
```python
# Agents subscribed to individual channels
await communication_service.subscribe_agent(
    agent_id="backend_agent_1",
    callback=message_handler
)
```

**After (VS 4.2):**
```python
# Agents join consumer groups for load balancing
await communication_service.join_consumer_group(
    agent_id="backend_agent_1",
    group_name="backend_engineers_consumers",
    message_handler=enhanced_message_handler
)
```

### Step 5: Update Message Sending

**Before (VS 4.1):**
```python
# Send to specific agent
message = AgentMessage(...)
await communication_service.send_message(message)
```

**After (VS 4.2):**
```python
# Send to consumer group with intelligent routing
message = AgentMessage(...)
await communication_service.send_to_consumer_group(
    group_name="backend_engineers_consumers",
    message=message,
    routing_mode=MessageRoutingMode.LOAD_BALANCED
)
```

### Step 6: Implement Workflow Routing

For complex multi-step workflows:

```python
# Define workflow with task dependencies
workflow_id = "feature_development_workflow"
tasks = [
    {
        "id": "architecture_task",
        "type": "architecture",
        "dependencies": [],
        "priority": 8
    },
    {
        "id": "backend_task", 
        "type": "backend",
        "dependencies": ["architecture_task"],
        "priority": 7
    },
    {
        "id": "frontend_task",
        "type": "frontend", 
        "dependencies": ["backend_task"],
        "priority": 6
    },
    {
        "id": "testing_task",
        "type": "testing",
        "dependencies": ["backend_task", "frontend_task"],
        "priority": 5
    }
]

# Route entire workflow with dependency management
result = await communication_service.route_workflow_message(
    workflow_id=workflow_id,
    tasks=tasks
)
```

### Step 7: Handle Task Completion

Signal task completion to trigger dependent tasks:

```python
# When a task completes, signal to trigger dependencies
triggered_tasks = await communication_service.signal_task_completion(
    workflow_id=workflow_id,
    task_id="architecture_task",
    result={"status": "completed", "artifacts": [...]}
)

print(f"Triggered {len(triggered_tasks)} dependent tasks")
```

## API Endpoints

VS 4.2 includes new REST API endpoints for consumer group management:

### Consumer Group Management
```bash
# List all consumer groups
GET /api/v1/consumer-groups/

# Create consumer group
POST /api/v1/consumer-groups/
{
  "name": "qa_engineers_consumers",
  "stream_name": "agent_messages:qa",
  "agent_type": "qa_engineers",
  "max_consumers": 8,
  "auto_scale_enabled": true
}

# Get consumer group details
GET /api/v1/consumer-groups/{group_name}

# Add consumer to group
POST /api/v1/consumer-groups/{group_name}/consumers?consumer_id=qa_agent_1

# Remove consumer from group
DELETE /api/v1/consumer-groups/{group_name}/consumers/{consumer_id}
```

### Message Routing
```bash
# Route message to consumer group
POST /api/v1/consumer-groups/route
{
  "message_id": "task_123",
  "from_agent": "orchestrator",
  "message_type": "task_request",
  "payload": {"task": "run_tests"},
  "workflow_id": "test_workflow_1",
  "dependencies": ["build_task"]
}

# Route entire workflow
POST /api/v1/consumer-groups/route/workflow
{
  "workflow_id": "deploy_workflow",
  "tasks": [...]
}
```

### Dead Letter Queue Management
```bash
# Get DLQ statistics
GET /api/v1/consumer-groups/dlq/statistics

# Replay failed message
POST /api/v1/consumer-groups/dlq/replay/{dlq_message_id}?priority_boost=true

# Batch replay with filters
POST /api/v1/consumer-groups/dlq/replay/batch
{
  "filter_criteria": {"failure_category": "timeout"},
  "max_messages": 10,
  "priority_boost": true
}

# Analyze failure patterns
GET /api/v1/consumer-groups/dlq/analyze
```

### Monitoring and Metrics
```bash
# Get comprehensive metrics
GET /api/v1/consumer-groups/metrics

# Health check
GET /api/v1/consumer-groups/health

# Trigger rebalancing
POST /api/v1/consumer-groups/rebalance
```

## Configuration Options

### Enhanced Redis Streams Manager
```python
EnhancedRedisStreamsManager(
    redis_url="redis://localhost:6379",
    connection_pool_size=20,
    default_stream_maxlen=1000000,
    default_batch_size=10,
    health_check_interval=30,
    metrics_collection_interval=60,
    auto_scaling_enabled=True
)
```

### Consumer Group Coordinator
```python
ConsumerGroupCoordinator(
    streams_manager=streams_manager,
    strategy=ConsumerGroupStrategy.HYBRID,
    provisioning_policy=ProvisioningPolicy.PREDICTIVE,
    allocation_mode=ResourceAllocationMode.ADAPTIVE,
    health_check_interval=30,
    rebalance_interval=300,
    enable_cross_group_coordination=True
)
```

### Workflow Message Router
```python
WorkflowMessageRouter(
    streams_manager=streams_manager,
    coordinator=coordinator,
    default_strategy=WorkflowRoutingStrategy.HYBRID,
    dependency_mode=DependencyResolutionMode.STRICT,
    max_parallel_tasks=50,
    routing_timeout_seconds=30,
    enable_workflow_optimization=True
)
```

### Dead Letter Queue Handler
```python
DeadLetterQueueHandler(
    streams_manager=streams_manager,
    dlq_stream_suffix=":dlq",
    analysis_interval=300,
    cleanup_interval=3600,
    max_dlq_age_days=7,
    enable_automatic_recovery=True,
    recovery_batch_size=10
)
```

## Testing

### Run Unit Tests
```bash
# Test consumer groups functionality
pytest tests/test_redis_streams_consumer_groups.py -v

# Test specific components
pytest tests/test_redis_streams_consumer_groups.py::TestEnhancedRedisStreamsManager -v
pytest tests/test_redis_streams_consumer_groups.py::TestConsumerGroupCoordinator -v
pytest tests/test_redis_streams_consumer_groups.py::TestWorkflowMessageRouter -v
pytest tests/test_redis_streams_consumer_groups.py::TestDeadLetterQueueHandler -v
```

### Run Performance Validation
```bash
# Validate all performance targets
python scripts/validate_vs_4_2_performance.py --verbose

# Test against custom Redis instance
python scripts/validate_vs_4_2_performance.py --redis-url redis://localhost:6380/0
```

Expected performance targets:
- **Message Throughput**: ≥10k messages/second sustained
- **Consumer Lag**: <5 seconds under normal load
- **Failure Recovery**: <30 seconds to reassign stalled messages
- **Consumer Group Join**: <1 second for new consumer registration
- **DLQ Processing**: <10 seconds to move poison messages

## Monitoring and Observability

### Key Metrics to Monitor

1. **Consumer Group Metrics**
   - Consumer count per group
   - Message lag per group
   - Throughput (messages/second)
   - Success rate (%)
   - Average processing time

2. **Workflow Routing Metrics**
   - Workflows routed
   - Tasks routed successfully
   - Dependency violations
   - Routing latency

3. **Dead Letter Queue Metrics**
   - Messages in DLQ
   - Recovery success rate
   - Failure patterns
   - DLQ growth rate

### Grafana Dashboard

Import the provided Grafana dashboard (`monitoring/grafana/dashboards/leanvibe-consumer-groups.json`) to visualize:

- Consumer group health and performance
- Message throughput and latency
- DLQ statistics and trends
- Auto-scaling events
- Failure recovery metrics

### Alerting Rules

Configure alerts for:
- Consumer group lag > 1000 messages
- DLQ message count > 100
- Consumer group success rate < 95%
- Workflow routing failures > 5%
- Auto-scaling events

## Troubleshooting

### Common Issues

1. **High Consumer Lag**
   - **Cause**: Insufficient consumers for workload
   - **Solution**: Enable auto-scaling or manually increase max_consumers
   - **Prevention**: Monitor lag trends and adjust thresholds

2. **Messages in DLQ**
   - **Cause**: Poison messages or handler failures
   - **Solution**: Analyze failure patterns and fix root cause
   - **Prevention**: Implement robust error handling in message handlers

3. **Consumer Join Failures**
   - **Cause**: Redis connection issues or resource limits
   - **Solution**: Check Redis health and consumer group limits
   - **Prevention**: Monitor Redis performance and connection pool size

4. **Workflow Dependencies Not Triggering**
   - **Cause**: Task completion not properly signaled
   - **Solution**: Ensure task completion is signaled with correct workflow_id
   - **Prevention**: Add logging for task completion events

### Debug Commands

```bash
# Check consumer group status
curl http://localhost:8000/api/v1/consumer-groups/

# Get specific group details
curl http://localhost:8000/api/v1/consumer-groups/backend_engineers_consumers

# Check DLQ messages
curl http://localhost:8000/api/v1/consumer-groups/dlq/statistics

# Health check
curl http://localhost:8000/api/v1/consumer-groups/health
```

### Log Analysis

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger('app.core.enhanced_redis_streams_manager').setLevel(logging.DEBUG)
logging.getLogger('app.core.consumer_group_coordinator').setLevel(logging.DEBUG)
logging.getLogger('app.core.workflow_message_router').setLevel(logging.DEBUG)
logging.getLogger('app.core.dead_letter_queue_handler').setLevel(logging.DEBUG)
```

## Best Practices

### Consumer Group Design

1. **Group by Agent Type**: Create consumer groups aligned with agent specializations
2. **Set Appropriate Limits**: Configure min/max consumers based on expected workload
3. **Enable Auto-Scaling**: Allow dynamic scaling based on message lag
4. **Monitor Performance**: Track group metrics and adjust configurations

### Message Design

1. **Include Workflow Context**: Add workflow_id and dependencies for complex flows
2. **Set Appropriate Priority**: Use priority levels to ensure critical tasks are processed first
3. **Add Correlation IDs**: Include correlation_id for request tracing
4. **Keep Payloads Lightweight**: Minimize message size for better performance

### Error Handling

1. **Implement Robust Handlers**: Handle exceptions gracefully in message handlers
2. **Use Exponential Backoff**: Implement backoff for retryable failures
3. **Monitor DLQ Growth**: Alert on DLQ message accumulation
4. **Regular DLQ Analysis**: Analyze failure patterns to improve system resilience

### Performance Optimization

1. **Batch Processing**: Process messages in batches where possible
2. **Connection Pooling**: Use appropriate connection pool sizes
3. **Message Batching**: Send messages in batches for higher throughput
4. **Resource Monitoring**: Monitor Redis memory and CPU usage

## Rollback Plan

If issues arise with VS 4.2, you can rollback to VS 4.1:

1. **Disable Enhanced Features**:
   ```python
   communication_service = AgentCommunicationService(
       enable_consumer_groups=False,
       enable_workflow_routing=False,
       enable_streams=True  # Keep basic streams
   )
   ```

2. **Revert Agent Registration**:
   ```python
   # Switch back to individual agent subscriptions
   await communication_service.subscribe_agent(agent_id, callback)
   ```

3. **Update Message Sending**:
   ```python
   # Use direct message sending
   await communication_service.send_message(message)
   ```

4. **Monitor Transition**: Ensure all messages are processed during rollback

## Support and Resources

- **Documentation**: `/docs/VS_4_2_MIGRATION_GUIDE.md` (this file)
- **API Reference**: `/app/api/v1/consumer_groups.py`
- **Test Examples**: `/tests/test_redis_streams_consumer_groups.py`
- **Performance Tests**: `/scripts/validate_vs_4_2_performance.py`
- **Monitoring**: `/monitoring/grafana/dashboards/leanvibe-consumer-groups.json`

For additional support, refer to the project's issue tracker or contact the development team.