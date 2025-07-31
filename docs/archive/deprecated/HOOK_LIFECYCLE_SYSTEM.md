# Hook Lifecycle System Documentation

## Overview

The Python-based Hook Lifecycle System provides comprehensive lifecycle event tracking with security safeguards, event aggregation, and real-time streaming capabilities for the LeanVibe Agent Hive 2.0.

## Architecture

### Core Components

#### 1. HookLifecycleSystem
- **Location**: `app/core/hook_lifecycle_system.py`
- **Purpose**: Central orchestrator for all hook processing
- **Features**:
  - Python-based hook processing with async support
  - Security validation integration
  - Event aggregation and batching
  - WebSocket streaming for real-time dashboard updates
  - Redis Streams integration for event persistence

#### 2. SecurityValidator
- **Purpose**: Dangerous command detection and blocking
- **Features**:
  - Configurable dangerous command patterns
  - Risk-based command classification (SAFE, LOW, MEDIUM, HIGH, CRITICAL)
  - Custom validator support
  - Caching for performance optimization

#### 3. EventAggregator
- **Purpose**: High-frequency event batching and aggregation
- **Features**:
  - Intelligent batching by priority
  - Configurable flush intervals
  - Custom aggregation rules
  - Background processing

#### 4. WebSocketStreamer
- **Purpose**: Real-time event streaming to dashboard clients
- **Features**:
  - Connection management
  - Event filtering
  - Broadcast optimization
  - Error handling and disconnection management

### Integration Components

#### 1. OrchestratorHookIntegration
- **Location**: `app/core/orchestrator_hook_integration.py`
- **Purpose**: Seamless integration with AgentOrchestrator
- **Features**:
  - Automatic hook generation for orchestrator operations
  - Method patching for transparent integration
  - Context managers for operation wrapping
  - Performance and lifecycle tracking

#### 2. Performance Benchmarks
- **Location**: `app/core/hook_performance_benchmarks.py`
- **Purpose**: Comprehensive performance validation
- **Features**:
  - <50ms processing requirement validation
  - Load testing capabilities
  - Detailed performance reporting
  - SLA compliance monitoring

## Security Features

### Dangerous Command Detection

The system includes comprehensive dangerous command detection with configurable patterns:

**Critical Risk Commands** (Blocked by default):
- `rm -rf /` - Recursive delete from root
- `sudo rm -rf` - Sudo recursive delete
- `mkfs.*` - Format filesystem
- `dd if=.*of=/dev/` - Direct disk operations
- Fork bomb patterns

**High Risk Commands** (Require approval):
- `sudo` commands
- `chmod 777` - Dangerous permissions
- Firewall modifications
- Download and execute scripts

**Medium Risk Commands** (Require approval):
- Force delete system files
- Cron job removal
- Process termination

### Security Configuration

```python
# Add custom dangerous command
validator.add_dangerous_command(DangerousCommand(
    pattern=r'custom_dangerous_pattern',
    risk_level=SecurityRisk.HIGH,
    description="Custom dangerous operation",
    block_execution=True
))

# Add custom validator
async def custom_validator(command: str) -> Tuple[bool, SecurityRisk, str]:
    # Custom validation logic
    return is_dangerous, risk_level, reason

validator.add_custom_validator(custom_validator)
```

## Performance Requirements

### SLA Targets
- **Average Processing Time**: <50ms
- **P95 Processing Time**: <75ms
- **Success Rate**: >95%
- **Throughput**: >100 events/second per component

### Benchmarking

Run performance validation:

```python
from app.core.hook_performance_benchmarks import run_performance_validation

# Validate performance requirements
meets_requirements = await run_performance_validation()
```

### Performance Optimizations

1. **Caching**: Command validation results cached for 5 minutes
2. **Batching**: Events aggregated in batches up to 100 items
3. **Async Processing**: All operations use async/await patterns
4. **Connection Pooling**: Redis connections pooled and reused
5. **Memory Management**: Limited cache sizes with LRU eviction

## Usage Examples

### Basic Hook Processing

```python
from app.core.hook_lifecycle_system import get_hook_lifecycle_system, HookType

# Get hook system instance
hook_system = await get_hook_lifecycle_system()

# Process PreToolUse hook
result = await hook_system.process_hook(
    hook_type=HookType.PRE_TOOL_USE,
    agent_id=agent_id,
    session_id=session_id,
    payload={
        "tool_name": "my_tool",
        "parameters": {"param1": "value1"}
    },
    priority=3
)

# Check result
if result.success:
    print(f"Hook processed in {result.processing_time_ms}ms")
else:
    print(f"Hook failed: {result.error}")
```

### Convenience Functions

```python
from app.core.hook_lifecycle_system import (
    process_pre_tool_use_hook,
    process_post_tool_use_hook,
    process_stop_hook,
    process_notification_hook
)

# PreToolUse hook
await process_pre_tool_use_hook(
    agent_id=agent_id,
    session_id=session_id,
    tool_name="example_tool",
    parameters={"key": "value"}
)

# PostToolUse hook
await process_post_tool_use_hook(
    agent_id=agent_id,
    session_id=session_id,
    tool_name="example_tool",
    success=True,
    result="Operation completed",
    execution_time_ms=123.45
)

# Stop hook
await process_stop_hook(
    agent_id=agent_id,
    session_id=session_id,
    reason="Task completed",
    details={"completion_status": "success"}
)

# Notification hook
await process_notification_hook(
    agent_id=agent_id,
    session_id=session_id,
    level="info",
    message="Operation status update",
    details={"progress": 75}
)
```

### Orchestrator Integration

```python
from app.core.orchestrator_hook_integration import initialize_orchestrator_hooks

# Initialize orchestrator with hook integration
orchestrator = AgentOrchestrator()
await orchestrator.start()

# Initialize hook integration (automatic hook generation)
integration = await initialize_orchestrator_hooks(orchestrator)

# Spawn agent (hooks automatically generated)
agent_id = await orchestrator.spawn_agent(AgentRole.STRATEGIC_PARTNER)

# Custom operation with hook context
async with integration.hook_context("custom_operation", agent_id) as ctx:
    result = await perform_custom_operation()
    ctx.set_result(result)

# Shutdown (hooks automatically generated)
await orchestrator.shutdown_agent(agent_id)
```

### WebSocket Dashboard Integration

```python
from fastapi import WebSocket
from app.core.hook_lifecycle_system import get_hook_lifecycle_system

@app.websocket("/hooks/stream")
async def websocket_endpoint(websocket: WebSocket):
    hook_system = await get_hook_lifecycle_system()
    
    # Connect client with filters
    filters = {
        "agent_ids": ["agent-123"],
        "hook_types": ["PreToolUse", "PostToolUse"],
        "min_priority": 3
    }
    
    await hook_system.websocket_streamer.connect(websocket, filters)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except Exception:
        await hook_system.websocket_streamer.disconnect(websocket)
```

## Configuration

### Environment Variables

```bash
# Hook system configuration
HOOK_BATCH_SIZE=100
HOOK_FLUSH_INTERVAL_MS=1000
HOOK_PERFORMANCE_THRESHOLD_MS=50
HOOK_MAX_PAYLOAD_SIZE=100000

# Redis configuration for hook streaming
REDIS_URL=redis://localhost:6379
REDIS_HOOK_STREAMS_TTL=86400

# Security configuration
HOOK_SECURITY_VALIDATION_ENABLED=true
HOOK_DANGEROUS_COMMAND_BLOCKING=true
```

### System Configuration

```python
# Configure hook system
hook_system = await get_hook_lifecycle_system()

# Update configuration
hook_system.config.update({
    "enable_security_validation": True,
    "enable_event_aggregation": True,
    "enable_websocket_streaming": True,
    "enable_redis_streaming": True,
    "performance_threshold_ms": 50.0
})

# Configure security validator
validator = hook_system.security_validator
validator.add_dangerous_command(custom_command)

# Configure event aggregator
aggregator = hook_system.event_aggregator
aggregator.add_aggregation_rule("CustomEvent", custom_aggregator)
```

## Monitoring and Observability

### Metrics

The system provides comprehensive metrics for monitoring:

```python
# Get comprehensive metrics
metrics = hook_system.get_comprehensive_metrics()

# Key metrics include:
# - hooks_processed: Total hooks processed
# - hooks_blocked: Hooks blocked by security
# - avg_processing_time_ms: Average processing time
# - performance_threshold_violations: SLA violations
# - events_aggregated: Events processed by aggregator
# - active_connections: WebSocket connections
# - security_validations: Command validations performed
```

### Logging

Structured logging with contextual information:

```python
import structlog

logger = structlog.get_logger()

# Automatic logging includes:
# - Hook type and processing time
# - Agent and session IDs
# - Security decisions and blocked commands
# - Performance metrics and SLA violations
# - Error details and stack traces
```

### Health Checks

```python
# Health check endpoint
@app.get("/health/hooks")
async def hooks_health():
    hook_system = await get_hook_lifecycle_system()
    metrics = hook_system.get_comprehensive_metrics()
    
    # Check system health
    avg_time = metrics["hook_lifecycle_system"]["avg_processing_time_ms"]
    violations = metrics["hook_lifecycle_system"]["performance_threshold_violations"]
    
    is_healthy = avg_time < 50.0 and violations < 10
    
    return {
        "status": "healthy" if is_healthy else "degraded",
        "avg_processing_time_ms": avg_time,
        "performance_violations": violations,
        "metrics": metrics
    }
```

## Error Handling

### Graceful Degradation

The system is designed for graceful degradation:

1. **Security Validation Failure**: Defaults to DENY for safety
2. **Event Aggregation Failure**: Falls back to individual processing
3. **WebSocket Streaming Failure**: Continues processing without streaming
4. **Redis Integration Failure**: Continues with in-memory processing

### Error Recovery

```python
# Automatic error recovery
try:
    result = await hook_system.process_hook(...)
except Exception as e:
    # System automatically logs error and continues
    # Failed operations are retried with exponential backoff
    # Circuit breaker prevents cascade failures
    pass
```

## Testing

### Unit Tests

```bash
# Run hook lifecycle system tests
pytest tests/core/test_hook_lifecycle_system.py -v

# Run security validator tests
pytest tests/core/test_security_validator.py -v

# Run integration tests
pytest tests/integration/test_orchestrator_hooks.py -v
```

### Performance Tests

```bash
# Run performance benchmarks
python -m app.core.hook_performance_benchmarks

# Run load tests
pytest tests/performance/test_hook_load.py -v
```

### Security Tests

```bash
# Run security validation tests
pytest tests/security/test_dangerous_commands.py -v

# Run penetration tests
pytest tests/security/test_command_injection.py -v
```

## Deployment

### Docker Configuration

```dockerfile
# Hook system requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Environment configuration
ENV HOOK_BATCH_SIZE=100
ENV HOOK_FLUSH_INTERVAL_MS=1000
ENV HOOK_PERFORMANCE_THRESHOLD_MS=50

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health/hooks || exit 1
```

### Production Considerations

1. **Resource Allocation**: Allocate sufficient CPU and memory for high-throughput scenarios
2. **Redis Configuration**: Configure Redis with appropriate memory and persistence settings
3. **Monitoring**: Set up alerts for performance threshold violations
4. **Security Updates**: Regularly update dangerous command patterns
5. **Capacity Planning**: Monitor throughput and scale horizontally as needed

## Troubleshooting

### Common Issues

**High Processing Times**:
- Check Redis connectivity and performance
- Review security validation patterns (may be too complex)
- Monitor system resource usage
- Consider increasing batch sizes

**Security False Positives**:
- Review dangerous command patterns
- Add custom validators for specific use cases
- Implement whitelisting for trusted operations
- Adjust risk level thresholds

**WebSocket Connection Issues**:
- Check client connection filters
- Monitor connection pool limits
- Review network configuration
- Validate WebSocket protocol compliance

**Event Aggregation Problems**:
- Check aggregation rule logic
- Monitor batch processing times
- Review flush interval configuration
- Validate event serialization

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger("app.core.hook_lifecycle_system").setLevel(logging.DEBUG)

# Enable performance profiling
hook_system.config["enable_performance_profiling"] = True

# Enable security audit mode
hook_system.security_validator.config["audit_mode"] = True
```

## Support and Maintenance

### Regular Maintenance Tasks

1. **Performance Review**: Weekly performance metrics analysis
2. **Security Updates**: Monthly dangerous command pattern updates
3. **Capacity Planning**: Quarterly throughput and scaling analysis
4. **Health Monitoring**: Daily health check and alert review

### Support Contacts

- **System Architecture**: DevOps Team
- **Security Issues**: Security Team
- **Performance Issues**: Platform Engineering Team
- **Integration Support**: Development Team

---

This documentation provides comprehensive guidance for implementing, configuring, and maintaining the Hook Lifecycle System in production environments.