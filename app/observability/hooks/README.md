# LeanVibe Agent Hive 2.0 - Observability Hooks

Claude Code integration hooks for comprehensive system observability including tool execution monitoring, session lifecycle tracking, and performance optimization.

## Overview

This hook system provides comprehensive observability for Claude Code sessions by capturing:

- **Pre-tool-use events**: Tool execution initiation, parameter validation, performance baselines
- **Post-tool-use events**: Tool results, performance metrics, error tracking
- **Session lifecycle events**: Session start/end, sleep/wake cycles, memory consolidation

## Components

### Hook Scripts

1. **`pre_tool_use.py`** - Captures tool execution initiation events
2. **`post_tool_use.py`** - Captures tool completion events with performance analysis
3. **`session_lifecycle.py`** - Manages session start/end, sleep/wake cycles

### Integration Components

4. **`hooks_config.py`** - Centralized configuration system
5. **`hooks_integration.py`** - Integration manager with existing observability infrastructure

## Usage

### Direct Script Execution

```bash
# Pre-tool-use hook via JSON input
echo '{"tool_name": "test_tool", "parameters": {"action": "test"}}' | python app/observability/hooks/pre_tool_use.py

# Post-tool-use hook via JSON input
echo '{"tool_name": "test_tool", "success": true, "execution_time_ms": 150}' | python app/observability/hooks/post_tool_use.py

# Session lifecycle events via command line arguments
python app/observability/hooks/session_lifecycle.py session_start
python app/observability/hooks/session_lifecycle.py session_end normal_completion
python app/observability/hooks/session_lifecycle.py sleep context_threshold_reached
python app/observability/hooks/session_lifecycle.py wake new_session_start
```

### Programmatic Integration

```python
from app.observability.hooks.hooks_integration import get_hook_integration_manager

# Get integration manager
integration_manager = get_hook_integration_manager()

# Capture tool execution
results = await integration_manager.capture_tool_execution(
    tool_name="example_tool",
    parameters={"param1": "value1"},
    execution_result={
        "success": True,
        "execution_time_ms": 250,
        "result": "Tool completed successfully"
    }
)

# Capture session lifecycle event
event_id = await integration_manager.capture_session_lifecycle_event(
    event_type="session_start",
    context_data={"initial_context": "value"}
)
```

## Configuration

### Environment Variables

Hook behavior is configured via environment variables:

```bash
# Hook enablement
export ENABLE_PRE_TOOL_USE_HOOK=true
export ENABLE_POST_TOOL_USE_HOOK=true
export ENABLE_SESSION_LIFECYCLE_HOOK=true
export ENABLE_ERROR_HOOKS=true

# Session identification
export CLAUDE_SESSION_ID="uuid-string"
export CLAUDE_AGENT_ID="uuid-string"

# Performance thresholds
export HOOK_SLOW_TOOL_THRESHOLD_MS=5000
export HOOK_VERY_SLOW_TOOL_THRESHOLD_MS=15000

# Security settings
export HOOK_MAX_PAYLOAD_SIZE=100000
export HOOK_SANITIZE_SENSITIVE_DATA=true

# Integration settings
export HOOK_USE_DATABASE=true
export HOOK_USE_REDIS_STREAMS=true
export HOOK_USE_PROMETHEUS=true
export HOOK_WEBHOOK_URLS="http://webhook1.com,http://webhook2.com"
```

### Configuration Environments

The system automatically adjusts settings based on the `ENVIRONMENT` variable:

- **`production`**: Strict thresholds, optimized for performance
- **`development`**: Relaxed thresholds, optimized for debugging
- **`testing`**: Fast feedback, minimal overhead

## Event Schema

### Pre-Tool-Use Event
```json
{
  "tool_name": "string",
  "parameters": {},
  "timestamp": "ISO8601",
  "correlation_id": "uuid",
  "hook_version": "1.0",
  "environment": "development|production|testing",
  "performance_monitoring": {
    "slow_threshold_ms": 5000,
    "very_slow_threshold_ms": 15000,
    "start_timestamp": 1234567890.123
  }
}
```

### Post-Tool-Use Event
```json
{
  "tool_name": "string",
  "success": true,
  "result": "string|object",
  "error": "string?",
  "error_type": "string?",
  "execution_time_ms": 150,
  "timestamp": "ISO8601",
  "correlation_id": "uuid",
  "hook_version": "1.0",
  "environment": "development|production|testing",
  "performance_analysis": {
    "category": "normal|slow|very_slow|fast",
    "is_slow": false,
    "is_very_slow": false,
    "recommendation": "string?"
  },
  "result_metadata": {
    "type": "string",
    "length": 123,
    "estimated_size_bytes": 456
  }
}
```

### Session Lifecycle Event
```json
{
  "event_subtype": "session_start|session_end|sleep|wake",
  "timestamp": "ISO8601",
  "session_duration_ms": 12345,
  "reason": "string",
  "environment": "development|production|testing",
  "context_stats": {
    "usage_percent": 85,
    "total_events": 100
  },
  "memory_usage_mb": 123.45
}
```

## Performance Optimizations

### Batching and Buffering

The hook system includes several performance optimizations:

- **Event batching**: Multiple events processed together for efficiency
- **Buffer management**: Events buffered before database writes
- **Asynchronous processing**: Non-blocking event capture
- **Timeout management**: Scripts timeout after 30 seconds to prevent hanging

### Resource Management

- **Memory monitoring**: Track and alert on memory usage
- **Payload size limits**: Prevent oversized event payloads
- **Connection pooling**: Reuse database and Redis connections
- **Graceful degradation**: Continue operation if individual components fail

## Error Handling

The hook system includes comprehensive error handling:

- **Script failures**: Logged but don't block tool execution
- **Database errors**: Fallback to local logging
- **Redis errors**: Continue without stream publishing
- **Timeout handling**: Scripts killed after 30-second timeout
- **Invalid data**: Sanitized and truncated automatically

## Security Features

### Data Sanitization

- **Sensitive pattern detection**: Automatically redact passwords, tokens, keys
- **Payload size limits**: Prevent payload bloat attacks
- **Input validation**: Validate all input data
- **Output sanitization**: Clean data before storage

### Access Control

- **Environment isolation**: Production/development separation
- **Process isolation**: Each script runs in isolated process
- **Resource limits**: Memory and CPU usage limits
- **Audit logging**: All actions logged with timestamps

## Monitoring and Alerting

### Metrics

The hook system exposes metrics for monitoring:

- `hook_executions_total{type, success}`: Total hook executions
- `hook_execution_duration_ms{type}`: Hook execution time
- `tool_executions_total{tool_name, success}`: Tool execution counts
- `tool_execution_duration_ms{tool_name}`: Tool execution time
- `session_lifecycle_events_total{type}`: Session event counts

### Alerts

Automatic alerts are triggered for:

- Tools taking longer than configured thresholds
- High error rates (>10% by default)
- Memory usage exceeding thresholds
- Hook script failures or timeouts
- Database or Redis connectivity issues

## Integration Status

Check integration health programmatically:

```python
from app.observability.hooks.hooks_integration import get_hook_integration_manager

integration_manager = get_hook_integration_manager()
status = await integration_manager.get_integration_status()

print(f"Hooks enabled: {status['hooks_enabled']}")
print(f"Scripts valid: {status['script_validation']}")
print(f"Integration health: {status['integration_health']}")
```

## Troubleshooting

### Common Issues

1. **Scripts not executable**: Run `chmod +x app/observability/hooks/*.py`
2. **UUID format errors**: Ensure `CLAUDE_SESSION_ID` and `CLAUDE_AGENT_ID` are valid UUIDs
3. **Database connection errors**: Expected when running scripts standalone
4. **Redis connection errors**: Expected when running scripts standalone
5. **Import errors**: Ensure you're running from the project root directory

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=debug
python app/observability/hooks/pre_tool_use.py
```

### Validation

Validate hook configuration:

```python
from app.observability.hooks.hooks_config import get_hook_config

config = get_hook_config()
validation_results = config.validate_configuration()
print("All valid:", all(validation_results.values()))
print("Results:", validation_results)
```

## Development

### Adding New Hooks

1. Create new hook script in `app/observability/hooks/`
2. Follow existing patterns for error handling and configuration
3. Add configuration options to `hooks_config.py`
4. Add integration methods to `hooks_integration.py`
5. Add tests to `tests/test_observability_hooks_integration.py`

### Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/test_observability_hooks_integration.py -v
```

Run individual tests:

```bash
python -m pytest tests/test_observability_hooks_integration.py::TestHookConfig::test_hook_config_initialization -v
```

## Architecture

The hook system is designed with:

- **Modularity**: Each hook is independent and can be enabled/disabled
- **Performance**: Asynchronous processing with minimal overhead
- **Reliability**: Comprehensive error handling and fallback mechanisms
- **Scalability**: Batch processing and resource optimization
- **Security**: Data sanitization and access controls
- **Observability**: Full metrics and logging integration

## License

This hook system is part of LeanVibe Agent Hive 2.0 and is subject to the same license terms.