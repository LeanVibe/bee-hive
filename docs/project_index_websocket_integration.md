# Project Index WebSocket Integration

## Overview

The Project Index WebSocket integration provides real-time communication capabilities for the LeanVibe Agent Hive project analysis system. It enables live updates for analysis progress, dependency changes, and context optimization results, delivering responsive dashboards and real-time collaboration features.

## Architecture

### Core Components

1. **Event Publisher** (`websocket_events.py`)
   - Publishes 4 main event types: `project_index_updated`, `analysis_progress`, `dependency_changed`, `context_optimized`
   - Manages subscriptions and event routing
   - Provides rate limiting and performance optimization

2. **Event Filtering** (`event_filters.py`)
   - Advanced filtering based on relevance scores
   - User preferences and custom filter rules
   - Performance impact filtering
   - Language and file pattern filtering

3. **Event History** (`event_history.py`)
   - Multi-tier storage (memory, Redis, database)
   - Event replay for client reconnection
   - Configurable persistence levels
   - Automatic cleanup and TTL management

4. **Performance Optimization** (`websocket_performance.py`)
   - Event batching and compression
   - Rate limiting with token bucket algorithm
   - Connection pooling and health tracking
   - Priority queue management

5. **WebSocket Integration** (`websocket_integration.py`)
   - Enhanced WebSocket manager
   - Message handling and routing
   - Client lifecycle management
   - Distributed event processing via Redis

6. **Monitoring System** (`websocket_monitoring.py`)
   - Prometheus metrics integration
   - Health checks and alerting
   - Performance analysis and anomaly detection
   - Comprehensive dashboard data

## Event Types

### 1. Project Index Updated (`project_index_updated`)

Triggered when project analysis is complete or updated.

**Event Structure:**
```json
{
  "type": "project_index_updated",
  "data": {
    "project_id": "uuid",
    "project_name": "string",
    "files_analyzed": 150,
    "files_updated": 75,
    "dependencies_updated": 25,
    "analysis_duration_seconds": 30.5,
    "status": "completed",
    "statistics": {
      "total_files": 150,
      "languages_detected": ["python", "javascript"],
      "dependency_count": 25,
      "complexity_score": 0.7
    },
    "error_count": 0,
    "warnings": []
  },
  "timestamp": "2025-01-15T12:00:00Z",
  "correlation_id": "uuid"
}
```

### 2. Analysis Progress (`analysis_progress`)

Real-time progress updates during analysis operations.

**Event Structure:**
```json
{
  "type": "analysis_progress",
  "data": {
    "session_id": "uuid",
    "project_id": "uuid",
    "analysis_type": "full",
    "progress_percentage": 75,
    "files_processed": 75,
    "total_files": 100,
    "current_file": "src/main.py",
    "estimated_completion": "2025-01-15T12:05:00Z",
    "processing_rate": 2.5,
    "performance_metrics": {
      "memory_usage_mb": 150.0,
      "cpu_usage_percent": 45.0,
      "parallel_tasks": 4
    },
    "errors_encountered": 1,
    "last_error": "File not found: missing.py"
  },
  "timestamp": "2025-01-15T12:00:00Z",
  "correlation_id": "uuid"
}
```

### 3. Dependency Changed (`dependency_changed`)

Notifies when dependency relationships change due to file modifications.

**Event Structure:**
```json
{
  "type": "dependency_changed",
  "data": {
    "project_id": "uuid",
    "file_path": "src/utils.py",
    "change_type": "modified",
    "dependency_details": {
      "target_file": "src/core.py",
      "target_external": null,
      "relationship_type": "import",
      "line_number": 15,
      "is_circular": false
    },
    "impact_analysis": {
      "affected_files": ["src/main.py", "src/app.py"],
      "potential_issues": ["breaking_change"],
      "recommendations": ["update_imports"]
    },
    "file_metadata": {
      "language": "python",
      "file_size": 2048,
      "last_modified": "2025-01-15T11:59:00Z"
    }
  },
  "timestamp": "2025-01-15T12:00:00Z",
  "correlation_id": "uuid"
}
```

### 4. Context Optimized (`context_optimized`)

Notifies when AI context optimization is complete.

**Event Structure:**
```json
{
  "type": "context_optimized",
  "data": {
    "context_id": "uuid",
    "project_id": "uuid",
    "task_description": "Optimize context for bug analysis",
    "task_type": "bug_analysis",
    "optimization_results": {
      "selected_files": 25,
      "total_tokens": 15000,
      "relevance_scores": {
        "high": 10,
        "medium": 15,
        "low": 5
      },
      "confidence_score": 0.92,
      "processing_time_ms": 250
    },
    "recommendations": {
      "architectural_patterns": ["mvc", "observer"],
      "potential_challenges": ["tight_coupling"],
      "suggested_approach": "Focus on controller and model layers"
    },
    "performance_metrics": {
      "cache_hit_rate": 0.88,
      "ml_analysis_time_ms": 180,
      "context_assembly_time_ms": 70
    }
  },
  "timestamp": "2025-01-15T12:00:00Z",
  "correlation_id": "uuid"
}
```

## Client Usage

### WebSocket Connection

Connect to the WebSocket endpoint:

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/project-index/websocket');

ws.onopen = function(event) {
    console.log('WebSocket connected');
    
    // Subscribe to events
    ws.send(JSON.stringify({
        action: 'subscribe',
        event_types: ['project_index_updated', 'analysis_progress'],
        project_id: 'your-project-uuid'
    }));
};

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('Received event:', message);
    
    switch(message.type) {
        case 'project_index_updated':
            handleProjectUpdate(message.data);
            break;
        case 'analysis_progress':
            handleProgressUpdate(message.data);
            break;
        case 'dependency_changed':
            handleDependencyChange(message.data);
            break;
        case 'context_optimized':
            handleContextOptimization(message.data);
            break;
    }
};
```

### Subscription Management

**Subscribe to specific events:**
```json
{
  "action": "subscribe",
  "event_types": ["project_index_updated", "analysis_progress"],
  "project_id": "uuid",
  "session_id": "uuid",
  "filters": {
    "min_progress_threshold": 25,
    "high_impact_only": true
  }
}
```

**Set user preferences:**
```json
{
  "action": "set_preferences",
  "preferences": {
    "preferred_languages": ["python", "javascript"],
    "ignored_file_patterns": ["*.test.js", "*.spec.py"],
    "min_progress_updates": 25,
    "high_impact_only": false,
    "notification_frequency": "normal"
  }
}
```

**Request event replay:**
```json
{
  "action": "replay",
  "project_id": "uuid",
  "since": "2025-01-15T11:00:00Z",
  "max_events": 50,
  "event_types": ["project_index_updated"]
}
```

## Performance Features

### Event Batching

Events are automatically batched for efficiency:
- **Batch size limit**: 10 events per batch
- **Time limit**: 100ms maximum batch time
- **Size limit**: 8KB maximum batch size

### Compression

Large events are automatically compressed:
- **Threshold**: 1KB minimum for compression
- **Algorithm**: GZIP compression
- **Benefit check**: Only compressed if >20% size reduction

### Rate Limiting

Prevents event flooding:
- **Default limit**: 100 events per minute per connection
- **Token bucket**: Refills at 10 tokens per second
- **Violations tracked**: Logged for monitoring

### Connection Health

Tracks connection quality:
- **Success rate**: Message delivery success
- **Latency**: Average response time
- **Health score**: 0.0 to 1.0 composite score

## Monitoring and Metrics

### REST API Endpoints

- `GET /api/v1/project-index/websocket/health` - System health status
- `GET /api/v1/project-index/websocket/metrics` - Comprehensive metrics
- `GET /api/v1/project-index/websocket/metrics/prometheus` - Prometheus format
- `GET /api/v1/project-index/websocket/performance` - Performance analysis
- `GET /api/v1/project-index/websocket/connections` - Connection status
- `GET /api/v1/project-index/websocket/events/statistics` - Event statistics
- `GET /api/v1/project-index/websocket/anomalies` - Anomaly detection

### Prometheus Metrics

Key metrics exported for monitoring:

- `project_index_websocket_events_published_total` - Events published
- `project_index_websocket_events_delivered_total` - Events delivered
- `project_index_websocket_active_connections` - Active connections
- `project_index_websocket_event_delivery_duration_seconds` - Delivery latency
- `project_index_websocket_errors_total` - Error count
- `project_index_websocket_health_score` - Overall health

### Health Checks

Automated health monitoring:

1. **Connection Pool Health** - Connection statistics and success rates
2. **Event Publisher Health** - Publishing success and error rates
3. **Redis Connectivity** - Redis connection status
4. **Memory Usage** - System memory consumption
5. **Error Rates** - Overall system error rates

### Alerting

Configurable alerts for:

- High event delivery latency (>500ms warning, >1000ms error)
- Low connection success rate (<90% warning, <80% error)
- High memory usage (>500MB warning, >1GB critical)
- High error rates (>5% warning)
- Large queue sizes (>1000 events error)

## Configuration

### Environment Variables

```bash
# WebSocket configuration
WEBSOCKET_MAX_CONNECTIONS=1000
WEBSOCKET_RATE_LIMIT_TOKENS=100
WEBSOCKET_RATE_LIMIT_REFILL=10.0
WEBSOCKET_COMPRESSION_THRESHOLD=1024

# Event history configuration
EVENT_HISTORY_TTL_HOURS=24
EVENT_HISTORY_MAX_EVENTS=100
EVENT_HISTORY_CLEANUP_INTERVAL=3600

# Performance configuration
WEBSOCKET_BATCH_SIZE=10
WEBSOCKET_BATCH_TIME_MS=100
WEBSOCKET_QUEUE_SIZE=10000

# Monitoring configuration
MONITORING_INTERVAL_SECONDS=30
HEALTH_CHECK_INTERVAL_SECONDS=60
ANOMALY_DETECTION_WINDOW_MINUTES=60
```

### Redis Configuration

Redis is used for:
- Event distribution across server instances
- Event caching and TTL management
- Rate limiting state
- Connection session data

```yaml
redis:
  host: localhost
  port: 6379
  db: 0
  password: null
  ssl: false
  connection_pool_size: 20
```

## Database Schema

The event history is stored in PostgreSQL:

```sql
CREATE TABLE project_index_event_history (
    id UUID PRIMARY KEY,
    project_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB NOT NULL,
    correlation_id VARCHAR(36) NOT NULL,
    persistence_level VARCHAR(20) NOT NULL,
    checksum VARCHAR(64) NOT NULL,
    client_delivered JSONB,
    replay_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for performance
CREATE INDEX idx_event_history_project_id ON project_index_event_history(project_id);
CREATE INDEX idx_event_history_event_type ON project_index_event_history(event_type);
CREATE INDEX idx_event_history_created_at ON project_index_event_history(created_at);
CREATE INDEX idx_event_history_expires_at ON project_index_event_history(expires_at);
```

## Testing

### Unit Tests

Run WebSocket integration tests:

```bash
pytest tests/test_project_index_websocket.py -v
```

### Load Testing

Test with concurrent connections:

```python
import asyncio
import websockets

async def test_concurrent_connections():
    connections = []
    for i in range(100):
        ws = await websockets.connect('ws://localhost:8000/api/v1/project-index/websocket')
        connections.append(ws)
    
    # Test event delivery
    # ... test implementation
```

### Performance Benchmarks

Expected performance targets:

- **Event delivery latency**: <50ms average
- **Connection scalability**: 500+ concurrent connections
- **Event throughput**: 1000+ events/second
- **Memory efficiency**: <10MB per 100 connections
- **Event delivery success**: >99.5%

## Troubleshooting

### Common Issues

1. **High Latency**
   - Check Redis connectivity
   - Monitor memory usage
   - Review event batching settings

2. **Connection Drops**
   - Check rate limiting configuration
   - Monitor connection health scores
   - Review error logs

3. **Missing Events**
   - Check event filtering rules
   - Verify subscription configuration
   - Review Redis pub/sub status

4. **Memory Issues**
   - Adjust event history TTL
   - Review connection pool size
   - Check for memory leaks

### Debug Endpoints

- `GET /api/v1/project-index/websocket/debug/internal-state` - Internal system state
- Log levels: Set `WEBSOCKET_LOG_LEVEL=DEBUG` for detailed logging

### Performance Tuning

1. **Batch Configuration**
   - Increase batch size for higher throughput
   - Decrease batch time for lower latency

2. **Rate Limiting**
   - Adjust token bucket size for burst handling
   - Modify refill rate for sustained load

3. **Compression**
   - Lower threshold for better compression
   - Disable for low-latency requirements

4. **Connection Pool**
   - Increase pool size for more concurrent connections
   - Adjust health check intervals

## Security Considerations

### Authentication

- WebSocket connections require valid authentication
- User permissions control event access
- Rate limiting prevents abuse

### Data Protection

- Event data is filtered based on user permissions
- Sensitive information is excluded from events
- Audit logging for security monitoring

### Resource Protection

- Connection limits prevent DoS attacks
- Rate limiting protects against flooding
- Memory limits prevent resource exhaustion

## Future Enhancements

1. **Event Compression**
   - Advanced compression algorithms
   - Selective field compression

2. **Advanced Filtering**
   - Machine learning-based relevance
   - Predictive event filtering

3. **Scalability**
   - Horizontal scaling support
   - Load balancing across instances

4. **Analytics**
   - Event pattern analysis
   - User behavior insights

5. **Integration**
   - Third-party webhook support
   - External monitoring systems