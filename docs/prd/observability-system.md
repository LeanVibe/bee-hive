# PRD: Real-Time Observability & Monitoring System

**Priority**: Must-Have (Phase 1) | **Estimated Effort**: 3-4 weeks | **Technical Complexity**: Medium–High

## Executive Summary

The Observability & Monitoring System delivers full-stack visibility into LeanVibe Agent Hive with hook-based event interception, structured logging, distributed tracing, and WebSocket streaming dashboards. Inspired by Claude Code hooks, SuperClaude, and industry best practices, it allows deterministic lifecycle event tracking (PreToolUse, PostToolUse, Notification, Stop, SubAgentStop) and granular performance metrics for debugging, optimization, and real-time analysis.

## Problem Statement

Agents operate autonomously at scale, making it difficult to understand failures, performance bottlenecks, or security issues. Current gaps include:
- **No unified event stream** across agents and tools
- **Limited insight** into performance bottlenecks and failures
- **Manual log inspection** with no dashboards or alerts
- **Lack of structured audit trails** for lifecycle hooks
- **Ad-hoc logging** lacking real-time visibility

Developers and operators need a robust observability stack to:
- **Trace agent actions** across tools and sub-agents
- **Diagnose errors** quickly with context-rich logs
- **Measure performance** (latency, token usage, retries)
- **Audit security** via deterministic event hooks

## Success Metrics

| KPI | Target |
|---|---|
| **Event capture coverage** | 100% lifecycle hooks instrumented |
| **Hook coverage** | 100% lifecycle events captured |
| **Event latency (P95)** | <150 ms from emit to storage |
| **Dashboard latency** | <2 seconds lag from event to visualization |
| **Dashboard refresh rate** | <1 s |
| **Mean Time To Detect (MTTD)** | <1 minute for critical failures |
| **Error detection MTTR** | <5 min |
| **Alert false-positive rate** | <5% |
| **Performance overhead** | <3% CPU per agent |
| **Retention compliance** | 30-day searchable logs with <1% data loss |

## User Stories

### Primary User Stories
1. **As a developer**, I can view real-time agent tool usage and outputs on a dashboard.
2. **As a security auditor**, I can receive alerts when a `PreToolUse` hook blocks a dangerous command.
3. **As an SRE**, I can query historical chat transcripts to diagnose anomalies.
4. **As a PM agent**, I can filter events by session and color-coded agent role.

### Extended User Stories & Acceptance Tests

#### Story 1: Real-Time Event Streaming
**As** an operator **I want** to see live agent events **So that** I can detect anomalies quickly.
```python
# Test: Event published appears on stream within 1s
start = time.time()
publish_event(event)
msg = redis.xread(streams={"agent_events": 0}, block=2000)
assert msg is not None and time.time() - start < 1.0
```

#### Story 2: Grafana Timeline Visualisation
**As** a developer **I want** colour-coded timelines per session **So that** I can trace tool calls and durations.
```python
def test_timeline_dashboard():
    dash = grafana.get_dashboard("session-timeline")
    assert dash is not None
    assert dash.variables["session_id"].query == "agent_events"
```

#### Story 3: Alert on Error Spike
```python
# Simulate 5 failed PostToolUse events in 30s
for _ in range(5):
    publish_event({"event_type": "PostToolUse", "payload": {"success": False}})
# Grafana alert should trigger
assert alertmanager.get_active_alerts(name="ToolFailureSpike")
```

## Technical Requirements

### Architecture
- **Hook Scripts**: Bash/Python scripts triggered by Claude Code events
- **Hook Interceptor**: Captures PreToolUse, PostToolUse, Notification, Stop, SubagentStop events
- **Event Collector**: Bun/TypeScript HTTP + WebSocket server (`/events`, `/stream`)
- **Event Stream Broker**: Redis Streams with consumer groups; supports at-least-once delivery
- **Event Processor**: Async FastAPI workers persisting events to PostgreSQL (JSONB) and emitting Prometheus metrics
- **Database**: PostgreSQL (events) + pgvector (embeddings for log search)
- **Message Broker**: Redis Streams for decoupled ingestion
- **Dashboard**: Vue 3 + Vite SPA
- **Metrics & Dashboards**: Prometheus + Grafana with colour-coded timelines, session filters, and drill-downs
- **Alerting Engine**: Grafana Alerting or Alertmanager rules for latency, error-rate, and throughput thresholds
- **Transcript Storage**: Optional S3/MinIO bucket for chat logs with indices referencing DB rows

### Event Schema
```json
{
  "id": "uuid",
  "session_id": "uuid",
  "agent_id": "agent-123",
  "event_type": "PreToolUse|PostToolUse|Notification|Stop|SubagentStop",
  "tool": "Write",
  "status": "success|blocked|error",
  "payload": {},
  "timestamp": 1699999999,
  "latency_ms": 150
}
```

### Hook Implementation Example (PreToolUse)
```bash
#!/usr/bin/env bash
set -euo pipefail
TOOL="$CLAUDE_TOOL_NAME"
if [[ "$TOOL" == "Bash" && "$CLAUDE_TOOL_INPUT" =~ "rm -rf /" ]]; then
  echo "Dangerous command blocked" >&2
  exit 2  # Block tool
fi

# Send event
jq -n --arg tool "$TOOL" --arg event "PreToolUse" '{tool:$tool,event_type:$event,timestamp:(now|floor)}' | \
  curl -X POST -H 'Content-Type: application/json' -d @- http://observability-server:7070/events
```

### API Specifications
```
POST /observability/event
{
  "event_type": "PreToolUse|PostToolUse|Notification|Stop|SubagentStop",
  "agent_id": "uuid",
  "session_id": "uuid",
  "payload": { ... },
  "timestamp": "iso8601"
}
Response: {"status": "queued"}

GET /observability/events?session_id={uuid}&from={ts}&to={ts}&type=PostToolUse
Response: {"events": [...]}  # Paginated

POST /events — ingest event (bulk supported)
GET /events/recent?session=uuid — recent events
WS /stream?filter=session:uuid — real-time stream

GET /observability/metrics
Response: Prometheus text format
```

### Database Schema
```sql
CREATE TABLE agent_events (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    agent_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL,
    latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE INDEX idx_events_session ON agent_events(session_id);
CREATE INDEX idx_events_type_time ON agent_events(event_type, created_at);

CREATE TABLE chat_transcripts (
    id UUID PRIMARY KEY,
    session_id UUID,
    agent_id UUID,
    s3_key VARCHAR(500),
    size_bytes BIGINT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Dashboard Features
- Live system timeline with session color coding
- Token usage heatmap per agent
- Error overlay with stack traces and payload excerpt
- Histogram of tool latencies
- Filter chips: event type, agent role, timeframe
- Colour-coded timelines per session for tool call tracing

## Implementation Strategy

### Phased Approach
| Phase | Duration | Deliverables |
|---|---|---|
| **Week 1** | 1 wk | Hook instrumentation & Redis Streams broker, PreToolUse & PostToolUse hooks, event collector |
| **Week 2** | 1 wk | Event processor, PostgreSQL persistence, Prometheus metrics, bulk API |
| **Week 3** | 1 wk | Grafana dashboards, alert rules, transcript storage, live timeline, filtering, color coding |
| **Week 4** | 1 wk | Load testing, data retention policies, Prometheus exporter, Grafana panels, Alertmanager rules, embeddings search |

## Testing Strategy
- **Unit**: Hook exit codes, schema validation, DB inserts
- **Integration**: Simulated agent run with 1k events/sec
- **Chaos**: Kill collector pod; ensure Redis buffer drains on restart
- **Load**: WebSocket fan-out to 100 dashboard clients

## Security Considerations
- **TLS** for Redis & API endpoints
- **Signed event payloads** to prevent tampering
- **RBAC** on Grafana & API
- **PII redaction** hook stage, RBAC on dashboard

## Risks & Mitigations
| Risk | Mitigation |
|---|---|
| **High event volume overload** | Redis buffer + batch inserts, Redis sharding, back-pressure |
| **High ingest rate** | Use Redis sharding, back-pressure and batch inserts |
| **Dashboard overload** | Pre-aggregate metrics; limit panel queries |
| **Sensitive data exposure** | PII redaction hook stage, RBAC on dashboard |
| **Hook misconfiguration blocking agents** | Canary hooks in non-blocking mode first |

## Dependencies
- Agent Orchestrator Core
- PostgreSQL with pgvector
- Redis ≥7.2
- Prometheus stack

## Acceptance Criteria
- All lifecycle events visible in dashboard within 1-2 seconds
- Prometheus metrics expose event_rate, error_rate
- Hooks block dangerous commands in e2e tests
- Performance overhead <3% CPU on benchmark
- Event flow from hook → DB verified end-to-end
- Grafana dashboard with token usage and error panels

## Definition of Done
1. Event flow from hook → DB verified end-to-end
2. Grafana dashboard with token usage and error panels
3. CI tests (>90% coverage) passing
4. Security review completed with no critical findings
5. Production-grade observability pipeline providing deep visibility into Agent Hive behaviour and performance

---
*This PRD consolidates requirements for a comprehensive observability and monitoring system that provides real-time visibility, performance metrics, and audit capabilities for the LeanVibe Agent Hive ecosystem.*