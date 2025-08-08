# Dashboard API Documentation

**LeanVibe Agent Hive - Multi-Agent Coordination Monitoring APIs**

## Overview

This documentation describes the comprehensive dashboard API infrastructure designed to monitor and control the multi-agent coordination system. These APIs address the critical 20% coordination success rate issue by providing real-time monitoring, failure analysis, and recovery controls.

## Architecture

The dashboard API infrastructure consists of four main components:

1. **Agent Status & Health APIs** - Monitor individual agent performance and health
2. **Coordination Monitoring APIs** - Track coordination success rates and failure patterns
3. **Task Distribution APIs** - Manage task queues and manual reassignment
4. **Recovery & Control APIs** - Emergency overrides and system recovery
5. **WebSocket APIs** - Real-time updates for dashboard clients
6. **Prometheus Metrics** - External monitoring integration

## Base Configuration

- **Base URL**: `http://localhost:8000`
- **WebSocket URL**: `ws://localhost:8000`
- **All endpoints return JSON** (except Prometheus metrics which return plain text)
- **Response times**: Target <100ms for dashboard endpoints
- **Authentication**: Currently open (production deployment should add auth)

---

## 1. Agent Status & Health APIs

### GET /api/dashboard/agents/status

Get real-time status of all agents with health indicators.

**Parameters:**
- `include_inactive` (query, bool): Include inactive agents (default: false)

**Response:**
```json
{
  "agents": [
    {
      "id": "uuid",
      "name": "Agent Name",
      "status": "active|inactive|busy|error|maintenance",
      "health_score": 85.5,
      "response_time_ms": 150.0,
      "task_success_rate": 92.3,
      "current_task": "Task title or null",
      "error_count": 2,
      "last_heartbeat": "2025-08-07T10:30:00Z"
    }
  ],
  "total_agents": 3,
  "active_agents": 2,
  "health_summary": {
    "healthy": 2,
    "degraded": 1,
    "unhealthy": 0
  },
  "last_updated": "2025-08-07T10:30:00Z"
}
```

### GET /api/dashboard/agents/{agent_id}/metrics

Get detailed performance metrics for a specific agent.

**Parameters:**
- `agent_id` (path, string): Agent UUID
- `time_range_hours` (query, int): Time range in hours (1-168, default: 24)

**Response:**
```json
{
  "agent": { /* agent details */ },
  "metrics": {
    "total_tasks": 45,
    "completed_tasks": 38,
    "failed_tasks": 5,
    "success_rate": 84.4,
    "average_completion_time_minutes": 12.5,
    "error_rate": 11.1
  },
  "recent_tasks": [ /* last 10 tasks */ ],
  "performance_trends": {
    "capacity_utilization": 0.65
  }
}
```

### POST /api/dashboard/agents/{agent_id}/restart

Restart a specific agent with optional force parameter.

**Parameters:**
- `agent_id` (path, string): Agent UUID
- `force` (query, bool): Force restart even with active tasks (default: false)
- `reason` (query, string): Reason for restart

**Response:**
```json
{
  "success": true,
  "agent_id": "uuid",
  "agent_name": "Agent Name",
  "status": "restart_initiated",
  "active_tasks_reassigned": 2,
  "timestamp": "2025-08-07T10:30:00Z"
}
```

### GET /api/dashboard/agents/heartbeat

Get last heartbeat timestamps for all agents.

**Response:**
```json
{
  "agents": [
    {
      "agent_id": "uuid",
      "name": "Agent Name",
      "status": "active",
      "last_heartbeat": "2025-08-07T10:29:45Z",
      "seconds_since_heartbeat": 15,
      "is_stale": false,
      "health_status": "healthy"
    }
  ],
  "summary": {
    "total_agents": 3,
    "healthy_agents": 2,
    "stale_heartbeats": 1,
    "overall_health": "healthy"
  }
}
```

---

## 2. Coordination Monitoring APIs

### GET /api/dashboard/coordination/success-rate

**CRITICAL ENDPOINT** - Monitor the coordination success rate issue.

**Parameters:**
- `time_range_hours` (query, int): Time range for analysis (1-168, default: 24)

**Response:**
```json
{
  "current_metrics": {
    "success_rate": 75.5,
    "total_tasks": 100,
    "successful_tasks": 75,
    "failed_tasks": 20,
    "pending_tasks": 5,
    "trend_direction": "improving|declining|stable"
  },
  "hourly_breakdown": [
    {
      "hour": "10:00",
      "total_tasks": 8,
      "successful_tasks": 6,
      "success_rate": 75.0
    }
  ],
  "failure_patterns": {
    "Redis connection timeout": 12,
    "Serialization error": 5
  },
  "critical_alerts": [
    {
      "level": "critical",
      "message": "Coordination success rate is 20.1%",
      "threshold": "Expected >90%",
      "impact": "Core autonomous development functionality affected"
    }
  ],
  "recommendations": [
    "Check Redis connectivity and message serialization",
    "Review agent assignment logic"
  ]
}
```

### GET /api/dashboard/coordination/failures

Get detailed failure analysis with breakdown by type and severity.

**Parameters:**
- `limit` (query, int): Number of failures to return (1-200, default: 50)
- `include_resolved` (query, bool): Include resolved failures (default: false)

**Response:**
```json
{
  "total_failures": 25,
  "analysis": {
    "by_type": {"feature_development": 15, "bug_fix": 8},
    "by_agent": {"agent-001": 10, "agent-002": 15},
    "by_error_pattern": {"timeout": 12, "serialization": 8}
  },
  "detailed_failures": [
    {
      "task_id": "uuid",
      "title": "Task title",
      "assigned_agent_id": "uuid",
      "error_message": "Redis connection timeout",
      "retry_count": 2,
      "can_retry": true,
      "minutes_since_failure": 15,
      "recovery_actions": ["retry_task", "reassign_agent"]
    }
  ],
  "recovery_recommendations": [
    {
      "action": "Restart affected agents",
      "applicable_to": 3,
      "urgency": "high"
    }
  ]
}
```

### POST /api/dashboard/coordination/reset

**EMERGENCY CONTROL** - Reset coordination system with different severity levels.

**Parameters:**
- `reset_type` (query, string): "soft|hard|full"
- `confirm` (query, bool): Confirmation required for reset

**Response:**
```json
{
  "success": true,
  "reset_type": "soft",
  "actions_performed": [
    "Reset 5 stuck tasks to pending",
    "Broadcast reset commands via Redis"
  ],
  "affected_components": {
    "tasks_reset": 5,
    "agents_notified": 3
  },
  "next_steps": [
    "Monitor agent health endpoints for recovery",
    "Check coordination success rate in 5 minutes"
  ]
}
```

### GET /api/dashboard/coordination/diagnostics

Get comprehensive diagnostic data for coordination system health.

**Response:**
```json
{
  "diagnostics": {
    "database": {
      "total_agents": 3,
      "active_agents": 2,
      "orphaned_tasks": 0
    },
    "redis": {
      "connectivity": "healthy",
      "active_streams": 5,
      "memory_usage": "15.2MB"
    },
    "coordination": {
      "stale_unassigned_tasks": 2,
      "long_running_tasks": 1,
      "agents_with_stale_heartbeats": 0
    }
  },
  "overall_health": {
    "score": 85,
    "status": "healthy",
    "issues": []
  },
  "recommendations": [
    "System is operating normally"
  ]
}
```

---

## 3. Task Distribution APIs

### GET /api/dashboard/tasks/queue

Get current task queue status with filtering and distribution metrics.

**Parameters:**
- `status_filter` (query, string): "pending|assigned|in_progress|blocked"
- `priority_filter` (query, string): "low|medium|high|critical" 
- `agent_filter` (query, string): Agent ID filter
- `limit` (query, int): Max tasks to return (1-500, default: 100)

**Response:**
```json
{
  "queue_statistics": {
    "by_status": {"pending": 10, "in_progress": 5},
    "by_priority": {"high": 8, "medium": 12},
    "agent_assignments": {
      "agent-001": {"name": "Dev Agent", "active_tasks": 3}
    }
  },
  "distribution_metrics": {
    "total_active_tasks": 15,
    "distribution_efficiency": 66.7,
    "average_wait_time_minutes": 8.5,
    "unassigned_tasks": 5
  },
  "tasks": [
    {
      "id": "uuid",
      "title": "Task title",
      "status": "pending",
      "priority": "high",
      "wait_time_minutes": 12.5,
      "urgency_score": 0.85
    }
  ]
}
```

### POST /api/dashboard/tasks/{task_id}/reassign

Manual task reassignment with automatic agent selection option.

**Parameters:**
- `task_id` (path, string): Task UUID
- `new_agent_id` (query, string): Specific agent ID (optional)
- `auto_select` (query, bool): Auto-select best agent (default: true)
- `priority_boost` (query, bool): Increase task priority (default: false)
- `reason` (query, string): Reassignment reason

**Response:**
```json
{
  "success": true,
  "reassignment": {
    "task_id": "uuid",
    "original_agent_id": "uuid",
    "new_agent_id": "uuid",
    "assignment_method": "auto_select_best_available",
    "priority_boost": {"from": "medium", "to": "high"}
  },
  "updated_task": { /* task object */ }
}
```

### GET /api/dashboard/tasks/distribution

Get task distribution data optimized for dashboard visualization.

**Parameters:**
- `time_range_hours` (query, int): Time range for analysis (1-168, default: 24)

**Response:**
```json
{
  "agent_distribution": [
    {
      "agent_id": "uuid",
      "agent_name": "Dev Agent",
      "task_count": 15,
      "percentage": 42.3
    }
  ],
  "type_distribution": [
    {"task_type": "feature_development", "count": 20, "percentage": 55.6}
  ],
  "hourly_timeline": [
    {
      "hour": "2025-08-07 10:00",
      "total_tasks": 8,
      "completed_tasks": 6,
      "success_rate": 75.0
    }
  ],
  "distribution_metrics": {
    "most_utilized_agent": "Dev Agent",
    "workload_variance": 15.2,
    "distribution_efficiency": 88.5
  }
}
```

### POST /api/dashboard/tasks/{task_id}/retry

Manual retry controls for failed tasks with enhanced options.

**Parameters:**
- `task_id` (path, string): Task UUID
- `reset_retry_count` (query, bool): Reset retry count to 0 (default: false)
- `increase_priority` (query, bool): Increase task priority (default: false)
- `new_agent_assignment` (query, bool): Allow new agent assignment (default: true)
- `reason` (query, string): Retry reason

**Response:**
```json
{
  "success": true,
  "retry_info": {
    "task_id": "uuid",
    "original_status": "failed",
    "new_retry_count": 3,
    "priority_increased": {"from": "medium", "to": "high"},
    "agent_reassignment": {"allow_new_assignment": true}
  },
  "next_steps": [
    "Task reset to pending status",
    "Will be picked up by orchestrator for reassignment"
  ]
}
```

---

## 4. Recovery & Control APIs

### POST /api/dashboard/system/emergency-override

**EMERGENCY CONTROL** - Emergency system override controls for critical failures.

**Parameters:**
- `action` (query, string): "stop_all_tasks|restart_all_agents|clear_task_queue|force_agent_restart|system_maintenance"
- `target_agent_id` (query, string): For targeted actions (optional)
- `confirm_emergency` (query, bool): Emergency confirmation required
- `reason` (query, string): Reason for emergency action

**Response:**
```json
{
  "success": true,
  "emergency_action": "restart_all_agents",
  "actions_performed": [
    "Set 3 agents to maintenance mode",
    "Broadcast restart commands via Redis"
  ],
  "affected_components": {
    "agents_restarted": 3,
    "tasks_reassigned": 8
  },
  "warning": "System may require manual intervention to resume normal operations"
}
```

### GET /api/dashboard/system/health

Comprehensive system health check with detailed component analysis.

**Parameters:**
- `include_historical` (query, bool): Include historical trends (default: false)

**Response:**
```json
{
  "overall_health": {
    "score": 92,
    "status": "healthy",
    "component_scores": {
      "database": 100,
      "redis": 100,
      "agents": 90,
      "tasks": 85
    }
  },
  "components": {
    "database": {
      "total_agents": 3,
      "total_tasks": 150,
      "tasks_last_hour": 12
    },
    "redis": {
      "connectivity": "healthy",
      "active_streams": 5,
      "memory_usage": "15.2MB"
    },
    "agents": {
      "active_agents": 2,
      "stale_heartbeats": 0,
      "status": "healthy"
    },
    "tasks": {
      "pending_tasks": 5,
      "long_running_tasks": 1,
      "recent_failures": 2,
      "status": "healthy"
    }
  },
  "alerts": [
    {
      "level": "warning",
      "component": "tasks",
      "message": "1 task has been running for >2 hours",
      "action": "Review task progress and consider intervention"
    }
  ]
}
```

### POST /api/dashboard/recovery/auto-heal

Trigger automatic recovery procedures for coordination system issues.

**Parameters:**
- `recovery_type` (query, string): "smart|aggressive|conservative"
- `dry_run` (query, bool): Perform dry run without changes (default: true)

**Response:**
```json
{
  "actions": [
    {
      "type": "reset_stuck_tasks",
      "description": "Reset 3 stuck tasks to pending",
      "impact": "medium",
      "execute": false
    }
  ],
  "analysis": {
    "stuck_tasks": 3,
    "stale_agents": 1,
    "failure_rate": 15.5
  },
  "success_likelihood": 85,
  "estimated_recovery_time": "6 minutes",
  "dry_run": true
}
```

### GET /api/dashboard/logs/coordination

Get coordination error logs with filtering and analysis.

**Parameters:**
- `hours` (query, int): Hours of logs to retrieve (1-168, default: 24)
- `error_level` (query, string): "all|error|critical|warning"
- `limit` (query, int): Max log entries (1-500, default: 100)

**Response:**
```json
{
  "summary": {
    "total_errors": 25,
    "critical_errors": 3,
    "error_rate_trend": "stable",
    "most_common_pattern": "Redis connection timeout"
  },
  "log_entries": [
    {
      "timestamp": "2025-08-07T10:25:30Z",
      "level": "error",
      "component": "coordination",
      "task_id": "uuid",
      "error_message": "Redis connection timeout",
      "context": {
        "task_type": "feature_development",
        "retry_count": 2
      }
    }
  ],
  "error_patterns": [
    {"pattern": "Redis connection timeout", "count": 12},
    {"pattern": "Serialization error", "count": 8}
  ],
  "recommendations": [
    {
      "issue": "Redis connectivity issues detected",
      "action": "Check Redis server health and network connectivity",
      "priority": "high"
    }
  ]
}
```

---

## 5. WebSocket APIs

Real-time endpoints for live dashboard updates.

### /api/dashboard/ws/agents

**WebSocket endpoint for real-time agent status updates.**

**Connection:**
```
ws://localhost:8000/api/dashboard/ws/agents?connection_id=optional_id
```

**Messages:**
- `ping` - Client heartbeat
- `subscribe` - Add subscriptions
- `unsubscribe` - Remove subscriptions
- `request_data` - Request specific data

**Server Events:**
- `agent_update` - Agent status changes
- `agent_event` - Agent-specific events

### /api/dashboard/ws/coordination

**WebSocket endpoint for real-time coordination monitoring.**

**Connection:**
```
ws://localhost:8000/api/dashboard/ws/coordination?connection_id=optional_id
```

**Server Events:**
- `coordination_update` - Success rate changes
- `critical_alert` - Critical coordination failures

### /api/dashboard/ws/tasks

**WebSocket endpoint for real-time task distribution monitoring.**

**Connection:**
```
ws://localhost:8000/api/dashboard/ws/tasks?connection_id=optional_id
```

**Server Events:**
- `task_update` - Queue status changes
- `task_assignment` - Task assignment events

### /api/dashboard/ws/system

**WebSocket endpoint for real-time system health monitoring.**

**Connection:**
```
ws://localhost:8000/api/dashboard/ws/system?connection_id=optional_id
```

**Server Events:**
- `system_update` - System health changes
- `critical_alert` - System alerts

### /api/dashboard/ws/dashboard

**Comprehensive WebSocket endpoint for full dashboard functionality.**

**Connection:**
```
ws://localhost:8000/api/dashboard/ws/dashboard?connection_id=optional_id&subscriptions=agents,coordination,tasks,system
```

**Subscriptions:**
- `agents` - Agent status updates
- `coordination` - Coordination monitoring
- `tasks` - Task distribution updates
- `system` - System health updates
- `alerts` - Critical alerts

---

## 6. Prometheus Metrics APIs

### GET /api/dashboard/metrics

**Prometheus-compatible metrics endpoint.**

**Response Format:** Plain text (Prometheus format)

**Sample Metrics:**
```
# HELP leanvibe_coordination_success_rate Coordination success rate percentage
# TYPE leanvibe_coordination_success_rate gauge
leanvibe_coordination_success_rate 75.5

# HELP leanvibe_agents_total Total number of agents by status
# TYPE leanvibe_agents_total gauge
leanvibe_agents_total{status="active"} 2
leanvibe_agents_total{status="inactive"} 1

# HELP leanvibe_tasks_total Total number of tasks by status
# TYPE leanvibe_tasks_total gauge
leanvibe_tasks_total{status="pending"} 5
leanvibe_tasks_total{status="completed"} 45
```

### GET /api/dashboard/metrics/coordination

**Coordination-focused metrics subset.**

### GET /api/dashboard/metrics/agents

**Agent-focused metrics subset.**

### GET /api/dashboard/metrics/system

**System health metrics subset.**

---

## Error Handling

All APIs use consistent error response format:

```json
{
  "error": "Error description",
  "request_id": "optional_request_id",
  "timestamp": "2025-08-07T10:30:00Z"
}
```

**Common HTTP Status Codes:**
- `200` - Success
- `400` - Bad request (invalid parameters)
- `404` - Resource not found
- `500` - Internal server error

---

## Performance Targets

- **Response Times**: <100ms for dashboard endpoints
- **WebSocket Updates**: <50ms latency
- **Prometheus Metrics**: <500ms generation time
- **Real-time Data**: 1-5 second refresh rates

---

## Security Considerations

**Current Status:** APIs are open for development
**Production Requirements:**
- Add authentication middleware
- Implement rate limiting
- Add input validation and sanitization
- Enable HTTPS only
- Add audit logging for control operations

---

## Monitoring Integration

### Grafana Dashboards

Use Prometheus metrics endpoints to create Grafana dashboards:

1. **Coordination Success Rate**: Monitor the critical 20% success rate issue
2. **Agent Health**: Track agent performance and heartbeats
3. **Task Distribution**: Monitor queue length and assignment efficiency
4. **System Health**: Overall system status and alerts

### Alerting Rules

**Critical Alerts:**
```
leanvibe_coordination_success_rate < 30
leanvibe_agents_stale_heartbeats > 2
leanvibe_task_queue_length > 100
leanvibe_database_healthy == 0
```

**Warning Alerts:**
```
leanvibe_coordination_success_rate < 70
leanvibe_tasks_long_running > 5
leanvibe_redis_healthy == 0
```

---

## Testing

**Run comprehensive tests:**
```bash
python test_dashboard_apis.py
```

**Test specific endpoint:**
```bash
python test_dashboard_apis.py /api/dashboard/agents/status
```

**Test WebSocket connectivity:**
```bash
# Comprehensive tests include WebSocket validation
python test_dashboard_apis.py
```

---

## Troubleshooting

### Common Issues

1. **502/503 Errors**: Server not running
   - Start server: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

2. **WebSocket Connection Failed**: 
   - Check WebSocket endpoint URLs
   - Verify connection_id parameter format

3. **Empty Metrics**: 
   - Check database connectivity
   - Verify agent data exists

4. **High Response Times**:
   - Check database performance
   - Review Redis connectivity
   - Monitor system resources

### Debug Endpoints

- `GET /health` - Basic system health
- `GET /debug-agents` - Agent debugging info
- `GET /api/dashboard/websocket/health` - WebSocket system health
- `GET /api/dashboard/metrics/health` - Metrics system health

---

## Support

For issues with the dashboard API infrastructure:

1. Check system logs for error details
2. Validate database and Redis connectivity  
3. Test individual endpoints with the test suite
4. Review WebSocket connection logs
5. Monitor Prometheus metrics for system health

The dashboard APIs are designed to provide comprehensive visibility into the coordination system's 20% success rate issue and enable effective monitoring and recovery operations.