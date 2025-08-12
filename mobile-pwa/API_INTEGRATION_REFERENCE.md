# 📡 Mobile PWA API Integration Reference

**Current API endpoints used by the LeanVibe Agent Hive 2.0 Mobile PWA Dashboard**

> **Note**: This documentation reflects the actual API endpoints currently implemented and used by the mobile PWA, with authentication postponed per project requirements.

---

## 🔧 Base Configuration

### API Base URL
```typescript
// Development
const API_BASE_URL = 'http://localhost:8000'

// Production  
const API_BASE_URL = 'https://api.leanvibe.dev'
```

### Authentication Status
🚫 **Authentication Currently Postponed**: All endpoints accessible without authentication tokens per project requirements.

---

## 🤖 Agent Management API

### Agent System Activation
```http
POST /api/agents/activate
```

**Request Body:**
```json
{
  "team_size": 5,
  "roles": ["product_manager", "architect", "backend_developer", "frontend_developer", "qa_engineer"],
  "auto_start_tasks": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully activated 5 agents",
  "active_agents": {
    "agent-123": {
      "id": "agent-123",
      "role": "product_manager",
      "status": "active",
      "capabilities": ["requirements_analysis", "project_planning"],
      "performance_score": 0.95
    }
  },
  "team_composition": {
    "product_manager": "agent-123",
    "architect": "agent-456"
  }
}
```

### Agent System Status
```http
GET /api/agents/status
```

**Response:**
```json
{
  "active": true,
  "agent_count": 5,
  "spawner_agents": 5,
  "orchestrator_agents": 0,
  "agents": {
    "agent-123": {
      "role": "product_manager",
      "status": "active",
      "current_task": "task-456",
      "performance_metrics": {
        "cpu_usage": 15.2,
        "memory_usage": 45.8,
        "task_completion_rate": 0.92
      }
    }
  },
  "system_ready": true,
  "hybrid_integration": true
}
```

### Spawn Specific Agent
```http
POST /api/agents/spawn/{role}
```

**Path Parameters:**
- `role`: Agent role (`product_manager`, `architect`, `backend_developer`, `frontend_developer`, `qa_engineer`, `devops_engineer`)

**Response:**
```json
{
  "success": true,
  "agent_id": "agent-789",
  "role": "backend_developer",
  "message": "Successfully spawned backend_developer agent"
}
```

### Agent System Deactivation
```http
DELETE /api/agents/deactivate
```

**Response:**
```json
{
  "success": true,
  "message": "Agent system deactivated successfully"
}
```

### Agent Capabilities
```http
GET /api/agents/capabilities
```

**Response:**
```json
{
  "total_agents": 5,
  "roles": {
    "product_manager": {
      "count": 1,
      "capabilities": ["requirements_analysis", "project_planning", "documentation"]
    },
    "backend_developer": {
      "count": 2,
      "capabilities": ["api_development", "database_design", "server_logic"]
    }
  },
  "system_capabilities": ["requirements_analysis", "api_development", "ui_development", "test_creation", "deployment"]
}
```

---

## 📋 Task Management API

### Create Task
```http
POST /api/v1/tasks/
```

**Request Body:**
```json
{
  "title": "Implement user authentication API",
  "description": "Create JWT-based authentication endpoints",
  "priority": "high",
  "agent": "agent-123",
  "estimated_duration": 120,
  "requirements": ["fastapi", "jwt", "database"]
}
```

### Get Tasks
```http
GET /api/v1/tasks/?status=pending&agent=agent-123&limit=10
```

**Query Parameters:**
- `status`: Filter by status (`pending`, `in-progress`, `review`, `done`)
- `agent`: Filter by assigned agent ID
- `priority`: Filter by priority (`low`, `medium`, `high`)
- `limit`: Number of tasks to return (default: 50)
- `offset`: Number of tasks to skip (default: 0)

### Update Task
```http
PUT /api/v1/tasks/{task_id}
```

**Request Body:**
```json
{
  "status": "in-progress",
  "progress": 45,
  "notes": "Working on JWT implementation"
}
```

### Assign Task to Agent
```http
POST /api/v1/tasks/{task_id}/assign/{agent_id}
```

### Start Task
```http
POST /api/v1/tasks/{task_id}/start
```

### Complete Task
```http
POST /api/v1/tasks/{task_id}/complete
```

**Request Body:**
```json
{
  "result": "Authentication API implemented successfully",
  "artifacts": ["auth.py", "models.py", "tests.py"],
  "time_spent": 110
}
```

---

## 📊 System Health API

### System Health Status
```http
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "database": "healthy",
    "redis": "healthy",
    "agents": "healthy",
    "websocket": "healthy"
  },
  "metrics": {
    "cpu_usage": 25.4,
    "memory_usage": 68.2,
    "active_connections": 12,
    "uptime": 3600
  }
}
```

### System Metrics
```http
GET /api/v1/metrics/system
```

**Response:**
```json
{
  "cpu_usage": 25.4,
  "memory_usage": 68.2,
  "disk_usage": 45.1,
  "network_io": {
    "bytes_sent": 1024000,
    "bytes_received": 2048000
  },
  "agent_performance": {
    "average_task_completion_time": 85.5,
    "success_rate": 0.94,
    "active_agents": 5
  }
}
```

---

## 📨 Event Management API

### Get Events
```http
GET /api/v1/events?severity=warning&limit=20
```

**Query Parameters:**
- `severity`: Filter by severity (`info`, `warning`, `error`, `critical`)
- `type`: Filter by event type
- `limit`: Number of events to return

**Response:**
```json
{
  "events": [
    {
      "id": "event-123",
      "type": "agent-status-change",
      "severity": "info",
      "title": "Agent activated",
      "description": "Backend developer agent activated successfully",
      "timestamp": "2024-01-15T10:30:00Z",
      "agent_id": "agent-456"
    }
  ],
  "total_count": 145,
  "has_more": true
}
```

---

## 🔌 WebSocket Integration

### WebSocket Connection
```javascript
// Connect to WebSocket for real-time updates
const ws = new WebSocket('ws://localhost:8000/api/dashboard/ws/dashboard');

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  
  switch (update.type) {
    case 'agent-status-change':
      updateAgentStatus(update.data);
      break;
    case 'task-updated':
      updateTaskInKanban(update.data);
      break;
    case 'system-metrics':
      updateSystemHealth(update.data);
      break;
  }
};
```

### WebSocket Message Types
- `agent-status-change`: Agent status updates
- `task-updated`: Task status or assignment changes
- `task-created`: New task created
- `task-deleted`: Task removed
- `system-metrics`: Real-time system performance data
- `new-event`: System events and notifications

---

## 🔄 Real-time Updates

### Polling Fallback
When WebSocket is unavailable, the PWA falls back to polling:

```typescript
// Polling intervals
const POLLING_INTERVALS = {
  agents: 5000,      // 5 seconds
  tasks: 3000,       // 3 seconds  
  health: 10000,     // 10 seconds
  events: 15000      // 15 seconds
};
```

### Update Patterns
- **Optimistic Updates**: UI updates immediately, reverts on API failure
- **Real-time Sync**: WebSocket updates override local state
- **Conflict Resolution**: Last-write-wins with user notification
- **Offline Queue**: Operations queued when offline, synced when online

---

## 🚨 Error Handling

### HTTP Status Codes
- `200`: Success
- `201`: Created successfully
- `400`: Bad request (validation error)
- `404`: Resource not found
- `500`: Internal server error
- `503`: Service unavailable

### Error Response Format
```json
{
  "error": true,
  "message": "Agent activation failed",
  "details": "Insufficient system resources",
  "code": "AGENT_ACTIVATION_FAILED",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Retry Logic
- **Network Errors**: Exponential backoff (1s, 2s, 4s, 8s)
- **Server Errors (5xx)**: Retry up to 3 times
- **Client Errors (4xx)**: No retry, show user error
- **WebSocket Reconnection**: Automatic with exponential backoff

---

## 📱 Mobile PWA Integration Examples

### Agent Activation Service
```typescript
import { BaseService } from './base-service';

export class AgentService extends BaseService {
  async activateAgentTeam(options: AgentActivationOptions) {
    return this.post('/api/agents/activate', {
      team_size: options.teamSize || 5,
      auto_start_tasks: options.autoStartTasks || true
    });
  }

  async getAgentSystemStatus() {
    return this.get('/api/agents/status');
  }

  async spawnAgent(role: AgentRole) {
    return this.post(`/api/agents/spawn/${role}`);
  }
}
```

### Task Service Integration
```typescript
export class TaskService extends BaseService {
  async createTask(taskData: CreateTaskRequest) {
    return this.post('/api/v1/tasks/', taskData);
  }

  async getTasks(filters: TaskFilters) {
    const params = new URLSearchParams(filters as any);
    return this.get(`/api/v1/tasks/?${params}`);
  }

  async updateTaskStatus(taskId: string, status: TaskStatus) {
    return this.put(`/api/v1/tasks/${taskId}`, { status });
  }
}
```

---

## 🔗 Related Documentation

- **[Mobile PWA README](README.md)**: Complete PWA documentation
- **[Testing Guide](README-TESTING.md)**: API integration testing
- **[Main API Reference](../docs/reference/API_REFERENCE_COMPREHENSIVE.md)**: Full backend API documentation

---

**📡 This reference covers all API integrations currently implemented in the Mobile PWA Dashboard as of the latest update.**