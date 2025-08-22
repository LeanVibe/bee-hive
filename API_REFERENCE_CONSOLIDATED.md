# 🚀 LeanVibe Agent Hive - Complete API Reference

**Comprehensive REST API documentation for multi-agent orchestration, real-time communication, and system monitoring.**

## 📋 Quick Reference

| Service | Base URL | Purpose |
|---------|----------|---------|
| **Main API** | `http://localhost:18080/api` | Core platform services |
| **Dashboard** | `http://localhost:18080/api/dashboard` | Real-time monitoring |
| **WebSocket** | `ws://localhost:18080/api/dashboard/ws/dashboard` | Live updates |
| **Metrics** | `http://localhost:18080/api/dashboard/metrics` | Prometheus metrics |

> **🔧 Ports**: Uses non-standard ports (18080, 18443) to avoid conflicts

## 🔐 Authentication

### Development Mode (No Auth Required)
```bash
# Most endpoints work without authentication in development
curl http://localhost:18080/health
```

### Production Authentication
```bash
# JWT Token Authentication
curl -H "Authorization: Bearer <jwt_token>" \
     http://localhost:18080/api/agents

# API Key Authentication  
curl -H "X-API-Key: <api_key>" \
     http://localhost:18080/api/agents
```

### Get Access Token
```bash
curl -X POST http://localhost:18080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "your-password"
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer", 
  "expires_in": 3600
}
```

## 🏥 Health & System Status

### System Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-22T10:30:00Z",
  "services": {
    "database": "healthy",
    "redis": "healthy", 
    "orchestrator": "healthy"
  },
  "version": "2.0.0"
}
```

### Comprehensive System Status
```http
GET /api/status
```

**Response:**
```json
{
  "system": {
    "status": "operational",
    "uptime": "24h 15m 32s",
    "agents_active": 12,
    "total_requests": 45789,
    "response_time_avg": "145ms"
  },
  "components": {
    "universal_orchestrator": "healthy",
    "domain_managers": 5,
    "specialized_engines": 8,
    "communication_hub": "active"
  }
}
```

## 🤖 Agent Management API

### List All Agents
```http
GET /api/agents
```

**Query Parameters:**
- `status` (optional): Filter by status (`active`, `inactive`, `deploying`)
- `type` (optional): Filter by agent type
- `limit` (optional): Limit results (default: 50)
- `offset` (optional): Pagination offset

**Response:**
```json
{
  "agents": [
    {
      "id": "agent_12345",
      "type": "backend-developer",
      "status": "active",
      "task": "Implement user authentication API",
      "created_at": "2025-08-22T10:00:00Z",
      "last_activity": "2025-08-22T10:29:45Z",
      "performance": {
        "tasks_completed": 15,
        "success_rate": 94.2,
        "avg_response_time": "2.3s"
      }
    }
  ],
  "total": 12,
  "active": 8,
  "pagination": {
    "limit": 50,
    "offset": 0,
    "has_more": false
  }
}
```

### Get Agent Details
```http
GET /api/agents/{agent_id}
```

**Response:**
```json
{
  "id": "agent_12345",
  "type": "backend-developer",
  "status": "active",
  "configuration": {
    "specialization": "FastAPI development",
    "tools": ["python", "fastapi", "postgresql"],
    "capabilities": ["api_design", "database_modeling", "testing"]
  },
  "current_task": {
    "id": "task_67890", 
    "description": "Implement user authentication API",
    "progress": 65,
    "estimated_completion": "2025-08-22T12:00:00Z"
  },
  "metrics": {
    "total_tasks": 47,
    "completed_tasks": 44,
    "success_rate": 93.6,
    "avg_completion_time": "45m"
  }
}
```

### Deploy New Agent
```http
POST /api/agents/deploy
```

**Request Body:**
```json
{
  "type": "backend-developer",
  "task": "Implement user authentication API", 
  "priority": "high",
  "configuration": {
    "specialization": "FastAPI",
    "timeout_minutes": 120,
    "tools": ["python", "fastapi", "postgresql"]
  }
}
```

**Response:**
```json
{
  "agent_id": "agent_12346",
  "status": "deploying",
  "estimated_ready": "2025-08-22T10:32:00Z",
  "deployment_id": "deploy_98765"
}
```

### Update Agent Task
```http
PUT /api/agents/{agent_id}/task
```

**Request Body:**
```json
{
  "task": "Add rate limiting to authentication API",
  "priority": "medium"
}
```

### Stop Agent
```http
DELETE /api/agents/{agent_id}
```

**Response:**
```json
{
  "message": "Agent stopped successfully",
  "final_status": "completed",
  "tasks_completed": 3,
  "total_runtime": "2h 15m"
}
```

## 📊 Real-Time Dashboard API

### Dashboard Data
```http
GET /api/dashboard/data
```

**Response:**
```json
{
  "system_overview": {
    "agents_active": 12,
    "tasks_in_progress": 8,
    "tasks_completed_today": 47,
    "system_load": 34.5,
    "response_time_avg": "145ms"
  },
  "agent_breakdown": {
    "backend-developer": 4,
    "frontend-developer": 3,
    "qa-engineer": 2,
    "devops-engineer": 2,
    "data-engineer": 1
  },
  "performance_metrics": {
    "requests_per_second": 125,
    "success_rate": 98.2,
    "error_rate": 1.8,
    "avg_task_completion": "42m"
  }
}
```

### Live Metrics Stream
```http
GET /api/dashboard/metrics/stream
```

**Server-Sent Events Response:**
```
data: {"timestamp": "2025-08-22T10:30:00Z", "active_agents": 12, "rps": 125}

data: {"timestamp": "2025-08-22T10:30:05Z", "active_agents": 12, "rps": 130}
```

## 🔌 WebSocket API

### Connection
```javascript
const ws = new WebSocket('ws://localhost:18080/api/dashboard/ws/dashboard');

// Authentication (if required)
ws.send(JSON.stringify({
  type: 'auth',
  token: 'your-jwt-token'
}));
```

### Subscribe to Events
```javascript
// Subscribe to agent status updates
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'agents',
  correlation_id: 'sub_001'
}));

// Subscribe to system metrics
ws.send(JSON.stringify({
  type: 'subscribe', 
  channel: 'system_metrics',
  correlation_id: 'sub_002'
}));
```

### WebSocket Message Types

#### Agent Status Updates
```json
{
  "type": "agent_status",
  "data": {
    "agent_id": "agent_12345",
    "status": "active",
    "task_progress": 75
  },
  "timestamp": "2025-08-22T10:30:00Z",
  "correlation_id": "sub_001"
}
```

#### System Metrics
```json
{
  "type": "system_metrics", 
  "data": {
    "cpu_usage": 45.2,
    "memory_usage": 68.5,
    "active_connections": 1247,
    "requests_per_second": 125
  },
  "timestamp": "2025-08-22T10:30:00Z",
  "correlation_id": "sub_002"
}
```

#### Task Completion
```json
{
  "type": "task_completed",
  "data": {
    "agent_id": "agent_12345",
    "task_id": "task_67890", 
    "result": "success",
    "duration": "42m 15s",
    "output": "Authentication API implemented successfully"
  },
  "timestamp": "2025-08-22T10:30:00Z"
}
```

### Error Handling
All WebSocket errors include:
```json
{
  "type": "error",
  "error": "Subscription limit exceeded",
  "error_code": "SUB_LIMIT",
  "timestamp": "2025-08-22T10:30:00Z",
  "correlation_id": "sub_003"
}
```

## 📋 Task Management API

### List Tasks  
```http
GET /api/tasks
```

**Query Parameters:**
- `status` (optional): Filter by status (`pending`, `in_progress`, `completed`, `failed`)
- `agent_id` (optional): Filter by agent
- `priority` (optional): Filter by priority (`low`, `medium`, `high`)

### Create Task
```http
POST /api/tasks
```

**Request Body:**
```json
{
  "title": "Implement user authentication",
  "description": "Create JWT-based authentication system with rate limiting",
  "agent_type": "backend-developer",
  "priority": "high",
  "estimated_duration": "2h",
  "requirements": [
    "FastAPI framework",
    "JWT authentication", 
    "Rate limiting",
    "Unit tests"
  ]
}
```

### Get Task Status
```http
GET /api/tasks/{task_id}
```

**Response:**
```json
{
  "id": "task_67890",
  "title": "Implement user authentication",
  "status": "in_progress",
  "progress": 65,
  "assigned_agent": "agent_12345",
  "created_at": "2025-08-22T09:00:00Z",
  "estimated_completion": "2025-08-22T12:00:00Z",
  "logs": [
    {
      "timestamp": "2025-08-22T09:15:00Z",
      "message": "Started JWT implementation"
    },
    {
      "timestamp": "2025-08-22T10:00:00Z", 
      "message": "Authentication endpoints created"
    }
  ]
}
```

## 🛠️ Configuration API

### Get System Configuration
```http
GET /api/admin/config
```

### Update Configuration
```http
PUT /api/admin/config
```

**Request Body:**
```json
{
  "orchestrator": {
    "max_agents": 100,
    "load_balancing": true
  },
  "rate_limiting": {
    "requests_per_second": 1000,
    "burst_limit": 2000
  }
}
```

### Hot Reload Configuration
```http
POST /api/admin/config/reload
```

## 📊 Metrics & Monitoring API

### Prometheus Metrics
```http
GET /api/dashboard/metrics
GET /api/dashboard/metrics/websockets
GET /api/dashboard/metrics/agents
```

### System Performance  
```http
GET /api/metrics/performance
```

**Response:**
```json
{
  "response_times": {
    "avg": "145ms",
    "p50": "120ms", 
    "p95": "300ms",
    "p99": "450ms"
  },
  "throughput": {
    "requests_per_second": 125,
    "success_rate": 98.2
  },
  "resources": {
    "cpu_usage": 45.2,
    "memory_usage": "1.2GB",
    "connections_active": 1247
  }
}
```

## 🔍 Search & Query API

### Search Agents
```http
GET /api/search/agents?q={search_term}
```

### Search Tasks
```http  
GET /api/search/tasks?q={search_term}
```

### Advanced Filtering
```http
GET /api/agents?filter={"type": "backend-developer", "status": "active", "performance.success_rate": {"$gt": 90}}
```

## 📡 GitHub Integration API

### Repository Management
```http
GET /api/github/repositories
POST /api/github/repositories/{repo_id}/deploy
GET /api/github/repositories/{repo_id}/status
```

### Webhook Handling
```http
POST /api/github/webhooks
```

## 🚨 Error Responses

### Standard Error Format
```json
{
  "error": "Agent not found",
  "error_code": "AGENT_NOT_FOUND", 
  "message": "Agent with ID 'agent_99999' does not exist",
  "timestamp": "2025-08-22T10:30:00Z",
  "request_id": "req_12345"
}
```

### Common HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized  
- `403` - Forbidden
- `404` - Not Found
- `422` - Validation Error
- `429` - Rate Limited
- `500` - Internal Server Error

## 🔧 Rate Limiting

### Limits
- **API Calls**: 1000 requests/minute per IP
- **WebSocket**: 20 messages/second per connection, burst 40
- **Agent Deployment**: 10 deployments/minute per user

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999  
X-RateLimit-Reset: 1692708000
```

## 📚 Interactive Documentation

### Swagger UI
- **Local**: http://localhost:18080/docs
- **Production**: https://api.leanvibe.com/docs

### ReDoc
- **Local**: http://localhost:18080/redoc
- **Production**: https://api.leanvibe.com/redoc

## 🧪 Testing Examples

### cURL Examples
```bash
# Health check
curl http://localhost:18080/health

# List agents
curl http://localhost:18080/api/agents

# Deploy new agent  
curl -X POST http://localhost:18080/api/agents/deploy \
  -H "Content-Type: application/json" \
  -d '{"type": "backend-developer", "task": "Create API endpoints"}'

# WebSocket connection test
wscat -c ws://localhost:18080/api/dashboard/ws/dashboard
```

### Python Examples
```python
import requests
import websocket

# API client example
response = requests.get('http://localhost:18080/api/agents')
agents = response.json()

# WebSocket client example
def on_message(ws, message):
    print(f"Received: {message}")

ws = websocket.WebSocketApp(
    "ws://localhost:18080/api/dashboard/ws/dashboard",
    on_message=on_message
)
ws.run_forever()
```

### JavaScript Examples
```javascript
// Fetch API example
const agents = await fetch('http://localhost:18080/api/agents')
  .then(response => response.json());

// WebSocket example
const ws = new WebSocket('ws://localhost:18080/api/dashboard/ws/dashboard');
ws.onmessage = (event) => {
  console.log('Received:', JSON.parse(event.data));
};
```

## 🔗 SDKs & Libraries

### Official SDKs
- **Python**: `pip install leanvibe-sdk`
- **JavaScript**: `npm install @leanvibe/sdk`
- **Go**: `go get github.com/leanvibe/go-sdk`

### Community Libraries
- **REST Client**: Available for all major languages
- **WebSocket Client**: Real-time event streaming
- **CLI Tools**: Command-line interface integration

---

## 📞 Support & Resources

### Getting Help
- **Documentation**: [docs/](docs/)
- **Issues**: GitHub Issues
- **Community**: Discord/Slack
- **Enterprise**: support@leanvibe.com

### Additional Resources
- **Architecture Guide**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Developer Guide**: [DEVELOPER_ONBOARDING_30MIN.md](DEVELOPER_ONBOARDING_30MIN.md)
- **Deployment Guide**: [docs/PRODUCTION_DEPLOYMENT_GUIDE.md](docs/PRODUCTION_DEPLOYMENT_GUIDE.md)
- **Troubleshooting**: [docs/runbooks/TROUBLESHOOTING_GUIDE_COMPREHENSIVE.md](docs/runbooks/TROUBLESHOOTING_GUIDE_COMPREHENSIVE.md)

---

*This API reference is automatically tested and validated. All examples are guaranteed to work with the current system version.*