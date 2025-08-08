# LeanVibe Agent Hive API Reference

## Overview

The LeanVibe Agent Hive 2.0 provides a comprehensive REST API for managing multi-agent orchestration, real-time communication, and system monitoring. This documentation provides detailed information about all available endpoints with interactive examples.

## Base URL

```
Production: https://api.leanvibe.dev/api/v1
Development: http://localhost:8000/api/v1
```

## Authentication

All API endpoints require JWT authentication. Include the token in the Authorization header:

```bash
Authorization: Bearer <your-jwt-token>
```

### Getting an Access Token

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
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

## Core API Endpoints

### Agents API

#### List All Agents
```http
GET /agents
```

**Query Parameters:**
- `limit` (optional): Number of agents to return (default: 50)
- `offset` (optional): Number of agents to skip (default: 0)
- `status` (optional): Filter by agent status (`active`, `inactive`, `busy`)
- `role` (optional): Filter by agent role (`developer`, `tester`, `reviewer`)

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/agents?limit=10&status=active" \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "agents": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "name": "backend-specialist",
      "role": "developer",
      "status": "active",
      "capabilities": ["python", "fastapi", "postgresql"],
      "max_concurrent_tasks": 3,
      "current_tasks": 1,
      "performance_score": 0.92,
      "created_at": "2024-01-01T00:00:00Z",
      "last_activity": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 25,
  "page": 1,
  "pages": 3
}
```

#### Create New Agent
```http
POST /agents
```

**Request Body:**
```json
{
  "name": "frontend-expert",
  "role": "developer",
  "capabilities": ["react", "typescript", "css", "testing"],
  "max_concurrent_tasks": 2,
  "specializations": ["ui_design", "responsive_web"],
  "experience_level": "expert"
}
```

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "frontend-expert",
    "role": "developer",
    "capabilities": ["react", "typescript", "css"],
    "max_concurrent_tasks": 2
  }'
```

**Response:**
```json
{
  "id": "456e7890-e89b-12d3-a456-426614174001",
  "name": "frontend-expert",
  "role": "developer",
  "status": "active",
  "capabilities": ["react", "typescript", "css"],
  "max_concurrent_tasks": 2,
  "created_at": "2024-01-15T12:00:00Z"
}
```

#### Get Agent Details
```http
GET /agents/{agent_id}
```

**Example Request:**
```bash
curl -X GET http://localhost:8000/api/v1/agents/123e4567-e89b-12d3-a456-426614174000 \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "name": "backend-specialist",
  "role": "developer",
  "status": "active",
  "capabilities": ["python", "fastapi", "postgresql", "docker"],
  "specializations": ["api_development", "database_optimization"],
  "max_concurrent_tasks": 3,
  "current_tasks": [
    {
      "id": "task-789",
      "title": "Implement user authentication API",
      "status": "in_progress",
      "assigned_at": "2024-01-15T09:00:00Z"
    }
  ],
  "performance_metrics": {
    "completed_tasks": 45,
    "success_rate": 0.96,
    "average_completion_time": 4.2,
    "quality_score": 0.89,
    "collaboration_score": 0.93
  },
  "created_at": "2024-01-01T00:00:00Z",
  "last_activity": "2024-01-15T10:30:00Z"
}
```

#### Update Agent
```http
PUT /agents/{agent_id}
```

**Request Body:**
```json
{
  "max_concurrent_tasks": 4,
  "capabilities": ["python", "fastapi", "postgresql", "docker", "kubernetes"],
  "status": "active"
}
```

**Example Request:**
```bash
curl -X PUT http://localhost:8000/api/v1/agents/123e4567-e89b-12d3-a456-426614174000 \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "max_concurrent_tasks": 4,
    "capabilities": ["python", "fastapi", "postgresql", "docker", "kubernetes"]
  }'
```

#### Delete Agent
```http
DELETE /agents/{agent_id}
```

**Example Request:**
```bash
curl -X DELETE http://localhost:8000/api/v1/agents/123e4567-e89b-12d3-a456-426614174000 \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "message": "Agent deleted successfully",
  "deleted_at": "2024-01-15T12:30:00Z"
}
```

### Tasks API

#### List Tasks
```http
GET /tasks
```

**Query Parameters:**
- `limit` (optional): Number of tasks to return (default: 50)
- `offset` (optional): Number of tasks to skip (default: 0)
- `status` (optional): Filter by status (`pending`, `assigned`, `in_progress`, `completed`, `failed`)
- `agent_id` (optional): Filter by assigned agent
- `priority` (optional): Filter by priority (`low`, `medium`, `high`, `critical`)

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/tasks?status=pending&priority=high" \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "tasks": [
    {
      "id": "task-123",
      "title": "Implement user authentication",
      "description": "Build JWT-based authentication system with refresh tokens",
      "task_type": "development",
      "status": "pending",
      "priority": "high",
      "estimated_effort": 8,
      "assigned_agent_id": null,
      "requirements": ["python", "fastapi", "jwt"],
      "created_at": "2024-01-15T08:00:00Z",
      "deadline": "2024-01-20T17:00:00Z"
    }
  ],
  "total": 15,
  "page": 1,
  "pages": 1
}
```

#### Create Task
```http
POST /tasks
```

**Request Body:**
```json
{
  "title": "Build user dashboard",
  "description": "Create responsive user dashboard with React and TypeScript",
  "task_type": "development",
  "priority": "medium",
  "estimated_effort": 12,
  "requirements": ["react", "typescript", "css"],
  "deadline": "2024-01-25T17:00:00Z",
  "metadata": {
    "complexity": "medium",
    "client_facing": true
  }
}
```

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Build user dashboard",
    "description": "Create responsive user dashboard",
    "task_type": "development",
    "priority": "medium",
    "estimated_effort": 12,
    "requirements": ["react", "typescript"]
  }'
```

#### Assign Task to Agent
```http
POST /tasks/{task_id}/assign/{agent_id}
```

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/v1/tasks/task-123/assign/123e4567-e89b-12d3-a456-426614174000 \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "id": "task-123",
  "title": "Implement user authentication",
  "status": "assigned",
  "assigned_agent_id": "123e4567-e89b-12d3-a456-426614174000",
  "assigned_at": "2024-01-15T12:45:00Z"
}
```

### Workflows API

#### List Workflows
```http
GET /workflows
```

**Query Parameters:**
- `status` (optional): Filter by status (`draft`, `active`, `paused`, `completed`, `failed`)
- `workflow_type` (optional): Filter by type (`sequential`, `parallel`, `dag`)

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/workflows?status=active" \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "workflows": [
    {
      "id": "workflow-456",
      "name": "E-commerce Platform Development",
      "description": "Complete development of e-commerce platform",
      "workflow_type": "dag",
      "status": "active",
      "progress": 0.65,
      "created_at": "2024-01-10T00:00:00Z",
      "started_at": "2024-01-10T09:00:00Z",
      "estimated_completion": "2024-02-10T17:00:00Z",
      "tasks_total": 25,
      "tasks_completed": 16,
      "agents_assigned": 4
    }
  ],
  "total": 8,
  "page": 1,
  "pages": 1
}
```

#### Create Workflow
```http
POST /workflows
```

**Request Body:**
```json
{
  "name": "Mobile App Development",
  "description": "Build cross-platform mobile application",
  "workflow_type": "dag",
  "tasks": [
    {
      "id": "task-ui-design",
      "title": "Design UI mockups",
      "dependencies": [],
      "requirements": ["ui_design", "figma"]
    },
    {
      "id": "task-api-backend", 
      "title": "Build API backend",
      "dependencies": [],
      "requirements": ["python", "fastapi"]
    },
    {
      "id": "task-mobile-app",
      "title": "Develop mobile app",
      "dependencies": ["task-ui-design", "task-api-backend"],
      "requirements": ["react_native", "typescript"]
    }
  ],
  "configuration": {
    "auto_assign": true,
    "parallel_execution": true,
    "failure_strategy": "retry"
  }
}
```

#### Start Workflow
```http
POST /workflows/{workflow_id}/start
```

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/v1/workflows/workflow-456/start \
  -H "Authorization: Bearer <token>"
```

### Multi-Agent Coordination API

#### Create Coordinated Project
```http
POST /coordination/projects
```

**Request Body:**
```json
{
  "name": "AI-Powered Analytics Dashboard",
  "description": "Build comprehensive analytics dashboard with ML insights",
  "requirements": {
    "capabilities": ["machine_learning", "data_visualization", "backend_api"],
    "complexity": "medium",
    "timeline": "2_weeks",
    "tech_stack": ["python", "react", "tensorflow", "postgresql"]
  },
  "coordination_mode": "parallel",
  "deadline": "2024-02-15T00:00:00Z",
  "quality_gates": {
    "test_coverage": 85,
    "code_review": true,
    "security_scan": true
  }
}
```

**Example Request:**
```bash
curl -X POST http://localhost:8000/api/v1/coordination/projects \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "AI Analytics Dashboard",
    "description": "ML-powered analytics dashboard",
    "requirements": {
      "capabilities": ["machine_learning", "data_visualization"],
      "complexity": "medium"
    },
    "coordination_mode": "parallel"
  }'
```

**Response:**
```json
{
  "project_id": "proj-789",
  "name": "AI-Powered Analytics Dashboard",
  "status": "created",
  "coordination_mode": "parallel",
  "agents_assigned": 3,
  "estimated_duration": "2 weeks",
  "created_at": "2024-01-15T13:00:00Z"
}
```

#### Get Project Status
```http
GET /coordination/projects/{project_id}
```

**Example Request:**
```bash
curl -X GET http://localhost:8000/api/v1/coordination/projects/proj-789 \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "project_id": "proj-789",
  "name": "AI-Powered Analytics Dashboard",
  "status": "in_progress",
  "progress": 0.35,
  "coordination_mode": "parallel",
  "agents": [
    {
      "agent_id": "agent-ml-001",
      "role": "data_scientist",
      "status": "active",
      "current_task": "Feature engineering pipeline"
    },
    {
      "agent_id": "agent-fe-002", 
      "role": "frontend_developer",
      "status": "active",
      "current_task": "Dashboard UI components"
    }
  ],
  "tasks": {
    "total": 12,
    "completed": 4,
    "in_progress": 3,
    "pending": 5
  },
  "quality_metrics": {
    "test_coverage": 78,
    "code_review_completion": 85,
    "security_scan_passed": true
  },
  "created_at": "2024-01-15T13:00:00Z",
  "estimated_completion": "2024-01-29T17:00:00Z"
}
```

### WebSocket API

#### Connect to Real-time Updates
```
ws://localhost:8000/ws/observability
```

**Connection Example (JavaScript):**
```javascript
const socket = new WebSocket('ws://localhost:8000/ws/observability');

socket.onopen = function(event) {
    console.log('Connected to WebSocket');
    // Subscribe to specific events
    socket.send(JSON.stringify({
        type: 'subscribe',
        events: ['agent_status', 'task_progress', 'system_alerts']
    }));
};

socket.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'agent_status_update':
            updateAgentStatus(data.agent_id, data.status);
            break;
        case 'task_progress':
            updateTaskProgress(data.task_id, data.progress);
            break;
        case 'system_alert':
            showAlert(data.message, data.severity);
            break;
    }
};
```

**Message Types:**
- `agent_status_update`: Agent status changes
- `task_progress`: Task completion updates
- `workflow_event`: Workflow state changes
- `system_alert`: System notifications
- `coordination_update`: Multi-agent coordination events

#### Agent-Specific WebSocket
```
ws://localhost:8000/ws/agents/{agent_id}
```

**Example Connection:**
```javascript
const agentSocket = new WebSocket('ws://localhost:8000/ws/agents/123e4567-e89b-12d3-a456-426614174000');

agentSocket.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'task_assigned') {
        console.log('New task assigned:', data.task);
    }
};
```

### System API

#### Health Check
```http
GET /system/health
```

**Example Request:**
```bash
curl -X GET http://localhost:8000/api/v1/system/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T14:00:00Z",
  "version": "2.0.0",
  "uptime": 172800,
  "dependencies": {
    "database": {
      "status": "healthy",
      "response_time_ms": 5.2,
      "connection_pool": {
        "active": 8,
        "idle": 12,
        "max": 20
      }
    },
    "redis": {
      "status": "healthy",
      "response_time_ms": 1.8,
      "memory_usage": "45.2MB"
    },
    "external_services": {
      "anthropic_api": {
        "status": "healthy",
        "last_check": "2024-01-15T13:59:00Z"
      }
    }
  },
  "system_metrics": {
    "active_agents": 15,
    "active_workflows": 5,
    "pending_tasks": 23,
    "websocket_connections": 8
  }
}
```

#### System Metrics
```http
GET /system/metrics
```

**Query Parameters:**
- `format` (optional): Response format (`json`, `prometheus`) (default: json)
- `timerange` (optional): Time range for metrics (`1h`, `24h`, `7d`) (default: 1h)

**Example Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/system/metrics?timerange=24h" \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "system_performance": {
    "avg_response_time_ms": 125.3,
    "p95_response_time_ms": 250.8,
    "p99_response_time_ms": 450.2,
    "error_rate": 0.02,
    "throughput_requests_per_minute": 1250
  },
  "agent_metrics": {
    "total_agents": 15,
    "active_agents": 12,
    "avg_utilization": 0.73,
    "task_completion_rate": 0.94,
    "avg_task_duration_hours": 3.2
  },
  "workflow_metrics": {
    "active_workflows": 5,
    "completed_workflows_24h": 8,
    "avg_workflow_duration_hours": 18.5,
    "success_rate": 0.96
  },
  "resource_usage": {
    "cpu_usage_percent": 45.2,
    "memory_usage_mb": 2048,
    "disk_usage_percent": 23.4,
    "network_io_mbps": 12.8
  }
}
```

## Error Handling

### Standard Error Response Format

All API errors follow this consistent format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "agent_id",
      "issue": "Agent ID must be a valid UUID"
    },
    "request_id": "req_123456789",
    "timestamp": "2024-01-15T14:30:00Z"
  }
}
```

### Common Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | `VALIDATION_ERROR` | Invalid request parameters |
| 401 | `AUTHENTICATION_REQUIRED` | Missing or invalid authentication |
| 403 | `INSUFFICIENT_PERMISSIONS` | User lacks required permissions |
| 404 | `RESOURCE_NOT_FOUND` | Requested resource doesn't exist |
| 409 | `RESOURCE_CONFLICT` | Resource conflict (e.g., duplicate name) |
| 422 | `BUSINESS_LOGIC_ERROR` | Request valid but violates business rules |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests |
| 500 | `INTERNAL_SERVER_ERROR` | Unexpected server error |
| 502 | `EXTERNAL_SERVICE_ERROR` | External service unavailable |
| 503 | `SERVICE_UNAVAILABLE` | Service temporarily unavailable |

### Error Handling Examples

#### 400 - Validation Error
```bash
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"name": ""}'  # Invalid empty name
```

**Response:**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Agent name cannot be empty",
    "details": {
      "field": "name",
      "constraint": "min_length_1"
    },
    "request_id": "req_987654321"
  }
}
```

#### 404 - Resource Not Found
```bash
curl -X GET http://localhost:8000/api/v1/agents/nonexistent-id \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "Agent not found",
    "details": {
      "resource_type": "agent",
      "resource_id": "nonexistent-id"
    },
    "request_id": "req_456789123"
  }
}
```

## Rate Limiting

API endpoints are rate-limited to ensure system stability:

| Endpoint Category | Rate Limit | Window |
|------------------|------------|---------|
| Authentication | 5 requests | 1 minute |
| Agent Operations | 100 requests | 1 minute |
| Task Operations | 200 requests | 1 minute |
| Workflow Operations | 50 requests | 1 minute |
| System Metrics | 30 requests | 1 minute |
| WebSocket Connections | 10 connections | Per IP |

Rate limit headers are included in all responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248000
```

## Pagination

List endpoints support cursor-based pagination:

**Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/agents?limit=10&cursor=eyJpZCI6IjEyMyJ9" \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "agents": [...],
  "pagination": {
    "has_next": true,
    "has_previous": false,
    "next_cursor": "eyJpZCI6IjQ1NiJ9",
    "previous_cursor": null,
    "total_count": 150
  }
}
```

## SDK Examples

### Python SDK

```python
import asyncio
from leanvibe_client import LeanVibeClient

async def main():
    # Initialize client
    client = LeanVibeClient(
        base_url="http://localhost:8000/api/v1",
        api_key="your-api-key"
    )
    
    # Create an agent
    agent = await client.agents.create(
        name="python-specialist",
        role="developer",
        capabilities=["python", "fastapi", "pytest"]
    )
    
    # Create and assign a task
    task = await client.tasks.create(
        title="Build authentication system",
        description="Implement JWT-based auth",
        requirements=["python", "fastapi"]
    )
    
    await client.tasks.assign(task.id, agent.id)
    
    # Create a coordinated project
    project = await client.coordination.create_project(
        name="Web Application",
        requirements={
            "capabilities": ["frontend", "backend", "testing"]
        },
        coordination_mode="parallel"
    )
    
    await client.coordination.start_project(project.id)

asyncio.run(main())
```

### JavaScript SDK

```javascript
import { LeanVibeClient } from '@leanvibe/client';

const client = new LeanVibeClient({
    baseUrl: 'http://localhost:8000/api/v1',
    apiKey: 'your-api-key'
});

// Create agent
const agent = await client.agents.create({
    name: 'javascript-expert',
    role: 'developer',
    capabilities: ['javascript', 'react', 'nodejs']
});

// Listen for real-time updates
client.websocket.connect('observability');
client.websocket.on('agent_status_update', (data) => {
    console.log('Agent status changed:', data);
});

// Create coordinated project
const project = await client.coordination.createProject({
    name: 'Mobile App Development',
    requirements: {
        capabilities: ['react_native', 'api_integration']
    },
    coordinationMode: 'collaborative'
});
```

## OpenAPI Specification

The complete OpenAPI specification is available at:
- **JSON**: http://localhost:8000/openapi.json
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Testing the API

### Using curl

```bash
# Test authentication
export API_TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password"}' | \
  jq -r '.access_token')

# Test agent creation
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"test-agent","role":"developer","capabilities":["python"]}'

# Test workflow creation
curl -X POST http://localhost:8000/api/v1/workflows \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d @workflow-example.json
```

### Using Postman

Import the OpenAPI specification into Postman:
1. Open Postman
2. Click "Import" 
3. Enter URL: `http://localhost:8000/openapi.json`
4. Configure authentication with your JWT token

### Using HTTPie

```bash
# Install HTTPie
pip install httpie

# Test API endpoints
http POST localhost:8000/api/v1/auth/login email=test@example.com password=password
http GET localhost:8000/api/v1/agents Authorization:"Bearer <token>"
http POST localhost:8000/api/v1/tasks Authorization:"Bearer <token>" title="Test Task"
```

## Support and Resources

- **GitHub Repository**: https://github.com/LeanVibe/bee-hive
- **Issue Tracker**: https://github.com/LeanVibe/bee-hive/issues
- **Discord Community**: https://discord.gg/leanvibe
- **Email Support**: api-support@leanvibe.com

---

*Last updated: January 15, 2024*