# Mobile PWA Backend API Specification

**Generated from LeanVibe Agent Hive 2.0 Mobile PWA Analysis**  
**Date**: 2025-08-20  
**Source**: Phase 1 Foundation Reality Check - Mobile PWA Component Analysis (85% functional)

## Executive Summary

The Mobile PWA is the strongest component in the LeanVibe Agent Hive 2.0 system at 85% functional. This specification documents the complete backend API surface required to support the PWA's functionality, derived from comprehensive code analysis of TypeScript services, components, and configuration files.

## Table of Contents

1. [Base Configuration](#base-configuration)
2. [Authentication & Authorization](#authentication--authorization)
3. [Agent Management API](#agent-management-api)
4. [Task Management API](#task-management-api)
5. [System Health & Monitoring API](#system-health--monitoring-api)
6. [Events & Notifications API](#events--notifications-api)
7. [Performance Metrics API](#performance-metrics-api)
8. [WebSocket Real-time API](#websocket-real-time-api)
9. [Data Models](#data-models)
10. [Error Handling](#error-handling)

---

## Base Configuration

### API Base URLs
```
Development: http://localhost:8000
Production: https://api.leanvibe.dev
```

### Expected Proxy Configuration
The PWA expects these proxy patterns:
- `/dashboard/api/*` → Backend API endpoints
- `/api/*` → Alternative API endpoints  
- `/health` → Health check endpoint

### CORS Requirements
```
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS
Access-Control-Allow-Headers: Origin, X-Requested-With, Content-Type, Accept, Authorization
```

---

## Authentication & Authorization

### Current Status
🚫 **Authentication Postponed**: The PWA includes comprehensive auth infrastructure but operates without authentication per project requirements.

### Authentication Infrastructure (Ready for Future Implementation)

#### Login Endpoints
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password123"
}
```

#### Response Format
```json
{
  "access_token": "jwt_token_here",
  "refresh_token": "refresh_token_here",
  "user": {
    "id": "user-123",
    "email": "user@example.com",
    "full_name": "John Doe",
    "role": "super_admin",
    "permissions": ["read:agents", "write:agents", "admin:system"],
    "company_name": "LeanVibe",
    "pilot_ids": ["pilot-001"],
    "is_active": true,
    "last_login": "2025-08-20T10:30:00Z",
    "auth_method": "password"
  }
}
```

#### Token Refresh
```http
POST /api/v1/auth/refresh
{
  "refresh_token": "refresh_token_here"
}
```

#### User Profile
```http
GET /api/v1/auth/me
Authorization: Bearer jwt_token_here
```

#### WebAuthn Support (Future)
```http
POST /api/v1/auth/webauthn/challenge
POST /api/v1/auth/webauthn/authenticate
```

#### Auth0 Integration Support
- Domain: Configurable via environment
- Client ID: Configurable via environment
- Audience: `https://leanvibe-agent-hive`
- Scope: `openid profile email read:agents write:agents`

---

## Agent Management API

### Agent System Activation
```http
POST /api/agents/activate
Content-Type: application/json

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
      "name": "Product Manager Agent",
      "capabilities": ["requirements_analysis", "project_planning", "documentation"],
      "created_at": "2025-08-20T10:00:00Z",
      "updated_at": "2025-08-20T10:00:00Z",
      "last_activity": "2025-08-20T10:30:00Z",
      "current_task_id": null,
      "performance_metrics": {
        "tasks_completed": 0,
        "tasks_failed": 0,
        "average_completion_time": 0,
        "cpu_usage": 15.2,
        "memory_usage": 45.8,
        "success_rate": 100.0,
        "uptime": 3600
      },
      "error_message": null
    }
  },
  "team_composition": {
    "product_manager": "agent-123",
    "architect": "agent-456",
    "backend_developer": "agent-789",
    "frontend_developer": "agent-101",
    "qa_engineer": "agent-112"
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
      "id": "agent-123",
      "role": "product_manager",
      "status": "idle",
      "name": "Product Manager Agent",
      "capabilities": ["requirements_analysis", "project_planning"],
      "created_at": "2025-08-20T10:00:00Z",
      "updated_at": "2025-08-20T10:30:00Z",
      "last_activity": "2025-08-20T10:30:00Z",
      "current_task_id": "task-456",
      "performance_metrics": {
        "tasks_completed": 12,
        "tasks_failed": 1,
        "average_completion_time": 1800,
        "cpu_usage": 25.5,
        "memory_usage": 67.8,
        "success_rate": 92.3,
        "uptime": 7200
      },
      "error_message": null
    }
  },
  "orchestrator_agents_detail": {},
  "system_ready": true,
  "hybrid_integration": true,
  "error": null
}
```

### Spawn Specific Agent
```http
POST /api/agents/spawn/{role}
```

**Path Parameters:**
- `role`: One of `product_manager`, `architect`, `backend_developer`, `frontend_developer`, `qa_engineer`, `devops_engineer`

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

### Individual Agent Deactivation
```http
DELETE /api/agents/{agent_id}
```

### Agent Configuration
```http
PUT /api/agents/{agent_id}/configure
{
  "capabilities": ["new_capability"],
  "name": "Updated Agent Name"
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
      "capabilities": ["api_development", "database_design", "server_logic", "testing"]
    },
    "frontend_developer": {
      "count": 1,
      "capabilities": ["ui_development", "responsive_design", "javascript"]
    },
    "qa_engineer": {
      "count": 1,
      "capabilities": ["test_automation", "quality_assurance", "bug_reporting"]
    }
  },
  "system_capabilities": ["requirements_analysis", "api_development", "ui_development", "test_creation", "deployment", "documentation"]
}
```

---

## Task Management API

### Create Task
```http
POST /api/v1/tasks/
Content-Type: application/json

{
  "title": "Implement user authentication API",
  "description": "Create JWT-based authentication endpoints with comprehensive error handling",
  "task_type": "feature",
  "priority": "high",
  "required_capabilities": ["api_development", "security", "testing"],
  "estimated_effort": 5,
  "context": {
    "requirements": ["fastapi", "jwt", "database"],
    "acceptance_criteria": ["JWT token generation", "Token validation", "User registration", "Password reset"]
  }
}
```

**Response:**
```json
{
  "id": "task-123",
  "title": "Implement user authentication API",
  "description": "Create JWT-based authentication endpoints with comprehensive error handling",
  "task_type": "feature",
  "priority": "high", 
  "status": "pending",
  "required_capabilities": ["api_development", "security", "testing"],
  "estimated_effort": 5,
  "actual_effort": null,
  "assigned_agent_id": null,
  "context": {
    "requirements": ["fastapi", "jwt", "database"],
    "acceptance_criteria": ["JWT token generation", "Token validation", "User registration", "Password reset"]
  },
  "result": null,
  "error_message": null,
  "retry_count": 0,
  "max_retries": 3,
  "created_at": "2025-08-20T10:00:00Z",
  "updated_at": "2025-08-20T10:00:00Z",
  "assigned_at": null,
  "started_at": null,
  "completed_at": null
}
```

### Get Tasks
```http
GET /api/v1/tasks/?status=pending&priority=high&limit=50&offset=0
```

**Query Parameters:**
- `status`: Filter by status (`pending`, `assigned`, `in_progress`, `completed`, `failed`, `cancelled`)
- `priority`: Filter by priority (`low`, `medium`, `high`, `critical`)
- `task_type`: Filter by type (`feature`, `bug_fix`, `refactor`, `test`, `documentation`, `deployment`)
- `assigned_agent_id`: Filter by assigned agent
- `limit`: Number of tasks to return (default: 50)
- `offset`: Number of tasks to skip (default: 0)

**Response:**
```json
{
  "tasks": [
    {
      "id": "task-123",
      "title": "Implement user authentication API",
      "description": "Create JWT-based authentication endpoints",
      "task_type": "feature",
      "priority": "high",
      "status": "in_progress",
      "required_capabilities": ["api_development", "security"],
      "estimated_effort": 5,
      "actual_effort": 3,
      "assigned_agent_id": "agent-789",
      "context": {},
      "result": null,
      "error_message": null,
      "retry_count": 0,
      "max_retries": 3,
      "created_at": "2025-08-20T10:00:00Z",
      "updated_at": "2025-08-20T11:00:00Z",
      "assigned_at": "2025-08-20T10:15:00Z",
      "started_at": "2025-08-20T10:30:00Z",
      "completed_at": null
    }
  ],
  "total": 25,
  "offset": 0,
  "limit": 50
}
```

### Get Specific Task
```http
GET /api/v1/tasks/{task_id}
```

### Update Task
```http
PUT /api/v1/tasks/{task_id}
{
  "status": "in_progress",
  "priority": "critical",
  "context": {
    "progress_notes": "50% complete, working on token validation"
  }
}
```

### Delete Task
```http
DELETE /api/v1/tasks/{task_id}
```

### Task Assignment
```http
POST /api/v1/tasks/{task_id}/assign/{agent_id}
```

### Task Lifecycle Operations
```http
POST /api/v1/tasks/{task_id}/start
POST /api/v1/tasks/{task_id}/complete
{
  "result": {
    "implementation": "JWT endpoints implemented successfully",
    "test_coverage": "95%",
    "performance_metrics": "Response time < 100ms"
  }
}

POST /api/v1/tasks/{task_id}/fail
{
  "error_message": "Database connection failed",
  "can_retry": true
}
```

### Agent-Specific Tasks
```http
GET /api/v1/tasks/agent/{agent_id}?status=in_progress&limit=10
```

---

## System Health & Monitoring API

### Primary Live Data Endpoint (Most Critical)
```http
GET /dashboard/api/live-data
```

**This is the PWA's primary data source. The backend adapter transforms this into all other service data.**

**Response:**
```json
{
  "metrics": {
    "active_projects": 3,
    "active_agents": 5,
    "agent_utilization": 75,
    "completed_tasks": 147,
    "active_conflicts": 1,
    "system_efficiency": 87,
    "system_status": "healthy",
    "last_updated": "2025-08-20T10:30:00Z"
  },
  "agent_activities": [
    {
      "agent_id": "agent-123",
      "name": "Product Manager Agent",
      "status": "active",
      "current_project": "Authentication System",
      "current_task": "Requirements analysis for OAuth integration",
      "task_progress": 65,
      "performance_score": 92,
      "specializations": ["requirements_analysis", "project_planning", "documentation"]
    },
    {
      "agent_id": "agent-789",
      "name": "Backend Developer Agent", 
      "status": "busy",
      "current_project": "API Development",
      "current_task": "Implementing JWT authentication endpoints",
      "task_progress": 45,
      "performance_score": 88,
      "specializations": ["api_development", "database_design", "testing"]
    }
  ],
  "project_snapshots": [
    {
      "name": "Authentication System",
      "status": "active",
      "progress_percentage": 75,
      "participating_agents": ["agent-123", "agent-789"],
      "completed_tasks": 8,
      "active_tasks": 3,
      "conflicts": 0,
      "quality_score": 95
    },
    {
      "name": "Dashboard Enhancement",
      "status": "completed",
      "progress_percentage": 100,
      "participating_agents": ["agent-456", "agent-101"],
      "completed_tasks": 12,
      "active_tasks": 0,
      "conflicts": 0,
      "quality_score": 98
    }
  ],
  "conflict_snapshots": [
    {
      "conflict_type": "Resource Contention",
      "severity": "medium",
      "project_name": "Authentication System",
      "description": "Multiple agents trying to access the same configuration file",
      "affected_agents": ["agent-123", "agent-789"],
      "impact_score": 3,
      "auto_resolvable": true
    }
  ]
}
```

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-20T10:30:00Z",
  "components": {
    "database": {
      "status": "healthy",
      "latency": 15,
      "errors": 0,
      "details": {
        "connection_pool": "8/20 connections",
        "query_avg_time": "12ms"
      },
      "lastCheck": "2025-08-20T10:29:55Z"
    },
    "redis": {
      "status": "healthy",
      "latency": 5,
      "errors": 0,
      "details": {
        "memory_usage": "45MB",
        "connected_clients": 12
      },
      "lastCheck": "2025-08-20T10:29:55Z"
    },
    "orchestrator": {
      "status": "degraded",
      "latency": 150,
      "errors": 2,
      "details": {
        "active_workflows": 5,
        "queued_tasks": 23
      },
      "lastCheck": "2025-08-20T10:29:50Z"
    },
    "agents": {
      "status": "healthy",
      "latency": 25,
      "errors": 0,
      "details": {
        "active_agents": 5,
        "total_capacity": "75%"
      },
      "lastCheck": "2025-08-20T10:29:58Z"
    }
  },
  "metrics": {
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "disk_usage": 23.5,
    "network_io": {
      "in": 1024000,
      "out": 2048000
    },
    "active_connections": 15,
    "uptime": 86400
  }
}
```

---

## Events & Notifications API

### Get Recent Events
```http
GET /api/v1/events/?limit=50&severity=high&agent_id=agent-123
```

**Query Parameters:**
- `limit`: Number of events to return (default: 50)
- `offset`: Number of events to skip (default: 0)
- `severity`: Filter by severity (`info`, `warning`, `error`, `critical`)
- `type`: Filter by event type (`agent_activated`, `task_created`, `task_completed`, etc.)
- `agent_id`: Filter by agent ID
- `start_date`: Filter events after this date
- `end_date`: Filter events before this date

**Response:**
```json
{
  "events": [
    {
      "id": "event-123",
      "type": "task_completed",
      "severity": "info",
      "title": "Task Completed Successfully",
      "description": "Authentication API implementation completed",
      "source": "agent-789",
      "agent_id": "agent-789",
      "task_id": "task-456",
      "data": {
        "completion_time": 3600,
        "quality_score": 95,
        "test_coverage": "98%"
      },
      "timestamp": "2025-08-20T10:30:00Z",
      "acknowledged": false
    },
    {
      "id": "event-124",
      "type": "agent_activated",
      "severity": "info",
      "title": "New Agent Activated",
      "description": "QA Engineer agent successfully spawned and ready",
      "source": "system",
      "agent_id": "agent-112",
      "task_id": null,
      "data": {
        "role": "qa_engineer",
        "capabilities": ["test_automation", "quality_assurance"]
      },
      "timestamp": "2025-08-20T10:25:00Z",
      "acknowledged": false
    }
  ],
  "total": 147,
  "offset": 0,
  "limit": 50
}
```

### Acknowledge Event
```http
POST /api/v1/events/{event_id}/acknowledge
```

---

## Performance Metrics API

### Get Current Performance
```http
GET /api/v1/metrics/performance/current
```

**Response:**
```json
{
  "system_metrics": {
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "network_usage": 23.1,
    "disk_usage": 34.5
  },
  "agent_metrics": {
    "agent-123": {
      "performance_score": 92,
      "task_completion_rate": 95.5,
      "error_rate": 2.3,
      "uptime": 99.8
    },
    "agent-789": {
      "performance_score": 88,
      "task_completion_rate": 87.2,
      "error_rate": 4.1,
      "uptime": 98.5
    }
  },
  "response_times": {
    "api_response_time": 125.5,
    "websocket_latency": 35.2,
    "database_query_time": 65.8
  },
  "throughput": {
    "requests_per_second": 75.3,
    "tasks_completed_per_hour": 148,
    "agent_operations_per_minute": 23.7
  },
  "alerts": [
    {
      "id": "alert-123",
      "type": "threshold",
      "severity": "warning",
      "message": "High memory usage detected on agent-789",
      "timestamp": "2025-08-20T10:28:00Z",
      "metric": "Memory Usage",
      "current_value": 85,
      "threshold_value": 80
    }
  ],
  "timestamp": "2025-08-20T10:30:00Z"
}
```

### Get Performance History
```http
GET /api/v1/metrics/performance/history?timeframe=24h&agent_id=agent-123
```

**Response:**
```json
{
  "timeframe": "24h",
  "agent_id": "agent-123",
  "data_points": [
    {
      "timestamp": "2025-08-20T09:30:00Z",
      "cpu_usage": 42.1,
      "memory_usage": 65.3,
      "task_completion_rate": 94.2,
      "error_rate": 1.8
    },
    {
      "timestamp": "2025-08-20T10:00:00Z", 
      "cpu_usage": 45.8,
      "memory_usage": 68.1,
      "task_completion_rate": 95.1,
      "error_rate": 2.1
    }
  ],
  "summary": {
    "avg_cpu": 43.9,
    "avg_memory": 66.7,
    "peak_cpu": 58.3,
    "peak_memory": 72.4,
    "trend": "stable"
  }
}
```

### Security Metrics
```http
GET /api/v1/metrics/security
```

**Response:**
```json
{
  "threat_detection": {
    "active_threats": 0,
    "resolved_today": 5,
    "false_positives": 2,
    "threat_level": "minimal"
  },
  "authentication": {
    "successful_logins": 87,
    "failed_attempts": 3,
    "suspicious_logins": 0,
    "active_sessions": 12,
    "mfa_compliance_rate": 95.5
  },
  "access_control": {
    "permission_violations": 1,
    "unauthorized_access_attempts": 2,
    "privilege_escalations": 0,
    "data_access_anomalies": 0
  },
  "network_security": {
    "blocked_connections": 23,
    "malicious_requests": 1,
    "rate_limit_violations": 8,
    "ddos_attempts": 0
  },
  "data_protection": {
    "encryption_status": "healthy",
    "backup_status": "current",
    "data_integrity_score": 98.7,
    "compliance_violations": 0
  },
  "system_security": {
    "vulnerability_score": 15,
    "patch_compliance": 95,
    "security_updates_pending": 2,
    "configuration_drift": 3
  },
  "timestamp": "2025-08-20T10:30:00Z"
}
```

---

## WebSocket Real-time API

### Connection URL
```
ws://localhost:8000/api/dashboard/ws/dashboard?access_token=jwt_token_here
```

### Connection Messages

#### Connection Established
```json
{
  "type": "connection_established",
  "connection_id": "conn-123",
  "subscriptions": ["agents", "tasks", "system", "alerts"],
  "server_time": "2025-08-20T10:30:00Z"
}
```

#### Dashboard Initialized
```json
{
  "type": "dashboard_initialized",
  "subscriptions": ["agents", "tasks", "system", "alerts"],
  "connection_info": {
    "client_id": "client-123",
    "session_id": "session-456"
  }
}
```

### Real-time Update Messages

#### Agent Status Update
```json
{
  "type": "agent_update",
  "subscription": "agents",
  "data": {
    "agent_id": "agent-123",
    "status": "busy",
    "current_task_id": "task-456",
    "performance_metrics": {
      "cpu_usage": 67.2,
      "memory_usage": 45.8,
      "tasks_completed": 13,
      "error_rate": 2.1,
      "response_time": 150
    }
  },
  "timestamp": "2025-08-20T10:30:00Z"
}
```

#### Task Status Update
```json
{
  "type": "task_update",
  "subscription": "tasks",
  "data": {
    "task_id": "task-456",
    "status": "completed",
    "assigned_agent_id": "agent-789",
    "progress": 100,
    "result": {
      "success": true,
      "completion_time": 3600,
      "quality_score": 95
    }
  },
  "timestamp": "2025-08-20T10:30:00Z"
}
```

#### Performance Metrics Update
```json
{
  "type": "performance_update",
  "subscription": "system",
  "data": {
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "agent_count": 5,
    "active_tasks": 12,
    "system_efficiency": 87.5
  },
  "timestamp": "2025-08-20T10:30:00Z"
}
```

#### Critical Alert
```json
{
  "type": "critical_alert",
  "subscription": "alerts",
  "data": {
    "alerts": [
      {
        "id": "alert-456",
        "type": "system_error",
        "severity": "critical",
        "title": "Database Connection Lost",
        "message": "Primary database connection has been lost, switching to backup",
        "source": "Database Monitor",
        "timestamp": "2025-08-20T10:30:00Z",
        "status": "active",
        "affected_agents": ["agent-123", "agent-789"],
        "metadata": {
          "connection_id": "db-conn-primary",
          "retry_count": 3,
          "backup_status": "active"
        }
      }
    ],
    "timestamp": "2025-08-20T10:30:00Z"
  },
  "timestamp": "2025-08-20T10:30:00Z"
}
```

### Client-to-Server Messages

#### Ping/Heartbeat
```json
{
  "type": "ping",
  "timestamp": "2025-08-20T10:30:00Z",
  "client_time": 1724148600000
}
```

#### Agent Commands
```json
{
  "type": "agent-command",
  "data": {
    "agentId": "agent-123",
    "command": "pause",
    "parameters": {
      "reason": "Manual pause from dashboard"
    },
    "timestamp": "2025-08-20T10:30:00Z"
  }
}
```

#### Emergency Controls
```json
{
  "type": "emergency-stop",
  "data": {
    "reason": "Critical system error detected",
    "timestamp": "2025-08-20T10:30:00Z"
  }
}
```

### Configuration Messages

#### Configure Streaming
```json
{
  "type": "configure-streaming",
  "data": {
    "metricType": "agent-metrics",
    "frequencyMs": 2000,
    "timestamp": "2025-08-20T10:30:00Z"
  }
}
```

#### Mobile Dashboard Mode
```json
{
  "type": "configure-client",
  "data": {
    "mode": "mobile_dashboard",
    "real_time": true,
    "priority_events": ["agent_status", "system_alerts", "performance_metrics"],
    "update_frequency": "high",
    "compression": true
  }
}
```

---

## Data Models

### Core Enums

#### Agent Role
```typescript
type AgentRole = 
  | 'product_manager'
  | 'architect'
  | 'backend_developer'
  | 'frontend_developer'
  | 'qa_engineer'
  | 'devops_engineer'
```

#### Agent Status
```typescript
type AgentStatus = 
  | 'active'
  | 'idle'
  | 'busy'
  | 'error'
  | 'offline'
```

#### Task Status
```typescript
type TaskStatus = 
  | 'pending'
  | 'assigned'
  | 'in_progress'
  | 'completed'
  | 'failed'
  | 'cancelled'
```

#### Task Priority
```typescript
type TaskPriority = 
  | 'low'
  | 'medium'
  | 'high'
  | 'critical'
```

#### Task Type
```typescript
type TaskType = 
  | 'feature'
  | 'bug_fix'
  | 'refactor'
  | 'test'
  | 'documentation'
  | 'deployment'
```

### Core Data Structures

#### Agent
```typescript
interface Agent {
  id: string
  role: AgentRole
  status: AgentStatus
  name: string
  capabilities: string[]
  created_at: string
  updated_at: string
  last_activity: string
  current_task_id?: string
  performance_metrics: AgentPerformanceMetrics
  error_message?: string
}

interface AgentPerformanceMetrics {
  tasks_completed: number
  tasks_failed: number
  average_completion_time: number
  cpu_usage: number
  memory_usage: number
  success_rate: number
  uptime: number
}
```

#### Task
```typescript
interface Task {
  id: string
  title: string
  description: string
  task_type: TaskType
  priority: TaskPriority
  status: TaskStatus
  required_capabilities: string[]
  estimated_effort?: number
  actual_effort?: number
  assigned_agent_id?: string
  context: Record<string, any>
  result?: Record<string, any>
  error_message?: string
  retry_count: number
  max_retries: number
  created_at: string
  updated_at: string
  assigned_at?: string
  started_at?: string
  completed_at?: string
}
```

#### System Event
```typescript
interface SystemEvent {
  id: string
  type: EventType
  severity: EventSeverity
  title: string
  description: string
  source: string
  agent_id?: string
  task_id?: string
  data: Record<string, any>
  timestamp: string
  acknowledged: boolean
}

type EventType = 
  | 'agent_activated'
  | 'agent_deactivated'
  | 'task_created'
  | 'task_assigned'
  | 'task_started'
  | 'task_completed'
  | 'task_failed'
  | 'system_error'
  | 'performance_alert'
  | 'health_check'

type EventSeverity = 
  | 'info'
  | 'warning'
  | 'error'
  | 'critical'
```

---

## Error Handling

### Standard Error Response
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid task priority provided",
    "details": {
      "field": "priority",
      "provided": "super_high",
      "allowed": ["low", "medium", "high", "critical"]
    },
    "timestamp": "2025-08-20T10:30:00Z"
  }
}
```

### HTTP Status Codes
- `200` - Success
- `201` - Created
- `400` - Bad Request (validation errors)
- `401` - Unauthorized (when auth is enabled)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found
- `409` - Conflict (resource already exists)
- `422` - Unprocessable Entity (business logic errors)
- `429` - Rate Limited
- `500` - Internal Server Error
- `503` - Service Unavailable

### Common Error Codes
- `VALIDATION_ERROR` - Invalid request data
- `NETWORK_ERROR` - Network connection issues
- `TIMEOUT` - Request timeout
- `AGENT_NOT_FOUND` - Agent does not exist
- `TASK_NOT_FOUND` - Task does not exist
- `AGENT_BUSY` - Agent is currently busy
- `SYSTEM_OVERLOADED` - System at capacity
- `UNAUTHORIZED` - Authentication required
- `FORBIDDEN` - Insufficient permissions

---

## Implementation Notes

### Backend Requirements

1. **Primary Endpoint**: The `/dashboard/api/live-data` endpoint is critical - it serves as the source of truth for the PWA's BackendAdapter service which transforms this data for all other services.

2. **WebSocket Integration**: Real-time updates are essential for the mobile dashboard experience. The WebSocket should support:
   - Agent status changes
   - Task progress updates
   - System performance metrics
   - Critical alerts

3. **Performance Considerations**: 
   - Cache `/dashboard/api/live-data` responses (5-second TTL recommended)
   - Support efficient filtering and pagination for tasks/events
   - Optimize WebSocket message frequency based on connection quality

4. **Mobile Optimization**:
   - Support connection quality detection
   - Handle network interruptions gracefully
   - Provide compressed data when requested

5. **Offline Capabilities**: The PWA includes offline sync infrastructure that expects:
   - Optimistic updates support
   - Conflict resolution mechanisms
   - Background sync when connectivity is restored

### Security Notes

- Authentication infrastructure is ready but not enforced
- JWT token validation should be implemented for production
- WebAuthn support is available for future enhancement
- Role-based access control (RBAC) is designed but not enforced

### Integration Priority

1. **Phase 1** (Minimal Viable Backend):
   - `/dashboard/api/live-data` endpoint
   - Basic WebSocket connection
   - `/health` endpoint

2. **Phase 2** (Core Functionality):
   - Agent management endpoints
   - Task CRUD operations
   - System events

3. **Phase 3** (Enhanced Features):
   - Performance metrics APIs
   - Authentication system
   - Advanced WebSocket features

This specification provides the complete API surface needed to support the Mobile PWA's current functionality at 85% completion level, making it the ideal foundation for Phase 2 minimal backend implementation.