"""
OpenAPI Documentation for Team Coordination API

This module provides comprehensive API documentation with examples,
security specifications, and detailed descriptions for all endpoints.
"""

from typing import Dict, Any

# OpenAPI Tags for endpoint organization
TAGS_METADATA = [
    {
        "name": "Agent Registration",
        "description": "Register and manage agents in the coordination system with capability matching",
        "externalDocs": {
            "description": "Agent Registration Guide",
            "url": "https://docs.leanvibe.dev/coordination/agents"
        }
    },
    {
        "name": "Task Distribution", 
        "description": "Intelligent task distribution and assignment to optimal agents",
        "externalDocs": {
            "description": "Task Distribution Guide",
            "url": "https://docs.leanvibe.dev/coordination/tasks"
        }
    },
    {
        "name": "Performance Metrics",
        "description": "Real-time performance monitoring and analytics",
        "externalDocs": {
            "description": "Metrics Guide",
            "url": "https://docs.leanvibe.dev/coordination/metrics"
        }
    },
    {
        "name": "Real-time Updates",
        "description": "WebSocket endpoints for live coordination updates",
        "externalDocs": {
            "description": "WebSocket Integration",
            "url": "https://docs.leanvibe.dev/coordination/websockets"
        }
    },
    {
        "name": "System Health",
        "description": "Health checks and system status monitoring",
        "externalDocs": {
            "description": "Health Check Guide", 
            "url": "https://docs.leanvibe.dev/coordination/health"
        }
    }
]

# API Response Examples
RESPONSE_EXAMPLES = {
    "agent_registration_success": {
        "summary": "Successful agent registration",
        "description": "Agent successfully registered with capabilities",
        "value": {
            "agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "name": "Claude Backend Developer",
            "type": "claude",
            "status": "active",
            "current_workload": 0.0,
            "available_capacity": 0.75,
            "capabilities": [
                {
                    "name": "Python Development",
                    "description": "Expert Python programming with FastAPI and Django",
                    "category": "backend",
                    "level": "expert",
                    "confidence_score": 0.95,
                    "technologies": ["Python", "FastAPI", "Django", "SQLAlchemy"]
                }
            ],
            "active_tasks": 0,
            "completed_today": 0,
            "average_response_time_ms": 150.0,
            "last_heartbeat": "2025-01-30T15:30:00Z",
            "performance_score": 0.85
        }
    },
    
    "task_distribution_success": {
        "summary": "Successful task distribution",
        "description": "Task successfully assigned to optimal agent",
        "value": {
            "task_id": "550e8400-e29b-41d4-a716-446655440001",
            "assigned_agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "agent_name": "Claude Backend Developer",
            "assignment_confidence": 0.92,
            "estimated_completion_time": "2025-01-30T17:30:00Z",
            "capability_match_details": {
                "capability_score": 0.95,
                "workload_factor": 0.85,
                "performance_factor": 0.90
            },
            "workload_impact": 0.25
        }
    },
    
    "system_metrics": {
        "summary": "System coordination metrics",
        "description": "Comprehensive system performance metrics",
        "value": {
            "timestamp": "2025-01-30T15:30:00Z",
            "measurement_window_hours": 24,
            "total_agents": 15,
            "active_agents": 12,
            "idle_agents": 2,
            "overloaded_agents": 1,
            "total_tasks": 247,
            "completed_tasks": 203,
            "failed_tasks": 8,
            "overall_utilization_rate": 0.72,
            "task_assignment_success_rate": 0.97,
            "average_queue_time_minutes": 2.3,
            "system_throughput_tasks_per_hour": 8.5,
            "deadline_adherence_rate": 0.94,
            "bottleneck_capabilities": ["Database Design", "DevOps"],
            "scaling_recommendations": ["Add 2 DevOps specialists", "Improve load balancing"]
        }
    },
    
    "websocket_message": {
        "summary": "WebSocket status update",
        "description": "Real-time agent status update via WebSocket",
        "value": {
            "type": "agent_status_update",
            "timestamp": "2025-01-30T15:30:00Z",
            "agent_id": "550e8400-e29b-41d4-a716-446655440000",
            "agent_name": "Claude Backend Developer",
            "old_status": "active",
            "new_status": "busy",
            "current_workload": 0.8,
            "active_tasks": 3,
            "metadata": {
                "new_task_assigned": {
                    "task_id": "550e8400-e29b-41d4-a716-446655440002",
                    "task_title": "Implement OAuth2 Authentication",
                    "priority": "high"
                }
            }
        }
    }
}

# Error Response Examples
ERROR_EXAMPLES = {
    "agent_not_found": {
        "summary": "Agent not found",
        "description": "Agent ID does not exist in coordination system",
        "value": {
            "error_code": "AGENT_NOT_FOUND",
            "message": "Agent not found: 550e8400-e29b-41d4-a716-446655440099",
            "details": "Agent with ID '550e8400-e29b-41d4-a716-446655440099' does not exist in the coordination system",
            "timestamp": "2025-01-30T15:30:00Z",
            "request_id": "req_123456789",
            "help_url": "https://docs.leanvibe.dev/errors/agent_not_found"
        }
    },
    
    "validation_error": {
        "summary": "Validation error",
        "description": "Request data failed validation",
        "value": {
            "error_code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": "Found 2 validation error(s)",
            "validation_errors": [
                {
                    "field": "capabilities.0.confidence_score",
                    "message": "ensure this value is less than or equal to 1.0",
                    "invalid_value": 1.5,
                    "constraint": "less_than_equal"
                },
                {
                    "field": "agent_name",
                    "message": "ensure this value has at least 1 characters",
                    "invalid_value": "",
                    "constraint": "min_length"
                }
            ],
            "timestamp": "2025-01-30T15:30:00Z",
            "request_id": "req_123456790"
        }
    },
    
    "insufficient_capacity": {
        "summary": "No suitable agents available",
        "description": "No agents with required capabilities are available",
        "value": {
            "error_code": "INSUFFICIENT_CAPACITY",
            "message": "No suitable agents available",
            "details": "No agents available with required capabilities: Python Development, Database Design, API Testing",
            "timestamp": "2025-01-30T15:30:00Z",
            "request_id": "req_123456791",
            "help_url": "https://docs.leanvibe.dev/errors/insufficient_capacity"
        }
    },
    
    "rate_limit_exceeded": {
        "summary": "Rate limit exceeded",
        "description": "Too many requests within the time window",
        "value": {
            "error_code": "RATE_LIMIT_EXCEEDED",
            "message": "Rate limit exceeded",
            "details": "Exceeded rate limit of 100 requests per 60 seconds for /team-coordination/tasks/distribute",
            "timestamp": "2025-01-30T15:30:00Z",
            "request_id": "req_123456792",
            "help_url": "https://docs.leanvibe.dev/errors/rate_limit_exceeded"
        }
    }
}

# Request Body Examples
REQUEST_EXAMPLES = {
    "agent_registration": {
        "summary": "Register backend developer agent",
        "description": "Register a Claude agent with backend development capabilities",
        "value": {
            "agent_name": "Claude Backend Developer",
            "agent_type": "claude",
            "description": "Full-stack backend developer specialized in Python and API development",
            "capabilities": [
                {
                    "name": "Python Development",
                    "description": "Expert Python programming with modern frameworks",
                    "category": "backend",
                    "level": "expert",
                    "confidence_score": 0.95,
                    "years_experience": 8.5,
                    "technologies": ["Python", "FastAPI", "Django", "SQLAlchemy", "Pydantic"],
                    "certifications": ["Python Professional Certification"]
                },
                {
                    "name": "API Design",
                    "description": "RESTful API design and implementation",
                    "category": "backend",
                    "level": "advanced",
                    "confidence_score": 0.90,
                    "years_experience": 6.0,
                    "technologies": ["OpenAPI", "REST", "GraphQL", "gRPC"]
                },
                {
                    "name": "Database Management",
                    "description": "Database design and optimization",
                    "category": "database",
                    "level": "advanced",
                    "confidence_score": 0.85,
                    "years_experience": 7.0,
                    "technologies": ["PostgreSQL", "Redis", "MongoDB", "SQLAlchemy"]
                }
            ],
            "primary_role": "Backend Developer",
            "secondary_roles": ["API Architect", "Database Designer"],
            "workload_preferences": {
                "max_concurrent_tasks": 5,
                "preferred_task_types": ["feature_development", "api_development", "database_design"],
                "working_hours_start": 9,
                "working_hours_end": 17,
                "timezone": "America/New_York"
            },
            "tags": ["python", "backend", "apis", "databases"],
            "team_assignments": ["backend-team", "api-team"]
        }
    },
    
    "task_distribution": {
        "summary": "Distribute API development task",
        "description": "Distribute a high-priority API development task",
        "value": {
            "title": "Implement User Authentication API",
            "description": "Design and implement comprehensive user authentication API with JWT tokens, password reset, and multi-factor authentication support",
            "task_type": "feature_development",
            "priority": "high",
            "requirements": {
                "required_capabilities": ["Python Development", "API Design", "Security Implementation"],
                "preferred_capabilities": ["OAuth2", "JWT", "Multi-factor Authentication"],
                "complexity_score": 0.8,
                "estimated_effort_hours": 16.0,
                "technologies_required": ["FastAPI", "SQLAlchemy", "JWT", "OAuth2"],
                "minimum_experience_years": 3.0
            },
            "constraints": {
                "deadline": "2025-02-05T17:00:00Z",
                "max_duration_days": 5,
                "requires_code_review": true,
                "allow_collaboration": true,
                "max_collaborators": 2
            },
            "assignment_strategy": "optimal_match",
            "project_context": "E-commerce platform authentication system upgrade",
            "labels": ["authentication", "security", "api", "high-priority"]
        }
    },
    
    "task_reassignment": {
        "summary": "Reassign task due to agent unavailability",
        "description": "Reassign task to different agent with explanation",
        "value": {
            "task_id": "550e8400-e29b-41d4-a716-446655440001",
            "target_agent_id": "550e8400-e29b-41d4-a716-446655440003",
            "reason": "Original agent became unavailable due to higher priority critical issue. Reassigning to agent with similar expertise and current availability.",
            "force_assignment": false,
            "preserve_context": true
        }
    }
}

# Security Scheme Definitions
SECURITY_SCHEMES = {
    "apiKey": {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "API key for authentication. Contact support to obtain an API key."
    },
    "bearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "JWT token for authenticated requests. Include 'Bearer ' prefix."
    }
}

# Common HTTP Headers
COMMON_HEADERS = {
    "X-Request-ID": {
        "description": "Unique request identifier for tracing and debugging",
        "schema": {"type": "string", "format": "uuid"}
    },
    "X-Rate-Limit-Remaining": {
        "description": "Number of requests remaining in current rate limit window",
        "schema": {"type": "integer"}
    },
    "X-Rate-Limit-Reset": {
        "description": "Time when rate limit window resets (ISO 8601)",
        "schema": {"type": "string", "format": "date-time"}
    }
}

# OpenAPI Extensions
OPENAPI_EXTENSIONS = {
    "x-logo": {
        "url": "https://leanvibe.dev/logo.png",
        "altText": "LeanVibe Agent Hive"
    },
    "x-code-samples": [
        {
            "lang": "Python",
            "source": """
import httpx
import asyncio

async def register_agent():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.leanvibe.dev/v1/team-coordination/agents/register",
            headers={"X-API-Key": "your-api-key"},
            json={
                "agent_name": "Claude Backend Developer",
                "capabilities": [
                    {
                        "name": "Python Development",
                        "description": "Expert Python programming",
                        "category": "backend",
                        "level": "expert",
                        "confidence_score": 0.95
                    }
                ]
            }
        )
        return response.json()

# Run the example
result = asyncio.run(register_agent())
print(result)
            """
        },
        {
            "lang": "curl",
            "source": """
curl -X POST "https://api.leanvibe.dev/v1/team-coordination/agents/register" \\
  -H "X-API-Key: your-api-key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "agent_name": "Claude Backend Developer",
    "capabilities": [
      {
        "name": "Python Development",
        "description": "Expert Python programming",
        "category": "backend", 
        "level": "expert",
        "confidence_score": 0.95
      }
    ]
  }'
            """
        },
        {
            "lang": "JavaScript",
            "source": """
const axios = require('axios');

async function registerAgent() {
  try {
    const response = await axios.post(
      'https://api.leanvibe.dev/v1/team-coordination/agents/register',
      {
        agent_name: 'Claude Backend Developer',
        capabilities: [
          {
            name: 'Python Development',
            description: 'Expert Python programming',
            category: 'backend',
            level: 'expert',
            confidence_score: 0.95
          }
        ]
      },
      {
        headers: {
          'X-API-Key': 'your-api-key',
          'Content-Type': 'application/json'
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error:', error.response.data);
  }
}

registerAgent().then(result => console.log(result));
            """
        }
    ]
}

# Complete OpenAPI specification additions
OPENAPI_CONFIG = {
    "title": "LeanVibe Agent Hive - Team Coordination API",
    "description": """
## Enterprise Multi-Agent Coordination System

The Team Coordination API enables intelligent management and orchestration of AI agents for collaborative software development. Built for enterprise-scale operations with comprehensive monitoring, real-time coordination, and intelligent task distribution.

### Key Features

- **Intelligent Agent Registration**: Register agents with detailed capability profiles and automatic skill matching
- **Smart Task Distribution**: AI-powered task assignment based on agent capabilities, workload, and performance
- **Real-time Coordination**: WebSocket-based live updates for agent status and task progress
- **Performance Analytics**: Comprehensive metrics and bottleneck detection for system optimization
- **Enterprise Security**: Rate limiting, circuit breakers, and comprehensive error handling

### Getting Started

1. **Obtain API Key**: Contact support for API access credentials
2. **Register Agents**: Register your AI agents with their capabilities
3. **Distribute Tasks**: Submit tasks for intelligent assignment
4. **Monitor Performance**: Use metrics endpoints to optimize system performance

### Rate Limits

- Agent Registration: 10 requests/minute
- Task Distribution: 100 requests/minute  
- Metrics: 30 requests/minute
- Default: 60 requests/minute

### WebSocket Integration

Connect to real-time updates via WebSocket at `/team-coordination/ws/{connection_id}`:

```javascript
const ws = new WebSocket('wss://api.leanvibe.dev/v1/team-coordination/ws/my-connection');
ws.onmessage = (event) => {
    const update = JSON.parse(event.data);
    console.log('Coordination update:', update);
};
```

### Error Handling

All errors follow RFC 7807 Problem Details format with structured error codes, detailed messages, and help URLs for resolution guidance.

### Support

- **Documentation**: https://docs.leanvibe.dev/coordination
- **Support**: support@leanvibe.dev  
- **Status Page**: https://status.leanvibe.dev
    """,
    "version": "2.0.0",
    "contact": {
        "name": "LeanVibe Support",
        "url": "https://leanvibe.dev/support",
        "email": "support@leanvibe.dev"
    },
    "license": {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    "servers": [
        {
            "url": "https://api.leanvibe.dev/v1",
            "description": "Production server"
        },
        {
            "url": "https://staging-api.leanvibe.dev/v1", 
            "description": "Staging server"
        },
        {
            "url": "http://localhost:8000/api/v1",
            "description": "Development server"
        }
    ],
    "tags": TAGS_METADATA,
    "security": [
        {"apiKey": []},
        {"bearerAuth": []}
    ],
    "components": {
        "securitySchemes": SECURITY_SCHEMES,
        "examples": {**RESPONSE_EXAMPLES, **ERROR_EXAMPLES, **REQUEST_EXAMPLES},
        "headers": COMMON_HEADERS
    },
    "externalDocs": {
        "description": "Complete API Documentation",
        "url": "https://docs.leanvibe.dev/api/coordination"
    },
    "x-logo": OPENAPI_EXTENSIONS["x-logo"]
}

# Endpoint-specific documentation
ENDPOINT_DOCS = {
    "register_agent": {
        "summary": "Register Agent for Coordination",
        "description": """
Register an AI agent in the coordination system with detailed capability profiles.

This endpoint enables:
- Comprehensive capability registration with confidence scores
- Workload preference configuration  
- Team assignment and role specification
- Automatic integration with task distribution system

The agent will immediately become available for task assignment based on its registered capabilities.
        """,
        "response_description": "Agent successfully registered with coordination details",
        "responses": {
            201: {
                "description": "Agent registered successfully",
                "content": {
                    "application/json": {
                        "example": RESPONSE_EXAMPLES["agent_registration_success"]["value"]
                    }
                }
            },
            400: {
                "description": "Invalid request data",
                "content": {
                    "application/json": {
                        "example": ERROR_EXAMPLES["validation_error"]["value"]
                    }
                }
            }
        }
    },
    
    "distribute_task": {
        "summary": "Intelligently Distribute Task",
        "description": """
Submit a task for intelligent distribution to the optimal agent.

The system uses advanced matching algorithms considering:
- Capability requirements vs agent skills
- Current agent workload and availability
- Historical performance and success rates
- Task complexity and priority levels

Returns detailed assignment analysis and estimated completion time.
        """,
        "response_description": "Task successfully assigned to optimal agent",
        "responses": {
            201: {
                "description": "Task distributed successfully", 
                "content": {
                    "application/json": {
                        "example": RESPONSE_EXAMPLES["task_distribution_success"]["value"]
                    }
                }
            },
            503: {
                "description": "No suitable agents available",
                "content": {
                    "application/json": {
                        "example": ERROR_EXAMPLES["insufficient_capacity"]["value"]
                    }
                }
            }
        }
    },
    
    "get_metrics": {
        "summary": "Get System Coordination Metrics",
        "description": """
Retrieve comprehensive system performance metrics and analytics.

Provides insights into:
- Agent utilization and performance trends
- Task completion rates and bottlenecks  
- System throughput and efficiency metrics
- Scaling recommendations and optimization opportunities

Use for monitoring system health and making data-driven scaling decisions.
        """,
        "response_description": "Current system coordination metrics",
        "responses": {
            200: {
                "description": "System metrics retrieved successfully",
                "content": {
                    "application/json": {
                        "example": RESPONSE_EXAMPLES["system_metrics"]["value"]
                    }
                }
            }
        }
    }
}

def get_openapi_config() -> Dict[str, Any]:
    """Get complete OpenAPI configuration."""
    return OPENAPI_CONFIG

def get_endpoint_docs(endpoint_name: str) -> Dict[str, Any]:
    """Get documentation for specific endpoint."""
    return ENDPOINT_DOCS.get(endpoint_name, {})