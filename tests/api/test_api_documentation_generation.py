"""
API Documentation Generation from Test Specifications
====================================================

Generates comprehensive API documentation based on contract tests and 
endpoint specifications. Creates OpenAPI/Swagger documentation that
reflects actual tested behavior and contracts.

Key Documentation Areas:
- OpenAPI 3.0 specification generation
- Endpoint documentation with examples
- Schema documentation from contracts
- Error response documentation
- Performance characteristics documentation
- Integration examples and workflows
"""

import pytest
import json
import yaml
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import httpx
import uvicorn
import threading
from collections import defaultdict


class APIDocumentationGenerator:
    """Generates API documentation from test specifications and live endpoints."""
    
    def __init__(self):
        self.api_spec = {
            "openapi": "3.0.3",
            "info": {
                "title": "LeanVibe Frontend API",
                "description": "Backend API for LeanVibe Agent Hive frontend integration",
                "version": "1.0.0",
                "contact": {
                    "name": "LeanVibe Team",
                    "url": "https://github.com/leanvibe/bee-hive"
                }
            },
            "servers": [
                {
                    "url": "http://localhost:8000",
                    "description": "Development server"
                },
                {
                    "url": "https://api.leanvibe.com",
                    "description": "Production server"
                }
            ],
            "paths": {},
            "components": {
                "schemas": {},
                "responses": {},
                "examples": {}
            }
        }
        self.performance_data = {}
        
    def add_endpoint(self, path: str, method: str, spec: Dict[str, Any]):
        """Add endpoint specification to the documentation."""
        if path not in self.api_spec["paths"]:
            self.api_spec["paths"][path] = {}
        
        self.api_spec["paths"][path][method.lower()] = spec
    
    def add_schema(self, name: str, schema: Dict[str, Any]):
        """Add schema definition to the documentation."""
        self.api_spec["components"]["schemas"][name] = schema
    
    def add_performance_data(self, endpoint: str, method: str, performance: Dict[str, Any]):
        """Add performance data for endpoint."""
        key = f"{method.upper()} {endpoint}"
        self.performance_data[key] = performance
    
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate complete OpenAPI specification."""
        return self.api_spec
    
    def save_documentation(self, output_path: Path):
        """Save documentation to files."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save OpenAPI JSON
        with open(output_path / "openapi.json", "w") as f:
            json.dump(self.api_spec, f, indent=2)
        
        # Save OpenAPI YAML
        with open(output_path / "openapi.yaml", "w") as f:
            yaml.dump(self.api_spec, f, default_flow_style=False)
        
        # Save performance data
        with open(output_path / "performance_characteristics.json", "w") as f:
            json.dump(self.performance_data, f, indent=2)


class TestAPIDocumentationGeneration:
    """Test API documentation generation from live endpoints and contracts."""
    
    @pytest.fixture(scope="class")
    def api_server(self):
        """Start API server for documentation generation."""
        def run_server():
            uvicorn.run(
                "frontend_api_server:app",
                host="127.0.0.1",
                port=8988,
                log_level="error"
            )
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        time.sleep(2)
        
        yield "http://127.0.0.1:8988"
    
    @pytest.fixture
    async def http_client(self, api_server):
        """Create HTTP client for documentation testing."""
        async with httpx.AsyncClient(base_url=api_server, timeout=30.0) as client:
            yield client
    
    @pytest.fixture
    def doc_generator(self):
        """Create documentation generator."""
        return APIDocumentationGenerator()
    
    async def test_generate_root_endpoint_documentation(self, http_client, doc_generator):
        """Generate documentation for root endpoint."""
        
        # Test the endpoint and measure performance
        start_time = time.time()
        response = await http_client.get("/")
        response_time_ms = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Add schema for root response
        doc_generator.add_schema("RootResponse", {
            "type": "object",
            "required": ["message", "version", "endpoints"],
            "properties": {
                "message": {"type": "string", "example": "LeanVibe Frontend API Server"},
                "version": {"type": "string", "example": "1.0.0"},
                "endpoints": {
                    "type": "object",
                    "properties": {
                        "health": {"type": "string", "example": "/health"},
                        "status": {"type": "string", "example": "/status"},
                        "api_v1": {"type": "string", "example": "/api/v1"},
                        "websocket": {"type": "string", "example": "/ws/updates"}
                    }
                }
            }
        })
        
        # Add endpoint documentation
        doc_generator.add_endpoint("/", "get", {
            "summary": "API Root Information",
            "description": "Returns basic API information including version and available endpoints",
            "tags": ["System"],
            "responses": {
                "200": {
                    "description": "API information successfully retrieved",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/RootResponse"},
                            "example": response_data
                        }
                    }
                }
            }
        })
        
        # Add performance data
        doc_generator.add_performance_data("/", "GET", {
            "average_response_time_ms": response_time_ms,
            "target_response_time_ms": 100.0,
            "performance_tier": "fast"
        })
    
    async def test_generate_health_endpoint_documentation(self, http_client, doc_generator):
        """Generate documentation for health endpoint."""
        
        start_time = time.time()
        response = await http_client.get("/health")
        response_time_ms = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Add schema
        doc_generator.add_schema("HealthResponse", {
            "type": "object",
            "required": ["status", "timestamp", "version"],
            "properties": {
                "status": {"type": "string", "enum": ["healthy"], "example": "healthy"},
                "timestamp": {"type": "string", "format": "date-time", "example": "2025-01-18T12:00:00Z"},
                "version": {"type": "string", "example": "1.0.0"}
            }
        })
        
        # Add endpoint documentation
        doc_generator.add_endpoint("/health", "get", {
            "summary": "Health Check",
            "description": "Returns the health status of the API server",
            "tags": ["Health"],
            "responses": {
                "200": {
                    "description": "Server is healthy",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/HealthResponse"},
                            "example": response_data
                        }
                    }
                }
            }
        })
        
        doc_generator.add_performance_data("/health", "GET", {
            "average_response_time_ms": response_time_ms,
            "target_response_time_ms": 50.0,
            "performance_tier": "fastest"
        })
    
    async def test_generate_agent_endpoints_documentation(self, http_client, doc_generator):
        """Generate documentation for agent management endpoints."""
        
        # Define schemas
        doc_generator.add_schema("CreateAgentRequest", {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {"type": "string", "minLength": 1, "maxLength": 100, "example": "Backend Developer Agent"},
                "type": {"type": "string", "enum": ["claude", "system", "custom"], "default": "claude", "example": "claude"},
                "role": {"type": "string", "maxLength": 50, "example": "backend_developer"},
                "capabilities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 20,
                    "example": ["coding", "testing", "debugging"]
                }
            }
        })
        
        doc_generator.add_schema("Agent", {
            "type": "object",
            "required": ["id", "name", "type", "status", "created_at", "updated_at"],
            "properties": {
                "id": {"type": "string", "pattern": "^agent-[a-f0-9]{8}$", "example": "agent-12345678"},
                "name": {"type": "string", "example": "Backend Developer Agent"},
                "type": {"type": "string", "enum": ["claude", "system", "custom"], "example": "claude"},
                "status": {"type": "string", "enum": ["active", "inactive", "error"], "example": "active"},
                "role": {"type": "string", "example": "backend_developer"},
                "capabilities": {"type": "array", "items": {"type": "string"}, "example": ["coding", "testing"]},
                "created_at": {"type": "string", "format": "date-time", "example": "2025-01-18T12:00:00Z"},
                "updated_at": {"type": "string", "format": "date-time", "example": "2025-01-18T12:00:00Z"}
            }
        })
        
        doc_generator.add_schema("AgentList", {
            "type": "object",
            "required": ["agents", "total", "offset", "limit"],
            "properties": {
                "agents": {"type": "array", "items": {"$ref": "#/components/schemas/Agent"}},
                "total": {"type": "integer", "minimum": 0, "example": 10},
                "offset": {"type": "integer", "minimum": 0, "example": 0},
                "limit": {"type": "integer", "minimum": 1, "example": 50}
            }
        })
        
        # Test and document GET /api/v1/agents
        start_time = time.time()
        list_response = await http_client.get("/api/v1/agents")
        list_time_ms = (time.time() - start_time) * 1000
        
        assert list_response.status_code == 200
        list_data = list_response.json()
        
        doc_generator.add_endpoint("/api/v1/agents", "get", {
            "summary": "List Agents",
            "description": "Retrieve a list of all agents with pagination support",
            "tags": ["Agents"],
            "parameters": [
                {
                    "name": "offset",
                    "in": "query",
                    "description": "Number of agents to skip",
                    "required": False,
                    "schema": {"type": "integer", "minimum": 0, "default": 0}
                },
                {
                    "name": "limit",
                    "in": "query", 
                    "description": "Maximum number of agents to return",
                    "required": False,
                    "schema": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50}
                }
            ],
            "responses": {
                "200": {
                    "description": "List of agents successfully retrieved",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/AgentList"},
                            "example": list_data
                        }
                    }
                }
            }
        })
        
        # Test and document POST /api/v1/agents
        create_payload = {
            "name": "Documentation Test Agent",
            "type": "claude",
            "role": "backend_developer",
            "capabilities": ["coding", "testing", "documentation"]
        }
        
        start_time = time.time()
        create_response = await http_client.post("/api/v1/agents", json=create_payload)
        create_time_ms = (time.time() - start_time) * 1000
        
        assert create_response.status_code in [200, 201]
        agent_data = create_response.json()
        agent_id = agent_data["id"]
        
        doc_generator.add_endpoint("/api/v1/agents", "post", {
            "summary": "Create Agent",
            "description": "Create a new agent with specified configuration",
            "tags": ["Agents"],
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/CreateAgentRequest"},
                        "example": create_payload
                    }
                }
            },
            "responses": {
                "200": {
                    "description": "Agent created successfully",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Agent"},
                            "example": agent_data
                        }
                    }
                },
                "422": {
                    "description": "Validation error",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ValidationError"}
                        }
                    }
                }
            }
        })
        
        # Test and document GET /api/v1/agents/{agent_id}
        start_time = time.time()
        get_response = await http_client.get(f"/api/v1/agents/{agent_id}")
        get_time_ms = (time.time() - start_time) * 1000
        
        assert get_response.status_code == 200
        get_data = get_response.json()
        
        doc_generator.add_endpoint("/api/v1/agents/{agent_id}", "get", {
            "summary": "Get Agent",
            "description": "Retrieve a specific agent by ID",
            "tags": ["Agents"],
            "parameters": [
                {
                    "name": "agent_id",
                    "in": "path",
                    "description": "Unique identifier of the agent",
                    "required": True,
                    "schema": {"type": "string", "pattern": "^agent-[a-f0-9]{8}$"}
                }
            ],
            "responses": {
                "200": {
                    "description": "Agent successfully retrieved",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Agent"},
                            "example": get_data
                        }
                    }
                },
                "404": {
                    "description": "Agent not found",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/NotFoundError"}
                        }
                    }
                }
            }
        })
        
        # Add performance data
        doc_generator.add_performance_data("/api/v1/agents", "GET", {
            "average_response_time_ms": list_time_ms,
            "target_response_time_ms": 200.0,
            "performance_tier": "fast"
        })
        
        doc_generator.add_performance_data("/api/v1/agents", "POST", {
            "average_response_time_ms": create_time_ms,
            "target_response_time_ms": 500.0,
            "performance_tier": "normal"
        })
        
        doc_generator.add_performance_data("/api/v1/agents/{agent_id}", "GET", {
            "average_response_time_ms": get_time_ms,
            "target_response_time_ms": 200.0,
            "performance_tier": "fast"
        })
        
        # Cleanup
        await http_client.delete(f"/api/v1/agents/{agent_id}")
    
    async def test_generate_task_endpoints_documentation(self, http_client, doc_generator):
        """Generate documentation for task management endpoints."""
        
        # Define task schemas
        doc_generator.add_schema("CreateTaskRequest", {
            "type": "object",
            "required": ["title"],
            "properties": {
                "title": {"type": "string", "minLength": 1, "maxLength": 200, "example": "Implement user authentication"},
                "description": {"type": "string", "maxLength": 1000, "example": "Add JWT-based authentication to the API"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"], "default": "medium", "example": "high"},
                "agent_id": {"type": "string", "pattern": "^agent-[a-f0-9]{8}$", "example": "agent-12345678"}
            }
        })
        
        doc_generator.add_schema("Task", {
            "type": "object",
            "required": ["id", "title", "status", "priority", "created_at", "updated_at"],
            "properties": {
                "id": {"type": "string", "pattern": "^task-[a-f0-9]{8}$", "example": "task-87654321"},
                "title": {"type": "string", "example": "Implement user authentication"},
                "description": {"type": "string", "example": "Add JWT-based authentication to the API"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "failed"], "example": "pending"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"], "example": "high"},
                "agent_id": {"type": "string", "pattern": "^agent-[a-f0-9]{8}$", "example": "agent-12345678"},
                "created_at": {"type": "string", "format": "date-time", "example": "2025-01-18T12:00:00Z"},
                "updated_at": {"type": "string", "format": "date-time", "example": "2025-01-18T12:00:00Z"}
            }
        })
        
        # Create agent for task testing
        agent_response = await http_client.post("/api/v1/agents", json={
            "name": "Task Documentation Agent",
            "type": "claude"
        })
        assert agent_response.status_code in [200, 201]
        agent_data = agent_response.json()
        agent_id = agent_data["id"]
        
        try:
            # Test task creation
            task_payload = {
                "title": "Documentation Task",
                "description": "Task created for API documentation",
                "priority": "medium",
                "agent_id": agent_id
            }
            
            start_time = time.time()
            create_response = await http_client.post("/api/v1/tasks", json=task_payload)
            create_time_ms = (time.time() - start_time) * 1000
            
            assert create_response.status_code in [200, 201]
            task_data = create_response.json()
            task_id = task_data["id"]
            
            # Document POST /api/v1/tasks
            doc_generator.add_endpoint("/api/v1/tasks", "post", {
                "summary": "Create Task",
                "description": "Create a new task and optionally assign it to an agent",
                "tags": ["Tasks"],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/CreateTaskRequest"},
                            "example": task_payload
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Task created successfully",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/Task"},
                                "example": task_data
                            }
                        }
                    },
                    "422": {
                        "description": "Validation error",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ValidationError"}
                            }
                        }
                    }
                }
            })
            
            # Add performance data
            doc_generator.add_performance_data("/api/v1/tasks", "POST", {
                "average_response_time_ms": create_time_ms,
                "target_response_time_ms": 300.0,
                "performance_tier": "normal"
            })
            
            # Cleanup
            await http_client.delete(f"/api/v1/tasks/{task_id}")
        
        finally:
            await http_client.delete(f"/api/v1/agents/{agent_id}")
    
    async def test_generate_error_response_documentation(self, http_client, doc_generator):
        """Generate documentation for error responses."""
        
        # Define error schemas
        doc_generator.add_schema("NotFoundError", {
            "type": "object",
            "required": ["detail"],
            "properties": {
                "detail": {"type": "string", "example": "Agent not found"}
            }
        })
        
        doc_generator.add_schema("ValidationError", {
            "type": "object",
            "required": ["detail"],
            "properties": {
                "detail": {
                    "oneOf": [
                        {"type": "string", "example": "Validation failed"},
                        {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "loc": {"type": "array", "items": {"type": "string"}},
                                    "msg": {"type": "string"},
                                    "type": {"type": "string"}
                                }
                            }
                        }
                    ]
                }
            }
        })
        
        doc_generator.add_schema("ServerError", {
            "type": "object",
            "required": ["detail"],
            "properties": {
                "detail": {"type": "string", "example": "Internal server error"}
            }
        })
        
        # Test 404 error
        not_found_response = await http_client.get("/api/v1/agents/non-existent-agent")
        assert not_found_response.status_code == 404
        not_found_data = not_found_response.json()
        
        # Test 422 error
        validation_response = await http_client.post("/api/v1/agents", json={})
        assert validation_response.status_code == 422
        validation_data = validation_response.json()
        
        # Add to components/responses
        doc_generator.api_spec["components"]["responses"].update({
            "NotFound": {
                "description": "The specified resource was not found",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/NotFoundError"},
                        "example": not_found_data
                    }
                }
            },
            "ValidationError": {
                "description": "Request validation failed",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ValidationError"},
                        "example": validation_data
                    }
                }
            },
            "ServerError": {
                "description": "Internal server error",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ServerError"}
                    }
                }
            }
        })
    
    async def test_generate_websocket_documentation(self, doc_generator):
        """Generate documentation for WebSocket endpoints."""
        
        # WebSocket is not easily testable with httpx, so we'll add schema documentation
        doc_generator.add_schema("WebSocketMessage", {
            "type": "object",
            "required": ["type", "timestamp"],
            "properties": {
                "type": {"type": "string", "enum": ["connection_established", "echo", "agent_created", "agent_updated", "agent_deleted", "task_created", "task_updated", "task_deleted"]},
                "timestamp": {"type": "string", "format": "date-time"},
                "data": {"type": "object", "description": "Message-specific data"},
                "message": {"type": "string", "description": "Human-readable message"}
            }
        })
        
        # Add WebSocket endpoint documentation (manually since it's not HTTP)
        # This would be added to a separate WebSocket documentation section
        websocket_spec = {
            "summary": "Real-time Updates WebSocket",
            "description": "WebSocket endpoint for receiving real-time updates about agents and tasks",
            "tags": ["WebSocket"],
            "url": "/ws/updates",
            "messages": {
                "connection_established": {
                    "description": "Sent when WebSocket connection is established",
                    "payload": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["connection_established"]},
                            "message": {"type": "string"},
                            "timestamp": {"type": "string", "format": "date-time"}
                        }
                    }
                },
                "agent_created": {
                    "description": "Sent when a new agent is created",
                    "payload": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["agent_created"]},
                            "data": {"$ref": "#/components/schemas/Agent"},
                            "timestamp": {"type": "string", "format": "date-time"}
                        }
                    }
                }
            }
        }
        
        # Store WebSocket documentation separately
        if "websockets" not in doc_generator.api_spec:
            doc_generator.api_spec["websockets"] = {}
        doc_generator.api_spec["websockets"]["/ws/updates"] = websocket_spec
    
    async def test_generate_complete_api_documentation(self, http_client, doc_generator):
        """Generate complete API documentation with examples and workflows."""
        
        # Add API-wide information
        doc_generator.api_spec["info"]["description"] = """
# LeanVibe Frontend API

This API provides backend services for the LeanVibe Agent Hive frontend application.

## Features

- **Agent Management**: Create, read, update, and delete agents
- **Task Management**: Create and manage tasks with agent assignment
- **Real-time Updates**: WebSocket support for live updates
- **Health Monitoring**: Health checks and system status endpoints
- **Performance Optimized**: All endpoints meet strict performance contracts

## Authentication

Currently, the API does not require authentication. This will be added in future versions.

## Rate Limiting

No rate limiting is currently implemented. Production deployments should implement appropriate rate limiting.

## Error Handling

All endpoints follow consistent error response patterns:

- **400/422**: Client errors with validation details
- **404**: Resource not found
- **500**: Server errors with error details

## Performance Characteristics

All endpoints are designed to meet specific performance targets:

- Health checks: < 50ms
- Read operations: < 200ms  
- Write operations: < 500ms
- System status: < 100ms
        """
        
        # Add examples
        doc_generator.api_spec["components"]["examples"] = {
            "AgentExample": {
                "summary": "Example backend developer agent",
                "value": {
                    "id": "agent-12345678",
                    "name": "Backend Developer",
                    "type": "claude",
                    "status": "active",
                    "role": "backend_developer",
                    "capabilities": ["coding", "testing", "debugging"],
                    "created_at": "2025-01-18T12:00:00Z",
                    "updated_at": "2025-01-18T12:00:00Z"
                }
            },
            "TaskExample": {
                "summary": "Example development task",
                "value": {
                    "id": "task-87654321",
                    "title": "Implement user authentication",
                    "description": "Add JWT-based authentication to the API",
                    "status": "pending",
                    "priority": "high",
                    "agent_id": "agent-12345678",
                    "created_at": "2025-01-18T12:00:00Z",
                    "updated_at": "2025-01-18T12:00:00Z"
                }
            }
        }
        
        # Add tags
        doc_generator.api_spec["tags"] = [
            {"name": "System", "description": "System information and health checks"},
            {"name": "Health", "description": "Health monitoring endpoints"},
            {"name": "Agents", "description": "Agent management operations"},
            {"name": "Tasks", "description": "Task management operations"},
            {"name": "WebSocket", "description": "Real-time communication via WebSocket"}
        ]
        
        # Test a complete workflow and document it
        workflow_example = {
            "title": "Complete Agent and Task Workflow",
            "description": "Example of creating an agent, assigning a task, and monitoring progress",
            "steps": [
                {
                    "step": 1,
                    "description": "Create a new agent",
                    "method": "POST",
                    "url": "/api/v1/agents",
                    "payload": {
                        "name": "Backend Developer",
                        "type": "claude",
                        "role": "backend_developer",
                        "capabilities": ["coding", "testing"]
                    }
                },
                {
                    "step": 2,
                    "description": "Create a task and assign to the agent",
                    "method": "POST",
                    "url": "/api/v1/tasks",
                    "payload": {
                        "title": "Implement authentication",
                        "description": "Add JWT authentication",
                        "priority": "high",
                        "agent_id": "agent-12345678"
                    }
                },
                {
                    "step": 3,
                    "description": "Monitor task progress",
                    "method": "GET",
                    "url": "/api/v1/tasks/task-87654321"
                },
                {
                    "step": 4,
                    "description": "Update task status",
                    "method": "PUT",
                    "url": "/api/v1/tasks/task-87654321",
                    "payload": {"status": "completed"}
                }
            ]
        }
        
        # Add workflow to documentation
        if "x-workflows" not in doc_generator.api_spec:
            doc_generator.api_spec["x-workflows"] = []
        doc_generator.api_spec["x-workflows"].append(workflow_example)
    
    async def test_save_generated_documentation(self, doc_generator):
        """Save the generated documentation to files."""
        
        # Ensure all documentation has been generated
        assert len(doc_generator.api_spec["paths"]) > 0, "No endpoints documented"
        assert len(doc_generator.api_spec["components"]["schemas"]) > 0, "No schemas documented"
        
        # Create output directory
        output_path = Path("/Users/bogdan/work/leanvibe-dev/bee-hive/docs/api")
        
        # Save documentation
        doc_generator.save_documentation(output_path)
        
        # Verify files were created
        assert (output_path / "openapi.json").exists(), "OpenAPI JSON not created"
        assert (output_path / "openapi.yaml").exists(), "OpenAPI YAML not created"
        assert (output_path / "performance_characteristics.json").exists(), "Performance data not created"
        
        # Validate JSON structure
        with open(output_path / "openapi.json", "r") as f:
            saved_spec = json.load(f)
        
        # Basic validation
        assert saved_spec["openapi"] == "3.0.3"
        assert saved_spec["info"]["title"] == "LeanVibe Frontend API"
        assert len(saved_spec["paths"]) > 0
        assert len(saved_spec["components"]["schemas"]) > 0
        
        print(f"✅ API documentation generated and saved to {output_path}")
        print(f"📝 Generated {len(saved_spec['paths'])} endpoint documentations")
        print(f"📋 Generated {len(saved_spec['components']['schemas'])} schema definitions")
        print(f"⚡ Performance data for {len(doc_generator.performance_data)} endpoints")


class TestDocumentationValidation:
    """Validate the generated documentation meets requirements."""
    
    def test_openapi_specification_compliance(self):
        """Test that generated OpenAPI spec is valid."""
        
        # Load generated documentation
        docs_path = Path("/Users/bogdan/work/leanvibe-dev/bee-hive/docs/api/openapi.json")
        
        if docs_path.exists():
            with open(docs_path, "r") as f:
                spec = json.load(f)
            
            # Validate required OpenAPI fields
            required_fields = ["openapi", "info", "paths"]
            for field in required_fields:
                assert field in spec, f"OpenAPI spec missing required field: {field}"
            
            # Validate OpenAPI version
            assert spec["openapi"].startswith("3.0"), "Should use OpenAPI 3.0+"
            
            # Validate info section
            info = spec["info"]
            required_info_fields = ["title", "version"]
            for field in required_info_fields:
                assert field in info, f"Info section missing required field: {field}"
            
            # Validate paths
            assert len(spec["paths"]) > 0, "Should have at least one endpoint documented"
            
            # Validate each path has required HTTP methods
            for path, methods in spec["paths"].items():
                assert isinstance(methods, dict), f"Path {path} should have method definitions"
                for method, definition in methods.items():
                    assert "summary" in definition, f"Method {method} on {path} missing summary"
                    assert "responses" in definition, f"Method {method} on {path} missing responses"
    
    def test_performance_documentation_completeness(self):
        """Test that performance documentation is complete."""
        
        perf_path = Path("/Users/bogdan/work/leanvibe-dev/bee-hive/docs/api/performance_characteristics.json")
        
        if perf_path.exists():
            with open(perf_path, "r") as f:
                performance_data = json.load(f)
            
            # Should have performance data for multiple endpoints
            assert len(performance_data) > 0, "Should have performance data for endpoints"
            
            # Validate performance data structure
            for endpoint, data in performance_data.items():
                assert "average_response_time_ms" in data, f"Performance data for {endpoint} missing response time"
                assert "target_response_time_ms" in data, f"Performance data for {endpoint} missing target time"
                assert "performance_tier" in data, f"Performance data for {endpoint} missing performance tier"
                
                # Validate performance tiers
                valid_tiers = ["fastest", "fast", "normal", "slow"]
                assert data["performance_tier"] in valid_tiers, f"Invalid performance tier for {endpoint}"
    
    def test_schema_documentation_completeness(self):
        """Test that schema documentation covers all required models."""
        
        docs_path = Path("/Users/bogdan/work/leanvibe-dev/bee-hive/docs/api/openapi.json")
        
        if docs_path.exists():
            with open(docs_path, "r") as f:
                spec = json.load(f)
            
            schemas = spec.get("components", {}).get("schemas", {})
            
            # Required schemas
            required_schemas = [
                "Agent",
                "CreateAgentRequest", 
                "Task",
                "CreateTaskRequest",
                "NotFoundError",
                "ValidationError"
            ]
            
            for schema_name in required_schemas:
                assert schema_name in schemas, f"Required schema {schema_name} not documented"
                
                schema = schemas[schema_name]
                assert "type" in schema, f"Schema {schema_name} missing type definition"
                assert "properties" in schema, f"Schema {schema_name} missing properties"