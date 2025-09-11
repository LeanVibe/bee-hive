#!/usr/bin/env python3
"""
Simple FastAPI server for frontend integration
Provides essential API endpoints that the frontend expects without heavy dependencies.
"""

import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

# Create FastAPI app
app = FastAPI(
    title="LeanVibe Frontend API",
    description="Simple API server for frontend integration",
    version="1.0.0"
)

# Configure CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3000", "http://127.0.0.1:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for demo purposes
agents_store = {}
tasks_store = {}
websocket_connections = []

# Pydantic models
class Agent(BaseModel):
    id: str
    name: str
    type: str = "claude"
    status: str = "active"
    role: Optional[str] = None
    capabilities: List[str] = []
    created_at: str
    updated_at: str

class Task(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    status: str = "pending"
    priority: str = "medium"
    agent_id: Optional[str] = None
    created_at: str
    updated_at: str

class CreateAgentRequest(BaseModel):
    name: str
    type: Optional[str] = "claude"
    role: Optional[str] = None
    capabilities: Optional[List[str]] = []

class CreateTaskRequest(BaseModel):
    title: str
    description: Optional[str] = None
    priority: Optional[str] = "medium"
    agent_id: Optional[str] = None

# WebSocket manager
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                # Remove dead connections
                self.disconnect(connection)

websocket_manager = WebSocketManager()

# Utility functions
def get_current_timestamp():
    return datetime.utcnow().isoformat() + "Z"

def create_agent_id():
    return f"agent-{str(uuid.uuid4())[:8]}"

def create_task_id():
    return f"task-{str(uuid.uuid4())[:8]}"

# Root endpoints
@app.get("/")
async def root():
    return {
        "message": "LeanVibe Frontend API Server",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "api_v1": "/api/v1",
            "websocket": "/ws/updates"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": get_current_timestamp(),
        "version": "1.0.0"
    }

@app.get("/status")
async def system_status():
    return {
        "status": "running",
        "timestamp": get_current_timestamp(),
        "components": {
            "api": "online",
            "websocket": "online",
            "agents": len(agents_store),
            "tasks": len(tasks_store)
        },
        "uptime": "active"
    }

# API v1 endpoints - System
@app.get("/api/v1/system/status")
async def get_system_status():
    return {
        "status": "healthy",
        "message": "LeanVibe Frontend API is running",
        "version": "1.0.0",
        "timestamp": get_current_timestamp(),
        "components": {
            "database": "online",
            "redis": "online", 
            "orchestrator": "online"
        }
    }

# API v1 endpoints - Agents
@app.get("/api/v1/agents")
async def list_agents():
    agents_list = list(agents_store.values())
    return {
        "agents": agents_list,
        "total": len(agents_list),
        "offset": 0,
        "limit": 50
    }

@app.post("/api/v1/agents")
async def create_agent(request: CreateAgentRequest):
    agent_id = create_agent_id()
    timestamp = get_current_timestamp()
    
    agent = Agent(
        id=agent_id,
        name=request.name,
        type=request.type or "claude",
        status="active",
        role=request.role,
        capabilities=request.capabilities or [],
        created_at=timestamp,
        updated_at=timestamp
    )
    
    agents_store[agent_id] = agent.dict()
    
    # Broadcast update
    await websocket_manager.broadcast({
        "type": "agent_created",
        "data": agent.dict(),
        "timestamp": timestamp
    })
    
    return agent.dict()

@app.get("/api/v1/agents/{agent_id}")
async def get_agent(agent_id: str):
    if agent_id not in agents_store:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agents_store[agent_id]

@app.put("/api/v1/agents/{agent_id}")
async def update_agent(agent_id: str, updates: Dict[str, Any]):
    if agent_id not in agents_store:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents_store[agent_id]
    agent.update(updates)
    agent["updated_at"] = get_current_timestamp()
    
    # Broadcast update
    await websocket_manager.broadcast({
        "type": "agent_updated",
        "data": agent,
        "timestamp": get_current_timestamp()
    })
    
    return agent

@app.delete("/api/v1/agents/{agent_id}")
async def delete_agent(agent_id: str):
    if agent_id not in agents_store:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents_store.pop(agent_id)
    
    # Broadcast update
    await websocket_manager.broadcast({
        "type": "agent_deleted",
        "data": {"id": agent_id},
        "timestamp": get_current_timestamp()
    })
    
    return {"message": "Agent deleted successfully"}

# API v1 endpoints - Tasks
@app.get("/api/v1/tasks")
async def list_tasks():
    tasks_list = list(tasks_store.values())
    return {
        "tasks": tasks_list,
        "total": len(tasks_list),
        "offset": 0,
        "limit": 50
    }

@app.post("/api/v1/tasks")
async def create_task(request: CreateTaskRequest):
    task_id = create_task_id()
    timestamp = get_current_timestamp()
    
    task = Task(
        id=task_id,
        title=request.title,
        description=request.description,
        status="pending",
        priority=request.priority or "medium",
        agent_id=request.agent_id,
        created_at=timestamp,
        updated_at=timestamp
    )
    
    tasks_store[task_id] = task.dict()
    
    # Broadcast update
    await websocket_manager.broadcast({
        "type": "task_created",
        "data": task.dict(),
        "timestamp": timestamp
    })
    
    return task.dict()

@app.get("/api/v1/tasks/{task_id}")
async def get_task(task_id: str):
    if task_id not in tasks_store:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks_store[task_id]

@app.put("/api/v1/tasks/{task_id}")
async def update_task(task_id: str, updates: Dict[str, Any]):
    if task_id not in tasks_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks_store[task_id]
    task.update(updates)
    task["updated_at"] = get_current_timestamp()
    
    # Broadcast update
    await websocket_manager.broadcast({
        "type": "task_updated",
        "data": task,
        "timestamp": get_current_timestamp()
    })
    
    return task

@app.delete("/api/v1/tasks/{task_id}")
async def delete_task(task_id: str):
    if task_id not in tasks_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks_store.pop(task_id)
    
    # Broadcast update
    await websocket_manager.broadcast({
        "type": "task_deleted",
        "data": {"id": task_id},
        "timestamp": get_current_timestamp()
    })
    
    return {"message": "Task deleted successfully"}

# WebSocket endpoint for real-time updates
@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "message": "Connected to LeanVibe real-time updates",
            "timestamp": get_current_timestamp()
        }))
        
        while True:
            # Keep connection alive and handle incoming messages
            try:
                data = await websocket.receive_text()
                # Echo back any received messages (for testing)
                await websocket.send_text(json.dumps({
                    "type": "echo",
                    "data": data,
                    "timestamp": get_current_timestamp()
                }))
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

# Additional endpoints that frontend might expect
@app.get("/observability/metrics")
async def get_metrics():
    return {
        "agents_total": len(agents_store),
        "tasks_total": len(tasks_store),
        "websocket_connections": len(websocket_manager.active_connections),
        "timestamp": get_current_timestamp()
    }

@app.get("/observability/health")
async def get_observability_health():
    return {
        "status": "healthy",
        "components": {
            "api": "online",
            "websocket": "online",
            "storage": "online"
        },
        "timestamp": get_current_timestamp()
    }

# Development utilities
@app.post("/dev/populate")
async def populate_demo_data():
    """Populate with demo data for frontend testing"""
    
    # Create demo agents
    demo_agents = [
        {"name": "Claude Assistant", "type": "claude", "role": "assistant", "capabilities": ["coding", "analysis"]},
        {"name": "Task Manager", "type": "system", "role": "manager", "capabilities": ["scheduling", "coordination"]},
        {"name": "Code Reviewer", "type": "claude", "role": "reviewer", "capabilities": ["code-review", "testing"]}
    ]
    
    created_agents = []
    for agent_data in demo_agents:
        agent_id = create_agent_id()
        timestamp = get_current_timestamp()
        
        agent = Agent(
            id=agent_id,
            name=agent_data["name"],
            type=agent_data["type"],
            status="active",
            role=agent_data["role"],
            capabilities=agent_data["capabilities"],
            created_at=timestamp,
            updated_at=timestamp
        )
        
        agents_store[agent_id] = agent.dict()
        created_agents.append(agent.dict())
    
    # Create demo tasks
    demo_tasks = [
        {"title": "Review API endpoints", "description": "Review and validate API endpoint implementations", "priority": "high"},
        {"title": "Update frontend components", "description": "Update Vue.js components for new API integration", "priority": "medium"},
        {"title": "Write integration tests", "description": "Create comprehensive tests for frontend-backend integration", "priority": "medium"}
    ]
    
    created_tasks = []
    for task_data in demo_tasks:
        task_id = create_task_id()
        timestamp = get_current_timestamp()
        
        task = Task(
            id=task_id,
            title=task_data["title"],
            description=task_data["description"],
            status="pending",
            priority=task_data["priority"],
            agent_id=created_agents[0]["id"] if created_agents else None,
            created_at=timestamp,
            updated_at=timestamp
        )
        
        tasks_store[task_id] = task.dict()
        created_tasks.append(task.dict())
    
    return {
        "message": "Demo data populated successfully",
        "agents_created": len(created_agents),
        "tasks_created": len(created_tasks),
        "agents": created_agents,
        "tasks": created_tasks
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting LeanVibe Frontend API Server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("📋 API Documentation: http://localhost:8000/docs")
    print("🔌 WebSocket endpoint: ws://localhost:8000/ws/updates")
    
    uvicorn.run(
        "frontend_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )