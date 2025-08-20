import asyncio
"""
Minimal FastAPI Application Entry Point
LeanVibe Agent Hive 2.0 API Server

This is the core API server entry point. Keep it minimal and functional.
"""
import os
import logging
from datetime import datetime
from typing import Dict, List, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="LeanVibe Agent Hive 2.0",
    description="Multi-Agent Orchestration System",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "LeanVibe Agent Hive 2.0",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "agents": "/api/v1/agents",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "service": "leanvibe-agent-hive"
    }

@app.get("/metrics")
async def metrics():
    """Basic system metrics"""
    return {
        "system": {
            "status": "operational",
            "uptime": "unknown",  # TODO: track actual uptime
            "agents": {
                "active": 0,  # TODO: connect to orchestrator
                "total": 0
            },
            "tasks": {
                "completed": 0,  # TODO: connect to task system
                "failed": 0,
                "in_progress": 0
            }
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# In-memory storage for basic agent management (temporary)
AGENTS: Dict[str, Dict[str, Any]] = {}
TASKS: Dict[str, Dict[str, Any]] = {}

@app.post("/api/v1/agents")
async def create_agent(agent_data: dict):
    """Create a new agent (in-memory for now)"""
    import uuid
    
    agent_id = str(uuid.uuid4())
    agent = {
        "id": agent_id,
        "name": agent_data.get("name", f"agent-{agent_id[:8]}"),
        "type": agent_data.get("type", "general-purpose"),
        "capabilities": agent_data.get("capabilities", []),
        "status": "active",
        "created_at": datetime.utcnow().isoformat(),
        "last_seen": datetime.utcnow().isoformat()
    }
    
    AGENTS[agent_id] = agent
    logger.info(f"Created agent: {agent['name']} ({agent_id})")
    
    return agent

@app.get("/api/v1/agents")
async def list_agents():
    """List all agents"""
    return {
        "agents": list(AGENTS.values()),
        "count": len(AGENTS),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get specific agent by ID"""
    if agent_id not in AGENTS:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return AGENTS[agent_id]

@app.delete("/api/v1/agents/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent"""
    if agent_id not in AGENTS:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = AGENTS.pop(agent_id)
    logger.info(f"Deleted agent: {agent['name']} ({agent_id})")
    
    return {"message": f"Agent {agent['name']} deleted successfully"}

@app.post("/api/v1/tasks")
async def create_task(task_data: dict):
    """Create a new task"""
    import uuid
    
    task_id = str(uuid.uuid4())
    task = {
        "id": task_id,
        "title": task_data.get("title", f"task-{task_id[:8]}"),
        "description": task_data.get("description", ""),
        "agent_id": task_data.get("agent_id"),
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    TASKS[task_id] = task
    logger.info(f"Created task: {task['title']} ({task_id})")
    
    return task

@app.get("/api/v1/tasks")
async def list_tasks():
    """List all tasks"""
    return {
        "tasks": list(TASKS.values()),
        "count": len(TASKS),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/tasks/{task_id}")
async def get_task(task_id: str):
    """Get specific task by ID"""
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TASKS[task_id]

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class MainScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            import uvicorn

            # Use port from environment or default to 8100 (non-standard to avoid conflicts)
            port = int(os.getenv("PORT", 8100))
            host = os.getenv("HOST", "0.0.0.0")

            logger.info(f"Starting LeanVibe Agent Hive 2.0 API server on {host}:{port}")
            uvicorn.run(app, host=host, port=port, reload=True)
            
            return {"status": "completed"}
    
    script_main(MainScript)