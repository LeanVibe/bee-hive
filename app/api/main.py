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
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import core initialization modules
try:
    from ..core.redis import init_redis, close_redis
    from ..core.database import init_database, close_database
except ImportError:
    # Fallback imports for when app.core is not in PYTHONPATH
    from app.core.redis import init_redis, close_redis
    from app.core.database import init_database, close_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    logger.info("Starting LeanVibe Agent Hive 2.0 API Server...")
    try:
        await init_redis()
        logger.info("✅ Redis initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️  Redis initialization failed: {e}")
    
    try:
        await init_database()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️  Database initialization failed: {e}")
    
    logger.info("🚀 API Server ready to accept connections")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LeanVibe Agent Hive 2.0 API Server...")
    try:
        await close_redis()
        logger.info("✅ Redis connections closed")
    except Exception as e:
        logger.warning(f"⚠️  Redis shutdown error: {e}")
    
    try:
        await close_database()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.warning(f"⚠️  Database shutdown error: {e}")
    
    logger.info("👋 API Server shutdown complete")


# Create FastAPI application with lifespan management
app = FastAPI(
    title="LeanVibe Agent Hive 2.0",
    description="Multi-Agent Orchestration System",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
try:
    from .routes import router as api_router
    app.include_router(api_router)
    logger.info("✅ API routes loaded successfully")
except Exception as e:
    logger.warning(f"⚠️  Some API routes may not be available: {e}")

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
            from uvicorn import Config, Server

            # Use port from environment or from settings to ensure consistency
            try:
                from ..core.config import settings
                port = int(os.getenv("PORT", settings.API_PORT))
            except Exception:
                # Fallback to default if config not available
                port = int(os.getenv("PORT", 18080))
            host = os.getenv("HOST", "0.0.0.0")

            logger.info(f"Starting LeanVibe Agent Hive 2.0 API server on {host}:{port}")
            
            # Create server config and run within existing event loop
            config = Config(app=app, host=host, port=port, log_level="info")
            server = Server(config)
            await server.serve()
            
            return {"status": "completed"}
    
    script_main(MainScript)