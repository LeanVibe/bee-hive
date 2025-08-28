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
            "tasks": "/api/v1/tasks",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive system health check endpoint"""
    health_status = {"healthy": True, "checks": {}}
    
    # Test Database connectivity
    try:
        from ..core.database import get_session
        async with get_session() as session:
            # Simple query to test database connectivity
            result = await session.execute("SELECT 1")
            health_status["checks"]["database"] = {
                "status": "healthy",
                "details": "PostgreSQL connection successful"
            }
    except Exception as e:
        health_status["healthy"] = False
        health_status["checks"]["database"] = {
            "status": "unhealthy", 
            "error": str(e),
            "details": "PostgreSQL connection failed"
        }
    
    # Test Redis connectivity
    try:
        from ..core.redis import get_redis_client
        redis_client = await get_redis_client()
        if redis_client:
            # Test basic Redis operation
            await redis_client.ping()
            health_status["checks"]["redis"] = {
                "status": "healthy",
                "details": "Redis connection successful"
            }
        else:
            raise Exception("Redis client not available")
    except Exception as e:
        health_status["healthy"] = False
        health_status["checks"]["redis"] = {
            "status": "unhealthy",
            "error": str(e), 
            "details": "Redis connection failed"
        }
    
    # System status summary
    overall_status = "healthy" if health_status["healthy"] else "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "service": "leanvibe-agent-hive",
        "components": health_status["checks"],
        "summary": {
            "healthy_components": len([c for c in health_status["checks"].values() if c["status"] == "healthy"]),
            "total_components": len(health_status["checks"]),
            "overall_healthy": health_status["healthy"]
        }
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

# Core API endpoints now handled by dedicated route files
# See app/api/routes/agents.py and app/api/routes/tasks.py

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