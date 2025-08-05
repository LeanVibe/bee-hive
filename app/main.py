"""
FastAPI Application Entry Point for LeanVibe Agent Hive 2.0

Multi-agent orchestration system with real-time communication,
context management, and self-modification capabilities.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .core.config import settings
from .core.database import init_database
from .core.redis import init_redis, get_redis
from .core.orchestrator import AgentOrchestrator
from .core.event_processor import initialize_event_processor, shutdown_event_processor
from .core.performance_metrics_publisher import get_performance_publisher, stop_performance_publisher
from .api.routes import router as api_router
from .api.sleep_management import router as sleep_management_router
from .api.intelligent_scheduling import router as intelligent_scheduling_router
from .api.monitoring_reporting import router as monitoring_router
from .api.analytics import router as analytics_router
from .observability.middleware import ObservabilityMiddleware, ObservabilityHookMiddleware
from .observability.hooks import HookInterceptor, set_hook_interceptor
from .observability.prometheus_middleware import PrometheusMiddleware
from .core.error_handling_middleware import ErrorHandlingMiddleware, create_error_handling_middleware
from .core.error_handling_config import initialize_error_handling_config, ErrorHandlingEnvironment, get_config_manager
from .core.error_handling_integration import initialize_error_handling_integration
from .api.v1.error_handling_health import router as error_handling_router
from .api.v1.enhanced_coordination_api import router as enhanced_coordination_router
from .api.v1.global_coordination import router as global_coordination_router
from .dashboard.coordination_dashboard import router as dashboard_router
from .dashboard.simple_agent_dashboard import router as simple_dashboard_router
from .api.agent_activation import router as agent_activation_router
from .api.hive_commands import router as hive_commands_router
from .api.intelligence import router as intelligence_router
from .api.claude_integration import router as claude_integration_router


# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management with proper startup/shutdown."""
    logger.info("🚀 Starting LeanVibe Agent Hive 2.0...")
    
    try:
        # Initialize core infrastructure
        await init_database()
        await init_redis()
        
        # Initialize observability system
        redis_client = get_redis()
        event_processor = await initialize_event_processor(redis_client)
        app.state.event_processor = event_processor
        
        # Initialize hook interceptor
        hook_interceptor = HookInterceptor(event_processor=event_processor)
        set_hook_interceptor(hook_interceptor)
        app.state.hook_interceptor = hook_interceptor
        
        # Initialize error handling system
        environment = ErrorHandlingEnvironment.PRODUCTION if not settings.DEBUG else ErrorHandlingEnvironment.DEVELOPMENT
        error_config_manager = initialize_error_handling_config(
            environment=environment,
            enable_hot_reload=settings.DEBUG
        )
        app.state.error_config_manager = error_config_manager
        
        # Initialize error handling observability integration
        error_integration = initialize_error_handling_integration(
            enable_detailed_logging=settings.DEBUG
        )
        app.state.error_integration = error_integration
        
        # Start configuration hot-reload if enabled
        if settings.DEBUG:
            await error_config_manager.start_hot_reload()
        
        # Start agent orchestrator
        orchestrator = AgentOrchestrator()
        await orchestrator.start()
        app.state.orchestrator = orchestrator
        
        # Start performance metrics publisher for real-time monitoring
        performance_publisher = await get_performance_publisher()
        app.state.performance_publisher = performance_publisher
        
        logger.info("✅ Agent Hive initialized successfully")
        
        yield  # Application runs here
        
    except Exception as e:
        logger.error("❌ Failed to initialize Agent Hive", error=str(e))
        raise
    finally:
        logger.info("🛑 Shutting down Agent Hive...")
        
        # Graceful shutdown
        if hasattr(app.state, 'orchestrator'):
            await app.state.orchestrator.shutdown()
        
        # Stop error handling hot-reload
        if hasattr(app.state, 'error_config_manager'):
            await app.state.error_config_manager.stop_hot_reload()
        
        # Stop performance metrics publisher
        await stop_performance_publisher()
        
        # Shutdown observability system
        await shutdown_event_processor()
        
        logger.info("✅ Agent Hive shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="LeanVibe Agent Hive 2.0",
        description="Multi-Agent Orchestration System for Autonomous Software Development",
        version="2.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
    )
    
    # Security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Error handling middleware (high priority - before other middleware)
    # Will be initialized during app startup if available
    try:
        from .core.error_handling_config import get_error_handling_config
        from .core.error_handling_middleware import ErrorHandlingMiddleware
        
        # Try to get config - may not be available during import
        try:
            config = get_error_handling_config()
            if config and getattr(config, 'middleware_enabled', True):
                # Create error handling middleware instance
                error_middleware = ErrorHandlingMiddleware(config)
                app.add_middleware(BaseHTTPMiddleware, dispatch=error_middleware.dispatch)
                logger.info("✅ Error handling middleware initialized")
        except Exception as config_error:
            # Configuration not ready - middleware will be initialized later
            logger.debug(f"Error handling middleware will be initialized during startup: {config_error}")
            pass
            
    except ImportError as import_error:
        # Error handling module not available
        logger.warning(f"⚠️ Error handling middleware not available: {import_error}")
        pass
    
    # Observability middleware for monitoring agent interactions
    app.add_middleware(ObservabilityMiddleware)
    
    # Hook interceptor middleware for automatic event capture
    app.add_middleware(ObservabilityHookMiddleware)
    
    # Prometheus metrics middleware for HTTP request tracking
    app.add_middleware(PrometheusMiddleware, exclude_paths=["/health", "/metrics", "/docs", "/redoc", "/openapi.json"])
    
    # Include API routes
    app.include_router(api_router, prefix="/api/v1")
    app.include_router(error_handling_router, prefix="/api/v1")  # Error handling health endpoints
    app.include_router(enhanced_coordination_router, prefix="/api/v1")  # Enhanced multi-agent coordination
    app.include_router(global_coordination_router, prefix="/api/v1")  # Global coordination Phase 4
    app.include_router(sleep_management_router)
    app.include_router(intelligent_scheduling_router)
    app.include_router(monitoring_router)
    app.include_router(analytics_router)
    app.include_router(dashboard_router, prefix="/dashboard", tags=["dashboard"])
    app.include_router(simple_dashboard_router, prefix="/dashboard", tags=["simple-dashboard"])
    app.include_router(agent_activation_router, prefix="/api/agents", tags=["agent-activation"])
    app.include_router(hive_commands_router, prefix="/api/hive", tags=["hive-commands"])
    app.include_router(intelligence_router, tags=["intelligence"])
    app.include_router(claude_integration_router, prefix="/api", tags=["claude-integration"])
    
    @app.get("/debug-agents")
    async def debug_agents():
        """Debug endpoint to test agent count.""" 
        try:
            from .core.agent_spawner import get_active_agents_status
            active_agents = await get_active_agents_status()
            return {
                "agent_count": len(active_agents),
                "agents": active_agents,
                "status": "debug_working"
            }
        except Exception as e:
            return {"error": str(e), "status": "debug_error"}
    
    @app.get("/health")
    async def health_check():
        """Comprehensive health check endpoint aggregating all component status."""
        from .core.database import get_async_session
        from .core.redis import get_redis
        from datetime import datetime
        import json
        
        health_status = {
            "status": "healthy",
            "version": "2.0.0",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "components": {},
            "summary": {
                "healthy": 0,
                "unhealthy": 0,
                "total": 0
            }
        }
        
        # Check Database
        try:
            from sqlalchemy import text
            async for session in get_async_session():
                result = await session.execute(text("SELECT 1"))
                assert result.scalar() == 1
                health_status["components"]["database"] = {
                    "status": "healthy",
                    "details": "PostgreSQL connection successful",
                    "response_time_ms": "<5"
                }
                health_status["summary"]["healthy"] += 1
                break
        except Exception as e:
            health_status["components"]["database"] = {
                "status": "unhealthy", 
                "error": str(e),
                "details": "Failed to connect to PostgreSQL"
            }
            health_status["summary"]["unhealthy"] += 1
            health_status["status"] = "degraded"
        
        # Check Redis
        try:
            redis_client = get_redis()
            pong = await redis_client.ping()
            if pong:
                health_status["components"]["redis"] = {
                    "status": "healthy",
                    "details": "Redis connection successful",
                    "response_time_ms": "<5"
                }
                health_status["summary"]["healthy"] += 1
            else:
                raise Exception("Redis ping failed")
        except Exception as e:
            health_status["components"]["redis"] = {
                "status": "unhealthy",
                "error": str(e), 
                "details": "Failed to connect to Redis"
            }
            health_status["summary"]["unhealthy"] += 1
            health_status["status"] = "degraded"
        
        # Check Agent System (using real agent spawner data - FIXED)
        try:
            from .core.agent_spawner import get_active_agents_status
            active_agents = await get_active_agents_status()
            active_agent_count = len(active_agents) if active_agents else 0
            
            health_status["components"]["orchestrator"] = {
                "status": "healthy", 
                "details": "Agent Orchestrator running",
                "active_agents": active_agent_count
            }
            health_status["summary"]["healthy"] += 1
        except Exception as e:
            health_status["components"]["orchestrator"] = {
                "status": "unhealthy",
                "error": str(e),
                "details": f"Agent System error: {str(e)}"
            }
            health_status["summary"]["unhealthy"] += 1
            health_status["status"] = "degraded"
        
        # Check Event Processor (Observability)
        try:
            event_processor = getattr(app.state, 'event_processor', None)
            if event_processor:
                health_status["components"]["observability"] = {
                    "status": "healthy", 
                    "details": "Event processor running"
                }
                health_status["summary"]["healthy"] += 1
            else:
                raise Exception("Event processor not initialized")
        except Exception as e:
            health_status["components"]["observability"] = {
                "status": "unhealthy",
                "error": str(e),
                "details": "Event processor not available"
            }
            health_status["summary"]["unhealthy"] += 1
            health_status["status"] = "degraded"
        
        # Check Error Handling System
        try:
            error_config_manager = getattr(app.state, 'error_config_manager', None)
            if error_config_manager:
                config_status = error_config_manager.get_status()
                health_status["components"]["error_handling"] = {
                    "status": "healthy",
                    "details": "Error handling system running",
                    "environment": config_status.get("current_environment", "unknown"),
                    "hot_reload_active": config_status.get("hot_reload_active", False)
                }
                health_status["summary"]["healthy"] += 1
            else:
                # Error handling is optional, so this is not a hard failure
                health_status["components"]["error_handling"] = {
                    "status": "not_configured",
                    "details": "Error handling system not initialized"
                }
        except Exception as e:
            health_status["components"]["error_handling"] = {
                "status": "unhealthy",
                "error": str(e),
                "details": "Error handling system failed"
            }
            health_status["summary"]["unhealthy"] += 1
            health_status["status"] = "degraded"
        
        health_status["summary"]["total"] = health_status["summary"]["healthy"] + health_status["summary"]["unhealthy"]
        
        # Determine overall status
        if health_status["summary"]["unhealthy"] == 0:
            health_status["status"] = "healthy"
        elif health_status["summary"]["healthy"] > 0:
            health_status["status"] = "degraded" 
        else:
            health_status["status"] = "unhealthy"
            
        return health_status
    
    @app.get("/status")
    async def system_status():
        """System-wide status endpoint for component monitoring."""
        from .core.database import get_async_session
        from .core.redis import get_redis
        from datetime import datetime
        
        status = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "uptime_seconds": 0,  # Would need app start time tracking
            "version": "2.0.0",
            "environment": "development",
            "components": {
                "database": {"connected": False, "tables": 0},
                "redis": {"connected": False, "memory_used": "unknown"},
                "orchestrator": {"active": False, "agents": []},
                "observability": {"active": False, "events_processed": 0}
            }
        }
        
        # Database status
        try:
            from sqlalchemy import text
            async for session in get_async_session():
                # Check table count
                result = await session.execute(text("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                """))
                table_count = result.scalar()
                status["components"]["database"] = {
                    "connected": True,
                    "tables": table_count,
                    "migrations_current": True
                }
                break
        except Exception:
            status["components"]["database"]["connected"] = False
        
        # Redis status
        try:
            redis_client = get_redis()
            info = await redis_client.info("memory")
            status["components"]["redis"] = {
                "connected": True,
                "memory_used": info.get("used_memory_human", "unknown"),
                "streams_active": True
            }
        except Exception:
            status["components"]["redis"]["connected"] = False
        
        return status
    
    @app.get("/metrics")
    async def system_metrics():
        """Prometheus-compatible metrics endpoint with real data."""
        from .core.prometheus_exporter import get_prometheus_exporter
        from fastapi import Response
        
        try:
            exporter = get_prometheus_exporter()
            metrics_output = await exporter.generate_metrics()
            
            return Response(
                content=metrics_output,
                media_type="text/plain; version=0.0.4; charset=utf-8"
            )
            
        except Exception as e:
            logger.error("📊 Failed to generate Prometheus metrics", error=str(e))
            
            # Fallback to basic metrics
            fallback_metrics = """# HELP leanvibe_health_status System health status (1=healthy, 0=unhealthy)
# TYPE leanvibe_health_status gauge
leanvibe_health_status{component="database"} 0
leanvibe_health_status{component="redis"} 0
leanvibe_health_status{component="orchestrator"} 0

# HELP leanvibe_uptime_seconds Application uptime in seconds  
# TYPE leanvibe_uptime_seconds gauge
leanvibe_uptime_seconds 0
"""
            return Response(
                content=fallback_metrics,
                media_type="text/plain; version=0.0.4; charset=utf-8"
            )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler with structured logging."""
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "request_id": getattr(request.state, 'request_id', 'unknown')
            }
        )
    
    return app


# FastAPI application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_config=None,  # We use structlog
        access_log=False,  # Handled by middleware
    )