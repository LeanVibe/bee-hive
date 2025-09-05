"""
Infrastructure Health API Endpoints for Epic 6 Phase 2

Provides REST API endpoints for infrastructure health monitoring, designed for 
production environments with comprehensive diagnostics and monitoring capabilities.
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from app.core.infrastructure_health import get_infrastructure_health, health_checker, HealthStatus
from app.core.auth import get_current_user_optional
from app.core.logging_service import get_component_logger

logger = get_component_logger("infrastructure_health_api")

router = APIRouter(prefix="/health", tags=["Infrastructure Health"])


@router.get("/", response_model=Dict[str, Any])
async def get_health_status(
    detailed: bool = False,
    current_user = Depends(get_current_user_optional)
) -> JSONResponse:
    """
    Get comprehensive infrastructure health status.
    
    Args:
        detailed: Include detailed diagnostics and recommendations
        current_user: Optional authenticated user (for detailed info)
    
    Returns:
        Health status summary with overall status and service details
    """
    try:
        health_summary = await get_infrastructure_health()
        
        # For unauthenticated requests, return basic status only
        if not current_user and detailed:
            health_summary = {
                "overall_status": health_summary["overall_status"],
                "timestamp": health_summary["timestamp"],
                "summary": health_summary["summary"]
            }
        
        # Determine HTTP status code based on health
        overall_status = health_summary["overall_status"]
        if overall_status == "critical":
            status_code = 503  # Service Unavailable
        elif overall_status == "down":
            status_code = 503  # Service Unavailable
        elif overall_status == "warning":
            status_code = 200  # OK but with warnings
        else:
            status_code = 200  # OK
        
        return JSONResponse(
            status_code=status_code,
            content=health_summary
        )
        
    except Exception as e:
        logger.error(f"Health check endpoint failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "overall_status": "critical",
                "timestamp": 0,
                "error": "Health check system failure",
                "details": {"error": str(e)}
            }
        )


@router.get("/database")
async def get_database_health() -> JSONResponse:
    """Get database-specific health status."""
    try:
        db_health = await health_checker.check_database_health()
        
        # Determine HTTP status code
        if db_health.status in [HealthStatus.CRITICAL, HealthStatus.DOWN]:
            status_code = 503
        elif db_health.status == HealthStatus.WARNING:
            status_code = 200
        else:
            status_code = 200
            
        return JSONResponse(
            status_code=status_code,
            content=db_health.to_dict()
        )
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "service": "postgresql",
                "status": "down",
                "message": f"Database health check failed: {str(e)}",
                "error": str(e)
            }
        )


@router.get("/redis")
async def get_redis_health() -> JSONResponse:
    """Get Redis-specific health status."""
    try:
        redis_health = await health_checker.check_redis_health()
        
        # Determine HTTP status code
        if redis_health.status in [HealthStatus.CRITICAL, HealthStatus.DOWN]:
            status_code = 503
        elif redis_health.status == HealthStatus.WARNING:
            status_code = 200
        else:
            status_code = 200
            
        return JSONResponse(
            status_code=status_code,
            content=redis_health.to_dict()
        )
        
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "service": "redis",
                "status": "down",
                "message": f"Redis health check failed: {str(e)}",
                "error": str(e)
            }
        )


@router.get("/docker")
async def get_docker_health(
    current_user = Depends(get_current_user_optional)
) -> JSONResponse:
    """Get Docker services health status (requires authentication for full details)."""
    try:
        docker_health = await health_checker.check_docker_services_health()
        
        # For unauthenticated requests, limit details
        if not current_user:
            limited_result = {
                "service": docker_health.service,
                "status": docker_health.status.value,
                "message": docker_health.message,
                "response_time_ms": docker_health.response_time_ms,
                "timestamp": docker_health.timestamp
            }
            return JSONResponse(content=limited_result)
        
        # Determine HTTP status code
        if docker_health.status in [HealthStatus.CRITICAL, HealthStatus.DOWN]:
            status_code = 503
        elif docker_health.status == HealthStatus.WARNING:
            status_code = 200
        else:
            status_code = 200
            
        return JSONResponse(
            status_code=status_code,
            content=docker_health.to_dict()
        )
        
    except Exception as e:
        logger.error(f"Docker health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "service": "docker_services",
                "status": "warning",
                "message": f"Docker health check failed: {str(e)}",
                "error": str(e)
            }
        )


@router.get("/ready")
async def readiness_check() -> JSONResponse:
    """
    Kubernetes-style readiness check.
    Returns 200 if system is ready to serve traffic, 503 otherwise.
    """
    try:
        health_summary = await get_infrastructure_health()
        
        # System is ready if no services are DOWN and not more than 1 CRITICAL
        critical_services = [
            name for name, service in health_summary["services"].items()
            if service["status"] in ["critical", "down"]
        ]
        
        if len(critical_services) == 0:
            return JSONResponse(
                status_code=200,
                content={"status": "ready", "message": "System ready to serve traffic"}
            )
        elif len(critical_services) == 1 and "docker_services" in critical_services:
            # Docker services being down doesn't prevent core functionality
            return JSONResponse(
                status_code=200,
                content={"status": "ready", "message": "System ready (Docker monitoring unavailable)"}
            )
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "message": f"Critical services unavailable: {critical_services}"
                }
            )
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "message": "Health check system failure",
                "error": str(e)
            }
        )


@router.get("/live")
async def liveness_check() -> JSONResponse:
    """
    Kubernetes-style liveness check.
    Returns 200 if the application is alive, 503 if it should be restarted.
    """
    try:
        # Basic check - if we can execute this code, we're alive
        # For more comprehensive check, verify core database connectivity
        db_health = await health_checker.check_database_health()
        
        if db_health.status == HealthStatus.DOWN:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "Core database connectivity lost"
                }
            )
        
        return JSONResponse(
            status_code=200,
            content={"status": "alive", "message": "Application is alive"}
        )
        
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "Application liveness check failed",
                "error": str(e)
            }
        )


@router.post("/refresh")
async def refresh_health_cache(
    current_user = Depends(get_current_user_optional)
) -> JSONResponse:
    """
    Force refresh of health check cache.
    Requires authentication to prevent abuse.
    """
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        # Clear any cached results and perform fresh check
        await health_checker.initialize()
        health_summary = await get_infrastructure_health()
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Health cache refreshed successfully",
                "health_summary": health_summary
            }
        )
        
    except Exception as e:
        logger.error(f"Health cache refresh failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "message": "Failed to refresh health cache",
                "error": str(e)
            }
        )