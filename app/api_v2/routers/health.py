"""
Health API - Consolidated health and diagnostics endpoints

Consolidates v1/error_handling_health.py and health-related functionality
from dx_debugging.py into a unified health monitoring resource.

Performance target: <25ms P95 response time
"""

import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime

import structlog
from fastapi import APIRouter, Request, HTTPException
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ...core.database import get_session_dependency
from ..middleware import get_current_user_from_request

logger = structlog.get_logger()
router = APIRouter()

@router.get("/status")
async def health_status():
    """
    Basic health check endpoint.
    
    Performance target: <25ms
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "service": "leanvibe-agent-hive"
    }

@router.get("/ready")
async def readiness_check(
    db: AsyncSession = Depends(get_session_dependency)
):
    """
    Readiness check for load balancers and orchestrators.
    
    Performance target: <25ms
    """
    try:
        # Check database connectivity
        start_time = time.time()
        await db.execute(text("SELECT 1"))
        db_time = (time.time() - start_time) * 1000
        
        checks = {
            "database": {
                "status": "healthy",
                "response_time_ms": round(db_time, 2)
            }
        }
        
        # All checks passed
        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": checks
        }
        
    except Exception as e:
        logger.error("readiness_check_failed", error=str(e))
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )

@router.get("/live")
async def liveness_check():
    """
    Liveness check for container orchestrators.
    
    Performance target: <25ms
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime_seconds": time.time()  # Would track actual uptime
    }

@router.get("/metrics")
async def health_metrics(
    request: Request,
    db: AsyncSession = Depends(get_session_dependency)
):
    """
    Basic health metrics for monitoring.
    
    Performance target: <25ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        start_time = time.time()
        
        # Database health check
        db_start = time.time()
        await db.execute(text("SELECT COUNT(*) FROM agents"))
        db_response_time = (time.time() - db_start) * 1000
        
        # Calculate total response time
        total_response_time = (time.time() - start_time) * 1000
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "response_time_ms": round(total_response_time, 2),
            "database": {
                "status": "healthy" if db_response_time < 100 else "slow",
                "response_time_ms": round(db_response_time, 2),
                "threshold_ms": 100
            },
            "performance": {
                "target_response_time_ms": 25,
                "current_response_time_ms": round(total_response_time, 2),
                "meets_target": total_response_time < 25
            }
        }
        
    except Exception as e:
        logger.error("health_metrics_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get health metrics: {str(e)}"
        )