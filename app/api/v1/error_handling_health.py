"""
Health Check Endpoints for Error Handling System - LeanVibe Agent Hive 2.0 - VS 3.3

Comprehensive health monitoring endpoints for all error handling components:
- Circuit breaker status and metrics
- Retry policy performance monitoring
- Graceful degradation health checks
- Workflow error handling status
- Performance target validation
- System-wide error handling health assessment
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
import structlog

from ...core.circuit_breaker import get_all_circuit_breakers, get_circuit_breaker_status
from ...core.graceful_degradation import get_degradation_manager
from ...core.workflow_engine_error_handling import get_workflow_recovery_manager
from ...core.error_handling_config import get_error_handling_config, get_config_manager
from ...core.error_handling_integration import get_error_handling_integration

logger = structlog.get_logger()

router = APIRouter(prefix="/error-handling", tags=["Error Handling Health"])


@router.get("/health")
async def get_error_handling_health():
    """
    Get comprehensive health status of error handling system.
    
    Returns overall health assessment including all components.
    """
    try:
        start_time = datetime.utcnow()
        
        # Initialize health assessment
        health_status = {
            "status": "healthy",
            "timestamp": start_time.isoformat() + "Z",
            "version": "3.3",
            "overall_score": 100,
            "components": {},
            "performance_targets": {},
            "issues": [],
            "recommendations": []
        }
        
        # Get configuration
        config = get_error_handling_config()
        
        # Check circuit breakers
        circuit_breaker_health = await _check_circuit_breaker_health()
        health_status["components"]["circuit_breakers"] = circuit_breaker_health
        
        if circuit_breaker_health["status"] != "healthy":
            health_status["status"] = "degraded"
            health_status["overall_score"] -= 20
            health_status["issues"].extend(circuit_breaker_health.get("issues", []))
        
        # Check graceful degradation
        degradation_health = await _check_graceful_degradation_health()
        health_status["components"]["graceful_degradation"] = degradation_health
        
        if degradation_health["status"] != "healthy":
            health_status["status"] = "degraded"
            health_status["overall_score"] -= 15
            health_status["issues"].extend(degradation_health.get("issues", []))
        
        # Check workflow error handling
        workflow_health = await _check_workflow_error_handling_health()
        health_status["components"]["workflow_error_handling"] = workflow_health
        
        if workflow_health["status"] != "healthy":
            health_status["status"] = "degraded"
            health_status["overall_score"] -= 25
            health_status["issues"].extend(workflow_health.get("issues", []))
        
        # Check configuration system
        config_health = await _check_configuration_health()
        health_status["components"]["configuration"] = config_health
        
        if config_health["status"] != "healthy":
            health_status["status"] = "degraded"
            health_status["overall_score"] -= 10
            health_status["issues"].extend(config_health.get("issues", []))
        
        # Check observability integration
        observability_health = await _check_observability_integration_health()
        health_status["components"]["observability_integration"] = observability_health
        
        if observability_health["status"] != "healthy":
            health_status["status"] = "degraded"
            health_status["overall_score"] -= 10
            health_status["issues"].extend(observability_health.get("issues", []))
        
        # Check performance targets
        performance_check = await _check_performance_targets(config)
        health_status["performance_targets"] = performance_check
        
        if not performance_check["targets_met"]:
            health_status["status"] = "degraded"
            health_status["overall_score"] -= 20
            health_status["issues"].extend(performance_check.get("issues", []))
        
        # Generate recommendations
        health_status["recommendations"] = await _generate_health_recommendations(health_status)
        
        # Calculate final status
        if health_status["overall_score"] < 50:
            health_status["status"] = "unhealthy"
        elif health_status["overall_score"] < 80:
            health_status["status"] = "degraded"
        
        # Calculate response time
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        health_status["response_time_ms"] = round(response_time, 2)
        
        logger.info(
            "🏥 Error handling health check completed",
            status=health_status["status"],
            overall_score=health_status["overall_score"],
            response_time_ms=response_time,
            issues=len(health_status["issues"])
        )
        
        # Return appropriate HTTP status
        if health_status["status"] == "healthy":
            return JSONResponse(status_code=200, content=health_status)
        elif health_status["status"] == "degraded":
            return JSONResponse(status_code=200, content=health_status)  # 200 but degraded
        else:
            return JSONResponse(status_code=503, content=health_status)  # Service unavailable
        
    except Exception as e:
        logger.error(f"❌ Error handling health check failed: {e}", exc_info=True)
        
        error_response = {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": "Health check system failure",
            "details": str(e),
            "overall_score": 0
        }
        
        return JSONResponse(status_code=503, content=error_response)


@router.get("/health/circuit-breakers")
async def get_circuit_breaker_health():
    """Get detailed health status of all circuit breakers."""
    try:
        return await _check_circuit_breaker_health()
    except Exception as e:
        logger.error(f"Circuit breaker health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/graceful-degradation")
async def get_graceful_degradation_health():
    """Get detailed health status of graceful degradation system."""
    try:
        return await _check_graceful_degradation_health()
    except Exception as e:
        logger.error(f"Graceful degradation health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/workflow-error-handling")
async def get_workflow_error_handling_health():
    """Get detailed health status of workflow error handling system."""
    try:
        return await _check_workflow_error_handling_health()
    except Exception as e:
        logger.error(f"Workflow error handling health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/configuration")
async def get_configuration_health():
    """Get detailed health status of configuration system."""
    try:
        return await _check_configuration_health()
    except Exception as e:
        logger.error(f"Configuration health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_error_handling_metrics(
    component: Optional[str] = Query(None, description="Specific component metrics"),
    include_history: bool = Query(False, description="Include historical metrics")
):
    """
    Get comprehensive metrics for error handling system.
    
    Args:
        component: Specific component to get metrics for
        include_history: Whether to include historical data
        
    Returns:
        Metrics data for requested components
    """
    try:
        metrics = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "components": {}
        }
        
        # Circuit breaker metrics
        if not component or component == "circuit_breakers":
            circuit_breakers = get_all_circuit_breakers()
            cb_metrics = {}
            
            for name, cb in circuit_breakers.items():
                cb_metrics[name] = cb.get_metrics()
            
            metrics["components"]["circuit_breakers"] = {
                "total_circuit_breakers": len(circuit_breakers),
                "circuit_breakers": cb_metrics
            }
        
        # Graceful degradation metrics
        if not component or component == "graceful_degradation":
            degradation_manager = get_degradation_manager()
            metrics["components"]["graceful_degradation"] = degradation_manager.get_metrics()
        
        # Workflow error handling metrics
        if not component or component == "workflow_error_handling":
            recovery_manager = get_workflow_recovery_manager()
            metrics["components"]["workflow_error_handling"] = recovery_manager.get_recovery_metrics()
        
        # Observability integration metrics
        if not component or component == "observability_integration":
            integration = get_error_handling_integration()
            metrics["components"]["observability_integration"] = integration.get_integration_metrics()
        
        # Configuration metrics
        if not component or component == "configuration":
            config_manager = get_config_manager()
            metrics["components"]["configuration"] = config_manager.get_status()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error handling metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_error_handling_status():
    """Get current status and configuration of error handling system."""
    try:
        config = get_error_handling_config()
        config_manager = get_config_manager()
        
        status = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "enabled": config.enabled,
            "environment": config.environment.value,
            "debug_mode": config.debug_mode,
            "middleware_enabled": config.middleware_enabled,
            "configuration_manager": config_manager.get_status(),
            "component_status": {
                "circuit_breaker_enabled": config.circuit_breaker.enabled,
                "retry_policy_enabled": config.retry_policy.enabled,
                "graceful_degradation_enabled": config.graceful_degradation.enabled,
                "workflow_error_handling_enabled": config.workflow_error_handling.enabled,
                "observability_enabled": config.observability.enabled
            },
            "performance_targets": {
                "max_processing_time_ms": config.performance_targets.max_processing_time_ms,
                "availability_target": config.performance_targets.availability_target,
                "recovery_time_target_ms": config.performance_targets.recovery_time_target_ms
            }
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error handling status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reset-metrics")
async def reset_error_handling_metrics(
    component: Optional[str] = Query(None, description="Specific component to reset")
):
    """
    Reset metrics for error handling components.
    
    Args:
        component: Specific component to reset, or all if not specified
        
    Returns:
        Confirmation of reset operation
    """
    try:
        reset_results = {}
        
        # Reset circuit breaker metrics
        if not component or component == "circuit_breakers":
            circuit_breakers = get_all_circuit_breakers()
            for name, cb in circuit_breakers.items():
                cb.reset()
            reset_results["circuit_breakers"] = f"Reset {len(circuit_breakers)} circuit breakers"
        
        # Reset graceful degradation metrics
        if not component or component == "graceful_degradation":
            degradation_manager = get_degradation_manager()
            degradation_manager.reset_metrics()
            reset_results["graceful_degradation"] = "Metrics reset successfully"
        
        # Note: Workflow error handling and other components don't have reset methods
        # This is intentional to preserve important error history
        
        logger.info(
            "🔄 Error handling metrics reset",
            component=component or "all",
            results=reset_results
        )
        
        return {
            "message": "Metrics reset completed",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "component": component or "all",
            "results": reset_results
        }
        
    except Exception as e:
        logger.error(f"Error handling metrics reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Private helper functions

async def _check_circuit_breaker_health() -> Dict[str, Any]:
    """Check health of all circuit breakers."""
    try:
        circuit_breakers = get_all_circuit_breakers()
        
        if not circuit_breakers:
            return {
                "status": "healthy",
                "message": "No circuit breakers configured",
                "circuit_breaker_count": 0
            }
        
        health_results = {}
        overall_healthy = True
        issues = []
        
        for name, cb in circuit_breakers.items():
            cb_health = await cb.health_check()
            health_results[name] = cb_health
            
            if cb_health["status"] != "healthy":
                overall_healthy = False
                issues.append(f"Circuit breaker {name} is {cb_health['status']}")
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "circuit_breaker_count": len(circuit_breakers),
            "circuit_breakers": health_results,
            "issues": issues
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "circuit_breaker_count": 0
        }


async def _check_graceful_degradation_health() -> Dict[str, Any]:
    """Check health of graceful degradation system."""
    try:
        degradation_manager = get_degradation_manager()
        return await degradation_manager.health_check()
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def _check_workflow_error_handling_health() -> Dict[str, Any]:
    """Check health of workflow error handling system."""
    try:
        recovery_manager = get_workflow_recovery_manager()
        return await recovery_manager.health_check()
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def _check_configuration_health() -> Dict[str, Any]:
    """Check health of configuration system."""
    try:
        config = get_error_handling_config()
        config_manager = get_config_manager()
        
        # Validate current configuration
        validation_issues = config.validate_configuration()
        
        # Check configuration manager status
        manager_status = config_manager.get_status()
        
        issues = validation_issues["errors"] + validation_issues["warnings"]
        
        return {
            "status": "healthy" if not validation_issues["errors"] else "degraded",
            "validation_errors": validation_issues["errors"],
            "validation_warnings": validation_issues["warnings"],
            "recommendations": validation_issues["recommendations"],
            "manager_status": manager_status,
            "issues": issues
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def _check_observability_integration_health() -> Dict[str, Any]:
    """Check health of observability integration."""
    try:
        integration = get_error_handling_integration()
        metrics = integration.get_integration_metrics()
        
        issues = []
        if not metrics["observability_hooks_available"]:
            issues.append("Observability hooks not available")
        
        return {
            "status": "healthy" if not issues else "degraded",
            "metrics": metrics,
            "issues": issues
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


async def _check_performance_targets(config) -> Dict[str, Any]:
    """Check if performance targets are being met."""
    try:
        targets = config.performance_targets
        
        # Collect performance data from components
        performance_data = {}
        issues = []
        targets_met = True
        
        # Circuit breaker performance
        circuit_breakers = get_all_circuit_breakers()
        cb_performance = []
        
        for name, cb in circuit_breakers.items():
            metrics = cb.get_metrics()
            cb_perf = metrics.get("performance", {})
            cb_performance.append({
                "name": name,
                "average_decision_time_ms": cb_perf.get("average_decision_time_ms", 0),
                "target_met": cb_perf.get("target_met", True)
            })
            
            if not cb_perf.get("target_met", True):
                targets_met = False
                issues.append(f"Circuit breaker {name} exceeds performance target")
        
        performance_data["circuit_breakers"] = cb_performance
        
        # Graceful degradation performance
        degradation_manager = get_degradation_manager()
        degradation_metrics = degradation_manager.get_metrics()
        degradation_perf = degradation_metrics.get("performance", {})
        
        performance_data["graceful_degradation"] = {
            "average_processing_time_ms": degradation_perf.get("average_processing_time_ms", 0),
            "target_met": degradation_perf.get("target_met", True)
        }
        
        if not degradation_perf.get("target_met", True):
            targets_met = False
            issues.append("Graceful degradation exceeds performance target")
        
        # Workflow error handling performance
        recovery_manager = get_workflow_recovery_manager()
        recovery_metrics = recovery_manager.get_recovery_metrics()
        
        performance_data["workflow_error_handling"] = {
            "average_recovery_time_ms": recovery_metrics.get("average_recovery_time_ms", 0),
            "recovery_target_met": recovery_metrics.get("recovery_target_met", True)
        }
        
        if not recovery_metrics.get("recovery_target_met", True):
            targets_met = False
            issues.append("Workflow error recovery exceeds time target")
        
        return {
            "targets_met": targets_met,
            "performance_data": performance_data,
            "targets": {
                "max_processing_time_ms": targets.max_processing_time_ms,
                "availability_target": targets.availability_target,
                "recovery_time_target_ms": targets.recovery_time_target_ms,
                "circuit_breaker_decision_time_ms": targets.circuit_breaker_decision_time_ms,
                "graceful_degradation_time_ms": targets.graceful_degradation_time_ms,
                "workflow_recovery_time_ms": targets.workflow_recovery_time_ms
            },
            "issues": issues
        }
        
    except Exception as e:
        return {
            "targets_met": False,
            "error": str(e),
            "issues": ["Performance target check failed"]
        }


async def _generate_health_recommendations(health_status: Dict[str, Any]) -> List[str]:
    """Generate health recommendations based on current status."""
    recommendations = []
    
    # Overall health recommendations
    if health_status["overall_score"] < 80:
        recommendations.append("System performance degraded - review component issues")
    
    # Component-specific recommendations
    components = health_status.get("components", {})
    
    # Circuit breaker recommendations
    cb_health = components.get("circuit_breakers", {})
    if cb_health.get("status") != "healthy":
        recommendations.append("Review circuit breaker thresholds and failure patterns")
    
    # Graceful degradation recommendations
    gd_health = components.get("graceful_degradation", {})
    if gd_health.get("status") != "healthy":
        recommendations.append("Optimize fallback strategies for better performance")
    
    # Workflow error handling recommendations
    weh_health = components.get("workflow_error_handling", {})
    if weh_health.get("status") != "healthy":
        recommendations.append("Review workflow error patterns and recovery strategies")
    
    # Performance target recommendations
    perf_targets = health_status.get("performance_targets", {})
    if not perf_targets.get("targets_met", True):
        recommendations.append("Performance targets not met - consider optimization")
    
    # General recommendations based on issue count
    issue_count = len(health_status.get("issues", []))
    if issue_count > 5:
        recommendations.append("High number of issues detected - consider system review")
    
    if not recommendations:
        recommendations.append("System operating within normal parameters")
    
    return recommendations