"""
API Endpoints for Project Index WebSocket Monitoring

Provides REST API endpoints for monitoring WebSocket performance,
health status, and metrics integration.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import PlainTextResponse
import structlog

from ..project_index.websocket_integration import get_websocket_manager, ProjectIndexWebSocketManager
from ..project_index.websocket_monitoring import (
    get_monitoring_system, export_prometheus_metrics, 
    get_health_status, get_performance_dashboard_data
)

logger = structlog.get_logger()

# Create router for WebSocket monitoring endpoints
router = APIRouter(
    prefix="/api/v1/project-index/websocket",
    tags=["project-index", "websocket", "monitoring"]
)


@router.get("/health")
async def get_websocket_health(
    manager: ProjectIndexWebSocketManager = Depends(get_websocket_manager)
) -> Dict[str, Any]:
    """
    Get WebSocket system health status.
    
    Returns comprehensive health information including:
    - Connection pool health
    - Event publisher status
    - Redis connectivity
    - Memory usage
    - Error rates
    """
    try:
        health_status = await get_health_status(manager)
        return {
            "status": "healthy" if health_status.get("overall_health", False) else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "health_checks": health_status
        }
    except Exception as e:
        logger.error("Failed to get health status", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve health status")


@router.get("/metrics")
async def get_websocket_metrics(
    manager: ProjectIndexWebSocketManager = Depends(get_websocket_manager)
) -> Dict[str, Any]:
    """
    Get comprehensive WebSocket metrics.
    
    Returns detailed metrics including:
    - Connection statistics
    - Event delivery metrics
    - Performance counters
    - Filter statistics
    """
    try:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": manager.get_metrics()
        }
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@router.get("/metrics/prometheus")
async def get_prometheus_metrics(
    manager: ProjectIndexWebSocketManager = Depends(get_websocket_manager)
) -> PlainTextResponse:
    """
    Export metrics in Prometheus format.
    
    Returns metrics in the standard Prometheus exposition format
    for integration with Prometheus monitoring.
    """
    try:
        prometheus_data = await export_prometheus_metrics(manager)
        return PlainTextResponse(
            content=prometheus_data,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    except Exception as e:
        logger.error("Failed to export Prometheus metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to export Prometheus metrics")


@router.get("/performance")
async def get_performance_data(
    window_minutes: int = Query(60, ge=1, le=1440, description="Time window in minutes"),
    manager: ProjectIndexWebSocketManager = Depends(get_websocket_manager)
) -> Dict[str, Any]:
    """
    Get performance analysis data.
    
    Args:
        window_minutes: Time window for analysis (1-1440 minutes)
    
    Returns:
        Performance data including:
        - Performance summary statistics
        - Anomaly detection results
        - Baseline comparisons
        - Health status
    """
    try:
        performance_data = await get_performance_dashboard_data(manager, window_minutes)
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "window_minutes": window_minutes,
            "data": performance_data
        }
    except Exception as e:
        logger.error("Failed to get performance data", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve performance data")


@router.get("/connections")
async def get_connection_status(
    manager: ProjectIndexWebSocketManager = Depends(get_websocket_manager)
) -> Dict[str, Any]:
    """
    Get detailed connection status.
    
    Returns information about active WebSocket connections including:
    - Connection count and details
    - User mappings
    - Subscription information
    - Health scores
    """
    try:
        metrics = manager.get_metrics()
        connection_details = metrics.get("connection_details", {})
        
        # Enhance connection details with health information
        enhanced_connections = {}
        for connection_id, details in connection_details.items():
            health_score = manager.performance_manager.connection_pool.get_connection_health(connection_id)
            enhanced_connections[connection_id] = {
                **details,
                "health_score": health_score,
                "health_status": "healthy" if health_score > 0.8 else "degraded" if health_score > 0.5 else "poor"
            }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "active_connections": metrics.get("active_connections", 0),
                "total_connections_handled": metrics["websocket_manager"].get("connections_handled", 0),
                "total_events_delivered": metrics["websocket_manager"].get("events_delivered", 0),
                "total_events_filtered": metrics["websocket_manager"].get("events_filtered", 0)
            },
            "connections": enhanced_connections
        }
    except Exception as e:
        logger.error("Failed to get connection status", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve connection status")


@router.get("/events/statistics")
async def get_event_statistics(
    window_minutes: int = Query(60, ge=1, le=1440, description="Time window in minutes"),
    manager: ProjectIndexWebSocketManager = Depends(get_websocket_manager)
) -> Dict[str, Any]:
    """
    Get event delivery statistics.
    
    Args:
        window_minutes: Time window for statistics (1-1440 minutes)
    
    Returns:
        Event statistics including:
        - Event counts by type
        - Delivery success rates
        - Filtering statistics
        - Performance metrics
    """
    try:
        monitoring_system = await get_monitoring_system(manager)
        performance_summary = monitoring_system.performance_monitor.get_performance_summary(window_minutes)
        
        # Get event publisher metrics
        publisher_metrics = manager.event_publisher.get_metrics()
        
        # Get event filter metrics
        filter_metrics = manager.event_filter.get_metrics()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "window_minutes": window_minutes,
            "event_publishing": {
                "total_published": publisher_metrics.get("events_published", 0),
                "total_delivered": publisher_metrics.get("events_delivered", 0),
                "total_failed": publisher_metrics.get("events_failed", 0),
                "subscribers_notified": publisher_metrics.get("subscribers_notified", 0),
                "delivery_rate": (
                    publisher_metrics.get("events_delivered", 0) / 
                    max(publisher_metrics.get("events_published", 1), 1)
                )
            },
            "event_filtering": {
                "events_passed": filter_metrics.get("events_passed", 0),
                "events_filtered": filter_metrics.get("events_filtered", 0),
                "filter_pass_rate": filter_metrics.get("filter_pass_rate", 0.0),
                "active_user_preferences": filter_metrics.get("active_user_preferences", 0)
            },
            "performance_metrics": performance_summary
        }
    except Exception as e:
        logger.error("Failed to get event statistics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve event statistics")


@router.get("/queue/status")
async def get_queue_status(
    manager: ProjectIndexWebSocketManager = Depends(get_websocket_manager)
) -> Dict[str, Any]:
    """
    Get event queue status.
    
    Returns information about event queues including:
    - Queue sizes
    - Processing rates
    - Backlog information
    """
    try:
        performance_metrics = manager.performance_manager.get_performance_summary()
        
        queue_info = {}
        if "priority_queue" in performance_metrics:
            queue_stats = performance_metrics["priority_queue"]
            queue_info = {
                "priority_queue": {
                    "current_size": queue_stats.get("current_queue_size", 0),
                    "utilization": queue_stats.get("queue_utilization", 0.0),
                    "events_queued": queue_stats.get("events_queued", 0),
                    "events_processed": queue_stats.get("events_processed", 0),
                    "events_dropped": queue_stats.get("events_dropped", 0),
                    "average_queue_time_ms": queue_stats.get("average_queue_time_ms", 0.0)
                }
            }
        
        if "batcher" in performance_metrics:
            batch_stats = performance_metrics["batcher"]
            queue_info["event_batcher"] = {
                "batches_created": batch_stats.get("batches_created", 0),
                "events_batched": batch_stats.get("events_batched", 0),
                "batch_efficiency": batch_stats.get("batch_efficiency", 0.0)
            }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "queue_status": queue_info
        }
    except Exception as e:
        logger.error("Failed to get queue status", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve queue status")


@router.get("/anomalies")
async def get_anomalies(
    window_minutes: int = Query(60, ge=1, le=1440, description="Time window in minutes"),
    metric_name: Optional[str] = Query(None, description="Specific metric to analyze"),
    manager: ProjectIndexWebSocketManager = Depends(get_websocket_manager)
) -> Dict[str, Any]:
    """
    Get anomaly detection results.
    
    Args:
        window_minutes: Time window for analysis (1-1440 minutes)
        metric_name: Optional specific metric to analyze
    
    Returns:
        Anomaly detection results for system metrics
    """
    try:
        monitoring_system = await get_monitoring_system(manager)
        
        metrics_to_check = ["event_delivery_latency_ms", "events_per_second", "error_rate", "memory_usage_mb"]
        if metric_name:
            if metric_name not in metrics_to_check:
                raise HTTPException(status_code=400, detail=f"Invalid metric name: {metric_name}")
            metrics_to_check = [metric_name]
        
        anomalies = {}
        for metric in metrics_to_check:
            metric_anomalies = monitoring_system.performance_monitor.detect_anomalies(metric, window_minutes)
            if metric_anomalies:
                anomalies[metric] = metric_anomalies
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "window_minutes": window_minutes,
            "anomalies_detected": len(anomalies) > 0,
            "anomalies": anomalies
        }
    except Exception as e:
        logger.error("Failed to get anomalies", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve anomalies")


@router.post("/monitoring/start")
async def start_monitoring(
    manager: ProjectIndexWebSocketManager = Depends(get_websocket_manager)
) -> Dict[str, Any]:
    """
    Start the monitoring system.
    
    Starts background monitoring tasks for metrics collection,
    health checks, and performance analysis.
    """
    try:
        monitoring_system = await get_monitoring_system(manager)
        await monitoring_system.start_monitoring()
        
        return {
            "status": "success",
            "message": "Monitoring system started",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("Failed to start monitoring", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to start monitoring system")


@router.post("/monitoring/stop")
async def stop_monitoring(
    manager: ProjectIndexWebSocketManager = Depends(get_websocket_manager)
) -> Dict[str, Any]:
    """
    Stop the monitoring system.
    
    Stops background monitoring tasks gracefully.
    """
    try:
        monitoring_system = await get_monitoring_system(manager)
        await monitoring_system.stop_monitoring()
        
        return {
            "status": "success", 
            "message": "Monitoring system stopped",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("Failed to stop monitoring", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to stop monitoring system")


@router.get("/debug/internal-state")
async def get_internal_state(
    manager: ProjectIndexWebSocketManager = Depends(get_websocket_manager)
) -> Dict[str, Any]:
    """
    Get internal system state for debugging.
    
    WARNING: This endpoint exposes internal system state and should
    only be used for debugging purposes. Not recommended for production.
    """
    try:
        monitoring_system = await get_monitoring_system(manager)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "warning": "This is internal debug information",
            "websocket_manager": {
                "active_connections": len(manager.active_connections),
                "connection_users": {k: v for k, v in manager.connection_users.items()},
                "redis_subscriber_active": manager.redis_subscriber_task is not None and not manager.redis_subscriber_task.done(),
                "metrics": manager.metrics
            },
            "event_publisher": {
                "subscriptions": {
                    "event_subscriptions": len(manager.event_publisher.subscriptions.event_subscriptions),
                    "project_subscriptions": len(manager.event_publisher.subscriptions.project_subscriptions),
                    "session_subscriptions": len(manager.event_publisher.subscriptions.session_subscriptions)
                },
                "event_history": {
                    "projects_tracked": len(manager.event_publisher.event_history.project_events),
                    "total_events": sum(
                        len(events) for events in manager.event_publisher.event_history.project_events.values()
                    )
                },
                "metrics": manager.event_publisher.metrics
            },
            "performance_manager": {
                "active_connections": len(manager.performance_manager.connection_pool.connections),
                "rate_limited_clients": len(manager.performance_manager.rate_limiter.tokens),
                "compression_stats": manager.performance_manager.compressor.compression_stats
            },
            "monitoring_system": {
                "monitoring_tasks_active": len(monitoring_system.monitoring_tasks),
                "alerts_configured": len(monitoring_system.performance_monitor.alerts),
                "baselines_configured": len(monitoring_system.performance_monitor.baselines)
            }
        }
    except Exception as e:
        logger.error("Failed to get internal state", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve internal state")