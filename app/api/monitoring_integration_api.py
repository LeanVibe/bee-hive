"""
Monitoring Integration API for LeanVibe Agent Hive 2.0

Unified API for accessing comprehensive monitoring data from all observability components:
- Prometheus metrics export
- Real-time performance dashboards
- Intelligent alert management
- Distributed tracing analytics
- System health and capacity planning
"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field
import structlog

from ..core.prometheus_metrics_exporter import get_prometheus_exporter, PrometheusMetricsExporter
from ..core.performance_monitoring import get_performance_intelligence_engine, PerformanceIntelligenceEngine
from ..core.intelligent_alerting_system import get_intelligent_alerting_system, IntelligentAlertingSystem, AlertRule, AlertInstance
from ..core.distributed_tracing_system import get_distributed_tracing_system, DistributedTracingSystem
from ..core.config import settings

logger = structlog.get_logger()

# Create router
monitoring_router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


# Request/Response Models
class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    components: Dict[str, Dict[str, Any]]


class MetricsExportResponse(BaseModel):
    """Metrics export response model."""
    total_metrics: int
    export_format: str
    timestamp: datetime
    collection_latency_ms: float


class PerformanceDashboardResponse(BaseModel):
    """Performance dashboard response model."""
    timestamp: datetime
    time_window_minutes: int
    system_health: Dict[str, Any]
    real_time_metrics: Dict[str, Any]
    alerts_summary: Dict[str, Any]
    performance_predictions: Dict[str, Any]
    anomalies: Dict[str, Any]
    capacity_status: Dict[str, Any]


class AlertRuleRequest(BaseModel):
    """Request model for creating/updating alert rules."""
    name: str
    description: str
    metric_query: str
    threshold: Optional[float] = None
    comparison_operator: str = ">"
    evaluation_window: int = 60
    severity: str = "medium"
    urgency: str = "medium"
    channels: List[str] = Field(default_factory=lambda: ["email", "slack"])
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    enabled: bool = True


class AlertInstanceResponse(BaseModel):
    """Response model for alert instances."""
    id: str
    rule_id: str
    name: str
    description: str
    severity: str
    urgency: str
    status: str
    fired_at: datetime
    resolved_at: Optional[datetime] = None
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    labels: Dict[str, str]
    annotations: Dict[str, str]
    suggested_actions: List[str]


class TraceAnalyticsResponse(BaseModel):
    """Response model for trace analytics."""
    time_period_hours: int
    total_traces: int
    performance_metrics: Dict[str, float]
    error_metrics: Dict[str, Any]
    sampling_metrics: Dict[str, float]
    service_metrics: Dict[str, Any]
    slow_operations: List[Dict[str, Any]]


class SystemCapacityResponse(BaseModel):
    """Response model for system capacity analysis."""
    timestamp: datetime
    resource_utilization: Dict[str, float]
    capacity_predictions: Dict[str, Any]
    scaling_recommendations: List[Dict[str, Any]]
    optimization_opportunities: List[Dict[str, Any]]


# Startup tracking
startup_time = time.time()


# Health and Status Endpoints
@monitoring_router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Returns detailed health status of all monitoring components.
    """
    try:
        current_time = datetime.utcnow()
        uptime = time.time() - startup_time
        
        components = {}
        
        # Check Prometheus metrics exporter
        try:
            prometheus_exporter = await get_prometheus_exporter()
            summary = await prometheus_exporter.get_metrics_summary()
            components["prometheus_exporter"] = {
                "status": "healthy" if summary.get("collection_active") else "degraded",
                "total_metrics": summary.get("total_metrics", 0),
                "last_collection": summary.get("last_collection"),
                "avg_latency_ms": summary.get("average_collection_latency_ms", 0)
            }
        except Exception as e:
            components["prometheus_exporter"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check performance intelligence engine
        try:
            perf_engine = await get_performance_intelligence_engine()
            components["performance_engine"] = {
                "status": "healthy",
                "active": True
            }
        except Exception as e:
            components["performance_engine"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check intelligent alerting system
        try:
            alerting_system = await get_intelligent_alerting_system()
            components["alerting_system"] = {
                "status": "healthy",
                "active_alerts": len(alerting_system.active_alerts),
                "alert_rules": len(alerting_system.alert_rules)
            }
        except Exception as e:
            components["alerting_system"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check distributed tracing system
        try:
            tracing_system = await get_distributed_tracing_system()
            components["tracing_system"] = {
                "status": "healthy",
                "active_traces": len(tracing_system.active_traces),
                "sampling_rate": tracing_system.sampler.current_rate
            }
        except Exception as e:
            components["tracing_system"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Determine overall status
        unhealthy_components = [
            name for name, component in components.items() 
            if component["status"] == "unhealthy"
        ]
        
        if unhealthy_components:
            overall_status = "degraded" if len(unhealthy_components) < len(components) // 2 else "unhealthy"
        else:
            overall_status = "healthy"
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=current_time,
            version="2.0.0",
            uptime_seconds=uptime,
            components=components
        )
        
    except Exception as e:
        logger.error("Health check error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@monitoring_router.get("/status")
async def system_status():
    """
    Quick system status endpoint for load balancers.
    
    Returns simple OK/ERROR status.
    """
    try:
        # Quick health checks
        prometheus_exporter = await get_prometheus_exporter()
        alerting_system = await get_intelligent_alerting_system()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "OK",
                "timestamp": datetime.utcnow().isoformat(),
                "monitoring_active": True
            }
        )
        
    except Exception:
        return JSONResponse(
            status_code=503,
            content={
                "status": "ERROR",
                "timestamp": datetime.utcnow().isoformat(),
                "monitoring_active": False
            }
        )


# Metrics Export Endpoints
@monitoring_router.get("/metrics", response_class=PlainTextResponse)
async def export_prometheus_metrics():
    """
    Export Prometheus metrics in standard format.
    
    This endpoint is scraped by Prometheus server.
    """
    try:
        prometheus_exporter = await get_prometheus_exporter()
        metrics_output = await prometheus_exporter.export_metrics()
        
        return PlainTextResponse(
            content=metrics_output,
            headers={"Content-Type": prometheus_exporter.get_content_type()}
        )
        
    except Exception as e:
        logger.error("Metrics export error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Metrics export failed: {str(e)}")


@monitoring_router.get("/metrics/summary", response_model=MetricsExportResponse)
async def get_metrics_summary():
    """
    Get summary of current metrics collection.
    
    Returns metadata about metrics collection performance.
    """
    try:
        prometheus_exporter = await get_prometheus_exporter()
        summary = await prometheus_exporter.get_metrics_summary()
        
        return MetricsExportResponse(
            total_metrics=summary.get("total_metrics", 0),
            export_format="prometheus",
            timestamp=datetime.utcnow(),
            collection_latency_ms=summary.get("average_collection_latency_ms", 0)
        )
        
    except Exception as e:
        logger.error("Metrics summary error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Metrics summary failed: {str(e)}")


# Performance Dashboard Endpoints
@monitoring_router.get("/dashboard/performance", response_model=PerformanceDashboardResponse)
async def get_performance_dashboard(
    time_window_minutes: int = Query(default=60, ge=5, le=1440, description="Time window in minutes")
):
    """
    Get comprehensive real-time performance dashboard data.
    
    Returns system health, metrics, alerts, predictions, and capacity status.
    """
    try:
        performance_engine = await get_performance_intelligence_engine()
        dashboard_data = await performance_engine.get_real_time_performance_dashboard(
            time_window_minutes=time_window_minutes
        )
        
        if "error" in dashboard_data:
            raise HTTPException(status_code=500, detail=dashboard_data["error"])
        
        return PerformanceDashboardResponse(**dashboard_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Performance dashboard error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Performance dashboard failed: {str(e)}")


@monitoring_router.get("/dashboard/performance/predictions")
async def get_performance_predictions(
    metric_names: List[str] = Query(description="List of metric names to predict"),
    horizon_hours: int = Query(default=1, ge=1, le=24, description="Prediction horizon in hours")
):
    """
    Get performance predictions for specified metrics.
    
    Uses ML-based forecasting to predict future metric values.
    """
    try:
        performance_engine = await get_performance_intelligence_engine()
        predictions = await performance_engine.predict_performance_metrics(
            metric_names=metric_names,
            horizon_hours=horizon_hours
        )
        
        return {
            "predictions": [asdict(pred) for pred in predictions],
            "horizon_hours": horizon_hours,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Performance predictions error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Performance predictions failed: {str(e)}")


@monitoring_router.get("/dashboard/performance/anomalies")
async def get_performance_anomalies(
    time_window_hours: int = Query(default=1, ge=1, le=24, description="Time window in hours")
):
    """
    Get detected performance anomalies.
    
    Returns anomalies detected in the specified time window.
    """
    try:
        performance_engine = await get_performance_intelligence_engine()
        anomalies = await performance_engine.detect_performance_anomalies(
            time_window_hours=time_window_hours
        )
        
        return {
            "anomalies": [asdict(anomaly) for anomaly in anomalies],
            "time_window_hours": time_window_hours,
            "total_anomalies": len(anomalies),
            "detected_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Performance anomalies error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Performance anomalies failed: {str(e)}")


# Alert Management Endpoints
@monitoring_router.get("/alerts", response_model=List[AlertInstanceResponse])
async def get_active_alerts(
    severity: Optional[str] = Query(default=None, description="Filter by severity"),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of alerts")
):
    """
    Get active alerts with optional filtering.
    
    Returns currently firing alerts with their details.
    """
    try:
        alerting_system = await get_intelligent_alerting_system()
        active_alerts = list(alerting_system.active_alerts.values())
        
        # Filter by severity if specified
        if severity:
            active_alerts = [
                alert for alert in active_alerts 
                if alert.severity.value.lower() == severity.lower()
            ]
        
        # Limit results
        active_alerts = active_alerts[:limit]
        
        # Convert to response model
        return [
            AlertInstanceResponse(
                id=alert.id,
                rule_id=alert.rule_id,
                name=alert.name,
                description=alert.description,
                severity=alert.severity.value,
                urgency=alert.urgency.value,
                status=alert.status.value,
                fired_at=alert.fired_at,
                resolved_at=alert.resolved_at,
                current_value=alert.current_value,
                threshold_value=alert.threshold_value,
                labels=alert.labels,
                annotations=alert.annotations,
                suggested_actions=alert.suggested_actions
            )
            for alert in active_alerts
        ]
        
    except Exception as e:
        logger.error("Get active alerts error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Get active alerts failed: {str(e)}")


@monitoring_router.post("/alerts/rules")
async def create_alert_rule(rule_request: AlertRuleRequest):
    """
    Create a new alert rule.
    
    Configures a new alerting rule with specified conditions and actions.
    """
    try:
        alerting_system = await get_intelligent_alerting_system()
        
        # Convert request to AlertRule
        from ..core.intelligent_alerting_system import AlertRule, AlertType, AlertSeverity, AlertUrgency, AlertChannel
        
        # Map string values to enums
        severity_map = {
            "low": AlertSeverity.LOW,
            "medium": AlertSeverity.MEDIUM,
            "high": AlertSeverity.HIGH,
            "critical": AlertSeverity.CRITICAL
        }
        
        urgency_map = {
            "low": AlertUrgency.LOW,
            "medium": AlertUrgency.MEDIUM,
            "high": AlertUrgency.HIGH,
            "critical": AlertUrgency.CRITICAL,
            "emergency": AlertUrgency.EMERGENCY
        }
        
        channel_map = {
            "email": AlertChannel.EMAIL,
            "slack": AlertChannel.SLACK,
            "webhook": AlertChannel.WEBHOOK,
            "sms": AlertChannel.SMS
        }
        
        alert_rule = AlertRule(
            id=f"custom_{int(time.time())}",
            name=rule_request.name,
            description=rule_request.description,
            alert_type=AlertType.THRESHOLD,  # Default to threshold
            metric_query=rule_request.metric_query,
            threshold=rule_request.threshold,
            comparison_operator=rule_request.comparison_operator,
            evaluation_window=rule_request.evaluation_window,
            severity=severity_map.get(rule_request.severity.lower(), AlertSeverity.MEDIUM),
            urgency=urgency_map.get(rule_request.urgency.lower(), AlertUrgency.MEDIUM),
            channels=[channel_map.get(ch, AlertChannel.EMAIL) for ch in rule_request.channels],
            labels=rule_request.labels,
            annotations=rule_request.annotations,
            enabled=rule_request.enabled
        )
        
        # Register the rule
        alerting_system.register_alert_rule(alert_rule)
        
        return {
            "rule_id": alert_rule.id,
            "status": "created",
            "message": f"Alert rule '{rule_request.name}' created successfully"
        }
        
    except Exception as e:
        logger.error("Create alert rule error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Create alert rule failed: {str(e)}")


@monitoring_router.delete("/alerts/rules/{rule_id}")
async def delete_alert_rule(rule_id: str = Path(description="Alert rule ID")):
    """
    Delete an alert rule.
    
    Removes the specified alert rule from the system.
    """
    try:
        alerting_system = await get_intelligent_alerting_system()
        
        if alerting_system.remove_alert_rule(rule_id):
            return {
                "rule_id": rule_id,
                "status": "deleted",
                "message": f"Alert rule '{rule_id}' deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Alert rule '{rule_id}' not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Delete alert rule error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Delete alert rule failed: {str(e)}")


# Distributed Tracing Endpoints
@monitoring_router.get("/tracing/analytics", response_model=TraceAnalyticsResponse)
async def get_trace_analytics(
    hours: int = Query(default=1, ge=1, le=24, description="Analysis time window in hours")
):
    """
    Get comprehensive trace analytics.
    
    Returns performance metrics, error rates, and slow operations from traces.
    """
    try:
        tracing_system = await get_distributed_tracing_system()
        analytics = await tracing_system.get_trace_analytics(hours=hours)
        
        if "error" in analytics:
            raise HTTPException(status_code=500, detail=analytics["error"])
        
        return TraceAnalyticsResponse(**analytics)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Trace analytics error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Trace analytics failed: {str(e)}")


@monitoring_router.get("/tracing/slow-operations")
async def get_slow_operations(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of operations")
):
    """
    Get slowest operations from trace data.
    
    Returns operations with highest latency for performance optimization.
    """
    try:
        tracing_system = await get_distributed_tracing_system()
        slow_ops = await tracing_system._get_slow_operations(limit=limit)
        
        return {
            "slow_operations": slow_ops,
            "limit": limit,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Slow operations error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Slow operations failed: {str(e)}")


# Capacity Planning Endpoints
@monitoring_router.get("/capacity/analysis", response_model=SystemCapacityResponse)
async def get_capacity_analysis(
    planning_horizon_days: int = Query(default=7, ge=1, le=90, description="Planning horizon in days")
):
    """
    Get system capacity analysis and recommendations.
    
    Returns current utilization, predictions, and scaling recommendations.
    """
    try:
        performance_engine = await get_performance_intelligence_engine()
        capacity_results = await performance_engine.generate_capacity_plan(
            planning_horizon_days=planning_horizon_days
        )
        
        # Aggregate results
        resource_utilization = {}
        capacity_predictions = {}
        scaling_recommendations = []
        optimization_opportunities = []
        
        for result in capacity_results:
            resource_utilization[result.resource_type] = result.current_utilization
            capacity_predictions[result.resource_type] = {
                "projected_utilization": result.projected_utilization,
                "time_to_threshold_days": result.time_to_threshold_days,
                "threshold": result.capacity_threshold
            }
            
            if result.recommended_actions:
                scaling_recommendations.append({
                    "resource_type": result.resource_type,
                    "actions": result.recommended_actions,
                    "priority": "high" if result.time_to_threshold_days and result.time_to_threshold_days < 7 else "medium"
                })
            
            # Add optimization opportunities
            if result.current_utilization < 0.3:
                optimization_opportunities.append({
                    "resource_type": result.resource_type,
                    "opportunity": "underutilized_resource",
                    "current_utilization": result.current_utilization,
                    "potential_savings": "Consider downsizing or consolidating"
                })
        
        return SystemCapacityResponse(
            timestamp=datetime.utcnow(),
            resource_utilization=resource_utilization,
            capacity_predictions=capacity_predictions,
            scaling_recommendations=scaling_recommendations,
            optimization_opportunities=optimization_opportunities
        )
        
    except Exception as e:
        logger.error("Capacity analysis error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Capacity analysis failed: {str(e)}")


@monitoring_router.get("/capacity/recommendations")
async def get_optimization_recommendations(
    component: Optional[str] = Query(default=None, description="Specific component to analyze")
):
    """
    Get AI-powered performance optimization recommendations.
    
    Returns specific recommendations for improving system performance.
    """
    try:
        performance_engine = await get_performance_intelligence_engine()
        recommendations = await performance_engine.get_performance_optimization_recommendations(
            component=component
        )
        
        return {
            "recommendations": recommendations,
            "component": component,
            "generated_at": datetime.utcnow().isoformat(),
            "total_recommendations": len(recommendations)
        }
        
    except Exception as e:
        logger.error("Optimization recommendations error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Optimization recommendations failed: {str(e)}")


# Unified Observability Endpoint
@monitoring_router.get("/observability/unified")
async def get_unified_observability(
    time_window_minutes: int = Query(default=30, ge=5, le=1440, description="Time window in minutes")
):
    """
    Get unified observability data from all monitoring components.
    
    Returns comprehensive view combining metrics, traces, alerts, and performance data.
    """
    try:
        # Gather data from all monitoring components
        results = await asyncio.gather(
            get_performance_dashboard(time_window_minutes),
            get_active_alerts(limit=10),
            get_trace_analytics(hours=max(1, time_window_minutes // 60)),
            get_capacity_analysis(planning_horizon_days=7),
            return_exceptions=True
        )
        
        performance_data, active_alerts, trace_analytics, capacity_analysis = results
        
        # Handle any exceptions
        errors = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                component_names = ["performance", "alerts", "tracing", "capacity"]
                errors.append(f"{component_names[i]}: {str(result)}")
        
        unified_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "time_window_minutes": time_window_minutes,
            "data_sources": {
                "performance": performance_data if not isinstance(performance_data, Exception) else None,
                "alerts": {
                    "active_alerts": active_alerts if not isinstance(active_alerts, Exception) else [],
                    "total_active": len(active_alerts) if not isinstance(active_alerts, Exception) else 0
                },
                "tracing": trace_analytics if not isinstance(trace_analytics, Exception) else None,
                "capacity": capacity_analysis if not isinstance(capacity_analysis, Exception) else None
            },
            "errors": errors,
            "health_status": "degraded" if errors else "healthy"
        }
        
        return unified_data
        
    except Exception as e:
        logger.error("Unified observability error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Unified observability failed: {str(e)}")


# Background Tasks Endpoint
@monitoring_router.post("/system/refresh")
async def refresh_monitoring_data(background_tasks: BackgroundTasks):
    """
    Trigger refresh of monitoring data.
    
    Forces collection of fresh metrics and analysis data.
    """
    try:
        async def refresh_all_components():
            """Background task to refresh all monitoring components."""
            try:
                # Refresh Prometheus metrics
                prometheus_exporter = await get_prometheus_exporter()
                logger.info("Triggered metrics refresh")
                
                # Refresh performance analysis
                performance_engine = await get_performance_intelligence_engine()
                logger.info("Triggered performance analysis refresh")
                
                # Refresh alerting system evaluation
                alerting_system = await get_intelligent_alerting_system()
                logger.info("Triggered alerting system refresh")
                
                logger.info("All monitoring components refreshed")
                
            except Exception as e:
                logger.error("Background refresh error", error=str(e))
        
        background_tasks.add_task(refresh_all_components)
        
        return {
            "status": "refresh_triggered",
            "message": "Monitoring data refresh initiated in background",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Refresh monitoring data error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Refresh failed: {str(e)}")


# Export router
__all__ = ["monitoring_router"]