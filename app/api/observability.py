"""
Observability API endpoints for LeanVibe Agent Hive 2.0

Provides REST API access to enterprise observability metrics,
dashboard data, and real-time monitoring capabilities.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel

from ..core.enterprise_observability import (
    EnterpriseObservability,
    DevelopmentMetrics,
    MetricCategory,
    create_enterprise_observability
)
from ..core.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/observability", tags=["observability"])

# Global observability instance (would be properly managed in production)
_observability_instance: Optional[EnterpriseObservability] = None


async def get_observability() -> EnterpriseObservability:
    """Get or create observability instance."""
    global _observability_instance
    if _observability_instance is None:
        _observability_instance = await create_enterprise_observability()
    return _observability_instance


class DevelopmentMetricsRequest(BaseModel):
    """Request model for recording development metrics."""
    task_id: str
    agent_type: str
    generation_time_seconds: float
    execution_time_seconds: float
    total_time_seconds: float
    code_length: int
    success: bool
    security_level: str
    quality_score: float


class AlertThresholdUpdate(BaseModel):
    """Request model for updating alert thresholds."""
    metric_name: str
    threshold_value: float
    severity: str = "warning"


@router.get("/health")
async def observability_health():
    """Health check for observability system."""
    try:
        observability = await get_observability()
        return {
            "status": "healthy",
            "metrics_server": f"http://localhost:{observability.metrics_port}/metrics",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Observability system unavailable: {str(e)}")


@router.get("/dashboard/enterprise")
async def get_enterprise_dashboard():
    """Get comprehensive enterprise dashboard data."""
    try:
        observability = await get_observability()
        dashboard_data = await observability.get_enterprise_dashboard_data()
        
        return {
            "status": "success",
            "data": dashboard_data,
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate dashboard data: {str(e)}")


@router.get("/dashboard/roi")
async def get_roi_dashboard():
    """Get ROI-focused dashboard data for executives."""
    try:
        observability = await get_observability()
        dashboard_data = await observability.get_enterprise_dashboard_data()
        
        # Extract ROI-specific metrics
        roi_data = {
            "summary": dashboard_data.get("summary", {}),
            "roi_metrics": dashboard_data.get("roi_metrics", {}),
            "cost_savings": {
                "hourly": dashboard_data.get("roi_metrics", {}).get("cost_savings_per_hour", 0),
                "daily": dashboard_data.get("roi_metrics", {}).get("cost_savings_per_hour", 0) * 24,
                "monthly": dashboard_data.get("roi_metrics", {}).get("cost_savings_per_hour", 0) * 24 * 30,
                "annual": dashboard_data.get("roi_metrics", {}).get("cost_savings_per_hour", 0) * 24 * 365
            },
            "productivity": {
                "velocity_multiplier": dashboard_data.get("roi_metrics", {}).get("development_velocity_improvement", 1.0),
                "hours_saved": dashboard_data.get("roi_metrics", {}).get("developer_hours_saved", 0),
                "quality_improvement": dashboard_data.get("roi_metrics", {}).get("quality_improvement_score", 0)
            }
        }
        
        return {
            "status": "success", 
            "data": roi_data,
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate ROI dashboard: {str(e)}")


@router.post("/metrics/development")
async def record_development_metrics(
    metrics: DevelopmentMetricsRequest,
    current_user: dict = Depends(get_current_user)
):
    """Record autonomous development metrics."""
    try:
        observability = await get_observability()
        
        development_metrics = DevelopmentMetrics(
            task_id=metrics.task_id,
            agent_type=metrics.agent_type,
            generation_time_seconds=metrics.generation_time_seconds,
            execution_time_seconds=metrics.execution_time_seconds,
            total_time_seconds=metrics.total_time_seconds,
            code_length=metrics.code_length,
            success=metrics.success,
            security_level=metrics.security_level,
            quality_score=metrics.quality_score,
            created_at=datetime.utcnow()
        )
        
        await observability.record_autonomous_development(development_metrics)
        
        return {
            "status": "success",
            "message": "Development metrics recorded",
            "task_id": metrics.task_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record metrics: {str(e)}")


@router.post("/metrics/cli-agent")
async def record_cli_agent_metrics(
    agent_type: str,
    response_time: float,
    success: bool,
    version: str = "unknown",
    current_user: dict = Depends(get_current_user)
):
    """Record CLI agent performance metrics."""
    try:
        observability = await get_observability()
        
        await observability.record_cli_agent_performance(
            agent_type=agent_type,
            response_time=response_time,
            success=success,
            version=version
        )
        
        return {
            "status": "success",
            "message": "CLI agent metrics recorded",
            "agent_type": agent_type
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record CLI agent metrics: {str(e)}")


@router.post("/metrics/security")
async def record_security_event(
    violation_type: str,
    severity: str = "medium",
    current_user: dict = Depends(get_current_user)
):
    """Record security violation or event."""
    try:
        observability = await get_observability()
        
        await observability.record_security_event(
            violation_type=violation_type,
            severity=severity
        )
        
        return {
            "status": "success",
            "message": "Security event recorded",
            "violation_type": violation_type,
            "severity": severity
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record security event: {str(e)}")


@router.post("/metrics/sandbox")
async def record_sandbox_execution(
    language: str,
    success: bool,
    security_level: str,
    execution_time: float,
    current_user: dict = Depends(get_current_user)
):
    """Record sandbox execution metrics."""
    try:
        observability = await get_observability()
        
        await observability.record_sandbox_execution(
            language=language,
            success=success,
            security_level=security_level,
            execution_time=execution_time
        )
        
        return {
            "status": "success",
            "message": "Sandbox execution metrics recorded",
            "language": language,
            "success": success
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record sandbox metrics: {str(e)}")


@router.get("/metrics/prometheus")
async def get_prometheus_endpoint():
    """Get Prometheus metrics endpoint information."""
    try:
        observability = await get_observability()
        return {
            "prometheus_endpoint": f"http://localhost:{observability.metrics_port}/metrics",
            "scrape_interval": "30s",
            "available_metrics": [
                "autonomous_development_duration_seconds",
                "autonomous_development_tasks_total", 
                "code_generation_time_seconds",
                "code_execution_time_seconds",
                "cli_agent_availability",
                "cli_agent_response_time_seconds",
                "security_violations_total",
                "sandbox_executions_total",
                "development_velocity_multiplier",
                "cost_savings_usd_per_hour",
                "developer_hours_saved_total",
                "code_quality_score",
                "system_health_score"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Prometheus info: {str(e)}")


@router.get("/alerts/status")
async def get_alert_status():
    """Get current alert status and active alerts."""
    try:
        observability = await get_observability()
        dashboard_data = await observability.get_enterprise_dashboard_data()
        
        # Check alert conditions
        alerts = await observability.alert_manager.check_thresholds(dashboard_data)
        
        return {
            "status": "success",
            "active_alerts": alerts,
            "alert_count": len(alerts),
            "system_status": "healthy" if len(alerts) == 0 else "warning",
            "checked_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check alert status: {str(e)}")


@router.put("/alerts/thresholds")
async def update_alert_thresholds(
    threshold_updates: List[AlertThresholdUpdate],
    current_user: dict = Depends(get_current_user)
):
    """Update alert thresholds."""
    try:
        observability = await get_observability()
        
        for update in threshold_updates:
            if update.metric_name in observability.alert_manager.alert_thresholds:
                observability.alert_manager.alert_thresholds[update.metric_name] = update.threshold_value
        
        return {
            "status": "success",
            "message": f"Updated {len(threshold_updates)} alert thresholds",
            "updated_thresholds": [u.metric_name for u in threshold_updates]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update alert thresholds: {str(e)}")


@router.get("/system/health")
async def get_system_health():
    """Get comprehensive system health status."""
    try:
        observability = await get_observability()
        dashboard_data = await observability.get_enterprise_dashboard_data()
        
        # Calculate overall health score
        summary = dashboard_data.get("summary", {})
        success_rate = summary.get("success_rate", 0)
        avg_response_time = summary.get("avg_total_time", 0)
        
        # Health scoring
        health_factors = []
        
        # Success rate factor (80% weight)
        if success_rate >= 0.95:
            health_factors.append(1.0)
        elif success_rate >= 0.8:
            health_factors.append(0.8)
        else:
            health_factors.append(0.5)
        
        # Response time factor (20% weight)
        if avg_response_time <= 30:
            health_factors.append(1.0)
        elif avg_response_time <= 60:
            health_factors.append(0.7)
        else:
            health_factors.append(0.4)
        
        overall_health = sum(health_factors) / len(health_factors)
        
        # Update system health metric
        await observability.update_system_health("overall", overall_health)
        
        return {
            "status": "success",
            "health_score": overall_health,
            "health_grade": "excellent" if overall_health >= 0.9 else "good" if overall_health >= 0.7 else "needs_attention",
            "components": {
                "success_rate": success_rate,
                "avg_response_time": avg_response_time,
                "active_agents": len([a for a in dashboard_data.get("agent_performance", {}).values() if a.get("count", 0) > 0])
            },
            "checked_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system health: {str(e)}")


@router.get("/reports/performance")
async def get_performance_report(
    period_hours: int = Query(24, description="Report period in hours"),
    include_trends: bool = Query(True, description="Include trend analysis")
):
    """Generate performance report for specified period."""
    try:
        observability = await get_observability()
        dashboard_data = await observability.get_enterprise_dashboard_data()
        
        report = {
            "period": f"Last {period_hours} hours",
            "generated_at": datetime.utcnow().isoformat(),
            "summary": dashboard_data.get("summary", {}),
            "roi_metrics": dashboard_data.get("roi_metrics", {}),
            "agent_performance": dashboard_data.get("agent_performance", {}),
            "security_status": dashboard_data.get("security", {}),
            "recommendations": []
        }
        
        # Add recommendations based on metrics
        summary = dashboard_data.get("summary", {})
        if summary.get("success_rate", 1.0) < 0.8:
            report["recommendations"].append({
                "priority": "high",
                "category": "performance",
                "message": "Success rate below 80% - investigate agent failures",
                "metric": "success_rate",
                "current_value": summary.get("success_rate", 0)
            })
        
        if summary.get("avg_total_time", 0) > 60:
            report["recommendations"].append({
                "priority": "medium", 
                "category": "performance",
                "message": "Average response time above 60s - consider optimization",
                "metric": "response_time",
                "current_value": summary.get("avg_total_time", 0)
            })
        
        return {
            "status": "success",
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate performance report: {str(e)}")


# Export router for inclusion in main app
__all__ = ["router"]