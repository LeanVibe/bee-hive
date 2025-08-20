"""
Monitoring and Reporting API endpoints for sleep-wake system observability.

Provides comprehensive monitoring, alerting, and reporting capabilities
for the automated sleep-wake management system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Query, Path, status, BackgroundTasks
from pydantic import BaseModel, Field, validator

from ..core.sleep_analytics import get_sleep_analytics_engine
from ..core.simple_orchestrator import SimpleOrchestrator, create_simple_orchestrator
from ..core.recovery_manager import get_recovery_manager
from ..core.sleep_wake_manager import get_sleep_wake_manager
from ..core.security import get_current_user, require_analytics_access


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/monitoring", tags=["Monitoring & Reporting"])


# Request/Response Models
class AlertConfigRequest(BaseModel):
    """Request model for alert configuration."""
    alert_type: str = Field(..., description="Type of alert")
    threshold_value: float = Field(..., description="Threshold value for triggering")
    agent_id: Optional[UUID] = Field(
        default=None, 
        description="Agent ID for agent-specific alerts"
    )
    notification_channels: List[str] = Field(
        default=["log", "webhook"],
        description="Notification channels for alerts"
    )
    cooldown_minutes: int = Field(
        default=15, 
        ge=1, 
        le=1440,
        description="Cooldown period between alerts"
    )
    enabled: bool = Field(default=True, description="Enable/disable alert")
    
    @validator('alert_type')
    def validate_alert_type(cls, v):
        allowed_types = [
            "efficiency_drop", "recovery_failure", "excessive_sleep_time",
            "consolidation_failure", "resource_threshold", "system_health"
        ]
        if v not in allowed_types:
            raise ValueError(f"alert_type must be one of {allowed_types}")
        return v


class ReportRequest(BaseModel):
    """Request model for report generation."""
    report_type: str = Field(..., description="Type of report to generate")
    time_range_hours: int = Field(
        default=24, 
        ge=1, 
        le=8760,
        description="Time range for report in hours"
    )
    agent_filter: Optional[List[UUID]] = Field(
        default=None,
        description="Filter by specific agent IDs"
    )
    include_raw_data: bool = Field(
        default=False,
        description="Include raw data in report"
    )
    export_format: str = Field(
        default="json",
        description="Export format for report"
    )
    
    @validator('report_type')
    def validate_report_type(cls, v):
        allowed_types = [
            "efficiency_summary", "performance_analysis", "resource_utilization",
            "error_summary", "optimization_impact", "comparative_analysis"
        ]
        if v not in allowed_types:
            raise ValueError(f"report_type must be one of {allowed_types}")
        return v
    
    @validator('export_format')
    def validate_export_format(cls, v):
        if v not in ["json", "csv", "pdf"]:
            raise ValueError("export_format must be json, csv, or pdf")
        return v


class DashboardConfig(BaseModel):
    """Configuration model for dashboard customization."""
    widget_layout: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Dashboard widget layout configuration"
    )
    refresh_interval_seconds: int = Field(
        default=30, 
        ge=5, 
        le=300,
        description="Dashboard refresh interval"
    )
    default_time_range_hours: int = Field(
        default=24, 
        ge=1, 
        le=168,
        description="Default time range for dashboard"
    )
    alert_display_limit: int = Field(
        default=10, 
        ge=1, 
        le=100,
        description="Number of alerts to display"
    )


# Real-time Monitoring Endpoints
@router.get("/realtime/status")
async def get_realtime_system_status(
    include_detailed_metrics: bool = Query(False, description="Include detailed metrics"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get real-time system status with live monitoring data.
    
    Features:
    - Live system health monitoring
    - Real-time performance metrics
    - Active operation tracking
    - Resource utilization monitoring
    """
    try:
        sleep_manager = await get_sleep_wake_manager()
        analytics_engine = get_sleep_analytics_engine()
        orchestrator = create_simple_orchestrator()
        
        # Get core system status
        system_status = await sleep_manager.get_system_status()
        
        # Get real-time analytics
        realtime_analytics = await analytics_engine.get_realtime_metrics()
        
        # Get orchestrator status
        orchestrator_status = await orchestrator.get_system_health()
        
        # Compile comprehensive status
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": {
                "overall_healthy": system_status["system_healthy"],
                "components": {
                    "sleep_manager": system_status["system_healthy"],
                    "orchestrator": orchestrator_status.get("healthy", False),
                    "analytics": realtime_analytics.get("healthy", True)
                }
            },
            "active_operations": {
                "total": system_status["active_operations"],
                "sleep_cycles": len([
                    agent for agent in system_status["agents"].values()
                    if agent["sleep_state"] in ["SLEEPING", "PREPARING_SLEEP"]
                ]),
                "wake_cycles": len([
                    agent for agent in system_status["agents"].values()
                    if agent["sleep_state"] == "PREPARING_WAKE"
                ]),
                "error_states": len([
                    agent for agent in system_status["agents"].values()
                    if agent["sleep_state"] == "ERROR"
                ])
            },
            "performance_summary": {
                "average_sleep_time_ms": system_status["metrics"].get("average_sleep_time_ms", 0),
                "average_wake_time_ms": system_status["metrics"].get("average_wake_time_ms", 0),
                "success_rate": _calculate_success_rate(system_status["metrics"]),
                "recent_throughput": realtime_analytics.get("throughput", 0)
            }
        }
        
        if include_detailed_metrics:
            status["detailed_metrics"] = {
                "system_metrics": system_status["metrics"],
                "realtime_analytics": realtime_analytics,
                "orchestrator_metrics": orchestrator_status.get("metrics", {}),
                "agent_details": system_status["agents"]
            }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting realtime status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get realtime status: {str(e)}"
        )


@router.get("/alerts/active")
async def get_active_alerts(
    severity_filter: Optional[str] = Query(None, description="Filter by severity"),
    limit: int = Query(50, ge=1, le=200, description="Maximum alerts to return"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get currently active alerts and warnings.
    
    Features:
    - Real-time alert monitoring
    - Severity-based filtering
    - Alert aging and prioritization
    - Action recommendations
    """
    try:
        analytics_engine = get_sleep_analytics_engine()
        
        # Get active alerts
        active_alerts = await analytics_engine.get_active_alerts(
            severity_filter=severity_filter,
            limit=limit
        )
        
        # Add alert context and recommendations
        enhanced_alerts = []
        for alert in active_alerts:
            enhanced_alert = alert.copy()
            
            # Add contextual information
            enhanced_alert["context"] = await _get_alert_context(alert)
            
            # Add recommended actions
            enhanced_alert["recommendations"] = await _get_alert_recommendations(alert)
            
            # Calculate alert age
            alert_time = datetime.fromisoformat(alert["timestamp"])
            enhanced_alert["age_minutes"] = (datetime.utcnow() - alert_time).total_seconds() / 60
            
            enhanced_alerts.append(enhanced_alert)
        
        return {
            "active_alerts": enhanced_alerts,
            "total_count": len(enhanced_alerts),
            "severity_counts": _count_alerts_by_severity(enhanced_alerts),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting active alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get active alerts: {str(e)}"
        )


@router.post("/alerts/configure")
async def configure_alert(
    request: AlertConfigRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Configure alert thresholds and notification settings.
    
    Features:
    - Flexible alert configuration
    - Multi-channel notifications
    - Intelligent threshold recommendations
    - Alert suppression and escalation
    """
    try:
        analytics_engine = get_sleep_analytics_engine()
        
        # Validate threshold value
        validation_result = await analytics_engine.validate_alert_threshold(
            alert_type=request.alert_type,
            threshold_value=request.threshold_value,
            agent_id=request.agent_id
        )
        
        if not validation_result["valid"]:
            return {
                "success": False,
                "message": "Invalid threshold configuration",
                "validation_errors": validation_result["errors"],
                "recommendations": validation_result.get("recommendations", [])
            }
        
        # Configure alert
        alert_config = await analytics_engine.configure_alert(
            alert_type=request.alert_type,
            threshold_value=request.threshold_value,
            agent_id=request.agent_id,
            notification_channels=request.notification_channels,
            cooldown_minutes=request.cooldown_minutes,
            enabled=request.enabled
        )
        
        return {
            "success": True,
            "message": f"Alert configuration updated for {request.alert_type}",
            "alert_id": alert_config["alert_id"],
            "configuration": alert_config,
            "estimated_alert_frequency": validation_result.get("estimated_frequency")
        }
        
    except Exception as e:
        logger.error(f"Error configuring alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Alert configuration failed: {str(e)}"
        )


# Reporting and Analytics Endpoints
@router.post("/reports/generate")
async def generate_report(
    request: ReportRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate comprehensive system reports.
    
    Features:
    - Multiple report types and formats
    - Scheduled report generation
    - Comparative analysis across time periods
    - Automated report distribution
    """
    try:
        analytics_engine = get_sleep_analytics_engine()
        
        # Validate report parameters
        validation = await analytics_engine.validate_report_parameters(
            report_type=request.report_type,
            time_range_hours=request.time_range_hours,
            agent_filter=request.agent_filter
        )
        
        if not validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid report parameters: {validation['errors']}"
            )
        
        # Generate report ID
        report_id = f"report_{request.report_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Start report generation in background
        background_tasks.add_task(
            _generate_report_background,
            report_id,
            request,
            analytics_engine
        )
        
        return {
            "report_id": report_id,
            "status": "generating",
            "report_type": request.report_type,
            "estimated_completion": datetime.utcnow() + timedelta(minutes=5),
            "download_url": f"/api/v1/monitoring/reports/{report_id}/download",
            "status_url": f"/api/v1/monitoring/reports/{report_id}/status"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}"
        )


@router.get("/reports/{report_id}/status")
async def get_report_status(
    report_id: str = Path(..., description="Report ID"),
    current_user: dict = Depends(get_current_user)
):
    """Get the status of a report generation task."""
    try:
        analytics_engine = get_sleep_analytics_engine()
        
        report_status = await analytics_engine.get_report_status(report_id)
        
        return {
            "report_id": report_id,
            "status": report_status["status"],
            "progress_percent": report_status.get("progress", 0),
            "generated_at": report_status.get("generated_at"),
            "error_message": report_status.get("error"),
            "download_ready": report_status["status"] == "completed",
            "expires_at": report_status.get("expires_at")
        }
        
    except Exception as e:
        logger.error(f"Error getting report status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get report status: {str(e)}"
        )


@router.get("/dashboard/data")
async def get_dashboard_data(
    time_range_hours: int = Query(24, ge=1, le=168, description="Time range for dashboard"),
    refresh_cache: bool = Query(False, description="Force cache refresh"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get comprehensive dashboard data for monitoring interface.
    
    Features:
    - Real-time metrics aggregation
    - Performance trend analysis
    - Resource utilization tracking
    - Alert summary and prioritization
    """
    try:
        analytics_engine = get_sleep_analytics_engine()
        sleep_manager = await get_sleep_wake_manager()
        
        # Get dashboard data
        dashboard_data = await analytics_engine.get_dashboard_data(
            time_range=timedelta(hours=time_range_hours),
            refresh_cache=refresh_cache
        )
        
        # Add real-time system status
        system_status = await sleep_manager.get_system_status()
        dashboard_data["realtime_status"] = system_status
        
        # Add trend analysis
        trends = await analytics_engine.get_performance_trends(
            time_range=timedelta(hours=time_range_hours)
        )
        dashboard_data["trends"] = trends
        
        # Add active alerts summary
        active_alerts = await analytics_engine.get_active_alerts(limit=10)
        dashboard_data["alerts_summary"] = {
            "total_active": len(active_alerts),
            "by_severity": _count_alerts_by_severity(active_alerts),
            "recent_alerts": active_alerts[:5]
        }
        
        return {
            "dashboard_id": f"dashboard_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "time_range_hours": time_range_hours,
            "data": dashboard_data,
            "generated_at": datetime.utcnow().isoformat(),
            "cache_status": "refreshed" if refresh_cache else "cached"
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dashboard data retrieval failed: {str(e)}"
        )


@router.post("/dashboard/configure")
async def configure_dashboard(
    config: DashboardConfig,
    current_user: dict = Depends(get_current_user)
):
    """
    Configure dashboard layout and settings.
    
    Features:
    - Customizable widget layouts
    - Personalized metric selection
    - Automatic refresh configuration
    - Multi-user dashboard profiles
    """
    try:
        analytics_engine = get_sleep_analytics_engine()
        
        # Validate dashboard configuration
        validation = await analytics_engine.validate_dashboard_config(config.dict())
        
        if not validation["valid"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid dashboard configuration: {validation['errors']}"
            )
        
        # Apply configuration
        config_result = await analytics_engine.update_dashboard_config(
            user_id=current_user["id"],
            config=config.dict()
        )
        
        return {
            "success": True,
            "message": "Dashboard configuration updated successfully",
            "config_id": config_result["config_id"],
            "applied_settings": config_result["settings"],
            "warnings": config_result.get("warnings", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error configuring dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dashboard configuration failed: {str(e)}"
        )


# System Diagnostics Endpoints
@router.get("/diagnostics/performance")
async def get_performance_diagnostics(
    deep_analysis: bool = Query(False, description="Perform deep performance analysis"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get comprehensive performance diagnostics.
    
    Features:
    - Performance bottleneck identification
    - Resource utilization analysis
    - Optimization recommendation engine
    - Predictive performance modeling
    """
    try:
        analytics_engine = get_sleep_analytics_engine()
        sleep_manager = await get_sleep_wake_manager()
        
        # Get basic performance metrics
        performance_metrics = await analytics_engine.get_performance_diagnostics()
        
        # Get system optimization results
        optimization_results = await sleep_manager.optimize_performance()
        
        diagnostics = {
            "timestamp": datetime.utcnow().isoformat(),
            "performance_metrics": performance_metrics,
            "optimization_results": optimization_results,
            "bottlenecks": await _identify_performance_bottlenecks(performance_metrics),
            "recommendations": await _generate_performance_recommendations(performance_metrics)
        }
        
        if deep_analysis:
            # Perform deep analysis
            deep_analysis_result = await analytics_engine.perform_deep_analysis()
            diagnostics["deep_analysis"] = deep_analysis_result
            
            # Add predictive modeling
            predictive_analysis = await analytics_engine.predict_performance_trends()
            diagnostics["predictive_analysis"] = predictive_analysis
        
        return diagnostics
        
    except Exception as e:
        logger.error(f"Error getting performance diagnostics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Performance diagnostics failed: {str(e)}"
        )


# Helper Functions
async def _generate_report_background(
    report_id: str,
    request: ReportRequest,
    analytics_engine: Any
) -> None:
    """Generate report in background task."""
    try:
        await analytics_engine.generate_report_async(
            report_id=report_id,
            report_type=request.report_type,
            time_range_hours=request.time_range_hours,
            agent_filter=request.agent_filter,
            include_raw_data=request.include_raw_data,
            export_format=request.export_format
        )
    except Exception as e:
        logger.error(f"Background report generation failed: {e}")
        await analytics_engine.mark_report_failed(report_id, str(e))


async def _get_alert_context(alert: Dict[str, Any]) -> Dict[str, Any]:
    """Get contextual information for an alert."""
    return {
        "affected_components": ["sleep_manager"],  # Placeholder
        "related_metrics": {},  # Placeholder
        "historical_occurrences": 0  # Placeholder
    }


async def _get_alert_recommendations(alert: Dict[str, Any]) -> List[str]:
    """Get recommended actions for an alert."""
    return [
        "Review system logs for detailed error information",
        "Check agent health status",
        "Consider triggering recovery procedures"
    ]


def _count_alerts_by_severity(alerts: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count alerts by severity level."""
    counts = {"critical": 0, "warning": 0, "info": 0}
    for alert in alerts:
        severity = alert.get("severity", "info")
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def _calculate_success_rate(metrics: Dict[str, Any]) -> float:
    """Calculate overall success rate from metrics."""
    try:
        total_operations = (
            metrics.get("total_sleep_cycles", 0) +
            metrics.get("total_wake_cycles", 0)
        )
        successful_operations = (
            metrics.get("successful_sleep_cycles", 0) +
            metrics.get("successful_wake_cycles", 0)
        )
        
        if total_operations == 0:
            return 1.0
        
        return successful_operations / total_operations
    except Exception:
        return 0.0


async def _identify_performance_bottlenecks(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify performance bottlenecks from metrics."""
    bottlenecks = []
    
    # Check sleep cycle performance
    avg_sleep_time = metrics.get("average_sleep_time_ms", 0)
    if avg_sleep_time > 5000:  # More than 5 seconds
        bottlenecks.append({
            "type": "sleep_performance",
            "severity": "medium",
            "description": f"Average sleep time is {avg_sleep_time}ms",
            "impact": "Reduced system responsiveness"
        })
    
    # Check wake cycle performance
    avg_wake_time = metrics.get("average_wake_time_ms", 0)
    if avg_wake_time > 2000:  # More than 2 seconds
        bottlenecks.append({
            "type": "wake_performance",
            "severity": "medium",
            "description": f"Average wake time is {avg_wake_time}ms",
            "impact": "Delayed agent availability"
        })
    
    return bottlenecks


async def _generate_performance_recommendations(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate performance improvement recommendations."""
    recommendations = []
    
    # Analyze consolidation efficiency
    if metrics.get("total_consolidations", 0) > 0:
        consolidation_rate = metrics.get("successful_consolidations", 0) / metrics["total_consolidations"]
        if consolidation_rate < 0.9:
            recommendations.append({
                "priority": "high",
                "category": "consolidation",
                "title": "Improve Consolidation Success Rate",
                "description": f"Current rate: {consolidation_rate:.1%}",
                "actions": [
                    "Review consolidation algorithms",
                    "Check resource availability during consolidation",
                    "Optimize memory usage patterns"
                ]
            })
    
    return recommendations