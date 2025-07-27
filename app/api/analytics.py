"""
Sleep Analytics API endpoints for monitoring and reporting.

Provides REST API access to comprehensive sleep analytics:
- Real-time performance metrics and dashboard data
- Efficiency reports and trend analysis
- Consolidated reporting and data export
- Performance alerts and health monitoring
"""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Path, Depends
from fastapi.responses import JSONResponse

from ..core.sleep_analytics import (
    get_sleep_analytics_engine, SleepAnalyticsEngine, AnalyticsTimeRange,
    collect_cycle_metrics, get_agent_efficiency_report, get_system_dashboard_data
)
from ..models.agent import Agent
from ..core.database import get_async_session
from ..core.security import get_current_user
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/dashboard")
async def get_analytics_dashboard(
    agent_id: Optional[UUID] = Query(None, description="Agent ID for agent-specific dashboard"),
    analytics_engine: SleepAnalyticsEngine = Depends(get_sleep_analytics_engine)
):
    """
    Get real-time analytics dashboard data.
    
    Returns comprehensive dashboard metrics including:
    - Active sleep cycles
    - Recent performance trends  
    - System health indicators
    - Current efficiency metrics
    - Performance alerts
    """
    try:
        dashboard_data = await analytics_engine.get_real_time_dashboard_data(agent_id)
        
        return {
            "status": "success",
            "data": dashboard_data,
            "agent_id": str(agent_id) if agent_id else None
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")


@router.get("/efficiency/{agent_id}")
async def get_agent_efficiency_metrics(
    agent_id: UUID = Path(..., description="Agent ID"),
    days: int = Query(30, ge=1, le=365, description="Number of days for analysis"),
    analytics_engine: SleepAnalyticsEngine = Depends(get_sleep_analytics_engine),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get comprehensive efficiency metrics for a specific agent.
    
    Returns detailed efficiency analysis including:
    - Token reduction effectiveness
    - Processing time optimization
    - Success rates and reliability
    - Resource utilization efficiency
    - Overall efficiency score
    """
    try:
        # Verify agent exists
        agent = await session.get(Agent, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Generate efficiency report
        time_range = AnalyticsTimeRange(
            start_date=date.today() - timedelta(days=days),
            end_date=date.today()
        )
        
        efficiency_metrics = await analytics_engine.generate_efficiency_report(agent_id, time_range)
        
        return {
            "status": "success",
            "data": {
                "agent_id": str(agent_id),
                "agent_name": agent.name,
                "analysis_period_days": days,
                "metrics": {
                    # Basic metrics
                    "total_cycles": efficiency_metrics.total_cycles,
                    "successful_cycles": efficiency_metrics.successful_cycles,
                    "failed_cycles": efficiency_metrics.failed_cycles,
                    "success_rate": efficiency_metrics.success_rate,
                    
                    # Token reduction metrics
                    "total_tokens_saved": efficiency_metrics.total_tokens_saved,
                    "average_token_reduction": efficiency_metrics.average_token_reduction,
                    "token_reduction_efficiency": efficiency_metrics.token_reduction_efficiency,
                    
                    # Timing metrics
                    "average_consolidation_time_ms": efficiency_metrics.average_consolidation_time_ms,
                    "average_recovery_time_ms": efficiency_metrics.average_recovery_time_ms,
                    "average_cycle_duration_minutes": efficiency_metrics.average_cycle_duration_minutes,
                    
                    # Quality metrics
                    "uptime_percentage": efficiency_metrics.uptime_percentage,
                    "efficiency_score": efficiency_metrics.efficiency_score,
                    "overall_score": efficiency_metrics.calculate_overall_score(),
                    
                    # Resource utilization
                    "cpu_efficiency": efficiency_metrics.cpu_efficiency,
                    "memory_optimization": efficiency_metrics.memory_optimization,
                    "consolidation_ratio": efficiency_metrics.consolidation_ratio,
                    
                    # Reliability metrics
                    "checkpoint_success_rate": efficiency_metrics.checkpoint_success_rate,
                    "recovery_success_rate": efficiency_metrics.recovery_success_rate,
                    "manual_intervention_rate": efficiency_metrics.manual_intervention_rate
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent efficiency metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get efficiency metrics: {str(e)}")


@router.get("/efficiency")
async def get_system_efficiency_metrics(
    days: int = Query(30, ge=1, le=365, description="Number of days for analysis"),
    analytics_engine: SleepAnalyticsEngine = Depends(get_sleep_analytics_engine)
):
    """
    Get system-wide efficiency metrics.
    
    Returns aggregated efficiency analysis across all agents.
    """
    try:
        time_range = AnalyticsTimeRange(
            start_date=date.today() - timedelta(days=days),
            end_date=date.today()
        )
        
        efficiency_metrics = await analytics_engine.generate_efficiency_report(None, time_range)
        
        return {
            "status": "success",
            "data": {
                "analysis_type": "system_wide",
                "analysis_period_days": days,
                "metrics": {
                    "total_cycles": efficiency_metrics.total_cycles,
                    "successful_cycles": efficiency_metrics.successful_cycles,
                    "failed_cycles": efficiency_metrics.failed_cycles,
                    "success_rate": efficiency_metrics.success_rate,
                    "total_tokens_saved": efficiency_metrics.total_tokens_saved,
                    "average_token_reduction": efficiency_metrics.average_token_reduction,
                    "token_reduction_efficiency": efficiency_metrics.token_reduction_efficiency,
                    "average_consolidation_time_ms": efficiency_metrics.average_consolidation_time_ms,
                    "uptime_percentage": efficiency_metrics.uptime_percentage,
                    "efficiency_score": efficiency_metrics.efficiency_score,
                    "overall_score": efficiency_metrics.calculate_overall_score()
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system efficiency metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system efficiency metrics: {str(e)}")


@router.get("/trends/{agent_id}")
async def get_agent_consolidation_trends(
    agent_id: UUID = Path(..., description="Agent ID"),
    days: int = Query(30, ge=7, le=365, description="Number of days for trend analysis"),
    analytics_engine: SleepAnalyticsEngine = Depends(get_sleep_analytics_engine),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get consolidation trends and patterns for a specific agent.
    
    Returns detailed trend analysis including:
    - Daily efficiency patterns
    - Weekly and hourly patterns
    - Performance trends over time
    - Optimization opportunities
    - Predictive analytics
    """
    try:
        # Verify agent exists
        agent = await session.get(Agent, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        trends = await analytics_engine.analyze_consolidation_trends(agent_id, days)
        
        return {
            "status": "success",
            "data": {
                "agent_id": str(agent_id),
                "agent_name": agent.name,
                "analysis_period_days": days,
                "trends": {
                    "daily_efficiency": trends.daily_efficiency,
                    "weekly_patterns": trends.weekly_patterns,
                    "hourly_patterns": trends.hourly_patterns,
                    "efficiency_trend": trends.efficiency_trend,
                    "peak_performance_time": trends.peak_performance_time,
                    "optimization_opportunities": trends.optimization_opportunities,
                    "predicted_efficiency": trends.predicted_efficiency,
                    "confidence_interval": {
                        "lower": trends.confidence_interval[0],
                        "upper": trends.confidence_interval[1]
                    },
                    "recommendation_score": trends.recommendation_score
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting consolidation trends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get consolidation trends: {str(e)}")


@router.get("/trends")
async def get_system_consolidation_trends(
    days: int = Query(30, ge=7, le=365, description="Number of days for trend analysis"),
    analytics_engine: SleepAnalyticsEngine = Depends(get_sleep_analytics_engine)
):
    """
    Get system-wide consolidation trends and patterns.
    
    Returns aggregated trend analysis across all agents.
    """
    try:
        trends = await analytics_engine.analyze_consolidation_trends(None, days)
        
        return {
            "status": "success",
            "data": {
                "analysis_type": "system_wide",
                "analysis_period_days": days,
                "trends": {
                    "daily_efficiency": trends.daily_efficiency,
                    "weekly_patterns": trends.weekly_patterns,
                    "hourly_patterns": trends.hourly_patterns,
                    "efficiency_trend": trends.efficiency_trend,
                    "peak_performance_time": trends.peak_performance_time,
                    "optimization_opportunities": trends.optimization_opportunities,
                    "predicted_efficiency": trends.predicted_efficiency,
                    "confidence_interval": {
                        "lower": trends.confidence_interval[0],
                        "upper": trends.confidence_interval[1]
                    },
                    "recommendation_score": trends.recommendation_score
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system consolidation trends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system consolidation trends: {str(e)}")


@router.get("/cycles/{cycle_id}/metrics")
async def get_cycle_metrics(
    cycle_id: UUID = Path(..., description="Sleep-wake cycle ID"),
    analytics_engine: SleepAnalyticsEngine = Depends(get_sleep_analytics_engine)
):
    """
    Get comprehensive metrics for a specific sleep-wake cycle.
    
    Returns detailed performance analysis for the cycle including:
    - Cycle performance metrics
    - Consolidation job analysis
    - Resource utilization data
    - Checkpoint metrics
    - Token reduction analysis
    """
    try:
        metrics = await analytics_engine.collect_sleep_cycle_metrics(cycle_id)
        
        if "error" in metrics:
            raise HTTPException(status_code=404, detail=f"Cycle not found or error collecting metrics: {metrics['error']}")
        
        return {
            "status": "success",
            "data": metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cycle metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cycle metrics: {str(e)}")


@router.post("/update-daily")
async def update_daily_analytics(
    target_date: Optional[date] = Query(None, description="Date to update (defaults to yesterday)"),
    analytics_engine: SleepAnalyticsEngine = Depends(get_sleep_analytics_engine)
):
    """
    Manually trigger daily analytics update.
    
    Updates aggregated daily analytics for the specified date.
    """
    try:
        success = await analytics_engine.update_daily_analytics(target_date)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update daily analytics")
        
        return {
            "status": "success",
            "message": f"Daily analytics updated for {target_date or 'yesterday'}",
            "target_date": (target_date or (date.today() - timedelta(days=1))).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating daily analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update daily analytics: {str(e)}")


@router.get("/export")
async def export_analytics_report(
    report_type: str = Query("comprehensive", regex="^(comprehensive|summary|trends)$", description="Type of report to export"),
    agent_id: Optional[UUID] = Query(None, description="Agent ID for agent-specific report"),
    days: int = Query(30, ge=1, le=365, description="Number of days for analysis"),
    format: str = Query("json", regex="^(json|csv)$", description="Export format"),
    analytics_engine: SleepAnalyticsEngine = Depends(get_sleep_analytics_engine),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Export comprehensive analytics report.
    
    Generates and exports detailed analytics reports in various formats:
    - comprehensive: Full analysis with all metrics and trends
    - summary: Key metrics and insights
    - trends: Focus on patterns and predictions
    """
    try:
        # Verify agent exists if specified
        if agent_id:
            agent = await session.get(Agent, agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")
        
        time_range = AnalyticsTimeRange(
            start_date=date.today() - timedelta(days=days),
            end_date=date.today()
        )
        
        report_data = await analytics_engine.export_analytics_report(
            report_type=report_type,
            agent_id=agent_id,
            time_range=time_range,
            format=format
        )
        
        if "error" in report_data:
            raise HTTPException(status_code=500, detail=f"Failed to generate report: {report_data['error']}")
        
        # Set appropriate response headers for download
        response_headers = {
            "Content-Disposition": f"attachment; filename=analytics_report_{report_type}_{date.today().isoformat()}.{format}"
        }
        
        if format == "json":
            return JSONResponse(content=report_data, headers=response_headers)
        elif format == "csv":
            # For CSV format, you would convert the data to CSV here
            # This is a simplified implementation
            return JSONResponse(
                content={"message": "CSV export not yet implemented", "data": report_data},
                headers=response_headers
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting analytics report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export report: {str(e)}")


@router.get("/health")
async def get_analytics_health(
    analytics_engine: SleepAnalyticsEngine = Depends(get_sleep_analytics_engine)
):
    """
    Get analytics system health status.
    
    Returns health indicators for the analytics system including:
    - System status
    - Recent error rates
    - Performance indicators
    - Data availability
    """
    try:
        # Get system-wide dashboard data as a health indicator
        dashboard_data = await analytics_engine.get_real_time_dashboard_data()
        
        health_status = {
            "status": "healthy",
            "timestamp": dashboard_data.get("timestamp"),
            "components": {
                "analytics_engine": "healthy",
                "data_collection": "healthy",
                "trend_analysis": "healthy",
                "reporting": "healthy"
            },
            "metrics": {
                "active_cycles": len(dashboard_data.get("active_cycles", [])),
                "system_health": dashboard_data.get("health_indicators", {}).get("status", "unknown"),
                "alerts_count": len(dashboard_data.get("alerts", []))
            }
        }
        
        # Check for any critical alerts
        alerts = dashboard_data.get("alerts", [])
        critical_alerts = [a for a in alerts if a.get("level") == "critical"]
        
        if critical_alerts:
            health_status["status"] = "degraded"
            health_status["components"]["data_collection"] = "degraded"
        
        if dashboard_data.get("health_indicators", {}).get("status") == "critical":
            health_status["status"] = "unhealthy"
        
        return {
            "status": "success",
            "data": health_status
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics health: {e}")
        return {
            "status": "error",
            "data": {
                "status": "unhealthy",
                "error": str(e),
                "components": {
                    "analytics_engine": "error",
                    "data_collection": "error",
                    "trend_analysis": "error",
                    "reporting": "error"
                }
            }
        }


# Quick access endpoints for common operations
@router.get("/quick/agent/{agent_id}/summary")
async def get_agent_quick_summary(
    agent_id: UUID = Path(..., description="Agent ID"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get quick summary metrics for an agent (last 7 days).
    
    Provides a fast overview of key metrics without detailed analysis.
    """
    try:
        # Verify agent exists
        agent = await session.get(Agent, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        # Get quick efficiency report
        efficiency_metrics = await get_agent_efficiency_report(agent_id, days=7)
        
        return {
            "status": "success",
            "data": {
                "agent_id": str(agent_id),
                "agent_name": agent.name,
                "period": "last_7_days",
                "summary": {
                    "total_cycles": efficiency_metrics.total_cycles,
                    "success_rate": efficiency_metrics.success_rate,
                    "token_reduction": efficiency_metrics.average_token_reduction,
                    "efficiency_score": efficiency_metrics.calculate_overall_score(),
                    "status": "excellent" if efficiency_metrics.calculate_overall_score() > 85 else 
                             "good" if efficiency_metrics.calculate_overall_score() > 70 else
                             "needs_attention"
                }
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent quick summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent summary: {str(e)}")


@router.get("/quick/system/status")
async def get_system_quick_status():
    """
    Get quick system status overview.
    
    Provides a fast overview of system health and performance.
    """
    try:
        dashboard_data = await get_system_dashboard_data()
        
        # Extract key status indicators
        health_indicators = dashboard_data.get("health_indicators", {})
        current_metrics = dashboard_data.get("current_metrics", {})
        alerts = dashboard_data.get("alerts", [])
        
        status = "healthy"
        if health_indicators.get("status") == "critical":
            status = "critical"
        elif health_indicators.get("status") == "warning" or len(alerts) > 0:
            status = "warning"
        
        return {
            "status": "success",
            "data": {
                "system_status": status,
                "active_cycles": len(dashboard_data.get("active_cycles", [])),
                "error_rate_24h": health_indicators.get("error_rate", 0.0),
                "average_efficiency": health_indicators.get("average_efficiency", 0.0),
                "meets_token_target": current_metrics.get("meets_target", False),
                "alerts_count": len(alerts),
                "critical_alerts": len([a for a in alerts if a.get("level") == "critical"])
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system quick status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")