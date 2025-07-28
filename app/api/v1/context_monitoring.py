"""
Context Engine Monitoring API endpoints.

Provides comprehensive context performance monitoring, analytics,
and optimization endpoints for the LeanVibe Agent Hive.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from ...core.database import get_async_session
from ...core.context_performance_monitor import (
    get_context_performance_monitor,
    ContextPerformanceMonitor,
    ContextOperation,
    PerformanceIssueType
)
from ...core.context_analytics import get_context_analytics_manager
from ...core.search_analytics import get_search_analytics
from ...observability.prometheus_exporter import metrics_exporter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/context-monitoring", tags=["Context Monitoring"])


# Request/Response Models
class ContextOperationRecord(BaseModel):
    """Model for recording context operations."""
    operation: str = Field(..., description="Type of operation (create, read, update, delete, search)")
    context_id: Optional[str] = Field(None, description="Context ID if applicable")
    context_type: Optional[str] = Field(None, description="Context type")
    duration_ms: Optional[float] = Field(None, description="Operation duration in milliseconds")
    success: bool = Field(True, description="Whether operation was successful")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class SearchPerformanceRecord(BaseModel):
    """Model for recording search performance."""
    search_type: str = Field(..., description="Type of search performed")
    query: str = Field(..., description="Search query")
    result_count: int = Field(..., description="Number of results returned")
    latency_ms: float = Field(..., description="Search latency in milliseconds")
    quality_score: Optional[float] = Field(None, description="Search quality score (0-1)")
    cache_hit: bool = Field(False, description="Whether search hit cache")


class EmbeddingAPIRecord(BaseModel):
    """Model for recording embedding API calls."""
    provider: str = Field(..., description="API provider (openai, anthropic, etc.)")
    model: str = Field(..., description="Model used")
    tokens: int = Field(..., description="Number of tokens processed")
    duration_ms: float = Field(..., description="API call duration in milliseconds")
    cost_usd: float = Field(..., description="Cost in USD")
    success: bool = Field(True, description="Whether API call was successful")


class PerformanceSummaryResponse(BaseModel):
    """Performance summary response model."""
    time_window_hours: int
    timestamp: str
    search_performance: Dict[str, Any]
    api_costs: Dict[str, Any]
    cache_performance: Dict[str, Any]
    active_alerts: int
    recommendations_count: int
    context_metrics: Dict[str, Any]


class CostAnalysisResponse(BaseModel):
    """Cost analysis response model."""
    time_window_hours: int
    total_cost_usd: float
    projected_daily_cost_usd: float
    projected_monthly_cost_usd: float
    costs_by_provider: Dict[str, float]
    costs_by_model: Dict[str, float]
    hourly_trend: Dict[str, float]
    total_api_calls: int
    avg_cost_per_call: float
    cost_alert: Optional[Dict[str, Any]] = None


class OptimizationRecommendationResponse(BaseModel):
    """Optimization recommendation response model."""
    recommendation_id: str
    category: str
    title: str
    description: str
    expected_improvement: float
    implementation_difficulty: str
    priority: int
    estimated_impact: str
    implementation_steps: List[str]
    created_at: str


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    components: Dict[str, Dict[str, Any]]
    overall_health: str


@router.post("/operations/record")
async def record_context_operation(
    operation_record: ContextOperationRecord,
    monitor: ContextPerformanceMonitor = Depends(get_context_performance_monitor)
) -> Dict[str, str]:
    """
    Record a context operation for performance monitoring.
    
    This endpoint allows agents and components to report context operations
    for comprehensive performance tracking and analytics.
    """
    try:
        # Convert operation string to enum
        try:
            operation = ContextOperation(operation_record.operation.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid operation type: {operation_record.operation}"
            )
        
        # Convert context_type string to enum if provided
        context_type = None
        if operation_record.context_type:
            from ...models.context import ContextType
            try:
                context_type = ContextType(operation_record.context_type.upper())
            except ValueError:
                # Use None if invalid context type provided
                pass
        
        await monitor.record_operation(
            operation=operation,
            context_id=operation_record.context_id,
            context_type=context_type,
            duration_ms=operation_record.duration_ms,
            success=operation_record.success,
            metadata=operation_record.metadata
        )
        
        return {"status": "recorded", "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Failed to record context operation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/record")
async def record_search_performance(
    search_record: SearchPerformanceRecord,
    monitor: ContextPerformanceMonitor = Depends(get_context_performance_monitor)
) -> Dict[str, str]:
    """
    Record search performance metrics.
    
    Used by the search system to report performance metrics for
    monitoring and optimization purposes.
    """
    try:
        await monitor.record_search_performance(
            search_type=search_record.search_type,
            query=search_record.query,
            result_count=search_record.result_count,
            latency_ms=search_record.latency_ms,
            quality_score=search_record.quality_score,
            cache_hit=search_record.cache_hit
        )
        
        return {"status": "recorded", "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Failed to record search performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embedding/record")
async def record_embedding_api_call(
    api_record: EmbeddingAPIRecord,
    monitor: ContextPerformanceMonitor = Depends(get_context_performance_monitor)
) -> Dict[str, str]:
    """
    Record embedding API call metrics for cost monitoring.
    
    Used by the embedding service to track API usage and costs.
    """
    try:
        await monitor.record_embedding_api_call(
            provider=api_record.provider,
            model=api_record.model,
            tokens=api_record.tokens,
            duration_ms=api_record.duration_ms,
            cost_usd=api_record.cost_usd,
            success=api_record.success
        )
        
        return {"status": "recorded", "timestamp": datetime.utcnow().isoformat()}
        
    except Exception as e:
        logger.error(f"Failed to record embedding API call: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance/summary", response_model=PerformanceSummaryResponse)
async def get_performance_summary(
    time_window_hours: int = Query(24, ge=1, le=168, description="Time window in hours (1-168)"),
    monitor: ContextPerformanceMonitor = Depends(get_context_performance_monitor)
) -> PerformanceSummaryResponse:
    """
    Get comprehensive performance summary for the specified time window.
    
    Provides aggregated metrics across all context engine components
    including search performance, API costs, cache performance, and alerts.
    """
    try:
        summary = await monitor.get_performance_summary(time_window_hours)
        
        if "error" in summary:
            raise HTTPException(status_code=500, detail=summary["error"])
        
        return PerformanceSummaryResponse(**summary)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get performance summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/costs/analysis", response_model=CostAnalysisResponse)
async def get_cost_analysis(
    time_window_hours: int = Query(24, ge=1, le=168, description="Time window in hours (1-168)"),
    monitor: ContextPerformanceMonitor = Depends(get_context_performance_monitor)
) -> CostAnalysisResponse:
    """
    Get detailed cost analysis for embedding API usage.
    
    Provides cost breakdown by provider and model, cost trends,
    and optimization recommendations.
    """
    try:
        analysis = await monitor.get_cost_analysis(time_window_hours)
        
        if "error" in analysis:
            raise HTTPException(status_code=500, detail=analysis["error"])
        
        return CostAnalysisResponse(**analysis)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get cost analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/optimization/recommendations", response_model=List[OptimizationRecommendationResponse])
async def get_optimization_recommendations(
    monitor: ContextPerformanceMonitor = Depends(get_context_performance_monitor)
) -> List[OptimizationRecommendationResponse]:
    """
    Get current optimization recommendations based on performance analysis.
    
    Returns intelligent recommendations for improving context engine
    performance, reducing costs, and optimizing search quality.
    """
    try:
        recommendations = await monitor.get_optimization_recommendations()
        
        return [
            OptimizationRecommendationResponse(**rec)
            for rec in recommendations
        ]
        
    except Exception as e:
        logger.error(f"Failed to get optimization recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    db: AsyncSession = Depends(get_async_session),
    monitor: ContextPerformanceMonitor = Depends(get_context_performance_monitor)
) -> HealthCheckResponse:
    """
    Comprehensive health check for all context engine components.
    
    Checks the health of:
    - Context Performance Monitor
    - Database connectivity
    - Redis connectivity
    - Search Analytics
    - Context Analytics Manager
    """
    try:
        timestamp = datetime.utcnow().isoformat()
        components = {}
        overall_health = "healthy"
        
        # Check Context Performance Monitor
        try:
            # Basic check - if we can get the monitor, it's healthy
            components["context_performance_monitor"] = {
                "status": "healthy",
                "active_alerts": len(monitor.active_alerts),
                "recommendations": len(monitor.recommendations)
            }
        except Exception as e:
            components["context_performance_monitor"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            overall_health = "degraded"
        
        # Check Database
        try:
            await db.execute("SELECT 1")
            components["database"] = {
                "status": "healthy",
                "connection": "active"
            }
        except Exception as e:
            components["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            overall_health = "unhealthy"
        
        # Check Redis
        try:
            await monitor.redis_client.ping()
            components["redis"] = {
                "status": "healthy",
                "connection": "active"
            }
        except Exception as e:
            components["redis"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            overall_health = "degraded"
        
        # Check Search Analytics
        try:
            search_analytics = await get_search_analytics()
            components["search_analytics"] = {
                "status": "healthy",
                "active": search_analytics is not None,
                "active_alerts": len(search_analytics.active_alerts) if search_analytics else 0
            }
        except Exception as e:
            components["search_analytics"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            overall_health = "degraded"
        
        # Check Context Analytics Manager
        try:
            analytics_manager = await get_context_analytics_manager(db, None)
            components["context_analytics"] = {
                "status": "healthy",
                "manager": analytics_manager is not None
            }
        except Exception as e:
            components["context_analytics"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            overall_health = "degraded"
        
        return HealthCheckResponse(
            status="ok",
            timestamp=timestamp,
            components=components,
            overall_health=overall_health
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="error",
            timestamp=datetime.utcnow().isoformat(),
            components={"error": str(e)},
            overall_health="unhealthy"
        )


@router.get("/metrics/realtime")
async def get_realtime_metrics(
    limit: int = Query(100, ge=1, le=1000, description="Number of recent metrics to return"),
    monitor: ContextPerformanceMonitor = Depends(get_context_performance_monitor)
) -> Dict[str, Any]:
    """
    Get real-time metrics from Redis for dashboard updates.
    
    Returns recent performance metrics suitable for real-time
    dashboard updates and monitoring.
    """
    try:
        # Get recent metrics from Redis
        metrics_raw = await monitor.redis_client.lrange(
            "context_monitor:realtime_metrics", 
            0, 
            limit - 1
        )
        
        metrics = []
        for metric_str in metrics_raw:
            try:
                metric = json.loads(metric_str)
                metrics.append(metric)
            except (json.JSONDecodeError, TypeError):
                continue
        
        # Get active alerts
        alert_keys = await monitor.redis_client.keys("context_monitor:alert:*")
        active_alerts = []
        
        for key in alert_keys:
            try:
                alert_data = await monitor.redis_client.get(key)
                if alert_data:
                    alert = json.loads(alert_data)
                    active_alerts.append(alert)
            except (json.JSONDecodeError, TypeError):
                continue
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "active_alerts": active_alerts,
            "metrics_count": len(metrics),
            "alerts_count": len(active_alerts)
        }
        
    except Exception as e:
        logger.error(f"Failed to get real-time metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/active")
async def get_active_alerts(
    monitor: ContextPerformanceMonitor = Depends(get_context_performance_monitor)
) -> Dict[str, Any]:
    """
    Get all currently active performance alerts.
    
    Returns detailed information about all active alerts
    including severity, component, and threshold information.
    """
    try:
        alerts = []
        
        for alert_id, alert in monitor.active_alerts.items():
            alerts.append({
                "alert_id": alert.alert_id,
                "issue_type": alert.issue_type.value,
                "severity": alert.severity,
                "component": alert.component,
                "message": alert.message,
                "threshold_value": alert.threshold_value,
                "current_value": alert.current_value,
                "triggered_at": alert.triggered_at.isoformat(),
                "metadata": alert.metadata
            })
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "active_alerts": alerts,
            "count": len(alerts),
            "by_severity": {
                "critical": len([a for a in alerts if a["severity"] == "critical"]),
                "warning": len([a for a in alerts if a["severity"] == "warning"]),
                "info": len([a for a in alerts if a["severity"] == "info"])
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get active alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/acknowledge/{alert_id}")
async def acknowledge_alert(
    alert_id: str,
    monitor: ContextPerformanceMonitor = Depends(get_context_performance_monitor)
) -> Dict[str, str]:
    """
    Acknowledge a performance alert.
    
    Marks an alert as acknowledged, which can be used to
    suppress notifications or update alert status.
    """
    try:
        if alert_id in monitor.active_alerts:
            alert = monitor.active_alerts[alert_id]
            alert.metadata["acknowledged"] = True
            alert.metadata["acknowledged_at"] = datetime.utcnow().isoformat()
            
            # Update in Redis
            await monitor.redis_client.setex(
                f"context_monitor:alert:{alert_id}",
                3600,  # 1 hour TTL
                json.dumps({
                    "alert_id": alert.alert_id,
                    "issue_type": alert.issue_type.value,
                    "severity": alert.severity,
                    "component": alert.component,
                    "message": alert.message,
                    "threshold_value": alert.threshold_value,
                    "current_value": alert.current_value,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "metadata": alert.metadata
                })
            )
            
            return {
                "status": "acknowledged",
                "alert_id": alert_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Alert not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/context-usage")
async def get_context_usage_analytics(
    context_id: Optional[str] = Query(None, description="Specific context ID to analyze"),
    agent_id: Optional[str] = Query(None, description="Specific agent ID to analyze"),
    days_back: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: AsyncSession = Depends(get_async_session)
) -> Dict[str, Any]:
    """
    Get context usage analytics from the analytics manager.
    
    Provides detailed analytics on context retrieval patterns,
    usage frequency, and performance metrics.
    """
    try:
        analytics_manager = await get_context_analytics_manager(db, None)
        
        # Convert string UUIDs to UUID objects if provided
        context_uuid = None
        agent_uuid = None
        
        if context_id:
            from uuid import UUID
            try:
                context_uuid = UUID(context_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid context_id format")
        
        if agent_id:
            from uuid import UUID
            try:
                agent_uuid = UUID(agent_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid agent_id format")
        
        analytics = await analytics_manager.get_context_usage_analytics(
            context_id=context_uuid,
            agent_id=agent_uuid,
            days_back=days_back
        )
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get context usage analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task for metrics collection
async def collect_background_metrics():
    """Background task to collect system metrics."""
    try:
        await metrics_exporter.collect_all_metrics()
    except Exception as e:
        logger.error(f"Background metrics collection failed: {e}")


@router.post("/metrics/collect")
async def trigger_metrics_collection(
    background_tasks: BackgroundTasks
) -> Dict[str, str]:
    """
    Trigger immediate metrics collection from all sources.
    
    Useful for testing or when immediate metrics updates are needed.
    """
    try:
        background_tasks.add_task(collect_background_metrics)
        
        return {
            "status": "triggered",
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Metrics collection started in background"
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger metrics collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capacity/trends")
async def get_capacity_trends(
    hours_back: int = Query(168, ge=1, le=720, description="Hours of history to analyze"),
    monitor: ContextPerformanceMonitor = Depends(get_context_performance_monitor)
) -> Dict[str, Any]:
    """
    Get capacity trends and growth projections.
    
    Analyzes historical data to provide capacity planning insights
    and growth trend predictions.
    """
    try:
        # Get capacity history from Redis
        history_raw = await monitor.redis_client.lrange(
            "context_monitor:capacity_history", 
            0, 
            -1
        )
        
        history = []
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        for entry_str in history_raw:
            try:
                entry = json.loads(entry_str)
                entry_time = datetime.fromisoformat(entry["timestamp"])
                
                if entry_time >= cutoff_time:
                    history.append(entry)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        
        if len(history) < 2:
            return {
                "error": "Insufficient historical data for trend analysis",
                "data_points": len(history)
            }
        
        # Sort by timestamp
        history.sort(key=lambda x: x["timestamp"])
        
        # Calculate trends
        oldest = history[0]
        newest = history[-1]
        
        time_diff_hours = (
            datetime.fromisoformat(newest["timestamp"]) - 
            datetime.fromisoformat(oldest["timestamp"])
        ).total_seconds() / 3600
        
        if time_diff_hours == 0:
            return {"error": "No time difference in data"}
        
        contexts_growth_per_hour = (
            newest["total_contexts"] - oldest["total_contexts"]
        ) / time_diff_hours
        
        size_growth_per_hour = (
            newest["total_size_bytes"] - oldest["total_size_bytes"]
        ) / time_diff_hours
        
        # Projections
        current_contexts = newest["total_contexts"]
        current_size_bytes = newest["total_size_bytes"]
        
        return {
            "analysis_period_hours": hours_back,
            "data_points": len(history),
            "current_metrics": {
                "total_contexts": current_contexts,
                "total_size_bytes": current_size_bytes,
                "total_size_gb": round(current_size_bytes / (1024**3), 2)
            },
            "growth_rates": {
                "contexts_per_hour": round(contexts_growth_per_hour, 2),
                "contexts_per_day": round(contexts_growth_per_hour * 24, 2),
                "size_bytes_per_hour": round(size_growth_per_hour, 2),
                "size_gb_per_day": round(size_growth_per_hour * 24 / (1024**3), 4)
            },
            "projections": {
                "contexts_in_30_days": int(current_contexts + (contexts_growth_per_hour * 24 * 30)),
                "size_gb_in_30_days": round((current_size_bytes + (size_growth_per_hour * 24 * 30)) / (1024**3), 2),
                "contexts_in_90_days": int(current_contexts + (contexts_growth_per_hour * 24 * 90)),
                "size_gb_in_90_days": round((current_size_bytes + (size_growth_per_hour * 24 * 90)) / (1024**3), 2)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get capacity trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))