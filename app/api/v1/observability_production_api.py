"""
Production Observability API for LeanVibe Agent Hive 2.0

Comprehensive API endpoints that integrate all advanced observability components:
- Agent workflow tracking with real-time state monitoring
- Intelligent alerting with ML-based anomaly detection  
- Dashboard metrics streaming with WebSocket updates
- Performance optimization advisor with automated recommendations

Provides unified access to production-grade observability stack supporting
50+ concurrent agents with <100ms latency and intelligent insights.
"""

import asyncio
import uuid
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query, Path, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.agent_workflow_tracker import (
    get_agent_workflow_tracker, AgentState, TaskProgressState, WorkflowPhase
)
from app.core.intelligent_alerting import (
    get_alert_manager, AlertSeverity, AlertCategory, ImpactLevel
)
from app.core.dashboard_metrics_streaming import (
    get_dashboard_metrics_streaming, DashboardType, MetricStreamType, DashboardFilter
)
from app.core.performance_optimization_advisor import (
    get_performance_optimization_advisor, OptimizationCategory, OptimizationPriority
)
from app.core.performance_monitoring import get_performance_intelligence_engine

logger = structlog.get_logger()

# Create router
router = APIRouter(prefix="/observability/production", tags=["production-observability"])


# API Models
class AgentStateTransitionRequest(BaseModel):
    """Request model for agent state transitions."""
    agent_id: str = Field(..., description="Agent UUID")
    new_state: str = Field(..., description="New agent state")
    transition_reason: str = Field(..., description="Reason for state transition")
    session_id: Optional[str] = Field(None, description="Session UUID")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    resource_allocation: Optional[Dict[str, Any]] = Field(None, description="Resource allocation info")


class TaskProgressRequest(BaseModel):
    """Request model for task progress updates."""
    task_id: str = Field(..., description="Task UUID")
    agent_id: str = Field(..., description="Agent UUID")
    new_state: str = Field(..., description="New task state")
    workflow_id: Optional[str] = Field(None, description="Workflow UUID")
    session_id: Optional[str] = Field(None, description="Session UUID")
    progress_percentage: float = Field(0.0, ge=0.0, le=100.0, description="Progress percentage")
    milestone_reached: Optional[str] = Field(None, description="Milestone reached")
    blocking_dependencies: Optional[List[str]] = Field(default_factory=list, description="Blocking dependencies")
    execution_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Execution context")


class SystemHealthResponse(BaseModel):
    """Response model for system health status."""
    overall_status: str = Field(..., description="Overall system health status")
    health_score: float = Field(..., description="Overall health score (0-1)")
    component_health: Dict[str, Dict[str, Any]] = Field(..., description="Individual component health")
    active_alerts: List[Dict[str, Any]] = Field(..., description="Active alerts")
    performance_summary: Dict[str, Any] = Field(..., description="Performance summary")
    optimization_recommendations: List[Dict[str, Any]] = Field(..., description="Top optimization recommendations")
    last_updated: str = Field(..., description="Last update timestamp")


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status."""
    workflow_id: str = Field(..., description="Workflow UUID")
    current_phase: str = Field(..., description="Current workflow phase")
    progress_percentage: float = Field(..., description="Overall progress percentage")
    active_agents: List[str] = Field(..., description="Active agent IDs")
    completed_tasks: int = Field(..., description="Number of completed tasks")
    total_tasks: int = Field(..., description="Total number of tasks")
    failed_tasks: int = Field(..., description="Number of failed tasks")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")
    performance_metrics: Dict[str, Any] = Field(..., description="Workflow performance metrics")


class MetricRequest(BaseModel):
    """Request model for metric evaluation."""
    metric_name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")
    tags: Optional[Dict[str, str]] = Field(default_factory=dict, description="Metric tags")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")


# WebSocket endpoints
@router.websocket("/stream/comprehensive")
async def comprehensive_observability_stream(
    websocket: WebSocket,
    dashboard_type: str = Query("operational", description="Dashboard type"),
    include_workflow_tracking: bool = Query(True, description="Include workflow tracking data"),
    include_performance_metrics: bool = Query(True, description="Include performance metrics"),
    include_alerts: bool = Query(True, description="Include alerts and anomalies"),
    include_optimizations: bool = Query(True, description="Include optimization insights"),
    update_rate_ms: int = Query(1000, ge=100, le=10000, description="Update rate in milliseconds"),
    agent_filter: Optional[str] = Query(None, description="Comma-separated agent IDs to filter"),
    workflow_filter: Optional[str] = Query(None, description="Comma-separated workflow IDs to filter")
):
    """
    Comprehensive WebSocket stream combining all observability data streams.
    
    Provides unified real-time updates for:
    - Agent state transitions and workflow progress
    - Performance metrics and system health
    - Intelligent alerts and anomalies  
    - Optimization recommendations and insights
    """
    try:
        await websocket.accept()
        
        # Parse filters
        agent_ids = agent_filter.split(",") if agent_filter else []
        workflow_ids = workflow_filter.split(",") if workflow_filter else []
        
        # Get dashboard type enum
        try:
            dashboard_type_enum = DashboardType(dashboard_type)
        except ValueError:
            dashboard_type_enum = DashboardType.OPERATIONAL
        
        # Get observability components
        workflow_tracker = await get_agent_workflow_tracker()
        alert_manager = await get_alert_manager()
        metrics_streaming = await get_dashboard_metrics_streaming()
        optimization_advisor = await get_performance_optimization_advisor()
        performance_engine = await get_performance_intelligence_engine()
        
        # Create dashboard filter
        dashboard_filter = DashboardFilter(
            agent_ids=agent_ids,
            workflow_ids=workflow_ids,
            update_rate_ms=update_rate_ms,
            enable_compression=True
        )
        
        # Connect to metrics streaming
        connection_id = await metrics_streaming.connect_dashboard(
            websocket=websocket,
            dashboard_type=dashboard_type_enum,
            filters=dashboard_filter
        )
        
        logger.info(
            "Comprehensive observability stream connected",
            connection_id=connection_id,
            dashboard_type=dashboard_type,
            agent_filter=agent_ids,
            workflow_filter=workflow_ids
        )
        
        # Stream updates
        while True:
            try:
                # Collect comprehensive data
                update_data = {
                    "type": "comprehensive_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "connection_id": connection_id
                }
                
                # Workflow tracking data
                if include_workflow_tracking:
                    workflow_status = await workflow_tracker.get_real_time_workflow_status(
                        include_agent_details=True,
                        include_communication_flow=True
                    )
                    update_data["workflow_tracking"] = workflow_status
                
                # Performance metrics
                if include_performance_metrics:
                    dashboard_metrics = await performance_engine.get_real_time_performance_dashboard()
                    update_data["performance_metrics"] = dashboard_metrics
                
                # Alerts and anomalies
                if include_alerts:
                    active_alerts = alert_manager.get_active_alerts(
                        severity_filter=[AlertSeverity.HIGH, AlertSeverity.CRITICAL]
                    )
                    update_data["alerts"] = {
                        "active_count": len(active_alerts),
                        "critical_alerts": active_alerts[:5]  # Top 5 most critical
                    }
                
                # Optimization insights
                if include_optimizations:
                    recommendations = await optimization_advisor.get_optimization_recommendations(
                        priority_filter=[OptimizationPriority.IMMEDIATE, OptimizationPriority.HIGH],
                        limit=3
                    )
                    insights = await optimization_advisor.get_performance_insights(hours_back=1)
                    
                    update_data["optimization"] = {
                        "top_recommendations": [rec.to_dict() for rec in recommendations],
                        "recent_insights": [insight.to_dict() for insight in insights[:5]]
                    }
                
                # Send comprehensive update
                await websocket.send_text(json.dumps(update_data))
                
                # Wait for next update cycle
                await asyncio.sleep(update_rate_ms / 1000.0)
                
            except WebSocketDisconnect:
                logger.info(f"Comprehensive stream client disconnected: {connection_id}")
                break
            except Exception as e:
                logger.error(f"Error in comprehensive stream: {e}")
                error_message = {
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket.send_text(json.dumps(error_message))
                await asyncio.sleep(5)  # Back off on error
        
    except Exception as e:
        logger.error(f"Comprehensive observability stream error: {e}")
        if websocket.client_state.name != "DISCONNECTED":
            await websocket.close(code=1011, reason="Internal server error")
    
    finally:
        # Cleanup connection
        if 'connection_id' in locals():
            await metrics_streaming.disconnect_dashboard(connection_id)


# REST API endpoints
@router.get("/health", response_model=SystemHealthResponse)
async def get_comprehensive_system_health():
    """
    Get comprehensive system health status across all observability components.
    
    Provides unified view of:
    - Overall system health score
    - Individual component health status
    - Active critical alerts
    - Performance summary with key metrics
    - Top optimization recommendations
    """
    try:
        # Get all observability components
        workflow_tracker = await get_agent_workflow_tracker()
        alert_manager = await get_alert_manager()
        optimization_advisor = await get_performance_optimization_advisor()
        performance_engine = await get_performance_intelligence_engine()
        
        # Collect component health
        component_health = {}
        
        # Workflow tracker health
        workflow_status = await workflow_tracker.get_real_time_workflow_status()
        component_health["workflow_tracker"] = {
            "status": "healthy" if workflow_status.get("system_overview") else "degraded",
            "active_agents": workflow_status.get("agent_summary", {}).get("total_agents", 0),
            "active_workflows": len(workflow_status.get("active_workflows", [])),
            "last_updated": workflow_status.get("timestamp")
        }
        
        # Alert manager health
        active_alerts = alert_manager.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.get("severity") == "critical"]
        
        component_health["alert_manager"] = {
            "status": "critical" if critical_alerts else ("degraded" if active_alerts else "healthy"),
            "active_alerts": len(active_alerts),
            "critical_alerts": len(critical_alerts),
            "alert_processing_rate": alert_manager.alert_metrics.get("alerts_generated", 0)
        }
        
        # Performance engine health
        performance_dashboard = await performance_engine.get_real_time_performance_dashboard()
        system_health = performance_dashboard.get("system_health", {})
        
        component_health["performance_engine"] = {
            "status": system_health.get("status", "unknown"),
            "overall_score": system_health.get("overall_score", 0.0),
            "component_scores": system_health.get("component_scores", {}),
            "capacity_status": performance_dashboard.get("capacity_status", {})
        }
        
        # Optimization advisor health
        recommendations = await optimization_advisor.get_optimization_recommendations(limit=3)
        insights = await optimization_advisor.get_performance_insights(hours_back=1)
        
        component_health["optimization_advisor"] = {
            "status": "healthy",
            "active_recommendations": len(recommendations),
            "recent_insights": len(insights),
            "high_priority_recommendations": len([r for r in recommendations if r.priority == OptimizationPriority.IMMEDIATE])
        }
        
        # Calculate overall health score
        health_scores = []
        for component, health in component_health.items():
            if health["status"] == "healthy":
                health_scores.append(1.0)
            elif health["status"] == "degraded":
                health_scores.append(0.7)
            elif health["status"] == "critical":
                health_scores.append(0.3)
            else:
                health_scores.append(0.5)
        
        overall_health_score = sum(health_scores) / len(health_scores) if health_scores else 0.0
        
        # Determine overall status
        if overall_health_score >= 0.9:
            overall_status = "healthy"
        elif overall_health_score >= 0.7:
            overall_status = "degraded"
        else:
            overall_status = "critical"
        
        # Get performance summary
        performance_summary = {
            "system_metrics": performance_dashboard.get("real_time_metrics", {}),
            "agent_performance": workflow_status.get("agent_summary", {}),
            "workflow_performance": workflow_status.get("performance_metrics", {}),
            "capacity_utilization": performance_dashboard.get("capacity_status", {})
        }
        
        return SystemHealthResponse(
            overall_status=overall_status,
            health_score=round(overall_health_score, 3),
            component_health=component_health,
            active_alerts=active_alerts[:10],  # Top 10 alerts
            performance_summary=performance_summary,
            optimization_recommendations=[rec.to_dict() for rec in recommendations],
            last_updated=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error("Failed to get comprehensive system health", error=str(e))
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.post("/agent/state-transition")
async def track_agent_state_transition(request: AgentStateTransitionRequest):
    """Track agent state transition with comprehensive monitoring."""
    try:
        workflow_tracker = await get_agent_workflow_tracker()
        
        # Parse agent state
        try:
            new_state = AgentState(request.new_state)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid agent state: {request.new_state}")
        
        # Track state transition
        await workflow_tracker.track_agent_state_transition(
            agent_id=uuid.UUID(request.agent_id),
            new_state=new_state,
            transition_reason=request.transition_reason,
            session_id=uuid.UUID(request.session_id) if request.session_id else None,
            context=request.context,
            resource_allocation=request.resource_allocation
        )
        
        return {
            "status": "success",
            "message": "Agent state transition tracked",
            "agent_id": request.agent_id,
            "new_state": request.new_state,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid UUID: {str(e)}")
    except Exception as e:
        logger.error("Failed to track agent state transition", error=str(e))
        raise HTTPException(status_code=500, detail=f"State transition tracking failed: {str(e)}")


@router.post("/task/progress")
async def track_task_progress(request: TaskProgressRequest):
    """Track task progress with milestone validation."""
    try:
        workflow_tracker = await get_agent_workflow_tracker()
        
        # Parse task state
        try:
            new_state = TaskProgressState(request.new_state)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid task state: {request.new_state}")
        
        # Track task progress
        await workflow_tracker.track_task_progress(
            task_id=uuid.UUID(request.task_id),
            agent_id=uuid.UUID(request.agent_id),
            new_state=new_state,
            workflow_id=uuid.UUID(request.workflow_id) if request.workflow_id else None,
            session_id=uuid.UUID(request.session_id) if request.session_id else None,
            milestone_reached=request.milestone_reached,
            progress_percentage=request.progress_percentage,
            blocking_dependencies=request.blocking_dependencies,
            execution_context=request.execution_context
        )
        
        return {
            "status": "success",
            "message": "Task progress tracked",
            "task_id": request.task_id,
            "new_state": request.new_state,
            "progress_percentage": request.progress_percentage,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid UUID: {str(e)}")
    except Exception as e:
        logger.error("Failed to track task progress", error=str(e))
        raise HTTPException(status_code=500, detail=f"Task progress tracking failed: {str(e)}")


@router.get("/workflow/{workflow_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str = Path(..., description="Workflow UUID")):
    """Get detailed workflow status and progress."""
    try:
        workflow_tracker = await get_agent_workflow_tracker()
        workflow_uuid = uuid.UUID(workflow_id)
        
        # Update workflow progress
        snapshot = await workflow_tracker.update_workflow_progress(
            workflow_id=workflow_uuid,
            current_phase=WorkflowPhase.EXECUTION
        )
        
        # Get real-time status
        status = await workflow_tracker.get_real_time_workflow_status(
            workflow_id=workflow_uuid,
            include_agent_details=True
        )
        
        return WorkflowStatusResponse(
            workflow_id=workflow_id,
            current_phase=snapshot.current_phase.value,
            progress_percentage=(snapshot.completed_tasks / max(snapshot.total_tasks, 1)) * 100,
            active_agents=[str(agent_id) for agent_id in snapshot.active_agents],
            completed_tasks=snapshot.completed_tasks,
            total_tasks=snapshot.total_tasks,
            failed_tasks=snapshot.failed_tasks,
            estimated_completion=snapshot.estimated_completion_time.isoformat() if snapshot.estimated_completion_time else None,
            performance_metrics=status.get("performance_metrics", {})
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid workflow UUID: {str(e)}")
    except Exception as e:
        logger.error("Failed to get workflow status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Workflow status retrieval failed: {str(e)}")


@router.post("/metrics/evaluate")
async def evaluate_metric_for_alerts(request: MetricRequest):
    """Evaluate metric against alert rules and anomaly detection."""
    try:
        alert_manager = await get_alert_manager()
        
        # Evaluate metric
        generated_alerts = await alert_manager.evaluate_metric_for_alerts(
            metric_name=request.metric_name,
            value=request.value,
            tags=request.tags,
            context=request.context
        )
        
        return {
            "status": "success",
            "metric_name": request.metric_name,
            "metric_value": request.value,
            "alerts_generated": len(generated_alerts),
            "alerts": [alert.to_dict() for alert in generated_alerts],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to evaluate metric for alerts", error=str(e))
        raise HTTPException(status_code=500, detail=f"Metric evaluation failed: {str(e)}")


@router.get("/alerts/active")
async def get_active_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity (critical, high, medium, low)"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of alerts to return")
):
    """Get currently active alerts with optional filtering."""
    try:
        alert_manager = await get_alert_manager()
        
        # Parse filters
        severity_filter = None
        if severity:
            try:
                severity_filter = [AlertSeverity(severity)]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
        
        category_filter = None
        if category:
            try:
                category_filter = [AlertCategory(category)]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        
        # Get active alerts
        active_alerts = alert_manager.get_active_alerts(
            severity_filter=severity_filter,
            category_filter=category_filter
        )
        
        return {
            "status": "success",
            "total_alerts": len(active_alerts),
            "alerts": active_alerts[:limit],
            "filters_applied": {
                "severity": severity,
                "category": category,
                "limit": limit
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get active alerts", error=str(e))
        raise HTTPException(status_code=500, detail=f"Alert retrieval failed: {str(e)}")


@router.get("/optimization/recommendations")
async def get_optimization_recommendations(
    category: Optional[str] = Query(None, description="Filter by optimization category"),
    priority: Optional[str] = Query(None, description="Filter by priority (immediate, high, medium, low)"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of recommendations to return")
):
    """Get performance optimization recommendations."""
    try:
        optimization_advisor = await get_performance_optimization_advisor()
        
        # Parse filters
        category_filter = None
        if category:
            try:
                category_filter = [OptimizationCategory(category)]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        
        priority_filter = None
        if priority:
            try:
                priority_filter = [OptimizationPriority(priority)]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid priority: {priority}")
        
        # Get recommendations
        recommendations = await optimization_advisor.get_optimization_recommendations(
            category_filter=category_filter,
            priority_filter=priority_filter,
            limit=limit
        )
        
        return {
            "status": "success",
            "total_recommendations": len(recommendations),
            "recommendations": [rec.to_dict() for rec in recommendations],
            "filters_applied": {
                "category": category,
                "priority": priority,
                "limit": limit
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get optimization recommendations", error=str(e))
        raise HTTPException(status_code=500, detail=f"Optimization recommendations retrieval failed: {str(e)}")


@router.get("/insights/performance")
async def get_performance_insights(
    hours_back: int = Query(24, ge=1, le=168, description="Hours of historical data to analyze"),
    category: Optional[str] = Query(None, description="Filter by insight category"),
    min_confidence: float = Query(0.7, ge=0.0, le=1.0, description="Minimum confidence score")
):
    """Get recent performance insights and analysis."""
    try:
        optimization_advisor = await get_performance_optimization_advisor()
        
        # Parse category filter
        category_filter = None
        if category:
            try:
                category_filter = [OptimizationCategory(category)]
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        
        # Get insights
        insights = await optimization_advisor.get_performance_insights(
            hours_back=hours_back,
            category_filter=category_filter
        )
        
        # Filter by confidence
        filtered_insights = [
            insight for insight in insights
            if insight.confidence_score >= min_confidence
        ]
        
        return {
            "status": "success",
            "total_insights": len(filtered_insights),
            "insights": [insight.to_dict() for insight in filtered_insights],
            "analysis_period": {
                "hours_back": hours_back,
                "start_time": (datetime.utcnow() - timedelta(hours=hours_back)).isoformat(),
                "end_time": datetime.utcnow().isoformat()
            },
            "filters_applied": {
                "category": category,
                "min_confidence": min_confidence
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get performance insights", error=str(e))
        raise HTTPException(status_code=500, detail=f"Performance insights retrieval failed: {str(e)}")


@router.get("/dashboard/metrics")
async def get_dashboard_performance_metrics():
    """Get dashboard and streaming service performance metrics."""
    try:
        metrics_streaming = await get_dashboard_metrics_streaming()
        
        # Get comprehensive metrics
        dashboard_metrics = await metrics_streaming.get_dashboard_metrics()
        
        return {
            "status": "success",
            "dashboard_metrics": dashboard_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get dashboard metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Dashboard metrics retrieval failed: {str(e)}")


@router.post("/system/trigger-analysis")
async def trigger_comprehensive_analysis(background_tasks: BackgroundTasks):
    """Trigger comprehensive system analysis and optimization recommendations."""
    try:
        async def run_analysis():
            """Background task for comprehensive analysis."""
            try:
                # Get all components
                workflow_tracker = await get_agent_workflow_tracker()
                alert_manager = await get_alert_manager() 
                optimization_advisor = await get_performance_optimization_advisor()
                performance_engine = await get_performance_intelligence_engine()
                
                # Trigger analysis across all components
                logger.info("Starting comprehensive system analysis")
                
                # Get current system state
                workflow_status = await workflow_tracker.get_real_time_workflow_status()
                performance_dashboard = await performance_engine.get_real_time_performance_dashboard()
                
                # Generate system-wide alerts for any issues
                system_metrics = performance_dashboard.get("real_time_metrics", {})
                for metric_name, value in system_metrics.items():
                    if isinstance(value, (int, float)):
                        await alert_manager.evaluate_metric_for_alerts(
                            metric_name=metric_name,
                            value=value,
                            context={"analysis_triggered": True}
                        )
                
                logger.info("Comprehensive system analysis completed")
                
            except Exception as e:
                logger.error("Comprehensive analysis failed", error=str(e))
        
        # Schedule analysis in background
        background_tasks.add_task(run_analysis)
        
        return {
            "status": "success",
            "message": "Comprehensive system analysis triggered",
            "analysis_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to trigger comprehensive analysis", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis trigger failed: {str(e)}")


@router.get("/status/summary")
async def get_observability_status_summary():
    """Get high-level status summary of all observability components."""
    try:
        # Get basic health checks from all components
        summary = {
            "observability_stack_status": "operational",
            "components": {},
            "key_metrics": {},
            "alerts_summary": {},
            "last_updated": datetime.utcnow().isoformat()
        }
        
        try:
            workflow_tracker = await get_agent_workflow_tracker()
            status = await workflow_tracker.get_real_time_workflow_status()
            summary["components"]["workflow_tracker"] = {
                "status": "healthy",
                "active_agents": status.get("agent_summary", {}).get("total_agents", 0),
                "active_workflows": len(status.get("active_workflows", []))
            }
        except Exception as e:
            summary["components"]["workflow_tracker"] = {"status": "error", "error": str(e)}
        
        try:
            alert_manager = await get_alert_manager()
            active_alerts = alert_manager.get_active_alerts()
            critical_alerts = [a for a in active_alerts if a.get("severity") == "critical"]
            summary["components"]["alert_manager"] = {
                "status": "critical" if critical_alerts else "healthy",
                "active_alerts": len(active_alerts),
                "critical_alerts": len(critical_alerts)
            }
            summary["alerts_summary"] = {
                "total_active": len(active_alerts),
                "critical": len(critical_alerts),
                "high": len([a for a in active_alerts if a.get("severity") == "high"])
            }
        except Exception as e:
            summary["components"]["alert_manager"] = {"status": "error", "error": str(e)}
        
        try:
            metrics_streaming = await get_dashboard_metrics_streaming()
            dashboard_metrics = await metrics_streaming.get_dashboard_metrics()
            summary["components"]["metrics_streaming"] = {
                "status": dashboard_metrics.get("service_status", "unknown"),
                "active_connections": dashboard_metrics.get("connections", {}).get("active", 0),
                "messages_per_second": dashboard_metrics.get("performance", {}).get("messages_per_second", 0)
            }
        except Exception as e:
            summary["components"]["metrics_streaming"] = {"status": "error", "error": str(e)}
        
        try:
            optimization_advisor = await get_performance_optimization_advisor()
            recommendations = await optimization_advisor.get_optimization_recommendations(limit=5)
            summary["components"]["optimization_advisor"] = {
                "status": "healthy",
                "active_recommendations": len(recommendations),
                "high_priority": len([r for r in recommendations if r.priority == OptimizationPriority.IMMEDIATE])
            }
        except Exception as e:
            summary["components"]["optimization_advisor"] = {"status": "error", "error": str(e)}
        
        # Determine overall status
        component_statuses = [comp.get("status", "unknown") for comp in summary["components"].values()]
        if "critical" in component_statuses:
            summary["observability_stack_status"] = "critical"
        elif "error" in component_statuses:
            summary["observability_stack_status"] = "degraded"
        elif all(status == "healthy" for status in component_statuses):
            summary["observability_stack_status"] = "healthy"
        else:
            summary["observability_stack_status"] = "degraded"
        
        return summary
        
    except Exception as e:
        logger.error("Failed to get observability status summary", error=str(e))
        raise HTTPException(status_code=500, detail=f"Status summary failed: {str(e)}")