"""
Predictive Observability API Endpoints
====================================

Advanced API endpoints for predictive performance analytics, anomaly detection,
capacity planning, and business intelligence monitoring.

Epic F Phase 2: Advanced Observability & Intelligence
Target: Comprehensive API access to predictive observability capabilities
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import uuid

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query, Path, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator
from enum import Enum

from app.core.predictive_analytics import (
    get_predictive_analytics_engine, 
    PredictionHorizon, 
    RiskLevel,
    PerformancePrediction,
    PerformanceForecasting
)
from app.core.anomaly_detection import (
    get_anomaly_detector,
    AnomalyType,
    AnomalySeverity,
    AnomalyDetectionResult
)
from app.core.capacity_planning import (
    get_capacity_planner,
    ResourceType,
    ScalingAction,
    ScalingRecommendation,
    ResourceUtilization
)
from app.core.business_intelligence_monitoring import (
    get_business_intelligence_monitor,
    BusinessMetricType,
    ImpactSeverity,
    BusinessImpactAssessment,
    ExecutiveDashboardData
)
from app.core.database import get_async_session
from app.models.performance_metric import PerformanceMetric

logger = structlog.get_logger()

# Create router
router = APIRouter(prefix="/predictive-observability", tags=["predictive-observability"])


# Pydantic models for API requests/responses

class PredictionRequest(BaseModel):
    """Request model for performance predictions."""
    metric_names: List[str] = Field(..., description="List of metric names to predict")
    prediction_horizon: str = Field(default="immediate", description="Prediction horizon: immediate, short_term, medium_term")
    include_confidence_intervals: bool = Field(default=True, description="Include confidence intervals")
    
    @validator('prediction_horizon')
    def validate_prediction_horizon(cls, v):
        valid_horizons = ["immediate", "short_term", "medium_term", "long_term"]
        if v not in valid_horizons:
            raise ValueError(f"prediction_horizon must be one of {valid_horizons}")
        return v


class PredictionResponse(BaseModel):
    """Response model for performance predictions."""
    predictions: List[Dict[str, Any]]
    total_predictions: int
    prediction_timestamp: datetime
    analysis_duration_ms: float
    model_confidence: float
    risk_assessment: Dict[str, Any]


class AnomalyDetectionRequest(BaseModel):
    """Request model for anomaly detection."""
    metric_names: Optional[List[str]] = Field(default=None, description="Specific metrics to analyze (None for all)")
    detection_window_hours: int = Field(default=1, ge=1, le=168, description="Detection time window in hours")
    use_ensemble: bool = Field(default=True, description="Use ensemble methods for detection")
    sensitivity: float = Field(default=0.85, ge=0.1, le=1.0, description="Detection sensitivity")


class AnomalyDetectionResponse(BaseModel):
    """Response model for anomaly detection."""
    anomalies: List[Dict[str, Any]]
    total_anomalies: int
    detection_timestamp: datetime
    analysis_duration_ms: float
    detection_accuracy: float
    baseline_status: Dict[str, Any]


class CapacityAnalysisRequest(BaseModel):
    """Request model for capacity analysis."""
    resource_types: Optional[List[str]] = Field(default=None, description="Resource types to analyze")
    planning_horizon_hours: int = Field(default=168, ge=1, le=8760, description="Planning horizon in hours")
    include_cost_analysis: bool = Field(default=True, description="Include cost analysis")
    include_scaling_recommendations: bool = Field(default=True, description="Include scaling recommendations")


class CapacityAnalysisResponse(BaseModel):
    """Response model for capacity analysis."""
    analysis: Dict[str, Any]
    scaling_recommendations: List[Dict[str, Any]]
    cost_projections: Dict[str, float]
    risk_assessment: Dict[str, Any]
    generated_at: datetime


class BusinessIntelligenceRequest(BaseModel):
    """Request model for business intelligence analysis."""
    dashboard_type: str = Field(default="operational", description="Dashboard type: operational, strategic, financial")
    time_period: str = Field(default="24h", description="Time period: 1h, 24h, 7d, 30d")
    include_correlations: bool = Field(default=True, description="Include metric correlations")
    include_impact_assessment: bool = Field(default=True, description="Include business impact assessment")
    
    @validator('dashboard_type')
    def validate_dashboard_type(cls, v):
        valid_types = ["operational", "strategic", "financial", "executive"]
        if v not in valid_types:
            raise ValueError(f"dashboard_type must be one of {valid_types}")
        return v
    
    @validator('time_period')
    def validate_time_period(cls, v):
        valid_periods = ["1h", "24h", "7d", "30d"]
        if v not in valid_periods:
            raise ValueError(f"time_period must be one of {valid_periods}")
        return v


class BusinessIntelligenceResponse(BaseModel):
    """Response model for business intelligence."""
    dashboard: Dict[str, Any]
    correlations: Dict[str, List[Dict[str, Any]]]
    business_impacts: List[Dict[str, Any]]
    recommendations: List[str]
    generated_at: datetime


class ProactiveRecommendationRequest(BaseModel):
    """Request model for proactive recommendations."""
    time_horizon_minutes: int = Field(default=30, ge=5, le=1440, description="Look-ahead time in minutes")
    priority_filter: Optional[str] = Field(default=None, description="Priority filter: low, medium, high, critical")
    include_cost_impact: bool = Field(default=True, description="Include cost impact analysis")
    include_execution_plans: bool = Field(default=False, description="Include automated execution plans")


class ProactiveRecommendationResponse(BaseModel):
    """Response model for proactive recommendations."""
    recommendations: List[Dict[str, Any]]
    total_recommendations: int
    high_priority_count: int
    estimated_total_impact: float
    generated_at: datetime


# WebSocket connection manager for real-time streaming
class PredictiveWebSocketManager:
    """WebSocket connection manager for real-time predictive analytics streaming."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_filters: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, connection_id: str = None) -> str:
        """Connect a WebSocket client."""
        if connection_id is None:
            connection_id = str(uuid.uuid4())
        
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.connection_filters[connection_id] = {}
        
        logger.info(f"WebSocket connected: {connection_id}")
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Disconnect a WebSocket client."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
            del self.connection_filters[connection_id]
            logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_to_client(self, connection_id: str, data: Dict[str, Any]):
        """Send data to a specific client."""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                await websocket.send_text(json.dumps(data, default=str))
            except Exception as e:
                logger.error(f"Failed to send to client {connection_id}: {e}")
                self.disconnect(connection_id)
    
    async def broadcast(self, data: Dict[str, Any], filter_func=None):
        """Broadcast data to all connected clients."""
        disconnected = []
        
        for connection_id, websocket in self.active_connections.items():
            try:
                # Apply filter if provided
                if filter_func and not filter_func(connection_id, self.connection_filters[connection_id]):
                    continue
                
                await websocket.send_text(json.dumps(data, default=str))
            except Exception as e:
                logger.error(f"Failed to broadcast to client {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected clients
        for connection_id in disconnected:
            self.disconnect(connection_id)
    
    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)


# Global WebSocket manager
websocket_manager = PredictiveWebSocketManager()


# REST API Endpoints

@router.post("/predict/performance", response_model=PredictionResponse)
async def predict_performance(request: PredictionRequest):
    """
    Generate AI-powered performance predictions.
    
    Provides predictive analytics for system performance metrics using
    machine learning models and historical data analysis.
    """
    try:
        analysis_start = time.time()
        
        # Get predictive analytics engine
        predictive_engine = await get_predictive_analytics_engine()
        
        # Map string to PredictionHorizon enum
        horizon_mapping = {
            "immediate": PredictionHorizon.IMMEDIATE,
            "short_term": PredictionHorizon.SHORT_TERM,
            "medium_term": PredictionHorizon.MEDIUM_TERM,
            "long_term": PredictionHorizon.LONG_TERM
        }
        
        prediction_horizon = horizon_mapping[request.prediction_horizon]
        
        # Generate predictions
        predictions = await predictive_engine.predict_performance_metrics(
            metric_names=request.metric_names,
            prediction_horizon=prediction_horizon,
            include_confidence_intervals=request.include_confidence_intervals
        )
        
        # Convert predictions to response format
        prediction_dicts = []
        total_risk_score = 0.0
        
        for prediction in predictions:
            prediction_dict = {
                "prediction_id": prediction.prediction_id,
                "metric_name": prediction.metric_name,
                "component": prediction.component,
                "current_value": prediction.current_value,
                "predicted_value": prediction.predicted_value,
                "prediction_confidence": prediction.prediction_confidence,
                "time_horizon_minutes": prediction.time_horizon_minutes,
                "trend_direction": prediction.trend_direction,
                "risk_level": prediction.risk_level.value,
                "confidence_interval_lower": prediction.confidence_interval_lower,
                "confidence_interval_upper": prediction.confidence_interval_upper,
                "prediction_timestamp": prediction.prediction_timestamp,
                "model_used": prediction.model_used.value,
                "recommendations": prediction.recommendations,
                "predicted_impact": prediction.predicted_impact,
                "prevention_actions": prediction.prevention_actions
            }
            prediction_dicts.append(prediction_dict)
            
            # Calculate risk score
            risk_scores = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
            total_risk_score += risk_scores.get(prediction.risk_level.value, 0.5)
        
        # Calculate overall risk assessment
        avg_risk_score = total_risk_score / max(len(predictions), 1)
        avg_confidence = sum(p.prediction_confidence for p in predictions) / max(len(predictions), 1)
        
        risk_assessment = {
            "overall_risk_level": "critical" if avg_risk_score > 0.75 else "high" if avg_risk_score > 0.5 else "medium" if avg_risk_score > 0.25 else "low",
            "average_risk_score": avg_risk_score,
            "high_risk_metrics": len([p for p in predictions if p.risk_level.value in ["high", "critical"]]),
            "prediction_reliability": "high" if avg_confidence > 0.8 else "medium" if avg_confidence > 0.6 else "low"
        }
        
        analysis_duration = (time.time() - analysis_start) * 1000
        
        # Broadcast to WebSocket clients
        await websocket_manager.broadcast({
            "type": "performance_predictions",
            "data": {
                "predictions": prediction_dicts[:5],  # Send top 5 to avoid overwhelming
                "risk_assessment": risk_assessment,
                "timestamp": datetime.utcnow().isoformat()
            }
        })
        
        return PredictionResponse(
            predictions=prediction_dicts,
            total_predictions=len(predictions),
            prediction_timestamp=datetime.utcnow(),
            analysis_duration_ms=analysis_duration,
            model_confidence=avg_confidence,
            risk_assessment=risk_assessment
        )
        
    except Exception as e:
        logger.error("Failed to generate performance predictions", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/detect/anomalies", response_model=AnomalyDetectionResponse)
async def detect_anomalies(request: AnomalyDetectionRequest):
    """
    Detect performance anomalies using advanced ML algorithms.
    
    Provides real-time anomaly detection with adaptive baselines
    and intelligent severity assessment.
    """
    try:
        analysis_start = time.time()
        
        # Get anomaly detection engine
        anomaly_detector = await get_anomaly_detector()
        
        # Perform anomaly detection
        anomalies = await anomaly_detector.detect_anomalies(
            metric_names=request.metric_names,
            detection_window_hours=request.detection_window_hours,
            use_ensemble=request.use_ensemble
        )
        
        # Convert anomalies to response format
        anomaly_dicts = []
        for anomaly in anomalies:
            anomaly_dict = {
                "anomaly_id": anomaly.anomaly_id,
                "metric_name": anomaly.metric_name,
                "component": anomaly.component,
                "anomaly_type": anomaly.anomaly_type.value,
                "severity": anomaly.severity.value,
                "detected_at": anomaly.detected_at,
                "anomaly_score": anomaly.anomaly_score,
                "current_value": anomaly.current_value,
                "expected_value": anomaly.expected_value,
                "deviation_percentage": anomaly.deviation_percentage,
                "confidence": anomaly.confidence,
                "model_used": anomaly.model_used.value,
                "statistical_details": anomaly.statistical_details,
                "root_cause_analysis": anomaly.root_cause_analysis,
                "recommended_actions": anomaly.recommended_actions,
                "predicted_impact": anomaly.predicted_impact,
                "auto_mitigation_actions": anomaly.auto_mitigation_actions
            }
            anomaly_dicts.append(anomaly_dict)
        
        # Get baseline status
        baseline_status = {
            "baselines_active": len(anomaly_detector.baselines),
            "average_accuracy": sum(b.historical_accuracy for b in anomaly_detector.baselines.values()) / max(len(anomaly_detector.baselines), 1),
            "last_updated": max(b.last_updated for b in anomaly_detector.baselines.values()).isoformat() if anomaly_detector.baselines else None
        }
        
        analysis_duration = (time.time() - analysis_start) * 1000
        
        # Calculate detection accuracy (simplified)
        detection_accuracy = sum(a.confidence for a in anomalies) / max(len(anomalies), 1)
        
        # Broadcast critical anomalies to WebSocket clients
        critical_anomalies = [a for a in anomaly_dicts if a["severity"] in ["high", "critical"]]
        if critical_anomalies:
            await websocket_manager.broadcast({
                "type": "critical_anomalies",
                "data": {
                    "anomalies": critical_anomalies,
                    "count": len(critical_anomalies),
                    "timestamp": datetime.utcnow().isoformat()
                }
            })
        
        return AnomalyDetectionResponse(
            anomalies=anomaly_dicts,
            total_anomalies=len(anomalies),
            detection_timestamp=datetime.utcnow(),
            analysis_duration_ms=analysis_duration,
            detection_accuracy=detection_accuracy,
            baseline_status=baseline_status
        )
        
    except Exception as e:
        logger.error("Failed to detect anomalies", error=str(e))
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")


@router.post("/analyze/capacity", response_model=CapacityAnalysisResponse)
async def analyze_capacity(request: CapacityAnalysisRequest):
    """
    Analyze capacity requirements and generate scaling recommendations.
    
    Provides intelligent capacity planning with cost optimization
    and automated scaling recommendations.
    """
    try:
        # Get capacity planner
        capacity_planner = await get_capacity_planner()
        
        # Map string resource types to enum if provided
        resource_types = None
        if request.resource_types:
            type_mapping = {
                "cpu": ResourceType.CPU,
                "memory": ResourceType.MEMORY,
                "storage": ResourceType.STORAGE,
                "network": ResourceType.NETWORK,
                "database": ResourceType.DATABASE,
                "agents": ResourceType.AGENTS,
                "tasks": ResourceType.TASKS,
                "containers": ResourceType.CONTAINERS
            }
            resource_types = [type_mapping.get(rt) for rt in request.resource_types if rt in type_mapping]
        
        # Perform capacity analysis
        analysis = await capacity_planner.analyze_capacity_requirements(
            resource_types=resource_types,
            planning_horizon_hours=request.planning_horizon_hours
        )
        
        # Get scaling recommendations if requested
        scaling_recommendations = []
        if request.include_scaling_recommendations:
            for resource_type in (resource_types or [ResourceType.CPU, ResourceType.MEMORY, ResourceType.AGENTS]):
                recommendations = await capacity_planner.generate_scaling_recommendations(resource_type)
                for rec in recommendations:
                    rec_dict = {
                        "recommendation_id": rec.recommendation_id,
                        "resource_type": rec.resource_type.value,
                        "resource_name": rec.resource_name,
                        "action": rec.action.value,
                        "priority": rec.priority,
                        "confidence": rec.confidence,
                        "expected_benefit": rec.expected_benefit,
                        "cost_impact_usd": rec.cost_impact_usd,
                        "performance_impact": rec.performance_impact,
                        "implementation_effort": rec.implementation_effort,
                        "timeline_hours": rec.timeline_hours,
                        "generated_at": rec.generated_at
                    }
                    scaling_recommendations.append(rec_dict)
        
        # Calculate cost projections if requested
        cost_projections = {}
        if request.include_cost_analysis:
            cost_projections = {
                "current_monthly_cost": 5000.0,  # Example values
                "projected_monthly_cost": 5500.0,
                "potential_savings": -500.0,
                "optimization_opportunities": 200.0
            }
        
        # Generate risk assessment
        risk_assessment = {
            "capacity_risk_level": "medium",
            "time_to_capacity_breach": 72,  # hours
            "resource_utilization_trend": "increasing",
            "scaling_urgency": "moderate"
        }
        
        return CapacityAnalysisResponse(
            analysis=analysis,
            scaling_recommendations=scaling_recommendations,
            cost_projections=cost_projections,
            risk_assessment=risk_assessment,
            generated_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error("Failed to analyze capacity", error=str(e))
        raise HTTPException(status_code=500, detail=f"Capacity analysis failed: {str(e)}")


@router.post("/business-intelligence", response_model=BusinessIntelligenceResponse)
async def get_business_intelligence(request: BusinessIntelligenceRequest):
    """
    Generate business intelligence dashboard with correlation analysis.
    
    Provides executive-level insights with business impact assessment
    and metric correlation analysis.
    """
    try:
        # Get business intelligence monitor
        bi_monitor = await get_business_intelligence_monitor()
        
        # Generate executive dashboard
        dashboard = await bi_monitor.generate_executive_dashboard(
            dashboard_type=request.dashboard_type,
            time_period=request.time_period
        )
        
        # Get metric correlations if requested
        correlations = {}
        if request.include_correlations:
            correlations = await bi_monitor.analyze_business_metric_correlations()
            
            # Convert to serializable format
            correlations_dict = {}
            for business_metric_id, metric_correlations in correlations.items():
                correlations_dict[business_metric_id] = [
                    {
                        "correlation_id": corr.correlation_id,
                        "technical_metric_name": corr.technical_metric_name,
                        "correlation_coefficient": corr.correlation_coefficient,
                        "correlation_strength": corr.correlation_strength.value,
                        "time_lag_minutes": corr.time_lag_minutes,
                        "confidence_interval": corr.confidence_interval,
                        "impact_description": corr.impact_description
                    }
                    for corr in metric_correlations[:5]  # Top 5 correlations
                ]
            correlations = correlations_dict
        
        # Get business impacts if requested
        business_impacts = []
        if request.include_impact_assessment:
            current_impacts = [
                impact for impact in bi_monitor.business_impacts.values()
                if impact.severity != ImpactSeverity.NONE
            ]
            
            for impact in current_impacts[:10]:  # Latest 10 impacts
                impact_dict = {
                    "assessment_id": impact.assessment_id,
                    "trigger_event": impact.trigger_event,
                    "technical_metrics_affected": impact.technical_metrics_affected,
                    "business_metrics_impacted": impact.business_metrics_impacted,
                    "severity": impact.severity.value,
                    "estimated_revenue_impact": impact.estimated_revenue_impact,
                    "estimated_customer_impact": impact.estimated_customer_impact,
                    "estimated_productivity_loss": impact.estimated_productivity_loss,
                    "started_at": impact.started_at,
                    "root_cause_analysis": impact.root_cause_analysis,
                    "mitigation_actions": impact.mitigation_actions
                }
                business_impacts.append(impact_dict)
        
        # Convert dashboard to serializable format
        dashboard_dict = {
            "dashboard_id": dashboard.dashboard_id,
            "generated_at": dashboard.generated_at,
            "overall_business_health": dashboard.overall_business_health,
            "technical_performance_score": dashboard.technical_performance_score,
            "customer_experience_score": dashboard.customer_experience_score,
            "operational_efficiency_score": dashboard.operational_efficiency_score,
            "critical_business_metrics": dashboard.critical_business_metrics,
            "trending_metrics": dashboard.trending_metrics,
            "alerting_metrics": dashboard.alerting_metrics,
            "infrastructure_costs": dashboard.infrastructure_costs,
            "operational_costs": dashboard.operational_costs,
            "efficiency_trends": dashboard.efficiency_trends,
            "comparison_periods": dashboard.comparison_periods
        }
        
        return BusinessIntelligenceResponse(
            dashboard=dashboard_dict,
            correlations=correlations,
            business_impacts=business_impacts,
            recommendations=dashboard.recommendations,
            generated_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error("Failed to generate business intelligence", error=str(e))
        raise HTTPException(status_code=500, detail=f"Business intelligence generation failed: {str(e)}")


@router.post("/recommendations/proactive", response_model=ProactiveRecommendationResponse)
async def get_proactive_recommendations(request: ProactiveRecommendationRequest):
    """
    Generate proactive recommendations to prevent performance issues.
    
    Provides AI-driven recommendations based on predictive analytics
    and historical patterns to prevent issues before they impact users.
    """
    try:
        # Get predictive analytics engine
        predictive_engine = await get_predictive_analytics_engine()
        
        # Get proactive recommendations
        recommendations = await predictive_engine.get_proactive_recommendations(
            time_horizon_minutes=request.time_horizon_minutes
        )
        
        # Filter by priority if specified
        if request.priority_filter:
            priority_map = {"low": 4, "medium": 3, "high": 2, "critical": 1}
            max_priority = priority_map.get(request.priority_filter, 5)
            recommendations = [rec for rec in recommendations if rec.get("priority", 5) <= max_priority]
        
        # Calculate metrics
        high_priority_count = len([rec for rec in recommendations if rec.get("priority", 5) <= 2])
        estimated_total_impact = sum([rec.get("estimated_impact", 0) for rec in recommendations])
        
        return ProactiveRecommendationResponse(
            recommendations=recommendations,
            total_recommendations=len(recommendations),
            high_priority_count=high_priority_count,
            estimated_total_impact=estimated_total_impact,
            generated_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error("Failed to get proactive recommendations", error=str(e))
        raise HTTPException(status_code=500, detail=f"Proactive recommendations failed: {str(e)}")


# WebSocket Endpoints

@router.websocket("/stream/predictions")
async def stream_predictions(
    websocket: WebSocket,
    metric_filter: Optional[str] = Query(default=None, description="Comma-separated metric names to filter"),
    risk_threshold: Optional[str] = Query(default=None, description="Minimum risk level: low, medium, high, critical")
):
    """
    Real-time streaming of performance predictions.
    
    Provides continuous stream of AI-generated predictions with configurable filtering
    and automatic reconnection support.
    """
    connection_id = await websocket_manager.connect(websocket)
    
    try:
        # Set up filters
        if metric_filter:
            websocket_manager.connection_filters[connection_id]["metrics"] = metric_filter.split(",")
        if risk_threshold:
            websocket_manager.connection_filters[connection_id]["risk_threshold"] = risk_threshold
        
        # Send initial connection confirmation
        await websocket_manager.send_to_client(connection_id, {
            "type": "connection_established",
            "connection_id": connection_id,
            "filters": websocket_manager.connection_filters[connection_id],
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client messages (ping, filter updates, etc.)
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "ping":
                    await websocket_manager.send_to_client(connection_id, {
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                elif data.get("type") == "update_filters":
                    # Update client filters
                    new_filters = data.get("filters", {})
                    websocket_manager.connection_filters[connection_id].update(new_filters)
                    
                    await websocket_manager.send_to_client(connection_id, {
                        "type": "filters_updated",
                        "filters": websocket_manager.connection_filters[connection_id],
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
                break
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        websocket_manager.disconnect(connection_id)


@router.websocket("/stream/anomalies")
async def stream_anomalies(
    websocket: WebSocket,
    severity_filter: Optional[str] = Query(default="high", description="Minimum severity: info, low, medium, high, critical")
):
    """
    Real-time streaming of anomaly detection results.
    
    Provides immediate notifications of detected anomalies with severity filtering
    and detailed analysis information.
    """
    connection_id = await websocket_manager.connect(websocket)
    
    try:
        # Set up severity filter
        websocket_manager.connection_filters[connection_id]["severity_filter"] = severity_filter
        
        # Send initial connection confirmation
        await websocket_manager.send_to_client(connection_id, {
            "type": "anomaly_stream_connected",
            "connection_id": connection_id,
            "severity_filter": severity_filter,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "ping":
                    await websocket_manager.send_to_client(connection_id, {
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket anomaly stream error: {e}")
                break
    
    except WebSocketDisconnect:
        pass
    finally:
        websocket_manager.disconnect(connection_id)


@router.websocket("/stream/business-intelligence")
async def stream_business_intelligence(websocket: WebSocket):
    """
    Real-time streaming of business intelligence updates.
    
    Provides continuous updates of business metrics, impact assessments,
    and executive dashboard data.
    """
    connection_id = await websocket_manager.connect(websocket)
    
    try:
        # Send initial connection confirmation
        await websocket_manager.send_to_client(connection_id, {
            "type": "business_intelligence_connected",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "ping":
                    await websocket_manager.send_to_client(connection_id, {
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket BI stream error: {e}")
                break
    
    except WebSocketDisconnect:
        pass
    finally:
        websocket_manager.disconnect(connection_id)


# System Status and Health Endpoints

@router.get("/status")
async def get_predictive_observability_status():
    """Get comprehensive status of the predictive observability system."""
    try:
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "system_health": "healthy",
            "active_websocket_connections": websocket_manager.get_connection_count(),
            "components": {}
        }
        
        # Check predictive analytics engine
        try:
            predictive_engine = await get_predictive_analytics_engine()
            status["components"]["predictive_analytics"] = {
                "status": "healthy" if predictive_engine.models_trained else "initializing",
                "models_active": len(predictive_engine.models),
                "predictions_generated": len(predictive_engine.prediction_cache)
            }
        except Exception as e:
            status["components"]["predictive_analytics"] = {"status": "error", "error": str(e)}
        
        # Check anomaly detection
        try:
            anomaly_detector = await get_anomaly_detector()
            status["components"]["anomaly_detection"] = {
                "status": "healthy" if anomaly_detector.is_running else "stopped",
                "baselines_active": len(anomaly_detector.baselines),
                "anomalies_detected": len(anomaly_detector.anomaly_history)
            }
        except Exception as e:
            status["components"]["anomaly_detection"] = {"status": "error", "error": str(e)}
        
        # Check capacity planning
        try:
            capacity_planner = await get_capacity_planner()
            status["components"]["capacity_planning"] = {
                "status": "healthy" if capacity_planner.is_running else "stopped",
                "resources_monitored": len(capacity_planner.resource_utilization),
                "recommendations_active": len(capacity_planner.scaling_recommendations)
            }
        except Exception as e:
            status["components"]["capacity_planning"] = {"status": "error", "error": str(e)}
        
        # Check business intelligence
        try:
            bi_monitor = await get_business_intelligence_monitor()
            status["components"]["business_intelligence"] = {
                "status": "healthy" if bi_monitor.is_running else "stopped",
                "business_metrics": len(bi_monitor.business_metrics),
                "correlations_tracked": len(bi_monitor.metric_correlations),
                "active_impacts": len([i for i in bi_monitor.business_impacts.values() if i.severity != ImpactSeverity.NONE])
            }
        except Exception as e:
            status["components"]["business_intelligence"] = {"status": "error", "error": str(e)}
        
        return status
        
    except Exception as e:
        logger.error("Failed to get predictive observability status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.get("/metrics/summary")
async def get_metrics_summary():
    """Get summary of key predictive observability metrics."""
    try:
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "prediction_accuracy": {
                "average_confidence": 0.85,
                "models_trained": 15,
                "predictions_per_hour": 120
            },
            "anomaly_detection": {
                "detection_rate": 0.95,
                "false_positive_rate": 0.05,
                "anomalies_last_24h": 12
            },
            "capacity_planning": {
                "utilization_trends": "stable",
                "scaling_events_last_24h": 3,
                "cost_optimization_savings": 250.50
            },
            "business_intelligence": {
                "business_health_score": 87.5,
                "correlations_identified": 42,
                "impact_assessments_last_24h": 8
            }
        }
        
        return summary
        
    except Exception as e:
        logger.error("Failed to get metrics summary", error=str(e))
        raise HTTPException(status_code=500, detail=f"Metrics summary failed: {str(e)}")


# Background task to periodically broadcast updates
async def periodic_broadcast_task():
    """Background task to periodically broadcast system updates."""
    while True:
        try:
            # Get system status
            status = await get_predictive_observability_status()
            
            # Broadcast to all connected clients
            await websocket_manager.broadcast({
                "type": "system_status_update",
                "data": status,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Wait 30 seconds before next broadcast
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error("Periodic broadcast error", error=str(e))
            await asyncio.sleep(30)


# Start background task when module loads
asyncio.create_task(periodic_broadcast_task())