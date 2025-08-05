"""
Intelligence API - Phase 3 Intelligence Layer Integration

This module provides API endpoints for the intelligence layer capabilities,
integrating alert analysis, user preferences, and ML framework.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.core.alert_analysis_engine import AlertAnalysisEngine, create_alert_analysis_engine
from app.core.user_preference_system import (
    UserPreferenceSystem, 
    create_user_preference_system,
    NotificationPreferences,
    DashboardPreferences,
    AlertPriority,
    NotificationChannel,
    DashboardLayout,
    ColorTheme
)
from app.core.intelligence_framework import (
    IntelligenceFramework,
    create_intelligence_framework,
    DataType,
    IntelligenceType
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/intelligence", tags=["intelligence"])


# Pydantic models for API requests/responses

class AlertAnalysisRequest(BaseModel):
    """Request model for alert analysis"""
    alert_type: str
    severity: str
    message: str
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AlertAnalysisResponse(BaseModel):
    """Response model for alert analysis"""
    alert_id: str
    analysis: Dict[str, Any]
    patterns: List[Dict[str, Any]]
    recommendations: List[str]
    priority_score: float


class NotificationPreferencesRequest(BaseModel):
    """Request model for notification preferences"""
    enabled_channels: List[str] = Field(default=["desktop"])
    alert_thresholds: Dict[str, str] = Field(default_factory=dict)
    quiet_hours_start: Optional[str] = "22:00"
    quiet_hours_end: Optional[str] = "08:00"
    focus_mode_duration: int = 60
    escalation_delay: int = 15
    custom_keywords: List[str] = Field(default_factory=list)


class DashboardPreferencesRequest(BaseModel):
    """Request model for dashboard preferences"""
    layout: str = "detailed"
    visible_widgets: List[str] = Field(default=["alerts", "agents", "performance", "tasks"])
    widget_order: List[str] = Field(default_factory=list)
    refresh_interval: int = 30
    color_theme: str = "auto"
    show_debug_info: bool = False
    mobile_layout: Optional[str] = "compact"


class UsageTrackingRequest(BaseModel):
    """Request model for usage tracking"""
    command: str
    response_time: Optional[float] = None
    success: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DataCollectionRequest(BaseModel):
    """Request model for ML data collection"""
    data_type: str
    value: Any
    labels: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IntelligencePredictionRequest(BaseModel):
    """Request model for intelligence predictions"""
    model_id: str
    input_data: Dict[str, Any]


class SignalRelevanceRequest(BaseModel):
    """Request model for signal relevance measurement"""
    signals: List[Dict[str, Any]]
    user_feedback: List[Dict[str, Any]]


# API Endpoints

@router.post("/alerts/analyze", response_model=AlertAnalysisResponse)
async def analyze_alert(
    request: AlertAnalysisRequest,
    analysis_engine: AlertAnalysisEngine = Depends(create_alert_analysis_engine)
):
    """Analyze an alert and return enriched data with patterns"""
    try:
        alert_data = {
            'id': str(uuid4()),
            'type': request.alert_type,
            'severity': request.severity,
            'message': request.message,
            'timestamp': request.timestamp.isoformat(),
            'metadata': request.metadata
        }
        
        # Perform analysis
        enriched_data = await analysis_engine.analyze_alert(alert_data)
        
        analysis_result = enriched_data.get('analysis', {})
        
        return AlertAnalysisResponse(
            alert_id=alert_data['id'],
            analysis=analysis_result,
            patterns=analysis_result.get('patterns', []),
            recommendations=[analysis_result.get('recommended_action', 'Monitor for patterns')],
            priority_score=analysis_result.get('priority_score', 0.5)
        )
        
    except Exception as e:
        logger.error(f"Error analyzing alert: {e}")
        raise HTTPException(status_code=500, detail=f"Alert analysis failed: {str(e)}")


@router.get("/alerts/metrics")
async def get_alert_metrics(
    hours: int = 24,
    analysis_engine: AlertAnalysisEngine = Depends(create_alert_analysis_engine)
):
    """Get comprehensive alert metrics"""
    try:
        time_window = timedelta(hours=hours)
        metrics = await analysis_engine.get_alert_metrics(time_window)
        
        return {
            'time_window_hours': hours,
            'frequency_per_hour': metrics.frequency,
            'severity_distribution': metrics.severity_distribution,
            'average_response_time': metrics.response_time_avg,
            'resolution_rate': metrics.resolution_rate,
            'peak_hours': metrics.peak_hours,
            'correlation_score': metrics.correlation_score
        }
        
    except Exception as e:
        logger.error(f"Error getting alert metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alert metrics: {str(e)}")


@router.get("/alerts/patterns")
async def get_alert_patterns(
    analysis_engine: AlertAnalysisEngine = Depends(create_alert_analysis_engine)
):
    """Get insights about detected alert patterns"""
    try:
        insights = await analysis_engine.get_pattern_insights()
        return insights
        
    except Exception as e:
        logger.error(f"Error getting alert patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alert patterns: {str(e)}")


@router.get("/preferences/{user_id}")
async def get_user_preferences(
    user_id: str,
    preference_system: UserPreferenceSystem = Depends(create_user_preference_system)
):
    """Get user preferences"""
    try:
        preferences = await preference_system.get_user_preferences(user_id)
        return preferences
        
    except Exception as e:
        logger.error(f"Error getting user preferences: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get preferences: {str(e)}")


@router.put("/preferences/{user_id}/notifications")
async def update_notification_preferences(
    user_id: str,
    request: NotificationPreferencesRequest,
    preference_system: UserPreferenceSystem = Depends(create_user_preference_system)
):
    """Update user notification preferences"""
    try:
        # Convert request to NotificationPreferences
        preferences = NotificationPreferences(
            enabled_channels=[NotificationChannel(ch) for ch in request.enabled_channels],
            alert_thresholds={k: AlertPriority(v) for k, v in request.alert_thresholds.items()},
            quiet_hours_start=request.quiet_hours_start,
            quiet_hours_end=request.quiet_hours_end,
            focus_mode_duration=request.focus_mode_duration,
            escalation_delay=request.escalation_delay,
            custom_keywords=request.custom_keywords
        )
        
        success = await preference_system.update_notification_preferences(user_id, preferences)
        
        if success:
            return {"status": "success", "message": "Notification preferences updated"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update preferences")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid preference value: {str(e)}")
    except Exception as e:
        logger.error(f"Error updating notification preferences: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update preferences: {str(e)}")


@router.put("/preferences/{user_id}/dashboard")
async def update_dashboard_preferences(
    user_id: str,
    request: DashboardPreferencesRequest,
    preference_system: UserPreferenceSystem = Depends(create_user_preference_system)
):
    """Update user dashboard preferences"""
    try:
        # Convert request to DashboardPreferences
        preferences = DashboardPreferences(
            layout=DashboardLayout(request.layout),
            visible_widgets=request.visible_widgets,
            widget_order=request.widget_order,
            refresh_interval=request.refresh_interval,
            color_theme=ColorTheme(request.color_theme),
            show_debug_info=request.show_debug_info,
            mobile_layout=DashboardLayout(request.mobile_layout) if request.mobile_layout else None
        )
        
        success = await preference_system.update_dashboard_preferences(user_id, preferences)
        
        if success:
            return {"status": "success", "message": "Dashboard preferences updated"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update preferences")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid preference value: {str(e)}")
    except Exception as e:
        logger.error(f"Error updating dashboard preferences: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update preferences: {str(e)}")


@router.post("/preferences/{user_id}/usage")
async def track_usage(
    user_id: str,
    request: UsageTrackingRequest,
    background_tasks: BackgroundTasks,
    preference_system: UserPreferenceSystem = Depends(create_user_preference_system)
):
    """Track user command usage for pattern analysis"""
    try:
        # Track usage in background to avoid blocking API response
        background_tasks.add_task(
            preference_system.track_usage,
            user_id,
            request.command,
            request.response_time,
            request.success,
            request.metadata
        )
        
        return {"status": "success", "message": "Usage tracked"}
        
    except Exception as e:
        logger.error(f"Error tracking usage: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to track usage: {str(e)}")


@router.get("/preferences/{user_id}/dashboard-config")
async def get_personalized_dashboard_config(
    user_id: str,
    preference_system: UserPreferenceSystem = Depends(create_user_preference_system)
):
    """Get personalized dashboard configuration"""
    try:
        config = await preference_system.get_personalized_dashboard_config(user_id)
        return config
        
    except Exception as e:
        logger.error(f"Error getting dashboard config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard config: {str(e)}")


@router.get("/preferences/{user_id}/insights")
async def get_productivity_insights(
    user_id: str,
    preference_system: UserPreferenceSystem = Depends(create_user_preference_system)
):
    """Get productivity insights for user"""
    try:
        insights = await preference_system.generate_productivity_insights(user_id)
        return {
            'user_id': insights.user_id,
            'efficiency_score': insights.efficiency_score,
            'most_used_commands': insights.most_used_commands,
            'improvement_suggestions': insights.improvement_suggestions,
            'workflow_optimizations': insights.workflow_optimizations,
            'productivity_trends': insights.productivity_trends,
            'generated_at': insights.generated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting productivity insights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")


@router.post("/preferences/{user_id}/notification-check")
async def check_notification_preferences(
    user_id: str,
    alert_type: str,
    severity: str,
    preference_system: UserPreferenceSystem = Depends(create_user_preference_system)
):
    """Check if notification should be sent based on user preferences"""
    try:
        should_send, channels = await preference_system.should_send_notification(
            user_id, alert_type, severity
        )
        
        return {
            'should_send': should_send,
            'channels': [ch.value for ch in channels],
            'user_id': user_id,
            'alert_type': alert_type,
            'severity': severity
        }
        
    except Exception as e:
        logger.error(f"Error checking notification preferences: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check preferences: {str(e)}")


@router.post("/ml/collect-data")
async def collect_ml_data(
    request: DataCollectionRequest,
    user_id: str,
    background_tasks: BackgroundTasks,
    intelligence_framework: IntelligenceFramework = Depends(create_intelligence_framework)
):
    """Collect data for ML training and analysis"""
    try:
        # Collect data in background
        background_tasks.add_task(
            intelligence_framework.collect_data,
            DataType(request.data_type),
            request.value,
            user_id,
            request.labels,
            request.metadata
        )
        
        return {"status": "success", "message": "Data collected for ML training"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid data type: {str(e)}")
    except Exception as e:
        logger.error(f"Error collecting ML data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to collect data: {str(e)}")


@router.post("/ml/predict")
async def get_intelligence_prediction(
    request: IntelligencePredictionRequest,
    intelligence_framework: IntelligenceFramework = Depends(create_intelligence_framework)
):
    """Get prediction from intelligence model"""
    try:
        prediction = await intelligence_framework.get_intelligence_prediction(
            request.model_id,
            request.input_data
        )
        
        if prediction is None:
            raise HTTPException(status_code=404, detail="Model not found or not deployed")
        
        return {
            'prediction_id': prediction.prediction_id,
            'model_id': prediction.model_id,
            'prediction': prediction.prediction,
            'confidence': prediction.confidence,
            'explanation': prediction.explanation,
            'alternatives': prediction.alternatives,
            'timestamp': prediction.timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/ml/measure-signal-relevance")
async def measure_signal_relevance(
    request: SignalRelevanceRequest,
    intelligence_framework: IntelligenceFramework = Depends(create_intelligence_framework)
):
    """Measure signal relevance improvement"""
    try:
        metrics = await intelligence_framework.measure_signal_relevance(
            request.signals,
            request.user_feedback
        )
        
        return {
            'total_signals': metrics.total_signals,
            'relevant_signals': metrics.relevant_signals,
            'false_positives': metrics.false_positives,
            'false_negatives': metrics.false_negatives,
            'relevance_ratio': metrics.relevance_ratio,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'noise_reduction_percentage': metrics.noise_reduction_percentage,
            'user_satisfaction_score': metrics.user_satisfaction_score
        }
        
    except Exception as e:
        logger.error(f"Error measuring signal relevance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to measure relevance: {str(e)}")


@router.get("/ml/insights")
async def get_intelligence_insights(
    intelligence_framework: IntelligenceFramework = Depends(create_intelligence_framework)
):
    """Get comprehensive intelligence system insights"""
    try:
        insights = await intelligence_framework.get_intelligence_insights()
        return insights
        
    except Exception as e:
        logger.error(f"Error getting intelligence insights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get insights: {str(e)}")


@router.get("/performance/trends")
async def get_performance_trends(
    metrics: Dict[str, float],
    analysis_engine: AlertAnalysisEngine = Depends(create_alert_analysis_engine)
):
    """Get performance trend analysis"""
    try:
        trends = await analysis_engine.detect_performance_trends(metrics)
        
        return {
            'trends': [
                {
                    'metric_name': trend.metric_name,
                    'baseline_value': trend.baseline_value,
                    'current_value': trend.current_value,
                    'trend_direction': trend.trend_direction,
                    'change_percentage': trend.change_percentage,
                    'confidence': trend.confidence,
                    'anomaly_detected': trend.anomaly_detected
                }
                for trend in trends
            ],
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance trends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")


@router.get("/status")
async def get_intelligence_system_status():
    """Get intelligence system status"""
    try:
        return {
            'status': 'operational',
            'phase': '3 - Intelligence Layer Foundation',
            'capabilities': [
                'Alert pattern recognition',
                'User preference personalization',
                'Basic ML framework',
                'Signal relevance measurement',
                'Performance trend detection'
            ],
            'progress_toward_90_percent_signal_relevance': 'Foundation established',
            'next_phase': 'Advanced ML model deployment',
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


# Health check endpoint
@router.get("/health")
async def health_check():
    """Health check for intelligence system"""
    return {
        'status': 'healthy',
        'service': 'intelligence-layer',
        'version': '3.0.0',
        'timestamp': datetime.now().isoformat()
    }