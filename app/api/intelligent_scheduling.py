"""
Intelligent Scheduling API endpoints for advanced sleep-wake automation.

Provides sophisticated scheduling capabilities with machine learning-based
optimization, predictive scheduling, and adaptive pattern recognition.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, Query, Path, status
from pydantic import BaseModel, Field, validator

from ..core.intelligent_sleep_manager import get_intelligent_sleep_manager
from ..core.automated_orchestrator import get_automated_orchestrator
from ..core.sleep_analytics import get_sleep_analytics_engine
from ..core.security import get_current_user, require_analytics_access


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/intelligent-scheduling", tags=["Intelligent Scheduling"])


# Request/Response Models
class PatternAnalysisRequest(BaseModel):
    """Request model for activity pattern analysis."""
    agent_id: Optional[UUID] = Field(
        default=None, 
        description="Agent ID for analysis, None for system-wide"
    )
    analysis_period_days: int = Field(
        default=7, 
        ge=1, 
        le=90,
        description="Number of days to analyze"
    )
    pattern_types: List[str] = Field(
        default=["activity", "sleep", "consolidation"],
        description="Types of patterns to analyze"
    )
    include_predictions: bool = Field(
        default=True,
        description="Include predictive scheduling recommendations"
    )


class OptimalScheduleRequest(BaseModel):
    """Request model for optimal schedule generation."""
    agent_id: UUID = Field(..., description="Agent ID for schedule optimization")
    optimization_goal: str = Field(
        default="efficiency", 
        description="Optimization goal (efficiency, performance, resource_usage)"
    )
    time_horizon_hours: int = Field(
        default=24, 
        ge=1, 
        le=168,
        description="Time horizon for schedule optimization"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Scheduling constraints and preferences"
    )
    
    @validator('optimization_goal')
    def validate_optimization_goal(cls, v):
        allowed_goals = ["efficiency", "performance", "resource_usage", "availability"]
        if v not in allowed_goals:
            raise ValueError(f"optimization_goal must be one of {allowed_goals}")
        return v


class AdaptiveConfigRequest(BaseModel):
    """Request model for adaptive configuration updates."""
    agent_id: Optional[UUID] = Field(
        default=None, 
        description="Agent ID for configuration, None for global"
    )
    learning_rate: float = Field(
        default=0.1, 
        ge=0.01, 
        le=1.0,
        description="Learning rate for adaptive algorithms"
    )
    adaptation_window_hours: int = Field(
        default=6, 
        ge=1, 
        le=48,
        description="Window for adaptation analysis"
    )
    enable_predictive_scheduling: bool = Field(
        default=True,
        description="Enable predictive scheduling features"
    )
    auto_optimization_threshold: float = Field(
        default=0.8, 
        ge=0.5, 
        le=0.95,
        description="Threshold for automatic optimization triggers"
    )


class ScheduleConflictRequest(BaseModel):
    """Request model for schedule conflict resolution."""
    conflict_resolution_strategy: str = Field(
        default="intelligent", 
        description="Strategy for resolving conflicts"
    )
    priority_weights: Dict[str, float] = Field(
        default_factory=lambda: {"efficiency": 0.4, "availability": 0.3, "performance": 0.3},
        description="Weights for different priority factors"
    )
    allow_rescheduling: bool = Field(
        default=True,
        description="Allow rescheduling of existing operations"
    )
    
    @validator('conflict_resolution_strategy')
    def validate_strategy(cls, v):
        allowed = ["intelligent", "priority_based", "first_come_first_serve", "optimal"]
        if v not in allowed:
            raise ValueError(f"strategy must be one of {allowed}")
        return v


# Pattern Analysis and Learning Endpoints
@router.post("/patterns/analyze")
async def analyze_activity_patterns(
    request: PatternAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Analyze agent activity patterns with machine learning insights.
    
    Features:
    - Historical pattern recognition
    - Predictive behavior modeling
    - Anomaly detection and alerting
    - Optimization opportunity identification
    """
    try:
        logger.info(f"Analyzing activity patterns for agent {request.agent_id}")
        
        intelligent_manager = get_intelligent_sleep_manager()
        analytics_engine = get_sleep_analytics_engine()
        
        # Perform pattern analysis
        patterns = await intelligent_manager.analyze_activity_patterns(
            agent_id=request.agent_id,
            analysis_period=timedelta(days=request.analysis_period_days),
            pattern_types=request.pattern_types
        )
        
        # Get additional analytics if requested
        if request.include_predictions:
            predictions = await analytics_engine.get_predictive_insights(
                agent_id=request.agent_id,
                forecast_hours=24
            )
            patterns["predictions"] = predictions
        
        # Generate actionable insights
        insights = await intelligent_manager.generate_optimization_insights(
            patterns, request.agent_id
        )
        
        return {
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "agent_id": str(request.agent_id) if request.agent_id else "system-wide",
            "analysis_period_days": request.analysis_period_days,
            "patterns": patterns,
            "insights": insights,
            "recommendations": await _generate_pattern_recommendations(patterns, insights)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing patterns: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pattern analysis failed: {str(e)}"
        )


@router.post("/schedule/optimize")
async def generate_optimal_schedule(
    request: OptimalScheduleRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate optimal schedule using machine learning optimization.
    
    Features:
    - Multi-objective optimization
    - Constraint satisfaction
    - Resource allocation optimization
    - Performance prediction and validation
    """
    try:
        logger.info(f"Generating optimal schedule for agent {request.agent_id}")
        
        intelligent_manager = get_intelligent_sleep_manager()
        
        # Generate optimal schedule
        optimal_schedule = await intelligent_manager.generate_optimal_schedule(
            agent_id=request.agent_id,
            optimization_goal=request.optimization_goal,
            time_horizon=timedelta(hours=request.time_horizon_hours),
            constraints=request.constraints
        )
        
        # Validate schedule feasibility
        validation_results = await intelligent_manager.validate_schedule_feasibility(
            optimal_schedule, request.agent_id
        )
        
        # Calculate performance predictions
        performance_predictions = await intelligent_manager.predict_schedule_performance(
            optimal_schedule, request.agent_id
        )
        
        return {
            "schedule_id": optimal_schedule["id"],
            "agent_id": str(request.agent_id),
            "optimization_goal": request.optimization_goal,
            "schedule": optimal_schedule,
            "validation": validation_results,
            "performance_predictions": performance_predictions,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=request.time_horizon_hours)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating optimal schedule: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Schedule optimization failed: {str(e)}"
        )


@router.put("/configuration/adaptive")
async def update_adaptive_configuration(
    request: AdaptiveConfigRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Update adaptive learning configuration with real-time validation.
    
    Features:
    - Runtime configuration updates
    - Learning parameter optimization
    - Performance impact assessment
    - Rollback capability for failed updates
    """
    try:
        logger.info("Updating adaptive configuration")
        
        intelligent_manager = get_intelligent_sleep_manager()
        
        # Validate configuration impact
        impact_assessment = await intelligent_manager.assess_configuration_impact(
            agent_id=request.agent_id,
            new_config={
                "learning_rate": request.learning_rate,
                "adaptation_window": request.adaptation_window_hours,
                "predictive_scheduling": request.enable_predictive_scheduling,
                "auto_optimization_threshold": request.auto_optimization_threshold
            }
        )
        
        # Apply configuration if impact is acceptable
        if impact_assessment["safe_to_apply"]:
            update_result = await intelligent_manager.update_adaptive_configuration(
                agent_id=request.agent_id,
                learning_rate=request.learning_rate,
                adaptation_window_hours=request.adaptation_window_hours,
                enable_predictive_scheduling=request.enable_predictive_scheduling,
                auto_optimization_threshold=request.auto_optimization_threshold
            )
            
            return {
                "success": True,
                "message": "Adaptive configuration updated successfully",
                "agent_id": str(request.agent_id) if request.agent_id else "global",
                "impact_assessment": impact_assessment,
                "update_result": update_result,
                "rollback_id": update_result.get("rollback_id")
            }
        else:
            return {
                "success": False,
                "message": "Configuration update rejected due to safety concerns",
                "impact_assessment": impact_assessment,
                "recommendations": impact_assessment.get("recommendations", [])
            }
        
    except Exception as e:
        logger.error(f"Error updating adaptive configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration update failed: {str(e)}"
        )


# Advanced Scheduling Features
@router.post("/conflicts/resolve")
async def resolve_schedule_conflicts(
    request: ScheduleConflictRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Intelligent resolution of scheduling conflicts.
    
    Features:
    - Multi-criteria conflict resolution
    - Optimal rescheduling algorithms
    - Stakeholder impact minimization
    - Performance preservation guarantees
    """
    try:
        logger.info("Resolving schedule conflicts")
        
        orchestrator = get_automated_orchestrator()
        
        # Detect existing conflicts
        conflicts = await orchestrator.detect_schedule_conflicts()
        
        if not conflicts:
            return {
                "conflicts_found": False,
                "message": "No schedule conflicts detected",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Resolve conflicts using specified strategy
        resolution_results = await orchestrator.resolve_conflicts(
            conflicts=conflicts,
            strategy=request.conflict_resolution_strategy,
            priority_weights=request.priority_weights,
            allow_rescheduling=request.allow_rescheduling
        )
        
        return {
            "conflicts_found": True,
            "conflicts_resolved": len(resolution_results["resolved"]),
            "conflicts_remaining": len(resolution_results["unresolved"]),
            "resolution_strategy": request.conflict_resolution_strategy,
            "resolution_details": resolution_results,
            "rescheduled_operations": resolution_results.get("rescheduled", []),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error resolving conflicts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conflict resolution failed: {str(e)}"
        )


@router.get("/predictions/forecast")
async def get_predictive_forecast(
    agent_id: Optional[UUID] = Query(None, description="Agent ID for forecast"),
    forecast_hours: int = Query(24, ge=1, le=168, description="Forecast time horizon"),
    include_confidence: bool = Query(True, description="Include confidence intervals"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get predictive forecast for agent behavior and optimal scheduling.
    
    Features:
    - Machine learning-based predictions
    - Confidence interval calculation
    - Scenario analysis and what-if modeling
    - Risk assessment and mitigation suggestions
    """
    try:
        intelligent_manager = get_intelligent_sleep_manager()
        analytics_engine = get_sleep_analytics_engine()
        
        # Generate predictive forecast
        forecast = await intelligent_manager.generate_predictive_forecast(
            agent_id=agent_id,
            forecast_horizon=timedelta(hours=forecast_hours),
            include_confidence=include_confidence
        )
        
        # Add behavioral predictions
        behavioral_predictions = await analytics_engine.predict_agent_behavior(
            agent_id=agent_id,
            prediction_window=timedelta(hours=forecast_hours)
        )
        
        # Generate optimization opportunities
        optimization_opportunities = await intelligent_manager.identify_optimization_opportunities(
            forecast, behavioral_predictions
        )
        
        return {
            "forecast_id": f"forecast_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "agent_id": str(agent_id) if agent_id else "system-wide",
            "forecast_horizon_hours": forecast_hours,
            "forecast": forecast,
            "behavioral_predictions": behavioral_predictions,
            "optimization_opportunities": optimization_opportunities,
            "generated_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=forecast_hours)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecast generation failed: {str(e)}"
        )


@router.post("/learning/train")
async def trigger_model_training(
    agent_id: Optional[UUID] = Query(None, description="Agent ID for training"),
    training_type: str = Query("incremental", description="Training type"),
    force_retrain: bool = Query(False, description="Force complete retraining"),
    current_user: dict = Depends(get_current_user)
):
    """
    Trigger machine learning model training for improved predictions.
    
    Features:
    - Incremental and full retraining modes
    - Performance validation and rollback
    - A/B testing for model improvements
    - Training progress monitoring
    """
    try:
        logger.info(f"Triggering model training for agent {agent_id}")
        
        intelligent_manager = get_intelligent_sleep_manager()
        
        # Validate training parameters
        if training_type not in ["incremental", "full", "adaptive"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="training_type must be 'incremental', 'full', or 'adaptive'"
            )
        
        # Start training process
        training_result = await intelligent_manager.trigger_model_training(
            agent_id=agent_id,
            training_type=training_type,
            force_retrain=force_retrain
        )
        
        return {
            "training_initiated": True,
            "training_id": training_result["training_id"],
            "agent_id": str(agent_id) if agent_id else "system-wide",
            "training_type": training_type,
            "estimated_completion": training_result["estimated_completion"],
            "training_status": training_result["status"],
            "model_version": training_result["model_version"],
            "started_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error triggering training: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model training initiation failed: {str(e)}"
        )


@router.get("/performance/metrics")
async def get_scheduling_performance_metrics(
    agent_id: Optional[UUID] = Query(None, description="Agent ID for metrics"),
    time_window_hours: int = Query(24, ge=1, le=168, description="Time window for metrics"),
    include_predictions: bool = Query(True, description="Include predictive metrics"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get comprehensive scheduling performance metrics.
    
    Features:
    - Real-time performance tracking
    - Comparative analysis against baselines
    - Trend identification and alerting
    - ROI and efficiency calculations
    """
    try:
        analytics_engine = get_sleep_analytics_engine()
        intelligent_manager = get_intelligent_sleep_manager()
        
        # Get performance metrics
        metrics = await analytics_engine.get_scheduling_performance_metrics(
            agent_id=agent_id,
            time_window=timedelta(hours=time_window_hours)
        )
        
        # Add intelligent scheduling specific metrics
        intelligent_metrics = await intelligent_manager.get_performance_metrics(
            agent_id=agent_id,
            time_window=timedelta(hours=time_window_hours)
        )
        
        # Calculate efficiency improvements
        efficiency_analysis = await intelligent_manager.calculate_efficiency_improvements(
            agent_id=agent_id,
            time_window=timedelta(hours=time_window_hours)
        )
        
        response = {
            "metrics_id": f"metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "agent_id": str(agent_id) if agent_id else "system-wide",
            "time_window_hours": time_window_hours,
            "performance_metrics": metrics,
            "intelligent_metrics": intelligent_metrics,
            "efficiency_analysis": efficiency_analysis,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        if include_predictions:
            # Add predictive performance metrics
            predicted_metrics = await intelligent_manager.predict_future_performance(
                agent_id=agent_id,
                prediction_window=timedelta(hours=24)
            )
            response["predicted_performance"] = predicted_metrics
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Performance metrics retrieval failed: {str(e)}"
        )


# Helper Functions
async def _generate_pattern_recommendations(
    patterns: Dict[str, Any],
    insights: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate actionable recommendations based on pattern analysis."""
    recommendations = []
    
    try:
        # Analyze sleep patterns
        if "sleep_patterns" in patterns:
            sleep_efficiency = patterns["sleep_patterns"].get("efficiency", 0)
            if sleep_efficiency < 0.8:
                recommendations.append({
                    "type": "optimization",
                    "priority": "high",
                    "title": "Improve Sleep Cycle Efficiency",
                    "description": f"Sleep efficiency is {sleep_efficiency:.1%}, below optimal threshold",
                    "action": "Review sleep timing and consolidation parameters"
                })
        
        # Analyze activity patterns
        if "activity_patterns" in patterns:
            activity_variance = patterns["activity_patterns"].get("variance", 0)
            if activity_variance > 0.5:
                recommendations.append({
                    "type": "scheduling",
                    "priority": "medium",
                    "title": "Stabilize Activity Patterns",
                    "description": f"High activity variance detected ({activity_variance:.1%})",
                    "action": "Consider more consistent scheduling intervals"
                })
        
        # Analyze resource utilization
        if "resource_utilization" in insights:
            cpu_efficiency = insights["resource_utilization"].get("cpu_efficiency", 1.0)
            if cpu_efficiency < 0.7:
                recommendations.append({
                    "type": "resource",
                    "priority": "medium",
                    "title": "Optimize Resource Usage",
                    "description": f"CPU efficiency is {cpu_efficiency:.1%}",
                    "action": "Review consolidation algorithms and background tasks"
                })
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return [{
            "type": "error",
            "priority": "low",
            "title": "Recommendation Generation Failed",
            "description": str(e),
            "action": "Review system logs for details"
        }]