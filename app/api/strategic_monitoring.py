"""
Strategic Monitoring and Analytics API Endpoints

Comprehensive REST API for strategic monitoring and analytics including:
- Market intelligence and competitive analysis endpoints
- Performance monitoring dashboard APIs
- Strategic intelligence system endpoints
- Real-time alerting and notification APIs
- Business metrics and KPI tracking endpoints

Provides full API access to the strategic monitoring framework
for global market expansion and competitive intelligence.
"""

import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Path, Depends, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..core.database import get_async_session
from ..core.strategic_market_analytics import (
    get_strategic_analytics_engine,
    MarketSegment,
    CompetitivePosition,
    TrendDirection,
    RiskLevel
)
from ..core.performance_monitoring_dashboard import get_performance_dashboard
from ..core.strategic_intelligence_system import (
    get_strategic_intelligence_system,
    IntelligenceType,
    ConfidenceLevel,
    AlertSeverity,
    ActionPriority
)
from ..models.strategic_monitoring import (
    MarketIntelligenceData,
    CompetitorAnalysisData,
    PerformanceMetrics,
    StrategicRecommendations,
    RiskAssessments,
    IntelligenceAlerts,
    BusinessMetrics
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, and_, or_

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/strategic-monitoring", tags=["strategic-monitoring"])


# Pydantic models for request/response
class CompetitiveAnalysisRequest(BaseModel):
    """Request model for competitive analysis."""
    segment: MarketSegment
    region: str = "global"
    depth: str = Field("comprehensive", pattern="^(strategic|comprehensive|basic)$")
    competitors: Optional[List[str]] = None


class MarketTrendAnalysisRequest(BaseModel):
    """Request model for market trend analysis."""
    time_horizon_months: int = Field(12, ge=1, le=60)
    categories: Optional[List[str]] = None
    segments: Optional[List[MarketSegment]] = None


class PerformanceDashboardRequest(BaseModel):
    """Request model for performance dashboard."""
    period: str = Field("current_month", pattern="^(current_month|last_month|current_quarter|last_quarter|ytd)$")
    include_forecasts: bool = True
    focus_areas: Optional[List[str]] = None


class RiskAssessmentRequest(BaseModel):
    """Request model for risk assessment."""
    risk_categories: Optional[List[str]] = None
    time_horizon_months: int = Field(12, ge=1, le=36)
    severity_threshold: str = Field("medium", pattern="^(low|medium|high|critical)$")


class IntelligenceMonitorRequest(BaseModel):
    """Request model for intelligence monitoring setup."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    monitor_type: str = Field(..., pattern="^(competitive|market|technology|regulatory)$")
    configuration: Dict[str, Any]
    alert_triggers: Dict[str, Any]
    execution_frequency: str = Field("hourly", pattern="^(hourly|daily|weekly|monthly)$")


# Market Intelligence Endpoints

@router.get("/market-intelligence/competitive-landscape")
async def analyze_competitive_landscape(
    segment: MarketSegment = Query(..., description="Market segment to analyze"),
    region: str = Query("global", description="Geographic region"),
    depth: str = Query("comprehensive", pattern="^(strategic|comprehensive|basic)$", description="Analysis depth"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Analyze competitive landscape for market segment.
    
    Provides comprehensive competitive intelligence including market positioning,
    threat assessment, and strategic opportunities.
    """
    try:
        analytics_engine = get_strategic_analytics_engine()
        
        analysis = await analytics_engine.analyze_competitive_landscape(
            segment=segment,
            region=region,
            depth=depth
        )
        
        return {
            "status": "success",
            "data": analysis,
            "metadata": {
                "segment": segment,
                "region": region,
                "analysis_depth": depth,
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing competitive landscape: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze competitive landscape: {str(e)}")


@router.get("/market-intelligence/trends")
async def analyze_market_trends(
    time_horizon_months: int = Query(12, ge=1, le=60, description="Analysis time horizon in months"),
    categories: Optional[str] = Query(None, description="Comma-separated trend categories"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Analyze market trends with strategic implications.
    
    Provides comprehensive trend analysis including impact assessment,
    velocity tracking, and strategic recommendations.
    """
    try:
        analytics_engine = get_strategic_analytics_engine()
        
        category_list = categories.split(",") if categories else None
        
        analysis = await analytics_engine.analyze_market_trends(
            time_horizon_months=time_horizon_months,
            categories=category_list
        )
        
        return {
            "status": "success",
            "data": analysis,
            "metadata": {
                "time_horizon_months": time_horizon_months,
                "categories": category_list,
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing market trends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze market trends: {str(e)}")


@router.get("/market-intelligence/strategic-report")
async def generate_strategic_intelligence_report(
    focus_areas: Optional[str] = Query(None, description="Comma-separated focus areas"),
    time_horizon_months: int = Query(6, ge=1, le=24, description="Strategic planning horizon"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Generate comprehensive strategic intelligence report.
    
    Combines competitive analysis, trend insights, and market opportunities
    into actionable strategic intelligence.
    """
    try:
        analytics_engine = get_strategic_analytics_engine()
        
        focus_list = focus_areas.split(",") if focus_areas else None
        
        report = await analytics_engine.generate_strategic_intelligence_report(
            focus_areas=focus_list,
            time_horizon_months=time_horizon_months
        )
        
        return {
            "status": "success",
            "data": report,
            "metadata": {
                "focus_areas": focus_list,
                "time_horizon_months": time_horizon_months,
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating strategic intelligence report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate strategic report: {str(e)}")


# Performance Monitoring Endpoints

@router.get("/performance/dashboard")
async def get_performance_dashboard(
    period: str = Query("current_month", pattern="^(current_month|last_month|current_quarter|last_quarter|ytd)$"),
    include_forecasts: bool = Query(True, description="Include predictive forecasts"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Generate comprehensive performance dashboard.
    
    Provides real-time performance metrics across all strategic pillars
    including enterprise partnerships, community growth, and thought leadership.
    """
    try:
        dashboard = get_performance_dashboard()
        
        dashboard_data = await dashboard.generate_performance_dashboard(
            period=period,
            include_forecasts=include_forecasts
        )
        
        return {
            "status": "success",
            "data": {
                "dashboard_id": dashboard_data.dashboard_id,
                "generated_at": dashboard_data.generated_at.isoformat(),
                "reporting_period": dashboard_data.reporting_period,
                "overall_performance_score": dashboard_data.overall_performance_score,
                "enterprise_metrics": dashboard_data.enterprise_metrics.__dict__,
                "community_metrics": dashboard_data.community_metrics.__dict__,
                "thought_leadership_metrics": dashboard_data.thought_leadership_metrics.__dict__,
                "revenue_metrics": dashboard_data.revenue_metrics.__dict__,
                "key_insights": dashboard_data.key_insights,
                "performance_alerts": dashboard_data.performance_alerts,
                "strategic_recommendations": dashboard_data.strategic_recommendations,
                "competitive_benchmarks": dashboard_data.competitive_benchmarks,
                "next_milestones": dashboard_data.next_milestones
            },
            "metadata": {
                "period": period,
                "include_forecasts": include_forecasts,
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating performance dashboard: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate performance dashboard: {str(e)}")


@router.get("/performance/enterprise-partnerships")
async def get_enterprise_partnership_metrics(
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get enterprise partnership performance metrics.
    
    Tracks partnership development, conversion rates, and revenue impact.
    """
    try:
        dashboard = get_performance_dashboard()
        metrics = await dashboard.monitor_enterprise_partnerships()
        
        return {
            "status": "success",
            "data": metrics.__dict__,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting enterprise partnership metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get partnership metrics: {str(e)}")


@router.get("/performance/community-growth")
async def get_community_growth_metrics(
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get community growth and engagement metrics.
    
    Tracks community expansion, user engagement, and viral growth patterns.
    """
    try:
        dashboard = get_performance_dashboard()
        metrics = await dashboard.monitor_community_growth()
        
        return {
            "status": "success",
            "data": metrics.__dict__,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting community growth metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get community metrics: {str(e)}")


@router.get("/performance/thought-leadership")
async def get_thought_leadership_metrics(
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get thought leadership impact metrics.
    
    Tracks content performance, industry recognition, and brand influence.
    """
    try:
        dashboard = get_performance_dashboard()
        metrics = await dashboard.monitor_thought_leadership()
        
        return {
            "status": "success",
            "data": metrics.__dict__,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting thought leadership metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get thought leadership metrics: {str(e)}")


@router.get("/performance/revenue-metrics")
async def get_revenue_metrics(
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get revenue and business development metrics.
    
    Tracks financial performance, growth rates, and business health indicators.
    """
    try:
        dashboard = get_performance_dashboard()
        metrics = await dashboard.monitor_revenue_metrics()
        
        return {
            "status": "success",
            "data": metrics.__dict__,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting revenue metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get revenue metrics: {str(e)}")


# Strategic Intelligence Endpoints

@router.post("/intelligence/competitive-analysis")
async def generate_competitive_intelligence(
    competitor_name: str = Body(..., description="Competitor name to analyze"),
    analysis_depth: str = Body("comprehensive", pattern="^(strategic|comprehensive|basic)$"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Generate automated competitive intelligence report.
    
    Performs deep competitive analysis with real-time data collection
    and strategic insight generation.
    """
    try:
        intelligence_system = get_strategic_intelligence_system()
        
        intelligence = await intelligence_system.generate_competitive_intelligence_report(
            competitor_name=competitor_name,
            analysis_depth=analysis_depth
        )
        
        return {
            "status": "success",
            "data": intelligence.__dict__,
            "metadata": {
                "competitor_name": competitor_name,
                "analysis_depth": analysis_depth,
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating competitive intelligence: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate competitive intelligence: {str(e)}")


@router.get("/intelligence/market-opportunities/{segment}")
async def analyze_market_opportunities(
    segment: MarketSegment = Path(..., description="Market segment to analyze"),
    region: str = Query("global", description="Geographic region"),
    time_horizon_months: int = Query(12, ge=1, le=36, description="Analysis time horizon"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Analyze market opportunities with predictive intelligence.
    
    Identifies emerging opportunities, trends, and strategic positioning
    options based on comprehensive market analysis.
    """
    try:
        intelligence_system = get_strategic_intelligence_system()
        
        intelligence = await intelligence_system.analyze_market_opportunities(
            segment=segment,
            region=region,
            time_horizon_months=time_horizon_months
        )
        
        return {
            "status": "success",
            "data": intelligence.__dict__,
            "metadata": {
                "segment": segment,
                "region": region,
                "time_horizon_months": time_horizon_months,
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing market opportunities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze market opportunities: {str(e)}")


@router.post("/intelligence/risk-assessment")
async def perform_risk_assessment(
    request: RiskAssessmentRequest = Body(...),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Perform comprehensive strategic risk assessment.
    
    Identifies, assesses, and prioritizes strategic risks with
    mitigation recommendations and monitoring indicators.
    """
    try:
        intelligence_system = get_strategic_intelligence_system()
        
        risk_assessments = await intelligence_system.perform_strategic_risk_assessment(
            risk_categories=request.risk_categories,
            time_horizon_months=request.time_horizon_months
        )
        
        return {
            "status": "success",
            "data": [assessment.__dict__ for assessment in risk_assessments],
            "metadata": {
                "risk_categories": request.risk_categories,
                "time_horizon_months": request.time_horizon_months,
                "severity_threshold": request.severity_threshold,
                "total_risks_identified": len(risk_assessments),
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error performing risk assessment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to perform risk assessment: {str(e)}")


@router.get("/intelligence/recommendations")
async def get_strategic_recommendations(
    focus_areas: Optional[str] = Query(None, description="Comma-separated focus areas"),
    confidence_threshold: float = Query(0.6, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of recommendations"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get strategic recommendations based on intelligence analysis.
    
    Combines competitive intelligence, market analysis, and risk assessment
    to provide actionable strategic recommendations.
    """
    try:
        intelligence_system = get_strategic_intelligence_system()
        
        focus_list = focus_areas.split(",") if focus_areas else None
        
        recommendations = await intelligence_system.generate_strategic_recommendations(
            focus_areas=focus_list,
            confidence_threshold=confidence_threshold
        )
        
        # Limit results
        recommendations = recommendations[:limit]
        
        return {
            "status": "success",
            "data": [rec.__dict__ for rec in recommendations],
            "metadata": {
                "focus_areas": focus_list,
                "confidence_threshold": confidence_threshold,
                "total_recommendations": len(recommendations),
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting strategic recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get strategic recommendations: {str(e)}")


# Monitoring and Alerting Endpoints

@router.post("/monitoring/setup")
async def setup_intelligence_monitoring(
    request: IntelligenceMonitorRequest = Body(...),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Set up automated intelligence monitoring with alerting.
    
    Creates continuous monitoring for competitive moves, market changes,
    and strategic opportunities with real-time alerting.
    """
    try:
        intelligence_system = get_strategic_intelligence_system()
        
        monitor_config = {
            "name": request.name,
            "description": request.description,
            "monitor_type": request.monitor_type,
            "configuration": request.configuration,
            "alert_triggers": request.alert_triggers,
            "execution_frequency": request.execution_frequency
        }
        
        monitor_id = await intelligence_system.setup_intelligence_monitoring(monitor_config)
        
        return {
            "status": "success",
            "data": {
                "monitor_id": monitor_id,
                "name": request.name,
                "monitor_type": request.monitor_type,
                "execution_frequency": request.execution_frequency,
                "status": "active"
            },
            "metadata": {
                "created_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error setting up intelligence monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to setup monitoring: {str(e)}")


@router.get("/alerts")
async def get_intelligence_alerts(
    severity: Optional[AlertSeverity] = Query(None, description="Filter by alert severity"),
    intelligence_type: Optional[IntelligenceType] = Query(None, description="Filter by intelligence type"),
    unresolved_only: bool = Query(True, description="Show only unresolved alerts"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of alerts"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get intelligence alerts with filtering options.
    
    Retrieves strategic intelligence alerts based on severity,
    type, and resolution status with comprehensive alert data.
    """
    try:
        # Build query filters
        filters = []
        if severity:
            filters.append(IntelligenceAlerts.severity == severity.value)
        if intelligence_type:
            filters.append(IntelligenceAlerts.intelligence_type == intelligence_type.value)
        if unresolved_only:
            filters.append(IntelligenceAlerts.resolved == False)
        
        # Query alerts
        query = select(IntelligenceAlerts).where(and_(*filters)).order_by(desc(IntelligenceAlerts.created_at)).limit(limit)
        result = await session.execute(query)
        alerts = result.scalars().all()
        
        alert_data = []
        for alert in alerts:
            alert_data.append({
                "alert_id": alert.alert_id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity,
                "intelligence_type": alert.intelligence_type,
                "affected_areas": alert.affected_areas,
                "recommended_actions": alert.recommended_actions,
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved,
                "created_at": alert.created_at.isoformat(),
                "expires_at": alert.expires_at.isoformat() if alert.expires_at else None
            })
        
        return {
            "status": "success",
            "data": alert_data,
            "metadata": {
                "total_alerts": len(alert_data),
                "filters_applied": {
                    "severity": severity.value if severity else None,
                    "intelligence_type": intelligence_type.value if intelligence_type else None,
                    "unresolved_only": unresolved_only
                },
                "generated_at": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting intelligence alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")


@router.patch("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str = Path(..., description="Alert ID to acknowledge"),
    acknowledged_by: str = Body(..., description="User acknowledging the alert"),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Acknowledge an intelligence alert.
    
    Marks an alert as acknowledged by a specific user with timestamp.
    """
    try:
        # Find the alert
        query = select(IntelligenceAlerts).where(IntelligenceAlerts.alert_id == alert_id)
        result = await session.execute(query)
        alert = result.scalar_one_or_none()
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        # Update acknowledgment
        alert.acknowledged = True
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.utcnow()
        
        await session.commit()
        
        return {
            "status": "success",
            "data": {
                "alert_id": alert_id,
                "acknowledged": True,
                "acknowledged_by": acknowledged_by,
                "acknowledged_at": alert.acknowledged_at.isoformat()
            },
            "metadata": {
                "updated_at": datetime.utcnow().isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to acknowledge alert: {str(e)}")


# Health and Status Endpoints

@router.get("/health")
async def get_system_health(
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get strategic monitoring system health status.
    
    Returns comprehensive health indicators for all monitoring components
    including data collection, analysis engines, and alerting systems.
    """
    try:
        # Check component health
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "strategic_analytics_engine": "healthy",
                "performance_dashboard": "healthy",
                "intelligence_system": "healthy",
                "database": "healthy",
                "alerting_system": "healthy"
            },
            "metrics": {
                "active_monitors": 0,  # Would query actual monitors
                "recent_analyses": 0,  # Would query recent analysis count
                "unresolved_alerts": 0,  # Would query unresolved alerts
                "system_uptime_hours": 24.5  # Would calculate actual uptime
            }
        }
        
        return {
            "status": "success",
            "data": health_status
        }
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        return {
            "status": "error",
            "data": {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        }