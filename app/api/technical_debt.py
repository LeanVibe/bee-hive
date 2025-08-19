"""
Technical Debt API Endpoints for LeanVibe Agent Hive 2.0

RESTful API endpoints providing comprehensive technical debt management capabilities,
integrating with the project index system for intelligent code quality analysis.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from urllib.parse import unquote

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, Path as FastAPIPath
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, delete, and_, or_, func, desc
from pydantic import BaseModel, Field, ValidationError

from ..core.config import settings
from ..core.database import get_session
from ..core.redis import get_redis_client, RedisClient

# Import project index dependencies
from ..models.project_index import (
    ProjectIndex, FileEntry, DependencyRelationship, AnalysisSession, 
    ProjectStatus, AnalysisStatus
)
from ..project_index.core import ProjectIndexer
from ..project_index.models import ProjectIndexConfig

# Import technical debt components
from ..project_index.debt_analyzer import TechnicalDebtAnalyzer, DebtAnalysisResult
from ..project_index.advanced_debt_detector import AdvancedDebtDetector
from ..project_index.historical_analyzer import HistoricalAnalyzer, DebtEvolutionResult
from ..project_index.debt_remediation_engine import (
    DebtRemediationEngine, RemediationPlan, RemediationRecommendation
)
from ..project_index.incremental_debt_analyzer import IncrementalDebtAnalyzer
from ..project_index.debt_monitor_integration import DebtMonitorIntegration, DebtMonitorConfig

# Mock auth functions (reuse from project_index.py pattern)
async def get_current_user():
    """Mock current user - replace with actual implementation."""
    return "test_user"

async def get_event_publisher():
    """Mock event publisher - replace with actual implementation."""
    class MockEventPublisher:
        async def publish(self, event):
            logger.debug("Mock event published", event=event)
    return MockEventPublisher()

logger = structlog.get_logger()

# Create API router for technical debt management
router = APIRouter(
    prefix="/api/technical-debt",
    tags=["Technical Debt"],
    responses={
        400: {"description": "Bad request - validation error or invalid parameters"},
        401: {"description": "Unauthorized - authentication required"},
        403: {"description": "Forbidden - insufficient permissions"},
        404: {"description": "Not found - resource does not exist"},
        409: {"description": "Conflict - resource state conflict"},
        429: {"description": "Too Many Requests - rate limit exceeded"},
        500: {"description": "Internal Server Error - unexpected server error"}
    }
)


# ================== PYDANTIC MODELS ==================

class StandardResponse(BaseModel):
    """Standard API response format."""
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = None


class DebtAnalysisRequest(BaseModel):
    """Request model for debt analysis."""
    include_advanced_patterns: bool = True
    include_historical_analysis: bool = False
    analysis_depth: str = Field("standard", pattern="^(quick|standard|comprehensive)$")
    file_patterns: Optional[List[str]] = Field(None, description="File patterns to include (e.g., ['*.py', '*.js'])")
    exclude_patterns: Optional[List[str]] = Field(None, description="File patterns to exclude")
    

class DebtItemResponse(BaseModel):
    """Response model for individual debt items."""
    id: str
    project_id: str
    file_id: str
    debt_type: str
    category: str
    severity: str
    status: str
    description: str
    location: Dict[str, Any]
    debt_score: float
    confidence_score: float
    remediation_suggestion: str
    estimated_effort_hours: int
    first_detected_at: Optional[datetime]
    last_detected_at: Optional[datetime]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DebtAnalysisResponse(BaseModel):
    """Response model for debt analysis results."""
    project_id: str
    analysis_id: str
    total_debt_score: float
    debt_items: List[DebtItemResponse]
    category_breakdown: Dict[str, float]
    severity_breakdown: Dict[str, int]
    file_count: int
    lines_of_code: int
    analysis_duration_seconds: float
    recommendations: List[str]
    analysis_timestamp: datetime
    advanced_patterns_detected: Optional[int] = None


class HistoricalAnalysisResponse(BaseModel):
    """Response model for historical debt analysis."""
    project_id: str
    lookback_days: int
    evolution_timeline: List[Dict[str, Any]]
    trend_analysis: Dict[str, Any]
    debt_hotspots: List[Dict[str, Any]]
    category_trends: Dict[str, Any]
    recommendations: List[str]
    analysis_timestamp: datetime


class RemediationPlanResponse(BaseModel):
    """Response model for remediation plans."""
    project_id: str
    plan_id: str
    scope: str
    target_path: str
    recommendations_count: int
    execution_phases: List[List[str]]
    total_debt_reduction: float
    total_effort_estimate: float
    total_risk_score: float
    estimated_duration_days: int
    immediate_actions: List[str]
    quick_wins: List[str]
    long_term_goals: List[str]
    success_criteria: List[str]
    potential_blockers: List[str]
    created_at: datetime


class RemediationRecommendationResponse(BaseModel):
    """Response model for individual remediation recommendations."""
    id: str
    strategy: str
    priority: str
    impact: str
    title: str
    description: str
    rationale: str
    file_path: str
    line_ranges: List[List[int]]
    debt_reduction_score: float
    implementation_effort: float
    risk_level: float
    cost_benefit_ratio: float
    suggested_approach: str
    code_examples: List[str]
    related_patterns: List[str]
    dependencies: List[str]
    debt_categories: List[str]


class MonitoringStatusResponse(BaseModel):
    """Response model for debt monitoring status."""
    enabled: bool
    active_since: Optional[datetime]
    monitored_projects_count: int
    total_files_monitored: int
    total_debt_events: int
    configuration: Dict[str, Any]
    projects: Dict[str, Any]


# ================== DEPENDENCY INJECTION ==================

async def get_technical_debt_analyzer(
    session: AsyncSession = Depends(get_session)
) -> TechnicalDebtAnalyzer:
    """Get TechnicalDebtAnalyzer instance."""
    return TechnicalDebtAnalyzer()


async def get_advanced_debt_detector(
    session: AsyncSession = Depends(get_session)
) -> AdvancedDebtDetector:
    """Get AdvancedDebtDetector instance."""
    debt_analyzer = TechnicalDebtAnalyzer()
    from ..project_index.ml_analyzer import MLAnalyzer
    ml_analyzer = MLAnalyzer()
    historical_analyzer = HistoricalAnalyzer()
    
    return AdvancedDebtDetector(
        debt_analyzer=debt_analyzer,
        ml_analyzer=ml_analyzer,
        historical_analyzer=historical_analyzer
    )


async def get_historical_analyzer() -> HistoricalAnalyzer:
    """Get HistoricalAnalyzer instance."""
    return HistoricalAnalyzer()


async def get_debt_remediation_engine(
    session: AsyncSession = Depends(get_session)
) -> DebtRemediationEngine:
    """Get DebtRemediationEngine instance."""
    debt_analyzer = TechnicalDebtAnalyzer()
    advanced_detector = await get_advanced_debt_detector(session)
    historical_analyzer = HistoricalAnalyzer()
    
    return DebtRemediationEngine(
        debt_analyzer=debt_analyzer,
        advanced_detector=advanced_detector,
        historical_analyzer=historical_analyzer
    )


async def get_debt_monitor_integration(
    redis_client: RedisClient = Depends(get_redis_client)
) -> DebtMonitorIntegration:
    """Get DebtMonitorIntegration instance."""
    config = DebtMonitorConfig(enabled=True)
    return DebtMonitorIntegration(config)


async def get_project_or_404(
    project_id: uuid.UUID,
    session: AsyncSession = Depends(get_session)
) -> ProjectIndex:
    """Get project by ID or raise 404."""
    stmt = select(ProjectIndex).where(ProjectIndex.id == project_id).options(
        selectinload(ProjectIndex.file_entries),
        selectinload(ProjectIndex.dependency_relationships)
    )
    result = await session.execute(stmt)
    project = result.scalar_one_or_none()
    
    if not project:
        raise HTTPException(
            status_code=404,
            detail=f"Project {project_id} not found"
        )
    
    return project


# ================== API ENDPOINTS ==================

@router.post(
    "/{project_id}/analyze",
    response_model=StandardResponse,
    summary="Analyze Technical Debt",
    description="Perform comprehensive technical debt analysis on a project"
)
async def analyze_technical_debt(
    project_id: uuid.UUID,
    request: DebtAnalysisRequest,
    background_tasks: BackgroundTasks,
    project: ProjectIndex = Depends(get_project_or_404),
    debt_analyzer: TechnicalDebtAnalyzer = Depends(get_technical_debt_analyzer),
    advanced_detector: AdvancedDebtDetector = Depends(get_advanced_debt_detector),
    session: AsyncSession = Depends(get_session),
    current_user: str = Depends(get_current_user)
):
    """
    Analyze technical debt for a specific project.
    
    - **project_id**: UUID of the project to analyze
    - **include_advanced_patterns**: Whether to include ML-powered pattern detection
    - **include_historical_analysis**: Whether to include historical trend analysis
    - **analysis_depth**: Level of analysis (quick, standard, comprehensive)
    """
    
    analysis_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    logger.info(
        "Starting technical debt analysis",
        project_id=str(project_id),
        analysis_id=analysis_id,
        analysis_depth=request.analysis_depth,
        user=current_user
    )
    
    try:
        # Perform basic debt analysis
        debt_analysis = await debt_analyzer.analyze_project_debt(
            project,
            session,
            include_advanced=request.include_advanced_patterns
        )
        
        # Add advanced pattern detection if requested
        advanced_patterns_count = 0
        if request.include_advanced_patterns:
            advanced_patterns = await advanced_detector.analyze_advanced_debt_patterns(
                project, debt_analysis
            )
            advanced_patterns_count = len(advanced_patterns)
            
            # Merge advanced patterns into debt analysis
            debt_analysis.recommendations.extend([
                f"Advanced pattern detected: {pattern.pattern_name} (confidence: {pattern.confidence:.2f})"
                for pattern in advanced_patterns[:5]  # Limit to top 5
            ])
        
        # Convert debt items to response format
        debt_items_response = [
            DebtItemResponse(
                id=item.id or f"debt_{i}",
                project_id=item.project_id or str(project_id),
                file_id=item.file_id or "unknown",
                debt_type=item.debt_type or "general",
                category=item.category.value,
                severity=item.severity.value,
                status=item.status.value,
                description=item.description,
                location=item.location,
                debt_score=item.debt_score,
                confidence_score=item.confidence_score,
                remediation_suggestion=item.remediation_suggestion,
                estimated_effort_hours=item.estimated_effort_hours,
                first_detected_at=item.first_detected_at,
                last_detected_at=item.last_detected_at,
                metadata=item.metadata
            )
            for i, item in enumerate(debt_analysis.debt_items)
        ]
        
        # Calculate category breakdown
        from collections import defaultdict
        category_breakdown = defaultdict(float)
        severity_breakdown = defaultdict(int)
        
        for item in debt_analysis.debt_items:
            category_breakdown[item.category.value] += item.debt_score
            severity_breakdown[item.severity.value] += 1
        
        analysis_duration = (datetime.utcnow() - start_time).total_seconds()
        
        response_data = DebtAnalysisResponse(
            project_id=str(project_id),
            analysis_id=analysis_id,
            total_debt_score=debt_analysis.total_debt_score,
            debt_items=debt_items_response,
            category_breakdown=dict(category_breakdown),
            severity_breakdown=dict(severity_breakdown),
            file_count=debt_analysis.file_count,
            lines_of_code=debt_analysis.lines_of_code,
            analysis_duration_seconds=analysis_duration,
            recommendations=debt_analysis.recommendations,
            analysis_timestamp=start_time,
            advanced_patterns_detected=advanced_patterns_count if request.include_advanced_patterns else None
        )
        
        logger.info(
            "Technical debt analysis completed",
            project_id=str(project_id),
            analysis_id=analysis_id,
            total_debt_score=debt_analysis.total_debt_score,
            debt_items_count=len(debt_analysis.debt_items),
            duration_seconds=analysis_duration
        )
        
        return StandardResponse(
            success=True,
            message=f"Technical debt analysis completed successfully",
            data=response_data.dict()
        )
        
    except Exception as e:
        logger.error(
            "Technical debt analysis failed",
            project_id=str(project_id),
            analysis_id=analysis_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Technical debt analysis failed: {str(e)}"
        )


@router.get(
    "/{project_id}/history",
    response_model=StandardResponse,
    summary="Get Technical Debt History",
    description="Retrieve historical technical debt evolution analysis"
)
async def get_debt_history(
    project_id: uuid.UUID,
    lookback_days: int = Query(90, ge=1, le=365, description="Days to look back for historical analysis"),
    sample_frequency_days: int = Query(7, ge=1, le=30, description="Sampling frequency in days"),
    project: ProjectIndex = Depends(get_project_or_404),
    historical_analyzer: HistoricalAnalyzer = Depends(get_historical_analyzer),
    current_user: str = Depends(get_current_user)
):
    """
    Get historical technical debt evolution for a project.
    
    - **project_id**: UUID of the project
    - **lookback_days**: Number of days to analyze (1-365)
    - **sample_frequency_days**: How often to sample debt data (1-30 days)
    """
    
    logger.info(
        "Retrieving debt history",
        project_id=str(project_id),
        lookback_days=lookback_days,
        sample_frequency_days=sample_frequency_days,
        user=current_user
    )
    
    try:
        # Perform historical analysis
        evolution_result = await historical_analyzer.analyze_debt_evolution(
            project_id=str(project_id),
            project_path=project.root_path,
            lookback_days=lookback_days,
            sample_frequency_days=sample_frequency_days
        )
        
        # Convert to response format
        response_data = HistoricalAnalysisResponse(
            project_id=str(project_id),
            lookback_days=lookback_days,
            evolution_timeline=[
                {
                    "date": point.date.isoformat(),
                    "total_debt_score": point.total_debt_score,
                    "category_scores": point.category_scores,
                    "files_analyzed": point.files_analyzed,
                    "lines_of_code": point.lines_of_code,
                    "debt_items_count": point.debt_items_count,
                    "debt_delta": point.debt_delta,
                    "commit_hash": point.commit_hash,
                    "commit_message": point.commit_message,
                    "author": point.author
                }
                for point in evolution_result.evolution_timeline
            ],
            trend_analysis={
                "trend_direction": evolution_result.trend_analysis.trend_direction,
                "trend_strength": evolution_result.trend_analysis.trend_strength,
                "velocity": evolution_result.trend_analysis.velocity,
                "acceleration": evolution_result.trend_analysis.acceleration,
                "projected_debt_30_days": evolution_result.trend_analysis.projected_debt_30_days,
                "projected_debt_90_days": evolution_result.trend_analysis.projected_debt_90_days,
                "confidence_level": evolution_result.trend_analysis.confidence_level,
                "seasonal_patterns": evolution_result.trend_analysis.seasonal_patterns,
                "risk_level": evolution_result.trend_analysis.risk_level
            },
            debt_hotspots=[
                {
                    "file_path": hotspot.file_path,
                    "debt_score": hotspot.debt_score,
                    "debt_velocity": hotspot.debt_velocity,
                    "stability_risk": hotspot.stability_risk,
                    "contributor_count": hotspot.contributor_count,
                    "priority": hotspot.priority,
                    "categories_affected": hotspot.categories_affected,
                    "recommendations": hotspot.recommendations
                }
                for hotspot in evolution_result.debt_hotspots
            ],
            category_trends={
                category: {
                    "trend_direction": trend.trend_direction,
                    "trend_strength": trend.trend_strength,
                    "velocity": trend.velocity,
                    "risk_level": trend.risk_level
                }
                for category, trend in evolution_result.category_trends.items()
            },
            recommendations=evolution_result.recommendations,
            analysis_timestamp=datetime.utcnow()
        )
        
        logger.info(
            "Debt history analysis completed",
            project_id=str(project_id),
            timeline_points=len(evolution_result.evolution_timeline),
            hotspots_found=len(evolution_result.debt_hotspots)
        )
        
        return StandardResponse(
            success=True,
            message="Historical debt analysis completed successfully",
            data=response_data.dict()
        )
        
    except Exception as e:
        logger.error(
            "Debt history analysis failed",
            project_id=str(project_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Historical debt analysis failed: {str(e)}"
        )


@router.post(
    "/{project_id}/remediation-plan",
    response_model=StandardResponse,
    summary="Generate Remediation Plan",
    description="Generate intelligent remediation plan for technical debt"
)
async def generate_remediation_plan(
    project_id: uuid.UUID,
    scope: str = Query("project", pattern="^(project|file|directory)$", description="Scope of remediation"),
    target_path: Optional[str] = Query(None, description="Specific path to focus on (for file/directory scope)"),
    project: ProjectIndex = Depends(get_project_or_404),
    remediation_engine: DebtRemediationEngine = Depends(get_debt_remediation_engine),
    current_user: str = Depends(get_current_user)
):
    """
    Generate an intelligent remediation plan for technical debt.
    
    - **project_id**: UUID of the project
    - **scope**: Scope of remediation (project, file, directory)
    - **target_path**: Specific path for file or directory scope
    """
    
    plan_id = str(uuid.uuid4())
    
    logger.info(
        "Generating remediation plan",
        project_id=str(project_id),
        plan_id=plan_id,
        scope=scope,
        target_path=target_path,
        user=current_user
    )
    
    try:
        # Generate remediation plan
        remediation_plan = await remediation_engine.generate_remediation_plan(
            project=project,
            scope=scope,
            target_path=target_path,
            context={"user": current_user, "plan_id": plan_id}
        )
        
        # Convert to response format
        response_data = RemediationPlanResponse(
            project_id=str(project_id),
            plan_id=plan_id,
            scope=remediation_plan.scope,
            target_path=remediation_plan.target_path,
            recommendations_count=len(remediation_plan.recommendations),
            execution_phases=remediation_plan.execution_phases,
            total_debt_reduction=remediation_plan.total_debt_reduction,
            total_effort_estimate=remediation_plan.total_effort_estimate,
            total_risk_score=remediation_plan.total_risk_score,
            estimated_duration_days=remediation_plan.estimated_duration_days,
            immediate_actions=remediation_plan.immediate_actions,
            quick_wins=remediation_plan.quick_wins,
            long_term_goals=remediation_plan.long_term_goals,
            success_criteria=remediation_plan.success_criteria,
            potential_blockers=remediation_plan.potential_blockers,
            created_at=remediation_plan.created_at
        )
        
        logger.info(
            "Remediation plan generated successfully",
            project_id=str(project_id),
            plan_id=plan_id,
            recommendations_count=len(remediation_plan.recommendations),
            estimated_duration_days=remediation_plan.estimated_duration_days
        )
        
        return StandardResponse(
            success=True,
            message="Remediation plan generated successfully",
            data=response_data.dict()
        )
        
    except Exception as e:
        logger.error(
            "Remediation plan generation failed",
            project_id=str(project_id),
            plan_id=plan_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Remediation plan generation failed: {str(e)}"
        )


@router.get(
    "/{project_id}/recommendations/{file_path:path}",
    response_model=StandardResponse,
    summary="Get File Recommendations",
    description="Get specific remediation recommendations for a file"
)
async def get_file_recommendations(
    project_id: uuid.UUID,
    file_path: str,
    project: ProjectIndex = Depends(get_project_or_404),
    remediation_engine: DebtRemediationEngine = Depends(get_debt_remediation_engine),
    current_user: str = Depends(get_current_user)
):
    """
    Get specific remediation recommendations for a file.
    
    - **project_id**: UUID of the project
    - **file_path**: Path to the specific file
    """
    
    # Decode URL-encoded file path
    decoded_file_path = unquote(file_path)
    
    logger.info(
        "Getting file recommendations",
        project_id=str(project_id),
        file_path=decoded_file_path,
        user=current_user
    )
    
    try:
        # Get file-specific recommendations
        recommendations = await remediation_engine.get_file_specific_recommendations(
            project=project,
            file_path=decoded_file_path,
            context={"user": current_user}
        )
        
        # Convert to response format
        recommendations_response = [
            RemediationRecommendationResponse(
                id=rec.id,
                strategy=rec.strategy.value,
                priority=rec.priority.value,
                impact=rec.impact.value,
                title=rec.title,
                description=rec.description,
                rationale=rec.rationale,
                file_path=rec.file_path,
                line_ranges=[[start, end] for start, end in rec.line_ranges],
                debt_reduction_score=rec.debt_reduction_score,
                implementation_effort=rec.implementation_effort,
                risk_level=rec.risk_level,
                cost_benefit_ratio=rec.cost_benefit_ratio,
                suggested_approach=rec.suggested_approach,
                code_examples=rec.code_examples,
                related_patterns=rec.related_patterns,
                dependencies=rec.dependencies,
                debt_categories=[cat.value for cat in rec.debt_categories]
            )
            for rec in recommendations
        ]
        
        logger.info(
            "File recommendations retrieved successfully",
            project_id=str(project_id),
            file_path=decoded_file_path,
            recommendations_count=len(recommendations)
        )
        
        return StandardResponse(
            success=True,
            message=f"Retrieved {len(recommendations)} recommendations for {decoded_file_path}",
            data={
                "file_path": decoded_file_path,
                "recommendations": [rec.dict() for rec in recommendations_response]
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "File recommendations retrieval failed",
            project_id=str(project_id),
            file_path=decoded_file_path,
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get file recommendations: {str(e)}"
        )


@router.get(
    "/{project_id}/monitoring/status",
    response_model=StandardResponse,
    summary="Get Monitoring Status",
    description="Get real-time debt monitoring status"
)
async def get_monitoring_status(
    project_id: uuid.UUID,
    project: ProjectIndex = Depends(get_project_or_404),
    monitor_integration: DebtMonitorIntegration = Depends(get_debt_monitor_integration),
    current_user: str = Depends(get_current_user)
):
    """
    Get real-time technical debt monitoring status.
    
    - **project_id**: UUID of the project
    """
    
    logger.info(
        "Getting monitoring status",
        project_id=str(project_id),
        user=current_user
    )
    
    try:
        # Get monitoring status
        status = await monitor_integration.get_monitoring_status()
        
        # Convert to response format
        response_data = MonitoringStatusResponse(
            enabled=status['enabled'],
            active_since=datetime.fromisoformat(status['active_since']) if status.get('active_since') else None,
            monitored_projects_count=status['monitored_projects_count'],
            total_files_monitored=status['total_files_monitored'],
            total_debt_events=status['total_debt_events'],
            configuration=status['configuration'],
            projects=status['projects']
        )
        
        return StandardResponse(
            success=True,
            message="Monitoring status retrieved successfully",
            data=response_data.dict()
        )
        
    except Exception as e:
        logger.error(
            "Failed to get monitoring status",
            project_id=str(project_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get monitoring status: {str(e)}"
        )


@router.post(
    "/{project_id}/monitoring/start",
    response_model=StandardResponse,
    summary="Start Debt Monitoring",
    description="Start real-time technical debt monitoring for project"
)
async def start_debt_monitoring(
    project_id: uuid.UUID,
    project: ProjectIndex = Depends(get_project_or_404),
    monitor_integration: DebtMonitorIntegration = Depends(get_debt_monitor_integration),
    current_user: str = Depends(get_current_user)
):
    """
    Start real-time technical debt monitoring for a project.
    
    - **project_id**: UUID of the project
    """
    
    logger.info(
        "Starting debt monitoring",
        project_id=str(project_id),
        user=current_user
    )
    
    try:
        # Initialize monitoring components if needed
        await monitor_integration.initialize_components()
        
        # Start monitoring for the project
        await monitor_integration.start_monitoring_project(project)
        
        logger.info(
            "Debt monitoring started successfully",
            project_id=str(project_id),
            files_monitored=len(project.file_entries)
        )
        
        return StandardResponse(
            success=True,
            message=f"Debt monitoring started for project {project.name}",
            data={
                "project_id": str(project_id),
                "files_monitored": len(project.file_entries),
                "monitoring_active": True
            }
        )
        
    except Exception as e:
        logger.error(
            "Failed to start debt monitoring",
            project_id=str(project_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start debt monitoring: {str(e)}"
        )


@router.post(
    "/{project_id}/monitoring/stop",
    response_model=StandardResponse,
    summary="Stop Debt Monitoring",
    description="Stop real-time technical debt monitoring for project"
)
async def stop_debt_monitoring(
    project_id: uuid.UUID,
    project: ProjectIndex = Depends(get_project_or_404),
    monitor_integration: DebtMonitorIntegration = Depends(get_debt_monitor_integration),
    current_user: str = Depends(get_current_user)
):
    """
    Stop real-time technical debt monitoring for a project.
    
    - **project_id**: UUID of the project
    """
    
    logger.info(
        "Stopping debt monitoring",
        project_id=str(project_id),
        user=current_user
    )
    
    try:
        # Stop monitoring for the project
        await monitor_integration.stop_monitoring_project(str(project_id))
        
        logger.info(
            "Debt monitoring stopped successfully",
            project_id=str(project_id)
        )
        
        return StandardResponse(
            success=True,
            message=f"Debt monitoring stopped for project {project.name}",
            data={
                "project_id": str(project_id),
                "monitoring_active": False
            }
        )
        
    except Exception as e:
        logger.error(
            "Failed to stop debt monitoring",
            project_id=str(project_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stop debt monitoring: {str(e)}"
        )


@router.post(
    "/{project_id}/analyze/force",
    response_model=StandardResponse,
    summary="Force Debt Analysis",
    description="Force immediate technical debt analysis bypassing cache"
)
async def force_debt_analysis(
    project_id: uuid.UUID,
    file_paths: Optional[List[str]] = Query(None, description="Specific file paths to analyze"),
    project: ProjectIndex = Depends(get_project_or_404),
    monitor_integration: DebtMonitorIntegration = Depends(get_debt_monitor_integration),
    current_user: str = Depends(get_current_user)
):
    """
    Force immediate technical debt analysis, bypassing cache.
    
    - **project_id**: UUID of the project
    - **file_paths**: Optional list of specific file paths to analyze
    """
    
    logger.info(
        "Forcing debt analysis",
        project_id=str(project_id),
        file_paths=file_paths,
        user=current_user
    )
    
    try:
        # Force analysis through monitoring integration
        result = await monitor_integration.force_debt_analysis(
            project_id=str(project_id),
            file_paths=file_paths
        )
        
        logger.info(
            "Forced debt analysis completed",
            project_id=str(project_id),
            files_analyzed=result['files_analyzed'],
            total_debt_score=result['total_debt_score']
        )
        
        return StandardResponse(
            success=True,
            message="Forced debt analysis completed successfully",
            data=result
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(
            "Forced debt analysis failed",
            project_id=str(project_id),
            error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Forced debt analysis failed: {str(e)}"
        )


# Health check endpoint
@router.get(
    "/health",
    response_model=StandardResponse,
    summary="Health Check",
    description="Health check for technical debt API"
)
async def health_check():
    """Technical debt API health check."""
    return StandardResponse(
        success=True,
        message="Technical Debt API is healthy",
        data={
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat()
        }
    )