"""
Self-Modification Engine API Endpoints

FastAPI routes for the self-modification engine providing comprehensive
code analysis, modification generation, application, and monitoring capabilities.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_session
from app.core.self_modification import SelfModificationService
from app.schemas.self_modification import (
    AnalyzeCodebaseRequest, AnalyzeCodebaseResponse,
    ApplyModificationsRequest, ApplyModificationsResponse,
    RollbackModificationRequest, RollbackModificationResponse,
    ProvideFeedbackRequest, ModificationSessionResponse,
    GetSessionsResponse, ModificationMetricsResponse,
    SystemHealthResponse, ErrorResponse, PaginationParams, FilterParams
)

logger = structlog.get_logger()

router = APIRouter(prefix="/self-modify", tags=["self-modification"])


# Dependency to get self-modification service
async def get_self_modification_service(
    session: AsyncSession = Depends(get_session)
) -> SelfModificationService:
    """Get self-modification service instance."""
    return SelfModificationService(session)


@router.post(
    "/analyze",
    response_model=AnalyzeCodebaseResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze codebase for modification opportunities",
    description="""
    Analyze a codebase to identify potential improvements and generate modification suggestions.
    This endpoint performs comprehensive code analysis including:
    
    - AST parsing and code structure analysis
    - Performance bottleneck detection  
    - Security vulnerability scanning
    - Code quality assessment
    - Anti-pattern identification
    
    The analysis respects the specified safety level and modification goals.
    """,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request parameters"},
        500: {"model": ErrorResponse, "description": "Analysis failed"}
    }
)
async def analyze_codebase(
    request: AnalyzeCodebaseRequest,
    background_tasks: BackgroundTasks,
    service: SelfModificationService = Depends(get_self_modification_service)
) -> AnalyzeCodebaseResponse:
    """Analyze codebase and generate modification suggestions."""
    
    logger.info(
        "Starting codebase analysis",
        codebase_path=request.codebase_path,
        goals=request.modification_goals,
        safety_level=request.safety_level.value
    )
    
    try:
        # Start analysis
        analysis_result = await service.analyze_codebase(
            codebase_path=request.codebase_path,
            modification_goals=request.modification_goals,
            safety_level=request.safety_level.value,
            repository_id=request.repository_id,
            analysis_context=request.analysis_context or {},
            include_patterns=request.include_patterns,
            exclude_patterns=request.exclude_patterns
        )
        
        # Schedule background processing if needed
        if len(analysis_result.suggestions) > 10:
            background_tasks.add_task(
                service.process_large_analysis,
                analysis_result.analysis_id
            )
        
        logger.info(
            "Codebase analysis completed",
            analysis_id=analysis_result.analysis_id,
            suggestions_count=len(analysis_result.suggestions)
        )
        
        return analysis_result
        
    except ValueError as e:
        logger.warning("Invalid analysis request", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        logger.error("Analysis failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Analysis failed due to internal error"
        )


@router.post(
    "/apply",
    response_model=ApplyModificationsResponse,
    status_code=status.HTTP_200_OK,
    summary="Apply selected modifications",
    description="""
    Apply selected modifications from an analysis session to the codebase.
    This endpoint:
    
    - Validates all selected modifications
    - Runs security and safety checks
    - Tests modifications in sandbox environment
    - Creates git branch and commits changes
    - Provides rollback capabilities
    
    High-risk modifications may require human approval via approval_token.
    """,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid modifications or approval required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions or approval token invalid"},
        500: {"model": ErrorResponse, "description": "Application failed"}
    }
)
async def apply_modifications(
    request: ApplyModificationsRequest,
    background_tasks: BackgroundTasks,
    service: SelfModificationService = Depends(get_self_modification_service)
) -> ApplyModificationsResponse:
    """Apply selected modifications to the codebase."""
    
    logger.info(
        "Applying modifications",
        analysis_id=request.analysis_id,
        modification_count=len(request.selected_modifications),
        dry_run=request.dry_run
    )
    
    try:
        # Apply modifications
        result = await service.apply_modifications(
            analysis_id=request.analysis_id,
            selected_modifications=request.selected_modifications,
            approval_token=request.approval_token,
            git_branch=request.git_branch,
            commit_message=request.commit_message,
            dry_run=request.dry_run
        )
        
        # Schedule background validation if not dry run
        if not request.dry_run and result.status == "applied":
            background_tasks.add_task(
                service.validate_applied_modifications,
                result.session_id,
                result.applied_modifications
            )
        
        logger.info(
            "Modifications applied",
            session_id=result.session_id,
            applied_count=len(result.applied_modifications),
            failed_count=len(result.failed_modifications)
        )
        
        return result
        
    except PermissionError as e:
        logger.warning("Insufficient permissions", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ValueError as e:
        logger.warning("Invalid modification request", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to apply modifications", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to apply modifications due to internal error"
        )


@router.post(
    "/rollback",
    response_model=RollbackModificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Rollback applied modifications",
    description="""
    Rollback previously applied modifications to restore the codebase to a previous state.
    This endpoint:
    
    - Validates rollback safety
    - Creates rollback branch for safety
    - Restores files to previous state
    - Updates modification status
    - Preserves rollback history
    
    Use force_rollback=true to override safety checks if necessary.
    """,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid rollback request"},
        404: {"model": ErrorResponse, "description": "Modification not found"},
        409: {"model": ErrorResponse, "description": "Rollback conflicts detected"},
        500: {"model": ErrorResponse, "description": "Rollback failed"}
    }
)
async def rollback_modification(
    request: RollbackModificationRequest,
    service: SelfModificationService = Depends(get_self_modification_service)
) -> RollbackModificationResponse:
    """Rollback applied modifications."""
    
    logger.info(
        "Rolling back modification",
        modification_id=request.modification_id,
        reason=request.rollback_reason,
        force=request.force_rollback
    )
    
    try:
        result = await service.rollback_modification(
            modification_id=request.modification_id,
            rollback_reason=request.rollback_reason,
            force_rollback=request.force_rollback
        )
        
        logger.info(
            "Modification rolled back",
            modification_id=request.modification_id,
            success=result.success
        )
        
        return result
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Modification not found"
        )
    except ValueError as e:
        logger.warning("Invalid rollback request", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except RuntimeError as e:
        logger.error("Rollback conflict", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Rollback failed due to conflicts: {str(e)}"
        )
    except Exception as e:
        logger.error("Rollback failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Rollback failed due to internal error"
        )


@router.get(
    "/sessions",
    response_model=GetSessionsResponse,
    status_code=status.HTTP_200_OK,
    summary="List modification sessions",
    description="""
    Retrieve a paginated list of modification sessions with optional filtering.
    Supports filtering by status, safety level, agent, date range, and more.
    """,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query parameters"}
    }
)
async def get_sessions(
    pagination: PaginationParams = Depends(),
    filters: FilterParams = Depends(),
    service: SelfModificationService = Depends(get_self_modification_service)
) -> GetSessionsResponse:
    """Get paginated list of modification sessions."""
    
    try:
        result = await service.get_modification_sessions(
            page=pagination.page,
            page_size=pagination.page_size,
            sort_by=pagination.sort_by,
            sort_order=pagination.sort_order,
            filters=filters.dict(exclude_none=True)
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get(
    "/sessions/{session_id}",
    response_model=ModificationSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get modification session details",
    description="""
    Retrieve detailed information about a specific modification session,
    including all modifications, sandbox executions, and performance metrics.
    """,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"}
    }
)
async def get_session(
    session_id: UUID,
    service: SelfModificationService = Depends(get_self_modification_service)
) -> ModificationSessionResponse:
    """Get detailed session information."""
    
    try:
        session = await service.get_modification_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        return session
        
    except Exception as e:
        logger.error("Failed to get session", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session"
        )


@router.get(
    "/metrics",
    response_model=ModificationMetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get performance metrics",
    description="""
    Retrieve aggregated performance metrics for modifications.
    Can be filtered by session, modification, or time range.
    """,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query parameters"}
    }
)
async def get_metrics(
    session_id: Optional[UUID] = None,
    modification_id: Optional[UUID] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    service: SelfModificationService = Depends(get_self_modification_service)
) -> ModificationMetricsResponse:
    """Get aggregated performance metrics."""
    
    try:
        metrics = await service.get_performance_metrics(
            session_id=session_id,
            modification_id=modification_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return metrics
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )


@router.post(
    "/feedback",
    status_code=status.HTTP_201_CREATED,
    summary="Provide feedback on modifications",
    description="""
    Provide feedback on applied modifications to improve future suggestions.
    Feedback is used for learning and adaptation of the modification engine.
    """,
    responses={
        404: {"model": ErrorResponse, "description": "Modification not found"},
        400: {"model": ErrorResponse, "description": "Invalid feedback"}
    }
)
async def provide_feedback(
    request: ProvideFeedbackRequest,
    service: SelfModificationService = Depends(get_self_modification_service)
) -> Dict[str, str]:
    """Provide feedback on modifications."""
    
    logger.info(
        "Providing modification feedback",
        modification_id=request.modification_id,
        feedback_type=request.feedback_type,
        rating=request.rating
    )
    
    try:
        await service.provide_feedback(
            modification_id=request.modification_id,
            feedback_source=request.feedback_source,
            feedback_type=request.feedback_type,
            rating=request.rating,
            feedback_text=request.feedback_text,
            patterns_identified=request.patterns_identified,
            improvement_suggestions=request.improvement_suggestions
        )
        
        return {"message": "Feedback recorded successfully"}
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Modification not found"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to record feedback", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record feedback"
        )


@router.get(
    "/health",
    response_model=SystemHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Get system health status",
    description="""
    Check the health status of the self-modification engine components:
    - Sandbox environment availability
    - Git integration status
    - Modification queue size
    - Overall system performance metrics
    """,
    responses={
        503: {"model": ErrorResponse, "description": "System unhealthy"}
    }
)
async def get_system_health(
    service: SelfModificationService = Depends(get_self_modification_service)
) -> SystemHealthResponse:
    """Get system health status."""
    
    try:
        health = await service.get_system_health()
        
        # Return 503 if critical components are unhealthy
        if not health.sandbox_environment_healthy or not health.git_integration_healthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Critical system components are unhealthy"
            )
        
        return health
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to check system health", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check system health"
        )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete modification session",
    description="""
    Delete a modification session and all associated data.
    This action is irreversible and will clean up:
    - Session records
    - Modification history
    - Performance metrics
    - Sandbox execution logs
    
    Only completed or failed sessions can be deleted.
    """,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
        409: {"model": ErrorResponse, "description": "Session cannot be deleted (still active)"}
    }
)
async def delete_session(
    session_id: UUID,
    service: SelfModificationService = Depends(get_self_modification_service)
) -> None:
    """Delete a modification session."""
    
    logger.info("Deleting modification session", session_id=session_id)
    
    try:
        await service.delete_modification_session(session_id)
        
        logger.info("Session deleted successfully", session_id=session_id)
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to delete session", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session"
        )


@router.post(
    "/sessions/{session_id}/archive",
    status_code=status.HTTP_200_OK,
    summary="Archive modification session",
    description="""
    Archive a completed modification session for long-term storage.
    Archived sessions are moved to cold storage and can be retrieved if needed.
    """,
    responses={
        404: {"model": ErrorResponse, "description": "Session not found"},
        409: {"model": ErrorResponse, "description": "Session cannot be archived"}
    }
)
async def archive_session(
    session_id: UUID,
    service: SelfModificationService = Depends(get_self_modification_service)
) -> Dict[str, str]:
    """Archive a modification session."""
    
    try:
        await service.archive_modification_session(session_id)
        
        return {"message": "Session archived successfully"}
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to archive session", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to archive session"
        )


# Export router
__all__ = ["router"]