"""
Projects API - Consolidated project management endpoints

Consolidates project_index.py, project_index_optimization.py,
project_index_websocket.py, and project_index_websocket_monitoring.py
into a unified RESTful resource for project indexing and analysis.

Performance target: <200ms P95 response time
"""

import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

import structlog
from fastapi import APIRouter, Request, HTTPException, Query, BackgroundTasks
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_
from sqlalchemy.orm import selectinload

from ...core.database import get_session_dependency
from ...project_index.core import ProjectIndexer
from ...project_index.analyzer import CodeAnalyzer
from ...models.project_index import ProjectIndex, ProjectStatus
from ...schemas.project_index import (
    ProjectIndexCreate,
    ProjectIndexUpdate,
    ProjectIndexResponse,
    ProjectIndexListResponse,
    ProjectStatistics,
    AnalysisSessionCreate
)
from ..middleware import (
    get_current_user_from_request
)

logger = structlog.get_logger()
router = APIRouter()

# Project index dependencies
async def get_project_index_core() -> ProjectIndexer:
    """Get project index core instance."""
    return ProjectIndexer()

async def get_project_analyzer() -> CodeAnalyzer:
    """Get project analyzer instance."""
    return CodeAnalyzer()

@router.post("/", response_model=ProjectIndexResponse, status_code=201)
async def create_project(
    request: Request,
    project_data: ProjectIndexCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session_dependency),
    project_core: ProjectIndexer = Depends(get_project_index_core)
) -> ProjectIndexResponse:
    """
    Create and index a new project.
    
    Performance target: <200ms (for creation, indexing happens async)
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Create project record
        project = ProjectIndex(
            id=str(uuid.uuid4()),
            name=project_data.name,
            description=project_data.description,
            repository_url=project_data.repository_url,
            local_path=project_data.local_path,
            status=ProjectStatus.PENDING,
            configuration=project_data.configuration or {},
            metadata={
                "created_by": current_user.id,
                "created_at": datetime.utcnow().isoformat(),
                "version": "2.0"
            }
        )
        
        db.add(project)
        await db.commit()
        await db.refresh(project)
        
        # Start indexing in background
        background_tasks.add_task(
            _index_project_background,
            project_core,
            project.id,
            project_data.repository_url,
            project_data.local_path,
            current_user.id
        )
        
        logger.info(
            "project_created",
            project_id=project.id,
            project_name=project.name,
            repository_url=project.repository_url,
            created_by=current_user.id
        )
        
        return ProjectIndexResponse.from_orm(project)
        
    except Exception as e:
        await db.rollback()
        logger.error("project_creation_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create project: {str(e)}"
        )

async def _index_project_background(
    project_core: ProjectIndexer,
    project_id: str,
    repository_url: Optional[str],
    local_path: Optional[str],
    user_id: str
):
    """Background task for project indexing."""
    try:
        # Update status to indexing
        async with get_session_dependency() as db:
            await db.execute(
                update(ProjectIndex)
                .where(ProjectIndex.id == project_id)
                .values(
                    status=ProjectStatus.INDEXING,
                    updated_at=datetime.utcnow()
                )
            )
            await db.commit()
        
        # Perform indexing
        result = await project_core.index_project(
            project_id=project_id,
            repository_url=repository_url,
            local_path=local_path
        )
        
        # Update with results
        async with get_session_dependency() as db:
            await db.execute(
                update(ProjectIndex)
                .where(ProjectIndex.id == project_id)
                .values(
                    status=ProjectStatus.COMPLETED,
                    index_data=result.index_data,
                    file_count=result.file_count,
                    language_breakdown=result.language_breakdown,
                    updated_at=datetime.utcnow()
                )
            )
            await db.commit()
        
        logger.info(
            "project_indexing_completed",
            project_id=project_id,
            file_count=result.file_count
        )
        
    except Exception as e:
        # Update status to failed
        async with get_session_dependency() as db:
            await db.execute(
                update(ProjectIndex)
                .where(ProjectIndex.id == project_id)
                .values(
                    status=ProjectStatus.FAILED,
                    error_message=str(e),
                    updated_at=datetime.utcnow()
                )
            )
            await db.commit()
        
        logger.error(
            "project_indexing_failed",
            project_id=project_id,
            error=str(e)
        )

@router.get("/", response_model=ProjectIndexListResponse)
async def list_projects(
    request: Request,
    skip: int = Query(0, ge=0, description="Number of projects to skip"),
    limit: int = Query(50, ge=1, le=1000, description="Number of projects to return"),
    status: Optional[ProjectStatus] = Query(None, description="Filter by index status"),
    search: Optional[str] = Query(None, description="Search by name or description"),
    db: AsyncSession = Depends(get_session_dependency)
) -> ProjectIndexListResponse:
    """
    List all projects with optional filtering.
    
    Performance target: <200ms
    """
    try:
        # Build query with filters
        query = select(ProjectIndex)
        
        filters = []
        if status:
            filters.append(ProjectIndex.status == status)
        if search:
            search_filter = f"%{search}%"
            filters.append(
                or_(
                    ProjectIndex.name.ilike(search_filter),
                    ProjectIndex.description.ilike(search_filter)
                )
            )
            
        if filters:
            query = query.where(and_(*filters))
            
        # Apply pagination
        query = query.offset(skip).limit(limit)
        
        # Execute query
        result = await db.execute(query)
        projects = result.scalars().all()
        
        # Get total count for pagination
        count_query = select(ProjectIndex)
        if filters:
            count_query = count_query.where(and_(*filters))
            
        total_result = await db.execute(count_query)
        total = len(total_result.scalars().all())
        
        return ProjectIndexListResponse(
            projects=[ProjectIndexResponse.from_orm(project) for project in projects],
            total=total,
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        logger.error("project_list_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list projects: {str(e)}"
        )

@router.get("/{project_id}", response_model=ProjectIndexResponse)
async def get_project(
    project_id: str,
    db: AsyncSession = Depends(get_session_dependency)
) -> ProjectIndexResponse:
    """
    Get details of a specific project.
    
    Performance target: <200ms
    """
    try:
        # Query project
        query = select(ProjectIndex).where(ProjectIndex.id == project_id)
        result = await db.execute(query)
        project = result.scalar_one_or_none()
        
        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"Project {project_id} not found"
            )
            
        return ProjectIndexResponse.from_orm(project)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("project_get_failed", project_id=project_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get project: {str(e)}"
        )

@router.put("/{project_id}", response_model=ProjectIndexResponse)
async def update_project(
    request: Request,
    project_id: str,
    project_data: ProjectIndexUpdate,
    db: AsyncSession = Depends(get_session_dependency)
) -> ProjectIndexResponse:
    """
    Update an existing project.
    
    Performance target: <200ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Get existing project
        query = select(ProjectIndex).where(ProjectIndex.id == project_id)
        result = await db.execute(query)
        project = result.scalar_one_or_none()
        
        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"Project {project_id} not found"
            )
        
        # Update project fields
        update_data = project_data.dict(exclude_unset=True)
        
        if update_data:
            update_data["updated_at"] = datetime.utcnow()
            update_data["updated_by"] = current_user.id
            
            # Update in database
            await db.execute(
                update(ProjectIndex)
                .where(ProjectIndex.id == project_id)
                .values(**update_data)
            )
            await db.commit()
        
        # Get updated project
        result = await db.execute(query)
        updated_project = result.scalar_one()
        
        logger.info(
            "project_updated",
            project_id=project_id,
            updated_by=current_user.id,
            updated_fields=list(update_data.keys())
        )
        
        return ProjectIndexResponse.from_orm(updated_project)
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("project_update_failed", project_id=project_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update project: {str(e)}"
        )

@router.delete("/{project_id}", status_code=204)
async def delete_project(
    request: Request,
    project_id: str,
    db: AsyncSession = Depends(get_session_dependency)
):
    """
    Delete a project from the system.
    
    Performance target: <200ms
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Check if project exists
        query = select(ProjectIndex).where(ProjectIndex.id == project_id)
        result = await db.execute(query)
        project = result.scalar_one_or_none()
        
        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"Project {project_id} not found"
            )
        
        # Delete from database
        await db.execute(delete(ProjectIndex).where(ProjectIndex.id == project_id))
        await db.commit()
        
        logger.info(
            "project_deleted",
            project_id=project_id,
            deleted_by=current_user.id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("project_delete_failed", project_id=project_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete project: {str(e)}"
        )

@router.post("/{project_id}/analyze")
async def analyze_project(
    request: Request,
    project_id: str,
    analysis_data: AnalysisSessionCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session_dependency),
    analyzer: CodeAnalyzer = Depends(get_project_analyzer)
):
    """
    Perform detailed analysis on a project.
    
    Performance target: <200ms (for initiation, analysis happens async)
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Get project
        query = select(ProjectIndex).where(ProjectIndex.id == project_id)
        result = await db.execute(query)
        project = result.scalar_one_or_none()
        
        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"Project {project_id} not found"
            )
        
        if project.status != ProjectStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Project must be indexed before analysis (current status: {project.status.value})"
            )
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Start analysis in background
        background_tasks.add_task(
            _analyze_project_background,
            analyzer,
            project_id,
            analysis_id,
            analysis_data.analysis_type,
            analysis_data.parameters,
            current_user.id
        )
        
        logger.info(
            "project_analysis_started",
            project_id=project_id,
            analysis_id=analysis_id,
            analysis_type=analysis_data.analysis_type,
            started_by=current_user.id
        )
        
        return {
            "analysis_id": analysis_id,
            "project_id": project_id,
            "analysis_type": analysis_data.analysis_type,
            "status": "started",
            "message": "Project analysis initiated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("project_analysis_failed", project_id=project_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start project analysis: {str(e)}"
        )

async def _analyze_project_background(
    analyzer: CodeAnalyzer,
    project_id: str,
    analysis_id: str,
    analysis_type: str,
    parameters: Dict[str, Any],
    user_id: str
):
    """Background task for project analysis."""
    try:
        # Perform analysis
        result = await analyzer.analyze_project(
            project_id=project_id,
            analysis_type=analysis_type,
            parameters=parameters
        )
        
        # Store analysis results (would be in a separate analysis_results table)
        logger.info(
            "project_analysis_completed",
            project_id=project_id,
            analysis_id=analysis_id,
            result=result
        )
        
    except Exception as e:
        logger.error(
            "project_analysis_error",
            project_id=project_id,
            analysis_id=analysis_id,
            error=str(e)
        )

@router.post("/{project_id}/reindex")
async def reindex_project(
    request: Request,
    project_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_session_dependency),
    project_core: ProjectIndexer = Depends(get_project_index_core)
):
    """
    Reindex an existing project.
    
    Performance target: <200ms (for initiation, reindexing happens async)
    """
    current_user = get_current_user_from_request(request)
    
    try:
        # Get project
        query = select(ProjectIndex).where(ProjectIndex.id == project_id)
        result = await db.execute(query)
        project = result.scalar_one_or_none()
        
        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"Project {project_id} not found"
            )
        
        # Update status to indexing
        await db.execute(
            update(ProjectIndex)
            .where(ProjectIndex.id == project_id)
            .values(
                status=ProjectStatus.INDEXING,
                updated_at=datetime.utcnow(),
                updated_by=current_user.id
            )
        )
        await db.commit()
        
        # Start reindexing in background
        background_tasks.add_task(
            _index_project_background,
            project_core,
            project.id,
            project.repository_url,
            project.local_path,
            current_user.id
        )
        
        logger.info(
            "project_reindexing_started",
            project_id=project_id,
            started_by=current_user.id
        )
        
        return {
            "project_id": project_id,
            "status": "reindexing",
            "message": "Project reindexing initiated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        logger.error("project_reindex_failed", project_id=project_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reindex project: {str(e)}"
        )

@router.get("/{project_id}/files")
async def list_project_files(
    project_id: str,
    path: Optional[str] = Query(None, description="Filter by file path"),
    extension: Optional[str] = Query(None, description="Filter by file extension"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_session_dependency)
):
    """
    List files in a project.
    
    Performance target: <200ms
    """
    try:
        # Verify project exists
        project_query = select(ProjectIndex).where(ProjectIndex.id == project_id)
        project_result = await db.execute(project_query)
        project = project_result.scalar_one_or_none()
        
        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"Project {project_id} not found"
            )
        
        if project.status != ProjectStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail="Project must be indexed to list files"
            )
        
        # For now, return placeholder file list
        # In production, this would query the actual indexed files
        files = []
        if project.index_data and isinstance(project.index_data, dict):
            files = project.index_data.get("files", [])
        
        # Apply filters
        if path:
            files = [f for f in files if path in f.get("path", "")]
        if extension:
            files = [f for f in files if f.get("path", "").endswith(f".{extension}")]
        
        # Apply pagination
        total = len(files)
        files = files[skip:skip + limit]
        
        return {
            "files": files,
            "total": total,
            "skip": skip,
            "limit": limit,
            "project_id": project_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("project_files_list_failed", project_id=project_id, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list project files: {str(e)}"
        )

@router.get("/stats/overview", response_model=ProjectStatistics)
async def get_project_stats(
    db: AsyncSession = Depends(get_session_dependency)
) -> ProjectStatistics:
    """
    Get system-wide project statistics.
    
    Performance target: <200ms
    """
    try:
        # Get all projects
        query = select(ProjectIndex)
        result = await db.execute(query)
        projects = result.scalars().all()
        
        # Calculate statistics
        total_projects = len(projects)
        
        status_counts = {}
        for status in ProjectStatus:
            status_counts[status.value] = len([p for p in projects if p.status == status])
        
        # Calculate language breakdown across all projects
        language_breakdown = {}
        total_files = 0
        
        for project in projects:
            if project.language_breakdown:
                for lang, count in project.language_breakdown.items():
                    language_breakdown[lang] = language_breakdown.get(lang, 0) + count
            if project.file_count:
                total_files += project.file_count
        
        return ProjectStatistics(
            total_projects=total_projects,
            status_breakdown=status_counts,
            total_indexed_files=total_files,
            language_breakdown=language_breakdown,
            average_files_per_project=round(total_files / total_projects, 2) if total_projects > 0 else 0
        )
        
    except Exception as e:
        logger.error("project_stats_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get project stats: {str(e)}"
        )