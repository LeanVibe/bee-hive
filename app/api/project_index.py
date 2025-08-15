"""
Project Index RESTful API Endpoints for LeanVibe Agent Hive 2.0

Comprehensive API endpoints providing full index management capabilities,
integrating with the core infrastructure for intelligent code analysis system.
"""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from urllib.parse import unquote

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks, Path as FastAPIPath, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, delete, and_, or_, func, desc
from pydantic import BaseModel, Field, ValidationError

from ..core.config import settings
from ..core.database import get_session
from ..core.redis import get_redis_client, RedisClient

# Mock auth functions for Project Index API
# These should be replaced with actual implementations when available
async def get_current_user():
    """Mock current user - replace with actual implementation."""
    return "test_user"

async def get_current_user_from_token(token: str):
    """Mock user from token - replace with actual implementation."""
    return "test_user"
from ..models.project_index import (
    ProjectIndex, FileEntry, DependencyRelationship, AnalysisSession, IndexSnapshot,
    ProjectStatus, FileType, DependencyType, AnalysisSessionType, AnalysisStatus
)
from ..schemas.project_index import (
    ProjectIndexCreate, ProjectIndexUpdate, ProjectIndexResponse, ProjectIndexListResponse,
    FileEntryResponse, FileEntryListResponse, DependencyRelationshipResponse, 
    DependencyRelationshipListResponse, AnalysisSessionCreate, AnalysisSessionResponse,
    AnalysisSessionListResponse, ProjectStatistics, DependencyGraph, 
    DependencyGraphNode, DependencyGraphEdge, AnalysisProgress,
    ProjectIndexFilter, FileEntryFilter, DependencyRelationshipFilter
)
from ..project_index.core import ProjectIndexer
from ..project_index.models import ProjectIndexConfig, AnalysisConfiguration
from ..models.project_index import AnalysisSessionType as CoreAnalysisSessionType
# Mock observability functions for Project Index API
# These should be replaced with actual implementations when available
async def get_event_publisher():
    """Mock event publisher - replace with actual implementation."""
    class MockEventPublisher:
        async def publish(self, event):
            logger.debug("Mock event published", event=event)
    return MockEventPublisher()

def create_analysis_event(event_type, **kwargs):
    """Mock analysis event creator - replace with actual implementation."""
    return {
        "type": event_type,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
    }

class EventType:
    """Mock event types - replace with actual implementation."""
    PROJECT_CREATED = "project_created"
    ERROR_OCCURRED = "error_occurred"
    ANALYSIS_PROGRESS = "analysis_progress"
    CACHE_HIT = "cache_hit"
    FILE_MODIFIED = "file_modified"
    FILE_CREATED = "file_created"
    FILE_DELETED = "file_deleted"
    FILE_MOVED = "file_moved"
from .project_index_websocket import (
    get_websocket_handler, websocket_auth_context, publish_analysis_progress,
    publish_file_change, publish_dependency_update, get_subscription_stats
)
from .project_index_optimization import (
    QueryOptimizer, CacheManager, PerformanceMonitor, CacheConfig,
    cache_response, create_optimized_session_factory, create_cache_manager_factory
)

logger = structlog.get_logger()

# Create API router with comprehensive OpenAPI documentation
router = APIRouter(
    prefix="/api/project-index", 
    tags=["Project Index"],
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


# ================== DEPENDENCY INJECTION ==================

async def get_project_indexer(
    session: AsyncSession = Depends(get_session),
    redis_client: RedisClient = Depends(get_redis_client)
) -> ProjectIndexer:
    """Get ProjectIndexer instance with dependencies."""
    return ProjectIndexer(
        session=session,
        redis_client=redis_client,
        config=ProjectIndexConfig(),
        event_publisher=await get_event_publisher()
    )


async def get_query_optimizer(
    session: AsyncSession = Depends(get_session)
) -> QueryOptimizer:
    """Get QueryOptimizer instance for database performance."""
    return QueryOptimizer(session)


async def get_cache_manager(
    redis_client: RedisClient = Depends(get_redis_client)
) -> CacheManager:
    """Get CacheManager instance for response caching."""
    return CacheManager(redis_client, CacheConfig())


async def get_performance_monitor(
    redis_client: RedisClient = Depends(get_redis_client)
) -> PerformanceMonitor:
    """Get PerformanceMonitor instance for metrics tracking."""
    return PerformanceMonitor(redis_client)


async def get_project_or_404(
    project_id: uuid.UUID,
    optimizer: QueryOptimizer = Depends(get_query_optimizer)
) -> ProjectIndex:
    """Get project by ID or raise 404 with optimized query."""
    project = await optimizer.get_project_with_stats(str(project_id))
    
    if not project:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "PROJECT_NOT_FOUND",
                "message": f"Project with ID {project_id} not found",
                "project_id": str(project_id)
            }
        )
    
    return project


# ================== REQUEST/RESPONSE SCHEMAS ==================

class StandardResponse(BaseModel):
    """Standard API response format with metadata."""
    data: Any = Field(..., description="Main response data")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    links: Dict[str, str] = Field(default_factory=dict, description="HATEOAS links")


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: Dict[str, Any] = Field(..., description="Error details")


class AnalyzeProjectRequest(BaseModel):
    """Request schema for project analysis."""
    file_paths: Optional[List[str]] = Field(None, description="Specific files to analyze")
    analysis_type: str = Field(default="full", description="Analysis type: full, incremental, context_optimization")
    force: bool = Field(default=False, description="Force reanalysis ignoring cache")
    configuration: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Analysis configuration overrides")


class RefreshProjectResponse(BaseModel):
    """Response schema for project refresh operation."""
    project_id: uuid.UUID = Field(..., description="Project ID")
    analysis_session_id: uuid.UUID = Field(..., description="Analysis session ID")
    status: str = Field(..., description="Analysis status")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


# ================== RATE LIMITING ==================

class RateLimiter:
    """Redis-based rate limiter for resource-intensive operations."""
    
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
    
    async def check_rate_limit(
        self,
        user_id: str,
        operation: str,
        limit: int,
        window_seconds: int
    ) -> bool:
        """Check if operation is within rate limits."""
        key = f"rate_limit:{user_id}:{operation}"
        
        # Get current count
        current = await self.redis.get(key)
        if current is None:
            # First request in window
            await self.redis.setex(key, window_seconds, 1)
            return True
        
        current_count = int(current)
        if current_count >= limit:
            return False
        
        # Increment counter
        await self.redis.incr(key)
        return True


async def rate_limit_analysis(
    user_id: str = Depends(get_current_user),
    redis_client: RedisClient = Depends(get_redis_client)
):
    """Rate limiting dependency for analysis operations."""
    limiter = RateLimiter(redis_client)
    
    # Allow 10 analysis requests per hour per user
    if not await limiter.check_rate_limit(user_id, "analysis", 10, 3600):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "RATE_LIMIT_EXCEEDED",
                "message": "Analysis rate limit exceeded. Try again later.",
                "limit": 10,
                "window": "1 hour"
            }
        )


# ================== PROJECT INDEX MANAGEMENT ENDPOINTS ==================

@router.post(
    "/create", 
    response_model=StandardResponse,
    status_code=200,
    summary="Create Project Index",
    description="Create and initialize a new project index with intelligent code analysis",
    response_description="Successfully created project index with analysis session scheduled",
    responses={
        200: {
            "description": "Project index created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "data": {
                            "id": "123e4567-e89b-12d3-a456-426614174000",
                            "name": "My Web Application",
                            "description": "Full-stack web application with React frontend",
                            "root_path": "/home/user/projects/webapp",
                            "git_repository_url": "https://github.com/user/webapp.git",
                            "git_branch": "main", 
                            "status": "inactive",
                            "file_count": 0,
                            "dependency_count": 0,
                            "created_at": "2024-01-15T10:30:00Z",
                            "updated_at": "2024-01-15T10:30:00Z"
                        },
                        "meta": {
                            "timestamp": "2024-01-15T10:30:00Z",
                            "correlation_id": "abc123",
                            "operation": "create_project"
                        },
                        "links": {
                            "self": "/api/project-index/123e4567-e89b-12d3-a456-426614174000",
                            "files": "/api/project-index/123e4567-e89b-12d3-a456-426614174000/files",
                            "dependencies": "/api/project-index/123e4567-e89b-12d3-a456-426614174000/dependencies"
                        }
                    }
                }
            }
        },
        400: {
            "description": "Invalid request parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "error": "INVALID_ROOT_PATH",
                            "message": "Root path does not exist: /invalid/path",
                            "field": "root_path"
                        }
                    }
                }
            }
        }
    }
)
async def create_project_index(
    request: ProjectIndexCreate,
    background_tasks: BackgroundTasks,
    indexer: ProjectIndexer = Depends(get_project_indexer),
    user_id: str = Depends(get_current_user)
):
    """
    Create and initialize a new project index.
    
    Creates a new project index with the specified configuration and triggers
    initial analysis in the background. The project index provides intelligent
    code analysis, dependency tracking, and context optimization.
    
    **Features:**
    - Automatic language detection and file classification
    - AST-based code analysis and dependency extraction  
    - Git integration for version tracking
    - Configurable file patterns and analysis settings
    - Background processing for non-blocking operation
    
    **Request Body Example:**
    ```json
    {
        "name": "My Web Application",
        "description": "Full-stack web application with React frontend",
        "root_path": "/home/user/projects/webapp",
        "git_repository_url": "https://github.com/user/webapp.git",
        "git_branch": "main",
        "configuration": {
            "languages": ["python", "javascript", "typescript"],
            "analysis_depth": 3,
            "enable_ai_analysis": true
        },
        "file_patterns": {
            "include": ["**/*.py", "**/*.js", "**/*.ts"],
            "exclude": ["node_modules/**", "__pycache__/**"]
        }
    }
    ```
    
    **Returns:**
    - Project index metadata with unique ID
    - HATEOAS links for related operations
    - Background analysis session information
    """
    try:
        # Validate root path exists
        root_path = Path(request.root_path).resolve()
        if not root_path.exists():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "INVALID_ROOT_PATH",
                    "message": f"Root path does not exist: {request.root_path}",
                    "field": "root_path"
                }
            )
        
        # Create project index
        project = await indexer.create_project(
            name=request.name,
            root_path=str(root_path),
            description=request.description,
            git_repository_url=request.git_repository_url,
            git_branch=request.git_branch,
            configuration=request.configuration,
            analysis_settings=request.analysis_settings,
            file_patterns=request.file_patterns,
            ignore_patterns=request.ignore_patterns
        )
        
        # Schedule initial analysis as background task
        background_tasks.add_task(
            _background_initial_analysis,
            str(project.id),
            indexer
        )
        
        # Create response
        response_data = ProjectIndexResponse.model_validate(project)
        
        return StandardResponse(
            data=response_data.model_dump(),
            meta={
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4()),
                "operation": "create_project"
            },
            links={
                "self": f"/api/project-index/{project.id}",
                "files": f"/api/project-index/{project.id}/files",
                "dependencies": f"/api/project-index/{project.id}/dependencies",
                "analyze": f"/api/project-index/{project.id}/analyze"
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "VALIDATION_ERROR",
                "message": str(e),
                "field": "root_path"
            }
        )
    except Exception as e:
        logger.error("Failed to create project index", error=str(e), user_id=user_id)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "Failed to create project index",
                "details": str(e)
            }
        )


@router.get(
    "/{project_id}",
    response_model=StandardResponse,
    summary="Get Project Index",
    description="Retrieve detailed project index information with comprehensive metadata",
    response_description="Project index details with statistics and analysis status",
    responses={
        200: {
            "description": "Project index retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "data": {
                            "id": "123e4567-e89b-12d3-a456-426614174000",
                            "name": "My Web Application",
                            "description": "Full-stack web application",
                            "root_path": "/home/user/projects/webapp",
                            "status": "active",
                            "file_count": 156,
                            "dependency_count": 342,
                            "last_indexed_at": "2024-01-15T10:45:00Z",
                            "last_analysis_at": "2024-01-15T10:45:00Z",
                            "configuration": {
                                "languages": ["python", "javascript"],
                                "analysis_depth": 3
                            }
                        },
                        "meta": {
                            "timestamp": "2024-01-15T11:00:00Z",
                            "correlation_id": "def456",
                            "cache_info": {"source": "database"}
                        },
                        "links": {
                            "self": "/api/project-index/123e4567-e89b-12d3-a456-426614174000",
                            "files": "/api/project-index/123e4567-e89b-12d3-a456-426614174000/files",
                            "dependencies": "/api/project-index/123e4567-e89b-12d3-a456-426614174000/dependencies"
                        }
                    }
                }
            }
        }
    }
)
async def get_project_index(
    project: ProjectIndex = Depends(get_project_or_404),
    cache_manager: CacheManager = Depends(get_cache_manager),
    monitor: PerformanceMonitor = Depends(get_performance_monitor)
):
    """
    Retrieve detailed project index information.
    
    Returns comprehensive project metadata including analysis statistics,
    file counts, dependency information, and current processing status.
    
    **Response includes:**
    - Project configuration and settings
    - File and dependency statistics
    - Analysis timestamps and status
    - Git repository information
    - HATEOAS navigation links
    
    **Use cases:**
    - Dashboard project overview
    - Status monitoring and health checks
    - Integration with external tools
    - Progress tracking for analysis operations
    
    **Performance Features:**
    - Response caching (5 minute TTL)
    - Optimized database queries
    - Performance monitoring
    """
    
    @monitor.monitor_endpoint("get_project_index")
    async def _get_project_with_caching():
        # Try cache first
        cached_response = await cache_manager.get_cached_response(
            "project", str(project.id)
        )
        
        if cached_response:
            logger.debug("Project cache hit", project_id=str(project.id))
            cached_response["meta"]["cache_info"] = {"source": "cache"}
            return cached_response
        
        # Build response
        response_data = ProjectIndexResponse.model_validate(project)
        
        response = StandardResponse(
            data=response_data.model_dump(),
            meta={
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4()),
                "cache_info": {"source": "database"}
            },
            links={
                "self": f"/api/project-index/{project.id}",
                "files": f"/api/project-index/{project.id}/files",
                "dependencies": f"/api/project-index/{project.id}/dependencies",
                "analyze": f"/api/project-index/{project.id}/analyze",
                "refresh": f"/api/project-index/{project.id}/refresh"
            }
        )
        
        # Cache response
        await cache_manager.set_cached_response(
            "project", str(project.id), data=response.model_dump()
        )
        
        return response.model_dump()
    
    try:
        return await _get_project_with_caching()
        
    except Exception as e:
        logger.error("Failed to get project index", project_id=str(project.id), error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "Failed to retrieve project index"
            }
        )


@router.put("/{project_id}/refresh", response_model=StandardResponse)
async def refresh_project_index(
    project_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    force: bool = Query(False, description="Force complete re-analysis"),
    project: ProjectIndex = Depends(get_project_or_404),
    indexer: ProjectIndexer = Depends(get_project_indexer),
    user_id: str = Depends(get_current_user),
    _rate_limited: None = Depends(rate_limit_analysis)
):
    """
    Force complete re-analysis of project.
    
    Triggers a comprehensive analysis session that re-processes all files
    and rebuilds the dependency graph.
    """
    try:
        # Check if project is already being analyzed
        session = indexer.session
        stmt = select(AnalysisSession).where(
            and_(
                AnalysisSession.project_id == project_id,
                AnalysisSession.status == AnalysisStatus.RUNNING
            )
        )
        result = await session.execute(stmt)
        active_session = result.scalar_one_or_none()
        
        if active_session:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "ANALYSIS_IN_PROGRESS",
                    "message": "Project analysis is already in progress",
                    "active_session_id": str(active_session.id)
                }
            )
        
        # Schedule analysis as background task
        analysis_session_id = str(uuid.uuid4())
        background_tasks.add_task(
            _background_full_analysis,
            str(project_id),
            analysis_session_id,
            force,
            indexer
        )
        
        response_data = RefreshProjectResponse(
            project_id=project_id,
            analysis_session_id=uuid.UUID(analysis_session_id),
            status="scheduled",
            estimated_completion=None
        )
        
        return StandardResponse(
            data=response_data.model_dump(),
            meta={
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4()),
                "operation": "refresh_project"
            },
            links={
                "self": f"/api/project-index/{project_id}",
                "analysis_status": f"/api/project-index/{project_id}/analysis/{analysis_session_id}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to refresh project", project_id=str(project_id), error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "Failed to refresh project index"
            }
        )


@router.delete("/{project_id}", response_model=StandardResponse)
async def delete_project_index(
    project_id: uuid.UUID,
    project: ProjectIndex = Depends(get_project_or_404),
    session: AsyncSession = Depends(get_session),
    user_id: str = Depends(get_current_user)
):
    """
    Remove project index and all associated data.
    
    Permanently deletes the project index, file entries, dependencies,
    analysis sessions, and snapshots.
    """
    try:
        # Get statistics before deletion
        files_count = await session.scalar(
            select(func.count(FileEntry.id)).where(FileEntry.project_id == project_id)
        )
        deps_count = await session.scalar(
            select(func.count(DependencyRelationship.id)).where(DependencyRelationship.project_id == project_id)
        )
        sessions_count = await session.scalar(
            select(func.count(AnalysisSession.id)).where(AnalysisSession.project_id == project_id)
        )
        
        # Delete project (cascade will handle related data)
        await session.delete(project)
        await session.commit()
        
        cleanup_summary = {
            "project_id": str(project_id),
            "project_name": project.name,
            "deleted_files": files_count or 0,
            "deleted_dependencies": deps_count or 0,
            "deleted_sessions": sessions_count or 0,
            "deleted_at": datetime.utcnow().isoformat()
        }
        
        return StandardResponse(
            data=cleanup_summary,
            meta={
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4()),
                "operation": "delete_project"
            }
        )
        
    except Exception as e:
        await session.rollback()
        logger.error("Failed to delete project", project_id=str(project_id), error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "Failed to delete project index"
            }
        )


# ================== FILE ANALYSIS ENDPOINTS ==================

@router.get("/{project_id}/files", response_model=StandardResponse)
async def list_project_files(
    project_id: uuid.UUID,
    language: Optional[str] = Query(None, description="Filter by programming language"),
    file_type: Optional[FileType] = Query(None, description="Filter by file type"),
    modified_after: Optional[datetime] = Query(None, description="Filter by modification date"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=500, description="Results per page"),
    project: ProjectIndex = Depends(get_project_or_404),
    session: AsyncSession = Depends(get_session)
):
    """
    List all analyzed files with filtering capabilities.
    
    Supports filtering by language, file type, modification date, and pagination.
    """
    try:
        # Build query with filters
        query = select(FileEntry).where(FileEntry.project_id == project_id)
        
        if language:
            query = query.where(FileEntry.language == language)
        
        if file_type:
            query = query.where(FileEntry.file_type == file_type)
        
        if modified_after:
            query = query.where(FileEntry.last_modified >= modified_after)
        
        # Add pagination
        offset = (page - 1) * limit
        query = query.offset(offset).limit(limit).order_by(FileEntry.relative_path)
        
        # Get total count
        count_query = select(func.count(FileEntry.id)).where(FileEntry.project_id == project_id)
        if language:
            count_query = count_query.where(FileEntry.language == language)
        if file_type:
            count_query = count_query.where(FileEntry.file_type == file_type)
        if modified_after:
            count_query = count_query.where(FileEntry.last_modified >= modified_after)
        
        # Execute queries
        result = await session.execute(query)
        files = result.scalars().all()
        
        total_result = await session.execute(count_query)
        total = total_result.scalar()
        
        # Convert to response format
        file_responses = [FileEntryResponse.model_validate(f) for f in files]
        
        list_response = FileEntryListResponse(
            files=file_responses,
            total=total,
            page=page,
            page_size=limit,
            has_next=(page * limit) < total,
            has_prev=page > 1
        )
        
        return StandardResponse(
            data=list_response.model_dump(),
            meta={
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4()),
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": total,
                    "total_pages": (total + limit - 1) // limit
                }
            },
            links={
                "self": f"/api/project-index/{project_id}/files?page={page}&limit={limit}",
                "first": f"/api/project-index/{project_id}/files?page=1&limit={limit}",
                "last": f"/api/project-index/{project_id}/files?page={(total + limit - 1) // limit}&limit={limit}"
            }
        )
        
    except Exception as e:
        logger.error("Failed to list project files", project_id=str(project_id), error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "Failed to retrieve project files"
            }
        )


@router.get("/{project_id}/files/{file_path:path}", response_model=StandardResponse)
async def get_file_analysis(
    project_id: uuid.UUID,
    file_path: str = FastAPIPath(..., description="File path within project"),
    project: ProjectIndex = Depends(get_project_or_404),
    session: AsyncSession = Depends(get_session)
):
    """
    Get detailed analysis for specific file.
    
    Returns complete file analysis including dependencies, functions, classes,
    and AI-generated insights.
    """
    try:
        # Decode URL-encoded file path
        decoded_path = unquote(file_path)
        
        # Query file entry
        stmt = select(FileEntry).where(
            and_(
                FileEntry.project_id == project_id,
                FileEntry.relative_path == decoded_path
            )
        ).options(
            selectinload(FileEntry.outgoing_dependencies),
            selectinload(FileEntry.incoming_dependencies)
        )
        
        result = await session.execute(stmt)
        file_entry = result.scalar_one_or_none()
        
        if not file_entry:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "FILE_NOT_FOUND",
                    "message": f"File not found: {decoded_path}",
                    "file_path": decoded_path
                }
            )
        
        # Convert to response format
        response_data = FileEntryResponse.model_validate(file_entry)
        
        # Add dependency information
        outgoing_deps = [
            DependencyRelationshipResponse.model_validate(dep) 
            for dep in file_entry.outgoing_dependencies
        ]
        incoming_deps = [
            DependencyRelationshipResponse.model_validate(dep) 
            for dep in file_entry.incoming_dependencies
        ]
        
        enhanced_data = response_data.model_dump()
        enhanced_data["outgoing_dependencies"] = [dep.model_dump() for dep in outgoing_deps]
        enhanced_data["incoming_dependencies"] = [dep.model_dump() for dep in incoming_deps]
        
        return StandardResponse(
            data=enhanced_data,
            meta={
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4()),
                "file_path": decoded_path
            },
            links={
                "self": f"/api/project-index/{project_id}/files/{file_path}",
                "project": f"/api/project-index/{project_id}",
                "dependencies": f"/api/project-index/{project_id}/dependencies?source_file_id={file_entry.id}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get file analysis", 
                    project_id=str(project_id), 
                    file_path=file_path, 
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "Failed to retrieve file analysis"
            }
        )


@router.post("/{project_id}/analyze", response_model=StandardResponse)
async def analyze_project(
    project_id: uuid.UUID,
    request: AnalyzeProjectRequest,
    background_tasks: BackgroundTasks,
    project: ProjectIndex = Depends(get_project_or_404),
    indexer: ProjectIndexer = Depends(get_project_indexer),
    user_id: str = Depends(get_current_user),
    _rate_limited: None = Depends(rate_limit_analysis)
):
    """
    Trigger analysis for specific files or entire project.
    
    Supports full analysis, incremental updates, or context optimization
    with configurable parameters.
    """
    try:
        # Validate analysis type
        analysis_type_map = {
            "full": CoreAnalysisSessionType.FULL_ANALYSIS,
            "incremental": CoreAnalysisSessionType.INCREMENTAL,
            "context_optimization": CoreAnalysisSessionType.CONTEXT_OPTIMIZATION,
            "dependency_mapping": CoreAnalysisSessionType.DEPENDENCY_MAPPING
        }
        
        if request.analysis_type not in analysis_type_map:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "INVALID_ANALYSIS_TYPE",
                    "message": f"Invalid analysis type: {request.analysis_type}",
                    "valid_types": list(analysis_type_map.keys())
                }
            )
        
        analysis_type = analysis_type_map[request.analysis_type]
        
        # Create analysis configuration
        config = AnalysisConfiguration(
            force_reanalysis=request.force,
            file_filters=request.file_paths,
            custom_settings=request.configuration
        )
        
        # Schedule analysis as background task
        analysis_session_id = str(uuid.uuid4())
        background_tasks.add_task(
            _background_targeted_analysis,
            str(project_id),
            analysis_session_id,
            analysis_type,
            config,
            indexer
        )
        
        response_data = {
            "project_id": str(project_id),
            "analysis_session_id": analysis_session_id,
            "analysis_type": request.analysis_type,
            "status": "scheduled",
            "configuration": config.to_dict() if hasattr(config, 'to_dict') else request.configuration
        }
        
        return StandardResponse(
            data=response_data,
            meta={
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4()),
                "operation": "analyze_project"
            },
            links={
                "self": f"/api/project-index/{project_id}/analyze",
                "project": f"/api/project-index/{project_id}",
                "analysis_status": f"/api/project-index/{project_id}/analysis/{analysis_session_id}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to start analysis", project_id=str(project_id), error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "Failed to start project analysis"
            }
        )


# ================== DEPENDENCIES & CONTEXT ENDPOINT ==================

@router.get(
    "/{project_id}/dependencies",
    response_model=StandardResponse,
    summary="Get Project Dependencies",
    description="Retrieve comprehensive dependency graph and relationships with advanced filtering",
    response_description="Dependency data in requested format with metadata and statistics",
    responses={
        200: {
            "description": "Dependencies retrieved successfully",
            "content": {
                "application/json": {
                    "examples": {
                        "json_format": {
                            "summary": "Standard JSON format with pagination",
                            "value": {
                                "data": {
                                    "dependencies": [
                                        {
                                            "id": "dep123",
                                            "source_file_id": "file456",
                                            "target_name": "requests",
                                            "dependency_type": "import",
                                            "is_external": True,
                                            "confidence_score": 1.0,
                                            "line_number": 1
                                        }
                                    ],
                                    "total": 150,
                                    "page": 1,
                                    "page_size": 100,
                                    "has_next": True
                                },
                                "meta": {
                                    "format": "json",
                                    "filters": {
                                        "include_external": True,
                                        "depth": 1
                                    }
                                }
                            }
                        },
                        "graph_format": {
                            "summary": "Graph format with nodes and edges",
                            "value": {
                                "data": {
                                    "nodes": [
                                        {
                                            "file_id": "file123",
                                            "file_path": "src/main.py",
                                            "file_type": "source",
                                            "language": "python",
                                            "in_degree": 2,
                                            "out_degree": 5
                                        }
                                    ],
                                    "edges": [
                                        {
                                            "source_file_id": "file123",
                                            "target_file_id": "file456",
                                            "dependency_type": "import",
                                            "is_external": False
                                        }
                                    ],
                                    "statistics": {
                                        "total_nodes": 50,
                                        "total_edges": 120,
                                        "external_dependencies": 30,
                                        "avg_in_degree": 2.4
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
)
async def get_project_dependencies(
    project_id: uuid.UUID,
    file_path: Optional[str] = Query(None, description="Dependencies for specific file"),
    depth: int = Query(1, ge=1, le=5, description="Dependency traversal depth"),
    include_external: bool = Query(True, description="Include external dependencies"),
    format: str = Query("json", description="Response format: json, graph, tree"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(100, ge=1, le=1000, description="Results per page"),
    project: ProjectIndex = Depends(get_project_or_404),
    session: AsyncSession = Depends(get_session)
):
    """
    Retrieve dependency graph and relationships.
    
    Provides comprehensive dependency analysis with multiple response formats
    and advanced filtering capabilities for different use cases.
    
    **Response Formats:**
    - `json`: Standard paginated list of dependencies
    - `graph`: Network graph with nodes and edges
    - `tree`: Hierarchical dependency tree (coming soon)
    
    **Filtering Options:**
    - Filter by specific file path
    - Control dependency traversal depth
    - Include/exclude external dependencies
    - Pagination for large result sets
    
    **Use Cases:**
    - Dependency visualization and analysis
    - Impact analysis for code changes
    - Architecture documentation
    - Circular dependency detection
    - External library usage tracking
    
    **Query Examples:**
    ```
    # Get all dependencies with pagination
    GET /api/project-index/{id}/dependencies?page=1&limit=50
    
    # Get dependencies for specific file
    GET /api/project-index/{id}/dependencies?file_path=src/main.py
    
    # Get graph format for visualization
    GET /api/project-index/{id}/dependencies?format=graph
    
    # Get only internal dependencies
    GET /api/project-index/{id}/dependencies?include_external=false
    ```
    """
    try:
        # Build base query
        query = select(DependencyRelationship).where(
            DependencyRelationship.project_id == project_id
        )
        
        # Apply filters
        if file_path:
            # Find file entry first
            file_stmt = select(FileEntry).where(
                and_(
                    FileEntry.project_id == project_id,
                    FileEntry.relative_path == file_path
                )
            )
            file_result = await session.execute(file_stmt)
            file_entry = file_result.scalar_one_or_none()
            
            if not file_entry:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "FILE_NOT_FOUND",
                        "message": f"File not found: {file_path}"
                    }
                )
            
            query = query.where(DependencyRelationship.source_file_id == file_entry.id)
        
        if not include_external:
            query = query.where(DependencyRelationship.is_external == False)
        
        # Handle different response formats
        if format == "graph":
            return await _get_dependency_graph(project_id, session, include_external, depth)
        elif format == "tree":
            return await _get_dependency_tree(project_id, session, file_path, depth)
        
        # Standard JSON format with pagination
        count_query = select(func.count(DependencyRelationship.id)).where(
            DependencyRelationship.project_id == project_id
        )
        if file_path and 'file_entry' in locals():
            count_query = count_query.where(DependencyRelationship.source_file_id == file_entry.id)
        if not include_external:
            count_query = count_query.where(DependencyRelationship.is_external == False)
        
        # Add pagination
        offset = (page - 1) * limit
        query = query.offset(offset).limit(limit).order_by(DependencyRelationship.target_name)
        
        # Execute queries
        result = await session.execute(query)
        dependencies = result.scalars().all()
        
        total_result = await session.execute(count_query)
        total = total_result.scalar()
        
        # Convert to response format
        dep_responses = [DependencyRelationshipResponse.model_validate(dep) for dep in dependencies]
        
        list_response = DependencyRelationshipListResponse(
            dependencies=dep_responses,
            total=total,
            page=page,
            page_size=limit,
            has_next=(page * limit) < total,
            has_prev=page > 1
        )
        
        return StandardResponse(
            data=list_response.model_dump(),
            meta={
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4()),
                "format": format,
                "filters": {
                    "file_path": file_path,
                    "include_external": include_external,
                    "depth": depth
                },
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": total
                }
            },
            links={
                "self": f"/api/project-index/{project_id}/dependencies",
                "project": f"/api/project-index/{project_id}",
                "graph": f"/api/project-index/{project_id}/dependencies?format=graph",
                "tree": f"/api/project-index/{project_id}/dependencies?format=tree"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get dependencies", project_id=str(project_id), error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "Failed to retrieve project dependencies"
            }
        )


# ================== BACKGROUND TASK FUNCTIONS ==================

async def _background_initial_analysis(project_id: str, indexer: ProjectIndexer):
    """Background task for initial project analysis."""
    try:
        logger.info("Starting initial project analysis", project_id=project_id)
        
        await indexer.analyze_project(
            project_id=project_id,
            analysis_type=CoreAnalysisSessionType.FULL_ANALYSIS,
            force_reanalysis=False
        )
        
        logger.info("Initial project analysis completed", project_id=project_id)
        
    except Exception as e:
        logger.error("Initial project analysis failed", project_id=project_id, error=str(e))


async def _background_full_analysis(
    project_id: str, 
    session_id: str, 
    force: bool, 
    indexer: ProjectIndexer
):
    """Background task for full project re-analysis."""
    try:
        logger.info("Starting full project re-analysis", 
                   project_id=project_id, session_id=session_id)
        
        await indexer.analyze_project(
            project_id=project_id,
            analysis_type=CoreAnalysisSessionType.FULL_ANALYSIS,
            force_reanalysis=force
        )
        
        logger.info("Full project re-analysis completed", 
                   project_id=project_id, session_id=session_id)
        
    except Exception as e:
        logger.error("Full project re-analysis failed", 
                    project_id=project_id, session_id=session_id, error=str(e))


async def _background_targeted_analysis(
    project_id: str,
    session_id: str,
    analysis_type: CoreAnalysisSessionType,
    config: AnalysisConfiguration,
    indexer: ProjectIndexer
):
    """Background task for targeted project analysis."""
    try:
        logger.info("Starting targeted project analysis",
                   project_id=project_id,
                   session_id=session_id,
                   analysis_type=analysis_type.value)
        
        await indexer.analyze_project(
            project_id=project_id,
            analysis_type=analysis_type,
            force_reanalysis=config.force_reanalysis if hasattr(config, 'force_reanalysis') else False,
            analysis_config=config
        )
        
        logger.info("Targeted project analysis completed",
                   project_id=project_id, session_id=session_id)
        
    except Exception as e:
        logger.error("Targeted project analysis failed",
                    project_id=project_id, session_id=session_id, error=str(e))


# ================== HELPER FUNCTIONS ==================

async def _get_dependency_graph(
    project_id: uuid.UUID, 
    session: AsyncSession, 
    include_external: bool, 
    depth: int
) -> StandardResponse:
    """Generate dependency graph response format."""
    try:
        # Get all files
        files_stmt = select(FileEntry).where(FileEntry.project_id == project_id)
        files_result = await session.execute(files_stmt)
        files = files_result.scalars().all()
        
        # Get dependencies
        deps_stmt = select(DependencyRelationship).where(
            DependencyRelationship.project_id == project_id
        )
        if not include_external:
            deps_stmt = deps_stmt.where(DependencyRelationship.is_external == False)
        
        deps_result = await session.execute(deps_stmt)
        dependencies = deps_result.scalars().all()
        
        # Build graph nodes
        nodes = []
        for file_entry in files:
            # Calculate in/out degree
            in_degree = len([d for d in dependencies if d.target_file_id == file_entry.id])
            out_degree = len([d for d in dependencies if d.source_file_id == file_entry.id])
            
            node = DependencyGraphNode(
                file_id=file_entry.id,
                file_path=file_entry.relative_path,
                file_type=file_entry.file_type,
                language=file_entry.language,
                in_degree=in_degree,
                out_degree=out_degree
            )
            nodes.append(node)
        
        # Build graph edges
        edges = []
        for dep in dependencies:
            edge = DependencyGraphEdge(
                source_file_id=dep.source_file_id,
                target_file_id=dep.target_file_id,
                target_name=dep.target_name,
                dependency_type=dep.dependency_type,
                is_external=dep.is_external
            )
            edges.append(edge)
        
        # Calculate statistics
        stats = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "external_dependencies": len([e for e in edges if e.is_external]),
            "internal_dependencies": len([e for e in edges if not e.is_external]),
            "avg_in_degree": sum(n.in_degree for n in nodes) / len(nodes) if nodes else 0,
            "avg_out_degree": sum(n.out_degree for n in nodes) / len(nodes) if nodes else 0
        }
        
        graph = DependencyGraph(
            nodes=nodes,
            edges=edges,
            statistics=stats
        )
        
        return StandardResponse(
            data=graph.model_dump(),
            meta={
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4()),
                "format": "graph",
                "statistics": stats
            }
        )
        
    except Exception as e:
        logger.error("Failed to generate dependency graph", project_id=str(project_id), error=str(e))
        raise


async def _get_dependency_tree(
    project_id: uuid.UUID,
    session: AsyncSession,
    root_file_path: Optional[str],
    depth: int
) -> StandardResponse:
    """Generate dependency tree response format."""
    try:
        # This is a simplified tree format
        # In a full implementation, you would build a hierarchical tree structure
        
        tree_data = {
            "root": root_file_path or "project_root",
            "depth": depth,
            "tree": "Tree format not yet implemented - use graph format",
            "note": "Tree format would show hierarchical dependency relationships"
        }
        
        return StandardResponse(
            data=tree_data,
            meta={
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4()),
                "format": "tree"
            }
        )
        
    except Exception as e:
        logger.error("Failed to generate dependency tree", project_id=str(project_id), error=str(e))
        raise


# ================== WEBSOCKET ENDPOINTS ==================

@router.websocket("/ws")
async def project_index_websocket(
    websocket: WebSocket,
    token: str = Query(..., description="Authentication token"),
    handler = Depends(get_websocket_handler)
):
    """
    WebSocket endpoint for real-time project index updates.
    
    Provides real-time notifications for:
    - Analysis progress updates
    - File change events
    - Dependency graph changes
    - Project status changes
    
    Authentication required via query parameter 'token'.
    
    Message format:
    ```json
    {
        "action": "subscribe",
        "event_types": ["analysis_progress", "file_change"],
        "project_id": "optional-project-uuid",
        "session_id": "optional-session-uuid"
    }
    ```
    """
    try:
        # Authenticate user
        async with websocket_auth_context(token) as user_id:
            await handler.handle_connection(websocket, user_id)
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    
    except Exception as e:
        logger.error("WebSocket connection error", error=str(e))
        try:
            await websocket.close(code=1011, reason="Authentication failed")
        except:
            pass


@router.get("/ws/stats", response_model=StandardResponse)
async def get_websocket_stats():
    """
    Get WebSocket connection and subscription statistics.
    
    Returns information about active connections, subscriptions,
    and real-time event distribution.
    """
    try:
        stats = await get_subscription_stats()
        
        return StandardResponse(
            data=stats,
            meta={
                "timestamp": datetime.utcnow().isoformat(),
                "correlation_id": str(uuid.uuid4()),
                "operation": "websocket_stats"
            }
        )
        
    except Exception as e:
        logger.error("Failed to get WebSocket stats", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "Failed to retrieve WebSocket statistics"
            }
        )