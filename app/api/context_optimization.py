"""
Enhanced Context Optimization API Endpoints for LeanVibe Agent Hive 2.0

Advanced AI-powered context optimization endpoints that provide intelligent
file selection and context assembly for development tasks. Integrates with
the complete context optimization engine including ML analysis, historical
data, and similarity-based caching.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import structlog
import psutil
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from pydantic import BaseModel, Field, ValidationError, validator

from ..core.database import get_session
from ..core.redis import get_redis_client, RedisClient
from ..models.project_index import ProjectIndex, FileEntry
from ..project_index.context_optimizer import (
    ContextOptimizer, ContextRequest, TaskType, OptimizedContext
)
from ..project_index.relevance_analyzer import RelevanceAnalyzer
from ..project_index.context_assembler import (
    ContextAssembler, AssemblyConfiguration, AssemblyStrategy, ContextFormat
)
from ..project_index.ml_analyzer import MLAnalyzer
from ..project_index.historical_analyzer import HistoricalAnalyzer, AnalysisScope
from ..project_index.context_cache import (
    ContextCacheManager, CacheConfiguration, CacheKey
)
from ..project_index.core import ProjectIndexer
from ..project_index.models import AnalysisConfiguration
from ..project_index.performance_optimizer import (
    PerformanceOptimizer, ResourceLimits, PerformanceMetrics,
    get_performance_optimizer, StreamingChunk
)

logger = structlog.get_logger()

# Create router
router = APIRouter(
    prefix="/api/context-optimization",
    tags=["Context Optimization"],
    responses={
        400: {"description": "Bad request - validation error"},
        401: {"description": "Unauthorized - authentication required"},
        403: {"description": "Forbidden - insufficient permissions"}, 
        404: {"description": "Not found - resource does not exist"},
        429: {"description": "Too many requests - rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)

# Mock auth function - replace with actual implementation
async def get_current_user():
    """Mock current user."""
    return "test_user"


# ================== REQUEST/RESPONSE SCHEMAS ==================

class AIModelInfo(BaseModel):
    """AI model information for context optimization."""
    model_name: str = Field(..., description="AI model name (e.g., gpt-4, claude-3)")
    context_window: int = Field(..., description="Model context window in tokens")
    preferred_format: str = Field(default="structured", description="Preferred context format")


class ContextPreferences(BaseModel):
    """Context optimization preferences."""
    max_files: int = Field(default=15, ge=1, le=50, description="Maximum files to include")
    max_tokens: int = Field(default=32000, ge=1000, le=200000, description="Maximum tokens")
    include_tests: bool = Field(default=False, description="Include test files")
    include_docs: bool = Field(default=True, description="Include documentation files")
    depth_limit: int = Field(default=3, ge=1, le=5, description="Dependency traversal depth")
    relevance_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum relevance score")


class HistoricalContext(BaseModel):
    """Historical context information."""
    recent_files: List[str] = Field(default_factory=list, description="Recently modified files")
    frequent_files: List[str] = Field(default_factory=list, description="Frequently accessed files")
    related_issues: List[str] = Field(default_factory=list, description="Related issue numbers")


class AdvancedContextRequest(BaseModel):
    """Advanced context optimization request schema."""
    task_description: str = Field(..., description="Natural language task description")
    task_type: str = Field(..., description="Task type: feature, bugfix, refactoring, analysis, documentation")
    files_mentioned: List[str] = Field(default_factory=list, description="Files explicitly mentioned")
    context_preferences: ContextPreferences = Field(default_factory=ContextPreferences)
    ai_model_info: Optional[AIModelInfo] = Field(None, description="AI model information")
    historical_context: HistoricalContext = Field(default_factory=HistoricalContext)
    assembly_strategy: str = Field(default="balanced", description="Assembly strategy")
    output_format: str = Field(default="structured", description="Output format")
    enable_caching: bool = Field(default=True, description="Enable result caching")
    stream_response: bool = Field(default=False, description="Stream response for large contexts")
    enable_parallel: bool = Field(default=True, description="Enable parallel processing")
    max_memory_mb: Optional[int] = Field(None, ge=512, le=8192, description="Maximum memory usage in MB")
    max_parallel_tasks: Optional[int] = Field(None, ge=1, le=32, description="Maximum parallel tasks")
    timeout_seconds: Optional[int] = Field(None, ge=30, le=1800, description="Processing timeout in seconds")
    
    @validator('task_type')
    def validate_task_type(cls, v):
        valid_types = ['feature', 'bugfix', 'refactoring', 'analysis', 'documentation', 'testing', 'performance', 'security']
        if v not in valid_types:
            raise ValueError(f"Task type must be one of: {valid_types}")
        return v
    
    @validator('assembly_strategy')
    def validate_assembly_strategy(cls, v):
        valid_strategies = ['hierarchical', 'dependency_first', 'task_focused', 'balanced', 'streaming', 'layered']
        if v not in valid_strategies:
            raise ValueError(f"Assembly strategy must be one of: {valid_strategies}")
        return v


class FileRelevanceInfo(BaseModel):
    """File relevance information in response."""
    file_path: str = Field(..., description="File path")
    relevance_score: float = Field(..., description="Relevance score (0-1)")
    relevance_reasons: List[str] = Field(..., description="Reasons for relevance")
    content_summary: str = Field(..., description="AI-generated content summary")
    key_functions: List[str] = Field(..., description="Key functions in file")
    key_classes: List[str] = Field(..., description="Key classes in file")
    import_relationships: List[str] = Field(..., description="Import relationships")
    estimated_tokens: int = Field(..., description="Estimated token count")


class DependencyGraphResponse(BaseModel):
    """Dependency graph in response."""
    nodes: List[Dict[str, Any]] = Field(..., description="Graph nodes")
    edges: List[Dict[str, Any]] = Field(..., description="Graph edges")


class ContextSummary(BaseModel):
    """Context summary information."""
    total_files: int = Field(..., description="Total files included")
    total_tokens: int = Field(..., description="Total estimated tokens")
    coverage_percentage: float = Field(..., description="Project coverage percentage")
    confidence_score: float = Field(..., description="Overall confidence score")
    architectural_patterns: List[str] = Field(..., description="Identified patterns")
    potential_challenges: List[str] = Field(..., description="Potential challenges")
    recommended_approach: str = Field(..., description="Recommended approach")


class OptimizationMetadata(BaseModel):
    """Optimization process metadata."""
    algorithm_used: str = Field(..., description="Optimization algorithm")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    cache_hit_rate: float = Field(..., description="Cache hit rate")
    relevance_distribution: Dict[str, int] = Field(..., description="Relevance score distribution")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="Performance optimization metrics")


class Suggestions(BaseModel):
    """Optimization suggestions."""
    additional_files: List[str] = Field(..., description="Additional relevant files")
    alternative_contexts: List[str] = Field(..., description="Alternative context strategies")
    optimization_tips: List[str] = Field(..., description="Optimization tips")


class OptimizedContextResponse(BaseModel):
    """Complete optimized context response."""
    core_files: List[FileRelevanceInfo] = Field(..., description="Core files for immediate attention")
    supporting_files: List[FileRelevanceInfo] = Field(..., description="Supporting files")
    dependency_graph: DependencyGraphResponse = Field(..., description="Dependency relationships")
    context_summary: ContextSummary = Field(..., description="Context summary")
    optimization_metadata: OptimizationMetadata = Field(..., description="Optimization metadata")
    suggestions: Suggestions = Field(..., description="Optimization suggestions")


class ContextCacheStats(BaseModel):
    """Context cache statistics."""
    total_requests: int = Field(..., description="Total cache requests")
    cache_hits: int = Field(..., description="Cache hits")
    cache_misses: int = Field(..., description="Cache misses")
    similarity_hits: int = Field(..., description="Similarity-based hits")
    hit_rate: float = Field(..., description="Overall hit rate")
    entries_cached: int = Field(..., description="Number of cached entries")
    avg_quality_score: float = Field(..., description="Average quality score")


# ================== DEPENDENCY INJECTION ==================

async def get_context_optimizer(
    session: AsyncSession = Depends(get_session),
    redis_client: RedisClient = Depends(get_redis_client)
) -> ContextOptimizer:
    """Get ContextOptimizer with all dependencies."""
    # Initialize analyzers
    cache_config = CacheConfiguration()
    cache_manager = ContextCacheManager(cache_config, redis_client)
    ml_analyzer = MLAnalyzer(cache_embeddings=True)
    historical_analyzer = HistoricalAnalyzer(cache_results=True)
    
    return ContextOptimizer(
        cache_manager=cache_manager,
        ml_analyzer=ml_analyzer,
        historical_analyzer=historical_analyzer
    )


async def get_context_assembler() -> ContextAssembler:
    """Get ContextAssembler instance."""
    return ContextAssembler()


async def get_cache_manager(
    redis_client: RedisClient = Depends(get_redis_client)
) -> ContextCacheManager:
    """Get ContextCacheManager instance."""
    cache_config = CacheConfiguration()
    return ContextCacheManager(cache_config, redis_client)


async def get_project_indexer(
    session: AsyncSession = Depends(get_session)
) -> ProjectIndexer:
    """Get ProjectIndexer instance."""
    return ProjectIndexer(session=session)


async def get_performance_optimizer_dep(
    request: AdvancedContextRequest = None
) -> PerformanceOptimizer:
    """Get PerformanceOptimizer with request-specific limits."""
    if request:
        # Create custom resource limits from request
        limits = ResourceLimits(
            max_memory_mb=request.max_memory_mb or ResourceLimits().max_memory_mb,
            max_parallel_tasks=request.max_parallel_tasks or ResourceLimits().max_parallel_tasks,
            timeout_seconds=request.timeout_seconds or ResourceLimits().timeout_seconds
        )
        return PerformanceOptimizer(limits)
    else:
        # Use system defaults
        return get_performance_optimizer()


# ================== MAIN CONTEXT OPTIMIZATION ENDPOINT ==================

@router.post(
    "/{project_id}/optimize",
    response_model=OptimizedContextResponse,
    summary="Optimize Context for AI Agent",
    description="Generate optimized context using advanced AI algorithms",
    response_description="Optimized context with intelligent file selection and assembly"
)
async def optimize_context(
    project_id: uuid.UUID,
    request: AdvancedContextRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
    context_optimizer: ContextOptimizer = Depends(get_context_optimizer),
    context_assembler: ContextAssembler = Depends(get_context_assembler),
    cache_manager: ContextCacheManager = Depends(get_cache_manager),
    user_id: str = Depends(get_current_user)
):
    """
    Generate optimized context for AI agents using advanced algorithms.
    
    This endpoint provides intelligent context optimization that:
    - Uses AI-powered relevance scoring with multiple algorithms
    - Performs semantic similarity analysis and structural importance ranking
    - Incorporates historical development patterns and team collaboration data
    - Applies machine learning for pattern recognition and anomaly detection
    - Provides intelligent context assembly with multiple strategies
    - Includes advanced caching with similarity-based retrieval
    
    **Context Optimization Process:**
    1. **Relevance Analysis**: Multi-algorithm scoring (semantic, structural, historical, ML)
    2. **Intelligent Filtering**: Token budget management and quality thresholds
    3. **Context Assembly**: Strategic organization based on task type and preferences
    4. **Quality Assessment**: Confidence scoring and optimization recommendations
    5. **Caching**: Smart caching with similarity-based retrieval for performance
    
    **Assembly Strategies:**
    - `hierarchical`: Organize by importance and structure
    - `dependency_first`: Start with dependencies and build outward
    - `task_focused`: Optimize for specific task type
    - `balanced`: Combine multiple organizational principles
    - `streaming`: Progressive context building for large projects
    - `layered`: Architectural layer-based organization
    
    **Output Formats:**
    - `structured`: Organized layers with metadata
    - `narrative`: Human-readable descriptions
    - `minimal`: Essential files only
    - `markdown`: Formatted for documentation
    """
    start_time = time.time()
    
    try:
        # Validate project exists
        project_query = select(ProjectIndex).where(ProjectIndex.id == project_id)
        result = await session.execute(project_query)
        project = result.scalar_one_or_none()
        
        if not project:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "PROJECT_NOT_FOUND",
                    "message": f"Project with ID {project_id} not found"
                }
            )
        
        # Get file analysis results
        files_query = select(FileEntry).where(FileEntry.project_id == project_id)
        files_result = await session.execute(files_query)
        file_entries = files_result.scalars().all()
        
        if not file_entries:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "NO_FILES_FOUND",
                    "message": "No analyzed files found for this project"
                }
            )
        
        # Convert to FileAnalysisResult objects
        file_results = []
        for entry in file_entries:
            from ..project_index.models import FileAnalysisResult
            file_result = FileAnalysisResult(
                file_path=entry.relative_path,
                file_name=Path(entry.relative_path).name,
                file_size=entry.file_size,
                line_count=entry.line_count,
                file_type=entry.file_type,
                language=entry.language,
                file_hash=entry.file_hash,
                analysis_successful=True,
                analysis_data=entry.analysis_data or {}
            )
            file_results.append(file_result)
        
        # Create context request
        task_type_map = {
            "feature": TaskType.FEATURE,
            "bugfix": TaskType.BUGFIX,
            "refactoring": TaskType.REFACTORING,
            "analysis": TaskType.ANALYSIS,
            "documentation": TaskType.DOCUMENTATION,
            "testing": TaskType.TESTING,
            "performance": TaskType.PERFORMANCE,
            "security": TaskType.SECURITY
        }
        
        context_request = ContextRequest(
            task_description=request.task_description,
            task_type=task_type_map.get(request.task_type, TaskType.ANALYSIS),
            files_mentioned=request.files_mentioned,
            context_preferences=request.context_preferences.dict(),
            ai_model_info=request.ai_model_info.dict() if request.ai_model_info else {},
            historical_context=request.historical_context.dict()
        )
        
        # Check cache if enabled
        if request.enable_caching:
            cache_key = CacheKey.from_request(
                context_request, str(project_id), file_results
            )
            
            cached_result = await cache_manager.get_cached_context(cache_key)
            if cached_result:
                logger.info("Context optimization cache hit",
                           project_id=str(project_id),
                           cache_key=cache_key.to_string())
                
                return await _convert_cached_result_to_response(cached_result)
        
        # Get dependency graph (simplified - would integrate with actual graph)
        from ..project_index.graph import DependencyGraph
        dependency_graph = DependencyGraph()
        
        # Load dependencies into graph
        for entry in file_entries:
            dependency_graph.add_node(entry.relative_path, {
                "file_type": entry.file_type.value if entry.file_type else "unknown",
                "language": entry.language
            })
        
        # Perform context optimization with performance optimization
        logger.info("Starting context optimization",
                   project_id=str(project_id),
                   task_type=request.task_type,
                   file_count=len(file_results),
                   enable_parallel=request.enable_parallel)
        
        # Check if we should use performance-optimized processing
        if request.enable_parallel and len(file_results) > 20:
            # Use performance optimizer for large projects
            performance_optimizer = await get_performance_optimizer_dep(request)
            
            # Get individual analyzers for parallel processing
            relevance_analyzer = RelevanceAnalyzer()
            
            optimized_context = await performance_optimizer.optimize_context_parallel(
                context_request=context_request,
                file_results=file_results,
                relevance_analyzer=relevance_analyzer,
                context_assembler=context_assembler,
                dependency_graph=dependency_graph
            )
            
            # Include performance metrics in response
            performance_metrics = performance_optimizer.get_performance_metrics()
            
        else:
            # Use standard optimization for smaller projects
            optimized_context = await context_optimizer.optimize_context(
                context_request=context_request,
                file_results=file_results,
                dependency_graph=dependency_graph,
                project_path=project.root_path
            )
            performance_metrics = None
        
        # Apply context assembly if requested
        if request.assembly_strategy != "none":
            assembly_config = AssemblyConfiguration(
                strategy=AssemblyStrategy(request.assembly_strategy),
                format=ContextFormat(request.output_format),
                max_tokens=request.context_preferences.max_tokens,
                include_navigation=True,
                include_summaries=True
            )
            
            # Convert relevance scores for assembly
            relevance_scores = optimized_context.core_files + optimized_context.supporting_files
            
            assembled_context = await context_assembler.assemble_context(
                relevance_scores=relevance_scores,
                context_request=context_request,
                dependency_graph=dependency_graph,
                config=assembly_config,
                file_results=file_results
            )
            
            # Use assembled context for response
            final_context = assembled_context
        else:
            final_context = optimized_context
        
        # Cache result if enabled
        if request.enable_caching and 'cache_key' in locals():
            quality_score = final_context.context_summary.get('confidence_score', 0.8)
            await cache_manager.cache_context(cache_key, final_context, quality_score)
        
        # Convert to response format
        response = await _convert_context_to_response(final_context, start_time, performance_metrics)
        
        # Schedule background optimization improvements
        background_tasks.add_task(
            _background_optimization_analysis,
            str(project_id),
            context_request,
            final_context
        )
        
        processing_time = (time.time() - start_time) * 1000
        logger.info("Context optimization completed",
                   project_id=str(project_id),
                   processing_time_ms=processing_time,
                   core_files=len(response.core_files),
                   supporting_files=len(response.supporting_files))
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Context optimization failed",
                    project_id=str(project_id),
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "OPTIMIZATION_FAILED",
                "message": "Context optimization failed",
                "details": str(e)
            }
        )


# ================== STREAMING CONTEXT ENDPOINT ==================

@router.post(
    "/{project_id}/optimize-stream",
    summary="Stream Optimized Context",
    description="Stream context optimization results progressively",
    response_class=StreamingResponse
)
async def optimize_context_stream(
    project_id: uuid.UUID,
    request: AdvancedContextRequest,
    session: AsyncSession = Depends(get_session),
    context_optimizer: ContextOptimizer = Depends(get_context_optimizer),
    user_id: str = Depends(get_current_user)
):
    """
    Stream context optimization results progressively for large projects.
    
    Returns Server-Sent Events (SSE) stream with:
    - Progress updates during optimization
    - Incremental file relevance results
    - Real-time performance metrics
    - Final optimized context
    """
    async def generate_context_stream():
        try:
            # Initial progress event
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'initialization', 'progress': 0})}\n\n"
            
            # Validate project and get files (same as above)
            project_query = select(ProjectIndex).where(ProjectIndex.id == project_id)
            result = await session.execute(project_query)
            project = result.scalar_one_or_none()
            
            if not project:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Project not found'})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'loading_files', 'progress': 10})}\n\n"
            
            # Get files (abbreviated for streaming example)
            files_query = select(FileEntry).where(FileEntry.project_id == project_id)
            files_result = await session.execute(files_query)
            file_entries = files_result.scalars().all()
            
            if not file_entries:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No files found'})}\n\n"
                return
            
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'analyzing_relevance', 'progress': 30, 'file_count': len(file_entries)})}\n\n"
            
            # Convert to FileAnalysisResult objects
            file_results = []
            for entry in file_entries:
                from ..project_index.models import FileAnalysisResult
                file_result = FileAnalysisResult(
                    file_path=entry.relative_path,
                    file_name=Path(entry.relative_path).name,
                    file_size=entry.file_size,
                    line_count=entry.line_count,
                    file_type=entry.file_type,
                    language=entry.language,
                    file_hash=entry.file_hash,
                    analysis_successful=True,
                    analysis_data=entry.analysis_data or {}
                )
                file_results.append(file_result)
            
            # Create context request
            task_type_map = {
                "feature": TaskType.FEATURE,
                "bugfix": TaskType.BUGFIX,
                "refactoring": TaskType.REFACTORING,
                "analysis": TaskType.ANALYSIS,
                "documentation": TaskType.DOCUMENTATION,
                "testing": TaskType.TESTING,
                "performance": TaskType.PERFORMANCE,
                "security": TaskType.SECURITY
            }
            
            context_request = ContextRequest(
                task_description=request.task_description,
                task_type=task_type_map.get(request.task_type, TaskType.ANALYSIS),
                files_mentioned=request.files_mentioned,
                context_preferences=request.context_preferences.dict(),
                ai_model_info=request.ai_model_info.dict() if request.ai_model_info else {},
                historical_context=request.historical_context.dict()
            )
            
            # Get dependency graph
            from ..project_index.graph import DependencyGraph
            dependency_graph = DependencyGraph()
            
            # Load dependencies into graph
            for entry in file_entries:
                dependency_graph.add_node(entry.relative_path, {
                    "file_type": entry.file_type.value if entry.file_type else "unknown",
                    "language": entry.language
                })
            
            # Use performance optimizer for streaming
            performance_optimizer = await get_performance_optimizer_dep(request)
            relevance_analyzer = RelevanceAnalyzer()
            context_assembler = ContextAssembler()
            
            # Stream optimization results
            async for chunk in performance_optimizer.stream_context_optimization(
                context_request=context_request,
                file_results=file_results,
                relevance_analyzer=relevance_analyzer,
                context_assembler=context_assembler,
                dependency_graph=dependency_graph
            ):
                yield f"data: {chunk.to_json()}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_context_stream(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


# ================== CONTEXT CACHE MANAGEMENT ENDPOINTS ==================

@router.get(
    "/cache/stats",
    response_model=ContextCacheStats,
    summary="Get Context Cache Statistics",
    description="Retrieve detailed context cache performance statistics"
)
async def get_cache_statistics(
    cache_manager: ContextCacheManager = Depends(get_cache_manager),
    user_id: str = Depends(get_current_user)
):
    """Get comprehensive context cache statistics and performance metrics."""
    try:
        stats = cache_manager.get_statistics()
        
        return ContextCacheStats(
            total_requests=stats.total_requests,
            cache_hits=stats.cache_hits,
            cache_misses=stats.cache_misses,
            similarity_hits=stats.similarity_hits,
            hit_rate=stats.hit_rate,
            entries_cached=stats.entries_created,
            avg_quality_score=stats.avg_quality_score
        )
        
    except Exception as e:
        logger.error("Failed to get cache statistics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "CACHE_STATS_FAILED",
                "message": "Failed to retrieve cache statistics"
            }
        )


@router.delete(
    "/cache/{project_id}",
    summary="Invalidate Project Cache",
    description="Invalidate all cached contexts for a project"
)
async def invalidate_project_cache(
    project_id: uuid.UUID,
    cache_manager: ContextCacheManager = Depends(get_cache_manager),
    user_id: str = Depends(get_current_user)
):
    """Invalidate all cached context optimizations for a specific project."""
    try:
        invalidated_count = await cache_manager.invalidate_cache(
            project_id=str(project_id)
        )
        
        return {
            "message": f"Invalidated {invalidated_count} cache entries",
            "project_id": str(project_id),
            "invalidated_count": invalidated_count
        }
        
    except Exception as e:
        logger.error("Cache invalidation failed",
                    project_id=str(project_id),
                    error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "CACHE_INVALIDATION_FAILED",
                "message": "Failed to invalidate cache"
            }
        )


@router.post(
    "/cache/optimize",
    summary="Optimize Context Cache",
    description="Optimize cache performance by removing stale entries"
)
async def optimize_cache(
    cache_manager: ContextCacheManager = Depends(get_cache_manager),
    user_id: str = Depends(get_current_user)
):
    """Optimize context cache by removing expired and low-quality entries."""
    try:
        optimization_results = await cache_manager.optimize_cache()
        
        return {
            "message": "Cache optimization completed",
            "results": optimization_results
        }
        
    except Exception as e:
        logger.error("Cache optimization failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "CACHE_OPTIMIZATION_FAILED",
                "message": "Failed to optimize cache"
            }
        )


# ================== PERFORMANCE MONITORING ENDPOINTS ==================

@router.get(
    "/performance/metrics",
    summary="Get Performance Metrics",
    description="Retrieve current context optimization performance metrics"
)
async def get_performance_metrics(
    user_id: str = Depends(get_current_user)
):
    """Get current performance metrics from the optimization engine."""
    try:
        performance_optimizer = get_performance_optimizer()
        metrics = performance_optimizer.get_performance_metrics()
        
        return {
            "metrics": metrics.to_dict(),
            "system_info": {
                "total_memory_mb": psutil.virtual_memory().total / (1024 * 1024),
                "available_memory_mb": psutil.virtual_memory().available / (1024 * 1024),
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to get performance metrics", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "METRICS_FAILED",
                "message": "Failed to retrieve performance metrics"
            }
        )


@router.post(
    "/performance/optimize",
    summary="Optimize Performance Settings",
    description="Optimize performance settings based on current system state"
)
async def optimize_performance_settings(
    user_id: str = Depends(get_current_user)
):
    """Optimize performance settings based on system capabilities."""
    try:
        # Get system capabilities
        total_memory = psutil.virtual_memory().total / (1024 * 1024)
        cpu_count = psutil.cpu_count(logical=False)
        
        # Create optimized resource limits
        optimized_limits = ResourceLimits(
            max_memory_mb=int(total_memory * 0.4),  # Use 40% of system memory
            max_parallel_tasks=min(max(cpu_count, 6), 20),  # 6-20 tasks based on CPU
            max_batch_size=max(100, cpu_count * 15),
            memory_threshold_mb=int(total_memory * 0.3)
        )
        
        return {
            "message": "Performance settings optimized",
            "optimized_settings": {
                "max_memory_mb": optimized_limits.max_memory_mb,
                "max_parallel_tasks": optimized_limits.max_parallel_tasks,
                "max_batch_size": optimized_limits.max_batch_size,
                "memory_threshold_mb": optimized_limits.memory_threshold_mb
            },
            "system_detected": {
                "total_memory_mb": total_memory,
                "cpu_count": cpu_count,
                "logical_cpu_count": psutil.cpu_count(logical=True)
            }
        }
        
    except Exception as e:
        logger.error("Performance optimization failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "error": "PERFORMANCE_OPTIMIZATION_FAILED",
                "message": "Failed to optimize performance settings"
            }
        )


# ================== HELPER FUNCTIONS ==================

async def _convert_context_to_response(
    context: Union[OptimizedContext, Any],
    start_time: float,
    performance_metrics: Optional[PerformanceMetrics] = None
) -> OptimizedContextResponse:
    """Convert optimized context to API response format."""
    try:
        # Convert core files
        core_files = []
        if hasattr(context, 'core_files'):
            for file_score in context.core_files:
                core_files.append(FileRelevanceInfo(
                    file_path=file_score.file_path,
                    relevance_score=file_score.relevance_score,
                    relevance_reasons=file_score.relevance_reasons,
                    content_summary=file_score.content_summary,
                    key_functions=file_score.key_functions,
                    key_classes=file_score.key_classes,
                    import_relationships=file_score.import_relationships,
                    estimated_tokens=file_score.estimated_tokens
                ))
        
        # Convert supporting files
        supporting_files = []
        if hasattr(context, 'supporting_files'):
            for file_score in context.supporting_files:
                supporting_files.append(FileRelevanceInfo(
                    file_path=file_score.file_path,
                    relevance_score=file_score.relevance_score,
                    relevance_reasons=file_score.relevance_reasons,
                    content_summary=file_score.content_summary,
                    key_functions=file_score.key_functions,
                    key_classes=file_score.key_classes,
                    import_relationships=file_score.import_relationships,
                    estimated_tokens=file_score.estimated_tokens
                ))
        
        # Convert dependency graph
        dependency_graph = DependencyGraphResponse(
            nodes=context.dependency_graph.get('nodes', []),
            edges=context.dependency_graph.get('edges', [])
        )
        
        # Convert context summary
        summary_data = context.context_summary
        context_summary = ContextSummary(
            total_files=summary_data.get('total_files', 0),
            total_tokens=summary_data.get('total_tokens', 0),
            coverage_percentage=summary_data.get('coverage_percentage', 0.0),
            confidence_score=summary_data.get('confidence_score', 0.0),
            architectural_patterns=summary_data.get('architectural_patterns', []),
            potential_challenges=summary_data.get('potential_challenges', []),
            recommended_approach=summary_data.get('recommended_approach', '')
        )
        
        # Convert optimization metadata
        metadata = context.optimization_metadata
        processing_time = int((time.time() - start_time) * 1000)
        
        # Include performance metrics if available
        perf_metrics_dict = None
        if performance_metrics:
            perf_metrics_dict = performance_metrics.to_dict()
        
        optimization_metadata = OptimizationMetadata(
            algorithm_used=metadata.get('algorithm_used', 'hybrid_ai_enhanced'),
            processing_time_ms=processing_time,
            cache_hit_rate=metadata.get('cache_hit_rate', 0.0),
            relevance_distribution=metadata.get('relevance_distribution', {"high": 0, "medium": 0, "low": 0}),
            performance_metrics=perf_metrics_dict
        )
        
        # Convert suggestions
        suggestions_data = context.suggestions
        suggestions = Suggestions(
            additional_files=suggestions_data.get('additional_files', []),
            alternative_contexts=suggestions_data.get('alternative_contexts', []),
            optimization_tips=suggestions_data.get('optimization_tips', [])
        )
        
        return OptimizedContextResponse(
            core_files=core_files,
            supporting_files=supporting_files,
            dependency_graph=dependency_graph,
            context_summary=context_summary,
            optimization_metadata=optimization_metadata,
            suggestions=suggestions
        )
        
    except Exception as e:
        logger.error("Context response conversion failed", error=str(e))
        raise


async def _convert_cached_result_to_response(cached_result: Any) -> OptimizedContextResponse:
    """Convert cached result to response format."""
    # This would be similar to _convert_context_to_response but handle cached data
    # For now, return a simplified response
    return OptimizedContextResponse(
        core_files=[],
        supporting_files=[],
        dependency_graph=DependencyGraphResponse(nodes=[], edges=[]),
        context_summary=ContextSummary(
            total_files=0,
            total_tokens=0,
            coverage_percentage=0.0,
            confidence_score=0.0,
            architectural_patterns=[],
            potential_challenges=[],
            recommended_approach="Cached result - details omitted"
        ),
        optimization_metadata=OptimizationMetadata(
            algorithm_used="cached",
            processing_time_ms=1,
            cache_hit_rate=1.0,
            relevance_distribution={"high": 0, "medium": 0, "low": 0}
        ),
        suggestions=Suggestions(
            additional_files=[],
            alternative_contexts=[],
            optimization_tips=["Using cached result for improved performance"]
        )
    )


async def _background_optimization_analysis(
    project_id: str,
    context_request: ContextRequest,
    optimization_result: Any
):
    """Background task for optimization analysis and improvement."""
    try:
        # Log optimization metrics for analysis
        logger.info("Background optimization analysis",
                   project_id=project_id,
                   task_type=context_request.task_type.value,
                   result_quality="analysis_placeholder")
        
        # This could include:
        # - Performance metric collection
        # - Quality assessment
        # - User feedback integration
        # - Model retraining triggers
        
    except Exception as e:
        logger.error("Background optimization analysis failed",
                    project_id=project_id,
                    error=str(e))