"""
Context management API endpoints.

Provides full CRUD operations for contexts with semantic search,
compression, and cross-agent knowledge sharing capabilities.
"""

import uuid
import time
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.context_manager import ContextManager, get_context_manager
from ...core.context_compression import CompressionLevel
from ...core.vector_search_engine import (
    VectorSearchEngine, 
    create_vector_search_engine,
    SearchConfiguration,
    BatchSearchRequest,
    BatchSearchResponse
)
from ...models.context import Context, ContextType
from ...schemas.context import (
    ContextCreate, 
    ContextUpdate, 
    ContextResponse, 
    ContextListResponse,
    ContextSearchRequest
)
from ...core.database import get_db


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/",
    response_model=ContextResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Store new context with automatic embedding generation"
)
async def create_context(
    context_data: ContextCreate,
    auto_embed: bool = Query(True, description="Generate embeddings automatically"),
    auto_compress: bool = Query(False, description="Auto-compress long content"),
    context_manager: ContextManager = Depends(get_context_manager)
) -> ContextResponse:
    """
    Store new context with automatic embedding generation.
    
    Features:
    - Automatic vector embedding generation for semantic search
    - Optional compression for long content (>10k characters)
    - Importance scoring and metadata handling
    - Cross-agent knowledge sharing setup
    
    Args:
        context_data: Context information to store
        auto_embed: Whether to generate embeddings automatically
        auto_compress: Whether to compress long content automatically
        
    Returns:
        Stored context with generated embedding and metadata
        
    Raises:
        400: Invalid context data
        500: Storage or embedding generation failed
    """
    try:
        logger.info(f"Creating context: {context_data.title}")
        
        # Validate context data
        if not context_data.title.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Context title cannot be empty"
            )
        
        if not context_data.content.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Context content cannot be empty"
            )
        
        # Store context
        stored_context = await context_manager.store_context(
            context_data=context_data,
            auto_embed=auto_embed,
            auto_compress=auto_compress
        )
        
        logger.info(f"Successfully created context: {stored_context.id}")
        
        # Convert to response schema
        return ContextResponse.model_validate(stored_context)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to store context: {str(e)}"
        )


@router.get(
    "/{context_id}",
    response_model=ContextResponse,
    summary="Retrieve specific context by ID"
)
async def get_context(
    context_id: uuid.UUID = Path(..., description="Context ID to retrieve"),
    context_manager: ContextManager = Depends(get_context_manager)
) -> ContextResponse:
    """
    Retrieve specific context by ID.
    
    Automatically marks context as accessed for relevance tracking.
    
    Args:
        context_id: Unique context identifier
        
    Returns:
        Context data with current relevance score
        
    Raises:
        404: Context not found
        500: Retrieval failed
    """
    try:
        context = await context_manager.retrieve_context(context_id)
        
        if not context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Context {context_id} not found"
            )
        
        return ContextResponse.model_validate(context)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve context {context_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve context: {str(e)}"
        )


@router.put(
    "/{context_id}",
    response_model=ContextResponse,
    summary="Update existing context"
)
async def update_context(
    context_id: uuid.UUID = Path(..., description="Context ID to update"),
    updates: ContextUpdate = ...,
    regenerate_embedding: bool = Query(False, description="Regenerate embedding after update"),
    context_manager: ContextManager = Depends(get_context_manager)
) -> ContextResponse:
    """
    Update existing context.
    
    Can update title, content, importance score, tags, and metadata.
    Optionally regenerates embeddings for updated content.
    
    Args:
        context_id: Context to update
        updates: Update data
        regenerate_embedding: Whether to regenerate embeddings
        
    Returns:
        Updated context data
        
    Raises:
        404: Context not found
        400: Invalid update data
        500: Update failed
    """
    try:
        updated_context = await context_manager.update_context(
            context_id=context_id,
            updates=updates,
            regenerate_embedding=regenerate_embedding
        )
        
        if not updated_context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Context {context_id} not found"
            )
        
        logger.info(f"Updated context: {context_id}")
        return ContextResponse.model_validate(updated_context)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update context {context_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update context: {str(e)}"
        )


@router.delete(
    "/{context_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete (archive) context"
)
async def delete_context(
    context_id: uuid.UUID = Path(..., description="Context ID to delete"),
    context_manager: ContextManager = Depends(get_context_manager)
) -> None:
    """
    Delete context (soft delete by archiving).
    
    Contexts are archived rather than hard deleted to preserve
    relationships and enable recovery if needed.
    
    Args:
        context_id: Context to delete
        
    Raises:
        404: Context not found
        500: Deletion failed
    """
    try:
        deleted = await context_manager.delete_context(context_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Context {context_id} not found"
            )
        
        logger.info(f"Deleted context: {context_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete context {context_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete context: {str(e)}"
        )


@router.post(
    "/search",
    response_model=List[Dict[str, Any]],
    summary="Semantic search across contexts"
)
async def search_contexts(
    search_request: ContextSearchRequest,
    include_cross_agent: bool = Query(True, description="Include other agents' contexts"),
    context_manager: ContextManager = Depends(get_context_manager)
) -> List[Dict[str, Any]]:
    """
    Perform semantic search across contexts.
    
    Uses vector embeddings for semantic similarity matching with
    advanced filtering and relevance scoring.
    
    Args:
        search_request: Search parameters and filters
        include_cross_agent: Whether to search other agents' contexts
        
    Returns:
        List of matching contexts with similarity scores
        
    Raises:
        400: Invalid search parameters
        500: Search failed
    """
    try:
        if not search_request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search query cannot be empty"
            )
        
        logger.info(f"Searching contexts: {search_request.query[:50]}...")
        
        # Perform search
        matches = await context_manager.search_contexts(
            request=search_request,
            include_cross_agent=include_cross_agent
        )
        
        # Convert to response format
        results = []
        for match in matches:
            result = {
                "context": ContextResponse.model_validate(match.context).model_dump(),
                "similarity_score": match.similarity_score,
                "relevance_score": match.relevance_score,
                "rank": match.rank
            }
            results.append(result)
        
        logger.info(f"Search returned {len(results)} results")
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post(
    "/{context_id}/compress",
    response_model=ContextResponse,
    summary="Compress context to reduce token usage"
)
async def compress_context(
    context_id: uuid.UUID = Path(..., description="Context ID to compress"),
    compression_level: CompressionLevel = Query(
        CompressionLevel.STANDARD,
        description="Compression level to apply"
    ),
    preserve_original: bool = Query(True, description="Preserve original content in metadata"),
    context_manager: ContextManager = Depends(get_context_manager)
) -> ContextResponse:
    """
    Intelligently compress context while preserving key information.
    
    Uses Claude to summarize content while preserving:
    - Key decisions and insights
    - Important patterns and learnings
    - Critical technical details
    
    Args:
        context_id: Context to compress
        compression_level: Level of compression (light/standard/aggressive)
        preserve_original: Whether to preserve original content
        
    Returns:
        Compressed context with metadata about compression
        
    Raises:
        404: Context not found
        400: Context already compressed or invalid compression level
        500: Compression failed
    """
    try:
        compressed_context = await context_manager.compress_context(
            context_id=context_id,
            compression_level=compression_level,
            preserve_original=preserve_original
        )
        
        if not compressed_context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Context {context_id} not found"
            )
        
        logger.info(f"Compressed context: {context_id}")
        return ContextResponse.model_validate(compressed_context)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to compress context {context_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compress context: {str(e)}"
        )


@router.get(
    "/{context_id}/similar",
    response_model=List[Dict[str, Any]],
    summary="Find contexts similar to a given context"
)
async def find_similar_contexts(
    context_id: uuid.UUID = Path(..., description="Reference context ID"),
    limit: int = Query(5, ge=1, le=20, description="Maximum number of similar contexts"),
    similarity_threshold: float = Query(0.7, ge=0.0, le=1.0, description="Minimum similarity score"),
    context_manager: ContextManager = Depends(get_context_manager)
) -> List[Dict[str, Any]]:
    """
    Find contexts similar to a given context.
    
    Uses vector similarity to find related contexts that might
    contain relevant information or patterns.
    
    Args:
        context_id: Reference context to find similarities for
        limit: Maximum number of similar contexts
        similarity_threshold: Minimum similarity score
        
    Returns:
        List of similar contexts with similarity scores
        
    Raises:
        404: Context not found
        500: Search failed
    """
    try:
        # First verify the context exists
        reference_context = await context_manager.retrieve_context(context_id)
        if not reference_context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Context {context_id} not found"
            )
        
        # Find similar contexts using search engine
        search_engine = await context_manager._ensure_search_engine()
        similar_matches = await search_engine.find_similar_contexts(
            context_id=context_id,
            limit=limit,
            similarity_threshold=similarity_threshold
        )
        
        # Convert to response format
        results = []
        for match in similar_matches:
            result = {
                "context": ContextResponse.model_validate(match.context).model_dump(),
                "similarity_score": match.similarity_score,
                "rank": match.rank
            }
            results.append(result)
        
        logger.info(f"Found {len(results)} similar contexts for {context_id}")
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to find similar contexts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find similar contexts: {str(e)}"
        )


@router.get(
    "/analytics/summary",
    response_model=Dict[str, Any],
    summary="Get context usage analytics"
)
async def get_context_analytics(
    agent_id: Optional[uuid.UUID] = Query(None, description="Filter by agent"),
    session_id: Optional[uuid.UUID] = Query(None, description="Filter by session"),
    context_manager: ContextManager = Depends(get_context_manager)
) -> Dict[str, Any]:
    """
    Get comprehensive context usage analytics.
    
    Provides insights into:
    - Context storage and retrieval patterns
    - Compression effectiveness
    - Search performance
    - Cross-agent knowledge sharing
    
    Args:
        agent_id: Filter analytics by specific agent
        session_id: Filter analytics by specific session
        
    Returns:
        Analytics data with performance metrics
    """
    try:
        analytics = await context_manager.get_context_analytics(
            agent_id=agent_id,
            session_id=session_id
        )
        
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to get context analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get analytics: {str(e)}"
        )


@router.post(
    "/maintenance/consolidate",
    response_model=Dict[str, Any],
    summary="Consolidate stale contexts"
)
async def consolidate_stale_contexts(
    agent_id: Optional[uuid.UUID] = Query(None, description="Specific agent to consolidate"),
    batch_size: int = Query(10, ge=1, le=50, description="Number of contexts to process"),
    context_manager: ContextManager = Depends(get_context_manager)
) -> Dict[str, Any]:
    """
    Automatically consolidate old, frequently accessed contexts.
    
    Compresses contexts that are:
    - Frequently accessed (5+ times)
    - High importance (>0.8 score)
    - Not already consolidated
    
    Args:
        agent_id: Specific agent to consolidate contexts for
        batch_size: Number of contexts to process in this batch
        
    Returns:
        Summary of consolidation results
    """
    try:
        consolidated_count = await context_manager.consolidate_stale_contexts(
            agent_id=agent_id,
            batch_size=batch_size
        )
        
        return {
            "consolidated_count": consolidated_count,
            "batch_size": batch_size,
            "agent_id": str(agent_id) if agent_id else None
        }
        
    except Exception as e:
        logger.error(f"Failed to consolidate contexts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to consolidate contexts: {str(e)}"
        )


@router.post(
    "/maintenance/cleanup",
    response_model=Dict[str, Any],
    summary="Clean up old, low-importance contexts"
)
async def cleanup_old_contexts(
    max_age_days: int = Query(90, ge=1, le=365, description="Maximum age in days"),
    min_importance_threshold: float = Query(0.3, ge=0.0, le=1.0, description="Minimum importance to preserve"),
    context_manager: ContextManager = Depends(get_context_manager)
) -> Dict[str, Any]:
    """
    Clean up old, low-importance contexts.
    
    Archives contexts that are:
    - Older than specified age
    - Below importance threshold
    - Not frequently accessed
    
    Args:
        max_age_days: Maximum age for contexts to keep
        min_importance_threshold: Minimum importance to preserve old contexts
        
    Returns:
        Summary of cleanup results
    """
    try:
        cleaned_count = await context_manager.cleanup_old_contexts(
            max_age_days=max_age_days,
            min_importance_threshold=min_importance_threshold
        )
        
        return {
            "cleaned_count": cleaned_count,
            "max_age_days": max_age_days,
            "min_importance_threshold": min_importance_threshold
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup contexts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup contexts: {str(e)}"
        )


@router.get(
    "/health",
    response_model=Dict[str, Any],
    summary="Context Engine health check"
)
async def health_check(
    context_manager: ContextManager = Depends(get_context_manager)
) -> Dict[str, Any]:
    """
    Perform comprehensive health check on Context Engine.
    
    Checks:
    - Database connectivity
    - Embedding service status
    - Compression service status
    - Search engine performance
    
    Returns:
        Health status for all components
    """
    try:
        health_status = await context_manager.health_check()
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "overall_status": "unhealthy",
            "error": str(e)
        }


# New Vector Search Engine endpoints

@router.post(
    "/search/semantic",
    response_model=List[Dict[str, Any]],
    summary="Advanced semantic search with configurable thresholds"
)
async def semantic_search_contexts(
    query: str = Query(..., description="Search query text", min_length=1),
    agent_id: Optional[uuid.UUID] = Query(None, description="Agent performing search"),
    limit: int = Query(10, ge=1, le=50, description="Maximum results to return"),
    similarity_threshold: Optional[float] = Query(0.7, ge=0.0, le=1.0, description="Minimum similarity score"),
    include_cross_agent: bool = Query(True, description="Include contexts from other agents"),
    context_types: Optional[List[ContextType]] = Query(None, description="Filter by context types"),
    min_importance: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum importance score"),
    max_age_days: Optional[int] = Query(None, ge=1, description="Maximum age in days"),
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Perform advanced semantic search using the Vector Search Engine.
    
    Features:
    - Configurable similarity thresholds for precision control
    - Cross-agent knowledge discovery with privacy controls
    - Advanced filtering by context type, importance, and age
    - Performance optimized for <50ms response times
    - Relevance scoring with multiple factors
    
    Args:
        query: Natural language search query
        agent_id: Agent ID for access control and personalization
        limit: Maximum number of results
        similarity_threshold: Minimum semantic similarity score (0.0-1.0)
        include_cross_agent: Whether to search other agents' contexts
        context_types: Filter by specific context types
        min_importance: Minimum importance score filter
        max_age_days: Maximum age of contexts in days
        
    Returns:
        List of context matches with similarity and relevance scores
        
    Raises:
        400: Invalid search parameters
        500: Search engine failure
    """
    try:
        # Create search engine
        search_engine = await create_vector_search_engine(db)
        
        # Build search filters
        from ...core.enhanced_vector_search import SearchFilters
        filters = SearchFilters(
            context_types=context_types,
            min_similarity=similarity_threshold or 0.7,
            min_importance=min_importance or 0.0,
            max_age_days=max_age_days
        )
        
        # Perform semantic search
        start_time = time.time()
        results = await search_engine.semantic_search(
            query=query,
            agent_id=agent_id,
            limit=limit,
            similarity_threshold=similarity_threshold,
            include_cross_agent=include_cross_agent,
            filters=filters
        )
        search_time_ms = (time.time() - start_time) * 1000
        
        # Convert to response format
        response_data = []
        for match in results:
            result = {
                "context": ContextResponse.model_validate(match.context).model_dump(),
                "similarity_score": round(match.similarity_score, 4),
                "relevance_score": round(match.relevance_score, 4),
                "rank": match.rank,
                "metadata": {
                    "search_time_ms": round(search_time_ms, 2),
                    "search_method": "semantic_vector_search"
                }
            }
            response_data.append(result)
        
        logger.info(
            f"Semantic search completed: {len(results)} results in {search_time_ms:.1f}ms for query: {query[:50]}..."
        )
        
        return response_data
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(e)}"
        )


@router.post(
    "/search/batch",
    response_model=Dict[str, Any],
    summary="Batch semantic search for multiple queries"
)
async def batch_semantic_search(
    queries: List[str] = Query(..., description="List of search queries", min_items=1, max_items=50),
    agent_id: Optional[uuid.UUID] = Query(None, description="Agent performing search"),
    limit: int = Query(10, ge=1, le=20, description="Maximum results per query"),
    include_cross_agent: bool = Query(True, description="Include cross-agent contexts"),
    similarity_threshold: float = Query(0.7, ge=0.0, le=1.0, description="Minimum similarity score"),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Perform batch semantic search for multiple queries efficiently.
    
    Optimized for processing multiple related queries with:
    - Concurrent processing for improved performance
    - Shared caching across queries
    - Bulk embedding generation
    - Performance analytics
    
    Args:
        queries: List of search queries to process
        agent_id: Agent ID for access control
        limit: Maximum results per query
        include_cross_agent: Include other agents' contexts
        similarity_threshold: Minimum similarity score
        
    Returns:
        Dictionary mapping queries to their results plus performance metrics
        
    Raises:
        400: Invalid batch request
        500: Batch search failure
    """
    try:
        # Validate batch size
        if len(queries) > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size cannot exceed 50 queries"
            )
        
        # Create search engine and batch request
        search_engine = await create_vector_search_engine(db)
        
        from ...core.enhanced_vector_search import SearchFilters
        filters = SearchFilters(min_similarity=similarity_threshold)
        
        batch_request = BatchSearchRequest(
            queries=queries,
            agent_id=agent_id,
            filters=filters,
            limit=limit,
            include_cross_agent=include_cross_agent
        )
        
        # Execute batch search
        start_time = time.time()
        batch_response = await search_engine.batch_search(batch_request)
        total_time_ms = (time.time() - start_time) * 1000
        
        # Format response
        formatted_results = {}
        for query, matches in batch_response.results.items():
            formatted_matches = []
            for match in matches:
                formatted_match = {
                    "context": ContextResponse.model_validate(match.context).model_dump(),
                    "similarity_score": round(match.similarity_score, 4),
                    "relevance_score": round(match.relevance_score, 4),
                    "rank": match.rank
                }
                formatted_matches.append(formatted_match)
            formatted_results[query] = formatted_matches
        
        response = {
            "results": formatted_results,
            "metadata": {
                **batch_response.metadata,
                "total_time_ms": round(total_time_ms, 2)
            },
            "performance_metrics": batch_response.performance_metrics
        }
        
        logger.info(
            f"Batch search completed: {len(queries)} queries, "
            f"{batch_response.performance_metrics['total_results']} total results in {total_time_ms:.1f}ms"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch search failed: {str(e)}"
        )


@router.post(
    "/search/cross-agent",
    response_model=List[Dict[str, Any]],
    summary="Cross-agent knowledge discovery search"
)
async def cross_agent_knowledge_search(
    query: str = Query(..., description="Search query", min_length=1),
    requesting_agent_id: uuid.UUID = Query(..., description="Agent requesting the search"),
    limit: int = Query(10, ge=1, le=20, description="Maximum results"),
    min_importance: float = Query(0.7, ge=0.0, le=1.0, description="Minimum importance for sharing"),
    similarity_threshold: float = Query(0.8, ge=0.0, le=1.0, description="Minimum similarity for cross-agent sharing"),
    db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Search for knowledge across all agents with privacy controls.
    
    Enables knowledge discovery and sharing between agents while respecting:
    - Importance thresholds for quality control
    - Higher similarity thresholds for cross-agent sharing
    - Access level controls and privacy boundaries
    
    Args:
        query: Search query for knowledge discovery
        requesting_agent_id: Agent making the request
        limit: Maximum results to return
        min_importance: Minimum importance score for sharing
        similarity_threshold: Higher threshold for cross-agent sharing
        
    Returns:
        List of contexts from other agents with high relevance
        
    Raises:
        400: Invalid search parameters
        500: Cross-agent search failure
    """
    try:
        # Create search engine
        search_engine = await create_vector_search_engine(db)
        
        # Perform cross-agent search
        start_time = time.time()
        results = await search_engine.cross_agent_search(
            query=query,
            requesting_agent_id=requesting_agent_id,
            limit=limit,
            min_importance=min_importance
        )
        search_time_ms = (time.time() - start_time) * 1000
        
        # Format response with cross-agent metadata
        response_data = []
        for match in results:
            result = {
                "context": ContextResponse.model_validate(match.context).model_dump(),
                "similarity_score": round(match.similarity_score, 4),
                "relevance_score": round(match.relevance_score, 4),
                "rank": match.rank,
                "cross_agent_metadata": {
                    "context_owner_agent_id": str(match.context.agent_id),
                    "sharing_score": round(match.relevance_score * match.context.importance_score, 4),
                    "knowledge_type": match.context.context_type.value if match.context.context_type else None
                },
                "search_metadata": {
                    "search_time_ms": round(search_time_ms, 2),
                    "min_importance_threshold": min_importance,
                    "similarity_threshold": similarity_threshold
                }
            }
            response_data.append(result)
        
        logger.info(
            f"Cross-agent search completed: {len(results)} shared contexts in {search_time_ms:.1f}ms "
            f"for agent {requesting_agent_id}"
        )
        
        return response_data
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Cross-agent search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cross-agent search failed: {str(e)}"
        )


@router.post(
    "/bulk/index",
    response_model=Dict[str, Any],
    summary="Bulk index contexts with embeddings"
)
async def bulk_index_contexts(
    context_ids: List[uuid.UUID] = Query(..., description="List of context IDs to index", min_items=1, max_items=100),
    batch_size: int = Query(20, ge=1, le=50, description="Batch size for processing"),
    regenerate_existing: bool = Query(False, description="Regenerate embeddings for contexts that already have them"),
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Bulk index multiple contexts with vector embeddings.
    
    Efficiently processes multiple contexts for semantic search:
    - Batch embedding generation for performance
    - Concurrent processing within batches
    - Progress tracking and error handling
    - Statistics on indexing success/failure
    
    Args:
        context_ids: List of context IDs to process
        batch_size: Number of contexts per batch
        regenerate_existing: Whether to regenerate existing embeddings
        
    Returns:
        Indexing statistics and performance metrics
        
    Raises:
        400: Invalid context IDs or batch parameters
        500: Bulk indexing failure
    """
    try:
        # Validate batch size
        if len(context_ids) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot index more than 100 contexts at once"
            )
        
        # Create search engine
        search_engine = await create_vector_search_engine(db)
        
        # Filter contexts that need indexing if not regenerating
        if not regenerate_existing:
            from sqlalchemy import select
            result = await db.execute(
                select(Context.id).where(
                    Context.id.in_(context_ids),
                    Context.embedding.is_(None)
                )
            )
            context_ids = [row[0] for row in result]
        
        # Perform bulk indexing
        start_time = time.time()
        stats = await search_engine.bulk_index_contexts(
            context_ids=context_ids,
            batch_size=batch_size
        )
        total_time_s = time.time() - start_time
        
        # Add performance metrics
        stats.update({
            "total_time_seconds": round(total_time_s, 2),
            "contexts_per_second": round(stats['successfully_indexed'] / max(1, total_time_s), 2),
            "batch_size": batch_size,
            "regenerate_existing": regenerate_existing
        })
        
        logger.info(
            f"Bulk indexing completed: {stats['successfully_indexed']}/{stats['total_contexts']} "
            f"contexts indexed in {total_time_s:.1f}s"
        )
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk indexing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bulk indexing failed: {str(e)}"
        )


@router.get(
    "/search/performance",
    response_model=Dict[str, Any],
    summary="Get search engine performance metrics"
)
async def get_search_performance_metrics(
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get comprehensive performance metrics for the Vector Search Engine.
    
    Provides insights into:
    - Search performance and response times
    - Cache hit rates and efficiency
    - Cross-agent sharing statistics
    - Embedding service performance
    - System health indicators
    
    Returns:
        Comprehensive performance and health metrics
    """
    try:
        # Create search engine and get metrics
        search_engine = await create_vector_search_engine(db)
        metrics = search_engine.get_performance_metrics()
        
        # Add timestamp
        metrics['timestamp'] = datetime.utcnow().isoformat()
        metrics['collection_time'] = time.time()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )


@router.get(
    "/search/health",
    response_model=Dict[str, Any],
    summary="Vector Search Engine health check"
)
async def search_engine_health_check(
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Perform comprehensive health check on the Vector Search Engine.
    
    Validates:
    - Database connectivity and pgvector availability
    - OpenAI Embedding Service status
    - Redis cache connectivity
    - Search performance with test queries
    - Index health and optimization status
    
    Returns:
        Detailed health status for all components
    """
    try:
        # Create search engine and perform health check
        search_engine = await create_vector_search_engine(db)
        health_status = await search_engine.health_check()
        
        return health_status
        
    except Exception as e:
        logger.error(f"Search engine health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }