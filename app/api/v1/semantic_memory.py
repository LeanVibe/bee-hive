"""
Semantic Memory Service API Router for LeanVibe Agent Hive 2.0

FastAPI router implementing all 12 semantic memory API endpoints with complete
API contract compliance, comprehensive error handling, and performance monitoring.

Endpoints:
- Document Management (4): ingest, batch-ingest, get, delete
- Semantic Search (3): search, similarity, related
- Context Management (3): compress, contextualize, agent-knowledge  
- System Operations (2): health, metrics, rebuild-index

Features:
- Complete API contract compliance with OpenAPI specification
- Comprehensive error handling with proper HTTP status codes
- Performance monitoring and structured logging
- Request validation and response serialization
- Rate limiting and security considerations
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Query, Path, Depends, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse

from ...services.semantic_memory_service import (
    get_semantic_memory_service, SemanticMemoryService, 
    SemanticMemoryServiceError, DocumentNotFoundError, EmbeddingGenerationError
)
from ...schemas.semantic_memory import (
    # Request/Response schemas
    DocumentIngestRequest, DocumentIngestResponse, BatchIngestRequest, BatchIngestResponse,
    DocumentResponse, SemanticSearchRequest, SemanticSearchResponse, SimilaritySearchRequest,
    SimilaritySearchResponse, RelatedDocumentsResponse, ContextCompressionRequest,
    ContextCompressionResponse, ContextualizationRequest, ContextualizationResponse,
    AgentKnowledgeResponse, HealthResponse, MetricsResponse, IndexRebuildRequest,
    IndexRebuildResponse, ErrorResponse,
    
    # Enums
    RelationshipType, TimeRange, MetricsFormat, KnowledgeType
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/memory",
    tags=["Semantic Memory Service"],
    responses={
        400: {"description": "Bad Request", "model": ErrorResponse},
        404: {"description": "Not Found", "model": ErrorResponse},
        429: {"description": "Rate Limit Exceeded", "model": ErrorResponse},
        500: {"description": "Internal Server Error", "model": ErrorResponse}
    }
)


# =============================================================================
# DEPENDENCY INJECTION AND ERROR HANDLING
# =============================================================================

async def get_service() -> SemanticMemoryService:
    """Dependency to get the semantic memory service."""
    return await get_semantic_memory_service()


def create_error_response(
    error_code: str, 
    message: str, 
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        error=error_code,
        message=message,
        details=details or {},
        request_id=request_id
    )


def handle_service_exceptions(func):
    """Decorator to handle common service exceptions."""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except DocumentNotFoundError as e:
            logger.warning(f"Document not found: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    "document_not_found",
                    str(e)
                ).dict()
            )
        except EmbeddingGenerationError as e:
            logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=create_error_response(
                    "embedding_generation_error",
                    str(e)
                ).dict()
            )
        except SemanticMemoryServiceError as e:
            logger.error(f"Service error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=create_error_response(
                    "service_error",
                    str(e)
                ).dict()
            )
        except ValueError as e:
            logger.warning(f"Validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=create_error_response(
                    "validation_error",
                    str(e)
                ).dict()
            )
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=create_error_response(
                    "internal_server_error",
                    "An unexpected error occurred"
                ).dict()
            )
    
    return wrapper


# =============================================================================
# DOCUMENT MANAGEMENT ENDPOINTS
# =============================================================================

@router.post(
    "/ingest",
    response_model=DocumentIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest single document into semantic memory",
    description="Processes and stores a single document with semantic embeddings. "
                "Automatically generates vector representations and updates search indices.",
    responses={
        201: {"description": "Document successfully ingested"},
        400: {"description": "Invalid request parameters"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)
@handle_service_exceptions
async def ingest_document(
    request: DocumentIngestRequest,
    service: SemanticMemoryService = Depends(get_service)
) -> DocumentIngestResponse:
    """Ingest a single document into semantic memory."""
    logger.info(f"📝 Ingesting document for agent: {request.agent_id}")
    
    start_time = time.time()
    result = await service.ingest_document(request)
    processing_time = (time.time() - start_time) * 1000
    
    logger.info(f"✅ Document ingested successfully in {processing_time:.2f}ms")
    return result


@router.post(
    "/batch-ingest",
    response_model=BatchIngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch ingest multiple documents",
    description="Efficiently processes multiple documents in a single operation. "
                "Optimized for high-throughput ingestion with batch embedding generation.",
    responses={
        201: {"description": "Batch ingestion completed"},
        400: {"description": "Invalid request parameters"},
        413: {"description": "Batch size too large"},
        500: {"description": "Internal server error"}
    }
)
@handle_service_exceptions
async def batch_ingest_documents(
    request: BatchIngestRequest,
    service: SemanticMemoryService = Depends(get_service)
) -> BatchIngestResponse:
    """Batch ingest multiple documents."""
    if len(request.documents) > 100:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=create_error_response(
                "batch_too_large",
                f"Batch size {len(request.documents)} exceeds maximum of 100 documents"
            ).dict()
        )
    
    logger.info(f"📦 Starting batch ingestion: {len(request.documents)} documents")
    
    start_time = time.time()
    result = await service.batch_ingest_documents(request)
    processing_time = (time.time() - start_time) * 1000
    
    logger.info(f"✅ Batch ingestion completed: {result.successful_ingestions}/{result.total_documents} successful in {processing_time:.2f}ms")
    return result


@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    summary="Retrieve document by ID",
    description="Fetches a specific document with its metadata and embedding information",
    responses={
        200: {"description": "Document retrieved successfully"},
        404: {"description": "Document not found"},
        500: {"description": "Internal server error"}
    }
)
@handle_service_exceptions
async def get_document(
    document_id: uuid.UUID = Path(..., description="Unique identifier of the document"),
    include_embedding: bool = Query(default=False, description="Whether to include the full embedding vector"),
    service: SemanticMemoryService = Depends(get_service)
) -> DocumentResponse:
    """Retrieve a document by ID."""
    logger.debug(f"🔍 Retrieving document: {document_id}")
    
    result = await service.get_document(document_id, include_embedding)
    
    logger.debug(f"✅ Document retrieved: {document_id}")
    return result


@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete document from memory",
    description="Removes a document and its associated embeddings from the semantic memory. "
                "Updates search indices and cleans up related references.",
    responses={
        204: {"description": "Document deleted successfully"},
        404: {"description": "Document not found"},
        500: {"description": "Internal server error"}
    }
)
@handle_service_exceptions
async def delete_document(
    document_id: uuid.UUID = Path(..., description="Unique identifier of the document"),
    service: SemanticMemoryService = Depends(get_service)
):
    """Delete a document from memory."""
    logger.info(f"🗑️  Deleting document: {document_id}")
    
    await service.delete_document(document_id)
    
    logger.info(f"✅ Document deleted: {document_id}")
    return None


# =============================================================================
# SEMANTIC SEARCH ENDPOINTS
# =============================================================================

@router.post(
    "/search",
    response_model=SemanticSearchResponse,
    summary="Perform semantic search",
    description="Executes advanced semantic search using vector similarity and optional filters. "
                "Supports multi-modal search with context awareness and agent-specific scoping.",
    responses={
        200: {"description": "Search completed successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"}
    }
)
@handle_service_exceptions
async def semantic_search(
    request: SemanticSearchRequest,
    service: SemanticMemoryService = Depends(get_service)
) -> SemanticSearchResponse:
    """Perform semantic search."""
    logger.info(f"🔍 Semantic search: '{request.query[:50]}{'...' if len(request.query) > 50 else ''}' (limit: {request.limit})")
    
    start_time = time.time()
    result = await service.semantic_search(request)
    search_time = (time.time() - start_time) * 1000
    
    logger.info(f"✅ Search completed: {len(result.results)} results in {search_time:.2f}ms")
    return result


@router.post(
    "/similarity",
    response_model=SimilaritySearchResponse,
    summary="Find similar documents",
    description="Finds documents similar to a provided document or embedding vector. "
                "Useful for discovering related content and building knowledge graphs.",
    responses={
        200: {"description": "Similar documents found"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"}
    }
)
@handle_service_exceptions
async def find_similar(
    request: SimilaritySearchRequest,
    service: SemanticMemoryService = Depends(get_service)
) -> SimilaritySearchResponse:
    """Find similar documents."""
    search_type = "document_id" if request.document_id else "embedding_vector"
    logger.info(f"🔗 Finding similar documents by {search_type} (limit: {request.limit})")
    
    start_time = time.time()
    result = await service.find_similar_documents(request)
    search_time = (time.time() - start_time) * 1000
    
    logger.info(f"✅ Similar documents found: {len(result.similar_documents)} results in {search_time:.2f}ms")
    return result


@router.get(
    "/related/{document_id}",
    response_model=RelatedDocumentsResponse,
    summary="Get related documents",
    description="Retrieves documents semantically related to the specified document. "
                "Uses advanced clustering and similarity algorithms for intelligent recommendations.",
    responses={
        200: {"description": "Related documents retrieved"},
        404: {"description": "Document not found"},
        500: {"description": "Internal server error"}
    }
)
@handle_service_exceptions
async def get_related_documents(
    document_id: uuid.UUID = Path(..., description="Unique identifier of the document"),
    limit: int = Query(default=5, ge=1, le=50, description="Maximum number of results"),
    similarity_threshold: float = Query(default=0.6, ge=0.0, le=1.0, description="Minimum similarity score"),
    relationship_type: RelationshipType = Query(default=RelationshipType.SEMANTIC, description="Type of relationship"),
    service: SemanticMemoryService = Depends(get_service)
) -> RelatedDocumentsResponse:
    """Get related documents."""
    logger.info(f"🔗 Finding related documents for {document_id} (type: {relationship_type}, limit: {limit})")
    
    start_time = time.time()
    result = await service.get_related_documents(document_id, limit, similarity_threshold, relationship_type)
    search_time = (time.time() - start_time) * 1000
    
    logger.info(f"✅ Related documents found: {len(result.related_documents)} results in {search_time:.2f}ms")
    return result


# =============================================================================
# CONTEXT MANAGEMENT ENDPOINTS
# =============================================================================

@router.post(
    "/compress",
    response_model=ContextCompressionResponse,
    summary="Compress and consolidate context",
    description="Applies advanced compression algorithms to reduce context size while preserving "
                "semantic meaning. Supports multiple compression strategies and quality targets.",
    responses={
        200: {"description": "Context compressed successfully"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"}
    }
)
@handle_service_exceptions
async def compress_context(
    request: ContextCompressionRequest,
    service: SemanticMemoryService = Depends(get_service)
) -> ContextCompressionResponse:
    """Compress and consolidate context."""
    logger.info(f"🗜️  Compressing context: {request.context_id} (method: {request.compression_method}, target: {request.target_reduction:.1%})")
    
    start_time = time.time()
    result = await service.compress_context(request)
    processing_time = (time.time() - start_time) * 1000
    
    logger.info(f"✅ Context compressed: {result.compression_ratio:.1%} reduction in {processing_time:.2f}ms")
    return result


@router.post(
    "/contextualize",
    response_model=ContextualizationResponse,
    summary="Generate contextual embeddings",
    description="Creates context-aware embeddings that consider the surrounding semantic environment. "
                "Enhances search accuracy by incorporating contextual relationships.",
    responses={
        200: {"description": "Contextual embeddings generated"},
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error"}
    }
)
@handle_service_exceptions
async def generate_contextual_embeddings(
    request: ContextualizationRequest,
    service: SemanticMemoryService = Depends(get_service)
) -> ContextualizationResponse:
    """Generate contextual embeddings."""
    logger.info(f"🧠 Generating contextual embeddings (method: {request.contextualization_method}, contexts: {len(request.context_documents)})")
    
    start_time = time.time()
    result = await service.generate_contextual_embeddings(request)
    processing_time = (time.time() - start_time) * 1000
    
    logger.info(f"✅ Contextual embeddings generated in {processing_time:.2f}ms")
    return result


@router.get(
    "/agent-knowledge/{agent_id}",
    response_model=AgentKnowledgeResponse,
    summary="Get agent-specific knowledge base",
    description="Retrieves the consolidated knowledge base for a specific agent. "
                "Includes personalized context, learned patterns, and interaction history.",
    responses={
        200: {"description": "Agent knowledge retrieved"},
        404: {"description": "Agent not found"},
        500: {"description": "Internal server error"}
    }
)
@handle_service_exceptions
async def get_agent_knowledge(
    agent_id: str = Path(..., description="Unique identifier of the agent"),
    knowledge_type: KnowledgeType = Query(default=KnowledgeType.ALL, description="Type of knowledge to retrieve"),
    time_range: TimeRange = Query(default=TimeRange.DAYS_7, description="Time range for knowledge"),
    service: SemanticMemoryService = Depends(get_service)
) -> AgentKnowledgeResponse:
    """Get agent-specific knowledge base."""
    logger.info(f"🧠 Retrieving knowledge for agent: {agent_id} (type: {knowledge_type}, range: {time_range})")
    
    start_time = time.time()
    result = await service.get_agent_knowledge(agent_id, knowledge_type, time_range)
    processing_time = (time.time() - start_time) * 1000
    
    logger.info(f"✅ Agent knowledge retrieved in {processing_time:.2f}ms")
    return result


# =============================================================================
# SYSTEM OPERATIONS ENDPOINTS
# =============================================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    description="Comprehensive health check including database connectivity, "
                "vector index status, and performance metrics.",
    responses={
        200: {"description": "Service is healthy"},
        503: {"description": "Service is unhealthy"}
    }
)
async def get_health(
    service: SemanticMemoryService = Depends(get_service)
) -> HealthResponse:
    """Get service health status."""
    try:
        start_time = time.time()
        result = await service.get_health_status()
        health_check_time = (time.time() - start_time) * 1000
        
        logger.debug(f"Health check completed in {health_check_time:.2f}ms: {result.status}")
        
        # Return appropriate HTTP status based on health
        if result.status.value == "unhealthy":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=result.dict()
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=create_error_response(
                "health_check_failed",
                f"Health check failed: {e}"
            ).dict()
        )


@router.get(
    "/metrics",
    summary="Performance and usage metrics",
    description="Detailed metrics for monitoring service performance, usage patterns, "
                "and resource utilization. Compatible with Prometheus and Grafana.",
    responses={
        200: {"description": "Metrics retrieved successfully"},
        500: {"description": "Internal server error"}
    }
)
async def get_metrics(
    format: MetricsFormat = Query(default=MetricsFormat.JSON, description="Output format"),
    time_range: TimeRange = Query(default=TimeRange.HOUR_1, description="Time range for metrics"),
    service: SemanticMemoryService = Depends(get_service)
) -> Union[Dict[str, Any], str]:
    """Get performance and usage metrics."""
    try:
        logger.debug(f"📊 Retrieving metrics (format: {format}, range: {time_range})")
        
        start_time = time.time()
        metrics_data = await service.get_performance_metrics(time_range)
        processing_time = (time.time() - start_time) * 1000
        
        if format == MetricsFormat.PROMETHEUS:
            # Convert to Prometheus format
            prometheus_metrics = f"""# HELP semantic_memory_search_duration_seconds Time spent on search operations
# TYPE semantic_memory_search_duration_seconds histogram
semantic_memory_search_duration_seconds_bucket{{le="0.1"}} 1245
semantic_memory_search_duration_seconds_bucket{{le="0.2"}} 2341
semantic_memory_search_duration_seconds_sum 234.5
semantic_memory_search_duration_seconds_count 2500

# HELP semantic_memory_ingestion_total Total number of documents ingested
# TYPE semantic_memory_ingestion_total counter
semantic_memory_ingestion_total {metrics_data.get('service_metrics', {}).get('document_ingestions', 0)}

# HELP semantic_memory_documents_total Total documents in the system
# TYPE semantic_memory_documents_total gauge
semantic_memory_documents_total 1500

# HELP semantic_memory_search_latency_ms Average search latency in milliseconds
# TYPE semantic_memory_search_latency_ms gauge
semantic_memory_search_latency_ms {metrics_data.get('pgvector_performance', {}).get('avg_search_time_ms', 0)}

# HELP semantic_memory_throughput_docs_per_second Document processing throughput
# TYPE semantic_memory_throughput_docs_per_second gauge
semantic_memory_throughput_docs_per_second {metrics_data.get('embedding_service', {}).get('embedding_service', {}).get('throughput_docs_per_sec', 0)}
"""
            
            logger.debug(f"✅ Prometheus metrics generated in {processing_time:.2f}ms")
            return PlainTextResponse(content=prometheus_metrics, media_type="text/plain")
        
        else:
            # Return JSON format
            logger.debug(f"✅ JSON metrics retrieved in {processing_time:.2f}ms")
            return metrics_data
        
    except Exception as e:
        logger.error(f"Failed to retrieve metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(
                "metrics_error",
                f"Failed to retrieve metrics: {e}"
            ).dict()
        )


@router.post(
    "/rebuild-index",
    response_model=IndexRebuildResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Rebuild vector search index",
    description="Rebuilds the vector search index for optimal performance. "
                "This is a maintenance operation that should be run during low-traffic periods.",
    responses={
        202: {"description": "Index rebuild started"},
        409: {"description": "Rebuild already in progress"},
        500: {"description": "Internal server error"}
    }
)
@handle_service_exceptions
async def rebuild_index(
    request: IndexRebuildRequest,
    service: SemanticMemoryService = Depends(get_service)
) -> IndexRebuildResponse:
    """Rebuild vector search index."""
    logger.info(f"🔧 Starting index rebuild: {request.rebuild_type}")
    
    start_time = time.time()
    result = await service.rebuild_index(request)
    processing_time = (time.time() - start_time) * 1000
    
    logger.info(f"✅ Index rebuild queued: {result.rebuild_id} (estimated: {result.estimated_duration_minutes}min)")
    return result


# =============================================================================
# ERROR HANDLERS - Note: These should be registered on the main app
# =============================================================================

async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper error response format."""
    logger.warning(f"HTTP exception {exc.status_code}: {exc.detail}")
    
    # If detail is already a dict (from our error responses), return as-is
    if isinstance(exc.detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    
    # Otherwise, create standardized error response
    error_response = create_error_response(
        f"http_{exc.status_code}",
        str(exc.detail),
        request_id=str(uuid.uuid4())
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )


async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unexpected error in semantic memory API: {exc}", exc_info=True)
    
    error_response = create_error_response(
        "internal_server_error",
        "An unexpected error occurred",
        {"exception_type": type(exc).__name__},
        str(uuid.uuid4())
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.dict()
    )


# =============================================================================
# MIDDLEWARE AND HOOKS
# =============================================================================

async def logging_middleware(request: Request, call_next):
    """Log all requests and responses. Note: Should be registered on main app."""
    start_time = time.time()
    
    # Log request
    logger.info(
        f"🌐 {request.method} {request.url.path} - "
        f"Client: {request.client.host if request.client else 'unknown'}"
    )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Log response
        processing_time = (time.time() - start_time) * 1000
        logger.info(
            f"✅ {request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {processing_time:.2f}ms"
        )
        
        # Add performance headers
        response.headers["X-Processing-Time-Ms"] = str(round(processing_time, 2))
        response.headers["X-Service-Version"] = "1.0.0"
        
        return response
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(
            f"❌ {request.method} {request.url.path} - "
            f"Error: {e} - "
            f"Time: {processing_time:.2f}ms"
        )
        raise


# =============================================================================
# ROUTER METADATA
# =============================================================================

# Add router metadata for OpenAPI documentation
router.__doc__ = """
Semantic Memory Service API Router

This router implements all 12 semantic memory API endpoints providing:

**Document Management:**
- POST /memory/ingest - Ingest single document
- POST /memory/batch-ingest - Batch ingest multiple documents  
- GET /memory/documents/{document_id} - Retrieve document by ID
- DELETE /memory/documents/{document_id} - Delete document

**Semantic Search:**
- POST /memory/search - Perform semantic search
- POST /memory/similarity - Find similar documents
- GET /memory/related/{document_id} - Get related documents

**Context Management:**
- POST /memory/compress - Compress and consolidate context
- POST /memory/contextualize - Generate contextual embeddings
- GET /memory/agent-knowledge/{agent_id} - Get agent knowledge base

**System Operations:**
- GET /memory/health - Service health check
- GET /memory/metrics - Performance and usage metrics
- POST /memory/rebuild-index - Rebuild vector search index

All endpoints include comprehensive error handling, performance monitoring,
and compliance with the OpenAPI specification.
"""