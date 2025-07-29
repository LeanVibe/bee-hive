"""
Pydantic schemas for Semantic Memory Service API.

This module defines all request/response models for the LeanVibe Agent Hive 2.0
Semantic Memory Service, providing type safety and validation for API contracts.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, validator


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class CompressionMethod(str, Enum):
    """Available context compression algorithms."""
    SEMANTIC_CLUSTERING = "semantic_clustering"
    IMPORTANCE_FILTERING = "importance_filtering"
    TEMPORAL_DECAY = "temporal_decay"
    HYBRID = "hybrid"


class ContextualizationMethod(str, Enum):
    """Methods for generating contextual embeddings."""
    WEIGHTED_AVERAGE = "weighted_average"
    ATTENTION_BASED = "attention_based"
    HIERARCHICAL = "hierarchical"


class RelationshipType(str, Enum):
    """Types of document relationships."""
    SEMANTIC = "semantic"
    TOPICAL = "topical"
    TEMPORAL = "temporal"
    AGENT_BASED = "agent-based"


class ProcessingPriority(str, Enum):
    """Processing priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentStatus(str, Enum):
    """Individual component status."""
    HEALTHY = "healthy"
    REBUILDING = "rebuilding"
    UNHEALTHY = "unhealthy"


class RebuildType(str, Enum):
    """Index rebuild types."""
    FULL = "full"
    INCREMENTAL = "incremental"
    OPTIMIZE = "optimize"


class IndexStatus(str, Enum):
    """Index rebuild status."""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class MetricsFormat(str, Enum):
    """Metrics output formats."""
    JSON = "json"
    PROMETHEUS = "prometheus"


class TimeRange(str, Enum):
    """Standard time range options."""
    HOUR_1 = "1h"
    HOURS_6 = "6h"
    HOURS_24 = "24h"
    DAYS_7 = "7d"
    DAYS_30 = "30d"
    ALL = "all"


class KnowledgeType(str, Enum):
    """Types of agent knowledge."""
    PATTERNS = "patterns"
    INTERACTIONS = "interactions"
    CONSOLIDATED = "consolidated"
    ALL = "all"


# =============================================================================
# DOCUMENT MANAGEMENT SCHEMAS
# =============================================================================

class ProcessingOptions(BaseModel):
    """Options for document processing."""
    generate_summary: bool = Field(default=False, description="Generate automatic summary")
    extract_entities: bool = Field(default=False, description="Extract named entities")
    priority: ProcessingPriority = Field(default=ProcessingPriority.NORMAL, description="Processing priority")


class DocumentMetadata(BaseModel):
    """Document metadata structure."""
    title: Optional[str] = Field(None, max_length=255, description="Document title")
    source: Optional[str] = Field(None, max_length=100, description="Source system or origin")
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score")
    created_by: Optional[str] = Field(None, max_length=100, description="Creator identifier")
    
    class Config:
        extra = "allow"  # Allow additional metadata fields


class DocumentIngestRequest(BaseModel):
    """Request schema for ingesting a single document."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "Agent coordination requires careful message ordering and failure recovery mechanisms.",
                "metadata": {
                    "title": "Agent Coordination Patterns",
                    "source": "technical_documentation",
                    "importance": 0.8
                },
                "agent_id": "orchestrator-001",
                "workflow_id": "12345678-1234-5678-9012-123456789abc",
                "tags": ["coordination", "distributed-systems", "patterns"]
            }
        }
    )
    
    content: str = Field(
        ..., 
        min_length=1, 
        max_length=100000, 
        description="Document content to be processed and indexed"
    )
    metadata: Optional[DocumentMetadata] = Field(
        default_factory=DocumentMetadata, 
        description="Additional document metadata"
    )
    agent_id: str = Field(
        ..., 
        pattern=r'^[a-zA-Z0-9\-_]+$', 
        max_length=100, 
        description="ID of the agent creating this document"
    )
    workflow_id: Optional[uuid.UUID] = Field(None, description="Optional workflow context identifier")
    tags: List[str] = Field(
        default_factory=list, 
        max_items=20, 
        description="Semantic tags for categorization"
    )
    processing_options: Optional[ProcessingOptions] = Field(
        default_factory=ProcessingOptions, 
        description="Processing configuration options"
    )
    
    @validator('tags')
    def validate_tags(cls, v):
        """Validate tag format and length."""
        for tag in v:
            if len(tag) > 50:
                raise ValueError(f"Tag '{tag}' exceeds maximum length of 50 characters")
            if not tag.strip():
                raise ValueError("Tags cannot be empty or whitespace")
        return v


class ExtractedEntity(BaseModel):
    """Extracted entity information."""
    entity: str = Field(..., description="Entity text")
    type: str = Field(..., description="Entity type (PERSON, ORG, etc.)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")


class DocumentIngestResponse(BaseModel):
    """Response schema for document ingestion."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "doc_12345678-1234-5678-9012-123456789abc",
                "embedding_id": "emb_87654321-4321-8765-2109-876543210fed",
                "processing_time_ms": 45.3,
                "vector_dimensions": 1536,
                "index_updated": True
            }
        }
    )
    
    document_id: uuid.UUID = Field(..., description="Unique identifier for the ingested document")
    embedding_id: uuid.UUID = Field(..., description="Identifier for the generated embedding vector")
    processing_time_ms: float = Field(..., ge=0, description="Time taken to process the document")
    vector_dimensions: int = Field(..., gt=0, description="Number of dimensions in the embedding vector")
    index_updated: bool = Field(..., description="Whether the search index was updated")
    summary: Optional[str] = Field(None, description="Auto-generated summary if requested")
    extracted_entities: List[ExtractedEntity] = Field(
        default_factory=list, 
        description="Extracted entities if requested"
    )


class BatchOptions(BaseModel):
    """Options for batch processing."""
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    generate_summary: bool = Field(default=False, description="Generate summaries for all documents")
    fail_on_error: bool = Field(default=False, description="Stop batch on first error")


class BatchIngestRequest(BaseModel):
    """Request schema for batch document ingestion."""
    documents: List[DocumentIngestRequest] = Field(
        ..., 
        min_items=1, 
        max_items=100, 
        description="Array of documents to ingest"
    )
    batch_options: Optional[BatchOptions] = Field(
        default_factory=BatchOptions, 
        description="Batch processing options"
    )


class BatchIngestResult(BaseModel):
    """Result for individual document in batch."""
    index: int = Field(..., description="Index in the original batch")
    status: str = Field(..., pattern=r'^(success|error)$', description="Processing status")
    document_id: Optional[uuid.UUID] = Field(None, description="Document ID if successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class BatchIngestResponse(BaseModel):
    """Response schema for batch document ingestion."""
    total_documents: int = Field(..., ge=0, description="Total number of documents processed")
    successful_ingestions: int = Field(..., ge=0, description="Number of successful ingestions")
    failed_ingestions: int = Field(..., ge=0, description="Number of failed ingestions")
    processing_time_ms: float = Field(..., ge=0, description="Total processing time")
    results: List[BatchIngestResult] = Field(..., description="Individual processing results")
    batch_summary: Optional[str] = Field(None, description="Overall batch summary if generated")


class DocumentResponse(BaseModel):
    """Response schema for document retrieval."""
    model_config = ConfigDict(from_attributes=True)
    
    document_id: uuid.UUID
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    agent_id: str
    workflow_id: Optional[uuid.UUID] = None
    tags: List[str] = Field(default_factory=list)
    embedding_vector: Optional[List[float]] = Field(None, description="Full embedding vector (if requested)")
    created_at: datetime
    updated_at: Optional[datetime] = None
    access_count: int = Field(default=0, ge=0)
    last_accessed: Optional[datetime] = None


# =============================================================================
# SEARCH SCHEMAS
# =============================================================================

class SearchFilters(BaseModel):
    """Advanced search filters."""
    tags: Optional[List[str]] = Field(None, description="Filter by document tags")
    importance_min: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum importance score")
    importance_max: Optional[float] = Field(None, ge=0.0, le=1.0, description="Maximum importance score")
    date_from: Optional[datetime] = Field(None, description="Filter documents from this date")
    date_to: Optional[datetime] = Field(None, description="Filter documents to this date")
    metadata_filters: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata filters")


class SearchOptions(BaseModel):
    """Advanced search options."""
    rerank: bool = Field(default=False, description="Apply advanced reranking algorithms")
    include_metadata: bool = Field(default=True, description="Include document metadata")
    include_embeddings: bool = Field(default=False, description="Include embedding vectors")
    explain_relevance: bool = Field(default=False, description="Include relevance explanations")


class SemanticSearchRequest(BaseModel):
    """Request schema for semantic search."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "How do agents coordinate in distributed workflows?",
                "limit": 10,
                "similarity_threshold": 0.7,
                "agent_id": "orchestrator-001",
                "filters": {
                    "tags": ["coordination", "workflows"],
                    "importance_min": 0.5
                },
                "search_options": {
                    "rerank": True,
                    "include_metadata": True
                }
            }
        }
    )
    
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1000, 
        description="Natural language search query"
    )
    limit: int = Field(
        default=10, 
        ge=1, 
        le=100, 
        description="Maximum number of results to return"
    )
    similarity_threshold: float = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0, 
        description="Minimum similarity score for results"
    )
    agent_id: Optional[str] = Field(None, description="Scope search to specific agent's context")
    workflow_id: Optional[uuid.UUID] = Field(None, description="Scope search to specific workflow")
    filters: Optional[SearchFilters] = Field(None, description="Advanced search filters")
    search_options: Optional[SearchOptions] = Field(
        default_factory=SearchOptions, 
        description="Search configuration options"
    )


class SearchResult(BaseModel):
    """Individual search result."""
    document_id: uuid.UUID
    content: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    agent_id: str
    tags: List[str] = Field(default_factory=list)
    relevance_explanation: Optional[str] = Field(None, description="Explanation of relevance")
    highlighted_content: Optional[str] = Field(None, description="Content with highlighted terms")
    embedding_vector: Optional[List[float]] = Field(None, description="Document embedding if requested")


class SemanticSearchResponse(BaseModel):
    """Response schema for semantic search."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "document_id": "doc_12345",
                        "content": "Agent coordination patterns...",
                        "similarity_score": 0.87,
                        "metadata": {"type": "pattern", "importance": 0.8},
                        "agent_id": "orchestrator-001",
                        "tags": ["coordination"],
                        "relevance_explanation": "High semantic similarity in coordination concepts"
                    }
                ],
                "total_found": 25,
                "search_time_ms": 143.7,
                "query_embedding_time_ms": 12.4
            }
        }
    )
    
    results: List[SearchResult] = Field(..., description="Search results")
    total_found: int = Field(..., ge=0, description="Total number of matching documents")
    search_time_ms: float = Field(..., ge=0, description="Time taken for the search operation")
    query_embedding_time_ms: float = Field(..., ge=0, description="Time to generate query embedding")
    reranking_applied: bool = Field(default=False, description="Whether reranking was applied")
    suggestions: List[str] = Field(default_factory=list, description="Query suggestions for better results")


class SimilaritySearchRequest(BaseModel):
    """Request schema for similarity search."""
    document_id: Optional[uuid.UUID] = Field(None, description="Find documents similar to this document")
    embedding_vector: Optional[List[float]] = Field(None, description="Find documents similar to this embedding")
    limit: int = Field(default=5, ge=1, le=50, description="Maximum number of results")
    similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum similarity score")
    exclude_self: bool = Field(default=True, description="Exclude the source document from results")
    
    @validator('document_id', 'embedding_vector')
    def validate_similarity_input(cls, v, values):
        """Ensure either document_id or embedding_vector is provided."""
        if 'document_id' in values and 'embedding_vector' in values:
            if not values['document_id'] and not values['embedding_vector']:
                raise ValueError("Either document_id or embedding_vector must be provided")
        return v


class SimilaritySearchResponse(BaseModel):
    """Response schema for similarity search."""
    similar_documents: List[SearchResult] = Field(..., description="Similar documents found")
    search_time_ms: float = Field(..., ge=0, description="Time taken for the search")


class RelatedDocument(SearchResult):
    """Extended search result with relationship information."""
    relationship_type: RelationshipType = Field(..., description="Type of relationship")
    relationship_strength: float = Field(..., ge=0.0, le=1.0, description="Strength of the relationship")


class RelationshipAnalysis(BaseModel):
    """Analysis of document relationships."""
    total_relationships_found: int = Field(..., ge=0)
    relationship_distribution: Dict[RelationshipType, int] = Field(default_factory=dict)
    analysis_time_ms: float = Field(..., ge=0)


class RelatedDocumentsResponse(BaseModel):
    """Response schema for related documents."""
    related_documents: List[RelatedDocument] = Field(..., description="Related documents")
    relationship_analysis: RelationshipAnalysis = Field(..., description="Relationship analysis")


# =============================================================================
# CONTEXT MANAGEMENT SCHEMAS
# =============================================================================

class CompressionOptions(BaseModel):
    """Options for context compression."""
    preserve_recent: bool = Field(default=True, description="Preserve recently accessed documents")
    maintain_relationships: bool = Field(default=True, description="Maintain document relationships")
    generate_summary: bool = Field(default=True, description="Generate compression summary")


class ContextCompressionRequest(BaseModel):
    """Request schema for context compression."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "context_id": "ctx_workflow_analysis_2024",
                "compression_method": "semantic_clustering",
                "target_reduction": 0.7,
                "preserve_importance_threshold": 0.8,
                "agent_id": "context-optimizer"
            }
        }
    )
    
    context_id: str = Field(..., min_length=1, description="Identifier for the context to compress")
    compression_method: CompressionMethod = Field(..., description="Algorithm to use for compression")
    target_reduction: float = Field(
        default=0.5, 
        ge=0.1, 
        le=0.9, 
        description="Target compression ratio (0.7 = 70% reduction)"
    )
    preserve_importance_threshold: float = Field(
        default=0.8, 
        ge=0.0, 
        le=1.0, 
        description="Preserve documents above this importance level"
    )
    agent_id: str = Field(..., description="Agent requesting the compression")
    compression_options: Optional[CompressionOptions] = Field(
        default_factory=CompressionOptions, 
        description="Compression configuration options"
    )


class ContextCompressionResponse(BaseModel):
    """Response schema for context compression."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "compressed_context_id": "ctx_compressed_workflow_analysis_2024",
                "original_size": 15420,
                "compressed_size": 4626,
                "compression_ratio": 0.7,
                "semantic_preservation_score": 0.94,
                "processing_time_ms": 234.5
            }
        }
    )
    
    compressed_context_id: str = Field(..., description="Identifier for the compressed context")
    original_size: int = Field(..., ge=0, description="Original context size in tokens/characters")
    compressed_size: int = Field(..., ge=0, description="Compressed context size")
    compression_ratio: float = Field(..., ge=0.0, le=1.0, description="Actual compression ratio achieved")
    semantic_preservation_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="How well semantic meaning was preserved"
    )
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
    compression_summary: Optional[str] = Field(None, description="Summary of compression actions")
    preserved_documents: List[uuid.UUID] = Field(
        default_factory=list, 
        description="IDs of documents that were preserved"
    )


class ContextualizationRequest(BaseModel):
    """Request schema for generating contextual embeddings."""
    content: str = Field(..., min_length=1, description="Content to generate contextual embeddings for")
    context_documents: List[uuid.UUID] = Field(..., min_items=1, description="Document IDs to use as context")
    contextualization_method: ContextualizationMethod = Field(
        default=ContextualizationMethod.ATTENTION_BASED, 
        description="Method for contextualization"
    )
    agent_id: Optional[str] = Field(None, description="Agent requesting contextualization")


class ContextualizationResponse(BaseModel):
    """Response schema for contextual embeddings."""
    contextual_embedding: List[float] = Field(..., description="Generated contextual embedding vector")
    context_influence_scores: Dict[str, float] = Field(
        ..., 
        description="How much each context document influenced the embedding"
    )
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
    context_summary: Optional[str] = Field(None, description="Summary of the contextual influences")


class LearnedPattern(BaseModel):
    """A learned pattern in agent knowledge."""
    pattern_id: str = Field(..., description="Unique pattern identifier")
    description: str = Field(..., description="Pattern description")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Pattern confidence score")
    occurrences: int = Field(..., ge=0, description="Number of times pattern was observed")


class AgentInteraction(BaseModel):
    """Record of agent interaction."""
    interaction_id: str = Field(..., description="Unique interaction identifier")
    timestamp: datetime = Field(..., description="When the interaction occurred")
    context: str = Field(..., description="Context of the interaction")
    outcome: str = Field(..., description="Outcome or result of the interaction")


class ConsolidatedKnowledge(BaseModel):
    """Consolidated knowledge base."""
    key_insights: List[str] = Field(default_factory=list, description="Key insights learned")
    expertise_areas: List[str] = Field(default_factory=list, description="Areas of expertise")
    learned_preferences: Dict[str, Any] = Field(default_factory=dict, description="Learned preferences")


class KnowledgeBase(BaseModel):
    """Agent knowledge base structure."""
    patterns: List[LearnedPattern] = Field(default_factory=list, description="Learned patterns")
    interactions: List[AgentInteraction] = Field(default_factory=list, description="Interaction history")
    consolidated_knowledge: ConsolidatedKnowledge = Field(
        default_factory=ConsolidatedKnowledge, 
        description="Consolidated knowledge"
    )


class KnowledgeStats(BaseModel):
    """Statistics about agent knowledge."""
    total_documents: int = Field(..., ge=0, description="Total documents in knowledge base")
    unique_patterns: int = Field(..., ge=0, description="Number of unique patterns")
    interaction_count: int = Field(..., ge=0, description="Total interactions recorded")
    knowledge_confidence: float = Field(..., ge=0.0, le=1.0, description="Overall knowledge confidence")


class AgentKnowledgeResponse(BaseModel):
    """Response schema for agent knowledge retrieval."""
    agent_id: str = Field(..., description="Agent identifier")
    knowledge_base: KnowledgeBase = Field(..., description="The agent's knowledge base")
    last_updated: datetime = Field(..., description="When knowledge was last updated")
    knowledge_stats: KnowledgeStats = Field(..., description="Knowledge statistics")


# =============================================================================
# SYSTEM OPERATION SCHEMAS
# =============================================================================

class DatabaseHealth(BaseModel):
    """Database health information."""
    status: ComponentStatus
    response_time_ms: float = Field(..., ge=0)
    connection_pool_size: int = Field(..., ge=0)


class VectorIndexHealth(BaseModel):
    """Vector index health information."""
    status: ComponentStatus
    documents_indexed: int = Field(..., ge=0)
    index_size_mb: float = Field(..., ge=0)
    last_rebuild: Optional[datetime] = None


class MemoryUsage(BaseModel):
    """Memory usage information."""
    heap_used_mb: float = Field(..., ge=0)
    heap_max_mb: float = Field(..., ge=0)
    gc_collections: int = Field(..., ge=0)


class HealthComponents(BaseModel):
    """Health status of system components."""
    database: DatabaseHealth
    vector_index: VectorIndexHealth
    memory_usage: MemoryUsage


class PerformanceMetrics(BaseModel):
    """System performance metrics."""
    avg_search_time_ms: float = Field(..., ge=0)
    avg_ingestion_time_ms: float = Field(..., ge=0)
    throughput_docs_per_sec: float = Field(..., ge=0)
    error_rate_percent: float = Field(..., ge=0.0, le=100.0)


class HealthResponse(BaseModel):
    """Response schema for health check."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T10:30:00Z",
                "version": "1.0.0",
                "components": {
                    "database": {
                        "status": "healthy",
                        "response_time_ms": 5.2,
                        "connection_pool_size": 10
                    },
                    "vector_index": {
                        "status": "healthy",
                        "documents_indexed": 15420,
                        "index_size_mb": 234.5
                    },
                    "memory_usage": {
                        "heap_used_mb": 512,
                        "heap_max_mb": 2048,
                        "gc_collections": 15
                    }
                },
                "performance_metrics": {
                    "avg_search_time_ms": 89.3,
                    "avg_ingestion_time_ms": 23.1,
                    "throughput_docs_per_sec": 847.2,
                    "error_rate_percent": 0.1
                }
            }
        }
    )
    
    status: HealthStatus
    timestamp: datetime
    version: str
    components: HealthComponents
    performance_metrics: PerformanceMetrics


class OperationMetrics(BaseModel):
    """Metrics for a specific operation type."""
    total_count: int = Field(..., ge=0)
    avg_duration_ms: float = Field(..., ge=0)
    p95_duration_ms: Optional[float] = Field(None, ge=0)
    p99_duration_ms: Optional[float] = Field(None, ge=0)
    throughput_docs_per_sec: Optional[float] = Field(None, ge=0)
    error_count: int = Field(..., ge=0)


class CompressionMetrics(BaseModel):
    """Metrics for compression operations."""
    total_count: int = Field(..., ge=0)
    avg_compression_ratio: float = Field(..., ge=0.0, le=1.0)
    avg_processing_time_ms: float = Field(..., ge=0)


class ResourceMetrics(BaseModel):
    """System resource utilization metrics."""
    memory_usage_mb: float = Field(..., ge=0)
    cpu_usage_percent: float = Field(..., ge=0.0, le=100.0)
    disk_usage_mb: float = Field(..., ge=0)
    network_io_mb: float = Field(..., ge=0)


class BusinessMetrics(BaseModel):
    """Business-level metrics."""
    total_documents: int = Field(..., ge=0)
    total_agents: int = Field(..., ge=0)
    active_workflows: int = Field(..., ge=0)
    knowledge_base_size_mb: float = Field(..., ge=0)


class DetailedPerformanceMetrics(BaseModel):
    """Detailed performance metrics breakdown."""
    search_operations: OperationMetrics
    ingestion_operations: OperationMetrics
    compression_operations: CompressionMetrics


class MetricsResponse(BaseModel):
    """Response schema for system metrics."""
    timestamp: datetime
    time_range: TimeRange
    performance_metrics: DetailedPerformanceMetrics
    resource_metrics: ResourceMetrics
    business_metrics: BusinessMetrics


class MaintenanceWindow(BaseModel):
    """Maintenance window configuration."""
    start_time: datetime
    max_duration_minutes: int = Field(..., ge=1, le=1440)


class IndexRebuildRequest(BaseModel):
    """Request schema for index rebuild."""
    rebuild_type: RebuildType = Field(default=RebuildType.INCREMENTAL)
    priority: ProcessingPriority = Field(default=ProcessingPriority.NORMAL)
    maintenance_window: Optional[MaintenanceWindow] = None


class IndexRebuildResponse(BaseModel):
    """Response schema for index rebuild."""
    rebuild_id: uuid.UUID
    status: IndexStatus
    estimated_duration_minutes: int = Field(..., ge=0)
    progress_url: Optional[str] = Field(None, description="URL to check rebuild progress")


# =============================================================================
# ERROR SCHEMAS
# =============================================================================

class ErrorResponse(BaseModel):
    """Standard error response schema."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "validation_error",
                "message": "Invalid similarity threshold: must be between 0.0 and 1.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req_12345"
            }
        }
    )
    
    error: str = Field(..., description="Error code or type")
    message: str = Field(..., description="Human-readable error description")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error context")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Unique request identifier for debugging")


# =============================================================================
# EXPORT ALL SCHEMAS
# =============================================================================

__all__ = [
    # Enums
    "CompressionMethod",
    "ContextualizationMethod", 
    "RelationshipType",
    "ProcessingPriority",
    "HealthStatus",
    "ComponentStatus",
    "RebuildType",
    "IndexStatus",
    "MetricsFormat",
    "TimeRange",
    "KnowledgeType",
    
    # Document Management
    "ProcessingOptions",
    "DocumentMetadata",
    "DocumentIngestRequest",
    "ExtractedEntity",
    "DocumentIngestResponse",
    "BatchOptions",
    "BatchIngestRequest",
    "BatchIngestResult",
    "BatchIngestResponse",
    "DocumentResponse",
    
    # Search Schemas
    "SearchFilters",
    "SearchOptions",
    "SemanticSearchRequest",
    "SearchResult",
    "SemanticSearchResponse",
    "SimilaritySearchRequest",
    "SimilaritySearchResponse",
    "RelatedDocument",
    "RelationshipAnalysis",
    "RelatedDocumentsResponse",
    
    # Context Management
    "CompressionOptions",
    "ContextCompressionRequest",
    "ContextCompressionResponse",
    "ContextualizationRequest",
    "ContextualizationResponse",
    "LearnedPattern",
    "AgentInteraction",
    "ConsolidatedKnowledge",
    "KnowledgeBase",
    "KnowledgeStats",
    "AgentKnowledgeResponse",
    
    # System Operations
    "DatabaseHealth",
    "VectorIndexHealth",
    "MemoryUsage",
    "HealthComponents",
    "PerformanceMetrics",
    "HealthResponse",
    "OperationMetrics",
    "CompressionMetrics",
    "ResourceMetrics",
    "BusinessMetrics",
    "DetailedPerformanceMetrics",
    "MetricsResponse",
    "MaintenanceWindow",
    "IndexRebuildRequest",
    "IndexRebuildResponse",
    
    # Error Handling
    "ErrorResponse",
]