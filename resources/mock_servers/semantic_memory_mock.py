"""
Mock Server for Semantic Memory Service API

Provides realistic mock responses for all Semantic Memory Service endpoints,
enabling parallel development of subagents while the actual service is being built.

Features:
- Realistic response data generation
- Configurable latency simulation
- Stateful document storage for testing
- Performance metrics simulation
- Error scenario simulation
"""

import asyncio
import uuid
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Query, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
import json
import logging
from dataclasses import dataclass, field

# Import our Pydantic schemas
from ..app.schemas.semantic_memory import (
    DocumentIngestRequest, DocumentIngestResponse, BatchIngestRequest, BatchIngestResponse,
    DocumentResponse, SemanticSearchRequest, SemanticSearchResponse, SimilaritySearchRequest,
    SimilaritySearchResponse, RelatedDocumentsResponse, ContextCompressionRequest,
    ContextCompressionResponse, ContextualizationRequest, ContextualizationResponse,
    AgentKnowledgeResponse, HealthResponse, MetricsResponse, IndexRebuildRequest,
    IndexRebuildResponse, ErrorResponse, SearchResult, RelatedDocument,
    HealthStatus, ComponentStatus, TimeRange, MetricsFormat, RelationshipType,
    ProcessingPriority, RebuildType, IndexStatus
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# MOCK DATA STORAGE AND MANAGEMENT
# =============================================================================

@dataclass
class MockDocument:
    """In-memory document for mock responses."""
    document_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_id: str = ""
    workflow_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    embedding_vector: List[float] = field(default_factory=lambda: [random.random() for _ in range(1536)])
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    similarity_score: float = 0.0  # Used for search results


@dataclass
class MockServerConfig:
    """Configuration for mock server behavior."""
    enable_latency_simulation: bool = True
    min_latency_ms: float = 10.0
    max_latency_ms: float = 200.0
    error_rate: float = 0.02  # 2% error rate
    enable_realistic_data: bool = True
    max_documents: int = 10000


class MockDocumentStore:
    """In-memory document store for realistic testing."""
    
    def __init__(self):
        self.documents: Dict[str, MockDocument] = {}
        self.agent_knowledge: Dict[str, Dict[str, Any]] = {}
        self.compression_contexts: Dict[str, Dict[str, Any]] = {}
        self.rebuild_operations: Dict[str, Dict[str, Any]] = {}
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with sample documents for realistic testing."""
        sample_docs = [
            {
                "content": "Agent coordination in distributed systems requires careful consideration of message ordering, failure recovery, and consensus mechanisms.",
                "agent_id": "orchestrator-001",
                "tags": ["coordination", "distributed-systems", "consensus"],
                "metadata": {"importance": 0.9, "type": "technical_knowledge"}
            },
            {
                "content": "Redis streams provide reliable message ordering and consumer group management for multi-agent communication.",
                "agent_id": "messaging-service",
                "tags": ["redis", "messaging", "streams"],
                "metadata": {"importance": 0.8, "type": "infrastructure"}
            },
            {
                "content": "Context compression algorithms can reduce memory usage by 70% while preserving semantic meaning through clustering techniques.",
                "agent_id": "context-optimizer",
                "tags": ["compression", "optimization", "memory"],
                "metadata": {"importance": 0.85, "type": "performance"}
            },
            {
                "content": "pgvector enables efficient semantic search with cosine similarity and can handle millions of embedding vectors.",
                "agent_id": "search-engine",
                "tags": ["pgvector", "embeddings", "search"],
                "metadata": {"importance": 0.9, "type": "database"}
            },
            {
                "content": "DAG workflows enable complex multi-step agent coordination with dependency management and error handling.",
                "agent_id": "workflow-manager",
                "tags": ["dag", "workflows", "orchestration"],
                "metadata": {"importance": 0.95, "type": "architecture"}
            }
        ]
        
        for doc_data in sample_docs:
            doc_id = str(uuid.uuid4())
            doc = MockDocument(
                document_id=doc_id,
                content=doc_data["content"],
                agent_id=doc_data["agent_id"],
                tags=doc_data["tags"],
                metadata=doc_data["metadata"]
            )
            self.documents[doc_id] = doc
    
    def add_document(self, doc: MockDocument) -> str:
        """Add a document to the store."""
        self.documents[doc.document_id] = doc
        return doc.document_id
    
    def get_document(self, doc_id: str) -> Optional[MockDocument]:
        """Retrieve a document by ID."""
        doc = self.documents.get(doc_id)
        if doc:
            doc.access_count += 1
            doc.last_accessed = datetime.utcnow()
        return doc
    
    def search_documents(self, query: str, limit: int = 10, similarity_threshold: float = 0.7,
                        agent_id: Optional[str] = None, tags: Optional[List[str]] = None) -> List[MockDocument]:
        """Simulate semantic search with realistic results."""
        results = []
        query_words = set(query.lower().split())
        
        for doc in self.documents.values():
            # Filter by agent if specified
            if agent_id and doc.agent_id != agent_id:
                continue
            
            # Filter by tags if specified
            if tags and not any(tag in doc.tags for tag in tags):
                continue
            
            # Calculate mock similarity score based on word overlap
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words.intersection(doc_words))
            similarity = min(0.95, max(0.3, overlap / max(len(query_words), 1) + random.uniform(0.1, 0.3)))
            
            if similarity >= similarity_threshold:
                doc.similarity_score = similarity
                results.append(doc)
        
        # Sort by similarity and limit results
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:limit]
    
    def get_related_documents(self, doc_id: str, limit: int = 5) -> List[MockDocument]:
        """Get documents related to the specified document."""
        source_doc = self.get_document(doc_id)
        if not source_doc:
            return []
        
        related = []
        source_tags = set(source_doc.tags)
        
        for other_doc in self.documents.values():
            if other_doc.document_id == doc_id:
                continue
            
            # Calculate relationship based on tag overlap and agent similarity
            tag_overlap = len(source_tags.intersection(set(other_doc.tags)))
            agent_match = 1.0 if source_doc.agent_id == other_doc.agent_id else 0.3
            
            similarity = min(0.9, tag_overlap * 0.3 + agent_match * 0.4 + random.uniform(0.1, 0.3))
            
            if similarity > 0.5:
                other_doc.similarity_score = similarity
                related.append(other_doc)
        
        related.sort(key=lambda x: x.similarity_score, reverse=True)
        return related[:limit]


# Global mock store instance
mock_store = MockDocumentStore()
config = MockServerConfig()


# =============================================================================
# FASTAPI APP SETUP
# =============================================================================

app = FastAPI(
    title="Semantic Memory Service Mock API",
    description="Mock server for LeanVibe Agent Hive 2.0 Semantic Memory Service",
    version="1.0.0-mock"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def simulate_latency():
    """Simulate realistic API latency."""
    if config.enable_latency_simulation:
        latency = random.uniform(config.min_latency_ms, config.max_latency_ms) / 1000
        await asyncio.sleep(latency)


def should_simulate_error() -> bool:
    """Determine if we should simulate an error response."""
    return random.random() < config.error_rate


def generate_processing_time() -> float:
    """Generate realistic processing time."""
    return random.uniform(15.0, 250.0)


def generate_embedding_vector(dimensions: int = 1536) -> List[float]:
    """Generate a realistic embedding vector."""
    return [random.gauss(0, 0.1) for _ in range(dimensions)]


# =============================================================================
# DOCUMENT MANAGEMENT ENDPOINTS
# =============================================================================

@app.post("/api/v1/memory/ingest", response_model=DocumentIngestResponse)
async def ingest_document(request: DocumentIngestRequest):
    """Mock document ingestion endpoint."""
    await simulate_latency()
    
    if should_simulate_error():
        raise HTTPException(status_code=500, detail="Simulated ingestion error")
    
    # Create mock document
    doc_id = str(uuid.uuid4())
    embedding_id = str(uuid.uuid4())
    
    doc = MockDocument(
        document_id=doc_id,
        content=request.content,
        metadata=request.metadata.dict() if request.metadata else {},
        agent_id=request.agent_id,
        workflow_id=str(request.workflow_id) if request.workflow_id else None,
        tags=request.tags or []
    )
    
    mock_store.add_document(doc)
    
    return DocumentIngestResponse(
        document_id=uuid.UUID(doc_id),
        embedding_id=uuid.UUID(embedding_id),
        processing_time_ms=generate_processing_time(),
        vector_dimensions=1536,
        index_updated=True,
        summary="Generated summary of the document" if request.processing_options and request.processing_options.generate_summary else None,
        extracted_entities=[]
    )


@app.post("/api/v1/memory/batch-ingest", response_model=BatchIngestResponse)
async def batch_ingest_documents(request: BatchIngestRequest):
    """Mock batch document ingestion endpoint."""
    await simulate_latency()
    
    if should_simulate_error():
        raise HTTPException(status_code=500, detail="Simulated batch ingestion error")
    
    results = []
    successful = 0
    failed = 0
    
    for i, doc_request in enumerate(request.documents):
        # Simulate occasional failures in batch processing
        if random.random() < 0.05:  # 5% failure rate
            results.append({
                "index": i,
                "status": "error",
                "document_id": None,
                "error_message": "Simulated processing error"
            })
            failed += 1
        else:
            doc_id = str(uuid.uuid4())
            doc = MockDocument(
                document_id=doc_id,
                content=doc_request.content,
                metadata=doc_request.metadata.dict() if doc_request.metadata else {},
                agent_id=doc_request.agent_id,
                tags=doc_request.tags or []
            )
            mock_store.add_document(doc)
            
            results.append({
                "index": i,
                "status": "success",
                "document_id": uuid.UUID(doc_id),
                "error_message": None
            })
            successful += 1
    
    return BatchIngestResponse(
        total_documents=len(request.documents),
        successful_ingestions=successful,
        failed_ingestions=failed,
        processing_time_ms=generate_processing_time() * len(request.documents),
        results=results,
        batch_summary="Batch processing completed with mixed results" if request.batch_options and request.batch_options.generate_summary else None
    )


@app.get("/api/v1/memory/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str = Path(...),
    include_embedding: bool = Query(default=False)
):
    """Mock document retrieval endpoint."""
    await simulate_latency()
    
    doc = mock_store.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        document_id=uuid.UUID(doc.document_id),
        content=doc.content,
        metadata=doc.metadata,
        agent_id=doc.agent_id,
        workflow_id=uuid.UUID(doc.workflow_id) if doc.workflow_id else None,
        tags=doc.tags,
        embedding_vector=doc.embedding_vector if include_embedding else None,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        access_count=doc.access_count,
        last_accessed=doc.last_accessed
    )


@app.delete("/api/v1/memory/documents/{document_id}", status_code=204)
async def delete_document(document_id: str = Path(...)):
    """Mock document deletion endpoint."""
    await simulate_latency()
    
    if document_id not in mock_store.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    del mock_store.documents[document_id]
    return None


# =============================================================================
# SEMANTIC SEARCH ENDPOINTS
# =============================================================================

@app.post("/api/v1/memory/search", response_model=SemanticSearchResponse)
async def semantic_search(request: SemanticSearchRequest):
    """Mock semantic search endpoint."""
    await simulate_latency()
    
    if should_simulate_error():
        raise HTTPException(status_code=500, detail="Simulated search error")
    
    # Perform mock search
    tags = request.filters.tags if request.filters else None
    results = mock_store.search_documents(
        query=request.query,
        limit=request.limit,
        similarity_threshold=request.similarity_threshold,
        agent_id=request.agent_id,
        tags=tags
    )
    
    search_results = []
    for doc in results:
        search_results.append(SearchResult(
            document_id=uuid.UUID(doc.document_id),
            content=doc.content,
            similarity_score=doc.similarity_score,
            metadata=doc.metadata,
            agent_id=doc.agent_id,
            tags=doc.tags,
            relevance_explanation=f"Semantic similarity: {doc.similarity_score:.2f}" if request.search_options and request.search_options.explain_relevance else None,
            highlighted_content=doc.content.replace(request.query.split()[0], f"**{request.query.split()[0]}**") if request.search_options and request.search_options.include_metadata else None,
            embedding_vector=doc.embedding_vector if request.search_options and request.search_options.include_embeddings else None
        ))
    
    return SemanticSearchResponse(
        results=search_results,
        total_found=len(mock_store.documents),  # Simplified
        search_time_ms=generate_processing_time(),
        query_embedding_time_ms=random.uniform(10.0, 30.0),
        reranking_applied=request.search_options.rerank if request.search_options else False,
        suggestions=["agent coordination patterns", "distributed system design"] if not search_results else []
    )


@app.post("/api/v1/memory/similarity", response_model=SimilaritySearchResponse)
async def find_similar(request: SimilaritySearchRequest):
    """Mock similarity search endpoint."""
    await simulate_latency()
    
    if request.document_id:
        results = mock_store.get_related_documents(str(request.document_id), request.limit)
    else:
        # For embedding vector similarity, return random results
        all_docs = list(mock_store.documents.values())
        results = random.sample(all_docs, min(request.limit, len(all_docs)))
        for doc in results:
            doc.similarity_score = random.uniform(request.similarity_threshold, 0.95)
    
    search_results = []
    for doc in results:
        search_results.append(SearchResult(
            document_id=uuid.UUID(doc.document_id),
            content=doc.content,
            similarity_score=doc.similarity_score,
            metadata=doc.metadata,
            agent_id=doc.agent_id,
            tags=doc.tags
        ))
    
    return SimilaritySearchResponse(
        similar_documents=search_results,
        search_time_ms=generate_processing_time()
    )


@app.get("/api/v1/memory/related/{document_id}", response_model=RelatedDocumentsResponse)
async def get_related_documents(
    document_id: str = Path(...),
    limit: int = Query(default=5, ge=1, le=50),
    similarity_threshold: float = Query(default=0.6, ge=0.0, le=1.0),
    relationship_type: RelationshipType = Query(default=RelationshipType.SEMANTIC)
):
    """Mock related documents endpoint."""
    await simulate_latency()
    
    if document_id not in mock_store.documents:
        raise HTTPException(status_code=404, detail="Document not found")
    
    related = mock_store.get_related_documents(document_id, limit)
    
    related_docs = []
    for doc in related:
        related_docs.append(RelatedDocument(
            document_id=uuid.UUID(doc.document_id),
            content=doc.content,
            similarity_score=doc.similarity_score,
            metadata=doc.metadata,
            agent_id=doc.agent_id,
            tags=doc.tags,
            relationship_type=relationship_type,
            relationship_strength=doc.similarity_score
        ))
    
    return RelatedDocumentsResponse(
        related_documents=related_docs,
        relationship_analysis={
            "total_relationships_found": len(related_docs),
            "relationship_distribution": {relationship_type: len(related_docs)},
            "analysis_time_ms": generate_processing_time()
        }
    )


# =============================================================================
# CONTEXT MANAGEMENT ENDPOINTS
# =============================================================================

@app.post("/api/v1/memory/compress", response_model=ContextCompressionResponse)
async def compress_context(request: ContextCompressionRequest):
    """Mock context compression endpoint."""
    await simulate_latency()
    
    if should_simulate_error():
        raise HTTPException(status_code=500, detail="Simulated compression error")
    
    # Simulate compression results
    original_size = random.randint(10000, 50000)
    compressed_size = int(original_size * (1 - request.target_reduction))
    actual_ratio = (original_size - compressed_size) / original_size
    
    compressed_context_id = f"compressed_{request.context_id}_{int(time.time())}"
    
    # Store compression context for later retrieval
    mock_store.compression_contexts[compressed_context_id] = {
        "original_context_id": request.context_id,
        "compression_method": request.compression_method,
        "created_at": datetime.utcnow()
    }
    
    return ContextCompressionResponse(
        compressed_context_id=compressed_context_id,
        original_size=original_size,
        compressed_size=compressed_size,
        compression_ratio=actual_ratio,
        semantic_preservation_score=random.uniform(0.85, 0.98),
        processing_time_ms=generate_processing_time() * 2,
        compression_summary=f"Applied {request.compression_method} compression, reduced size by {actual_ratio:.1%}",
        preserved_documents=[uuid.uuid4() for _ in range(random.randint(3, 8))]
    )


@app.post("/api/v1/memory/contextualize", response_model=ContextualizationResponse)
async def generate_contextual_embeddings(request: ContextualizationRequest):
    """Mock contextualization endpoint."""
    await simulate_latency()
    
    # Generate mock contextual embedding
    contextual_embedding = generate_embedding_vector()
    
    # Generate influence scores for context documents
    influence_scores = {}
    for doc_id in request.context_documents:
        influence_scores[str(doc_id)] = random.uniform(0.1, 0.9)
    
    return ContextualizationResponse(
        contextual_embedding=contextual_embedding,
        context_influence_scores=influence_scores,
        processing_time_ms=generate_processing_time(),
        context_summary="Contextual embedding generated using attention-based method with strong influence from coordination patterns"
    )


@app.get("/api/v1/memory/agent-knowledge/{agent_id}", response_model=AgentKnowledgeResponse)
async def get_agent_knowledge(
    agent_id: str = Path(...),
    knowledge_type: str = Query(default="all"),
    time_range: TimeRange = Query(default=TimeRange.DAYS_7)
):
    """Mock agent knowledge retrieval endpoint."""
    await simulate_latency()
    
    # Generate mock knowledge base
    patterns = [
        {
            "pattern_id": f"pattern_{i}",
            "description": f"Learned pattern {i} for {agent_id}",
            "confidence": random.uniform(0.6, 0.95),
            "occurrences": random.randint(5, 50)
        }
        for i in range(random.randint(3, 8))
    ]
    
    interactions = [
        {
            "interaction_id": f"interaction_{i}",
            "timestamp": datetime.utcnow() - timedelta(hours=random.randint(1, 168)),
            "context": f"Context for interaction {i}",
            "outcome": f"Successful outcome {i}"
        }
        for i in range(random.randint(10, 30))
    ]
    
    return AgentKnowledgeResponse(
        agent_id=agent_id,
        knowledge_base={
            "patterns": patterns,
            "interactions": interactions,
            "consolidated_knowledge": {
                "key_insights": [
                    "Agent coordination improves with consistent messaging patterns",
                    "Error recovery is critical for distributed operations",
                    "Context compression preserves semantic meaning effectively"
                ],
                "expertise_areas": ["coordination", "messaging", "optimization"],
                "learned_preferences": {
                    "communication_style": "structured",
                    "error_handling": "proactive",
                    "optimization_focus": "memory_efficiency"
                }
            }
        },
        last_updated=datetime.utcnow() - timedelta(minutes=random.randint(5, 120)),
        knowledge_stats={
            "total_documents": random.randint(50, 500),
            "unique_patterns": len(patterns),
            "interaction_count": len(interactions),
            "knowledge_confidence": random.uniform(0.7, 0.95)
        }
    )


# =============================================================================
# SYSTEM OPERATIONS ENDPOINTS
# =============================================================================

@app.get("/api/v1/memory/health", response_model=HealthResponse)
async def get_health():
    """Mock health check endpoint."""
    await asyncio.sleep(0.005)  # Very fast health check
    
    # Simulate occasional degraded status
    status = HealthStatus.HEALTHY
    if random.random() < 0.1:  # 10% chance of degraded
        status = HealthStatus.DEGRADED
    
    return HealthResponse(
        status=status,
        timestamp=datetime.utcnow(),
        version="1.0.0-mock",
        components={
            "database": {
                "status": ComponentStatus.HEALTHY,
                "response_time_ms": random.uniform(2.0, 8.0),
                "connection_pool_size": 10
            },
            "vector_index": {
                "status": ComponentStatus.HEALTHY,
                "documents_indexed": len(mock_store.documents),
                "index_size_mb": random.uniform(100.0, 500.0),
                "last_rebuild": datetime.utcnow() - timedelta(hours=random.randint(1, 24))
            },
            "memory_usage": {
                "heap_used_mb": random.uniform(200.0, 800.0),
                "heap_max_mb": 2048.0,
                "gc_collections": random.randint(10, 100)
            }
        },
        performance_metrics={
            "avg_search_time_ms": random.uniform(50.0, 150.0),
            "avg_ingestion_time_ms": random.uniform(20.0, 80.0),
            "throughput_docs_per_sec": random.uniform(500.0, 1200.0),
            "error_rate_percent": random.uniform(0.0, 2.0)
        }
    )


@app.get("/api/v1/memory/metrics")
async def get_metrics(
    format: MetricsFormat = Query(default=MetricsFormat.JSON),
    time_range: TimeRange = Query(default=TimeRange.HOUR_1)
):
    """Mock metrics endpoint."""
    await simulate_latency()
    
    if format == MetricsFormat.PROMETHEUS:
        # Return Prometheus format metrics
        metrics = f"""# HELP semantic_memory_search_duration_seconds Time spent on search operations
# TYPE semantic_memory_search_duration_seconds histogram
semantic_memory_search_duration_seconds_bucket{{le="0.1"}} {random.randint(1000, 2000)}
semantic_memory_search_duration_seconds_bucket{{le="0.2"}} {random.randint(2000, 4000)}
semantic_memory_search_duration_seconds_sum {random.uniform(200.0, 500.0)}
semantic_memory_search_duration_seconds_count {random.randint(2000, 5000)}

# HELP semantic_memory_ingestion_total Total number of documents ingested
# TYPE semantic_memory_ingestion_total counter
semantic_memory_ingestion_total {random.randint(10000, 50000)}

# HELP semantic_memory_documents_total Total documents in the system
# TYPE semantic_memory_documents_total gauge
semantic_memory_documents_total {len(mock_store.documents)}
"""
        return PlainTextResponse(content=metrics, media_type="text/plain")
    
    else:
        # Return JSON format metrics
        return MetricsResponse(
            timestamp=datetime.utcnow(),
            time_range=time_range,
            performance_metrics={
                "search_operations": {
                    "total_count": random.randint(1000, 5000),
                    "avg_duration_ms": random.uniform(80.0, 120.0),
                    "p95_duration_ms": random.uniform(150.0, 200.0),
                    "p99_duration_ms": random.uniform(200.0, 300.0),
                    "error_count": random.randint(0, 50)
                },
                "ingestion_operations": {
                    "total_count": random.randint(500, 2000),
                    "avg_duration_ms": random.uniform(40.0, 80.0),
                    "throughput_docs_per_sec": random.uniform(500.0, 1000.0),
                    "error_count": random.randint(0, 20)
                },
                "compression_operations": {
                    "total_count": random.randint(50, 200),
                    "avg_compression_ratio": random.uniform(0.6, 0.8),
                    "avg_processing_time_ms": random.uniform(200.0, 500.0)
                }
            },
            resource_metrics={
                "memory_usage_mb": random.uniform(400.0, 1200.0),
                "cpu_usage_percent": random.uniform(20.0, 80.0),
                "disk_usage_mb": random.uniform(1000.0, 5000.0),
                "network_io_mb": random.uniform(100.0, 500.0)
            },
            business_metrics={
                "total_documents": len(mock_store.documents),
                "total_agents": random.randint(5, 20),
                "active_workflows": random.randint(10, 50),
                "knowledge_base_size_mb": random.uniform(500.0, 2000.0)
            }
        )


@app.post("/api/v1/memory/rebuild-index", response_model=IndexRebuildResponse)
async def rebuild_index(request: IndexRebuildRequest):
    """Mock index rebuild endpoint."""
    await simulate_latency()
    
    if should_simulate_error():
        raise HTTPException(status_code=500, detail="Simulated rebuild error")
    
    # Check if rebuild is already in progress
    active_rebuilds = [r for r in mock_store.rebuild_operations.values() 
                      if r["status"] in [IndexStatus.QUEUED, IndexStatus.IN_PROGRESS]]
    
    if active_rebuilds:
        raise HTTPException(status_code=409, detail="Rebuild already in progress")
    
    rebuild_id = str(uuid.uuid4())
    estimated_duration = random.randint(10, 60)  # 10-60 minutes
    
    # Store rebuild operation
    mock_store.rebuild_operations[rebuild_id] = {
        "rebuild_id": rebuild_id,
        "status": IndexStatus.QUEUED,
        "rebuild_type": request.rebuild_type,
        "started_at": datetime.utcnow(),
        "estimated_duration": estimated_duration
    }
    
    return IndexRebuildResponse(
        rebuild_id=uuid.UUID(rebuild_id),
        status=IndexStatus.QUEUED,
        estimated_duration_minutes=estimated_duration,
        progress_url=f"/api/v1/memory/rebuild-index/{rebuild_id}/progress"
    )


# =============================================================================
# ADDITIONAL HELPER ENDPOINTS
# =============================================================================

@app.get("/api/v1/memory/rebuild-index/{rebuild_id}/progress")
async def get_rebuild_progress(rebuild_id: str = Path(...)):
    """Get rebuild progress (additional helper endpoint)."""
    await simulate_latency()
    
    rebuild_info = mock_store.rebuild_operations.get(rebuild_id)
    if not rebuild_info:
        raise HTTPException(status_code=404, detail="Rebuild operation not found")
    
    # Simulate progress
    elapsed_minutes = (datetime.utcnow() - rebuild_info["started_at"]).total_seconds() / 60
    progress = min(100, int((elapsed_minutes / rebuild_info["estimated_duration"]) * 100))
    
    if progress >= 100:
        status = IndexStatus.COMPLETED
    elif progress > 0:
        status = IndexStatus.IN_PROGRESS
    else:
        status = IndexStatus.QUEUED
    
    return {
        "rebuild_id": rebuild_id,
        "status": status,
        "progress_percent": progress,
        "estimated_completion": rebuild_info["started_at"] + timedelta(minutes=rebuild_info["estimated_duration"])
    }


@app.get("/config")
async def get_mock_config():
    """Get current mock server configuration."""
    return {
        "enable_latency_simulation": config.enable_latency_simulation,
        "min_latency_ms": config.min_latency_ms,
        "max_latency_ms": config.max_latency_ms,
        "error_rate": config.error_rate,
        "enable_realistic_data": config.enable_realistic_data,
        "total_documents": len(mock_store.documents),
        "server_info": "LeanVibe Semantic Memory Service Mock Server v1.0.0"
    }


@app.post("/config")
async def update_mock_config(new_config: dict):
    """Update mock server configuration."""
    global config
    
    if "enable_latency_simulation" in new_config:
        config.enable_latency_simulation = new_config["enable_latency_simulation"]
    if "min_latency_ms" in new_config:
        config.min_latency_ms = new_config["min_latency_ms"]
    if "max_latency_ms" in new_config:
        config.max_latency_ms = new_config["max_latency_ms"]
    if "error_rate" in new_config:
        config.error_rate = max(0.0, min(1.0, new_config["error_rate"]))
    
    return {"message": "Configuration updated successfully", "new_config": new_config}


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper error response format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=f"http_{exc.status_code}",
            message=exc.detail,
            timestamp=datetime.utcnow(),
            request_id=f"mock_{int(time.time())}"
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred in the mock server",
            details={"exception_type": type(exc).__name__},
            timestamp=datetime.utcnow(),
            request_id=f"mock_{int(time.time())}"
        ).dict()
    )


# =============================================================================
# STARTUP AND SHUTDOWN EVENTS
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize mock server on startup."""
    logger.info("🚀 Starting Semantic Memory Service Mock Server")
    logger.info(f"📊 Initialized with {len(mock_store.documents)} sample documents")
    logger.info("🔧 Configuration loaded successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("🛑 Shutting down Semantic Memory Service Mock Server")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "mock_servers.semantic_memory_mock:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )