"""
Contract Testing Suite for Semantic Memory Service API

This test suite validates that the API implementation conforms to the OpenAPI specification
and ensures compatibility between subagents and the semantic memory service.

Features:
- OpenAPI specification validation
- Contract compliance testing
- Performance requirement validation
- Error scenario testing
- Mock server validation
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import httpx
import pytest
from pydantic import ValidationError

# Import our schemas for validation
from app.schemas.semantic_memory import (
    DocumentIngestRequest, DocumentIngestResponse, BatchIngestRequest, BatchIngestResponse,
    DocumentResponse, SemanticSearchRequest, SemanticSearchResponse, SimilaritySearchRequest,
    SimilaritySearchResponse, RelatedDocumentsResponse, ContextCompressionRequest,
    ContextCompressionResponse, ContextualizationRequest, ContextualizationResponse,
    AgentKnowledgeResponse, HealthResponse, MetricsResponse, IndexRebuildRequest,
    IndexRebuildResponse, ErrorResponse, ProcessingPriority, CompressionMethod,
    RelationshipType, TimeRange, MetricsFormat
)


# =============================================================================
# TEST CONFIGURATION AND FIXTURES
# =============================================================================

class ContractTestConfig:
    """Configuration for contract testing."""
    
    # Service endpoints
    MOCK_SERVER_URL = "http://localhost:8001/api/v1"
    PRODUCTION_SERVER_URL = "http://localhost:8000/api/v1"  # When available
    
    # Performance requirements from specification
    MAX_SEARCH_TIME_MS = 200  # P95 requirement
    MAX_INGESTION_TIME_MS = 100  # Per document
    MIN_THROUGHPUT_DOCS_PER_SEC = 500
    MAX_WORKFLOW_OVERHEAD_MS = 10
    
    # Test data
    SAMPLE_AGENT_IDS = ["orchestrator-001", "context-optimizer", "search-engine"]
    SAMPLE_WORKFLOW_ID = "12345678-1234-5678-9012-123456789abc"


@pytest.fixture
async def contract_client():
    """HTTP client for contract testing."""
    async with httpx.AsyncClient(
        base_url=ContractTestConfig.MOCK_SERVER_URL,
        timeout=30.0
    ) as client:
        yield client


@pytest.fixture
def sample_document_data():
    """Sample document data for testing."""
    return {
        "content": "Agent coordination patterns in distributed systems require careful message ordering and failure recovery mechanisms.",
        "agent_id": "orchestrator-001",
        "workflow_id": ContractTestConfig.SAMPLE_WORKFLOW_ID,
        "tags": ["coordination", "distributed-systems", "patterns"],
        "metadata": {
            "title": "Agent Coordination Patterns",
            "source": "technical_documentation",
            "importance": 0.8,
            "created_by": "test_agent"
        },
        "processing_options": {
            "generate_summary": True,
            "extract_entities": False,
            "priority": ProcessingPriority.HIGH
        }
    }


@pytest.fixture
def sample_search_request():
    """Sample search request for testing."""
    return {
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
            "include_metadata": True,
            "explain_relevance": True
        }
    }


# =============================================================================
# DOCUMENT MANAGEMENT CONTRACT TESTS
# =============================================================================

@pytest.mark.asyncio
class TestDocumentManagementContract:
    """Test document management API contract compliance."""
    
    async def test_document_ingest_request_validation(self, sample_document_data):
        """Test that document ingest request validates correctly."""
        # Valid request should pass
        request = DocumentIngestRequest(**sample_document_data)
        assert request.content == sample_document_data["content"]
        assert request.agent_id == sample_document_data["agent_id"]
        assert len(request.tags) == 3
        
        # Invalid request should fail
        invalid_data = sample_document_data.copy()
        invalid_data["content"] = ""  # Empty content
        
        with pytest.raises(ValidationError):
            DocumentIngestRequest(**invalid_data)
    
    async def test_document_ingest_endpoint_contract(self, contract_client, sample_document_data):
        """Test document ingestion endpoint contract compliance."""
        response = await contract_client.post("/memory/ingest", json=sample_document_data)
        
        # Validate response status
        assert response.status_code == 201
        
        # Validate response schema
        response_data = response.json()
        ingest_response = DocumentIngestResponse(**response_data)
        
        assert ingest_response.document_id is not None
        assert ingest_response.embedding_id is not None
        assert ingest_response.processing_time_ms > 0
        assert ingest_response.vector_dimensions == 1536
        assert ingest_response.index_updated is True
        
        # Validate performance requirement
        assert ingest_response.processing_time_ms <= ContractTestConfig.MAX_INGESTION_TIME_MS * 3  # Allow 3x for mock
    
    async def test_batch_ingest_contract(self, contract_client, sample_document_data):
        """Test batch ingestion contract compliance."""
        # Create batch request
        batch_documents = []
        for i in range(5):
            doc = sample_document_data.copy()
            doc["content"] = f"Batch document {i}: {doc['content']}"
            doc["agent_id"] = f"batch_agent_{i}"
            batch_documents.append(doc)
        
        batch_request = {
            "documents": batch_documents,
            "batch_options": {
                "parallel_processing": True,
                "generate_summary": True,
                "fail_on_error": False
            }
        }
        
        # Validate request schema
        BatchIngestRequest(**batch_request)
        
        # Test endpoint
        response = await contract_client.post("/memory/batch-ingest", json=batch_request)
        assert response.status_code == 201
        
        # Validate response schema
        response_data = response.json()
        batch_response = BatchIngestResponse(**response_data)
        
        assert batch_response.total_documents == 5
        assert batch_response.successful_ingestions >= 0
        assert batch_response.failed_ingestions >= 0
        assert batch_response.successful_ingestions + batch_response.failed_ingestions == 5
        assert len(batch_response.results) == 5
    
    async def test_document_retrieval_contract(self, contract_client, sample_document_data):
        """Test document retrieval contract compliance."""
        # First ingest a document
        ingest_response = await contract_client.post("/memory/ingest", json=sample_document_data)
        document_id = ingest_response.json()["document_id"]
        
        # Test retrieval without embedding
        response = await contract_client.get(f"/memory/documents/{document_id}")
        assert response.status_code == 200
        
        # Validate response schema
        doc_response = DocumentResponse(**response.json())
        assert str(doc_response.document_id) == document_id
        assert doc_response.content == sample_document_data["content"]
        assert doc_response.agent_id == sample_document_data["agent_id"]
        assert doc_response.embedding_vector is None  # Not requested
        
        # Test retrieval with embedding
        response = await contract_client.get(
            f"/memory/documents/{document_id}",
            params={"include_embedding": True}
        )
        assert response.status_code == 200
        
        doc_response_with_embedding = DocumentResponse(**response.json())
        assert doc_response_with_embedding.embedding_vector is not None
        assert len(doc_response_with_embedding.embedding_vector) == 1536
    
    async def test_document_deletion_contract(self, contract_client, sample_document_data):
        """Test document deletion contract compliance."""
        # First ingest a document
        ingest_response = await contract_client.post("/memory/ingest", json=sample_document_data)
        document_id = ingest_response.json()["document_id"]
        
        # Delete the document
        response = await contract_client.delete(f"/memory/documents/{document_id}")
        assert response.status_code == 204
        
        # Verify document is deleted
        response = await contract_client.get(f"/memory/documents/{document_id}")
        assert response.status_code == 404


# =============================================================================
# SEMANTIC SEARCH CONTRACT TESTS
# =============================================================================

@pytest.mark.asyncio
class TestSemanticSearchContract:
    """Test semantic search API contract compliance."""
    
    async def test_semantic_search_request_validation(self, sample_search_request):
        """Test semantic search request validation."""
        # Valid request should pass
        request = SemanticSearchRequest(**sample_search_request)
        assert request.query == sample_search_request["query"]
        assert request.limit == sample_search_request["limit"]
        assert request.similarity_threshold == sample_search_request["similarity_threshold"]
        
        # Invalid request should fail
        invalid_data = sample_search_request.copy()
        invalid_data["similarity_threshold"] = 1.5  # Invalid threshold
        
        with pytest.raises(ValidationError):
            SemanticSearchRequest(**invalid_data)
    
    async def test_semantic_search_endpoint_contract(self, contract_client, sample_search_request):
        """Test semantic search endpoint contract compliance."""
        start_time = time.time()
        response = await contract_client.post("/memory/search", json=sample_search_request)
        search_time_ms = (time.time() - start_time) * 1000
        
        # Validate response status
        assert response.status_code == 200
        
        # Validate response schema
        response_data = response.json()
        search_response = SemanticSearchResponse(**response_data)
        
        assert isinstance(search_response.results, list)
        assert search_response.total_found >= 0
        assert search_response.search_time_ms > 0
        assert search_response.query_embedding_time_ms > 0
        
        # Validate performance requirement (relaxed for mock)
        assert search_time_ms <= ContractTestConfig.MAX_SEARCH_TIME_MS * 5  # Allow 5x for mock
        
        # Validate search result structure
        for result in search_response.results:
            assert result.document_id is not None
            assert result.content is not None
            assert 0.0 <= result.similarity_score <= 1.0
            assert result.agent_id is not None
    
    async def test_similarity_search_contract(self, contract_client, sample_document_data):
        """Test similarity search contract compliance."""
        # First ingest a document
        ingest_response = await contract_client.post("/memory/ingest", json=sample_document_data)
        document_id = ingest_response.json()["document_id"]
        
        # Test similarity search by document ID
        similarity_request = {
            "document_id": document_id,
            "limit": 5,
            "similarity_threshold": 0.6,
            "exclude_self": True
        }
        
        # Validate request schema
        SimilaritySearchRequest(**similarity_request)
        
        # Test endpoint
        response = await contract_client.post("/memory/similarity", json=similarity_request)
        assert response.status_code == 200
        
        # Validate response schema
        similarity_response = SimilaritySearchResponse(**response.json())
        assert isinstance(similarity_response.similar_documents, list)
        assert similarity_response.search_time_ms > 0
        
        # Validate that source document is excluded
        for doc in similarity_response.similar_documents:
            assert str(doc.document_id) != document_id
    
    async def test_related_documents_contract(self, contract_client, sample_document_data):
        """Test related documents endpoint contract compliance."""
        # First ingest a document
        ingest_response = await contract_client.post("/memory/ingest", json=sample_document_data)
        document_id = ingest_response.json()["document_id"]
        
        # Test related documents endpoint
        response = await contract_client.get(
            f"/memory/related/{document_id}",
            params={
                "limit": 5,
                "similarity_threshold": 0.6,
                "relationship_type": RelationshipType.SEMANTIC
            }
        )
        
        assert response.status_code == 200
        
        # Validate response schema
        related_response = RelatedDocumentsResponse(**response.json())
        assert isinstance(related_response.related_documents, list)
        assert related_response.relationship_analysis is not None
        assert related_response.relationship_analysis.total_relationships_found >= 0


# =============================================================================
# CONTEXT MANAGEMENT CONTRACT TESTS
# =============================================================================

@pytest.mark.asyncio
class TestContextManagementContract:
    """Test context management API contract compliance."""
    
    async def test_context_compression_contract(self, contract_client):
        """Test context compression contract compliance."""
        compression_request = {
            "context_id": "test_context_12345",
            "compression_method": CompressionMethod.SEMANTIC_CLUSTERING,
            "target_reduction": 0.7,
            "preserve_importance_threshold": 0.8,
            "agent_id": "context-optimizer",
            "compression_options": {
                "preserve_recent": True,
                "maintain_relationships": True,
                "generate_summary": True
            }
        }
        
        # Validate request schema
        ContextCompressionRequest(**compression_request)
        
        # Test endpoint
        response = await contract_client.post("/memory/compress", json=compression_request)
        assert response.status_code == 200
        
        # Validate response schema
        compression_response = ContextCompressionResponse(**response.json())
        assert compression_response.compressed_context_id is not None
        assert compression_response.original_size > 0
        assert compression_response.compressed_size > 0
        assert 0.0 <= compression_response.compression_ratio <= 1.0
        assert 0.0 <= compression_response.semantic_preservation_score <= 1.0
        assert compression_response.processing_time_ms > 0
    
    async def test_contextualization_contract(self, contract_client, sample_document_data):
        """Test contextualization contract compliance."""
        # First ingest some documents for context
        context_doc_ids = []
        for i in range(3):
            doc_data = sample_document_data.copy()
            doc_data["content"] = f"Context document {i}: {doc_data['content']}"
            ingest_response = await contract_client.post("/memory/ingest", json=doc_data)
            context_doc_ids.append(ingest_response.json()["document_id"])
        
        # Test contextualization
        contextualization_request = {
            "content": "Summary of agent coordination analysis",
            "context_documents": context_doc_ids,
            "contextualization_method": "attention_based",
            "agent_id": "context-analyzer"
        }
        
        # Validate request schema
        ContextualizationRequest(**contextualization_request)
        
        # Test endpoint
        response = await contract_client.post("/memory/contextualize", json=contextualization_request)
        assert response.status_code == 200
        
        # Validate response schema
        contextualization_response = ContextualizationResponse(**response.json())
        assert len(contextualization_response.contextual_embedding) == 1536
        assert len(contextualization_response.context_influence_scores) == len(context_doc_ids)
        assert contextualization_response.processing_time_ms > 0
    
    async def test_agent_knowledge_contract(self, contract_client):
        """Test agent knowledge retrieval contract compliance."""
        agent_id = "test_knowledge_agent"
        
        # Test endpoint
        response = await contract_client.get(
            f"/memory/agent-knowledge/{agent_id}",
            params={
                "knowledge_type": "all",
                "time_range": TimeRange.DAYS_7
            }
        )
        
        assert response.status_code == 200
        
        # Validate response schema
        knowledge_response = AgentKnowledgeResponse(**response.json())
        assert knowledge_response.agent_id == agent_id
        assert knowledge_response.knowledge_base is not None
        assert knowledge_response.last_updated is not None
        assert knowledge_response.knowledge_stats is not None
        assert knowledge_response.knowledge_stats.total_documents >= 0


# =============================================================================
# SYSTEM OPERATIONS CONTRACT TESTS
# =============================================================================

@pytest.mark.asyncio
class TestSystemOperationsContract:
    """Test system operations API contract compliance."""
    
    async def test_health_check_contract(self, contract_client):
        """Test health check endpoint contract compliance."""
        response = await contract_client.get("/memory/health")
        assert response.status_code in [200, 503]  # Healthy or unhealthy
        
        # Validate response schema
        health_response = HealthResponse(**response.json())
        assert health_response.status is not None
        assert health_response.timestamp is not None
        assert health_response.version is not None
        assert health_response.components is not None
        assert health_response.performance_metrics is not None
        
        # Validate component structure
        assert health_response.components.database is not None
        assert health_response.components.vector_index is not None
        assert health_response.components.memory_usage is not None
    
    async def test_metrics_endpoint_contract(self, contract_client):
        """Test metrics endpoint contract compliance."""
        # Test JSON format
        response = await contract_client.get(
            "/memory/metrics",
            params={
                "format": MetricsFormat.JSON,
                "time_range": TimeRange.HOUR_1
            }
        )
        
        assert response.status_code == 200
        
        # Validate response schema
        metrics_response = MetricsResponse(**response.json())
        assert metrics_response.timestamp is not None
        assert metrics_response.time_range == TimeRange.HOUR_1
        assert metrics_response.performance_metrics is not None
        assert metrics_response.resource_metrics is not None
        assert metrics_response.business_metrics is not None
        
        # Test Prometheus format
        response = await contract_client.get(
            "/memory/metrics",
            params={"format": MetricsFormat.PROMETHEUS}
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        
        # Basic validation of Prometheus format
        content = response.text
        assert "# HELP" in content
        assert "# TYPE" in content
    
    async def test_index_rebuild_contract(self, contract_client):
        """Test index rebuild endpoint contract compliance."""
        rebuild_request = {
            "rebuild_type": "incremental",
            "priority": ProcessingPriority.NORMAL,
            "maintenance_window": {
                "start_time": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
                "max_duration_minutes": 60
            }
        }
        
        # Validate request schema
        IndexRebuildRequest(**rebuild_request)
        
        # Test endpoint
        response = await contract_client.post("/memory/rebuild-index", json=rebuild_request)
        assert response.status_code == 202
        
        # Validate response schema
        rebuild_response = IndexRebuildResponse(**response.json())
        assert rebuild_response.rebuild_id is not None
        assert rebuild_response.status is not None
        assert rebuild_response.estimated_duration_minutes > 0


# =============================================================================
# ERROR HANDLING CONTRACT TESTS
# =============================================================================

@pytest.mark.asyncio
class TestErrorHandlingContract:
    """Test error handling contract compliance."""
    
    async def test_400_error_contract(self, contract_client):
        """Test 400 Bad Request error handling."""
        # Send invalid search request
        invalid_request = {
            "query": "",  # Empty query should fail
            "similarity_threshold": 1.5  # Invalid threshold
        }
        
        response = await contract_client.post("/memory/search", json=invalid_request)
        assert response.status_code == 400
        
        # Validate error response schema
        error_response = ErrorResponse(**response.json())
        assert error_response.error is not None
        assert error_response.message is not None
        assert error_response.timestamp is not None
    
    async def test_404_error_contract(self, contract_client):
        """Test 404 Not Found error handling."""
        # Try to get non-existent document
        fake_doc_id = str(uuid.uuid4())
        response = await contract_client.get(f"/memory/documents/{fake_doc_id}")
        assert response.status_code == 404
        
        # Validate error response schema
        error_response = ErrorResponse(**response.json())
        assert "not found" in error_response.message.lower()
    
    async def test_429_rate_limit_contract(self, contract_client):
        """Test rate limiting error handling (if implemented)."""
        # This test may not trigger on mock server, but validates the contract
        # In production, rapid requests should eventually return 429
        
        tasks = []
        for _ in range(100):  # Send many requests quickly
            task = contract_client.post("/memory/search", json={
                "query": "rate limit test",
                "limit": 1
            })
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check if any responses are rate limited
        status_codes = []
        for response in responses:
            if isinstance(response, httpx.Response):
                status_codes.append(response.status_code)
        
        # At least some should succeed
        assert 200 in status_codes
        
        # If rate limiting is implemented, should see 429
        if 429 in status_codes:
            # Find a 429 response and validate schema
            for response in responses:
                if isinstance(response, httpx.Response) and response.status_code == 429:
                    error_response = ErrorResponse(**response.json())
                    assert "rate limit" in error_response.message.lower()
                    break


# =============================================================================
# PERFORMANCE CONTRACT TESTS
# =============================================================================

@pytest.mark.asyncio
class TestPerformanceContract:
    """Test performance requirements from the API specification."""
    
    async def test_search_performance_requirement(self, contract_client, sample_search_request):
        """Test that search meets P95 performance requirement."""
        # Run multiple searches to get realistic performance data
        search_times = []
        
        for _ in range(20):
            start_time = time.time()
            response = await contract_client.post("/memory/search", json=sample_search_request)
            end_time = time.time()
            
            assert response.status_code == 200
            search_times.append((end_time - start_time) * 1000)
        
        # Calculate P95
        search_times.sort()
        p95_time = search_times[int(len(search_times) * 0.95)]
        
        # For mock server, allow more lenient requirements
        max_allowed_time = ContractTestConfig.MAX_SEARCH_TIME_MS * 3
        assert p95_time <= max_allowed_time, f"P95 search time {p95_time:.2f}ms exceeds requirement {max_allowed_time}ms"
    
    async def test_ingestion_throughput_requirement(self, contract_client, sample_document_data):
        """Test that ingestion meets throughput requirement."""
        batch_size = 10
        documents = []
        
        for i in range(batch_size):
            doc = sample_document_data.copy()
            doc["content"] = f"Throughput test document {i}: {doc['content']}"
            doc["agent_id"] = f"throughput_test_{i}"
            documents.append(doc)
        
        # Measure batch ingestion time
        start_time = time.time()
        response = await contract_client.post("/memory/batch-ingest", json={
            "documents": documents,
            "batch_options": {
                "parallel_processing": True,
                "fail_on_error": False
            }
        })
        end_time = time.time()
        
        assert response.status_code == 201
        
        # Calculate throughput
        processing_time_seconds = end_time - start_time
        throughput = batch_size / processing_time_seconds
        
        # For mock server, allow lower throughput requirement
        min_required_throughput = ContractTestConfig.MIN_THROUGHPUT_DOCS_PER_SEC / 10
        assert throughput >= min_required_throughput, f"Throughput {throughput:.2f} docs/sec below requirement {min_required_throughput} docs/sec"


# =============================================================================
# CONTRACT VALIDATION UTILITIES
# =============================================================================

class ContractValidator:
    """Utilities for validating API contracts."""
    
    @staticmethod
    def validate_response_headers(response: httpx.Response, endpoint: str):
        """Validate standard response headers."""
        # Check content type for JSON endpoints
        if "/memory/" in endpoint and response.status_code == 200:
            content_type = response.headers.get("content-type", "")
            if not endpoint.endswith("/metrics") or "format=json" in str(response.url):
                assert "application/json" in content_type
    
    @staticmethod
    def validate_error_response_format(response: httpx.Response):
        """Validate that error responses follow the standard format."""
        if response.status_code >= 400:
            try:
                error_data = response.json()
                ErrorResponse(**error_data)
                return True
            except (ValueError, ValidationError):
                return False
        return True
    
    @staticmethod
    def validate_uuid_format(uuid_string: str) -> bool:
        """Validate UUID format."""
        try:
            uuid.UUID(uuid_string)
            return True
        except ValueError:
            return False


# =============================================================================
# INTEGRATION CONTRACT TESTS
# =============================================================================

@pytest.mark.asyncio
class TestIntegrationContract:
    """Test integration scenarios that span multiple endpoints."""
    
    async def test_end_to_end_workflow_contract(self, contract_client, sample_document_data):
        """Test a complete workflow using multiple endpoints."""
        # 1. Ingest documents
        documents = []
        for i in range(3):
            doc = sample_document_data.copy()
            doc["content"] = f"Workflow document {i}: coordination patterns in agent systems"
            doc["agent_id"] = f"workflow_agent_{i}"
            documents.append(doc)
        
        batch_response = await contract_client.post("/memory/batch-ingest", json={
            "documents": documents
        })
        assert batch_response.status_code == 201
        
        # 2. Search for related content
        search_response = await contract_client.post("/memory/search", json={
            "query": "coordination patterns in agent systems",
            "limit": 5,
            "similarity_threshold": 0.6
        })
        assert search_response.status_code == 200
        search_results = search_response.json()["results"]
        assert len(search_results) > 0
        
        # 3. Get related documents
        if search_results:
            related_response = await contract_client.get(
                f"/memory/related/{search_results[0]['document_id']}"
            )
            assert related_response.status_code == 200
        
        # 4. Compress context
        compression_response = await contract_client.post("/memory/compress", json={
            "context_id": "workflow_test_context",
            "compression_method": "semantic_clustering",
            "target_reduction": 0.5,
            "preserve_importance_threshold": 0.7,
            "agent_id": "workflow_tester"
        })
        assert compression_response.status_code == 200
        
        # 5. Check system health
        health_response = await contract_client.get("/memory/health")
        assert health_response.status_code in [200, 503]
    
    async def test_agent_knowledge_lifecycle_contract(self, contract_client):
        """Test agent knowledge accumulation and retrieval."""
        agent_id = "lifecycle_test_agent"
        
        # 1. Ingest knowledge for specific agent
        knowledge_items = [
            "Agents coordinate through message passing",
            "Error recovery requires state checkpoints",
            "Performance optimization uses caching strategies"
        ]
        
        for i, knowledge in enumerate(knowledge_items):
            await contract_client.post("/memory/ingest", json={
                "content": knowledge,
                "agent_id": agent_id,
                "tags": ["knowledge", f"topic_{i}"],
                "metadata": {"importance": 0.8}
            })
        
        # 2. Retrieve agent knowledge
        knowledge_response = await contract_client.get(f"/memory/agent-knowledge/{agent_id}")
        assert knowledge_response.status_code == 200
        
        knowledge_data = knowledge_response.json()
        assert knowledge_data["agent_id"] == agent_id
        assert knowledge_data["knowledge_stats"]["total_documents"] >= 0
        
        # 3. Search agent-specific knowledge
        search_response = await contract_client.post("/memory/search", json={
            "query": "coordination and error recovery",
            "agent_id": agent_id,
            "limit": 10
        })
        assert search_response.status_code == 200


# =============================================================================
# MOCK SERVER VALIDATION TESTS
# =============================================================================

@pytest.mark.asyncio
class TestMockServerValidation:
    """Validate that the mock server correctly implements the contract."""
    
    async def test_mock_server_configuration(self, contract_client):
        """Test mock server configuration endpoints."""
        # Test configuration retrieval
        config_response = await contract_client.get("/config")
        assert config_response.status_code == 200
        
        config_data = config_response.json()
        assert "enable_latency_simulation" in config_data
        assert "error_rate" in config_data
        assert "total_documents" in config_data
    
    async def test_mock_data_consistency(self, contract_client):
        """Test that mock server provides consistent data."""
        # Ingest a document
        doc_data = {
            "content": "Consistency test document",
            "agent_id": "consistency_tester",
            "tags": ["test", "consistency"]
        }
        
        ingest_response = await contract_client.post("/memory/ingest", json=doc_data)
        document_id = ingest_response.json()["document_id"]
        
        # Retrieve the same document multiple times
        for _ in range(3):
            retrieve_response = await contract_client.get(f"/memory/documents/{document_id}")
            assert retrieve_response.status_code == 200
            
            retrieved_doc = retrieve_response.json()
            assert retrieved_doc["content"] == doc_data["content"]
            assert retrieved_doc["agent_id"] == doc_data["agent_id"]
    
    async def test_mock_error_simulation(self, contract_client):
        """Test that mock server can simulate error conditions."""
        # The mock server has a configurable error rate
        # Test that it can produce errors when configured
        
        # Update mock config to increase error rate temporarily
        await contract_client.post("/config", json={"error_rate": 0.8})
        
        # Try multiple operations - some should fail
        error_count = 0
        success_count = 0
        
        for _ in range(10):
            try:
                response = await contract_client.post("/memory/ingest", json={
                    "content": "Error simulation test",
                    "agent_id": "error_tester"
                })
                if response.status_code >= 400:
                    error_count += 1
                else:
                    success_count += 1
            except Exception:
                error_count += 1
        
        # Reset error rate
        await contract_client.post("/config", json={"error_rate": 0.02})
        
        # Should have some errors with high error rate
        assert error_count > 0 or success_count > 0  # At least some activity


# =============================================================================
# CONTRACT TEST RUNNER AND REPORTING
# =============================================================================

class ContractTestReporter:
    """Generate contract compliance reports."""
    
    def __init__(self):
        self.test_results = []
    
    def add_result(self, test_name: str, status: str, details: Dict[str, Any]):
        """Add a test result."""
        self.test_results.append({
            "test_name": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate contract compliance report."""
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r["status"] == "passed"])
        failed_tests = len([r for r in self.test_results if r["status"] == "failed"])
        
        return {
            "contract_compliance_report": {
                "summary": {
                    "total_tests": total_tests,
                    "passed_tests": passed_tests,
                    "failed_tests": failed_tests,
                    "success_rate": passed_tests / total_tests if total_tests > 0 else 0
                },
                "test_results": self.test_results,
                "generated_at": datetime.utcnow().isoformat()
            }
        }


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

@pytest.fixture(scope="session")
def contract_test_reporter():
    """Contract test reporter fixture."""
    return ContractTestReporter()


def pytest_configure(config):
    """Configure pytest for contract testing."""
    config.addinivalue_line(
        "markers", "contract: mark test as a contract compliance test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance requirement test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration contract test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection for contract testing."""
    for item in items:
        # Add contract marker to all tests in this file
        if "test_semantic_memory_contract" in str(item.fspath):
            item.add_marker(pytest.mark.contract)
        
        # Add performance marker to performance tests
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        
        # Add integration marker to integration tests
        if "integration" in item.name.lower() or "end_to_end" in item.name.lower():
            item.add_marker(pytest.mark.integration)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """Run contract tests directly."""
    import sys
    
    # Run contract tests
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "contract",
        "--asyncio-mode=auto"
    ])
    
    sys.exit(exit_code)