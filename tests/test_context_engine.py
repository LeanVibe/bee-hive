"""
Comprehensive test suite for Context Engine - Long-Term Memory System.

Tests cover semantic memory storage, vector search, context compression,
and cross-agent knowledge sharing with performance validation.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.embeddings import EmbeddingService
from app.core.vector_search import VectorSearchEngine
from app.core.context_compression import ContextCompressor
from app.core.context_manager import ContextManager
from app.models.context import Context, ContextType
from app.schemas.context import ContextCreate, ContextSearchRequest


class TestEmbeddingService:
    """Test suite for embedding generation service."""

    @pytest.fixture
    async def embedding_service(self):
        """Create embedding service instance for testing."""
        return EmbeddingService(model_name="text-embedding-ada-002")

    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, embedding_service):
        """Test successful embedding generation."""
        with patch.object(embedding_service, 'client') as mock_client:
            # Mock OpenAI response
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            text = "Test context for Redis cluster setup"
            embedding = await embedding_service.generate_embedding(text)
            
            assert embedding is not None
            assert len(embedding) == 1536
            assert all(isinstance(x, float) for x in embedding)
            mock_client.embeddings.create.assert_called_once_with(
                model="text-embedding-ada-002",
                input=text
            )

    @pytest.mark.asyncio
    async def test_batch_generate_embeddings(self, embedding_service):
        """Test batch embedding generation for efficiency."""
        with patch.object(embedding_service, 'client') as mock_client:
            # Mock batch response
            mock_response = Mock()
            mock_response.data = [
                Mock(embedding=[0.1] * 1536),
                Mock(embedding=[0.2] * 1536),
                Mock(embedding=[0.3] * 1536)
            ]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            texts = [
                "Redis configuration guide",
                "PostgreSQL optimization tips", 
                "Agent communication patterns"
            ]
            embeddings = await embedding_service.batch_generate_embeddings(texts)
            
            assert len(embeddings) == 3
            assert all(len(emb) == 1536 for emb in embeddings)
            mock_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_embedding_caching(self, embedding_service):
        """Test embedding caching for performance optimization."""
        with patch.object(embedding_service, 'client') as mock_client:
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create = AsyncMock(return_value=mock_response)
            
            text = "Repeated context for caching test"
            
            # First call should hit API
            embedding1 = await embedding_service.generate_embedding(text)
            # Second call should use cache
            embedding2 = await embedding_service.generate_embedding(text)
            
            assert embedding1 == embedding2
            # API should only be called once due to caching
            mock_client.embeddings.create.assert_called_once()


class TestVectorSearchEngine:
    """Test suite for vector search functionality."""

    @pytest.fixture
    async def mock_db_session(self):
        """Create mock database session."""
        return AsyncMock(spec=AsyncSession)

    @pytest.fixture
    async def mock_embedding_service(self):
        """Create mock embedding service."""
        service = AsyncMock(spec=EmbeddingService)
        service.generate_embedding.return_value = [0.1] * 1536
        return service

    @pytest.fixture
    async def search_engine(self, mock_db_session, mock_embedding_service):
        """Create vector search engine instance."""
        return VectorSearchEngine(mock_db_session, mock_embedding_service)

    @pytest.mark.asyncio
    async def test_semantic_search_basic(self, search_engine, mock_db_session):
        """Test basic semantic search functionality."""
        # Mock database results
        mock_context = Mock(spec=Context)
        mock_context.id = uuid.uuid4()
        mock_context.title = "Redis Cluster Setup"
        mock_context.similarity_score = 0.85
        
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [mock_context]
        mock_db_session.execute.return_value = mock_result
        
        query = "database clustering configuration"
        agent_id = uuid.uuid4()
        
        results = await search_engine.semantic_search(
            query=query,
            agent_id=agent_id,
            limit=10
        )
        
        assert len(results) == 1
        assert results[0].id == mock_context.id
        mock_db_session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_semantic_search_with_filters(self, search_engine, mock_db_session):
        """Test semantic search with context type and time filters."""
        mock_contexts = [
            Mock(id=uuid.uuid4(), context_type=ContextType.DOCUMENTATION, similarity_score=0.9),
            Mock(id=uuid.uuid4(), context_type=ContextType.CODE_SNIPPET, similarity_score=0.8)
        ]
        
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = mock_contexts
        mock_db_session.execute.return_value = mock_result
        
        filters = {
            'context_types': [ContextType.DOCUMENTATION],
            'min_similarity': 0.7,
            'created_after': datetime.utcnow() - timedelta(days=7)
        }
        
        results = await search_engine.semantic_search(
            query="API documentation",
            agent_id=uuid.uuid4(),
            filters=filters
        )
        
        assert len(results) >= 1
        mock_db_session.execute.assert_called_once()

    @pytest.mark.asyncio 
    async def test_search_performance_target(self, search_engine):
        """Test that search meets <50ms performance target."""
        import time
        
        start_time = time.time()
        
        await search_engine.semantic_search(
            query="performance test query",
            agent_id=uuid.uuid4(),
            limit=10
        )
        
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Performance target: <50ms
        assert elapsed_time < 50, f"Search took {elapsed_time}ms, target is <50ms"


class TestContextCompressor:
    """Test suite for context compression functionality."""

    @pytest.fixture
    async def mock_llm_client(self):
        """Create mock LLM client for compression."""
        client = AsyncMock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Compressed summary preserving key insights")]
        client.messages.create = AsyncMock(return_value=mock_response)
        return client

    @pytest.fixture
    async def compressor(self, mock_llm_client):
        """Create context compressor instance."""
        return ContextCompressor(mock_llm_client)

    @pytest.mark.asyncio
    async def test_compress_conversation_standard(self, compressor):
        """Test standard conversation compression."""
        long_conversation = "A" * 50000  # 50k character conversation
        
        compressed = await compressor.compress_conversation(
            conversation_content=long_conversation,
            compression_ratio=0.3
        )
        
        assert compressed is not None
        assert len(compressed.summary) < len(long_conversation) * 0.4
        assert compressed.importance_score > 0.0
        assert "key insight" in compressed.summary.lower() or "summary" in compressed.summary.lower()

    @pytest.mark.asyncio
    async def test_preserve_critical_information(self, compressor):
        """Test that compression preserves critical decisions and patterns."""
        conversation_with_decisions = """
        Agent: I've identified a critical security vulnerability in the Redis configuration.
        Agent: The solution is to enable AUTH and set up SSL/TLS encryption.
        Agent: This pattern has worked successfully in 3 previous deployments.
        Agent: Key learning: Always validate connection security in production.
        """
        
        compressed = await compressor.compress_conversation(
            conversation_content=conversation_with_decisions
        )
        
        summary = compressed.summary.lower()
        # Should preserve critical elements
        assert any(keyword in summary for keyword in ["security", "redis", "ssl", "auth"])
        assert any(keyword in summary for keyword in ["solution", "pattern", "learning"])

    @pytest.mark.asyncio
    async def test_compression_ratio_adherence(self, compressor):
        """Test that compression adheres to specified ratios."""
        original_content = "X" * 10000  # 10k characters
        
        # Test different compression ratios
        ratios = [0.2, 0.3, 0.5]
        
        for ratio in ratios:
            compressed = await compressor.compress_conversation(
                conversation_content=original_content,
                compression_ratio=ratio
            )
            
            expected_max_length = len(original_content) * (ratio + 0.1)  # 10% tolerance
            assert len(compressed.summary) <= expected_max_length


class TestContextManager:
    """Test suite for high-level context management interface."""

    @pytest.fixture
    async def mock_dependencies(self):
        """Create mock dependencies for context manager."""
        return {
            'db_session': AsyncMock(spec=AsyncSession),
            'embedding_service': AsyncMock(spec=EmbeddingService),
            'search_engine': AsyncMock(spec=VectorSearchEngine),
            'compressor': AsyncMock(spec=ContextCompressor)
        }

    @pytest.fixture
    async def context_manager(self, mock_dependencies):
        """Create context manager instance with mocked dependencies."""
        return ContextManager(**mock_dependencies)

    @pytest.mark.asyncio
    async def test_store_context_with_embedding(self, context_manager, mock_dependencies):
        """Test storing context with automatic embedding generation."""
        context_data = ContextCreate(
            title="Redis Configuration Guide",
            content="Detailed guide for setting up Redis cluster...",
            context_type=ContextType.DOCUMENTATION,
            agent_id=uuid.uuid4(),
            importance_score=0.8
        )
        
        # Mock embedding generation
        mock_dependencies['embedding_service'].generate_embedding.return_value = [0.1] * 1536
        
        # Mock database save
        mock_context = Mock(spec=Context)
        mock_context.id = uuid.uuid4()
        mock_dependencies['db_session'].add = Mock()
        mock_dependencies['db_session'].commit = AsyncMock()
        mock_dependencies['db_session'].refresh = AsyncMock()
        
        result = await context_manager.store_context(context_data)
        
        assert result is not None
        mock_dependencies['embedding_service'].generate_embedding.assert_called_once()
        mock_dependencies['db_session'].commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_relevant_contexts(self, context_manager, mock_dependencies):
        """Test retrieving relevant contexts for agent query."""
        query = "How to optimize PostgreSQL performance?"
        agent_id = uuid.uuid4()
        
        # Mock search results
        mock_contexts = [
            Mock(id=uuid.uuid4(), title="PostgreSQL Tuning", similarity_score=0.9),
            Mock(id=uuid.uuid4(), title="Database Optimization", similarity_score=0.8)
        ]
        mock_dependencies['search_engine'].semantic_search.return_value = mock_contexts
        
        results = await context_manager.retrieve_relevant_contexts(
            query=query,
            agent_id=agent_id,
            limit=5
        )
        
        assert len(results) == 2
        assert all(result.similarity_score >= 0.7 for result in results)
        mock_dependencies['search_engine'].semantic_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_cross_agent_knowledge_sharing(self, context_manager, mock_dependencies):
        """Test that agents can access shared knowledge appropriately."""
        # Agent A stores public knowledge
        context_data = ContextCreate(
            title="Redis Cluster Setup Best Practices",
            content="Comprehensive guide for Redis clustering...",
            context_type=ContextType.DOCUMENTATION,
            agent_id=uuid.uuid4(),
            metadata={"access_level": "public"}
        )
        
        # Mock storage
        mock_dependencies['embedding_service'].generate_embedding.return_value = [0.1] * 1536
        stored_context = await context_manager.store_context(context_data)
        
        # Agent B searches for Redis knowledge
        agent_b_id = uuid.uuid4()
        mock_shared_context = Mock(
            id=stored_context.id if stored_context else uuid.uuid4(),
            title="Redis Cluster Setup Best Practices",
            similarity_score=0.95
        )
        mock_dependencies['search_engine'].semantic_search.return_value = [mock_shared_context]
        
        results = await context_manager.retrieve_relevant_contexts(
            query="Redis cluster configuration",
            agent_id=agent_b_id
        )
        
        assert len(results) >= 1
        assert any(result.title == "Redis Cluster Setup Best Practices" for result in results)

    @pytest.mark.asyncio
    async def test_context_consolidation_workflow(self, context_manager, mock_dependencies):
        """Test automatic context consolidation for frequently accessed contexts."""
        context_id = uuid.uuid4()
        
        # Mock context that should be consolidated
        mock_context = Mock(spec=Context)
        mock_context.access_count = "10"  # Highly accessed
        mock_context.importance_score = 0.9
        mock_context.should_be_consolidated.return_value = True
        
        # Mock compression result
        mock_compressed = Mock()
        mock_compressed.summary = "Consolidated summary of key insights"
        mock_compressed.importance_score = 0.95
        mock_dependencies['compressor'].compress_conversation.return_value = mock_compressed
        
        result = await context_manager.consolidate_context(context_id)
        
        assert result is not None
        mock_dependencies['compressor'].compress_conversation.assert_called_once()


class TestContextAPI:
    """Test suite for Context API endpoints."""

    @pytest.mark.asyncio
    async def test_create_context_endpoint(self):
        """Test context creation API endpoint."""
        # This will be implemented after API is built
        pass

    @pytest.mark.asyncio
    async def test_search_contexts_endpoint(self):
        """Test context search API endpoint."""
        # This will be implemented after API is built
        pass

    @pytest.mark.asyncio
    async def test_compress_context_endpoint(self):
        """Test context compression API endpoint."""
        # This will be implemented after API is built
        pass


class TestPerformanceTargets:
    """Test suite for validating performance targets."""

    @pytest.mark.asyncio
    async def test_context_retrieval_speed_target(self):
        """Test that context retrieval meets <50ms target."""
        # Performance validation will be implemented with real components
        pass

    @pytest.mark.asyncio
    async def test_token_reduction_target(self):
        """Test that context compression achieves 60-80% token reduction."""
        # Token counting validation will be implemented
        pass

    @pytest.mark.asyncio
    async def test_memory_accuracy_target(self):
        """Test that semantic search achieves >90% relevance precision."""
        # Relevance scoring validation will be implemented
        pass

    @pytest.mark.asyncio
    async def test_storage_efficiency_target(self):
        """Test storage efficiency <1GB per 10,000 contexts."""
        # Storage analysis will be implemented
        pass

    @pytest.mark.asyncio
    async def test_concurrent_access_target(self):
        """Test support for 50+ agents with <100ms latency."""
        # Concurrency testing will be implemented
        pass


# Test fixtures and utilities
@pytest.fixture
async def sample_contexts():
    """Create sample contexts for testing."""
    return [
        {
            "title": "Redis Cluster Configuration",
            "content": "Guide for setting up Redis in cluster mode with high availability...",
            "context_type": ContextType.DOCUMENTATION,
            "importance_score": 0.8
        },
        {
            "title": "PostgreSQL Performance Tuning", 
            "content": "Best practices for optimizing PostgreSQL query performance...",
            "context_type": ContextType.DOCUMENTATION,
            "importance_score": 0.9
        },
        {
            "title": "Agent Communication Error Fix",
            "content": "Fixed timeout issue in agent message bus by increasing Redis timeout...",
            "context_type": ContextType.ERROR_RESOLUTION,
            "importance_score": 0.7
        }
    ]


@pytest.fixture
async def mock_agent():
    """Create mock agent for testing."""
    return Mock(
        id=uuid.uuid4(),
        name="test-agent",
        type="CLAUDE",
        status="ACTIVE"
    )


@pytest.fixture 
async def mock_session():
    """Create mock session for testing."""
    return Mock(
        id=uuid.uuid4(),
        name="test-session",
        session_type="FEATURE_DEVELOPMENT",
        status="ACTIVE"
    )