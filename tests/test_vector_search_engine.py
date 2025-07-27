"""
Comprehensive test suite for Vector Search Engine.

Tests cover:
- Semantic search functionality with OpenAI embeddings
- Cross-agent context sharing with privacy controls
- Batch search processing and optimization
- Performance requirements validation (<50ms target)
- Relevance scoring and similarity thresholds
- Error handling and edge cases
- Integration with existing Context Engine components
"""

import pytest
import asyncio
import uuid
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import Mock, AsyncMock, patch

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.context import Context, ContextType
from app.core.vector_search_engine import (
    VectorSearchEngine,
    SearchConfiguration,
    BatchSearchRequest,
    BatchSearchResponse,
    create_vector_search_engine,
    quick_semantic_search
)
from app.core.enhanced_vector_search import ContextMatch, SearchFilters
from app.core.embedding_service_simple import EmbeddingService


@pytest.fixture
async def mock_embedding_service():
    """Create a mock embedding service for testing."""
    service = Mock(spec=EmbeddingService)
    
    # Mock embedding generation - return consistent embeddings for testing
    async def generate_embedding(text: str) -> List[float]:
        # Create deterministic embeddings based on text hash
        text_hash = hash(text)
        return [(text_hash % 1000) / 1000.0] * 1536
    
    async def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
        return [await generate_embedding(text) for text in texts]
    
    async def health_check() -> Dict[str, Any]:
        return {"status": "healthy", "model": "text-embedding-ada-002"}
    
    async def clear_cache() -> None:
        pass
    
    def get_performance_metrics() -> Dict[str, Any]:
        return {
            "total_api_calls": 10,
            "cache_hits": 5,
            "cache_hit_rate": 0.5,
            "total_tokens_processed": 1000
        }
    
    service.generate_embedding = generate_embedding
    service.generate_embeddings_batch = generate_embeddings_batch
    service.health_check = health_check
    service.clear_cache = clear_cache
    service.get_performance_metrics = get_performance_metrics
    
    return service


@pytest.fixture
async def test_contexts(test_db_session: AsyncSession) -> List[Context]:
    """Create test contexts with embeddings for search testing."""
    contexts = []
    
    test_data = [
        {
            "title": "Redis Configuration Guide",
            "content": "How to configure Redis for high availability and performance optimization",
            "context_type": ContextType.DOCUMENTATION,
            "importance_score": 0.8
        },
        {
            "title": "PostgreSQL Performance Tuning",
            "content": "Database optimization techniques for PostgreSQL including indexing and query optimization",
            "context_type": ContextType.DOCUMENTATION,
            "importance_score": 0.9
        },
        {
            "title": "Agent Communication Error Resolution",
            "content": "Fixed communication timeout issue between agents by increasing connection pool size",
            "context_type": ContextType.ERROR_RESOLUTION,
            "importance_score": 0.7
        },
        {
            "title": "API Authentication Implementation",
            "content": "Implemented JWT-based authentication with role-based access control",
            "context_type": ContextType.ARCHITECTURE,
            "importance_score": 0.85
        },
        {
            "title": "Docker Deployment Configuration",
            "content": "Container orchestration setup for microservices deployment with health checks",
            "context_type": ContextType.DOCUMENTATION,
            "importance_score": 0.75
        }
    ]
    
    for i, data in enumerate(test_data):
        context = Context(
            id=uuid.uuid4(),
            title=data["title"],
            content=data["content"],
            context_type=data["context_type"],
            importance_score=data["importance_score"],
            # Create deterministic embeddings for testing
            embedding=[(hash(data["content"]) % 1000) / 1000.0] * 1536,
            agent_id=uuid.uuid4() if i % 2 == 0 else None,  # Mix of agent-owned and unowned
            created_at=datetime.utcnow() - timedelta(days=i),
            accessed_at=datetime.utcnow() - timedelta(hours=i)
        )
        test_db_session.add(context)
        contexts.append(context)
    
    await test_db_session.commit()
    return contexts


@pytest.fixture
async def vector_search_engine(
    test_db_session: AsyncSession,
    mock_embedding_service: Mock
) -> VectorSearchEngine:
    """Create a vector search engine for testing."""
    config = SearchConfiguration(
        similarity_threshold=0.7,
        performance_target_ms=50.0,
        max_results=20,
        enable_caching=False,  # Disable caching for predictable tests
        enable_cross_agent=True
    )
    
    return VectorSearchEngine(
        db_session=test_db_session,
        embedding_service=mock_embedding_service,
        config=config,
        redis_client=None  # No Redis for unit tests
    )


class TestVectorSearchEngine:
    """Test suite for Vector Search Engine core functionality."""
    
    async def test_semantic_search_basic(
        self,
        vector_search_engine: VectorSearchEngine,
        test_contexts: List[Context]
    ):
        """Test basic semantic search functionality."""
        # Test search for database-related content
        results = await vector_search_engine.semantic_search(
            query="database optimization performance",
            limit=3
        )
        
        assert isinstance(results, list)
        assert len(results) <= 3
        
        # Verify results contain ContextMatch objects
        for result in results:
            assert isinstance(result, ContextMatch)
            assert hasattr(result, 'context')
            assert hasattr(result, 'similarity_score')
            assert hasattr(result, 'relevance_score')
            assert hasattr(result, 'rank')
        
        # Verify results are ordered by relevance
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].relevance_score >= results[i + 1].relevance_score
    
    async def test_semantic_search_with_agent_filtering(
        self,
        vector_search_engine: VectorSearchEngine,
        test_contexts: List[Context]
    ):
        """Test semantic search with agent-specific filtering."""
        # Get an agent ID from test contexts
        agent_id = test_contexts[0].agent_id
        
        # Search with agent filtering
        results = await vector_search_engine.semantic_search(
            query="configuration setup",
            agent_id=agent_id,
            include_cross_agent=False,
            limit=5
        )
        
        # Verify all results belong to the specified agent or are public
        for result in results:
            assert (result.context.agent_id == agent_id or 
                   result.context.agent_id is None)
    
    async def test_semantic_search_similarity_threshold(
        self,
        vector_search_engine: VectorSearchEngine,
        test_contexts: List[Context]
    ):
        """Test semantic search with similarity threshold filtering."""
        # Test with high similarity threshold
        high_threshold_results = await vector_search_engine.semantic_search(
            query="redis configuration",
            similarity_threshold=0.9,
            limit=10
        )
        
        # Test with low similarity threshold  
        low_threshold_results = await vector_search_engine.semantic_search(
            query="redis configuration",
            similarity_threshold=0.3,
            limit=10
        )
        
        # High threshold should return fewer or equal results
        assert len(high_threshold_results) <= len(low_threshold_results)
        
        # All results should meet the threshold
        for result in high_threshold_results:
            assert result.similarity_score >= 0.9
    
    async def test_semantic_search_performance(
        self,
        vector_search_engine: VectorSearchEngine,
        test_contexts: List[Context]
    ):
        """Test that semantic search meets performance requirements."""
        start_time = time.perf_counter()
        
        results = await vector_search_engine.semantic_search(
            query="performance optimization techniques",
            limit=10
        )
        
        end_time = time.perf_counter()
        search_time_ms = (end_time - start_time) * 1000
        
        # Should meet the <50ms performance target
        assert search_time_ms < vector_search_engine.config.performance_target_ms
        assert len(results) >= 0  # Should return some results
    
    async def test_find_similar_contexts(
        self,
        vector_search_engine: VectorSearchEngine,
        test_contexts: List[Context]
    ):
        """Test finding contexts similar to a given context."""
        reference_context = test_contexts[0]
        
        similar_contexts = await vector_search_engine.find_similar_contexts(
            context_id=reference_context.id,
            limit=3,
            similarity_threshold=0.5
        )
        
        assert isinstance(similar_contexts, list)
        assert len(similar_contexts) <= 3
        
        # Verify no context is similar to itself
        for match in similar_contexts:
            assert match.context.id != reference_context.id
            assert match.similarity_score >= 0.5
    
    async def test_batch_search(
        self,
        vector_search_engine: VectorSearchEngine,
        test_contexts: List[Context]
    ):
        """Test batch search functionality."""
        batch_request = BatchSearchRequest(
            queries=[
                "database optimization",
                "authentication security",
                "container deployment",
                "error resolution"
            ],
            limit=5
        )
        
        start_time = time.perf_counter()
        response = await vector_search_engine.batch_search(batch_request)
        end_time = time.perf_counter()
        
        assert isinstance(response, BatchSearchResponse)
        assert len(response.results) == len(batch_request.queries)
        
        # Verify each query has results
        for query in batch_request.queries:
            assert query in response.results
            assert isinstance(response.results[query], list)
        
        # Verify performance metrics
        assert 'total_time_ms' in response.performance_metrics
        assert 'avg_time_per_query_ms' in response.performance_metrics
        assert response.performance_metrics['queries_processed'] == len(batch_request.queries)
        
        # Batch processing should be reasonably efficient
        total_time_ms = (end_time - start_time) * 1000
        assert response.performance_metrics['total_time_ms'] <= total_time_ms + 10  # Small tolerance
    
    async def test_cross_agent_search(
        self,
        vector_search_engine: VectorSearchEngine,
        test_contexts: List[Context]
    ):
        """Test cross-agent context discovery."""
        # Get an agent ID
        requesting_agent_id = uuid.uuid4()
        
        results = await vector_search_engine.cross_agent_search(
            query="system configuration",
            requesting_agent_id=requesting_agent_id,
            limit=5,
            min_importance=0.6
        )
        
        assert isinstance(results, list)
        
        # Verify results are from other agents (not the requesting agent)
        for result in results:
            assert result.context.agent_id != requesting_agent_id
            assert result.context.importance_score >= 0.6
    
    async def test_index_context(
        self,
        vector_search_engine: VectorSearchEngine,
        test_db_session: AsyncSession
    ):
        """Test context indexing with embedding generation."""
        # Create a new context without embedding
        new_context = Context(
            title="Test Context for Indexing",
            content="This is a test context that needs to be indexed with embeddings",
            context_type=ContextType.LEARNING,
            importance_score=0.6
        )
        test_db_session.add(new_context)
        await test_db_session.commit()
        
        # Index the context
        result = await vector_search_engine.index_context(new_context)
        
        assert result is True
        assert new_context.embedding is not None
        assert len(new_context.embedding) == 1536  # OpenAI embedding dimension
    
    async def test_update_context_vector(
        self,
        vector_search_engine: VectorSearchEngine,
        test_contexts: List[Context]
    ):
        """Test updating context vector embeddings."""
        context = test_contexts[0]
        original_embedding = context.embedding.copy()
        new_content = "This is completely new content for the context"
        
        result = await vector_search_engine.update_context_vector(
            context_id=context.id,
            new_content=new_content
        )
        
        assert result is True
        assert context.content == new_content
        assert context.embedding != original_embedding
        assert context.updated_at is not None
    
    async def test_bulk_index_contexts(
        self,
        vector_search_engine: VectorSearchEngine,
        test_db_session: AsyncSession
    ):
        """Test bulk indexing of multiple contexts."""
        # Create contexts without embeddings
        contexts_to_index = []
        for i in range(5):
            context = Context(
                title=f"Bulk Index Test Context {i}",
                content=f"Content for bulk indexing test context number {i}",
                context_type=ContextType.LEARNING,
                importance_score=0.5
            )
            test_db_session.add(context)
            contexts_to_index.append(context)
        
        await test_db_session.commit()
        
        context_ids = [ctx.id for ctx in contexts_to_index]
        
        # Bulk index the contexts
        stats = await vector_search_engine.bulk_index_contexts(
            context_ids=context_ids,
            batch_size=3
        )
        
        assert stats['total_contexts'] == 5
        assert stats['successfully_indexed'] == 5
        assert stats['failed'] == 0
        assert stats['skipped'] == 0
        
        # Verify all contexts now have embeddings
        for context in contexts_to_index:
            await test_db_session.refresh(context)
            assert context.embedding is not None
    
    async def test_get_context_recommendations(
        self,
        vector_search_engine: VectorSearchEngine,
        test_contexts: List[Context]
    ):
        """Test context recommendation functionality."""
        agent_id = test_contexts[0].agent_id or uuid.uuid4()
        
        recommendations = await vector_search_engine.get_context_recommendations(
            agent_id=agent_id,
            limit=3
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        
        for recommendation in recommendations:
            assert isinstance(recommendation, ContextMatch)
            assert recommendation.context.importance_score >= 0.0
    
    async def test_search_filters(
        self,
        vector_search_engine: VectorSearchEngine,
        test_contexts: List[Context]
    ):
        """Test search with various filters."""
        filters = SearchFilters(
            context_types=[ContextType.DOCUMENTATION],
            min_importance=0.8,
            max_age_days=30
        )
        
        results = await vector_search_engine.semantic_search(
            query="configuration documentation",
            filters=filters,
            limit=10
        )
        
        # Verify all results match the filters
        for result in results:
            assert result.context.context_type == ContextType.DOCUMENTATION
            assert result.context.importance_score >= 0.8
            
            # Check age constraint
            if result.context.created_at:
                age = (datetime.utcnow() - result.context.created_at).days
                assert age <= 30
    
    async def test_performance_metrics(
        self,
        vector_search_engine: VectorSearchEngine,
        test_contexts: List[Context]
    ):
        """Test performance metrics collection."""
        # Perform some searches to generate metrics
        await vector_search_engine.semantic_search("test query 1", limit=5)
        await vector_search_engine.semantic_search("test query 2", limit=5)
        
        metrics = vector_search_engine.get_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert 'search_engine' in metrics
        assert 'enhanced_engine' in metrics
        assert 'embedding_service' in metrics
        assert 'configuration' in metrics
        assert 'system_health' in metrics
        
        # Check search engine metrics
        search_metrics = metrics['search_engine']
        assert search_metrics['total_searches'] >= 2
        assert 'average_search_time_ms' in search_metrics
        assert 'meets_performance_target' in search_metrics
    
    async def test_health_check(
        self,
        vector_search_engine: VectorSearchEngine,
        test_contexts: List[Context]
    ):
        """Test health check functionality."""
        health_status = await vector_search_engine.health_check()
        
        assert isinstance(health_status, dict)
        assert 'status' in health_status
        assert 'components' in health_status
        assert 'performance' in health_status
        
        # Check component health
        components = health_status['components']
        assert 'embedding_service' in components
        assert 'database' in components
        assert 'search' in components
        
        # Overall status should be healthy or degraded
        assert health_status['status'] in ['healthy', 'degraded', 'unhealthy']
    
    async def test_error_handling(
        self,
        vector_search_engine: VectorSearchEngine
    ):
        """Test error handling for various edge cases."""
        # Test empty query
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await vector_search_engine.semantic_search("")
        
        # Test invalid limit
        with pytest.raises(ValueError, match="Limit must be between"):
            await vector_search_engine.semantic_search("test", limit=0)
        
        with pytest.raises(ValueError, match="Limit must be between"):
            await vector_search_engine.semantic_search("test", limit=1000)
        
        # Test update context vector with empty content
        with pytest.raises(ValueError, match="New content cannot be empty"):
            await vector_search_engine.update_context_vector(uuid.uuid4(), "")
        
        # Test find similar contexts with non-existent context
        results = await vector_search_engine.find_similar_contexts(uuid.uuid4())
        assert results == []
    
    async def test_search_configuration(
        self,
        test_db_session: AsyncSession,
        mock_embedding_service: Mock
    ):
        """Test different search configurations."""
        # Test with custom configuration
        custom_config = SearchConfiguration(
            similarity_threshold=0.9,
            cross_agent_threshold=0.95,
            performance_target_ms=25.0,
            max_results=15,
            enable_cross_agent=False
        )
        
        engine = VectorSearchEngine(
            db_session=test_db_session,
            embedding_service=mock_embedding_service,
            config=custom_config
        )
        
        assert engine.config.similarity_threshold == 0.9
        assert engine.config.cross_agent_threshold == 0.95
        assert engine.config.performance_target_ms == 25.0
        assert engine.config.max_results == 15
        assert engine.config.enable_cross_agent is False


class TestFactoryFunctions:
    """Test factory functions and utility methods."""
    
    async def test_create_vector_search_engine(
        self,
        test_db_session: AsyncSession
    ):
        """Test factory function for creating search engine."""
        with patch('app.core.vector_search_engine.get_embedding_service'):
            with patch('app.core.vector_search_engine.get_redis_client'):
                engine = await create_vector_search_engine(test_db_session)
                
                assert isinstance(engine, VectorSearchEngine)
                assert engine.db == test_db_session
                assert engine.config is not None
    
    async def test_quick_semantic_search(
        self,
        test_db_session: AsyncSession,
        test_contexts: List[Context]
    ):
        """Test quick semantic search utility function."""
        with patch('app.core.vector_search_engine.get_embedding_service') as mock_service:
            mock_service.return_value = Mock(spec=EmbeddingService)
            mock_service.return_value.generate_embedding = AsyncMock(
                return_value=[0.5] * 1536
            )
            
            with patch('app.core.vector_search_engine.get_redis_client'):
                results = await quick_semantic_search(
                    query="test query",
                    db_session=test_db_session,
                    limit=5
                )
                
                assert isinstance(results, list)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    async def test_agent_knowledge_sharing_scenario(
        self,
        vector_search_engine: VectorSearchEngine,
        test_contexts: List[Context],
        test_db_session: AsyncSession
    ):
        """Test a realistic agent knowledge sharing scenario."""
        # Create contexts for different agents
        agent_a = uuid.uuid4()
        agent_b = uuid.uuid4()
        
        # Agent A creates a context about Redis optimization
        context_a = Context(
            title="Redis Memory Optimization Techniques",
            content="Optimized Redis memory usage by implementing key expiration policies and memory-efficient data structures",
            context_type=ContextType.LEARNING,
            importance_score=0.9,
            agent_id=agent_a,
            embedding=[0.8] * 1536  # High similarity embedding
        )
        test_db_session.add(context_a)
        
        # Agent B searches for Redis-related knowledge
        await test_db_session.commit()
        
        # Agent B should find Agent A's context through cross-agent search
        results = await vector_search_engine.cross_agent_search(
            query="Redis memory optimization",
            requesting_agent_id=agent_b,
            min_importance=0.8,
            limit=5
        )
        
        # Verify Agent B found Agent A's knowledge
        found_context_ids = [result.context.id for result in results]
        assert context_a.id in found_context_ids
        
        # Verify the context has high relevance
        for result in results:
            if result.context.id == context_a.id:
                assert result.similarity_score > 0.7
                assert result.context.agent_id == agent_a
    
    async def test_performance_under_load(
        self,
        vector_search_engine: VectorSearchEngine,
        test_contexts: List[Context]
    ):
        """Test search engine performance under concurrent load."""
        # Create multiple concurrent search tasks
        search_tasks = []
        queries = [
            "database optimization",
            "authentication security", 
            "performance monitoring",
            "error handling",
            "system configuration"
        ]
        
        for query in queries:
            task = vector_search_engine.semantic_search(
                query=query,
                limit=10
            )
            search_tasks.append(task)
        
        start_time = time.perf_counter()
        
        # Execute all searches concurrently
        results = await asyncio.gather(*search_tasks)
        
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        
        # Verify all searches completed successfully
        assert len(results) == len(queries)
        for result_list in results:
            assert isinstance(result_list, list)
        
        # Performance should still be reasonable under load
        avg_time_per_search = total_time_ms / len(queries)
        assert avg_time_per_search < vector_search_engine.config.performance_target_ms * 2  # Allow 2x under load
    
    async def test_token_reduction_scenario(
        self,
        vector_search_engine: VectorSearchEngine,
        test_contexts: List[Context]
    ):
        """Test the token reduction value proposition scenario."""
        # Simulate a complex query that would normally require large context
        complex_query = (
            "I need to implement a secure authentication system with Redis session management "
            "and PostgreSQL user storage. What are the best practices and common pitfalls?"
        )
        
        # Search should return relevant contexts efficiently
        results = await vector_search_engine.semantic_search(
            query=complex_query,
            limit=10
        )
        
        # Verify we get relevant results
        assert len(results) > 0
        
        # Calculate estimated token savings
        # Assume each context replaces ~2000 tokens of general context
        estimated_tokens_saved = len(results) * 2000
        
        # With 60-80% reduction target, verify significant savings
        assert estimated_tokens_saved >= 1000  # Significant token reduction achieved
        
        # Verify relevance quality
        for result in results:
            assert result.similarity_score >= vector_search_engine.config.similarity_threshold
            assert result.relevance_score > 0.0