"""
Semantic Memory Service - Component Isolation Tests
==================================================

Tests the semantic memory and context management system in complete isolation.
This validates knowledge storage, retrieval, compression, and graph building
without relying on pgvector, database, or external embedding services.

Testing Strategy:
- Mock all external dependencies (pgvector, OpenAI embeddings, database)
- Test core semantic memory operations and algorithms
- Validate context compression and consolidation logic
- Ensure proper knowledge graph construction and traversal
- Test cross-agent knowledge sharing mechanisms
"""

import asyncio
import uuid
import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.semantic_memory_service import SemanticMemoryService
from app.core.enhanced_context_engine import EnhancedContextEngine
from app.core.context_compression_engine import ContextCompressionEngine
from app.core.knowledge_graph_builder import KnowledgeGraphBuilder
from app.core.vector_search_engine import VectorSearchEngine


@pytest.mark.isolation
@pytest.mark.unit
@pytest.mark.context_management
class TestSemanticMemoryServiceIsolated:
    """Test semantic memory service functionality in isolation."""
    
    @pytest.fixture
    async def isolated_semantic_memory(
        self,
        mock_database_session,
        mock_vector_search,
        isolated_test_environment,
        assert_isolated
    ):
        """Create isolated semantic memory service with all dependencies mocked."""
        
        # Mock vector database operations
        mock_pgvector = AsyncMock()
        mock_pgvector.search_similar = AsyncMock(return_value=[
            {"id": "doc_1", "content": "Similar document", "score": 0.95}
        ])
        mock_pgvector.store_vector = AsyncMock(return_value="vec_123")
        mock_pgvector.delete_vector = AsyncMock(return_value=True)
        
        # Mock embedding service
        mock_embeddings = AsyncMock()
        mock_embeddings.embed_text = AsyncMock(return_value=np.random.random(1536).tolist())
        mock_embeddings.embed_batch = AsyncMock(return_value=[
            np.random.random(1536).tolist() for _ in range(3)
        ])
        
        with patch('app.services.semantic_memory_service.get_database_session', return_value=mock_database_session), \
             patch('app.services.semantic_memory_service.get_pgvector_manager', return_value=mock_pgvector), \
             patch('app.services.semantic_memory_service.get_embedding_service', return_value=mock_embeddings):
            
            memory_service = SemanticMemoryService()
            await memory_service.initialize()
            
            # Assert complete isolation
            assert_isolated(memory_service, {
                "database": mock_database_session,
                "pgvector": mock_pgvector,
                "embeddings": mock_embeddings,
                "vector_search": mock_vector_search
            })
            
            yield memory_service
            
            await memory_service.shutdown()
    
    async def test_memory_storage_and_retrieval_isolated(
        self,
        isolated_semantic_memory,
        capture_component_calls
    ):
        """Test basic memory storage and retrieval in isolation."""
        memory_service = isolated_semantic_memory
        
        # Capture method calls
        calls, _ = capture_component_calls(memory_service, [
            "store_memory", "retrieve_memory", "search_memories"
        ])
        
        # Store a memory
        memory_data = {
            "content": "The user prefers Python over JavaScript for backend development",
            "type": "preference",
            "agent_id": str(uuid.uuid4()),
            "context": {"domain": "programming", "confidence": 0.9},
            "metadata": {"source": "conversation", "timestamp": datetime.utcnow().isoformat()}
        }
        
        result = await memory_service.store_memory(**memory_data)
        assert result["success"] is True
        assert "memory_id" in result
        memory_id = result["memory_id"]
        
        # Retrieve the memory
        retrieved = await memory_service.retrieve_memory(memory_id)
        assert retrieved is not None
        assert retrieved["content"] == memory_data["content"]
        assert retrieved["type"] == memory_data["type"]
        
        # Search for related memories
        search_results = await memory_service.search_memories(
            query="Python programming preferences",
            limit=5
        )
        assert len(search_results) > 0
        assert any(memory_id == result["id"] for result in search_results)
        
        # Verify method calls
        assert len(calls) == 3
        assert calls[0]["method"] == "store_memory"
        assert calls[1]["method"] == "retrieve_memory"
        assert calls[2]["method"] == "search_memories"
    
    async def test_context_compression_isolated(
        self,
        isolated_semantic_memory,
        isolated_test_environment
    ):
        """Test context compression algorithms in isolation."""
        memory_service = isolated_semantic_memory
        
        # Create large context to compress
        large_context = {
            "conversation_history": [
                {"role": "user", "content": f"Message {i}: This is test content for compression"} 
                for i in range(100)
            ],
            "code_snippets": [
                {"language": "python", "code": f"def function_{i}():\n    return {i}"} 
                for i in range(50)
            ],
            "documentation": [
                {"title": f"Doc {i}", "content": f"Documentation content {i} " * 100}
                for i in range(20)
            ]
        }
        
        # Test compression
        compression_result = await memory_service.compress_context(
            context=large_context,
            target_size_ratio=0.3,
            preserve_important=True
        )
        
        assert compression_result["success"] is True
        assert "compressed_context" in compression_result
        assert "compression_ratio" in compression_result
        assert "preserved_elements" in compression_result
        
        # Verify compression effectiveness
        original_size = len(str(large_context))
        compressed_size = len(str(compression_result["compressed_context"]))
        actual_ratio = compressed_size / original_size
        
        assert actual_ratio <= 0.4  # Allow some variance from target 0.3
        assert compression_result["compression_ratio"] > 0.5  # At least 50% compression
        
        # Verify important elements are preserved
        preserved = compression_result["preserved_elements"]
        assert len(preserved) > 0
        assert any("function" in elem for elem in preserved)  # Code should be preserved
    
    async def test_knowledge_graph_construction_isolated(
        self,
        isolated_semantic_memory,
        isolated_agent_config
    ):
        """Test knowledge graph construction and traversal in isolation."""
        memory_service = isolated_semantic_memory
        
        # Create agents with different knowledge domains
        agents = []
        for i, domain in enumerate(["backend", "frontend", "database", "testing"]):
            agent_config = isolated_agent_config(
                role=f"{domain}-specialist",
                capabilities=[domain, "general"]
            )
            agents.append(agent_config)
        
        # Store knowledge for each agent
        knowledge_entries = []
        for i, agent in enumerate(agents):
            for j in range(3):
                knowledge = {
                    "content": f"Knowledge about {agent['role']}: Fact {j}",
                    "type": "expertise",
                    "agent_id": agent["id"],
                    "domain": agent["role"].split("-")[0],
                    "concepts": [agent["role"], "best_practices", f"technique_{j}"]
                }
                result = await memory_service.store_knowledge(**knowledge)
                knowledge_entries.append(result["knowledge_id"])
        
        # Build knowledge graph
        graph_result = await memory_service.build_knowledge_graph(
            scope="all_agents",
            min_connection_strength=0.1
        )
        
        assert graph_result["success"] is True
        assert "graph" in graph_result
        
        graph = graph_result["graph"]
        assert "nodes" in graph
        assert "edges" in graph
        assert "clusters" in graph
        
        # Verify graph structure
        assert len(graph["nodes"]) > 0
        assert len(graph["edges"]) > 0
        
        # Test graph traversal
        traversal_result = await memory_service.traverse_knowledge_graph(
            start_concept="backend",
            max_depth=3,
            relationship_types=["relates_to", "depends_on", "enables"]
        )
        
        assert "path" in traversal_result
        assert "connected_concepts" in traversal_result
        assert len(traversal_result["connected_concepts"]) > 0
    
    async def test_cross_agent_knowledge_sharing_isolated(
        self,
        isolated_semantic_memory,
        isolated_agent_config
    ):
        """Test cross-agent knowledge sharing mechanisms in isolation."""
        memory_service = isolated_semantic_memory
        
        # Create specialized agents
        backend_agent = isolated_agent_config(
            role="backend-engineer",
            capabilities=["python", "fastapi", "postgresql"]
        )
        frontend_agent = isolated_agent_config(
            role="frontend-developer",
            capabilities=["react", "typescript", "css"]
        )
        
        # Backend agent stores API knowledge
        api_knowledge = await memory_service.store_knowledge(
            content="FastAPI best practice: Use dependency injection for database sessions",
            type="best_practice",
            agent_id=backend_agent["id"],
            domain="backend",
            shareability="high",
            tags=["api", "database", "architecture"]
        )
        
        # Frontend agent requests related knowledge
        shared_knowledge = await memory_service.request_shared_knowledge(
            requesting_agent_id=frontend_agent["id"],
            query="API best practices",
            domains=["backend", "api"],
            max_results=5
        )
        
        assert len(shared_knowledge) > 0
        assert any("FastAPI" in item["content"] for item in shared_knowledge)
        
        # Test knowledge adaptation for frontend context
        adapted_knowledge = await memory_service.adapt_knowledge_for_agent(
            knowledge_items=shared_knowledge,
            target_agent_id=frontend_agent["id"],
            target_context="frontend API integration"
        )
        
        assert len(adapted_knowledge) > 0
        adapted_item = adapted_knowledge[0]
        assert "adapted_content" in adapted_item
        assert "relevance_score" in adapted_item
        assert adapted_item["relevance_score"] > 0.5
    
    async def test_memory_consolidation_isolated(
        self,
        isolated_semantic_memory,
        isolated_agent_config
    ):
        """Test memory consolidation and deduplication in isolation."""
        memory_service = isolated_semantic_memory
        agent_config = isolated_agent_config()
        
        # Store similar memories
        similar_memories = [
            {
                "content": "Python is great for web development",
                "type": "opinion",
                "agent_id": agent_config["id"],
                "context": {"topic": "programming"}
            },
            {
                "content": "Python is excellent for building web applications",
                "type": "opinion", 
                "agent_id": agent_config["id"],
                "context": {"topic": "programming"}
            },
            {
                "content": "Web development with Python is very effective",
                "type": "opinion",
                "agent_id": agent_config["id"], 
                "context": {"topic": "programming"}
            }
        ]
        
        memory_ids = []
        for memory in similar_memories:
            result = await memory_service.store_memory(**memory)
            memory_ids.append(result["memory_id"])
        
        # Run consolidation
        consolidation_result = await memory_service.consolidate_memories(
            agent_id=agent_config["id"],
            similarity_threshold=0.8,
            consolidation_strategy="merge_similar"
        )
        
        assert consolidation_result["success"] is True
        assert "consolidated_memories" in consolidation_result
        assert "removed_duplicates" in consolidation_result
        
        # Verify consolidation effectiveness
        assert consolidation_result["removed_duplicates"] > 0
        consolidated = consolidation_result["consolidated_memories"]
        assert len(consolidated) < len(similar_memories)
        
        # Verify consolidated memory contains key information
        consolidated_memory = consolidated[0]
        assert "Python" in consolidated_memory["content"]
        assert "web development" in consolidated_memory["content"]
    
    async def test_contextual_memory_retrieval_isolated(
        self,
        isolated_semantic_memory,
        isolated_agent_config
    ):
        """Test contextual memory retrieval with different scenarios."""
        memory_service = isolated_semantic_memory
        agent_config = isolated_agent_config()
        
        # Store memories in different contexts
        memories = [
            {
                "content": "Use React hooks for state management",
                "type": "technical_tip",
                "agent_id": agent_config["id"],
                "context": {"domain": "frontend", "framework": "react", "complexity": "intermediate"}
            },
            {
                "content": "FastAPI dependency injection pattern",
                "type": "technical_tip",
                "agent_id": agent_config["id"],
                "context": {"domain": "backend", "framework": "fastapi", "complexity": "advanced"}
            },
            {
                "content": "Basic HTML structure guidelines",
                "type": "technical_tip",
                "agent_id": agent_config["id"],
                "context": {"domain": "frontend", "technology": "html", "complexity": "beginner"}
            }
        ]
        
        for memory in memories:
            await memory_service.store_memory(**memory)
        
        # Test context-aware retrieval
        frontend_memories = await memory_service.retrieve_memories_by_context(
            agent_id=agent_config["id"],
            context_filter={"domain": "frontend"},
            limit=10
        )
        
        assert len(frontend_memories) == 2
        assert all("frontend" in mem["context"]["domain"] for mem in frontend_memories)
        
        # Test complexity-based retrieval
        advanced_memories = await memory_service.retrieve_memories_by_context(
            agent_id=agent_config["id"],
            context_filter={"complexity": "advanced"},
            limit=10
        )
        
        assert len(advanced_memories) == 1
        assert "FastAPI" in advanced_memories[0]["content"]
        
        # Test combined context retrieval
        react_memories = await memory_service.retrieve_memories_by_context(
            agent_id=agent_config["id"],
            context_filter={"domain": "frontend", "framework": "react"},
            limit=10
        )
        
        assert len(react_memories) == 1
        assert "React hooks" in react_memories[0]["content"]
    
    async def test_memory_performance_metrics_isolated(
        self,
        isolated_semantic_memory,
        isolated_agent_config
    ):
        """Test memory performance monitoring in isolation."""
        memory_service = isolated_semantic_memory
        agent_config = isolated_agent_config()
        
        # Perform various memory operations to generate metrics
        operations = []
        
        # Storage operations
        for i in range(10):
            start_time = asyncio.get_event_loop().time()
            await memory_service.store_memory(
                content=f"Performance test memory {i}",
                type="test",
                agent_id=agent_config["id"]
            )
            end_time = asyncio.get_event_loop().time()
            operations.append(("store", end_time - start_time))
        
        # Retrieval operations
        for i in range(5):
            start_time = asyncio.get_event_loop().time()
            await memory_service.search_memories(
                query=f"Performance test {i}",
                limit=5
            )
            end_time = asyncio.get_event_loop().time()
            operations.append(("search", end_time - start_time))
        
        # Get performance metrics
        metrics = await memory_service.get_performance_metrics()
        
        assert "storage_operations" in metrics
        assert "retrieval_operations" in metrics
        assert "average_storage_time" in metrics
        assert "average_retrieval_time" in metrics
        assert "memory_usage" in metrics
        assert "total_memories_stored" in metrics
        
        # Verify metrics accuracy
        assert metrics["storage_operations"] >= 10
        assert metrics["retrieval_operations"] >= 5
        assert metrics["average_storage_time"] > 0
        assert metrics["average_retrieval_time"] > 0
        
        # Test performance optimization suggestions
        optimization_suggestions = await memory_service.get_optimization_suggestions()
        assert isinstance(optimization_suggestions, list)
        assert len(optimization_suggestions) > 0
        
        for suggestion in optimization_suggestions:
            assert "category" in suggestion
            assert "description" in suggestion
            assert "impact" in suggestion


@pytest.mark.isolation
@pytest.mark.unit
@pytest.mark.context_management
class TestContextCompressionEngineIsolated:
    """Test context compression engine in isolation."""
    
    @pytest.fixture
    async def isolated_compression_engine(
        self,
        mock_vector_search,
        isolated_test_environment
    ):
        """Create isolated compression engine."""
        
        with patch('app.core.context_compression_engine.get_embedding_service', return_value=mock_vector_search):
            engine = ContextCompressionEngine()
            await engine.initialize()
            
            yield engine
            
            await engine.shutdown()
    
    async def test_intelligent_compression_strategies_isolated(
        self,
        isolated_compression_engine
    ):
        """Test different compression strategies in isolation."""
        engine = isolated_compression_engine
        
        # Test data with different importance levels
        context_data = {
            "critical_info": [
                "User authentication token: abc123",
                "Database connection string: postgresql://...",
                "Current task ID: task_456"
            ],
            "important_info": [
                "User preference: dark mode",
                "Last action: created new file",
                "Current directory: /home/user/project"
            ],
            "supplementary_info": [
                "Weather: sunny",
                "Random fact: cats have 9 lives",
                "Joke: Why did the chicken cross the road?"
            ],
            "verbose_logs": [
                f"Debug log entry {i}: Some detailed information here" 
                for i in range(50)
            ]
        }
        
        # Test conservative compression (preserve most information)
        conservative_result = await engine.compress_context(
            context=context_data,
            strategy="conservative",
            target_ratio=0.7
        )
        
        assert conservative_result["compression_ratio"] >= 0.3
        assert len(conservative_result["preserved_critical"]) == 3
        assert "authentication token" in str(conservative_result["compressed_context"])
        
        # Test aggressive compression (maximum space saving)
        aggressive_result = await engine.compress_context(
            context=context_data,
            strategy="aggressive", 
            target_ratio=0.3
        )
        
        assert aggressive_result["compression_ratio"] >= 0.7
        assert "authentication token" in str(aggressive_result["compressed_context"])  # Critical info preserved
        assert "Random fact" not in str(aggressive_result["compressed_context"])  # Supplementary removed
        
        # Test semantic compression (preserve meaning)
        semantic_result = await engine.compress_context(
            context=context_data,
            strategy="semantic",
            target_ratio=0.5
        )
        
        assert semantic_result["compression_ratio"] >= 0.5
        assert len(semantic_result["semantic_clusters"]) > 0
        assert "preserved_semantics" in semantic_result
    
    async def test_context_relevance_scoring_isolated(
        self,
        isolated_compression_engine
    ):
        """Test context relevance scoring algorithms in isolation."""
        engine = isolated_compression_engine
        
        # Test context with varying relevance
        context_items = [
            {"content": "Current task: implement user authentication", "type": "task"},
            {"content": "Related code: def authenticate_user(token):", "type": "code"},
            {"content": "Documentation: Authentication best practices", "type": "docs"},
            {"content": "Unrelated: Today's lunch menu", "type": "misc"},
            {"content": "Old task: fix CSS styling from last week", "type": "old_task"}
        ]
        
        current_context = {
            "current_task": "user authentication",
            "focus_areas": ["security", "backend", "API"],
            "recent_actions": ["read auth docs", "wrote auth function"]
        }
        
        # Score relevance
        relevance_scores = await engine.score_context_relevance(
            context_items=context_items,
            current_context=current_context,
            scoring_factors=["semantic_similarity", "temporal_relevance", "task_alignment"]
        )
        
        assert len(relevance_scores) == len(context_items)
        
        # Verify scoring logic
        auth_task_score = next(s for s in relevance_scores if "implement user authentication" in s["content"])
        lunch_score = next(s for s in relevance_scores if "lunch menu" in s["content"])
        
        assert auth_task_score["relevance_score"] > lunch_score["relevance_score"]
        assert auth_task_score["relevance_score"] > 0.8  # High relevance
        assert lunch_score["relevance_score"] < 0.3      # Low relevance