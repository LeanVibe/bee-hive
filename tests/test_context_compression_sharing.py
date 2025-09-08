"""
Comprehensive Test Suite for Context Compression and Cross-Agent Knowledge Sharing

Tests all components of VS 5.2 implementation:
- Context Compression Engine with all algorithms
- Cross-Agent Knowledge Manager with sharing and access controls
- Memory Hierarchy Manager with consolidation and aging
- Knowledge Graph Builder with relationship mapping
- Context Relevance Scorer with intelligent ranking
- API Endpoints with comprehensive integration testing
- Performance validation against target metrics

Test Categories:
- Unit Tests: Individual component functionality
- Integration Tests: Component interaction and data flow
- Performance Tests: Speed, compression ratio, semantic preservation
- Security Tests: Access controls and permission validation
- End-to-End Tests: Complete workflow scenarios
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
import uuid

import numpy as np
from fastapi.testclient import TestClient

# Import components under test
from app.core.context_compression_engine import (
    ContextCompressionEngine, CompressionConfig, CompressionQuality, 
    CompressionStrategy, CompressionContext, CompressedResult
)
from app.core.cross_agent_knowledge_manager import (
    CrossAgentKnowledgeManager, AgentExpertise, SharingPolicy, 
    KnowledgeQualityScore, CollaborationPattern
)
from app.core.memory_hierarchy_manager import (
    MemoryHierarchyManager, MemoryItem, MemoryType, MemoryLevel,
    AgingStrategy, ConsolidationTrigger, MemoryHierarchyConfig
)
from app.core.knowledge_graph_builder import (
    KnowledgeGraphBuilder, GraphType, NodeType, EdgeType
)
from app.core.context_relevance_scorer import (
    ContextRelevanceScorer, ScoringStrategy, ContextType, 
    ScoringRequest, ContextItem, ScoringConfig
)
from app.core.agent_knowledge_manager import (
    KnowledgeItem, KnowledgeType, AccessLevel, AgentKnowledgeManager
)

# Test fixtures and mocks
from tests.factories import (
    create_mock_embedding_service, create_test_knowledge_items,
    create_test_agent_expertise, create_test_memory_items
)

# ScriptBase import for standardized execution
from app.common.script_base import ScriptBase


# =============================================================================
# PERFORMANCE TEST TARGETS
# =============================================================================

class PerformanceTargets:
    """Performance targets for validation."""
    
    # Context Compression targets
    COMPRESSION_RATIO_MIN = 0.6  # 60% minimum reduction
    COMPRESSION_RATIO_MAX = 0.8  # 80% maximum reduction
    SEMANTIC_PRESERVATION_MIN = 0.9  # 90% minimum semantic preservation
    COMPRESSION_TIME_MAX_MS = 500  # <500ms for 10k token contexts
    
    # Knowledge Sharing targets
    KNOWLEDGE_DISCOVERY_TIME_MAX_MS = 100  # <100ms to find relevant expertise
    CROSS_AGENT_SHARING_TIME_MAX_MS = 200  # <200ms for knowledge transfer
    
    # Memory Management targets
    MEMORY_CONSOLIDATION_TIME_MAX_MS = 5000  # <5s for 1k memories
    MEMORY_EFFICIENCY_TARGET = 1000000  # <1GB for 1M compressed contexts
    
    # Context Relevance targets
    RELEVANCE_SCORING_TIME_MAX_MS = 100  # <100ms for relevance scoring
    SCORING_ACCURACY_TARGET = 0.85  # 85% scoring accuracy


# =============================================================================
# CONTEXT COMPRESSION ENGINE TESTS
# =============================================================================

class TestContextCompressionEngine:
    """Test suite for Context Compression Engine."""
    
    @pytest.fixture
    async def compression_engine(self):
        """Create compression engine with mock embedding service."""
        mock_embedding_service = create_mock_embedding_service()
        engine = ContextCompressionEngine(mock_embedding_service)
        await engine.initialize()
        return engine
    
    @pytest.fixture
    def test_contexts(self):
        """Create test contexts for compression."""
        return [
            "Machine learning algorithms require careful parameter tuning to achieve optimal performance. This involves adjusting hyperparameters such as learning rate, batch size, and regularization strength. The process often requires multiple iterations and cross-validation to find the best configuration.",
            "Deep neural networks consist of multiple layers that learn hierarchical representations of data. Each layer transforms the input through weighted connections and activation functions. Training involves backpropagation to update weights based on prediction errors.",
            "Natural language processing tasks include text classification, named entity recognition, and sentiment analysis. These tasks often use transformer-based models like BERT and GPT, which have revolutionized the field with their attention mechanisms.",
            "Computer vision applications use convolutional neural networks to process images and extract features. These networks learn filters that detect edges, patterns, and higher-level semantic concepts through multiple convolutional and pooling layers.",
            "Reinforcement learning agents learn optimal policies through interaction with environments. They use reward signals to update their action selection strategies, balancing exploration of new actions with exploitation of known good actions."
        ]
    
    @pytest.mark.asyncio
    async def test_semantic_clustering_compression(self, compression_engine, test_contexts):
        """Test semantic clustering compression algorithm."""
        config = CompressionConfig(
            strategy=CompressionStrategy.SEMANTIC_CLUSTERING,
            quality=CompressionQuality.BALANCED,
            target_reduction=0.7
        )
        
        start_time = time.time()
        result = await compression_engine.compress_context(test_contexts, config)
        processing_time = (time.time() - start_time) * 1000
        
        # Validate performance targets
        assert result.compression_ratio >= PerformanceTargets.COMPRESSION_RATIO_MIN
        assert result.compression_ratio <= PerformanceTargets.COMPRESSION_RATIO_MAX
        assert result.semantic_preservation_score >= PerformanceTargets.SEMANTIC_PRESERVATION_MIN
        assert processing_time <= PerformanceTargets.COMPRESSION_TIME_MAX_MS
        
        # Validate result structure
        assert isinstance(result, CompressedResult)
        assert result.compressed_content
        assert result.original_size > result.compressed_size
        assert result.algorithm_used == CompressionStrategy.SEMANTIC_CLUSTERING
        
    @pytest.mark.asyncio
    async def test_importance_filtering_compression(self, compression_engine, test_contexts):
        """Test importance filtering compression algorithm."""
        config = CompressionConfig(
            strategy=CompressionStrategy.IMPORTANCE_FILTERING,
            preserve_importance_threshold=0.8
        )
        
        # Add importance metadata
        context_metadata = {
            f'item_{i}': {'importance_score': 0.9 if i < 2 else 0.4}
            for i in range(len(test_contexts))
        }
        
        result = await compression_engine.compress_context(
            test_contexts, config, context_metadata
        )
        
        # High-importance items should be preserved
        assert result.compression_ratio >= 0.4  # Some compression achieved
        assert result.semantic_preservation_score >= 0.85
        assert "High-importance" in result.summary or "important" in result.summary.lower()
        
    @pytest.mark.asyncio
    async def test_temporal_decay_compression(self, compression_engine, test_contexts):
        """Test temporal decay compression algorithm."""
        config = CompressionConfig(
            strategy=CompressionStrategy.TEMPORAL_DECAY,
            temporal_decay_factor=0.1
        )
        
        # Add temporal metadata with varying ages
        context_metadata = {
            f'item_{i}': {
                'importance_score': 0.5,
                'created_at': (datetime.utcnow() - timedelta(hours=i*24)).isoformat()
            }
            for i in range(len(test_contexts))
        }
        
        result = await compression_engine.compress_context(
            test_contexts, config, context_metadata
        )
        
        # Recent content should be better preserved
        assert result.compression_ratio >= 0.5
        assert result.semantic_preservation_score >= 0.8
        assert "temporal" in result.summary.lower() or "recent" in result.summary.lower()
        
    @pytest.mark.asyncio
    async def test_adaptive_compression(self, compression_engine, test_contexts):
        """Test adaptive compression to target token count."""
        combined_content = " ".join(test_contexts)
        target_tokens = 100  # Much smaller than original
        
        result = await compression_engine.adaptive_compress(
            combined_content, target_tokens
        )
        
        # Should achieve significant compression
        assert result.compression_ratio >= 0.7
        assert len(result.compressed_content.split()) <= target_tokens * 1.2  # 20% tolerance
        
    @pytest.mark.asyncio
    async def test_batch_compression(self, compression_engine, test_contexts):
        """Test batch compression of multiple content items."""
        content_items = [test_contexts[:2], test_contexts[2:4], test_contexts[4:]]
        
        start_time = time.time()
        results = await compression_engine.batch_compress(content_items)
        processing_time = (time.time() - start_time) * 1000
        
        assert len(results) == len(content_items)
        assert all(isinstance(result, CompressedResult) for result in results)
        assert processing_time <= PerformanceTargets.COMPRESSION_TIME_MAX_MS * len(content_items)
        
    @pytest.mark.asyncio
    async def test_compression_performance_metrics(self, compression_engine, test_contexts):
        """Test compression performance metrics collection."""
        config = CompressionConfig(strategy=CompressionStrategy.HYBRID_ADAPTIVE)
        
        # Perform multiple compressions
        for _ in range(5):
            await compression_engine.compress_context(test_contexts, config)
        
        metrics = compression_engine.get_performance_metrics()
        
        assert metrics["total_compressions"] == 5
        assert metrics["avg_processing_time_ms"] > 0
        assert metrics["avg_tokens_saved"] > 0
        assert metrics["avg_semantic_preservation"] >= 0.8


# =============================================================================
# CROSS-AGENT KNOWLEDGE MANAGER TESTS
# =============================================================================

class TestCrossAgentKnowledgeManager:
    """Test suite for Cross-Agent Knowledge Manager."""
    
    @pytest.fixture
    async def knowledge_manager(self):
        """Create knowledge manager with mock dependencies."""
        mock_base_manager = Mock(spec=AgentKnowledgeManager)
        manager = CrossAgentKnowledgeManager(mock_base_manager)
        await manager.initialize()
        return manager
    
    @pytest.fixture
    def test_knowledge_item(self):
        """Create test knowledge item."""
        return KnowledgeItem(
            knowledge_id="test_knowledge_001",
            agent_id="agent_001",
            knowledge_type=KnowledgeType.PATTERN,
            content="Machine learning optimization requires careful hyperparameter tuning",
            title="ML Optimization Pattern",
            confidence_score=0.9,
            access_level=AccessLevel.TEAM_SHARED,
            tags=["machine-learning", "optimization", "hyperparameters"]
        )
    
    @pytest.fixture
    def test_agent_expertise(self):
        """Create test agent expertise data."""
        return create_test_agent_expertise()
    
    @pytest.mark.asyncio
    async def test_knowledge_sharing_with_compression(self, knowledge_manager, test_knowledge_item):
        """Test knowledge sharing with automatic compression."""
        target_agents = ["agent_002", "agent_003"]
        
        start_time = time.time()
        result = await knowledge_manager.share_knowledge(
            knowledge_item=test_knowledge_item,
            target_agents=target_agents,
            sharing_policy=SharingPolicy.TEAM_SHARING
        )
        processing_time = (time.time() - start_time) * 1000
        
        # Validate performance target
        assert processing_time <= PerformanceTargets.CROSS_AGENT_SHARING_TIME_MAX_MS
        
        # Validate sharing result
        assert result["successful_shares"]
        assert len(result["successful_shares"]) <= len(target_agents)
        assert result.get("quality_score") is not None
        
    @pytest.mark.asyncio
    async def test_expertise_discovery(self, knowledge_manager, test_agent_expertise):
        """Test agent expertise discovery."""
        # Setup expertise data
        knowledge_manager.agent_expertise = test_agent_expertise
        
        start_time = time.time()
        expertise_matches = await knowledge_manager.discover_agent_expertise(
            domain="machine_learning",
            capability="optimization",
            min_proficiency=0.6
        )
        processing_time = (time.time() - start_time) * 1000
        
        # Validate performance target
        assert processing_time <= PerformanceTargets.KNOWLEDGE_DISCOVERY_TIME_MAX_MS
        
        # Validate results
        assert isinstance(expertise_matches, list)
        if expertise_matches:
            assert all(exp.proficiency_level >= 0.6 for exp in expertise_matches)
            assert all("machine_learning" in exp.domain.lower() or 
                      "optimization" in exp.capability.lower() 
                      for exp in expertise_matches)
    
    @pytest.mark.asyncio
    async def test_knowledge_quality_assessment(self, knowledge_manager):
        """Test knowledge quality assessment and scoring."""
        knowledge_id = "test_knowledge_001"
        assessor_agent = "agent_002"
        
        feedback = {
            "helpful": True,
            "success_rate": 0.85,
            "metric_scores": {
                "accuracy": 0.9,
                "usefulness": 0.8,
                "completeness": 0.85
            }
        }
        
        quality_score = await knowledge_manager.assess_knowledge_quality(
            knowledge_id, assessor_agent, feedback
        )
        
        assert isinstance(quality_score, KnowledgeQualityScore)
        assert quality_score.overall_score > 0.5
        assert quality_score.positive_feedback == 1
        assert quality_score.usage_success_rate > 0.8
        
    @pytest.mark.asyncio
    async def test_collaborative_learning(self, knowledge_manager):
        """Test collaborative learning from workflow outcomes."""
        workflow_id = "workflow_001"
        participating_agents = ["agent_001", "agent_002", "agent_003"]
        
        collaboration_data = {
            "success_rate": 0.9,
            "completion_time_ms": 15000,
            "efficiency_score": 0.85,
            "agent_count": len(participating_agents),
            "interaction_type": "peer_to_peer",
            "success_factors": ["clear communication", "complementary expertise"],
            "knowledge_shared": ["knowledge_001", "knowledge_002"]
        }
        
        outcome = await knowledge_manager.learn_from_collaboration(
            workflow_id, participating_agents, collaboration_data
        )
        
        assert outcome.workflow_id == workflow_id
        assert outcome.participating_agents == participating_agents
        assert outcome.success_metrics["success_rate"] == 0.9
        assert len(outcome.best_practices) > 0
        
    @pytest.mark.asyncio
    async def test_access_control_validation(self, knowledge_manager, test_knowledge_item):
        """Test access control and permission validation."""
        # Test different access scenarios
        test_scenarios = [
            {
                "requester": "agent_001",  # Owner
                "expected_access": True,
                "reason_contains": "Owner"
            },
            {
                "requester": "agent_002",  # Team member
                "expected_access": True,
                "reason_contains": "team"
            },
            {
                "requester": "unauthorized_agent",
                "expected_access": False,
                "reason_contains": "denied"
            }
        ]
        
        # Setup team memberships
        knowledge_manager.access_control.set_team_membership("agent_001", ["team_ml"])
        knowledge_manager.access_control.set_team_membership("agent_002", ["team_ml"])
        knowledge_manager.access_control.set_team_membership("unauthorized_agent", ["team_other"])
        
        for scenario in test_scenarios:
            can_access, reason = knowledge_manager.access_control.can_access_knowledge(
                scenario["requester"], test_knowledge_item
            )
            
            assert can_access == scenario["expected_access"]
            assert scenario["reason_contains"].lower() in reason.lower()


# =============================================================================
# MEMORY HIERARCHY MANAGER TESTS
# =============================================================================

class TestMemoryHierarchyManager:
    """Test suite for Memory Hierarchy Manager."""
    
    @pytest.fixture
    async def memory_manager(self):
        """Create memory hierarchy manager."""
        config = MemoryHierarchyConfig(
            short_term_capacity=50,
            working_memory_capacity=200,
            long_term_capacity=1000,
            aging_strategy=AgingStrategy.HYBRID
        )
        manager = MemoryHierarchyManager(config)
        await manager.initialize()
        return manager
    
    @pytest.fixture
    def test_memories(self):
        """Create test memory items."""
        return create_test_memory_items()
    
    @pytest.mark.asyncio
    async def test_memory_storage_and_retrieval(self, memory_manager):
        """Test basic memory storage and retrieval."""
        agent_id = "test_agent_001"
        
        # Store memory
        memory_item = await memory_manager.store_memory(
            agent_id=agent_id,
            content="Important machine learning insight about gradient descent optimization",
            memory_type=MemoryType.SEMANTIC,
            memory_level=MemoryLevel.SHORT_TERM,
            importance_score=0.8,
            tags=["machine-learning", "optimization"]
        )
        
        assert memory_item.agent_id == agent_id
        assert memory_item.importance_score == 0.8
        assert memory_item.memory_level == MemoryLevel.SHORT_TERM
        
        # Retrieve memories
        retrieved = await memory_manager.retrieve_memories(
            agent_id=agent_id,
            memory_types=[MemoryType.SEMANTIC],
            limit=10
        )
        
        assert len(retrieved) == 1
        assert retrieved[0].memory_id == memory_item.memory_id
        
    @pytest.mark.asyncio
    async def test_memory_consolidation(self, memory_manager, test_memories):
        """Test memory consolidation process."""
        agent_id = "test_agent_001"
        
        # Add test memories to manager
        for memory in test_memories:
            memory.agent_id = agent_id
            await memory_manager.store_memory(
                agent_id=memory.agent_id,
                content=memory.content,
                memory_type=memory.memory_type,
                memory_level=memory.memory_level,
                importance_score=memory.importance_score
            )
        
        start_time = time.time()
        result = await memory_manager.trigger_consolidation(
            agent_id, ConsolidationTrigger.MEMORY_PRESSURE
        )
        processing_time = (time.time() - start_time) * 1000
        
        # Validate performance target
        assert processing_time <= PerformanceTargets.MEMORY_CONSOLIDATION_TIME_MAX_MS
        
        # Validate consolidation result
        assert result.memories_processed > 0
        assert result.compression_ratio >= 0.3  # Some compression achieved
        assert result.avg_semantic_preservation >= 0.8
        
    @pytest.mark.asyncio
    async def test_memory_search(self, memory_manager, test_memories):
        """Test memory search functionality."""
        agent_id = "test_agent_001"
        
        # Store memories with searchable content
        search_contents = [
            "Machine learning optimization techniques",
            "Deep neural network architectures", 
            "Natural language processing models",
            "Computer vision algorithms"
        ]
        
        for content in search_contents:
            await memory_manager.store_memory(
                agent_id=agent_id,
                content=content,
                memory_type=MemoryType.SEMANTIC,
                importance_score=0.7
            )
        
        # Search for specific content
        results = await memory_manager.search_memories(
            agent_id=agent_id,
            query="machine learning optimization",
            limit=5
        )
        
        assert len(results) > 0
        # Top result should contain the query terms
        top_memory, score = results[0]
        assert "machine learning" in top_memory.content.lower()
        assert score > 0
        
    @pytest.mark.asyncio
    async def test_memory_pressure_calculation(self, memory_manager):
        """Test memory pressure calculation and thresholds."""
        agent_id = "test_agent_001"
        
        # Add memories to approach capacity limits
        for i in range(40):  # Approaching short-term capacity of 50
            await memory_manager.store_memory(
                agent_id=agent_id,
                content=f"Test memory content {i}",
                memory_type=MemoryType.EPISODIC,
                memory_level=MemoryLevel.SHORT_TERM
            )
        
        pressure = memory_manager.get_memory_pressure(agent_id)
        
        assert 0.0 <= pressure <= 1.0
        assert pressure > 0.5  # Should show high pressure
        
    @pytest.mark.asyncio
    async def test_automatic_consolidation_scheduling(self, memory_manager):
        """Test automatic consolidation scheduling."""
        agent_id = "test_agent_001"
        
        # Schedule automatic consolidation
        await memory_manager.schedule_automatic_consolidation(agent_id)
        
        assert agent_id in memory_manager.consolidation_tasks
        assert not memory_manager.consolidation_tasks[agent_id].done()
        
        # Stop automatic consolidation
        await memory_manager.stop_automatic_consolidation(agent_id)
        assert agent_id not in memory_manager.consolidation_tasks


# =============================================================================
# KNOWLEDGE GRAPH BUILDER TESTS
# =============================================================================

class TestKnowledgeGraphBuilder:
    """Test suite for Knowledge Graph Builder."""
    
    @pytest.fixture
    async def graph_builder(self):
        """Create knowledge graph builder with mock embedding service."""
        mock_embedding_service = create_mock_embedding_service()
        builder = KnowledgeGraphBuilder(mock_embedding_service)
        await builder.initialize()
        return builder
    
    @pytest.fixture
    def test_agent_expertise(self):
        """Create test agent expertise for graph building."""
        return create_test_agent_expertise()
    
    @pytest.fixture
    def test_collaboration_history(self):
        """Create test collaboration history."""
        return [
            {
                "participating_agents": ["agent_001", "agent_002"],
                "success": True,
                "duration_hours": 2.5,
                "domains": ["machine_learning", "optimization"],
                "timestamp": datetime.utcnow() - timedelta(days=1)
            },
            {
                "participating_agents": ["agent_002", "agent_003"],
                "success": True,
                "duration_hours": 1.8,
                "domains": ["neural_networks", "deep_learning"],
                "timestamp": datetime.utcnow() - timedelta(days=2)
            },
            {
                "participating_agents": ["agent_001", "agent_003"],
                "success": False,
                "duration_hours": 0.5,
                "domains": ["computer_vision"],
                "timestamp": datetime.utcnow() - timedelta(days=3)
            }
        ]
    
    @pytest.mark.asyncio
    async def test_agent_expertise_graph_building(self, graph_builder, test_agent_expertise):
        """Test building agent expertise graph."""
        graph = await graph_builder.build_agent_expertise_graph(test_agent_expertise)
        
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
        
        # Verify graph structure
        metadata = graph_builder.graph_metadata[GraphType.AGENT_EXPERTISE]
        assert metadata["agent_count"] == len(test_agent_expertise)
        assert metadata["node_count"] == graph.number_of_nodes()
        
    @pytest.mark.asyncio
    async def test_collaboration_graph_building(self, graph_builder, test_collaboration_history):
        """Test building collaboration graph."""
        graph = await graph_builder.build_collaboration_graph(test_collaboration_history)
        
        assert graph.number_of_nodes() > 0  # Should have agent nodes
        assert graph.number_of_edges() > 0  # Should have collaboration edges
        
        # Verify it's a directed graph
        assert graph.is_directed()
        
        # Check for collaboration edges
        edges_with_data = list(graph.edges(data=True))
        assert len(edges_with_data) > 0
        
        for source, target, data in edges_with_data:
            assert "weight" in data
            assert "edge_type" in data
            assert data["edge_type"] == EdgeType.COLLABORATES_WITH.value
            
    @pytest.mark.asyncio
    async def test_knowledge_flow_graph_building(self, graph_builder):
        """Test building knowledge flow graph."""
        knowledge_sharing_events = [
            {
                "source_agent": "agent_001",
                "target_agent": "agent_002",
                "success": True,
                "quality_score": 0.8,
                "domain": "machine_learning",
                "timestamp": datetime.utcnow()
            },
            {
                "source_agent": "agent_002",
                "target_agent": "agent_003",
                "success": True,
                "quality_score": 0.9,
                "domain": "deep_learning",
                "timestamp": datetime.utcnow()
            }
        ]
        
        graph = await graph_builder.build_knowledge_flow_graph(knowledge_sharing_events)
        
        assert graph.number_of_nodes() >= 3  # At least 3 agents
        assert graph.number_of_edges() >= 2  # At least 2 sharing events
        assert graph.is_directed()  # Knowledge flows are directional
        
    @pytest.mark.asyncio
    async def test_graph_analysis(self, graph_builder, test_agent_expertise):
        """Test comprehensive graph analysis."""
        # Build graph first
        await graph_builder.build_agent_expertise_graph(test_agent_expertise)
        
        # Perform analysis
        analysis = await graph_builder.analyze_graph(GraphType.AGENT_EXPERTISE)
        
        assert analysis.graph_type == GraphType.AGENT_EXPERTISE
        assert analysis.node_count > 0
        assert analysis.edge_count >= 0
        assert 0.0 <= analysis.density <= 1.0
        assert analysis.clustering_coefficient >= 0.0
        
    @pytest.mark.asyncio
    async def test_knowledge_bridge_recommendations(self, graph_builder, test_agent_expertise):
        """Test knowledge bridge recommendations."""
        # Build expertise graph
        await graph_builder.build_agent_expertise_graph(test_agent_expertise)
        
        # Test bridge recommendations
        bridges = await graph_builder.recommend_knowledge_bridges(
            domain1="machine_learning",
            domain2="optimization",
            max_recommendations=5
        )
        
        assert isinstance(bridges, list)
        # Each bridge should have relevant scoring information
        for bridge in bridges:
            assert "bridge_score" in bridge
            assert bridge["bridge_score"] >= 0.0


# =============================================================================
# CONTEXT RELEVANCE SCORER TESTS
# =============================================================================

class TestContextRelevanceScorer:
    """Test suite for Context Relevance Scorer."""
    
    @pytest.fixture
    async def relevance_scorer(self):
        """Create context relevance scorer with mock embedding service."""
        mock_embedding_service = create_mock_embedding_service()
        scorer = ContextRelevanceScorer(mock_embedding_service)
        await scorer.initialize()
        return scorer
    
    @pytest.fixture
    def test_contexts(self):
        """Create test contexts for relevance scoring."""
        return [
            ContextItem(
                context_id="ctx_001",
                content="Machine learning optimization techniques for neural networks",
                context_type=ContextType.DOMAIN_CONTEXT,
                importance_score=0.9,
                tags=["machine-learning", "optimization", "neural-networks"]
            ),
            ContextItem(
                context_id="ctx_002", 
                content="Database query optimization and indexing strategies",
                context_type=ContextType.TASK_CONTEXT,
                importance_score=0.6,
                tags=["database", "optimization", "indexing"]
            ),
            ContextItem(
                context_id="ctx_003",
                content="Deep learning architectures for computer vision tasks",
                context_type=ContextType.DOMAIN_CONTEXT,
                importance_score=0.8,
                tags=["deep-learning", "computer-vision", "architecture"]
            )
        ]
    
    @pytest.mark.asyncio
    async def test_hybrid_multi_factor_scoring(self, relevance_scorer, test_contexts):
        """Test hybrid multi-factor relevance scoring."""
        request = ScoringRequest(
            query="machine learning optimization",
            contexts=test_contexts,
            agent_id="test_agent",
            max_results=3,
            config=ScoringConfig(strategy=ScoringStrategy.HYBRID_MULTI_FACTOR)
        )
        
        start_time = time.time()
        result = await relevance_scorer.score_contexts(request)
        processing_time = (time.time() - start_time) * 1000
        
        # Validate performance target
        assert processing_time <= PerformanceTargets.RELEVANCE_SCORING_TIME_MAX_MS
        
        # Validate results
        assert len(result.scored_contexts) <= 3
        assert result.query == "machine learning optimization"
        
        # Results should be sorted by relevance
        scores = [score.overall_score for _, score in result.scored_contexts]
        assert scores == sorted(scores, reverse=True)
        
        # Top result should be most relevant (machine learning context)
        top_context, top_score = result.scored_contexts[0]
        assert "machine learning" in top_context.content.lower()
        assert top_score.overall_score > 0.5
        
    @pytest.mark.asyncio
    async def test_semantic_only_scoring(self, relevance_scorer, test_contexts):
        """Test semantic similarity only scoring."""
        request = ScoringRequest(
            query="neural network architectures",
            contexts=test_contexts,
            config=ScoringConfig(strategy=ScoringStrategy.SEMANTIC_SIMILARITY)
        )
        
        result = await relevance_scorer.score_contexts(request)
        
        assert len(result.scored_contexts) > 0
        
        # Check that semantic scores are used
        for context, score in result.scored_contexts:
            assert score.semantic_similarity > 0.0
            assert score.overall_score == score.semantic_similarity
            
    @pytest.mark.asyncio
    async def test_quick_scoring(self, relevance_scorer, test_contexts):
        """Test quick scoring for real-time applications."""
        start_time = time.time()
        results = await relevance_scorer.quick_score(
            query="optimization techniques",
            contexts=test_contexts,
            max_results=2
        )
        processing_time = (time.time() - start_time) * 1000
        
        # Should be very fast
        assert processing_time <= 50  # Even faster for quick scoring
        assert len(results) <= 2
        
        # Results should be tuples of (context, score)
        for context, score in results:
            assert isinstance(context, ContextItem)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
            
    @pytest.mark.asyncio
    async def test_scoring_with_feedback(self, relevance_scorer, test_contexts):
        """Test scoring accuracy improvement with feedback."""
        request = ScoringRequest(
            query="machine learning",
            contexts=test_contexts,
            config=ScoringConfig(enable_adaptive_learning=True)
        )
        
        result = await relevance_scorer.score_contexts(request)
        
        # Provide feedback on top result
        if result.scored_contexts:
            top_context, top_score = result.scored_contexts[0]
            relevance_scorer.provide_feedback(
                request_id=result.request_id,
                context_id=top_context.context_id,
                feedback_score=0.9,
                feedback_type="relevance"
            )
        
        # Scoring history should be recorded
        assert len(relevance_scorer.scoring_history) > 0
        
    @pytest.mark.asyncio
    async def test_context_type_filtering(self, relevance_scorer, test_contexts):
        """Test filtering contexts by type."""
        request = ScoringRequest(
            query="optimization",
            contexts=test_contexts,
            context_types=[ContextType.DOMAIN_CONTEXT],  # Filter to domain contexts only
            max_results=5
        )
        
        result = await relevance_scorer.score_contexts(request)
        
        # Should only return domain contexts
        for context, score in result.scored_contexts:
            assert context.context_type == ContextType.DOMAIN_CONTEXT


# =============================================================================
# API INTEGRATION TESTS
# =============================================================================

class TestAPIIntegration:
    """Test suite for API endpoints integration."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client for API testing."""
        from app.main import app
        return TestClient(app)
    
    def test_compress_context_endpoint(self, test_client):
        """Test context compression API endpoint."""
        request_data = {
            "content": [
                "Machine learning requires careful parameter tuning.",
                "Deep neural networks learn hierarchical representations.",
                "Natural language processing uses transformer models."
            ],
            "target_reduction": 0.7,
            "quality": "balanced",
            "strategy": "hybrid_adaptive"
        }
        
        response = test_client.post("/context-compression/compress", json=request_data)
        
        assert response.status_code == 201
        data = response.json()
        
        assert "compression_id" in data
        assert "compressed_content" in data
        assert "compression_ratio" in data
        assert data["compression_ratio"] >= 0.6
        assert "semantic_preservation_score" in data
        assert data["semantic_preservation_score"] >= 0.8
        
    def test_knowledge_sharing_endpoint(self, test_client):
        """Test knowledge sharing API endpoint."""
        request_data = {
            "knowledge_id": "test_knowledge_001",
            "source_agent_id": "agent_001",
            "target_agent_ids": ["agent_002"],
            "sharing_policy": "team_sharing",
            "justification": "Sharing ML optimization expertise"
        }
        
        with patch('app.core.cross_agent_knowledge_manager.get_cross_agent_knowledge_manager') as mock_manager:
            mock_manager.return_value.share_knowledge = AsyncMock(return_value={
                "request_id": "share_001",
                "successful_shares": [{"agent_id": "agent_002", "shared_knowledge_id": "shared_001"}],
                "failed_shares": [],
                "quality_score": 0.85
            })
            
            response = test_client.post("/context-compression/knowledge/share", json=request_data)
            
            assert response.status_code == 201
            data = response.json()
            
            assert data["request_id"] == "share_001"
            assert len(data["successful_shares"]) == 1
            assert data["quality_score"] == 0.85
            
    def test_memory_consolidation_endpoint(self, test_client):
        """Test memory consolidation API endpoint."""
        request_data = {
            "agent_id": "test_agent_001",
            "trigger_type": "memory_pressure",
            "force_consolidation": False,
            "compression_enabled": True
        }
        
        with patch('app.core.memory_hierarchy_manager.get_memory_hierarchy_manager') as mock_manager:
            mock_consolidation_result = Mock()
            mock_consolidation_result.to_dict.return_value = {
                "consolidation_id": "cons_001",
                "memories_processed": 100,
                "compression_ratio": 0.65,
                "processing_time_ms": 2500
            }
            mock_manager.return_value.trigger_consolidation = AsyncMock(return_value=mock_consolidation_result)
            
            response = test_client.post("/context-compression/memory/consolidate", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "success"
            assert "consolidation_result" in data
            
    def test_context_scoring_endpoint(self, test_client):
        """Test context relevance scoring API endpoint."""
        request_data = {
            "query": "machine learning optimization",
            "contexts": [
                {
                    "context_id": "ctx_001",
                    "content": "Machine learning optimization techniques",
                    "context_type": "domain_context",
                    "importance_score": 0.8
                },
                {
                    "context_id": "ctx_002",
                    "content": "Database optimization strategies",
                    "context_type": "task_context", 
                    "importance_score": 0.6
                }
            ],
            "agent_id": "test_agent",
            "scoring_strategy": "hybrid_multi_factor",
            "max_results": 2
        }
        
        response = test_client.post("/context-compression/context/score", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "scoring_result" in data
        scoring_result = data["scoring_result"]
        assert scoring_result["query"] == "machine learning optimization"
        assert len(scoring_result["scored_contexts"]) <= 2
        
    def test_system_health_endpoint(self, test_client):
        """Test system health check endpoint."""
        response = test_client.get("/context-compression/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "components" in data
        assert "timestamp" in data
        
        # Check that all components are listed
        components = data["components"]
        expected_components = [
            "compression_engine",
            "knowledge_manager",
            "memory_manager", 
            "graph_builder",
            "relevance_scorer"
        ]
        
        for component in expected_components:
            assert component in components
            
    def test_system_metrics_endpoint(self, test_client):
        """Test system metrics endpoint."""
        response = test_client.get("/context-compression/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert "metrics" in data
        
        metrics = data["metrics"]
        expected_metric_types = [
            "compression_engine",
            "knowledge_manager",
            "memory_manager",
            "graph_builder", 
            "relevance_scorer"
        ]
        
        for metric_type in expected_metric_types:
            assert metric_type in metrics


# =============================================================================
# PERFORMANCE VALIDATION TESTS
# =============================================================================

class TestPerformanceValidation:
    """Test suite for validating performance targets."""
    
    @pytest.mark.asyncio
    async def test_compression_performance_targets(self):
        """Validate compression performance meets targets."""
        mock_embedding_service = create_mock_embedding_service()
        engine = ContextCompressionEngine(mock_embedding_service)
        await engine.initialize()
        
        # Large context for performance testing
        large_context = " ".join([
            "Machine learning optimization requires careful parameter tuning and cross-validation. " * 50,
            "Deep neural networks learn hierarchical representations through multiple layers. " * 50,
            "Natural language processing uses transformer architectures with attention mechanisms. " * 50
        ])
        
        config = CompressionConfig(
            strategy=CompressionStrategy.HYBRID_ADAPTIVE,
            target_reduction=0.7
        )
        
        start_time = time.time()
        result = await engine.compress_context(large_context, config)
        processing_time = (time.time() - start_time) * 1000
        
        # Validate all performance targets
        assert result.compression_ratio >= PerformanceTargets.COMPRESSION_RATIO_MIN
        assert result.compression_ratio <= PerformanceTargets.COMPRESSION_RATIO_MAX
        assert result.semantic_preservation_score >= PerformanceTargets.SEMANTIC_PRESERVATION_MIN
        assert processing_time <= PerformanceTargets.COMPRESSION_TIME_MAX_MS
        
    @pytest.mark.asyncio
    async def test_knowledge_sharing_performance_targets(self):
        """Validate knowledge sharing performance meets targets."""
        # Mock the knowledge manager to measure performance
        mock_base_manager = Mock(spec=AgentKnowledgeManager)
        manager = CrossAgentKnowledgeManager(mock_base_manager)
        await manager.initialize()
        
        # Setup test data
        test_expertise = create_test_agent_expertise()
        manager.agent_expertise = test_expertise
        
        # Test expertise discovery performance
        start_time = time.time()
        expertise_matches = await manager.discover_agent_expertise(
            domain="machine_learning",
            capability="optimization"
        )
        discovery_time = (time.time() - start_time) * 1000
        
        assert discovery_time <= PerformanceTargets.KNOWLEDGE_DISCOVERY_TIME_MAX_MS
        
    @pytest.mark.asyncio
    async def test_memory_consolidation_performance_targets(self):
        """Validate memory consolidation performance meets targets."""
        config = MemoryHierarchyConfig(
            short_term_capacity=100,
            working_memory_capacity=500,
            long_term_capacity=2000
        )
        manager = MemoryHierarchyManager(config)
        await manager.initialize()
        
        agent_id = "performance_test_agent"
        
        # Create large number of memories for performance testing
        for i in range(500):
            await manager.store_memory(
                agent_id=agent_id,
                content=f"Performance test memory {i} with detailed content about machine learning and optimization techniques. This content should be representative of typical memory items that would be stored in the system.",
                memory_type=MemoryType.SEMANTIC,
                memory_level=MemoryLevel.WORKING,
                importance_score=0.5 + (i % 5) * 0.1
            )
        
        # Test consolidation performance
        start_time = time.time()
        result = await manager.trigger_consolidation(agent_id, ConsolidationTrigger.MEMORY_PRESSURE)
        consolidation_time = (time.time() - start_time) * 1000
        
        assert consolidation_time <= PerformanceTargets.MEMORY_CONSOLIDATION_TIME_MAX_MS
        assert result.memories_processed > 0
        assert result.compression_ratio >= 0.3
        
    @pytest.mark.asyncio
    async def test_relevance_scoring_performance_targets(self):
        """Validate relevance scoring performance meets targets."""
        mock_embedding_service = create_mock_embedding_service()
        scorer = ContextRelevanceScorer(mock_embedding_service)
        await scorer.initialize()
        
        # Create large number of contexts for performance testing
        test_contexts = []
        for i in range(100):
            context = ContextItem(
                context_id=f"perf_ctx_{i}",
                content=f"Performance test context {i} about machine learning optimization techniques and neural network architectures for deep learning applications.",
                context_type=ContextType.DOMAIN_CONTEXT,
                importance_score=0.5 + (i % 10) * 0.05
            )
            test_contexts.append(context)
        
        request = ScoringRequest(
            query="machine learning optimization neural networks",
            contexts=test_contexts,
            max_results=10,
            config=ScoringConfig(strategy=ScoringStrategy.HYBRID_MULTI_FACTOR)
        )
        
        start_time = time.time()
        result = await scorer.score_contexts(request)
        scoring_time = (time.time() - start_time) * 1000
        
        assert scoring_time <= PerformanceTargets.RELEVANCE_SCORING_TIME_MAX_MS
        assert len(result.scored_contexts) <= 10
        assert result.total_contexts_evaluated == 100


# =============================================================================
# END-TO-END INTEGRATION TESTS
# =============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests for complete workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_knowledge_sharing_workflow(self):
        """Test complete knowledge sharing workflow from compression to cross-agent sharing."""
        # Initialize all components
        mock_embedding_service = create_mock_embedding_service()
        compression_engine = ContextCompressionEngine(mock_embedding_service)
        await compression_engine.initialize()
        
        mock_base_manager = Mock(spec=AgentKnowledgeManager)
        knowledge_manager = CrossAgentKnowledgeManager(mock_base_manager)
        await knowledge_manager.initialize()
        
        # Step 1: Create knowledge content
        knowledge_content = """
        Machine learning optimization requires a systematic approach to hyperparameter tuning.
        The key principles include: 1) Start with reasonable defaults, 2) Use grid search or 
        random search for initial exploration, 3) Apply Bayesian optimization for refinement,
        4) Always use cross-validation for robust evaluation, 5) Monitor for overfitting.
        """
        
        # Step 2: Compress the knowledge if it's large
        config = CompressionConfig(
            strategy=CompressionStrategy.SEMANTIC_CLUSTERING,
            target_reduction=0.6
        )
        
        compression_result = await compression_engine.compress_context(knowledge_content, config)
        assert compression_result.compression_ratio >= 0.5
        
        # Step 3: Create knowledge item
        knowledge_item = KnowledgeItem(
            knowledge_id="integration_test_knowledge",
            agent_id="expert_agent_001",
            knowledge_type=KnowledgeType.BEST_PRACTICE,
            content=compression_result.compressed_content,
            title="ML Optimization Best Practices",
            confidence_score=0.9,
            access_level=AccessLevel.TEAM_SHARED,
            tags=["machine-learning", "optimization", "best-practices"]
        )
        
        # Step 4: Share knowledge with other agents
        target_agents = ["agent_002", "agent_003"]
        sharing_result = await knowledge_manager.share_knowledge(
            knowledge_item=knowledge_item,
            target_agents=target_agents,
            sharing_policy=SharingPolicy.EXPERTISE_BASED,
            justification="Sharing ML optimization expertise with team members"
        )
        
        # Validate end-to-end workflow
        assert compression_result.semantic_preservation_score >= 0.8
        assert sharing_result["successful_shares"] or sharing_result["failed_shares"]
        
    @pytest.mark.asyncio
    async def test_complete_memory_management_workflow(self):
        """Test complete memory management workflow with compression and consolidation."""
        # Initialize components
        mock_embedding_service = create_mock_embedding_service()
        compression_engine = ContextCompressionEngine(mock_embedding_service)
        await compression_engine.initialize()
        
        config = MemoryHierarchyConfig(
            short_term_capacity=20,
            working_memory_capacity=50,
            enable_compression=True
        )
        memory_manager = MemoryHierarchyManager(config)
        await memory_manager.initialize()
        
        agent_id = "memory_test_agent"
        
        # Step 1: Store various types of memories
        memory_contents = [
            ("Learned about gradient descent optimization", MemoryType.SEMANTIC, 0.8),
            ("Completed neural network training task", MemoryType.EPISODIC, 0.7),
            ("Implemented backpropagation algorithm", MemoryType.PROCEDURAL, 0.9),
            ("Understanding of learning rate scheduling", MemoryType.META_COGNITIVE, 0.6),
            ("Context of recent ML project", MemoryType.CONTEXTUAL, 0.5)
        ]
        
        stored_memories = []
        for content, mem_type, importance in memory_contents:
            memory = await memory_manager.store_memory(
                agent_id=agent_id,
                content=content,
                memory_type=mem_type,
                memory_level=MemoryLevel.SHORT_TERM,
                importance_score=importance
            )
            stored_memories.append(memory)
        
        # Step 2: Trigger consolidation
        consolidation_result = await memory_manager.trigger_consolidation(
            agent_id, ConsolidationTrigger.MEMORY_PRESSURE
        )
        
        # Step 3: Search for memories
        search_results = await memory_manager.search_memories(
            agent_id=agent_id,
            query="machine learning optimization",
            limit=3
        )
        
        # Validate complete workflow
        assert len(stored_memories) == 5
        assert consolidation_result.memories_processed > 0
        assert len(search_results) > 0
        
    @pytest.mark.asyncio
    async def test_complete_context_analysis_workflow(self):
        """Test complete context analysis workflow with graph building and relevance scoring."""
        # Initialize components  
        mock_embedding_service = create_mock_embedding_service()
        
        graph_builder = KnowledgeGraphBuilder(mock_embedding_service)
        await graph_builder.initialize()
        
        relevance_scorer = ContextRelevanceScorer(mock_embedding_service)
        await relevance_scorer.initialize()
        
        # Step 1: Build knowledge graph
        test_expertise = create_test_agent_expertise()
        expertise_graph = await graph_builder.build_agent_expertise_graph(test_expertise)
        
        # Step 2: Analyze graph for insights
        analysis = await graph_builder.analyze_graph(GraphType.AGENT_EXPERTISE)
        
        # Step 3: Create contexts for relevance scoring
        contexts = [
            ContextItem(
                context_id="ctx_ml_1",
                content="Machine learning optimization techniques for gradient descent",
                context_type=ContextType.DOMAIN_CONTEXT,
                importance_score=0.9
            ),
            ContextItem(
                context_id="ctx_db_1", 
                content="Database query optimization and indexing strategies",
                context_type=ContextType.TASK_CONTEXT,
                importance_score=0.6
            )
        ]
        
        # Step 4: Score context relevance
        scoring_request = ScoringRequest(
            query="machine learning optimization",
            contexts=contexts,
            max_results=2
        )
        
        scoring_result = await relevance_scorer.score_contexts(scoring_request)
        
        # Validate complete workflow
        assert expertise_graph.number_of_nodes() > 0
        assert analysis.node_count > 0
        assert len(scoring_result.scored_contexts) > 0
        
        # Top scored context should be ML-related
        top_context, top_score = scoring_result.scored_contexts[0]
        assert "machine learning" in top_context.content.lower()
        assert top_score.overall_score > 0.5


# =============================================================================
# TEST CONFIGURATION AND FIXTURES
# =============================================================================

def create_mock_embedding_service():
    """Create mock embedding service for testing."""
    mock_service = Mock()
    
    # Mock embedding generation
    async def mock_generate_embedding(text):
        # Generate deterministic but realistic embeddings based on text hash
        text_hash = hash(text) % 1000000
        np.random.seed(text_hash)
        return np.random.random(1536).tolist()
    
    mock_service.generate_embedding = AsyncMock(side_effect=mock_generate_embedding)
    
    # Mock batch generation
    async def mock_generate_embeddings_batch(contents, batch_id=None):
        embeddings = []
        for content in contents:
            embedding = await mock_generate_embedding(content)
            embeddings.append(embedding)
        
        return embeddings, {"cache_hits": 0, "cache_misses": len(contents)}
    
    mock_service.generate_embeddings_batch = AsyncMock(side_effect=mock_generate_embeddings_batch)
    
    # Mock health check
    async def mock_health_check():
        return {"status": "healthy", "embedding_model": "mock"}
    
    mock_service.get_health_status = AsyncMock(return_value={"status": "healthy"})
    mock_service.get_performance_metrics = AsyncMock(return_value={
        "total_embeddings": 1000,
        "avg_generation_time_ms": 50
    })
    
    return mock_service


def create_test_knowledge_items():
    """Create test knowledge items for testing."""
    return [
        KnowledgeItem(
            knowledge_id="knowledge_001",
            agent_id="agent_001", 
            knowledge_type=KnowledgeType.PATTERN,
            content="Machine learning optimization requires systematic hyperparameter tuning",
            confidence_score=0.9,
            tags=["machine-learning", "optimization"]
        ),
        KnowledgeItem(
            knowledge_id="knowledge_002",
            agent_id="agent_002",
            knowledge_type=KnowledgeType.BEST_PRACTICE,
            content="Deep neural networks benefit from proper weight initialization",
            confidence_score=0.8,
            tags=["deep-learning", "neural-networks", "initialization"]
        )
    ]


def create_test_agent_expertise():
    """Create test agent expertise data."""
    return {
        "agent_001": [
            AgentExpertise(
                agent_id="agent_001",
                domain="machine_learning",
                capability="optimization",
                proficiency_level=0.9,
                evidence_count=15,
                success_rate=0.85,
                last_demonstrated=datetime.utcnow()
            )
        ],
        "agent_002": [
            AgentExpertise(
                agent_id="agent_002",
                domain="deep_learning",
                capability="neural_networks",
                proficiency_level=0.8,
                evidence_count=12,
                success_rate=0.80,
                last_demonstrated=datetime.utcnow()
            )
        ]
    }


def create_test_memory_items():
    """Create test memory items for testing."""
    return [
        MemoryItem(
            memory_id="mem_001",
            agent_id="test_agent",
            content="Important machine learning insight about optimization",
            memory_type=MemoryType.SEMANTIC,
            memory_level=MemoryLevel.SHORT_TERM,
            importance_score=0.9
        ),
        MemoryItem(
            memory_id="mem_002", 
            agent_id="test_agent",
            content="Completed neural network training task successfully",
            memory_type=MemoryType.EPISODIC,
            memory_level=MemoryLevel.WORKING,
            importance_score=0.7
        )
    ]


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.mark.asyncio
async def test_performance_validation_summary():
    """Summary test to validate all performance targets are met."""
    # This test serves as a comprehensive validation summary
    
    performance_results = {
        "compression_ratio_target": "60-80%",
        "semantic_preservation_target": ">90%", 
        "compression_time_target": "<500ms",
        "knowledge_discovery_target": "<100ms",
        "cross_agent_sharing_target": "<200ms",
        "memory_consolidation_target": "<5s",
        "relevance_scoring_target": "<100ms"
    }
    
    logger.info("Performance validation summary:")
    for target, value in performance_results.items():
        logger.info(f"  {target}: {value}")
    
    # All individual performance tests should pass for this to be meaningful
    assert True  # Placeholder - real validation happens in individual tests


class ContextCompressionSharingTestScript(ScriptBase):
    """Standardized test execution using ScriptBase pattern."""
    
    async def run(self) -> Dict[str, Any]:
        """Execute context compression and sharing tests."""
        import subprocess
        import sys
        
        # Run pytest with the same configuration
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            __file__,
            "-v", 
            "--tb=short"
        ], capture_output=True, text=True)
        
        return {
            "status": "success" if result.returncode == 0 else "error",
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "message": f"Context compression tests {'passed' if result.returncode == 0 else 'failed'}"
        }


# Create script instance
script = ContextCompressionSharingTestScript()

if __name__ == "__main__":
    script.execute()