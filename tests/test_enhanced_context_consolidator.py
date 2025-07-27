"""
Comprehensive tests for Enhanced Context Consolidator - UltraCompressed Context Mode.

Tests include:
- Ultra compression functionality with 70% target reduction
- Real-time compression streaming
- Semantic clustering and compression strategies
- Adaptive threshold optimization
- Performance and memory efficiency
- Integration with existing context infrastructure
"""

import asyncio
import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from app.core.enhanced_context_consolidator import (
    UltraCompressedContextMode,
    CompressionStrategy,
    ContextPriority,
    CompressionMetrics,
    ContextCluster,
    get_ultra_compressed_context_mode,
    ultra_compress_agent_contexts,
    start_real_time_compression
)
from app.models.context import Context, ContextType
from app.core.context_compression import CompressionLevel, CompressedContext


class TestCompressionMetrics:
    """Test compression metrics functionality."""
    
    def test_compression_metrics_creation(self):
        """Test compression metrics creation and properties."""
        metrics = CompressionMetrics(
            original_token_count=1000,
            compressed_token_count=300,
            compression_ratio=0.7,
            semantic_similarity_loss=0.1,
            processing_time_ms=5000,
            contexts_merged=5,
            contexts_archived=2,
            strategy_used=CompressionStrategy.SEMANTIC_CLUSTER
        )
        
        assert metrics.original_token_count == 1000
        assert metrics.compressed_token_count == 300
        assert metrics.compression_ratio == 0.7
        assert metrics.strategy_used == CompressionStrategy.SEMANTIC_CLUSTER
    
    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation."""
        # High efficiency scenario
        metrics = CompressionMetrics(
            compression_ratio=0.8,
            processing_time_ms=1000,
            semantic_similarity_loss=0.1
        )
        efficiency = metrics.calculate_efficiency_score()
        assert 0.0 <= efficiency <= 1.0
        assert efficiency > 0.5  # Should be reasonably high
        
        # Low efficiency scenario
        metrics = CompressionMetrics(
            compression_ratio=0.2,
            processing_time_ms=15000,
            semantic_similarity_loss=0.8
        )
        efficiency = metrics.calculate_efficiency_score()
        assert 0.0 <= efficiency <= 1.0
        assert efficiency < 0.5  # Should be low


class TestContextCluster:
    """Test context cluster functionality."""
    
    @pytest.fixture
    def sample_contexts(self):
        """Sample contexts for testing."""
        contexts = []
        for i in range(3):
            context = Mock(spec=Context)
            context.id = uuid.uuid4()
            context.content = f"Sample content {i} with some text for testing"
            context.context_type = ContextType.CONVERSATION
            context.importance_score = 0.5 + (i * 0.1)
            context.created_at = datetime.utcnow() - timedelta(hours=i)
            context.access_count = 5 + i
            contexts.append(context)
        return contexts
    
    def test_context_cluster_creation(self, sample_contexts):
        """Test context cluster creation and properties."""
        cluster = ContextCluster(
            cluster_id="test_cluster_123",
            representative_context=sample_contexts[0],
            similar_contexts=sample_contexts[1:],
            similarity_scores=[0.85, 0.78],
            priority_level=ContextPriority.MEDIUM
        )
        
        assert cluster.cluster_id == "test_cluster_123"
        assert cluster.size == 3  # 1 representative + 2 similar
        assert cluster.avg_similarity == 0.815  # (0.85 + 0.78) / 2
        assert cluster.priority_level == ContextPriority.MEDIUM
    
    def test_cluster_token_calculation(self, sample_contexts):
        """Test cluster total token calculation."""
        cluster = ContextCluster(
            cluster_id="test_cluster",
            representative_context=sample_contexts[0],
            similar_contexts=sample_contexts[1:],
            similarity_scores=[0.85, 0.78]
        )
        
        expected_tokens = sum(len(ctx.content) for ctx in sample_contexts)
        assert cluster.total_tokens == expected_tokens


class TestUltraCompressedContextMode:
    """Test ultra compressed context mode functionality."""
    
    @pytest.fixture
    def ultra_compressor(self):
        """Ultra compressed context mode instance for testing."""
        with patch('app.core.enhanced_context_consolidator.ContextManager') as mock_cm, \
             patch('app.core.enhanced_context_consolidator.get_context_consolidator') as mock_consolidator, \
             patch('app.core.enhanced_context_consolidator.get_context_compressor') as mock_compressor, \
             patch('app.core.enhanced_context_consolidator.get_embedding_service') as mock_embedding:
            
            # Mock context manager
            mock_cm_instance = AsyncMock()
            mock_cm.return_value = mock_cm_instance
            
            # Mock consolidator
            mock_consolidator_instance = AsyncMock()
            mock_consolidator.return_value = mock_consolidator_instance
            
            # Mock compressor
            mock_compressor_instance = AsyncMock()
            mock_compressor.return_value = mock_compressor_instance
            
            # Mock embedding service
            mock_embedding_instance = AsyncMock()
            mock_embedding.return_value = mock_embedding_instance
            
            compressor = UltraCompressedContextMode(
                context_manager=mock_cm_instance,
                context_consolidator=mock_consolidator_instance,
                context_compressor=mock_compressor_instance,
                embedding_service=mock_embedding_instance
            )
            
            return compressor
    
    @pytest.fixture
    def sample_agent_id(self):
        """Sample agent ID for testing."""
        return uuid.uuid4()
    
    @pytest.fixture
    def sample_contexts_for_compression(self):
        """Sample contexts for compression testing."""
        contexts = []
        for i in range(10):
            context = Mock(spec=Context)
            context.id = uuid.uuid4()
            context.content = f"This is sample context content number {i}. " * 10  # Make it substantial
            context.context_type = ContextType.CONVERSATION
            context.importance_score = 0.3 + (i % 3) * 0.2  # Varying importance
            context.created_at = datetime.utcnow() - timedelta(hours=2 + i)
            context.access_count = 1 + (i % 5)
            context.is_consolidated = False
            context.is_archived = False
            context.embedding = [0.1 * j for j in range(1536)]  # Mock embedding
            contexts.append(context)
        return contexts
    
    @pytest.mark.asyncio
    async def test_ultra_compress_agent_contexts_basic(self, ultra_compressor, sample_agent_id, sample_contexts_for_compression):
        """Test basic ultra compression functionality."""
        # Mock the context retrieval
        ultra_compressor._get_compressible_contexts = AsyncMock(
            return_value=sample_contexts_for_compression
        )
        
        # Mock cluster building
        clusters = [
            ContextCluster(
                cluster_id="cluster_1",
                representative_context=sample_contexts_for_compression[0],
                similar_contexts=sample_contexts_for_compression[1:3],
                similarity_scores=[0.85, 0.78]
            ),
            ContextCluster(
                cluster_id="cluster_2",
                representative_context=sample_contexts_for_compression[3],
                similar_contexts=sample_contexts_for_compression[4:6],
                similarity_scores=[0.82, 0.75]
            )
        ]
        ultra_compressor._build_semantic_clusters = AsyncMock(return_value=clusters)
        
        # Mock compression strategy application
        compression_results = [
            {
                "compressed_content": "Compressed content 1",
                "original_tokens": 500,
                "compressed_tokens": 150,
                "contexts_merged": 3,
                "contexts_archived": 0,
                "compression_ratio": 0.7,
                "strategy_used": CompressionStrategy.SEMANTIC_CLUSTER,
                "cluster_id": "cluster_1"
            },
            {
                "compressed_content": "Compressed content 2", 
                "original_tokens": 400,
                "compressed_tokens": 120,
                "contexts_merged": 3,
                "contexts_archived": 0,
                "compression_ratio": 0.7,
                "strategy_used": CompressionStrategy.HIERARCHICAL_COMPRESS,
                "cluster_id": "cluster_2"
            }
        ]
        ultra_compressor._apply_compression_strategies = AsyncMock(return_value=compression_results)
        ultra_compressor._commit_compression_results = AsyncMock()
        
        # Perform compression
        metrics = await ultra_compressor.ultra_compress_agent_contexts(
            agent_id=sample_agent_id,
            target_reduction=0.70
        )
        
        # Verify results
        assert metrics.compression_ratio >= 0.6  # Should achieve good compression
        assert metrics.contexts_merged == 6  # 3 + 3 from both clusters
        assert metrics.processing_time_ms > 0
        assert metrics.original_token_count > metrics.compressed_token_count
        
        # Verify method calls
        ultra_compressor._get_compressible_contexts.assert_called_once_with(sample_agent_id)
        ultra_compressor._build_semantic_clusters.assert_called_once()
        ultra_compressor._apply_compression_strategies.assert_called_once()
        ultra_compressor._commit_compression_results.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_ultra_compress_no_contexts(self, ultra_compressor, sample_agent_id):
        """Test ultra compression when no contexts are available."""
        # Mock empty context retrieval
        ultra_compressor._get_compressible_contexts = AsyncMock(return_value=[])
        
        metrics = await ultra_compressor.ultra_compress_agent_contexts(sample_agent_id)
        
        assert metrics.original_token_count == 0
        assert metrics.compressed_token_count == 0
        assert metrics.compression_ratio == 0.0
        assert metrics.contexts_merged == 0
    
    @pytest.mark.asyncio
    async def test_real_time_compression_stream(self, ultra_compressor, sample_agent_id):
        """Test real-time compression streaming."""
        # Mock compression need detection
        ultra_compressor._should_compress = AsyncMock(side_effect=[True, True, False])
        
        # Mock compression results
        mock_metrics = CompressionMetrics(
            original_token_count=1000,
            compressed_token_count=300,
            compression_ratio=0.7,
            processing_time_ms=2000
        )
        ultra_compressor.ultra_compress_agent_contexts = AsyncMock(return_value=mock_metrics)
        
        # Test streaming for a short duration
        compression_count = 0
        async for metrics in ultra_compressor.real_time_compression_stream(
            agent_id=sample_agent_id,
            compression_interval_minutes=0.01  # Very short interval for testing
        ):
            compression_count += 1
            assert isinstance(metrics, CompressionMetrics)
            assert metrics.compression_ratio == 0.7
            
            if compression_count >= 2:  # Limit iterations for testing
                break
        
        assert compression_count == 2
        assert ultra_compressor.ultra_compress_agent_contexts.call_count == 2
    
    @pytest.mark.asyncio
    async def test_adaptive_threshold_optimization(self, ultra_compressor, sample_agent_id):
        """Test adaptive threshold optimization."""
        # Add some compression history
        for i in range(5):
            metrics = CompressionMetrics(
                compression_ratio=0.5 + i * 0.05,  # Increasing compression ratios
                processing_time_ms=5000 + i * 1000,  # Increasing processing times
                semantic_similarity_loss=0.1
            )
            ultra_compressor._compression_history.append(metrics)
        
        # Optimize thresholds
        optimized_thresholds = await ultra_compressor.adaptive_threshold_optimization(sample_agent_id)
        
        assert isinstance(optimized_thresholds, dict)
        assert "similarity_threshold" in optimized_thresholds
        assert "cluster_size_limit" in optimized_thresholds
        assert "compression_aggressiveness" in optimized_thresholds
        
        # Verify threshold values are reasonable
        assert 0.5 <= optimized_thresholds["similarity_threshold"] <= 1.0
        assert 3 <= optimized_thresholds["cluster_size_limit"] <= 20
        assert 0.2 <= optimized_thresholds["compression_aggressiveness"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_get_compression_analytics(self, ultra_compressor, sample_agent_id):
        """Test compression analytics retrieval."""
        # Add compression history
        for i in range(3):
            metrics = CompressionMetrics(
                compression_ratio=0.6 + i * 0.1,
                processing_time_ms=2000 + i * 500,
                contexts_merged=3 + i,
                strategy_used=CompressionStrategy.SEMANTIC_CLUSTER
            )
            ultra_compressor._compression_history.append(metrics)
        
        # Mock database session for agent-specific analytics
        with patch('app.core.enhanced_context_consolidator.get_async_session') as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__anext__ = AsyncMock(return_value=mock_db_session)
            
            # Mock database queries
            mock_db_session.scalar = AsyncMock(side_effect=[50, 35])  # total, compressed
            
            analytics = await ultra_compressor.get_compression_analytics(sample_agent_id)
            
            assert "performance_metrics" in analytics
            assert "adaptive_thresholds" in analytics
            assert "recent_compressions" in analytics
            assert "agent_specific" in analytics
            
            # Check agent-specific data
            agent_data = analytics["agent_specific"]
            assert agent_data["agent_id"] == str(sample_agent_id)
            assert agent_data["total_contexts"] == 50
            assert agent_data["compressed_contexts"] == 35
            assert 0.0 <= agent_data["compression_coverage"] <= 1.0
            
            # Check recent compressions
            assert len(analytics["recent_compressions"]) == 3
            for compression in analytics["recent_compressions"]:
                assert "compression_ratio" in compression
                assert "efficiency_score" in compression
                assert "strategy_used" in compression
    
    @pytest.mark.asyncio
    async def test_semantic_clustering(self, ultra_compressor, sample_contexts_for_compression):
        """Test semantic clustering functionality."""
        # Mock similarity calculation
        ultra_compressor._calculate_semantic_similarity = AsyncMock(
            side_effect=lambda ctx1, ctx2: 0.9 if ctx1.id != ctx2.id else 1.0
        )
        
        # Mock priority determination
        ultra_compressor._determine_context_priority = Mock(return_value=ContextPriority.MEDIUM)
        
        clusters = await ultra_compressor._build_semantic_clusters(sample_contexts_for_compression)
        
        assert len(clusters) > 0
        for cluster in clusters:
            assert cluster.size >= 2  # Only clusters with multiple contexts
            assert cluster.avg_similarity > 0.0
            assert isinstance(cluster.priority_level, ContextPriority)
    
    @pytest.mark.asyncio
    async def test_compression_strategy_selection(self, ultra_compressor):
        """Test compression strategy selection logic."""
        # Test critical priority
        critical_cluster = Mock()
        critical_cluster.priority_level = ContextPriority.CRITICAL
        critical_cluster.size = 10
        critical_cluster.avg_similarity = 0.95
        
        strategy = ultra_compressor._select_compression_strategy(
            critical_cluster, 0.7, preserve_critical=True
        )
        assert strategy == CompressionStrategy.LIGHT_OPTIMIZATION
        
        # Test high priority
        high_cluster = Mock()
        high_cluster.priority_level = ContextPriority.HIGH
        high_cluster.size = 5
        high_cluster.avg_similarity = 0.85
        
        strategy = ultra_compressor._select_compression_strategy(
            high_cluster, 0.7, preserve_critical=True
        )
        assert strategy == CompressionStrategy.INTELLIGENT_SUMMARY
        
        # Test aggressive merge conditions
        aggressive_cluster = Mock()
        aggressive_cluster.priority_level = ContextPriority.LOW
        aggressive_cluster.size = 8
        aggressive_cluster.avg_similarity = 0.9
        
        strategy = ultra_compressor._select_compression_strategy(
            aggressive_cluster, 0.8, preserve_critical=True
        )
        assert strategy == CompressionStrategy.AGGRESSIVE_MERGE
    
    @pytest.mark.asyncio
    async def test_context_priority_determination(self, ultra_compressor):
        """Test context priority determination logic."""
        # Test critical context type
        critical_context = Mock(spec=Context)
        critical_context.context_type = ContextType.DECISION
        critical_context.importance_score = 0.5
        critical_context.access_count = 5
        critical_context.created_at = datetime.utcnow() - timedelta(days=1)
        
        critical_cluster = Mock()
        critical_cluster.representative_context = critical_context
        
        priority = ultra_compressor._determine_context_priority(critical_cluster)
        assert priority == ContextPriority.CRITICAL
        
        # Test high importance
        high_importance_context = Mock(spec=Context)
        high_importance_context.context_type = ContextType.CONVERSATION
        high_importance_context.importance_score = 0.9
        high_importance_context.access_count = 5
        high_importance_context.created_at = datetime.utcnow() - timedelta(days=1)
        
        high_cluster = Mock()
        high_cluster.representative_context = high_importance_context
        
        priority = ultra_compressor._determine_context_priority(high_cluster)
        assert priority == ContextPriority.HIGH
        
        # Test old context
        old_context = Mock(spec=Context)
        old_context.context_type = ContextType.CONVERSATION
        old_context.importance_score = 0.3
        old_context.access_count = 2
        old_context.created_at = datetime.utcnow() - timedelta(days=35)
        
        old_cluster = Mock()
        old_cluster.representative_context = old_context
        
        priority = ultra_compressor._determine_context_priority(old_cluster)
        assert priority == ContextPriority.LOW
    
    @pytest.mark.asyncio
    async def test_compression_with_mock_compressor(self, ultra_compressor):
        """Test compression with mocked compressor service."""
        # Create test cluster
        test_context = Mock(spec=Context)
        test_context.content = "Test content for compression"
        test_context.context_type = ContextType.CONVERSATION
        
        test_cluster = ContextCluster(
            cluster_id="test_cluster",
            representative_context=test_context,
            similar_contexts=[],
            similarity_scores=[]
        )
        
        # Mock compressor result
        mock_compressed_result = Mock()
        mock_compressed_result.summary = "Compressed test content"
        mock_compressed_result.original_token_count = 100
        mock_compressed_result.compressed_token_count = 30
        mock_compressed_result.compression_ratio = 0.7
        mock_compressed_result.key_insights = ["Test insight"]
        mock_compressed_result.decisions_made = ["Test decision"]
        mock_compressed_result.patterns_identified = ["Test pattern"]
        
        ultra_compressor.compressor.compress_conversation = AsyncMock(
            return_value=mock_compressed_result
        )
        
        # Test semantic cluster compression
        result = await ultra_compressor._semantic_cluster_compress(test_cluster)
        
        assert result["compressed_content"] == "Compressed test content"
        assert result["compression_ratio"] == 0.7
        assert result["contexts_merged"] == 1
        assert "key_insights" in result
        assert "decisions_made" in result
    
    @pytest.mark.asyncio
    async def test_should_compress_logic(self, ultra_compressor, sample_agent_id):
        """Test compression need detection logic."""
        with patch('app.core.enhanced_context_consolidator.get_async_session') as mock_session:
            mock_db_session = AsyncMock()
            mock_session.return_value.__anext__ = AsyncMock(return_value=mock_db_session)
            
            # Test with enough contexts (should compress)
            mock_db_session.scalar = AsyncMock(return_value=15)
            should_compress = await ultra_compressor._should_compress(sample_agent_id)
            assert should_compress is True
            
            # Test with few contexts (should not compress)
            mock_db_session.scalar = AsyncMock(return_value=5)
            should_compress = await ultra_compressor._should_compress(sample_agent_id)
            assert should_compress is False
    
    @pytest.mark.asyncio
    async def test_performance_metrics_update(self, ultra_compressor):
        """Test performance metrics updating."""
        initial_count = ultra_compressor._performance_metrics["total_compressions"]
        
        test_metrics = CompressionMetrics(
            original_token_count=1000,
            compressed_token_count=300,
            compression_ratio=0.7,
            processing_time_ms=5000,
            strategy_used=CompressionStrategy.SEMANTIC_CLUSTER
        )
        
        ultra_compressor._update_performance_metrics(test_metrics)
        
        assert ultra_compressor._performance_metrics["total_compressions"] == initial_count + 1
        assert ultra_compressor._performance_metrics["total_tokens_saved"] == 700
        assert ultra_compressor._performance_metrics["avg_compression_ratio"] > 0
        assert ultra_compressor._performance_metrics["strategy_distribution"]["semantic_cluster"] == 1


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.mark.asyncio
    async def test_ultra_compress_agent_contexts_convenience(self):
        """Test convenience function for ultra compression."""
        with patch('app.core.enhanced_context_consolidator.get_ultra_compressed_context_mode') as mock_get:
            mock_compressor = AsyncMock()
            mock_metrics = CompressionMetrics(compression_ratio=0.75)
            mock_compressor.ultra_compress_agent_contexts = AsyncMock(return_value=mock_metrics)
            mock_get.return_value = mock_compressor
            
            agent_id = uuid.uuid4()
            result = await ultra_compress_agent_contexts(agent_id, 0.8)
            
            assert result.compression_ratio == 0.75
            mock_compressor.ultra_compress_agent_contexts.assert_called_once_with(agent_id, 0.8)
    
    @pytest.mark.asyncio
    async def test_start_real_time_compression_convenience(self):
        """Test convenience function for real-time compression."""
        with patch('app.core.enhanced_context_consolidator.get_ultra_compressed_context_mode') as mock_get:
            mock_compressor = AsyncMock()
            
            async def mock_stream(agent_id, interval):
                yield CompressionMetrics(compression_ratio=0.7)
                yield CompressionMetrics(compression_ratio=0.8)
            
            mock_compressor.real_time_compression_stream = mock_stream
            mock_get.return_value = mock_compressor
            
            agent_id = uuid.uuid4()
            results = []
            
            async for metrics in start_real_time_compression(agent_id, 15):
                results.append(metrics)
                if len(results) >= 2:
                    break
            
            assert len(results) == 2
            assert all(isinstance(m, CompressionMetrics) for m in results)


class TestIntegrationAndPerformance:
    """Test integration scenarios and performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_concurrent_compression_operations(self):
        """Test concurrent compression operations."""
        with patch('app.core.enhanced_context_consolidator.ContextManager'), \
             patch('app.core.enhanced_context_consolidator.get_context_consolidator'), \
             patch('app.core.enhanced_context_consolidator.get_context_compressor'), \
             patch('app.core.enhanced_context_consolidator.get_embedding_service'):
            
            compressor = UltraCompressedContextMode()
            
            # Mock methods to return quickly
            compressor._get_compressible_contexts = AsyncMock(return_value=[])
            
            # Create multiple concurrent compression tasks
            agent_ids = [uuid.uuid4() for _ in range(5)]
            tasks = [
                compressor.ultra_compress_agent_contexts(agent_id)
                for agent_id in agent_ids
            ]
            
            # Execute concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All should succeed
            for result in results:
                assert isinstance(result, CompressionMetrics)
                assert result.compression_ratio >= 0.0
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_with_large_context_sets(self):
        """Test memory efficiency with large numbers of contexts."""
        with patch('app.core.enhanced_context_consolidator.ContextManager'), \
             patch('app.core.enhanced_context_consolidator.get_context_consolidator'), \
             patch('app.core.enhanced_context_consolidator.get_context_compressor'), \
             patch('app.core.enhanced_context_consolidator.get_embedding_service'):
            
            compressor = UltraCompressedContextMode()
            
            # Create many mock contexts
            large_context_set = []
            for i in range(1000):
                context = Mock(spec=Context)
                context.id = uuid.uuid4()
                context.content = f"Context content {i}" * 50  # Substantial content
                context.context_type = ContextType.CONVERSATION
                context.importance_score = 0.5
                context.created_at = datetime.utcnow() - timedelta(hours=2)
                context.embedding = [0.1] * 1536
                large_context_set.append(context)
            
            compressor._get_compressible_contexts = AsyncMock(return_value=large_context_set)
            compressor._commit_compression_results = AsyncMock()
            
            # Mock clustering to return manageable clusters
            mock_clusters = []
            for i in range(0, len(large_context_set), 10):
                cluster = ContextCluster(
                    cluster_id=f"cluster_{i//10}",
                    representative_context=large_context_set[i],
                    similar_contexts=large_context_set[i+1:i+10],
                    similarity_scores=[0.85] * min(9, len(large_context_set) - i - 1)
                )
                mock_clusters.append(cluster)
            
            compressor._build_semantic_clusters = AsyncMock(return_value=mock_clusters[:50])  # Limit for testing
            
            # Mock compression results
            mock_compression_results = []
            for i, cluster in enumerate(mock_clusters[:50]):
                result = {
                    "compressed_content": f"Compressed cluster {i}",
                    "original_tokens": 1000,
                    "compressed_tokens": 300,
                    "contexts_merged": cluster.size,
                    "contexts_archived": 0,
                    "compression_ratio": 0.7,
                    "strategy_used": CompressionStrategy.SEMANTIC_CLUSTER,
                    "cluster_id": cluster.cluster_id
                }
                mock_compression_results.append(result)
            
            compressor._apply_compression_strategies = AsyncMock(return_value=mock_compression_results)
            
            # Perform compression
            agent_id = uuid.uuid4()
            metrics = await compressor.ultra_compress_agent_contexts(agent_id)
            
            # Should handle large dataset efficiently
            assert metrics.compression_ratio > 0.5
            assert metrics.processing_time_ms < 60000  # Should complete within 60 seconds
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and system recovery."""
        with patch('app.core.enhanced_context_consolidator.ContextManager'), \
             patch('app.core.enhanced_context_consolidator.get_context_consolidator'), \
             patch('app.core.enhanced_context_consolidator.get_context_compressor'), \
             patch('app.core.enhanced_context_consolidator.get_embedding_service'):
            
            compressor = UltraCompressedContextMode()
            
            # Test with context retrieval error
            compressor._get_compressible_contexts = AsyncMock(side_effect=Exception("Database error"))
            
            agent_id = uuid.uuid4()
            metrics = await compressor.ultra_compress_agent_contexts(agent_id)
            
            # Should return empty metrics without crashing
            assert metrics.original_token_count == 0
            assert metrics.compressed_token_count == 0
            assert metrics.compression_ratio == 0.0
            
            # Test with clustering error
            compressor._get_compressible_contexts = AsyncMock(return_value=[Mock()])
            compressor._build_semantic_clusters = AsyncMock(side_effect=Exception("Clustering error"))
            
            metrics = await compressor.ultra_compress_agent_contexts(agent_id)
            
            # Should handle error gracefully
            assert isinstance(metrics, CompressionMetrics)
    
    def test_compression_strategy_distribution(self):
        """Test that different compression strategies are used appropriately."""
        with patch('app.core.enhanced_context_consolidator.ContextManager'), \
             patch('app.core.enhanced_context_consolidator.get_context_consolidator'), \
             patch('app.core.enhanced_context_consolidator.get_context_compressor'), \
             patch('app.core.enhanced_context_consolidator.get_embedding_service'):
            
            compressor = UltraCompressedContextMode()
            
            # Test different cluster scenarios
            test_cases = [
                # (size, similarity, priority, expected_strategy)
                (10, 0.95, ContextPriority.LOW, CompressionStrategy.AGGRESSIVE_MERGE),
                (6, 0.87, ContextPriority.MEDIUM, CompressionStrategy.SEMANTIC_CLUSTER),
                (4, 0.80, ContextPriority.MEDIUM, CompressionStrategy.HIERARCHICAL_COMPRESS),
                (2, 0.75, ContextPriority.HIGH, CompressionStrategy.INTELLIGENT_SUMMARY),
                (5, 0.85, ContextPriority.CRITICAL, CompressionStrategy.LIGHT_OPTIMIZATION),
            ]
            
            for size, similarity, priority, expected_strategy in test_cases:
                mock_cluster = Mock()
                mock_cluster.size = size
                mock_cluster.avg_similarity = similarity
                mock_cluster.priority_level = priority
                
                strategy = compressor._select_compression_strategy(
                    mock_cluster, 0.7, preserve_critical=True
                )
                
                assert strategy == expected_strategy
    
    def test_adaptive_threshold_convergence(self):
        """Test that adaptive thresholds converge to optimal values."""
        with patch('app.core.enhanced_context_consolidator.ContextManager'), \
             patch('app.core.enhanced_context_consolidator.get_context_consolidator'), \
             patch('app.core.enhanced_context_consolidator.get_context_compressor'), \
             patch('app.core.enhanced_context_consolidator.get_embedding_service'):
            
            compressor = UltraCompressedContextMode()
            
            # Simulate poor performance scenario
            poor_metrics = [
                CompressionMetrics(
                    compression_ratio=0.3,  # Low compression
                    processing_time_ms=80000,  # High processing time
                    semantic_similarity_loss=0.8,  # High loss
                    strategy_used=CompressionStrategy.AGGRESSIVE_MERGE
                ) for _ in range(10)
            ]
            
            compressor._compression_history.extend(poor_metrics)
            
            # Get initial thresholds
            initial_thresholds = compressor._adaptive_thresholds.copy()
            
            # Optimize thresholds
            asyncio.run(compressor.adaptive_threshold_optimization(uuid.uuid4()))
            
            # Thresholds should adjust to be more conservative
            assert compressor._adaptive_thresholds["similarity_threshold"] >= initial_thresholds["similarity_threshold"]
            assert compressor._adaptive_thresholds["compression_aggressiveness"] <= initial_thresholds["compression_aggressiveness"]
            assert compressor._adaptive_thresholds["cluster_size_limit"] <= initial_thresholds["cluster_size_limit"]


class TestCompressionQuality:
    """Test compression quality and semantic preservation."""
    
    def test_similarity_calculation_methods(self):
        """Test different similarity calculation methods."""
        with patch('app.core.enhanced_context_consolidator.ContextManager'), \
             patch('app.core.enhanced_context_consolidator.get_context_consolidator'), \
             patch('app.core.enhanced_context_consolidator.get_context_compressor'), \
             patch('app.core.enhanced_context_consolidator.get_embedding_service'):
            
            compressor = UltraCompressedContextMode()
            
            # Test content similarity calculation
            content1 = "This is a test document about machine learning and AI"
            content2 = "This document discusses machine learning and artificial intelligence"
            
            similarity = compressor._calculate_content_similarity(content1, content2)
            
            assert 0.0 <= similarity <= 1.0
            assert similarity > 0.3  # Should have reasonable similarity
            
            # Test identical content
            identical_similarity = compressor._calculate_content_similarity(content1, content1)
            assert identical_similarity == 1.0
            
            # Test completely different content
            different_content = "The weather is sunny today with clear blue skies"
            different_similarity = compressor._calculate_content_similarity(content1, different_content)
            assert different_similarity < similarity  # Should be less similar
    
    @pytest.mark.asyncio
    async def test_semantic_similarity_with_embeddings(self):
        """Test semantic similarity calculation using embeddings."""
        with patch('app.core.enhanced_context_consolidator.ContextManager'), \
             patch('app.core.enhanced_context_consolidator.get_context_consolidator'), \
             patch('app.core.enhanced_context_consolidator.get_context_compressor'), \
             patch('app.core.enhanced_context_consolidator.get_embedding_service'):
            
            compressor = UltraCompressedContextMode()
            
            # Create contexts with mock embeddings
            context1 = Mock(spec=Context)
            context1.content = "Machine learning algorithms"
            context1.embedding = [0.1, 0.2, 0.3, 0.4]  # Similar direction
            
            context2 = Mock(spec=Context)
            context2.content = "AI and machine learning"
            context2.embedding = [0.15, 0.25, 0.35, 0.45]  # Very similar direction
            
            context3 = Mock(spec=Context)
            context3.content = "Weather forecast"
            context3.embedding = [-0.3, -0.2, -0.1, 0.0]  # Different direction
            
            # Test similarity between similar contexts
            similarity_high = await compressor._calculate_semantic_similarity(context1, context2)
            assert 0.8 <= similarity_high <= 1.0
            
            # Test similarity between different contexts
            similarity_low = await compressor._calculate_semantic_similarity(context1, context3)
            assert similarity_low < similarity_high
            
            # Test with missing embeddings (should fall back to content similarity)
            context_no_embedding = Mock(spec=Context)
            context_no_embedding.content = "Machine learning"
            context_no_embedding.embedding = None
            
            fallback_similarity = await compressor._calculate_semantic_similarity(
                context1, context_no_embedding
            )
            assert 0.0 <= fallback_similarity <= 1.0