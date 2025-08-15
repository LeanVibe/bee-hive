"""
Unit tests for Project Index cache management.

Tests for Redis-based caching functionality with TTL management
and cache invalidation strategies for analysis results.
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, Optional

from app.project_index.cache import CacheManager
from app.project_index.models import (
    FileAnalysisResult,
    AnalysisResult,
    DependencyResult,
    ComplexityMetrics
)


class TestCacheManager:
    """Test CacheManager functionality."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing."""
        redis_mock = AsyncMock()
        redis_mock.ping.return_value = True
        redis_mock.get.return_value = None
        redis_mock.setex.return_value = True
        redis_mock.delete.return_value = 1
        redis_mock.exists.return_value = False
        redis_mock.keys.return_value = []
        redis_mock.mget.return_value = []
        redis_mock.mset.return_value = True
        redis_mock.expire.return_value = True
        return redis_mock
    
    @pytest.fixture
    def cache_manager(self, mock_redis):
        """Create CacheManager instance with mocked Redis."""
        return CacheManager(redis_client=mock_redis)
    
    @pytest.mark.asyncio
    async def test_cache_manager_initialization(self, mock_redis):
        """Test cache manager initialization."""
        cache_manager = CacheManager(redis_client=mock_redis)
        
        assert cache_manager.redis == mock_redis
        assert cache_manager.default_ttl == 3600  # 1 hour
        assert cache_manager.long_ttl == 86400    # 24 hours
        assert cache_manager.short_ttl == 900     # 15 minutes
    
    @pytest.mark.asyncio
    async def test_cache_manager_initialization_with_custom_ttl(self, mock_redis):
        """Test cache manager initialization with custom TTL values."""
        cache_manager = CacheManager(
            redis_client=mock_redis,
            default_ttl=7200,
            long_ttl=172800,
            short_ttl=1800
        )
        
        assert cache_manager.default_ttl == 7200
        assert cache_manager.long_ttl == 172800
        assert cache_manager.short_ttl == 1800
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, cache_manager, mock_redis):
        """Test health check when Redis is healthy."""
        mock_redis.ping.return_value = True
        
        is_healthy = await cache_manager.health_check()
        assert is_healthy is True
        mock_redis.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, cache_manager, mock_redis):
        """Test health check when Redis is unhealthy."""
        mock_redis.ping.side_effect = Exception("Redis connection failed")
        
        is_healthy = await cache_manager.health_check()
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_get_analysis_result_cache_hit(self, cache_manager, mock_redis):
        """Test getting analysis result from cache (cache hit)."""
        # Mock cached data
        cached_result = {
            "project_id": "proj_123",
            "session_id": "sess_456",
            "analysis_type": "full",
            "files_processed": 5,
            "analysis_duration": 30.0
        }
        mock_redis.get.return_value = json.dumps(cached_result)
        
        result = await cache_manager.get_analysis_result("proj_123", "sess_456")
        
        assert result is not None
        assert isinstance(result, AnalysisResult)
        assert result.project_id == "proj_123"
        assert result.session_id == "sess_456"
        assert result.files_processed == 5
        
        mock_redis.get.assert_called_once_with("analysis:proj_123:sess_456")
    
    @pytest.mark.asyncio
    async def test_get_analysis_result_cache_miss(self, cache_manager, mock_redis):
        """Test getting analysis result from cache (cache miss)."""
        mock_redis.get.return_value = None
        
        result = await cache_manager.get_analysis_result("proj_123", "sess_456")
        
        assert result is None
        mock_redis.get.assert_called_once_with("analysis:proj_123:sess_456")
    
    @pytest.mark.asyncio
    async def test_get_analysis_result_invalid_json(self, cache_manager, mock_redis):
        """Test getting analysis result with invalid JSON in cache."""
        mock_redis.get.return_value = "invalid json {"
        
        result = await cache_manager.get_analysis_result("proj_123", "sess_456")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_set_analysis_result(self, cache_manager, mock_redis):
        """Test setting analysis result in cache."""
        analysis_result = AnalysisResult(
            project_id="proj_123",
            session_id="sess_456",
            analysis_type="full",
            files_processed=5,
            analysis_duration=30.0
        )
        
        await cache_manager.set_analysis_result(analysis_result)
        
        # Check that setex was called with correct parameters
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        
        assert call_args[0][0] == "analysis:proj_123:sess_456"  # key
        assert call_args[0][1] == cache_manager.default_ttl     # TTL
        
        # Verify the JSON data
        stored_data = json.loads(call_args[0][2])
        assert stored_data["project_id"] == "proj_123"
        assert stored_data["files_processed"] == 5
    
    @pytest.mark.asyncio
    async def test_set_analysis_result_with_custom_ttl(self, cache_manager, mock_redis):
        """Test setting analysis result with custom TTL."""
        analysis_result = AnalysisResult(
            project_id="proj_123",
            session_id="sess_456",
            analysis_type="full"
        )
        
        custom_ttl = 7200
        await cache_manager.set_analysis_result(analysis_result, ttl=custom_ttl)
        
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == custom_ttl
    
    @pytest.mark.asyncio
    async def test_get_file_analysis_cache_hit(self, cache_manager, mock_redis):
        """Test getting file analysis from cache (cache hit)."""
        cached_file_result = {
            "file_path": "/path/to/file.py",
            "file_size": 1024,
            "line_count": 50,
            "language": "python",
            "analysis_successful": True
        }
        mock_redis.get.return_value = json.dumps(cached_file_result)
        
        result = await cache_manager.get_file_analysis("proj_123", "/path/to/file.py")
        
        assert result is not None
        assert isinstance(result, FileAnalysisResult)
        assert result.file_path == "/path/to/file.py"
        assert result.file_size == 1024
        assert result.language == "python"
        
        mock_redis.get.assert_called_once_with("file:proj_123:/path/to/file.py")
    
    @pytest.mark.asyncio
    async def test_get_file_analysis_cache_miss(self, cache_manager, mock_redis):
        """Test getting file analysis from cache (cache miss)."""
        mock_redis.get.return_value = None
        
        result = await cache_manager.get_file_analysis("proj_123", "/path/to/file.py")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_set_file_analysis(self, cache_manager, mock_redis):
        """Test setting file analysis in cache."""
        file_result = FileAnalysisResult(
            file_path="/path/to/file.py",
            file_size=1024,
            line_count=50,
            language="python"
        )
        
        await cache_manager.set_file_analysis("proj_123", file_result)
        
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        
        assert call_args[0][0] == "file:proj_123:/path/to/file.py"
        assert call_args[0][1] == cache_manager.long_ttl
        
        stored_data = json.loads(call_args[0][2])
        assert stored_data["file_path"] == "/path/to/file.py"
        assert stored_data["file_size"] == 1024
    
    @pytest.mark.asyncio
    async def test_get_dependency_graph_cache_hit(self, cache_manager, mock_redis):
        """Test getting dependency graph from cache (cache hit)."""
        cached_graph = {
            "nodes": [{"id": "a", "name": "A", "type": "internal_file"}],
            "edges": [{"source": "a", "target": "b", "dependency_type": "import"}],
            "metrics": {"node_count": 2, "edge_count": 1}
        }
        mock_redis.get.return_value = json.dumps(cached_graph)
        
        result = await cache_manager.get_dependency_graph("proj_123")
        
        assert result is not None
        assert isinstance(result, dict)
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) == 1
        
        mock_redis.get.assert_called_once_with("graph:proj_123")
    
    @pytest.mark.asyncio
    async def test_set_dependency_graph(self, cache_manager, mock_redis):
        """Test setting dependency graph in cache."""
        graph_data = {
            "nodes": [{"id": "a", "name": "A", "type": "internal_file"}],
            "edges": [{"source": "a", "target": "b", "dependency_type": "import"}],
            "metrics": {"node_count": 2, "edge_count": 1}
        }
        
        await cache_manager.set_dependency_graph("proj_123", graph_data)
        
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        
        assert call_args[0][0] == "graph:proj_123"
        assert call_args[0][1] == cache_manager.long_ttl
        
        stored_data = json.loads(call_args[0][2])
        assert stored_data == graph_data
    
    @pytest.mark.asyncio
    async def test_get_context_optimization_cache_hit(self, cache_manager, mock_redis):
        """Test getting context optimization from cache (cache hit)."""
        cached_context = {
            "context_type": "development",
            "total_files": 100,
            "recommended_files": [],
            "clusters": [],
            "optimization_metrics": {"average_relevance": 0.75}
        }
        mock_redis.get.return_value = json.dumps(cached_context)
        
        result = await cache_manager.get_context_optimization("proj_123", "development")
        
        assert result is not None
        assert isinstance(result, dict)
        assert result["context_type"] == "development"
        assert result["total_files"] == 100
        
        mock_redis.get.assert_called_once_with("context:proj_123:development")
    
    @pytest.mark.asyncio
    async def test_set_context_optimization(self, cache_manager, mock_redis):
        """Test setting context optimization in cache."""
        context_data = {
            "context_type": "development",
            "total_files": 100,
            "recommended_files": [],
            "clusters": []
        }
        
        await cache_manager.set_context_optimization("proj_123", "development", context_data)
        
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        
        assert call_args[0][0] == "context:proj_123:development"
        assert call_args[0][1] == cache_manager.default_ttl
    
    @pytest.mark.asyncio
    async def test_invalidate_project_cache(self, cache_manager, mock_redis):
        """Test invalidating all cache entries for a project."""
        # Mock keys that exist for the project
        mock_redis.keys.return_value = [
            b"analysis:proj_123:sess_1",
            b"analysis:proj_123:sess_2",
            b"file:proj_123:/path/to/file1.py",
            b"file:proj_123:/path/to/file2.py",
            b"graph:proj_123",
            b"context:proj_123:development"
        ]
        mock_redis.delete.return_value = 6
        
        deleted_count = await cache_manager.invalidate_project("proj_123")
        
        assert deleted_count == 6
        mock_redis.keys.assert_called_once_with("*proj_123*")
        mock_redis.delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_invalidate_project_cache_no_entries(self, cache_manager, mock_redis):
        """Test invalidating cache when no entries exist."""
        mock_redis.keys.return_value = []
        
        deleted_count = await cache_manager.invalidate_project("proj_123")
        
        assert deleted_count == 0
        mock_redis.keys.assert_called_once_with("*proj_123*")
        mock_redis.delete.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_invalidate_file_cache(self, cache_manager, mock_redis):
        """Test invalidating cache for specific file."""
        mock_redis.delete.return_value = 1
        
        deleted = await cache_manager.invalidate_file("proj_123", "/path/to/file.py")
        
        assert deleted is True
        mock_redis.delete.assert_called_once_with("file:proj_123:/path/to/file.py")
    
    @pytest.mark.asyncio
    async def test_invalidate_file_cache_not_found(self, cache_manager, mock_redis):
        """Test invalidating cache for file that doesn't exist in cache."""
        mock_redis.delete.return_value = 0
        
        deleted = await cache_manager.invalidate_file("proj_123", "/path/to/file.py")
        
        assert deleted is False
    
    @pytest.mark.asyncio
    async def test_get_cache_stats(self, cache_manager, mock_redis):
        """Test getting cache statistics."""
        # Mock Redis info response
        mock_redis.info.return_value = {
            'used_memory': 1024000,
            'used_memory_human': '1.00M',
            'keyspace_hits': 150,
            'keyspace_misses': 50,
            'total_commands_processed': 1000
        }
        
        # Mock keys for project cache entries
        mock_redis.keys.side_effect = [
            [b"analysis:proj_123:sess_1", b"analysis:proj_123:sess_2"],  # analysis keys
            [b"file:proj_123:/file1.py", b"file:proj_123:/file2.py"],    # file keys
            [b"graph:proj_123"],                                         # graph keys
            [b"context:proj_123:dev"]                                    # context keys
        ]
        
        stats = await cache_manager.get_cache_stats("proj_123")
        
        assert stats['memory_usage'] == 1024000
        assert stats['memory_usage_human'] == '1.00M'
        assert stats['hit_rate'] == 0.75  # 150 / (150 + 50)
        assert stats['total_commands'] == 1000
        assert stats['analysis_cache_entries'] == 2
        assert stats['file_cache_entries'] == 2
        assert stats['graph_cache_entries'] == 1
        assert stats['context_cache_entries'] == 1
    
    @pytest.mark.asyncio
    async def test_batch_get_file_analyses(self, cache_manager, mock_redis):
        """Test batch getting multiple file analyses."""
        file_paths = ["/path/to/file1.py", "/path/to/file2.py", "/path/to/file3.py"]
        
        # Mock mget response - file1 and file3 in cache, file2 not
        cached_results = [
            json.dumps({"file_path": "/path/to/file1.py", "language": "python"}),
            None,  # file2 not in cache
            json.dumps({"file_path": "/path/to/file3.py", "language": "python"})
        ]
        mock_redis.mget.return_value = cached_results
        
        results = await cache_manager.batch_get_file_analyses("proj_123", file_paths)
        
        assert len(results) == 3
        assert results[0] is not None
        assert results[0].file_path == "/path/to/file1.py"
        assert results[1] is None  # Cache miss
        assert results[2] is not None
        assert results[2].file_path == "/path/to/file3.py"
        
        # Check mget was called with correct keys
        expected_keys = [f"file:proj_123:{path}" for path in file_paths]
        mock_redis.mget.assert_called_once_with(expected_keys)
    
    @pytest.mark.asyncio
    async def test_batch_set_file_analyses(self, cache_manager, mock_redis):
        """Test batch setting multiple file analyses."""
        file_results = [
            FileAnalysisResult(file_path="/path/to/file1.py", language="python"),
            FileAnalysisResult(file_path="/path/to/file2.py", language="javascript")
        ]
        
        await cache_manager.batch_set_file_analyses("proj_123", file_results)
        
        # Check mset was called
        mock_redis.mset.assert_called_once()
        call_args = mock_redis.mset.call_args[0][0]
        
        # Verify the keys and data
        assert "file:proj_123:/path/to/file1.py" in call_args
        assert "file:proj_123:/path/to/file2.py" in call_args
        
        # Verify expire was called for each key
        assert mock_redis.expire.call_count == 2
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_entries(self, cache_manager, mock_redis):
        """Test cleaning up expired cache entries."""
        # Mock scan_iter to return some keys
        mock_redis.scan_iter.return_value = [
            b"analysis:proj_123:old_session",
            b"file:proj_123:/old_file.py",
            b"graph:proj_123"
        ]
        
        # Mock ttl to return different values
        mock_redis.ttl.side_effect = [-1, 3600, -2]  # -1: no expiry, 3600: valid, -2: expired
        mock_redis.delete.return_value = 1
        
        deleted_count = await cache_manager.cleanup_expired_entries("proj_123")
        
        # Should delete the expired key (ttl = -2)
        assert deleted_count == 1
        mock_redis.delete.assert_called_once_with(b"file:proj_123:/old_file.py")
    
    @pytest.mark.asyncio
    async def test_extend_ttl(self, cache_manager, mock_redis):
        """Test extending TTL for cache entries."""
        mock_redis.expire.return_value = True
        
        extended = await cache_manager.extend_ttl("analysis:proj_123:sess_456", 7200)
        
        assert extended is True
        mock_redis.expire.assert_called_once_with("analysis:proj_123:sess_456", 7200)
    
    @pytest.mark.asyncio
    async def test_extend_ttl_key_not_found(self, cache_manager, mock_redis):
        """Test extending TTL for non-existent key."""
        mock_redis.expire.return_value = False
        
        extended = await cache_manager.extend_ttl("nonexistent:key", 7200)
        
        assert extended is False
    
    @pytest.mark.asyncio
    async def test_cache_serialization_with_complex_objects(self, cache_manager, mock_redis):
        """Test caching complex objects with nested data."""
        # Create complex analysis result with nested objects
        complexity_metrics = ComplexityMetrics(
            cyclomatic_complexity=5,
            cognitive_complexity=8,
            nesting_depth=3
        )
        
        dependency = DependencyResult(
            source_file_path="/path/to/source.py",
            target_name="requests",
            dependency_type="import",
            confidence_score=0.9
        )
        
        file_result = FileAnalysisResult(
            file_path="/path/to/file.py",
            language="python",
            complexity_metrics=complexity_metrics,
            dependencies=[dependency]
        )
        
        analysis_result = AnalysisResult(
            project_id="proj_123",
            session_id="sess_456",
            analysis_type="full",
            file_results=[file_result],
            dependency_results=[dependency]
        )
        
        # Set and get the complex object
        await cache_manager.set_analysis_result(analysis_result)
        
        # Mock the return value for get
        stored_call_args = mock_redis.setex.call_args[0][2]
        mock_redis.get.return_value = stored_call_args
        
        retrieved_result = await cache_manager.get_analysis_result("proj_123", "sess_456")
        
        assert retrieved_result is not None
        assert retrieved_result.project_id == "proj_123"
        assert len(retrieved_result.file_results) == 1
        assert len(retrieved_result.dependency_results) == 1
    
    @pytest.mark.asyncio
    async def test_cache_error_handling(self, cache_manager, mock_redis):
        """Test cache error handling when Redis operations fail."""
        # Mock Redis operations to raise exceptions
        mock_redis.get.side_effect = Exception("Redis connection error")
        mock_redis.setex.side_effect = Exception("Redis write error")
        
        # Get operation should return None on error
        result = await cache_manager.get_analysis_result("proj_123", "sess_456")
        assert result is None
        
        # Set operation should not raise exception
        analysis_result = AnalysisResult(
            project_id="proj_123",
            session_id="sess_456",
            analysis_type="full"
        )
        
        try:
            await cache_manager.set_analysis_result(analysis_result)
            # Should not raise exception
        except Exception:
            pytest.fail("Cache set operation should handle Redis errors gracefully")
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, cache_manager):
        """Test cache key generation for different data types."""
        # Test different key patterns
        analysis_key = cache_manager._get_analysis_key("proj_123", "sess_456")
        assert analysis_key == "analysis:proj_123:sess_456"
        
        file_key = cache_manager._get_file_key("proj_123", "/path/to/file.py")
        assert file_key == "file:proj_123:/path/to/file.py"
        
        graph_key = cache_manager._get_graph_key("proj_123")
        assert graph_key == "graph:proj_123"
        
        context_key = cache_manager._get_context_key("proj_123", "development")
        assert context_key == "context:proj_123:development"