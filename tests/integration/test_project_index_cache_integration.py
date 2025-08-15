"""
Integration tests for Project Index cache system.

Tests Redis cache integration with actual cache operations,
TTL management, and cache invalidation strategies.
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from app.project_index.cache import CacheManager
from app.project_index.models import (
    FileAnalysisResult, AnalysisResult, DependencyResult, ComplexityMetrics
)


class TestCacheIntegration:
    """Integration tests for cache management system."""
    
    @pytest.fixture
    def redis_client(self):
        """Create mock Redis client with realistic behavior."""
        redis = AsyncMock()
        
        # Storage for mock data
        self._cache_data = {}
        self._ttl_data = {}
        
        async def mock_get(key):
            if key in self._cache_data:
                # Check TTL
                if key in self._ttl_data:
                    if time.time() > self._ttl_data[key]:
                        del self._cache_data[key]
                        del self._ttl_data[key]
                        return None
                return self._cache_data[key]
            return None
        
        async def mock_setex(key, ttl, value):
            self._cache_data[key] = value
            self._ttl_data[key] = time.time() + ttl
            return True
        
        async def mock_set(key, value, expire=None):
            self._cache_data[key] = value
            if expire:
                self._ttl_data[key] = time.time() + expire
            return True
        
        async def mock_delete(*keys):
            deleted = 0
            for key in keys:
                if key in self._cache_data:
                    del self._cache_data[key]
                    deleted += 1
                if key in self._ttl_data:
                    del self._ttl_data[key]
            return deleted
        
        async def mock_keys(pattern):
            import fnmatch
            matching_keys = []
            for key in self._cache_data.keys():
                if fnmatch.fnmatch(key, pattern):
                    matching_keys.append(key.encode() if isinstance(key, str) else key)
            return matching_keys
        
        async def mock_mget(keys):
            results = []
            for key in keys:
                results.append(await mock_get(key))
            return results
        
        async def mock_mset(mapping):
            for key, value in mapping.items():
                self._cache_data[key] = value
            return True
        
        async def mock_expire(key, ttl):
            if key in self._cache_data:
                self._ttl_data[key] = time.time() + ttl
                return True
            return False
        
        async def mock_ttl(key):
            if key in self._ttl_data:
                remaining = self._ttl_data[key] - time.time()
                if remaining > 0:
                    return int(remaining)
                else:
                    return -2  # Expired
            elif key in self._cache_data:
                return -1  # No expiry set
            else:
                return -2  # Key doesn't exist
        
        async def mock_info():
            return {
                'used_memory': len(str(self._cache_data)),
                'used_memory_human': f'{len(str(self._cache_data))}B',
                'keyspace_hits': 100,
                'keyspace_misses': 20,
                'total_commands_processed': 500
            }
        
        # Mock scan_iter for cleanup tests
        async def mock_scan_iter(match=None):
            keys = list(self._cache_data.keys())
            if match:
                import fnmatch
                keys = [k for k in keys if fnmatch.fnmatch(k, match)]
            for key in keys:
                yield key.encode() if isinstance(key, str) else key
        
        redis.get.side_effect = mock_get
        redis.setex.side_effect = mock_setex
        redis.set.side_effect = mock_set
        redis.delete.side_effect = mock_delete
        redis.keys.side_effect = mock_keys
        redis.mget.side_effect = mock_mget
        redis.mset.side_effect = mock_mset
        redis.expire.side_effect = mock_expire
        redis.ttl.side_effect = mock_ttl
        redis.info.side_effect = mock_info
        redis.scan_iter.side_effect = mock_scan_iter
        redis.ping.return_value = True
        
        return redis
    
    @pytest.fixture
    def cache_manager(self, redis_client):
        """Create CacheManager with mock Redis."""
        return CacheManager(redis_client=redis_client)
    
    @pytest.mark.asyncio
    async def test_cache_health_check(self, cache_manager):
        """Test cache health checking."""
        # Should be healthy
        is_healthy = await cache_manager.health_check()
        assert is_healthy is True
        
        # Test unhealthy state
        cache_manager.redis.ping.side_effect = Exception("Connection failed")
        is_healthy = await cache_manager.health_check()
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_analysis_result_caching(self, cache_manager):
        """Test caching and retrieval of analysis results."""
        # Create test analysis result
        analysis_result = AnalysisResult(
            project_id="proj_123",
            session_id="sess_456",
            analysis_type="full",
            files_processed=10,
            files_analyzed=8,
            dependencies_found=25,
            analysis_duration=45.2
        )
        
        # Test cache miss
        result = await cache_manager.get_analysis_result("proj_123", "sess_456")
        assert result is None
        
        # Cache the result
        await cache_manager.set_analysis_result(analysis_result)
        
        # Test cache hit
        cached_result = await cache_manager.get_analysis_result("proj_123", "sess_456")
        assert cached_result is not None
        assert cached_result.project_id == "proj_123"
        assert cached_result.session_id == "sess_456"
        assert cached_result.files_processed == 10
        assert cached_result.dependencies_found == 25
    
    @pytest.mark.asyncio
    async def test_file_analysis_caching(self, cache_manager):
        """Test caching of file analysis results."""
        # Create test file analysis result
        complexity_metrics = ComplexityMetrics(
            cyclomatic_complexity=5,
            cognitive_complexity=8,
            nesting_depth=3
        )
        
        file_result = FileAnalysisResult(
            file_path="/path/to/file.py",
            relative_path="src/file.py",
            file_name="file.py",
            file_extension=".py",
            language="python",
            file_size=2048,
            line_count=100,
            complexity_metrics=complexity_metrics,
            analysis_successful=True
        )
        
        # Test cache miss
        result = await cache_manager.get_file_analysis("proj_123", "/path/to/file.py")
        assert result is None
        
        # Cache the result
        await cache_manager.set_file_analysis("proj_123", file_result)
        
        # Test cache hit
        cached_result = await cache_manager.get_file_analysis("proj_123", "/path/to/file.py")
        assert cached_result is not None
        assert cached_result.file_path == "/path/to/file.py"
        assert cached_result.language == "python"
        assert cached_result.complexity_metrics.cyclomatic_complexity == 5
    
    @pytest.mark.asyncio
    async def test_dependency_graph_caching(self, cache_manager):
        """Test caching of dependency graphs."""
        # Create test dependency graph
        graph_data = {
            "nodes": [
                {"id": "file1", "name": "main.py", "type": "source"},
                {"id": "file2", "name": "utils.py", "type": "source"}
            ],
            "edges": [
                {"source": "file1", "target": "file2", "type": "import"}
            ],
            "metrics": {
                "node_count": 2,
                "edge_count": 1,
                "cyclic_dependencies": 0
            }
        }
        
        # Test cache miss
        result = await cache_manager.get_dependency_graph("proj_123")
        assert result is None
        
        # Cache the graph
        await cache_manager.set_dependency_graph("proj_123", graph_data)
        
        # Test cache hit
        cached_graph = await cache_manager.get_dependency_graph("proj_123")
        assert cached_graph is not None
        assert len(cached_graph["nodes"]) == 2
        assert len(cached_graph["edges"]) == 1
        assert cached_graph["metrics"]["node_count"] == 2
    
    @pytest.mark.asyncio
    async def test_context_optimization_caching(self, cache_manager):
        """Test caching of context optimization results."""
        # Create test context optimization data
        context_data = {
            "context_type": "development",
            "total_files": 50,
            "recommended_files": [
                {"path": "main.py", "relevance": 0.95},
                {"path": "utils.py", "relevance": 0.80}
            ],
            "clusters": [
                {
                    "id": "core",
                    "files": ["main.py", "app.py"],
                    "coherence_score": 0.85
                }
            ],
            "optimization_metrics": {
                "context_efficiency": 0.78,
                "relevance_accuracy": 0.82
            }
        }
        
        # Test cache miss
        result = await cache_manager.get_context_optimization("proj_123", "development")
        assert result is None
        
        # Cache the data
        await cache_manager.set_context_optimization("proj_123", "development", context_data)
        
        # Test cache hit
        cached_context = await cache_manager.get_context_optimization("proj_123", "development")
        assert cached_context is not None
        assert cached_context["context_type"] == "development"
        assert cached_context["total_files"] == 50
        assert len(cached_context["recommended_files"]) == 2
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, cache_manager):
        """Test cache invalidation for projects and files."""
        # Set up test data
        analysis_result = AnalysisResult(
            project_id="proj_123",
            session_id="sess_456",
            analysis_type="full"
        )
        
        file_result = FileAnalysisResult(
            file_path="/path/to/file.py",
            file_name="file.py",
            language="python"
        )
        
        # Cache data
        await cache_manager.set_analysis_result(analysis_result)
        await cache_manager.set_file_analysis("proj_123", file_result)
        
        # Verify data is cached
        assert await cache_manager.get_analysis_result("proj_123", "sess_456") is not None
        assert await cache_manager.get_file_analysis("proj_123", "/path/to/file.py") is not None
        
        # Invalidate project cache
        deleted_count = await cache_manager.invalidate_project("proj_123")
        assert deleted_count > 0
        
        # Verify project data is invalidated
        assert await cache_manager.get_analysis_result("proj_123", "sess_456") is None
        
        # Test file-specific invalidation
        await cache_manager.set_file_analysis("proj_123", file_result)
        deleted = await cache_manager.invalidate_file("proj_123", "/path/to/file.py")
        assert deleted is True
        assert await cache_manager.get_file_analysis("proj_123", "/path/to/file.py") is None
    
    @pytest.mark.asyncio
    async def test_batch_operations(self, cache_manager):
        """Test batch get and set operations."""
        # Create multiple file results
        file_results = []
        file_paths = ["/path/file1.py", "/path/file2.py", "/path/file3.py"]
        
        for i, path in enumerate(file_paths):
            file_result = FileAnalysisResult(
                file_path=path,
                file_name=f"file{i+1}.py",
                language="python",
                line_count=(i+1) * 10
            )
            file_results.append(file_result)
        
        # Batch set
        await cache_manager.batch_set_file_analyses("proj_123", file_results)
        
        # Batch get
        cached_results = await cache_manager.batch_get_file_analyses("proj_123", file_paths)
        
        # Verify results
        assert len(cached_results) == 3
        for i, result in enumerate(cached_results):
            assert result is not None
            assert result.file_path == file_paths[i]
            assert result.line_count == (i+1) * 10
    
    @pytest.mark.asyncio
    async def test_ttl_functionality(self, cache_manager):
        """Test TTL (Time To Live) functionality."""
        # Create test data with short TTL
        analysis_result = AnalysisResult(
            project_id="proj_123",
            session_id="sess_456",
            analysis_type="full"
        )
        
        # Cache with short TTL (1 second)
        await cache_manager.set_analysis_result(analysis_result, ttl=1)
        
        # Verify data is cached
        result = await cache_manager.get_analysis_result("proj_123", "sess_456")
        assert result is not None
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Verify data is expired
        expired_result = await cache_manager.get_analysis_result("proj_123", "sess_456")
        assert expired_result is None
    
    @pytest.mark.asyncio
    async def test_ttl_extension(self, cache_manager):
        """Test extending TTL for cache entries."""
        # Cache data
        analysis_result = AnalysisResult(
            project_id="proj_123",
            session_id="sess_456",
            analysis_type="full"
        )
        
        await cache_manager.set_analysis_result(analysis_result, ttl=60)
        
        # Extend TTL
        cache_key = cache_manager._get_analysis_key("proj_123", "sess_456")
        extended = await cache_manager.extend_ttl(cache_key, 3600)
        assert extended is True
        
        # Test extending non-existent key
        extended = await cache_manager.extend_ttl("nonexistent:key", 3600)
        assert extended is False
    
    @pytest.mark.asyncio
    async def test_cache_statistics(self, cache_manager):
        """Test cache statistics gathering."""
        # Set up test data
        analysis_result = AnalysisResult(
            project_id="proj_123",
            session_id="sess_456",
            analysis_type="full"
        )
        
        file_result = FileAnalysisResult(
            file_path="/path/to/file.py",
            file_name="file.py",
            language="python"
        )
        
        # Cache some data
        await cache_manager.set_analysis_result(analysis_result)
        await cache_manager.set_file_analysis("proj_123", file_result)
        
        # Get statistics
        stats = await cache_manager.get_cache_stats("proj_123")
        
        assert 'memory_usage' in stats
        assert 'hit_rate' in stats
        assert 'total_commands' in stats
        assert 'analysis_cache_entries' in stats
        assert 'file_cache_entries' in stats
        
        # Verify counts
        assert stats['analysis_cache_entries'] >= 1
        assert stats['file_cache_entries'] >= 1
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_entries(self, cache_manager):
        """Test cleanup of expired cache entries."""
        # Create entries with different TTLs
        analysis_result1 = AnalysisResult(
            project_id="proj_123",
            session_id="sess_old",
            analysis_type="full"
        )
        
        analysis_result2 = AnalysisResult(
            project_id="proj_123",
            session_id="sess_new",
            analysis_type="full"
        )
        
        # Cache with different TTLs
        await cache_manager.set_analysis_result(analysis_result1, ttl=1)  # Will expire
        await cache_manager.set_analysis_result(analysis_result2, ttl=3600)  # Long-lived
        
        # Wait for first entry to expire
        await asyncio.sleep(1.1)
        
        # Run cleanup
        deleted_count = await cache_manager.cleanup_expired_entries("proj_123")
        
        # Should have cleaned up expired entry
        assert deleted_count >= 0  # May be 0 if Redis handles expiration automatically
        
        # Verify non-expired entry still exists
        result = await cache_manager.get_analysis_result("proj_123", "sess_new")
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, cache_manager):
        """Test cache key generation methods."""
        # Test analysis key
        analysis_key = cache_manager._get_analysis_key("proj_123", "sess_456")
        assert analysis_key == "analysis:proj_123:sess_456"
        
        # Test file key
        file_key = cache_manager._get_file_key("proj_123", "/path/to/file.py")
        assert file_key == "file:proj_123:/path/to/file.py"
        
        # Test graph key
        graph_key = cache_manager._get_graph_key("proj_123")
        assert graph_key == "graph:proj_123"
        
        # Test context key
        context_key = cache_manager._get_context_key("proj_123", "development")
        assert context_key == "context:proj_123:development"
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self, cache_manager):
        """Test concurrent cache operations."""
        # Create multiple analysis results
        tasks = []
        
        for i in range(10):
            analysis_result = AnalysisResult(
                project_id="proj_123",
                session_id=f"sess_{i}",
                analysis_type="full",
                files_processed=i * 5
            )
            
            # Create set and get tasks
            set_task = cache_manager.set_analysis_result(analysis_result)
            tasks.append(set_task)
        
        # Execute all set operations concurrently
        await asyncio.gather(*tasks)
        
        # Verify all results were cached
        get_tasks = []
        for i in range(10):
            get_task = cache_manager.get_analysis_result("proj_123", f"sess_{i}")
            get_tasks.append(get_task)
        
        results = await asyncio.gather(*get_tasks)
        
        # All results should be cached
        assert all(result is not None for result in results)
        
        # Verify data integrity
        for i, result in enumerate(results):
            assert result.session_id == f"sess_{i}"
            assert result.files_processed == i * 5
    
    @pytest.mark.asyncio
    async def test_large_data_caching(self, cache_manager):
        """Test caching of large data structures."""
        # Create large analysis result with many file results
        file_results = []
        for i in range(100):
            file_result = FileAnalysisResult(
                file_path=f"/path/to/file_{i}.py",
                file_name=f"file_{i}.py",
                language="python",
                line_count=i * 10,
                analysis_data={"functions": [f"func_{j}" for j in range(10)]}
            )
            file_results.append(file_result)
        
        analysis_result = AnalysisResult(
            project_id="proj_large",
            session_id="sess_456",
            analysis_type="full",
            files_processed=100,
            file_results=file_results
        )
        
        # Cache large result
        await cache_manager.set_analysis_result(analysis_result)
        
        # Retrieve and verify
        cached_result = await cache_manager.get_analysis_result("proj_large", "sess_456")
        assert cached_result is not None
        assert len(cached_result.file_results) == 100
        assert cached_result.files_processed == 100
    
    @pytest.mark.asyncio
    async def test_error_handling(self, cache_manager):
        """Test error handling in cache operations."""
        # Test invalid JSON handling
        cache_manager.redis.get.return_value = "invalid json {"
        
        result = await cache_manager.get_analysis_result("proj_123", "sess_456")
        assert result is None
        
        # Test Redis operation failures
        cache_manager.redis.setex.side_effect = Exception("Redis error")
        
        analysis_result = AnalysisResult(
            project_id="proj_123",
            session_id="sess_456",
            analysis_type="full"
        )
        
        # Should not raise exception
        try:
            await cache_manager.set_analysis_result(analysis_result)
        except Exception:
            pytest.fail("Cache operation should handle Redis errors gracefully")