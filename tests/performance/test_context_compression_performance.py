"""
Performance and load tests for Context Compression
Tests performance targets, scalability, and resource usage
"""

import pytest
import asyncio
import time
import psutil
import json
from unittest.mock import Mock, AsyncMock, patch
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, median
from typing import List, Dict, Any

from app.core.context_compression import (
    ContextCompressor,
    CompressionLevel
)
from app.core.hive_slash_commands import HiveCompactCommand


class TestContextCompressionPerformance:
    """Performance test suite for context compression"""
    
    @pytest.fixture
    def compressor(self):
        """Create ContextCompressor with mocked client for performance testing"""
        mock_client = AsyncMock()
        
        # Mock consistent response for performance testing
        def create_mock_response():
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = json.dumps({
                "summary": "Performance test compression summary with sufficient content to measure token reduction",
                "key_insights": ["Performance insight 1", "Performance insight 2", "Performance insight 3"],
                "decisions_made": ["Performance decision 1", "Performance decision 2"],
                "patterns_identified": ["Performance pattern 1", "Performance pattern 2"],
                "importance_score": 0.75
            })
            return mock_response
        
        mock_client.messages.create.side_effect = lambda *args, **kwargs: create_mock_response()
        return ContextCompressor(llm_client=mock_client)
    
    @pytest.fixture
    def compact_command(self):
        """Create HiveCompactCommand for performance testing"""
        return HiveCompactCommand()
    
    def generate_test_content(self, size_category: str) -> str:
        """Generate test content of different sizes"""
        base_content = """
        This is a comprehensive development session discussing the implementation
        of a new feature in our application. The team has gathered to review
        requirements, discuss architecture decisions, and plan the development
        approach. We need to consider performance implications, security concerns,
        and user experience improvements.
        
        Key discussion points include:
        1. Database schema modifications required
        2. API endpoint design and versioning
        3. Frontend component architecture
        4. Testing strategy and coverage requirements
        5. Deployment pipeline modifications
        6. Monitoring and observability improvements
        
        The team has made several important decisions during this session
        that will impact the overall system design and implementation approach.
        """
        
        if size_category == "small":
            return base_content
        elif size_category == "medium":
            return base_content * 10  # ~5K characters
        elif size_category == "large":
            return base_content * 50  # ~25K characters
        elif size_category == "extra_large":
            return base_content * 200  # ~100K characters
        else:
            return base_content
    
    @pytest.mark.asyncio
    async def test_compression_speed_target(self, compressor):
        """Test that compression meets <15 second target"""
        content = self.generate_test_content("medium")
        
        start_time = time.time()
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD
        )
        end_time = time.time()
        
        compression_time = end_time - start_time
        
        # Target: <15 seconds
        assert compression_time < 15.0, f"Compression took {compression_time:.2f}s, exceeding 15s target"
        assert result.summary
        
        # Log performance for monitoring
        print(f"Compression time: {compression_time:.2f}s for {len(content)} characters")
    
    @pytest.mark.asyncio
    async def test_compression_speed_by_content_size(self, compressor):
        """Test compression speed across different content sizes"""
        sizes = ["small", "medium", "large"]
        results = {}
        
        for size in sizes:
            content = self.generate_test_content(size)
            
            # Measure compression time
            start_time = time.time()
            result = await compressor.compress_conversation(
                conversation_content=content,
                compression_level=CompressionLevel.STANDARD
            )
            end_time = time.time()
            
            compression_time = end_time - start_time
            results[size] = {
                "time": compression_time,
                "content_length": len(content),
                "original_tokens": result.original_token_count,
                "compressed_tokens": result.compressed_token_count,
                "compression_ratio": result.compression_ratio
            }
            
            # All sizes should meet the 15s target
            assert compression_time < 15.0, f"{size} content compression took {compression_time:.2f}s"
        
        # Log performance scaling
        for size, metrics in results.items():
            print(f"{size}: {metrics['time']:.2f}s, {metrics['content_length']} chars, "
                  f"{metrics['original_tokens']} → {metrics['compressed_tokens']} tokens "
                  f"({metrics['compression_ratio']:.1%} reduction)")
        
        # Verify performance scales reasonably with content size
        small_time = results["small"]["time"]
        large_time = results["large"]["time"]
        
        # Large content shouldn't take more than 10x longer than small content
        assert large_time < small_time * 10, "Performance doesn't scale well with content size"
    
    @pytest.mark.asyncio
    async def test_compression_speed_by_level(self, compressor):
        """Test compression speed across different compression levels"""
        content = self.generate_test_content("medium")
        levels = [CompressionLevel.LIGHT, CompressionLevel.STANDARD, CompressionLevel.AGGRESSIVE]
        results = {}
        
        for level in levels:
            start_time = time.time()
            result = await compressor.compress_conversation(
                conversation_content=content,
                compression_level=level
            )
            end_time = time.time()
            
            compression_time = end_time - start_time
            results[level.value] = {
                "time": compression_time,
                "compression_ratio": result.compression_ratio
            }
            
            # All levels should meet the 15s target
            assert compression_time < 15.0, f"{level.value} compression took {compression_time:.2f}s"
        
        # Log performance by level
        for level, metrics in results.items():
            print(f"{level}: {metrics['time']:.2f}s, {metrics['compression_ratio']:.1%} reduction")
    
    @pytest.mark.asyncio
    async def test_concurrent_compression_performance(self, compressor):
        """Test performance under concurrent compression load"""
        content = self.generate_test_content("medium")
        num_concurrent = 10
        
        async def single_compression():
            start_time = time.time()
            result = await compressor.compress_conversation(
                conversation_content=content,
                compression_level=CompressionLevel.STANDARD
            )
            end_time = time.time()
            return end_time - start_time, result
        
        # Run concurrent compressions
        start_time = time.time()
        tasks = [single_compression() for _ in range(num_concurrent)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Verify no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception), f"Concurrent compression failed: {result}"
        
        # Extract timing data
        times = [result[0] for result in results]
        compression_results = [result[1] for result in results]
        
        # Performance assertions
        avg_time = mean(times)
        max_time = max(times)
        
        # Each compression should still meet target
        assert max_time < 15.0, f"Slowest concurrent compression took {max_time:.2f}s"
        
        # Average should be reasonable
        assert avg_time < 10.0, f"Average concurrent compression time {avg_time:.2f}s too slow"
        
        # Total time should show parallelization benefits
        assert total_time < num_concurrent * avg_time * 0.7, "Poor parallelization performance"
        
        # Verify all compressions succeeded
        for comp_result in compression_results:
            assert comp_result.summary
            assert comp_result.compression_ratio > 0
        
        print(f"Concurrent performance: {num_concurrent} compressions in {total_time:.2f}s "
              f"(avg: {avg_time:.2f}s, max: {max_time:.2f}s)")
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_compression(self, compressor):
        """Test memory usage during compression operations"""
        process = psutil.Process()
        content = self.generate_test_content("large")
        
        # Baseline memory usage
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform compression while monitoring memory
        memory_samples = []
        
        async def monitor_memory():
            while True:
                memory_samples.append(process.memory_info().rss / 1024 / 1024)
                await asyncio.sleep(0.1)
        
        # Start monitoring
        monitor_task = asyncio.create_task(monitor_memory())
        
        try:
            # Perform compression
            result = await compressor.compress_conversation(
                conversation_content=content,
                compression_level=CompressionLevel.STANDARD
            )
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        # Analyze memory usage
        if memory_samples:
            max_memory = max(memory_samples)
            avg_memory = mean(memory_samples)
            memory_increase = max_memory - baseline_memory
            
            # Memory increase should be reasonable (< 100MB for single compression)
            assert memory_increase < 100, f"Memory usage increased by {memory_increase:.1f}MB"
            
            print(f"Memory usage: baseline {baseline_memory:.1f}MB, "
                  f"max {max_memory:.1f}MB, increase {memory_increase:.1f}MB")
        
        assert result.summary
    
    @pytest.mark.asyncio
    async def test_compression_throughput(self, compressor):
        """Test compression throughput (compressions per minute)"""
        content = self.generate_test_content("small")
        num_compressions = 20
        
        start_time = time.time()
        
        # Perform compressions sequentially
        for i in range(num_compressions):
            result = await compressor.compress_conversation(
                conversation_content=f"{content} - Iteration {i}",
                compression_level=CompressionLevel.LIGHT  # Use light for speed
            )
            assert result.summary
        
        total_time = time.time() - start_time
        throughput = num_compressions / total_time * 60  # compressions per minute
        
        # Target: at least 30 compressions per minute for small content
        assert throughput >= 30, f"Throughput {throughput:.1f} compressions/min below target of 30"
        
        print(f"Throughput: {throughput:.1f} compressions per minute")
    
    @pytest.mark.asyncio
    async def test_adaptive_compression_performance(self, compressor):
        """Test performance of adaptive compression with various targets"""
        content = self.generate_test_content("large")
        original_tokens = compressor.count_tokens(content)
        
        # Test different target token ratios
        target_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
        results = {}
        
        for ratio in target_ratios:
            target_tokens = int(original_tokens * ratio)
            
            start_time = time.time()
            result = await compressor.adaptive_compress(
                content=content,
                target_token_count=target_tokens
            )
            end_time = time.time()
            
            compression_time = end_time - start_time
            results[ratio] = {
                "time": compression_time,
                "target_tokens": target_tokens,
                "actual_tokens": result.compressed_token_count,
                "compression_ratio": result.compression_ratio
            }
            
            # Should meet time target
            assert compression_time < 15.0, f"Adaptive compression for {ratio} ratio took {compression_time:.2f}s"
            
            # Should be reasonably close to target (within 20%)
            if target_tokens > 100:  # Skip for very small targets
                token_diff = abs(result.compressed_token_count - target_tokens)
                tolerance = target_tokens * 0.2
                assert token_diff <= tolerance, f"Compression missed target by {token_diff} tokens"
        
        # Log adaptive compression performance
        for ratio, metrics in results.items():
            print(f"Target {ratio:.1%}: {metrics['time']:.2f}s, "
                  f"{metrics['target_tokens']} → {metrics['actual_tokens']} tokens")
    
    @pytest.mark.asyncio
    async def test_batch_compression_performance(self, compressor):
        """Test performance of batch compression operations"""
        # Create mock contexts for batch compression
        num_contexts = 20
        contexts = []
        
        for i in range(num_contexts):
            mock_context = Mock()
            mock_context.id = f"context-{i}"
            mock_context.content = self.generate_test_content("small")
            mock_context.context_type = Mock()
            mock_context.context_type.value = "general"
            contexts.append(mock_context)
        
        start_time = time.time()
        results = await compressor.compress_context_batch(
            contexts=contexts,
            compression_level=CompressionLevel.STANDARD
        )
        total_time = time.time() - start_time
        
        # Verify all compressions completed
        assert len(results) == num_contexts
        for result in results:
            assert result.summary
        
        # Performance assertions
        avg_time_per_compression = total_time / num_contexts
        
        # Batch should be more efficient than individual compressions
        assert avg_time_per_compression < 5.0, f"Batch compression too slow: {avg_time_per_compression:.2f}s per item"
        
        # Total time should be reasonable
        assert total_time < 60.0, f"Batch compression took {total_time:.2f}s for {num_contexts} items"
        
        print(f"Batch compression: {num_contexts} items in {total_time:.2f}s "
              f"({avg_time_per_compression:.2f}s per item)")
    
    @pytest.mark.asyncio
    async def test_hive_command_performance_end_to_end(self, compact_command):
        """Test end-to-end performance of HiveCompactCommand"""
        # Mock session and compression components
        mock_session = Mock()
        mock_session.description = self.generate_test_content("medium")
        mock_session.objectives = ["Test objective 1", "Test objective 2"]
        mock_session.shared_context = {"project": "test", "phase": "development"}
        mock_session.state = {"status": "active"}
        mock_session.session_type.value = "development"
        mock_session.status.value = "active"
        mock_session.created_at = Mock()
        mock_session.last_activity = Mock()
        mock_session.update_shared_context = Mock()
        
        mock_db_session = AsyncMock()
        mock_db_session.get.return_value = mock_session
        mock_db_session.commit = AsyncMock()
        
        mock_compressor = AsyncMock()
        mock_result = Mock()
        mock_result.summary = "End-to-end test summary"
        mock_result.key_insights = ["E2E insight"]
        mock_result.decisions_made = ["E2E decision"]
        mock_result.patterns_identified = ["E2E pattern"]
        mock_result.importance_score = 0.8
        mock_result.compression_ratio = 0.6
        mock_result.original_token_count = 1000
        mock_result.compressed_token_count = 400
        mock_compressor.compress_conversation.return_value = mock_result
        
        with patch('app.core.hive_slash_commands.get_context_compressor', return_value=mock_compressor):
            with patch('app.core.hive_slash_commands.get_db_session') as mock_get_db:
                mock_get_db.return_value.__aenter__.return_value = mock_db_session
                
                start_time = time.time()
                result = await compact_command.execute(
                    args=["test-session-123", "--level=standard"],
                    context={}
                )
                end_time = time.time()
        
        execution_time = end_time - start_time
        
        # End-to-end should still meet performance target
        assert execution_time < 15.0, f"End-to-end compression took {execution_time:.2f}s"
        assert result["success"] is True
        assert result["performance_met"] is True
        
        print(f"End-to-end performance: {execution_time:.2f}s")
    
    def test_compression_time_estimation_accuracy(self, compressor):
        """Test accuracy of compression time estimation"""
        # Test different token counts
        token_counts = [100, 1000, 5000, 10000, 50000]
        
        for token_count in token_counts:
            estimated_time = compressor.estimate_compression_time(token_count)
            
            # Estimates should be reasonable
            assert estimated_time > 0, "Estimated time should be positive"
            assert estimated_time < 60, f"Estimated time {estimated_time}s too high for {token_count} tokens"
            
            # Should scale with token count
            if token_count > 1000:
                base_estimate = compressor.estimate_compression_time(1000)
                assert estimated_time >= base_estimate, "Estimates should scale with token count"
        
        print(f"Time estimates: {[(tc, compressor.estimate_compression_time(tc)) for tc in token_counts]}")
    
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking_overhead(self, compressor):
        """Test that performance metrics tracking doesn't impact performance"""
        content = self.generate_test_content("medium")
        
        # Measure compression time with metrics tracking
        start_time = time.time()
        result1 = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD
        )
        time_with_metrics = time.time() - start_time
        
        # Get metrics (this exercises the tracking code)
        metrics = compressor.get_performance_metrics()
        assert metrics["total_compressions"] > 0
        
        # Perform another compression
        start_time = time.time()
        result2 = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD
        )
        time_with_more_metrics = time.time() - start_time
        
        # Metrics tracking shouldn't add significant overhead
        overhead = abs(time_with_more_metrics - time_with_metrics)
        assert overhead < 1.0, f"Metrics tracking added {overhead:.2f}s overhead"
        
        print(f"Metrics tracking overhead: {overhead:.3f}s")
    
    @pytest.mark.asyncio
    async def test_compression_performance_under_load(self, compressor):
        """Test compression performance under sustained load"""
        content = self.generate_test_content("medium")
        num_iterations = 50
        times = []
        
        # Perform sustained compression operations
        for i in range(num_iterations):
            start_time = time.time()
            result = await compressor.compress_conversation(
                conversation_content=f"{content} - Load test iteration {i}",
                compression_level=CompressionLevel.STANDARD
            )
            end_time = time.time()
            
            times.append(end_time - start_time)
            assert result.summary
        
        # Analyze performance consistency
        avg_time = mean(times)
        median_time = median(times)
        max_time = max(times)
        min_time = min(times)
        
        # Performance should be consistent
        assert max_time < 15.0, f"Slowest compression under load: {max_time:.2f}s"
        assert avg_time < 10.0, f"Average compression time under load: {avg_time:.2f}s"
        
        # Variance should be reasonable (max shouldn't be more than 3x min)
        variance_ratio = max_time / min_time if min_time > 0 else float('inf')
        assert variance_ratio < 3.0, f"Performance variance too high: {variance_ratio:.2f}x"
        
        print(f"Load test performance: avg {avg_time:.2f}s, median {median_time:.2f}s, "
              f"range {min_time:.2f}s-{max_time:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__])