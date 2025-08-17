"""
Edge case and error scenario tests for Context Compression
Tests various edge cases, error conditions, and boundary scenarios
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import json

from app.core.context_compression import (
    ContextCompressor,
    CompressedContext,
    CompressionLevel
)
from app.core.hive_slash_commands import HiveCompactCommand
from app.models.context import ContextType


class TestContextCompressionEdgeCases:
    """Test suite for edge cases in context compression"""
    
    @pytest.fixture
    def compressor(self):
        """Create ContextCompressor with mocked client"""
        mock_client = AsyncMock()
        return ContextCompressor(llm_client=mock_client)
    
    @pytest.fixture
    def compact_command(self):
        """Create HiveCompactCommand instance"""
        return HiveCompactCommand()
    
    @pytest.mark.asyncio
    async def test_compression_with_empty_content(self, compressor):
        """Test compression behavior with empty content"""
        content = ""
        
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD
        )
        
        # Should not compress empty content
        assert result.compression_ratio == 0.0
        assert result.summary == content
        assert result.original_token_count == 0
        assert result.compressed_token_count == 0
    
    @pytest.mark.asyncio
    async def test_compression_with_whitespace_only(self, compressor):
        """Test compression with only whitespace content"""
        content = "   \n\t   \n   "
        
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD
        )
        
        # Should handle whitespace-only content gracefully
        assert result.compression_ratio == 0.0
        assert result.summary == content
    
    @pytest.mark.asyncio
    async def test_compression_with_very_long_content(self, compressor):
        """Test compression with extremely long content"""
        # Create very long content (>100k characters)
        long_content = "This is a test sentence. " * 5000  # ~125k characters
        
        # Mock response for long content
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "summary": "Long content compressed summary",
            "key_insights": ["Long content insight"],
            "decisions_made": ["Long content decision"],
            "patterns_identified": ["Long content pattern"],
            "importance_score": 0.7
        })
        compressor.llm_client.messages.create.return_value = mock_response
        
        result = await compressor.compress_conversation(
            conversation_content=long_content,
            compression_level=CompressionLevel.AGGRESSIVE
        )
        
        assert result.summary
        assert result.original_token_count > 10000  # Should be very large
        assert result.compression_ratio > 0.5  # Should achieve significant compression
    
    @pytest.mark.asyncio
    async def test_compression_with_special_characters(self, compressor):
        """Test compression with special Unicode characters"""
        content = """
        Content with special characters: 🚀 ñ á é í ó ú ü
        Emojis: 😀 🎉 🔥 💯 
        Mathematical symbols: ∑ ∆ π ∞ ≤ ≥
        Other Unicode: ™ © ® § ¿ ¡
        Chinese: 你好世界
        Arabic: مرحبا بالعالم
        Russian: Привет мир
        """
        
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "summary": "Content with international characters and symbols",
            "key_insights": ["Unicode handling"],
            "decisions_made": ["Support international content"],
            "patterns_identified": ["Multilingual pattern"],
            "importance_score": 0.6
        })
        compressor.llm_client.messages.create.return_value = mock_response
        
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD
        )
        
        assert result.summary
        assert result.compression_ratio > 0
    
    @pytest.mark.asyncio
    async def test_compression_with_malformed_json_in_content(self, compressor):
        """Test compression when content contains malformed JSON"""
        content = """
        Here's some malformed JSON: {"key": "value", "broken": }
        And some more: [1, 2, 3, 
        Invalid syntax: {key: value}
        """
        
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "summary": "Content discussing JSON with syntax errors",
            "key_insights": ["JSON parsing issues"],
            "decisions_made": ["Handle malformed data"],
            "patterns_identified": ["Error handling pattern"],
            "importance_score": 0.5
        })
        compressor.llm_client.messages.create.return_value = mock_response
        
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD
        )
        
        assert result.summary
        assert "JSON" in result.summary
    
    @pytest.mark.asyncio
    async def test_compression_with_extremely_high_target_tokens(self, compressor):
        """Test adaptive compression with unrealistically high target tokens"""
        content = "Short content that should not need compression."
        target_tokens = 100000  # Much larger than content
        
        result = await compressor.adaptive_compress(
            content=content,
            target_token_count=target_tokens
        )
        
        # Should return original content without compression
        assert result.compression_ratio == 0.0
        assert result.summary == content
        assert result.compressed_token_count == result.original_token_count
    
    @pytest.mark.asyncio
    async def test_compression_with_zero_target_tokens(self, compressor):
        """Test adaptive compression with zero target tokens"""
        content = "Some content to compress to zero tokens somehow."
        target_tokens = 1  # Minimal target
        
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "summary": "Minimal",
            "key_insights": [],
            "decisions_made": [],
            "patterns_identified": [],
            "importance_score": 0.1
        })
        compressor.llm_client.messages.create.return_value = mock_response
        
        result = await compressor.adaptive_compress(
            content=content,
            target_token_count=target_tokens
        )
        
        # Should attempt aggressive compression
        assert result.compression_ratio > 0
        assert len(result.summary) < len(content)
    
    @pytest.mark.asyncio
    async def test_compression_api_timeout(self, compressor):
        """Test compression behavior when API times out"""
        content = "Content that will cause API timeout"
        
        # Mock timeout error
        compressor.llm_client.messages.create.side_effect = asyncio.TimeoutError("Request timed out")
        
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD
        )
        
        # Should return fallback result
        assert result.compression_ratio == 0.0
        assert result.summary == content
        assert "error" in result.metadata
    
    @pytest.mark.asyncio
    async def test_compression_api_rate_limit(self, compressor):
        """Test compression behavior when API rate limit is hit"""
        content = "Content that will hit rate limit"
        
        # Mock rate limit error
        class RateLimitError(Exception):
            pass
        
        compressor.llm_client.messages.create.side_effect = RateLimitError("Rate limit exceeded")
        
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD
        )
        
        # Should return fallback result
        assert result.compression_ratio == 0.0
        assert result.summary == content
        assert "error" in result.metadata
    
    @pytest.mark.asyncio
    async def test_compression_with_invalid_response_structure(self, compressor):
        """Test compression when API returns invalid response structure"""
        content = "Content for invalid response test"
        
        # Mock response with missing content
        mock_response = Mock()
        mock_response.content = []  # Empty content array
        compressor.llm_client.messages.create.return_value = mock_response
        
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD
        )
        
        # Should handle gracefully
        assert result.compression_ratio == 0.0
        assert result.summary == content
    
    @pytest.mark.asyncio
    async def test_compression_with_partial_json_response(self, compressor):
        """Test compression when API returns partial JSON"""
        content = "Content for partial JSON test"
        
        # Mock response with partial JSON
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = '{"summary": "Partial summary", "key_insights": ["insight"]'  # Missing closing brace
        compressor.llm_client.messages.create.return_value = mock_response
        
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD
        )
        
        # Should use fallback parsing
        assert result.summary
        assert isinstance(result.key_insights, list)
    
    @pytest.mark.asyncio
    async def test_compression_performance_with_concurrent_requests(self, compressor):
        """Test compression performance under concurrent load"""
        content = "Content for concurrent compression testing"
        
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "summary": "Concurrent compression result",
            "key_insights": ["Concurrency insight"],
            "decisions_made": [],
            "patterns_identified": [],
            "importance_score": 0.6
        })
        compressor.llm_client.messages.create.return_value = mock_response
        
        # Create multiple concurrent compression tasks
        tasks = [
            compressor.compress_conversation(
                conversation_content=f"{content} - Request {i}",
                compression_level=CompressionLevel.STANDARD
            )
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All requests should succeed
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, CompressedContext)
            assert result.summary
    
    @pytest.mark.asyncio
    async def test_hive_command_with_invalid_session_format(self, compact_command):
        """Test HiveCompactCommand with invalid session ID format"""
        invalid_session_ids = [
            "not-a-uuid",
            "123",
            "",
            "session-with-spaces in name",
            "session@with!special$characters",
            "too-long-" + "a" * 100,
            None
        ]
        
        for invalid_id in invalid_session_ids:
            if invalid_id is None:
                args = []
            else:
                args = [str(invalid_id)]
            
            with patch.object(compact_command, '_extract_session_context') as mock_extract:
                mock_extract.return_value = None
                
                result = await compact_command.execute(args=args, context={})
                
                # Should handle gracefully (either extract from context or fail gracefully)
                assert "success" in result
    
    @pytest.mark.asyncio
    async def test_hive_command_with_invalid_arguments(self, compact_command):
        """Test HiveCompactCommand with various invalid arguments"""
        invalid_arg_sets = [
            ["--level=invalid"],
            ["--target-tokens=not-a-number"],
            ["--target-tokens=-100"],
            ["--unknown-flag"],
            ["--level="],
            ["--target-tokens="],
            ["session-123", "--level=standard", "--invalid-combo"]
        ]
        
        for args in invalid_arg_sets:
            # Some should fail validation, others should be handled gracefully
            result = await compact_command.execute(args=args, context={})
            assert "success" in result
            # If success is False, should have error message
            if not result.get("success", True):
                assert "error" in result
    
    @pytest.mark.asyncio
    async def test_compression_memory_usage_with_large_history(self, compressor):
        """Test memory usage when compression history grows large"""
        # Simulate large compression history
        large_content = "Large content for memory test. " * 1000
        
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "summary": "Memory test summary",
            "key_insights": ["Memory insight"],
            "decisions_made": [],
            "patterns_identified": [],
            "importance_score": 0.5
        })
        compressor.llm_client.messages.create.return_value = mock_response
        
        # Perform many compressions to build up metrics
        for i in range(100):
            await compressor.compress_conversation(
                conversation_content=f"{large_content} - Iteration {i}",
                compression_level=CompressionLevel.LIGHT
            )
        
        metrics = compressor.get_performance_metrics()
        
        # Should track metrics correctly
        assert metrics["total_compressions"] == 100
        assert metrics["average_compression_time_s"] > 0
        assert metrics["total_tokens_saved"] >= 0
    
    @pytest.mark.asyncio
    async def test_compression_with_network_interruption(self, compressor):
        """Test compression behavior during network interruption"""
        content = "Content during network interruption"
        
        # Mock network error
        class NetworkError(Exception):
            pass
        
        compressor.llm_client.messages.create.side_effect = NetworkError("Network connection lost")
        
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD
        )
        
        # Should return fallback result
        assert result.compression_ratio == 0.0
        assert result.summary == content
        assert "error" in result.metadata
    
    @pytest.mark.asyncio
    async def test_compression_context_type_edge_cases(self, compressor):
        """Test compression with various context type edge cases"""
        content = "Test content for context type edge cases"
        
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps({
            "summary": "Context type test summary",
            "key_insights": ["Type insight"],
            "decisions_made": ["Type decision"],
            "patterns_identified": ["Type pattern"],
            "importance_score": 0.7
        })
        compressor.llm_client.messages.create.return_value = mock_response
        
        # Test with all context types
        context_types = [None] + list(ContextType)
        
        for context_type in context_types:
            result = await compressor.compress_conversation(
                conversation_content=content,
                compression_level=CompressionLevel.STANDARD,
                context_type=context_type
            )
            
            assert result.summary
            if context_type:
                assert result.metadata["context_type"] == context_type.value
            else:
                assert result.metadata["context_type"] is None
    
    @pytest.mark.asyncio
    async def test_batch_compression_with_mixed_success_failure(self, compressor):
        """Test batch compression when some compressions succeed and others fail"""
        # Create mock contexts
        contexts = []
        for i in range(5):
            mock_context = Mock()
            mock_context.id = f"context-{i}"
            mock_context.content = f"Content {i}"
            mock_context.context_type = ContextType.GENERAL
            contexts.append(mock_context)
        
        # Mock API to fail on specific calls
        def mock_api_call(*args, **kwargs):
            # Fail on every other call
            if hasattr(mock_api_call, 'call_count'):
                mock_api_call.call_count += 1
            else:
                mock_api_call.call_count = 1
            
            if mock_api_call.call_count % 2 == 0:
                raise Exception("Simulated API failure")
            
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = json.dumps({
                "summary": f"Batch summary {mock_api_call.call_count}",
                "key_insights": ["Batch insight"],
                "decisions_made": [],
                "patterns_identified": [],
                "importance_score": 0.5
            })
            return mock_response
        
        compressor.llm_client.messages.create.side_effect = mock_api_call
        
        results = await compressor.compress_context_batch(
            contexts=contexts,
            compression_level=CompressionLevel.STANDARD
        )
        
        # Should return results for all contexts
        assert len(results) == 5
        
        # Some should be successful, others should have errors
        successful_results = [r for r in results if r.compression_ratio > 0]
        failed_results = [r for r in results if "error" in r.metadata]
        
        assert len(successful_results) > 0
        assert len(failed_results) > 0
        assert len(successful_results) + len(failed_results) == 5
    
    @pytest.mark.asyncio
    async def test_compression_with_extremely_nested_json_response(self, compressor):
        """Test compression when API returns deeply nested JSON"""
        content = "Content for nested JSON test"
        
        # Create deeply nested JSON response
        nested_response = {
            "summary": "Nested summary",
            "key_insights": [
                {
                    "insight": "Nested insight",
                    "metadata": {
                        "level": 1,
                        "nested": {
                            "level": 2,
                            "deeply": {
                                "level": 3,
                                "data": ["item1", "item2"]
                            }
                        }
                    }
                }
            ],
            "decisions_made": ["Simple decision"],
            "patterns_identified": ["Simple pattern"],
            "importance_score": 0.8
        }
        
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = json.dumps(nested_response)
        compressor.llm_client.messages.create.return_value = mock_response
        
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD
        )
        
        # Should handle nested response gracefully
        assert result.summary == "Nested summary"
        # Should flatten nested insights or handle appropriately
        assert isinstance(result.key_insights, list)
    
    @pytest.mark.asyncio
    async def test_health_check_under_stress(self, compressor):
        """Test health check functionality under stress conditions"""
        # Mock slow API response
        async def slow_api_call(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slow response
            mock_response = Mock()
            mock_response.content = [Mock()]
            mock_response.content[0].text = json.dumps({
                "summary": "Health check summary",
                "key_insights": [],
                "decisions_made": [],
                "patterns_identified": [],
                "importance_score": 0.5
            })
            return mock_response
        
        compressor.llm_client.messages.create.side_effect = slow_api_call
        
        # Run health check
        health_result = await compressor.health_check()
        
        assert health_result["status"] == "healthy"
        assert "test_compression_ratio" in health_result
        assert "performance" in health_result


if __name__ == "__main__":
    pytest.main([__file__])