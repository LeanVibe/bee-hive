"""
Unit tests for Context Compression Service
Tests the ContextCompressor class and related functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from app.core.context_compression import (
    ContextCompressor,
    CompressedContext,
    CompressionLevel,
    get_context_compressor
)
from app.models.context import ContextType


class TestContextCompressor:
    """Test suite for ContextCompressor class"""
    
    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock Anthropic client for testing"""
        client = AsyncMock()
        
        # Mock response structure
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = '''
        {
            "summary": "This is a compressed summary of the conversation.",
            "key_insights": ["Important insight 1", "Important insight 2"],
            "decisions_made": ["Decision 1", "Decision 2"],
            "patterns_identified": ["Pattern 1", "Pattern 2"],
            "importance_score": 0.8
        }
        '''
        
        client.messages.create.return_value = mock_response
        return client
    
    @pytest.fixture
    def compressor(self, mock_anthropic_client):
        """Create ContextCompressor instance with mocked client"""
        return ContextCompressor(llm_client=mock_anthropic_client)
    
    def test_context_compressor_initialization(self, compressor):
        """Test that ContextCompressor initializes correctly"""
        assert compressor.model_name == "claude-3-haiku-20240307"
        assert compressor._compression_count == 0
        assert compressor._total_compression_time == 0.0
        assert compressor._total_tokens_saved == 0
        assert compressor._compression_ratios == []
    
    def test_token_counting(self, compressor):
        """Test token counting functionality"""
        text = "This is a test message for token counting."
        token_count = compressor.count_tokens(text)
        
        assert isinstance(token_count, int)
        assert token_count > 0
    
    def test_compression_time_estimation(self, compressor):
        """Test compression time estimation"""
        # Test various token counts
        estimated_time_small = compressor.estimate_compression_time(100)
        estimated_time_large = compressor.estimate_compression_time(10000)
        
        assert estimated_time_small > 0
        assert estimated_time_large > estimated_time_small
    
    @pytest.mark.asyncio
    async def test_compress_conversation_standard(self, compressor):
        """Test standard conversation compression"""
        content = """
        User: How do I implement a Redis cache?
        Agent: To implement Redis cache, you need to:
        1. Install Redis server
        2. Configure connection settings
        3. Implement cache logic
        4. Handle cache invalidation
        This is a comprehensive conversation about Redis implementation.
        """
        
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD
        )
        
        assert isinstance(result, CompressedContext)
        assert result.summary
        assert result.compression_ratio >= 0
        assert result.original_token_count > 0
        assert result.compressed_token_count > 0
        assert len(result.key_insights) > 0
        assert len(result.decisions_made) > 0
        assert 0 <= result.importance_score <= 1
    
    @pytest.mark.asyncio
    async def test_compress_conversation_light(self, compressor):
        """Test light compression level"""
        content = "This is a short test conversation for light compression."
        
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.LIGHT
        )
        
        assert result.metadata["compression_level"] == "light"
        # Light compression should preserve more content
        assert result.compression_ratio < 0.4  # Less than 40% reduction
    
    @pytest.mark.asyncio
    async def test_compress_conversation_aggressive(self, compressor):
        """Test aggressive compression level"""
        content = """
        This is a very long conversation that contains lots of details.
        We need to test aggressive compression to see if it can reduce
        the content significantly while preserving the key information.
        There are many sentences here to provide sufficient content for
        the compression algorithm to work with and demonstrate its capabilities.
        """
        
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.AGGRESSIVE
        )
        
        assert result.metadata["compression_level"] == "aggressive"
        # Aggressive compression should achieve higher reduction
        assert result.compression_ratio > 0.3  # More than 30% reduction
    
    @pytest.mark.asyncio
    async def test_compress_short_content(self, compressor):
        """Test compression of very short content"""
        content = "Short message"
        
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD
        )
        
        # Short content should not be compressed
        assert result.compression_ratio == 0.0
        assert result.summary == content
    
    @pytest.mark.asyncio
    async def test_compress_with_context_type_decision(self, compressor):
        """Test compression with decision context type"""
        content = """
        Team meeting about architecture decision.
        Decision: We will use PostgreSQL instead of MongoDB.
        Rationale: Better ACID compliance and SQL support.
        Outcome: Migration plan to be implemented next sprint.
        """
        
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD,
            context_type=ContextType.DECISION
        )
        
        assert result.metadata["context_type"] == "decision"
        assert len(result.decisions_made) > 0
    
    @pytest.mark.asyncio
    async def test_compress_with_context_type_error_resolution(self, compressor):
        """Test compression with error resolution context type"""
        content = """
        Bug report: Application crashes on user login.
        Root cause: NULL pointer exception in authentication service.
        Solution: Added null checks and improved error handling.
        Verification: Issue resolved in testing environment.
        """
        
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD,
            context_type=ContextType.ERROR_RESOLUTION
        )
        
        assert result.metadata["context_type"] == "error_resolution"
        # Should preserve error resolution details
        assert "error" in result.summary.lower() or "bug" in result.summary.lower()
    
    @pytest.mark.asyncio
    async def test_adaptive_compress_small_content(self, compressor):
        """Test adaptive compression for content already under target"""
        content = "Small content that doesn't need compression."
        target_tokens = 100
        
        result = await compressor.adaptive_compress(
            content=content,
            target_token_count=target_tokens
        )
        
        # Should return original content if already under target
        assert result.compression_ratio == 0.0
        assert result.summary == content
    
    @pytest.mark.asyncio
    async def test_adaptive_compress_large_content(self, compressor):
        """Test adaptive compression for content over target"""
        content = """
        This is a much longer piece of content that definitely exceeds
        our target token count and should trigger adaptive compression.
        The system should automatically choose the appropriate compression
        level based on the required reduction ratio to meet the target.
        """ * 10  # Repeat to make it longer
        
        target_tokens = 50
        
        result = await compressor.adaptive_compress(
            content=content,
            target_token_count=target_tokens
        )
        
        assert result.compression_ratio > 0
        assert result.compressed_token_count <= target_tokens * 1.1  # Allow 10% margin
    
    @pytest.mark.asyncio
    async def test_compress_with_api_error(self, mock_anthropic_client):
        """Test compression behavior when API call fails"""
        # Mock API error
        mock_anthropic_client.messages.create.side_effect = Exception("API Error")
        
        compressor = ContextCompressor(llm_client=mock_anthropic_client)
        content = "Test content for error handling"
        
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD
        )
        
        # Should return fallback result
        assert result.compression_ratio == 0.0
        assert result.summary == content
        assert "error" in result.metadata
    
    @pytest.mark.asyncio
    async def test_compress_with_malformed_response(self, mock_anthropic_client):
        """Test compression with malformed JSON response"""
        # Mock malformed response
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Not valid JSON response"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        compressor = ContextCompressor(llm_client=mock_anthropic_client)
        content = "Test content for malformed response"
        
        result = await compressor.compress_conversation(
            conversation_content=content,
            compression_level=CompressionLevel.STANDARD
        )
        
        # Should handle gracefully with fallback parsing
        assert result.summary
        assert isinstance(result.key_insights, list)
        assert isinstance(result.decisions_made, list)
    
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, compressor):
        """Test that performance metrics are tracked correctly"""
        content = "Test content for metrics tracking"
        
        # Perform multiple compressions
        for _ in range(3):
            await compressor.compress_conversation(
                conversation_content=content,
                compression_level=CompressionLevel.STANDARD
            )
        
        metrics = compressor.get_performance_metrics()
        
        assert metrics["total_compressions"] == 3
        assert metrics["average_compression_time_s"] > 0
        assert metrics["total_tokens_saved"] >= 0
        assert metrics["model_used"] == compressor.model_name
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, compressor):
        """Test successful health check"""
        health_result = await compressor.health_check()
        
        assert health_result["status"] == "healthy"
        assert health_result["model"] == compressor.model_name
        assert "test_compression_ratio" in health_result
        assert "performance" in health_result
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_anthropic_client):
        """Test health check when service is unhealthy"""
        # Mock API failure
        mock_anthropic_client.messages.create.side_effect = Exception("Service unavailable")
        
        compressor = ContextCompressor(llm_client=mock_anthropic_client)
        health_result = await compressor.health_check()
        
        assert health_result["status"] == "unhealthy"
        assert "error" in health_result
        assert health_result["model"] == compressor.model_name
    
    def test_build_compression_prompt_standard(self, compressor):
        """Test compression prompt building for standard level"""
        content = "Test content"
        
        prompt = compressor._build_compression_prompt(
            content=content,
            compression_level=CompressionLevel.STANDARD,
            context_type=None,
            preserve_decisions=True,
            preserve_patterns=True
        )
        
        assert "Compress this content by 50-60%" in prompt
        assert "decisions made" in prompt
        assert "patterns" in prompt
        assert content in prompt
        assert "JSON" in prompt
    
    def test_build_compression_prompt_context_specific(self, compressor):
        """Test compression prompt building with context type"""
        content = "Test content"
        
        prompt = compressor._build_compression_prompt(
            content=content,
            compression_level=CompressionLevel.STANDARD,
            context_type=ContextType.DECISION,
            preserve_decisions=True,
            preserve_patterns=False
        )
        
        assert "decision made" in prompt
        assert "rationale" in prompt
        assert "patterns" not in prompt  # Should not preserve patterns
    
    def test_parse_compression_response_valid_json(self, compressor):
        """Test parsing of valid JSON response"""
        response = '''
        {
            "summary": "Test summary",
            "key_insights": ["insight1", "insight2"],
            "decisions_made": ["decision1"],
            "patterns_identified": ["pattern1"],
            "importance_score": 0.7
        }
        '''
        
        parsed = compressor._parse_compression_response(response)
        
        assert parsed["summary"] == "Test summary"
        assert len(parsed["key_insights"]) == 2
        assert len(parsed["decisions_made"]) == 1
        assert parsed["importance_score"] == 0.7
    
    def test_parse_compression_response_invalid_json(self, compressor):
        """Test parsing of invalid JSON response"""
        response = "This is just a plain text response without JSON"
        
        parsed = compressor._parse_compression_response(response)
        
        assert parsed["summary"] == response.strip()
        assert parsed["key_insights"] == []
        assert parsed["importance_score"] == 0.5


class TestCompressedContext:
    """Test suite for CompressedContext class"""
    
    def test_compressed_context_initialization(self):
        """Test CompressedContext initialization"""
        context = CompressedContext(
            original_id="test-123",
            summary="Test summary",
            key_insights=["insight1", "insight2"],
            decisions_made=["decision1"],
            patterns_identified=["pattern1"],
            importance_score=0.8,
            compression_ratio=0.5,
            original_token_count=1000,
            compressed_token_count=500
        )
        
        assert context.original_id == "test-123"
        assert context.summary == "Test summary"
        assert len(context.key_insights) == 2
        assert context.importance_score == 0.8
        assert context.compression_ratio == 0.5
        assert isinstance(context.compressed_at, datetime)
    
    def test_compressed_context_to_dict(self):
        """Test CompressedContext serialization"""
        context = CompressedContext(
            summary="Test summary",
            importance_score=0.7,
            compression_ratio=0.4,
            original_token_count=800,
            compressed_token_count=480
        )
        
        data = context.to_dict()
        
        assert data["summary"] == "Test summary"
        assert data["importance_score"] == 0.7
        assert data["compression_ratio"] == 0.4
        assert "compressed_at" in data
        assert isinstance(data["compressed_at"], str)


class TestCompressionSingleton:
    """Test suite for compression service singleton"""
    
    def test_get_context_compressor_singleton(self):
        """Test that singleton returns same instance"""
        compressor1 = get_context_compressor()
        compressor2 = get_context_compressor()
        
        assert compressor1 is compressor2
    
    @patch('app.core.context_compression._compressor', None)
    def test_get_context_compressor_creates_new_instance(self):
        """Test that singleton creates new instance when needed"""
        compressor = get_context_compressor()
        
        assert isinstance(compressor, ContextCompressor)


class TestCompressionLevelEnum:
    """Test suite for CompressionLevel enum"""
    
    def test_compression_level_values(self):
        """Test CompressionLevel enum values"""
        assert CompressionLevel.LIGHT.value == "light"
        assert CompressionLevel.STANDARD.value == "standard"
        assert CompressionLevel.AGGRESSIVE.value == "aggressive"
    
    def test_compression_level_iteration(self):
        """Test CompressionLevel enum iteration"""
        levels = list(CompressionLevel)
        assert len(levels) == 3
        assert CompressionLevel.LIGHT in levels
        assert CompressionLevel.STANDARD in levels
        assert CompressionLevel.AGGRESSIVE in levels


if __name__ == "__main__":
    pytest.main([__file__])