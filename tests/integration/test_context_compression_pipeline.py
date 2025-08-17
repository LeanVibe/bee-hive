"""
Integration tests for Context Compression Pipeline
Tests the full end-to-end compression flow from API to database storage
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from httpx import AsyncClient

from app.main import app
from app.core.context_compression import get_context_compressor, ContextCompressor
from app.core.hive_slash_commands import get_hive_command_registry


class TestContextCompressionPipeline:
    """Test suite for full context compression pipeline"""
    
    @pytest.fixture
    def mock_session_data(self):
        """Mock session data for testing"""
        return {
            "id": "test-session-123",
            "name": "Test Development Session",
            "description": "A test session for compression testing",
            "objectives": ["Implement feature X", "Fix bug Y"],
            "shared_context": {
                "project": "test-project",
                "phase": "development"
            },
            "state": {"status": "active"},
            "session_type": "development",
            "status": "active",
            "created_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "participant_agents": ["agent-1", "agent-2"]
        }
    
    @pytest.fixture
    def mock_anthropic_response(self):
        """Mock Anthropic API response for compression"""
        return '''
        {
            "summary": "Development session focused on implementing feature X and fixing bug Y. Team discussed architecture decisions and implemented solutions.",
            "key_insights": [
                "Feature X requires async implementation for performance",
                "Bug Y was caused by race condition in concurrent operations",
                "Team decided to use PostgreSQL for data persistence"
            ],
            "decisions_made": [
                "Use async/await pattern for feature X implementation",
                "Implement connection pooling to prevent race conditions",
                "Migrate from SQLite to PostgreSQL for production"
            ],
            "patterns_identified": [
                "Async programming pattern for I/O intensive operations",
                "Database connection management best practices",
                "Error handling patterns for concurrent systems"
            ],
            "importance_score": 0.85
        }
        '''
    
    @pytest.mark.asyncio
    async def test_end_to_end_session_compression(self, mock_session_data, mock_anthropic_response):
        """Test complete end-to-end session compression flow"""
        session_id = mock_session_data["id"]
        
        # Mock Anthropic client
        mock_anthropic_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = mock_anthropic_response
        mock_anthropic_client.messages.create.return_value = mock_response
        
        # Mock database session
        mock_session = Mock()
        mock_session.id = session_id
        mock_session.name = mock_session_data["name"]
        mock_session.description = mock_session_data["description"]
        mock_session.objectives = mock_session_data["objectives"]
        mock_session.shared_context = mock_session_data["shared_context"]
        mock_session.state = mock_session_data["state"]
        mock_session.session_type.value = mock_session_data["session_type"]
        mock_session.status.value = mock_session_data["status"]
        mock_session.created_at = mock_session_data["created_at"]
        mock_session.last_activity = mock_session_data["last_activity"]
        mock_session.update_shared_context = Mock()
        
        mock_db_session = AsyncMock()
        mock_db_session.get.return_value = mock_session
        mock_db_session.commit = AsyncMock()
        
        with patch('app.core.context_compression.AsyncAnthropic', return_value=mock_anthropic_client):
            with patch('app.core.hive_slash_commands.get_db_session') as mock_get_db:
                mock_get_db.return_value.__aenter__.return_value = mock_db_session
                
                # Execute compression via hive command
                registry = get_hive_command_registry()
                result = await registry.execute_command(
                    f"/hive:compact {session_id} --level=standard --preserve-decisions --preserve-patterns",
                    context={"api_request": True}
                )
        
        # Verify successful compression
        assert result["success"] is True
        assert result["session_id"] == session_id
        assert result["compression_level"] == "standard"
        assert result["original_tokens"] > 0
        assert result["compressed_tokens"] > 0
        assert result["compression_ratio"] > 0
        assert result["summary"]
        assert len(result["key_insights"]) > 0
        assert len(result["decisions_made"]) > 0
        assert len(result["patterns_identified"]) > 0
        assert 0 <= result["importance_score"] <= 1
        
        # Verify database storage
        mock_session.update_shared_context.assert_called()
        mock_db_session.commit.assert_called()
        
        # Verify performance target met
        assert result["performance_met"] is True
        assert result["compression_time_seconds"] < 15.0
    
    @pytest.mark.asyncio
    async def test_compression_with_different_levels(self, mock_session_data, mock_anthropic_response):
        """Test compression with different compression levels"""
        session_id = mock_session_data["id"]
        
        # Mock setup
        mock_anthropic_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = mock_anthropic_response
        mock_anthropic_client.messages.create.return_value = mock_response
        
        mock_session = Mock()
        mock_session.id = session_id
        mock_session.description = "Test content for compression level testing"
        mock_session.objectives = ["Test objective"]
        mock_session.shared_context = {"test": "data"}
        mock_session.state = {"status": "active"}
        mock_session.session_type.value = "development"
        mock_session.status.value = "active"
        mock_session.created_at = datetime.utcnow()
        mock_session.last_activity = datetime.utcnow()
        mock_session.update_shared_context = Mock()
        
        mock_db_session = AsyncMock()
        mock_db_session.get.return_value = mock_session
        mock_db_session.commit = AsyncMock()
        
        registry = get_hive_command_registry()
        
        with patch('app.core.context_compression.AsyncAnthropic', return_value=mock_anthropic_client):
            with patch('app.core.hive_slash_commands.get_db_session') as mock_get_db:
                mock_get_db.return_value.__aenter__.return_value = mock_db_session
                
                # Test different compression levels
                levels = ["light", "standard", "aggressive"]
                results = {}
                
                for level in levels:
                    result = await registry.execute_command(
                        f"/hive:compact {session_id} --level={level}",
                        context={}
                    )
                    results[level] = result
                    
                    assert result["success"] is True
                    assert result["compression_level"] == level
        
        # Verify all levels worked
        assert len(results) == 3
        for level, result in results.items():
            assert result["compression_level"] == level
    
    @pytest.mark.asyncio
    async def test_compression_with_target_tokens(self, mock_session_data, mock_anthropic_response):
        """Test adaptive compression with target token count"""
        session_id = mock_session_data["id"]
        target_tokens = 200
        
        # Mock setup
        mock_anthropic_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = mock_anthropic_response
        mock_anthropic_client.messages.create.return_value = mock_response
        
        mock_session = Mock()
        mock_session.description = "A longer test session description that should be compressed to meet the target token count specified in the compression request."
        mock_session.objectives = ["Long objective 1", "Long objective 2"]
        mock_session.shared_context = {"detailed": "context", "with": "multiple", "key": "value", "pairs": "here"}
        mock_session.state = {"status": "active", "progress": "50%"}
        mock_session.session_type.value = "development"
        mock_session.status.value = "active"
        mock_session.created_at = datetime.utcnow()
        mock_session.last_activity = datetime.utcnow()
        mock_session.update_shared_context = Mock()
        
        mock_db_session = AsyncMock()
        mock_db_session.get.return_value = mock_session
        mock_db_session.commit = AsyncMock()
        
        with patch('app.core.context_compression.AsyncAnthropic', return_value=mock_anthropic_client):
            with patch('app.core.hive_slash_commands.get_db_session') as mock_get_db:
                mock_get_db.return_value.__aenter__.return_value = mock_db_session
                
                registry = get_hive_command_registry()
                result = await registry.execute_command(
                    f"/hive:compact {session_id} --target-tokens={target_tokens}",
                    context={}
                )
        
        assert result["success"] is True
        # Should use adaptive compression when target tokens specified
        assert result["compressed_tokens"] <= target_tokens * 1.2  # Allow some margin
    
    @pytest.mark.asyncio
    async def test_compression_error_handling(self, mock_session_data):
        """Test compression pipeline error handling"""
        session_id = mock_session_data["id"]
        
        # Mock Anthropic client failure
        mock_anthropic_client = AsyncMock()
        mock_anthropic_client.messages.create.side_effect = Exception("API rate limit exceeded")
        
        mock_session = Mock()
        mock_session.description = "Test content"
        mock_session.objectives = []
        mock_session.shared_context = {}
        mock_session.state = {}
        mock_session.session_type.value = "development"
        mock_session.status.value = "active"
        mock_session.created_at = datetime.utcnow()
        mock_session.last_activity = datetime.utcnow()
        
        mock_db_session = AsyncMock()
        mock_db_session.get.return_value = mock_session
        
        with patch('app.core.context_compression.AsyncAnthropic', return_value=mock_anthropic_client):
            with patch('app.core.hive_slash_commands.get_db_session') as mock_get_db:
                mock_get_db.return_value.__aenter__.return_value = mock_db_session
                
                registry = get_hive_command_registry()
                result = await registry.execute_command(
                    f"/hive:compact {session_id}",
                    context={}
                )
        
        # Should handle error gracefully
        assert result["success"] is True  # Returns fallback result
        assert result["compression_ratio"] == 0.0  # No compression due to error
        assert "error" in result.get("metadata", {})
    
    @pytest.mark.asyncio
    async def test_compression_with_empty_session(self):
        """Test compression with empty/minimal session content"""
        session_id = "empty-session-123"
        
        # Mock empty session
        mock_session = Mock()
        mock_session.description = ""
        mock_session.objectives = []
        mock_session.shared_context = {}
        mock_session.state = {}
        mock_session.session_type.value = "development"
        mock_session.status.value = "active"
        mock_session.created_at = datetime.utcnow()
        mock_session.last_activity = datetime.utcnow()
        mock_session.update_shared_context = Mock()
        
        mock_db_session = AsyncMock()
        mock_db_session.get.return_value = mock_session
        mock_db_session.commit = AsyncMock()
        
        mock_anthropic_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = '{"summary": "Empty session", "key_insights": [], "decisions_made": [], "patterns_identified": [], "importance_score": 0.1}'
        mock_anthropic_client.messages.create.return_value = mock_response
        
        with patch('app.core.context_compression.AsyncAnthropic', return_value=mock_anthropic_client):
            with patch('app.core.hive_slash_commands.get_db_session') as mock_get_db:
                mock_get_db.return_value.__aenter__.return_value = mock_db_session
                
                registry = get_hive_command_registry()
                result = await registry.execute_command(
                    f"/hive:compact {session_id}",
                    context={}
                )
        
        assert result["success"] is True
        # Should still compress minimal content
        assert result["summary"]
        assert result["importance_score"] >= 0
    
    @pytest.mark.asyncio
    async def test_compression_session_not_found(self):
        """Test compression when session doesn't exist"""
        session_id = "nonexistent-session-456"
        
        # Mock database returning None for session
        mock_db_session = AsyncMock()
        mock_db_session.get.return_value = None
        
        with patch('app.core.hive_slash_commands.get_db_session') as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_db_session
            
            registry = get_hive_command_registry()
            result = await registry.execute_command(
                f"/hive:compact {session_id}",
                context={}
            )
        
        assert result["success"] is False
        assert "No context found to compress" in result["error"]
    
    @pytest.mark.asyncio
    async def test_compression_database_error(self, mock_session_data):
        """Test compression when database access fails"""
        session_id = mock_session_data["id"]
        
        # Mock database error
        with patch('app.core.hive_slash_commands.get_db_session') as mock_get_db:
            mock_get_db.side_effect = Exception("Database connection failed")
            
            registry = get_hive_command_registry()
            result = await registry.execute_command(
                f"/hive:compact {session_id}",
                context={}
            )
        
        assert result["success"] is False
        assert "Context compression failed" in result["message"]
    
    @pytest.mark.asyncio
    async def test_compression_performance_monitoring(self, mock_session_data, mock_anthropic_response):
        """Test performance monitoring during compression"""
        session_id = mock_session_data["id"]
        
        # Mock setup with controlled timing
        mock_anthropic_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = mock_anthropic_response
        mock_anthropic_client.messages.create.return_value = mock_response
        
        mock_session = Mock()
        mock_session.description = "Performance test session"
        mock_session.objectives = ["Test performance"]
        mock_session.shared_context = {"test": "data"}
        mock_session.state = {"status": "active"}
        mock_session.session_type.value = "development"
        mock_session.status.value = "active"
        mock_session.created_at = datetime.utcnow()
        mock_session.last_activity = datetime.utcnow()
        mock_session.update_shared_context = Mock()
        
        mock_db_session = AsyncMock()
        mock_db_session.get.return_value = mock_session
        mock_db_session.commit = AsyncMock()
        
        with patch('app.core.context_compression.AsyncAnthropic', return_value=mock_anthropic_client):
            with patch('app.core.hive_slash_commands.get_db_session') as mock_get_db:
                mock_get_db.return_value.__aenter__.return_value = mock_db_session
                
                start_time = datetime.utcnow()
                
                registry = get_hive_command_registry()
                result = await registry.execute_command(
                    f"/hive:compact {session_id}",
                    context={}
                )
                
                end_time = datetime.utcnow()
                total_time = (end_time - start_time).total_seconds()
        
        assert result["success"] is True
        assert "compression_time_seconds" in result
        assert result["compression_time_seconds"] <= total_time + 1  # Allow for small timing differences
        assert "performance_met" in result
        
        # Performance target is <15 seconds
        if result["compression_time_seconds"] < 15.0:
            assert result["performance_met"] is True
    
    @pytest.mark.asyncio
    async def test_concurrent_compressions(self, mock_session_data, mock_anthropic_response):
        """Test handling of concurrent compression requests"""
        # Create multiple session IDs
        session_ids = [f"test-session-{i}" for i in range(3)]
        
        # Mock setup
        mock_anthropic_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = mock_anthropic_response
        mock_anthropic_client.messages.create.return_value = mock_response
        
        def create_mock_session(session_id):
            mock_session = Mock()
            mock_session.id = session_id
            mock_session.description = f"Test session {session_id}"
            mock_session.objectives = ["Test objective"]
            mock_session.shared_context = {"test": "data"}
            mock_session.state = {"status": "active"}
            mock_session.session_type.value = "development"
            mock_session.status.value = "active"
            mock_session.created_at = datetime.utcnow()
            mock_session.last_activity = datetime.utcnow()
            mock_session.update_shared_context = Mock()
            return mock_session
        
        mock_db_session = AsyncMock()
        mock_db_session.get.side_effect = lambda session_id: create_mock_session(session_id)
        mock_db_session.commit = AsyncMock()
        
        with patch('app.core.context_compression.AsyncAnthropic', return_value=mock_anthropic_client):
            with patch('app.core.hive_slash_commands.get_db_session') as mock_get_db:
                mock_get_db.return_value.__aenter__.return_value = mock_db_session
                
                registry = get_hive_command_registry()
                
                # Execute concurrent compressions
                tasks = [
                    registry.execute_command(f"/hive:compact {session_id}", context={})
                    for session_id in session_ids
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all compressions succeeded
        assert len(results) == 3
        for i, result in enumerate(results):
            assert not isinstance(result, Exception)
            assert result["success"] is True
            assert result["session_id"] == session_ids[i]
    
    @pytest.mark.asyncio
    async def test_compression_with_malformed_json_response(self, mock_session_data):
        """Test compression handling of malformed API response"""
        session_id = mock_session_data["id"]
        
        # Mock malformed response
        mock_anthropic_client = AsyncMock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "This is not valid JSON { malformed"
        mock_anthropic_client.messages.create.return_value = mock_response
        
        mock_session = Mock()
        mock_session.description = "Test content"
        mock_session.objectives = []
        mock_session.shared_context = {}
        mock_session.state = {}
        mock_session.session_type.value = "development"
        mock_session.status.value = "active"
        mock_session.created_at = datetime.utcnow()
        mock_session.last_activity = datetime.utcnow()
        mock_session.update_shared_context = Mock()
        
        mock_db_session = AsyncMock()
        mock_db_session.get.return_value = mock_session
        mock_db_session.commit = AsyncMock()
        
        with patch('app.core.context_compression.AsyncAnthropic', return_value=mock_anthropic_client):
            with patch('app.core.hive_slash_commands.get_db_session') as mock_get_db:
                mock_get_db.return_value.__aenter__.return_value = mock_db_session
                
                registry = get_hive_command_registry()
                result = await registry.execute_command(
                    f"/hive:compact {session_id}",
                    context={}
                )
        
        # Should handle malformed response gracefully
        assert result["success"] is True
        assert result["summary"]  # Should use fallback parsing
        assert isinstance(result["key_insights"], list)
        assert isinstance(result["decisions_made"], list)
        assert isinstance(result["patterns_identified"], list)


if __name__ == "__main__":
    pytest.main([__file__])