"""
Unit tests for HiveCompactCommand
Tests the /hive:compact slash command functionality
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from app.core.hive_slash_commands import (
    HiveCompactCommand,
    HiveSlashCommandRegistry,
    execute_hive_command
)


class TestHiveCompactCommand:
    """Test suite for HiveCompactCommand class"""
    
    @pytest.fixture
    def command(self):
        """Create HiveCompactCommand instance"""
        return HiveCompactCommand()
    
    @pytest.fixture
    def mock_compressor(self):
        """Mock ContextCompressor for testing"""
        mock_result = Mock()
        mock_result.summary = "Compressed conversation summary"
        mock_result.key_insights = ["Key insight 1", "Key insight 2"]
        mock_result.decisions_made = ["Decision 1"]
        mock_result.patterns_identified = ["Pattern 1", "Pattern 2"]
        mock_result.importance_score = 0.8
        mock_result.compression_ratio = 0.6
        mock_result.original_token_count = 1000
        mock_result.compressed_token_count = 400
        mock_result.to_dict.return_value = {
            "summary": mock_result.summary,
            "importance_score": mock_result.importance_score
        }
        
        compressor = AsyncMock()
        compressor.compress_conversation.return_value = mock_result
        compressor.adaptive_compress.return_value = mock_result
        return compressor
    
    def test_command_initialization(self, command):
        """Test HiveCompactCommand initialization"""
        assert command.name == "compact"
        assert "compress" in command.description.lower()
        assert "/hive:compact" in command.usage
    
    def test_validate_args_all_optional(self, command):
        """Test that all arguments are optional"""
        # No arguments should be valid
        assert command.validate_args([]) is True
        assert command.validate_args(None) is True
        
        # Various argument combinations should be valid
        assert command.validate_args(["session-123"]) is True
        assert command.validate_args(["--level=standard"]) is True
        assert command.validate_args(["session-123", "--level=aggressive"]) is True
    
    @pytest.mark.asyncio
    async def test_execute_basic_compression(self, command, mock_compressor):
        """Test basic compression execution without session ID"""
        with patch('app.core.hive_slash_commands.get_context_compressor', return_value=mock_compressor):
            result = await command.execute(args=[], context={"test": "context"})
        
        assert result["success"] is True
        assert "compression_level" in result
        assert result["original_tokens"] == 1000
        assert result["compressed_tokens"] == 400
        assert result["compression_ratio"] == 0.6
        assert result["tokens_saved"] == 600
        assert result["summary"] == "Compressed conversation summary"
        assert len(result["key_insights"]) == 2
        assert result["performance_met"] is True  # Should be under 15 seconds
    
    @pytest.mark.asyncio
    async def test_execute_with_session_id(self, command, mock_compressor):
        """Test compression execution with session ID"""
        session_id = "test-session-123"
        
        with patch('app.core.hive_slash_commands.get_context_compressor', return_value=mock_compressor):
            with patch.object(command, '_extract_session_context') as mock_extract:
                mock_extract.return_value = {
                    "content": "Session conversation content",
                    "metadata": {"session_name": "Test Session"}
                }
                
                result = await command.execute(
                    args=[session_id],
                    context={}
                )
        
        assert result["success"] is True
        assert result["session_id"] == session_id
        mock_extract.assert_called_once_with(session_id)
    
    @pytest.mark.asyncio
    async def test_execute_with_compression_level(self, command, mock_compressor):
        """Test compression with specific level"""
        with patch('app.core.hive_slash_commands.get_context_compressor', return_value=mock_compressor):
            result = await command.execute(
                args=["--level=aggressive"],
                context={"test": "context"}
            )
        
        assert result["success"] is True
        assert result["compression_level"] == "aggressive"
        
        # Verify compressor was called with correct level
        call_args = mock_compressor.compress_conversation.call_args
        assert call_args is not None
    
    @pytest.mark.asyncio
    async def test_execute_with_target_tokens(self, command, mock_compressor):
        """Test compression with target token count"""
        target_tokens = 500
        
        with patch('app.core.hive_slash_commands.get_context_compressor', return_value=mock_compressor):
            result = await command.execute(
                args=[f"--target-tokens={target_tokens}"],
                context={"test": "context"}
            )
        
        assert result["success"] is True
        
        # Should call adaptive_compress when target tokens specified
        mock_compressor.adaptive_compress.assert_called_once()
        call_args = mock_compressor.adaptive_compress.call_args
        assert call_args[1]["target_token_count"] == target_tokens
    
    @pytest.mark.asyncio
    async def test_execute_with_invalid_target_tokens(self, command):
        """Test compression with invalid target token value"""
        result = await command.execute(
            args=["--target-tokens=invalid"],
            context={}
        )
        
        assert result["success"] is False
        assert "Invalid target-tokens value" in result["error"]
        assert command.usage in result["usage"]
    
    @pytest.mark.asyncio
    async def test_execute_with_preserve_flags(self, command, mock_compressor):
        """Test compression with preserve flags"""
        with patch('app.core.hive_slash_commands.get_context_compressor', return_value=mock_compressor):
            # Test with preserve flags
            result = await command.execute(
                args=["--preserve-decisions", "--preserve-patterns"],
                context={"test": "context"}
            )
        
        assert result["success"] is True
        
        # Test with no-preserve flags
        with patch('app.core.hive_slash_commands.get_context_compressor', return_value=mock_compressor):
            result = await command.execute(
                args=["--no-preserve-decisions", "--no-preserve-patterns"],
                context={"test": "context"}
            )
        
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_execute_no_context_found(self, command):
        """Test compression when no context is available"""
        with patch.object(command, '_extract_context') as mock_extract:
            mock_extract.return_value = ("", {})
            
            result = await command.execute(
                args=[],
                context={}
            )
        
        assert result["success"] is False
        assert "No context found to compress" in result["error"]
        assert "message" in result
    
    @pytest.mark.asyncio
    async def test_execute_compression_failure(self, command):
        """Test compression execution when compression fails"""
        mock_compressor = AsyncMock()
        mock_compressor.compress_conversation.side_effect = Exception("Compression failed")
        
        with patch('app.core.hive_slash_commands.get_context_compressor', return_value=mock_compressor):
            result = await command.execute(
                args=[],
                context={"test": "context"}
            )
        
        assert result["success"] is False
        assert "Compression failed" in result["error"]
        assert result["message"] == "Context compression failed"
    
    @pytest.mark.asyncio
    async def test_extract_context_with_session_id(self, command):
        """Test context extraction with session ID"""
        session_id = "test-session-123"
        
        with patch.object(command, '_extract_session_context') as mock_extract:
            mock_extract.return_value = {
                "content": "Session content",
                "metadata": {"source": "session"}
            }
            
            content, metadata = await command._extract_context(session_id, {})
        
        assert content == "Session content"
        assert metadata["source"] == "session"
        assert metadata["session_id"] == session_id
        mock_extract.assert_called_once_with(session_id)
    
    @pytest.mark.asyncio
    async def test_extract_context_from_current_context(self, command):
        """Test context extraction from current context"""
        context = {
            "conversation_history": "User: Hello\nAgent: Hi there!"
        }
        
        content, metadata = await command._extract_context(None, context)
        
        assert "User: Hello" in content
        assert metadata["source"] == "current_context"
    
    @pytest.mark.asyncio
    async def test_extract_context_fallback(self, command):
        """Test context extraction fallback behavior"""
        context = {"random_data": "some data"}
        
        content, metadata = await command._extract_context(None, context)
        
        assert "some data" in content
        assert metadata["source"] == "full_context"
    
    @pytest.mark.asyncio
    async def test_extract_context_empty(self, command):
        """Test context extraction when no context available"""
        content, metadata = await command._extract_context(None, {})
        
        assert content == "No conversation context available for compression."
        assert "extraction_time" in metadata
    
    @pytest.mark.asyncio
    async def test_extract_session_context_success(self, command):
        """Test successful session context extraction"""
        session_id = "test-session-123"
        
        # Mock session object
        mock_session = Mock()
        mock_session.description = "Test session description"
        mock_session.objectives = ["Objective 1", "Objective 2"]
        mock_session.shared_context = {"key": "value"}
        mock_session.state = {"status": "active"}
        mock_session.name = "Test Session"
        mock_session.session_type.value = "development"
        mock_session.status.value = "active"
        mock_session.created_at = datetime.utcnow()
        mock_session.last_activity = datetime.utcnow()
        
        # Mock database session
        mock_db_session = AsyncMock()
        mock_db_session.get.return_value = mock_session
        
        with patch('app.core.hive_slash_commands.get_db_session') as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_db_session
            
            result = await command._extract_session_context(session_id)
        
        assert result is not None
        assert "Session Description" in result["content"]
        assert "Objective 1" in result["content"]
        assert result["metadata"]["session_name"] == "Test Session"
        assert result["metadata"]["session_type"] == "development"
    
    @pytest.mark.asyncio
    async def test_extract_session_context_not_found(self, command):
        """Test session context extraction when session not found"""
        session_id = "nonexistent-session"
        
        # Mock database session returning None
        mock_db_session = AsyncMock()
        mock_db_session.get.return_value = None
        
        with patch('app.core.hive_slash_commands.get_db_session') as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_db_session
            
            result = await command._extract_session_context(session_id)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_extract_session_context_database_error(self, command):
        """Test session context extraction with database error"""
        session_id = "test-session-123"
        
        with patch('app.core.hive_slash_commands.get_db_session') as mock_get_db:
            mock_get_db.side_effect = Exception("Database error")
            
            result = await command._extract_session_context(session_id)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_store_compressed_context_success(self, command):
        """Test successful storage of compressed context"""
        session_id = "test-session-123"
        
        # Mock compressed result
        mock_result = Mock()
        mock_result.to_dict.return_value = {"summary": "test"}
        mock_result.original_token_count = 1000
        mock_result.compressed_token_count = 400
        mock_result.compression_ratio = 0.6
        
        # Mock session object
        mock_session = Mock()
        mock_session.update_shared_context = Mock()
        
        # Mock database session
        mock_db_session = AsyncMock()
        mock_db_session.get.return_value = mock_session
        mock_db_session.commit = AsyncMock()
        
        with patch('app.core.hive_slash_commands.get_db_session') as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_db_session
            
            await command._store_compressed_context(session_id, mock_result)
        
        # Verify session was updated
        assert mock_session.update_shared_context.call_count == 2
        mock_db_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_store_compressed_context_session_not_found(self, command):
        """Test storing compressed context when session not found"""
        session_id = "nonexistent-session"
        mock_result = Mock()
        
        # Mock database session returning None
        mock_db_session = AsyncMock()
        mock_db_session.get.return_value = None
        
        with patch('app.core.hive_slash_commands.get_db_session') as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_db_session
            
            # Should not raise exception
            await command._store_compressed_context(session_id, mock_result)
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, command, mock_compressor):
        """Test that compression performance is tracked"""
        with patch('app.core.hive_slash_commands.get_context_compressor', return_value=mock_compressor):
            result = await command.execute(
                args=[],
                context={"test": "context"}
            )
        
        assert result["success"] is True
        assert "compression_time_seconds" in result
        assert isinstance(result["compression_time_seconds"], float)
        assert result["compression_time_seconds"] >= 0
        assert "performance_met" in result
        assert isinstance(result["performance_met"], bool)
        
        # Performance should be met if under 15 seconds
        if result["compression_time_seconds"] < 15.0:
            assert result["performance_met"] is True


class TestHiveCompactCommandIntegration:
    """Integration tests for HiveCompactCommand within the command registry"""
    
    @pytest.fixture
    def registry(self):
        """Create command registry with HiveCompactCommand"""
        registry = HiveSlashCommandRegistry()
        return registry
    
    def test_compact_command_registered(self, registry):
        """Test that compact command is registered in registry"""
        compact_command = registry.get_command("compact")
        assert compact_command is not None
        assert isinstance(compact_command, HiveCompactCommand)
    
    @pytest.mark.asyncio
    async def test_execute_via_registry(self, registry):
        """Test executing compact command via registry"""
        with patch('app.core.hive_slash_commands.get_context_compressor') as mock_get_compressor:
            mock_compressor = AsyncMock()
            mock_result = Mock()
            mock_result.summary = "Test summary"
            mock_result.key_insights = []
            mock_result.decisions_made = []
            mock_result.patterns_identified = []
            mock_result.importance_score = 0.5
            mock_result.compression_ratio = 0.4
            mock_result.original_token_count = 100
            mock_result.compressed_token_count = 60
            mock_compressor.compress_conversation.return_value = mock_result
            mock_get_compressor.return_value = mock_compressor
            
            result = await registry.execute_command(
                "/hive:compact --level=standard",
                context={"test": "context"}
            )
        
        assert result["success"] is True
        assert result["compression_level"] == "standard"
    
    @pytest.mark.asyncio
    async def test_execute_hive_command_function(self):
        """Test the execute_hive_command convenience function"""
        with patch('app.core.hive_slash_commands.get_context_compressor') as mock_get_compressor:
            mock_compressor = AsyncMock()
            mock_result = Mock()
            mock_result.summary = "Test summary"
            mock_result.key_insights = []
            mock_result.decisions_made = []
            mock_result.patterns_identified = []
            mock_result.importance_score = 0.5
            mock_result.compression_ratio = 0.4
            mock_result.original_token_count = 100
            mock_result.compressed_token_count = 60
            mock_compressor.compress_conversation.return_value = mock_result
            mock_get_compressor.return_value = mock_compressor
            
            result = await execute_hive_command(
                "/hive:compact session-123 --level=light",
                context={"test": "context"}
            )
        
        assert result["success"] is True
    
    @pytest.mark.asyncio
    async def test_invalid_command_format(self, registry):
        """Test handling of invalid command format"""
        result = await registry.execute_command(
            "compact --level=standard",  # Missing /hive: prefix
            context={}
        )
        
        assert result["success"] is False
        assert "Invalid hive command format" in result["error"]
    
    @pytest.mark.asyncio
    async def test_unknown_command(self, registry):
        """Test handling of unknown command"""
        result = await registry.execute_command(
            "/hive:unknown_command",
            context={}
        )
        
        assert result["success"] is False
        assert "Unknown command" in result["error"]
        assert "available_commands" in result


if __name__ == "__main__":
    pytest.main([__file__])