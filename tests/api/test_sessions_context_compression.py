"""
API tests for Context Compression endpoints
Tests the REST API endpoints for session context compression
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from httpx import AsyncClient

from app.main import app


class TestSessionsContextCompressionAPI:
    """Test suite for session context compression API endpoints"""
    
    @pytest.fixture
    async def client(self):
        """Create test HTTP client"""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.fixture
    def mock_session(self):
        """Mock session object for testing"""
        mock_session = Mock()
        mock_session.id = "test-session-123"
        mock_session.name = "Test Session"
        mock_session.description = "A test session for API testing"
        mock_session.objectives = ["Test objective 1", "Test objective 2"]
        mock_session.shared_context = {"project": "test", "phase": "development"}
        mock_session.state = {"status": "active"}
        mock_session.session_type.value = "development"
        mock_session.status.value = "active"
        mock_session.created_at = datetime.utcnow()
        mock_session.last_activity = datetime.utcnow()
        mock_session.update_shared_context = Mock()
        mock_session.get_shared_context = Mock()
        return mock_session
    
    @pytest.fixture
    def mock_compression_result(self):
        """Mock compression result for testing"""
        result = Mock()
        result.summary = "Compressed summary of the test session"
        result.key_insights = ["Insight 1", "Insight 2"]
        result.decisions_made = ["Decision 1", "Decision 2"]
        result.patterns_identified = ["Pattern 1", "Pattern 2"]
        result.importance_score = 0.8
        result.compression_ratio = 0.6
        result.original_token_count = 1000
        result.compressed_token_count = 400
        return result
    
    @pytest.mark.asyncio
    async def test_compact_session_success(self, client, mock_session, mock_compression_result):
        """Test successful session compression"""
        session_id = "test-session-123"
        
        # Mock database and compression
        mock_db_session = AsyncMock()
        mock_db_session.get.return_value = mock_session
        mock_db_session.commit = AsyncMock()
        
        with patch('app.api.v1.sessions.get_hive_command_registry') as mock_registry:
            mock_compact_command = AsyncMock()
            mock_compact_command.execute.return_value = {
                "success": True,
                "session_id": session_id,
                "compression_level": "standard",
                "original_tokens": 1000,
                "compressed_tokens": 400,
                "compression_ratio": 0.6,
                "tokens_saved": 600,
                "compression_time_seconds": 2.5,
                "summary": mock_compression_result.summary,
                "key_insights": mock_compression_result.key_insights,
                "decisions_made": mock_compression_result.decisions_made,
                "patterns_identified": mock_compression_result.patterns_identified,
                "importance_score": mock_compression_result.importance_score,
                "message": "Context compression completed",
                "performance_met": True,
                "timestamp": "2024-01-01T12:00:00Z"
            }
            
            mock_registry_instance = Mock()
            mock_registry_instance.get_command.return_value = mock_compact_command
            mock_registry.return_value = mock_registry_instance
            
            response = await client.post(
                f"/api/v1/sessions/{session_id}/compact",
                json={
                    "compression_level": "standard",
                    "preserve_decisions": True,
                    "preserve_patterns": True
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["session_id"] == session_id
        assert data["compression_level"] == "standard"
        assert data["original_tokens"] == 1000
        assert data["compressed_tokens"] == 400
        assert data["compression_ratio"] == 0.6
        assert data["tokens_saved"] == 600
        assert data["summary"] == mock_compression_result.summary
        assert len(data["key_insights"]) == 2
        assert len(data["decisions_made"]) == 2
        assert len(data["patterns_identified"]) == 2
        assert data["importance_score"] == 0.8
        assert data["performance_met"] is True
    
    @pytest.mark.asyncio
    async def test_compact_session_with_target_tokens(self, client, mock_session, mock_compression_result):
        """Test session compression with target token count"""
        session_id = "test-session-123"
        target_tokens = 300
        
        with patch('app.api.v1.sessions.get_hive_command_registry') as mock_registry:
            mock_compact_command = AsyncMock()
            mock_compact_command.execute.return_value = {
                "success": True,
                "session_id": session_id,
                "compression_level": "aggressive",
                "original_tokens": 1000,
                "compressed_tokens": 280,
                "compression_ratio": 0.72,
                "tokens_saved": 720,
                "compression_time_seconds": 3.2,
                "summary": "Aggressively compressed summary",
                "key_insights": ["Key insight"],
                "decisions_made": ["Key decision"],
                "patterns_identified": ["Key pattern"],
                "importance_score": 0.9,
                "message": "Context compression completed",
                "performance_met": True,
                "timestamp": "2024-01-01T12:00:00Z"
            }
            
            mock_registry_instance = Mock()
            mock_registry_instance.get_command.return_value = mock_compact_command
            mock_registry.return_value = mock_registry_instance
            
            response = await client.post(
                f"/api/v1/sessions/{session_id}/compact",
                json={
                    "compression_level": "standard",
                    "target_tokens": target_tokens,
                    "preserve_decisions": True,
                    "preserve_patterns": False
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["compressed_tokens"] <= target_tokens * 1.1  # Allow some margin
        
        # Verify command was called with correct arguments
        call_args = mock_compact_command.execute.call_args
        args = call_args[1]["args"]
        assert f"--target-tokens={target_tokens}" in args
        assert "--no-preserve-patterns" in args
    
    @pytest.mark.asyncio
    async def test_compact_session_invalid_session_id_format(self, client):
        """Test compression with invalid session ID format"""
        invalid_session_id = "not-a-valid-uuid"
        
        response = await client.post(
            f"/api/v1/sessions/{invalid_session_id}/compact",
            json={"compression_level": "standard"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid session_id format" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_compact_session_not_found(self, client):
        """Test compression when session doesn't exist"""
        session_id = "00000000-0000-0000-0000-000000000000"
        
        with patch('app.api.v1.sessions.get_hive_command_registry') as mock_registry:
            mock_compact_command = AsyncMock()
            mock_compact_command.execute.return_value = {
                "success": False,
                "error": "Session test-session-123 not found"
            }
            
            mock_registry_instance = Mock()
            mock_registry_instance.get_command.return_value = mock_compact_command
            mock_registry.return_value = mock_registry_instance
            
            response = await client.post(
                f"/api/v1/sessions/{session_id}/compact",
                json={"compression_level": "standard"}
            )
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_compact_session_no_context(self, client):
        """Test compression when session has no context"""
        session_id = "test-session-456"
        
        with patch('app.api.v1.sessions.get_hive_command_registry') as mock_registry:
            mock_compact_command = AsyncMock()
            mock_compact_command.execute.return_value = {
                "success": False,
                "error": "No context found to compress"
            }
            
            mock_registry_instance = Mock()
            mock_registry_instance.get_command.return_value = mock_compact_command
            mock_registry.return_value = mock_registry_instance
            
            response = await client.post(
                f"/api/v1/sessions/{session_id}/compact",
                json={"compression_level": "standard"}
            )
        
        assert response.status_code == 400
        data = response.json()
        assert "No conversation context found" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_compact_session_compression_service_unavailable(self, client):
        """Test compression when service is unavailable"""
        session_id = "test-session-789"
        
        with patch('app.api.v1.sessions.get_hive_command_registry') as mock_registry:
            mock_registry_instance = Mock()
            mock_registry_instance.get_command.return_value = None  # Service unavailable
            mock_registry.return_value = mock_registry_instance
            
            response = await client.post(
                f"/api/v1/sessions/{session_id}/compact",
                json={"compression_level": "standard"}
            )
        
        assert response.status_code == 500
        data = response.json()
        assert "Context compression service not available" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_compact_session_internal_error(self, client):
        """Test compression when internal error occurs"""
        session_id = "test-session-error"
        
        with patch('app.api.v1.sessions.get_hive_command_registry') as mock_registry:
            mock_compact_command = AsyncMock()
            mock_compact_command.execute.return_value = {
                "success": False,
                "error": "Internal compression error"
            }
            
            mock_registry_instance = Mock()
            mock_registry_instance.get_command.return_value = mock_compact_command
            mock_registry.return_value = mock_registry_instance
            
            response = await client.post(
                f"/api/v1/sessions/{session_id}/compact",
                json={"compression_level": "standard"}
            )
        
        assert response.status_code == 500
        data = response.json()
        assert "Internal compression error" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_compact_session_invalid_compression_level(self, client):
        """Test compression with invalid compression level"""
        session_id = "test-session-123"
        
        response = await client.post(
            f"/api/v1/sessions/{session_id}/compact",
            json={"compression_level": "invalid_level"}
        )
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.asyncio
    async def test_compact_session_invalid_target_tokens(self, client):
        """Test compression with invalid target tokens"""
        session_id = "test-session-123"
        
        response = await client.post(
            f"/api/v1/sessions/{session_id}/compact",
            json={
                "compression_level": "standard",
                "target_tokens": -100  # Invalid negative value
            }
        )
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.asyncio
    async def test_compact_session_missing_request_body(self, client):
        """Test compression with missing request body"""
        session_id = "test-session-123"
        
        with patch('app.api.v1.sessions.get_hive_command_registry') as mock_registry:
            mock_compact_command = AsyncMock()
            mock_compact_command.execute.return_value = {
                "success": True,
                "session_id": session_id,
                "compression_level": "standard",  # Should use default
                "original_tokens": 500,
                "compressed_tokens": 250,
                "compression_ratio": 0.5,
                "tokens_saved": 250,
                "compression_time_seconds": 1.8,
                "summary": "Default compression",
                "key_insights": [],
                "decisions_made": [],
                "patterns_identified": [],
                "importance_score": 0.5,
                "message": "Context compression completed",
                "performance_met": True,
                "timestamp": "2024-01-01T12:00:00Z"
            }
            
            mock_registry_instance = Mock()
            mock_registry_instance.get_command.return_value = mock_compact_command
            mock_registry.return_value = mock_registry_instance
            
            response = await client.post(
                f"/api/v1/sessions/{session_id}/compact",
                json={}  # Empty request body, should use defaults
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["compression_level"] == "standard"  # Default level
    
    @pytest.mark.asyncio
    async def test_get_compression_status_success(self, client, mock_session):
        """Test successful retrieval of compression status"""
        session_id = "test-session-123"
        
        # Mock compressed context data
        compressed_context = {
            "summary": "Test summary",
            "key_insights": ["Insight 1", "Insight 2"],
            "decisions_made": ["Decision 1"],
            "patterns_identified": ["Pattern 1"],
            "importance_score": 0.8,
            "compressed_at": "2024-01-01T12:00:00Z"
        }
        
        compression_history = {
            "compressed_at": "2024-01-01T12:00:00Z",
            "original_tokens": 1000,
            "compressed_tokens": 400,
            "compression_ratio": 0.6
        }
        
        mock_session.get_shared_context.side_effect = lambda key: {
            "compressed_context": compressed_context,
            "compression_history": compression_history
        }.get(key)
        
        mock_db_session = AsyncMock()
        mock_db_session.get.return_value = mock_session
        
        with patch('app.api.v1.sessions.get_db_session') as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_db_session
            
            response = await client.get(f"/api/v1/sessions/{session_id}/compact/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == session_id
        assert data["has_compressed_context"] is True
        assert data["compression_history"] == compression_history
        assert data["compressed_data"]["summary_length"] == len(compressed_context["summary"])
        assert data["compressed_data"]["key_insights_count"] == 2
        assert data["compressed_data"]["decisions_count"] == 1
        assert data["compressed_data"]["patterns_count"] == 1
        assert data["compressed_data"]["importance_score"] == 0.8
        assert data["session_info"]["name"] == mock_session.name
        assert data["session_info"]["type"] == mock_session.session_type.value
        assert data["session_info"]["status"] == mock_session.status.value
    
    @pytest.mark.asyncio
    async def test_get_compression_status_no_data(self, client, mock_session):
        """Test compression status when no compression data exists"""
        session_id = "test-session-456"
        
        mock_session.get_shared_context.return_value = None
        
        mock_db_session = AsyncMock()
        mock_db_session.get.return_value = mock_session
        
        with patch('app.api.v1.sessions.get_db_session') as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_db_session
            
            response = await client.get(f"/api/v1/sessions/{session_id}/compact/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == session_id
        assert data["has_compressed_context"] is False
        assert "No compression data available" in data["message"]
    
    @pytest.mark.asyncio
    async def test_get_compression_status_session_not_found(self, client):
        """Test compression status when session doesn't exist"""
        session_id = "00000000-0000-0000-0000-000000000000"
        
        mock_db_session = AsyncMock()
        mock_db_session.get.return_value = None
        
        with patch('app.api.v1.sessions.get_db_session') as mock_get_db:
            mock_get_db.return_value.__aenter__.return_value = mock_db_session
            
            response = await client.get(f"/api/v1/sessions/{session_id}/compact/status")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_get_compression_status_invalid_session_id(self, client):
        """Test compression status with invalid session ID format"""
        invalid_session_id = "not-a-valid-uuid"
        
        response = await client.get(f"/api/v1/sessions/{invalid_session_id}/compact/status")
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid session_id format" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_get_compression_status_database_error(self, client):
        """Test compression status when database error occurs"""
        session_id = "test-session-789"
        
        with patch('app.api.v1.sessions.get_db_session') as mock_get_db:
            mock_get_db.side_effect = Exception("Database connection failed")
            
            response = await client.get(f"/api/v1/sessions/{session_id}/compact/status")
        
        assert response.status_code == 500
        data = response.json()
        assert "Error retrieving compression status" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_compression_request_validation(self, client):
        """Test comprehensive request validation"""
        session_id = "test-session-123"
        
        # Test all valid compression levels
        valid_levels = ["light", "standard", "aggressive"]
        
        for level in valid_levels:
            with patch('app.api.v1.sessions.get_hive_command_registry') as mock_registry:
                mock_compact_command = AsyncMock()
                mock_compact_command.execute.return_value = {
                    "success": True,
                    "session_id": session_id,
                    "compression_level": level,
                    "original_tokens": 100,
                    "compressed_tokens": 50,
                    "compression_ratio": 0.5,
                    "tokens_saved": 50,
                    "compression_time_seconds": 1.0,
                    "summary": "Test",
                    "key_insights": [],
                    "decisions_made": [],
                    "patterns_identified": [],
                    "importance_score": 0.5,
                    "message": "Test",
                    "performance_met": True,
                    "timestamp": "2024-01-01T12:00:00Z"
                }
                
                mock_registry_instance = Mock()
                mock_registry_instance.get_command.return_value = mock_compact_command
                mock_registry.return_value = mock_registry_instance
                
                response = await client.post(
                    f"/api/v1/sessions/{session_id}/compact",
                    json={"compression_level": level}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["compression_level"] == level
    
    @pytest.mark.asyncio
    async def test_compression_request_all_options(self, client):
        """Test compression request with all available options"""
        session_id = "test-session-comprehensive"
        
        request_data = {
            "compression_level": "aggressive",
            "target_tokens": 200,
            "preserve_decisions": False,
            "preserve_patterns": True
        }
        
        with patch('app.api.v1.sessions.get_hive_command_registry') as mock_registry:
            mock_compact_command = AsyncMock()
            mock_compact_command.execute.return_value = {
                "success": True,
                "session_id": session_id,
                "compression_level": "aggressive",
                "original_tokens": 800,
                "compressed_tokens": 180,
                "compression_ratio": 0.775,
                "tokens_saved": 620,
                "compression_time_seconds": 4.2,
                "summary": "Comprehensive test summary",
                "key_insights": ["Insight A", "Insight B"],
                "decisions_made": [],  # Should be empty since preserve_decisions=False
                "patterns_identified": ["Pattern X", "Pattern Y"],
                "importance_score": 0.9,
                "message": "Context compression completed",
                "performance_met": True,
                "timestamp": "2024-01-01T12:00:00Z"
            }
            
            mock_registry_instance = Mock()
            mock_registry_instance.get_command.return_value = mock_compact_command
            mock_registry.return_value = mock_registry_instance
            
            response = await client.post(
                f"/api/v1/sessions/{session_id}/compact",
                json=request_data
            )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["compression_level"] == "aggressive"
        assert data["compressed_tokens"] <= 220  # Close to target
        assert len(data["decisions_made"]) == 0  # Shouldn't preserve decisions
        assert len(data["patterns_identified"]) > 0  # Should preserve patterns
        
        # Verify correct arguments were passed
        call_args = mock_compact_command.execute.call_args
        args = call_args[1]["args"]
        assert "--level=aggressive" in args
        assert "--target-tokens=200" in args
        assert "--no-preserve-decisions" in args
        assert "--preserve-patterns" in args


if __name__ == "__main__":
    pytest.main([__file__])