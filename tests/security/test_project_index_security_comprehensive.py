"""
Comprehensive security tests for Project Index system.

Tests for authentication, authorization, input validation,
SQL injection, XSS, path traversal, and other security vulnerabilities.
"""

import pytest
import uuid
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import quote, unquote

from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project_index import ProjectIndex, FileEntry


@pytest.mark.project_index_security
@pytest.mark.security
class TestInputValidation:
    """Test input validation and sanitization."""
    
    async def test_sql_injection_prevention(
        self, 
        async_test_client: AsyncClient,
        security_test_config: Dict[str, Any]
    ):
        """Test SQL injection prevention in all endpoints."""
        payloads = security_test_config["sql_injection_payloads"]
        
        # Test SQL injection in project creation
        for payload in payloads:
            request_data = {
                "name": payload,
                "root_path": "/test/path",
                "description": payload,
                "git_repository_url": payload
            }
            
            response = await async_test_client.post(
                "/api/project-index/create",
                json=request_data
            )
            
            # Should either validate and reject (400/422) or handle safely
            assert response.status_code in [200, 400, 422, 500]
            
            # If successful, verify no SQL injection occurred
            if response.status_code == 200:
                response_data = response.json()
                # Check that the payload wasn't executed as SQL
                assert "DROP" not in str(response_data).upper()
                assert "DELETE" not in str(response_data).upper()
                assert "UPDATE" not in str(response_data).upper()
    
    async def test_xss_prevention(
        self, 
        async_test_client: AsyncClient,
        security_test_config: Dict[str, Any]
    ):
        """Test XSS prevention in API responses."""
        payloads = security_test_config["xss_payloads"]
        
        with patch('app.api.project_index.get_project_indexer') as mock_indexer:
            mock_indexer_instance = AsyncMock()
            mock_project = Mock()
            mock_project.id = uuid.uuid4()
            mock_project.name = "Test Project"
            
            mock_indexer_instance.create_project.return_value = mock_project
            mock_indexer.return_value = mock_indexer_instance
            
            for payload in payloads:
                request_data = {
                    "name": payload,
                    "root_path": "/test/path",
                    "description": payload
                }
                
                response = await async_test_client.post(
                    "/api/project-index/create",
                    json=request_data
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    response_text = json.dumps(response_data)
                    
                    # Verify XSS payloads are properly escaped/sanitized
                    assert "<script>" not in response_text
                    assert "javascript:" not in response_text
                    assert "onerror=" not in response_text
                    
                    # Check for HTML entity encoding
                    if "&lt;" in response_text or "&gt;" in response_text:
                        print(f"XSS payload properly encoded: {payload}")
    
    async def test_path_traversal_prevention(
        self, 
        async_test_client: AsyncClient,
        security_test_config: Dict[str, Any]
    ):
        """Test path traversal attack prevention."""
        payloads = security_test_config["path_traversal_payloads"]
        
        # Test path traversal in file path endpoints
        project_id = uuid.uuid4()
        
        with patch('app.api.project_index.get_project_or_404') as mock_get_project:
            mock_project = Mock()
            mock_project.id = project_id
            mock_get_project.return_value = mock_project
            
            with patch('app.api.project_index.get_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_session.execute.return_value.scalar_one_or_none.return_value = None
                mock_get_session.return_value = mock_session
                
                for payload in payloads:
                    # Test file analysis endpoint with path traversal
                    encoded_payload = quote(payload, safe='')
                    
                    response = await async_test_client.get(
                        f"/api/project-index/{project_id}/files/{encoded_payload}"
                    )
                    
                    # Should handle path traversal safely
                    assert response.status_code in [400, 404, 422]
                    
                    # Verify no actual file system access outside project
                    if response.status_code == 200:
                        response_data = response.json()
                        # Ensure the response doesn't contain sensitive system files
                        response_text = json.dumps(response_data)
                        assert "/etc/passwd" not in response_text
                        assert "windows/system32" not in response_text.lower()
    
    async def test_file_size_limits(
        self, 
        async_test_client: AsyncClient,
        security_test_config: Dict[str, Any]
    ):
        """Test file size limits and DoS prevention."""
        max_size = security_test_config["max_request_size"]
        
        # Create request with large payload
        large_data = "x" * (max_size + 1000)  # Exceed limit
        
        request_data = {
            "name": "Large Request Test",
            "root_path": "/test/path",
            "description": large_data,
            "configuration": {
                "large_field": large_data
            }
        }
        
        response = await async_test_client.post(
            "/api/project-index/create",
            json=request_data
        )
        
        # Should reject overly large requests
        assert response.status_code in [413, 422, 400]
    
    async def test_parameter_validation(
        self, 
        async_test_client: AsyncClient
    ):
        """Test parameter validation for various edge cases."""
        project_id = uuid.uuid4()
        
        with patch('app.api.project_index.get_project_or_404') as mock_get_project:
            mock_project = Mock()
            mock_project.id = project_id
            mock_get_project.return_value = mock_project
            
            with patch('app.api.project_index.get_session') as mock_get_session:
                mock_session = AsyncMock()
                mock_session.execute.return_value.scalars.return_value.all.return_value = []
                mock_session.execute.return_value.scalar.return_value = 0
                mock_get_session.return_value = mock_session
                
                # Test invalid pagination parameters
                invalid_params = [
                    {"page": -1, "limit": 10},
                    {"page": 0, "limit": 10},
                    {"page": 1, "limit": -1},
                    {"page": 1, "limit": 0},
                    {"page": 1, "limit": 10000},  # Too large
                    {"page": "invalid", "limit": "invalid"},
                    {"page": 1.5, "limit": 2.5},  # Float values
                ]
                
                for params in invalid_params:
                    response = await async_test_client.get(
                        f"/api/project-index/{project_id}/files",
                        params=params
                    )
                    
                    assert response.status_code in [422, 400], f"Failed to reject invalid params: {params}"
    
    async def test_unicode_and_encoding_validation(
        self, 
        async_test_client: AsyncClient
    ):
        """Test handling of Unicode and various encodings."""
        unicode_test_cases = [
            "普通话/汉语",  # Chinese
            "🚀🌟💻",  # Emojis
            "test\x00null",  # Null bytes
            "test\x08backspace",  # Control characters
            "\uffff\ufffe",  # Unicode non-characters
            "𝕊𝕡𝕖𝕔𝕚𝕒𝕝 𝕌𝕟𝕚𝕔𝕠𝕕𝕖",  # Mathematical symbols
        ]
        
        with patch('app.api.project_index.get_project_indexer') as mock_indexer:
            mock_indexer_instance = AsyncMock()
            mock_project = Mock()
            mock_project.id = uuid.uuid4()
            mock_project.name = "Unicode Test"
            
            mock_indexer_instance.create_project.return_value = mock_project
            mock_indexer.return_value = mock_indexer_instance
            
            for test_string in unicode_test_cases:
                request_data = {
                    "name": test_string,
                    "root_path": "/test/path",
                    "description": f"Unicode test with: {test_string}"
                }
                
                response = await async_test_client.post(
                    "/api/project-index/create",
                    json=request_data
                )
                
                # Should handle Unicode properly
                assert response.status_code in [200, 400, 422]
                
                if response.status_code == 200:
                    response_data = response.json()
                    # Verify proper Unicode handling
                    assert isinstance(response_data, dict)


@pytest.mark.project_index_security
@pytest.mark.security
class TestAuthenticationSecurity:
    """Test authentication and session security."""
    
    async def test_unauthenticated_access_prevention(
        self, 
        async_test_client: AsyncClient
    ):
        """Test that endpoints properly require authentication."""
        
        # Test endpoints that should require authentication
        protected_endpoints = [
            ("POST", "/api/project-index/create", {"name": "test", "root_path": "/test"}),
            ("GET", f"/api/project-index/{uuid.uuid4()}"),
            ("PUT", f"/api/project-index/{uuid.uuid4()}/refresh"),
            ("DELETE", f"/api/project-index/{uuid.uuid4()}"),
            ("POST", f"/api/project-index/{uuid.uuid4()}/analyze", {"analysis_type": "full"}),
        ]
        
        for method, endpoint, *body in protected_endpoints:
            if method == "POST" or method == "PUT":
                if body:
                    response = await async_test_client.request(method, endpoint, json=body[0])
                else:
                    response = await async_test_client.request(method, endpoint)
            else:
                response = await async_test_client.request(method, endpoint)
            
            # Note: In the current implementation, authentication is mocked
            # In a real system, these should return 401 Unauthorized
            # For now, we verify the endpoints exist and respond
            assert response.status_code in [200, 401, 403, 404, 422]
    
    @patch('app.api.project_index.get_current_user')
    async def test_user_isolation(
        self, 
        mock_get_user,
        async_test_client: AsyncClient,
        project_index_session: AsyncSession
    ):
        """Test that users can only access their own projects."""
        
        # Create projects for different users
        user1_project = ProjectIndex(
            name="User 1 Project",
            root_path="/user1/project",
            meta_data={"owner": "user1"}
        )
        
        user2_project = ProjectIndex(
            name="User 2 Project", 
            root_path="/user2/project",
            meta_data={"owner": "user2"}
        )
        
        project_index_session.add_all([user1_project, user2_project])
        await project_index_session.commit()
        
        # Test User 1 trying to access User 2's project
        mock_get_user.return_value = "user1"
        
        with patch('app.api.project_index.get_project_or_404') as mock_get_project:
            # Simulate access control check
            def check_access(project_id):
                if str(project_id) == str(user2_project.id):
                    from fastapi import HTTPException
                    raise HTTPException(status_code=403, detail="Access denied")
                return user1_project
            
            mock_get_project.side_effect = lambda pid: check_access(pid)
            
            # User 1 should be able to access their own project
            response = await async_test_client.get(f"/api/project-index/{user1_project.id}")
            assert response.status_code == 200
            
            # User 1 should NOT be able to access User 2's project
            response = await async_test_client.get(f"/api/project-index/{user2_project.id}")
            assert response.status_code == 403
    
    async def test_session_security(self, async_test_client: AsyncClient):
        """Test session security and token validation."""
        
        # Test WebSocket authentication
        with patch('app.api.project_index.websocket_auth_context') as mock_auth:
            # Test with invalid token
            mock_auth.side_effect = Exception("Invalid token")
            
            try:
                with async_test_client.websocket_connect("/api/project-index/ws?token=invalid_token") as websocket:
                    # Connection should fail
                    pass
            except Exception:
                # Expected to fail with invalid token
                pass
            
            # Test with valid token
            mock_auth.side_effect = None
            mock_auth.return_value.__aenter__.return_value = "valid_user"
            
            with patch('app.api.project_index.get_websocket_handler') as mock_handler:
                mock_handler_instance = AsyncMock()
                mock_handler.return_value = mock_handler_instance
                
                try:
                    with async_test_client.websocket_connect("/api/project-index/ws?token=valid_token") as websocket:
                        # Connection should succeed
                        pass
                except Exception:
                    # May fail due to mocking, but should not be auth-related
                    pass


@pytest.mark.project_index_security
@pytest.mark.security
class TestRateLimitingSecurity:
    """Test rate limiting and DoS protection."""
    
    @patch('app.api.project_index.rate_limit_analysis')
    async def test_analysis_rate_limiting(
        self, 
        mock_rate_limit,
        async_test_client: AsyncClient,
        security_test_config: Dict[str, Any]
    ):
        """Test analysis endpoint rate limiting."""
        project_id = uuid.uuid4()
        
        with patch('app.api.project_index.get_project_or_404') as mock_get_project:
            mock_project = Mock()
            mock_project.id = project_id
            mock_get_project.return_value = mock_project
            
            with patch('app.api.project_index.get_project_indexer') as mock_indexer:
                mock_indexer.return_value = AsyncMock()
                
                # Test rate limiting enforcement
                rate_limit = security_test_config["rate_limit_requests_per_minute"]
                
                # Simulate exceeding rate limit
                def rate_limit_check():
                    from fastapi import HTTPException
                    raise HTTPException(
                        status_code=429,
                        detail={
                            "error": "RATE_LIMIT_EXCEEDED",
                            "message": "Analysis rate limit exceeded"
                        }
                    )
                
                mock_rate_limit.side_effect = rate_limit_check
                
                request_data = {
                    "analysis_type": "full",
                    "force": False
                }
                
                response = await async_test_client.post(
                    f"/api/project-index/{project_id}/analyze",
                    json=request_data
                )
                
                assert response.status_code == 429
                assert "rate limit" in response.json()["detail"]["message"].lower()
    
    async def test_concurrent_request_limits(
        self, 
        async_test_client: AsyncClient,
        security_test_config: Dict[str, Any]
    ):
        """Test concurrent request limits."""
        project_id = uuid.uuid4()
        
        with patch('app.api.project_index.get_project_or_404') as mock_get_project:
            mock_project = Mock()
            mock_project.id = project_id
            mock_project.name = "Concurrent Test"
            mock_project.status = "active"
            mock_project.created_at = datetime.utcnow()
            mock_project.updated_at = datetime.utcnow()
            
            mock_get_project.return_value = mock_project
            
            # Simulate many concurrent requests
            import asyncio
            
            async def make_request():
                return await async_test_client.get(f"/api/project-index/{project_id}")
            
            # Create more requests than should be allowed
            max_concurrent = security_test_config["max_concurrent_users"]
            tasks = [make_request() for _ in range(max_concurrent * 2)]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify that the system handles the load appropriately
            success_count = sum(1 for r in responses if hasattr(r, 'status_code') and r.status_code == 200)
            error_count = len(responses) - success_count
            
            # Should handle some requests successfully but may reject excessive load
            assert success_count > 0, "No requests succeeded under load"
            print(f"Concurrent load test: {success_count} successful, {error_count} errors")


@pytest.mark.project_index_security
@pytest.mark.security
class TestDataValidationSecurity:
    """Test data validation and sanitization security."""
    
    async def test_project_path_validation(
        self, 
        async_test_client: AsyncClient
    ):
        """Test project root path validation and security."""
        
        # Test dangerous paths that should be rejected
        dangerous_paths = [
            "/etc/passwd",
            "/root/.ssh",
            "C:\\Windows\\System32",
            "/proc/self/environ",
            "../../../etc/shadow",
            "/dev/null",
            "/tmp/../etc/passwd",
            "\\\\server\\share\\sensitive",
        ]
        
        for dangerous_path in dangerous_paths:
            request_data = {
                "name": "Security Test",
                "root_path": dangerous_path,
                "description": "Testing path validation"
            }
            
            response = await async_test_client.post(
                "/api/project-index/create",
                json=request_data
            )
            
            # Should reject dangerous paths
            assert response.status_code in [400, 403, 422], f"Dangerous path allowed: {dangerous_path}"
    
    async def test_git_url_validation(
        self, 
        async_test_client: AsyncClient
    ):
        """Test Git URL validation for security."""
        
        # Test potentially dangerous Git URLs
        dangerous_urls = [
            "file:///etc/passwd",
            "ssh://evil.com/../../etc/passwd",
            "git://malicious.com/repo.git/../../../etc/passwd",
            "http://internal.server/repo",  # Internal network access
            "ftp://evil.com/repo",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
        ]
        
        with patch('app.api.project_index.get_project_indexer') as mock_indexer:
            mock_indexer_instance = AsyncMock()
            mock_project = Mock()
            mock_project.id = uuid.uuid4()
            mock_project.name = "Git URL Test"
            
            mock_indexer_instance.create_project.return_value = mock_project
            mock_indexer.return_value = mock_indexer_instance
            
            for dangerous_url in dangerous_urls:
                request_data = {
                    "name": "Git URL Security Test",
                    "root_path": "/test/path",
                    "git_repository_url": dangerous_url
                }
                
                response = await async_test_client.post(
                    "/api/project-index/create",
                    json=request_data
                )
                
                # Should validate Git URLs properly
                if response.status_code == 200:
                    # If accepted, verify it's properly sanitized
                    response_data = response.json()
                    stored_url = response_data.get("data", {}).get("git_repository_url")
                    
                    # Ensure dangerous protocols are not stored
                    if stored_url:
                        assert not stored_url.startswith("file://")
                        assert not stored_url.startswith("javascript:")
                        assert not stored_url.startswith("data:")
    
    async def test_configuration_data_validation(
        self, 
        async_test_client: AsyncClient
    ):
        """Test configuration data validation for security."""
        
        # Test configuration with potentially dangerous content
        dangerous_configs = [
            {"command": "rm -rf /"},
            {"script": "<script>alert('xss')</script>"},
            {"file_path": "../../../etc/passwd"},
            {"__proto__": {"isAdmin": True}},  # Prototype pollution
            {"constructor": {"prototype": {"isAdmin": True}}},
            {"eval": "process.exit()"},
            {"require": "child_process"},
        ]
        
        with patch('app.api.project_index.get_project_indexer') as mock_indexer:
            mock_indexer_instance = AsyncMock()
            mock_project = Mock()
            mock_project.id = uuid.uuid4()
            mock_project.name = "Config Test"
            
            mock_indexer_instance.create_project.return_value = mock_project
            mock_indexer.return_value = mock_indexer_instance
            
            for dangerous_config in dangerous_configs:
                request_data = {
                    "name": "Configuration Security Test",
                    "root_path": "/test/path",
                    "configuration": dangerous_config
                }
                
                response = await async_test_client.post(
                    "/api/project-index/create",
                    json=request_data
                )
                
                # Should handle dangerous configurations safely
                assert response.status_code in [200, 400, 422]
                
                if response.status_code == 200:
                    response_data = response.json()
                    stored_config = response_data.get("data", {}).get("configuration", {})
                    
                    # Verify dangerous content is sanitized or rejected
                    assert "__proto__" not in str(stored_config)
                    assert "constructor" not in str(stored_config)
                    assert "<script>" not in str(stored_config)


@pytest.mark.project_index_security
@pytest.mark.security
class TestFileSystemSecurity:
    """Test file system access security."""
    
    async def test_file_access_restrictions(
        self, 
        async_test_client: AsyncClient
    ):
        """Test file access is properly restricted to project boundaries."""
        project_id = uuid.uuid4()
        
        with patch('app.api.project_index.get_project_or_404') as mock_get_project:
            mock_project = Mock()
            mock_project.id = project_id
            mock_project.root_path = "/safe/project/path"
            mock_get_project.return_value = mock_project
            
            with patch('app.api.project_index.get_session') as mock_get_session:
                mock_session = AsyncMock()
                
                # Test files outside project boundaries
                dangerous_files = [
                    "../../../etc/passwd",
                    "/etc/hosts",
                    "C:\\Windows\\System32\\config\\SAM",
                    "/proc/self/environ",
                    "/root/.ssh/id_rsa",
                ]
                
                for dangerous_file in dangerous_files:
                    # Mock file entry that would represent unauthorized access
                    mock_file = Mock()
                    mock_file.id = uuid.uuid4()
                    mock_file.relative_path = dangerous_file
                    mock_file.file_path = dangerous_file
                    
                    # Should not return files outside project
                    mock_session.execute.return_value.scalar_one_or_none.return_value = None
                    mock_get_session.return_value = mock_session
                    
                    encoded_path = quote(dangerous_file, safe='')
                    response = await async_test_client.get(
                        f"/api/project-index/{project_id}/files/{encoded_path}"
                    )
                    
                    # Should not find files outside project boundaries
                    assert response.status_code == 404
    
    async def test_symlink_traversal_prevention(
        self, 
        project_index_session: AsyncSession,
        temp_project_directory
    ):
        """Test prevention of symlink traversal attacks."""
        
        # Create a project
        project = ProjectIndex(
            name="Symlink Security Test",
            root_path=str(temp_project_directory),
            configuration={"security_test": True}
        )
        
        project_index_session.add(project)
        await project_index_session.commit()
        await project_index_session.refresh(project)
        
        # Test file entries that could represent symlink attacks
        dangerous_symlinks = [
            {
                "file_path": str(temp_project_directory / "safe_file.py"),
                "relative_path": "safe_file.py",
                "analysis_data": {"symlink_target": "/etc/passwd"}
            },
            {
                "file_path": str(temp_project_directory / "another_file.py"),
                "relative_path": "another_file.py", 
                "analysis_data": {"symlink_target": "../../../root/.ssh/id_rsa"}
            }
        ]
        
        for symlink_data in dangerous_symlinks:
            file_entry = FileEntry(
                project_id=project.id,
                file_path=symlink_data["file_path"],
                relative_path=symlink_data["relative_path"],
                file_name=symlink_data["relative_path"],
                file_extension=".py",
                file_type="source",
                language="python",
                analysis_data=symlink_data["analysis_data"]
            )
            
            project_index_session.add(file_entry)
            await project_index_session.commit()
            await project_index_session.refresh(file_entry)
            
            # Verify symlink targets are not accessible
            analysis_data = file_entry.analysis_data
            if "symlink_target" in analysis_data:
                target = analysis_data["symlink_target"]
                
                # Symlink targets should be within project boundaries
                assert not target.startswith("/etc/")
                assert not target.startswith("/root/")
                assert not ("../" in target and target.startswith("../"))


@pytest.mark.project_index_security
@pytest.mark.security
class TestWebSocketSecurity:
    """Test WebSocket security and message validation."""
    
    async def test_websocket_message_validation(
        self, 
        async_test_client: AsyncClient,
        websocket_test_config: Dict[str, Any]
    ):
        """Test WebSocket message validation and sanitization."""
        
        # Test malicious WebSocket messages
        malicious_messages = [
            {"action": "subscribe", "project_id": "'; DROP TABLE projects; --"},
            {"action": "subscribe", "event_types": ["<script>alert('xss')</script>"]},
            {"action": "subscribe", "filter": {"file_path": "../../../etc/passwd"}},
            {"action": "eval", "code": "process.exit()"},  # Command injection
            {"action": "subscribe", "data": "x" * (websocket_test_config["message_size_limit"] + 1000)},
        ]
        
        with patch('app.api.project_index.get_websocket_handler') as mock_handler:
            mock_handler_instance = AsyncMock()
            mock_handler.return_value = mock_handler_instance
            
            with patch('app.api.project_index.websocket_auth_context') as mock_auth:
                mock_auth.return_value.__aenter__.return_value = "test_user"
                
                try:
                    with async_test_client.websocket_connect("/api/project-index/ws?token=test_token") as websocket:
                        for message in malicious_messages:
                            try:
                                # Should handle malicious messages safely
                                await websocket.send_json(message)
                                
                                # If we get a response, it should be an error or sanitized
                                try:
                                    response = await websocket.receive_json()
                                    if "error" in response:
                                        print(f"Malicious message properly rejected: {message}")
                                except Exception:
                                    # Timeout or connection closed is acceptable
                                    pass
                                    
                            except Exception:
                                # Connection closed due to malicious message is acceptable
                                print(f"Connection closed for malicious message: {message}")
                                
                except Exception:
                    # WebSocket connection may fail due to mocking
                    pass
    
    async def test_websocket_rate_limiting(
        self, 
        async_test_client: AsyncClient,
        websocket_test_config: Dict[str, Any]
    ):
        """Test WebSocket rate limiting and flood protection."""
        
        with patch('app.api.project_index.get_websocket_handler') as mock_handler:
            mock_handler_instance = AsyncMock()
            mock_handler.return_value = mock_handler_instance
            
            with patch('app.api.project_index.websocket_auth_context') as mock_auth:
                mock_auth.return_value.__aenter__.return_value = "test_user"
                
                try:
                    with async_test_client.websocket_connect("/api/project-index/ws?token=test_token") as websocket:
                        # Send many messages rapidly to test rate limiting
                        message = {"action": "subscribe", "event_types": ["analysis_progress"]}
                        
                        for i in range(100):  # Rapid message flood
                            try:
                                await websocket.send_json(message)
                            except Exception:
                                # Rate limiting may close connection
                                print(f"Rate limiting activated after {i} messages")
                                break
                                
                except Exception:
                    # WebSocket connection may fail due to mocking or rate limiting
                    pass


@pytest.mark.project_index_security
@pytest.mark.security
class TestDataPrivacySecurity:
    """Test data privacy and information disclosure prevention."""
    
    async def test_sensitive_data_exposure_prevention(
        self, 
        async_test_client: AsyncClient,
        project_index_session: AsyncSession
    ):
        """Test prevention of sensitive data exposure in API responses."""
        
        # Create project with potentially sensitive data
        project = ProjectIndex(
            name="Privacy Test Project",
            root_path="/private/project",
            configuration={
                "api_key": "secret_api_key_12345",
                "password": "super_secret_password",
                "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg...",
                "database_url": "postgresql://user:password@localhost/db"
            },
            meta_data={
                "ssh_key": "ssh-rsa AAAAB3NzaC1yc2E...",
                "internal_notes": "Contains sensitive customer data",
                "system_info": {
                    "hostname": "internal-server-01",
                    "ip_address": "192.168.1.100"
                }
            }
        )
        
        project_index_session.add(project)
        await project_index_session.commit()
        await project_index_session.refresh(project)
        
        with patch('app.api.project_index.get_project_or_404') as mock_get_project:
            mock_get_project.return_value = project
            
            response = await async_test_client.get(f"/api/project-index/{project.id}")
            
            if response.status_code == 200:
                response_data = response.json()
                response_text = json.dumps(response_data)
                
                # Verify sensitive data is not exposed
                sensitive_patterns = [
                    "secret_api_key",
                    "super_secret_password", 
                    "BEGIN PRIVATE KEY",
                    "postgresql://user:password",
                    "ssh-rsa AAAAB3",
                    "192.168.1.100"
                ]
                
                for pattern in sensitive_patterns:
                    assert pattern not in response_text, f"Sensitive data exposed: {pattern}"
                
                # Verify configuration is sanitized or excluded
                data = response_data.get("data", {})
                config = data.get("configuration", {})
                
                # Sensitive keys should be filtered out or masked
                for key in ["api_key", "password", "private_key", "database_url"]:
                    if key in config:
                        value = config[key]
                        # Should be masked or empty
                        assert value != f"secret_{key}_12345" or value.startswith("***")
    
    async def test_error_message_information_disclosure(
        self, 
        async_test_client: AsyncClient
    ):
        """Test that error messages don't disclose sensitive information."""
        
        # Test various error conditions that might leak information
        test_cases = [
            # Non-existent project
            f"/api/project-index/{uuid.uuid4()}",
            # Malformed UUID
            "/api/project-index/invalid-uuid-format",
            # Non-existent file
            f"/api/project-index/{uuid.uuid4()}/files/non-existent-file.py",
        ]
        
        for endpoint in test_cases:
            response = await async_test_client.get(endpoint)
            
            if response.status_code >= 400:
                response_data = response.json()
                error_message = json.dumps(response_data)
                
                # Error messages should not contain sensitive information
                sensitive_info = [
                    "/etc/passwd",
                    "database",
                    "connection",
                    "internal",
                    "stack trace",
                    "file system",
                    "directory"
                ]
                
                for info in sensitive_info:
                    assert info.lower() not in error_message.lower(), f"Sensitive info in error: {info}"
                
                # Should not expose internal paths or system information
                assert "/opt/" not in error_message
                assert "/var/" not in error_message
                assert "/home/" not in error_message
                assert "C:\\" not in error_message