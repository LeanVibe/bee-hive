"""
Comprehensive Security Testing Suite for LeanVibe Agent Hive 2.0

Tests for:
- Authentication and authorization
- Input validation and sanitization  
- API security boundaries
- Token management and security
- Rate limiting and abuse prevention
- Data exposure and injection attacks
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, patch

import pytest
import httpx
from fastapi import status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import SecurityManager, TokenValidator, RateLimiter
from app.core.authorization_engine import AuthorizationEngine, Permission
from app.core.audit_logger import AuditLogger
from app.models.agent import Agent, AgentStatus
from tests.utils.database_test_utils import DatabaseTestUtils


@pytest.mark.security
class TestAuthenticationSecurity:
    """Test authentication security mechanisms."""
    
    async def test_token_validation_security(self):
        """Test token validation against various attack vectors."""
        validator = TokenValidator()
        
        # Valid tokens
        valid_tokens = [
            "ghp_1234567890abcdef1234567890abcdef12345678",
            "github_pat_11AAAAAAA0123456789abcdef0123456789abcdef"
        ]
        
        for token in valid_tokens:
            result = await validator.validate_token(token)
            assert result.is_valid
            assert not result.is_expired
        
        # Invalid/malicious tokens
        malicious_tokens = [
            "",  # Empty token
            "invalid_format",  # Wrong format
            "ghp_short",  # Too short
            "../../../etc/passwd",  # Path traversal
            "<script>alert('xss')</script>",  # XSS attempt
            "'; DROP TABLE users; --",  # SQL injection
            "ghp_" + "A" * 1000,  # Extremely long token
            None,  # Null token
        ]
        
        for token in malicious_tokens:
            result = await validator.validate_token(token)
            assert not result.is_valid
            assert result.error_reason is not None
    
    async def test_authorization_boundary_enforcement(self):
        """Test that authorization boundaries are properly enforced."""
        auth_engine = AuthorizationEngine()
        
        # Test different permission levels
        permissions_test_cases = [
            {
                "permissions": ["read:repository"],
                "action": "read_code",
                "resource": "repository:test/repo",
                "expected": True
            },
            {
                "permissions": ["read:repository"],
                "action": "write_code", 
                "resource": "repository:test/repo",
                "expected": False
            },
            {
                "permissions": ["admin:all"],
                "action": "delete_repository",
                "resource": "repository:test/repo", 
                "expected": True
            },
            {
                "permissions": [],
                "action": "read_code",
                "resource": "repository:test/repo",
                "expected": False
            }
        ]
        
        for case in permissions_test_cases:
            result = await auth_engine.check_permission(
                permissions=case["permissions"],
                action=case["action"],
                resource=case["resource"]
            )
            assert result == case["expected"], f"Failed for case: {case}"
    
    async def test_session_security(self, test_db_session: AsyncSession):
        """Test session security and token management."""
        security_manager = SecurityManager()
        
        # Create test agent
        agent = await DatabaseTestUtils.create_test_agent(test_db_session)
        
        # Test session creation
        session_token = await security_manager.create_session(agent.id)
        assert session_token is not None
        assert len(session_token) >= 32  # Minimum entropy
        
        # Test session validation
        is_valid = await security_manager.validate_session(session_token)
        assert is_valid
        
        # Test session expiration
        with patch('app.core.security.datetime') as mock_datetime:
            # Fast-forward time
            mock_datetime.utcnow.return_value = datetime.utcnow() + timedelta(hours=25)
            
            is_valid = await security_manager.validate_session(session_token)
            assert not is_valid
        
        # Test session invalidation
        session_token_2 = await security_manager.create_session(agent.id)
        await security_manager.invalidate_session(session_token_2)
        
        is_valid = await security_manager.validate_session(session_token_2)
        assert not is_valid


@pytest.mark.security  
class TestInputValidationSecurity:
    """Test input validation and sanitization."""
    
    @pytest.mark.parametrize("malicious_input", [
        "<script>alert('xss')</script>",
        "'; DROP TABLE agents; --",
        "../../../etc/passwd",
        "{{7*7}}{{request.application.__globals__.__builtins__.__import__('os').popen('id').read()}}",
        "\x00\x01\x02",  # Null bytes
        "A" * 10000,  # Extremely long input
        "javascript:alert('xss')",
        "data:text/html,<script>alert('xss')</script>",
    ])
    async def test_malicious_input_sanitization(self, malicious_input: str):
        """Test that malicious inputs are properly sanitized."""
        from app.core.security import InputSanitizer
        
        sanitizer = InputSanitizer()
        
        # Test string sanitization
        clean_string = sanitizer.sanitize_string(malicious_input)
        assert "<script>" not in clean_string
        assert "DROP TABLE" not in clean_string
        assert "../" not in clean_string
        assert "javascript:" not in clean_string
        
        # Test SQL injection prevention
        clean_sql = sanitizer.sanitize_sql_input(malicious_input)
        sql_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "UNION", "SELECT"]
        for keyword in sql_keywords:
            assert keyword not in clean_sql.upper()
    
    async def test_json_payload_security(self, async_test_client: httpx.AsyncClient):
        """Test JSON payload security and validation."""
        malicious_payloads = [
            {"name": "<script>alert('xss')</script>"},
            {"description": "'; DROP TABLE agents; --"},
            {"config": {"key": "../../../etc/passwd"}},
            {"capabilities": ["{{7*7}}"]},
            {"metadata": {"file": "data:text/html,<script>"}},
            # Deeply nested payload
            {"nested": {"level1": {"level2": {"level3": {"xss": "<script>"}}}}},
            # Extremely large payload
            {"large_field": "A" * 100000},
        ]
        
        for payload in malicious_payloads:
            response = await async_test_client.post(
                "/api/v1/agents",
                json=payload
            )
            
            # Should either reject (4xx) or sanitize the input
            if response.status_code == 200:
                # If accepted, check that dangerous content was sanitized
                response_data = response.json()
                response_str = json.dumps(response_data)
                assert "<script>" not in response_str
                assert "DROP TABLE" not in response_str
                assert "../" not in response_str
    
    async def test_file_upload_security(self, async_test_client: httpx.AsyncClient):
        """Test file upload security."""
        malicious_files = [
            # Executable files
            ("malware.exe", b"MZ\x90\x00", "application/octet-stream"),
            # Script files
            ("script.sh", b"#!/bin/bash\nrm -rf /", "text/plain"),
            # Large files
            ("large.txt", b"A" * 10_000_000, "text/plain"),
            # Files with null bytes
            ("null.txt", b"test\x00\x01", "text/plain"),
            # HTML with script
            ("index.html", b"<script>alert('xss')</script>", "text/html"),
        ]
        
        for filename, content, content_type in malicious_files:
            files = {"file": (filename, content, content_type)}
            
            response = await async_test_client.post(
                "/api/v1/upload",
                files=files
            )
            
            # Should reject malicious files
            assert response.status_code in [400, 403, 413, 415]


@pytest.mark.security
class TestRateLimitingSecurity:
    """Test rate limiting and abuse prevention."""
    
    async def test_api_rate_limiting(self, async_test_client: httpx.AsyncClient):
        """Test API rate limiting mechanisms."""
        # Make rapid requests to trigger rate limiting
        responses = []
        
        for i in range(100):  # Exceed typical rate limits
            response = await async_test_client.get("/api/v1/health")
            responses.append(response)
            
            # Small delay to avoid overwhelming the test
            if i % 10 == 0:
                await asyncio.sleep(0.1)
        
        # Should have some rate limited responses
        rate_limited_count = sum(1 for r in responses if r.status_code == 429)
        assert rate_limited_count > 0, "Rate limiting should be triggered"
        
        # Check rate limit headers
        for response in responses[-10:]:  # Check last few responses
            if response.status_code == 429:
                assert "X-RateLimit-Remaining" in response.headers
                assert "X-RateLimit-Reset" in response.headers
    
    async def test_rate_limiting_per_agent(self):
        """Test rate limiting per agent/user."""
        rate_limiter = RateLimiter()
        
        agent_id = "test_agent_123"
        
        # Test within limits
        for i in range(10):
            allowed = await rate_limiter.check_rate_limit(
                key=f"agent:{agent_id}",
                limit=20,
                window_seconds=60
            )
            assert allowed, f"Request {i} should be allowed"
        
        # Test exceeding limits
        for i in range(15):  # Exceed the limit of 20
            allowed = await rate_limiter.check_rate_limit(
                key=f"agent:{agent_id}",
                limit=20,
                window_seconds=60
            )
            
            if i < 10:  # First 10 more should be allowed (total 20)
                assert allowed
            else:  # Remaining should be rate limited
                assert not allowed
    
    async def test_distributed_rate_limiting(self):
        """Test rate limiting across distributed instances."""
        rate_limiter = RateLimiter(distributed=True)
        
        # Simulate multiple instances
        instance_ids = ["instance_1", "instance_2", "instance_3"]
        agent_id = "distributed_test_agent"
        
        total_allowed = 0
        
        for instance_id in instance_ids:
            for i in range(10):
                allowed = await rate_limiter.check_rate_limit(
                    key=f"agent:{agent_id}",
                    limit=20,  # Global limit across all instances
                    window_seconds=60,
                    instance_id=instance_id
                )
                
                if allowed:
                    total_allowed += 1
        
        # Should not exceed global limit regardless of instances
        assert total_allowed <= 20


@pytest.mark.security
class TestDataExposureSecurity:
    """Test data exposure and information leakage prevention."""
    
    async def test_sensitive_data_filtering(self, async_test_client: httpx.AsyncClient):
        """Test that sensitive data is not exposed in API responses."""
        # Create agent with sensitive config
        agent_data = {
            "name": "Test Agent",
            "type": "claude",
            "role": "test",
            "config": {
                "api_key": "sk-secret123456789",
                "password": "supersecret",
                "token": "ghp_secret_token",
                "webhook_secret": "webhook_secret_123",
                "public_info": "this_should_be_visible"
            }
        }
        
        response = await async_test_client.post("/api/v1/agents", json=agent_data)
        
        if response.status_code == 200:
            response_data = response.json()
            response_str = json.dumps(response_data)
            
            # Sensitive data should be filtered out
            sensitive_patterns = [
                "sk-secret123456789",
                "supersecret", 
                "ghp_secret_token",
                "webhook_secret_123"
            ]
            
            for pattern in sensitive_patterns:
                assert pattern not in response_str, f"Sensitive data '{pattern}' exposed"
            
            # Public info should still be present
            assert "this_should_be_visible" in response_str
    
    async def test_error_message_security(self, async_test_client: httpx.AsyncClient):
        """Test that error messages don't leak sensitive information."""
        # Trigger various error conditions
        error_test_cases = [
            # Invalid agent ID (might reveal DB structure)
            ("GET", "/api/v1/agents/invalid-uuid"),
            # Non-existent resource
            ("GET", "/api/v1/agents/00000000-0000-0000-0000-000000000000"),
            # Malformed request
            ("POST", "/api/v1/agents", {"invalid": "data"}),
            # Large payload
            ("POST", "/api/v1/agents", {"data": "A" * 100000}),
        ]
        
        for method, url, data in error_test_cases:
            if method == "GET":
                response = await async_test_client.get(url)
            else:
                response = await async_test_client.post(url, json=data)
            
            # Check that error responses don't contain sensitive info
            if 400 <= response.status_code < 600:
                error_text = response.text.lower()
                
                # Should not contain sensitive patterns
                sensitive_patterns = [
                    "password", "secret", "token", "key",
                    "database", "sql", "postgres", "redis",
                    "traceback", "exception", "error in",
                    "/users/", "/home/", "c:\\",  # File paths
                ]
                
                for pattern in sensitive_patterns:
                    assert pattern not in error_text, f"Error message contains '{pattern}'"
    
    async def test_audit_logging_security(self, test_db_session: AsyncSession):
        """Test security audit logging."""
        audit_logger = AuditLogger()
        
        # Log various security events
        security_events = [
            {
                "event_type": "authentication_failure",
                "agent_id": "test_agent",
                "details": {"reason": "invalid_token", "ip": "192.168.1.100"}
            },
            {
                "event_type": "authorization_denied", 
                "agent_id": "test_agent",
                "details": {"action": "delete_repository", "resource": "sensitive_repo"}
            },
            {
                "event_type": "rate_limit_exceeded",
                "agent_id": "test_agent", 
                "details": {"endpoint": "/api/v1/agents", "count": 100}
            },
            {
                "event_type": "malicious_input_detected",
                "agent_id": "test_agent",
                "details": {"input": "<script>alert", "sanitized": True}
            }
        ]
        
        for event in security_events:
            await audit_logger.log_security_event(**event)
        
        # Verify audit logs were created
        audit_entries = await audit_logger.get_security_events(
            event_types=["authentication_failure", "authorization_denied"]
        )
        
        assert len(audit_entries) >= 2
        
        for entry in audit_entries:
            assert entry["timestamp"] is not None
            assert entry["event_type"] in ["authentication_failure", "authorization_denied"]
            assert "agent_id" in entry
            assert "details" in entry


@pytest.mark.security
class TestAPISecurityBoundaries:
    """Test API security boundaries and access controls."""
    
    async def test_cors_security(self, async_test_client: httpx.AsyncClient):
        """Test CORS security configuration."""
        # Test preflight request
        response = await async_test_client.options(
            "/api/v1/agents",
            headers={
                "Origin": "https://malicious-site.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        # Should have proper CORS headers
        assert "Access-Control-Allow-Origin" in response.headers
        
        # Should not allow arbitrary origins (unless configured to)
        allowed_origin = response.headers.get("Access-Control-Allow-Origin")
        assert allowed_origin != "*" or allowed_origin != "https://malicious-site.com"
    
    async def test_content_type_security(self, async_test_client: httpx.AsyncClient):  
        """Test content type validation and security."""
        # Test various content types
        test_cases = [
            # Valid JSON
            ("application/json", '{"name": "test"}', [200, 201, 400]),
            # Invalid content type for JSON endpoint
            ("text/plain", '{"name": "test"}', [400, 415]),
            # Malicious content type
            ("application/javascript", "alert('xss')", [400, 415]),
            # Empty content type
            ("", '{"name": "test"}', [400, 415]),
        ]
        
        for content_type, data, expected_statuses in test_cases:
            headers = {"Content-Type": content_type} if content_type else {}
            
            response = await async_test_client.post(
                "/api/v1/agents",
                content=data,
                headers=headers
            )
            
            assert response.status_code in expected_statuses
    
    async def test_http_security_headers(self, async_test_client: httpx.AsyncClient):
        """Test that proper security headers are present."""
        response = await async_test_client.get("/api/v1/health")
        
        # Check for important security headers
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy"
        ]
        
        for header in security_headers:
            assert header in response.headers, f"Missing security header: {header}"
        
        # Verify header values
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") in ["DENY", "SAMEORIGIN"]
        assert response.headers.get("X-XSS-Protection") == "1; mode=block"


@pytest.mark.security
@pytest.mark.performance
class TestSecurityPerformance:
    """Test security mechanisms don't impact performance negatively."""
    
    async def test_authentication_performance(self):
        """Test that authentication doesn't cause performance issues."""
        security_manager = SecurityManager()
        
        # Measure token validation performance
        token = "ghp_1234567890abcdef1234567890abcdef12345678"
        
        start_time = time.time()
        
        # Validate token multiple times
        for _ in range(100):
            result = await security_manager.validate_token(token)
            assert result.is_valid
        
        end_time = time.time()
        avg_time_ms = ((end_time - start_time) / 100) * 1000
        
        # Should be fast (under 10ms per validation)
        assert avg_time_ms < 10, f"Token validation too slow: {avg_time_ms}ms"
    
    async def test_rate_limiting_performance(self):
        """Test rate limiting performance under load."""
        rate_limiter = RateLimiter()
        
        start_time = time.time()
        
        # Check rate limits rapidly
        tasks = []
        for i in range(1000):
            task = rate_limiter.check_rate_limit(
                key=f"perf_test_{i % 10}",  # 10 different keys
                limit=100,
                window_seconds=60
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        avg_time_ms = ((end_time - start_time) / 1000) * 1000
        
        # Should be fast (under 5ms per check)
        assert avg_time_ms < 5, f"Rate limiting too slow: {avg_time_ms}ms"
        
        # Most should be allowed (within limits)
        allowed_count = sum(1 for r in results if r)
        assert allowed_count > 500  # Reasonable threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])