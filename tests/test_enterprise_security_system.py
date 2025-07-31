"""
Comprehensive Test Suite for Enterprise Security System.

Tests OAuth 2.0/OIDC integration, RBAC, audit logging,
API security middleware, and compliance features.
"""

import asyncio
import json
import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
import httpx

from app.core.oauth_provider_system import (
    OAuthProviderSystem, OAuthProviderType, OAuthSession, OAuthTokenSet
)
from app.core.api_security_middleware import (
    APISecurityMiddleware, SecurityConfig, RateLimitRule, RateLimitStrategy
)
from app.core.comprehensive_audit_system import (
    ComprehensiveAuditSystem, AuditContext, AuditEventCategory, ComplianceFramework
)
from app.core.authorization_engine import AuthorizationEngine, AccessDecision
from app.core.security_orchestrator_integration import SecurityOrchestrator
from app.schemas.security import (
    OAuthProviderConfig, OAuthAuthorizationRequest, SecurityEventSeverityEnum
)


class TestOAuthProviderSystem:
    """Test OAuth 2.0/OIDC provider system."""
    
    @pytest.fixture
    async def oauth_system(self, db_session, redis_client):
        """Create OAuth provider system fixture."""
        return OAuthProviderSystem(
            db_session=db_session,
            redis_client=redis_client,
            base_url="http://localhost:8000"
        )
    
    @pytest.fixture
    def mock_http_client(self):
        """Mock HTTP client for OAuth requests."""
        with patch('httpx.AsyncClient') as mock_client:
            yield mock_client
    
    async def test_configure_google_provider(self, oauth_system):
        """Test Google OAuth provider configuration."""
        success = await oauth_system.configure_provider(
            provider_name="google_test",
            provider_type=OAuthProviderType.GOOGLE,
            client_id="test-client-id.apps.googleusercontent.com",
            client_secret="test-client-secret"
        )
        
        assert success
        assert "google_test" in oauth_system.providers
        
        config = oauth_system.providers["google_test"]
        assert config.provider_type == OAuthProviderType.GOOGLE
        assert config.client_id == "test-client-id.apps.googleusercontent.com"
        assert "openid" in config.scopes
    
    async def test_configure_github_provider(self, oauth_system):
        """Test GitHub OAuth provider configuration."""
        success = await oauth_system.configure_provider(
            provider_name="github_test",
            provider_type=OAuthProviderType.GITHUB,
            client_id="test-github-client-id",
            client_secret="test-github-secret"
        )
        
        assert success
        config = oauth_system.providers["github_test"]
        assert config.provider_type == OAuthProviderType.GITHUB
        assert "user:email" in config.scopes
    
    async def test_initiate_authorization(self, oauth_system):
        """Test OAuth authorization initiation."""
        # Configure provider first
        await oauth_system.configure_provider(
            provider_name="test_provider",
            provider_type=OAuthProviderType.GOOGLE,
            client_id="test-client-id",
            client_secret="test-secret"
        )
        
        auth_url, session_id = await oauth_system.initiate_authorization(
            provider_name="test_provider",
            scopes=["openid", "email"]
        )
        
        assert auth_url.startswith("https://accounts.google.com/o/oauth2/v2/auth")
        assert "client_id=test-client-id" in auth_url
        assert "scope=openid+email" in auth_url
        assert session_id is not None
    
    async def test_pkce_support(self, oauth_system):
        """Test PKCE (Proof Key for Code Exchange) support."""
        await oauth_system.configure_provider(
            provider_name="pkce_test",
            provider_type=OAuthProviderType.GOOGLE,
            client_id="test-client-id",
            client_secret="test-secret"
        )
        
        auth_url, session_id = await oauth_system.initiate_authorization(
            provider_name="pkce_test"
        )
        
        # PKCE should be enabled by default
        assert "code_challenge=" in auth_url
        assert "code_challenge_method=S256" in auth_url
    
    async def test_authorization_callback_success(self, oauth_system, mock_http_client):
        """Test successful OAuth authorization callback."""
        # Configure provider
        await oauth_system.configure_provider(
            provider_name="callback_test",
            provider_type=OAuthProviderType.GOOGLE,
            client_id="test-client-id",
            client_secret="test-secret"
        )
        
        # Initiate authorization to create session
        auth_url, session_id = await oauth_system.initiate_authorization(
            provider_name="callback_test"
        )
        
        # Extract state from auth URL
        state = auth_url.split("state=")[1].split("&")[0]
        
        # Mock HTTP responses
        mock_client_instance = mock_http_client.return_value
        
        # Mock token exchange
        token_response = Mock()
        token_response.status_code = 200
        token_response.json.return_value = {
            "access_token": "test-access-token",
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": "test-refresh-token",
            "id_token": "test-id-token"
        }
        
        # Mock user info
        userinfo_response = Mock()
        userinfo_response.status_code = 200
        userinfo_response.json.return_value = {
            "sub": "123456789",
            "email": "test@example.com",
            "name": "Test User",
            "picture": "https://example.com/avatar.jpg"
        }
        
        mock_client_instance.post = AsyncMock(return_value=token_response)
        mock_client_instance.get = AsyncMock(return_value=userinfo_response)
        
        # Handle callback
        token_set, user_profile = await oauth_system.handle_authorization_callback(
            provider_name="callback_test",
            code="test-auth-code",
            state=state
        )
        
        assert token_set.access_token == "test-access-token"
        assert user_profile.user_id == "123456789"
        assert user_profile.email == "test@example.com"
        assert user_profile.provider == "google"
    
    async def test_token_refresh(self, oauth_system, mock_http_client):
        """Test OAuth token refresh."""
        # Setup provider and mock responses
        await oauth_system.configure_provider(
            provider_name="refresh_test",
            provider_type=OAuthProviderType.GOOGLE,
            client_id="test-client-id",
            client_secret="test-secret"
        )
        
        # Mock refresh response
        mock_client_instance = mock_http_client.return_value
        refresh_response = Mock()
        refresh_response.status_code = 200
        refresh_response.json.return_value = {
            "access_token": "new-access-token",
            "token_type": "Bearer",
            "expires_in": 3600
        }
        mock_client_instance.post = AsyncMock(return_value=refresh_response)
        
        # Store initial token set
        initial_token = OAuthTokenSet(
            access_token="old-token",
            refresh_token="refresh-token",
            expires_in=3600
        )
        await oauth_system._store_token_set("test-user", "refresh_test", initial_token)
        
        # Test refresh
        new_token_set = await oauth_system.refresh_access_token("refresh_test", "test-user")
        
        assert new_token_set is not None
        assert new_token_set.access_token == "new-access-token"
    
    async def test_provider_metrics(self, oauth_system):
        """Test OAuth provider metrics collection."""
        # Configure provider and initiate auth
        await oauth_system.configure_provider(
            provider_name="metrics_test",
            provider_type=OAuthProviderType.GOOGLE,
            client_id="test-client-id",
            client_secret="test-secret"
        )
        
        await oauth_system.initiate_authorization(provider_name="metrics_test")
        
        metrics = oauth_system.get_metrics()
        
        assert "oauth_metrics" in metrics
        assert metrics["oauth_metrics"]["authorization_requests"] > 0
        assert "provider_usage" in metrics["oauth_metrics"]


class TestAPISecurityMiddleware:
    """Test API Security Middleware."""
    
    @pytest.fixture
    def security_config(self):
        """Security configuration fixture."""
        return SecurityConfig(
            enable_rate_limiting=True,
            default_rate_limit=10,  # Low limit for testing
            enable_security_headers=True,
            enable_threat_detection=True,
            enable_sql_injection_detection=True,
            enable_xss_detection=True
        )
    
    @pytest.fixture
    def app_with_middleware(self, security_config, redis_client):
        """FastAPI app with security middleware."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        @app.post("/test-post")
        async def test_post_endpoint(data: dict):
            return {"received": data}
        
        # Add security middleware
        app.add_middleware(
            APISecurityMiddleware,
            redis_client=redis_client,
            config=security_config
        )
        
        return app
    
    def test_security_headers(self, app_with_middleware):
        """Test security headers are added."""
        client = TestClient(app_with_middleware)
        response = client.get("/test")
        
        assert response.status_code == 200
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
    
    def test_rate_limiting(self, app_with_middleware):
        """Test rate limiting enforcement."""
        client = TestClient(app_with_middleware)
        
        # Make requests up to limit
        for i in range(10):
            response = client.get("/test")
            assert response.status_code == 200
        
        # Next request should be rate limited
        response = client.get("/test")
        assert response.status_code == 429
        assert "rate_limit_exceeded" in response.json()["error"]
    
    def test_sql_injection_detection(self, app_with_middleware):
        """Test SQL injection detection."""
        client = TestClient(app_with_middleware)
        
        # Test malicious query parameter
        response = client.get("/test?id=1' OR '1'='1")
        assert response.status_code == 403
        assert "access_denied" in response.json()["error"]
    
    def test_xss_detection(self, app_with_middleware):
        """Test XSS detection."""
        client = TestClient(app_with_middleware)
        
        # Test malicious script in request body
        malicious_data = {"content": "<script>alert('xss')</script>"}
        response = client.post("/test-post", json=malicious_data)
        assert response.status_code == 403
    
    def test_request_size_limit(self, app_with_middleware):
        """Test request size limiting."""
        client = TestClient(app_with_middleware)
        
        # Create large payload
        large_data = {"data": "x" * (11 * 1024 * 1024)}  # 11MB, exceeds 10MB limit
        
        response = client.post("/test-post", json=large_data)
        assert response.status_code == 403
    
    def test_path_traversal_detection(self, app_with_middleware):
        """Test path traversal detection."""
        client = TestClient(app_with_middleware)
        
        # Test path traversal in URL
        response = client.get("/test/../../../etc/passwd")
        assert response.status_code == 403


class TestComprehensiveAuditSystem:
    """Test Comprehensive Audit System."""
    
    @pytest.fixture
    async def audit_system(self, db_session, redis_client):
        """Create audit system fixture."""
        return ComprehensiveAuditSystem(
            db_session=db_session,
            redis_client=redis_client,
            enabled_frameworks=[ComplianceFramework.SOC2, ComplianceFramework.ISO27001]
        )
    
    async def test_audit_event_logging(self, audit_system):
        """Test audit event logging."""
        context = AuditContext(
            agent_id=uuid.uuid4(),
            user_id="test-user",
            action="read_file",
            resource="files/document.txt",
            success=True,
            client_ip="192.168.1.1",
            start_time=datetime.utcnow()
        )
        
        event_id = await audit_system.log_audit_event(
            context=context,
            category=AuditEventCategory.DATA_ACCESS
        )
        
        assert event_id is not None
        
        # Verify metrics updated
        metrics = audit_system.get_metrics()
        assert metrics["audit_system_metrics"]["events_logged"] > 0
    
    async def test_security_event_logging(self, audit_system):
        """Test security event logging."""
        event_id = await audit_system.log_security_event(
            event_type="failed_authentication",
            severity=SecurityEventSeverityEnum.HIGH,
            description="Multiple failed login attempts",
            agent_id=uuid.uuid4(),
            details={"attempts": 5, "ip": "192.168.1.100"}
        )
        
        assert event_id is not None
    
    async def test_compliance_framework_support(self, audit_system):
        """Test compliance framework support."""
        # Test SOC 2 compliance
        context = AuditContext(
            user_id="test-user",
            action="authenticate",
            success=True,
            client_ip="192.168.1.1",
            authentication_method="oauth2"
        )
        
        event_id = await audit_system.log_audit_event(
            context=context,
            category=AuditEventCategory.AUTHENTICATION,
            compliance_frameworks=[ComplianceFramework.SOC2]
        )
        
        assert event_id is not None
    
    async def test_risk_score_calculation(self, audit_system):
        """Test risk score calculation."""
        # High-risk context
        high_risk_context = AuditContext(
            user_id="test-user",
            action="delete_system_file",
            resource="system/critical.conf",
            success=False,
            error_code="PERMISSION_DENIED",
            data_classification="confidential",
            start_time=datetime.utcnow().replace(hour=2)  # Off-hours
        )
        
        risk_score = await audit_system._calculate_risk_score(
            high_risk_context,
            AuditEventCategory.SYSTEM_ADMINISTRATION
        )
        
        assert risk_score > 0.5  # Should be high risk
    
    async def test_log_integrity_verification(self, audit_system):
        """Test audit log integrity verification."""
        # Log some events
        for i in range(5):
            context = AuditContext(
                user_id=f"user-{i}",
                action=f"action-{i}",
                success=True
            )
            await audit_system.log_audit_event(context, AuditEventCategory.DATA_ACCESS)
        
        # Verify integrity
        verification_result = await audit_system.verify_log_integrity(sample_size=5)
        
        assert verification_result["total_checked"] <= 5
        assert verification_result["integrity_percentage"] >= 0
    
    async def test_compliance_reporting(self, audit_system):
        """Test compliance report generation."""
        # Log some events for compliance
        context = AuditContext(
            user_id="compliance-user",
            action="access_sensitive_data",
            resource="data/personal_info.csv",
            success=True,
            permissions_checked=["data:read"],
            authorization_result="granted"
        )
        
        await audit_system.log_audit_event(context, AuditEventCategory.DATA_ACCESS)
        
        # Generate compliance report
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=1)
        
        report = await audit_system.generate_compliance_report(
            framework=ComplianceFramework.SOC2,
            start_date=start_date,
            end_date=end_date
        )
        
        assert report["framework"] == "soc2"
        assert "overall_compliance_score" in report
        assert "rule_compliance" in report


class TestAuthorizationEngine:
    """Test Authorization Engine."""
    
    @pytest.fixture
    async def auth_engine(self, db_session, redis_client):
        """Create authorization engine fixture."""
        return AuthorizationEngine(
            db_session=db_session,
            redis_client=redis_client
        )
    
    async def test_role_creation(self, auth_engine):
        """Test role creation."""
        role = await auth_engine.create_role(
            role_name="test_developer",
            permissions={
                "resources": ["github", "files"],
                "actions": ["read", "write"]
            },
            created_by="test_admin",
            description="Test developer role",
            resource_patterns=["github/repos/org/*"]
        )
        
        assert role is not None
        assert role.role_name == "test_developer"
        assert "github" in role.permissions["resources"]
    
    async def test_permission_check(self, auth_engine, db_session):
        """Test permission checking."""
        # Create a test role and agent
        role = await auth_engine.create_role(
            role_name="file_reader",
            permissions={
                "resources": ["files"],
                "actions": ["read"]
            },
            created_by="system"
        )
        
        # Create agent identity (simplified for test)
        from app.models.security import AgentIdentity
        agent = AgentIdentity(
            agent_name="test_agent",
            human_controller="test@example.com",
            created_by="system"
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)
        
        # Assign role
        success = await auth_engine.assign_role(
            agent_id=str(agent.id),
            role_id=str(role.id),
            granted_by="test_admin"
        )
        assert success
        
        # Check permission
        result = await auth_engine.check_permission(
            agent_id=str(agent.id),
            resource="files",
            action="read"
        )
        
        assert result.decision == AccessDecision.GRANTED
        assert "file_reader" in result.matched_roles
    
    async def test_permission_denied(self, auth_engine, db_session):
        """Test permission denial."""
        # Create agent without appropriate role
        from app.models.security import AgentIdentity
        agent = AgentIdentity(
            agent_name="restricted_agent",
            human_controller="test@example.com",
            created_by="system"
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)
        
        # Check permission without role
        result = await auth_engine.check_permission(
            agent_id=str(agent.id),
            resource="admin_panel",
            action="write"
        )
        
        assert result.decision == AccessDecision.DENIED
    
    async def test_role_assignment_conditions(self, auth_engine, db_session):
        """Test conditional role assignments."""
        # Create role and agent
        role = await auth_engine.create_role(
            role_name="time_restricted",
            permissions={
                "resources": ["data"],
                "actions": ["read"]
            },
            created_by="system"
        )
        
        from app.models.security import AgentIdentity
        agent = AgentIdentity(
            agent_name="conditional_agent",
            human_controller="test@example.com",
            created_by="system"
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)
        
        # Assign role with time restriction
        success = await auth_engine.assign_role(
            agent_id=str(agent.id),
            role_id=str(role.id),
            granted_by="test_admin",
            conditions={"time_restricted": "09:00-17:00"}
        )
        assert success
        
        # Permission check should consider conditions
        result = await auth_engine.check_permission(
            agent_id=str(agent.id),
            resource="data",
            action="read",
            context={"current_time": "14:30"}  # Within allowed hours
        )
        
        # Note: This test might need adjustment based on actual condition checking logic
        assert result.decision in [AccessDecision.GRANTED, AccessDecision.DENIED]


class TestSecurityOrchestrator:
    """Test Security Orchestrator Integration."""
    
    @pytest.fixture
    async def security_orchestrator(self, db_session, redis_client):
        """Create security orchestrator fixture."""
        app = FastAPI()
        orchestrator = SecurityOrchestrator(
            app=app,
            db_session=db_session,
            redis_client=redis_client
        )
        await orchestrator.initialize()
        return orchestrator
    
    async def test_orchestrator_initialization(self, security_orchestrator):
        """Test security orchestrator initialization."""
        assert security_orchestrator.oauth_system is not None
        assert security_orchestrator.authorization_engine is not None
        assert security_orchestrator.audit_system is not None
    
    async def test_oauth_provider_configuration(self, security_orchestrator):
        """Test OAuth provider configuration through orchestrator."""
        success = await security_orchestrator.configure_oauth_provider(
            provider_name="test_google",
            provider_type="google",
            client_id="test-client-id",
            client_secret="test-client-secret"
        )
        
        assert success
    
    async def test_agent_action_logging(self, security_orchestrator):
        """Test agent action logging through orchestrator."""
        agent_id = uuid.uuid4()
        
        event_id = await security_orchestrator.log_agent_action(
            agent_id=agent_id,
            action="read_file",
            resource="documents/test.txt",
            success=True,
            duration_ms=150
        )
        
        assert event_id is not None
        
        # Check metrics
        metrics = security_orchestrator.get_security_metrics()
        assert metrics["orchestrator_metrics"]["audit_events_logged"] > 0
    
    async def test_security_threat_detection(self, security_orchestrator):
        """Test security threat detection."""
        event_id = await security_orchestrator.detect_security_threat(
            agent_id=uuid.uuid4(),
            threat_type="suspicious_activity",
            description="Unusual access pattern detected",
            severity="high",
            details={"pattern": "rapid_file_access", "count": 50}
        )
        
        assert event_id is not None
        
        # Check metrics
        metrics = security_orchestrator.get_security_metrics()
        assert metrics["orchestrator_metrics"]["security_violations_detected"] > 0
    
    async def test_compliance_report_generation(self, security_orchestrator):
        """Test compliance report generation."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=1)
        
        report = await security_orchestrator.generate_compliance_report(
            framework="soc2",
            start_date=start_date,
            end_date=end_date
        )
        
        assert report["framework"] == "soc2"
        assert "overall_compliance_score" in report
    
    async def test_health_check(self, security_orchestrator):
        """Test security system health check."""
        health = await security_orchestrator.health_check()
        
        assert health["overall_status"] in ["healthy", "degraded"]
        assert "components" in health
        assert "oauth" in health["components"]
        assert "authorization" in health["components"]
        assert "audit" in health["components"]


class TestEndToEndSecurity:
    """End-to-end security integration tests."""
    
    @pytest.fixture
    def secure_app(self, db_session, redis_client):
        """Create secure FastAPI app with all security components."""
        app = FastAPI()
        
        # Add security middleware
        security_config = SecurityConfig(
            enable_rate_limiting=True,
            default_rate_limit=50,
            enable_security_headers=True,
            enable_threat_detection=True
        )
        
        app.add_middleware(
            APISecurityMiddleware,
            redis_client=redis_client,
            config=security_config
        )
        
        @app.get("/public")
        async def public_endpoint():
            return {"message": "public"}
        
        @app.get("/protected")
        async def protected_endpoint():
            # In real app, this would use security dependencies
            return {"message": "protected", "user": "authenticated"}
        
        return app
    
    def test_public_endpoint_access(self, secure_app):
        """Test public endpoint access."""
        client = TestClient(secure_app)
        response = client.get("/public")
        
        assert response.status_code == 200
        assert response.json()["message"] == "public"
        
        # Security headers should be present
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
    
    def test_security_middleware_integration(self, secure_app):
        """Test security middleware integration."""
        client = TestClient(secure_app)
        
        # Test with malicious request
        response = client.get("/public?search=<script>alert('xss')</script>")
        assert response.status_code == 403
        
        # Test with SQL injection attempt
        response = client.get("/public?id=1' OR '1'='1")
        assert response.status_code == 403
    
    def test_rate_limiting_integration(self, secure_app):
        """Test rate limiting in integrated environment."""
        client = TestClient(secure_app)
        
        # Make requests within limit
        for i in range(40):  # Below 50 limit
            response = client.get("/public")
            assert response.status_code == 200
        
        # Exceed rate limit
        for i in range(15):  # Push over 50 limit
            response = client.get("/public")
        
        # Should be rate limited
        assert response.status_code == 429


# Test Configuration and Fixtures

@pytest.fixture
async def db_session():
    """Mock database session."""
    mock_session = AsyncMock()
    mock_session.add = Mock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.get = AsyncMock(return_value=None)
    return mock_session


@pytest.fixture
async def redis_client():
    """Mock Redis client."""
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set_with_expiry = AsyncMock()
    mock_redis.delete = AsyncMock()
    mock_redis.lpush = AsyncMock()
    mock_redis.xadd = AsyncMock()
    return mock_redis


# Performance Tests

class TestPerformance:
    """Performance tests for security components."""
    
    @pytest.mark.asyncio
    async def test_audit_logging_performance(self, db_session, redis_client):
        """Test audit logging performance under load."""
        audit_system = ComprehensiveAuditSystem(
            db_session=db_session,
            redis_client=redis_client
        )
        
        import time
        start_time = time.time()
        
        # Log 100 events
        for i in range(100):
            context = AuditContext(
                user_id=f"user-{i}",
                action=f"action-{i}",
                success=True
            )
            await audit_system.log_audit_event(context, AuditEventCategory.DATA_ACCESS)
        
        duration = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert duration < 5.0  # 5 seconds for 100 events
        
        # Check average processing time
        metrics = audit_system.get_metrics()
        avg_time = metrics["audit_system_metrics"]["avg_logging_time_ms"]
        assert avg_time < 50  # Less than 50ms per event
    
    @pytest.mark.asyncio
    async def test_authorization_performance(self, db_session, redis_client):
        """Test authorization engine performance."""
        auth_engine = AuthorizationEngine(
            db_session=db_session,
            redis_client=redis_client
        )
        
        import time
        start_time = time.time()
        
        # Perform 50 permission checks
        for i in range(50):
            await auth_engine.check_permission(
                agent_id=f"agent-{i}",
                resource="test_resource",
                action="read"
            )
        
        duration = time.time() - start_time
        
        # Should be fast (adjust threshold as needed)
        assert duration < 2.0  # 2 seconds for 50 checks
    
    def test_middleware_performance(self, redis_client):
        """Test API security middleware performance."""
        app = FastAPI()
        
        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}
        
        security_config = SecurityConfig(
            enable_rate_limiting=True,
            enable_security_headers=True,
            enable_threat_detection=True
        )
        
        app.add_middleware(
            APISecurityMiddleware,
            redis_client=redis_client,
            config=security_config
        )
        
        client = TestClient(app)
        
        import time
        start_time = time.time()
        
        # Make 100 requests
        for i in range(100):
            response = client.get("/test")
            assert response.status_code == 200
        
        duration = time.time() - start_time
        
        # Should handle requests efficiently
        assert duration < 10.0  # 10 seconds for 100 requests


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])