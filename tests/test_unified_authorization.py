"""
Comprehensive tests for Unified Authorization Engine
Tests all consolidated security functionality including authentication, authorization, and threat detection
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from app.core.unified_authorization_engine import (
    UnifiedAuthorizationEngine,
    get_unified_authorization_engine,
    User,
    Role,
    Permission,
    AuthenticationToken,
    AuthorizationContext,
    AuthorizationResult,
    SecurityValidationResult,
    PermissionLevel,
    ResourceType,
    AuthenticationMethod,
    AccessDecision,
    ThreatLevel,
    require_permission,
    require_role
)

from app.core.security_integration import (
    SecurityOrchestrationIntegration,
    get_security_integration
)


class TestUnifiedAuthorizationEngine:
    """Test suite for the unified authorization engine."""
    
    @pytest.fixture
    def auth_engine(self):
        """Create auth engine for testing."""
        # Mock the singleton to create fresh instance for each test
        UnifiedAuthorizationEngine._instance = None
        
        with patch('app.core.unified_authorization_engine.get_component_logger'):
            with patch('app.core.unified_authorization_engine.ConfigurationService'):
                with patch('app.core.unified_authorization_engine.get_redis_service'):
                    with patch('app.core.unified_authorization_engine.CircuitBreakerService'):
                        engine = UnifiedAuthorizationEngine()
                        return engine
    
    @pytest.fixture
    def sample_user(self):
        """Create sample user for testing."""
        return User(
            user_id="test_user_123",
            username="testuser",
            email="test@example.com",
            roles=["user"],
            is_active=True
        )
    
    @pytest.fixture
    def sample_admin_user(self):
        """Create sample admin user for testing."""
        return User(
            user_id="admin_user_123",
            username="adminuser",
            email="admin@example.com",
            roles=["admin"],
            is_active=True
        )
    
    @pytest.fixture
    def sample_permission(self):
        """Create sample permission for testing."""
        return Permission(
            id="test_permission",
            resource_type=ResourceType.API,
            action="read",
            permission_level=PermissionLevel.READ
        )
    
    @pytest.fixture
    def sample_role(self, sample_permission):
        """Create sample role for testing."""
        role = Role(
            id="test_role",
            name="Test Role",
            description="Test role for testing"
        )
        role.permissions.add(sample_permission)
        return role


class TestAuthentication:
    """Test authentication functionality."""
    
    @pytest.mark.asyncio
    async def test_create_user(self, auth_engine):
        """Test user creation."""
        success = await auth_engine.create_user(
            user_id="new_user_123",
            username="newuser",
            email="new@example.com",
            password="secure_password",
            roles=["user"]
        )
        
        assert success is True
        assert "new_user_123" in auth_engine._users
        
        user = auth_engine._users["new_user_123"]
        assert user.username == "newuser"
        assert user.email == "new@example.com"
        assert "user" in user.roles
        assert user.is_active is True
        assert user.password_hash is not None
    
    @pytest.mark.asyncio
    async def test_create_duplicate_user(self, auth_engine, sample_user):
        """Test creating duplicate user fails."""
        auth_engine._users[sample_user.user_id] = sample_user
        
        success = await auth_engine.create_user(
            user_id=sample_user.user_id,
            username="duplicate",
            email="duplicate@example.com"
        )
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_create_user_invalid_role(self, auth_engine):
        """Test creating user with invalid role fails."""
        success = await auth_engine.create_user(
            user_id="invalid_role_user",
            username="invaliduser",
            roles=["nonexistent_role"]
        )
        
        assert success is False
    
    @pytest.mark.asyncio
    async def test_authenticate_user_success(self, auth_engine):
        """Test successful user authentication."""
        # Create user with password
        password = "test_password"
        await auth_engine.create_user(
            user_id="auth_test_user",
            username="authtest",
            password=password
        )
        
        with patch.object(auth_engine, '_verify_password', return_value=True):
            token = await auth_engine.authenticate_user(
                username="authtest",
                password=password,
                client_ip="127.0.0.1"
            )
        
        assert token is not None
        assert token.user_id == "auth_test_user"
        assert token.method == AuthenticationMethod.JWT
        assert token.client_ip == "127.0.0.1"
        assert token.token_id in auth_engine._active_tokens
    
    @pytest.mark.asyncio
    async def test_authenticate_user_invalid_password(self, auth_engine):
        """Test authentication with invalid password."""
        await auth_engine.create_user(
            user_id="auth_test_user",
            username="authtest",
            password="correct_password"
        )
        
        with patch.object(auth_engine, '_verify_password', return_value=False):
            token = await auth_engine.authenticate_user(
                username="authtest",
                password="wrong_password"
            )
        
        assert token is None
        assert "authtest" in auth_engine._failed_attempts
        assert auth_engine._failed_attempts["authtest"] == 1
    
    @pytest.mark.asyncio
    async def test_authenticate_nonexistent_user(self, auth_engine):
        """Test authentication with nonexistent user."""
        token = await auth_engine.authenticate_user(
            username="nonexistent",
            password="password"
        )
        
        assert token is None
    
    @pytest.mark.asyncio
    async def test_validate_token_success(self, auth_engine, sample_user):
        """Test successful token validation."""
        auth_engine._users[sample_user.user_id] = sample_user
        
        # Create a token
        token = auth_engine._generate_token(sample_user, AuthenticationMethod.JWT)
        auth_engine._active_tokens[token.token_id] = token
        
        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = {
                "user_id": sample_user.user_id,
                "username": sample_user.username
            }
            
            validated_user = await auth_engine.validate_token(token.token_id)
        
        assert validated_user is not None
        assert validated_user.user_id == sample_user.user_id
    
    @pytest.mark.asyncio
    async def test_validate_expired_token(self, auth_engine, sample_user):
        """Test validation of expired token."""
        auth_engine._users[sample_user.user_id] = sample_user
        
        # Create expired token
        token = auth_engine._generate_token(sample_user, AuthenticationMethod.JWT)
        token.expires_at = datetime.utcnow() - timedelta(hours=1)  # Expired
        auth_engine._active_tokens[token.token_id] = token
        
        with patch('jwt.decode') as mock_decode:
            mock_decode.side_effect = Exception("Token expired")
            
            validated_user = await auth_engine.validate_token(token.token_id)
        
        assert validated_user is None
    
    @pytest.mark.asyncio
    async def test_validate_revoked_token(self, auth_engine, sample_user):
        """Test validation of revoked token."""
        auth_engine._users[sample_user.user_id] = sample_user
        
        token = auth_engine._generate_token(sample_user, AuthenticationMethod.JWT)
        auth_engine._revoked_tokens.add(token.token_id)
        
        validated_user = await auth_engine.validate_token(token.token_id)
        
        assert validated_user is None


class TestAuthorization:
    """Test authorization functionality."""
    
    @pytest.mark.asyncio
    async def test_check_permission_granted(self, auth_engine, sample_user, sample_permission):
        """Test permission check that should be granted."""
        # Setup user with permission
        auth_engine._users[sample_user.user_id] = sample_user
        sample_user.direct_permissions.append(sample_permission)
        
        context = AuthorizationContext(
            user_id=sample_user.user_id,
            resource_type=ResourceType.API,
            action="read",
            permission_level=PermissionLevel.READ
        )
        
        with patch.object(auth_engine, '_validate_security_context') as mock_security:
            mock_security.return_value = SecurityValidationResult(
                is_valid=True,
                threat_level=ThreatLevel.SAFE,
                threats_detected=[]
            )
            
            with patch.object(auth_engine, '_check_rate_limits', return_value=True):
                result = await auth_engine.check_permission(context)
        
        assert result.decision == AccessDecision.GRANTED
        assert "direct_permission" in result.matched_roles
        assert result.threat_level == ThreatLevel.SAFE
    
    @pytest.mark.asyncio
    async def test_check_permission_denied_no_permission(self, auth_engine, sample_user):
        """Test permission check that should be denied due to lack of permission."""
        auth_engine._users[sample_user.user_id] = sample_user
        
        context = AuthorizationContext(
            user_id=sample_user.user_id,
            resource_type=ResourceType.ADMIN,
            action="delete",
            permission_level=PermissionLevel.ADMIN
        )
        
        with patch.object(auth_engine, '_validate_security_context') as mock_security:
            mock_security.return_value = SecurityValidationResult(
                is_valid=True,
                threat_level=ThreatLevel.SAFE,
                threats_detected=[]
            )
            
            with patch.object(auth_engine, '_check_rate_limits', return_value=True):
                result = await auth_engine.check_permission(context)
        
        assert result.decision == AccessDecision.DENIED
        assert result.reason == "No matching permissions found"
        assert len(result.matched_roles) == 0
    
    @pytest.mark.asyncio
    async def test_check_permission_denied_security_threat(self, auth_engine, sample_user):
        """Test permission check denied due to security threat."""
        auth_engine._users[sample_user.user_id] = sample_user
        
        context = AuthorizationContext(
            user_id=sample_user.user_id,
            resource_type=ResourceType.API,
            action="read"
        )
        
        with patch.object(auth_engine, '_validate_security_context') as mock_security:
            mock_security.return_value = SecurityValidationResult(
                is_valid=False,
                threat_level=ThreatLevel.HIGH,
                threats_detected=["sql_injection_detected"],
                blocked_reason="SQL injection attempt detected"
            )
            
            result = await auth_engine.check_permission(context)
        
        assert result.decision == AccessDecision.DENIED
        assert "Security threat detected" in result.reason
        assert result.threat_level == ThreatLevel.HIGH
    
    @pytest.mark.asyncio
    async def test_check_permission_denied_rate_limit(self, auth_engine, sample_user, sample_permission):
        """Test permission check denied due to rate limiting."""
        auth_engine._users[sample_user.user_id] = sample_user
        sample_user.direct_permissions.append(sample_permission)
        
        context = AuthorizationContext(
            user_id=sample_user.user_id,
            resource_type=ResourceType.API,
            action="read",
            client_ip="127.0.0.1"
        )
        
        with patch.object(auth_engine, '_validate_security_context') as mock_security:
            mock_security.return_value = SecurityValidationResult(
                is_valid=True,
                threat_level=ThreatLevel.SAFE,
                threats_detected=[]
            )
            
            with patch.object(auth_engine, '_check_rate_limits', return_value=False):
                result = await auth_engine.check_permission(context)
        
        assert result.decision == AccessDecision.DENIED
        assert "Rate limit exceeded" in result.reason
        assert result.threat_level == ThreatLevel.MEDIUM
    
    @pytest.mark.asyncio
    async def test_check_permission_role_based(self, auth_engine, sample_user, sample_role, sample_permission):
        """Test role-based permission checking."""
        # Setup user with role
        auth_engine._users[sample_user.user_id] = sample_user
        auth_engine._roles[sample_role.id] = sample_role
        sample_user.roles = [sample_role.id]
        
        context = AuthorizationContext(
            user_id=sample_user.user_id,
            resource_type=ResourceType.API,
            action="read",
            permission_level=PermissionLevel.READ
        )
        
        with patch.object(auth_engine, '_validate_security_context') as mock_security:
            mock_security.return_value = SecurityValidationResult(
                is_valid=True,
                threat_level=ThreatLevel.SAFE,
                threats_detected=[]
            )
            
            with patch.object(auth_engine, '_check_rate_limits', return_value=True):
                with patch.object(auth_engine, '_get_role_permissions_with_inheritance') as mock_perms:
                    mock_perms.return_value = {sample_permission}
                    
                    result = await auth_engine.check_permission(context)
        
        assert result.decision == AccessDecision.GRANTED
        assert sample_role.id in result.matched_roles


class TestSecurityValidation:
    """Test security validation functionality."""
    
    @pytest.mark.asyncio
    async def test_security_validation_safe_content(self, auth_engine):
        """Test security validation with safe content."""
        context = AuthorizationContext(
            user_id="test_user",
            resource_type=ResourceType.API,
            action="read",
            request_path="/api/safe/endpoint"
        )
        
        result = await auth_engine._validate_security_context(context)
        
        assert result.is_valid is True
        assert result.threat_level == ThreatLevel.SAFE
        assert len(result.threats_detected) == 0
    
    @pytest.mark.asyncio
    async def test_security_validation_sql_injection(self, auth_engine):
        """Test detection of SQL injection attempts."""
        context = AuthorizationContext(
            user_id="test_user",
            resource_type=ResourceType.API,
            action="' OR 1=1 --",  # SQL injection in action
            request_path="/api/endpoint"
        )
        
        result = await auth_engine._validate_security_context(context)
        
        assert result.is_valid is False
        assert result.threat_level == ThreatLevel.HIGH
        assert any("sql_injection" in threat for threat in result.threats_detected)
    
    @pytest.mark.asyncio
    async def test_security_validation_xss_attempt(self, auth_engine):
        """Test detection of XSS attempts."""
        context = AuthorizationContext(
            user_id="test_user",
            resource_type=ResourceType.API,
            resource_id="<script>alert('xss')</script>",  # XSS in resource_id
            action="read"
        )
        
        result = await auth_engine._validate_security_context(context)
        
        assert result.is_valid is False
        assert result.threat_level == ThreatLevel.HIGH
        assert any("xss" in threat for threat in result.threats_detected)
    
    @pytest.mark.asyncio
    async def test_security_validation_blocked_ip(self, auth_engine):
        """Test validation with blocked IP."""
        auth_engine._blocked_ips.add("192.168.1.100")
        
        context = AuthorizationContext(
            user_id="test_user",
            resource_type=ResourceType.API,
            action="read",
            client_ip="192.168.1.100"
        )
        
        result = await auth_engine._validate_security_context(context)
        
        assert result.is_valid is False
        assert result.threat_level == ThreatLevel.CRITICAL
        assert "blocked_ip" in result.threats_detected


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting_within_limit(self, auth_engine):
        """Test rate limiting when within limits."""
        with patch.object(auth_engine.redis, 'incr') as mock_incr:
            with patch.object(auth_engine.redis, 'expire') as mock_expire:
                mock_incr.return_value = 50  # Within limit
                
                result = await auth_engine._check_rate_limits("127.0.0.1")
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_rate_limiting_exceeds_minute_limit(self, auth_engine):
        """Test rate limiting when minute limit is exceeded."""
        with patch.object(auth_engine.redis, 'incr') as mock_incr:
            with patch.object(auth_engine.redis, 'expire') as mock_expire:
                # Return value higher than minute limit
                mock_incr.side_effect = [150, 50]  # minute exceeds, hour within
                
                result = await auth_engine._check_rate_limits("127.0.0.1")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_rate_limiting_exceeds_hour_limit(self, auth_engine):
        """Test rate limiting when hour limit is exceeded."""
        with patch.object(auth_engine.redis, 'incr') as mock_incr:
            with patch.object(auth_engine.redis, 'expire') as mock_expire:
                # Return values: minute ok, hour exceeds
                mock_incr.side_effect = [50, 1500]  # minute within, hour exceeds
                
                result = await auth_engine._check_rate_limits("127.0.0.1")
        
        assert result is False


class TestDecorators:
    """Test authorization decorators."""
    
    def test_require_permission_decorator(self):
        """Test require_permission decorator."""
        
        @require_permission(ResourceType.API, "read", PermissionLevel.READ)
        async def protected_function(current_user=None, request=None):
            return "success"
        
        # Test function has decorator applied
        assert hasattr(protected_function, '__wrapped__')
    
    def test_require_role_decorator(self):
        """Test require_role decorator."""
        
        @require_role("admin")
        async def admin_function(current_user=None):
            return "admin_success"
        
        # Test function has decorator applied
        assert hasattr(admin_function, '__wrapped__')


class TestSecurityIntegration:
    """Test security integration functionality."""
    
    @pytest.fixture
    def security_integration(self):
        """Create security integration for testing."""
        return SecurityOrchestrationIntegration()
    
    @pytest.mark.asyncio
    async def test_secure_agent_registration_success(self, security_integration, sample_admin_user):
        """Test successful secure agent registration."""
        agent_config = {
            "type": "worker",
            "security_level": "standard",
            "capabilities": ["task_execution"]
        }
        
        with patch.object(security_integration.auth_engine, 'check_permission') as mock_check:
            mock_check.return_value = AuthorizationResult(
                decision=AccessDecision.GRANTED,
                reason="Permission granted"
            )
            
            with patch.object(security_integration, '_validate_agent_security_config') as mock_validate:
                mock_validate.return_value = {"is_valid": True}
                
                with patch('app.core.production_orchestrator.get_production_orchestrator') as mock_orchestrator:
                    mock_orchestrator.return_value.register_agent = AsyncMock(
                        return_value={"agent_id": "new_agent_123"}
                    )
                    
                    result = await security_integration.secure_agent_registration(
                        agent_config, sample_admin_user
                    )
        
        assert result["success"] is True
        assert "agent_id" in result
    
    @pytest.mark.asyncio
    async def test_secure_agent_registration_permission_denied(self, security_integration, sample_user):
        """Test agent registration with insufficient permissions."""
        agent_config = {
            "type": "worker",
            "security_level": "standard", 
            "capabilities": ["task_execution"]
        }
        
        with patch.object(security_integration.auth_engine, 'check_permission') as mock_check:
            mock_check.return_value = AuthorizationResult(
                decision=AccessDecision.DENIED,
                reason="Insufficient permissions"
            )
            
            with pytest.raises(PermissionError):
                await security_integration.secure_agent_registration(
                    agent_config, sample_user
                )
    
    @pytest.mark.asyncio
    async def test_validate_agent_security_config_valid(self, security_integration):
        """Test validation of valid agent security configuration."""
        agent_config = {
            "type": "worker",
            "security_level": "standard",
            "capabilities": ["task_execution", "data_processing"]
        }
        
        result = await security_integration._validate_agent_security_config(agent_config)
        
        assert result["is_valid"] is True
        assert result["security_level"] == "standard"
    
    @pytest.mark.asyncio
    async def test_validate_agent_security_config_missing_fields(self, security_integration):
        """Test validation with missing required fields."""
        agent_config = {
            "type": "worker"
            # Missing security_level and capabilities
        }
        
        result = await security_integration._validate_agent_security_config(agent_config)
        
        assert result["is_valid"] is False
        assert "Missing required security fields" in result["reason"]
    
    @pytest.mark.asyncio
    async def test_validate_agent_security_config_high_risk_low_security(self, security_integration):
        """Test validation of high-risk capabilities with low security level."""
        agent_config = {
            "type": "worker",
            "security_level": "low",
            "capabilities": ["system_access", "network_access"]  # High-risk capabilities
        }
        
        result = await security_integration._validate_agent_security_config(agent_config)
        
        assert result["is_valid"] is False
        assert "High-risk capabilities require high or critical security level" in result["reason"]
    
    def test_assess_task_complexity(self, security_integration):
        """Test task complexity assessment."""
        # Low complexity task
        low_complexity_config = {
            "type": "simple_task",
            "resources": {"memory_mb": 100, "cpu_cores": 1},
            "estimated_duration_minutes": 5
        }
        
        complexity = security_integration._assess_task_complexity(low_complexity_config)
        assert complexity == "low"
        
        # High complexity task
        high_complexity_config = {
            "type": "ml_training",
            "resources": {"memory_mb": 1000, "cpu_cores": 4},
            "estimated_duration_minutes": 120
        }
        
        complexity = security_integration._assess_task_complexity(high_complexity_config)
        assert complexity == "high"
    
    def test_get_required_permission_level(self, security_integration):
        """Test getting required permission level based on complexity."""
        assert security_integration._get_required_permission_level("low") == PermissionLevel.WRITE
        assert security_integration._get_required_permission_level("medium") == PermissionLevel.EXECUTE
        assert security_integration._get_required_permission_level("high") == PermissionLevel.ADMIN
    
    def test_get_metrics_permission_level(self, security_integration):
        """Test getting required permission level for metrics access."""
        assert security_integration._get_metrics_permission_level("security") == PermissionLevel.ADMIN
        assert security_integration._get_metrics_permission_level("performance") == PermissionLevel.EXECUTE
        assert security_integration._get_metrics_permission_level("basic") == PermissionLevel.READ


class TestMetrics:
    """Test metrics and monitoring functionality."""
    
    @pytest.mark.asyncio
    async def test_get_security_metrics(self, auth_engine):
        """Test getting security metrics."""
        # Simulate some activity
        auth_engine.metrics["total_authentications"] = 100
        auth_engine.metrics["successful_authentications"] = 95
        auth_engine.metrics["total_authorizations"] = 200
        auth_engine.metrics["granted_authorizations"] = 180
        auth_engine.metrics["cache_hits"] = 150
        auth_engine.metrics["cache_misses"] = 50
        
        metrics = await auth_engine.get_security_metrics()
        
        assert "authentication_metrics" in metrics
        assert "authorization_metrics" in metrics
        assert "security_metrics" in metrics
        assert "cache_metrics" in metrics
        assert "system_status" in metrics
        
        # Check calculated rates
        assert metrics["authentication_metrics"]["success_rate"] == 0.95
        assert metrics["authorization_metrics"]["success_rate"] == 0.9
        assert metrics["cache_metrics"]["cache_hit_rate"] == 0.75


class TestPerformance:
    """Test performance aspects of the authorization engine."""
    
    @pytest.mark.asyncio
    async def test_authorization_performance(self, auth_engine, sample_user, sample_permission):
        """Test that authorization checks complete within performance targets."""
        auth_engine._users[sample_user.user_id] = sample_user
        sample_user.direct_permissions.append(sample_permission)
        
        context = AuthorizationContext(
            user_id=sample_user.user_id,
            resource_type=ResourceType.API,
            action="read"
        )
        
        with patch.object(auth_engine, '_validate_security_context') as mock_security:
            mock_security.return_value = SecurityValidationResult(
                is_valid=True,
                threat_level=ThreatLevel.SAFE,
                threats_detected=[]
            )
            
            with patch.object(auth_engine, '_check_rate_limits', return_value=True):
                start_time = time.time()
                result = await auth_engine.check_permission(context)
                end_time = time.time()
        
        # Should complete within 100ms (performance target)
        execution_time_ms = (end_time - start_time) * 1000
        assert execution_time_ms < 100
        
        assert result.decision == AccessDecision.GRANTED


# Integration Tests
class TestEndToEndScenarios:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_auth_flow(self, auth_engine):
        """Test complete authentication and authorization flow."""
        # 1. Create user
        success = await auth_engine.create_user(
            user_id="e2e_user",
            username="e2euser",
            password="secure_password"
        )
        assert success is True
        
        # 2. Authenticate user
        with patch.object(auth_engine, '_verify_password', return_value=True):
            token = await auth_engine.authenticate_user(
                username="e2euser",
                password="secure_password"
            )
        assert token is not None
        
        # 3. Validate token
        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = {
                "user_id": "e2e_user",
                "username": "e2euser"
            }
            
            validated_user = await auth_engine.validate_token(token.token_id)
        assert validated_user is not None
        
        # 4. Check permissions
        context = AuthorizationContext(
            user_id="e2e_user",
            resource_type=ResourceType.API,
            action="read"
        )
        
        with patch.object(auth_engine, '_validate_security_context') as mock_security:
            mock_security.return_value = SecurityValidationResult(
                is_valid=True,
                threat_level=ThreatLevel.SAFE,
                threats_detected=[]
            )
            
            with patch.object(auth_engine, '_check_rate_limits', return_value=True):
                result = await auth_engine.check_permission(context)
        
        # Should be denied due to no permissions
        assert result.decision == AccessDecision.DENIED
    
    @pytest.mark.asyncio
    async def test_security_threat_handling(self, auth_engine, sample_user):
        """Test that security threats are properly detected and handled."""
        auth_engine._users[sample_user.user_id] = sample_user
        
        # Create context with malicious content
        context = AuthorizationContext(
            user_id=sample_user.user_id,
            resource_type=ResourceType.API,
            action="'; DROP TABLE users; --",  # SQL injection
            client_ip="192.168.1.100"
        )
        
        result = await auth_engine.check_permission(context)
        
        # Should be denied due to security threat
        assert result.decision == AccessDecision.DENIED
        assert "Security threat detected" in result.reason
        assert result.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__ + "::TestAuthentication::test_create_user", "-v"])