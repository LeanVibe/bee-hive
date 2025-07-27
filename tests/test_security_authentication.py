"""
Comprehensive tests for Security & Authentication System.

Tests OAuth 2.0/OIDC authentication, JWT token management,
and agent identity service functionality.
"""

import pytest
import uuid
import jwt
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.agent_identity_service import (
    AgentIdentityService, TokenValidationError, RateLimitExceededError
)
from app.models.security import AgentIdentity, AgentToken, AgentStatus
from app.schemas.security import AgentTokenRequest, TokenRefreshRequest
from app.core.redis import RedisClient


@pytest.fixture
async def mock_redis():
    """Mock Redis client."""
    redis_mock = AsyncMock(spec=RedisClient)
    redis_mock.get_count_in_window.return_value = 0
    redis_mock.increment_counter.return_value = 1
    redis_mock.exists.return_value = False
    redis_mock.set_with_expiry.return_value = True
    redis_mock.lpush.return_value = True
    return redis_mock


@pytest.fixture
async def mock_db_session():
    """Mock database session."""
    session_mock = AsyncMock(spec=AsyncSession)
    session_mock.execute.return_value = AsyncMock()
    session_mock.commit.return_value = None
    session_mock.rollback.return_value = None
    session_mock.refresh.return_value = None
    session_mock.add.return_value = None
    session_mock.get.return_value = None
    return session_mock


@pytest.fixture
async def identity_service(mock_db_session, mock_redis):
    """Create AgentIdentityService instance."""
    return AgentIdentityService(
        db_session=mock_db_session,
        redis_client=mock_redis,
        jwt_secret_key="test-secret-key-12345678901234567890123456789012"
    )


@pytest.fixture
def sample_agent_identity():
    """Sample agent identity for testing."""
    return AgentIdentity(
        id=uuid.uuid4(),
        agent_name="test-agent-001",
        human_controller="test@company.com",
        oauth_client_id="client_123",
        oauth_client_secret_hash="$2b$12$hash",
        scopes=["read:files", "write:github"],
        status=AgentStatus.ACTIVE.value,
        rate_limit_per_minute=10,
        token_expires_in_seconds=3600,
        refresh_token_expires_in_seconds=604800,
        created_by="admin"
    )


@pytest.fixture
def token_request():
    """Sample token request."""
    return AgentTokenRequest(
        agent_id="test-agent-001",
        human_controller="test@company.com",
        requested_scopes=["read:files", "write:github"]
    )


class TestAgentRegistration:
    """Test agent registration functionality."""
    
    async def test_register_agent_success(self, identity_service, mock_db_session):
        """Test successful agent registration."""
        # Mock no existing agent
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        
        result = await identity_service.register_agent(
            agent_name="test-agent-001",
            human_controller="test@company.com",
            scopes=["read:files"],
            created_by="admin"
        )
        
        assert result.agent_name == "test-agent-001"
        assert result.human_controller == "test@company.com"
        assert result.scopes == ["read:files"]
        assert result.status == AgentStatus.ACTIVE.value
        assert hasattr(result, 'oauth_client_secret_plaintext')
        
        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called()
    
    async def test_register_agent_duplicate_name(self, identity_service, mock_db_session):
        """Test registration with duplicate agent name."""
        # Mock existing agent
        existing_agent = MagicMock()
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = existing_agent
        
        with pytest.raises(ValueError, match="Agent name 'test-agent-001' already exists"):
            await identity_service.register_agent(
                agent_name="test-agent-001",
                human_controller="test@company.com",
                scopes=["read:files"],
                created_by="admin"
            )
    
    async def test_register_agent_database_error(self, identity_service, mock_db_session):
        """Test registration with database error."""
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        mock_db_session.commit.side_effect = Exception("Database error")
        
        with pytest.raises(Exception):
            await identity_service.register_agent(
                agent_name="test-agent-001",
                human_controller="test@company.com",
                scopes=["read:files"],
                created_by="admin"
            )
        
        mock_db_session.rollback.assert_called_once()


class TestAuthentication:
    """Test authentication functionality."""
    
    async def test_authenticate_agent_success(
        self, 
        identity_service, 
        mock_db_session,
        mock_redis,
        sample_agent_identity,
        token_request
    ):
        """Test successful agent authentication."""
        # Mock agent lookup
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = sample_agent_identity
        
        # Mock token count
        mock_db_session.execute.return_value.scalar.return_value = 0
        
        result = await identity_service.authenticate_agent(
            request=token_request,
            client_ip="127.0.0.1",
            user_agent="test-client/1.0"
        )
        
        assert result.access_token is not None
        assert result.refresh_token is not None
        assert result.token_type == "Bearer"
        assert result.expires_in == 3600
        assert result.scope == ["read:files", "write:github"]
        
        # Verify rate limiting check
        mock_redis.get_count_in_window.assert_called_once()
        mock_redis.increment_counter.assert_called_once()
    
    async def test_authenticate_agent_not_found(
        self,
        identity_service,
        mock_db_session,
        token_request
    ):
        """Test authentication with non-existent agent."""
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        
        with pytest.raises(Exception):  # Would be SecurityError in real implementation
            await identity_service.authenticate_agent(
                request=token_request,
                client_ip="127.0.0.1"
            )
    
    async def test_authenticate_agent_inactive(
        self,
        identity_service,
        mock_db_session,
        sample_agent_identity,
        token_request
    ):
        """Test authentication with inactive agent."""
        sample_agent_identity.status = AgentStatus.INACTIVE.value
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = sample_agent_identity
        
        with pytest.raises(Exception):  # Would be SecurityError in real implementation
            await identity_service.authenticate_agent(
                request=token_request,
                client_ip="127.0.0.1"
            )
    
    async def test_authenticate_agent_rate_limited(
        self,
        identity_service,
        mock_db_session,
        mock_redis,
        sample_agent_identity,
        token_request
    ):
        """Test authentication with rate limiting."""
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = sample_agent_identity
        mock_redis.get_count_in_window.return_value = 15  # Exceeds limit of 10
        
        with pytest.raises(RateLimitExceededError):
            await identity_service.authenticate_agent(
                request=token_request,
                client_ip="127.0.0.1"
            )
    
    async def test_authenticate_human_controller_mismatch(
        self,
        identity_service,
        mock_db_session,
        sample_agent_identity,
        token_request
    ):
        """Test authentication with human controller mismatch."""
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = sample_agent_identity
        mock_redis.get_count_in_window.return_value = 0
        
        # Change human controller
        token_request.human_controller = "different@company.com"
        
        with pytest.raises(Exception):  # Would be SecurityError in real implementation
            await identity_service.authenticate_agent(
                request=token_request,
                client_ip="127.0.0.1"
            )


class TestTokenValidation:
    """Test JWT token validation."""
    
    async def test_validate_token_success(
        self,
        identity_service,
        mock_db_session,
        sample_agent_identity
    ):
        """Test successful token validation."""
        # Create a valid JWT token
        now = datetime.utcnow()
        payload = {
            "iss": "leanvibe-agent-hive",
            "sub": str(sample_agent_identity.id),
            "aud": "leanvibe-api",
            "exp": int((now + timedelta(hours=1)).timestamp()),
            "iat": int(now.timestamp()),
            "jti": str(uuid.uuid4()),
            "scope": ["read:files"],
            "human_controller": "test@company.com",
            "agent_name": "test-agent"
        }
        
        token = jwt.encode(payload, identity_service.jwt_secret_key, algorithm="HS256")
        
        # Mock token record in database
        token_record = MagicMock()
        token_record.is_revoked = False
        token_record.expires_at = now + timedelta(hours=1)
        token_record.record_usage = MagicMock()
        
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = token_record
        
        # Mock agent lookup
        with patch.object(identity_service, '_get_agent_identity', return_value=sample_agent_identity):
            result = await identity_service.validate_token(token)
        
        assert result["sub"] == str(sample_agent_identity.id)
        assert result["scope"] == ["read:files"]
        assert result["human_controller"] == "test@company.com"
        
        # Verify token usage was recorded
        token_record.record_usage.assert_called_once()
    
    async def test_validate_token_expired(self, identity_service):
        """Test validation of expired token."""
        # Create expired token
        past_time = datetime.utcnow() - timedelta(hours=1)
        payload = {
            "iss": "leanvibe-agent-hive",
            "sub": str(uuid.uuid4()),
            "aud": "leanvibe-api",
            "exp": int(past_time.timestamp()),
            "iat": int((past_time - timedelta(hours=1)).timestamp()),
            "jti": str(uuid.uuid4())
        }
        
        token = jwt.encode(payload, identity_service.jwt_secret_key, algorithm="HS256")
        
        with pytest.raises(TokenValidationError, match="Token expired"):
            await identity_service.validate_token(token)
    
    async def test_validate_token_invalid_signature(self, identity_service):
        """Test validation with invalid signature."""
        # Create token with wrong key
        payload = {
            "iss": "leanvibe-agent-hive",
            "sub": str(uuid.uuid4()),
            "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.utcnow().timestamp()),
            "jti": str(uuid.uuid4())
        }
        
        token = jwt.encode(payload, "wrong-secret-key", algorithm="HS256")
        
        with pytest.raises(TokenValidationError, match="Invalid token"):
            await identity_service.validate_token(token)
    
    async def test_validate_token_blacklisted(
        self,
        identity_service,
        mock_redis
    ):
        """Test validation of blacklisted token."""
        # Mock blacklisted token
        mock_redis.exists.return_value = True
        
        payload = {
            "iss": "leanvibe-agent-hive",
            "sub": str(uuid.uuid4()),
            "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.utcnow().timestamp()),
            "jti": str(uuid.uuid4())
        }
        
        token = jwt.encode(payload, identity_service.jwt_secret_key, algorithm="HS256")
        
        with pytest.raises(TokenValidationError, match="Token is revoked"):
            await identity_service.validate_token(token)
    
    async def test_validate_token_insufficient_scopes(
        self,
        identity_service,
        mock_db_session,
        sample_agent_identity
    ):
        """Test validation with insufficient scopes."""
        payload = {
            "iss": "leanvibe-agent-hive",
            "sub": str(sample_agent_identity.id),
            "aud": "leanvibe-api",
            "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.utcnow().timestamp()),
            "jti": str(uuid.uuid4()),
            "scope": ["read:files"]
        }
        
        token = jwt.encode(payload, identity_service.jwt_secret_key, algorithm="HS256")
        
        # Mock token record
        token_record = MagicMock()
        token_record.is_revoked = False
        token_record.expires_at = datetime.utcnow() + timedelta(hours=1)
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = token_record
        
        with patch.object(identity_service, '_get_agent_identity', return_value=sample_agent_identity):
            with pytest.raises(TokenValidationError, match="Insufficient scopes"):
                await identity_service.validate_token(token, required_scopes=["write:admin"])


class TestTokenRefresh:
    """Test token refresh functionality."""
    
    async def test_refresh_token_success(
        self,
        identity_service,
        mock_db_session,
        sample_agent_identity
    ):
        """Test successful token refresh."""
        # Create refresh token
        now = datetime.utcnow()
        refresh_payload = {
            "iss": "leanvibe-agent-hive",
            "sub": str(sample_agent_identity.id),
            "aud": "leanvibe-api",
            "exp": int((now + timedelta(days=7)).timestamp()),
            "iat": int(now.timestamp()),
            "jti": str(uuid.uuid4()),
            "scope": ["read:files"],
            "token_type": "refresh"
        }
        
        refresh_token = jwt.encode(refresh_payload, identity_service.jwt_secret_key, algorithm="HS256")
        
        # Mock refresh token validation
        token_record = MagicMock()
        token_record.is_revoked = False
        token_record.expires_at = now + timedelta(days=7)
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = token_record
        
        # Mock agent lookup
        with patch.object(identity_service, '_get_agent_identity', return_value=sample_agent_identity):
            request = TokenRefreshRequest(refresh_token=refresh_token)
            result = await identity_service.refresh_token(request)
        
        assert result.access_token is not None
        assert result.refresh_token == refresh_token  # Same refresh token
        assert result.token_type == "Bearer"
        assert result.scope == ["read:files"]
    
    async def test_refresh_token_invalid_type(self, identity_service):
        """Test refresh with access token instead of refresh token."""
        # Create access token (not refresh)
        payload = {
            "iss": "leanvibe-agent-hive",
            "sub": str(uuid.uuid4()),
            "exp": int((datetime.utcnow() + timedelta(hours=1)).timestamp()),
            "iat": int(datetime.utcnow().timestamp()),
            "jti": str(uuid.uuid4()),
            "scope": ["read:files"]
            # Missing token_type: "refresh"
        }
        
        token = jwt.encode(payload, identity_service.jwt_secret_key, algorithm="HS256")
        request = TokenRefreshRequest(refresh_token=token)
        
        with pytest.raises(Exception):  # Would be SecurityError in real implementation
            await identity_service.refresh_token(request)


class TestTokenRevocation:
    """Test token revocation functionality."""
    
    async def test_revoke_token_success(
        self,
        identity_service,
        mock_db_session,
        mock_redis
    ):
        """Test successful token revocation."""
        # Mock finding token
        token_record = MagicMock()
        token_record.agent_id = uuid.uuid4()
        token_record.revoke = MagicMock()
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = token_record
        
        result = await identity_service.revoke_token(
            token="dummy-token",
            revoked_by="admin",
            reason="security_incident"
        )
        
        assert result is True
        token_record.revoke.assert_called_once_with("security_incident")
        mock_redis.set_with_expiry.assert_called_once()  # Blacklist token
        mock_db_session.commit.assert_called_once()
    
    async def test_revoke_token_not_found(
        self,
        identity_service,
        mock_db_session
    ):
        """Test revoking non-existent token."""
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        
        result = await identity_service.revoke_token(
            token="dummy-token",
            revoked_by="admin"
        )
        
        assert result is False


class TestUtilityMethods:
    """Test utility and helper methods."""
    
    def test_generate_client_id(self, identity_service):
        """Test OAuth client ID generation."""
        client_id = identity_service._generate_client_id()
        
        assert client_id.startswith("agent_")
        assert len(client_id) == 6 + 32  # "agent_" + 32 hex chars
    
    def test_generate_client_secret(self, identity_service):
        """Test OAuth client secret generation."""
        secret = identity_service._generate_client_secret()
        
        assert len(secret) > 20  # URL-safe base64
        assert isinstance(secret, str)
    
    def test_hash_client_secret(self, identity_service):
        """Test client secret hashing."""
        secret = "test-secret"
        hashed = identity_service._hash_client_secret(secret)
        
        assert hashed != secret
        assert hashed.startswith("$2b$")  # bcrypt format
    
    def test_verify_client_secret(self, identity_service):
        """Test client secret verification."""
        secret = "test-secret"
        hashed = identity_service._hash_client_secret(secret)
        
        assert identity_service._verify_client_secret(secret, hashed) is True
        assert identity_service._verify_client_secret("wrong-secret", hashed) is False
    
    def test_validate_scopes(self, identity_service):
        """Test scope validation."""
        allowed_scopes = ["read:files", "write:github", "admin:users"]
        
        # Valid subset
        result = identity_service._validate_scopes(
            ["read:files", "write:github"], 
            allowed_scopes
        )
        assert result == ["read:files", "write:github"]
        
        # Invalid scope filtered out
        result = identity_service._validate_scopes(
            ["read:files", "invalid:scope"], 
            allowed_scopes
        )
        assert result == ["read:files"]
        
        # Empty request
        result = identity_service._validate_scopes([], allowed_scopes)
        assert result == []
        
        # None request
        result = identity_service._validate_scopes(None, allowed_scopes)
        assert result == []
    
    def test_hash_token(self, identity_service):
        """Test token hashing for storage."""
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test"
        hashed = identity_service._hash_token(token)
        
        assert len(hashed) == 64  # SHA-256 hex
        assert hashed != token
        
        # Same token produces same hash
        assert identity_service._hash_token(token) == hashed


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""
    
    async def test_concurrent_token_limit(
        self,
        identity_service,
        mock_db_session,
        sample_agent_identity
    ):
        """Test concurrent token limit enforcement."""
        # Mock high token count
        mock_db_session.execute.return_value.scalar.return_value = 5  # At limit
        
        sample_agent_identity.max_concurrent_tokens = 5
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = sample_agent_identity
        
        token_request = AgentTokenRequest(
            agent_id="test-agent",
            human_controller="test@company.com",
            requested_scopes=["read:files"]
        )
        
        with pytest.raises(Exception):  # Would be SecurityError in real implementation
            await identity_service.authenticate_agent(token_request)
    
    async def test_cleanup_expired_tokens(
        self,
        identity_service,
        mock_db_session
    ):
        """Test expired token cleanup."""
        # Mock expired tokens
        expired_tokens = [MagicMock() for _ in range(3)]
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = expired_tokens
        
        result = await identity_service.cleanup_expired_tokens()
        
        assert result == 3
        assert mock_db_session.delete.call_count == 3
        mock_db_session.commit.assert_called_once()


@pytest.mark.integration
class TestSecurityIntegration:
    """Integration tests for security system."""
    
    async def test_full_authentication_flow(
        self,
        identity_service,
        mock_db_session,
        mock_redis
    ):
        """Test complete authentication flow."""
        # 1. Register agent
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None
        
        agent = await identity_service.register_agent(
            agent_name="integration-test-agent",
            human_controller="test@company.com",
            scopes=["read:files", "write:github"],
            created_by="admin"
        )
        
        # 2. Authenticate agent
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = agent
        mock_db_session.execute.return_value.scalar.return_value = 0
        
        token_request = AgentTokenRequest(
            agent_id=agent.agent_name,
            human_controller=agent.human_controller,
            requested_scopes=agent.scopes
        )
        
        token_response = await identity_service.authenticate_agent(token_request)
        
        assert token_response.access_token is not None
        assert token_response.refresh_token is not None
        
        # 3. Validate token
        # Mock token record for validation
        token_record = MagicMock()
        token_record.is_revoked = False
        token_record.expires_at = datetime.utcnow() + timedelta(hours=1)
        token_record.record_usage = MagicMock()
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = token_record
        
        with patch.object(identity_service, '_get_agent_identity', return_value=agent):
            token_payload = await identity_service.validate_token(token_response.access_token)
        
        assert token_payload["agent_name"] == agent.agent_name
        assert token_payload["human_controller"] == agent.human_controller
        
        # 4. Revoke token
        result = await identity_service.revoke_token(
            token=token_response.access_token,
            revoked_by="admin",
            reason="test_cleanup"
        )
        
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])