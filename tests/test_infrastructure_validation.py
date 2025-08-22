"""
Infrastructure validation tests for LeanVibe Agent Hive 2.0.

Tests basic database connectivity, Redis integration, and core functionality.
"""

import pytest
import asyncio


class TestInfrastructureValidation:
    """Test infrastructure components are operational."""
    
    @pytest.mark.asyncio
    @pytest.mark.database
    async def test_database_connectivity(self):
        """Test that database connection is working."""
        from app.core.database import init_database, DatabaseHealthCheck, close_database
        
        await init_database()
        
        # Test connection
        is_connected = await DatabaseHealthCheck.check_connection()
        assert is_connected, "Database connection should be active"
        
        # Test extensions
        extensions = await DatabaseHealthCheck.check_extensions()
        assert extensions.get("pgvector") is True, "pgvector extension should be available"
        assert extensions.get("uuid-ossp") is True, "uuid-ossp extension should be available"
        
        await close_database()
    
    @pytest.mark.asyncio
    @pytest.mark.redis
    async def test_redis_connectivity(self):
        """Test that Redis connection is working."""
        import redis.asyncio as redis
        from app.core.config import settings
        
        redis_client = redis.from_url(settings.REDIS_URL)
        
        # Test basic connection
        response = await redis_client.ping()
        assert response is True, "Redis ping should return True"
        
        # Test set/get operations
        await redis_client.set('test_key', 'test_value')
        value = await redis_client.get('test_key')
        assert value.decode() == 'test_value', "Redis get/set should work correctly"
        
        # Clean up
        await redis_client.delete('test_key')
        await redis_client.aclose()
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_simple_orchestrator_initialization(self):
        """Test that SimpleOrchestrator can be initialized."""
        from app.core.simple_orchestrator import SimpleOrchestrator
        from app.core.database import init_database, close_database
        
        await init_database()
        
        orchestrator = SimpleOrchestrator()
        assert orchestrator is not None, "SimpleOrchestrator should initialize"
        
        # Test system status
        status = await orchestrator.get_system_status()
        assert isinstance(status, dict), "System status should return dict"
        assert "agents" in status, "Status should contain agents key"
        
        await close_database()
    
    @pytest.mark.unit
    def test_config_loading(self):
        """Test that configuration loads correctly."""
        from app.core.config import settings
        
        assert settings.DATABASE_URL is not None, "Database URL should be configured"
        assert settings.REDIS_URL is not None, "Redis URL should be configured"
        assert settings.ENVIRONMENT == "development", "Environment should be development"
        assert settings.DEBUG is True, "Debug should be enabled in development"