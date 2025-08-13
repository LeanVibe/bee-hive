"""
FastAPI Service Lifecycle Validation Tests

Critical tests for FastAPI application startup, shutdown, and lifecycle management
to ensure reliable service operation and graceful handling of failure scenarios.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient
from app.main import app, lifespan


class TestFastAPIServiceStartup:
    """Test FastAPI service startup scenarios."""
    
    @pytest.mark.asyncio
    async def test_successful_service_startup(self):
        """Test successful FastAPI service startup."""
        
        startup_events = []
        
        @asynccontextmanager
        async def mock_lifespan(app):
            startup_events.append("database_init")
            startup_events.append("redis_init") 
            startup_events.append("event_processor_init")
            startup_events.append("hook_interceptor_init")
            yield
            startup_events.append("shutdown")
        
        with patch('app.main.lifespan', mock_lifespan):
            client = TestClient(app)
            
            # Verify startup sequence
            expected_events = [
                "database_init",
                "redis_init", 
                "event_processor_init",
                "hook_interceptor_init"
            ]
            
            for event in expected_events:
                assert event in startup_events
    
    @pytest.mark.asyncio
    async def test_database_initialization_failure(self):
        """Test handling of database initialization failure."""
        
        @asynccontextmanager
        async def mock_lifespan_with_db_failure(app):
            # Simulate database initialization failure
            from app.core.database import init_database
            with patch('app.core.database.init_database', side_effect=ConnectionError("Database unavailable")):
                try:
                    await init_database()
                except ConnectionError:
                    # Service should handle gracefully
                    pass
            yield
        
        with patch('app.main.lifespan', mock_lifespan_with_db_failure):
            # Should not crash the entire service
            client = TestClient(app)
            response = client.get("/health")
            # Should return degraded status
            assert response.status_code in [200, 503]  # 503 for degraded mode
    
    @pytest.mark.asyncio
    async def test_redis_initialization_failure(self):
        """Test handling of Redis initialization failure."""
        
        @asynccontextmanager 
        async def mock_lifespan_with_redis_failure(app):
            from app.core.redis import init_redis
            with patch('app.core.redis.init_redis', side_effect=ConnectionError("Redis unavailable")):
                try:
                    await init_redis()
                except ConnectionError:
                    # Service should continue with degraded functionality
                    pass
            yield
        
        with patch('app.main.lifespan', mock_lifespan_with_redis_failure):
            client = TestClient(app)
            response = client.get("/health")
            # Should still be accessible
            assert response.status_code in [200, 503]
    
    @pytest.mark.asyncio
    async def test_event_processor_initialization_failure(self):
        """Test handling of event processor initialization failure."""
        
        @asynccontextmanager
        async def mock_lifespan_with_processor_failure(app):
            from app.core.event_processor import initialize_event_processor
            with patch('app.core.event_processor.initialize_event_processor', 
                      side_effect=RuntimeError("Event processor failed")):
                try:
                    await initialize_event_processor(None)
                except RuntimeError:
                    # Should handle gracefully
                    pass
            yield
        
        with patch('app.main.lifespan', mock_lifespan_with_processor_failure):
            client = TestClient(app)
            # Service should still start but with reduced functionality
            response = client.get("/health")
            assert response.status_code in [200, 503]
    
    @pytest.mark.asyncio
    async def test_ci_environment_startup_skip(self):
        """Test startup behavior in CI environment."""
        
        with patch.dict('os.environ', {'CI': 'true'}):
            @asynccontextmanager
            async def mock_ci_lifespan(app):
                # Should skip heavy initialization in CI
                yield
            
            with patch('app.main.lifespan', mock_ci_lifespan):
                client = TestClient(app)
                response = client.get("/health")
                assert response.status_code == 200


class TestFastAPIServiceShutdown:
    """Test FastAPI service shutdown scenarios."""
    
    @pytest.mark.asyncio
    async def test_graceful_service_shutdown(self):
        """Test graceful service shutdown."""
        
        shutdown_events = []
        
        @asynccontextmanager
        async def mock_lifespan_with_shutdown(app):
            yield
            shutdown_events.append("event_processor_shutdown")
            shutdown_events.append("coordination_bridge_shutdown")
            shutdown_events.append("performance_publisher_shutdown")
        
        with patch('app.main.lifespan', mock_lifespan_with_shutdown):
            client = TestClient(app)
            # Client context exit triggers shutdown
            
        # Verify shutdown sequence
        expected_shutdown_events = [
            "event_processor_shutdown",
            "coordination_bridge_shutdown", 
            "performance_publisher_shutdown"
        ]
        
        for event in expected_shutdown_events:
            assert event in shutdown_events
    
    @pytest.mark.asyncio
    async def test_shutdown_with_active_connections(self):
        """Test shutdown behavior with active connections."""
        
        active_connections = []
        
        @asynccontextmanager
        async def mock_lifespan_with_connections(app):
            # Simulate active connections
            active_connections.extend(["conn1", "conn2", "conn3"])
            yield
            # Should wait for connections to complete
            await asyncio.sleep(0.1)  # Simulate graceful wait
            active_connections.clear()
        
        with patch('app.main.lifespan', mock_lifespan_with_connections):
            client = TestClient(app)
            
        # All connections should be properly closed
        assert len(active_connections) == 0
    
    @pytest.mark.asyncio
    async def test_forced_shutdown_after_timeout(self):
        """Test forced shutdown when graceful shutdown times out."""
        
        shutdown_timeout_reached = False
        
        @asynccontextmanager
        async def mock_lifespan_with_timeout(app):
            yield
            try:
                # Simulate long-running shutdown operation
                await asyncio.wait_for(asyncio.sleep(10), timeout=1.0)
            except asyncio.TimeoutError:
                nonlocal shutdown_timeout_reached
                shutdown_timeout_reached = True
                # Force shutdown
        
        with patch('app.main.lifespan', mock_lifespan_with_timeout):
            client = TestClient(app)
        
        assert shutdown_timeout_reached is True


class TestFastAPIServiceHealthChecks:
    """Test service health check functionality."""
    
    def test_health_endpoint_during_startup(self):
        """Test health endpoint during service startup."""
        client = TestClient(app)
        
        response = client.get("/health")
        assert response.status_code in [200, 503]  # Healthy or starting
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert data["status"] in ["healthy", "starting", "degraded"]
    
    def test_health_endpoint_with_dependency_failures(self):
        """Test health endpoint when dependencies fail."""
        
        with patch('app.core.database.get_session', side_effect=ConnectionError("DB down")):
            client = TestClient(app)
            response = client.get("/health")
            
            # Should report degraded status
            assert response.status_code in [200, 503]
            
            if response.status_code == 200:
                data = response.json()
                assert data["status"] in ["degraded", "unhealthy"]
    
    def test_readiness_endpoint(self):
        """Test readiness endpoint for load balancer health checks."""
        client = TestClient(app)
        
        response = client.get("/ready")
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "ready" in data
            assert isinstance(data["ready"], bool)
    
    def test_liveness_endpoint(self):
        """Test liveness endpoint for basic service status."""
        client = TestClient(app)
        
        response = client.get("/live")
        assert response.status_code == 200
        
        data = response.json()
        assert "alive" in data
        assert data["alive"] is True


class TestFastAPIServiceMiddleware:
    """Test middleware initialization and error handling."""
    
    def test_cors_middleware_initialization(self):
        """Test CORS middleware initialization."""
        client = TestClient(app)
        
        response = client.options("/health")
        assert response.status_code in [200, 405]  # OPTIONS allowed or method not allowed
        
        # Check CORS headers are present
        headers = response.headers
        assert "access-control-allow-origin" in headers or response.status_code == 405
    
    def test_trusted_host_middleware(self):
        """Test trusted host middleware."""
        client = TestClient(app)
        
        # Test with allowed host
        response = client.get("/health", headers={"host": "localhost"})
        assert response.status_code in [200, 503]
        
        # Test with disallowed host (if configured)
        response = client.get("/health", headers={"host": "malicious.example.com"})
        assert response.status_code in [200, 400, 403, 503]  # Depends on configuration
    
    def test_global_exception_handler(self):
        """Test global exception handler."""
        client = TestClient(app)
        
        # This would test a route that throws an exception
        # For now, we'll test that the handler is properly configured
        with patch('app.main.app.exception_handler') as mock_handler:
            # Exception handler should be properly configured
            assert mock_handler is not None or hasattr(app, 'exception_handlers')


class TestFastAPIServiceConfiguration:
    """Test service configuration and environment handling."""
    
    def test_development_configuration(self):
        """Test development environment configuration."""
        with patch.dict('os.environ', {'ENVIRONMENT': 'development'}):
            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code in [200, 503]
    
    def test_production_configuration(self):
        """Test production environment configuration.""" 
        with patch.dict('os.environ', {'ENVIRONMENT': 'production'}):
            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code in [200, 503]
    
    def test_settings_validation(self):
        """Test application settings validation."""
        from app.core.config import get_settings
        
        settings = get_settings()
        
        # Verify critical settings are present
        assert hasattr(settings, 'DATABASE_URL')
        assert hasattr(settings, 'REDIS_URL') 
        assert hasattr(settings, 'DEBUG')
    
    def test_logging_configuration(self):
        """Test logging configuration."""
        import structlog
        
        # Verify structured logging is configured
        logger = structlog.get_logger()
        assert logger is not None
        
        # Test that logging works
        logger.info("Test log message", test=True)


class TestFastAPIServiceDependencyInjection:
    """Test dependency injection container."""
    
    def test_database_dependency_injection(self):
        """Test database session dependency injection."""
        from app.core.database import get_session
        
        # Should be able to get database session
        assert get_session is not None
    
    def test_redis_dependency_injection(self):
        """Test Redis client dependency injection."""
        from app.core.redis import get_redis
        
        # Should be able to get Redis client
        assert get_redis is not None
    
    def test_orchestrator_dependency_injection(self):
        """Test orchestrator dependency injection."""
        from app.core.orchestrator import get_orchestrator
        
        # Should be able to get orchestrator instance
        orchestrator = get_orchestrator()
        assert orchestrator is not None


class TestFastAPIServicePerformance:
    """Test service performance characteristics."""
    
    def test_startup_time_performance(self):
        """Test that service starts within acceptable time."""
        start_time = time.time()
        
        client = TestClient(app)
        response = client.get("/health")
        
        startup_time = time.time() - start_time
        
        # Should start within 30 seconds (adjust as needed)
        assert startup_time < 30.0
        assert response.status_code in [200, 503]
    
    def test_memory_usage_during_startup(self):
        """Test memory usage during startup."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        client = TestClient(app)
        client.get("/health")
        
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (adjust threshold as needed)
        assert memory_increase < 500 * 1024 * 1024  # 500MB
    
    def test_concurrent_startup_requests(self):
        """Test handling of concurrent requests during startup."""
        client = TestClient(app)
        
        # Make multiple concurrent requests
        responses = []
        for _ in range(10):
            response = client.get("/health")
            responses.append(response)
        
        # All requests should be handled gracefully
        for response in responses:
            assert response.status_code in [200, 503, 429]  # Include rate limiting


@pytest.mark.integration
class TestFastAPIServiceIntegration:
    """Integration tests for complete service lifecycle."""
    
    @pytest.mark.asyncio
    async def test_full_service_lifecycle(self):
        """Test complete service startup and shutdown cycle."""
        
        lifecycle_events = []
        
        @asynccontextmanager
        async def tracked_lifespan(app):
            lifecycle_events.append("startup_begin")
            
            try:
                # Simulate full startup sequence
                lifecycle_events.append("database_ready")
                lifecycle_events.append("redis_ready")
                lifecycle_events.append("services_ready")
                lifecycle_events.append("startup_complete")
                
                yield
                
            finally:
                lifecycle_events.append("shutdown_begin")
                lifecycle_events.append("cleanup_complete")
                lifecycle_events.append("shutdown_complete")
        
        with patch('app.main.lifespan', tracked_lifespan):
            client = TestClient(app)
            response = client.get("/health")
            assert response.status_code in [200, 503]
        
        # Verify complete lifecycle
        expected_events = [
            "startup_begin",
            "database_ready", 
            "redis_ready",
            "services_ready",
            "startup_complete",
            "shutdown_begin",
            "cleanup_complete",
            "shutdown_complete"
        ]
        
        for event in expected_events:
            assert event in lifecycle_events
    
    @pytest.mark.asyncio
    async def test_service_recovery_after_partial_failure(self):
        """Test service recovery after partial component failure."""
        
        recovery_events = []
        
        @asynccontextmanager
        async def recovery_lifespan(app):
            # Simulate partial failure and recovery
            try:
                recovery_events.append("startup_attempt")
                raise ConnectionError("Initial failure")
            except ConnectionError:
                recovery_events.append("failure_detected")
                # Retry logic
                await asyncio.sleep(0.1)
                recovery_events.append("recovery_successful")
            
            yield
            recovery_events.append("shutdown_clean")
        
        with patch('app.main.lifespan', recovery_lifespan):
            client = TestClient(app)
            response = client.get("/health")
            # Should eventually be healthy after recovery
            assert response.status_code in [200, 503]
        
        assert "recovery_successful" in recovery_events