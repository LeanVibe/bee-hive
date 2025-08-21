"""
Shared fixtures for component isolation testing.

Provides isolated test environments, mocked dependencies, and performance
measurement utilities for systematic component validation.
"""

import asyncio
import os
import psutil
import time
from typing import AsyncGenerator, Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import asynccontextmanager

import pytest
import fakeredis
import redis.asyncio as redis
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Test environment configuration
TEST_DATABASE_URL = "sqlite+aioredis:///:memory:"
TEST_REDIS_URL = "redis://localhost:6379/15"  # Test database


class ComponentTestMetrics:
    """Performance and resource usage metrics for component testing."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.start_time: Optional[float] = None
        self.start_memory: Optional[int] = None
        self.metrics: Dict[str, Any] = {}
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss
        
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics."""
        if self.start_time is None:
            return {}
            
        duration = time.time() - self.start_time
        current_memory = psutil.Process().memory_info().rss
        memory_delta = current_memory - (self.start_memory or 0)
        
        self.metrics.update({
            "component": self.component_name,
            "duration_seconds": duration,
            "memory_delta_mb": memory_delta / (1024 * 1024),
            "peak_memory_mb": current_memory / (1024 * 1024)
        })
        
        return self.metrics
    
    @asynccontextmanager
    async def measure_async(self):
        """Context manager for measuring async operations."""
        self.start_monitoring()
        try:
            yield self
        finally:
            self.stop_monitoring()


@pytest.fixture
def component_metrics():
    """Factory for creating component performance metrics."""
    def _create_metrics(component_name: str) -> ComponentTestMetrics:
        return ComponentTestMetrics(component_name)
    return _create_metrics


@pytest.fixture
async def isolated_redis():
    """Isolated Redis instance for testing with fakeredis."""
    fake_redis = fakeredis.FakeAsyncRedis()
    yield fake_redis
    await fake_redis.aclose()


@pytest.fixture
async def isolated_database():
    """Isolated SQLite database for testing."""
    # Use in-memory SQLite for fast, isolated testing
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    
    # Create tables (if needed - will be component specific)
    async with engine.begin() as conn:
        # Basic health check table for testing
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS test_health (
                id INTEGER PRIMARY KEY,
                status TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
    
    yield engine
    await engine.dispose()


@pytest.fixture
def mock_settings():
    """Mock settings for isolated component testing."""
    return MagicMock(**{
        'DEBUG': True,
        'REDIS_URL': 'redis://localhost:6379/15',
        'DATABASE_URL': 'sqlite+aiosqlite:///:memory:',
        'SECRET_KEY': 'test-secret-key',
        'JWT_SECRET_KEY': 'test-jwt-secret',
        'JWT_ALGORITHM': 'HS256',
        'REDIS_STREAM_MAX_LEN': 1000,
        'ALLOWED_HOSTS': ['testserver', 'localhost', '127.0.0.1'],
        'CORS_ORIGINS': ['http://testserver', 'http://localhost:8000']
    })


@pytest.fixture
def isolated_environment(mock_settings):
    """Completely isolated environment for component testing."""
    with patch.dict(os.environ, {
        'SKIP_STARTUP_INIT': 'true',
        'CI': 'false',
        'DEBUG': 'true'
    }):
        with patch('app.core.config.settings', mock_settings):
            yield mock_settings


class MockRedisConnection:
    """Mock Redis connection for testing components without Redis dependency."""
    
    def __init__(self):
        self.data = {}
        self.streams = {}
        self.sorted_sets = {}
        
    async def ping(self):
        return True
    
    async def set(self, key: str, value: str, ex: Optional[int] = None):
        self.data[key] = value
        
    async def get(self, key: str):
        return self.data.get(key)
        
    async def delete(self, key: str):
        self.data.pop(key, None)
        
    async def lpush(self, key: str, *values):
        if key not in self.data:
            self.data[key] = []
        self.data[key].extend(reversed(values))
        
    async def ltrim(self, key: str, start: int, end: int):
        if key in self.data:
            self.data[key] = self.data[key][start:end+1]
    
    async def zadd(self, key: str, mapping: Dict[str, float]):
        if key not in self.sorted_sets:
            self.sorted_sets[key] = {}
        self.sorted_sets[key].update(mapping)
        
    async def zcard(self, key: str):
        return len(self.sorted_sets.get(key, {}))
        
    async def zremrangebyscore(self, key: str, min_score: float, max_score: float):
        if key in self.sorted_sets:
            to_remove = [k for k, v in self.sorted_sets[key].items() 
                        if min_score <= v <= max_score]
            for k in to_remove:
                del self.sorted_sets[key][k]
    
    async def expire(self, key: str, seconds: int):
        # Mock expiration - in real testing we'd track this
        pass
    
    async def close(self):
        pass


@pytest.fixture
async def mock_redis():
    """Mock Redis connection for components that need Redis-like behavior."""
    mock_conn = MockRedisConnection()
    yield mock_conn
    await mock_conn.close()


@pytest.fixture
def isolated_component_environment(isolated_environment, mock_redis):
    """Complete isolation environment for component testing."""
    with patch('app.core.redis.get_redis', return_value=mock_redis):
        with patch('app.core.redis._redis_client', mock_redis):
            yield {
                'settings': isolated_environment,
                'redis': mock_redis
            }


class ComponentTestCase:
    """Base class for component isolation tests."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.metrics = ComponentTestMetrics(component_name)
    
    async def setup_component(self):
        """Override in subclasses for component-specific setup."""
        pass
    
    async def teardown_component(self):
        """Override in subclasses for component-specific teardown."""
        pass
    
    async def run_isolated_test(self, test_func):
        """Run a test function with component isolation."""
        async with self.metrics.measure_async():
            await self.setup_component()
            try:
                result = await test_func()
                return result
            finally:
                await self.teardown_component()


@pytest.fixture
def component_test_case():
    """Factory for creating component test cases."""
    def _create_test_case(component_name: str) -> ComponentTestCase:
        return ComponentTestCase(component_name)
    return _create_test_case


# Performance assertion helpers

def assert_performance_target(metrics: Dict[str, Any], max_duration: float, max_memory_mb: float):
    """Assert that component meets performance targets."""
    duration = metrics.get('duration_seconds', 0)
    memory = metrics.get('memory_delta_mb', 0)
    
    assert duration <= max_duration, f"Component took {duration}s, expected <={max_duration}s"
    assert memory <= max_memory_mb, f"Component used {memory}MB, expected <={max_memory_mb}MB"


def assert_redis_performance(metrics: Dict[str, Any]):
    """Assert Redis component performance targets."""
    assert_performance_target(metrics, max_duration=0.1, max_memory_mb=10)


def assert_database_performance(metrics: Dict[str, Any]):
    """Assert database component performance targets."""
    assert_performance_target(metrics, max_duration=0.5, max_memory_mb=25)


def assert_security_performance(metrics: Dict[str, Any]):
    """Assert security component performance targets."""
    assert_performance_target(metrics, max_duration=0.2, max_memory_mb=15)


# Async test utilities

@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Component boundary validation helpers

class ComponentBoundary:
    """Helper for validating component boundaries and interfaces."""
    
    @staticmethod
    def validate_interface(component, expected_methods: list):
        """Validate that component has expected interface methods."""
        for method in expected_methods:
            assert hasattr(component, method), f"Component missing required method: {method}"
    
    @staticmethod
    def validate_async_interface(component, expected_async_methods: list):
        """Validate that component has expected async methods."""
        for method in expected_async_methods:
            assert hasattr(component, method), f"Component missing async method: {method}"
            assert asyncio.iscoroutinefunction(getattr(component, method)), \
                f"Method {method} is not async"


@pytest.fixture
def boundary_validator():
    """Component boundary validation helper."""
    return ComponentBoundary()


# Test environment markers

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "isolation: Component isolation tests")
    config.addinivalue_line("markers", "performance: Performance baseline tests") 
    config.addinivalue_line("markers", "boundary: Integration boundary tests")
    config.addinivalue_line("markers", "consolidation: Consolidation readiness tests")