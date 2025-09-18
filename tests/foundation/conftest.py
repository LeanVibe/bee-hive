"""
Foundation Test Configuration - Testing Pyramid Base Layer

Provides fixtures, configuration, and utilities for foundation testing.
Foundation tests focus on basic system integrity validation.

TESTING PYRAMID LEVEL: Foundation (Base Layer)
PURPOSE: Validate imports, configurations, models, and core dependencies
EXECUTION TIME TARGET: <30 seconds total
"""

import pytest
import os
import sys
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch, MagicMock

# Add app to path for testing
app_path = Path(__file__).parent.parent.parent / "app"
if app_path.exists():
    sys.path.insert(0, str(app_path.parent))

# Foundation test configuration
FOUNDATION_TIMEOUT = 30
FOUNDATION_ENV_VARS = {
    "TESTING": "true",
    "CI": "true",
    "SKIP_STARTUP_INIT": "true",
    "DEBUG": "false",
    "LOG_LEVEL": "ERROR",
    "SECRET_KEY": "foundation-test-secret-key",
    "DATABASE_URL": "sqlite:///:memory:",
    "REDIS_URL": "redis://localhost:6379/99",
}

@pytest.fixture(scope="session", autouse=True)
def foundation_test_environment():
    """Set up foundation test environment."""
    # Set foundation test environment variables
    original_env = {}
    
    for key, value in FOUNDATION_ENV_VARS.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
        
    yield
    
    # Restore original environment
    for key, original_value in original_env.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value

@pytest.fixture
def mock_database():
    """Mock database for foundation tests."""
    mock_session = MagicMock()
    mock_session.execute.return_value.scalar.return_value = 1
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None
    
    with patch('app.core.database.get_session') as mock_get_session:
        mock_get_session.return_value = lambda: mock_session
        yield mock_session

@pytest.fixture
def mock_redis():
    """Mock Redis for foundation tests."""
    mock_redis_client = MagicMock()
    mock_redis_client.ping.return_value = True
    
    with patch('app.core.redis.get_redis') as mock_get_redis:
        mock_get_redis.return_value = mock_redis_client
        yield mock_redis_client

@pytest.fixture
def foundation_config():
    """Provide foundation test configuration."""
    return {
        "timeout": FOUNDATION_TIMEOUT,
        "env_vars": FOUNDATION_ENV_VARS,
        "test_level": "foundation",
        "fast_mode": True,
        "mock_external_deps": True
    }

@pytest.fixture
def app_path():
    """Provide path to app directory."""
    return Path(__file__).parent.parent.parent / "app"

@pytest.fixture
def test_data_factory():
    """Factory for generating test data."""
    class TestDataFactory:
        @staticmethod
        def valid_config() -> Dict[str, Any]:
            return {
                "SECRET_KEY": "test-secret-key",
                "DATABASE_URL": "sqlite:///./test.db",
                "REDIS_URL": "redis://localhost:6379/0",
                "DEBUG": True,
                "ENVIRONMENT": "testing"
            }
            
        @staticmethod
        def invalid_config() -> Dict[str, Any]:
            return {
                "SECRET_KEY": "",  # Invalid: empty
                "DATABASE_URL": "invalid-url",  # Invalid: bad format
                "REDIS_URL": "not-redis://localhost",  # Invalid: wrong scheme
            }
            
        @staticmethod
        def model_data() -> Dict[str, Any]:
            from datetime import datetime, timezone
            return {
                "string_field": "test_value",
                "integer_field": 42,
                "float_field": 3.14,
                "boolean_field": True,
                "datetime_field": datetime.now(timezone.utc),
                "optional_field": None
            }
    
    return TestDataFactory()

# Foundation-specific markers
def pytest_configure(config):
    """Configure foundation-specific pytest markers."""
    config.addinivalue_line(
        "markers", "foundation: mark test as foundation layer test"
    )
    config.addinivalue_line(
        "markers", "import_test: mark test as import resolution test"
    )
    config.addinivalue_line(
        "markers", "config_test: mark test as configuration validation test"
    )
    config.addinivalue_line(
        "markers", "model_test: mark test as model integrity test"
    )
    config.addinivalue_line(
        "markers", "dependency_test: mark test as core dependency test"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection for foundation tests."""
    for item in items:
        # Auto-add foundation marker to all tests in foundation directory
        if "foundation" in str(item.fspath):
            item.add_marker(pytest.mark.foundation)
            
        # Add timeout to foundation tests
        if item.get_closest_marker("foundation"):
            if not item.get_closest_marker("timeout"):
                item.add_marker(pytest.mark.timeout(FOUNDATION_TIMEOUT))

# Foundation test utilities
class FoundationTestHelper:
    """Helper utilities for foundation tests."""
    
    @staticmethod
    def is_ci_environment() -> bool:
        """Check if running in CI environment."""
        return os.environ.get("CI") == "true"
    
    @staticmethod
    def is_testing_environment() -> bool:
        """Check if running in testing environment."""
        return (
            os.environ.get("TESTING") == "true" or
            os.environ.get("PYTEST_CURRENT_TEST") is not None
        )
    
    @staticmethod
    def mock_external_service(service_name: str):
        """Create a mock for external services."""
        if service_name == "database":
            mock = MagicMock()
            mock.execute.return_value.scalar.return_value = 1
            return mock
        elif service_name == "redis":
            mock = MagicMock()
            mock.ping.return_value = True
            return mock
        else:
            return MagicMock()
    
    @staticmethod
    def assert_fast_execution(execution_time: float, max_time: float = 5.0):
        """Assert that execution was fast enough for foundation tests."""
        assert execution_time < max_time, \
            f"Foundation test too slow: {execution_time:.2f}s (max: {max_time}s)"
    
    @staticmethod
    def validate_import_result(result):
        """Validate import test result structure."""
        assert hasattr(result, 'module_name')
        assert hasattr(result, 'success')
        assert hasattr(result, 'import_time')
        assert hasattr(result, 'error')
        assert isinstance(result.success, bool)
        assert isinstance(result.import_time, (int, float))
    
    @staticmethod
    def validate_config_result(result):
        """Validate configuration test result structure."""
        assert hasattr(result, 'environment')
        assert hasattr(result, 'success')
        assert hasattr(result, 'errors')
        assert hasattr(result, 'warnings')
        assert isinstance(result.success, bool)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)

@pytest.fixture
def foundation_helper():
    """Provide foundation test helper."""
    return FoundationTestHelper()

# Error handling for missing dependencies
def handle_missing_dependency(dependency_name: str, import_error: Exception):
    """Handle missing dependencies gracefully in foundation tests."""
    import warnings
    warnings.warn(
        f"Dependency {dependency_name} not available: {import_error}. "
        f"Some foundation tests may be skipped."
    )

# Mock providers for external dependencies
@pytest.fixture
def mock_anthropic():
    """Mock Anthropic API client."""
    with patch('anthropic.Anthropic') as mock:
        client = MagicMock()
        client.messages.create.return_value.content = [
            MagicMock(text="Mocked response")
        ]
        mock.return_value = client
        yield client

@pytest.fixture
def mock_openai():
    """Mock OpenAI API client."""
    with patch('openai.OpenAI') as mock:
        client = MagicMock()
        client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="Mocked response"))
        ]
        mock.return_value = client
        yield client

# Performance monitoring for foundation tests
@pytest.fixture(autouse=True)
def monitor_foundation_test_performance(request):
    """Monitor performance of foundation tests."""
    import time
    
    start_time = time.time()
    yield
    execution_time = time.time() - start_time
    
    # Warn about slow foundation tests
    if execution_time > 5.0:
        import warnings
        warnings.warn(
            f"Foundation test {request.node.name} took {execution_time:.2f}s "
            f"(foundation tests should be <5s)"
        )

# Cleanup utilities
@pytest.fixture
def cleanup_test_files():
    """Clean up test files after foundation tests."""
    test_files = []
    
    def register_file(file_path):
        """Register a file for cleanup."""
        test_files.append(Path(file_path))
    
    yield register_file
    
    # Cleanup registered files
    for file_path in test_files:
        try:
            if file_path.exists():
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    import shutil
                    shutil.rmtree(file_path)
        except Exception:
            # Don't fail tests due to cleanup issues
            pass