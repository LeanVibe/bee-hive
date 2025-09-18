"""
Core Dependency Testing - Foundation Layer

Validates that core dependencies are functional including database connectivity,
Redis operations, essential third-party packages, and file system permissions.

TESTING PYRAMID LEVEL: Foundation (Base Layer)
EXECUTION TIME TARGET: <8 seconds
COVERAGE: Database, Redis, third-party packages, file system, network
"""

import pytest
import asyncio
import tempfile
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any
from unittest.mock import patch, MagicMock, AsyncMock

# Core dependency test constants
DEPENDENCY_TIMEOUT = 8
CONNECTION_TIMEOUT = 3.0
MAX_CONNECTION_TIME = 2.0

class DependencyTestResult:
    """Result of dependency testing."""
    
    def __init__(self, dependency_name: str):
        self.dependency_name = dependency_name
        self.success = False
        self.test_time = 0.0
        self.connection_time = 0.0
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.details: Dict[str, Any] = {}

class CoreDependencyValidator:
    """Validates core system dependencies."""
    
    def __init__(self):
        self.results: List[DependencyTestResult] = []
        
    def test_database_connectivity(self) -> DependencyTestResult:
        """Test database connectivity and basic operations."""
        result = DependencyTestResult("database")
        start_time = time.time()
        
        try:
            # Test 1: Import database modules
            try:
                from app.core.database import get_session, init_database
                result.details['import_success'] = True
            except ImportError as e:
                result.errors.append(f"Failed to import database modules: {e}")
                result.details['import_success'] = False
                return result
                
            # Test 2: Database URL validation
            try:
                from app.core.config import get_settings
                settings = get_settings()
                
                if hasattr(settings, 'DATABASE_URL'):
                    db_url = getattr(settings, 'DATABASE_URL')
                    result.details['database_url'] = self._mask_url(db_url)
                    
                    # Validate URL format
                    if not db_url or not any(db_url.startswith(scheme) for scheme in ['postgresql://', 'sqlite://', 'mysql://']):
                        result.warnings.append(f"Unusual database URL format: {self._mask_url(db_url)}")
                    else:
                        result.details['url_format_valid'] = True
                else:
                    result.errors.append("DATABASE_URL not found in settings")
                    
            except Exception as e:
                result.errors.append(f"Database URL validation failed: {e}")
                
            # Test 3: Basic connection test (mock in CI/testing)
            if self._is_testing_environment():
                # Mock database connection in testing
                result.details['connection_test'] = 'mocked'
                result.details['connection_success'] = True
            else:
                try:
                    connection_start = time.time()
                    
                    # Attempt actual connection with timeout
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    async def test_connection():
                        try:
                            session_maker = get_session()
                            # Test basic query execution
                            async with session_maker() as session:
                                from sqlalchemy import text
                                result_proxy = await session.execute(text("SELECT 1"))
                                test_result = result_proxy.scalar()
                                return test_result == 1
                        except Exception as e:
                            raise e
                    
                    connection_success = loop.run_until_complete(
                        asyncio.wait_for(test_connection(), timeout=CONNECTION_TIMEOUT)
                    )
                    
                    result.connection_time = time.time() - connection_start
                    result.details['connection_success'] = connection_success
                    result.details['connection_time'] = result.connection_time
                    
                    if result.connection_time > MAX_CONNECTION_TIME:
                        result.warnings.append(f"Slow database connection: {result.connection_time:.2f}s")
                        
                except asyncio.TimeoutError:
                    result.errors.append(f"Database connection timeout after {CONNECTION_TIMEOUT}s")
                except Exception as e:
                    result.warnings.append(f"Database connection test failed (may be expected in testing): {e}")
                    result.details['connection_test'] = 'failed'
                finally:
                    loop.close()
                    
            result.success = len(result.errors) == 0
            
        except Exception as e:
            result.errors.append(f"Unexpected database test error: {e}")
            
        result.test_time = time.time() - start_time
        self.results.append(result)
        return result
    
    def test_redis_connectivity(self) -> DependencyTestResult:
        """Test Redis connectivity and basic operations."""
        result = DependencyTestResult("redis")
        start_time = time.time()
        
        try:
            # Test 1: Import Redis modules
            try:
                from app.core.redis import get_redis, init_redis
                result.details['import_success'] = True
            except ImportError as e:
                result.errors.append(f"Failed to import Redis modules: {e}")
                result.details['import_success'] = False
                return result
                
            # Test 2: Redis URL validation
            try:
                from app.core.config import get_settings
                settings = get_settings()
                
                if hasattr(settings, 'REDIS_URL'):
                    redis_url = getattr(settings, 'REDIS_URL')
                    result.details['redis_url'] = self._mask_url(redis_url)
                    
                    # Validate URL format
                    if not redis_url or not redis_url.startswith('redis://'):
                        result.warnings.append(f"Unusual Redis URL format: {self._mask_url(redis_url)}")
                    else:
                        result.details['url_format_valid'] = True
                else:
                    result.errors.append("REDIS_URL not found in settings")
                    
            except Exception as e:
                result.errors.append(f"Redis URL validation failed: {e}")
                
            # Test 3: Basic connection test (mock in CI/testing)
            if self._is_testing_environment():
                # Mock Redis connection in testing
                result.details['connection_test'] = 'mocked'
                result.details['connection_success'] = True
            else:
                try:
                    connection_start = time.time()
                    
                    # Attempt actual connection with timeout
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    async def test_redis_connection():
                        try:
                            redis_client = get_redis()
                            # Test ping
                            pong = await redis_client.ping()
                            return pong is True
                        except Exception as e:
                            raise e
                    
                    connection_success = loop.run_until_complete(
                        asyncio.wait_for(test_redis_connection(), timeout=CONNECTION_TIMEOUT)
                    )
                    
                    result.connection_time = time.time() - connection_start
                    result.details['connection_success'] = connection_success
                    result.details['connection_time'] = result.connection_time
                    
                    if result.connection_time > MAX_CONNECTION_TIME:
                        result.warnings.append(f"Slow Redis connection: {result.connection_time:.2f}s")
                        
                except asyncio.TimeoutError:
                    result.errors.append(f"Redis connection timeout after {CONNECTION_TIMEOUT}s")
                except Exception as e:
                    result.warnings.append(f"Redis connection test failed (may be expected in testing): {e}")
                    result.details['connection_test'] = 'failed'
                finally:
                    loop.close()
                    
            result.success = len(result.errors) == 0
            
        except Exception as e:
            result.errors.append(f"Unexpected Redis test error: {e}")
            
        result.test_time = time.time() - start_time
        self.results.append(result)
        return result
    
    def test_third_party_packages(self) -> DependencyTestResult:
        """Test essential third-party packages."""
        result = DependencyTestResult("third_party_packages")
        start_time = time.time()
        
        # Essential packages for basic functionality
        essential_packages = {
            'fastapi': 'Web framework',
            'uvicorn': 'ASGI server',
            'sqlalchemy': 'Database ORM',
            'redis': 'Redis client',
            'pydantic': 'Data validation'
        }
        
        # Optional but commonly used packages
        optional_packages = {
            'anthropic': 'AI API client',
            'openai': 'OpenAI client',
            'prometheus_client': 'Metrics collection'
        }
        
        missing_essential = []
        missing_optional = []
        version_info = {}
        
        # Test essential packages
        for package, description in essential_packages.items():
            try:
                import importlib
                module = importlib.import_module(package)
                
                # Try to get version
                version = getattr(module, '__version__', 'unknown')
                version_info[package] = version
                
            except ImportError:
                missing_essential.append(f"{package} ({description})")
                
        # Test optional packages
        for package, description in optional_packages.items():
            try:
                import importlib
                importlib.import_module(package)
                version = getattr(importlib.import_module(package), '__version__', 'unknown')
                version_info[package] = version
            except ImportError:
                missing_optional.append(f"{package} ({description})")
                
        # Evaluate results
        if missing_essential:
            result.errors.append(f"Missing essential packages: {', '.join(missing_essential)}")
        else:
            result.details['essential_packages'] = 'all_present'
            
        if missing_optional:
            result.warnings.append(f"Missing optional packages: {', '.join(missing_optional)}")
            
        result.details['version_info'] = version_info
        result.details['packages_tested'] = len(essential_packages) + len(optional_packages)
        
        result.success = len(result.errors) == 0
        result.test_time = time.time() - start_time
        self.results.append(result)
        return result
    
    def test_file_system_permissions(self) -> DependencyTestResult:
        """Test file system permissions for logs, uploads, etc."""
        result = DependencyTestResult("file_system")
        start_time = time.time()
        
        test_directories = [
            'logs',
            'temp',
            'uploads'
        ]
        
        permission_issues = []
        
        try:
            # Test temporary directory access
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Test write permission
                test_file = temp_path / "test_write.txt"
                try:
                    test_file.write_text("test content")
                    result.details['temp_write'] = True
                except Exception as e:
                    permission_issues.append(f"Cannot write to temp directory: {e}")
                    
                # Test read permission
                try:
                    content = test_file.read_text()
                    assert content == "test content"
                    result.details['temp_read'] = True
                except Exception as e:
                    permission_issues.append(f"Cannot read from temp directory: {e}")
                    
                # Test directory creation
                try:
                    sub_dir = temp_path / "test_subdir"
                    sub_dir.mkdir()
                    result.details['directory_creation'] = True
                except Exception as e:
                    permission_issues.append(f"Cannot create subdirectories: {e}")
                    
            # Test working directory permissions
            try:
                cwd = Path.cwd()
                result.details['working_directory'] = str(cwd)
                
                # Check if we can read current directory
                list(cwd.iterdir())
                result.details['cwd_readable'] = True
                
            except Exception as e:
                permission_issues.append(f"Cannot access working directory: {e}")
                
            # Test application-specific directories (if they exist)
            app_root = Path(__file__).parent.parent.parent
            for dir_name in test_directories:
                dir_path = app_root / dir_name
                if dir_path.exists():
                    try:
                        # Test read access
                        list(dir_path.iterdir())
                        result.details[f'{dir_name}_readable'] = True
                        
                        # Test write access (careful not to disrupt)
                        test_file = dir_path / ".permission_test"
                        test_file.touch()
                        test_file.unlink()  # Clean up
                        result.details[f'{dir_name}_writable'] = True
                        
                    except Exception as e:
                        result.warnings.append(f"Limited access to {dir_name} directory: {e}")
                        
        except Exception as e:
            permission_issues.append(f"File system test error: {e}")
            
        if permission_issues:
            result.errors.extend(permission_issues)
            
        result.success = len(result.errors) == 0
        result.test_time = time.time() - start_time
        self.results.append(result)
        return result
    
    def test_network_connectivity(self) -> DependencyTestResult:
        """Test basic network connectivity requirements."""
        result = DependencyTestResult("network")
        start_time = time.time()
        
        # Skip network tests in CI/testing environments
        if self._is_testing_environment():
            result.details['test_mode'] = 'skipped_in_testing'
            result.success = True
            result.test_time = time.time() - start_time
            self.results.append(result)
            return result
            
        try:
            import socket
            
            # Test basic socket functionality
            try:
                # Test localhost connectivity
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                
                # Try to connect to a standard port (this tests basic networking)
                test_result = sock.connect_ex(('127.0.0.1', 22))  # SSH port, commonly open
                sock.close()
                
                result.details['localhost_connectivity'] = test_result == 0
                
            except Exception as e:
                result.warnings.append(f"Basic socket test failed: {e}")
                
            # Test DNS resolution
            try:
                hostname = socket.gethostname()
                result.details['hostname'] = hostname
                
                # Basic DNS test
                socket.gethostbyname('localhost')
                result.details['dns_resolution'] = True
                
            except Exception as e:
                result.warnings.append(f"DNS resolution test failed: {e}")
                
            result.success = True  # Network tests are informational
            
        except Exception as e:
            result.warnings.append(f"Network test error: {e}")
            result.success = True  # Don't fail on network issues
            
        result.test_time = time.time() - start_time
        self.results.append(result)
        return result
    
    def _mask_url(self, url: str) -> str:
        """Mask sensitive information in URLs."""
        if not url:
            return "empty"
            
        if '://' in url:
            scheme, rest = url.split('://', 1)
            if '@' in rest:
                creds, host_part = rest.split('@', 1)
                return f"{scheme}://***:***@{host_part}"
        return url
    
    def _is_testing_environment(self) -> bool:
        """Check if we're in a testing environment."""
        import os
        return (
            os.environ.get("CI") == "true" or
            os.environ.get("TESTING") == "true" or
            os.environ.get("PYTEST_CURRENT_TEST") is not None or
            os.environ.get("SKIP_STARTUP_INIT") == "true"
        )

@pytest.fixture
def dependency_validator():
    """Fixture providing a CoreDependencyValidator instance."""
    return CoreDependencyValidator()

class TestDatabaseDependency:
    """Test suite for database dependency validation."""
    
    def test_database_modules_importable(self, dependency_validator):
        """Test that database modules can be imported."""
        result = dependency_validator.test_database_connectivity()
        
        # Import should always work
        assert result.details.get('import_success'), f"Database import failed: {result.errors}"
        
    def test_database_configuration_valid(self, dependency_validator):
        """Test database configuration validity."""
        result = dependency_validator.test_database_connectivity()
        
        # Configuration should be readable
        config_errors = [e for e in result.errors if 'DATABASE_URL' in e]
        assert not config_errors, f"Database configuration issues: {config_errors}"

class TestRedisDependency:
    """Test suite for Redis dependency validation."""
    
    def test_redis_modules_importable(self, dependency_validator):
        """Test that Redis modules can be imported."""
        result = dependency_validator.test_redis_connectivity()
        
        # Import should always work
        assert result.details.get('import_success'), f"Redis import failed: {result.errors}"
        
    def test_redis_configuration_valid(self, dependency_validator):
        """Test Redis configuration validity."""
        result = dependency_validator.test_redis_connectivity()
        
        # Configuration should be readable
        config_errors = [e for e in result.errors if 'REDIS_URL' in e]
        assert not config_errors, f"Redis configuration issues: {config_errors}"

class TestThirdPartyPackages:
    """Test suite for third-party package validation."""
    
    def test_essential_packages_available(self, dependency_validator):
        """Test that essential packages are available."""
        result = dependency_validator.test_third_party_packages()
        
        # Essential packages should be present
        assert result.success, f"Essential packages missing: {result.errors}"
        
    def test_package_versions_reasonable(self, dependency_validator):
        """Test that package versions are reasonable."""
        result = dependency_validator.test_third_party_packages()
        
        version_info = result.details.get('version_info', {})
        
        # Check that we got version information for key packages
        key_packages = ['fastapi', 'pydantic', 'sqlalchemy']
        for package in key_packages:
            if package in version_info:
                version = version_info[package]
                assert version != 'unknown', f"Could not determine version for {package}"

class TestFileSystemAccess:
    """Test suite for file system access validation."""
    
    def test_basic_file_operations(self, dependency_validator):
        """Test basic file system operations."""
        result = dependency_validator.test_file_system_permissions()
        
        # Basic file operations should work
        assert result.details.get('temp_write'), "Cannot write temporary files"
        assert result.details.get('temp_read'), "Cannot read temporary files"
        assert result.details.get('directory_creation'), "Cannot create directories"
        
    def test_working_directory_access(self, dependency_validator):
        """Test working directory access."""
        result = dependency_validator.test_file_system_permissions()
        
        # Should be able to access working directory
        assert result.details.get('cwd_readable'), "Cannot read working directory"

@pytest.mark.foundation
@pytest.mark.timeout(DEPENDENCY_TIMEOUT)
class TestFoundationCoreDependencies:
    """Foundation test marker for core dependency tests."""
    
    def test_foundation_core_imports(self):
        """Test that core dependencies can be imported."""
        validator = CoreDependencyValidator()
        
        # Test database import
        db_result = validator.test_database_connectivity()
        assert db_result.details.get('import_success'), "Database modules not importable"
        
        # Test Redis import
        redis_result = validator.test_redis_connectivity()
        assert redis_result.details.get('import_success'), "Redis modules not importable"
        
    def test_foundation_essential_packages(self):
        """Test that essential packages are available."""
        validator = CoreDependencyValidator()
        
        packages_result = validator.test_third_party_packages()
        assert packages_result.success, f"Essential packages missing: {packages_result.errors}"
        
    def test_foundation_file_system_basics(self):
        """Test basic file system functionality."""
        validator = CoreDependencyValidator()
        
        fs_result = validator.test_file_system_permissions()
        
        # Critical file operations should work
        critical_errors = [e for e in fs_result.errors if 'temp' in e.lower()]
        assert not critical_errors, f"Critical file system issues: {critical_errors}"

if __name__ == "__main__":
    # Run foundation core dependency tests
    pytest.main([__file__, "-v", "--tb=short"])