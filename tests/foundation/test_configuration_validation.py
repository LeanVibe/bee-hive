"""
Configuration Validation Testing - Foundation Layer

Validates that all environment variable combinations work correctly,
configuration classes instantiate properly, and all deployment scenarios
have valid configurations.

TESTING PYRAMID LEVEL: Foundation (Base Layer)
EXECUTION TIME TARGET: <8 seconds
COVERAGE: All environment configurations, settings validation, secrets management
"""

import pytest
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock
import warnings
from dataclasses import dataclass
from enum import Enum

# Configuration test constants
CONFIG_TIMEOUT = 8
REQUIRED_ENV_VARS = [
    "DATABASE_URL",
    "REDIS_URL", 
    "SECRET_KEY"
]
OPTIONAL_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "DEBUG",
    "ENVIRONMENT"
]

class EnvironmentType(Enum):
    """Environment types for testing."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    CI = "ci"

@dataclass
class ConfigTestResult:
    """Result of configuration testing."""
    environment: str
    success: bool
    errors: List[str]
    warnings: List[str]
    validation_time: float
    config_data: Optional[Dict[str, Any]] = None

class ConfigurationValidator:
    """Validates configuration across different environments."""
    
    def __init__(self):
        self.results: List[ConfigTestResult] = []
        
    def create_test_environment(self, env_type: EnvironmentType) -> Dict[str, str]:
        """Create test environment variables for given environment type."""
        base_config = {
            "SECRET_KEY": "test-secret-key-for-testing-only",
            "DATABASE_URL": "postgresql://test:test@localhost:5432/test_db",
            "REDIS_URL": "redis://localhost:6379/0",
            "ENVIRONMENT": env_type.value,
        }
        
        if env_type == EnvironmentType.DEVELOPMENT:
            base_config.update({
                "DEBUG": "true",
                "LOG_LEVEL": "DEBUG",
                "CORS_ORIGINS": "http://localhost:3000,http://localhost:8080",
                "ALLOWED_HOSTS": "localhost,127.0.0.1",
            })
            
        elif env_type == EnvironmentType.PRODUCTION:
            base_config.update({
                "DEBUG": "false",
                "LOG_LEVEL": "INFO", 
                "CORS_ORIGINS": "https://app.leanvibe.com",
                "ALLOWED_HOSTS": "app.leanvibe.com",
                "DATABASE_URL": "postgresql://prod_user:secure_pass@prod-db:5432/hive_prod",
                "REDIS_URL": "redis://prod-redis:6379/0",
            })
            
        elif env_type == EnvironmentType.TESTING:
            base_config.update({
                "DEBUG": "true",
                "LOG_LEVEL": "WARNING",
                "TESTING": "true",
                "DATABASE_URL": "sqlite:///./test.db",
                "REDIS_URL": "redis://localhost:6379/1",
            })
            
        elif env_type == EnvironmentType.CI:
            base_config.update({
                "CI": "true",
                "DEBUG": "false",
                "LOG_LEVEL": "ERROR",
                "SKIP_STARTUP_INIT": "true",
                "DATABASE_URL": "sqlite:///:memory:",
                "REDIS_URL": "redis://localhost:6379/2",
            })
            
        return base_config
    
    def validate_configuration(self, env_vars: Dict[str, str]) -> ConfigTestResult:
        """Validate configuration with given environment variables."""
        import time
        start_time = time.time()
        
        result = ConfigTestResult(
            environment=env_vars.get("ENVIRONMENT", "unknown"),
            success=False,
            errors=[],
            warnings=[],
            validation_time=0.0
        )
        
        try:
            # Test configuration loading with environment isolation
            with patch.dict(os.environ, env_vars, clear=False):
                # Try to import and validate settings
                try:
                    from app.core.config import get_settings, Settings
                    
                    # Test settings instantiation
                    settings = get_settings()
                    result.config_data = self._extract_config_data(settings)
                    
                    # Validate required settings
                    validation_errors = self._validate_settings(settings)
                    result.errors.extend(validation_errors)
                    
                    # Check for configuration warnings
                    config_warnings = self._check_config_warnings(settings)
                    result.warnings.extend(config_warnings)
                    
                    result.success = len(result.errors) == 0
                    
                except ImportError as e:
                    result.errors.append(f"Failed to import settings: {e}")
                except Exception as e:
                    result.errors.append(f"Settings instantiation failed: {e}")
                    
        except Exception as e:
            result.errors.append(f"Configuration validation failed: {e}")
            
        result.validation_time = time.time() - start_time
        self.results.append(result)
        return result
    
    def _extract_config_data(self, settings) -> Dict[str, Any]:
        """Extract configuration data for validation."""
        config_data = {}
        
        # Extract common settings attributes
        for attr in ['DEBUG', 'ENVIRONMENT', 'LOG_LEVEL', 'DATABASE_URL', 'REDIS_URL']:
            if hasattr(settings, attr):
                value = getattr(settings, attr)
                # Mask sensitive data
                if 'URL' in attr and isinstance(value, str):
                    config_data[attr] = self._mask_sensitive_url(value)
                else:
                    config_data[attr] = value
                    
        return config_data
    
    def _mask_sensitive_url(self, url: str) -> str:
        """Mask sensitive information in URLs."""
        if '://' in url:
            scheme, rest = url.split('://', 1)
            if '@' in rest:
                creds, host_part = rest.split('@', 1)
                return f"{scheme}://***:***@{host_part}"
        return url
    
    def _validate_settings(self, settings) -> List[str]:
        """Validate settings object for required attributes."""
        errors = []
        
        # Check required attributes exist
        required_attrs = ['DATABASE_URL', 'REDIS_URL', 'SECRET_KEY']
        for attr in required_attrs:
            if not hasattr(settings, attr):
                errors.append(f"Missing required setting: {attr}")
            elif not getattr(settings, attr):
                errors.append(f"Empty required setting: {attr}")
                
        # Validate URL formats
        if hasattr(settings, 'DATABASE_URL'):
            db_url = getattr(settings, 'DATABASE_URL')
            if not self._is_valid_database_url(db_url):
                errors.append(f"Invalid DATABASE_URL format: {db_url}")
                
        if hasattr(settings, 'REDIS_URL'):
            redis_url = getattr(settings, 'REDIS_URL')
            if not self._is_valid_redis_url(redis_url):
                errors.append(f"Invalid REDIS_URL format: {redis_url}")
                
        return errors
    
    def _is_valid_database_url(self, url: str) -> bool:
        """Validate database URL format."""
        if not url:
            return False
        valid_schemes = ['postgresql', 'sqlite', 'mysql']
        return any(url.startswith(f'{scheme}://') for scheme in valid_schemes)
    
    def _is_valid_redis_url(self, url: str) -> bool:
        """Validate Redis URL format."""
        if not url:
            return False
        return url.startswith('redis://')
    
    def _check_config_warnings(self, settings) -> List[str]:
        """Check for configuration warnings."""
        warnings = []
        
        # Check for insecure configurations
        if hasattr(settings, 'DEBUG') and getattr(settings, 'DEBUG'):
            if hasattr(settings, 'SECRET_KEY'):
                secret = getattr(settings, 'SECRET_KEY')
                if secret and 'test' in secret.lower():
                    warnings.append("Using test secret key in debug mode")
                    
        # Check for missing optional but recommended settings
        if hasattr(settings, 'CORS_ORIGINS'):
            cors = getattr(settings, 'CORS_ORIGINS', [])
            if not cors:
                warnings.append("CORS_ORIGINS not configured")
                
        return warnings

@pytest.fixture
def config_validator():
    """Fixture providing a ConfigurationValidator instance."""
    return ConfigurationValidator()

class TestConfigurationValidation:
    """Test suite for configuration validation."""
    
    def test_development_config(self, config_validator):
        """Test development environment configuration."""
        env_vars = config_validator.create_test_environment(EnvironmentType.DEVELOPMENT)
        result = config_validator.validate_configuration(env_vars)
        
        assert result.success, f"Development config validation failed: {result.errors}"
        assert result.config_data is not None
        assert result.config_data.get('DEBUG') is True
        
    def test_production_config(self, config_validator):
        """Test production environment configuration."""
        env_vars = config_validator.create_test_environment(EnvironmentType.PRODUCTION)
        result = config_validator.validate_configuration(env_vars)
        
        assert result.success, f"Production config validation failed: {result.errors}"
        assert result.config_data is not None
        assert result.config_data.get('DEBUG') is False
        
    def test_testing_config(self, config_validator):
        """Test testing environment configuration."""
        env_vars = config_validator.create_test_environment(EnvironmentType.TESTING)
        result = config_validator.validate_configuration(env_vars)
        
        assert result.success, f"Testing config validation failed: {result.errors}"
        assert result.config_data is not None
        
    def test_ci_config(self, config_validator):
        """Test CI environment configuration."""
        env_vars = config_validator.create_test_environment(EnvironmentType.CI)
        result = config_validator.validate_configuration(env_vars)
        
        assert result.success, f"CI config validation failed: {result.errors}"
        assert result.config_data is not None
        
    def test_missing_required_vars(self, config_validator):
        """Test configuration with missing required variables."""
        # Create incomplete configuration
        incomplete_config = {
            "SECRET_KEY": "test-key",
            # Missing DATABASE_URL and REDIS_URL
        }
        
        result = config_validator.validate_configuration(incomplete_config)
        
        # Should fail validation
        assert not result.success
        assert any("DATABASE_URL" in error for error in result.errors)
        assert any("REDIS_URL" in error for error in result.errors)
        
    def test_invalid_url_formats(self, config_validator):
        """Test configuration with invalid URL formats."""
        invalid_config = {
            "SECRET_KEY": "test-key",
            "DATABASE_URL": "invalid-database-url",
            "REDIS_URL": "invalid-redis-url",
        }
        
        result = config_validator.validate_configuration(invalid_config)
        
        # Should fail validation due to invalid URLs
        assert not result.success
        assert any("Invalid DATABASE_URL" in error for error in result.errors)
        assert any("Invalid REDIS_URL" in error for error in result.errors)
        
    def test_config_performance(self, config_validator):
        """Test that configuration validation is fast."""
        env_vars = config_validator.create_test_environment(EnvironmentType.DEVELOPMENT)
        result = config_validator.validate_configuration(env_vars)
        
        # Configuration loading should be fast
        assert result.validation_time < 3.0, f"Config validation too slow: {result.validation_time}s"

class TestEnvironmentSpecificSettings:
    """Test environment-specific configuration behaviors."""
    
    def test_debug_mode_settings(self):
        """Test debug mode specific settings."""
        debug_env = {
            "DEBUG": "true",
            "SECRET_KEY": "test-key",
            "DATABASE_URL": "sqlite:///./test.db",
            "REDIS_URL": "redis://localhost:6379/0",
        }
        
        with patch.dict(os.environ, debug_env, clear=False):
            try:
                from app.core.config import get_settings
                settings = get_settings()
                
                # Debug mode should enable certain features
                assert hasattr(settings, 'DEBUG')
                assert getattr(settings, 'DEBUG') is True
                
            except ImportError:
                pytest.skip("Configuration module not available")
                
    def test_production_security_settings(self):
        """Test production security settings."""
        prod_env = {
            "DEBUG": "false",
            "SECRET_KEY": "production-secret-key",
            "DATABASE_URL": "postgresql://prod:pass@db:5432/prod_db",
            "REDIS_URL": "redis://redis:6379/0",
            "ENVIRONMENT": "production",
        }
        
        with patch.dict(os.environ, prod_env, clear=False):
            try:
                from app.core.config import get_settings
                settings = get_settings()
                
                # Production should have debug disabled
                assert hasattr(settings, 'DEBUG')
                assert getattr(settings, 'DEBUG') is False
                
            except ImportError:
                pytest.skip("Configuration module not available")

class TestConfigurationResilience:
    """Test configuration resilience and error handling."""
    
    def test_config_with_extra_vars(self, config_validator):
        """Test configuration with extra environment variables."""
        base_config = config_validator.create_test_environment(EnvironmentType.DEVELOPMENT)
        
        # Add extra variables that shouldn't break configuration
        base_config.update({
            "UNKNOWN_VAR": "some_value",
            "EXTRA_SETTING": "12345",
            "RANDOM_CONFIG": "true",
        })
        
        result = config_validator.validate_configuration(base_config)
        
        # Configuration should still work with extra variables
        assert result.success, f"Config failed with extra vars: {result.errors}"
        
    def test_config_type_conversion(self, config_validator):
        """Test configuration type conversion from environment variables."""
        type_test_config = {
            "SECRET_KEY": "test-key",
            "DATABASE_URL": "sqlite:///./test.db",
            "REDIS_URL": "redis://localhost:6379/0",
            "DEBUG": "true",  # String that should become boolean
            "PORT": "8000",   # String that should become integer
            "TIMEOUT": "30.5", # String that should become float
        }
        
        result = config_validator.validate_configuration(type_test_config)
        
        # Type conversion should work correctly
        assert result.success, f"Type conversion failed: {result.errors}"
        
        if result.config_data:
            # Debug should be converted to boolean
            assert isinstance(result.config_data.get('DEBUG'), bool)

@pytest.mark.foundation
@pytest.mark.timeout(CONFIG_TIMEOUT)
class TestFoundationConfiguration:
    """Foundation test marker for configuration tests."""
    
    def test_foundation_config_integrity(self):
        """High-level test ensuring basic configuration integrity."""
        validator = ConfigurationValidator()
        
        # Test all environment types work
        all_pass = True
        failed_envs = []
        
        for env_type in EnvironmentType:
            env_vars = validator.create_test_environment(env_type)
            result = validator.validate_configuration(env_vars)
            
            if not result.success:
                all_pass = False
                failed_envs.append(f"{env_type.value}: {result.errors}")
                
        assert all_pass, f"Configuration validation failed for environments: {failed_envs}"
        
    def test_foundation_required_settings_present(self):
        """Ensure all required settings are properly defined."""
        # Test with minimal required configuration
        minimal_config = {
            "SECRET_KEY": "test-foundation-key",
            "DATABASE_URL": "sqlite:///./foundation_test.db",
            "REDIS_URL": "redis://localhost:6379/99",
        }
        
        validator = ConfigurationValidator()
        result = validator.validate_configuration(minimal_config)
        
        # Minimal required configuration should work
        assert result.success, f"Minimal config failed: {result.errors}"

if __name__ == "__main__":
    # Run foundation configuration tests
    pytest.main([__file__, "-v", "--tb=short"])