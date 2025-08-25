"""
Lightweight Configuration Loader for CLI Performance

Ultra-fast configuration loading optimized for CLI commands with <20ms load time.
This bypasses the heavy Pydantic validation for CLI operations where speed matters
more than comprehensive validation.

Performance Targets:
- Configuration load: <20ms
- Memory footprint: <5MB
- Cold start optimization
"""

import os
import time
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class LightweightConfig:
    """
    Minimal configuration class for CLI performance optimization.
    
    Contains only essential configuration needed for CLI operations,
    avoiding heavy Pydantic validation and complex nested structures.
    """
    
    # Core settings
    environment: str = "development"
    debug: bool = False
    
    # Database
    database_url: str = "sqlite:///./agent_hive.db"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # API settings
    api_port: str = "18080"
    api_host: str = "localhost"
    
    # Security
    secret_key: str = "dev-secret-key"
    
    # External services
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    github_token: Optional[str] = None
    
    # Sandbox mode
    sandbox_mode: bool = False
    
    # Performance settings
    max_workers: int = 4
    request_timeout: int = 300
    
    def __post_init__(self):
        """Auto-detect sandbox mode if API keys are missing."""
        if not self.anthropic_api_key:
            self.sandbox_mode = True
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() in ["production", "prod"]
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment.lower() in ["development", "dev"]
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        return self.database_url
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        return self.redis_url
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility."""
        return {
            'environment': self.environment,
            'debug': self.debug,
            'database_url': self.database_url,
            'redis_url': self.redis_url,
            'api_port': self.api_port,
            'api_host': self.api_host,
            'secret_key': self.secret_key,
            'anthropic_api_key': self.anthropic_api_key,
            'openai_api_key': self.openai_api_key,
            'github_token': self.github_token,
            'sandbox_mode': self.sandbox_mode,
            'max_workers': self.max_workers,
            'request_timeout': self.request_timeout
        }


class FastConfigLoader:
    """
    High-performance configuration loader for CLI commands.
    
    Features:
    - <20ms load time
    - Environment variable parsing
    - Simple caching
    - Minimal memory footprint
    - No external dependencies beyond stdlib
    """
    
    _cached_config: Optional[LightweightConfig] = None
    _cache_timestamp: Optional[float] = None
    _cache_ttl: float = 60.0  # 60 seconds TTL
    
    @classmethod
    def load_config(cls, force_reload: bool = False) -> LightweightConfig:
        """Load configuration with caching for performance."""
        now = time.time()
        
        # Check cache validity
        if (not force_reload and 
            cls._cached_config is not None and 
            cls._cache_timestamp is not None and 
            now - cls._cache_timestamp < cls._cache_ttl):
            return cls._cached_config
        
        # Load fresh configuration
        start_time = time.time()
        config = cls._load_from_environment()
        load_time = time.time() - start_time
        
        # Cache the result
        cls._cached_config = config
        cls._cache_timestamp = now
        
        # Log performance if over threshold
        if load_time > 0.02:  # 20ms threshold
            print(f"⚠️ Config loading took {load_time:.3f}s")
        
        return config
    
    @classmethod
    def _load_from_environment(cls) -> LightweightConfig:
        """Load configuration from environment variables."""
        return LightweightConfig(
            # Core settings
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() in ("true", "1", "yes"),
            
            # Database
            database_url=os.getenv("DATABASE_URL", "sqlite:///./agent_hive.db"),
            
            # Redis
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            
            # API settings
            api_port=os.getenv("API_PORT", "18080"),
            api_host=os.getenv("API_HOST", "localhost"),
            
            # Security
            secret_key=os.getenv("SECRET_KEY", "dev-secret-key"),
            
            # External services
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            github_token=os.getenv("GITHUB_TOKEN"),
            
            # Sandbox mode
            sandbox_mode=os.getenv("SANDBOX_MODE", "false").lower() in ("true", "1", "yes"),
            
            # Performance settings
            max_workers=int(os.getenv("MAX_WORKERS", "4")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "300"))
        )
    
    @classmethod
    def clear_cache(cls):
        """Clear cached configuration."""
        cls._cached_config = None
        cls._cache_timestamp = None
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """Get cache statistics."""
        now = time.time()
        is_cached = cls._cached_config is not None
        age = now - cls._cache_timestamp if cls._cache_timestamp else None
        is_expired = age and age > cls._cache_ttl
        
        return {
            'cached': is_cached,
            'age_seconds': age,
            'expired': is_expired,
            'ttl_seconds': cls._cache_ttl,
            'cache_timestamp': cls._cache_timestamp
        }


# Compatibility layer for existing code
class CompatibilityWrapper:
    """
    Wrapper to make LightweightConfig compatible with existing configuration patterns.
    
    This allows CLI code to use the fast loader while maintaining compatibility
    with existing code that expects the full configuration service interface.
    """
    
    def __init__(self, lightweight_config: LightweightConfig):
        self._config = lightweight_config
    
    @property
    def config(self) -> LightweightConfig:
        """Get configuration (primary interface)."""
        return self._config
    
    def get_settings(self) -> LightweightConfig:
        """Backwards compatibility method."""
        return self._config
    
    def get_database_url(self) -> str:
        """Get database URL."""
        return self._config.get_database_url()
    
    def get_redis_url(self) -> str:
        """Get Redis URL."""
        return self._config.get_redis_url()
    
    def is_production(self) -> bool:
        """Check if production environment."""
        return self._config.is_production
    
    def is_development(self) -> bool:
        """Check if development environment."""
        return self._config.is_development
    
    def get_anthropic_api_key(self) -> Optional[str]:
        """Get Anthropic API key."""
        return self._config.anthropic_api_key
    
    def get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key."""
        return self._config.openai_api_key
    
    def get_github_token(self) -> Optional[str]:
        """Get GitHub token."""
        return self._config.github_token


# Global cached instances for CLI performance
_global_config: Optional[LightweightConfig] = None
_global_config_service: Optional[CompatibilityWrapper] = None


@lru_cache(maxsize=1)
def get_fast_config() -> LightweightConfig:
    """Get cached lightweight configuration."""
    return FastConfigLoader.load_config()


def get_fast_config_service() -> CompatibilityWrapper:
    """Get fast configuration service for CLI commands."""
    global _global_config_service
    
    if _global_config_service is None:
        config = get_fast_config()
        _global_config_service = CompatibilityWrapper(config)
    
    return _global_config_service


def clear_fast_config_cache():
    """Clear fast configuration cache."""
    global _global_config, _global_config_service
    
    FastConfigLoader.clear_cache()
    _global_config = None
    _global_config_service = None
    
    # Clear lru_cache
    get_fast_config.cache_clear()


def get_fast_config_stats() -> Dict[str, Any]:
    """Get fast configuration performance statistics."""
    return {
        'loader_stats': FastConfigLoader.get_cache_stats(),
        'lru_cache_info': get_fast_config.cache_info()._asdict(),
        'global_cached': _global_config_service is not None
    }


# Performance measurement utilities
class ConfigLoadTimer:
    """Timer for measuring configuration load performance."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        load_time = self.end_time - self.start_time
        
        if load_time > 0.02:  # 20ms threshold
            print(f"🐌 Config load: {load_time*1000:.1f}ms (target: <20ms)")
        elif load_time < 0.005:  # Under 5ms is excellent
            print(f"⚡ Config load: {load_time*1000:.1f}ms")
    
    @property
    def execution_time_ms(self) -> Optional[float]:
        """Get execution time in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None


def benchmark_config_loading(iterations: int = 10) -> Dict[str, Any]:
    """Benchmark configuration loading performance."""
    times = []
    
    for _ in range(iterations):
        # Clear cache to measure cold load time
        clear_fast_config_cache()
        
        start_time = time.time()
        config = get_fast_config()
        end_time = time.time()
        
        times.append((end_time - start_time) * 1000)  # Convert to ms
    
    return {
        'iterations': iterations,
        'average_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'all_times_ms': times,
        'under_20ms_count': len([t for t in times if t < 20]),
        'success_rate': len([t for t in times if t < 20]) / len(times)
    }