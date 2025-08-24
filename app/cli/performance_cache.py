"""
CLI Performance Optimization Cache

Provides caching mechanisms to improve CLI command execution speed
by avoiding repeated initialization of heavy components.
"""

import time
from typing import Optional, Any
from pathlib import Path
import json


class CLIPerformanceCache:
    """
    Performance cache for CLI operations.
    
    Caches expensive operations like configuration loading to improve
    CLI command execution speed from >700ms to <500ms target.
    """
    
    def __init__(self):
        self._config_cache: Optional[Any] = None
        self._config_timestamp: Optional[float] = None
        self._cache_ttl = 300  # 5 minutes TTL for configuration
        
    def get_cached_config(self):
        """Get cached configuration service, avoiding expensive reinitialization."""
        now = time.time()
        
        # Check if cache is valid
        if (self._config_cache is not None and 
            self._config_timestamp is not None and 
            now - self._config_timestamp < self._cache_ttl):
            return self._config_cache
            
        # Cache miss or expired - lazy load configuration
        try:
            from ..core.configuration_service import get_configuration_service
            self._config_cache = get_configuration_service()
            self._config_timestamp = now
            return self._config_cache
        except ImportError:
            return None
    
    def clear_cache(self):
        """Clear all caches - useful for testing or forced refresh."""
        self._config_cache = None
        self._config_timestamp = None


# Global cache instance for CLI performance optimization
_cli_cache = CLIPerformanceCache()


def get_cached_config():
    """Get cached configuration service for CLI commands."""
    return _cli_cache.get_cached_config()


def clear_cli_cache():
    """Clear CLI performance cache."""
    _cli_cache.clear_cache()