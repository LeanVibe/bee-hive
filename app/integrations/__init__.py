"""
Framework Integration Adapters for Project Index

Provides minimal-friction integration patterns for popular web frameworks
and project types, allowing seamless Project Index integration with 1-3 lines of code.
"""

from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from ..project_index import ProjectIndexer, AnalysisConfiguration, ProjectIndexConfig


class BaseFrameworkAdapter(ABC):
    """
    Abstract base class for framework adapters.
    
    All framework adapters should inherit from this class and implement
    the required methods to provide consistent integration patterns.
    """
    
    def __init__(self, config: Optional[ProjectIndexConfig] = None):
        """
        Initialize adapter with optional configuration.
        
        Args:
            config: Project Index configuration. If None, uses smart defaults.
        """
        self.config = config or self._create_default_config()
        self.indexer: Optional[ProjectIndexer] = None
        self._initialized = False
    
    @abstractmethod
    def integrate(self, app: Any, **kwargs) -> None:
        """
        Integrate Project Index with the framework application.
        
        Args:
            app: Framework application instance
            **kwargs: Framework-specific configuration options
        """
        pass
    
    @abstractmethod
    def _setup_routes(self, app: Any) -> None:
        """Setup framework-specific routes for Project Index API."""
        pass
    
    @abstractmethod
    def _setup_middleware(self, app: Any) -> None:
        """Setup framework-specific middleware for request/response handling."""
        pass
    
    def _create_default_config(self) -> ProjectIndexConfig:
        """Create optimized default configuration for the framework."""
        return ProjectIndexConfig(
            cache_enabled=True,
            context_optimization_enabled=True,
            monitoring_enabled=True,
            incremental_updates=True,
            max_concurrent_analyses=4
        )
    
    async def start(self) -> None:
        """Start the Project Index service."""
        if not self._initialized:
            self.indexer = ProjectIndexer(config=self.config)
            await self.indexer.__aenter__()
            self._initialized = True
    
    async def stop(self) -> None:
        """Stop the Project Index service."""
        if self._initialized and self.indexer:
            await self.indexer.__aexit__(None, None, None)
            self._initialized = False


class IntegrationManager:
    """
    Central manager for framework integrations.
    
    Provides utilities for discovering, configuring, and managing
    different framework adapters.
    """
    
    _adapters: Dict[str, type] = {}
    
    @classmethod
    def register_adapter(cls, framework_name: str, adapter_class: type) -> None:
        """
        Register a framework adapter.
        
        Args:
            framework_name: Name of the framework (e.g., 'fastapi', 'django')
            adapter_class: Adapter class for the framework
        """
        cls._adapters[framework_name] = adapter_class
    
    @classmethod
    def get_adapter(cls, framework_name: str) -> Optional[type]:
        """
        Get adapter class for a framework.
        
        Args:
            framework_name: Name of the framework
            
        Returns:
            Adapter class or None if not found
        """
        return cls._adapters.get(framework_name)
    
    @classmethod
    def list_supported_frameworks(cls) -> List[str]:
        """
        Get list of supported frameworks.
        
        Returns:
            List of framework names
        """
        return list(cls._adapters.keys())
    
    @classmethod
    def create_adapter(cls, framework_name: str, config: Optional[ProjectIndexConfig] = None) -> Optional[BaseFrameworkAdapter]:
        """
        Create adapter instance for a framework.
        
        Args:
            framework_name: Name of the framework
            config: Optional configuration
            
        Returns:
            Adapter instance or None if framework not supported
        """
        adapter_class = cls.get_adapter(framework_name)
        if adapter_class:
            return adapter_class(config=config)
        return None


def detect_framework() -> Optional[str]:
    """
    Auto-detect the framework in the current environment.
    
    Returns:
        Framework name if detected, None otherwise
    """
    try:
        # FastAPI detection
        import fastapi
        return 'fastapi'
    except ImportError:
        pass
    
    try:
        # Django detection
        import django
        return 'django'
    except ImportError:
        pass
    
    try:
        # Flask detection
        import flask
        return 'flask'
    except ImportError:
        pass
    
    try:
        # Express.js detection (through subprocess or package.json)
        import json
        import os
        if os.path.exists('package.json'):
            with open('package.json', 'r') as f:
                package_data = json.load(f)
                dependencies = {**package_data.get('dependencies', {}), **package_data.get('devDependencies', {})}
                if 'express' in dependencies:
                    return 'express'
                elif 'next' in dependencies:
                    return 'nextjs'
                elif 'react' in dependencies:
                    return 'react'
                elif 'vue' in dependencies:
                    return 'vue'
                elif '@angular/core' in dependencies:
                    return 'angular'
    except Exception:
        pass
    
    return None


def quick_integrate(app: Any = None, framework: Optional[str] = None, config: Optional[ProjectIndexConfig] = None) -> Optional[BaseFrameworkAdapter]:
    """
    Quick integration helper that auto-detects framework and sets up Project Index.
    
    Args:
        app: Framework application instance (optional for auto-detection)
        framework: Framework name (optional, will auto-detect if not provided)
        config: Project Index configuration (optional)
        
    Returns:
        Configured adapter instance or None if integration failed
        
    Example:
        # Auto-detect and integrate
        adapter = quick_integrate(app)
        
        # Explicit framework
        adapter = quick_integrate(app, framework='fastapi')
    """
    if not framework:
        framework = detect_framework()
    
    if not framework:
        raise ValueError("Could not detect framework. Please specify framework explicitly.")
    
    adapter = IntegrationManager.create_adapter(framework, config)
    if adapter and app:
        adapter.integrate(app)
    
    return adapter


# Export main components
__all__ = [
    'BaseFrameworkAdapter',
    'IntegrationManager', 
    'detect_framework',
    'quick_integrate'
]