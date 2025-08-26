"""
Base Plugin Architecture for LeanVibe Agent Hive 2.0

Epic 1 Phase 2.2: Consolidated plugin architecture
Provides base classes and interfaces for all orchestrator plugins.

Key Features:
- Abstract base plugin interface
- Plugin metadata management
- Performance monitoring integration
- Cleanup and lifecycle management
- Error handling with PluginError

Epic 1 Performance Targets:
- <1ms plugin hook overhead
- Memory-efficient metadata storage
- Lazy loading plugin capabilities
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime


class PluginError(Exception):
    """Exception raised by plugin operations."""
    pass


@dataclass
class PluginMetadata:
    """Metadata for orchestrator plugins."""
    name: str
    version: str
    description: str
    author: str
    capabilities: List[str]
    dependencies: List[str]
    epic_phase: str
    enabled: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class OrchestratorPlugin(ABC):
    """
    Base class for all orchestrator plugins in Epic 1 Phase 2.2.
    
    Provides common interface and functionality for:
    - Plugin lifecycle management
    - Performance monitoring
    - Error handling
    - Integration with SimpleOrchestrator
    """
    
    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self._enabled = metadata.enabled
        self._initialized = False
        
        # Performance tracking
        self._hook_call_count = 0
        self._initialization_time = None
        
    @property
    def enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self._enabled
    
    @property
    def initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized
    
    def enable(self):
        """Enable the plugin."""
        self._enabled = True
        
    def disable(self):
        """Disable the plugin."""
        self._enabled = False
    
    @abstractmethod
    async def initialize(self, context: Dict[str, Any]) -> None:
        """
        Initialize the plugin with orchestrator context.
        
        Args:
            context: Dictionary containing orchestrator and configuration
        """
        self._initialization_time = datetime.utcnow()
        self._initialized = True
    
    async def pre_task_execution(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called before task execution.
        
        Args:
            task_context: Task context dictionary
            
        Returns:
            Modified task context
        """
        if not self._enabled:
            return task_context
        
        self._hook_call_count += 1
        return task_context
    
    async def post_task_execution(self, task_context: Dict[str, Any], result: Any) -> Any:
        """
        Hook called after task execution.
        
        Args:
            task_context: Task context dictionary
            result: Task execution result
            
        Returns:
            Modified result
        """
        if not self._enabled:
            return result
            
        self._hook_call_count += 1
        return result
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Return plugin health status.
        
        Returns:
            Health status dictionary
        """
        return {
            "plugin": self.metadata.name,
            "version": self.metadata.version,
            "enabled": self.enabled,
            "initialized": self.initialized,
            "status": "healthy" if self.enabled and self.initialized else "inactive",
            "hook_calls": self._hook_call_count,
            "uptime_seconds": (datetime.utcnow() - self._initialization_time).total_seconds() if self._initialization_time else 0
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get plugin performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        return {
            "plugin_name": self.metadata.name,
            "hook_calls": self._hook_call_count,
            "initialization_time": self._initialization_time.isoformat() if self._initialization_time else None,
            "capabilities": len(self.metadata.capabilities),
            "dependencies": len(self.metadata.dependencies)
        }
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup plugin resources.
        Must be implemented by all plugins.
        """
        self._initialized = False
        self._hook_call_count = 0
        self._initialization_time = None


# Export base classes
__all__ = [
    'OrchestratorPlugin',
    'PluginMetadata', 
    'PluginError'
]