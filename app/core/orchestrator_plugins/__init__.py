"""
Orchestrator Plugin Architecture for LeanVibe Agent Hive 2.0

This module provides a plugin-based architecture for extending orchestrator functionality
while maintaining a clean separation of concerns.

Plugin Types:
- PerformancePlugin: Performance monitoring, optimization, and resource management
- SecurityPlugin: Authentication, authorization, and security monitoring  
- ContextPlugin: Context compression, memory management, and session optimization
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum


class PluginType(Enum):
    """Types of orchestrator plugins."""
    PERFORMANCE = "performance"
    SECURITY = "security"
    CONTEXT = "context"
    WORKFLOW = "workflow"
    COMMUNICATION = "communication"


@dataclass
class PluginMetadata:
    """Metadata for orchestrator plugins."""
    name: str
    version: str
    plugin_type: PluginType
    description: str
    dependencies: List[str]
    enabled: bool = True


class OrchestratorPlugin(ABC):
    """Base class for all orchestrator plugins."""
    
    def __init__(self, metadata: PluginMetadata):
        self.metadata = metadata
        self._enabled = metadata.enabled
        
    @property
    def enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self._enabled
    
    def enable(self):
        """Enable the plugin."""
        self._enabled = True
        
    def disable(self):
        """Disable the plugin."""
        self._enabled = False
        
    @abstractmethod
    async def initialize(self, orchestrator_context: Dict[str, Any]) -> bool:
        """Initialize the plugin with orchestrator context."""
        pass
        
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        pass
        
    async def pre_task_execution(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called before task execution."""
        return task_context
        
    async def post_task_execution(self, task_context: Dict[str, Any], result: Any) -> Any:
        """Hook called after task execution."""
        return result
        
    async def health_check(self) -> Dict[str, Any]:
        """Return plugin health status."""
        return {
            "plugin": self.metadata.name,
            "enabled": self.enabled,
            "status": "healthy"
        }


class PluginManager:
    """Manager for orchestrator plugins."""
    
    def __init__(self):
        self._plugins: Dict[PluginType, List[OrchestratorPlugin]] = {}
        self._plugin_registry: Dict[str, OrchestratorPlugin] = {}
        
    def register_plugin(self, plugin: OrchestratorPlugin):
        """Register a plugin with the manager."""
        plugin_type = plugin.metadata.plugin_type
        if plugin_type not in self._plugins:
            self._plugins[plugin_type] = []
            
        self._plugins[plugin_type].append(plugin)
        self._plugin_registry[plugin.metadata.name] = plugin
        
    def get_plugins(self, plugin_type: PluginType) -> List[OrchestratorPlugin]:
        """Get all plugins of a specific type."""
        return self._plugins.get(plugin_type, [])
        
    def get_plugin(self, name: str) -> Optional[OrchestratorPlugin]:
        """Get a plugin by name."""
        return self._plugin_registry.get(name)
        
    async def initialize_all(self, orchestrator_context: Dict[str, Any]):
        """Initialize all registered plugins."""
        for plugins in self._plugins.values():
            for plugin in plugins:
                if plugin.enabled:
                    await plugin.initialize(orchestrator_context)
                    
    async def cleanup_all(self):
        """Cleanup all plugins."""
        for plugins in self._plugins.values():
            for plugin in plugins:
                await plugin.cleanup()
                
    async def execute_hooks(self, hook_name: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute hooks across all enabled plugins."""
        result = context
        
        for plugins in self._plugins.values():
            for plugin in plugins:
                if plugin.enabled and hasattr(plugin, hook_name):
                    hook_method = getattr(plugin, hook_name)
                    result = await hook_method(result, **kwargs)
                    
        return result


# Global plugin manager instance
_plugin_manager = PluginManager()


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    return _plugin_manager