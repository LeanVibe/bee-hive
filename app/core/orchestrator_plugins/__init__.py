"""
Orchestrator Plugin Architecture for LeanVibe Agent Hive 2.0

Epic 1 Phase 2.2: Consolidated plugin architecture
This module provides a plugin-based architecture for extending orchestrator functionality
while maintaining a clean separation of concerns.

Legacy Plugin Types (maintained for backward compatibility):
- PerformancePlugin: Performance monitoring, optimization, and resource management
- SecurityPlugin: Authentication, authorization, and security monitoring  
- ContextPlugin: Context compression, memory management, and session optimization

New Epic 1 Phase 2.2 Plugins:
- DemoOrchestratorPlugin: Realistic multi-agent development scenarios
- MasterOrchestratorPlugin: Advanced orchestration with production monitoring
- ManagementOrchestratorPlugin: Project management system integration
- MigrationOrchestratorPlugin: Backward compatibility layer
- UnifiedOrchestratorPlugin: Advanced multi-agent coordination
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

# Import new base classes
from .base_plugin import OrchestratorPlugin as NewOrchestratorPlugin, PluginMetadata as NewPluginMetadata, PluginError

# Import Epic 1 Phase 2.2 plugins
from .demo_orchestrator_plugin import DemoOrchestratorPlugin, create_demo_orchestrator_plugin
from .master_orchestrator_plugin import MasterOrchestratorPlugin, create_master_orchestrator_plugin
from .management_orchestrator_plugin import ManagementOrchestratorPlugin, create_management_orchestrator_plugin  
from .migration_orchestrator_plugin import MigrationOrchestratorPlugin, create_migration_orchestrator_plugin
from .unified_orchestrator_plugin import UnifiedOrchestratorPlugin, create_unified_orchestrator_plugin


class PluginType(Enum):
    """Types of orchestrator plugins (legacy compatibility)."""
    PERFORMANCE = "performance"
    SECURITY = "security"
    CONTEXT = "context"
    WORKFLOW = "workflow"
    COMMUNICATION = "communication"
    # New Epic 1 Phase 2.2 plugin types
    DEMO = "demo"
    MASTER = "master"
    MANAGEMENT = "management"
    MIGRATION = "migration"
    UNIFIED = "unified"


@dataclass
class PluginMetadata:
    """Metadata for orchestrator plugins (legacy compatibility)."""
    name: str
    version: str
    plugin_type: PluginType
    description: str
    dependencies: List[str]
    enabled: bool = True


class OrchestratorPlugin(ABC):
    """Base class for all orchestrator plugins (legacy compatibility)."""
    
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
    """Manager for orchestrator plugins with Epic 1 Phase 2.2 support."""
    
    def __init__(self):
        # Legacy plugin storage
        self._plugins: Dict[PluginType, List[OrchestratorPlugin]] = {}
        self._plugin_registry: Dict[str, OrchestratorPlugin] = {}
        
        # New Epic 1 Phase 2.2 plugin storage
        self._new_plugins: Dict[str, NewOrchestratorPlugin] = {}
        
        # Performance tracking
        self._plugin_call_count = 0
        
    def register_plugin(self, plugin: OrchestratorPlugin):
        """Register a legacy plugin with the manager."""
        plugin_type = plugin.metadata.plugin_type
        if plugin_type not in self._plugins:
            self._plugins[plugin_type] = []
            
        self._plugins[plugin_type].append(plugin)
        self._plugin_registry[plugin.metadata.name] = plugin
        
    def register_new_plugin(self, plugin: NewOrchestratorPlugin):
        """Register an Epic 1 Phase 2.2 plugin with the manager."""
        self._new_plugins[plugin.metadata.name] = plugin
        
    def get_plugins(self, plugin_type: PluginType) -> List[OrchestratorPlugin]:
        """Get all legacy plugins of a specific type."""
        return self._plugins.get(plugin_type, [])
        
    def get_plugin(self, name: str) -> Optional[OrchestratorPlugin]:
        """Get a legacy plugin by name."""
        return self._plugin_registry.get(name)
        
    def get_new_plugin(self, name: str) -> Optional[NewOrchestratorPlugin]:
        """Get an Epic 1 Phase 2.2 plugin by name."""
        return self._new_plugins.get(name)
        
    def get_all_new_plugins(self) -> List[NewOrchestratorPlugin]:
        """Get all Epic 1 Phase 2.2 plugins."""
        return list(self._new_plugins.values())
        
    async def initialize_all(self, orchestrator_context: Dict[str, Any]):
        """Initialize all registered plugins (legacy and new)."""
        # Initialize legacy plugins
        for plugins in self._plugins.values():
            for plugin in plugins:
                if plugin.enabled:
                    await plugin.initialize(orchestrator_context)
                    
        # Initialize Epic 1 Phase 2.2 plugins
        for plugin in self._new_plugins.values():
            if plugin.enabled:
                try:
                    await plugin.initialize(orchestrator_context)
                except Exception as e:
                    # Log error but continue with other plugins
                    print(f"Failed to initialize plugin {plugin.metadata.name}: {e}")
                    
    async def cleanup_all(self):
        """Cleanup all plugins (legacy and new)."""
        # Cleanup legacy plugins
        for plugins in self._plugins.values():
            for plugin in plugins:
                await plugin.cleanup()
                
        # Cleanup Epic 1 Phase 2.2 plugins
        for plugin in self._new_plugins.values():
            try:
                await plugin.cleanup()
            except Exception as e:
                # Log error but continue cleanup
                print(f"Failed to cleanup plugin {plugin.metadata.name}: {e}")
                
    async def execute_hooks(self, hook_name: str, context: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute hooks across all enabled plugins."""
        result = context
        self._plugin_call_count += 1
        
        # Execute legacy plugin hooks
        for plugins in self._plugins.values():
            for plugin in plugins:
                if plugin.enabled and hasattr(plugin, hook_name):
                    hook_method = getattr(plugin, hook_name)
                    result = await hook_method(result, **kwargs)
        
        # Execute Epic 1 Phase 2.2 plugin hooks
        for plugin in self._new_plugins.values():
            if plugin.enabled and hasattr(plugin, hook_name):
                hook_method = getattr(plugin, hook_name)
                result = await hook_method(result, **kwargs)
                    
        return result
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all plugins."""
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "legacy_plugins": {
                "count": len(self._plugin_registry),
                "by_type": {plugin_type.value: len(plugins) for plugin_type, plugins in self._plugins.items()},
                "plugins": []
            },
            "epic1_plugins": {
                "count": len(self._new_plugins),
                "plugins": []
            },
            "performance": {
                "total_plugin_calls": self._plugin_call_count
            }
        }
        
        # Get legacy plugin status
        for plugin in self._plugin_registry.values():
            plugin_status = await plugin.health_check()
            status["legacy_plugins"]["plugins"].append(plugin_status)
        
        # Get Epic 1 Phase 2.2 plugin status
        for plugin in self._new_plugins.values():
            try:
                plugin_status = await plugin.health_check()
                plugin_performance = await plugin.get_performance_metrics()
                status["epic1_plugins"]["plugins"].append({
                    **plugin_status,
                    "performance": plugin_performance
                })
            except Exception as e:
                status["epic1_plugins"]["plugins"].append({
                    "plugin": plugin.metadata.name,
                    "status": "error",
                    "error": str(e)
                })
        
        return status


# Global plugin manager instance
_plugin_manager = PluginManager()


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    return _plugin_manager


def initialize_epic1_plugins(orchestrator_context: Dict[str, Any]) -> Dict[str, NewOrchestratorPlugin]:
    """
    Initialize all Epic 1 Phase 2.2 plugins with the orchestrator context.
    
    Args:
        orchestrator_context: Context containing orchestrator instance and config
        
    Returns:
        Dictionary of initialized plugins
    """
    plugins = {}
    
    try:
        # Create and register demo orchestrator plugin
        demo_plugin = create_demo_orchestrator_plugin()
        _plugin_manager.register_new_plugin(demo_plugin)
        plugins["demo"] = demo_plugin
        
        # Create and register master orchestrator plugin
        master_plugin = create_master_orchestrator_plugin()
        _plugin_manager.register_new_plugin(master_plugin)
        plugins["master"] = master_plugin
        
        # Create and register management orchestrator plugin
        management_plugin = create_management_orchestrator_plugin()
        _plugin_manager.register_new_plugin(management_plugin)
        plugins["management"] = management_plugin
        
        # Create and register migration orchestrator plugin
        migration_plugin = create_migration_orchestrator_plugin()
        _plugin_manager.register_new_plugin(migration_plugin)
        plugins["migration"] = migration_plugin
        
        # Create and register unified orchestrator plugin
        unified_plugin = create_unified_orchestrator_plugin()
        _plugin_manager.register_new_plugin(unified_plugin)
        plugins["unified"] = unified_plugin
        
        print(f"Epic 1 Phase 2.2: Initialized {len(plugins)} orchestrator plugins")
        
    except Exception as e:
        print(f"Failed to initialize Epic 1 Phase 2.2 plugins: {e}")
        
    return plugins


# Export everything for backward compatibility and Epic 1 Phase 2.2 support
__all__ = [
    # Legacy exports
    'PluginType',
    'PluginMetadata', 
    'OrchestratorPlugin',
    'PluginManager',
    'get_plugin_manager',
    
    # Epic 1 Phase 2.2 exports
    'NewOrchestratorPlugin',
    'NewPluginMetadata', 
    'PluginError',
    'DemoOrchestratorPlugin',
    'MasterOrchestratorPlugin',
    'ManagementOrchestratorPlugin',
    'MigrationOrchestratorPlugin', 
    'UnifiedOrchestratorPlugin',
    'create_demo_orchestrator_plugin',
    'create_master_orchestrator_plugin',
    'create_management_orchestrator_plugin',
    'create_migration_orchestrator_plugin',
    'create_unified_orchestrator_plugin',
    'initialize_epic1_plugins'
]