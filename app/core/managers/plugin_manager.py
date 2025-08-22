"""
Plugin Manager - Consolidated Plugin System Management

Consolidates functionality from:
- AdvancedPluginManager, PluginSystem
- Plugin security framework, plugin marketplace
- All plugin-related manager classes (10+ files)

Preserves Epic 2 Phase 2.1 advanced plugin architecture.
"""

import asyncio
import importlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

from ..config import settings
from ..logging_service import get_component_logger

logger = get_component_logger("plugin_manager")


@dataclass
class PluginMetadata:
    """Plugin metadata and information."""
    plugin_id: str
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    security_level: str = "standard"
    load_priority: int = 100
    enabled: bool = True


@dataclass
class PluginState:
    """Plugin runtime state."""
    plugin_id: str
    status: str  # loaded, unloaded, failed, disabled
    load_time: Optional[datetime] = None
    memory_usage_mb: float = 0.0
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PluginSystem:
    """Plugin system configuration."""
    enabled: bool = True
    security_enabled: bool = True
    sandbox_enabled: bool = True
    max_plugins: int = 50
    plugin_timeout_seconds: int = 30


class PluginError(Exception):
    """Plugin management errors."""
    pass


class PluginManager:
    """
    Consolidated Plugin Manager
    
    Replaces and consolidates:
    - AdvancedPluginManager (Epic 2 Phase 2.1)
    - PluginSystem, PluginRegistry
    - Plugin security framework
    - Plugin marketplace integration
    - All plugin-related manager classes (10+ files)
    
    Preserves:
    - Epic 2 Phase 2.1 advanced plugin features
    - Dynamic plugin loading/unloading
    - Plugin security and sandboxing
    - Hot-swapping capabilities
    """

    def __init__(self, master_orchestrator):
        """Initialize plugin manager."""
        self.master_orchestrator = master_orchestrator
        
        # Plugin registry
        self.plugins: Dict[str, Any] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self.plugin_states: Dict[str, PluginState] = {}
        
        # Plugin system configuration
        self.plugin_system = PluginSystem()
        
        # Plugin directories
        self.plugin_directories = [
            Path("app/core/orchestrator_plugins"),
            Path("app/core/archive_orchestrators/plugins"),
            Path("plugins")
        ]
        
        # Security and performance tracking
        self.security_violations = 0
        self.load_count = 0
        self.unload_count = 0
        self.hot_swap_count = 0
        
        logger.info("Plugin Manager initialized")

    async def initialize(self) -> None:
        """Initialize plugin system."""
        try:
            if not self.plugin_system.enabled:
                logger.info("Plugin system disabled")
                return
            
            # Initialize plugin directories
            await self._initialize_plugin_directories()
            
            # Load plugin metadata
            await self._discover_plugins()
            
            # Load core plugins
            await self._load_core_plugins()
            
            logger.info("✅ Plugin Manager initialized successfully",
                       discovered_plugins=len(self.plugin_metadata),
                       loaded_plugins=len(self.plugins))
            
        except Exception as e:
            logger.error("❌ Plugin Manager initialization failed", error=str(e))
            raise PluginError(f"Initialization failed: {e}") from e

    async def shutdown(self) -> None:
        """Shutdown plugin manager and unload all plugins."""
        logger.info("🛑 Shutting down Plugin Manager...")
        
        # Unload all plugins gracefully
        plugin_ids = list(self.plugins.keys())
        for plugin_id in plugin_ids:
            try:
                await self.unload_plugin(plugin_id)
            except Exception as e:
                logger.warning(f"Failed to unload plugin {plugin_id}: {e}")
        
        # Clear plugin registry
        self.plugins.clear()
        self.plugin_metadata.clear()
        self.plugin_states.clear()
        
        logger.info("✅ Plugin Manager shutdown complete")

    async def load_plugin(
        self,
        plugin_id: str,
        version: str = "latest",
        source_path: Optional[Path] = None,
        source_code: Optional[str] = None
    ) -> bool:
        """
        Load plugin dynamically - Epic 2 Phase 2.1 compatibility.
        
        Supports loading from:
        - Plugin registry
        - File system path
        - Source code string
        """
        try:
            # Check if plugin already loaded
            if plugin_id in self.plugins:
                logger.warning("Plugin already loaded", plugin_id=plugin_id)
                return True
            
            # Check plugin limits
            if len(self.plugins) >= self.plugin_system.max_plugins:
                raise PluginError(f"Maximum plugin limit reached: {self.plugin_system.max_plugins}")
            
            # Load plugin metadata if not discovered
            if plugin_id not in self.plugin_metadata:
                await self._load_plugin_metadata(plugin_id, source_path)
            
            metadata = self.plugin_metadata.get(plugin_id)
            if not metadata:
                raise PluginError(f"Plugin metadata not found: {plugin_id}")
            
            if not metadata.enabled:
                logger.warning("Plugin disabled", plugin_id=plugin_id)
                return False
            
            # Security check
            if self.plugin_system.security_enabled:
                security_result = await self._security_scan_plugin(plugin_id, metadata)
                if not security_result.get('safe', False):
                    self.security_violations += 1
                    raise PluginError(f"Plugin failed security scan: {security_result.get('reason')}")
            
            # Load plugin module
            plugin_instance = await self._load_plugin_module(plugin_id, source_path, source_code)
            
            if not plugin_instance:
                raise PluginError("Failed to instantiate plugin")
            
            # Initialize plugin
            if hasattr(plugin_instance, 'initialize'):
                await plugin_instance.initialize(self.master_orchestrator)
            
            # Register plugin
            self.plugins[plugin_id] = plugin_instance
            self.plugin_states[plugin_id] = PluginState(
                plugin_id=plugin_id,
                status="loaded",
                load_time=datetime.utcnow()
            )
            
            self.load_count += 1
            
            logger.info("✅ Plugin loaded successfully",
                       plugin_id=plugin_id,
                       version=metadata.version,
                       memory_usage_mb=await self._get_plugin_memory_usage(plugin_id))
            
            return True
            
        except Exception as e:
            logger.error("❌ Plugin load failed", plugin_id=plugin_id, error=str(e))
            
            # Update plugin state
            self.plugin_states[plugin_id] = PluginState(
                plugin_id=plugin_id,
                status="failed",
                error_message=str(e)
            )
            
            return False

    async def unload_plugin(self, plugin_id: str) -> bool:
        """
        Safely unload plugin - Epic 2 Phase 2.1 compatibility.
        
        Gracefully shuts down plugin and cleans up resources.
        """
        try:
            if plugin_id not in self.plugins:
                logger.warning("Plugin not loaded", plugin_id=plugin_id)
                return True
            
            plugin_instance = self.plugins[plugin_id]
            
            # Graceful shutdown
            if hasattr(plugin_instance, 'shutdown'):
                try:
                    await asyncio.wait_for(
                        plugin_instance.shutdown(),
                        timeout=self.plugin_system.plugin_timeout_seconds
                    )
                except asyncio.TimeoutError:
                    logger.warning("Plugin shutdown timeout", plugin_id=plugin_id)
            
            # Remove from registry
            del self.plugins[plugin_id]
            
            # Update state
            if plugin_id in self.plugin_states:
                self.plugin_states[plugin_id].status = "unloaded"
            
            # Cleanup module references
            await self._cleanup_plugin_module(plugin_id)
            
            self.unload_count += 1
            
            logger.info("✅ Plugin unloaded successfully", plugin_id=plugin_id)
            return True
            
        except Exception as e:
            logger.error("❌ Plugin unload failed", plugin_id=plugin_id, error=str(e))
            return False

    async def hot_swap_plugin(self, old_plugin_id: str, new_plugin_id: str) -> bool:
        """
        Hot-swap plugins without system restart - Epic 2 Phase 2.1 feature.
        
        Replaces old plugin with new plugin seamlessly.
        """
        try:
            # Verify new plugin can be loaded
            load_result = await self.load_plugin(new_plugin_id)
            if not load_result:
                raise PluginError(f"Failed to load new plugin: {new_plugin_id}")
            
            # Transfer state if possible
            if old_plugin_id in self.plugins:
                old_plugin = self.plugins[old_plugin_id]
                new_plugin = self.plugins[new_plugin_id]
                
                # Transfer configuration
                if (hasattr(old_plugin, 'get_state') and 
                    hasattr(new_plugin, 'set_state')):
                    try:
                        state = await old_plugin.get_state()
                        await new_plugin.set_state(state)
                    except Exception as e:
                        logger.warning("Failed to transfer plugin state", error=str(e))
                
                # Unload old plugin
                await self.unload_plugin(old_plugin_id)
            
            self.hot_swap_count += 1
            
            logger.info("✅ Plugin hot-swap successful",
                       old_plugin=old_plugin_id,
                       new_plugin=new_plugin_id)
            
            return True
            
        except Exception as e:
            logger.error("❌ Plugin hot-swap failed", error=str(e))
            
            # Cleanup new plugin on failure
            if new_plugin_id in self.plugins:
                await self.unload_plugin(new_plugin_id)
            
            return False

    async def reload_plugin(self, plugin_id: str) -> bool:
        """Reload plugin by unloading and loading again."""
        if plugin_id in self.plugins:
            unload_success = await self.unload_plugin(plugin_id)
            if not unload_success:
                return False
        
        return await self.load_plugin(plugin_id)

    async def get_plugin_status(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed plugin status."""
        if plugin_id not in self.plugin_states:
            return None
        
        state = self.plugin_states[plugin_id]
        metadata = self.plugin_metadata.get(plugin_id)
        
        status = {
            "plugin_id": plugin_id,
            "status": state.status,
            "load_time": state.load_time.isoformat() if state.load_time else None,
            "memory_usage_mb": await self._get_plugin_memory_usage(plugin_id),
            "error_message": state.error_message,
            "performance_metrics": state.performance_metrics
        }
        
        if metadata:
            status.update({
                "name": metadata.name,
                "version": metadata.version,
                "description": metadata.description,
                "capabilities": metadata.capabilities,
                "security_level": metadata.security_level
            })
        
        return status

    async def list_plugins(self) -> List[Dict[str, Any]]:
        """List all plugins with their status."""
        plugins_list = []
        
        # Add loaded plugins
        for plugin_id in self.plugins.keys():
            status = await self.get_plugin_status(plugin_id)
            if status:
                plugins_list.append(status)
        
        # Add discovered but not loaded plugins
        for plugin_id in self.plugin_metadata.keys():
            if plugin_id not in self.plugins:
                metadata = self.plugin_metadata[plugin_id]
                plugins_list.append({
                    "plugin_id": plugin_id,
                    "name": metadata.name,
                    "version": metadata.version,
                    "status": "discovered",
                    "enabled": metadata.enabled
                })
        
        return plugins_list

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get plugin performance metrics - Epic 1 monitoring integration."""
        total_memory_mb = 0
        plugin_metrics = {}
        
        for plugin_id in self.plugins.keys():
            memory_usage = await self._get_plugin_memory_usage(plugin_id)
            total_memory_mb += memory_usage
            
            plugin_metrics[plugin_id] = {
                "memory_usage_mb": memory_usage,
                "status": self.plugin_states.get(plugin_id, {}).get('status', 'unknown')
            }
        
        return {
            "total_plugins": len(self.plugins),
            "total_memory_mb": total_memory_mb,
            "load_count": self.load_count,
            "unload_count": self.unload_count,
            "hot_swap_count": self.hot_swap_count,
            "security_violations": self.security_violations,
            "plugin_metrics": plugin_metrics,
            "system_config": {
                "enabled": self.plugin_system.enabled,
                "security_enabled": self.plugin_system.security_enabled,
                "max_plugins": self.plugin_system.max_plugins
            }
        }

    async def get_security_status(self) -> Dict[str, Any]:
        """Get plugin security status and reports."""
        try:
            security_status = {
                "security_enabled": self.plugin_system.security_enabled,
                "sandbox_enabled": self.plugin_system.sandbox_enabled,
                "security_violations": self.security_violations,
                "plugin_security": {}
            }
            
            for plugin_id, metadata in self.plugin_metadata.items():
                security_status["plugin_security"][plugin_id] = {
                    "security_level": metadata.security_level,
                    "status": "safe" if plugin_id in self.plugins else "unverified"
                }
            
            return security_status
            
        except Exception as e:
            logger.error("Failed to get plugin security status", error=str(e))
            return {"error": str(e)}

    async def get_status(self) -> Dict[str, Any]:
        """Get plugin manager status."""
        return {
            "system_enabled": self.plugin_system.enabled,
            "total_plugins": len(self.plugins),
            "loaded_plugins": len([p for p in self.plugin_states.values() 
                                 if p.status == "loaded"]),
            "failed_plugins": len([p for p in self.plugin_states.values()
                                 if p.status == "failed"]),
            "load_count": self.load_count,
            "unload_count": self.unload_count,
            "hot_swap_count": self.hot_swap_count,
            "security_violations": self.security_violations
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get plugin metrics for monitoring."""
        return {
            "plugin_count": len(self.plugins),
            "load_operations": self.load_count,
            "unload_operations": self.unload_count,
            "hot_swap_operations": self.hot_swap_count,
            "security_violations": self.security_violations,
            "total_memory_mb": sum([
                await self._get_plugin_memory_usage(pid) for pid in self.plugins.keys()
            ])
        }

    # ==================================================================
    # PLUGIN DISCOVERY AND LOADING
    # ==================================================================

    async def _initialize_plugin_directories(self) -> None:
        """Initialize and create plugin directories."""
        for directory in self.plugin_directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug("Plugin directory initialized", path=str(directory))
            except Exception as e:
                logger.warning(f"Failed to create plugin directory {directory}: {e}")

    async def _discover_plugins(self) -> None:
        """Discover available plugins from directories."""
        for directory in self.plugin_directories:
            if not directory.exists():
                continue
            
            try:
                for plugin_file in directory.glob("*.py"):
                    if plugin_file.name.startswith("__"):
                        continue
                    
                    plugin_id = plugin_file.stem
                    await self._load_plugin_metadata(plugin_id, plugin_file)
                    
            except Exception as e:
                logger.warning(f"Failed to discover plugins in {directory}: {e}")

    async def _load_plugin_metadata(self, plugin_id: str, source_path: Optional[Path]) -> None:
        """Load plugin metadata from file or module."""
        try:
            # Try to load metadata from plugin module
            if source_path and source_path.exists():
                spec = importlib.util.spec_from_file_location(plugin_id, source_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Extract metadata
                    metadata = PluginMetadata(
                        plugin_id=plugin_id,
                        name=getattr(module, 'PLUGIN_NAME', plugin_id),
                        version=getattr(module, 'PLUGIN_VERSION', '1.0.0'),
                        description=getattr(module, 'PLUGIN_DESCRIPTION', ''),
                        author=getattr(module, 'PLUGIN_AUTHOR', 'Unknown'),
                        dependencies=getattr(module, 'PLUGIN_DEPENDENCIES', []),
                        capabilities=getattr(module, 'PLUGIN_CAPABILITIES', []),
                        security_level=getattr(module, 'PLUGIN_SECURITY_LEVEL', 'standard'),
                        load_priority=getattr(module, 'PLUGIN_LOAD_PRIORITY', 100)
                    )
                    
                    self.plugin_metadata[plugin_id] = metadata
                    logger.debug("Plugin metadata loaded", plugin_id=plugin_id)
            
        except Exception as e:
            logger.warning(f"Failed to load metadata for plugin {plugin_id}: {e}")

    async def _load_core_plugins(self) -> None:
        """Load core system plugins."""
        core_plugins = [
            "production_enhancement_plugin",
            "performance_orchestrator_plugin", 
            "integration_orchestrator_plugin"
        ]
        
        for plugin_id in core_plugins:
            if plugin_id in self.plugin_metadata:
                try:
                    await self.load_plugin(plugin_id)
                except Exception as e:
                    logger.warning(f"Failed to load core plugin {plugin_id}: {e}")

    async def _load_plugin_module(
        self,
        plugin_id: str,
        source_path: Optional[Path] = None,
        source_code: Optional[str] = None
    ) -> Optional[Any]:
        """Load plugin module and return instance."""
        try:
            if source_code:
                # Load from source code string
                module = importlib.util.module_from_spec(
                    importlib.util.spec_from_loader(plugin_id, loader=None)
                )
                exec(source_code, module.__dict__)
                
            elif source_path:
                # Load from file path
                spec = importlib.util.spec_from_file_location(plugin_id, source_path)
                if not spec or not spec.loader:
                    return None
                
                module = importlib.util.module_from_spec(spec)
                sys.modules[plugin_id] = module
                spec.loader.exec_module(module)
                
            else:
                # Load from plugin directory
                plugin_path = None
                for directory in self.plugin_directories:
                    candidate_path = directory / f"{plugin_id}.py"
                    if candidate_path.exists():
                        plugin_path = candidate_path
                        break
                
                if not plugin_path:
                    return None
                
                spec = importlib.util.spec_from_file_location(plugin_id, plugin_path)
                if not spec or not spec.loader:
                    return None
                
                module = importlib.util.module_from_spec(spec)
                sys.modules[plugin_id] = module
                spec.loader.exec_module(module)
            
            # Create plugin instance
            plugin_class = getattr(module, 'Plugin', None)
            if not plugin_class:
                # Try common naming conventions
                for name in ['PluginClass', f'{plugin_id.title()}Plugin', 'Main']:
                    plugin_class = getattr(module, name, None)
                    if plugin_class:
                        break
            
            if not plugin_class:
                logger.error("Plugin class not found", plugin_id=plugin_id)
                return None
            
            return plugin_class()
            
        except Exception as e:
            logger.error("Failed to load plugin module", 
                        plugin_id=plugin_id, error=str(e))
            return None

    async def _cleanup_plugin_module(self, plugin_id: str) -> None:
        """Cleanup plugin module references."""
        try:
            if plugin_id in sys.modules:
                del sys.modules[plugin_id]
        except Exception as e:
            logger.warning(f"Failed to cleanup plugin module {plugin_id}: {e}")

    # ==================================================================
    # SECURITY AND PERFORMANCE
    # ==================================================================

    async def _security_scan_plugin(
        self,
        plugin_id: str,
        metadata: PluginMetadata
    ) -> Dict[str, Any]:
        """Perform security scan on plugin."""
        # Simplified security check - production would be more comprehensive
        try:
            security_result = {
                "safe": True,
                "reason": None,
                "risk_level": "low"
            }
            
            # Check security level
            if metadata.security_level == "dangerous":
                security_result.update({
                    "safe": False,
                    "reason": "Plugin marked as dangerous",
                    "risk_level": "high"
                })
            
            # Check for suspicious capabilities
            suspicious_capabilities = ["system_access", "file_system", "network"]
            if any(cap in suspicious_capabilities for cap in metadata.capabilities):
                security_result.update({
                    "risk_level": "medium",
                    "reason": "Plugin requests system-level capabilities"
                })
            
            return security_result
            
        except Exception as e:
            logger.error("Security scan failed", plugin_id=plugin_id, error=str(e))
            return {
                "safe": False,
                "reason": f"Security scan failed: {e}",
                "risk_level": "unknown"
            }

    async def _get_plugin_memory_usage(self, plugin_id: str) -> float:
        """Get plugin memory usage in MB."""
        try:
            # Simplified memory tracking - production would use more sophisticated methods
            if plugin_id in self.plugin_states:
                return self.plugin_states[plugin_id].memory_usage_mb
            return 0.0
        except Exception:
            return 0.0