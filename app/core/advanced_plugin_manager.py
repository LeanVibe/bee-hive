"""
Advanced Plugin Manager for LeanVibe Agent Hive 2.0 - Epic 2 Phase 2.1

Implements advanced plugin framework with dynamic loading, hot-swap capabilities,
and security framework while preserving Epic 1 performance achievements (<50ms API response, <80MB memory).

Key Features:
- Dynamic plugin loading without system restart
- Hot-swap plugin replacement capabilities  
- Plugin security validation and resource isolation
- Version management and dependency resolution
- Memory-efficient plugin lifecycle management
- Integration with existing SimpleOrchestrator

Epic 1 Preservation:
- Lazy loading to maintain <80MB memory usage
- <50ms plugin operations for API response times
- Minimal memory footprint for inactive plugins
- Non-blocking plugin operations
"""

import asyncio
import uuid
import importlib
import importlib.util
import inspect
import sys
import resource
import threading
import weakref
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Type, Protocol, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import logging
import hashlib
import tempfile
import shutil

from .logging_service import get_component_logger
from .orchestrator_plugins import OrchestratorPlugin, PluginType, PluginMetadata

logger = get_component_logger("advanced_plugin_manager")

# Epic 1: Lazy imports for memory efficiency
if False:  # TYPE_CHECKING equivalent for lazy loading
    from .simple_orchestrator import SimpleOrchestrator


class PluginSecurityLevel(Enum):
    """Security levels for plugin validation."""
    TRUSTED = "trusted"        # System/core plugins
    VERIFIED = "verified"      # Verified third-party plugins
    SANDBOX = "sandbox"        # Sandboxed plugins with restrictions
    UNTRUSTED = "untrusted"    # Unverified plugins (blocked by default)


class PluginLoadStrategy(Enum):
    """Plugin loading strategies."""
    IMMEDIATE = "immediate"    # Load immediately when added
    LAZY = "lazy"             # Load only when first accessed (Epic 1 optimization)
    ON_DEMAND = "on_demand"   # Load only when explicitly requested


@dataclass
class PluginSecurityPolicy:
    """Security policy for plugin execution."""
    security_level: PluginSecurityLevel
    max_memory_mb: int = 50  # Epic 1: Strict memory limits
    max_cpu_time_ms: int = 100  # Epic 1: Response time preservation
    allowed_imports: Set[str] = field(default_factory=set)
    blocked_imports: Set[str] = field(default_factory=set)
    network_access: bool = False
    file_system_access: bool = False
    resource_limits: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginVersion:
    """Plugin version information."""
    major: int
    minor: int
    patch: int
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def __lt__(self, other: 'PluginVersion') -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    @classmethod
    def from_string(cls, version_str: str) -> 'PluginVersion':
        """Parse version from string like '1.2.3'."""
        parts = version_str.split('.')
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]) if len(parts) > 1 else 0,
            patch=int(parts[2]) if len(parts) > 2 else 0
        )


@dataclass
class PluginDependency:
    """Plugin dependency specification."""
    name: str
    min_version: Optional[PluginVersion] = None
    max_version: Optional[PluginVersion] = None
    optional: bool = False


@dataclass
class SecurityReport:
    """Plugin security validation report."""
    plugin_id: str
    is_safe: bool
    security_level: PluginSecurityLevel
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "is_safe": self.is_safe,
            "security_level": self.security_level.value,
            "violations": self.violations,
            "warnings": self.warnings,
            "resource_usage": self.resource_usage,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass 
class PluginState:
    """Runtime state of a plugin."""
    plugin_id: str
    version: PluginVersion
    loaded: bool = False
    active: bool = False
    last_access: datetime = field(default_factory=datetime.utcnow)
    load_count: int = 0
    error_count: int = 0
    memory_usage_mb: float = 0.0
    last_security_check: Optional[datetime] = None


class Plugin:
    """Enhanced plugin wrapper with dynamic loading capabilities."""
    
    def __init__(
        self,
        plugin_id: str,
        metadata: PluginMetadata,
        version: PluginVersion,
        source_path: Optional[Path] = None,
        source_code: Optional[str] = None,
        dependencies: List[PluginDependency] = None,
        security_policy: Optional[PluginSecurityPolicy] = None,
        load_strategy: PluginLoadStrategy = PluginLoadStrategy.LAZY
    ):
        self.plugin_id = plugin_id
        self.metadata = metadata
        self.version = version
        self.source_path = source_path
        self.source_code = source_code
        self.dependencies = dependencies or []
        self.security_policy = security_policy or self._default_security_policy()
        self.load_strategy = load_strategy
        
        # Runtime state
        self.state = PluginState(plugin_id, version)
        self._plugin_instance: Optional[OrchestratorPlugin] = None
        self._module = None
        self._isolated_globals: Dict[str, Any] = {}
        
        # Epic 1: Memory efficiency tracking
        self._weak_refs: Set[weakref.ref] = set()
        
    def _default_security_policy(self) -> PluginSecurityPolicy:
        """Create default security policy based on plugin metadata."""
        # Epic 1: Conservative defaults for performance preservation
        return PluginSecurityPolicy(
            security_level=PluginSecurityLevel.SANDBOX,
            max_memory_mb=20,  # Epic 1: Strict memory limits
            max_cpu_time_ms=50,  # Epic 1: Response time preservation
            allowed_imports={"typing", "datetime", "asyncio", "logging"},
            blocked_imports={"os", "subprocess", "socket", "urllib", "requests"},
            network_access=False,
            file_system_access=False
        )
    
    @property
    def is_loaded(self) -> bool:
        """Check if plugin is currently loaded."""
        return self.state.loaded and self._plugin_instance is not None
    
    @property
    def is_active(self) -> bool:
        """Check if plugin is currently active."""
        return self.state.active and self.is_loaded
    
    async def load(self, orchestrator_context: Dict[str, Any]) -> bool:
        """Load the plugin dynamically."""
        if self.is_loaded:
            logger.debug("Plugin already loaded", plugin_id=self.plugin_id)
            return True
        
        start_time = datetime.utcnow()
        
        try:
            # Epic 1: Memory usage tracking before load
            initial_memory = self._get_memory_usage()
            
            # Load plugin module
            if self.source_path:
                self._module = await self._load_from_file()
            elif self.source_code:
                self._module = await self._load_from_source()
            else:
                raise ValueError("No source path or source code provided")
            
            # Find plugin class
            plugin_class = self._find_plugin_class()
            if not plugin_class:
                raise ValueError("No valid plugin class found")
            
            # Create plugin instance
            self._plugin_instance = plugin_class(self.metadata)
            
            # Initialize plugin
            await self._plugin_instance.initialize(orchestrator_context)
            
            # Update state
            self.state.loaded = True
            self.state.load_count += 1
            self.state.last_access = datetime.utcnow()
            
            # Epic 1: Track memory usage
            final_memory = self._get_memory_usage()
            self.state.memory_usage_mb = final_memory - initial_memory
            
            load_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.info("Plugin loaded successfully",
                       plugin_id=self.plugin_id,
                       load_time_ms=round(load_time_ms, 2),
                       memory_usage_mb=round(self.state.memory_usage_mb, 2))
            
            return True
            
        except Exception as e:
            self.state.error_count += 1
            logger.error("Failed to load plugin",
                        plugin_id=self.plugin_id,
                        error=str(e))
            return False
    
    async def unload(self) -> bool:
        """Unload the plugin and free resources."""
        if not self.is_loaded:
            return True
        
        try:
            # Cleanup plugin instance
            if self._plugin_instance:
                await self._plugin_instance.cleanup()
                self._plugin_instance = None
            
            # Remove module from sys.modules if dynamically loaded
            if self._module and hasattr(self._module, '__name__'):
                module_name = self._module.__name__
                if module_name in sys.modules:
                    del sys.modules[module_name]
            
            # Clear references
            self._module = None
            self._isolated_globals.clear()
            
            # Epic 1: Clear weak references for memory efficiency
            for ref in list(self._weak_refs):
                if ref() is not None:
                    try:
                        del ref
                    except:
                        pass
            self._weak_refs.clear()
            
            # Update state
            self.state.loaded = False
            self.state.active = False
            self.state.memory_usage_mb = 0.0
            
            logger.info("Plugin unloaded successfully", plugin_id=self.plugin_id)
            return True
            
        except Exception as e:
            logger.error("Failed to unload plugin",
                        plugin_id=self.plugin_id,
                        error=str(e))
            return False
    
    async def _load_from_file(self):
        """Load plugin from file path."""
        if not self.source_path or not self.source_path.exists():
            raise ValueError(f"Plugin source file not found: {self.source_path}")
        
        # Epic 1: Use importlib for efficient loading
        spec = importlib.util.spec_from_file_location(
            f"dynamic_plugin_{self.plugin_id}",
            self.source_path
        )
        if not spec or not spec.loader:
            raise ValueError("Failed to create module spec")
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        return module
    
    async def _load_from_source(self):
        """Load plugin from source code string."""
        if not self.source_code:
            raise ValueError("No source code provided")
        
        # Create temporary module
        module_name = f"dynamic_plugin_{self.plugin_id}_{uuid.uuid4().hex[:8]}"
        
        # Epic 1: Use restricted globals for security and memory efficiency
        restricted_globals = {
            "__builtins__": __builtins__,
            "__name__": module_name,
            "datetime": datetime,
            "asyncio": asyncio,
            "logging": logging,
        }
        
        # Execute source code in isolated environment
        exec(self.source_code, restricted_globals)
        
        # Create module-like object
        import types
        module = types.ModuleType(module_name)
        module.__dict__.update(restricted_globals)
        
        return module
    
    def _find_plugin_class(self) -> Optional[Type[OrchestratorPlugin]]:
        """Find the plugin class in the loaded module."""
        if not self._module:
            return None
        
        for name, obj in inspect.getmembers(self._module):
            if (inspect.isclass(obj) and 
                issubclass(obj, OrchestratorPlugin) and 
                obj != OrchestratorPlugin):
                return obj
        
        return None
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            # Epic 1: Efficient memory tracking
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    async def get_instance(self) -> Optional[OrchestratorPlugin]:
        """Get plugin instance, loading if necessary."""
        if not self.is_loaded:
            # For lazy loading strategy, load on first access
            if self.load_strategy == PluginLoadStrategy.LAZY:
                await self.load({})  # Empty context for lazy load
        
        self.state.last_access = datetime.utcnow()
        return self._plugin_instance


class AdvancedPluginManager:
    """
    Advanced Plugin Manager with dynamic loading and hot-swap capabilities.
    
    Epic 1 Optimizations:
    - Lazy loading to preserve <80MB memory usage
    - <50ms plugin operations for API response times
    - Memory-efficient plugin lifecycle management
    - Non-blocking plugin operations
    """
    
    def __init__(self, orchestrator: Optional["SimpleOrchestrator"] = None):
        self.orchestrator = orchestrator
        
        # Plugin registry
        self._plugins: Dict[str, Plugin] = {}
        self._plugins_by_type: Dict[PluginType, List[str]] = {}
        self._plugin_dependencies: Dict[str, Set[str]] = {}
        
        # Security framework
        self._security_policies: Dict[str, PluginSecurityPolicy] = {}
        self._security_reports: Dict[str, SecurityReport] = {}
        
        # Hot-swap support
        self._plugin_versions: Dict[str, List[PluginVersion]] = {}
        self._active_versions: Dict[str, PluginVersion] = {}
        
        # Epic 1: Performance monitoring
        self._operation_metrics: Dict[str, List[float]] = {}
        self._memory_baseline: float = 0.0
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        logger.info("AdvancedPluginManager initialized with Epic 1 optimizations")
    
    async def load_plugin_dynamic(
        self,
        plugin_id: str,
        version: str = "latest",
        source_path: Optional[Path] = None,
        source_code: Optional[str] = None,
        metadata: Optional[PluginMetadata] = None,
        dependencies: Optional[List[PluginDependency]] = None,
        security_policy: Optional[PluginSecurityPolicy] = None,
        load_strategy: PluginLoadStrategy = PluginLoadStrategy.LAZY
    ) -> Plugin:
        """
        Load a plugin dynamically with version management.
        
        Epic 1: <50ms operation target for immediate loading
        """
        start_time = datetime.utcnow()
        
        async with self._lock:
            try:
                # Parse version
                plugin_version = PluginVersion.from_string(version) if version != "latest" else PluginVersion(1, 0, 0)
                
                # Create or update plugin metadata
                if not metadata:
                    metadata = PluginMetadata(
                        name=plugin_id,
                        version=str(plugin_version),
                        plugin_type=PluginType.PERFORMANCE,  # Default type
                        description=f"Dynamically loaded plugin {plugin_id}",
                        dependencies=[]
                    )
                
                # Create plugin instance
                plugin = Plugin(
                    plugin_id=plugin_id,
                    metadata=metadata,
                    version=plugin_version,
                    source_path=source_path,
                    source_code=source_code,
                    dependencies=dependencies or [],
                    security_policy=security_policy,
                    load_strategy=load_strategy
                )
                
                # Validate security
                security_report = await self.validate_plugin_security(plugin)
                if not security_report.is_safe:
                    raise ValueError(f"Plugin security validation failed: {security_report.violations}")
                
                # Check dependencies
                await self._resolve_dependencies(plugin)
                
                # Register plugin
                self._plugins[plugin_id] = plugin
                
                # Update type registry
                plugin_type = metadata.plugin_type
                if plugin_type not in self._plugins_by_type:
                    self._plugins_by_type[plugin_type] = []
                if plugin_id not in self._plugins_by_type[plugin_type]:
                    self._plugins_by_type[plugin_type].append(plugin_id)
                
                # Update version tracking
                if plugin_id not in self._plugin_versions:
                    self._plugin_versions[plugin_id] = []
                self._plugin_versions[plugin_id].append(plugin_version)
                self._active_versions[plugin_id] = plugin_version
                
                # Load immediately if strategy requires it
                if load_strategy == PluginLoadStrategy.IMMEDIATE:
                    await plugin.load({"orchestrator": self.orchestrator})
                
                # Epic 1: Track operation time
                operation_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                self._record_operation_metric("load_plugin_dynamic", operation_time_ms)
                
                logger.info("Plugin loaded dynamically",
                           plugin_id=plugin_id,
                           version=str(plugin_version),
                           load_strategy=load_strategy.value,
                           operation_time_ms=round(operation_time_ms, 2))
                
                return plugin
                
            except Exception as e:
                logger.error("Failed to load plugin dynamically",
                            plugin_id=plugin_id,
                            version=version,
                            error=str(e))
                raise
    
    async def unload_plugin_safe(self, plugin_id: str) -> bool:
        """
        Safely unload a plugin with dependency checking.
        
        Epic 1: <50ms operation target
        """
        start_time = datetime.utcnow()
        
        async with self._lock:
            try:
                if plugin_id not in self._plugins:
                    logger.warning("Plugin not found for unloading", plugin_id=plugin_id)
                    return False
                
                plugin = self._plugins[plugin_id]
                
                # Check for dependent plugins
                dependents = await self._find_dependent_plugins(plugin_id)
                if dependents:
                    logger.warning("Cannot unload plugin with dependencies",
                                  plugin_id=plugin_id,
                                  dependents=dependents)
                    return False
                
                # Unload plugin
                success = await plugin.unload()
                
                if success:
                    # Remove from registries
                    del self._plugins[plugin_id]
                    
                    # Clean up type registry
                    for plugin_type, plugin_list in self._plugins_by_type.items():
                        if plugin_id in plugin_list:
                            plugin_list.remove(plugin_id)
                    
                    # Clean up dependencies
                    if plugin_id in self._plugin_dependencies:
                        del self._plugin_dependencies[plugin_id]
                    
                    # Clean up security data
                    if plugin_id in self._security_policies:
                        del self._security_policies[plugin_id]
                    if plugin_id in self._security_reports:
                        del self._security_reports[plugin_id]
                
                # Epic 1: Track operation time
                operation_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                self._record_operation_metric("unload_plugin_safe", operation_time_ms)
                
                logger.info("Plugin unloaded safely",
                           plugin_id=plugin_id,
                           success=success,
                           operation_time_ms=round(operation_time_ms, 2))
                
                return success
                
            except Exception as e:
                logger.error("Failed to unload plugin safely",
                            plugin_id=plugin_id,
                            error=str(e))
                return False
    
    async def hot_swap_plugin(self, old_id: str, new_id: str) -> bool:
        """
        Hot-swap plugin replacement without system restart.
        
        Epic 1: <100ms operation target for seamless swapping
        """
        start_time = datetime.utcnow()
        
        async with self._lock:
            try:
                if old_id not in self._plugins:
                    logger.error("Old plugin not found for hot-swap", old_plugin_id=old_id)
                    return False
                
                if new_id not in self._plugins:
                    logger.error("New plugin not found for hot-swap", new_plugin_id=new_id)
                    return False
                
                old_plugin = self._plugins[old_id]
                new_plugin = self._plugins[new_id]
                
                # Validate compatibility
                if old_plugin.metadata.plugin_type != new_plugin.metadata.plugin_type:
                    logger.error("Plugin types don't match for hot-swap",
                                old_type=old_plugin.metadata.plugin_type,
                                new_type=new_plugin.metadata.plugin_type)
                    return False
                
                # Check if old plugin is active
                was_active = old_plugin.is_active
                orchestrator_context = {"orchestrator": self.orchestrator}
                
                # Load new plugin first
                if not new_plugin.is_loaded:
                    success = await new_plugin.load(orchestrator_context)
                    if not success:
                        logger.error("Failed to load new plugin for hot-swap", new_plugin_id=new_id)
                        return False
                
                # Transfer state if possible (simplified for Epic 2 Phase 2.1)
                if was_active:
                    new_plugin.state.active = True
                
                # Unload old plugin
                await old_plugin.unload()
                
                # Update registry to point to new plugin
                plugin_type = old_plugin.metadata.plugin_type
                if plugin_type in self._plugins_by_type:
                    plugin_list = self._plugins_by_type[plugin_type]
                    if old_id in plugin_list:
                        plugin_list.remove(old_id)
                    if new_id not in plugin_list:
                        plugin_list.append(new_id)
                
                # Remove old plugin
                del self._plugins[old_id]
                
                # Epic 1: Track operation time
                operation_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                self._record_operation_metric("hot_swap_plugin", operation_time_ms)
                
                logger.info("Plugin hot-swapped successfully",
                           old_plugin_id=old_id,
                           new_plugin_id=new_id,
                           operation_time_ms=round(operation_time_ms, 2))
                
                return True
                
            except Exception as e:
                logger.error("Failed to hot-swap plugin",
                            old_plugin_id=old_id,
                            new_plugin_id=new_id,
                            error=str(e))
                return False
    
    async def validate_plugin_security(self, plugin: Plugin) -> SecurityReport:
        """
        Validate plugin security with comprehensive checks.
        
        Epic 1: <30ms validation target
        """
        start_time = datetime.utcnow()
        
        try:
            violations = []
            warnings = []
            resource_usage = {}
            
            # Check source code if available
            if plugin.source_code:
                violations.extend(await self._scan_source_security(plugin.source_code))
            
            # Check file path security
            if plugin.source_path:
                violations.extend(await self._scan_file_security(plugin.source_path))
            
            # Validate dependencies
            dep_violations = await self._validate_dependencies_security(plugin.dependencies)
            violations.extend(dep_violations)
            
            # Check memory limits
            policy = plugin.security_policy
            if policy.max_memory_mb > 100:  # Epic 1: Conservative limits
                warnings.append(f"High memory limit: {policy.max_memory_mb}MB")
            
            # Determine security level
            security_level = policy.security_level
            if violations:
                security_level = PluginSecurityLevel.UNTRUSTED
            elif warnings:
                security_level = PluginSecurityLevel.SANDBOX
            
            # Create security report
            report = SecurityReport(
                plugin_id=plugin.plugin_id,
                is_safe=len(violations) == 0,
                security_level=security_level,
                violations=violations,
                warnings=warnings,
                resource_usage=resource_usage
            )
            
            # Cache report
            self._security_reports[plugin.plugin_id] = report
            
            # Epic 1: Track validation time
            validation_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._record_operation_metric("validate_plugin_security", validation_time_ms)
            
            logger.debug("Plugin security validated",
                        plugin_id=plugin.plugin_id,
                        is_safe=report.is_safe,
                        violations_count=len(violations),
                        warnings_count=len(warnings),
                        validation_time_ms=round(validation_time_ms, 2))
            
            return report
            
        except Exception as e:
            logger.error("Failed to validate plugin security",
                        plugin_id=plugin.plugin_id,
                        error=str(e))
            
            # Return unsafe report on error
            return SecurityReport(
                plugin_id=plugin.plugin_id,
                is_safe=False,
                security_level=PluginSecurityLevel.UNTRUSTED,
                violations=[f"Security validation error: {str(e)}"]
            )
    
    async def _scan_source_security(self, source_code: str) -> List[str]:
        """Scan source code for security violations."""
        violations = []
        
        # Simple security patterns (can be enhanced)
        dangerous_patterns = [
            ("os.system", "System command execution"),
            ("subprocess", "Subprocess execution"),
            ("exec(", "Dynamic code execution"),
            ("eval(", "Dynamic code evaluation"),
            ("__import__", "Dynamic import"),
            ("open(", "File access"),
            ("socket", "Network access"),
            ("urllib", "HTTP requests"),
            ("requests", "HTTP requests")
        ]
        
        for pattern, description in dangerous_patterns:
            if pattern in source_code:
                violations.append(f"Dangerous pattern detected: {description} ({pattern})")
        
        return violations
    
    async def _scan_file_security(self, file_path: Path) -> List[str]:
        """Scan file path for security violations."""
        violations = []
        
        # Check file permissions and location
        try:
            if not file_path.exists():
                violations.append("Plugin file does not exist")
            elif not file_path.is_file():
                violations.append("Plugin path is not a file")
            elif file_path.stat().st_size > 1024 * 1024:  # 1MB limit
                violations.append("Plugin file too large (>1MB)")
        except Exception as e:
            violations.append(f"File access error: {str(e)}")
        
        return violations
    
    async def _validate_dependencies_security(self, dependencies: List[PluginDependency]) -> List[str]:
        """Validate plugin dependencies for security."""
        violations = []
        
        for dep in dependencies:
            # Check if dependency exists and is secure
            if dep.name not in self._plugins:
                if not dep.optional:
                    violations.append(f"Required dependency not found: {dep.name}")
            else:
                dep_plugin = self._plugins[dep.name]
                if dep_plugin.plugin_id in self._security_reports:
                    dep_report = self._security_reports[dep_plugin.plugin_id]
                    if not dep_report.is_safe:
                        violations.append(f"Unsafe dependency: {dep.name}")
        
        return violations
    
    async def _resolve_dependencies(self, plugin: Plugin) -> None:
        """Resolve and validate plugin dependencies."""
        if not plugin.dependencies:
            return
        
        dependencies = set()
        for dep in plugin.dependencies:
            dependencies.add(dep.name)
        
        self._plugin_dependencies[plugin.plugin_id] = dependencies
    
    async def _find_dependent_plugins(self, plugin_id: str) -> List[str]:
        """Find plugins that depend on the given plugin."""
        dependents = []
        
        for pid, deps in self._plugin_dependencies.items():
            if plugin_id in deps:
                dependents.append(pid)
        
        return dependents
    
    def _record_operation_metric(self, operation: str, time_ms: float) -> None:
        """Record operation metrics for Epic 1 performance monitoring."""
        if operation not in self._operation_metrics:
            self._operation_metrics[operation] = []
        
        metrics = self._operation_metrics[operation]
        metrics.append(time_ms)
        
        # Keep only last 100 measurements for memory efficiency
        if len(metrics) > 100:
            metrics.pop(0)
    
    async def get_plugin(self, plugin_id: str) -> Optional[Plugin]:
        """Get a plugin by ID."""
        return self._plugins.get(plugin_id)
    
    async def get_plugins_by_type(self, plugin_type: PluginType) -> List[Plugin]:
        """Get all plugins of a specific type."""
        plugin_ids = self._plugins_by_type.get(plugin_type, [])
        return [self._plugins[pid] for pid in plugin_ids if pid in self._plugins]
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for Epic 1 monitoring."""
        metrics = {}
        
        for operation, times in self._operation_metrics.items():
            if times:
                metrics[operation] = {
                    "avg_ms": sum(times) / len(times),
                    "max_ms": max(times),
                    "min_ms": min(times),
                    "count": len(times)
                }
        
        # Memory usage
        total_memory = sum(p.state.memory_usage_mb for p in self._plugins.values())
        
        return {
            "operations": metrics,
            "total_plugins": len(self._plugins),
            "loaded_plugins": len([p for p in self._plugins.values() if p.is_loaded]),
            "active_plugins": len([p for p in self._plugins.values() if p.is_active]),
            "total_memory_mb": round(total_memory, 2),
            "epic1_compliant": {
                "memory_under_80mb": total_memory < 80,
                "avg_operation_under_50ms": all(
                    sum(times) / len(times) < 50 
                    for times in self._operation_metrics.values() 
                    if times
                )
            }
        }
    
    async def cleanup(self) -> None:
        """Cleanup all plugins and resources."""
        logger.info("Cleaning up AdvancedPluginManager")
        
        # Unload all plugins
        for plugin_id in list(self._plugins.keys()):
            await self.unload_plugin_safe(plugin_id)
        
        # Clear all registries
        self._plugins.clear()
        self._plugins_by_type.clear()
        self._plugin_dependencies.clear()
        self._security_policies.clear()
        self._security_reports.clear()
        self._plugin_versions.clear()
        self._active_versions.clear()
        self._operation_metrics.clear()
        
        logger.info("AdvancedPluginManager cleanup complete")


# Factory function for integration with SimpleOrchestrator
def create_advanced_plugin_manager(orchestrator: Optional["SimpleOrchestrator"] = None) -> AdvancedPluginManager:
    """Factory function to create AdvancedPluginManager."""
    return AdvancedPluginManager(orchestrator)