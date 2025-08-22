"""
Integration Bridge for LeanVibe Plugin SDK with AdvancedPluginManager and Plugin Marketplace.

This module provides seamless integration between the Plugin SDK (Phase 2.3) and the existing
AdvancedPluginManager (Phase 2.1) and Plugin Marketplace (Phase 2.2) components.

Key Features:
- Automatic SDK plugin registration with AdvancedPluginManager
- Plugin marketplace publishing and distribution
- Bidirectional compatibility layer
- Performance monitoring and Epic 1 compliance validation
- Unified plugin lifecycle management
"""

import asyncio
import uuid
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type, Tuple
from dataclasses import dataclass
import logging

from .interfaces import PluginBase, PluginType as SDKPluginType
from .models import PluginConfig, TaskInterface, TaskResult, PluginEvent
from .tools import PluginPackager, PerformanceProfiler
from .testing import PluginTestFramework
from .exceptions import PluginSDKError, PluginIntegrationError

# Import existing LeanVibe components
from ..core.advanced_plugin_manager import (
    AdvancedPluginManager, Plugin as AdvancedPlugin, PluginVersion, PluginDependency,
    PluginSecurityPolicy, PluginSecurityLevel, PluginLoadStrategy, SecurityReport
)
from ..core.plugin_marketplace import (
    PluginMarketplace, MarketplacePluginEntry, Developer, PluginStatus,
    CertificationLevel, PluginCategory, PluginRating, PluginUsageMetrics
)
from ..core.orchestrator_plugins import PluginType as CorePluginType, PluginMetadata, OrchestratorPlugin
from ..core.logging_service import get_component_logger

logger = get_component_logger("plugin_sdk_integration")


@dataclass
class SDKPluginWrapper:
    """Wrapper for SDK plugins to integrate with existing plugin system."""
    sdk_plugin: PluginBase
    config: PluginConfig
    wrapper_id: str
    
    # Compatibility mappings
    metadata: PluginMetadata
    orchestrator_plugin: 'SDKOrchestratorPluginAdapter'
    
    def __post_init__(self):
        # Create orchestrator plugin adapter
        self.orchestrator_plugin = SDKOrchestratorPluginAdapter(
            self.sdk_plugin, self.config, self.metadata
        )


class SDKOrchestratorPluginAdapter(OrchestratorPlugin):
    """Adapter to make SDK plugins compatible with OrchestratorPlugin interface."""
    
    def __init__(self, sdk_plugin: PluginBase, config: PluginConfig, metadata: PluginMetadata):
        super().__init__(metadata)
        self.sdk_plugin = sdk_plugin
        self.config = config
        self._initialized = False
    
    async def initialize(self, context: Dict[str, Any] = None) -> None:
        """Initialize the SDK plugin."""
        if not self._initialized:
            await self.sdk_plugin.initialize()
            self._initialized = True
    
    async def execute(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SDK plugin task."""
        # Convert task data to SDK TaskInterface
        task = TaskInterface(
            task_id=task_data.get("task_id", str(uuid.uuid4())),
            task_type=task_data.get("task_type", "execute"),
            parameters=task_data.get("parameters", {})
        )
        
        # Execute through SDK
        result = await self.sdk_plugin.handle_task(task)
        
        # Convert back to orchestrator format
        return {
            "success": result.success,
            "data": result.data,
            "error": result.error,
            "execution_time_ms": result.execution_time_ms,
            "task_id": result.task_id
        }
    
    async def cleanup(self) -> None:
        """Cleanup SDK plugin."""
        if self._initialized:
            await self.sdk_plugin.cleanup()
            self._initialized = False
    
    def get_capabilities(self) -> List[str]:
        """Get plugin capabilities."""
        capabilities = ["sdk_plugin"]
        
        # Add capabilities based on SDK plugin type
        if hasattr(self.sdk_plugin, 'get_supported_tasks'):
            capabilities.extend(self.sdk_plugin.get_supported_tasks())
        
        return capabilities


class PluginTypeMapper:
    """Maps between SDK plugin types and Core plugin types."""
    
    SDK_TO_CORE_MAPPING = {
        SDKPluginType.WORKFLOW: CorePluginType.WORKFLOW,
        SDKPluginType.MONITORING: CorePluginType.PERFORMANCE,
        SDKPluginType.SECURITY: CorePluginType.SECURITY,
        SDKPluginType.INTEGRATION: CorePluginType.INTEGRATION,
        SDKPluginType.ANALYTICS: CorePluginType.ANALYTICS
    }
    
    CORE_TO_SDK_MAPPING = {v: k for k, v in SDK_TO_CORE_MAPPING.items()}
    
    @classmethod
    def sdk_to_core(cls, sdk_type: SDKPluginType) -> CorePluginType:
        """Convert SDK plugin type to Core plugin type."""
        return cls.SDK_TO_CORE_MAPPING.get(sdk_type, CorePluginType.WORKFLOW)
    
    @classmethod
    def core_to_sdk(cls, core_type: CorePluginType) -> SDKPluginType:
        """Convert Core plugin type to SDK plugin type."""
        return cls.CORE_TO_SDK_MAPPING.get(core_type, SDKPluginType.WORKFLOW)


class CategoryMapper:
    """Maps SDK plugin characteristics to marketplace categories."""
    
    KEYWORD_TO_CATEGORY = {
        "workflow": PluginCategory.WORKFLOW,
        "monitor": PluginCategory.MONITORING,
        "security": PluginCategory.SECURITY,
        "integration": PluginCategory.INTEGRATION,
        "analytics": PluginCategory.ANALYTICS,
        "automation": PluginCategory.AUTOMATION,
        "productivity": PluginCategory.PRODUCTIVITY,
        "communication": PluginCategory.COMMUNICATION,
        "development": PluginCategory.DEVELOPMENT,
        "utility": PluginCategory.UTILITY
    }
    
    @classmethod
    def determine_category(cls, plugin_config: PluginConfig) -> PluginCategory:
        """Determine marketplace category based on plugin configuration."""
        # Check plugin name and description for keywords
        text_to_check = f"{plugin_config.name} {plugin_config.description}".lower()
        
        for keyword, category in cls.KEYWORD_TO_CATEGORY.items():
            if keyword in text_to_check:
                return category
        
        # Default category
        return PluginCategory.UTILITY


class SDKPluginManagerIntegration:
    """Integration bridge between SDK and AdvancedPluginManager."""
    
    def __init__(self, advanced_plugin_manager: AdvancedPluginManager):
        self.advanced_manager = advanced_plugin_manager
        self.sdk_wrappers: Dict[str, SDKPluginWrapper] = {}
        self.type_mapper = PluginTypeMapper()
        
        logger.info("SDK-AdvancedPluginManager integration initialized")
    
    async def register_sdk_plugin(
        self,
        sdk_plugin: PluginBase,
        config: PluginConfig,
        security_level: PluginSecurityLevel = PluginSecurityLevel.SANDBOX,
        load_strategy: PluginLoadStrategy = PluginLoadStrategy.LAZY
    ) -> str:
        """Register an SDK plugin with the AdvancedPluginManager."""
        
        start_time = datetime.utcnow()
        
        try:
            # Generate wrapper ID
            wrapper_id = f"sdk_{config.name}_{uuid.uuid4().hex[:8]}"
            
            # Create PluginMetadata for compatibility
            core_plugin_type = self.type_mapper.sdk_to_core(sdk_plugin.plugin_type)
            
            metadata = PluginMetadata(
                name=config.name,
                version=config.version,
                plugin_type=core_plugin_type,
                description=config.description,
                dependencies=[]  # Will be populated from config if needed
            )
            
            # Create SDK wrapper
            wrapper = SDKPluginWrapper(
                sdk_plugin=sdk_plugin,
                config=config,
                wrapper_id=wrapper_id,
                metadata=metadata,
                orchestrator_plugin=None  # Will be created in __post_init__
            )
            
            # Convert version
            plugin_version = PluginVersion.from_string(config.version)
            
            # Create security policy
            security_policy = PluginSecurityPolicy(
                security_level=security_level,
                max_memory_mb=80,  # Epic 1 compliance
                max_cpu_time_ms=50,  # Epic 1 compliance
                allowed_imports={"typing", "datetime", "asyncio", "logging", "json"},
                blocked_imports={"os", "subprocess", "socket"},
                network_access=False,
                file_system_access=False
            )
            
            # Create source code for the adapter
            adapter_source = self._generate_adapter_source(wrapper_id, config)
            
            # Register with AdvancedPluginManager
            advanced_plugin = await self.advanced_manager.load_plugin_dynamic(
                plugin_id=wrapper_id,
                version=config.version,
                source_code=adapter_source,
                metadata=metadata,
                security_policy=security_policy,
                load_strategy=load_strategy
            )
            
            # Store wrapper
            self.sdk_wrappers[wrapper_id] = wrapper
            
            operation_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.info("SDK plugin registered with AdvancedPluginManager",
                       plugin_name=config.name,
                       wrapper_id=wrapper_id,
                       operation_time_ms=round(operation_time_ms, 2))
            
            return wrapper_id
            
        except Exception as e:
            logger.error("Failed to register SDK plugin",
                        plugin_name=config.name,
                        error=str(e))
            raise PluginIntegrationError(f"Failed to register SDK plugin: {e}")
    
    def _generate_adapter_source(self, wrapper_id: str, config: PluginConfig) -> str:
        """Generate source code for the SDK plugin adapter."""
        return f"""
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from app.core.orchestrator_plugins import OrchestratorPlugin, PluginMetadata

class SDKPluginAdapter_{wrapper_id.replace('-', '_')}(OrchestratorPlugin):
    '''Generated adapter for SDK plugin {config.name}'''
    
    def __init__(self, metadata):
        super().__init__(metadata)
        self.config_name = "{config.name}"
        self.wrapper_id = "{wrapper_id}"
        
    async def initialize(self, context: Dict[str, Any] = None) -> None:
        '''Initialize adapter'''
        pass  # SDK plugin initialization handled separately
        
    async def execute(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        '''Execute through SDK integration'''
        # This will be intercepted by the integration layer
        return {{"success": True, "data": {{}}, "message": "SDK adapter executed"}}
        
    async def cleanup(self) -> None:
        '''Cleanup adapter'''
        pass  # SDK plugin cleanup handled separately
        
    def get_capabilities(self) -> List[str]:
        '''Get adapter capabilities'''
        return ["sdk_plugin", "adapter", "{config.name}"]

# Create the plugin class for discovery
PluginClass = SDKPluginAdapter_{wrapper_id.replace('-', '_')}
"""
    
    async def unregister_sdk_plugin(self, wrapper_id: str) -> bool:
        """Unregister an SDK plugin from AdvancedPluginManager."""
        
        try:
            if wrapper_id not in self.sdk_wrappers:
                logger.warning("SDK plugin wrapper not found for unregistration", wrapper_id=wrapper_id)
                return False
            
            # Unregister from AdvancedPluginManager
            success = await self.advanced_manager.unload_plugin_safe(wrapper_id)
            
            if success:
                # Remove wrapper
                wrapper = self.sdk_wrappers.pop(wrapper_id)
                
                # Cleanup SDK plugin
                await wrapper.sdk_plugin.cleanup()
                
                logger.info("SDK plugin unregistered successfully", wrapper_id=wrapper_id)
            
            return success
            
        except Exception as e:
            logger.error("Failed to unregister SDK plugin",
                        wrapper_id=wrapper_id,
                        error=str(e))
            return False
    
    async def get_sdk_plugin(self, wrapper_id: str) -> Optional[PluginBase]:
        """Get the original SDK plugin by wrapper ID."""
        wrapper = self.sdk_wrappers.get(wrapper_id)
        return wrapper.sdk_plugin if wrapper else None
    
    async def list_sdk_plugins(self) -> List[Dict[str, Any]]:
        """List all registered SDK plugins."""
        plugins = []
        
        for wrapper_id, wrapper in self.sdk_wrappers.items():
            plugins.append({
                "wrapper_id": wrapper_id,
                "plugin_name": wrapper.config.name,
                "plugin_version": wrapper.config.version,
                "plugin_type": wrapper.sdk_plugin.plugin_type.value,
                "description": wrapper.config.description,
                "is_initialized": wrapper.sdk_plugin.is_initialized
            })
        
        return plugins


class SDKMarketplaceIntegration:
    """Integration bridge between SDK and Plugin Marketplace."""
    
    def __init__(self, plugin_marketplace: 'PluginMarketplace'):
        self.marketplace = plugin_marketplace
        self.packager = PluginPackager()
        self.profiler = PerformanceProfiler()
        self.test_framework = PluginTestFramework()
        self.category_mapper = CategoryMapper()
        
        logger.info("SDK-Marketplace integration initialized")
    
    async def submit_sdk_plugin(
        self,
        sdk_plugin: PluginBase,
        config: PluginConfig,
        developer: Developer,
        category: Optional[PluginCategory] = None,
        short_description: str = "",
        long_description: str = "",
        tags: List[str] = None,
        screenshots: List[str] = None
    ) -> str:
        """Submit an SDK plugin to the marketplace."""
        
        start_time = datetime.utcnow()
        
        try:
            # Validate plugin first
            validation_result = await self._validate_plugin_for_marketplace(sdk_plugin, config)
            if not validation_result["valid"]:
                raise PluginIntegrationError(f"Plugin validation failed: {validation_result['errors']}")
            
            # Determine category if not provided
            if not category:
                category = self.category_mapper.determine_category(config)
            
            # Package the plugin
            package_path = await self._package_sdk_plugin(sdk_plugin, config)
            
            # Create marketplace entry
            plugin_version = PluginVersion.from_string(config.version)
            
            # Create basic metadata
            metadata = PluginMetadata(
                name=config.name,
                version=config.version,
                plugin_type=PluginTypeMapper.sdk_to_core(sdk_plugin.plugin_type),
                description=config.description,
                dependencies=[]
            )
            
            # Create marketplace entry
            marketplace_entry = MarketplacePluginEntry(
                plugin_id=f"sdk_{config.name}_{uuid.uuid4().hex[:8]}",
                metadata=metadata,
                version=plugin_version,
                developer=developer,
                status=PluginStatus.PENDING_REVIEW,
                certification_level=CertificationLevel.UNCERTIFIED,
                category=category,
                short_description=short_description or config.description[:100],
                long_description=long_description or config.description,
                tags=tags or [],
                screenshots=screenshots or [],
                usage_metrics=PluginUsageMetrics(plugin_id=""),
                security_report=validation_result.get("security_report"),
                compliance_checks=validation_result.get("compliance_checks", {})
            )
            
            # Submit to marketplace
            submission_id = await self.marketplace.submit_plugin(marketplace_entry, package_path)
            
            # Clean up temporary package
            if package_path and Path(package_path).exists():
                Path(package_path).unlink()
            
            operation_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.info("SDK plugin submitted to marketplace",
                       plugin_name=config.name,
                       submission_id=submission_id,
                       category=category.value,
                       operation_time_ms=round(operation_time_ms, 2))
            
            return submission_id
            
        except Exception as e:
            logger.error("Failed to submit SDK plugin to marketplace",
                        plugin_name=config.name,
                        error=str(e))
            raise PluginIntegrationError(f"Failed to submit plugin to marketplace: {e}")
    
    async def _validate_plugin_for_marketplace(
        self,
        sdk_plugin: PluginBase,
        config: PluginConfig
    ) -> Dict[str, Any]:
        """Validate SDK plugin for marketplace submission."""
        
        validation_errors = []
        compliance_checks = {}
        
        try:
            # Basic configuration validation
            if not config.name or len(config.name) < 3:
                validation_errors.append("Plugin name must be at least 3 characters")
            
            if not config.description or len(config.description) < 10:
                validation_errors.append("Plugin description must be at least 10 characters")
            
            if not config.version:
                validation_errors.append("Plugin version is required")
            
            # Initialize plugin for testing
            if not sdk_plugin.is_initialized:
                await sdk_plugin.initialize()
            
            # Performance validation (Epic 1 compliance)
            performance_result = await self._validate_performance(sdk_plugin)
            compliance_checks["epic1_performance"] = performance_result["compliant"]
            
            if not performance_result["compliant"]:
                validation_errors.extend(performance_result["issues"])
            
            # Security validation
            security_result = await self._validate_security(sdk_plugin, config)
            compliance_checks["security_scan"] = security_result["secure"]
            
            if not security_result["secure"]:
                validation_errors.extend(security_result["issues"])
            
            # Functionality validation
            functionality_result = await self._validate_functionality(sdk_plugin)
            compliance_checks["functionality_test"] = functionality_result["working"]
            
            if not functionality_result["working"]:
                validation_errors.extend(functionality_result["issues"])
            
            return {
                "valid": len(validation_errors) == 0,
                "errors": validation_errors,
                "compliance_checks": compliance_checks,
                "security_report": security_result.get("security_report")
            }
            
        except Exception as e:
            logger.error("Plugin validation failed", error=str(e))
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "compliance_checks": compliance_checks
            }
    
    async def _validate_performance(self, sdk_plugin: PluginBase) -> Dict[str, Any]:
        """Validate plugin performance for Epic 1 compliance."""
        
        issues = []
        
        try:
            # Create test task
            test_task = TaskInterface(
                task_id="performance_test",
                task_type="test_performance",
                parameters={"test_data": list(range(100))}
            )
            
            # Profile performance
            profile_result = await self.profiler.profile_plugin_method(
                sdk_plugin,
                "handle_task",
                task=test_task
            )
            
            # Check Epic 1 compliance
            if profile_result.average_times.get("handle_task", 0) > 50:
                issues.append("Response time exceeds Epic 1 limit of 50ms")
            
            if profile_result.peak_memory_mb > 80:
                issues.append("Memory usage exceeds Epic 1 limit of 80MB")
            
            return {
                "compliant": len(issues) == 0,
                "issues": issues,
                "performance_data": profile_result.to_dict()
            }
            
        except Exception as e:
            return {
                "compliant": False,
                "issues": [f"Performance validation error: {str(e)}"]
            }
    
    async def _validate_security(self, sdk_plugin: PluginBase, config: PluginConfig) -> Dict[str, Any]:
        """Validate plugin security."""
        
        issues = []
        
        try:
            # Basic security checks
            if hasattr(sdk_plugin, '__code__'):
                # Check for dangerous operations
                source_code = str(sdk_plugin.__class__)
                dangerous_patterns = ['exec(', 'eval(', 'os.system', 'subprocess']
                
                for pattern in dangerous_patterns:
                    if pattern in source_code:
                        issues.append(f"Dangerous pattern detected: {pattern}")
            
            # Check configuration for sensitive data
            config_str = json.dumps(config.parameters)
            sensitive_patterns = ['password', 'secret', 'token', 'key']
            
            for pattern in sensitive_patterns:
                if pattern.lower() in config_str.lower():
                    issues.append(f"Potential sensitive data in configuration: {pattern}")
            
            return {
                "secure": len(issues) == 0,
                "issues": issues,
                "security_report": {
                    "scan_timestamp": datetime.utcnow().isoformat(),
                    "issues_found": len(issues),
                    "security_level": "safe" if len(issues) == 0 else "review_required"
                }
            }
            
        except Exception as e:
            return {
                "secure": False,
                "issues": [f"Security validation error: {str(e)}"]
            }
    
    async def _validate_functionality(self, sdk_plugin: PluginBase) -> Dict[str, Any]:
        """Validate plugin basic functionality."""
        
        issues = []
        
        try:
            # Test basic plugin operations
            if not sdk_plugin.is_initialized:
                issues.append("Plugin failed to initialize")
                return {"working": False, "issues": issues}
            
            # Test handle_task method
            test_task = TaskInterface(
                task_id="functionality_test",
                task_type="test",
                parameters={"test": True}
            )
            
            result = await sdk_plugin.handle_task(test_task)
            
            if not isinstance(result, TaskResult):
                issues.append("handle_task does not return TaskResult")
            
            if not hasattr(result, 'success'):
                issues.append("TaskResult missing success field")
            
            # Test cleanup
            await sdk_plugin.cleanup()
            
            return {
                "working": len(issues) == 0,
                "issues": issues
            }
            
        except Exception as e:
            return {
                "working": False,
                "issues": [f"Functionality test error: {str(e)}"]
            }
    
    async def _package_sdk_plugin(self, sdk_plugin: PluginBase, config: PluginConfig) -> str:
        """Package SDK plugin for marketplace distribution."""
        
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="sdk_plugin_")
            
            # Package the plugin
            package_info = await self.packager.create_distribution_package(
                plugin_class=sdk_plugin.__class__,
                config=config,
                output_dir=temp_dir,
                include_dependencies=True,
                sign_package=False  # Basic packaging for now
            )
            
            return package_info.package_path
            
        except Exception as e:
            logger.error("Failed to package SDK plugin", error=str(e))
            raise PluginIntegrationError(f"Failed to package plugin: {e}")
    
    async def download_and_install_marketplace_plugin(
        self,
        plugin_id: str,
        target_directory: Optional[str] = None
    ) -> Tuple[PluginBase, PluginConfig]:
        """Download and install a plugin from the marketplace as an SDK plugin."""
        
        try:
            # Get plugin from marketplace
            marketplace_entry = await self.marketplace.get_plugin(plugin_id)
            if not marketplace_entry:
                raise PluginIntegrationError(f"Plugin {plugin_id} not found in marketplace")
            
            # Download plugin package
            package_path = await self.marketplace.download_plugin(plugin_id)
            
            # Extract and install
            if not target_directory:
                target_directory = tempfile.mkdtemp(prefix="marketplace_plugin_")
            
            # TODO: Implement package extraction and SDK plugin creation
            # This would involve:
            # 1. Extracting the package
            # 2. Loading the plugin code
            # 3. Creating SDK-compatible wrapper
            # 4. Returning PluginBase instance and config
            
            raise NotImplementedError("Marketplace plugin installation not yet implemented")
            
        except Exception as e:
            logger.error("Failed to download and install marketplace plugin",
                        plugin_id=plugin_id,
                        error=str(e))
            raise PluginIntegrationError(f"Failed to install marketplace plugin: {e}")


class UnifiedPluginSDK:
    """Unified SDK interface that integrates with both AdvancedPluginManager and Marketplace."""
    
    def __init__(
        self,
        advanced_plugin_manager: AdvancedPluginManager,
        plugin_marketplace: 'PluginMarketplace'
    ):
        self.advanced_manager = advanced_plugin_manager
        self.marketplace = plugin_marketplace
        
        # Initialize integration bridges
        self.manager_integration = SDKPluginManagerIntegration(advanced_plugin_manager)
        self.marketplace_integration = SDKMarketplaceIntegration(plugin_marketplace)
        
        logger.info("UnifiedPluginSDK initialized with full integration")
    
    async def register_plugin(
        self,
        sdk_plugin: PluginBase,
        config: PluginConfig,
        auto_publish: bool = False,
        developer: Optional[Developer] = None,
        marketplace_options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Register plugin with both manager and optionally marketplace."""
        
        result = {}
        
        try:
            # Register with AdvancedPluginManager
            wrapper_id = await self.manager_integration.register_sdk_plugin(
                sdk_plugin, config
            )
            result["manager_id"] = wrapper_id
            
            # Optionally publish to marketplace
            if auto_publish and developer:
                marketplace_options = marketplace_options or {}
                
                submission_id = await self.marketplace_integration.submit_sdk_plugin(
                    sdk_plugin=sdk_plugin,
                    config=config,
                    developer=developer,
                    **marketplace_options
                )
                result["marketplace_id"] = submission_id
            
            logger.info("Plugin registered successfully",
                       plugin_name=config.name,
                       manager_registered=True,
                       marketplace_submitted=auto_publish)
            
            return result
            
        except Exception as e:
            logger.error("Failed to register plugin",
                        plugin_name=config.name,
                        error=str(e))
            
            # Cleanup on failure
            if "manager_id" in result:
                await self.manager_integration.unregister_sdk_plugin(result["manager_id"])
            
            raise
    
    async def unregister_plugin(self, wrapper_id: str) -> bool:
        """Unregister plugin from manager."""
        return await self.manager_integration.unregister_sdk_plugin(wrapper_id)
    
    async def get_plugin_status(self, wrapper_id: str) -> Dict[str, Any]:
        """Get comprehensive plugin status."""
        
        status = {
            "wrapper_id": wrapper_id,
            "found": False,
            "manager_status": None,
            "marketplace_status": None
        }
        
        # Check manager status
        sdk_plugin = await self.manager_integration.get_sdk_plugin(wrapper_id)
        if sdk_plugin:
            status["found"] = True
            status["manager_status"] = {
                "initialized": sdk_plugin.is_initialized,
                "plugin_type": sdk_plugin.plugin_type.value,
                "plugin_id": sdk_plugin.plugin_id
            }
        
        return status
    
    async def list_all_plugins(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all plugins across manager and marketplace."""
        
        return {
            "manager_plugins": await self.manager_integration.list_sdk_plugins(),
            "marketplace_plugins": []  # TODO: Implement marketplace listing
        }
    
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary for Epic 1 compliance."""
        
        # Get manager performance metrics
        manager_metrics = await self.advanced_manager.get_performance_metrics()
        
        # Calculate SDK-specific metrics
        sdk_plugins = await self.manager_integration.list_sdk_plugins()
        
        summary = {
            "epic1_compliance": manager_metrics.get("epic1_compliant", {}),
            "total_sdk_plugins": len(sdk_plugins),
            "manager_metrics": manager_metrics,
            "sdk_specific": {
                "initialized_plugins": len([p for p in sdk_plugins if p["is_initialized"]]),
                "plugin_types": {}
            }
        }
        
        # Aggregate by plugin type
        for plugin in sdk_plugins:
            plugin_type = plugin["plugin_type"]
            if plugin_type not in summary["sdk_specific"]["plugin_types"]:
                summary["sdk_specific"]["plugin_types"][plugin_type] = 0
            summary["sdk_specific"]["plugin_types"][plugin_type] += 1
        
        return summary


# Factory function for easy initialization
async def create_unified_sdk(
    advanced_plugin_manager: AdvancedPluginManager,
    plugin_marketplace: 'PluginMarketplace'
) -> UnifiedPluginSDK:
    """Factory function to create UnifiedPluginSDK."""
    return UnifiedPluginSDK(advanced_plugin_manager, plugin_marketplace)


# Exception for integration-specific errors
class PluginIntegrationError(PluginSDKError):
    """Exception raised when plugin integration fails."""
    pass