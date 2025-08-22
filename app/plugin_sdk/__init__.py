"""
LeanVibe Plugin SDK - Epic 2 Phase 2.3: Developer SDK & Documentation

Comprehensive Plugin Development Kit for LeanVibe Agent Hive 2.0.
Enables third-party developers to create high-quality plugins effortlessly 
with full integration to AdvancedPluginManager and Plugin Marketplace.

Key Features:
- Intuitive plugin development interfaces and base classes
- Complete SDK with type hints and comprehensive documentation
- Plugin testing framework and validation utilities
- Real-time debugging and profiling tools
- Integration with AdvancedPluginManager (Phase 2.1)
- Plugin Marketplace integration (Phase 2.2)
- Example plugins demonstrating SDK capabilities
- Epic 1 performance compliance monitoring

Epic 1 Preservation:
- <50ms SDK operations for fast development feedback
- <80MB memory footprint for development tools
- Lazy loading of SDK components and examples
- Efficient plugin runtime performance monitoring
"""

from .interfaces import (
    PluginBase,
    WorkflowPlugin,
    MonitoringPlugin,
    SecurityPlugin,
    AgentInterface,
    TaskInterface,
    OrchestratorInterface,
    MonitoringInterface,
    PluginType
)

from .models import (
    TaskResult,
    CoordinationResult,
    PluginConfig,
    PluginEvent,
    PluginError,
    EventSeverity
)

from .testing import (
    PluginTestFramework,
    MockOrchestrator,
    MockAgent,
    MockTask,
    MockMonitoring
)

from .tools import (
    PluginGenerator,
    PluginPackager,
    PerformanceProfiler,
    DebugConsole
)

from .decorators import (
    plugin_method,
    performance_tracked,
    error_handled,
    cached_result,
    validate_inputs,
    retry_on_failure,
    timeout_after,
    rate_limited,
    requires_capability,
    log_execution,
    circuit_breaker
)

from .exceptions import (
    PluginSDKError,
    PluginConfigurationError,
    PluginExecutionError,
    PluginValidationError,
    PluginTimeoutError,
    PluginResourceError,
    PluginSecurityError,
    PluginCompatibilityError
)

# Integration components
from .integration import (
    UnifiedPluginSDK,
    SDKPluginManagerIntegration,
    SDKMarketplaceIntegration,
    PluginTypeMapper,
    CategoryMapper,
    PluginIntegrationError,
    create_unified_sdk
)

# Example plugins (lazy-loaded)
from .examples import (
    DataPipelinePlugin,
    SystemMonitorPlugin,
    SecurityScannerPlugin,
    WebhookIntegrationPlugin,
    EXAMPLE_CONFIGS
)

# Version information
__version__ = "2.3.0"
__author__ = "LeanVibe Development Team"
__description__ = "Complete Plugin Development Kit for LeanVibe Agent Hive 2.0"

# Epic 1 Performance Constants
EPIC1_MAX_RESPONSE_TIME_MS = 50
EPIC1_MAX_MEMORY_USAGE_MB = 80
EPIC1_MAX_INITIALIZATION_TIME_MS = 10

# Epic 1: Memory-efficient exports
__all__ = [
    # Core interfaces
    "PluginBase",
    "WorkflowPlugin",
    "MonitoringPlugin", 
    "SecurityPlugin",
    "AgentInterface", 
    "TaskInterface",
    "OrchestratorInterface",
    "MonitoringInterface",
    "PluginType",
    
    # Data models
    "TaskResult",
    "CoordinationResult", 
    "PluginConfig",
    "PluginEvent",
    "PluginError",
    "EventSeverity",
    
    # Testing framework
    "PluginTestFramework",
    "MockOrchestrator",
    "MockAgent",
    "MockTask",
    "MockMonitoring",
    
    # Development tools
    "PluginGenerator",
    "PluginPackager",
    "PerformanceProfiler",
    "DebugConsole",
    
    # Decorators
    "plugin_method",
    "performance_tracked",
    "error_handled",
    "cached_result",
    "validate_inputs",
    "retry_on_failure",
    "timeout_after",
    "rate_limited",
    "requires_capability",
    "log_execution",
    "circuit_breaker",
    
    # Exceptions
    "PluginSDKError",
    "PluginConfigurationError",
    "PluginExecutionError", 
    "PluginValidationError",
    "PluginTimeoutError",
    "PluginResourceError",
    "PluginSecurityError",
    "PluginCompatibilityError",
    
    # Integration components
    "UnifiedPluginSDK",
    "SDKPluginManagerIntegration",
    "SDKMarketplaceIntegration",
    "PluginTypeMapper",
    "CategoryMapper",
    "PluginIntegrationError",
    "create_unified_sdk",
    
    # Example plugins
    "DataPipelinePlugin",
    "SystemMonitorPlugin",
    "SecurityScannerPlugin",
    "WebhookIntegrationPlugin",
    "EXAMPLE_CONFIGS",
    
    # Constants
    "EPIC1_MAX_RESPONSE_TIME_MS",
    "EPIC1_MAX_MEMORY_USAGE_MB", 
    "EPIC1_MAX_INITIALIZATION_TIME_MS",
    
    # Version info
    "__version__",
    "__author__",
    "__description__"
]

# SDK initialization
def get_sdk_info():
    """Get SDK information and status."""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "epic1_optimized": True,
        "features": {
            "plugin_interfaces": True,
            "testing_framework": True,
            "development_tools": True,
            "integration_components": True,
            "performance_profiling": True,
            "real_time_debugging": True,
            "marketplace_integration": True,
            "advanced_plugin_manager": True
        }
    }

# Quick start function for developers
def create_plugin(plugin_type: PluginType, name: str, version: str = "1.0.0", **kwargs) -> PluginBase:
    """
    Quick start function to create a basic plugin.
    
    Args:
        plugin_type: Type of plugin to create
        name: Plugin name
        version: Plugin version
        **kwargs: Additional configuration parameters
    
    Returns:
        PluginBase: Configured plugin instance
    """
    config = PluginConfig(
        name=name,
        version=version,
        description=kwargs.get("description", f"Auto-generated {plugin_type.value} plugin"),
        parameters=kwargs.get("parameters", {})
    )
    
    plugin_classes = {
        PluginType.WORKFLOW: WorkflowPlugin,
        PluginType.MONITORING: MonitoringPlugin,
        PluginType.SECURITY: SecurityPlugin,
        PluginType.INTEGRATION: WorkflowPlugin,  # Use WorkflowPlugin as base
        PluginType.ANALYTICS: WorkflowPlugin     # Use WorkflowPlugin as base
    }
    
    plugin_class = plugin_classes.get(plugin_type, WorkflowPlugin)
    return plugin_class(config)

# Development environment setup
def setup_development_environment(workspace_dir: str = "./plugin_development") -> str:
    """
    Set up a complete plugin development environment.
    
    Args:
        workspace_dir: Directory to create workspace in
    
    Returns:
        str: Path to created workspace
    """
    from .tools import PluginGenerator
    
    generator = PluginGenerator()
    workspace_path = generator.create_development_workspace(
        workspace_name="plugin_workspace",
        output_dir=workspace_dir
    )
    
    return workspace_path

# Epic 1 compliance validation
def validate_epic1_compliance(plugin: PluginBase, test_iterations: int = 10) -> dict:
    """
    Validate that a plugin meets Epic 1 performance requirements.
    
    Args:
        plugin: Plugin to validate
        test_iterations: Number of test iterations to run
    
    Returns:
        dict: Compliance validation results
    """
    import asyncio
    import time
    
    async def _validate():
        results = []
        
        for i in range(test_iterations):
            test_task = TaskInterface(
                task_id=f"epic1_test_{i}",
                task_type="epic1_compliance_test",
                parameters={"test_data": list(range(100))}
            )
            
            start_time = time.perf_counter()
            result = await plugin.handle_task(test_task)
            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            
            results.append({
                "iteration": i + 1,
                "execution_time_ms": execution_time_ms,
                "success": result.success,
                "epic1_compliant": execution_time_ms < EPIC1_MAX_RESPONSE_TIME_MS
            })
        
        avg_time = sum(r["execution_time_ms"] for r in results) / len(results)
        max_time = max(r["execution_time_ms"] for r in results)
        compliance_rate = sum(1 for r in results if r["epic1_compliant"]) / len(results)
        
        return {
            "overall_compliant": avg_time < EPIC1_MAX_RESPONSE_TIME_MS and compliance_rate >= 0.95,
            "average_time_ms": avg_time,
            "max_time_ms": max_time,
            "compliance_rate": compliance_rate,
            "test_results": results
        }
    
    return asyncio.run(_validate())

# Unified integration function
def register_plugin_with_system(
    plugin: PluginBase,
    config: PluginConfig,
    advanced_manager=None,
    marketplace=None,
    auto_publish: bool = False,
    developer=None
) -> dict:
    """
    Register a plugin with the LeanVibe system (AdvancedPluginManager and optionally Marketplace).
    
    Args:
        plugin: Plugin instance to register
        config: Plugin configuration
        advanced_manager: AdvancedPluginManager instance
        marketplace: Plugin Marketplace instance  
        auto_publish: Whether to automatically publish to marketplace
        developer: Developer information for marketplace submission
    
    Returns:
        dict: Registration results with IDs and status
    """
    import asyncio
    
    async def _register():
        results = {"success": False, "manager_id": None, "marketplace_id": None}
        
        try:
            # Create unified SDK if both components provided
            if advanced_manager and marketplace:
                unified_sdk = await create_unified_sdk(advanced_manager, marketplace)
                
                registration_result = await unified_sdk.register_plugin(
                    plugin, config, auto_publish=auto_publish, developer=developer
                )
                
                results.update(registration_result)
                results["success"] = True
                
            # Register only with manager
            elif advanced_manager:
                manager_integration = SDKPluginManagerIntegration(advanced_manager)
                wrapper_id = await manager_integration.register_sdk_plugin(plugin, config)
                
                results["manager_id"] = wrapper_id
                results["success"] = True
            
            else:
                raise ValueError("At least AdvancedPluginManager must be provided")
            
            return results
            
        except Exception as e:
            results["error"] = str(e)
            return results
    
    return asyncio.run(_register())