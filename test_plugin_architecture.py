#!/usr/bin/env python3
"""
Test script for Epic 2 Phase 2.1: Plugin Architecture Foundation

Tests the core functionality of the dynamic plugin system:
1. AdvancedPluginManager creation and basic operations
2. Plugin security framework validation
3. Dynamic plugin loading capabilities
4. Hot-swap functionality
5. Performance targets compliance

Epic 1 Requirements:
- <50ms API response times
- <80MB memory usage
- Plugin operations must be non-blocking
"""

import asyncio
import time
import psutil
import tempfile
from pathlib import Path
from typing import Dict, Any

# Test if the core imports work
try:
    import sys
    sys.path.append('.')
    
    from app.core.advanced_plugin_manager import (
        AdvancedPluginManager,
        create_advanced_plugin_manager,
        Plugin,
        PluginVersion,
        PluginSecurityLevel,
        PluginSecurityPolicy,
        PluginLoadStrategy
    )
    
    from app.core.plugin_security_framework import (
        PluginSecurityFramework,
        get_plugin_security_framework
    )
    
    from app.core.orchestrator_plugins import (
        PluginMetadata,
        PluginType,
        OrchestratorPlugin
    )
    
    print("✅ All core plugin architecture imports successful")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Cannot proceed with tests due to import failures")
    sys.exit(1)

class TestPlugin(OrchestratorPlugin):
    """Simple test plugin for dynamic loading tests."""
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self.initialized = False
        self.operations_count = 0
    
    async def initialize(self, orchestrator_context: Dict[str, Any]) -> bool:
        """Initialize the test plugin."""
        self.initialized = True
        return True
    
    async def cleanup(self) -> bool:
        """Cleanup the test plugin."""
        self.initialized = False
        return True
    
    async def test_operation(self) -> str:
        """Test operation for performance measurement."""
        self.operations_count += 1
        return f"Test operation {self.operations_count} completed"

class PluginArchitectureTests:
    """Comprehensive tests for the plugin architecture."""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all plugin architecture tests."""
        print("🚀 Starting Epic 2 Phase 2.1 Plugin Architecture Tests\n")
        
        # Test 1: Basic Plugin Manager Creation
        await self.test_plugin_manager_creation()
        
        # Test 2: Plugin Security Framework
        await self.test_security_framework()
        
        # Test 3: Dynamic Plugin Loading
        await self.test_dynamic_plugin_loading()
        
        # Test 4: Plugin Hot-Swap
        await self.test_plugin_hot_swap()
        
        # Test 5: Performance Validation
        await self.test_performance_compliance()
        
        # Summary
        self.print_test_summary()
        
        return {
            "test_results": self.test_results,
            "performance_metrics": self.performance_metrics
        }
    
    async def test_plugin_manager_creation(self):
        """Test 1: AdvancedPluginManager creation and basic operations."""
        print("Test 1: AdvancedPluginManager Creation")
        start_time = time.time()
        
        try:
            # Create plugin manager
            plugin_manager = create_advanced_plugin_manager()
            
            # Verify initialization
            assert plugin_manager is not None
            assert isinstance(plugin_manager, AdvancedPluginManager)
            assert len(plugin_manager._plugins) == 0
            
            # Test performance metrics
            metrics = await plugin_manager.get_performance_metrics()
            assert isinstance(metrics, dict)
            
            creation_time = (time.time() - start_time) * 1000
            self.test_results["plugin_manager_creation"] = {
                "status": "PASSED",
                "creation_time_ms": round(creation_time, 2)
            }
            
            print(f"✅ Plugin manager created successfully in {creation_time:.2f}ms")
            
        except Exception as e:
            self.test_results["plugin_manager_creation"] = {
                "status": "FAILED",
                "error": str(e)
            }
            print(f"❌ Plugin manager creation failed: {e}")
    
    async def test_security_framework(self):
        """Test 2: Plugin Security Framework validation."""
        print("\nTest 2: Plugin Security Framework")
        start_time = time.time()
        
        try:
            # Get security framework
            security_framework = get_plugin_security_framework()
            assert security_framework is not None
            
            # Test security validation with sample code
            test_code = '''
import asyncio
from datetime import datetime

async def safe_operation():
    return datetime.utcnow().isoformat()
'''
            
            validation_start = time.time()
            security_report = await security_framework.validate_plugin_security(
                plugin_id="test_plugin",
                source_code=test_code,
                security_level=PluginSecurityLevel.SANDBOX
            )
            validation_time = (time.time() - validation_start) * 1000
            
            assert security_report is not None
            assert security_report.plugin_id == "test_plugin"
            
            # Test security context
            async with security_framework.secure_execution_context("test_plugin") as context:
                assert context is not None
                assert context.plugin_id == "test_plugin"
            
            total_time = (time.time() - start_time) * 1000
            self.test_results["security_framework"] = {
                "status": "PASSED",
                "validation_time_ms": round(validation_time, 2),
                "total_time_ms": round(total_time, 2),
                "is_safe": security_report.is_safe
            }
            
            print(f"✅ Security framework operational in {total_time:.2f}ms")
            print(f"   Validation time: {validation_time:.2f}ms (target: <30ms)")
            
        except Exception as e:
            self.test_results["security_framework"] = {
                "status": "FAILED",
                "error": str(e)
            }
            print(f"❌ Security framework test failed: {e}")
    
    async def test_dynamic_plugin_loading(self):
        """Test 3: Dynamic plugin loading without system restart."""
        print("\nTest 3: Dynamic Plugin Loading")
        start_time = time.time()
        
        try:
            plugin_manager = create_advanced_plugin_manager()
            
            # Create test plugin source code
            test_plugin_code = '''
from app.core.orchestrator_plugins import OrchestratorPlugin, PluginMetadata
from typing import Dict, Any

class DynamicTestPlugin(OrchestratorPlugin):
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self.test_value = "dynamic_loaded"
    
    async def initialize(self, orchestrator_context: Dict[str, Any]) -> bool:
        return True
    
    async def cleanup(self) -> bool:
        return True
    
    def get_test_value(self) -> str:
        return self.test_value
'''
            
            # Test dynamic loading
            loading_start = time.time()
            plugin = await plugin_manager.load_plugin_dynamic(
                plugin_id="dynamic_test_plugin",
                version="1.0.0",
                source_code=test_plugin_code,
                metadata=PluginMetadata(
                    name="dynamic_test_plugin",
                    version="1.0.0",
                    plugin_type=PluginType.PERFORMANCE,
                    description="Test plugin for dynamic loading",
                    dependencies=[]
                ),
                load_strategy=PluginLoadStrategy.IMMEDIATE
            )
            loading_time = (time.time() - loading_start) * 1000
            
            assert plugin is not None
            assert plugin.plugin_id == "dynamic_test_plugin"
            assert plugin.is_loaded
            
            # Test plugin retrieval
            retrieved_plugin = await plugin_manager.get_plugin("dynamic_test_plugin")
            assert retrieved_plugin is not None
            assert retrieved_plugin.plugin_id == "dynamic_test_plugin"
            
            total_time = (time.time() - start_time) * 1000
            self.test_results["dynamic_loading"] = {
                "status": "PASSED",
                "loading_time_ms": round(loading_time, 2),
                "total_time_ms": round(total_time, 2)
            }
            
            print(f"✅ Dynamic plugin loading successful in {total_time:.2f}ms")
            print(f"   Loading time: {loading_time:.2f}ms (target: <50ms)")
            
        except Exception as e:
            self.test_results["dynamic_loading"] = {
                "status": "FAILED",
                "error": str(e)
            }
            print(f"❌ Dynamic plugin loading failed: {e}")
    
    async def test_plugin_hot_swap(self):
        """Test 4: Plugin hot-swap functionality."""
        print("\nTest 4: Plugin Hot-Swap")
        start_time = time.time()
        
        try:
            plugin_manager = create_advanced_plugin_manager()
            
            # Create first plugin version
            plugin_v1_code = '''
from app.core.orchestrator_plugins import OrchestratorPlugin, PluginMetadata
from typing import Dict, Any

class HotSwapTestPlugin(OrchestratorPlugin):
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self.version = "1.0"
    
    async def initialize(self, orchestrator_context: Dict[str, Any]) -> bool:
        return True
    
    async def cleanup(self) -> bool:
        return True
    
    def get_version(self) -> str:
        return self.version
'''
            
            # Create second plugin version
            plugin_v2_code = '''
from app.core.orchestrator_plugins import OrchestratorPlugin, PluginMetadata
from typing import Dict, Any

class HotSwapTestPlugin(OrchestratorPlugin):
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self.version = "2.0"
    
    async def initialize(self, orchestrator_context: Dict[str, Any]) -> bool:
        return True
    
    async def cleanup(self) -> bool:
        return True
    
    def get_version(self) -> str:
        return self.version
'''
            
            # Load first version
            await plugin_manager.load_plugin_dynamic(
                plugin_id="hotswap_test_v1",
                version="1.0.0",
                source_code=plugin_v1_code,
                metadata=PluginMetadata(
                    name="hotswap_test_v1",
                    version="1.0.0",
                    plugin_type=PluginType.PERFORMANCE,
                    description="Test plugin v1 for hot-swap",
                    dependencies=[]
                ),
                load_strategy=PluginLoadStrategy.IMMEDIATE
            )
            
            # Load second version
            await plugin_manager.load_plugin_dynamic(
                plugin_id="hotswap_test_v2",
                version="2.0.0",
                source_code=plugin_v2_code,
                metadata=PluginMetadata(
                    name="hotswap_test_v2",
                    version="2.0.0",
                    plugin_type=PluginType.PERFORMANCE,
                    description="Test plugin v2 for hot-swap",
                    dependencies=[]
                ),
                load_strategy=PluginLoadStrategy.IMMEDIATE
            )
            
            # Test hot-swap
            swap_start = time.time()
            success = await plugin_manager.hot_swap_plugin("hotswap_test_v1", "hotswap_test_v2")
            swap_time = (time.time() - swap_start) * 1000
            
            assert success
            
            # Verify old plugin is removed
            old_plugin = await plugin_manager.get_plugin("hotswap_test_v1")
            assert old_plugin is None
            
            # Verify new plugin is available
            new_plugin = await plugin_manager.get_plugin("hotswap_test_v2")
            assert new_plugin is not None
            
            total_time = (time.time() - start_time) * 1000
            self.test_results["hot_swap"] = {
                "status": "PASSED",
                "swap_time_ms": round(swap_time, 2),
                "total_time_ms": round(total_time, 2)
            }
            
            print(f"✅ Plugin hot-swap successful in {total_time:.2f}ms")
            print(f"   Swap time: {swap_time:.2f}ms (target: <100ms)")
            
        except Exception as e:
            self.test_results["hot_swap"] = {
                "status": "FAILED",
                "error": str(e)
            }
            print(f"❌ Plugin hot-swap failed: {e}")
    
    async def test_performance_compliance(self):
        """Test 5: Epic 1 performance targets compliance."""
        print("\nTest 5: Epic 1 Performance Compliance")
        start_time = time.time()
        
        try:
            # Memory usage check
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            plugin_manager = create_advanced_plugin_manager()
            security_framework = get_plugin_security_framework()
            
            # Perform multiple operations to test performance
            operations_start = time.time()
            
            for i in range(10):
                # Create and load plugin
                test_code = f'''
from app.core.orchestrator_plugins import OrchestratorPlugin, PluginMetadata
from typing import Dict, Any

class PerfTestPlugin{i}(OrchestratorPlugin):
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self.iteration = {i}
    
    async def initialize(self, orchestrator_context: Dict[str, Any]) -> bool:
        return True
    
    async def cleanup(self) -> bool:
        return True
'''
                
                plugin = await plugin_manager.load_plugin_dynamic(
                    plugin_id=f"perf_test_{i}",
                    version="1.0.0",
                    source_code=test_code,
                    metadata=PluginMetadata(
                        name=f"perf_test_{i}",
                        version="1.0.0",
                        plugin_type=PluginType.PERFORMANCE,
                        description=f"Performance test plugin {i}",
                        dependencies=[]
                    ),
                    load_strategy=PluginLoadStrategy.LAZY
                )
                
                # Security validation
                await security_framework.validate_plugin_security(
                    plugin_id=f"perf_test_{i}",
                    source_code=test_code,
                    security_level=PluginSecurityLevel.SANDBOX
                )
            
            operations_time = (time.time() - operations_start) * 1000
            avg_operation_time = operations_time / 10
            
            # Final memory check
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = final_memory - initial_memory
            
            # Performance metrics
            metrics = await plugin_manager.get_performance_metrics()
            security_metrics = security_framework.get_performance_metrics()
            
            # Epic 1 compliance checks
            epic1_compliant = {
                "memory_under_80mb": memory_usage < 80,
                "avg_operation_under_50ms": avg_operation_time < 50,
                "total_plugins": metrics.get("total_plugins", 0),
                "loaded_plugins": metrics.get("loaded_plugins", 0)
            }
            
            total_time = (time.time() - start_time) * 1000
            self.performance_metrics = {
                "memory_usage_mb": round(memory_usage, 2),
                "avg_operation_time_ms": round(avg_operation_time, 2),
                "total_operations_time_ms": round(operations_time, 2),
                "epic1_compliant": epic1_compliant,
                "plugin_metrics": metrics,
                "security_metrics": security_metrics
            }
            
            self.test_results["performance_compliance"] = {
                "status": "PASSED" if all(epic1_compliant.values()) else "WARNING",
                "total_time_ms": round(total_time, 2),
                "memory_usage_mb": round(memory_usage, 2),
                "avg_operation_time_ms": round(avg_operation_time, 2)
            }
            
            print(f"✅ Performance compliance test completed in {total_time:.2f}ms")
            print(f"   Memory usage: {memory_usage:.2f}MB (target: <80MB)")
            print(f"   Avg operation time: {avg_operation_time:.2f}ms (target: <50ms)")
            print(f"   Epic 1 compliant: {all(epic1_compliant.values())}")
            
        except Exception as e:
            self.test_results["performance_compliance"] = {
                "status": "FAILED",
                "error": str(e)
            }
            print(f"❌ Performance compliance test failed: {e}")
    
    def print_test_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "="*60)
        print("Epic 2 Phase 2.1 Plugin Architecture Test Summary")
        print("="*60)
        
        passed = sum(1 for result in self.test_results.values() if result["status"] == "PASSED")
        total = len(self.test_results)
        
        print(f"Tests Passed: {passed}/{total}")
        
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result["status"] == "PASSED" else "❌" if result["status"] == "FAILED" else "⚠️"
            print(f"{status_icon} {test_name}: {result['status']}")
            
            if "error" in result:
                print(f"    Error: {result['error']}")
        
        print("\nPerformance Summary:")
        if self.performance_metrics:
            print(f"  Memory Usage: {self.performance_metrics['memory_usage_mb']}MB (target: <80MB)")
            print(f"  Avg Operation: {self.performance_metrics['avg_operation_time_ms']}ms (target: <50ms)")
            print(f"  Epic 1 Compliant: {all(self.performance_metrics['epic1_compliant'].values())}")
        
        print("\n" + "="*60)

async def main():
    """Main test execution."""
    print("Epic 2 Phase 2.1: Plugin Architecture Foundation Test Suite")
    print("Testing dynamic plugin loading, hot-swap, and security framework\n")
    
    # Run comprehensive tests
    tester = PluginArchitectureTests()
    results = await tester.run_all_tests()
    
    # Determine overall success
    all_passed = all(result["status"] == "PASSED" for result in results["test_results"].values())
    
    print(f"\n🎯 Epic 2 Phase 2.1 Implementation Status: {'SUCCESS' if all_passed else 'NEEDS ATTENTION'}")
    
    if all_passed:
        print("✅ All core plugin architecture features are operational")
        print("✅ Dynamic plugin loading without system restart: WORKING")
        print("✅ Plugin security validation framework: OPERATIONAL")
        print("✅ Hot-swap capability: FUNCTIONAL")
        
        if results["performance_metrics"]:
            epic1_compliant = all(results["performance_metrics"]["epic1_compliant"].values())
            print(f"✅ Epic 1 performance targets: {'PRESERVED' if epic1_compliant else 'NEEDS OPTIMIZATION'}")
    else:
        print("⚠️  Some plugin architecture features need attention")
        print("   Check individual test results for specific issues")
    
    return results

if __name__ == "__main__":
    try:
        results = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()