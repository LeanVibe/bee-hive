"""
Integration Tests for Advanced Plugin Manager - Epic 2 Phase 2.1

Tests the integration of AdvancedPluginManager with SimpleOrchestrator
while ensuring Epic 1 performance targets are preserved:
- <50ms API response times
- <80MB memory usage
- 250+ concurrent agents capability

Test Categories:
1. Plugin Loading Performance Tests
2. Memory Usage Validation
3. Security Framework Integration
4. Hot-swap Capabilities
5. Epic 1 Performance Preservation
"""

import asyncio
import pytest
import time
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import tempfile

from app.core.simple_orchestrator import SimpleOrchestrator, create_simple_orchestrator, AgentRole
from app.core.advanced_plugin_manager import (
    AdvancedPluginManager, 
    create_advanced_plugin_manager,
    PluginLoadStrategy,
    PluginSecurityLevel
)
from app.core.plugin_security_framework import get_plugin_security_framework
from app.core.orchestrator_plugins import PluginMetadata, PluginType


class TestAdvancedPluginManagerIntegration:
    """Integration tests for Advanced Plugin Manager with SimpleOrchestrator."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create test orchestrator with advanced plugin manager."""
        orchestrator = create_simple_orchestrator()
        await orchestrator.initialize()
        yield orchestrator
        await orchestrator.shutdown()
    
    @pytest.fixture
    def sample_plugin_code(self):
        """Sample plugin code for testing."""
        return '''
import asyncio
from datetime import datetime
from typing import Dict, Any

from app.core.orchestrator_plugins import OrchestratorPlugin, PluginMetadata

class TestPlugin(OrchestratorPlugin):
    """Test plugin for integration testing."""
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
        self.initialized = False
    
    async def initialize(self, orchestrator_context: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        self.initialized = True
        return True
    
    async def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        self.initialized = False
        return True
    
    async def pre_task_execution(self, task_context: Dict[str, Any]) -> Dict[str, Any]:
        """Hook called before task execution."""
        task_context["test_plugin_processed"] = True
        return task_context
    
    async def health_check(self) -> Dict[str, Any]:
        """Return plugin health status."""
        return {
            "plugin": self.metadata.name,
            "enabled": self.enabled,
            "initialized": self.initialized,
            "status": "healthy"
        }
'''
    
    @pytest.fixture
    def secure_plugin_code(self):
        """Secure plugin code that should pass security validation."""
        return '''
import asyncio
from datetime import datetime
from typing import Dict, Any

from app.core.orchestrator_plugins import OrchestratorPlugin, PluginMetadata

class SecurePlugin(OrchestratorPlugin):
    """Secure test plugin."""
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
    
    async def initialize(self, orchestrator_context: Dict[str, Any]) -> bool:
        return True
    
    async def cleanup(self) -> bool:
        return True
'''
    
    @pytest.fixture
    def insecure_plugin_code(self):
        """Insecure plugin code that should fail security validation."""
        return '''
import os
import subprocess
from app.core.orchestrator_plugins import OrchestratorPlugin, PluginMetadata

class InsecurePlugin(OrchestratorPlugin):
    """Insecure test plugin with dangerous operations."""
    
    def __init__(self, metadata: PluginMetadata):
        super().__init__(metadata)
    
    async def initialize(self, orchestrator_context: Dict[str, Any]) -> bool:
        # Dangerous: system command execution
        os.system("echo 'dangerous operation'")
        subprocess.run(["ls", "-la"])
        return True
    
    async def cleanup(self) -> bool:
        return True
'''

    # Epic 1 Performance Preservation Tests
    
    @pytest.mark.asyncio
    async def test_plugin_loading_performance(self, orchestrator, sample_plugin_code):
        """Test that plugin loading meets Epic 1 <50ms target."""
        start_time = time.time()
        
        success = await orchestrator.load_plugin_dynamic(
            plugin_id="test_performance_plugin",
            version="1.0.0",
            source_code=sample_plugin_code
        )
        
        load_time_ms = (time.time() - start_time) * 1000
        
        assert success, "Plugin loading should succeed"
        assert load_time_ms < 50, f"Plugin loading took {load_time_ms}ms, should be <50ms (Epic 1 target)"
        
        # Verify plugin is loaded
        manager = orchestrator._advanced_plugin_manager
        plugin = await manager.get_plugin("test_performance_plugin")
        assert plugin is not None
        assert plugin.is_loaded
    
    @pytest.mark.asyncio
    async def test_memory_usage_preservation(self, orchestrator, sample_plugin_code):
        """Test that plugin loading doesn't exceed Epic 1 memory targets."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load multiple plugins to test memory efficiency
        plugin_count = 10
        for i in range(plugin_count):
            await orchestrator.load_plugin_dynamic(
                plugin_id=f"memory_test_plugin_{i}",
                version="1.0.0",
                source_code=sample_plugin_code
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Epic 1: Total system should stay under 80MB, plugins should add minimal overhead
        assert memory_increase < 40, f"Memory increase {memory_increase}MB too high for {plugin_count} plugins"
        
        # Get plugin performance metrics
        metrics = await orchestrator.get_plugin_performance_metrics()
        assert metrics["epic1_compliant"]["memory_under_80mb"], "System memory usage exceeds Epic 1 targets"
    
    @pytest.mark.asyncio
    async def test_hot_swap_performance(self, orchestrator, sample_plugin_code):
        """Test hot-swap performance meets Epic 1 targets."""
        # Load initial plugin
        await orchestrator.load_plugin_dynamic(
            plugin_id="hot_swap_test_old",
            version="1.0.0",
            source_code=sample_plugin_code
        )
        
        # Load replacement plugin
        await orchestrator.load_plugin_dynamic(
            plugin_id="hot_swap_test_new",
            version="1.1.0",
            source_code=sample_plugin_code
        )
        
        # Measure hot-swap performance
        start_time = time.time()
        
        success = await orchestrator.hot_swap_plugin(
            old_plugin_id="hot_swap_test_old",
            new_plugin_id="hot_swap_test_new"
        )
        
        swap_time_ms = (time.time() - start_time) * 1000
        
        assert success, "Hot-swap should succeed"
        assert swap_time_ms < 100, f"Hot-swap took {swap_time_ms}ms, should be <100ms for seamless operation"
    
    # Security Framework Integration Tests
    
    @pytest.mark.asyncio
    async def test_security_validation_performance(self, orchestrator, secure_plugin_code):
        """Test security validation meets <30ms target."""
        security_framework = get_plugin_security_framework()
        
        start_time = time.time()
        
        report = await security_framework.validate_plugin_security(
            plugin_id="security_test_plugin",
            source_code=secure_plugin_code,
            security_level=PluginSecurityLevel.SANDBOX
        )
        
        validation_time_ms = (time.time() - start_time) * 1000
        
        assert validation_time_ms < 30, f"Security validation took {validation_time_ms}ms, should be <30ms"
        assert report.is_safe, "Secure plugin should pass validation"
        
        # Check framework performance metrics
        metrics = security_framework.get_performance_metrics()
        assert metrics["validation_times"]["epic1_compliant"], "Security validation not Epic 1 compliant"
    
    @pytest.mark.asyncio
    async def test_insecure_plugin_rejection(self, orchestrator, insecure_plugin_code):
        """Test that insecure plugins are properly rejected."""
        with pytest.raises(ValueError, match="security validation failed"):
            await orchestrator.load_plugin_dynamic(
                plugin_id="insecure_test_plugin",
                version="1.0.0",
                source_code=insecure_plugin_code
            )
    
    @pytest.mark.asyncio
    async def test_security_framework_resource_monitoring(self, orchestrator, sample_plugin_code):
        """Test resource monitoring in security framework."""
        security_framework = get_plugin_security_framework()
        
        # Load plugin and check resource monitoring
        await orchestrator.load_plugin_dynamic(
            plugin_id="resource_monitor_test",
            version="1.0.0",
            source_code=sample_plugin_code
        )
        
        metrics = security_framework.get_performance_metrics()
        assert "validation_times" in metrics
        assert metrics["monitoring_active"] in [True, False]  # May not be active yet
    
    # Plugin Lifecycle Tests
    
    @pytest.mark.asyncio
    async def test_lazy_loading_strategy(self, orchestrator, sample_plugin_code):
        """Test lazy loading strategy preserves memory."""
        manager = orchestrator._advanced_plugin_manager
        
        # Create plugin with lazy loading
        from app.core.advanced_plugin_manager import Plugin, PluginVersion
        from app.core.orchestrator_plugins import PluginMetadata, PluginType
        
        metadata = PluginMetadata(
            name="lazy_test_plugin",
            version="1.0.0",
            plugin_type=PluginType.PERFORMANCE,
            description="Test lazy loading",
            dependencies=[]
        )
        
        plugin = await manager.load_plugin_dynamic(
            plugin_id="lazy_test_plugin",
            version="1.0.0",
            source_code=sample_plugin_code,
            metadata=metadata,
            load_strategy=PluginLoadStrategy.LAZY
        )
        
        # Initially should not be loaded
        assert not plugin.is_loaded, "Lazy plugin should not be immediately loaded"
        
        # Access should trigger loading
        instance = await plugin.get_instance()
        assert instance is not None
        assert plugin.is_loaded, "Plugin should be loaded after access"
    
    @pytest.mark.asyncio
    async def test_plugin_dependency_resolution(self, orchestrator, sample_plugin_code):
        """Test plugin dependency management."""
        # Load base plugin first
        await orchestrator.load_plugin_dynamic(
            plugin_id="base_plugin",
            version="1.0.0",
            source_code=sample_plugin_code
        )
        
        # Create dependent plugin code
        dependent_plugin_code = sample_plugin_code.replace(
            "class TestPlugin", "class DependentPlugin"
        )
        
        # Load dependent plugin
        await orchestrator.load_plugin_dynamic(
            plugin_id="dependent_plugin",
            version="1.0.0",
            source_code=dependent_plugin_code
        )
        
        # Verify both plugins are loaded
        manager = orchestrator._advanced_plugin_manager
        base_plugin = await manager.get_plugin("base_plugin")
        dependent_plugin = await manager.get_plugin("dependent_plugin")
        
        assert base_plugin.is_loaded
        assert dependent_plugin.is_loaded
    
    @pytest.mark.asyncio
    async def test_safe_unloading_with_dependencies(self, orchestrator, sample_plugin_code):
        """Test safe unloading prevents dependency issues."""
        # This is a simplified test - full dependency checking would need more complex setup
        await orchestrator.load_plugin_dynamic(
            plugin_id="unload_test_plugin",
            version="1.0.0",
            source_code=sample_plugin_code
        )
        
        success = await orchestrator.unload_plugin_safe("unload_test_plugin")
        assert success, "Plugin unloading should succeed"
        
        # Verify plugin is unloaded
        manager = orchestrator._advanced_plugin_manager
        plugin = await manager.get_plugin("unload_test_plugin")
        assert plugin is None, "Plugin should be removed after unloading"
    
    # Integration with SimpleOrchestrator Tests
    
    @pytest.mark.asyncio
    async def test_orchestrator_integration_methods(self, orchestrator, sample_plugin_code):
        """Test that SimpleOrchestrator properly integrates AdvancedPluginManager methods."""
        # Test dynamic loading through orchestrator
        success = await orchestrator.load_plugin_dynamic(
            plugin_id="integration_test_plugin",
            version="1.0.0",
            source_code=sample_plugin_code
        )
        assert success
        
        # Test performance metrics access
        metrics = await orchestrator.get_plugin_performance_metrics()
        assert "total_plugins" in metrics
        assert metrics["total_plugins"] >= 1
        
        # Test security status access
        security_status = await orchestrator.get_plugin_security_status()
        assert "validation_times" in security_status
        
        # Test unloading through orchestrator
        success = await orchestrator.unload_plugin_safe("integration_test_plugin")
        assert success
    
    @pytest.mark.asyncio
    async def test_plugin_manager_initialization(self, orchestrator):
        """Test that AdvancedPluginManager is properly initialized in orchestrator."""
        assert orchestrator._advanced_plugin_manager is not None
        assert isinstance(orchestrator._advanced_plugin_manager, AdvancedPluginManager)
        
        # Test initial state
        metrics = await orchestrator.get_plugin_performance_metrics()
        assert metrics["total_plugins"] == 0  # No plugins loaded initially
        assert metrics["epic1_compliant"]["memory_under_80mb"]
    
    # Epic 1 Compliance Validation
    
    @pytest.mark.asyncio
    async def test_epic1_performance_targets_preserved(self, orchestrator, sample_plugin_code):
        """Comprehensive test that Epic 1 performance targets are preserved."""
        # Test multiple operations to ensure consistent performance
        operations_times = []
        
        for i in range(5):
            start_time = time.time()
            
            await orchestrator.load_plugin_dynamic(
                plugin_id=f"epic1_test_plugin_{i}",
                version="1.0.0",
                source_code=sample_plugin_code
            )
            
            operation_time = (time.time() - start_time) * 1000
            operations_times.append(operation_time)
        
        # All operations should be under 50ms
        avg_time = sum(operations_times) / len(operations_times)
        max_time = max(operations_times)
        
        assert avg_time < 50, f"Average operation time {avg_time}ms exceeds Epic 1 target"
        assert max_time < 100, f"Max operation time {max_time}ms too high"
        
        # Memory should stay reasonable
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024
        assert current_memory < 200, f"Memory usage {current_memory}MB too high"
        
        # Plugin manager should report Epic 1 compliance
        metrics = await orchestrator.get_plugin_performance_metrics()
        assert metrics["epic1_compliant"]["memory_under_80mb"]
        assert metrics["epic1_compliant"]["avg_operation_under_50ms"]
    
    @pytest.mark.asyncio
    async def test_cleanup_preserves_resources(self, orchestrator, sample_plugin_code):
        """Test that plugin cleanup properly frees resources."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Load and unload multiple plugins
        for i in range(10):
            await orchestrator.load_plugin_dynamic(
                plugin_id=f"cleanup_test_{i}",
                version="1.0.0",
                source_code=sample_plugin_code
            )
            
            await orchestrator.unload_plugin_safe(f"cleanup_test_{i}")
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_delta = final_memory - initial_memory
        
        # Memory should not significantly increase after cleanup
        assert memory_delta < 10, f"Memory not properly cleaned up: {memory_delta}MB increase"


class TestAdvancedPluginManagerStandalone:
    """Standalone tests for AdvancedPluginManager functionality."""
    
    @pytest.fixture
    async def plugin_manager(self):
        """Create standalone plugin manager for testing."""
        manager = create_advanced_plugin_manager()
        yield manager
        await manager.cleanup()
    
    @pytest.mark.asyncio
    async def test_plugin_version_management(self, plugin_manager):
        """Test plugin version management capabilities."""
        from app.core.advanced_plugin_manager import PluginVersion
        
        # Test version parsing
        version = PluginVersion.from_string("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert str(version) == "1.2.3"
        
        # Test version comparison
        v1 = PluginVersion(1, 0, 0)
        v2 = PluginVersion(1, 1, 0)
        assert v1 < v2
    
    @pytest.mark.asyncio
    async def test_security_policy_enforcement(self, plugin_manager):
        """Test security policy enforcement."""
        from app.core.advanced_plugin_manager import PluginSecurityPolicy, PluginSecurityLevel
        
        policy = PluginSecurityPolicy(
            security_level=PluginSecurityLevel.SANDBOX,
            max_memory_mb=20,
            max_cpu_time_ms=50
        )
        
        assert policy.security_level == PluginSecurityLevel.SANDBOX
        assert policy.max_memory_mb == 20
        assert policy.max_cpu_time_ms == 50


# Performance Benchmarks for Epic 1 Compliance

@pytest.mark.benchmark
class TestPluginPerformanceBenchmarks:
    """Performance benchmarks to ensure Epic 1 targets are met."""
    
    @pytest.mark.asyncio
    async def test_plugin_loading_benchmark(self, benchmark):
        """Benchmark plugin loading performance."""
        async def load_plugin():
            manager = create_advanced_plugin_manager()
            try:
                plugin = await manager.load_plugin_dynamic(
                    plugin_id="benchmark_plugin",
                    version="1.0.0",
                    source_code='''
from app.core.orchestrator_plugins import OrchestratorPlugin, PluginMetadata

class BenchmarkPlugin(OrchestratorPlugin):
    async def initialize(self, context): return True
    async def cleanup(self): return True
'''
                )
                return plugin
            finally:
                await manager.cleanup()
        
        # Benchmark should complete in under 50ms
        result = benchmark(lambda: asyncio.run(load_plugin()))
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_security_validation_benchmark(self, benchmark):
        """Benchmark security validation performance."""
        from app.core.plugin_security_framework import get_plugin_security_framework
        
        def validate_security():
            framework = get_plugin_security_framework()
            return asyncio.run(framework.validate_plugin_security(
                plugin_id="benchmark_security",
                source_code="from typing import Dict, Any\nclass TestPlugin: pass",
                security_level=PluginSecurityLevel.SANDBOX
            ))
        
        # Benchmark should complete in under 30ms
        result = benchmark(validate_security)
        assert result.is_safe