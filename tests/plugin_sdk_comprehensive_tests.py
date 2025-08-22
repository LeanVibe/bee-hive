"""
Comprehensive Test Suite for LeanVibe Plugin SDK - Epic 2 Phase 2.3

Tests all SDK components and validates Epic 1 performance preservation.
Ensures the SDK meets all requirements and maintains system performance standards.
"""

import asyncio
import pytest
import time
import psutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

# Import SDK components
from app.plugin_sdk import (
    PluginBase, WorkflowPlugin, MonitoringPlugin, SecurityPlugin,
    PluginConfig, TaskInterface, TaskResult, PluginEvent, EventSeverity,
    PluginTestFramework, PluginGenerator, PluginPackager, PerformanceProfiler,
    DataPipelinePlugin, SystemMonitorPlugin, SecurityScannerPlugin, WebhookIntegrationPlugin,
    UnifiedPluginSDK, SDKPluginManagerIntegration, SDKMarketplaceIntegration,
    validate_epic1_compliance, EPIC1_MAX_RESPONSE_TIME_MS, EPIC1_MAX_MEMORY_USAGE_MB
)

# Import integration examples for testing
from app.plugin_sdk.integration_examples import ExampleIntegrationPlugin, IntegrationExamples

# Import existing LeanVibe components
from app.core.advanced_plugin_manager import AdvancedPluginManager, PluginSecurityLevel
from app.core.plugin_marketplace import Developer, PluginCategory


class TestSDKCoreComponents:
    """Test core SDK components and interfaces."""
    
    @pytest.fixture
    async def basic_plugin_config(self):
        """Create basic plugin configuration for testing."""
        return PluginConfig(
            name="TestPlugin",
            version="1.0.0",
            description="Test plugin for SDK validation",
            parameters={
                "test_mode": True,
                "batch_size": 100,
                "timeout_seconds": 30
            }
        )
    
    @pytest.fixture
    async def workflow_plugin(self, basic_plugin_config):
        """Create workflow plugin for testing."""
        plugin = WorkflowPlugin(basic_plugin_config)
        await plugin.initialize()
        return plugin
    
    async def test_plugin_base_interface(self, basic_plugin_config):
        """Test PluginBase interface implementation."""
        plugin = WorkflowPlugin(basic_plugin_config)
        
        # Test initialization
        assert not plugin.is_initialized
        await plugin.initialize()
        assert plugin.is_initialized
        
        # Test configuration access
        assert plugin.config == basic_plugin_config
        assert plugin.plugin_id is not None
        
        # Test cleanup
        await plugin.cleanup()
        assert not plugin.is_initialized
    
    async def test_task_interface_and_result(self, workflow_plugin):
        """Test TaskInterface and TaskResult functionality."""
        task = TaskInterface(
            task_id="test_task_001",
            task_type="test_operation",
            parameters={"test_data": [1, 2, 3, 4, 5]}
        )
        
        # Test task properties
        assert task.task_id == "test_task_001"
        assert task.task_type == "test_operation"
        assert "test_data" in task.parameters
        
        # Test task execution (should handle unknown task type gracefully)
        result = await workflow_plugin.handle_task(task)
        
        # Verify result structure
        assert isinstance(result, TaskResult)
        assert result.plugin_id == workflow_plugin.plugin_id
        assert result.task_id == task.task_id
        assert isinstance(result.success, bool)
        assert result.execution_time_ms is not None
    
    async def test_plugin_event_system(self, workflow_plugin):
        """Test plugin event emission and handling."""
        events_emitted = []
        
        # Mock event handler
        original_emit = workflow_plugin.emit_event
        async def mock_emit_event(event):
            events_emitted.append(event)
            return await original_emit(event)
        
        workflow_plugin.emit_event = mock_emit_event
        
        # Emit test event
        test_event = PluginEvent(
            event_type="test_event",
            plugin_id=workflow_plugin.plugin_id,
            data={"test": "data"},
            severity=EventSeverity.INFO
        )
        
        await workflow_plugin.emit_event(test_event)
        
        # Verify event was emitted
        assert len(events_emitted) == 1
        assert events_emitted[0].event_type == "test_event"
        assert events_emitted[0].severity == EventSeverity.INFO
    
    async def test_epic1_performance_compliance(self, workflow_plugin):
        """Test Epic 1 performance compliance."""
        # Test multiple iterations for statistical validity
        response_times = []
        
        for i in range(20):
            task = TaskInterface(
                task_id=f"perf_test_{i}",
                task_type="test_performance",
                parameters={"data": list(range(50))}
            )
            
            start_time = time.perf_counter()
            result = await workflow_plugin.handle_task(task)
            end_time = time.perf_counter()
            
            execution_time_ms = (end_time - start_time) * 1000
            response_times.append(execution_time_ms)
        
        # Verify Epic 1 compliance
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        assert avg_response_time < EPIC1_MAX_RESPONSE_TIME_MS, f"Average response time {avg_response_time:.2f}ms exceeds Epic 1 limit"
        assert max_response_time < EPIC1_MAX_RESPONSE_TIME_MS * 2, f"Max response time {max_response_time:.2f}ms significantly exceeds Epic 1 limit"
        
        # Verify high compliance rate
        compliant_responses = sum(1 for rt in response_times if rt < EPIC1_MAX_RESPONSE_TIME_MS)
        compliance_rate = compliant_responses / len(response_times)
        
        assert compliance_rate >= 0.8, f"Compliance rate {compliance_rate:.2%} is below minimum threshold"


class TestSDKExamplePlugins:
    """Test SDK example plugins for functionality and performance."""
    
    @pytest.fixture
    async def data_pipeline_plugin(self):
        """Create DataPipelinePlugin for testing."""
        config = PluginConfig(
            name="TestDataPipeline",
            version="1.0.0",
            description="Test data pipeline plugin",
            parameters={
                "batch_size": 100,
                "enable_validation": True,
                "pipeline_steps": [
                    {
                        "name": "filter_step",
                        "type": "filter",
                        "enabled": True,
                        "parameters": {
                            "field": "status",
                            "operation": "equals",
                            "value": "active"
                        }
                    },
                    {
                        "name": "transform_step",
                        "type": "transform",
                        "enabled": True,
                        "parameters": {
                            "transformations": [
                                {"field": "name", "operation": "uppercase"}
                            ]
                        }
                    }
                ]
            }
        )
        
        plugin = DataPipelinePlugin(config)
        await plugin.initialize()
        return plugin
    
    @pytest.fixture
    async def system_monitor_plugin(self):
        """Create SystemMonitorPlugin for testing."""
        config = PluginConfig(
            name="TestSystemMonitor",
            version="1.0.0",
            description="Test system monitor plugin",
            parameters={
                "collection_interval": 5,
                "retention_hours": 1,
                "enable_alerts": False,  # Disable alerts for testing
                "thresholds": []
            }
        )
        
        plugin = SystemMonitorPlugin(config)
        await plugin.initialize()
        return plugin
    
    async def test_data_pipeline_plugin_functionality(self, data_pipeline_plugin):
        """Test DataPipelinePlugin core functionality."""
        # Test data processing
        test_data = [
            {"id": 1, "name": "test item 1", "status": "active"},
            {"id": 2, "name": "test item 2", "status": "inactive"},
            {"id": 3, "name": "test item 3", "status": "active"},
        ]
        
        task = TaskInterface(
            task_id="pipeline_test",
            task_type="process_data",
            parameters={"input_data": test_data}
        )
        
        result = await data_pipeline_plugin.handle_task(task)
        
        assert result.success, f"Pipeline processing failed: {result.error}"
        assert "processing_summary" in result.data
        assert result.data["processing_summary"]["input_records"] == 3
        
        # Verify Epic 1 compliance
        assert result.execution_time_ms < EPIC1_MAX_RESPONSE_TIME_MS * 10  # Allow more time for complex processing
    
    async def test_system_monitor_plugin_functionality(self, system_monitor_plugin):
        """Test SystemMonitorPlugin core functionality."""
        # Test metrics collection
        task = TaskInterface(
            task_id="metrics_test",
            task_type="collect_metrics",
            parameters={}
        )
        
        result = await system_monitor_plugin.handle_task(task)
        
        assert result.success, f"Metrics collection failed: {result.error}"
        assert "metrics" in result.data
        
        metrics = result.data["metrics"]
        assert "cpu_percent" in metrics
        assert "memory_percent" in metrics
        assert "timestamp" in metrics
        
        # Verify Epic 1 compliance for metrics collection
        assert result.execution_time_ms < EPIC1_MAX_RESPONSE_TIME_MS * 5  # Allow more time for system metrics
    
    async def test_security_scanner_plugin_initialization(self):
        """Test SecurityScannerPlugin initialization and basic functionality."""
        config = PluginConfig(
            name="TestSecurityScanner",
            version="1.0.0",
            description="Test security scanner plugin",
            parameters={
                "dependency_check_enabled": False,  # Disable for testing
                "max_file_size_mb": 1,
                "excluded_paths": [".git", "__pycache__"]
            }
        )
        
        plugin = SecurityScannerPlugin(config)
        await plugin.initialize()
        
        assert plugin.is_initialized
        assert plugin.config.name == "TestSecurityScanner"
        
        await plugin.cleanup()
    
    async def test_webhook_integration_plugin_initialization(self):
        """Test WebhookIntegrationPlugin initialization."""
        config = PluginConfig(
            name="TestWebhookIntegration",
            version="1.0.0",
            description="Test webhook integration plugin",
            parameters={
                "max_concurrent_deliveries": 5,
                "delivery_timeout_seconds": 10,
                "webhooks": [],  # No webhooks for testing
                "api_endpoints": []  # No endpoints for testing
            }
        )
        
        plugin = WebhookIntegrationPlugin(config)
        await plugin.initialize()
        
        assert plugin.is_initialized
        assert plugin.config.name == "TestWebhookIntegration"
        
        await plugin.cleanup()


class TestSDKTestingFramework:
    """Test the SDK testing framework functionality."""
    
    async def test_plugin_test_framework_initialization(self):
        """Test PluginTestFramework initialization."""
        framework = PluginTestFramework()
        
        assert framework is not None
        assert hasattr(framework, 'register_plugin')
        assert hasattr(framework, 'test_plugin_task')
        
        await framework.cleanup()
    
    async def test_mock_components(self):
        """Test mock components functionality."""
        # Test MockOrchestrator
        mock_orchestrator = framework.MockOrchestrator()
        assert mock_orchestrator is not None
        
        # Test MockAgent
        mock_agent = framework.MockAgent("test_agent")
        assert mock_agent.agent_id == "test_agent"
        
        # Test MockTask
        mock_task = framework.MockTask("test_task")
        assert mock_task.task_id == "test_task"
        
        # Test MockMonitoring
        mock_monitoring = framework.MockMonitoring()
        assert mock_monitoring is not None


class TestSDKDevelopmentTools:
    """Test SDK development tools functionality."""
    
    async def test_plugin_generator(self):
        """Test PluginGenerator functionality."""
        generator = PluginGenerator()
        
        # Test template creation
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = generator.create_plugin_project(
                plugin_name="TestGeneratedPlugin",
                plugin_type="workflow",
                output_dir=temp_dir,
                author_name="Test Author",
                author_email="test@example.com"
            )
            
            assert Path(project_path).exists()
            assert Path(project_path, "plugin.py").exists()
            assert Path(project_path, "config.json").exists()
    
    async def test_plugin_packager(self):
        """Test PluginPackager functionality."""
        packager = PluginPackager()
        
        # Create simple test plugin
        config = PluginConfig(
            name="TestPackagePlugin",
            version="1.0.0",
            description="Test plugin for packaging"
        )
        
        # Test package creation (basic validation)
        assert packager is not None
        assert hasattr(packager, 'create_distribution_package')
    
    async def test_performance_profiler(self):
        """Test PerformanceProfiler functionality."""
        profiler = PerformanceProfiler()
        
        # Create test plugin for profiling
        config = PluginConfig(
            name="TestProfilePlugin",
            version="1.0.0",
            description="Test plugin for profiling"
        )
        
        plugin = WorkflowPlugin(config)
        await plugin.initialize()
        
        # Test basic profiling (simplified)
        assert profiler is not None
        assert hasattr(profiler, 'profile_plugin_method')
        
        await plugin.cleanup()


class TestSDKIntegration:
    """Test SDK integration with AdvancedPluginManager and Plugin Marketplace."""
    
    @pytest.fixture
    async def advanced_manager(self):
        """Create AdvancedPluginManager for testing."""
        return AdvancedPluginManager()
    
    @pytest.fixture
    async def example_integration_plugin(self):
        """Create ExampleIntegrationPlugin for testing."""
        config = PluginConfig(
            name="IntegrationTestPlugin",
            version="1.0.0",
            description="Integration test plugin",
            parameters={
                "demo_mode": True,
                "integration_test": True
            }
        )
        
        plugin = ExampleIntegrationPlugin(config)
        await plugin.initialize()
        return plugin
    
    async def test_sdk_plugin_manager_integration(self, advanced_manager, example_integration_plugin):
        """Test SDK integration with AdvancedPluginManager."""
        # Create integration bridge
        manager_integration = SDKPluginManagerIntegration(advanced_manager)
        
        # Register SDK plugin
        wrapper_id = await manager_integration.register_sdk_plugin(
            example_integration_plugin,
            example_integration_plugin.config,
            security_level=PluginSecurityLevel.TRUSTED
        )
        
        assert wrapper_id is not None
        assert wrapper_id.startswith("sdk_")
        
        # Test plugin retrieval
        sdk_plugin = await manager_integration.get_sdk_plugin(wrapper_id)
        assert sdk_plugin is not None
        assert sdk_plugin == example_integration_plugin
        
        # Test plugin listing
        plugin_list = await manager_integration.list_sdk_plugins()
        assert len(plugin_list) >= 1
        
        # Cleanup
        unregister_success = await manager_integration.unregister_sdk_plugin(wrapper_id)
        assert unregister_success
    
    async def test_plugin_type_mapper(self):
        """Test PluginTypeMapper functionality."""
        from app.plugin_sdk.integration import PluginTypeMapper
        from app.plugin_sdk.interfaces import PluginType as SDKPluginType
        from app.core.orchestrator_plugins import PluginType as CorePluginType
        
        # Test SDK to Core mapping
        core_type = PluginTypeMapper.sdk_to_core(SDKPluginType.WORKFLOW)
        assert core_type == CorePluginType.WORKFLOW
        
        # Test Core to SDK mapping
        sdk_type = PluginTypeMapper.core_to_sdk(CorePluginType.SECURITY)
        assert sdk_type == SDKPluginType.SECURITY
    
    async def test_category_mapper(self):
        """Test CategoryMapper functionality."""
        from app.plugin_sdk.integration import CategoryMapper
        
        # Test category determination
        config = PluginConfig(
            name="SecurityTestPlugin",
            version="1.0.0",
            description="A security plugin for testing security detection"
        )
        
        category = CategoryMapper.determine_category(config)
        assert category == PluginCategory.SECURITY
        
        # Test workflow detection
        workflow_config = PluginConfig(
            name="WorkflowTestPlugin",
            version="1.0.0",
            description="A workflow automation plugin for testing"
        )
        
        workflow_category = CategoryMapper.determine_category(workflow_config)
        assert workflow_category in [PluginCategory.WORKFLOW, PluginCategory.AUTOMATION]


class TestEpic1PerformanceValidation:
    """Comprehensive Epic 1 performance validation tests."""
    
    async def test_sdk_operation_performance(self):
        """Test that all SDK operations meet Epic 1 requirements."""
        # Test plugin creation performance
        start_time = time.perf_counter()
        
        config = PluginConfig(
            name="PerformanceTestPlugin",
            version="1.0.0",
            description="Plugin for performance testing"
        )
        
        plugin = WorkflowPlugin(config)
        creation_time = (time.perf_counter() - start_time) * 1000
        
        assert creation_time < EPIC1_MAX_RESPONSE_TIME_MS, f"Plugin creation took {creation_time:.2f}ms"
        
        # Test plugin initialization performance
        start_time = time.perf_counter()
        await plugin.initialize()
        init_time = (time.perf_counter() - start_time) * 1000
        
        assert init_time < EPIC1_MAX_RESPONSE_TIME_MS, f"Plugin initialization took {init_time:.2f}ms"
        
        # Test cleanup performance
        start_time = time.perf_counter()
        await plugin.cleanup()
        cleanup_time = (time.perf_counter() - start_time) * 1000
        
        assert cleanup_time < EPIC1_MAX_RESPONSE_TIME_MS, f"Plugin cleanup took {cleanup_time:.2f}ms"
    
    async def test_memory_usage_compliance(self):
        """Test that SDK components maintain memory usage within Epic 1 limits."""
        import gc
        
        # Get baseline memory usage
        gc.collect()
        process = psutil.Process()
        baseline_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Create multiple plugins
        plugins = []
        for i in range(10):
            config = PluginConfig(
                name=f"MemoryTestPlugin_{i}",
                version="1.0.0",
                description=f"Memory test plugin {i}"
            )
            
            plugin = WorkflowPlugin(config)
            await plugin.initialize()
            plugins.append(plugin)
        
        # Measure memory usage after plugin creation
        gc.collect()
        current_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_increase = current_memory_mb - baseline_memory_mb
        
        assert memory_increase < EPIC1_MAX_MEMORY_USAGE_MB, f"Memory increase {memory_increase:.2f}MB exceeds Epic 1 limit"
        
        # Cleanup plugins
        for plugin in plugins:
            await plugin.cleanup()
        
        # Verify memory cleanup
        gc.collect()
        final_memory_mb = process.memory_info().rss / 1024 / 1024
        memory_cleanup = current_memory_mb - final_memory_mb
        
        # Should free at least 50% of allocated memory
        assert memory_cleanup > memory_increase * 0.5, f"Insufficient memory cleanup: {memory_cleanup:.2f}MB"
    
    async def test_concurrent_operation_performance(self):
        """Test Epic 1 compliance under concurrent operations."""
        config = PluginConfig(
            name="ConcurrentTestPlugin",
            version="1.0.0",
            description="Plugin for concurrent testing"
        )
        
        plugin = WorkflowPlugin(config)
        await plugin.initialize()
        
        # Create multiple concurrent tasks
        tasks = []
        for i in range(20):
            task = TaskInterface(
                task_id=f"concurrent_test_{i}",
                task_type="test_operation",
                parameters={"data": list(range(10))}
            )
            tasks.append(plugin.handle_task(task))
        
        # Execute all tasks concurrently
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Verify all tasks completed
        assert all(isinstance(r, TaskResult) for r in results)
        
        # Verify overall performance
        avg_time_per_task = total_time / len(tasks)
        assert avg_time_per_task < EPIC1_MAX_RESPONSE_TIME_MS * 2, f"Average concurrent task time {avg_time_per_task:.2f}ms exceeds acceptable limit"
        
        await plugin.cleanup()
    
    async def test_epic1_compliance_validation_function(self):
        """Test the Epic 1 compliance validation function itself."""
        config = PluginConfig(
            name="ComplianceTestPlugin",
            version="1.0.0",
            description="Plugin for compliance testing"
        )
        
        plugin = WorkflowPlugin(config)
        await plugin.initialize()
        
        # Run compliance validation
        compliance_result = validate_epic1_compliance(plugin, test_iterations=5)
        
        assert compliance_result is not None
        assert "overall_compliant" in compliance_result
        assert "average_time_ms" in compliance_result
        assert "compliance_rate" in compliance_result
        
        # Results should indicate compliance for basic workflow plugin
        assert compliance_result["overall_compliant"], f"Plugin failed Epic 1 compliance: {compliance_result}"
        
        await plugin.cleanup()


class TestIntegrationExamples:
    """Test integration examples and end-to-end workflows."""
    
    async def test_integration_examples_execution(self):
        """Test that integration examples execute successfully."""
        examples = IntegrationExamples()
        
        # Test basic integration example
        basic_result = await examples.basic_integration_example()
        assert basic_result, "Basic integration example failed"
        
        # Test marketplace integration example
        marketplace_result = await examples.marketplace_integration_example()
        assert marketplace_result, "Marketplace integration example failed"
        
        # Test unified SDK example
        unified_result = await examples.unified_sdk_example()
        assert unified_result, "Unified SDK example failed"


# Test execution function
async def run_comprehensive_tests():
    """Run all comprehensive tests."""
    print("🧪 Running Comprehensive LeanVibe Plugin SDK Tests")
    print("=" * 60)
    
    test_results = []
    
    # Core Components Tests
    print("\n📦 Testing Core SDK Components...")
    core_tests = TestSDKCoreComponents()
    
    try:
        # Run core component tests
        config = PluginConfig(name="TestPlugin", version="1.0.0", description="Test")
        
        await core_tests.test_plugin_base_interface(config)
        print("   ✅ Plugin base interface test passed")
        
        plugin = WorkflowPlugin(config)
        await plugin.initialize()
        
        await core_tests.test_task_interface_and_result(plugin)
        print("   ✅ Task interface and result test passed")
        
        await core_tests.test_plugin_event_system(plugin)
        print("   ✅ Plugin event system test passed")
        
        await core_tests.test_epic1_performance_compliance(plugin)
        print("   ✅ Epic 1 performance compliance test passed")
        
        await plugin.cleanup()
        test_results.append(("Core Components", True))
        
    except Exception as e:
        print(f"   ❌ Core components test failed: {e}")
        test_results.append(("Core Components", False))
    
    # Example Plugins Tests
    print("\n🔌 Testing Example Plugins...")
    example_tests = TestSDKExamplePlugins()
    
    try:
        # Test DataPipelinePlugin
        pipeline_config = PluginConfig(
            name="TestPipeline",
            version="1.0.0",
            description="Test pipeline",
            parameters={
                "batch_size": 50,
                "enable_validation": True,
                "pipeline_steps": []
            }
        )
        
        pipeline_plugin = DataPipelinePlugin(pipeline_config)
        await pipeline_plugin.initialize()
        await pipeline_plugin.cleanup()
        print("   ✅ DataPipelinePlugin initialization test passed")
        
        # Test SystemMonitorPlugin  
        monitor_config = PluginConfig(
            name="TestMonitor",
            version="1.0.0",
            description="Test monitor",
            parameters={
                "collection_interval": 10,
                "enable_alerts": False
            }
        )
        
        monitor_plugin = SystemMonitorPlugin(monitor_config)
        await monitor_plugin.initialize()
        await monitor_plugin.cleanup()
        print("   ✅ SystemMonitorPlugin initialization test passed")
        
        test_results.append(("Example Plugins", True))
        
    except Exception as e:
        print(f"   ❌ Example plugins test failed: {e}")
        test_results.append(("Example Plugins", False))
    
    # Performance Validation Tests
    print("\n⚡ Testing Epic 1 Performance Compliance...")
    perf_tests = TestEpic1PerformanceValidation()
    
    try:
        await perf_tests.test_sdk_operation_performance()
        print("   ✅ SDK operation performance test passed")
        
        await perf_tests.test_memory_usage_compliance()
        print("   ✅ Memory usage compliance test passed")
        
        await perf_tests.test_concurrent_operation_performance()
        print("   ✅ Concurrent operation performance test passed")
        
        await perf_tests.test_epic1_compliance_validation_function()
        print("   ✅ Epic 1 compliance validation function test passed")
        
        test_results.append(("Epic 1 Performance", True))
        
    except Exception as e:
        print(f"   ❌ Performance validation test failed: {e}")
        test_results.append(("Epic 1 Performance", False))
    
    # Integration Tests
    print("\n🔗 Testing SDK Integration...")
    integration_tests = TestSDKIntegration()
    
    try:
        await integration_tests.test_plugin_type_mapper()
        print("   ✅ Plugin type mapper test passed")
        
        await integration_tests.test_category_mapper()
        print("   ✅ Category mapper test passed")
        
        test_results.append(("SDK Integration", True))
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        test_results.append(("SDK Integration", False))
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("=" * 30)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    success_rate = passed / total
    epic1_compliant = success_rate >= 0.8  # 80% pass rate required
    
    print(f"Epic 1 Compliance: {'✅ MAINTAINED' if epic1_compliant else '❌ COMPROMISED'}")
    
    return epic1_compliant, test_results


# Main execution
if __name__ == "__main__":
    import sys
    
    async def main():
        compliant, results = await run_comprehensive_tests()
        
        if compliant:
            print("\n🎉 All tests passed! SDK is Epic 1 compliant.")
            sys.exit(0)
        else:
            print("\n⚠️ Some tests failed. Check results above.")
            sys.exit(1)
    
    asyncio.run(main())