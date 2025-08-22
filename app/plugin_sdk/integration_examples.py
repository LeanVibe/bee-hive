"""
Integration Examples for LeanVibe Plugin SDK.

Demonstrates how to use the SDK with AdvancedPluginManager and Plugin Marketplace integration.
Shows complete end-to-end workflows for plugin development, registration, and deployment.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from .interfaces import WorkflowPlugin, PluginType
from .models import PluginConfig, TaskInterface, TaskResult, PluginEvent, EventSeverity
from .decorators import plugin_method, performance_tracked, error_handled
from .integration import (
    UnifiedPluginSDK, SDKPluginManagerIntegration, SDKMarketplaceIntegration,
    PluginIntegrationError
)
from .examples import DataPipelinePlugin, SystemMonitorPlugin

# Import existing LeanVibe components (for example setup)
from ..core.advanced_plugin_manager import AdvancedPluginManager, PluginSecurityLevel
from ..core.plugin_marketplace import Developer, PluginCategory
from ..core.logging_service import get_component_logger

logger = get_component_logger("plugin_sdk_integration_examples")


class ExampleIntegrationPlugin(WorkflowPlugin):
    """Example plugin demonstrating SDK integration capabilities."""
    
    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self.processed_tasks = 0
        self.processing_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_response_time_ms": 0.0
        }
    
    async def _on_initialize(self) -> None:
        """Initialize the integration example plugin."""
        await self.log_info("ExampleIntegrationPlugin initialized for SDK integration demo")
        
        # Validate configuration
        required_params = ["demo_mode", "integration_test"]
        for param in required_params:
            if param not in self.config.parameters:
                raise ValueError(f"Missing required parameter: {param}")
    
    @performance_tracked(alert_threshold_ms=40, memory_limit_mb=20)
    @plugin_method(timeout_seconds=30, max_retries=2)
    async def handle_task(self, task: TaskInterface) -> TaskResult:
        """Handle integration example tasks."""
        
        start_time = datetime.utcnow()
        
        try:
            task_type = task.task_type
            
            if task_type == "process_integration_data":
                return await self._process_integration_data(task)
            elif task_type == "test_performance":
                return await self._test_performance(task)
            elif task_type == "demonstrate_features":
                return await self._demonstrate_features(task)
            elif task_type == "get_statistics":
                return await self._get_statistics(task)
            else:
                return TaskResult(
                    success=False,
                    plugin_id=self.plugin_id,
                    task_id=task.task_id,
                    error=f"Unknown task type: {task_type}",
                    error_code="INVALID_TASK_TYPE"
                )
        
        except Exception as e:
            self.processing_stats["failed_executions"] += 1
            await self.log_error(f"Task execution failed: {e}")
            
            return TaskResult(
                success=False,
                plugin_id=self.plugin_id,
                task_id=task.task_id,
                error=str(e),
                error_code="EXECUTION_ERROR"
            )
        
        finally:
            # Update statistics
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.processing_stats["total_executions"] += 1
            
            # Update average response time
            current_avg = self.processing_stats["average_response_time_ms"]
            total_executions = self.processing_stats["total_executions"]
            new_avg = ((current_avg * (total_executions - 1)) + execution_time) / total_executions
            self.processing_stats["average_response_time_ms"] = new_avg
    
    async def _process_integration_data(self, task: TaskInterface) -> TaskResult:
        """Process data demonstrating integration capabilities."""
        
        input_data = task.parameters.get("data", [])
        processing_mode = task.parameters.get("mode", "standard")
        
        await task.update_status("running", progress=0.2)
        
        # Simulate different processing modes
        if processing_mode == "fast":
            # Epic 1 optimized processing
            processed_data = [{"id": item.get("id", i), "processed": True, "mode": "fast"} 
                            for i, item in enumerate(input_data)]
            await asyncio.sleep(0.01)  # Minimal processing time
            
        elif processing_mode == "comprehensive":
            # More thorough processing
            processed_data = []
            for i, item in enumerate(input_data):
                processed_item = {
                    "id": item.get("id", i),
                    "original_data": item,
                    "processed": True,
                    "mode": "comprehensive",
                    "processed_at": datetime.utcnow().isoformat(),
                    "processing_metadata": {
                        "version": "1.0.0",
                        "processor": "ExampleIntegrationPlugin"
                    }
                }
                processed_data.append(processed_item)
                
                # Update progress
                progress = 0.2 + (0.6 * (i + 1) / len(input_data))
                await task.update_status("running", progress=progress)
                
                # Small delay to simulate processing
                await asyncio.sleep(0.001)
        
        else:
            # Standard processing
            processed_data = [{"id": item.get("id", i), "processed": True, "mode": "standard"} 
                            for i, item in enumerate(input_data)]
        
        await task.update_status("running", progress=0.9)
        
        # Emit processing event
        processing_event = PluginEvent(
            event_type="data_processed",
            plugin_id=self.plugin_id,
            data={
                "records_processed": len(processed_data),
                "processing_mode": processing_mode,
                "task_id": task.task_id
            },
            task_id=task.task_id
        )
        await self.emit_event(processing_event)
        
        self.processing_stats["successful_executions"] += 1
        
        return TaskResult(
            success=True,
            plugin_id=self.plugin_id,
            task_id=task.task_id,
            data={
                "processed_data": processed_data,
                "processing_summary": {
                    "input_count": len(input_data),
                    "output_count": len(processed_data),
                    "processing_mode": processing_mode
                }
            }
        )
    
    async def _test_performance(self, task: TaskInterface) -> TaskResult:
        """Test performance characteristics for Epic 1 compliance."""
        
        test_iterations = task.parameters.get("iterations", 10)
        payload_size = task.parameters.get("payload_size", 100)
        
        performance_results = []
        
        for i in range(test_iterations):
            start_time = datetime.utcnow()
            
            # Simulate workload
            test_data = [{"id": j, "value": f"test_value_{j}"} for j in range(payload_size)]
            
            # Process data
            result_data = []
            for item in test_data:
                result_data.append({
                    "processed_id": item["id"],
                    "processed_value": item["value"].upper(),
                    "processed_at": datetime.utcnow().isoformat()
                })
            
            end_time = datetime.utcnow()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            performance_results.append({
                "iteration": i + 1,
                "execution_time_ms": execution_time_ms,
                "payload_size": payload_size,
                "epic1_compliant": execution_time_ms < 50
            })
            
            # Brief pause between iterations
            await asyncio.sleep(0.001)
        
        # Calculate summary statistics
        execution_times = [r["execution_time_ms"] for r in performance_results]
        avg_time = sum(execution_times) / len(execution_times)
        max_time = max(execution_times)
        min_time = min(execution_times)
        epic1_compliant = all(r["epic1_compliant"] for r in performance_results)
        
        return TaskResult(
            success=True,
            plugin_id=self.plugin_id,
            task_id=task.task_id,
            data={
                "performance_results": performance_results,
                "summary": {
                    "average_time_ms": avg_time,
                    "max_time_ms": max_time,
                    "min_time_ms": min_time,
                    "total_iterations": test_iterations,
                    "epic1_compliant": epic1_compliant,
                    "compliance_rate": sum(1 for r in performance_results if r["epic1_compliant"]) / len(performance_results)
                }
            }
        )
    
    async def _demonstrate_features(self, task: TaskInterface) -> TaskResult:
        """Demonstrate various SDK features."""
        
        features_to_demo = task.parameters.get("features", ["events", "logging", "caching"])
        demo_results = {}
        
        # Demonstrate event emission
        if "events" in features_to_demo:
            demo_event = PluginEvent(
                event_type="feature_demonstration",
                plugin_id=self.plugin_id,
                data={
                    "feature": "events",
                    "description": "Demonstrating event emission capabilities",
                    "timestamp": datetime.utcnow().isoformat()
                },
                severity=EventSeverity.INFO,
                task_id=task.task_id
            )
            await self.emit_event(demo_event)
            demo_results["events"] = "Event emitted successfully"
        
        # Demonstrate logging
        if "logging" in features_to_demo:
            await self.log_info("Demonstrating info-level logging")
            await self.log_warning("Demonstrating warning-level logging")
            await self.log_debug("Demonstrating debug-level logging")
            demo_results["logging"] = "Various log levels demonstrated"
        
        # Demonstrate caching (simulated)
        if "caching" in features_to_demo:
            cache_key = f"demo_cache_{task.task_id}"
            cache_value = {
                "cached_at": datetime.utcnow().isoformat(),
                "demo_data": "This would be cached in a real implementation"
            }
            # In a real implementation, this would use the @cached_result decorator
            demo_results["caching"] = f"Cache simulation with key: {cache_key}"
        
        # Demonstrate error handling
        if "error_handling" in features_to_demo:
            try:
                # Simulate a controlled error
                if task.parameters.get("trigger_error", False):
                    raise ValueError("Controlled error for demonstration")
                demo_results["error_handling"] = "Error handling ready (no error triggered)"
            except ValueError as e:
                demo_results["error_handling"] = f"Error caught and handled: {e}"
        
        return TaskResult(
            success=True,
            plugin_id=self.plugin_id,
            task_id=task.task_id,
            data={
                "demonstrated_features": demo_results,
                "total_features": len(features_to_demo),
                "sdk_capabilities": [
                    "Event emission",
                    "Structured logging",
                    "Performance tracking",
                    "Error handling",
                    "Task status updates",
                    "Result caching",
                    "Resource management"
                ]
            }
        )
    
    async def _get_statistics(self, task: TaskInterface) -> TaskResult:
        """Get plugin processing statistics."""
        
        return TaskResult(
            success=True,
            plugin_id=self.plugin_id,
            task_id=task.task_id,
            data={
                "processing_stats": self.processing_stats,
                "plugin_info": {
                    "plugin_id": self.plugin_id,
                    "plugin_type": self.plugin_type.value,
                    "config_name": self.config.name,
                    "config_version": self.config.version,
                    "is_initialized": self.is_initialized
                }
            }
        )
    
    async def _on_cleanup(self) -> None:
        """Cleanup plugin resources."""
        await self.log_info("ExampleIntegrationPlugin cleanup completed")


class IntegrationExamples:
    """Collection of integration examples and workflows."""
    
    def __init__(self):
        self.logger = logger
    
    async def basic_integration_example(self):
        """Basic example of SDK integration with AdvancedPluginManager."""
        
        print("🔌 Basic SDK Integration Example")
        print("=" * 50)
        
        try:
            # Create AdvancedPluginManager (normally would be injected)
            advanced_manager = AdvancedPluginManager()
            
            # Create SDK integration
            manager_integration = SDKPluginManagerIntegration(advanced_manager)
            
            # Create example plugin
            plugin_config = PluginConfig(
                name="ExampleIntegrationPlugin",
                version="1.0.0",
                description="Example plugin demonstrating SDK integration",
                parameters={
                    "demo_mode": True,
                    "integration_test": True,
                    "batch_size": 100
                }
            )
            
            example_plugin = ExampleIntegrationPlugin(plugin_config)
            await example_plugin.initialize()
            
            print(f"✅ Created plugin: {plugin_config.name}")
            
            # Register with AdvancedPluginManager
            wrapper_id = await manager_integration.register_sdk_plugin(
                example_plugin,
                plugin_config,
                security_level=PluginSecurityLevel.TRUSTED
            )
            
            print(f"✅ Registered with AdvancedPluginManager: {wrapper_id}")
            
            # Test plugin execution
            test_task = TaskInterface(
                task_id="basic_test",
                task_type="process_integration_data",
                parameters={
                    "data": [{"id": i, "value": f"test_{i}"} for i in range(10)],
                    "mode": "fast"
                }
            )
            
            result = await example_plugin.handle_task(test_task)
            
            print(f"✅ Task execution: {'SUCCESS' if result.success else 'FAILED'}")
            if result.success:
                processed_count = len(result.data.get("processed_data", []))
                print(f"   Processed {processed_count} records")
                print(f"   Execution time: {result.execution_time_ms:.2f}ms")
            
            # Get performance metrics
            performance_metrics = await advanced_manager.get_performance_metrics()
            
            print(f"\n📊 Performance Metrics:")
            print(f"   Total plugins: {performance_metrics['total_plugins']}")
            print(f"   Loaded plugins: {performance_metrics['loaded_plugins']}")
            print(f"   Memory usage: {performance_metrics['total_memory_mb']:.2f}MB")
            print(f"   Epic 1 compliant: {performance_metrics['epic1_compliant']}")
            
            # Cleanup
            await manager_integration.unregister_sdk_plugin(wrapper_id)
            print(f"✅ Unregistered plugin: {wrapper_id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Integration example failed: {e}")
            return False
    
    async def marketplace_integration_example(self):
        """Example of SDK integration with Plugin Marketplace."""
        
        print("\n🏪 Marketplace Integration Example")
        print("=" * 50)
        
        try:
            # Note: This is a simplified example since we don't have a real marketplace instance
            print("📝 This example demonstrates the workflow for marketplace integration:")
            
            # Create example plugin
            plugin_config = PluginConfig(
                name="MarketplaceExamplePlugin",
                version="1.2.0",
                description="Example plugin for marketplace submission",
                parameters={
                    "demo_mode": True,
                    "marketplace_ready": True
                }
            )
            
            example_plugin = ExampleIntegrationPlugin(plugin_config)
            await example_plugin.initialize()
            
            print(f"✅ Created marketplace plugin: {plugin_config.name}")
            
            # Create developer profile
            developer = Developer(
                developer_id="example_developer",
                name="Example Developer",
                email="developer@example.com",
                organization="Example Corp",
                verified=True,
                reputation_score=4.5
            )
            
            print(f"✅ Created developer profile: {developer.name}")
            
            # Simulate marketplace integration setup
            print("\n📋 Marketplace submission would include:")
            print(f"   • Plugin Name: {plugin_config.name}")
            print(f"   • Version: {plugin_config.version}")
            print(f"   • Developer: {developer.name} ({developer.email})")
            print(f"   • Category: {PluginCategory.UTILITY.value}")
            print(f"   • Security Level: SANDBOX")
            print(f"   • Performance: Epic 1 Compliant")
            
            # Simulate validation
            print("\n🔍 Validation checks:")
            
            # Performance validation
            perf_task = TaskInterface(
                task_id="perf_validation",
                task_type="test_performance",
                parameters={"iterations": 5, "payload_size": 50}
            )
            
            perf_result = await example_plugin.handle_task(perf_task)
            
            if perf_result.success:
                summary = perf_result.data["summary"]
                print(f"   ✅ Performance: Avg {summary['average_time_ms']:.2f}ms")
                print(f"   ✅ Epic 1 Compliance: {summary['epic1_compliant']}")
            
            # Feature demonstration
            feature_task = TaskInterface(
                task_id="feature_demo",
                task_type="demonstrate_features",
                parameters={"features": ["events", "logging", "error_handling"]}
            )
            
            feature_result = await example_plugin.handle_task(feature_task)
            
            if feature_result.success:
                features = feature_result.data["demonstrated_features"]
                print(f"   ✅ Features demonstrated: {len(features)}")
            
            print("\n✅ Marketplace integration simulation completed")
            
            return True
            
        except Exception as e:
            print(f"❌ Marketplace integration example failed: {e}")
            return False
    
    async def unified_sdk_example(self):
        """Example of unified SDK usage with both manager and marketplace."""
        
        print("\n🎯 Unified SDK Example")
        print("=" * 50)
        
        try:
            # Create components (normally would be dependency injected)
            advanced_manager = AdvancedPluginManager()
            # marketplace = PluginMarketplace()  # Would be real instance
            
            print("📝 This example demonstrates unified SDK workflow:")
            
            # Create example plugin
            plugin_config = PluginConfig(
                name="UnifiedSDKExample",
                version="2.0.0",
                description="Example plugin for unified SDK demonstration",
                parameters={
                    "unified_mode": True,
                    "epic1_optimized": True
                }
            )
            
            example_plugin = ExampleIntegrationPlugin(plugin_config)
            await example_plugin.initialize()
            
            print(f"✅ Created unified plugin: {plugin_config.name}")
            
            # Simulate unified registration
            print("\n🔗 Unified registration would include:")
            print("   • AdvancedPluginManager registration")
            print("   • Performance validation")
            print("   • Security scanning")
            print("   • Optional marketplace submission")
            print("   • Unified monitoring and metrics")
            
            # Test comprehensive functionality
            test_cases = [
                {
                    "name": "Performance Test",
                    "task": TaskInterface(
                        task_id="unified_perf_test",
                        task_type="test_performance",
                        parameters={"iterations": 3, "payload_size": 25}
                    )
                },
                {
                    "name": "Feature Demo",
                    "task": TaskInterface(
                        task_id="unified_feature_test",
                        task_type="demonstrate_features",
                        parameters={"features": ["events", "logging"]}
                    )
                },
                {
                    "name": "Data Processing",
                    "task": TaskInterface(
                        task_id="unified_data_test",
                        task_type="process_integration_data",
                        parameters={
                            "data": [{"id": i, "value": f"unified_{i}"} for i in range(5)],
                            "mode": "comprehensive"
                        }
                    )
                }
            ]
            
            print(f"\n🧪 Running {len(test_cases)} test cases:")
            
            all_passed = True
            for test_case in test_cases:
                result = await example_plugin.handle_task(test_case["task"])
                status = "✅ PASS" if result.success else "❌ FAIL"
                print(f"   {status} {test_case['name']}")
                
                if not result.success:
                    all_passed = False
                    print(f"      Error: {result.error}")
            
            # Get final statistics
            stats_task = TaskInterface(
                task_id="final_stats",
                task_type="get_statistics",
                parameters={}
            )
            
            stats_result = await example_plugin.handle_task(stats_task)
            
            if stats_result.success:
                stats = stats_result.data["processing_stats"]
                print(f"\n📊 Final Statistics:")
                print(f"   Total executions: {stats['total_executions']}")
                print(f"   Successful: {stats['successful_executions']}")
                print(f"   Failed: {stats['failed_executions']}")
                print(f"   Avg response time: {stats['average_response_time_ms']:.2f}ms")
            
            print(f"\n{'✅ All tests passed!' if all_passed else '⚠️ Some tests failed'}")
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Unified SDK example failed: {e}")
            return False
    
    async def real_world_integration_example(self):
        """Real-world integration example using existing SDK plugins."""
        
        print("\n🌍 Real-World Integration Example")
        print("=" * 50)
        
        try:
            # Use existing SDK example plugins
            pipeline_config = PluginConfig(
                name="IntegratedDataPipeline",
                version="1.0.0",
                description="Data pipeline plugin integrated with LeanVibe system",
                parameters={
                    "batch_size": 500,
                    "max_retries": 3,
                    "output_format": "json",
                    "enable_validation": True,
                    "pipeline_steps": [
                        {
                            "name": "validation_step",
                            "type": "validate",
                            "parameters": {
                                "validations": [
                                    {"field": "id", "rule": "required"},
                                    {"field": "value", "rule": "type", "type": "str"}
                                ]
                            }
                        },
                        {
                            "name": "transform_step",
                            "type": "transform",
                            "parameters": {
                                "transformations": [
                                    {"field": "value", "operation": "uppercase"},
                                    {"field": "processed_at", "operation": "timestamp"}
                                ]
                            }
                        }
                    ]
                }
            )
            
            pipeline_plugin = DataPipelinePlugin(pipeline_config)
            await pipeline_plugin.initialize()
            
            print(f"✅ Initialized DataPipelinePlugin")
            
            # Create AdvancedPluginManager integration
            advanced_manager = AdvancedPluginManager()
            manager_integration = SDKPluginManagerIntegration(advanced_manager)
            
            # Register the real-world plugin
            wrapper_id = await manager_integration.register_sdk_plugin(
                pipeline_plugin,
                pipeline_config,
                security_level=PluginSecurityLevel.VERIFIED
            )
            
            print(f"✅ Registered with AdvancedPluginManager: {wrapper_id}")
            
            # Test with realistic data
            realistic_data = [
                {"id": 1, "value": "customer_data_batch_1", "priority": "high"},
                {"id": 2, "value": "customer_data_batch_2", "priority": "medium"},
                {"id": 3, "value": "customer_data_batch_3", "priority": "low"},
                {"id": 4, "value": "customer_data_batch_4", "priority": "high"},
                {"id": 5, "value": "customer_data_batch_5", "priority": "medium"}
            ]
            
            processing_task = TaskInterface(
                task_id="real_world_processing",
                task_type="process_data",
                parameters={"input_data": realistic_data}
            )
            
            print(f"\n🔄 Processing {len(realistic_data)} records...")
            
            result = await pipeline_plugin.handle_task(processing_task)
            
            if result.success:
                summary = result.data["processing_summary"]
                print(f"✅ Processing completed successfully")
                print(f"   Input records: {summary['input_records']}")
                print(f"   Output records: {summary['final_records']}")
                print(f"   Processing time: {result.execution_time_ms:.2f}ms")
                print(f"   Epic 1 compliant: {result.execution_time_ms < 50}")
            else:
                print(f"❌ Processing failed: {result.error}")
            
            # Test monitoring integration
            monitor_config = PluginConfig(
                name="IntegratedSystemMonitor",
                version="1.0.0", 
                description="System monitor integrated with LeanVibe",
                parameters={
                    "collection_interval": 5,
                    "retention_hours": 1,
                    "enable_alerts": True
                }
            )
            
            monitor_plugin = SystemMonitorPlugin(monitor_config)
            await monitor_plugin.initialize()
            
            print(f"✅ Initialized SystemMonitorPlugin")
            
            # Test metrics collection
            metrics_task = TaskInterface(
                task_id="collect_metrics",
                task_type="collect_metrics",
                parameters={}
            )
            
            metrics_result = await monitor_plugin.handle_task(metrics_task)
            
            if metrics_result.success:
                metrics = metrics_result.data["metrics"]
                print(f"✅ Metrics collected:")
                print(f"   CPU usage: {metrics['cpu_percent']:.1f}%")
                print(f"   Memory usage: {metrics['memory_percent']:.1f}%")
                print(f"   Collection time: {metrics_result.execution_time_ms:.2f}ms")
            
            # Cleanup
            await manager_integration.unregister_sdk_plugin(wrapper_id)
            await pipeline_plugin.cleanup()
            await monitor_plugin.cleanup()
            
            print(f"\n✅ Real-world integration example completed successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Real-world integration example failed: {e}")
            return False
    
    async def run_all_examples(self):
        """Run all integration examples."""
        
        print("🚀 LeanVibe Plugin SDK Integration Examples")
        print("=" * 60)
        
        examples = [
            ("Basic Integration", self.basic_integration_example),
            ("Marketplace Integration", self.marketplace_integration_example),
            ("Unified SDK", self.unified_sdk_example),
            ("Real-World Integration", self.real_world_integration_example)
        ]
        
        results = []
        
        for name, example_func in examples:
            try:
                result = await example_func()
                results.append((name, result))
            except Exception as e:
                print(f"❌ {name} failed with exception: {e}")
                results.append((name, False))
        
        # Summary
        print(f"\n📊 Integration Examples Summary")
        print("=" * 40)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"   {status} {name}")
        
        print(f"\nOverall: {passed}/{total} examples passed")
        
        return passed == total


# Usage example
async def main():
    """Main function to run integration examples."""
    examples = IntegrationExamples()
    success = await examples.run_all_examples()
    
    if success:
        print("\n🎉 All integration examples completed successfully!")
    else:
        print("\n⚠️ Some integration examples failed. Check the output above.")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())