#!/usr/bin/env python3
"""
Test Suite for OrchestratorV2 - Phase 0 POC Week 2
LeanVibe Agent Hive 2.0 - Orchestrator Consolidation Testing

This test suite validates the core functionality of OrchestratorV2 and its plugin system:
1. Core orchestrator operations (spawn, delegate, health)
2. Plugin system (dependency resolution, hooks, performance monitoring)  
3. Migration adapters (shadow testing, comparison, fallback)
4. Performance benchmarks (spawn time, delegation time, plugin overhead)
5. Error handling and circuit breaker functionality

Tests designed to work without external dependencies for CI/CD compatibility.
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List

# Test imports
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from app.core.orchestrator_v2 import (
        OrchestratorV2,
        OrchestratorConfig,
        OrchestratorPlugin,
        PluginManager,
        PluginStateManager,
        PluginPerformanceMonitor,
        HookManager,
        AgentRole,
        AgentStatus,
        Task,
        TaskExecution,
        TaskExecutionState,
        MessagePriority
    )
    
    from app.core.orchestrator_v2_plugins import (
        ProductionPlugin,
        PerformancePlugin,
        AutomationPlugin,
        DevelopmentPlugin,
        MonitoringPlugin,
        create_standard_plugin_set
    )
    
    from app.core.orchestrator_v2_migration import (
        MigrationMode,
        MigrationConfig,
        MigrationMetrics,
        ProductionOrchestratorAdapter,
        OrchestratorMigrationManager
    )
    
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# ================================================================================
# Test Configuration
# ================================================================================

class TestConfig:
    """Test configuration."""
    PERFORMANCE_TEST_COUNT = 100
    PLUGIN_TIMEOUT_TEST_MS = 50
    CONCURRENT_AGENT_TEST_COUNT = 10
    MIGRATION_TEST_SAMPLES = 25

# ================================================================================
# Mock Plugin for Testing
# ================================================================================

class MockTestPlugin(OrchestratorPlugin):
    """Mock plugin for testing plugin system."""
    
    plugin_name = "MockTestPlugin"
    dependencies = []
    hook_timeout_ms = 25
    
    def __init__(self, state_manager, performance_monitor):
        super().__init__(state_manager, performance_monitor)
        self.hook_calls = []
    
    async def before_agent_spawn(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        self.hook_calls.append(("before_agent_spawn", agent_config))
        agent_config["mock_enhanced"] = True
        return agent_config
    
    async def after_agent_spawn(self, agent: 'AgentInstance') -> None:
        self.hook_calls.append(("after_agent_spawn", agent.id))
    
    async def before_task_delegate(self, task: Task) -> Task:
        self.hook_calls.append(("before_task_delegate", task.id))
        task.metadata["mock_processed"] = True
        return task
    
    async def after_task_delegate(self, task: Task, agent_id: str) -> None:
        self.hook_calls.append(("after_task_delegate", task.id, agent_id))
    
    async def on_performance_metric(self, metric_name: str, value: float, metadata: Dict[str, Any]) -> None:
        self.hook_calls.append(("on_performance_metric", metric_name, value))

class SlowMockPlugin(OrchestratorPlugin):
    """Plugin that intentionally takes too long (for timeout testing)."""
    
    plugin_name = "SlowMockPlugin"
    dependencies = []
    hook_timeout_ms = 25
    
    async def before_agent_spawn(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.1)  # 100ms - exceeds timeout
        return agent_config

class DependentMockPlugin(OrchestratorPlugin):
    """Plugin with dependencies for testing dependency resolution."""
    
    plugin_name = "DependentMockPlugin"
    dependencies = ["MockTestPlugin"]
    
    async def after_agent_spawn(self, agent: 'AgentInstance') -> None:
        # This should be called after MockTestPlugin
        pass

# ================================================================================
# Core Orchestrator Tests
# ================================================================================

async def test_orchestrator_initialization():
    """Test basic orchestrator initialization."""
    print("\n🧪 Testing orchestrator initialization...")
    
    config = OrchestratorConfig(
        max_concurrent_agents=10,
        agent_spawn_timeout_ms=100,
        task_delegation_timeout_ms=500
    )
    
    orchestrator = OrchestratorV2(config, [])
    await orchestrator.initialize()
    
    assert orchestrator._running == True, "Orchestrator should be running after initialization"
    assert orchestrator.communication_manager is not None, "Communication manager should be initialized"
    
    # Test health status
    health = await orchestrator.get_health_status()
    assert health["status"] == "healthy", "Health status should be healthy"
    assert health["active_agents"] == 0, "Should start with 0 active agents"
    
    await orchestrator.shutdown()
    
    print("✅ Orchestrator initialization tests passed")
    return True

async def test_agent_spawning():
    """Test agent spawning functionality."""
    print("\n🧪 Testing agent spawning...")
    
    config = OrchestratorConfig(max_concurrent_agents=5)
    orchestrator = OrchestratorV2(config, [MockTestPlugin])
    await orchestrator.initialize()
    
    # Test single agent spawn
    agent_id = await orchestrator.spawn_agent(
        AgentRole.CLAUDE_CODE, 
        capabilities=["code_generation", "debugging"]
    )
    
    assert agent_id is not None, "Agent ID should be returned"
    assert agent_id in orchestrator.active_agents, "Agent should be in active agents"
    
    agent = orchestrator.active_agents[agent_id]
    assert agent.role == AgentRole.CLAUDE_CODE, "Agent role should match"
    assert agent.status == AgentStatus.ACTIVE, "Agent should be active"
    assert "code_generation" in agent.capabilities, "Agent should have specified capabilities"
    
    # Test concurrent agent limit
    try:
        for i in range(6):  # Try to spawn 6 more agents (total 7, limit is 5)
            await orchestrator.spawn_agent(AgentRole.TASK_EXECUTOR)
    except ValueError as e:
        assert "Maximum concurrent agents" in str(e), "Should raise max agents error"
    
    await orchestrator.shutdown()
    
    print("✅ Agent spawning tests passed")
    return True

async def test_task_delegation():
    """Test task delegation functionality."""
    print("\n🧪 Testing task delegation...")
    
    config = OrchestratorConfig()
    orchestrator = OrchestratorV2(config, [MockTestPlugin])
    await orchestrator.initialize()
    
    # Spawn an agent first
    agent_id = await orchestrator.spawn_agent(AgentRole.TASK_EXECUTOR)
    
    # Create and delegate a task
    task = Task(
        type="test_task",
        description="Test task for delegation",
        payload={"test_data": "value"},
        requirements=["basic_processing"]
    )
    
    task_id = await orchestrator.delegate_task(task, agent_id)
    
    assert task_id == task.id, "Returned task ID should match task ID"
    assert task_id in orchestrator.task_executions, "Task should be in executions"
    
    execution = orchestrator.task_executions[task_id]
    assert execution.agent_id == agent_id, "Task should be assigned to specified agent"
    assert execution.state == TaskExecutionState.RUNNING, "Task should be running"
    
    # Check agent status
    agent = orchestrator.active_agents[agent_id]
    assert agent.current_task_id == task_id, "Agent should have current task ID"
    assert agent.status == AgentStatus.BUSY, "Agent should be busy"
    
    await orchestrator.shutdown()
    
    print("✅ Task delegation tests passed")
    return True

# ================================================================================
# Plugin System Tests
# ================================================================================

async def test_plugin_dependency_resolution():
    """Test plugin dependency resolution and loading order."""
    print("\n🧪 Testing plugin dependency resolution...")
    
    state_manager = PluginStateManager()
    performance_monitor = PluginPerformanceMonitor()
    plugin_manager = PluginManager(state_manager, performance_monitor)
    
    # Register plugins with dependencies
    plugin_manager.register_plugin(MockTestPlugin)
    plugin_manager.register_plugin(DependentMockPlugin)
    
    # Create mock orchestrator
    config = OrchestratorConfig()
    orchestrator = OrchestratorV2(config, [])
    await orchestrator.initialize()
    
    # Load plugins - should resolve dependencies
    await plugin_manager.load_plugins(orchestrator)
    
    # Check loading order
    plugin_names = [name for name, _ in plugin_manager.sorted_plugins]
    mock_index = plugin_names.index("MockTestPlugin")
    dependent_index = plugin_names.index("DependentMockPlugin")
    
    assert mock_index < dependent_index, "MockTestPlugin should load before DependentMockPlugin"
    
    # Test circular dependency detection
    class CircularPlugin1(OrchestratorPlugin):
        plugin_name = "CircularPlugin1"
        dependencies = ["CircularPlugin2"]
    
    class CircularPlugin2(OrchestratorPlugin):
        plugin_name = "CircularPlugin2"  
        dependencies = ["CircularPlugin1"]
    
    plugin_manager.register_plugin(CircularPlugin1)
    plugin_manager.register_plugin(CircularPlugin2)
    
    try:
        await plugin_manager.load_plugins(orchestrator)
        assert False, "Should raise CircularDependencyError"
    except Exception as e:
        assert "circular dependency" in str(e).lower(), "Should detect circular dependency"
    
    await orchestrator.shutdown()
    
    print("✅ Plugin dependency resolution tests passed")
    return True

async def test_plugin_hooks():
    """Test plugin hook system."""
    print("\n🧪 Testing plugin hook system...")
    
    config = OrchestratorConfig()
    orchestrator = OrchestratorV2(config, [MockTestPlugin])
    await orchestrator.initialize()
    
    # Get the mock plugin instance
    mock_plugin = orchestrator.plugin_manager.get_plugin("MockTestPlugin")
    assert mock_plugin is not None, "Mock plugin should be loaded"
    
    # Test agent spawning hooks
    agent_id = await orchestrator.spawn_agent(AgentRole.CLAUDE_CODE)
    
    # Check that hooks were called
    hook_calls = mock_plugin.hook_calls
    before_spawn_calls = [call for call in hook_calls if call[0] == "before_agent_spawn"]
    after_spawn_calls = [call for call in hook_calls if call[0] == "after_agent_spawn"]
    
    assert len(before_spawn_calls) == 1, "before_agent_spawn should be called once"
    assert len(after_spawn_calls) == 1, "after_agent_spawn should be called once"
    assert after_spawn_calls[0][1] == agent_id, "after_agent_spawn should receive agent ID"
    
    # Test task delegation hooks
    task = Task(type="test_task")
    task_id = await orchestrator.delegate_task(task, agent_id)
    
    before_delegate_calls = [call for call in hook_calls if call[0] == "before_task_delegate"]
    after_delegate_calls = [call for call in hook_calls if call[0] == "after_task_delegate"]
    
    assert len(before_delegate_calls) == 1, "before_task_delegate should be called once"
    assert len(after_delegate_calls) == 1, "after_task_delegate should be called once"
    
    await orchestrator.shutdown()
    
    print("✅ Plugin hook tests passed")
    return True

async def test_plugin_performance_monitoring():
    """Test plugin performance monitoring and timeout handling."""
    print("\n🧪 Testing plugin performance monitoring...")
    
    performance_monitor = PluginPerformanceMonitor(default_timeout_ms=50)
    
    # Test normal operation timing
    async with performance_monitor.time_hook("TestPlugin", "test_hook"):
        await asyncio.sleep(0.01)  # 10ms - should be fine
    
    metrics = performance_monitor.get_performance_summary()
    assert metrics["total_hooks_executed"] == 1, "Should record hook execution"
    
    # Test timeout detection
    try:
        async with performance_monitor.time_hook("SlowPlugin", "slow_hook", timeout_ms=25):
            await asyncio.sleep(0.05)  # 50ms - should timeout
        assert False, "Should have raised timeout error"
    except asyncio.TimeoutError:
        pass  # Expected
    
    # Test plugin disabling after repeated violations
    for i in range(6):  # Trigger 6 timeouts
        try:
            async with performance_monitor.time_hook("BadPlugin", "bad_hook", timeout_ms=10):
                await asyncio.sleep(0.02)  # Always timeout
        except asyncio.TimeoutError:
            pass
    
    assert performance_monitor.is_plugin_disabled("BadPlugin"), "Plugin should be disabled after repeated timeouts"
    
    print("✅ Plugin performance monitoring tests passed")
    return True

async def test_plugin_state_sandboxing():
    """Test plugin state management and sandboxing."""
    print("\n🧪 Testing plugin state sandboxing...")
    
    state_manager = PluginStateManager()
    
    # Get state for different plugins
    plugin1_state = state_manager.get_plugin_state("Plugin1")
    plugin2_state = state_manager.get_plugin_state("Plugin2")
    
    # Modify states independently
    plugin1_state["data"] = "plugin1_data"
    plugin2_state["data"] = "plugin2_data"
    
    # Verify isolation
    assert plugin1_state["data"] == "plugin1_data", "Plugin1 state should be isolated"
    assert plugin2_state["data"] == "plugin2_data", "Plugin2 state should be isolated"
    assert plugin1_state is not plugin2_state, "State dictionaries should be separate"
    
    # Test state snapshot and restore
    snapshot = await state_manager.snapshot_state()
    assert "Plugin1" in snapshot, "Snapshot should include Plugin1 state"
    assert "Plugin2" in snapshot, "Snapshot should include Plugin2 state"
    
    # Modify state and restore
    plugin1_state["data"] = "modified"
    await state_manager.restore_state(snapshot)
    
    restored_state = state_manager.get_plugin_state("Plugin1")
    assert restored_state["data"] == "plugin1_data", "State should be restored from snapshot"
    
    print("✅ Plugin state sandboxing tests passed")
    return True

# ================================================================================
# Migration System Tests
# ================================================================================

async def test_migration_metrics():
    """Test migration metrics tracking."""
    print("\n🧪 Testing migration metrics...")
    
    metrics = MigrationMetrics()
    
    # Record some legacy requests
    metrics.record_legacy_request(True, 100.0)
    metrics.record_legacy_request(True, 150.0)
    metrics.record_legacy_request(False, 200.0)  # Failed request
    
    # Record some V2 requests  
    metrics.record_v2_request(True, 75.0)
    metrics.record_v2_request(True, 80.0)
    
    # Record comparisons
    metrics.record_comparison(True, True, False)  # Results match, V2 faster, no error diff
    metrics.record_comparison(False, False, True)  # Results differ, V2 slower, error diff
    
    summary = metrics.get_summary()
    
    # Verify legacy metrics
    assert summary["legacy_metrics"]["requests"] == 3, "Should track 3 legacy requests"
    assert summary["legacy_metrics"]["failures"] == 1, "Should track 1 legacy failure"
    assert summary["legacy_metrics"]["avg_response_time_ms"] == 150.0, "Should calculate average response time"
    
    # Verify V2 metrics
    assert summary["v2_metrics"]["requests"] == 2, "Should track 2 V2 requests"
    assert summary["v2_metrics"]["failures"] == 0, "Should track 0 V2 failures"
    assert summary["v2_metrics"]["avg_response_time_ms"] == 77.5, "Should calculate average response time"
    
    # Verify comparison metrics
    assert summary["comparison_results"]["total_comparisons"] == 2, "Should track 2 comparisons"
    assert summary["comparison_results"]["identical_results"] == 1, "Should track 1 identical result"
    
    print("✅ Migration metrics tests passed")
    return True

async def test_migration_adapter():
    """Test migration adapter functionality."""
    print("\n🧪 Testing migration adapter...")
    
    # Test shadow testing mode
    config = MigrationConfig(
        mode=MigrationMode.SHADOW_TESTING,
        enable_comparison=True
    )
    
    adapter = ProductionOrchestratorAdapter(config)
    
    # Test task delegation (will use mock legacy implementation)
    task_data = {
        "id": "test_task_123",
        "type": "test_task",
        "description": "Test task for migration",
        "payload": {"test": "data"}
    }
    
    result = await adapter.delegate_task_with_migration(task_data)
    
    assert result is not None, "Should return result from migration adapter"
    assert result["orchestrator_type"] == "legacy_production", "Should use legacy production orchestrator"
    
    # Check metrics were recorded
    metrics_summary = adapter.metrics.get_summary()
    assert metrics_summary["legacy_metrics"]["requests"] > 0, "Should record legacy request"
    
    print("✅ Migration adapter tests passed")
    return True

# ================================================================================
# Performance Tests
# ================================================================================

async def test_agent_spawn_performance():
    """Test agent spawning performance benchmarks."""
    print(f"\n🚀 Testing agent spawn performance ({TestConfig.PERFORMANCE_TEST_COUNT} agents)...")
    
    config = OrchestratorConfig(max_concurrent_agents=TestConfig.PERFORMANCE_TEST_COUNT + 10)
    orchestrator = OrchestratorV2(config, [])
    await orchestrator.initialize()
    
    spawn_times = []
    
    for i in range(TestConfig.PERFORMANCE_TEST_COUNT):
        start_time = time.perf_counter()
        
        agent_id = await orchestrator.spawn_agent(
            AgentRole.TASK_EXECUTOR,
            capabilities=[f"capability_{i % 5}"]
        )
        
        spawn_time = (time.perf_counter() - start_time) * 1000
        spawn_times.append(spawn_time)
        
        assert agent_id in orchestrator.active_agents, f"Agent {i} should be spawned successfully"
    
    # Calculate statistics
    avg_spawn_time = sum(spawn_times) / len(spawn_times)
    max_spawn_time = max(spawn_times)
    min_spawn_time = min(spawn_times)
    
    print(f"📊 Agent Spawn Performance:")
    print(f"   Average: {avg_spawn_time:.2f}ms")
    print(f"   Min: {min_spawn_time:.2f}ms")
    print(f"   Max: {max_spawn_time:.2f}ms")
    print(f"   Target: <{config.agent_spawn_timeout_ms}ms")
    
    # Performance assertions
    assert avg_spawn_time < config.agent_spawn_timeout_ms, f"Average spawn time should be under {config.agent_spawn_timeout_ms}ms"
    assert len(orchestrator.active_agents) == TestConfig.PERFORMANCE_TEST_COUNT, "All agents should be spawned"
    
    await orchestrator.shutdown()
    
    print("✅ Agent spawn performance tests passed")
    return True

async def test_task_delegation_performance():
    """Test task delegation performance benchmarks."""
    print(f"\n🚀 Testing task delegation performance ({TestConfig.PERFORMANCE_TEST_COUNT} tasks)...")
    
    config = OrchestratorConfig()
    orchestrator = OrchestratorV2(config, [])
    await orchestrator.initialize()
    
    # Spawn some agents first
    agent_ids = []
    for i in range(min(10, TestConfig.PERFORMANCE_TEST_COUNT)):
        agent_id = await orchestrator.spawn_agent(AgentRole.TASK_EXECUTOR)
        agent_ids.append(agent_id)
    
    delegation_times = []
    
    for i in range(TestConfig.PERFORMANCE_TEST_COUNT):
        task = Task(
            type="performance_test_task",
            description=f"Performance test task {i}",
            payload={"task_number": i}
        )
        
        start_time = time.perf_counter()
        
        task_id = await orchestrator.delegate_task(task)
        
        delegation_time = (time.perf_counter() - start_time) * 1000
        delegation_times.append(delegation_time)
        
        assert task_id in orchestrator.task_executions, f"Task {i} should be delegated successfully"
    
    # Calculate statistics
    avg_delegation_time = sum(delegation_times) / len(delegation_times)
    max_delegation_time = max(delegation_times)
    min_delegation_time = min(delegation_times)
    
    print(f"📊 Task Delegation Performance:")
    print(f"   Average: {avg_delegation_time:.2f}ms")
    print(f"   Min: {min_delegation_time:.2f}ms")
    print(f"   Max: {max_delegation_time:.2f}ms")
    print(f"   Target: <{config.task_delegation_timeout_ms}ms")
    
    # Performance assertions
    assert avg_delegation_time < config.task_delegation_timeout_ms, f"Average delegation time should be under {config.task_delegation_timeout_ms}ms"
    assert len(orchestrator.task_executions) == TestConfig.PERFORMANCE_TEST_COUNT, "All tasks should be delegated"
    
    await orchestrator.shutdown()
    
    print("✅ Task delegation performance tests passed")
    return True

async def test_concurrent_operations():
    """Test concurrent agent and task operations."""
    print(f"\n🚀 Testing concurrent operations ({TestConfig.CONCURRENT_AGENT_TEST_COUNT} concurrent agents)...")
    
    config = OrchestratorConfig(max_concurrent_agents=TestConfig.CONCURRENT_AGENT_TEST_COUNT + 5)
    orchestrator = OrchestratorV2(config, [MockTestPlugin])
    await orchestrator.initialize()
    
    # Test concurrent agent spawning
    spawn_tasks = []
    for i in range(TestConfig.CONCURRENT_AGENT_TEST_COUNT):
        task = orchestrator.spawn_agent(
            AgentRole.TASK_EXECUTOR,
            capabilities=[f"concurrent_capability_{i}"]
        )
        spawn_tasks.append(task)
    
    start_time = time.perf_counter()
    agent_ids = await asyncio.gather(*spawn_tasks)
    concurrent_spawn_time = (time.perf_counter() - start_time) * 1000
    
    print(f"📊 Concurrent Agent Spawn: {concurrent_spawn_time:.2f}ms for {TestConfig.CONCURRENT_AGENT_TEST_COUNT} agents")
    
    assert len(agent_ids) == TestConfig.CONCURRENT_AGENT_TEST_COUNT, "All agents should be spawned concurrently"
    assert len(orchestrator.active_agents) == TestConfig.CONCURRENT_AGENT_TEST_COUNT, "All agents should be active"
    
    # Test concurrent task delegation
    delegation_tasks = []
    for i, agent_id in enumerate(agent_ids):
        task = Task(
            type="concurrent_test_task",
            description=f"Concurrent task {i}",
            payload={"task_number": i}
        )
        delegation_task = orchestrator.delegate_task(task, agent_id)
        delegation_tasks.append(delegation_task)
    
    start_time = time.perf_counter()
    task_ids = await asyncio.gather(*delegation_tasks)
    concurrent_delegation_time = (time.perf_counter() - start_time) * 1000
    
    print(f"📊 Concurrent Task Delegation: {concurrent_delegation_time:.2f}ms for {TestConfig.CONCURRENT_AGENT_TEST_COUNT} tasks")
    
    assert len(task_ids) == TestConfig.CONCURRENT_AGENT_TEST_COUNT, "All tasks should be delegated concurrently"
    assert len(orchestrator.task_executions) == TestConfig.CONCURRENT_AGENT_TEST_COUNT, "All tasks should be tracked"
    
    await orchestrator.shutdown()
    
    print("✅ Concurrent operations tests passed")
    return True

# ================================================================================
# Integration Tests
# ================================================================================

async def test_full_integration():
    """Test full integration with all components."""
    print("\n🧪 Testing full integration...")
    
    config = OrchestratorConfig()
    orchestrator = OrchestratorV2(config, create_standard_plugin_set())
    await orchestrator.initialize()
    
    # Test with production plugins loaded
    plugins = orchestrator.plugin_manager.get_all_plugins()
    assert len(plugins) == 5, "Should load all standard plugins"
    assert "ProductionPlugin" in plugins, "Should load ProductionPlugin"
    assert "PerformancePlugin" in plugins, "Should load PerformancePlugin"
    
    # Test complete workflow
    agent_id = await orchestrator.spawn_agent(
        AgentRole.CLAUDE_CODE,
        capabilities=["code_generation", "debugging", "testing"]
    )
    
    task = Task(
        type="integration_test_task",
        description="Full integration test task",
        payload={
            "code": "def hello_world(): return 'Hello, World!'",
            "action": "analyze_and_test"
        },
        requirements=["code_generation", "testing"],
        timeout_seconds=60
    )
    
    task_id = await orchestrator.delegate_task(task, agent_id)
    
    # Verify everything is working
    assert agent_id in orchestrator.active_agents, "Agent should be active"
    assert task_id in orchestrator.task_executions, "Task should be tracked"
    
    agent = orchestrator.active_agents[agent_id]
    execution = orchestrator.task_executions[task_id]
    
    assert agent.current_task_id == task_id, "Agent should have current task"
    assert execution.agent_id == agent_id, "Execution should reference agent"
    assert execution.state == TaskExecutionState.RUNNING, "Task should be running"
    
    # Test health status with plugins
    health = await orchestrator.get_health_status()
    assert health["status"] == "healthy", "System should be healthy"
    assert health["active_agents"] == 1, "Should show 1 active agent"
    assert health["running_tasks"] == 1, "Should show 1 running task"
    
    await orchestrator.shutdown()
    
    print("✅ Full integration tests passed")
    return True

# ================================================================================
# Test Runner
# ================================================================================

async def run_all_tests():
    """Run complete test suite."""
    print("🧪 ORCHESTRATORV2 COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    test_results = []
    start_time = time.perf_counter()
    
    # Core orchestrator tests
    tests = [
        ("Orchestrator Initialization", test_orchestrator_initialization),
        ("Agent Spawning", test_agent_spawning), 
        ("Task Delegation", test_task_delegation),
        ("Plugin Dependency Resolution", test_plugin_dependency_resolution),
        ("Plugin Hooks", test_plugin_hooks),
        ("Plugin Performance Monitoring", test_plugin_performance_monitoring),
        ("Plugin State Sandboxing", test_plugin_state_sandboxing),
        ("Migration Metrics", test_migration_metrics),
        ("Migration Adapter", test_migration_adapter),
        ("Agent Spawn Performance", test_agent_spawn_performance),
        ("Task Delegation Performance", test_task_delegation_performance),
        ("Concurrent Operations", test_concurrent_operations),
        ("Full Integration", test_full_integration)
    ]
    
    # Run tests
    for test_name, test_func in tests:
        try:
            result = await test_func()
            test_results.append((test_name, result))
            
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            test_results.append((test_name, False))
    
    # Print summary
    total_time = (time.perf_counter() - start_time) * 1000
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    print(f"Total time: {total_time:.2f}ms")
    
    # Print detailed results
    print("\n📋 Detailed Results:")
    for test_name, result in test_results:
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! OrchestratorV2 is working correctly.")
        print("\n✅ Phase 0 POC Week 2 - OrchestratorV2 with Plugin Architecture - VALIDATED")
        return True
    else:
        print(f"\n⚠️ {total - passed} tests failed. Review the output above for details.")
        return False

def main():
    """Main test function."""
    try:
        print("🔧 Test Configuration:")
        print(f"   Performance test count: {TestConfig.PERFORMANCE_TEST_COUNT}")
        print(f"   Plugin timeout test: {TestConfig.PLUGIN_TIMEOUT_TEST_MS}ms")
        print(f"   Concurrent agent test: {TestConfig.CONCURRENT_AGENT_TEST_COUNT}")
        
        # Run tests
        success = asyncio.run(run_all_tests())
        
        if success:
            print("\n✅ OrchestratorV2 - Phase 0 POC Week 2 - COMPLETED")
            return 0
        else:
            print("\n❌ Some tests failed - review and fix before proceeding")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())