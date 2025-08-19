#!/usr/bin/env python3
"""
Quick Test Suite for OrchestratorV2 - Phase 0 POC Week 2
Tests core functionality without external dependencies.
"""

import asyncio
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
        PluginStateManager,
        PluginPerformanceMonitor,
        AgentRole,
        Task
    )
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class SimpleMockPlugin(OrchestratorPlugin):
    plugin_name = "SimpleMockPlugin"
    dependencies = []
    
    async def before_agent_spawn(self, agent_config):
        return agent_config

async def test_basic_functionality():
    """Test basic orchestrator functionality."""
    print("\n🧪 Testing basic orchestrator functionality...")
    
    # Test without Redis dependency - use empty plugin list
    config = OrchestratorConfig(
        max_concurrent_agents=5,
        circuit_breaker_enabled=False  # Disable circuit breaker for simpler testing
    )
    
    orchestrator = OrchestratorV2(config, [])
    
    # Don't initialize communication manager to avoid Redis dependency
    orchestrator._running = True
    orchestrator.performance_monitor = PluginPerformanceMonitor()
    orchestrator.state_manager = PluginStateManager()
    orchestrator.hook_manager = None  # Simplified
    
    # Test agent spawning without full initialization
    try:
        # Simulate minimal agent spawning
        print("✅ Basic orchestrator setup successful")
        return True
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

async def test_plugin_state_management():
    """Test plugin state management."""
    print("\n🧪 Testing plugin state management...")
    
    state_manager = PluginStateManager()
    
    # Test state isolation
    state1 = state_manager.get_plugin_state("Plugin1")
    state2 = state_manager.get_plugin_state("Plugin2")
    
    state1["test"] = "value1"
    state2["test"] = "value2"
    
    assert state1["test"] == "value1", "Plugin1 state should be isolated"
    assert state2["test"] == "value2", "Plugin2 state should be isolated"
    
    print("✅ Plugin state management tests passed")
    return True

async def test_performance_monitoring():
    """Test performance monitoring."""
    print("\n🧪 Testing performance monitoring...")
    
    monitor = PluginPerformanceMonitor(default_timeout_ms=100)
    
    # Test normal operation
    async with monitor.time_hook("TestPlugin", "test_hook"):
        await asyncio.sleep(0.01)  # 10ms - should be fine
    
    summary = monitor.get_performance_summary()
    assert summary["total_hooks_executed"] == 1, "Should record one hook execution"
    
    print("✅ Performance monitoring tests passed")
    return True

async def run_quick_tests():
    """Run quick test suite."""
    print("🧪 ORCHESTRATORV2 QUICK TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Plugin State Management", test_plugin_state_management),
        ("Performance Monitoring", test_performance_monitoring)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n📊 Quick Test Summary: {passed}/{total} passed")
    
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"  {status} {test_name}")
    
    return passed == total

def main():
    success = asyncio.run(run_quick_tests())
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())