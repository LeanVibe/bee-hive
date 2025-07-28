#!/usr/bin/env python3
"""
Simple validation test for Vertical Slice 1.1 components.

This script tests the core functionality without complex dependencies
to ensure the implementation is working correctly.
"""

import asyncio
import uuid
from datetime import datetime

# Test the basic components
from app.core.agent_lifecycle_manager import AgentLifecycleManager, LifecycleEventType
from app.core.task_execution_engine import TaskExecutionEngine, ExecutionOutcome
from app.core.agent_messaging_service import AgentMessagingService, MessageType
from app.core.agent_lifecycle_hooks import AgentLifecycleHooks, SecurityAction
from app.models.agent import AgentType, AgentStatus
from app.models.task import TaskType, TaskPriority


async def test_basic_functionality():
    """Test basic functionality of all components."""
    print("🧪 Testing Vertical Slice 1.1 Components")
    print("=" * 50)
    
    # Test 1: Agent Lifecycle Manager
    print("\n1. Testing Agent Lifecycle Manager...")
    try:
        lifecycle_manager = AgentLifecycleManager()
        print("   ✅ AgentLifecycleManager created successfully")
        
        # Test system metrics
        metrics = await lifecycle_manager.get_system_metrics()
        print(f"   ✅ System metrics retrieved: {len(metrics)} entries")
        
    except Exception as e:
        print(f"   ❌ AgentLifecycleManager test failed: {e}")
    
    # Test 2: Task Execution Engine
    print("\n2. Testing Task Execution Engine...")
    try:
        execution_engine = TaskExecutionEngine()
        print("   ✅ TaskExecutionEngine created successfully")
        
        # Test performance metrics
        perf_metrics = await execution_engine.get_performance_metrics()
        print(f"   ✅ Performance metrics retrieved: {len(perf_metrics)} entries")
        
    except Exception as e:
        print(f"   ❌ TaskExecutionEngine test failed: {e}")
    
    # Test 3: Agent Messaging Service
    print("\n3. Testing Agent Messaging Service...")
    try:
        messaging_service = AgentMessagingService()
        print("   ✅ AgentMessagingService created successfully")
        
        # Test messaging metrics
        msg_metrics = await messaging_service.get_messaging_metrics()
        print(f"   ✅ Messaging metrics retrieved: {len(msg_metrics)} entries")
        
    except Exception as e:
        print(f"   ❌ AgentMessagingService test failed: {e}")
    
    # Test 4: Agent Lifecycle Hooks
    print("\n4. Testing Agent Lifecycle Hooks...")
    try:
        lifecycle_hooks = AgentLifecycleHooks()
        print("   ✅ AgentLifecycleHooks created successfully")
        
        # Test hook metrics
        hook_metrics = await lifecycle_hooks.get_hook_metrics()
        print(f"   ✅ Hook metrics retrieved: {len(hook_metrics)} entries")
        
    except Exception as e:
        print(f"   ❌ AgentLifecycleHooks test failed: {e}")
    
    # Test 5: Data Models
    print("\n5. Testing Data Models...")
    try:
        # Test enum values
        assert AgentType.CLAUDE.value == "claude"
        assert AgentStatus.ACTIVE.value == "active"
        assert TaskType.FEATURE_DEVELOPMENT.value == "feature_development"
        assert TaskPriority.HIGH.value == 8
        
        print("   ✅ All data models working correctly")
        
    except Exception as e:
        print(f"   ❌ Data models test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Basic component validation completed!")
    print("\n💡 Next steps:")
    print("   - Run full integration tests with database")
    print("   - Execute performance benchmarks")
    print("   - Test API endpoints")


if __name__ == "__main__":
    asyncio.run(test_basic_functionality())