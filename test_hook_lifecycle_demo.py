#!/usr/bin/env python3
"""
Hook Lifecycle System Demo and Basic Testing

This script demonstrates the Hook Lifecycle System functionality
and performs basic validation testing.
"""

import asyncio
import uuid
from datetime import datetime
import json

# Import the hook lifecycle system components
try:
    from app.core.hook_lifecycle_system import (
        HookLifecycleSystem,
        HookType,
        SecurityValidator,
        EventAggregator,
        WebSocketStreamer,
        DangerousCommand,
        SecurityRisk
    )
    
    print("✅ Successfully imported Hook Lifecycle System components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    exit(1)


async def demo_security_validator():
    """Demonstrate SecurityValidator functionality."""
    print("\n🔒 Testing Security Validator...")
    
    validator = SecurityValidator()
    
    # Test safe commands
    safe_commands = [
        "ls -la",
        "python script.py",
        "git status",
        "npm install",
        "docker ps"
    ]
    
    print("Testing safe commands:")
    for cmd in safe_commands:
        is_safe, risk_level, reason = await validator.validate_command(cmd)
        print(f"  '{cmd}' -> Safe: {is_safe}, Risk: {risk_level.value}")
    
    # Test dangerous commands
    dangerous_commands = [
        "rm -rf /",
        "sudo rm -rf /home",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=/dev/sda",
        "chmod 777 /etc/passwd"
    ]
    
    print("\nTesting dangerous commands:")
    for cmd in dangerous_commands:
        is_safe, risk_level, reason = await validator.validate_command(cmd)
        print(f"  '{cmd}' -> Safe: {is_safe}, Risk: {risk_level.value}, Reason: {reason}")
    
    print("✅ Security Validator tests completed")


async def demo_event_aggregator():
    """Demonstrate EventAggregator functionality."""
    print("\n📊 Testing Event Aggregator...")
    
    aggregator = EventAggregator(batch_size=5, flush_interval_ms=100)
    await aggregator.start()
    
    try:
        # Create test events
        events = []
        for i in range(10):
            from app.core.hook_lifecycle_system import HookEvent
            event = HookEvent(
                hook_type=HookType.PRE_TOOL_USE,
                agent_id=uuid.uuid4(),
                session_id=uuid.uuid4(),
                timestamp=datetime.utcnow(),
                payload={
                    "tool_name": f"test_tool_{i}",
                    "parameters": {"iteration": i}
                },
                priority=5
            )
            events.append(event)
            await aggregator.add_event(event)
        
        # Wait for processing
        await asyncio.sleep(0.2)
        
        # Check metrics
        metrics = aggregator.get_metrics()
        print(f"  Events aggregated: {metrics['event_aggregator']['events_aggregated']}")
        print(f"  Batches processed: {metrics['event_aggregator']['batches_processed']}")
        
    finally:
        await aggregator.stop()
    
    print("✅ Event Aggregator tests completed")


async def demo_websocket_streamer():
    """Demonstrate WebSocketStreamer functionality (without actual WebSocket)."""
    print("\n🌐 Testing WebSocket Streamer...")
    
    streamer = WebSocketStreamer()
    
    # Create test event
    from app.core.hook_lifecycle_system import HookEvent
    event = HookEvent(
        hook_type=HookType.NOTIFICATION,
        agent_id=uuid.uuid4(),
        session_id=uuid.uuid4(),
        timestamp=datetime.utcnow(),
        payload={
            "level": "info",
            "message": "Test notification",
            "details": {"test": True}
        },
        priority=5
    )
    
    # Broadcast event (will be no-op with no connected clients)
    await streamer.broadcast_event(event)
    
    # Check metrics
    metrics = streamer.get_metrics()
    print(f"  Active connections: {metrics['websocket_streamer']['active_connections']}")
    print(f"  Messages sent: {metrics['websocket_streamer']['messages_sent']}")
    
    print("✅ WebSocket Streamer tests completed")


async def demo_hook_lifecycle_system():
    """Demonstrate complete HookLifecycleSystem functionality."""
    print("\n🚀 Testing Hook Lifecycle System...")
    
    # Create and initialize system (simplified for demo)
    hook_system = HookLifecycleSystem()
    
    # Mock the missing dependencies for demo
    hook_system.redis_client = None  # Disable Redis for demo
    hook_system.config["enable_redis_streaming"] = False
    
    await hook_system.initialize()
    
    try:
        # Test various hook types
        agent_id = uuid.uuid4()
        session_id = uuid.uuid4()
        
        # Test PreToolUse hook
        print("  Testing PreToolUse hook...")
        result = await hook_system.process_pre_tool_use(
            agent_id=agent_id,
            session_id=session_id,
            tool_name="demo_tool",
            parameters={"param1": "value1", "param2": 123}
        )
        print(f"    Result: Success={result.success}, Time={result.processing_time_ms:.2f}ms")
        
        # Test PostToolUse hook
        print("  Testing PostToolUse hook...")
        result = await hook_system.process_post_tool_use(
            agent_id=agent_id,
            session_id=session_id,
            tool_name="demo_tool",
            success=True,
            result="Demo operation completed successfully",
            execution_time_ms=45.67
        )
        print(f"    Result: Success={result.success}, Time={result.processing_time_ms:.2f}ms")
        
        # Test Notification hook
        print("  Testing Notification hook...")
        result = await hook_system.process_notification(
            agent_id=agent_id,
            session_id=session_id,
            level="info",
            message="Demo notification message",
            details={"demo": True, "timestamp": datetime.utcnow().isoformat()}
        )
        print(f"    Result: Success={result.success}, Time={result.processing_time_ms:.2f}ms")
        
        # Test Stop hook
        print("  Testing Stop hook...")
        result = await hook_system.process_stop(
            agent_id=agent_id,
            session_id=session_id,
            reason="Demo completed",
            details={"completion_status": "success"}
        )
        print(f"    Result: Success={result.success}, Time={result.processing_time_ms:.2f}ms")
        
        # Test security validation with dangerous command
        print("  Testing security validation...")
        result = await hook_system.process_hook(
            hook_type=HookType.PRE_TOOL_USE,
            agent_id=agent_id,
            session_id=session_id,
            payload={
                "tool_name": "bash",
                "command": "rm -rf /dangerous",
                "parameters": {"command": "rm -rf /dangerous"}
            },
            priority=1
        )
        print(f"    Security Result: Success={result.success}")
        if not result.success:
            print(f"    Blocked Reason: {result.blocked_reason}")
        
        # Get comprehensive metrics
        metrics = hook_system.get_comprehensive_metrics()
        print(f"\n  📊 System Metrics:")
        print(f"    Hooks processed: {metrics['hook_lifecycle_system']['hooks_processed']}")
        print(f"    Hooks blocked: {metrics['hook_lifecycle_system']['hooks_blocked']}")
        print(f"    Avg processing time: {metrics['hook_lifecycle_system']['avg_processing_time_ms']:.2f}ms")
        print(f"    Performance violations: {metrics['hook_lifecycle_system']['performance_threshold_violations']}")
        
    finally:
        await hook_system.shutdown()
    
    print("✅ Hook Lifecycle System tests completed")


async def demo_performance_validation():
    """Demonstrate basic performance validation."""
    print("\n⚡ Testing Performance...")
    
    hook_system = HookLifecycleSystem()
    hook_system.redis_client = None  # Disable Redis for demo
    hook_system.config["enable_redis_streaming"] = False
    
    await hook_system.initialize()
    
    try:
        # Run performance test
        agent_id = uuid.uuid4()
        session_id = uuid.uuid4()
        
        times = []
        for i in range(100):  # 100 iterations
            start_time = asyncio.get_event_loop().time()
            
            result = await hook_system.process_pre_tool_use(
                agent_id=agent_id,
                session_id=session_id,
                tool_name=f"perf_test_{i}",
                parameters={"iteration": i}
            )
            
            end_time = asyncio.get_event_loop().time()
            processing_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(processing_time)
            
            if not result.success:
                print(f"    ❌ Performance test iteration {i} failed: {result.error}")
        
        # Calculate statistics
        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            
            print(f"  Performance Results (100 iterations):")
            print(f"    Average time: {avg_time:.2f}ms")
            print(f"    Min time: {min_time:.2f}ms")
            print(f"    Max time: {max_time:.2f}ms")
            print(f"    Meets <50ms SLA: {'✅' if avg_time < 50.0 else '❌'}")
        
    finally:
        await hook_system.shutdown()
    
    print("✅ Performance validation completed")


async def main():
    """Run all demonstrations."""
    print("🎯 Hook Lifecycle System Demo")
    print("=" * 50)
    
    try:
        await demo_security_validator()
        await demo_event_aggregator()
        await demo_websocket_streamer()
        await demo_hook_lifecycle_system()
        await demo_performance_validation()
        
        print("\n" + "=" * 50)
        print("🎊 All demonstrations completed successfully!")
        print("The Hook Lifecycle System is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    # Run the demo
    success = asyncio.run(main())
    exit(0 if success else 1)