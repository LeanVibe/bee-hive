#!/usr/bin/env python3
"""
Simple validation script for Enhanced Simple Orchestrator

This script quickly validates that all core functionality works:
1. Agent spawning
2. Task delegation 
3. Task completion
4. System status
5. Agent shutdown

Returns 0 for success, 1 for failure.
"""

import asyncio
import sys
from app.core.simple_orchestrator_enhanced import (
    EnhancedSimpleOrchestrator,
    OrchestratorConfig,
    OrchestratorMode,
    AgentRole,
    TaskAssignmentStrategy
)
from app.models.task import TaskPriority


async def validate_orchestrator():
    """Validate core orchestrator functionality."""
    
    print("🔍 Validating Enhanced Simple Orchestrator...")
    
    # Create test configuration
    config = OrchestratorConfig(
        mode=OrchestratorMode.TEST,
        max_concurrent_agents=3,
        enable_database_persistence=False,
        enable_performance_monitoring=False,  # Disable for faster testing
        heartbeat_interval_seconds=60  # Longer interval to avoid issues
    )
    
    # Create orchestrator
    orch = EnhancedSimpleOrchestrator(config=config)
    
    try:
        # Test 1: Start orchestrator
        print("1. Starting orchestrator...", end=" ")
        await orch.start()
        print("✅")
        
        # Test 2: Spawn agent
        print("2. Spawning agent...", end=" ")
        agent_id = await orch.spawn_agent(
            role=AgentRole.BACKEND_DEVELOPER,
            capabilities=["python", "testing"]
        )
        assert agent_id is not None
        assert agent_id in orch._agents
        print("✅")
        
        # Test 3: Get system status
        print("3. Getting system status...", end=" ")
        status = await orch.get_system_status()
        assert status["agents"]["total"] == 1
        assert status["agents"]["healthy"] == 1
        print("✅")
        
        # Test 4: Delegate task
        print("4. Delegating task...", end=" ")
        task_id = await orch.delegate_task(
            task_description="Test validation task",
            task_type="validation",
            priority=TaskPriority.HIGH
        )
        assert task_id is not None
        assert task_id in orch._task_assignments
        print("✅")
        
        # Test 5: Complete task
        print("5. Completing task...", end=" ")
        result = await orch.complete_task(
            task_id,
            result={"status": "validation_passed"}
        )
        assert result is True
        assignment = orch._task_assignments[task_id]
        assert assignment.status.value == "completed"
        print("✅")
        
        # Test 6: Agent availability after task completion
        print("6. Checking agent availability...", end=" ")
        agent = orch._agents[agent_id]
        assert agent.current_task_id is None
        assert agent.is_available_for_task() is True
        print("✅")
        
        # Test 7: Shutdown agent
        print("7. Shutting down agent...", end=" ")
        result = await orch.shutdown_agent(agent_id, graceful=False)  # Force shutdown
        assert result is True
        assert agent_id not in orch._agents
        print("✅")
        
        # Test 8: Final status check
        print("8. Final status check...", end=" ")
        status = await orch.get_system_status()
        assert status["agents"]["total"] == 0
        print("✅")
        
        print("\n🎉 All validations passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Always shutdown orchestrator
        try:
            await orch.shutdown()
        except Exception as e:
            print(f"Warning: Error during shutdown: {e}")


async def main():
    """Main function."""
    success = await validate_orchestrator()
    
    if success:
        print("\n✅ Enhanced Simple Orchestrator validation PASSED")
        sys.exit(0)
    else:
        print("\n❌ Enhanced Simple Orchestrator validation FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())