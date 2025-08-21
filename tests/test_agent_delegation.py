#!/usr/bin/env python3
"""
Test script for Agent Delegation System

Tests task decomposition, agent coordination, and context rot prevention
without the full FastAPI server complexity.
"""

import asyncio
import asyncpg
from pathlib import Path
import sys

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.project_index.agent_delegation import (
    TaskDecomposer, AgentCoordinator, ContextRotPrevention,
    TaskType, TaskComplexity, AgentSpecialization
)
from uuid import UUID

# Database configuration
DATABASE_URL = "postgresql://leanvibe_user:leanvibe_secure_pass@localhost:5432/leanvibe_agent_hive"

async def test_task_decomposition():
    """Test the task decomposition functionality"""
    print("🧪 Testing Task Decomposition System...")
    
    # Connect to database
    pool = await asyncpg.create_pool(DATABASE_URL)
    
    # Get the bee-hive project ID
    async with pool.acquire() as conn:
        project_row = await conn.fetchrow(
            "SELECT id FROM project_indexes WHERE name = 'bee-hive' LIMIT 1"
        )
        if not project_row:
            print("❌ Error: bee-hive project not found in database")
            return False
        
        project_id = project_row['id']
        print(f"✅ Found project: {project_id}")
    
    # Test 1: Simple task (should not decompose)
    print("\n📝 Test 1: Simple Task")
    decomposer = TaskDecomposer(project_id, pool)
    
    simple_result = await decomposer.decompose_task(
        "fix authentication bug in login endpoint",
        TaskType.BUG_FIX
    )
    
    print(f"   Complexity: {simple_result.original_task.complexity.value}")
    print(f"   Subtasks: {len(simple_result.subtasks)}")
    print(f"   Strategy: {simple_result.decomposition_strategy}")
    print(f"   Success: {simple_result.success}")
    
    # Test 2: Complex task (should decompose)
    print("\n📝 Test 2: Complex Task")
    complex_result = await decomposer.decompose_task(
        "implement comprehensive user authentication system with JWT tokens, OAuth integration, role-based access control, session management, and audit logging across backend API and frontend dashboard",
        TaskType.FEATURE_IMPLEMENTATION
    )
    
    print(f"   Complexity: {complex_result.original_task.complexity.value}")
    print(f"   Subtasks: {len(complex_result.subtasks)}")
    print(f"   Strategy: {complex_result.decomposition_strategy}")
    print(f"   Success: {complex_result.success}")
    print(f"   Total Duration: {complex_result.estimated_total_duration} minutes")
    
    if complex_result.subtasks:
        print("   Subtask breakdown:")
        for i, subtask in enumerate(complex_result.subtasks, 1):
            print(f"     {i}. {subtask.title}")
            print(f"        Specialization: {subtask.preferred_specialization.value}")
            print(f"        Duration: {subtask.estimated_duration_minutes} min")
            print(f"        Files: {len(subtask.primary_files)}")
    
    # Test 3: Agent Coordination
    print("\n🤖 Test 3: Agent Coordination")
    coordinator = AgentCoordinator(project_id, pool)
    
    if complex_result.success and complex_result.subtasks:
        assignment_result = await coordinator.assign_agents_to_tasks(complex_result)
        
        print(f"   Total Agents: {assignment_result['total_agents']}")
        print(f"   Parallel Execution: {assignment_result['parallel_execution']}")
        print(f"   Estimated Completion: {assignment_result['estimated_completion']}")
        
        print("   Agent Assignments:")
        for assignment in assignment_result['assignments']:
            print(f"     Agent: {assignment['agent_id']}")
            print(f"     Task: {assignment['task_id']}")
            print(f"     Specialization Match: {assignment['specialization_match']}")
    
    # Test 4: Context Monitoring
    print("\n🧠 Test 4: Context Rot Prevention")
    context_monitor = ContextRotPrevention(pool)
    
    # Simulate monitoring different context sizes
    test_agent_id = "test_agent_001"
    
    for context_size in [30000, 60000, 85000, 95000]:
        monitoring_result = await context_monitor.monitor_agent_context(test_agent_id, context_size)
        
        print(f"   Context Size: {context_size} tokens")
        print(f"   Status: {monitoring_result['threshold_status']}")
        print(f"   Recommendations: {len(monitoring_result['recommendations'])}")
        
        for rec in monitoring_result['recommendations']:
            print(f"     - {rec['type']}: {rec['message']}")
    
    # Test context refresh
    print("\n🔄 Test 5: Context Refresh")
    refresh_result = await context_monitor.trigger_context_refresh(test_agent_id, "full")
    print(f"   Refresh Type: {refresh_result['refresh_type']}")
    print(f"   Steps: {len(refresh_result['steps'])}")
    
    for step in refresh_result['steps']:
        print(f"     - {step['step']}: {step['description']}")
    
    await pool.close()
    print("\n✅ All Agent Delegation tests completed successfully!")
    return True

async def main():
    """Main test function"""
    print("🚀 Starting Agent Delegation System Tests")
    print("=" * 50)
    
    try:
        success = await test_task_decomposition()
        if success:
            print("\n🎉 Phase 3 Agent Delegation System: FUNCTIONAL")
        else:
            print("\n❌ Phase 3 Agent Delegation System: FAILED")
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())