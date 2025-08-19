#!/usr/bin/env python3
"""
Demo script for Enhanced Simple Orchestrator

This script demonstrates the key features of the enhanced orchestrator:
1. Agent lifecycle management
2. Task delegation with different strategies
3. Error handling and recovery
4. Performance monitoring
5. System status reporting

Run this script to see the orchestrator in action.
"""

import asyncio
import json
from datetime import datetime
from app.core.simple_orchestrator_enhanced import (
    EnhancedSimpleOrchestrator,
    OrchestratorConfig,
    OrchestratorMode,
    AgentRole,
    TaskAssignmentStrategy,
    create_enhanced_orchestrator
)
from app.models.task import TaskPriority


async def demo_basic_operations():
    """Demonstrate basic orchestrator operations."""
    print("🚀 Starting Enhanced Simple Orchestrator Demo")
    print("=" * 60)
    
    # Create test configuration
    config = OrchestratorConfig(
        mode=OrchestratorMode.TEST,
        max_concurrent_agents=5,
        enable_database_persistence=False,  # Disable for demo
        enable_performance_monitoring=True,
        heartbeat_interval_seconds=10
    )
    
    # Create orchestrator
    orch = create_enhanced_orchestrator(config=config)
    
    try:
        # Start orchestrator
        print("🔧 Starting orchestrator...")
        await orch.start()
        print("✅ Orchestrator started successfully")
        
        # Test 1: Spawn agents with different roles
        print("\n📋 Test 1: Spawning agents with different roles")
        backend_dev = await orch.spawn_agent(
            role=AgentRole.BACKEND_DEVELOPER,
            capabilities=["python", "django", "api", "database"]
        )
        print(f"✅ Spawned backend developer: {backend_dev}")
        
        frontend_dev = await orch.spawn_agent(
            role=AgentRole.FRONTEND_DEVELOPER,
            capabilities=["javascript", "react", "ui", "css"]
        )
        print(f"✅ Spawned frontend developer: {frontend_dev}")
        
        devops_eng = await orch.spawn_agent(
            role=AgentRole.DEVOPS_ENGINEER,
            capabilities=["docker", "kubernetes", "ci/cd", "monitoring"]
        )
        print(f"✅ Spawned DevOps engineer: {devops_eng}")
        
        qa_eng = await orch.spawn_agent(
            role=AgentRole.QA_ENGINEER,
            capabilities=["testing", "automation", "selenium", "quality"]
        )
        print(f"✅ Spawned QA engineer: {qa_eng}")
        
        # Test 2: System status
        print("\n📊 Test 2: Getting system status")
        status = await orch.get_system_status()
        print(f"Active agents: {status['agents']['total']}")
        print(f"Agent health: {status['agents']['healthy']}/{status['agents']['total']}")
        print(f"Overall health: {status['health']['overall']}")
        print(f"Configuration mode: {status['configuration']['max_concurrent_agents']} max agents")
        
        # Test 3: Task delegation with different strategies
        print("\n🎯 Test 3: Task delegation with different strategies")
        
        # Strategy 1: Capability matching
        api_task = await orch.delegate_task(
            task_description="Implement user authentication API endpoints",
            task_type="backend_api",
            required_capabilities=["python", "api"],
            priority=TaskPriority.HIGH,
            assignment_strategy=TaskAssignmentStrategy.CAPABILITY_MATCH,
            estimated_duration=120
        )
        print(f"✅ API task assigned: {api_task}")
        
        # Strategy 2: Availability-based
        ui_task = await orch.delegate_task(
            task_description="Create responsive user dashboard",
            task_type="frontend_ui",
            required_capabilities=["react", "ui"],
            priority=TaskPriority.HIGH,
            assignment_strategy=TaskAssignmentStrategy.AVAILABILITY_BASED,
            estimated_duration=90
        )
        print(f"✅ UI task assigned: {ui_task}")
        
        # Strategy 3: Round-robin
        deployment_task = await orch.delegate_task(
            task_description="Setup CI/CD pipeline",
            task_type="deployment",
            required_capabilities=["ci/cd", "docker"],
            priority=TaskPriority.MEDIUM,
            assignment_strategy=TaskAssignmentStrategy.ROUND_ROBIN,
            estimated_duration=60
        )
        print(f"✅ Deployment task assigned: {deployment_task}")
        
        # Test 4: Check task assignments
        print("\n📝 Test 4: Task assignment details")
        for task_id, assignment in orch._task_assignments.items():
            agent = orch._agents[assignment.agent_id]
            print(f"Task {task_id[:8]}... -> Agent {agent.role.value} (score: {assignment.suitability_score:.2f})")
        
        # Test 5: Complete some tasks
        print("\n✅ Test 5: Completing tasks")
        
        # Complete API task
        await orch.complete_task(
            api_task,
            result={
                "endpoints": ["POST /auth/login", "POST /auth/register", "GET /auth/profile"],
                "status": "completed",
                "tests_passed": True
            }
        )
        print(f"✅ Completed API task: {api_task}")
        
        # Complete UI task
        await orch.complete_task(
            ui_task,
            result={
                "components": ["Dashboard", "UserProfile", "Navigation"],
                "responsive": True,
                "accessibility_score": 95
            }
        )
        print(f"✅ Completed UI task: {ui_task}")
        
        # Test 6: System status after task completion
        print("\n📊 Test 6: System status after task completion")
        status = await orch.get_system_status()
        print(f"Active tasks: {status['tasks']['active_assignments']}")
        print(f"Performance metrics:")
        print(f"  - Total operations: {status['performance']['operations_total']}")
        print(f"  - Success rate: {status['performance']['success_rate']:.1%}")
        print(f"  - Avg response time: {status['performance']['avg_response_time_ms']:.1f}ms")
        
        # Test 7: Agent performance metrics
        print("\n🏆 Test 7: Agent performance metrics")
        for agent_id, agent in orch._agents.items():
            metrics = agent.performance_metrics
            print(f"Agent {agent.role.value}:")
            print(f"  - Load score: {agent.load_score:.2f}")
            print(f"  - Completed tasks: {metrics.get('completed_tasks', 0)}")
            print(f"  - Total duration: {metrics.get('total_duration', 0)} minutes")
            print(f"  - Available: {agent.is_available_for_task()}")
        
        # Test 8: Error handling
        print("\n⚠️  Test 8: Testing error handling")
        
        try:
            # Try to spawn agent with duplicate ID
            await orch.spawn_agent(
                role=AgentRole.BACKEND_DEVELOPER,
                agent_id=backend_dev  # This should fail
            )
        except Exception as e:
            print(f"✅ Correctly caught duplicate agent error: {type(e).__name__}")
        
        try:
            # Try to delegate task with no available agents by maxing out all agents
            for i in range(10):  # Try to create more tasks than agents can handle
                await orch.delegate_task(f"Overload task {i}", "backend")
        except Exception as e:
            print(f"✅ Correctly caught task delegation error: {type(e).__name__}")
        
        # Test 9: Resource management
        print("\n🔧 Test 9: Resource management")
        print(f"Memory usage: {len(orch._agents)} agents, {len(orch._task_assignments)} tasks")
        print(f"Cache enabled: {orch.config.enable_caching}")
        print(f"Performance monitoring: {orch.config.enable_performance_monitoring}")
        
        # Test 10: Graceful shutdown of agents
        print("\n🛑 Test 10: Graceful agent shutdown")
        
        # Shutdown one agent gracefully
        result = await orch.shutdown_agent(devops_eng, graceful=True)
        print(f"✅ DevOps agent shutdown: {result}")
        
        # Check remaining agents
        status = await orch.get_system_status()
        print(f"Remaining active agents: {status['agents']['total']}")
        
        # Final status report
        print("\n📋 Final Status Report")
        print("=" * 40)
        final_status = await orch.get_system_status()
        print(json.dumps({
            "agents": {
                "total": final_status["agents"]["total"],
                "healthy": final_status["agents"]["healthy"],
                "by_role": final_status["agents"]["by_role"]
            },
            "tasks": final_status["tasks"],
            "performance": final_status["performance"],
            "health": final_status["health"]
        }, indent=2))
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Always shutdown orchestrator
        print("\n🛑 Shutting down orchestrator...")
        await orch.shutdown()
        print("✅ Orchestrator shutdown complete")


async def demo_advanced_features():
    """Demonstrate advanced orchestrator features."""
    print("\n🔬 Advanced Features Demo")
    print("=" * 40)
    
    config = OrchestratorConfig(
        mode=OrchestratorMode.TEST,
        max_concurrent_agents=3,
        enable_database_persistence=False,
        default_task_assignment_strategy=TaskAssignmentStrategy.PERFORMANCE_BASED
    )
    
    orch = create_enhanced_orchestrator(config=config)
    
    try:
        await orch.start()
        
        # Create agents with different performance profiles
        print("📈 Creating agents with performance tracking...")
        
        fast_agent = await orch.spawn_agent(
            role=AgentRole.BACKEND_DEVELOPER,
            capabilities=["python", "fast_execution"]
        )
        
        # Simulate some completed tasks for performance history
        agent = orch._agents[fast_agent]
        agent.performance_metrics = {
            "completed_tasks": 10,
            "total_duration": 300,  # 30 min average
            "success_rate": 0.95
        }
        
        slow_agent = await orch.spawn_agent(
            role=AgentRole.BACKEND_DEVELOPER,
            capabilities=["python", "detailed_analysis"]
        )
        
        # Simulate different performance profile
        agent = orch._agents[slow_agent]
        agent.performance_metrics = {
            "completed_tasks": 5,
            "total_duration": 600,  # 120 min average
            "success_rate": 0.98
        }
        
        # Test performance-based assignment
        print("🎯 Testing performance-based task assignment...")
        
        task1 = await orch.delegate_task(
            "Quick API fix",
            "backend_fix",
            assignment_strategy=TaskAssignmentStrategy.PERFORMANCE_BASED
        )
        
        task2 = await orch.delegate_task(
            "Complex algorithm implementation",
            "backend_algorithm",
            assignment_strategy=TaskAssignmentStrategy.PERFORMANCE_BASED
        )
        
        # Show assignments
        for task_id, assignment in orch._task_assignments.items():
            agent = orch._agents[assignment.agent_id]
            metrics = agent.performance_metrics
            avg_duration = metrics.get("total_duration", 0) / max(metrics.get("completed_tasks", 1), 1)
            print(f"Task {task_id[:8]}... assigned to agent with {avg_duration:.0f}min avg duration")
        
        # Test load balancing
        print("⚖️  Testing load balancing...")
        
        # Assign multiple tasks to see load distribution
        task_ids = []
        for i in range(4):
            task_id = await orch.delegate_task(
                f"Load test task {i+1}",
                "backend",
                assignment_strategy=TaskAssignmentStrategy.AVAILABILITY_BASED
            )
            task_ids.append(task_id)
        
        # Show load distribution
        for agent_id, agent in orch._agents.items():
            task_count = sum(1 for a in orch._task_assignments.values() if a.agent_id == agent_id)
            print(f"Agent {agent.role.value}: {task_count} tasks, load score: {agent.load_score:.2f}")
        
        # Complete some tasks to test load rebalancing
        await orch.complete_task(task_ids[0])
        await orch.complete_task(task_ids[1])
        
        print("📊 Load after task completion:")
        for agent_id, agent in orch._agents.items():
            task_count = sum(1 for a in orch._task_assignments.values() 
                           if a.agent_id == agent_id and a.status.value != "completed")
            print(f"Agent {agent.role.value}: {task_count} active tasks, load score: {agent.load_score:.2f}")
        
    finally:
        await orch.shutdown()


if __name__ == "__main__":
    print("🎭 Enhanced Simple Orchestrator Demo")
    print("=====================================")
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run basic demo
    asyncio.run(demo_basic_operations())
    
    # Run advanced demo
    asyncio.run(demo_advanced_features())
    
    print(f"\n🎉 Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Thank you for trying the Enhanced Simple Orchestrator!")