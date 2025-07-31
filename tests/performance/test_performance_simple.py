#!/usr/bin/env python3
"""
Simplified Performance Validation for Custom Commands System
Phase 6.1 - LeanVibe Agent Hive 2.0

Tests core component performance without requiring full Redis infrastructure.
"""

import asyncio
import time
import psutil
import uuid
from datetime import datetime
from typing import List, Dict, Any
import statistics

# Import our custom commands system
from app.core.command_registry import CommandRegistry
from app.core.task_distributor import TaskDistributor, DistributionStrategy
from app.core.agent_registry import AgentRegistry
from app.schemas.custom_commands import (
    CommandDefinition, CommandExecutionRequest, AgentRequirement, 
    WorkflowStep, SecurityPolicy, AgentRole
)

class SimplePerformanceValidator:
    """Simplified performance validation for core components."""
    
    def __init__(self):
        self.results = {}
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
    async def setup_system(self):
        """Setup system components for testing."""
        print("🚀 Setting up system components...")
        
        # Create system components (without Redis dependencies)
        self.agent_registry = AgentRegistry()
        self.command_registry = CommandRegistry(agent_registry=self.agent_registry)
        self.task_distributor = TaskDistributor(agent_registry=self.agent_registry)
        
        print("✅ System components initialized")
        
    def create_test_command(self, complexity_level: str = "simple") -> CommandDefinition:
        """Create test command definition."""
        if complexity_level == "simple":
            workflow = [
                WorkflowStep(
                    step="generate_content",
                    task="Generate content using AI",
                    agent=AgentRole.TECHNICAL_WRITER,
                    timeout_minutes=5
                ),
                WorkflowStep(
                    step="review_content", 
                    task="Review generated content",
                    agent=AgentRole.QA_TEST_GUARDIAN,
                    depends_on=["generate_content"],
                    timeout_minutes=3
                )
            ]
            agents = [
                AgentRequirement(
                    role=AgentRole.TECHNICAL_WRITER,
                    required_capabilities=["text_generation"]
                ),
                AgentRequirement(
                    role=AgentRole.QA_TEST_GUARDIAN, 
                    required_capabilities=["content_review"]
                )
            ]
        else:  # complex
            workflow = [
                WorkflowStep(
                    step="analyze_requirements",
                    task="Analyze project requirements",
                    agent=AgentRole.DATA_ANALYST,
                    timeout_minutes=10
                ),
                WorkflowStep(
                    step="design_architecture", 
                    task="Design system architecture",
                    agent=AgentRole.BACKEND_ENGINEER,
                    depends_on=["analyze_requirements"],
                    timeout_minutes=15
                ),
                WorkflowStep(
                    step="implement_backend",
                    task="Implement backend services",
                    agent=AgentRole.BACKEND_ENGINEER,
                    depends_on=["design_architecture"],
                    timeout_minutes=30
                ),
                WorkflowStep(
                    step="implement_frontend",
                    task="Implement frontend interface", 
                    agent=AgentRole.FRONTEND_BUILDER,
                    depends_on=["design_architecture"],
                    timeout_minutes=25
                ),
                WorkflowStep(
                    step="integration_testing",
                    task="Perform integration testing",
                    agent=AgentRole.QA_TEST_GUARDIAN,
                    depends_on=["implement_backend", "implement_frontend"],
                    timeout_minutes=20
                )
            ]
            agents = [
                AgentRequirement(role=AgentRole.DATA_ANALYST, required_capabilities=["analysis"]),
                AgentRequirement(role=AgentRole.BACKEND_ENGINEER, required_capabilities=["system_design"]),
                AgentRequirement(role=AgentRole.BACKEND_ENGINEER, required_capabilities=["backend_development"]),
                AgentRequirement(role=AgentRole.FRONTEND_BUILDER, required_capabilities=["frontend_development"]),
                AgentRequirement(role=AgentRole.QA_TEST_GUARDIAN, required_capabilities=["testing"])
            ]
        
        return CommandDefinition(
            name=f"test_command_{complexity_level}_{uuid.uuid4().hex[:8]}",
            version="1.0.0",
            description=f"Test command for {complexity_level} performance validation",
            category="testing",
            tags=["performance", "validation", complexity_level],
            agents=agents,
            workflow=workflow,
            security_policy=SecurityPolicy(
                requires_approval=False,
                network_access=False,
                allowed_operations=["read", "write", "compute"],
                restricted_paths=[],
                resource_limits={"max_memory_mb": 512, "max_cpu_time_seconds": 1800}
            ),
            failure_strategy="continue_on_failure"
        )
    
    async def test_command_parsing_performance(self) -> Dict[str, float]:
        """Test Target: Command parsing <100ms"""
        print("\n📊 Testing command parsing performance...")
        
        parse_times = []
        
        for i in range(50):  # Test with 50 commands
            command_def = self.create_test_command("simple")
            
            start_time = time.perf_counter()
            
            # Register command (includes parsing and validation)
            success, validation_result = await self.command_registry.register_command(
                command_def, 
                author_id="performance_test",
                validate_agents=False,  # Skip agent validation for pure parsing test
                dry_run=True  # Only validate, don't store
            )
            
            end_time = time.perf_counter()
            parse_time_ms = (end_time - start_time) * 1000
            parse_times.append(parse_time_ms)
            
            if not success:
                print(f"❌ Command parsing failed: {validation_result.errors}")
        
        avg_parse_time = statistics.mean(parse_times)
        max_parse_time = max(parse_times)
        p95_parse_time = statistics.quantiles(parse_times, n=20)[18]  # 95th percentile
        
        results = {
            "average_ms": avg_parse_time,
            "max_ms": max_parse_time,
            "p95_ms": p95_parse_time,
            "target_ms": 100,
            "passed": avg_parse_time < 100 and p95_parse_time < 100
        }
        
        print(f"  Average: {avg_parse_time:.2f}ms")
        print(f"  Max: {max_parse_time:.2f}ms") 
        print(f"  95th percentile: {p95_parse_time:.2f}ms")
        print(f"  Target: <100ms")
        print(f"  Result: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        
        return results
    
    async def test_task_distribution_performance(self) -> Dict[str, float]:
        """Test Target: Agent task distribution <200ms per agent"""
        print("\n📊 Testing task distribution performance...")
        
        distribution_times_per_agent = []
        
        for strategy in [DistributionStrategy.ROUND_ROBIN, DistributionStrategy.LEAST_LOADED, 
                        DistributionStrategy.CAPABILITY_MATCH]:
            
            for complexity in ["simple", "complex"]:
                command_def = self.create_test_command(complexity)
                agent_count = len(command_def.agents)
                
                start_time = time.perf_counter()
                
                try:
                    distribution_result = await self.task_distributor.distribute_tasks(
                        workflow_steps=command_def.workflow,
                        agent_requirements=command_def.agents,
                        execution_context={"test": "performance"},
                        strategy_override=strategy
                    )
                    
                    end_time = time.perf_counter()
                    total_time_ms = (end_time - start_time) * 1000
                    time_per_agent_ms = total_time_ms / max(agent_count, 1)
                    distribution_times_per_agent.append(time_per_agent_ms)
                    
                    print(f"    {strategy} ({complexity}): {time_per_agent_ms:.2f}ms per agent")
                    
                except Exception as e:
                    print(f"❌ Task distribution failed for {strategy} ({complexity}): {e}")
                    continue
        
        if not distribution_times_per_agent:
            return {"passed": False, "error": "No successful distributions"}
        
        avg_time_per_agent = statistics.mean(distribution_times_per_agent)
        max_time_per_agent = max(distribution_times_per_agent)
        
        results = {
            "average_ms_per_agent": avg_time_per_agent,
            "max_ms_per_agent": max_time_per_agent,
            "target_ms_per_agent": 200,
            "passed": avg_time_per_agent < 200 and max_time_per_agent < 200
        }
        
        print(f"  Average per agent: {avg_time_per_agent:.2f}ms")
        print(f"  Max per agent: {max_time_per_agent:.2f}ms")
        print(f"  Target: <200ms per agent")
        print(f"  Result: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        
        return results
    
    async def test_command_validation_performance(self) -> Dict[str, float]:
        """Test comprehensive command validation performance"""
        print("\n📊 Testing command validation performance...")
        
        validation_times = []
        
        # Test various command complexities
        for complexity in ["simple", "complex"]:
            for i in range(10):
                command_def = self.create_test_command(complexity)
                
                start_time = time.perf_counter()
                
                try:
                    validation_result = await self.command_registry.validate_command(
                        command_def, 
                        validate_agents=False  # Skip agent validation for consistent timing
                    )
                    
                    end_time = time.perf_counter()
                    validation_time_ms = (end_time - start_time) * 1000
                    validation_times.append(validation_time_ms)
                    
                    if not validation_result.is_valid:
                        print(f"❌ Validation failed: {validation_result.errors}")
                    
                except Exception as e:
                    print(f"❌ Validation error: {e}")
                    continue
        
        if not validation_times:
            return {"passed": False, "error": "No successful validations"}
        
        avg_validation_time = statistics.mean(validation_times)
        max_validation_time = max(validation_times)
        
        results = {
            "average_ms": avg_validation_time,
            "max_ms": max_validation_time,
            "target_ms": 50,  # Internal target for validation
            "passed": avg_validation_time < 50
        }
        
        print(f"  Average: {avg_validation_time:.2f}ms")
        print(f"  Max: {max_validation_time:.2f}ms")
        print(f"  Target: <50ms")
        print(f"  Result: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        
        return results
    
    def test_memory_usage(self) -> Dict[str, Any]:
        """Test Target: <500MB memory usage under normal load"""
        print("\n📊 Testing memory usage...")
        
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = current_memory - self.start_memory
        
        results = {
            "start_memory_mb": self.start_memory,
            "current_memory_mb": current_memory,
            "memory_increase_mb": memory_increase,
            "target_max_mb": 100,  # Adjusted for component testing
            "passed": memory_increase < 100
        }
        
        print(f"  Start memory: {self.start_memory:.2f}MB")
        print(f"  Current memory: {current_memory:.2f}MB")
        print(f"  Memory increase: {memory_increase:.2f}MB")
        print(f"  Target increase: <100MB")
        print(f"  Result: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        
        return results
    
    async def test_concurrent_operations(self) -> Dict[str, Any]:
        """Test concurrent command registration and validation"""
        print("\n📊 Testing concurrent operations...")
        
        async def register_and_validate_command(i):
            try:
                command_def = self.create_test_command("simple")
                
                # Register command
                success, validation_result = await self.command_registry.register_command(
                    command_def,
                    author_id=f"concurrent_test_{i}",
                    validate_agents=False,
                    dry_run=True
                )
                
                return {"success": success, "command_name": command_def.name}
                
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        start_time = time.perf_counter()
        
        # Run 20 concurrent operations
        tasks = [register_and_validate_command(i) for i in range(20)]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        successful_operations = sum(1 for r in results_list if isinstance(r, dict) and r.get("success"))
        
        results = {
            "total_operations": len(tasks),
            "successful_operations": successful_operations,
            "total_time_seconds": total_time,
            "operations_per_second": successful_operations / total_time,
            "target_concurrent": 15,
            "passed": successful_operations >= 15
        }
        
        print(f"  Total operations: {len(tasks)}")
        print(f"  Successful operations: {successful_operations}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Operations per second: {results['operations_per_second']:.2f}")
        print(f"  Target: ≥15 successful concurrent operations")
        print(f"  Result: {'✅ PASSED' if results['passed'] else '❌ FAILED'}")
        
        return results
    
    async def run_validation(self) -> Dict[str, Any]:
        """Run simplified performance validation."""
        print("🔥 Starting Custom Commands System Performance Validation (Simplified)")
        print("=" * 70)
        
        await self.setup_system()
        
        # Run all performance tests
        self.results = {
            "command_parsing": await self.test_command_parsing_performance(),
            "task_distribution": await self.test_task_distribution_performance(),
            "command_validation": await self.test_command_validation_performance(),
            "concurrent_operations": await self.test_concurrent_operations(),
            "memory_usage": self.test_memory_usage()
        }
        
        # Generate summary
        print("\n" + "=" * 70)
        print("📋 PERFORMANCE VALIDATION SUMMARY")
        print("=" * 70)
        
        all_passed = True
        for test_name, result in self.results.items():
            status = "✅ PASSED" if result.get("passed", False) else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
            if not result.get("passed", False):
                all_passed = False
        
        print("=" * 70)
        overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
        print(f"Overall Result: {overall_status}")
        
        if all_passed:
            print("\n🎉 Custom Commands System core components meet performance targets!")
            print("Ready for integration with full infrastructure.")
        else:
            print("\n⚠️  Some performance targets not met. Review results above.")
        
        return {
            "overall_passed": all_passed,
            "detailed_results": self.results,
            "timestamp": datetime.utcnow().isoformat()
        }

async def main():
    """Main entry point for simplified performance validation."""
    validator = SimplePerformanceValidator()
    results = await validator.run_validation()
    
    # Save results to file
    import json
    with open("performance_validation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Detailed results saved to: performance_validation_results.json")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())