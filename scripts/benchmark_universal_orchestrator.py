#!/usr/bin/env python3
"""
Performance Benchmark Suite for Universal Orchestrator

Validates all critical performance requirements:
- Agent registration: <100ms per agent
- Concurrent agents: 50+ simultaneous agents
- Task delegation: <500ms for complex routing
- Memory usage: <50MB base overhead
- System initialization: <2000ms

This script provides comprehensive benchmarking to validate that the
consolidation from 28+ orchestrator files maintains performance while
providing all required functionality.
"""

import asyncio
import json
import time
import psutil
import statistics
from datetime import datetime
from typing import Dict, List, Tuple, Any
import uuid
import argparse
import sys
import os

# Add the parent directory to the path to import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.universal_orchestrator import (
    UniversalOrchestrator, 
    OrchestratorConfig, 
    OrchestratorMode,
    AgentRole,
    get_universal_orchestrator,
    shutdown_universal_orchestrator
)
from app.models.task import TaskPriority


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = time.time()
        self.end_time = None
        self.duration_ms = None
        self.success = False
        self.details = {}
        self.errors = []
    
    def finish(self, success: bool = True, details: Dict = None):
        """Mark benchmark as finished."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.success = success
        self.details = details or {}
    
    def add_error(self, error: str):
        """Add an error to the benchmark."""
        self.errors.append(error)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'duration_ms': self.duration_ms,
            'success': self.success,
            'details': self.details,
            'errors': self.errors,
            'timestamp': datetime.fromtimestamp(self.start_time).isoformat()
        }


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmark suite for Universal Orchestrator."""
    
    def __init__(self, config: OrchestratorConfig = None):
        """Initialize benchmark suite."""
        self.config = config or OrchestratorConfig(
            mode=OrchestratorMode.TESTING,
            max_agents=100,
            max_concurrent_tasks=1000,
            enable_performance_plugin=True,
            enable_security_plugin=False,
            enable_context_plugin=False,
            enable_automation_plugin=False
        )
        self.orchestrator = None
        self.benchmark_results = []
        self.system_info = {}
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        print("🚀 Starting Universal Orchestrator Performance Benchmark Suite")
        print("=" * 80)
        
        # Collect system information
        await self._collect_system_info()
        
        # Run benchmarks in order
        benchmarks = [
            ("System Initialization", self._benchmark_initialization),
            ("Agent Registration Performance", self._benchmark_agent_registration),
            ("Concurrent Agent Support", self._benchmark_concurrent_agents),
            ("Task Delegation Performance", self._benchmark_task_delegation),
            ("Memory Usage", self._benchmark_memory_usage),
            ("High Load Performance", self._benchmark_high_load),
            ("System Recovery", self._benchmark_system_recovery),
            ("Plugin Performance", self._benchmark_plugin_performance),
        ]
        
        for benchmark_name, benchmark_func in benchmarks:
            print(f"\n📊 Running: {benchmark_name}")
            print("-" * 50)
            
            try:
                result = await benchmark_func()
                self.benchmark_results.append(result)
                self._print_benchmark_result(result)
                
            except Exception as e:
                error_result = BenchmarkResult(benchmark_name)
                error_result.finish(success=False, details={'error': str(e)})
                error_result.add_error(f"Benchmark failed: {str(e)}")
                self.benchmark_results.append(error_result)
                print(f"❌ FAILED: {str(e)}")
        
        # Generate final report
        report = self._generate_final_report()
        return report
    
    async def _collect_system_info(self):
        """Collect system information for benchmarking context."""
        try:
            self.system_info = {
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'platform': sys.platform,
                'python_version': sys.version,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"💻 System Info:")
            print(f"   CPU Cores: {self.system_info['cpu_count']} physical, {self.system_info['cpu_count_logical']} logical")
            print(f"   Memory: {self.system_info['memory_total_gb']:.1f} GB")
            print(f"   Platform: {self.system_info['platform']}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not collect system info: {e}")
    
    async def _benchmark_initialization(self) -> BenchmarkResult:
        """Benchmark orchestrator initialization performance."""
        result = BenchmarkResult("System Initialization")
        
        try:
            # Test initialization time
            start_time = time.time()
            self.orchestrator = UniversalOrchestrator(self.config)
            init_success = await self.orchestrator.initialize()
            init_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Performance requirement: <2000ms
            target_time = self.config.max_system_initialization_ms
            meets_requirement = init_time < target_time
            
            result.finish(
                success=init_success and meets_requirement,
                details={
                    'initialization_time_ms': init_time,
                    'target_time_ms': target_time,
                    'meets_requirement': meets_requirement,
                    'orchestrator_id': self.orchestrator.orchestrator_id if self.orchestrator else None
                }
            )
            
            if not meets_requirement:
                result.add_error(f"Initialization time ({init_time:.1f}ms) exceeds requirement ({target_time}ms)")
            
        except Exception as e:
            result.finish(success=False, details={'error': str(e)})
            result.add_error(f"Initialization failed: {str(e)}")
        
        return result
    
    async def _benchmark_agent_registration(self) -> BenchmarkResult:
        """Benchmark agent registration performance."""
        result = BenchmarkResult("Agent Registration Performance")
        
        try:
            if not self.orchestrator:
                raise Exception("Orchestrator not initialized")
            
            # Test single agent registration times
            num_tests = 10
            registration_times = []
            
            for i in range(num_tests):
                agent_id = f"perf_test_agent_{i:03d}"
                capabilities = ["python", "javascript", "testing", "performance"]
                
                start_time = time.time()
                success = await self.orchestrator.register_agent(
                    agent_id, AgentRole.WORKER, capabilities
                )
                registration_time = (time.time() - start_time) * 1000
                
                if success:
                    registration_times.append(registration_time)
                else:
                    result.add_error(f"Failed to register agent {agent_id}")
            
            if registration_times:
                avg_time = statistics.mean(registration_times)
                max_time = max(registration_times)
                min_time = min(registration_times)
                std_dev = statistics.stdev(registration_times) if len(registration_times) > 1 else 0
                
                # Performance requirement: <100ms
                target_time = self.config.max_agent_registration_ms
                meets_requirement = max_time < target_time
                
                result.finish(
                    success=meets_requirement,
                    details={
                        'average_registration_time_ms': avg_time,
                        'max_registration_time_ms': max_time,
                        'min_registration_time_ms': min_time,
                        'std_dev_ms': std_dev,
                        'target_time_ms': target_time,
                        'meets_requirement': meets_requirement,
                        'successful_registrations': len(registration_times),
                        'total_attempts': num_tests
                    }
                )
                
                if not meets_requirement:
                    result.add_error(f"Max registration time ({max_time:.1f}ms) exceeds requirement ({target_time}ms)")
            else:
                result.finish(success=False, details={'error': 'No successful registrations'})
                result.add_error("No agents could be registered successfully")
        
        except Exception as e:
            result.finish(success=False, details={'error': str(e)})
            result.add_error(f"Agent registration benchmark failed: {str(e)}")
        
        return result
    
    async def _benchmark_concurrent_agents(self) -> BenchmarkResult:
        """Benchmark concurrent agent support."""
        result = BenchmarkResult("Concurrent Agent Support")
        
        try:
            if not self.orchestrator:
                raise Exception("Orchestrator not initialized")
            
            # Clear existing agents for clean test
            self.orchestrator.agents.clear()
            self.orchestrator.agent_capabilities_index.clear()
            
            # Test concurrent registration of 50+ agents
            target_concurrent_agents = 55
            
            async def register_concurrent_agent(agent_index):
                agent_id = f"concurrent_agent_{agent_index:03d}"
                capabilities = [
                    "python", "concurrent", "load_test", 
                    f"specialization_{agent_index % 5}"  # Add some variety
                ]
                
                start_time = time.time()
                success = await self.orchestrator.register_agent(
                    agent_id, AgentRole.WORKER, capabilities
                )
                registration_time = (time.time() - start_time) * 1000
                
                return success, registration_time, agent_id
            
            # Execute concurrent registrations
            start_time = time.time()
            tasks = [register_concurrent_agent(i) for i in range(target_concurrent_agents)]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = (time.time() - start_time) * 1000
            
            # Analyze results
            successful_registrations = []
            registration_times = []
            failed_registrations = []
            
            for i, result_item in enumerate(results_list):
                if isinstance(result_item, Exception):
                    failed_registrations.append(f"Agent {i}: {str(result_item)}")
                else:
                    success, reg_time, agent_id = result_item
                    if success:
                        successful_registrations.append(agent_id)
                        registration_times.append(reg_time)
                    else:
                        failed_registrations.append(f"Agent {i}: Registration failed")
            
            # Performance evaluation
            meets_concurrent_requirement = len(successful_registrations) >= 50
            avg_registration_time = statistics.mean(registration_times) if registration_times else 0
            max_registration_time = max(registration_times) if registration_times else 0
            
            result.finish(
                success=meets_concurrent_requirement,
                details={
                    'target_concurrent_agents': target_concurrent_agents,
                    'successful_registrations': len(successful_registrations),
                    'failed_registrations': len(failed_registrations),
                    'total_time_ms': total_time,
                    'average_registration_time_ms': avg_registration_time,
                    'max_registration_time_ms': max_registration_time,
                    'meets_requirement': meets_concurrent_requirement,
                    'agents_per_second': len(successful_registrations) / (total_time / 1000) if total_time > 0 else 0
                }
            )
            
            if not meets_concurrent_requirement:
                result.add_error(f"Could not register 50+ concurrent agents (achieved: {len(successful_registrations)})")
            
            for failure in failed_registrations[:5]:  # Show first 5 failures
                result.add_error(failure)
        
        except Exception as e:
            result.finish(success=False, details={'error': str(e)})
            result.add_error(f"Concurrent agents benchmark failed: {str(e)}")
        
        return result
    
    async def _benchmark_task_delegation(self) -> BenchmarkResult:
        """Benchmark task delegation performance."""
        result = BenchmarkResult("Task Delegation Performance")
        
        try:
            if not self.orchestrator:
                raise Exception("Orchestrator not initialized")
            
            # Ensure we have registered agents
            if len(self.orchestrator.agents) < 10:
                for i in range(15):
                    await self.orchestrator.register_agent(
                        f"delegation_agent_{i:02d}",
                        AgentRole.WORKER,
                        ["python", "javascript", "testing", f"skill_{i % 3}"]
                    )
            
            # Test task delegation performance
            num_tasks = 20
            delegation_times = []
            successful_delegations = []
            failed_delegations = []
            
            for i in range(num_tasks):
                task_id = f"perf_task_{i:03d}"
                required_capabilities = ["python", "testing"]
                
                start_time = time.time()
                assigned_agent = await self.orchestrator.delegate_task(
                    task_id, "performance_test", required_capabilities, TaskPriority.MEDIUM
                )
                delegation_time = (time.time() - start_time) * 1000
                
                if assigned_agent:
                    delegation_times.append(delegation_time)
                    successful_delegations.append((task_id, assigned_agent))
                    
                    # Complete task immediately to free agent for next delegation
                    await self.orchestrator.complete_task(
                        task_id, assigned_agent, {"result": "success"}, success=True
                    )
                else:
                    failed_delegations.append(task_id)
            
            # Analyze performance
            if delegation_times:
                avg_time = statistics.mean(delegation_times)
                max_time = max(delegation_times)
                min_time = min(delegation_times)
                
                # Performance requirement: <500ms
                target_time = self.config.max_task_delegation_ms
                meets_requirement = max_time < target_time
                
                result.finish(
                    success=meets_requirement and len(failed_delegations) == 0,
                    details={
                        'average_delegation_time_ms': avg_time,
                        'max_delegation_time_ms': max_time,
                        'min_delegation_time_ms': min_time,
                        'target_time_ms': target_time,
                        'meets_requirement': meets_requirement,
                        'successful_delegations': len(successful_delegations),
                        'failed_delegations': len(failed_delegations),
                        'total_tasks': num_tasks,
                        'delegation_success_rate': len(successful_delegations) / num_tasks * 100
                    }
                )
                
                if not meets_requirement:
                    result.add_error(f"Max delegation time ({max_time:.1f}ms) exceeds requirement ({target_time}ms)")
                
                if failed_delegations:
                    result.add_error(f"Failed to delegate {len(failed_delegations)} tasks")
            else:
                result.finish(success=False, details={'error': 'No successful task delegations'})
                result.add_error("No tasks could be delegated successfully")
        
        except Exception as e:
            result.finish(success=False, details={'error': str(e)})
            result.add_error(f"Task delegation benchmark failed: {str(e)}")
        
        return result
    
    async def _benchmark_memory_usage(self) -> BenchmarkResult:
        """Benchmark memory usage."""
        result = BenchmarkResult("Memory Usage")
        
        try:
            import gc
            
            # Force garbage collection to get accurate baseline
            gc.collect()
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Create load to measure memory overhead
            if not self.orchestrator:
                raise Exception("Orchestrator not initialized")
            
            # Add more agents and tasks to stress memory
            additional_agents = 30
            for i in range(additional_agents):
                await self.orchestrator.register_agent(
                    f"memory_agent_{i:03d}",
                    AgentRole.WORKER,
                    ["python", "memory_test", f"specialization_{i % 5}"]
                )
            
            # Create and complete many tasks to build history
            num_tasks = 100
            for i in range(num_tasks):
                task_id = f"memory_task_{i:04d}"
                assigned_agent = await self.orchestrator.delegate_task(
                    task_id, "memory_test", ["python"], TaskPriority.MEDIUM
                )
                if assigned_agent:
                    await self.orchestrator.complete_task(
                        task_id, assigned_agent, {"result": f"task_{i}_completed"}, True
                    )
            
            # Force garbage collection again
            gc.collect()
            final_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_increase = final_memory - initial_memory
            
            # Performance requirement: <50MB base overhead
            target_memory = self.config.max_memory_mb
            meets_requirement = memory_increase < target_memory
            
            # Get memory breakdown
            status = await self.orchestrator.get_system_status()
            
            result.finish(
                success=meets_requirement,
                details={
                    'initial_memory_mb': initial_memory,
                    'final_memory_mb': final_memory,
                    'memory_increase_mb': memory_increase,
                    'target_memory_mb': target_memory,
                    'meets_requirement': meets_requirement,
                    'total_agents': len(self.orchestrator.agents),
                    'completed_tasks': status.get('tasks', {}).get('completed_total', 0),
                    'memory_per_agent_kb': (memory_increase * 1024) / max(len(self.orchestrator.agents), 1)
                }
            )
            
            if not meets_requirement:
                result.add_error(f"Memory increase ({memory_increase:.1f}MB) exceeds requirement ({target_memory}MB)")
        
        except Exception as e:
            result.finish(success=False, details={'error': str(e)})
            result.add_error(f"Memory usage benchmark failed: {str(e)}")
        
        return result
    
    async def _benchmark_high_load(self) -> BenchmarkResult:
        """Benchmark performance under high load."""
        result = BenchmarkResult("High Load Performance")
        
        try:
            if not self.orchestrator:
                raise Exception("Orchestrator not initialized")
            
            # High load scenario: Many concurrent operations
            load_start_time = time.time()
            
            # Clear existing state for clean test
            for task_id in list(self.orchestrator.active_tasks.keys()):
                task_execution = self.orchestrator.active_tasks[task_id]
                await self.orchestrator.complete_task(
                    task_id, task_execution.agent_id, {"result": "cleanup"}, True
                )
            
            # Ensure we have enough agents for high load
            current_agent_count = len(self.orchestrator.agents)
            target_agents = 40
            
            if current_agent_count < target_agents:
                for i in range(current_agent_count, target_agents):
                    await self.orchestrator.register_agent(
                        f"load_agent_{i:03d}",
                        AgentRole.WORKER,
                        ["python", "high_load", "concurrent"]
                    )
            
            # Create high concurrent load
            concurrent_operations = []
            num_concurrent_tasks = 80  # More tasks than agents to test queuing
            
            # Create all tasks concurrently
            async def create_and_complete_task(task_index):
                task_id = f"load_task_{task_index:04d}"
                
                # Task delegation
                start_time = time.time()
                assigned_agent = await self.orchestrator.delegate_task(
                    task_id, "high_load_test", ["python"], TaskPriority.MEDIUM
                )
                delegation_time = time.time() - start_time
                
                if assigned_agent:
                    # Simulate some work
                    await asyncio.sleep(0.001)  # 1ms simulated work
                    
                    # Task completion
                    completion_success = await self.orchestrator.complete_task(
                        task_id, assigned_agent, {"load_test": True}, True
                    )
                    
                    return True, delegation_time * 1000, assigned_agent  # Convert to ms
                else:
                    return False, delegation_time * 1000, None
            
            # Execute all operations concurrently
            load_tasks = [create_and_complete_task(i) for i in range(num_concurrent_tasks)]
            load_results = await asyncio.gather(*load_tasks, return_exceptions=True)
            
            total_load_time = (time.time() - load_start_time) * 1000
            
            # Analyze results
            successful_operations = []
            delegation_times = []
            failed_operations = 0
            
            for i, load_result in enumerate(load_results):
                if isinstance(load_result, Exception):
                    failed_operations += 1
                else:
                    success, delegation_time, agent = load_result
                    if success:
                        successful_operations.append((i, agent))
                        delegation_times.append(delegation_time)
                    else:
                        failed_operations += 1
            
            # Performance evaluation
            success_rate = len(successful_operations) / num_concurrent_tasks * 100
            avg_delegation_time = statistics.mean(delegation_times) if delegation_times else 0
            max_delegation_time = max(delegation_times) if delegation_times else 0
            throughput = len(successful_operations) / (total_load_time / 1000)  # operations per second
            
            # Requirements: Should handle high load gracefully
            meets_requirement = (
                success_rate >= 75 and  # At least 75% success rate
                max_delegation_time < self.config.max_task_delegation_ms * 2  # Allow some variance under load
            )
            
            result.finish(
                success=meets_requirement,
                details={
                    'total_load_time_ms': total_load_time,
                    'concurrent_tasks': num_concurrent_tasks,
                    'successful_operations': len(successful_operations),
                    'failed_operations': failed_operations,
                    'success_rate_percent': success_rate,
                    'average_delegation_time_ms': avg_delegation_time,
                    'max_delegation_time_ms': max_delegation_time,
                    'throughput_ops_per_second': throughput,
                    'meets_requirement': meets_requirement,
                    'available_agents': len(self.orchestrator.agents)
                }
            )
            
            if success_rate < 75:
                result.add_error(f"Success rate ({success_rate:.1f}%) below acceptable threshold (75%)")
            
            if max_delegation_time > self.config.max_task_delegation_ms * 2:
                result.add_error(f"Max delegation time under load ({max_delegation_time:.1f}ms) too high")
        
        except Exception as e:
            result.finish(success=False, details={'error': str(e)})
            result.add_error(f"High load benchmark failed: {str(e)}")
        
        return result
    
    async def _benchmark_system_recovery(self) -> BenchmarkResult:
        """Benchmark system recovery capabilities."""
        result = BenchmarkResult("System Recovery")
        
        try:
            if not self.orchestrator:
                raise Exception("Orchestrator not initialized")
            
            # Test circuit breaker recovery
            initial_status = await self.orchestrator.get_system_status()
            
            # Simulate system stress that might trigger circuit breakers
            # Force some circuit breaker failures
            breaker = self.orchestrator.circuit_breakers['agent_registration']
            original_state = breaker.state
            
            # Open circuit breaker manually to test recovery
            breaker.state = CircuitBreakerState.OPEN
            breaker.last_failure_time = datetime.now()
            
            # Wait a short time and test recovery
            await asyncio.sleep(0.1)
            
            # Attempt operation that should fail due to circuit breaker
            registration_result = await self.orchestrator.register_agent(
                "recovery_test_agent", AgentRole.WORKER, ["recovery"]
            )
            
            # Reset circuit breaker for recovery test
            breaker.record_success()
            
            # Test that system can recover
            recovery_registration = await self.orchestrator.register_agent(
                "recovery_success_agent", AgentRole.WORKER, ["recovery"]
            )
            
            final_status = await self.orchestrator.get_system_status()
            
            recovery_successful = (
                registration_result == False and  # Should fail with circuit breaker open
                recovery_registration == True and  # Should succeed after recovery
                final_status['health_status'] in ['healthy', 'degraded']  # System should be functional
            )
            
            result.finish(
                success=recovery_successful,
                details={
                    'circuit_breaker_test_passed': registration_result == False,
                    'recovery_test_passed': recovery_registration == True,
                    'final_health_status': final_status['health_status'],
                    'total_agents_after_recovery': final_status['agents']['total'],
                    'system_functional': final_status['agents']['total'] > 0
                }
            )
            
            if not recovery_successful:
                result.add_error("System did not recover properly from circuit breaker test")
        
        except Exception as e:
            result.finish(success=False, details={'error': str(e)})
            result.add_error(f"System recovery benchmark failed: {str(e)}")
        
        return result
    
    async def _benchmark_plugin_performance(self) -> BenchmarkResult:
        """Benchmark plugin system performance impact."""
        result = BenchmarkResult("Plugin Performance")
        
        try:
            if not self.orchestrator:
                raise Exception("Orchestrator not initialized")
            
            # Test plugin system overhead
            plugin_manager = self.orchestrator.plugin_manager
            
            # Measure performance with plugins
            num_operations = 100
            operations_with_plugins = []
            
            for i in range(num_operations):
                start_time = time.time()
                
                # Simulate plugin hook execution
                test_context = {
                    'test_operation': i,
                    'agent_id': f'plugin_test_agent_{i}',
                    'metadata': {'benchmark': True}
                }
                
                # This would normally execute plugin hooks
                # For benchmarking, we test the hook execution infrastructure
                processed_context = await plugin_manager.execute_hooks('pre_task_execution', test_context)
                
                operation_time = (time.time() - start_time) * 1000  # Convert to ms
                operations_with_plugins.append(operation_time)
            
            # Analyze plugin performance impact
            if operations_with_plugins:
                avg_plugin_overhead = statistics.mean(operations_with_plugins)
                max_plugin_overhead = max(operations_with_plugins)
                
                # Plugin overhead should be minimal (< 10ms per operation)
                acceptable_overhead = 10.0  # ms
                meets_requirement = max_plugin_overhead < acceptable_overhead
                
                result.finish(
                    success=meets_requirement,
                    details={
                        'average_plugin_overhead_ms': avg_plugin_overhead,
                        'max_plugin_overhead_ms': max_plugin_overhead,
                        'acceptable_overhead_ms': acceptable_overhead,
                        'meets_requirement': meets_requirement,
                        'operations_tested': num_operations,
                        'plugin_system_functional': True
                    }
                )
                
                if not meets_requirement:
                    result.add_error(f"Plugin overhead ({max_plugin_overhead:.1f}ms) exceeds acceptable limit ({acceptable_overhead}ms)")
            else:
                result.finish(success=False, details={'error': 'No plugin operations completed'})
                result.add_error("Plugin performance test failed to complete any operations")
        
        except Exception as e:
            result.finish(success=False, details={'error': str(e)})
            result.add_error(f"Plugin performance benchmark failed: {str(e)}")
        
        return result
    
    def _print_benchmark_result(self, result: BenchmarkResult):
        """Print formatted benchmark result."""
        status = "✅ PASSED" if result.success else "❌ FAILED"
        print(f"{status} {result.name} ({result.duration_ms:.1f}ms)")
        
        # Print key metrics
        if result.details:
            for key, value in result.details.items():
                if key.endswith('_ms') and isinstance(value, (int, float)):
                    print(f"   📏 {key.replace('_', ' ').title()}: {value:.1f}ms")
                elif key.endswith('_mb') and isinstance(value, (int, float)):
                    print(f"   💾 {key.replace('_', ' ').title()}: {value:.1f}MB")
                elif key.endswith('_percent') and isinstance(value, (int, float)):
                    print(f"   📊 {key.replace('_', ' ').title()}: {value:.1f}%")
                elif key == 'meets_requirement':
                    req_status = "✅" if value else "❌"
                    print(f"   {req_status} Meets Performance Requirement: {value}")
        
        # Print errors
        for error in result.errors:
            print(f"   🔴 {error}")
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        total_benchmarks = len(self.benchmark_results)
        passed_benchmarks = sum(1 for r in self.benchmark_results if r.success)
        failed_benchmarks = total_benchmarks - passed_benchmarks
        
        # Calculate overall performance score
        performance_score = (passed_benchmarks / total_benchmarks * 100) if total_benchmarks > 0 else 0
        
        # Analyze critical requirements
        critical_requirements = {
            'initialization_time': False,
            'agent_registration_time': False,
            'concurrent_agents_support': False,
            'task_delegation_time': False,
            'memory_usage': False
        }
        
        for result in self.benchmark_results:
            if result.name == "System Initialization" and result.success:
                critical_requirements['initialization_time'] = True
            elif result.name == "Agent Registration Performance" and result.success:
                critical_requirements['agent_registration_time'] = True
            elif result.name == "Concurrent Agent Support" and result.success:
                critical_requirements['concurrent_agents_support'] = True
            elif result.name == "Task Delegation Performance" and result.success:
                critical_requirements['task_delegation_time'] = True
            elif result.name == "Memory Usage" and result.success:
                critical_requirements['memory_usage'] = True
        
        all_critical_passed = all(critical_requirements.values())
        
        # Create final report
        report = {
            'summary': {
                'overall_status': 'PASSED' if all_critical_passed else 'FAILED',
                'performance_score': performance_score,
                'total_benchmarks': total_benchmarks,
                'passed_benchmarks': passed_benchmarks,
                'failed_benchmarks': failed_benchmarks,
                'all_critical_requirements_met': all_critical_passed,
                'critical_requirements': critical_requirements
            },
            'system_info': self.system_info,
            'benchmark_results': [r.to_dict() for r in self.benchmark_results],
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'max_agents': self.config.max_agents,
                'max_concurrent_tasks': self.config.max_concurrent_tasks,
                'max_agent_registration_ms': self.config.max_agent_registration_ms,
                'max_task_delegation_ms': self.config.max_task_delegation_ms,
                'max_system_initialization_ms': self.config.max_system_initialization_ms,
                'max_memory_mb': self.config.max_memory_mb
            }
        }
        
        return report
    
    async def cleanup(self):
        """Cleanup benchmark resources."""
        if self.orchestrator:
            try:
                await self.orchestrator.shutdown()
            except Exception as e:
                print(f"⚠️ Warning: Error during orchestrator cleanup: {e}")


async def main():
    """Main benchmark execution function."""
    parser = argparse.ArgumentParser(description="Universal Orchestrator Performance Benchmark Suite")
    parser.add_argument('--output', '-o', help='Output file for benchmark results (JSON)', default=None)
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()
    
    # Create and run benchmark suite
    benchmark_suite = PerformanceBenchmarkSuite()
    
    try:
        report = await benchmark_suite.run_all_benchmarks()
        
        # Print final summary
        print("\n" + "=" * 80)
        print("🎯 FINAL BENCHMARK RESULTS")
        print("=" * 80)
        
        summary = report['summary']
        status_emoji = "🎉" if summary['overall_status'] == 'PASSED' else "💥"
        
        print(f"{status_emoji} Overall Status: {summary['overall_status']}")
        print(f"📈 Performance Score: {summary['performance_score']:.1f}%")
        print(f"📊 Benchmarks: {summary['passed_benchmarks']}/{summary['total_benchmarks']} passed")
        
        print("\n🎯 Critical Requirements:")
        for req, status in summary['critical_requirements'].items():
            status_emoji = "✅" if status else "❌"
            req_name = req.replace('_', ' ').title()
            print(f"   {status_emoji} {req_name}")
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {args.output}")
        
        # Exit with appropriate code
        exit_code = 0 if summary['overall_status'] == 'PASSED' else 1
        
        if summary['overall_status'] == 'PASSED':
            print("\n🚀 Universal Orchestrator is ready for production!")
            print("   All critical performance requirements have been met.")
        else:
            print("\n⚠️  Universal Orchestrator needs optimization before production.")
            print("   Some critical performance requirements were not met.")
        
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"\n💥 Benchmark suite failed: {e}")
        sys.exit(1)
        
    finally:
        await benchmark_suite.cleanup()


if __name__ == "__main__":
    from app.common.utilities.script_base import BaseScript, script_main
    
    class BenchmarkUniversalOrchestratorScript(BaseScript):
        """Refactored script using standardized pattern."""
        
        async def execute(self):
            """Execute the main script logic."""
            await main()
            
            return {"status": "completed"}
    
    script_main(BenchmarkUniversalOrchestratorScript)