"""
Performance Benchmarks for Unified Production Orchestrator

Validates all performance requirements:
- Agent Registration: <100ms
- Task Delegation: <500ms for complex routing  
- Concurrent Agents: 50+ simultaneous agents
- Memory Efficiency: <50MB base overhead
"""

import asyncio
import time
import uuid
import statistics
import psutil
import gc
from typing import List, Dict, Any
from datetime import datetime

from app.core.unified_production_orchestrator import (
    UnifiedProductionOrchestrator,
    OrchestratorConfig,
    AgentProtocol,
    AgentState,
    AgentCapability
)
from app.models.task import Task, TaskStatus, TaskPriority


class BenchmarkAgent:
    """Mock agent optimized for benchmarking."""
    
    def __init__(self, agent_id: str = None, capabilities: List[AgentCapability] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.state = AgentState.ACTIVE
        self.capabilities = capabilities or [
            AgentCapability(
                name="benchmark",
                description="Benchmark test agent",
                confidence_level=0.9,
                specialization_areas=["testing", "benchmark"]
            )
        ]
        self.task_count = 0
        self.execution_times = []
        
    async def execute_task(self, task: Task) -> Any:
        """Fast mock task execution."""
        start_time = time.time()
        self.task_count += 1
        
        # Simulate minimal processing time
        await asyncio.sleep(0.001)  # 1ms simulation
        
        execution_time = time.time() - start_time
        self.execution_times.append(execution_time)
        
        return f"Task {task.id} completed in {execution_time:.3f}s"
        
    async def get_status(self) -> AgentState:
        return self.state
        
    async def get_capabilities(self) -> List[AgentCapability]:
        return self.capabilities
        
    async def shutdown(self, graceful: bool = True) -> None:
        self.state = AgentState.TERMINATED


class PerformanceBenchmark:
    """Performance benchmark suite for orchestrator."""
    
    def __init__(self):
        self.results = {}
        self.process = psutil.Process()
        
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        print("🚀 Starting Unified Production Orchestrator Performance Benchmarks")
        print("=" * 80)
        
        # Run individual benchmarks
        self.results['agent_registration'] = await self.benchmark_agent_registration()
        self.results['task_delegation'] = await self.benchmark_task_delegation()
        self.results['concurrent_capacity'] = await self.benchmark_concurrent_capacity()
        self.results['memory_efficiency'] = await self.benchmark_memory_efficiency()
        self.results['load_balancing'] = await self.benchmark_load_balancing()
        self.results['intelligent_routing'] = await self.benchmark_intelligent_routing()
        
        # Generate summary
        self.results['summary'] = self._generate_summary()
        
        print("\n" + "=" * 80)
        print("📊 Performance Benchmark Results Summary")
        print("=" * 80)
        
        self._print_summary()
        
        return self.results
    
    async def benchmark_agent_registration(self) -> Dict[str, Any]:
        """Benchmark agent registration performance (Target: <100ms)."""
        print("\n📋 Agent Registration Performance Test")
        print("-" * 40)
        
        config = OrchestratorConfig(
            max_concurrent_agents=30,
            registration_target_ms=100.0
        )
        
        orchestrator = UnifiedProductionOrchestrator(config)
        await orchestrator.start()
        
        try:
            registration_times = []
            successful_registrations = 0
            
            # Test registration performance
            for i in range(25):
                agent = BenchmarkAgent()
                
                start_time = time.perf_counter()
                try:
                    agent_id = await orchestrator.register_agent(agent)
                    registration_time = (time.perf_counter() - start_time) * 1000  # ms
                    
                    registration_times.append(registration_time)
                    successful_registrations += 1
                    
                    if i < 5 or i % 5 == 0:  # Log first 5 and every 5th
                        print(f"  Agent {i+1:2d}: {registration_time:6.2f}ms")
                        
                except Exception as e:
                    print(f"  Agent {i+1:2d}: FAILED - {e}")
            
            # Calculate statistics
            avg_time = statistics.mean(registration_times)
            median_time = statistics.median(registration_times)
            max_time = max(registration_times)
            min_time = min(registration_times)
            p95_time = statistics.quantiles(registration_times, n=20)[18]  # 95th percentile
            
            target_met = avg_time < config.registration_target_ms
            
            results = {
                'total_attempts': 25,
                'successful_registrations': successful_registrations,
                'average_time_ms': avg_time,
                'median_time_ms': median_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'p95_time_ms': p95_time,
                'target_ms': config.registration_target_ms,
                'target_met': target_met,
                'success_rate': successful_registrations / 25
            }
            
            print(f"\n  Results:")
            print(f"    Average: {avg_time:6.2f}ms (Target: <{config.registration_target_ms}ms)")
            print(f"    Median:  {median_time:6.2f}ms")
            print(f"    95th %:  {p95_time:6.2f}ms")
            print(f"    Range:   {min_time:6.2f}ms - {max_time:6.2f}ms")
            print(f"    Success: {successful_registrations}/25 ({results['success_rate']*100:.1f}%)")
            print(f"    Status:  {'✅ PASSED' if target_met else '❌ FAILED'}")
            
            return results
            
        finally:
            await orchestrator.shutdown()
    
    async def benchmark_task_delegation(self) -> Dict[str, Any]:
        """Benchmark task delegation performance (Target: <500ms)."""
        print("\n🎯 Task Delegation Performance Test")
        print("-" * 40)
        
        config = OrchestratorConfig(
            max_concurrent_agents=15,
            delegation_target_ms=500.0,
            routing_strategy="intelligent"
        )
        
        orchestrator = UnifiedProductionOrchestrator(config)
        await orchestrator.start()
        
        try:
            # Register agents with different capabilities for complex routing
            agents = []
            for i in range(10):
                capabilities = [
                    AgentCapability(
                        name=f"skill_{i % 3}",
                        description=f"Skill {i % 3}",
                        confidence_level=0.7 + (i % 3) * 0.1,
                        specialization_areas=[f"area_{i % 3}", "general"]
                    )
                ]
                agent = BenchmarkAgent(capabilities=capabilities)
                agent_id = await orchestrator.register_agent(agent)
                agents.append(agent_id)
            
            # Test delegation performance with complex routing
            delegation_times = []
            successful_delegations = 0
            
            for i in range(30):
                task = Task(
                    id=str(uuid.uuid4()),
                    title=f"Benchmark Task {i}",
                    description=f"Complex routing test for area_{i % 3}",
                    priority=TaskPriority.MEDIUM,
                    status=TaskStatus.PENDING
                )
                
                start_time = time.perf_counter()
                try:
                    agent_id = await orchestrator.delegate_task(task)
                    delegation_time = (time.perf_counter() - start_time) * 1000  # ms
                    
                    delegation_times.append(delegation_time)
                    successful_delegations += 1
                    
                    if i < 5 or i % 5 == 0:  # Log first 5 and every 5th
                        print(f"  Task {i+1:2d}: {delegation_time:6.2f}ms -> Agent {agent_id[:8]}")
                        
                except Exception as e:
                    print(f"  Task {i+1:2d}: FAILED - {e}")
            
            # Calculate statistics
            avg_time = statistics.mean(delegation_times)
            median_time = statistics.median(delegation_times)
            max_time = max(delegation_times)
            min_time = min(delegation_times)
            p95_time = statistics.quantiles(delegation_times, n=20)[18]  # 95th percentile
            
            target_met = avg_time < config.delegation_target_ms
            
            results = {
                'total_attempts': 30,
                'successful_delegations': successful_delegations,
                'average_time_ms': avg_time,
                'median_time_ms': median_time,
                'min_time_ms': min_time,
                'max_time_ms': max_time,
                'p95_time_ms': p95_time,
                'target_ms': config.delegation_target_ms,
                'target_met': target_met,
                'success_rate': successful_delegations / 30
            }
            
            print(f"\n  Results:")
            print(f"    Average: {avg_time:6.2f}ms (Target: <{config.delegation_target_ms}ms)")
            print(f"    Median:  {median_time:6.2f}ms")
            print(f"    95th %:  {p95_time:6.2f}ms")
            print(f"    Range:   {min_time:6.2f}ms - {max_time:6.2f}ms")
            print(f"    Success: {successful_delegations}/30 ({results['success_rate']*100:.1f}%)")
            print(f"    Status:  {'✅ PASSED' if target_met else '❌ FAILED'}")
            
            return results
            
        finally:
            await orchestrator.shutdown()
    
    async def benchmark_concurrent_capacity(self) -> Dict[str, Any]:
        """Benchmark concurrent agent capacity (Target: 50+ agents)."""
        print("\n🔥 Concurrent Agent Capacity Test")
        print("-" * 40)
        
        config = OrchestratorConfig(
            max_concurrent_agents=60,  # Test above 50
            max_agent_pool=70,
            registration_target_ms=150.0,  # Allow slight degradation at scale
            delegation_target_ms=750.0
        )
        
        orchestrator = UnifiedProductionOrchestrator(config)
        await orchestrator.start()
        
        try:
            # Register 55 agents concurrently to test 50+ capacity
            target_agents = 55
            print(f"  Registering {target_agents} agents concurrently...")
            
            agents = [BenchmarkAgent() for _ in range(target_agents)]
            
            start_time = time.perf_counter()
            
            # Register agents in batches for realistic load
            batch_size = 10
            successful_agents = []
            
            for batch_start in range(0, target_agents, batch_size):
                batch_end = min(batch_start + batch_size, target_agents)
                batch_agents = agents[batch_start:batch_end]
                
                batch_tasks = [
                    orchestrator.register_agent(agent) for agent in batch_agents
                ]
                
                try:
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    for i, result in enumerate(batch_results):
                        if isinstance(result, Exception):
                            print(f"    Agent {batch_start + i + 1}: FAILED - {result}")
                        else:
                            successful_agents.append(result)
                            
                    print(f"    Batch {batch_start//batch_size + 1}: {len([r for r in batch_results if not isinstance(r, Exception)])}/{len(batch_agents)} registered")
                    
                except Exception as e:
                    print(f"    Batch {batch_start//batch_size + 1}: FAILED - {e}")
            
            registration_time = time.perf_counter() - start_time
            
            print(f"\n  Registration completed in {registration_time:.2f}s")
            print(f"  Successfully registered: {len(successful_agents)}/{target_agents}")
            
            # Test task delegation at scale
            print(f"  Testing task delegation with {len(successful_agents)} agents...")
            
            tasks = [
                Task(
                    id=str(uuid.uuid4()),
                    title=f"Scale Test Task {i}",
                    description=f"Testing delegation at scale",
                    priority=TaskPriority.MEDIUM,
                    status=TaskStatus.PENDING
                )
                for i in range(100)  # More tasks than agents
            ]
            
            delegation_start = time.perf_counter()
            delegation_tasks = [orchestrator.delegate_task(task) for task in tasks]
            
            try:
                delegation_results = await asyncio.gather(*delegation_tasks, return_exceptions=True)
                successful_delegations = len([r for r in delegation_results if not isinstance(r, Exception)])
                
            except Exception as e:
                print(f"    Delegation batch failed: {e}")
                successful_delegations = 0
            
            delegation_time = time.perf_counter() - delegation_start
            
            # Get system status for verification
            status = await orchestrator.get_system_status()
            
            target_met = len(successful_agents) >= 50
            
            results = {
                'target_agents': target_agents,
                'successful_agents': len(successful_agents),
                'registration_time_seconds': registration_time,
                'delegation_test_tasks': len(tasks),
                'successful_delegations': successful_delegations,
                'delegation_time_seconds': delegation_time,
                'target_met': target_met,
                'system_status': {
                    'total_registered': status['agents']['total_registered'],
                    'idle_count': status['agents']['idle_count'],
                    'busy_count': status['agents']['busy_count']
                }
            }
            
            print(f"\n  Results:")
            print(f"    Agents registered: {len(successful_agents)}/{target_agents} (Target: ≥50)")
            print(f"    Registration time: {registration_time:.2f}s")
            print(f"    Tasks delegated:   {successful_delegations}/{len(tasks)}")
            print(f"    Delegation time:   {delegation_time:.2f}s")
            print(f"    System status:     {status['agents']['total_registered']} total, {status['agents']['idle_count']} idle")
            print(f"    Status:           {'✅ PASSED' if target_met else '❌ FAILED'}")
            
            return results
            
        finally:
            await orchestrator.shutdown()
    
    async def benchmark_memory_efficiency(self) -> Dict[str, Any]:
        """Benchmark memory efficiency (Target: <50MB base overhead)."""
        print("\n💾 Memory Efficiency Test")
        print("-" * 40)
        
        # Get baseline memory usage
        gc.collect()  # Force garbage collection
        baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        config = OrchestratorConfig(max_concurrent_agents=25)
        orchestrator = UnifiedProductionOrchestrator(config)
        
        # Measure orchestrator creation overhead
        gc.collect()
        creation_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        creation_overhead = creation_memory - baseline_memory
        
        await orchestrator.start()
        
        try:
            # Measure startup overhead
            gc.collect()
            startup_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            startup_overhead = startup_memory - baseline_memory
            
            print(f"  Baseline memory:    {baseline_memory:.2f} MB")
            print(f"  After creation:     {creation_memory:.2f} MB (+{creation_overhead:.2f} MB)")
            print(f"  After startup:      {startup_memory:.2f} MB (+{startup_overhead:.2f} MB)")
            
            # Register agents and measure memory growth
            agents = []
            memory_measurements = []
            
            for i in range(20):
                agent = BenchmarkAgent()
                agent_id = await orchestrator.register_agent(agent)
                agents.append(agent_id)
                
                if i % 5 == 4:  # Measure every 5 agents
                    gc.collect()
                    current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                    memory_growth = current_memory - baseline_memory
                    memory_measurements.append((i + 1, memory_growth))
                    print(f"  With {i+1:2d} agents:     {current_memory:.2f} MB (+{memory_growth:.2f} MB)")
            
            # Final memory measurement
            gc.collect()
            final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            final_overhead = final_memory - baseline_memory
            memory_per_agent = (final_overhead - startup_overhead) / len(agents) if agents else 0
            
            # Test memory cleanup
            print(f"\n  Testing memory cleanup...")
            agents_to_remove = agents[:10]  # Remove half the agents
            
            for agent_id in agents_to_remove:
                await orchestrator.unregister_agent(agent_id)
            
            gc.collect()
            cleanup_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            memory_freed = final_memory - cleanup_memory
            
            target_met = startup_overhead < 50.0  # Base overhead target
            
            results = {
                'baseline_memory_mb': baseline_memory,
                'creation_overhead_mb': creation_overhead,
                'startup_overhead_mb': startup_overhead,
                'final_memory_mb': final_memory,
                'final_overhead_mb': final_overhead,
                'memory_per_agent_mb': memory_per_agent,
                'agents_registered': len(agents),
                'cleanup_memory_freed_mb': memory_freed,
                'target_overhead_mb': 50.0,
                'target_met': target_met,
                'memory_measurements': memory_measurements
            }
            
            print(f"\n  Results:")
            print(f"    Base overhead:      {startup_overhead:.2f} MB (Target: <50.0 MB)")
            print(f"    Total with agents:  {final_overhead:.2f} MB")
            print(f"    Memory per agent:   {memory_per_agent:.2f} MB")
            print(f"    Memory freed:       {memory_freed:.2f} MB")
            print(f"    Status:            {'✅ PASSED' if target_met else '❌ FAILED'}")
            
            return results
            
        finally:
            await orchestrator.shutdown()
    
    async def benchmark_load_balancing(self) -> Dict[str, Any]:
        """Benchmark load balancing effectiveness."""
        print("\n⚖️  Load Balancing Performance Test")
        print("-" * 40)
        
        config = OrchestratorConfig(
            max_concurrent_agents=8,
            routing_strategy="least_loaded"
        )
        
        orchestrator = UnifiedProductionOrchestrator(config)
        await orchestrator.start()
        
        try:
            # Register agents
            agents = []
            for i in range(6):
                agent = BenchmarkAgent()
                agent_id = await orchestrator.register_agent(agent)
                agents.append(agent_id)
            
            # Delegate tasks and measure distribution
            tasks = []
            for i in range(30):  # 5 tasks per agent on average
                task = Task(
                    id=str(uuid.uuid4()),
                    title=f"Load Balance Test {i}",
                    description="Testing load balancing",
                    priority=TaskPriority.MEDIUM,
                    status=TaskStatus.PENDING
                )
                
                agent_id = await orchestrator.delegate_task(task)
                tasks.append((task.id, agent_id))
            
            # Wait for tasks to complete
            await asyncio.sleep(0.2)
            
            # Analyze task distribution
            agent_task_counts = {}
            for task_id, agent_id in tasks:
                agent_task_counts[agent_id] = agent_task_counts.get(agent_id, 0) + 1
            
            # Calculate load balancing metrics
            task_counts = list(agent_task_counts.values())
            if task_counts:
                min_tasks = min(task_counts)
                max_tasks = max(task_counts)
                avg_tasks = sum(task_counts) / len(task_counts)
                load_variance = statistics.variance(task_counts) if len(task_counts) > 1 else 0
                balance_ratio = min_tasks / max_tasks if max_tasks > 0 else 1.0
            else:
                min_tasks = max_tasks = avg_tasks = load_variance = balance_ratio = 0
            
            # Good load balancing: balance_ratio > 0.7, low variance
            target_met = balance_ratio > 0.6 and load_variance < 4.0
            
            results = {
                'total_agents': len(agents),
                'total_tasks': len(tasks),
                'task_distribution': agent_task_counts,
                'min_tasks_per_agent': min_tasks,
                'max_tasks_per_agent': max_tasks,
                'avg_tasks_per_agent': avg_tasks,
                'load_variance': load_variance,
                'balance_ratio': balance_ratio,
                'target_met': target_met
            }
            
            print(f"  Task distribution:")
            for agent_id, count in agent_task_counts.items():
                print(f"    Agent {agent_id[:8]}: {count:2d} tasks")
            
            print(f"\n  Results:")
            print(f"    Tasks per agent: {min_tasks}-{max_tasks} (avg: {avg_tasks:.1f})")
            print(f"    Balance ratio:   {balance_ratio:.2f} (Target: >0.6)")
            print(f"    Load variance:   {load_variance:.2f} (Target: <4.0)")
            print(f"    Status:         {'✅ PASSED' if target_met else '❌ FAILED'}")
            
            return results
            
        finally:
            await orchestrator.shutdown()
    
    async def benchmark_intelligent_routing(self) -> Dict[str, Any]:
        """Benchmark intelligent routing effectiveness."""
        print("\n🧠 Intelligent Routing Performance Test")
        print("-" * 40)
        
        config = OrchestratorConfig(
            max_concurrent_agents=6,
            routing_strategy="intelligent"
        )
        
        orchestrator = UnifiedProductionOrchestrator(config)
        await orchestrator.start()
        
        try:
            # Register agents with specific capabilities
            agent_capabilities = {
                'backend': ['api', 'database', 'backend'],
                'frontend': ['ui', 'react', 'frontend'],
                'devops': ['docker', 'kubernetes', 'devops'],
                'qa': ['testing', 'automation', 'qa']
            }
            
            agents = {}
            for role, areas in agent_capabilities.items():
                capabilities = [
                    AgentCapability(
                        name=role,
                        description=f"{role.title()} specialist",
                        confidence_level=0.9,
                        specialization_areas=areas
                    )
                ]
                agent = BenchmarkAgent(capabilities=capabilities)
                agent_id = await orchestrator.register_agent(agent)
                agents[role] = agent_id
            
            # Create tasks that should route to specific agents
            test_tasks = [
                ('api', 'backend'),
                ('database', 'backend'),
                ('ui', 'frontend'),
                ('react', 'frontend'),
                ('docker', 'devops'),
                ('testing', 'qa'),
                ('kubernetes', 'devops'),
                ('automation', 'qa'),
            ]
            
            routing_results = []
            correct_routings = 0
            
            for keyword, expected_role in test_tasks:
                task = Task(
                    id=str(uuid.uuid4()),
                    title=f"Task requiring {keyword}",
                    description=f"This task involves {keyword} work and should route to {expected_role}",
                    priority=TaskPriority.MEDIUM,
                    status=TaskStatus.PENDING
                )
                
                start_time = time.perf_counter()
                assigned_agent = await orchestrator.delegate_task(task)
                routing_time = (time.perf_counter() - start_time) * 1000  # ms
                
                # Check if routed correctly
                routed_correctly = assigned_agent == agents.get(expected_role)
                if routed_correctly:
                    correct_routings += 1
                
                routing_results.append({
                    'keyword': keyword,
                    'expected_role': expected_role,
                    'assigned_agent': assigned_agent,
                    'correct': routed_correctly,
                    'routing_time_ms': routing_time
                })
                
                print(f"  {keyword:10s} -> {expected_role:8s}: {'✅' if routed_correctly else '❌'} ({routing_time:.1f}ms)")
            
            routing_accuracy = correct_routings / len(test_tasks)
            avg_routing_time = statistics.mean([r['routing_time_ms'] for r in routing_results])
            
            # Good intelligent routing: >80% accuracy, reasonable speed
            target_met = routing_accuracy > 0.8 and avg_routing_time < 500.0
            
            results = {
                'total_tasks': len(test_tasks),
                'correct_routings': correct_routings,
                'routing_accuracy': routing_accuracy,
                'average_routing_time_ms': avg_routing_time,
                'routing_details': routing_results,
                'target_met': target_met
            }
            
            print(f"\n  Results:")
            print(f"    Routing accuracy: {correct_routings}/{len(test_tasks)} ({routing_accuracy*100:.1f}%)")
            print(f"    Average time:     {avg_routing_time:.1f}ms")
            print(f"    Status:          {'✅ PASSED' if target_met else '❌ FAILED'}")
            
            return results
            
        finally:
            await orchestrator.shutdown()
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate overall benchmark summary."""
        total_tests = len(self.results) - 1  # Exclude summary itself
        passed_tests = sum(1 for test, result in self.results.items() 
                          if test != 'summary' and result.get('target_met', False))
        
        # Key performance metrics
        key_metrics = {
            'agent_registration_avg_ms': self.results.get('agent_registration', {}).get('average_time_ms', 0),
            'task_delegation_avg_ms': self.results.get('task_delegation', {}).get('average_time_ms', 0),
            'concurrent_agents_achieved': self.results.get('concurrent_capacity', {}).get('successful_agents', 0),
            'memory_overhead_mb': self.results.get('memory_efficiency', {}).get('startup_overhead_mb', 0),
            'load_balance_ratio': self.results.get('load_balancing', {}).get('balance_ratio', 0),
            'routing_accuracy': self.results.get('intelligent_routing', {}).get('routing_accuracy', 0)
        }
        
        # Overall performance grade
        performance_grade = 'A' if passed_tests == total_tests else \
                          'B' if passed_tests >= total_tests * 0.8 else \
                          'C' if passed_tests >= total_tests * 0.6 else 'D'
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'performance_grade': performance_grade,
            'key_metrics': key_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def _print_summary(self):
        """Print benchmark summary."""
        summary = self.results['summary']
        metrics = summary['key_metrics']
        
        print(f"Overall Performance Grade: {summary['performance_grade']}")
        print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']} ({summary['success_rate']*100:.1f}%)")
        print()
        print("Key Performance Metrics:")
        print(f"  Agent Registration:  {metrics['agent_registration_avg_ms']:6.1f}ms (Target: <100ms)")
        print(f"  Task Delegation:     {metrics['task_delegation_avg_ms']:6.1f}ms (Target: <500ms)")
        print(f"  Concurrent Capacity: {metrics['concurrent_agents_achieved']:6d} agents (Target: ≥50)")
        print(f"  Memory Overhead:     {metrics['memory_overhead_mb']:6.1f}MB (Target: <50MB)")
        print(f"  Load Balance Ratio:  {metrics['load_balance_ratio']:6.2f} (Target: >0.6)")
        print(f"  Routing Accuracy:    {metrics['routing_accuracy']*100:6.1f}% (Target: >80%)")


async def main():
    """Run all performance benchmarks."""
    benchmark = PerformanceBenchmark()
    results = await benchmark.run_all_benchmarks()
    
    # Save results to file
    import json
    with open('orchestrator_performance_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Full results saved to: orchestrator_performance_results.json")
    
    # Return summary for validation
    return results['summary']


if __name__ == "__main__":
    import json
    results = asyncio.run(main())
    print(f"\n🎯 Final Grade: {results['performance_grade']}")
    print(f"✨ Success Rate: {results['success_rate']*100:.1f}%")