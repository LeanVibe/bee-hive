#!/usr/bin/env python3
"""
Performance Benchmarking Script for Vertical Slice 1.1: Agent Lifecycle

This script provides comprehensive performance benchmarking for the agent
lifecycle system, focusing on the 500ms task assignment target and overall
system performance metrics.
"""

import asyncio
import time
import statistics
import json
from datetime import datetime
from typing import Dict, List, Any
import argparse

import structlog
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID
from rich import print as rprint

from app.core.vertical_slice_orchestrator import VerticalSliceOrchestrator
from app.core.database import get_async_session
from app.models.agent import Agent, AgentType
from app.models.task import Task, TaskType, TaskPriority

logger = structlog.get_logger()
console = Console()


class VerticalSliceBenchmark:
    """Comprehensive benchmarking suite for the vertical slice."""
    
    def __init__(self):
        self.orchestrator = VerticalSliceOrchestrator()
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "benchmarks": {},
            "summary": {},
            "system_info": {}
        }
    
    async def setup(self):
        """Initialize the benchmarking environment."""
        rprint("[bold blue]🚀 Setting up Vertical Slice Benchmark Environment[/bold blue]")
        
        # Start the orchestrator
        success = await self.orchestrator.start_system()
        if not success:
            raise RuntimeError("Failed to start orchestrator system")
        
        # Clean up any existing test data
        await self._cleanup_test_data()
        
        rprint("[green]✅ Benchmark environment ready[/green]")
    
    async def teardown(self):
        """Clean up the benchmarking environment."""
        rprint("[bold blue]🧹 Cleaning up benchmark environment[/bold blue]")
        
        await self._cleanup_test_data()
        await self.orchestrator.stop_system()
        
        rprint("[green]✅ Cleanup completed[/green]")
    
    async def run_all_benchmarks(self, num_agents: int = 10, num_tasks: int = 50) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        rprint("[bold yellow]📊 Starting Comprehensive Performance Benchmarks[/bold yellow]")
        
        with Progress() as progress:
            main_task = progress.add_task("[cyan]Running benchmarks...", total=6)
            
            # 1. Agent Registration Benchmark
            progress.update(main_task, description="[cyan]Benchmarking agent registration...")
            await self.benchmark_agent_registration(num_agents)
            progress.advance(main_task)
            
            # 2. Task Assignment Benchmark (Primary Target: <500ms)
            progress.update(main_task, description="[cyan]Benchmarking task assignment...")
            await self.benchmark_task_assignment(num_agents, num_tasks)
            progress.advance(main_task)
            
            # 3. Task Execution Benchmark
            progress.update(main_task, description="[cyan]Benchmarking task execution...")
            await self.benchmark_task_execution(min(num_tasks, 10))  # Limit execution tests
            progress.advance(main_task)
            
            # 4. Hook System Benchmark
            progress.update(main_task, description="[cyan]Benchmarking hook system...")
            await self.benchmark_hook_system(100)  # 100 hook executions
            progress.advance(main_task)
            
            # 5. Messaging System Benchmark
            progress.update(main_task, description="[cyan]Benchmarking messaging system...")
            await self.benchmark_messaging_system(num_agents, 200)  # 200 messages
            progress.advance(main_task)
            
            # 6. System Integration Benchmark
            progress.update(main_task, description="[cyan]Running integration benchmark...")
            await self.benchmark_system_integration()
            progress.advance(main_task)
        
        # Generate summary
        self._generate_summary()
        
        return self.results
    
    async def benchmark_agent_registration(self, num_agents: int):
        """Benchmark agent registration performance."""
        rprint(f"[bold]📝 Benchmarking Agent Registration ({num_agents} agents)[/bold]")
        
        registration_times = []
        successful_registrations = 0
        
        roles = ["backend_developer", "frontend_developer", "qa_engineer", "devops_engineer"]
        
        for i in range(num_agents):
            role = roles[i % len(roles)]
            
            start_time = time.time()
            result = await self.orchestrator.lifecycle_manager.register_agent(
                name=f"benchmark_agent_{i}",
                agent_type=AgentType.CLAUDE,
                role=role,
                capabilities=[{
                    "name": f"{role}_skills",
                    "description": f"Skills for {role}",
                    "confidence_level": 0.8 + (i % 3) * 0.05,
                    "specialization_areas": [role.replace("_", " ")]
                }]
            )
            registration_time = (time.time() - start_time) * 1000
            registration_times.append(registration_time)
            
            if result.success:
                successful_registrations += 1
        
        # Store results
        self.results["benchmarks"]["agent_registration"] = {
            "total_agents": num_agents,
            "successful_registrations": successful_registrations,
            "success_rate": successful_registrations / num_agents,
            "times_ms": {
                "average": statistics.mean(registration_times),
                "median": statistics.median(registration_times),
                "min": min(registration_times),
                "max": max(registration_times),
                "stdev": statistics.stdev(registration_times) if len(registration_times) > 1 else 0
            },
            "target_met": all(t < 2000 for t in registration_times),  # 2 second target
            "all_times": registration_times
        }
        
        # Display results
        avg_time = statistics.mean(registration_times)
        success_rate = successful_registrations / num_agents * 100
        
        rprint(f"  Average time: [green]{avg_time:.2f}ms[/green]")
        rprint(f"  Success rate: [green]{success_rate:.1f}%[/green]")
        rprint(f"  Target met: [{'green' if avg_time < 2000 else 'red'}]{avg_time < 2000}[/{'green' if avg_time < 2000 else 'red'}]")
    
    async def benchmark_task_assignment(self, num_agents: int, num_tasks: int):
        """Benchmark task assignment performance (PRIMARY TARGET: <500ms)."""
        rprint(f"[bold]🎯 Benchmarking Task Assignment ({num_tasks} tasks) - TARGET: <500ms[/bold]")
        
        # Create tasks
        task_ids = []
        async with get_async_session() as db:
            task_types = [TaskType.FEATURE_DEVELOPMENT, TaskType.BUG_FIX, TaskType.TESTING, TaskType.REFACTORING]
            priorities = [TaskPriority.LOW, TaskPriority.MEDIUM, TaskPriority.HIGH, TaskPriority.CRITICAL]
            
            for i in range(num_tasks):
                task = Task(
                    title=f"Benchmark Task {i}",
                    description=f"Performance benchmark task {i}",
                    task_type=task_types[i % len(task_types)],
                    priority=priorities[i % len(priorities)],
                    required_capabilities=["backend_developer_skills", "qa_engineer_skills"][i % 2:i % 2 + 1],
                    estimated_effort=30 + (i % 120)  # 30-150 minutes
                )
                db.add(task)
                await db.commit()
                await db.refresh(task)
                task_ids.append(task.id)
        
        # Benchmark assignments
        assignment_times = []
        successful_assignments = 0
        failed_assignments = 0
        confidence_scores = []
        
        for i, task_id in enumerate(task_ids):
            start_time = time.time()
            result = await self.orchestrator.lifecycle_manager.assign_task_to_agent(
                task_id=task_id,
                max_assignment_time_ms=500.0
            )
            assignment_time = (time.time() - start_time) * 1000
            assignment_times.append(assignment_time)
            
            if result.success:
                successful_assignments += 1
                if result.confidence_score:
                    confidence_scores.append(result.confidence_score)
            else:
                failed_assignments += 1
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                rprint(f"  Assigned {i + 1}/{num_tasks} tasks...")
        
        # Store results
        target_met_count = sum(1 for t in assignment_times if t < 500)
        
        self.results["benchmarks"]["task_assignment"] = {
            "total_tasks": num_tasks,
            "successful_assignments": successful_assignments,
            "failed_assignments": failed_assignments,
            "success_rate": successful_assignments / num_tasks,
            "times_ms": {
                "average": statistics.mean(assignment_times),
                "median": statistics.median(assignment_times),
                "min": min(assignment_times),
                "max": max(assignment_times),
                "stdev": statistics.stdev(assignment_times) if len(assignment_times) > 1 else 0,
                "p95": sorted(assignment_times)[int(0.95 * len(assignment_times))],
                "p99": sorted(assignment_times)[int(0.99 * len(assignment_times))]
            },
            "target_performance": {
                "target_ms": 500,
                "assignments_under_target": target_met_count,
                "percentage_under_target": target_met_count / num_tasks * 100,
                "target_met": target_met_count == num_tasks
            },
            "confidence_scores": {
                "average": statistics.mean(confidence_scores) if confidence_scores else 0,
                "min": min(confidence_scores) if confidence_scores else 0,
                "max": max(confidence_scores) if confidence_scores else 0
            },
            "all_times": assignment_times
        }
        
        # Display results
        avg_time = statistics.mean(assignment_times)
        success_rate = successful_assignments / num_tasks * 100
        target_rate = target_met_count / num_tasks * 100
        
        rprint(f"  Average time: [{'green' if avg_time < 500 else 'red'}]{avg_time:.2f}ms[/{'green' if avg_time < 500 else 'red'}]")
        rprint(f"  Success rate: [green]{success_rate:.1f}%[/green]")
        rprint(f"  Under target: [{'green' if target_rate == 100 else 'red'}]{target_rate:.1f}%[/{'green' if target_rate == 100 else 'red'}]")
        rprint(f"  P95 time: [yellow]{sorted(assignment_times)[int(0.95 * len(assignment_times))]:.2f}ms[/yellow]")
    
    async def benchmark_task_execution(self, num_executions: int):
        """Benchmark task execution performance."""
        rprint(f"[bold]⚙️ Benchmarking Task Execution ({num_executions} executions)[/bold]")
        
        execution_times = []
        successful_executions = 0
        
        # Get some assigned tasks
        async with get_async_session() as db:
            result = await db.execute(
                db.query(Task).filter(Task.status == TaskStatus.ASSIGNED).limit(num_executions)
            )
            tasks = result.scalars().all()
        
        if len(tasks) < num_executions:
            rprint(f"[yellow]Warning: Only {len(tasks)} assigned tasks available, expected {num_executions}[/yellow]")
        
        for task in tasks[:num_executions]:
            start_time = time.time()
            
            # Start execution
            await self.orchestrator.execution_engine.start_task_execution(
                task_id=task.id,
                agent_id=task.assigned_agent_id,
                execution_context={"benchmark_mode": True}
            )
            
            # Simulate execution with progress updates
            await asyncio.sleep(0.1)  # Simulate work
            await self.orchestrator.execution_engine.update_execution_progress(
                task_id=task.id,
                phase=self.orchestrator.execution_engine.ExecutionPhase.EXECUTION,
                progress_percentage=75.0
            )
            
            # Complete execution
            result = await self.orchestrator.execution_engine.complete_task_execution(
                task_id=task.id,
                outcome=self.orchestrator.execution_engine.ExecutionOutcome.SUCCESS,
                result_data={"benchmark": True, "completed": True}
            )
            
            execution_time = (time.time() - start_time) * 1000
            execution_times.append(execution_time)
            
            if result.outcome == self.orchestrator.execution_engine.ExecutionOutcome.SUCCESS:
                successful_executions += 1
        
        if execution_times:
            self.results["benchmarks"]["task_execution"] = {
                "total_executions": len(execution_times),
                "successful_executions": successful_executions,
                "success_rate": successful_executions / len(execution_times),
                "times_ms": {
                    "average": statistics.mean(execution_times),
                    "median": statistics.median(execution_times),
                    "min": min(execution_times),
                    "max": max(execution_times),
                    "stdev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0
                }
            }
            
            avg_time = statistics.mean(execution_times)
            success_rate = successful_executions / len(execution_times) * 100
            
            rprint(f"  Average time: [green]{avg_time:.2f}ms[/green]")
            rprint(f"  Success rate: [green]{success_rate:.1f}%[/green]")
    
    async def benchmark_hook_system(self, num_hooks: int):
        """Benchmark hook system performance."""
        rprint(f"[bold]🪝 Benchmarking Hook System ({num_hooks} executions)[/bold]")
        
        # Get a registered agent
        agents = list(self.orchestrator.lifecycle_manager.active_agents)
        if not agents:
            rprint("[red]No active agents found for hook benchmark[/red]")
            return
        
        agent_id = agents[0]
        
        pre_hook_times = []
        post_hook_times = []
        security_violations = 0
        
        for i in range(num_hooks):
            # Test PreToolUse hook
            start_time = time.time()
            pre_result = await self.orchestrator.lifecycle_hooks.execute_pre_tool_hooks(
                agent_id=agent_id,
                session_id=None,
                tool_name="python_interpreter",
                parameters={"code": f"print('Benchmark test {i}')"}
            )
            pre_hook_time = (time.time() - start_time) * 1000
            pre_hook_times.append(pre_hook_time)
            
            if not pre_result.success or pre_result.security_action != pre_result.SecurityAction.ALLOW:
                security_violations += 1
            
            # Test PostToolUse hook
            start_time = time.time()
            post_result = await self.orchestrator.lifecycle_hooks.execute_post_tool_hooks(
                agent_id=agent_id,
                session_id=None,
                tool_name="python_interpreter",
                parameters={"code": f"print('Benchmark test {i}')"},
                result={"output": f"Benchmark test {i}", "exit_code": 0},
                success=True,
                execution_time_ms=50.0
            )
            post_hook_time = (time.time() - start_time) * 1000
            post_hook_times.append(post_hook_time)
        
        self.results["benchmarks"]["hook_system"] = {
            "total_hooks": num_hooks,
            "security_violations": security_violations,
            "pre_hook_times_ms": {
                "average": statistics.mean(pre_hook_times),
                "median": statistics.median(pre_hook_times),
                "min": min(pre_hook_times),
                "max": max(pre_hook_times)
            },
            "post_hook_times_ms": {
                "average": statistics.mean(post_hook_times),
                "median": statistics.median(post_hook_times),
                "min": min(post_hook_times),
                "max": max(post_hook_times)
            }
        }
        
        avg_pre_time = statistics.mean(pre_hook_times)
        avg_post_time = statistics.mean(post_hook_times)
        
        rprint(f"  PreHook avg: [green]{avg_pre_time:.2f}ms[/green]")
        rprint(f"  PostHook avg: [green]{avg_post_time:.2f}ms[/green]")
        rprint(f"  Security violations: [yellow]{security_violations}[/yellow]")
    
    async def benchmark_messaging_system(self, num_agents: int, num_messages: int):
        """Benchmark messaging system performance."""
        rprint(f"[bold]📡 Benchmarking Messaging System ({num_messages} messages)[/bold]")
        
        agents = list(self.orchestrator.lifecycle_manager.active_agents)
        if len(agents) < 2:
            rprint("[red]Need at least 2 agents for messaging benchmark[/red]")
            return
        
        message_times = []
        successful_messages = 0
        
        for i in range(num_messages):
            sender = agents[i % len(agents)]
            receiver = agents[(i + 1) % len(agents)]
            
            start_time = time.time()
            message_id = await self.orchestrator.messaging_service.send_lifecycle_message(
                message_type=self.orchestrator.messaging_service.MessageType.HEARTBEAT_REQUEST,
                from_agent=str(sender),
                to_agent=str(receiver),
                payload={"benchmark_id": i, "timestamp": datetime.utcnow().isoformat()}
            )
            message_time = (time.time() - start_time) * 1000
            message_times.append(message_time)
            
            if message_id:
                successful_messages += 1
        
        self.results["benchmarks"]["messaging_system"] = {
            "total_messages": num_messages,
            "successful_messages": successful_messages,
            "success_rate": successful_messages / num_messages,
            "times_ms": {
                "average": statistics.mean(message_times),
                "median": statistics.median(message_times),
                "min": min(message_times),
                "max": max(message_times)
            }
        }
        
        avg_time = statistics.mean(message_times)
        success_rate = successful_messages / num_messages * 100
        
        rprint(f"  Average time: [green]{avg_time:.2f}ms[/green]")
        rprint(f"  Success rate: [green]{success_rate:.1f}%[/green]")
    
    async def benchmark_system_integration(self):
        """Benchmark the complete system integration."""
        rprint("[bold]🔄 Benchmarking System Integration[/bold]")
        
        start_time = time.time()
        demo_results = await self.orchestrator.demonstrate_complete_lifecycle()
        integration_time = (time.time() - start_time) * 1000
        
        self.results["benchmarks"]["system_integration"] = {
            "integration_time_ms": integration_time,
            "demo_success": demo_results["success"],
            "demo_duration_seconds": demo_results.get("duration_seconds", 0),
            "steps_completed": len(demo_results.get("steps_completed", [])),
            "errors": len(demo_results.get("errors", [])),
            "demo_metrics": demo_results.get("metrics", {}).get("demonstration", {})
        }
        
        rprint(f"  Integration time: [green]{integration_time:.2f}ms[/green]")
        rprint(f"  Demo success: [{'green' if demo_results['success'] else 'red'}]{demo_results['success']}[/{'green' if demo_results['success'] else 'red'}]")
        rprint(f"  Steps completed: [green]{len(demo_results.get('steps_completed', []))}[/green]")
    
    def _generate_summary(self):
        """Generate benchmark summary."""
        benchmarks = self.results["benchmarks"]
        
        # Task assignment summary (primary focus)
        task_assignment = benchmarks.get("task_assignment", {})
        target_performance = task_assignment.get("target_performance", {})
        
        summary = {
            "overall_success": True,
            "primary_target_met": target_performance.get("target_met", False),
            "key_metrics": {
                "agent_registration_avg_ms": benchmarks.get("agent_registration", {}).get("times_ms", {}).get("average", 0),
                "task_assignment_avg_ms": task_assignment.get("times_ms", {}).get("average", 0),
                "task_assignment_target_met": target_performance.get("target_met", False),
                "task_assignment_under_target_pct": target_performance.get("percentage_under_target", 0),
                "hook_system_avg_ms": benchmarks.get("hook_system", {}).get("pre_hook_times_ms", {}).get("average", 0),
                "messaging_avg_ms": benchmarks.get("messaging_system", {}).get("times_ms", {}).get("average", 0)
            },
            "performance_grade": self._calculate_performance_grade()
        }
        
        self.results["summary"] = summary
    
    def _calculate_performance_grade(self) -> str:
        """Calculate overall performance grade."""
        benchmarks = self.results["benchmarks"]
        
        # Primary criteria: Task assignment under 500ms
        task_assignment = benchmarks.get("task_assignment", {})
        target_pct = task_assignment.get("target_performance", {}).get("percentage_under_target", 0)
        
        if target_pct >= 99:
            return "A+"
        elif target_pct >= 95:
            return "A"
        elif target_pct >= 90:
            return "B+"
        elif target_pct >= 80:
            return "B"
        elif target_pct >= 70:
            return "C"
        else:
            return "F"
    
    def display_results(self):
        """Display comprehensive benchmark results."""
        rprint("\n[bold blue]📊 VERTICAL SLICE PERFORMANCE BENCHMARK RESULTS[/bold blue]")
        rprint("=" * 60)
        
        summary = self.results["summary"]
        
        # Overall Summary
        grade = summary["performance_grade"]
        grade_color = {
            "A+": "bright_green", "A": "green", "B+": "yellow", 
            "B": "yellow", "C": "orange", "F": "red"
        }.get(grade, "white")
        
        rprint(f"\n[bold]Overall Performance Grade: [{grade_color}]{grade}[/{grade_color}][/bold]")
        rprint(f"Primary Target Met: [{'green' if summary['primary_target_met'] else 'red'}]{summary['primary_target_met']}[/{'green' if summary['primary_target_met'] else 'red'}]")
        
        # Key Metrics Table
        table = Table(title="Key Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Target", style="yellow")
        table.add_column("Status", style="green")
        
        metrics = summary["key_metrics"]
        
        table.add_row(
            "Agent Registration",
            f"{metrics['agent_registration_avg_ms']:.2f}ms",
            "<2000ms",
            "✅" if metrics['agent_registration_avg_ms'] < 2000 else "❌"
        )
        
        table.add_row(
            "Task Assignment (PRIMARY)",
            f"{metrics['task_assignment_avg_ms']:.2f}ms",
            "<500ms",
            "✅" if metrics['task_assignment_target_met'] else "❌"
        )
        
        table.add_row(
            "Task Assignment Success Rate",
            f"{metrics['task_assignment_under_target_pct']:.1f}%",
            "100%",
            "✅" if metrics['task_assignment_under_target_pct'] >= 95 else "❌"
        )
        
        table.add_row(
            "Hook System",
            f"{metrics['hook_system_avg_ms']:.2f}ms",
            "<100ms",
            "✅" if metrics['hook_system_avg_ms'] < 100 else "❌"
        )
        
        table.add_row(
            "Messaging System",
            f"{metrics['messaging_avg_ms']:.2f}ms",
            "<50ms",
            "✅" if metrics['messaging_avg_ms'] < 50 else "❌"
        )
        
        console.print(table)
        
        # Detailed Results
        if self.results["benchmarks"].get("task_assignment"):
            task_data = self.results["benchmarks"]["task_assignment"]
            rprint(f"\n[bold]📈 Task Assignment Details (Primary Target)[/bold]")
            rprint(f"  Total Tasks: {task_data['total_tasks']}")
            rprint(f"  Success Rate: {task_data['success_rate'] * 100:.1f}%")
            rprint(f"  Average Time: {task_data['times_ms']['average']:.2f}ms")
            rprint(f"  P95 Time: {task_data['times_ms']['p95']:.2f}ms")
            rprint(f"  P99 Time: {task_data['times_ms']['p99']:.2f}ms")
            rprint(f"  Under Target: {task_data['target_performance']['percentage_under_target']:.1f}%")
    
    def save_results(self, filename: str):
        """Save benchmark results to file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        rprint(f"[green]Results saved to {filename}[/green]")
    
    async def _cleanup_test_data(self):
        """Clean up test data from database."""
        async with get_async_session() as db:
            # Delete benchmark agents
            await db.execute(
                db.query(Agent).filter(Agent.name.like("benchmark_%")).delete()
            )
            
            # Delete benchmark tasks
            await db.execute(
                db.query(Task).filter(Task.title.like("Benchmark Task%")).delete()
            )
            
            await db.commit()


async def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(description="Vertical Slice Performance Benchmark")
    parser.add_argument("--agents", type=int, default=10, help="Number of agents to create")
    parser.add_argument("--tasks", type=int, default=50, help="Number of tasks to test")
    parser.add_argument("--output", type=str, default=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", help="Output file")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with fewer iterations")
    
    args = parser.parse_args()
    
    if args.quick:
        args.agents = min(args.agents, 5)
        args.tasks = min(args.tasks, 20)
    
    benchmark = VerticalSliceBenchmark()
    
    try:
        await benchmark.setup()
        results = await benchmark.run_all_benchmarks(args.agents, args.tasks)
        benchmark.display_results()
        benchmark.save_results(args.output)
        
        # Print final verdict
        summary = results["summary"]
        if summary["primary_target_met"]:
            rprint("\n[bold green]🎉 PRIMARY TARGET ACHIEVED: Task assignment <500ms! 🎉[/bold green]")
        else:
            rprint("\n[bold red]❌ PRIMARY TARGET MISSED: Task assignment >=500ms[/bold red]")
            
    except Exception as e:
        rprint(f"[bold red]❌ Benchmark failed: {str(e)}[/bold red]")
        raise
    finally:
        await benchmark.teardown()


if __name__ == "__main__":
    asyncio.run(main())