#!/usr/bin/env python3
"""
Performance and Reliability Testing Framework

Comprehensive testing for multi-CLI agent system performance, scalability,
and reliability under various load conditions and failure scenarios.

This framework validates:
- System performance under normal and peak loads
- Scalability with increasing number of agents and tasks
- Reliability and stability during extended operations
- Memory usage and resource management
- Recovery from various failure scenarios
- Performance degradation patterns
"""

import asyncio
import time
import psutil
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import pytest
import threading
import queue
import random
import gc
import tracemalloc
from contextlib import asynccontextmanager
import concurrent.futures

class LoadPattern(Enum):
    """Different load patterns for testing."""
    CONSTANT = "constant"
    RAMP_UP = "ramp_up"
    SPIKE = "spike"
    STEP = "step"
    RANDOM = "random"
    BURST = "burst"

class PerformanceMetric(Enum):
    """Performance metrics to track."""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    QUEUE_SIZE = "queue_size"
    ERROR_RATE = "error_rate"
    AGENT_UTILIZATION = "agent_utilization"

@dataclass
class PerformanceTarget:
    """Performance targets for validation."""
    max_response_time: float = 5.0  # seconds
    min_throughput: float = 10.0  # operations per second
    max_cpu_usage: float = 80.0  # percentage
    max_memory_usage: float = 1024.0  # MB
    max_error_rate: float = 5.0  # percentage
    max_queue_size: int = 1000

@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    test_name: str
    pattern: LoadPattern
    duration: int  # seconds
    max_concurrent_tasks: int
    task_generation_rate: float  # tasks per second
    agent_count: int
    ramp_up_time: int = 60  # seconds
    cool_down_time: int = 30  # seconds

@dataclass
class PerformanceResult:
    """Results from performance testing."""
    test_name: str
    start_time: float
    end_time: float
    duration: float
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    summary_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    targets_met: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

class MockTaskGenerator:
    """Generates mock tasks for performance testing."""
    
    def __init__(self):
        self.task_templates = [
            {
                "type": "code_analysis",
                "complexity": "low",
                "expected_duration": 1.0,
                "resources_required": ["cpu"]
            },
            {
                "type": "implementation",
                "complexity": "medium", 
                "expected_duration": 3.0,
                "resources_required": ["cpu", "memory"]
            },
            {
                "type": "testing",
                "complexity": "medium",
                "expected_duration": 2.0,
                "resources_required": ["cpu"]
            },
            {
                "type": "documentation",
                "complexity": "low",
                "expected_duration": 1.5,
                "resources_required": ["cpu"]
            },
            {
                "type": "optimization",
                "complexity": "high",
                "expected_duration": 5.0,
                "resources_required": ["cpu", "memory"]
            }
        ]
    
    def generate_task(self, task_id: str) -> Dict[str, Any]:
        """Generate a random task for testing."""
        template = random.choice(self.task_templates)
        
        return {
            "task_id": task_id,
            "type": template["type"],
            "complexity": template["complexity"],
            "expected_duration": template["expected_duration"] * (0.8 + 0.4 * random.random()),
            "payload": {
                "data_size": random.randint(100, 10000),
                "iteration_count": random.randint(10, 100),
                "nested_level": random.randint(1, 5)
            },
            "metadata": {
                "priority": random.choice(["low", "normal", "high"]),
                "created_at": time.time()
            }
        }

class PerformanceAgent:
    """High-performance mock agent for load testing."""
    
    def __init__(self, agent_id: str, processing_capacity: float = 1.0):
        self.agent_id = agent_id
        self.processing_capacity = processing_capacity
        self.is_busy = False
        self.current_task = None
        self.tasks_completed = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        self.performance_history = []
        self.resource_usage = {
            "cpu_time": 0.0,
            "memory_peak": 0.0,
            "io_operations": 0
        }
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task with realistic performance characteristics."""
        if self.is_busy:
            return {
                "status": "rejected",
                "reason": "agent_busy",
                "agent_id": self.agent_id
            }
        
        self.is_busy = True
        self.current_task = task
        start_time = time.time()
        
        try:
            # Simulate processing based on task complexity and agent capacity
            base_duration = task["expected_duration"]
            actual_duration = base_duration / self.processing_capacity
            
            # Add some realistic variability
            actual_duration *= (0.8 + 0.4 * random.random())
            
            # Simulate different types of processing
            await self._simulate_task_processing(task, actual_duration)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Update metrics
            self.tasks_completed += 1
            self.total_processing_time += processing_time
            
            # Record performance data
            self.performance_history.append({
                "task_id": task["task_id"],
                "task_type": task["type"],
                "processing_time": processing_time,
                "timestamp": start_time,
                "cpu_usage": random.uniform(20, 80),  # Simulated CPU usage
                "memory_usage": random.uniform(50, 200)  # Simulated memory usage in MB
            })
            
            return {
                "status": "completed",
                "agent_id": self.agent_id,
                "task_id": task["task_id"],
                "processing_time": processing_time,
                "result": {
                    "output_size": random.randint(100, 1000),
                    "quality_score": random.uniform(0.8, 1.0)
                }
            }
        
        except Exception as e:
            self.error_count += 1
            return {
                "status": "failed",
                "agent_id": self.agent_id,
                "task_id": task["task_id"],
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        
        finally:
            self.is_busy = False
            self.current_task = None
    
    async def _simulate_task_processing(self, task: Dict[str, Any], duration: float):
        """Simulate realistic task processing with resource usage."""
        task_type = task["type"]
        
        if task_type == "code_analysis":
            # CPU-intensive simulation
            await self._simulate_cpu_work(duration * 0.8)
            await asyncio.sleep(duration * 0.2)
        
        elif task_type == "implementation":
            # Mixed CPU and I/O simulation
            await self._simulate_cpu_work(duration * 0.6)
            await self._simulate_io_work(duration * 0.4)
        
        elif task_type == "testing":
            # Burst patterns simulation
            for _ in range(random.randint(3, 8)):
                await self._simulate_cpu_work(duration * 0.1)
                await asyncio.sleep(duration * 0.02)
        
        elif task_type == "optimization":
            # Heavy processing simulation
            await self._simulate_memory_work(duration * 0.3)
            await self._simulate_cpu_work(duration * 0.7)
        
        else:
            # Default processing
            await asyncio.sleep(duration)
    
    async def _simulate_cpu_work(self, duration: float):
        """Simulate CPU-intensive work."""
        start_time = time.time()
        while time.time() - start_time < duration:
            # Simulate CPU work with brief yields
            for _ in range(1000):
                _ = sum(range(100))
            await asyncio.sleep(0.001)  # Brief yield
    
    async def _simulate_io_work(self, duration: float):
        """Simulate I/O work."""
        await asyncio.sleep(duration)
    
    async def _simulate_memory_work(self, duration: float):
        """Simulate memory-intensive work."""
        start_time = time.time()
        memory_data = []
        
        while time.time() - start_time < duration:
            # Allocate and deallocate memory
            chunk = list(range(random.randint(1000, 10000)))
            memory_data.append(chunk)
            
            if len(memory_data) > 10:
                memory_data.pop(0)  # Release some memory
            
            await asyncio.sleep(0.01)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the agent."""
        if not self.performance_history:
            return {"no_data": True}
        
        processing_times = [p["processing_time"] for p in self.performance_history]
        cpu_usages = [p["cpu_usage"] for p in self.performance_history]
        memory_usages = [p["memory_usage"] for p in self.performance_history]
        
        return {
            "agent_id": self.agent_id,
            "tasks_completed": self.tasks_completed,
            "error_count": self.error_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": statistics.mean(processing_times),
            "median_processing_time": statistics.median(processing_times),
            "p95_processing_time": statistics.quantiles(processing_times, n=20)[18] if len(processing_times) > 20 else max(processing_times),
            "average_cpu_usage": statistics.mean(cpu_usages),
            "average_memory_usage": statistics.mean(memory_usages),
            "throughput": self.tasks_completed / self.total_processing_time if self.total_processing_time > 0 else 0
        }

class SystemMonitor:
    """Monitors system performance during testing."""
    
    def __init__(self, sample_interval: float = 1.0):
        self.sample_interval = sample_interval
        self.monitoring = False
        self.metrics_history = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start system monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "memory_percent": psutil.virtual_memory().percent,
                    "memory_used_mb": psutil.virtual_memory().used / (1024 * 1024),
                    "disk_io_read": psutil.disk_io_counters().read_bytes if psutil.disk_io_counters() else 0,
                    "disk_io_write": psutil.disk_io_counters().write_bytes if psutil.disk_io_counters() else 0,
                    "network_sent": psutil.net_io_counters().bytes_sent if psutil.net_io_counters() else 0,
                    "network_recv": psutil.net_io_counters().bytes_recv if psutil.net_io_counters() else 0,
                    "process_count": len(psutil.pids()),
                    "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
                }
                
                self.metrics_history.append(metrics)
                
                # Keep history manageable
                if len(self.metrics_history) > 10000:
                    self.metrics_history = self.metrics_history[-5000:]
                
                time.sleep(self.sample_interval)
            
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.sample_interval)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        if not self.metrics_history:
            return {"no_data": True}
        
        # Calculate summary statistics for each metric
        summary = {}
        metric_names = ["cpu_percent", "memory_percent", "memory_used_mb", "load_average"]
        
        for metric in metric_names:
            values = [m[metric] for m in self.metrics_history if metric in m]
            if values:
                summary[metric] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0
                }
        
        return summary

class LoadTestOrchestrator:
    """Orchestrates load testing scenarios."""
    
    def __init__(self):
        self.agents = {}
        self.task_generator = MockTaskGenerator()
        self.system_monitor = SystemMonitor()
        self.active_tasks = {}
        self.completed_tasks = []
        self.failed_tasks = []
        self.performance_targets = PerformanceTarget()
    
    def register_agent(self, agent: PerformanceAgent):
        """Register an agent for load testing."""
        self.agents[agent.agent_id] = agent
    
    def create_agents(self, count: int, capacity_range: Tuple[float, float] = (0.8, 1.2)):
        """Create multiple agents for testing."""
        for i in range(count):
            capacity = random.uniform(capacity_range[0], capacity_range[1])
            agent = PerformanceAgent(f"agent_{i}", capacity)
            self.register_agent(agent)
    
    async def execute_load_test(self, config: LoadTestConfig) -> PerformanceResult:
        """Execute a load test scenario."""
        print(f"🚀 Starting load test: {config.test_name}")
        
        result = PerformanceResult(
            test_name=config.test_name,
            start_time=time.time(),
            end_time=0.0,
            duration=0.0
        )
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        # Start memory tracking
        tracemalloc.start()
        
        try:
            if config.pattern == LoadPattern.CONSTANT:
                await self._execute_constant_load(config, result)
            elif config.pattern == LoadPattern.RAMP_UP:
                await self._execute_ramp_up_load(config, result)
            elif config.pattern == LoadPattern.SPIKE:
                await self._execute_spike_load(config, result)
            elif config.pattern == LoadPattern.STEP:
                await self._execute_step_load(config, result)
            elif config.pattern == LoadPattern.RANDOM:
                await self._execute_random_load(config, result)
            elif config.pattern == LoadPattern.BURST:
                await self._execute_burst_load(config, result)
        
        except Exception as e:
            result.errors.append(str(e))
        
        finally:
            result.end_time = time.time()
            result.duration = result.end_time - result.start_time
            
            # Stop monitoring
            self.system_monitor.stop_monitoring()
            
            # Get memory snapshot
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Collect final metrics
            await self._collect_performance_metrics(result)
            
            # Add memory metrics
            result.metrics["memory_current_mb"] = [current / (1024 * 1024)]
            result.metrics["memory_peak_mb"] = [peak / (1024 * 1024)]
            
            # Calculate summary statistics
            self._calculate_summary_stats(result)
            
            # Validate against targets
            self._validate_performance_targets(result)
            
            print(f"✅ Load test completed: {config.test_name}")
        
        return result
    
    async def _execute_constant_load(self, config: LoadTestConfig, result: PerformanceResult):
        """Execute constant load pattern."""
        tasks = []
        task_count = 0
        
        end_time = time.time() + config.duration
        
        while time.time() < end_time:
            # Generate tasks at specified rate
            if len(self.active_tasks) < config.max_concurrent_tasks:
                task_id = f"task_{task_count}"
                task = self.task_generator.generate_task(task_id)
                
                # Submit task
                task_future = asyncio.create_task(self._process_task_with_agent(task))
                self.active_tasks[task_id] = {
                    "task": task,
                    "future": task_future,
                    "start_time": time.time()
                }
                
                task_count += 1
            
            # Wait based on generation rate
            await asyncio.sleep(1.0 / config.task_generation_rate)
            
            # Clean up completed tasks
            await self._cleanup_completed_tasks(result)
        
        # Wait for remaining tasks
        await self._wait_for_active_tasks(result)
    
    async def _execute_ramp_up_load(self, config: LoadTestConfig, result: PerformanceResult):
        """Execute ramp-up load pattern."""
        ramp_duration = min(config.ramp_up_time, config.duration // 2)
        steady_duration = config.duration - ramp_duration - config.cool_down_time
        
        task_count = 0
        
        # Ramp-up phase
        for phase_time in range(ramp_duration):
            progress = phase_time / ramp_duration
            current_rate = config.task_generation_rate * progress
            
            if current_rate > 0:
                await self._generate_tasks_for_interval(current_rate, 1.0, task_count, config, result)
                task_count += int(current_rate)
        
        # Steady phase
        for _ in range(steady_duration):
            await self._generate_tasks_for_interval(config.task_generation_rate, 1.0, task_count, config, result)
            task_count += int(config.task_generation_rate)
        
        # Cool-down phase
        for phase_time in range(config.cool_down_time):
            progress = 1.0 - (phase_time / config.cool_down_time)
            current_rate = config.task_generation_rate * progress
            
            if current_rate > 0:
                await self._generate_tasks_for_interval(current_rate, 1.0, task_count, config, result)
                task_count += int(current_rate)
        
        await self._wait_for_active_tasks(result)
    
    async def _execute_spike_load(self, config: LoadTestConfig, result: PerformanceResult):
        """Execute spike load pattern."""
        normal_rate = config.task_generation_rate * 0.3
        spike_rate = config.task_generation_rate * 3.0
        spike_duration = 10  # 10 second spikes
        
        task_count = 0
        end_time = time.time() + config.duration
        
        while time.time() < end_time:
            # Normal load
            await self._generate_tasks_for_interval(normal_rate, 20.0, task_count, config, result)
            task_count += int(normal_rate * 20)
            
            if time.time() < end_time:
                # Spike load
                await self._generate_tasks_for_interval(spike_rate, spike_duration, task_count, config, result)
                task_count += int(spike_rate * spike_duration)
        
        await self._wait_for_active_tasks(result)
    
    async def _execute_step_load(self, config: LoadTestConfig, result: PerformanceResult):
        """Execute step load pattern."""
        steps = 5
        step_duration = config.duration // steps
        
        task_count = 0
        
        for step in range(steps):
            step_rate = config.task_generation_rate * (step + 1) / steps
            await self._generate_tasks_for_interval(step_rate, step_duration, task_count, config, result)
            task_count += int(step_rate * step_duration)
        
        await self._wait_for_active_tasks(result)
    
    async def _execute_random_load(self, config: LoadTestConfig, result: PerformanceResult):
        """Execute random load pattern."""
        task_count = 0
        end_time = time.time() + config.duration
        
        while time.time() < end_time:
            # Random rate between 10% and 150% of configured rate
            random_rate = config.task_generation_rate * random.uniform(0.1, 1.5)
            interval_duration = random.uniform(1.0, 5.0)
            
            await self._generate_tasks_for_interval(random_rate, interval_duration, task_count, config, result)
            task_count += int(random_rate * interval_duration)
        
        await self._wait_for_active_tasks(result)
    
    async def _execute_burst_load(self, config: LoadTestConfig, result: PerformanceResult):
        """Execute burst load pattern."""
        burst_size = config.max_concurrent_tasks
        burst_interval = 30  # 30 seconds between bursts
        
        task_count = 0
        end_time = time.time() + config.duration
        
        while time.time() < end_time:
            # Create burst of tasks
            burst_tasks = []
            for _ in range(burst_size):
                task_id = f"task_{task_count}"
                task = self.task_generator.generate_task(task_id)
                burst_tasks.append(task)
                task_count += 1
            
            # Submit all burst tasks
            for task in burst_tasks:
                if len(self.active_tasks) < config.max_concurrent_tasks:
                    task_future = asyncio.create_task(self._process_task_with_agent(task))
                    self.active_tasks[task["task_id"]] = {
                        "task": task,
                        "future": task_future,
                        "start_time": time.time()
                    }
            
            # Wait for burst interval
            await asyncio.sleep(burst_interval)
            await self._cleanup_completed_tasks(result)
        
        await self._wait_for_active_tasks(result)
    
    async def _generate_tasks_for_interval(self, rate: float, duration: float, task_count_start: int, 
                                         config: LoadTestConfig, result: PerformanceResult):
        """Generate tasks at specified rate for given duration."""
        interval_end = time.time() + duration
        task_count = task_count_start
        
        while time.time() < interval_end:
            if len(self.active_tasks) < config.max_concurrent_tasks:
                task_id = f"task_{task_count}"
                task = self.task_generator.generate_task(task_id)
                
                task_future = asyncio.create_task(self._process_task_with_agent(task))
                self.active_tasks[task_id] = {
                    "task": task,
                    "future": task_future,
                    "start_time": time.time()
                }
                
                task_count += 1
            
            # Wait based on generation rate
            if rate > 0:
                await asyncio.sleep(1.0 / rate)
            else:
                await asyncio.sleep(1.0)
            
            await self._cleanup_completed_tasks(result)
    
    async def _process_task_with_agent(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process task using available agent."""
        # Find available agent
        available_agents = [agent for agent in self.agents.values() if not agent.is_busy]
        
        if not available_agents:
            # All agents busy, wait briefly and retry
            await asyncio.sleep(0.1)
            available_agents = [agent for agent in self.agents.values() if not agent.is_busy]
        
        if available_agents:
            # Select agent (simple round-robin)
            agent = min(available_agents, key=lambda a: a.tasks_completed)
            return await agent.process_task(task)
        else:
            return {
                "status": "failed",
                "task_id": task["task_id"],
                "error": "no_available_agents"
            }
    
    async def _cleanup_completed_tasks(self, result: PerformanceResult):
        """Clean up completed tasks and record metrics."""
        completed_task_ids = []
        
        for task_id, task_info in self.active_tasks.items():
            if task_info["future"].done():
                try:
                    task_result = task_info["future"].result()
                    
                    # Record metrics
                    end_time = time.time()
                    response_time = end_time - task_info["start_time"]
                    
                    if "response_time" not in result.metrics:
                        result.metrics["response_time"] = []
                    result.metrics["response_time"].append(response_time)
                    
                    if task_result["status"] == "completed":
                        self.completed_tasks.append({
                            "task": task_info["task"],
                            "result": task_result,
                            "response_time": response_time
                        })
                    else:
                        self.failed_tasks.append({
                            "task": task_info["task"],
                            "result": task_result,
                            "response_time": response_time
                        })
                
                except Exception as e:
                    self.failed_tasks.append({
                        "task": task_info["task"],
                        "error": str(e),
                        "response_time": time.time() - task_info["start_time"]
                    })
                
                completed_task_ids.append(task_id)
        
        # Remove completed tasks
        for task_id in completed_task_ids:
            del self.active_tasks[task_id]
    
    async def _wait_for_active_tasks(self, result: PerformanceResult):
        """Wait for all active tasks to complete."""
        while self.active_tasks:
            await self._cleanup_completed_tasks(result)
            if self.active_tasks:
                await asyncio.sleep(0.1)
    
    async def _collect_performance_metrics(self, result: PerformanceResult):
        """Collect performance metrics from agents and system monitor."""
        # Collect agent metrics
        for agent in self.agents.values():
            agent_summary = agent.get_performance_summary()
            
            for metric, value in agent_summary.items():
                if isinstance(value, (int, float)):
                    if metric not in result.metrics:
                        result.metrics[metric] = []
                    result.metrics[metric].append(value)
        
        # Collect system metrics
        system_summary = self.system_monitor.get_metrics_summary()
        for metric, stats in system_summary.items():
            if isinstance(stats, dict) and "mean" in stats:
                if metric not in result.metrics:
                    result.metrics[metric] = []
                result.metrics[metric].append(stats["mean"])
        
        # Calculate throughput
        total_completed = len(self.completed_tasks)
        if result.duration > 0:
            throughput = total_completed / result.duration
            if "throughput" not in result.metrics:
                result.metrics["throughput"] = []
            result.metrics["throughput"].append(throughput)
        
        # Calculate error rate
        total_tasks = len(self.completed_tasks) + len(self.failed_tasks)
        if total_tasks > 0:
            error_rate = (len(self.failed_tasks) / total_tasks) * 100
            if "error_rate" not in result.metrics:
                result.metrics["error_rate"] = []
            result.metrics["error_rate"].append(error_rate)
    
    def _calculate_summary_stats(self, result: PerformanceResult):
        """Calculate summary statistics for all metrics."""
        for metric, values in result.metrics.items():
            if values:
                result.summary_stats[metric] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "count": len(values)
                }
                
                if len(values) >= 20:
                    result.summary_stats[metric]["p95"] = statistics.quantiles(values, n=20)[18]
                    result.summary_stats[metric]["p99"] = statistics.quantiles(values, n=100)[98]
    
    def _validate_performance_targets(self, result: PerformanceResult):
        """Validate performance against targets."""
        targets = self.performance_targets
        
        # Response time check
        if "response_time" in result.summary_stats:
            avg_response_time = result.summary_stats["response_time"]["mean"]
            result.targets_met["response_time"] = avg_response_time <= targets.max_response_time
        
        # Throughput check
        if "throughput" in result.summary_stats:
            avg_throughput = result.summary_stats["throughput"]["mean"]
            result.targets_met["throughput"] = avg_throughput >= targets.min_throughput
        
        # CPU usage check
        if "cpu_percent" in result.summary_stats:
            avg_cpu = result.summary_stats["cpu_percent"]["mean"]
            result.targets_met["cpu_usage"] = avg_cpu <= targets.max_cpu_usage
        
        # Memory usage check
        if "memory_used_mb" in result.summary_stats:
            avg_memory = result.summary_stats["memory_used_mb"]["mean"]
            result.targets_met["memory_usage"] = avg_memory <= targets.max_memory_usage
        
        # Error rate check
        if "error_rate" in result.summary_stats:
            avg_error_rate = result.summary_stats["error_rate"]["mean"]
            result.targets_met["error_rate"] = avg_error_rate <= targets.max_error_rate

def create_performance_test_configs() -> List[LoadTestConfig]:
    """Create comprehensive performance test configurations."""
    
    configs = [
        # Baseline performance test
        LoadTestConfig(
            test_name="Baseline Performance",
            pattern=LoadPattern.CONSTANT,
            duration=60,
            max_concurrent_tasks=10,
            task_generation_rate=5.0,
            agent_count=3
        ),
        
        # Ramp-up load test
        LoadTestConfig(
            test_name="Ramp-up Load Test",
            pattern=LoadPattern.RAMP_UP,
            duration=180,
            max_concurrent_tasks=50,
            task_generation_rate=20.0,
            agent_count=5,
            ramp_up_time=60,
            cool_down_time=30
        ),
        
        # Spike load test
        LoadTestConfig(
            test_name="Spike Load Test",
            pattern=LoadPattern.SPIKE,
            duration=300,
            max_concurrent_tasks=100,
            task_generation_rate=15.0,
            agent_count=8
        ),
        
        # High concurrency test
        LoadTestConfig(
            test_name="High Concurrency Test",
            pattern=LoadPattern.CONSTANT,
            duration=120,
            max_concurrent_tasks=200,
            task_generation_rate=50.0,
            agent_count=15
        ),
        
        # Endurance test
        LoadTestConfig(
            test_name="Endurance Test",
            pattern=LoadPattern.CONSTANT,
            duration=600,  # 10 minutes
            max_concurrent_tasks=30,
            task_generation_rate=10.0,
            agent_count=5
        ),
        
        # Burst load test
        LoadTestConfig(
            test_name="Burst Load Test",
            pattern=LoadPattern.BURST,
            duration=180,
            max_concurrent_tasks=100,
            task_generation_rate=25.0,
            agent_count=10
        )
    ]
    
    return configs

class PerformanceTestSuite:
    """Main test suite for performance and reliability testing."""
    
    def __init__(self):
        self.orchestrator = LoadTestOrchestrator()
        self.test_results = []
    
    async def run_comprehensive_performance_tests(self) -> Dict[str, Any]:
        """Run comprehensive performance and reliability tests."""
        suite_results = {
            "test_suite": "Performance and Reliability",
            "start_time": time.time(),
            "tests_executed": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "detailed_results": []
        }
        
        configs = create_performance_test_configs()
        
        for config in configs:
            print(f"🏃 Executing performance test: {config.test_name}")
            
            try:
                # Create agents for this test
                self.orchestrator.agents.clear()
                self.orchestrator.create_agents(config.agent_count)
                
                # Execute load test
                result = await self.orchestrator.execute_load_test(config)
                suite_results["detailed_results"].append(result)
                suite_results["tests_executed"] += 1
                
                # Determine if test passed based on targets
                targets_passed = sum(1 for passed in result.targets_met.values() if passed)
                total_targets = len(result.targets_met)
                
                if total_targets == 0 or targets_passed / total_targets >= 0.8:
                    suite_results["tests_passed"] += 1
                    print(f"✅ {config.test_name} - PASSED")
                else:
                    suite_results["tests_failed"] += 1
                    print(f"❌ {config.test_name} - FAILED")
                
                # Clear completed/failed task lists for next test
                self.orchestrator.completed_tasks.clear()
                self.orchestrator.failed_tasks.clear()
                
                # Brief pause between tests
                await asyncio.sleep(5)
            
            except Exception as e:
                suite_results["tests_failed"] += 1
                suite_results["detailed_results"].append({
                    "test_name": config.test_name,
                    "status": "error",
                    "error": str(e)
                })
                print(f"❌ {config.test_name} - ERROR: {str(e)}")
        
        suite_results["end_time"] = time.time()
        suite_results["total_duration"] = suite_results["end_time"] - suite_results["start_time"]
        
        return suite_results

# Pytest integration
@pytest.fixture
async def performance_test_suite():
    """Pytest fixture for performance testing."""
    suite = PerformanceTestSuite()
    yield suite

@pytest.mark.asyncio
async def test_baseline_performance(performance_test_suite):
    """Test baseline performance."""
    suite = performance_test_suite
    
    config = LoadTestConfig(
        test_name="Pytest Baseline",
        pattern=LoadPattern.CONSTANT,
        duration=30,
        max_concurrent_tasks=5,
        task_generation_rate=2.0,
        agent_count=2
    )
    
    suite.orchestrator.create_agents(config.agent_count)
    result = await suite.orchestrator.execute_load_test(config)
    
    assert result.duration > 0
    assert len(result.metrics) > 0
    assert "response_time" in result.metrics

@pytest.mark.asyncio
async def test_agent_scalability(performance_test_suite):
    """Test system scalability with increasing agents."""
    suite = performance_test_suite
    
    # Test with different agent counts
    agent_counts = [2, 5, 10]
    results = []
    
    for agent_count in agent_counts:
        config = LoadTestConfig(
            test_name=f"Scalability Test - {agent_count} agents",
            pattern=LoadPattern.CONSTANT,
            duration=20,
            max_concurrent_tasks=agent_count * 3,
            task_generation_rate=agent_count * 2.0,
            agent_count=agent_count
        )
        
        suite.orchestrator.agents.clear()
        suite.orchestrator.create_agents(config.agent_count)
        result = await suite.orchestrator.execute_load_test(config)
        results.append(result)
    
    # Verify throughput scales with agents
    throughputs = [r.summary_stats.get("throughput", {}).get("mean", 0) for r in results]
    assert len(throughputs) == 3
    assert throughputs[1] > throughputs[0]  # 5 agents > 2 agents

if __name__ == "__main__":
    async def main():
        """Run performance and reliability tests standalone."""
        print("⚡ Performance and Reliability Testing Suite")
        print("=" * 60)
        
        test_suite = PerformanceTestSuite()
        
        try:
            results = await test_suite.run_comprehensive_performance_tests()
            
            print("\n" + "=" * 60)
            print("📊 PERFORMANCE TEST RESULTS")
            print("=" * 60)
            print(f"Tests Executed: {results['tests_executed']}")
            print(f"Tests Passed: {results['tests_passed']}")
            print(f"Tests Failed: {results['tests_failed']}")
            print(f"Success Rate: {results['tests_passed']/results['tests_executed']*100:.1f}%")
            print(f"Total Duration: {results['total_duration']:.2f}s")
            
            # Save detailed results
            with open('performance_reliability_test_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n📄 Detailed results saved to: performance_reliability_test_results.json")
            
        except Exception as e:
            print(f"❌ Test suite error: {str(e)}")
    
    asyncio.run(main())