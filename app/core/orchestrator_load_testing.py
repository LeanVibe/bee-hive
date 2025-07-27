"""
Agent Orchestrator Load Testing Framework.

Comprehensive performance testing to validate PRD requirements:
- >50 concurrent agents with <10s spawn time
- <500ms orchestration operation latency  
- >95% agent uptime
- >85% task completion rate
- Memory efficiency: <2GB RAM per 10 agents
"""

import asyncio
import time
import uuid
import statistics
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import json

import structlog
from sqlalchemy import select, func

from .orchestrator import AgentOrchestrator, AgentRole, AgentInstance
from .database import get_session
from ..models.agent import Agent, AgentStatus
from ..models.task import Task, TaskStatus, TaskPriority, TaskType
from ..models.performance_metric import PerformanceMetric

logger = structlog.get_logger()


class LoadTestPhase(str, Enum):
    """Load test execution phases."""
    PREPARATION = "preparation"
    AGENT_SPAWNING = "agent_spawning" 
    TASK_DELEGATION = "task_delegation"
    CONCURRENT_OPERATIONS = "concurrent_operations"
    STRESS_TEST = "stress_test"
    CLEANUP = "cleanup"


@dataclass
class OrchestratorLoadTestConfig:
    """Configuration for orchestrator load testing."""
    
    # Agent spawn testing
    max_concurrent_agents: int = 75  # Test above target of 50
    agent_spawn_batch_size: int = 10
    agent_spawn_interval_seconds: float = 2.0
    
    # Task delegation testing  
    tasks_per_agent: int = 5
    task_delegation_concurrency: int = 20
    task_complexity_variation: bool = True
    
    # Performance targets (from PRD)
    max_agent_spawn_time_seconds: float = 10.0
    max_orchestration_latency_ms: float = 500.0
    min_agent_uptime_percent: float = 95.0
    min_task_completion_rate: float = 85.0
    max_memory_gb_per_10_agents: float = 2.0
    
    # Test duration and phases
    preparation_duration_seconds: int = 30
    spawning_duration_seconds: int = 120
    delegation_duration_seconds: int = 180
    concurrent_operations_duration_seconds: int = 300
    stress_test_duration_seconds: int = 120
    cleanup_duration_seconds: int = 60
    
    # Workload simulation
    task_types_distribution: Dict[str, float] = field(default_factory=lambda: {
        "code_review": 0.3,
        "api_development": 0.2,
        "testing": 0.15,
        "documentation": 0.15,
        "deployment": 0.1,
        "planning": 0.1
    })
    
    # System resource monitoring
    resource_monitoring_interval_seconds: float = 5.0
    memory_usage_alert_threshold_gb: float = 8.0


@dataclass
class OrchestratorPerformanceMetrics:
    """Performance metrics for orchestrator testing."""
    
    phase: LoadTestPhase
    timestamp: float
    
    # Agent management metrics
    agents_spawned: int = 0
    agents_active: int = 0
    agents_failed: int = 0
    agent_spawn_times: List[float] = field(default_factory=list)
    agent_uptime_percent: float = 0.0
    
    # Task delegation metrics
    tasks_submitted: int = 0
    tasks_assigned: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    task_delegation_times: List[float] = field(default_factory=list)
    
    # System performance metrics
    orchestrator_response_times: List[float] = field(default_factory=list)
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Orchestration quality metrics
    routing_accuracy: float = 0.0
    load_balancing_effectiveness: float = 0.0
    system_health_score: float = 0.0
    
    def calculate_derived_metrics(self) -> None:
        """Calculate derived performance metrics."""
        # Task completion rate
        self.task_completion_rate = (
            self.tasks_completed / self.tasks_submitted 
            if self.tasks_submitted > 0 else 0.0
        )
        
        # Average spawn time
        self.avg_agent_spawn_time = (
            statistics.mean(self.agent_spawn_times) 
            if self.agent_spawn_times else 0.0
        )
        
        # Average orchestrator response time
        self.avg_orchestrator_response_time = (
            statistics.mean(self.orchestrator_response_times) 
            if self.orchestrator_response_times else 0.0
        )
        
        # P95 response times
        if self.orchestrator_response_times:
            sorted_times = sorted(self.orchestrator_response_times)
            n = len(sorted_times)
            self.p95_orchestrator_response_time = sorted_times[int(n * 0.95)]
        else:
            self.p95_orchestrator_response_time = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        self.calculate_derived_metrics()
        return {
            "phase": self.phase.value,
            "timestamp": self.timestamp,
            "agents_spawned": self.agents_spawned,
            "agents_active": self.agents_active,
            "agents_failed": self.agents_failed,
            "agent_uptime_percent": self.agent_uptime_percent,
            "avg_agent_spawn_time_seconds": self.avg_agent_spawn_time,
            "tasks_submitted": self.tasks_submitted,
            "tasks_assigned": self.tasks_assigned,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "task_completion_rate": getattr(self, 'task_completion_rate', 0.0),
            "avg_orchestrator_response_time_ms": self.avg_orchestrator_response_time * 1000,
            "p95_orchestrator_response_time_ms": getattr(self, 'p95_orchestrator_response_time', 0.0) * 1000,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "routing_accuracy": self.routing_accuracy,
            "load_balancing_effectiveness": self.load_balancing_effectiveness,
            "system_health_score": self.system_health_score
        }


class OrchestratorLoadTestFramework:
    """
    Comprehensive load testing framework for Agent Orchestrator.
    
    Validates all PRD performance targets with realistic workload simulation.
    """
    
    def __init__(
        self,
        config: Optional[OrchestratorLoadTestConfig] = None
    ):
        self.config = config or OrchestratorLoadTestConfig()
        
        # Test infrastructure
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.test_agents: List[str] = []
        self.test_tasks: List[str] = []
        
        # Metrics collection
        self.metrics_history: List[OrchestratorPerformanceMetrics] = []
        self.current_metrics = OrchestratorPerformanceMetrics(
            LoadTestPhase.PREPARATION, time.time()
        )
        self.metrics_lock = asyncio.Lock()
        
        # Test state
        self.test_start_time = 0.0
        self.current_phase = LoadTestPhase.PREPARATION
        self.is_running = False
        
        # Resource monitoring
        self.process = psutil.Process()
        self.baseline_memory_mb = 0.0
    
    async def setup(self) -> None:
        """Setup test environment."""
        try:
            logger.info("Setting up orchestrator load test environment")
            
            # Initialize orchestrator
            self.orchestrator = AgentOrchestrator()
            await self.orchestrator.start()
            
            # Record baseline memory usage
            self.baseline_memory_mb = self.process.memory_info().rss / 1024 / 1024
            
            # Clear any existing test data
            await self._clear_test_data()
            
            logger.info("Orchestrator load test environment setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup orchestrator load test: {e}")
            raise
    
    async def teardown(self) -> None:
        """Cleanup test environment."""
        try:
            # Stop orchestrator
            if self.orchestrator:
                await self.orchestrator.shutdown()
            
            # Clean up test data
            await self._clear_test_data()
            
            logger.info("Orchestrator load test environment cleaned up")
            
        except Exception as e:
            logger.error(f"Error during orchestrator load test teardown: {e}")
    
    async def run_full_load_test(self) -> Dict[str, Any]:
        """Run complete orchestrator load test with all phases."""
        self.test_start_time = time.time()
        self.is_running = True
        
        try:
            # Start metrics collection
            metrics_task = asyncio.create_task(self._metrics_collection_loop())
            
            # Run test phases
            await self._run_preparation_phase()
            await self._run_agent_spawning_phase()
            await self._run_task_delegation_phase()
            await self._run_concurrent_operations_phase()
            await self._run_stress_test_phase()
            await self._run_cleanup_phase()
            
            # Stop metrics collection
            self.is_running = False
            await metrics_task
            
            # Generate test report
            return self._generate_test_report()
            
        except Exception as e:
            logger.error(f"Error during orchestrator load test: {e}")
            self.is_running = False
            raise
    
    async def _run_preparation_phase(self) -> None:
        """Prepare test environment and validate baseline performance."""
        self.current_phase = LoadTestPhase.PREPARATION
        logger.info("Starting orchestrator preparation phase")
        
        # Test basic orchestrator functionality
        test_agent_id = await self._test_basic_agent_spawn()
        if test_agent_id:
            await self._test_basic_task_delegation(test_agent_id)
            await self.orchestrator.shutdown_agent(test_agent_id)
        
        # Warm up system
        await asyncio.sleep(self.config.preparation_duration_seconds)
        
        logger.info("Preparation phase complete")
    
    async def _run_agent_spawning_phase(self) -> None:
        """Test agent spawning performance and capacity."""
        self.current_phase = LoadTestPhase.AGENT_SPAWNING
        logger.info(f"Starting agent spawning phase - target: {self.config.max_concurrent_agents} agents")
        
        start_time = time.time()
        spawned_agents = []
        
        # Spawn agents in batches
        for batch_start in range(0, self.config.max_concurrent_agents, self.config.agent_spawn_batch_size):
            batch_end = min(batch_start + self.config.agent_spawn_batch_size, self.config.max_concurrent_agents)
            batch_size = batch_end - batch_start
            
            # Spawn batch concurrently
            batch_tasks = []
            for i in range(batch_size):
                role = self._select_random_agent_role()
                task = asyncio.create_task(self._spawn_agent_with_timing(role))
                batch_tasks.append(task)
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process batch results
            for result in batch_results:
                if isinstance(result, tuple):
                    agent_id, spawn_time = result
                    if agent_id:
                        spawned_agents.append(agent_id)
                        self.current_metrics.agent_spawn_times.append(spawn_time)
                        self.current_metrics.agents_spawned += 1
                    else:
                        self.current_metrics.agents_failed += 1
                else:
                    self.current_metrics.agents_failed += 1
            
            # Brief pause between batches
            if batch_end < self.config.max_concurrent_agents:
                await asyncio.sleep(self.config.agent_spawn_interval_seconds)
        
        # Wait for all agents to be active
        await self._wait_for_agents_active(spawned_agents)
        
        self.test_agents = spawned_agents
        total_time = time.time() - start_time
        
        logger.info(
            f"Agent spawning phase complete: {len(spawned_agents)} agents spawned in {total_time:.2f}s"
        )
    
    async def _run_task_delegation_phase(self) -> None:
        """Test task delegation performance and routing accuracy."""
        self.current_phase = LoadTestPhase.TASK_DELEGATION
        logger.info("Starting task delegation phase")
        
        if not self.test_agents:
            logger.warning("No agents available for task delegation testing")
            return
        
        # Calculate total tasks to create
        total_tasks = len(self.test_agents) * self.config.tasks_per_agent
        
        # Create and delegate tasks
        delegation_tasks = []
        for i in range(total_tasks):
            task_type = self._select_random_task_type()
            priority = self._select_random_priority()
            
            task = asyncio.create_task(
                self._delegate_task_with_timing(
                    f"Load test task {i}",
                    task_type,
                    priority
                )
            )
            delegation_tasks.append(task)
            
            # Control concurrency
            if len(delegation_tasks) >= self.config.task_delegation_concurrency:
                # Wait for a batch to complete
                done, pending = await asyncio.wait(
                    delegation_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for completed_task in done:
                    try:
                        result = await completed_task
                        if result:
                            task_id, delegation_time = result
                            self.test_tasks.append(task_id)
                            self.current_metrics.task_delegation_times.append(delegation_time)
                            self.current_metrics.tasks_submitted += 1
                        else:
                            self.current_metrics.tasks_failed += 1
                    except Exception as e:
                        logger.error(f"Task delegation failed: {e}")
                        self.current_metrics.tasks_failed += 1
                
                # Update task list
                delegation_tasks = list(pending)
        
        # Wait for remaining tasks
        if delegation_tasks:
            remaining_results = await asyncio.gather(*delegation_tasks, return_exceptions=True)
            for result in remaining_results:
                if isinstance(result, tuple):
                    task_id, delegation_time = result
                    self.test_tasks.append(task_id)
                    self.current_metrics.task_delegation_times.append(delegation_time)
                    self.current_metrics.tasks_submitted += 1
                else:
                    self.current_metrics.tasks_failed += 1
        
        # Wait for task processing to begin
        await asyncio.sleep(30)
        
        logger.info(f"Task delegation phase complete: {len(self.test_tasks)} tasks created")
    
    async def _run_concurrent_operations_phase(self) -> None:
        """Test sustained concurrent operations."""
        self.current_phase = LoadTestPhase.CONCURRENT_OPERATIONS
        logger.info("Starting concurrent operations phase")
        
        # Simulate various orchestrator operations concurrently
        operation_tasks = []
        
        # Add continuous monitoring
        operation_tasks.append(
            asyncio.create_task(self._continuous_system_monitoring())
        )
        
        # Add periodic agent health checks
        operation_tasks.append(
            asyncio.create_task(self._periodic_health_checks())
        )
        
        # Add dynamic task creation
        operation_tasks.append(
            asyncio.create_task(self._dynamic_task_creation())
        )
        
        # Add load balancing operations
        operation_tasks.append(
            asyncio.create_task(self._periodic_load_balancing())
        )
        
        # Run concurrent operations for specified duration
        await asyncio.sleep(self.config.concurrent_operations_duration_seconds)
        
        # Stop all operations
        for task in operation_tasks:
            task.cancel()
        
        await asyncio.gather(*operation_tasks, return_exceptions=True)
        
        logger.info("Concurrent operations phase complete")
    
    async def _run_stress_test_phase(self) -> None:
        """Run stress test to find performance limits."""
        self.current_phase = LoadTestPhase.STRESS_TEST
        logger.info("Starting stress test phase")
        
        # Spawn additional agents beyond target
        stress_agents = []
        try:
            for i in range(25):  # Add 25 more agents (total 100)
                role = self._select_random_agent_role()
                agent_id = await self.orchestrator.spawn_agent(role)
                if agent_id:
                    stress_agents.append(agent_id)
                    await asyncio.sleep(0.5)  # Aggressive spawning
        except Exception as e:
            logger.info(f"Stress test agent limit reached: {e}")
        
        # Create burst of tasks
        burst_tasks = []
        for i in range(100):  # 100 rapid tasks
            task_type = self._select_random_task_type()
            priority = TaskPriority.HIGH  # High priority stress
            
            try:
                task_id = await self.orchestrator.delegate_task(
                    f"Stress test task {i}",
                    task_type,
                    priority
                )
                burst_tasks.append(task_id)
            except Exception as e:
                logger.warning(f"Stress task creation failed: {e}")
                break
            
            await asyncio.sleep(0.1)  # 10 tasks per second
        
        # Monitor system under stress
        await asyncio.sleep(self.config.stress_test_duration_seconds)
        
        # Clean up stress test agents
        for agent_id in stress_agents:
            try:
                await self.orchestrator.shutdown_agent(agent_id)
            except Exception:
                pass
        
        logger.info(f"Stress test phase complete: {len(burst_tasks)} burst tasks created")
    
    async def _run_cleanup_phase(self) -> None:
        """Clean up test environment and collect final metrics."""
        self.current_phase = LoadTestPhase.CLEANUP
        logger.info("Starting cleanup phase")
        
        # Collect final task completion statistics
        await self._collect_final_task_metrics()
        
        # Shutdown all test agents gracefully
        shutdown_tasks = []
        for agent_id in self.test_agents:
            task = asyncio.create_task(self.orchestrator.shutdown_agent(agent_id))
            shutdown_tasks.append(task)
        
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        # Wait for cleanup
        await asyncio.sleep(self.config.cleanup_duration_seconds)
        
        logger.info("Cleanup phase complete")
    
    async def _spawn_agent_with_timing(self, role: AgentRole) -> Tuple[Optional[str], float]:
        """Spawn agent and measure timing."""
        start_time = time.perf_counter()
        
        try:
            agent_id = await self.orchestrator.spawn_agent(role)
            spawn_time = time.perf_counter() - start_time
            return agent_id, spawn_time
        except Exception as e:
            spawn_time = time.perf_counter() - start_time
            logger.error(f"Agent spawn failed in {spawn_time:.3f}s: {e}")
            return None, spawn_time
    
    async def _delegate_task_with_timing(
        self,
        description: str,
        task_type: str,
        priority: TaskPriority
    ) -> Tuple[Optional[str], float]:
        """Delegate task and measure timing."""
        start_time = time.perf_counter()
        
        try:
            task_id = await self.orchestrator.delegate_task(
                description,
                task_type,
                priority
            )
            delegation_time = time.perf_counter() - start_time
            return task_id, delegation_time
        except Exception as e:
            delegation_time = time.perf_counter() - start_time
            logger.error(f"Task delegation failed in {delegation_time:.3f}s: {e}")
            return None, delegation_time
    
    async def _test_basic_agent_spawn(self) -> Optional[str]:
        """Test basic agent spawning functionality."""
        try:
            start_time = time.perf_counter()
            agent_id = await self.orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
            spawn_time = time.perf_counter() - start_time
            
            logger.info(f"Basic agent spawn test: {spawn_time:.3f}s")
            return agent_id
        except Exception as e:
            logger.error(f"Basic agent spawn test failed: {e}")
            return None
    
    async def _test_basic_task_delegation(self, agent_id: str) -> bool:
        """Test basic task delegation functionality."""
        try:
            start_time = time.perf_counter()
            task_id = await self.orchestrator.delegate_task(
                "Test task",
                "testing",
                TaskPriority.MEDIUM
            )
            delegation_time = time.perf_counter() - start_time
            
            logger.info(f"Basic task delegation test: {delegation_time:.3f}s")
            return task_id is not None
        except Exception as e:
            logger.error(f"Basic task delegation test failed: {e}")
            return False
    
    async def _wait_for_agents_active(self, agent_ids: List[str], timeout_seconds: float = 60.0) -> None:
        """Wait for agents to become active."""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            active_count = 0
            for agent_id in agent_ids:
                if (agent_id in self.orchestrator.agents and 
                    self.orchestrator.agents[agent_id].status == AgentStatus.ACTIVE):
                    active_count += 1
            
            if active_count == len(agent_ids):
                logger.info(f"All {len(agent_ids)} agents are now active")
                return
            
            await asyncio.sleep(1)
        
        logger.warning(f"Timeout waiting for agents to become active: {active_count}/{len(agent_ids)}")
    
    async def _continuous_system_monitoring(self) -> None:
        """Continuously monitor system performance."""
        while self.current_phase == LoadTestPhase.CONCURRENT_OPERATIONS:
            try:
                # Test orchestrator response time
                start_time = time.perf_counter()
                await self.orchestrator.get_system_status()
                response_time = time.perf_counter() - start_time
                
                async with self.metrics_lock:
                    self.current_metrics.orchestrator_response_times.append(response_time)
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _periodic_health_checks(self) -> None:
        """Perform periodic health checks."""
        while self.current_phase == LoadTestPhase.CONCURRENT_OPERATIONS:
            try:
                # Check agent health
                for agent_id in self.test_agents[:10]:  # Check first 10 agents
                    if agent_id in self.orchestrator.agents:
                        agent = self.orchestrator.agents[agent_id]
                        if agent.status == AgentStatus.ACTIVE:
                            self.current_metrics.agents_active += 1
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)
    
    async def _dynamic_task_creation(self) -> None:
        """Create tasks dynamically during concurrent operations."""
        task_counter = 0
        
        while self.current_phase == LoadTestPhase.CONCURRENT_OPERATIONS:
            try:
                task_type = self._select_random_task_type()
                priority = self._select_random_priority()
                
                task_id = await self.orchestrator.delegate_task(
                    f"Dynamic task {task_counter}",
                    task_type,
                    priority
                )
                
                if task_id:
                    self.current_metrics.tasks_submitted += 1
                
                task_counter += 1
                await asyncio.sleep(random.uniform(5, 15))  # Random interval
                
            except Exception as e:
                logger.error(f"Dynamic task creation error: {e}")
                await asyncio.sleep(10)
    
    async def _periodic_load_balancing(self) -> None:
        """Perform periodic load balancing operations."""
        while self.current_phase == LoadTestPhase.CONCURRENT_OPERATIONS:
            try:
                # Test load balancing
                start_time = time.perf_counter()
                result = await self.orchestrator.rebalance_agent_workloads()
                load_balance_time = time.perf_counter() - start_time
                
                if result and not result.get("error"):
                    self.current_metrics.load_balancing_effectiveness += 1
                
                async with self.metrics_lock:
                    self.current_metrics.orchestrator_response_times.append(load_balance_time)
                
                await asyncio.sleep(30)  # Every 30 seconds
                
            except Exception as e:
                logger.error(f"Load balancing error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_final_task_metrics(self) -> None:
        """Collect final task completion metrics."""
        try:
            async with get_session() as db_session:
                # Count completed tasks
                completed_result = await db_session.execute(
                    select(func.count(Task.id))
                    .where(Task.id.in_(self.test_tasks))
                    .where(Task.status == TaskStatus.COMPLETED.value)
                )
                self.current_metrics.tasks_completed = completed_result.scalar() or 0
                
                # Count assigned tasks
                assigned_result = await db_session.execute(
                    select(func.count(Task.id))
                    .where(Task.id.in_(self.test_tasks))
                    .where(Task.assigned_agent_id.isnot(None))
                )
                self.current_metrics.tasks_assigned = assigned_result.scalar() or 0
                
        except Exception as e:
            logger.error(f"Error collecting final task metrics: {e}")
    
    async def _metrics_collection_loop(self) -> None:
        """Collect metrics during test execution."""
        while self.is_running:
            try:
                # Record resource usage
                memory_info = self.process.memory_info()
                self.current_metrics.memory_usage_mb = memory_info.rss / 1024 / 1024
                self.current_metrics.cpu_usage_percent = self.process.cpu_percent()
                
                # Calculate agent uptime
                if self.test_agents:
                    active_agents = sum(
                        1 for agent_id in self.test_agents
                        if (agent_id in self.orchestrator.agents and 
                            self.orchestrator.agents[agent_id].status == AgentStatus.ACTIVE)
                    )
                    self.current_metrics.agent_uptime_percent = (active_agents / len(self.test_agents)) * 100
                
                # Get system health
                health_status = await self.orchestrator._check_system_health()
                self.current_metrics.system_health_score = (
                    sum(1 for v in health_status.values() if v) / len(health_status)
                ) * 100
                
                # Get routing analytics
                if self.orchestrator.intelligent_router:
                    routing_analytics = await self.orchestrator.get_routing_analytics()
                    self.current_metrics.routing_accuracy = routing_analytics.get("routing_accuracy", 0.0) * 100
                
                # Store snapshot
                async with self.metrics_lock:
                    self.metrics_history.append(
                        OrchestratorPerformanceMetrics(**self.current_metrics.__dict__)
                    )
                
                await asyncio.sleep(self.config.resource_monitoring_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)
    
    def _select_random_agent_role(self) -> AgentRole:
        """Select random agent role for testing."""
        roles = [
            AgentRole.BACKEND_DEVELOPER,
            AgentRole.FRONTEND_DEVELOPER,
            AgentRole.DEVOPS_ENGINEER,
            AgentRole.QA_ENGINEER,
            AgentRole.ARCHITECT,
            AgentRole.PRODUCT_MANAGER
        ]
        return random.choice(roles)
    
    def _select_random_task_type(self) -> str:
        """Select random task type based on distribution."""
        rand = random.random()
        cumulative = 0.0
        
        for task_type, probability in self.config.task_types_distribution.items():
            cumulative += probability
            if rand <= cumulative:
                return task_type
        
        return "general"  # Fallback
    
    def _select_random_priority(self) -> TaskPriority:
        """Select random task priority."""
        priorities = [TaskPriority.LOW, TaskPriority.MEDIUM, TaskPriority.HIGH]
        return random.choice(priorities)
    
    async def _clear_test_data(self) -> None:
        """Clear test data from database."""
        try:
            async with get_session() as db_session:
                # Clear test tasks
                if self.test_tasks:
                    await db_session.execute(
                        select(Task).where(Task.id.in_(self.test_tasks))
                    )
                
                # Clear test agents
                if self.test_agents:
                    await db_session.execute(
                        select(Agent).where(Agent.id.in_(self.test_agents))
                    )
                
                await db_session.commit()
                
        except Exception as e:
            logger.error(f"Error clearing test data: {e}")
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if not self.metrics_history:
            return {"error": "No metrics collected"}
        
        # Calculate overall statistics
        final_metrics = self.metrics_history[-1]
        final_metrics.calculate_derived_metrics()
        
        # Performance targets validation
        targets_met = {
            "agent_spawn_time": final_metrics.avg_agent_spawn_time <= self.config.max_agent_spawn_time_seconds,
            "orchestrator_latency": final_metrics.avg_orchestrator_response_time * 1000 <= self.config.max_orchestration_latency_ms,
            "agent_uptime": final_metrics.agent_uptime_percent >= self.config.min_agent_uptime_percent,
            "task_completion_rate": getattr(final_metrics, 'task_completion_rate', 0.0) * 100 >= self.config.min_task_completion_rate,
            "memory_efficiency": self._check_memory_efficiency()
        }
        
        # Phase-specific metrics
        phase_metrics = {}
        for phase in LoadTestPhase:
            phase_data = [m for m in self.metrics_history if m.phase == phase]
            if phase_data:
                phase_metrics[phase.value] = self._aggregate_phase_metrics(phase_data)
        
        test_duration = time.time() - self.test_start_time
        
        return {
            "test_config": {
                "max_concurrent_agents": self.config.max_concurrent_agents,
                "tasks_per_agent": self.config.tasks_per_agent,
                "test_duration_seconds": test_duration
            },
            "performance_summary": final_metrics.to_dict(),
            "targets_validation": targets_met,
            "targets_summary": {
                "all_targets_met": all(targets_met.values()),
                "failed_targets": [k for k, v in targets_met.items() if not v]
            },
            "phase_metrics": phase_metrics,
            "resource_usage": {
                "baseline_memory_mb": self.baseline_memory_mb,
                "peak_memory_mb": max(m.memory_usage_mb for m in self.metrics_history),
                "memory_overhead_mb": final_metrics.memory_usage_mb - self.baseline_memory_mb,
                "peak_cpu_percent": max(m.cpu_usage_percent for m in self.metrics_history)
            },
            "recommendations": self._generate_recommendations(targets_met, final_metrics)
        }
    
    def _check_memory_efficiency(self) -> bool:
        """Check if memory efficiency meets target."""
        if not self.metrics_history or not self.test_agents:
            return False
        
        peak_memory_mb = max(m.memory_usage_mb for m in self.metrics_history)
        memory_overhead_mb = peak_memory_mb - self.baseline_memory_mb
        
        # Calculate memory per 10 agents
        agents_count = len(self.test_agents)
        memory_per_10_agents_gb = (memory_overhead_mb / agents_count) * 10 / 1024
        
        return memory_per_10_agents_gb <= self.config.max_memory_gb_per_10_agents
    
    def _aggregate_phase_metrics(self, phase_data: List[OrchestratorPerformanceMetrics]) -> Dict[str, Any]:
        """Aggregate metrics for a specific phase."""
        if not phase_data:
            return {}
        
        latest_metrics = phase_data[-1]
        latest_metrics.calculate_derived_metrics()
        
        return {
            "duration_seconds": (phase_data[-1].timestamp - phase_data[0].timestamp),
            "agents_spawned": latest_metrics.agents_spawned,
            "agents_active": latest_metrics.agents_active,
            "tasks_submitted": latest_metrics.tasks_submitted,
            "tasks_completed": latest_metrics.tasks_completed,
            "avg_agent_spawn_time": latest_metrics.avg_agent_spawn_time,
            "avg_orchestrator_response_time_ms": latest_metrics.avg_orchestrator_response_time * 1000,
            "p95_orchestrator_response_time_ms": latest_metrics.p95_orchestrator_response_time * 1000,
            "agent_uptime_percent": latest_metrics.agent_uptime_percent,
            "task_completion_rate": getattr(latest_metrics, 'task_completion_rate', 0.0) * 100,
            "peak_memory_mb": max(m.memory_usage_mb for m in phase_data),
            "system_health_score": latest_metrics.system_health_score
        }
    
    def _generate_recommendations(
        self,
        targets_met: Dict[str, bool],
        final_metrics: OrchestratorPerformanceMetrics
    ) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if not targets_met["agent_spawn_time"]:
            recommendations.append(
                f"Agent spawn time ({final_metrics.avg_agent_spawn_time:.2f}s) exceeds target "
                f"({self.config.max_agent_spawn_time_seconds}s). Consider optimizing agent initialization."
            )
        
        if not targets_met["orchestrator_latency"]:
            recommendations.append(
                f"Orchestrator response time ({final_metrics.avg_orchestrator_response_time * 1000:.1f}ms) "
                f"exceeds target ({self.config.max_orchestration_latency_ms}ms). Optimize database queries and caching."
            )
        
        if not targets_met["agent_uptime"]:
            recommendations.append(
                f"Agent uptime ({final_metrics.agent_uptime_percent:.1f}%) below target "
                f"({self.config.min_agent_uptime_percent}%). Improve agent health monitoring and recovery."
            )
        
        if not targets_met["task_completion_rate"]:
            completion_rate = getattr(final_metrics, 'task_completion_rate', 0.0) * 100
            recommendations.append(
                f"Task completion rate ({completion_rate:.1f}%) below target "
                f"({self.config.min_task_completion_rate}%). Enhance task routing and agent matching."
            )
        
        if not targets_met["memory_efficiency"]:
            recommendations.append(
                "Memory usage exceeds efficiency target. Implement memory optimization and agent cleanup."
            )
        
        if all(targets_met.values()):
            recommendations.append(
                "All performance targets met. System is ready for production workloads."
            )
        
        return recommendations


# Factory function for creating load test framework
async def create_orchestrator_load_test(
    config: Optional[OrchestratorLoadTestConfig] = None
) -> OrchestratorLoadTestFramework:
    """
    Create orchestrator load test framework instance.
    
    Args:
        config: Load test configuration
        
    Returns:
        OrchestratorLoadTestFramework instance
    """
    framework = OrchestratorLoadTestFramework(config)
    await framework.setup()
    return framework