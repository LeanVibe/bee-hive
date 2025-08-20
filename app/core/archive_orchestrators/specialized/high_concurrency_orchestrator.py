"""
High Concurrency Orchestrator Extensions for 50+ Agent Management

This module provides specialized enhancements for managing 50+ concurrent agents
with optimized resource allocation, advanced load balancing, and performance monitoring.
"""

import asyncio
import json
import time
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable, Tuple
from enum import Enum
import weakref
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
import resource
import threading

import structlog
import psutil
from prometheus_client import Counter, Histogram, Gauge, Summary

from .production_orchestrator import ProductionOrchestrator, ProductionEventSeverity
from .orchestrator import AgentInstance, AgentRole, AgentStatus
from .config import settings

logger = structlog.get_logger()

# High Concurrency Metrics
AGENT_POOL_SIZE = Gauge('agent_pool_size', 'Current agent pool size')
AGENT_SPAWN_RATE = Gauge('agent_spawn_rate_per_minute', 'Agent spawn rate per minute')
RESOURCE_PRESSURE = Gauge('resource_pressure', 'System resource pressure (0-1)')
LOAD_BALANCING_EFFICIENCY = Gauge('load_balancing_efficiency', 'Load balancing efficiency ratio')
CONTEXT_SWITCHING_OVERHEAD = Histogram('context_switching_time_seconds', 'Context switching overhead')
AGENT_LIFECYCLE_DURATION = Histogram('agent_lifecycle_duration_seconds', 'Agent full lifecycle duration')


@dataclass
class ConcurrencyConfig:
    """Configuration for high concurrency operations."""
    max_concurrent_agents: int = 50
    min_agent_pool: int = 5
    max_agent_pool: int = 75  # Buffer above max concurrent
    spawn_batch_size: int = 5
    shutdown_batch_size: int = 3
    resource_check_interval: int = 15  # seconds
    load_balance_interval: int = 30  # seconds
    agent_timeout_seconds: int = 300
    memory_pressure_threshold: float = 0.85
    cpu_pressure_threshold: float = 0.80
    aggressive_scaling: bool = True


@dataclass
class AgentPoolStats:
    """Statistics for agent pool management."""
    total_agents: int = 0
    active_agents: int = 0
    idle_agents: int = 0
    spawning_agents: int = 0
    terminating_agents: int = 0
    failed_agents: int = 0
    avg_utilization: float = 0.0
    peak_concurrent: int = 0
    spawn_success_rate: float = 1.0
    last_scale_action: Optional[str] = None


@dataclass 
class ResourceSnapshot:
    """System resource snapshot for monitoring."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    disk_percent: float = 0.0
    network_io: Dict[str, int] = field(default_factory=dict)
    open_files: int = 0
    thread_count: int = 0


class AgentLifecyclePhase(Enum):
    """Detailed agent lifecycle phases for tracking."""
    REQUESTED = "requested"
    SPAWNING = "spawning" 
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    IDLE = "idle"
    DRAINING = "draining"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    FAILED = "failed"


class HighConcurrencyOrchestrator:
    """
    Enhanced orchestrator optimized for 50+ concurrent agents.
    
    Features:
    - Agent pool management with pre-warming
    - Advanced resource monitoring and pressure detection
    - Intelligent load balancing and task distribution
    - Circuit breakers for cascading failure prevention
    - Memory leak detection and prevention
    - Performance analytics and optimization
    """
    
    def __init__(self, base_orchestrator: ProductionOrchestrator):
        self.base_orchestrator = base_orchestrator
        self.config = ConcurrencyConfig(
            max_concurrent_agents=getattr(settings, 'MAX_CONCURRENT_AGENTS', 50)
        )
        
        # Agent pool management
        self.agent_pool: Dict[str, AgentInstance] = {}
        self.agent_lifecycle_phases: Dict[str, AgentLifecyclePhase] = {}
        self.agent_performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Resource monitoring
        self.resource_snapshots: deque = deque(maxlen=1000)
        self.resource_pressure_history: deque = deque(maxlen=100)
        
        # Load balancing
        self.task_assignment_history: deque = deque(maxlen=1000)
        self.agent_load_scores: Dict[str, float] = {}
        self.load_balance_lock = asyncio.Lock()
        
        # Performance optimization
        self.spawn_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent spawns
        self.shutdown_semaphore = asyncio.Semaphore(3)  # Max 3 concurrent shutdowns
        self.context_switching_times = deque(maxlen=100)
        
        # Background tasks
        self.monitoring_tasks: List[asyncio.Task] = []
        self.executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="hc_orchestrator")
        
        # Circuit breakers
        self.spawn_circuit_breaker = {'failures': 0, 'last_failure': 0, 'state': 'closed'}
        self.resource_circuit_breaker = {'triggered': False, 'trigger_time': 0}
        
        # Weak references to prevent memory leaks
        self.agent_refs = weakref.WeakValueDictionary()
        
        logger.info(f"🚀 High Concurrency Orchestrator initialized for {self.config.max_concurrent_agents} agents")
    
    async def start_high_concurrency_services(self):
        """Start high concurrency management services."""
        # Start monitoring tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._resource_pressure_monitor()),
            asyncio.create_task(self._agent_pool_manager()),
            asyncio.create_task(self._load_balancer_optimizer()),
            asyncio.create_task(self._performance_analytics_collector()),
            asyncio.create_task(self._memory_leak_detector()),
        ]
        
        # Pre-warm agent pool
        await self._prewarm_agent_pool()
        
        logger.info("📊 High concurrency services started")
    
    async def shutdown_high_concurrency_services(self):
        """Graceful shutdown of high concurrency services."""
        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("🛑 High concurrency services shutdown")
    
    async def spawn_agents_batch(
        self, 
        count: int, 
        roles: List[AgentRole] = None,
        priority: str = "normal"
    ) -> Dict[str, Any]:
        """
        Spawn multiple agents in optimized batches.
        
        Args:
            count: Number of agents to spawn
            roles: List of roles to assign (round-robin if fewer than count)
            priority: Spawn priority level
        """
        if count <= 0:
            return {"spawned": 0, "failed": 0, "agent_ids": []}
        
        # Check resource constraints
        if not await self._can_spawn_agents(count):
            return {"error": "Insufficient resources for batch spawn", "spawned": 0, "failed": count}
        
        # Prepare role assignment
        if not roles:
            roles = [AgentRole.BACKEND_DEVELOPER] * count
        elif len(roles) < count:
            # Repeat roles to match count
            roles = (roles * ((count // len(roles)) + 1))[:count]
        
        # Spawn in batches to prevent resource contention
        batch_size = self.config.spawn_batch_size
        spawned_agents = []
        failed_spawns = 0
        
        start_time = time.time()
        
        for i in range(0, count, batch_size):
            batch_roles = roles[i:i + batch_size]
            batch_results = await self._spawn_agent_batch(batch_roles)
            
            spawned_agents.extend(batch_results['spawned'])
            failed_spawns += batch_results['failed']
            
            # Small delay between batches to prevent resource spike
            if i + batch_size < count:
                await asyncio.sleep(0.1)
        
        spawn_duration = time.time() - start_time
        
        # Update metrics
        AGENT_SPAWN_RATE.set(len(spawned_agents) / max(spawn_duration / 60, 0.1))
        AGENT_POOL_SIZE.set(len(self.agent_pool))
        
        logger.info(
            f"✅ Batch spawn complete: {len(spawned_agents)}/{count} agents in {spawn_duration:.2f}s"
        )
        
        return {
            "spawned": len(spawned_agents),
            "failed": failed_spawns,
            "agent_ids": spawned_agents,
            "duration_seconds": spawn_duration
        }
    
    async def _spawn_agent_batch(self, roles: List[AgentRole]) -> Dict[str, Any]:
        """Spawn a single batch of agents concurrently."""
        spawned = []
        failed = 0
        
        # Use semaphore to limit concurrent spawns
        async with self.spawn_semaphore:
            spawn_tasks = []
            
            for role in roles:
                spawn_tasks.append(self._spawn_single_agent_optimized(role))
            
            # Execute spawns concurrently
            results = await asyncio.gather(*spawn_tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Agent spawn failed: {result}")
                    failed += 1
                elif result:
                    spawned.append(result)
                else:
                    failed += 1
        
        return {"spawned": spawned, "failed": failed}
    
    async def _spawn_single_agent_optimized(self, role: AgentRole) -> Optional[str]:
        """Spawn a single agent with optimized resource management."""
        agent_id = None
        start_time = time.time()
        
        try:
            # Update lifecycle phase
            agent_id = f"agent_{int(time.time() * 1000)}_{role.value}"
            self.agent_lifecycle_phases[agent_id] = AgentLifecyclePhase.SPAWNING
            
            # Use base orchestrator to spawn
            spawned_id = await self.base_orchestrator.agent_orchestrator.spawn_agent(role, agent_id)
            
            if spawned_id:
                # Add to our tracking
                if spawned_id in self.base_orchestrator.agent_orchestrator.agents:
                    agent_instance = self.base_orchestrator.agent_orchestrator.agents[spawned_id]
                    self.agent_pool[spawned_id] = agent_instance
                    self.agent_lifecycle_phases[spawned_id] = AgentLifecyclePhase.READY
                    
                    # Initialize performance tracking
                    self.agent_performance_history[spawned_id] = deque(maxlen=100)
                    self.agent_load_scores[spawned_id] = 0.0
                    
                    spawn_time = time.time() - start_time
                    AGENT_LIFECYCLE_DURATION.observe(spawn_time)
                    
                    # Reset circuit breaker on success
                    self.spawn_circuit_breaker['failures'] = 0
                    
                    return spawned_id
            
            # Spawn failed
            self._update_spawn_circuit_breaker(False)
            return None
            
        except Exception as e:
            logger.error(f"Agent spawn error: {e}")
            self._update_spawn_circuit_breaker(False)
            
            # Cleanup on failure
            if agent_id and agent_id in self.agent_lifecycle_phases:
                self.agent_lifecycle_phases[agent_id] = AgentLifecyclePhase.FAILED
            
            return None
    
    async def shutdown_agents_batch(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Shutdown multiple agents in optimized batches."""
        if not agent_ids:
            return {"shutdown": 0, "failed": 0}
        
        batch_size = self.config.shutdown_batch_size
        shutdown_agents = []
        failed_shutdowns = []
        
        start_time = time.time()
        
        # Shutdown in batches
        for i in range(0, len(agent_ids), batch_size):
            batch_ids = agent_ids[i:i + batch_size]
            batch_results = await self._shutdown_agent_batch(batch_ids)
            
            shutdown_agents.extend(batch_results['shutdown'])
            failed_shutdowns.extend(batch_results['failed'])
            
            # Small delay between batches
            if i + batch_size < len(agent_ids):
                await asyncio.sleep(0.05)
        
        shutdown_duration = time.time() - start_time
        
        logger.info(
            f"🛑 Batch shutdown complete: {len(shutdown_agents)}/{len(agent_ids)} agents in {shutdown_duration:.2f}s"
        )
        
        return {
            "shutdown": len(shutdown_agents),
            "failed": len(failed_shutdowns),
            "duration_seconds": shutdown_duration
        }
    
    async def _shutdown_agent_batch(self, agent_ids: List[str]) -> Dict[str, Any]:
        """Shutdown a single batch of agents concurrently."""
        shutdown = []
        failed = []
        
        async with self.shutdown_semaphore:
            shutdown_tasks = []
            
            for agent_id in agent_ids:
                shutdown_tasks.append(self._shutdown_single_agent_optimized(agent_id))
            
            results = await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                agent_id = agent_ids[i]
                if isinstance(result, Exception):
                    logger.error(f"Agent {agent_id} shutdown failed: {result}")
                    failed.append(agent_id)
                elif result:
                    shutdown.append(agent_id)
                else:
                    failed.append(agent_id)
        
        return {"shutdown": shutdown, "failed": failed}
    
    async def _shutdown_single_agent_optimized(self, agent_id: str) -> bool:
        """Shutdown a single agent with cleanup."""
        start_time = time.time()
        
        try:
            # Update lifecycle phase
            if agent_id in self.agent_lifecycle_phases:
                self.agent_lifecycle_phases[agent_id] = AgentLifecyclePhase.TERMINATING
            
            # Use base orchestrator to shutdown
            success = await self.base_orchestrator.agent_orchestrator.shutdown_agent(agent_id, graceful=True)
            
            if success:
                # Cleanup tracking
                self._cleanup_agent_tracking(agent_id)
                shutdown_time = time.time() - start_time
                AGENT_LIFECYCLE_DURATION.observe(shutdown_time)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Agent {agent_id} shutdown error: {e}")
            return False
    
    def _cleanup_agent_tracking(self, agent_id: str):
        """Clean up all tracking data for an agent."""
        # Remove from tracking dictionaries
        self.agent_pool.pop(agent_id, None)
        self.agent_lifecycle_phases.pop(agent_id, None)
        self.agent_performance_history.pop(agent_id, None)
        self.agent_load_scores.pop(agent_id, None)
        
        # Update pool size metric
        AGENT_POOL_SIZE.set(len(self.agent_pool))
    
    async def optimize_agent_distribution(self) -> Dict[str, Any]:
        """Optimize agent distribution based on workload patterns."""
        async with self.load_balance_lock:
            start_time = time.time()
            
            # Analyze current distribution
            role_distribution = defaultdict(int)
            utilization_by_role = defaultdict(list)
            
            for agent_id, agent in self.agent_pool.items():
                role = agent.role
                role_distribution[role] += 1
                
                # Get utilization score
                utilization = self._calculate_agent_utilization(agent_id)
                utilization_by_role[role].append(utilization)
            
            # Calculate optimal distribution
            total_agents = len(self.agent_pool)
            target_distribution = await self._calculate_optimal_distribution(total_agents)
            
            rebalance_actions = []
            
            # Determine rebalancing actions
            for role, target_count in target_distribution.items():
                current_count = role_distribution[role]
                
                if current_count < target_count:
                    # Need more agents of this role
                    needed = target_count - current_count
                    rebalance_actions.append({
                        'action': 'spawn',
                        'role': role,
                        'count': needed
                    })
                elif current_count > target_count:
                    # Need fewer agents of this role
                    excess = current_count - target_count
                    
                    # Find lowest performing agents of this role
                    role_agents = [
                        (aid, self._calculate_agent_utilization(aid))
                        for aid, agent in self.agent_pool.items()
                        if agent.role == role
                    ]
                    role_agents.sort(key=lambda x: x[1])  # Sort by utilization
                    
                    agents_to_shutdown = [aid for aid, _ in role_agents[:excess]]
                    rebalance_actions.append({
                        'action': 'shutdown',
                        'role': role,
                        'agent_ids': agents_to_shutdown
                    })
            
            # Execute rebalancing actions
            results = await self._execute_rebalance_actions(rebalance_actions)
            
            optimization_time = time.time() - start_time
            
            # Update efficiency metric
            efficiency = self._calculate_load_balancing_efficiency()
            LOAD_BALANCING_EFFICIENCY.set(efficiency)
            
            logger.info(f"🎯 Agent distribution optimized in {optimization_time:.2f}s (efficiency: {efficiency:.2f})")
            
            return {
                "optimization_time": optimization_time,
                "actions_executed": len(rebalance_actions),
                "efficiency": efficiency,
                "results": results
            }
    
    async def _execute_rebalance_actions(self, actions: List[Dict]) -> List[Dict]:
        """Execute rebalancing actions."""
        results = []
        
        for action in actions:
            if action['action'] == 'spawn':
                result = await self.spawn_agents_batch(
                    count=action['count'],
                    roles=[action['role']] * action['count']
                )
                results.append({'action': 'spawn', 'role': action['role'], 'result': result})
                
            elif action['action'] == 'shutdown':
                result = await self.shutdown_agents_batch(action['agent_ids'])
                results.append({'action': 'shutdown', 'role': action['role'], 'result': result})
        
        return results
    
    async def _calculate_optimal_distribution(self, total_agents: int) -> Dict[AgentRole, int]:
        """Calculate optimal agent distribution based on workload patterns."""
        # Base distribution ratios (could be made configurable)
        base_ratios = {
            AgentRole.BACKEND_DEVELOPER: 0.30,
            AgentRole.FRONTEND_DEVELOPER: 0.25,
            AgentRole.QA_ENGINEER: 0.15,
            AgentRole.DEVOPS_ENGINEER: 0.15,
            AgentRole.ARCHITECT: 0.10,
            AgentRole.PRODUCT_MANAGER: 0.03,
            AgentRole.STRATEGIC_PARTNER: 0.02,
        }
        
        # Adjust ratios based on recent task patterns
        task_patterns = await self._analyze_recent_task_patterns()
        adjusted_ratios = self._adjust_ratios_for_workload(base_ratios, task_patterns)
        
        # Calculate target counts
        target_distribution = {}
        allocated = 0
        
        for role, ratio in adjusted_ratios.items():
            target_count = max(1, int(total_agents * ratio))
            target_distribution[role] = target_count
            allocated += target_count
        
        # Distribute remaining agents to most needed roles
        remaining = total_agents - allocated
        if remaining > 0:
            # Give to roles with highest demand
            high_demand_roles = sorted(
                adjusted_ratios.items(),
                key=lambda x: x[1],
                reverse=True
            )[:remaining]
            
            for role, _ in high_demand_roles:
                target_distribution[role] += 1
        
        return target_distribution
    
    async def _analyze_recent_task_patterns(self) -> Dict[str, float]:
        """Analyze recent task patterns to inform distribution."""
        # This would analyze recent task types and requirements
        # For now, return default patterns
        return {
            'backend_heavy': 0.5,
            'frontend_heavy': 0.3,
            'testing_intensive': 0.2,
            'infrastructure_focus': 0.1
        }
    
    def _adjust_ratios_for_workload(
        self, 
        base_ratios: Dict[AgentRole, float],
        task_patterns: Dict[str, float]
    ) -> Dict[AgentRole, float]:
        """Adjust role ratios based on workload patterns."""
        adjusted = base_ratios.copy()
        
        # Adjust based on task patterns
        if task_patterns.get('backend_heavy', 0) > 0.6:
            adjusted[AgentRole.BACKEND_DEVELOPER] *= 1.2
            adjusted[AgentRole.FRONTEND_DEVELOPER] *= 0.9
        
        if task_patterns.get('testing_intensive', 0) > 0.3:
            adjusted[AgentRole.QA_ENGINEER] *= 1.3
        
        if task_patterns.get('infrastructure_focus', 0) > 0.2:
            adjusted[AgentRole.DEVOPS_ENGINEER] *= 1.2
        
        # Normalize to ensure sum is 1.0
        total = sum(adjusted.values())
        for role in adjusted:
            adjusted[role] /= total
        
        return adjusted
    
    def _calculate_agent_utilization(self, agent_id: str) -> float:
        """Calculate current utilization for an agent."""
        if agent_id not in self.agent_pool:
            return 0.0
        
        agent = self.agent_pool[agent_id]
        
        # Calculate based on context window usage and active tasks
        context_utilization = getattr(agent, 'context_window_usage', 0.0)
        task_utilization = 1.0 if hasattr(agent, 'current_task') and agent.current_task else 0.0
        
        return (context_utilization + task_utilization) / 2.0
    
    def _calculate_load_balancing_efficiency(self) -> float:
        """Calculate load balancing efficiency score."""
        if not self.agent_pool:
            return 1.0
        
        utilizations = [self._calculate_agent_utilization(aid) for aid in self.agent_pool.keys()]
        
        if not utilizations:
            return 1.0
        
        # Efficiency is inverse of standard deviation (lower std = higher efficiency)
        mean_util = statistics.mean(utilizations)
        if mean_util == 0:
            return 1.0
        
        std_util = statistics.stdev(utilizations) if len(utilizations) > 1 else 0
        efficiency = max(0, 1 - (std_util / mean_util))
        
        return efficiency
    
    async def _can_spawn_agents(self, count: int) -> bool:
        """Check if system can handle spawning additional agents."""
        current_count = len(self.agent_pool)
        
        # Check hard limits
        if current_count + count > self.config.max_agent_pool:
            return False
        
        # Check resource pressure
        resource_pressure = await self._calculate_resource_pressure()
        if resource_pressure > self.config.memory_pressure_threshold:
            return False
        
        # Check circuit breaker
        if self.spawn_circuit_breaker['state'] == 'open':
            # Check if cooldown period has passed
            if time.time() - self.spawn_circuit_breaker['last_failure'] > 300:  # 5 min cooldown
                self.spawn_circuit_breaker['state'] = 'closed'
                self.spawn_circuit_breaker['failures'] = 0
            else:
                return False
        
        return True
    
    async def _calculate_resource_pressure(self) -> float:
        """Calculate current system resource pressure (0-1)."""
        # Get current system metrics
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        
        # Calculate pressure scores
        memory_pressure = memory.percent / 100.0
        cpu_pressure = cpu / 100.0
        
        # Get process-specific metrics
        try:
            process = psutil.Process()
            process_memory = process.memory_percent()
            open_files = len(process.open_files())
            thread_count = process.num_threads()
            
            # Additional pressure factors
            file_pressure = min(open_files / 1000.0, 1.0)  # Normalize to file limit
            thread_pressure = min(thread_count / 500.0, 1.0)  # Normalize to thread limit
            
        except Exception:
            process_memory = 0
            file_pressure = 0
            thread_pressure = 0
        
        # Weighted pressure calculation
        total_pressure = (
            memory_pressure * 0.4 +
            cpu_pressure * 0.3 +
            (process_memory / 100.0) * 0.2 +
            file_pressure * 0.05 +
            thread_pressure * 0.05
        )
        
        RESOURCE_PRESSURE.set(total_pressure)
        return min(total_pressure, 1.0)
    
    def _update_spawn_circuit_breaker(self, success: bool):
        """Update spawn circuit breaker state."""
        if success:
            self.spawn_circuit_breaker['failures'] = 0
        else:
            self.spawn_circuit_breaker['failures'] += 1
            self.spawn_circuit_breaker['last_failure'] = time.time()
            
            # Trip circuit breaker after 5 failures
            if self.spawn_circuit_breaker['failures'] >= 5:
                self.spawn_circuit_breaker['state'] = 'open'
                logger.warning("🚨 Spawn circuit breaker OPENED due to repeated failures")
    
    async def _prewarm_agent_pool(self):
        """Pre-warm the agent pool with minimum required agents."""
        current_count = len(self.agent_pool)
        needed = max(0, self.config.min_agent_pool - current_count)
        
        if needed > 0:
            logger.info(f"🔥 Pre-warming agent pool with {needed} agents")
            await self.spawn_agents_batch(needed)
    
    async def _resource_pressure_monitor(self):
        """Background monitor for resource pressure."""
        while True:
            try:
                # Take resource snapshot
                snapshot = ResourceSnapshot(
                    memory_percent=psutil.virtual_memory().percent,
                    cpu_percent=psutil.cpu_percent(),
                    disk_percent=psutil.disk_usage('/').percent,
                    network_io=dict(psutil.net_io_counters()._asdict()),
                    open_files=len(psutil.Process().open_files()),
                    thread_count=psutil.Process().num_threads()
                )
                
                self.resource_snapshots.append(snapshot)
                
                # Calculate pressure
                pressure = await self._calculate_resource_pressure()
                self.resource_pressure_history.append(pressure)
                
                # Check for pressure spikes
                if pressure > 0.90:
                    logger.warning(f"🚨 Critical resource pressure: {pressure:.2f}")
                    await self._handle_resource_pressure(pressure)
                
            except Exception as e:
                logger.error(f"❌ Resource monitoring error: {e}")
            
            await asyncio.sleep(self.config.resource_check_interval)
    
    async def _handle_resource_pressure(self, pressure: float):
        """Handle high resource pressure."""
        if pressure > 0.95 and not self.resource_circuit_breaker['triggered']:
            # Emergency scale down
            self.resource_circuit_breaker['triggered'] = True
            self.resource_circuit_breaker['trigger_time'] = time.time()
            
            # Scale down least utilized agents
            agents_to_shutdown = await self._get_emergency_shutdown_candidates()
            if agents_to_shutdown:
                logger.warning(f"🚨 Emergency shutdown of {len(agents_to_shutdown)} agents due to resource pressure")
                await self.shutdown_agents_batch(agents_to_shutdown)
    
    async def _get_emergency_shutdown_candidates(self) -> List[str]:
        """Get agents to shutdown in emergency situations."""
        candidates = []
        
        # Find idle and low-utilization agents
        for agent_id in self.agent_pool.keys():
            phase = self.agent_lifecycle_phases.get(agent_id)
            utilization = self._calculate_agent_utilization(agent_id)
            
            if phase == AgentLifecyclePhase.IDLE or utilization < 0.1:
                candidates.append((agent_id, utilization))
        
        # Sort by utilization (lowest first)
        candidates.sort(key=lambda x: x[1])
        
        # Return up to 25% of agent pool
        max_shutdown = max(1, len(self.agent_pool) // 4)
        return [agent_id for agent_id, _ in candidates[:max_shutdown]]
    
    async def _agent_pool_manager(self):
        """Background agent pool management."""
        while True:
            try:
                stats = await self._calculate_pool_stats()
                
                # Check if we need to adjust pool size
                if stats.active_agents < self.config.min_agent_pool:
                    needed = self.config.min_agent_pool - stats.total_agents
                    if needed > 0:
                        await self.spawn_agents_batch(needed)
                
                # Remove failed agents
                failed_agents = [
                    aid for aid, phase in self.agent_lifecycle_phases.items()
                    if phase == AgentLifecyclePhase.FAILED
                ]
                
                if failed_agents:
                    for agent_id in failed_agents:
                        self._cleanup_agent_tracking(agent_id)
                    logger.info(f"🧹 Cleaned up {len(failed_agents)} failed agents")
                
            except Exception as e:
                logger.error(f"❌ Agent pool management error: {e}")
            
            await asyncio.sleep(60)  # Check every minute
    
    async def _load_balancer_optimizer(self):
        """Background load balancing optimization."""
        while True:
            try:
                await asyncio.sleep(self.config.load_balance_interval)
                
                # Only optimize if we have enough agents
                if len(self.agent_pool) >= 5:
                    await self.optimize_agent_distribution()
                
            except Exception as e:
                logger.error(f"❌ Load balancer optimization error: {e}")
    
    async def _performance_analytics_collector(self):
        """Background performance analytics collection."""
        while True:
            try:
                # Collect performance metrics for all agents
                for agent_id in self.agent_pool.keys():
                    metrics = {
                        'timestamp': time.time(),
                        'utilization': self._calculate_agent_utilization(agent_id),
                        'phase': self.agent_lifecycle_phases.get(agent_id, AgentLifecyclePhase.ACTIVE).value
                    }
                    
                    self.agent_performance_history[agent_id].append(metrics)
                
                # Update global metrics
                stats = await self._calculate_pool_stats()
                AGENT_POOL_SIZE.set(stats.total_agents)
                
            except Exception as e:
                logger.error(f"❌ Performance analytics error: {e}")
            
            await asyncio.sleep(30)  # Collect every 30 seconds
    
    async def _memory_leak_detector(self):
        """Background memory leak detection."""
        baseline_memory = None
        
        while True:
            try:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                if baseline_memory is None:
                    baseline_memory = current_memory
                
                # Check for significant memory growth
                memory_growth = current_memory - baseline_memory
                growth_ratio = memory_growth / baseline_memory if baseline_memory > 0 else 0
                
                if growth_ratio > 0.5:  # 50% growth
                    logger.warning(f"🧠 Potential memory leak detected: {memory_growth:.1f}MB growth ({growth_ratio:.1%})")
                    
                    # Force garbage collection
                    import gc
                    collected = gc.collect()
                    logger.info(f"🗑️ Garbage collection freed {collected} objects")
                    
                    # Reset baseline after cleanup
                    baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
            except Exception as e:
                logger.error(f"❌ Memory leak detection error: {e}")
            
            await asyncio.sleep(300)  # Check every 5 minutes
    
    async def _calculate_pool_stats(self) -> AgentPoolStats:
        """Calculate comprehensive agent pool statistics."""
        stats = AgentPoolStats()
        
        stats.total_agents = len(self.agent_pool)
        
        # Count by phase
        phase_counts = defaultdict(int)
        for phase in self.agent_lifecycle_phases.values():
            phase_counts[phase] += 1
        
        stats.active_agents = phase_counts[AgentLifecyclePhase.ACTIVE]
        stats.idle_agents = phase_counts[AgentLifecyclePhase.IDLE]
        stats.spawning_agents = phase_counts[AgentLifecyclePhase.SPAWNING]
        stats.terminating_agents = phase_counts[AgentLifecyclePhase.TERMINATING]
        stats.failed_agents = phase_counts[AgentLifecyclePhase.FAILED]
        
        # Calculate utilization
        if self.agent_pool:
            utilizations = [self._calculate_agent_utilization(aid) for aid in self.agent_pool.keys()]
            stats.avg_utilization = statistics.mean(utilizations)
        
        return stats
    
    async def get_high_concurrency_metrics(self) -> Dict[str, Any]:
        """Get comprehensive high concurrency metrics."""
        stats = await self._calculate_pool_stats()
        pressure = await self._calculate_resource_pressure()
        efficiency = self._calculate_load_balancing_efficiency()
        
        # Recent performance metrics
        recent_performance = {}
        for agent_id, history in self.agent_performance_history.items():
            if history:
                recent_metrics = list(history)[-10:]  # Last 10 metrics
                recent_performance[agent_id] = {
                    'avg_utilization': statistics.mean([m['utilization'] for m in recent_metrics]),
                    'current_phase': self.agent_lifecycle_phases.get(agent_id, AgentLifecyclePhase.ACTIVE).value
                }
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'pool_stats': {
                'total_agents': stats.total_agents,
                'active_agents': stats.active_agents,
                'idle_agents': stats.idle_agents,
                'spawning_agents': stats.spawning_agents,
                'failed_agents': stats.failed_agents,
                'avg_utilization': stats.avg_utilization
            },
            'resource_metrics': {
                'pressure': pressure,
                'memory_percent': psutil.virtual_memory().percent,
                'cpu_percent': psutil.cpu_percent(),
                'open_files': len(psutil.Process().open_files()),
                'thread_count': psutil.Process().num_threads()
            },
            'performance_metrics': {
                'load_balancing_efficiency': efficiency,
                'spawn_circuit_breaker_state': self.spawn_circuit_breaker['state'],
                'resource_circuit_breaker_triggered': self.resource_circuit_breaker['triggered']
            },
            'agent_performance': recent_performance,
            'configuration': {
                'max_concurrent_agents': self.config.max_concurrent_agents,
                'min_agent_pool': self.config.min_agent_pool,
                'max_agent_pool': self.config.max_agent_pool,
                'spawn_batch_size': self.config.spawn_batch_size
            }
        }


# Global high concurrency orchestrator instance
_high_concurrency_orchestrator: Optional[HighConcurrencyOrchestrator] = None


async def get_high_concurrency_orchestrator() -> HighConcurrencyOrchestrator:
    """Get or create the global high concurrency orchestrator instance."""
    global _high_concurrency_orchestrator
    
    if _high_concurrency_orchestrator is None:
        from .production_orchestrator import ProductionOrchestrator
        
        # Get base production orchestrator
        base_orchestrator = ProductionOrchestrator()
        await base_orchestrator.start()
        
        # Create high concurrency orchestrator
        _high_concurrency_orchestrator = HighConcurrencyOrchestrator(base_orchestrator)
        await _high_concurrency_orchestrator.start_high_concurrency_services()
    
    return _high_concurrency_orchestrator


async def shutdown_high_concurrency_orchestrator():
    """Shutdown the global high concurrency orchestrator."""
    global _high_concurrency_orchestrator
    
    if _high_concurrency_orchestrator:
        await _high_concurrency_orchestrator.shutdown_high_concurrency_services()
        _high_concurrency_orchestrator = None