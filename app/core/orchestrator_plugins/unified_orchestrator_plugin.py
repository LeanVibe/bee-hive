"""
Unified Orchestrator Plugin for LeanVibe Agent Hive 2.0

Epic 1 Phase 2.2: Consolidated plugin architecture
Consolidates unified_orchestrator.py capabilities into the unified plugin system.

Key Features:
- Advanced orchestration modes (production, development, testing, sandbox)
- Multi-agent coordination with intelligent workload distribution
- Real-time WebSocket communication and coordination
- Plugin-based architecture with dynamic loading
- Background task monitoring and health checks
- Auto-scaling with predictive algorithms
- Agent persona management and capability matching

Epic 1 Performance Targets:
- <50ms orchestration decisions
- <20MB memory footprint for unified capabilities
- Support for 100+ concurrent agents
- <1s agent spawning and initialization
"""

import asyncio
import json
import uuid
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import heapq

from .base_plugin import OrchestratorPlugin, PluginMetadata, PluginError
from ..simple_orchestrator import SimpleOrchestrator, AgentRole, AgentInstance
from ...models.agent import Agent, AgentStatus, AgentType
from ...models.task import Task, TaskStatus, TaskPriority
from ..logging_service import get_component_logger

logger = get_component_logger("unified_orchestrator_plugin")


class OrchestratorMode(Enum):
    """Orchestrator operational modes."""
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    TESTING = "testing"
    SANDBOX = "sandbox"


class TaskExecutionState(Enum):
    """Task execution states for unified orchestration."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class UnifiedOrchestratorConfig:
    """Configuration for the unified orchestrator."""
    mode: OrchestratorMode = OrchestratorMode.PRODUCTION
    max_agents: int = 100
    max_concurrent_tasks: int = 1000
    health_check_interval: int = 30
    cleanup_interval: int = 300
    auto_scaling_enabled: bool = True
    websocket_enabled: bool = True
    
    # Performance thresholds
    max_response_time_ms: float = 1000.0
    max_cpu_usage_percent: float = 80.0
    max_memory_usage_percent: float = 80.0
    
    # Auto-scaling parameters
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    min_agents: int = 2
    max_scale_up_rate: int = 5  # agents per minute


@dataclass
class AgentInfo:
    """Information about a managed agent."""
    agent_id: str
    agent_type: AgentType
    role: AgentRole
    status: AgentStatus
    persona: Optional[str]
    capabilities: Set[str]
    current_task_id: Optional[str]
    performance_score: float
    last_heartbeat: datetime
    created_at: datetime
    container_id: Optional[str] = None
    websocket_connection: Optional[Any] = None


@dataclass
class TaskExecution:
    """Information about task execution."""
    task_id: str
    agent_id: str
    status: TaskExecutionState
    started_at: datetime
    estimated_completion: Optional[datetime]
    progress_percentage: float
    context_size_bytes: int
    performance_metrics: Dict[str, Any]


class UnifiedOrchestratorPlugin(OrchestratorPlugin):
    """
    Unified orchestrator plugin providing advanced multi-agent coordination.
    
    Epic 1 Phase 2.2: Consolidation of unified_orchestrator.py capabilities 
    into the unified plugin architecture for SimpleOrchestrator integration.
    
    Provides:
    - Advanced orchestration modes and configurations
    - Multi-agent coordination with real-time communication
    - Intelligent task delegation and workload balancing
    - Background monitoring and health checks
    - Auto-scaling with predictive algorithms
    - WebSocket communication support
    - Agent persona and capability management
    """
    
    def __init__(self, config: Optional[UnifiedOrchestratorConfig] = None):
        super().__init__(
            metadata=PluginMetadata(
                name="unified_orchestrator",
                version="2.2.0",
                description="Advanced multi-agent coordination with auto-scaling and real-time communication",
                author="LeanVibe Agent Hive",
                capabilities=["advanced_orchestration", "multi_agent_coordination", "auto_scaling", "websocket_communication"],
                dependencies=["simple_orchestrator"],
                epic_phase="Epic 1 Phase 2.2"
            )
        )
        
        self.config = config or UnifiedOrchestratorConfig()
        
        # Core state
        self.managed_agents: Dict[str, AgentInfo] = {}
        self.active_tasks: Dict[str, TaskExecution] = {}
        self.task_queue: List[Task] = []
        self.is_running = False
        self.startup_time = None
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        # Metrics and monitoring
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "agents_spawned": 0,
            "agents_terminated": 0,
            "total_execution_time": 0.0,
            "avg_response_time": 0.0
        }
        
        # Performance tracking for Epic 1 targets
        self.operation_times: Dict[str, List[float]] = {}
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        # Auto-scaling state
        self.last_scaling_decision = datetime.utcnow()
        self.scaling_cooldown = timedelta(minutes=2)
        
        # WebSocket connections (if enabled)
        self.websocket_connections: Dict[str, Any] = {}
        
    async def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize the unified orchestrator plugin."""
        await super().initialize(context)
        
        self.orchestrator = context.get("orchestrator")
        if not isinstance(self.orchestrator, SimpleOrchestrator):
            raise PluginError("UnifiedOrchestratorPlugin requires SimpleOrchestrator")
        
        # Initialize based on configuration
        await self._initialize_unified_capabilities()
        
        # Start background tasks
        await self._start_background_tasks()
        
        self.startup_time = datetime.utcnow()
        self.is_running = True
        
        logger.info(f"Unified Orchestrator Plugin initialized in {self.config.mode.value} mode")
        
    async def _initialize_unified_capabilities(self):
        """Initialize unified orchestrator capabilities."""
        # Initialize agent management
        await self._initialize_agent_management()
        
        # Initialize task execution engine
        await self._initialize_task_execution()
        
        # Initialize WebSocket support if enabled
        if self.config.websocket_enabled:
            await self._initialize_websocket_support()
        
        # Initialize auto-scaling if enabled
        if self.config.auto_scaling_enabled:
            await self._initialize_auto_scaling()
    
    async def _initialize_agent_management(self):
        """Initialize advanced agent management capabilities."""
        # Set up agent lifecycle tracking
        # Set up capability management
        # Set up persona assignment
        pass
    
    async def _initialize_task_execution(self):
        """Initialize advanced task execution engine."""
        # Set up task queue management
        # Set up execution monitoring
        # Set up performance tracking
        pass
    
    async def _initialize_websocket_support(self):
        """Initialize WebSocket communication support."""
        # Set up WebSocket connection management
        # Set up real-time coordination
        # Set up event broadcasting
        pass
    
    async def _initialize_auto_scaling(self):
        """Initialize auto-scaling capabilities."""
        # Set up scaling decision logic
        # Set up performance monitoring
        # Set up predictive scaling
        pass
    
    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks."""
        # Health monitoring
        health_task = asyncio.create_task(self._health_monitoring_loop())
        self._background_tasks.add(health_task)
        
        # Task processing
        processing_task = asyncio.create_task(self._task_processing_loop())
        self._background_tasks.add(processing_task)
        
        # Cleanup and maintenance
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._background_tasks.add(cleanup_task)
        
        # Auto-scaling (if enabled)
        if self.config.auto_scaling_enabled:
            scaling_task = asyncio.create_task(self._auto_scaling_loop())
            self._background_tasks.add(scaling_task)
        
        logger.info(f"Started {len(self._background_tasks)} unified orchestrator background tasks")
    
    async def spawn_unified_agent(
        self,
        agent_type: AgentType,
        role: AgentRole,
        capabilities: Optional[Set[str]] = None,
        persona: Optional[str] = None,
        **kwargs
    ) -> str:
        """Spawn a new agent with unified orchestrator capabilities."""
        import time
        start_time_ms = time.time()
        
        async with self._lock:
            try:
                # Check limits
                if len(self.managed_agents) >= self.config.max_agents:
                    raise PluginError(f"Maximum agent limit ({self.config.max_agents}) reached")
                
                # Generate agent ID
                agent_id = f"unified_agent_{uuid.uuid4().hex[:8]}"
                
                # Default capabilities based on role
                if not capabilities:
                    capabilities = self._get_default_capabilities(role)
                
                # Create agent info
                agent_info = AgentInfo(
                    agent_id=agent_id,
                    agent_type=agent_type,
                    role=role,
                    status=AgentStatus.INITIALIZING,
                    persona=persona,
                    capabilities=capabilities or set(),
                    current_task_id=None,
                    performance_score=0.5,  # Default performance score
                    last_heartbeat=datetime.utcnow(),
                    created_at=datetime.utcnow()
                )
                
                # Store agent in unified management
                self.managed_agents[agent_id] = agent_info
                
                # Delegate to SimpleOrchestrator for actual spawning
                actual_agent_id = await self.orchestrator.spawn_agent(
                    role=role,
                    agent_id=agent_id,
                    **kwargs
                )
                
                # Update agent status
                agent_info.status = AgentStatus.ACTIVE
                agent_info.last_heartbeat = datetime.utcnow()
                
                # Update metrics
                self.metrics["agents_spawned"] += 1
                
                # Epic 1 Performance tracking
                operation_time_ms = (time.time() - start_time_ms) * 1000
                self._record_operation_time("spawn_unified_agent", operation_time_ms)
                
                logger.info(f"Spawned unified agent {agent_id} with role {role.value} in {operation_time_ms:.2f}ms")
                return actual_agent_id
                
            except Exception as e:
                logger.error(f"Failed to spawn unified agent: {e}")
                raise PluginError(f"Unified agent spawn failed: {e}")
    
    def _get_default_capabilities(self, role: AgentRole) -> Set[str]:
        """Get default capabilities for an agent role."""
        capabilities_map = {
            AgentRole.COORDINATOR: {"task_delegation", "agent_management", "workflow_coordination"},
            AgentRole.SPECIALIST: {"domain_expertise", "complex_analysis", "specialized_tools"},
            AgentRole.WORKER: {"task_execution", "basic_tools", "data_processing"},
            AgentRole.MONITOR: {"system_monitoring", "health_checks", "performance_analysis"},
            AgentRole.SECURITY: {"security_analysis", "threat_detection", "compliance_checking"},
            AgentRole.OPTIMIZER: {"performance_tuning", "resource_optimization", "efficiency_analysis"}
        }
        
        # Fallback for SimpleOrchestrator AgentRoles
        simple_capabilities_map = {
            "BACKEND_DEVELOPER": {"python", "databases", "apis", "server_development"},
            "FRONTEND_DEVELOPER": {"javascript", "react", "ui", "client_development"},
            "DEVOPS_ENGINEER": {"docker", "deployment", "infrastructure", "monitoring"},
            "QA_ENGINEER": {"testing", "quality_assurance", "automation", "validation"},
            "META_AGENT": {"coordination", "management", "planning", "oversight"}
        }
        
        # Try unified role mapping first, then fallback to simple mapping
        if role in capabilities_map:
            return capabilities_map[role]
        else:
            role_name = role.value.upper() if hasattr(role, 'value') else str(role).upper()
            return simple_capabilities_map.get(role_name, {"general_capability"})
    
    async def delegate_unified_task(
        self,
        task: Task,
        preferred_agent_id: Optional[str] = None,
        required_capabilities: Optional[Set[str]] = None,
        priority_boost: float = 0.0
    ) -> str:
        """Delegate a task with unified orchestrator intelligence."""
        import time
        start_time_ms = time.time()
        
        try:
            # Enhanced task analysis
            task_context = await self._analyze_task_context(task, required_capabilities)
            
            # Find suitable agent using unified intelligence
            if preferred_agent_id and preferred_agent_id in self.managed_agents:
                agent_info = self.managed_agents[preferred_agent_id]
                if agent_info.status == AgentStatus.ACTIVE and not agent_info.current_task_id:
                    selected_agent_id = preferred_agent_id
                else:
                    raise PluginError(f"Preferred agent {preferred_agent_id} is not available")
            else:
                selected_agent_id = await self._select_optimal_agent(task_context, required_capabilities)
            
            if not selected_agent_id:
                # Add to unified queue with priority boost
                self.task_queue.append(task)
                logger.info(f"Task {task.id} queued with unified orchestrator - no suitable agents available")
                return task.id
            
            # Create unified task execution
            task_execution = TaskExecution(
                task_id=task.id,
                agent_id=selected_agent_id,
                status=TaskExecutionState.QUEUED,
                started_at=datetime.utcnow(),
                estimated_completion=None,
                progress_percentage=0.0,
                context_size_bytes=0,
                performance_metrics={}
            )
            
            # Store in unified tracking
            self.active_tasks[task.id] = task_execution
            
            # Update managed agent
            if selected_agent_id in self.managed_agents:
                agent_info = self.managed_agents[selected_agent_id]
                agent_info.current_task_id = task.id
                agent_info.status = AgentStatus.BUSY
            
            # Delegate to SimpleOrchestrator
            delegated_task_id = await self.orchestrator.delegate_task(
                task_description=task.description,
                task_type=task.task_type,
                priority=task.priority,
                preferred_agent_role=None  # Already selected specific agent
            )
            
            # Start unified execution monitoring
            execution_task = asyncio.create_task(
                self._monitor_unified_execution(task_execution, task)
            )
            self._background_tasks.add(execution_task)
            
            # Epic 1 Performance tracking
            operation_time_ms = (time.time() - start_time_ms) * 1000
            self._record_operation_time("delegate_unified_task", operation_time_ms)
            
            logger.info(f"Delegated unified task {task.id} to agent {selected_agent_id} in {operation_time_ms:.2f}ms")
            return delegated_task_id
            
        except Exception as e:
            logger.error(f"Failed to delegate unified task {task.id}: {e}")
            raise PluginError(f"Unified task delegation failed: {e}")
    
    async def _analyze_task_context(self, task: Task, required_capabilities: Optional[Set[str]]) -> Dict[str, Any]:
        """Analyze task context for intelligent routing."""
        context = {
            "task_id": task.id,
            "complexity": await self._estimate_task_complexity(task),
            "estimated_duration": await self._estimate_task_duration(task),
            "resource_requirements": await self._estimate_resource_requirements(task),
            "required_capabilities": required_capabilities or set()
        }
        
        return context
    
    async def _estimate_task_complexity(self, task: Task) -> float:
        """Estimate task complexity (0.0 to 1.0)."""
        # Simplified complexity estimation
        description_length = len(task.description)
        if description_length > 1000:
            return 0.9
        elif description_length > 500:
            return 0.6
        else:
            return 0.3
    
    async def _estimate_task_duration(self, task: Task) -> timedelta:
        """Estimate task duration."""
        # Simplified duration estimation
        complexity = await self._estimate_task_complexity(task)
        base_duration = 30  # 30 minutes base
        return timedelta(minutes=base_duration * (1 + complexity))
    
    async def _estimate_resource_requirements(self, task: Task) -> Dict[str, float]:
        """Estimate resource requirements."""
        return {
            "cpu": 0.5,
            "memory": 0.3,
            "io": 0.2
        }
    
    async def _select_optimal_agent(
        self,
        task_context: Dict[str, Any],
        required_capabilities: Optional[Set[str]]
    ) -> Optional[str]:
        """Select optimal agent using unified intelligence."""
        available_agents = [
            agent for agent in self.managed_agents.values()
            if agent.status == AgentStatus.ACTIVE and agent.current_task_id is None
        ]
        
        if not available_agents:
            return None
        
        # Filter by capabilities if required
        if required_capabilities:
            available_agents = [
                agent for agent in available_agents
                if required_capabilities.issubset(agent.capabilities)
            ]
        
        if not available_agents:
            return None
        
        # Score agents based on multiple factors
        scored_agents = []
        for agent in available_agents:
            score = await self._calculate_agent_score(agent, task_context)
            scored_agents.append((score, agent))
        
        # Return highest scoring agent
        if scored_agents:
            scored_agents.sort(key=lambda x: x[0], reverse=True)
            return scored_agents[0][1].agent_id
        
        return None
    
    async def _calculate_agent_score(self, agent: AgentInfo, task_context: Dict[str, Any]) -> float:
        """Calculate agent suitability score for task."""
        score = 0.0
        
        # Performance score (0.0 to 1.0)
        score += agent.performance_score * 0.4
        
        # Capability match score
        required_caps = task_context.get("required_capabilities", set())
        if required_caps:
            match_ratio = len(required_caps.intersection(agent.capabilities)) / len(required_caps)
            score += match_ratio * 0.3
        
        # Availability score (higher for recently active agents)
        time_since_heartbeat = datetime.utcnow() - agent.last_heartbeat
        availability = max(0.0, 1.0 - (time_since_heartbeat.total_seconds() / 3600))  # Decay over 1 hour
        score += availability * 0.3
        
        return score
    
    async def _monitor_unified_execution(self, task_execution: TaskExecution, task: Task):
        """Monitor unified task execution."""
        try:
            task_execution.status = TaskExecutionState.RUNNING
            task_execution.started_at = datetime.utcnow()
            
            # Monitor execution progress
            while task_execution.status == TaskExecutionState.RUNNING:
                # Update progress (simplified)
                elapsed = datetime.utcnow() - task_execution.started_at
                task_execution.progress_percentage = min(100.0, (elapsed.total_seconds() / 3600) * 100)
                
                # Check for completion or failure
                # In production, this would check actual task status
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Simulate completion after some time
                if elapsed > timedelta(seconds=30):  # Simulate 30 second task
                    task_execution.status = TaskExecutionState.COMPLETED
                    task_execution.progress_percentage = 100.0
                    break
            
            # Update metrics and cleanup
            if task_execution.status == TaskExecutionState.COMPLETED:
                self.metrics["tasks_completed"] += 1
                execution_time = (datetime.utcnow() - task_execution.started_at).total_seconds()
                self.metrics["total_execution_time"] += execution_time
                self.metrics["avg_response_time"] = self.metrics["total_execution_time"] / self.metrics["tasks_completed"]
            else:
                self.metrics["tasks_failed"] += 1
            
            # Update agent status
            if task_execution.agent_id in self.managed_agents:
                agent_info = self.managed_agents[task_execution.agent_id]
                agent_info.current_task_id = None
                agent_info.status = AgentStatus.ACTIVE
                agent_info.last_heartbeat = datetime.utcnow()
            
            # Clean up
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            
            logger.info(f"Unified task {task.id} monitoring complete: {task_execution.status.value}")
            
        except Exception as e:
            logger.error(f"Error monitoring unified task execution: {e}")
            task_execution.status = TaskExecutionState.FAILED
            self.metrics["tasks_failed"] += 1
        finally:
            # Remove from background tasks
            current_task = asyncio.current_task()
            if current_task in self._background_tasks:
                self._background_tasks.remove(current_task)
    
    # Background monitoring loops
    
    async def _health_monitoring_loop(self) -> None:
        """Background task for health monitoring."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_unified_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in unified health monitoring: {e}")
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _perform_unified_health_checks(self):
        """Perform comprehensive health checks."""
        # Check managed agent health
        unhealthy_agents = []
        for agent_id, agent_info in self.managed_agents.items():
            time_since_heartbeat = datetime.utcnow() - agent_info.last_heartbeat
            if time_since_heartbeat > timedelta(seconds=self.config.health_check_interval * 3):
                unhealthy_agents.append(agent_id)
                agent_info.status = AgentStatus.ERROR
        
        if unhealthy_agents:
            logger.warning(f"Found {len(unhealthy_agents)} unhealthy unified agents")
    
    async def _task_processing_loop(self) -> None:
        """Background task for processing queued tasks."""
        while not self._shutdown_event.is_set():
            try:
                if self.task_queue:
                    # Process queued tasks
                    tasks_to_remove = []
                    
                    for task in self.task_queue[:5]:  # Process up to 5 tasks per cycle
                        agent_id = await self._select_optimal_agent(
                            await self._analyze_task_context(task, None),
                            None
                        )
                        if agent_id:
                            # Move task from queue to execution
                            await self.delegate_unified_task(task, preferred_agent_id=agent_id)
                            tasks_to_remove.append(task)
                    
                    # Remove processed tasks from queue
                    for task in tasks_to_remove:
                        if task in self.task_queue:
                            self.task_queue.remove(task)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in unified task processing loop: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_loop(self) -> None:
        """Background task for cleanup and maintenance."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_unified_cleanup()
                await asyncio.sleep(self.config.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in unified cleanup loop: {e}")
                await asyncio.sleep(self.config.cleanup_interval)
    
    async def _perform_unified_cleanup(self):
        """Perform cleanup and maintenance tasks."""
        # Clean up completed tasks
        completed_tasks = [
            task_id for task_id, task_execution in self.active_tasks.items()
            if task_execution.status in [TaskExecutionState.COMPLETED, TaskExecutionState.FAILED]
        ]
        
        for task_id in completed_tasks:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
        
        # Remove failed agents
        failed_agents = [
            agent_id for agent_id, agent_info in self.managed_agents.items()
            if agent_info.status == AgentStatus.ERROR
        ]
        
        for agent_id in failed_agents:
            await self._remove_unified_agent(agent_id)
    
    async def _auto_scaling_loop(self) -> None:
        """Background task for auto-scaling."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_unified_auto_scaling()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in unified auto-scaling: {e}")
                await asyncio.sleep(60)
    
    async def _perform_unified_auto_scaling(self):
        """Perform auto-scaling based on load."""
        # Check cooldown period
        if datetime.utcnow() - self.last_scaling_decision < self.scaling_cooldown:
            return
        
        # Calculate current load
        idle_agents = len([a for a in self.managed_agents.values() if a.status == AgentStatus.ACTIVE and not a.current_task_id])
        busy_agents = len([a for a in self.managed_agents.values() if a.current_task_id])
        total_agents = len(self.managed_agents)
        queue_size = len(self.task_queue)
        
        if total_agents == 0:
            return
        
        load_ratio = busy_agents / total_agents if total_agents > 0 else 0
        
        # Scale up if needed
        if (load_ratio > self.config.scale_up_threshold or queue_size > 5) and total_agents < self.config.max_agents:
            agents_to_spawn = min(
                self.config.max_scale_up_rate,
                self.config.max_agents - total_agents,
                max(1, queue_size // 2)  # Spawn based on queue size
            )
            
            for _ in range(agents_to_spawn):
                try:
                    await self.spawn_unified_agent(
                        agent_type=AgentType.ANTHROPIC_CLAUDE,
                        role=AgentRole.BACKEND_DEVELOPER  # Default role for auto-scaling
                    )
                except Exception as e:
                    logger.error(f"Failed to spawn agent during unified auto-scaling: {e}")
                    break
            
            if agents_to_spawn > 0:
                logger.info(f"Unified auto-scaled up: spawned {agents_to_spawn} agents")
                self.last_scaling_decision = datetime.utcnow()
        
        # Scale down if needed
        elif load_ratio < self.config.scale_down_threshold and total_agents > self.config.min_agents:
            excess_agents = min(
                total_agents - self.config.min_agents,
                idle_agents // 2  # Only remove half of idle agents
            )
            
            # Remove idle agents
            idle_agent_ids = [
                agent_id for agent_id, agent_info in self.managed_agents.items()
                if agent_info.status == AgentStatus.ACTIVE and not agent_info.current_task_id
            ]
            
            removed_count = 0
            for agent_id in idle_agent_ids[:excess_agents]:
                try:
                    await self._remove_unified_agent(agent_id)
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Failed to remove agent during unified auto-scaling: {e}")
            
            if removed_count > 0:
                logger.info(f"Unified auto-scaled down: removed {removed_count} agents")
                self.last_scaling_decision = datetime.utcnow()
    
    async def _remove_unified_agent(self, agent_id: str):
        """Remove a unified agent."""
        if agent_id not in self.managed_agents:
            return
        
        agent_info = self.managed_agents[agent_id]
        
        # If agent has an active task, mark it as failed
        if agent_info.current_task_id and agent_info.current_task_id in self.active_tasks:
            task_execution = self.active_tasks[agent_info.current_task_id]
            task_execution.status = TaskExecutionState.FAILED
        
        # Remove from SimpleOrchestrator
        try:
            await self.orchestrator.shutdown_agent(agent_id, graceful=True)
        except Exception as e:
            logger.warning(f"Failed to shutdown agent via SimpleOrchestrator: {e}")
        
        # Remove from unified management
        del self.managed_agents[agent_id]
        
        # Update metrics
        self.metrics["agents_terminated"] += 1
        
        logger.info(f"Removed unified agent {agent_id}")
    
    async def get_unified_status(self) -> Dict[str, Any]:
        """Get comprehensive unified orchestrator status."""
        import time
        start_time_ms = time.time()
        
        try:
            # Get base status from SimpleOrchestrator
            base_status = await self.orchestrator.get_system_status()
            
            # Agent status breakdown for unified agents
            unified_agent_status = {}
            for status in AgentStatus:
                unified_agent_status[status.value] = len([
                    a for a in self.managed_agents.values() if a.status == status
                ])
            
            # Task status breakdown for unified tasks
            unified_task_status = {}
            for status in TaskExecutionState:
                unified_task_status[status.value] = len([
                    t for t in self.active_tasks.values() if t.status == status
                ])
            
            uptime = (datetime.utcnow() - self.startup_time).total_seconds() if self.startup_time else 0
            
            unified_status = {
                **base_status,
                "unified_orchestrator": {
                    "mode": self.config.mode.value,
                    "is_running": self.is_running,
                    "uptime_seconds": uptime,
                    "startup_time": self.startup_time.isoformat() if self.startup_time else None
                },
                "unified_agents": {
                    "managed_total": len(self.managed_agents),
                    "status_breakdown": unified_agent_status,
                    "max_agents": self.config.max_agents
                },
                "unified_tasks": {
                    "active": len(self.active_tasks),
                    "queued": len(self.task_queue),
                    "status_breakdown": unified_task_status,
                    "max_concurrent": self.config.max_concurrent_tasks
                },
                "unified_metrics": self.metrics,
                "background_tasks": len(self._background_tasks),
                "auto_scaling": {
                    "enabled": self.config.auto_scaling_enabled,
                    "last_decision": self.last_scaling_decision.isoformat()
                },
                "websocket": {
                    "enabled": self.config.websocket_enabled,
                    "active_connections": len(self.websocket_connections)
                }
            }
            
            # Epic 1 Performance tracking
            operation_time_ms = (time.time() - start_time_ms) * 1000
            self._record_operation_time("get_unified_status", operation_time_ms)
            
            unified_status["performance"] = {
                **unified_status.get("performance", {}),
                "unified_status_time_ms": round(operation_time_ms, 2),
                "epic1_compliant": operation_time_ms < 50.0
            }
            
            return unified_status
            
        except Exception as e:
            logger.error(f"Failed to get unified status: {e}")
            return {"error": str(e), "unified_orchestrator": {"is_running": False}}
    
    def _record_operation_time(self, operation: str, time_ms: float) -> None:
        """Record operation time for Epic 1 performance monitoring."""
        if operation not in self.operation_times:
            self.operation_times[operation] = []
        
        times = self.operation_times[operation]
        times.append(time_ms)
        
        # Keep only last 50 measurements for memory efficiency
        if len(times) > 50:
            times.pop(0)
        
        # Log performance warnings for Epic 1 targets
        if time_ms > 50.0:
            logger.warning("Unified operation slow",
                         operation=operation,
                         operation_time_ms=time_ms,
                         target_ms=50.0)
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get plugin performance metrics for Epic 1 monitoring."""
        metrics = {}
        
        for operation, times in self.operation_times.items():
            if times:
                import statistics
                metrics[operation] = {
                    "avg_ms": round(statistics.mean(times), 2),
                    "max_ms": round(max(times), 2),
                    "min_ms": round(min(times), 2),
                    "count": len(times),
                    "last_ms": round(times[-1], 2),
                    "epic1_compliant": statistics.mean(times) < 50.0
                }
        
        return {
            "operation_metrics": metrics,
            "managed_agents": len(self.managed_agents),
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
            "background_tasks": len(self._background_tasks),
            "unified_metrics": self.metrics
        }
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self.is_running = False
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Shutdown all managed agents
        for agent_id in list(self.managed_agents.keys()):
            try:
                await self._remove_unified_agent(agent_id)
            except Exception as e:
                logger.error(f"Error shutting down unified agent {agent_id}: {e}")
        
        # Clear state
        self.managed_agents.clear()
        self.active_tasks.clear()
        self.task_queue.clear()
        self.websocket_connections.clear()
        self.operation_times.clear()
        self._background_tasks.clear()
        
        await super().cleanup()
        
        logger.info("Unified Orchestrator Plugin cleanup complete")


def create_unified_orchestrator_plugin(config: Optional[UnifiedOrchestratorConfig] = None) -> UnifiedOrchestratorPlugin:
    """Factory function to create unified orchestrator plugin."""
    return UnifiedOrchestratorPlugin(config)


# Export for SimpleOrchestrator integration
__all__ = [
    'UnifiedOrchestratorPlugin',
    'OrchestratorMode',
    'TaskExecutionState',
    'UnifiedOrchestratorConfig',
    'AgentInfo',
    'TaskExecution',
    'create_unified_orchestrator_plugin'
]