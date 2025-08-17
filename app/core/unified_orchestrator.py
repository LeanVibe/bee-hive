"""
Unified Agent Orchestrator for LeanVibe Agent Hive 2.0

This is the consolidated orchestrator that merges functionality from 25+ separate 
orchestrator files into a single, plugin-based architecture.

Consolidated Features:
- Central agent lifecycle management
- Task delegation and workflow coordination  
- Performance monitoring and optimization
- Security enforcement and monitoring
- Context compression and memory management
- WebSocket communication and real-time coordination
- Production-ready monitoring and alerting
- Development and testing support

Plugin Architecture:
- PerformancePlugin: Resource monitoring, optimization, metrics collection
- SecurityPlugin: Authentication, authorization, threat detection
- ContextPlugin: Memory management, context compression, session optimization
"""

import asyncio
import json
import os
import uuid
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import heapq

from anthropic import AsyncAnthropic

# Core dependencies
from .config import settings
from .storage_manager import get_session_cache, SessionCache, StorageManager, CacheManager
from .communication_manager import CommunicationManager, MessagingService, Message, MessageType, MessagePriority
from .storage_manager import StorageManager, DatabaseManager
from .workflow_manager import WorkflowManager, WorkflowEngine, WorkflowResult, TaskExecutionState, WorkflowDefinition
from .intelligent_task_router import IntelligentTaskRouter, TaskRoutingContext, RoutingStrategy
from .capability_matcher import CapabilityMatcher
from .agent_persona_system import AgentPersonaSystem, get_agent_persona_system
from .logging_service import get_component_logger

# Plugin system
from .orchestrator_plugins import get_plugin_manager, PluginType
from .orchestrator_plugins.performance_plugin import PerformancePlugin
from .orchestrator_plugins.security_plugin import SecurityPlugin, SecurityError
from .orchestrator_plugins.context_plugin import ContextPlugin

# Data models
from ..models.agent import Agent, AgentStatus, AgentType
from ..models.session import Session, SessionStatus
from ..models.task import Task, TaskStatus, TaskPriority
from ..models.workflow import Workflow, WorkflowStatus
from ..models.agent_performance import AgentPerformanceHistory, TaskRoutingDecision
from sqlalchemy import select, update, func

logger = get_component_logger("unified_orchestrator")


class OrchestratorMode(Enum):
    """Orchestrator operational modes."""
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    TESTING = "testing"
    SANDBOX = "sandbox"


class AgentRole(Enum):
    """Agent roles in the multi-agent system."""
    COORDINATOR = "coordinator"      # Central coordination and task delegation
    SPECIALIST = "specialist"        # Domain-specific expertise
    WORKER = "worker"               # Task execution
    MONITOR = "monitor"             # System monitoring and health checks
    SECURITY = "security"           # Security and compliance
    OPTIMIZER = "optimizer"         # Performance optimization


@dataclass
class OrchestratorConfig:
    """Configuration for the unified orchestrator."""
    mode: OrchestratorMode = OrchestratorMode.PRODUCTION
    max_agents: int = 100
    max_concurrent_tasks: int = 1000
    health_check_interval: int = 30
    cleanup_interval: int = 300
    auto_scaling_enabled: bool = True
    performance_monitoring_enabled: bool = True
    security_monitoring_enabled: bool = True
    context_compression_enabled: bool = True
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


class UnifiedOrchestrator:
    """
    Unified orchestrator that consolidates functionality from multiple orchestrator files.
    
    This orchestrator provides:
    - Agent lifecycle management
    - Task delegation and coordination
    - Performance monitoring and optimization
    - Security enforcement
    - Context compression and memory management
    - Real-time WebSocket communication
    - Production monitoring and alerting
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        
        # Core state
        self.agents: Dict[str, AgentInfo] = {}
        self.active_tasks: Dict[str, TaskExecution] = {}
        self.task_queue: List[Task] = []
        self.is_running = False
        self.startup_time = None
        
        # Plugin manager
        self.plugin_manager = get_plugin_manager()
        
        # Core services (initialized during startup)
        self.redis = None
        self.messaging_service = None
        self.workflow_engine = None
        self.task_router = None
        self.capability_matcher = None
        self.persona_system = None
        
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
        
        # Thread safety
        self._lock = asyncio.Lock()
        
    async def initialize(self) -> bool:
        """Initialize the unified orchestrator."""
        try:
            logger.info("Initializing Unified Orchestrator...")
            
            # Initialize core services
            self.redis = await get_redis()
            self.messaging_service = await get_messaging_service()
            self.workflow_engine = WorkflowEngine()
            self.task_router = IntelligentTaskRouter()
            self.capability_matcher = CapabilityMatcher()
            self.persona_system = await get_agent_persona_system()
            
            # Initialize plugins based on configuration
            await self._initialize_plugins()
            
            # Load existing state from Redis
            await self._load_state()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.startup_time = datetime.utcnow()
            self.is_running = True
            
            logger.info(f"Unified Orchestrator initialized successfully in {self.config.mode.value} mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            await self.cleanup()
            return False
            
    async def _initialize_plugins(self):
        """Initialize orchestrator plugins based on configuration."""
        plugin_context = {
            "orchestrator": self,
            "config": self.config,
            "redis": self.redis,
            "mode": self.config.mode.value
        }
        
        # Always initialize performance plugin if monitoring is enabled
        if self.config.performance_monitoring_enabled:
            performance_plugin = PerformancePlugin()
            self.plugin_manager.register_plugin(performance_plugin)
            
        # Initialize security plugin if security monitoring is enabled
        if self.config.security_monitoring_enabled:
            security_plugin = SecurityPlugin()
            self.plugin_manager.register_plugin(security_plugin)
            
        # Initialize context plugin if context compression is enabled
        if self.config.context_compression_enabled:
            context_plugin = ContextPlugin()
            self.plugin_manager.register_plugin(context_plugin)
            
        # Initialize all plugins
        await self.plugin_manager.initialize_all(plugin_context)
        logger.info(f"Initialized {len(self.plugin_manager._plugin_registry)} plugins")
        
    async def _load_state(self):
        """Load existing orchestrator state from Redis."""
        try:
            # Load agents
            agent_keys = await self.redis.keys("orchestrator:agent:*")
            for key in agent_keys:
                agent_data = await self.redis.get(key)
                if agent_data:
                    agent_dict = json.loads(agent_data)
                    agent_info = AgentInfo(
                        agent_id=agent_dict["agent_id"],
                        agent_type=AgentType(agent_dict["agent_type"]),
                        role=AgentRole(agent_dict["role"]),
                        status=AgentStatus(agent_dict["status"]),
                        persona=agent_dict.get("persona"),
                        capabilities=set(agent_dict.get("capabilities", [])),
                        current_task_id=agent_dict.get("current_task_id"),
                        performance_score=agent_dict.get("performance_score", 0.5),
                        last_heartbeat=datetime.fromisoformat(agent_dict["last_heartbeat"]),
                        created_at=datetime.fromisoformat(agent_dict["created_at"]),
                        container_id=agent_dict.get("container_id")
                    )
                    self.agents[agent_info.agent_id] = agent_info
                    
            # Load active tasks
            task_keys = await self.redis.keys("orchestrator:task:*")
            for key in task_keys:
                task_data = await self.redis.get(key)
                if task_data:
                    task_dict = json.loads(task_data)
                    task_execution = TaskExecution(
                        task_id=task_dict["task_id"],
                        agent_id=task_dict["agent_id"],
                        status=TaskExecutionState(task_dict["status"]),
                        started_at=datetime.fromisoformat(task_dict["started_at"]),
                        estimated_completion=datetime.fromisoformat(task_dict["estimated_completion"]) if task_dict.get("estimated_completion") else None,
                        progress_percentage=task_dict.get("progress_percentage", 0.0),
                        context_size_bytes=task_dict.get("context_size_bytes", 0),
                        performance_metrics=task_dict.get("performance_metrics", {})
                    )
                    self.active_tasks[task_execution.task_id] = task_execution
                    
            logger.info(f"Loaded state: {len(self.agents)} agents, {len(self.active_tasks)} active tasks")
            
        except Exception as e:
            logger.error(f"Error loading orchestrator state: {e}")
            
    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks."""
        # Health monitoring
        health_task = asyncio.create_task(self._health_monitoring_loop())
        self._background_tasks.add(health_task)
        
        # Cleanup and maintenance
        cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._background_tasks.add(cleanup_task)
        
        # Auto-scaling (if enabled)
        if self.config.auto_scaling_enabled:
            scaling_task = asyncio.create_task(self._auto_scaling_loop())
            self._background_tasks.add(scaling_task)
            
        # Task processing
        task_processing_task = asyncio.create_task(self._task_processing_loop())
        self._background_tasks.add(task_processing_task)
        
        logger.info(f"Started {len(self._background_tasks)} background tasks")
        
    async def cleanup(self):
        """Cleanup orchestrator resources."""
        try:
            logger.info("Shutting down Unified Orchestrator...")
            
            self.is_running = False
            self._shutdown_event.set()
            
            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()
                
            # Wait for tasks to complete with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._background_tasks, return_exceptions=True),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.warning("Background tasks did not complete within timeout")
                
            # Save state to Redis
            await self._save_state()
            
            # Cleanup plugins
            await self.plugin_manager.cleanup_all()
            
            # Shutdown agents gracefully
            await self._shutdown_all_agents()
            
            logger.info("Unified Orchestrator shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during orchestrator cleanup: {e}")
            
    async def _save_state(self):
        """Save orchestrator state to Redis."""
        try:
            # Save agents
            for agent_info in self.agents.values():
                agent_data = {
                    "agent_id": agent_info.agent_id,
                    "agent_type": agent_info.agent_type.value,
                    "role": agent_info.role.value,
                    "status": agent_info.status.value,
                    "persona": agent_info.persona,
                    "capabilities": list(agent_info.capabilities),
                    "current_task_id": agent_info.current_task_id,
                    "performance_score": agent_info.performance_score,
                    "last_heartbeat": agent_info.last_heartbeat.isoformat(),
                    "created_at": agent_info.created_at.isoformat(),
                    "container_id": agent_info.container_id
                }
                
                await self.redis.set(
                    f"orchestrator:agent:{agent_info.agent_id}",
                    json.dumps(agent_data),
                    ex=3600  # 1 hour TTL
                )
                
            # Save active tasks
            for task_execution in self.active_tasks.values():
                task_data = {
                    "task_id": task_execution.task_id,
                    "agent_id": task_execution.agent_id,
                    "status": task_execution.status.value,
                    "started_at": task_execution.started_at.isoformat(),
                    "estimated_completion": task_execution.estimated_completion.isoformat() if task_execution.estimated_completion else None,
                    "progress_percentage": task_execution.progress_percentage,
                    "context_size_bytes": task_execution.context_size_bytes,
                    "performance_metrics": task_execution.performance_metrics
                }
                
                await self.redis.set(
                    f"orchestrator:task:{task_execution.task_id}",
                    json.dumps(task_data),
                    ex=3600  # 1 hour TTL
                )
                
        except Exception as e:
            logger.error(f"Error saving orchestrator state: {e}")
            
    async def spawn_agent(
        self,
        agent_type: AgentType,
        role: AgentRole,
        capabilities: Optional[Set[str]] = None,
        persona: Optional[str] = None,
        **kwargs
    ) -> str:
        """Spawn a new agent."""
        async with self._lock:
            try:
                # Check limits
                if len(self.agents) >= self.config.max_agents:
                    raise ValueError(f"Maximum agent limit ({self.config.max_agents}) reached")
                    
                # Generate agent ID
                agent_id = f"agent_{uuid.uuid4().hex[:8]}"
                
                # Get persona if not provided
                if not persona and self.persona_system:
                    persona = await self.persona_system.assign_persona(agent_type, role)
                    
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
                
                # Store agent
                self.agents[agent_id] = agent_info
                
                # Initialize agent (implementation would depend on agent type)
                await self._initialize_agent(agent_info, **kwargs)
                
                # Update metrics
                self.metrics["agents_spawned"] += 1
                
                logger.info(f"Spawned agent {agent_id} with role {role.value}")
                return agent_id
                
            except Exception as e:
                logger.error(f"Failed to spawn agent: {e}")
                raise
                
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
        return capabilities_map.get(role, set())
        
    async def _initialize_agent(self, agent_info: AgentInfo, **kwargs):
        """Initialize a newly spawned agent."""
        try:
            # Set status to starting
            agent_info.status = AgentStatus.STARTING
            
            # Initialize based on agent type
            if agent_info.agent_type == AgentType.ANTHROPIC_CLAUDE:
                # Initialize Claude agent
                await self._initialize_claude_agent(agent_info, **kwargs)
            elif agent_info.agent_type == AgentType.CONTAINER:
                # Initialize container-based agent
                await self._initialize_container_agent(agent_info, **kwargs)
            else:
                # Initialize other agent types
                await self._initialize_generic_agent(agent_info, **kwargs)
                
            # Mark as ready
            agent_info.status = AgentStatus.IDLE
            agent_info.last_heartbeat = datetime.utcnow()
            
        except Exception as e:
            agent_info.status = AgentStatus.ERROR
            logger.error(f"Failed to initialize agent {agent_info.agent_id}: {e}")
            raise
            
    async def _initialize_claude_agent(self, agent_info: AgentInfo, **kwargs):
        """Initialize a Claude-based agent."""
        # Implementation would initialize Claude API connection
        # For now, just mark as initialized
        logger.debug(f"Initializing Claude agent {agent_info.agent_id}")
        
    async def _initialize_container_agent(self, agent_info: AgentInfo, **kwargs):
        """Initialize a container-based agent."""
        # Implementation would spawn container and establish communication
        # For now, just mark as initialized
        logger.debug(f"Initializing container agent {agent_info.agent_id}")
        
    async def _initialize_generic_agent(self, agent_info: AgentInfo, **kwargs):
        """Initialize a generic agent."""
        # Implementation would initialize generic agent
        # For now, just mark as initialized
        logger.debug(f"Initializing generic agent {agent_info.agent_id}")
        
    async def delegate_task(
        self,
        task: Task,
        preferred_agent_id: Optional[str] = None,
        required_capabilities: Optional[Set[str]] = None,
        security_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Delegate a task to an appropriate agent."""
        try:
            # Security validation through plugin
            if self.config.security_monitoring_enabled:
                security_plugins = self.plugin_manager.get_plugins(PluginType.SECURITY)
                for plugin in security_plugins:
                    if plugin.enabled:
                        task_context = {
                            "task_id": task.id,
                            "user_id": security_context.get("user_id") if security_context else None,
                            "source_ip": security_context.get("source_ip") if security_context else None,
                            "agent_id": preferred_agent_id
                        }
                        task_context = await plugin.pre_task_execution(task_context)
                        
            # Find suitable agent
            if preferred_agent_id and preferred_agent_id in self.agents:
                agent_info = self.agents[preferred_agent_id]
                if agent_info.status == AgentStatus.IDLE:
                    selected_agent_id = preferred_agent_id
                else:
                    raise ValueError(f"Preferred agent {preferred_agent_id} is not available")
            else:
                selected_agent_id = await self._select_agent_for_task(task, required_capabilities)
                
            if not selected_agent_id:
                # Add to queue if no agent available
                self.task_queue.append(task)
                logger.info(f"Task {task.id} queued - no suitable agents available")
                return task.id
                
            # Create task execution
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
            
            # Store task execution
            self.active_tasks[task.id] = task_execution
            
            # Update agent
            agent_info = self.agents[selected_agent_id]
            agent_info.current_task_id = task.id
            agent_info.status = AgentStatus.BUSY
            
            # Start task execution
            execution_task = asyncio.create_task(
                self._execute_task(task_execution, task)
            )
            self._background_tasks.add(execution_task)
            
            logger.info(f"Delegated task {task.id} to agent {selected_agent_id}")
            return task.id
            
        except SecurityError as e:
            logger.error(f"Security error delegating task {task.id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to delegate task {task.id}: {e}")
            raise
            
    async def _select_agent_for_task(
        self,
        task: Task,
        required_capabilities: Optional[Set[str]] = None
    ) -> Optional[str]:
        """Select the best agent for a task."""
        available_agents = [
            agent for agent in self.agents.values()
            if agent.status == AgentStatus.IDLE
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
            
        # Use task router to select best agent
        routing_context = TaskRoutingContext(
            task=task,
            available_agents=[
                {
                    "agent_id": agent.agent_id,
                    "capabilities": agent.capabilities,
                    "performance_score": agent.performance_score,
                    "current_load": 0.0  # Since agent is idle
                }
                for agent in available_agents
            ]
        )
        
        routing_result = await self.task_router.route_task(routing_context)
        return routing_result.selected_agent_id if routing_result else None
        
    async def _execute_task(self, task_execution: TaskExecution, task: Task):
        """Execute a task on an agent."""
        try:
            task_execution.status = TaskExecutionState.RUNNING
            task_execution.started_at = datetime.utcnow()
            
            # Context optimization through plugin
            task_context = {
                "task_id": task.id,
                "agent_id": task_execution.agent_id,
                "session_id": getattr(task, 'session_id', None)
            }
            
            if self.config.context_compression_enabled:
                context_plugins = self.plugin_manager.get_plugins(PluginType.CONTEXT)
                for plugin in context_plugins:
                    if plugin.enabled:
                        task_context = await plugin.pre_task_execution(task_context)
                        
            # Performance monitoring
            if self.config.performance_monitoring_enabled:
                performance_plugins = self.plugin_manager.get_plugins(PluginType.PERFORMANCE)
                for plugin in performance_plugins:
                    if plugin.enabled:
                        task_context = await plugin.pre_task_execution(task_context)
                        
            # Execute task (implementation would depend on task type and agent)
            result = await self._perform_task_execution(task_execution, task, task_context)
            
            # Post-execution hooks
            if self.config.context_compression_enabled:
                context_plugins = self.plugin_manager.get_plugins(PluginType.CONTEXT)
                for plugin in context_plugins:
                    if plugin.enabled:
                        result = await plugin.post_task_execution(task_context, result)
                        
            if self.config.performance_monitoring_enabled:
                performance_plugins = self.plugin_manager.get_plugins(PluginType.PERFORMANCE)
                for plugin in performance_plugins:
                    if plugin.enabled:
                        result = await plugin.post_task_execution(task_context, result)
                        
            if self.config.security_monitoring_enabled:
                security_plugins = self.plugin_manager.get_plugins(PluginType.SECURITY)
                for plugin in security_plugins:
                    if plugin.enabled:
                        result = await plugin.post_task_execution(task_context, result)
                        
            # Mark as completed
            task_execution.status = TaskExecutionState.COMPLETED
            task_execution.progress_percentage = 100.0
            
            # Update agent
            agent_info = self.agents[task_execution.agent_id]
            agent_info.current_task_id = None
            agent_info.status = AgentStatus.IDLE
            agent_info.last_heartbeat = datetime.utcnow()
            
            # Update metrics
            execution_time = (datetime.utcnow() - task_execution.started_at).total_seconds()
            self.metrics["tasks_completed"] += 1
            self.metrics["total_execution_time"] += execution_time
            self.metrics["avg_response_time"] = self.metrics["total_execution_time"] / self.metrics["tasks_completed"]
            
            # Clean up
            del self.active_tasks[task.id]
            
            logger.info(f"Task {task.id} completed successfully in {execution_time:.2f}s")
            
        except Exception as e:
            # Mark as failed
            task_execution.status = TaskExecutionState.FAILED
            
            # Update agent
            agent_info = self.agents[task_execution.agent_id]
            agent_info.current_task_id = None
            agent_info.status = AgentStatus.ERROR
            
            # Update metrics
            self.metrics["tasks_failed"] += 1
            
            logger.error(f"Task {task.id} failed: {e}")
            
        finally:
            # Remove from background tasks
            current_task = asyncio.current_task()
            if current_task in self._background_tasks:
                self._background_tasks.remove(current_task)
                
    async def _perform_task_execution(
        self,
        task_execution: TaskExecution,
        task: Task,
        context: Dict[str, Any]
    ) -> Any:
        """Perform the actual task execution."""
        # This is a simplified implementation
        # In production, this would route to the appropriate agent implementation
        
        # Simulate task execution
        await asyncio.sleep(1.0)  # Simulate work
        
        return {
            "task_id": task.id,
            "result": "Task completed successfully",
            "execution_time": 1.0,
            "agent_id": task_execution.agent_id
        }
        
    async def _health_monitoring_loop(self):
        """Background task for health monitoring."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(self.config.health_check_interval)
                
    async def _perform_health_checks(self):
        """Perform health checks on all components."""
        # Check agent health
        unhealthy_agents = []
        for agent_id, agent_info in self.agents.items():
            time_since_heartbeat = datetime.utcnow() - agent_info.last_heartbeat
            if time_since_heartbeat > timedelta(seconds=self.config.health_check_interval * 3):
                unhealthy_agents.append(agent_id)
                agent_info.status = AgentStatus.ERROR
                
        if unhealthy_agents:
            logger.warning(f"Found {len(unhealthy_agents)} unhealthy agents")
            
        # Check plugin health
        for plugin_type in PluginType:
            plugins = self.plugin_manager.get_plugins(plugin_type)
            for plugin in plugins:
                if plugin.enabled:
                    health_status = await plugin.health_check()
                    if health_status.get("status") != "healthy":
                        logger.warning(f"Plugin {plugin.metadata.name} health check failed: {health_status}")
                        
    async def _cleanup_loop(self):
        """Background task for cleanup and maintenance."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_cleanup()
                await asyncio.sleep(self.config.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(self.config.cleanup_interval)
                
    async def _perform_cleanup(self):
        """Perform cleanup and maintenance tasks."""
        # Clean up completed tasks
        completed_tasks = [
            task_id for task_id, task_execution in self.active_tasks.items()
            if task_execution.status in [TaskExecutionState.COMPLETED, TaskExecutionState.FAILED]
        ]
        
        for task_id in completed_tasks:
            del self.active_tasks[task_id]
            
        # Remove failed agents
        failed_agents = [
            agent_id for agent_id, agent_info in self.agents.items()
            if agent_info.status == AgentStatus.ERROR
        ]
        
        for agent_id in failed_agents:
            await self._remove_agent(agent_id)
            
        # Save state periodically
        await self._save_state()
        
    async def _auto_scaling_loop(self):
        """Background task for auto-scaling."""
        while not self._shutdown_event.is_set():
            try:
                await self._perform_auto_scaling()
                await asyncio.sleep(60)  # Check every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-scaling: {e}")
                await asyncio.sleep(60)
                
    async def _perform_auto_scaling(self):
        """Perform auto-scaling based on load."""
        # Calculate current load
        idle_agents = len([a for a in self.agents.values() if a.status == AgentStatus.IDLE])
        busy_agents = len([a for a in self.agents.values() if a.status == AgentStatus.BUSY])
        total_agents = len(self.agents)
        
        if total_agents == 0:
            return
            
        load_ratio = busy_agents / total_agents
        queue_size = len(self.task_queue)
        
        # Scale up if needed
        if (load_ratio > self.config.scale_up_threshold or queue_size > 5) and total_agents < self.config.max_agents:
            agents_to_spawn = min(
                self.config.max_scale_up_rate,
                self.config.max_agents - total_agents,
                queue_size  # Don't spawn more than queued tasks
            )
            
            for _ in range(agents_to_spawn):
                try:
                    await self.spawn_agent(
                        agent_type=AgentType.ANTHROPIC_CLAUDE,
                        role=AgentRole.WORKER
                    )
                except Exception as e:
                    logger.error(f"Failed to spawn agent during auto-scaling: {e}")
                    break
                    
            logger.info(f"Auto-scaled up: spawned {agents_to_spawn} agents")
            
        # Scale down if needed
        elif load_ratio < self.config.scale_down_threshold and total_agents > self.config.min_agents:
            excess_agents = min(
                total_agents - self.config.min_agents,
                idle_agents // 2  # Only remove half of idle agents
            )
            
            # Remove idle agents
            idle_agent_ids = [
                agent_id for agent_id, agent_info in self.agents.items()
                if agent_info.status == AgentStatus.IDLE
            ]
            
            for i, agent_id in enumerate(idle_agent_ids[:excess_agents]):
                try:
                    await self._remove_agent(agent_id)
                except Exception as e:
                    logger.error(f"Failed to remove agent during auto-scaling: {e}")
                    
            if excess_agents > 0:
                logger.info(f"Auto-scaled down: removed {excess_agents} agents")
                
    async def _task_processing_loop(self):
        """Background task for processing queued tasks."""
        while not self._shutdown_event.is_set():
            try:
                if self.task_queue:
                    # Try to assign queued tasks to available agents
                    tasks_to_remove = []
                    
                    for task in self.task_queue:
                        agent_id = await self._select_agent_for_task(task)
                        if agent_id:
                            # Move task from queue to execution
                            await self.delegate_task(task, preferred_agent_id=agent_id)
                            tasks_to_remove.append(task)
                            
                    # Remove assigned tasks from queue
                    for task in tasks_to_remove:
                        self.task_queue.remove(task)
                        
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in task processing loop: {e}")
                await asyncio.sleep(5)
                
    async def _remove_agent(self, agent_id: str):
        """Remove an agent from the orchestrator."""
        if agent_id not in self.agents:
            return
            
        agent_info = self.agents[agent_id]
        
        # If agent has an active task, mark it as failed
        if agent_info.current_task_id and agent_info.current_task_id in self.active_tasks:
            task_execution = self.active_tasks[agent_info.current_task_id]
            task_execution.status = TaskExecutionState.FAILED
            
        # Remove from Redis
        await self.redis.delete(f"orchestrator:agent:{agent_id}")
        
        # Remove from memory
        del self.agents[agent_id]
        
        # Update metrics
        self.metrics["agents_terminated"] += 1
        
        logger.info(f"Removed agent {agent_id}")
        
    async def _shutdown_all_agents(self):
        """Shutdown all agents gracefully."""
        for agent_id in list(self.agents.keys()):
            try:
                await self._remove_agent(agent_id)
            except Exception as e:
                logger.error(f"Error shutting down agent {agent_id}: {e}")
                
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status."""
        # Collect plugin health statuses
        plugin_health = {}
        for plugin_type in PluginType:
            plugins = self.plugin_manager.get_plugins(plugin_type)
            for plugin in plugins:
                if plugin.enabled:
                    health_status = await plugin.health_check()
                    plugin_health[plugin.metadata.name] = health_status
                    
        # Agent status breakdown
        agent_status_counts = {}
        for status in AgentStatus:
            agent_status_counts[status.value] = len([
                a for a in self.agents.values() if a.status == status
            ])
            
        # Task status breakdown
        task_status_counts = {}
        for status in TaskExecutionState:
            task_status_counts[status.value] = len([
                t for t in self.active_tasks.values() if t.status == status
            ])
            
        uptime = (datetime.utcnow() - self.startup_time).total_seconds() if self.startup_time else 0
        
        return {
            "orchestrator": {
                "mode": self.config.mode.value,
                "is_running": self.is_running,
                "uptime_seconds": uptime,
                "startup_time": self.startup_time.isoformat() if self.startup_time else None
            },
            "agents": {
                "total": len(self.agents),
                "status_breakdown": agent_status_counts,
                "max_agents": self.config.max_agents
            },
            "tasks": {
                "active": len(self.active_tasks),
                "queued": len(self.task_queue),
                "status_breakdown": task_status_counts,
                "max_concurrent": self.config.max_concurrent_tasks
            },
            "metrics": self.metrics,
            "plugins": plugin_health,
            "background_tasks": len(self._background_tasks)
        }


# Global orchestrator instance
_orchestrator: Optional[UnifiedOrchestrator] = None


async def get_orchestrator(config: Optional[OrchestratorConfig] = None) -> UnifiedOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator
    
    if _orchestrator is None:
        _orchestrator = UnifiedOrchestrator(config)
        await _orchestrator.initialize()
        
    return _orchestrator


async def shutdown_orchestrator():
    """Shutdown the global orchestrator instance."""
    global _orchestrator
    
    if _orchestrator:
        await _orchestrator.cleanup()
        _orchestrator = None