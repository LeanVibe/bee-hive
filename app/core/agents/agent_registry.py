"""
Agent Registry for Multi-CLI Coordination

This module provides centralized registration and discovery of CLI agents
participating in coordinated workflows. It manages agent lifecycle,
capability tracking, load balancing, and health monitoring.

Key Components:
- AgentRegistry: Central registry for agent management
- Agent discovery and registration
- Capability-based routing
- Load balancing and health monitoring
- Agent lifecycle management
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

from .universal_agent_interface import (
    UniversalAgentInterface,
    AgentType,
    AgentCapability,
    CapabilityType,
    HealthStatus,
    HealthState,
    AgentTask,
    AgentResult
)
from .models import (
    AgentStatus,
    AgentConfiguration,
    SystemStatus,
    ErrorReport,
    ErrorSeverity
)

logger = logging.getLogger(__name__)

# ================================================================================
# Agent Registry Core
# ================================================================================

class AgentRegistrationError(Exception):
    """Raised when agent registration fails"""
    pass

class AgentNotFoundError(Exception):
    """Raised when requested agent is not found"""
    pass

class NoCapableAgentError(Exception):
    """Raised when no agent can handle requested capability"""
    pass

@dataclass
class RegisteredAgent:
    """Information about a registered agent"""
    interface: UniversalAgentInterface
    configuration: AgentConfiguration
    status: AgentStatus
    last_health_check: datetime = field(default_factory=datetime.utcnow)
    registration_time: datetime = field(default_factory=datetime.utcnow)
    capabilities: List[AgentCapability] = field(default_factory=list)
    task_history: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def update_health(self, health: HealthStatus) -> None:
        """Update agent health information"""
        self.status.health = health.state
        self.status.current_tasks = health.active_tasks
        self.status.load_score = health.calculate_load_score()
        self.status.last_activity = datetime.utcnow()
        self.last_health_check = datetime.utcnow()
        
        # Update performance score based on health
        if health.error_rate > 0.15:  # High error rate
            self.status.performance_score *= 0.9
        elif health.error_rate < 0.05:  # Low error rate
            self.status.performance_score = min(1.0, self.status.performance_score * 1.01)

class AgentRegistry:
    """
    Central registry for managing CLI agents in the coordination system.
    
    Provides agent registration, discovery, capability routing, load balancing,
    and health monitoring for heterogeneous CLI agent coordination.
    
    Features:
    - Agent registration and lifecycle management
    - Capability-based task routing
    - Load balancing across agents
    - Health monitoring and failure detection
    - Performance tracking and optimization
    - Event-driven notifications
    
    Thread Safety:
    - All methods are thread-safe
    - Supports concurrent agent operations
    - Uses async/await for non-blocking operations
    """
    
    def __init__(self):
        self._agents: Dict[str, RegisteredAgent] = {}
        self._agents_by_type: Dict[AgentType, Set[str]] = {}
        self._capabilities_index: Dict[CapabilityType, Set[str]] = {}
        self._lock = threading.RLock()
        self._health_check_interval = 60  # seconds
        self._health_check_task: Optional[asyncio.Task] = None
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._shutdown = False
        
        # Initialize capability index
        for capability in CapabilityType:
            self._capabilities_index[capability] = set()
        
        # Initialize agent type index
        for agent_type in AgentType:
            self._agents_by_type[agent_type] = set()
    
    # ================================================================================
    # Agent Registration and Lifecycle
    # ================================================================================
    
    async def register_agent(
        self,
        agent: UniversalAgentInterface,
        configuration: AgentConfiguration
    ) -> bool:
        """
        Register a new agent in the coordination system.
        
        Args:
            agent: Agent interface implementation
            configuration: Agent configuration
            
        Returns:
            bool: True if registration successful
            
        Raises:
            AgentRegistrationError: If registration fails
        """
        try:
            with self._lock:
                if agent.agent_id in self._agents:
                    raise AgentRegistrationError(f"Agent {agent.agent_id} already registered")
                
                # Validate configuration
                config_errors = configuration.validate()
                if config_errors:
                    raise AgentRegistrationError(f"Invalid configuration: {config_errors}")
                
                # Initialize agent
                initialization_success = await agent.initialize(configuration.custom_settings)
                if not initialization_success:
                    raise AgentRegistrationError(f"Agent {agent.agent_id} initialization failed")
                
                # Get agent capabilities
                capabilities = await agent.get_capabilities()
                
                # Create agent status
                status = AgentStatus(
                    agent_id=agent.agent_id,
                    agent_type=agent.agent_type,
                    health=HealthState.HEALTHY,
                    capabilities=capabilities,
                    last_activity=datetime.utcnow()
                )
                
                # Create registered agent record
                registered_agent = RegisteredAgent(
                    interface=agent,
                    configuration=configuration,
                    status=status,
                    capabilities=capabilities
                )
                
                # Add to registry
                self._agents[agent.agent_id] = registered_agent
                self._agents_by_type[agent.agent_type].add(agent.agent_id)
                
                # Update capability index
                for capability in capabilities:
                    self._capabilities_index[capability.type].add(agent.agent_id)
                
                # Start health monitoring if this is the first agent
                if len(self._agents) == 1 and not self._health_check_task:
                    self._health_check_task = asyncio.create_task(self._health_check_loop())
                
                # Emit registration event
                await self._emit_event("agent_registered", {
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type.value,
                    "capabilities": [cap.type.value for cap in capabilities]
                })
                
                logger.info(f"Agent {agent.agent_id} ({agent.agent_type.value}) registered successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register agent {agent.agent_id}: {e}")
            raise AgentRegistrationError(f"Registration failed: {e}")
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent from the coordination system.
        
        Args:
            agent_id: ID of agent to unregister
            
        Returns:
            bool: True if unregistration successful
        """
        try:
            with self._lock:
                if agent_id not in self._agents:
                    logger.warning(f"Attempted to unregister unknown agent: {agent_id}")
                    return False
                
                registered_agent = self._agents[agent_id]
                
                # Shutdown agent gracefully
                try:
                    await registered_agent.interface.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down agent {agent_id}: {e}")
                
                # Remove from capability index
                for capability in registered_agent.capabilities:
                    self._capabilities_index[capability.type].discard(agent_id)
                
                # Remove from type index
                self._agents_by_type[registered_agent.interface.agent_type].discard(agent_id)
                
                # Remove from main registry
                del self._agents[agent_id]
                
                # Emit unregistration event
                await self._emit_event("agent_unregistered", {
                    "agent_id": agent_id,
                    "agent_type": registered_agent.interface.agent_type.value
                })
                
                logger.info(f"Agent {agent_id} unregistered successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to unregister agent {agent_id}: {e}")
            return False
    
    # ================================================================================
    # Agent Discovery and Querying
    # ================================================================================
    
    def get_agent(self, agent_id: str) -> Optional[UniversalAgentInterface]:
        """
        Get agent interface by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            UniversalAgentInterface: Agent interface or None if not found
        """
        with self._lock:
            registered_agent = self._agents.get(agent_id)
            return registered_agent.interface if registered_agent else None
    
    def list_agents(
        self,
        agent_type: Optional[AgentType] = None,
        health_state: Optional[HealthState] = None,
        capability: Optional[CapabilityType] = None
    ) -> List[str]:
        """
        List agents matching specified criteria.
        
        Args:
            agent_type: Filter by agent type
            health_state: Filter by health state
            capability: Filter by capability
            
        Returns:
            List[str]: List of matching agent IDs
        """
        with self._lock:
            matching_agents = []
            
            for agent_id, registered_agent in self._agents.items():
                # Filter by agent type
                if agent_type and registered_agent.interface.agent_type != agent_type:
                    continue
                
                # Filter by health state
                if health_state and registered_agent.status.health != health_state:
                    continue
                
                # Filter by capability
                if capability:
                    has_capability = any(
                        cap.type == capability for cap in registered_agent.capabilities
                    )
                    if not has_capability:
                        continue
                
                matching_agents.append(agent_id)
            
            return matching_agents
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[str]:
        """Get all agents of specified type."""
        with self._lock:
            return list(self._agents_by_type.get(agent_type, set()))
    
    def get_agents_by_capability(self, capability: CapabilityType) -> List[str]:
        """Get all agents with specified capability."""
        with self._lock:
            return list(self._capabilities_index.get(capability, set()))
    
    # ================================================================================
    # Capability-Based Routing
    # ================================================================================
    
    async def find_best_agent(
        self,
        task: AgentTask,
        exclude_agents: Optional[Set[str]] = None
    ) -> Optional[str]:
        """
        Find the best agent for a given task based on capabilities and load.
        
        Args:
            task: Task to be executed
            exclude_agents: Set of agent IDs to exclude from selection
            
        Returns:
            str: ID of best agent, or None if no suitable agent found
        """
        exclude_agents = exclude_agents or set()
        
        with self._lock:
            # Get agents with required capability
            capable_agents = self._capabilities_index.get(task.type, set())
            available_agents = capable_agents - exclude_agents
            
            if not available_agents:
                return None
            
            # Score agents based on multiple factors
            agent_scores = []
            
            for agent_id in available_agents:
                registered_agent = self._agents.get(agent_id)
                if not registered_agent:
                    continue
                
                # Skip unhealthy agents
                if registered_agent.status.health == HealthState.UNHEALTHY:
                    continue
                
                # Find capability match
                capability_match = None
                for cap in registered_agent.capabilities:
                    if cap.type == task.type:
                        capability_match = cap
                        break
                
                if not capability_match:
                    continue
                
                # Calculate composite score
                score = self._calculate_agent_score(registered_agent, capability_match, task)
                agent_scores.append((agent_id, score))
            
            if not agent_scores:
                return None
            
            # Sort by score (higher is better) and return best agent
            agent_scores.sort(key=lambda x: x[1], reverse=True)
            return agent_scores[0][0]
    
    def _calculate_agent_score(
        self,
        registered_agent: RegisteredAgent,
        capability: AgentCapability,
        task: AgentTask
    ) -> float:
        """Calculate agent suitability score for a task."""
        # Base score from capability confidence and performance
        score = capability.confidence * capability.performance_score
        
        # Adjust for agent performance history
        score *= registered_agent.status.performance_score
        
        # Adjust for current load (prefer less loaded agents)
        load_penalty = registered_agent.status.load_score * 0.3
        score *= (1.0 - load_penalty)
        
        # Adjust for health state
        if registered_agent.status.health == HealthState.DEGRADED:
            score *= 0.7
        elif registered_agent.status.health == HealthState.MAINTENANCE:
            score *= 0.5
        
        # Priority bonus (higher priority tasks get preference for better agents)
        if task.priority <= 3:  # High priority
            score *= 1.2
        elif task.priority >= 8:  # Low priority
            score *= 0.8
        
        return score
    
    async def route_task(
        self,
        task: AgentTask,
        max_retries: int = 3
    ) -> Optional[AgentResult]:
        """
        Route task to best available agent and execute.
        
        Args:
            task: Task to execute
            max_retries: Maximum retry attempts
            
        Returns:
            AgentResult: Task execution result or None if all attempts failed
        """
        exclude_agents = set()
        
        for attempt in range(max_retries):
            # Find best agent for this task
            agent_id = await self.find_best_agent(task, exclude_agents)
            
            if not agent_id:
                logger.warning(f"No capable agent found for task {task.id} (attempt {attempt + 1})")
                continue
            
            try:
                # Execute task
                registered_agent = self._agents[agent_id]
                result = await registered_agent.interface.execute_task(task)
                
                # Update agent performance
                success = result.status.value in ["completed"]
                registered_agent.status.update_performance(success, result.execution_time_seconds)
                
                # Update task history
                registered_agent.task_history.append(task.id)
                if len(registered_agent.task_history) > 100:  # Keep last 100 tasks
                    registered_agent.task_history = registered_agent.task_history[-100:]
                
                return result
                
            except Exception as e:
                logger.error(f"Task {task.id} failed on agent {agent_id}: {e}")
                exclude_agents.add(agent_id)
                
                # Report error
                await self._report_error(ErrorReport(
                    severity=ErrorSeverity.MEDIUM,
                    source_agent=agent_id,
                    error_type="task_execution_error",
                    error_message=str(e),
                    related_task_id=task.id
                ))
        
        return None
    
    # ================================================================================
    # Health Monitoring
    # ================================================================================
    
    async def _health_check_loop(self) -> None:
        """Continuous health monitoring loop."""
        while not self._shutdown:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self._health_check_interval)
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all registered agents."""
        with self._lock:
            agent_ids = list(self._agents.keys())
        
        # Perform health checks concurrently
        health_check_tasks = []
        for agent_id in agent_ids:
            task = asyncio.create_task(self._check_agent_health(agent_id))
            health_check_tasks.append(task)
        
        if health_check_tasks:
            await asyncio.gather(*health_check_tasks, return_exceptions=True)
    
    async def _check_agent_health(self, agent_id: str) -> None:
        """Check health of a specific agent."""
        try:
            with self._lock:
                registered_agent = self._agents.get(agent_id)
                if not registered_agent:
                    return
            
            # Perform health check
            health = await registered_agent.interface.health_check()
            
            with self._lock:
                if agent_id in self._agents:  # Agent might have been unregistered
                    old_health = self._agents[agent_id].status.health
                    self._agents[agent_id].update_health(health)
                    
                    # Emit health change event if state changed
                    if old_health != health.state:
                        await self._emit_event("agent_health_changed", {
                            "agent_id": agent_id,
                            "old_health": old_health.value,
                            "new_health": health.state.value
                        })
        
        except Exception as e:
            logger.error(f"Health check failed for agent {agent_id}: {e}")
            
            # Mark agent as unhealthy if health check fails
            with self._lock:
                if agent_id in self._agents:
                    self._agents[agent_id].status.health = HealthState.UNHEALTHY
    
    # ================================================================================
    # System Status and Monitoring
    # ================================================================================
    
    def get_system_status(self) -> SystemStatus:
        """Get overall system status and health."""
        with self._lock:
            total_agents = len(self._agents)
            active_agents = sum(
                1 for agent in self._agents.values()
                if agent.status.health in [HealthState.HEALTHY, HealthState.DEGRADED]
            )
            
            active_tasks = sum(len(agent.status.current_tasks) for agent in self._agents.values())
            completed_tasks = sum(agent.status.completed_tasks for agent in self._agents.values())
            failed_tasks = sum(agent.status.failed_tasks for agent in self._agents.values())
            
            # Calculate system load
            if total_agents > 0:
                avg_load = sum(agent.status.load_score for agent in self._agents.values()) / total_agents
            else:
                avg_load = 0.0
            
            status = SystemStatus(
                total_agents=total_agents,
                active_agents=active_agents,
                active_tasks=active_tasks,
                completed_tasks=completed_tasks,
                failed_tasks=failed_tasks,
                system_load=avg_load
            )
            
            status.overall_health = status.calculate_health()
            return status
    
    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """Get status of specific agent."""
        with self._lock:
            registered_agent = self._agents.get(agent_id)
            return registered_agent.status if registered_agent else None
    
    # ================================================================================
    # Event Handling
    # ================================================================================
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add event handler for registry events."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: str, handler: Callable) -> bool:
        """Remove event handler."""
        if event_type in self._event_handlers:
            try:
                self._event_handlers[event_type].remove(handler)
                return True
            except ValueError:
                pass
        return False
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit event to registered handlers."""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event_type, data)
                else:
                    handler(event_type, data)
            except Exception as e:
                logger.error(f"Event handler error for {event_type}: {e}")
    
    async def _report_error(self, error: ErrorReport) -> None:
        """Report error to monitoring system."""
        await self._emit_event("error_reported", {
            "error_id": error.id,
            "severity": error.severity.value,
            "source_agent": error.source_agent,
            "error_type": error.error_type,
            "message": error.error_message,
            "timestamp": error.timestamp.isoformat()
        })
    
    # ================================================================================
    # Cleanup and Shutdown
    # ================================================================================
    
    async def shutdown(self) -> None:
        """Shutdown the agent registry."""
        self._shutdown = True
        
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Shutdown all agents
        with self._lock:
            agent_ids = list(self._agents.keys())
        
        for agent_id in agent_ids:
            await self.unregister_agent(agent_id)
        
        # Shutdown executor
        self._executor.shutdown(wait=True)
        
        logger.info("Agent registry shutdown completed")

# ================================================================================
# Global Registry Instance
# ================================================================================

_global_registry: Optional[AgentRegistry] = None

def get_registry() -> AgentRegistry:
    """Get global agent registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry

# ================================================================================
# Convenience Functions
# ================================================================================

async def register_agent(
    agent: UniversalAgentInterface,
    configuration: AgentConfiguration
) -> bool:
    """Register agent with global registry."""
    return await get_registry().register_agent(agent, configuration)

async def unregister_agent(agent_id: str) -> bool:
    """Unregister agent from global registry."""
    return await get_registry().unregister_agent(agent_id)

def get_agent(agent_id: str) -> Optional[UniversalAgentInterface]:
    """Get agent from global registry."""
    return get_registry().get_agent(agent_id)

def list_available_agents(
    agent_type: Optional[AgentType] = None,
    capability: Optional[CapabilityType] = None
) -> List[str]:
    """List available agents in global registry."""
    return get_registry().list_agents(
        agent_type=agent_type,
        health_state=HealthState.HEALTHY,
        capability=capability
    )

async def find_best_agent_for_task(task: AgentTask) -> Optional[str]:
    """Find best agent for task using global registry."""
    return await get_registry().find_best_agent(task)

def get_system_status() -> SystemStatus:
    """Get system status from global registry."""
    return get_registry().get_system_status()