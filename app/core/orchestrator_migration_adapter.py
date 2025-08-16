"""
Orchestrator Migration Adapter for LeanVibe Agent Hive
Provides backward compatibility while migrating to unified production orchestrator

Epic 1, Phase 2 Week 3: Orchestrator Consolidation Migration Strategy
This adapter provides seamless migration from legacy orchestrator implementations
to the unified production orchestrator while maintaining full API compatibility.

Migration Strategy:
1. Wrap unified orchestrator with legacy API compatibility
2. Gradually migrate consumers to new unified API
3. Deprecate and remove legacy orchestrator implementations
4. Maintain performance and functionality during transition

Compatibility Layer:
- AgentOrchestrator (orchestrator.py) -> UnifiedProductionOrchestrator
- ProductionOrchestrator (production_orchestrator.py) -> UnifiedProductionOrchestrator
- AutomatedOrchestrator (automated_orchestrator.py) -> UnifiedProductionOrchestrator
- PerformanceOrchestrator (performance_orchestrator.py) -> UnifiedProductionOrchestrator
- HighConcurrencyOrchestrator (high_concurrency_orchestrator.py) -> UnifiedProductionOrchestrator
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Callable, Union
from datetime import datetime, timedelta
from dataclasses import asdict
from enum import Enum

from app.core.production_orchestrator_unified import (
    UnifiedProductionOrchestrator,
    get_production_orchestrator,
    AgentState as UnifiedAgentState,
    OrchestrationStrategy,
    TaskPriority as UnifiedTaskPriority,
    AgentCapability,
    RegisteredAgent,
    OrchestrationTask,
    submit_orchestration_task,
    register_orchestration_agent
)
from app.core.logging_service import get_component_logger

logger = get_component_logger("orchestrator_migration")


# Legacy compatibility enums
class AgentStatus(Enum):
    """Legacy agent status compatibility"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    SLEEPING = "sleeping"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"


class AgentRole(Enum):
    """Legacy agent role compatibility"""
    STRATEGIC_PARTNER = "strategic_partner"
    PRODUCT_MANAGER = "product_manager" 
    ARCHITECT = "architect"
    BACKEND_DEVELOPER = "backend_developer"
    FRONTEND_DEVELOPER = "frontend_developer"
    DEVOPS_ENGINEER = "devops_engineer"
    QA_ENGINEER = "qa_engineer"
    META_AGENT = "meta_agent"


class TaskStatus(Enum):
    """Legacy task status compatibility"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Legacy task priority compatibility"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


# Legacy compatibility data classes
class AgentInstance:
    """Legacy AgentInstance compatibility wrapper"""
    def __init__(self, id: str, role: AgentRole, status: AgentStatus,
                 tmux_session: Optional[str] = None,
                 capabilities: Optional[List[Any]] = None,
                 current_task: Optional[str] = None,
                 context_window_usage: float = 0.0,
                 last_heartbeat: Optional[datetime] = None,
                 anthropic_client: Optional[Any] = None):
        self.id = id
        self.role = role
        self.status = status
        self.tmux_session = tmux_session
        self.capabilities = capabilities or []
        self.current_task = current_task
        self.context_window_usage = context_window_usage
        self.last_heartbeat = last_heartbeat or datetime.utcnow()
        self.anthropic_client = anthropic_client
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility"""
        return {
            'id': self.id,
            'role': self.role.value,
            'status': self.status.value,
            'tmux_session': self.tmux_session,
            'capabilities': [asdict(cap) if hasattr(cap, '__dict__') else cap for cap in self.capabilities],
            'current_task': self.current_task,
            'context_window_usage': self.context_window_usage,
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'anthropic_client': None  # Don't serialize the client
        }


class AgentOrchestrator:
    """
    Legacy AgentOrchestrator compatibility wrapper
    
    Provides backward compatibility for existing code while delegating
    to the unified production orchestrator under the hood.
    """
    
    def __init__(self):
        self._unified_orchestrator = get_production_orchestrator()
        self._legacy_agents: Dict[str, AgentInstance] = {}
        self._status_mapping = self._create_status_mapping()
        self._priority_mapping = self._create_priority_mapping()
        
        logger.info("Legacy AgentOrchestrator initialized with unified backend")
    
    def _create_status_mapping(self) -> Dict[str, str]:
        """Create mapping between legacy and unified agent states"""
        return {
            AgentStatus.INITIALIZING.value: UnifiedAgentState.INITIALIZING.value,
            AgentStatus.ACTIVE.value: UnifiedAgentState.READY.value,
            AgentStatus.BUSY.value: UnifiedAgentState.BUSY.value,
            AgentStatus.IDLE.value: UnifiedAgentState.IDLE.value,
            AgentStatus.SLEEPING.value: UnifiedAgentState.SLEEPING.value,
            AgentStatus.ERROR.value: UnifiedAgentState.ERROR.value,
            AgentStatus.SHUTTING_DOWN.value: UnifiedAgentState.TERMINATING.value,
            AgentStatus.TERMINATED.value: UnifiedAgentState.TERMINATED.value
        }
    
    def _create_priority_mapping(self) -> Dict[str, int]:
        """Create mapping between legacy and unified task priorities"""
        return {
            TaskPriority.LOW.value: UnifiedTaskPriority.LOW,
            TaskPriority.NORMAL.value: UnifiedTaskPriority.NORMAL,
            TaskPriority.HIGH.value: UnifiedTaskPriority.HIGH,
            TaskPriority.CRITICAL.value: UnifiedTaskPriority.CRITICAL,
            TaskPriority.EMERGENCY.value: UnifiedTaskPriority.EMERGENCY
        }
    
    async def start(self):
        """Start the orchestrator"""
        await self._unified_orchestrator.start_orchestrator()
        logger.info("Legacy AgentOrchestrator started")
    
    async def stop(self):
        """Stop the orchestrator"""
        await self._unified_orchestrator.stop_orchestrator()
        logger.info("Legacy AgentOrchestrator stopped")
    
    async def spawn_agent(self, role: AgentRole, capabilities: Optional[List[Any]] = None,
                         tmux_session: Optional[str] = None,
                         anthropic_client: Optional[Any] = None) -> str:
        """
        Spawn new agent (legacy compatibility)
        
        Maps to unified orchestrator agent registration
        """
        agent_id = f"agent_{role.value}_{int(datetime.utcnow().timestamp())}"
        
        # Convert legacy capabilities to unified format
        unified_capabilities = []
        if capabilities:
            for cap in capabilities:
                if hasattr(cap, 'name'):
                    # AgentCapability object
                    unified_cap = AgentCapability(
                        capability_type=cap.name,
                        skill_level=getattr(cap, 'confidence_level', 0.5) * 10,  # Convert 0-1 to 1-10
                        max_concurrent_tasks=1,
                        specialization_areas=getattr(cap, 'specialization_areas', [])
                    )
                else:
                    # String capability
                    unified_cap = AgentCapability(
                        capability_type=str(cap),
                        skill_level=5,  # Default skill level
                        max_concurrent_tasks=1
                    )
                unified_capabilities.append(unified_cap)
        else:
            # Default capabilities based on role
            unified_capabilities = [
                AgentCapability(
                    capability_type=role.value,
                    skill_level=7,
                    max_concurrent_tasks=2
                )
            ]
        
        # Register with unified orchestrator
        success = await self._unified_orchestrator.register_agent(
            agent_id=agent_id,
            agent_type=role.value,
            capabilities=unified_capabilities,
            anthropic_client=anthropic_client,
            tmux_session=tmux_session
        )
        
        if success:
            # Create legacy agent instance for compatibility
            legacy_agent = AgentInstance(
                id=agent_id,
                role=role,
                status=AgentStatus.ACTIVE,
                tmux_session=tmux_session,
                capabilities=capabilities or [],
                anthropic_client=anthropic_client
            )
            self._legacy_agents[agent_id] = legacy_agent
            
            logger.info("Agent spawned successfully", agent_id=agent_id, role=role.value)
            return agent_id
        else:
            logger.error("Failed to spawn agent", role=role.value)
            raise RuntimeError(f"Failed to spawn agent with role {role.value}")
    
    async def terminate_agent(self, agent_id: str):
        """Terminate agent (legacy compatibility)"""
        success = await self._unified_orchestrator.deregister_agent(agent_id)
        
        if success and agent_id in self._legacy_agents:
            del self._legacy_agents[agent_id]
            logger.info("Agent terminated successfully", agent_id=agent_id)
        else:
            logger.warning("Failed to terminate agent or agent not found", agent_id=agent_id)
    
    async def delegate_task(self, task_data: Dict[str, Any], 
                          agent_id: Optional[str] = None,
                          priority: Union[str, TaskPriority] = TaskPriority.NORMAL,
                          deadline: Optional[datetime] = None) -> str:
        """
        Delegate task to agent (legacy compatibility)
        
        Maps to unified orchestrator task submission
        """
        # Convert legacy priority to unified
        if isinstance(priority, str):
            unified_priority = self._priority_mapping.get(priority, UnifiedTaskPriority.NORMAL)
        elif isinstance(priority, TaskPriority):
            unified_priority = self._priority_mapping.get(priority.value, UnifiedTaskPriority.NORMAL)
        else:
            unified_priority = UnifiedTaskPriority.NORMAL
        
        # Determine required capabilities from task or agent
        required_capabilities = []
        if agent_id and agent_id in self._legacy_agents:
            # Use specific agent's capabilities
            agent = self._legacy_agents[agent_id]
            if hasattr(agent.role, 'value'):
                required_capabilities = [agent.role.value]
        else:
            # Infer capabilities from task type
            task_type = task_data.get('type', 'generic')
            required_capabilities = [task_type]
        
        # Create orchestration task
        task = OrchestrationTask(
            task_type=task_data.get('type', 'generic'),
            priority=unified_priority,
            required_capabilities=required_capabilities,
            payload=task_data,
            deadline=deadline
        )
        
        # Submit to unified orchestrator
        success = await self._unified_orchestrator.submit_task(task)
        
        if success:
            logger.info("Task delegated successfully", task_id=task.task_id, agent_id=agent_id)
            return task.task_id
        else:
            logger.error("Failed to delegate task", agent_id=agent_id)
            raise RuntimeError("Failed to delegate task")
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent status (legacy compatibility)"""
        # Try unified orchestrator first
        unified_status = self._unified_orchestrator.get_agent_status(agent_id)
        if unified_status:
            # Convert to legacy format
            legacy_status = {
                'id': unified_status['agent_id'],
                'type': unified_status['agent_type'],
                'state': unified_status['state'],
                'current_tasks': list(unified_status['current_tasks']),
                'last_heartbeat': unified_status['last_heartbeat'],
                'performance_metrics': unified_status['performance_metrics'],
                'total_tasks_completed': unified_status['total_tasks_completed'],
                'error_count': unified_status['error_count']
            }
            return legacy_status
        
        # Fallback to legacy agent if exists
        if agent_id in self._legacy_agents:
            return self._legacy_agents[agent_id].to_dict()
        
        return None
    
    def get_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get all agents status (legacy compatibility)"""
        all_agents = {}
        
        # Get from unified orchestrator
        unified_agents = self._unified_orchestrator.get_all_agents_status()
        for agent_id, agent_data in unified_agents.items():
            all_agents[agent_id] = {
                'id': agent_data['agent_id'],
                'type': agent_data['agent_type'],
                'state': agent_data['state'],
                'current_tasks': list(agent_data['current_tasks']),
                'last_heartbeat': agent_data['last_heartbeat'],
                'performance_metrics': agent_data['performance_metrics']
            }
        
        # Add any legacy-only agents (should be rare)
        for agent_id, agent in self._legacy_agents.items():
            if agent_id not in all_agents:
                all_agents[agent_id] = agent.to_dict()
        
        return all_agents
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get orchestration statistics (legacy compatibility)"""
        unified_metrics = self._unified_orchestrator.get_orchestration_metrics()
        
        # Convert to legacy format
        legacy_stats = {
            'total_agents': unified_metrics['total_agents'],
            'active_agents': unified_metrics['active_agents'],
            'pending_tasks': unified_metrics['pending_tasks'],
            'completed_tasks': unified_metrics['tasks_processed'],
            'failed_tasks': unified_metrics['tasks_failed'],
            'average_task_time': unified_metrics['average_task_time'],
            'system_load': unified_metrics['current_load'],
            'uptime_seconds': (datetime.utcnow() - datetime.fromisoformat(unified_metrics['timestamp'])).total_seconds() if unified_metrics.get('timestamp') else 0
        }
        
        return legacy_stats
    
    async def update_agent_heartbeat(self, agent_id: str, context_usage: Optional[float] = None):
        """Update agent heartbeat (legacy compatibility)"""
        await self._unified_orchestrator.update_agent_heartbeat(agent_id, context_usage)
        
        # Update legacy agent if exists
        if agent_id in self._legacy_agents:
            agent = self._legacy_agents[agent_id]
            agent.last_heartbeat = datetime.utcnow()
            if context_usage is not None:
                agent.context_window_usage = context_usage


class ProductionOrchestrator:
    """
    Legacy ProductionOrchestrator compatibility wrapper
    
    Provides backward compatibility for production-specific features
    while delegating to the unified production orchestrator.
    """
    
    def __init__(self):
        self._unified_orchestrator = get_production_orchestrator()
        logger.info("Legacy ProductionOrchestrator initialized with unified backend")
    
    async def start_monitoring(self):
        """Start production monitoring"""
        await self._unified_orchestrator.start_orchestrator()
        logger.info("Production monitoring started")
    
    async def stop_monitoring(self):
        """Stop production monitoring"""
        await self._unified_orchestrator.stop_orchestrator()
        logger.info("Production monitoring stopped")
    
    def get_production_metrics(self) -> Dict[str, Any]:
        """Get production metrics"""
        return self._unified_orchestrator.get_orchestration_metrics()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        metrics = self._unified_orchestrator.get_orchestration_metrics()
        
        health_status = {
            'status': 'healthy' if metrics['system_health_score'] > 0.8 else 'degraded',
            'health_score': metrics['system_health_score'],
            'cpu_usage': metrics['cpu_usage'],
            'memory_usage': metrics['memory_usage'],
            'active_agents': metrics['active_agents'],
            'pending_tasks': metrics['pending_tasks']
        }
        
        return health_status


class AutomatedOrchestrator:
    """
    Legacy AutomatedOrchestrator compatibility wrapper
    
    Provides backward compatibility for automation features
    while delegating to the unified production orchestrator.
    """
    
    def __init__(self):
        self._unified_orchestrator = get_production_orchestrator()
        logger.info("Legacy AutomatedOrchestrator initialized with unified backend")
    
    async def start_automation(self):
        """Start automation features"""
        await self._unified_orchestrator.start_orchestrator()
        self._unified_orchestrator.set_auto_scaling(True)
        logger.info("Automation features started")
    
    async def stop_automation(self):
        """Stop automation features"""
        self._unified_orchestrator.set_auto_scaling(False)
        await self._unified_orchestrator.stop_orchestrator()
        logger.info("Automation features stopped")
    
    def enable_auto_scaling(self):
        """Enable auto-scaling"""
        self._unified_orchestrator.set_auto_scaling(True)
    
    def disable_auto_scaling(self):
        """Disable auto-scaling"""
        self._unified_orchestrator.set_auto_scaling(False)


class PerformanceOrchestrator:
    """
    Legacy PerformanceOrchestrator compatibility wrapper
    
    Provides backward compatibility for performance features
    while delegating to the unified production orchestrator.
    """
    
    def __init__(self):
        self._unified_orchestrator = get_production_orchestrator()
        logger.info("Legacy PerformanceOrchestrator initialized with unified backend")
    
    async def start_performance_monitoring(self):
        """Start performance monitoring"""
        await self._unified_orchestrator.start_orchestrator()
        self._unified_orchestrator.set_orchestration_strategy(OrchestrationStrategy.PERFORMANCE_OPTIMIZED)
        logger.info("Performance monitoring started")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self._unified_orchestrator.get_orchestration_metrics()
    
    def optimize_performance(self):
        """Optimize performance settings"""
        self._unified_orchestrator.set_orchestration_strategy(OrchestrationStrategy.PERFORMANCE_OPTIMIZED)


class HighConcurrencyOrchestrator:
    """
    Legacy HighConcurrencyOrchestrator compatibility wrapper
    
    Provides backward compatibility for high-concurrency features
    while delegating to the unified production orchestrator.
    """
    
    def __init__(self):
        self._unified_orchestrator = get_production_orchestrator()
        logger.info("Legacy HighConcurrencyOrchestrator initialized with unified backend")
    
    async def start_high_concurrency_mode(self):
        """Start high concurrency mode"""
        await self._unified_orchestrator.start_orchestrator()
        self._unified_orchestrator.set_orchestration_strategy(OrchestrationStrategy.LOAD_BALANCED)
        self._unified_orchestrator.set_auto_scaling(True)
        logger.info("High concurrency mode started")
    
    def get_concurrency_metrics(self) -> Dict[str, Any]:
        """Get concurrency metrics"""
        return self._unified_orchestrator.get_orchestration_metrics()


# Migration utility functions
async def migrate_orchestrator_instance(legacy_orchestrator: Any) -> UnifiedProductionOrchestrator:
    """
    Migrate a legacy orchestrator instance to unified orchestrator
    
    Args:
        legacy_orchestrator: Legacy orchestrator instance
        
    Returns:
        UnifiedProductionOrchestrator: Unified orchestrator instance
    """
    unified_orchestrator = get_production_orchestrator()
    
    # Start unified orchestrator if not already running
    await unified_orchestrator.start_orchestrator()
    
    logger.info("Legacy orchestrator migrated to unified production orchestrator")
    
    return unified_orchestrator


def get_legacy_compatible_orchestrator(orchestrator_type: str = "agent") -> Any:
    """
    Get legacy-compatible orchestrator instance
    
    Args:
        orchestrator_type: Type of orchestrator needed
        
    Returns:
        Legacy-compatible orchestrator wrapper
    """
    orchestrator_map = {
        "agent": AgentOrchestrator,
        "production": ProductionOrchestrator,
        "automated": AutomatedOrchestrator,
        "performance": PerformanceOrchestrator,
        "high_concurrency": HighConcurrencyOrchestrator
    }
    
    orchestrator_class = orchestrator_map.get(orchestrator_type, AgentOrchestrator)
    return orchestrator_class()


# Export legacy compatibility classes and functions
__all__ = [
    'AgentOrchestrator',
    'ProductionOrchestrator', 
    'AutomatedOrchestrator',
    'PerformanceOrchestrator',
    'HighConcurrencyOrchestrator',
    'AgentStatus',
    'AgentRole',
    'TaskStatus',
    'TaskPriority',
    'AgentInstance',
    'migrate_orchestrator_instance',
    'get_legacy_compatible_orchestrator'
]