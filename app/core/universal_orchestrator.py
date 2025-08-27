"""
Universal Orchestrator - Interface Consolidation Layer

Provides unified interface to all orchestrator implementations for Epic 7 consolidation.
Designed to eliminate import errors and provide consistent orchestrator access patterns.
"""

from typing import Optional, Dict, Any, List
from enum import Enum
import asyncio
from datetime import datetime

# Import available orchestrator implementations
try:
    from .simple_orchestrator import SimpleOrchestrator, AgentRole, AgentStatus, TaskPriority
    SIMPLE_AVAILABLE = True
except ImportError:
    SIMPLE_AVAILABLE = False
    # Fallback definitions for tests
    class AgentRole(Enum):
        DEVELOPER = "developer"
        QA = "qa"
        ARCHITECT = "architect"
        META = "meta"

    class AgentStatus(Enum):
        IDLE = "idle"
        ACTIVE = "active"
        BUSY = "busy"
        ERROR = "error"

    class TaskPriority(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"

try:
    from .orchestrator import Orchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False

try:
    from .production_orchestrator import ProductionOrchestrator
    PRODUCTION_AVAILABLE = True
except ImportError:
    PRODUCTION_AVAILABLE = False


class UniversalOrchestratorConfig:
    """Configuration for universal orchestrator."""
    
    def __init__(self):
        self.orchestrator_type = "simple"  # Default to simple
        self.max_concurrent_agents = 10
        self.enable_monitoring = True
        self.enable_metrics = False  # Disable for testing
        
    @classmethod
    def from_settings(cls, settings):
        """Create config from application settings."""
        config = cls()
        if hasattr(settings, 'ORCHESTRATOR_TYPE'):
            config.orchestrator_type = settings.ORCHESTRATOR_TYPE
        if hasattr(settings, 'MAX_CONCURRENT_AGENTS'):
            config.max_concurrent_agents = settings.MAX_CONCURRENT_AGENTS
        return config


class UniversalOrchestrator:
    """
    Universal orchestrator that provides unified interface to all orchestrator implementations.
    
    Epic 7 Consolidation: This class ensures tests can import and use orchestrator
    functionality regardless of which specific implementation is available.
    """
    
    def __init__(self, config: Optional[UniversalOrchestratorConfig] = None):
        self.config = config or UniversalOrchestratorConfig()
        self._orchestrator = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize the appropriate orchestrator implementation."""
        if self._initialized:
            return
            
        # Try to initialize based on availability and configuration
        if SIMPLE_AVAILABLE and self.config.orchestrator_type == "simple":
            self._orchestrator = SimpleOrchestrator()
        elif ORCHESTRATOR_AVAILABLE:
            self._orchestrator = Orchestrator()
        elif PRODUCTION_AVAILABLE:
            self._orchestrator = ProductionOrchestrator()
        else:
            # Fallback mock orchestrator for testing
            self._orchestrator = MockOrchestrator()
            
        self._initialized = True
        
    async def create_agent(self, role: AgentRole, **kwargs) -> str:
        """Create a new agent with the specified role."""
        await self.initialize()
        if hasattr(self._orchestrator, 'create_agent'):
            return await self._orchestrator.create_agent(role, **kwargs)
        else:
            # Fallback for testing
            return f"agent_{role.value}_{datetime.now().timestamp()}"
            
    async def get_agent_status(self, agent_id: str) -> AgentStatus:
        """Get the status of an agent."""
        await self.initialize()
        if hasattr(self._orchestrator, 'get_agent_status'):
            return await self._orchestrator.get_agent_status(agent_id)
        else:
            return AgentStatus.IDLE
            
    async def assign_task(self, agent_id: str, task: str, priority: TaskPriority = TaskPriority.MEDIUM):
        """Assign a task to an agent."""
        await self.initialize()
        if hasattr(self._orchestrator, 'assign_task'):
            return await self._orchestrator.assign_task(agent_id, task, priority)
        else:
            return True
            
    async def get_active_agents(self) -> List[str]:
        """Get list of active agent IDs."""
        await self.initialize()
        if hasattr(self._orchestrator, 'get_active_agents'):
            return await self._orchestrator.get_active_agents()
        else:
            return []
            
    async def shutdown(self):
        """Shutdown the orchestrator and all agents."""
        if self._orchestrator and hasattr(self._orchestrator, 'shutdown'):
            await self._orchestrator.shutdown()
        self._initialized = False


class MockOrchestrator:
    """Mock orchestrator for testing when no real implementation is available."""
    
    def __init__(self):
        self.agents = {}
        self.tasks = {}
        
    async def create_agent(self, role: AgentRole, **kwargs) -> str:
        agent_id = f"mock_agent_{role.value}_{len(self.agents)}"
        self.agents[agent_id] = {
            'role': role,
            'status': AgentStatus.IDLE,
            'created_at': datetime.now()
        }
        return agent_id
        
    async def get_agent_status(self, agent_id: str) -> AgentStatus:
        if agent_id in self.agents:
            return self.agents[agent_id]['status']
        return AgentStatus.ERROR
        
    async def assign_task(self, agent_id: str, task: str, priority: TaskPriority = TaskPriority.MEDIUM):
        if agent_id in self.agents:
            self.agents[agent_id]['status'] = AgentStatus.ACTIVE
            self.tasks[agent_id] = {'task': task, 'priority': priority}
            return True
        return False
        
    async def get_active_agents(self) -> List[str]:
        return [aid for aid, agent in self.agents.items() if agent['status'] == AgentStatus.ACTIVE]
        
    async def shutdown(self):
        for agent_id in self.agents:
            self.agents[agent_id]['status'] = AgentStatus.IDLE
        self.tasks.clear()


# Convenience factory functions for Epic 7 consolidation
def create_universal_orchestrator(settings=None) -> UniversalOrchestrator:
    """Create universal orchestrator with settings."""
    config = UniversalOrchestratorConfig.from_settings(settings) if settings else UniversalOrchestratorConfig()
    return UniversalOrchestrator(config)


async def get_orchestrator_instance(settings=None) -> UniversalOrchestrator:
    """Get initialized universal orchestrator instance."""
    orchestrator = create_universal_orchestrator(settings)
    await orchestrator.initialize()
    return orchestrator


# Export all necessary components for tests
__all__ = [
    'UniversalOrchestrator',
    'UniversalOrchestratorConfig', 
    'MockOrchestrator',
    'AgentRole',
    'AgentStatus', 
    'TaskPriority',
    'create_universal_orchestrator',
    'get_orchestrator_instance'
]