"""
Consolidated Orchestrator for LeanVibe Agent Hive 2.0

ORCHESTRATOR CONSOLIDATION: This module now serves as the single production
orchestrator by directly aliasing SimpleOrchestrator, which already contains
all production features including Redis, tmux, plugins, and monitoring.

All imports should use this module for consistency, but it directly uses
SimpleOrchestrator as the implementation.
"""

# Direct import and alias of SimpleOrchestrator as the production orchestrator
from .simple_orchestrator import (
    SimpleOrchestrator as Orchestrator,  # Main orchestrator class
    SimpleOrchestrator,  # For explicit imports
    create_simple_orchestrator,
    create_enhanced_simple_orchestrator,
    get_simple_orchestrator,
    set_simple_orchestrator,
    AgentRole,
    AgentStatus, 
    TaskPriority,
    SimpleOrchestratorError,
    AgentNotFoundError,
    TaskDelegationError,
    AgentInstance,
    AgentLauncherType,
    TaskAssignment
)

# Additional compatibility imports for legacy code
from .simple_orchestrator import SimpleOrchestrator as AgentOrchestrator  # Legacy alias
from dataclasses import dataclass
from typing import List

@dataclass  
class AgentCapability:
    """Agent capability definition for compatibility with agent_spawner."""
    name: str
    description: str
    confidence_level: float = 0.8
    specialization_areas: List[str] = None

    def __post_init__(self):
        if self.specialization_areas is None:
            self.specialization_areas = []

# Main orchestrator factory function
def get_orchestrator():
    """Get the production orchestrator instance."""
    return get_simple_orchestrator()

# Configuration compatibility
@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator behavior."""
    max_agents: int = 50
    task_timeout: int = 300
    plugin_dir: str = "app/core/orchestrator_plugins"
    enable_plugins: bool = True
    use_simple_orchestrator: bool = True
    enable_advanced_features: bool = True

# Ensure we're using the singleton pattern correctly
orchestrator = get_simple_orchestrator()