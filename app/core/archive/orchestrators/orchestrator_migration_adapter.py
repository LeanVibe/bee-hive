"""
Orchestrator Migration Adapter for LeanVibe Agent Hive 2.0
Provides backward compatibility during orchestrator consolidation

This adapter provides a simple compatibility layer to ensure existing code
continues to work while the full migration to unified orchestrator is completed.
"""

# Import from the simple adapter for backward compatibility
from .simple_orchestrator_adapter import (
    AgentOrchestrator,
    AgentInstance,
    AgentRole,
    get_agent_orchestrator,
    initialize_orchestrator,
    shutdown_orchestrator
)

# For backward compatibility, export the main classes
__all__ = [
    'AgentOrchestrator',
    'AgentInstance', 
    'AgentRole',
    'get_agent_orchestrator',
    'initialize_orchestrator',
    'shutdown_orchestrator'
]