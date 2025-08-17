"""
Backward Compatibility Layer for agent_spawner.py

This module provides the exact same interface as the original agent_spawner.py
but routes all calls to the new unified AgentManager.

Usage:
    Replace "from .agent_spawner import" with "from .agent_spawner_compat import"
    or temporarily rename this file to agent_spawner.py during transition.
"""

from ._compatibility_adapters import get_adapter

# Get the adapter instance
_adapter = get_adapter('agent_spawner')

# Expose all original functions with the same signatures
async def spawn_agent(agent_spec, **kwargs):
    """Spawn agent using new unified manager."""
    return await _adapter.spawn_agent(agent_spec, **kwargs)

async def stop_agent(agent_id, **kwargs):
    """Stop agent using new unified manager.""" 
    return await _adapter.stop_agent(agent_id, **kwargs)

async def get_agent_status(agent_id):
    """Get agent status using new unified manager."""
    return await _adapter.get_agent_status(agent_id)

# Maintain any classes that were exported from original module
class AgentSpawner:
    """Legacy AgentSpawner class for backward compatibility."""
    
    def __init__(self, *args, **kwargs):
        self._adapter = get_adapter('agent_spawner')
    
    async def spawn(self, agent_spec, **kwargs):
        return await self._adapter.spawn_agent(agent_spec, **kwargs)
    
    async def stop(self, agent_id, **kwargs):
        return await self._adapter.stop_agent(agent_id, **kwargs)
    
    async def status(self, agent_id):
        return await self._adapter.get_agent_status(agent_id)

# Export everything that was originally exported
__all__ = [
    'spawn_agent',
    'stop_agent', 
    'get_agent_status',
    'AgentSpawner'
]