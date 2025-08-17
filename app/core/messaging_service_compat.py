"""
Backward Compatibility Layer for messaging_service.py

This module provides the exact same interface as the original messaging_service.py
but routes all calls to the new unified CommunicationManager.

Usage:
    Replace "from .messaging_service import" with "from .messaging_service_compat import"
    or temporarily rename this file to messaging_service.py during transition.
"""

from ._compatibility_adapters import get_adapter

# Get the adapter instance
_adapter = get_adapter('messaging_service')

# Expose all original functions with the same signatures
async def send_message(message, **kwargs):
    """Send message using new unified manager."""
    return await _adapter.send_message(message, **kwargs)

async def receive_message(agent_id, **kwargs):
    """Receive message using new unified manager."""
    return await _adapter.receive_message(agent_id, **kwargs)

async def broadcast_message(message, **kwargs):
    """Broadcast message using new unified manager."""
    return await _adapter.broadcast_message(message, **kwargs)

# Maintain any classes that were exported from original module
class MessagingService:
    """Legacy MessagingService class for backward compatibility."""
    
    def __init__(self, *args, **kwargs):
        self._adapter = get_adapter('messaging_service')
    
    async def send(self, message, **kwargs):
        return await self._adapter.send_message(message, **kwargs)
    
    async def receive(self, agent_id, **kwargs):
        return await self._adapter.receive_message(agent_id, **kwargs)
    
    async def broadcast(self, message, **kwargs):
        return await self._adapter.broadcast_message(message, **kwargs)

# Export everything that was originally exported
__all__ = [
    'send_message',
    'receive_message',
    'broadcast_message', 
    'MessagingService'
]