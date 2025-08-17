"""
Backward Compatibility Layer for security_audit.py

This module provides the exact same interface as the original security_audit.py
but routes all calls to the new unified SecurityManager.

Usage:
    Replace "from .security_audit import" with "from .security_audit_compat import"
    or temporarily rename this file to security_audit.py during transition.
"""

from ._compatibility_adapters import get_adapter

# Get the adapter instance
_adapter = get_adapter('security_audit')

# Expose all original functions with the same signatures
async def log_security_event(event):
    """Log security event using new unified manager."""
    return await _adapter.log_security_event(event)

async def audit_access(user_id, resource, action):
    """Audit access using new unified manager."""
    return await _adapter.audit_access(user_id, resource, action)

# Maintain any classes that were exported from original module
class SecurityAuditor:
    """Legacy SecurityAuditor class for backward compatibility."""
    
    def __init__(self, *args, **kwargs):
        self._adapter = get_adapter('security_audit')
    
    async def log_event(self, event):
        return await self._adapter.log_security_event(event)
    
    async def audit(self, user_id, resource, action):
        return await self._adapter.audit_access(user_id, resource, action)

class AuditLogger:
    """Legacy AuditLogger class for backward compatibility."""
    
    def __init__(self, *args, **kwargs):
        self._adapter = get_adapter('security_audit')
    
    async def log(self, event):
        return await self._adapter.log_security_event(event)
    
    async def audit_user_action(self, user_id, resource, action):
        return await self._adapter.audit_access(user_id, resource, action)

# Export everything that was originally exported
__all__ = [
    'log_security_event',
    'audit_access',
    'SecurityAuditor',
    'AuditLogger'
]