"""
Backward Compatibility Layer for workflow_engine.py

This module provides the exact same interface as the original workflow_engine.py
but routes all calls to the new unified WorkflowManager.

Usage:
    Replace "from .workflow_engine import" with "from .workflow_engine_compat import"
    or temporarily rename this file to workflow_engine.py during transition.
"""

from ._compatibility_adapters import get_adapter

# Get the adapter instance
_adapter = get_adapter('workflow_engine')

# Expose all original functions with the same signatures
async def execute_workflow(workflow_def, **kwargs):
    """Execute workflow using new unified manager."""
    return await _adapter.execute_workflow(workflow_def, **kwargs)

async def pause_workflow(workflow_id):
    """Pause workflow using new unified manager."""
    return await _adapter.pause_workflow(workflow_id)

async def resume_workflow(workflow_id):
    """Resume workflow using new unified manager."""
    return await _adapter.resume_workflow(workflow_id)

# Maintain any classes that were exported from original module
class WorkflowEngine:
    """Legacy WorkflowEngine class for backward compatibility."""
    
    def __init__(self, *args, **kwargs):
        self._adapter = get_adapter('workflow_engine')
    
    async def execute(self, workflow_def, **kwargs):
        return await self._adapter.execute_workflow(workflow_def, **kwargs)
    
    async def pause(self, workflow_id):
        return await self._adapter.pause_workflow(workflow_id)
    
    async def resume(self, workflow_id):
        return await self._adapter.resume_workflow(workflow_id)

class WorkflowExecutor:
    """Legacy WorkflowExecutor class for backward compatibility."""
    
    def __init__(self, *args, **kwargs):
        self._adapter = get_adapter('workflow_engine')
    
    async def run_workflow(self, workflow_def, **kwargs):
        return await self._adapter.execute_workflow(workflow_def, **kwargs)

# Export everything that was originally exported
__all__ = [
    'execute_workflow',
    'pause_workflow',
    'resume_workflow',
    'WorkflowEngine',
    'WorkflowExecutor'
]