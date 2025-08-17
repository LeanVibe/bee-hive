"""
Backward Compatibility Layer for performance_optimizer.py

This module provides the exact same interface as the original performance_optimizer.py
but routes all calls to the new unified ResourceManager.

Usage:
    Replace "from .performance_optimizer import" with "from .performance_optimizer_compat import"
    or temporarily rename this file to performance_optimizer.py during transition.
"""

from ._compatibility_adapters import get_adapter

# Get the adapter instance
_adapter = get_adapter('performance_optimizer')

# Expose all original functions with the same signatures
async def optimize_performance(target, **kwargs):
    """Optimize performance using new unified manager."""
    return await _adapter.optimize_performance(target, **kwargs)

async def get_performance_metrics():
    """Get performance metrics using new unified manager."""
    return await _adapter.get_performance_metrics()

# Maintain any classes that were exported from original module
class PerformanceOptimizer:
    """Legacy PerformanceOptimizer class for backward compatibility."""
    
    def __init__(self, *args, **kwargs):
        self._adapter = get_adapter('performance_optimizer')
    
    async def optimize(self, target, **kwargs):
        return await self._adapter.optimize_performance(target, **kwargs)
    
    async def get_metrics(self):
        return await self._adapter.get_performance_metrics()

class PerformanceMonitor:
    """Legacy PerformanceMonitor class for backward compatibility."""
    
    def __init__(self, *args, **kwargs):
        self._adapter = get_adapter('performance_optimizer')
    
    async def monitor_performance(self, target, **kwargs):
        return await self._adapter.optimize_performance(target, **kwargs)
    
    async def collect_metrics(self):
        return await self._adapter.get_performance_metrics()

# Export everything that was originally exported
__all__ = [
    'optimize_performance',
    'get_performance_metrics',
    'PerformanceOptimizer',
    'PerformanceMonitor'
]