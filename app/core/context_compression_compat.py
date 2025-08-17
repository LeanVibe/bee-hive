"""
Backward Compatibility Layer for context_compression.py

This module provides the exact same interface as the original context_compression.py
but routes all calls to the new unified ContextManagerUnified.

Usage:
    Replace "from .context_compression import" with "from .context_compression_compat import"
    or temporarily rename this file to context_compression.py during transition.
"""

from ._compatibility_adapters import get_adapter

# Get the adapter instance
_adapter = get_adapter('context_compression')

# Expose all original functions with the same signatures
async def compress_context(context_data, **kwargs):
    """Compress context using new unified manager."""
    return await _adapter.compress_context(context_data, **kwargs)

async def decompress_context(compressed_data, **kwargs):
    """Decompress context using new unified manager."""
    return await _adapter.decompress_context(compressed_data, **kwargs)

# Maintain any classes that were exported from original module
class ContextCompressor:
    """Legacy ContextCompressor class for backward compatibility."""
    
    def __init__(self, *args, **kwargs):
        self._adapter = get_adapter('context_compression')
    
    async def compress(self, context_data, **kwargs):
        return await self._adapter.compress_context(context_data, **kwargs)
    
    async def decompress(self, compressed_data, **kwargs):
        return await self._adapter.decompress_context(compressed_data, **kwargs)

class ContextCompressionEngine:
    """Legacy ContextCompressionEngine class for backward compatibility."""
    
    def __init__(self, *args, **kwargs):
        self._adapter = get_adapter('context_compression')
    
    async def compress_data(self, data, **kwargs):
        return await self._adapter.compress_context(data, **kwargs)
    
    async def decompress_data(self, data, **kwargs):
        return await self._adapter.decompress_context(data, **kwargs)

# Export everything that was originally exported
__all__ = [
    'compress_context',
    'decompress_context',
    'ContextCompressor',
    'ContextCompressionEngine'
]