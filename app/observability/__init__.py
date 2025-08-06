"""
Observability module for monitoring and metrics.

Comprehensive observability infrastructure including:
- Hook system for Claude Code integration
- Event tracking and analysis
- Performance monitoring
- Session lifecycle management
"""

from .hooks.hooks_integration import get_hook_integration_manager, HookIntegrationManager

__all__ = [
    "get_hook_integration_manager",
    "HookIntegrationManager"
]