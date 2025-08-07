"""
Observability Hooks Module for LeanVibe Agent Hive 2.0

Claude Code integration hooks for comprehensive system observability including
tool execution monitoring, session lifecycle tracking, and performance optimization.
"""

from .hooks_config import HookConfig, get_hook_config, reload_hook_config
from .hooks_integration import EventProcessor, get_hook_integration_manager, HookInterceptor, set_hook_integration_manager
# Note: HookIntegrationManager imports moved to avoid circular imports

# Import new classes from parent hooks.py module
try:
    import importlib.util
    import os
    hooks_py_path = os.path.join(os.path.dirname(__file__), '..', 'hooks.py')
    spec = importlib.util.spec_from_file_location("hooks_module", hooks_py_path)
    hooks_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hooks_module)
    RealTimeEventProcessor = hooks_module.RealTimeEventProcessor
    EventCapture = hooks_module.EventCapture
    # Override the stub HookInterceptor with the full implementation
    HookInterceptor = hooks_module.HookInterceptor
except Exception as e:
    # Fallback if import fails
    RealTimeEventProcessor = None
    EventCapture = None
    # Keep the stub HookInterceptor from hooks_integration if import fails

__all__ = [
    "HookConfig",
    "get_hook_config", 
    "reload_hook_config",
    "EventProcessor",
    "get_hook_integration_manager",
    "HookInterceptor",
    "set_hook_integration_manager",
    "RealTimeEventProcessor",
    "EventCapture"
]