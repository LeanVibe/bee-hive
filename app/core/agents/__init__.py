"""
Universal Agent System for Multi-CLI Coordination

This package provides the foundation for heterogeneous CLI agent coordination,
enabling Claude Code, Cursor, Gemini CLI, OpenCode, and other CLI tools to work
together in coordinated workflows with proper isolation and security.
"""

from .universal_agent_interface import (
    UniversalAgentInterface,
    AgentTask,
    AgentResult,
    AgentCapability,
    ExecutionContext,
    HealthStatus
)

from .agent_registry import (
    AgentRegistry,
    register_agent,
    get_agent,
    list_available_agents
)

from .models import (
    AgentMessage,
    MessageType,
    MessageMetadata,
    TaskStatus,
    AgentType
)

__all__ = [
    # Core interfaces
    'UniversalAgentInterface',
    'AgentTask',
    'AgentResult', 
    'AgentCapability',
    'ExecutionContext',
    'HealthStatus',
    
    # Registry functions
    'AgentRegistry',
    'register_agent',
    'get_agent',
    'list_available_agents',
    
    # Data models
    'AgentMessage',
    'MessageType',
    'MessageMetadata',
    'TaskStatus',
    'AgentType'
]