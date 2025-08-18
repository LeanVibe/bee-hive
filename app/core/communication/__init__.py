"""
Multi-CLI Communication Protocol for Heterogeneous Agent Coordination

This module provides the communication infrastructure for coordinating multiple
heterogeneous CLI agents (Claude Code, Cursor, Gemini CLI, etc.) with different
message formats, protocols, and capabilities.

Architecture Components:
- MultiCLIProtocol: Core protocol for message translation and routing
- MessageTranslator: Converts between CLI-specific message formats
- ContextPreserver: Maintains context during agent handoffs
- CommunicationBridge: Bridges different communication channels
- ProtocolAdapter: Adapts various CLI protocols to universal format

Key Features:
- Universal message format with CLI-specific translations
- Context preservation during agent handoffs
- Real-time bidirectional communication
- Protocol bridging for different CLI tools
- Message queuing and reliable delivery
- Performance optimization and caching
"""

from .multi_cli_protocol import MultiCLIProtocol
from .message_translator import MessageTranslator
from .context_preserver import ContextPreserver
from .communication_bridge import CommunicationBridge
from .protocol_models import (
    UniversalMessage,
    CLIMessage,
    ContextPackage,
    HandoffRequest,
    ProtocolConfig,
    MessageRoute
)

__all__ = [
    "MultiCLIProtocol",
    "MessageTranslator",
    "ContextPreserver", 
    "CommunicationBridge",
    "UniversalMessage",
    "CLIMessage",
    "ContextPackage",
    "HandoffRequest",
    "ProtocolConfig",
    "MessageRoute"
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "LeanVibe Agent Hive"
__description__ = "Multi-CLI Communication Protocol for Heterogeneous Agent Coordination"