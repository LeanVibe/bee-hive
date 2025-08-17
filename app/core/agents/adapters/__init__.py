"""
CLI Agent Adapters for Multi-CLI Coordination

This package contains adapter implementations for different CLI tools
that enable them to participate in coordinated multi-agent workflows.

Supported CLI Tools:
- Claude Code: Advanced AI coding assistant
- Cursor: AI-powered code editor
- Gemini CLI: Google's AI CLI tool
- OpenCode: Open-source AI coding tool
- GitHub Copilot: GitHub's AI coding assistant

Each adapter implements the UniversalAgentInterface to provide standardized
communication and task execution capabilities.

Usage:
    from app.core.agents.adapters import ClaudeCodeAdapter, CursorAdapter
    
    # Create and register adapters
    claude_adapter = ClaudeCodeAdapter("claude-1", AgentType.CLAUDE_CODE)
    cursor_adapter = CursorAdapter("cursor-1", AgentType.CURSOR)
    
    await register_agent(claude_adapter, claude_config)
    await register_agent(cursor_adapter, cursor_config)
"""

from .claude_code_adapter import ClaudeCodeAdapter
from .cursor_adapter import CursorAdapter
from .gemini_cli_adapter import GeminiCLIAdapter
from .opencode_adapter import OpenCodeAdapter
from .github_copilot_adapter import GitHubCopilotAdapter

__all__ = [
    'ClaudeCodeAdapter',
    'CursorAdapter', 
    'GeminiCLIAdapter',
    'OpenCodeAdapter',
    'GitHubCopilotAdapter'
]