"""
Git Worktree Isolation System

This module provides secure git worktree-based isolation for multi-agent coordination.
It enables each agent to work in isolated environments while maintaining security
boundaries and preventing unauthorized access to system resources.

Components:
- WorktreeManager: Core worktree lifecycle management
- PathValidator: Security validation for file system access
- SecurityEnforcer: Resource monitoring and constraint enforcement

Key Features:
- Path traversal attack prevention
- Symlink escape protection  
- System directory access blocking
- Resource usage monitoring and limits
- Process isolation and sandboxing
"""

from .worktree_manager import WorktreeManager
from .path_validator import PathValidator
from .security_enforcer import SecurityEnforcer

__all__ = [
    "WorktreeManager",
    "PathValidator", 
    "SecurityEnforcer"
]

# Version information
__version__ = "1.0.0"
__author__ = "LeanVibe Agent Hive"
__description__ = "Git Worktree Isolation System for Multi-Agent Coordination"