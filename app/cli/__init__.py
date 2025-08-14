"""
Unix Philosophy CLI Package for LeanVibe Agent Hive 2.0

This package provides kubectl/docker-style individual commands that follow Unix principles:
- Each command does one thing well
- Commands are composable and pipeable
- Consistent interface patterns
- JSON output for programmatic use
"""

from .unix_commands import unix_commands

__all__ = ['unix_commands']