"""
Unix Philosophy CLI Package for LeanVibe Agent Hive 2.0

This package provides kubectl/docker-style individual commands that follow Unix principles:
- Each command does one thing well
- Commands are composable and pipeable
- Consistent interface patterns
- JSON output for programmatic use

REFACTORED: Phase 1.2 Technical Debt Remediation - Standardized __init__.py pattern
"""

# Standard utility imports
from typing import Any, Callable, Dict, List, Optional

# Utility imports - Auto-generated
from .unix_commands import unix_commands
from .agent_hive_cli import AgentHiveCLI

# Utility exports - Auto-generated
__all__ = [
    "unix_commands",
    "AgentHiveCLI",
]

# Standard module initialization
import logging
logger = logging.getLogger(__name__)
logger.debug(f"CLI package initialized: {__name__}")