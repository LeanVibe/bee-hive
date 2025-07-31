"""
LeanVibe Agent Hive 2.0 - Multi-Agent Orchestration System

A self-improving development environment where AI agents collaborate
to build software autonomously while maintaining human oversight.
"""

__version__ = "2.0.0"
__author__ = "LeanVibe Agent Hive Team"
__email__ = "dev@leanvibe.com"

# Load environment variables before importing config
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If python-dotenv is not installed, continue without it
    pass

from .core.config import settings

__all__ = ["settings"]