"""
LeanVibe Agent Hive 2.0 - Multi-Agent Orchestration System

A self-improving development environment where AI agents collaborate
to build software autonomously while maintaining human oversight.
"""

__version__ = "2.0.0"
__author__ = "LeanVibe Agent Hive Team"
__email__ = "dev@leanvibe.com"

# Load environment variables (optional). Avoid importing settings here to
# prevent requiring env vars at import time (breaks CI and tests).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

__all__: list[str] = []