"""
API Endpoints Module - Epic C Phase 1

Core API endpoints for LeanVibe Agent Hive 2.0 multi-agent orchestration system.
Provides comprehensive REST API endpoints for agent and task management with
full orchestrator integration.

Features:
- Agent lifecycle management (creation, status control, deletion)
- Task management (creation, assignment, progress tracking)
- Real-time status monitoring and health checks
- Performance-optimized responses (<200ms target)
- Production-ready error handling and logging

Epic C Phase 1: API Endpoint Implementation
"""

from .agents import router as agents_router
from .tasks import router as tasks_router

__all__ = ["agents_router", "tasks_router"]