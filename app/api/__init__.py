"""
API Package for LeanVibe Agent Hive 2.0

Main API router configuration and endpoint organization for enterprise
autonomous development platform.
"""

import os
from fastapi import APIRouter
from .enterprise_pilots import router as pilots_router
from .project_index import router as project_index_router

# Conditional auth import to prevent PyO3/cryptography issues during testing
if os.environ.get("TESTING"):
    # Use mock auth router during testing to avoid cryptography import
    from unittest.mock import MagicMock
    auth_router = MagicMock()
    auth_router.routes = []
else:
    from ..core.auth import auth_router

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth_router)
api_router.include_router(pilots_router)
api_router.include_router(project_index_router)

__all__ = ["api_router"]