"""
Test compatibility layer for auth-related imports.

Provides mock implementations to avoid PyO3/cryptography initialization issues
during test collection phase.
"""

import os
from unittest.mock import MagicMock, Mock
from typing import Any

# Set test environment before any auth imports
os.environ["TESTING"] = "true"

# Create mock JWT module to avoid cryptography import
class MockJWT:
    """Mock JWT implementation for testing."""
    
    @staticmethod
    def encode(payload, key, algorithm="HS256"):
        return "mock.jwt.token"
    
    @staticmethod
    def decode(token, key, algorithms=None):
        return {"user_id": "test_user", "exp": 9999999999}
    
    class ExpiredSignatureError(Exception):
        pass
    
    class InvalidTokenError(Exception):
        pass

# Mock auth router to prevent auth module import during testing
mock_auth_router = MagicMock()
mock_auth_router.routes = []

def get_mock_auth_dependencies():
    """Get mocked auth dependencies for testing."""
    return {
        "jwt": MockJWT(),
        "auth_router": mock_auth_router,
        "get_current_user": lambda: {"id": "test_user", "role": "admin"},
        "verify_token": lambda token: {"user_id": "test_user"},
    }