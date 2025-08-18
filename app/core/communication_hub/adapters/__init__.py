"""
Protocol Adapters for CommunicationHub

This package contains all protocol adapters that enable CommunicationHub
to communicate over different protocols including Redis, WebSocket, HTTP, etc.
"""

from .base_adapter import (
    BaseProtocolAdapter,
    AdapterStatus,
    HealthStatus,
    AdapterMetrics,
    ConnectionInfo,
    MessageHandler,
    AdapterRegistry
)

from .redis_adapter import RedisAdapter
from .websocket_adapter import WebSocketAdapter

__all__ = [
    "BaseProtocolAdapter",
    "AdapterStatus", 
    "HealthStatus",
    "AdapterMetrics",
    "ConnectionInfo",
    "MessageHandler",
    "AdapterRegistry",
    "RedisAdapter",
    "WebSocketAdapter"
]