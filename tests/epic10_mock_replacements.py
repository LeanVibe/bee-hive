"""
Epic 10 Test Mock Replacements

Provides mock implementations for problematic imports to enable test execution.
"""

from unittest.mock import MagicMock, AsyncMock
from enum import Enum
from typing import Dict, List, Any, Optional
import asyncio


class MockAgentRole(Enum):
    """Mock agent roles for testing."""
    DEVELOPER = "developer"
    QA = "qa"
    ARCHITECT = "architect"
    META = "meta"


class MockAgentStatus(Enum):
    """Mock agent status for testing."""
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"


class MockTaskPriority(Enum):
    """Mock task priorities for testing."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MockOrchestrator:
    """Mock orchestrator for testing."""
    
    def __init__(self):
        self.status = MockAgentStatus.IDLE
        self.agents = []
        
    async def execute_task(self, task: str, **kwargs):
        """Mock task execution."""
        return {"status": "completed", "result": f"Mock execution of: {task}"}
        
    def get_agent_status(self, agent_id: str):
        """Mock agent status."""
        return MockAgentStatus.IDLE
        
    def assign_task(self, task_id: str, agent_id: str):
        """Mock task assignment."""
        return {"assigned": True, "agent": agent_id, "task": task_id}


class MockWebSocketManager:
    """Mock WebSocket manager for testing."""
    
    async def connect(self, websocket, path):
        """Mock WebSocket connection."""
        pass
        
    async def broadcast(self, message: str):
        """Mock broadcast."""
        pass


class MockDatabase:
    """Mock database for testing."""
    
    async def connect(self):
        """Mock database connection."""
        pass
        
    async def execute(self, query: str):
        """Mock query execution."""
        return {"success": True, "rows": []}
        
    async def fetch_one(self, query: str):
        """Mock single row fetch."""
        return {"id": 1, "name": "test"}
        
    async def fetch_many(self, query: str):
        """Mock multiple rows fetch.""" 
        return [{"id": 1, "name": "test1"}, {"id": 2, "name": "test2"}]


class MockRedisManager:
    """Mock Redis manager for testing."""
    
    async def get(self, key: str):
        """Mock Redis get."""
        return "mock_value"
        
    async def set(self, key: str, value: str):
        """Mock Redis set."""
        return True
        
    async def delete(self, key: str):
        """Mock Redis delete."""
        return True


class MockPerformanceMonitor:
    """Mock performance monitor for testing."""
    
    def start_monitoring(self):
        """Mock monitoring start."""
        pass
        
    def stop_monitoring(self):
        """Mock monitoring stop."""
        return {"duration": 0.001, "memory": 100, "cpu": 0.1}
        
    def get_metrics(self):
        """Mock metrics retrieval."""
        return {"response_time": 0.001, "throughput": 1000}


class MockContextEngine:
    """Mock context engine for testing."""
    
    async def process_context(self, context: str):
        """Mock context processing."""
        return {"processed": True, "tokens": 100}
        
    def get_context_summary(self):
        """Mock context summary."""
        return {"total_contexts": 10, "active": 2}


# Export all mocks for easy importing
__all__ = [
    'MockAgentRole', 'MockAgentStatus', 'MockTaskPriority',
    'MockOrchestrator', 'MockWebSocketManager', 'MockDatabase',
    'MockRedisManager', 'MockPerformanceMonitor', 'MockContextEngine'
]
