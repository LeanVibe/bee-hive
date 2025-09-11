"""
Consolidation Test Configuration

This conftest.py file sets up fixtures and configuration for orchestrator
consolidation tests.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

# Import all fixtures from enhanced_fixtures
from .enhanced_fixtures import (
    ConsolidatedComponentMock,
    consolidated_orchestrator_mock,
    task_manager_mock,
    agent_manager_mock,
    workflow_manager_mock,
    quality_gate_checker,
    performance_monitor,
    consolidation_validator
)


@pytest.fixture(scope="function")
def consolidated_managers_suite(
    task_manager_mock,
    agent_manager_mock, 
    workflow_manager_mock,
    resource_manager_mock,
    cache_manager_mock
):
    """Provide a suite of consolidated manager mocks."""
    return {
        "task_manager": task_manager_mock,
        "agent_manager": agent_manager_mock,
        "workflow_manager": workflow_manager_mock,
        "resource_manager": resource_manager_mock,
        "cache_manager": cache_manager_mock,
    }


@pytest.fixture(scope="function")
def resource_manager_mock():
    """Mock of consolidated ResourceManager."""
    mock = MagicMock()
    mock.allocate_memory = AsyncMock()
    mock.allocate_cpu = AsyncMock()
    mock.deallocate_resources = AsyncMock()
    mock.initialize = AsyncMock()
    mock.get_resource_usage = MagicMock(return_value={
        "memory_usage": 0.5,
        "cpu_usage": 0.3,
        "disk_usage": 0.2
    })
    return mock


@pytest.fixture(scope="function") 
def cache_manager_mock():
    """Mock of consolidated CacheManager."""
    mock = MagicMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock()
    mock.delete = AsyncMock()
    mock.clear = AsyncMock()
    mock.initialize = AsyncMock()
    mock.get_stats = MagicMock(return_value={
        "hit_rate": 0.8,
        "miss_rate": 0.2,
        "total_keys": 100
    })
    return mock