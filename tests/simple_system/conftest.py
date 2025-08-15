"""
Component Isolation Testing Configuration
========================================

Provides test fixtures and utilities for isolated component testing.
This file supports the comprehensive testing strategy for LeanVibe Agent Hive 2.0.

Key Principles:
- Complete isolation of components from external dependencies
- Mock all database, Redis, and API connections
- Test components in controlled, predictable environments
- Focus on component behavior and contracts
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, Mock
import uuid
from datetime import datetime, timedelta

# Test Data Factories for Component Isolation


@pytest.fixture
def isolated_agent_config():
    """Factory for creating isolated agent configurations."""
    def create_config(
        agent_type: str = "claude",
        capabilities: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        if capabilities is None:
            capabilities = ["python", "testing", "analysis"]
            
        return {
            "id": str(uuid.uuid4()),
            "name": f"test-agent-{uuid.uuid4().hex[:8]}",
            "type": agent_type,
            "role": kwargs.get("role", "backend-engineer"),
            "capabilities": [
                {
                    "name": cap,
                    "description": f"{cap} capability",
                    "confidence_level": 0.8,
                    "specialization_areas": ["testing"]
                }
                for cap in capabilities
            ],
            "status": kwargs.get("status", "active"),
            "config": kwargs.get("config", {"max_context": 8000}),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    
    return create_config


@pytest.fixture  
def isolated_task_config():
    """Factory for creating isolated task configurations."""
    def create_config(
        task_type: str = "feature_development",
        priority: str = "medium",
        **kwargs
    ) -> Dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "title": kwargs.get("title", f"Test Task {uuid.uuid4().hex[:8]}"),
            "description": kwargs.get("description", "Test task for component isolation"),
            "task_type": task_type,
            "status": kwargs.get("status", "pending"),
            "priority": priority,
            "required_capabilities": kwargs.get("required_capabilities", ["python"]),
            "estimated_effort": kwargs.get("estimated_effort", 60),
            "timeout_seconds": kwargs.get("timeout_seconds", 3600),
            "context": kwargs.get("context", {"test": True}),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    
    return create_config


@pytest.fixture
def isolated_workflow_config():
    """Factory for creating isolated workflow configurations.""" 
    def create_config(
        workflow_type: str = "sequential",
        **kwargs
    ) -> Dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "name": kwargs.get("name", f"Test Workflow {uuid.uuid4().hex[:8]}"),
            "description": kwargs.get("description", "Test workflow for isolation"),
            "status": kwargs.get("status", "created"),
            "priority": kwargs.get("priority", "medium"),
            "definition": {
                "type": workflow_type,
                "steps": kwargs.get("steps", ["task1", "task2"])
            },
            "context": kwargs.get("context", {"project": "test"}),
            "variables": kwargs.get("variables", {"env": "testing"}),
            "estimated_duration": kwargs.get("estimated_duration", 120),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
    
    return create_config


# Mock Infrastructure Components


@pytest.fixture
def mock_database_session():
    """Mock database session for component isolation."""
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.execute = AsyncMock()
    session.scalar_one_or_none = AsyncMock()
    session.scalars = AsyncMock()
    session.add = AsyncMock()
    session.delete = AsyncMock()
    session.refresh = AsyncMock()
    session.merge = AsyncMock()
    return session


@pytest.fixture
def mock_redis_streams():
    """Mock Redis client focused on stream operations."""
    redis = AsyncMock()
    
    # Stream operations
    redis.xadd = AsyncMock(return_value="1234567890-0")
    redis.xread = AsyncMock(return_value=[])
    redis.xreadgroup = AsyncMock(return_value=[])
    redis.xgroup_create = AsyncMock(return_value=True)
    redis.xinfo_groups = AsyncMock(return_value=[])
    
    # Pub/sub operations
    redis.publish = AsyncMock(return_value=1)
    redis.subscribe = AsyncMock()
    redis.unsubscribe = AsyncMock()
    
    # Key-value operations
    redis.set = AsyncMock(return_value=True)
    redis.get = AsyncMock(return_value=None)
    redis.delete = AsyncMock(return_value=1)
    redis.exists = AsyncMock(return_value=0)
    redis.expire = AsyncMock(return_value=True)
    
    # Health checks
    redis.ping = AsyncMock(return_value=True)
    
    return redis


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic API client for component isolation."""
    client = AsyncMock()
    
    # Mock message creation
    mock_message = Mock()
    mock_message.content = [Mock(text="Mock response")]
    mock_message.id = "msg_test_123"
    mock_message.model = "claude-3-sonnet-20240229"
    mock_message.role = "assistant"
    mock_message.stop_reason = "end_turn"
    
    client.messages.create = AsyncMock(return_value=mock_message)
    
    # Mock streaming responses
    async def mock_stream():
        yield Mock(delta=Mock(text="Mock "), type="content_block_delta")
        yield Mock(delta=Mock(text="streaming "), type="content_block_delta") 
        yield Mock(delta=Mock(text="response"), type="content_block_delta")
        yield Mock(type="message_stop")
    
    client.messages.stream = AsyncMock(return_value=mock_stream())
    
    return client


@pytest.fixture
def mock_vector_search():
    """Mock vector search engine for semantic memory isolation."""
    search_engine = AsyncMock()
    
    # Mock embedding operations
    search_engine.embed_text = AsyncMock(return_value=[0.1] * 1536)
    search_engine.embed_batch = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536])
    
    # Mock search operations
    search_engine.similarity_search = AsyncMock(return_value=[
        {
            "id": "doc_1",
            "content": "Mock search result 1",
            "score": 0.95,
            "metadata": {"source": "test"}
        }
    ])
    
    search_engine.semantic_search = AsyncMock(return_value=[
        {
            "id": "doc_2", 
            "content": "Mock semantic result",
            "score": 0.88,
            "metadata": {"type": "semantic"}
        }
    ])
    
    # Mock storage operations
    search_engine.store_document = AsyncMock(return_value="doc_123")
    search_engine.delete_document = AsyncMock(return_value=True)
    search_engine.update_document = AsyncMock(return_value=True)
    
    return search_engine


# Component Isolation Base Classes


@pytest.fixture
def isolated_orchestrator_base():
    """Base mock for orchestrator component isolation."""
    orchestrator = AsyncMock()
    
    # Agent management
    orchestrator.register_agent = AsyncMock(return_value={"success": True})
    orchestrator.unregister_agent = AsyncMock(return_value={"success": True})
    orchestrator.get_agent_status = AsyncMock(return_value="active")
    orchestrator.list_agents = AsyncMock(return_value=[])
    
    # Task management
    orchestrator.submit_task = AsyncMock(return_value={"task_id": str(uuid.uuid4())})
    orchestrator.assign_task = AsyncMock(return_value={"success": True})
    orchestrator.get_task_status = AsyncMock(return_value="pending")
    orchestrator.cancel_task = AsyncMock(return_value={"success": True})
    
    # Resource management
    orchestrator.get_system_status = AsyncMock(return_value={
        "healthy": True,
        "active_agents": 0,
        "pending_tasks": 0,
        "system_load": 0.1
    })
    
    return orchestrator


@pytest.fixture
def isolated_context_engine_base():
    """Base mock for context engine component isolation."""
    context_engine = AsyncMock()
    
    # Context management
    context_engine.store_context = AsyncMock(return_value=str(uuid.uuid4()))
    context_engine.retrieve_context = AsyncMock(return_value={
        "id": str(uuid.uuid4()),
        "content": "Mock context",
        "metadata": {"test": True}
    })
    context_engine.search_context = AsyncMock(return_value=[])
    context_engine.compress_context = AsyncMock(return_value={
        "compressed": "Mock compressed context",
        "compression_ratio": 0.3
    })
    
    # Memory operations
    context_engine.store_memory = AsyncMock(return_value=str(uuid.uuid4()))
    context_engine.recall_memory = AsyncMock(return_value=[])
    context_engine.consolidate_memories = AsyncMock(return_value={"success": True})
    
    # Knowledge graph
    context_engine.build_knowledge_graph = AsyncMock(return_value={
        "nodes": 10,
        "edges": 15,
        "clusters": 3
    })
    
    return context_engine


@pytest.fixture
def isolated_communication_base():
    """Base mock for communication system isolation."""
    comm_system = AsyncMock()
    
    # Message routing
    comm_system.send_message = AsyncMock(return_value={"success": True})
    comm_system.broadcast_message = AsyncMock(return_value={"delivered": 5})
    comm_system.route_message = AsyncMock(return_value={"routed_to": "agent_123"})
    
    # Connection management
    comm_system.establish_connection = AsyncMock(return_value={"connection_id": str(uuid.uuid4())})
    comm_system.close_connection = AsyncMock(return_value={"success": True})
    comm_system.get_connection_status = AsyncMock(return_value="connected")
    
    # Event streaming
    comm_system.subscribe_to_events = AsyncMock(return_value={"subscription_id": str(uuid.uuid4())})
    comm_system.publish_event = AsyncMock(return_value={"published": True})
    comm_system.unsubscribe_from_events = AsyncMock(return_value={"success": True})
    
    return comm_system


# Test Environment Setup


@pytest.fixture(autouse=True)
def isolated_test_environment():
    """Setup completely isolated test environment."""
    import os
    
    # Override environment variables for isolation
    test_env = {
        "TESTING": "true",
        "ANTHROPIC_API_KEY": "test-key-isolated",
        "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
        "REDIS_URL": "redis://mock-redis:6379/15",
        "SECRET_KEY": "test-secret-isolated",
        "LOG_LEVEL": "ERROR",
        "SKIP_DATABASE_INIT": "true",
        "SKIP_REDIS_INIT": "true",
        "ENABLE_COMPONENT_ISOLATION": "true"
    }
    
    # Temporarily override environment
    original_env = {}
    for key, value in test_env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield
    
    # Restore original environment
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# Event Loop Management for Isolation Tests


@pytest_asyncio.fixture(scope="function")
async def isolated_event_loop():
    """Create isolated event loop for each test."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    yield loop
    
    # Clean shutdown
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    
    loop.close()


# Test Utilities for Component Isolation


@pytest.fixture
def assert_isolated():
    """Utility for asserting component isolation."""
    def _assert_isolated(component, dependencies):
        """Assert that component does not have access to real dependencies."""
        for dep_name, dep_instance in dependencies.items():
            # Check that dependency is mocked
            assert hasattr(dep_instance, "_mock_name") or hasattr(dep_instance, "call_count"), \
                f"Dependency {dep_name} is not properly mocked for isolation"
        
        # Check that component does not have real network/database connections
        if hasattr(component, "_redis_client"):
            assert hasattr(component._redis_client, "_mock_name"), \
                "Component has real Redis connection, breaking isolation"
        
        if hasattr(component, "_db_session"):
            assert hasattr(component._db_session, "_mock_name"), \
                "Component has real database connection, breaking isolation"
    
    return _assert_isolated


@pytest.fixture
def capture_component_calls():
    """Utility for capturing and analyzing component method calls."""
    calls = []
    
    def _capture_calls(component, method_names):
        """Wrap component methods to capture calls."""
        original_methods = {}
        
        for method_name in method_names:
            if hasattr(component, method_name):
                original_method = getattr(component, method_name)
                original_methods[method_name] = original_method
                
                async def wrapped_method(*args, _method_name=method_name, _original=original_method, **kwargs):
                    call_info = {
                        "method": _method_name,
                        "args": args,
                        "kwargs": kwargs,
                        "timestamp": datetime.utcnow()
                    }
                    calls.append(call_info)
                    
                    if asyncio.iscoroutinefunction(_original):
                        return await _original(*args, **kwargs)
                    else:
                        return _original(*args, **kwargs)
                
                setattr(component, method_name, wrapped_method)
        
        return calls, original_methods
    
    return _capture_calls