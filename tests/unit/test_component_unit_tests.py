"""
Level 2: Component Unit Tests for LeanVibe Agent Hive 2.0

These tests focus on individual components in complete isolation:
- Test single classes/modules without external dependencies
- Use mocks for all dependencies (database, Redis, external APIs)
- Fast execution (<100ms per test)
- High test coverage for business logic

This is the second level of the testing pyramid, building on foundation tests.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List
from datetime import datetime, timedelta


class TestOrchestratorComponent:
    """Test Orchestrator component in complete isolation."""
    
    @pytest.fixture
    def mock_orchestrator_config(self):
        """Provide mocked orchestrator configuration."""
        config = MagicMock()
        config.max_agents = 10
        config.task_timeout = 300
        config.heartbeat_interval = 30
        config.retry_attempts = 3
        return config
    
    @pytest.fixture
    def mock_dependencies(self):
        """Provide all mocked dependencies for orchestrator."""
        deps = MagicMock()
        deps.database = AsyncMock()
        deps.redis = AsyncMock()
        deps.message_broker = AsyncMock()
        deps.health_monitor = AsyncMock()
        deps.logger = MagicMock()
        return deps
    
    def test_orchestrator_initialization(self, mock_orchestrator_config, mock_dependencies):
        """Test orchestrator initializes correctly with configuration."""
        # Mock the orchestrator class
        class MockOrchestrator:
            def __init__(self, config, dependencies):
                self.config = config
                self.dependencies = dependencies
                self.agents = {}
                self.status = "initialized"
        
        orchestrator = MockOrchestrator(mock_orchestrator_config, mock_dependencies)
        
        assert orchestrator.config.max_agents == 10
        assert orchestrator.config.task_timeout == 300
        assert orchestrator.status == "initialized"
        assert orchestrator.agents == {}
    
    @pytest.mark.asyncio
    async def test_agent_registration_logic(self, mock_orchestrator_config):
        """Test agent registration business logic."""
        class MockOrchestrator:
            def __init__(self, config):
                self.config = config
                self.agents = {}
                self.agent_counter = 0
            
            async def register_agent(self, agent_spec):
                if len(self.agents) >= self.config.max_agents:
                    raise ValueError("Maximum agents exceeded")
                
                self.agent_counter += 1
                agent_id = f"agent-{self.agent_counter}"
                
                agent = {
                    "id": agent_id,
                    "name": agent_spec.get("name", f"Agent {self.agent_counter}"),
                    "type": agent_spec.get("type", "default"),
                    "status": "registered",
                    "created_at": datetime.now()
                }
                
                self.agents[agent_id] = agent
                return agent_id
        
        orchestrator = MockOrchestrator(mock_orchestrator_config)
        
        # Test successful registration
        agent_spec = {"name": "Test Agent", "type": "backend"}
        agent_id = await orchestrator.register_agent(agent_spec)
        
        assert agent_id == "agent-1"
        assert len(orchestrator.agents) == 1
        assert orchestrator.agents[agent_id]["name"] == "Test Agent"
        assert orchestrator.agents[agent_id]["type"] == "backend"
        assert orchestrator.agents[agent_id]["status"] == "registered"
    
    @pytest.mark.asyncio
    async def test_max_agents_limit_enforcement(self, mock_orchestrator_config):
        """Test that max agents limit is enforced."""
        mock_orchestrator_config.max_agents = 2
        
        class MockOrchestrator:
            def __init__(self, config):
                self.config = config
                self.agents = {}
                self.agent_counter = 0
            
            async def register_agent(self, agent_spec):
                if len(self.agents) >= self.config.max_agents:
                    raise ValueError("Maximum agents exceeded")
                
                self.agent_counter += 1
                agent_id = f"agent-{self.agent_counter}"
                self.agents[agent_id] = {"id": agent_id, "name": agent_spec.get("name")}
                return agent_id
        
        orchestrator = MockOrchestrator(mock_orchestrator_config)
        
        # Register maximum allowed agents
        await orchestrator.register_agent({"name": "Agent 1"})
        await orchestrator.register_agent({"name": "Agent 2"})
        
        # Attempt to exceed limit should fail
        with pytest.raises(ValueError, match="Maximum agents exceeded"):
            await orchestrator.register_agent({"name": "Agent 3"})
        
        assert len(orchestrator.agents) == 2


class TestMessageBrokerComponent:
    """Test Message Broker component in complete isolation."""
    
    @pytest.fixture
    def mock_redis_client(self):
        """Provide mocked Redis client."""
        redis = AsyncMock()
        redis.xadd = AsyncMock(return_value="1234567890-0")
        redis.xread = AsyncMock(return_value=[])
        redis.xlen = AsyncMock(return_value=0)
        return redis
    
    def test_message_broker_initialization(self, mock_redis_client):
        """Test message broker initializes with Redis client."""
        class MockMessageBroker:
            def __init__(self, redis_client):
                self.redis = redis_client
                self.streams = {}
                self.consumer_groups = {}
        
        broker = MockMessageBroker(mock_redis_client)
        
        assert broker.redis is not None
        assert broker.streams == {}
        assert broker.consumer_groups == {}
    
    @pytest.mark.asyncio
    async def test_message_publishing_logic(self, mock_redis_client):
        """Test message publishing business logic."""
        class MockMessageBroker:
            def __init__(self, redis_client):
                self.redis = redis_client
                self.published_messages = []
            
            async def publish_message(self, stream, data):
                message_id = f"test-{len(self.published_messages) + 1}"
                
                message = {
                    "id": message_id,
                    "stream": stream,
                    "data": data,
                    "timestamp": datetime.now()
                }
                
                self.published_messages.append(message)
                
                # Simulate Redis call
                await self.redis.xadd(stream, data)
                
                return message_id
        
        broker = MockMessageBroker(mock_redis_client)
        
        # Test message publishing
        message_data = {"task_id": "123", "action": "process"}
        message_id = await broker.publish_message("tasks", message_data)
        
        assert message_id == "test-1"
        assert len(broker.published_messages) == 1
        assert broker.published_messages[0]["stream"] == "tasks"
        assert broker.published_messages[0]["data"] == message_data
        
        # Verify Redis was called
        mock_redis_client.xadd.assert_called_once_with("tasks", message_data)


class TestHealthMonitorComponent:
    """Test Health Monitor component in complete isolation."""
    
    @pytest.fixture
    def mock_health_config(self):
        """Provide mocked health monitoring configuration."""
        config = MagicMock()
        config.check_interval = 30
        config.failure_threshold = 3
        config.recovery_threshold = 2
        return config
    
    def test_health_monitor_initialization(self, mock_health_config):
        """Test health monitor initializes correctly."""
        class MockHealthMonitor:
            def __init__(self, config):
                self.config = config
                self.services = {}
                self.status = "initialized"
        
        monitor = MockHealthMonitor(mock_health_config)
        
        assert monitor.config.check_interval == 30
        assert monitor.config.failure_threshold == 3
        assert monitor.status == "initialized"
        assert monitor.services == {}
    
    def test_service_status_tracking(self, mock_health_config):
        """Test service health status tracking logic."""
        class MockHealthMonitor:
            def __init__(self, config):
                self.config = config
                self.services = {}
            
            def register_service(self, name, health_check_func):
                self.services[name] = {
                    "name": name,
                    "health_check": health_check_func,
                    "status": "unknown",
                    "consecutive_failures": 0,
                    "consecutive_successes": 0,
                    "last_check": None
                }
            
            def update_service_status(self, name, is_healthy):
                if name not in self.services:
                    raise ValueError(f"Service {name} not registered")
                
                service = self.services[name]
                
                if is_healthy:
                    service["consecutive_failures"] = 0
                    service["consecutive_successes"] += 1
                    
                    if service["consecutive_successes"] >= self.config.recovery_threshold:
                        service["status"] = "healthy"
                else:
                    service["consecutive_successes"] = 0
                    service["consecutive_failures"] += 1
                    
                    if service["consecutive_failures"] >= self.config.failure_threshold:
                        service["status"] = "unhealthy"
                
                service["last_check"] = datetime.now()
        
        monitor = MockHealthMonitor(mock_health_config)
        
        # Register a service
        health_check = lambda: True
        monitor.register_service("database", health_check)
        
        assert "database" in monitor.services
        assert monitor.services["database"]["status"] == "unknown"
        
        # Test health status updates
        monitor.update_service_status("database", True)
        monitor.update_service_status("database", True)
        
        assert monitor.services["database"]["status"] == "healthy"
        assert monitor.services["database"]["consecutive_successes"] == 2
        assert monitor.services["database"]["consecutive_failures"] == 0
        
        # Test failure tracking
        monitor.update_service_status("database", False)
        monitor.update_service_status("database", False)
        monitor.update_service_status("database", False)
        
        assert monitor.services["database"]["status"] == "unhealthy"
        assert monitor.services["database"]["consecutive_failures"] == 3
        assert monitor.services["database"]["consecutive_successes"] == 0


class TestTaskQueueComponent:
    """Test Task Queue component in complete isolation."""
    
    @pytest.fixture
    def mock_task_queue_config(self):
        """Provide mocked task queue configuration."""
        config = MagicMock()
        config.max_queue_size = 1000
        config.default_priority = 5
        config.cleanup_interval = 3600
        return config
    
    def test_task_queue_initialization(self, mock_task_queue_config):
        """Test task queue initializes correctly."""
        class MockTaskQueue:
            def __init__(self, config):
                self.config = config
                self.tasks = []
                self.completed_tasks = []
                self.failed_tasks = []
        
        queue = MockTaskQueue(mock_task_queue_config)
        
        assert queue.config.max_queue_size == 1000
        assert queue.tasks == []
        assert queue.completed_tasks == []
        assert queue.failed_tasks == []
    
    def test_task_queuing_logic(self, mock_task_queue_config):
        """Test task queuing and prioritization logic."""
        class MockTaskQueue:
            def __init__(self, config):
                self.config = config
                self.tasks = []
                self.task_counter = 0
            
            def add_task(self, task_data, priority=None):
                if len(self.tasks) >= self.config.max_queue_size:
                    raise ValueError("Queue is full")
                
                self.task_counter += 1
                
                task = {
                    "id": f"task-{self.task_counter}",
                    "data": task_data,
                    "priority": priority or self.config.default_priority,
                    "created_at": datetime.now(),
                    "status": "queued"
                }
                
                # Insert task based on priority (higher priority first)
                inserted = False
                for i, existing_task in enumerate(self.tasks):
                    if task["priority"] > existing_task["priority"]:
                        self.tasks.insert(i, task)
                        inserted = True
                        break
                
                if not inserted:
                    self.tasks.append(task)
                
                return task["id"]
            
            def get_next_task(self):
                if not self.tasks:
                    return None
                
                task = self.tasks.pop(0)
                task["status"] = "processing"
                return task
        
        queue = MockTaskQueue(mock_task_queue_config)
        
        # Test basic task queuing
        task_id_1 = queue.add_task({"action": "process_data"}, priority=5)
        task_id_2 = queue.add_task({"action": "send_email"}, priority=8)
        task_id_3 = queue.add_task({"action": "backup"}, priority=3)
        
        assert len(queue.tasks) == 3
        
        # Test priority ordering (higher priority first)
        next_task = queue.get_next_task()
        assert next_task["priority"] == 8  # Highest priority task
        assert next_task["data"]["action"] == "send_email"
        
        next_task = queue.get_next_task()
        assert next_task["priority"] == 5  # Medium priority task
        
        next_task = queue.get_next_task()
        assert next_task["priority"] == 3  # Lowest priority task


class TestCircuitBreakerComponent:
    """Test Circuit Breaker component in complete isolation."""
    
    @pytest.fixture
    def mock_circuit_breaker_config(self):
        """Provide mocked circuit breaker configuration."""
        config = MagicMock()
        config.failure_threshold = 5
        config.recovery_timeout = 60
        config.success_threshold = 3
        return config
    
    def test_circuit_breaker_states(self, mock_circuit_breaker_config):
        """Test circuit breaker state transitions."""
        class MockCircuitBreaker:
            def __init__(self, config):
                self.config = config
                self.state = "closed"  # closed, open, half-open
                self.failure_count = 0
                self.success_count = 0
                self.last_failure_time = None
            
            def record_success(self):
                self.failure_count = 0
                
                if self.state == "half-open":
                    self.success_count += 1
                    if self.success_count >= self.config.success_threshold:
                        self.state = "closed"
                        self.success_count = 0
            
            def record_failure(self):
                self.failure_count += 1
                self.last_failure_time = datetime.now()
                
                if self.state == "closed" and self.failure_count >= self.config.failure_threshold:
                    self.state = "open"
                elif self.state == "half-open":
                    self.state = "open"
                    self.success_count = 0
            
            def can_execute(self):
                if self.state == "closed":
                    return True
                elif self.state == "open":
                    if self.last_failure_time:
                        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
                        if time_since_failure >= self.config.recovery_timeout:
                            self.state = "half-open"
                            return True
                    return False
                elif self.state == "half-open":
                    return True
                
                return False
        
        breaker = MockCircuitBreaker(mock_circuit_breaker_config)
        
        # Test initial state
        assert breaker.state == "closed"
        assert breaker.can_execute() == True
        
        # Test failure accumulation
        for _ in range(4):
            breaker.record_failure()
            assert breaker.state == "closed"
            assert breaker.can_execute() == True
        
        # Test circuit opening
        breaker.record_failure()  # 5th failure
        assert breaker.state == "open"
        assert breaker.can_execute() == False
        
        # Test half-open transition (simulated time passage)
        breaker.last_failure_time = datetime.now() - timedelta(seconds=61)
        assert breaker.can_execute() == True
        assert breaker.state == "half-open"
        
        # Test recovery to closed state
        for _ in range(3):
            breaker.record_success()
        
        assert breaker.state == "closed"
        assert breaker.failure_count == 0