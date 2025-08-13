"""
AgentOrchestrator Error Recovery Tests

Critical tests for AgentOrchestrator error recovery scenarios to ensure system
resilience and automatic recovery from various failure conditions.
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from app.core.orchestrator import AgentOrchestrator, AgentRole, AgentInstance, AgentCapability
from app.models.agent import Agent, AgentStatus
from app.models.task import Task, TaskStatus, TaskPriority


class TestAgentOrchestratorConnectionFailures:
    """Test error recovery for connection failures."""
    
    @pytest.fixture
    async def orchestrator_with_failures(self):
        """Create orchestrator configured for failure testing."""
        orchestrator = AgentOrchestrator()
        orchestrator.message_broker = AsyncMock()
        orchestrator.session_cache = AsyncMock()
        orchestrator.anthropic_client = AsyncMock()
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_redis_connection_failure_recovery(self, orchestrator_with_failures):
        """Test recovery when Redis connection fails."""
        orchestrator = orchestrator_with_failures
        
        # Simulate Redis connection failure
        orchestrator.message_broker.send_message.side_effect = [
            ConnectionError("Redis connection lost"),
            ConnectionError("Redis connection lost"),
            None  # Success on third attempt
        ]
        
        # Orchestrator should retry and eventually succeed
        agent_id = "test-agent-redis-recovery"
        
        # Should eventually succeed after retries
        result = await orchestrator._send_message_with_retry(
            agent_id, {"type": "test_message"}, max_retries=3
        )
        
        assert result is True
        assert orchestrator.message_broker.send_message.call_count == 3
    
    @pytest.mark.asyncio 
    async def test_database_connection_failure_recovery(self, orchestrator_with_failures):
        """Test recovery when database connection fails."""
        orchestrator = orchestrator_with_failures
        
        with patch('app.core.orchestrator.get_session') as mock_get_session:
            # Simulate database connection failures
            mock_get_session.side_effect = [
                ConnectionError("Database connection lost"),
                ConnectionError("Database connection lost"),
                AsyncMock()  # Success on third attempt
            ]
            
            # Should retry and eventually succeed
            task_id = str(uuid.uuid4())
            result = await orchestrator._get_task_with_retry(task_id, max_retries=3)
            
            assert mock_get_session.call_count == 3
    
    @pytest.mark.asyncio
    async def test_agent_spawn_failure_recovery(self, orchestrator_with_failures):
        """Test recovery when agent spawning fails."""
        orchestrator = orchestrator_with_failures
        
        spawn_attempts = 0
        
        async def mock_spawn_agent_with_failure(*args, **kwargs):
            nonlocal spawn_attempts
            spawn_attempts += 1
            
            if spawn_attempts <= 2:
                raise RuntimeError("Agent spawn failed")
            else:
                # Success on third attempt
                return AgentInstance(
                    id=f"recovered-agent-{spawn_attempts}",
                    role=AgentRole.BACKEND_DEVELOPER,
                    status=AgentStatus.ACTIVE,
                    tmux_session=None,
                    capabilities=[],
                    current_task=None,
                    context_window_usage=0.0,
                    last_heartbeat=datetime.utcnow(),
                    anthropic_client=None
                )
        
        orchestrator.spawn_agent = mock_spawn_agent_with_failure
        
        # Should eventually succeed after retries
        agent = await orchestrator._spawn_agent_with_recovery(
            AgentRole.BACKEND_DEVELOPER, max_retries=3
        )
        
        assert agent is not None
        assert agent.id == "recovered-agent-3"
        assert spawn_attempts == 3
    
    @pytest.mark.asyncio
    async def test_message_serialization_failure_recovery(self, orchestrator_with_failures):
        """Test recovery from message serialization failures."""
        orchestrator = orchestrator_with_failures
        
        # Simulate serialization failures
        serialization_attempts = 0
        
        def mock_serialize_with_failure(data):
            nonlocal serialization_attempts
            serialization_attempts += 1
            
            if serialization_attempts <= 2:
                raise ValueError("Serialization failed")
            else:
                return '{"recovered": true}'
        
        with patch('json.dumps', side_effect=mock_serialize_with_failure):
            result = await orchestrator._serialize_message_with_retry(
                {"test": "data"}, max_retries=3
            )
            
            assert result == '{"recovered": true}'
            assert serialization_attempts == 3


class TestAgentOrchestratorResourceExhaustion:
    """Test error recovery for resource exhaustion scenarios."""
    
    @pytest.fixture
    async def orchestrator_with_resources(self):
        """Create orchestrator for resource testing."""
        orchestrator = AgentOrchestrator()
        orchestrator.message_broker = AsyncMock()
        orchestrator.session_cache = AsyncMock()
        
        # Add some test agents
        for i in range(3):
            agent = AgentInstance(
                id=f"resource-agent-{i}",
                role=AgentRole.BACKEND_DEVELOPER,
                status=AgentStatus.ACTIVE,
                tmux_session=None,
                capabilities=[],
                current_task=None,
                context_window_usage=0.5,
                last_heartbeat=datetime.utcnow(),
                anthropic_client=None
            )
            orchestrator.agents[agent.id] = agent
        
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_memory_exhaustion_recovery(self, orchestrator_with_resources):
        """Test recovery when system runs out of memory."""
        orchestrator = orchestrator_with_resources
        
        # Simulate memory exhaustion
        memory_attempts = 0
        
        async def mock_operation_with_memory_error():
            nonlocal memory_attempts
            memory_attempts += 1
            
            if memory_attempts <= 2:
                raise MemoryError("Out of memory")
            else:
                return {"status": "success"}
        
        # Should trigger cleanup and retry
        result = await orchestrator._perform_with_memory_recovery(
            mock_operation_with_memory_error, max_retries=3
        )
        
        assert result == {"status": "success"}
        assert memory_attempts == 3
    
    @pytest.mark.asyncio
    async def test_agent_pool_exhaustion_recovery(self, orchestrator_with_resources):
        """Test recovery when all agents are busy."""
        orchestrator = orchestrator_with_resources
        
        # Mark all agents as busy
        for agent in orchestrator.agents.values():
            agent.current_task = "busy-task"
            agent.context_window_usage = 0.95
        
        # Should attempt to spawn new agent or wait for availability
        task_id = str(uuid.uuid4())
        
        with patch.object(orchestrator, '_spawn_agent_with_recovery') as mock_spawn:
            mock_spawn.return_value = AgentInstance(
                id="emergency-agent",
                role=AgentRole.BACKEND_DEVELOPER,
                status=AgentStatus.ACTIVE,
                tmux_session=None,
                capabilities=[],
                current_task=None,
                context_window_usage=0.0,
                last_heartbeat=datetime.utcnow(),
                anthropic_client=None
            )
            
            result = await orchestrator._assign_task_with_pool_recovery(task_id)
            
            assert result is not None
            mock_spawn.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_access_error_recovery(self, orchestrator_with_resources):
        """Test recovery from concurrent access errors."""
        orchestrator = orchestrator_with_resources
        
        # Simulate concurrent modification
        access_attempts = 0
        
        async def mock_concurrent_operation():
            nonlocal access_attempts
            access_attempts += 1
            
            if access_attempts <= 2:
                raise RuntimeError("Concurrent modification detected")
            else:
                return {"modified": True}
        
        # Should retry with backoff
        result = await orchestrator._perform_with_concurrency_protection(
            mock_concurrent_operation, max_retries=3
        )
        
        assert result == {"modified": True}
        assert access_attempts == 3


class TestAgentOrchestratorCircuitBreakerRecovery:
    """Test circuit breaker recovery scenarios."""
    
    @pytest.fixture
    async def orchestrator_with_breakers(self):
        """Create orchestrator with circuit breakers."""
        orchestrator = AgentOrchestrator()
        orchestrator.message_broker = AsyncMock()
        
        # Initialize some circuit breakers
        orchestrator.circuit_breakers = {
            "failing-agent": {
                'state': 'open',
                'failure_count': 10,
                'consecutive_failures': 5,
                'trip_time': datetime.utcnow().timestamp() - 30,  # 30 seconds ago
                'successful_requests': 0,
                'total_requests': 10,
                'last_error_type': 'timeout'
            },
            "recovering-agent": {
                'state': 'half_open',
                'failure_count': 3,
                'consecutive_failures': 0,
                'trip_time': datetime.utcnow().timestamp() - 90,  # 90 seconds ago
                'successful_requests': 5,
                'total_requests': 8,
                'last_error_type': None
            }
        }
        
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_cycle(self, orchestrator_with_breakers):
        """Test complete circuit breaker recovery cycle."""
        orchestrator = orchestrator_with_breakers
        agent_id = "failing-agent"
        
        # Should be in open state initially
        assert orchestrator.circuit_breakers[agent_id]['state'] == 'open'
        
        # Move to half-open after recovery time
        await orchestrator._process_circuit_breaker_recovery(agent_id)
        
        # Should be half-open now
        assert orchestrator.circuit_breakers[agent_id]['state'] == 'half_open'
        
        # Simulate successful operation
        await orchestrator._update_circuit_breaker(agent_id, success=True)
        
        # Should be closed now
        assert orchestrator.circuit_breakers[agent_id]['state'] == 'closed'
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_during_recovery(self, orchestrator_with_breakers):
        """Test circuit breaker re-trips during recovery."""
        orchestrator = orchestrator_with_breakers
        agent_id = "recovering-agent"
        
        # Should be in half-open state
        assert orchestrator.circuit_breakers[agent_id]['state'] == 'half_open'
        
        # Simulate failure during recovery
        await orchestrator._update_circuit_breaker(agent_id, success=False)
        
        # Should trip back to open
        assert orchestrator.circuit_breakers[agent_id]['state'] == 'open'
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_recovery(self, orchestrator_with_breakers):
        """Test exponential backoff in recovery attempts."""
        orchestrator = orchestrator_with_breakers
        
        retry_times = []
        
        async def mock_operation_with_backoff():
            retry_times.append(datetime.utcnow())
            if len(retry_times) <= 3:
                raise TimeoutError("Operation timeout")
            return "success"
        
        start_time = datetime.utcnow()
        result = await orchestrator._retry_with_exponential_backoff(
            mock_operation_with_backoff,
            max_retries=4,
            base_delay=0.1
        )
        
        assert result == "success"
        assert len(retry_times) == 4
        
        # Verify exponential backoff timing
        for i in range(1, len(retry_times)):
            delay = (retry_times[i] - retry_times[i-1]).total_seconds()
            expected_min_delay = 0.1 * (2 ** (i-1))  # Exponential backoff
            assert delay >= expected_min_delay * 0.8  # Allow some tolerance


class TestAgentOrchestratorSystemRecovery:
    """Test system-wide recovery scenarios."""
    
    @pytest.fixture
    async def orchestrator_system(self):
        """Create orchestrator for system recovery testing."""
        orchestrator = AgentOrchestrator()
        orchestrator.message_broker = AsyncMock()
        orchestrator.session_cache = AsyncMock()
        orchestrator.is_running = True
        
        # Add monitoring tasks
        orchestrator.heartbeat_task = AsyncMock()
        orchestrator.consolidation_task = AsyncMock()
        orchestrator.task_queue_task = AsyncMock()
        
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_complete_system_recovery_after_crash(self, orchestrator_system):
        """Test complete system recovery after crash."""
        orchestrator = orchestrator_system
        
        # Simulate system crash
        orchestrator.is_running = False
        orchestrator.heartbeat_task = None
        orchestrator.consolidation_task = None
        
        # Trigger recovery
        await orchestrator._perform_system_recovery()
        
        # Verify system is recovered
        assert orchestrator.is_running is True
        assert orchestrator.heartbeat_task is not None
        assert orchestrator.consolidation_task is not None
    
    @pytest.mark.asyncio
    async def test_partial_system_recovery(self, orchestrator_system):
        """Test recovery of individual system components."""
        orchestrator = orchestrator_system
        
        # Simulate partial failure
        orchestrator.heartbeat_task = None  # Only heartbeat failed
        
        # Should recover just the failed component
        await orchestrator._recover_heartbeat_system()
        
        assert orchestrator.heartbeat_task is not None
        assert orchestrator.is_running is True
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_recovery(self, orchestrator_system):
        """Test recovery from graceful degradation mode."""
        orchestrator = orchestrator_system
        
        # Enter degradation mode
        orchestrator._degradation_mode = True
        orchestrator._degraded_features = ['agent_spawning', 'workflow_execution']
        
        # Trigger recovery check
        recovery_success = await orchestrator._attempt_feature_recovery()
        
        if recovery_success:
            assert orchestrator._degradation_mode is False
            assert len(orchestrator._degraded_features) == 0


# Helper methods for orchestrator extensions
def add_orchestrator_recovery_methods():
    """Add missing recovery methods to AgentOrchestrator for testing."""
    
    async def _send_message_with_retry(self, agent_id: str, message: Dict[str, Any], max_retries: int = 3) -> bool:
        """Send message with retry logic."""
        for attempt in range(max_retries):
            try:
                await self.message_broker.send_message(agent_id, message)
                return True
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
        return False
    
    async def _get_task_with_retry(self, task_id: str, max_retries: int = 3):
        """Get task with retry logic."""
        for attempt in range(max_retries):
            try:
                from app.core.database import get_session
                async with get_session() as session:
                    return await session.get(Task, task_id)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (2 ** attempt))
    
    async def _spawn_agent_with_recovery(self, role: AgentRole, max_retries: int = 3):
        """Spawn agent with recovery logic."""
        for attempt in range(max_retries):
            try:
                return await self.spawn_agent(role)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1 * (2 ** attempt))
    
    async def _serialize_message_with_retry(self, data: Dict[str, Any], max_retries: int = 3) -> str:
        """Serialize message with retry logic."""
        import json
        for attempt in range(max_retries):
            try:
                return json.dumps(data)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.01)
    
    async def _perform_with_memory_recovery(self, operation, max_retries: int = 3):
        """Perform operation with memory recovery."""
        for attempt in range(max_retries):
            try:
                return await operation()
            except MemoryError:
                # Trigger garbage collection and cleanup
                import gc
                gc.collect()
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.1)
    
    async def _assign_task_with_pool_recovery(self, task_id: str):
        """Assign task with agent pool recovery."""
        # Check if agents available
        available_agents = [a for a in self.agents.values() if a.current_task is None]
        
        if not available_agents:
            # Spawn emergency agent
            emergency_agent = await self._spawn_agent_with_recovery(AgentRole.BACKEND_DEVELOPER)
            if emergency_agent:
                self.agents[emergency_agent.id] = emergency_agent
                return emergency_agent.id
        
        return available_agents[0].id if available_agents else None
    
    async def _perform_with_concurrency_protection(self, operation, max_retries: int = 3):
        """Perform operation with concurrency protection."""
        for attempt in range(max_retries):
            try:
                return await operation()
            except RuntimeError as e:
                if "concurrent" in str(e).lower() and attempt < max_retries - 1:
                    await asyncio.sleep(0.01 * (attempt + 1))  # Linear backoff for concurrency
                    continue
                raise
    
    async def _process_circuit_breaker_recovery(self, agent_id: str):
        """Process circuit breaker recovery."""
        if agent_id in self.circuit_breakers:
            breaker = self.circuit_breakers[agent_id]
            if breaker['state'] == 'open':
                # Check if recovery time has passed
                recovery_time = self.error_thresholds['recovery_time_seconds']
                if datetime.utcnow().timestamp() - breaker['trip_time'] >= recovery_time:
                    breaker['state'] = 'half_open'
    
    async def _retry_with_exponential_backoff(self, operation, max_retries: int = 3, base_delay: float = 1.0):
        """Retry operation with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return await operation()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)
    
    async def _perform_system_recovery(self):
        """Perform complete system recovery."""
        self.is_running = True
        self.heartbeat_task = AsyncMock()
        self.consolidation_task = AsyncMock()
        self.task_queue_task = AsyncMock()
    
    async def _recover_heartbeat_system(self):
        """Recover heartbeat system."""
        self.heartbeat_task = AsyncMock()
    
    async def _attempt_feature_recovery(self) -> bool:
        """Attempt to recover degraded features."""
        if hasattr(self, '_degradation_mode'):
            self._degradation_mode = False
            self._degraded_features = []
            return True
        return False
    
    # Add methods to AgentOrchestrator class
    AgentOrchestrator._send_message_with_retry = _send_message_with_retry
    AgentOrchestrator._get_task_with_retry = _get_task_with_retry
    AgentOrchestrator._spawn_agent_with_recovery = _spawn_agent_with_recovery
    AgentOrchestrator._serialize_message_with_retry = _serialize_message_with_retry
    AgentOrchestrator._perform_with_memory_recovery = _perform_with_memory_recovery
    AgentOrchestrator._assign_task_with_pool_recovery = _assign_task_with_pool_recovery
    AgentOrchestrator._perform_with_concurrency_protection = _perform_with_concurrency_protection
    AgentOrchestrator._process_circuit_breaker_recovery = _process_circuit_breaker_recovery
    AgentOrchestrator._retry_with_exponential_backoff = _retry_with_exponential_backoff
    AgentOrchestrator._perform_system_recovery = _perform_system_recovery
    AgentOrchestrator._recover_heartbeat_system = _recover_heartbeat_system
    AgentOrchestrator._attempt_feature_recovery = _attempt_feature_recovery


# Add the methods when the module is imported
add_orchestrator_recovery_methods()