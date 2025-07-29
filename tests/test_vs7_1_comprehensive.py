"""
Comprehensive Test Suite for VS 7.1: Sleep/Wake API with Checkpointing

Tests all components of the VS 7.1 implementation:
- Atomic checkpoint creation with <5s performance
- Recovery operations with <10s restoration
- Secure API endpoints with <2s response time
- State management with Redis/PostgreSQL integration
- Observability and monitoring integration
"""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.core.checkpoint_manager import CheckpointManager, get_checkpoint_manager
from app.core.recovery_manager import RecoveryManager, get_recovery_manager
from app.core.enhanced_state_manager import EnhancedStateManager, get_enhanced_state_manager
from app.api.v1.sleep_wake_vs7_1 import router as vs71_router
from app.observability.vs7_1_hooks import VS71ObservabilityHooks, get_vs71_observability_hooks
from app.models.sleep_wake import CheckpointType, SleepState
from app.models.agent import Agent
from app.main import app


class TestVS71AtomicCheckpointing:
    """Test atomic checkpoint creation with performance requirements."""
    
    @pytest.fixture
    async def checkpoint_manager(self):
        """Create checkpoint manager for testing."""
        manager = CheckpointManager()
        # Override settings for testing
        manager.target_creation_time_ms = 5000
        manager.enable_atomic_operations = True
        manager.enable_distributed_locking = True
        return manager
    
    @pytest.fixture
    async def test_agent(self, db_session):
        """Create test agent."""
        agent = Agent(
            id=uuid4(),
            name="test-agent",
            current_sleep_state=SleepState.AWAKE
        )
        db_session.add(agent)
        await db_session.commit()
        await db_session.refresh(agent)
        return agent
    
    @pytest.mark.asyncio
    async def test_atomic_checkpoint_creation_performance(self, checkpoint_manager, test_agent):
        """Test that atomic checkpoint creation meets <5s performance target."""
        start_time = time.time()
        
        checkpoint = await checkpoint_manager.create_atomic_checkpoint(
            agent_id=test_agent.id,
            checkpoint_type=CheckpointType.MANUAL,
            metadata={"test": "performance"},
            idempotency_key=f"test_{int(time.time())}"
        )
        
        creation_time_ms = (time.time() - start_time) * 1000
        
        assert checkpoint is not None
        assert creation_time_ms < 5000, f"Checkpoint creation took {creation_time_ms:.0f}ms, exceeds 5s target"
        assert checkpoint.is_valid
        assert checkpoint.agent_id == test_agent.id
        
        # Verify performance metadata
        perf_metrics = checkpoint.checkpoint_metadata.get("performance_metrics", {})
        assert perf_metrics.get("meets_target", False)
        assert perf_metrics.get("total_creation_time_ms") < 5000
    
    @pytest.mark.asyncio
    async def test_distributed_locking(self, checkpoint_manager, test_agent):
        """Test distributed locking prevents concurrent checkpoint creation."""
        
        # Start two concurrent checkpoint operations
        task1 = asyncio.create_task(checkpoint_manager.create_atomic_checkpoint(
            agent_id=test_agent.id,
            checkpoint_type=CheckpointType.MANUAL,
            metadata={"concurrent": "task1"}
        ))
        
        task2 = asyncio.create_task(checkpoint_manager.create_atomic_checkpoint(
            agent_id=test_agent.id,
            checkpoint_type=CheckpointType.MANUAL,
            metadata={"concurrent": "task2"}
        ))
        
        # Wait for both tasks
        results = await asyncio.gather(task1, task2, return_exceptions=True)
        
        # One should succeed, one should fail (return None due to lock failure)
        successful_checkpoints = [r for r in results if r is not None and not isinstance(r, Exception)]
        assert len(successful_checkpoints) == 1, "Distributed locking should allow only one concurrent checkpoint"
    
    @pytest.mark.asyncio
    async def test_idempotency_key_handling(self, checkpoint_manager, test_agent):
        """Test idempotency key prevents duplicate checkpoints."""
        idempotency_key = f"idempotent_test_{int(time.time())}"
        
        # Create first checkpoint
        checkpoint1 = await checkpoint_manager.create_atomic_checkpoint(
            agent_id=test_agent.id,
            checkpoint_type=CheckpointType.MANUAL,
            idempotency_key=idempotency_key
        )
        
        # Create second checkpoint with same idempotency key
        checkpoint2 = await checkpoint_manager.create_atomic_checkpoint(
            agent_id=test_agent.id,
            checkpoint_type=CheckpointType.MANUAL,
            idempotency_key=idempotency_key
        )
        
        assert checkpoint1 is not None
        assert checkpoint2 is not None
        assert checkpoint1.id == checkpoint2.id, "Same idempotency key should return same checkpoint"
    
    @pytest.mark.asyncio
    async def test_parallel_state_collection(self, checkpoint_manager, test_agent):
        """Test parallel state collection improves performance."""
        # Enable parallel collection
        checkpoint_manager.parallel_state_collection = True
        
        start_time = time.time()
        state_data = await checkpoint_manager._collect_state_data_parallel(test_agent.id)
        parallel_time = time.time() - start_time
        
        # Disable parallel collection for comparison
        checkpoint_manager.parallel_state_collection = False
        
        start_time = time.time()
        sequential_data = await checkpoint_manager._collect_state_data(test_agent.id)
        sequential_time = time.time() - start_time
        
        # Verify data completeness
        assert "redis_offsets" in state_data
        assert "agent_states" in state_data
        assert "timestamp" in state_data
        assert state_data["checkpoint_version"] == "1.1"
        
        # Parallel should be faster or similar (within 20% margin)
        assert parallel_time <= sequential_time * 1.2, "Parallel collection should not be significantly slower"
    
    @pytest.mark.asyncio
    async def test_checkpoint_validation_integrity(self, checkpoint_manager, test_agent):
        """Test checkpoint validation ensures 100% data integrity."""
        checkpoint = await checkpoint_manager.create_atomic_checkpoint(
            agent_id=test_agent.id,
            checkpoint_type=CheckpointType.MANUAL
        )
        
        assert checkpoint is not None
        
        # Validate the checkpoint
        is_valid, errors = await checkpoint_manager.validate_checkpoint(checkpoint.id)
        
        assert is_valid, f"Checkpoint validation failed: {errors}"
        assert len(errors) == 0
        assert checkpoint.sha256 is not None
        assert len(checkpoint.sha256) == 64  # SHA-256 hex length


class TestVS71FastRecovery:
    """Test fast recovery with <10s restoration capability."""
    
    @pytest.fixture
    async def recovery_manager(self):
        """Create recovery manager for testing."""
        manager = RecoveryManager()
        # Override settings for testing
        manager.target_recovery_time_ms = 10000
        manager.enable_parallel_validation = True
        manager.enable_fast_health_checks = True
        manager.enable_recovery_caching = True
        return manager
    
    @pytest.fixture
    async def test_checkpoint(self, checkpoint_manager, test_agent):
        """Create test checkpoint for recovery."""
        checkpoint = await checkpoint_manager.create_atomic_checkpoint(
            agent_id=test_agent.id,
            checkpoint_type=CheckpointType.PRE_SLEEP
        )
        return checkpoint
    
    @pytest.mark.asyncio
    async def test_fast_recovery_performance(self, recovery_manager, test_agent, test_checkpoint):
        """Test that fast recovery meets <10s restoration target."""
        start_time = time.time()
        
        success, details = await recovery_manager.fast_recovery_with_caching(
            agent_id=test_agent.id,
            checkpoint_id=test_checkpoint.id
        )
        
        recovery_time_ms = (time.time() - start_time) * 1000
        
        assert success, f"Fast recovery failed: {details}"
        assert recovery_time_ms < 10000, f"Recovery took {recovery_time_ms:.0f}ms, exceeds 10s target"
        assert details.get("meets_target", False)
    
    @pytest.mark.asyncio
    async def test_parallel_recovery_validation(self, recovery_manager, test_agent, test_checkpoint):
        """Test parallel validation improves recovery performance."""
        # Test with parallel validation enabled
        recovery_manager.enable_parallel_validation = True
        
        start_time = time.time()
        success1, details1 = await recovery_manager._parallel_recovery_validation(
            test_agent.id, test_checkpoint
        )
        parallel_time = time.time() - start_time
        
        # Test comprehensive recovery for comparison
        start_time = time.time()
        success2, details2 = await recovery_manager.comprehensive_wake_restoration(
            test_agent.id, test_checkpoint, "standard"
        )
        comprehensive_time = time.time() - start_time
        
        assert success1 and success2
        assert parallel_time < comprehensive_time, "Parallel validation should be faster"
        assert details1.get("parallel_validation", False)
    
    @pytest.mark.asyncio
    async def test_recovery_caching(self, recovery_manager, test_agent, test_checkpoint):
        """Test recovery caching improves repeated operations."""
        # First recovery (no cache)
        start_time = time.time()
        success1, details1 = await recovery_manager.fast_recovery_with_caching(
            test_agent.id, test_checkpoint.id
        )
        first_time = time.time() - start_time
        
        # Reset agent state for second recovery
        await recovery_manager._restore_agent_state(test_agent.id, {"agent_states": {}})
        
        # Second recovery (with cache)
        start_time = time.time()
        success2, details2 = await recovery_manager.fast_recovery_with_caching(
            test_agent.id, test_checkpoint.id
        )
        cached_time = time.time() - start_time
        
        assert success1 and success2
        # Second recovery might use cache (should be faster or similar)
        assert cached_time <= first_time * 1.5, "Cached recovery should not be significantly slower"
    
    @pytest.mark.asyncio
    async def test_fast_health_checks(self, recovery_manager, test_agent):
        """Test fast health checks for parallel execution."""
        start_time = time.time()
        health_result = await recovery_manager._fast_health_check_async(test_agent.id)
        health_check_time = (time.time() - start_time) * 1000
        
        assert health_result.get("fast_mode", False)
        assert health_check_time < 2000, f"Fast health check took {health_check_time:.0f}ms, should be <2s"
        assert "checks" in health_result
        assert health_result.get("passed", False)
    
    @pytest.mark.asyncio
    async def test_recovery_fallback_logic(self, recovery_manager, test_agent):
        """Test recovery fallback with multiple checkpoint generations."""
        # Mock multiple checkpoints for fallback
        with patch.object(recovery_manager, '_get_recovery_checkpoints') as mock_get_checkpoints:
            mock_checkpoint1 = MagicMock()
            mock_checkpoint1.id = uuid4()
            mock_checkpoint2 = MagicMock()
            mock_checkpoint2.id = uuid4()
            
            mock_get_checkpoints.return_value = (mock_checkpoint1, [mock_checkpoint2])
            
            # Mock first checkpoint failure, second success
            with patch.object(recovery_manager.checkpoint_manager, 'restore_checkpoint') as mock_restore:
                mock_restore.side_effect = [(False, {}), (True, {"agent_states": {}})]
                
                success, checkpoint = await recovery_manager._attempt_recovery_with_fallbacks(
                    uuid4(), mock_checkpoint1, [mock_checkpoint2], test_agent.id
                )
                
                assert success
                assert checkpoint == mock_checkpoint2  # Should use fallback


class TestVS71SecureAPI:
    """Test secure API endpoints with <2s response time."""
    
    @pytest.fixture
    def test_client(self):
        """Create test client."""
        app.include_router(vs71_router)
        return TestClient(app)
    
    @pytest.fixture
    def valid_jwt_token(self):
        """Create valid JWT token for testing."""
        payload = {
            "sub": "test-user",
            "username": "test-user",
            "roles": ["admin"],
            "permissions": ["checkpoint:create", "agent:wake", "system:distributed-sleep", "system:read", "metrics:read"],
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        
        # Use test secret key
        secret_key = "test-secret-key-for-jwt-signing-in-tests-only"
        token = jwt.encode(payload, secret_key, algorithm="HS256")
        return token
    
    @pytest.mark.asyncio
    async def test_create_atomic_checkpoint_api_performance(self, test_client, valid_jwt_token, test_agent):
        """Test atomic checkpoint creation API meets <2s response time."""
        headers = {"Authorization": f"Bearer {valid_jwt_token}"}
        
        request_data = {
            "agent_id": str(test_agent.id),
            "checkpoint_type": "manual",
            "metadata": {"api_test": True},
            "force_creation": True
        }
        
        start_time = time.time()
        response = test_client.post(
            "/api/v1/sleep-wake/vs7.1/checkpoint/create",
            json=request_data,
            headers=headers
        )
        api_response_time = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        assert api_response_time < 2000, f"API response took {api_response_time:.0f}ms, exceeds 2s target"
        
        response_data = response.json()
        assert response_data["success"]
        assert response_data["checkpoint_id"] is not None
        assert response_data["performance_metrics"]["response_time_ms"] < 2000
    
    @pytest.mark.asyncio
    async def test_wake_agent_api_performance(self, test_client, valid_jwt_token, test_agent):
        """Test wake agent API meets response time requirements."""
        headers = {"Authorization": f"Bearer {valid_jwt_token}"}
        
        # First put agent to sleep
        await test_client.post(
            "/api/v1/sleep-wake/vs7.1/agent/sleep",
            json={"agent_id": str(test_agent.id)},
            headers=headers
        )
        
        request_data = {
            "agent_id": str(test_agent.id),
            "validation_level": "minimal",
            "recovery_mode": False
        }
        
        start_time = time.time()
        response = test_client.post(
            "/api/v1/sleep-wake/vs7.1/agent/wake",
            json=request_data,
            headers=headers
        )
        api_response_time = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        # Wake operations can take up to 10s, but API response should be quick
        assert api_response_time < 15000, f"Wake API response took {api_response_time:.0f}ms"
        
        response_data = response.json()
        assert response_data["success"]
        assert response_data["agent_id"] == str(test_agent.id)
    
    @pytest.mark.asyncio
    async def test_jwt_authentication_validation(self, test_client):
        """Test JWT authentication and authorization."""
        # Test without token
        response = test_client.post("/api/v1/sleep-wake/vs7.1/checkpoint/create")
        assert response.status_code == 403  # Forbidden due to missing token
        
        # Test with invalid token
        headers = {"Authorization": "Bearer invalid-token"}
        response = test_client.post(
            "/api/v1/sleep-wake/vs7.1/checkpoint/create",
            headers=headers
        )
        assert response.status_code == 401  # Unauthorized
        
        # Test with valid token but insufficient permissions
        payload = {
            "sub": "limited-user",
            "username": "limited-user",
            "roles": ["user"],
            "permissions": ["system:read"],  # Missing checkpoint:create
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        
        secret_key = "test-secret-key-for-jwt-signing-in-tests-only"
        limited_token = jwt.encode(payload, secret_key, algorithm="HS256")
        headers = {"Authorization": f"Bearer {limited_token}"}
        
        response = test_client.post(
            "/api/v1/sleep-wake/vs7.1/checkpoint/create",
            headers=headers
        )
        assert response.status_code == 403  # Forbidden due to insufficient permissions
    
    @pytest.mark.asyncio
    async def test_distributed_sleep_coordination(self, test_client, valid_jwt_token, db_session):
        """Test distributed sleep operations with coordination."""
        # Create multiple test agents
        agents = []
        for i in range(3):
            agent = Agent(
                id=uuid4(),
                name=f"test-agent-{i}",
                current_sleep_state=SleepState.AWAKE
            )
            db_session.add(agent)
            agents.append(agent)
        
        await db_session.commit()
        
        headers = {"Authorization": f"Bearer {valid_jwt_token}"}
        
        request_data = {
            "agent_ids": [str(agent.id) for agent in agents],
            "coordination_strategy": "parallel",
            "max_concurrent": 2,
            "rollback_on_failure": True,
            "timeout_seconds": 300
        }
        
        start_time = time.time()
        response = test_client.post(
            "/api/v1/sleep-wake/vs7.1/system/distributed-sleep",
            json=request_data,
            headers=headers
        )
        api_response_time = (time.time() - start_time) * 1000
        
        assert response.status_code == 200
        assert api_response_time < 20000, f"Distributed sleep API took {api_response_time:.0f}ms"
        
        response_data = response.json()
        assert response_data["total_agents"] == 3
        assert response_data["coordination_strategy"] == "parallel"
        assert "results" in response_data


class TestVS71StateManagement:
    """Test enhanced state management with Redis/PostgreSQL integration."""
    
    @pytest.fixture
    async def state_manager(self):
        """Create enhanced state manager for testing."""
        manager = EnhancedStateManager()
        await manager.initialize()
        return manager
    
    @pytest.mark.asyncio
    async def test_hybrid_state_access_performance(self, state_manager, test_agent):
        """Test hybrid Redis/PostgreSQL state access performance."""
        # First access (cache miss, should hit PostgreSQL)
        start_time = time.time()
        state1 = await state_manager.get_agent_state(test_agent.id, use_cache=True)
        first_access_time = (time.time() - start_time) * 1000
        
        # Second access (cache hit, should be faster)
        start_time = time.time()
        state2 = await state_manager.get_agent_state(test_agent.id, use_cache=True)
        cached_access_time = (time.time() - start_time) * 1000
        
        assert state1 is not None
        assert state2 is not None
        assert state1["id"] == state2["id"]
        
        # Cached access should be significantly faster
        assert cached_access_time < first_access_time * 0.5, "Cached access should be much faster"
        assert cached_access_time < 10, f"Cached access took {cached_access_time:.1f}ms, should be <10ms"
    
    @pytest.mark.asyncio
    async def test_write_through_caching(self, state_manager, test_agent):
        """Test write-through caching maintains consistency."""
        # Update state
        state_updates = {
            "current_sleep_state": "SLEEPING",
            "last_sleep_time": datetime.utcnow().isoformat()
        }
        
        success = await state_manager.set_agent_state(
            test_agent.id, state_updates, persist_to_db=True
        )
        assert success
        
        # Verify state from cache
        cached_state = await state_manager.get_agent_state(test_agent.id, use_cache=True)
        assert cached_state["current_sleep_state"] == "SLEEPING"
        
        # Verify state from database (bypass cache)
        db_state = await state_manager.get_agent_state(test_agent.id, use_cache=False)
        assert db_state["current_sleep_state"] == "SLEEPING"
        
        # Both should match
        assert cached_state["current_sleep_state"] == db_state["current_sleep_state"]
    
    @pytest.mark.asyncio
    async def test_batch_operations_efficiency(self, state_manager, db_session):
        """Test batch operations improve efficiency."""
        # Create multiple agents
        agents = []
        for i in range(10):
            agent = Agent(
                id=uuid4(),
                name=f"batch-test-agent-{i}",
                current_sleep_state=SleepState.AWAKE
            )
            db_session.add(agent)
            agents.append(agent)
        
        await db_session.commit()
        
        agent_ids = [agent.id for agent in agents]
        
        # Test batch get performance
        start_time = time.time()
        batch_results = await state_manager.batch_get_agent_states(agent_ids)
        batch_time = time.time() - start_time
        
        # Test individual gets for comparison
        start_time = time.time()
        individual_results = {}
        for agent_id in agent_ids:
            state = await state_manager.get_agent_state(agent_id)
            if state:
                individual_results[agent_id] = state
        individual_time = time.time() - start_time
        
        assert len(batch_results) == len(individual_results)
        assert batch_time < individual_time, "Batch operations should be faster"
    
    @pytest.mark.asyncio
    async def test_state_consistency_validation(self, state_manager, test_agent):
        """Test state consistency validation between Redis and PostgreSQL."""
        # Ensure state exists in both systems
        await state_manager.get_agent_state(test_agent.id, use_cache=True)
        
        # Validate consistency
        consistency_result = await state_manager.validate_state_consistency(test_agent.id)
        assert consistency_result["consistent"]
        
        # Simulate inconsistency (corrupt cache)
        redis_client = await state_manager.get_redis()
        cache_key = f"{state_manager.redis_key_prefix}agent:{test_agent.id}"
        await redis_client.set(cache_key, json.dumps({
            "id": str(test_agent.id),
            "current_sleep_state": "INCONSISTENT_STATE"
        }))
        
        # Validate consistency again
        consistency_result = await state_manager.validate_state_consistency(test_agent.id)
        assert not consistency_result["consistent"]
        assert "inconsistencies" in consistency_result
        
        # Repair consistency
        repair_success = await state_manager.repair_state_consistency(test_agent.id)
        assert repair_success
        
        # Verify consistency is restored
        consistency_result = await state_manager.validate_state_consistency(test_agent.id)
        assert consistency_result["consistent"]


class TestVS71ObservabilityIntegration:
    """Test observability and monitoring integration."""
    
    @pytest.fixture
    async def observability_hooks(self):
        """Create VS 7.1 observability hooks for testing."""
        hooks = VS71ObservabilityHooks()
        await hooks.initialize()
        return hooks
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, observability_hooks):
        """Test performance metrics collection and reporting."""
        # Get dashboard data
        dashboard_data = await observability_hooks.get_vs71_dashboard_data()
        
        assert "timestamp" in dashboard_data
        assert "summary" in dashboard_data
        assert "performance" in dashboard_data
        assert "thresholds" in dashboard_data
        
        # Verify performance thresholds
        thresholds = dashboard_data["thresholds"]
        assert thresholds["checkpoint_creation_ms"] == 5000
        assert thresholds["recovery_time_ms"] == 10000
        assert thresholds["api_response_ms"] == 2000
    
    @pytest.mark.asyncio
    async def test_alerting_on_performance_degradation(self, observability_hooks):
        """Test alerting when performance thresholds are exceeded."""
        # Mock alert manager
        with patch.object(observability_hooks.alert_manager, 'send_alert') as mock_alert:
            # Simulate slow checkpoint creation
            await observability_hooks._checkpoint_metrics.update({
                "test_checkpoint": {
                    "start_time": time.time() - 10,  # 10 seconds ago
                    "agent_id": uuid4(),
                    "checkpoint_type": "manual",
                    "status": "completed"
                }
            })
            
            # Trigger completion event with slow duration
            from app.observability.hooks import HookEvent
            event = HookEvent(
                event_type="checkpoint.creation.completed",
                data={
                    "checkpoint_id": "test_checkpoint",
                    "success": True
                }
            )
            
            # This should trigger an alert due to slow creation time
            mock_alert.assert_called()  # Verify alert was sent
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_monitoring(self, observability_hooks):
        """Test circuit breaker state monitoring and alerting."""
        # Mock circuit breaker state change
        with patch.object(observability_hooks.alert_manager, 'send_alert') as mock_alert:
            from app.observability.hooks import HookEvent
            
            # Simulate circuit breaker opening
            event = HookEvent(
                event_type="circuit_breaker.state_changed",
                data={
                    "circuit_name": "checkpoint_circuit_breaker",
                    "old_state": "closed",
                    "new_state": "open"
                }
            )
            
            # This should trigger a critical alert
            mock_alert.assert_called_with(
                "circuit_breaker_opened",
                "Circuit breaker checkpoint_circuit_breaker opened (was closed)",
                severity="critical",
                metadata={
                    "circuit_name": "checkpoint_circuit_breaker",
                    "old_state": "closed",
                    "new_state": "open"
                }
            )


class TestVS71ProductionReadiness:
    """Test production readiness and compliance."""
    
    @pytest.mark.asyncio
    async def test_all_performance_targets_met(self, checkpoint_manager, recovery_manager, test_agent):
        """Comprehensive test that all performance targets are met."""
        results = {}
        
        # Test checkpoint creation performance
        start_time = time.time()
        checkpoint = await checkpoint_manager.create_atomic_checkpoint(
            agent_id=test_agent.id,
            checkpoint_type=CheckpointType.MANUAL
        )
        checkpoint_time = (time.time() - start_time) * 1000
        results["checkpoint_creation_ms"] = checkpoint_time
        
        # Test recovery performance
        start_time = time.time()
        success, details = await recovery_manager.fast_recovery_with_caching(
            agent_id=test_agent.id,
            checkpoint_id=checkpoint.id
        )
        recovery_time = (time.time() - start_time) * 1000
        results["recovery_time_ms"] = recovery_time
        
        # Verify all targets are met
        assert checkpoint_time < 5000, f"Checkpoint creation: {checkpoint_time:.0f}ms > 5000ms"
        assert recovery_time < 10000, f"Recovery time: {recovery_time:.0f}ms > 10000ms"
        assert success, "Recovery operation failed"
        
        # Log results for reporting
        print(f"Performance Results: {results}")
    
    @pytest.mark.asyncio
    async def test_data_integrity_guarantees(self, checkpoint_manager, recovery_manager, test_agent):
        """Test 100% data integrity is maintained."""
        # Create checkpoint
        checkpoint = await checkpoint_manager.create_atomic_checkpoint(
            agent_id=test_agent.id,
            checkpoint_type=CheckpointType.MANUAL,
            metadata={"integrity_test": True}
        )
        
        assert checkpoint is not None
        assert checkpoint.is_valid
        
        # Validate integrity
        is_valid, errors = await checkpoint_manager.validate_checkpoint(checkpoint.id)
        assert is_valid, f"Data integrity check failed: {errors}"
        
        # Test recovery maintains integrity
        success, state_data = await checkpoint_manager.restore_checkpoint(checkpoint.id)
        assert success, "Checkpoint restoration failed"
        assert "timestamp" in state_data
        assert "checkpoint_version" in state_data
    
    @pytest.mark.asyncio
    async def test_scalability_limits(self, state_manager, db_session):
        """Test system handles expected load levels."""
        # Create many agents for load testing
        agents = []
        for i in range(50):  # Test with 50 agents
            agent = Agent(
                id=uuid4(),
                name=f"load-test-agent-{i}",
                current_sleep_state=SleepState.AWAKE
            )
            db_session.add(agent)
            agents.append(agent)
        
        await db_session.commit()
        
        # Test batch operations at scale
        agent_ids = [agent.id for agent in agents]
        
        start_time = time.time()
        results = await state_manager.batch_get_agent_states(agent_ids)
        batch_time = time.time() - start_time
        
        assert len(results) == len(agents)
        assert batch_time < 5.0, f"Batch operation took {batch_time:.2f}s for {len(agents)} agents"
        
        # Test concurrent state updates
        async def update_agent_state(agent_id):
            return await state_manager.set_agent_state(
                agent_id,
                {"last_wake_time": datetime.utcnow().isoformat()}
            )
        
        start_time = time.time()
        update_tasks = [update_agent_state(agent.id) for agent in agents[:10]]  # Test 10 concurrent
        update_results = await asyncio.gather(*update_tasks)
        concurrent_time = time.time() - start_time
        
        assert all(update_results), "Some concurrent updates failed"
        assert concurrent_time < 2.0, f"Concurrent updates took {concurrent_time:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])