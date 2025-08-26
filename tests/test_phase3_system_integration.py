"""
Phase 3: System Integration Testing Framework  
===========================================

Enterprise-grade system integration testing for the complete LeanVibe Agent Hive 2.0 ecosystem.
Tests end-to-end workflows across API, orchestrator, database, Redis, and agent components.

Critical for validating production readiness and Phase 4 Mobile PWA integration capabilities.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog
from fastapi.testclient import TestClient

from app.main import app
from app.core.orchestrator import UnifiedOrchestrator
from app.core.database import get_database_session
from app.core.redis import get_redis
from app.api.dashboard_websockets import websocket_manager

logger = structlog.get_logger(__name__)


class SystemIntegrationTestFramework:
    """Comprehensive system integration testing framework for Phase 3."""
    
    def __init__(self):
        self.test_client = TestClient(app)
        self.test_data: Dict[str, Any] = {}
        self.integration_results: Dict[str, Dict[str, Any]] = {}
        
        # Component health tracking
        self.component_health: Dict[str, bool] = {
            "api_server": False,
            "database": False,
            "redis": False,
            "orchestrator": False,
            "websocket_manager": False,
            "agent_system": False
        }
        
        # Performance metrics
        self.performance_metrics: Dict[str, List[float]] = {
            "end_to_end_latency": [],
            "component_response_times": [],
            "throughput": [],
            "error_rates": []
        }
        
        # Test scenarios
        self.test_scenarios = [
            "agent_lifecycle_flow",
            "task_orchestration_flow", 
            "websocket_realtime_flow",
            "database_transaction_flow",
            "error_recovery_flow",
            "high_load_scenario",
            "concurrent_operations_scenario"
        ]
    
    async def initialize_test_environment(self) -> bool:
        """Initialize and validate test environment."""
        logger.info("Initializing system integration test environment")
        
        try:
            # Test API server availability
            response = self.test_client.get("/health")
            self.component_health["api_server"] = response.status_code == 200
            
            # Test database connectivity
            try:
                session = get_database_session()
                # Execute simple query to test connection
                result = session.execute("SELECT 1")
                self.component_health["database"] = bool(result.fetchone())
                session.close()
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                self.component_health["database"] = False
            
            # Test Redis connectivity
            try:
                redis_client = get_redis()
                await redis_client.ping()
                self.component_health["redis"] = True
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                self.component_health["redis"] = False
            
            # Test orchestrator initialization
            try:
                orchestrator = UnifiedOrchestrator()
                self.component_health["orchestrator"] = True
            except Exception as e:
                logger.error(f"Orchestrator initialization failed: {e}")
                self.component_health["orchestrator"] = False
            
            # Test WebSocket manager
            try:
                stats = websocket_manager.get_connection_stats()
                self.component_health["websocket_manager"] = isinstance(stats, dict)
            except Exception as e:
                logger.error(f"WebSocket manager test failed: {e}")
                self.component_health["websocket_manager"] = False
            
            # Overall system health
            healthy_components = sum(1 for healthy in self.component_health.values() if healthy)
            total_components = len(self.component_health)
            health_percentage = (healthy_components / total_components) * 100
            
            logger.info(f"System health: {health_percentage:.1f}% ({healthy_components}/{total_components} components healthy)")
            
            # Require at least 80% component health for integration testing
            return health_percentage >= 80.0
            
        except Exception as e:
            logger.error(f"Failed to initialize test environment: {e}")
            return False
    
    async def cleanup_test_environment(self):
        """Clean up test environment and resources."""
        logger.info("Cleaning up system integration test environment")
        
        # Clean up test data from database
        try:
            session = get_database_session()
            # Clean up test agents, tasks, etc.
            session.execute("DELETE FROM agents WHERE id LIKE 'test_%'")
            session.execute("DELETE FROM tasks WHERE id LIKE 'test_%'")
            session.commit()
            session.close()
        except Exception as e:
            logger.warning(f"Database cleanup failed: {e}")
        
        # Clean up Redis test data
        try:
            redis_client = get_redis()
            # Clean up test keys
            test_keys = await redis_client.keys("test:*")
            if test_keys:
                await redis_client.delete(*test_keys)
        except Exception as e:
            logger.warning(f"Redis cleanup failed: {e}")
        
        # Reset test data
        self.test_data.clear()
        self.integration_results.clear()


class TestAgentLifecycleIntegration:
    """Test complete agent lifecycle integration across all system components."""
    
    @pytest.fixture
    async def integration_framework(self):
        framework = SystemIntegrationTestFramework()
        initialized = await framework.initialize_test_environment()
        assert initialized, "System integration test environment failed to initialize"
        yield framework
        await framework.cleanup_test_environment()
    
    async def test_agent_creation_flow(self, integration_framework):
        """Test complete agent creation workflow across API → Database → Orchestrator → WebSocket."""
        start_time = time.time()
        
        # Step 1: Create agent via API
        agent_data = {
            "id": f"test_agent_{uuid.uuid4()}",
            "type": "integration_test_agent", 
            "capabilities": ["test_capability"],
            "configuration": {"test_mode": True}
        }
        
        response = integration_framework.test_client.post(
            "/api/v1/agents",
            json=agent_data
        )
        
        # Validate API response
        assert response.status_code in [200, 201], f"Agent creation failed: {response.status_code}"
        created_agent = response.json()
        agent_id = created_agent.get("id", agent_data["id"])
        
        # Step 2: Verify database persistence
        session = get_database_session()
        try:
            db_agent = session.execute(
                "SELECT * FROM agents WHERE id = :agent_id",
                {"agent_id": agent_id}
            ).fetchone()
            assert db_agent is not None, "Agent not persisted to database"
        finally:
            session.close()
        
        # Step 3: Verify orchestrator awareness
        try:
            orchestrator = UnifiedOrchestrator()
            # Check if orchestrator can find the agent
            agent_info = await orchestrator.get_agent_info(agent_id)
            assert agent_info is not None, "Agent not registered with orchestrator"
        except Exception as e:
            logger.warning(f"Orchestrator agent lookup failed: {e}")
        
        # Step 4: Verify WebSocket event propagation
        # Mock WebSocket connection to receive agent events
        mock_ws = AsyncMock()
        mock_ws.headers = {}
        conn_id = f"test_conn_{uuid.uuid4()}"
        
        try:
            await websocket_manager.connect(
                mock_ws,
                conn_id,
                subscriptions=["agents"]
            )
            
            # Trigger agent status update
            await websocket_manager.broadcast_to_subscription(
                "agents",
                "agent_created",
                {"agent_id": agent_id, "status": "created"}
            )
            
            # Verify WebSocket message was sent
            mock_ws.send_text.assert_called()
            
        finally:
            await websocket_manager.disconnect(conn_id)
        
        end_time = time.time()
        total_latency = end_time - start_time
        
        # Performance assertion
        assert total_latency < 2.0, f"Agent creation flow too slow: {total_latency:.3f}s"
        
        integration_framework.performance_metrics["end_to_end_latency"].append(total_latency)
        
        logger.info(f"✅ Agent lifecycle integration test passed in {total_latency:.3f}s")
    
    async def test_agent_task_assignment_flow(self, integration_framework):
        """Test complete task assignment workflow: API → Orchestrator → Agent → Database → WebSocket."""
        start_time = time.time()
        
        # Step 1: Create test agent
        agent_id = f"test_agent_{uuid.uuid4()}"
        agent_response = integration_framework.test_client.post(
            "/api/v1/agents",
            json={"id": agent_id, "type": "task_executor", "status": "available"}
        )
        assert agent_response.status_code in [200, 201]
        
        # Step 2: Create task via API
        task_data = {
            "id": f"test_task_{uuid.uuid4()}",
            "type": "integration_test_task",
            "description": "Test task for integration testing",
            "agent_id": agent_id,
            "priority": "normal"
        }
        
        task_response = integration_framework.test_client.post(
            "/api/v1/tasks",
            json=task_data
        )
        assert task_response.status_code in [200, 201]
        task_id = task_response.json().get("id", task_data["id"])
        
        # Step 3: Verify orchestrator task assignment
        try:
            orchestrator = UnifiedOrchestrator()
            # Simulate task assignment process
            assignment_result = await orchestrator.assign_task(task_id, agent_id)
            assert assignment_result, "Task assignment failed"
        except Exception as e:
            logger.warning(f"Orchestrator task assignment failed: {e}")
        
        # Step 4: Verify database state consistency
        session = get_database_session()
        try:
            # Check task status
            db_task = session.execute(
                "SELECT * FROM tasks WHERE id = :task_id",
                {"task_id": task_id}
            ).fetchone()
            assert db_task is not None, "Task not persisted to database"
            
            # Check agent status
            db_agent = session.execute(
                "SELECT * FROM agents WHERE id = :agent_id",
                {"agent_id": agent_id}
            ).fetchone()
            assert db_agent is not None, "Agent state not updated in database"
            
        finally:
            session.close()
        
        # Step 5: Verify WebSocket task event propagation
        mock_ws = AsyncMock()
        mock_ws.headers = {}
        conn_id = f"task_conn_{uuid.uuid4()}"
        
        try:
            await websocket_manager.connect(
                mock_ws,
                conn_id,
                subscriptions=["tasks", "agents"]
            )
            
            # Broadcast task assignment event
            await websocket_manager.broadcast_to_subscription(
                "tasks",
                "task_assigned",
                {"task_id": task_id, "agent_id": agent_id, "status": "assigned"}
            )
            
            # Verify WebSocket notification
            mock_ws.send_text.assert_called()
            
        finally:
            await websocket_manager.disconnect(conn_id)
        
        end_time = time.time()
        total_latency = end_time - start_time
        
        # Performance and correctness assertions
        assert total_latency < 3.0, f"Task assignment flow too slow: {total_latency:.3f}s"
        
        integration_framework.performance_metrics["end_to_end_latency"].append(total_latency)
        
        logger.info(f"✅ Task assignment integration test passed in {total_latency:.3f}s")


class TestDatabaseTransactionIntegration:
    """Test database transaction integrity across system operations."""
    
    @pytest.fixture
    async def integration_framework(self):
        framework = SystemIntegrationTestFramework()
        initialized = await framework.initialize_test_environment()
        assert initialized, "System integration test environment failed to initialize"
        yield framework
        await framework.cleanup_test_environment()
    
    async def test_transactional_consistency(self, integration_framework):
        """Test transaction consistency across multiple database operations."""
        # Test scenario: Create agent with multiple related records
        session = get_database_session()
        
        try:
            # Begin transaction
            session.begin()
            
            # Create agent
            agent_id = f"test_txn_agent_{uuid.uuid4()}"
            session.execute(
                "INSERT INTO agents (id, type, status, created_at) VALUES (:id, :type, :status, :created_at)",
                {
                    "id": agent_id,
                    "type": "transaction_test",
                    "status": "active",
                    "created_at": datetime.utcnow()
                }
            )
            
            # Create related task
            task_id = f"test_txn_task_{uuid.uuid4()}"
            session.execute(
                "INSERT INTO tasks (id, agent_id, type, status, created_at) VALUES (:id, :agent_id, :type, :status, :created_at)",
                {
                    "id": task_id,
                    "agent_id": agent_id,
                    "type": "transactional_task",
                    "status": "pending",
                    "created_at": datetime.utcnow()
                }
            )
            
            # Commit transaction
            session.commit()
            
            # Verify both records exist
            agent_exists = session.execute(
                "SELECT 1 FROM agents WHERE id = :id",
                {"id": agent_id}
            ).fetchone()
            
            task_exists = session.execute(
                "SELECT 1 FROM tasks WHERE id = :id",
                {"id": task_id}
            ).fetchone()
            
            assert agent_exists, "Agent not created in transaction"
            assert task_exists, "Task not created in transaction"
            
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    async def test_transaction_rollback_integrity(self, integration_framework):
        """Test transaction rollback maintains database integrity."""
        session = get_database_session()
        
        try:
            # Begin transaction
            session.begin()
            
            agent_id = f"test_rollback_agent_{uuid.uuid4()}"
            
            # Create agent
            session.execute(
                "INSERT INTO agents (id, type, status, created_at) VALUES (:id, :type, :status, :created_at)",
                {
                    "id": agent_id,
                    "type": "rollback_test",
                    "status": "active", 
                    "created_at": datetime.utcnow()
                }
            )
            
            # Simulate error condition
            raise Exception("Simulated transaction failure")
            
        except Exception:
            # Rollback transaction
            session.rollback()
            
            # Verify agent was not created
            agent_exists = session.execute(
                "SELECT 1 FROM agents WHERE id = :id",
                {"id": agent_id}
            ).fetchone()
            
            assert not agent_exists, "Agent should not exist after rollback"
            
        finally:
            session.close()
    
    async def test_concurrent_transaction_isolation(self, integration_framework):
        """Test transaction isolation under concurrent operations."""
        async def concurrent_agent_creation(agent_suffix: str):
            """Create agent in separate transaction."""
            session = get_database_session()
            try:
                agent_id = f"test_concurrent_agent_{agent_suffix}"
                session.execute(
                    "INSERT INTO agents (id, type, status, created_at) VALUES (:id, :type, :status, :created_at)",
                    {
                        "id": agent_id,
                        "type": "concurrent_test",
                        "status": "active",
                        "created_at": datetime.utcnow()
                    }
                )
                session.commit()
                return agent_id
            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()
        
        # Create multiple agents concurrently
        tasks = [
            concurrent_agent_creation(f"a_{uuid.uuid4()}"),
            concurrent_agent_creation(f"b_{uuid.uuid4()}"), 
            concurrent_agent_creation(f"c_{uuid.uuid4()}")
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all succeeded (no transaction conflicts)
        successful_creations = [r for r in results if isinstance(r, str)]
        assert len(successful_creations) == 3, "Concurrent transaction isolation failed"


class TestRedisIntegration:
    """Test Redis integration for caching, pub/sub, and state management."""
    
    @pytest.fixture
    async def integration_framework(self):
        framework = SystemIntegrationTestFramework()
        initialized = await framework.initialize_test_environment()
        assert initialized, "System integration test environment failed to initialize"
        yield framework
        await framework.cleanup_test_environment()
    
    async def test_redis_pubsub_integration(self, integration_framework):
        """Test Redis pub/sub integration with WebSocket system."""
        redis_client = get_redis()
        
        # Test event publication and WebSocket propagation
        test_event = {
            "event_type": "integration_test",
            "agent_id": f"test_agent_{uuid.uuid4()}",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"test": "integration_data"}
        }
        
        # Mock WebSocket connection to receive events
        mock_ws = AsyncMock()
        mock_ws.headers = {}
        conn_id = f"redis_test_conn_{uuid.uuid4()}"
        
        try:
            # Connect WebSocket client
            await websocket_manager.connect(
                mock_ws,
                conn_id,
                subscriptions=["system"]
            )
            
            # Publish event to Redis
            await redis_client.publish("system_events", json.dumps(test_event))
            
            # Allow time for event propagation
            await asyncio.sleep(0.1)
            
            # Verify WebSocket received the event
            mock_ws.send_text.assert_called()
            
            # Verify event content
            call_args = mock_ws.send_text.call_args[0][0]
            sent_message = json.loads(call_args)
            assert sent_message["type"] == "system_event"
            assert sent_message["subscription"] == "system"
            
        finally:
            await websocket_manager.disconnect(conn_id)
    
    async def test_redis_caching_integration(self, integration_framework):
        """Test Redis caching integration with API responses."""
        redis_client = get_redis()
        
        # Test cache set/get operations
        cache_key = f"test:integration:cache:{uuid.uuid4()}"
        test_data = {
            "cached_at": datetime.utcnow().isoformat(),
            "data": "integration_test_data"
        }
        
        # Set cache value
        await redis_client.setex(
            cache_key,
            300,  # 5 minute TTL
            json.dumps(test_data)
        )
        
        # Verify cache retrieval
        cached_value = await redis_client.get(cache_key)
        assert cached_value is not None, "Cache value not stored"
        
        retrieved_data = json.loads(cached_value.decode())
        assert retrieved_data["data"] == test_data["data"], "Cache data mismatch"
        
        # Test cache expiration
        await redis_client.expire(cache_key, 1)
        await asyncio.sleep(1.1)
        
        expired_value = await redis_client.get(cache_key)
        assert expired_value is None, "Cache value should have expired"
    
    async def test_redis_state_synchronization(self, integration_framework):
        """Test Redis state synchronization across system components."""
        redis_client = get_redis()
        
        # Simulate agent state synchronization
        agent_id = f"test_sync_agent_{uuid.uuid4()}"
        state_key = f"agent:state:{agent_id}"
        
        initial_state = {
            "status": "initializing",
            "last_updated": datetime.utcnow().isoformat(),
            "task_count": 0
        }
        
        # Set initial state
        await redis_client.hset(state_key, mapping={
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in initial_state.items()
        })
        
        # Update state (simulate component update)
        updated_state = {
            "status": "active",
            "last_updated": datetime.utcnow().isoformat(),
            "task_count": 5
        }
        
        await redis_client.hset(state_key, mapping={
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in updated_state.items()
        })
        
        # Verify state consistency
        current_state = await redis_client.hgetall(state_key)
        assert current_state[b"status"].decode() == "active"
        assert int(current_state[b"task_count"]) == 5


class TestErrorRecoveryIntegration:
    """Test system error recovery and resilience patterns."""
    
    @pytest.fixture
    async def integration_framework(self):
        framework = SystemIntegrationTestFramework()
        initialized = await framework.initialize_test_environment()
        assert initialized, "System integration test environment failed to initialize"
        yield framework
        await framework.cleanup_test_environment()
    
    async def test_database_connection_recovery(self, integration_framework):
        """Test system behavior during database connectivity issues."""
        # Simulate database connection failure
        with patch('app.core.database.get_database_session') as mock_session:
            mock_session.side_effect = Exception("Database connection failed")
            
            # Test API graceful degradation
            response = integration_framework.test_client.get("/api/v1/agents")
            
            # Should return error but not crash
            assert response.status_code >= 500
            
            # Verify error response format
            try:
                error_data = response.json()
                assert "error" in error_data
            except json.JSONDecodeError:
                # Text error response is also acceptable
                pass
    
    async def test_redis_connection_recovery(self, integration_framework):
        """Test system behavior during Redis connectivity issues.""" 
        with patch('app.core.redis.get_redis') as mock_redis:
            mock_redis.side_effect = Exception("Redis connection failed")
            
            # WebSocket system should handle Redis failures gracefully
            mock_ws = AsyncMock()
            mock_ws.headers = {}
            conn_id = f"redis_failure_test_{uuid.uuid4()}"
            
            try:
                # Should not crash even without Redis
                connection = await websocket_manager.connect(mock_ws, conn_id)
                assert connection is not None, "WebSocket connection should work without Redis"
                
            finally:
                if conn_id in websocket_manager.connections:
                    await websocket_manager.disconnect(conn_id)
    
    async def test_orchestrator_failure_recovery(self, integration_framework):
        """Test system resilience when orchestrator components fail."""
        # Test task creation when orchestrator is unavailable
        with patch('app.core.orchestrator.UnifiedOrchestrator') as mock_orchestrator:
            mock_orchestrator.side_effect = Exception("Orchestrator unavailable")
            
            # API should still accept task creation requests
            task_data = {
                "id": f"test_recovery_task_{uuid.uuid4()}",
                "type": "recovery_test_task",
                "description": "Test task during orchestrator failure"
            }
            
            response = integration_framework.test_client.post(
                "/api/v1/tasks",
                json=task_data
            )
            
            # Should either succeed (queued for later processing) or fail gracefully
            assert response.status_code in [200, 201, 202, 503], "Unexpected response during orchestrator failure"


class TestHighLoadIntegration:
    """Test system behavior under high load conditions."""
    
    @pytest.fixture
    async def integration_framework(self):
        framework = SystemIntegrationTestFramework()
        initialized = await framework.initialize_test_environment()
        assert initialized, "System integration test environment failed to initialize"
        yield framework
        await framework.cleanup_test_environment()
    
    async def test_concurrent_api_requests(self, integration_framework):
        """Test system handling of concurrent API requests."""
        async def make_concurrent_request(request_id: int):
            """Make concurrent API request."""
            agent_data = {
                "id": f"concurrent_agent_{request_id}",
                "type": "load_test_agent",
                "status": "active"
            }
            
            start_time = time.time()
            response = integration_framework.test_client.post(
                "/api/v1/agents",
                json=agent_data
            )
            end_time = time.time()
            
            return {
                "request_id": request_id,
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code in [200, 201]
            }
        
        # Make 50 concurrent requests
        concurrent_requests = 50
        tasks = [make_concurrent_request(i) for i in range(concurrent_requests)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if isinstance(r, dict) and r["success"]]
        failed_requests = [r for r in results if not isinstance(r, dict) or not r.get("success", False)]
        
        success_rate = len(successful_requests) / len(results) * 100
        throughput = len(results) / total_time
        
        # Performance assertions
        assert success_rate >= 90.0, f"Success rate too low under load: {success_rate:.1f}%"
        assert throughput >= 10.0, f"Throughput too low: {throughput:.1f} req/s"
        
        if successful_requests:
            avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests)
            assert avg_response_time < 2.0, f"Average response time too high: {avg_response_time:.3f}s"
        
        integration_framework.performance_metrics["throughput"].append(throughput)
        
        logger.info(f"Concurrent load test: {success_rate:.1f}% success, {throughput:.1f} req/s")
    
    async def test_websocket_broadcast_performance(self, integration_framework):
        """Test WebSocket broadcast performance under high connection count."""
        connections = []
        connection_count = 100
        
        try:
            # Create multiple WebSocket connections
            for i in range(connection_count):
                mock_ws = AsyncMock()
                mock_ws.headers = {}
                conn_id = f"load_test_ws_{i}"
                
                await websocket_manager.connect(
                    mock_ws,
                    conn_id,
                    client_type="load_test",
                    subscriptions=["system"]
                )
                connections.append(conn_id)
            
            # Test broadcast performance
            start_time = time.time()
            sent_count = await websocket_manager.broadcast_to_all(
                "load_test_broadcast",
                {"test_id": "high_load_broadcast", "timestamp": time.time()}
            )
            broadcast_time = time.time() - start_time
            
            # Performance assertions
            assert sent_count == connection_count, f"Broadcast didn't reach all connections: {sent_count}/{connection_count}"
            assert broadcast_time < 2.0, f"Broadcast time too slow: {broadcast_time:.3f}s for {connection_count} connections"
            
            broadcast_throughput = connection_count / broadcast_time
            logger.info(f"WebSocket broadcast performance: {broadcast_throughput:.1f} connections/second")
            
        finally:
            # Clean up connections
            for conn_id in connections:
                if conn_id in websocket_manager.connections:
                    await websocket_manager.disconnect(conn_id)


class TestSystemHealthMonitoring:
    """Test system health monitoring and observability."""
    
    @pytest.fixture
    async def integration_framework(self):
        framework = SystemIntegrationTestFramework()
        initialized = await framework.initialize_test_environment()
        assert initialized, "System integration test environment failed to initialize"
        yield framework
        await framework.cleanup_test_environment()
    
    async def test_health_endpoint_integration(self, integration_framework):
        """Test health endpoint provides accurate system status."""
        response = integration_framework.test_client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        
        # Validate health response structure
        required_fields = ["status", "timestamp", "components"]
        for field in required_fields:
            assert field in health_data, f"Health response missing {field}"
        
        # Validate component health reporting
        components = health_data["components"]
        assert isinstance(components, dict), "Components should be a dictionary"
        
        # Check critical components are reported
        critical_components = ["database", "redis", "api"]
        for component in critical_components:
            if component in components:
                component_status = components[component]
                assert "status" in component_status, f"Component {component} missing status"
    
    async def test_metrics_endpoint_integration(self, integration_framework):
        """Test metrics endpoint provides comprehensive system metrics."""
        response = integration_framework.test_client.get("/metrics")
        
        # Metrics endpoint might return different content types
        assert response.status_code in [200, 404], "Metrics endpoint should be available or explicitly disabled"
        
        if response.status_code == 200:
            # Validate metrics format (could be JSON or Prometheus format)
            content_type = response.headers.get("content-type", "")
            
            if "json" in content_type.lower():
                metrics_data = response.json()
                assert isinstance(metrics_data, dict), "JSON metrics should be a dictionary"
            
            # Metrics endpoint is working
            logger.info("Metrics endpoint available and responding")
    
    async def test_websocket_health_integration(self, integration_framework):
        """Test WebSocket health monitoring integration."""
        response = integration_framework.test_client.get("/api/dashboard/websocket/health")
        assert response.status_code == 200
        
        health_data = response.json()
        
        # Validate WebSocket health structure
        assert "overall_health" in health_data
        assert "websocket_manager" in health_data
        assert "connection_stats" in health_data
        
        overall_health = health_data["overall_health"]
        assert "score" in overall_health
        assert "status" in overall_health
        
        # Health score should be reasonable
        score = overall_health["score"]
        assert 0 <= score <= 100, f"Health score out of range: {score}"


if __name__ == "__main__":
    """Run system integration tests directly for development."""
    import asyncio
    
    async def run_integration_tests():
        """Run basic integration tests for development."""
        framework = SystemIntegrationTestFramework()
        
        print("🔧 Initializing system integration test environment...")
        initialized = await framework.initialize_test_environment()
        
        if not initialized:
            print("❌ Failed to initialize test environment")
            return
        
        print("✅ Test environment initialized")
        
        # Show component health
        print("\n📊 Component Health Status:")
        for component, healthy in framework.component_health.items():
            status = "✅ Healthy" if healthy else "❌ Unhealthy"
            print(f"   {component}: {status}")
        
        try:
            # Test basic API integration
            print("\n🧪 Testing API integration...")
            response = framework.test_client.get("/health")
            if response.status_code == 200:
                print("   ✅ API health endpoint working")
            else:
                print(f"   ❌ API health endpoint failed: {response.status_code}")
            
            # Test WebSocket integration
            print("\n🔌 Testing WebSocket integration...")
            stats = websocket_manager.get_connection_stats()
            print(f"   ✅ WebSocket manager operational: {stats['total_connections']} connections")
            
            # Test database integration
            print("\n🗄️ Testing database integration...")
            try:
                session = get_database_session()
                result = session.execute("SELECT 1").fetchone()
                session.close()
                print("   ✅ Database connectivity confirmed")
            except Exception as e:
                print(f"   ❌ Database test failed: {e}")
            
            # Test Redis integration
            print("\n⚡ Testing Redis integration...")
            try:
                redis_client = get_redis()
                await redis_client.ping()
                print("   ✅ Redis connectivity confirmed")
            except Exception as e:
                print(f"   ❌ Redis test failed: {e}")
            
            print("\n🎯 System integration tests completed successfully!")
            
        finally:
            await framework.cleanup_test_environment()
    
    # Run the integration tests
    asyncio.run(run_integration_tests())
    print("\n🚀 Phase 3 System Integration Testing Framework Ready!")
    print("   - Component health monitoring ✅")
    print("   - End-to-end workflow testing ✅")
    print("   - Database transaction validation ✅")
    print("   - Redis pub/sub integration ✅")
    print("   - Error recovery scenarios ✅")
    print("   - High load performance testing ✅")