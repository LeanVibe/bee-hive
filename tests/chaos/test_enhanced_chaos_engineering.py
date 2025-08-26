"""
Enhanced Chaos Engineering Test Suite for LeanVibe Agent Hive 2.0

Tests system resilience under various failure scenarios:
- Network partitions and latency
- Database failures and connection issues
- Redis failures and message loss
- Resource exhaustion (memory, CPU, disk)
- Cascading failures and recovery
- Byzantine failures and corruption
"""

import asyncio
import random
import time
import gc
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from unittest.mock import AsyncMock, patch, MagicMock
from contextlib import asynccontextmanager

import pytest
import structlog
from sqlalchemy.exc import OperationalError, DisconnectionError
from sqlalchemy.ext.asyncio import AsyncSession

# Import with fallback for testing isolation
try:
    from app.core.orchestrator import AgentOrchestrator
    from app.core.communication import MessageBroker
    from app.core.database import get_async_session
    from app.core.redis import AgentMessageBroker
    from app.core.health_monitor import HealthMonitor, HealthStatus
    from app.core.circuit_breaker import CircuitBreaker
    from app.core.recovery_manager import RecoveryManager
    from app.models.agent import Agent, AgentStatus
except ImportError:
    # Mock imports for isolated testing
    AgentOrchestrator = AsyncMock
    MessageBroker = AsyncMock
    get_async_session = AsyncMock
    AgentMessageBroker = AsyncMock
    HealthMonitor = MagicMock
    HealthStatus = MagicMock
    CircuitBreaker = MagicMock
    RecoveryManager = AsyncMock
    Agent = MagicMock
    AgentStatus = MagicMock

logger = structlog.get_logger()


class ChaosScenario:
    """Base class for chaos engineering scenarios."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.metrics: Dict[str, Any] = {}
    
    async def setup(self) -> None:
        """Setup the chaos scenario."""
        self.start_time = datetime.utcnow()
        logger.info(f"Starting chaos scenario: {self.name}")
    
    async def execute(self) -> None:
        """Execute the chaos scenario."""
        raise NotImplementedError
    
    async def cleanup(self) -> None:
        """Cleanup after the chaos scenario."""
        self.end_time = datetime.utcnow()
        logger.info(f"Completed chaos scenario: {self.name}")
    
    def duration_seconds(self) -> float:
        """Get scenario duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class DatabaseFailureScenario(ChaosScenario):
    """Simulate database failures and recovery."""
    
    def __init__(self):
        super().__init__(
            "database_failure",
            "Simulate database connection failures and recovery"
        )
        self.original_session_factory = None
    
    @asynccontextmanager
    async def failing_database(self, failure_rate: float = 0.3):
        """Context manager that simulates database failures."""
        
        async def failing_session_factory():
            if random.random() < failure_rate:
                raise OperationalError("Database connection failed", None, None)
            
            # Simulate slow database
            if random.random() < 0.2:
                await asyncio.sleep(random.uniform(1.0, 3.0))
            
            # Return mock session for successful connections
            mock_session = AsyncMock(spec=AsyncSession)
            return mock_session
        
        # Patch the session factory
        with patch('app.core.database.get_async_session', failing_session_factory):
            yield
    
    async def execute(self) -> None:
        """Execute database failure scenario."""
        orchestrator = AgentOrchestrator()
        success_count = 0
        failure_count = 0
        recovery_count = 0
        
        async with self.failing_database(failure_rate=0.4):
            # Simulate operations under database stress
            for i in range(50):
                try:
                    # Attempt database operation
                    result = await orchestrator.get_agent_status("test_agent")
                    success_count += 1
                    
                    # If we had failures before, this is a recovery
                    if failure_count > 0 and success_count == 1:
                        recovery_count += 1
                        
                except (OperationalError, DisconnectionError):
                    failure_count += 1
                    logger.warning(f"Database operation failed (attempt {i+1})")
                    
                    # Simulate retry logic
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                
                # Add jitter to operations
                await asyncio.sleep(random.uniform(0.01, 0.1))
        
        self.metrics.update({
            "total_operations": 50,
            "success_count": success_count,
            "failure_count": failure_count,
            "recovery_count": recovery_count,
            "success_rate": success_count / 50,
            "failure_rate": failure_count / 50
        })


class RedisFailureScenario(ChaosScenario):
    """Simulate Redis failures and message loss."""
    
    def __init__(self):
        super().__init__(
            "redis_failure", 
            "Simulate Redis connection failures and message loss"
        )
    
    @asynccontextmanager
    async def failing_redis(self, failure_rate: float = 0.2):
        """Context manager that simulates Redis failures."""
        
        class FailingRedis:
            def __init__(self):
                self.failure_rate = failure_rate
                self.connected = True
            
            async def ping(self):
                if random.random() < self.failure_rate:
                    self.connected = False
                    raise ConnectionError("Redis connection lost")
                return True
            
            async def xadd(self, stream: str, fields: dict):
                if not self.connected or random.random() < self.failure_rate:
                    raise ConnectionError("Redis not available")
                return f"{int(time.time() * 1000)}-0"
            
            async def xreadgroup(self, group: str, consumer: str, streams: dict, count: int = 1):
                if not self.connected or random.random() < self.failure_rate:
                    raise ConnectionError("Redis not available")
                return []
            
            async def publish(self, channel: str, message: str):
                if not self.connected or random.random() < self.failure_rate:
                    return 0  # No subscribers (connection failed)
                return 1
        
        failing_redis = FailingRedis()
        
        with patch('app.core.redis.get_redis', return_value=failing_redis):
            yield failing_redis
    
    async def execute(self) -> None:
        """Execute Redis failure scenario."""
        message_broker = AgentMessageBroker()
        sent_count = 0
        failed_count = 0
        recovered_count = 0
        
        async with self.failing_redis(failure_rate=0.3) as redis_client:
            # Simulate message operations under Redis stress
            for i in range(100):
                try:
                    # Attempt to send message
                    await message_broker.send_message(
                        from_agent="test_sender",
                        to_agent="test_receiver", 
                        message_type="test_message",
                        payload={"test": True, "sequence": i}
                    )
                    sent_count += 1
                    
                    # Check if this is a recovery
                    if failed_count > 0 and sent_count > 0:
                        redis_client.connected = True
                        recovered_count += 1
                        
                except ConnectionError:
                    failed_count += 1
                    logger.warning(f"Redis operation failed (attempt {i+1})")
                    
                    # Simulate reconnection attempts
                    if i % 10 == 0:  # Try to reconnect every 10 failures
                        try:
                            await redis_client.ping()
                            redis_client.connected = True
                        except ConnectionError:
                            pass
                
                await asyncio.sleep(random.uniform(0.01, 0.05))
        
        self.metrics.update({
            "total_messages": 100,
            "sent_count": sent_count,
            "failed_count": failed_count,
            "recovered_count": recovered_count,
            "delivery_rate": sent_count / 100,
            "failure_rate": failed_count / 100
        })


class ResourceExhaustionScenario(ChaosScenario):
    """Simulate resource exhaustion (memory, CPU)."""
    
    def __init__(self):
        super().__init__(
            "resource_exhaustion",
            "Simulate memory and CPU exhaustion"
        )
        self.memory_hogs = []
        self.cpu_tasks = []
    
    async def consume_memory(self, target_mb: int = 100):
        """Consume memory to simulate exhaustion."""
        chunk_size = 1024 * 1024  # 1MB chunks
        chunks_needed = target_mb
        
        for _ in range(chunks_needed):
            # Allocate memory chunk
            chunk = bytearray(chunk_size)
            self.memory_hogs.append(chunk)
            await asyncio.sleep(0.01)  # Small delay to allow other operations
    
    async def consume_cpu(self, duration_seconds: int = 5):
        """Consume CPU to simulate high load."""
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time:
            # CPU-intensive calculation
            sum([i**2 for i in range(1000)])
            await asyncio.sleep(0.001)  # Tiny yield to event loop
    
    async def execute(self) -> None:
        """Execute resource exhaustion scenario."""
        orchestrator = AgentOrchestrator()
        
        # Record initial resource usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()
        
        # Start resource consumption in background
        memory_task = asyncio.create_task(self.consume_memory(50))  # 50MB
        cpu_task = asyncio.create_task(self.consume_cpu(10))  # 10 seconds
        
        # Monitor system behavior under resource stress
        operation_times = []
        failures = 0
        
        for i in range(20):
            start_time = time.time()
            
            try:
                # Perform operations under resource stress
                await orchestrator.get_system_status()
                
                operation_time = (time.time() - start_time) * 1000  # ms
                operation_times.append(operation_time)
                
            except Exception as e:
                failures += 1
                logger.warning(f"Operation failed under resource stress: {e}")
            
            await asyncio.sleep(0.5)  # 500ms between operations
        
        # Wait for resource consumption to complete
        await asyncio.gather(memory_task, cpu_task, return_exceptions=True)
        
        # Record final resource usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        final_cpu = process.cpu_percent()
        
        # Calculate metrics
        if operation_times:
            avg_response_time = sum(operation_times) / len(operation_times)
            max_response_time = max(operation_times)
        else:
            avg_response_time = max_response_time = 0
        
        self.metrics.update({
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": final_memory - initial_memory,
            "initial_cpu_percent": initial_cpu,
            "final_cpu_percent": final_cpu,
            "total_operations": 20,
            "failed_operations": failures,
            "success_rate": (20 - failures) / 20,
            "avg_response_time_ms": avg_response_time,
            "max_response_time_ms": max_response_time
        })
    
    async def cleanup(self) -> None:
        """Cleanup allocated resources."""
        # Release memory
        self.memory_hogs.clear()
        gc.collect()  # Force garbage collection
        
        # Cancel CPU tasks
        for task in self.cpu_tasks:
            if not task.done():
                task.cancel()
        
        await super().cleanup()


class NetworkPartitionScenario(ChaosScenario):
    """Simulate network partitions and latency."""
    
    def __init__(self):
        super().__init__(
            "network_partition",
            "Simulate network partitions and high latency"
        )
    
    @asynccontextmanager
    async def network_chaos(self, 
                           partition_rate: float = 0.1,
                           latency_ms: int = 1000):
        """Context manager for network chaos."""
        
        original_methods = {}
        
        def add_network_delay(original_method):
            async def delayed_method(*args, **kwargs):
                # Simulate network partition
                if random.random() < partition_rate:
                    raise ConnectionError("Network partition")
                
                # Simulate high latency
                delay = random.uniform(0, latency_ms / 1000.0)
                await asyncio.sleep(delay)
                
                return await original_method(*args, **kwargs)
            
            return delayed_method
        
        # Mock network-dependent methods
        with patch.multiple(
            'app.core.communication.MessageBroker',
            send_message=add_network_delay(AsyncMock()),
            receive_message=add_network_delay(AsyncMock()),
        ):
            yield
    
    async def execute(self) -> None:
        """Execute network partition scenario."""
        message_broker = MessageBroker()
        
        successful_ops = 0
        failed_ops = 0
        response_times = []
        
        async with self.network_chaos(partition_rate=0.15, latency_ms=500):
            # Simulate operations under network stress
            for i in range(30):
                start_time = time.time()
                
                try:
                    await message_broker.send_message(
                        "test_agent",
                        {"type": "ping", "sequence": i}
                    )
                    
                    response_time = (time.time() - start_time) * 1000
                    response_times.append(response_time)
                    successful_ops += 1
                    
                except ConnectionError:
                    failed_ops += 1
                    logger.warning(f"Network partition detected (operation {i+1})")
                
                except Exception as e:
                    failed_ops += 1
                    logger.error(f"Unexpected network error: {e}")
                
                await asyncio.sleep(0.2)  # 200ms between operations
        
        # Calculate network performance metrics
        if response_times:
            avg_latency = sum(response_times) / len(response_times)
            p95_latency = sorted(response_times)[int(len(response_times) * 0.95)]
            p99_latency = sorted(response_times)[int(len(response_times) * 0.99)]
        else:
            avg_latency = p95_latency = p99_latency = 0
        
        self.metrics.update({
            "total_operations": 30,
            "successful_operations": successful_ops,
            "failed_operations": failed_ops,
            "success_rate": successful_ops / 30,
            "partition_rate": failed_ops / 30,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency
        })


class CascadingFailureScenario(ChaosScenario):
    """Simulate cascading failures across components."""
    
    def __init__(self):
        super().__init__(
            "cascading_failure",
            "Simulate cascading failures across system components"
        )
        self.failure_chain = []
    
    async def trigger_database_failure(self):
        """Trigger database failure."""
        self.failure_chain.append(("database", datetime.utcnow()))
        
        # Simulate database becoming unavailable
        with patch('app.core.database.get_async_session') as mock_session:
            mock_session.side_effect = OperationalError("Database down", None, None)
            
            # Let failure propagate for a short time
            await asyncio.sleep(2.0)
    
    async def trigger_redis_failure(self):
        """Trigger Redis failure (usually follows database failure)."""
        self.failure_chain.append(("redis", datetime.utcnow()))
        
        # Simulate Redis becoming unavailable
        with patch('app.core.redis.get_redis') as mock_redis:
            mock_redis.side_effect = ConnectionError("Redis connection lost")
            
            await asyncio.sleep(2.0)
    
    async def trigger_orchestrator_failure(self):
        """Trigger orchestrator failure (follows Redis failure)."""
        self.failure_chain.append(("orchestrator", datetime.utcnow()))
        
        # Simulate orchestrator failing
        with patch.object(AgentOrchestrator, 'process_message') as mock_process:
            mock_process.side_effect = Exception("Orchestrator overloaded")
            
            await asyncio.sleep(2.0)
    
    async def execute(self) -> None:
        """Execute cascading failure scenario."""
        orchestrator = AgentOrchestrator()
        health_monitor = HealthMonitor()
        
        # Start monitoring system health
        initial_health = await health_monitor.get_system_health()
        
        # Trigger cascading failures
        failure_tasks = [
            asyncio.create_task(self.trigger_database_failure()),
            asyncio.create_task(self.trigger_redis_failure()),
            asyncio.create_task(self.trigger_orchestrator_failure())
        ]
        
        # Monitor system behavior during cascading failures
        health_checks = []
        operation_results = []
        
        for i in range(10):
            # Check system health
            try:
                health = await health_monitor.get_system_health()
                health_checks.append({
                    "timestamp": datetime.utcnow(),
                    "status": health.status,
                    "healthy_components": len([c for c in health.components if c.healthy])
                })
            except Exception as e:
                health_checks.append({
                    "timestamp": datetime.utcnow(),
                    "status": "error",
                    "error": str(e)
                })
            
            # Try system operations
            try:
                result = await orchestrator.get_system_status()
                operation_results.append({"success": True, "timestamp": datetime.utcnow()})
            except Exception as e:
                operation_results.append({
                    "success": False, 
                    "error": str(e),
                    "timestamp": datetime.utcnow()
                })
            
            await asyncio.sleep(1.0)
        
        # Wait for failure scenarios to complete
        await asyncio.gather(*failure_tasks, return_exceptions=True)
        
        # Analyze cascading effect
        successful_operations = sum(1 for r in operation_results if r.get("success"))
        healthy_checks = sum(1 for h in health_checks if h.get("status") != "error")
        
        # Calculate failure propagation time
        if len(self.failure_chain) >= 2:
            propagation_time = (
                self.failure_chain[-1][1] - self.failure_chain[0][1]
            ).total_seconds()
        else:
            propagation_time = 0
        
        self.metrics.update({
            "failure_chain": [f[0] for f in self.failure_chain],
            "failure_count": len(self.failure_chain),
            "propagation_time_seconds": propagation_time,
            "total_operations": len(operation_results),
            "successful_operations": successful_operations,
            "operation_success_rate": successful_operations / len(operation_results),
            "total_health_checks": len(health_checks),
            "healthy_checks": healthy_checks,
            "health_check_success_rate": healthy_checks / len(health_checks)
        })


@pytest.mark.chaos
class TestChaosEngineering:
    """Main chaos engineering test class."""
    
    @pytest.mark.asyncio
    async def test_database_failure_resilience(self):
        """Test system resilience to database failures."""
        scenario = DatabaseFailureScenario()
        
        await scenario.setup()
        await scenario.execute()
        await scenario.cleanup()
        
        # Verify resilience metrics
        assert scenario.metrics["success_rate"] > 0.3, "System should handle some database failures"
        assert scenario.metrics["recovery_count"] > 0, "System should recover from failures"
        
        logger.info(f"Database failure scenario metrics: {scenario.metrics}")
    
    @pytest.mark.asyncio
    async def test_redis_failure_resilience(self):
        """Test system resilience to Redis failures."""
        scenario = RedisFailureScenario()
        
        await scenario.setup()
        await scenario.execute()
        await scenario.cleanup()
        
        # Verify resilience metrics
        assert scenario.metrics["delivery_rate"] > 0.4, "System should deliver some messages despite failures"
        assert scenario.metrics["recovered_count"] > 0, "System should recover from Redis failures"
        
        logger.info(f"Redis failure scenario metrics: {scenario.metrics}")
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion_resilience(self):
        """Test system resilience to resource exhaustion."""
        scenario = ResourceExhaustionScenario()
        
        await scenario.setup()
        await scenario.execute()
        await scenario.cleanup()
        
        # Verify resilience metrics
        assert scenario.metrics["success_rate"] > 0.5, "System should handle resource pressure"
        assert scenario.metrics["avg_response_time_ms"] < 5000, "Response times should remain reasonable"
        
        logger.info(f"Resource exhaustion scenario metrics: {scenario.metrics}")
    
    @pytest.mark.asyncio
    async def test_network_partition_resilience(self):
        """Test system resilience to network partitions."""
        scenario = NetworkPartitionScenario()
        
        await scenario.setup()
        await scenario.execute()
        await scenario.cleanup()
        
        # Verify resilience metrics
        assert scenario.metrics["success_rate"] > 0.6, "System should handle network partitions"
        assert scenario.metrics["p95_latency_ms"] < 2000, "Latency should be manageable"
        
        logger.info(f"Network partition scenario metrics: {scenario.metrics}")
    
    @pytest.mark.asyncio
    async def test_cascading_failure_resilience(self):
        """Test system resilience to cascading failures."""
        scenario = CascadingFailureScenario()
        
        await scenario.setup()
        await scenario.execute()
        await scenario.cleanup()
        
        # Verify resilience metrics
        assert scenario.metrics["operation_success_rate"] > 0.2, "Some operations should succeed despite cascading failures"
        assert scenario.metrics["health_check_success_rate"] > 0.3, "Health monitoring should partially work"
        assert scenario.metrics["propagation_time_seconds"] < 10, "Failures shouldn't propagate too slowly"
        
        logger.info(f"Cascading failure scenario metrics: {scenario.metrics}")
    
    @pytest.mark.asyncio
    async def test_recovery_mechanisms(self):
        """Test system recovery mechanisms."""
        recovery_manager = RecoveryManager()
        
        # Simulate failure detection
        failure_event = {
            "component": "database",
            "error": "Connection timeout",
            "timestamp": datetime.utcnow(),
            "severity": "high"
        }
        
        # Test recovery initiation
        recovery_plan = await recovery_manager.create_recovery_plan(failure_event)
        assert recovery_plan is not None
        assert "steps" in recovery_plan
        assert len(recovery_plan["steps"]) > 0
        
        # Test recovery execution
        recovery_result = await recovery_manager.execute_recovery_plan(recovery_plan)
        assert recovery_result["status"] in ["success", "partial", "failed"]
        
        # Verify recovery metrics
        if recovery_result["status"] == "success":
            assert recovery_result["recovery_time_seconds"] < 30
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior(self):
        """Test circuit breaker behavior under failures."""
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=5,
            expected_exception=Exception
        )
        
        # Function that fails predictably
        call_count = 0
        
        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count <= 5:  # Fail first 5 calls
                raise Exception("Simulated failure")
            return "success"
        
        results = []
        
        # Test circuit breaker behavior
        for i in range(10):
            try:
                result = await circuit_breaker.call(failing_function)
                results.append(("success", result))
            except Exception as e:
                results.append(("failure", str(e)))
            
            await asyncio.sleep(0.1)
        
        # Verify circuit breaker opened after threshold
        failure_count = sum(1 for r in results if r[0] == "failure")
        success_count = sum(1 for r in results if r[0] == "success")
        
        assert failure_count >= 3, "Circuit breaker should detect failures"
        assert success_count >= 1, "Circuit breaker should allow recovery"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "chaos"])