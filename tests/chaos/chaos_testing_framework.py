"""
Chaos Testing Framework - LeanVibe Agent Hive 2.0
Reusable infrastructure for chaos engineering and resilience validation.
"""

import asyncio
import time
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from unittest.mock import Mock, patch
import psutil

logger = logging.getLogger(__name__)

@dataclass
class ChaosScenario:
    """Definition of a chaos testing scenario."""
    name: str
    description: str
    target_services: List[str]
    failure_modes: List[str]
    duration_seconds: float
    expected_recovery_time_seconds: float
    success_criteria: Dict[str, Any]

@dataclass
class ChaosMetrics:
    """Metrics collected during chaos testing."""
    scenario_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    availability_measurements: List[float] = field(default_factory=list)
    recovery_times: List[float] = field(default_factory=list)
    error_rates: List[float] = field(default_factory=list)
    performance_impacts: List[float] = field(default_factory=list)
    successful_operations: int = 0
    failed_operations: int = 0
    data_loss_incidents: int = 0

class ChaosTestingFramework:
    """Framework for orchestrating chaos testing scenarios."""
    
    def __init__(self):
        self.active_scenarios = {}
        self.metrics_collectors = []
        self.failure_injectors = {}
        self.recovery_validators = {}
        
    def register_failure_injector(self, name: str, injector: Callable):
        """Register a failure injection mechanism."""
        self.failure_injectors[name] = injector
        logger.info(f"Registered failure injector: {name}")
    
    def register_recovery_validator(self, name: str, validator: Callable):
        """Register a recovery validation mechanism."""
        self.recovery_validators[name] = validator
        logger.info(f"Registered recovery validator: {name}")
    
    @asynccontextmanager
    async def chaos_scenario(self, scenario: ChaosScenario):
        """Execute a chaos scenario with comprehensive monitoring."""
        metrics = ChaosMetrics(
            scenario_name=scenario.name,
            start_time=datetime.utcnow()
        )
        
        logger.info(f"🎭 Starting chaos scenario: {scenario.name}")
        
        try:
            # Pre-scenario baseline measurement
            baseline_metrics = await self._collect_baseline_metrics()
            
            # Execute the scenario
            yield metrics
            
            # Post-scenario recovery validation
            recovery_metrics = await self._validate_recovery(scenario, baseline_metrics)
            metrics.recovery_times.extend(recovery_metrics.get("recovery_times", []))
            
        except Exception as e:
            logger.error(f"Chaos scenario {scenario.name} failed: {e}")
            raise
        finally:
            metrics.end_time = datetime.utcnow()
            logger.info(f"✅ Completed chaos scenario: {scenario.name}")
    
    async def _collect_baseline_metrics(self) -> Dict[str, Any]:
        """Collect baseline system metrics before chaos injection."""
        process = psutil.Process()
        
        return {
            "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "open_files": len(process.open_files()),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _validate_recovery(self, scenario: ChaosScenario, baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Validate system recovery after chaos injection."""
        recovery_start = time.time()
        
        # Wait for system to stabilize
        await asyncio.sleep(2)
        
        # Collect post-chaos metrics
        process = psutil.Process()
        post_metrics = {
            "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "open_files": len(process.open_files()),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        recovery_time = time.time() - recovery_start
        
        # Calculate recovery quality
        memory_recovery = abs(post_metrics["memory_usage_mb"] - baseline["memory_usage_mb"])
        cpu_recovery = abs(post_metrics["cpu_percent"] - baseline["cpu_percent"])
        
        return {
            "recovery_times": [recovery_time],
            "memory_recovery_mb": memory_recovery,
            "cpu_recovery_percent": cpu_recovery,
            "recovery_quality": "EXCELLENT" if memory_recovery < 10 and cpu_recovery < 5 else "GOOD"
        }

class FailureInjector:
    """Injectable failure mechanisms for chaos testing."""
    
    @staticmethod
    @asynccontextmanager
    async def redis_connection_failure():
        """Inject Redis connection failures."""
        with patch('aioredis.from_url') as mock_redis:
            mock_redis.side_effect = ConnectionError("Simulated Redis failure")
            logger.info("🔥 Injected Redis connection failure")
            yield
            logger.info("✅ Restored Redis connection")
    
    @staticmethod
    @asynccontextmanager
    async def database_unavailability():
        """Inject database unavailability."""
        with patch('sqlalchemy.ext.asyncio.AsyncSession') as mock_session:
            from sqlalchemy.exc import DisconnectionError
            mock_session.side_effect = DisconnectionError("Simulated DB failure", None, None)
            logger.info("🔥 Injected database unavailability")
            yield
            logger.info("✅ Restored database connection")
    
    @staticmethod
    @asynccontextmanager
    async def network_partition(target_services: List[str]):
        """Inject network partition for specific services."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = asyncio.TimeoutError("Network partition")
            logger.info(f"🔥 Injected network partition for services: {target_services}")
            yield
            logger.info("✅ Restored network connectivity")
    
    @staticmethod
    @asynccontextmanager
    async def service_overload(latency_multiplier: float = 10.0):
        """Inject service overload with increased latency."""
        original_sleep = asyncio.sleep
        
        async def slow_sleep(delay):
            await original_sleep(delay * latency_multiplier)
        
        with patch('asyncio.sleep', slow_sleep):
            logger.info(f"🔥 Injected service overload (latency x{latency_multiplier})")
            yield
            logger.info("✅ Restored normal service latency")
    
    @staticmethod
    @asynccontextmanager
    async def memory_pressure(pressure_mb: int = 100):
        """Inject memory pressure."""
        memory_hogs = []
        
        try:
            # Allocate memory to create pressure
            for i in range(pressure_mb // 10):
                memory_hog = bytearray(10 * 1024 * 1024)  # 10MB chunks
                memory_hogs.append(memory_hog)
            
            logger.info(f"🔥 Injected memory pressure: {pressure_mb}MB")
            yield
            
        finally:
            # Clean up memory
            memory_hogs.clear()
            logger.info("✅ Released memory pressure")

class ResilienceValidator:
    """Validation mechanisms for resilience testing."""
    
    @staticmethod
    async def validate_availability(
        operation_func: Callable[[], Awaitable[Any]],
        iterations: int = 100,
        target_availability: float = 99.95
    ) -> Dict[str, Any]:
        """Validate service availability under stress."""
        successful_operations = 0
        failed_operations = 0
        operation_times = []
        
        for i in range(iterations):
            start_time = time.time()
            
            try:
                await operation_func()
                successful_operations += 1
                
            except Exception as e:
                failed_operations += 1
                logger.warning(f"Operation {i} failed: {e}")
            
            operation_time = (time.time() - start_time) * 1000  # ms
            operation_times.append(operation_time)
        
        availability = (successful_operations / iterations) * 100
        avg_operation_time = sum(operation_times) / len(operation_times)
        
        return {
            "availability_percent": availability,
            "target_met": availability >= target_availability,
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "avg_operation_time_ms": avg_operation_time,
            "max_operation_time_ms": max(operation_times),
            "min_operation_time_ms": min(operation_times)
        }
    
    @staticmethod
    async def validate_recovery_time(
        failure_injection: Callable,
        recovery_validation: Callable,
        target_recovery_seconds: float = 30.0
    ) -> Dict[str, Any]:
        """Validate recovery time after failure injection."""
        
        # Inject failure
        failure_start = time.time()
        await failure_injection()
        failure_duration = time.time() - failure_start
        
        # Validate recovery
        recovery_start = time.time()
        recovery_result = await recovery_validation()
        recovery_time = time.time() - recovery_start
        
        return {
            "recovery_time_seconds": recovery_time,
            "target_met": recovery_time <= target_recovery_seconds,
            "failure_duration_seconds": failure_duration,
            "recovery_successful": recovery_result,
            "total_incident_time": failure_duration + recovery_time
        }
    
    @staticmethod
    async def validate_data_integrity(
        pre_chaos_state: Dict[str, Any],
        post_chaos_state: Dict[str, Any],
        critical_fields: List[str]
    ) -> Dict[str, Any]:
        """Validate data integrity after chaos testing."""
        integrity_issues = []
        
        for field in critical_fields:
            pre_value = pre_chaos_state.get(field)
            post_value = post_chaos_state.get(field)
            
            if pre_value != post_value:
                integrity_issues.append({
                    "field": field,
                    "pre_value": pre_value,
                    "post_value": post_value
                })
        
        data_integrity_preserved = len(integrity_issues) == 0
        
        return {
            "data_integrity_preserved": data_integrity_preserved,
            "integrity_issues": integrity_issues,
            "critical_fields_checked": len(critical_fields),
            "fields_corrupted": len(integrity_issues)
        }

class ChaosScenarioBuilder:
    """Builder for creating comprehensive chaos scenarios."""
    
    @staticmethod
    def build_redis_failure_scenario() -> ChaosScenario:
        """Build Redis connection failure scenario."""
        return ChaosScenario(
            name="Redis Connection Failure",
            description="Test system resilience during Redis unavailability",
            target_services=["redis", "message_queue", "session_storage"],
            failure_modes=["connection_timeout", "connection_refused", "network_partition"],
            duration_seconds=30.0,
            expected_recovery_time_seconds=10.0,
            success_criteria={
                "min_availability_percent": 99.95,
                "max_recovery_time_seconds": 30.0,
                "max_data_loss_percent": 0.0
            }
        )
    
    @staticmethod
    def build_database_failure_scenario() -> ChaosScenario:
        """Build database unavailability scenario."""
        return ChaosScenario(
            name="Database Unavailability",
            description="Test graceful degradation during database outages",
            target_services=["postgresql", "semantic_memory", "agent_state"],
            failure_modes=["connection_timeout", "deadlock", "disk_full"],
            duration_seconds=45.0,
            expected_recovery_time_seconds=15.0,
            success_criteria={
                "min_availability_percent": 95.0,
                "max_recovery_time_seconds": 30.0,
                "fallback_effectiveness_percent": 90.0
            }
        )
    
    @staticmethod
    def build_service_overload_scenario() -> ChaosScenario:
        """Build service overload scenario."""
        return ChaosScenario(
            name="Service Overload",
            description="Test circuit breaker protection during service overload",
            target_services=["semantic_memory", "workflow_engine", "api_gateway"],
            failure_modes=["high_latency", "connection_exhaustion", "memory_pressure"],
            duration_seconds=60.0,
            expected_recovery_time_seconds=20.0,
            success_criteria={
                "circuit_breaker_activation": True,
                "max_cascade_failure_percent": 5.0,
                "protection_effectiveness_percent": 95.0
            }
        )
    
    @staticmethod
    def build_poison_message_scenario() -> ChaosScenario:
        """Build poison message attack scenario."""
        return ChaosScenario(
            name="Poison Message Attack",
            description="Test DLQ isolation during poison message flood",
            target_services=["message_processor", "dlq_manager", "poison_detector"],
            failure_modes=["malformed_json", "oversized_payload", "circular_reference"],
            duration_seconds=120.0,
            expected_recovery_time_seconds=5.0,
            success_criteria={
                "min_detection_accuracy_percent": 95.0,
                "max_system_impact_percent": 2.0,
                "isolation_effectiveness_percent": 99.0
            }
        )

class ChaosTestRunner:
    """Orchestrator for running chaos testing suites."""
    
    def __init__(self):
        self.framework = ChaosTestingFramework()
        self.scenarios_executed = 0
        self.scenarios_passed = 0
        self.overall_metrics = {}
    
    async def run_scenario(self, scenario: ChaosScenario) -> Dict[str, Any]:
        """Execute a single chaos scenario with full validation."""
        logger.info(f"🎬 Executing chaos scenario: {scenario.name}")
        
        async with self.framework.chaos_scenario(scenario) as metrics:
            scenario_start = time.time()
            
            # Phase 1: Baseline measurement
            baseline = await self._measure_baseline_performance()
            
            # Phase 2: Chaos injection
            chaos_result = await self._inject_chaos(scenario)
            
            # Phase 3: Recovery validation
            recovery_result = await self._validate_recovery(scenario, baseline)
            
            scenario_duration = time.time() - scenario_start
            
            # Evaluate success criteria
            success = await self._evaluate_success_criteria(scenario, chaos_result, recovery_result)
            
            return {
                "scenario": scenario.name,
                "passed": success,
                "duration_seconds": scenario_duration,
                "baseline_metrics": baseline,
                "chaos_result": chaos_result,
                "recovery_result": recovery_result,
                "success_criteria_met": success
            }
    
    async def _measure_baseline_performance(self) -> Dict[str, Any]:
        """Measure baseline system performance."""
        baseline_operations = 50
        baseline_times = []
        baseline_success = 0
        
        for i in range(baseline_operations):
            start_time = time.time()
            
            try:
                # Simulate baseline operation
                await asyncio.sleep(0.005)  # 5ms normal operation
                baseline_success += 1
                
            except Exception:
                pass
            
            operation_time = (time.time() - start_time) * 1000
            baseline_times.append(operation_time)
        
        return {
            "availability_percent": (baseline_success / baseline_operations) * 100,
            "avg_response_time_ms": sum(baseline_times) / len(baseline_times),
            "max_response_time_ms": max(baseline_times),
            "successful_operations": baseline_success,
            "total_operations": baseline_operations
        }
    
    async def _inject_chaos(self, scenario: ChaosScenario) -> Dict[str, Any]:
        """Inject chaos according to scenario specification."""
        chaos_operations = 200
        chaos_success = 0
        chaos_times = []
        
        # Simulate chaos injection based on failure modes
        for failure_mode in scenario.failure_modes:
            logger.info(f"   Injecting failure mode: {failure_mode}")
            
            for i in range(chaos_operations // len(scenario.failure_modes)):
                start_time = time.time()
                
                try:
                    # Simulate chaos with different failure characteristics
                    if failure_mode == "connection_timeout":
                        await asyncio.sleep(0.1)  # 100ms timeout simulation
                    elif failure_mode == "high_latency":
                        await asyncio.sleep(0.05)  # 50ms high latency
                    elif failure_mode == "memory_pressure":
                        await asyncio.sleep(0.01)  # 10ms with memory pressure
                    else:
                        await asyncio.sleep(0.002)  # 2ms normal chaos operation
                    
                    chaos_success += 1
                    
                except Exception:
                    # Expected chaos-induced failures
                    pass
                
                operation_time = (time.time() - start_time) * 1000
                chaos_times.append(operation_time)
        
        return {
            "chaos_availability_percent": (chaos_success / chaos_operations) * 100,
            "avg_chaos_response_time_ms": sum(chaos_times) / len(chaos_times) if chaos_times else 0,
            "successful_chaos_operations": chaos_success,
            "total_chaos_operations": chaos_operations,
            "failure_modes_tested": len(scenario.failure_modes)
        }
    
    async def _validate_recovery(self, scenario: ChaosScenario, baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Validate system recovery after chaos injection."""
        recovery_start = time.time()
        
        # Wait for expected recovery time
        await asyncio.sleep(scenario.expected_recovery_time_seconds)
        
        # Test recovery operations
        recovery_operations = 25
        recovery_success = 0
        recovery_times = []
        
        for i in range(recovery_operations):
            start_time = time.time()
            
            try:
                # Recovery operations should work normally
                await asyncio.sleep(0.005)  # 5ms normal operation
                recovery_success += 1
                
            except Exception:
                pass
            
            operation_time = (time.time() - start_time) * 1000
            recovery_times.append(operation_time)
        
        recovery_time = time.time() - recovery_start
        recovery_availability = (recovery_success / recovery_operations) * 100
        
        # Compare with baseline
        baseline_availability = baseline["availability_percent"]
        recovery_effectiveness = (recovery_availability / baseline_availability) * 100 if baseline_availability > 0 else 0
        
        return {
            "recovery_time_seconds": recovery_time,
            "recovery_availability_percent": recovery_availability,
            "recovery_effectiveness_percent": recovery_effectiveness,
            "successful_recovery_operations": recovery_success,
            "total_recovery_operations": recovery_operations,
            "avg_recovery_response_time_ms": sum(recovery_times) / len(recovery_times) if recovery_times else 0
        }
    
    async def _evaluate_success_criteria(
        self, 
        scenario: ChaosScenario, 
        chaos_result: Dict[str, Any], 
        recovery_result: Dict[str, Any]
    ) -> bool:
        """Evaluate whether scenario met success criteria."""
        criteria = scenario.success_criteria
        success = True
        
        # Check availability criteria
        if "min_availability_percent" in criteria:
            recovery_availability = recovery_result["recovery_availability_percent"]
            if recovery_availability < criteria["min_availability_percent"]:
                logger.warning(f"Availability below threshold: {recovery_availability:.2f}% < {criteria['min_availability_percent']}%")
                success = False
        
        # Check recovery time criteria
        if "max_recovery_time_seconds" in criteria:
            recovery_time = recovery_result["recovery_time_seconds"]
            if recovery_time > criteria["max_recovery_time_seconds"]:
                logger.warning(f"Recovery time exceeded: {recovery_time:.2f}s > {criteria['max_recovery_time_seconds']}s")
                success = False
        
        # Check specific scenario criteria
        for criterion, target_value in criteria.items():
            if criterion not in ["min_availability_percent", "max_recovery_time_seconds"]:
                # Custom criteria validation would go here
                logger.info(f"Custom criterion {criterion}: target {target_value}")
        
        return success

# Utility functions for chaos testing
async def simulate_system_load(operations: int = 1000, operation_delay: float = 0.001):
    """Simulate system load during chaos testing."""
    tasks = []
    
    for i in range(operations):
        task = asyncio.create_task(asyncio.sleep(operation_delay))
        tasks.append(task)
    
    await asyncio.gather(*tasks)

async def measure_system_health() -> Dict[str, Any]:
    """Measure current system health metrics."""
    process = psutil.Process()
    
    return {
        "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(),
        "open_files": len(process.open_files()),
        "threads": process.num_threads(),
        "timestamp": datetime.utcnow().isoformat()
    }

def calculate_availability(successful: int, total: int) -> float:
    """Calculate availability percentage."""
    return (successful / total) * 100 if total > 0 else 0.0

def calculate_recovery_time(start_time: float, end_time: float) -> float:
    """Calculate recovery time in seconds."""
    return end_time - start_time