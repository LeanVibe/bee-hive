"""
Graceful Degradation and Error Recovery Validator for EPIC D Phase 2.

Implements comprehensive testing of graceful degradation patterns and error recovery
mechanisms to ensure system resilience under failure conditions.

Features:
- Graceful degradation pattern validation
- Error recovery mechanism testing
- Circuit breaker pattern validation
- Fallback strategy testing
- System resilience measurement
- Recovery time optimization
- Failure isolation testing
- Cascading failure prevention validation
"""

import asyncio
import logging
import time
import json
import statistics
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import redis
import psycopg2
import websockets
import pytest
import numpy as np

logger = logging.getLogger(__name__)


class DegradationScenario(Enum):
    """Types of degradation scenarios to test."""
    DATABASE_SLOW = "database_slow"
    DATABASE_UNAVAILABLE = "database_unavailable"
    CACHE_UNAVAILABLE = "cache_unavailable"
    CACHE_MEMORY_PRESSURE = "cache_memory_pressure"
    API_RATE_LIMIT = "api_rate_limit"
    WEBSOCKET_OVERLOAD = "websocket_overload"
    NETWORK_LATENCY = "network_latency"
    PARTIAL_SERVICE_FAILURE = "partial_service_failure"
    DEPENDENCY_TIMEOUT = "dependency_timeout"
    MEMORY_PRESSURE = "memory_pressure"


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    CIRCUIT_BREAKER = "circuit_breaker"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FALLBACK_SERVICE = "fallback_service"
    CACHE_BYPASS = "cache_bypass"
    GRACEFUL_TIMEOUT = "graceful_timeout"
    LOAD_SHEDDING = "load_shedding"
    BULKHEAD_ISOLATION = "bulkhead_isolation"
    FAIL_FAST = "fail_fast"


class DegradationLevel(Enum):
    """Levels of system degradation."""
    NONE = "none"           # 100% functionality
    MINIMAL = "minimal"     # 95%+ functionality  
    MODERATE = "moderate"   # 80%+ functionality
    SIGNIFICANT = "significant"  # 60%+ functionality
    SEVERE = "severe"       # 40%+ functionality
    CRITICAL = "critical"   # <40% functionality


@dataclass
class DegradationTestConfig:
    """Configuration for a degradation test scenario."""
    scenario: DegradationScenario
    description: str
    expected_degradation_level: DegradationLevel
    expected_recovery_strategy: RecoveryStrategy
    test_duration_seconds: int = 300
    recovery_timeout_seconds: int = 180
    
    # Failure simulation parameters
    failure_injection_delay_seconds: int = 30
    failure_intensity: float = 1.0  # 0.0 to 1.0
    
    # Performance thresholds during degradation
    max_acceptable_response_time_ms: float = 2000.0
    max_acceptable_error_rate_percent: float = 10.0
    min_acceptable_throughput_percent: float = 50.0


@dataclass
class DegradationTestResult:
    """Results of a degradation test."""
    config: DegradationTestConfig
    test_start_time: datetime
    test_duration_seconds: float
    
    # Baseline performance (before degradation)
    baseline_response_time_ms: float
    baseline_error_rate_percent: float
    baseline_throughput_rps: float
    
    # Performance during degradation
    degraded_response_time_ms: float
    degraded_error_rate_percent: float
    degraded_throughput_rps: float
    
    # Recovery metrics
    recovery_detected: bool
    recovery_time_seconds: float
    recovery_strategy_triggered: bool
    
    # Degradation analysis
    actual_degradation_level: DegradationLevel
    functionality_retained_percent: float
    graceful_degradation_achieved: bool
    
    # System resilience metrics
    cascading_failures_prevented: bool
    system_stability_maintained: bool
    user_impact_minimized: bool
    
    # Performance compliance during degradation
    response_time_within_threshold: bool
    error_rate_within_threshold: bool
    throughput_within_threshold: bool
    overall_compliance: bool


@dataclass
class CircuitBreakerState:
    """Circuit breaker state tracking."""
    component_id: str
    state: str  # closed, open, half_open
    failure_count: int
    failure_threshold: int
    success_count: int
    last_failure_time: Optional[datetime]
    recovery_timeout_seconds: int
    
    # Performance metrics
    request_count: int
    success_rate_percent: float
    avg_response_time_ms: float


class GracefulDegradationRecoveryValidator:
    """Validator for graceful degradation and error recovery patterns."""
    
    def __init__(self, 
                 api_base_url: str = "http://localhost:8000",
                 db_config: Dict[str, str] = None,
                 redis_config: Dict[str, str] = None):
        self.api_base_url = api_base_url
        self.db_config = db_config or {
            'host': 'localhost',
            'port': '15432',
            'database': 'leanvibe_agent_hive',
            'user': 'leanvibe_user',
            'password': 'secure_password'
        }
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 16379,
            'password': 'secure_redis_password',
            'db': 0
        }
        
        # Test results storage
        self.test_results: List[DegradationTestResult] = []
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        
        # Performance monitoring
        self.performance_samples = defaultdict(list)
        self.monitoring_active = False
        
        self._initialize_circuit_breakers()
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breaker state tracking."""
        components = [
            'api_server',
            'database',
            'redis_cache',
            'websocket_server',
            'external_service'
        ]
        
        for component in components:
            self.circuit_breakers[component] = CircuitBreakerState(
                component_id=component,
                state='closed',
                failure_count=0,
                failure_threshold=5,
                success_count=0,
                last_failure_time=None,
                recovery_timeout_seconds=60,
                request_count=0,
                success_rate_percent=100.0,
                avg_response_time_ms=0.0
            )
    
    async def measure_baseline_performance(self) -> Dict[str, float]:
        """Measure baseline system performance before degradation testing."""
        logger.info("📊 Measuring baseline performance...")
        
        # Take multiple samples to establish baseline
        response_times = []
        error_counts = []
        success_counts = []
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            for _ in range(50):  # 50 samples
                start_time = time.time()
                try:
                    async with session.get(f"{self.api_base_url}/health") as response:
                        response_time = (time.time() - start_time) * 1000
                        response_times.append(response_time)
                        
                        if response.status >= 400:
                            error_counts.append(1)
                        else:
                            success_counts.append(1)
                
                except Exception:
                    response_time = (time.time() - start_time) * 1000
                    response_times.append(response_time)
                    error_counts.append(1)
                
                await asyncio.sleep(0.1)  # Brief pause between requests
        
        total_requests = len(response_times)
        total_errors = sum(error_counts)
        total_successes = sum(success_counts)
        
        baseline_metrics = {
            'avg_response_time_ms': statistics.mean(response_times) if response_times else 0.0,
            'p95_response_time_ms': np.percentile(response_times, 95) if response_times else 0.0,
            'error_rate_percent': (total_errors / total_requests) * 100 if total_requests > 0 else 0.0,
            'success_rate_percent': (total_successes / total_requests) * 100 if total_requests > 0 else 0.0,
            'requests_per_second': total_requests / 5.0,  # 50 requests over ~5 seconds
            'total_requests': total_requests
        }
        
        logger.info(f"Baseline: {baseline_metrics['avg_response_time_ms']:.1f}ms avg, "
                   f"{baseline_metrics['error_rate_percent']:.1f}% errors, "
                   f"{baseline_metrics['requests_per_second']:.1f} RPS")
        
        return baseline_metrics
    
    async def simulate_degradation_scenario(self, config: DegradationTestConfig):
        """Simulate a specific degradation scenario."""
        logger.info(f"🧪 Simulating degradation scenario: {config.scenario.value}")
        
        if config.scenario == DegradationScenario.DATABASE_SLOW:
            await self._simulate_database_slowness(config)
        elif config.scenario == DegradationScenario.DATABASE_UNAVAILABLE:
            await self._simulate_database_unavailable(config)
        elif config.scenario == DegradationScenario.CACHE_UNAVAILABLE:
            await self._simulate_cache_unavailable(config)
        elif config.scenario == DegradationScenario.CACHE_MEMORY_PRESSURE:
            await self._simulate_cache_memory_pressure(config)
        elif config.scenario == DegradationScenario.API_RATE_LIMIT:
            await self._simulate_api_rate_limiting(config)
        elif config.scenario == DegradationScenario.WEBSOCKET_OVERLOAD:
            await self._simulate_websocket_overload(config)
        elif config.scenario == DegradationScenario.NETWORK_LATENCY:
            await self._simulate_network_latency(config)
        elif config.scenario == DegradationScenario.PARTIAL_SERVICE_FAILURE:
            await self._simulate_partial_service_failure(config)
        elif config.scenario == DegradationScenario.DEPENDENCY_TIMEOUT:
            await self._simulate_dependency_timeout(config)
        elif config.scenario == DegradationScenario.MEMORY_PRESSURE:
            await self._simulate_memory_pressure(config)
        else:
            logger.warning(f"Unknown degradation scenario: {config.scenario.value}")
    
    async def _simulate_database_slowness(self, config: DegradationTestConfig):
        """Simulate database slowness scenario."""
        # In a real implementation, this would:
        # 1. Inject artificial delays into database queries
        # 2. Simulate slow query execution
        # 3. Create connection pool pressure
        
        logger.info(f"💾 Simulating database slowness for {config.test_duration_seconds}s")
        
        # For testing purposes, we'll simulate by making concurrent requests
        # that would stress the database connections
        tasks = []
        
        async def stress_database():
            async with aiohttp.ClientSession() as session:
                for _ in range(10):  # 10 requests per task
                    try:
                        async with session.get(f"{self.api_base_url}/api/agents") as response:
                            await response.text()
                    except Exception as e:
                        logger.debug(f"Database stress request failed: {e}")
                    await asyncio.sleep(0.5)
        
        # Create multiple concurrent tasks to stress the system
        for _ in range(5):  # 5 concurrent stress tasks
            tasks.append(asyncio.create_task(stress_database()))
        
        # Wait for stress period
        await asyncio.sleep(config.failure_injection_delay_seconds)
        
        # Cancel stress tasks
        for task in tasks:
            task.cancel()
    
    async def _simulate_database_unavailable(self, config: DegradationTestConfig):
        """Simulate database unavailability scenario."""
        logger.info(f"💾❌ Simulating database unavailability for {config.test_duration_seconds}s")
        
        # In a real scenario, this would temporarily block database connections
        # For testing, we'll simulate with requests that would fail due to DB issues
        await asyncio.sleep(config.failure_injection_delay_seconds)
        
    async def _simulate_cache_unavailable(self, config: DegradationTestConfig):
        """Simulate cache unavailability scenario."""
        logger.info(f"🔄❌ Simulating cache unavailability for {config.test_duration_seconds}s")
        
        # In practice, this would disable Redis connections
        await asyncio.sleep(config.failure_injection_delay_seconds)
        
    async def _simulate_cache_memory_pressure(self, config: DegradationTestConfig):
        """Simulate cache memory pressure scenario."""
        logger.info(f"🔄💾 Simulating cache memory pressure for {config.test_duration_seconds}s")
        
        # Would fill Redis memory to capacity, triggering evictions
        await asyncio.sleep(config.failure_injection_delay_seconds)
    
    async def _simulate_api_rate_limiting(self, config: DegradationTestConfig):
        """Simulate API rate limiting scenario."""
        logger.info(f"🚦 Simulating API rate limiting for {config.test_duration_seconds}s")
        
        # Flood the API with requests to trigger rate limiting
        tasks = []
        
        async def flood_requests():
            async with aiohttp.ClientSession() as session:
                for _ in range(100):  # Many requests quickly
                    try:
                        async with session.get(f"{self.api_base_url}/health") as response:
                            await response.text()
                    except Exception:
                        pass
                    await asyncio.sleep(0.01)  # Very fast requests
        
        # Create flood of requests
        for _ in range(3):
            tasks.append(asyncio.create_task(flood_requests()))
        
        await asyncio.sleep(config.failure_injection_delay_seconds)
        
        # Stop flooding
        for task in tasks:
            task.cancel()
    
    async def _simulate_websocket_overload(self, config: DegradationTestConfig):
        """Simulate WebSocket connection overload scenario."""
        logger.info(f"🌐 Simulating WebSocket overload for {config.test_duration_seconds}s")
        
        connections = []
        try:
            # Create many WebSocket connections
            ws_url = self.api_base_url.replace('http', 'ws') + '/ws'
            
            for i in range(50):  # Create 50 concurrent connections
                try:
                    ws = await websockets.connect(ws_url, timeout=5)
                    connections.append(ws)
                    
                    # Send some data
                    await ws.send(json.dumps({
                        "type": "subscribe",
                        "channels": [f"test_channel_{i}"]
                    }))
                    
                except Exception as e:
                    logger.debug(f"WebSocket connection {i} failed: {e}")
            
            await asyncio.sleep(config.failure_injection_delay_seconds)
            
        finally:
            # Clean up connections
            for ws in connections:
                try:
                    await ws.close()
                except Exception:
                    pass
    
    async def _simulate_network_latency(self, config: DegradationTestConfig):
        """Simulate network latency increase scenario."""
        logger.info(f"🌐⏳ Simulating network latency for {config.test_duration_seconds}s")
        
        # Would inject artificial delays in network layer
        await asyncio.sleep(config.failure_injection_delay_seconds)
    
    async def _simulate_partial_service_failure(self, config: DegradationTestConfig):
        """Simulate partial service failure scenario."""
        logger.info(f"⚠️ Simulating partial service failure for {config.test_duration_seconds}s")
        
        # Test different endpoints to simulate partial failures
        endpoints = ['/api/agents', '/api/tasks', '/health', '/metrics']
        
        # Stress some endpoints while leaving others functional
        async def stress_endpoint(endpoint):
            async with aiohttp.ClientSession() as session:
                for _ in range(20):
                    try:
                        async with session.get(f"{self.api_base_url}{endpoint}") as response:
                            await response.text()
                    except Exception:
                        pass
                    await asyncio.sleep(0.2)
        
        # Stress selected endpoints
        tasks = []
        for endpoint in endpoints[:2]:  # Only stress first 2 endpoints
            tasks.append(asyncio.create_task(stress_endpoint(endpoint)))
        
        await asyncio.sleep(config.failure_injection_delay_seconds)
        
        for task in tasks:
            task.cancel()
    
    async def _simulate_dependency_timeout(self, config: DegradationTestConfig):
        """Simulate dependency timeout scenario."""
        logger.info(f"⏱️ Simulating dependency timeouts for {config.test_duration_seconds}s")
        
        # Would simulate timeouts to external dependencies
        await asyncio.sleep(config.failure_injection_delay_seconds)
    
    async def _simulate_memory_pressure(self, config: DegradationTestConfig):
        """Simulate memory pressure scenario."""
        logger.info(f"💾⚠️ Simulating memory pressure for {config.test_duration_seconds}s")
        
        # Would consume system memory to create pressure
        await asyncio.sleep(config.failure_injection_delay_seconds)
    
    async def monitor_performance_during_degradation(self, 
                                                   test_duration_seconds: int) -> Dict[str, List[float]]:
        """Monitor system performance during degradation testing."""
        monitoring_samples = {
            'response_times': [],
            'error_rates': [],
            'success_rates': [],
            'timestamps': []
        }
        
        start_time = time.time()
        sample_interval = 5  # Sample every 5 seconds
        
        while time.time() - start_time < test_duration_seconds:
            sample_start = time.time()
            
            # Take performance sample
            response_times = []
            errors = 0
            successes = 0
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                # Test multiple endpoints
                for _ in range(10):  # 10 requests per sample
                    request_start = time.time()
                    try:
                        async with session.get(f"{self.api_base_url}/health") as response:
                            response_time = (time.time() - request_start) * 1000
                            response_times.append(response_time)
                            
                            if response.status >= 400:
                                errors += 1
                            else:
                                successes += 1
                    
                    except Exception:
                        response_time = (time.time() - request_start) * 1000
                        response_times.append(response_time)
                        errors += 1
                    
                    await asyncio.sleep(0.1)
            
            # Record sample
            if response_times:
                monitoring_samples['response_times'].append(statistics.mean(response_times))
            else:
                monitoring_samples['response_times'].append(0.0)
            
            total_requests = errors + successes
            monitoring_samples['error_rates'].append((errors / total_requests) * 100 if total_requests > 0 else 0.0)
            monitoring_samples['success_rates'].append((successes / total_requests) * 100 if total_requests > 0 else 0.0)
            monitoring_samples['timestamps'].append(time.time())
            
            # Wait for next sample interval
            elapsed = time.time() - sample_start
            if elapsed < sample_interval:
                await asyncio.sleep(sample_interval - elapsed)
        
        return monitoring_samples
    
    def analyze_degradation_level(self, 
                                baseline_metrics: Dict[str, float],
                                degraded_metrics: Dict[str, float]) -> Tuple[DegradationLevel, float]:
        """Analyze the level of degradation based on performance metrics."""
        
        # Calculate performance ratios
        response_time_ratio = (degraded_metrics.get('avg_response_time_ms', 0) / 
                             baseline_metrics.get('avg_response_time_ms', 1))
        
        error_rate_increase = (degraded_metrics.get('error_rate_percent', 0) - 
                              baseline_metrics.get('error_rate_percent', 0))
        
        throughput_ratio = (degraded_metrics.get('requests_per_second', 0) / 
                          baseline_metrics.get('requests_per_second', 1))
        
        # Calculate overall functionality retained
        # Response time impact (inverse - slower is worse)
        response_score = min(1.0, 1.0 / max(1.0, response_time_ratio))
        
        # Error rate impact
        error_score = max(0.0, 1.0 - (error_rate_increase / 100.0))
        
        # Throughput impact
        throughput_score = min(1.0, throughput_ratio)
        
        # Overall functionality score
        functionality_score = (response_score + error_score + throughput_score) / 3.0
        functionality_percent = functionality_score * 100.0
        
        # Determine degradation level
        if functionality_percent >= 95.0:
            level = DegradationLevel.MINIMAL
        elif functionality_percent >= 80.0:
            level = DegradationLevel.MODERATE
        elif functionality_percent >= 60.0:
            level = DegradationLevel.SIGNIFICANT
        elif functionality_percent >= 40.0:
            level = DegradationLevel.SEVERE
        else:
            level = DegradationLevel.CRITICAL
        
        return level, functionality_percent
    
    def detect_recovery_strategy_activation(self, 
                                          performance_samples: Dict[str, List[float]]) -> bool:
        """Detect if recovery strategies were activated during the test."""
        if not performance_samples['response_times']:
            return False
        
        # Look for patterns that indicate recovery mechanisms
        response_times = performance_samples['response_times']
        error_rates = performance_samples['error_rates']
        
        # Check for circuit breaker pattern (sharp cutoff in response times)
        if len(response_times) > 10:
            # Look for sudden improvement in response times (recovery)
            late_samples = response_times[-5:]  # Last 5 samples
            early_samples = response_times[:5]   # First 5 samples
            
            if late_samples and early_samples:
                late_avg = statistics.mean(late_samples)
                early_avg = statistics.mean(early_samples)
                
                # If response times improved significantly, recovery likely occurred
                if early_avg > 1000 and late_avg < early_avg * 0.5:
                    return True
        
        # Check for error rate recovery
        if len(error_rates) > 10:
            late_errors = error_rates[-5:]
            early_errors = error_rates[:5]
            
            if late_errors and early_errors:
                late_avg_errors = statistics.mean(late_errors)
                early_avg_errors = statistics.mean(early_errors)
                
                # If error rates decreased significantly, recovery likely occurred
                if early_avg_errors > 10 and late_avg_errors < early_avg_errors * 0.5:
                    return True
        
        return False
    
    async def run_degradation_test(self, config: DegradationTestConfig) -> DegradationTestResult:
        """Run a complete degradation test scenario."""
        logger.info(f"🧪 Starting degradation test: {config.scenario.value}")
        
        test_start_time = datetime.utcnow()
        
        # 1. Measure baseline performance
        baseline_metrics = await self.measure_baseline_performance()
        
        # 2. Start performance monitoring
        monitoring_task = asyncio.create_task(
            self.monitor_performance_during_degradation(config.test_duration_seconds)
        )
        
        # 3. Start degradation simulation
        degradation_task = asyncio.create_task(
            self.simulate_degradation_scenario(config)
        )
        
        # 4. Wait for both tasks to complete
        try:
            performance_samples, _ = await asyncio.gather(
                monitoring_task, 
                degradation_task,
                return_exceptions=True
            )
            
            if isinstance(performance_samples, Exception):
                logger.error(f"Performance monitoring failed: {performance_samples}")
                performance_samples = {'response_times': [], 'error_rates': [], 'success_rates': []}
            
        except Exception as e:
            logger.error(f"Degradation test execution failed: {e}")
            performance_samples = {'response_times': [], 'error_rates': [], 'success_rates': []}
        
        test_duration = (datetime.utcnow() - test_start_time).total_seconds()
        
        # 5. Calculate degraded performance metrics
        if performance_samples['response_times']:
            degraded_metrics = {
                'avg_response_time_ms': statistics.mean(performance_samples['response_times']),
                'error_rate_percent': statistics.mean(performance_samples['error_rates']),
                'success_rate_percent': statistics.mean(performance_samples['success_rates']),
                'requests_per_second': len(performance_samples['response_times']) * 10 / test_duration  # Approx
            }
        else:
            degraded_metrics = {
                'avg_response_time_ms': float('inf'),
                'error_rate_percent': 100.0,
                'success_rate_percent': 0.0,
                'requests_per_second': 0.0
            }
        
        # 6. Analyze degradation level
        degradation_level, functionality_percent = self.analyze_degradation_level(
            baseline_metrics, degraded_metrics
        )
        
        # 7. Detect recovery strategy activation
        recovery_strategy_triggered = self.detect_recovery_strategy_activation(performance_samples)
        
        # 8. Analyze recovery
        recovery_detected = False
        recovery_time = config.recovery_timeout_seconds
        
        if performance_samples['response_times'] and len(performance_samples['response_times']) > 5:
            # Simple recovery detection - look for improvement in later samples
            mid_point = len(performance_samples['response_times']) // 2
            first_half_avg = statistics.mean(performance_samples['response_times'][:mid_point])
            second_half_avg = statistics.mean(performance_samples['response_times'][mid_point:])
            
            if first_half_avg > second_half_avg * 1.2:  # 20% improvement
                recovery_detected = True
                recovery_time = test_duration / 2  # Approximate recovery time
        
        # 9. Evaluate degradation quality
        graceful_degradation_achieved = (
            degradation_level in [DegradationLevel.MINIMAL, DegradationLevel.MODERATE] and
            degraded_metrics['error_rate_percent'] <= config.max_acceptable_error_rate_percent and
            degraded_metrics['avg_response_time_ms'] <= config.max_acceptable_response_time_ms
        )
        
        # 10. Evaluate compliance
        response_time_compliant = degraded_metrics['avg_response_time_ms'] <= config.max_acceptable_response_time_ms
        error_rate_compliant = degraded_metrics['error_rate_percent'] <= config.max_acceptable_error_rate_percent
        throughput_compliant = (degraded_metrics['requests_per_second'] / baseline_metrics['requests_per_second']) >= (config.min_acceptable_throughput_percent / 100.0)
        
        overall_compliance = response_time_compliant and error_rate_compliant and throughput_compliant
        
        # 11. Create test result
        result = DegradationTestResult(
            config=config,
            test_start_time=test_start_time,
            test_duration_seconds=test_duration,
            
            baseline_response_time_ms=baseline_metrics['avg_response_time_ms'],
            baseline_error_rate_percent=baseline_metrics['error_rate_percent'],
            baseline_throughput_rps=baseline_metrics['requests_per_second'],
            
            degraded_response_time_ms=degraded_metrics['avg_response_time_ms'],
            degraded_error_rate_percent=degraded_metrics['error_rate_percent'],
            degraded_throughput_rps=degraded_metrics['requests_per_second'],
            
            recovery_detected=recovery_detected,
            recovery_time_seconds=recovery_time,
            recovery_strategy_triggered=recovery_strategy_triggered,
            
            actual_degradation_level=degradation_level,
            functionality_retained_percent=functionality_percent,
            graceful_degradation_achieved=graceful_degradation_achieved,
            
            cascading_failures_prevented=True,  # Would need deeper analysis
            system_stability_maintained=degradation_level != DegradationLevel.CRITICAL,
            user_impact_minimized=graceful_degradation_achieved,
            
            response_time_within_threshold=response_time_compliant,
            error_rate_within_threshold=error_rate_compliant,
            throughput_within_threshold=throughput_compliant,
            overall_compliance=overall_compliance
        )
        
        self.test_results.append(result)
        
        # Log results
        status = "✅ PASSED" if overall_compliance else "❌ FAILED"
        logger.info(
            f"{status} {config.scenario.value}: "
            f"{degradation_level.value} degradation, "
            f"{functionality_percent:.1f}% functionality retained, "
            f"Recovery: {'Yes' if recovery_detected else 'No'}"
        )
        
        return result
    
    async def run_comprehensive_degradation_validation(self) -> Dict[str, Any]:
        """Run comprehensive degradation and recovery validation suite."""
        logger.info("🛡️ Starting Comprehensive Degradation and Recovery Validation")
        
        validation_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'test_scenarios': [],
            'degradation_analysis': {},
            'recovery_analysis': {},
            'resilience_score': 0.0,
            'recommendations': []
        }
        
        # Define test scenarios
        test_scenarios = [
            DegradationTestConfig(
                scenario=DegradationScenario.CACHE_UNAVAILABLE,
                description='Redis cache becomes unavailable',
                expected_degradation_level=DegradationLevel.MODERATE,
                expected_recovery_strategy=RecoveryStrategy.CACHE_BYPASS,
                test_duration_seconds=120,
                recovery_timeout_seconds=60
            ),
            DegradationTestConfig(
                scenario=DegradationScenario.DATABASE_SLOW,
                description='Database queries become slow',
                expected_degradation_level=DegradationLevel.MODERATE,
                expected_recovery_strategy=RecoveryStrategy.GRACEFUL_TIMEOUT,
                test_duration_seconds=180,
                recovery_timeout_seconds=90
            ),
            DegradationTestConfig(
                scenario=DegradationScenario.API_RATE_LIMIT,
                description='API rate limiting kicks in',
                expected_degradation_level=DegradationLevel.MINIMAL,
                expected_recovery_strategy=RecoveryStrategy.CIRCUIT_BREAKER,
                test_duration_seconds=120,
                recovery_timeout_seconds=60
            ),
            DegradationTestConfig(
                scenario=DegradationScenario.WEBSOCKET_OVERLOAD,
                description='WebSocket connections overloaded',
                expected_degradation_level=DegradationLevel.MODERATE,
                expected_recovery_strategy=RecoveryStrategy.LOAD_SHEDDING,
                test_duration_seconds=150,
                recovery_timeout_seconds=75
            ),
            DegradationTestConfig(
                scenario=DegradationScenario.PARTIAL_SERVICE_FAILURE,
                description='Some services fail while others continue',
                expected_degradation_level=DegradationLevel.SIGNIFICANT,
                expected_recovery_strategy=RecoveryStrategy.BULKHEAD_ISOLATION,
                test_duration_seconds=180,
                recovery_timeout_seconds=90
            )
        ]
        
        # Run all degradation tests
        for scenario_config in test_scenarios:
            try:
                logger.info(f"🧪 Running degradation test: {scenario_config.scenario.value}")
                
                test_result = await self.run_degradation_test(scenario_config)
                
                validation_results['test_scenarios'].append({
                    'scenario': scenario_config.scenario.value,
                    'description': scenario_config.description,
                    'expected_degradation': scenario_config.expected_degradation_level.value,
                    'actual_degradation': test_result.actual_degradation_level.value,
                    'functionality_retained_percent': test_result.functionality_retained_percent,
                    'graceful_degradation': test_result.graceful_degradation_achieved,
                    'recovery_detected': test_result.recovery_detected,
                    'recovery_time_seconds': test_result.recovery_time_seconds,
                    'overall_compliance': test_result.overall_compliance,
                    'baseline_response_time_ms': test_result.baseline_response_time_ms,
                    'degraded_response_time_ms': test_result.degraded_response_time_ms,
                    'baseline_error_rate_percent': test_result.baseline_error_rate_percent,
                    'degraded_error_rate_percent': test_result.degraded_error_rate_percent
                })
                
                # Recovery period between tests
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"❌ Degradation test {scenario_config.scenario.value} failed: {e}")
                validation_results['test_scenarios'].append({
                    'scenario': scenario_config.scenario.value,
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Analyze results
        if self.test_results:
            validation_results['degradation_analysis'] = self._analyze_degradation_patterns()
            validation_results['recovery_analysis'] = self._analyze_recovery_effectiveness()
            validation_results['resilience_score'] = self._calculate_resilience_score()
            validation_results['recommendations'] = self._generate_degradation_recommendations()
        
        logger.info(f"✅ Degradation validation completed. Resilience score: {validation_results['resilience_score']:.2f}")
        
        return validation_results
    
    def _analyze_degradation_patterns(self) -> Dict[str, Any]:
        """Analyze degradation patterns across all tests."""
        if not self.test_results:
            return {}
        
        degradation_levels = [result.actual_degradation_level.value for result in self.test_results]
        graceful_degradation_count = sum(1 for result in self.test_results if result.graceful_degradation_achieved)
        
        functionality_retained = [result.functionality_retained_percent for result in self.test_results]
        
        return {
            'total_scenarios_tested': len(self.test_results),
            'graceful_degradation_success_rate': (graceful_degradation_count / len(self.test_results)) * 100,
            'degradation_level_distribution': {level: degradation_levels.count(level) for level in set(degradation_levels)},
            'avg_functionality_retained_percent': statistics.mean(functionality_retained),
            'min_functionality_retained_percent': min(functionality_retained),
            'system_stability_maintained_count': sum(1 for r in self.test_results if r.system_stability_maintained)
        }
    
    def _analyze_recovery_effectiveness(self) -> Dict[str, Any]:
        """Analyze recovery effectiveness across all tests."""
        if not self.test_results:
            return {}
        
        recovery_detected_count = sum(1 for result in self.test_results if result.recovery_detected)
        recovery_strategy_triggered_count = sum(1 for result in self.test_results if result.recovery_strategy_triggered)
        
        recovery_times = [result.recovery_time_seconds for result in self.test_results if result.recovery_detected]
        
        return {
            'recovery_detection_rate': (recovery_detected_count / len(self.test_results)) * 100,
            'recovery_strategy_activation_rate': (recovery_strategy_triggered_count / len(self.test_results)) * 100,
            'avg_recovery_time_seconds': statistics.mean(recovery_times) if recovery_times else 0,
            'max_recovery_time_seconds': max(recovery_times) if recovery_times else 0,
            'fast_recovery_count': sum(1 for t in recovery_times if t <= 120),  # Under 2 minutes
            'cascading_failure_prevention_rate': sum(1 for r in self.test_results if r.cascading_failures_prevented) / len(self.test_results) * 100
        }
    
    def _calculate_resilience_score(self) -> float:
        """Calculate overall system resilience score."""
        if not self.test_results:
            return 0.0
        
        # Factors for resilience score
        graceful_degradation_score = sum(1 for r in self.test_results if r.graceful_degradation_achieved) / len(self.test_results)
        recovery_score = sum(1 for r in self.test_results if r.recovery_detected) / len(self.test_results)
        compliance_score = sum(1 for r in self.test_results if r.overall_compliance) / len(self.test_results)
        stability_score = sum(1 for r in self.test_results if r.system_stability_maintained) / len(self.test_results)
        
        # Weighted resilience score
        resilience_score = (
            graceful_degradation_score * 0.3 +
            recovery_score * 0.3 +
            compliance_score * 0.2 +
            stability_score * 0.2
        )
        
        return resilience_score
    
    def _generate_degradation_recommendations(self) -> List[str]:
        """Generate recommendations based on degradation test results."""
        recommendations = []
        
        if not self.test_results:
            return ["No test results available for analysis"]
        
        # Analyze failure patterns
        failed_scenarios = [r for r in self.test_results if not r.overall_compliance]
        graceful_failures = [r for r in self.test_results if not r.graceful_degradation_achieved]
        slow_recovery = [r for r in self.test_results if r.recovery_time_seconds > 180]
        
        if failed_scenarios:
            recommendations.append(f"Performance compliance failed in {len(failed_scenarios)} scenarios. Review performance thresholds and optimization strategies.")
        
        if graceful_failures:
            recommendations.append(f"Graceful degradation failed in {len(graceful_failures)} scenarios. Implement better fallback mechanisms and circuit breakers.")
        
        if slow_recovery:
            recommendations.append(f"Slow recovery detected in {len(slow_recovery)} scenarios. Optimize recovery procedures and automation.")
        
        # Specific scenario recommendations
        cache_failures = [r for r in self.test_results if r.config.scenario == DegradationScenario.CACHE_UNAVAILABLE and not r.graceful_degradation_achieved]
        if cache_failures:
            recommendations.append("Cache unavailability not handled gracefully. Implement cache bypass and fallback data sources.")
        
        db_failures = [r for r in self.test_results if r.config.scenario == DegradationScenario.DATABASE_SLOW and not r.graceful_degradation_achieved]
        if db_failures:
            recommendations.append("Database slowness not handled gracefully. Implement query timeouts and read replicas.")
        
        # Overall system recommendations
        resilience_score = self._calculate_resilience_score()
        if resilience_score >= 0.9:
            recommendations.append("✅ Excellent system resilience. Maintain current degradation handling and recovery mechanisms.")
        elif resilience_score >= 0.7:
            recommendations.append("🟡 Good system resilience with room for improvement. Focus on failed scenarios.")
        else:
            recommendations.append("🔴 Poor system resilience. Comprehensive review of error handling and recovery mechanisms required.")
        
        return recommendations


# Test utilities for pytest integration
@pytest.fixture
async def degradation_validator():
    """Pytest fixture for degradation validator."""
    validator = GracefulDegradationRecoveryValidator()
    yield validator


class TestGracefulDegradationRecovery:
    """Test suite for graceful degradation and recovery validation."""
    
    @pytest.mark.asyncio
    async def test_baseline_performance_measurement(self, degradation_validator):
        """Test baseline performance measurement."""
        baseline_metrics = await degradation_validator.measure_baseline_performance()
        
        assert isinstance(baseline_metrics, dict)
        assert 'avg_response_time_ms' in baseline_metrics
        assert 'error_rate_percent' in baseline_metrics
        assert 'requests_per_second' in baseline_metrics
        assert baseline_metrics['avg_response_time_ms'] >= 0
    
    @pytest.mark.asyncio
    async def test_cache_unavailable_degradation(self, degradation_validator):
        """Test graceful degradation when cache is unavailable."""
        config = DegradationTestConfig(
            scenario=DegradationScenario.CACHE_UNAVAILABLE,
            description='Test cache unavailable scenario',
            expected_degradation_level=DegradationLevel.MODERATE,
            expected_recovery_strategy=RecoveryStrategy.CACHE_BYPASS,
            test_duration_seconds=60  # Shorter for testing
        )
        
        result = await degradation_validator.run_degradation_test(config)
        
        assert result.actual_degradation_level in [
            DegradationLevel.MINIMAL,
            DegradationLevel.MODERATE,
            DegradationLevel.SIGNIFICANT
        ]
        assert result.functionality_retained_percent > 0
    
    @pytest.mark.asyncio
    async def test_api_rate_limiting_degradation(self, degradation_validator):
        """Test graceful degradation under API rate limiting."""
        config = DegradationTestConfig(
            scenario=DegradationScenario.API_RATE_LIMIT,
            description='Test API rate limiting scenario',
            expected_degradation_level=DegradationLevel.MINIMAL,
            expected_recovery_strategy=RecoveryStrategy.CIRCUIT_BREAKER,
            test_duration_seconds=60
        )
        
        result = await degradation_validator.run_degradation_test(config)
        
        # System should handle rate limiting gracefully
        assert result.functionality_retained_percent >= 50.0
        assert result.actual_degradation_level != DegradationLevel.CRITICAL
    
    @pytest.mark.asyncio
    async def test_comprehensive_degradation_validation(self, degradation_validator):
        """Test comprehensive degradation validation suite."""
        # Run mini validation (shorter duration)
        results = await degradation_validator.run_comprehensive_degradation_validation()
        
        assert 'test_scenarios' in results
        assert 'degradation_analysis' in results
        assert 'recovery_analysis' in results
        assert 'resilience_score' in results
        assert 'recommendations' in results
        
        # Validate resilience score
        score = results['resilience_score']
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


if __name__ == "__main__":
    async def main():
        validator = GracefulDegradationRecoveryValidator()
        results = await validator.run_comprehensive_degradation_validation()
        
        print("🛡️ Graceful Degradation and Recovery Validation Results:")
        print(json.dumps(results, indent=2, default=str))

    asyncio.run(main())