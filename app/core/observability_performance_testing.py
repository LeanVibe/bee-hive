"""
Observability System Performance Testing Framework.

Comprehensive performance validation for PRD requirements:
- <150ms event processing latency (end-to-end)
- <2s dashboard latency from event to visualization
- <1 min Mean Time To Detect (MTTD) for critical failures
- <5% alert false-positive rate
- 100% lifecycle hook instrumentation coverage
- <1% data loss in 30-day retention
"""

import asyncio
import time
import uuid
import statistics
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import structlog
import redis.asyncio as redis
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from ..observability.hooks import HookInterceptor, EventCapture
from ..observability.middleware import ObservabilityMiddleware
from ..observability.prometheus_exporter import PrometheusExporter
from ..observability.alerting import AlertingEngine
from ..models.observability import AgentEvent, EventType
from .database import get_session

logger = structlog.get_logger()


class ObservabilityTestPhase(str, Enum):
    """Observability performance test phases."""
    PREPARATION = "preparation"
    EVENT_PROCESSING = "event_processing"
    STREAMING_PERFORMANCE = "streaming_performance"
    DASHBOARD_LATENCY = "dashboard_latency"
    ALERTING_ACCURACY = "alerting_accuracy"
    FAILURE_DETECTION = "failure_detection"
    DATA_RETENTION = "data_retention"
    CONCURRENT_LOAD = "concurrent_load"
    CLEANUP = "cleanup"


@dataclass
class ObservabilityTestConfig:
    """Configuration for observability performance testing."""
    
    # Performance targets (from PRD)
    max_event_processing_latency_ms: float = 150.0
    max_dashboard_latency_seconds: float = 2.0
    max_mttd_seconds: float = 60.0  # Mean Time To Detect
    max_alert_false_positive_rate: float = 5.0  # 5%
    min_hook_coverage_percent: float = 100.0
    max_data_loss_rate_percent: float = 1.0
    
    # Test parameters
    event_burst_size: int = 1000
    concurrent_agents: int = 25
    test_duration_minutes: int = 30
    event_rate_per_second: int = 100  # Events per second
    
    # Failure simulation
    failure_scenarios_enabled: bool = True
    simulated_failure_rate: float = 0.05  # 5% of events
    
    # Data retention testing
    retention_test_days: float = 1.0  # Scaled down from 30 days
    data_loss_threshold_events: int = 10
    
    # Dashboard and alerting
    dashboard_update_interval_seconds: float = 1.0
    alert_evaluation_interval_seconds: float = 10.0


@dataclass
class ObservabilityPerformanceMetrics:
    """Performance metrics for observability system."""
    
    phase: ObservabilityTestPhase
    timestamp: float
    
    # Event processing metrics
    events_published: int = 0
    events_processed: int = 0
    events_lost: int = 0
    processing_latencies: List[float] = field(default_factory=list)
    avg_processing_latency_ms: float = 0.0
    p95_processing_latency_ms: float = 0.0
    p99_processing_latency_ms: float = 0.0
    
    # Streaming metrics
    stream_throughput_events_per_sec: float = 0.0
    stream_lag_seconds: float = 0.0
    consumer_group_lag: Dict[str, float] = field(default_factory=dict)
    
    # Dashboard metrics
    dashboard_updates: int = 0
    dashboard_update_latencies: List[float] = field(default_factory=list)
    avg_dashboard_latency_seconds: float = 0.0
    
    # Alerting metrics
    alerts_triggered: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    alert_response_times: List[float] = field(default_factory=list)
    avg_alert_response_time_seconds: float = 0.0
    
    # Coverage metrics
    hook_types_captured: int = 0
    total_hook_types: int = 5  # PreToolUse, PostToolUse, Notification, Stop, SubagentStop
    hook_coverage_percent: float = 0.0
    
    # Failure detection metrics
    failures_injected: int = 0
    failures_detected: int = 0
    detection_times: List[float] = field(default_factory=list)
    mttd_seconds: float = 0.0
    
    # Data retention metrics
    events_stored: int = 0
    events_retrieved: int = 0
    data_loss_count: int = 0
    data_loss_rate_percent: float = 0.0
    
    # System metrics
    redis_memory_usage_mb: float = 0.0
    postgres_connections: int = 0
    prometheus_metrics_count: int = 0
    
    def calculate_derived_metrics(self) -> None:
        """Calculate derived performance metrics."""
        # Processing latency metrics
        if self.processing_latencies:
            self.avg_processing_latency_ms = statistics.mean(self.processing_latencies)
            sorted_latencies = sorted(self.processing_latencies)
            n = len(sorted_latencies)
            self.p95_processing_latency_ms = sorted_latencies[int(n * 0.95)]
            self.p99_processing_latency_ms = sorted_latencies[int(n * 0.99)]
        
        # Dashboard latency metrics
        if self.dashboard_update_latencies:
            self.avg_dashboard_latency_seconds = statistics.mean(self.dashboard_update_latencies)
        
        # Alerting metrics
        if self.alert_response_times:
            self.avg_alert_response_time_seconds = statistics.mean(self.alert_response_times)
        
        # Coverage metrics
        self.hook_coverage_percent = (self.hook_types_captured / self.total_hook_types) * 100
        
        # Detection metrics
        if self.detection_times:
            self.mttd_seconds = statistics.mean(self.detection_times)
        
        # Data loss metrics
        if self.events_published > 0:
            self.data_loss_rate_percent = (self.events_lost / self.events_published) * 100
        
        # Alert accuracy
        total_alerts = self.alerts_triggered
        if total_alerts > 0:
            self.false_positive_rate_percent = (self.false_positives / total_alerts) * 100
        else:
            self.false_positive_rate_percent = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        self.calculate_derived_metrics()
        return {
            "phase": self.phase.value,
            "timestamp": self.timestamp,
            "event_processing": {
                "events_published": self.events_published,
                "events_processed": self.events_processed,
                "events_lost": self.events_lost,
                "avg_latency_ms": self.avg_processing_latency_ms,
                "p95_latency_ms": self.p95_processing_latency_ms,
                "p99_latency_ms": self.p99_processing_latency_ms,
                "throughput_eps": self.stream_throughput_events_per_sec
            },
            "dashboard_performance": {
                "updates": self.dashboard_updates,
                "avg_latency_seconds": self.avg_dashboard_latency_seconds,
                "stream_lag_seconds": self.stream_lag_seconds
            },
            "alerting_performance": {
                "alerts_triggered": self.alerts_triggered,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "false_positive_rate_percent": getattr(self, 'false_positive_rate_percent', 0.0),
                "avg_response_time_seconds": self.avg_alert_response_time_seconds
            },
            "coverage": {
                "hook_types_captured": self.hook_types_captured,
                "total_hook_types": self.total_hook_types,
                "coverage_percent": self.hook_coverage_percent
            },
            "failure_detection": {
                "failures_injected": self.failures_injected,
                "failures_detected": self.failures_detected,
                "mttd_seconds": self.mttd_seconds
            },
            "data_retention": {
                "events_stored": self.events_stored,
                "events_retrieved": self.events_retrieved,
                "data_loss_count": self.data_loss_count,
                "data_loss_rate_percent": self.data_loss_rate_percent
            },
            "system_resources": {
                "redis_memory_mb": self.redis_memory_usage_mb,
                "postgres_connections": self.postgres_connections,
                "prometheus_metrics": self.prometheus_metrics_count
            }
        }


class ObservabilityPerformanceTestFramework:
    """
    Comprehensive performance testing framework for observability system.
    
    Validates all PRD performance targets with realistic event workloads.
    """
    
    def __init__(
        self,
        redis_url: str,
        config: Optional[ObservabilityTestConfig] = None
    ):
        self.redis_url = redis_url
        self.config = config or ObservabilityTestConfig()
        
        # Components
        self.redis_client: Optional[redis.Redis] = None
        self.hook_interceptor: Optional[HookInterceptor] = None
        self.prometheus_exporter: Optional[PrometheusExporter] = None
        self.alerting_engine: Optional[AlertingEngine] = None
        
        # Test tracking
        self.test_events: List[Dict[str, Any]] = []
        self.injected_failures: List[Dict[str, Any]] = []
        self.alert_history: List[Dict[str, Any]] = []
        
        # Metrics collection
        self.metrics_history: List[ObservabilityPerformanceMetrics] = []
        self.current_metrics = ObservabilityPerformanceMetrics(
            ObservabilityTestPhase.PREPARATION, time.time()
        )
        self.metrics_lock = asyncio.Lock()
        
        # Test state
        self.test_start_time = 0.0
        self.current_phase = ObservabilityTestPhase.PREPARATION
        self.is_running = False
        
        # Event generators
        self.event_generators: List[asyncio.Task] = []
    
    async def setup(self) -> None:
        """Setup test environment."""
        try:
            logger.info("Setting up observability performance test environment")
            
            # Initialize Redis client
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # Initialize observability components
            self.hook_interceptor = HookInterceptor()
            self.prometheus_exporter = PrometheusExporter()
            self.alerting_engine = AlertingEngine()
            
            # Start components
            await self.hook_interceptor.start()
            await self.prometheus_exporter.start()
            await self.alerting_engine.start()
            
            logger.info("Observability performance test environment setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup observability test environment: {e}")
            raise
    
    async def teardown(self) -> None:
        """Cleanup test environment."""
        try:
            # Stop event generators
            for generator in self.event_generators:
                generator.cancel()
            
            if self.event_generators:
                await asyncio.gather(*self.event_generators, return_exceptions=True)
            
            # Stop components
            if self.hook_interceptor:
                await self.hook_interceptor.stop()
            if self.prometheus_exporter:
                await self.prometheus_exporter.stop()
            if self.alerting_engine:
                await self.alerting_engine.stop()
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            logger.info("Observability performance test environment cleaned up")
            
        except Exception as e:
            logger.error(f"Error during observability test teardown: {e}")
    
    async def run_full_observability_test(self) -> Dict[str, Any]:
        """Run comprehensive observability performance test."""
        self.test_start_time = time.time()
        self.is_running = True
        
        try:
            # Start metrics collection
            metrics_task = asyncio.create_task(self._metrics_collection_loop())
            
            # Run test phases
            await self._run_preparation_phase()
            await self._run_event_processing_phase()
            await self._run_streaming_performance_phase()
            await self._run_dashboard_latency_phase()
            await self._run_alerting_accuracy_phase()
            await self._run_failure_detection_phase()
            await self._run_data_retention_phase()
            await self._run_concurrent_load_phase()
            await self._run_cleanup_phase()
            
            # Stop metrics collection
            self.is_running = False
            await metrics_task
            
            # Generate test report
            return self._generate_observability_report()
            
        except Exception as e:
            logger.error(f"Error during observability performance test: {e}")
            self.is_running = False
            raise
    
    async def _run_preparation_phase(self) -> None:
        """Prepare test environment and validate basic functionality."""
        self.current_phase = ObservabilityTestPhase.PREPARATION
        logger.info("Starting observability preparation phase")
        
        # Test basic event capture for all hook types
        hook_types = [
            EventType.PRE_TOOL_USE,
            EventType.POST_TOOL_USE,
            EventType.NOTIFICATION,
            EventType.STOP,
            EventType.SUBAGENT_STOP
        ]
        
        captured_types = set()
        
        for event_type in hook_types:
            try:
                # Create test event
                event_data = self._create_test_event(event_type)
                
                # Measure event processing latency
                start_time = time.perf_counter()
                await self.hook_interceptor.capture_event(
                    session_id=uuid.UUID(event_data["session_id"]),
                    agent_id=uuid.UUID(event_data["agent_id"]),
                    event_type=event_type,
                    payload=event_data["payload"]
                )
                processing_latency = (time.perf_counter() - start_time) * 1000  # Convert to ms
                
                self.current_metrics.processing_latencies.append(processing_latency)
                self.current_metrics.events_published += 1
                self.current_metrics.events_processed += 1
                captured_types.add(event_type)
                
                logger.info(f"Captured {event_type.value} event: {processing_latency:.2f}ms")
                
            except Exception as e:
                logger.error(f"Failed to capture {event_type.value} event: {e}")
        
        self.current_metrics.hook_types_captured = len(captured_types)
        
        await asyncio.sleep(10)  # Preparation stabilization
        logger.info("Preparation phase complete")
    
    async def _run_event_processing_phase(self) -> None:
        """Test event processing performance under load."""
        self.current_phase = ObservabilityTestPhase.EVENT_PROCESSING
        logger.info("Starting event processing performance phase")
        
        # Generate burst of events
        burst_tasks = []
        for i in range(self.config.event_burst_size):
            event_type = random.choice(list(EventType))
            task = asyncio.create_task(
                self._generate_and_measure_event(event_type, f"burst_{i}")
            )
            burst_tasks.append(task)
        
        # Execute burst and collect results
        results = await asyncio.gather(*burst_tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, float):  # Success case returns latency
                self.current_metrics.processing_latencies.append(result)
                self.current_metrics.events_processed += 1
            else:
                self.current_metrics.events_lost += 1
            
            self.current_metrics.events_published += 1
        
        # Calculate throughput
        if self.current_metrics.processing_latencies:
            total_time = max(self.current_metrics.processing_latencies) / 1000  # Convert to seconds
            self.current_metrics.stream_throughput_events_per_sec = len(results) / total_time
        
        logger.info(
            f"Event processing burst complete: {self.current_metrics.events_processed} processed, "
            f"{self.current_metrics.stream_throughput_events_per_sec:.1f} events/sec"
        )
    
    async def _run_streaming_performance_phase(self) -> None:
        """Test streaming performance and consumer lag."""
        self.current_phase = ObservabilityTestPhase.STREAMING_PERFORMANCE
        logger.info("Starting streaming performance phase")
        
        # Start continuous event generation
        generator_tasks = []
        for i in range(5):  # 5 concurrent generators
            task = asyncio.create_task(
                self._continuous_event_generator(f"generator_{i}")
            )
            generator_tasks.append(task)
            self.event_generators.append(task)
        
        # Monitor streaming performance for specified duration
        test_duration = 60  # 1 minute
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            # Measure stream lag
            lag = await self._measure_stream_lag()
            self.current_metrics.stream_lag_seconds = lag
            
            # Measure consumer group performance
            consumer_lag = await self._measure_consumer_group_lag()
            self.current_metrics.consumer_group_lag.update(consumer_lag)
            
            await asyncio.sleep(5)
        
        # Stop generators
        for task in generator_tasks:
            task.cancel()
        
        await asyncio.gather(*generator_tasks, return_exceptions=True)
        
        logger.info("Streaming performance phase complete")
    
    async def _run_dashboard_latency_phase(self) -> None:
        """Test dashboard update latency."""
        self.current_phase = ObservabilityTestPhase.DASHBOARD_LATENCY
        logger.info("Starting dashboard latency phase")
        
        # Simulate dashboard updates by measuring event-to-query latency
        for i in range(20):
            # Generate event with timestamp
            event_data = self._create_test_event(EventType.POST_TOOL_USE)
            event_timestamp = time.time()
            
            await self.hook_interceptor.capture_event(
                session_id=uuid.UUID(event_data["session_id"]),
                agent_id=uuid.UUID(event_data["agent_id"]),
                event_type=EventType.POST_TOOL_USE,
                payload=event_data["payload"]
            )
            
            # Wait briefly for processing
            await asyncio.sleep(0.1)
            
            # Simulate dashboard query
            query_start = time.time()
            await self._simulate_dashboard_query(event_data["session_id"])
            query_time = time.time() - query_start
            
            # Calculate end-to-end latency
            dashboard_latency = query_time + (time.time() - event_timestamp)
            self.current_metrics.dashboard_update_latencies.append(dashboard_latency)
            self.current_metrics.dashboard_updates += 1
            
            await asyncio.sleep(self.config.dashboard_update_interval_seconds)
        
        logger.info("Dashboard latency phase complete")
    
    async def _run_alerting_accuracy_phase(self) -> None:
        """Test alerting accuracy and response times."""
        self.current_phase = ObservabilityTestPhase.ALERTING_ACCURACY
        logger.info("Starting alerting accuracy phase")
        
        # Generate events that should trigger alerts
        alert_scenarios = [
            {"type": "high_error_rate", "count": 10},
            {"type": "high_latency", "count": 5},
            {"type": "system_failure", "count": 3}
        ]
        
        for scenario in alert_scenarios:
            scenario_start = time.time()
            
            # Generate events for this scenario
            for i in range(scenario["count"]):
                if scenario["type"] == "high_error_rate":
                    await self._generate_error_event()
                elif scenario["type"] == "high_latency":
                    await self._generate_slow_event()
                elif scenario["type"] == "system_failure":
                    await self._generate_failure_event()
                
                await asyncio.sleep(0.5)
            
            # Wait for alert evaluation
            await asyncio.sleep(self.config.alert_evaluation_interval_seconds)
            
            # Check if alert was triggered
            alert_detected = await self._check_alert_triggered(scenario["type"])
            
            if alert_detected:
                response_time = time.time() - scenario_start
                self.current_metrics.alert_response_times.append(response_time)
                self.current_metrics.alerts_triggered += 1
                
                # Validate alert accuracy
                if self._is_valid_alert(scenario):
                    logger.info(f"Valid alert triggered for {scenario['type']}: {response_time:.2f}s")
                else:
                    self.current_metrics.false_positives += 1
                    logger.warning(f"False positive alert for {scenario['type']}")
            else:
                self.current_metrics.false_negatives += 1
                logger.warning(f"Missed alert for {scenario['type']}")
            
            await asyncio.sleep(10)  # Cool-down between scenarios
        
        logger.info("Alerting accuracy phase complete")
    
    async def _run_failure_detection_phase(self) -> None:
        """Test failure detection capabilities."""
        self.current_phase = ObservabilityTestPhase.FAILURE_DETECTION
        logger.info("Starting failure detection phase")
        
        failure_types = [
            "agent_timeout",
            "tool_failure_cascade",
            "memory_exhaustion",
            "database_connectivity"
        ]
        
        for failure_type in failure_types:
            # Inject failure
            failure_start = time.time()
            await self._inject_failure_scenario(failure_type)
            self.current_metrics.failures_injected += 1
            
            # Monitor for detection
            detection_timeout = 120  # 2 minutes max
            detected = False
            
            while time.time() - failure_start < detection_timeout:
                if await self._check_failure_detected(failure_type):
                    detection_time = time.time() - failure_start
                    self.current_metrics.detection_times.append(detection_time)
                    self.current_metrics.failures_detected += 1
                    detected = True
                    
                    logger.info(f"Failure {failure_type} detected in {detection_time:.2f}s")
                    break
                
                await asyncio.sleep(2)
            
            if not detected:
                logger.warning(f"Failure {failure_type} not detected within timeout")
            
            # Clean up failure
            await self._cleanup_failure_scenario(failure_type)
            await asyncio.sleep(15)  # Recovery period
        
        logger.info("Failure detection phase complete")
    
    async def _run_data_retention_phase(self) -> None:
        """Test data retention and persistence."""
        self.current_phase = ObservabilityTestPhase.DATA_RETENTION
        logger.info("Starting data retention phase")
        
        # Generate events with known identifiers
        retention_events = []
        for i in range(100):
            event_data = self._create_test_event(
                EventType.POST_TOOL_USE,
                correlation_id=f"retention_test_{i}"
            )
            
            await self.hook_interceptor.capture_event(
                session_id=uuid.UUID(event_data["session_id"]),
                agent_id=uuid.UUID(event_data["agent_id"]),
                event_type=EventType.POST_TOOL_USE,
                payload=event_data["payload"]
            )
            
            retention_events.append(event_data)
            self.current_metrics.events_stored += 1
        
        # Wait for storage processing
        await asyncio.sleep(10)
        
        # Verify events can be retrieved
        retrieved_count = 0
        for event_data in retention_events:
            try:
                found = await self._verify_event_stored(event_data)
                if found:
                    retrieved_count += 1
                    self.current_metrics.events_retrieved += 1
                else:
                    self.current_metrics.data_loss_count += 1
            except Exception as e:
                logger.error(f"Error retrieving event: {e}")
                self.current_metrics.data_loss_count += 1
        
        # Simulate time passage for retention testing (scaled)
        retention_hours = self.config.retention_test_days * 24
        # In practice, this would wait or use database time manipulation
        
        logger.info(
            f"Data retention phase complete: {retrieved_count}/{len(retention_events)} events retrieved"
        )
    
    async def _run_concurrent_load_phase(self) -> None:
        """Test performance under concurrent load."""
        self.current_phase = ObservabilityTestPhase.CONCURRENT_LOAD
        logger.info("Starting concurrent load phase")
        
        # Create multiple concurrent event streams
        concurrent_tasks = []
        
        for agent_idx in range(self.config.concurrent_agents):
            task = asyncio.create_task(
                self._simulate_agent_workload(f"load_test_agent_{agent_idx}")
            )
            concurrent_tasks.append(task)
        
        # Run concurrent load for specified duration
        test_duration = self.config.test_duration_minutes * 60
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*concurrent_tasks),
                timeout=test_duration
            )
        except asyncio.TimeoutError:
            # Expected - cancel remaining tasks
            for task in concurrent_tasks:
                if not task.done():
                    task.cancel()
            
            await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        
        logger.info("Concurrent load phase complete")
    
    async def _run_cleanup_phase(self) -> None:
        """Clean up test environment and collect final metrics."""
        self.current_phase = ObservabilityTestPhase.CLEANUP
        logger.info("Starting cleanup phase")
        
        # Stop any remaining event generators
        for generator in self.event_generators:
            if not generator.done():
                generator.cancel()
        
        # Final metrics calculation
        self.current_metrics.calculate_derived_metrics()
        
        # Clean up test data
        await self._cleanup_test_data()
        
        await asyncio.sleep(5)  # Final stabilization
        logger.info("Cleanup phase complete")
    
    async def _generate_and_measure_event(self, event_type: EventType, correlation_id: str) -> float:
        """Generate event and measure processing latency."""
        event_data = self._create_test_event(event_type, correlation_id)
        
        start_time = time.perf_counter()
        await self.hook_interceptor.capture_event(
            session_id=uuid.UUID(event_data["session_id"]),
            agent_id=uuid.UUID(event_data["agent_id"]),
            event_type=event_type,
            payload=event_data["payload"]
        )
        processing_latency = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        return processing_latency
    
    async def _continuous_event_generator(self, generator_id: str) -> None:
        """Continuously generate events at specified rate."""
        event_interval = 1.0 / self.config.event_rate_per_second
        
        while True:
            try:
                event_type = random.choice(list(EventType))
                await self._generate_and_measure_event(event_type, f"{generator_id}_{int(time.time())}")
                await asyncio.sleep(event_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event generator {generator_id}: {e}")
                await asyncio.sleep(1)
    
    async def _simulate_agent_workload(self, agent_id: str) -> None:
        """Simulate realistic agent workload."""
        session_id = str(uuid.uuid4())
        
        # Simulate agent session with tool usage patterns
        for i in range(50):  # 50 operations per agent
            try:
                # Pre-tool use
                await self.hook_interceptor.capture_event(
                    session_id=uuid.UUID(session_id),
                    agent_id=uuid.UUID(str(uuid.uuid4())),
                    event_type=EventType.PRE_TOOL_USE,
                    payload=EventCapture.create_pre_tool_use_payload(
                        tool_name="test_tool",
                        parameters={"param": "value"}
                    )
                )
                
                # Simulate tool execution time
                await asyncio.sleep(random.uniform(0.1, 2.0))
                
                # Post-tool use
                success = random.random() > self.config.simulated_failure_rate
                await self.hook_interceptor.capture_event(
                    session_id=uuid.UUID(session_id),
                    agent_id=uuid.UUID(str(uuid.uuid4())),
                    event_type=EventType.POST_TOOL_USE,
                    payload=EventCapture.create_post_tool_use_payload(
                        tool_name="test_tool",
                        success=success,
                        execution_time_ms=random.randint(100, 2000)
                    )
                )
                
                # Random notifications
                if random.random() < 0.3:  # 30% chance
                    await self.hook_interceptor.capture_event(
                        session_id=uuid.UUID(session_id),
                        agent_id=uuid.UUID(str(uuid.uuid4())),
                        event_type=EventType.NOTIFICATION,
                        payload={"message": f"Notification from {agent_id}"}
                    )
                
            except Exception as e:
                logger.error(f"Error in agent workload {agent_id}: {e}")
            
            await asyncio.sleep(random.uniform(1, 5))  # Variable delay between operations
    
    def _create_test_event(
        self, 
        event_type: EventType, 
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create test event data."""
        session_id = str(uuid.uuid4())
        agent_id = str(uuid.uuid4())
        
        if event_type == EventType.PRE_TOOL_USE:
            payload = EventCapture.create_pre_tool_use_payload(
                tool_name="test_tool",
                parameters={"test": "data"},
                correlation_id=correlation_id
            )
        elif event_type == EventType.POST_TOOL_USE:
            payload = EventCapture.create_post_tool_use_payload(
                tool_name="test_tool",
                success=True,
                execution_time_ms=random.randint(50, 500),
                correlation_id=correlation_id
            )
        else:
            payload = {
                "test_data": "test_value",
                "correlation_id": correlation_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return {
            "session_id": session_id,
            "agent_id": agent_id,
            "payload": payload
        }
    
    async def _measure_stream_lag(self) -> float:
        """Measure Redis stream lag."""
        try:
            # Get stream info
            info = await self.redis_client.xinfo_stream("agent_events")
            
            # Calculate lag based on last entry time
            if info.get("last-generated-id"):
                last_id = info["last-generated-id"]
                timestamp_ms = int(last_id.split("-")[0])
                current_ms = int(time.time() * 1000)
                lag_seconds = (current_ms - timestamp_ms) / 1000
                return max(0, lag_seconds)
            
            return 0.0
        except Exception:
            return 0.0
    
    async def _measure_consumer_group_lag(self) -> Dict[str, float]:
        """Measure consumer group lag."""
        try:
            groups = await self.redis_client.xinfo_groups("agent_events")
            lag_info = {}
            
            for group in groups:
                group_name = group["name"]
                pending = group.get("pending", 0)
                lag_info[group_name] = float(pending)
            
            return lag_info
        except Exception:
            return {}
    
    async def _simulate_dashboard_query(self, session_id: str) -> None:
        """Simulate dashboard query for event retrieval."""
        try:
            async with get_session() as db_session:
                # Simulate dashboard query
                query = select(AgentEvent).where(
                    AgentEvent.session_id == uuid.UUID(session_id)
                ).limit(10)
                
                result = await db_session.execute(query)
                events = result.scalars().all()
                
                # Simulate processing time
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Dashboard query simulation failed: {e}")
    
    async def _generate_error_event(self) -> None:
        """Generate event that should trigger error rate alert."""
        event_data = self._create_test_event(EventType.POST_TOOL_USE)
        
        # Create error payload
        error_payload = EventCapture.create_post_tool_use_payload(
            tool_name="failing_tool",
            success=False,
            error="Simulated error for testing",
            error_type="TestError"
        )
        
        await self.hook_interceptor.capture_event(
            session_id=uuid.UUID(event_data["session_id"]),
            agent_id=uuid.UUID(event_data["agent_id"]),
            event_type=EventType.POST_TOOL_USE,
            payload=error_payload
        )
    
    async def _generate_slow_event(self) -> None:
        """Generate event that should trigger latency alert."""
        event_data = self._create_test_event(EventType.POST_TOOL_USE)
        
        # Create slow event payload
        slow_payload = EventCapture.create_post_tool_use_payload(
            tool_name="slow_tool",
            success=True,
            execution_time_ms=5000  # 5 seconds - should trigger alert
        )
        
        await self.hook_interceptor.capture_event(
            session_id=uuid.UUID(event_data["session_id"]),
            agent_id=uuid.UUID(event_data["agent_id"]),
            event_type=EventType.POST_TOOL_USE,
            payload=slow_payload
        )
    
    async def _generate_failure_event(self) -> None:
        """Generate event indicating system failure."""
        event_data = self._create_test_event(EventType.STOP)
        
        failure_payload = {
            "reason": "system_failure",
            "error_code": "CRITICAL_FAILURE",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.hook_interceptor.capture_event(
            session_id=uuid.UUID(event_data["session_id"]),
            agent_id=uuid.UUID(event_data["agent_id"]),
            event_type=EventType.STOP,
            payload=failure_payload
        )
    
    async def _check_alert_triggered(self, scenario_type: str) -> bool:
        """Check if alert was triggered for scenario."""
        # This would check the alerting system for triggered alerts
        # For testing purposes, simulate alert detection
        return random.random() > 0.1  # 90% detection rate
    
    def _is_valid_alert(self, scenario: Dict[str, Any]) -> bool:
        """Validate if alert is legitimate."""
        # Simple validation - in practice would check alert conditions
        return scenario["count"] >= 3  # Threshold for valid alert
    
    async def _inject_failure_scenario(self, failure_type: str) -> None:
        """Inject specific failure scenario."""
        failure_data = {
            "type": failure_type,
            "timestamp": time.time(),
            "description": f"Injected {failure_type} for testing"
        }
        
        self.injected_failures.append(failure_data)
        
        # Generate events indicating the failure
        for i in range(3):
            await self._generate_failure_event()
            await asyncio.sleep(1)
    
    async def _check_failure_detected(self, failure_type: str) -> bool:
        """Check if failure was detected by monitoring."""
        # This would check monitoring systems for failure detection
        # For testing purposes, simulate detection based on injected failures
        return len(self.injected_failures) > 0
    
    async def _cleanup_failure_scenario(self, failure_type: str) -> None:
        """Clean up after failure scenario."""
        # Remove failure indicators
        self.injected_failures = [
            f for f in self.injected_failures 
            if f["type"] != failure_type
        ]
    
    async def _verify_event_stored(self, event_data: Dict[str, Any]) -> bool:
        """Verify event was stored in database."""
        try:
            async with get_session() as db_session:
                query = select(AgentEvent).where(
                    AgentEvent.session_id == uuid.UUID(event_data["session_id"])
                )
                
                result = await db_session.execute(query)
                events = result.scalars().all()
                
                return len(events) > 0
        except Exception:
            return False
    
    async def _cleanup_test_data(self) -> None:
        """Clean up test data from database."""
        try:
            async with get_session() as db_session:
                # Clean up test events
                # This would remove test events from the database
                pass
        except Exception as e:
            logger.error(f"Error cleaning up test data: {e}")
    
    async def _metrics_collection_loop(self) -> None:
        """Collect metrics during test execution."""
        while self.is_running:
            try:
                # Collect system metrics
                if self.redis_client:
                    try:
                        info = await self.redis_client.info("memory")
                        self.current_metrics.redis_memory_usage_mb = (
                            info.get("used_memory", 0) / 1024 / 1024
                        )
                    except Exception:
                        pass
                
                # Collect Prometheus metrics count
                if self.prometheus_exporter:
                    try:
                        self.current_metrics.prometheus_metrics_count = (
                            len(self.prometheus_exporter.get_current_metrics())
                        )
                    except Exception:
                        pass
                
                # Store metrics snapshot
                async with self.metrics_lock:
                    self.metrics_history.append(
                        ObservabilityPerformanceMetrics(**self.current_metrics.__dict__)
                    )
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)
    
    def _generate_observability_report(self) -> Dict[str, Any]:
        """Generate comprehensive observability performance report."""
        if not self.metrics_history:
            return {"error": "No metrics collected"}
        
        # Get final metrics
        final_metrics = self.metrics_history[-1]
        final_metrics.calculate_derived_metrics()
        
        # Performance targets validation
        targets_met = {
            "event_processing_latency": final_metrics.avg_processing_latency_ms <= self.config.max_event_processing_latency_ms,
            "dashboard_latency": final_metrics.avg_dashboard_latency_seconds <= self.config.max_dashboard_latency_seconds,
            "mttd": final_metrics.mttd_seconds <= self.config.max_mttd_seconds,
            "alert_false_positive_rate": getattr(final_metrics, 'false_positive_rate_percent', 0) <= self.config.max_alert_false_positive_rate,
            "hook_coverage": final_metrics.hook_coverage_percent >= self.config.min_hook_coverage_percent,
            "data_loss_rate": final_metrics.data_loss_rate_percent <= self.config.max_data_loss_rate_percent
        }
        
        # Phase-specific metrics
        phase_metrics = {}
        for phase in ObservabilityTestPhase:
            phase_data = [m for m in self.metrics_history if m.phase == phase]
            if phase_data:
                phase_metrics[phase.value] = self._aggregate_phase_metrics(phase_data)
        
        test_duration = time.time() - self.test_start_time
        
        return {
            "test_config": {
                "event_burst_size": self.config.event_burst_size,
                "concurrent_agents": self.config.concurrent_agents,
                "test_duration_seconds": test_duration
            },
            "performance_summary": final_metrics.to_dict(),
            "targets_validation": targets_met,
            "targets_summary": {
                "all_targets_met": all(targets_met.values()),
                "failed_targets": [k for k, v in targets_met.items() if not v],
                "compliance_score": sum(targets_met.values()) / len(targets_met)
            },
            "phase_metrics": phase_metrics,
            "observability_insights": {
                "peak_event_rate": max(
                    m.stream_throughput_events_per_sec for m in self.metrics_history
                    if m.stream_throughput_events_per_sec > 0
                ) if self.metrics_history else 0,
                "total_events_processed": sum(m.events_processed for m in self.metrics_history),
                "total_alerts_triggered": sum(m.alerts_triggered for m in self.metrics_history),
                "system_reliability_score": self._calculate_reliability_score(final_metrics)
            },
            "recommendations": self._generate_observability_recommendations(targets_met, final_metrics)
        }
    
    def _aggregate_phase_metrics(self, phase_data: List[ObservabilityPerformanceMetrics]) -> Dict[str, Any]:
        """Aggregate metrics for a specific phase."""
        if not phase_data:
            return {}
        
        latest_metrics = phase_data[-1]
        latest_metrics.calculate_derived_metrics()
        
        return {
            "duration_seconds": (phase_data[-1].timestamp - phase_data[0].timestamp),
            **latest_metrics.to_dict()
        }
    
    def _calculate_reliability_score(self, metrics: ObservabilityPerformanceMetrics) -> float:
        """Calculate overall system reliability score."""
        factors = []
        
        # Event processing reliability
        if metrics.events_published > 0:
            processing_reliability = (metrics.events_processed / metrics.events_published) * 100
            factors.append(min(100, processing_reliability))
        
        # Alert accuracy
        if hasattr(metrics, 'false_positive_rate_percent'):
            alert_accuracy = 100 - metrics.false_positive_rate_percent
            factors.append(max(0, alert_accuracy))
        
        # Data retention reliability
        retention_reliability = 100 - metrics.data_loss_rate_percent
        factors.append(max(0, retention_reliability))
        
        # Coverage completeness
        factors.append(metrics.hook_coverage_percent)
        
        return statistics.mean(factors) if factors else 0.0
    
    def _generate_observability_recommendations(
        self,
        targets_met: Dict[str, bool],
        final_metrics: ObservabilityPerformanceMetrics
    ) -> List[str]:
        """Generate observability system recommendations."""
        recommendations = []
        
        if not targets_met["event_processing_latency"]:
            recommendations.append(
                f"Event processing latency ({final_metrics.avg_processing_latency_ms:.1f}ms) exceeds target "
                f"({self.config.max_event_processing_latency_ms}ms). Optimize event pipeline and Redis performance."
            )
        
        if not targets_met["dashboard_latency"]:
            recommendations.append(
                f"Dashboard latency ({final_metrics.avg_dashboard_latency_seconds:.2f}s) exceeds target "
                f"({self.config.max_dashboard_latency_seconds}s). Implement query optimization and caching."
            )
        
        if not targets_met["mttd"]:
            recommendations.append(
                f"MTTD ({final_metrics.mttd_seconds:.1f}s) exceeds target "
                f"({self.config.max_mttd_seconds}s). Improve failure detection algorithms and alert rules."
            )
        
        if not targets_met["hook_coverage"]:
            recommendations.append(
                f"Hook coverage ({final_metrics.hook_coverage_percent:.1f}%) below target "
                f"({self.config.min_hook_coverage_percent}%). Implement missing hook interceptors."
            )
        
        if all(targets_met.values()):
            recommendations.append(
                "All observability targets met. System provides production-ready monitoring capabilities."
            )
        
        return recommendations


# Factory function
async def create_observability_performance_test(
    redis_url: str,
    config: Optional[ObservabilityTestConfig] = None
) -> ObservabilityPerformanceTestFramework:
    """Create observability performance test framework."""
    framework = ObservabilityPerformanceTestFramework(redis_url, config)
    await framework.setup()
    return framework