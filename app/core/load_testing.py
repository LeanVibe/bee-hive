"""
Load Testing Framework for Redis Streams Communication System.

Comprehensive performance testing to validate 10k+ msg/sec throughput 
and <200ms P95 latency requirements with realistic workload simulation.
"""

import asyncio
import time
import uuid
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import json

import structlog
import redis.asyncio as redis
from redis.asyncio import Redis

from ..models.message import StreamMessage, MessageType, MessagePriority
from .performance_optimizations import HighPerformanceMessageBroker, BatchConfig, CompressionConfig, ConnectionConfig
from .stream_monitor import StreamMonitor
from .backpressure_manager import BackPressureManager, BackPressureConfig
from ..core.config import settings

logger = structlog.get_logger()


class TestPhase(str, Enum):
    """Load test phases."""
    WARMUP = "warmup"
    RAMP_UP = "ramp_up" 
    STEADY_STATE = "steady_state"
    SPIKE = "spike"
    RAMP_DOWN = "ramp_down"
    COOLDOWN = "cooldown"


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    
    # Test duration and phases
    warmup_duration_seconds: int = 30
    ramp_up_duration_seconds: int = 60
    steady_state_duration_seconds: int = 300  # 5 minutes
    spike_duration_seconds: int = 60
    ramp_down_duration_seconds: int = 60
    cooldown_duration_seconds: int = 30
    
    # Load parameters
    target_messages_per_second: int = 10000
    spike_multiplier: float = 2.0  # 2x normal load during spike
    concurrent_producers: int = 50
    concurrent_consumers: int = 25
    
    # Message characteristics
    message_size_bytes: Tuple[int, int] = (100, 10000)  # min, max payload size
    message_types_distribution: Dict[MessageType, float] = field(default_factory=lambda: {
        MessageType.TASK_REQUEST: 0.4,
        MessageType.TASK_RESULT: 0.3,
        MessageType.EVENT: 0.2,
        MessageType.HEARTBEAT: 0.1
    })
    
    # Performance targets
    max_p95_latency_ms: float = 200.0
    max_p99_latency_ms: float = 500.0
    min_success_rate: float = 0.999  # 99.9%
    max_error_rate: float = 0.001  # 0.1%
    
    # Consumer behavior
    consumer_processing_time_ms: Tuple[int, int] = (10, 100)  # min, max processing time
    consumer_error_rate: float = 0.001  # 0.1% simulated failures
    
    # Streams configuration
    stream_count: int = 10
    stream_name_prefix: str = "load_test_stream"


@dataclass
class TestMetrics:
    """Metrics collected during load testing."""
    
    phase: TestPhase
    timestamp: float
    
    # Throughput metrics
    messages_sent: int = 0
    messages_received: int = 0
    messages_acknowledged: int = 0
    messages_failed: int = 0
    
    # Latency metrics (milliseconds)
    latencies: List[float] = field(default_factory=list)
    min_latency: float = 0.0
    max_latency: float = 0.0
    avg_latency: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    
    # Throughput rates
    send_rate_msg_per_sec: float = 0.0
    receive_rate_msg_per_sec: float = 0.0
    ack_rate_msg_per_sec: float = 0.0
    
    # Error rates
    error_rate: float = 0.0
    success_rate: float = 0.0
    
    # System metrics
    active_producers: int = 0
    active_consumers: int = 0
    queue_depths: Dict[str, int] = field(default_factory=dict)
    
    def calculate_derived_metrics(self, window_duration_seconds: float) -> None:
        """Calculate derived metrics from raw data."""
        if self.latencies:
            self.min_latency = min(self.latencies)
            self.max_latency = max(self.latencies)
            self.avg_latency = statistics.mean(self.latencies)
            self.p50_latency = statistics.median(self.latencies)
            
            sorted_latencies = sorted(self.latencies)
            n = len(sorted_latencies)
            self.p95_latency = sorted_latencies[int(n * 0.95)] if n > 0 else 0.0
            self.p99_latency = sorted_latencies[int(n * 0.99)] if n > 0 else 0.0
        
        # Calculate rates
        if window_duration_seconds > 0:
            self.send_rate_msg_per_sec = self.messages_sent / window_duration_seconds
            self.receive_rate_msg_per_sec = self.messages_received / window_duration_seconds
            self.ack_rate_msg_per_sec = self.messages_acknowledged / window_duration_seconds
        
        # Calculate success/error rates
        total_operations = self.messages_sent + self.messages_failed
        if total_operations > 0:
            self.success_rate = self.messages_acknowledged / total_operations
            self.error_rate = self.messages_failed / total_operations


class LoadTestProducer:
    """Producer worker for load testing."""
    
    def __init__(
        self,
        producer_id: str,
        broker: HighPerformanceMessageBroker,
        config: LoadTestConfig,
        metrics_callback: Callable[[str, float, bool], None]
    ):
        self.producer_id = producer_id
        self.broker = broker
        self.config = config
        self.metrics_callback = metrics_callback
        
        self.is_running = False
        self.messages_sent = 0
        self.current_rate = 0.0
        
    async def start(self, target_rate: float) -> None:
        """Start producing messages at target rate."""
        self.is_running = True
        self.current_rate = target_rate
        
        # Calculate delay between messages
        if target_rate > 0:
            delay_seconds = 1.0 / target_rate
        else:
            delay_seconds = 1.0
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # Create test message
                message = self._create_test_message()
                
                # Send message
                await self.broker.send_message(message)
                self.messages_sent += 1
                
                # Record latency
                latency_ms = (time.time() - start_time) * 1000
                self.metrics_callback(self.producer_id, latency_ms, True)
                
                # Wait for next message
                await asyncio.sleep(delay_seconds)
                
            except Exception as e:
                logger.error(f"Producer {self.producer_id} error: {e}")
                self.metrics_callback(self.producer_id, 0.0, False)
                await asyncio.sleep(0.1)  # Brief pause on error
    
    def stop(self) -> None:
        """Stop the producer."""
        self.is_running = False
    
    def update_rate(self, new_rate: float) -> None:
        """Update target message rate."""
        self.current_rate = new_rate
    
    def _create_test_message(self) -> StreamMessage:
        """Create a realistic test message."""
        # Select message type based on distribution
        message_type = self._select_message_type()
        
        # Generate payload of appropriate size
        payload_size = random.randint(*self.config.message_size_bytes)
        payload = self._generate_payload(message_type, payload_size)
        
        # Select target stream
        stream_suffix = random.randint(1, self.config.stream_count)
        target_agent = f"{self.config.stream_name_prefix}_{stream_suffix}"
        
        return StreamMessage(
            from_agent=self.producer_id,
            to_agent=target_agent,
            message_type=message_type,
            payload=payload,
            priority=MessagePriority.NORMAL
        )
    
    def _select_message_type(self) -> MessageType:
        """Select message type based on distribution."""
        rand = random.random()
        cumulative = 0.0
        
        for msg_type, probability in self.config.message_types_distribution.items():
            cumulative += probability
            if rand <= cumulative:
                return msg_type
        
        return MessageType.EVENT  # Fallback
    
    def _generate_payload(self, message_type: MessageType, target_size: int) -> Dict[str, Any]:
        """Generate realistic payload of target size."""
        base_payload = {
            "timestamp": time.time(),
            "producer_id": self.producer_id,
            "sequence": self.messages_sent,
            "message_type": message_type.value
        }
        
        if message_type == MessageType.TASK_REQUEST:
            base_payload.update({
                "task_id": str(uuid.uuid4()),
                "task_type": random.choice(["data_processing", "model_training", "analysis"]),
                "parameters": {"key": "value", "number": random.randint(1, 1000)}
            })
        elif message_type == MessageType.TASK_RESULT:
            base_payload.update({
                "task_id": str(uuid.uuid4()),
                "success": random.choice([True, True, True, False]),  # 75% success rate
                "result": {"output": "processed_data", "metrics": {"accuracy": random.random()}}
            })
        elif message_type == MessageType.EVENT:
            base_payload.update({
                "event_type": random.choice(["status_change", "alert", "notification"]),
                "severity": random.choice(["info", "warning", "error"]),
                "details": {"component": "agent", "status": "active"}
            })
        
        # Pad to target size with random data
        current_size = len(json.dumps(base_payload, separators=(',', ':')))
        if current_size < target_size:
            padding_size = target_size - current_size - 20  # Leave some buffer
            padding_data = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=max(0, padding_size)))
            base_payload["_padding"] = padding_data
        
        return base_payload


class LoadTestConsumer:
    """Consumer worker for load testing."""
    
    def __init__(
        self,
        consumer_id: str,
        redis_client: Redis,
        config: LoadTestConfig,
        metrics_callback: Callable[[str, bool], None]
    ):
        self.consumer_id = consumer_id
        self.redis = redis_client
        self.config = config
        self.metrics_callback = metrics_callback
        
        self.is_running = False
        self.messages_processed = 0
    
    async def start(self, streams: List[str]) -> None:
        """Start consuming from specified streams."""
        self.is_running = True
        
        # Create consumer groups for all streams
        for stream_name in streams:
            try:
                group_name = f"load_test_group_{stream_name}"
                await self.redis.xgroup_create(
                    stream_name, group_name, id='0', mkstream=True
                )
            except Exception:
                pass  # Group might already exist
        
        # Start consuming
        while self.is_running:
            try:
                await self._consume_batch(streams)
                await asyncio.sleep(0.01)  # Small delay between batches
                
            except Exception as e:
                logger.error(f"Consumer {self.consumer_id} error: {e}")
                await asyncio.sleep(0.1)
    
    def stop(self) -> None:
        """Stop the consumer."""
        self.is_running = False
    
    async def _consume_batch(self, streams: List[str]) -> None:
        """Consume a batch of messages from streams."""
        # Build stream reading dict
        stream_dict = {}
        for stream_name in streams:
            stream_dict[stream_name] = '>'
        
        # Try to read from all streams
        for stream_name in streams:
            try:
                group_name = f"load_test_group_{stream_name}"
                
                messages = await self.redis.xreadgroup(
                    group_name,
                    self.consumer_id,
                    {stream_name: '>'},
                    count=10,
                    block=100  # 100ms timeout
                )
                
                # Process messages
                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        await self._process_message(stream.decode(), msg_id.decode(), fields)
                        
            except Exception as e:
                if "NOGROUP" not in str(e):  # Ignore "no group" errors
                    logger.error(f"Error reading from {stream_name}: {e}")
    
    async def _process_message(self, stream_name: str, msg_id: str, fields: Dict) -> None:
        """Process a single message."""
        try:
            # Simulate processing time
            processing_time = random.uniform(*self.config.consumer_processing_time_ms) / 1000.0
            await asyncio.sleep(processing_time)
            
            # Simulate occasional errors
            if random.random() < self.config.consumer_error_rate:
                self.metrics_callback(self.consumer_id, False)
                return
            
            # Acknowledge message
            group_name = f"load_test_group_{stream_name}"
            await self.redis.xack(stream_name, group_name, msg_id)
            
            self.messages_processed += 1
            self.metrics_callback(self.consumer_id, True)
            
        except Exception as e:
            logger.error(f"Error processing message {msg_id}: {e}")
            self.metrics_callback(self.consumer_id, False)


class LoadTestFramework:
    """
    Comprehensive load testing framework for Redis Streams.
    
    Validates 10k+ msg/sec throughput and <200ms P95 latency requirements
    under realistic workload conditions.
    """
    
    def __init__(
        self,
        redis_url: str,
        config: Optional[LoadTestConfig] = None
    ):
        self.redis_url = redis_url
        self.config = config or LoadTestConfig()
        
        # Components
        self.redis_client: Optional[Redis] = None
        self.broker: Optional[HighPerformanceMessageBroker] = None
        self.monitor: Optional[StreamMonitor] = None
        self.backpressure_manager: Optional[BackPressureManager] = None
        
        # Test workers
        self.producers: List[LoadTestProducer] = []
        self.consumers: List[LoadTestConsumer] = []
        self.producer_tasks: List[asyncio.Task] = []
        self.consumer_tasks: List[asyncio.Task] = []
        
        # Metrics collection
        self.test_metrics: List[TestMetrics] = []
        self.current_metrics = TestMetrics(TestPhase.WARMUP, time.time())
        self.metrics_lock = asyncio.Lock()
        
        # Test state
        self.test_start_time = 0.0
        self.current_phase = TestPhase.WARMUP
        self.is_running = False
    
    async def setup(self) -> None:
        """Setup test environment."""
        try:
            # Initialize Redis client
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # Initialize high-performance broker
            batch_config = BatchConfig(
                max_batch_size=50,
                max_batch_wait_ms=25,
                adaptive_batching=True
            )
            compression_config = CompressionConfig(
                min_payload_size=500,
                adaptive_compression=True
            )
            connection_config = ConnectionConfig(
                pool_size=50,
                max_connections=200,
                adaptive_scaling=True
            )
            
            self.broker = HighPerformanceMessageBroker(
                self.redis_url,
                batch_config=batch_config,
                compression_config=compression_config,
                connection_config=connection_config
            )
            await self.broker.start()
            
            # Initialize monitoring
            self.monitor = StreamMonitor(self.redis_client, enable_prometheus=False)
            await self.monitor.start()
            
            # Initialize backpressure manager
            bp_config = BackPressureConfig(
                monitoring_interval_seconds=2,
                throttling_enabled=True
            )
            self.backpressure_manager = BackPressureManager(self.redis_client, bp_config)
            await self.backpressure_manager.start()
            
            # Create test streams
            await self._create_test_streams()
            
            # Create producers
            for i in range(self.config.concurrent_producers):
                producer = LoadTestProducer(
                    producer_id=f"producer_{i}",
                    broker=self.broker,
                    config=self.config,
                    metrics_callback=self._record_producer_metrics
                )
                self.producers.append(producer)
            
            # Create consumers
            stream_names = [
                f"{self.config.stream_name_prefix}_{i}"
                for i in range(1, self.config.stream_count + 1)
            ]
            
            for i in range(self.config.concurrent_consumers):
                consumer = LoadTestConsumer(
                    consumer_id=f"consumer_{i}",
                    redis_client=self.redis_client,
                    config=self.config,
                    metrics_callback=self._record_consumer_metrics
                )
                self.consumers.append(consumer)
            
            logger.info("Load test environment setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup load test environment: {e}")
            raise
    
    async def teardown(self) -> None:
        """Cleanup test environment."""
        try:
            # Stop workers
            await self._stop_all_workers()
            
            # Stop components
            if self.broker:
                await self.broker.stop()
            
            if self.monitor:
                await self.monitor.stop()
            
            if self.backpressure_manager:
                await self.backpressure_manager.stop()
            
            # Close Redis client
            if self.redis_client:
                await self.redis_client.close()
            
            # Cleanup test streams
            await self._cleanup_test_streams()
            
            logger.info("Load test environment cleaned up")
            
        except Exception as e:
            logger.error(f"Error during teardown: {e}")
    
    async def run_full_test(self) -> Dict[str, Any]:
        """Run complete load test with all phases."""
        self.test_start_time = time.time()
        self.is_running = True
        
        try:
            # Start metrics collection
            metrics_task = asyncio.create_task(self._metrics_collection_loop())
            
            # Run test phases
            await self._run_warmup_phase()
            await self._run_ramp_up_phase()
            await self._run_steady_state_phase()
            await self._run_spike_phase()
            await self._run_ramp_down_phase()
            await self._run_cooldown_phase()
            
            # Stop metrics collection
            self.is_running = False
            await metrics_task
            
            # Generate test report
            return self._generate_test_report()
            
        except Exception as e:
            logger.error(f"Error during load test: {e}")
            self.is_running = False
            raise
        finally:
            await self._stop_all_workers()
    
    async def _run_warmup_phase(self) -> None:
        """Run warmup phase."""
        self.current_phase = TestPhase.WARMUP
        logger.info("Starting warmup phase")
        
        # Start consumers
        await self._start_consumers()
        
        # Start producers at low rate
        warmup_rate = self.config.target_messages_per_second * 0.1
        await self._start_producers(warmup_rate)
        
        # Wait for warmup duration
        await asyncio.sleep(self.config.warmup_duration_seconds)
        
        logger.info("Warmup phase complete")
    
    async def _run_ramp_up_phase(self) -> None:
        """Run ramp-up phase."""
        self.current_phase = TestPhase.RAMP_UP
        logger.info("Starting ramp-up phase")
        
        # Gradually increase rate
        start_rate = self.config.target_messages_per_second * 0.1
        end_rate = self.config.target_messages_per_second
        
        steps = 10
        step_duration = self.config.ramp_up_duration_seconds / steps
        
        for step in range(steps):
            progress = (step + 1) / steps
            current_rate = start_rate + (end_rate - start_rate) * progress
            
            await self._update_producer_rates(current_rate)
            await asyncio.sleep(step_duration)
        
        logger.info("Ramp-up phase complete")
    
    async def _run_steady_state_phase(self) -> None:
        """Run steady-state phase."""
        self.current_phase = TestPhase.STEADY_STATE
        logger.info("Starting steady-state phase")
        
        # Maintain target rate
        await self._update_producer_rates(self.config.target_messages_per_second)
        await asyncio.sleep(self.config.steady_state_duration_seconds)
        
        logger.info("Steady-state phase complete")
    
    async def _run_spike_phase(self) -> None:
        """Run spike phase."""
        self.current_phase = TestPhase.SPIKE
        logger.info("Starting spike phase")
        
        # Increase to spike rate
        spike_rate = self.config.target_messages_per_second * self.config.spike_multiplier
        await self._update_producer_rates(spike_rate)
        await asyncio.sleep(self.config.spike_duration_seconds)
        
        logger.info("Spike phase complete")
    
    async def _run_ramp_down_phase(self) -> None:
        """Run ramp-down phase."""
        self.current_phase = TestPhase.RAMP_DOWN
        logger.info("Starting ramp-down phase")
        
        # Gradually decrease rate
        start_rate = self.config.target_messages_per_second * self.config.spike_multiplier
        end_rate = self.config.target_messages_per_second * 0.1
        
        steps = 10
        step_duration = self.config.ramp_down_duration_seconds / steps
        
        for step in range(steps):
            progress = (step + 1) / steps
            current_rate = start_rate - (start_rate - end_rate) * progress
            
            await self._update_producer_rates(current_rate)
            await asyncio.sleep(step_duration)
        
        logger.info("Ramp-down phase complete")
    
    async def _run_cooldown_phase(self) -> None:
        """Run cooldown phase."""
        self.current_phase = TestPhase.COOLDOWN
        logger.info("Starting cooldown phase")
        
        # Maintain low rate
        await self._update_producer_rates(self.config.target_messages_per_second * 0.1)
        await asyncio.sleep(self.config.cooldown_duration_seconds)
        
        logger.info("Cooldown phase complete")
    
    async def _start_producers(self, rate: float) -> None:
        """Start all producers at specified rate."""
        rate_per_producer = rate / len(self.producers)
        
        for producer in self.producers:
            task = asyncio.create_task(producer.start(rate_per_producer))
            self.producer_tasks.append(task)
    
    async def _start_consumers(self) -> None:
        """Start all consumers."""
        stream_names = [
            f"{self.config.stream_name_prefix}_{i}"
            for i in range(1, self.config.stream_count + 1)
        ]
        
        for consumer in self.consumers:
            task = asyncio.create_task(consumer.start(stream_names))
            self.consumer_tasks.append(task)
    
    async def _update_producer_rates(self, total_rate: float) -> None:
        """Update producer rates."""
        rate_per_producer = total_rate / len(self.producers)
        
        for producer in self.producers:
            producer.update_rate(rate_per_producer)
    
    async def _stop_all_workers(self) -> None:
        """Stop all producers and consumers."""
        # Stop producers
        for producer in self.producers:
            producer.stop()
        
        for task in self.producer_tasks:
            task.cancel()
        
        if self.producer_tasks:
            await asyncio.gather(*self.producer_tasks, return_exceptions=True)
        
        # Stop consumers
        for consumer in self.consumers:
            consumer.stop()
        
        for task in self.consumer_tasks:
            task.cancel()
        
        if self.consumer_tasks:
            await asyncio.gather(*self.consumer_tasks, return_exceptions=True)
        
        self.producer_tasks.clear()
        self.consumer_tasks.clear()
    
    async def _create_test_streams(self) -> None:
        """Create test streams."""
        for i in range(1, self.config.stream_count + 1):
            stream_name = f"agent_messages:{self.config.stream_name_prefix}_{i}"
            try:
                # Add a dummy message to create the stream
                await self.redis_client.xadd(stream_name, {"init": "true"})
                # Delete the dummy message
                await self.redis_client.xtrim(stream_name, maxlen=0)
            except Exception as e:
                logger.error(f"Error creating stream {stream_name}: {e}")
    
    async def _cleanup_test_streams(self) -> None:
        """Cleanup test streams."""
        for i in range(1, self.config.stream_count + 1):
            stream_name = f"agent_messages:{self.config.stream_name_prefix}_{i}"
            try:
                await self.redis_client.delete(stream_name)
            except Exception as e:
                logger.error(f"Error deleting stream {stream_name}: {e}")
    
    async def _metrics_collection_loop(self) -> None:
        """Collect metrics during test run."""
        last_collection_time = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                window_duration = current_time - last_collection_time
                
                # Collect current metrics
                async with self.metrics_lock:
                    self.current_metrics.calculate_derived_metrics(window_duration)
                    self.test_metrics.append(self.current_metrics)
                    
                    # Start new metrics collection
                    self.current_metrics = TestMetrics(self.current_phase, current_time)
                
                last_collection_time = current_time
                await asyncio.sleep(5)  # Collect metrics every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)
    
    async def _record_producer_metrics(self, producer_id: str, latency_ms: float, success: bool) -> None:
        """Record producer metrics."""
        async with self.metrics_lock:
            if success:
                self.current_metrics.messages_sent += 1
                self.current_metrics.latencies.append(latency_ms)
            else:
                self.current_metrics.messages_failed += 1
            
            self.current_metrics.active_producers = len([p for p in self.producers if p.is_running])
    
    async def _record_consumer_metrics(self, consumer_id: str, success: bool) -> None:
        """Record consumer metrics."""
        async with self.metrics_lock:
            if success:
                self.current_metrics.messages_acknowledged += 1
            else:
                self.current_metrics.messages_failed += 1
            
            self.current_metrics.active_consumers = len([c for c in self.consumers if c.is_running])
    
    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if not self.test_metrics:
            return {"error": "No metrics collected"}
        
        # Aggregate metrics by phase
        phase_metrics = {}
        for phase in TestPhase:
            phase_data = [m for m in self.test_metrics if m.phase == phase]
            if phase_data:
                phase_metrics[phase.value] = self._aggregate_phase_metrics(phase_data)
        
        # Overall statistics
        all_latencies = []
        total_sent = 0
        total_acked = 0
        total_failed = 0
        
        for metrics in self.test_metrics:
            all_latencies.extend(metrics.latencies)
            total_sent += metrics.messages_sent
            total_acked += metrics.messages_acknowledged
            total_failed += metrics.messages_failed
        
        # Calculate overall performance
        test_duration = time.time() - self.test_start_time
        overall_throughput = total_sent / test_duration if test_duration > 0 else 0
        overall_success_rate = total_acked / total_sent if total_sent > 0 else 0
        
        # Performance targets validation
        steady_state_metrics = phase_metrics.get(TestPhase.STEADY_STATE.value, {})
        targets_met = {
            "throughput_target": steady_state_metrics.get("avg_send_rate", 0) >= self.config.target_messages_per_second,
            "latency_p95_target": steady_state_metrics.get("p95_latency", float('inf')) <= self.config.max_p95_latency_ms,
            "latency_p99_target": steady_state_metrics.get("p99_latency", float('inf')) <= self.config.max_p99_latency_ms,
            "success_rate_target": overall_success_rate >= self.config.min_success_rate
        }
        
        # Get system performance metrics
        broker_metrics = {}
        if self.broker:
            broker_metrics = self.broker.get_performance_metrics()
        
        return {
            "test_config": {
                "target_messages_per_second": self.config.target_messages_per_second,
                "concurrent_producers": self.config.concurrent_producers,
                "concurrent_consumers": self.config.concurrent_consumers,
                "test_duration_seconds": test_duration
            },
            "overall_performance": {
                "total_messages_sent": total_sent,
                "total_messages_acknowledged": total_acked,
                "total_messages_failed": total_failed,
                "overall_throughput_msg_per_sec": overall_throughput,
                "overall_success_rate": overall_success_rate,
                "test_duration_seconds": test_duration
            },
            "latency_statistics": self._calculate_latency_stats(all_latencies),
            "phase_metrics": phase_metrics,
            "targets_validation": targets_met,
            "targets_summary": {
                "all_targets_met": all(targets_met.values()),
                "failed_targets": [k for k, v in targets_met.items() if not v]
            },
            "system_metrics": {
                "broker_performance": broker_metrics,
                "monitoring_enabled": self.monitor is not None,
                "backpressure_enabled": self.backpressure_manager is not None
            }
        }
    
    def _aggregate_phase_metrics(self, phase_data: List[TestMetrics]) -> Dict[str, Any]:
        """Aggregate metrics for a phase."""
        if not phase_data:
            return {}
        
        all_latencies = []
        total_sent = 0
        total_acked = 0
        total_failed = 0
        send_rates = []
        
        for metrics in phase_data:
            all_latencies.extend(metrics.latencies)
            total_sent += metrics.messages_sent
            total_acked += metrics.messages_acknowledged
            total_failed += metrics.messages_failed
            if metrics.send_rate_msg_per_sec > 0:
                send_rates.append(metrics.send_rate_msg_per_sec)
        
        return {
            "total_sent": total_sent,
            "total_acknowledged": total_acked,
            "total_failed": total_failed,
            "success_rate": total_acked / total_sent if total_sent > 0 else 0,
            "avg_send_rate": statistics.mean(send_rates) if send_rates else 0,
            "max_send_rate": max(send_rates) if send_rates else 0,
            **self._calculate_latency_stats(all_latencies)
        }
    
    def _calculate_latency_stats(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate latency statistics."""
        if not latencies:
            return {
                "min_latency": 0.0,
                "max_latency": 0.0,
                "avg_latency": 0.0,
                "p50_latency": 0.0,
                "p95_latency": 0.0,
                "p99_latency": 0.0
            }
        
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        
        return {
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "avg_latency": statistics.mean(latencies),
            "p50_latency": sorted_latencies[n // 2],
            "p95_latency": sorted_latencies[int(n * 0.95)],
            "p99_latency": sorted_latencies[int(n * 0.99)]
        }