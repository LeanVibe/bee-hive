"""
Enterprise Back-pressure Management System for LeanVibe Agent Hive 2.0

Provides comprehensive back-pressure management with:
- Adaptive flow control
- Load shedding mechanisms  
- Circuit breaker integration
- Real-time monitoring
- Predictive load management

Performance targets:
- Handle >10,000 messages/second without degradation
- <10ms latency impact under normal conditions
- Graceful degradation under extreme load
- Auto-recovery within 30 seconds
"""

import asyncio
import time
import logging
import statistics
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import math

import structlog
import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = structlog.get_logger()


class BackPressureLevel(str, Enum):
    """Back-pressure severity levels."""
    NONE = "none"           # 0-60% capacity
    LOW = "low"             # 60-75% capacity  
    MEDIUM = "medium"       # 75-85% capacity
    HIGH = "high"           # 85-95% capacity
    CRITICAL = "critical"   # >95% capacity


class LoadSheddingStrategy(str, Enum):
    """Load shedding strategies."""
    PRIORITY_BASED = "priority_based"
    RANDOM = "random"
    OLDEST_FIRST = "oldest_first"
    LEAST_IMPORTANT = "least_important"


class FlowControlAction(str, Enum):
    """Flow control actions."""
    ALLOW = "allow"
    THROTTLE = "throttle"
    DELAY = "delay"
    REJECT = "reject"
    SHED = "shed"


@dataclass
class BackPressureMetrics:
    """Back-pressure system metrics."""
    
    # Current system state
    current_load_percent: float = 0.0
    pressure_level: BackPressureLevel = BackPressureLevel.NONE
    messages_per_second: float = 0.0
    average_latency_ms: float = 0.0
    
    # Flow control metrics
    messages_allowed: int = 0
    messages_throttled: int = 0
    messages_delayed: int = 0
    messages_rejected: int = 0
    messages_shed: int = 0
    
    # System resource metrics
    redis_memory_usage_mb: float = 0.0
    redis_cpu_percent: float = 0.0
    connection_pool_utilization: float = 0.0
    queue_depths: Dict[str, int] = field(default_factory=dict)
    
    # Performance metrics
    throughput_efficiency: float = 1.0
    latency_overhead_ms: float = 0.0
    recovery_time_seconds: float = 0.0
    
    def calculate_efficiency(self) -> None:
        """Calculate system efficiency metrics."""
        total_messages = (
            self.messages_allowed + self.messages_throttled + 
            self.messages_delayed + self.messages_rejected + self.messages_shed
        )
        
        if total_messages > 0:
            self.throughput_efficiency = self.messages_allowed / total_messages
        else:
            self.throughput_efficiency = 1.0


@dataclass
class FlowControlDecision:
    """Flow control decision for a message."""
    action: FlowControlAction
    delay_ms: int = 0
    reason: str = ""
    priority_boost: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter using token bucket algorithm with dynamic adjustment.
    """
    
    def __init__(
        self,
        initial_rate: float = 1000.0,
        burst_capacity: int = 100,
        min_rate: float = 10.0,
        max_rate: float = 50000.0,
        adaptation_factor: float = 0.1
    ):
        self.current_rate = initial_rate
        self.burst_capacity = burst_capacity
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.adaptation_factor = adaptation_factor
        
        # Token bucket state
        self.tokens = float(burst_capacity)
        self.last_refill = time.time()
        
        # Adaptation state
        self.recent_latencies = deque(maxlen=100)
        self.recent_success_rates = deque(maxlen=50)
        self.last_adaptation = time.time()
        
    async def check_rate_limit(
        self, 
        message_priority: str = "normal",
        current_latency_ms: float = 0.0
    ) -> Tuple[bool, float]:
        """
        Check if message should be rate limited.
        
        Returns:
            (allowed, wait_time_seconds)
        """
        current_time = time.time()
        
        # Refill tokens
        self._refill_tokens(current_time)
        
        # Record metrics for adaptation
        if current_latency_ms > 0:
            self.recent_latencies.append(current_latency_ms)
        
        # Priority adjustments
        tokens_needed = self._calculate_tokens_needed(message_priority)
        
        if self.tokens >= tokens_needed:
            self.tokens -= tokens_needed
            self.recent_success_rates.append(1.0)
            return True, 0.0
        else:
            # Calculate wait time
            wait_time = tokens_needed / self.current_rate
            self.recent_success_rates.append(0.0)
            return False, wait_time
    
    def _refill_tokens(self, current_time: float) -> None:
        """Refill token bucket based on current rate."""
        time_passed = current_time - self.last_refill
        tokens_to_add = time_passed * self.current_rate
        
        self.tokens = min(self.burst_capacity, self.tokens + tokens_to_add)
        self.last_refill = current_time
    
    def _calculate_tokens_needed(self, priority: str) -> float:
        """Calculate tokens needed based on message priority."""
        priority_multipliers = {
            "critical": 0.5,   # Require fewer tokens
            "high": 0.75,
            "normal": 1.0,
            "low": 1.5,        # Require more tokens
            "bulk": 2.0
        }
        return priority_multipliers.get(priority, 1.0)
    
    async def adapt_rate(self, system_metrics: BackPressureMetrics) -> None:
        """Adapt rate based on system performance."""
        current_time = time.time()
        
        # Only adapt every few seconds
        if current_time - self.last_adaptation < 5.0:
            return
        
        self.last_adaptation = current_time
        
        # Calculate adaptation signals
        avg_latency = statistics.mean(self.recent_latencies) if self.recent_latencies else 0.0
        success_rate = statistics.mean(self.recent_success_rates) if self.recent_success_rates else 1.0
        
        # Determine rate adjustment
        rate_adjustment = 1.0
        
        # High latency or low success rate -> decrease rate
        if avg_latency > 200.0 or success_rate < 0.8:  # 200ms latency threshold
            rate_adjustment = 1.0 - self.adaptation_factor
        
        # High back-pressure -> decrease rate
        elif system_metrics.pressure_level in [BackPressureLevel.HIGH, BackPressureLevel.CRITICAL]:
            rate_adjustment = 1.0 - (self.adaptation_factor * 2)
        
        # Low latency and good throughput -> increase rate
        elif avg_latency < 50.0 and success_rate > 0.95 and system_metrics.pressure_level == BackPressureLevel.NONE:
            rate_adjustment = 1.0 + self.adaptation_factor
        
        # Apply adjustment
        new_rate = self.current_rate * rate_adjustment
        self.current_rate = max(self.min_rate, min(self.max_rate, new_rate))
        
        logger.debug(
            "Rate limiter adaptation",
            old_rate=self.current_rate / rate_adjustment,
            new_rate=self.current_rate,
            avg_latency=avg_latency,
            success_rate=success_rate,
            pressure_level=system_metrics.pressure_level.value
        )


class LoadShedder:
    """
    Emergency load shedding with priority-based message dropping.
    """
    
    def __init__(
        self,
        shedding_threshold: float = 0.9,  # 90% load
        recovery_threshold: float = 0.7,   # 70% load
        max_shed_rate: float = 0.5         # Max 50% of messages
    ):
        self.shedding_threshold = shedding_threshold
        self.recovery_threshold = recovery_threshold
        self.max_shed_rate = max_shed_rate
        
        self.is_shedding = False
        self.current_shed_rate = 0.0
        self.shed_count = 0
        self.total_processed = 0
        
        # Priority thresholds for shedding
        self.priority_thresholds = {
            "critical": 0.99,   # Almost never shed
            "high": 0.9,
            "normal": 0.7,
            "low": 0.3,
            "bulk": 0.1         # Shed first
        }
    
    async def should_shed_message(
        self,
        message_priority: str,
        current_load: float,
        queue_depth: int = 0
    ) -> Tuple[bool, str]:
        """
        Determine if message should be shed.
        
        Returns:
            (should_shed, reason)
        """
        self.total_processed += 1
        
        # Check if we should start/continue shedding
        if not self.is_shedding and current_load > self.shedding_threshold:
            self.is_shedding = True
            self.current_shed_rate = min(current_load - self.shedding_threshold, self.max_shed_rate)
            logger.warning(
                "Load shedding activated",
                current_load=current_load,
                shed_rate=self.current_shed_rate
            )
        
        elif self.is_shedding and current_load < self.recovery_threshold:
            self.is_shedding = False
            self.current_shed_rate = 0.0
            logger.info(
                "Load shedding deactivated",
                current_load=current_load,
                total_shed=self.shed_count
            )
        
        # If not shedding, allow all messages
        if not self.is_shedding:
            return False, ""
        
        # Calculate shedding probability for this message
        priority_threshold = self.priority_thresholds.get(message_priority, 0.5)
        shed_probability = self.current_shed_rate * (1.0 - priority_threshold)
        
        # Factor in queue depth
        if queue_depth > 1000:  # High queue depth increases shedding
            shed_probability *= 1.5
        
        # Make shedding decision
        import random
        should_shed = random.random() < shed_probability
        
        if should_shed:
            self.shed_count += 1
            reason = f"Load shedding active (load: {current_load:.1%}, priority: {message_priority})"
            return True, reason
        
        return False, ""
    
    def get_shedding_stats(self) -> Dict[str, Any]:
        """Get current shedding statistics."""
        shed_rate = self.shed_count / self.total_processed if self.total_processed > 0 else 0.0
        
        return {
            "is_shedding": self.is_shedding,
            "current_shed_rate": self.current_shed_rate,
            "total_messages_processed": self.total_processed,
            "total_messages_shed": self.shed_count,
            "overall_shed_rate": shed_rate,
            "shedding_threshold": self.shedding_threshold,
            "recovery_threshold": self.recovery_threshold
        }


class EnterpriseBackPressureManager:
    """
    Enterprise-grade back-pressure management system.
    
    Provides comprehensive flow control with:
    - Real-time load monitoring
    - Adaptive rate limiting
    - Emergency load shedding
    - Circuit breaker integration
    - Predictive load management
    """
    
    def __init__(
        self,
        redis_client: Redis,
        max_throughput: float = 10000.0,  # msgs/sec
        target_latency_ms: float = 50.0,
        monitoring_interval: float = 1.0   # seconds
    ):
        self.redis = redis_client
        self.max_throughput = max_throughput
        self.target_latency_ms = target_latency_ms
        self.monitoring_interval = monitoring_interval
        
        # Core components
        self.rate_limiter = AdaptiveRateLimiter(
            initial_rate=max_throughput * 0.8,  # Start at 80% capacity
            max_rate=max_throughput
        )
        
        self.load_shedder = LoadShedder(
            shedding_threshold=0.85,  # 85% load threshold
            recovery_threshold=0.65   # 65% recovery threshold
        )
        
        # Monitoring state
        self.current_metrics = BackPressureMetrics()
        self.historical_metrics = deque(maxlen=300)  # 5 minutes at 1s intervals
        
        # Load tracking
        self.message_timestamps = deque(maxlen=10000)
        self.latency_samples = deque(maxlen=1000)
        self.queue_depth_samples = defaultdict(lambda: deque(maxlen=100))
        
        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict] = {}
        
        # Background tasks
        self.monitor_task: Optional[asyncio.Task] = None
        self.adaptation_task: Optional[asyncio.Task] = None
        
        self.is_running = False
    
    async def start(self) -> None:
        """Start back-pressure management system."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start monitoring tasks
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        self.adaptation_task = asyncio.create_task(self._adaptation_loop())
        
        logger.info(
            "Enterprise back-pressure manager started",
            max_throughput=self.max_throughput,
            target_latency_ms=self.target_latency_ms
        )
    
    async def stop(self) -> None:
        """Stop back-pressure management system."""
        self.is_running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
        if self.adaptation_task:
            self.adaptation_task.cancel()
        
        tasks = [t for t in [self.monitor_task, self.adaptation_task] if t]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Enterprise back-pressure manager stopped")
    
    async def check_flow_control(
        self,
        stream_name: str,
        message_priority: str = "normal",
        message_size_bytes: int = 1024,
        current_latency_ms: float = 0.0
    ) -> FlowControlDecision:
        """
        Make flow control decision for incoming message.
        
        Returns:
            FlowControlDecision with action and parameters
        """
        current_time = time.time()
        
        # Update metrics
        await self._update_real_time_metrics(current_time, current_latency_ms)
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(stream_name):
            return FlowControlDecision(
                action=FlowControlAction.REJECT,
                reason=f"Circuit breaker open for stream {stream_name}"
            )
        
        # Check load shedding
        queue_depth = self.queue_depth_samples[stream_name][-1] if self.queue_depth_samples[stream_name] else 0
        should_shed, shed_reason = await self.load_shedder.should_shed_message(
            message_priority, 
            self.current_metrics.current_load_percent / 100.0,
            queue_depth
        )
        
        if should_shed:
            return FlowControlDecision(
                action=FlowControlAction.SHED,
                reason=shed_reason
            )
        
        # Check rate limiting
        allowed, wait_time = await self.rate_limiter.check_rate_limit(
            message_priority, current_latency_ms
        )
        
        if not allowed:
            if wait_time > 1.0:  # If wait time is too long, reject
                return FlowControlDecision(
                    action=FlowControlAction.REJECT,
                    reason=f"Rate limit exceeded, wait time {wait_time:.2f}s too long"
                )
            else:
                return FlowControlDecision(
                    action=FlowControlAction.DELAY,
                    delay_ms=int(wait_time * 1000),
                    reason=f"Rate limited, delay {wait_time:.3f}s"
                )
        
        # Check back-pressure level for throttling
        if self.current_metrics.pressure_level == BackPressureLevel.HIGH:
            return FlowControlDecision(
                action=FlowControlAction.THROTTLE,
                delay_ms=50,  # 50ms delay
                reason="High back-pressure detected"
            )
        
        elif self.current_metrics.pressure_level == BackPressureLevel.MEDIUM:
            return FlowControlDecision(
                action=FlowControlAction.THROTTLE,
                delay_ms=20,  # 20ms delay
                reason="Medium back-pressure detected"
            )
        
        # Allow message
        return FlowControlDecision(
            action=FlowControlAction.ALLOW,
            reason="Normal flow conditions"
        )
    
    async def record_message_processed(
        self,
        stream_name: str,
        processing_time_ms: float,
        success: bool,
        queue_depth: int = 0
    ) -> None:
        """Record message processing metrics."""
        current_time = time.time()
        
        # Record timing
        self.message_timestamps.append(current_time)
        self.latency_samples.append(processing_time_ms)
        self.queue_depth_samples[stream_name].append(queue_depth)
        
        # Update circuit breaker
        await self._update_circuit_breaker(stream_name, success, processing_time_ms)
        
        # Update flow control metrics
        if success:
            self.current_metrics.messages_allowed += 1
        else:
            self.current_metrics.messages_rejected += 1
    
    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.is_running:
            try:
                await self._collect_system_metrics()
                await self._update_pressure_level()
                await self._store_historical_metrics()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _adaptation_loop(self) -> None:
        """Background adaptation loop."""
        while self.is_running:
            try:
                # Adapt rate limiter
                await self.rate_limiter.adapt_rate(self.current_metrics)
                
                # Update circuit breaker states
                await self._update_all_circuit_breakers()
                
                await asyncio.sleep(5.0)  # Adapt every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in adaptation loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _update_real_time_metrics(
        self, 
        current_time: float, 
        latency_ms: float
    ) -> None:
        """Update real-time metrics."""
        # Calculate throughput (messages per second)
        recent_timestamps = [
            t for t in self.message_timestamps 
            if current_time - t <= 1.0
        ]
        self.current_metrics.messages_per_second = len(recent_timestamps)
        
        # Calculate average latency
        if self.latency_samples:
            self.current_metrics.average_latency_ms = statistics.mean(
                list(self.latency_samples)[-100:]  # Last 100 samples
            )
        
        # Calculate load percentage
        load_ratio = self.current_metrics.messages_per_second / self.max_throughput
        self.current_metrics.current_load_percent = min(100.0, load_ratio * 100.0)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system resource metrics."""
        try:
            # Redis memory usage
            redis_info = await self.redis.info("memory")
            used_memory = redis_info.get("used_memory", 0)
            self.current_metrics.redis_memory_usage_mb = used_memory / (1024 * 1024)
            
            # Connection pool utilization (simplified)
            if hasattr(self.redis.connection_pool, '_available_connections'):
                available = len(self.redis.connection_pool._available_connections)
                total = self.redis.connection_pool.max_connections
                self.current_metrics.connection_pool_utilization = (total - available) / total
            
            # Queue depths
            for stream_name, depths in self.queue_depth_samples.items():
                if depths:
                    self.current_metrics.queue_depths[stream_name] = depths[-1]
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _update_pressure_level(self) -> None:
        """Update current back-pressure level."""
        load_percent = self.current_metrics.current_load_percent
        
        if load_percent >= 95:
            self.current_metrics.pressure_level = BackPressureLevel.CRITICAL
        elif load_percent >= 85:
            self.current_metrics.pressure_level = BackPressureLevel.HIGH
        elif load_percent >= 75:
            self.current_metrics.pressure_level = BackPressureLevel.MEDIUM
        elif load_percent >= 60:
            self.current_metrics.pressure_level = BackPressureLevel.LOW
        else:
            self.current_metrics.pressure_level = BackPressureLevel.NONE
    
    async def _store_historical_metrics(self) -> None:
        """Store current metrics in historical data."""
        self.current_metrics.calculate_efficiency()
        
        # Create a copy for historical storage
        historical_entry = BackPressureMetrics(
            current_load_percent=self.current_metrics.current_load_percent,
            pressure_level=self.current_metrics.pressure_level,
            messages_per_second=self.current_metrics.messages_per_second,
            average_latency_ms=self.current_metrics.average_latency_ms,
            messages_allowed=self.current_metrics.messages_allowed,
            messages_throttled=self.current_metrics.messages_throttled,
            messages_delayed=self.current_metrics.messages_delayed,
            messages_rejected=self.current_metrics.messages_rejected,
            messages_shed=self.current_metrics.messages_shed,
            throughput_efficiency=self.current_metrics.throughput_efficiency
        )
        
        self.historical_metrics.append(historical_entry)
    
    def _is_circuit_breaker_open(self, stream_name: str) -> bool:
        """Check if circuit breaker is open for stream."""
        if stream_name not in self.circuit_breakers:
            return False
        
        cb_state = self.circuit_breakers[stream_name]
        return cb_state.get("state") == "open"
    
    async def _update_circuit_breaker(
        self, 
        stream_name: str, 
        success: bool, 
        latency_ms: float
    ) -> None:
        """Update circuit breaker state for stream."""
        current_time = time.time()
        
        if stream_name not in self.circuit_breakers:
            self.circuit_breakers[stream_name] = {
                "state": "closed",
                "failure_count": 0,
                "success_count": 0,
                "last_failure_time": 0,
                "next_test_time": 0
            }
        
        cb_state = self.circuit_breakers[stream_name]
        
        if success and latency_ms < self.target_latency_ms * 2:
            cb_state["success_count"] += 1
            
            # Reset failure count on successful operations
            if cb_state["success_count"] >= 5:
                cb_state["failure_count"] = 0
                if cb_state["state"] == "half_open":
                    cb_state["state"] = "closed"
                    logger.info(f"Circuit breaker closed for stream {stream_name}")
        else:
            cb_state["failure_count"] += 1
            cb_state["success_count"] = 0
            cb_state["last_failure_time"] = current_time
            
            # Open circuit breaker on repeated failures
            if cb_state["failure_count"] >= 5 and cb_state["state"] == "closed":
                cb_state["state"] = "open"
                cb_state["next_test_time"] = current_time + 30  # 30 second timeout
                logger.warning(f"Circuit breaker opened for stream {stream_name}")
    
    async def _update_all_circuit_breakers(self) -> None:
        """Update all circuit breaker states."""
        current_time = time.time()
        
        for stream_name, cb_state in self.circuit_breakers.items():
            if (cb_state["state"] == "open" and 
                current_time >= cb_state["next_test_time"]):
                cb_state["state"] = "half_open"
                logger.info(f"Circuit breaker half-open for stream {stream_name}")
    
    async def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive back-pressure metrics."""
        # Calculate historical trends
        historical_loads = [m.current_load_percent for m in self.historical_metrics]
        historical_latencies = [m.average_latency_ms for m in self.historical_metrics]
        
        return {
            "current_metrics": {
                "load_percent": self.current_metrics.current_load_percent,
                "pressure_level": self.current_metrics.pressure_level.value,
                "messages_per_second": self.current_metrics.messages_per_second,
                "average_latency_ms": self.current_metrics.average_latency_ms,
                "throughput_efficiency": self.current_metrics.throughput_efficiency
            },
            "flow_control_stats": {
                "messages_allowed": self.current_metrics.messages_allowed,
                "messages_throttled": self.current_metrics.messages_throttled,
                "messages_delayed": self.current_metrics.messages_delayed,
                "messages_rejected": self.current_metrics.messages_rejected,
                "messages_shed": self.current_metrics.messages_shed
            },
            "rate_limiter": {
                "current_rate": self.rate_limiter.current_rate,
                "burst_capacity": self.rate_limiter.burst_capacity,
                "tokens_available": self.rate_limiter.tokens
            },
            "load_shedder": self.load_shedder.get_shedding_stats(),
            "circuit_breakers": {
                stream: {
                    "state": cb["state"],
                    "failure_count": cb["failure_count"],
                    "success_count": cb["success_count"]
                }
                for stream, cb in self.circuit_breakers.items()
            },
            "historical_trends": {
                "avg_load_5min": statistics.mean(historical_loads) if historical_loads else 0.0,
                "max_load_5min": max(historical_loads) if historical_loads else 0.0,
                "avg_latency_5min": statistics.mean(historical_latencies) if historical_latencies else 0.0,
                "max_latency_5min": max(historical_latencies) if historical_latencies else 0.0
            },
            "system_resources": {
                "redis_memory_mb": self.current_metrics.redis_memory_usage_mb,
                "connection_pool_utilization": self.current_metrics.connection_pool_utilization,
                "queue_depths": dict(self.current_metrics.queue_depths)
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of back-pressure system."""
        try:
            # Check if system is responsive
            current_load = self.current_metrics.current_load_percent
            pressure_level = self.current_metrics.pressure_level
            avg_latency = self.current_metrics.average_latency_ms
            
            # Determine health status
            is_healthy = (
                self.is_running and
                current_load < 90.0 and  # Under 90% load
                avg_latency < self.target_latency_ms * 2 and  # Latency under 2x target
                pressure_level != BackPressureLevel.CRITICAL
            )
            
            status = "healthy" if is_healthy else "degraded"
            if pressure_level == BackPressureLevel.CRITICAL:
                status = "critical"
            
            return {
                "status": status,
                "is_running": self.is_running,
                "current_load_percent": current_load,
                "pressure_level": pressure_level.value,
                "average_latency_ms": avg_latency,
                "target_latency_ms": self.target_latency_ms,
                "rate_limiter_active": self.rate_limiter.tokens < self.rate_limiter.burst_capacity,
                "load_shedding_active": self.load_shedder.is_shedding,
                "circuit_breakers_open": sum(
                    1 for cb in self.circuit_breakers.values() 
                    if cb["state"] == "open"
                ),
                "recommendations": self._generate_health_recommendations(
                    current_load, pressure_level, avg_latency
                )
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "is_running": self.is_running
            }
    
    def _generate_health_recommendations(
        self,
        current_load: float,
        pressure_level: BackPressureLevel,
        avg_latency: float
    ) -> List[str]:
        """Generate health recommendations."""
        recommendations = []
        
        if current_load > 85.0:
            recommendations.append(
                f"High system load ({current_load:.1f}%). Consider scaling horizontally or optimizing consumers."
            )
        
        if avg_latency > self.target_latency_ms * 1.5:
            recommendations.append(
                f"High latency ({avg_latency:.1f}ms vs {self.target_latency_ms}ms target). Check Redis performance and network."
            )
        
        if pressure_level == BackPressureLevel.CRITICAL:
            recommendations.append(
                "Critical back-pressure detected. Emergency load shedding may be active. Check system capacity."
            )
        
        if self.load_shedder.is_shedding:
            recommendations.append(
                f"Load shedding active ({self.load_shedder.current_shed_rate:.1%} shed rate). Increase system capacity."
            )
        
        open_breakers = sum(1 for cb in self.circuit_breakers.values() if cb["state"] == "open")
        if open_breakers > 0:
            recommendations.append(
                f"{open_breakers} circuit breaker(s) open. Check stream health and consumer performance."
            )
        
        if not recommendations:
            recommendations.append("System operating within normal parameters.")
        
        return recommendations