"""
Back-pressure Management System for Redis Streams Communication.

Provides consumer lag monitoring, auto-scaling triggers, circuit breaker patterns,
and dynamic throttling to maintain system stability under high load.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum

import structlog
import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError

from ..core.config import settings

logger = structlog.get_logger()


class BackPressureState(str, Enum):
    """Back-pressure system states."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ScalingAction(str, Enum):
    """Consumer scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class BackPressureConfig:
    """Configuration for back-pressure management."""
    
    # Lag monitoring thresholds
    warning_lag_threshold: int = 1000  # Messages
    critical_lag_threshold: int = 5000  # Messages
    emergency_lag_threshold: int = 10000  # Messages
    
    # Consumer scaling parameters
    min_consumers: int = 1
    max_consumers: int = 10
    scale_up_threshold: float = 0.8  # Scale up at 80% capacity
    scale_down_threshold: float = 0.3  # Scale down at 30% capacity
    scale_cooldown_seconds: int = 60  # Minimum time between scaling events
    
    # Circuit breaker configuration
    circuit_breaker_failure_threshold: int = 10  # Failures to trip breaker
    circuit_breaker_timeout_seconds: int = 60  # Time before half-open
    circuit_breaker_success_threshold: int = 5  # Successes to close
    
    # Throttling parameters
    throttling_enabled: bool = True
    max_throttle_factor: float = 0.1  # Minimum processing rate (10%)
    throttle_recovery_rate: float = 0.1  # Recovery rate per second
    
    # Monitoring intervals
    monitoring_interval_seconds: int = 5
    metrics_retention_minutes: int = 60
    
    # Alert callbacks
    alert_callback: Optional[Callable[[BackPressureState, str, Dict[str, Any]], None]] = None


@dataclass
class ConsumerMetrics:
    """Metrics for a single consumer."""
    
    consumer_name: str
    stream_name: str
    group_name: str
    pending_messages: int
    lag: int
    last_delivered_id: str
    idle_time_ms: int
    processing_rate: float  # messages per second
    error_rate: float  # percentage
    last_update: float


@dataclass
class StreamMetrics:
    """Aggregated metrics for a stream."""
    
    stream_name: str
    total_length: int
    total_consumers: int
    total_pending: int
    total_lag: int
    avg_processing_rate: float
    avg_error_rate: float
    backpressure_state: BackPressureState
    recommended_action: ScalingAction
    last_update: float


class BackPressureManager:
    """
    Advanced back-pressure management system for Redis Streams.
    
    Monitors consumer lag, manages auto-scaling, implements circuit breakers,
    and provides dynamic throttling to maintain system stability.
    """
    
    def __init__(
        self,
        redis_client: Redis,
        config: Optional[BackPressureConfig] = None
    ):
        self.redis = redis_client
        self.config = config or BackPressureConfig()
        
        # State tracking
        self._consumer_metrics: Dict[str, ConsumerMetrics] = {}
        self._stream_metrics: Dict[str, StreamMetrics] = {}
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._throttle_factors: Dict[str, float] = {}  # Stream -> throttle factor
        self._last_scaling_events: Dict[str, float] = {}
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._scaling_task: Optional[asyncio.Task] = None
        self._circuit_breaker_task: Optional[asyncio.Task] = None
        
        # Performance tracking
        self._performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Callbacks for scaling actions
        self._scaling_callbacks: Dict[ScalingAction, List[Callable]] = {
            ScalingAction.SCALE_UP: [],
            ScalingAction.SCALE_DOWN: [],
            ScalingAction.MAINTAIN: [],
            ScalingAction.EMERGENCY_STOP: []
        }
    
    async def start(self) -> None:
        """Start back-pressure monitoring and management."""
        try:
            # Start monitoring task
            self._monitoring_task = asyncio.create_task(
                self._monitoring_loop()
            )
            
            # Start scaling management task
            self._scaling_task = asyncio.create_task(
                self._scaling_management_loop()
            )
            
            # Start circuit breaker management
            self._circuit_breaker_task = asyncio.create_task(
                self._circuit_breaker_loop()
            )
            
            logger.info("BackPressure Manager started")
            
        except Exception as e:
            logger.error("Failed to start BackPressure Manager", error=str(e))
            raise
    
    async def stop(self) -> None:
        """Stop back-pressure management."""
        tasks = [
            self._monitoring_task,
            self._scaling_task,
            self._circuit_breaker_task
        ]
        
        for task in tasks:
            if task and not task.done():
                task.cancel()
        
        completed_tasks = [t for t in tasks if t is not None]
        if completed_tasks:
            await asyncio.gather(*completed_tasks, return_exceptions=True)
        
        logger.info("BackPressure Manager stopped")
    
    def register_scaling_callback(
        self,
        action: ScalingAction,
        callback: Callable[[str, Dict[str, Any]], None]
    ) -> None:
        """Register callback for scaling events."""
        self._scaling_callbacks[action].append(callback)
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for consumer lag and performance."""
        while True:
            try:
                current_time = time.time()
                
                # Get all streams with active consumers
                streams = await self._discover_active_streams()
                
                for stream_name in streams:
                    # Update consumer metrics
                    await self._update_consumer_metrics(stream_name, current_time)
                    
                    # Update stream metrics
                    await self._update_stream_metrics(stream_name, current_time)
                
                # Clean up old metrics
                await self._cleanup_old_metrics(current_time)
                
                await asyncio.sleep(self.config.monitoring_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.config.monitoring_interval_seconds)
    
    async def _discover_active_streams(self) -> List[str]:
        """Discover active streams with consumers."""
        try:
            # Get all keys matching agent message streams
            stream_keys = await self.redis.keys("agent_messages:*")
            
            # Filter for streams with active consumer groups
            active_streams = []
            for key in stream_keys:
                try:
                    if isinstance(key, bytes):
                        key = key.decode()
                    
                    groups = await self.redis.xinfo_groups(key)
                    if groups:  # Has active consumer groups
                        active_streams.append(key)
                except RedisError:
                    # Stream might not exist or have groups
                    continue
            
            return active_streams
            
        except Exception as e:
            logger.error(f"Error discovering active streams: {e}")
            return []
    
    async def _update_consumer_metrics(self, stream_name: str, current_time: float) -> None:
        """Update metrics for consumers on a specific stream."""
        try:
            # Get consumer group information
            groups = await self.redis.xinfo_groups(stream_name)
            
            for group in groups:
                group_name = group["name"]
                
                # Get consumers in this group
                consumers = await self.redis.xinfo_consumers(stream_name, group_name)
                
                for consumer in consumers:
                    consumer_name = consumer["name"]
                    consumer_key = f"{stream_name}:{group_name}:{consumer_name}"
                    
                    # Calculate processing rate
                    processing_rate = await self._calculate_processing_rate(
                        consumer_key, consumer["idle"], current_time
                    )
                    
                    # Calculate error rate (simplified - would need more tracking)
                    error_rate = 0.0  # TODO: Track actual error rates
                    
                    # Create/update consumer metrics
                    self._consumer_metrics[consumer_key] = ConsumerMetrics(
                        consumer_name=consumer_name,
                        stream_name=stream_name,
                        group_name=group_name,
                        pending_messages=consumer["pending"],
                        lag=group.get("lag", 0),
                        last_delivered_id=group["last-delivered-id"],
                        idle_time_ms=consumer["idle"],
                        processing_rate=processing_rate,
                        error_rate=error_rate,
                        last_update=current_time
                    )
            
        except Exception as e:
            logger.error(f"Error updating consumer metrics for {stream_name}: {e}")
    
    async def _calculate_processing_rate(
        self,
        consumer_key: str,
        idle_time_ms: int,
        current_time: float
    ) -> float:
        """Calculate message processing rate for a consumer."""
        # Simple rate calculation based on idle time
        # In production, this would track actual message counts over time
        
        if idle_time_ms > 60000:  # If idle for more than 1 minute
            return 0.0
        
        # Estimate based on how recently the consumer was active
        activity_factor = max(0, 1 - (idle_time_ms / 60000))
        
        # Return estimated messages per second (simplified)
        return activity_factor * 10.0  # Assume max 10 msg/sec per consumer
    
    async def _update_stream_metrics(self, stream_name: str, current_time: float) -> None:
        """Update aggregated metrics for a stream."""
        try:
            # Get stream info
            stream_info = await self.redis.xinfo_stream(stream_name)
            
            # Aggregate consumer metrics for this stream
            stream_consumers = [
                metrics for metrics in self._consumer_metrics.values()
                if metrics.stream_name == stream_name
            ]
            
            if not stream_consumers:
                return
            
            total_pending = sum(c.pending_messages for c in stream_consumers)
            total_lag = sum(c.lag for c in stream_consumers)
            avg_processing_rate = sum(c.processing_rate for c in stream_consumers) / len(stream_consumers)
            avg_error_rate = sum(c.error_rate for c in stream_consumers) / len(stream_consumers)
            
            # Determine back-pressure state
            backpressure_state = self._determine_backpressure_state(total_lag)
            
            # Determine recommended scaling action
            recommended_action = self._determine_scaling_action(
                stream_name, total_lag, len(stream_consumers), avg_processing_rate
            )
            
            # Update stream metrics
            self._stream_metrics[stream_name] = StreamMetrics(
                stream_name=stream_name,
                total_length=stream_info["length"],
                total_consumers=len(stream_consumers),
                total_pending=total_pending,
                total_lag=total_lag,
                avg_processing_rate=avg_processing_rate,
                avg_error_rate=avg_error_rate,
                backpressure_state=backpressure_state,
                recommended_action=recommended_action,
                last_update=current_time
            )
            
            # Update performance history
            self._update_performance_history(stream_name, current_time)
            
            # Update throttling if needed
            await self._update_throttling(stream_name, backpressure_state)
            
            # Trigger alerts if necessary
            await self._check_alerts(stream_name, backpressure_state)
            
        except Exception as e:
            logger.error(f"Error updating stream metrics for {stream_name}: {e}")
    
    def _determine_backpressure_state(self, total_lag: int) -> BackPressureState:
        """Determine back-pressure state based on lag."""
        if total_lag >= self.config.emergency_lag_threshold:
            return BackPressureState.EMERGENCY
        elif total_lag >= self.config.critical_lag_threshold:
            return BackPressureState.CRITICAL
        elif total_lag >= self.config.warning_lag_threshold:
            return BackPressureState.WARNING
        else:
            return BackPressureState.NORMAL
    
    def _determine_scaling_action(
        self,
        stream_name: str,
        total_lag: int,
        current_consumers: int,
        avg_processing_rate: float
    ) -> ScalingAction:
        """Determine recommended scaling action."""
        
        # Check cooldown period
        last_scaling = self._last_scaling_events.get(stream_name, 0)
        if time.time() - last_scaling < self.config.scale_cooldown_seconds:
            return ScalingAction.MAINTAIN
        
        # Emergency conditions
        if total_lag >= self.config.emergency_lag_threshold:
            return ScalingAction.EMERGENCY_STOP
        
        # Calculate current capacity utilization
        max_capacity = current_consumers * 10.0  # Assume 10 msg/sec per consumer
        utilization = avg_processing_rate / max_capacity if max_capacity > 0 else 0
        
        # Scaling decisions
        if (utilization >= self.config.scale_up_threshold and 
            current_consumers < self.config.max_consumers):
            return ScalingAction.SCALE_UP
        elif (utilization <= self.config.scale_down_threshold and 
              current_consumers > self.config.min_consumers):
            return ScalingAction.SCALE_DOWN
        else:
            return ScalingAction.MAINTAIN
    
    async def _update_throttling(self, stream_name: str, state: BackPressureState) -> None:
        """Update throttling factor based on back-pressure state."""
        if not self.config.throttling_enabled:
            return
        
        current_factor = self._throttle_factors.get(stream_name, 1.0)
        
        if state == BackPressureState.EMERGENCY:
            new_factor = self.config.max_throttle_factor
        elif state == BackPressureState.CRITICAL:
            new_factor = max(0.5, current_factor * 0.8)
        elif state == BackPressureState.WARNING:
            new_factor = max(0.7, current_factor * 0.9)
        else:  # NORMAL
            # Gradually recover
            new_factor = min(1.0, current_factor + self.config.throttle_recovery_rate)
        
        self._throttle_factors[stream_name] = new_factor
        
        if abs(new_factor - current_factor) > 0.05:  # Significant change
            logger.info(
                "Throttling updated",
                stream=stream_name,
                old_factor=current_factor,
                new_factor=new_factor,
                state=state.value
            )
    
    async def _check_alerts(self, stream_name: str, state: BackPressureState) -> None:
        """Check if alerts should be triggered."""
        if state != BackPressureState.NORMAL and self.config.alert_callback:
            stream_metrics = self._stream_metrics.get(stream_name)
            if stream_metrics:
                alert_data = {
                    "stream_name": stream_name,
                    "state": state.value,
                    "total_lag": stream_metrics.total_lag,
                    "total_consumers": stream_metrics.total_consumers,
                    "recommended_action": stream_metrics.recommended_action.value
                }
                
                message = f"Back-pressure {state.value} on stream {stream_name}"
                self.config.alert_callback(state, message, alert_data)
    
    def _update_performance_history(self, stream_name: str, current_time: float) -> None:
        """Update performance history for trend analysis."""
        if stream_name not in self._performance_history:
            self._performance_history[stream_name] = []
        
        stream_metrics = self._stream_metrics.get(stream_name)
        if stream_metrics:
            history_entry = {
                "timestamp": current_time,
                "total_lag": stream_metrics.total_lag,
                "total_consumers": stream_metrics.total_consumers,
                "avg_processing_rate": stream_metrics.avg_processing_rate,
                "backpressure_state": stream_metrics.backpressure_state.value
            }
            
            self._performance_history[stream_name].append(history_entry)
            
            # Keep only recent history
            cutoff_time = current_time - (self.config.metrics_retention_minutes * 60)
            self._performance_history[stream_name] = [
                entry for entry in self._performance_history[stream_name]
                if entry["timestamp"] > cutoff_time
            ]
    
    async def _scaling_management_loop(self) -> None:
        """Loop to execute scaling actions."""
        while True:
            try:
                current_time = time.time()
                
                for stream_name, metrics in self._stream_metrics.items():
                    if metrics.recommended_action != ScalingAction.MAINTAIN:
                        # Check if we should execute the scaling action
                        last_scaling = self._last_scaling_events.get(stream_name, 0)
                        
                        if current_time - last_scaling >= self.config.scale_cooldown_seconds:
                            await self._execute_scaling_action(
                                stream_name, metrics.recommended_action, metrics
                            )
                            self._last_scaling_events[stream_name] = current_time
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in scaling management loop: {e}")
                await asyncio.sleep(10)
    
    async def _execute_scaling_action(
        self,
        stream_name: str,
        action: ScalingAction,
        metrics: StreamMetrics
    ) -> None:
        """Execute a scaling action."""
        try:
            action_data = {
                "stream_name": stream_name,
                "action": action.value,
                "current_consumers": metrics.total_consumers,
                "total_lag": metrics.total_lag,
                "backpressure_state": metrics.backpressure_state.value
            }
            
            # Execute callbacks for this action
            for callback in self._scaling_callbacks[action]:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, callback, stream_name, action_data
                    )
                except Exception as e:
                    logger.error(f"Error in scaling callback: {e}")
            
            logger.info(
                "Scaling action executed",
                stream=stream_name,
                action=action.value,
                data=action_data
            )
            
        except Exception as e:
            logger.error(f"Error executing scaling action: {e}")
    
    async def _circuit_breaker_loop(self) -> None:
        """Manage circuit breaker states."""
        while True:
            try:
                current_time = time.time()
                
                for stream_name in list(self._circuit_breakers.keys()):
                    breaker = self._circuit_breakers[stream_name]
                    
                    if breaker["state"] == "open":
                        # Check if should transition to half-open
                        if current_time >= breaker["retry_time"]:
                            breaker["state"] = "half-open"
                            breaker["success_count"] = 0
                            logger.info(f"Circuit breaker half-open for {stream_name}")
                    
                    elif breaker["state"] == "half-open":
                        # Check if should close or re-open
                        if breaker["success_count"] >= self.config.circuit_breaker_success_threshold:
                            breaker["state"] = "closed"
                            breaker["failure_count"] = 0
                            logger.info(f"Circuit breaker closed for {stream_name}")
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in circuit breaker loop: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_old_metrics(self, current_time: float) -> None:
        """Clean up old metrics to prevent memory leaks."""
        cutoff_time = current_time - (self.config.metrics_retention_minutes * 60)
        
        # Clean up consumer metrics
        old_consumers = [
            key for key, metrics in self._consumer_metrics.items()
            if metrics.last_update < cutoff_time
        ]
        
        for key in old_consumers:
            del self._consumer_metrics[key]
        
        # Clean up stream metrics
        old_streams = [
            key for key, metrics in self._stream_metrics.items()
            if metrics.last_update < cutoff_time
        ]
        
        for key in old_streams:
            del self._stream_metrics[key]
    
    def get_throttle_factor(self, stream_name: str) -> float:
        """Get current throttle factor for a stream."""
        return self._throttle_factors.get(stream_name, 1.0)
    
    def get_stream_metrics(self, stream_name: Optional[str] = None) -> Dict[str, Any]:
        """Get current stream metrics."""
        if stream_name:
            metrics = self._stream_metrics.get(stream_name)
            return metrics.__dict__ if metrics else {}
        
        return {
            name: metrics.__dict__ 
            for name, metrics in self._stream_metrics.items()
        }
    
    def get_consumer_metrics(self, stream_name: Optional[str] = None) -> Dict[str, Any]:
        """Get current consumer metrics."""
        if stream_name:
            return {
                key: metrics.__dict__
                for key, metrics in self._consumer_metrics.items()
                if metrics.stream_name == stream_name
            }
        
        return {
            key: metrics.__dict__
            for key, metrics in self._consumer_metrics.items()
        }
    
    def get_performance_history(self, stream_name: str) -> List[Dict[str, Any]]:
        """Get performance history for a stream."""
        return self._performance_history.get(stream_name, [])
    
    def is_circuit_breaker_open(self, stream_name: str) -> bool:
        """Check if circuit breaker is open for a stream."""
        breaker = self._circuit_breakers.get(stream_name)
        return breaker and breaker["state"] == "open"
    
    def record_circuit_breaker_failure(self, stream_name: str) -> None:
        """Record a failure for circuit breaker tracking."""
        if stream_name not in self._circuit_breakers:
            self._circuit_breakers[stream_name] = {
                "state": "closed",
                "failure_count": 0,
                "success_count": 0,
                "retry_time": 0
            }
        
        breaker = self._circuit_breakers[stream_name]
        breaker["failure_count"] += 1
        
        # Check if should open circuit breaker
        if (breaker["state"] == "closed" and 
            breaker["failure_count"] >= self.config.circuit_breaker_failure_threshold):
            
            breaker["state"] = "open"
            breaker["retry_time"] = time.time() + self.config.circuit_breaker_timeout_seconds
            
            logger.warning(f"Circuit breaker opened for {stream_name}")
    
    def record_circuit_breaker_success(self, stream_name: str) -> None:
        """Record a success for circuit breaker tracking."""
        if stream_name in self._circuit_breakers:
            breaker = self._circuit_breakers[stream_name]
            
            if breaker["state"] == "half-open":
                breaker["success_count"] += 1
            elif breaker["state"] == "closed":
                # Reset failure count on success
                breaker["failure_count"] = max(0, breaker["failure_count"] - 1)