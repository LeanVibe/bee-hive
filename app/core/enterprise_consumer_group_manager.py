"""
Enterprise Consumer Group Management System for LeanVibe Agent Hive 2.0

Provides advanced consumer group management with:
- Automatic consumer scaling and balancing
- Health monitoring and recovery
- Intelligent failover mechanisms
- Performance optimization
- Real-time coordination

Performance targets:
- 99.9% message delivery reliability
- Auto-scaling within 30 seconds
- Consumer failover within 10 seconds
- Load balancing efficiency >95%
"""

import asyncio
import time
import logging
import statistics
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import json

import structlog
import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError

from .enterprise_backpressure_manager import EnterpriseBackPressureManager

logger = structlog.get_logger()


class ConsumerState(str, Enum):
    """Consumer states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNRESPONSIVE = "unresponsive"
    FAILED = "failed"
    SCALING = "scaling"


class ScalingAction(str, Enum):
    """Auto-scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    REBALANCE = "rebalance"
    NO_ACTION = "no_action"


class FailoverStrategy(str, Enum):
    """Failover strategies."""
    IMMEDIATE = "immediate"
    GRACEFUL = "graceful"
    LOAD_AWARE = "load_aware"


@dataclass
class ConsumerMetrics:
    """Consumer performance metrics."""
    consumer_id: str
    group_name: str
    stream_name: str
    
    # Performance metrics
    messages_processed: int = 0
    messages_per_second: float = 0.0
    average_latency_ms: float = 0.0  
    error_rate: float = 0.0
    
    # Health metrics
    last_heartbeat: float = 0.0
    response_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    # Queue metrics
    pending_messages: int = 0
    claimed_messages: int = 0
    acknowledged_messages: int = 0
    
    # State
    state: ConsumerState = ConsumerState.HEALTHY
    last_state_change: float = 0.0
    consecutive_errors: int = 0
    
    def calculate_health_score(self) -> float:
        """Calculate consumer health score (0.0 to 1.0)."""
        score = 1.0
        
        # Penalize high error rate
        score -= min(self.error_rate * 0.5, 0.3)
        
        # Penalize high latency
        if self.average_latency_ms > 100:
            score -= min((self.average_latency_ms - 100) / 1000, 0.2)
        
        # Penalize unresponsiveness
        time_since_heartbeat = time.time() - self.last_heartbeat
        if time_since_heartbeat > 60:  # 1 minute
            score -= min(time_since_heartbeat / 300, 0.3)  # Max 0.3 penalty for 5 min
        
        # Penalize consecutive errors
        if self.consecutive_errors > 0:
            score -= min(self.consecutive_errors * 0.05, 0.2)
        
        return max(0.0, score)


@dataclass
class ConsumerGroupState:
    """Consumer group state and metrics."""
    group_name: str
    stream_name: str
    
    # Consumers
    consumers: Dict[str, ConsumerMetrics] = field(default_factory=dict)
    target_consumer_count: int = 3
    
    # Performance metrics
    total_throughput: float = 0.0
    average_latency_ms: float = 0.0
    overall_error_rate: float = 0.0
    
    # Health metrics
    healthy_consumers: int = 0
    degraded_consumers: int = 0
    failed_consumers: int = 0
    
    # Scaling metrics
    last_scaling_action: Optional[ScalingAction] = None
    last_scaling_time: float = 0.0
    scaling_cooldown_seconds: float = 60.0
    
    # Load balancing
    load_distribution_score: float = 1.0  # 1.0 = perfectly balanced
    rebalancing_needed: bool = False
    
    def calculate_group_health(self) -> float:
        """Calculate overall group health score."""
        if not self.consumers:
            return 0.0
        
        consumer_scores = [c.calculate_health_score() for c in self.consumers.values()]
        return statistics.mean(consumer_scores)
    
    def needs_scaling(self) -> Tuple[bool, ScalingAction]:
        """Determine if group needs scaling."""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_scaling_time < self.scaling_cooldown_seconds:
            return False, ScalingAction.NO_ACTION
        
        active_consumers = len([c for c in self.consumers.values() 
                               if c.state in [ConsumerState.HEALTHY, ConsumerState.DEGRADED]])
        
        # Scale up conditions
        if active_consumers < self.target_consumer_count:
            return True, ScalingAction.SCALE_UP
        
        # Scale up on high load
        if (self.total_throughput > 0 and 
            self.average_latency_ms > 200 and  # High latency
            active_consumers < 10):  # Don't scale infinitely
            return True, ScalingAction.SCALE_UP
        
        # Scale down conditions
        if (active_consumers > self.target_consumer_count and
            self.total_throughput < 100 and  # Low throughput
            self.average_latency_ms < 50):   # Low latency
            return True, ScalingAction.SCALE_DOWN
        
        # Rebalance conditions
        if self.load_distribution_score < 0.7:  # Poor load distribution
            return True, ScalingAction.REBALANCE
        
        return False, ScalingAction.NO_ACTION


class ConsumerHealthMonitor:
    """
    Monitors consumer health and performance.
    """
    
    def __init__(
        self,
        redis_client: Redis,
        heartbeat_interval: float = 10.0,  # seconds
        health_check_timeout: float = 5.0,  # seconds
        unhealthy_threshold: int = 3       # consecutive failures
    ):
        self.redis = redis_client
        self.heartbeat_interval = heartbeat_interval
        self.health_check_timeout = health_check_timeout
        self.unhealthy_threshold = unhealthy_threshold
        
        # Monitoring state
        self.consumer_metrics: Dict[str, ConsumerMetrics] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Health check callbacks
        self.health_callbacks: List[Callable[[str, ConsumerState], None]] = []
        
        self.is_running = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start health monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Consumer health monitor started")
    
    async def stop(self) -> None:
        """Stop health monitoring."""
        self.is_running = False
        
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Consumer health monitor stopped")
    
    def register_health_callback(
        self, 
        callback: Callable[[str, ConsumerState], None]
    ) -> None:
        """Register callback for health state changes."""
        self.health_callbacks.append(callback)
    
    async def record_consumer_metrics(
        self,
        consumer_id: str,
        group_name: str,
        stream_name: str,
        messages_processed: int = 0,
        latency_ms: float = 0.0,
        error_occurred: bool = False
    ) -> None:
        """Record consumer performance metrics."""
        current_time = time.time()
        
        if consumer_id not in self.consumer_metrics:
            self.consumer_metrics[consumer_id] = ConsumerMetrics(
                consumer_id=consumer_id,
                group_name=group_name,
                stream_name=stream_name,
                last_heartbeat=current_time
            )
        
        metrics = self.consumer_metrics[consumer_id]
        
        # Update performance metrics
        metrics.messages_processed += messages_processed
        if latency_ms > 0:
            # Update rolling average latency
            if metrics.average_latency_ms == 0:
                metrics.average_latency_ms = latency_ms
            else:
                metrics.average_latency_ms = (metrics.average_latency_ms * 0.9 + latency_ms * 0.1)
        
        # Update error tracking
        if error_occurred:
            metrics.consecutive_errors += 1
        else:
            metrics.consecutive_errors = 0
        
        # Update heartbeat
        metrics.last_heartbeat = current_time
        
        # Calculate throughput (messages per second)
        time_window = 60.0  # 1 minute window
        recent_history = [
            entry for entry in self.health_history[consumer_id]
            if current_time - entry["timestamp"] <= time_window
        ]
        
        if recent_history:
            total_messages = sum(entry["messages_processed"] for entry in recent_history)
            metrics.messages_per_second = total_messages / time_window
        
        # Store in history
        self.health_history[consumer_id].append({
            "timestamp": current_time,
            "messages_processed": messages_processed,
            "latency_ms": latency_ms,
            "error_occurred": error_occurred
        })
        
        # Update consumer state
        await self._update_consumer_state(consumer_id)
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_running:
            try:
                await self._perform_health_checks()
                await self._update_redis_consumer_info()
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all consumers."""
        current_time = time.time()
        
        for consumer_id, metrics in self.consumer_metrics.items():
            # Check heartbeat timeout
            time_since_heartbeat = current_time - metrics.last_heartbeat
            
            if time_since_heartbeat > 120:  # 2 minutes timeout
                await self._mark_consumer_failed(consumer_id)
            elif time_since_heartbeat > 60:  # 1 minute warning
                await self._mark_consumer_unresponsive(consumer_id)
            
            # Check error rate
            error_rate = self._calculate_error_rate(consumer_id)
            metrics.error_rate = error_rate
            
            if error_rate > 0.1:  # 10% error rate
                await self._mark_consumer_degraded(consumer_id)
            elif metrics.consecutive_errors >= self.unhealthy_threshold:
                await self._mark_consumer_degraded(consumer_id)
    
    def _calculate_error_rate(self, consumer_id: str) -> float:
        """Calculate consumer error rate."""
        history = self.health_history[consumer_id]
        if not history:
            return 0.0
        
        # Calculate error rate over last 5 minutes
        current_time = time.time()
        recent_entries = [
            entry for entry in history
            if current_time - entry["timestamp"] <= 300
        ]
        
        if not recent_entries:
            return 0.0
        
        error_count = sum(1 for entry in recent_entries if entry["error_occurred"])
        return error_count / len(recent_entries)
    
    async def _update_consumer_state(self, consumer_id: str) -> None:
        """Update consumer state based on metrics."""
        metrics = self.consumer_metrics[consumer_id]
        old_state = metrics.state
        
        # Determine new state
        health_score = metrics.calculate_health_score()
        
        if health_score >= 0.8:
            new_state = ConsumerState.HEALTHY
        elif health_score >= 0.5:
            new_state = ConsumerState.DEGRADED
        elif health_score >= 0.2:
            new_state = ConsumerState.UNRESPONSIVE
        else:
            new_state = ConsumerState.FAILED
        
        # Update state if changed
        if new_state != old_state:
            metrics.state = new_state
            metrics.last_state_change = time.time()
            
            logger.info(
                "Consumer state changed",
                consumer_id=consumer_id,
                old_state=old_state.value,
                new_state=new_state.value,
                health_score=health_score
            )
            
            # Notify callbacks
            for callback in self.health_callbacks:
                try:
                    callback(consumer_id, new_state)
                except Exception as e:
                    logger.error(f"Error in health callback: {e}")
    
    async def _mark_consumer_failed(self, consumer_id: str) -> None:
        """Mark consumer as failed."""
        if consumer_id in self.consumer_metrics:
            metrics = self.consumer_metrics[consumer_id]
            if metrics.state != ConsumerState.FAILED:
                metrics.state = ConsumerState.FAILED
                metrics.last_state_change = time.time()
                
                logger.warning(f"Consumer {consumer_id} marked as failed")
    
    async def _mark_consumer_unresponsive(self, consumer_id: str) -> None:
        """Mark consumer as unresponsive."""
        if consumer_id in self.consumer_metrics:
            metrics = self.consumer_metrics[consumer_id]
            if metrics.state == ConsumerState.HEALTHY:
                metrics.state = ConsumerState.UNRESPONSIVE
                metrics.last_state_change = time.time()
                
                logger.warning(f"Consumer {consumer_id} marked as unresponsive")
    
    async def _mark_consumer_degraded(self, consumer_id: str) -> None:
        """Mark consumer as degraded."""
        if consumer_id in self.consumer_metrics:
            metrics = self.consumer_metrics[consumer_id]
            if metrics.state == ConsumerState.HEALTHY:
                metrics.state = ConsumerState.DEGRADED
                metrics.last_state_change = time.time()
                
                logger.warning(f"Consumer {consumer_id} marked as degraded")
    
    async def _update_redis_consumer_info(self) -> None:
        """Update Redis with consumer information."""
        try:
            for consumer_id, metrics in self.consumer_metrics.items():
                consumer_info = {
                    "consumer_id": consumer_id,
                    "group_name": metrics.group_name,
                    "stream_name": metrics.stream_name,
                    "state": metrics.state.value,
                    "health_score": metrics.calculate_health_score(),
                    "messages_per_second": metrics.messages_per_second,
                    "average_latency_ms": metrics.average_latency_ms,
                    "error_rate": metrics.error_rate,
                    "last_heartbeat": metrics.last_heartbeat,
                    "last_update": time.time()
                }
                
                await self.redis.setex(
                    f"consumer_health:{consumer_id}",
                    300,  # 5 minute expiry
                    json.dumps(consumer_info)
                )
        
        except Exception as e:
            logger.error(f"Error updating Redis consumer info: {e}")
    
    def get_consumer_metrics(self, consumer_id: str) -> Optional[ConsumerMetrics]:
        """Get metrics for specific consumer."""
        return self.consumer_metrics.get(consumer_id)
    
    def get_all_consumer_metrics(self) -> Dict[str, ConsumerMetrics]:
        """Get all consumer metrics."""
        return self.consumer_metrics.copy()


class ConsumerGroupBalancer:
    """
    Automatically balances load across consumer groups.
    """
    
    def __init__(
        self,
        redis_client: Redis,
        health_monitor: ConsumerHealthMonitor,
        backpressure_manager: Optional[EnterpriseBackPressureManager] = None
    ):
        self.redis = redis_client
        self.health_monitor = health_monitor
        self.backpressure_manager = backpressure_manager
        
        # Group state tracking
        self.group_states: Dict[str, ConsumerGroupState] = {}
        
        # Scaling callbacks
        self.scaling_callbacks: List[Callable[[str, ScalingAction], None]] = []
        
        self.is_running = False
        self.balancer_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start load balancer."""
        if self.is_running:
            return
        
        self.is_running = True
        self.balancer_task = asyncio.create_task(self._balancing_loop())
        
        # Register for health callbacks
        self.health_monitor.register_health_callback(self._on_consumer_state_change)
        
        logger.info("Consumer group balancer started")
    
    async def stop(self) -> None:
        """Stop load balancer."""
        self.is_running = False
        
        if self.balancer_task:
            self.balancer_task.cancel()
            try:
                await self.balancer_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Consumer group balancer stopped")
    
    def register_scaling_callback(
        self, 
        callback: Callable[[str, ScalingAction], None]
    ) -> None:
        """Register callback for scaling actions."""
        self.scaling_callbacks.append(callback)
    
    async def register_consumer_group(
        self,
        group_name: str,
        stream_name: str,
        target_consumer_count: int = 3
    ) -> None:
        """Register a consumer group for management."""
        if group_name not in self.group_states:
            self.group_states[group_name] = ConsumerGroupState(
                group_name=group_name,
                stream_name=stream_name,
                target_consumer_count=target_consumer_count
            )
            
            logger.info(
                "Consumer group registered",
                group_name=group_name,
                stream_name=stream_name,
                target_consumers=target_consumer_count
            )
    
    async def _balancing_loop(self) -> None:
        """Main load balancing loop."""
        while self.is_running:
            try:
                await self._update_group_states()
                await self._perform_load_balancing()
                await self._check_scaling_needs()
                
                await asyncio.sleep(30)  # Balance every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in balancing loop: {e}")
                await asyncio.sleep(30)
    
    async def _update_group_states(self) -> None:
        """Update consumer group states."""
        all_metrics = self.health_monitor.get_all_consumer_metrics()
        
        for group_name, group_state in self.group_states.items():
            # Find consumers for this group
            group_consumers = {
                cid: metrics for cid, metrics in all_metrics.items()
                if metrics.group_name == group_name
            }
            
            group_state.consumers = group_consumers
            
            # Calculate group metrics
            if group_consumers:
                group_state.total_throughput = sum(
                    c.messages_per_second for c in group_consumers.values()
                )
                
                latencies = [c.average_latency_ms for c in group_consumers.values() 
                           if c.average_latency_ms > 0]
                group_state.average_latency_ms = statistics.mean(latencies) if latencies else 0.0
                
                error_rates = [c.error_rate for c in group_consumers.values()]
                group_state.overall_error_rate = statistics.mean(error_rates) if error_rates else 0.0
                
                # Count consumer states
                state_counts = defaultdict(int)
                for consumer in group_consumers.values():
                    state_counts[consumer.state] += 1
                
                group_state.healthy_consumers = state_counts[ConsumerState.HEALTHY]
                group_state.degraded_consumers = state_counts[ConsumerState.DEGRADED]
                group_state.failed_consumers = state_counts[ConsumerState.FAILED]
                
                # Calculate load distribution
                group_state.load_distribution_score = self._calculate_load_distribution(
                    group_consumers
                )
    
    def _calculate_load_distribution(
        self, 
        consumers: Dict[str, ConsumerMetrics]
    ) -> float:
        """Calculate load distribution score (1.0 = perfect balance)."""
        if not consumers:
            return 1.0
        
        throughputs = [c.messages_per_second for c in consumers.values()]
        
        if not throughputs or max(throughputs) == 0:
            return 1.0
        
        # Calculate coefficient of variation (lower is better)
        mean_throughput = statistics.mean(throughputs)
        if mean_throughput == 0:
            return 1.0
        
        std_dev = statistics.stdev(throughputs) if len(throughputs) > 1 else 0
        cv = std_dev / mean_throughput
        
        # Convert to score (1.0 = perfect, 0.0 = terrible)
        return max(0.0, 1.0 - cv)
    
    async def _perform_load_balancing(self) -> None:
        """Perform load balancing across consumer groups."""
        for group_name, group_state in self.group_states.items():
            if group_state.load_distribution_score < 0.7:
                await self._rebalance_group(group_name)
    
    async def _rebalance_group(self, group_name: str) -> None:
        """Rebalance load within a consumer group."""
        group_state = self.group_states[group_name]
        
        logger.info(
            "Rebalancing consumer group",
            group_name=group_name,
            load_distribution_score=group_state.load_distribution_score
        )
        
        # Notify callbacks
        for callback in self.scaling_callbacks:
            try:
                callback(group_name, ScalingAction.REBALANCE)
            except Exception as e:
                logger.error(f"Error in scaling callback: {e}")
        
        group_state.last_scaling_action = ScalingAction.REBALANCE
        group_state.last_scaling_time = time.time()
    
    async def _check_scaling_needs(self) -> None:
        """Check if any groups need scaling."""
        for group_name, group_state in self.group_states.items():
            needs_scaling, action = group_state.needs_scaling()
            
            if needs_scaling:
                await self._execute_scaling_action(group_name, action)
    
    async def _execute_scaling_action(
        self, 
        group_name: str, 
        action: ScalingAction
    ) -> None:
        """Execute scaling action."""
        group_state = self.group_states[group_name]
        
        logger.info(
            "Executing scaling action",
            group_name=group_name,
            action=action.value,
            current_consumers=len(group_state.consumers),
            target_consumers=group_state.target_consumer_count
        )
        
        # Notify callbacks
        for callback in self.scaling_callbacks:
            try:
                callback(group_name, action)
            except Exception as e:
                logger.error(f"Error in scaling callback: {e}")
        
        # Update group state
        group_state.last_scaling_action = action
        group_state.last_scaling_time = time.time()
    
    def _on_consumer_state_change(
        self, 
        consumer_id: str, 
        new_state: ConsumerState
    ) -> None:
        """Handle consumer state changes."""
        if new_state == ConsumerState.FAILED:
            # Find group and trigger scaling if needed
            for group_name, group_state in self.group_states.items():
                if consumer_id in group_state.consumers:
                    # Check if we need emergency scaling
                    active_consumers = len([
                        c for c in group_state.consumers.values()
                        if c.state in [ConsumerState.HEALTHY, ConsumerState.DEGRADED]
                    ])
                    
                    if active_consumers < group_state.target_consumer_count:
                        logger.warning(
                            "Emergency scaling needed due to consumer failure",
                            group_name=group_name,
                            failed_consumer=consumer_id,
                            active_consumers=active_consumers
                        )
                        
                        # Trigger immediate scaling callback
                        for callback in self.scaling_callbacks:
                            try:
                                callback(group_name, ScalingAction.SCALE_UP)
                            except Exception as e:
                                logger.error(f"Error in emergency scaling callback: {e}")
                    break
    
    def get_group_state(self, group_name: str) -> Optional[ConsumerGroupState]:
        """Get state for specific group."""
        return self.group_states.get(group_name)
    
    def get_all_group_states(self) -> Dict[str, ConsumerGroupState]:
        """Get all group states."""
        return self.group_states.copy()


class EnterpriseConsumerGroupManager:
    """
    Enterprise consumer group management system combining health monitoring,
    load balancing, and automatic scaling.
    """
    
    def __init__(
        self,
        redis_client: Redis,
        backpressure_manager: Optional[EnterpriseBackPressureManager] = None
    ):
        self.redis = redis_client
        
        # Initialize components
        self.health_monitor = ConsumerHealthMonitor(redis_client)
        self.load_balancer = ConsumerGroupBalancer(
            redis_client, 
            self.health_monitor,
            backpressure_manager
        )
        
        self.is_running = False
    
    async def start(self) -> None:
        """Start enterprise consumer group management."""
        if self.is_running:
            return
        
        self.is_running = True
        
        await self.health_monitor.start()
        await self.load_balancer.start()
        
        logger.info("Enterprise consumer group manager started")
    
    async def stop(self) -> None:
        """Stop enterprise consumer group management."""
        self.is_running = False
        
        await self.load_balancer.stop()
        await self.health_monitor.stop()
        
        logger.info("Enterprise consumer group manager stopped")
    
    async def register_consumer_group(
        self,
        group_name: str,
        stream_name: str,
        target_consumer_count: int = 3
    ) -> None:
        """Register a consumer group for management."""
        await self.load_balancer.register_consumer_group(
            group_name, stream_name, target_consumer_count
        )
    
    async def record_consumer_activity(
        self,
        consumer_id: str,
        group_name: str,
        stream_name: str,
        messages_processed: int = 0,
        latency_ms: float = 0.0,
        error_occurred: bool = False
    ) -> None:
        """Record consumer activity for monitoring."""
        await self.health_monitor.record_consumer_metrics(
            consumer_id, group_name, stream_name,
            messages_processed, latency_ms, error_occurred
        )
    
    def register_health_callback(
        self, 
        callback: Callable[[str, ConsumerState], None]
    ) -> None:
        """Register callback for health state changes."""
        self.health_monitor.register_health_callback(callback)
    
    def register_scaling_callback(
        self, 
        callback: Callable[[str, ScalingAction], None]
    ) -> None:
        """Register callback for scaling actions."""
        self.load_balancer.register_scaling_callback(callback)
    
    async def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive consumer group status."""
        all_metrics = self.health_monitor.get_all_consumer_metrics()
        all_groups = self.load_balancer.get_all_group_states()
        
        return {
            "is_running": self.is_running,
            "total_consumers": len(all_metrics),
            "total_groups": len(all_groups),
            "consumer_metrics": {
                cid: {
                    "state": metrics.state.value,
                    "health_score": metrics.calculate_health_score(),
                    "messages_per_second": metrics.messages_per_second,
                    "average_latency_ms": metrics.average_latency_ms,
                    "error_rate": metrics.error_rate
                }
                for cid, metrics in all_metrics.items()
            },
            "group_states": {
                gname: {
                    "total_throughput": state.total_throughput,
                    "average_latency_ms": state.average_latency_ms,
                    "overall_error_rate": state.overall_error_rate,
                    "healthy_consumers": state.healthy_consumers,
                    "degraded_consumers": state.degraded_consumers,
                    "failed_consumers": state.failed_consumers,
                    "load_distribution_score": state.load_distribution_score,
                    "last_scaling_action": state.last_scaling_action.value if state.last_scaling_action else None
                }
                for gname, state in all_groups.items()
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        all_groups = self.load_balancer.get_all_group_states()
        
        # Calculate overall health
        group_healths = [group.calculate_group_health() for group in all_groups.values()]
        overall_health = statistics.mean(group_healths) if group_healths else 1.0
        
        # Count issues
        failed_consumers = sum(group.failed_consumers for group in all_groups.values())
        degraded_consumers = sum(group.degraded_consumers for group in all_groups.values())
        
        status = "healthy"
        if overall_health < 0.5 or failed_consumers > 0:
            status = "critical"
        elif overall_health < 0.7 or degraded_consumers > 0:
            status = "degraded"
        
        return {
            "status": status,
            "is_running": self.is_running,
            "overall_health_score": overall_health,
            "total_groups": len(all_groups),
            "failed_consumers": failed_consumers,
            "degraded_consumers": degraded_consumers,
            "recommendations": self._generate_health_recommendations(
                all_groups, overall_health, failed_consumers, degraded_consumers
            )
        }
    
    def _generate_health_recommendations(
        self,
        groups: Dict[str, ConsumerGroupState],
        overall_health: float,
        failed_consumers: int,
        degraded_consumers: int
    ) -> List[str]:
        """Generate health recommendations."""
        recommendations = []
        
        if failed_consumers > 0:
            recommendations.append(
                f"{failed_consumers} consumer(s) have failed. Check logs and restart if needed."
            )
        
        if degraded_consumers > 0:
            recommendations.append(
                f"{degraded_consumers} consumer(s) are degraded. Monitor performance and consider scaling."
            )
        
        for group_name, group in groups.items():
            if group.load_distribution_score < 0.6:
                recommendations.append(
                    f"Group {group_name} has poor load distribution ({group.load_distribution_score:.1%}). Consider rebalancing."
                )
            
            if group.overall_error_rate > 0.05:  # 5% error rate
                recommendations.append(
                    f"Group {group_name} has high error rate ({group.overall_error_rate:.1%}). Check consumer health."
                )
        
        if overall_health < 0.7:
            recommendations.append(
                f"Overall system health is low ({overall_health:.1%}). Consider scaling up or optimizing consumers."
            )
        
        if not recommendations:
            recommendations.append("All consumer groups are healthy and operating normally.")
        
        return recommendations