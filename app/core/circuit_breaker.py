"""
Circuit Breaker Pattern Implementation for LeanVibe Agent Hive 2.0 - VS 3.3

Production-ready circuit breaker with intelligent failure detection:
- Configurable failure thresholds and recovery mechanisms
- State management with automatic transitions
- Performance monitoring with <1ms overhead
- Integration with observability system
- Half-open state for intelligent recovery testing
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import structlog

logger = structlog.get_logger()


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failures detected, requests blocked
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 10          # Failures to trigger open state
    success_threshold: int = 5           # Successes to close from half-open
    timeout_seconds: int = 60           # Time before half-open attempt
    monitoring_window_seconds: int = 300 # Window for failure rate calculation
    min_requests_threshold: int = 5      # Minimum requests before evaluation
    failure_rate_threshold: float = 0.5  # Failure rate to trigger open (50%)
    half_open_max_requests: int = 3      # Max requests in half-open state
    
    # Performance settings
    max_processing_time_ms: float = 1.0  # Target <1ms overhead
    enable_adaptive_timeout: bool = True  # Adaptive timeout based on performance
    
    # Observability settings
    enable_metrics: bool = True
    enable_logging: bool = True


@dataclass
class CircuitBreakerMetrics:
    """Metrics tracking for circuit breaker."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    state_transitions: int = 0
    time_in_open_state: float = 0.0
    time_in_half_open_state: float = 0.0
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    
    # Performance metrics
    average_decision_time_ms: float = 0.0
    max_decision_time_ms: float = 0.0
    
    # Time-based metrics
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    last_state_change: Optional[datetime] = None


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after


class CircuitBreaker:
    """
    Circuit breaker pattern implementation with intelligent failure detection.
    
    Features:
    - Automatic state management (closed -> open -> half-open -> closed)
    - Configurable failure thresholds and recovery mechanisms
    - Time-based failure rate analysis
    - Performance monitoring with <1ms overhead target
    - Comprehensive metrics and observability
    """
    
    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 10,
        success_threshold: int = 5,
        timeout_seconds: int = 60,
        config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Unique name for this circuit breaker
            failure_threshold: Number of failures to trigger open state
            success_threshold: Number of successes to close from half-open
            timeout_seconds: Time before attempting recovery
            config: Detailed configuration object
        """
        self.name = name
        self.config = config or CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout_seconds=timeout_seconds
        )
        
        # State management
        self._state = CircuitBreakerState.CLOSED
        self._state_lock = asyncio.Lock()
        
        # Counters
        self._failure_count = 0
        self._success_count = 0
        self._half_open_requests = 0
        
        # Timing
        self._last_failure_time: Optional[datetime] = None
        self._state_changed_time = datetime.utcnow()
        
        # Request tracking for rate-based evaluation
        self._request_history: List[Tuple[datetime, bool]] = []  # (timestamp, success)
        self._history_lock = asyncio.Lock()
        
        # Metrics
        self.metrics = CircuitBreakerMetrics()
        self._decision_times: List[float] = []
        
        # Performance optimization
        self._cached_state_check_time = 0.0
        self._cached_state = CircuitBreakerState.CLOSED
        self._cache_invalidation_time = 0.1  # 100ms cache
        
        logger.info(
            "⚡ Circuit breaker initialized",
            name=self.name,
            failure_threshold=self.config.failure_threshold,
            success_threshold=self.config.success_threshold,
            timeout_seconds=self.config.timeout_seconds
        )
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit breaker is open
        """
        start_time = time.time()
        
        try:
            # Check if request is allowed
            await self._check_request_allowed()
            
            # Execute function
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Record success
            await self.record_success()
            
            return result
            
        except Exception as error:
            # Record failure
            await self.record_failure()
            raise
        
        finally:
            # Record decision time
            decision_time = (time.time() - start_time) * 1000
            await self._record_decision_time(decision_time)
    
    async def record_success(self) -> None:
        """Record a successful operation."""
        start_time = time.time()
        
        async with self._state_lock:
            self._success_count += 1
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = datetime.utcnow()
            
            # Add to request history
            await self._add_to_history(True)
            
            # Handle state transitions based on current state
            if self._state == CircuitBreakerState.HALF_OPEN:
                await self._handle_half_open_success()
            elif self._state == CircuitBreakerState.OPEN:
                # Success in open state shouldn't happen, but handle gracefully
                logger.warning(
                    f"⚠️ Unexpected success in OPEN state for circuit breaker {self.name}"
                )
        
        # Record decision time
        decision_time = (time.time() - start_time) * 1000
        await self._record_decision_time(decision_time)
        
        if self.config.enable_logging:
            logger.debug(
                "✅ Circuit breaker recorded success",
                name=self.name,
                state=self._state.value,
                success_count=self._success_count
            )
    
    async def record_failure(self) -> None:
        """Record a failed operation."""
        start_time = time.time()
        
        async with self._state_lock:
            self._failure_count += 1
            self.metrics.failed_requests += 1
            self._last_failure_time = datetime.utcnow()
            self.metrics.last_failure_time = self._last_failure_time
            
            # Add to request history
            await self._add_to_history(False)
            
            # Check if we should transition to OPEN
            if self._state == CircuitBreakerState.CLOSED:
                await self._check_failure_threshold()
            elif self._state == CircuitBreakerState.HALF_OPEN:
                await self._handle_half_open_failure()
        
        # Record decision time
        decision_time = (time.time() - start_time) * 1000
        await self._record_decision_time(decision_time)
        
        if self.config.enable_logging:
            logger.warning(
                "❌ Circuit breaker recorded failure",
                name=self.name,
                state=self._state.value,
                failure_count=self._failure_count
            )
    
    async def get_state(self) -> CircuitBreakerState:
        """
        Get current circuit breaker state with performance optimization.
        
        Returns:
            Current circuit breaker state
        """
        current_time = time.time()
        
        # Use cached state if still valid
        if (current_time - self._cached_state_check_time) < self._cache_invalidation_time:
            return self._cached_state
        
        async with self._state_lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitBreakerState.OPEN:
                await self._check_timeout()
            
            # Update cache
            self._cached_state = self._state
            self._cached_state_check_time = current_time
            
            return self._state
    
    async def force_open(self) -> None:
        """Force circuit breaker to OPEN state."""
        async with self._state_lock:
            await self._transition_to_open("forced")
        
        logger.warning(f"🔴 Circuit breaker {self.name} forced to OPEN state")
    
    async def force_close(self) -> None:
        """Force circuit breaker to CLOSED state."""
        async with self._state_lock:
            await self._transition_to_closed("forced")
        
        logger.info(f"🟢 Circuit breaker {self.name} forced to CLOSED state")
    
    async def force_half_open(self) -> None:
        """Force circuit breaker to HALF_OPEN state."""
        async with self._state_lock:
            await self._transition_to_half_open("forced")
        
        logger.info(f"🟡 Circuit breaker {self.name} forced to HALF_OPEN state")
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_requests = 0
        self._last_failure_time = None
        self._state_changed_time = datetime.utcnow()
        self._request_history.clear()
        
        # Reset metrics
        self.metrics = CircuitBreakerMetrics()
        self._decision_times.clear()
        
        # Reset cache
        self._cached_state = CircuitBreakerState.CLOSED
        self._cached_state_check_time = 0.0
        
        logger.info(f"🔄 Circuit breaker {self.name} reset to initial state")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker metrics."""
        current_time = datetime.utcnow()
        
        # Calculate failure rate
        failure_rate = 0.0
        if self.metrics.total_requests > 0:
            failure_rate = self.metrics.failed_requests / self.metrics.total_requests
        
        # Calculate time in current state
        time_in_current_state = (current_time - self._state_changed_time).total_seconds()
        
        # Calculate recent failure rate (last 5 minutes)
        recent_failure_rate = self._calculate_recent_failure_rate()
        
        metrics = {
            "name": self.name,
            "state": self._state.value,
            "time_in_current_state_seconds": time_in_current_state,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "total_requests": self.metrics.total_requests,
            "failure_rate": failure_rate,
            "recent_failure_rate": recent_failure_rate,
            "state_transitions": self.metrics.state_transitions,
            "recovery_attempts": self.metrics.recovery_attempts,
            "successful_recoveries": self.metrics.successful_recoveries,
            "performance": {
                "average_decision_time_ms": self.metrics.average_decision_time_ms,
                "max_decision_time_ms": self.metrics.max_decision_time_ms,
                "target_met": self.metrics.average_decision_time_ms <= self.config.max_processing_time_ms
            },
            "configuration": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "monitoring_window_seconds": self.config.monitoring_window_seconds,
                "failure_rate_threshold": self.config.failure_rate_threshold
            },
            "timestamps": {
                "last_failure": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
                "last_success": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
                "last_state_change": self.metrics.last_state_change.isoformat() if self.metrics.last_state_change else None
            }
        }
        
        return metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of circuit breaker."""
        state = await self.get_state()
        metrics = self.get_metrics()
        
        # Determine health status
        if state == CircuitBreakerState.OPEN:
            status = "degraded"
            issues = ["Circuit breaker is open due to failures"]
        elif state == CircuitBreakerState.HALF_OPEN:
            status = "recovering"
            issues = ["Circuit breaker is testing recovery"]
        elif metrics["performance"]["target_met"]:
            status = "healthy"
            issues = []
        else:
            status = "degraded"
            issues = ["Performance target not met"]
        
        return {
            "status": status,
            "state": state.value,
            "issues": issues,
            "metrics": metrics,
            "recommendations": self._get_health_recommendations(metrics)
        }
    
    # Private methods
    
    async def _check_request_allowed(self) -> None:
        """Check if request is allowed through circuit breaker."""
        state = await self.get_state()
        
        if state == CircuitBreakerState.OPEN:
            raise CircuitBreakerError(
                f"Circuit breaker {self.name} is OPEN",
                retry_after=self.config.timeout_seconds
            )
        elif state == CircuitBreakerState.HALF_OPEN:
            async with self._state_lock:
                if self._half_open_requests >= self.config.half_open_max_requests:
                    raise CircuitBreakerError(
                        f"Circuit breaker {self.name} is HALF_OPEN with max requests reached",
                        retry_after=10  # Short retry for half-open
                    )
                self._half_open_requests += 1
        
        # Update total requests
        self.metrics.total_requests += 1
    
    async def _add_to_history(self, success: bool) -> None:
        """Add request result to history for rate-based analysis."""
        async with self._history_lock:
            current_time = datetime.utcnow()
            self._request_history.append((current_time, success))
            
            # Clean old entries outside monitoring window
            cutoff_time = current_time - timedelta(seconds=self.config.monitoring_window_seconds)
            self._request_history = [
                (timestamp, result) for timestamp, result in self._request_history
                if timestamp > cutoff_time
            ]
    
    async def _check_failure_threshold(self) -> None:
        """Check if failure threshold is reached and transition to OPEN if needed."""
        
        # Count-based threshold check
        if self._failure_count >= self.config.failure_threshold:
            await self._transition_to_open("failure_count_threshold")
            return
        
        # Rate-based threshold check
        if len(self._request_history) >= self.config.min_requests_threshold:
            recent_failure_rate = self._calculate_recent_failure_rate()
            if recent_failure_rate >= self.config.failure_rate_threshold:
                await self._transition_to_open("failure_rate_threshold")
    
    async def _check_timeout(self) -> None:
        """Check if timeout has passed and transition to HALF_OPEN if needed."""
        if self._last_failure_time:
            time_since_failure = datetime.utcnow() - self._last_failure_time
            if time_since_failure.total_seconds() >= self.config.timeout_seconds:
                await self._transition_to_half_open("timeout_recovery")
    
    async def _handle_half_open_success(self) -> None:
        """Handle success in HALF_OPEN state."""
        if self._success_count >= self.config.success_threshold:
            await self._transition_to_closed("recovery_success")
            self.metrics.successful_recoveries += 1
    
    async def _handle_half_open_failure(self) -> None:
        """Handle failure in HALF_OPEN state."""
        await self._transition_to_open("half_open_failure")
    
    async def _transition_to_open(self, reason: str) -> None:
        """Transition to OPEN state."""
        if self._state != CircuitBreakerState.OPEN:
            old_state = self._state
            self._state = CircuitBreakerState.OPEN
            self._state_changed_time = datetime.utcnow()
            self.metrics.state_transitions += 1
            self.metrics.last_state_change = self._state_changed_time
            
            # Reset half-open counters
            self._half_open_requests = 0
            
            # Invalidate cache
            self._cached_state_check_time = 0.0
            
            if self.config.enable_logging:
                logger.warning(
                    f"🔴 Circuit breaker {self.name} transitioned to OPEN",
                    reason=reason,
                    previous_state=old_state.value,
                    failure_count=self._failure_count,
                    timeout_seconds=self.config.timeout_seconds
                )
    
    async def _transition_to_half_open(self, reason: str) -> None:
        """Transition to HALF_OPEN state."""
        if self._state != CircuitBreakerState.HALF_OPEN:
            old_state = self._state
            self._state = CircuitBreakerState.HALF_OPEN
            self._state_changed_time = datetime.utcnow()
            self.metrics.state_transitions += 1
            self.metrics.last_state_change = self._state_changed_time
            self.metrics.recovery_attempts += 1
            
            # Reset counters
            self._success_count = 0
            self._half_open_requests = 0
            
            # Invalidate cache
            self._cached_state_check_time = 0.0
            
            if self.config.enable_logging:
                logger.info(
                    f"🟡 Circuit breaker {self.name} transitioned to HALF_OPEN",
                    reason=reason,
                    previous_state=old_state.value,
                    max_test_requests=self.config.half_open_max_requests
                )
    
    async def _transition_to_closed(self, reason: str) -> None:
        """Transition to CLOSED state."""
        if self._state != CircuitBreakerState.CLOSED:
            old_state = self._state
            self._state = CircuitBreakerState.CLOSED
            self._state_changed_time = datetime.utcnow()
            self.metrics.state_transitions += 1
            self.metrics.last_state_change = self._state_changed_time
            
            # Reset counters
            self._failure_count = 0
            self._success_count = 0
            self._half_open_requests = 0
            
            # Invalidate cache
            self._cached_state_check_time = 0.0
            
            if self.config.enable_logging:
                logger.info(
                    f"🟢 Circuit breaker {self.name} transitioned to CLOSED",
                    reason=reason,
                    previous_state=old_state.value
                )
    
    def _calculate_recent_failure_rate(self) -> float:
        """Calculate recent failure rate within monitoring window."""
        if not self._request_history:
            return 0.0
        
        total_requests = len(self._request_history)
        failed_requests = sum(1 for _, success in self._request_history if not success)
        
        return failed_requests / total_requests if total_requests > 0 else 0.0
    
    async def _record_decision_time(self, decision_time_ms: float) -> None:
        """Record decision time for performance monitoring."""
        self._decision_times.append(decision_time_ms)
        
        # Keep only recent decisions for performance
        if len(self._decision_times) > 1000:
            self._decision_times = self._decision_times[-500:]
        
        # Update metrics
        if self._decision_times:
            self.metrics.average_decision_time_ms = sum(self._decision_times) / len(self._decision_times)
            self.metrics.max_decision_time_ms = max(self._decision_times)
    
    def _get_health_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Get health recommendations based on current metrics."""
        recommendations = []
        
        if metrics["state"] == "open":
            recommendations.append("Investigate underlying service issues causing failures")
            recommendations.append("Consider increasing timeout or failure threshold if appropriate")
        
        if metrics["recent_failure_rate"] > 0.3:
            recommendations.append("High recent failure rate detected - monitor service health")
        
        if not metrics["performance"]["target_met"]:
            recommendations.append("Circuit breaker decision time exceeds target - consider optimization")
        
        if metrics["state_transitions"] > 10:
            recommendations.append("Frequent state transitions detected - review thresholds")
        
        return recommendations


# Global circuit breaker registry for management
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 10,
    success_threshold: int = 5,
    timeout_seconds: int = 60,
    config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """
    Get or create a circuit breaker instance.
    
    Args:
        name: Unique name for the circuit breaker
        failure_threshold: Number of failures to trigger open state
        success_threshold: Number of successes to close from half-open
        timeout_seconds: Time before attempting recovery
        config: Detailed configuration object
        
    Returns:
        Circuit breaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            success_threshold=success_threshold,
            timeout_seconds=timeout_seconds,
            config=config
        )
    
    return _circuit_breakers[name]


def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Get all registered circuit breakers."""
    return _circuit_breakers.copy()


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers to initial state."""
    for circuit_breaker in _circuit_breakers.values():
        circuit_breaker.reset()
    
    logger.info("🔄 All circuit breakers reset")


async def get_circuit_breaker_status() -> Dict[str, Any]:
    """Get status of all circuit breakers."""
    status = {
        "total_circuit_breakers": len(_circuit_breakers),
        "circuit_breakers": {}
    }
    
    for name, cb in _circuit_breakers.items():
        cb_status = await cb.health_check()
        status["circuit_breakers"][name] = cb_status
    
    return status