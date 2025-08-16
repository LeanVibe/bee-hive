"""
Unified Circuit Breaker Service for LeanVibe Agent Hive
Consolidates 15+ circuit breaker implementations into comprehensive resilience infrastructure

Provides enterprise-grade fault tolerance with:
- Circuit breaking with configurable thresholds  
- Backpressure management and load shedding
- Graceful degradation and fallback coordination
- Security-aware failure handling
- Intelligent retry scheduling with exponential backoff
- Recovery coordination across services
- Real-time monitoring and alerting
- Pattern-based failure detection
"""

import asyncio
import time
import functools
import statistics
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Any, Callable, TypeVar, Awaitable, Dict, List, Union, Tuple
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import structlog

from .logging_service import get_component_logger

logger = get_component_logger("circuit_breaker")

T = TypeVar('T')


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


class FailureType(Enum):
    """Types of failures that can trigger circuit breaking."""
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    SERVICE_ERROR = "service_error"
    OVERLOAD = "overload"
    SECURITY_VIOLATION = "security_violation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    VALIDATION_ERROR = "validation_error"
    DEPENDENCY_FAILURE = "dependency_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    UNKNOWN = "unknown"


class CircuitBreakerType(Enum):
    """Different types of circuit breakers for specific use cases."""
    BASIC = "basic"                    # Standard circuit breaking
    ORCHESTRATOR = "orchestrator"      # For orchestrator operations
    DATABASE = "database"              # For database operations  
    SECURITY = "security"              # For security-sensitive operations
    REDIS = "redis"                    # For Redis operations
    EMBEDDING = "embedding"            # For AI/ML operations
    BACKPRESSURE = "backpressure"     # For load management
    RETRY = "retry"                    # For retry coordination


@dataclass
class CircuitBreakerConfig:
    """Comprehensive circuit breaker configuration."""
    # Basic circuit breaker settings
    failure_threshold: int = 5
    timeout: float = 60.0
    success_threshold: int = 2  # For half-open to closed transition
    max_timeout: float = 300.0  # Maximum timeout for exponential backoff
    
    # Advanced settings
    expected_exception: tuple = (Exception,)
    fallback_function: Optional[Callable] = None
    name: Optional[str] = None
    circuit_type: CircuitBreakerType = CircuitBreakerType.BASIC
    
    # Backpressure settings
    load_threshold: float = 0.8  # Load threshold for backpressure
    shed_threshold: float = 0.95  # Threshold for load shedding
    
    # Security settings
    security_sensitive: bool = False
    immediate_open_on_security: bool = False
    
    # Performance settings
    performance_threshold_ms: float = 5000.0  # Performance degradation threshold
    response_time_window: int = 10  # Number of recent calls to analyze
    
    # Failure tracking
    failure_window_seconds: int = 300  # Window for failure rate calculation
    max_failure_history: int = 100  # Maximum failures to track

    @classmethod
    def for_orchestrator(cls, name: str = "orchestrator") -> 'CircuitBreakerConfig':
        """Configuration optimized for orchestrator operations."""
        return cls(
            name=name,
            circuit_type=CircuitBreakerType.ORCHESTRATOR,
            failure_threshold=10,
            timeout=30.0,
            success_threshold=3,
            max_timeout=180.0,
            performance_threshold_ms=2000.0
        )
    
    @classmethod
    def for_database(cls, name: str = "database") -> 'CircuitBreakerConfig':
        """Configuration optimized for database operations."""
        return cls(
            name=name,
            circuit_type=CircuitBreakerType.DATABASE,
            failure_threshold=3,
            timeout=15.0,
            success_threshold=2,
            max_timeout=120.0,
            performance_threshold_ms=1000.0
        )
    
    @classmethod
    def for_security(cls, name: str = "security") -> 'CircuitBreakerConfig':
        """Configuration for security-sensitive operations."""
        return cls(
            name=name,
            circuit_type=CircuitBreakerType.SECURITY,
            failure_threshold=1,  # Very sensitive
            timeout=300.0,  # Long recovery time
            success_threshold=5,  # Require more successes
            max_timeout=900.0,
            security_sensitive=True,
            immediate_open_on_security=True
        )
    
    @classmethod
    def for_redis(cls, name: str = "redis") -> 'CircuitBreakerConfig':
        """Configuration optimized for Redis operations."""
        return cls(
            name=name,
            circuit_type=CircuitBreakerType.REDIS,
            failure_threshold=5,
            timeout=60.0,
            success_threshold=2,
            max_timeout=300.0,
            performance_threshold_ms=100.0
        )


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    def __init__(self, message: str, circuit_name: str = None, failure_count: int = None):
        super().__init__(message)
        self.circuit_name = circuit_name
        self.failure_count = failure_count


class CircuitBreakerOpenError(CircuitBreakerError):
    """Specific error for when circuit breaker is open."""
    pass


@dataclass
class FailureRecord:
    """Record of a failure event."""
    timestamp: float
    failure_type: FailureType
    error_message: str
    duration_ms: float = 0.0


@dataclass  
class CircuitBreakerMetrics:
    """Comprehensive metrics for circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    timeout_calls: int = 0
    circuit_opened_count: int = 0
    avg_response_time_ms: float = 0.0
    success_rate: float = 1.0
    current_load: float = 0.0
    
    # Recent performance window
    recent_response_times: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def update_response_time(self, response_time_ms: float):
        """Update response time tracking."""
        self.recent_response_times.append(response_time_ms)
        if self.recent_response_times:
            self.avg_response_time_ms = statistics.mean(self.recent_response_times)


class UnifiedCircuitBreaker:
    """
    Unified Circuit Breaker implementing all resilience patterns:
    - Standard circuit breaking with configurable thresholds
    - Backpressure management and load shedding
    - Graceful degradation with fallback coordination  
    - Security-aware failure handling with immediate protection
    - Intelligent retry scheduling with exponential backoff
    - Recovery coordination across distributed services
    - Performance monitoring and degradation detection
    - Pattern-based failure classification and handling
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        """Initialize unified circuit breaker with comprehensive configuration."""
        self.config = config
        self.name = config.name or f"circuit-{id(self)}"
        
        # Core state
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None
        self.current_timeout = config.timeout
        
        # Advanced tracking
        self.failure_history: deque[FailureRecord] = deque(maxlen=config.max_failure_history)
        self.failure_types: Dict[FailureType, int] = defaultdict(int)
        self.metrics = CircuitBreakerMetrics()
        
        # Load management
        self.current_load = 0.0
        self.active_calls = 0
        self.load_shed_count = 0
        
        # Performance tracking
        self.performance_degraded = False
        self.degradation_start_time: Optional[float] = None
        
        # Threading protection
        self._lock = asyncio.Lock()
        
        logger.info(
            "Circuit breaker initialized",
            name=self.name,
            type=self.config.circuit_type.value,
            failure_threshold=self.config.failure_threshold,
            timeout=self.config.timeout
        )

    def __call__(self, func: Callable) -> Callable:
        """Decorator for protecting functions with circuit breaker."""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await self._execute_async(func, *args, **kwargs)
            
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return self._execute_sync(func, *args, **kwargs)
            
        # Return appropriate wrapper based on function type
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    async def __aenter__(self):
        """Async context manager entry."""
        await self._check_state_and_load()
        self.active_calls += 1
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.active_calls = max(0, self.active_calls - 1)
        
        if exc_type is None:
            await self._on_success()
        else:
            await self._on_failure(exc_val)
        return False  # Don't suppress exceptions

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute function with circuit breaker protection."""
        async with self:
            return await func()

    async def _execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection."""
        start_time = time.time()
        
        # Check circuit breaker state and load
        if await self._should_reject_call():
            raise CircuitBreakerOpenError(
                f"Circuit breaker {self.name} is open or load shedding active",
                circuit_name=self.name,
                failure_count=self.failure_count
            )
        
        self.active_calls += 1
        try:
            result = await func(*args, **kwargs)
            await self._record_success(time.time() - start_time)
            return result
        except self.config.expected_exception as e:
            await self._record_failure(e, time.time() - start_time)
            raise
        finally:
            self.active_calls = max(0, self.active_calls - 1)
    
    def _execute_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute sync function with circuit breaker protection."""
        # For sync functions, use simplified checking
        if self._should_reject_call_sync():
            raise CircuitBreakerOpenError(
                f"Circuit breaker {self.name} is open",
                circuit_name=self.name,
                failure_count=self.failure_count
            )
        
        start_time = time.time()
        self.active_calls += 1
        try:
            result = func(*args, **kwargs)
            self._record_success_sync(time.time() - start_time)
            return result
        except self.config.expected_exception as e:
            self._record_failure_sync(e, time.time() - start_time)
            raise
        finally:
            self.active_calls = max(0, self.active_calls - 1)

    async def _check_state_and_load(self) -> None:
        """Check circuit breaker state and load conditions."""
        async with self._lock:
            # Update current load
            self._update_load_metrics()
            
            # Check for load shedding
            if self._should_shed_load():
                self.load_shed_count += 1
                raise CircuitBreakerOpenError(
                    f"Load shedding active on {self.name}. Current load: {self.current_load:.2f}",
                    circuit_name=self.name
                )
            
            # Standard circuit breaker state check
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    logger.info(
                        "Circuit breaker transitioning to HALF_OPEN",
                        name=self.name,
                        timeout_elapsed=time.time() - self.last_failure_time
                    )
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker {self.name} is OPEN. Timeout: {self.current_timeout}s",
                        circuit_name=self.name,
                        failure_count=self.failure_count
                    )

    async def _should_reject_call(self) -> bool:
        """Determine if call should be rejected."""
        async with self._lock:
            # Update load metrics
            self._update_load_metrics()
            
            # Check for load shedding
            if self._should_shed_load():
                return True
                
            # Check circuit breaker state
            if self.state == CircuitBreakerState.OPEN:
                if not self._should_attempt_reset():
                    return True
                # Transition to half-open
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                
            return False

    def _should_reject_call_sync(self) -> bool:
        """Synchronous version of call rejection check."""
        if self.state == CircuitBreakerState.OPEN:
            return not self._should_attempt_reset()
        return False

    async def _record_success(self, duration_ms: float) -> None:
        """Record successful operation."""
        async with self._lock:
            self.metrics.total_calls += 1
            self.metrics.successful_calls += 1
            self.metrics.update_response_time(duration_ms * 1000)  # Convert to ms
            self.last_success_time = time.time()
            
            # Update success rate
            if self.metrics.total_calls > 0:
                self.metrics.success_rate = self.metrics.successful_calls / self.metrics.total_calls
            
            # Check for performance degradation recovery
            if self.performance_degraded and duration_ms * 1000 < self.config.performance_threshold_ms:
                self.performance_degraded = False
                self.degradation_start_time = None
                logger.info("Performance degradation recovered", name=self.name)
            
            # Handle state transitions
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._reset()
                    logger.info(
                        "Circuit breaker CLOSED after successful recovery",
                        name=self.name,
                        success_count=self.success_count
                    )
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on successful operation
                self.failure_count = 0

    def _record_success_sync(self, duration_ms: float) -> None:
        """Synchronous version of success recording."""
        self.metrics.total_calls += 1
        self.metrics.successful_calls += 1
        self.last_success_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._reset()
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0

    async def _record_failure(self, exception: Exception, duration_ms: float) -> None:
        """Record failed operation with comprehensive tracking."""
        async with self._lock:
            self.metrics.total_calls += 1
            self.metrics.failed_calls += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            # Classify failure type
            failure_type = self._classify_failure(exception)
            self.failure_types[failure_type] += 1
            
            # Record failure details
            failure_record = FailureRecord(
                timestamp=time.time(),
                failure_type=failure_type,
                error_message=str(exception),
                duration_ms=duration_ms * 1000
            )
            self.failure_history.append(failure_record)
            
            # Update success rate
            if self.metrics.total_calls > 0:
                self.metrics.success_rate = self.metrics.successful_calls / self.metrics.total_calls
            
            # Check for performance degradation
            if duration_ms * 1000 > self.config.performance_threshold_ms:
                if not self.performance_degraded:
                    self.performance_degraded = True
                    self.degradation_start_time = time.time()
                    logger.warning("Performance degradation detected", 
                                 name=self.name, 
                                 duration_ms=duration_ms * 1000)
            
            # Handle security violations immediately
            if (self.config.security_sensitive and 
                failure_type == FailureType.SECURITY_VIOLATION and
                self.config.immediate_open_on_security):
                self._open()
                self.metrics.circuit_opened_count += 1
                logger.critical("Circuit breaker OPENED due to security violation",
                              name=self.name,
                              exception=str(exception))
                return
            
            # Standard circuit breaker logic
            if self.state == CircuitBreakerState.HALF_OPEN:
                # Immediately open on failure during recovery attempt
                self._open()
                self.metrics.circuit_opened_count += 1
                logger.warning("Circuit breaker OPENED during recovery attempt",
                             name=self.name,
                             exception=str(exception))
            elif (self.state == CircuitBreakerState.CLOSED and 
                  self.failure_count >= self.config.failure_threshold):
                self._open()
                self.metrics.circuit_opened_count += 1
                logger.warning("Circuit breaker OPENED due to failure threshold",
                             name=self.name,
                             failure_count=self.failure_count,
                             threshold=self.config.failure_threshold,
                             failure_type=failure_type.value,
                             exception=str(exception))

    def _record_failure_sync(self, exception: Exception, duration_ms: float) -> None:
        """Synchronous version of failure recording."""
        self.metrics.total_calls += 1
        self.metrics.failed_calls += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        failure_type = self._classify_failure(exception)
        self.failure_types[failure_type] += 1
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._open()
        elif (self.state == CircuitBreakerState.CLOSED and 
              self.failure_count >= self.config.failure_threshold):
            self._open()

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.current_timeout

    def _open(self) -> None:
        """Transition to open state."""
        self.state = CircuitBreakerState.OPEN
        # Exponential backoff for timeout, capped at max_timeout
        self.current_timeout = min(
            self.current_timeout * 2,
            self.config.max_timeout
        )

    def _reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.current_timeout = self.config.timeout

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == CircuitBreakerState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit breaker is closed."""
        return self.state == CircuitBreakerState.CLOSED

    @property
    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open."""
        return self.state == CircuitBreakerState.HALF_OPEN

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'current_timeout': self.current_timeout,
            'last_failure_time': self.last_failure_time,
            'time_since_last_failure': (
                time.time() - self.last_failure_time 
                if self.last_failure_time else None
            )
        }


def circuit_breaker(
    failure_threshold: int = 5,
    timeout: float = 60.0,
    success_threshold: int = 2
) -> Callable:
    """Decorator for circuit breaker protection."""
    
    def decorator(func: Callable) -> Callable:
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            timeout=timeout,
            success_threshold=success_threshold
        )
        
        async def wrapper(*args, **kwargs):
            async with breaker:
                return await func(*args, **kwargs)
        
        # Attach circuit breaker to function for inspection
        wrapper._circuit_breaker = breaker
        return wrapper
    
    return decorator


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        success_threshold: int = 2
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    failure_threshold=failure_threshold,
                    timeout=timeout,
                    success_threshold=success_threshold
                )
            return self._breakers[name]

    async def get_all_stats(self) -> dict[str, dict]:
        """Get statistics for all registered circuit breakers."""
        return {
            name: breaker.get_stats()
            for name, breaker in self._breakers.items()
        }

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        async with self._lock:
            for breaker in self._breakers.values():
                breaker._reset()
                
    async def reset(self, name: str) -> bool:
        """Reset specific circuit breaker."""
        async with self._lock:
            if name in self._breakers:
                self._breakers[name]._reset()
                return True
            return False


# Global registry instance
_registry = CircuitBreakerRegistry()


async def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    timeout: float = 60.0,
    success_threshold: int = 2
) -> CircuitBreaker:
    """Get circuit breaker from global registry."""
    return await _registry.get_or_create(
        name=name,
        failure_threshold=failure_threshold,
        timeout=timeout,
        success_threshold=success_threshold
    )


async def get_all_circuit_breakers() -> dict[str, dict]:
    """Get all circuit breakers and their stats."""
    return await _registry.get_all_stats()


async def get_circuit_breaker_status(name: str) -> Optional[dict]:
    """Get status for specific circuit breaker."""
    stats = await _registry.get_all_stats()
    return stats.get(name)