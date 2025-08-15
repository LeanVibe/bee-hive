"""
Circuit Breaker Pattern Implementation for Production Orchestrator

Provides fault tolerance and resilience by preventing cascading failures
in distributed systems through automatic failure detection and recovery.
"""

import asyncio
import time
from enum import Enum
from typing import Optional, Any, Callable, TypeVar, Awaitable
from dataclasses import dataclass
from contextlib import asynccontextmanager
import structlog

logger = structlog.get_logger()

T = TypeVar('T')


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    timeout: float = 60.0
    success_threshold: int = 2  # For half-open to closed transition
    max_timeout: float = 300.0  # Maximum timeout for exponential backoff


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    Monitors failures and automatically opens to prevent cascading failures,
    then attempts recovery after a timeout period.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        success_threshold: int = 2,
        max_timeout: float = 300.0
    ):
        """Initialize circuit breaker."""
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            timeout=timeout,
            success_threshold=success_threshold,
            max_timeout=max_timeout
        )
        
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.current_timeout = timeout
        
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        """Async context manager entry."""
        await self._check_state()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_type is None:
            await self._on_success()
        else:
            await self._on_failure(exc_val)
        return False  # Don't suppress exceptions

    async def call(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute function with circuit breaker protection."""
        async with self:
            return await func()

    async def _check_state(self) -> None:
        """Check and update circuit breaker state."""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker is OPEN. Timeout: {self.current_timeout}s"
                    )

    async def _on_success(self) -> None:
        """Handle successful operation."""
        async with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._reset()
                    logger.info("Circuit breaker CLOSED after successful recovery")
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on successful operation
                self.failure_count = 0

    async def _on_failure(self, exception: Exception) -> None:
        """Handle failed operation."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                # Immediately open on failure during recovery attempt
                self._open()
                logger.warning("Circuit breaker OPENED during recovery attempt", 
                             exception=str(exception))
            elif (self.state == CircuitBreakerState.CLOSED and 
                  self.failure_count >= self.config.failure_threshold):
                self._open()
                logger.warning("Circuit breaker OPENED due to failure threshold", 
                             failure_count=self.failure_count,
                             threshold=self.config.failure_threshold,
                             exception=str(exception))

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