"""
Retry Policies for Production Orchestrator

Implements various retry strategies with exponential backoff, jitter,
and circuit breaker integration for robust error handling.
"""

import asyncio
import random
import time
from typing import Callable, TypeVar, Awaitable, Union, Type, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger()

T = TypeVar('T')


class RetryStrategy(Enum):
    """Available retry strategies."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    FIBONACCI_BACKOFF = "fibonacci_backoff"


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    jitter_range: float = 0.1  # ±10% jitter
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)


class RetryExhaustedError(Exception):
    """Raised when all retry attempts have been exhausted."""
    
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Retry exhausted after {attempts} attempts. "
            f"Last exception: {last_exception}"
        )


@dataclass
class RetryResult:
    """Result of a retry operation."""
    success: bool
    result: Optional[Any] = None
    exception: Optional[Exception] = None
    attempts: int = 0
    total_time: float = 0.0
    
    @property
    def failed(self) -> bool:
        """Check if retry failed."""
        return not self.success


class RetryPolicy:
    """Base retry policy implementation."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """Initialize retry policy."""
        self.config = config or RetryConfig()
        
    async def execute(
        self,
        func: Callable[[], Awaitable[T]],
        *args,
        **kwargs
    ) -> T:
        """Execute function with retry policy."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt)
                    logger.debug("Retrying after delay",
                               attempt=attempt,
                               delay=delay,
                               function=func.__name__)
                    await asyncio.sleep(delay)
                
                result = await func(*args, **kwargs)
                
                if attempt > 0:
                    logger.info("Retry successful",
                              attempt=attempt,
                              function=func.__name__)
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not self._is_retryable(e):
                    logger.debug("Non-retryable exception, not retrying",
                               exception=str(e),
                               exception_type=type(e).__name__)
                    raise
                
                if attempt == self.config.max_retries:
                    logger.warning("Retry exhausted",
                                 attempts=attempt + 1,
                                 function=func.__name__,
                                 last_exception=str(e))
                    break
                
                logger.warning("Attempt failed, will retry",
                             attempt=attempt + 1,
                             max_retries=self.config.max_retries,
                             function=func.__name__,
                             exception=str(e))
        
        raise RetryExhaustedError(
            attempts=self.config.max_retries + 1,
            last_exception=last_exception
        )
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt based on strategy."""
        if self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (
                self.config.backoff_multiplier ** (attempt - 1)
            )
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        elif self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = self.config.base_delay * self._fibonacci(attempt)
        else:
            delay = self.config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Apply jitter if enabled
        if self.config.jitter:
            jitter_amount = delay * self.config.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)  # Ensure delay is not negative
        
        return delay
    
    def _fibonacci(self, n: int) -> int:
        """Calculate nth Fibonacci number."""
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
    
    def _is_retryable(self, exception: Exception) -> bool:
        """Check if exception is retryable."""
        return any(
            isinstance(exception, exc_type) 
            for exc_type in self.config.retryable_exceptions
        )


def exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_multiplier: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """Decorator for exponential backoff retry."""
    
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        backoff_multiplier=backoff_multiplier,
        jitter=jitter,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        retryable_exceptions=retryable_exceptions
    )
    
    def decorator(func: Callable) -> Callable:
        policy = RetryPolicy(config)
        
        async def wrapper(*args, **kwargs):
            return await policy.execute(
                lambda: func(*args, **kwargs)
            )
        
        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._retry_policy = policy
        
        return wrapper
    
    return decorator


def linear_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """Decorator for linear backoff retry."""
    
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        jitter=jitter,
        strategy=RetryStrategy.LINEAR_BACKOFF,
        retryable_exceptions=retryable_exceptions
    )
    
    def decorator(func: Callable) -> Callable:
        policy = RetryPolicy(config)
        
        async def wrapper(*args, **kwargs):
            return await policy.execute(
                lambda: func(*args, **kwargs)
            )
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._retry_policy = policy
        
        return wrapper
    
    return decorator


def fixed_delay(
    max_retries: int = 3,
    delay: float = 1.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """Decorator for fixed delay retry."""
    
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=delay,
        max_delay=delay,
        jitter=jitter,
        strategy=RetryStrategy.FIXED_DELAY,
        retryable_exceptions=retryable_exceptions
    )
    
    def decorator(func: Callable) -> Callable:
        policy = RetryPolicy(config)
        
        async def wrapper(*args, **kwargs):
            return await policy.execute(
                lambda: func(*args, **kwargs)
            )
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._retry_policy = policy
        
        return wrapper
    
    return decorator


def fibonacci_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable:
    """Decorator for Fibonacci sequence backoff retry."""
    
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        jitter=jitter,
        strategy=RetryStrategy.FIBONACCI_BACKOFF,
        retryable_exceptions=retryable_exceptions
    )
    
    def decorator(func: Callable) -> Callable:
        policy = RetryPolicy(config)
        
        async def wrapper(*args, **kwargs):
            return await policy.execute(
                lambda: func(*args, **kwargs)
            )
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper._retry_policy = policy
        
        return wrapper
    
    return decorator


class RetryableOperation:
    """Context manager for retryable operations."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """Initialize retryable operation."""
        self.policy = RetryPolicy(config)
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return False  # Don't suppress exceptions
    
    async def execute(self, func: Callable[[], Awaitable[T]]) -> T:
        """Execute function with retry policy."""
        return await self.policy.execute(func)


# Predefined retry policies for common scenarios
DATABASE_RETRY = RetryConfig(
    max_retries=3,
    base_delay=0.5,
    max_delay=10.0,
    backoff_multiplier=2.0,
    jitter=True,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        # Add database-specific exceptions here
    )
)

API_RETRY = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    backoff_multiplier=2.0,
    jitter=True,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        # Add HTTP-specific exceptions here
    )
)

REDIS_RETRY = RetryConfig(
    max_retries=2,
    base_delay=0.1,
    max_delay=5.0,
    backoff_multiplier=3.0,
    jitter=True,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    retryable_exceptions=(
        ConnectionError,
        TimeoutError,
        # Add Redis-specific exceptions here
    )
)


async def retry_with_policy(
    func: Callable[[], Awaitable[T]],
    config: Optional[RetryConfig] = None
) -> T:
    """Utility function to retry with custom policy."""
    policy = RetryPolicy(config)
    return await policy.execute(func)


class RetryMetrics:
    """Metrics collection for retry operations."""
    
    def __init__(self):
        self.total_attempts = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.retry_counts = {}
        
    def record_attempt(self, attempt_number: int, success: bool):
        """Record retry attempt."""
        self.total_attempts += 1
        
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
            
        if attempt_number not in self.retry_counts:
            self.retry_counts[attempt_number] = 0
        self.retry_counts[attempt_number] += 1
    
    def get_stats(self) -> dict:
        """Get retry statistics."""
        success_rate = (
            self.successful_operations / self.total_attempts
            if self.total_attempts > 0 else 0
        )
        
        return {
            'total_attempts': self.total_attempts,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'success_rate': success_rate,
            'retry_distribution': self.retry_counts
        }


# Global metrics instance
_retry_metrics = RetryMetrics()


def get_retry_metrics() -> RetryMetrics:
    """Get global retry metrics."""
    return _retry_metrics


class JitterType(Enum):
    """Types of jitter for retry delays."""
    NONE = "none"
    UNIFORM = "uniform"
    EXPONENTIAL = "exponential"


class RetryPolicyFactory:
    """Factory for creating retry policies."""
    
    @staticmethod
    def create_exponential_backoff(
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_multiplier: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ) -> RetryPolicy:
        """Create exponential backoff retry policy."""
        config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_multiplier=backoff_multiplier,
            jitter=jitter,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            retryable_exceptions=retryable_exceptions
        )
        return RetryPolicy(config)
    
    @staticmethod
    def create_linear_backoff(
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ) -> RetryPolicy:
        """Create linear backoff retry policy."""
        config = RetryConfig(
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter=jitter,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            retryable_exceptions=retryable_exceptions
        )
        return RetryPolicy(config)
    
    @staticmethod
    def create_fixed_delay(
        max_retries: int = 3,
        delay: float = 1.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ) -> RetryPolicy:
        """Create fixed delay retry policy."""
        config = RetryConfig(
            max_retries=max_retries,
            base_delay=delay,
            max_delay=delay,
            jitter=jitter,
            strategy=RetryStrategy.FIXED_DELAY,
            retryable_exceptions=retryable_exceptions
        )
        return RetryPolicy(config)


class RetryExecutor:
    """Executor for retry operations with metrics and monitoring."""
    
    def __init__(self, metrics: Optional[RetryMetrics] = None):
        """Initialize retry executor."""
        self.metrics = metrics or _retry_metrics
        
    async def execute_with_retry(
        self,
        func: Callable[[], Awaitable[T]],
        policy: RetryPolicy
    ) -> T:
        """Execute function with retry policy and metrics collection."""
        attempt = 0
        last_exception = None
        
        while attempt <= policy.config.max_retries:
            try:
                result = await func()
                self.metrics.record_attempt(attempt, success=True)
                return result
            except Exception as e:
                last_exception = e
                attempt += 1
                self.metrics.record_attempt(attempt, success=False)
                
                if attempt > policy.config.max_retries:
                    break
                    
                if not policy._is_retryable(e):
                    raise
                    
                delay = policy._calculate_delay(attempt)
                await asyncio.sleep(delay)
        
        raise RetryExhaustedError(attempt, last_exception)