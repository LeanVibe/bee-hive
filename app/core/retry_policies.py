"""
Configurable Retry Logic System for LeanVibe Agent Hive 2.0 - VS 3.3

Production-ready retry policies with exponential backoff and jitter:
- Multiple retry strategies (exponential, linear, fixed, adaptive)
- Intelligent jitter algorithms for avoiding thundering herd
- Configurable retry conditions and stop conditions
- Performance monitoring with minimal overhead
- Integration with circuit breaker and observability systems
"""

import asyncio
import time
import random
import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import structlog

logger = structlog.get_logger()


class RetryStrategy(Enum):
    """Retry strategy types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    ADAPTIVE = "adaptive"
    FIBONACCI = "fibonacci"


class JitterType(Enum):
    """Jitter algorithm types."""
    NONE = "none"
    FULL = "full"           # Random between 0 and calculated delay
    EQUAL = "equal"         # Random between delay/2 and delay
    DECORRELATED = "decorrelated"  # Decorrelated jitter algorithm


class StopCondition(Enum):
    """Stop condition types for retry logic."""
    MAX_ATTEMPTS = "max_attempts"
    MAX_DURATION = "max_duration"
    EXPONENTIAL_BACKOFF_LIMIT = "exponential_backoff_limit"
    CUSTOM_CONDITION = "custom_condition"


@dataclass
class RetryResult:
    """Result of retry delay calculation."""
    should_retry: bool
    delay_ms: float
    attempt_number: int
    total_elapsed_ms: float
    next_delay_ms: Optional[float] = None
    reason: Optional[str] = None


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay_ms: int = 100
    max_delay_ms: int = 30000  # 30 seconds
    max_duration_ms: int = 300000  # 5 minutes
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter_type: JitterType = JitterType.EQUAL
    backoff_multiplier: float = 2.0
    jitter_factor: float = 0.1
    
    # Adaptive strategy parameters
    adaptive_success_threshold: float = 0.8
    adaptive_failure_threshold: float = 0.3
    adaptive_adjustment_factor: float = 1.5
    
    # Fibonacci sequence parameters
    fibonacci_max_sequence: int = 20
    
    # Custom conditions
    custom_stop_condition: Optional[Callable] = None
    custom_retry_condition: Optional[Callable] = None
    
    # Performance settings
    enable_metrics: bool = True
    enable_logging: bool = True


@dataclass
class RetryMetrics:
    """Metrics for retry policy performance."""
    total_attempts: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    total_delay_time_ms: float = 0.0
    average_delay_ms: float = 0.0
    max_delay_ms: float = 0.0
    strategy_adjustments: int = 0
    
    # Time-based metrics
    first_attempt_time: Optional[datetime] = None
    last_attempt_time: Optional[datetime] = None


class RetryPolicy(ABC):
    """Abstract base class for retry policies."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.metrics = RetryMetrics()
        self._start_time: Optional[datetime] = None
        self._attempt_history: List[Tuple[datetime, bool, float]] = []  # (time, success, delay)
    
    @abstractmethod
    async def calculate_delay(self, attempt: int, last_error: Optional[Exception] = None) -> RetryResult:
        """Calculate delay for retry attempt."""
        pass
    
    async def should_retry(
        self,
        attempt: int,
        error: Optional[Exception] = None,
        elapsed_time_ms: Optional[float] = None
    ) -> bool:
        """Determine if retry should be attempted."""
        
        # Check max attempts
        if attempt >= self.config.max_attempts:
            return False
        
        # Check max duration
        if elapsed_time_ms and elapsed_time_ms >= self.config.max_duration_ms:
            return False
        
        # Check custom retry condition
        if self.config.custom_retry_condition:
            return await self.config.custom_retry_condition(attempt, error, elapsed_time_ms)
        
        # Default: retry if we haven't exceeded limits
        return True
    
    def _apply_jitter(self, delay_ms: float, attempt: int) -> float:
        """Apply jitter to delay based on configured jitter type."""
        
        if self.config.jitter_type == JitterType.NONE:
            return delay_ms
        
        elif self.config.jitter_type == JitterType.FULL:
            # Random between 0 and delay
            return random.uniform(0, delay_ms)
        
        elif self.config.jitter_type == JitterType.EQUAL:
            # Random between delay/2 and delay
            return random.uniform(delay_ms / 2, delay_ms)
        
        elif self.config.jitter_type == JitterType.DECORRELATED:
            # Decorrelated jitter algorithm
            if not hasattr(self, '_last_delay'):
                self._last_delay = delay_ms
            
            # sleep = min(cap, random_between(base, last_sleep * 3))
            jittered_delay = random.uniform(
                self.config.base_delay_ms,
                min(self.config.max_delay_ms, self._last_delay * 3)
            )
            self._last_delay = jittered_delay
            return jittered_delay
        
        return delay_ms
    
    def _record_attempt(self, attempt: int, success: bool, delay_ms: float) -> None:
        """Record attempt for metrics and adaptive behavior."""
        current_time = datetime.utcnow()
        
        if self._start_time is None:
            self._start_time = current_time
            self.metrics.first_attempt_time = current_time
        
        self.metrics.last_attempt_time = current_time
        self.metrics.total_attempts += 1
        
        if success:
            self.metrics.successful_retries += 1
        else:
            self.metrics.failed_retries += 1
        
        self.metrics.total_delay_time_ms += delay_ms
        self.metrics.average_delay_ms = self.metrics.total_delay_time_ms / self.metrics.total_attempts
        self.metrics.max_delay_ms = max(self.metrics.max_delay_ms, delay_ms)
        
        # Add to history for adaptive behavior
        self._attempt_history.append((current_time, success, delay_ms))
        
        # Keep history manageable
        if len(self._attempt_history) > 100:
            self._attempt_history = self._attempt_history[-50:]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retry policy metrics."""
        return {
            "strategy": self.config.strategy.value,
            "total_attempts": self.metrics.total_attempts,
            "successful_retries": self.metrics.successful_retries,
            "failed_retries": self.metrics.failed_retries,
            "success_rate": self.metrics.successful_retries / max(1, self.metrics.total_attempts),
            "average_delay_ms": self.metrics.average_delay_ms,
            "max_delay_ms": self.metrics.max_delay_ms,
            "total_delay_time_ms": self.metrics.total_delay_time_ms,
            "strategy_adjustments": self.metrics.strategy_adjustments,
            "configuration": {
                "max_attempts": self.config.max_attempts,
                "base_delay_ms": self.config.base_delay_ms,
                "max_delay_ms": self.config.max_delay_ms,
                "jitter_type": self.config.jitter_type.value,
                "backoff_multiplier": self.config.backoff_multiplier
            }
        }


class ExponentialBackoffPolicy(RetryPolicy):
    """Exponential backoff retry policy with jitter."""
    
    async def calculate_delay(self, attempt: int, last_error: Optional[Exception] = None) -> RetryResult:
        """Calculate exponential backoff delay."""
        
        # Calculate base exponential delay
        exponential_delay = self.config.base_delay_ms * (self.config.backoff_multiplier ** attempt)
        
        # Apply maximum delay limit
        capped_delay = min(exponential_delay, self.config.max_delay_ms)
        
        # Apply jitter
        jittered_delay = self._apply_jitter(capped_delay, attempt)
        
        # Calculate elapsed time
        elapsed_time = 0.0
        if self._start_time:
            elapsed_time = (datetime.utcnow() - self._start_time).total_seconds() * 1000
        
        # Check if should retry
        should_retry = await self.should_retry(attempt, last_error, elapsed_time)
        
        # Calculate next delay for preview
        next_delay = None
        if should_retry and attempt + 1 < self.config.max_attempts:
            next_exponential = self.config.base_delay_ms * (self.config.backoff_multiplier ** (attempt + 1))
            next_delay = min(next_exponential, self.config.max_delay_ms)
        
        result = RetryResult(
            should_retry=should_retry,
            delay_ms=jittered_delay,
            attempt_number=attempt,
            total_elapsed_ms=elapsed_time,
            next_delay_ms=next_delay,
            reason=f"exponential_backoff_attempt_{attempt}"
        )
        
        # Record metrics
        self._record_attempt(attempt, should_retry, jittered_delay)
        
        if self.config.enable_logging:
            logger.debug(
                "⏱️ Exponential backoff delay calculated",
                attempt=attempt,
                delay_ms=round(jittered_delay, 2),
                base_delay=exponential_delay,
                capped_delay=capped_delay,
                should_retry=should_retry,
                elapsed_time_ms=round(elapsed_time, 2)
            )
        
        return result


class LinearBackoffPolicy(RetryPolicy):
    """Linear backoff retry policy with jitter."""
    
    async def calculate_delay(self, attempt: int, last_error: Optional[Exception] = None) -> RetryResult:
        """Calculate linear backoff delay."""
        
        # Calculate linear delay
        linear_delay = self.config.base_delay_ms * (1 + attempt * self.config.backoff_multiplier)
        
        # Apply maximum delay limit
        capped_delay = min(linear_delay, self.config.max_delay_ms)
        
        # Apply jitter
        jittered_delay = self._apply_jitter(capped_delay, attempt)
        
        # Calculate elapsed time
        elapsed_time = 0.0
        if self._start_time:
            elapsed_time = (datetime.utcnow() - self._start_time).total_seconds() * 1000
        
        # Check if should retry
        should_retry = await self.should_retry(attempt, last_error, elapsed_time)
        
        result = RetryResult(
            should_retry=should_retry,
            delay_ms=jittered_delay,
            attempt_number=attempt,
            total_elapsed_ms=elapsed_time,
            reason=f"linear_backoff_attempt_{attempt}"
        )
        
        # Record metrics
        self._record_attempt(attempt, should_retry, jittered_delay)
        
        return result


class FixedDelayPolicy(RetryPolicy):
    """Fixed delay retry policy with optional jitter."""
    
    async def calculate_delay(self, attempt: int, last_error: Optional[Exception] = None) -> RetryResult:
        """Calculate fixed delay."""
        
        # Use base delay as fixed delay
        fixed_delay = self.config.base_delay_ms
        
        # Apply jitter
        jittered_delay = self._apply_jitter(fixed_delay, attempt)
        
        # Calculate elapsed time
        elapsed_time = 0.0
        if self._start_time:
            elapsed_time = (datetime.utcnow() - self._start_time).total_seconds() * 1000
        
        # Check if should retry
        should_retry = await self.should_retry(attempt, last_error, elapsed_time)
        
        result = RetryResult(
            should_retry=should_retry,
            delay_ms=jittered_delay,
            attempt_number=attempt,
            total_elapsed_ms=elapsed_time,
            reason=f"fixed_delay_attempt_{attempt}"
        )
        
        # Record metrics
        self._record_attempt(attempt, should_retry, jittered_delay)
        
        return result


class AdaptiveBackoffPolicy(RetryPolicy):
    """Adaptive backoff policy that adjusts based on success/failure patterns."""
    
    def __init__(self, config: RetryConfig):
        super().__init__(config)
        self._current_multiplier = config.backoff_multiplier
        self._adjustment_history: List[float] = []
    
    async def calculate_delay(self, attempt: int, last_error: Optional[Exception] = None) -> RetryResult:
        """Calculate adaptive backoff delay."""
        
        # Adjust multiplier based on recent success/failure pattern
        await self._adjust_multiplier()
        
        # Calculate adaptive delay
        adaptive_delay = self.config.base_delay_ms * (self._current_multiplier ** attempt)
        
        # Apply maximum delay limit
        capped_delay = min(adaptive_delay, self.config.max_delay_ms)
        
        # Apply jitter
        jittered_delay = self._apply_jitter(capped_delay, attempt)
        
        # Calculate elapsed time
        elapsed_time = 0.0
        if self._start_time:
            elapsed_time = (datetime.utcnow() - self._start_time).total_seconds() * 1000
        
        # Check if should retry
        should_retry = await self.should_retry(attempt, last_error, elapsed_time)
        
        result = RetryResult(
            should_retry=should_retry,
            delay_ms=jittered_delay,
            attempt_number=attempt,
            total_elapsed_ms=elapsed_time,
            reason=f"adaptive_backoff_attempt_{attempt}_multiplier_{self._current_multiplier:.2f}"
        )
        
        # Record metrics
        self._record_attempt(attempt, should_retry, jittered_delay)
        
        if self.config.enable_logging:
            logger.debug(
                "🔄 Adaptive backoff delay calculated",
                attempt=attempt,
                delay_ms=round(jittered_delay, 2),
                current_multiplier=round(self._current_multiplier, 2),
                should_retry=should_retry
            )
        
        return result
    
    async def _adjust_multiplier(self) -> None:
        """Adjust backoff multiplier based on recent performance."""
        
        if len(self._attempt_history) < 5:  # Need minimum history
            return
        
        # Calculate recent success rate (last 10 attempts)
        recent_attempts = self._attempt_history[-10:]
        success_count = sum(1 for _, success, _ in recent_attempts if success)
        success_rate = success_count / len(recent_attempts)
        
        old_multiplier = self._current_multiplier
        
        # Adjust multiplier based on success rate
        if success_rate >= self.config.adaptive_success_threshold:
            # High success rate - reduce delays (more aggressive retries)
            self._current_multiplier = max(
                1.1,  # Minimum multiplier
                self._current_multiplier / self.config.adaptive_adjustment_factor
            )
        elif success_rate <= self.config.adaptive_failure_threshold:
            # High failure rate - increase delays (more conservative retries)
            self._current_multiplier = min(
                5.0,  # Maximum multiplier
                self._current_multiplier * self.config.adaptive_adjustment_factor
            )
        
        # Record adjustment if changed
        if abs(old_multiplier - self._current_multiplier) > 0.01:
            self.metrics.strategy_adjustments += 1
            self._adjustment_history.append(self._current_multiplier)
            
            if self.config.enable_logging:
                logger.info(
                    "📊 Adaptive retry multiplier adjusted",
                    old_multiplier=round(old_multiplier, 2),
                    new_multiplier=round(self._current_multiplier, 2),
                    success_rate=round(success_rate, 2),
                    recent_attempts=len(recent_attempts)
                )


class FibonacciBackoffPolicy(RetryPolicy):
    """Fibonacci sequence backoff retry policy."""
    
    def __init__(self, config: RetryConfig):
        super().__init__(config)
        self._fibonacci_sequence = self._generate_fibonacci_sequence(config.fibonacci_max_sequence)
    
    def _generate_fibonacci_sequence(self, max_length: int) -> List[int]:
        """Generate Fibonacci sequence up to max_length."""
        if max_length <= 0:
            return [1]
        elif max_length == 1:
            return [1, 1]
        
        sequence = [1, 1]
        for i in range(2, max_length):
            sequence.append(sequence[i-1] + sequence[i-2])
        
        return sequence
    
    async def calculate_delay(self, attempt: int, last_error: Optional[Exception] = None) -> RetryResult:
        """Calculate Fibonacci backoff delay."""
        
        # Get Fibonacci multiplier
        fib_index = min(attempt, len(self._fibonacci_sequence) - 1)
        fib_multiplier = self._fibonacci_sequence[fib_index]
        
        # Calculate delay
        fibonacci_delay = self.config.base_delay_ms * fib_multiplier
        
        # Apply maximum delay limit
        capped_delay = min(fibonacci_delay, self.config.max_delay_ms)
        
        # Apply jitter
        jittered_delay = self._apply_jitter(capped_delay, attempt)
        
        # Calculate elapsed time
        elapsed_time = 0.0
        if self._start_time:
            elapsed_time = (datetime.utcnow() - self._start_time).total_seconds() * 1000
        
        # Check if should retry
        should_retry = await self.should_retry(attempt, last_error, elapsed_time)
        
        result = RetryResult(
            should_retry=should_retry,
            delay_ms=jittered_delay,
            attempt_number=attempt,
            total_elapsed_ms=elapsed_time,
            reason=f"fibonacci_backoff_attempt_{attempt}_multiplier_{fib_multiplier}"
        )
        
        # Record metrics
        self._record_attempt(attempt, should_retry, jittered_delay)
        
        return result


class RetryPolicyFactory:
    """Factory for creating retry policies."""
    
    @staticmethod
    def create_policy(config: RetryConfig) -> RetryPolicy:
        """Create retry policy based on configuration."""
        
        if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return ExponentialBackoffPolicy(config)
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            return LinearBackoffPolicy(config)
        elif config.strategy == RetryStrategy.FIXED_DELAY:
            return FixedDelayPolicy(config)
        elif config.strategy == RetryStrategy.ADAPTIVE:
            return AdaptiveBackoffPolicy(config)
        elif config.strategy == RetryStrategy.FIBONACCI:
            return FibonacciBackoffPolicy(config)
        else:
            raise ValueError(f"Unknown retry strategy: {config.strategy}")


class RetryExecutor:
    """Executor for running functions with retry policies."""
    
    def __init__(self, retry_policy: RetryPolicy):
        self.retry_policy = retry_policy
        self.execution_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_attempts': 0,
            'average_attempts_per_execution': 0.0
        }
    
    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute function with retry policy.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: Last exception if all retries failed
        """
        self.execution_metrics['total_executions'] += 1
        
        last_exception = None
        attempt = 0
        
        while True:
            self.execution_metrics['total_attempts'] += 1
            
            try:
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Success - record and return
                self.execution_metrics['successful_executions'] += 1
                self._update_average_attempts()
                
                logger.debug(
                    "✅ Function executed successfully with retry",
                    attempt=attempt,
                    function_name=getattr(func, '__name__', 'unknown')
                )
                
                return result
                
            except Exception as error:
                last_exception = error
                
                # Calculate retry delay
                retry_result = await self.retry_policy.calculate_delay(attempt, error)
                
                if not retry_result.should_retry:
                    # No more retries - record failure and raise
                    self.execution_metrics['failed_executions'] += 1
                    self._update_average_attempts()
                    
                    logger.error(
                        "❌ Function execution failed after all retries",
                        attempt=attempt,
                        function_name=getattr(func, '__name__', 'unknown'),
                        final_error=str(error),
                        total_delay_ms=retry_result.total_elapsed_ms
                    )
                    
                    raise error
                
                # Wait before retry
                if retry_result.delay_ms > 0:
                    await asyncio.sleep(retry_result.delay_ms / 1000.0)
                
                attempt += 1
                
                logger.warning(
                    "🔄 Function execution failed, retrying",
                    attempt=attempt,
                    function_name=getattr(func, '__name__', 'unknown'),
                    error=str(error),
                    delay_ms=retry_result.delay_ms,
                    next_delay_ms=retry_result.next_delay_ms
                )
    
    def _update_average_attempts(self) -> None:
        """Update average attempts per execution metric."""
        if self.execution_metrics['total_executions'] > 0:
            self.execution_metrics['average_attempts_per_execution'] = (
                self.execution_metrics['total_attempts'] / self.execution_metrics['total_executions']
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics combined with retry policy metrics."""
        return {
            "execution_metrics": self.execution_metrics,
            "retry_policy_metrics": self.retry_policy.get_metrics()
        }


# Convenience functions for common retry patterns

async def retry_with_exponential_backoff(
    func: Callable,
    max_attempts: int = 3,
    base_delay_ms: int = 100,
    max_delay_ms: int = 30000,
    backoff_multiplier: float = 2.0,
    jitter_type: JitterType = JitterType.EQUAL,
    *args,
    **kwargs
) -> Any:
    """
    Execute function with exponential backoff retry.
    
    Args:
        func: Function to execute
        max_attempts: Maximum retry attempts
        base_delay_ms: Base delay in milliseconds
        max_delay_ms: Maximum delay in milliseconds  
        backoff_multiplier: Exponential backoff multiplier
        jitter_type: Type of jitter to apply
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay_ms=base_delay_ms,
        max_delay_ms=max_delay_ms,
        strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
        backoff_multiplier=backoff_multiplier,
        jitter_type=jitter_type
    )
    
    policy = RetryPolicyFactory.create_policy(config)
    executor = RetryExecutor(policy)
    
    return await executor.execute(func, *args, **kwargs)


async def retry_with_fixed_delay(
    func: Callable,
    max_attempts: int = 3,
    delay_ms: int = 1000,
    jitter_type: JitterType = JitterType.EQUAL,
    *args,
    **kwargs
) -> Any:
    """
    Execute function with fixed delay retry.
    
    Args:
        func: Function to execute
        max_attempts: Maximum retry attempts
        delay_ms: Fixed delay in milliseconds
        jitter_type: Type of jitter to apply
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay_ms=delay_ms,
        strategy=RetryStrategy.FIXED_DELAY,
        jitter_type=jitter_type
    )
    
    policy = RetryPolicyFactory.create_policy(config)
    executor = RetryExecutor(policy)
    
    return await executor.execute(func, *args, **kwargs)


async def retry_with_adaptive_backoff(
    func: Callable,
    max_attempts: int = 5,
    base_delay_ms: int = 100,
    max_delay_ms: int = 30000,
    *args,
    **kwargs
) -> Any:
    """
    Execute function with adaptive backoff retry.
    
    Args:
        func: Function to execute
        max_attempts: Maximum retry attempts
        base_delay_ms: Base delay in milliseconds
        max_delay_ms: Maximum delay in milliseconds
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Function result
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay_ms=base_delay_ms,
        max_delay_ms=max_delay_ms,
        strategy=RetryStrategy.ADAPTIVE,
        jitter_type=JitterType.EQUAL
    )
    
    policy = RetryPolicyFactory.create_policy(config)
    executor = RetryExecutor(policy)
    
    return await executor.execute(func, *args, **kwargs)