"""
Comprehensive Error Handling Middleware for LeanVibe Agent Hive 2.0 - VS 3.3

Production-ready error handling middleware with <5ms overhead performance targets:
- Circuit breaker pattern integration for cascade failure prevention
- Exponential backoff retry logic with jitter for transient failures
- Graceful degradation mechanisms for service dependencies
- Integration with observability hooks for error tracking
- Performance monitoring with >99.95% availability targets
"""

import asyncio
import time
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
import structlog

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .observability_hooks import get_observability_hooks

logger = structlog.get_logger()


class ErrorSeverity(Enum):
    """Error severity levels for monitoring and alerting."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification and handling."""
    NETWORK = "network"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERNAL = "internal"
    DEPENDENCY = "dependency"


@dataclass
class ErrorContext:
    """Context information for error handling decisions."""
    request_id: str
    path: str
    method: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    retry_count: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)
    service_name: Optional[str] = None
    dependency_chain: List[str] = field(default_factory=list)


@dataclass
class ErrorHandlingConfig:
    """Configuration for error handling middleware."""
    enabled: bool = True
    max_retries: int = 3
    base_delay_ms: int = 100
    max_delay_ms: int = 5000
    jitter_enabled: bool = True
    circuit_breaker_enabled: bool = True
    graceful_degradation_enabled: bool = True
    observability_enabled: bool = True
    
    # Performance targets
    max_processing_time_ms: float = 5.0
    availability_target: float = 0.9995  # 99.95%
    
    # Circuit breaker thresholds
    circuit_breaker_failure_threshold: int = 10
    circuit_breaker_success_threshold: int = 5
    circuit_breaker_timeout_seconds: int = 60
    
    # Retry configurations
    retryable_status_codes: List[int] = field(default_factory=lambda: [502, 503, 504, 429])
    retryable_exceptions: List[Type[Exception]] = field(default_factory=lambda: [
        ConnectionError, TimeoutError, asyncio.TimeoutError
    ])
    
    # Degradation settings
    degradation_timeout_ms: int = 30000  # 30 seconds


class ErrorAnalyzer:
    """Analyzes errors to determine handling strategy."""
    
    def __init__(self):
        self.error_patterns = {}
        self.error_frequency = {}
        
    def analyze_error(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """
        Analyze an error to determine severity, category, and handling strategy.
        
        Args:
            error: The exception that occurred
            context: Request context information
            
        Returns:
            Analysis results with handling recommendations
        """
        analysis = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": self._determine_severity(error, context),
            "category": self._determine_category(error, context),
            "is_retryable": self._is_retryable(error, context),
            "should_circuit_break": self._should_circuit_break(error, context),
            "degradation_level": self._determine_degradation_level(error, context),
            "recovery_strategy": self._determine_recovery_strategy(error, context)
        }
        
        return analysis
    
    def _determine_severity(self, error: Exception, context: ErrorContext) -> ErrorSeverity:
        """Determine error severity based on error type and context."""
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, HTTPException):
            if error.status_code >= 500:
                return ErrorSeverity.HIGH
            elif error.status_code == 429:  # Rate limiting
                return ErrorSeverity.MEDIUM
            elif error.status_code >= 400:
                return ErrorSeverity.LOW
        elif isinstance(error, asyncio.TimeoutError):
            return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.MEDIUM
    
    def _determine_category(self, error: Exception, context: ErrorContext) -> ErrorCategory:
        """Determine error category for appropriate handling."""
        if isinstance(error, ConnectionError):
            return ErrorCategory.NETWORK
        elif isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return ErrorCategory.TIMEOUT
        elif isinstance(error, HTTPException):
            if error.status_code == 401:
                return ErrorCategory.AUTHENTICATION
            elif error.status_code == 403:
                return ErrorCategory.AUTHORIZATION
            elif error.status_code == 422:
                return ErrorCategory.VALIDATION
            elif error.status_code == 429:
                return ErrorCategory.RATE_LIMIT
            elif error.status_code >= 500:
                return ErrorCategory.SERVICE_UNAVAILABLE
        
        return ErrorCategory.INTERNAL
    
    def _is_retryable(self, error: Exception, context: ErrorContext) -> bool:
        """Determine if error is retryable."""
        if context.retry_count >= 3:  # Max retries reached
            return False
        
        if isinstance(error, (ConnectionError, TimeoutError, asyncio.TimeoutError)):
            return True
        
        if isinstance(error, HTTPException):
            return error.status_code in [502, 503, 504, 429]
        
        return False
    
    def _should_circuit_break(self, error: Exception, context: ErrorContext) -> bool:
        """Determine if error should trigger circuit breaker."""
        if isinstance(error, (ConnectionError, TimeoutError)):
            return True
        
        if isinstance(error, HTTPException) and error.status_code >= 500:
            return True
        
        return False
    
    def _determine_degradation_level(self, error: Exception, context: ErrorContext):
        """Determine appropriate degradation level."""
        from .graceful_degradation import DegradationLevel
        
        if isinstance(error, ConnectionError):
            return DegradationLevel.PARTIAL
        elif isinstance(error, TimeoutError):
            return DegradationLevel.MINIMAL
        elif isinstance(error, HTTPException) and error.status_code >= 500:
            return DegradationLevel.PARTIAL
        
        return DegradationLevel.NONE
    
    def _determine_recovery_strategy(self, error: Exception, context: ErrorContext) -> str:
        """Determine recovery strategy based on error analysis."""
        if self._is_retryable(error, context):
            return "retry_with_backoff"
        elif self._should_circuit_break(error, context):
            return "circuit_break"
        else:
            return "fail_fast"


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive error handling middleware for FastAPI applications.
    
    Features:
    - Circuit breaker pattern integration
    - Intelligent retry logic with exponential backoff
    - Graceful degradation mechanisms
    - Performance monitoring with <5ms overhead
    - Integration with observability system
    """
    
    def __init__(
        self,
        app: ASGIApp,
        config: Optional[ErrorHandlingConfig] = None,
        circuit_breaker: Optional['CircuitBreaker'] = None,
        degradation_manager: Optional['GracefulDegradationManager'] = None
    ):
        """Initialize error handling middleware."""
        super().__init__(app)
        self.config = config or ErrorHandlingConfig()
        
        # Initialize components
        self.error_analyzer = ErrorAnalyzer()
        
        # Initialize circuit breaker
        if circuit_breaker:
            self.circuit_breaker = circuit_breaker
        else:
            from .circuit_breaker import CircuitBreaker
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=self.config.circuit_breaker_failure_threshold,
                success_threshold=self.config.circuit_breaker_success_threshold,
                timeout_seconds=self.config.circuit_breaker_timeout_seconds
            )
        
        # Initialize degradation manager
        if degradation_manager:
            self.degradation_manager = degradation_manager
        else:
            from .graceful_degradation import GracefulDegradationManager
            self.degradation_manager = GracefulDegradationManager()
        
        # Initialize retry policy
        from .retry_policies import ExponentialBackoffPolicy, RetryConfig
        retry_config = RetryConfig(
            max_attempts=self.config.max_retries,
            base_delay_ms=self.config.base_delay_ms,
            max_delay_ms=self.config.max_delay_ms,
            jitter_enabled=self.config.jitter_enabled
        )
        self.retry_policy = ExponentialBackoffPolicy(retry_config)
        
        # Performance tracking
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        self.availability_violations = 0
        
        # Observability integration
        self.observability_hooks = get_observability_hooks()
        
        logger.info(
            "🛡️ Error handling middleware initialized",
            circuit_breaker_enabled=self.config.circuit_breaker_enabled,
            graceful_degradation_enabled=self.config.graceful_degradation_enabled,
            max_retries=self.config.max_retries,
            availability_target=self.config.availability_target
        )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Main middleware dispatch method with comprehensive error handling.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware/route handler
            
        Returns:
            Response with error handling applied
        """
        if not self.config.enabled:
            return await call_next(request)
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Create error context
        context = ErrorContext(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
            user_id=getattr(request.state, 'user_id', None),
            agent_id=getattr(request.state, 'agent_id', None),
            session_id=getattr(request.state, 'session_id', None)
        )
        
        # Store context in request state
        request.state.error_context = context
        request.state.request_id = request_id
        
        try:
            # Check circuit breaker state
            if self.config.circuit_breaker_enabled:
                from .circuit_breaker import CircuitBreakerState
                circuit_state = await self.circuit_breaker.get_state()
                if circuit_state == CircuitBreakerState.OPEN:
                    return await self._handle_circuit_breaker_open(context)
            
            # Execute request with retry logic
            response = await self._execute_with_retry(request, call_next, context)
            
            # Record success metrics
            processing_time = (time.time() - start_time) * 1000
            await self._record_success_metrics(context, processing_time)
            
            # Update circuit breaker on success
            if self.config.circuit_breaker_enabled:
                await self.circuit_breaker.record_success()
            
            return response
            
        except Exception as error:
            # Handle error with comprehensive analysis
            processing_time = (time.time() - start_time) * 1000
            return await self._handle_error(error, context, processing_time)
        
        finally:
            # Update request metrics
            self.request_count += 1
            self.total_processing_time += (time.time() - start_time) * 1000
    
    async def _execute_with_retry(
        self,
        request: Request,
        call_next: Callable,
        context: ErrorContext
    ) -> Response:
        """Execute request with intelligent retry logic."""
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            context.retry_count = attempt
            
            try:
                # Add retry attempt to request state
                request.state.retry_attempt = attempt
                
                # Execute request
                response = await call_next(request)
                
                # Check if response indicates a retryable error
                if hasattr(response, 'status_code') and response.status_code in self.config.retryable_status_codes:
                    if attempt < self.config.max_retries:
                        await self._wait_for_retry(attempt)
                        continue
                
                return response
                
            except Exception as error:
                last_error = error
                
                # Analyze error to determine if retryable
                analysis = self.error_analyzer.analyze_error(error, context)
                
                if not analysis["is_retryable"] or attempt >= self.config.max_retries:
                    break
                
                # Wait before retry with exponential backoff
                await self._wait_for_retry(attempt)
                
                logger.warning(
                    "🔄 Retrying request after error",
                    request_id=context.request_id,
                    attempt=attempt + 1,
                    max_retries=self.config.max_retries,
                    error=str(error),
                    error_type=type(error).__name__
                )
        
        # All retries exhausted, raise the last error
        if last_error:
            raise last_error
        
        # Should not reach here, but safety fallback
        raise RuntimeError("Request execution failed without specific error")
    
    async def _wait_for_retry(self, attempt: int) -> None:
        """Wait for retry with exponential backoff and jitter."""
        retry_result = await self.retry_policy.calculate_delay(attempt)
        
        if retry_result.should_retry:
            await asyncio.sleep(retry_result.delay_ms / 1000.0)
    
    async def _handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        processing_time: float
    ) -> Response:
        """Handle error with comprehensive analysis and response generation."""
        
        # Analyze error
        analysis = self.error_analyzer.analyze_error(error, context)
        
        # Update circuit breaker
        if self.config.circuit_breaker_enabled and analysis["should_circuit_break"]:
            await self.circuit_breaker.record_failure()
        
        # Record error metrics
        await self._record_error_metrics(error, context, analysis, processing_time)
        
        # Apply graceful degradation if needed
        degraded_response = None
        if self.config.graceful_degradation_enabled:
            from .graceful_degradation import DegradationLevel
            if analysis["degradation_level"] != DegradationLevel.NONE:
                degraded_response = await self.degradation_manager.apply_degradation(
                    context.path,
                    analysis["degradation_level"],
                    error_context=analysis
                )
        
        # Emit error event to observability system
        if self.config.observability_enabled and self.observability_hooks:
            try:
                await self.observability_hooks.failure_detected(
                    failure_type=analysis["error_type"],
                    failure_description=analysis["error_message"],
                    affected_component="error_handling_middleware",
                    severity=analysis["severity"].value,
                    error_details={
                        "request_id": context.request_id,
                        "path": context.path,
                        "method": context.method,
                        "retry_count": context.retry_count,
                        "processing_time_ms": processing_time,
                        "category": analysis["category"].value,
                        "recovery_strategy": analysis["recovery_strategy"]
                    },
                    agent_id=uuid.UUID(context.agent_id) if context.agent_id else None,
                    session_id=uuid.UUID(context.session_id) if context.session_id else None,
                    detection_method="middleware_analysis",
                    impact_assessment={
                        "availability_impact": processing_time > self.config.max_processing_time_ms,
                        "degradation_applied": degraded_response is not None,
                        "circuit_breaker_triggered": analysis["should_circuit_break"]
                    }
                )
            except Exception as obs_error:
                logger.warning("Failed to emit error event to observability", error=str(obs_error))
        
        # Return degraded response if available
        if degraded_response:
            return degraded_response
        
        # Generate appropriate error response
        return await self._generate_error_response(error, context, analysis)
    
    async def _handle_circuit_breaker_open(self, context: ErrorContext) -> Response:
        """Handle requests when circuit breaker is open."""
        
        # Try graceful degradation first
        if self.config.graceful_degradation_enabled:
            from .graceful_degradation import DegradationLevel
            degraded_response = await self.degradation_manager.apply_degradation(
                context.path,
                DegradationLevel.FULL,
                error_context={"circuit_breaker": "open"}
            )
            
            if degraded_response:
                return degraded_response
        
        # Return circuit breaker error response
        return JSONResponse(
            status_code=503,
            content={
                "error": "Service temporarily unavailable",
                "reason": "Circuit breaker is open",
                "request_id": context.request_id,
                "retry_after": self.config.circuit_breaker_timeout_seconds
            },
            headers={
                "Retry-After": str(self.config.circuit_breaker_timeout_seconds)
            }
        )
    
    async def _generate_error_response(
        self,
        error: Exception,
        context: ErrorContext,
        analysis: Dict[str, Any]
    ) -> Response:
        """Generate appropriate error response based on error analysis."""
        
        # Determine status code
        if isinstance(error, HTTPException):
            status_code = error.status_code
            detail = error.detail
        elif analysis["category"] == ErrorCategory.TIMEOUT:
            status_code = 408
            detail = "Request timeout"
        elif analysis["category"] == ErrorCategory.RATE_LIMIT:
            status_code = 429
            detail = "Rate limit exceeded"
        elif analysis["category"] == ErrorCategory.SERVICE_UNAVAILABLE:
            status_code = 503
            detail = "Service temporarily unavailable"
        else:
            status_code = 500
            detail = "Internal server error"
        
        # Create error response
        error_response = {
            "error": detail,
            "request_id": context.request_id,
            "error_type": analysis["error_type"],
            "category": analysis["category"].value,
            "severity": analysis["severity"].value
        }
        
        # Add retry information if applicable
        if analysis["is_retryable"] and context.retry_count < self.config.max_retries:
            error_response["retryable"] = True
            error_response["retry_after"] = self.config.base_delay_ms / 1000.0
        
        # Add debug information in development
        if logger.isEnabledFor(logging.DEBUG):
            error_response["debug"] = {
                "retry_count": context.retry_count,
                "recovery_strategy": analysis["recovery_strategy"],
                "error_message": str(error)
            }
        
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )
    
    async def _record_success_metrics(self, context: ErrorContext, processing_time: float) -> None:
        """Record success metrics for monitoring."""
        
        # Check availability target
        if processing_time > self.config.max_processing_time_ms:
            self.availability_violations += 1
        
        # Log success with performance metrics
        logger.debug(
            "✅ Request completed successfully",
            request_id=context.request_id,
            path=context.path,
            method=context.method,
            processing_time_ms=round(processing_time, 2),
            retry_count=context.retry_count
        )
    
    async def _record_error_metrics(
        self,
        error: Exception,
        context: ErrorContext,
        analysis: Dict[str, Any],
        processing_time: float
    ) -> None:
        """Record error metrics for monitoring and alerting."""
        
        self.error_count += 1
        
        # Check availability target
        if processing_time > self.config.max_processing_time_ms:
            self.availability_violations += 1
        
        # Log error with comprehensive details
        logger.error(
            "❌ Request failed with error",
            request_id=context.request_id,
            path=context.path,
            method=context.method,
            error_type=analysis["error_type"],
            error_message=analysis["error_message"],
            category=analysis["category"].value,
            severity=analysis["severity"].value,
            retry_count=context.retry_count,
            processing_time_ms=round(processing_time, 2),
            recovery_strategy=analysis["recovery_strategy"],
            is_retryable=analysis["is_retryable"],
            should_circuit_break=analysis["should_circuit_break"]
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive error handling metrics."""
        
        availability = 1.0 - (self.error_count / max(1, self.request_count))
        avg_processing_time = self.total_processing_time / max(1, self.request_count)
        
        metrics = {
            "requests_total": self.request_count,
            "errors_total": self.error_count,
            "availability": availability,
            "availability_target_met": availability >= self.config.availability_target,
            "average_processing_time_ms": avg_processing_time,
            "performance_target_met": avg_processing_time <= self.config.max_processing_time_ms,
            "availability_violations": self.availability_violations,
            "configuration": {
                "enabled": self.config.enabled,
                "max_retries": self.config.max_retries,
                "circuit_breaker_enabled": self.config.circuit_breaker_enabled,
                "graceful_degradation_enabled": self.config.graceful_degradation_enabled,
                "availability_target": self.config.availability_target,
                "max_processing_time_ms": self.config.max_processing_time_ms
            }
        }
        
        # Add circuit breaker metrics
        if self.circuit_breaker:
            metrics["circuit_breaker"] = self.circuit_breaker.get_metrics()
        
        # Add degradation manager metrics
        if self.degradation_manager:
            metrics["graceful_degradation"] = self.degradation_manager.get_metrics()
        
        return metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of error handling components."""
        
        health_status = {
            "status": "healthy",
            "components": {},
            "metrics": self.get_metrics()
        }
        
        # Check circuit breaker health
        if self.circuit_breaker:
            cb_health = await self.circuit_breaker.health_check()
            health_status["components"]["circuit_breaker"] = cb_health
            
            if cb_health["status"] != "healthy":
                health_status["status"] = "degraded"
        
        # Check degradation manager health
        if self.degradation_manager:
            dg_health = await self.degradation_manager.health_check()
            health_status["components"]["graceful_degradation"] = dg_health
            
            if dg_health["status"] != "healthy":
                health_status["status"] = "degraded"
        
        # Check performance targets
        metrics = health_status["metrics"]
        if not metrics["availability_target_met"] or not metrics["performance_target_met"]:
            health_status["status"] = "degraded"
            health_status["performance_issues"] = {
                "availability_target_met": metrics["availability_target_met"],
                "performance_target_met": metrics["performance_target_met"]
            }
        
        return health_status
    
    def reset_metrics(self) -> None:
        """Reset metrics for testing or monitoring resets."""
        self.request_count = 0
        self.error_count = 0
        self.total_processing_time = 0.0
        self.availability_violations = 0
        
        if self.circuit_breaker:
            self.circuit_breaker.reset()
        
        logger.info("🔄 Error handling middleware metrics reset")


def create_error_handling_middleware(
    config: Optional[ErrorHandlingConfig] = None,
    circuit_breaker: Optional['CircuitBreaker'] = None,
    degradation_manager: Optional['GracefulDegradationManager'] = None
) -> Callable[[ASGIApp], ErrorHandlingMiddleware]:
    """
    Factory function to create error handling middleware.
    
    Args:
        config: Error handling configuration
        circuit_breaker: Circuit breaker instance
        degradation_manager: Graceful degradation manager
        
    Returns:
        Middleware factory function
    """
    def middleware_factory(app: ASGIApp) -> ErrorHandlingMiddleware:
        return ErrorHandlingMiddleware(
            app=app,
            config=config,
            circuit_breaker=circuit_breaker,
            degradation_manager=degradation_manager
        )
    
    return middleware_factory