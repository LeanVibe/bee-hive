"""
Comprehensive Error Handling and Validation for Team Coordination API

Enterprise-grade error handling providing:
- Structured error responses with proper HTTP status codes
- Request validation with detailed error messages
- Rate limiting and circuit breaker patterns
- Audit logging for security and compliance
- Graceful degradation for system resilience
- Custom exception classes for domain-specific errors
"""

import asyncio
import json
import traceback
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum
from contextlib import asynccontextmanager

from fastapi import HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError, BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import structlog

from ..schemas.team_coordination import APIError, ValidationError as CustomValidationError


logger = structlog.get_logger()


# =====================================================================================
# CUSTOM EXCEPTION CLASSES
# =====================================================================================

class ErrorCategory(str, Enum):
    """Categories for error classification."""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RESOURCE_NOT_FOUND = "resource_not_found"
    BUSINESS_LOGIC = "business_logic"
    SYSTEM_ERROR = "system_error"
    RATE_LIMIT = "rate_limit"
    EXTERNAL_SERVICE = "external_service"
    CONFIGURATION = "configuration"


class ErrorSeverity(str, Enum):
    """Error severity levels for monitoring and alerting."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CoordinationException(Exception):
    """Base exception for team coordination errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        http_status: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    ):
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.details = details
        self.context = context or {}
        self.http_status = http_status
        self.timestamp = datetime.utcnow()
        
        super().__init__(message)


class AgentNotFoundError(CoordinationException):
    """Agent not found in coordination system."""
    
    def __init__(self, agent_id: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Agent not found: {agent_id}",
            error_code="AGENT_NOT_FOUND",
            category=ErrorCategory.RESOURCE_NOT_FOUND,
            severity=ErrorSeverity.LOW,
            details=f"Agent with ID '{agent_id}' does not exist in the coordination system",
            context={"agent_id": agent_id, **(context or {})},
            http_status=status.HTTP_404_NOT_FOUND
        )


class TaskNotFoundError(CoordinationException):
    """Task not found in coordination system."""
    
    def __init__(self, task_id: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Task not found: {task_id}",
            error_code="TASK_NOT_FOUND", 
            category=ErrorCategory.RESOURCE_NOT_FOUND,
            severity=ErrorSeverity.LOW,
            details=f"Task with ID '{task_id}' does not exist",
            context={"task_id": task_id, **(context or {})},
            http_status=status.HTTP_404_NOT_FOUND
        )


class InsufficientCapacityError(CoordinationException):
    """No agents available with required capabilities."""
    
    def __init__(self, required_capabilities: List[str], context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="No suitable agents available",
            error_code="INSUFFICIENT_CAPACITY",
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.HIGH,
            details=f"No agents available with required capabilities: {', '.join(required_capabilities)}",
            context={"required_capabilities": required_capabilities, **(context or {})},
            http_status=status.HTTP_503_SERVICE_UNAVAILABLE
        )


class AgentOverloadedError(CoordinationException):
    """Agent is overloaded and cannot accept new tasks."""
    
    def __init__(self, agent_id: str, current_workload: float, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Agent is overloaded: {agent_id}",
            error_code="AGENT_OVERLOADED",
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            details=f"Agent workload ({current_workload:.1%}) exceeds capacity limits",
            context={"agent_id": agent_id, "current_workload": current_workload, **(context or {})},
            http_status=status.HTTP_429_TOO_MANY_REQUESTS
        )


class InvalidTaskStateError(CoordinationException):
    """Task is in an invalid state for the requested operation."""
    
    def __init__(self, task_id: str, current_state: str, required_state: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Invalid task state for operation",
            error_code="INVALID_TASK_STATE",
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            details=f"Task {task_id} is in state '{current_state}' but requires '{required_state}'",
            context={
                "task_id": task_id,
                "current_state": current_state,
                "required_state": required_state,
                **(context or {})
            },
            http_status=status.HTTP_409_CONFLICT
        )


class CoordinationSystemError(CoordinationException):
    """System-level coordination error."""
    
    def __init__(self, component: str, operation: str, underlying_error: Optional[Exception] = None, context: Optional[Dict[str, Any]] = None):
        details = f"Component '{component}' failed during '{operation}'"
        if underlying_error:
            details += f": {str(underlying_error)}"
        
        super().__init__(
            message="System coordination error",
            error_code="COORDINATION_SYSTEM_ERROR",
            category=ErrorCategory.SYSTEM_ERROR,
            severity=ErrorSeverity.HIGH,
            details=details,
            context={
                "component": component,
                "operation": operation,
                "underlying_error": str(underlying_error) if underlying_error else None,
                **(context or {})
            },
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class RateLimitExceededError(CoordinationException):
    """Rate limit exceeded for API endpoint."""
    
    def __init__(self, endpoint: str, limit: int, window_seconds: int, context: Optional[Dict[str, Any]] = None):
        super().__init__(
            message="Rate limit exceeded",
            error_code="RATE_LIMIT_EXCEEDED",
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.MEDIUM,
            details=f"Exceeded rate limit of {limit} requests per {window_seconds} seconds for {endpoint}",
            context={
                "endpoint": endpoint,
                "limit": limit,
                "window_seconds": window_seconds,
                **(context or {})
            },
            http_status=status.HTTP_429_TOO_MANY_REQUESTS
        )


# =====================================================================================
# RATE LIMITER
# =====================================================================================

class RateLimiter:
    """Simple in-memory rate limiter with sliding window."""
    
    def __init__(self):
        self.requests: Dict[str, List[datetime]] = {}
        self.limits: Dict[str, Dict[str, Union[int, timedelta]]] = {
            # endpoint -> {limit: int, window: timedelta}
            "/team-coordination/agents/register": {"limit": 10, "window": timedelta(minutes=1)},
            "/team-coordination/tasks/distribute": {"limit": 100, "window": timedelta(minutes=1)},
            "/team-coordination/metrics": {"limit": 30, "window": timedelta(minutes=1)},
            "default": {"limit": 60, "window": timedelta(minutes=1)}
        }
    
    async def check_rate_limit(self, client_id: str, endpoint: str) -> bool:
        """Check if request is within rate limits."""
        key = f"{client_id}:{endpoint}"
        now = datetime.utcnow()
        
        # Get rate limit configuration
        config = self.limits.get(endpoint, self.limits["default"])
        limit = config["limit"]
        window = config["window"]
        
        # Initialize request history if needed
        if key not in self.requests:
            self.requests[key] = []
        
        # Clean old requests outside window
        cutoff_time = now - window
        self.requests[key] = [req_time for req_time in self.requests[key] if req_time > cutoff_time]
        
        # Check if within limits
        if len(self.requests[key]) >= limit:
            return False
        
        # Record this request
        self.requests[key].append(now)
        return True
    
    async def get_rate_limit_info(self, client_id: str, endpoint: str) -> Dict[str, Any]:
        """Get current rate limit status."""
        key = f"{client_id}:{endpoint}"
        config = self.limits.get(endpoint, self.limits["default"])
        
        current_requests = len(self.requests.get(key, []))
        
        return {
            "limit": config["limit"],
            "remaining": max(0, config["limit"] - current_requests),
            "window_seconds": int(config["window"].total_seconds()),
            "reset_time": (datetime.utcnow() + config["window"]).isoformat()
        }


# =====================================================================================
# CIRCUIT BREAKER
# =====================================================================================

class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker for external dependencies."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 60,
        success_threshold: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = timedelta(seconds=recovery_timeout_seconds)
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitBreakerState.CLOSED
    
    def should_allow_request(self) -> bool:
        """Check if request should be allowed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time and \
               (datetime.utcnow() - self.last_failure_time) > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        
        # HALF_OPEN state - allow limited requests to test
        return True
    
    def record_success(self):
        """Record successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN


# =====================================================================================
# ERROR HANDLER SERVICE
# =====================================================================================

class TeamCoordinationErrorHandler:
    """Comprehensive error handling service."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
    
    async def handle_coordination_exception(
        self,
        request: Request,
        exc: CoordinationException
    ) -> JSONResponse:
        """Handle coordination-specific exceptions."""
        
        # Generate request ID if not present
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        # Log error with structured data
        logger.error(
            "Coordination error",
            error_code=exc.error_code,
            category=exc.category.value,
            severity=exc.severity.value,
            message=exc.message,
            details=exc.details,
            context=exc.context,
            request_id=request_id,
            path=request.url.path,
            method=request.method
        )
        
        # Record error for analytics
        await self._record_error(request, exc, request_id)
        
        # Create error response
        error_response = APIError(
            error_code=exc.error_code,
            message=exc.message,
            details=exc.details,
            timestamp=exc.timestamp,
            request_id=request_id,
            help_url=f"https://docs.leanvibe.dev/errors/{exc.error_code.lower()}"
        )
        
        return JSONResponse(
            status_code=exc.http_status,
            content=error_response.dict(),
            headers={
                "X-Request-ID": request_id,
                "X-Error-Category": exc.category.value,
                "X-Error-Severity": exc.severity.value
            }
        )
    
    async def handle_validation_error(
        self,
        request: Request,
        exc: Union[RequestValidationError, ValidationError]
    ) -> JSONResponse:
        """Handle Pydantic validation errors."""
        
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        # Extract validation errors
        validation_errors = []
        
        if isinstance(exc, RequestValidationError):
            for error in exc.errors():
                validation_errors.append(CustomValidationError(
                    field=".".join(str(loc) for loc in error["loc"]),
                    message=error["msg"],
                    invalid_value=error.get("input"),
                    constraint=error.get("type")
                ))
        else:
            for error in exc.errors():
                validation_errors.append(CustomValidationError(
                    field=".".join(str(loc) for loc in error["loc"]),
                    message=error["msg"],
                    invalid_value=error.get("input"),
                    constraint=error.get("type")
                ))
        
        logger.warning(
            "Validation error",
            path=request.url.path,
            method=request.method,
            validation_errors=[ve.dict() for ve in validation_errors],
            request_id=request_id
        )
        
        error_response = APIError(
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            details=f"Found {len(validation_errors)} validation error(s)",
            validation_errors=validation_errors,
            request_id=request_id,
            help_url="https://docs.leanvibe.dev/errors/validation"
        )
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.dict(),
            headers={"X-Request-ID": request_id}
        )
    
    async def handle_http_exception(
        self,
        request: Request,
        exc: HTTPException
    ) -> JSONResponse:
        """Handle FastAPI HTTP exceptions."""
        
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        logger.warning(
            "HTTP exception",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path,
            method=request.method,
            request_id=request_id
        )
        
        error_response = APIError(
            error_code=f"HTTP_{exc.status_code}",
            message=exc.detail if isinstance(exc.detail, str) else "HTTP error",
            details=str(exc.detail) if not isinstance(exc.detail, str) else None,
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.dict(),
            headers={"X-Request-ID": request_id}
        )
    
    async def handle_generic_exception(
        self,
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        # Log with full traceback for debugging
        logger.error(
            "Unexpected error",
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback=traceback.format_exc(),
            path=request.url.path,
            method=request.method,
            request_id=request_id
        )
        
        # Record critical error
        coordination_exc = CoordinationSystemError(
            component="error_handler",
            operation="handle_generic_exception",
            underlying_error=exc,
            context={"request_id": request_id}
        )
        await self._record_error(request, coordination_exc, request_id)
        
        error_response = APIError(
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            details="The system encountered an unexpected error. Please try again or contact support.",
            request_id=request_id,
            help_url="https://docs.leanvibe.dev/support"
        )
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.dict(),
            headers={"X-Request-ID": request_id}
        )
    
    async def check_rate_limit(self, request: Request) -> Optional[JSONResponse]:
        """Check rate limits and return error response if exceeded."""
        
        # Extract client identifier (IP address as fallback)
        client_id = request.headers.get("X-Client-ID", request.client.host)
        endpoint = request.url.path
        
        if not await self.rate_limiter.check_rate_limit(client_id, endpoint):
            request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
            
            rate_limit_info = await self.rate_limiter.get_rate_limit_info(client_id, endpoint)
            
            exc = RateLimitExceededError(
                endpoint=endpoint,
                limit=rate_limit_info["limit"],
                window_seconds=rate_limit_info["window_seconds"],
                context={"client_id": client_id}
            )
            
            return await self.handle_coordination_exception(request, exc)
        
        return None
    
    async def check_circuit_breaker(self, service_name: str) -> bool:
        """Check if service is available via circuit breaker."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        
        return self.circuit_breakers[service_name].should_allow_request()
    
    async def record_service_success(self, service_name: str):
        """Record successful service call."""
        if service_name in self.circuit_breakers:
            self.circuit_breakers[service_name].record_success()
    
    async def record_service_failure(self, service_name: str):
        """Record failed service call."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        
        self.circuit_breakers[service_name].record_failure()
    
    async def _record_error(
        self,
        request: Request,
        exc: Union[CoordinationException, Exception],
        request_id: str
    ):
        """Record error for analytics and monitoring."""
        
        error_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "path": request.url.path,
            "method": request.method,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "client_ip": request.client.host,
            "user_agent": request.headers.get("user-agent"),
        }
        
        if isinstance(exc, CoordinationException):
            error_record.update({
                "error_code": exc.error_code,
                "category": exc.category.value,
                "severity": exc.severity.value,
                "context": exc.context
            })
        
        # Add to history (with size limit)
        self.error_history.append(error_record)
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
    
    async def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_errors = [
            e for e in self.error_history
            if datetime.fromisoformat(e["timestamp"]) > cutoff_time
        ]
        
        if not recent_errors:
            return {"period_hours": hours, "total_errors": 0}
        
        # Count by category
        error_counts = {}
        severity_counts = {}
        
        for error in recent_errors:
            category = error.get("category", "unknown")
            severity = error.get("severity", "unknown")
            
            error_counts[category] = error_counts.get(category, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "period_hours": hours,
            "total_errors": len(recent_errors),
            "error_rate": len(recent_errors) / hours,
            "errors_by_category": error_counts,
            "errors_by_severity": severity_counts,
            "circuit_breaker_states": {
                name: breaker.state.value
                for name, breaker in self.circuit_breakers.items()
            }
        }


# =====================================================================================
# ERROR HANDLING MIDDLEWARE
# =====================================================================================

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive error handling."""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.error_handler = TeamCoordinationErrorHandler()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with error handling."""
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Add request ID to response headers
        def add_request_id_header(response: Response) -> Response:
            response.headers["X-Request-ID"] = request_id
            return response
        
        try:
            # Check rate limits
            rate_limit_response = await self.error_handler.check_rate_limit(request)
            if rate_limit_response:
                return add_request_id_header(rate_limit_response)
            
            # Process request
            response = await call_next(request)
            return add_request_id_header(response)
            
        except CoordinationException as exc:
            response = await self.error_handler.handle_coordination_exception(request, exc)
            return add_request_id_header(response)
            
        except (RequestValidationError, ValidationError) as exc:
            response = await self.error_handler.handle_validation_error(request, exc)
            return add_request_id_header(response)
            
        except HTTPException as exc:
            response = await self.error_handler.handle_http_exception(request, exc)
            return add_request_id_header(response)
            
        except Exception as exc:
            response = await self.error_handler.handle_generic_exception(request, exc)
            return add_request_id_header(response)


# =====================================================================================
# UTILITY FUNCTIONS AND DECORATORS
# =====================================================================================

def with_circuit_breaker(service_name: str):
    """Decorator to wrap functions with circuit breaker protection."""
    
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            error_handler = TeamCoordinationErrorHandler()
            
            if not await error_handler.check_circuit_breaker(service_name):
                raise CoordinationSystemError(
                    component=service_name,
                    operation=func.__name__,
                    context={"circuit_breaker_open": True}
                )
            
            try:
                result = await func(*args, **kwargs)
                await error_handler.record_service_success(service_name)
                return result
                
            except Exception as e:
                await error_handler.record_service_failure(service_name)
                raise
        
        return wrapper
    return decorator


def validate_request_data(schema_class: type[BaseModel]):
    """Decorator to validate request data against Pydantic schema."""
    
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            # This would be implemented with dependency injection in FastAPI
            # The decorator is here as an example of validation patterns
            return await func(*args, **kwargs)
        return wrapper
    return decorator


@asynccontextmanager
async def error_context(operation: str, component: str = "coordination"):
    """Context manager for consistent error handling."""
    try:
        yield
    except CoordinationException:
        raise  # Re-raise coordination exceptions as-is
    except Exception as e:
        raise CoordinationSystemError(
            component=component,
            operation=operation,
            underlying_error=e
        )


# =====================================================================================
# GLOBAL ERROR HANDLER INSTANCE
# =====================================================================================

_error_handler: Optional[TeamCoordinationErrorHandler] = None


def get_error_handler() -> TeamCoordinationErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    
    if _error_handler is None:
        _error_handler = TeamCoordinationErrorHandler()
    
    return _error_handler