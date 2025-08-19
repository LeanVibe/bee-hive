"""
Enhanced Logging and Error Handling for Production-Ready System

Provides comprehensive structured logging, error handling, and observability
improvements with correlation IDs, performance metrics, and security event logging.
"""

import json
import time
import uuid
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Callable
from functools import wraps
from contextlib import asynccontextmanager
from enum import Enum

import structlog
from structlog.types import EventDict, Processor

from .logging_service import LoggingService, get_component_logger


class LogLevel(Enum):
    """Enhanced log levels for production monitoring."""
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    FATAL = "fatal"
    AUDIT = "audit"
    SECURITY = "security"
    PERFORMANCE = "performance"


class CorrelationContext:
    """Thread-local correlation context for request tracing."""
    
    def __init__(self):
        self._context: Dict[str, Any] = {}
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for current context."""
        self._context["correlation_id"] = correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        """Get correlation ID from current context."""
        return self._context.get("correlation_id")
    
    def set_request_id(self, request_id: str) -> None:
        """Set request ID for current context."""
        self._context["request_id"] = request_id
    
    def get_request_id(self) -> Optional[str]:
        """Get request ID from current context."""
        return self._context.get("request_id")
    
    def set_user_context(self, user_id: str, user_role: str = None) -> None:
        """Set user context for security logging."""
        self._context["user_id"] = user_id
        if user_role:
            self._context["user_role"] = user_role
    
    def get_user_context(self) -> Dict[str, Any]:
        """Get user context."""
        return {
            "user_id": self._context.get("user_id"),
            "user_role": self._context.get("user_role")
        }
    
    def set_operation_context(self, operation: str, component: str = None) -> None:
        """Set operation context for tracing."""
        self._context["operation"] = operation
        if component:
            self._context["component"] = component
    
    def get_full_context(self) -> Dict[str, Any]:
        """Get complete correlation context."""
        return self._context.copy()
    
    def clear(self) -> None:
        """Clear correlation context."""
        self._context.clear()


# Global correlation context instance
correlation_context = CorrelationContext()


def correlation_processor(logger, method_name, event_dict: EventDict) -> EventDict:
    """Processor to inject correlation context into log events."""
    context = correlation_context.get_full_context()
    event_dict.update(context)
    return event_dict


def performance_processor(logger, method_name, event_dict: EventDict) -> EventDict:
    """Processor to add performance context to log events."""
    if "duration_ms" in event_dict:
        # Add performance classification
        duration_ms = event_dict["duration_ms"]
        if duration_ms > 1000:
            event_dict["performance_class"] = "slow"
        elif duration_ms > 500:
            event_dict["performance_class"] = "medium"
        else:
            event_dict["performance_class"] = "fast"
    
    # Add timestamp for performance analysis
    event_dict["log_timestamp_ms"] = int(time.time() * 1000)
    return event_dict


def security_processor(logger, method_name, event_dict: EventDict) -> EventDict:
    """Processor to enhance security-related log events."""
    # Mark security-sensitive events
    security_keywords = ["auth", "login", "permission", "access", "token", "security"]
    event_text = str(event_dict.get("event", "")).lower()
    
    if any(keyword in event_text for keyword in security_keywords):
        event_dict["security_relevant"] = True
        event_dict["log_category"] = "security"
    
    return event_dict


class EnhancedLogger:
    """
    Enhanced logger with production-ready features.
    
    Features:
    - Automatic correlation ID injection
    - Performance metrics logging
    - Security event classification
    - Error context enhancement
    - Request/response logging
    - Structured JSON output for production
    """
    
    def __init__(self, component: str, context: Dict[str, Any] = None):
        """Initialize enhanced logger for component."""
        # Configure enhanced structlog processors
        self._configure_enhanced_processors()
        
        # Get component logger with enhanced configuration
        self.logger = get_component_logger(component, context)
        self.component = component
        
        # Performance tracking
        self._operation_start_times: Dict[str, float] = {}
    
    def _configure_enhanced_processors(self) -> None:
        """Configure enhanced processors for production logging."""
        processors = [
            structlog.stdlib.filter_by_level,
            correlation_processor,
            performance_processor,
            security_processor,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(serializer=self._custom_json_serializer)
        ]
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def _custom_json_serializer(self, obj: Any, default=None, **kwargs) -> str:
        """Custom JSON serializer for complex objects."""
        def default_serializer(o):
            if isinstance(o, datetime):
                return o.isoformat()
            if hasattr(o, 'to_dict'):
                return o.to_dict()
            if hasattr(o, '__dict__'):
                return o.__dict__
            return str(o)
        
        # Use provided default or our default_serializer
        serializer = default or default_serializer
        return json.dumps(obj, default=serializer, separators=(',', ':'), **kwargs)
    
    def start_operation(self, operation_name: str, **context) -> str:
        """Start operation timing and context tracking."""
        operation_id = str(uuid.uuid4())
        self._operation_start_times[operation_id] = time.time()
        
        # Set operation context
        correlation_context.set_operation_context(operation_name, self.component)
        
        self.logger.info(
            "operation_started",
            operation=operation_name,
            operation_id=operation_id,
            **context
        )
        
        return operation_id
    
    def end_operation(self, operation_id: str, success: bool = True, **context) -> None:
        """End operation timing and log completion."""
        if operation_id not in self._operation_start_times:
            self.logger.warn("operation_end_without_start", operation_id=operation_id)
            return
        
        duration_ms = (time.time() - self._operation_start_times[operation_id]) * 1000
        del self._operation_start_times[operation_id]
        
        log_level = "info" if success else "error"
        getattr(self.logger, log_level)(
            "operation_completed",
            operation_id=operation_id,
            success=success,
            duration_ms=duration_ms,
            **context
        )
    
    def log_request(self, method: str, path: str, **context) -> None:
        """Log API request with full context."""
        self.logger.info(
            "api_request",
            http_method=method,
            path=path,
            request_type="incoming",
            **context
        )
    
    def log_response(self, status_code: int, duration_ms: float, **context) -> None:
        """Log API response with performance metrics."""
        log_level = "info" if 200 <= status_code < 400 else "error"
        getattr(self.logger, log_level)(
            "api_response",
            status_code=status_code,
            duration_ms=duration_ms,
            response_type="outgoing",
            **context
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log error with enhanced context and stack trace."""
        error_context = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "stack_trace": traceback.format_exc(),
            "error_id": str(uuid.uuid4())
        }
        
        if context:
            error_context.update(context)
        
        self.logger.error("error_occurred", **error_context)
    
    def log_security_event(self, event_type: str, severity: str, **context) -> None:
        """Log security-related events with special handling."""
        self.logger.bind(
            security_event=True,
            severity=severity,
            event_type=event_type
        ).warning("security_event", **context)
    
    def log_performance_metric(self, metric_name: str, value: Union[int, float], 
                             unit: str = "ms", **context) -> None:
        """Log performance metrics for monitoring."""
        self.logger.bind(
            metric_type="performance",
            metric_name=metric_name,
            metric_value=value,
            unit=unit
        ).info("performance_metric", **context)
    
    def log_audit_event(self, action: str, resource: str, success: bool, **context) -> None:
        """Log audit events for compliance."""
        self.logger.bind(
            audit_event=True,
            action=action,
            resource=resource,
            success=success
        ).info("audit_event", **context)


class PerformanceTracker:
    """Context manager for automatic performance tracking."""
    
    def __init__(self, logger: EnhancedLogger, operation_name: str, **context):
        self.logger = logger
        self.operation_name = operation_name
        self.context = context
        self.operation_id = None
        self.start_time = None
    
    def __enter__(self):
        self.operation_id = self.logger.start_operation(self.operation_name, **self.context)
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        if exc_val:
            self.context["error"] = str(exc_val)
        self.logger.end_operation(self.operation_id, success=success, **self.context)


def with_correlation_id(correlation_id: str = None):
    """Decorator to set correlation ID for operation scope."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            old_correlation_id = correlation_context.get_correlation_id()
            try:
                # Set correlation ID
                cid = correlation_id or str(uuid.uuid4())
                correlation_context.set_correlation_id(cid)
                
                # Execute function
                result = await func(*args, **kwargs)
                return result
            finally:
                # Restore previous correlation ID
                if old_correlation_id:
                    correlation_context.set_correlation_id(old_correlation_id)
                else:
                    correlation_context.clear()
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            old_correlation_id = correlation_context.get_correlation_id()
            try:
                # Set correlation ID
                cid = correlation_id or str(uuid.uuid4())
                correlation_context.set_correlation_id(cid)
                
                # Execute function
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore previous correlation ID
                if old_correlation_id:
                    correlation_context.set_correlation_id(old_correlation_id)
                else:
                    correlation_context.clear()
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def with_performance_logging(operation_name: str = None):
    """Decorator to automatically log performance metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create logger for the function's module
            logger = EnhancedLogger(func.__module__)
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with PerformanceTracker(logger, op_name):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Create logger for the function's module
            logger = EnhancedLogger(func.__module__)
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with PerformanceTracker(logger, op_name):
                return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class LogAggregator:
    """Production log aggregation and analytics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}
        self.security_events: List[Dict[str, Any]] = []
    
    def aggregate_performance_metric(self, metric_name: str, value: float) -> None:
        """Aggregate performance metrics."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def record_error(self, error_type: str) -> None:
        """Record error occurrence."""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def record_security_event(self, event: Dict[str, Any]) -> None:
        """Record security event for analysis."""
        event["timestamp"] = datetime.utcnow().isoformat()
        self.security_events.append(event)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        summary = {}
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "p95": sorted(values)[int(len(values) * 0.95)] if len(values) >= 20 else max(values)
                }
        return summary
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error statistics."""
        total_errors = sum(self.error_counts.values())
        return {
            "total_errors": total_errors,
            "by_type": self.error_counts.copy(),
            "error_rate": total_errors / (len(self.metrics.get("requests", [])) or 1)
        }
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security events summary."""
        return {
            "total_events": len(self.security_events),
            "recent_events": self.security_events[-10:] if self.security_events else [],
            "event_types": list(set(event.get("event_type") for event in self.security_events))
        }


# Global log aggregator instance
log_aggregator = LogAggregator()


# Convenience functions
def get_enhanced_logger(component: str, context: Dict[str, Any] = None) -> EnhancedLogger:
    """Get enhanced logger for component."""
    return EnhancedLogger(component, context)


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current context."""
    correlation_context.set_correlation_id(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return correlation_context.get_correlation_id()


def set_request_context(request_id: str, user_id: str = None, user_role: str = None) -> None:
    """Set request context for logging."""
    correlation_context.set_request_id(request_id)
    if user_id:
        correlation_context.set_user_context(user_id, user_role)


@asynccontextmanager
async def operation_context(logger: EnhancedLogger, operation_name: str, **context):
    """Async context manager for operation tracking."""
    operation_id = logger.start_operation(operation_name, **context)
    try:
        yield operation_id
    except Exception as e:
        logger.log_error(e, {"operation_id": operation_id})
        raise
    finally:
        logger.end_operation(operation_id, success=True)