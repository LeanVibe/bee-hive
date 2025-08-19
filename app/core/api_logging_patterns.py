"""
Consistent API Logging Patterns for Production-Ready System

Provides standardized logging patterns for API endpoints, request/response
logging, performance monitoring, and error handling across the system.
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from functools import wraps

from fastapi import Request, Response, HTTPException, status
from fastapi.routing import APIRoute

from .enhanced_logging import (
    EnhancedLogger,
    PerformanceTracker,
    correlation_context,
    log_aggregator
)


class APIEndpointLogger:
    """
    Enhanced logging for individual API endpoints.
    
    Provides consistent logging patterns for request/response cycles,
    error handling, and performance monitoring.
    """
    
    def __init__(self, endpoint_name: str, component: str = "api"):
        """Initialize endpoint logger."""
        self.endpoint_name = endpoint_name
        self.component = component
        self.logger = EnhancedLogger(f"{component}.{endpoint_name}")
        
        # Performance thresholds by endpoint type
        self.performance_thresholds = {
            "health": 25,
            "auth": 100,
            "agents": 150,
            "tasks": 100,
            "workflows": 200,
            "projects": 250,
            "admin": 100,
            "security": 75
        }
    
    def get_performance_threshold(self) -> int:
        """Get performance threshold for this endpoint."""
        for endpoint_type, threshold in self.performance_thresholds.items():
            if endpoint_type in self.endpoint_name.lower():
                return threshold
        return 150  # Default threshold
    
    def log_request_start(self, request: Request, **context) -> str:
        """Log the start of a request."""
        operation_id = self.logger.start_operation(
            f"{self.endpoint_name}_request",
            method=request.method,
            path=str(request.url.path),
            query_params=dict(request.query_params),
            **context
        )
        
        # Log request details
        self.logger.log_request(
            request.method,
            str(request.url.path),
            endpoint=self.endpoint_name,
            body_size=int(request.headers.get("Content-Length", 0)),
            content_type=request.headers.get("Content-Type"),
            accept=request.headers.get("Accept"),
            operation_id=operation_id
        )
        
        return operation_id
    
    def log_request_end(self, operation_id: str, response: Response, 
                       duration_ms: float, **context) -> None:
        """Log the end of a request."""
        success = 200 <= response.status_code < 400
        threshold = self.get_performance_threshold()
        
        # End operation tracking
        self.logger.end_operation(
            operation_id,
            success=success,
            status_code=response.status_code,
            duration_ms=duration_ms,
            performance_status="fast" if duration_ms <= threshold else "slow",
            **context
        )
        
        # Log response details
        self.logger.log_response(
            response.status_code,
            duration_ms,
            endpoint=self.endpoint_name,
            response_size=int(response.headers.get("Content-Length", 0)),
            content_type=response.headers.get("Content-Type"),
            performance_threshold=threshold
        )
        
        # Log performance metrics
        self.logger.log_performance_metric(
            f"{self.endpoint_name}_response_time",
            duration_ms,
            unit="ms",
            status_code=response.status_code
        )
        
        # Aggregate metrics for monitoring
        log_aggregator.aggregate_performance_metric(
            f"{self.endpoint_name}_response_time", 
            duration_ms
        )
        
        # Log slow requests
        if duration_ms > threshold:
            self.logger.logger.warning(
                "slow_endpoint_response",
                endpoint=self.endpoint_name,
                duration_ms=duration_ms,
                threshold_ms=threshold,
                status_code=response.status_code
            )
    
    def log_validation_error(self, error: Exception, request_data: Dict[str, Any] = None) -> None:
        """Log validation errors with context."""
        self.logger.log_error(error, {
            "error_category": "validation",
            "endpoint": self.endpoint_name,
            "request_data_keys": list(request_data.keys()) if request_data else []
        })
        
        log_aggregator.record_error("validation_error")
    
    def log_business_logic_error(self, error: Exception, operation: str, **context) -> None:
        """Log business logic errors."""
        self.logger.log_error(error, {
            "error_category": "business_logic",
            "endpoint": self.endpoint_name,
            "operation": operation,
            **context
        })
        
        log_aggregator.record_error("business_logic_error")
    
    def log_database_error(self, error: Exception, operation: str, **context) -> None:
        """Log database-related errors."""
        self.logger.log_error(error, {
            "error_category": "database",
            "endpoint": self.endpoint_name,
            "operation": operation,
            **context
        })
        
        self.logger.log_security_event(
            "database_error_occurred",
            "MEDIUM",
            endpoint=self.endpoint_name,
            operation=operation,
            error_type=type(error).__name__
        )
        
        log_aggregator.record_error("database_error")
    
    def log_external_service_error(self, service: str, error: Exception, **context) -> None:
        """Log external service errors."""
        self.logger.log_error(error, {
            "error_category": "external_service",
            "endpoint": self.endpoint_name,
            "service": service,
            **context
        })
        
        log_aggregator.record_error(f"{service}_error")
    
    def log_audit_event(self, action: str, resource: str, success: bool, **context) -> None:
        """Log audit events for compliance."""
        self.logger.log_audit_event(
            action,
            resource,
            success=success,
            endpoint=self.endpoint_name,
            **context
        )
    
    def log_security_event(self, event_type: str, severity: str, **context) -> None:
        """Log security events."""
        self.logger.log_security_event(
            event_type,
            severity,
            endpoint=self.endpoint_name,
            **context
        )


def with_api_logging(endpoint_name: str, component: str = "api"):
    """
    Decorator for automatic API endpoint logging.
    
    Provides comprehensive request/response logging, error handling,
    and performance monitoring for API endpoints.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(request: Request, *args, **kwargs):
            logger = APIEndpointLogger(endpoint_name, component)
            start_time = time.time()
            operation_id = None
            
            try:
                # Log request start
                operation_id = logger.log_request_start(request)
                
                # Execute endpoint function
                result = await func(request, *args, **kwargs)
                
                # Create response object if result is not already a Response
                if hasattr(result, 'status_code'):
                    response = result
                else:
                    from fastapi.responses import JSONResponse
                    response = JSONResponse(content=result)
                
                # Log successful completion
                duration_ms = (time.time() - start_time) * 1000
                logger.log_request_end(operation_id, response, duration_ms)
                
                return result
                
            except HTTPException as e:
                # Log HTTP exceptions
                duration_ms = (time.time() - start_time) * 1000
                logger.logger.warning(
                    "http_exception_in_endpoint",
                    endpoint=endpoint_name,
                    status_code=e.status_code,
                    detail=e.detail,
                    duration_ms=duration_ms
                )
                
                if operation_id:
                    logger.logger.end_operation(operation_id, success=False)
                
                raise
                
            except Exception as e:
                # Log unexpected errors
                duration_ms = (time.time() - start_time) * 1000
                logger.log_business_logic_error(e, "endpoint_execution", duration_ms=duration_ms)
                
                if operation_id:
                    logger.logger.end_operation(operation_id, success=False)
                
                raise
        
        @wraps(func)
        def sync_wrapper(request: Request, *args, **kwargs):
            logger = APIEndpointLogger(endpoint_name, component)
            start_time = time.time()
            operation_id = None
            
            try:
                # Log request start
                operation_id = logger.log_request_start(request)
                
                # Execute endpoint function
                result = func(request, *args, **kwargs)
                
                # Create response object if result is not already a Response
                if hasattr(result, 'status_code'):
                    response = result
                else:
                    from fastapi.responses import JSONResponse
                    response = JSONResponse(content=result)
                
                # Log successful completion
                duration_ms = (time.time() - start_time) * 1000
                logger.log_request_end(operation_id, response, duration_ms)
                
                return result
                
            except HTTPException as e:
                # Log HTTP exceptions
                duration_ms = (time.time() - start_time) * 1000
                logger.logger.warning(
                    "http_exception_in_endpoint",
                    endpoint=endpoint_name,
                    status_code=e.status_code,
                    detail=e.detail,
                    duration_ms=duration_ms
                )
                
                if operation_id:
                    logger.logger.end_operation(operation_id, success=False)
                
                raise
                
            except Exception as e:
                # Log unexpected errors
                duration_ms = (time.time() - start_time) * 1000
                logger.log_business_logic_error(e, "endpoint_execution", duration_ms=duration_ms)
                
                if operation_id:
                    logger.logger.end_operation(operation_id, success=False)
                
                raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class DatabaseOperationLogger:
    """Logger for database operations with enhanced context."""
    
    def __init__(self, component: str):
        self.logger = EnhancedLogger(f"database.{component}")
        self.component = component
    
    def log_query_start(self, operation: str, table: str = None, **context) -> str:
        """Log start of database query."""
        return self.logger.start_operation(
            f"db_{operation}",
            table=table,
            component=self.component,
            **context
        )
    
    def log_query_end(self, operation_id: str, success: bool, 
                     rows_affected: int = None, **context) -> None:
        """Log end of database query."""
        self.logger.end_operation(
            operation_id,
            success=success,
            rows_affected=rows_affected,
            **context
        )
        
        if rows_affected is not None:
            self.logger.log_performance_metric(
                f"db_rows_{self.component}",
                rows_affected,
                unit="rows"
            )
    
    def log_transaction_start(self, operation: str, **context) -> str:
        """Log start of database transaction."""
        return self.logger.start_operation(
            f"db_transaction_{operation}",
            component=self.component,
            **context
        )
    
    def log_transaction_end(self, operation_id: str, success: bool, **context) -> None:
        """Log end of database transaction."""
        self.logger.end_operation(operation_id, success=success, **context)
        
        # Log audit event for transactions
        self.logger.log_audit_event(
            f"database_transaction",
            f"component:{self.component}",
            success=success,
            **context
        )
    
    def log_connection_event(self, event_type: str, **context) -> None:
        """Log database connection events."""
        self.logger.logger.info(
            f"database_connection_{event_type}",
            component=self.component,
            **context
        )
        
        if event_type == "error":
            self.logger.log_security_event(
                "database_connection_error",
                "HIGH",
                component=self.component,
                **context
            )


def with_database_logging(operation: str, table: str = None):
    """Decorator for database operation logging."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract component name from function module
            component = func.__module__.split('.')[-1]
            db_logger = DatabaseOperationLogger(component)
            
            operation_id = db_logger.log_query_start(operation, table)
            
            try:
                result = await func(*args, **kwargs)
                
                # Try to extract rows affected from result
                rows_affected = None
                if hasattr(result, 'rowcount'):
                    rows_affected = result.rowcount
                elif isinstance(result, list):
                    rows_affected = len(result)
                
                db_logger.log_query_end(operation_id, True, rows_affected)
                return result
                
            except Exception as e:
                db_logger.log_query_end(operation_id, False)
                db_logger.logger.log_error(e, {
                    "operation": operation,
                    "table": table,
                    "component": component
                })
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Extract component name from function module
            component = func.__module__.split('.')[-1]
            db_logger = DatabaseOperationLogger(component)
            
            operation_id = db_logger.log_query_start(operation, table)
            
            try:
                result = func(*args, **kwargs)
                
                # Try to extract rows affected from result
                rows_affected = None
                if hasattr(result, 'rowcount'):
                    rows_affected = result.rowcount
                elif isinstance(result, list):
                    rows_affected = len(result)
                
                db_logger.log_query_end(operation_id, True, rows_affected)
                return result
                
            except Exception as e:
                db_logger.log_query_end(operation_id, False)
                db_logger.logger.log_error(e, {
                    "operation": operation,
                    "table": table,
                    "component": component
                })
                raise
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class SecurityEventLogger:
    """Specialized logger for security events."""
    
    def __init__(self, component: str):
        self.logger = EnhancedLogger(f"security.{component}")
        self.component = component
    
    def log_authentication_attempt(self, success: bool, user_id: str = None, 
                                 client_ip: str = None, **context) -> None:
        """Log authentication attempts."""
        event_type = "authentication_success" if success else "authentication_failure"
        severity = "INFO" if success else "HIGH"
        
        self.logger.log_security_event(
            event_type,
            severity,
            user_id=user_id,
            client_ip=client_ip,
            component=self.component,
            **context
        )
        
        self.logger.log_audit_event(
            "authentication_attempt",
            f"user:{user_id or 'unknown'}",
            success=success,
            client_ip=client_ip,
            **context
        )
    
    def log_authorization_check(self, success: bool, user_id: str, 
                              resource: str, permission: str, **context) -> None:
        """Log authorization checks."""
        self.logger.log_audit_event(
            "authorization_check",
            resource,
            success=success,
            user_id=user_id,
            permission=permission,
            component=self.component,
            **context
        )
        
        if not success:
            self.logger.log_security_event(
                "authorization_denied",
                "MEDIUM",
                user_id=user_id,
                resource=resource,
                permission=permission,
                **context
            )
    
    def log_suspicious_activity(self, activity_type: str, severity: str, 
                              user_id: str = None, **context) -> None:
        """Log suspicious security activities."""
        self.logger.log_security_event(
            f"suspicious_{activity_type}",
            severity,
            user_id=user_id,
            component=self.component,
            **context
        )
        
        # Record security event for aggregation
        log_aggregator.record_security_event({
            "event_type": f"suspicious_{activity_type}",
            "severity": severity,
            "user_id": user_id,
            "component": self.component,
            **context
        })


# Convenience functions for common logging patterns
def get_api_logger(endpoint_name: str, component: str = "api") -> APIEndpointLogger:
    """Get API endpoint logger."""
    return APIEndpointLogger(endpoint_name, component)


def get_db_logger(component: str) -> DatabaseOperationLogger:
    """Get database operation logger."""
    return DatabaseOperationLogger(component)


def get_security_logger(component: str) -> SecurityEventLogger:
    """Get security event logger."""
    return SecurityEventLogger(component)


# Performance monitoring helpers
class PerformanceMonitor:
    """Monitor and alert on performance metrics."""
    
    def __init__(self):
        self.logger = EnhancedLogger("performance_monitor")
        
        # Performance thresholds
        self.thresholds = {
            "api_response_time": 200,
            "database_query_time": 100,
            "authentication_time": 50,
            "task_delegation_time": 150
        }
    
    def check_performance_threshold(self, metric_name: str, value: float) -> None:
        """Check if metric exceeds performance threshold."""
        threshold = self.thresholds.get(metric_name)
        if threshold and value > threshold:
            self.logger.logger.warning(
                "performance_threshold_exceeded",
                metric=metric_name,
                value=value,
                threshold=threshold,
                severity="HIGH" if value > threshold * 2 else "MEDIUM"
            )
    
    def log_system_performance_summary(self) -> None:
        """Log system-wide performance summary."""
        summary = log_aggregator.get_performance_summary()
        
        self.logger.logger.info(
            "system_performance_summary",
            **summary
        )
        
        # Check for concerning trends
        for metric, stats in summary.items():
            if isinstance(stats, dict) and 'avg' in stats:
                self.check_performance_threshold(metric, stats['avg'])


# Global performance monitor instance
performance_monitor = PerformanceMonitor()