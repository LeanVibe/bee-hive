"""
Enhanced Middleware for Production-Ready API

Provides comprehensive request/response logging, error handling with correlation IDs,
performance monitoring, and security event logging for production environments.
"""

import json
import time
import uuid
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable

from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from .enhanced_logging import (
    EnhancedLogger,
    correlation_context,
    set_request_context,
    log_aggregator
)
from .auth import (
    decode_jwt_token, 
    get_user_by_id,
    Permission,
    UserRole
)
from .database import get_session


class CorrelationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to inject correlation IDs for request tracing.
    
    This middleware ensures every request has a unique correlation ID
    that can be used to trace requests across the entire system.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = EnhancedLogger("correlation_middleware")
    
    async def dispatch(self, request: Request, call_next):
        # Generate or extract correlation ID
        correlation_id = (
            request.headers.get("X-Correlation-ID") or
            request.headers.get("X-Request-ID") or
            str(uuid.uuid4())
        )
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Set correlation context
        correlation_context.set_correlation_id(correlation_id)
        correlation_context.set_request_id(request_id)
        
        # Add to request state for other middleware
        request.state.correlation_id = correlation_id
        request.state.request_id = request_id
        request.state.request_start_time = time.time()
        
        # Log request initiation
        self.logger.log_request(
            request.method,
            str(request.url.path),
            correlation_id=correlation_id,
            request_id=request_id,
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
            content_type=request.headers.get("Content-Type")
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Add correlation headers to response
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Log unhandled exceptions
            self.logger.log_error(e, {
                "correlation_id": correlation_id,
                "request_id": request_id,
                "path": str(request.url.path),
                "method": request.method
            })
            raise
        finally:
            # Clear correlation context
            correlation_context.clear()


class EnhancedLoggingMiddleware(BaseHTTPMiddleware):
    """
    Enhanced logging middleware for comprehensive request/response monitoring.
    
    Provides detailed logging of API requests, responses, performance metrics,
    and error contexts for production observability.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = EnhancedLogger("api_requests")
        
        # Performance targets by endpoint category
        self.performance_targets = {
            "agents": 100,      # milliseconds
            "workflows": 150,
            "tasks": 100,
            "projects": 200,
            "coordination": 100,
            "observability": 50,
            "security": 75,
            "resources": 100,
            "contexts": 150,
            "enterprise": 200,
            "websocket": 50,
            "health": 25,
            "admin": 100,
            "integrations": 200,
            "dashboard": 100
        }
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Extract request context
        correlation_id = getattr(request.state, "correlation_id", None)
        request_id = getattr(request.state, "request_id", None)
        
        # Extract resource type from path
        path_parts = request.url.path.split("/")
        resource_type = "unknown"
        if len(path_parts) >= 4 and path_parts[2] == "v2":
            resource_type = path_parts[3]
        
        # Log detailed request information
        request_body_size = int(request.headers.get("Content-Length", 0))
        
        self.logger.logger.info(
            "api_request_detailed",
            method=request.method,
            path=str(request.url.path),
            query_params=dict(request.query_params),
            resource_type=resource_type,
            request_body_size_bytes=request_body_size,
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
            accept=request.headers.get("Accept"),
            content_type=request.headers.get("Content-Type")
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate performance metrics
        duration_ms = (time.time() - start_time) * 1000
        
        # Check performance targets
        target_ms = self.performance_targets.get(resource_type, 100)
        performance_status = "fast" if duration_ms <= target_ms else "slow"
        
        # Log response information
        response_size = int(response.headers.get("Content-Length", 0))
        
        self.logger.log_response(
            response.status_code,
            duration_ms,
            resource_type=resource_type,
            performance_status=performance_status,
            target_ms=target_ms,
            response_size_bytes=response_size,
            response_type=response.headers.get("Content-Type")
        )
        
        # Log performance metrics
        self.logger.log_performance_metric(
            f"{resource_type}_response_time",
            duration_ms,
            unit="ms",
            status_code=response.status_code,
            performance_status=performance_status
        )
        
        # Aggregate metrics for monitoring
        log_aggregator.aggregate_performance_metric("api_response_time", duration_ms)
        log_aggregator.aggregate_performance_metric(f"{resource_type}_response_time", duration_ms)
        
        # Log slow requests as warnings
        if duration_ms > target_ms:
            self.logger.logger.warning(
                "slow_api_request",
                duration_ms=duration_ms,
                target_ms=target_ms,
                resource_type=resource_type,
                path=str(request.url.path),
                method=request.method,
                status_code=response.status_code
            )
        
        # Add performance headers to response
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        response.headers["X-Performance-Target"] = f"{target_ms}ms"
        response.headers["X-Performance-Status"] = performance_status
        
        return response


class EnhancedAuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Enhanced authentication middleware with security logging and monitoring.
    
    Provides authentication with detailed security event logging,
    brute force detection, and audit trails.
    """
    
    # Public endpoints that don't require authentication
    PUBLIC_ENDPOINTS = {
        "/api/v2/",
        "/api/v2/health/status",
        "/api/v2/health/ready",
        "/api/v2/security/login",
        "/api/v2/security/refresh",
        "/docs",
        "/redoc",
        "/openapi.json"
    }
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = EnhancedLogger("auth_middleware")
        
        # Security monitoring
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.max_attempts = 5
        self.lockout_duration_minutes = 15
    
    async def dispatch(self, request: Request, call_next):
        # Skip authentication for public endpoints
        if request.url.path in self.PUBLIC_ENDPOINTS:
            return await call_next(request)
        
        # Extract request context
        correlation_id = getattr(request.state, "correlation_id", None)
        request_id = getattr(request.state, "request_id", None)
        client_ip = request.client.host if request.client else "unknown"
        
        # Set user context for logging
        set_request_context(request_id, correlation_id=correlation_id)
        
        # Extract and validate token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            # Log authentication attempt without token
            self.logger.log_security_event(
                "authentication_missing_token",
                "MEDIUM",
                client_ip=client_ip,
                path=str(request.url.path),
                user_agent=request.headers.get("User-Agent")
            )
            
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "authentication_required",
                    "message": "Bearer token required for this endpoint",
                    "request_id": request_id,
                    "correlation_id": correlation_id
                }
            )
        
        token = auth_header.split(" ")[1]
        
        # Check for brute force attempts
        if self._is_ip_locked_out(client_ip):
            self.logger.log_security_event(
                "authentication_brute_force_lockout",
                "HIGH",
                client_ip=client_ip,
                path=str(request.url.path),
                lockout_duration_minutes=self.lockout_duration_minutes
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "too_many_attempts",
                    "message": f"IP temporarily locked due to too many failed attempts",
                    "request_id": request_id,
                    "correlation_id": correlation_id
                }
            )
        
        try:
            # Decode and validate JWT token
            payload = decode_jwt_token(token)
            user_id = payload.get("sub")
            
            if not user_id:
                raise ValueError("Token missing user ID")
            
            # Get user from database
            async with get_session() as db:
                user = await get_user_by_id(db, user_id)
                if not user:
                    raise ValueError("User not found")
            
            # Log successful authentication
            self.logger.log_security_event(
                "authentication_success",
                "INFO",
                user_id=user_id,
                user_role=user.role.value if user.role else None,
                client_ip=client_ip,
                path=str(request.url.path)
            )
            
            # Clear failed attempts on successful authentication
            if client_ip in self.failed_attempts:
                del self.failed_attempts[client_ip]
            
            # Inject user context into request
            request.state.current_user = user
            request.state.user_id = user_id
            request.state.user_role = user.role
            request.state.permissions = user.permissions
            
            # Update correlation context with user info
            correlation_context.set_user_context(user_id, user.role.value if user.role else None)
            
            # Log audit event for authenticated request
            self.logger.log_audit_event(
                "authenticated_request",
                str(request.url.path),
                success=True,
                user_id=user_id,
                method=request.method,
                client_ip=client_ip
            )
            
        except Exception as e:
            # Record failed attempt
            self._record_failed_attempt(client_ip)
            
            # Log authentication failure with details
            self.logger.log_security_event(
                "authentication_failed",
                "HIGH",
                error=str(e),
                client_ip=client_ip,
                path=str(request.url.path),
                user_agent=request.headers.get("User-Agent"),
                token_present=bool(token),
                failed_attempts_count=len(self.failed_attempts.get(client_ip, []))
            )
            
            # Aggregate error for monitoring
            log_aggregator.record_error("authentication_error")
            
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "authentication_failed",
                    "message": "Invalid or expired token",
                    "request_id": request_id,
                    "correlation_id": correlation_id
                }
            )
        
        return await call_next(request)
    
    def _is_ip_locked_out(self, client_ip: str) -> bool:
        """Check if IP address is currently locked out."""
        if client_ip not in self.failed_attempts:
            return False
        
        attempts = self.failed_attempts[client_ip]
        cutoff_time = datetime.utcnow().timestamp() - (self.lockout_duration_minutes * 60)
        
        # Count recent attempts
        recent_attempts = [
            attempt for attempt in attempts 
            if attempt.timestamp() > cutoff_time
        ]
        
        # Update attempts list
        self.failed_attempts[client_ip] = recent_attempts
        
        return len(recent_attempts) >= self.max_attempts
    
    def _record_failed_attempt(self, client_ip: str) -> None:
        """Record a failed authentication attempt."""
        if client_ip not in self.failed_attempts:
            self.failed_attempts[client_ip] = []
        
        self.failed_attempts[client_ip].append(datetime.utcnow())
        
        # Keep only recent attempts
        cutoff_time = datetime.utcnow().timestamp() - (self.lockout_duration_minutes * 60)
        self.failed_attempts[client_ip] = [
            attempt for attempt in self.failed_attempts[client_ip]
            if attempt.timestamp() > cutoff_time
        ]


class EnhancedErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Enhanced error handling middleware with detailed error context and monitoring.
    
    Provides comprehensive error logging, error classification,
    and error recovery patterns for production systems.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = EnhancedLogger("error_handler")
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # Extract request context
            correlation_id = getattr(request.state, "correlation_id", None)
            request_id = getattr(request.state, "request_id", None)
            user_id = getattr(request.state, "user_id", None)
            
            # Log HTTP exception with context
            self.logger.logger.warning(
                "http_exception_occurred",
                status_code=e.status_code,
                detail=e.detail,
                path=str(request.url.path),
                method=request.method,
                user_id=user_id
            )
            
            # Aggregate error for monitoring
            log_aggregator.record_error(f"http_{e.status_code}")
            
            # Return formatted error response
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": "http_exception",
                    "message": e.detail,
                    "status_code": e.status_code,
                    "request_id": request_id,
                    "correlation_id": correlation_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url.path)
                }
            )
            
        except Exception as e:
            # Extract request context
            correlation_id = getattr(request.state, "correlation_id", None)
            request_id = getattr(request.state, "request_id", None)
            user_id = getattr(request.state, "user_id", None)
            
            # Generate error ID for tracking
            error_id = str(uuid.uuid4())
            
            # Enhanced error logging with full context
            error_context = {
                "error_id": error_id,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "stack_trace": traceback.format_exc(),
                "path": str(request.url.path),
                "method": request.method,
                "user_id": user_id,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "client_ip": request.client.host if request.client else None
            }
            
            # Log error with full context
            self.logger.log_error(e, error_context)
            
            # Log security event if error might be security-related
            if self._is_security_related_error(e, str(request.url.path)):
                self.logger.log_security_event(
                    "potential_security_error",
                    "MEDIUM",
                    error_type=type(e).__name__,
                    path=str(request.url.path),
                    user_id=user_id,
                    client_ip=request.client.host if request.client else None
                )
            
            # Aggregate error for monitoring
            log_aggregator.record_error(type(e).__name__)
            
            # Return safe error response (don't leak internal details)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "internal_server_error",
                    "message": "An unexpected error occurred",
                    "error_id": error_id,
                    "request_id": request_id,
                    "correlation_id": correlation_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": str(request.url.path)
                }
            )
    
    def _is_security_related_error(self, error: Exception, path: str) -> bool:
        """Determine if an error might be security-related."""
        security_keywords = [
            "permission", "access", "auth", "token", "forbidden", 
            "unauthorized", "sql", "injection", "xss"
        ]
        
        error_text = str(error).lower()
        path_text = path.lower()
        
        return any(
            keyword in error_text or keyword in path_text 
            for keyword in security_keywords
        )


class SecurityAuditMiddleware(BaseHTTPMiddleware):
    """
    Security audit middleware for comprehensive security event logging.
    
    Monitors and logs security-relevant events, suspicious activities,
    and compliance-required audit trails.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = EnhancedLogger("security_audit")
        
        # Sensitive endpoints that require audit logging
        self.sensitive_endpoints = {
            "/api/v2/security/",
            "/api/v2/admin/",
            "/api/v2/agents/",
            "/api/v2/contexts/"
        }
    
    async def dispatch(self, request: Request, call_next):
        # Check if this is a sensitive endpoint
        is_sensitive = any(
            request.url.path.startswith(endpoint) 
            for endpoint in self.sensitive_endpoints
        )
        
        if is_sensitive:
            # Extract request context
            correlation_id = getattr(request.state, "correlation_id", None)
            request_id = getattr(request.state, "request_id", None)
            user_id = getattr(request.state, "user_id", None)
            
            # Log sensitive endpoint access
            self.logger.log_security_event(
                "sensitive_endpoint_access",
                "INFO",
                path=str(request.url.path),
                method=request.method,
                user_id=user_id,
                client_ip=request.client.host if request.client else None,
                user_agent=request.headers.get("User-Agent")
            )
        
        # Process request
        response = await call_next(request)
        
        # Log security audit events based on response
        if is_sensitive and hasattr(response, 'status_code'):
            success = 200 <= response.status_code < 300
            
            self.logger.log_audit_event(
                f"sensitive_endpoint_{request.method.lower()}",
                str(request.url.path),
                success=success,
                user_id=getattr(request.state, "user_id", None),
                status_code=response.status_code,
                client_ip=request.client.host if request.client else None
            )
        
        return response


# Factory functions for middleware instances
def create_correlation_middleware() -> Callable[[ASGIApp], CorrelationMiddleware]:
    """Create correlation middleware instance."""
    return CorrelationMiddleware


def create_enhanced_logging_middleware() -> Callable[[ASGIApp], EnhancedLoggingMiddleware]:
    """Create enhanced logging middleware instance."""
    return EnhancedLoggingMiddleware


def create_enhanced_auth_middleware() -> Callable[[ASGIApp], EnhancedAuthenticationMiddleware]:
    """Create enhanced authentication middleware instance."""
    return EnhancedAuthenticationMiddleware


def create_enhanced_error_middleware() -> Callable[[ASGIApp], EnhancedErrorHandlingMiddleware]:
    """Create enhanced error handling middleware instance."""
    return EnhancedErrorHandlingMiddleware


def create_security_audit_middleware() -> Callable[[ASGIApp], SecurityAuditMiddleware]:
    """Create security audit middleware instance."""
    return SecurityAuditMiddleware


# Complete middleware stack for production
PRODUCTION_MIDDLEWARE_STACK = [
    create_correlation_middleware(),
    create_enhanced_logging_middleware(),
    create_enhanced_auth_middleware(),
    create_enhanced_error_middleware(),
    create_security_audit_middleware()
]