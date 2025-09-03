"""
TaskExecutionAPI Middleware - Enterprise Security and Performance Layer

Consolidated middleware providing OAuth2 authentication, task-specific permissions,
rate limiting, audit logging, and performance optimization following Phase 2-3 patterns.

Features:
- OAuth2 + task-specific RBAC permissions
- Intelligent rate limiting with burst protection
- Comprehensive audit logging for compliance
- Performance monitoring and caching
- Request/response validation and sanitization
- Circuit breaker patterns for resilience
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from uuid import uuid4
from functools import wraps

from fastapi import Request, Response, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import structlog
import redis.asyncio as redis

from ....core.security import verify_token, get_current_user
from ....core.redis_integration import get_redis_service
from ....core.audit_logger import create_audit_logger, AuditLogger
from ....models.task import TaskStatus, TaskPriority


logger = structlog.get_logger(__name__)
security = HTTPBearer()


# ===============================================================================
# AUTHENTICATION AND AUTHORIZATION MIDDLEWARE
# ===============================================================================

class TaskPermissions:
    """Task-specific permission definitions."""
    
    # Core task operations
    CREATE_TASK = "tasks:create"
    READ_TASK = "tasks:read"  
    UPDATE_TASK = "tasks:update"
    DELETE_TASK = "tasks:delete"
    ASSIGN_TASK = "tasks:assign"
    
    # Workflow operations
    CREATE_WORKFLOW = "workflows:create"
    EXECUTE_WORKFLOW = "workflows:execute"
    MANAGE_WORKFLOW = "workflows:manage"
    
    # Scheduling operations
    SCHEDULE_ANALYZE = "scheduling:analyze"
    SCHEDULE_OPTIMIZE = "scheduling:optimize"
    SCHEDULE_RESOLVE = "scheduling:resolve"
    
    # Administrative operations  
    TASK_ADMIN = "tasks:admin"
    SYSTEM_MONITOR = "system:monitor"
    AUDIT_ACCESS = "audit:access"
    
    # Team coordination
    COORDINATE_TEAM = "team:coordinate"
    MANAGE_AGENTS = "agents:manage"


class TaskAuthenticationError(Exception):
    """Custom authentication error for task operations."""
    def __init__(self, message: str, error_code: str = "AUTH_FAILED"):
        self.message = message
        self.error_code = error_code
        super().__init__(message)


async def verify_task_permission(
    token: HTTPAuthorizationCredentials = Depends(security),
    required_permission: str = None
) -> Dict[str, Any]:
    """
    Verify OAuth2 token and check task-specific permissions.
    
    Args:
        token: OAuth2 bearer token
        required_permission: Required permission for operation
        
    Returns:
        Dict containing user information and permissions
        
    Raises:
        HTTPException: On authentication or authorization failure
    """
    try:
        # Verify OAuth2 token
        user_info = await verify_oauth2_token(token.credentials)
        if not user_info:
            raise TaskAuthenticationError("Invalid or expired token")
        
        # Check if specific permission is required
        if required_permission:
            user_permissions = user_info.get("permissions", [])
            
            # Check for admin permissions (bypass specific checks)
            if TaskPermissions.TASK_ADMIN in user_permissions:
                logger.info("Admin access granted", user_id=user_info.get("user_id"))
                return user_info
            
            # Check specific permission
            if required_permission not in user_permissions:
                raise TaskAuthenticationError(
                    f"Insufficient permissions. Required: {required_permission}",
                    error_code="INSUFFICIENT_PERMISSIONS"
                )
        
        logger.info("Task permission verified", 
                   user_id=user_info.get("user_id"),
                   permission=required_permission)
        return user_info
        
    except TaskAuthenticationError:
        raise
    except Exception as e:
        logger.error("Token verification failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"}
        )


# Permission dependency factories
def require_permission(permission: str):
    """Create a dependency that requires a specific permission."""
    async def permission_dependency(
        user: Dict[str, Any] = Depends(lambda: verify_task_permission(required_permission=permission))
    ):
        return user
    return permission_dependency


# Common permission dependencies
require_task_create = require_permission(TaskPermissions.CREATE_TASK)
require_task_read = require_permission(TaskPermissions.READ_TASK)
require_task_update = require_permission(TaskPermissions.UPDATE_TASK)
require_task_delete = require_permission(TaskPermissions.DELETE_TASK)
require_workflow_create = require_permission(TaskPermissions.CREATE_WORKFLOW)
require_workflow_execute = require_permission(TaskPermissions.EXECUTE_WORKFLOW)
require_schedule_optimize = require_permission(TaskPermissions.SCHEDULE_OPTIMIZE)
require_team_coordinate = require_permission(TaskPermissions.COORDINATE_TEAM)
require_admin_access = require_permission(TaskPermissions.TASK_ADMIN)


# ===============================================================================
# RATE LIMITING MIDDLEWARE
# ===============================================================================

class IntelligentRateLimiter:
    """
    Intelligent rate limiting with burst protection and user-based quotas.
    
    Features:
    - Per-user rate limiting
    - Burst protection with token bucket
    - Priority-based rate limiting
    - Redis-based distributed limiting
    """
    
    def __init__(self, redis_service=None):
        self.redis_service = redis_service
        
        # Rate limit configurations
        self.default_limits = {
            "requests_per_minute": 100,
            "requests_per_hour": 3000,
            "burst_size": 20
        }
        
        self.admin_limits = {
            "requests_per_minute": 500,
            "requests_per_hour": 15000, 
            "burst_size": 100
        }
        
        self.endpoint_limits = {
            "/tasks/create": {"requests_per_minute": 50, "burst_size": 10},
            "/workflows/execute": {"requests_per_minute": 20, "burst_size": 5},
            "/scheduling/optimize": {"requests_per_minute": 10, "burst_size": 3}
        }
    
    async def check_rate_limit(
        self, 
        user_id: str, 
        endpoint: str = None,
        user_permissions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Check rate limit for user and endpoint.
        
        Returns:
            Dict with rate limit information and whether request is allowed
        """
        try:
            # Determine rate limits based on user permissions
            limits = self.admin_limits if TaskPermissions.TASK_ADMIN in (user_permissions or []) else self.default_limits
            
            # Apply endpoint-specific limits if configured
            if endpoint and endpoint in self.endpoint_limits:
                endpoint_config = self.endpoint_limits[endpoint]
                limits = {**limits, **endpoint_config}
            
            current_time = int(time.time())
            minute_window = current_time // 60
            hour_window = current_time // 3600
            
            # Redis keys for rate limiting
            minute_key = f"rate_limit:{user_id}:minute:{minute_window}"
            hour_key = f"rate_limit:{user_id}:hour:{hour_window}"
            burst_key = f"rate_limit:{user_id}:burst"
            
            if self.redis_service:
                # Check current usage
                minute_count = await self.redis_service.get(minute_key) or 0
                hour_count = await self.redis_service.get(hour_key) or 0
                burst_tokens = await self.redis_service.get(burst_key) or limits["burst_size"]
                
                minute_count = int(minute_count)
                hour_count = int(hour_count)
                burst_tokens = int(burst_tokens)
                
                # Check limits
                if hour_count >= limits["requests_per_hour"]:
                    return {
                        "allowed": False,
                        "reason": "hourly_limit_exceeded",
                        "reset_time": (hour_window + 1) * 3600,
                        "limits": limits,
                        "current_usage": {"hour": hour_count, "minute": minute_count}
                    }
                
                if minute_count >= limits["requests_per_minute"] and burst_tokens <= 0:
                    return {
                        "allowed": False,
                        "reason": "minute_limit_exceeded",
                        "reset_time": (minute_window + 1) * 60,
                        "limits": limits,
                        "current_usage": {"hour": hour_count, "minute": minute_count}
                    }
                
                # Increment counters
                await self.redis_service.incr(minute_key)
                await self.redis_service.expire(minute_key, 60)
                await self.redis_service.incr(hour_key)
                await self.redis_service.expire(hour_key, 3600)
                
                # Handle burst tokens
                if minute_count >= limits["requests_per_minute"]:
                    await self.redis_service.decr(burst_key)
                    await self.redis_service.expire(burst_key, 60)
                else:
                    # Replenish burst tokens
                    await self.redis_service.set(burst_key, min(limits["burst_size"], burst_tokens + 1))
                    await self.redis_service.expire(burst_key, 60)
            
            return {
                "allowed": True,
                "limits": limits,
                "current_usage": {"hour": hour_count + 1, "minute": minute_count + 1},
                "burst_tokens_remaining": max(0, burst_tokens - 1) if minute_count >= limits["requests_per_minute"] else burst_tokens
            }
            
        except Exception as e:
            logger.error("Rate limit check failed", user_id=user_id, endpoint=endpoint, error=str(e))
            # Allow request on rate limiter failure (fail open)
            return {"allowed": True, "fallback": True}


rate_limiter = IntelligentRateLimiter()


async def check_rate_limit_dependency(
    request: Request,
    user: Dict[str, Any] = Depends(get_current_user)
) -> None:
    """Dependency to check rate limits for authenticated requests."""
    user_id = user.get("user_id", "anonymous")
    user_permissions = user.get("permissions", [])
    endpoint = request.url.path
    
    # Initialize rate limiter with Redis if not done
    if not rate_limiter.redis_service:
        rate_limiter.redis_service = get_redis_service()
    
    rate_check = await rate_limiter.check_rate_limit(
        user_id=user_id,
        endpoint=endpoint, 
        user_permissions=user_permissions
    )
    
    if not rate_check["allowed"]:
        logger.warning("Rate limit exceeded",
                      user_id=user_id,
                      endpoint=endpoint,
                      reason=rate_check["reason"])
        
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "Rate limit exceeded",
                "reason": rate_check["reason"],
                "reset_time": rate_check.get("reset_time"),
                "limits": rate_check.get("limits")
            },
            headers={
                "Retry-After": str(rate_check.get("reset_time", 60)),
                "X-RateLimit-Limit": str(rate_check["limits"]["requests_per_minute"]),
                "X-RateLimit-Remaining": str(max(0, rate_check["limits"]["requests_per_minute"] - rate_check["current_usage"]["minute"])),
                "X-RateLimit-Reset": str(rate_check.get("reset_time", int(time.time()) + 60))
            }
        )


# ===============================================================================
# AUDIT LOGGING MIDDLEWARE 
# ===============================================================================

class TaskAuditLogger:
    """
    Comprehensive audit logging for task execution operations.
    
    Logs all task, workflow, and scheduling operations for compliance
    and security monitoring.
    """
    
    def __init__(self):
        self.logger = structlog.get_logger("task_execution_api_audit")
        self.sensitive_fields = {
            "password", "token", "secret", "key", "credentials", "authorization"
        }
    
    def sanitize_data(self, data: Any) -> Any:
        """Sanitize data by removing sensitive information."""
        if isinstance(data, dict):
            return {
                k: "***REDACTED***" if k.lower() in self.sensitive_fields else self.sanitize_data(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self.sanitize_data(item) for item in data]
        else:
            return data
    
    async def log_operation(
        self,
        operation: str,
        user_id: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None,
        request_id: Optional[str] = None
    ):
        """Log a task execution operation for audit purposes."""
        try:
            audit_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id or str(uuid4()),
                "operation": operation,
                "user_id": user_id,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "success": success,
                "details": self.sanitize_data(details) if details else None,
                "error": error
            }
            
            self.logger.info("Audit log", **audit_entry)
            
        except Exception as e:
            logger.error("Audit logging failed", 
                        operation=operation,
                        user_id=user_id,
                        error=str(e))


audit_logger = TaskAuditLogger()


def audit_operation(operation: str, resource_type: str):
    """Decorator to automatically audit task operations."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request_id = str(uuid4())
            user_id = "unknown"
            resource_id = None
            success = True
            error = None
            details = None
            
            try:
                # Extract user information from dependencies
                if "user" in kwargs:
                    user_id = kwargs["user"].get("user_id", "unknown")
                
                # Extract resource ID from path parameters
                if "task_id" in kwargs:
                    resource_id = kwargs["task_id"]
                elif "workflow_id" in kwargs:
                    resource_id = kwargs["workflow_id"]
                
                # Extract request details
                if "request" in kwargs and hasattr(kwargs["request"], "dict"):
                    details = kwargs["request"].dict()
                
                # Execute the operation
                result = await func(*args, **kwargs)
                
                # Log successful operation
                await audit_logger.log_operation(
                    operation=operation,
                    user_id=user_id,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    details=details,
                    success=True,
                    request_id=request_id
                )
                
                return result
                
            except Exception as e:
                success = False
                error = str(e)
                
                # Log failed operation
                await audit_logger.log_operation(
                    operation=operation,
                    user_id=user_id,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    details=details,
                    success=False,
                    error=error,
                    request_id=request_id
                )
                
                raise
        
        return wrapper
    return decorator


# ===============================================================================
# PERFORMANCE MONITORING MIDDLEWARE
# ===============================================================================

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Performance monitoring middleware for task execution operations.
    
    Tracks response times, monitors performance targets, and provides
    metrics for optimization.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.performance_targets = {
            "task_creation": 200,  # 200ms target
            "task_status": 50,     # 50ms target
            "workflow_execution": 500,  # 500ms target
            "schedule_optimization": 2000  # 2 second target
        }
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = str(uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Determine operation type from path
            operation_type = self._determine_operation_type(request.url.path)
            target_time = self.performance_targets.get(operation_type, 1000)
            
            # Log performance metrics
            logger.info("Request completed",
                       request_id=request_id,
                       method=request.method,
                       path=request.url.path,
                       response_time_ms=response_time,
                       status_code=response.status_code,
                       operation_type=operation_type,
                       target_time_ms=target_time,
                       performance_met=response_time <= target_time)
            
            # Add performance headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = str(int(response_time))
            response.headers["X-Performance-Target"] = str(target_time)
            
            # Warn if performance target missed
            if response_time > target_time:
                logger.warning("Performance target missed",
                             request_id=request_id,
                             response_time_ms=response_time,
                             target_time_ms=target_time,
                             operation_type=operation_type)
            
            return response
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            logger.error("Request failed",
                        request_id=request_id,
                        method=request.method,
                        path=request.url.path,
                        response_time_ms=response_time,
                        error=str(e))
            
            # Return error response with request ID
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat()
                },
                headers={"X-Request-ID": request_id}
            )
    
    def _determine_operation_type(self, path: str) -> str:
        """Determine operation type from request path."""
        if "/tasks" in path and path.endswith("/tasks"):
            return "task_creation"
        elif "/status" in path:
            return "task_status" 
        elif "/workflows" in path and "/execute" in path:
            return "workflow_execution"
        elif "/scheduling" in path and "/optimize" in path:
            return "schedule_optimization"
        else:
            return "general"


# ===============================================================================
# CIRCUIT BREAKER PATTERNS
# ===============================================================================

class CircuitBreaker:
    """
    Circuit breaker for resilient task execution operations.
    
    Provides fail-fast behavior when downstream services are unavailable.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def is_available(self) -> bool:
        """Check if circuit breaker allows requests."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        elif self.state == "HALF_OPEN":
            return True
        
    def record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"
        
    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning("Circuit breaker opened due to failures",
                          failure_count=self.failure_count,
                          threshold=self.failure_threshold)


# Global circuit breakers for key services
orchestrator_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)
scheduling_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
workflow_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=45)


def with_circuit_breaker(circuit_breaker: CircuitBreaker):
    """Decorator to apply circuit breaker pattern."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not circuit_breaker.is_available():
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service temporarily unavailable"
                )
            
            try:
                result = await func(*args, **kwargs)
                circuit_breaker.record_success()
                return result
            except Exception as e:
                circuit_breaker.record_failure()
                raise
        
        return wrapper
    return decorator