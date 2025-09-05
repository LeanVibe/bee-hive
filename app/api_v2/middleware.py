"""
Unified middleware for LeanVibe Agent Hive 2.0 API

Provides consistent authentication, error handling, and performance
monitoring across all consolidated API endpoints.
"""

import time
import uuid
from typing import Optional, Dict, Any
from datetime import datetime

import structlog
from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..services.user_service import get_user_service
from ..models.user import UserRole
from ..core.database import get_session

logger = structlog.get_logger()
security = HTTPBearer(auto_error=False)

# Performance metrics
PERFORMANCE_TARGETS = {
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

class PerformanceMiddleware(BaseHTTPMiddleware):
    """Monitors and optimizes API performance to meet sub-100ms targets."""
    
    async def dispatch(self, request: Request, call_next):
        """Track request performance and optimize response times."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request tracking
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        # Process request
        response = await call_next(request)
        
        # Calculate performance metrics
        duration_ms = (time.time() - start_time) * 1000
        
        # Extract resource type from path
        path_parts = request.url.path.split("/")
        resource_type = "unknown"
        if len(path_parts) >= 4 and path_parts[2] == "v2":
            resource_type = path_parts[3]
        
        # Check performance targets
        target_ms = PERFORMANCE_TARGETS.get(resource_type, 100)
        if duration_ms > target_ms:
            logger.warning(
                "performance_threshold_exceeded",
                request_id=request_id,
                resource=resource_type,
                duration_ms=duration_ms,
                target_ms=target_ms,
                path=request.url.path
            )
        
        # Add performance headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        response.headers["X-Performance-Target"] = f"{target_ms}ms"
        
        # Log performance metrics
        logger.info(
            "api_request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            duration_ms=duration_ms,
            status_code=response.status_code,
            resource=resource_type
        )
        
        return response

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """Unified authentication middleware for all API endpoints."""
    
    # Public endpoints that don't require authentication
    PUBLIC_ENDPOINTS = {
        "/api/v2/",
        "/api/v2/health/status",
        "/api/v2/health/ready", 
        "/api/v2/health",
        "/api/v2/auth/register",
        "/api/v2/auth/login",
        "/api/v2/auth/refresh",
        "/api/v2/auth/health",
        "/api/v2/security/login",
        "/api/v2/security/refresh",
        "/docs",
        "/redoc",
        "/openapi.json"
    }
    
    async def dispatch(self, request: Request, call_next):
        """Authenticate requests and inject user context."""
        # Skip authentication for public endpoints
        if request.url.path in self.PUBLIC_ENDPOINTS:
            return await call_next(request)
        
        # Extract and validate token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "authentication_required",
                    "message": "Bearer token required for this endpoint",
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
        
        token = auth_header.split(" ")[1]
        
        try:
            # Decode and validate JWT token
            user_service = get_user_service()
            payload = user_service.verify_token(token)
            
            if not payload:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token"
                )
            
            user_id = payload.get("sub")
            if not user_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: missing user ID"
                )
            
            # Get user from database
            async with get_session() as db:
                user = await user_service.get_user_by_id(db, user_id)
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="User not found"
                    )
            
            # Inject user context into request
            request.state.current_user = user
            request.state.user_id = user_id
            request.state.user_roles = user.roles or []
            request.state.permissions = user.permissions or []
            
            # Log authentication success
            logger.debug(
                "authentication_success",
                user_id=user_id,
                roles=user.roles,
                request_id=getattr(request.state, "request_id", None)
            )
            
        except Exception as e:
            logger.warning(
                "authentication_failed",
                error=str(e),
                request_id=getattr(request.state, "request_id", None)
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "error": "authentication_failed",
                    "message": "Invalid or expired token",
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
        
        return await call_next(request)

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Unified error handling and response formatting."""
    
    async def dispatch(self, request: Request, call_next):
        """Handle and format all API errors consistently."""
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": "http_exception",
                    "message": e.detail,
                    "status_code": e.status_code,
                    "request_id": getattr(request.state, "request_id", None),
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": request.url.path
                }
            )
            
        except Exception as e:
            # Handle unexpected errors
            request_id = getattr(request.state, "request_id", None)
            
            logger.error(
                "unexpected_api_error",
                error=str(e),
                error_type=type(e).__name__,
                request_id=request_id,
                path=request.url.path,
                method=request.method
            )
            
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "internal_server_error",
                    "message": "An unexpected error occurred",
                    "request_id": request_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "path": request.url.path
                }
            )

# Middleware instances
performance_middleware = PerformanceMiddleware
auth_middleware = AuthenticationMiddleware
error_middleware = ErrorHandlingMiddleware

# Utility functions for route handlers
def require_permission(permission: str):
    """Decorator to require specific permission for endpoint access."""
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            user_permissions = getattr(request.state, "permissions", [])
            if permission not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required"
                )
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

def require_role(role: str):
    """Decorator to require specific role for endpoint access."""
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            user_roles = getattr(request.state, "user_roles", [])
            if isinstance(role, UserRole):
                role = role.value
            if role not in user_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role '{role}' required"
                )
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

def get_current_user_from_request(request: Request):
    """Get current user from request state."""
    return getattr(request.state, "current_user", None)

def get_user_id_from_request(request: Request):
    """Get user ID from request state."""
    return getattr(request.state, "user_id", None)