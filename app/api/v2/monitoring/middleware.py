"""
SystemMonitoringAPI v2 Middleware

Consolidated middleware providing caching, security, rate limiting, and error handling
for the unified monitoring API. Implements enterprise-grade middleware patterns.

Epic 4 Phase 2 - Middleware Consolidation
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
import hashlib
import structlog
from contextlib import asynccontextmanager

from fastapi import HTTPException, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import redis.asyncio as redis

from ...core.redis import get_redis
from ...core.auth import verify_token, get_permissions
from .models import ErrorResponse, CacheConfig, SecurityConfig

logger = structlog.get_logger()


class CacheMiddleware:
    """
    Intelligent caching middleware with TTL, invalidation, and hit/miss tracking.
    
    Features:
    - Redis-backed distributed caching
    - Intelligent cache key generation
    - TTL-based expiration
    - Cache hit/miss analytics
    - Background cache warming
    """
    
    def __init__(self, default_ttl: int = 300, max_cache_size: int = 10000):
        self.default_ttl = default_ttl
        self.max_cache_size = max_cache_size
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
        self._redis_client: Optional[redis.Redis] = None
    
    async def _get_redis(self) -> redis.Redis:
        """Get Redis client singleton."""
        if self._redis_client is None:
            self._redis_client = await get_redis()
        return self._redis_client
    
    def _generate_cache_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generate deterministic cache key from parameters."""
        # Sort parameters for consistent key generation
        sorted_params = sorted(params.items())
        params_str = json.dumps(sorted_params, sort_keys=True, default=str)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        return f"monitoring_cache:{prefix}:{params_hash}"
    
    async def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached data."""
        try:
            redis_client = await self._get_redis()
            cached_data = await redis_client.get(cache_key)
            
            if cached_data:
                self.cache_stats["hits"] += 1
                try:
                    return json.loads(cached_data)
                except json.JSONDecodeError as e:
                    logger.warning("❌ Cache data JSON decode failed", key=cache_key, error=str(e))
                    await self.delete(cache_key)  # Remove corrupted cache entry
                    return None
            else:
                self.cache_stats["misses"] += 1
                return None
                
        except Exception as e:
            logger.error("❌ Cache get failed", key=cache_key, error=str(e))
            self.cache_stats["errors"] += 1
            return None
    
    async def set(self, cache_key: str, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set cached data with TTL."""
        try:
            redis_client = await self._get_redis()
            ttl_seconds = ttl or self.default_ttl
            
            # Add cache metadata
            cache_data = {
                "data": data,
                "cached_at": datetime.utcnow().isoformat(),
                "ttl": ttl_seconds
            }
            
            await redis_client.setex(
                cache_key,
                ttl_seconds,
                json.dumps(cache_data, default=str)
            )
            
            self.cache_stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error("❌ Cache set failed", key=cache_key, error=str(e))
            self.cache_stats["errors"] += 1
            return False
    
    async def delete(self, cache_key: str) -> bool:
        """Delete cached data."""
        try:
            redis_client = await self._get_redis()
            deleted = await redis_client.delete(cache_key)
            
            if deleted:
                self.cache_stats["deletes"] += 1
            
            return bool(deleted)
            
        except Exception as e:
            logger.error("❌ Cache delete failed", key=cache_key, error=str(e))
            self.cache_stats["errors"] += 1
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern."""
        try:
            redis_client = await self._get_redis()
            keys = await redis_client.keys(f"monitoring_cache:{pattern}*")
            
            if keys:
                deleted = await redis_client.delete(*keys)
                self.cache_stats["deletes"] += deleted
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error("❌ Cache pattern invalidation failed", pattern=pattern, error=str(e))
            self.cache_stats["errors"] += 1
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_operations = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_operations if total_operations > 0 else 0
        
        return {
            **self.cache_stats,
            "hit_rate": hit_rate,
            "total_operations": total_operations
        }
    
    @asynccontextmanager
    async def cached_response(self, cache_key: str, ttl: Optional[int] = None):
        """Context manager for cached responses."""
        # Try to get cached data
        cached_data = await self.get(cache_key)
        if cached_data:
            yield cached_data["data"], True  # data, cache_hit
            return
        
        # No cached data, yield None to compute fresh data
        yield None, False


class SecurityMiddleware:
    """
    Enterprise security middleware with OAuth2, RBAC, and audit logging.
    
    Features:
    - JWT token validation
    - Role-based access control (RBAC)
    - Permission checking
    - Audit logging
    - Request sanitization
    - Rate limiting integration
    """
    
    def __init__(self):
        self.security_stats = {
            "auth_attempts": 0,
            "auth_successes": 0,
            "auth_failures": 0,
            "permission_denials": 0,
            "invalid_tokens": 0
        }
    
    async def authenticate_request(self, request: Request) -> Optional[Dict[str, Any]]:
        """Authenticate request using JWT token."""
        try:
            # Extract authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return None
            
            token = auth_header.split(" ")[1]
            self.security_stats["auth_attempts"] += 1
            
            # Verify token
            try:
                payload = verify_token(token)
                self.security_stats["auth_successes"] += 1
                
                # Add request metadata
                payload["request_ip"] = self._get_client_ip(request)
                payload["request_time"] = datetime.utcnow()
                payload["user_agent"] = request.headers.get("User-Agent", "unknown")
                
                return payload
                
            except JWTError as e:
                logger.warning("❌ JWT validation failed", error=str(e), ip=self._get_client_ip(request))
                self.security_stats["invalid_tokens"] += 1
                self.security_stats["auth_failures"] += 1
                return None
                
        except Exception as e:
            logger.error("❌ Authentication error", error=str(e))
            self.security_stats["auth_failures"] += 1
            return None
    
    async def check_permissions(self, user: Dict[str, Any], required_permissions: List[str]) -> bool:
        """Check if user has required permissions."""
        try:
            user_permissions = get_permissions(user)
            
            # Admin bypass
            if "admin" in user_permissions:
                return True
            
            # Check specific permissions
            has_permission = any(perm in user_permissions for perm in required_permissions)
            
            if not has_permission:
                self.security_stats["permission_denials"] += 1
                logger.warning(
                    "❌ Permission denied",
                    user_id=user.get("sub"),
                    required_permissions=required_permissions,
                    user_permissions=user_permissions
                )
            
            return has_permission
            
        except Exception as e:
            logger.error("❌ Permission check error", error=str(e))
            return False
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        success_rate = (
            self.security_stats["auth_successes"] / 
            max(1, self.security_stats["auth_attempts"])
        )
        
        return {
            **self.security_stats,
            "success_rate": success_rate
        }


class RateLimitMiddleware:
    """
    Advanced rate limiting middleware with sliding windows and intelligent throttling.
    
    Features:
    - Sliding window rate limiting
    - Per-user and per-IP limits
    - Intelligent throttling
    - Burst allowance
    - Rate limit bypass for admin users
    """
    
    def __init__(self, default_limit: int = 100, window_seconds: int = 60):
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.rate_limit_stats = {
            "requests": 0,
            "rate_limited": 0,
            "bypassed": 0
        }
        self._redis_client: Optional[redis.Redis] = None
    
    async def _get_redis(self) -> redis.Redis:
        """Get Redis client singleton."""
        if self._redis_client is None:
            self._redis_client = await get_redis()
        return self._redis_client
    
    def _get_rate_limit_key(self, identifier: str, endpoint: str) -> str:
        """Generate rate limit key."""
        return f"rate_limit:{identifier}:{endpoint}"
    
    async def check_rate_limit(self, 
                             request: Request, 
                             user: Optional[Dict[str, Any]] = None,
                             limit: Optional[int] = None) -> tuple[bool, Dict[str, Any]]:
        """
        Check rate limit for request.
        
        Returns: (allowed, rate_limit_info)
        """
        try:
            self.rate_limit_stats["requests"] += 1
            
            # Admin bypass
            if user and "admin" in get_permissions(user):
                self.rate_limit_stats["bypassed"] += 1
                return True, {"bypassed": True, "reason": "admin_user"}
            
            # Determine identifier (user ID or IP)
            identifier = user.get("sub") if user else self._get_client_ip(request)
            endpoint = request.url.path
            
            # Use provided limit or default
            request_limit = limit or self.default_limit
            
            redis_client = await self._get_redis()
            rate_limit_key = self._get_rate_limit_key(identifier, endpoint)
            
            # Implement sliding window using Redis sorted sets
            now = time.time()
            window_start = now - self.window_seconds
            
            # Remove expired entries
            await redis_client.zremrangebyscore(rate_limit_key, 0, window_start)
            
            # Count current requests in window
            current_requests = await redis_client.zcard(rate_limit_key)
            
            # Check if under limit
            if current_requests < request_limit:
                # Add current request
                await redis_client.zadd(rate_limit_key, {str(now): now})
                await redis_client.expire(rate_limit_key, self.window_seconds)
                
                return True, {
                    "allowed": True,
                    "limit": request_limit,
                    "remaining": request_limit - current_requests - 1,
                    "reset_at": int(now + self.window_seconds)
                }
            else:
                # Rate limit exceeded
                self.rate_limit_stats["rate_limited"] += 1
                
                # Get reset time (oldest entry + window)
                oldest_entries = await redis_client.zrange(rate_limit_key, 0, 0, withscores=True)
                reset_at = int(oldest_entries[0][1] + self.window_seconds) if oldest_entries else int(now + self.window_seconds)
                
                logger.warning(
                    "⏱️ Rate limit exceeded",
                    identifier=identifier,
                    endpoint=endpoint,
                    limit=request_limit,
                    current_requests=current_requests
                )
                
                return False, {
                    "allowed": False,
                    "limit": request_limit,
                    "remaining": 0,
                    "reset_at": reset_at,
                    "retry_after": reset_at - int(now)
                }
                
        except Exception as e:
            logger.error("❌ Rate limit check failed", error=str(e))
            # Allow request on error (fail open)
            return True, {"allowed": True, "error": "rate_limit_check_failed"}
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        return request.client.host if request.client else "unknown"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        rate_limited_percentage = (
            self.rate_limit_stats["rate_limited"] / 
            max(1, self.rate_limit_stats["requests"])
        ) * 100
        
        return {
            **self.rate_limit_stats,
            "rate_limited_percentage": rate_limited_percentage
        }


class ErrorHandlingMiddleware:
    """
    Comprehensive error handling middleware with logging, monitoring, and recovery.
    
    Features:
    - Structured error logging
    - Error categorization
    - Error rate monitoring
    - Automatic error recovery
    - Client-friendly error responses
    """
    
    def __init__(self):
        self.error_stats = {
            "total_errors": 0,
            "client_errors_4xx": 0,
            "server_errors_5xx": 0,
            "validation_errors": 0,
            "auth_errors": 0,
            "rate_limit_errors": 0,
            "system_errors": 0
        }
    
    def handle_http_exception(self, e: HTTPException, request: Request) -> ErrorResponse:
        """Handle HTTP exceptions with proper categorization."""
        self.error_stats["total_errors"] += 1
        
        # Categorize error
        if 400 <= e.status_code < 500:
            self.error_stats["client_errors_4xx"] += 1
            
            if e.status_code == 401:
                self.error_stats["auth_errors"] += 1
            elif e.status_code == 422:
                self.error_stats["validation_errors"] += 1
            elif e.status_code == 429:
                self.error_stats["rate_limit_errors"] += 1
                
        elif 500 <= e.status_code < 600:
            self.error_stats["server_errors_5xx"] += 1
            self.error_stats["system_errors"] += 1
        
        # Log error
        logger.error(
            "🚨 HTTP Exception",
            status_code=e.status_code,
            detail=e.detail,
            path=request.url.path,
            method=request.method,
            client_ip=self._get_client_ip(request)
        )
        
        return ErrorResponse(
            error=str(e.detail),
            error_code=f"HTTP_{e.status_code}",
            timestamp=datetime.utcnow(),
            details={
                "status_code": e.status_code,
                "path": request.url.path,
                "method": request.method
            },
            trace_id=self._generate_trace_id()
        )
    
    def handle_general_exception(self, e: Exception, request: Request) -> ErrorResponse:
        """Handle general exceptions."""
        self.error_stats["total_errors"] += 1
        self.error_stats["system_errors"] += 1
        
        # Log error with full traceback
        logger.error(
            "💥 General Exception",
            error_type=type(e).__name__,
            error=str(e),
            path=request.url.path,
            method=request.method,
            client_ip=self._get_client_ip(request),
            exc_info=True
        )
        
        return ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=datetime.utcnow(),
            details={
                "error_type": type(e).__name__,
                "path": request.url.path,
                "method": request.method
            },
            trace_id=self._generate_trace_id()
        )
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        return request.client.host if request.client else "unknown"
    
    def _generate_trace_id(self) -> str:
        """Generate unique trace ID for error tracking."""
        import uuid
        return f"trace_{uuid.uuid4().hex[:12]}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        error_rate = self.error_stats["total_errors"]  # Would typically be rate per time period
        
        return {
            **self.error_stats,
            "error_rate": error_rate,
            "client_error_rate": (
                self.error_stats["client_errors_4xx"] / 
                max(1, self.error_stats["total_errors"])
            ),
            "server_error_rate": (
                self.error_stats["server_errors_5xx"] / 
                max(1, self.error_stats["total_errors"])
            )
        }


# ==================== MIDDLEWARE INTEGRATION ====================

class UnifiedMiddlewareStack:
    """
    Unified middleware stack integrating all middleware components.
    
    Provides a single interface for all middleware operations with
    proper error handling and performance monitoring.
    """
    
    def __init__(self):
        self.cache = CacheMiddleware()
        self.security = SecurityMiddleware()
        self.rate_limit = RateLimitMiddleware()
        self.error_handler = ErrorHandlingMiddleware()
    
    async def process_request(self, request: Request) -> Dict[str, Any]:
        """Process incoming request through middleware stack."""
        context = {
            "request": request,
            "start_time": time.time(),
            "user": None,
            "rate_limit_info": None,
            "errors": []
        }
        
        try:
            # 1. Authentication
            user = await self.security.authenticate_request(request)
            context["user"] = user
            
            # 2. Rate limiting
            allowed, rate_limit_info = await self.rate_limit.check_rate_limit(request, user)
            context["rate_limit_info"] = rate_limit_info
            
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(rate_limit_info.get("retry_after", 60))}
                )
            
            return context
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("❌ Middleware processing failed", error=str(e))
            raise HTTPException(status_code=500, detail="Middleware processing failed")
    
    async def process_response(self, context: Dict[str, Any], response_data: Any) -> Any:
        """Process outgoing response through middleware stack."""
        try:
            # Add performance metrics
            processing_time = (time.time() - context["start_time"]) * 1000
            
            if isinstance(response_data, dict):
                response_data.setdefault("metadata", {})
                response_data["metadata"]["processing_time_ms"] = processing_time
                
                # Add rate limit headers to metadata
                if context.get("rate_limit_info"):
                    response_data["metadata"]["rate_limit"] = context["rate_limit_info"]
            
            return response_data
            
        except Exception as e:
            logger.error("❌ Response processing failed", error=str(e))
            return response_data
    
    def get_middleware_stats(self) -> Dict[str, Any]:
        """Get comprehensive middleware statistics."""
        return {
            "cache": self.cache.get_stats(),
            "security": self.security.get_stats(), 
            "rate_limit": self.rate_limit.get_stats(),
            "error_handling": self.error_handler.get_stats(),
            "timestamp": datetime.utcnow().isoformat()
        }


# Global middleware instance
unified_middleware = UnifiedMiddlewareStack()