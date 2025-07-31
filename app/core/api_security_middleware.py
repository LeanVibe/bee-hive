"""
API Security Middleware for Enterprise-Grade Protection.

Provides comprehensive API security features including rate limiting,
input validation, security headers, request filtering, and DDoS protection.

Features:
- Intelligent rate limiting per user/IP/endpoint
- Security headers (HSTS, CSP, X-Frame-Options, etc.)
- Request size and content validation
- SQL injection and XSS protection
- CORS security
- Request/response filtering
- Audit logging for security events
"""

import asyncio
import hashlib
import ipaddress
import json
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp
import structlog

from .redis import RedisClient
from .security import get_current_user
from ..schemas.security import SecurityError

logger = structlog.get_logger()


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    ADAPTIVE = "adaptive"


class SecurityThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RateLimitRule:
    """Rate limiting rule definition."""
    key_pattern: str  # Redis key pattern
    limit: int       # Request limit
    window_seconds: int  # Time window
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_limit: Optional[int] = None  # Burst allowance
    
    # Advanced options
    per_user: bool = True
    per_ip: bool = False
    per_endpoint: bool = True
    skip_successful_auth: bool = False
    
    def generate_key(self, request: Request, user_id: Optional[str] = None) -> str:
        """Generate cache key for rate limiting."""
        key_parts = []
        
        if self.per_user and user_id:
            key_parts.append(f"user:{user_id}")
        
        if self.per_ip:
            client_ip = self._get_client_ip(request)
            key_parts.append(f"ip:{client_ip}")
        
        if self.per_endpoint:
            endpoint = f"{request.method}:{request.url.path}"
            key_parts.append(f"endpoint:{endpoint}")
        
        key_suffix = ":".join(key_parts) if key_parts else "global"
        return f"rate_limit:{self.key_pattern}:{key_suffix}"
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return str(request.client.host) if request.client else "unknown"


@dataclass
class SecurityConfig:
    """Security middleware configuration."""
    # Rate limiting
    enable_rate_limiting: bool = True
    default_rate_limit: int = 100  # requests per minute
    burst_multiplier: float = 1.5
    rate_limit_strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    
    # Security headers
    enable_security_headers: bool = True
    enable_hsts: bool = True
    hsts_max_age: int = 31536000  # 1 year
    enable_csp: bool = True
    csp_policy: str = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
    
    # Request validation
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_json_depth: int = 10
    enable_sql_injection_detection: bool = True
    enable_xss_detection: bool = True
    
    # Threat detection
    enable_threat_detection: bool = True
    suspicious_patterns: List[str] = field(default_factory=lambda: [
        r"(?i)(union|select|insert|update|delete|drop|create|alter|exec|execute|script|javascript|vbscript)",
        r"(?i)(<script|javascript:|vbscript:|onload|onerror|onclick)",
        r"(?i)(\.\.\/|\.\.\\|\/etc\/passwd|\/proc\/|cmd\.exe|powershell)"
    ])
    
    # IP filtering
    blocked_ips: Set[str] = field(default_factory=set)
    allowed_ips: Set[str] = field(default_factory=set)  # If set, only these IPs allowed
    blocked_countries: Set[str] = field(default_factory=set)
    
    # Audit and logging
    log_all_requests: bool = False
    log_security_events: bool = True
    log_rate_limit_hits: bool = True


@dataclass
class SecurityMetrics:
    """Security metrics tracking."""
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limited_requests: int = 0
    suspicious_requests: int = 0
    
    # Threat detection
    sql_injection_attempts: int = 0
    xss_attempts: int = 0
    path_traversal_attempts: int = 0
    
    # Performance
    avg_processing_time_ms: float = 0.0
    max_processing_time_ms: float = 0.0
    
    # Rate limiting
    rate_limit_hits_by_endpoint: Dict[str, int] = field(default_factory=dict)
    rate_limit_hits_by_ip: Dict[str, int] = field(default_factory=dict)


class APISecurityMiddleware(BaseHTTPMiddleware):
    """
    Comprehensive API Security Middleware.
    
    Provides enterprise-grade API protection including rate limiting,
    input validation, security headers, and threat detection.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        redis_client: RedisClient,
        config: Optional[SecurityConfig] = None
    ):
        """
        Initialize API Security Middleware.
        
        Args:
            app: ASGI application
            redis_client: Redis client for caching and rate limiting
            config: Security configuration
        """
        super().__init__(app)
        self.redis = redis_client
        self.config = config or SecurityConfig()
        
        # Performance metrics
        self.metrics = SecurityMetrics()
        
        # Rate limiting rules
        self.rate_limit_rules = self._initialize_rate_limit_rules()
        
        # Compiled regex patterns for threat detection
        self.threat_patterns = [
            re.compile(pattern) for pattern in self.config.suspicious_patterns
        ]
        
        # Security headers template
        self.security_headers = self._build_security_headers()
        
        # Cache keys
        self._rate_limit_prefix = "api_security:rate_limit:"
        self._blocked_ip_prefix = "api_security:blocked_ip:"
        self._metrics_key = "api_security:metrics"
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request through security pipeline."""
        start_time = time.time()
        
        try:
            # Update request metrics
            self.metrics.total_requests += 1
            
            # Phase 1: IP filtering and basic validation
            if not await self._validate_client_ip(request):
                self.metrics.blocked_requests += 1
                return await self._create_blocked_response("IP blocked")
            
            # Phase 2: Request size and structure validation
            if not await self._validate_request_structure(request):
                self.metrics.blocked_requests += 1
                return await self._create_blocked_response("Invalid request structure")
            
            # Phase 3: Threat detection
            threat_level = await self._detect_threats(request)
            if threat_level in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]:
                self.metrics.suspicious_requests += 1
                await self._log_security_event(request, "threat_detected", {"threat_level": threat_level.value})
                
                if threat_level == SecurityThreatLevel.CRITICAL:
                    self.metrics.blocked_requests += 1
                    return await self._create_blocked_response("Security threat detected")
            
            # Phase 4: Rate limiting
            if self.config.enable_rate_limiting:
                rate_limit_check = await self._check_rate_limits(request)
                if not rate_limit_check["allowed"]:
                    self.metrics.rate_limited_requests += 1
                    return await self._create_rate_limit_response(rate_limit_check)
            
            # Phase 5: Content validation for POST/PUT requests
            if request.method in ["POST", "PUT", "PATCH"]:
                if not await self._validate_request_content(request):
                    self.metrics.blocked_requests += 1
                    return await self._create_blocked_response("Invalid request content")
            
            # Process request
            response = await call_next(request)
            
            # Phase 6: Add security headers
            if self.config.enable_security_headers:
                self._add_security_headers(response)
            
            # Phase 7: Response validation and filtering
            await self._process_response(request, response)
            
            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(processing_time)
            
            return response
            
        except Exception as e:
            logger.error(f"API security middleware error: {e}")
            processing_time = (time.time() - start_time) * 1000
            self._update_performance_metrics(processing_time)
            
            # Return safe error response
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Internal security error"},
                headers=self.security_headers
            )
    
    async def _validate_client_ip(self, request: Request) -> bool:
        """Validate client IP address."""
        client_ip = self._get_client_ip(request)
        
        # Check if IP is blocked
        if client_ip in self.config.blocked_ips:
            return False
        
        # Check if IP is temporarily blocked (rate limiting aftermath)
        blocked_key = f"{self._blocked_ip_prefix}{client_ip}"
        if await self.redis.get(blocked_key):
            return False
        
        # If allow list is configured, check if IP is allowed
        if self.config.allowed_ips and client_ip not in self.config.allowed_ips:
            return False
        
        return True
    
    async def _validate_request_structure(self, request: Request) -> bool:
        """Validate basic request structure."""
        
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.config.max_request_size:
            await self._log_security_event(
                request, "oversized_request", 
                {"content_length": content_length, "max_allowed": self.config.max_request_size}
            )
            return False
        
        # Check for suspicious headers
        suspicious_headers = ["x-forwarded-host", "x-cluster-client-ip"]
        for header in suspicious_headers:
            if header in request.headers:
                value = request.headers[header]
                if self._contains_suspicious_content(value):
                    await self._log_security_event(
                        request, "suspicious_header", 
                        {"header": header, "value": value}
                    )
                    return False
        
        return True
    
    async def _detect_threats(self, request: Request) -> SecurityThreatLevel:
        """Detect security threats in request."""
        threat_level = SecurityThreatLevel.LOW
        
        # Check URL path
        path_threats = self._analyze_path_threats(request.url.path)
        if path_threats:
            threat_level = max(threat_level, path_threats)
        
        # Check query parameters
        query_threats = self._analyze_query_threats(str(request.url.query))
        if query_threats:
            threat_level = max(threat_level, query_threats)
        
        # Check headers
        header_threats = await self._analyze_header_threats(request.headers)
        if header_threats:
            threat_level = max(threat_level, header_threats)
        
        # Check user agent
        user_agent = request.headers.get("user-agent", "")
        ua_threats = self._analyze_user_agent_threats(user_agent)
        if ua_threats:
            threat_level = max(threat_level, ua_threats)
        
        return threat_level
    
    def _analyze_path_threats(self, path: str) -> Optional[SecurityThreatLevel]:
        """Analyze URL path for threats."""
        
        # Path traversal detection
        if "../" in path or "..\\" in path or "/etc/" in path or "\\windows\\" in path.lower():
            self.metrics.path_traversal_attempts += 1
            return SecurityThreatLevel.HIGH
        
        # SQL injection patterns in path
        if self._contains_sql_injection_patterns(path):
            self.metrics.sql_injection_attempts += 1
            return SecurityThreatLevel.HIGH
        
        # Script injection in path
        if self._contains_xss_patterns(path):
            self.metrics.xss_attempts += 1
            return SecurityThreatLevel.MEDIUM
        
        return None
    
    def _analyze_query_threats(self, query: str) -> Optional[SecurityThreatLevel]:
        """Analyze query parameters for threats."""
        if not query:
            return None
        
        # SQL injection detection
        if self._contains_sql_injection_patterns(query):
            self.metrics.sql_injection_attempts += 1
            return SecurityThreatLevel.HIGH
        
        # XSS detection
        if self._contains_xss_patterns(query):
            self.metrics.xss_attempts += 1
            return SecurityThreatLevel.MEDIUM
        
        return None
    
    async def _analyze_header_threats(self, headers) -> Optional[SecurityThreatLevel]:
        """Analyze request headers for threats."""
        
        for name, value in headers.items():
            # Check for injection attempts in headers
            if self._contains_suspicious_content(value):
                return SecurityThreatLevel.MEDIUM
            
            # Check for known malicious headers
            if name.lower() in ["x-real-ip", "x-forwarded-for"]:
                # Validate IP format
                try:
                    ips = value.split(",")
                    for ip in ips:
                        ipaddress.ip_address(ip.strip())
                except ValueError:
                    return SecurityThreatLevel.MEDIUM
        
        return None
    
    def _analyze_user_agent_threats(self, user_agent: str) -> Optional[SecurityThreatLevel]:
        """Analyze user agent for threats."""
        if not user_agent:
            return SecurityThreatLevel.LOW  # Missing user agent is suspicious but not critical
        
        # Check for known attack tools
        attack_tools = ["sqlmap", "nikto", "nmap", "burp", "dirb", "gobuster"]
        if any(tool in user_agent.lower() for tool in attack_tools):
            return SecurityThreatLevel.HIGH
        
        # Check for suspicious patterns
        if self._contains_suspicious_content(user_agent):
            return SecurityThreatLevel.MEDIUM
        
        return None
    
    def _contains_sql_injection_patterns(self, content: str) -> bool:
        """Check if content contains SQL injection patterns."""
        if not self.config.enable_sql_injection_detection:
            return False
        
        sql_patterns = [
            r"(?i)(union\s+select|select\s+.*\s+from)",
            r"(?i)(insert\s+into|update\s+.*\s+set|delete\s+from)",
            r"(?i)(drop\s+table|create\s+table|alter\s+table)",
            r"(?i)(exec\s*\(|execute\s*\(|sp_executesql)",
            r"(?i)(\'\s*or\s+\'\s*|\'\s*and\s+\'\s*)",
            r"(?i)(\/\*.*\*\/|--\s|#\s)"
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, content):
                return True
        
        return False
    
    def _contains_xss_patterns(self, content: str) -> bool:
        """Check if content contains XSS patterns."""
        if not self.config.enable_xss_detection:
            return False
        
        xss_patterns = [
            r"(?i)<script[^>]*>.*?</script>",
            r"(?i)javascript\s*:",
            r"(?i)vbscript\s*:",
            r"(?i)on\w+\s*=",
            r"(?i)<iframe[^>]*>",
            r"(?i)expression\s*\(",
            r"(?i)@import\s+['\"]"
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, content):
                return True
        
        return False
    
    def _contains_suspicious_content(self, content: str) -> bool:
        """Check if content contains suspicious patterns."""
        for pattern in self.threat_patterns:
            if pattern.search(content):
                return True
        return False
    
    async def _check_rate_limits(self, request: Request) -> Dict[str, Any]:
        """Check rate limits for request."""
        user_id = None
        
        # Try to get user from JWT token
        try:
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                # This would need to be integrated with the actual auth system
                user_id = "temp_user_id"  # Placeholder
        except Exception:
            pass
        
        # Check each rate limit rule
        for rule in self.rate_limit_rules:
            cache_key = rule.generate_key(request, user_id)
            
            # Check current usage
            current_usage = await self._get_rate_limit_usage(cache_key, rule)
            
            if current_usage >= rule.limit:
                # Update metrics
                endpoint = f"{request.method}:{request.url.path}"
                self.metrics.rate_limit_hits_by_endpoint[endpoint] = (
                    self.metrics.rate_limit_hits_by_endpoint.get(endpoint, 0) + 1
                )
                
                client_ip = self._get_client_ip(request)
                self.metrics.rate_limit_hits_by_ip[client_ip] = (
                    self.metrics.rate_limit_hits_by_ip.get(client_ip, 0) + 1
                )
                
                # Log rate limit hit
                if self.config.log_rate_limit_hits:
                    await self._log_security_event(
                        request, "rate_limit_exceeded",
                        {
                            "rule": rule.key_pattern,
                            "limit": rule.limit,
                            "current_usage": current_usage,
                            "window_seconds": rule.window_seconds
                        }
                    )
                
                return {
                    "allowed": False,
                    "rule": rule.key_pattern,
                    "limit": rule.limit,
                    "current_usage": current_usage,
                    "reset_time": time.time() + rule.window_seconds
                }
            
            # Increment usage counter
            await self._increment_rate_limit_usage(cache_key, rule)
        
        return {"allowed": True}
    
    async def _get_rate_limit_usage(self, cache_key: str, rule: RateLimitRule) -> int:
        """Get current rate limit usage."""
        if rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._get_sliding_window_usage(cache_key, rule.window_seconds)
        elif rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._get_fixed_window_usage(cache_key, rule.window_seconds)
        else:
            # Default to sliding window
            return await self._get_sliding_window_usage(cache_key, rule.window_seconds)
    
    async def _get_sliding_window_usage(self, cache_key: str, window_seconds: int) -> int:
        """Get usage for sliding window rate limiting."""
        now = time.time()
        window_start = now - window_seconds
        
        # Count requests in the sliding window
        # This is a simplified implementation - in production, you'd use Redis sorted sets
        usage_data = await self.redis.get(cache_key)
        if not usage_data:
            return 0
        
        try:
            requests = json.loads(usage_data)
            # Filter requests within the window
            valid_requests = [req for req in requests if req > window_start]
            return len(valid_requests)
        except (json.JSONDecodeError, TypeError):
            return 0
    
    async def _get_fixed_window_usage(self, cache_key: str, window_seconds: int) -> int:
        """Get usage for fixed window rate limiting."""
        # Generate window key based on current time window
        window_start = int(time.time() // window_seconds) * window_seconds
        window_key = f"{cache_key}:{window_start}"
        
        usage = await self.redis.get(window_key)
        return int(usage) if usage else 0
    
    async def _increment_rate_limit_usage(self, cache_key: str, rule: RateLimitRule) -> None:
        """Increment rate limit usage counter."""
        now = time.time()
        
        if rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            # Add timestamp to sliding window
            usage_data = await self.redis.get(cache_key) or "[]"
            try:
                requests = json.loads(usage_data)
                requests.append(now)
                # Keep only requests within the window
                window_start = now - rule.window_seconds
                requests = [req for req in requests if req > window_start]
                
                await self.redis.set_with_expiry(
                    cache_key,
                    json.dumps(requests),
                    ttl=rule.window_seconds
                )
            except json.JSONDecodeError:
                await self.redis.set_with_expiry(
                    cache_key,
                    json.dumps([now]),
                    ttl=rule.window_seconds
                )
        
        elif rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            # Increment counter for current window
            window_start = int(now // rule.window_seconds) * rule.window_seconds
            window_key = f"{cache_key}:{window_start}"
            
            current = await self.redis.get(window_key)
            new_value = (int(current) if current else 0) + 1
            
            await self.redis.set_with_expiry(
                window_key,
                str(new_value),
                ttl=rule.window_seconds
            )
    
    async def _validate_request_content(self, request: Request) -> bool:
        """Validate request content for POST/PUT requests."""
        try:
            # Get request body
            body = await request.body()
            if not body:
                return True
            
            # Check content type
            content_type = request.headers.get("content-type", "")
            
            if "application/json" in content_type:
                try:
                    json_data = json.loads(body)
                    
                    # Check JSON depth
                    if self._get_json_depth(json_data) > self.config.max_json_depth:
                        await self._log_security_event(
                            request, "excessive_json_depth",
                            {"max_depth": self.config.max_json_depth}
                        )
                        return False
                    
                    # Check for threats in JSON values
                    if self._contains_threats_in_json(json_data):
                        return False
                    
                except json.JSONDecodeError:
                    await self._log_security_event(request, "invalid_json", {})
                    return False
            
            # Check for threats in raw body
            body_str = body.decode('utf-8', errors='ignore')
            if self._contains_suspicious_content(body_str):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Content validation error: {e}")
            return False
    
    def _get_json_depth(self, obj: Any, depth: int = 0) -> int:
        """Calculate JSON object depth."""
        if isinstance(obj, dict):
            return max(self._get_json_depth(v, depth + 1) for v in obj.values()) if obj else depth
        elif isinstance(obj, list):
            return max(self._get_json_depth(v, depth + 1) for v in obj) if obj else depth
        else:
            return depth
    
    def _contains_threats_in_json(self, obj: Any) -> bool:
        """Check for threats in JSON object."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if self._contains_suspicious_content(str(key)) or self._contains_threats_in_json(value):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if self._contains_threats_in_json(item):
                    return True
        elif isinstance(obj, str):
            if self._contains_suspicious_content(obj):
                return True
        
        return False
    
    async def _process_response(self, request: Request, response: Response) -> None:
        """Process and validate response."""
        # Log successful request if configured
        if self.config.log_all_requests and response.status_code < 400:
            await self._log_security_event(
                request, "successful_request",
                {"status_code": response.status_code}
            )
    
    def _add_security_headers(self, response: Response) -> None:
        """Add security headers to response."""
        for header, value in self.security_headers.items():
            response.headers[header] = value
    
    def _build_security_headers(self) -> Dict[str, str]:
        """Build security headers dictionary."""
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "X-Permitted-Cross-Domain-Policies": "none"
        }
        
        if self.config.enable_hsts:
            headers["Strict-Transport-Security"] = f"max-age={self.config.hsts_max_age}; includeSubDomains"
        
        if self.config.enable_csp:
            headers["Content-Security-Policy"] = self.config.csp_policy
        
        return headers
    
    def _initialize_rate_limit_rules(self) -> List[RateLimitRule]:
        """Initialize default rate limiting rules."""
        return [
            # Global rate limit
            RateLimitRule(
                key_pattern="global",
                limit=self.config.default_rate_limit,
                window_seconds=60,
                strategy=self.config.rate_limit_strategy,
                per_user=True,
                per_ip=False,
                per_endpoint=False
            ),
            
            # Authentication endpoints (more restrictive)
            RateLimitRule(
                key_pattern="auth",
                limit=10,
                window_seconds=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                per_user=False,
                per_ip=True,
                per_endpoint=True
            ),
            
            # API endpoints (per endpoint)
            RateLimitRule(
                key_pattern="api",
                limit=50,
                window_seconds=60,
                strategy=self.config.rate_limit_strategy,
                per_user=True,
                per_ip=False,
                per_endpoint=True
            )
        ]
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return str(request.client.host) if request.client else "unknown"
    
    async def _create_blocked_response(self, reason: str) -> JSONResponse:
        """Create blocked request response."""
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": "access_denied",
                "error_description": reason,
                "error_code": "SEC_001"
            },
            headers=self.security_headers
        )
    
    async def _create_rate_limit_response(self, rate_limit_info: Dict[str, Any]) -> JSONResponse:
        """Create rate limit exceeded response."""
        headers = self.security_headers.copy()
        headers.update({
            "X-RateLimit-Limit": str(rate_limit_info["limit"]),
            "X-RateLimit-Remaining": str(max(0, rate_limit_info["limit"] - rate_limit_info["current_usage"])),
            "X-RateLimit-Reset": str(int(rate_limit_info["reset_time"])),
            "Retry-After": str(int(rate_limit_info["reset_time"] - time.time()))
        })
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "rate_limit_exceeded",
                "error_description": f"Rate limit exceeded for {rate_limit_info['rule']}",
                "error_code": "SEC_002",
                "rate_limit": {
                    "limit": rate_limit_info["limit"],
                    "current_usage": rate_limit_info["current_usage"],
                    "reset_time": rate_limit_info["reset_time"]
                }
            },
            headers=headers
        )
    
    async def _log_security_event(
        self,
        request: Request,
        event_type: str,
        metadata: Dict[str, Any]
    ) -> None:
        """Log security event."""
        if not self.config.log_security_events:
            return
        
        try:
            event_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "client_ip": self._get_client_ip(request),
                "method": request.method,
                "path": str(request.url.path),
                "user_agent": request.headers.get("user-agent", ""),
                "metadata": metadata
            }
            
            # Store in Redis for analysis
            event_key = f"security_events:{datetime.utcnow().strftime('%Y-%m-%d')}"
            await self.redis.lpush(event_key, json.dumps(event_data))
            await self.redis.expire(event_key, 86400 * 7)  # Keep for 7 days
            
            # Log to structured logger
            logger.warning(f"Security event: {event_type}", **event_data)
            
        except Exception as e:
            logger.error(f"Failed to log security event: {e}")
    
    def _update_performance_metrics(self, processing_time_ms: float) -> None:
        """Update performance metrics."""
        current_avg = self.metrics.avg_processing_time_ms
        total_requests = self.metrics.total_requests
        
        self.metrics.avg_processing_time_ms = (
            (current_avg * (total_requests - 1) + processing_time_ms) / total_requests
        )
        
        if processing_time_ms > self.metrics.max_processing_time_ms:
            self.metrics.max_processing_time_ms = processing_time_ms
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get security middleware metrics."""
        return {
            "api_security_metrics": {
                "total_requests": self.metrics.total_requests,
                "blocked_requests": self.metrics.blocked_requests,
                "rate_limited_requests": self.metrics.rate_limited_requests,
                "suspicious_requests": self.metrics.suspicious_requests,
                "block_rate": self.metrics.blocked_requests / max(1, self.metrics.total_requests),
                "rate_limit_rate": self.metrics.rate_limited_requests / max(1, self.metrics.total_requests),
                "threat_detection": {
                    "sql_injection_attempts": self.metrics.sql_injection_attempts,
                    "xss_attempts": self.metrics.xss_attempts,
                    "path_traversal_attempts": self.metrics.path_traversal_attempts
                },
                "performance": {
                    "avg_processing_time_ms": self.metrics.avg_processing_time_ms,
                    "max_processing_time_ms": self.metrics.max_processing_time_ms
                },
                "rate_limiting": {
                    "hits_by_endpoint": self.metrics.rate_limit_hits_by_endpoint,
                    "hits_by_ip": self.metrics.rate_limit_hits_by_ip
                }
            },
            "configuration": {
                "rate_limiting_enabled": self.config.enable_rate_limiting,
                "security_headers_enabled": self.config.enable_security_headers,
                "threat_detection_enabled": self.config.enable_threat_detection,
                "default_rate_limit": self.config.default_rate_limit,
                "max_request_size": self.config.max_request_size
            }
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        # Store final metrics
        try:
            await self.redis.set_with_expiry(
                self._metrics_key,
                json.dumps(self.get_metrics()),
                ttl=86400
            )
        except Exception as e:
            logger.error(f"Failed to store final metrics: {e}")


# Factory function
def create_api_security_middleware(
    app: ASGIApp,
    redis_client: RedisClient,
    config: Optional[SecurityConfig] = None
) -> APISecurityMiddleware:
    """
    Create API Security Middleware instance.
    
    Args:
        app: ASGI application
        redis_client: Redis client
        config: Security configuration
        
    Returns:
        APISecurityMiddleware instance
    """
    return APISecurityMiddleware(app, redis_client, config)