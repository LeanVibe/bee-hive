"""
Production API Security Middleware for LeanVibe Agent Hive 2.0
Implements enterprise-grade API security with comprehensive protection
"""

import asyncio
import hashlib
import hmac
import ipaddress
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from urllib.parse import urlparse

import aioredis
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import RequestResponseEndpoint

logger = logging.getLogger(__name__)

class SecurityConfig(BaseModel):
    """Production security configuration"""
    # Rate limiting configuration
    rate_limit_enabled: bool = True
    max_requests_per_minute: int = 100
    max_requests_per_hour: int = 1000
    burst_capacity: int = 150
    
    # DDoS protection
    ddos_protection_enabled: bool = True
    max_connections_per_ip: int = 10
    suspicious_activity_threshold: int = 50
    
    # Input validation
    input_validation_enabled: bool = True
    max_request_size_mb: int = 10
    max_json_depth: int = 10
    
    # Security headers
    security_headers_enabled: bool = True
    hsts_max_age: int = 31536000
    
    # IP filtering
    ip_allowlist: List[str] = Field(default_factory=list)
    ip_blocklist: List[str] = Field(default_factory=list)
    
    # API key validation
    api_key_validation_enabled: bool = True
    require_api_key_for_paths: List[str] = Field(default_factory=lambda: ["/api/v1/"])
    
    # Content security
    content_security_enabled: bool = True
    allowed_content_types: List[str] = Field(default_factory=lambda: [
        "application/json",
        "application/x-www-form-urlencoded",
        "multipart/form-data"
    ])

class ThreatDetectionEngine:
    """Advanced threat detection with ML-based analysis"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
        self.threat_patterns = self._load_threat_patterns()
        self.behavioral_baseline = {}
        
    def _load_threat_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Load comprehensive threat detection patterns"""
        return {
            "sql_injection": [
                re.compile(r"(union\s+select|insert\s+into|drop\s+table)", re.IGNORECASE),
                re.compile(r"(or\s+1=1|and\s+1=1|'.*'.*=.*')", re.IGNORECASE),
                re.compile(r"(exec\s*\(|execute\s*\(|sp_)", re.IGNORECASE),
            ],
            "xss": [
                re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
                re.compile(r"javascript:", re.IGNORECASE),
                re.compile(r"on\w+\s*=\s*[\"'][^\"']*[\"']", re.IGNORECASE),
            ],
            "command_injection": [
                re.compile(r"(;|\|{1,2}|&&|\$\()", re.IGNORECASE),
                re.compile(r"(rm\s+-rf|chmod|chown|wget|curl)", re.IGNORECASE),
                re.compile(r"(\/bin\/|\/usr\/bin\/|\/sbin\/)", re.IGNORECASE),
            ],
            "path_traversal": [
                re.compile(r"(\.\.\/|\.\.\\|%2e%2e)", re.IGNORECASE),
                re.compile(r"(\/etc\/passwd|\/proc\/|\/sys\/)", re.IGNORECASE),
            ],
            "nosql_injection": [
                re.compile(r"(\$where|\$ne|\$gt|\$lt)", re.IGNORECASE),
                re.compile(r"(this\.|\$regex)", re.IGNORECASE),
            ]
        }
    
    async def analyze_request(self, request: Request, content: str) -> Dict[str, Any]:
        """Comprehensive request threat analysis"""
        threats_detected = []
        confidence_score = 0.0
        
        # Pattern-based detection
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if pattern.search(content) or pattern.search(str(request.url)):
                    threats_detected.append({
                        "type": threat_type,
                        "pattern": pattern.pattern,
                        "confidence": 0.8
                    })
                    confidence_score = max(confidence_score, 0.8)
        
        # Behavioral analysis
        client_ip = self._get_client_ip(request)
        behavioral_score = await self._analyze_behavioral_patterns(client_ip, request)
        confidence_score = max(confidence_score, behavioral_score)
        
        # Rate-based analysis
        rate_score = await self._analyze_request_rate(client_ip)
        confidence_score = max(confidence_score, rate_score)
        
        return {
            "threats_detected": threats_detected,
            "confidence_score": confidence_score,
            "is_threat": confidence_score > 0.7,
            "client_ip": client_ip,
            "risk_level": self._calculate_risk_level(confidence_score)
        }
    
    async def _analyze_behavioral_patterns(self, client_ip: str, request: Request) -> float:
        """Analyze behavioral patterns for anomaly detection"""
        try:
            # Track request patterns
            pattern_key = f"behavior:{client_ip}"
            current_pattern = {
                "path": str(request.url.path),
                "method": request.method,
                "user_agent": request.headers.get("user-agent", ""),
                "timestamp": time.time()
            }
            
            # Store pattern
            await self.redis.lpush(pattern_key, json.dumps(current_pattern))
            await self.redis.ltrim(pattern_key, 0, 99)  # Keep last 100 requests
            await self.redis.expire(pattern_key, 3600)  # 1 hour expiry
            
            # Get recent patterns
            patterns = await self.redis.lrange(pattern_key, 0, -1)
            if len(patterns) < 10:
                return 0.0  # Need baseline
            
            # Analyze patterns for anomalies
            parsed_patterns = [json.loads(p.decode()) for p in patterns]
            
            # Check for suspicious patterns
            unique_paths = set(p["path"] for p in parsed_patterns[-20:])
            unique_user_agents = set(p["user_agent"] for p in parsed_patterns[-20:])
            
            # Scoring based on diversity and frequency
            score = 0.0
            if len(unique_paths) > 15:  # Too many different paths
                score += 0.3
            if len(unique_user_agents) > 5:  # Multiple user agents
                score += 0.4
            
            # Check request frequency
            recent_requests = [p for p in parsed_patterns if time.time() - p["timestamp"] < 300]
            if len(recent_requests) > 30:  # More than 30 requests in 5 minutes
                score += 0.5
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error in behavioral analysis: {e}")
            return 0.0
    
    async def _analyze_request_rate(self, client_ip: str) -> float:
        """Analyze request rate for DDoS detection"""
        try:
            rate_key = f"rate:{client_ip}"
            current_time = int(time.time())
            
            # Sliding window rate limiting
            await self.redis.zadd(rate_key, {str(current_time): current_time})
            await self.redis.zremrangebyscore(rate_key, 0, current_time - 300)  # 5 minute window
            await self.redis.expire(rate_key, 300)
            
            # Count requests in last 5 minutes
            request_count = await self.redis.zcard(rate_key)
            
            # Score based on request rate
            if request_count > 100:  # More than 100 requests in 5 minutes
                return 0.9
            elif request_count > 50:
                return 0.6
            elif request_count > 25:
                return 0.3
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error in rate analysis: {e}")
            return 0.0
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP with proxy support"""
        # Check for forwarded IP headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _calculate_risk_level(self, confidence_score: float) -> str:
        """Calculate risk level based on confidence score"""
        if confidence_score >= 0.9:
            return "CRITICAL"
        elif confidence_score >= 0.7:
            return "HIGH"
        elif confidence_score >= 0.5:
            return "MEDIUM"
        elif confidence_score >= 0.3:
            return "LOW"
        else:
            return "MINIMAL"

class ProductionApiSecurityMiddleware(BaseHTTPMiddleware):
    """Enterprise-grade API security middleware"""
    
    def __init__(
        self,
        app: FastAPI,
        config: SecurityConfig,
        redis_client: aioredis.Redis
    ):
        super().__init__(app)
        self.config = config
        self.redis = redis_client
        self.threat_detector = ThreatDetectionEngine(redis_client)
        self.blocked_ips: Set[str] = set()
        self.api_keys: Dict[str, Dict[str, Any]] = {}
        
    async def dispatch(
        self, 
        request: Request, 
        call_next: RequestResponseEndpoint
    ) -> Response:
        """Main security processing pipeline"""
        start_time = time.time()
        
        try:
            # Pre-request security checks
            security_result = await self._perform_security_checks(request)
            
            if security_result["blocked"]:
                return await self._create_security_response(
                    security_result["reason"],
                    security_result["status_code"]
                )
            
            # Process request
            response = await call_next(request)
            
            # Post-request security headers
            if self.config.security_headers_enabled:
                response = await self._add_security_headers(response)
            
            # Log security metrics
            await self._log_security_metrics(request, response, time.time() - start_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            return await self._create_security_response(
                "Internal security error",
                status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    async def _perform_security_checks(self, request: Request) -> Dict[str, Any]:
        """Comprehensive security validation pipeline"""
        client_ip = self.threat_detector._get_client_ip(request)
        
        # IP filtering
        if await self._is_ip_blocked(client_ip):
            return {
                "blocked": True,
                "reason": "IP address blocked",
                "status_code": status.HTTP_403_FORBIDDEN
            }
        
        # Rate limiting
        if self.config.rate_limit_enabled:
            if await self._is_rate_limited(client_ip):
                return {
                    "blocked": True,
                    "reason": "Rate limit exceeded",
                    "status_code": status.HTTP_429_TOO_MANY_REQUESTS
                }
        
        # Request size validation
        content_length = int(request.headers.get("content-length", 0))
        if content_length > self.config.max_request_size_mb * 1024 * 1024:
            return {
                "blocked": True,
                "reason": "Request too large",
                "status_code": status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            }
        
        # Content type validation
        if self.config.content_security_enabled:
            content_type = request.headers.get("content-type", "")
            if content_type and not any(ct in content_type for ct in self.config.allowed_content_types):
                return {
                    "blocked": True,
                    "reason": "Invalid content type",
                    "status_code": status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
                }
        
        # API key validation
        if self.config.api_key_validation_enabled:
            if any(str(request.url.path).startswith(path) for path in self.config.require_api_key_for_paths):
                if not await self._validate_api_key(request):
                    return {
                        "blocked": True,
                        "reason": "Invalid or missing API key",
                        "status_code": status.HTTP_401_UNAUTHORIZED
                    }
        
        # Input validation and threat detection
        if self.config.input_validation_enabled:
            try:
                body = await request.body()
                content = body.decode("utf-8", errors="ignore")
                
                threat_analysis = await self.threat_detector.analyze_request(request, content)
                
                if threat_analysis["is_threat"]:
                    # Log threat
                    await self._log_threat(client_ip, threat_analysis, request)
                    
                    # Block high-confidence threats
                    if threat_analysis["confidence_score"] > 0.8:
                        # Temporarily block IP for repeated threats
                        await self._add_temp_ip_block(client_ip, 300)  # 5 minutes
                        
                        return {
                            "blocked": True,
                            "reason": "Potential security threat detected",
                            "status_code": status.HTTP_403_FORBIDDEN
                        }
                
            except Exception as e:
                logger.error(f"Error in threat detection: {e}")
        
        return {"blocked": False}
    
    async def _is_ip_blocked(self, client_ip: str) -> bool:
        """Check if IP is in blocklist or temporarily blocked"""
        # Check static blocklist
        for blocked_range in self.config.ip_blocklist:
            try:
                if ipaddress.ip_address(client_ip) in ipaddress.ip_network(blocked_range, strict=False):
                    return True
            except ValueError:
                continue
        
        # Check dynamic blocks
        temp_block_key = f"temp_block:{client_ip}"
        is_temp_blocked = await self.redis.get(temp_block_key)
        return is_temp_blocked is not None
    
    async def _is_rate_limited(self, client_ip: str) -> bool:
        """Advanced rate limiting with multiple windows"""
        current_time = int(time.time())
        
        # Check minute-based rate limit
        minute_key = f"rate_limit:minute:{client_ip}"
        minute_requests = await self.redis.incr(minute_key)
        await self.redis.expire(minute_key, 60)
        
        if minute_requests > self.config.max_requests_per_minute:
            return True
        
        # Check hour-based rate limit
        hour_key = f"rate_limit:hour:{client_ip}"
        hour_requests = await self.redis.incr(hour_key)
        await self.redis.expire(hour_key, 3600)
        
        if hour_requests > self.config.max_requests_per_hour:
            return True
        
        # Check burst capacity
        burst_key = f"rate_limit:burst:{client_ip}"
        burst_requests = await self.redis.incr(burst_key)
        await self.redis.expire(burst_key, 10)  # 10-second window
        
        if burst_requests > self.config.burst_capacity:
            return True
        
        return False
    
    async def _validate_api_key(self, request: Request) -> bool:
        """Validate API key from header or query parameter"""
        # Check Authorization header
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
        else:
            # Check X-API-Key header
            api_key = request.headers.get("x-api-key")
        
        if not api_key:
            # Check query parameter
            api_key = request.query_params.get("api_key")
        
        if not api_key:
            return False
        
        # Validate key against Redis store
        key_info = await self.redis.hgetall(f"api_key:{api_key}")
        if not key_info:
            return False
        
        # Check if key is active and not expired
        if key_info.get(b"status") != b"active":
            return False
        
        expires_at = key_info.get(b"expires_at")
        if expires_at and int(expires_at) < time.time():
            return False
        
        # Update last used timestamp
        await self.redis.hset(f"api_key:{api_key}", "last_used", int(time.time()))
        
        return True
    
    async def _add_security_headers(self, response: Response) -> Response:
        """Add comprehensive security headers"""
        # HSTS
        response.headers["Strict-Transport-Security"] = f"max-age={self.config.hsts_max_age}; includeSubDomains"
        
        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'"
        )
        
        # Other security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response
    
    async def _add_temp_ip_block(self, client_ip: str, duration: int):
        """Add temporary IP block"""
        temp_block_key = f"temp_block:{client_ip}"
        await self.redis.setex(temp_block_key, duration, "blocked")
        logger.warning(f"Temporarily blocked IP {client_ip} for {duration} seconds")
    
    async def _log_threat(self, client_ip: str, threat_analysis: Dict, request: Request):
        """Log security threat for monitoring"""
        threat_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "client_ip": client_ip,
            "path": str(request.url.path),
            "method": request.method,
            "user_agent": request.headers.get("user-agent", ""),
            "threats": threat_analysis["threats_detected"],
            "confidence_score": threat_analysis["confidence_score"],
            "risk_level": threat_analysis["risk_level"]
        }
        
        # Store in Redis for monitoring
        await self.redis.lpush("security_threats", json.dumps(threat_log))
        await self.redis.ltrim("security_threats", 0, 999)  # Keep last 1000
        
        logger.warning(f"Security threat detected: {threat_log}")
    
    async def _log_security_metrics(self, request: Request, response: Response, processing_time: float):
        """Log security metrics for monitoring"""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "client_ip": self.threat_detector._get_client_ip(request),
            "path": str(request.url.path),
            "method": request.method,
            "status_code": response.status_code,
            "processing_time": processing_time,
            "request_size": int(request.headers.get("content-length", 0)),
            "response_size": len(response.body) if hasattr(response, 'body') else 0
        }
        
        # Store metrics for monitoring
        await self.redis.lpush("security_metrics", json.dumps(metrics))
        await self.redis.ltrim("security_metrics", 0, 9999)  # Keep last 10000
    
    async def _create_security_response(self, reason: str, status_code: int) -> JSONResponse:
        """Create standardized security error response"""
        return JSONResponse(
            status_code=status_code,
            content={
                "error": "Security validation failed",
                "message": reason,
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": f"sec_{int(time.time())}"
            }
        )

async def create_production_security_middleware(
    app: FastAPI,
    redis_url: str,
    config: Optional[SecurityConfig] = None
) -> ProductionApiSecurityMiddleware:
    """Factory function to create production security middleware"""
    if config is None:
        config = SecurityConfig()
    
    # Create Redis connection
    redis_client = aioredis.from_url(redis_url)
    
    # Initialize middleware
    middleware = ProductionApiSecurityMiddleware(app, config, redis_client)
    
    return middleware

# Security utilities for API key management
class ApiKeyManager:
    """Utility class for API key management"""
    
    def __init__(self, redis_client: aioredis.Redis):
        self.redis = redis_client
    
    async def create_api_key(
        self, 
        key_id: str, 
        permissions: List[str],
        expires_in_days: Optional[int] = None
    ) -> str:
        """Create new API key with permissions"""
        api_key = self._generate_secure_key()
        
        key_data = {
            "key_id": key_id,
            "permissions": ",".join(permissions),
            "created_at": int(time.time()),
            "status": "active",
            "last_used": 0
        }
        
        if expires_in_days:
            key_data["expires_at"] = int(time.time()) + (expires_in_days * 24 * 3600)
        
        await self.redis.hset(f"api_key:{api_key}", mapping=key_data)
        
        return api_key
    
    async def revoke_api_key(self, api_key: str) -> bool:
        """Revoke API key"""
        exists = await self.redis.exists(f"api_key:{api_key}")
        if exists:
            await self.redis.hset(f"api_key:{api_key}", "status", "revoked")
            return True
        return False
    
    def _generate_secure_key(self) -> str:
        """Generate cryptographically secure API key"""
        import secrets
        return f"lv_{secrets.token_urlsafe(32)}"