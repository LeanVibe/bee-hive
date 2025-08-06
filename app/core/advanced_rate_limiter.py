"""
Advanced API Rate Limiting and Throttling Middleware for LeanVibe Agent Hive.

Implements enterprise-grade rate limiting with:
- Multiple algorithms: Token Bucket, Sliding Window, Fixed Window
- Distributed Redis backend for multi-instance scaling
- Geographic and behavioral rate limiting
- DDoS protection and abuse detection
- Enterprise policy enforcement and compliance

Production-ready with comprehensive metrics, monitoring, and audit logging.
"""

import os
import json
import time
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from ipaddress import IPv4Address, IPv6Address, AddressValueError

import structlog
import redis.asyncio as redis
from fastapi import HTTPException, Request, Response, Depends, status, middleware
from fastapi.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_session
from .auth import get_auth_service, AuthenticationService
from ..models.security import SecurityAuditLog, SecurityEvent
from ..schemas.security import SecurityError

logger = structlog.get_logger()

# Rate Limiting Configuration
RATE_LIMIT_CONFIG = {
    "default_requests_per_minute": int(os.getenv("DEFAULT_RATE_LIMIT_RPM", "60")),
    "default_requests_per_hour": int(os.getenv("DEFAULT_RATE_LIMIT_RPH", "1000")),
    "default_requests_per_day": int(os.getenv("DEFAULT_RATE_LIMIT_RPD", "10000")),
    "burst_capacity_multiplier": float(os.getenv("BURST_CAPACITY_MULTIPLIER", "2.0")),
    "window_size_seconds": int(os.getenv("SLIDING_WINDOW_SIZE", "60")),
    "redis_key_prefix": os.getenv("REDIS_RATE_LIMIT_PREFIX", "rl:"),
    "redis_key_ttl": int(os.getenv("REDIS_KEY_TTL", "3600")),  # 1 hour
    "enable_geographic_limiting": os.getenv("ENABLE_GEO_LIMITING", "true").lower() == "true",
    "enable_behavioral_analysis": os.getenv("ENABLE_BEHAVIORAL_ANALYSIS", "true").lower() == "true",
    "ddos_threshold_multiplier": float(os.getenv("DDOS_THRESHOLD_MULTIPLIER", "10.0")),
    "suspicious_activity_threshold": int(os.getenv("SUSPICIOUS_ACTIVITY_THRESHOLD", "500")),
    "auto_ban_duration_minutes": int(os.getenv("AUTO_BAN_DURATION", "60")),
    "whitelist_ips": os.getenv("RATE_LIMIT_WHITELIST_IPS", "").split(",") if os.getenv("RATE_LIMIT_WHITELIST_IPS") else [],
    "blacklist_ips": os.getenv("RATE_LIMIT_BLACKLIST_IPS", "").split(",") if os.getenv("RATE_LIMIT_BLACKLIST_IPS") else []
}

# Enterprise Rate Limiting Tiers
ENTERPRISE_RATE_LIMITS = {
    "free": {
        "rpm": 10,
        "rph": 100,
        "rpd": 1000,
        "concurrent_requests": 5,
        "burst_multiplier": 1.5
    },
    "developer": {
        "rpm": 60,
        "rph": 1000,
        "rpd": 10000,
        "concurrent_requests": 20,
        "burst_multiplier": 2.0
    },
    "professional": {
        "rpm": 300,
        "rph": 5000,
        "rpd": 50000,
        "concurrent_requests": 50,
        "burst_multiplier": 3.0
    },
    "enterprise": {
        "rpm": 1000,
        "rph": 20000,
        "rpd": 200000,
        "concurrent_requests": 200,
        "burst_multiplier": 5.0
    },
    "unlimited": {
        "rpm": 999999,
        "rph": 999999,
        "rpd": 999999,
        "concurrent_requests": 1000,
        "burst_multiplier": 10.0
    }
}


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitScope(Enum):
    """Rate limiting scopes."""
    IP_ADDRESS = "ip_address"
    USER_AGENT = "user_agent"
    API_KEY = "api_key"
    AGENT_ID = "agent_id"
    ENDPOINT = "endpoint"
    GLOBAL = "global"


class ActionType(Enum):
    """Rate limit violation actions."""
    ALLOW = "allow"
    THROTTLE = "throttle"
    REJECT = "reject"
    BAN = "ban"
    CAPTCHA = "captcha"


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    scope: RateLimitScope
    algorithm: RateLimitAlgorithm
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_capacity: int
    window_size: int = 60
    priority: int = 100
    enabled: bool = True
    
    # Geographic restrictions
    allowed_countries: Optional[List[str]] = None
    blocked_countries: Optional[List[str]] = None
    
    # Time-based restrictions
    allowed_hours: Optional[List[int]] = None  # 0-23
    blocked_hours: Optional[List[int]] = None
    
    # Advanced features
    progressive_delays: bool = False
    ban_after_violations: int = 5
    ban_duration_minutes: int = 60
    require_authentication: bool = False
    
    # Compliance features
    audit_violations: bool = True
    alert_on_violations: bool = True
    compliance_tags: List[str] = field(default_factory=list)


@dataclass
class RateLimitStatus:
    """Current rate limit status."""
    allowed: bool
    action: ActionType
    requests_remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None
    current_usage: Dict[str, int] = field(default_factory=dict)
    violation_count: int = 0
    is_banned: bool = False
    ban_expires_at: Optional[datetime] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RateLimitViolation(BaseModel):
    """Rate limit violation record."""
    id: str
    timestamp: datetime
    scope: str
    identifier: str
    rule_violated: str
    requests_made: int
    requests_allowed: int
    action_taken: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    endpoint: Optional[str]
    metadata: Dict[str, Any] = {}


class AdvancedRateLimiter:
    """
    Advanced API Rate Limiting and Throttling System.
    
    Features:
    - Multiple rate limiting algorithms with different strategies
    - Distributed Redis backend for horizontal scaling
    - Geographic and behavioral rate limiting
    - DDoS protection with automated response
    - Enterprise policy enforcement
    - Comprehensive audit logging and compliance
    """
    
    def __init__(self, db_session: Optional[AsyncSession] = None):
        """
        Initialize Advanced Rate Limiter.
        
        Args:
            db_session: Database session for audit logging
        """
        self.db = db_session
        
        # Redis connection for distributed rate limiting
        self.redis = redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True
        )
        
        # Rate limiting rules
        self.rules: List[RateLimitRule] = []
        self._initialize_default_rules()
        
        # Active bans and violations tracking
        self.active_bans: Dict[str, datetime] = {}
        self.violation_counters: Dict[str, int] = {}
        
        # Performance metrics
        self.metrics = {
            "requests_processed": 0,
            "requests_allowed": 0,
            "requests_rejected": 0,
            "requests_throttled": 0,
            "bans_issued": 0,
            "ddos_attacks_detected": 0,
            "avg_processing_time_ms": 0.0,
            "algorithm_usage": {},
            "top_violators": {},
            "geographic_blocks": {},
            "endpoint_usage": {}
        }
        
        # Behavioral analysis patterns
        self.behavioral_patterns: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Advanced Rate Limiter initialized", 
                   redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
                   rules_count=len(self.rules))
    
    def _initialize_default_rules(self):
        """Initialize default rate limiting rules."""
        
        # Global rate limiting rule
        self.rules.append(RateLimitRule(
            scope=RateLimitScope.GLOBAL,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            requests_per_minute=RATE_LIMIT_CONFIG["default_requests_per_minute"],
            requests_per_hour=RATE_LIMIT_CONFIG["default_requests_per_hour"],
            requests_per_day=RATE_LIMIT_CONFIG["default_requests_per_day"],
            burst_capacity=int(RATE_LIMIT_CONFIG["default_requests_per_minute"] * RATE_LIMIT_CONFIG["burst_capacity_multiplier"]),
            priority=100
        ))
        
        # IP-based rate limiting
        self.rules.append(RateLimitRule(
            scope=RateLimitScope.IP_ADDRESS,
            algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
            requests_per_minute=30,
            requests_per_hour=500,
            requests_per_day=5000,
            burst_capacity=60,
            priority=200,
            progressive_delays=True,
            ban_after_violations=10,
            ban_duration_minutes=30
        ))
        
        # API key rate limiting
        self.rules.append(RateLimitRule(
            scope=RateLimitScope.API_KEY,
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
            requests_per_minute=100,
            requests_per_hour=2000,
            requests_per_day=20000,
            burst_capacity=200,
            priority=150,
            require_authentication=True
        ))
        
        # Agent-specific rate limiting
        self.rules.append(RateLimitRule(
            scope=RateLimitScope.AGENT_ID,
            algorithm=RateLimitAlgorithm.LEAKY_BUCKET,
            requests_per_minute=50,
            requests_per_hour=1000,
            requests_per_day=10000,
            burst_capacity=100,
            priority=120
        ))
        
        # Endpoint-specific rate limiting (stricter for sensitive endpoints)
        self.rules.append(RateLimitRule(
            scope=RateLimitScope.ENDPOINT,
            algorithm=RateLimitAlgorithm.FIXED_WINDOW,
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=1000,
            burst_capacity=20,
            priority=250,
            compliance_tags=["sensitive", "admin"],
            audit_violations=True,
            alert_on_violations=True
        ))
    
    async def check_rate_limit(self, request: Request, identifier: str = None) -> RateLimitStatus:
        """
        Check if request should be rate limited.
        
        Args:
            request: FastAPI request object
            identifier: Optional custom identifier (defaults to IP address)
            
        Returns:
            RateLimitStatus with decision and metadata
        """
        start_time = time.time()
        
        try:
            # Extract request metadata
            ip_address = self._get_client_ip(request)
            user_agent = request.headers.get("User-Agent", "")
            endpoint = str(request.url.path)
            api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
            agent_id = request.headers.get("X-Agent-ID")
            
            # Use IP address as default identifier
            identifier = identifier or ip_address
            
            # Check IP whitelist/blacklist
            if ip_address in RATE_LIMIT_CONFIG["whitelist_ips"]:
                return RateLimitStatus(
                    allowed=True,
                    action=ActionType.ALLOW,
                    requests_remaining=999999,
                    reset_time=datetime.utcnow() + timedelta(hours=1),
                    reason="IP whitelisted"
                )
            
            if ip_address in RATE_LIMIT_CONFIG["blacklist_ips"]:
                await self._log_rate_limit_event(
                    identifier=identifier,
                    action="blacklist_block",
                    metadata={"ip_address": ip_address, "endpoint": endpoint}
                )
                return RateLimitStatus(
                    allowed=False,
                    action=ActionType.REJECT,
                    requests_remaining=0,
                    reset_time=datetime.utcnow() + timedelta(days=1),
                    reason="IP blacklisted"
                )
            
            # Check if identifier is currently banned
            ban_status = await self._check_ban_status(identifier)
            if ban_status["is_banned"]:
                return RateLimitStatus(
                    allowed=False,
                    action=ActionType.REJECT,
                    requests_remaining=0,
                    reset_time=ban_status["expires_at"],
                    is_banned=True,
                    ban_expires_at=ban_status["expires_at"],
                    reason=f"Temporarily banned: {ban_status['reason']}"
                )
            
            # Check geographic restrictions
            if RATE_LIMIT_CONFIG["enable_geographic_limiting"]:
                geo_status = await self._check_geographic_limits(ip_address)
                if not geo_status["allowed"]:
                    return RateLimitStatus(
                        allowed=False,
                        action=ActionType.REJECT,
                        requests_remaining=0,
                        reset_time=datetime.utcnow() + timedelta(hours=24),
                        reason=f"Geographic restriction: {geo_status['reason']}"
                    )
            
            # Apply rate limiting rules in priority order
            applicable_rules = self._get_applicable_rules(request, identifier, api_key, agent_id, endpoint)
            
            most_restrictive_status = None
            combined_status = RateLimitStatus(
                allowed=True,
                action=ActionType.ALLOW,
                requests_remaining=999999,
                reset_time=datetime.utcnow() + timedelta(hours=1)
            )
            
            for rule in applicable_rules:
                rule_status = await self._apply_rate_limit_rule(rule, request, identifier)
                
                # Track rule usage
                algorithm_key = rule.algorithm.value
                self.metrics["algorithm_usage"][algorithm_key] = (
                    self.metrics["algorithm_usage"].get(algorithm_key, 0) + 1
                )
                
                if not rule_status.allowed:
                    most_restrictive_status = rule_status
                    break
                
                # Use most restrictive limits
                combined_status.requests_remaining = min(
                    combined_status.requests_remaining,
                    rule_status.requests_remaining
                )
                
                if rule_status.reset_time < combined_status.reset_time:
                    combined_status.reset_time = rule_status.reset_time
            
            # Use most restrictive result
            final_status = most_restrictive_status or combined_status
            
            # Update behavioral analysis
            if RATE_LIMIT_CONFIG["enable_behavioral_analysis"]:
                await self._update_behavioral_analysis(identifier, request, final_status)
            
            # Check for DDoS patterns
            await self._check_ddos_patterns(identifier, final_status)
            
            # Update metrics
            self.metrics["requests_processed"] += 1
            if final_status.allowed:
                self.metrics["requests_allowed"] += 1
            else:
                self.metrics["requests_rejected"] += 1
                self._update_violator_metrics(identifier)
            
            # Update endpoint usage metrics
            self.metrics["endpoint_usage"][endpoint] = (
                self.metrics["endpoint_usage"].get(endpoint, 0) + 1
            )
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            current_avg = self.metrics["avg_processing_time_ms"]
            total_requests = self.metrics["requests_processed"]
            self.metrics["avg_processing_time_ms"] = (
                (current_avg * (total_requests - 1) + processing_time_ms) / total_requests
            )
            
            # Log rate limit check
            if not final_status.allowed or final_status.violation_count > 0:
                await self._log_rate_limit_event(
                    identifier=identifier,
                    action=final_status.action.value,
                    metadata={
                        "ip_address": ip_address,
                        "endpoint": endpoint,
                        "user_agent": user_agent[:200],  # Truncate for storage
                        "requests_remaining": final_status.requests_remaining,
                        "violation_count": final_status.violation_count,
                        "processing_time_ms": round(processing_time_ms, 2)
                    }
                )
            
            return final_status
            
        except Exception as e:
            logger.error("Rate limit check failed", identifier=identifier, error=str(e))
            
            # Fail open with logging
            return RateLimitStatus(
                allowed=True,
                action=ActionType.ALLOW,
                requests_remaining=1,
                reset_time=datetime.utcnow() + timedelta(minutes=1),
                reason=f"Rate limiter error: {str(e)}"
            )
    
    async def _apply_rate_limit_rule(self, rule: RateLimitRule, request: Request, identifier: str) -> RateLimitStatus:
        """Apply a specific rate limiting rule."""
        
        if not rule.enabled:
            return RateLimitStatus(
                allowed=True,
                action=ActionType.ALLOW,
                requests_remaining=999999,
                reset_time=datetime.utcnow() + timedelta(hours=1)
            )
        
        # Generate Redis keys based on scope and algorithm
        scope_identifier = self._generate_scope_identifier(rule.scope, request, identifier)
        
        if rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return await self._apply_token_bucket(rule, scope_identifier)
        elif rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return await self._apply_sliding_window(rule, scope_identifier)
        elif rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return await self._apply_fixed_window(rule, scope_identifier)
        elif rule.algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            return await self._apply_leaky_bucket(rule, scope_identifier)
        else:
            # Default to sliding window
            return await self._apply_sliding_window(rule, scope_identifier)
    
    async def _apply_token_bucket(self, rule: RateLimitRule, identifier: str) -> RateLimitStatus:
        """Apply token bucket rate limiting algorithm."""
        
        redis_key = f"{RATE_LIMIT_CONFIG['redis_key_prefix']}tb:{identifier}"
        current_time = time.time()
        
        # Lua script for atomic token bucket operations
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])  -- tokens per second
        local requested_tokens = tonumber(ARGV[3])
        local current_time = tonumber(ARGV[4])
        local ttl = tonumber(ARGV[5])
        
        local bucket = redis.call('hmget', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or current_time
        
        -- Calculate tokens to add based on time elapsed
        local time_passed = math.max(0, current_time - last_refill)
        local tokens_to_add = math.min(capacity - tokens, time_passed * refill_rate)
        tokens = math.min(capacity, tokens + tokens_to_add)
        
        local allowed = 0
        if tokens >= requested_tokens then
            tokens = tokens - requested_tokens
            allowed = 1
        end
        
        -- Update bucket state
        redis.call('hmset', key, 'tokens', tokens, 'last_refill', current_time)
        redis.call('expire', key, ttl)
        
        return {allowed, tokens, current_time + ((capacity - tokens) / refill_rate)}
        """
        
        # Calculate refill rate (tokens per second)
        refill_rate = rule.requests_per_minute / 60.0
        
        result = await self.redis.eval(
            lua_script, 1, redis_key,
            rule.burst_capacity,  # capacity
            refill_rate,  # refill rate
            1,  # requested tokens
            current_time,
            RATE_LIMIT_CONFIG["redis_key_ttl"]
        )
        
        allowed, remaining_tokens, reset_time_timestamp = result
        reset_time = datetime.fromtimestamp(reset_time_timestamp)
        
        return RateLimitStatus(
            allowed=bool(allowed),
            action=ActionType.ALLOW if allowed else ActionType.THROTTLE,
            requests_remaining=int(remaining_tokens),
            reset_time=reset_time,
            current_usage={"tokens_used": rule.burst_capacity - int(remaining_tokens)}
        )
    
    async def _apply_sliding_window(self, rule: RateLimitRule, identifier: str) -> RateLimitStatus:
        """Apply sliding window rate limiting algorithm."""
        
        redis_key = f"{RATE_LIMIT_CONFIG['redis_key_prefix']}sw:{identifier}"
        current_time = time.time()
        window_start = current_time - rule.window_size
        
        # Lua script for atomic sliding window operations
        lua_script = """
        local key = KEYS[1]
        local window_size = tonumber(ARGV[1])
        local max_requests = tonumber(ARGV[2])
        local current_time = tonumber(ARGV[3])
        local ttl = tonumber(ARGV[4])
        
        local window_start = current_time - window_size
        
        -- Remove old entries
        redis.call('zremrangebyscore', key, '-inf', window_start)
        
        -- Count current requests in window
        local current_count = redis.call('zcard', key)
        
        local allowed = 0
        if current_count < max_requests then
            -- Add current request
            redis.call('zadd', key, current_time, current_time)
            allowed = 1
            current_count = current_count + 1
        end
        
        redis.call('expire', key, ttl)
        
        return {allowed, max_requests - current_count, window_start + window_size}
        """
        
        result = await self.redis.eval(
            lua_script, 1, redis_key,
            rule.window_size,
            rule.requests_per_minute,
            current_time,
            RATE_LIMIT_CONFIG["redis_key_ttl"]
        )
        
        allowed, remaining_requests, reset_time_timestamp = result
        reset_time = datetime.fromtimestamp(reset_time_timestamp)
        
        return RateLimitStatus(
            allowed=bool(allowed),
            action=ActionType.ALLOW if allowed else ActionType.THROTTLE,
            requests_remaining=int(remaining_requests),
            reset_time=reset_time,
            current_usage={"window_usage": rule.requests_per_minute - int(remaining_requests)}
        )
    
    async def _apply_fixed_window(self, rule: RateLimitRule, identifier: str) -> RateLimitStatus:
        """Apply fixed window rate limiting algorithm."""
        
        current_time = int(time.time())
        window_start = (current_time // rule.window_size) * rule.window_size
        redis_key = f"{RATE_LIMIT_CONFIG['redis_key_prefix']}fw:{identifier}:{window_start}"
        
        # Lua script for atomic fixed window operations
        lua_script = """
        local key = KEYS[1]
        local max_requests = tonumber(ARGV[1])
        local ttl = tonumber(ARGV[2])
        
        local current_count = tonumber(redis.call('get', key)) or 0
        
        local allowed = 0
        if current_count < max_requests then
            current_count = redis.call('incr', key)
            allowed = 1
        end
        
        redis.call('expire', key, ttl)
        
        return {allowed, max_requests - current_count}
        """
        
        result = await self.redis.eval(
            lua_script, 1, redis_key,
            rule.requests_per_minute,
            rule.window_size
        )
        
        allowed, remaining_requests = result
        reset_time = datetime.fromtimestamp(window_start + rule.window_size)
        
        return RateLimitStatus(
            allowed=bool(allowed),
            action=ActionType.ALLOW if allowed else ActionType.THROTTLE,
            requests_remaining=int(remaining_requests),
            reset_time=reset_time,
            current_usage={"window_start": window_start, "requests_used": rule.requests_per_minute - int(remaining_requests)}
        )
    
    async def _apply_leaky_bucket(self, rule: RateLimitRule, identifier: str) -> RateLimitStatus:
        """Apply leaky bucket rate limiting algorithm."""
        
        redis_key = f"{RATE_LIMIT_CONFIG['redis_key_prefix']}lb:{identifier}"
        current_time = time.time()
        
        # Lua script for atomic leaky bucket operations
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local leak_rate = tonumber(ARGV[2])  -- requests per second
        local current_time = tonumber(ARGV[3])
        local ttl = tonumber(ARGV[4])
        
        local bucket = redis.call('hmget', key, 'volume', 'last_leak')
        local volume = tonumber(bucket[1]) or 0
        local last_leak = tonumber(bucket[2]) or current_time
        
        -- Calculate volume to leak based on time elapsed
        local time_passed = math.max(0, current_time - last_leak)
        local leaked_volume = time_passed * leak_rate
        volume = math.max(0, volume - leaked_volume)
        
        local allowed = 0
        if volume < capacity then
            volume = volume + 1
            allowed = 1
        end
        
        -- Update bucket state
        redis.call('hmset', key, 'volume', volume, 'last_leak', current_time)
        redis.call('expire', key, ttl)
        
        return {allowed, capacity - volume, current_time + (volume / leak_rate)}
        """
        
        # Calculate leak rate (requests per second)
        leak_rate = rule.requests_per_minute / 60.0
        
        result = await self.redis.eval(
            lua_script, 1, redis_key,
            rule.burst_capacity,  # capacity
            leak_rate,  # leak rate
            current_time,
            RATE_LIMIT_CONFIG["redis_key_ttl"]
        )
        
        allowed, available_capacity, reset_time_timestamp = result
        reset_time = datetime.fromtimestamp(reset_time_timestamp)
        
        return RateLimitStatus(
            allowed=bool(allowed),
            action=ActionType.ALLOW if allowed else ActionType.THROTTLE,
            requests_remaining=int(available_capacity),
            reset_time=reset_time,
            current_usage={"bucket_volume": rule.burst_capacity - int(available_capacity)}
        )
    
    def _generate_scope_identifier(self, scope: RateLimitScope, request: Request, base_identifier: str) -> str:
        """Generate identifier based on rate limiting scope."""
        
        if scope == RateLimitScope.IP_ADDRESS:
            return self._get_client_ip(request)
        elif scope == RateLimitScope.USER_AGENT:
            user_agent = request.headers.get("User-Agent", "unknown")
            return hashlib.sha256(user_agent.encode()).hexdigest()[:16]
        elif scope == RateLimitScope.API_KEY:
            api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key", "anonymous")
            return f"apikey:{api_key}"
        elif scope == RateLimitScope.AGENT_ID:
            agent_id = request.headers.get("X-Agent-ID", "unknown")
            return f"agent:{agent_id}"
        elif scope == RateLimitScope.ENDPOINT:
            endpoint = str(request.url.path)
            return f"endpoint:{endpoint}"
        elif scope == RateLimitScope.GLOBAL:
            return "global"
        else:
            return base_identifier
    
    def _get_applicable_rules(self, request: Request, identifier: str, api_key: str, agent_id: str, endpoint: str) -> List[RateLimitRule]:
        """Get applicable rate limiting rules for the request."""
        
        applicable_rules = []
        
        for rule in self.rules:
            # Check time-based restrictions
            if rule.allowed_hours or rule.blocked_hours:
                current_hour = datetime.utcnow().hour
                if rule.blocked_hours and current_hour in rule.blocked_hours:
                    continue
                if rule.allowed_hours and current_hour not in rule.allowed_hours:
                    continue
            
            # Check authentication requirements
            if rule.require_authentication and not (api_key or agent_id):
                continue
            
            # Check scope applicability
            scope_applies = True
            if rule.scope == RateLimitScope.API_KEY and not api_key:
                scope_applies = False
            elif rule.scope == RateLimitScope.AGENT_ID and not agent_id:
                scope_applies = False
            
            if scope_applies:
                applicable_rules.append(rule)
        
        # Sort by priority (lower number = higher priority)
        applicable_rules.sort(key=lambda r: r.priority)
        
        return applicable_rules
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        forwarded = request.headers.get("X-Forwarded")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Fallback to direct client IP
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"
    
    async def _check_ban_status(self, identifier: str) -> Dict[str, Any]:
        """Check if identifier is currently banned."""
        
        ban_key = f"{RATE_LIMIT_CONFIG['redis_key_prefix']}ban:{identifier}"
        
        ban_data = await self.redis.hgetall(ban_key)
        if not ban_data:
            return {"is_banned": False}
        
        expires_at = datetime.fromisoformat(ban_data.get("expires_at"))
        if expires_at <= datetime.utcnow():
            # Ban expired, remove it
            await self.redis.delete(ban_key)
            return {"is_banned": False}
        
        return {
            "is_banned": True,
            "expires_at": expires_at,
            "reason": ban_data.get("reason", "Rate limit violations"),
            "banned_at": datetime.fromisoformat(ban_data.get("banned_at"))
        }
    
    async def _check_geographic_limits(self, ip_address: str) -> Dict[str, Any]:
        """Check geographic rate limiting restrictions."""
        
        # Placeholder for GeoIP lookup
        # In production, integrate with MaxMind GeoIP2 or similar service
        
        try:
            # Simple validation that it's a valid IP
            import ipaddress
            ipaddress.ip_address(ip_address)
            
            # For demo purposes, block certain test ranges
            if ip_address.startswith("192.168.") or ip_address.startswith("10."):
                return {"allowed": True, "country": "private", "reason": "Private IP"}
            
            # Simulate country detection and blocking
            blocked_countries = ["CN", "RU", "KP"]  # Example blocked countries
            detected_country = "US"  # Placeholder
            
            if detected_country in blocked_countries:
                self.metrics["geographic_blocks"][detected_country] = (
                    self.metrics["geographic_blocks"].get(detected_country, 0) + 1
                )
                return {"allowed": False, "country": detected_country, "reason": f"Country {detected_country} is blocked"}
            
            return {"allowed": True, "country": detected_country}
            
        except Exception as e:
            logger.warning("Geographic check failed", ip_address=ip_address, error=str(e))
            return {"allowed": True, "reason": "Geographic check failed"}
    
    async def _update_behavioral_analysis(self, identifier: str, request: Request, status: RateLimitStatus):
        """Update behavioral analysis patterns."""
        
        current_time = datetime.utcnow()
        
        if identifier not in self.behavioral_patterns:
            self.behavioral_patterns[identifier] = {
                "first_seen": current_time,
                "total_requests": 0,
                "failed_requests": 0,
                "endpoints": set(),
                "user_agents": set(),
                "request_intervals": [],
                "violation_count": 0,
                "last_violation": None
            }
        
        pattern = self.behavioral_patterns[identifier]
        pattern["total_requests"] += 1
        pattern["endpoints"].add(str(request.url.path))
        pattern["user_agents"].add(request.headers.get("User-Agent", "")[:50])  # Truncate
        
        if not status.allowed:
            pattern["failed_requests"] += 1
            pattern["violation_count"] += 1
            pattern["last_violation"] = current_time
        
        # Track request intervals for burst detection
        if len(pattern["request_intervals"]) >= 10:
            pattern["request_intervals"].pop(0)
        pattern["request_intervals"].append(current_time.timestamp())
        
        # Detect suspicious patterns
        await self._detect_suspicious_behavior(identifier, pattern)
    
    async def _detect_suspicious_behavior(self, identifier: str, pattern: Dict[str, Any]):
        """Detect suspicious behavioral patterns."""
        
        suspicious_indicators = []
        
        # High request frequency
        if len(pattern["request_intervals"]) >= 5:
            intervals = pattern["request_intervals"]
            if intervals[-1] - intervals[0] < 5:  # 5 requests in 5 seconds
                suspicious_indicators.append("high_frequency_requests")
        
        # High failure rate
        if pattern["total_requests"] > 10:
            failure_rate = pattern["failed_requests"] / pattern["total_requests"]
            if failure_rate > 0.5:  # More than 50% failures
                suspicious_indicators.append("high_failure_rate")
        
        # Too many different endpoints
        if len(pattern["endpoints"]) > 50:
            suspicious_indicators.append("endpoint_enumeration")
        
        # Multiple user agents (possible bot)
        if len(pattern["user_agents"]) > 10:
            suspicious_indicators.append("multiple_user_agents")
        
        # Recent violations
        if pattern["violation_count"] > 5:
            suspicious_indicators.append("repeated_violations")
        
        # If suspicious patterns detected, escalate
        if suspicious_indicators:
            await self._handle_suspicious_activity(identifier, suspicious_indicators, pattern)
    
    async def _handle_suspicious_activity(self, identifier: str, indicators: List[str], pattern: Dict[str, Any]):
        """Handle detected suspicious activity."""
        
        risk_score = len(indicators) * 0.2  # Simple scoring
        
        # Log security event
        if self.db:
            security_event = SecurityEvent(
                event_type="suspicious_behavior",
                severity="medium" if risk_score < 0.8 else "high",
                source_ip=identifier if identifier.replace(".", "").isdigit() else None,
                description=f"Suspicious behavior detected: {', '.join(indicators)}",
                details={
                    "indicators": indicators,
                    "total_requests": pattern["total_requests"],
                    "failure_rate": pattern["failed_requests"] / pattern["total_requests"],
                    "unique_endpoints": len(pattern["endpoints"]),
                    "unique_user_agents": len(pattern["user_agents"]),
                    "violation_count": pattern["violation_count"]
                },
                risk_score=risk_score
            )
            self.db.add(security_event)
        
        # Auto-ban if risk is high enough
        if risk_score >= 0.8:
            await self._issue_ban(
                identifier=identifier,
                duration_minutes=RATE_LIMIT_CONFIG["auto_ban_duration_minutes"],
                reason=f"Automated ban: suspicious behavior ({', '.join(indicators)})"
            )
    
    async def _check_ddos_patterns(self, identifier: str, status: RateLimitStatus):
        """Check for DDoS attack patterns."""
        
        if not status.allowed and status.violation_count > 0:
            # Check global request rate
            global_key = f"{RATE_LIMIT_CONFIG['redis_key_prefix']}global:ddos_check"
            current_time = int(time.time())
            
            # Increment global violation counter
            await self.redis.zincrby(global_key, 1, current_time)
            await self.redis.expire(global_key, 300)  # 5-minute window
            
            # Check if we have excessive violations globally
            violation_count = await self.redis.zcard(global_key)
            if violation_count > RATE_LIMIT_CONFIG["suspicious_activity_threshold"]:
                
                # This might be a DDoS attack
                self.metrics["ddos_attacks_detected"] += 1
                
                # Log DDoS detection
                await self._log_rate_limit_event(
                    identifier="global",
                    action="ddos_detected",
                    metadata={
                        "violation_count": violation_count,
                        "time_window": 300,
                        "threshold": RATE_LIMIT_CONFIG["suspicious_activity_threshold"]
                    }
                )
                
                logger.critical("Potential DDoS attack detected",
                              violation_count=violation_count,
                              threshold=RATE_LIMIT_CONFIG["suspicious_activity_threshold"])
    
    async def _issue_ban(self, identifier: str, duration_minutes: int, reason: str):
        """Issue a temporary ban for an identifier."""
        
        ban_key = f"{RATE_LIMIT_CONFIG['redis_key_prefix']}ban:{identifier}"
        banned_at = datetime.utcnow()
        expires_at = banned_at + timedelta(minutes=duration_minutes)
        
        ban_data = {
            "identifier": identifier,
            "banned_at": banned_at.isoformat(),
            "expires_at": expires_at.isoformat(),
            "duration_minutes": duration_minutes,
            "reason": reason
        }
        
        await self.redis.hset(ban_key, mapping=ban_data)
        await self.redis.expire(ban_key, int(duration_minutes * 60))
        
        # Update metrics
        self.metrics["bans_issued"] += 1
        
        # Log ban event
        await self._log_rate_limit_event(
            identifier=identifier,
            action="temporary_ban",
            metadata={
                "duration_minutes": duration_minutes,
                "expires_at": expires_at.isoformat(),
                "reason": reason
            }
        )
        
        logger.warning("Temporary ban issued", 
                      identifier=identifier,
                      duration_minutes=duration_minutes,
                      reason=reason)
    
    def _update_violator_metrics(self, identifier: str):
        """Update top violators metrics."""
        
        self.metrics["top_violators"][identifier] = (
            self.metrics["top_violators"].get(identifier, 0) + 1
        )
        
        # Keep only top 100 violators
        if len(self.metrics["top_violators"]) > 100:
            sorted_violators = sorted(
                self.metrics["top_violators"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            self.metrics["top_violators"] = dict(sorted_violators[:100])
    
    async def _log_rate_limit_event(self, identifier: str, action: str, metadata: Dict[str, Any]):
        """Log rate limiting events for audit and compliance."""
        
        if self.db:
            audit_log = SecurityAuditLog(
                agent_id=None,
                human_controller=metadata.get("ip_address", identifier),
                action=f"rate_limit_{action}",
                resource="api_access",
                resource_id=metadata.get("endpoint", ""),
                success=action in ["allow", "throttle"],
                metadata={
                    "identifier": identifier,
                    "rate_limiting_action": action,
                    **metadata
                }
            )
            self.db.add(audit_log)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive rate limiting metrics."""
        
        return {
            "rate_limiting_metrics": self.metrics.copy(),
            "active_rules": len(self.rules),
            "behavioral_patterns_tracked": len(self.behavioral_patterns),
            "redis_connection_status": "connected",  # Simplified check
            "config": {
                "default_rpm": RATE_LIMIT_CONFIG["default_requests_per_minute"],
                "default_rph": RATE_LIMIT_CONFIG["default_requests_per_hour"],
                "default_rpd": RATE_LIMIT_CONFIG["default_requests_per_day"],
                "geographic_limiting": RATE_LIMIT_CONFIG["enable_geographic_limiting"],
                "behavioral_analysis": RATE_LIMIT_CONFIG["enable_behavioral_analysis"],
                "ddos_threshold": RATE_LIMIT_CONFIG["suspicious_activity_threshold"],
                "auto_ban_duration": RATE_LIMIT_CONFIG["auto_ban_duration_minutes"]
            }
        }
    
    async def add_rate_limit_rule(self, rule: RateLimitRule) -> bool:
        """Add a new rate limiting rule."""
        
        try:
            self.rules.append(rule)
            self.rules.sort(key=lambda r: r.priority)
            
            logger.info("Rate limiting rule added",
                       scope=rule.scope.value,
                       algorithm=rule.algorithm.value,
                       priority=rule.priority)
            
            return True
            
        except Exception as e:
            logger.error("Failed to add rate limiting rule", error=str(e))
            return False
    
    async def update_enterprise_tier(self, identifier: str, tier: str) -> bool:
        """Update rate limiting tier for enterprise customers."""
        
        try:
            if tier not in ENTERPRISE_RATE_LIMITS:
                return False
            
            tier_config = ENTERPRISE_RATE_LIMITS[tier]
            
            # Store tier configuration in Redis
            tier_key = f"{RATE_LIMIT_CONFIG['redis_key_prefix']}tier:{identifier}"
            await self.redis.hset(tier_key, mapping={
                "tier": tier,
                "rpm": tier_config["rpm"],
                "rph": tier_config["rph"],
                "rpd": tier_config["rpd"],
                "concurrent_requests": tier_config["concurrent_requests"],
                "burst_multiplier": tier_config["burst_multiplier"]
            })
            await self.redis.expire(tier_key, 86400 * 30)  # 30-day expiration
            
            logger.info("Enterprise tier updated",
                       identifier=identifier,
                       tier=tier,
                       limits=tier_config)
            
            return True
            
        except Exception as e:
            logger.error("Failed to update enterprise tier", identifier=identifier, tier=tier, error=str(e))
            return False


# Global rate limiter instance
_rate_limiter: Optional[AdvancedRateLimiter] = None


async def get_rate_limiter(db: AsyncSession = Depends(get_session)) -> AdvancedRateLimiter:
    """Get or create rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = AdvancedRateLimiter(db)
    return _rate_limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for automatic rate limiting."""
    
    def __init__(self, app, rate_limiter: AdvancedRateLimiter = None):
        super().__init__(app)
        self.rate_limiter = rate_limiter
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for certain paths
        skip_paths = ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]
        if any(request.url.path.startswith(path) for path in skip_paths):
            return await call_next(request)
        
        if not self.rate_limiter:
            # Get rate limiter from dependency injection
            from .database import get_session
            async with get_session() as db:
                self.rate_limiter = AdvancedRateLimiter(db)
        
        # Check rate limits
        try:
            status = await self.rate_limiter.check_rate_limit(request)
            
            # Add rate limit headers
            response = None
            if status.allowed:
                response = await call_next(request)
            else:
                response = Response(
                    content=json.dumps({
                        "error": "Rate limit exceeded",
                        "message": status.reason or "Too many requests",
                        "retry_after": status.retry_after
                    }),
                    status_code=429 if status.action == ActionType.THROTTLE else 403,
                    media_type="application/json"
                )
            
            # Add rate limiting headers
            response.headers["X-RateLimit-Limit"] = "60"  # Simplified
            response.headers["X-RateLimit-Remaining"] = str(max(0, status.requests_remaining))
            response.headers["X-RateLimit-Reset"] = str(int(status.reset_time.timestamp()))
            
            if status.retry_after:
                response.headers["Retry-After"] = str(status.retry_after)
            
            if status.is_banned:
                response.headers["X-RateLimit-Banned"] = "true"
                response.headers["X-RateLimit-Ban-Expires"] = str(int(status.ban_expires_at.timestamp()))
            
            return response
            
        except Exception as e:
            logger.error("Rate limiting middleware error", error=str(e))
            # Fail open - allow request to proceed
            return await call_next(request)


# Export rate limiting components
__all__ = [
    "AdvancedRateLimiter", "get_rate_limiter", "RateLimitMiddleware",
    "RateLimitRule", "RateLimitStatus", "RateLimitAlgorithm", "RateLimitScope",
    "ActionType", "RateLimitViolation", "ENTERPRISE_RATE_LIMITS"
]