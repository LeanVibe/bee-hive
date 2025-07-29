"""
Advanced Rate Limiter and DDoS Protection for LeanVibe Agent Hive 2.0.

Implements sophisticated rate limiting with multiple algorithms, DDoS protection,
adaptive throttling, and intelligent threat detection with Redis backend.
"""

import time
import uuid
import hashlib
import json
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import asyncio
import math

import structlog
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .redis import RedisClient

logger = structlog.get_logger()


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithm types."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"


class ThreatLevel(Enum):
    """DDoS threat level enumeration."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    name: str
    requests_per_second: int
    requests_per_minute: int
    requests_per_hour: int
    burst_capacity: int
    algorithm: RateLimitAlgorithm
    
    # Scoping
    apply_to_paths: Optional[List[str]] = None
    apply_to_methods: Optional[List[str]] = None
    apply_to_user_agents: Optional[List[str]] = None
    exclude_paths: Optional[List[str]] = None
    
    # Advanced features
    enable_progressive_penalties: bool = True
    penalty_multiplier: float = 2.0
    max_penalty_duration_minutes: int = 60
    whitelist_ips: Set[str] = field(default_factory=set)
    blacklist_ips: Set[str] = field(default_factory=set)
    
    # DDoS protection
    enable_ddos_protection: bool = True
    ddos_threshold_multiplier: float = 10.0
    ddos_detection_window_minutes: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "requests_per_second": self.requests_per_second,
            "requests_per_minute": self.requests_per_minute,
            "requests_per_hour": self.requests_per_hour,
            "burst_capacity": self.burst_capacity,
            "algorithm": self.algorithm.value,
            "apply_to_paths": self.apply_to_paths,
            "apply_to_methods": self.apply_to_methods,
            "apply_to_user_agents": self.apply_to_user_agents,
            "exclude_paths": self.exclude_paths,
            "enable_progressive_penalties": self.enable_progressive_penalties,
            "penalty_multiplier": self.penalty_multiplier,
            "max_penalty_duration_minutes": self.max_penalty_duration_minutes,
            "whitelist_ips": list(self.whitelist_ips),
            "blacklist_ips": list(self.blacklist_ips),
            "enable_ddos_protection": self.enable_ddos_protection,
            "ddos_threshold_multiplier": self.ddos_threshold_multiplier,
            "ddos_detection_window_minutes": self.ddos_detection_window_minutes
        }


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    limit_exceeded: bool
    current_usage: int
    limit: int
    reset_time: datetime
    retry_after_seconds: int
    threat_level: ThreatLevel
    applied_rule: str
    penalty_active: bool = False
    penalty_expires_at: Optional[datetime] = None
    ddos_detected: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "limit_exceeded": self.limit_exceeded,
            "current_usage": self.current_usage,
            "limit": self.limit,
            "reset_time": self.reset_time.isoformat(),
            "retry_after_seconds": self.retry_after_seconds,
            "threat_level": self.threat_level.value,
            "applied_rule": self.applied_rule,
            "penalty_active": self.penalty_active,
            "penalty_expires_at": self.penalty_expires_at.isoformat() if self.penalty_expires_at else None,
            "ddos_detected": self.ddos_detected
        }


@dataclass
class DDoSDetectionResult:
    """Result of DDoS detection analysis."""
    is_ddos: bool
    threat_level: ThreatLevel
    confidence_score: float
    attack_pattern: str
    source_ips: List[str]
    request_rate: float
    anomaly_indicators: List[str]
    recommended_action: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_ddos": self.is_ddos,  
            "threat_level": self.threat_level.value,
            "confidence_score": self.confidence_score,
            "attack_pattern": self.attack_pattern,
            "source_ips": self.source_ips,
            "request_rate": self.request_rate,
            "anomaly_indicators": self.anomaly_indicators,
            "recommended_action": self.recommended_action
        }


class AdvancedRateLimiter:
    """
    Advanced rate limiter with multiple algorithms and DDoS protection.
    
    Features:
    - Multiple rate limiting algorithms (sliding window, token bucket, etc.)
    - Progressive penalties for repeat offenders
    - DDoS detection and mitigation
    - Adaptive rate limiting based on system load
    - IP geolocation-based rules
    - User agent fingerprinting
    - Real-time threat analysis
    - Comprehensive metrics and monitoring
    """
    
    def __init__(
        self,
        redis_client: RedisClient,
        default_rules: Optional[List[RateLimitRule]] = None,
        enable_adaptive_limiting: bool = True,
        enable_ddos_protection: bool = True
    ):
        """Initialize advanced rate limiter."""
        self.redis = redis_client
        self.enable_adaptive_limiting = enable_adaptive_limiting
        self.enable_ddos_protection = enable_ddos_protection
        
        # Default rate limiting rules
        self.rules = default_rules or self._create_default_rules()
        
        # Redis key prefixes
        self.rate_limit_prefix = "rate_limit:"
        self.penalty_prefix = "penalty:"
        self.ddos_detection_prefix = "ddos:"
        self.metrics_prefix = "rate_metrics:"
        self.threat_analysis_prefix = "threat:"
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "penalties_applied": 0,
            "ddos_attacks_detected": 0,
            "adaptive_adjustments": 0,
            "algorithm_usage": {
                "fixed_window": 0,
                "sliding_window": 0,
                "token_bucket": 0,
                "leaky_bucket": 0,
                "adaptive": 0
            },
            "threat_levels": {
                "none": 0,
                "low": 0,
                "medium": 0,
                "high": 0,
                "critical": 0
            }
        }
        
        # Configuration
        self.config = {
            "adaptive_factor": 0.1,  # How much to adjust limits based on load
            "ddos_sensitivity": 0.8,  # DDoS detection sensitivity
            "max_adaptive_multiplier": 5.0,  # Maximum adaptive rate limit multiplier
            "min_adaptive_multiplier": 0.1,  # Minimum adaptive rate limit multiplier
            "cleanup_interval_seconds": 300,  # 5 minutes
            "metrics_retention_hours": 24,
            "enable_geolocation_blocking": False,
            "suspicious_user_agents": [
                "bot", "crawler", "spider", "scraper", "harvest"
            ]
        }
        
        # Background task control
        self._cleanup_task: Optional[asyncio.Task] = None
        self._ddos_monitoring_task: Optional[asyncio.Task] = None
    
    async def check_rate_limit(
        self,
        identifier: str,
        request: Optional[Request] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RateLimitResult:
        """
        Check if request is within rate limits.
        
        Args:
            identifier: Client identifier (IP, user ID, API key, etc.)
            request: FastAPI request object for additional context
            context: Additional context data
            
        Returns:
            RateLimitResult with decision and metadata
        """
        try:
            self.metrics["total_requests"] += 1
            
            # Find applicable rule
            applicable_rule = self._find_applicable_rule(request, context)
            
            # Check if IP is blacklisted
            if identifier in applicable_rule.blacklist_ips:
                return RateLimitResult(
                    allowed=False,
                    limit_exceeded=True,
                    current_usage=0,
                    limit=0,
                    reset_time=datetime.utcnow() + timedelta(hours=24),
                    retry_after_seconds=86400,  # 24 hours
                    threat_level=ThreatLevel.CRITICAL,
                    applied_rule=applicable_rule.name,
                    ddos_detected=False
                )
            
            # Check if IP is whitelisted (bypass rate limiting)
            if identifier in applicable_rule.whitelist_ips:
                return RateLimitResult(
                    allowed=True,
                    limit_exceeded=False,
                    current_usage=0,
                    limit=float('inf'),
                    reset_time=datetime.utcnow() + timedelta(hours=1),
                    retry_after_seconds=0,
                    threat_level=ThreatLevel.NONE,
                    applied_rule=applicable_rule.name,
                    ddos_detected=False
                )
            
            # Check for active penalties
            penalty_result = await self._check_penalty(identifier)
            if penalty_result and penalty_result.penalty_active:
                return penalty_result
            
            # Apply rate limiting algorithm
            rate_limit_result = await self._apply_rate_limiting_algorithm(
                identifier, applicable_rule, request, context
            )
            
            # DDoS detection
            if applicable_rule.enable_ddos_protection and self.enable_ddos_protection:
                ddos_result = await self._check_ddos_patterns(identifier, request, context)
                if ddos_result.is_ddos:
                    rate_limit_result.ddos_detected = True
                    rate_limit_result.threat_level = max(
                        rate_limit_result.threat_level, ddos_result.threat_level,
                        key=lambda x: list(ThreatLevel).index(x)
                    )
                    rate_limit_result.allowed = False
                    
                    # Apply emergency blocking
                    await self._apply_emergency_block(identifier, ddos_result)
            
            # Apply progressive penalties if limit exceeded
            if rate_limit_result.limit_exceeded and applicable_rule.enable_progressive_penalties:
                await self._apply_progressive_penalty(identifier, applicable_rule)
                rate_limit_result.penalty_active = True
            
            # Update metrics
            self._update_metrics(rate_limit_result)
            
            # Log significant events
            if rate_limit_result.limit_exceeded or rate_limit_result.ddos_detected:
                logger.warning(
                    "Rate limit exceeded or DDoS detected",
                    identifier=identifier,
                    rule=applicable_rule.name,
                    current_usage=rate_limit_result.current_usage,
                    limit=rate_limit_result.limit,
                    threat_level=rate_limit_result.threat_level.value,
                    ddos_detected=rate_limit_result.ddos_detected
                )
            
            return rate_limit_result
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            
            # Fail-safe: allow request but log error
            return RateLimitResult(
                allowed=True,
                limit_exceeded=False,
                current_usage=0,
                limit=1000,  # Conservative default
                reset_time=datetime.utcnow() + timedelta(minutes=1),
                retry_after_seconds=0,
                threat_level=ThreatLevel.NONE,
                applied_rule="error_fallback",
                ddos_detected=False
            )
    
    async def add_rule(self, rule: RateLimitRule) -> bool:
        """Add a new rate limiting rule."""
        try:
            # Validate rule
            if not self._validate_rule(rule):
                return False
            
            # Check for duplicate names
            if any(r.name == rule.name for r in self.rules):
                raise ValueError(f"Rule with name '{rule.name}' already exists")
            
            self.rules.append(rule)
            
            # Store in Redis for persistence
            await self._store_rule(rule)
            
            logger.info(f"Rate limiting rule added: {rule.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add rate limiting rule: {e}")
            return False
    
    async def remove_rule(self, rule_name: str) -> bool:
        """Remove a rate limiting rule."""
        try:
            # Find and remove rule
            self.rules = [r for r in self.rules if r.name != rule_name]
            
            # Remove from Redis
            await self.redis.delete(f"rule:{rule_name}")
            
            logger.info(f"Rate limiting rule removed: {rule_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove rate limiting rule: {e}")
            return False
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive rate limiting metrics."""
        try:
            # Get current metrics
            current_metrics = self.metrics.copy()
            
            # Add Redis-based metrics
            redis_metrics = await self._get_redis_metrics()
            current_metrics.update(redis_metrics)
            
            # Calculate rates
            total_requests = current_metrics["total_requests"]
            if total_requests > 0:
                current_metrics["block_rate"] = current_metrics["blocked_requests"] / total_requests
                current_metrics["penalty_rate"] = current_metrics["penalties_applied"] / total_requests
                current_metrics["ddos_detection_rate"] = current_metrics["ddos_attacks_detected"] / total_requests
            else:
                current_metrics["block_rate"] = 0.0
                current_metrics["penalty_rate"] = 0.0
                current_metrics["ddos_detection_rate"] = 0.0
            
            # Add configuration
            current_metrics["configuration"] = self.config.copy()
            current_metrics["rules"] = [rule.to_dict() for rule in self.rules]
            
            return current_metrics
            
        except Exception as e:
            logger.error(f"Failed to get rate limiting metrics: {e}")
            return {"error": str(e)}
    
    # Private methods
    
    def _create_default_rules(self) -> List[RateLimitRule]:
        """Create default rate limiting rules."""
        return [
            # General API rate limiting
            RateLimitRule(
                name="general_api",
                requests_per_second=10,
                requests_per_minute=100,
                requests_per_hour=1000,
                burst_capacity=20,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                exclude_paths=["/health", "/metrics"],
                enable_progressive_penalties=True,
                enable_ddos_protection=True
            ),
            
            # Authentication endpoints (more restrictive)
            RateLimitRule(
                name="authentication",
                requests_per_second=2,
                requests_per_minute=10,
                requests_per_hour=50,
                burst_capacity=5,
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                apply_to_paths=["/auth", "/login", "/register"],
                enable_progressive_penalties=True,
                penalty_multiplier=3.0,
                enable_ddos_protection=True
            ),
            
            # Code execution (very restrictive)
            RateLimitRule(
                name="code_execution",
                requests_per_second=1,
                requests_per_minute=5,
                requests_per_hour=20,
                burst_capacity=2,
                algorithm=RateLimitAlgorithm.LEAKY_BUCKET,
                apply_to_paths=["/api/v1/code"],
                enable_progressive_penalties=True,
                penalty_multiplier=5.0,
                enable_ddos_protection=True,
                ddos_threshold_multiplier=2.0
            ),
            
            # File uploads
            RateLimitRule(
                name="file_uploads",
                requests_per_second=1,
                requests_per_minute=10,
                requests_per_hour=100,
                burst_capacity=3,
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                apply_to_methods=["POST", "PUT"],
                enable_progressive_penalties=True,
                enable_ddos_protection=True
            )
        ]
    
    def _find_applicable_rule(
        self, 
        request: Optional[Request] = None, 
        context: Optional[Dict[str, Any]] = None
    ) -> RateLimitRule:
        """Find the most specific applicable rule for the request."""
        
        if not request:
            return self.rules[0]  # Return default rule
        
        path = request.url.path
        method = request.method
        user_agent = request.headers.get("user-agent", "").lower()
        
        # Find matching rules (most specific first)
        matching_rules = []
        
        for rule in self.rules:
            # Check exclusions first
            if rule.exclude_paths and any(path.startswith(excl) for excl in rule.exclude_paths):
                continue
            
            # Check path matching
            if rule.apply_to_paths and not any(path.startswith(p) for p in rule.apply_to_paths):
                continue
            
            # Check method matching
            if rule.apply_to_methods and method not in rule.apply_to_methods:
                continue
            
            # Check user agent matching
            if rule.apply_to_user_agents and not any(ua in user_agent for ua in rule.apply_to_user_agents):
                continue
            
            matching_rules.append(rule)
        
        # Return most specific rule (or default if none match)
        return matching_rules[0] if matching_rules else self.rules[0]
    
    async def _apply_rate_limiting_algorithm(
        self,
        identifier: str,
        rule: RateLimitRule,
        request: Optional[Request] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RateLimitResult:
        """Apply the specified rate limiting algorithm."""
        
        algorithm = rule.algorithm
        self.metrics["algorithm_usage"][algorithm.value] += 1
        
        if algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return await self._fixed_window_limit(identifier, rule)
        elif algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return await self._sliding_window_limit(identifier, rule)
        elif algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return await self._token_bucket_limit(identifier, rule)
        elif algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
            return await self._leaky_bucket_limit(identifier, rule)
        elif algorithm == RateLimitAlgorithm.ADAPTIVE:
            return await self._adaptive_limit(identifier, rule, request, context)
        else:
            # Default to sliding window
            return await self._sliding_window_limit(identifier, rule)
    
    async def _sliding_window_limit(self, identifier: str, rule: RateLimitRule) -> RateLimitResult:
        """Implement sliding window rate limiting."""
        now = time.time()
        window_size = 60  # 1 minute window
        key = f"{self.rate_limit_prefix}sliding:{identifier}:{rule.name}"
        
        # Use Redis sorted set for sliding window
        pipe = self.redis.redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, now - window_size)
        
        # Count current entries
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(uuid.uuid4()): now})
        
        # Set expiration
        pipe.expire(key, int(window_size) + 10)
        
        results = await pipe.execute()
        current_count = results[1]
        
        # Check limits
        limit = rule.requests_per_minute
        reset_time = datetime.fromtimestamp(now + window_size - (now % window_size))
        
        if current_count > limit:
            return RateLimitResult(
                allowed=False,
                limit_exceeded=True,
                current_usage=current_count,
                limit=limit,
                reset_time=reset_time,
                retry_after_seconds=int(window_size - (now % window_size)),
                threat_level=self._calculate_threat_level(current_count, limit),
                applied_rule=rule.name
            )
        
        return RateLimitResult(
            allowed=True,
            limit_exceeded=False,
            current_usage=current_count,
            limit=limit,
            reset_time=reset_time,
            retry_after_seconds=0,
            threat_level=ThreatLevel.NONE,
            applied_rule=rule.name
        )
    
    async def _token_bucket_limit(self, identifier: str, rule: RateLimitRule) -> RateLimitResult:
        """Implement token bucket rate limiting."""
        now = time.time()
        key = f"{self.rate_limit_prefix}bucket:{identifier}:{rule.name}"
        
        # Get current bucket state
        bucket_data = await self.redis.get(key)
        
        if bucket_data:
            bucket_state = json.loads(bucket_data.decode('utf-8'))
            tokens = bucket_state["tokens"]
            last_refill = bucket_state["last_refill"]
        else:
            tokens = rule.burst_capacity
            last_refill = now
        
        # Calculate tokens to add based on time elapsed
        time_elapsed = now - last_refill
        tokens_to_add = time_elapsed * (rule.requests_per_second)
        tokens = min(rule.burst_capacity, tokens + tokens_to_add)
        
        # Check if request can be served
        if tokens >= 1:
            tokens -= 1
            allowed = True
            limit_exceeded = False
        else:
            allowed = False
            limit_exceeded = True
        
        # Store updated state
        bucket_state = {
            "tokens": tokens,
            "last_refill": now
        }
        await self.redis.set_with_expiry(key, json.dumps(bucket_state), 3600)
        
        # Calculate reset time
        if tokens < 1:
            time_to_refill = (1 - tokens) / rule.requests_per_second
            reset_time = datetime.fromtimestamp(now + time_to_refill)
            retry_after = int(time_to_refill) + 1
        else:
            reset_time = datetime.fromtimestamp(now + 60)
            retry_after = 0
        
        current_usage = rule.burst_capacity - int(tokens)
        
        return RateLimitResult(
            allowed=allowed,
            limit_exceeded=limit_exceeded,
            current_usage=current_usage,
            limit=rule.burst_capacity,
            reset_time=reset_time,
            retry_after_seconds=retry_after,
            threat_level=self._calculate_threat_level(current_usage, rule.burst_capacity),
            applied_rule=rule.name
        )
    
    async def _leaky_bucket_limit(self, identifier: str, rule: RateLimitRule) -> RateLimitResult:
        """Implement leaky bucket rate limiting."""
        now = time.time()
        key = f"{self.rate_limit_prefix}leaky:{identifier}:{rule.name}"
        
        # Get current bucket state
        bucket_data = await self.redis.get(key)
        
        if bucket_data:
            bucket_state = json.loads(bucket_data.decode('utf-8'))
            volume = bucket_state["volume"]
            last_leak = bucket_state["last_leak"]
        else:
            volume = 0
            last_leak = now
        
        # Calculate leak (outflow)
        time_elapsed = now - last_leak
        leak_amount = time_elapsed * rule.requests_per_second
        volume = max(0, volume - leak_amount)
        
        # Check if request can be added
        if volume < rule.burst_capacity:
            volume += 1
            allowed = True
            limit_exceeded = False
        else:
            allowed = False
            limit_exceeded = True
        
        # Store updated state
        bucket_state = {
            "volume": volume,
            "last_leak": now
        }
        await self.redis.set_with_expiry(key, json.dumps(bucket_state), 3600)
        
        # Calculate reset time
        if volume >= rule.burst_capacity:
            time_to_leak = (volume - rule.burst_capacity + 1) / rule.requests_per_second
            reset_time = datetime.fromtimestamp(now + time_to_leak)
            retry_after = int(time_to_leak) + 1
        else:
            reset_time = datetime.fromtimestamp(now + 60)
            retry_after = 0
        
        return RateLimitResult(
            allowed=allowed,
            limit_exceeded=limit_exceeded,
            current_usage=int(volume),
            limit=rule.burst_capacity,
            reset_time=reset_time,
            retry_after_seconds=retry_after,
            threat_level=self._calculate_threat_level(int(volume), rule.burst_capacity),
            applied_rule=rule.name
        )
    
    async def _fixed_window_limit(self, identifier: str, rule: RateLimitRule) -> RateLimitResult:
        """Implement fixed window rate limiting."""
        now = time.time()
        window_start = int(now // 60) * 60  # 1-minute windows
        key = f"{self.rate_limit_prefix}fixed:{identifier}:{rule.name}:{window_start}"
        
        # Increment counter
        current_count = await self.redis.increment_counter(key, 60)
        
        limit = rule.requests_per_minute
        reset_time = datetime.fromtimestamp(window_start + 60)
        
        if current_count > limit:
            return RateLimitResult(
                allowed=False,
                limit_exceeded=True,
                current_usage=current_count,
                limit=limit,
                reset_time=reset_time,
                retry_after_seconds=int(window_start + 60 - now),
                threat_level=self._calculate_threat_level(current_count, limit),
                applied_rule=rule.name
            )
        
        return RateLimitResult(
            allowed=True,
            limit_exceeded=False,
            current_usage=current_count,
            limit=limit,
            reset_time=reset_time,
            retry_after_seconds=0,
            threat_level=ThreatLevel.NONE,
            applied_rule=rule.name
        )
    
    async def _adaptive_limit(
        self,
        identifier: str,
        rule: RateLimitRule,
        request: Optional[Request] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> RateLimitResult:
        """Implement adaptive rate limiting based on system load and behavior."""
        
        # Get system load metrics (simplified - would integrate with system monitoring)
        system_load = await self._get_system_load()
        
        # Calculate adaptive multiplier
        if system_load > 0.8:  # High load
            multiplier = self.config["min_adaptive_multiplier"]
        elif system_load < 0.3:  # Low load
            multiplier = self.config["max_adaptive_multiplier"]
        else:
            multiplier = 1.0 - (system_load - 0.3) * self.config["adaptive_factor"]
        
        # Apply multiplier to limits
        adapted_rule = RateLimitRule(
            name=f"{rule.name}_adaptive",
            requests_per_second=int(rule.requests_per_second * multiplier),
            requests_per_minute=int(rule.requests_per_minute * multiplier),
            requests_per_hour=int(rule.requests_per_hour * multiplier),
            burst_capacity=int(rule.burst_capacity * multiplier),
            algorithm=RateLimitAlgorithm.SLIDING_WINDOW  # Use sliding window for adaptive
        )
        
        self.metrics["adaptive_adjustments"] += 1
        
        # Apply sliding window with adapted limits
        return await self._sliding_window_limit(identifier, adapted_rule)
    
    async def _check_penalty(self, identifier: str) -> Optional[RateLimitResult]:
        """Check if identifier is under penalty."""
        key = f"{self.penalty_prefix}{identifier}"
        penalty_data = await self.redis.get(key)
        
        if penalty_data:
            penalty_info = json.loads(penalty_data.decode('utf-8'))
            expires_at = datetime.fromisoformat(penalty_info["expires_at"])
            
            if datetime.utcnow() < expires_at:
                return RateLimitResult(
                    allowed=False,
                    limit_exceeded=True,
                    current_usage=0,
                    limit=0,
                    reset_time=expires_at,
                    retry_after_seconds=int((expires_at - datetime.utcnow()).total_seconds()),
                    threat_level=ThreatLevel(penalty_info["threat_level"]),
                    applied_rule=penalty_info["rule"],
                    penalty_active=True,
                    penalty_expires_at=expires_at
                )
            else:
                # Penalty expired, clean up
                await self.redis.delete(key)
        
        return None
    
    async def _apply_progressive_penalty(self, identifier: str, rule: RateLimitRule) -> None:
        """Apply progressive penalty for repeated violations."""
        penalty_key = f"{self.penalty_prefix}{identifier}"
        violations_key = f"{self.penalty_prefix}violations:{identifier}"
        
        # Get violation count
        violations = await self.redis.increment_counter(violations_key, 3600)  # 1 hour window
        
        # Calculate penalty duration (exponential backoff)
        base_penalty_minutes = 5
        penalty_minutes = min(
            base_penalty_minutes * (rule.penalty_multiplier ** (violations - 1)),
            rule.max_penalty_duration_minutes
        )
        
        expires_at = datetime.utcnow() + timedelta(minutes=penalty_minutes)
        
        # Store penalty
        penalty_data = {
            "violations": violations,
            "expires_at": expires_at.isoformat(),
            "rule": rule.name,
            "threat_level": ThreatLevel.MEDIUM.value
        }
        
        await self.redis.set_with_expiry(
            penalty_key,
            json.dumps(penalty_data),
            int(penalty_minutes * 60)
        )
        
        self.metrics["penalties_applied"] += 1
        
        logger.warning(
            f"Progressive penalty applied to {identifier}",
            violations=violations,
            penalty_minutes=penalty_minutes,
            rule=rule.name
        )
    
    async def _check_ddos_patterns(
        self,
        identifier: str,
        request: Optional[Request] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> DDoSDetectionResult:
        """Detect DDoS attack patterns."""
        
        # Analyze request patterns
        now = time.time()
        window_minutes = 5
        detection_key = f"{self.ddos_detection_prefix}{identifier}"
        
        # Get recent request timestamps
        pipe = self.redis.redis.pipeline()
        pipe.zremrangebyscore(detection_key, 0, now - (window_minutes * 60))
        pipe.zcard(detection_key)
        pipe.zadd(detection_key, {str(uuid.uuid4()): now})
        pipe.expire(detection_key, window_minutes * 60)
        
        results = await pipe.execute()
        request_count = results[1]
        
        # Calculate request rate
        request_rate = request_count / window_minutes
        
        # Anomaly detection
        anomaly_indicators = []
        confidence_score = 0.0
        
        # High request rate
        if request_rate > 100:  # More than 100 requests per minute
            anomaly_indicators.append("high_request_rate")
            confidence_score += 0.3
        
        # Burst pattern detection
        if request_count > 50:  # 50 requests in 5 minutes
            anomaly_indicators.append("burst_pattern")
            confidence_score += 0.2
        
        # Add more sophisticated detection logic here:
        # - User agent analysis
        # - Geographic distribution
        # - Request pattern analysis
        # - Payload analysis
        
        if request and request.headers.get("user-agent"):
            user_agent = request.headers["user-agent"].lower()
            if any(suspicious in user_agent for suspicious in self.config["suspicious_user_agents"]):
                anomaly_indicators.append("suspicious_user_agent")
                confidence_score += 0.2
        
        # Determine threat level and action
        if confidence_score >= 0.8:
            threat_level = ThreatLevel.CRITICAL
            is_ddos = True
            recommended_action = "immediate_block"
        elif confidence_score >= 0.6:
            threat_level = ThreatLevel.HIGH  
            is_ddos = True
            recommended_action = "temporary_block"
        elif confidence_score >= 0.4:
            threat_level = ThreatLevel.MEDIUM
            is_ddos = False
            recommended_action = "enhanced_monitoring"
        else:
            threat_level = ThreatLevel.LOW
            is_ddos = False
            recommended_action = "continue_monitoring"
        
        return DDoSDetectionResult(
            is_ddos=is_ddos,
            threat_level=threat_level,
            confidence_score=confidence_score,
            attack_pattern="volumetric" if request_rate > 100 else "burst",
            source_ips=[identifier],
            request_rate=request_rate,
            anomaly_indicators=anomaly_indicators,
            recommended_action=recommended_action
        )
    
    async def _apply_emergency_block(self, identifier: str, ddos_result: DDoSDetectionResult) -> None:
        """Apply emergency blocking for DDoS attacks."""
        block_duration_minutes = 60  # 1 hour
        
        if ddos_result.threat_level == ThreatLevel.CRITICAL:
            block_duration_minutes = 240  # 4 hours
        elif ddos_result.threat_level == ThreatLevel.HIGH:
            block_duration_minutes = 120  # 2 hours
        
        block_key = f"{self.penalty_prefix}emergency:{identifier}"
        block_data = {
            "blocked_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(minutes=block_duration_minutes)).isoformat(),
            "reason": "ddos_detected",
            "threat_level": ddos_result.threat_level.value,
            "confidence_score": ddos_result.confidence_score,
            "attack_pattern": ddos_result.attack_pattern
        }
        
        await self.redis.set_with_expiry(
            block_key,
            json.dumps(block_data),
            block_duration_minutes * 60
        )
        
        self.metrics["ddos_attacks_detected"] += 1
        
        logger.critical(
            f"Emergency DDoS block applied to {identifier}",
            threat_level=ddos_result.threat_level.value,
            confidence_score=ddos_result.confidence_score,
            block_duration_minutes=block_duration_minutes
        )
    
    def _calculate_threat_level(self, current_usage: int, limit: int) -> ThreatLevel:
        """Calculate threat level based on usage vs limit."""
        ratio = current_usage / max(limit, 1)
        
        if ratio >= 5.0:
            return ThreatLevel.CRITICAL
        elif ratio >= 3.0:
            return ThreatLevel.HIGH
        elif ratio >= 2.0:
            return ThreatLevel.MEDIUM
        elif ratio >= 1.5:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.NONE
    
    async def _get_system_load(self) -> float:
        """Get current system load (simplified implementation)."""
        # In production, this would integrate with system monitoring
        # For now, return a mock value based on current metrics
        try:
            total_requests = self.metrics["total_requests"]
            blocked_requests = self.metrics["blocked_requests"]
            
            if total_requests > 0:
                return min(1.0, blocked_requests / total_requests * 2)
            return 0.1
        except:
            return 0.5  # Default moderate load
    
    def _validate_rule(self, rule: RateLimitRule) -> bool:
        """Validate rate limiting rule configuration."""
        if rule.requests_per_second <= 0:
            return False
        if rule.requests_per_minute <= 0:
            return False
        if rule.requests_per_hour <= 0:
            return False
        if rule.burst_capacity <= 0:
            return False
        
        # Logical consistency checks
        if rule.requests_per_minute > rule.requests_per_second * 60:
            return False
        if rule.requests_per_hour > rule.requests_per_minute * 60:
            return False
        
        return True
    
    async def _store_rule(self, rule: RateLimitRule) -> None:
        """Store rule in Redis for persistence."""
        key = f"rule:{rule.name}"
        await self.redis.set(key, json.dumps(rule.to_dict()))
    
    async def _get_redis_metrics(self) -> Dict[str, Any]:
        """Get metrics from Redis."""
        try:
            # Get active penalties count
            penalty_keys = await self.redis.scan_pattern(f"{self.penalty_prefix}*")
            active_penalties = len(penalty_keys)
            
            # Get active rate limit entries
            rate_limit_keys = await self.redis.scan_pattern(f"{self.rate_limit_prefix}*")
            active_rate_limits = len(rate_limit_keys)
            
            return {
                "active_penalties": active_penalties,
                "active_rate_limits": active_rate_limits,
                "redis_key_count": active_penalties + active_rate_limits
            }
        except Exception as e:
            logger.debug(f"Failed to get Redis metrics: {e}")
            return {}
    
    def _update_metrics(self, result: RateLimitResult) -> None:
        """Update internal metrics."""
        if result.limit_exceeded:
            self.metrics["blocked_requests"] += 1
        
        self.metrics["threat_levels"][result.threat_level.value] += 1


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""
    
    def __init__(self, app, rate_limiter: AdvancedRateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter
    
    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiter."""
        
        # Skip rate limiting for certain paths
        skip_paths = {"/health", "/metrics", "/docs", "/redoc", "/openapi.json"}
        if request.url.path in skip_paths:
            return await call_next(request)
        
        # Get client identifier
        client_ip = self._get_client_ip(request)
        
        # Check rate limits
        result = await self.rate_limiter.check_rate_limit(client_ip, request)
        
        if not result.allowed:
            # Return rate limit exceeded response
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Rate limit exceeded. Please try again later.",
                    "limit": result.limit,
                    "current_usage": result.current_usage,
                    "reset_time": result.reset_time.isoformat(),
                    "retry_after": result.retry_after_seconds,
                    "threat_level": result.threat_level.value
                },
                headers={
                    "Retry-After": str(result.retry_after_seconds),
                    "X-RateLimit-Limit": str(result.limit),
                    "X-RateLimit-Remaining": str(max(0, result.limit - result.current_usage)),
                    "X-RateLimit-Reset": str(int(result.reset_time.timestamp())),
                    "X-RateLimit-Rule": result.applied_rule
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to successful responses
        response.headers["X-RateLimit-Limit"] = str(result.limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, result.limit - result.current_usage))
        response.headers["X-RateLimit-Reset"] = str(int(result.reset_time.timestamp()))
        response.headers["X-RateLimit-Rule"] = result.applied_rule
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        if request.client:
            return request.client.host
        
        return "unknown"


# Factory functions
async def create_advanced_rate_limiter(redis_client: RedisClient) -> AdvancedRateLimiter:
    """Create advanced rate limiter instance."""
    return AdvancedRateLimiter(redis_client)


def create_rate_limit_middleware(rate_limiter: AdvancedRateLimiter) -> RateLimitMiddleware:
    """Create rate limit middleware instance."""
    return RateLimitMiddleware(None, rate_limiter)  # App will be set by FastAPI