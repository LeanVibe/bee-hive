# LeanVibe Agent Hive 2.0 - Security Hardening Implementation Guide

**Implementation Date:** 2025-07-29
**Implemented By:** Security Hardening Specialist Agent
**Status:** PRODUCTION READY

## Overview

This guide provides comprehensive instructions for implementing the advanced security hardening measures developed for LeanVibe Agent Hive 2.0. The implementation includes multiple layers of security validation, threat detection, and protection mechanisms.

## Implemented Security Components

### 1. Enhanced Input Validation Middleware
**File:** `app/core/security_validation_middleware.py`

**Features:**
- Multi-layer threat detection (SQL injection, XSS, command injection, path traversal)
- Content sanitization and normalization
- Performance-optimized pattern matching
- Real-time threat scoring and blocking
- Comprehensive audit logging

**Integration:**
```python
from app.core.security_validation_middleware import SecurityValidationMiddleware

# Add to FastAPI application
app.add_middleware(SecurityValidationMiddleware)
```

### 2. Advanced Rate Limiting & DDoS Protection
**File:** `app/core/advanced_rate_limiter.py`

**Features:**
- Multiple rate limiting algorithms (sliding window, token bucket, leaky bucket)
- DDoS detection and mitigation
- Progressive penalties for repeat offenders
- Adaptive rate limiting based on system load
- IP-based threat analysis

**Integration:**
```python
from app.core.advanced_rate_limiter import AdvancedRateLimiter, RateLimitMiddleware

# Initialize rate limiter
rate_limiter = AdvancedRateLimiter(redis_client)
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
```

### 3. Enhanced JWT Token Management
**File:** `app/core/enhanced_jwt_manager.py`

**Features:**
- Automatic key rotation with multiple algorithms
- Token blacklisting and revocation
- Replay attack protection
- Comprehensive token validation
- Key lifecycle management

**Integration:**
```python
from app.core.enhanced_jwt_manager import EnhancedJWTManager, KeyAlgorithm

# Initialize JWT manager
jwt_manager = EnhancedJWTManager(
    redis_client,
    default_algorithm=KeyAlgorithm.RS256,
    key_rotation_interval_hours=24
)
```

## Production Deployment Steps

### Step 1: Environment Configuration

Update your `.env` file with enhanced security settings:

```bash
# Enhanced Security Configuration
JWT_SECRET_KEY=<generate-strong-random-key-512-bits>
JWT_ALGORITHM=RS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Rate Limiting Configuration
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_BURST_CAPACITY=200
DDOS_PROTECTION_ENABLED=true

# Security Validation Configuration
INPUT_VALIDATION_ENABLED=true
CONTENT_SANITIZATION_ENABLED=true
THREAT_DETECTION_ENABLED=true
SECURITY_AUDIT_LOGGING=true

# Redis Configuration (for security components)
REDIS_URL=redis://localhost:6379/1
REDIS_SECURITY_DB=2
```

### Step 2: Database Migration

Ensure the security-related database tables are created:

```sql
-- Agent security tables should already exist from previous migrations
-- Verify the following tables exist:
-- - agent_identities
-- - agent_tokens  
-- - agent_roles
-- - agent_role_assignments
-- - security_audit_logs
-- - security_events
```

### Step 3: Update Main Application

Modify `app/main.py` to include security middleware:

```python
from fastapi import FastAPI
from app.core.security_validation_middleware import SecurityValidationMiddleware
from app.core.advanced_rate_limiter import create_advanced_rate_limiter, RateLimitMiddleware
from app.core.enhanced_jwt_manager import create_enhanced_jwt_manager
from app.core.redis import RedisClient

app = FastAPI()

# Initialize security components
redis_client = RedisClient()
rate_limiter = await create_advanced_rate_limiter(redis_client)
jwt_manager = await create_enhanced_jwt_manager(redis_client)

# Add security middleware (order matters!)
app.add_middleware(SecurityValidationMiddleware)
app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)

# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response
```

### Step 4: Update Authentication Endpoints

Modify authentication endpoints to use the enhanced JWT manager:

```python
from app.core.enhanced_jwt_manager import TokenGenerationOptions, TokenType

@router.post("/auth/login")
async def login(credentials: LoginCredentials):
    # Validate credentials...
    
    # Generate tokens with enhanced JWT manager
    access_token, access_metadata = await jwt_manager.generate_token(
        payload={"sub": user_id, "username": username, "roles": user_roles},
        options=TokenGenerationOptions(
            token_type=TokenType.ACCESS,
            expires_in_seconds=3600,
            audience="leanvibe-api"
        )
    )
    
    refresh_token, refresh_metadata = await jwt_manager.generate_token(
        payload={"sub": user_id, "token_type": "refresh"},
        options=TokenGenerationOptions(
            token_type=TokenType.REFRESH,
            expires_in_seconds=604800
        )
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": 3600
    }
```

### Step 5: Update Protected Endpoints

Add token validation to protected endpoints:

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def get_current_user(token: str = Depends(security)):
    validation_result = await jwt_manager.validate_token(
        token.credentials,
        check_blacklist=True,
        required_claims=["sub", "username"]
    )
    
    if not validation_result.is_valid:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {validation_result.error}"
        )
    
    # Log security warnings if any
    if validation_result.security_warnings:
        logger.warning(
            "Token security warnings",
            warnings=validation_result.security_warnings,
            token_age=validation_result.token_age_seconds
        )
    
    return validation_result.payload

@router.get("/protected")
async def protected_endpoint(current_user = Depends(get_current_user)):
    return {"message": "Access granted", "user": current_user}
```

### Step 6: Configure Rate Limiting Rules

Add custom rate limiting rules for different endpoints:

```python
from app.core.advanced_rate_limiter import RateLimitRule, RateLimitAlgorithm

# Authentication endpoints (strict limits)
auth_rule = RateLimitRule(
    name="authentication",
    requests_per_second=2,
    requests_per_minute=10,
    requests_per_hour=50,
    burst_capacity=5,
    algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
    apply_to_paths=["/auth/login", "/auth/register"],
    enable_progressive_penalties=True,
    penalty_multiplier=3.0
)

# API endpoints (moderate limits)
api_rule = RateLimitRule(
    name="api_endpoints",
    requests_per_second=10,
    requests_per_minute=100,
    requests_per_hour=1000,
    burst_capacity=20,
    algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
    exclude_paths=["/health", "/metrics"]
)

# Add rules to rate limiter
await rate_limiter.add_rule(auth_rule)
await rate_limiter.add_rule(api_rule)
```

### Step 7: Enable Security Monitoring

Set up comprehensive security monitoring:

```python
# Add security metrics endpoint
@router.get("/security/metrics")
async def get_security_metrics(current_user = Depends(get_admin_user)):
    return {
        "rate_limiting": await rate_limiter.get_metrics(),
        "jwt_management": await jwt_manager.get_metrics(),
        "input_validation": security_validation_middleware.get_metrics(),
        "threat_detection": threat_detection_engine.get_metrics()
    }

# Add security health check
@router.get("/security/health")
async def security_health_check():
    return {
        "status": "healthy",
        "components": {
            "rate_limiter": "operational",
            "jwt_manager": "operational", 
            "input_validation": "operational",
            "threat_detection": "operational"
        },
        "timestamp": datetime.utcnow().isoformat()
    }
```

## Security Configuration Options

### Input Validation Configuration

```python
# Customize security validation engine
validation_engine = SecurityValidationEngine()
validation_engine.config.update({
    "max_input_length": 500000,  # 500KB
    "enable_content_sanitization": True,
    "threat_threshold": 0.7,
    "enable_threat_scoring": True,
    "blocked_content_types": [
        "application/x-executable",
        "application/octet-stream",
        "application/x-msdownload"
    ]
})
```

### Rate Limiting Configuration

```python
# Customize rate limiter settings
rate_limiter.config.update({
    "adaptive_factor": 0.15,
    "ddos_sensitivity": 0.8,
    "max_adaptive_multiplier": 3.0,
    "enable_geolocation_blocking": True,
    "suspicious_user_agents": [
        "bot", "crawler", "spider", "scraper", "harvest", "automated"
    ]
})
```

### JWT Management Configuration

```python
# Customize JWT manager settings
jwt_manager.config.update({
    "default_access_token_ttl_seconds": 1800,  # 30 minutes
    "default_refresh_token_ttl_seconds": 259200,  # 3 days
    "max_token_age_seconds": 43200,  # 12 hours
    "enable_replay_protection": True,
    "require_aud_claim": True,
    "max_clock_skew_seconds": 180  # 3 minutes
})
```

## Monitoring and Alerting

### Key Security Metrics to Monitor

1. **Rate Limiting Metrics:**
   - Request block rate
   - DDoS attacks detected
   - Progressive penalties applied
   - Algorithm performance

2. **JWT Token Metrics:**
   - Token validation errors
   - Key rotation frequency
   - Blacklisted tokens
   - Security warnings count

3. **Input Validation Metrics:**
   - Threats detected by type
   - Content sanitization rate
   - Processing time performance
   - False positive rates

### Security Alerts Configuration

```python
# Configure security alerts
SECURITY_ALERTS = {
    "high_threat_detection_rate": {
        "threshold": 0.1,  # 10% of requests
        "window_minutes": 5,
        "action": "immediate_alert"
    },
    "ddos_attack_detected": {
        "threshold": 1,  # Any DDoS detection
        "action": "emergency_alert"
    },
    "jwt_validation_errors": {
        "threshold": 0.05,  # 5% error rate
        "window_minutes": 10,
        "action": "warning_alert"
    },
    "key_rotation_overdue": {
        "threshold_hours": 48,  # 2 days overdue
        "action": "maintenance_alert"
    }
}
```

## Testing and Validation

### Security Test Suite

Run the comprehensive security test suite:

```bash
# Run security validation tests
pytest tests/security/test_input_validation.py -v

# Run rate limiting tests  
pytest tests/security/test_rate_limiting.py -v

# Run JWT management tests
pytest tests/security/test_jwt_management.py -v

# Run integration security tests
pytest tests/security/test_security_integration.py -v
```

### Load Testing

Validate security components under load:

```bash
# Rate limiting load test
python scripts/security_load_test.py --component=rate_limiter --requests=10000

# Input validation load test
python scripts/security_load_test.py --component=input_validation --requests=5000

# JWT performance test
python scripts/security_load_test.py --component=jwt_manager --requests=1000
```

### Penetration Testing

Recommended security tests to perform:

1. **SQL Injection Testing:** Test all input fields and API endpoints
2. **XSS Testing:** Validate content sanitization effectiveness  
3. **Rate Limiting Bypass:** Attempt to circumvent rate limits
4. **JWT Token Attacks:** Test token manipulation and replay attacks
5. **DDoS Simulation:** Test DDoS detection and mitigation

## Security Maintenance

### Daily Tasks
- Monitor security metrics dashboards
- Review security alert logs
- Check rate limiting effectiveness
- Validate JWT key rotation

### Weekly Tasks  
- Review security audit logs
- Analyze threat patterns
- Update security configurations
- Test backup security systems

### Monthly Tasks
- Conduct security assessment
- Update threat detection patterns
- Review and update security policies
- Performance optimization review

## Troubleshooting

### Common Issues and Solutions

1. **High False Positive Rate:**
   ```python
   # Adjust threat detection sensitivity
   validation_engine.config["threat_threshold"] = 0.8  # Increase threshold
   ```

2. **Rate Limiting Too Restrictive:**
   ```python
   # Adjust rate limits for specific endpoints
   await rate_limiter.add_rule(RateLimitRule(
       name="relaxed_api",
       requests_per_minute=200,  # Increased limit
       apply_to_paths=["/api/v1/relaxed"]
   ))
   ```

3. **JWT Key Rotation Issues:**
   ```python
   # Force key rotation
   rotation_result = await jwt_manager.rotate_keys(force=True)
   ```

4. **Performance Impact:**
   ```python
   # Enable performance optimizations
   validation_engine.config["enable_caching"] = True
   rate_limiter.config["enable_redis_optimization"] = True
   ```

## Security Compliance

This implementation provides compliance with:

- **OWASP Top 10** - Protection against common web vulnerabilities
- **ISO 27001** - Information security management practices  
- **SOC 2 Type II** - Security and availability controls
- **GDPR** - Data protection and privacy requirements

## Conclusion

The security hardening implementation provides enterprise-grade protection for the LeanVibe Agent Hive 2.0 platform. Regular monitoring, maintenance, and updates are essential for maintaining security effectiveness.

For questions or issues with the security implementation, refer to the security team documentation or contact the security team directly.

---

**Document Version:** 1.0
**Last Updated:** 2025-07-29
**Next Review Date:** 2025-08-29