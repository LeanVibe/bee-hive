# Enterprise Security System Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the enhanced LeanVibe Agent Hive security system with enterprise-grade OAuth 2.0/OIDC integration, RBAC, audit logging, and API security features.

## Prerequisites

### System Requirements
- Python 3.11+
- PostgreSQL 13+ with pgvector extension
- Redis 6.0+
- Docker and Docker Compose (recommended)
- SSL/TLS certificates for production

### Dependencies
All required dependencies are included in `pyproject.toml`:
- `authlib>=1.2.1` - OAuth 2.0/OIDC support
- `oauthlib>=3.2.2` - OAuth utilities
- `cryptography>=41.0.7` - Cryptographic operations
- `python-jose[cryptography]>=3.3.0` - JWT handling
- `passlib[bcrypt]>=1.7.4` - Password hashing

## Security Components

### 1. OAuth 2.0/OIDC Provider System
**Location**: `app/core/oauth_provider_system.py`

**Features**:
- Google, GitHub, Microsoft, Azure AD integration
- Custom OIDC provider support
- PKCE (Proof Key for Code Exchange) support
- Token refresh and validation
- Multi-tenant support

### 2. API Security Middleware
**Location**: `app/core/api_security_middleware.py`

**Features**:
- Intelligent rate limiting (per user/IP/endpoint)
- SQL injection and XSS detection
- Request/response validation
- Security headers (HSTS, CSP, X-Frame-Options)
- Real-time threat detection

### 3. Comprehensive Audit System
**Location**: `app/core/comprehensive_audit_system.py`

**Features**:
- SOC 2, ISO 27001, NIST compliance support
- Real-time audit event logging
- Integrity verification with HMAC signatures
- Automated compliance reporting
- Event correlation and analysis

### 4. Enhanced Authorization Engine
**Location**: `app/core/authorization_engine.py`

**Features**:
- Role-based access control (RBAC)
- Fine-grained permission checking
- Resource pattern matching
- Condition-based access control
- Performance-optimized caching

## Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```env
# Database Configuration
DATABASE_URL=postgresql+asyncpg://leanvibe_user:password@localhost:5432/leanvibe_agent_hive

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-here
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60

# OAuth 2.0 Configuration
# Google OAuth
GOOGLE_OAUTH_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_OAUTH_CLIENT_SECRET=GOCSPX-your-google-client-secret

# GitHub OAuth
GITHUB_OAUTH_CLIENT_ID=your-github-client-id
GITHUB_OAUTH_CLIENT_SECRET=your-github-client-secret

# Microsoft OAuth
MICROSOFT_OAUTH_CLIENT_ID=your-microsoft-client-id
MICROSOFT_OAUTH_CLIENT_SECRET=your-microsoft-client-secret
MICROSOFT_TENANT_ID=your-tenant-id

# Security Configuration
SECURITY_AUDIT_INTEGRITY_KEY=your-audit-integrity-key
SECURITY_RATE_LIMIT_DEFAULT=100
SECURITY_MAX_REQUEST_SIZE=10485760

# Compliance Configuration
COMPLIANCE_FRAMEWORKS=soc2,iso27001,nist
COMPLIANCE_RETENTION_DAYS=2555

# Application Configuration
BASE_URL=https://your-domain.com
ENVIRONMENT=production
```

### OAuth Provider Setup

#### Google OAuth 2.0
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google+ API
4. Go to Credentials → Create Credentials → OAuth 2.0 Client IDs
5. Set authorized redirect URIs: `https://your-domain.com/api/v1/oauth/callback/google`

#### GitHub OAuth
1. Go to GitHub Settings → Developer settings → OAuth Apps
2. Click "New OAuth App"
3. Set Authorization callback URL: `https://your-domain.com/api/v1/oauth/callback/github`

#### Microsoft OAuth
1. Go to [Azure Portal](https://portal.azure.com/)
2. Navigate to Azure Active Directory → App registrations
3. Click "New registration"
4. Set Redirect URI: `https://your-domain.com/api/v1/oauth/callback/microsoft`

## Database Setup

### 1. Install PostgreSQL with pgvector

```bash
# Ubuntu/Debian
sudo apt-get install postgresql-13 postgresql-contrib
sudo apt-get install postgresql-13-pgvector

# Or using Docker
docker run -d \
  --name postgres-leanvibe \
  -e POSTGRES_DB=leanvibe_agent_hive \
  -e POSTGRES_USER=leanvibe_user \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  pgvector/pgvector:pg13
```

### 2. Run Database Migrations

```bash
# Install dependencies
pip install -e .

# Run migrations
alembic upgrade head
```

### 3. Create Initial Roles and Users

```python
# Run this script to create initial system roles
from app.core.authorization_engine import create_authorization_engine
from app.core.database import get_session
from app.core.redis import get_redis_client

async def setup_initial_roles():
    async with get_session() as db:
        redis = await get_redis_client()
        auth_engine = await create_authorization_engine(db, redis)
        
        # Create system admin role
        await auth_engine.create_role(
            role_name="system_admin",
            permissions={
                "resources": ["*"],
                "actions": ["*"]
            },
            created_by="system",
            description="Full system administrator access",
            max_access_level="admin",
            can_delegate=True
        )
        
        # Create developer role
        await auth_engine.create_role(
            role_name="developer",
            permissions={
                "resources": ["github", "files", "workspaces"],
                "actions": ["read", "write", "create"]
            },
            created_by="system",
            description="Developer access to code and files",
            resource_patterns=["github/repos/org/*", "files/workspace/*"]
        )
        
        # Create read-only role
        await auth_engine.create_role(
            role_name="readonly",
            permissions={
                "resources": ["*"],
                "actions": ["read"]
            },
            created_by="system",
            description="Read-only access to all resources"
        )

# Run the setup
import asyncio
asyncio.run(setup_initial_roles())
```

## Deployment

### Using Docker Compose (Recommended)

Create `docker-compose.security.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg13
    environment:
      POSTGRES_DB: leanvibe_agent_hive
      POSTGRES_USER: leanvibe_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  app:
    build: .
    environment:
      - DATABASE_URL=postgresql+asyncpg://leanvibe_user:${POSTGRES_PASSWORD}@postgres:5432/leanvibe_agent_hive
      - REDIS_URL=redis://redis:6379/0
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - GOOGLE_OAUTH_CLIENT_ID=${GOOGLE_OAUTH_CLIENT_ID}
      - GOOGLE_OAUTH_CLIENT_SECRET=${GOOGLE_OAUTH_CLIENT_SECRET}
      - GITHUB_OAUTH_CLIENT_ID=${GITHUB_OAUTH_CLIENT_ID}
      - GITHUB_OAUTH_CLIENT_SECRET=${GITHUB_OAUTH_CLIENT_SECRET}
      - MICROSOFT_OAUTH_CLIENT_ID=${MICROSOFT_OAUTH_CLIENT_ID}
      - MICROSOFT_OAUTH_CLIENT_SECRET=${MICROSOFT_OAUTH_CLIENT_SECRET}
      - BASE_URL=${BASE_URL}
      - ENVIRONMENT=production
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000

volumes:
  postgres_data:
  redis_data:
```

Deploy with:
```bash
docker-compose -f docker-compose.security.yml up -d
```

### Manual Deployment

1. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

2. **Start Services**:
   ```bash
   # Start PostgreSQL and Redis
   sudo systemctl start postgresql redis

   # Run database migrations
   alembic upgrade head

   # Start the application
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --loop astral
   ```

## Security Configuration

### 1. Rate Limiting Configuration

```python
# In app/core/api_security_middleware.py
security_config = SecurityConfig(
    enable_rate_limiting=True,
    default_rate_limit=100,  # requests per minute
    burst_multiplier=1.5,
    rate_limit_strategy=RateLimitStrategy.SLIDING_WINDOW,
    
    # Per endpoint limits
    rate_limit_rules=[
        RateLimitRule(
            key_pattern="auth",
            limit=10,  # Auth endpoints more restrictive
            window_seconds=60,
            per_ip=True
        ),
        RateLimitRule(
            key_pattern="api",
            limit=100,
            window_seconds=60,
            per_user=True,
            per_endpoint=True
        )
    ]
)
```

### 2. Security Headers Configuration

```python
security_config = SecurityConfig(
    enable_security_headers=True,
    enable_hsts=True,
    hsts_max_age=31536000,  # 1 year
    enable_csp=True,
    csp_policy="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
)
```

### 3. Audit Logging Configuration

```python
# Enable compliance frameworks
audit_system = ComprehensiveAuditSystem(
    db_session=db,
    redis_client=redis,
    enabled_frameworks=[
        ComplianceFramework.SOC2,
        ComplianceFramework.ISO27001,
        ComplianceFramework.NIST
    ]
)
```

## API Integration

### 1. Protecting API Endpoints

```python
from fastapi import Depends
from app.core.security_orchestrator_integration import SecurityOrchestrator

@app.get("/protected-endpoint")
async def protected_endpoint(
    current_user: dict = Depends(security_orchestrator.authenticate_request)
):
    if not current_user["authenticated"]:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    # Check authorization
    auth_result = await security_orchestrator.authorize_request(
        user_context=current_user["user"],
        resource="api/data",
        action="read"
    )
    
    if not auth_result["authorized"]:
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    return {"message": "Access granted", "user": current_user["user"]}
```

### 2. OAuth Authentication Flow

```python
# Initiate OAuth flow
@app.get("/auth/{provider}")
async def initiate_oauth(provider: str):
    auth_url, session_id = await oauth_system.initiate_authorization(
        provider_name=provider,
        scopes=["openid", "email", "profile"]
    )
    return RedirectResponse(url=auth_url)

# Handle OAuth callback
@app.get("/auth/callback/{provider}")
async def oauth_callback(provider: str, code: str, state: str):
    token_set, user_profile = await oauth_system.handle_authorization_callback(
        provider_name=provider,
        code=code,
        state=state
    )
    
    # Create internal JWT
    jwt_token = create_access_token({
        "sub": user_profile.user_id,
        "email": user_profile.email,
        "name": user_profile.name,
        "provider": user_profile.provider
    })
    
    return {"access_token": jwt_token, "user": user_profile}
```

## Monitoring and Alerting

### 1. Security Metrics Endpoint

```python
@app.get("/security/metrics")
async def get_security_metrics():
    return security_orchestrator.get_security_metrics()
```

### 2. Health Check Endpoint

```python
@app.get("/security/health")
async def security_health_check():
    return await security_orchestrator.health_check()
```

### 3. Real-time Security Monitoring

```python
# Add custom security event handler
async def custom_security_handler(event_type: str, severity: str, details: dict):
    if severity in ["high", "critical"]:
        # Send alert to external system
        await send_security_alert(event_type, severity, details)

security_orchestrator.add_security_event_handler(custom_security_handler)
```

## Compliance Reporting

### 1. Generate Compliance Reports

```python
@app.get("/compliance/report/{framework}")
async def generate_compliance_report(
    framework: str,
    start_date: datetime,
    end_date: datetime
):
    report = await security_orchestrator.generate_compliance_report(
        framework=framework,
        start_date=start_date,
        end_date=end_date
    )
    return report
```

### 2. Automated Compliance Monitoring

```python
# Schedule daily compliance checks
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

@scheduler.scheduled_job('cron', hour=0)  # Daily at midnight
async def daily_compliance_check():
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=1)
    
    for framework in ["soc2", "iso27001", "nist"]:
        report = await security_orchestrator.generate_compliance_report(
            framework=framework,
            start_date=start_date,
            end_date=end_date
        )
        
        # Alert if compliance score is low
        if report["overall_compliance_score"] < 0.9:
            await send_compliance_alert(framework, report)
```

## Testing

### 1. Security Integration Tests

```python
import pytest
from fastapi.testclient import TestClient

def test_oauth_flow():
    # Test OAuth authorization initiation
    response = client.get("/api/v1/oauth/authorize/google")
    assert response.status_code == 302
    
    # Test OAuth callback
    response = client.get("/api/v1/oauth/callback/google?code=test&state=test")
    assert response.status_code == 200

def test_rate_limiting():
    # Test rate limiting enforcement
    for _ in range(150):  # Exceed default limit
        response = client.get("/api/test")
    
    assert response.status_code == 429

def test_audit_logging():
    # Test audit event creation
    response = client.post("/api/protected", json={"test": "data"})
    
    # Verify audit log entry
    audit_logs = db.query(SecurityAuditLog).filter_by(action="test").all()
    assert len(audit_logs) > 0
```

### 2. Load Testing

```bash
# Install locust
pip install locust

# Run load tests
locust -f tests/performance/security_load_test.py --host=http://localhost:8000
```

## Production Considerations

### 1. SSL/TLS Configuration

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.pem;
    ssl_certificate_key /path/to/private-key.pem;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 2. Database Security

```sql
-- Create read-only user for reporting
CREATE USER leanvibe_readonly WITH PASSWORD 'readonly_password';
GRANT CONNECT ON DATABASE leanvibe_agent_hive TO leanvibe_readonly;
GRANT USAGE ON SCHEMA public TO leanvibe_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO leanvibe_readonly;

-- Enable row-level security
ALTER TABLE security_audit_log ENABLE ROW LEVEL SECURITY;
CREATE POLICY audit_policy ON security_audit_log
    FOR ALL TO leanvibe_user
    USING (human_controller = current_user);
```

### 3. Monitoring and Alerting

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, generate_latest

security_events_counter = Counter('security_events_total', 'Total security events', ['event_type', 'severity'])
auth_latency_histogram = Histogram('auth_request_duration_seconds', 'Authentication request latency')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### 4. Backup and Recovery

```bash
# Database backup
pg_dump -h localhost -U leanvibe_user leanvibe_agent_hive > backup_$(date +%Y%m%d).sql

# Redis backup
redis-cli --rdb dump.rdb

# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h localhost -U leanvibe_user leanvibe_agent_hive | gzip > /backups/db_$DATE.sql.gz
redis-cli --rdb /backups/redis_$DATE.rdb
aws s3 cp /backups/ s3://your-backup-bucket/leanvibe/ --recursive
```

## Troubleshooting

### Common Issues

1. **OAuth Redirect URI Mismatch**
   - Ensure redirect URIs in OAuth provider match exactly
   - Check BASE_URL environment variable

2. **Database Connection Issues**
   - Verify DATABASE_URL format
   - Check PostgreSQL service status
   - Ensure pgvector extension is installed

3. **Rate Limiting Issues**
   - Check Redis connectivity
   - Verify rate limit configuration
   - Monitor Redis memory usage

4. **JWT Token Issues**
   - Verify JWT_SECRET_KEY is set and consistent
   - Check token expiration settings
   - Ensure clock synchronization

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
uvicorn app.main:app --reload --log-level debug
```

### Performance Monitoring

```python
# Add performance monitoring
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        duration = time.time() - start_time
        
        logger.info(f"{func.__name__} took {duration:.3f}s")
        return result
    return wrapper
```

## Security Best Practices

1. **Secret Management**
   - Use environment variables for secrets
   - Consider using HashiCorp Vault or AWS Secrets Manager
   - Rotate secrets regularly

2. **Database Security**
   - Use connection pooling
   - Enable SSL connections
   - Regular security updates

3. **API Security**
   - Implement request signing for sensitive operations
   - Use CORS policies appropriately
   - Validate all input data

4. **Monitoring**
   - Set up alerting for security events
   - Monitor failed authentication attempts
   - Track unusual access patterns

5. **Compliance**
   - Regular compliance audits
   - Document security procedures
   - Maintain audit trails

## Support and Maintenance

For ongoing support and maintenance:

1. **Regular Updates**
   - Keep dependencies updated
   - Monitor security advisories
   - Test updates in staging environment

2. **Performance Monitoring**
   - Monitor response times
   - Track resource usage
   - Optimize slow queries

3. **Security Reviews**
   - Quarterly security assessments
   - Penetration testing
   - Code security reviews

This comprehensive deployment guide provides everything needed to deploy and maintain the enterprise-grade security system for LeanVibe Agent Hive.