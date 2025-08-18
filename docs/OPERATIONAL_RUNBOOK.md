# LeanVibe Agent Hive 2.0 - Operational Runbook

## 📋 Production Operations Guide

This runbook provides comprehensive guidance for deploying, monitoring, and maintaining LeanVibe Agent Hive 2.0 in production environments.

**Status**: ✅ Production-ready unified architecture  
**Last Updated**: 2024 (Configuration consolidation complete)

## 🎯 Quick Reference

| Environment | Purpose | Performance Target | Security Level |
|-------------|---------|------------------|----------------|
| **Production** | Live system | 99.9% uptime, <200ms response | Maximum security |
| **Staging** | Pre-production testing | 95% uptime, <300ms response | Production-like |
| **Development** | Local development | N/A | Minimal security |
| **Testing** | Automated testing | Fast startup | Isolated |

## 🚀 Production Deployment

### Prerequisites
- Docker & Docker Compose
- PostgreSQL 14+ (with pgvector extension)
- Redis 7.0+
- Nginx (for reverse proxy)
- SSL certificates
- Monitoring infrastructure (Prometheus/Grafana)

### 1. Environment Configuration

#### Production Environment Variables
```bash
# Core application
export ENVIRONMENT=production
export DEBUG=false
export LOG_LEVEL=INFO
export SECRET_KEY=$(openssl rand -hex 32)
export JWT_SECRET_KEY=$(openssl rand -hex 64)

# Database configuration
export DATABASE_URL=postgresql://user:pass@prod-postgres:5432/leanvibe_prod
export DATABASE_POOL_SIZE=50
export DATABASE_MAX_OVERFLOW=100

# Redis configuration  
export REDIS_URL=redis://prod-redis:6379/0
export REDIS_CONNECTION_POOL_SIZE=100
export REDIS_MAX_CONNECTIONS=500

# Security settings
export API_KEY_REQUIRED=true
export RATE_LIMITING_ENABLED=true
export SECURITY_ENABLED=true
export MFA_ENABLED=true
export THREAT_DETECTION_ENABLED=true

# Performance settings
export MAX_CONCURRENT_AGENTS=100
export TARGET_RESPONSE_TIME_MS=150
export TARGET_THROUGHPUT_RPS=20000
export AUTO_SCALING_ENABLED=true

# Monitoring
export METRICS_ENABLED=true
export PROMETHEUS_PORT=9090
export AUDIT_LOGGING_ENABLED=true
```

#### Unified Configuration Migration
```bash
# Validate production configuration
python scripts/migrate_configurations.py --environment production --validate-only

# Migrate to unified configuration with backup
python scripts/migrate_configurations.py --environment production --backup

# Verify migration success
python -c "from app.config.unified_config import get_unified_config; print('✅ Configuration loaded successfully')"
```

### 2. Infrastructure Deployment

#### Docker Compose Production Setup
```bash
# Use production compose configuration
docker compose -f docker-compose.production.yml up -d

# Verify all services are healthy
docker compose -f docker-compose.production.yml ps
```

#### Database Setup
```sql
-- Create database with required extensions
CREATE DATABASE leanvibe_prod;
\c leanvibe_prod;
CREATE EXTENSION IF NOT EXISTS pgvector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
```

#### Application Deployment
```bash
# Build production image
docker build -t leanvibe-hive:production .

# Deploy with configuration
docker run -d \
  --name leanvibe-production \
  --env-file .env.production \
  -p 8000:8000 \
  --restart unless-stopped \
  leanvibe-hive:production

# Initialize unified configuration system
docker exec leanvibe-production python -c "
from app.config.unified_config import initialize_unified_config, Environment
initialize_unified_config(environment=Environment.PRODUCTION)
"
```

### 3. Production Validation

#### Health Checks
```bash
# Core system health
curl -f http://localhost:8000/health || exit 1

# Component health verification
curl -f http://localhost:8000/api/observability/health

# Configuration system health
curl -f http://localhost:8000/api/admin/config/status

# Database connectivity
curl -f http://localhost:8000/api/observability/database

# Redis connectivity  
curl -f http://localhost:8000/api/observability/redis
```

#### Performance Validation
```bash
# Load test critical endpoints
ab -n 1000 -c 10 http://localhost:8000/health
ab -n 500 -c 5 http://localhost:8000/api/agents/

# WebSocket connection test
wscat -c ws://localhost:8000/api/dashboard/ws/dashboard
```

## 📊 Monitoring & Observability

### Key Metrics to Monitor

#### System Performance
- **Response Time**: P95 < 200ms, P99 < 500ms
- **Throughput**: > 10,000 RPS capacity
- **Error Rate**: < 0.1% (99.9% success rate)
- **Uptime**: > 99.9% availability

#### Resource Utilization
- **CPU Usage**: < 80% average
- **Memory Usage**: < 8GB per instance
- **Database Connections**: < 80% of pool
- **Redis Connections**: < 80% of pool

#### Component Health
- **Universal Orchestrator**: Agent spawn time < 100ms
- **Domain Managers**: Processing time < 100ms
- **Specialized Engines**: Queue depth < 1000
- **Communication Hub**: Connection success rate > 99%

### Monitoring Setup

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'leanvibe-hive'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']
```

#### Grafana Dashboards
```bash
# Import pre-configured dashboards
curl -X POST http://admin:admin@grafana:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana/leanvibe-dashboard.json
```

#### Alert Rules
```yaml
# alerts.yml
groups:
  - name: leanvibe-hive-alerts
    rules:
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, http_request_duration_seconds) > 0.2
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.01
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          
      - alert: DatabaseConnectionPoolExhaustion
        expr: database_connections_active / database_connections_max > 0.9
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection pool nearly exhausted"
```

### Log Management

#### Structured Logging Configuration
```python
# Logging configuration in unified config
config.monitoring.structured_logging = True
config.monitoring.log_format = "json"
config.monitoring.log_level = "INFO"
```

#### Log Aggregation
```bash
# ELK Stack configuration
docker run -d \
  --name elasticsearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  elasticsearch:7.17.0

docker run -d \
  --name logstash \
  -p 5044:5044 \
  logstash:7.17.0

docker run -d \
  --name kibana \
  -p 5601:5601 \
  kibana:7.17.0
```

## 🔐 Security Operations

### Security Hardening Checklist

#### Application Security
- [ ] JWT secrets are properly secured (64+ characters)
- [ ] API key authentication enabled in production
- [ ] Rate limiting configured and active
- [ ] CORS origins properly restricted
- [ ] Input validation enabled on all endpoints
- [ ] SQL injection protection active
- [ ] XSS protection headers configured

#### Infrastructure Security
- [ ] TLS/SSL certificates installed and valid
- [ ] Database connections encrypted
- [ ] Redis AUTH enabled
- [ ] Network segmentation implemented
- [ ] Firewall rules configured
- [ ] Regular security updates applied

#### Compliance Verification
```bash
# SOC2 compliance check
curl -f http://localhost:8000/api/security/compliance/soc2

# GDPR compliance check
curl -f http://localhost:8000/api/security/compliance/gdpr

# Security audit
curl -f http://localhost:8000/api/security/audit/status
```

### Security Monitoring
```bash
# Security events monitoring
tail -f logs/security_audit.log | grep -i "threat\|alert\|violation"

# Failed authentication attempts
grep "authentication_failed" logs/app.log | tail -20

# Suspicious activity detection
curl http://localhost:8000/api/security/threats/recent
```

## 🚨 Incident Response

### Incident Response Procedures

#### High Severity Incidents (System Down)

1. **Immediate Response (0-5 minutes)**
   ```bash
   # Check system health
   curl -f http://localhost:8000/health || echo "SYSTEM DOWN"
   
   # Check all components
   docker compose -f docker-compose.production.yml ps
   
   # Check logs for errors
   docker logs leanvibe-production --tail 100
   ```

2. **Diagnosis (5-15 minutes)**
   ```bash
   # Check resource utilization
   docker stats leanvibe-production
   
   # Database connectivity
   psql $DATABASE_URL -c "SELECT 1;"
   
   # Redis connectivity
   redis-cli -u $REDIS_URL ping
   
   # Configuration status
   curl http://localhost:8000/api/admin/config/status
   ```

3. **Recovery Actions**
   ```bash
   # Restart application if needed
   docker restart leanvibe-production
   
   # Rollback configuration if needed
   python scripts/migrate_configurations.py --rollback config_backup_*.json
   
   # Scale up if resource constrained
   docker compose -f docker-compose.production.yml up -d --scale app=3
   ```

#### Medium Severity Incidents (Performance Degradation)

1. **Performance Analysis**
   ```bash
   # Check response times
   curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health
   
   # Monitor system resources
   htop
   
   # Database performance
   psql $DATABASE_URL -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"
   ```

2. **Optimization Actions**
   ```bash
   # Enable auto-scaling
   export AUTO_SCALING_ENABLED=true
   
   # Restart unified configuration hot-reload
   curl -X POST http://localhost:8000/api/admin/config/reload
   
   # Clear Redis cache if needed
   redis-cli -u $REDIS_URL FLUSHDB
   ```

#### Low Severity Incidents (Minor Issues)

1. **Investigation**
   ```bash
   # Review error logs
   grep ERROR logs/app.log | tail -50
   
   # Check configuration drift
   python scripts/migrate_configurations.py --validate-only
   
   # Monitor trends
   curl http://localhost:8000/api/observability/metrics
   ```

### Escalation Procedures

| Severity | Response Time | Escalation Path |
|----------|--------------|----------------|
| **Critical** (System Down) | 5 minutes | On-call engineer → Engineering manager → CTO |
| **High** (Major Feature Down) | 15 minutes | On-call engineer → Engineering manager |
| **Medium** (Performance Issues) | 30 minutes | Engineering team → Team lead |
| **Low** (Minor Issues) | 2 hours | Engineering team |

## 🔄 Maintenance Operations

### Regular Maintenance Tasks

#### Daily Operations
```bash
# Health check verification
./scripts/health_check.sh

# Log rotation
logrotate /etc/logrotate.d/leanvibe-hive

# Configuration validation
python scripts/migrate_configurations.py --validate-only

# Security scan
./scripts/security_scan.sh
```

#### Weekly Operations
```bash
# Database maintenance
psql $DATABASE_URL -c "VACUUM ANALYZE;"

# Redis memory optimization
redis-cli -u $REDIS_URL MEMORY PURGE

# Performance baseline
python scripts/performance_validation.py --baseline

# Configuration backup
python scripts/migrate_configurations.py --environment production --backup
```

#### Monthly Operations
```bash
# Security updates
apt update && apt upgrade -y
docker pull leanvibe-hive:latest

# Database optimization
psql $DATABASE_URL -c "REINDEX DATABASE leanvibe_prod;"

# SSL certificate renewal
certbot renew

# Disaster recovery test
./scripts/dr_test.sh
```

### Configuration Management

#### Hot Configuration Reload
```python
# Enable hot-reload in production (use with caution)
from app.config.unified_config import get_config_manager

config_manager = get_config_manager()
await config_manager.start_hot_reload()
```

#### Configuration Rollback
```bash
# List available backups
ls -la config_backups/

# Rollback to specific backup
python scripts/migrate_configurations.py --rollback config_backup_20240101_120000.json

# Verify rollback success
python scripts/migrate_configurations.py --validate-only
```

### Database Operations

#### Backup and Restore
```bash
# Create backup
pg_dump $DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore from backup
psql $DATABASE_URL < backup_20240101_120000.sql

# Point-in-time recovery
pg_restore --clean --if-exists -d $DATABASE_URL backup.dump
```

#### Performance Tuning
```sql
-- Monitor slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Monitor connection usage
SELECT count(*), state 
FROM pg_stat_activity 
GROUP BY state;
```

## 📈 Scaling Operations

### Horizontal Scaling

#### Application Scaling
```bash
# Scale up application instances
docker compose -f docker-compose.production.yml up -d --scale app=5

# Load balancer configuration
nginx -s reload
```

#### Database Scaling
```bash
# Read replica setup
pg_basebackup -h primary-db -D /var/lib/postgresql/replica -U replicator -v -P

# Connection pooling with PgBouncer
docker run -d --name pgbouncer -p 5432:5432 pgbouncer/pgbouncer
```

### Auto-scaling Configuration
```python
# Auto-scaling settings in unified config
config.performance.auto_scaling_enabled = True
config.performance.max_concurrent_agents = 200  # Scale limit
config.performance.target_response_time_ms = 150
```

## 🔍 Troubleshooting Guide

### Common Issues and Solutions

#### Configuration Issues
```bash
# Problem: Configuration validation fails
# Solution: Check environment variables and migrate
python scripts/migrate_configurations.py --validate-only
export MISSING_VAR=value
python scripts/migrate_configurations.py --environment production --backup

# Problem: Hot-reload not working
# Solution: Restart config manager
curl -X POST http://localhost:8000/api/admin/config/reload
```

#### Performance Issues
```bash
# Problem: High response times
# Solution: Check resource usage and scale
docker stats
docker compose -f docker-compose.production.yml up -d --scale app=3

# Problem: Database connection pool exhaustion
# Solution: Increase pool size
export DATABASE_POOL_SIZE=100
python scripts/migrate_configurations.py --environment production
```

#### Security Issues
```bash
# Problem: Authentication failures
# Solution: Check JWT configuration
curl http://localhost:8000/api/security/jwt/validate

# Problem: Rate limiting too aggressive
# Solution: Adjust rate limits
export RATE_LIMIT_REQUESTS_PER_MINUTE=2000
curl -X POST http://localhost:8000/api/admin/config/reload
```

### Emergency Procedures

#### Complete System Recovery
```bash
#!/bin/bash
# emergency_recovery.sh

echo "🚨 Starting emergency recovery..."

# 1. Stop all services
docker compose -f docker-compose.production.yml down

# 2. Backup current state
python scripts/migrate_configurations.py --environment production --backup

# 3. Reset to known good configuration
python scripts/migrate_configurations.py --rollback config_backup_known_good.json

# 4. Restart services
docker compose -f docker-compose.production.yml up -d

# 5. Verify health
sleep 30
curl -f http://localhost:8000/health || echo "❌ Recovery failed"

echo "✅ Emergency recovery completed"
```

## 📚 Additional Resources

### Documentation
- `docs/ARCHITECTURE.md` - System architecture overview
- `docs/GETTING_STARTED.md` - 2-day developer onboarding
- `docs/TECHNICAL_DEBT_ANALYSIS.md` - Known issues and improvements
- `app/config/unified_config.py` - Configuration system source

### Monitoring Dashboards
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Application metrics: http://localhost:8000/metrics

### Support Contacts
- **Engineering Team**: engineering@leanvibe.com
- **DevOps Team**: devops@leanvibe.com
- **Security Team**: security@leanvibe.com
- **On-call**: +1-XXX-XXX-XXXX

---

**🎯 This runbook is maintained by the DevOps team and updated with each major release.**

Last Updated: Configuration & Documentation Consolidation Phase Complete