# System Health Monitoring Runbook

## Overview

This runbook provides step-by-step procedures for monitoring and maintaining the health of the LeanVibe Agent Hive monitoring infrastructure, including Prometheus, Grafana, AlertManager, and dashboard services.

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI App   │───▶│   Prometheus    │───▶│    Grafana      │
│   (Metrics)     │    │   (Collection)  │    │  (Visualization)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dashboard     │    │  AlertManager   │    │   Mobile UI     │
│   Components    │    │ (Notifications) │    │  (QR Access)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Health Check

### 1. Service Status Check

```bash
# Check all monitoring services
docker-compose ps

# Expected output: All services should be "Up"
# - leanvibe_prometheus (Up)
# - leanvibe_grafana (Up)
# - leanvibe_alertmanager (Up)
# - leanvibe_api (Up)
```

### 2. Health Endpoints

```bash
# Check main application health
curl http://localhost:8000/health

# Check mobile monitoring health
curl http://localhost:8000/api/mobile/health

# Check dashboard metrics health
curl http://localhost:8000/api/dashboard/metrics/health
```

### 3. Quick Metrics Validation

```bash
# Test Prometheus metrics endpoint
curl http://localhost:8000/metrics

# Test dashboard-specific metrics
curl http://localhost:8000/api/dashboard/metrics/coordination
```

## Alert Response Procedures

### Critical Alert: System Down

**Alert:** `ApplicationDown`
**Severity:** Critical
**Response Time:** Immediate

#### Symptoms
- Application health check fails
- HTTP 500/503 errors
- Dashboard inaccessible

#### Investigation Steps

1. **Check service status:**
   ```bash
   docker-compose ps
   docker-compose logs api --tail=50
   ```

2. **Check resource usage:**
   ```bash
   docker stats
   ```

3. **Check database connectivity:**
   ```bash
   # Test PostgreSQL
   docker exec leanvibe_postgres pg_isready -U leanvibe_user
   
   # Test Redis
   docker exec leanvibe_redis redis-cli ping
   ```

#### Recovery Actions

1. **Restart application container:**
   ```bash
   docker-compose restart api
   ```

2. **If database issues:**
   ```bash
   docker-compose restart postgres redis
   sleep 30
   docker-compose restart api
   ```

3. **Check logs for errors:**
   ```bash
   docker-compose logs api --tail=100 | grep ERROR
   ```

#### Escalation
If issue persists after basic recovery:
- Alert on-call engineer via PagerDuty
- Document findings in incident response channel

---

### Warning Alert: High Coordination Failure Rate

**Alert:** `LowCoordinationSuccessRate` 
**Severity:** Warning
**Response Time:** 15 minutes

#### Symptoms
- Coordination success rate below 80%
- High task failure rates
- Agent communication issues

#### Investigation Steps

1. **Check agent status:**
   ```bash
   curl http://localhost:8000/api/dashboard/metrics/agents
   ```

2. **Review coordination metrics:**
   ```bash
   curl http://localhost:8000/api/dashboard/metrics/coordination
   ```

3. **Check Redis streams:**
   ```bash
   docker exec leanvibe_redis redis-cli INFO replication
   ```

#### Recovery Actions

1. **Restart agent orchestrator:**
   ```bash
   curl -X POST http://localhost:8000/api/v1/agents/restart-orchestrator
   ```

2. **Clear Redis message queues if needed:**
   ```bash
   docker exec leanvibe_redis redis-cli FLUSHDB
   ```

3. **Monitor recovery:**
   ```bash
   # Watch coordination success rate
   watch -n 5 'curl -s http://localhost:8000/api/dashboard/metrics/coordination | grep success_rate'
   ```

---

### Performance Alert: High Response Times

**Alert:** `HighResponseTime`
**Severity:** Warning
**Response Time:** 30 minutes

#### Symptoms
- P95 response times > 2 seconds
- Dashboard loading slowly
- User complaints about performance

#### Investigation Steps

1. **Check system resources:**
   ```bash
   docker stats
   htop
   ```

2. **Review database performance:**
   ```bash
   # Check active connections
   docker exec leanvibe_postgres psql -U leanvibe_user -d leanvibe_agent_hive -c "SELECT count(*) FROM pg_stat_activity;"
   
   # Check slow queries
   docker exec leanvibe_postgres psql -U leanvibe_user -d leanvibe_agent_hive -c "SELECT query, query_start, state FROM pg_stat_activity WHERE state != 'idle';"
   ```

3. **Check Redis performance:**
   ```bash
   docker exec leanvibe_redis redis-cli INFO stats
   ```

#### Recovery Actions

1. **Scale resources if needed:**
   ```bash
   # Increase memory limits in docker-compose.yml
   docker-compose up -d --scale api=2
   ```

2. **Clear caches:**
   ```bash
   curl -X POST http://localhost:8000/api/dashboard/metrics/cache/clear
   ```

3. **Restart services in order:**
   ```bash
   docker-compose restart redis
   sleep 10
   docker-compose restart postgres  
   sleep 20
   docker-compose restart api
   ```

---

## Dashboard Maintenance

### Daily Checks

1. **Verify dashboard accessibility:**
   - Open Grafana: http://localhost:3001
   - Check all dashboards load properly
   - Verify real-time data updates

2. **Mobile dashboard validation:**
   ```bash
   # Generate QR code
   curl http://localhost:8000/api/mobile/qr-access
   
   # Test mobile dashboard
   curl http://localhost:8000/api/mobile/dashboard
   ```

3. **Alert validation:**
   ```bash
   # Check AlertManager status
   curl http://localhost:9093/-/healthy
   
   # Review active alerts
   curl http://localhost:9093/api/v1/alerts
   ```

### Weekly Maintenance

1. **Data cleanup:**
   ```bash
   # Prometheus data retention (automated)
   # Elasticsearch index cleanup
   docker exec leanvibe_elasticsearch curator --config config.yml action.yml
   ```

2. **Performance review:**
   - Review Grafana performance dashboards
   - Check resource utilization trends
   - Plan capacity adjustments

3. **Security updates:**
   ```bash
   # Update container images
   docker-compose pull
   docker-compose up -d
   ```

### Monthly Tasks

1. **Backup validation:**
   ```bash
   # Test database backup
   docker exec leanvibe_postgres pg_dump -U leanvibe_user leanvibe_agent_hive > backup_test.sql
   
   # Test Prometheus data backup
   docker cp leanvibe_prometheus:/prometheus ./prometheus_backup_test
   ```

2. **Security scan:**
   ```bash
   # Vulnerability scan
   docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
     -v $PWD:/app -w /app anchore/grype:latest ./
   ```

---

## Troubleshooting Common Issues

### Dashboard Not Loading

**Problem:** Grafana dashboard shows "No data" or fails to load

**Diagnosis:**
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check metrics endpoint
curl http://localhost:8000/metrics | head -20
```

**Solutions:**
1. Restart Prometheus to reload configuration
2. Check FastAPI metrics endpoint is responding
3. Verify Prometheus can reach metrics endpoint

### Mobile QR Code Issues  

**Problem:** QR code generation fails or mobile dashboard doesn't load

**Diagnosis:**
```bash
# Test QR generation
curl http://localhost:8000/api/mobile/qr-access

# Check mobile service health  
curl http://localhost:8000/api/mobile/health
```

**Solutions:**
1. Install missing Python dependencies (qrcode, PIL)
2. Check mobile dashboard HTML rendering
3. Verify mobile-responsive CSS

### High Memory Usage

**Problem:** Containers consuming excessive memory

**Diagnosis:**
```bash
# Check container resource usage
docker stats

# Check Prometheus memory usage
curl http://localhost:9090/api/v1/status/runtimeinfo
```

**Solutions:**
1. Adjust Prometheus retention settings
2. Increase container memory limits
3. Optimize metric collection frequency

---

## Contact Information

### Escalation Matrix

| Severity | Contact | Response Time |
|----------|---------|---------------|
| Critical | On-call Engineer + PagerDuty | 15 minutes |
| High | DevOps Team Slack | 1 hour |
| Medium | Engineering Team | 4 hours |
| Low | Create ticket | Next business day |

### Key Contacts

- **DevOps Lead:** @devops-lead (Slack)
- **System Administrator:** @sysadmin (Slack)  
- **Engineering Manager:** @eng-manager (Email)
- **On-call Rotation:** PagerDuty schedule

### Useful Links

- **Grafana Dashboards:** http://localhost:3001
- **Prometheus:** http://localhost:9090
- **AlertManager:** http://localhost:9093
- **Mobile Dashboard:** http://localhost:8000/api/mobile/dashboard
- **System Health:** http://localhost:8000/health
- **Documentation:** /infrastructure/runbooks/

---

**Last Updated:** {{current_date}}
**Version:** 1.0
**Owner:** DevOps Team