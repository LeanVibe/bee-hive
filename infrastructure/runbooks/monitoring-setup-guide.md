# LeanVibe Monitoring Infrastructure Setup Guide

## Overview

This guide provides complete instructions for setting up the comprehensive monitoring infrastructure for the LeanVibe Agent Hive dashboard system, including Prometheus, Grafana, AlertManager, and mobile monitoring capabilities.

## Prerequisites

- Docker and Docker Compose installed
- LeanVibe Agent Hive system running
- Basic knowledge of Prometheus, Grafana, and container orchestration
- Network access to monitoring ports (3001, 9090, 9093)

## Quick Setup (5-12 Minutes)

### 1. Start Monitoring Stack

```bash
# Navigate to project root
cd /path/to/leanvibe-agent-hive

# Start complete monitoring stack
docker-compose --profile monitoring --profile logging up -d

# Verify all services are running
docker-compose ps
```

Expected services should be running:
- `leanvibe_prometheus` (Up)
- `leanvibe_grafana` (Up)  
- `leanvibe_alertmanager` (Up)
- `leanvibe_postgres_exporter` (Up)
- `leanvibe_redis_exporter` (Up)
- `leanvibe_cadvisor` (Up)
- `leanvibe_node_exporter` (Up)

### 2. Configure Environment Variables

Create or update `.env.local`:

```bash
# Monitoring Configuration
GRAFANA_PASSWORD=secure_admin_password
SMTP_HOST=smtp.your-domain.com
SMTP_FROM=alerts@your-domain.com
SMTP_USERNAME=alerts@your-domain.com
SMTP_PASSWORD=your_smtp_password
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK

# Security Configuration
CRITICAL_EMAIL_LIST=admin@your-domain.com,ops@your-domain.com
SECURITY_EMAIL_LIST=security@your-domain.com
PAGERDUTY_WEBHOOK_URL=https://events.pagerduty.com/YOUR-INTEGRATION-KEY
MOBILE_PUSH_WEBHOOK_URL=https://your-mobile-push-service.com/webhook
```

### 3. Initial Validation

```bash
# Test Prometheus metrics collection
curl http://localhost:9090/api/v1/targets

# Test Grafana access
curl http://localhost:3001/api/health

# Test AlertManager
curl http://localhost:9093/-/healthy

# Test mobile dashboard
curl http://localhost:8000/api/mobile/health
```

## Detailed Setup Instructions

### Prometheus Configuration

1. **Verify Prometheus Configuration:**
   ```bash
   # Check Prometheus config syntax
   docker exec leanvibe_prometheus promtool check config /etc/prometheus/prometheus.yml
   
   # Check alert rules
   docker exec leanvibe_prometheus promtool check rules /etc/prometheus/alertmanager/rules.yml
   ```

2. **Dashboard-Specific Metrics Endpoints:**
   ```bash
   # Test coordination metrics
   curl http://localhost:8000/api/dashboard/metrics/coordination
   
   # Test security metrics  
   curl http://localhost:8000/api/dashboard/metrics/security
   
   # Test business metrics
   curl http://localhost:8000/api/dashboard/metrics/business
   ```

### Grafana Dashboard Setup

1. **Access Grafana:**
   - URL: http://localhost:3001
   - Username: `admin`
   - Password: Value from `GRAFANA_PASSWORD` env var

2. **Import Dashboards:**
   
   Dashboards are automatically provisioned from:
   - `infrastructure/monitoring/grafana/dashboards/multi-agent-coordination.json`
   - `infrastructure/monitoring/grafana/dashboards/performance-analytics.json`
   - `infrastructure/monitoring/grafana/dashboards/security-monitoring.json`
   - `infrastructure/monitoring/grafana/dashboards/business-intelligence.json`

3. **Verify Data Sources:**
   - Go to Configuration > Data Sources
   - Ensure Prometheus is configured and working
   - Test connection should show "Data source is working"

### AlertManager Configuration

1. **Test Alert Routing:**
   ```bash
   # Send test alert
   curl -XPOST http://localhost:9093/api/v1/alerts -H "Content-Type: application/json" -d '[
     {
       "labels": {
         "alertname": "TestAlert",
         "severity": "warning",
         "component": "test"
       },
       "annotations": {
         "summary": "Test alert for monitoring setup",
         "description": "This is a test alert to validate AlertManager configuration"
       }
     }
   ]'
   ```

2. **Check Alert Routing:**
   ```bash
   # View active alerts
   curl http://localhost:9093/api/v1/alerts
   
   # Check AlertManager status
   curl http://localhost:9093/api/v1/status
   ```

### Mobile Monitoring Setup

1. **Install QR Code Dependencies:**
   ```bash
   # Add to requirements or install directly
   pip install qrcode[pil] Pillow
   ```

2. **Test Mobile Access:**
   ```bash
   # Generate QR code for mobile access
   curl http://localhost:8000/api/mobile/qr-access
   
   # Test mobile dashboard
   curl http://localhost:8000/api/mobile/dashboard
   ```

3. **Mobile Dashboard Features:**
   - Touch-optimized interface
   - Auto-refresh every 30 seconds
   - QR code access for quick mobile setup
   - Real-time alerts display

## Configuration Validation

### Health Check Script

Create `infrastructure/scripts/monitoring-health-check.sh`:

```bash
#!/bin/bash
# Comprehensive monitoring health check

echo "=== LeanVibe Monitoring Health Check ==="
echo "Timestamp: $(date)"
echo

# Check service availability
services=("prometheus:9090" "grafana:3001" "alertmanager:9093")
for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if curl -s --max-time 5 "http://localhost:$port" > /dev/null 2>&1; then
        echo "✅ $name is accessible on port $port"
    else
        echo "❌ $name is NOT accessible on port $port"
    fi
done

# Check metrics endpoints
echo
echo "--- Metrics Endpoints ---"
metrics_endpoints=(
    "coordination:8000/api/dashboard/metrics/coordination"
    "security:8000/api/dashboard/metrics/security" 
    "business:8000/api/dashboard/metrics/business"
    "mobile:8000/api/mobile/metrics"
)

for endpoint in "${metrics_endpoints[@]}"; do
    name=$(echo $endpoint | cut -d: -f1)
    url=$(echo $endpoint | cut -d: -f2-)
    
    if curl -s --max-time 5 "http://localhost:$url" > /dev/null 2>&1; then
        echo "✅ $name metrics endpoint is working"
    else
        echo "❌ $name metrics endpoint is NOT working"
    fi
done

# Check Prometheus targets
echo
echo "--- Prometheus Targets ---"
targets_up=$(curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets | map(select(.health=="up")) | length')
targets_total=$(curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets | length')
echo "Healthy targets: $targets_up/$targets_total"

# Check Grafana dashboards
echo
echo "--- Grafana Dashboards ---"
dashboards=$(curl -s -u admin:${GRAFANA_PASSWORD:-admin_password} http://localhost:3001/api/search?type=dash-db | jq '. | length')
echo "Available dashboards: $dashboards"

echo
echo "=== Health Check Complete ==="
```

Make executable and run:
```bash
chmod +x infrastructure/scripts/monitoring-health-check.sh
./infrastructure/scripts/monitoring-health-check.sh
```

## Performance Optimization

### 1. Prometheus Optimization

```yaml
# Add to docker-compose.yml prometheus service
command:
  - '--config.file=/etc/prometheus/prometheus.yml'
  - '--storage.tsdb.path=/prometheus'
  - '--storage.tsdb.retention.time=30d'
  - '--storage.tsdb.retention.size=10GB'
  - '--storage.tsdb.wal-compression'
  - '--web.enable-lifecycle'
  - '--web.enable-admin-api'
  - '--query.max-concurrency=20'
  - '--query.timeout=2m'
```

### 2. Grafana Performance

```bash
# Add to grafana environment in docker-compose.yml
environment:
  - GF_SECURITY_ADMIN_USER=admin
  - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin_password}
  - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
  - GF_FEATURE_TOGGLES_ENABLE=publicDashboards
  - GF_DATABASE_WAL=true
  - GF_PANELS_DISABLE_SANITIZE_HTML=true
```

### 3. Resource Limits

```yaml
# Update resource limits in docker-compose.yml
deploy:
  resources:
    limits:
      memory: 512M
      cpus: '0.5'
    reservations:
      memory: 256M
      cpus: '0.25'
```

## Security Configuration

### 1. Enable HTTPS (Production)

```yaml
# Add to nginx service in docker-compose.yml
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
    - "443:443"
  volumes:
    - ./infrastructure/nginx/nginx.conf:/etc/nginx/nginx.conf
    - ./infrastructure/ssl:/etc/ssl/certs
```

### 2. Authentication Setup

```bash
# Generate bcrypt password for basic auth
htpasswd -nB username

# Add to prometheus.yml
basic_auth_users:
  username: $2b$12$hashed_password
```

### 3. Network Security

```yaml
# Restrict network access in docker-compose.yml
networks:
  monitoring:
    driver: bridge
    internal: true  # Isolate monitoring network
  
  public:
    driver: bridge
```

## Troubleshooting

### Common Issues

1. **"No data in Grafana dashboards"**
   ```bash
   # Check Prometheus targets
   curl http://localhost:9090/api/v1/targets
   
   # Verify metrics endpoints
   curl http://localhost:8000/metrics | head -20
   ```

2. **"AlertManager not sending notifications"**
   ```bash
   # Check AlertManager configuration
   docker exec leanvibe_alertmanager amtool config show
   
   # Test SMTP settings
   docker exec leanvibe_alertmanager amtool config routes test
   ```

3. **"Mobile dashboard not accessible"**
   ```bash
   # Check mobile service health
   curl http://localhost:8000/api/mobile/health
   
   # Install missing dependencies
   pip install qrcode[pil] Pillow
   ```

### Debug Commands

```bash
# View service logs
docker-compose logs prometheus --tail=50
docker-compose logs grafana --tail=50
docker-compose logs alertmanager --tail=50

# Check resource usage
docker stats

# Validate configuration files
docker exec leanvibe_prometheus promtool check config /etc/prometheus/prometheus.yml
```

## Maintenance

### Daily Tasks
- Monitor dashboard accessibility
- Check alert status
- Verify mobile QR code generation

### Weekly Tasks  
- Review Grafana dashboard performance
- Clean up old metrics data
- Update security configurations

### Monthly Tasks
- Update container images
- Review and optimize alert rules
- Backup configuration files

---

## Next Steps

After setup completion:

1. **Configure Business-Specific Alerts:**
   - Set appropriate thresholds for your environment
   - Customize notification channels
   - Test alert escalation procedures

2. **Dashboard Customization:**
   - Modify dashboards for your specific metrics
   - Add custom business KPIs
   - Configure user access and permissions

3. **Integration:**
   - Connect with existing monitoring tools
   - Set up log forwarding to external systems  
   - Configure backup and disaster recovery

4. **Training:**
   - Train operations team on dashboard usage
   - Document custom procedures and runbooks
   - Set up monitoring playbooks

---

**Last Updated:** 2025-08-07
**Version:** 1.0  
**Owner:** DevOps Team