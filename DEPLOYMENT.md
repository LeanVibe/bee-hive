# LeanVibe Agent Hive 2.0 - Production Deployment Guide

This guide provides comprehensive instructions for deploying LeanVibe Agent Hive 2.0 in a production environment.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Production Deployment](#production-deployment)
- [Security Configuration](#security-configuration)
- [Monitoring and Observability](#monitoring-and-observability)
- [Backup and Recovery](#backup-and-recovery)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Scaling Guidelines](#scaling-guidelines)

## Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- **CPU**: Minimum 4 cores (8+ cores recommended)
- **RAM**: Minimum 8GB (16GB+ recommended)
- **Storage**: Minimum 100GB SSD (500GB+ recommended)
- **Network**: Stable internet connection with ports 80/443 accessible

### Software Requirements
- Docker 24.0+ and Docker Compose 2.20+
- Git for repository management
- OpenSSL for SSL certificate management
- (Optional) Kubernetes 1.26+ for container orchestration

### Domain and DNS
- Registered domain name
- DNS records configured:
  - A record: `yourdomain.com` → Server IP
  - A record: `www.yourdomain.com` → Server IP  
  - A record: `api.yourdomain.com` → Server IP (optional)
  - A record: `monitor.yourdomain.com` → Server IP (optional)

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/leanvibe/agent-hive.git
cd agent-hive
```

### 2. Configure Environment
```bash
# Copy production environment template
cp .env.production.template .env.production

# Edit environment variables (IMPORTANT: Replace all placeholder values)
nano .env.production
```

### 3. Create Data Directories
```bash
# Create persistent data directories
sudo mkdir -p /var/lib/leanvibe/{postgres,redis,prometheus,grafana,logs,workspaces,static}
sudo chown -R $USER:$USER /var/lib/leanvibe
```

### 4. Generate SSL Certificates
```bash
# Option A: Let's Encrypt (Recommended)
docker-compose -f docker-compose.production.yml run --rm certbot

# Option B: Self-signed for testing
./scripts/generate-ssl-certs.sh
```

### 5. Deploy Production Stack
```bash
# Start production services
docker-compose -f docker-compose.production.yml up -d

# Verify deployment
docker-compose -f docker-compose.production.yml ps
```

### 6. Health Check
```bash
# Check application health
curl -H "Accept: application/json" https://yourdomain.com/health

# Expected response:
# {"status": "healthy", "checks": {...}, "timestamp": "..."}
```

## Environment Setup

### Production Environment Variables

**Critical Security Variables (MUST BE CHANGED):**
```env
# Generate with: openssl rand -hex 32
SECRET_KEY=your-super-secure-secret-key-minimum-64-chars

# Generate with: openssl rand -hex 64
JWT_SECRET_KEY=your-jwt-secret-key-minimum-128-chars

# Strong database credentials
POSTGRES_PASSWORD=your-secure-database-password-replace-this
REDIS_PASSWORD=your-secure-redis-password-replace-this

# Monitoring credentials
GRAFANA_PASSWORD=your-secure-grafana-password-replace-this
GRAFANA_SECRET_KEY=your-grafana-secret-key-replace-this
```

**Domain Configuration:**
```env
# Replace with your actual domain
DOMAIN_NAME=yourdomain.com
API_BASE_URL=https://yourdomain.com/api
FRONTEND_URL=https://yourdomain.com

# Update CORS origins
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
ALLOWED_HOSTS=yourdomain.com,www.yourdomain.com,api
```

**AI Service Configuration:**
```env
# Required: Get from https://console.anthropic.com/
ANTHROPIC_API_KEY=your-anthropic-api-key-here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Optional: OpenAI fallback
OPENAI_API_KEY=your-openai-api-key-here
```

### Directory Structure
```
/var/lib/leanvibe/          # Production data directory
├── postgres/               # PostgreSQL data
├── redis/                  # Redis data
├── prometheus/             # Metrics data
├── grafana/               # Dashboard data
├── logs/                  # Application logs
├── workspaces/            # Agent workspaces
├── static/                # Static files
└── backups/               # Database backups
```

## Production Deployment

### Docker Compose Services

The production stack includes:

1. **PostgreSQL**: Database with pgvector extension
2. **Redis**: Caching and message broker
3. **FastAPI Application**: Main API server
4. **Nginx**: Reverse proxy with SSL termination
5. **Prometheus**: Metrics collection
6. **Grafana**: Monitoring dashboards
7. **Postgres Backup**: Automated database backups

### Service Configuration

**Resource Limits:**
```yaml
# API Service
api:
  deploy:
    resources:
      limits:
        memory: 4G
        cpus: '2.0'
      reservations:
        memory: 2G
        cpus: '1.0'

# PostgreSQL
postgres:
  deploy:
    resources:
      limits:
        memory: 2G
        cpus: '1.0'
      reservations:
        memory: 1G
        cpus: '0.5'
```

### Deployment Commands

**Start Production Stack:**
```bash
# Full stack deployment
docker-compose -f docker-compose.production.yml up -d

# Start specific services
docker-compose -f docker-compose.production.yml up -d postgres redis api nginx

# Check service status
docker-compose -f docker-compose.production.yml ps
docker-compose -f docker-compose.production.yml logs -f api
```

**Update Deployment:**
```bash
# Pull latest images
docker-compose -f docker-compose.production.yml pull

# Recreate containers with zero-downtime
docker-compose -f docker-compose.production.yml up -d --no-deps api

# Verify update
curl https://yourdomain.com/health
```

### SSL Certificate Management

**Automatic Renewal with Let's Encrypt:**
```bash
# Initial certificate generation
docker-compose -f docker-compose.production.yml run --rm certbot

# Set up automatic renewal (add to crontab)
echo "0 12 * * * /usr/bin/docker-compose -f /path/to/docker-compose.production.yml run --rm certbot renew --quiet" | sudo crontab -
```

**Manual Certificate Installation:**
```bash
# Copy certificates to the correct location
sudo cp your-cert.pem config/ssl/fullchain.pem
sudo cp your-key.pem config/ssl/privkey.pem
sudo chown root:root config/ssl/*
sudo chmod 644 config/ssl/fullchain.pem
sudo chmod 600 config/ssl/privkey.pem
```

## Security Configuration

### Firewall Setup
```bash
# Ubuntu/Debian with UFW
sudo ufw allow 22/tcp          # SSH
sudo ufw allow 80/tcp          # HTTP
sudo ufw allow 443/tcp         # HTTPS
sudo ufw --force enable

# CentOS/RHEL with firewalld
sudo firewall-cmd --permanent --add-service=ssh
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

### Security Headers

The Nginx configuration includes comprehensive security headers:
- **HSTS**: HTTP Strict Transport Security
- **CSP**: Content Security Policy
- **X-Frame-Options**: Clickjacking protection
- **X-Content-Type-Options**: MIME type sniffing protection
- **Referrer-Policy**: Referrer information control

### Rate Limiting

Built-in rate limiting protects against abuse:
```nginx
# API endpoints: 100 requests/minute
# Authentication: 5 requests/minute  
# WebSocket: 5 connections/second
# Static files: 100 requests/second
```

### Access Control

**Monitoring Endpoints:**
- Prometheus metrics: Restricted to internal network
- Grafana dashboards: HTTP Basic Auth protected
- Health checks: Public (but no sensitive data)

## Monitoring and Observability

### Metrics Collection

**Prometheus Targets:**
- API application metrics (`/metrics`)
- PostgreSQL metrics (postgres-exporter)
- Redis metrics (redis-exporter)
- System metrics (node-exporter)
- Container metrics (cadvisor)

**Key Metrics:**
```prometheus
# Request metrics
http_requests_total
http_request_duration_seconds

# Database metrics  
postgresql_up
postgresql_connections_active

# Redis metrics
redis_up
redis_memory_used_bytes

# Resource metrics
container_memory_usage_bytes
container_cpu_usage_seconds_total
```

### Grafana Dashboards

Access monitoring at: `https://yourdomain.com/grafana`

**Default Dashboards:**
1. **Application Overview**: Request rates, response times, errors
2. **Infrastructure**: CPU, memory, disk, network usage
3. **Database Performance**: Connection pools, query performance
4. **Redis Monitoring**: Memory usage, command statistics
5. **Security Metrics**: Failed logins, blocked requests

### Log Management

**Structured Logging:**
All services output JSON logs for easy parsing:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "service": "api",
  "message": "Request processed",
  "request_id": "req-123",
  "duration_ms": 45
}
```

**Log Locations:**
```bash
# Application logs
/var/lib/leanvibe/logs/

# Container logs
docker-compose -f docker-compose.production.yml logs api
docker-compose -f docker-compose.production.yml logs postgres
docker-compose -f docker-compose.production.yml logs nginx
```

### Alerting

**Prometheus Alerting Rules:**
- High error rate (>5% 5xx responses)
- High response time (>2s p95)
- Database connection issues
- High memory usage (>80%)
- Disk space low (<20% free)
- SSL certificate expiry (<30 days)

## Backup and Recovery

### Automated Database Backup

The `postgres-backup` service provides automated backups:

**Schedule**: Daily at 2 AM (configurable)
**Retention**: 7 days, 4 weeks, 6 months
**Location**: `/var/lib/leanvibe/backups/`

**Manual Backup:**
```bash
# Create immediate backup
docker-compose -f docker-compose.production.yml exec postgres-backup backup

# List available backups
ls -la /var/lib/leanvibe/backups/
```

### Restore Procedures

**Database Restore:**
```bash
# Stop application
docker-compose -f docker-compose.production.yml stop api

# Restore from backup
docker-compose -f docker-compose.production.yml exec postgres \
  psql -U leanvibe_user -d leanvibe_agent_hive < /backups/backup-YYYY-MM-DD.sql

# Start application
docker-compose -f docker-compose.production.yml start api
```

### Disaster Recovery

**Full System Backup:**
```bash
# Create system snapshot
tar -czf leanvibe-backup-$(date +%Y%m%d).tar.gz \
  /var/lib/leanvibe/ \
  docker-compose.production.yml \
  .env.production \
  config/
```

**Recovery Steps:**
1. Provision new server with same specifications
2. Install Docker and Docker Compose
3. Restore backup files to `/var/lib/leanvibe/`
4. Update DNS records if IP changed
5. Deploy production stack
6. Verify all services are healthy

## Troubleshooting

### Common Issues

**Services Won't Start:**
```bash
# Check service logs
docker-compose -f docker-compose.production.yml logs api
docker-compose -f docker-compose.production.yml logs postgres

# Check resource usage
docker stats

# Verify environment variables
docker-compose -f docker-compose.production.yml config
```

**Database Connection Issues:**
```bash
# Test database connectivity
docker-compose -f docker-compose.production.yml exec postgres \
  psql -U leanvibe_user -d leanvibe_agent_hive -c "SELECT 1"

# Check PostgreSQL logs
docker-compose -f docker-compose.production.yml logs postgres
```

**SSL Certificate Problems:**
```bash
# Verify certificate validity
openssl x509 -in config/ssl/fullchain.pem -text -noout

# Check Nginx configuration
docker-compose -f docker-compose.production.yml exec nginx nginx -t

# Reload Nginx
docker-compose -f docker-compose.production.yml exec nginx nginx -s reload
```

**High Memory Usage:**
```bash
# Check container memory usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Scale down if needed
docker-compose -f docker-compose.production.yml up -d --scale api=1
```

### Performance Issues

**Slow Response Times:**
1. Check database query performance in Grafana
2. Monitor Redis cache hit rates
3. Review application logs for bottlenecks
4. Consider horizontal scaling

**High CPU Usage:**
1. Review Prometheus metrics for CPU usage patterns
2. Check for inefficient queries or algorithms
3. Consider vertical scaling (more CPU cores)

### Health Check Debugging

**API Health Check:**
```bash
# Basic health check
curl -v https://yourdomain.com/health

# Detailed health check with timing
curl -w "@curl-format.txt" -s -H "Accept: application/json" https://yourdomain.com/health

# Internal health check (from container)
docker-compose -f docker-compose.production.yml exec api curl -f http://localhost:8000/health
```

## Performance Optimization

### Database Optimization

**PostgreSQL Configuration (`config/postgresql.production.conf`):**
```ini
# Memory settings
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB

# Connection settings
max_connections = 100
shared_preload_libraries = 'pg_stat_statements'

# Performance settings
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
```

**Redis Configuration:**
```ini
# Memory management
maxmemory 512mb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000
```

### Application Optimization

**Gunicorn Workers:**
Adjust based on server resources:
```env
# For 4 CPU cores
MAX_WORKERS=4
WORKER_CONNECTIONS=1000

# For 8 CPU cores  
MAX_WORKERS=8
WORKER_CONNECTIONS=2000
```

**Caching Strategy:**
- Redis for session data and frequently accessed content
- Nginx for static file caching
- Database query result caching

### Nginx Optimization

**Worker Processes:**
```nginx
worker_processes auto;  # Matches CPU cores
worker_connections 4096;
```

**Caching:**
```nginx
# Static file caching
location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

## Scaling Guidelines

### Vertical Scaling (Scale Up)

**When to Scale Up:**
- CPU usage consistently >70%
- Memory usage consistently >80%
- Response times increasing
- Queue lengths growing

**Scaling Steps:**
1. Increase server resources (CPU, RAM)
2. Update Docker resource limits
3. Adjust worker/connection counts
4. Monitor performance improvements

### Horizontal Scaling (Scale Out)

**Load Balancer Setup:**
```bash
# Multiple API instances
docker-compose -f docker-compose.production.yml up -d --scale api=3

# Update Nginx upstream configuration
upstream api_backend {
    server api_1:8000;
    server api_2:8000; 
    server api_3:8000;
}
```

**Database Scaling:**
- Read replicas for query scaling
- Connection pooling with PgBouncer
- Database sharding for very large datasets

**Redis Scaling:**
- Redis Cluster for horizontal scaling
- Redis Sentinel for high availability
- Separate Redis instances by use case

### Kubernetes Deployment

For large-scale deployments, consider Kubernetes:

**Prerequisites:**
- Kubernetes cluster 1.26+
- kubectl configured
- Helm 3.x for package management

**Deployment:**
```bash
# Install using Helm chart
helm install leanvibe ./k8s/helm-chart \
  --values values.production.yaml \
  --namespace leanvibe \
  --create-namespace
```

## Support and Maintenance

### Regular Maintenance Tasks

**Weekly:**
- Review monitoring dashboards
- Check backup integrity
- Update security patches
- Review error logs

**Monthly:**
- Update Docker images
- Clean up old logs and backups
- Review resource usage trends
- Test disaster recovery procedures

**Quarterly:**
- Security audit and penetration testing
- Performance review and optimization
- Capacity planning
- Documentation updates

### Getting Help

**Support Channels:**
- GitHub Issues: Bug reports and feature requests
- Documentation: Detailed guides and API reference
- Community Forums: Discussion and troubleshooting
- Enterprise Support: Priority support for enterprise customers

**Before Contacting Support:**
1. Check the troubleshooting section
2. Review recent logs for error messages
3. Gather system information (versions, resource usage)
4. Document steps to reproduce issues

---

## Security Checklist

Before going to production, ensure:

- [ ] All default passwords changed
- [ ] SECRET_KEY and JWT_SECRET_KEY are cryptographically secure
- [ ] Database credentials are unique and complex
- [ ] SSL certificates are valid and properly configured
- [ ] CORS origins are restrictive and accurate
- [ ] Debug features are disabled
- [ ] Monitoring and alerting are configured
- [ ] Backup and recovery procedures are tested
- [ ] Security scanning is enabled
- [ ] Network isolation is properly configured
- [ ] Access logs are being collected
- [ ] Rate limiting is configured
- [ ] Security headers are enabled

---

**Need help?** Contact our support team or check the troubleshooting section above.
**Security issues?** Please report to security@leanvibe.com immediately.