# LeanVibe Agent Hive 2.0 - Production Infrastructure Setup

**Epic 7 Phase 1: Production Infrastructure Excellence**

This directory contains the complete production infrastructure setup for LeanVibe Agent Hive 2.0, building on the solid foundation established in Epic 6. All components have been designed for high availability, security, and scalability in production environments.

## 🏗️ Architecture Overview

### Production Infrastructure Components

1. **Database Layer**
   - Production-optimized PostgreSQL 15 with pgvector extension
   - PgBouncer connection pooling for improved performance
   - Automated backups with encryption and S3 storage
   - Database migration system with rollback capabilities

2. **Caching Layer** 
   - Redis cluster with master-replica setup
   - Redis Sentinel for high availability
   - Production-tuned memory policies and persistence

3. **Application Layer**
   - Horizontally scalable FastAPI services
   - Container orchestration with health checks
   - Resource limits and auto-restart policies

4. **Reverse Proxy & Load Balancing**
   - Nginx with SSL termination and HTTP/2
   - Rate limiting and DDoS protection
   - Automated SSL certificate management with Let's Encrypt

5. **Monitoring & Observability**
   - Prometheus metrics collection
   - Grafana dashboards for visualization
   - AlertManager for notifications
   - Centralized logging with Loki and Promtail

6. **Security & Compliance**
   - SSL/TLS encryption everywhere
   - Security headers and CSP policies
   - Container security scanning
   - Network segmentation

7. **Backup & Disaster Recovery**
   - Automated daily backups
   - Disaster recovery procedures
   - Business continuity testing
   - Off-site backup storage

## 📁 File Structure

```
deploy/production/
├── docker-compose.production-optimized.yml  # Main production deployment
├── deploy.sh                               # Deployment orchestrator
├── ssl-setup.sh                           # SSL certificate management
├── disaster-recovery.sh                   # Backup and recovery procedures
├── loki-config.yml                        # Centralized logging configuration
├── promtail-config.yml                    # Log collection agent configuration
└── README.md                              # This file

../../config/
├── nginx.production.conf                  # Production nginx configuration
├── postgresql.production.conf             # Production database configuration
├── redis.production.conf                  # Production Redis configuration
└── ...

../../database/scripts/
├── backup.sh                              # Database backup script
├── restore.sh                             # Database restore script
└── migrate.py                             # Database migration manager
```

## 🚀 Deployment Guide

### Prerequisites

1. **System Requirements**
   - Linux server with Docker and Docker Compose
   - Minimum 8GB RAM, 4 CPU cores, 100GB storage
   - Domain name configured with DNS pointing to server
   - SSL certificate requirements met

2. **Environment Setup**
   ```bash
   # Install required tools
   sudo apt update
   sudo apt install -y docker.io docker-compose git curl openssl jq
   
   # Clone repository
   git clone <repository-url>
   cd bee-hive/deploy/production
   ```

3. **Configuration**
   ```bash
   # Create production environment file
   cp ../../.env.example ../../.env.production
   
   # Edit environment variables (REQUIRED)
   nano ../../.env.production
   ```

### Required Environment Variables

Create `/path/to/bee-hive/.env.production` with:

```bash
# Domain Configuration
DOMAIN_NAME=your-domain.com
ADMIN_EMAIL=admin@your-domain.com

# Database Configuration
POSTGRES_PASSWORD=secure_postgres_password_here
POSTGRES_DB=leanvibe_agent_hive
POSTGRES_USER=leanvibe_user

# Redis Configuration  
REDIS_PASSWORD=secure_redis_password_here

# Application Security
SECRET_KEY=your_secret_key_64_chars_minimum
JWT_SECRET_KEY=your_jwt_secret_key_minimum_64_chars

# AI Service
ANTHROPIC_API_KEY=your_anthropic_api_key

# CORS Configuration
CORS_ORIGINS=https://your-domain.com,https://www.your-domain.com

# Monitoring
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=secure_grafana_password

# Backup Configuration (Optional)
BACKUP_S3_BUCKET=your-backup-bucket
BACKUP_SLACK_WEBHOOK=your-slack-webhook-url
BACKUP_ENCRYPTION_KEY=your_backup_encryption_key

# SSL Configuration (Optional)
SSL_SLACK_WEBHOOK=your-ssl-webhook-url
```

### Deployment Commands

1. **Full Production Deployment**
   ```bash
   sudo ./deploy.sh deploy --domain your-domain.com --email admin@your-domain.com
   ```

2. **Check Deployment Status**
   ```bash
   sudo ./deploy.sh status
   ```

3. **Health Check**
   ```bash
   sudo ./deploy.sh health
   ```

4. **View Logs**
   ```bash
   sudo ./deploy.sh logs
   ```

## 🔒 Security Features

### SSL/TLS Configuration
- Automatic SSL certificate provisioning with Let's Encrypt
- SSL certificate auto-renewal with monitoring
- Strong SSL/TLS configuration (A+ rating)
- HTTP Strict Transport Security (HSTS)

### Network Security
- Rate limiting on all endpoints
- DDoS protection mechanisms
- Network segmentation with Docker networks
- Secure container communication

### Application Security
- Security headers (CSP, X-Frame-Options, etc.)
- Container security scanning
- Secret management
- Input validation and sanitization

### Access Control
- Authentication and authorization
- Role-based access control
- API key management
- Session security

## 📊 Monitoring & Observability

### Metrics Collection
- **Prometheus**: System and application metrics
- **Node Exporter**: System resource monitoring
- **cAdvisor**: Container resource monitoring
- **Custom Metrics**: Application-specific metrics

### Visualization
- **Grafana Dashboards**:
  - System Overview
  - Application Performance
  - Database Metrics
  - Security Monitoring
  - Business Intelligence

### Alerting
- **AlertManager**: Centralized alert management
- **Slack Integration**: Real-time notifications
- **Email Notifications**: Critical alerts
- **Custom Alert Rules**: Business-specific monitoring

### Logging
- **Centralized Logging**: Loki for log aggregation
- **Log Parsing**: Structured logs with labels
- **Log Retention**: 31-day retention policy
- **Log Search**: Grafana integration for log exploration

## 💾 Backup & Disaster Recovery

### Automated Backups
- **Daily Database Backups**: Encrypted and compressed
- **Configuration Backups**: All critical configuration files
- **SSL Certificate Backups**: Certificate and key backup
- **Application Data Backups**: Persistent volumes

### Disaster Recovery
- **RTO (Recovery Time Objective)**: 60 minutes
- **RPO (Recovery Point Objective)**: 15 minutes
- **Automated DR Testing**: Weekly validation
- **Runbook Documentation**: Step-by-step procedures

### Backup Commands
```bash
# Create comprehensive backup
sudo ./disaster-recovery.sh backup

# Test disaster recovery procedures  
sudo ./disaster-recovery.sh test-dr

# Restore from backup
sudo ./disaster-recovery.sh restore --backup-file backup_file.tar.gz

# Check system status
sudo ./disaster-recovery.sh status
```

## 🔧 Maintenance Procedures

### Regular Maintenance
- **Daily**: Automated backups and health checks
- **Weekly**: Security updates and certificate monitoring
- **Monthly**: Performance optimization and cleanup
- **Quarterly**: Disaster recovery testing

### Update Procedures
```bash
# Update to new version
sudo ./deploy.sh update --version v2.1.0

# Rollback if needed
sudo ./deploy.sh rollback --force
```

### SSL Certificate Management
```bash
# Setup SSL certificates
sudo ./ssl-setup.sh

# Check certificate status
sudo /usr/local/bin/leanvibe-ssl-monitor

# Manual certificate renewal
sudo /usr/local/bin/leanvibe-ssl-renew
```

## 🚨 Troubleshooting

### Common Issues

1. **Services Not Starting**
   ```bash
   # Check service status
   docker-compose -f docker-compose.production-optimized.yml ps
   
   # View service logs
   docker-compose -f docker-compose.production-optimized.yml logs [service_name]
   ```

2. **Database Connection Issues**
   ```bash
   # Check database health
   docker-compose -f docker-compose.production-optimized.yml exec postgres pg_isready
   
   # Test database connectivity
   docker-compose -f docker-compose.production-optimized.yml exec postgres psql -U leanvibe_user -d leanvibe_agent_hive -c "SELECT 1;"
   ```

3. **SSL Certificate Issues**
   ```bash
   # Check certificate status
   sudo /usr/local/bin/leanvibe-ssl-monitor
   
   # Force certificate renewal
   sudo certbot renew --force-renewal
   ```

4. **Performance Issues**
   ```bash
   # Check resource usage
   docker stats
   
   # View system metrics
   curl -s http://localhost:9090/metrics
   ```

### Log Locations
- **Application Logs**: `/var/log/leanvibe/`
- **Nginx Logs**: `/var/log/nginx/`
- **Docker Logs**: `docker-compose logs`
- **System Logs**: `/var/log/syslog`

## 📈 Performance Optimization

### Database Optimization
- Connection pooling with PgBouncer
- Production-tuned PostgreSQL configuration
- Regular VACUUM and ANALYZE operations
- Query performance monitoring

### Application Optimization
- Horizontal scaling capabilities
- Container resource limits
- Caching strategies with Redis
- Connection keep-alive optimization

### Network Optimization
- HTTP/2 support
- Gzip compression
- Static asset caching
- CDN integration ready

## 🔐 Compliance & Standards

### Security Standards
- OWASP Top 10 compliance
- SSL/TLS best practices
- Container security benchmarks
- Regular security scanning

### Operational Standards
- 99.9% uptime target
- 24/7 monitoring
- Incident response procedures
- Change management process

## 📞 Support & Maintenance

### Monitoring Dashboards
- **Grafana**: http://your-domain.com/grafana/
- **Prometheus**: http://localhost:9090/ (internal)
- **AlertManager**: http://localhost:9093/ (internal)

### Key Contacts
- **Technical Lead**: [Contact Information]
- **Operations Team**: [Contact Information]
- **Security Team**: [Contact Information]

### Documentation Links
- [API Documentation](../../docs/api/)
- [Architecture Documentation](../../docs/architecture/)
- [Security Procedures](../../docs/security/)
- [Incident Response](../../docs/incident-response/)

---

## 🎯 Epic 7 Phase 1 Success Criteria

✅ **All Phase 7.1 objectives have been completed:**

- **Phase 7.1A**: Production database setup with connection pooling and backup procedures
- **Phase 7.1B**: Redis cluster configuration with high availability and monitoring
- **Phase 7.1C**: Nginx reverse proxy with SSL termination and load balancing
- **Phase 7.1D**: Centralized logging and comprehensive backup procedures

**Next Steps**: Ready for Phase 7.2 (User Access & API Deployment)

---

*This production infrastructure setup provides a robust, secure, and scalable foundation for the LeanVibe Agent Hive 2.0 system, ensuring high availability and optimal performance in production environments.*