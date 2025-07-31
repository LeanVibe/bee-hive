# ⚠️ DEPRECATED - Enterprise Deployment & Security Configuration Guide

**NOTICE**: This file has been deprecated and consolidated into `docs/implementation/security-implementation-guide.md`

**Deprecation Date**: 2025-07-31
**Reason**: Content redundancy - consolidated with other security implementation files

**For current security information, please refer to:**
- `SECURITY.md` (root) - External security policy
- `docs/prd/security-auth-system.md` - Technical specifications
- `docs/implementation/security-implementation-guide.md` - Implementation procedures

---

# Enterprise Deployment & Security Configuration Guide

## Overview

This guide provides comprehensive instructions for deploying LeanVibe Agent Hive 2.0 in enterprise environments with advanced security configurations, OAuth 2.0/OIDC integration, and production-grade infrastructure setup.

## Prerequisites

### System Requirements

**Minimum Production Environment:**
- **CPU**: 8 cores (Intel Xeon or AMD EPYC)
- **RAM**: 32GB
- **Storage**: 500GB SSD (NVMe recommended)
- **Network**: 10Gbps
- **OS**: Ubuntu 22.04 LTS or RHEL 9

**Recommended Production Environment:**
- **CPU**: 16 cores (Intel Xeon or AMD EPYC)
- **RAM**: 64GB
- **Storage**: 1TB NVMe SSD
- **Network**: 25Gbps
- **OS**: Ubuntu 22.04 LTS or RHEL 9

### Required Software

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Python 3.12
sudo apt install -y python3.12 python3.12-venv python3.12-dev

# Install PostgreSQL client
sudo apt install -y postgresql-client-14

# Install Redis tools
sudo apt install -y redis-tools

# Install monitoring tools
sudo apt install -y htop iotop netstat-nat
```

## Enterprise Docker Configuration

### Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  # PostgreSQL with pgvector for semantic search
  postgres:
    image: pgvector/pgvector:0.5.1-pg15
    container_name: leanvibe-postgres-prod
    environment:
      POSTGRES_DB: leanvibe_prod
      POSTGRES_USER: leanvibe_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_HOST_AUTH_METHOD: scram-sha-256
      POSTGRES_INITDB_ARGS: "--auth-host=scram-sha-256"
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./config/postgresql.conf:/etc/postgresql/postgresql.conf
      - ./config/pg_hba.conf:/etc/postgresql/pg_hba.conf
      - ./logs/postgres:/var/log/postgresql
    networks:
      - leanvibe-network
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '4'
        reservations:
          memory: 8G
          cpus: '2'
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U leanvibe_user -d leanvibe_prod"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: json-file
      options:
        max-size: "100m"
        max-file: "10"

  # Redis Streams for message bus
  redis:
    image: redis:7.2-alpine
    container_name: leanvibe-redis-prod
    command: redis-server /usr/local/etc/redis/redis.conf
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      - ./config/redis.conf:/usr/local/etc/redis/redis.conf
      - ./logs/redis:/var/log/redis
    networks:
      - leanvibe-network
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '2'
        reservations:
          memory: 4G
          cpus: '1'
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "5"

  # LeanVibe Agent Hive Application
  app:
    build:
      context: .
      dockerfile: Dockerfile.prod
      args:
        BUILD_ENV: production
    container_name: leanvibe-app-prod
    environment:
      # Database Configuration
      DATABASE_URL: postgresql://leanvibe_user:${POSTGRES_PASSWORD}@postgres:5432/leanvibe_prod
      DATABASE_POOL_SIZE: 20
      DATABASE_MAX_OVERFLOW: 10
      
      # Redis Configuration
      REDIS_URL: redis://redis:6379/0
      REDIS_MAX_CONNECTIONS: 100
      
      # Security Configuration
      SECRET_KEY: ${SECRET_KEY}
      JWT_SECRET_KEY: ${JWT_SECRET_KEY}
      JWT_ALGORITHM: RS256
      JWT_ACCESS_TOKEN_EXPIRE_MINUTES: 60
      JWT_REFRESH_TOKEN_EXPIRE_DAYS: 30
      
      # OAuth 2.0/OIDC Configuration
      OAUTH_CLIENT_ID: ${OAUTH_CLIENT_ID}
      OAUTH_CLIENT_SECRET: ${OAUTH_CLIENT_SECRET}
      OAUTH_ISSUER_URL: ${OAUTH_ISSUER_URL}
      OAUTH_REDIRECT_URI: ${OAUTH_REDIRECT_URI}
      
      # GitHub Integration
      GITHUB_CLIENT_ID: ${GITHUB_CLIENT_ID}
      GITHUB_CLIENT_SECRET: ${GITHUB_CLIENT_SECRET}
      GITHUB_WEBHOOK_SECRET: ${GITHUB_WEBHOOK_SECRET}
      
      # Monitoring
      PROMETHEUS_ENABLED: true
      PROMETHEUS_PORT: 9090
      GRAFANA_ENABLED: true
      
      # Logging
      LOG_LEVEL: INFO
      LOG_FORMAT: json
      STRUCTURED_LOGGING: true
      
      # Performance
      WORKER_PROCESSES: 4
      WORKER_CONNECTIONS: 1000
      
      # Security Headers
      SECURITY_HEADERS_ENABLED: true
      CORS_ORIGINS: ${CORS_ORIGINS}
      
    ports:
      - "8000:8000"
      - "9090:9090"  # Prometheus metrics
    volumes:
      - ./logs/app:/app/logs
      - ./workspaces:/app/workspaces
      - ./checkpoints:/app/checkpoints
      - ./config/ssl:/app/ssl
    networks:
      - leanvibe-network
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '8'
        reservations:
          memory: 8G
          cpus: '4'
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging:
      driver: json-file
      options:
        max-size: "200m"
        max-file: "20"

  # Nginx Reverse Proxy with SSL
  nginx:
    image: nginx:1.24-alpine
    container_name: leanvibe-nginx-prod
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./config/nginx/sites-enabled:/etc/nginx/sites-enabled
      - ./config/ssl:/etc/nginx/ssl
      - ./logs/nginx:/var/log/nginx
    networks:
      - leanvibe-network
    depends_on:
      - app
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Prometheus for Metrics Collection
  prometheus:
    image: prom/prometheus:v2.47.0
    container_name: leanvibe-prometheus-prod
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    ports:
      - "9091:9090"
    volumes:
      - ./config/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    networks:
      - leanvibe-network
    restart: unless-stopped

  # Grafana for Monitoring Dashboards
  grafana:
    image: grafana/grafana:10.1.0
    container_name: leanvibe-grafana-prod
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD}
      GF_SECURITY_ADMIN_USER: admin
      GF_USERS_ALLOW_SIGN_UP: false
      GF_SECURITY_ALLOW_EMBEDDING: true
      GF_AUTH_ANONYMOUS_ENABLED: false
      GF_SECURITY_COOKIE_SECURE: true
      GF_SECURITY_COOKIE_SAMESITE: strict
    ports:
      - "3001:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./config/grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - leanvibe-network
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local

networks:
  leanvibe-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Production Environment Configuration

Create `.env.prod`:

```bash
# Database Configuration
POSTGRES_PASSWORD=your_super_secure_postgres_password_here_32_chars_min

# Application Security
SECRET_KEY=your_secret_key_for_app_encryption_64_chars_minimum_recommended
JWT_SECRET_KEY=your_jwt_secret_key_for_token_signing_64_chars_minimum_here
JWT_PRIVATE_KEY_PATH=/app/ssl/jwt_private_key.pem
JWT_PUBLIC_KEY_PATH=/app/ssl/jwt_public_key.pem

# OAuth 2.0/OIDC Configuration
OAUTH_CLIENT_ID=leanvibe-agent-hive-production
OAUTH_CLIENT_SECRET=your_oauth_client_secret_from_provider
OAUTH_ISSUER_URL=https://your-oauth-provider.com
OAUTH_REDIRECT_URI=https://your-domain.com/auth/callback
OAUTH_SCOPES=openid,profile,email,groups

# GitHub Integration
GITHUB_CLIENT_ID=your_github_app_client_id
GITHUB_CLIENT_SECRET=your_github_app_client_secret
GITHUB_WEBHOOK_SECRET=your_github_webhook_secret_key
GITHUB_PRIVATE_KEY_PATH=/app/ssl/github_private_key.pem

# CORS Configuration
CORS_ORIGINS=https://your-domain.com,https://dashboard.your-domain.com

# Monitoring
GRAFANA_ADMIN_PASSWORD=your_secure_grafana_admin_password

# SSL/TLS Configuration
SSL_CERT_PATH=/app/ssl/certificate.crt
SSL_KEY_PATH=/app/ssl/private.key
SSL_CA_PATH=/app/ssl/ca_bundle.crt

# Encryption Keys
ENCRYPTION_KEY=your_32_byte_encryption_key_for_secrets_storage_here
AUDIT_LOG_SIGNING_KEY=your_audit_log_signing_key_for_immutable_logs
```

## Security Configuration

### PostgreSQL Security Configuration

Create `config/postgresql.conf`:

```ini
# PostgreSQL Production Configuration with Security Hardening

# Connection Settings
listen_addresses = '*'
port = 5432
max_connections = 200
superuser_reserved_connections = 3

# SSL Configuration
ssl = on
ssl_cert_file = '/etc/ssl/certs/server.crt'
ssl_key_file = '/etc/ssl/private/server.key'
ssl_ca_file = '/etc/ssl/certs/ca.crt'
ssl_prefer_server_ciphers = on
ssl_ciphers = 'ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384'
ssl_min_protocol_version = 'TLSv1.2'

# Authentication
password_encryption = scram-sha-256
auth_delay.milliseconds = 2000

# Memory Configuration
shared_buffers = 8GB
effective_cache_size = 24GB
work_mem = 256MB
maintenance_work_mem = 2GB

# WAL Configuration
wal_level = replica
max_wal_size = 4GB
min_wal_size = 1GB
checkpoint_completion_target = 0.9

# Query Performance
random_page_cost = 1.1
effective_io_concurrency = 200

# Security Settings
log_connections = on
log_disconnections = on
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_checkpoints = on
log_lock_waits = on
log_min_error_statement = error
log_min_messages = warning
log_statement = 'ddl'

# Prevent SQL Injection
standard_conforming_strings = on
escape_string_warning = on

# Resource Limits
max_files_per_process = 4000
max_locks_per_transaction = 256
```

Create `config/pg_hba.conf`:

```ini
# PostgreSQL Host-Based Authentication Configuration

# TYPE  DATABASE        USER            ADDRESS                 METHOD

# Local connections
local   all             postgres                                peer
local   all             all                                     scram-sha-256

# IPv4 local connections
host    leanvibe_prod   leanvibe_user   172.20.0.0/16          scram-sha-256
host    leanvibe_prod   leanvibe_user   127.0.0.1/32           scram-sha-256

# IPv6 local connections
host    leanvibe_prod   leanvibe_user   ::1/128                scram-sha-256

# Reject all other connections
host    all             all             0.0.0.0/0              reject
```

### Redis Security Configuration

Create `config/redis.conf`:

```ini
# Redis Production Configuration with Security Hardening

# Network Security
bind 127.0.0.1 172.20.0.0/16
protected-mode yes
port 6379

# Authentication
requirepass your_redis_password_here_minimum_32_characters_long
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command EVAL ""
rename-command DEBUG ""
rename-command CONFIG "CONFIG_b835d3c4e5f6a7b8c9d0e1f2"

# SSL/TLS Configuration
tls-port 6380
tls-cert-file /etc/redis/ssl/redis.crt
tls-key-file /etc/redis/ssl/redis.key
tls-ca-cert-file /etc/redis/ssl/ca.crt
tls-ciphers ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256
tls-protocols TLSv1.2 TLSv1.3

# Memory Configuration
maxmemory 6gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# Persistence Configuration
save 900 1
save 300 10
save 60 10000
rdbcompression yes
rdbchecksum yes
dbfilename leanvibe-prod.rdb
dir /data

# AOF Configuration
appendonly yes
appendfilename "leanvibe-prod.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Security Logging
syslog-enabled yes
syslog-ident redis-prod
loglevel notice
logfile /var/log/redis/redis-server.log

# Performance Tuning
tcp-keepalive 300
timeout 300
tcp-backlog 511
databases 16

# Client Output Buffer Limits
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60

# Slow Log
slowlog-log-slower-than 10000
slowlog-max-len 128
```

### Nginx Security Configuration

Create `config/nginx/nginx.conf`:

```nginx
# Nginx Production Configuration with Security Hardening

user nginx;
worker_processes auto;
worker_rlimit_nofile 65535;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    # Security Headers
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self'; connect-src 'self' wss:" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

    # Hide Nginx Version
    server_tokens off;

    # Logging Configuration
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                   '$status $body_bytes_sent "$http_referer" '
                   '"$http_user_agent" "$http_x_forwarded_for" '
                   'rt=$request_time uct="$upstream_connect_time" '
                   'uht="$upstream_header_time" urt="$upstream_response_time"';

    access_log /var/log/nginx/access.log main;

    # Performance Optimization
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 100M;
    client_body_timeout 60s;
    client_header_timeout 60s;
    send_timeout 60s;

    # Gzip Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        application/atom+xml
        application/javascript
        application/json
        application/ld+json
        application/manifest+json
        application/rss+xml
        application/vnd.geo+json
        application/vnd.ms-fontobject
        application/x-font-ttf
        application/x-web-app-manifest+json
        application/xhtml+xml
        application/xml
        font/opentype
        image/bmp
        image/svg+xml
        image/x-icon
        text/cache-manifest
        text/css
        text/plain
        text/vcard
        text/vnd.rim.location.xloc
        text/vtt
        text/x-component
        text/x-cross-domain-policy;

    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/s;
    limit_req_zone $binary_remote_addr zone=webhooks:10m rate=100r/s;

    # SSL Configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-SHA384:ECDHE-RSA-AES128-SHA256;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_stapling on;
    ssl_stapling_verify on;

    include /etc/nginx/sites-enabled/*;
}
```

Create `config/nginx/sites-enabled/leanvibe.conf`:

```nginx
# LeanVibe Agent Hive Virtual Host Configuration

upstream leanvibe_app {
    least_conn;
    server app:8000 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

upstream grafana {
    server grafana:3000;
}

# HTTP to HTTPS Redirect
server {
    listen 80;
    server_name your-domain.com dashboard.your-domain.com;
    return 301 https://$server_name$request_uri;
}

# Main Application Server
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/certificate.crt;
    ssl_certificate_key /etc/nginx/ssl/private.key;
    ssl_trusted_certificate /etc/nginx/ssl/ca_bundle.crt;

    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;

    # API Endpoints
    location /api/ {
        limit_req zone=api burst=20 nodelay;
        
        proxy_pass http://leanvibe_app;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Security Headers
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Server $host;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Authentication Endpoints (Stricter Rate Limiting)
    location /api/v1/auth/ {
        limit_req zone=auth burst=10 nodelay;
        
        proxy_pass http://leanvibe_app;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Additional Security for Auth
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Server $host;
    }

    # Webhook Endpoints
    location /webhooks/ {
        limit_req zone=webhooks burst=200 nodelay;
        
        proxy_pass http://leanvibe_app;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Webhook-specific headers
        proxy_set_header X-GitHub-Delivery $http_x_github_delivery;
        proxy_set_header X-GitHub-Event $http_x_github_event;
        proxy_set_header X-Hub-Signature-256 $http_x_hub_signature_256;
    }

    # WebSocket Support
    location /ws/ {
        proxy_pass http://leanvibe_app;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket-specific timeouts
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
    }

    # Health Check Endpoint
    location /health {
        access_log off;
        proxy_pass http://leanvibe_app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Static Files (if any)
    location /static/ {
        alias /app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
    }

    # Deny access to sensitive files
    location ~ /\. {
        deny all;
        access_log off;
        log_not_found off;
    }

    location ~ \.(env|yml|yaml|conf)$ {
        deny all;
        access_log off;
        log_not_found off;
    }
}

# Grafana Dashboard Server
server {
    listen 443 ssl http2;
    server_name dashboard.your-domain.com;

    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/certificate.crt;
    ssl_certificate_key /etc/nginx/ssl/private.key;
    ssl_trusted_certificate /etc/nginx/ssl/ca_bundle.crt;

    # Grafana Proxy
    location / {
        proxy_pass http://grafana;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## SSL/TLS Certificate Configuration

### Generate Self-Signed Certificates (Development)

```bash
#!/bin/bash
# generate_ssl_certs.sh

# Create SSL directory
mkdir -p config/ssl

# Generate private key
openssl genrsa -out config/ssl/private.key 4096

# Generate certificate signing request
openssl req -new -key config/ssl/private.key -out config/ssl/certificate.csr -subj "/C=US/ST=State/L=City/O=LeanVibe/OU=Agent Hive/CN=your-domain.com"

# Generate self-signed certificate
openssl x509 -req -days 365 -in config/ssl/certificate.csr -signkey config/ssl/private.key -out config/ssl/certificate.crt

# Generate JWT signing keys
openssl genrsa -out config/ssl/jwt_private_key.pem 4096
openssl rsa -in config/ssl/jwt_private_key.pem -pubout -out config/ssl/jwt_public_key.pem

# Generate GitHub App private key (placeholder - replace with actual GitHub App key)
cp config/ssl/jwt_private_key.pem config/ssl/github_private_key.pem

# Set proper permissions
chmod 600 config/ssl/*.key config/ssl/*.pem
chmod 644 config/ssl/*.crt config/ssl/*.csr

echo "SSL certificates generated successfully!"
```

### Production SSL Certificate Setup (Let's Encrypt)

```bash
#!/bin/bash
# setup_letsencrypt.sh

# Install Certbot
sudo apt update
sudo apt install -y certbot python3-certbot-nginx

# Generate certificates
sudo certbot --nginx -d your-domain.com -d dashboard.your-domain.com

# Setup auto-renewal
sudo crontab -e
# Add this line:
# 0 12 * * * /usr/bin/certbot renew --quiet --renew-hook "docker-compose -f /path/to/docker-compose.prod.yml restart nginx"
```

## OAuth 2.0/OIDC Integration

### Keycloak Integration Example

Create `config/keycloak-integration.json`:

```json
{
  "oauth_provider": "keycloak",
  "configuration": {
    "issuer_url": "https://your-keycloak.company.com/auth/realms/leanvibe",
    "client_id": "leanvibe-agent-hive",
    "client_secret": "your-keycloak-client-secret",
    "redirect_uri": "https://your-domain.com/auth/callback",
    "scopes": ["openid", "profile", "email", "groups", "roles"],
    "authorization_endpoint": "https://your-keycloak.company.com/auth/realms/leanvibe/protocol/openid-connect/auth",
    "token_endpoint": "https://your-keycloak.company.com/auth/realms/leanvibe/protocol/openid-connect/token",
    "userinfo_endpoint": "https://your-keycloak.company.com/auth/realms/leanvibe/protocol/openid-connect/userinfo",
    "jwks_uri": "https://your-keycloak.company.com/auth/realms/leanvibe/protocol/openid-connect/certs",
    "end_session_endpoint": "https://your-keycloak.company.com/auth/realms/leanvibe/protocol/openid-connect/logout"
  },
  "role_mapping": {
    "admin": ["orchestrator.admin", "security.admin", "github.admin"],
    "senior-agent": ["orchestrator.manage", "github.repository.write", "context.compress"],
    "agent": ["orchestrator.read", "github.repository.read", "context.read"],
    "viewer": ["orchestrator.read", "github.repository.read"]
  },
  "security_settings": {
    "enforce_ssl": true,
    "token_validation": "strict",
    "session_timeout": 3600,
    "refresh_token_rotation": true,
    "pkce_required": true,
    "state_parameter_required": true
  }
}
```

### Azure AD Integration Example

Create `config/azure-ad-integration.json`:

```json
{
  "oauth_provider": "azure_ad",
  "configuration": {
    "tenant_id": "your-azure-tenant-id",
    "client_id": "your-azure-app-registration-id",
    "client_secret": "your-azure-client-secret",
    "redirect_uri": "https://your-domain.com/auth/callback",
    "scopes": ["openid", "profile", "email", "User.Read", "Directory.Read.All"],
    "authority": "https://login.microsoftonline.com/your-azure-tenant-id",
    "discovery_url": "https://login.microsoftonline.com/your-azure-tenant-id/v2.0/.well-known/openid_configuration"
  },
  "group_mapping": {
    "LeanVibe-Admins": ["orchestrator.admin", "security.admin", "github.admin"],
    "LeanVibe-Senior-Agents": ["orchestrator.manage", "github.repository.write", "context.compress"],
    "LeanVibe-Agents": ["orchestrator.read", "github.repository.read", "context.read"],
    "LeanVibe-Viewers": ["orchestrator.read", "github.repository.read"]
  },
  "security_settings": {
    "enforce_ssl": true,
    "validate_issuer": true,
    "validate_audience": true,
    "require_signed_tokens": true,
    "clock_skew_tolerance": 300,
    "token_cache_duration": 3600
  }
}
```

## Monitoring and Alerting Configuration

### Prometheus Configuration

Create `config/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'leanvibe-prod'
    replica: '1'

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # LeanVibe Application Metrics
  - job_name: 'leanvibe-app'
    static_configs:
      - targets: ['app:9090']
    scrape_interval: 10s
    metrics_path: '/metrics'
    
  # PostgreSQL Metrics
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres_exporter:9187']
    
  # Redis Metrics
  - job_name: 'redis'
    static_configs:
      - targets: ['redis_exporter:9121']
    
  # Nginx Metrics
  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx_exporter:9113']
    
  # Node Metrics
  - job_name: 'node'
    static_configs:
      - targets: ['node_exporter:9100']
```

### Alert Rules

Create `config/prometheus/rules/leanvibe-alerts.yml`:

```yaml
groups:
  - name: leanvibe-application
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ $value }}s"

      - alert: AgentAuthenticationFailure
        expr: rate(agent_authentication_failures_total[5m]) > 0.2
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High agent authentication failure rate"
          description: "Authentication failure rate is {{ $value }} per second"

  - name: leanvibe-infrastructure
    rules:
      - alert: PostgreSQLDown
        expr: pg_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL is down"
          description: "PostgreSQL database is not responding"

      - alert: RedisDown
        expr: redis_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis is down"
          description: "Redis server is not responding"

      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }}"
```

## Deployment Scripts

### Production Deployment Script

Create `deploy-production.sh`:

```bash
#!/bin/bash
# Production Deployment Script for LeanVibe Agent Hive

set -e

# Configuration
DEPLOY_ENV="production"
PROJECT_DIR="/opt/leanvibe-agent-hive"
BACKUP_DIR="/opt/backups/leanvibe"
LOG_FILE="/var/log/leanvibe-deploy.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_info() {
    log "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    log "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

# Pre-deployment checks
pre_deployment_checks() {
    log_info "Running pre-deployment checks..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running!"
        exit 1
    fi
    
    # Check if required files exist
    required_files=(
        ".env.prod"
        "docker-compose.prod.yml"
        "config/ssl/certificate.crt"
        "config/ssl/private.key"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file $file not found!"
            exit 1
        fi
    done
    
    # Check available disk space
    available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [[ $available_space -lt 10 ]]; then
        log_error "Insufficient disk space! At least 10GB required."
        exit 1
    fi
    
    log_info "Pre-deployment checks passed!"
}

# Database backup
backup_database() {
    log_info "Creating database backup..."
    
    mkdir -p "$BACKUP_DIR"
    backup_file="$BACKUP_DIR/leanvibe_$(date +%Y%m%d_%H%M%S).sql"
    
    docker-compose -f docker-compose.prod.yml exec -T postgres \
        pg_dump -U leanvibe_user leanvibe_prod > "$backup_file"
    
    if [[ $? -eq 0 ]]; then
        log_info "Database backup created: $backup_file"
        # Compress backup
        gzip "$backup_file"
        log_info "Backup compressed: ${backup_file}.gz"
    else
        log_error "Database backup failed!"
        exit 1
    fi
}

# Pull latest images
pull_images() {
    log_info "Pulling latest Docker images..."
    docker-compose -f docker-compose.prod.yml pull
}

# Stop services gracefully
stop_services() {
    log_info "Stopping services gracefully..."
    docker-compose -f docker-compose.prod.yml down --timeout 30
}

# Start services
start_services() {
    log_info "Starting services..."
    docker-compose -f docker-compose.prod.yml up -d
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Wait for database to be ready
    log_info "Waiting for database to be ready..."
    docker-compose -f docker-compose.prod.yml exec -T app \
        python -c "
import time
import psycopg2
import os

max_attempts = 30
attempt = 0

while attempt < max_attempts:
    try:
        conn = psycopg2.connect(os.environ['DATABASE_URL'])
        conn.close()
        print('Database is ready!')
        break
    except psycopg2.OperationalError:
        attempt += 1
        print(f'Database not ready, attempt {attempt}/{max_attempts}')
        time.sleep(10)
else:
    print('Database failed to become ready!')
    exit(1)
"
    
    # Run migrations
    docker-compose -f docker-compose.prod.yml exec -T app \
        alembic upgrade head
    
    if [[ $? -eq 0 ]]; then
        log_info "Database migrations completed successfully!"
    else
        log_error "Database migrations failed!"
        exit 1
    fi
}

# Health checks
health_checks() {
    log_info "Running health checks..."
    
    max_attempts=30
    attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -f -s http://localhost:8000/health >/dev/null; then
            log_info "Application health check passed!"
            return 0
        fi
        
        attempt=$((attempt + 1))
        log_warn "Health check attempt $attempt/$max_attempts failed, retrying in 10 seconds..."
        sleep 10
    done
    
    log_error "Application health checks failed!"
    return 1
}

# Post-deployment verification
post_deployment_verification() {
    log_info "Running post-deployment verification..."
    
    # Check all containers are running
    failed_containers=$(docker-compose -f docker-compose.prod.yml ps -q | xargs docker inspect -f '{{.Name}} {{.State.Status}}' | grep -v "running" || true)
    
    if [[ -n "$failed_containers" ]]; then
        log_error "Some containers are not running:"
        echo "$failed_containers"
        return 1
    fi
    
    # Test API authentication
    log_info "Testing API authentication..."
    auth_response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/v1/auth/health)
    if [[ "$auth_response" != "200" ]]; then
        log_error "API authentication endpoint failed (HTTP $auth_response)"
        return 1
    fi
    
    # Test database connection
    log_info "Testing database connection..."
    db_test=$(docker-compose -f docker-compose.prod.yml exec -T app python -c "
import os
import psycopg2
try:
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    cur = conn.cursor()
    cur.execute('SELECT 1')
    result = cur.fetchone()
    conn.close()
    print('OK' if result[0] == 1 else 'FAIL')
except Exception as e:
    print('FAIL')
")
    
    if [[ "$db_test" != "OK" ]]; then
        log_error "Database connection test failed"
        return 1
    fi
    
    log_info "Post-deployment verification completed successfully!"
    return 0
}

# Rollback function
rollback() {
    log_error "Deployment failed! Initiating rollback..."
    
    # Stop current services
    docker-compose -f docker-compose.prod.yml down --timeout 30
    
    # Restore database from backup
    latest_backup=$(ls -t "$BACKUP_DIR"/*.sql.gz 2>/dev/null | head -1)
    if [[ -n "$latest_backup" ]]; then
        log_info "Restoring database from backup: $latest_backup"
        gunzip -c "$latest_backup" | docker-compose -f docker-compose.prod.yml exec -T postgres \
            psql -U leanvibe_user -d leanvibe_prod
    fi
    
    # Start services with previous version
    docker-compose -f docker-compose.prod.yml up -d
    
    log_error "Rollback completed. Please investigate the deployment failure."
    exit 1
}

# Main deployment process
main() {
    log_info "Starting LeanVibe Agent Hive production deployment..."
    
    # Trap errors for rollback
    trap rollback ERR
    
    pre_deployment_checks
    backup_database
    pull_images
    stop_services
    start_services
    run_migrations
    
    if ! health_checks; then
        rollback
    fi
    
    if ! post_deployment_verification; then
        rollback
    fi
    
    log_info "Deployment completed successfully!"
    log_info "Application is available at: https://your-domain.com"
    log_info "Dashboard is available at: https://dashboard.your-domain.com"
}

# Run main function
main "$@"
```

### Security Hardening Script

Create `security-hardening.sh`:

```bash
#!/bin/bash
# Security Hardening Script for LeanVibe Agent Hive

set -e

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install security tools
sudo apt install -y fail2ban ufw lynis rkhunter chkrootkit

# Configure UFW Firewall
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (adjust port as needed)
sudo ufw allow 22/tcp

# Allow HTTP and HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow specific application ports
sudo ufw allow 8000/tcp  # LeanVibe App
sudo ufw allow 3001/tcp  # Grafana
sudo ufw allow 9091/tcp  # Prometheus

# Enable firewall
sudo ufw --force enable

# Configure Fail2Ban
sudo tee /etc/fail2ban/jail.local > /dev/null <<EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
maxretry = 3

[nginx-http-auth]
enabled = true
filter = nginx-http-auth
port = http,https
logpath = /var/log/nginx/error.log

[nginx-noscript]
enabled = true
port = http,https
filter = nginx-noscript
logpath = /var/log/nginx/access.log
maxretry = 6
EOF

# Start and enable Fail2Ban
sudo systemctl start fail2ban
sudo systemctl enable fail2ban

# Secure shared memory
echo "tmpfs /run/shm tmpfs defaults,noexec,nosuid 0 0" | sudo tee -a /etc/fstab

# Disable unused network protocols
echo "install dccp /bin/true" | sudo tee -a /etc/modprobe.d/blacklist-rare-network.conf
echo "install sctp /bin/true" | sudo tee -a /etc/modprobe.d/blacklist-rare-network.conf
echo "install rds /bin/true" | sudo tee -a /etc/modprobe.d/blacklist-rare-network.conf
echo "install tipc /bin/true" | sudo tee -a /etc/modprobe.d/blacklist-rare-network.conf

# Set kernel parameters for security
sudo tee /etc/sysctl.d/99-security.conf > /dev/null <<EOF
# IP Spoofing protection
net.ipv4.conf.default.rp_filter = 1
net.ipv4.conf.all.rp_filter = 1

# Ignore ICMP redirects
net.ipv4.conf.all.accept_redirects = 0
net.ipv6.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0

# Ignore send redirects
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0

# Disable source packet routing
net.ipv4.conf.all.accept_source_route = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv6.conf.default.accept_source_route = 0

# Log Martians
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1

# Ignore ICMP ping requests
net.ipv4.icmp_echo_ignore_all = 1

# Ignore Directed pings
net.ipv4.icmp_echo_ignore_broadcasts = 1

# Disable IPv6 if not needed
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
net.ipv6.conf.lo.disable_ipv6 = 1

# TCP SYN Cookies
net.ipv4.tcp_syncookies = 1

# Controls the maximum size of a message, in bytes
kernel.msgmnb = 65536

# Controls the default maxmimum size of a mesage queue
kernel.msgmax = 65536

# Restrict core dumps
fs.suid_dumpable = 0

# Hide kernel pointers
kernel.kptr_restrict = 1
EOF

# Apply sysctl settings
sudo sysctl -p /etc/sysctl.d/99-security.conf

# Set file permissions for Docker configurations
sudo chmod 600 .env.prod
sudo chmod 644 docker-compose.prod.yml
sudo chmod -R 644 config/
sudo chmod 600 config/ssl/*.key config/ssl/*.pem

# Create log rotation for LeanVibe logs
sudo tee /etc/logrotate.d/leanvibe > /dev/null <<EOF
/opt/leanvibe-agent-hive/logs/*/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
    postrotate
        docker-compose -f /opt/leanvibe-agent-hive/docker-compose.prod.yml restart nginx
    endscript
}
EOF

echo "Security hardening completed!"
echo "Please reboot the system to ensure all changes take effect."
```

## Monitoring Setup

### System Monitoring Script

Create `setup-monitoring.sh`:

```bash
#!/bin/bash
# Monitoring Setup Script

# Install Node Exporter
wget https://github.com/prometheus/node_exporter/releases/latest/download/node_exporter-1.6.1.linux-amd64.tar.gz
tar xvfz node_exporter-1.6.1.linux-amd64.tar.gz
sudo mv node_exporter-1.6.1.linux-amd64/node_exporter /usr/local/bin/
sudo useradd --no-create-home --shell /bin/false node_exporter
sudo chown node_exporter:node_exporter /usr/local/bin/node_exporter

# Create systemd service for Node Exporter
sudo tee /etc/systemd/system/node_exporter.service > /dev/null <<EOF
[Unit]
Description=Node Exporter
Wants=network-online.target
After=network-online.target

[Service]
User=node_exporter
Group=node_exporter
Type=simple
ExecStart=/usr/local/bin/node_exporter --collector.systemd --collector.processes

[Install]
WantedBy=multi-user.target
EOF

# Start and enable Node Exporter
sudo systemctl daemon-reload
sudo systemctl start node_exporter
sudo systemctl enable node_exporter

# Install Postgres Exporter
wget https://github.com/prometheus-community/postgres_exporter/releases/latest/download/postgres_exporter-0.13.2.linux-amd64.tar.gz
tar xvfz postgres_exporter-0.13.2.linux-amd64.tar.gz
sudo mv postgres_exporter-0.13.2.linux-amd64/postgres_exporter /usr/local/bin/
sudo useradd --no-create-home --shell /bin/false postgres_exporter
sudo chown postgres_exporter:postgres_exporter /usr/local/bin/postgres_exporter

# Create environment file for Postgres Exporter
sudo tee /etc/default/postgres_exporter > /dev/null <<EOF
DATA_SOURCE_NAME="postgresql://leanvibe_user:${POSTGRES_PASSWORD}@localhost:5432/leanvibe_prod?sslmode=disable"
EOF

# Create systemd service for Postgres Exporter
sudo tee /etc/systemd/system/postgres_exporter.service > /dev/null <<EOF
[Unit]
Description=Postgres Exporter
Wants=network-online.target
After=network-online.target

[Service]
User=postgres_exporter
Group=postgres_exporter
Type=simple
EnvironmentFile=/etc/default/postgres_exporter
ExecStart=/usr/local/bin/postgres_exporter

[Install]
WantedBy=multi-user.target
EOF

# Start and enable Postgres Exporter
sudo systemctl daemon-reload
sudo systemctl start postgres_exporter
sudo systemctl enable postgres_exporter

# Install Redis Exporter
wget https://github.com/oliver006/redis_exporter/releases/latest/download/redis_exporter-v1.54.0.linux-amd64.tar.gz
tar xvfz redis_exporter-v1.54.0.linux-amd64.tar.gz
sudo mv redis_exporter-v1.54.0.linux-amd64/redis_exporter /usr/local/bin/
sudo useradd --no-create-home --shell /bin/false redis_exporter
sudo chown redis_exporter:redis_exporter /usr/local/bin/redis_exporter

# Create systemd service for Redis Exporter
sudo tee /etc/systemd/system/redis_exporter.service > /dev/null <<EOF
[Unit]
Description=Redis Exporter
Wants=network-online.target
After=network-online.target

[Service]
User=redis_exporter
Group=redis_exporter
Type=simple
ExecStart=/usr/local/bin/redis_exporter -redis.addr=redis://localhost:6379

[Install]
WantedBy=multi-user.target
EOF

# Start and enable Redis Exporter
sudo systemctl daemon-reload
sudo systemctl start redis_exporter
sudo systemctl enable redis_exporter

echo "Monitoring setup completed!"
```

## Conclusion

This comprehensive enterprise deployment guide provides:

1. **Production-ready Docker configuration** with security hardening
2. **OAuth 2.0/OIDC integration** with major identity providers
3. **Advanced security configurations** for all components
4. **SSL/TLS certificate management** for production environments
5. **Monitoring and alerting setup** with Prometheus and Grafana
6. **Automated deployment scripts** with rollback capabilities
7. **Security hardening procedures** following industry best practices

The LeanVibe Agent Hive 2.0 platform is now ready for enterprise deployment with comprehensive security, monitoring, and reliability features.