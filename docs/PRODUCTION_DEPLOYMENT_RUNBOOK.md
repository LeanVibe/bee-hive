# LeanVibe Agent Hive Production Deployment Runbook

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Environment Configuration](#environment-configuration)
4. [Database Setup](#database-setup)
5. [Application Deployment](#application-deployment)
6. [Post-Deployment Verification](#post-deployment-verification)
7. [Monitoring Setup](#monitoring-setup)
8. [Backup and Recovery](#backup-and-recovery)
9. [Scaling Procedures](#scaling-procedures)
10. [Maintenance Procedures](#maintenance-procedures)
11. [Emergency Procedures](#emergency-procedures)
12. [Rollback Procedures](#rollback-procedures)

## Pre-Deployment Checklist

### System Requirements Verification

#### Minimum Hardware Requirements

**Application Server (Single Instance):**
```
CPU: 4 vCPUs (Intel/AMD x64)
Memory: 8 GB RAM
Storage: 100 GB SSD
Network: 1 Gbps connection
OS: Ubuntu 20.04 LTS or CentOS 8+
```

**Database Server:**
```
CPU: 4 vCPUs
Memory: 16 GB RAM
Storage: 500 GB SSD (RAID 10 recommended)
Network: 1 Gbps connection
Backup Storage: 1 TB for database backups
```

**Redis Cache Server:**
```
CPU: 2 vCPUs
Memory: 8 GB RAM
Storage: 50 GB SSD
Network: 1 Gbps connection
```

#### Production Hardware Requirements

**Load Balanced Setup (Recommended):**
```
Application Servers: 3x instances (behind load balancer)
Database: PostgreSQL cluster (Primary + Read Replica)
Cache: Redis cluster (3 nodes)
Load Balancer: HAProxy or cloud-native (AWS ALB, GCP LB)
```

### Software Dependencies

#### Required Software Versions

```bash
# Python and Dependencies
Python: 3.11.0+
pip: 23.0+
Poetry: 1.5.0+ (for dependency management)

# Database
PostgreSQL: 15.0+
pgvector extension: 0.5.0+
Redis: 7.0+

# Container Runtime
Docker: 24.0+
Docker Compose: 2.20+

# Web Server
Nginx: 1.22+ (for reverse proxy)

# Monitoring
Prometheus: 2.40+
Grafana: 9.0+
Node Exporter: 1.5+
```

#### Security Requirements

```bash
# SSL/TLS
Valid SSL certificates (Let's Encrypt or commercial)
TLS 1.2+ only
Strong cipher suites configured

# Network Security
Firewall configured (iptables/ufw or cloud security groups)
VPN access for administrative tasks
Network segmentation between tiers

# Access Control
SSH key-based authentication only
Sudo access limited to authorized users
Service accounts with minimal privileges
```

### Pre-Deployment Validation

#### Code Quality Gates

```bash
# Run complete test suite
pytest -v --cov=app --cov-report=html tests/

# Security scanning
bandit -r app/
safety check

# Code quality
flake8 app/
mypy app/

# Frontend tests
cd frontend && npm test
cd mobile-pwa && npm test

# Integration tests
pytest tests/integration/ -v
```

#### Infrastructure Validation

```bash
# Verify connectivity
ping database-server
ping redis-server
ping monitoring-server

# Check resource availability
df -h  # Disk space
free -h  # Memory
nproc  # CPU cores

# Verify DNS resolution
nslookup your-domain.com
nslookup api.your-domain.com
```

## Infrastructure Setup

### Cloud Infrastructure (AWS Example)

#### VPC and Network Configuration

```yaml
# terraform/vpc.tf
resource "aws_vpc" "leanvibe_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostname  = true
  enable_dns_support   = true
  
  tags = {
    Name = "leanvibe-production-vpc"
    Environment = "production"
  }
}

# Public subnets for load balancers
resource "aws_subnet" "public_subnets" {
  count             = 2
  vpc_id            = aws_vpc.leanvibe_vpc.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  map_public_ip_on_launch = true
  
  tags = {
    Name = "leanvibe-public-subnet-${count.index + 1}"
    Type = "Public"
  }
}

# Private subnets for applications
resource "aws_subnet" "private_subnets" {
  count             = 2
  vpc_id            = aws_vpc.leanvibe_vpc.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {
    Name = "leanvibe-private-subnet-${count.index + 1}"
    Type = "Private"
  }
}

# Database subnets
resource "aws_subnet" "database_subnets" {
  count             = 2
  vpc_id            = aws_vpc.leanvibe_vpc.id
  cidr_block        = "10.0.${count.index + 20}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {
    Name = "leanvibe-database-subnet-${count.index + 1}"
    Type = "Database"
  }
}
```

#### Security Groups

```yaml
# Application Security Group
resource "aws_security_group" "app_sg" {
  name_prefix = "leanvibe-app-"
  vpc_id      = aws_vpc.leanvibe_vpc.id

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    security_groups = [aws_security_group.alb_sg.id]
  }

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]  # VPC only
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "leanvibe-app-security-group"
  }
}

# Database Security Group
resource "aws_security_group" "db_sg" {
  name_prefix = "leanvibe-db-"
  vpc_id      = aws_vpc.leanvibe_vpc.id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    security_groups = [aws_security_group.app_sg.id]
  }

  tags = {
    Name = "leanvibe-database-security-group"
  }
}
```

#### RDS Database Setup

```yaml
# RDS Subnet Group
resource "aws_db_subnet_group" "leanvibe_db_subnet_group" {
  name       = "leanvibe-db-subnet-group"
  subnet_ids = aws_subnet.database_subnets[*].id

  tags = {
    Name = "LeanVibe DB subnet group"
  }
}

# RDS Instance
resource "aws_db_instance" "leanvibe_db" {
  identifier = "leanvibe-production-db"
  
  # Database Configuration
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = "db.t3.large"
  
  # Storage
  allocated_storage     = 500
  max_allocated_storage = 1000
  storage_type         = "gp3"
  storage_encrypted    = true
  
  # Database Settings
  db_name  = "leanvibe_production"
  username = "leanvibe_admin"
  password = var.db_password
  
  # Network
  vpc_security_group_ids = [aws_security_group.db_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.leanvibe_db_subnet_group.name
  
  # Backup
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  # Monitoring
  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring_role.arn
  
  # Security
  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "leanvibe-db-final-snapshot"
  
  tags = {
    Name = "LeanVibe Production Database"
    Environment = "production"
  }
}

# Read Replica for Performance
resource "aws_db_instance" "leanvibe_db_replica" {
  identifier = "leanvibe-production-db-replica"
  
  replicate_source_db = aws_db_instance.leanvibe_db.id
  instance_class      = "db.t3.medium"
  
  vpc_security_group_ids = [aws_security_group.db_sg.id]
  
  tags = {
    Name = "LeanVibe Production Database Replica"
    Environment = "production"
  }
}
```

### On-Premises Infrastructure

#### Server Preparation

```bash
#!/bin/bash
# server-setup.sh - Prepare Ubuntu 20.04 server for LeanVibe deployment

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y \
    curl \
    wget \
    git \
    vim \
    htop \
    unzip \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Install Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create docker group and add user
sudo groupadd docker
sudo usermod -aG docker $USER

# Install Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Install Nginx
sudo apt install -y nginx

# Install and configure firewall
sudo ufw enable
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw reload

# Create application user
sudo useradd -m -s /bin/bash leanvibe
sudo usermod -aG docker leanvibe

# Create application directories
sudo mkdir -p /opt/leanvibe/{app,logs,backups,ssl}
sudo chown -R leanvibe:leanvibe /opt/leanvibe

echo "Server preparation completed. Please reboot to ensure all changes take effect."
```

#### Database Server Setup

```bash
#!/bin/bash
# database-setup.sh - Set up PostgreSQL with pgvector

# Install PostgreSQL 15
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt update
sudo apt install -y postgresql-15 postgresql-client-15 postgresql-contrib-15

# Install pgvector extension
sudo apt install -y postgresql-15-pgvector

# Configure PostgreSQL
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'secure_password_here';"

# Create database and user
sudo -u postgres createdb leanvibe_production
sudo -u postgres psql -c "CREATE USER leanvibe_admin WITH PASSWORD 'secure_db_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE leanvibe_production TO leanvibe_admin;"
sudo -u postgres psql -d leanvibe_production -c "CREATE EXTENSION IF NOT EXISTS vector;"
sudo -u postgres psql -d leanvibe_production -c "GRANT CREATE ON SCHEMA public TO leanvibe_admin;"

# Configure PostgreSQL settings
sudo tee /etc/postgresql/15/main/postgresql.conf << EOF
# Connection settings
listen_addresses = '*'
port = 5432
max_connections = 200

# Memory settings
shared_buffers = 2GB
effective_cache_size = 6GB
work_mem = 32MB
maintenance_work_mem = 512MB

# Checkpoint settings
checkpoint_completion_target = 0.9
wal_buffers = 16MB

# Query planner
random_page_cost = 1.1
effective_io_concurrency = 200

# Logging
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
log_temp_files = 0

# Performance
shared_preload_libraries = 'pg_stat_statements'
EOF

# Configure pg_hba.conf for secure access
sudo tee /etc/postgresql/15/main/pg_hba.conf << EOF
# TYPE  DATABASE        USER            ADDRESS                 METHOD
local   all             postgres                                peer
local   all             all                                     peer
host    leanvibe_production    leanvibe_admin     10.0.0.0/16          md5
host    all             all             127.0.0.1/32            md5
host    all             all             ::1/128                 md5
EOF

# Restart PostgreSQL
sudo systemctl restart postgresql
sudo systemctl enable postgresql

echo "PostgreSQL setup completed with pgvector extension."
```

#### Redis Setup

```bash
#!/bin/bash
# redis-setup.sh - Install and configure Redis

# Install Redis
sudo apt update
sudo apt install -y redis-server

# Configure Redis for production
sudo tee /etc/redis/redis.conf << EOF
# Network
bind 127.0.0.1 10.0.0.0/16
port 6379
protected-mode yes

# General
daemonize yes
supervised systemd
pidfile /var/run/redis/redis-server.pid

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log
syslog-enabled yes

# Snapshotting
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /var/lib/redis

# Security
requirepass your_redis_password_here
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command DEBUG ""

# Memory management
maxmemory 4gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# Append only file
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Slow log
slowlog-log-slower-than 10000
slowlog-max-len 128

# Client output buffer limits
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60
EOF

# Set Redis password (replace with secure password)
sudo sed -i 's/your_redis_password_here/SecureRedisPassword123!/' /etc/redis/redis.conf

# Start and enable Redis
sudo systemctl restart redis-server
sudo systemctl enable redis-server

# Test Redis connection
redis-cli -a SecureRedisPassword123! ping

echo "Redis setup completed."
```

## Environment Configuration

### Environment Variables Setup

#### Production Environment File

```bash
# /opt/leanvibe/app/.env.production

# Application Configuration
DEBUG=false
LOG_LEVEL=INFO
ENVIRONMENT=production

# Database Configuration
DATABASE_URL=postgresql+asyncpg://leanvibe_admin:secure_db_password@db-server:5432/leanvibe_production
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10
DATABASE_POOL_TIMEOUT=30
DATABASE_POOL_RECYCLE=3600

# Redis Configuration
REDIS_URL=redis://:SecureRedisPassword123!@redis-server:6379/0
REDIS_POOL_SIZE=20
REDIS_POOL_TIMEOUT=30

# Security Configuration
JWT_SECRET_KEY=your-super-secure-jwt-secret-key-here-min-32-chars
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_DAYS=30

# CORS Settings
CORS_ORIGINS=["https://your-domain.com", "https://app.your-domain.com"]
ALLOWED_HOSTS=["your-domain.com", "app.your-domain.com", "api.your-domain.com"]

# External APIs
ANTHROPIC_API_KEY=your-anthropic-api-key-here
OPENAI_API_KEY=your-openai-api-key-here

# Monitoring and Observability
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
JAEGER_ENABLED=true
SENTRY_DSN=your-sentry-dsn-here

# Email Configuration (for notifications)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=noreply@your-domain.com
SMTP_PASSWORD=your-email-password
SMTP_TLS=true

# File Storage
UPLOAD_PATH=/opt/leanvibe/uploads
MAX_FILE_SIZE=50MB

# Performance Settings
WORKER_PROCESSES=4
WORKER_THREADS=8
WORKER_TIMEOUT=30

# Security Headers
SECURITY_HEADERS_ENABLED=true
HSTS_MAX_AGE=31536000
CONTENT_SECURITY_POLICY="default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_BURST=200

# Session Configuration
SESSION_TIMEOUT=3600
SESSION_CLEANUP_INTERVAL=300

# Backup Configuration
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM
BACKUP_RETENTION_DAYS=30
S3_BACKUP_BUCKET=leanvibe-backups
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-west-2
```

#### Environment Variables Validation Script

```bash
#!/bin/bash
# validate-env.sh - Validate environment configuration

echo "Validating production environment configuration..."

# Check required environment variables
required_vars=(
    "DATABASE_URL"
    "REDIS_URL" 
    "JWT_SECRET_KEY"
    "ANTHROPIC_API_KEY"
)

missing_vars=()

for var in "${required_vars[@]}"; do
    if [[ -z "${!var}" ]]; then
        missing_vars+=("$var")
    fi
done

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    echo "Error: Missing required environment variables:"
    printf '%s\n' "${missing_vars[@]}"
    exit 1
fi

# Validate JWT secret key length
if [[ ${#JWT_SECRET_KEY} -lt 32 ]]; then
    echo "Error: JWT_SECRET_KEY must be at least 32 characters long"
    exit 1
fi

# Test database connection
echo "Testing database connection..."
python3 -c "
import asyncio
import asyncpg
import os

async def test_db():
    try:
        conn = await asyncpg.connect(os.getenv('DATABASE_URL'))
        await conn.fetchval('SELECT 1')
        await conn.close()
        print('✓ Database connection successful')
    except Exception as e:
        print(f'✗ Database connection failed: {e}')
        exit(1)

asyncio.run(test_db())
"

# Test Redis connection
echo "Testing Redis connection..."
python3 -c "
import redis
import os
from urllib.parse import urlparse

url = urlparse(os.getenv('REDIS_URL'))
r = redis.Redis(
    host=url.hostname,
    port=url.port,
    password=url.password,
    decode_responses=True
)

try:
    r.ping()
    print('✓ Redis connection successful')
except Exception as e:
    print(f'✗ Redis connection failed: {e}')
    exit(1)
"

echo "Environment validation completed successfully!"
```

### SSL Certificate Setup

#### Let's Encrypt Setup (Recommended for Production)

```bash
#!/bin/bash
# ssl-setup.sh - Set up SSL certificates with Let's Encrypt

# Install Certbot
sudo apt update
sudo apt install -y certbot python3-certbot-nginx

# Generate certificates for your domains
sudo certbot --nginx -d your-domain.com -d app.your-domain.com -d api.your-domain.com

# Set up automatic renewal
sudo crontab -e
# Add this line to renew certificates twice daily:
# 0 12 * * * /usr/bin/certbot renew --quiet

# Test renewal process
sudo certbot renew --dry-run

echo "SSL certificates configured successfully!"
```

#### Manual SSL Certificate Installation

```bash
#!/bin/bash
# manual-ssl-setup.sh - Install manually provided certificates

# Create SSL directory
sudo mkdir -p /opt/leanvibe/ssl

# Copy certificate files (replace with your actual files)
sudo cp your-domain.com.crt /opt/leanvibe/ssl/
sudo cp your-domain.com.key /opt/leanvibe/ssl/
sudo cp ca-bundle.crt /opt/leanvibe/ssl/

# Set proper permissions
sudo chmod 600 /opt/leanvibe/ssl/*.key
sudo chmod 644 /opt/leanvibe/ssl/*.crt
sudo chown -R leanvibe:leanvibe /opt/leanvibe/ssl

# Verify certificate
openssl x509 -in /opt/leanvibe/ssl/your-domain.com.crt -text -noout

echo "Manual SSL certificates installed successfully!"
```

## Database Setup

### Database Initialization

#### Run Database Migrations

```bash
#!/bin/bash
# database-init.sh - Initialize database with migrations

cd /opt/leanvibe/app

# Load environment variables
source .env.production

# Install Python dependencies
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Create initial admin user
echo "Creating initial admin user..."
python3 -c "
import asyncio
from app.core.database import get_session
from app.models.user import User
from app.core.security import get_password_hash

async def create_admin():
    async for db in get_session():
        admin_user = User(
            email='admin@your-domain.com',
            username='admin',
            hashed_password=get_password_hash('SecureAdminPassword123!'),
            is_active=True,
            is_superuser=True
        )
        db.add(admin_user)
        await db.commit()
        print('Admin user created successfully')
        break

asyncio.run(create_admin())
"

echo "Database initialization completed!"
```

#### Database Performance Tuning

```sql
-- database-tuning.sql - Optimize database for production

-- Create indexes for better performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_capabilities ON agents USING GIN(capabilities);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_priority ON tasks(priority);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_workflows_status ON workflows(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_contexts_agent_id ON contexts(agent_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_contexts_embedding ON contexts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Update table statistics
ANALYZE agents;
ANALYZE tasks;
ANALYZE workflows;
ANALYZE contexts;

-- Configure auto-vacuum for high-activity tables
ALTER TABLE tasks SET (autovacuum_vacuum_scale_factor = 0.1);
ALTER TABLE contexts SET (autovacuum_vacuum_scale_factor = 0.05);

-- Create materialized views for complex queries
CREATE MATERIALIZED VIEW agent_performance_summary AS
SELECT 
    agent_id,
    COUNT(*) as total_tasks,
    AVG(actual_effort) as avg_completion_time,
    COUNT(*) FILTER (WHERE status = 'completed') as completed_tasks,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_tasks,
    (COUNT(*) FILTER (WHERE status = 'completed')::float / COUNT(*)::float) as success_rate
FROM tasks 
WHERE created_at > NOW() - INTERVAL '30 days'
GROUP BY agent_id;

CREATE UNIQUE INDEX ON agent_performance_summary (agent_id);

-- Set up automatic refresh of materialized views
-- This should be run via cron job: */15 * * * * psql -d leanvibe_production -c "REFRESH MATERIALIZED VIEW CONCURRENTLY agent_performance_summary;"
```

#### Database Backup Setup

```bash
#!/bin/bash
# backup-setup.sh - Set up automated database backups

# Create backup directory
sudo mkdir -p /opt/leanvibe/backups/database
sudo chown leanvibe:leanvibe /opt/leanvibe/backups/database

# Create backup script
sudo tee /opt/leanvibe/scripts/db-backup.sh << 'EOF'
#!/bin/bash
# Database backup script

BACKUP_DIR="/opt/leanvibe/backups/database"
DB_NAME="leanvibe_production"
DB_USER="leanvibe_admin"
DB_HOST="localhost"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="$BACKUP_DIR/leanvibe_backup_$TIMESTAMP.sql"

# Create backup
PGPASSWORD="$DATABASE_PASSWORD" pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > $BACKUP_FILE

# Compress backup
gzip $BACKUP_FILE

# Upload to S3 (if configured)
if [[ -n "$S3_BACKUP_BUCKET" ]]; then
    aws s3 cp "$BACKUP_FILE.gz" "s3://$S3_BACKUP_BUCKET/database/"
fi

# Clean up old backups (keep 30 days)
find $BACKUP_DIR -name "leanvibe_backup_*.sql.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_FILE.gz"
EOF

sudo chmod +x /opt/leanvibe/scripts/db-backup.sh

# Set up cron job for daily backups
sudo crontab -u leanvibe -e
# Add this line:
# 0 2 * * * /opt/leanvibe/scripts/db-backup.sh >> /opt/leanvibe/logs/backup.log 2>&1

echo "Database backup system configured!"
```

## Application Deployment

### Docker-based Deployment

#### Production Docker Compose Configuration

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile.prod
    image: leanvibe/agent-hive:latest
    container_name: leanvibe-app
    restart: unless-stopped
    env_file:
      - .env.production
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads
      - ./ssl:/app/ssl:ro
    depends_on:
      - postgres
      - redis
    networks:
      - leanvibe-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  postgres:
    image: pgvector/pgvector:pg15
    container_name: leanvibe-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: leanvibe_production
      POSTGRES_USER: leanvibe_admin
      POSTGRES_PASSWORD: ${DATABASE_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backups:/backups
    ports:
      - "5432:5432"
    networks:
      - leanvibe-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U leanvibe_admin -d leanvibe_production"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: leanvibe-redis
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD} --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - leanvibe-network
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD}", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: leanvibe-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/sites-available:/etc/nginx/sites-available:ro
      - ./ssl:/etc/nginx/ssl:ro
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - app
    networks:
      - leanvibe-network

  prometheus:
    image: prom/prometheus:latest
    container_name: leanvibe-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    networks:
      - leanvibe-network

  grafana:
    image: grafana/grafana:latest
    container_name: leanvibe-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    networks:
      - leanvibe-network

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  leanvibe-network:
    driver: bridge
```

#### Production Dockerfile

```dockerfile
# Dockerfile.prod
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY migrations/ ./migrations/
COPY alembic.ini .

# Create non-root user
RUN groupadd -r leanvibe && useradd -r -g leanvibe leanvibe
RUN chown -R leanvibe:leanvibe /app
USER leanvibe

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

#### Nginx Configuration

```nginx
# nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream leanvibe_app {
        server app:8000;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/s;
    
    server {
        listen 80;
        server_name your-domain.com app.your-domain.com api.your-domain.com;
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl http2;
        server_name your-domain.com app.your-domain.com api.your-domain.com;
        
        # SSL Configuration
        ssl_certificate /etc/nginx/ssl/your-domain.com.crt;
        ssl_certificate_key /etc/nginx/ssl/your-domain.com.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-SHA384;
        ssl_prefer_server_ciphers on;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;
        
        # Security Headers
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        add_header X-Frame-Options DENY always;
        add_header X-Content-Type-Options nosniff always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header Referrer-Policy "strict-origin-when-cross-origin" always;
        
        # Gzip compression
        gzip on;
        gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
        
        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://leanvibe_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Authentication endpoints (stricter rate limiting)
        location /api/v1/auth/ {
            limit_req zone=auth burst=10 nodelay;
            proxy_pass http://leanvibe_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # WebSocket connections
        location /ws/ {
            proxy_pass http://leanvibe_app;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 86400;
        }
        
        # Static files
        location /static/ {
            alias /var/www/static/;
            expires 30d;
            add_header Cache-Control "public, immutable";
        }
        
        # Frontend application
        location / {
            proxy_pass http://leanvibe_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

### Deployment Script

```bash
#!/bin/bash
# deploy.sh - Production deployment script

set -e

echo "Starting LeanVibe Agent Hive production deployment..."

# Configuration
DEPLOY_DIR="/opt/leanvibe"
APP_DIR="$DEPLOY_DIR/app"
BACKUP_DIR="$DEPLOY_DIR/backups"
LOG_FILE="$DEPLOY_DIR/logs/deploy.log"

# Create log entry
echo "$(date): Starting deployment" >> $LOG_FILE

# Pre-deployment checks
echo "Running pre-deployment checks..."

# Check if services are running
if ! systemctl is-active --quiet docker; then
    echo "Error: Docker is not running"
    exit 1
fi

# Check disk space
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    echo "Warning: Disk usage is at ${DISK_USAGE}%"
fi

# Check memory
FREE_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7*100/$2}')
if [ $FREE_MEM -lt 20 ]; then
    echo "Warning: Available memory is at ${FREE_MEM}%"
fi

# Backup current version
echo "Creating backup of current version..."
if [ -d "$APP_DIR" ]; then
    BACKUP_NAME="backup_$(date +%Y%m%d_%H%M%S)"
    cp -r "$APP_DIR" "$BACKUP_DIR/$BACKUP_NAME"
    echo "Backup created: $BACKUP_DIR/$BACKUP_NAME"
fi

# Pull latest code
echo "Pulling latest code..."
cd $APP_DIR
git fetch --all
git checkout main
git pull origin main

# Update dependencies
echo "Updating dependencies..."
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Run database migrations
echo "Running database migrations..."
alembic upgrade head

# Build frontend
echo "Building frontend applications..."
cd frontend
npm ci --production
npm run build

cd ../mobile-pwa
npm ci --production
npm run build

# Run tests
echo "Running tests..."
cd $APP_DIR
pytest tests/ -v --tb=short

# Build and deploy with Docker Compose
echo "Building and deploying containers..."
docker-compose -f docker-compose.prod.yml build --no-cache
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Health checks
echo "Running health checks..."
HEALTH_CHECK_URL="http://localhost:8000/health"
for i in {1..10}; do
    if curl -f -s $HEALTH_CHECK_URL > /dev/null; then
        echo "Health check passed"
        break
    else
        echo "Health check attempt $i failed, retrying..."
        sleep 10
    fi
    
    if [ $i -eq 10 ]; then
        echo "Health checks failed, rolling back..."
        docker-compose -f docker-compose.prod.yml down
        # Restore from backup if needed
        exit 1
    fi
done

# Update reverse proxy configuration
echo "Updating Nginx configuration..."
sudo systemctl reload nginx

# Clean up old Docker images
echo "Cleaning up old Docker images..."
docker image prune -f

echo "Deployment completed successfully!"
echo "$(date): Deployment completed successfully" >> $LOG_FILE

# Send notification (optional)
if [ -n "$SLACK_WEBHOOK_URL" ]; then
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"LeanVibe Agent Hive deployment completed successfully"}' \
        $SLACK_WEBHOOK_URL
fi
```

## Post-Deployment Verification

### Automated Verification Script

```bash
#!/bin/bash
# verify-deployment.sh - Comprehensive post-deployment verification

echo "Starting post-deployment verification..."

# Service health checks
echo "Checking service health..."

# Check application health
APP_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ $APP_HEALTH -eq 200 ]; then
    echo "✓ Application service is healthy"
else
    echo "✗ Application service health check failed (HTTP $APP_HEALTH)"
    exit 1
fi

# Check database connection
DB_CHECK=$(curl -s http://localhost:8000/health | jq -r '.dependencies.database.status')
if [ "$DB_CHECK" = "healthy" ]; then
    echo "✓ Database connection is healthy"
else
    echo "✗ Database connection failed"
    exit 1
fi

# Check Redis connection
REDIS_CHECK=$(curl -s http://localhost:8000/health | jq -r '.dependencies.redis.status')
if [ "$REDIS_CHECK" = "healthy" ]; then
    echo "✓ Redis connection is healthy"
else
    echo "✗ Redis connection failed"
    exit 1
fi

# API endpoint tests
echo "Testing API endpoints..."

# Test authentication endpoint
AUTH_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/auth/test \
    -H "Content-Type: application/json" \
    -d '{"test": true}' \
    -w "%{http_code}")

if [[ $AUTH_RESPONSE == *"200"* ]]; then
    echo "✓ Authentication endpoint is working"
else
    echo "✗ Authentication endpoint failed"
fi

# Test agents endpoint
AGENTS_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    http://localhost:8000/api/v1/agents \
    -H "Authorization: Bearer test-token")

if [ $AGENTS_RESPONSE -eq 200 ] || [ $AGENTS_RESPONSE -eq 401 ]; then
    echo "✓ Agents endpoint is accessible"
else
    echo "✗ Agents endpoint failed (HTTP $AGENTS_RESPONSE)"
fi

# WebSocket connectivity test
echo "Testing WebSocket connectivity..."
timeout 10s wscat -c ws://localhost:8000/ws/observability && echo "✓ WebSocket connection successful" || echo "! WebSocket test skipped (wscat not available)"

# Frontend verification
echo "Checking frontend applications..."

# Check if frontend files are served
FRONTEND_CHECK=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/)
if [ $FRONTEND_CHECK -eq 200 ]; then
    echo "✓ Frontend application is accessible"
else
    echo "✗ Frontend application failed (HTTP $FRONTEND_CHECK)"
fi

# SSL certificate verification (if HTTPS enabled)
if [ -f "/etc/nginx/ssl/your-domain.com.crt" ]; then
    echo "Checking SSL certificate..."
    SSL_EXPIRY=$(openssl x509 -enddate -noout -in /etc/nginx/ssl/your-domain.com.crt | cut -d= -f2)
    SSL_DAYS=$(( ($(date -d "$SSL_EXPIRY" +%s) - $(date +%s)) / 86400 ))
    
    if [ $SSL_DAYS -gt 30 ]; then
        echo "✓ SSL certificate is valid (expires in $SSL_DAYS days)"
    else
        echo "! SSL certificate expires soon ($SSL_DAYS days)"
    fi
fi

# Performance verification
echo "Running performance verification..."

# Check response times
RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}\n' http://localhost:8000/health)
if (( $(echo "$RESPONSE_TIME < 1.0" | bc -l) )); then
    echo "✓ API response time is acceptable ($RESPONSE_TIME seconds)"
else
    echo "! API response time is slow ($RESPONSE_TIME seconds)"
fi

# Check system resources
echo "Checking system resources..."

# Memory usage
MEM_USAGE=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100.0)}')
if [ $MEM_USAGE -lt 80 ]; then
    echo "✓ Memory usage is acceptable ($MEM_USAGE%)"
else
    echo "! High memory usage ($MEM_USAGE%)"
fi

# Disk usage
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -lt 80 ]; then
    echo "✓ Disk usage is acceptable ($DISK_USAGE%)"
else
    echo "! High disk usage ($DISK_USAGE%)"
fi

# Container status
echo "Checking container status..."
CONTAINERS=$(docker-compose -f docker-compose.prod.yml ps --services --filter "status=running" | wc -l)
EXPECTED_CONTAINERS=6  # app, postgres, redis, nginx, prometheus, grafana

if [ $CONTAINERS -eq $EXPECTED_CONTAINERS ]; then
    echo "✓ All containers are running ($CONTAINERS/$EXPECTED_CONTAINERS)"
else
    echo "! Some containers may not be running ($CONTAINERS/$EXPECTED_CONTAINERS)"
    docker-compose -f docker-compose.prod.yml ps
fi

# Log verification
echo "Checking application logs..."
ERROR_COUNT=$(docker logs leanvibe-app --since=5m 2>&1 | grep -i error | wc -l)
if [ $ERROR_COUNT -eq 0 ]; then
    echo "✓ No recent errors in application logs"
else
    echo "! Found $ERROR_COUNT recent errors in application logs"
fi

echo "Post-deployment verification completed!"
```

### Manual Verification Checklist

```markdown
# Manual Verification Checklist

## Functional Testing
- [ ] User registration and login works
- [ ] Agent creation and management functions properly
- [ ] Task creation and assignment works
- [ ] Workflow execution runs successfully
- [ ] Real-time updates via WebSocket work
- [ ] Dashboard displays accurate data
- [ ] API endpoints respond correctly
- [ ] File uploads work (if applicable)
- [ ] Email notifications send properly

## Performance Testing
- [ ] Page load times are under 3 seconds
- [ ] API response times are under 500ms
- [ ] Database queries complete quickly
- [ ] WebSocket connections are stable
- [ ] System handles expected concurrent users
- [ ] Memory usage is within limits
- [ ] CPU usage is reasonable

## Security Testing
- [ ] HTTPS redirects work properly
- [ ] Authentication is required for protected endpoints
- [ ] Authorization rules are enforced
- [ ] Input validation prevents injection attacks
- [ ] Rate limiting is active
- [ ] Security headers are present
- [ ] File upload restrictions work
- [ ] Session management is secure

## Monitoring and Alerting
- [ ] Prometheus metrics are collecting
- [ ] Grafana dashboards display data
- [ ] Alert rules are configured
- [ ] Log aggregation is working
- [ ] Health check endpoints respond
- [ ] Backup systems are running
- [ ] Monitoring notifications work
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"
  - "recording_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'leanvibe-app'
    static_configs:
      - targets: ['app:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx-exporter:9113']
```

### Alert Rules

```yaml
# monitoring/alert_rules.yml
groups:
  - name: leanvibe_alerts
    rules:
      - alert: HighAPIResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High API response time detected"
          description: "95th percentile response time is {{ $value }}s"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: DatabaseConnectionHigh
        expr: postgres_stat_activity_count > 80
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "High database connection count"
          description: "Current connection count: {{ $value }}"

      - alert: RedisMemoryHigh
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.8
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Redis memory usage is high"
          description: "Memory usage: {{ $value | humanizePercentage }}"

      - alert: ContainerDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Container is down"
          description: "{{ $labels.job }} container is not responding"
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "id": null,
    "title": "LeanVibe Agent Hive Overview",
    "tags": ["leanvibe"],
    "timezone": "browser",
    "panels": [
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Active Agents",
        "type": "singlestat",
        "targets": [
          {
            "expr": "leanvibe_active_agents_total"
          }
        ]
      },
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(cpu_usage_seconds_total[5m]) * 100",
            "legendFormat": "CPU Usage %"
          },
          {
            "expr": "memory_usage_bytes / memory_total_bytes * 100",
            "legendFormat": "Memory Usage %"
          }
        ]
      }
    ]
  }
}
```

This comprehensive deployment runbook provides detailed procedures for setting up LeanVibe Agent Hive in production environments. The documentation covers everything from infrastructure setup to monitoring configuration, ensuring a reliable and scalable deployment.