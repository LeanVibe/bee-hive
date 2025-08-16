# Project Index Universal Installer - Complete Integration Guide

## 🚀 Quick Start

The Project Index Universal Installer provides a complete containerized deployment of intelligent project analysis and context optimization for any codebase. Get running in under 5 minutes:

```bash
# 1. Clone the repository
git clone https://github.com/leanvibe/bee-hive.git
cd bee-hive

# 2. Copy environment template and configure
cp .env.template .env
# Edit .env file with your settings (see Configuration section)

# 3. Start all services
./start-project-index.sh start

# 4. Access the services
# - API: http://localhost:8100
# - Dashboard: http://localhost:8101
# - Health: http://localhost:8100/health
```

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Deployment Profiles](#deployment-profiles)
- [Host Project Integration](#host-project-integration)
- [API Reference](#api-reference)
- [WebSocket Integration](#websocket-integration)
- [Monitoring & Observability](#monitoring--observability)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)
- [Security Considerations](#security-considerations)
- [Performance Tuning](#performance-tuning)

## 🔧 Prerequisites

### Required Software

- **Docker Engine**: 20.10+ with Docker Compose V2
- **Available Resources**: 
  - RAM: 2GB minimum, 4GB recommended
  - Disk: 1GB free space minimum
  - CPU: 2 cores recommended
- **Network**: Ports 8100-8101, 5433, 6380, 9090, 3001 available

### Operating System Support

- ✅ Linux (Ubuntu 20.04+, CentOS 8+, RHEL 8+)
- ✅ macOS (10.15+)
- ✅ Windows 10/11 with WSL2

### Verification Commands

```bash
# Check Docker
docker --version
docker compose version
docker info

# Check available ports
netstat -tuln | grep -E ':(8100|8101|5433|6380|9090|3001)'

# Check system resources
free -h
df -h
```

## 📦 Installation

### Method 1: Quick Start Script

```bash
# Download and run the installer
curl -fsSL https://raw.githubusercontent.com/leanvibe/bee-hive/main/quick-install.sh | bash

# Or with custom options
curl -fsSL https://raw.githubusercontent.com/leanvibe/bee-hive/main/quick-install.sh | bash -s -- --profile medium --path /my/project
```

### Method 2: Manual Installation

```bash
# 1. Clone repository
git clone https://github.com/leanvibe/bee-hive.git
cd bee-hive

# 2. Choose deployment profile
cp .env.template .env          # Default medium profile
# OR
cp .env.small .env            # For small projects
cp .env.large .env            # For large projects

# 3. Configure environment
editor .env

# 4. Start services
./start-project-index.sh start
```

### Method 3: Docker Compose Only

```bash
# For advanced users who prefer direct Docker Compose control
docker compose -f docker-compose.universal.yml --profile production up -d
```

## ⚙️ Configuration

### Essential Configuration

Edit your `.env` file with these required settings:

```bash
# Project Configuration - REQUIRED
PROJECT_NAME=my-awesome-project
HOST_PROJECT_PATH=/absolute/path/to/your/project
DATA_DIR=./data

# Security - REQUIRED: Change these passwords!
PROJECT_INDEX_PASSWORD=your_secure_database_password
REDIS_PASSWORD=your_secure_redis_password

# Network Configuration - Adjust if ports conflict
PROJECT_INDEX_PORT=8100
DASHBOARD_PORT=8101
POSTGRES_PORT=5433
REDIS_PORT=6380
```

### Optional Advanced Configuration

```bash
# Analysis Configuration
ANALYSIS_MODE=intelligent          # simple|smart|intelligent|comprehensive
MONITOR_PATTERNS=**/*.py,**/*.js,**/*.ts,**/*.go,**/*.rs
IGNORE_PATTERNS=**/node_modules/**,**/.git/**,**/__pycache__/**

# Performance Tuning
WORKER_CONCURRENCY=4
MAX_FILE_SIZE_MB=50
ANALYSIS_TIMEOUT_SECONDS=600

# Feature Toggles
ENABLE_ML_ANALYSIS=true
ENABLE_SEMANTIC_SEARCH=true
ENABLE_REAL_TIME_MONITORING=true
ENABLE_METRICS=true
```

## 🎛️ Deployment Profiles

Choose the right profile for your project size and team:

### Small Profile (< 1,000 files, 1 developer)
```bash
./start-project-index.sh --profile small start
```
- **Resources**: 1GB RAM, 0.5 CPU cores
- **Features**: Essential analysis only
- **Use case**: Personal projects, prototypes

### Medium Profile (1k-10k files, 2-5 developers) [Default]
```bash
./start-project-index.sh --profile medium start
```
- **Resources**: 2GB RAM, 1 CPU core
- **Features**: Full analysis with caching
- **Use case**: Most development projects

### Large Profile (10k+ files, 5+ developers)
```bash
./start-project-index.sh --profile large start
```
- **Resources**: 4GB RAM, 2 CPU cores
- **Features**: All features + advanced analytics
- **Use case**: Enterprise projects, complex codebases

### Custom Profile
```bash
# Copy and modify existing profile
cp .env.medium .env.custom
# Edit resource limits and features
./start-project-index.sh --env .env.custom start
```

## 🔌 Host Project Integration

### REST API Integration

#### 1. Basic Project Analysis

```python
import requests

# Start project analysis
response = requests.post('http://localhost:8100/api/v1/projects/analyze', json={
    'project_path': '/path/to/project',
    'analysis_mode': 'intelligent',
    'include_dependencies': True
})

analysis_id = response.json()['analysis_id']
print(f"Analysis started: {analysis_id}")
```

#### 2. Get Analysis Results

```python
# Poll for results
import time

while True:
    status = requests.get(f'http://localhost:8100/api/v1/projects/{analysis_id}/status')
    if status.json()['status'] == 'completed':
        results = requests.get(f'http://localhost:8100/api/v1/projects/{analysis_id}/results')
        break
    time.sleep(5)

print(f"Analysis complete: {results.json()}")
```

#### 3. Real-time File Monitoring

```python
# Enable real-time monitoring for your project
requests.post('http://localhost:8100/api/v1/monitoring/enable', json={
    'project_path': '/path/to/project',
    'patterns': ['**/*.py', '**/*.js'],
    'webhook_url': 'http://your-app/webhooks/file-changed'
})
```

### WebSocket Integration for Real-time Updates

```javascript
// Connect to real-time updates
const ws = new WebSocket('ws://localhost:8100/ws');

ws.onopen = function() {
    // Subscribe to project events
    ws.send(JSON.stringify({
        type: 'subscribe',
        channels: ['project_updates', 'analysis_progress']
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'file_changed':
            console.log('File changed:', data.file_path);
            break;
        case 'analysis_complete':
            console.log('Analysis complete:', data.analysis_id);
            break;
        case 'dependency_updated':
            console.log('Dependencies changed:', data.changes);
            break;
    }
};
```

### Integration Libraries

#### Python Client

```bash
pip install project-index-client
```

```python
from project_index_client import ProjectIndexClient

client = ProjectIndexClient('http://localhost:8100')

# Analyze project
analysis = client.analyze_project('/path/to/project')

# Get context for specific files
context = client.get_context(['src/main.py', 'src/utils.py'])

# Search for similar code
similar = client.find_similar_code('def process_data')
```

#### JavaScript/Node.js Client

```bash
npm install project-index-client
```

```javascript
const { ProjectIndexClient } = require('project-index-client');

const client = new ProjectIndexClient('http://localhost:8100');

// Analyze project
const analysis = await client.analyzeProject('/path/to/project');

// Real-time monitoring
client.on('fileChanged', (file) => {
    console.log(`File changed: ${file.path}`);
});

client.enableMonitoring('/path/to/project');
```

## 📊 API Reference

### Core Endpoints

#### Project Analysis
```
POST /api/v1/projects/analyze
GET  /api/v1/projects/{id}/status
GET  /api/v1/projects/{id}/results
GET  /api/v1/projects/{id}/summary
```

#### File Operations
```
GET  /api/v1/files/analyze/{file_path}
POST /api/v1/files/context
GET  /api/v1/files/dependencies/{file_path}
POST /api/v1/files/similar
```

#### Monitoring
```
POST /api/v1/monitoring/enable
GET  /api/v1/monitoring/status
POST /api/v1/monitoring/disable
GET  /api/v1/monitoring/events
```

#### Search & Context
```
POST /api/v1/search/semantic
POST /api/v1/search/code
GET  /api/v1/context/optimize
POST /api/v1/context/assemble
```

### WebSocket Events

#### Subscribe to Events
```json
{
    "type": "subscribe",
    "channels": ["project_updates", "analysis_progress", "file_changes"]
}
```

#### Event Types
- `file_changed`: Real-time file modification events
- `analysis_progress`: Analysis progress updates
- `analysis_complete`: Analysis completion notification
- `dependency_changed`: Dependency graph updates
- `context_optimized`: Context optimization results

## 📈 Monitoring & Observability

### Health Monitoring

```bash
# Basic health check
curl http://localhost:8100/health

# Comprehensive health check
./start-project-index.sh health

# Continuous monitoring
python3 docker/health-check.py --output json
```

### Prometheus Metrics

Access metrics at `http://localhost:9090` when metrics are enabled:

```yaml
# Key metrics to monitor
- project_index_analysis_duration_seconds
- project_index_files_processed_total
- project_index_cache_hit_ratio
- project_index_memory_usage_bytes
- project_index_api_requests_total
```

### Grafana Dashboards

Pre-configured dashboards available at `http://localhost:3001`:

1. **Project Index Overview**: System health and performance
2. **Analysis Performance**: Analysis timing and throughput
3. **Resource Usage**: Memory, CPU, and storage metrics
4. **File Monitoring**: Real-time file change events

Default login: `admin` / `admin`

### Logging

```bash
# View all logs
./start-project-index.sh logs

# View specific service logs
docker compose -f docker-compose.universal.yml logs -f project-index

# Export logs for analysis
docker compose -f docker-compose.universal.yml logs --no-color > project-index.log
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Port Conflicts
```bash
# Check what's using the port
lsof -i :8100

# Change ports in .env file
PROJECT_INDEX_PORT=8200
DASHBOARD_PORT=8201
```

#### 2. Database Connection Issues
```bash
# Check PostgreSQL container
docker compose -f docker-compose.universal.yml logs postgres

# Test database connection
docker compose -f docker-compose.universal.yml exec postgres psql -U project_user -d project_index -c "SELECT 1;"
```

#### 3. Out of Memory
```bash
# Check container memory usage
docker stats

# Reduce resource limits in .env
PROJECT_INDEX_MEMORY_LIMIT=1G
POSTGRES_MEMORY_LIMIT=512M
```

#### 4. Analysis Failing
```bash
# Check project path accessibility
ls -la $HOST_PROJECT_PATH

# Check file permissions
docker compose -f docker-compose.universal.yml exec project-index ls -la /workspace

# Review analysis logs
docker compose -f docker-compose.universal.yml logs project-index | grep -i error
```

### Debug Mode

```bash
# Start in debug mode
DEBUG=true ./start-project-index.sh start

# Enable verbose logging
LOG_LEVEL=DEBUG ./start-project-index.sh start

# Run individual health checks
python3 docker/health-check.py --timeout 60 --output json
```

### Getting Help

1. **Check Logs**: Always start with service logs
2. **Health Check**: Run comprehensive health checks
3. **Resource Monitor**: Check system resources
4. **GitHub Issues**: Search existing issues or create new ones
5. **Community**: Join our Discord for real-time help

## 🔒 Security Considerations

### Production Deployment

```bash
# 1. Change default passwords
PROJECT_INDEX_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)

# 2. Enable authentication
API_KEY_REQUIRED=true
DASHBOARD_AUTH=true

# 3. Configure network security
CORS_ORIGINS=https://your-domain.com
ALLOWED_HOSTS=your-domain.com,localhost

# 4. Use HTTPS (external reverse proxy recommended)
```

### Network Security

```yaml
# docker-compose.override.yml for production
services:
  project-index:
    networks:
      - internal
    ports: []  # Remove external port exposure
  
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
```

### Data Protection

- **Encryption**: All data at rest is encrypted in PostgreSQL
- **Access Control**: Role-based access for API endpoints
- **Audit Logging**: All operations are logged
- **Backup**: Regular automated backups of analysis data

## ⚡ Performance Tuning

### Resource Optimization

```bash
# For CPU-intensive workloads
WORKER_CONCURRENCY=8
ANALYSIS_TIMEOUT_SECONDS=1200

# For memory-constrained environments
CACHE_TTL_HOURS=6
CONTEXT_CACHE_SIZE=500

# For high-throughput scenarios
POSTGRES_MAX_CONNECTIONS=200
REDIS_MAX_CONNECTIONS=100
```

### Caching Strategy

```bash
# Enable aggressive caching
ENABLE_SMART_CACHING=true
CACHE_COMPRESSION=true
CACHE_TTL_HOURS=48

# Pre-warm caches
curl -X POST http://localhost:8100/api/v1/cache/warm
```

### Database Tuning

```bash
# High-performance PostgreSQL settings
POSTGRES_SHARED_BUFFERS=512MB
POSTGRES_CACHE_SIZE=2GB
POSTGRES_MAINTENANCE_MEM=128MB
```

### Monitoring Performance

```bash
# Enable detailed metrics
ENABLE_PERFORMANCE_MONITORING=true
METRICS_COLLECTION_INTERVAL=30

# Performance benchmarking
python3 scripts/benchmark_performance.py --project-path $HOST_PROJECT_PATH
```

## 🔄 Updates and Maintenance

### Updating Project Index

```bash
# Pull latest changes
git pull origin main

# Rebuild containers
docker compose -f docker-compose.universal.yml build --no-cache

# Restart with new version
./start-project-index.sh restart
```

### Database Migrations

```bash
# Run database migrations
docker compose -f docker-compose.universal.yml exec project-index alembic upgrade head

# Backup before major updates
docker compose -f docker-compose.universal.yml exec postgres pg_dump -U project_user project_index > backup.sql
```

### Data Cleanup

```bash
# Clean old analysis data
curl -X POST http://localhost:8100/api/v1/maintenance/cleanup

# Vacuum database
docker compose -f docker-compose.universal.yml exec postgres psql -U project_user -d project_index -c "VACUUM ANALYZE;"
```

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/leanvibe/bee-hive.git
cd bee-hive

# Start in development mode
BUILD_TARGET=development ./start-project-index.sh start

# Run tests
docker compose -f docker-compose.universal.yml exec project-index pytest
```

## 📞 Support

- 📖 **Documentation**: [docs.leanvibe.com](https://docs.leanvibe.com)
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/leanvibe/bee-hive/issues)
- 💬 **Community**: [Discord Server](https://discord.gg/leanvibe)
- 📧 **Enterprise**: [enterprise@leanvibe.com](mailto:enterprise@leanvibe.com)

## 📄 License

Project Index Universal Installer is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

**Made with ❤️ by the LeanVibe team**