# Universal Project Index Installer - Usage Guide

## Quick Start (TL;DR)

```bash
# One-command installation
curl -fsSL https://install.leanvibe.com/project-index.sh | bash

# Access your Project Index
open http://localhost:8100  # API
open http://localhost:8101  # Dashboard
```

## Table of Contents

1. [Installation Options](#installation-options)
2. [Project Size Detection](#project-size-detection)
3. [Language-Specific Optimizations](#language-specific-optimizations)
4. [Deployment Modes](#deployment-modes)
5. [Client Library Integration](#client-library-integration)
6. [API Usage Examples](#api-usage-examples)
7. [Real-Time Monitoring](#real-time-monitoring)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Configuration](#advanced-configuration)
10. [Performance Tuning](#performance-tuning)

## Installation Options

### Basic Installation

```bash
# Install with auto-detected settings
curl -fsSL https://install.leanvibe.com/project-index.sh | bash
```

### Custom Installation

```bash
# Install with specific options
curl -fsSL https://install.leanvibe.com/project-index.sh | bash -s -- \
  --path /path/to/my/project \
  --mode production \
  --enable-monitoring \
  --install-dir /opt/my-project-index
```

### Installation Options Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--path PATH` | Project path to analyze | Current directory |
| `--mode MODE` | Deployment mode (development/production/ci) | development |
| `--enable-dashboard` | Enable web dashboard | true |
| `--disable-dashboard` | Disable web dashboard | false |
| `--enable-monitoring` | Enable Prometheus/Grafana | false |
| `--install-dir DIR` | Installation directory | /opt/project-index |
| `--data-dir DIR` | Data storage directory | /var/lib/project-index |
| `--force` | Force reinstall | false |
| `--quiet` | Suppress output | false |
| `--debug` | Enable debug logging | false |

## Project Size Detection

The installer automatically detects your project size and optimizes configuration:

### Size Categories

| Size | File Count | Memory | CPU | Features |
|------|------------|--------|-----|----------|
| **Small** | < 100 | 896MB | 0.4 CPU | Basic analysis, dashboard |
| **Medium** | 100-1000 | 1.75GB | 0.85 CPU | Smart analysis, advanced search |
| **Large** | 1000-5000 | 3.5GB | 1.7 CPU | Full analysis, ML features |
| **Enterprise** | 5000+ | 8GB+ | 3.4+ CPU | Comprehensive analysis, all features |

### Manual Size Override

```bash
# Force specific size configuration
export PROJECT_SIZE="large"
curl -fsSL https://install.leanvibe.com/project-index.sh | bash
```

## Language-Specific Optimizations

### Supported Languages

The installer automatically detects your primary language and optimizes accordingly:

#### Python Projects
```bash
# Detected via: requirements.txt, pyproject.toml, setup.py
# Optimizations:
# - AST parsing for Python syntax
# - Import dependency analysis
# - Virtual environment exclusions
# - pytest and coverage integration
```

#### Node.js Projects
```bash
# Detected via: package.json, yarn.lock
# Optimizations:
# - ES6/TypeScript AST parsing
# - npm/yarn dependency analysis
# - node_modules exclusions
# - Framework-specific patterns (React, Vue, Angular)
```

#### Go Projects
```bash
# Detected via: go.mod, go.sum
# Optimizations:
# - Go module dependency analysis
# - Package relationship mapping
# - vendor directory exclusions
```

#### Multi-Language Projects
```bash
# Fallback for complex projects
# Features:
# - Universal file pattern detection
# - Cross-language dependency analysis
# - Intelligent language boundary detection
```

## Deployment Modes

### Development Mode (Default)

```bash
curl -fsSL https://install.leanvibe.com/project-index.sh | bash -s -- --mode development
```

**Features:**
- Debug logging enabled
- Hot reload for configuration changes
- Detailed error messages
- Performance profiling
- No authentication required
- CORS enabled for local development

**Resource Usage:** Moderate
**Security:** Relaxed for development ease

### Production Mode

```bash
curl -fsSL https://install.leanvibe.com/project-index.sh | bash -s -- --mode production
```

**Features:**
- Optimized logging (INFO level)
- Security hardening enabled
- Authentication required
- Rate limiting enabled
- SSL/TLS support
- Monitoring and alerting

**Resource Usage:** Optimized
**Security:** Hardened for production

### CI/CD Mode

```bash
curl -fsSL https://install.leanvibe.com/project-index.sh | bash -s -- --mode ci
```

**Features:**
- Fast startup optimizations
- Minimal logging (WARNING level)
- No persistent state
- Optimized for batch analysis
- Lightweight resource usage

**Resource Usage:** Minimal
**Security:** Basic (suitable for CI environments)

## Client Library Integration

### JavaScript/Node.js Integration

```javascript
// Install client library
npm install @leanvibe/project-index-client

// Basic usage
import { ProjectIndexClient } from '@leanvibe/project-index-client';

const client = new ProjectIndexClient({
    baseUrl: 'http://localhost:8100',
    debug: true
});

// Get project information
const project = await client.getProject();
console.log(`Project: ${project.name}, Files: ${project.file_count}`);

// Trigger analysis
const analysis = await client.analyzeProject({
    type: 'smart',
    includeDependencies: true
});

// Real-time monitoring
client.connectWebSocket({ events: ['file_change', 'analysis_progress'] });
client.on('file_change', (event) => {
    console.log(`File changed: ${event.file_path}`);
});
```

### Python Integration

```python
# Install client library
pip install project-index-client

# Basic usage
import asyncio
from project_index_client import ProjectIndexClient

async def main():
    async with ProjectIndexClient(debug=True) as client:
        # Get project info
        project = await client.get_project()
        print(f"Project: {project['name']}")
        
        # Trigger analysis
        analysis = await client.analyze_project({
            'type': 'smart',
            'include_dependencies': True
        })
        
        # Get results
        results = await client.get_analysis_results(analysis['analysis_id'])
        print(f"Analyzed {results['files_analyzed']} files")

asyncio.run(main())
```

### Quick Analysis Functions

```python
# Python quick analysis
from project_index_client import analyze_project

results = await analyze_project('/path/to/project')
print(f"Dependencies: {results['dependencies_found']}")
```

```javascript
// JavaScript quick analysis
import { analyzeProject } from '@leanvibe/project-index-client';

const results = await analyzeProject('/path/to/project');
console.log(`Dependencies: ${results.dependencies_found}`);
```

## API Usage Examples

### REST API Examples

#### Get Project Information
```bash
curl http://localhost:8100/api/projects
```

#### Trigger Analysis
```bash
curl -X POST http://localhost:8100/api/projects/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_type": "smart",
    "include_dependencies": true,
    "include_complexity": true
  }'
```

#### Get File Analysis
```bash
curl "http://localhost:8100/api/files/analyze?path=src/main.py"
```

#### Get Dependency Graph
```bash
curl "http://localhost:8100/api/dependencies/graph?format=json&include_external=true"
```

#### Search Files
```bash
curl -X POST http://localhost:8100/api/files/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "function calculateTotal",
    "language": "javascript",
    "include_content": true,
    "limit": 10
  }'
```

### cURL Examples with jq

```bash
# Get project stats with pretty output
curl -s http://localhost:8100/api/projects/stats | jq '.'

# Get top 10 most complex files
curl -s http://localhost:8100/api/analysis/latest | \
  jq '.files[] | select(.complexity.cyclomatic > 10) | {path: .file_path, complexity: .complexity.cyclomatic}' | \
  head -20

# Get dependency cycles
curl -s http://localhost:8100/api/dependencies/cycles | \
  jq '.cycles[] | {files: [.nodes[].file_path], cycle_length: (.nodes | length)}'
```

## Real-Time Monitoring

### WebSocket Connection

```javascript
// Connect to real-time updates
const ws = new WebSocket('ws://localhost:8100/api/ws');

ws.onopen = () => {
    // Subscribe to specific events
    ws.send(JSON.stringify({
        action: 'subscribe',
        event_types: ['file_change', 'analysis_progress', 'dependency_changed'],
        filters: {
            file_extensions: ['.py', '.js', '.ts']
        }
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'file_change':
            console.log(`File ${data.change_type}: ${data.file_path}`);
            break;
        case 'analysis_progress':
            console.log(`Analysis: ${data.progress_percentage}% complete`);
            break;
        case 'dependency_changed':
            console.log(`Dependency updated: ${data.dependency_details.source_file} -> ${data.dependency_details.target_file}`);
            break;
    }
};
```

### Event Types

| Event Type | Description | Frequency |
|------------|-------------|-----------|
| `file_change` | File was added, modified, or deleted | Real-time |
| `analysis_progress` | Analysis progress updates | Every 1-5 seconds |
| `dependency_changed` | Dependency relationships changed | On detection |
| `analysis_complete` | Full analysis completed | Per analysis |
| `error_occurred` | System errors or warnings | As needed |
| `cache_invalidated` | Cache entries were invalidated | On cache updates |

## Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check if ports are in use
netstat -tulpn | grep -E "(8100|8101|5433|6380)"

# Use alternative ports
curl -fsSL https://install.leanvibe.com/project-index.sh | bash -s -- \
  --env PROJECT_INDEX_PORT=8200 \
  --env DASHBOARD_PORT=8201
```

#### Memory Issues
```bash
# Check available memory
free -h

# Use smaller project profile
export PROJECT_SIZE="small"
curl -fsSL https://install.leanvibe.com/project-index.sh | bash
```

#### Docker Issues
```bash
# Check Docker status
docker version
docker compose version

# Restart Docker daemon
sudo systemctl restart docker

# Clean up Docker resources
docker system prune -f
```

### Service Status Commands

```bash
# Installation directory
cd /opt/project-index

# Check service status
docker compose ps

# View service logs
docker compose logs -f project-index
docker compose logs -f postgres
docker compose logs -f redis

# Restart services
docker compose restart project-index

# Full restart
docker compose down && docker compose up -d

# Update to latest version
docker compose pull && docker compose up -d
```

### Health Checks

```bash
# API health check
curl http://localhost:8100/health

# Database connectivity
curl http://localhost:8100/api/health/database

# Cache connectivity  
curl http://localhost:8100/api/health/cache

# Full system status
curl http://localhost:8100/api/health/all
```

### Log Analysis

```bash
# View application logs
docker compose logs project-index | grep ERROR

# View database logs
docker compose logs postgres | tail -50

# View all logs with timestamps
docker compose logs -t --since="1h"

# Follow logs in real-time
docker compose logs -f project-index
```

## Advanced Configuration

### Custom Configuration

```bash
# Edit configuration after installation
cd /opt/project-index
nano .env

# Apply configuration changes
docker compose restart
```

### Environment Variables

```bash
# Performance tuning
export WORKER_CONCURRENCY=4
export ANALYSIS_BATCH_SIZE=50
export CACHE_TTL_HOURS=48

# Feature flags
export ENABLE_ML_ANALYSIS=true
export ENABLE_ADVANCED_SEARCH=true
export ENABLE_METRICS=true

# Security settings
export ENABLE_AUTHENTICATION=true
export API_KEY_REQUIRED=true
export CORS_ORIGINS="https://yourdomain.com"
```

### Volume Customization

```bash
# Custom data directory
export DATA_DIR="/custom/data/path"

# Custom cache directory  
export CACHE_DIR="/fast/ssd/cache"

# Backup directory
export BACKUP_DIR="/backup/project-index"
```

## Performance Tuning

### Resource Optimization

```bash
# For high-performance systems
export PROJECT_INDEX_MEMORY_LIMIT=4G
export PROJECT_INDEX_CPU_LIMIT=2.0
export WORKER_REPLICAS=4

# For resource-constrained systems
export PROJECT_INDEX_MEMORY_LIMIT=512M
export PROJECT_INDEX_CPU_LIMIT=0.25
export WORKER_REPLICAS=1
```

### Analysis Optimization

```bash
# Fast analysis mode
export ANALYSIS_MODE=fast
export MAX_FILE_SIZE_MB=5
export ANALYSIS_TIMEOUT_SECONDS=60

# Comprehensive analysis mode
export ANALYSIS_MODE=comprehensive
export MAX_FILE_SIZE_MB=25
export ANALYSIS_TIMEOUT_SECONDS=1200
```

### Database Tuning

```bash
# PostgreSQL optimization
export POSTGRES_SHARED_BUFFERS=256MB
export POSTGRES_EFFECTIVE_CACHE_SIZE=1GB
export POSTGRES_WORK_MEM=16MB

# Connection pooling
export POSTGRES_MAX_CONNECTIONS=100
export DATABASE_POOL_SIZE=20
```

### Cache Optimization

```bash
# Redis optimization
export REDIS_MEMORY_LIMIT=512M
export REDIS_MAXMEMORY_POLICY=allkeys-lru
export CACHE_COMPRESSION=true
```

## Integration Examples

### CI/CD Integration

#### GitHub Actions

```yaml
name: Project Analysis
on: [push, pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Project Index
        run: |
          curl -fsSL https://install.leanvibe.com/project-index.sh | \
          bash -s -- --mode ci --quiet
      
      - name: Run Analysis
        run: |
          # Wait for services to start
          sleep 30
          
          # Trigger analysis
          curl -X POST http://localhost:8100/api/projects/analyze
          
          # Wait for completion
          sleep 60
          
          # Export results
          curl http://localhost:8100/api/projects/export > analysis.json
      
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: project-analysis
          path: analysis.json
```

#### GitLab CI

```yaml
stages:
  - analyze

project-analysis:
  stage: analyze
  image: docker:20.10
  services:
    - docker:20.10-dind
  script:
    - curl -fsSL https://install.leanvibe.com/project-index.sh | bash -s -- --mode ci
    - sleep 30
    - curl -X POST http://localhost:8100/api/projects/analyze
    - sleep 60
    - curl http://localhost:8100/api/projects/export > analysis.json
  artifacts:
    paths:
      - analysis.json
    expire_in: 1 week
```

### IDE Integration

#### VSCode Extension

```json
{
  "projectIndex.apiUrl": "http://localhost:8100",
  "projectIndex.enableRealTime": true,
  "projectIndex.autoAnalyze": true
}
```

### Webhook Integration

```bash
# Configure webhooks for external integrations
curl -X POST http://localhost:8100/api/webhooks \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-webhook-endpoint.com/project-index",
    "events": ["analysis_complete", "dependency_changed"],
    "secret": "your-webhook-secret"
  }'
```

## Backup and Recovery

### Automated Backups

```bash
# Enable automated backups
cd /opt/project-index
echo "ENABLE_BACKUPS=true" >> .env
echo "BACKUP_SCHEDULE=0 2 * * *" >> .env  # Daily at 2 AM
docker compose restart
```

### Manual Backup

```bash
# Create manual backup
cd /opt/project-index
./scripts/backup.sh

# Restore from backup
./scripts/restore.sh backup-20240116-020000.tar.gz
```

### Data Export

```bash
# Export all project data
curl http://localhost:8100/api/projects/export?format=json > project-backup.json

# Export specific analysis
curl http://localhost:8100/api/analysis/12345/export > analysis-backup.json
```

## Monitoring and Alerting

### Prometheus Metrics

Access metrics at: `http://localhost:9090` (if monitoring enabled)

Key metrics to monitor:
- `project_index_api_requests_total`
- `project_index_analysis_duration_seconds`
- `project_index_files_processed_total`
- `project_index_cache_hit_ratio`
- `project_index_database_connections`

### Grafana Dashboards

Access dashboards at: `http://localhost:3001` (if monitoring enabled)

Default login: `admin/admin`

Pre-configured dashboards:
- Project Index Overview
- Performance Metrics
- System Resources
- Analysis Workflows

### Custom Alerts

```bash
# Add custom alert rules
cd /opt/project-index/monitoring
nano alert_rules.yml

# Reload Prometheus configuration
curl -X POST http://localhost:9090/-/reload
```

## Support and Community

### Documentation
- Full API Documentation: `http://localhost:8100/docs`
- Architecture Guide: [docs/ARCHITECTURE.md](../ARCHITECTURE.md)
- Troubleshooting: [docs/TROUBLESHOOTING.md](../TROUBLESHOOTING.md)

### Community Resources
- GitHub Repository: https://github.com/leanvibe/project-index
- Documentation Site: https://docs.leanvibe.com/project-index
- Community Discord: https://discord.gg/leanvibe
- Issue Tracker: https://github.com/leanvibe/project-index/issues

### Support Channels
- Bug Reports: GitHub Issues
- Feature Requests: GitHub Discussions
- Community Support: Discord
- Enterprise Support: support@leanvibe.com

## License and Legal

Project Index Universal Installer is released under the MIT License.

For enterprise licensing and support options, contact: enterprise@leanvibe.com