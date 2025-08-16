# Project Index Universal CLI Installer - Usage Guide

## Overview

The Project Index Universal CLI Installer is a comprehensive, user-friendly system that provides seamless one-command installation of the Project Index system for any codebase. It automatically detects your project's characteristics and sets up an optimized, production-ready Project Index environment in under 5 minutes.

## Features

### 🚀 One-Command Installation
```bash
python install_project_index.py /path/to/your/project
```

### 🔍 Intelligent Project Detection
- Automatically detects programming languages and frameworks
- Analyzes project structure and complexity
- Recommends optimal configuration profiles

### 🐳 Docker Infrastructure Automation
- Generates optimized Docker Compose configurations
- Sets up PostgreSQL with vector extensions
- Configures Redis for caching
- Includes monitoring and health checks

### 🔧 Framework Integration Generation
- Creates framework-specific integration code
- Supports Flask, React, Django, Vue.js, and more
- Provides ready-to-use API clients and components

### ⚡ Performance Optimization
- System-aware resource allocation
- Intelligent caching strategies
- Optimized for different project sizes

### 🛡️ Security Configuration
- Secure password generation
- SSL/TLS configuration
- Rate limiting and access control

### 📊 Comprehensive Validation
- Pre and post-installation health checks
- Performance monitoring
- Automated troubleshooting

## Quick Start

### Prerequisites

1. **Docker Desktop** (or compatible Docker engine)
2. **Python 3.8+**
3. **Docker Compose V2**
4. **Minimum 2GB RAM** (4GB+ recommended)
5. **1GB free disk space**

### Installation

#### Option 1: One-Command Installation
```bash
# Download and run installer
curl -fsSL https://raw.githubusercontent.com/leanvibe/bee-hive/main/install_project_index.py | python - /path/to/your/project
```

#### Option 2: Manual Installation
```bash
# Clone repository
git clone https://github.com/leanvibe/bee-hive.git
cd bee-hive

# Install dependencies
pip install -r requirements.txt

# Run installer
python install_project_index.py /path/to/your/project
```

## Usage Examples

### Basic Installation
```bash
# Analyze project and install with recommended settings
python install_project_index.py /home/user/my-web-app
```

### Specify Installation Profile
```bash
# Use specific profile for large projects
python install_project_index.py /home/user/enterprise-app --profile large
```

### Automated Installation (CI/CD)
```bash
# Non-interactive installation
python install_project_index.py /project --auto-confirm --skip-validation
```

### Development Installation
```bash
# Install with verbose logging
python install_project_index.py /home/user/prototype --profile small --verbose
```

## Installation Profiles

### Small Projects (< 1k files, 1 developer)
- **Memory**: 1GB RAM, 0.5 CPU cores
- **Services**: API, Database, Cache
- **Features**: Basic indexing, simple validation
- **Best for**: Prototypes, personal projects, scripts

```bash
python install_project_index.py /project --profile small
```

### Medium Projects (1k-10k files, 2-5 developers)
- **Memory**: 2GB RAM, 1 CPU core
- **Services**: API, Database, Cache, Monitoring
- **Features**: Advanced indexing, team coordination, performance optimization
- **Best for**: Web applications, mobile apps, typical business projects

```bash
python install_project_index.py /project --profile medium
```

### Large Projects (> 10k files, team development)
- **Memory**: 4GB RAM, 2 CPU cores
- **Services**: API, Database, Cache, Monitoring, Analytics
- **Features**: Enterprise features, advanced analytics, high availability
- **Best for**: Large applications, microservices, enterprise systems

```bash
python install_project_index.py /project --profile large
```

### Enterprise Projects (Mission-critical, large teams)
- **Memory**: 8GB RAM, 4 CPU cores
- **Services**: Full enterprise stack with clustering
- **Features**: Full enterprise features, compliance, advanced security
- **Best for**: Mission-critical applications, financial systems, healthcare

```bash
python install_project_index.py /project --profile enterprise
```

## Command-Line Options

```
Usage: python install_project_index.py [OPTIONS] PROJECT_PATH

Options:
  --profile {small|medium|large|enterprise}  Installation profile
  --auto-confirm                             Skip confirmation prompts
  --skip-validation                          Skip post-deployment validation
  --verbose, -v                              Enable verbose logging
  --help, -h                                 Show help message

Examples:
  python install_project_index.py /path/to/project
  python install_project_index.py /path/to/project --profile large
  python install_project_index.py /path/to/project --auto-confirm
```

## What Gets Installed

### Docker Services
- **PostgreSQL 15** with pgvector extension for semantic search
- **Redis 7** for caching and session management
- **Project Index API** for search and analysis
- **Optional Dashboard** for web interface
- **Optional Monitoring** (Prometheus + Grafana)

### Configuration Files
- `docker-compose.yml` - Service orchestration
- `.env` - Environment variables
- `project-index-config.json` - Complete configuration
- `scripts/` - Management scripts
- `config/` - Service configurations

### Framework Integrations
- **Python Flask**: API client, middleware, CLI commands
- **React**: Hooks, components, context providers
- **Vue.js**: Plugin, composables, components
- **Django**: Middleware, admin integration, management commands
- **Express.js**: Middleware, routes, helpers

### Management Scripts
- `scripts/start.sh` - Start all services
- `scripts/stop.sh` - Stop all services
- `scripts/health-check.sh` - Health validation
- `scripts/backup.sh` - Data backup (enterprise)

## Post-Installation

### Verify Installation
```bash
# Check service status
docker-compose ps

# Run health check
./scripts/health-check.sh

# Test API
curl http://localhost:8100/health
```

### Access Services
- **API**: http://localhost:8100
- **API Documentation**: http://localhost:8100/docs
- **Dashboard**: http://localhost:8101 (if enabled)
- **Metrics**: http://localhost:9090 (if monitoring enabled)

### Integration Examples

#### Python Flask
```python
from project_index_client import project_index

app = Flask(__name__)
project_index.init_app(app)

@app.route('/search')
@track_context
def search():
    client = app.extensions['project_index']
    results = client.search_code(request.args.get('q'))
    return jsonify(results)
```

#### React
```javascript
import { ProjectIndexProvider, useCodeSearch } from './hooks/useProjectIndex';

function App() {
  return (
    <ProjectIndexProvider>
      <SearchComponent />
    </ProjectIndexProvider>
  );
}

function SearchComponent() {
  const { query, setQuery, results, loading } = useCodeSearch();
  
  return (
    <div>
      <input value={query} onChange={(e) => setQuery(e.target.value)} />
      {results.map(result => <div key={result.id}>{result.content}</div>)}
    </div>
  );
}
```

## Management Commands

### Service Management
```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Restart specific service
docker-compose restart project-index-api

# View logs
docker-compose logs -f project-index-api
```

### Health Monitoring
```bash
# System health check
./scripts/health-check.sh

# Service status
docker-compose ps

# Resource usage
docker stats

# API health
curl http://localhost:8100/health
```

### Data Management
```bash
# Backup database
docker-compose exec postgres pg_dump -U project_index project_index_db > backup.sql

# Restore database
docker-compose exec -T postgres psql -U project_index project_index_db < backup.sql

# Clear cache
docker-compose exec redis redis-cli FLUSHALL
```

### Configuration Updates
```bash
# Update environment variables
nano .env

# Reload configuration
docker-compose down && docker-compose up -d

# Update resource limits
nano docker-compose.yml
docker-compose up -d --force-recreate
```

## Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check port usage
lsof -i :8100

# Stop conflicting service
sudo kill -9 $(lsof -t -i:8100)

# Or change ports in .env file
echo "API_PORT=8200" >> .env
docker-compose down && docker-compose up -d
```

#### Memory Issues
```bash
# Check memory usage
free -h
docker stats

# Reduce memory allocation in docker-compose.yml
# Or upgrade system memory
```

#### Docker Issues
```bash
# Check Docker status
docker info

# Restart Docker daemon
sudo systemctl restart docker

# Clean up Docker resources
docker system prune -f
```

#### Service Startup Issues
```bash
# Check service logs
docker-compose logs project-index-api

# Check health status
docker-compose exec project-index-api curl localhost:8100/health

# Restart services
docker-compose restart
```

### Validation and Diagnostics

#### Run Comprehensive Validation
```bash
# Standard validation
python -c "
import asyncio
from cli.validation_framework import ValidationFramework, ValidationLevel
async def main():
    validator = ValidationFramework()
    report = await validator.run_validation(ValidationLevel.COMPREHENSIVE)
    print(f'Status: {report.overall_status.value}')
    for result in report.validation_results:
        if result.status.value != 'healthy':
            print(f'Issue: {result.check_name} - {result.message}')
asyncio.run(main())
"
```

#### Performance Analysis
```bash
# Check system resources
htop

# Check Docker resource usage
docker stats

# Test API performance
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8100/api/search?q=test
```

### Getting Help

#### Log Analysis
```bash
# View installer logs
tail -f logs/installation.log

# View service logs
docker-compose logs --tail=100 -f

# View system logs
journalctl -f -u docker
```

#### Support Resources
- **Documentation**: Check the `docs/` directory for detailed guides
- **GitHub Issues**: Report bugs and request features
- **Validation Reports**: Use the HTML health reports for detailed diagnostics
- **Community**: Join our community for support and best practices

## Performance Tuning

### Resource Optimization
```bash
# For development (reduce resources)
export DEPLOYMENT_PROFILE=small
docker-compose down && docker-compose up -d

# For production (increase resources)
export DEPLOYMENT_PROFILE=large
docker-compose down && docker-compose up -d
```

### Index Optimization
```yaml
# In project-index-config.yaml
indexing:
  batch_size: 100          # Increase for faster indexing
  concurrent_indexers: 4   # Increase with more CPU cores
  index_frequency: "5m"    # More frequent updates
```

### Database Tuning
```bash
# Increase PostgreSQL memory
# Edit docker-compose.yml
environment:
  - POSTGRES_SHARED_BUFFERS=256MB
  - POSTGRES_EFFECTIVE_CACHE_SIZE=1GB
```

## Security Considerations

### Production Deployment
```bash
# Generate secure passwords
openssl rand -base64 32

# Enable SSL
export SSL_ENABLED=true

# Configure firewall
sudo ufw allow 8100/tcp
sudo ufw deny 5432/tcp  # Block direct database access
```

### Access Control
```yaml
# In project-index-config.yaml
security:
  api_authentication: true
  rate_limiting: true
  cors_origins: ["https://yourdomain.com"]
```

## Advanced Usage

### Custom Configuration
```python
# Generate custom configuration
from cli.config_generator import ConfigurationGenerator
from cli.project_detector import ProjectDetector

detector = ProjectDetector()
analysis = detector.analyze_project("/path/to/project")

generator = ConfigurationGenerator()
config = generator.generate_configuration(
    analysis=analysis,
    deployment_profile=DeploymentProfile.MEDIUM,
    user_preferences={
        "optimization_level": "performance",
        "enable_monitoring": True,
        "enable_ssl": True
    }
)
```

### Integration Development
```python
# Create custom framework adapter
from cli.framework_adapters import FrameworkAdapter

class MyFrameworkAdapter(FrameworkAdapter):
    def can_handle(self, framework, analysis):
        return framework.name == "MyFramework"
    
    def generate_integration(self, framework, analysis, config):
        # Generate integration code
        pass
```

### Monitoring Integration
```yaml
# Custom Prometheus metrics
prometheus:
  scrape_configs:
    - job_name: 'project-index'
      static_configs:
        - targets: ['project-index-api:8100']
      metrics_path: '/metrics'
```

This comprehensive guide should help users understand and effectively use the Project Index Universal CLI Installer. The system is designed to be both powerful for advanced users and simple for beginners, with intelligent defaults and comprehensive validation to ensure successful installations.