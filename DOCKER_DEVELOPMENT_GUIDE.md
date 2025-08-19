# 🐳 Docker Development Guide

Complete Docker-based development environment for LeanVibe Agent Hive 2.0, designed for one-command setup and optimal developer experience.

## 🚀 Quick Start

### Prerequisites
- Docker 24.0+ with BuildKit enabled
- Docker Compose 2.0+
- 8GB+ RAM recommended
- 10GB+ free disk space

### One-Command Setup
```bash
# 1. Set up the Docker environment
make docker-setup

# 2. Add your API keys to .env.docker
# Edit the file and add your ANTHROPIC_API_KEY

# 3. Start the development environment
make docker-dev
```

That's it! Your complete development environment is now running.

## 🛠️ Available Services

### Core Services
- **🔌 API Server**: FastAPI backend at http://localhost:8100
- **🎨 Frontend PWA**: React/Lit PWA at http://localhost:3001  
- **🐘 PostgreSQL**: Database with pgvector extension
- **🔴 Redis**: Cache and message broker

### Development Tools  
- **🗄️ pgAdmin**: Database management at http://localhost:5150
- **📦 Redis Commander**: Redis management at http://localhost:8201
- **🔍 Adminer**: Lightweight DB tool at http://localhost:8203

### Monitoring Stack (Optional)
- **📊 Prometheus**: Metrics collection at http://localhost:9090
- **📈 Grafana**: Dashboards at http://localhost:3101
- **🔍 Node Exporter**: System metrics
- **📊 cAdvisor**: Container metrics

### Development Features
- **🔥 Hot Reload**: Both backend and frontend automatically reload on code changes
- **🐛 Debug Support**: Debug port exposed for IDE attachment
- **💾 Data Persistence**: All data persists between container restarts
- **🧪 Testing Environment**: Isolated test containers
- **📝 Development Logs**: Centralized logging with proper retention

## 📋 Docker Commands Reference

### Setup & Management
```bash
# Initial setup
make docker-setup              # Set up environment and create .env.docker
make docker-dev                # Start full development stack
make docker-dev-bg             # Start in background
make docker-dev-minimal        # Start only essential services
make docker-dev-full           # Start everything including monitoring

# Management
make docker-stop               # Stop all services
make docker-restart            # Restart services  
make docker-logs               # Show logs from all services
make docker-health             # Check health of all services
make docker-ps                 # Show running containers
make docker-stats              # Show resource usage
```

### Development Tools
```bash
# Shell Access
make docker-shell              # Open shell in API container
make docker-db-shell           # Open PostgreSQL shell
make docker-redis-shell        # Open Redis CLI

# Building & Testing
make docker-build              # Build all images
make docker-build-no-cache     # Build without cache
make docker-test               # Run tests in Docker
```

### Cleanup
```bash
make docker-clean              # Clean up containers and images
make docker-clean-all          # Complete cleanup (DESTRUCTIVE)
```

## 🏗️ Development Workflow

### Daily Development Flow
```bash
# Morning routine
make docker-dev-bg             # Start services in background
make docker-health             # Verify everything is healthy

# Development work
# - Edit code (hot reload active)
# - View logs: make docker-logs
# - Test: make docker-test

# Evening routine  
make docker-stop               # Stop services
```

### Code Changes
1. **Backend Changes**: Auto-reload enabled via volume mounting
2. **Frontend Changes**: Vite HMR provides instant updates
3. **Database Changes**: Use `make docker-db-shell` for schema work
4. **Dependencies**: Rebuild containers after package.json/requirements.txt changes

### Testing Workflow
```bash
# Run tests in isolated environment
make docker-test

# Interactive testing
make docker-shell
# Inside container: pytest -xvs tests/your_test.py
```

## 🔧 Configuration

### Environment Files
- **`.env.docker.example`**: Template with all available settings
- **`.env.docker`**: Your local configuration (created by `make docker-setup`)

### Key Configuration Areas

#### Database Configuration
```bash
# PostgreSQL settings
POSTGRES_DB=leanvibe_agent_hive
POSTGRES_USER=leanvibe_user
POSTGRES_PASSWORD=your_secure_password
DATABASE_URL=postgresql+asyncpg://user:pass@postgres:5432/dbname
```

#### API Configuration  
```bash
# FastAPI settings
ANTHROPIC_API_KEY=your-api-key-here
DEBUG=true
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3001,http://localhost:8100
```

#### Frontend Configuration
```bash
# Vite PWA settings
VITE_API_BASE_URL=http://localhost:8100
VITE_WS_URL=ws://localhost:8100/api/dashboard/ws/dashboard
VITE_ENABLE_DEBUG=true
```

### Port Configuration
All ports are configurable via environment variables:

| Service | Default Port | Environment Variable |
|---------|-------------|---------------------|
| API Server | 8100 | `MAIN_API_PORT` |
| Frontend PWA | 3001 | `DASHBOARD_PORT` |
| PostgreSQL | 5433 | `POSTGRES_PORT` |
| Redis | 6380 | `REDIS_PORT` |
| pgAdmin | 5150 | `PGADMIN_PORT` |
| Redis Commander | 8201 | `REDIS_COMMANDER_PORT` |
| Prometheus | 9090 | `PROMETHEUS_PORT` |
| Grafana | 3101 | `GRAFANA_PORT` |

## 🔄 Service Profiles

Docker Compose uses profiles to control which services run:

### Development Profile (`--profile development`)
- API server with hot reload
- Frontend PWA with HMR  
- Database management tools (pgAdmin, Redis Commander)
- Development optimizations enabled

### Tools Profile (`--profile tools`)
- Additional database tools (Adminer)
- Redis Insight for advanced Redis management
- Jupyter notebook for data analysis

### Monitoring Profile (`--profile monitoring`)
- Prometheus metrics collection
- Grafana dashboards
- System metrics exporters
- Container monitoring with cAdvisor

### Testing Profile (`--profile testing`)
- Isolated test environment
- Test database and Redis instance
- Coverage reporting setup

## 🐛 Debugging

### Debug API Server
```bash
# Start with debug port exposed
make docker-dev

# In your IDE, attach to localhost:5678
# Or use VS Code launch configuration
```

### Database Debugging
```bash
# Access database directly
make docker-db-shell

# Or use pgAdmin at http://localhost:5150
# Login: admin@leanvibe.dev / admin_password_docker
```

### Redis Debugging
```bash  
# Command line access
make docker-redis-shell

# Or use Redis Commander at http://localhost:8201
# Login: admin / admin_docker
```

### Log Analysis
```bash
# All services
make docker-logs

# Specific service
docker compose --env-file=.env.docker logs api -f

# With timestamps
docker compose --env-file=.env.docker logs -t api
```

## 📊 Monitoring & Performance

### Health Checks
All services include comprehensive health checks:
- **API**: `/health` endpoint with component status
- **PostgreSQL**: Connection and query verification  
- **Redis**: Ping response verification
- **Frontend**: HTTP response verification

### Metrics Collection
When monitoring profile is enabled:
- **Application Metrics**: Custom business logic metrics
- **System Metrics**: CPU, memory, disk, network
- **Container Metrics**: Per-container resource usage
- **Database Metrics**: Query performance, connection pools

### Performance Optimization
Development containers are optimized for:
- **Fast Builds**: Multi-stage builds with intelligent caching
- **Quick Startup**: Dependency pre-installation and health checks
- **Low Resources**: Reasonable memory limits for development
- **Hot Reload**: Efficient file watching and change detection

## 🔒 Security Considerations

### Development Security
- Non-root users in containers
- Restricted network access  
- Secret management via environment variables
- Security scanning with bandit/safety

### Production Readiness
- Security headers configured
- HTTPS termination via nginx
- Database connection encryption
- API rate limiting enabled

## 🚨 Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check which process is using a port
lsof -i :8100

# Change port in .env.docker
MAIN_API_PORT=8200
```

#### Permission Issues
```bash
# Reset permissions
make docker-clean
make docker-setup
```

#### Database Connection Issues
```bash
# Check database health
make docker-health

# Reset database
make docker-clean-all
make docker-setup
```

#### Build Failures
```bash
# Clean rebuild
make docker-build-no-cache

# Check Docker space
docker system df
```

### Getting Help
1. Check service health: `make docker-health`
2. Check logs: `make docker-logs`
3. Verify configuration: Review `.env.docker`
4. Clean restart: `make docker-clean && make docker-dev`

## 🎯 Advanced Usage

### Custom Service Combinations
```bash
# Just database services
docker compose --env-file=.env.docker up postgres redis

# API + Frontend only
docker compose --env-file=.env.docker up api frontend

# Full monitoring stack
make docker-dev-full
```

### Development with External Services
Edit `.env.docker` to point to external services:
```bash
# Use external Redis
REDIS_URL=redis://your-redis-server:6379

# Use external PostgreSQL  
DATABASE_URL=postgresql+asyncpg://user:pass@your-db:5432/dbname
```

### Custom Docker Images
```bash
# Build with custom args
docker compose --env-file=.env.docker build --build-arg PYTHON_VERSION=3.11 api

# Use development image
docker compose --env-file=.env.docker build --target development api
```

## 📝 Best Practices

### Development Best Practices
1. **Keep .env.docker up to date** with .env.docker.example
2. **Use profiles** to run only needed services
3. **Monitor resource usage** with `make docker-stats`
4. **Regular cleanup** with `make docker-clean`
5. **Version control** your .env.docker (without secrets)

### Performance Best Practices
1. **Use cached volumes** for node_modules and Python packages
2. **Enable BuildKit** for faster builds
3. **Prune regularly** to reclaim disk space
4. **Monitor logs** for performance issues
5. **Use minimal service sets** during development

### Security Best Practices
1. **Never commit secrets** in Docker files
2. **Use development passwords** only in development
3. **Keep base images updated** regularly
4. **Scan for vulnerabilities** before deployment
5. **Use least privilege** principles

## 📚 Additional Resources

- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Development Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [PostgreSQL Docker Documentation](https://hub.docker.com/_/postgres)
- [Redis Docker Documentation](https://hub.docker.com/_/redis)

---

🎉 **Happy Coding!** Your Docker development environment is now ready for productive development.