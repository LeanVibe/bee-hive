# 🌐 Project Index Multi-Project Deployment Guide

## 📋 Overview

This guide enables deployment of the Project Index system to any software project, providing intelligent code analysis, dependency tracking, and context optimization across different codebases and technology stacks.

## ✅ System Requirements

### Minimum Requirements
- **Python**: 3.11+
- **PostgreSQL**: 12+ 
- **Redis**: 6+
- **Memory**: 2GB RAM minimum
- **Storage**: 500MB + project size
- **CPU**: 2 cores minimum

### Recommended Production Setup
- **Python**: 3.12
- **PostgreSQL**: 15+ with extensions
- **Redis**: 7+ with persistence
- **Memory**: 8GB RAM for large projects
- **Storage**: 2GB + 2x project size
- **CPU**: 4+ cores for parallel analysis

## 🚀 Quick Deployment Steps

### 1. Copy Project Index Components

Copy these directories from bee-hive to your target project:

```bash
# Core Project Index system
cp -r app/project_index/ /path/to/your-project/app/
cp -r app/api/project_index.py /path/to/your-project/app/api/
cp -r app/schemas/project_index.py /path/to/your-project/app/schemas/
cp -r app/models/project_index.py /path/to/your-project/app/models/

# Database migration
cp migrations/versions/022_add_project_index_system.py /path/to/your-project/migrations/versions/

# Tests (optional but recommended)
cp -r tests/project_index/ /path/to/your-project/tests/
```

### 2. Install Dependencies

Add to your `requirements.txt` or `pyproject.toml`:

```txt
# Project Index System Dependencies
tree-sitter>=0.20.0          # Code parsing and analysis
networkx>=3.0                # Dependency graph analysis
tree-sitter-python>=0.20.0   # Python language parser
tree-sitter-javascript>=0.20.0  # JavaScript language parser
tree-sitter-typescript>=0.20.0  # TypeScript language parser
watchdog>=3.0.0              # File system event monitoring
apscheduler>=3.10.4          # Background task scheduling
```

### 3. Database Setup

```bash
# Run migration to create tables
alembic upgrade head

# Verify tables created
psql your_database -c "\dt" | grep project
```

### 4. Configuration

Add to your `.env` file:

```env
# Project Index Configuration
PROJECT_INDEX_ENABLED=true
PROJECT_INDEX_REAL_TIME_MONITORING=true
PROJECT_INDEX_CACHE_ENABLED=true
PROJECT_INDEX_MAX_FILE_SIZE_MB=10
PROJECT_INDEX_ANALYSIS_BATCH_SIZE=50
PROJECT_INDEX_MAX_CONCURRENT_ANALYSES=3
```

### 5. API Integration

Add to your FastAPI app:

```python
# main.py or similar
from app.api.project_index import router as project_index_router

app.include_router(
    project_index_router,
    prefix="/api/project-index",
    tags=["Project Index"]
)
```

## 🎯 Project-Specific Configurations

### Python Projects

```python
config = ProjectIndexConfig(
    project_name="my-python-project",
    root_path="/path/to/project",
    file_patterns={
        "include": [
            "**/*.py",
            "**/*.pyx",      # Cython files
            "**/*.pyi",      # Type stubs
            "**/pyproject.toml",
            "**/setup.py",
            "**/requirements*.txt"
        ]
    },
    ignore_patterns={
        "exclude": [
            "**/__pycache__/**",
            "**/build/**",
            "**/dist/**",
            "**/.tox/**",
            "**/.pytest_cache/**",
            "**/venv/**",
            "**/.venv/**"
        ]
    },
    analysis_config={
        "languages": ["python"],
        "analysis_depth": 3,
        "include_tests": True,
        "include_documentation": True
    }
)
```

### JavaScript/TypeScript Projects

```python
config = ProjectIndexConfig(
    project_name="my-web-project",
    root_path="/path/to/project",
    file_patterns={
        "include": [
            "**/*.js",
            "**/*.jsx",
            "**/*.ts",
            "**/*.tsx",
            "**/*.vue",       # Vue.js files
            "**/*.svelte",    # Svelte files
            "**/package*.json",
            "**/tsconfig*.json",
            "**/*.config.js"
        ]
    },
    ignore_patterns={
        "exclude": [
            "**/node_modules/**",
            "**/build/**",
            "**/dist/**",
            "**/.next/**",
            "**/.nuxt/**",
            "**/coverage/**",
            "**/*.min.js"
        ]
    },
    analysis_config={
        "languages": ["javascript", "typescript"],
        "analysis_depth": 4,
        "include_tests": True
    }
)
```

### Full-Stack Projects

```python
config = ProjectIndexConfig(
    project_name="my-fullstack-project",
    root_path="/path/to/project",
    file_patterns={
        "include": [
            # Backend
            "**/*.py",
            "**/*.java",
            "**/*.go",
            "**/*.rs",        # Rust files
            # Frontend
            "**/*.js",
            "**/*.jsx",
            "**/*.ts",
            "**/*.tsx",
            "**/*.vue",
            # Configuration
            "**/*.json",
            "**/*.yaml",
            "**/*.yml",
            "**/*.toml",
            "**/*.xml",
            # Documentation
            "**/*.md",
            "**/*.rst"
        ]
    },
    ignore_patterns={
        "exclude": [
            # Python
            "**/__pycache__/**",
            "**/venv/**",
            # JavaScript
            "**/node_modules/**",
            # Build artifacts
            "**/build/**",
            "**/dist/**",
            "**/target/**",    # Rust/Java builds
            # IDE files
            "**/.vscode/**",
            "**/.idea/**",
            # Version control
            "**/.git/**"
        ]
    },
    analysis_config={
        "languages": ["python", "javascript", "typescript", "java", "go", "rust"],
        "analysis_depth": 3,
        "include_tests": True,
        "include_documentation": True
    }
)
```

### Microservices Architecture

```python
# Configuration for each service
def create_microservice_config(service_name, service_path):
    return ProjectIndexConfig(
        project_name=f"microservice-{service_name}",
        root_path=service_path,
        file_patterns={
            "include": [
                "**/*.py",
                "**/*.js",
                "**/*.ts",
                "**/Dockerfile*",
                "**/*.yaml",
                "**/*.yml",
                "**/requirements*.txt",
                "**/package*.json"
            ]
        },
        ignore_patterns={
            "exclude": [
                "**/__pycache__/**",
                "**/node_modules/**",
                "**/venv/**",
                "**/.git/**"
            ]
        },
        analysis_config={
            "languages": ["python", "javascript", "typescript"],
            "analysis_depth": 2,  # Reduced for microservices
            "include_tests": True,
            "service_boundaries": True
        }
    )

# Index multiple services
services = [
    ("auth-service", "/path/to/auth-service"),
    ("user-service", "/path/to/user-service"),
    ("payment-service", "/path/to/payment-service")
]

for service_name, service_path in services:
    config = create_microservice_config(service_name, service_path)
    # Create project index for each service
```

## 🛠️ Advanced Deployment Patterns

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Run migrations and start server
CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/project_index
      - REDIS_URL=redis://redis:6379
      - PROJECT_INDEX_ENABLED=true
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: project_index
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: project-index
spec:
  replicas: 3
  selector:
    matchLabels:
      app: project-index
  template:
    metadata:
      labels:
        app: project-index
    spec:
      containers:
      - name: project-index
        image: your-registry/project-index:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: project-index-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: project-index-secrets
              key: redis-url
        - name: PROJECT_INDEX_ENABLED
          value: "true"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30

---
apiVersion: v1
kind: Service
metadata:
  name: project-index-service
spec:
  selector:
    app: project-index
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### CI/CD Integration

```yaml
# .github/workflows/project-index.yml
name: Project Index Analysis

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  analyze:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
          
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        
    - name: Run database migrations
      run: |
        alembic upgrade head
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        
    - name: Run Project Index analysis
      run: |
        python scripts/analyze_project.py
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        REDIS_URL: redis://localhost:6379
        PROJECT_INDEX_ENABLED: true
        
    - name: Upload analysis results
      uses: actions/upload-artifact@v3
      with:
        name: project-analysis
        path: analysis-results/
```

## 📊 Performance Optimization

### Large Projects (>10k files)

```python
# Optimized configuration for large projects
config = ProjectIndexConfig(
    project_name="large-project",
    root_path="/path/to/project",
    
    # Performance settings
    analysis_batch_size=10,        # Smaller batches
    max_concurrent_analyses=2,     # Limit concurrency
    
    # Caching settings
    cache_enabled=True,
    cache_config={
        "max_memory_mb": 1000,     # Increase cache size
        "enable_compression": True,
        "compression_threshold": 512
    },
    
    # Monitoring settings
    monitoring_config={
        "debounce_seconds": 5.0,   # Longer debounce
        "max_file_size_mb": 5      # Skip very large files
    },
    
    # Selective analysis
    file_patterns={
        "include": [
            "src/**/*.py",         # Only analyze source code
            "lib/**/*.js",
            "!**/*test*/**"        # Exclude test directories
        ]
    }
)
```

### Resource Limits

```python
# Resource-constrained environments
config = ProjectIndexConfig(
    project_name="resource-limited",
    root_path="/path/to/project",
    
    # Minimal resource usage
    analysis_batch_size=5,
    max_concurrent_analyses=1,
    enable_real_time_monitoring=False,  # Disable file watching
    enable_ml_analysis=False,           # Disable ML features
    
    cache_config={
        "max_memory_mb": 100,           # Minimal cache
        "enable_compression": True
    }
)
```

## 🔧 Integration Examples

### Custom Analysis Workflow

```python
import asyncio
from app.project_index.core import ProjectIndexer
from app.project_index.models import ProjectIndexConfig

async def custom_analysis_workflow():
    """Custom analysis workflow for specific project needs."""
    
    config = ProjectIndexConfig(
        project_name="custom-workflow",
        root_path="/path/to/project"
    )
    
    async with ProjectIndexer(config=config) as indexer:
        # Step 1: Create project
        project = await indexer.create_project(
            name="Custom Project",
            root_path="/path/to/project",
            description="Project with custom analysis workflow"
        )
        
        # Step 2: Initial full analysis
        print("Running initial analysis...")
        initial_result = await indexer.analyze_project(
            project_id=str(project.id),
            analysis_type="full_analysis"
        )
        print(f"Initial: {initial_result.files_processed} files analyzed")
        
        # Step 3: Dependency-focused analysis
        print("Analyzing dependencies...")
        dep_result = await indexer.analyze_project(
            project_id=str(project.id),
            analysis_type="dependency_mapping"
        )
        print(f"Dependencies: {dep_result.dependencies_found} found")
        
        # Step 4: Context optimization
        print("Optimizing context...")
        context_result = await indexer.analyze_project(
            project_id=str(project.id),
            analysis_type="context_optimization"
        )
        print("Context optimization completed")
        
        # Step 5: Get comprehensive statistics
        stats = await indexer.get_analysis_statistics()
        print(f"Total analysis time: {stats['analysis_time']:.2f}s")
        print(f"Cache efficiency: {stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']):.2%}")

# Run the workflow
if __name__ == "__main__":
    asyncio.run(custom_analysis_workflow())
```

### Multi-Project Dashboard Integration

```python
from typing import List, Dict
from app.project_index.core import ProjectIndexer

class MultiProjectDashboard:
    """Dashboard for managing multiple projects."""
    
    def __init__(self):
        self.projects: Dict[str, ProjectIndexer] = {}
    
    async def add_project(self, project_config: ProjectIndexConfig):
        """Add a new project to the dashboard."""
        indexer = ProjectIndexer(config=project_config)
        project = await indexer.create_project(
            name=project_config.project_name,
            root_path=project_config.root_path
        )
        self.projects[str(project.id)] = indexer
        return project.id
    
    async def get_all_project_stats(self):
        """Get statistics for all projects."""
        stats = {}
        for project_id, indexer in self.projects.items():
            project_stats = await indexer.get_analysis_statistics()
            stats[project_id] = project_stats
        return stats
    
    async def trigger_analysis_all(self):
        """Trigger analysis for all projects."""
        tasks = []
        for project_id, indexer in self.projects.items():
            task = indexer.analyze_project(
                project_id=project_id,
                analysis_type="incremental"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

# Usage
dashboard = MultiProjectDashboard()

# Add projects
config1 = ProjectIndexConfig(project_name="frontend", root_path="/path/to/frontend")
config2 = ProjectIndexConfig(project_name="backend", root_path="/path/to/backend")

await dashboard.add_project(config1)
await dashboard.add_project(config2)

# Get combined statistics
all_stats = await dashboard.get_all_project_stats()
print(f"Managing {len(all_stats)} projects")
```

## 🚨 Troubleshooting Guide

### Common Issues and Solutions

#### 1. High Memory Usage

**Problem**: Project Index consuming too much memory

**Solutions**:
```python
# Reduce cache size
config.cache_config["max_memory_mb"] = 200

# Enable compression
config.cache_config["enable_compression"] = True

# Reduce batch size
config.analysis_batch_size = 10

# Limit concurrent analyses
config.max_concurrent_analyses = 1
```

#### 2. Slow Analysis Performance

**Problem**: Analysis taking too long

**Solutions**:
```python
# Increase batch size (if you have memory)
config.analysis_batch_size = 100

# Increase concurrency (if you have CPU cores)
config.max_concurrent_analyses = 4

# Use selective file patterns
config.file_patterns = {
    "include": ["src/**/*.py"]  # Only analyze core source
}

# Enable incremental updates
config.incremental_updates = True
```

#### 3. Database Connection Issues

**Problem**: Cannot connect to database

**Check**:
```bash
# Test PostgreSQL connection
pg_isready -h localhost -p 5432

# Test database access
psql -h localhost -p 5432 -U username -d database_name -c "SELECT 1;"

# Check migration status
alembic current
alembic history
```

#### 4. Redis Connection Issues

**Problem**: Cannot connect to Redis

**Check**:
```bash
# Test Redis connection
redis-cli ping

# Check Redis configuration
redis-cli config get "*"

# Monitor Redis
redis-cli monitor
```

#### 5. File Analysis Errors

**Problem**: Certain files fail to analyze

**Solutions**:
```python
# Add file size limits
config.monitoring_config["max_file_size_mb"] = 5

# Skip binary files
config.ignore_patterns["exclude"].append("**/*.bin")

# Handle encoding issues
config.default_encoding = "utf-8"
```

## ✅ Deployment Checklist

### Pre-Deployment
- [ ] Dependencies installed
- [ ] Database configured and accessible
- [ ] Redis configured and accessible
- [ ] Environment variables set
- [ ] Migration files copied
- [ ] Project Index components copied

### Deployment
- [ ] Run database migrations
- [ ] Start Redis service
- [ ] Start application server
- [ ] Verify API endpoints respond
- [ ] Test WebSocket connections

### Post-Deployment
- [ ] Create initial project index
- [ ] Run test analysis
- [ ] Verify file monitoring
- [ ] Check performance metrics
- [ ] Set up monitoring/alerting

### Production Monitoring
- [ ] Database performance monitoring
- [ ] Redis memory usage monitoring
- [ ] Analysis performance tracking
- [ ] Error rate monitoring
- [ ] Resource usage alerts

## 📚 Additional Resources

### Documentation Links
- **API Reference**: `/docs` endpoint when server is running
- **Database Schema**: `migrations/versions/022_add_project_index_system.py`
- **Configuration Reference**: `app/project_index/models.py`

### Example Projects
- **Python Package**: Minimal Python package setup
- **React App**: Frontend JavaScript project
- **Django Backend**: Full-stack web application
- **Microservices**: Multi-service architecture

### Support and Community
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Comprehensive setup and usage guides
- **Examples**: Real-world deployment scenarios

---

**The Project Index system is now ready for deployment to any project!** 🚀

This guide provides everything needed to integrate intelligent code analysis and dependency tracking into your development workflow.