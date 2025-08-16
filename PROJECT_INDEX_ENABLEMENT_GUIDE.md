# 🚀 Project Index Enablement Guide for Bee-Hive

## 📋 Overview

This guide enables the Project Index system for the LeanVibe Agent Hive 2.0 project, providing intelligent code analysis, dependency tracking, and context optimization.

## ✅ Pre-Enablement Validation

### System Status ✅ READY
- ✅ **Dependencies Resolved**: All Project Index dependencies installed
- ✅ **Implementation Complete**: Core, API, schemas, and models working
- ✅ **Test Coverage**: 325 tests across 6 test files (4,450 lines)
- ✅ **Database Schema**: Migration `022_add_project_index_system.py` ready
- ✅ **Integration Tested**: All components validated and functional

## 🎯 Enablement Steps

### Step 1: Database Setup

```bash
# 1. Start PostgreSQL service
# For Docker:
docker run --name postgres-dev -e POSTGRES_PASSWORD=leanvibe -e POSTGRES_USER=leanvibe_user -e POSTGRES_DB=leanvibe_agent_hive -p 5434:5432 -d postgres:15

# For local PostgreSQL:
brew services start postgresql
createdb leanvibe_agent_hive

# 2. Run migrations to create Project Index tables
alembic upgrade head

# 3. Verify tables created
psql leanvibe_agent_hive -c "\dt" | grep project
```

### Step 2: Redis Setup

```bash
# Start Redis for caching and real-time events
# For Docker:
docker run --name redis-dev -p 6381:6379 -d redis:7-alpine

# For local Redis:
brew services start redis
```

### Step 3: Environment Configuration

Create or update `.env` file:

```bash
# Database Configuration
DATABASE_URL=postgresql://leanvibe_user:leanvibe@localhost:5434/leanvibe_agent_hive

# Redis Configuration  
REDIS_URL=redis://localhost:6381

# Project Index Settings
PROJECT_INDEX_ENABLED=true
PROJECT_INDEX_REAL_TIME_MONITORING=true
PROJECT_INDEX_CACHE_ENABLED=true
PROJECT_INDEX_MAX_FILE_SIZE_MB=10
PROJECT_INDEX_ANALYSIS_BATCH_SIZE=50
PROJECT_INDEX_MAX_CONCURRENT_ANALYSES=5

# Optional: Enable advanced features
PROJECT_INDEX_INCREMENTAL_UPDATES=true
PROJECT_INDEX_EVENTS_ENABLED=true
PROJECT_INDEX_ML_ANALYSIS=false
```

### Step 4: API Integration

The Project Index API is automatically available at:

```
Base URL: /api/project-index/
WebSocket: /api/project-index/ws
```

Key endpoints:
- `POST /api/project-index/create` - Create project index
- `GET /api/project-index/{project_id}` - Get project details
- `GET /api/project-index/{project_id}/files` - List analyzed files
- `GET /api/project-index/{project_id}/dependencies` - Get dependency graph
- `POST /api/project-index/{project_id}/analyze` - Trigger analysis

### Step 5: Create Bee-Hive Project Index

```python
# Create index for current project
import asyncio
from pathlib import Path
from app.project_index.core import ProjectIndexer
from app.project_index.models import ProjectIndexConfig

async def create_bee_hive_index():
    config = ProjectIndexConfig(
        project_name="bee-hive",
        root_path=str(Path.cwd()),
        enable_real_time_monitoring=True,
        enable_ml_analysis=False,
        analysis_config={
            "languages": ["python", "javascript", "typescript"],
            "analysis_depth": 3,
            "include_tests": True
        },
        file_patterns={
            "include": [
                "**/*.py",
                "**/*.js", 
                "**/*.ts",
                "**/*.md",
                "**/*.yml",
                "**/*.yaml",
                "**/*.json"
            ]
        },
        ignore_patterns={
            "exclude": [
                "**/__pycache__/**",
                "**/node_modules/**",
                "**/.git/**",
                "**/.venv/**",
                "**/venv/**",
                "**/*.pyc",
                "**/*.pyo",
                "**/build/**",
                "**/dist/**"
            ]
        }
    )
    
    async with ProjectIndexer(config=config) as indexer:
        project = await indexer.create_project(
            name="LeanVibe Agent Hive 2.0",
            root_path=str(Path.cwd()),
            description="Multi-Agent Orchestration System for Autonomous Software Development",
            git_repository_url="https://github.com/leanvibe/agent-hive.git",
            git_branch="main",
            configuration=config.to_dict()
        )
        
        print(f"✅ Created project index: {project.id}")
        
        # Trigger initial analysis
        result = await indexer.analyze_project(
            project_id=str(project.id),
            analysis_type="full_analysis"
        )
        
        print(f"✅ Analysis completed: {result.files_processed} files, {result.dependencies_found} dependencies")
        return project.id

# Run the setup
if __name__ == "__main__":
    project_id = asyncio.run(create_bee_hive_index())
    print(f"Bee-Hive project indexed with ID: {project_id}")
```

## 🔧 Usage Examples

### Via API (curl)

```bash
# Create project index
curl -X POST "http://localhost:8000/api/project-index/create" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "LeanVibe Agent Hive 2.0",
    "root_path": "/path/to/bee-hive",
    "description": "Multi-Agent System",
    "git_repository_url": "https://github.com/leanvibe/agent-hive.git",
    "git_branch": "main",
    "file_patterns": {
      "include": ["**/*.py", "**/*.js", "**/*.ts"]
    }
  }'

# Get project info
curl -X GET "http://localhost:8000/api/project-index/{project_id}"

# Get dependency graph
curl -X GET "http://localhost:8000/api/project-index/{project_id}/dependencies?format=graph"

# Trigger analysis
curl -X POST "http://localhost:8000/api/project-index/{project_id}/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_type": "full",
    "force": false
  }'
```

### Via Python SDK

```python
from app.project_index.core import ProjectIndexer
from app.project_index.models import ProjectIndexConfig

# Initialize indexer
config = ProjectIndexConfig(project_name="my-project")
async with ProjectIndexer(config=config) as indexer:
    
    # Create project
    project = await indexer.create_project(
        name="My Project",
        root_path="/path/to/project"
    )
    
    # Analyze project
    result = await indexer.analyze_project(project.id)
    print(f"Analyzed {result.files_processed} files")
    
    # Get statistics
    stats = await indexer.get_analysis_statistics()
    print(f"Total dependencies: {stats['dependencies_found']}")
```

### WebSocket Real-time Updates

```javascript
// Connect to WebSocket for real-time updates
const ws = new WebSocket('ws://localhost:8000/api/project-index/ws?token=your-auth-token');

ws.onopen = () => {
    // Subscribe to analysis progress
    ws.send(JSON.stringify({
        action: 'subscribe',
        event_types: ['analysis_progress', 'file_change', 'dependency_changed'],
        project_id: 'your-project-id'
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'analysis_progress':
            console.log(`Analysis progress: ${data.progress_percentage}%`);
            break;
        case 'file_change':
            console.log(`File changed: ${data.file_path}`);
            break;
        case 'dependency_changed':
            console.log(`Dependency updated: ${data.dependency_details.target_name}`);
            break;
    }
};
```

## 📊 Monitoring & Validation

### Health Check

```bash
# Test Project Index API health
curl -X GET "http://localhost:8000/api/project-index/health"

# Check WebSocket stats
curl -X GET "http://localhost:8000/api/project-index/ws/stats"
```

### Performance Monitoring

```python
# Get performance statistics
async def check_performance():
    async with ProjectIndexer() as indexer:
        stats = await indexer.get_analysis_statistics()
        
        print(f"Files processed: {stats['files_processed']}")
        print(f"Analysis time: {stats['analysis_time']:.2f}s")
        print(f"Cache hit rate: {stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']):.2%}")
        print(f"Events published: {stats['events_published']}")
```

### Database Verification

```sql
-- Check project index tables
SELECT name, status, file_count, dependency_count, created_at 
FROM project_indexes 
ORDER BY created_at DESC;

-- Check file analysis
SELECT file_type, language, COUNT(*) as count
FROM file_entries 
GROUP BY file_type, language
ORDER BY count DESC;

-- Check dependency relationships
SELECT dependency_type, is_external, COUNT(*) as count
FROM dependency_relationships
GROUP BY dependency_type, is_external
ORDER BY count DESC;
```

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Error**
   ```bash
   # Check PostgreSQL is running
   pg_isready -h localhost -p 5434
   
   # Check database exists
   psql -h localhost -p 5434 -U leanvibe_user -l
   ```

2. **Redis Connection Error**
   ```bash
   # Check Redis is running
   redis-cli -p 6381 ping
   
   # Should return: PONG
   ```

3. **Import Errors**
   ```bash
   # Verify all dependencies installed
   python -c "from app.project_index.core import ProjectIndexer; print('✅ Import successful')"
   ```

4. **Analysis Performance Issues**
   - Reduce `analysis_batch_size` in config
   - Increase `max_concurrent_analyses` (if you have more CPU cores)
   - Enable caching: `cache_enabled=True`
   - Use incremental analysis: `incremental_updates=True`

### Performance Tuning

```python
# Optimized configuration for large projects
config = ProjectIndexConfig(
    analysis_batch_size=25,  # Reduce batch size
    max_concurrent_analyses=3,  # Limit concurrency
    cache_enabled=True,
    incremental_updates=True,
    cache_config={
        "max_memory_mb": 500,
        "enable_compression": True,
        "compression_threshold": 1024
    }
)
```

## ✅ Success Criteria

The Project Index is successfully enabled when:

- ✅ Database tables created and accessible
- ✅ API endpoints responding correctly
- ✅ WebSocket connections working
- ✅ Project can be created and analyzed
- ✅ Files and dependencies are extracted
- ✅ Real-time file monitoring active
- ✅ Performance meets requirements (<2s analysis for typical files)

## 🎯 Next Steps

After enablement:

1. **Integration with Agents**: Connect Project Index to agent orchestration for intelligent context
2. **Dashboard Integration**: Add Project Index widgets to the admin dashboard
3. **CI/CD Integration**: Automatic analysis on code changes
4. **Multi-Project Support**: Index multiple repositories
5. **Advanced Analytics**: Dependency visualization and code quality metrics

## 📚 Additional Resources

- **API Documentation**: `/docs` endpoint when server is running
- **WebSocket Events**: See `app/project_index/websocket_events.py` for event types
- **Configuration Options**: See `app/project_index/models.py` for all config options
- **Performance Tuning**: See `app/project_index/cache.py` for cache configurations