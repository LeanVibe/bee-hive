# Quick Reference Guide for Claude Code
## LeanVibe Agent Hive 2.0

### 🚀 System Startup & Validation

```bash
# Quick start (development)
python start_hive.py

# Fast startup for testing
./start-fast.sh

# Full system validation
python scripts/validate_system_integration.py

# Project index server
python project_index_server.py

# Run comprehensive tests
pytest tests/ --cov=app
```

### 🔍 Finding Code Patterns

#### Agent-Related Code
```bash
# Find all orchestrator implementations
find app/core -name "*orchestrator*.py"

# Find agent coordination logic
find app -name "*agent*" -name "*.py"

# Find communication protocols
grep -r "communication" app/core/
```

#### Project Index System
```bash
# Core indexing files
ls app/project_index/

# API endpoints
ls app/api/project_index*

# Database models
cat app/models/project_index.py
```

#### WebSocket Integration
```bash
# WebSocket handlers
find app -name "*websocket*.py"

# Real-time events
find app -name "*event*.py"
```

### 🗃️ Database Operations

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Check current migration
alembic current

# Project index tables
psql -f create_project_index_tables.sql
```

### 🧪 Testing Patterns

```bash
# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Test specific component
pytest tests/test_orchestrator*.py
pytest tests/test_project_index*.py

# Run with coverage
pytest --cov=app tests/

# Performance testing
python scripts/run_performance_demo.py
```

### 📊 Common File Patterns

| Pattern | Purpose | Location |
|---------|---------|----------|
| `*orchestrator*.py` | Agent orchestration logic | `app/core/` |
| `*project_index*.py` | Project indexing system | `app/project_index/`, `app/api/` |
| `*websocket*.py` | Real-time communication | `app/api/`, `app/project_index/` |
| `*agent*.py` | Agent implementations | `app/agents/`, `app/models/` |
| `*coordination*.py` | Multi-agent coordination | `app/core/`, `app/api/` |
| `*debt*.py` | Technical debt analysis | `app/project_index/` |

### 🔧 Configuration Files

| File | Purpose |
|------|---------|
| `bee-hive-config.json` | Main project configuration |
| `project-index-config.json` | Project index settings |
| `pyproject.toml` | Python project configuration |
| `docker-compose.yml` | Development environment |
| `docker-compose.production.yml` | Production deployment |
| `.claude/project-index.json` | Claude-specific index |

### 🌐 API Endpoints Structure

```python
# REST API Pattern
app/api/
├── main.py                    # FastAPI app setup
├── routes.py                  # Main routes
├── project_index.py          # Project index endpoints
├── agent_coordination.py     # Agent API
├── dashboard_websockets.py   # WebSocket handlers
└── *_endpoints.py            # Specific feature endpoints
```

### 📱 Frontend Structure

```typescript
mobile-pwa/src/
├── components/               # UI components
├── services/                # API clients
├── types/                   # TypeScript types
└── pages/                   # Application pages
```

### 🐳 Docker Commands

```bash
# Development environment
docker-compose up -d

# Production environment
docker-compose -f docker-compose.production.yml up -d

# Enterprise demo
docker-compose -f docker-compose.enterprise-demo.yml up -d

# Check services
docker-compose ps

# View logs
docker-compose logs -f api
```

### 📈 Monitoring & Debugging

```bash
# Check system health
curl http://localhost:8000/health

# WebSocket connection test
python examples/websocket_client_example.py

# Database connection test
python -c "from app.core.database import get_db; print('DB OK')"

# Redis connection test
python -c "from app.core.redis import get_redis; print('Redis OK')"
```

### 🔐 Security & Authentication

```bash
# Generate API key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Check security configurations
cat app/core/security*.py

# Run security tests
pytest tests/security/
```

### 📚 Documentation Patterns

| Document | When to Update |
|----------|----------------|
| `README.md` | Project overview changes |
| `docs/ARCHITECTURE.md` | System design changes |
| `docs/CLAUDE.md` | Claude-specific guidelines |
| `docs/DEVELOPER_GUIDE.md` | Development process changes |
| API docstrings | New/changed endpoints |

### ⚡ Performance Optimization

```bash
# Performance benchmarks
python scripts/benchmark_*.py

# Memory profiling
python -m memory_profiler app/main.py

# Database query analysis
python scripts/analyze_db_performance.py

# Load testing
python scripts/load_testing.py
```

### 🔄 Common Workflows

#### Adding New Agent Type:
1. Create agent class in `app/agents/`
2. Update `app/models/agent.py`
3. Modify orchestrator in `app/core/orchestrator*.py`
4. Add API endpoints in `app/api/agent_coordination.py`
5. Add tests in `tests/test_agents.py`

#### Extending Project Index:
1. Add analyzer to `app/project_index/analyzer.py`
2. Update models in `app/models/project_index.py`
3. Create migration if needed
4. Add API endpoints in `app/api/project_index.py`
5. Update WebSocket events in `app/project_index/websocket_events.py`

#### Creating API Endpoint:
1. Add endpoint to appropriate file in `app/api/`
2. Define request/response schemas in `app/schemas/`
3. Add business logic to `app/core/`
4. Write tests in `tests/test_api*.py`
5. Update API documentation

### 🧮 Useful Code Snippets

#### Database Session Pattern:
```python
from app.core.database import get_db_session

async def my_function():
    async with get_db_session() as session:
        # Your database operations here
        pass
```

#### WebSocket Event Pattern:
```python
from app.api.dashboard_websockets import websocket_manager

await websocket_manager.broadcast({
    "type": "event_type",
    "data": your_data
})
```

#### Agent Orchestration Pattern:
```python
from app.core.orchestrator import UnifiedOrchestrator

orchestrator = UnifiedOrchestrator()
result = await orchestrator.execute_task(task)
```

### 🎯 Key Metrics to Monitor

- Agent response times
- Project index update rates
- WebSocket connection health
- Database query performance
- Memory usage patterns
- Technical debt trends

### 📞 Getting Help

1. Check `docs/` directory for detailed documentation
2. Look at `tests/` for usage examples
3. Review `examples/` for code samples
4. Check logs in `logs/` directory
5. Run validation scripts in `scripts/`