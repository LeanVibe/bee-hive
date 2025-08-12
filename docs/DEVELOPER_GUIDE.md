> NOTE: ARCHIVED. Use `docs/CORE.md` (overview) and `docs/ARCHITECTURE.md` (implementation). For navigation, see `docs/NAV_INDEX.md`.

# Developer Guide

This guide provides comprehensive information for developers working on LeanVibe Agent Hive.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [API Documentation](#api-documentation)
- [Database Schema](#database-schema)
- [Testing Strategy](#testing-strategy)
- [Deployment Guide](#deployment-guide)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

LeanVibe Agent Hive is built with a modern, scalable architecture:

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Mobile PWA    │    │   Vue.js Web    │    │  FastAPI API    │
│   Dashboard     │◄──►│   Dashboard     │◄──►│   Gateway       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                         ┌─────────────────────────────┼─────────────────────────────┐
                         │                             │                             │
                ┌─────────▼──────┐          ┌─────────▼──────┐          ┌─────────▼──────┐
                │  Agent         │          │  Context       │          │  Communication │
                │  Orchestrator  │          │  Engine        │          │  Bus           │
                └────────────────┘          └────────────────┘          └────────────────┘
                         │                             │                             │
                ┌─────────▼──────┐          ┌─────────▼──────┐          ┌─────────▼──────┐
                │  PostgreSQL    │          │  pgvector      │          │  Redis         │
                │  Database      │          │  Embeddings    │          │  Streams       │
                └────────────────┘          └────────────────┘          └────────────────┘
```

### Technology Stack

**Backend:**
- **FastAPI**: Async web framework with automatic OpenAPI docs
- **PostgreSQL 15+**: Primary database with pgvector for embeddings
- **Redis 7+**: Message bus using Redis Streams
- **SQLAlchemy 2.0**: Async ORM with declarative models
- **Alembic**: Database migrations
- **Pydantic**: Data validation and serialization

**Frontend:**
- **Vue.js 3**: Reactive web framework with Composition API
- **Lit**: Web components for the mobile PWA
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **Vite**: Fast build tool and dev server

**Infrastructure:**
- **Docker**: Containerization
- **Prometheus**: Metrics collection
- **Grafana**: Dashboards and alerting
- **Nginx**: Reverse proxy (production)

## Getting Started

For the canonical, tested steps, see `docs/GETTING_STARTED.md`. This guide focuses on deeper reference material.

### Prerequisites

- **Python 3.11+** with pip
- **Node.js 18+** with npm
- **Docker** and **Docker Compose**
- **Git** with SSH keys

### Quick Setup (summary)

```bash
# 1. Clone and enter directory
git clone https://github.com/LeanVibe/bee-hive.git
cd bee-hive

# 2. Start infrastructure
docker-compose up -d postgres redis

# 3. Set up Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .

# 4. Configure environment
cp .env.example .env
# Edit .env with your configuration

# 5. Initialize database
alembic upgrade head

# 6. Start backend
uvicorn app.main:app --reload

# 7. Start frontend (in separate terminals)
cd frontend && npm install && npm run dev
cd mobile-pwa && npm install && npm run dev
```

### Verification

- Backend API: http://localhost:8000/docs
- Web Dashboard: http://localhost:3000
- Mobile PWA: http://localhost:3001
- Health Check: http://localhost:8000/health

## Development Workflow

### Code Organization

```
app/
├── api/v1/              # API endpoints
│   ├── agents.py        # Agent management
│   ├── tasks.py         # Task operations
│   ├── websocket.py     # Real-time communication
│   └── ...
├── core/                # Core business logic
│   ├── orchestrator.py  # Agent orchestration
│   ├── communication.py # Message bus
│   ├── context_manager.py # Context engine
│   └── ...
├── models/              # Database models
│   ├── agent.py         # Agent entity
│   ├── task.py          # Task entity
│   └── ...
├── schemas/             # API schemas
│   ├── agent.py         # Agent DTOs
│   ├── task.py          # Task DTOs
│   └── ...
└── observability/       # Monitoring
    ├── hooks.py         # Event hooks
    ├── middleware.py    # Request middleware
    └── ...
```

### Development Commands

```bash
# Backend development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Database operations
alembic revision --autogenerate -m "Description"
alembic upgrade head
alembic downgrade -1

# Testing
pytest -v --cov=app
pytest tests/unit/
pytest tests/integration/

# Code quality
black app/ tests/
ruff check app/ tests/
mypy app/

# Frontend development
cd frontend
npm run dev          # Development server
npm run build        # Production build
npm run test         # Unit tests
npm run lint         # Linting
npm run type-check   # TypeScript check

cd mobile-pwa
npm run dev          # Development server
npm run build        # Production build
npm run test         # Unit tests
npm run test:e2e     # E2E tests
npm run lighthouse   # PWA audit
```

### Git Workflow

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Make changes** with tests
3. **Commit with conventional commits**: `feat: add new feature`
4. **Push and create PR**: Use the provided PR template
5. **Code review** and address feedback
6. **Merge** after approval and passing CI

### Conventional Commits

We use conventional commits for clear change history:

```bash
feat: add user authentication
fix: resolve memory leak in agent orchestrator
docs: update API documentation
refactor: simplify context manager logic
test: add integration tests for WebSocket
chore: update dependencies
```

## API Documentation

### Authentication

All API endpoints require JWT authentication except for health checks and public endpoints.

```bash
# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}'

# Use token
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/agents/
```

### Core Endpoints

**Agents:**
- `GET /api/v1/agents/` - List all agents
- `POST /api/v1/agents/` - Create new agent
- `GET /api/v1/agents/{agent_id}` - Get specific agent
- `PUT /api/v1/agents/{agent_id}` - Update agent
- `DELETE /api/v1/agents/{agent_id}` - Delete agent

**Tasks:**
- `GET /api/v1/tasks/` - List tasks with filtering
- `POST /api/v1/tasks/` - Create new task
- `PUT /api/v1/tasks/{task_id}` - Update task
- `POST /api/v1/tasks/{task_id}/assign/{agent_id}` - Assign task

**WebSocket:**
- `ws://localhost:8000/ws/observability` - Real-time events
- `ws://localhost:8000/ws/agents/{agent_id}` - Agent-specific events

### Request/Response Examples

**Create Agent:**
```json
POST /api/v1/agents/
{
  "name": "backend-developer",
  "role": "developer",
  "capabilities": ["python", "fastapi", "postgresql"],
  "max_concurrent_tasks": 3
}
```

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "name": "backend-developer",
  "role": "developer",
  "status": "active",
  "capabilities": ["python", "fastapi", "postgresql"],
  "max_concurrent_tasks": 3,
  "created_at": "2024-01-01T00:00:00Z"
}
```

## Database Schema

### Core Tables

**agents:**
```sql
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL UNIQUE,
    role VARCHAR(100) NOT NULL,
    status agent_status NOT NULL DEFAULT 'active',
    capabilities JSONB,
    max_concurrent_tasks INTEGER DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**tasks:**
```sql
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(500) NOT NULL,
    description TEXT,
    task_type task_type NOT NULL,
    status task_status NOT NULL DEFAULT 'pending',
    priority task_priority NOT NULL DEFAULT 'medium',
    assigned_agent_id UUID REFERENCES agents(id),
    estimated_effort INTEGER,
    actual_effort INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**contexts:**
```sql
CREATE TABLE contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agents(id),
    session_id UUID,
    content_hash VARCHAR(64) UNIQUE,
    content TEXT NOT NULL,
    embedding vector(1536),
    context_type context_type NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Migrations

Database migrations are managed with Alembic:

```bash
# Create new migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# View migration history
alembic history
```

## Testing Strategy

### Test Structure

```
tests/
├── unit/                # Unit tests
│   ├── test_agents.py
│   ├── test_tasks.py
│   └── ...
├── integration/         # Integration tests
│   ├── test_api.py
│   ├── test_database.py
│   └── ...
├── e2e/                # End-to-end tests
│   ├── test_workflows.py
│   └── ...
├── conftest.py         # Pytest configuration
└── factories.py        # Test data factories
```

### Running Tests

```bash
# All tests with coverage
pytest -v --cov=app --cov-report=html

# Specific test types
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# Test specific module
pytest tests/unit/test_agents.py

# Test with pattern
pytest -k "test_agent"

# Run with debugging
pytest -v -s tests/unit/test_agents.py::test_create_agent
```

### Writing Tests

**Unit Test Example:**
```python
import pytest
from app.core.orchestrator import AgentOrchestrator
from app.models.agent import Agent, AgentStatus

@pytest.fixture
def orchestrator():
    return AgentOrchestrator()

@pytest.mark.asyncio
async def test_assign_task_to_agent(orchestrator):
    # Arrange
    agent = Agent(
        name="test-agent",
        role="developer",
        status=AgentStatus.ACTIVE
    )
    task_data = {"title": "Test task", "priority": "high"}
    
    # Act
    result = await orchestrator.assign_task(agent.id, task_data)
    
    # Assert
    assert result.assigned_agent_id == agent.id
    assert result.status == "assigned"
```

**Integration Test Example:**
```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_create_agent_api(client: AsyncClient, auth_headers):
    # Arrange
    agent_data = {
        "name": "integration-test-agent",
        "role": "developer",
        "capabilities": ["python", "testing"]
    }
    
    # Act
    response = await client.post(
        "/api/v1/agents/",
        json=agent_data,
        headers=auth_headers
    )
    
    # Assert
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == agent_data["name"]
    assert data["role"] == agent_data["role"]
```

### Test Configuration

**conftest.py:**
```python
import pytest
import asyncio
from httpx import AsyncClient
from app.main import app
from app.core.database import get_session_dependency

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
async def auth_headers():
    # Return headers with valid JWT token
    return {"Authorization": "Bearer test-token"}
```

## Deployment Guide

### Development Environment

Use Docker Compose for local development:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up --build
```

### Production Deployment

**1. Environment Setup:**
```bash
# Create production environment file
cp .env.example .env.production

# Update with production values
DEBUG=false
DATABASE_URL=postgresql://user:pass@prod-db:5432/agent_hive
REDIS_URL=redis://prod-redis:6379/0
JWT_SECRET_KEY=<secure-random-key>
```

**2. Build Production Images:**
```bash
# Build images
docker-compose -f docker-compose.prod.yml build

# Push to registry
docker tag bee-hive:latest your-registry/bee-hive:latest
docker push your-registry/bee-hive:latest
```

**3. Deploy:**
```bash
# Deploy with production config
docker-compose -f docker-compose.prod.yml up -d

# Run migrations
docker-compose -f docker-compose.prod.yml exec api alembic upgrade head
```

### Infrastructure Requirements

**Minimum Production Requirements:**
- **CPU**: 2 vCPUs
- **Memory**: 4 GB RAM
- **Storage**: 20 GB SSD
- **Network**: 1 Gbps

**Recommended Production Setup:**
- **CPU**: 4 vCPUs
- **Memory**: 8 GB RAM
- **Storage**: 100 GB SSD
- **Database**: Managed PostgreSQL with backups
- **Cache**: Managed Redis with persistence
- **Load Balancer**: Nginx or cloud LB
- **Monitoring**: Prometheus + Grafana
- **SSL**: Let's Encrypt or managed certificates

### Health Checks

The application provides comprehensive health checks:

```bash
# Basic health
curl http://localhost:8000/health

# Detailed health with dependencies
curl http://localhost:8000/health/detailed
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "2.0.0",
  "dependencies": {
    "database": "healthy",
    "redis": "healthy",
    "disk_space": "healthy"
  }
}
```

## Troubleshooting

### Common Issues

**1. Database Connection Issues:**
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Test connection
psql postgresql://postgres:password@localhost:5432/agent_hive
```

**2. Redis Connection Issues:**
```bash
# Check if Redis is running
docker-compose ps redis

# Test connection
redis-cli -h localhost -p 6379 ping
```

**3. Migration Issues:**
```bash
# Check current migration status
alembic current

# View migration history
alembic history

# Reset database (development only)
alembic downgrade base
alembic upgrade head
```

**4. WebSocket Connection Issues:**
```bash
# Test WebSocket connection
wscat -c ws://localhost:8000/ws/observability

# Check browser console for errors
# Verify authentication token
```

### Performance Tuning

**Database:**
```sql
-- Monitor slow queries
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC LIMIT 10;

-- Check connection pool usage
SELECT count(*) as active_connections 
FROM pg_stat_activity 
WHERE state = 'active';
```

**Redis:**
```bash
# Monitor Redis performance
redis-cli info memory
redis-cli info replication

# Monitor stream usage
redis-cli xinfo stream observability_events
```

**Application:**
```bash
# Check process memory usage
ps aux | grep uvicorn

# Monitor API response times
curl -w "%{time_total}\\n" -s http://localhost:8000/api/v1/agents/
```

### Debugging

**Backend Debugging:**
```python
# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debugger
import pdb; pdb.set_trace()

# Async debugging
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

**Frontend Debugging:**
```javascript
// Enable debug mode
localStorage.setItem('debug', 'true')

// Check service worker
navigator.serviceWorker.getRegistrations()

// Monitor WebSocket
window.wsDebug = true
```

### Log Analysis

**Backend Logs:**
```bash
# Follow logs
docker-compose logs -f api

# Search for errors
docker-compose logs api | grep ERROR

# Filter by correlation ID
docker-compose logs api | grep "correlation_id=abc123"
```

**Frontend Logs:**
```bash
# Check browser console
# Enable verbose logging in development

# Service worker logs
# Open DevTools > Application > Service Workers
```

For additional help:
- Check [GitHub Issues](https://github.com/LeanVibe/bee-hive/issues)
- Join our [Discord Community](https://discord.gg/leanvibe)
- Email: support@leanvibe.com