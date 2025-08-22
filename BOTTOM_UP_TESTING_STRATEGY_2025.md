# 🧪 LeanVibe Agent Hive 2.0 - Bottom-Up Testing Strategy

**Date**: August 22, 2025  
**Version**: 2.0.0  
**Strategy**: Component Isolation → Integration → Contracts → System Validation  
**Objective**: Establish reliable, scalable testing infrastructure from ground up

---

## 📋 Executive Summary

This bottom-up testing strategy addresses the critical testing infrastructure breakdown identified in the comprehensive audit. With 361+ test files present but pytest execution blocked, we need a systematic approach to rebuild testing reliability while maintaining development velocity.

**Strategy**: Start with isolated component testing, progressively integrate systems, validate contracts, and culminate with full system validation.

**Success Criteria**:
- 95% test suite reliability
- <5 minute feedback loops
- Automated regression detection
- Zero-false-positive quality gates

---

## 🏗️ Testing Architecture Foundation

### Current State Assessment

**Assets Available**:
- 361+ test files across multiple categories
- Multiple testing frameworks: pytest, playwright, locust  
- Comprehensive test coverage areas
- Existing CI/CD pipeline infrastructure

**Critical Issues**:
- Pytest configuration conflicts preventing execution
- Import resolution problems
- Environment variable conflicts  
- Complex dependency chains

**Testing Framework Stack**:
- **Unit Testing**: pytest with async support
- **Integration Testing**: pytest with database fixtures
- **Contract Testing**: Custom framework for API/WebSocket contracts
- **E2E Testing**: Playwright for browser automation
- **Performance Testing**: Locust for load testing
- **Security Testing**: bandit, safety, semgrep

---

## 🎯 Phase 1: Component Isolation Testing

### 1.1 Test Infrastructure Recovery

**Priority**: CRITICAL (Days 1-2)

**Objectives**:
- Restore pytest execution capability
- Establish isolated test environments
- Create reliable test fixtures

**Action Items**:

1. **Configuration Consolidation**
   ```bash
   # Audit existing configurations
   find tests/ -name "conftest.py" -exec echo "=== {} ===" \; -exec head -20 {} \;
   find . -name "pytest.ini" -o -name "pyproject.toml" -o -name "setup.cfg"
   
   # Create single authoritative conftest.py
   mv tests/conftest.py tests/conftest_backup.py
   mv tests/conftest_enhanced.py tests/conftest_enhanced_backup.py
   mv tests/conftest_fixed.py tests/conftest_fixed_backup.py
   ```

2. **Environment Isolation**
   ```bash
   # Create isolated test environment
   python -m venv test-env
   source test-env/bin/activate
   pip install -r requirements-test.txt
   ```

3. **Import Path Resolution**
   ```python
   # tests/conftest.py - Simplified configuration
   import os
   import sys
   from pathlib import Path
   
   # Add project root to Python path
   project_root = Path(__file__).parent.parent
   sys.path.insert(0, str(project_root))
   
   # Basic pytest configuration
   pytest_plugins = [
       "pytest_asyncio",
   ]
   ```

**Success Criteria**:
- `pytest --version` executes without errors
- Basic smoke test passes
- Test discovery works correctly

### 1.2 Unit Testing Foundation

**Priority**: HIGH (Days 3-5)

**Strategy**: Test individual components in isolation with mocked dependencies

**Component Categories**:
1. **Core Components**: Configuration, logging, utilities
2. **Business Logic**: Agent management, task execution, coordination
3. **Data Access**: Database models, Redis operations, vector search
4. **API Layer**: Request/response handling, validation, serialization

**Testing Approach**:
```python
# Example: Component isolation test
import pytest
from unittest.mock import Mock, patch
from app.core.simple_orchestrator import SimpleOrchestrator

@pytest.fixture
def mock_redis():
    return Mock()

@pytest.fixture  
def mock_database():
    return Mock()

@pytest.fixture
def isolated_orchestrator(mock_redis, mock_database):
    with patch('app.core.redis.get_redis', return_value=mock_redis):
        with patch('app.core.database.get_database', return_value=mock_database):
            orchestrator = SimpleOrchestrator()
            yield orchestrator
```

**Coverage Targets**:
- Core utilities: 90%
- Business logic: 85%
- API handlers: 80%
- Data access: 75%

### 1.3 Component Test Categories

**Core System Tests** (`tests/unit/core/`):
- Configuration management
- Logging system
- Error handling
- Security components
- Performance monitoring

**Agent System Tests** (`tests/unit/agents/`):
- Agent lifecycle management
- Agent communication
- Agent persona system
- Agent registry operations

**API Layer Tests** (`tests/unit/api/`):
- Request validation
- Response serialization
- Error handling
- Authentication/authorization
- Rate limiting

**Data Layer Tests** (`tests/unit/data/`):
- Database models
- Redis operations
- Vector search
- Caching mechanisms

---

## 🔗 Phase 2: Integration Testing

### 2.1 Service Integration Testing

**Priority**: HIGH (Days 6-10)

**Strategy**: Test component interactions with real dependencies in controlled environments

**Integration Categories**:

1. **Database Integration**
   ```python
   # tests/integration/database/
   @pytest.fixture(scope="module")
   def test_database():
       # Create isolated test database
       db = create_test_database()
       yield db
       cleanup_test_database(db)
   
   def test_agent_crud_operations(test_database):
       # Test full CRUD cycle with real database
       agent = create_agent({"name": "test-agent"})
       assert agent.id is not None
       # ... full CRUD testing
   ```

2. **Redis Integration** 
   ```python
   # tests/integration/redis/
   @pytest.fixture(scope="module")
   def test_redis():
       redis_client = redis.Redis(host='localhost', port=6379, db=15)
       redis_client.flushdb()  # Clean test database
       yield redis_client
       redis_client.flushdb()
   ```

3. **API Integration**
   ```python
   # tests/integration/api/
   @pytest.fixture
   def api_client():
       from fastapi.testclient import TestClient
       from app.main import app
       return TestClient(app)
   
   def test_api_agent_workflow(api_client):
       # Test full API workflow
       response = api_client.post("/api/v2/agents", json={...})
       assert response.status_code == 201
   ```

**Test Environment Setup**:
- Isolated test databases (PostgreSQL + Redis)
- Mock external services (APIs, webhooks)
- Containerized dependencies (Docker)
- Environment variable isolation

### 2.2 WebSocket Integration Testing

**Priority**: MEDIUM (Days 8-12)

**Strategy**: Test real-time communication patterns

```python
# tests/integration/websocket/
import pytest
import websockets
import asyncio
import json

@pytest.mark.asyncio
async def test_websocket_agent_coordination():
    uri = "ws://localhost:8000/api/dashboard/ws/dashboard"
    
    async with websockets.connect(uri) as websocket:
        # Subscribe to agent events
        subscribe_msg = {
            "type": "subscribe",
            "subscriptions": ["agents", "tasks"]
        }
        await websocket.send(json.dumps(subscribe_msg))
        
        # Trigger agent creation via API
        # ... API call to create agent ...
        
        # Verify WebSocket event received
        response = await websocket.recv()
        data = json.loads(response)
        assert data["type"] == "data_response"
        assert "agent_created" in data["data"]
```

**WebSocket Test Categories**:
- Connection management
- Subscription handling  
- Message broadcasting
- Rate limiting
- Error handling

### 2.3 CLI Integration Testing

**Priority**: MEDIUM (Days 10-14)

**Strategy**: Test CLI commands with real system interaction

```python
# tests/integration/cli/
import subprocess
import json
from pathlib import Path

def test_cli_agent_deployment():
    # Test CLI agent deployment
    result = subprocess.run([
        "python", "hive", "agent", "deploy", "backend-developer",
        "--task", "test implementation"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0
    assert "Agent deployed successfully" in result.stdout
    
    # Verify agent created in system
    status_result = subprocess.run([
        "python", "hive", "status", "--json"
    ], capture_output=True, text=True)
    
    status_data = json.loads(status_result.stdout)
    assert status_data["agents"]["total"] > 0
```

---

## 📋 Phase 3: Contract Testing

### 3.1 API Contract Testing

**Priority**: HIGH (Days 12-16)

**Strategy**: Validate API contracts against OpenAPI specifications

**Contract Categories**:

1. **Request/Response Contracts**
   ```python
   # tests/contracts/api/
   from openapi_core import create_spec
   from openapi_core.validation.request import RequestValidator
   from openapi_core.validation.response import ResponseValidator
   
   @pytest.fixture
   def api_spec():
       spec_path = Path("docs/openapi.json")
       return create_spec(spec_path)
   
   def test_agent_creation_contract(api_client, api_spec):
       response = api_client.post("/api/v2/agents", json={
           "name": "test-agent",
           "role": "backend_developer"
       })
       
       # Validate response against OpenAPI spec
       validator = ResponseValidator(api_spec)
       result = validator.validate(response)
       assert result.errors == []
   ```

2. **Database Schema Contracts**
   ```python
   # tests/contracts/database/
   def test_agent_model_schema():
       from app.models.agent import Agent
       from sqlalchemy import inspect
       
       # Verify database schema matches model
       inspector = inspect(Agent.__table__)
       columns = {col.name: col.type for col in inspector.columns}
       
       assert "id" in columns
       assert "name" in columns
       assert "role" in columns
   ```

3. **WebSocket Message Contracts**
   ```python
   # tests/contracts/websocket/
   import jsonschema
   
   def test_websocket_message_contracts():
       schema_path = Path("schemas/ws_messages.schema.json")
       with open(schema_path) as f:
           schema = json.load(f)
       
       # Test various message types
       agent_event = {
           "type": "data_response",
           "data_type": "agent_status",
           "data": {"agent_id": "123", "status": "active"}
       }
       
       jsonschema.validate(agent_event, schema)
   ```

### 3.2 Service Contract Testing

**Priority**: MEDIUM (Days 14-18)

**Strategy**: Test inter-service communication contracts

**Service Boundaries**:
- Orchestrator ↔ Agent communication
- Agent ↔ Task management  
- Dashboard ↔ WebSocket manager
- CLI ↔ API service

```python
# tests/contracts/services/
def test_orchestrator_agent_contract():
    orchestrator = create_test_orchestrator()
    
    # Test agent spawning contract
    agent_config = {
        "role": "backend_developer",
        "specializations": ["python", "fastapi"]
    }
    
    agent_id = orchestrator.spawn_agent(agent_config)
    assert isinstance(agent_id, str)
    assert len(agent_id) > 0
    
    # Test agent communication contract  
    task = {"type": "implementation", "description": "test task"}
    result = orchestrator.assign_task(agent_id, task)
    
    assert "task_id" in result
    assert result["status"] in ["accepted", "queued"]
```

---

## 🎯 Phase 4: System-Level Testing

### 4.1 End-to-End Testing

**Priority**: HIGH (Days 16-20)

**Strategy**: Test complete user workflows from start to finish

**E2E Test Categories**:

1. **Agent Lifecycle Workflows**
   ```python
   # tests/e2e/workflows/
   @pytest.mark.e2e
   async def test_complete_agent_workflow():
       # 1. Deploy agent via CLI
       deploy_result = run_cli_command([
           "hive", "agent", "deploy", "backend-developer"
       ])
       assert deploy_result.success
       
       # 2. Assign task via API
       async with httpx.AsyncClient() as client:
           response = await client.post("/api/v2/tasks", json={
               "agent_id": deploy_result.agent_id,
               "task": {"type": "implementation", "spec": "..."}
           })
           assert response.status_code == 201
       
       # 3. Monitor progress via WebSocket
       async with websockets.connect(ws_uri) as ws:
           await ws.send(json.dumps({
               "type": "subscribe", 
               "subscriptions": ["tasks"]
           }))
           
           # Wait for task completion event
           while True:
               message = await ws.recv()
               data = json.loads(message)
               if data["data"]["task_status"] == "completed":
                   break
       
       # 4. Verify results
       assert task_completed_successfully()
   ```

2. **Dashboard Workflows**
   ```python
   # tests/e2e/dashboard/
   @pytest.mark.playwright
   async def test_dashboard_agent_management(page):
       # Navigate to dashboard
       await page.goto("http://localhost:3001")
       
       # Activate agent team
       await page.click('[data-testid="activate-team"]')
       
       # Verify agents appear in dashboard
       await page.wait_for_selector('[data-testid="agent-card"]')
       agent_cards = await page.query_selector_all('[data-testid="agent-card"]')
       assert len(agent_cards) == 5
       
       # Test task assignment
       await page.fill('[data-testid="task-input"]', "Test implementation")
       await page.click('[data-testid="assign-task"]')
       
       # Verify task appears in kanban board
       await page.wait_for_selector('[data-testid="task-item"]')
   ```

### 4.2 Performance Testing

**Priority**: MEDIUM (Days 18-22)

**Strategy**: Validate system performance under load

**Performance Test Categories**:

1. **API Load Testing**
   ```python
   # tests/performance/api/
   from locust import HttpUser, task, between
   
   class AgentAPIUser(HttpUser):
       wait_time = between(1, 5)
       
       @task(3)
       def get_agents(self):
           self.client.get("/api/v2/agents")
       
       @task(1) 
       def create_agent(self):
           self.client.post("/api/v2/agents", json={
               "name": f"agent-{self.get_unique_id()}",
               "role": "backend_developer"
           })
   ```

2. **WebSocket Load Testing**
   ```python
   # tests/performance/websocket/
   import asyncio
   import websockets
   
   async def websocket_load_test():
       tasks = []
       for i in range(100):  # 100 concurrent connections
           task = asyncio.create_task(websocket_client(i))
           tasks.append(task)
       
       await asyncio.gather(*tasks)
   
   async def websocket_client(client_id):
       uri = "ws://localhost:8000/api/dashboard/ws/dashboard"
       async with websockets.connect(uri) as websocket:
           # Subscribe to all channels
           await websocket.send(json.dumps({
               "type": "subscribe",
               "subscriptions": ["agents", "tasks", "system"]
           }))
           
           # Listen for 60 seconds
           try:
               for _ in range(60):
                   await asyncio.wait_for(websocket.recv(), timeout=1.0)
           except asyncio.TimeoutError:
               pass
   ```

3. **System Resource Testing**
   ```python
   # tests/performance/system/
   import psutil
   import time
   
   def test_system_resource_usage():
       # Baseline measurements
       initial_memory = psutil.virtual_memory().used
       initial_cpu = psutil.cpu_percent()
       
       # Start system under load
       spawn_multiple_agents(25)
       run_concurrent_tasks(100)
       
       # Monitor resource usage
       peak_memory = initial_memory
       peak_cpu = initial_cpu
       
       for _ in range(60):  # Monitor for 60 seconds
           memory = psutil.virtual_memory().used
           cpu = psutil.cpu_percent()
           
           peak_memory = max(peak_memory, memory)
           peak_cpu = max(peak_cpu, cpu)
           
           time.sleep(1)
       
       # Validate resource constraints
       memory_increase = peak_memory - initial_memory
       assert memory_increase < 500_000_000  # <500MB increase
       assert peak_cpu < 80  # <80% CPU usage
   ```

### 4.3 Chaos Testing

**Priority**: LOW (Days 20-24)

**Strategy**: Test system resilience under failure conditions

```python
# tests/chaos/
def test_database_failure_recovery():
    # Start system
    system = start_test_system()
    
    # Verify normal operation
    assert system.health_check()
    
    # Simulate database failure
    simulate_database_failure()
    
    # System should degrade gracefully
    assert system.health_check(include_database=False)
    assert "database_offline" in system.get_status()["warnings"]
    
    # Restore database
    restore_database()
    
    # System should recover
    wait_for_condition(lambda: system.health_check(), timeout=30)

def test_redis_failure_recovery():
    # Similar pattern for Redis failure
    pass

def test_high_load_graceful_degradation():
    # Test system behavior under extreme load
    pass
```

---

## 📊 Testing Infrastructure Setup

### Test Environment Configuration

**Development Environment**:
```yaml
# docker-compose.test.yml
version: '3.8'
services:
  postgres-test:
    image: postgres:13
    environment:
      POSTGRES_DB: hive_test
      POSTGRES_USER: test_user
      POSTGRES_PASSWORD: test_pass
    ports:
      - "15433:5432"  # Different port for isolation
  
  redis-test:
    image: redis:6
    ports:
      - "16380:6379"  # Different port for isolation
    command: redis-server --appendonly yes
```

**Test Configuration**:
```python
# tests/conftest.py (Consolidated)
import pytest
import asyncio
import pytest_asyncio
from fastapi.testclient import TestClient

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_database():
    """Create test database for session."""
    from app.core.database import create_test_database
    db = create_test_database()
    yield db
    cleanup_test_database(db)

@pytest.fixture
def api_client():
    """Create FastAPI test client."""
    from app.main import app
    return TestClient(app)

# Test markers
pytest_markers = [
    "unit: Unit tests",
    "integration: Integration tests", 
    "e2e: End-to-end tests",
    "performance: Performance tests",
    "chaos: Chaos engineering tests",
    "playwright: Browser-based tests"
]
```

### CI/CD Integration

**GitHub Actions Workflow**:
```yaml
# .github/workflows/test.yml
name: Testing Pipeline
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements-test.txt
      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=app --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v1

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: test_pass
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:6
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Run integration tests
        run: |
          pytest tests/integration/ -v

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Node.js
        uses: actions/setup-node@v2
        with:
          node-version: '18'
      - name: Install Playwright
        run: |
          npm install playwright
          npx playwright install
      - name: Run E2E tests
        run: |
          pytest tests/e2e/ -v --playwright
```

---

## 🎯 Success Criteria and Metrics

### Quality Gates

**Unit Testing**:
- Test execution time: <2 minutes
- Code coverage: >85%
- Test reliability: >98%
- Zero flaky tests

**Integration Testing**:
- Test execution time: <10 minutes
- Database tests: 100% pass rate
- API tests: 100% pass rate  
- WebSocket tests: 100% pass rate

**System Testing**:
- E2E test execution: <30 minutes
- Performance benchmarks met: 100%
- Chaos tests: 100% pass rate
- User workflow coverage: >90%

### Performance Benchmarks

**API Performance**:
- Response time p95: <200ms
- Response time p99: <500ms
- Throughput: >1000 req/sec
- Error rate: <0.1%

**WebSocket Performance**:
- Connection setup: <100ms
- Message latency: <50ms
- Concurrent connections: >500
- Message throughput: >10k msg/sec

**System Performance**:
- Memory usage: <500MB under load
- CPU usage: <80% under load
- Database connections: <100
- File descriptors: <1000

---

## 🚀 Implementation Timeline

### Week 1: Foundation Recovery
- **Days 1-2**: Pytest configuration fix, basic test execution
- **Days 3-5**: Unit testing infrastructure, core component tests
- **Days 6-7**: Database and Redis integration tests

### Week 2: Integration Testing
- **Days 8-10**: API integration tests, WebSocket testing
- **Days 11-12**: CLI integration tests
- **Days 13-14**: Service contract testing

### Week 3: System Testing  
- **Days 15-17**: End-to-end workflow testing
- **Days 18-20**: Performance testing and benchmarking
- **Days 21**: Chaos testing implementation

### Week 4: Optimization and Monitoring
- **Days 22-24**: CI/CD pipeline optimization
- **Days 25-26**: Test monitoring and alerting
- **Days 27-28**: Documentation and training

---

## 📋 Conclusion

This bottom-up testing strategy provides a systematic approach to rebuilding testing reliability while maintaining development velocity. By starting with component isolation and progressively integrating systems, we can establish confidence in each layer before building upon it.

**Key Success Factors**:
1. **Configuration Consolidation**: Single source of truth for test configuration
2. **Environment Isolation**: Clean separation between test environments
3. **Progressive Integration**: Build confidence layer by layer
4. **Automated Quality Gates**: Continuous validation and feedback

**Expected Outcomes**:
- 95% test suite reliability within 4 weeks
- <5 minute feedback loops for developers  
- Automated regression detection
- Production-ready quality assurance

The strategy prioritizes immediate infrastructure recovery while building toward comprehensive system validation, ensuring both short-term success and long-term sustainability.

---

**Document Version**: 1.0  
**Last Updated**: August 22, 2025  
**Next Review**: Weekly during implementation  
**Owner**: QA Agent and Infrastructure Agent collaboration