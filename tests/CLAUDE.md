# CLAUDE.md - Testing Infrastructure & Quality Assurance

## 🎯 **Context: Comprehensive Testing Strategy**

You are working in the **testing infrastructure** of LeanVibe Agent Hive 2.0. This directory implements the comprehensive testing pyramid with **135 tests across 6 testing levels** that build confidence from foundation to end-to-end validation.

## 🏗️ **Testing Pyramid Architecture**

### **Epic 2 Focus: Testing Infrastructure Implementation**
```
                    E2E CLI Testing (37 tests) 
                 🔺
               REST API Testing (54 tests)
            🔺
         Component Integration (28 tests) 
      🔺
   Foundation Unit Tests (44 tests)
🔺

Total: 135 Tests | Coverage: 34.67% | Strategy: Bottom-Up Confidence Building
```

### **Test Organization Structure**
```
tests/
├── simple_system/          # Level 1: Foundation Unit Tests
├── unit/                   # Level 2: Component Unit Tests
├── integration/            # Level 3: Component Integration
├── contracts/              # Level 4: Contract Testing
├── api/                    # Level 5: REST API Testing
├── cli/                    # Level 6: CLI Testing
├── e2e-validation/         # Level 7: End-to-End Workflows
├── performance/            # Load & Performance Testing
├── security/               # Security Testing
└── chaos/                  # Chaos Engineering
```

## 🧪 **Testing Standards & Patterns**

### **Foundation Unit Tests** (`simple_system/`)
**Purpose**: Build basic confidence in core system components

```python
# test_foundation_unit_tests.py
import pytest
from app.core.orchestrator import ProductionOrchestrator
from app.models.agent import Agent

class TestFoundationComponents:
    """Foundation-level tests with zero external dependencies"""
    
    def test_core_imports_successful(self):
        """Verify core modules import without errors"""
        import app.core.config
        import app.core.database
        import app.core.orchestrator
        assert True  # If we get here, imports worked
    
    def test_model_instantiation(self):
        """Test Pydantic models create correctly"""
        agent = Agent(
            name="test-agent",
            type="backend-engineer",
            capabilities=["python", "fastapi"]
        )
        assert agent.name == "test-agent"
        assert agent.type == "backend-engineer"
    
    @pytest.mark.asyncio
    async def test_orchestrator_creation(self):
        """Test orchestrator instantiates without external dependencies"""
        from app.core.config import OrchestratorConfig
        config = OrchestratorConfig(max_agents=10, task_timeout=300)
        orchestrator = ProductionOrchestrator(config)
        assert orchestrator.config.max_agents == 10
```

### **Component Integration Tests** (`integration/`)
**Purpose**: Test component interactions with controlled test environments

```python
# test_component_integration.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
import fakeredis

@pytest.fixture
async def test_database():
    """Isolated test database with SQLite in-memory"""
    from sqlalchemy.ext.asyncio import create_async_engine
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    
    # Create tables and yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    await engine.dispose()

@pytest.fixture
def test_redis():
    """Isolated Redis using fakeredis"""
    return fakeredis.FakeStrictRedis(decode_responses=True)

class TestComponentIntegration:
    """Test component interactions with isolated dependencies"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_database_integration(self, test_database):
        """Test orchestrator saves agent state to database"""
        orchestrator = ProductionOrchestrator(database=test_database)
        
        agent_spec = AgentSpec(name="test", type="backend-engineer")
        agent_id = await orchestrator.register_agent(agent_spec)
        
        # Verify agent persisted
        saved_agent = await orchestrator.get_agent(agent_id)
        assert saved_agent.name == "test"
    
    @pytest.mark.asyncio
    async def test_redis_messaging_integration(self, test_redis):
        """Test Redis stream operations with realistic message flow"""
        broker = AgentMessageBroker(test_redis)
        
        # Test message publishing
        message_id = await broker.publish_task(
            stream="agent:tasks",
            data={"task_id": "123", "type": "code_review"}
        )
        assert message_id is not None
        
        # Test message consumption
        messages = await broker.consume_messages("agent:tasks", count=1)
        assert len(messages) == 1
        assert messages[0]["task_id"] == "123"
```

### **Contract Testing** (`contracts/`)
**Purpose**: Validate interfaces between components

```python
# test_orchestrator_contracts.py
import pytest
from pydantic import ValidationError
from app.schemas.agent import AgentCreateSchema, AgentResponseSchema

class TestOrchestratorContracts:
    """Validate contracts between orchestrator and consumers"""
    
    def test_agent_create_schema_validation(self):
        """Test input schema validation"""
        # Valid input
        valid_data = {
            "name": "test-agent",
            "type": "backend-engineer",
            "capabilities": ["python", "fastapi"]
        }
        schema = AgentCreateSchema(**valid_data)
        assert schema.name == "test-agent"
        
        # Invalid input
        with pytest.raises(ValidationError):
            AgentCreateSchema(name="", type="invalid-type")
    
    def test_orchestrator_response_contract(self):
        """Test output schema validation"""
        response_data = {
            "id": "agent-123",
            "name": "test-agent", 
            "type": "backend-engineer",
            "status": "active",
            "created_at": "2025-08-15T10:00:00Z"
        }
        
        response = AgentResponseSchema(**response_data)
        assert response.id == "agent-123"
        assert response.status == "active"
```

### **Performance Testing** (`performance/`)
**Purpose**: Validate system performance under load

```python
# test_performance_benchmarks.py
import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

class TestPerformanceBenchmarks:
    """Validate performance requirements"""
    
    @pytest.mark.asyncio
    async def test_agent_registration_performance(self, production_orchestrator):
        """Test <100ms agent registration requirement"""
        start_time = time.time()
        
        agent_id = await production_orchestrator.register_agent(
            AgentSpec(name="perf-test", type="backend-engineer")
        )
        
        registration_time = (time.time() - start_time) * 1000  # Convert to ms
        assert registration_time < 100, f"Registration took {registration_time}ms, expected <100ms"
    
    @pytest.mark.asyncio
    async def test_concurrent_agent_capacity(self, production_orchestrator):
        """Test 50+ concurrent agents requirement"""
        async def create_agent(index):
            return await production_orchestrator.register_agent(
                AgentSpec(name=f"agent-{index}", type="backend-engineer")
            )
        
        # Create 50 agents concurrently
        tasks = [create_agent(i) for i in range(50)]
        agent_ids = await asyncio.gather(*tasks)
        
        assert len(agent_ids) == 50
        assert all(agent_id is not None for agent_id in agent_ids)
```

## 🎨 **Test Development Guidelines**

### **Isolation Principles**
1. **External Dependencies**: Always mock or use test doubles
2. **Environment Variables**: Control test environment completely
3. **Database State**: Use transactions and proper cleanup
4. **Time Dependencies**: Mock time-based functionality

### **Test Naming Conventions**
```python
def test_[component]_[scenario]_[expected_outcome]:
    """
    Clear test naming that describes:
    - What component is being tested
    - What scenario is being tested
    - What the expected outcome is
    """
    pass

# Examples:
def test_orchestrator_agent_registration_succeeds_with_valid_input():
def test_api_endpoint_returns_404_when_agent_not_found():
def test_websocket_connection_recovers_after_redis_disconnect():
```

### **Fixture Organization**
```python
# conftest.py - Shared fixtures
@pytest.fixture
async def isolated_orchestrator():
    """Production orchestrator with all dependencies mocked"""
    from unittest.mock import AsyncMock
    
    config = OrchestratorConfig(max_agents=10)
    orchestrator = ProductionOrchestrator(config)
    
    # Mock external dependencies
    orchestrator.database = AsyncMock()
    orchestrator.redis = MagicMock()
    orchestrator.message_broker = AsyncMock()
    
    return orchestrator

@pytest.fixture(scope="session")
def event_loop():
    """Provide event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
```

## 📊 **Test Execution & Reporting**

### **Test Categories & Execution Times**
- **Foundation Unit Tests**: <5 seconds total
- **Component Integration**: <10 seconds with setup/teardown  
- **API Tests**: <30 seconds including server startup
- **CLI Tests**: <60 seconds for complete validation
- **Performance Tests**: <120 seconds for load scenarios

### **Coverage Requirements**
- **Core Components**: 85%+ coverage required
- **API Endpoints**: 80%+ coverage required
- **Critical Paths**: 95%+ coverage required
- **Overall Target**: 75%+ project coverage

### **CI Integration**
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  test-foundation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Foundation Tests
        run: |
          pytest tests/simple_system/ -v --cov=app/core
          
  test-integration:
    runs-on: ubuntu-latest
    needs: test-foundation
    steps:
      - uses: actions/checkout@v3
      - name: Run Integration Tests
        run: |
          pytest tests/integration/ -v --cov-append
          
  test-performance:
    runs-on: ubuntu-latest
    needs: test-integration
    steps:
      - uses: actions/checkout@v3
      - name: Run Performance Tests
        run: |
          pytest tests/performance/ -v --benchmark-only
```

## 🚦 **Quality Gates & Automation**

### **Pre-Commit Quality Gates**
- All new tests must pass
- Test coverage must not decrease
- Performance tests must meet benchmarks
- No flaky tests (>98% reliability)

### **Quality Metrics Tracking**
```python
# test_quality_metrics.py
class TestQualityMetrics:
    """Track and validate quality metrics"""
    
    def test_flaky_test_rate_under_threshold(self):
        """Ensure flaky test rate stays below 2%"""
        test_results = get_recent_test_results()
        flaky_rate = calculate_flaky_test_rate(test_results)
        assert flaky_rate < 0.02, f"Flaky test rate {flaky_rate} exceeds 2% threshold"
    
    def test_test_execution_time_reasonable(self):
        """Ensure test suite execution time stays reasonable"""
        execution_time = measure_test_suite_time()
        assert execution_time < 300, f"Test suite took {execution_time}s, expected <5 minutes"
```

## 🎯 **Epic 2 Implementation Status**

### **Current Achievement: 135 Tests Implemented**
- ✅ **Foundation Tests**: 44 tests passing (100% success rate)
- ✅ **Component Integration**: 28 tests with isolation patterns
- ✅ **API Testing**: 54 tests covering 219 endpoints
- ✅ **CLI Testing**: 37 tests for complete user experience
- 🔄 **Performance Testing**: Framework ready, benchmarks in progress
- ❌ **End-to-End Workflows**: Framework established, implementation pending

### **Next Priorities for Epic 2 Completion**
1. **Complete Performance Test Suite**: Load testing for 50+ concurrent agents
2. **Implement E2E Workflow Tests**: Full user journey validation
3. **Enhance Contract Testing**: Cross-component interface validation
4. **Optimize Test Execution**: Parallel execution and faster feedback

## ⚠️ **Critical Testing Considerations**

### **Environment Management**
- Tests must run reliably in CI/CD environment
- Local development environment isolation required
- Database and Redis cleanup after test runs
- Proper async test handling and cleanup

### **Epic Integration Testing**
- **Epic 1**: Orchestrator consolidation requires integration tests
- **Epic 2**: Meta-testing of testing infrastructure itself
- **Epic 3**: Security and performance testing integration
- **Epic 4**: Context engine testing with semantic validation

## ✅ **Success Criteria**

Your work in `/tests/` is successful when:
- **Coverage**: 80%+ test coverage across critical components
- **Reliability**: <2% flaky test rate consistently
- **Performance**: Test suite execution <5 minutes total
- **Quality Gates**: Automated prevention of regressions
- **Epic Support**: Complete testing support for all 4 epics

Focus on **completing Epic 2 testing infrastructure** by implementing missing performance and end-to-end test suites while maintaining the high-quality foundation already established.