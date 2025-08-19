# LeanVibe Agent Hive 2.0 - Smoke Test Suite

Comprehensive smoke test suite to validate that the LeanVibe Agent Hive system works correctly end-to-end. These tests are designed to run quickly (<30 seconds) and reliably validate critical functionality.

## Overview

The smoke test suite covers:

- **Core Functionality**: Essential system operations
- **API Endpoints**: All major REST endpoints
- **SimpleOrchestrator**: Agent management and task delegation
- **Database Connectivity**: PostgreSQL/SQLite operations
- **Redis Connectivity**: Caching and messaging
- **Performance**: <100ms response time validation
- **Integration**: Cross-component functionality
- **Error Handling**: Graceful degradation scenarios

## Quick Start

### Run All Tests
```bash
# Use the test runner script (recommended)
./run-smoke-tests.sh

# Or run directly with pytest
python -m pytest tests/smoke/ -c pytest-smoke.ini -v
```

### Run Specific Test Categories
```bash
# Core functionality only
./run-smoke-tests.sh core

# Performance tests only
./run-smoke-tests.sh performance

# Integration tests only
./run-smoke-tests.sh integration

# Run in Docker environment
./run-smoke-tests.sh docker
```

### Run with Pytest Markers
```bash
# Fast tests only (<100ms)
python -m pytest tests/smoke/ -m fast -v

# API endpoint tests
python -m pytest tests/smoke/ -m api -v

# Orchestrator tests
python -m pytest tests/smoke/ -m orchestrator -v

# Database tests
python -m pytest tests/smoke/ -m database -v
```

## Test Structure

### Test Files

- **`test_core_functionality.py`** - Essential system operations
  - API health checks
  - Database connectivity
  - Redis operations
  - SimpleOrchestrator basic operations
  - Import resolution validation

- **`test_api_endpoints.py`** - Comprehensive API testing
  - Authentication endpoints
  - Agent management APIs
  - Task management APIs
  - Workflow APIs
  - Observability endpoints
  - Response time validation

- **`test_integration_smoke.py`** - Cross-component integration
  - Database + Redis integration
  - Orchestrator + API integration
  - WebSocket integration
  - Error handling integration
  - Performance under load

- **`test_performance_smoke.py`** - Performance validation
  - <100ms response time targets
  - Concurrency handling
  - Memory efficiency
  - Performance regression detection

### Configuration Files

- **`pytest-smoke.ini`** - Pytest configuration optimized for smoke tests
- **`conftest.py`** - Test fixtures and helpers
- **`docker-compose.smoke-tests.yml`** - Docker test environment

## Performance Targets

The smoke tests validate Epic 1 performance requirements:

| Operation | Target | Limit |
|-----------|--------|---------|
| Health endpoint | <100ms | <200ms |
| Status endpoint | <100ms | <200ms |
| Orchestrator operations | <100ms | <300ms |
| Database queries | <10ms | <50ms |
| API endpoints | <200ms | <1000ms |
| Agent lifecycle | <300ms | <500ms |

## Test Environment Setup

### Prerequisites

```bash
# Install Python dependencies
pip install pytest pytest-asyncio httpx fastapi

# Or install all dev dependencies
pip install -r requirements-test.txt
```

### Environment Variables

The tests automatically configure:

```bash
TESTING=true
DEBUG=true
LOG_LEVEL=ERROR
SKIP_STARTUP_INIT=true
DATABASE_URL=sqlite+aiosqlite:///:memory:
REDIS_URL=redis://localhost:6379/1
PYTHONHASHSEED=0
```

### Docker Environment

Run tests in isolated Docker environment:

```bash
# Build and run smoke tests
docker-compose -f docker-compose.smoke-tests.yml up --build

# Clean up
docker-compose -f docker-compose.smoke-tests.yml down -v
```

## Test Categories and Markers

### Markers

- `@pytest.mark.smoke` - Core smoke tests
- `@pytest.mark.performance` - Performance validation
- `@pytest.mark.integration` - Cross-component tests
- `@pytest.mark.api` - API endpoint tests
- `@pytest.mark.database` - Database operations
- `@pytest.mark.redis` - Redis operations
- `@pytest.mark.orchestrator` - SimpleOrchestrator tests
- `@pytest.mark.fast` - Tests completing <100ms
- `@pytest.mark.slow` - Tests taking >1 second

### Running Specific Categories

```bash
# Run only fast tests
python -m pytest tests/smoke/ -m "fast and not slow" -v

# Run performance tests excluding slow ones
python -m pytest tests/smoke/ -m "performance and not slow" -v

# Run integration tests
python -m pytest tests/smoke/ -m integration -v
```

## Continuous Integration

### GitHub Actions Integration

Add to `.github/workflows/smoke-tests.yml`:

```yaml
name: Smoke Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  smoke-tests:
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
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements-test.txt
    
    - name: Run smoke tests
      run: |
        ./run-smoke-tests.sh all
    
    - name: Upload test results
      if: always()
      uses: actions/upload-artifact@v3
      with:
        name: smoke-test-results
        path: reports/smoke-test-results.xml
```

## Troubleshooting

### Common Issues

#### Test Timeouts
```bash
# Increase timeout (default: 30s per test)
export PYTEST_TIMEOUT=60
python -m pytest tests/smoke/ -c pytest-smoke.ini -v
```

#### Database Connection Issues
```bash
# Use in-memory SQLite (default for tests)
export DATABASE_URL="sqlite+aiosqlite:///:memory:"

# Or use test PostgreSQL
export DATABASE_URL="postgresql+asyncpg://test:test@localhost:5432/test_db"
```

#### Redis Connection Issues
```bash
# Skip Redis-dependent tests
python -m pytest tests/smoke/ -m "not redis" -v

# Use different Redis database
export REDIS_URL="redis://localhost:6379/15"
```

#### Performance Test Failures
```bash
# Run on dedicated machine for accurate timing
# Exclude performance tests if system is under load
python -m pytest tests/smoke/ -m "not performance" -v
```

### Debug Mode

```bash
# Enable debug output
export DEBUG=true
export LOG_LEVEL=DEBUG
python -m pytest tests/smoke/ -c pytest-smoke.ini -v -s
```

### Verbose Output

```bash
# Maximum verbosity
python -m pytest tests/smoke/ -c pytest-smoke.ini -vvv --tb=long
```

## Test Development Guidelines

### Writing New Smoke Tests

1. **Keep tests fast** - Target <100ms per test
2. **Test critical paths** - Focus on essential functionality
3. **Use appropriate markers** - Mark tests with relevant categories
4. **Include performance assertions** - Validate response times
5. **Handle degraded states** - Allow for test environment limitations

### Example Test Structure

```python
import pytest
import time

class TestNewFeature:
    """Test new feature smoke tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.api
    async def test_new_endpoint_response_time(self, async_test_client):
        """New endpoint should respond within 100ms."""
        start_time = time.time()
        response = await async_test_client.get("/api/new-feature")
        response_time = (time.time() - start_time) * 1000
        
        assert response_time < 100, f"Response time {response_time:.2f}ms exceeds 100ms target"
        assert response.status_code in [200, 401]  # Allow auth requirement
        
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
```

### Performance Test Guidelines

1. **Use statistical analysis** - Calculate mean, max, percentiles
2. **Test multiple iterations** - Get reliable measurements
3. **Account for test environment** - Allow reasonable margins
4. **Test concurrency** - Validate performance under load
5. **Check for regression** - Compare before/after performance

## Reporting and Metrics

### Test Reports

Generate JUnit XML reports:

```bash
# Generate report
./run-smoke-tests.sh report

# View report location
ls reports/smoke-test-results.xml
```

### Performance Metrics

The tests collect and validate:

- Response times (mean, max, P95)
- Concurrency handling
- Memory usage patterns
- Error rates
- System stability

### Success Criteria

✅ **All critical paths functional**
✅ **Performance targets met**
✅ **Integration points working**
✅ **Error handling graceful**
✅ **System stable under load**

## Integration with Development Workflow

### Pre-commit Hook

```bash
# Add to .git/hooks/pre-commit
#!/bin/bash
echo "Running smoke tests..."
if ./run-smoke-tests.sh core; then
    echo "Smoke tests passed!"
else
    echo "Smoke tests failed. Please fix before committing."
    exit 1
fi
```

### Development Cycle

1. **Make changes** to code
2. **Run smoke tests** - `./run-smoke-tests.sh`
3. **Fix any failures** - Critical paths must work
4. **Commit changes** - With confidence system works
5. **CI runs full suite** - Including smoke tests

## Maintenance

### Regular Tasks

- **Review performance trends** - Watch for degradation
- **Update test timeouts** - As system evolves
- **Add new critical paths** - For new features
- **Remove obsolete tests** - Clean up deprecated functionality
- **Update documentation** - Keep README current

### Performance Tuning

If smoke tests become slow:

1. **Profile test execution** - Find bottlenecks
2. **Optimize fixtures** - Reduce setup/teardown time
3. **Use mocking** - For external dependencies
4. **Parallelize tests** - Run independent tests concurrently
5. **Split test categories** - Run subsets as needed

---

**🎯 Goal**: Validate system readiness in <30 seconds with high confidence

**📈 Success**: Green smoke tests = System ready for production use
