# Project Index Testing Guide

This document provides comprehensive guidance for testing the Project Index system, including test organization, execution strategies, and quality assurance processes.

## Table of Contents

1. [Test Architecture](#test-architecture)
2. [Test Categories](#test-categories)
3. [Test Data & Fixtures](#test-data--fixtures)
4. [Running Tests](#running-tests)
5. [Coverage Requirements](#coverage-requirements)
6. [CI/CD Integration](#cicd-integration)
7. [Performance Testing](#performance-testing)
8. [Security Testing](#security-testing)
9. [Troubleshooting](#troubleshooting)

## Test Architecture

The Project Index testing framework follows a comprehensive multi-layered approach:

```
tests/
├── unit/                           # Unit tests (fast, isolated)
│   ├── test_project_index_database_models.py
│   ├── test_project_index_api_comprehensive.py
│   ├── test_project_index_cache.py
│   ├── test_project_index_utils.py
│   ├── test_project_index_graph.py
│   ├── test_project_index_file_monitor.py
│   └── test_project_index_models.py
├── integration/                    # Integration tests (moderate speed)
│   ├── test_project_index_end_to_end_workflows.py
│   ├── test_project_index_cache_integration.py
│   ├── test_project_index_file_monitor_integration.py
│   └── test_project_index_integration.py
├── performance/                    # Performance tests (slower)
│   └── test_project_index_performance_benchmarks.py
├── security/                       # Security tests (validation focus)
│   └── test_project_index_security_comprehensive.py
├── frontend/                       # Frontend/API contract tests
│   └── test_project_index_pwa_components.py
├── fixtures/                       # Test data and utilities
│   └── project_index_test_data.py
└── project_index_conftest.py       # Shared fixtures and configuration
```

## Test Categories

### 1. Unit Tests (`@pytest.mark.unit`)

**Purpose**: Test individual components in isolation
**Speed**: Fast (< 1 second per test)
**Coverage Target**: 95%+

- **Database Models**: Validation, relationships, serialization
- **API Endpoints**: Request/response handling, validation, error cases
- **Core Logic**: Business logic, utilities, algorithms
- **Cache Operations**: Redis interactions, cache policies
- **Graph Operations**: Dependency analysis, graph algorithms

```bash
# Run unit tests only
pytest tests/unit/ -m unit -v
```

### 2. Integration Tests (`@pytest.mark.integration`)

**Purpose**: Test component interactions and workflows
**Speed**: Moderate (5-30 seconds per test)
**Coverage Target**: 85%+

- **End-to-End Workflows**: Complete project lifecycle testing
- **Database Integration**: Real database operations with transactions
- **Cache Integration**: Redis integration with real connections
- **File System Integration**: File monitoring and analysis workflows

```bash
# Run integration tests
pytest tests/integration/ -m integration -v
```

### 3. Performance Tests (`@pytest.mark.performance`)

**Purpose**: Validate performance characteristics and benchmarks
**Speed**: Slower (30+ seconds per test)
**Coverage Target**: Key performance paths

- **Load Testing**: Handle multiple concurrent operations
- **Benchmark Validation**: Ensure performance targets are met
- **Memory Usage**: Validate memory consumption limits
- **Database Performance**: Query optimization and response times

```bash
# Run performance tests
pytest tests/performance/ -m performance -v --benchmark-only
```

### 4. Security Tests (`@pytest.mark.security`)

**Purpose**: Validate security controls and vulnerability prevention
**Speed**: Moderate (5-15 seconds per test)
**Coverage Target**: All security-critical paths

- **Input Validation**: SQL injection, XSS, path traversal prevention
- **Authentication**: Access control and authorization
- **Rate Limiting**: DoS protection and resource limits
- **Data Privacy**: Information disclosure prevention

```bash
# Run security tests
pytest tests/security/ -m security -v
```

### 5. Frontend Tests (`@pytest.mark.frontend`)

**Purpose**: Validate frontend-backend contracts and API compatibility
**Speed**: Fast to moderate (1-10 seconds per test)
**Coverage Target**: All public API contracts

- **API Response Formats**: Ensure consistency with frontend expectations
- **PWA Functionality**: Progressive Web App features
- **Mobile Compatibility**: Responsive design validation
- **WebSocket Contracts**: Real-time communication validation

```bash
# Run frontend contract tests
pytest tests/frontend/ -m frontend -v
```

## Test Data & Fixtures

### Shared Fixtures (`project_index_conftest.py`)

**Core Database Fixtures**:
- `project_index_engine`: Async database engine for testing
- `project_index_session`: Database session with proper cleanup
- `sample_project_index`: Realistic project instance
- `sample_file_entries`: File entries with analysis data
- `sample_dependencies`: Dependency relationships
- `sample_analysis_session`: Analysis session with results

**Infrastructure Fixtures**:
- `mock_redis_client`: Redis client mock for cache testing
- `mock_event_publisher`: Event publishing mock
- `temp_project_directory`: Temporary directory for file operations
- `async_test_client`: FastAPI test client for API testing

**Configuration Fixtures**:
- `performance_test_config`: Performance testing parameters
- `security_test_config`: Security testing scenarios
- `websocket_test_config`: WebSocket testing configuration

### Test Data Factory (`project_index_test_data.py`)

The `ProjectIndexTestDataFactory` provides realistic test data generation:

```python
from tests.fixtures.project_index_test_data import test_data_factory

# Create small project for quick tests
small_project = test_data_factory.create_project_data(
    project_type="python_webapp",
    complexity="simple"
)

# Create large project for stress testing
large_project = test_data_factory.create_project_data(
    project_type="python_webapp", 
    complexity="large"
)

# Create data science project
ds_project = test_data_factory.create_project_data(
    project_type="data_science",
    complexity="medium"
)
```

**Predefined Scenarios**:
- `create_small_python_project()`: Quick testing scenarios
- `create_large_enterprise_project()`: Stress testing scenarios
- `create_data_science_project()`: ML/analytics workflows
- `create_microservice_project()`: Service-oriented architecture

## Running Tests

### Quick Test Commands

```bash
# Run all Project Index tests
pytest tests/ -k "project_index" -v

# Run with coverage
pytest tests/ -k "project_index" --cov=app.project_index --cov-report=html

# Run specific test category
pytest tests/unit/ -m project_index_unit -v
pytest tests/integration/ -m project_index_integration -v
pytest tests/performance/ -m project_index_performance --benchmark-only
pytest tests/security/ -m project_index_security -v

# Run tests in parallel (faster execution)
pytest tests/ -k "project_index" -n auto -v

# Run with detailed output
pytest tests/ -k "project_index" -v -s --tb=long
```

### Environment Setup

**Required Environment Variables**:
```bash
export TESTING=true
export DATABASE_URL=postgresql://test_user:test_password@localhost:5432/test_db
export REDIS_URL=redis://localhost:6379/0
```

**Docker Services** (for integration tests):
```bash
# Start test services
docker-compose -f tests/docker-compose.test.yml up -d

# Stop test services
docker-compose -f tests/docker-compose.test.yml down
```

### Test Markers

Use pytest markers to run specific test categories:

```bash
# Unit tests only
pytest -m "unit and project_index"

# Integration tests only  
pytest -m "integration and project_index"

# Performance tests only
pytest -m "performance and project_index"

# Security tests only
pytest -m "security and project_index"

# Slow tests (for CI pipelines)
pytest -m "slow and project_index"

# Fast tests (for development)
pytest -m "not slow and project_index"
```

## Coverage Requirements

### Coverage Targets

| Test Category | Minimum Coverage | Target Coverage |
|--------------|------------------|-----------------|
| Unit Tests | 90% | 95% |
| Integration Tests | 80% | 85% |
| Overall Project Index | 85% | 90% |

### Coverage Commands

```bash
# Generate coverage report
pytest tests/ -k "project_index" \
  --cov=app.models.project_index \
  --cov=app.schemas.project_index \
  --cov=app.api.project_index \
  --cov=app.project_index \
  --cov-branch \
  --cov-report=html:htmlcov-project-index \
  --cov-report=xml:coverage-project-index.xml \
  --cov-report=term-missing \
  --cov-fail-under=85

# View HTML coverage report
open htmlcov-project-index/index.html
```

### Coverage Exclusions

Lines excluded from coverage requirements:
- Error handling for impossible conditions
- Debug logging statements
- Type checking blocks (`if TYPE_CHECKING`)
- Platform-specific code paths
- Defensive programming assertions

## CI/CD Integration

### GitHub Actions Workflow

The Project Index quality pipeline (`.github/workflows/project_index_quality_pipeline.yml`) provides:

1. **Static Analysis**: Code linting, type checking, security scanning
2. **Unit Tests**: Fast feedback with parallel execution
3. **Integration Tests**: Database and Redis integration validation
4. **Performance Tests**: Benchmark validation and regression detection
5. **Security Tests**: Vulnerability scanning and input validation
6. **Frontend Tests**: API contract validation
7. **Coverage Consolidation**: Overall quality reporting
8. **Deployment Readiness**: Quality gates for production deployment

### Quality Gates

| Gate | Requirement | Blocking |
|------|-------------|----------|
| Static Analysis | No linting errors | Yes |
| Unit Test Coverage | ≥ 90% | Yes |
| Integration Tests | All pass | Yes |
| Performance Benchmarks | Within 10% of baseline | Yes |
| Security Tests | All pass | Yes |
| Overall Coverage | ≥ 85% | Yes |

### Local Quality Check

```bash
# Run the same checks as CI locally
./scripts/run_quality_checks.sh

# Or manually:
black --check app/project_index/
isort --check-only app/project_index/
flake8 app/project_index/
mypy app/project_index/
bandit -r app/project_index/
pytest tests/ -k "project_index" --cov-fail-under=85
```

## Performance Testing

### Performance Targets

| Operation | Target | Maximum |
|-----------|--------|---------|
| Project Creation | < 2s | 5s |
| File Analysis | < 100ms | 500ms |
| Dependency Graph | < 1s | 3s |
| Cache Operations | < 10ms | 50ms |
| API Response | < 200ms | 1s |

### Benchmark Tests

```bash
# Run performance benchmarks
pytest tests/performance/ --benchmark-only --benchmark-json=results.json

# Compare with baseline
pytest tests/performance/ --benchmark-compare=baseline.json

# Generate performance report
pytest tests/performance/ --benchmark-histogram=performance-histogram
```

### Load Testing

```bash
# Concurrent operations test
pytest tests/performance/ -k "concurrent" -v

# Large dataset test
pytest tests/performance/ -k "large_project" -v

# Memory usage validation
pytest tests/performance/ -k "memory" -v
```

## Security Testing

### Security Test Categories

1. **Input Validation**
   - SQL injection prevention
   - XSS prevention
   - Path traversal prevention
   - Unicode handling

2. **Authentication & Authorization**
   - Access control validation
   - Session security
   - Rate limiting

3. **Data Privacy**
   - Information disclosure prevention
   - Error message sanitization
   - Sensitive data masking

4. **Infrastructure Security**
   - File system access restrictions
   - Network security
   - WebSocket security

### Security Test Execution

```bash
# Run all security tests
pytest tests/security/ -m security -v

# Run specific security categories
pytest tests/security/ -k "input_validation" -v
pytest tests/security/ -k "authentication" -v
pytest tests/security/ -k "data_privacy" -v

# Generate security report
pytest tests/security/ --html=security-report.html --self-contained-html
```

## Troubleshooting

### Common Issues

**1. Database Connection Errors**
```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Reset test database
psql -h localhost -U test_user -c "DROP DATABASE IF EXISTS test_project_index_db;"
psql -h localhost -U test_user -c "CREATE DATABASE test_project_index_db;"
```

**2. Redis Connection Errors**
```bash
# Check Redis is running
redis-cli ping

# Clear Redis test data
redis-cli FLUSHALL
```

**3. SQLite/JSONB Compatibility Issues**
```bash
# Use PostgreSQL for integration tests
export DATABASE_URL=postgresql://test_user:test_password@localhost:5432/test_db

# Or skip tests with SQLite issues
pytest -m "not sqlite_incompatible"
```

**4. Import Errors**
```bash
# Install test dependencies
pip install -e .[test]

# Or with poetry
poetry install --with test
```

**5. Slow Test Performance**
```bash
# Run only fast tests during development
pytest -m "not slow"

# Use parallel execution
pytest -n auto

# Profile slow tests
pytest --profile-svg
```

### Debug Commands

```bash
# Run with debug output
pytest tests/ -k "project_index" -v -s --log-cli-level=DEBUG

# Run single test with pdb
pytest tests/unit/test_project_index_models.py::TestProjectIndex::test_create_project -v -s --pdb

# Run with coverage debug
pytest tests/ -k "project_index" --cov-report=html --cov-debug=trace1
```

### Test Data Cleanup

```bash
# Clean test databases
python -c "
import asyncio
from tests.project_index_conftest import cleanup_test_data
asyncio.run(cleanup_test_data())
"

# Clean Redis test data
redis-cli EVAL "return redis.call('del', unpack(redis.call('keys', ARGV[1])))" 0 "test:*"

# Clean temporary files
rm -rf /tmp/test_project_*
rm -rf tests/.pytest_cache
```

## Best Practices

### Writing Tests

1. **Test Organization**
   - One test class per component
   - Descriptive test method names
   - Proper test categorization with markers

2. **Test Data**
   - Use fixtures for common test data
   - Generate realistic test scenarios
   - Clean up test data after each test

3. **Assertions**
   - Use specific assertions
   - Include helpful error messages
   - Test both positive and negative cases

4. **Performance**
   - Keep unit tests fast (< 1 second)
   - Use mocks for external dependencies
   - Parallelize independent tests

### Maintenance

1. **Regular Updates**
   - Update test data as schema evolves
   - Maintain performance baselines
   - Review and update security test scenarios

2. **Monitoring**
   - Track test execution times
   - Monitor coverage trends
   - Identify flaky tests

3. **Documentation**
   - Keep this guide updated
   - Document new test scenarios
   - Maintain troubleshooting guide

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [SQLAlchemy Testing](https://docs.sqlalchemy.org/en/20/orm/session_transaction.html#joining-a-session-into-an-external-transaction-such-as-for-test-suites)
- [Performance Testing Best Practices](https://pytest-benchmark.readthedocs.io/)
- [Security Testing Guidelines](https://owasp.org/www-project-web-security-testing-guide/)

---

**Last Updated**: 2024-08-15
**Version**: 1.0.0
**Maintainer**: Project Index QA Team