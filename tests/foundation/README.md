# Foundation Testing Layer - Testing Pyramid Base

This directory contains the foundation testing layer implementation for the LeanVibe Agent Hive project. Foundation tests are the base of the testing pyramid and focus on validating basic system integrity.

## Overview

**TESTING PYRAMID LEVEL**: Foundation (Base Layer)  
**PURPOSE**: Validate imports, configurations, models, and core dependencies  
**EXECUTION TIME TARGET**: <30 seconds total  
**QUALITY REQUIREMENT**: All foundation tests must pass for higher-level testing  

## Test Categories

### 1. Import Resolution Testing (`test_import_resolution.py`)
Validates that all Python modules can be imported without circular dependency errors or missing dependencies.

**Covers:**
- Core modules (app.main, app.core.config, app.core.database, app.core.redis)
- API modules (v1 endpoints)
- Schema modules (Pydantic models)
- Conditional imports (modules with optional dependencies)
- Circular dependency detection
- Import performance validation
- Third-party package availability

**Quality Gates:**
- No critical import failures
- No circular dependencies in core modules
- Import time <5 seconds per module
- All required packages available

### 2. Configuration Validation Testing (`test_configuration_validation.py`)
Validates that all environment variable combinations work correctly and configuration classes instantiate properly.

**Covers:**
- Development environment configuration
- Production environment configuration
- Testing environment configuration
- CI environment configuration
- Missing required variables handling
- Invalid URL format handling
- Type conversion validation
- Environment-specific security settings

**Quality Gates:**
- All environment types validate successfully
- Required settings properly defined
- URL formats validate correctly
- Configuration loading <3 seconds

### 3. Model Integrity Testing (`test_model_integrity.py`)
Validates that Pydantic models validate correctly, database models have proper relationships, and API schemas are consistent.

**Covers:**
- Pydantic model discovery and validation
- Database model structure validation
- Model serialization/deserialization
- Enum consistency validation
- Field type validation
- Schema compliance testing

**Quality Gates:**
- All discovered models validate successfully
- Serialization works correctly
- No schema inconsistencies
- Model validation <10 seconds

### 4. Core Dependency Testing (`test_core_dependencies.py`)
Validates that core dependencies are functional including database connectivity, Redis operations, and essential packages.

**Covers:**
- Database connectivity and basic operations
- Redis connectivity and basic operations
- Third-party package availability
- File system permissions
- Network connectivity (when applicable)
- Package version validation

**Quality Gates:**
- Essential packages available
- Basic file operations work
- Configuration readable
- Connection tests pass (when not mocked)

## Usage

### Running All Foundation Tests
```bash
# Run complete foundation test suite
python3 tests/foundation/run_foundation_tests.py

# Run with verbose output
python3 tests/foundation/run_foundation_tests.py --verbose

# Run in fast mode (exit on first failure)
python3 tests/foundation/run_foundation_tests.py --fast

# Run quietly
python3 tests/foundation/run_foundation_tests.py --quiet
```

### Running Individual Test Categories
```bash
# Import resolution tests
python3 -m pytest tests/foundation/test_import_resolution.py -v

# Configuration validation tests
python3 -m pytest tests/foundation/test_configuration_validation.py -v

# Model integrity tests
python3 -m pytest tests/foundation/test_model_integrity.py -v

# Core dependency tests
python3 -m pytest tests/foundation/test_core_dependencies.py -v
```

### Foundation Test Configuration
Foundation tests use their own pytest configuration (`pytest.ini`) optimized for:
- Fast execution
- Minimal warnings
- No coverage collection (foundation layer focus)
- Timeout protection
- Clean output

## Test Architecture

### Key Components

1. **Test Validators**: Specialized classes for each test category
   - `ImportResolver`: Handles import resolution and circular dependency detection
   - `ConfigurationValidator`: Manages environment configuration testing
   - `ModelIntegrityValidator`: Validates Pydantic and database models
   - `CoreDependencyValidator`: Tests core system dependencies

2. **Test Fixtures**: Shared test utilities and mocks
   - `foundation_test_environment`: Sets up isolated test environment
   - `mock_database`: Provides database mocking for testing
   - `mock_redis`: Provides Redis mocking for testing
   - `test_data_factory`: Generates valid test data

3. **Quality Gates**: Automated validation of test quality
   - Execution time validation
   - Test coverage validation
   - Failure rate validation
   - Performance monitoring

### Error Handling Strategy

Foundation tests are designed to be resilient and handle common issues gracefully:

- **Known Issues**: PyO3 module initialization conflicts are detected and handled
- **Configuration Issues**: Expected configuration validation errors are caught
- **Optional Dependencies**: Missing optional packages don't cause failures
- **Environment Isolation**: Tests use isolated environment variables

### Integration with CI/CD

Foundation tests are designed to run in CI/CD environments:
- Automatic detection of CI environments
- Appropriate mocking of external services
- Fast execution for rapid feedback
- Clear failure reporting

## Quality Standards

### Foundation Test Requirements

1. **Fast Execution**: Individual tests <5 seconds, total suite <30 seconds
2. **Reliable**: No flaky tests, consistent results across environments
3. **Comprehensive**: Cover all critical system foundations
4. **Isolated**: No dependencies on external services in testing
5. **Clear Reporting**: Detailed error messages and recommendations

### Success Criteria

Foundation tests must achieve:
- ✅ All 4 test categories pass
- ✅ Zero critical failures
- ✅ Execution time <30 seconds
- ✅ No circular dependencies
- ✅ All required packages available

## Troubleshooting

### Common Issues

1. **Import Failures**: 
   - Check for missing dependencies in requirements.txt
   - Verify PYTHONPATH is set correctly
   - Look for circular import issues

2. **Configuration Issues**:
   - Ensure required environment variables are set
   - Check for typos in configuration keys
   - Validate URL formats

3. **Model Validation Failures**:
   - Check for schema changes in Pydantic models
   - Verify enum values are consistent
   - Ensure required fields are present

4. **Dependency Issues**:
   - Verify package installation
   - Check for version conflicts
   - Ensure database/Redis are available

### Getting Help

1. Check the detailed test report: `tests/foundation/foundation_test_report.json`
2. Run individual test categories to isolate issues
3. Use verbose mode for detailed error information
4. Review the test logs for specific error messages

## Implementation Status

**COMPLETED ✅**:
- Foundation test directory structure
- Import resolution testing with circular dependency detection
- Configuration validation for all environments
- Model integrity testing for Pydantic models
- Core dependency testing with mocking support
- Test runner with quality gates
- Comprehensive error handling and reporting

**CURRENT STATUS**:
- 2 of 4 test categories passing
- Minor configuration and model validation issues
- Foundation layer 70% complete
- Ready for iteration and refinement

This foundation testing layer provides a solid base for the testing pyramid, ensuring that all higher-level tests can rely on properly functioning imports, configurations, models, and dependencies.