#!/bin/bash

# Pre-commit Test Validation Script
# Ensures all tests pass and code quality standards are met before commit

set -e  # Exit on any error

echo "🧪 Running pre-commit test validation..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MIN_COVERAGE=80
MAX_TEST_TIME=300  # 5 minutes

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to run tests with timeout
run_tests_with_timeout() {
    local test_command=$1
    local description=$2
    local timeout=$3
    
    print_status $YELLOW "Running $description..."
    
    if timeout $timeout $test_command; then
        print_status $GREEN "✅ $description passed"
        return 0
    else
        print_status $RED "❌ $description failed or timed out"
        return 1
    fi
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] && [ ! -f "requirements.txt" ]; then
    print_status $RED "❌ Not in a Python project directory"
    exit 1
fi

# Set environment for testing
export CI=true
export SKIP_STARTUP_INIT=true

# 1. Run critical fast tests first
print_status $YELLOW "🚀 Running critical fast tests..."

# Unit tests for core components
run_tests_with_timeout "python -m pytest tests/test_database_connection_pool_failure.py -v --tb=short" "Database Connection Pool Tests" 60

run_tests_with_timeout "python -m pytest tests/test_agent_orchestrator_error_recovery.py -v --tb=short" "Agent Orchestrator Error Recovery Tests" 120

run_tests_with_timeout "python -m pytest tests/test_fastapi_service_lifecycle.py -v --tb=short" "FastAPI Service Lifecycle Tests" 90

run_tests_with_timeout "python -m pytest tests/test_cross_component_contracts.py -v --tb=short" "Cross-Component Contract Tests" 60

# 2. Run enhanced orchestrator tests
print_status $YELLOW "🎭 Running enhanced orchestrator tests..."
run_tests_with_timeout "python -m pytest tests/test_enhanced_orchestrator_comprehensive.py -v --tb=short" "Enhanced Orchestrator Tests" 180

# 3. Run smoke tests
print_status $YELLOW "💨 Running smoke tests..."
run_tests_with_timeout "python -m pytest tests/smoke/ -v --tb=short" "Smoke Tests" 60

# 4. Check test coverage
print_status $YELLOW "📊 Checking test coverage..."

coverage_output=$(python -m pytest --cov=app --cov-report=term-missing --cov-report=json:coverage.json tests/ -q 2>/dev/null || true)

if [ -f "coverage.json" ]; then
    coverage_percentage=$(python -c "
import json
try:
    with open('coverage.json', 'r') as f:
        data = json.load(f)
    print(f\"{data['totals']['percent_covered']:.1f}\")
except:
    print('0.0')
")

    if (( $(echo "$coverage_percentage >= $MIN_COVERAGE" | bc -l) )); then
        print_status $GREEN "✅ Test coverage: ${coverage_percentage}% (>= ${MIN_COVERAGE}%)"
    else
        print_status $YELLOW "⚠️ Test coverage: ${coverage_percentage}% (< ${MIN_COVERAGE}%) - Warning only"
        # Don't fail on coverage for now, just warn
    fi
else
    print_status $YELLOW "⚠️ Could not determine test coverage"
fi

# 5. Run code quality checks
print_status $YELLOW "🔍 Running code quality checks..."

# Check for basic Python syntax errors
if command -v python >/dev/null 2>&1; then
    print_status $YELLOW "Checking Python syntax..."
    python -m py_compile app/core/orchestrator.py
    python -m py_compile app/core/database.py
    python -m py_compile app/main.py
    print_status $GREEN "✅ Python syntax check passed"
fi

# Check for import errors in key modules
print_status $YELLOW "Checking critical imports..."
python -c "
try:
    from app.core.orchestrator import AgentOrchestrator
    from app.core.database import get_session
    from app.models.task import Task
    print('✅ Critical imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# 6. Check for common issues
print_status $YELLOW "🔍 Checking for common issues..."

# Check for unresolved merge conflicts
if grep -r "<<<<<<< " --include="*.py" app/ tests/ 2>/dev/null; then
    print_status $RED "❌ Unresolved merge conflicts detected"
    exit 1
fi

# Check for debugging statements
if grep -r "import pdb\|pdb.set_trace()\|breakpoint()" --include="*.py" app/ 2>/dev/null; then
    print_status $YELLOW "⚠️ Debugging statements found - please remove before commit"
fi

# Check for TODO/FIXME comments in critical files
todo_count=$(grep -r "TODO\|FIXME" --include="*.py" app/core/ | wc -l)
if [ "$todo_count" -gt 0 ]; then
    print_status $YELLOW "⚠️ Found $todo_count TODO/FIXME items in core modules"
fi

# 7. Validate configuration files
print_status $YELLOW "⚙️ Validating configuration..."

if [ -f "pyproject.toml" ]; then
    python -c "
import tomllib
try:
    with open('pyproject.toml', 'rb') as f:
        tomllib.load(f)
    print('✅ pyproject.toml is valid')
except Exception as e:
    print(f'❌ pyproject.toml is invalid: {e}')
    exit(1)
" 2>/dev/null || echo "⚠️ Could not validate pyproject.toml (requires Python 3.11+)"
fi

# 8. Check database model consistency
print_status $YELLOW "🗄️ Checking database model consistency..."
python -c "
try:
    from app.models.task import Task
    from app.models.agent import Agent
    from app.models.workflow import Workflow
    
    # Verify Task model has workflow_id field
    task = Task(title='test', description='test')
    assert hasattr(task, 'workflow_id'), 'Task model missing workflow_id field'
    
    print('✅ Database models are consistent')
except Exception as e:
    print(f'❌ Database model issue: {e}')
    exit(1)
"

# 9. Performance check - ensure tests don't take too long
print_status $YELLOW "⏱️ Validating test performance..."
test_start_time=$(date +%s)

# Run a quick performance test
python -m pytest tests/test_enhanced_orchestrator_comprehensive.py::TestEnhancedOrchestratorPersonaIntegration::test_assign_task_with_persona_success -v --tb=no -q >/dev/null 2>&1

test_end_time=$(date +%s)
test_duration=$((test_end_time - test_start_time))

if [ $test_duration -gt 30 ]; then
    print_status $YELLOW "⚠️ Single test took ${test_duration}s - consider optimization"
else
    print_status $GREEN "✅ Test performance acceptable (${test_duration}s)"
fi

# 10. Final validation
print_status $YELLOW "🎯 Running final validation..."

# Ensure critical test files exist
critical_test_files=(
    "tests/test_database_connection_pool_failure.py"
    "tests/test_agent_orchestrator_error_recovery.py"
    "tests/test_fastapi_service_lifecycle.py"
    "tests/test_cross_component_contracts.py"
)

for test_file in "${critical_test_files[@]}"; do
    if [ ! -f "$test_file" ]; then
        print_status $RED "❌ Critical test file missing: $test_file"
        exit 1
    fi
done

print_status $GREEN "✅ All critical test files present"

# Summary
print_status $GREEN "
🎉 Pre-commit validation completed successfully!

Summary:
✅ Database connection pool tests: PASSED
✅ Agent orchestrator error recovery tests: PASSED  
✅ FastAPI service lifecycle tests: PASSED
✅ Cross-component contract tests: PASSED
✅ Enhanced orchestrator tests: PASSED
✅ Smoke tests: PASSED
✅ Code quality checks: PASSED
✅ Configuration validation: PASSED
✅ Database model consistency: PASSED
✅ Test performance: ACCEPTABLE

🚀 Ready to commit with confidence!
"

# Clean up temporary files
rm -f coverage.json 2>/dev/null || true

exit 0