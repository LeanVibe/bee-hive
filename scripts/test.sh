#!/bin/bash

# LeanVibe Agent Hive 2.0 - Comprehensive Testing Script
# Professional testing framework with multiple test suites and reporting
#
# Usage: ./scripts/test.sh [SUITE] [OPTIONS]
# Suites: all (default), unit, integration, performance, security, e2e
#
# Environment Variables:
#   TEST_SUITE        - Override suite selection
#   COVERAGE_THRESHOLD - Minimum coverage percentage (default: 90)
#   PARALLEL_JOBS     - Number of parallel test processes (default: auto)
#   SKIP_SLOW         - Skip slow tests (true/false)
#   GENERATE_REPORT   - Generate HTML test report (true/false)

set -euo pipefail

# Color codes for professional output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Script metadata
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_VERSION="2.0.0"

# Configuration
readonly DEFAULT_SUITE="all"
readonly DEFAULT_COVERAGE_THRESHOLD=90
readonly DEFAULT_PARALLEL_JOBS="auto"

# Test suite configurations (using functions for compatibility)
get_suite_description() {
    case "$1" in
        "all") echo "Run all test suites with comprehensive reporting" ;;
        "unit") echo "Run unit tests only (fast execution)" ;;
        "integration") echo "Run integration tests with external dependencies" ;;
        "performance") echo "Run performance benchmarks and load tests" ;;
        "security") echo "Run security scans and vulnerability tests" ;;
        "e2e") echo "Run end-to-end tests with full system simulation" ;;
        "smoke") echo "Run smoke tests for quick validation" ;;
        *) echo "Unknown test suite" ;;
    esac
}

get_suite_command() {
    case "$1" in
        "unit") echo "pytest tests/ -v -m 'not integration and not performance and not e2e and not slow'" ;;
        "integration") echo "pytest tests/ -v -m 'integration'" ;;
        "performance") echo "pytest tests/performance/ -v -m 'performance or benchmark'" ;;
        "security") echo "bandit -r app/ && safety check && pytest tests/security/ -v" ;;
        "e2e") echo "pytest tests/ -v -m 'e2e'" ;;
        "smoke") echo "pytest tests/ -v -m 'smoke' -x" ;;
        *) echo "" ;;
    esac
}

# Global variables
TEST_SUITE="${TEST_SUITE:-${1:-$DEFAULT_SUITE}}"
COVERAGE_THRESHOLD="${COVERAGE_THRESHOLD:-$DEFAULT_COVERAGE_THRESHOLD}"
PARALLEL_JOBS="${PARALLEL_JOBS:-$DEFAULT_PARALLEL_JOBS}"
SKIP_SLOW="${SKIP_SLOW:-false}"
GENERATE_REPORT="${GENERATE_REPORT:-true}"
START_TIME=""
TEST_LOG=""
COVERAGE_REPORT=""

#======================================
# Utility Functions
#======================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")  echo -e "${BLUE}[INFO]${NC}  $message" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC}  $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $message" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} $message" ;;
        "STEP") echo -e "${PURPLE}[STEP]${NC} $message" ;;
        "TEST") echo -e "${CYAN}[TEST]${NC} $message" ;;
    esac
    
    # Log to file if available
    if [[ -n "$TEST_LOG" ]]; then
        echo "[$timestamp] [$level] $message" >> "$TEST_LOG"
    fi
}

show_header() {
    clear
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                          LeanVibe Agent Hive 2.0                            ║
║                          Testing Framework                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo
    log "INFO" "Running test suite: ${TEST_SUITE}"
    log "INFO" "Suite: $(get_suite_description "$TEST_SUITE")"
    echo
    START_TIME=$(date +%s)
}

show_help() {
    cat << EOF
${CYAN}LeanVibe Agent Hive 2.0 - Comprehensive Testing Script${NC}

${YELLOW}USAGE:${NC}
    $SCRIPT_NAME [SUITE] [OPTIONS]

${YELLOW}TEST SUITES:${NC}
EOF
    for suite in all unit integration performance security e2e smoke; do
        echo "    ${GREEN}$suite${NC} - $(get_suite_description "$suite")"
    done
    cat << EOF

${YELLOW}ENVIRONMENT VARIABLES:${NC}
    TEST_SUITE          Override suite selection
    COVERAGE_THRESHOLD  Minimum coverage percentage (default: 90)
    PARALLEL_JOBS       Number of parallel test processes (default: auto)
    SKIP_SLOW           Skip slow tests (true/false)
    GENERATE_REPORT     Generate HTML test report (true/false)

${YELLOW}EXAMPLES:${NC}
    $SCRIPT_NAME                      # Run all tests
    $SCRIPT_NAME unit                 # Run unit tests only
    $SCRIPT_NAME integration          # Run integration tests
    SKIP_SLOW=true $SCRIPT_NAME       # Skip slow tests
    COVERAGE_THRESHOLD=95 $SCRIPT_NAME # Require 95% coverage

${YELLOW}REPORTS:${NC}
    Test results: reports/test-results.xml
    Coverage: htmlcov/index.html
    Performance: reports/performance/

${YELLOW}MORE INFO:${NC}
    Quality gates: docs/QUALITY_GATES_AUTOMATION.md
    CI/CD integration: docs/ci_cd_enhanced_pipeline.yml
EOF
}

setup_logging() {
    local log_dir="$PROJECT_ROOT/logs"
    local report_dir="$PROJECT_ROOT/reports"
    mkdir -p "$log_dir" "$report_dir"
    
    TEST_LOG="$log_dir/test-$(date '+%Y%m%d-%H%M%S').log"
    COVERAGE_REPORT="$report_dir/coverage-$(date '+%Y%m%d-%H%M%S').json"
    
    log "INFO" "Test logging to: $TEST_LOG"
    log "INFO" "Coverage report: $COVERAGE_REPORT"
}

check_prerequisites() {
    log "STEP" "Checking testing prerequisites..."
    
    local errors=0
    
    # Check virtual environment
    if [[ ! -d "$PROJECT_ROOT/venv" ]]; then
        log "ERROR" "Python virtual environment not found. Run 'make setup' first."
        errors=$((errors + 1))
    fi
    
    # Check test dependencies
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    local required_packages=("pytest" "pytest-cov" "pytest-xdist")
    for package in "${required_packages[@]}"; do
        if ! python -c "import $package" 2>/dev/null; then
            log "WARN" "Test package '$package' not found - installing..."
            pip install "$package" || errors=$((errors + 1))
        fi
    done
    
    # Check Docker for integration tests
    if [[ "$TEST_SUITE" == "all" ]] || [[ "$TEST_SUITE" == "integration" ]] || [[ "$TEST_SUITE" == "e2e" ]]; then
        if ! command -v docker &> /dev/null; then
            log "WARN" "Docker not available - integration/e2e tests may fail"
        elif ! docker info &> /dev/null; then
            log "WARN" "Docker daemon not running - integration/e2e tests may fail"
        fi
    fi
    
    # Check for test services
    if [[ "$TEST_SUITE" == "all" ]] || [[ "$TEST_SUITE" == "integration" ]]; then
        if ! docker-compose ps postgres 2>/dev/null | grep -q "Up"; then
            log "WARN" "PostgreSQL not running - some integration tests may fail"
            log "INFO" "Start services with: make start"
        fi
        
        if ! docker-compose ps redis 2>/dev/null | grep -q "Up"; then
            log "WARN" "Redis not running - some integration tests may fail"
            log "INFO" "Start services with: make start"
        fi
    fi
    
    if [[ $errors -gt 0 ]]; then
        log "ERROR" "Prerequisites check failed"
        exit 1
    fi
    
    log "SUCCESS" "Prerequisites check passed"
}

prepare_test_environment() {
    log "STEP" "Preparing test environment..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    # Set test environment variables
    export TESTING=true
    export DATABASE_URL="postgresql://test_user:test_password@localhost:5432/test_leanvibe_agent_hive"
    export REDIS_URL="redis://localhost:6379/15"  # Use test database
    export LOG_LEVEL="WARNING"  # Reduce log noise during tests
    
    # Create test database if needed
    if [[ "$TEST_SUITE" == "all" ]] || [[ "$TEST_SUITE" == "integration" ]]; then
        log "INFO" "Setting up test database..."
        # This would typically create/reset test database
        # Commented out to avoid interfering with existing setup
        # createdb -h localhost -U leanvibe_user test_leanvibe_agent_hive 2>/dev/null || true
    fi
    
    # Clear previous test artifacts
    rm -rf .pytest_cache/ htmlcov/ .coverage reports/test-results.xml 2>/dev/null || true
    
    log "SUCCESS" "Test environment prepared"
}

run_code_quality_checks() {
    log "STEP" "Running code quality checks..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    local quality_errors=0
    
    # Run linting (ruff)
    if command -v ruff &> /dev/null; then
        log "TEST" "Running ruff linting..."
        if ruff check . --output-format=junit > reports/ruff-results.xml 2>/dev/null; then
            log "SUCCESS" "Ruff linting passed"
        else
            log "WARN" "Ruff linting found issues"
            quality_errors=$((quality_errors + 1))
        fi
    fi
    
    # Run code formatting check (black)
    if command -v black &> /dev/null; then
        log "TEST" "Checking code formatting..."
        if black --check . --quiet; then
            log "SUCCESS" "Code formatting check passed"
        else
            log "WARN" "Code formatting issues found (run: black .)"
            quality_errors=$((quality_errors + 1))
        fi
    fi
    
    # Run type checking (mypy)
    if command -v mypy &> /dev/null; then
        log "TEST" "Running type checking..."
        if mypy app --junit-xml reports/mypy-results.xml 2>/dev/null; then
            log "SUCCESS" "Type checking passed"
        else
            log "WARN" "Type checking found issues"
            quality_errors=$((quality_errors + 1))
        fi
    fi
    
    if [[ $quality_errors -gt 0 ]]; then
        log "WARN" "Code quality checks found $quality_errors issue(s)"
    else
        log "SUCCESS" "All code quality checks passed"
    fi
}

run_unit_tests() {
    log "STEP" "Running unit tests..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    local pytest_args="-v --tb=short"
    local markers="not integration and not performance and not e2e"
    
    # Add slow test exclusion if requested
    if [[ "$SKIP_SLOW" == "true" ]]; then
        markers="$markers and not slow"
    fi
    
    # Add parallel execution
    if [[ "$PARALLEL_JOBS" != "1" ]]; then
        if [[ "$PARALLEL_JOBS" == "auto" ]]; then
            pytest_args="$pytest_args -n auto"
        else
            pytest_args="$pytest_args -n $PARALLEL_JOBS"
        fi
    fi
    
    # Add coverage reporting
    pytest_args="$pytest_args --cov=app --cov-report=html --cov-report=xml --cov-report=json:$COVERAGE_REPORT"
    pytest_args="$pytest_args --cov-fail-under=$COVERAGE_THRESHOLD"
    
    # Add JUnit XML output
    pytest_args="$pytest_args --junit-xml=reports/unit-test-results.xml"
    
    log "TEST" "Command: pytest tests/ $pytest_args -m '$markers'"
    
    if pytest tests/ $pytest_args -m "$markers"; then
        log "SUCCESS" "Unit tests passed"
        return 0
    else
        log "ERROR" "Unit tests failed"
        return 1
    fi
}

run_integration_tests() {
    log "STEP" "Running integration tests..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    local pytest_args="-v --tb=short"
    local markers="integration"
    
    # Add slow test exclusion if requested
    if [[ "$SKIP_SLOW" == "true" ]]; then
        markers="$markers and not slow"
    fi
    
    # Add JUnit XML output
    pytest_args="$pytest_args --junit-xml=reports/integration-test-results.xml"
    
    log "TEST" "Command: pytest tests/ $pytest_args -m '$markers'"
    
    if pytest tests/ $pytest_args -m "$markers"; then
        log "SUCCESS" "Integration tests passed"
        return 0
    else
        log "ERROR" "Integration tests failed"
        return 1
    fi
}

run_performance_tests() {
    log "STEP" "Running performance tests..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    # Create performance report directory
    mkdir -p reports/performance
    
    local pytest_args="-v --tb=short"
    local markers="performance or benchmark"
    
    # Add JUnit XML output
    pytest_args="$pytest_args --junit-xml=reports/performance-test-results.xml"
    
    # Add performance reporting
    pytest_args="$pytest_args --benchmark-json=reports/performance/benchmark-results.json"
    
    log "TEST" "Command: pytest tests/performance/ $pytest_args -m '$markers'"
    
    if pytest tests/performance/ $pytest_args -m "$markers"; then
        log "SUCCESS" "Performance tests passed"
        return 0
    else
        log "ERROR" "Performance tests failed"
        return 1
    fi
}

run_security_tests() {
    log "STEP" "Running security tests..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    local security_errors=0
    
    # Run bandit security scan
    if command -v bandit &> /dev/null; then
        log "TEST" "Running bandit security scan..."
        if bandit -r app/ -f json -o reports/bandit-results.json; then
            log "SUCCESS" "Bandit security scan passed"
        else
            log "WARN" "Bandit found security issues"
            security_errors=$((security_errors + 1))
        fi
    fi
    
    # Run safety check for vulnerable dependencies
    if command -v safety &> /dev/null; then
        log "TEST" "Running safety check..."
        if safety check --json --output reports/safety-results.json; then
            log "SUCCESS" "Safety check passed"
        else
            log "WARN" "Safety found vulnerable dependencies"
            security_errors=$((security_errors + 1))
        fi
    fi
    
    # Run security-specific tests
    if [[ -d "tests/security" ]]; then
        log "TEST" "Running security tests..."
        if pytest tests/security/ -v --junit-xml=reports/security-test-results.xml; then
            log "SUCCESS" "Security tests passed"
        else
            log "ERROR" "Security tests failed"
            security_errors=$((security_errors + 1))
        fi
    fi
    
    if [[ $security_errors -gt 0 ]]; then
        log "ERROR" "Security tests failed with $security_errors issue(s)"
        return 1
    else
        log "SUCCESS" "All security tests passed"
        return 0
    fi
}

run_e2e_tests() {
    log "STEP" "Running end-to-end tests..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    local pytest_args="-v --tb=short"
    local markers="e2e"
    
    # Add JUnit XML output
    pytest_args="$pytest_args --junit-xml=reports/e2e-test-results.xml"
    
    log "TEST" "Command: pytest tests/ $pytest_args -m '$markers'"
    
    if pytest tests/ $pytest_args -m "$markers"; then
        log "SUCCESS" "End-to-end tests passed"
        return 0
    else
        log "ERROR" "End-to-end tests failed"
        return 1
    fi
}

run_smoke_tests() {
    log "STEP" "Running smoke tests..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    local pytest_args="-v --tb=short -x"  # Stop on first failure
    local markers="smoke"
    
    log "TEST" "Command: pytest tests/ $pytest_args -m '$markers'"
    
    if pytest tests/ $pytest_args -m "$markers"; then
        log "SUCCESS" "Smoke tests passed"
        return 0
    else
        log "ERROR" "Smoke tests failed"
        return 1
    fi
}

generate_test_report() {
    if [[ "$GENERATE_REPORT" != "true" ]]; then
        return 0
    fi
    
    log "STEP" "Generating test report..."
    
    cd "$PROJECT_ROOT"
    
    # Create comprehensive test report
    local report_file="reports/test-summary-$(date '+%Y%m%d-%H%M%S').html"
    
    cat > "$report_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>LeanVibe Agent Hive 2.0 - Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f8f9fa; padding: 20px; border-radius: 8px; }
        .success { color: #28a745; }
        .warning { color: #ffc107; }
        .error { color: #dc3545; }
        .section { margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; }
    </style>
</head>
<body>
    <div class="header">
        <h1>LeanVibe Agent Hive 2.0 - Test Report</h1>
        <p>Generated: $(date)</p>
        <p>Test Suite: $TEST_SUITE</p>
    </div>
EOF
    
    # Add test results summary
    echo "    <div class='section'>" >> "$report_file"
    echo "        <h2>Test Results Summary</h2>" >> "$report_file"
    echo "        <p>Comprehensive test execution completed.</p>" >> "$report_file"
    
    # Add links to detailed reports
    if [[ -f "htmlcov/index.html" ]]; then
        echo "        <p><a href='htmlcov/index.html'>Coverage Report</a></p>" >> "$report_file"
    fi
    
    echo "    </div>" >> "$report_file"
    echo "</body></html>" >> "$report_file"
    
    log "SUCCESS" "Test report generated: $report_file"
}

show_test_summary() {
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    echo
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                              TESTING COMPLETE                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo
    log "SUCCESS" "Testing completed in ${minutes}m ${seconds}s"
    echo
    
    echo -e "${YELLOW}TEST REPORTS:${NC}"
    if [[ -f "htmlcov/index.html" ]]; then
        echo "   ${CYAN}Coverage Report:${NC} htmlcov/index.html"
    fi
    if [[ -f "reports/test-results.xml" ]]; then
        echo "   ${CYAN}JUnit Results:${NC}  reports/test-results.xml"
    fi
    if [[ -d "reports/performance" ]]; then
        echo "   ${CYAN}Performance:${NC}    reports/performance/"
    fi
    echo
    
    echo -e "${YELLOW}QUALITY METRICS:${NC}"
    if [[ -f "$COVERAGE_REPORT" ]]; then
        local coverage=$(python -c "
import json
try:
    with open('$COVERAGE_REPORT') as f:
        data = json.load(f)
    print(f\"{data['totals']['percent_covered']:.1f}%\")
except:
    print('N/A')" 2>/dev/null)
        echo "   ${CYAN}Test Coverage:${NC}  $coverage"
    fi
    echo
    
    if [[ -n "$TEST_LOG" ]]; then
        echo -e "${CYAN}Test log saved to:${NC} $TEST_LOG"
    fi
    echo
}

handle_test_error() {
    local exit_code=$?
    log "ERROR" "Testing failed with exit code $exit_code"
    
    echo
    echo -e "${RED}TESTING FAILED${NC}"
    echo
    echo "Troubleshooting:"
    echo "- Check test log: $TEST_LOG"
    echo "- Review failed tests in reports/"
    echo "- Run specific test suite: $SCRIPT_NAME unit"
    echo "- Check service status: make ps"
    echo
    
    exit $exit_code
}

#======================================
# Main Testing Flow
#======================================

main() {
    # Handle help request
    if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
        show_help
        exit 0
    fi
    
    # Validate suite
    if [[ -n "${1:-}" ]] && [[ "$1" != "all" && "$1" != "unit" && "$1" != "integration" && "$1" != "performance" && "$1" != "security" && "$1" != "e2e" && "$1" != "smoke" ]]; then
        log "ERROR" "Invalid test suite: $1"
        echo "Valid suites: all unit integration performance security e2e smoke"
        exit 1
    fi
    
    # Set up error handling
    trap handle_test_error ERR
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Initialize
    show_header
    setup_logging
    check_prerequisites
    prepare_test_environment
    
    # Track test results
    local test_failures=0
    
    # Run code quality checks first
    run_code_quality_checks || test_failures=$((test_failures + 1))
    
    # Run test suites based on selection
    case "$TEST_SUITE" in
        "all")
            run_unit_tests || test_failures=$((test_failures + 1))
            run_integration_tests || test_failures=$((test_failures + 1))
            run_performance_tests || test_failures=$((test_failures + 1))
            run_security_tests || test_failures=$((test_failures + 1))
            run_e2e_tests || test_failures=$((test_failures + 1))
            ;;
        "unit")
            run_unit_tests || test_failures=$((test_failures + 1))
            ;;
        "integration")
            run_integration_tests || test_failures=$((test_failures + 1))
            ;;
        "performance")
            run_performance_tests || test_failures=$((test_failures + 1))
            ;;
        "security")
            run_security_tests || test_failures=$((test_failures + 1))
            ;;
        "e2e")
            run_e2e_tests || test_failures=$((test_failures + 1))
            ;;
        "smoke")
            run_smoke_tests || test_failures=$((test_failures + 1))
            ;;
    esac
    
    # Generate reports
    generate_test_report
    
    # Show summary
    show_test_summary
    
    # Exit with appropriate code
    if [[ $test_failures -gt 0 ]]; then
        log "ERROR" "$test_failures test suite(s) failed"
        exit 1
    else
        log "SUCCESS" "All tests passed successfully"
        exit 0
    fi
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi