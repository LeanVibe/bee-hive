#!/bin/bash
# LeanVibe Agent Hive 2.0 - Smoke Test Runner
# Comprehensive smoke test execution with multiple environments

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SMOKE_TEST_TIMEOUT=120  # 2 minutes max
PERFORMANCE_TARGET_MS=100
MAX_FAILURES=5

echo -e "${BLUE}==========================================${NC}"
echo -e "${BLUE}LeanVibe Agent Hive 2.0 - Smoke Tests${NC}"
echo -e "${BLUE}==========================================${NC}"
echo

# Function to print status
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi
    
    # Check pytest
    if ! python -m pytest --version &> /dev/null; then
        print_error "pytest not found. Please install: pip install pytest pytest-asyncio"
        exit 1
    fi
    
    # Check if we're in the right directory
    if [ ! -f "pytest-smoke.ini" ]; then
        print_error "pytest-smoke.ini not found. Please run from project root."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to set up test environment
setup_environment() {
    print_status "Setting up test environment..."
    
    export TESTING=true
    export DEBUG=true
    export LOG_LEVEL=ERROR
    export SKIP_STARTUP_INIT=true
    export CI=false
    export PYTHONHASHSEED=0
    export DATABASE_URL="sqlite+aiosqlite:///:memory:"
    export REDIS_URL="redis://localhost:6379/1"
    
    print_success "Environment configured"
}

# Function to run smoke tests with different configurations
run_smoke_tests() {
    local test_type="$1"
    local additional_args="$2"
    
    print_status "Running $test_type smoke tests..."
    
    local start_time=$(date +%s)
    
    # Run pytest with smoke test configuration
    if timeout $SMOKE_TEST_TIMEOUT python -m pytest tests/smoke/ \
        -c pytest-smoke.ini \
        --tb=short \
        --durations=10 \
        --color=yes \
        --maxfail=$MAX_FAILURES \
        -v \
        $additional_args; then
        
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        print_success "$test_type tests completed in ${duration}s"
        return 0
    else
        local exit_code=$?
        print_error "$test_type tests failed with exit code $exit_code"
        return $exit_code
    fi
}

# Function to run performance validation
run_performance_validation() {
    print_status "Running performance validation..."
    
    if run_smoke_tests "Performance" "-m performance"; then
        print_success "Performance targets met (<${PERFORMANCE_TARGET_MS}ms)"
    else
        print_warning "Performance tests had issues - check output above"
        return 1
    fi
}

# Function to run core functionality tests
run_core_tests() {
    print_status "Running core functionality tests..."
    
    if run_smoke_tests "Core" "-m 'not performance'"; then
        print_success "Core functionality validated"
    else
        print_error "Core functionality tests failed"
        return 1
    fi
}

# Function to run integration tests
run_integration_tests() {
    print_status "Running integration tests..."
    
    if run_smoke_tests "Integration" "-m integration"; then
        print_success "Integration tests passed"
    else
        print_error "Integration tests failed"
        return 1
    fi
}

# Function to run all tests
run_all_tests() {
    print_status "Running complete smoke test suite..."
    
    if run_smoke_tests "Complete" ""; then
        print_success "All smoke tests passed!"
    else
        print_error "Some smoke tests failed"
        return 1
    fi
}

# Function to generate test report
generate_report() {
    print_status "Generating test report..."
    
    # Run tests with JUnit XML output for reporting
    python -m pytest tests/smoke/ \
        -c pytest-smoke.ini \
        --junitxml=reports/smoke-test-results.xml \
        --tb=short \
        --quiet \
        2>/dev/null || true
    
    if [ -f "reports/smoke-test-results.xml" ]; then
        print_success "Test report generated: reports/smoke-test-results.xml"
    fi
}

# Function to run tests in Docker
run_docker_tests() {
    print_status "Running smoke tests in Docker environment..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -f docker-compose.smoke-tests.yml up --build --abort-on-container-exit smoke-test-runner
        local exit_code=$?
        docker-compose -f docker-compose.smoke-tests.yml down -v
        
        if [ $exit_code -eq 0 ]; then
            print_success "Docker smoke tests passed"
        else
            print_error "Docker smoke tests failed"
            return $exit_code
        fi
    else
        print_warning "Docker Compose not available, skipping Docker tests"
    fi
}

# Function to display usage
usage() {
    echo "Usage: $0 [command]"
    echo
    echo "Commands:"
    echo "  all         Run all smoke tests (default)"
    echo "  core        Run core functionality tests only"
    echo "  performance Run performance tests only"
    echo "  integration Run integration tests only"
    echo "  docker      Run tests in Docker environment"
    echo "  report      Generate test report"
    echo "  help        Show this help message"
    echo
    echo "Options:"
    echo "  --fast      Run only fast tests (<100ms)"
    echo "  --verbose   Enable verbose output"
    echo "  --no-cov    Disable coverage reporting"
    echo
}

# Main execution
main() {
    local command="${1:-all}"
    local exit_code=0
    
    case "$command" in
        help|--help|-h)
            usage
            exit 0
            ;;
        
        all)
            check_prerequisites
            setup_environment
            mkdir -p reports
            run_all_tests
            exit_code=$?
            generate_report
            ;;
        
        core)
            check_prerequisites
            setup_environment
            run_core_tests
            exit_code=$?
            ;;
        
        performance)
            check_prerequisites
            setup_environment
            run_performance_validation
            exit_code=$?
            ;;
        
        integration)
            check_prerequisites
            setup_environment
            run_integration_tests
            exit_code=$?
            ;;
        
        docker)
            run_docker_tests
            exit_code=$?
            ;;
        
        report)
            check_prerequisites
            setup_environment
            mkdir -p reports
            generate_report
            ;;
        
        *)
            print_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
    
    echo
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}==========================================${NC}"
        echo -e "${GREEN}🎉 Smoke tests completed successfully!${NC}"
        echo -e "${GREEN}The system is ready for use.${NC}"
        echo -e "${GREEN}==========================================${NC}"
    else
        echo -e "${RED}==========================================${NC}"
        echo -e "${RED}⚠️  Smoke tests failed!${NC}"
        echo -e "${RED}Please check the output above and fix issues.${NC}"
        echo -e "${RED}==========================================${NC}"
    fi
    
    exit $exit_code
}

# Execute main function with all arguments
main "$@"
