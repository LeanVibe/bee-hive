#!/bin/bash

# LeanVibe Agent Hive 2.0 - Setup Performance Validation Script
# Measures and compares setup performance between original and optimized scripts

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_FILE="${SCRIPT_DIR}/setup-performance-results.json"
LOG_DIR="${SCRIPT_DIR}/performance-logs"

# Performance targets
TARGET_FAST_MIN=300   # 5 minutes
TARGET_FAST_MAX=900   # 15 minutes
TARGET_SUCCESS_RATE=95 # 95% success rate

# Test configurations
declare -A TEST_CONFIGS=(
    ["fresh_system"]="Complete clean setup (no cache)"
    ["cached_system"]="Setup with Docker/pip caches present"
    ["development_mode"]="Setup with development profile"
    ["minimal_setup"]="Core services only"
)

print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_header() {
    echo ""
    print_status "$BOLD$BLUE" "============================================"
    print_status "$BOLD$BLUE" "$1"
    print_status "$BOLD$BLUE" "============================================"
    echo ""
}

# Function to clean environment for fresh test
cleanup_environment() {
    local level=${1:-"partial"}
    
    print_status "$YELLOW" "🧹 Cleaning environment ($level)..."
    
    # Stop and remove containers
    docker compose -f docker-compose.yml down -v 2>/dev/null || true
    docker compose -f docker-compose.fast.yml down -v 2>/dev/null || true
    
    # Remove volumes if fresh cleanup
    if [[ "$level" == "fresh" ]]; then
        docker volume prune -f 2>/dev/null || true
        docker system prune -f 2>/dev/null || true
        
        # Remove Python virtual environment
        rm -rf "${SCRIPT_DIR}/venv" 2>/dev/null || true
        
        # Remove pip cache
        rm -rf "${SCRIPT_DIR}/.pip-cache" 2>/dev/null || true
        
        # Remove development data
        rm -rf "${SCRIPT_DIR}/dev-state" 2>/dev/null || true
    fi
    
    # Remove log files
    rm -f "${SCRIPT_DIR}/setup.log" "${SCRIPT_DIR}/setup-fast.log"
    
    print_status "$GREEN" "✅ Environment cleaned"
}

# Function to run setup with timing
run_setup_test() {
    local script_name=$1
    local test_name=$2
    local cleanup_level=${3:-"partial"}
    
    print_status "$BLUE" "🚀 Running $test_name..."
    print_status "$CYAN" "   Script: $script_name"
    print_status "$CYAN" "   Cleanup: $cleanup_level"
    
    # Cleanup before test
    cleanup_environment "$cleanup_level"
    sleep 5  # Let system settle
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    local test_log="${LOG_DIR}/${test_name}_$(date +%Y%m%d_%H%M%S).log"
    
    # Run setup with timing
    local start_time=$(date +%s)
    local exit_code=0
    
    echo "Starting $test_name at $(date)" > "$test_log"
    
    # Run the setup script
    if timeout 1800 bash "$script_name" >> "$test_log" 2>&1; then
        exit_code=0
        print_status "$GREEN" "✅ $test_name completed successfully"
    else
        exit_code=$?
        print_status "$RED" "❌ $test_name failed (exit code: $exit_code)"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    echo "Completed $test_name at $(date)" >> "$test_log"
    echo "Duration: ${minutes}m ${seconds}s" >> "$test_log"
    echo "Exit code: $exit_code" >> "$test_log"
    
    print_status "$PURPLE" "   Duration: ${minutes}m ${seconds}s"
    print_status "$PURPLE" "   Log: $test_log"
    
    # Return results as JSON-like string
    echo "{\"test\":\"$test_name\",\"script\":\"$script_name\",\"duration\":$duration,\"success\":$([ $exit_code -eq 0 ] && echo true || echo false),\"log\":\"$test_log\"}"
}

# Function to validate setup after completion
validate_setup_state() {
    local test_name=$1
    
    print_status "$CYAN" "🔍 Validating setup state for $test_name..."
    
    local validation_results=()
    
    # Check Docker containers
    if docker compose -f docker-compose.fast.yml ps | grep -q "Up"; then
        validation_results+=("docker_services:true")
        print_status "$GREEN" "  ✅ Docker services running"
    else
        validation_results+=("docker_services:false")
        print_status "$RED" "  ❌ Docker services not running"
    fi
    
    # Check database connectivity
    if docker compose -f docker-compose.fast.yml exec -T postgres pg_isready -U leanvibe_user -d leanvibe_agent_hive >/dev/null 2>&1; then
        validation_results+=("database:true")
        print_status "$GREEN" "  ✅ Database accessible"
    else
        validation_results+=("database:false")
        print_status "$RED" "  ❌ Database not accessible"
    fi
    
    # Check Redis connectivity
    if docker compose -f docker-compose.fast.yml exec -T redis redis-cli -a leanvibe_redis_pass ping >/dev/null 2>&1; then
        validation_results+=("redis:true")
        print_status "$GREEN" "  ✅ Redis accessible"
    else
        validation_results+=("redis:false")
        print_status "$RED" "  ❌ Redis not accessible"
    fi
    
    # Check Python environment
    if [[ -f "${SCRIPT_DIR}/venv/bin/activate" ]]; then
        validation_results+=("python_env:true")
        print_status "$GREEN" "  ✅ Python environment exists"
        
        # Check key packages
        source "${SCRIPT_DIR}/venv/bin/activate"
        if python -c "import fastapi, uvicorn, sqlalchemy" 2>/dev/null; then
            validation_results+=("python_packages:true")
            print_status "$GREEN" "  ✅ Key Python packages available"
        else
            validation_results+=("python_packages:false")
            print_status "$RED" "  ❌ Key Python packages missing"
        fi
    else
        validation_results+=("python_env:false")
        validation_results+=("python_packages:false")
        print_status "$RED" "  ❌ Python environment missing"
    fi
    
    # Check configuration
    if [[ -f "${SCRIPT_DIR}/.env.local" ]]; then
        validation_results+=("config:true")
        print_status "$GREEN" "  ✅ Configuration file exists"
    else
        validation_results+=("config:false")
        print_status "$RED" "  ❌ Configuration file missing"
    fi
    
    # Return validation summary
    local success_count=$(printf '%s\n' "${validation_results[@]}" | grep -c ":true" || echo 0)
    local total_count=${#validation_results[@]}
    local success_rate=$((success_count * 100 / total_count))
    
    print_status "$PURPLE" "  Validation: $success_count/$total_count passed ($success_rate%)"
    
    echo "{\"validation_rate\":$success_rate,\"checks\":{$(printf '%s\n' "${validation_results[@]}" | sed 's/:/":"/g' | sed 's/^/"/' | paste -sd ',' -)}}"
}

# Function to run comprehensive performance tests
run_performance_suite() {
    local test_results=()
    
    print_header "LEANVIBE AGENT HIVE 2.0 - SETUP PERFORMANCE VALIDATION"
    
    print_status "$CYAN" "🎯 Performance Targets:"
    print_status "$NC" "  • Fast setup: 5-15 minutes (${TARGET_FAST_MIN}-${TARGET_FAST_MAX}s)"
    print_status "$NC" "  • Success rate: ${TARGET_SUCCESS_RATE}%+"
    print_status "$NC" "  • Improvement: 50%+ time reduction vs original"
    echo ""
    
    # Test 1: Original setup (baseline)
    print_header "TEST 1: ORIGINAL SETUP (BASELINE)"
    if [[ -f "${SCRIPT_DIR}/setup.sh" ]]; then
        local original_result=$(run_setup_test "${SCRIPT_DIR}/setup.sh" "original_fresh" "fresh")
        test_results+=("$original_result")
        
        local original_validation=$(validate_setup_state "original_fresh")
    else
        print_status "$YELLOW" "⚠️  Original setup.sh not found, skipping baseline test"
    fi
    
    # Test 2: Fast setup (fresh system)
    print_header "TEST 2: FAST SETUP (FRESH SYSTEM)"
    local fast_fresh_result=$(run_setup_test "${SCRIPT_DIR}/setup-fast.sh" "fast_fresh" "fresh")
    test_results+=("$fast_fresh_result")
    
    local fast_fresh_validation=$(validate_setup_state "fast_fresh")
    
    # Test 3: Fast setup (cached system)
    print_header "TEST 3: FAST SETUP (CACHED SYSTEM)"
    local fast_cached_result=$(run_setup_test "${SCRIPT_DIR}/setup-fast.sh" "fast_cached" "partial")
    test_results+=("$fast_cached_result")
    
    local fast_cached_validation=$(validate_setup_state "fast_cached")
    
    # Test 4: Reliability test (multiple runs)
    print_header "TEST 4: RELIABILITY TEST (3 RUNS)"
    local reliability_results=()
    local success_count=0
    
    for i in {1..3}; do
        print_status "$BLUE" "🔄 Reliability run $i/3..."
        local reliability_result=$(run_setup_test "${SCRIPT_DIR}/setup-fast.sh" "reliability_run_$i" "partial")
        reliability_results+=("$reliability_result")
        
        if echo "$reliability_result" | grep -q '"success":true'; then
            success_count=$((success_count + 1))
        fi
        
        # Brief pause between runs
        sleep 10
    done
    
    local reliability_rate=$((success_count * 100 / 3))
    print_status "$PURPLE" "📊 Reliability rate: $success_count/3 ($reliability_rate%)"
    
    # Generate comprehensive report
    generate_performance_report "${test_results[@]}" "$reliability_rate"
}

# Function to generate performance report
generate_performance_report() {
    local test_results=("$@")
    local reliability_rate="${test_results[-1]}"
    unset 'test_results[-1]'  # Remove reliability rate from test results
    
    print_header "PERFORMANCE ANALYSIS REPORT"
    
    local report_data="{\"timestamp\":\"$(date -Iseconds)\",\"tests\":["
    local test_count=0
    local total_improvement=0
    local original_time=0
    local fast_fresh_time=0
    
    # Process test results
    for result in "${test_results[@]}"; do
        if [[ $test_count -gt 0 ]]; then
            report_data+=","
        fi
        report_data+="$result"
        
        # Extract timing data
        local duration=$(echo "$result" | grep -o '"duration":[0-9]*' | cut -d: -f2)
        local test_name=$(echo "$result" | grep -o '"test":"[^"]*"' | cut -d'"' -f4)
        
        if [[ "$test_name" == "original_fresh" ]]; then
            original_time=$duration
        elif [[ "$test_name" == "fast_fresh" ]]; then
            fast_fresh_time=$duration
        fi
        
        # Display individual results
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))
        local success=$(echo "$result" | grep -o '"success":[^,}]*' | cut -d: -f2)
        
        print_status "$BLUE" "📊 $test_name:"
        print_status "$NC" "   Duration: ${minutes}m ${seconds}s"
        print_status "$NC" "   Success: $success"
        
        # Check against targets
        if [[ "$test_name" == "fast_"* ]]; then
            if [[ $duration -le $TARGET_FAST_MAX ]]; then
                print_status "$GREEN" "   ✅ Within target (≤15 minutes)"
            else
                print_status "$RED" "   ❌ Exceeds target (>15 minutes)"
            fi
        fi
        
        test_count=$((test_count + 1))
    done
    
    report_data+="],\"reliability_rate\":$reliability_rate"
    
    # Calculate improvements
    if [[ $original_time -gt 0 && $fast_fresh_time -gt 0 ]]; then
        local time_saved=$((original_time - fast_fresh_time))
        local improvement_percent=$(((original_time - fast_fresh_time) * 100 / original_time))
        
        print_status "$CYAN" "📈 Performance Comparison:"
        print_status "$NC" "   Original setup: $((original_time / 60))m $((original_time % 60))s"
        print_status "$NC" "   Fast setup: $((fast_fresh_time / 60))m $((fast_fresh_time % 60))s"
        print_status "$NC" "   Time saved: $((time_saved / 60))m $((time_saved % 60))s"
        
        if [[ $improvement_percent -gt 0 ]]; then
            print_status "$GREEN" "   Improvement: ${improvement_percent}% faster"
        else
            print_status "$RED" "   Regression: $((0 - improvement_percent))% slower"
        fi
        
        report_data+=",\"improvement_percent\":$improvement_percent"
    fi
    
    report_data+="}"
    
    # Save detailed report
    echo "$report_data" | python3 -m json.tool > "$RESULTS_FILE" 2>/dev/null || echo "$report_data" > "$RESULTS_FILE"
    
    # Final assessment
    print_header "FINAL ASSESSMENT"
    
    local targets_met=0
    local total_targets=3
    
    # Check fast setup time target
    if [[ $fast_fresh_time -le $TARGET_FAST_MAX && $fast_fresh_time -ge $TARGET_FAST_MIN ]]; then
        print_status "$GREEN" "✅ Fast setup time target met (5-15 minutes)"
        targets_met=$((targets_met + 1))
    else
        print_status "$RED" "❌ Fast setup time target missed"
    fi
    
    # Check reliability target
    if [[ $reliability_rate -ge $TARGET_SUCCESS_RATE ]]; then
        print_status "$GREEN" "✅ Reliability target met (≥${TARGET_SUCCESS_RATE}%)"
        targets_met=$((targets_met + 1))
    else
        print_status "$RED" "❌ Reliability target missed ($reliability_rate% < ${TARGET_SUCCESS_RATE}%)"
    fi
    
    # Check improvement target
    if [[ $improvement_percent -ge 50 ]]; then
        print_status "$GREEN" "✅ Performance improvement target met (≥50%)"
        targets_met=$((targets_met + 1))
    else
        print_status "$RED" "❌ Performance improvement target missed ($improvement_percent% < 50%)"
    fi
    
    # Overall result
    local success_rate=$((targets_met * 100 / total_targets))
    
    if [[ $targets_met -eq $total_targets ]]; then
        print_status "$BOLD$GREEN" "🏆 ALL TARGETS ACHIEVED! Setup optimization successful."
    elif [[ $targets_met -ge 2 ]]; then
        print_status "$YELLOW" "⚠️  Most targets achieved ($targets_met/$total_targets). Minor optimization needed."
    else
        print_status "$RED" "❌ Significant optimization needed ($targets_met/$total_targets targets met)."
    fi
    
    print_status "$CYAN" "📄 Detailed report saved: $RESULTS_FILE"
    print_status "$CYAN" "📂 Test logs available in: $LOG_DIR"
}

# Function to show quick performance check
quick_performance_check() {
    print_header "QUICK PERFORMANCE CHECK"
    
    print_status "$BLUE" "🚀 Running single fast setup test..."
    
    cleanup_environment "partial"
    local start_time=$(date +%s)
    
    if timeout 1200 bash "${SCRIPT_DIR}/setup-fast.sh" >/dev/null 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))
        
        print_status "$GREEN" "✅ Setup completed in ${minutes}m ${seconds}s"
        
        if [[ $duration -le $TARGET_FAST_MAX ]]; then
            print_status "$GREEN" "🎯 Target achieved (≤15 minutes)"
        else
            print_status "$YELLOW" "⚠️  Exceeds 15-minute target"
        fi
        
        # Quick validation
        validate_setup_state "quick_check" >/dev/null
        
    else
        print_status "$RED" "❌ Setup failed or timed out"
    fi
}

# Main function
main() {
    local mode=${1:-"full"}
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    case $mode in
        "quick")
            quick_performance_check
            ;;
        "full")
            run_performance_suite
            ;;
        "cleanup")
            cleanup_environment "fresh"
            print_status "$GREEN" "✅ Environment cleaned"
            ;;
        *)
            print_status "$RED" "Usage: $0 [quick|full|cleanup]"
            print_status "$NC" "  quick   - Run single fast setup test"
            print_status "$NC" "  full    - Run comprehensive performance suite"
            print_status "$NC" "  cleanup - Clean environment for fresh testing"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"