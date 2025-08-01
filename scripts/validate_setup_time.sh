#!/bin/bash
# Setup Time Validation Script
# 
# Measures and validates setup time claims with professional methodology
# Based on expert recommendations from Gemini CLI for enterprise-grade quality

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMP_TEST_DIR=""
MEASUREMENTS_FILE="$PROJECT_ROOT/scratchpad/setup_time_measurements.json"
NUM_RUNS=5
BASELINE_HARDWARE="$(uname -m) $(uname -s) $(nproc 2>/dev/null || echo 'unknown') cores"

# Cleanup function
cleanup() {
    if [[ -n "$TEMP_TEST_DIR" && -d "$TEMP_TEST_DIR" ]]; then
        cd "$PROJECT_ROOT"
        echo "🧹 Cleaning up test directory: $TEMP_TEST_DIR"
        rm -rf "$TEMP_TEST_DIR"
    fi
}
trap cleanup EXIT

# Logging functions
log_info() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO] $*"
}

log_error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $*" >&2
}

# JSON output functions
init_measurements_file() {
    mkdir -p "$(dirname "$MEASUREMENTS_FILE")"
    cat > "$MEASUREMENTS_FILE" << EOF
{
    "test_metadata": {
        "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)",
        "hostname": "$(hostname)",
        "baseline_hardware": "$BASELINE_HARDWARE",
        "num_runs": $NUM_RUNS,
        "methodology": "Multiple clean environment runs with time measurement"
    },
    "measurements": [],
    "summary": {}
}
EOF
}

add_measurement() {
    local run_number=$1
    local git_clone_time=$2
    local setup_time=$3
    local total_time=$4
    local success=$5
    local error_message=$6
    
    # Create temporary JSON entry
    local temp_entry=$(cat << EOF
{
    "run": $run_number,
    "git_clone_time": $git_clone_time,
    "setup_time": $setup_time,
    "total_time": $total_time,
    "success": $success,
    "error_message": "$error_message"
}
EOF
)
    
    # Add to measurements array using jq
    if command -v jq >/dev/null 2>&1; then
        local temp_file=$(mktemp)
        jq ".measurements += [$temp_entry]" "$MEASUREMENTS_FILE" > "$temp_file"
        mv "$temp_file" "$MEASUREMENTS_FILE"
    else
        # Fallback if jq not available
        log_info "jq not available, using basic JSON append"
        sed -i.bak 's/"measurements": \[\]/"measurements": ['"$temp_entry"']/' "$MEASUREMENTS_FILE" || true
    fi
}

calculate_summary() {
    if command -v jq >/dev/null 2>&1; then
        # Calculate statistics using jq
        jq '
        .summary = {
            "successful_runs": [.measurements[] | select(.success == true)] | length,
            "failed_runs": [.measurements[] | select(.success == false)] | length,
            "avg_total_time": ([.measurements[] | select(.success == true) | .total_time] | add / length),
            "median_total_time": ([.measurements[] | select(.success == true) | .total_time] | sort | if length % 2 == 0 then .[length/2-1:length/2+1] | add / 2 else .[length/2] end),
            "min_total_time": ([.measurements[] | select(.success == true) | .total_time] | min),
            "max_total_time": ([.measurements[] | select(.success == true) | .total_time] | max),
            "std_dev": ([.measurements[] | select(.success == true) | .total_time] as $times | ($times | add / length) as $avg | ($times | map(. - $avg | . * .) | add / length | sqrt)),
            "setup_time_claim_validation": {
                "claimed_time_seconds": 120,
                "claim_description": "< 2 minutes",
                "passes_claim": ([.measurements[] | select(.success == true) | .total_time] | max) < 120
            }
        }' "$MEASUREMENTS_FILE" > "${MEASUREMENTS_FILE}.tmp"
        mv "${MEASUREMENTS_FILE}.tmp" "$MEASUREMENTS_FILE"
    fi
}

# Validate prerequisites
check_prerequisites() {
    log_info "🔍 Checking prerequisites..."
    
    local missing_tools=()
    
    # Check for required tools
    for tool in git docker python3; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_error "Please install missing tools before running setup time validation"
        exit 1
    fi
    
    # Check Docker is running
    if ! docker info >/dev/null 2>&1; then
        log_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    
    log_info "✅ Prerequisites check passed"
}

# Run a single setup time measurement
run_single_measurement() {
    local run_number=$1
    log_info "🏃 Running measurement $run_number/$NUM_RUNS"
    
    # Create fresh test environment
    TEMP_TEST_DIR=$(mktemp -d -t "leanvibe_setup_time_test_XXXXXX")
    local project_clone_dir="$TEMP_TEST_DIR/bee-hive"
    
    # Measure git clone time
    log_info "📥 Cloning repository..."
    local clone_start_time=$(date +%s.%N)
    
    if git clone "file://$PROJECT_ROOT" "$project_clone_dir" >/dev/null 2>&1; then
        local clone_end_time=$(date +%s.%N)
        local git_clone_time=$(echo "$clone_end_time - $clone_start_time" | bc -l)
        log_info "✅ Git clone completed in ${git_clone_time}s"
    else
        log_error "❌ Git clone failed"
        add_measurement "$run_number" "0" "0" "0" "false" "Git clone failed"
        return 1
    fi
    
    # Change to project directory
    cd "$project_clone_dir"
    
    # Measure setup script time
    log_info "⚡ Running setup script..."
    local setup_start_time=$(date +%s.%N)
    
    if timeout 300 ./setup-fast.sh >/dev/null 2>&1; then
        local setup_end_time=$(date +%s.%N)
        local setup_time=$(echo "$setup_end_time - $setup_start_time" | bc -l)
        local total_time=$(echo "$git_clone_time + $setup_time" | bc -l)
        
        log_info "✅ Setup completed in ${setup_time}s (total: ${total_time}s)"
        add_measurement "$run_number" "$git_clone_time" "$setup_time" "$total_time" "true" ""
        
        # Quick health check
        sleep 10  # Allow services to start
        if curl -s http://localhost:8000/health >/dev/null 2>&1; then
            log_info "✅ Health check passed"
        else
            log_info "⚠️  Health check failed (services may still be starting)"
        fi
        
    else
        local setup_end_time=$(date +%s.%N)
        local setup_time=$(echo "$setup_end_time - $setup_start_time" | bc -l)
        local total_time=$(echo "$git_clone_time + $setup_time" | bc -l)
        
        log_error "❌ Setup failed or timed out"
        add_measurement "$run_number" "$git_clone_time" "$setup_time" "$total_time" "false" "Setup script failed or timed out"
    fi
    
    # Stop services before cleanup
    docker compose down -v >/dev/null 2>&1 || true
    
    # Clean up test directory
    cd "$PROJECT_ROOT"
    rm -rf "$TEMP_TEST_DIR"
    TEMP_TEST_DIR=""
    
    log_info "🧹 Measurement $run_number completed and cleaned up"
}

# Main function
main() {
    log_info "🚀 Starting setup time validation with $NUM_RUNS runs"
    log_info "📊 Hardware baseline: $BASELINE_HARDWARE"
    
    check_prerequisites
    init_measurements_file
    
    # Run multiple measurements
    for ((i=1; i<=NUM_RUNS; i++)); do
        run_single_measurement "$i"
        
        # Wait between runs to avoid interference
        if [[ $i -lt $NUM_RUNS ]]; then
            log_info "⏱️  Waiting 30 seconds before next run..."
            sleep 30
        fi
    done
    
    # Calculate summary statistics
    log_info "📊 Calculating summary statistics..."
    calculate_summary
    
    # Display results
    log_info "✅ Setup time validation completed!"
    log_info "📄 Results saved to: $MEASUREMENTS_FILE"
    
    if command -v jq >/dev/null 2>&1; then
        echo ""
        echo "=== SETUP TIME VALIDATION RESULTS ==="
        jq -r '
        "Successful runs: " + (.summary.successful_runs | tostring) + "/" + (.test_metadata.num_runs | tostring),
        "Average setup time: " + (.summary.avg_total_time | tostring | .[0:5]) + "s",
        "Median setup time: " + (.summary.median_total_time | tostring | .[0:5]) + "s", 
        "Min setup time: " + (.summary.min_total_time | tostring | .[0:5]) + "s",
        "Max setup time: " + (.summary.max_total_time | tostring | .[0:5]) + "s",
        "Standard deviation: " + (.summary.std_dev | tostring | .[0:5]) + "s",
        "",
        "CLAIM VALIDATION:",
        "Claimed time: " + .summary.setup_time_claim_validation.claim_description,
        "Validation result: " + (if .summary.setup_time_claim_validation.passes_claim then "✅ PASSED" else "❌ FAILED" end)
        ' "$MEASUREMENTS_FILE"
    else
        echo "Install jq to see formatted results summary"
    fi
}

# Check if bc is available for floating point arithmetic
if ! command -v bc >/dev/null 2>&1; then
    log_error "bc (calculator) is required for time calculations"
    log_error "Please install bc: brew install bc (macOS) or apt-get install bc (Ubuntu)"
    exit 1
fi

# Run main function
main "$@"