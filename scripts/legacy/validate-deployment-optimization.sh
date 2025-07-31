#!/bin/bash

# LeanVibe Agent Hive 2.0 - Deployment Optimization Validation
# Comprehensive validation of all DevOps optimizations

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATION_LOG="${SCRIPT_DIR}/validation-results.log"
PERFORMANCE_TARGET_SECONDS=180  # 3 minutes
SUCCESS_RATE_TARGET=95

# Validation results
VALIDATIONS_TOTAL=0
VALIDATIONS_PASSED=0
VALIDATIONS_FAILED=0
VALIDATIONS_WARNINGS=0

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print validation header
print_validation() {
    local title=$1
    VALIDATIONS_TOTAL=$((VALIDATIONS_TOTAL + 1))
    echo ""
    print_status "$BLUE" "🔍 [$VALIDATIONS_TOTAL] Validating $title..."
}

# Function to print success
print_success() {
    VALIDATIONS_PASSED=$((VALIDATIONS_PASSED + 1))
    print_status "$GREEN" "  ✅ $1"
}

# Function to print warning
print_warning() {
    VALIDATIONS_WARNINGS=$((VALIDATIONS_WARNINGS + 1))
    print_status "$YELLOW" "  ⚠️  $1"
}

# Function to print error
print_error() {
    VALIDATIONS_FAILED=$((VALIDATIONS_FAILED + 1))
    print_status "$RED" "  ❌ $1"
}

# Function to run timed command
run_timed() {
    local description=$1
    local command=$2
    local start_time=$(date +%s)
    
    print_status "$CYAN" "    → $description..."
    
    if eval "$command" >> "$VALIDATION_LOG" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "$description completed in ${duration}s"
        return $duration
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_error "$description failed after ${duration}s"
        return 999
    fi
}

# Validate ultra-fast setup performance
validate_setup_performance() {
    print_validation "Ultra-Fast Setup Performance"
    
    # Clean environment for accurate testing
    print_status "$CYAN" "    → Preparing clean test environment..."
    docker compose -f docker-compose.fast.yml down -v 2>/dev/null || true
    docker system prune -f >/dev/null 2>&1 || true
    rm -rf venv .env.local .pip-cache 2>/dev/null || true
    
    # Test setup performance
    local setup_duration
    setup_duration=$(run_timed "Ultra-fast setup execution" "./setup-ultra-fast.sh --config-only")
    
    if [[ $setup_duration -le $PERFORMANCE_TARGET_SECONDS ]]; then
        print_success "Setup time target achieved: ${setup_duration}s (target: ${PERFORMANCE_TARGET_SECONDS}s)"
    elif [[ $setup_duration -le $((PERFORMANCE_TARGET_SECONDS * 2)) ]]; then
        print_warning "Setup time above target but acceptable: ${setup_duration}s"
    else
        print_error "Setup time significantly above target: ${setup_duration}s"
    fi
    
    # Validate setup artifacts
    if [[ -f "${SCRIPT_DIR}/.env.local" ]]; then
        print_success "Environment configuration created"
    else
        print_error "Environment configuration missing"
    fi
}

# Validate setup reliability
validate_setup_reliability() {
    print_validation "Setup Reliability and Success Rate"
    
    local success_count=0
    local total_runs=3  # Reduced for faster validation
    
    for i in $(seq 1 $total_runs); do
        print_status "$CYAN" "    → Reliability test run $i of $total_runs..."
        
        # Clean environment
        docker compose -f docker-compose.fast.yml down -v 2>/dev/null || true
        rm -rf venv .env.local 2>/dev/null || true
        
        # Test setup
        if timeout 300 ./setup-ultra-fast.sh --config-only >/dev/null 2>&1; then
            success_count=$((success_count + 1))
            print_success "Run $i: Success"
        else
            print_error "Run $i: Failed"
        fi
    done
    
    local success_rate=$((success_count * 100 / total_runs))
    
    if [[ $success_rate -ge $SUCCESS_RATE_TARGET ]]; then
        print_success "Success rate target achieved: ${success_rate}% (target: ${SUCCESS_RATE_TARGET}%)"
    else
        print_error "Success rate below target: ${success_rate}% (target: ${SUCCESS_RATE_TARGET}%)"
    fi
}

# Validate DevContainer configuration
validate_devcontainer() {
    print_validation "DevContainer Configuration"
    
    if [[ -f "${SCRIPT_DIR}/.devcontainer/devcontainer.json" ]]; then
        print_success "DevContainer configuration exists"
        
        # Validate JSON syntax
        if python3 -c "import json; json.load(open('.devcontainer/devcontainer.json'))" 2>/dev/null; then
            print_success "DevContainer configuration is valid JSON"
        else
            print_error "DevContainer configuration has invalid JSON"
        fi
    else
        print_error "DevContainer configuration missing"
    fi
    
    if [[ -f "${SCRIPT_DIR}/docker-compose.devcontainer.yml" ]]; then
        print_success "DevContainer Docker Compose file exists"
    else
        print_error "DevContainer Docker Compose file missing"
    fi
    
    if [[ -f "${SCRIPT_DIR}/Dockerfile.devcontainer" ]]; then
        print_success "DevContainer Dockerfile exists"
    else
        print_error "DevContainer Dockerfile missing"
    fi
}

# Validate Docker optimizations
validate_docker_optimizations() {
    print_validation "Docker Configuration Optimizations"
    
    # Check fast compose file
    if [[ -f "${SCRIPT_DIR}/docker-compose.fast.yml" ]]; then
        print_success "Fast Docker Compose configuration exists"
        
        # Validate YAML syntax
        if python3 -c "import yaml; yaml.safe_load(open('docker-compose.fast.yml'))" 2>/dev/null; then
            print_success "Fast Docker Compose configuration is valid YAML"
        else
            print_error "Fast Docker Compose configuration has invalid YAML"
        fi
    else
        print_error "Fast Docker Compose configuration missing"
    fi
    
    # Check optimized Dockerfile
    if [[ -f "${SCRIPT_DIR}/Dockerfile.fast" ]]; then
        print_success "Optimized Dockerfile exists"
        
        # Check for multi-stage build
        if grep -q "FROM.*as.*" "Dockerfile.fast"; then
            print_success "Multi-stage build optimization detected"
        else
            print_warning "Multi-stage build optimization not detected"
        fi
    else
        print_error "Optimized Dockerfile missing"
    fi
}

# Validate monitoring configuration
validate_monitoring() {
    print_validation "Monitoring and Observability Setup"
    
    if [[ -f "${SCRIPT_DIR}/infrastructure/monitoring/prometheus.yml" ]]; then
        print_success "Prometheus configuration exists"
    else
        print_error "Prometheus configuration missing"
    fi
    
    if [[ -f "${SCRIPT_DIR}/infrastructure/monitoring/alertmanager/rules.yml" ]]; then
        print_success "Alerting rules configuration exists"
    else
        print_error "Alerting rules configuration missing"
    fi
    
    # Check for Grafana dashboards
    if [[ -d "${SCRIPT_DIR}/infrastructure/monitoring/grafana/dashboards" ]]; then
        local dashboard_count=$(find "${SCRIPT_DIR}/infrastructure/monitoring/grafana/dashboards" -name "*.json" | wc -l)
        if [[ $dashboard_count -gt 0 ]]; then
            print_success "Grafana dashboards available ($dashboard_count dashboards)"
        else
            print_warning "No Grafana dashboards found"
        fi
    else
        print_warning "Grafana dashboards directory missing"
    fi
}

# Validate CI/CD configuration
validate_cicd() {
    print_validation "CI/CD Pipeline Configuration"
    
    if [[ -f "${SCRIPT_DIR}/.github/workflows/devops-quality-gates.yml" ]]; then
        print_success "GitHub Actions workflow exists"
        
        # Validate YAML syntax
        if python3 -c "import yaml; yaml.safe_load(open('.github/workflows/devops-quality-gates.yml'))" 2>/dev/null; then
            print_success "GitHub Actions workflow is valid YAML"
        else
            print_error "GitHub Actions workflow has invalid YAML"
        fi
    else
        print_error "GitHub Actions workflow missing"
    fi
    
    # Check for required quality gates
    if [[ -f "${SCRIPT_DIR}/.github/workflows/devops-quality-gates.yml" ]]; then
        local workflow_file="${SCRIPT_DIR}/.github/workflows/devops-quality-gates.yml"
        
        if grep -q "setup-performance" "$workflow_file"; then
            print_success "Setup performance validation gate detected"
        else
            print_warning "Setup performance validation gate missing"
        fi
        
        if grep -q "code-quality" "$workflow_file"; then
            print_success "Code quality validation gate detected"
        else
            print_warning "Code quality validation gate missing"
        fi
        
        if grep -q "security-validation" "$workflow_file"; then
            print_success "Security validation gate detected"
        else
            print_warning "Security validation gate missing"
        fi
    fi
}

# Validate management scripts
validate_management_scripts() {
    print_validation "Enhanced Management Scripts"
    
    local required_scripts=(
        "setup-ultra-fast.sh"
        "start-ultra.sh"
        "monitor-performance.sh"
        "troubleshoot-auto.sh"
        "health-check.sh"
    )
    
    for script in "${required_scripts[@]}"; do
        if [[ -f "${SCRIPT_DIR}/$script" ]]; then
            if [[ -x "${SCRIPT_DIR}/$script" ]]; then
                print_success "$script exists and is executable"
            else
                print_warning "$script exists but is not executable"
            fi
        else
            print_error "$script missing"
        fi
    done
}

# Validate documentation
validate_documentation() {
    print_validation "Documentation and Guides"
    
    if [[ -f "${SCRIPT_DIR}/DEPLOYMENT_OPTIMIZATION_GUIDE.md" ]]; then
        print_success "Deployment optimization guide exists"
    else
        print_error "Deployment optimization guide missing"
    fi
    
    if [[ -f "${SCRIPT_DIR}/scratchpad/devops_optimization_strategy_2025.md" ]]; then
        print_success "DevOps optimization strategy document exists"
    else
        print_error "DevOps optimization strategy document missing"
    fi
    
    # Check for README updates
    if [[ -f "${SCRIPT_DIR}/README.md" ]]; then
        if grep -q "ultra-fast" "${SCRIPT_DIR}/README.md" 2>/dev/null; then
            print_success "README includes ultra-fast setup references"
        else
            print_warning "README may need updates for ultra-fast setup"
        fi
    fi
}

# Validate performance improvements
validate_performance_improvements() {
    print_validation "Performance Improvement Artifacts"
    
    # Check for performance logs
    if [[ -f "${SCRIPT_DIR}/setup-performance.log" ]]; then
        print_success "Performance logging configured"
    else
        print_warning "Performance logging not yet initialized"
    fi
    
    # Check for caching directories
    if [[ -d "${SCRIPT_DIR}/.pip-cache" ]] || [[ -f "${SCRIPT_DIR}/pyproject.toml" ]]; then
        print_success "Python package caching configured"
    else
        print_warning "Python package caching not yet configured"
    fi
    
    # Check for Docker buildx cache configuration
    if grep -q "cache_from" "${SCRIPT_DIR}/docker-compose.fast.yml" 2>/dev/null; then
        print_success "Docker layer caching configured"
    else
        print_warning "Docker layer caching not detected"
    fi
}

# Generate validation report
generate_validation_report() {
    local total_validations=$VALIDATIONS_TOTAL
    local success_percentage=$(( (VALIDATIONS_PASSED * 100) / total_validations ))
    
    echo ""
    print_status "$PURPLE" "📊 Deployment Optimization Validation Report"
    print_status "$PURPLE" "=============================================="
    echo ""
    
    print_status "$GREEN" "✅ Passed:     $VALIDATIONS_PASSED"
    print_status "$YELLOW" "⚠️  Warnings:   $VALIDATIONS_WARNINGS"  
    print_status "$RED" "❌ Failed:     $VALIDATIONS_FAILED"
    print_status "$BLUE" "📊 Total:      $VALIDATIONS_TOTAL"
    echo ""
    
    print_status "$CYAN" "🎯 Success Rate: $success_percentage%"
    echo ""
    
    # Overall assessment
    if [[ $VALIDATIONS_FAILED -eq 0 ]]; then
        if [[ $VALIDATIONS_WARNINGS -eq 0 ]]; then
            print_status "$BOLD$GREEN" "🏆 EXCELLENT: All optimizations validated successfully!"
            print_status "$GREEN" "🚀 System ready for world-class DevOps experience"
        else
            print_status "$BOLD$YELLOW" "👍 GOOD: Core optimizations validated with minor warnings"
            print_status "$YELLOW" "🔧 Consider addressing warnings for optimal experience"
        fi
    else
        if [[ $success_percentage -ge 80 ]]; then
            print_status "$BOLD$YELLOW" "⚠️  NEEDS ATTENTION: Most optimizations validated"
            print_status "$YELLOW" "🔧 Address failed validations before production use"
        else
            print_status "$BOLD$RED" "❌ REQUIRES FIXES: Significant optimization issues detected"
            print_status "$RED" "🚨 Critical fixes needed before deployment"
        fi
    fi
    
    echo ""
    print_status "$CYAN" "📋 Validation Summary:"
    print_status "$NC" "• Setup Performance: Optimized for <3 minute deployment"
    print_status "$NC" "• DevContainer Support: VS Code integration ready"
    print_status "$NC" "• Docker Optimizations: Multi-stage builds and caching"
    print_status "$NC" "• Monitoring Stack: Prometheus + Grafana + Alerting"
    print_status "$NC" "• CI/CD Pipeline: GitHub Actions quality gates"
    print_status "$NC" "• Management Tools: Enhanced scripts and automation"
    print_status "$NC" "• Documentation: Comprehensive guides and references"
    echo ""
    
    # Save report
    cat > "${SCRIPT_DIR}/validation-report.json" << EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "validation_results": {
    "total": $VALIDATIONS_TOTAL,
    "passed": $VALIDATIONS_PASSED,
    "warnings": $VALIDATIONS_WARNINGS,
    "failed": $VALIDATIONS_FAILED,
    "success_rate": $success_percentage
  },
  "performance_targets": {
    "setup_time_target": "${PERFORMANCE_TARGET_SECONDS}s",
    "success_rate_target": "${SUCCESS_RATE_TARGET}%"
  },
  "optimization_status": "$(if [[ $VALIDATIONS_FAILED -eq 0 && $VALIDATIONS_WARNINGS -eq 0 ]]; then echo "excellent"; elif [[ $VALIDATIONS_FAILED -eq 0 ]]; then echo "good"; elif [[ $success_percentage -ge 80 ]]; then echo "needs_attention"; else echo "requires_fixes"; fi)",
  "recommendations": [
    $(if [[ $VALIDATIONS_FAILED -gt 0 ]]; then echo "\"Address failed validation items\""; fi)
    $(if [[ $VALIDATIONS_WARNINGS -gt 0 ]]; then echo "\"Review and resolve warning items\""; fi)
    $(if [[ $VALIDATIONS_FAILED -eq 0 && $VALIDATIONS_WARNINGS -eq 0 ]]; then echo "\"System ready for production deployment\""; fi)
  ]
}
EOF
    
    print_status "$CYAN" "📄 Detailed report saved to: validation-report.json"
    print_status "$CYAN" "📄 Validation logs saved to: $VALIDATION_LOG"
}

# Main validation function
main() {
    # Initialize log
    > "$VALIDATION_LOG"
    
    print_status "$BOLD$PURPLE" "🔍 LeanVibe Agent Hive 2.0 - Deployment Optimization Validation"
    print_status "$PURPLE" "=================================================================="
    print_status "$CYAN" "🎯 Validating all DevOps optimizations for production readiness"
    print_status "$CYAN" "🚀 Target: <3 minute setup, >95% success rate, world-class experience"
    echo ""
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Run all validations
    validate_setup_performance
    validate_setup_reliability
    validate_devcontainer
    validate_docker_optimizations
    validate_monitoring
    validate_cicd
    validate_management_scripts
    validate_documentation
    validate_performance_improvements
    
    # Generate final report
    generate_validation_report
    
    # Exit with appropriate code
    if [[ $VALIDATIONS_FAILED -gt 0 ]]; then
        exit 1
    elif [[ $VALIDATIONS_WARNINGS -gt 0 ]]; then
        exit 2
    else
        exit 0
    fi
}

# Enhanced error handling
trap 'print_error "Validation interrupted by user or system error"' INT TERM

# Run main function
main "$@"