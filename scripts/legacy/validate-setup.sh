#!/bin/bash

# LeanVibe Agent Hive 2.0 - Setup Validation Script
# Quick validation tool for common setup issues

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATION_LOG="${SCRIPT_DIR}/validation.log"

# Validation results
VALIDATIONS_TOTAL=0
VALIDATIONS_PASSED=0
VALIDATIONS_FAILED=0

print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

print_validation() {
    local title=$1
    VALIDATIONS_TOTAL=$((VALIDATIONS_TOTAL + 1))
    print_status "$BLUE" "🔍 [$VALIDATIONS_TOTAL] $title"
}

print_pass() {
    VALIDATIONS_PASSED=$((VALIDATIONS_PASSED + 1))
    print_status "$GREEN" "  ✅ $1"
}

print_fail() {
    VALIDATIONS_FAILED=$((VALIDATIONS_FAILED + 1))
    print_status "$RED" "  ❌ $1"
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

validate_system_requirements() {
    print_validation "System Requirements"
    
    # Check Python
    if command_exists python3; then
        local python_version=$(python3 --version | cut -d' ' -f2)
        if [[ $(echo "$python_version" | cut -d. -f1-2) = "3.11" ]] || [[ $(echo "$python_version" | cut -d. -f1-2) = "3.12" ]]; then
            print_pass "Python $python_version (compatible)"
        else
            print_fail "Python $python_version (requires 3.11+)"
        fi
    else
        print_fail "Python 3 not found"
    fi
    
    # Check Docker
    if command_exists docker; then
        if docker info >/dev/null 2>&1; then
            print_pass "Docker is running"
        else
            print_fail "Docker is installed but not running"
        fi
    else
        print_fail "Docker not installed"
    fi
    
    # Check Docker Compose
    if docker compose version >/dev/null 2>&1; then
        print_pass "Docker Compose v2 available"
    elif command_exists docker-compose; then
        print_pass "Docker Compose v1 available (legacy)"
    else
        print_fail "Docker Compose not available"
    fi
}

validate_project_files() {
    print_validation "Project Structure"
    
    local required_files=(
        "pyproject.toml:Project configuration"
        "docker-compose.yml:Docker services"
        "app/main.py:Application entry point"
        "setup.sh:Setup script"
        "health-check.sh:Health check script"
        "Makefile:Development commands"
    )
    
    for file_info in "${required_files[@]}"; do
        local file=$(echo "$file_info" | cut -d: -f1)
        local description=$(echo "$file_info" | cut -d: -f2)
        
        if [[ -f "${SCRIPT_DIR}/$file" ]]; then
            print_pass "$description ($file)"
        else
            print_fail "$description missing ($file)"
        fi
    done
}

validate_environment() {
    print_validation "Environment Configuration"
    
    local env_file="${SCRIPT_DIR}/.env.local"
    
    if [[ -f "$env_file" ]]; then
        print_pass "Environment file exists (.env.local)"
        
        # Check for placeholder values
        if grep -q "your_.*_key_here" "$env_file" 2>/dev/null; then
            print_fail "Environment file contains placeholder API keys"
        else
            print_pass "No placeholder values detected"
        fi
        
        # Check required variables
        local required_vars=("SECRET_KEY" "DATABASE_URL" "REDIS_URL")
        source "$env_file" 2>/dev/null
        
        for var in "${required_vars[@]}"; do
            if [[ -n "${!var:-}" ]]; then
                print_pass "$var is set"
            else
                print_fail "$var is not set"
            fi
        done
    else
        print_fail "Environment file not found (.env.local)"
    fi
}

validate_virtual_environment() {
    print_validation "Python Virtual Environment"
    
    if [[ -d "${SCRIPT_DIR}/venv" ]]; then
        print_pass "Virtual environment directory exists"
        
        if [[ -f "${SCRIPT_DIR}/venv/bin/activate" ]]; then
            print_pass "Virtual environment is properly configured"
            
            # Check if key packages are installed
            source "${SCRIPT_DIR}/venv/bin/activate" 2>/dev/null
            local packages=("fastapi" "uvicorn" "sqlalchemy")
            
            for package in "${packages[@]}"; do
                if pip show "$package" >/dev/null 2>&1; then
                    print_pass "$package is installed"
                else
                    print_fail "$package is not installed"
                fi
            done
        else
            print_fail "Virtual environment activation script missing"
        fi
    else
        print_fail "Virtual environment not found"
    fi
}

validate_docker_services() {
    print_validation "Docker Services Status"
    
    if docker compose ps >/dev/null 2>&1; then
        local services=("postgres" "redis")
        
        for service in "${services[@]}"; do
            if docker compose ps --services --filter "status=running" | grep -q "$service"; then
                print_pass "$service service is running"
            else
                print_fail "$service service is not running"
            fi
        done
    else
        print_fail "Cannot check Docker services (docker compose not available)"
    fi
}

validate_api_access() {
    print_validation "API Accessibility"
    
    # Check if API port is accessible
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:8000/health" | grep -q "200"; then
        print_pass "API health endpoint is accessible"
    else
        print_fail "API health endpoint is not accessible"
    fi
}

provide_quick_fixes() {
    echo ""
    print_status "$PURPLE" "🔧 Quick Fixes"
    print_status "$PURPLE" "=============="
    echo ""
    
    if [[ $VALIDATIONS_FAILED -gt 0 ]]; then
        print_status "$YELLOW" "Try these commands to resolve issues:"
        echo ""
        
        if [[ ! -f "${SCRIPT_DIR}/.env.local" ]] || grep -q "your_.*_key_here" "${SCRIPT_DIR}/.env.local" 2>/dev/null; then
            print_status "$NC" "1. Fix environment configuration:"
            print_status "$GREEN" "   ./setup.sh"
            print_status "$GREEN" "   # Then edit .env.local with your API keys"
            echo ""
        fi
        
        if [[ ! -d "${SCRIPT_DIR}/venv" ]]; then
            print_status "$NC" "2. Create virtual environment:"
            print_status "$GREEN" "   python3 -m venv venv"
            print_status "$GREEN" "   source venv/bin/activate"
            print_status "$GREEN" "   pip install -e .[dev,monitoring,ai-extended]"
            echo ""
        fi
        
        if ! docker compose ps --services --filter "status=running" | grep -q "postgres\\|redis"; then
            print_status "$NC" "3. Start required services:"
            print_status "$GREEN" "   docker compose up -d postgres redis"
            echo ""
        fi
        
        print_status "$NC" "4. Run comprehensive health check:"
        print_status "$GREEN" "   ./health-check.sh"
        echo ""
        
        print_status "$NC" "5. Start the application:"
        print_status "$GREEN" "   make dev"
        print_status "$GREEN" "   # or"
        print_status "$GREEN" "   ./start.sh"
    else
        print_status "$GREEN" "🎉 All validations passed! Your setup looks good."
        print_status "$BLUE" "Next steps:"
        print_status "$NC" "  • Run './health-check.sh' for detailed system health"
        print_status "$NC" "  • Start the application with 'make dev' or './start.sh'"
        print_status "$NC" "  • Access the API at http://localhost:8000"
    fi
}

print_summary() {
    echo ""
    print_status "$PURPLE" "📊 Validation Summary"
    print_status "$PURPLE" "===================="
    echo ""
    print_status "$GREEN" "✅ Passed: $VALIDATIONS_PASSED"
    print_status "$RED" "❌ Failed: $VALIDATIONS_FAILED"
    print_status "$BLUE" "📊 Total:  $VALIDATIONS_TOTAL"
    echo ""
    
    local success_rate=$(( (VALIDATIONS_PASSED * 100) / VALIDATIONS_TOTAL ))
    
    if [[ $VALIDATIONS_FAILED -eq 0 ]]; then
        print_status "$GREEN" "🎉 Setup Status: Ready for Development! ($success_rate%)"
    elif [[ $success_rate -ge 70 ]]; then
        print_status "$YELLOW" "⚠️  Setup Status: Mostly Ready ($success_rate%) - minor fixes needed"
    else
        print_status "$RED" "❌ Setup Status: Needs Attention ($success_rate%) - run setup script"
    fi
}

main() {
    # Clear log
    > "$VALIDATION_LOG"
    
    print_status "$PURPLE" "⚡ LeanVibe Agent Hive 2.0 - Quick Setup Validation"
    print_status "$PURPLE" "===================================================="
    echo ""
    
    cd "$SCRIPT_DIR"
    
    validate_system_requirements
    validate_project_files
    validate_environment
    validate_virtual_environment
    validate_docker_services
    validate_api_access
    
    print_summary
    provide_quick_fixes
    
    # Exit with appropriate code
    if [[ $VALIDATIONS_FAILED -gt 0 ]]; then
        exit 1
    else
        exit 0
    fi
}

main "$@"