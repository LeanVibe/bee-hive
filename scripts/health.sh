#!/bin/bash

# LeanVibe Agent Hive 2.0 - Comprehensive Health Check Script
# Validates system state and dependencies for troubleshooting

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HEALTH_LOG="${PROJECT_ROOT}/health-check.log"
PYTHON_VERSION="3.11"
MIN_DOCKER_VERSION="20.10.0"

# Health check results
CHECKS_TOTAL=0
CHECKS_PASSED=0
CHECKS_FAILED=0
CHECKS_WARNINGS=0

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to print check header
print_check() {
    local title=$1
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
    echo ""
    print_status "$BLUE" "🔍 [$CHECKS_TOTAL] Checking $title..."
}

# Function to print success
print_success() {
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
    print_status "$GREEN" "  ✅ $1"
}

# Function to print warning
print_warning() {
    CHECKS_WARNINGS=$((CHECKS_WARNINGS + 1))
    print_status "$YELLOW" "  ⚠️  $1"
}

# Function to print error
print_error() {
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
    print_status "$RED" "  ❌ $1"
}

# Function to log command output
log_command() {
    echo "=== $(date): $1 ===" >> "$HEALTH_LOG"
    eval "$1" >> "$HEALTH_LOG" 2>&1
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to compare versions
version_ge() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# Function to get container status
get_container_status() {
    local container_name=$1
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "$container_name"; then
        docker ps --format "table {{.Names}}\t{{.Status}}" | grep "$container_name" | awk '{print $2}'
    else
        echo "not_running"
    fi
}

# Function to test database connection
test_database_connection() {
    local env_file="${PROJECT_ROOT}/.env.local"
    
    if [[ ! -f "$env_file" ]]; then
        print_error "Environment file not found: $env_file"
        return 1
    fi
    
    # Source environment variables
    set -a
    source "$env_file" 2>/dev/null || true
    set +a
    
    # Extract database connection details
    local db_host=$(echo "$DATABASE_URL" | sed -n 's/.*@\([^:]*\):.*/\1/p')
    local db_port=$(echo "$DATABASE_URL" | sed -n 's/.*:\([0-9]*\)\/.*/\1/p')
    local db_name=$(echo "$DATABASE_URL" | sed -n 's/.*\/\([^?]*\).*/\1/p')
    local db_user=$(echo "$DATABASE_URL" | sed -n 's/.*\/\/\([^:]*\):.*/\1/p')
    
    # Test connection using docker
    if docker compose exec -T postgres pg_isready -U "$db_user" -d "$db_name" >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to test Redis connection
test_redis_connection() {
    if docker compose exec -T redis redis-cli ping >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to test API endpoint
test_api_endpoint() {
    local endpoint=$1
    local expected_status=${2:-200}
    
    local response_code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:8000$endpoint" 2>/dev/null || echo "000")
    
    if [[ "$response_code" == "$expected_status" ]]; then
        return 0
    else
        echo "HTTP $response_code" >&2
        return 1
    fi
}

# Main health checks

check_system_requirements() {
    print_check "System Requirements"
    
    # Check Python
    if command_exists python3; then
        local python_version=$(python3 --version | cut -d' ' -f2)
        if version_ge "$python_version" "$PYTHON_VERSION"; then
            print_success "Python $python_version installed"
        else
            print_error "Python $python_version installed, but $PYTHON_VERSION+ required"
        fi
    else
        print_error "Python 3 not installed"
    fi
    
    # Check Docker
    if command_exists docker; then
        local docker_version=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        if version_ge "$docker_version" "$MIN_DOCKER_VERSION"; then
            print_success "Docker $docker_version installed"
        else
            print_error "Docker $docker_version installed, but $MIN_DOCKER_VERSION+ required"
        fi
    else
        print_error "Docker not installed"
    fi
    
    # Check Docker Compose
    if docker compose version >/dev/null 2>&1; then
        local compose_version=$(docker compose version --short)
        print_success "Docker Compose $compose_version installed"
    elif command_exists docker-compose; then
        local compose_version=$(docker-compose --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        print_success "Docker Compose $compose_version installed (legacy)"
    else
        print_error "Docker Compose not installed"
    fi
    
    # Check Git
    if command_exists git; then
        local git_version=$(git --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        print_success "Git $git_version installed"
    else
        print_error "Git not installed"
    fi
    
    # Check tmux
    if command_exists tmux; then
        local tmux_version=$(tmux -V | grep -oE '[0-9]+\.[0-9]+' | head -1)
        print_success "tmux $tmux_version installed"
    else
        print_warning "tmux not installed (optional)"
    fi
}

check_project_structure() {
    print_check "Project Structure"
    
    # Check key directories
    local required_dirs=("app" "migrations" "tests" "docs" "frontend")
    for dir in "${required_dirs[@]}"; do
        if [[ -d "${PROJECT_ROOT}/$dir" ]]; then
            print_success "Directory exists: $dir"
        else
            print_error "Missing directory: $dir"
        fi
    done
    
    # Check key files
    local required_files=("pyproject.toml" "docker-compose.yml" "alembic.ini" "app/main.py")
    for file in "${required_files[@]}"; do
        if [[ -f "${PROJECT_ROOT}/$file" ]]; then
            print_success "File exists: $file"
        else
            print_error "Missing file: $file"
        fi
    done
}

check_environment_config() {
    print_check "Environment Configuration"
    
    local env_file="${PROJECT_ROOT}/.env.local"
    
    if [[ -f "$env_file" ]]; then
        print_success "Environment file exists: .env.local"
        
        # Check for required variables
        local required_vars=("SECRET_KEY" "DATABASE_URL" "REDIS_URL" "ANTHROPIC_API_KEY" "OPENAI_API_KEY")
        
        source "$env_file" 2>/dev/null || true
        
        for var in "${required_vars[@]}"; do
            if [[ -n "${!var:-}" ]]; then
                if [[ "${!var}" == *"your_"*"_key_here"* ]]; then
                    print_warning "$var is set but contains placeholder value"
                else
                    print_success "$var is configured"
                fi
            else
                print_error "$var is not set"
            fi
        done
    else
        print_error "Environment file not found: $env_file"
        print_status "$CYAN" "Run ./setup.sh to create environment configuration"
    fi
}

check_virtual_environment() {
    print_check "Python Virtual Environment"
    
    local venv_path="${PROJECT_ROOT}/venv"
    
    if [[ -d "$venv_path" ]]; then
        print_success "Virtual environment exists"
        
        # Check if it's activated
        if [[ "${VIRTUAL_ENV:-}" == "$venv_path" ]]; then
            print_success "Virtual environment is activated"
        else
            print_warning "Virtual environment exists but not activated"
            print_status "$CYAN" "Run: source venv/bin/activate"
        fi
        
        # Check key packages
        local packages=("fastapi" "uvicorn" "sqlalchemy" "redis" "anthropic")
        source "$venv_path/bin/activate" 2>/dev/null
        
        for package in "${packages[@]}"; do
            if pip show "$package" >/dev/null 2>&1; then
                local version=$(pip show "$package" | grep Version | cut -d' ' -f2)
                print_success "$package $version installed"
            else
                print_error "$package not installed"
            fi
        done
    else
        print_error "Virtual environment not found"
        print_status "$CYAN" "Run ./setup.sh to create virtual environment"
    fi
}

check_docker_services() {
    print_check "Docker Services"
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running"
        return
    fi
    
    print_success "Docker daemon is running"
    
    # Check individual services
    local services=("leanvibe_postgres" "leanvibe_redis")
    
    for service in "${services[@]}"; do
        local status=$(get_container_status "$service")
        case $status in
            "Up"*)
                print_success "$service: $status"
                ;;
            "not_running")
                print_error "$service: Not running"
                ;;
            *)
                print_warning "$service: $status"
                ;;
        esac
    done
}

check_database_connectivity() {
    print_check "Database Connectivity"
    
    if test_database_connection; then
        print_success "PostgreSQL connection successful"
        
        # Check database tables
        if docker compose exec -T postgres psql -U leanvibe_user -d leanvibe_agent_hive -c "\dt" >/dev/null 2>&1; then
            local table_count=$(docker compose exec -T postgres psql -U leanvibe_user -d leanvibe_agent_hive -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE';" 2>/dev/null | tr -d ' \n' || echo "0")
            if [[ "$table_count" -gt 0 ]]; then
                print_success "Database has $table_count tables (migrations applied)"
            else
                print_warning "Database is empty (migrations may not be applied)"
                print_status "$CYAN" "Run: alembic upgrade head"
            fi
        else
            print_warning "Could not check database tables"
        fi
    else
        print_error "PostgreSQL connection failed"
        print_status "$CYAN" "Check if postgres container is running: docker compose ps"
    fi
}

check_redis_connectivity() {
    print_check "Redis Connectivity"
    
    if test_redis_connection; then
        print_success "Redis connection successful"
        
        # Check Redis info
        local redis_info=$(docker compose exec -T redis redis-cli info memory 2>/dev/null || echo "")
        if [[ -n "$redis_info" ]]; then
            local memory_used=$(echo "$redis_info" | grep "used_memory_human" | cut -d: -f2 | tr -d '\r')
            print_success "Redis memory usage: $memory_used"
        fi
    else
        print_error "Redis connection failed"
        print_status "$CYAN" "Check if redis container is running: docker compose ps"
    fi
}

check_application_health() {
    print_check "Application Health"
    
    # Check if API is running
    if test_api_endpoint "/health"; then
        print_success "API health endpoint responding"
        
        # Get detailed health status
        local health_response=$(curl -s "http://localhost:8000/health" 2>/dev/null || echo '{"status":"unknown"}')
        local overall_status=$(echo "$health_response" | grep -o '"status":"[^"]*"' | cut -d'"' -f4 2>/dev/null || echo "unknown")
        
        case $overall_status in
            "healthy")
                print_success "Overall system status: $overall_status"
                ;;
            "degraded")
                print_warning "Overall system status: $overall_status"
                ;;
            *)
                print_error "Overall system status: $overall_status"
                ;;
        esac
    else
        print_error "API health endpoint not responding"
        print_status "$CYAN" "Check if the application is running: ./start.sh"
    fi
    
    # Check other endpoints
    local endpoints=("/status" "/metrics")
    for endpoint in "${endpoints[@]}"; do
        if test_api_endpoint "$endpoint"; then
            print_success "Endpoint $endpoint responding"
        else
            print_warning "Endpoint $endpoint not responding"
        fi
    done
}

check_ports() {
    print_check "Port Availability"
    
    local ports=("5432:PostgreSQL" "6380:Redis" "8000:API" "3000:Frontend" "9090:Prometheus")
    
    for port_info in "${ports[@]}"; do
        local port=$(echo "$port_info" | cut -d: -f1)
        local service=$(echo "$port_info" | cut -d: -f2)
        
        if command_exists netstat; then
            if netstat -tuln 2>/dev/null | grep -q ":$port "; then
                print_success "Port $port ($service) is in use"
            else
                print_warning "Port $port ($service) is not in use"
            fi
        elif command_exists lsof; then
            if lsof -i ":$port" >/dev/null 2>&1; then
                print_success "Port $port ($service) is in use"
            else
                print_warning "Port $port ($service) is not in use"
            fi
        else
            print_warning "Cannot check port $port (no netstat or lsof available)"
        fi
    done
}

check_disk_space() {
    print_check "Disk Space"
    
    local available_space=$(df -h "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    local available_bytes=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    
    print_success "Available disk space: $available_space"
    
    # Warn if less than 5GB available
    if [[ $available_bytes -lt 5242880 ]]; then  # 5GB in KB
        print_warning "Low disk space detected"
    fi
    
    # Check Docker space usage
    if command_exists docker; then
        local docker_info=$(docker system df 2>/dev/null || echo "")
        if [[ -n "$docker_info" ]]; then
            print_success "Docker system space usage available in logs"
            log_command "docker system df"
        fi
    fi
}

# Function to provide troubleshooting suggestions
provide_troubleshooting() {
    echo ""
    print_status "$PURPLE" "🔧 Troubleshooting Guide"
    print_status "$PURPLE" "======================="
    echo ""
    
    if [[ $CHECKS_FAILED -gt 0 ]]; then
        print_status "$RED" "❌ Issues Found ($CHECKS_FAILED failures)"
        echo ""
        print_status "$YELLOW" "Common Solutions:"
        print_status "$NC" "1. Re-run setup script: ./setup.sh"
        print_status "$NC" "2. Check Docker is running: docker info"
        print_status "$NC" "3. Start services: docker compose up -d postgres redis"
        print_status "$NC" "4. Apply migrations: alembic upgrade head"
        print_status "$NC" "5. Update API keys in .env.local"
        echo ""
    fi
    
    if [[ $CHECKS_WARNINGS -gt 0 ]]; then
        print_status "$YELLOW" "⚠️  Warnings ($CHECKS_WARNINGS warnings)"
        echo ""
        print_status "$YELLOW" "Recommended Actions:"
        print_status "$NC" "1. Update placeholder API keys in .env.local"
        print_status "$NC" "2. Activate virtual environment: source venv/bin/activate"
        print_status "$NC" "3. Start all services: ./start.sh"
        echo ""
    fi
    
    print_status "$CYAN" "📚 Additional Resources:"
    print_status "$NC" "• Setup guide: docs/GETTING_STARTED.md"
    print_status "$NC" "• Developer guide: docs/DEVELOPER_GUIDE.md"
    print_status "$NC" "• Troubleshooting: docs/TROUBLESHOOTING_GUIDE_COMPREHENSIVE.md"
    print_status "$NC" "• Logs: $HEALTH_LOG"
}

# Function to print summary
print_summary() {
    echo ""
    print_status "$PURPLE" "📊 Health Check Summary"
    print_status "$PURPLE" "======================"
    echo ""
    print_status "$GREEN" "✅ Passed:  $CHECKS_PASSED"
    print_status "$YELLOW" "⚠️  Warnings: $CHECKS_WARNINGS"
    print_status "$RED" "❌ Failed:  $CHECKS_FAILED"
    print_status "$BLUE" "📊 Total:   $CHECKS_TOTAL"
    echo ""
    
    local health_percentage=$(( (CHECKS_PASSED * 100) / CHECKS_TOTAL ))
    
    if [[ $CHECKS_FAILED -eq 0 ]]; then
        if [[ $CHECKS_WARNINGS -eq 0 ]]; then
            print_status "$GREEN" "🎉 System Health: Excellent ($health_percentage%)"
        else
            print_status "$YELLOW" "👍 System Health: Good ($health_percentage%) with minor issues"
        fi
    else
        if [[ $health_percentage -ge 70 ]]; then
            print_status "$YELLOW" "⚠️  System Health: Fair ($health_percentage%) - needs attention"
        else
            print_status "$RED" "❌ System Health: Poor ($health_percentage%) - requires fixes"
        fi
    fi
}

# Main function
main() {
    # Clear log file
    > "$HEALTH_LOG"
    
    print_status "$PURPLE" "🏥 LeanVibe Agent Hive 2.0 - Health Check"
    print_status "$PURPLE" "========================================"
    print_status "$CYAN" "Checking system health and dependencies..."
    
    # Change to project root directory
    cd "$PROJECT_ROOT"
    
    # Run all health checks
    check_system_requirements
    check_project_structure
    check_environment_config
    check_virtual_environment
    check_docker_services
    check_database_connectivity
    check_redis_connectivity
    check_application_health
    check_ports
    check_disk_space
    
    # Print results
    print_summary
    provide_troubleshooting
    
    # Exit with appropriate code
    if [[ $CHECKS_FAILED -gt 0 ]]; then
        exit 1
    elif [[ $CHECKS_WARNINGS -gt 0 ]]; then
        exit 2
    else
        exit 0
    fi
}

# Run main function
main "$@"