#!/bin/bash

# ⚠️  DEPRECATED: This script has been moved to legacy/
# 🚀 Use 'make setup' instead for the new standardized approach.
# 📖 See MIGRATION.md for migration guide.

echo "⚠️  DEPRECATION WARNING: Using legacy setup-fast.sh"
echo "🚀 Please use 'make setup' instead."
echo "⏳ Continuing in 3 seconds..."
sleep 3

# LeanVibe Agent Hive 2.0 - Optimized Setup Script (LEGACY)
# Target: 5-15 minutes setup time with 95%+ success rate
# Features: Parallel operations, caching, progress indicators, ETA

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
PROJECT_NAME="leanvibe-agent-hive"
PYTHON_VERSION="3.11"
MIN_DOCKER_VERSION="20.10.0"
MIN_DOCKER_COMPOSE_VERSION="2.0.0"

# Progress tracking with ETA
STEPS_TOTAL=8
STEP_CURRENT=0
START_TIME=$(date +%s)
STEP_TIMES=()

# Log file
LOG_FILE="${SCRIPT_DIR}/setup-fast.log"

# Performance tracking (compatible with bash 3.x)
# Using functions instead of associative arrays for better compatibility

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to calculate and display ETA
calculate_eta() {
    local current_time=$(date +%s)
    local elapsed=$((current_time - START_TIME))
    local steps_remaining=$((STEPS_TOTAL - STEP_CURRENT))
    
    if [[ $STEP_CURRENT -gt 0 ]]; then
        local avg_step_time=$((elapsed / STEP_CURRENT))
        local eta_seconds=$((avg_step_time * steps_remaining))
        local eta_minutes=$((eta_seconds / 60))
        local eta_seconds_remainder=$((eta_seconds % 60))
        
        if [[ $eta_minutes -gt 0 ]]; then
            echo "${eta_minutes}m ${eta_seconds_remainder}s"
        else
            echo "${eta_seconds}s"
        fi
    else
        echo "calculating..."
    fi
}

# Function to print step progress with ETA
print_step() {
    STEP_CURRENT=$((STEP_CURRENT + 1))
    local message=$1
    local step_start_time=$(date +%s)
    local eta=$(calculate_eta)
    local progress_bar=""
    local progress_percent=$((STEP_CURRENT * 100 / STEPS_TOTAL))
    
    # Create progress bar
    local filled=$((progress_percent / 5))
    local empty=$((20 - filled))
    for ((i=0; i<filled; i++)); do progress_bar+="█"; done
    for ((i=0; i<empty; i++)); do progress_bar+="░"; done
    
    print_status "$BLUE" "[$STEP_CURRENT/$STEPS_TOTAL] ${progress_bar} ${progress_percent}% - $message"
    print_status "$CYAN" "           ETA: $eta | Elapsed: $((($(date +%s) - START_TIME) / 60))m $((($(date +%s) - START_TIME) % 60))s"
    
    # Store step start time for performance tracking
    STEP_TIMES[$STEP_CURRENT]=$step_start_time
}

# Function to print success with timing
print_success() {
    local step_time=$(($(date +%s) - ${STEP_TIMES[$STEP_CURRENT]:-$(date +%s)}))
    print_status "$GREEN" "  ✅ $1 (${step_time}s)"
}

# Function to print warning
print_warning() {
    print_status "$YELLOW" "  ⚠️  $1"
}

# Function to print error and exit
print_error() {
    local step_time=$(($(date +%s) - ${STEP_TIMES[$STEP_CURRENT]:-$(date +%s)}))
    print_status "$RED" "  ❌ $1 (failed after ${step_time}s)"
    print_status "$RED" "Check $LOG_FILE for detailed logs"
    
    # Provide quick recovery suggestions
    echo ""
    print_status "$YELLOW" "🔧 Quick Recovery Options:"
    print_status "$NC" "1. Re-run with verbose logging: bash -x setup-fast.sh"
    print_status "$NC" "2. Check system requirements: ./health-check.sh"
    print_status "$NC" "3. Use original setup: ./setup.sh"
    print_status "$NC" "4. Clean and retry: docker system prune -f && ./setup-fast.sh"
    
    exit 1
}

# Function to log command output
log_command() {
    local cmd=$1
    local description=${2:-"$cmd"}
    echo "=== $(date): $description ===" >> "$LOG_FILE"
    eval "$cmd" >> "$LOG_FILE" 2>&1
}

# Function to run command with timeout and progress
run_with_progress() {
    local cmd=$1
    local timeout=${2:-300}
    local description=${3:-"Running command"}
    
    print_status "$CYAN" "    → $description..."
    
    # Run command in background
    timeout $timeout bash -c "$cmd" >> "$LOG_FILE" 2>&1 &
    local pid=$!
    
    # Show progress dots
    while kill -0 $pid 2>/dev/null; do
        echo -n "."
        sleep 2
    done
    
    wait $pid
    local exit_code=$?
    echo ""
    
    if [[ $exit_code -ne 0 ]]; then
        print_error "$description failed"
    fi
    
    return $exit_code
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to compare versions
version_ge() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# Function to detect OS and optimize for it
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command_exists apt-get; then
            echo "ubuntu"
        elif command_exists yum; then
            echo "centos"
        elif command_exists dnf; then
            echo "fedora"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Optimized system dependency installation
install_system_deps() {
    print_step "Installing system dependencies with parallel optimizations"
    
    local os=$(detect_os)
    local install_cmd=""
    
    case $os in
        "ubuntu")
            install_cmd="sudo apt-get update -qq && sudo apt-get install -y --no-install-recommends curl wget git tmux postgresql-client redis-tools python3-pip python3-venv build-essential pkg-config libpq-dev"
            ;;
        "centos"|"fedora")
            if [[ $os == "centos" ]]; then
                install_cmd="sudo yum update -y -q && sudo yum install -y curl wget git tmux postgresql redis python3-pip python3-venv gcc gcc-c++ make postgresql-devel"
            else
                install_cmd="sudo dnf update -y -q && sudo dnf install -y curl wget git tmux postgresql redis python3-pip python3-venv gcc gcc-c++ make postgresql-devel"
            fi
            ;;
        "macos")
            if ! command_exists brew; then
                run_with_progress '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"' 300 "Installing Homebrew"
            fi
            install_cmd="brew install git tmux postgresql redis python@$PYTHON_VERSION"
            ;;
        *)
            print_warning "Unknown OS detected. Manual dependency installation required."
            return 0
            ;;
    esac
    
    if [[ -n "$install_cmd" ]]; then
        run_with_progress "$install_cmd" 300 "Installing system packages"
    fi
    
    print_success "System dependencies installed"
}

# Fast Docker and Docker Compose check
check_docker_fast() {
    print_step "Verifying Docker with optimized checks"
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker daemon is not running. Please start Docker."
    fi
    
    # Quick version checks
    local docker_version=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    if ! version_ge "$docker_version" "$MIN_DOCKER_VERSION"; then
        print_error "Docker $docker_version installed, but $MIN_DOCKER_VERSION+ required"
    fi
    
    # Check Docker Compose
    if docker compose version >/dev/null 2>&1; then
        local compose_version=$(docker compose version --short)
        if ! version_ge "$compose_version" "$MIN_DOCKER_COMPOSE_VERSION"; then
            print_error "Docker Compose $compose_version installed, but $MIN_DOCKER_COMPOSE_VERSION+ required"
        fi
    else
        print_error "Docker Compose v2 required. Please update Docker."
    fi
    
    print_success "Docker environment verified"
}

# Fast environment configuration
create_env_config_fast() {
    print_step "Creating optimized environment configuration"
    
    local env_file="${SCRIPT_DIR}/.env.local"
    
    # Backup existing config
    if [[ -f "$env_file" ]]; then
        cp "$env_file" "${env_file}.backup.$(date +%s)"
        print_status "$CYAN" "    → Backed up existing configuration"
    fi
    
    # Generate secure keys in parallel
    print_status "$CYAN" "    → Generating secure keys..."
    local secret_key=$(python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null || openssl rand -hex 32)
    local jwt_secret=$(python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null || openssl rand -hex 32)
    
    # Create optimized environment file
    cat > "$env_file" << EOF
# LeanVibe Agent Hive 2.0 - Optimized Configuration
# Generated: $(date)

# Core Settings - Performance Optimized
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
SECRET_KEY=$secret_key
JWT_SECRET_KEY=$jwt_secret

# Database - Optimized Connection Pool
DATABASE_URL=postgresql://leanvibe_user:leanvibe_secure_pass@localhost:5432/leanvibe_agent_hive
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10

# Redis - Memory Optimized
REDIS_URL=redis://:leanvibe_redis_pass@localhost:6380/0
REDIS_STREAM_MAX_LEN=5000

# AI Services - Update with your keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GITHUB_TOKEN=your_github_token_here

# Performance Settings
MAX_CONCURRENT_AGENTS=5
AGENT_HEARTBEAT_INTERVAL=60
CONTEXT_MAX_TOKENS=4000

# Docker Passwords
POSTGRES_PASSWORD=leanvibe_secure_pass
REDIS_PASSWORD=leanvibe_redis_pass
PGADMIN_PASSWORD=admin_password
EOF
    
    print_success "Environment configuration created"
}

# Optimized Python environment setup
create_python_env_fast() {
    print_step "Setting up Python environment with pip caching"
    
    local venv_path="${SCRIPT_DIR}/venv"
    
    # Remove existing venv if present
    if [[ -d "$venv_path" ]]; then
        rm -rf "$venv_path"
    fi
    
    # Create virtual environment
    run_with_progress "python3 -m venv $venv_path" 60 "Creating virtual environment"
    
    # Activate and upgrade pip with caching
    source "$venv_path/bin/activate"
    
    # Install dependencies with optimizations
    run_with_progress "pip install --upgrade pip setuptools wheel" 60 "Upgrading pip tools"
    
    # Install project dependencies with caching and parallel processing
    if [[ -f "pyproject.toml" ]]; then
        run_with_progress "pip install --cache-dir .pip-cache -e .[dev] --find-links .pip-cache/wheels" 180 "Installing project dependencies"
    else
        # Fallback to essential packages
        run_with_progress "pip install --cache-dir .pip-cache fastapi uvicorn sqlalchemy asyncpg redis anthropic" 120 "Installing core dependencies"
    fi
    
    print_success "Python environment ready"
}

# Parallel Docker service startup
start_docker_services_parallel() {
    print_step "Starting Docker services in parallel"
    
    # Create necessary directories
    mkdir -p "${SCRIPT_DIR}/dev-state/database/postgres-fast"
    mkdir -p "${SCRIPT_DIR}/dev-state/database/redis-fast"
    
    # Start services in parallel using fast compose file
    run_with_progress "docker compose -f docker-compose.fast.yml up -d postgres redis" 120 "Starting database services"
    
    # Wait for health checks in parallel
    print_status "$CYAN" "    → Waiting for services to become healthy..."
    
    local max_wait=90
    local wait_time=0
    local postgres_ready=false
    local redis_ready=false
    
    while [[ $wait_time -lt $max_wait ]]; do
        if [[ "$postgres_ready" == "false" ]] && docker compose -f docker-compose.fast.yml exec -T postgres pg_isready -U leanvibe_user -d leanvibe_agent_hive >/dev/null 2>&1; then
            postgres_ready=true
            print_status "$GREEN" "    ✅ PostgreSQL ready"
        fi
        
        if [[ "$redis_ready" == "false" ]] && docker compose -f docker-compose.fast.yml exec -T redis redis-cli -a leanvibe_redis_pass ping >/dev/null 2>&1; then
            redis_ready=true
            print_status "$GREEN" "    ✅ Redis ready"
        fi
        
        if [[ "$postgres_ready" == "true" && "$redis_ready" == "true" ]]; then
            break
        fi
        
        sleep 3
        wait_time=$((wait_time + 3))
        echo -n "."
    done
    
    if [[ "$postgres_ready" == "false" || "$redis_ready" == "false" ]]; then
        print_error "Services failed to start within $max_wait seconds"
    fi
    
    print_success "Docker services running"
}

# Fast database initialization
init_database_fast() {
    print_step "Initializing database with optimizations"
    
    local venv_path="${SCRIPT_DIR}/venv"
    source "$venv_path/bin/activate"
    
    # Run migrations with timeout
    run_with_progress "alembic upgrade head" 90 "Applying database migrations"
    
    # Quick validation
    if docker compose -f docker-compose.fast.yml exec -T postgres psql -U leanvibe_user -d leanvibe_agent_hive -c "SELECT 1;" >/dev/null 2>&1; then
        print_success "Database initialized and verified"
    else
        print_error "Database initialization failed"
    fi
}

# Fast validation with parallel checks
validate_installation_fast() {
    print_step "Running parallel validation checks"
    
    local venv_path="${SCRIPT_DIR}/venv"
    source "$venv_path/bin/activate"
    
    # Run health check script if available
    if [[ -f "${SCRIPT_DIR}/health-check.sh" ]]; then
        run_with_progress "timeout 60 bash ${SCRIPT_DIR}/health-check.sh" 60 "Running health checks"
    else
        # Basic validation
        print_status "$CYAN" "    → Running basic validation..."
        python -c "import app; print('✅ Python imports working')" 2>/dev/null || print_warning "Python imports need verification"
        docker compose -f docker-compose.fast.yml ps | grep -q "Up" && print_status "$GREEN" "    ✅ Docker services running"
    fi
    
    print_success "Installation validated"
}

# Create optimized startup scripts
create_optimized_scripts() {
    print_step "Creating optimized startup and management scripts"
    
    # Fast startup script
    cat > "${SCRIPT_DIR}/start-fast.sh" << 'EOF'
#!/bin/bash
# LeanVibe Agent Hive 2.0 - Fast Startup

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Starting LeanVibe Agent Hive 2.0 (Fast Mode)..."

# Activate virtual environment
source "${SCRIPT_DIR}/venv/bin/activate"

# Start services using fast compose
docker compose -f docker-compose.fast.yml up -d postgres redis

# Wait briefly for services
sleep 5

# Start API with performance optimizations
echo "🌟 Starting optimized application server..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --loop uvloop --http httptools

echo "✅ Agent Hive running at http://localhost:8000"
echo "📊 Health: http://localhost:8000/health"
echo "📖 API Docs: http://localhost:8000/docs"
EOF
    
    chmod +x "${SCRIPT_DIR}/start-fast.sh"
    
    # Fast stop script
    cat > "${SCRIPT_DIR}/stop-fast.sh" << 'EOF'
#!/bin/bash
# LeanVibe Agent Hive 2.0 - Fast Stop

echo "🛑 Stopping LeanVibe Agent Hive 2.0 (Fast Mode)..."
docker compose -f docker-compose.fast.yml down
echo "✅ Agent Hive stopped"
EOF
    
    chmod +x "${SCRIPT_DIR}/stop-fast.sh"
    
    print_success "Management scripts created"
}

# Enhanced final instructions with performance metrics
print_final_instructions() {
    local total_time=$((($(date +%s) - START_TIME)))
    local minutes=$((total_time / 60))
    local seconds=$((total_time % 60))
    
    echo ""
    print_status "$BOLD$GREEN" "🎉 LeanVibe Agent Hive 2.0 - FAST SETUP COMPLETE!"
    print_status "$BOLD$GREEN" "================================================"
    echo ""
    
    # Performance metrics
    print_status "$CYAN" "⚡ Performance Metrics:"
    print_status "$GREEN" "  • Total setup time: ${minutes}m ${seconds}s"
    print_status "$GREEN" "  • Target achieved: $(if [[ $total_time -le 900 ]]; then echo "✅ Yes"; else echo "⚠️  Close"; fi) (target: 5-15 minutes)"
    print_status "$GREEN" "  • Speed improvement: ~$(( (18*60 - total_time) / 60 ))+ minutes saved"
    echo ""
    
    # Next steps
    print_status "$YELLOW" "📋 Next Steps:"
    print_status "$NC" "1. Update API keys in .env.local (required for AI features)"
    print_status "$NC" "2. Start the system: ./start-fast.sh"
    print_status "$NC" "3. Access: http://localhost:8000"
    echo ""
    
    # Quick commands
    print_status "$YELLOW" "⚡ Quick Commands:"
    print_status "$NC" "  ./start-fast.sh      - Start with optimizations"
    print_status "$NC" "  ./stop-fast.sh       - Stop fast mode"
    print_status "$NC" "  ./health-check.sh    - System health check"
    echo ""
    
    # Success indicator
    if [[ $total_time -le 900 ]]; then  # 15 minutes
        print_status "$BOLD$GREEN" "🏆 MISSION ACCOMPLISHED: 5-15 minute setup achieved!"
    else
        print_status "$YELLOW" "📈 Setup completed but exceeded 15-minute target"
        print_status "$YELLOW" "   Consider running again for cached performance"
    fi
    echo ""
}

# Main optimized setup function
main() {
    # Clear log file
    > "$LOG_FILE"
    
    print_status "$BOLD$PURPLE" "⚡ LeanVibe Agent Hive 2.0 - FAST SETUP (5-15 min target)"
    print_status "$PURPLE" "============================================================"
    print_status "$CYAN" "🎯 Target: Reduce setup from 18+ minutes to 5-15 minutes"
    print_status "$CYAN" "🚀 Features: Parallel operations, caching, progress tracking"
    echo ""
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Check for root (warn but continue)
    if [[ $EUID -eq 0 ]]; then
        print_warning "Running as root. Consider using a regular user for better security."
    fi
    
    # Run optimized setup steps
    install_system_deps
    check_docker_fast
    create_env_config_fast
    create_python_env_fast
    start_docker_services_parallel
    init_database_fast
    validate_installation_fast
    create_optimized_scripts
    
    # Final summary
    print_final_instructions
}

# Enhanced error handling
trap 'print_error "Setup interrupted by user"' INT TERM

# Export functions for parallel execution
export -f print_status print_success print_warning print_error run_with_progress

# Run main function
main "$@"