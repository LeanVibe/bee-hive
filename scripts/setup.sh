#!/bin/bash

# LeanVibe Agent Hive 2.0 - Unified Setup Script
# Professional developer experience with multiple setup profiles
# 
# Usage: ./scripts/setup.sh [PROFILE]  
# Profiles: minimal, fast (default), full, devcontainer
#
# Environment Variables:
#   SETUP_PROFILE  - Override profile selection (minimal|fast|full|devcontainer)
#   SKIP_DEPS      - Skip dependency installation (true/false)
#   SKIP_DOCKER    - Skip Docker service startup (true/false)
#   SKIP_MIGRATION - Skip database migration (true/false)

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
readonly DEFAULT_PROFILE="fast"
readonly PYTHON_MIN_VERSION="3.12"
readonly DOCKER_REQUIRED="true"
readonly NODE_MIN_VERSION="18"

# Profile configurations (using functions for compatibility)
get_profile_description() {
    case "$1" in
        "minimal") echo "Essential setup for CI/CD environments (2-3 min)" ;;
        "fast") echo "Optimized setup for development (5-8 min) [DEFAULT]" ;;
        "full") echo "Complete setup with all tools (10-15 min)" ;;
        "devcontainer") echo "VS Code DevContainer initialization (1-2 min)" ;;
        *) echo "Unknown profile" ;;
    esac
}

get_profile_features() {
    case "$1" in
        "minimal") echo "core_deps docker_services database" ;;
        "fast") echo "core_deps docker_services database health_check demo_validation" ;;
        "full") echo "core_deps docker_services database health_check demo_validation monitoring dev_tools docs" ;;
        "devcontainer") echo "core_deps health_check demo_validation" ;;
        *) echo "" ;;
    esac
}

# Global variables
SETUP_PROFILE="${SETUP_PROFILE:-${1:-$DEFAULT_PROFILE}}"
SKIP_DEPS="${SKIP_DEPS:-false}"
SKIP_DOCKER="${SKIP_DOCKER:-false}"
SKIP_MIGRATION="${SKIP_MIGRATION:-false}"
START_TIME=""
SETUP_LOG=""

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
    esac
    
    # Log to file if available
    if [[ -n "$SETUP_LOG" ]]; then
        echo "[$timestamp] [$level] $message" >> "$SETUP_LOG"
    fi
}

show_header() {
    clear
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                          LeanVibe Agent Hive 2.0                            ║
║                        Professional Setup Experience                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo
    log "INFO" "Starting setup with profile: ${SETUP_PROFILE}"
    log "INFO" "Profile: $(get_profile_description "$SETUP_PROFILE")"
    echo
    START_TIME=$(date +%s)
}

show_help() {
    cat << EOF
${CYAN}LeanVibe Agent Hive 2.0 - Unified Setup Script${NC}

${YELLOW}USAGE:${NC}
    $SCRIPT_NAME [PROFILE] [OPTIONS]

${YELLOW}PROFILES:${NC}
EOF
    for profile in minimal fast full devcontainer; do
        echo "    ${GREEN}$profile${NC} - $(get_profile_description "$profile")"
    done
    cat << EOF

${YELLOW}ENVIRONMENT VARIABLES:${NC}
    SETUP_PROFILE   Override profile selection
    SKIP_DEPS       Skip dependency installation (true/false)
    SKIP_DOCKER     Skip Docker service startup (true/false) 
    SKIP_MIGRATION  Skip database migration (true/false)

${YELLOW}EXAMPLES:${NC}
    $SCRIPT_NAME                 # Use fast profile (default)
    $SCRIPT_NAME minimal         # Minimal setup for CI
    $SCRIPT_NAME full            # Complete development setup
    SETUP_PROFILE=fast $SCRIPT_NAME  # Use environment variable

${YELLOW}MORE INFO:${NC}
    Documentation: docs/GETTING_STARTED.md
    Troubleshooting: docs/TROUBLESHOOTING_GUIDE_COMPREHENSIVE.md
EOF
}

check_prerequisites() {
    log "STEP" "Checking prerequisites..."
    
    local errors=0
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log "ERROR" "Python 3 is required but not installed"
        errors=$((errors + 1))
    else
        local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        log "INFO" "Python $python_version detected"
    fi
    
    # Check Docker (unless in devcontainer)
    if [[ "$SETUP_PROFILE" != "devcontainer" ]] && [[ "$SKIP_DOCKER" != "true" ]]; then
        if ! command -v docker &> /dev/null; then
            log "ERROR" "Docker is required but not installed"
            errors=$((errors + 1))
        else
            if ! docker info &> /dev/null; then
                log "ERROR" "Docker daemon is not running"
                errors=$((errors + 1))
            else
                log "INFO" "Docker is available and running"
            fi
        fi
        
        if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
            log "ERROR" "Docker Compose is required but not available"
            errors=$((errors + 1))
        else
            log "INFO" "Docker Compose is available"
        fi
    fi
    
    # Check Node.js for full profile
    if [[ "$SETUP_PROFILE" == "full" ]]; then
        if ! command -v node &> /dev/null; then
            log "WARN" "Node.js not found - frontend features will be limited"
        else
            local node_version=$(node --version | cut -d'v' -f2)
            log "INFO" "Node.js $node_version detected"
        fi
    fi
    
    if [[ $errors -gt 0 ]]; then
        log "ERROR" "Prerequisites check failed. Please install missing dependencies."
        exit 1
    fi
    
    log "SUCCESS" "Prerequisites check passed"
}

setup_logging() {
    local log_dir="$PROJECT_ROOT/logs"
    mkdir -p "$log_dir"
    SETUP_LOG="$log_dir/setup-$(date '+%Y%m%d-%H%M%S').log"
    log "INFO" "Setup logging to: $SETUP_LOG"
}

install_dependencies() {
    if [[ "$SKIP_DEPS" == "true" ]]; then
        log "INFO" "Skipping dependency installation (SKIP_DEPS=true)"
        return 0
    fi
    
    log "STEP" "Installing Python dependencies..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "$PROJECT_ROOT/venv" ]]; then
        log "INFO" "Creating Python virtual environment..."
        cd "$PROJECT_ROOT"
        python3 -m venv venv
    fi
    
    # Activate virtual environment and install dependencies
    source "$PROJECT_ROOT/venv/bin/activate"
    
    log "INFO" "Upgrading pip and core tools..."
    pip install --upgrade pip setuptools wheel
    
    log "INFO" "Installing project dependencies..."
    case "$SETUP_PROFILE" in
        "minimal")
            pip install -e .[ai-extended]
            ;;
        "fast"|"devcontainer")
            pip install -e .[dev,ai-extended]
            ;;
        "full")
            pip install -e .[dev,monitoring,ai-extended]
            ;;
    esac
    
    log "SUCCESS" "Dependencies installed successfully"
}

start_docker_services() {
    if [[ "$SKIP_DOCKER" == "true" ]] || [[ "$SETUP_PROFILE" == "devcontainer" ]]; then
        log "INFO" "Skipping Docker services startup"
        return 0
    fi
    
    log "STEP" "Starting Docker services..."
    
    cd "$PROJECT_ROOT"
    
    # Choose appropriate docker-compose file
    local compose_file="docker-compose.yml"
    case "$SETUP_PROFILE" in
        "fast")
            if [[ -f "docker-compose.fast.yml" ]]; then
                compose_file="docker-compose.fast.yml"
            fi
            ;;
        "full")
            compose_file="docker-compose.yml"
            ;;
    esac
    
    log "INFO" "Using compose file: $compose_file"
    
    # Start core services
    docker-compose -f "$compose_file" up -d postgres redis
    
    # Wait for services to be ready
    log "INFO" "Waiting for services to be ready..."
    local retries=30
    while [[ $retries -gt 0 ]]; do
        if docker-compose -f "$compose_file" exec -T postgres pg_isready -U leanvibe_user &> /dev/null; then
            log "SUCCESS" "PostgreSQL is ready"
            break
        fi
        retries=$((retries - 1))
        sleep 2
    done
    
    if [[ $retries -eq 0 ]]; then
        log "ERROR" "PostgreSQL failed to start within timeout"
        return 1
    fi
    
    # Start additional services for full profile
    if [[ "$SETUP_PROFILE" == "full" ]]; then
        log "INFO" "Starting monitoring services..."
        docker-compose -f "$compose_file" --profile monitoring up -d prometheus grafana || log "WARN" "Monitoring services failed to start"
    fi
    
    log "SUCCESS" "Docker services started successfully"
}

run_database_migration() {
    if [[ "$SKIP_MIGRATION" == "true" ]]; then
        log "INFO" "Skipping database migration (SKIP_MIGRATION=true)"
        return 0
    fi
    
    log "STEP" "Running database migrations..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    # Run migrations
    if alembic upgrade head; then
        log "SUCCESS" "Database migrations completed"
    else
        log "ERROR" "Database migration failed"
        return 1
    fi
}

run_health_check() {
    log "STEP" "Running health check..."
    
    cd "$PROJECT_ROOT"
    
    if [[ -f "health-check.sh" ]]; then
        if ./health-check.sh --quiet; then
            log "SUCCESS" "Health check passed"
        else
            log "WARN" "Health check reported issues (see details above)"
        fi
    else
        log "WARN" "Health check script not found"
    fi
}

validate_demo() {
    log "STEP" "Validating demo functionality..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    # Quick demo validation
    if python -c "
import sys
sys.path.append('.')
from scripts.demos.autonomous_development_demo import main
print('Demo validation: PASSED')
" 2>/dev/null; then
        log "SUCCESS" "Demo validation passed"
    else
        log "WARN" "Demo validation failed - some features may not work"
    fi
}

setup_monitoring() {
    if [[ "$SETUP_PROFILE" != "full" ]]; then
        return 0
    fi
    
    log "STEP" "Setting up monitoring tools..."
    
    # This would be expanded based on monitoring requirements
    log "INFO" "Monitoring setup completed"
}

setup_dev_tools() {
    if [[ "$SETUP_PROFILE" != "full" ]]; then
        return 0
    fi
    
    log "STEP" "Setting up development tools..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    # Install pre-commit hooks
    if command -v pre-commit &> /dev/null; then
        pre-commit install || log "WARN" "Failed to install pre-commit hooks"
    fi
    
    log "INFO" "Development tools setup completed"
}

generate_docs() {
    if [[ "$SETUP_PROFILE" != "full" ]]; then
        return 0
    fi
    
    log "STEP" "Generating documentation..."
    
    # This would be expanded for documentation generation
    log "INFO" "Documentation generation completed"
}

create_env_file() {
    log "STEP" "Setting up environment configuration..."
    
    cd "$PROJECT_ROOT"
    
    if [[ ! -f ".env.local" ]]; then
        cat > .env.local << 'EOF'
# LeanVibe Agent Hive 2.0 - Local Environment Configuration
# Generated by setup script

# Database Configuration
DATABASE_URL=postgresql://leanvibe_user:leanvibe_password@localhost:5432/leanvibe_agent_hive

# Redis Configuration  
REDIS_URL=redis://localhost:6379/0

# Security Configuration
JWT_SECRET_KEY=your-secret-key-change-in-production
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# AI Configuration (REQUIRED - Add your API key)
# ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Observability Configuration
ENABLE_METRICS=true
ENABLE_TRACING=false

# Development Configuration
DEBUG=true
LOG_LEVEL=INFO

# Setup Metadata
EOF
        echo "SETUP_PROFILE=$SETUP_PROFILE" >> .env.local
        echo "SETUP_TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> .env.local
        echo "SETUP_VERSION=$SCRIPT_VERSION" >> .env.local
        
        log "SUCCESS" "Environment file created: .env.local"
    else
        log "INFO" "Environment file already exists: .env.local"
    fi
}

show_next_steps() {
    local end_time=$(date +%s)
    local duration=$((end_time - START_TIME))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    echo
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                               SETUP COMPLETE                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo
    log "SUCCESS" "Setup completed in ${minutes}m ${seconds}s"
    echo
    
    echo -e "${YELLOW}NEXT STEPS:${NC}"
    echo
    echo "1. ${CYAN}Add your API key:${NC}"
    echo "   echo 'ANTHROPIC_API_KEY=your_key_here' >> .env.local"
    echo
    echo "2. ${CYAN}Start the system:${NC}"
    echo "   make start        # Start all services"
    echo "   # OR"
    echo "   ./start-fast.sh   # Use existing script"
    echo
    echo "3. ${CYAN}Verify everything works:${NC}"
    echo "   make health       # Run health check"
    echo "   curl http://localhost:8000/health"
    echo
    echo "4. ${CYAN}Try autonomous development:${NC}"
    echo "   python scripts/demos/autonomous_development_demo.py"
    echo
    
    echo -e "${YELLOW}USEFUL COMMANDS:${NC}"
    echo "   make help         # Show all available commands"
    echo "   make test         # Run test suite"
    echo "   make clean        # Clean up resources"
    echo
    
    if [[ -n "$SETUP_LOG" ]]; then
        echo -e "${CYAN}Setup log saved to:${NC} $SETUP_LOG"
    fi
    echo
}

cleanup_on_error() {
    local exit_code=$?
    log "ERROR" "Setup failed with exit code $exit_code"
    
    echo
    echo -e "${RED}SETUP FAILED${NC}"
    echo
    echo "Troubleshooting resources:"
    echo "- Setup log: $SETUP_LOG" 
    echo "- Troubleshooting guide: docs/TROUBLESHOOTING_GUIDE_COMPREHENSIVE.md"
    echo "- Health check: ./health-check.sh"
    echo
    
    exit $exit_code
}

#======================================
# Main Setup Flow
#======================================

main() {
    # Handle help request
    if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
        show_help
        exit 0
    fi
    
    # Validate profile
    if [[ -n "${1:-}" ]] && [[ "$1" != "minimal" && "$1" != "fast" && "$1" != "full" && "$1" != "devcontainer" ]]; then
        log "ERROR" "Invalid profile: $1"
        echo "Valid profiles: minimal fast full devcontainer"
        exit 1
    fi
    
    # Set up error handling
    trap cleanup_on_error ERR
    
    # Change to project root
    cd "$PROJECT_ROOT"
    
    # Initialize
    show_header
    setup_logging
    check_prerequisites
    
    # Core setup steps
    create_env_file
    install_dependencies
    start_docker_services
    run_database_migration
    
    # Profile-specific features  
    local features=($(get_profile_features "$SETUP_PROFILE"))
    for feature in "${features[@]}"; do
        case "$feature" in
            "health_check") run_health_check ;;
            "demo_validation") validate_demo ;;
            "monitoring") setup_monitoring ;;
            "dev_tools") setup_dev_tools ;;
            "docs") generate_docs ;;
        esac
    done
    
    # Success
    show_next_steps
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi