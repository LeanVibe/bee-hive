#!/bin/bash
set -euo pipefail

# Project Index Universal Installer - Startup Orchestration Script
# Manages service startup, dependency validation, and graceful initialization

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="${PROJECT_NAME:-project-index}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.universal.yml}"
ENV_FILE="${ENV_FILE:-.env}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Cleanup function for graceful shutdown
cleanup() {
    log_info "Cleaning up on exit..."
    if [[ -n "${DOCKER_COMPOSE_PID:-}" ]]; then
        kill $DOCKER_COMPOSE_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Help function
show_help() {
    cat << EOF
Project Index Universal Installer - Startup Script

USAGE:
    $0 [OPTIONS] [COMMAND]

COMMANDS:
    start       Start all Project Index services (default)
    stop        Stop all services
    restart     Restart all services
    status      Show service status
    logs        Show service logs
    health      Run health checks
    clean       Clean up all data and containers

OPTIONS:
    -e, --env FILE          Environment file (default: .env)
    -f, --compose FILE      Docker compose file (default: docker-compose.universal.yml)
    -p, --profile PROFILE   Deployment profile (small|medium|large|xlarge)
    -t, --timeout SECONDS   Startup timeout (default: 300)
    -v, --verbose           Verbose output
    -h, --help              Show this help

PROFILES:
    small       For small projects (< 1k files, 1 developer)
    medium      For medium projects (1k-10k files, 2-5 developers) [default]
    large       For large projects (> 10k files, team development)
    xlarge      For enterprise projects (> 100k files, multiple teams)

EXAMPLES:
    # Start with default medium profile
    $0 start

    # Start with small profile for personal projects
    $0 --profile small start

    # Start with custom environment file
    $0 --env .env.production start

    # Check service health
    $0 health

    # View logs for debugging
    $0 logs

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--env)
                ENV_FILE="$2"
                shift 2
                ;;
            -f|--compose)
                COMPOSE_FILE="$2"
                shift 2
                ;;
            -p|--profile)
                DEPLOYMENT_PROFILE="$2"
                shift 2
                ;;
            -t|--timeout)
                STARTUP_TIMEOUT="$2"
                shift 2
                ;;
            -v|--verbose)
                LOG_LEVEL="DEBUG"
                set -x
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            start|stop|restart|status|logs|health|clean)
                COMMAND="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Set default command
    COMMAND="${COMMAND:-start}"
}

# Validate prerequisites
validate_prerequisites() {
    log_step "Validating prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose V2 is not installed or not working"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Prerequisites validated"
}

# Load and validate environment configuration
load_environment() {
    log_step "Loading environment configuration..."
    
    # Apply deployment profile if specified
    if [[ -n "${DEPLOYMENT_PROFILE:-}" ]]; then
        PROFILE_FILE=".env.${DEPLOYMENT_PROFILE}"
        if [[ -f "$PROFILE_FILE" ]]; then
            log_info "Loading profile: $DEPLOYMENT_PROFILE"
            # Load profile first, then override with custom env file
            set -a
            source "$PROFILE_FILE"
            set +a
        else
            log_warn "Profile file $PROFILE_FILE not found, using defaults"
        fi
    fi
    
    # Load main environment file
    if [[ -f "$ENV_FILE" ]]; then
        log_info "Loading environment from: $ENV_FILE"
        set -a
        source "$ENV_FILE"
        set +a
    else
        log_warn "Environment file $ENV_FILE not found"
        if [[ "$ENV_FILE" == ".env" ]] && [[ -f ".env.template" ]]; then
            log_info "Creating .env from template..."
            cp .env.template .env
            log_warn "Please edit .env file with your configuration before starting"
            exit 1
        fi
    fi
    
    # Validate required environment variables
    local required_vars=(
        "PROJECT_INDEX_PASSWORD"
        "REDIS_PASSWORD"
        "HOST_PROJECT_PATH"
    )
    
    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "Missing required environment variables:"
        printf '%s\n' "${missing_vars[@]}" | sed 's/^/  - /'
        log_info "Please check your $ENV_FILE file"
        exit 1
    fi
    
    # Validate project path
    if [[ ! -d "${HOST_PROJECT_PATH}" ]]; then
        log_error "Project path does not exist: ${HOST_PROJECT_PATH}"
        log_info "Please set HOST_PROJECT_PATH to a valid directory"
        exit 1
    fi
    
    log_success "Environment configuration loaded"
}

# Setup data directories
setup_data_directories() {
    log_step "Setting up data directories..."
    
    local data_dir="${DATA_DIR:-./data}"
    local dirs=(
        "$data_dir/postgres"
        "$data_dir/redis"
        "$data_dir/cache"
        "$data_dir/project"
        "$data_dir/snapshots"
        "$data_dir/logs"
        "$data_dir/worker/temp"
        "$data_dir/worker/logs"
        "$data_dir/monitor/state"
        "$data_dir/monitor/temp"
        "$data_dir/monitor/logs"
        "$data_dir/prometheus"
        "$data_dir/grafana"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log_info "Creating directory: $dir"
            mkdir -p "$dir"
            
            # Set appropriate permissions
            if [[ "$dir" == *"postgres"* ]]; then
                chmod 750 "$dir"
            elif [[ "$dir" == *"redis"* ]]; then
                chmod 750 "$dir"
            else
                chmod 755 "$dir"
            fi
        fi
    done
    
    log_success "Data directories ready"
}

# Check for port conflicts
check_port_conflicts() {
    log_step "Checking for port conflicts..."
    
    local ports=(
        "${PROJECT_INDEX_PORT:-8100}"
        "${POSTGRES_PORT:-5433}"
        "${REDIS_PORT:-6380}"
        "${DASHBOARD_PORT:-8101}"
        "${PROMETHEUS_PORT:-9090}"
        "${GRAFANA_PORT:-3001}"
    )
    
    local conflicts=()
    for port in "${ports[@]}"; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            conflicts+=("$port")
        fi
    done
    
    if [[ ${#conflicts[@]} -gt 0 ]]; then
        log_warn "Port conflicts detected:"
        printf '%s\n' "${conflicts[@]}" | sed 's/^/  - Port /'
        log_info "Consider changing ports in your environment configuration"
        
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        log_success "No port conflicts detected"
    fi
}

# Start services with proper dependency ordering
start_services() {
    log_step "Starting Project Index services..."
    
    # Determine profiles to start
    local profiles="production"
    
    # Add optional profiles based on configuration
    if [[ "${ENABLE_METRICS:-false}" == "true" ]]; then
        profiles="$profiles,metrics"
    fi
    
    if [[ "${ENABLE_REAL_TIME_MONITORING:-true}" == "true" ]]; then
        profiles="$profiles,monitoring"
    fi
    
    if [[ "${WORKER_REPLICAS:-1}" -gt 0 ]]; then
        profiles="$profiles,workers"
    fi
    
    if [[ "${DASHBOARD_PORT:-}" != "" ]]; then
        profiles="$profiles,dashboard"
    fi
    
    log_info "Starting with profiles: $profiles"
    
    # Start core services first
    log_info "Starting core infrastructure (PostgreSQL, Redis)..."
    docker compose -f "$COMPOSE_FILE" --profile production up -d postgres redis
    
    # Wait for core services to be healthy
    log_info "Waiting for core services to be ready..."
    local retries=0
    local max_retries=30
    
    while [[ $retries -lt $max_retries ]]; do
        if docker compose -f "$COMPOSE_FILE" ps postgres | grep -q "Up (healthy)" && \
           docker compose -f "$COMPOSE_FILE" ps redis | grep -q "Up (healthy)"; then
            log_success "Core services are healthy"
            break
        fi
        
        log_info "Waiting for core services... ($((retries + 1))/$max_retries)"
        sleep 5
        ((retries++))
    done
    
    if [[ $retries -eq $max_retries ]]; then
        log_error "Core services failed to start within timeout"
        docker compose -f "$COMPOSE_FILE" logs postgres redis
        exit 1
    fi
    
    # Start Project Index API
    log_info "Starting Project Index API..."
    docker compose -f "$COMPOSE_FILE" --profile production up -d project-index
    
    # Wait for API to be ready
    log_info "Waiting for Project Index API to be ready..."
    retries=0
    max_retries=20
    
    while [[ $retries -lt $max_retries ]]; do
        if curl -f -s "http://localhost:${PROJECT_INDEX_PORT:-8100}/health" > /dev/null; then
            log_success "Project Index API is ready"
            break
        fi
        
        log_info "Waiting for API... ($((retries + 1))/$max_retries)"
        sleep 5
        ((retries++))
    done
    
    if [[ $retries -eq $max_retries ]]; then
        log_error "Project Index API failed to start within timeout"
        docker compose -f "$COMPOSE_FILE" logs project-index
        exit 1
    fi
    
    # Start additional services
    if [[ "$profiles" != "production" ]]; then
        log_info "Starting additional services..."
        docker compose -f "$COMPOSE_FILE" $(printf " --profile %s" ${profiles//,/ }) up -d
    fi
    
    log_success "All services started successfully"
}

# Show service status
show_status() {
    log_step "Showing service status..."
    
    docker compose -f "$COMPOSE_FILE" ps
    
    echo
    log_info "Service URLs:"
    echo "  - Project Index API: http://localhost:${PROJECT_INDEX_PORT:-8100}"
    echo "  - API Documentation: http://localhost:${PROJECT_INDEX_PORT:-8100}/docs"
    echo "  - Health Check: http://localhost:${PROJECT_INDEX_PORT:-8100}/health"
    
    if [[ "${DASHBOARD_PORT:-}" != "" ]]; then
        echo "  - Dashboard: http://localhost:${DASHBOARD_PORT}"
    fi
    
    if [[ "${PROMETHEUS_PORT:-}" != "" ]]; then
        echo "  - Prometheus: http://localhost:${PROMETHEUS_PORT}"
    fi
    
    if [[ "${GRAFANA_PORT:-}" != "" ]]; then
        echo "  - Grafana: http://localhost:${GRAFANA_PORT}"
    fi
}

# Run health checks
run_health_checks() {
    log_step "Running comprehensive health checks..."
    
    if command -v python3 &> /dev/null; then
        # Install health check dependencies if needed
        if ! python3 -c "import aiohttp, asyncpg, redis" 2>/dev/null; then
            log_info "Installing health check dependencies..."
            pip3 install aiohttp asyncpg redis psutil 2>/dev/null || {
                log_warn "Could not install health check dependencies, using basic checks"
                curl -f "http://localhost:${PROJECT_INDEX_PORT:-8100}/health" || exit 1
                return
            }
        fi
        
        # Run comprehensive health check
        python3 docker/health-check.py --output text
    else
        log_warn "Python3 not available, running basic health check"
        curl -f "http://localhost:${PROJECT_INDEX_PORT:-8100}/health" || exit 1
    fi
}

# Stop services gracefully
stop_services() {
    log_step "Stopping Project Index services..."
    
    docker compose -f "$COMPOSE_FILE" down
    
    log_success "Services stopped"
}

# Clean up all data and containers
clean_all() {
    log_step "Cleaning up all Project Index data and containers..."
    
    read -p "This will delete all data and containers. Are you sure? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Cleanup cancelled"
        exit 0
    fi
    
    # Stop and remove containers, networks, volumes
    docker compose -f "$COMPOSE_FILE" down -v --remove-orphans
    
    # Remove data directories
    local data_dir="${DATA_DIR:-./data}"
    if [[ -d "$data_dir" ]]; then
        log_info "Removing data directory: $data_dir"
        rm -rf "$data_dir"
    fi
    
    # Remove Docker images (optional)
    read -p "Also remove Docker images? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker images --format "table {{.Repository}}:{{.Tag}}" | grep "project-index" | xargs -r docker rmi
    fi
    
    log_success "Cleanup completed"
}

# Show logs
show_logs() {
    docker compose -f "$COMPOSE_FILE" logs -f "${@}"
}

# Main execution
main() {
    log_info "Project Index Universal Installer - Startup Orchestrator"
    log_info "================================================="
    
    parse_arguments "$@"
    
    case "$COMMAND" in
        start)
            validate_prerequisites
            load_environment
            setup_data_directories
            check_port_conflicts
            start_services
            show_status
            log_success "Project Index is ready! 🚀"
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            sleep 2
            validate_prerequisites
            load_environment
            start_services
            show_status
            ;;
        status)
            show_status
            ;;
        health)
            run_health_checks
            ;;
        logs)
            show_logs "$@"
            ;;
        clean)
            clean_all
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@"