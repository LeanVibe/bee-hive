#!/bin/bash
# Universal Project Index Installer
# Deploys containerized Project Index to any existing project with a single command

set -euo pipefail

# Configuration
readonly INSTALLER_VERSION="1.0.0"
readonly GITHUB_REPO="leanvibe/project-index"
readonly DEFAULT_INSTALL_DIR="/opt/project-index"
readonly DEFAULT_DATA_DIR="/var/lib/project-index"
readonly COMPOSE_FILE_URL="https://raw.githubusercontent.com/leanvibe/project-index/main/docker-compose.universal.yml"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

# Global variables
INSTALL_DIR=""
DATA_DIR=""
PROJECT_PATH=""
DEPLOYMENT_MODE="development"
ENABLE_DASHBOARD=true
ENABLE_MONITORING=false
FORCE_INSTALL=false
QUIET_MODE=false

# Logging functions
log() { 
    [[ "$QUIET_MODE" == "true" ]] || echo -e "${BLUE}[INFO]${NC} $1" >&2
}

success() { 
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

warn() { 
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

error() { 
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

debug() {
    [[ "${DEBUG:-false}" == "true" ]] && echo -e "${CYAN}[DEBUG]${NC} $1" >&2 || true
}

print_header() {
    echo -e "${PURPLE}$1${NC}" >&2
}

# Help function
show_help() {
    cat << EOF
Universal Project Index Installer v${INSTALLER_VERSION}

USAGE:
    curl -fsSL https://install.leanvibe.com/project-index.sh | bash
    curl -fsSL https://install.leanvibe.com/project-index.sh | bash -s -- [OPTIONS]

OPTIONS:
    -h, --help                  Show this help message
    -v, --version              Show version information
    -p, --path PATH            Project path to analyze (default: current directory)
    -i, --install-dir DIR      Installation directory (default: /opt/project-index)
    -d, --data-dir DIR         Data directory (default: /var/lib/project-index)
    -m, --mode MODE            Deployment mode: development|production|ci (default: development)
    --enable-dashboard         Enable web dashboard (default: true)
    --disable-dashboard        Disable web dashboard
    --enable-monitoring        Enable Prometheus/Grafana monitoring
    --disable-monitoring       Disable monitoring (default)
    --force                    Force reinstall if already installed
    --quiet                    Quiet output (only errors and final status)
    --debug                    Enable debug output

DEPLOYMENT MODES:
    development     Full features, development optimizations, local access
    production      High performance, security hardened, resource optimized
    ci              Lightweight, fast startup, CI/CD optimized

EXAMPLES:
    # Install with defaults
    curl -fsSL https://install.leanvibe.com/project-index.sh | bash

    # Install in production mode with monitoring
    curl -fsSL https://install.leanvibe.com/project-index.sh | bash -s -- --mode production --enable-monitoring

    # Install for specific project path
    curl -fsSL https://install.leanvibe.com/project-index.sh | bash -s -- --path /home/user/my-project

    # Force reinstall in quiet mode
    curl -fsSL https://install.leanvibe.com/project-index.sh | bash -s -- --force --quiet

REQUIREMENTS:
    - Docker 20.10+ with Docker Compose
    - curl and bash
    - 2GB+ available memory (1GB for small projects)
    - 1GB+ available disk space

ACCESS POINTS (after installation):
    - Project Index API: http://localhost:8100
    - Web Dashboard: http://localhost:8101 (if enabled)
    - PostgreSQL: localhost:5433 (admin access)
    - Prometheus: http://localhost:9090 (if monitoring enabled)
    - Grafana: http://localhost:3001 (if monitoring enabled)

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--version)
                echo "Universal Project Index Installer v${INSTALLER_VERSION}"
                exit 0
                ;;
            -p|--path)
                PROJECT_PATH="$2"
                shift 2
                ;;
            -i|--install-dir)
                INSTALL_DIR="$2"
                shift 2
                ;;
            -d|--data-dir)
                DATA_DIR="$2"
                shift 2
                ;;
            -m|--mode)
                DEPLOYMENT_MODE="$2"
                shift 2
                ;;
            --enable-dashboard)
                ENABLE_DASHBOARD=true
                shift
                ;;
            --disable-dashboard)
                ENABLE_DASHBOARD=false
                shift
                ;;
            --enable-monitoring)
                ENABLE_MONITORING=true
                shift
                ;;
            --disable-monitoring)
                ENABLE_MONITORING=false
                shift
                ;;
            --force)
                FORCE_INSTALL=true
                shift
                ;;
            --quiet)
                QUIET_MODE=true
                shift
                ;;
            --debug)
                DEBUG=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Set defaults
    PROJECT_PATH="${PROJECT_PATH:-$(pwd)}"
    INSTALL_DIR="${INSTALL_DIR:-$DEFAULT_INSTALL_DIR}"
    DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
    
    # Validate deployment mode
    case "$DEPLOYMENT_MODE" in
        development|production|ci)
            ;;
        *)
            error "Invalid deployment mode: $DEPLOYMENT_MODE"
            error "Valid modes: development, production, ci"
            exit 1
            ;;
    esac
}

# Environment validation
validate_environment() {
    log "Validating environment requirements..."
    
    local issues=()
    
    # Check if running as root (not recommended)
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root is not recommended for security reasons"
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        issues+=("Docker is required but not installed. Install from: https://docs.docker.com/get-docker/")
    else
        # Check Docker version
        local docker_version
        docker_version=$(docker version --format '{{.Client.Version}}' 2>/dev/null || echo "unknown")
        debug "Docker version: $docker_version"
        
        # Check if Docker daemon is running
        if ! docker info &> /dev/null; then
            issues+=("Docker daemon is not running. Please start Docker.")
        fi
    fi
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        issues+=("Docker Compose is required but not available")
    else
        local compose_version
        compose_version=$(docker compose version --short 2>/dev/null || echo "unknown")
        debug "Docker Compose version: $compose_version"
    fi
    
    # Check available memory
    if command -v free &> /dev/null; then
        local available_memory
        available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
        debug "Available memory: ${available_memory}MB"
        
        local required_memory=1024
        case "$DEPLOYMENT_MODE" in
            production) required_memory=2048 ;;
            development) required_memory=1536 ;;
            ci) required_memory=768 ;;
        esac
        
        if (( available_memory < required_memory )); then
            warn "Low available memory: ${available_memory}MB (recommended: ${required_memory}MB+)"
        fi
    fi
    
    # Check available disk space
    local available_disk
    available_disk=$(df -h "$INSTALL_DIR" 2>/dev/null | awk 'NR==2{print $4}' || echo "unknown")
    debug "Available disk space: $available_disk"
    
    # Check network connectivity
    if ! curl -fsSL --connect-timeout 5 https://hub.docker.com &> /dev/null; then
        warn "Limited internet connectivity detected. Docker image pulls may fail."
    fi
    
    # Report issues
    if [[ ${#issues[@]} -gt 0 ]]; then
        error "Environment validation failed:"
        for issue in "${issues[@]}"; do
            error "  - $issue"
        done
        exit 1
    fi
    
    success "Environment validation passed"
}

# Project detection and analysis
detect_project_configuration() {
    log "Analyzing project structure..."
    
    # Validate project path
    if [[ ! -d "$PROJECT_PATH" ]]; then
        error "Project path does not exist: $PROJECT_PATH"
        exit 1
    fi
    
    # Get absolute path
    PROJECT_PATH=$(cd "$PROJECT_PATH" && pwd)
    log "Project path: $PROJECT_PATH"
    
    # Detect project language
    local project_language
    project_language=$(detect_project_language "$PROJECT_PATH")
    log "Detected language: $project_language"
    
    # Estimate project size
    local project_size
    project_size=$(estimate_project_size "$PROJECT_PATH")
    log "Estimated size: $project_size"
    
    # Get project name
    local project_name
    project_name=$(basename "$PROJECT_PATH")
    log "Project name: $project_name"
    
    # Store configuration for later use
    export DETECTED_PROJECT_LANGUAGE="$project_language"
    export DETECTED_PROJECT_SIZE="$project_size"
    export DETECTED_PROJECT_NAME="$project_name"
    
    success "Project analysis complete"
}

# Language detection
detect_project_language() {
    local project_path="$1"
    
    # Check for common language indicators
    if [[ -f "$project_path/package.json" ]]; then
        echo "nodejs"
    elif [[ -f "$project_path/requirements.txt" ]] || [[ -f "$project_path/pyproject.toml" ]] || [[ -f "$project_path/setup.py" ]]; then
        echo "python"
    elif [[ -f "$project_path/go.mod" ]]; then
        echo "go"
    elif [[ -f "$project_path/Cargo.toml" ]]; then
        echo "rust"
    elif [[ -f "$project_path/pom.xml" ]] || [[ -f "$project_path/build.gradle" ]] || [[ -f "$project_path/build.gradle.kts" ]]; then
        echo "java"
    elif [[ -f "$project_path/CMakeLists.txt" ]] || [[ -f "$project_path/Makefile" ]]; then
        echo "cpp"
    elif [[ -f "$project_path/composer.json" ]]; then
        echo "php"
    elif [[ -f "$project_path/Gemfile" ]]; then
        echo "ruby"
    elif [[ -f "$project_path/.csproj" ]] || [[ -f "$project_path/project.json" ]]; then
        echo "csharp"
    else
        # Count file types to determine primary language
        local py_count js_count go_count rs_count java_count
        py_count=$(find "$project_path" -name "*.py" -type f 2>/dev/null | wc -l)
        js_count=$(find "$project_path" -name "*.js" -o -name "*.ts" -type f 2>/dev/null | wc -l)
        go_count=$(find "$project_path" -name "*.go" -type f 2>/dev/null | wc -l)
        rs_count=$(find "$project_path" -name "*.rs" -type f 2>/dev/null | wc -l)
        java_count=$(find "$project_path" -name "*.java" -type f 2>/dev/null | wc -l)
        
        # Return most common language
        if (( py_count > js_count && py_count > go_count && py_count > rs_count && py_count > java_count )); then
            echo "python"
        elif (( js_count > go_count && js_count > rs_count && js_count > java_count )); then
            echo "javascript"
        elif (( go_count > rs_count && go_count > java_count )); then
            echo "go"
        elif (( rs_count > java_count )); then
            echo "rust"
        elif (( java_count > 0 )); then
            echo "java"
        else
            echo "multi-language"
        fi
    fi
}

# Project size estimation
estimate_project_size() {
    local project_path="$1"
    
    # Count code files
    local code_files
    code_files=$(find "$project_path" -type f \( \
        -name "*.py" -o -name "*.js" -o -name "*.ts" -o -name "*.tsx" -o \
        -name "*.go" -o -name "*.rs" -o -name "*.java" -o -name "*.cpp" -o \
        -name "*.c" -o -name "*.h" -o -name "*.hpp" -o -name "*.php" -o \
        -name "*.rb" -o -name "*.cs" -o -name "*.swift" -o -name "*.kt" \
        \) 2>/dev/null | wc -l)
    
    debug "Code files found: $code_files"
    
    # Categorize project size
    if (( code_files < 50 )); then
        echo "small"
    elif (( code_files < 500 )); then
        echo "medium"
    elif (( code_files < 2000 )); then
        echo "large"
    else
        echo "enterprise"
    fi
}

# Generate optimized configuration
generate_configuration() {
    log "Generating optimized configuration..."
    
    # Create installation directory
    if [[ ! -d "$INSTALL_DIR" ]]; then
        debug "Creating install directory: $INSTALL_DIR"
        mkdir -p "$INSTALL_DIR"
    fi
    
    # Create data directory
    if [[ ! -d "$DATA_DIR" ]]; then
        debug "Creating data directory: $DATA_DIR"
        mkdir -p "$DATA_DIR"/{postgres,redis,cache,project,prometheus,grafana}
    fi
    
    # Generate secure passwords
    local project_index_password redis_password grafana_password
    project_index_password=$(generate_secure_password)
    redis_password=$(generate_secure_password)
    grafana_password=$(generate_secure_password)
    
    # Create environment configuration
    cat > "$INSTALL_DIR/.env" << EOF
# Universal Project Index Configuration
# Auto-generated on $(date)

# Project Information
HOST_PROJECT_PATH=$PROJECT_PATH
PROJECT_NAME=$DETECTED_PROJECT_NAME
PROJECT_LANGUAGE=$DETECTED_PROJECT_LANGUAGE
PROJECT_SIZE=$DETECTED_PROJECT_SIZE

# Installation Paths
INSTALL_DIR=$INSTALL_DIR
DATA_DIR=$DATA_DIR

# Security Configuration
PROJECT_INDEX_PASSWORD=$project_index_password
REDIS_PASSWORD=$redis_password
GRAFANA_ADMIN_PASSWORD=$grafana_password

# Service Ports (non-conflicting)
PROJECT_INDEX_PORT=8100
DASHBOARD_PORT=8101
POSTGRES_PORT=5433
REDIS_PORT=6380
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001

# Performance Configuration (optimized for $DETECTED_PROJECT_SIZE project)
$(generate_performance_config "$DETECTED_PROJECT_SIZE")

# Feature Configuration
ENABLE_REAL_TIME_MONITORING=true
ENABLE_ML_ANALYSIS=false
ENABLE_DASHBOARD=$ENABLE_DASHBOARD
ENABLE_METRICS=$ENABLE_MONITORING

# File Monitoring Patterns (optimized for $DETECTED_PROJECT_LANGUAGE)
$(generate_file_patterns "$DETECTED_PROJECT_LANGUAGE")

# Deployment Mode
DEPLOYMENT_MODE=$DEPLOYMENT_MODE

# Logging Configuration
LOG_LEVEL=$(get_log_level "$DEPLOYMENT_MODE")
EOF
    
    success "Configuration generated"
}

# Generate performance configuration based on project size
generate_performance_config() {
    local size="$1"
    
    case "$size" in
        small)
            cat << EOF
# Small Project Configuration
WORKER_CONCURRENCY=1
WORKER_REPLICAS=1
ANALYSIS_MODE=fast
PROJECT_INDEX_MEMORY_LIMIT=512M
PROJECT_INDEX_CPU_LIMIT=0.25
POSTGRES_MEMORY_LIMIT=256M
POSTGRES_CPU_LIMIT=0.1
REDIS_MEMORY_LIMIT=128M
REDIS_CPU_LIMIT=0.05
WORKER_MEMORY_LIMIT=256M
WORKER_CPU_LIMIT=0.1
MAX_FILE_SIZE_MB=5
ANALYSIS_TIMEOUT_SECONDS=120
CACHE_TTL_HOURS=12
EOF
            ;;
        medium)
            cat << EOF
# Medium Project Configuration
WORKER_CONCURRENCY=2
WORKER_REPLICAS=1
ANALYSIS_MODE=smart
PROJECT_INDEX_MEMORY_LIMIT=1G
PROJECT_INDEX_CPU_LIMIT=0.5
POSTGRES_MEMORY_LIMIT=512M
POSTGRES_CPU_LIMIT=0.25
REDIS_MEMORY_LIMIT=256M
REDIS_CPU_LIMIT=0.1
WORKER_MEMORY_LIMIT=512M
WORKER_CPU_LIMIT=0.25
MAX_FILE_SIZE_MB=10
ANALYSIS_TIMEOUT_SECONDS=300
CACHE_TTL_HOURS=24
EOF
            ;;
        large)
            cat << EOF
# Large Project Configuration
WORKER_CONCURRENCY=4
WORKER_REPLICAS=2
ANALYSIS_MODE=full
PROJECT_INDEX_MEMORY_LIMIT=2G
PROJECT_INDEX_CPU_LIMIT=1.0
POSTGRES_MEMORY_LIMIT=1G
POSTGRES_CPU_LIMIT=0.5
REDIS_MEMORY_LIMIT=512M
REDIS_CPU_LIMIT=0.2
WORKER_MEMORY_LIMIT=1G
WORKER_CPU_LIMIT=0.5
MAX_FILE_SIZE_MB=15
ANALYSIS_TIMEOUT_SECONDS=600
CACHE_TTL_HOURS=48
EOF
            ;;
        enterprise)
            cat << EOF
# Enterprise Project Configuration
WORKER_CONCURRENCY=8
WORKER_REPLICAS=4
ANALYSIS_MODE=comprehensive
PROJECT_INDEX_MEMORY_LIMIT=4G
PROJECT_INDEX_CPU_LIMIT=2.0
POSTGRES_MEMORY_LIMIT=2G
POSTGRES_CPU_LIMIT=1.0
REDIS_MEMORY_LIMIT=1G
REDIS_CPU_LIMIT=0.5
WORKER_MEMORY_LIMIT=2G
WORKER_CPU_LIMIT=1.0
MAX_FILE_SIZE_MB=25
ANALYSIS_TIMEOUT_SECONDS=1200
CACHE_TTL_HOURS=72
EOF
            ;;
    esac
}

# Generate file patterns based on detected language
generate_file_patterns() {
    local language="$1"
    
    case "$language" in
        python)
            cat << EOF
MONITOR_PATTERNS=**/*.py,**/*.pyx,**/*.pyi,**/*.ipynb,**/*.yml,**/*.yaml,**/*.toml,**/*.cfg,**/*.ini
IGNORE_PATTERNS=**/__pycache__/**,**/.pytest_cache/**,**/.mypy_cache/**,**/venv/**,**/.venv/**,**/*.pyc,**/*.pyo,**/build/**,**/dist/**,**/.tox/**
EOF
            ;;
        nodejs|javascript)
            cat << EOF
MONITOR_PATTERNS=**/*.js,**/*.ts,**/*.jsx,**/*.tsx,**/*.json,**/*.vue,**/*.svelte,**/*.mjs
IGNORE_PATTERNS=**/node_modules/**,**/build/**,**/dist/**,**/.next/**,**/.nuxt/**,**/coverage/**,**/*.min.js
EOF
            ;;
        go)
            cat << EOF
MONITOR_PATTERNS=**/*.go,**/go.mod,**/go.sum,**/*.yml,**/*.yaml
IGNORE_PATTERNS=**/vendor/**,**/bin/**,**/.git/**
EOF
            ;;
        rust)
            cat << EOF
MONITOR_PATTERNS=**/*.rs,**/Cargo.toml,**/Cargo.lock,**/*.yml,**/*.yaml
IGNORE_PATTERNS=**/target/**,**/Cargo.lock,**/.git/**
EOF
            ;;
        java)
            cat << EOF
MONITOR_PATTERNS=**/*.java,**/*.kt,**/*.scala,**/pom.xml,**/build.gradle,**/build.gradle.kts,**/*.properties
IGNORE_PATTERNS=**/target/**,**/build/**,**/.gradle/**,**/bin/**,**/*.class
EOF
            ;;
        *)
            cat << EOF
MONITOR_PATTERNS=**/*.py,**/*.js,**/*.ts,**/*.go,**/*.rs,**/*.java,**/*.cpp,**/*.c,**/*.h,**/*.yml,**/*.yaml,**/*.json
IGNORE_PATTERNS=**/node_modules/**,**/__pycache__/**,**/target/**,**/build/**,**/dist/**,**/.git/**,**/vendor/**
EOF
            ;;
    esac
}

# Get log level based on deployment mode
get_log_level() {
    local mode="$1"
    
    case "$mode" in
        development) echo "DEBUG" ;;
        production) echo "INFO" ;;
        ci) echo "WARNING" ;;
        *) echo "INFO" ;;
    esac
}

# Generate secure password
generate_secure_password() {
    if command -v openssl &> /dev/null; then
        openssl rand -base64 32 | tr -d "=+/" | cut -c1-25
    else
        # Fallback to /dev/urandom
        LC_ALL=C tr -dc 'a-zA-Z0-9' < /dev/urandom | head -c 25
    fi
}

# Download and setup Docker Compose configuration
download_and_setup() {
    log "Downloading Project Index components..."
    
    # Download Docker Compose file
    if ! curl -fsSL "$COMPOSE_FILE_URL" -o "$INSTALL_DIR/docker-compose.yml"; then
        error "Failed to download Docker Compose configuration"
        exit 1
    fi
    
    # Download any additional configuration files
    download_additional_configs
    
    # Set proper permissions
    chmod 644 "$INSTALL_DIR/docker-compose.yml"
    chmod 600 "$INSTALL_DIR/.env"
    
    success "Components downloaded and configured"
}

# Download additional configuration files
download_additional_configs() {
    local config_dir="$INSTALL_DIR/config"
    mkdir -p "$config_dir"
    
    # Create minimal init script for PostgreSQL
    cat > "$INSTALL_DIR/init-scripts/01-init.sql" << 'EOF'
-- Project Index Database Initialization
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_files_path ON project_files USING gin(file_path gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_dependencies_source ON dependencies(source_file_id);
CREATE INDEX IF NOT EXISTS idx_dependencies_target ON dependencies(target_file_id);
EOF
    
    # Create monitoring configuration if enabled
    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        mkdir -p "$INSTALL_DIR/monitoring"
        
        cat > "$INSTALL_DIR/monitoring/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'project-index'
    static_configs:
      - targets: ['project-index:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
    scrape_interval: 30s
EOF
    fi
}

# Start services with proper orchestration
start_services() {
    log "Starting Project Index services..."
    
    cd "$INSTALL_DIR"
    
    # Pull all required images first
    log "Pulling Docker images..."
    if ! docker compose pull --quiet; then
        warn "Some images failed to pull, attempting to continue..."
    fi
    
    # Start core services first
    log "Starting core infrastructure (PostgreSQL, Redis)..."
    docker compose up -d postgres redis
    
    # Wait for core services to be healthy
    wait_for_service "postgres" 60
    wait_for_service "redis" 30
    
    # Start main application
    log "Starting Project Index API..."
    docker compose up -d project-index
    wait_for_service "project-index" 120
    
    # Start worker services
    if [[ "$DETECTED_PROJECT_SIZE" != "small" ]]; then
        log "Starting analysis workers..."
        docker compose --profile workers up -d
        sleep 10  # Give workers time to connect
    fi
    
    # Start monitoring services if enabled
    if [[ "$ENABLE_REAL_TIME_MONITORING" == "true" ]]; then
        log "Starting file monitoring..."
        docker compose --profile monitoring up -d
    fi
    
    # Start dashboard if enabled
    if [[ "$ENABLE_DASHBOARD" == "true" ]]; then
        log "Starting web dashboard..."
        docker compose --profile dashboard up -d
        wait_for_service "dashboard" 60
    fi
    
    # Start metrics if enabled
    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        log "Starting monitoring stack..."
        docker compose --profile metrics up -d
    fi
    
    success "All services started successfully"
}

# Wait for service to become healthy
wait_for_service() {
    local service_name="$1"
    local max_wait="${2:-60}"
    local attempt=0
    
    debug "Waiting for $service_name to become healthy (max ${max_wait}s)..."
    
    while (( attempt < max_wait )); do
        if docker compose ps "$service_name" --format json 2>/dev/null | grep -q '"Health":"healthy"'; then
            debug "$service_name is healthy"
            return 0
        fi
        
        if docker compose ps "$service_name" --format json 2>/dev/null | grep -q '"State":"running"'; then
            # Service is running but no health check, assume it's working
            debug "$service_name is running (no health check)"
            return 0
        fi
        
        sleep 2
        ((attempt += 2))
        [[ "$QUIET_MODE" == "true" ]] || echo -n "."
    done
    
    echo >&2
    error "$service_name failed to become healthy within ${max_wait} seconds"
    
    # Show logs for debugging
    if [[ "$DEBUG" == "true" ]]; then
        debug "Service logs:"
        docker compose logs "$service_name" --tail 20
    fi
    
    return 1
}

# Verify installation
verify_installation() {
    log "Verifying installation..."
    
    local verification_errors=()
    
    # Check service health
    local required_services=("postgres" "redis" "project-index")
    for service in "${required_services[@]}"; do
        if ! docker compose ps "$service" --format json 2>/dev/null | grep -q '"State":"running"'; then
            verification_errors+=("Service $service is not running")
        fi
    done
    
    # Test API endpoint
    local api_url="http://localhost:${PROJECT_INDEX_PORT:-8100}"
    if ! curl -fsSL --connect-timeout 10 "$api_url/health" > /dev/null; then
        verification_errors+=("Project Index API is not responding at $api_url")
    fi
    
    # Test dashboard if enabled
    if [[ "$ENABLE_DASHBOARD" == "true" ]]; then
        local dashboard_url="http://localhost:${DASHBOARD_PORT:-8101}"
        if ! curl -fsSL --connect-timeout 10 "$dashboard_url/health" > /dev/null; then
            verification_errors+=("Dashboard is not responding at $dashboard_url")
        fi
    fi
    
    # Check data directory permissions
    if [[ ! -w "$DATA_DIR" ]]; then
        verification_errors+=("Data directory is not writable: $DATA_DIR")
    fi
    
    # Report verification results
    if [[ ${#verification_errors[@]} -gt 0 ]]; then
        error "Installation verification failed:"
        for err in "${verification_errors[@]}"; do
            error "  - $err"
        done
        
        # Show troubleshooting information
        show_troubleshooting_info
        exit 1
    fi
    
    success "Installation verification passed"
}

# Show troubleshooting information
show_troubleshooting_info() {
    cat << EOF

TROUBLESHOOTING:
    1. Check service status: cd $INSTALL_DIR && docker compose ps
    2. View service logs: cd $INSTALL_DIR && docker compose logs [service-name]
    3. Restart services: cd $INSTALL_DIR && docker compose restart
    4. Check system resources: docker stats
    5. Verify network connectivity: curl -I https://hub.docker.com

COMMON ISSUES:
    - Port conflicts: Check if ports 8100, 8101, 5433, 6380 are in use
    - Insufficient memory: Ensure at least 2GB RAM available
    - Docker permissions: Add user to docker group or run with sudo
    - Firewall blocking: Allow Docker bridge networks

SUPPORT:
    - Documentation: https://docs.leanvibe.com/project-index
    - GitHub Issues: https://github.com/leanvibe/project-index/issues
    - Community: https://discord.gg/leanvibe

EOF
}

# Display success information and usage instructions
display_success_info() {
    print_header ""
    print_header "🎉 Universal Project Index Installation Complete!"
    print_header "================================================="
    print_header ""
    
    success "Project Index is now running and analyzing your project"
    echo >&2
    
    echo -e "${BLUE}📊 Service Status:${NC}" >&2
    echo -e "   ${GREEN}✅${NC} Project Index API: http://localhost:${PROJECT_INDEX_PORT:-8100}" >&2
    
    if [[ "$ENABLE_DASHBOARD" == "true" ]]; then
        echo -e "   ${GREEN}✅${NC} Web Dashboard: http://localhost:${DASHBOARD_PORT:-8101}" >&2
    fi
    
    echo -e "   ${GREEN}✅${NC} PostgreSQL: localhost:${POSTGRES_PORT:-5433}" >&2
    echo -e "   ${GREEN}✅${NC} Redis: localhost:${REDIS_PORT:-6380}" >&2
    
    if [[ "$ENABLE_MONITORING" == "true" ]]; then
        echo -e "   ${GREEN}✅${NC} Prometheus: http://localhost:${PROMETHEUS_PORT:-9090}" >&2
        echo -e "   ${GREEN}✅${NC} Grafana: http://localhost:${GRAFANA_PORT:-3001}" >&2
    fi
    
    echo >&2
    echo -e "${BLUE}🚀 Quick Start Commands:${NC}" >&2
    echo -e "   # Check project status" >&2
    echo -e "   curl http://localhost:${PROJECT_INDEX_PORT:-8100}/api/projects" >&2
    echo >&2
    echo -e "   # Trigger project analysis" >&2
    echo -e "   curl -X POST http://localhost:${PROJECT_INDEX_PORT:-8100}/api/projects/analyze" >&2
    echo >&2
    echo -e "   # Get dependency graph" >&2
    echo -e "   curl http://localhost:${PROJECT_INDEX_PORT:-8100}/api/dependencies/graph" >&2
    echo >&2
    
    echo -e "${BLUE}🔧 Management Commands:${NC}" >&2
    echo -e "   # Service status" >&2
    echo -e "   cd $INSTALL_DIR && docker compose ps" >&2
    echo >&2
    echo -e "   # View logs" >&2
    echo -e "   cd $INSTALL_DIR && docker compose logs -f [service-name]" >&2
    echo >&2
    echo -e "   # Stop services" >&2
    echo -e "   cd $INSTALL_DIR && docker compose down" >&2
    echo >&2
    echo -e "   # Start services" >&2
    echo -e "   cd $INSTALL_DIR && docker compose up -d" >&2
    echo >&2
    echo -e "   # Update services" >&2
    echo -e "   cd $INSTALL_DIR && docker compose pull && docker compose up -d" >&2
    echo >&2
    
    echo -e "${BLUE}📁 Installation Paths:${NC}" >&2
    echo -e "   Install Directory: $INSTALL_DIR" >&2
    echo -e "   Data Directory: $DATA_DIR" >&2
    echo -e "   Project Path: $PROJECT_PATH" >&2
    echo -e "   Configuration: $INSTALL_DIR/.env" >&2
    echo >&2
    
    echo -e "${BLUE}📚 Resources:${NC}" >&2
    echo -e "   Documentation: https://docs.leanvibe.com/project-index" >&2
    echo -e "   API Reference: http://localhost:${PROJECT_INDEX_PORT:-8100}/docs" >&2
    echo -e "   GitHub: https://github.com/leanvibe/project-index" >&2
    echo >&2
    
    success "Happy coding with enhanced project intelligence! 🚀"
}

# Cleanup function for error handling
cleanup_on_error() {
    local exit_code=$?
    
    if [[ $exit_code -ne 0 ]]; then
        error "Installation failed with exit code $exit_code"
        
        if [[ -d "$INSTALL_DIR" ]] && [[ "$FORCE_INSTALL" == "true" ]]; then
            warn "Cleaning up failed installation..."
            cd "$INSTALL_DIR" && docker compose down --volumes 2>/dev/null || true
        fi
        
        echo >&2
        echo -e "${YELLOW}For support:${NC}" >&2
        echo -e "  - Check troubleshooting guide: https://docs.leanvibe.com/project-index/troubleshooting" >&2
        echo -e "  - Report issues: https://github.com/leanvibe/project-index/issues" >&2
        echo -e "  - Join community: https://discord.gg/leanvibe" >&2
    fi
}

# Check if already installed
check_existing_installation() {
    if [[ -f "$INSTALL_DIR/docker-compose.yml" ]] && [[ "$FORCE_INSTALL" != "true" ]]; then
        warn "Project Index appears to be already installed at: $INSTALL_DIR"
        warn "Use --force to reinstall or --install-dir to use a different location"
        
        # Check if services are running
        cd "$INSTALL_DIR"
        if docker compose ps --format json 2>/dev/null | grep -q '"State":"running"'; then
            echo >&2
            echo -e "${GREEN}Current services are running:${NC}" >&2
            docker compose ps
            echo >&2
            echo -e "Access your Project Index at: http://localhost:${PROJECT_INDEX_PORT:-8100}" >&2
        fi
        
        exit 0
    fi
}

# Main installation function
main() {
    # Set up error handling
    trap cleanup_on_error EXIT
    
    # Show header
    if [[ "$QUIET_MODE" != "true" ]]; then
        print_header "🚀 Universal Project Index Installer v${INSTALLER_VERSION}"
        print_header "==========================================="
        print_header ""
        log "Deploying containerized Project Index to any project with zero friction"
        print_header ""
    fi
    
    # Parse arguments
    parse_arguments "$@"
    
    # Check for existing installation
    check_existing_installation
    
    # Run installation steps
    validate_environment
    detect_project_configuration
    generate_configuration
    download_and_setup
    start_services
    verify_installation
    
    # Show success information
    if [[ "$QUIET_MODE" != "true" ]]; then
        display_success_info
    else
        success "Project Index installed successfully at http://localhost:${PROJECT_INDEX_PORT:-8100}"
    fi
    
    # Clear trap since we completed successfully
    trap - EXIT
}

# Run main function with all arguments
main "$@"