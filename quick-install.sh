#!/bin/bash
set -euo pipefail

# Project Index Universal Installer - Quick Installation Script
# One-command installation for any project

# Script metadata
SCRIPT_VERSION="2.0.0"
REPO_URL="https://github.com/leanvibe/bee-hive.git"
INSTALL_DIR="project-index"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default values
DEPLOYMENT_PROFILE="medium"
PROJECT_PATH=""
AUTO_START="true"
CLONE_REPO="true"
INTERACTIVE="true"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Show banner
show_banner() {
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🚀 Project Index Universal Installer                              ║
║                                                                      ║
║   Intelligent project analysis and context optimization             ║
║   for any codebase in under 5 minutes                              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
EOF
}

# Show help
show_help() {
    cat << EOF
Project Index Universal Installer - Quick Installation

USAGE:
    curl -fsSL https://raw.githubusercontent.com/leanvibe/bee-hive/main/quick-install.sh | bash
    
    Or with options:
    curl -fsSL https://raw.githubusercontent.com/leanvibe/bee-hive/main/quick-install.sh | bash -s -- [OPTIONS]

OPTIONS:
    --profile PROFILE       Deployment profile (small|medium|large) [default: medium]
    --path PATH            Path to your project directory
    --no-start             Don't start services after installation
    --no-clone             Use existing directory (skip git clone)
    --yes                  Non-interactive mode (accept all defaults)
    --help                 Show this help message

PROFILES:
    small     < 1k files, 1 developer    (1GB RAM, 0.5 CPU)
    medium    1k-10k files, 2-5 devs     (2GB RAM, 1 CPU)
    large     > 10k files, team dev      (4GB RAM, 2 CPU)

EXAMPLES:
    # Basic installation with defaults
    curl -fsSL https://raw.githubusercontent.com/leanvibe/bee-hive/main/quick-install.sh | bash

    # Small project installation
    curl -fsSL https://raw.githubusercontent.com/leanvibe/bee-hive/main/quick-install.sh | bash -s -- --profile small --path /my/project

    # Non-interactive installation
    curl -fsSL https://raw.githubusercontent.com/leanvibe/bee-hive/main/quick-install.sh | bash -s -- --yes --path /my/project

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --profile)
                DEPLOYMENT_PROFILE="$2"
                shift 2
                ;;
            --path)
                PROJECT_PATH="$2"
                shift 2
                ;;
            --no-start)
                AUTO_START="false"
                shift
                ;;
            --no-clone)
                CLONE_REPO="false"
                shift
                ;;
            --yes)
                INTERACTIVE="false"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate system requirements
validate_system() {
    log_step "Validating system requirements..."
    
    local errors=()
    
    # Check operating system
    if [[ "$OSTYPE" != "linux-gnu"* ]] && [[ "$OSTYPE" != "darwin"* ]]; then
        errors+=("Unsupported operating system: $OSTYPE")
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        errors+=("Docker is not installed. Install from: https://docs.docker.com/get-docker/")
    elif ! docker info &> /dev/null; then
        errors+=("Docker daemon is not running")
    fi
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        errors+=("Docker Compose V2 is not available")
    fi
    
    # Check Git (only if cloning)
    if [[ "$CLONE_REPO" == "true" ]] && ! command -v git &> /dev/null; then
        errors+=("Git is not installed")
    fi
    
    # Check available disk space (need at least 1GB)
    local available_space
    if [[ "$OSTYPE" == "darwin"* ]]; then
        available_space=$(df -g . | awk 'NR==2 {print $4}')
    else
        available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    fi
    
    if [[ $available_space -lt 1 ]]; then
        errors+=("Insufficient disk space. Need at least 1GB, have ${available_space}GB")
    fi
    
    # Check memory
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        local total_mem=$(free -g | awk 'NR==2{print $2}')
        if [[ $total_mem -lt 2 ]]; then
            log_warn "Low memory detected (${total_mem}GB). Consider using --profile small"
        fi
    fi
    
    # Report errors
    if [[ ${#errors[@]} -gt 0 ]]; then
        log_error "System validation failed:"
        printf '%s\n' "${errors[@]}" | sed 's/^/  - /'
        exit 1
    fi
    
    log_success "System requirements satisfied"
}

# Interactive configuration
interactive_config() {
    if [[ "$INTERACTIVE" == "false" ]]; then
        return
    fi
    
    log_step "Interactive configuration..."
    
    # Ask for project path if not provided
    if [[ -z "$PROJECT_PATH" ]]; then
        read -p "Enter the path to your project directory: " -r PROJECT_PATH
        
        # Validate project path
        if [[ ! -d "$PROJECT_PATH" ]]; then
            log_error "Directory does not exist: $PROJECT_PATH"
            exit 1
        fi
    fi
    
    # Confirm deployment profile
    echo
    echo "Deployment profiles:"
    echo "  small  - < 1k files, 1 developer (1GB RAM, 0.5 CPU)"
    echo "  medium - 1k-10k files, 2-5 devs (2GB RAM, 1 CPU) [current: $DEPLOYMENT_PROFILE]"
    echo "  large  - > 10k files, team dev (4GB RAM, 2 CPU)"
    echo
    
    read -p "Keep profile '$DEPLOYMENT_PROFILE'? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        read -p "Enter profile (small/medium/large): " -r DEPLOYMENT_PROFILE
    fi
    
    # Validate profile
    if [[ ! "$DEPLOYMENT_PROFILE" =~ ^(small|medium|large)$ ]]; then
        log_error "Invalid profile: $DEPLOYMENT_PROFILE"
        exit 1
    fi
}

# Clone or setup repository
setup_repository() {
    log_step "Setting up Project Index repository..."
    
    if [[ "$CLONE_REPO" == "true" ]]; then
        if [[ -d "$INSTALL_DIR" ]]; then
            log_warn "Directory $INSTALL_DIR already exists"
            if [[ "$INTERACTIVE" == "true" ]]; then
                read -p "Remove and re-clone? (y/N): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    rm -rf "$INSTALL_DIR"
                else
                    log_info "Using existing directory"
                    CLONE_REPO="false"
                fi
            else
                rm -rf "$INSTALL_DIR"
            fi
        fi
        
        if [[ "$CLONE_REPO" == "true" ]]; then
            log_info "Cloning Project Index repository..."
            git clone "$REPO_URL" "$INSTALL_DIR"
        fi
    fi
    
    # Change to install directory
    cd "$INSTALL_DIR"
    log_success "Repository ready in $(pwd)"
}

# Generate environment configuration
generate_config() {
    log_step "Generating environment configuration..."
    
    # Start with profile template
    local env_source=".env.template"
    if [[ -f ".env.${DEPLOYMENT_PROFILE}" ]]; then
        env_source=".env.${DEPLOYMENT_PROFILE}"
        log_info "Using profile template: $env_source"
    fi
    
    # Copy template to .env
    cp "$env_source" .env
    
    # Generate secure passwords
    local db_password=$(openssl rand -base64 32 2>/dev/null || head -c 32 /dev/urandom | base64)
    local redis_password=$(openssl rand -base64 32 2>/dev/null || head -c 32 /dev/urandom | base64)
    
    # Update configuration
    if [[ -n "$PROJECT_PATH" ]]; then
        # Convert to absolute path
        PROJECT_PATH=$(cd "$PROJECT_PATH" && pwd)
        
        # Update .env file
        sed -i.bak "s|HOST_PROJECT_PATH=.*|HOST_PROJECT_PATH=$PROJECT_PATH|" .env
    fi
    
    # Set deployment profile
    sed -i.bak "s|DEPLOYMENT_PROFILE=.*|DEPLOYMENT_PROFILE=$DEPLOYMENT_PROFILE|" .env
    
    # Set secure passwords
    sed -i.bak "s|PROJECT_INDEX_PASSWORD=.*|PROJECT_INDEX_PASSWORD=$db_password|" .env
    sed -i.bak "s|REDIS_PASSWORD=.*|REDIS_PASSWORD=$redis_password|" .env
    
    # Set project name based on directory
    if [[ -n "$PROJECT_PATH" ]]; then
        local project_name=$(basename "$PROJECT_PATH")
        sed -i.bak "s|PROJECT_NAME=.*|PROJECT_NAME=$project_name|" .env
    fi
    
    # Remove backup file
    rm -f .env.bak
    
    log_success "Environment configuration generated"
}

# Download Docker images
pull_images() {
    log_step "Downloading Docker images..."
    
    # Build images locally (faster than pulling for now)
    log_info "Building Project Index images..."
    docker compose -f docker-compose.universal.yml build --parallel
    
    # Pull external images
    log_info "Pulling external images..."
    docker compose -f docker-compose.universal.yml pull postgres redis prometheus grafana
    
    log_success "Docker images ready"
}

# Start services
start_services() {
    if [[ "$AUTO_START" != "true" ]]; then
        return
    fi
    
    log_step "Starting Project Index services..."
    
    # Use the startup script
    ./start-project-index.sh --profile "$DEPLOYMENT_PROFILE" start
    
    log_success "Services started successfully!"
}

# Show completion message
show_completion() {
    log_success "Project Index installation completed! 🎉"
    echo
    log_info "Services running at:"
    echo "  - Project Index API: http://localhost:${PROJECT_INDEX_PORT:-8100}"
    echo "  - API Documentation: http://localhost:${PROJECT_INDEX_PORT:-8100}/docs"
    echo "  - Health Check: http://localhost:${PROJECT_INDEX_PORT:-8100}/health"
    
    if [[ "${DASHBOARD_PORT:-}" != "" ]]; then
        echo "  - Dashboard: http://localhost:${DASHBOARD_PORT:-8101}"
    fi
    
    if [[ "${PROMETHEUS_PORT:-}" != "" ]] && [[ "${ENABLE_METRICS:-false}" == "true" ]]; then
        echo "  - Prometheus: http://localhost:${PROMETHEUS_PORT:-9090}"
    fi
    
    if [[ "${GRAFANA_PORT:-}" != "" ]] && [[ "${ENABLE_METRICS:-false}" == "true" ]]; then
        echo "  - Grafana: http://localhost:${GRAFANA_PORT:-3001} (admin/admin)"
    fi
    
    echo
    log_info "Useful commands:"
    echo "  ./start-project-index.sh status    # Check service status"
    echo "  ./start-project-index.sh health    # Run health checks"
    echo "  ./start-project-index.sh logs      # View service logs"
    echo "  ./start-project-index.sh stop      # Stop all services"
    echo
    log_info "Documentation: PROJECT_INDEX_UNIVERSAL_INSTALLER_GUIDE.md"
    
    if [[ "$AUTO_START" != "true" ]]; then
        echo
        log_info "To start services manually:"
        echo "  cd $INSTALL_DIR"
        echo "  ./start-project-index.sh start"
    fi
}

# Main installation function
main() {
    show_banner
    parse_arguments "$@"
    
    log_info "Starting Project Index Universal Installer v$SCRIPT_VERSION"
    
    validate_system
    interactive_config
    setup_repository
    generate_config
    pull_images
    start_services
    show_completion
}

# Error handling
handle_error() {
    log_error "Installation failed at step: $1"
    log_info "Check the logs above for details"
    log_info "For help, visit: https://github.com/leanvibe/bee-hive/issues"
    exit 1
}

# Trap errors
trap 'handle_error "${BASH_COMMAND}"' ERR

# Run main function
main "$@"