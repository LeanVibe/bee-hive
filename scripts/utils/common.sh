#!/bin/bash

# LeanVibe Agent Hive 2.0 - Common Utilities
# Shared functions and utilities for all scripts
# 
# Usage: source scripts/utils/common.sh

# Color definitions
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly PURPLE='\033[0;35m'
readonly CYAN='\033[0;36m'
readonly WHITE='\033[1;37m'
readonly NC='\033[0m' # No Color

# Common constants
readonly SCRIPT_VERSION="2.0.0"
readonly PROJECT_NAME="LeanVibe Agent Hive 2.0"
readonly API_PORT=8000
readonly SANDBOX_PORT=8001
readonly PROMETHEUS_PORT=9090
readonly GRAFANA_PORT=3001

# Global variables (set by including scripts)
PROJECT_ROOT=""
SCRIPT_DIR=""
LOG_FILE=""
VERBOSE=${VERBOSE:-false}
DEBUG=${DEBUG:-false}

#======================================
# Logging Functions
#======================================

# Initialize logging
init_logging() {
    local script_name="${1:-unknown}"
    local log_dir="${PROJECT_ROOT}/logs"
    
    mkdir -p "$log_dir"
    LOG_FILE="$log_dir/${script_name}-$(date '+%Y%m%d-%H%M%S').log"
    
    if [[ "$VERBOSE" == "true" ]]; then
        log "INFO" "Logging initialized: $LOG_FILE"
    fi
}

# Main logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Console output with colors
    case "$level" in
        "DEBUG")   [[ "$DEBUG" == "true" ]] && echo -e "${PURPLE}[DEBUG]${NC} $message" ;;
        "INFO")    echo -e "${BLUE}[INFO]${NC}  $message" ;;
        "WARN")    echo -e "${YELLOW}[WARN]${NC}  $message" ;;
        "ERROR")   echo -e "${RED}[ERROR]${NC} $message" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} $message" ;;
        "STEP")    echo -e "${CYAN}[STEP]${NC} $message" ;;
        "HEADER")  echo -e "${WHITE}[HEADER]${NC} $message" ;;
    esac
    
    # File output (if log file is set)
    if [[ -n "$LOG_FILE" ]]; then
        echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
    fi
}

# Specialized logging functions
log_debug() { log "DEBUG" "$@"; }
log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }
log_success() { log "SUCCESS" "$@"; }
log_step() { log "STEP" "$@"; }
log_header() { log "HEADER" "$@"; }

#======================================
# Display Functions
#======================================

show_banner() {
    local title="${1:-$PROJECT_NAME}"
    local subtitle="${2:-Professional Development Platform}"
    
    clear
    cat << EOF
╔══════════════════════════════════════════════════════════════════════════════╗
║$(printf "%*s" $(((80-${#title})/2)) "")${title}$(printf "%*s" $(((80-${#title})/2)) "")║
║$(printf "%*s" $(((80-${#subtitle})/2)) "")${subtitle}$(printf "%*s" $(((80-${#subtitle})/2)) "")║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
    echo
}

show_progress() {
    local current="$1"
    local total="$2"
    local description="${3:-Processing}"
    local width=50
    
    local percentage=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))
    
    printf "\r${CYAN}%s${NC} [" "$description"
    printf "%*s" $filled | tr ' ' '█'
    printf "%*s" $empty | tr ' ' '░'
    printf "] %d%% (%d/%d)" $percentage $current $total
    
    if [[ $current -eq $total ]]; then
        echo
    fi
}

show_spinner() {
    local pid=$1
    local message="${2:-Processing...}"
    local spinner_chars="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    local i=0
    
    while kill -0 $pid 2>/dev/null; do
        printf "\r${BLUE}%s${NC} %s" "${spinner_chars:$i:1}" "$message"
        i=$(((i + 1) % ${#spinner_chars}))
        sleep 0.1
    done
    
    printf "\r%*s\r" $((${#message} + 10)) ""
}

#======================================
# Validation Functions
#======================================

check_command() {
    local cmd="$1"
    local description="${2:-$cmd}"
    
    if command -v "$cmd" &> /dev/null; then
        log_debug "$description is available"
        return 0
    else
        log_error "$description is not available"
        return 1
    fi
}

check_port() {
    local port="$1"
    local description="${2:-Port $port}"
    
    if lsof -i ":$port" &> /dev/null; then
        log_debug "$description is in use"
        return 0
    else
        log_debug "$description is available"
        return 1
    fi
}

check_url() {
    local url="$1"
    local timeout="${2:-10}"
    local description="${3:-$url}"
    
    if curl -s --max-time "$timeout" "$url" > /dev/null 2>&1; then
        log_debug "$description is accessible"
        return 0
    else
        log_debug "$description is not accessible"
        return 1
    fi
}

validate_python_version() {
    local min_version="${1:-3.12}"
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        return 1
    fi
    
    local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    local python_major=$(echo "$python_version" | cut -d'.' -f1)
    local python_minor=$(echo "$python_version" | cut -d'.' -f2)
    local min_major=$(echo "$min_version" | cut -d'.' -f1)
    local min_minor=$(echo "$min_version" | cut -d'.' -f2)
    
    if [[ $python_major -gt $min_major ]] || \
       [[ $python_major -eq $min_major && $python_minor -ge $min_minor ]]; then
        log_debug "Python $python_version meets minimum requirement ($min_version)"
        return 0
    else
        log_error "Python $python_version does not meet minimum requirement ($min_version)"
        return 1
    fi
}

validate_docker() {
    if ! check_command "docker" "Docker"; then
        return 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        return 1
    fi
    
    # Check Docker Compose
    if ! check_command "docker-compose" "Docker Compose" && \
       ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available"
        return 1
    fi
    
    log_debug "Docker environment is valid"
    return 0
}

#======================================
# File System Functions
#======================================

ensure_directory() {
    local dir="$1"
    local description="${2:-Directory $dir}"
    
    if [[ ! -d "$dir" ]]; then
        log_debug "Creating $description"
        mkdir -p "$dir"
    fi
    
    return 0
}

backup_file() {
    local file="$1"
    local backup_suffix="${2:-.backup.$(date +%s)}"
    
    if [[ -f "$file" ]]; then
        local backup_file="${file}${backup_suffix}"
        cp "$file" "$backup_file"
        log_debug "Backed up $file to $backup_file"
        echo "$backup_file"
    fi
}

safe_remove() {
    local path="$1"
    local description="${2:-$path}"
    
    if [[ -e "$path" ]]; then
        rm -rf "$path"
        log_debug "Removed $description"
    fi
}

#======================================
# Process Management Functions
#======================================

wait_for_service() {
    local url="$1"
    local timeout="${2:-60}"
    local description="${3:-Service}"
    local interval="${4:-2}"
    
    log_info "Waiting for $description to be ready..."
    
    local retries=$((timeout / interval))
    while [[ $retries -gt 0 ]]; do
        if check_url "$url" 5; then
            log_success "$description is ready"
            return 0
        fi
        
        retries=$((retries - 1))
        sleep "$interval"
        echo -n "."
    done
    
    echo
    log_error "$description failed to start within $timeout seconds"
    return 1
}

kill_process_by_port() {
    local port="$1"
    local signal="${2:-TERM}"
    
    local pid=$(lsof -ti ":$port" 2>/dev/null)
    if [[ -n "$pid" ]]; then
        kill "-$signal" "$pid" 2>/dev/null || true
        log_debug "Sent $signal signal to process $pid on port $port"
        return 0
    else
        log_debug "No process found on port $port"
        return 1
    fi
}

cleanup_pids() {
    local pid_files=("$@")
    
    for pid_file in "${pid_files[@]}"; do
        if [[ -f "$pid_file" ]]; then
            local pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                log_debug "Stopping process $pid from $pid_file"
                kill -TERM "$pid" 2>/dev/null || true
                
                # Wait a moment for graceful shutdown
                sleep 2
                
                # Force kill if still running
                if kill -0 "$pid" 2>/dev/null; then
                    kill -KILL "$pid" 2>/dev/null || true
                fi
            fi
            
            rm -f "$pid_file"
        fi
    done
}

#======================================
# Configuration Functions
#======================================

load_env_file() {
    local env_file="${1:-$PROJECT_ROOT/.env.local}"
    
    if [[ -f "$env_file" ]]; then
        log_debug "Loading environment from $env_file"
        set -a  # Export all variables
        source "$env_file"
        set +a
        return 0
    else
        log_warn "Environment file not found: $env_file"
        return 1
    fi
}

get_config_value() {
    local key="$1"
    local default_value="${2:-}"
    local config_file="${3:-$PROJECT_ROOT/.env.local}"
    
    if [[ -f "$config_file" ]]; then
        local value=$(grep "^$key=" "$config_file" 2>/dev/null | cut -d'=' -f2-)
        echo "${value:-$default_value}"
    else
        echo "$default_value"
    fi
}

set_config_value() {
    local key="$1"
    local value="$2"
    local config_file="${3:-$PROJECT_ROOT/.env.local}"
    
    ensure_directory "$(dirname "$config_file")"
    
    if [[ -f "$config_file" ]] && grep -q "^$key=" "$config_file"; then
        # Update existing value
        sed -i.bak "s/^$key=.*/$key=$value/" "$config_file"
        rm -f "$config_file.bak"
        log_debug "Updated $key in $config_file"
    else
        # Add new value
        echo "$key=$value" >> "$config_file"
        log_debug "Added $key to $config_file"
    fi
}

#======================================
# Docker Functions
#======================================

get_compose_command() {
    if command -v docker-compose &> /dev/null; then
        echo "docker-compose"
    else
        echo "docker compose"
    fi
}

docker_service_status() {
    local service="$1"
    local compose_file="${2:-docker-compose.yml}"
    local compose_cmd=$(get_compose_command)
    
    cd "$PROJECT_ROOT"
    if $compose_cmd -f "$compose_file" ps "$service" 2>/dev/null | grep -q "Up"; then
        return 0
    else
        return 1
    fi
}

start_docker_service() {
    local service="$1"
    local compose_file="${2:-docker-compose.yml}"
    local compose_cmd=$(get_compose_command)
    
    cd "$PROJECT_ROOT"
    log_info "Starting Docker service: $service"
    
    if $compose_cmd -f "$compose_file" up -d "$service"; then
        log_success "Started $service"
        return 0
    else
        log_error "Failed to start $service"
        return 1
    fi
}

stop_docker_service() {
    local service="$1"
    local compose_file="${2:-docker-compose.yml}"
    local compose_cmd=$(get_compose_command)
    
    cd "$PROJECT_ROOT"
    log_info "Stopping Docker service: $service"
    
    if $compose_cmd -f "$compose_file" stop "$service"; then
        log_success "Stopped $service"
        return 0
    else
        log_error "Failed to stop $service"
        return 1
    fi
}

#======================================
# Python Environment Functions
#======================================

activate_venv() {
    local venv_path="${1:-$PROJECT_ROOT/venv}"
    
    if [[ -f "$venv_path/bin/activate" ]]; then
        source "$venv_path/bin/activate"
        log_debug "Activated virtual environment: $venv_path"
        return 0
    else
        log_error "Virtual environment not found: $venv_path"
        return 1
    fi
}

install_python_package() {
    local package="$1"
    local version="${2:-}"
    
    local pkg_spec="$package"
    if [[ -n "$version" ]]; then
        pkg_spec="$package==$version"
    fi
    
    log_info "Installing Python package: $pkg_spec"
    if pip install "$pkg_spec"; then
        log_success "Installed $pkg_spec"
        return 0
    else
        log_error "Failed to install $pkg_spec"
        return 1
    fi
}

#======================================
# Health Check Functions
#======================================

health_check_api() {
    local url="${1:-http://localhost:$API_PORT/health}"
    local timeout="${2:-10}"
    
    if check_url "$url" "$timeout" "API Health Check"; then
        log_success "API health check passed"
        return 0
    else
        log_error "API health check failed"
        return 1
    fi
}

health_check_database() {
    local compose_file="${1:-docker-compose.yml}"
    local compose_cmd=$(get_compose_command)
    
    cd "$PROJECT_ROOT"
    if $compose_cmd -f "$compose_file" exec -T postgres pg_isready -U leanvibe_user &> /dev/null; then
        log_success "Database health check passed"
        return 0
    else
        log_error "Database health check failed"
        return 1
    fi
}

health_check_redis() {
    local compose_file="${1:-docker-compose.yml}"
    local compose_cmd=$(get_compose_command)
    
    cd "$PROJECT_ROOT"
    if $compose_cmd -f "$compose_file" exec -T redis redis-cli ping &> /dev/null; then
        log_success "Redis health check passed"
        return 0
    else
        log_error "Redis health check failed"
        return 1
    fi
}

#======================================
# Error Handling Functions
#======================================

setup_error_handling() {
    local cleanup_function="${1:-cleanup_default}"
    
    trap "$cleanup_function" ERR INT TERM
    set -euo pipefail
}

cleanup_default() {
    local exit_code=$?
    
    log_error "Script failed with exit code $exit_code"
    
    # Clean up any PID files
    cleanup_pids "$PROJECT_ROOT"/.*.pid
    
    exit $exit_code
}

#======================================
# Utility Functions
#======================================

confirm() {
    local message="${1:-Are you sure?}"
    local default="${2:-n}"
    
    local prompt="$message"
    case "$default" in
        [Yy]|[Yy][Ee][Ss]) prompt="$prompt [Y/n]" ;;
        *) prompt="$prompt [y/N]" ;;
    esac
    
    read -p "$prompt " -n 1 -r
    echo
    
    case "$REPLY" in
        [Yy]|[Yy][Ee][Ss]) return 0 ;;
        [Nn]|[Nn][Oo]) return 1 ;;
        "") 
            case "$default" in
                [Yy]|[Yy][Ee][Ss]) return 0 ;;
                *) return 1 ;;
            esac
            ;;
        *) return 1 ;;
    esac
}

format_duration() {
    local seconds="$1"
    local minutes=$((seconds / 60))
    local remaining_seconds=$((seconds % 60))
    
    if [[ $minutes -gt 0 ]]; then
        echo "${minutes}m ${remaining_seconds}s"
    else
        echo "${seconds}s"
    fi
}

get_timestamp() {
    local format="${1:-iso}"
    
    case "$format" in
        "iso") date -u +%Y-%m-%dT%H:%M:%SZ ;;
        "filename") date +%Y%m%d-%H%M%S ;;
        "readable") date '+%Y-%m-%d %H:%M:%S' ;;
        *) date ;;
    esac
}

# Initialize common variables when sourced
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    # Script is being sourced
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    
    # Set defaults if not already set
    VERBOSE=${VERBOSE:-false}
    DEBUG=${DEBUG:-false}
fi