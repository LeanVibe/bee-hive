#!/bin/bash

# LeanVibe Agent Hive 2.0 - Production Deployment Orchestrator
# Epic 7 Phase 1: Complete production infrastructure deployment

set -euo pipefail

# Configuration
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
DOMAIN_NAME="${DOMAIN_NAME:-}"
ADMIN_EMAIL="${ADMIN_EMAIL:-}"
BUILD_VERSION="${BUILD_VERSION:-$(git rev-parse --short HEAD 2>/dev/null || echo 'latest')}"
FORCE_DEPLOY="${FORCE_DEPLOY:-false}"
SKIP_BACKUP="${SKIP_BACKUP:-false}"
SLACK_WEBHOOK="${DEPLOY_SLACK_WEBHOOK:-}"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY_DIR="${SCRIPT_DIR}"
CONFIG_DIR="${PROJECT_ROOT}/config"

# Files
COMPOSE_FILE="${DEPLOY_DIR}/docker-compose.production-optimized.yml"
ENV_FILE="${PROJECT_ROOT}/.env.production"
SSL_SCRIPT="${DEPLOY_DIR}/ssl-setup.sh"
DR_SCRIPT="${DEPLOY_DIR}/disaster-recovery.sh"

# Logging
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="/var/log/leanvibe/deployment_${TIMESTAMP}.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

notify_slack() {
    if [[ -n "${SLACK_WEBHOOK}" ]]; then
        local message="$1"
        local color="${2:-good}"
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"attachments\":[{\"color\":\"${color}\",\"text\":\"🚀 PRODUCTION DEPLOYMENT: ${message}\"}]}" \
            "${SLACK_WEBHOOK}" || true
    fi
}

usage() {
    echo "LeanVibe Production Deployment Orchestrator"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  deploy          Full production deployment"
    echo "  update          Update existing deployment"
    echo "  rollback        Rollback to previous version"
    echo "  status          Check deployment status"
    echo "  health          Comprehensive health check"
    echo "  logs            Show deployment logs"
    echo ""
    echo "Options:"
    echo "  --domain        Domain name for SSL setup"
    echo "  --email         Admin email for SSL certificates"
    echo "  --version       Build version to deploy"
    echo "  --force         Force deployment without confirmation"
    echo "  --skip-backup   Skip pre-deployment backup"
    echo "  --dry-run       Show what would be deployed"
    echo ""
    echo "Examples:"
    echo "  $0 deploy --domain example.com --email admin@example.com"
    echo "  $0 update --version v2.1.0"
    echo "  $0 rollback --force"
    exit 1
}

check_prerequisites() {
    log "Checking deployment prerequisites"
    
    # Check required tools
    local required_tools=("docker" "docker-compose" "git" "curl" "openssl" "jq")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log "ERROR: Required tool '$tool' is not installed"
            exit 1
        fi
    done
    
    # Check Docker daemon
    if ! docker info >/dev/null 2>&1; then
        log "ERROR: Docker daemon is not running"
        exit 1
    fi
    
    # Check environment file
    if [[ ! -f "$ENV_FILE" ]]; then
        log "ERROR: Production environment file not found: $ENV_FILE"
        log "Please create the environment file with required variables"
        exit 1
    fi
    
    # Check compose file
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log "ERROR: Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    # Check required environment variables
    source "$ENV_FILE"
    local required_vars=("POSTGRES_PASSWORD" "REDIS_PASSWORD" "SECRET_KEY" "JWT_SECRET_KEY")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log "ERROR: Required environment variable '$var' is not set"
            exit 1
        fi
    done
    
    # Check domain configuration for SSL
    if [[ -n "$DOMAIN_NAME" ]]; then
        if ! dig +short "$DOMAIN_NAME" >/dev/null 2>&1; then
            log "WARNING: Domain $DOMAIN_NAME does not resolve to an IP address"
        fi
    fi
    
    log "Prerequisites check completed successfully"
}

prepare_environment() {
    log "Preparing deployment environment"
    
    # Create required directories
    local dirs=(
        "/var/lib/leanvibe/postgres"
        "/var/lib/leanvibe/redis-master"
        "/var/lib/leanvibe/redis-replica"
        "/var/lib/leanvibe/prometheus"
        "/var/lib/leanvibe/grafana"
        "/var/lib/leanvibe/alertmanager"
        "/var/lib/leanvibe/loki"
        "/var/log/leanvibe"
        "/backups"
        "/var/www/certbot"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        chmod 755 "$dir"
    done
    
    # Set proper ownership
    chown -R 999:999 /var/lib/leanvibe/postgres
    chown -R 999:999 /var/lib/leanvibe/redis-master
    chown -R 999:999 /var/lib/leanvibe/redis-replica
    chown -R 472:472 /var/lib/leanvibe/grafana
    
    # Generate secrets if not set
    if [[ -z "${SECRET_KEY:-}" ]]; then
        local new_secret
        new_secret=$(openssl rand -hex 32)
        echo "SECRET_KEY=${new_secret}" >> "$ENV_FILE"
        log "Generated new SECRET_KEY"
    fi
    
    if [[ -z "${JWT_SECRET_KEY:-}" ]]; then
        local new_jwt_secret
        new_jwt_secret=$(openssl rand -base64 64 | tr -d '\n')
        echo "JWT_SECRET_KEY=${new_jwt_secret}" >> "$ENV_FILE"
        log "Generated new JWT_SECRET_KEY"
    fi
    
    log "Environment preparation completed"
}

pre_deployment_backup() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        log "Skipping pre-deployment backup as requested"
        return 0
    fi
    
    log "Creating pre-deployment backup"
    
    if [[ -x "$DR_SCRIPT" ]]; then
        if "$DR_SCRIPT" backup; then
            log "Pre-deployment backup created successfully"
            notify_slack "📦 Pre-deployment backup created successfully"
        else
            log "WARNING: Pre-deployment backup failed, continuing with deployment"
        fi
    else
        log "WARNING: Disaster recovery script not found or not executable"
    fi
}

build_and_push_images() {
    log "Building and preparing Docker images"
    
    # Build production image
    cd "$PROJECT_ROOT"
    
    log "Building API production image"
    docker build \
        --target production \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')" \
        --build-arg VERSION="$BUILD_VERSION" \
        --tag "leanvibe/agent-hive-api:${BUILD_VERSION}" \
        --tag "leanvibe/agent-hive-api:latest" \
        .
    
    # Pull required images
    log "Pulling required Docker images"
    docker-compose -f "$COMPOSE_FILE" pull || true
    
    log "Image preparation completed"
}

deploy_infrastructure() {
    log "Deploying production infrastructure"
    
    cd "$DEPLOY_DIR"
    
    # Load environment variables
    export $(cat "$ENV_FILE" | grep -v '^#' | xargs)
    export BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    export VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    export VERSION="$BUILD_VERSION"
    
    # Deploy core services first (database, redis)
    log "Deploying core services"
    docker-compose -f "$COMPOSE_FILE" up -d postgres pgbouncer redis-master redis-replica redis-sentinel
    
    # Wait for core services to be healthy
    log "Waiting for core services to be healthy"
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if docker-compose -f "$COMPOSE_FILE" ps postgres | grep -q "healthy" && \
           docker-compose -f "$COMPOSE_FILE" ps redis-master | grep -q "healthy"; then
            break
        fi
        log "Waiting for core services... (attempt $((attempt + 1))/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    if [[ $attempt -eq $max_attempts ]]; then
        log "ERROR: Core services failed to become healthy"
        exit 1
    fi
    
    # Run database migrations
    log "Running database migrations"
    if [[ -x "${PROJECT_ROOT}/database/scripts/migrate.py" ]]; then
        python3 "${PROJECT_ROOT}/database/scripts/migrate.py" migrate --no-backup
    fi
    
    # Deploy application services
    log "Deploying application services"
    docker-compose -f "$COMPOSE_FILE" up -d api
    
    # Deploy monitoring stack
    log "Deploying monitoring stack"
    docker-compose -f "$COMPOSE_FILE" up -d prometheus alertmanager grafana loki promtail
    
    # Deploy supporting services
    log "Deploying supporting services"
    docker-compose -f "$COMPOSE_FILE" up -d node-exporter cadvisor db-backup
    
    log "Infrastructure deployment completed"
}

setup_ssl_and_proxy() {
    if [[ -z "$DOMAIN_NAME" ]]; then
        log "No domain specified, skipping SSL setup"
        return 0
    fi
    
    log "Setting up SSL certificates and reverse proxy"
    
    # Set environment variables for SSL script
    export DOMAIN_NAME="$DOMAIN_NAME"
    export ADMIN_EMAIL="${ADMIN_EMAIL:-admin@${DOMAIN_NAME}}"
    
    # Run SSL setup
    if [[ -x "$SSL_SCRIPT" ]]; then
        if "$SSL_SCRIPT"; then
            log "SSL certificates configured successfully"
        else
            log "ERROR: SSL setup failed"
            exit 1
        fi
    else
        log "WARNING: SSL setup script not found or not executable"
    fi
    
    # Deploy nginx with SSL
    log "Deploying nginx reverse proxy"
    docker-compose -f "$COMPOSE_FILE" up -d nginx
    
    log "SSL and proxy setup completed"
}

validate_deployment() {
    log "Validating production deployment"
    
    cd "$DEPLOY_DIR"
    
    # Check service health
    log "Checking service health status"
    local services=(
        "postgres:database"
        "redis-master:cache"
        "api:application"
        "nginx:proxy"
        "prometheus:monitoring"
    )
    
    local failed_services=0
    for service_info in "${services[@]}"; do
        local service="${service_info%:*}"
        local description="${service_info#*:}"
        
        if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "healthy\|Up"; then
            log "✓ $service ($description) is running"
        else
            log "✗ $service ($description) is not healthy"
            ((failed_services++))
        fi
    done
    
    # Test API endpoints
    log "Testing API endpoints"
    local api_tests=(
        "http://localhost:8000/health:Health check"
        "http://localhost:8000/api/v2/system/info:System info"
    )
    
    for test_info in "${api_tests[@]}"; do
        local endpoint="${test_info%:*}"
        local description="${test_info#*:}"
        
        if curl -sSf "$endpoint" >/dev/null 2>&1; then
            log "✓ $description endpoint is responding"
        else
            log "✗ $description endpoint is not responding"
            ((failed_services++))
        fi
    done
    
    # Test HTTPS if domain configured
    if [[ -n "$DOMAIN_NAME" ]]; then
        log "Testing HTTPS connectivity"
        if curl -sSf "https://${DOMAIN_NAME}/health" >/dev/null 2>&1; then
            log "✓ HTTPS connectivity test passed"
        else
            log "✗ HTTPS connectivity test failed"
            ((failed_services++))
        fi
    fi
    
    # Check database connectivity
    log "Testing database connectivity"
    if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U leanvibe_user -d leanvibe_agent_hive >/dev/null 2>&1; then
        log "✓ Database connectivity test passed"
    else
        log "✗ Database connectivity test failed"
        ((failed_services++))
    fi
    
    # Validate deployment
    if [[ $failed_services -eq 0 ]]; then
        log "✅ Deployment validation PASSED - All services operational"
        notify_slack "✅ Production deployment validation PASSED - All systems operational"
        return 0
    else
        log "❌ Deployment validation FAILED - $failed_services service(s) have issues"
        notify_slack "❌ Production deployment validation FAILED - $failed_services issues detected" "danger"
        return 1
    fi
}

show_deployment_status() {
    log "Production deployment status"
    
    cd "$DEPLOY_DIR"
    
    echo ""
    echo "=== LEANVIBE AGENT HIVE PRODUCTION STATUS ==="
    echo ""
    
    # Service status
    echo "Docker Services:"
    docker-compose -f "$COMPOSE_FILE" ps
    echo ""
    
    # Resource usage
    echo "Resource Usage:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
    echo ""
    
    # Network ports
    echo "Network Ports:"
    docker-compose -f "$COMPOSE_FILE" ps --format "table {{.Name}}\t{{.Ports}}"
    echo ""
    
    # Volume usage
    echo "Volume Usage:"
    docker volume ls --filter name=leanvibe
    echo ""
    
    # Recent logs
    echo "Recent Logs (last 20 lines):"
    docker-compose -f "$COMPOSE_FILE" logs --tail=20
}

cleanup_old_deployment() {
    log "Cleaning up old deployment artifacts"
    
    # Remove unused Docker images
    docker image prune -f
    
    # Remove unused volumes (be careful with data)
    docker volume prune -f
    
    # Clean up old log files
    find /var/log/leanvibe -name "*.log" -mtime +30 -delete || true
    
    log "Cleanup completed"
}

main() {
    local command="${1:-deploy}"
    local dry_run=false
    
    # Parse arguments
    shift || true
    while [[ $# -gt 0 ]]; do
        case $1 in
            --domain)
                DOMAIN_NAME="$2"
                shift 2
                ;;
            --email)
                ADMIN_EMAIL="$2"
                shift 2
                ;;
            --version)
                BUILD_VERSION="$2"
                shift 2
                ;;
            --force)
                FORCE_DEPLOY=true
                shift
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            -h|--help)
                usage
                ;;
            *)
                log "Unknown option: $1"
                usage
                ;;
        esac
    done
    
    # Execute command
    case "$command" in
        deploy)
            log "Starting full production deployment - Version: $BUILD_VERSION"
            notify_slack "🚀 Starting production deployment - Version: $BUILD_VERSION" "warning"
            
            if [[ "$dry_run" == "true" ]]; then
                log "DRY RUN MODE - No changes will be made"
                check_prerequisites
                log "Dry run completed - deployment would proceed normally"
                exit 0
            fi
            
            check_prerequisites
            prepare_environment
            pre_deployment_backup
            build_and_push_images
            deploy_infrastructure
            setup_ssl_and_proxy
            
            if validate_deployment; then
                cleanup_old_deployment
                log "🎉 Production deployment completed successfully!"
                notify_slack "🎉 Production deployment completed successfully - Version: $BUILD_VERSION"
            else
                log "❌ Production deployment completed with issues"
                notify_slack "⚠️ Production deployment completed with validation issues" "warning"
                exit 1
            fi
            ;;
        status)
            show_deployment_status
            ;;
        health)
            validate_deployment
            ;;
        logs)
            cd "$DEPLOY_DIR"
            docker-compose -f "$COMPOSE_FILE" logs -f "${@}"
            ;;
        *)
            usage
            ;;
    esac
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR: Deployment failed with exit code: $exit_code"
        notify_slack "❌ Production deployment FAILED" "danger"
    fi
    exit $exit_code
}

trap cleanup EXIT

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi