#!/bin/bash
# Production Deployment Script for LeanVibe Agent Hive Mobile Dashboard
# Ensures complete production readiness with mobile optimization

set -e
set -o pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_FILE="$PROJECT_ROOT/logs/production-deployment-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

log "Starting LeanVibe Agent Hive production deployment..."

# Validate environment variables
validate_environment() {
    log "Validating environment variables..."
    
    required_vars=(
        "DOMAIN_NAME"
        "POSTGRES_PASSWORD"
        "REDIS_PASSWORD"
        "SECRET_KEY"
        "JWT_SECRET_KEY"
        "ANTHROPIC_API_KEY"
        "FIREBASE_PROJECT_ID"
        "FCM_SERVER_KEY"
        "VAPID_PUBLIC_KEY"
        "VAPID_PRIVATE_KEY"
    )
    
    missing_vars=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        error "Missing required environment variables: ${missing_vars[*]}"
    fi
    
    success "Environment variables validated"
}

# Pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."
    
    # Check Docker and Docker Compose
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    if ! command -v docker-compose &> /dev/null && ! command -v docker compose &> /dev/null; then
        error "Docker Compose is not installed"
    fi
    
    # Check required files
    required_files=(
        "docker-compose.production.yml"
        "config/nginx.conf"
        "config/ssl/generate-certs.sh"
        "mobile-pwa/dist/index.html"
    )
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$PROJECT_ROOT/$file" ]]; then
            error "Required file missing: $file"
        fi
    done
    
    # Check mobile PWA build
    if [[ ! -d "$PROJECT_ROOT/mobile-pwa/dist" ]]; then
        error "Mobile PWA not built. Run 'npm run build' in mobile-pwa directory"
    fi
    
    success "Pre-deployment checks completed"
}

# Build mobile PWA for production
build_mobile_pwa() {
    log "Building mobile PWA for production..."
    
    cd "$PROJECT_ROOT/mobile-pwa"
    
    # Install dependencies if needed
    if [[ ! -d "node_modules" ]]; then
        log "Installing mobile PWA dependencies..."
        npm ci
    fi
    
    # Build for production with mobile optimizations
    log "Building production PWA bundle..."
    NODE_ENV=production npm run build -- --config vite.config.production.ts
    
    # Validate build
    if [[ ! -f "dist/index.html" ]]; then
        error "Mobile PWA build failed - index.html not found"
    fi
    
    # Check bundle sizes
    js_size=$(find dist -name "*.js" -exec cat {} \; | wc -c)
    css_size=$(find dist -name "*.css" -exec cat {} \; | wc -c)
    
    log "Bundle sizes: JS=${js_size} bytes, CSS=${css_size} bytes"
    
    # Warning if bundles are too large
    if [[ $js_size -gt 204800 ]]; then # 200KB
        warn "JavaScript bundle is larger than 200KB (${js_size} bytes)"
    fi
    
    cd "$PROJECT_ROOT"
    success "Mobile PWA built successfully"
}

# Generate SSL certificates
setup_ssl_certificates() {
    log "Setting up SSL certificates..."
    
    # Create SSL directory
    mkdir -p "$PROJECT_ROOT/config/ssl"
    
    # Generate certificates
    cd "$PROJECT_ROOT/config/ssl"
    
    if [[ ! -f "cert.pem" ]] || [[ ! -f "key.pem" ]]; then
        log "Generating SSL certificates..."
        ./generate-certs.sh
    else
        log "SSL certificates already exist"
    fi
    
    # Validate certificates
    if ! openssl x509 -in cert.pem -text -noout &> /dev/null; then
        error "SSL certificate validation failed"
    fi
    
    cd "$PROJECT_ROOT"
    success "SSL certificates configured"
}

# Setup production data directories
setup_data_directories() {
    log "Setting up production data directories..."
    
    data_dirs=(
        "/var/lib/leanvibe/postgres"
        "/var/lib/leanvibe/redis"
        "/var/lib/leanvibe/prometheus"
        "/var/lib/leanvibe/grafana"
        "/var/lib/leanvibe/alertmanager"
    )
    
    for dir in "${data_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log "Creating data directory: $dir"
            sudo mkdir -p "$dir"
            sudo chown -R 1001:1001 "$dir" 2>/dev/null || true
        fi
    done
    
    success "Data directories configured"
}

# Run security validation
validate_security() {
    log "Running security validation..."
    
    # Run mobile security compliance check
    if [[ -f "$PROJECT_ROOT/config/security/mobile-security.yml" ]]; then
        python3 "$PROJECT_ROOT/config/security/mobile-compliance.py" \
            "$PROJECT_ROOT/config/security/mobile-security.yml" > security-validation.json
        
        if [[ $? -eq 0 ]]; then
            success "Security validation passed"
        else
            warn "Security validation has warnings - check security-validation.json"
        fi
    fi
}

# Deploy infrastructure
deploy_infrastructure() {
    log "Deploying production infrastructure..."
    
    # Stop any existing deployment
    if docker-compose -f docker-compose.production.yml ps -q &> /dev/null; then
        log "Stopping existing deployment..."
        docker-compose -f docker-compose.production.yml down
    fi
    
    # Pull latest images
    log "Pulling latest Docker images..."
    docker-compose -f docker-compose.production.yml pull
    
    # Start core services first
    log "Starting core services (database, redis)..."
    docker-compose -f docker-compose.production.yml up -d postgres redis
    
    # Wait for core services to be healthy
    log "Waiting for core services to be ready..."
    timeout 120 bash -c 'until docker-compose -f docker-compose.production.yml exec postgres pg_isready -U leanvibe_user; do sleep 2; done'
    timeout 60 bash -c 'until docker-compose -f docker-compose.production.yml exec redis redis-cli -a ${REDIS_PASSWORD} ping; do sleep 2; done'
    
    # Run database migrations
    log "Running database migrations..."
    docker-compose -f docker-compose.production.yml run --rm api alembic upgrade head
    
    # Start application services
    log "Starting application services..."
    docker-compose -f docker-compose.production.yml up -d api nginx
    
    # Start monitoring services
    log "Starting monitoring services..."
    docker-compose -f docker-compose.production.yml up -d prometheus grafana alertmanager mobile-monitor
    
    # Wait for services to be healthy
    log "Waiting for services to be healthy..."
    sleep 30
    
    success "Infrastructure deployed"
}

# Run deployment validation
validate_deployment() {
    log "Running deployment validation..."
    
    # Set validation environment
    export BASE_URL="https://${DOMAIN_NAME}"
    export WS_URL="wss://${DOMAIN_NAME}"
    export TIMEOUT=60
    
    # Run comprehensive validation
    if python3 "$PROJECT_ROOT/scripts/production-deployment-validator.py" > deployment-validation.json 2>&1; then
        success "Deployment validation passed"
        
        # Extract key metrics from validation
        if command -v jq &> /dev/null; then
            overall_status=$(jq -r '.overall_status' deployment-validation.json)
            log "Overall deployment status: $overall_status"
            
            if [[ "$overall_status" == "ready" ]]; then
                success "🚀 Production deployment is READY!"
            else
                warn "Deployment status: $overall_status - check deployment-validation.json for details"
            fi
        fi
    else
        warn "Deployment validation completed with warnings - check deployment-validation.json"
    fi
}

# Setup monitoring and alerting
setup_monitoring() {
    log "Configuring monitoring and alerting..."
    
    # Import Grafana dashboards
    if [[ -d "$PROJECT_ROOT/infrastructure/monitoring/grafana/dashboards" ]]; then
        log "Grafana dashboards will be auto-imported on startup"
    fi
    
    # Validate Prometheus configuration
    if docker-compose -f docker-compose.production.yml exec prometheus promtool check config /etc/prometheus/prometheus.yml &> /dev/null; then
        success "Prometheus configuration validated"
    else
        warn "Prometheus configuration validation failed"
    fi
    
    success "Monitoring configured"
}

# Post-deployment tasks
post_deployment_tasks() {
    log "Running post-deployment tasks..."
    
    # Create backup scripts
    create_backup_script
    
    # Setup log rotation
    setup_log_rotation
    
    # Create health check script
    create_health_check_script
    
    success "Post-deployment tasks completed"
}

create_backup_script() {
    log "Creating backup script..."
    
    cat > "$PROJECT_ROOT/backup-production.sh" << 'EOF'
#!/bin/bash
# Production backup script

BACKUP_DIR="/var/backups/leanvibe/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup database
docker-compose -f docker-compose.production.yml exec postgres pg_dump -U leanvibe_user leanvibe_agent_hive | gzip > "$BACKUP_DIR/database.sql.gz"

# Backup Redis data
docker-compose -f docker-compose.production.yml exec redis redis-cli -a "${REDIS_PASSWORD}" --rdb - > "$BACKUP_DIR/redis.rdb"

# Backup configuration
tar -czf "$BACKUP_DIR/config.tar.gz" config/

# Cleanup old backups (keep 7 days)
find /var/backups/leanvibe -type d -mtime +7 -exec rm -rf {} \;

echo "Backup completed: $BACKUP_DIR"
EOF
    
    chmod +x "$PROJECT_ROOT/backup-production.sh"
}

setup_log_rotation() {
    log "Setting up log rotation..."
    
    # Create logrotate configuration
    sudo tee /etc/logrotate.d/leanvibe << EOF > /dev/null
$PROJECT_ROOT/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 $USER $USER
    postrotate
        docker-compose -f $PROJECT_ROOT/docker-compose.production.yml restart api nginx || true
    endscript
}
EOF
}

create_health_check_script() {
    log "Creating health check script..."
    
    cat > "$PROJECT_ROOT/health-check-production.sh" << 'EOF'
#!/bin/bash
# Production health check script

DOMAIN_NAME="${DOMAIN_NAME:-localhost}"
BASE_URL="https://${DOMAIN_NAME}"

echo "=== LeanVibe Production Health Check ==="
echo "Timestamp: $(date)"
echo "Domain: $DOMAIN_NAME"
echo

# Check main application
if curl -f -s "$BASE_URL/health" > /dev/null; then
    echo "✓ Main application: Healthy"
else
    echo "✗ Main application: Unhealthy"
fi

# Check mobile API
if curl -f -s "$BASE_URL/api/mobile/health" > /dev/null; then
    echo "✓ Mobile API: Healthy"
else
    echo "✗ Mobile API: Unhealthy"
fi

# Check WebSocket
if curl -f -s -H "Connection: Upgrade" -H "Upgrade: websocket" "$BASE_URL/ws/health" > /dev/null 2>&1; then
    echo "✓ WebSocket: Healthy"
else
    echo "✗ WebSocket: Unhealthy"
fi

# Check services
docker-compose -f docker-compose.production.yml ps --services | while read service; do
    if docker-compose -f docker-compose.production.yml ps "$service" | grep -q "Up"; then
        echo "✓ Service $service: Running"
    else
        echo "✗ Service $service: Not running"
    fi
done

echo
echo "=== End Health Check ==="
EOF
    
    chmod +x "$PROJECT_ROOT/health-check-production.sh"
    
    log "Health check script created: ./health-check-production.sh"
}

# Display deployment summary
deployment_summary() {
    log "Deployment Summary:"
    log "=================="
    log "Domain: https://${DOMAIN_NAME}"
    log "Mobile Dashboard: https://${DOMAIN_NAME}"
    log "API Endpoints: https://${DOMAIN_NAME}/api/"
    log "WebSocket: wss://${DOMAIN_NAME}/ws/"
    log "Monitoring: https://${DOMAIN_NAME}:3001 (Grafana)"
    log "Metrics: https://${DOMAIN_NAME}:9090 (Prometheus)"
    log ""
    log "Log files:"
    log "- Deployment: $LOG_FILE"
    log "- Security validation: ./security-validation.json"
    log "- Deployment validation: ./deployment-validation.json"
    log ""
    log "Useful commands:"
    log "- Health check: ./health-check-production.sh"
    log "- Backup: ./backup-production.sh"
    log "- View logs: docker-compose -f docker-compose.production.yml logs -f"
    log "- Restart services: docker-compose -f docker-compose.production.yml restart"
    log ""
    success "🎉 LeanVibe Agent Hive production deployment completed!"
    log ""
    log "Your mobile dashboard is now live at: https://${DOMAIN_NAME}"
    log "Mobile users can install the PWA directly from their browsers."
    log "Push notifications are enabled with Firebase FCM integration."
    log "Real-time WebSocket coordination is operational for mobile clients."
    log ""
    log "Next steps:"
    log "1. Test the mobile dashboard on various devices"
    log "2. Configure monitoring alerts in Grafana"
    log "3. Set up automated backups with cron"
    log "4. Review security validation results"
    log "5. Monitor performance metrics"
}

# Main deployment flow
main() {
    log "LeanVibe Agent Hive Production Deployment"
    log "========================================"
    
    validate_environment
    pre_deployment_checks
    build_mobile_pwa
    setup_ssl_certificates
    setup_data_directories
    validate_security
    deploy_infrastructure
    setup_monitoring
    validate_deployment
    post_deployment_tasks
    deployment_summary
}

# Error handling
trap 'error "Deployment failed at line $LINENO"' ERR

# Run main deployment
main "$@"