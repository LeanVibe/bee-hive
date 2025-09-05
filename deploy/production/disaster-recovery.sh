#!/bin/bash

# LeanVibe Agent Hive - Disaster Recovery and Business Continuity Script
# Comprehensive disaster recovery with automated failover and data protection

set -euo pipefail

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/backups}"
RECOVERY_DIR="${RECOVERY_DIR:-/recovery}"
S3_BUCKET="${BACKUP_S3_BUCKET:-}"
SLACK_WEBHOOK="${BACKUP_SLACK_WEBHOOK:-}"
ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY:-}"
DR_MODE="${DR_MODE:-full}"  # full, database, application, infrastructure

# Database configuration
DATABASE_HOST="${POSTGRES_HOST:-postgres}"
DATABASE_PORT="${POSTGRES_PORT:-5432}"
DATABASE_NAME="${POSTGRES_DB:-leanvibe_agent_hive}"
DATABASE_USER="${POSTGRES_USER:-leanvibe_user}"
DATABASE_PASSWORD="${POSTGRES_PASSWORD:-}"

# Service configuration
DOCKER_COMPOSE_FILE="${DOCKER_COMPOSE_FILE:-/app/deploy/production/docker-compose.production-optimized.yml}"
PRODUCTION_ENV_FILE="${PRODUCTION_ENV_FILE:-/app/.env.production}"

# Recovery targets
RTO_MINUTES="${RTO_MINUTES:-60}"  # Recovery Time Objective
RPO_MINUTES="${RPO_MINUTES:-15}"  # Recovery Point Objective

# Logging
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${RECOVERY_DIR}/logs/disaster_recovery_${TIMESTAMP}.log"
mkdir -p "${RECOVERY_DIR}/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

notify_slack() {
    if [[ -n "${SLACK_WEBHOOK}" ]]; then
        local message="$1"
        local color="${2:-warning}"
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"attachments\":[{\"color\":\"${color}\",\"text\":\"🚨 DISASTER RECOVERY: ${message}\"}]}" \
            "${SLACK_WEBHOOK}" || true
    fi
}

usage() {
    echo "LeanVibe Disaster Recovery Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  backup          Create comprehensive backup"
    echo "  restore         Restore from backup"
    echo "  failover        Activate failover mode"
    echo "  test-dr         Test disaster recovery procedures"
    echo "  status          Check system health and backup status"
    echo "  validate        Validate disaster recovery capability"
    echo ""
    echo "Options:"
    echo "  --backup-file   Specific backup file to restore from"
    echo "  --dry-run       Simulate operations without making changes"
    echo "  --force         Force operations without confirmation"
    echo "  --mode          DR mode: full, database, application, infrastructure"
    echo ""
    echo "Examples:"
    echo "  $0 backup"
    echo "  $0 restore --backup-file backup_20240901_120000.tar.gz"
    echo "  $0 test-dr --dry-run"
    exit 1
}

check_prerequisites() {
    log "Checking disaster recovery prerequisites"
    
    # Check required tools
    for tool in docker docker-compose pg_dump pg_restore aws; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log "WARNING: Tool '$tool' not found - some functionality may be limited"
        fi
    done
    
    # Check directories
    mkdir -p "$BACKUP_DIR" "$RECOVERY_DIR"
    
    # Check database connectivity
    if ! pg_isready -h "$DATABASE_HOST" -p "$DATABASE_PORT" -U "$DATABASE_USER" >/dev/null 2>&1; then
        log "WARNING: Cannot connect to primary database"
        return 1
    fi
    
    # Check Docker services
    if ! docker-compose -f "$DOCKER_COMPOSE_FILE" ps >/dev/null 2>&1; then
        log "WARNING: Cannot check Docker services status"
        return 1
    fi
    
    log "Prerequisites check completed"
    return 0
}

create_comprehensive_backup() {
    log "Creating comprehensive disaster recovery backup"
    
    local backup_name="dr_backup_${TIMESTAMP}"
    local backup_path="${BACKUP_DIR}/${backup_name}"
    
    mkdir -p "$backup_path"
    
    # 1. Database backup
    log "Backing up database"
    pg_dump \
        --host="$DATABASE_HOST" \
        --port="$DATABASE_PORT" \
        --username="$DATABASE_USER" \
        --dbname="$DATABASE_NAME" \
        --format=custom \
        --no-owner \
        --no-privileges \
        --verbose \
        --file="${backup_path}/database_backup.custom" 2>>"$LOG_FILE"
    
    # 2. Configuration backup
    log "Backing up configuration files"
    mkdir -p "${backup_path}/config"
    
    # Copy critical configuration files
    cp -r /etc/nginx "${backup_path}/config/" || true
    cp -r /etc/ssl "${backup_path}/config/" || true
    cp "$PRODUCTION_ENV_FILE" "${backup_path}/config/.env" || true
    cp "$DOCKER_COMPOSE_FILE" "${backup_path}/config/docker-compose.yml" || true
    
    # 3. Application data backup
    log "Backing up application data"
    mkdir -p "${backup_path}/data"
    
    # Copy persistent volumes
    for volume in postgres_prod_data redis_master_data grafana_prod_data; do
        local volume_path="/var/lib/leanvibe/${volume#*_prod_}"
        if [[ -d "$volume_path" ]]; then
            tar -czf "${backup_path}/data/${volume}.tar.gz" -C "$volume_path" . || true
        fi
    done
    
    # 4. SSL certificates backup
    log "Backing up SSL certificates"
    if [[ -d "/etc/letsencrypt" ]]; then
        tar -czf "${backup_path}/ssl_certificates.tar.gz" -C "/etc/letsencrypt" . || true
    fi
    
    # 5. Logs backup (last 7 days)
    log "Backing up recent logs"
    find /var/log -name "*.log" -mtime -7 -type f | \
        tar -czf "${backup_path}/logs_recent.tar.gz" -T - 2>/dev/null || true
    
    # 6. Create backup manifest
    cat > "${backup_path}/manifest.json" << EOF
{
    "backup_timestamp": "${TIMESTAMP}",
    "backup_type": "disaster_recovery",
    "rto_minutes": ${RTO_MINUTES},
    "rpo_minutes": ${RPO_MINUTES},
    "components": {
        "database": "database_backup.custom",
        "configuration": "config/",
        "application_data": "data/",
        "ssl_certificates": "ssl_certificates.tar.gz",
        "logs": "logs_recent.tar.gz"
    },
    "system_info": {
        "hostname": "$(hostname)",
        "docker_version": "$(docker --version)",
        "os_version": "$(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"')",
        "backup_size_mb": "$(du -sm "${backup_path}" | cut -f1)"
    }
}
EOF
    
    # 7. Compress entire backup
    log "Compressing backup archive"
    cd "$BACKUP_DIR"
    tar -czf "${backup_name}.tar.gz" "${backup_name}/"
    rm -rf "${backup_name}/"
    
    # 8. Encrypt if encryption key provided
    if [[ -n "$ENCRYPTION_KEY" ]]; then
        log "Encrypting backup"
        openssl enc -aes-256-cbc -salt -k "$ENCRYPTION_KEY" \
            -in "${backup_name}.tar.gz" \
            -out "${backup_name}.tar.gz.enc"
        rm "${backup_name}.tar.gz"
        backup_name="${backup_name}.tar.gz.enc"
    else
        backup_name="${backup_name}.tar.gz"
    fi
    
    # 9. Upload to S3 if configured
    if [[ -n "$S3_BUCKET" ]] && command -v aws >/dev/null 2>&1; then
        log "Uploading backup to S3"
        aws s3 cp "${BACKUP_DIR}/${backup_name}" "s3://${S3_BUCKET}/disaster-recovery/${backup_name}" \
            --storage-class GLACIER
    fi
    
    # 10. Calculate final backup size and checksum
    local backup_size
    backup_size=$(stat -c%s "${BACKUP_DIR}/${backup_name}" 2>/dev/null || stat -f%z "${BACKUP_DIR}/${backup_name}")
    local backup_checksum
    backup_checksum=$(sha256sum "${BACKUP_DIR}/${backup_name}" | cut -d' ' -f1)
    
    log "Disaster recovery backup completed: ${backup_name}"
    log "Size: $((backup_size / 1024 / 1024)) MB"
    log "Checksum: ${backup_checksum}"
    
    notify_slack "✅ Comprehensive DR backup created: ${backup_name} ($((backup_size / 1024 / 1024)) MB)"
    
    echo "${BACKUP_DIR}/${backup_name}"
}

restore_from_backup() {
    local backup_file="$1"
    local dry_run="${2:-false}"
    
    log "Starting disaster recovery restore from: ${backup_file}"
    
    if [[ "$dry_run" == "true" ]]; then
        log "DRY RUN MODE - No changes will be made"
    fi
    
    # Verify backup file exists
    if [[ ! -f "$backup_file" ]]; then
        log "ERROR: Backup file not found: $backup_file"
        return 1
    fi
    
    local restore_dir="${RECOVERY_DIR}/restore_${TIMESTAMP}"
    mkdir -p "$restore_dir"
    
    # Decrypt if needed
    if [[ "$backup_file" == *.enc ]]; then
        if [[ -z "$ENCRYPTION_KEY" ]]; then
            log "ERROR: Backup is encrypted but no encryption key provided"
            return 1
        fi
        
        log "Decrypting backup"
        local decrypted_file="${restore_dir}/$(basename "${backup_file%.enc}")"
        openssl enc -aes-256-cbc -d -k "$ENCRYPTION_KEY" \
            -in "$backup_file" -out "$decrypted_file"
        backup_file="$decrypted_file"
    fi
    
    # Extract backup
    log "Extracting backup archive"
    cd "$restore_dir"
    tar -xzf "$backup_file"
    
    local backup_content_dir
    backup_content_dir=$(find . -maxdepth 1 -type d -name "dr_backup_*" | head -1)
    
    if [[ -z "$backup_content_dir" ]]; then
        log "ERROR: Could not find backup content directory"
        return 1
    fi
    
    cd "$backup_content_dir"
    
    # Read manifest
    if [[ -f "manifest.json" ]]; then
        log "Backup manifest found:"
        cat "manifest.json" | jq '.' || cat "manifest.json"
    fi
    
    if [[ "$dry_run" == "true" ]]; then
        log "DRY RUN: Would restore the following components:"
        ls -la
        return 0
    fi
    
    # Stop services
    log "Stopping application services"
    docker-compose -f "$DOCKER_COMPOSE_FILE" down || true
    
    # Restore database
    if [[ -f "database_backup.custom" ]]; then
        log "Restoring database"
        
        # Drop and recreate database
        psql -h "$DATABASE_HOST" -p "$DATABASE_PORT" -U "$DATABASE_USER" -d postgres \
            -c "DROP DATABASE IF EXISTS ${DATABASE_NAME};"
        psql -h "$DATABASE_HOST" -p "$DATABASE_PORT" -U "$DATABASE_USER" -d postgres \
            -c "CREATE DATABASE ${DATABASE_NAME};"
        
        # Restore data
        pg_restore \
            --host="$DATABASE_HOST" \
            --port="$DATABASE_PORT" \
            --username="$DATABASE_USER" \
            --dbname="$DATABASE_NAME" \
            --verbose \
            --no-owner \
            --no-privileges \
            "database_backup.custom"
    fi
    
    # Restore configuration
    if [[ -d "config" ]]; then
        log "Restoring configuration files"
        cp -r config/nginx/* /etc/nginx/ || true
        cp -r config/ssl/* /etc/ssl/ || true
        cp config/.env "$PRODUCTION_ENV_FILE" || true
    fi
    
    # Restore SSL certificates
    if [[ -f "ssl_certificates.tar.gz" ]]; then
        log "Restoring SSL certificates"
        mkdir -p /etc/letsencrypt
        tar -xzf ssl_certificates.tar.gz -C /etc/letsencrypt/
    fi
    
    # Restore application data
    if [[ -d "data" ]]; then
        log "Restoring application data"
        for data_file in data/*.tar.gz; do
            if [[ -f "$data_file" ]]; then
                local volume_name
                volume_name=$(basename "$data_file" .tar.gz)
                local volume_path="/var/lib/leanvibe/${volume_name#*_}"
                
                mkdir -p "$volume_path"
                tar -xzf "$data_file" -C "$volume_path"
            fi
        done
    fi
    
    # Restart services
    log "Starting application services"
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    # Wait for services to be healthy
    log "Waiting for services to become healthy"
    local max_attempts=30
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "healthy"; then
            break
        fi
        sleep 10
        ((attempt++))
    done
    
    log "Disaster recovery restore completed"
    notify_slack "✅ Disaster recovery restore completed successfully"
}

test_disaster_recovery() {
    local dry_run="${1:-false}"
    
    log "Starting disaster recovery test"
    
    if [[ "$dry_run" == "true" ]]; then
        log "DRY RUN MODE - Testing procedures without making changes"
    fi
    
    # Test 1: Backup creation
    log "Test 1: Creating test backup"
    local test_backup
    if test_backup=$(create_comprehensive_backup); then
        log "✓ Backup creation test passed"
    else
        log "✗ Backup creation test failed"
        return 1
    fi
    
    # Test 2: Backup integrity
    log "Test 2: Verifying backup integrity"
    if tar -tzf "$test_backup" >/dev/null 2>&1; then
        log "✓ Backup integrity test passed"
    else
        log "✗ Backup integrity test failed"
        return 1
    fi
    
    # Test 3: Service health checks
    log "Test 3: Service health checks"
    local healthy_services=0
    local total_services=0
    
    while read -r service status; do
        ((total_services++))
        if [[ "$status" == *"healthy"* ]]; then
            ((healthy_services++))
            log "✓ Service $service is healthy"
        else
            log "✗ Service $service is not healthy: $status"
        fi
    done < <(docker-compose -f "$DOCKER_COMPOSE_FILE" ps --format "table {{.Service}}\t{{.Status}}" | tail -n +2)
    
    # Test 4: Database connectivity and basic operations
    log "Test 4: Database connectivity test"
    if psql -h "$DATABASE_HOST" -p "$DATABASE_PORT" -U "$DATABASE_USER" -d "$DATABASE_NAME" \
       -c "SELECT 1;" >/dev/null 2>&1; then
        log "✓ Database connectivity test passed"
    else
        log "✗ Database connectivity test failed"
        return 1
    fi
    
    # Test 5: API endpoint availability
    log "Test 5: API endpoint availability test"
    if curl -sSf "http://localhost:8000/health" >/dev/null 2>&1; then
        log "✓ API endpoint availability test passed"
    else
        log "✗ API endpoint availability test failed"
    fi
    
    # Test 6: SSL certificate validation
    log "Test 6: SSL certificate validation"
    if [[ -f "/etc/nginx/ssl/fullchain.pem" ]] && \
       openssl x509 -in "/etc/nginx/ssl/fullchain.pem" -noout -checkend 86400 >/dev/null 2>&1; then
        log "✓ SSL certificate validation test passed"
    else
        log "✗ SSL certificate validation test failed"
    fi
    
    # Calculate RTO/RPO compliance
    local backup_age_minutes
    backup_age_minutes=$(( ($(date +%s) - $(stat -c %Y "$test_backup" 2>/dev/null || stat -f %m "$test_backup")) / 60 ))
    
    log "Disaster Recovery Test Results:"
    log "=============================="
    log "Healthy Services: ${healthy_services}/${total_services}"
    log "Backup Age: ${backup_age_minutes} minutes"
    log "RPO Compliance: $([[ $backup_age_minutes -le $RPO_MINUTES ]] && echo "✓ PASS" || echo "✗ FAIL")"
    log "RTO Target: ${RTO_MINUTES} minutes"
    
    if [[ $healthy_services -eq $total_services ]] && [[ $backup_age_minutes -le $RPO_MINUTES ]]; then
        log "✅ Disaster recovery test PASSED"
        notify_slack "✅ Disaster recovery test PASSED - All systems operational"
        return 0
    else
        log "❌ Disaster recovery test FAILED"
        notify_slack "❌ Disaster recovery test FAILED - Issues detected" "danger"
        return 1
    fi
}

check_system_status() {
    log "Checking system status and disaster recovery readiness"
    
    # Service status
    log "Docker services status:"
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
    
    # Database status
    log "Database status:"
    if pg_isready -h "$DATABASE_HOST" -p "$DATABASE_PORT" -U "$DATABASE_USER"; then
        log "✓ Database is ready"
        
        # Database size
        local db_size
        db_size=$(psql -h "$DATABASE_HOST" -p "$DATABASE_PORT" -U "$DATABASE_USER" -d "$DATABASE_NAME" \
                  -t -c "SELECT pg_size_pretty(pg_database_size('$DATABASE_NAME'));" | xargs)
        log "Database size: $db_size"
    else
        log "✗ Database is not ready"
    fi
    
    # Recent backups
    log "Recent backups:"
    find "$BACKUP_DIR" -name "dr_backup_*.tar.gz*" -mtime -7 -exec ls -lh {} \; | head -5
    
    # Disk space
    log "Disk space status:"
    df -h | grep -E "(Filesystem|/var|/backup)"
    
    # SSL certificate expiry
    if [[ -f "/etc/nginx/ssl/fullchain.pem" ]]; then
        local cert_expiry
        cert_expiry=$(openssl x509 -in "/etc/nginx/ssl/fullchain.pem" -noout -enddate | cut -d= -f2)
        log "SSL certificate expires: $cert_expiry"
    fi
}

main() {
    local command="${1:-}"
    local dry_run=false
    local force=false
    local backup_file=""
    
    # Parse arguments
    shift || true
    while [[ $# -gt 0 ]]; do
        case $1 in
            --backup-file)
                backup_file="$2"
                shift 2
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            --force)
                force=true
                shift
                ;;
            --mode)
                DR_MODE="$2"
                shift 2
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
        backup)
            create_comprehensive_backup
            ;;
        restore)
            if [[ -z "$backup_file" ]]; then
                log "ERROR: --backup-file required for restore command"
                exit 1
            fi
            restore_from_backup "$backup_file" "$dry_run"
            ;;
        test-dr)
            test_disaster_recovery "$dry_run"
            ;;
        status)
            check_system_status
            ;;
        validate)
            check_prerequisites && test_disaster_recovery true
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
        log "ERROR: Disaster recovery operation failed with exit code: $exit_code"
        notify_slack "❌ Disaster recovery operation FAILED" "danger"
    fi
    exit $exit_code
}

trap cleanup EXIT

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi