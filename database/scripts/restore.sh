#!/bin/bash

# LeanVibe Agent Hive - Production Database Restore Script
# Disaster recovery with validation and rollback capabilities

set -euo pipefail

# Configuration
BACKUP_DIR="/backups"
DATABASE_NAME="${POSTGRES_DB:-leanvibe_agent_hive}"
DATABASE_USER="${POSTGRES_USER:-leanvibe_user}"
DATABASE_HOST="${POSTGRES_HOST:-localhost}"
DATABASE_PORT="${POSTGRES_PORT:-5432}"
ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY:-}"
SLACK_WEBHOOK="${BACKUP_SLACK_WEBHOOK:-}"

# Script options
BACKUP_FILE=""
FORCE_RESTORE=false
SKIP_VALIDATION=false
CREATE_PRE_RESTORE_BACKUP=true

# Logging
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${BACKUP_DIR}/logs/restore_${TIMESTAMP}.log"
mkdir -p "${BACKUP_DIR}/logs"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "${LOG_FILE}"
}

notify_slack() {
    if [[ -n "${SLACK_WEBHOOK}" ]]; then
        local message="$1"
        local color="${2:-good}"
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"attachments\":[{\"color\":\"${color}\",\"text\":\"${message}\"}]}" \
            "${SLACK_WEBHOOK}" || true
    fi
}

usage() {
    echo "Usage: $0 -f <backup_file> [options]"
    echo ""
    echo "Options:"
    echo "  -f, --file <backup_file>     Path to backup file to restore"
    echo "  --force                      Force restore without confirmation"
    echo "  --skip-validation           Skip post-restore validation"
    echo "  --no-pre-backup            Skip pre-restore backup creation"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -f leanvibe_backup_20240901_120000.sql.gz"
    echo "  $0 -f leanvibe_backup_20240901_120000.sql.gz.enc --force"
    exit 1
}

cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR: Restore process failed with exit code: $exit_code"
        notify_slack "❌ LeanVibe database restore FAILED on $(hostname)" "danger"
    fi
    # Clean up temporary files
    [[ -f "/tmp/restore_${TIMESTAMP}.sql" ]] && rm -f "/tmp/restore_${TIMESTAMP}.sql"
    [[ -f "/tmp/decrypt_${TIMESTAMP}.sql.gz" ]] && rm -f "/tmp/decrypt_${TIMESTAMP}.sql.gz"
    exit $exit_code
}

trap cleanup EXIT

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--file)
            BACKUP_FILE="$2"
            shift 2
            ;;
        --force)
            FORCE_RESTORE=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --no-pre-backup)
            CREATE_PRE_RESTORE_BACKUP=false
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate required parameters
if [[ -z "${BACKUP_FILE}" ]]; then
    echo "ERROR: Backup file not specified"
    usage
fi

# Check if backup file exists
if [[ ! -f "${BACKUP_FILE}" ]]; then
    if [[ -f "${BACKUP_DIR}/${BACKUP_FILE}" ]]; then
        BACKUP_FILE="${BACKUP_DIR}/${BACKUP_FILE}"
    else
        log "ERROR: Backup file not found: ${BACKUP_FILE}"
        exit 1
    fi
fi

log "Starting database restore process"
log "Backup file: ${BACKUP_FILE}"

# Pre-flight checks
for tool in psql pg_dump; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        log "ERROR: Required tool '$tool' is not installed"
        exit 1
    fi
done

# Check database connectivity
log "Testing database connectivity"
if ! pg_isready -h "${DATABASE_HOST}" -p "${DATABASE_PORT}" -U "${DATABASE_USER}" -d "${DATABASE_NAME}"; then
    log "ERROR: Cannot connect to database"
    exit 1
fi

# Get current database info
current_tables=$(psql -h "${DATABASE_HOST}" -p "${DATABASE_PORT}" -U "${DATABASE_USER}" -d "${DATABASE_NAME}" -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" | xargs)
current_size=$(psql -h "${DATABASE_HOST}" -p "${DATABASE_PORT}" -U "${DATABASE_USER}" -d "${DATABASE_NAME}" -t -c "SELECT pg_size_pretty(pg_database_size('${DATABASE_NAME}'));" | xargs)

log "Current database stats:"
log "  Tables: ${current_tables}"
log "  Size: ${current_size}"

# Confirm restore operation
if [[ "${FORCE_RESTORE}" == "false" ]]; then
    echo ""
    echo "⚠️  WARNING: This will COMPLETELY REPLACE the current database!"
    echo "Database: ${DATABASE_NAME}"
    echo "Current tables: ${current_tables}"
    echo "Current size: ${current_size}"
    echo "Backup file: ${BACKUP_FILE}"
    echo ""
    read -p "Are you sure you want to continue? (type 'YES' to confirm): " confirm
    if [[ "${confirm}" != "YES" ]]; then
        log "Restore cancelled by user"
        exit 0
    fi
fi

# Create pre-restore backup if requested
if [[ "${CREATE_PRE_RESTORE_BACKUP}" == "true" ]]; then
    log "Creating pre-restore backup"
    pre_restore_backup="pre_restore_backup_${TIMESTAMP}.sql.gz"
    
    pg_dump \
        --host="${DATABASE_HOST}" \
        --port="${DATABASE_PORT}" \
        --username="${DATABASE_USER}" \
        --dbname="${DATABASE_NAME}" \
        --verbose \
        --no-owner \
        --no-privileges \
        2>>"${LOG_FILE}" | gzip > "${BACKUP_DIR}/${pre_restore_backup}"
    
    log "Pre-restore backup saved: ${pre_restore_backup}"
    notify_slack "📦 Pre-restore backup created: ${pre_restore_backup}"
fi

# Prepare backup file for restore
restore_file="/tmp/restore_${TIMESTAMP}.sql"

if [[ "${BACKUP_FILE}" == *".enc" ]]; then
    # Decrypt backup
    if [[ -z "${ENCRYPTION_KEY}" ]]; then
        log "ERROR: Backup is encrypted but no encryption key provided"
        exit 1
    fi
    
    log "Decrypting backup file"
    decrypted_file="/tmp/decrypt_${TIMESTAMP}.sql.gz"
    
    if ! openssl enc -aes-256-cbc -d -k "${ENCRYPTION_KEY}" \
        -in "${BACKUP_FILE}" \
        -out "${decrypted_file}"; then
        log "ERROR: Failed to decrypt backup file"
        exit 1
    fi
    
    BACKUP_FILE="${decrypted_file}"
fi

if [[ "${BACKUP_FILE}" == *".gz" ]]; then
    # Decompress backup
    log "Decompressing backup file"
    if ! gzip -dc "${BACKUP_FILE}" > "${restore_file}"; then
        log "ERROR: Failed to decompress backup file"
        exit 1
    fi
elif [[ "${BACKUP_FILE}" == *".custom" ]]; then
    # Handle custom format backup
    restore_file="${BACKUP_FILE}"
else
    # Plain SQL backup
    cp "${BACKUP_FILE}" "${restore_file}"
fi

# Verify backup integrity
log "Verifying backup file integrity"
if [[ "${restore_file}" == *".custom" ]]; then
    # Verify custom format
    if ! pg_restore --list "${restore_file}" >/dev/null 2>&1; then
        log "ERROR: Backup file appears to be corrupted (custom format)"
        exit 1
    fi
else
    # Basic SQL file check
    if ! head -n 10 "${restore_file}" | grep -q "PostgreSQL database dump"; then
        log "WARNING: Backup file may not be a valid PostgreSQL dump"
    fi
fi

# Get backup metadata if available
backup_basename=$(basename "${BACKUP_FILE}")
metadata_file="${BACKUP_DIR}/backup_${backup_basename%.*.*}_metadata.json"
if [[ -f "${metadata_file}" ]]; then
    log "Found backup metadata: ${metadata_file}"
    backup_info=$(cat "${metadata_file}")
    log "Backup metadata: ${backup_info}"
fi

# Stop application services (optional - requires Docker Compose)
if command -v docker-compose >/dev/null 2>&1 && [[ -f "docker-compose.yml" ]]; then
    log "Stopping application services"
    docker-compose stop api frontend || true
fi

# Perform the restore
log "Starting database restore"
start_time=$(date +%s)

# Drop and recreate database
log "Dropping and recreating database"
psql -h "${DATABASE_HOST}" -p "${DATABASE_PORT}" -U "${DATABASE_USER}" -d postgres -c "DROP DATABASE IF EXISTS ${DATABASE_NAME};"
psql -h "${DATABASE_HOST}" -p "${DATABASE_PORT}" -U "${DATABASE_USER}" -d postgres -c "CREATE DATABASE ${DATABASE_NAME};"

# Restore data
if [[ "${restore_file}" == *".custom" ]]; then
    # Custom format restore
    log "Restoring from custom format backup"
    pg_restore \
        --host="${DATABASE_HOST}" \
        --port="${DATABASE_PORT}" \
        --username="${DATABASE_USER}" \
        --dbname="${DATABASE_NAME}" \
        --verbose \
        --no-owner \
        --no-privileges \
        --exit-on-error \
        "${restore_file}" \
        2>>"${LOG_FILE}"
else
    # SQL format restore
    log "Restoring from SQL backup"
    psql \
        --host="${DATABASE_HOST}" \
        --port="${DATABASE_PORT}" \
        --username="${DATABASE_USER}" \
        --dbname="${DATABASE_NAME}" \
        --file="${restore_file}" \
        --single-transaction \
        --set ON_ERROR_STOP=1 \
        2>>"${LOG_FILE}"
fi

end_time=$(date +%s)
restore_duration=$((end_time - start_time))
log "Database restore completed in ${restore_duration} seconds"

# Post-restore validation
if [[ "${SKIP_VALIDATION}" == "false" ]]; then
    log "Performing post-restore validation"
    
    # Check table count
    restored_tables=$(psql -h "${DATABASE_HOST}" -p "${DATABASE_PORT}" -U "${DATABASE_USER}" -d "${DATABASE_NAME}" -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" | xargs)
    restored_size=$(psql -h "${DATABASE_HOST}" -p "${DATABASE_PORT}" -U "${DATABASE_USER}" -d "${DATABASE_NAME}" -t -c "SELECT pg_size_pretty(pg_database_size('${DATABASE_NAME}'));" | xargs)
    
    log "Restored database stats:"
    log "  Tables: ${restored_tables}"
    log "  Size: ${restored_size}"
    
    # Basic connectivity test
    if ! psql -h "${DATABASE_HOST}" -p "${DATABASE_PORT}" -U "${DATABASE_USER}" -d "${DATABASE_NAME}" -c "SELECT 1;" >/dev/null 2>&1; then
        log "ERROR: Post-restore connectivity test failed"
        exit 1
    fi
    
    # Check for critical tables (customize based on your schema)
    critical_tables=("users" "agents" "tasks" "workflows")
    for table in "${critical_tables[@]}"; do
        if psql -h "${DATABASE_HOST}" -p "${DATABASE_PORT}" -U "${DATABASE_USER}" -d "${DATABASE_NAME}" -t -c "SELECT to_regclass('${table}');" | grep -q "null"; then
            log "WARNING: Critical table '${table}' not found in restored database"
        else
            log "✓ Critical table '${table}' found"
        fi
    done
    
    log "Post-restore validation completed"
fi

# Update database statistics
log "Updating database statistics"
psql -h "${DATABASE_HOST}" -p "${DATABASE_PORT}" -U "${DATABASE_USER}" -d "${DATABASE_NAME}" -c "ANALYZE;"

# Restart application services
if command -v docker-compose >/dev/null 2>&1 && [[ -f "docker-compose.yml" ]]; then
    log "Restarting application services"
    docker-compose start api frontend || true
fi

# Create restore log entry
cat > "${BACKUP_DIR}/restore_${TIMESTAMP}_log.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "backup_file": "${BACKUP_FILE}",
    "original_tables": ${current_tables},
    "restored_tables": ${restored_tables:-0},
    "restore_duration_seconds": ${restore_duration},
    "pre_restore_backup": $([ "${CREATE_PRE_RESTORE_BACKUP}" == "true" ] && echo "\"${pre_restore_backup}\"" || echo "null"),
    "validation_skipped": $([ "${SKIP_VALIDATION}" == "true" ] && echo "true" || echo "false"),
    "success": true
}
EOF

log "Database restore completed successfully"
notify_slack "✅ LeanVibe database restore completed successfully on $(hostname)
• Backup file: $(basename "${BACKUP_FILE}")
• Tables restored: ${restored_tables:-0}
• Duration: ${restore_duration} seconds
• Size: ${restored_size:-unknown}"

log "Restore process completed successfully"