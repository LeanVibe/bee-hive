#!/bin/bash

# LeanVibe Agent Hive - Production Database Backup Script
# Automated backup with encryption, compression, and retention management

set -euo pipefail

# Configuration
BACKUP_DIR="/backups"
ARCHIVE_DIR="/backups/archive"
DATABASE_NAME="${POSTGRES_DB:-leanvibe_agent_hive}"
DATABASE_USER="${POSTGRES_USER:-leanvibe_user}"
DATABASE_HOST="${POSTGRES_HOST:-localhost}"
DATABASE_PORT="${POSTGRES_PORT:-5432}"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY:-}"
COMPRESSION_LEVEL="${BACKUP_COMPRESSION_LEVEL:-6}"
S3_BUCKET="${BACKUP_S3_BUCKET:-}"
SLACK_WEBHOOK="${BACKUP_SLACK_WEBHOOK:-}"

# Timestamps
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
DATE=$(date '+%Y-%m-%d')

# Backup filenames
BACKUP_FILE="leanvibe_backup_${TIMESTAMP}.sql"
COMPRESSED_BACKUP="leanvibe_backup_${TIMESTAMP}.sql.gz"
ENCRYPTED_BACKUP="leanvibe_backup_${TIMESTAMP}.sql.gz.enc"

# Logging
LOG_FILE="${BACKUP_DIR}/logs/backup_${DATE}.log"
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

cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log "ERROR: Backup failed with exit code: $exit_code"
        notify_slack "❌ LeanVibe database backup FAILED on $(hostname)" "danger"
    fi
    # Clean up temporary files
    [[ -f "${BACKUP_DIR}/${BACKUP_FILE}" ]] && rm -f "${BACKUP_DIR}/${BACKUP_FILE}"
    exit $exit_code
}

trap cleanup EXIT

# Pre-flight checks
log "Starting database backup process"

# Check required tools
for tool in pg_dump gzip; do
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

# Create backup directories
mkdir -p "${BACKUP_DIR}" "${ARCHIVE_DIR}"

# Perform the backup
log "Starting PostgreSQL backup"
start_time=$(date +%s)

pg_dump \
    --host="${DATABASE_HOST}" \
    --port="${DATABASE_PORT}" \
    --username="${DATABASE_USER}" \
    --dbname="${DATABASE_NAME}" \
    --verbose \
    --format=custom \
    --no-owner \
    --no-privileges \
    --compress=0 \
    --file="${BACKUP_DIR}/${BACKUP_FILE}.custom" \
    2>>"${LOG_FILE}"

# Also create a plain SQL backup for easier restoration
pg_dump \
    --host="${DATABASE_HOST}" \
    --port="${DATABASE_PORT}" \
    --username="${DATABASE_USER}" \
    --dbname="${DATABASE_NAME}" \
    --verbose \
    --no-owner \
    --no-privileges \
    --file="${BACKUP_DIR}/${BACKUP_FILE}" \
    2>>"${LOG_FILE}"

end_time=$(date +%s)
backup_duration=$((end_time - start_time))
log "Database backup completed in ${backup_duration} seconds"

# Get backup size
backup_size=$(stat -f%z "${BACKUP_DIR}/${BACKUP_FILE}" 2>/dev/null || stat -c%s "${BACKUP_DIR}/${BACKUP_FILE}")
backup_size_mb=$((backup_size / 1024 / 1024))
log "Backup size: ${backup_size_mb} MB"

# Compress the backup
log "Compressing backup"
gzip -${COMPRESSION_LEVEL} "${BACKUP_DIR}/${BACKUP_FILE}"
compressed_size=$(stat -f%z "${BACKUP_DIR}/${COMPRESSED_BACKUP}" 2>/dev/null || stat -c%s "${BACKUP_DIR}/${COMPRESSED_BACKUP}")
compressed_size_mb=$((compressed_size / 1024 / 1024))
compression_ratio=$(( (backup_size - compressed_size) * 100 / backup_size ))
log "Compressed backup size: ${compressed_size_mb} MB (${compression_ratio}% compression)"

# Encrypt the backup if encryption key is provided
if [[ -n "${ENCRYPTION_KEY}" ]]; then
    log "Encrypting backup"
    if command -v openssl >/dev/null 2>&1; then
        openssl enc -aes-256-cbc -salt -k "${ENCRYPTION_KEY}" \
            -in "${BACKUP_DIR}/${COMPRESSED_BACKUP}" \
            -out "${BACKUP_DIR}/${ENCRYPTED_BACKUP}"
        rm "${BACKUP_DIR}/${COMPRESSED_BACKUP}"
        FINAL_BACKUP="${ENCRYPTED_BACKUP}"
    else
        log "WARNING: OpenSSL not available, backup not encrypted"
        FINAL_BACKUP="${COMPRESSED_BACKUP}"
    fi
else
    FINAL_BACKUP="${COMPRESSED_BACKUP}"
fi

# Copy to archive
cp "${BACKUP_DIR}/${FINAL_BACKUP}" "${ARCHIVE_DIR}/"
log "Backup copied to archive directory"

# Upload to S3 if configured
if [[ -n "${S3_BUCKET}" ]] && command -v aws >/dev/null 2>&1; then
    log "Uploading backup to S3"
    if aws s3 cp "${BACKUP_DIR}/${FINAL_BACKUP}" "s3://${S3_BUCKET}/database-backups/${FINAL_BACKUP}"; then
        log "Backup successfully uploaded to S3"
    else
        log "WARNING: Failed to upload backup to S3"
    fi
fi

# Create backup metadata
cat > "${BACKUP_DIR}/backup_${TIMESTAMP}_metadata.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "date": "${DATE}",
    "database": "${DATABASE_NAME}",
    "backup_file": "${FINAL_BACKUP}",
    "custom_backup_file": "$(basename "${BACKUP_DIR}/${BACKUP_FILE}.custom")",
    "original_size_mb": ${backup_size_mb},
    "compressed_size_mb": ${compressed_size_mb},
    "compression_ratio": ${compression_ratio},
    "backup_duration_seconds": ${backup_duration},
    "encrypted": $([ -n "${ENCRYPTION_KEY}" ] && echo "true" || echo "false"),
    "s3_uploaded": $([ -n "${S3_BUCKET}" ] && echo "true" || echo "false"),
    "checksum": "$(sha256sum "${BACKUP_DIR}/${FINAL_BACKUP}" | cut -d' ' -f1)"
}
EOF

# Clean up old backups
log "Cleaning up old backups (retention: ${RETENTION_DAYS} days)"
find "${BACKUP_DIR}" -name "leanvibe_backup_*.sql*" -mtime +${RETENTION_DAYS} -delete
find "${BACKUP_DIR}" -name "backup_*_metadata.json" -mtime +${RETENTION_DAYS} -delete
find "${BACKUP_DIR}/logs" -name "backup_*.log" -mtime +30 -delete
find "${ARCHIVE_DIR}" -name "leanvibe_backup_*.sql*" -mtime +$((RETENTION_DAYS * 2)) -delete

# Verify backup integrity
log "Verifying backup integrity"
if [[ "${FINAL_BACKUP}" == *".enc" ]]; then
    # Decrypt and verify
    temp_file="/tmp/backup_verify_${TIMESTAMP}.sql.gz"
    openssl enc -aes-256-cbc -d -k "${ENCRYPTION_KEY}" \
        -in "${BACKUP_DIR}/${FINAL_BACKUP}" \
        -out "${temp_file}"
    if gzip -t "${temp_file}"; then
        log "Backup integrity verified (encrypted)"
    else
        log "ERROR: Backup integrity check failed (encrypted)"
        exit 1
    fi
    rm -f "${temp_file}"
elif [[ "${FINAL_BACKUP}" == *".gz" ]]; then
    if gzip -t "${BACKUP_DIR}/${FINAL_BACKUP}"; then
        log "Backup integrity verified"
    else
        log "ERROR: Backup integrity check failed"
        exit 1
    fi
fi

# Generate backup report
backup_count=$(find "${BACKUP_DIR}" -name "leanvibe_backup_*.sql*" | wc -l)
total_backup_size=$(find "${BACKUP_DIR}" -name "leanvibe_backup_*.sql*" -exec stat -f%z {} + 2>/dev/null | awk '{sum+=$1} END {print int(sum/1024/1024)}' || echo "0")

log "Backup completed successfully"
log "Current backup count: ${backup_count}"
log "Total backup storage: ${total_backup_size} MB"

# Send success notification
notify_slack "✅ LeanVibe database backup completed successfully on $(hostname)
• Size: ${compressed_size_mb} MB (${compression_ratio}% compression)
• Duration: ${backup_duration} seconds
• Retention: ${backup_count} backups (${total_backup_size} MB total)"

log "Backup process completed successfully"