#!/bin/bash

# LeanVibe Agent Hive - SSL Certificate Setup and Management
# Automated Let's Encrypt certificate provisioning with monitoring

set -euo pipefail

# Configuration
DOMAIN_NAME="${DOMAIN_NAME:-example.com}"
ADMIN_EMAIL="${ADMIN_EMAIL:-admin@example.com}"
WEBROOT_PATH="${WEBROOT_PATH:-/var/www/certbot}"
SSL_DIR="${SSL_DIR:-/etc/nginx/ssl}"
NGINX_CONFIG_DIR="${NGINX_CONFIG_DIR:-/etc/nginx}"
SLACK_WEBHOOK="${SSL_SLACK_WEBHOOK:-}"
CERT_EXPIRY_DAYS="${CERT_EXPIRY_DAYS:-30}"

# Additional domains
ADDITIONAL_DOMAINS="${ADDITIONAL_DOMAINS:-www.${DOMAIN_NAME} api.${DOMAIN_NAME}}"

# Logging
LOG_FILE="/var/log/leanvibe/ssl_management.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
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

check_prerequisites() {
    log "Checking prerequisites"
    
    # Check if certbot is installed
    if ! command -v certbot >/dev/null 2>&1; then
        log "Installing certbot"
        if command -v apt-get >/dev/null 2>&1; then
            apt-get update
            apt-get install -y certbot
        elif command -v yum >/dev/null 2>&1; then
            yum install -y certbot
        else
            log "ERROR: Cannot install certbot - unsupported package manager"
            exit 1
        fi
    fi
    
    # Create directories
    mkdir -p "$WEBROOT_PATH" "$SSL_DIR"
    
    # Check nginx configuration
    if ! nginx -t; then
        log "ERROR: Nginx configuration test failed"
        exit 1
    fi
}

create_temporary_nginx_config() {
    log "Creating temporary nginx configuration for certificate verification"
    
    cat > "$NGINX_CONFIG_DIR/sites-available/temp-ssl" << EOF
server {
    listen 80;
    server_name ${DOMAIN_NAME} ${ADDITIONAL_DOMAINS};
    
    location /.well-known/acme-challenge/ {
        root ${WEBROOT_PATH};
        try_files \$uri =404;
    }
    
    location / {
        return 301 https://\$server_name\$request_uri;
    }
}
EOF
    
    # Enable the temporary configuration
    ln -sf "$NGINX_CONFIG_DIR/sites-available/temp-ssl" "$NGINX_CONFIG_DIR/sites-enabled/temp-ssl"
    
    # Remove default if it exists
    rm -f "$NGINX_CONFIG_DIR/sites-enabled/default"
    
    # Test and reload nginx
    nginx -t && systemctl reload nginx
}

obtain_certificate() {
    log "Obtaining SSL certificate for ${DOMAIN_NAME}"
    
    # Build domain list
    local domain_args="-d ${DOMAIN_NAME}"
    for domain in ${ADDITIONAL_DOMAINS}; do
        domain_args="${domain_args} -d ${domain}"
    done
    
    # Request certificate
    certbot certonly \
        --webroot \
        --webroot-path="${WEBROOT_PATH}" \
        --email "${ADMIN_EMAIL}" \
        --agree-tos \
        --no-eff-email \
        --non-interactive \
        --expand \
        ${domain_args} || {
        log "ERROR: Certificate request failed"
        notify_slack "❌ SSL certificate request failed for ${DOMAIN_NAME}" "danger"
        exit 1
    }
    
    log "Certificate obtained successfully"
}

install_certificate() {
    log "Installing certificate to nginx"
    
    local cert_path="/etc/letsencrypt/live/${DOMAIN_NAME}"
    
    # Copy certificates to nginx ssl directory
    cp "${cert_path}/fullchain.pem" "${SSL_DIR}/fullchain.pem"
    cp "${cert_path}/privkey.pem" "${SSL_DIR}/privkey.pem"
    cp "${cert_path}/chain.pem" "${SSL_DIR}/chain.pem"
    
    # Set proper permissions
    chmod 644 "${SSL_DIR}/fullchain.pem" "${SSL_DIR}/chain.pem"
    chmod 600 "${SSL_DIR}/privkey.pem"
    chown root:root "${SSL_DIR}"/*
    
    log "Certificates installed successfully"
}

create_production_nginx_config() {
    log "Creating production nginx configuration"
    
    # Remove temporary configuration
    rm -f "$NGINX_CONFIG_DIR/sites-enabled/temp-ssl"
    
    # Copy production configuration
    cp "${NGINX_CONFIG_DIR}/nginx.production.conf" "${NGINX_CONFIG_DIR}/nginx.conf"
    
    # Test configuration
    if ! nginx -t; then
        log "ERROR: Production nginx configuration test failed"
        exit 1
    fi
    
    # Reload nginx with SSL configuration
    systemctl reload nginx
    log "Production nginx configuration activated"
}

setup_auto_renewal() {
    log "Setting up automatic certificate renewal"
    
    # Create renewal script
    cat > "/usr/local/bin/leanvibe-ssl-renew" << 'EOF'
#!/bin/bash

# LeanVibe SSL Certificate Renewal Script

set -euo pipefail

LOG_FILE="/var/log/leanvibe/ssl_renewal.log"
SLACK_WEBHOOK="${SSL_SLACK_WEBHOOK:-}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
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

# Attempt renewal
log "Starting certificate renewal check"

if certbot renew --webroot --webroot-path=/var/www/certbot --quiet --no-self-upgrade --post-hook "systemctl reload nginx"; then
    log "Certificate renewal completed successfully"
    
    # Check if certificates were actually renewed
    if [[ -n "$(find /etc/letsencrypt/live -name "cert.pem" -mtime -1)" ]]; then
        log "Certificates were renewed"
        
        # Copy new certificates to nginx
        for cert_dir in /etc/letsencrypt/live/*/; do
            domain=$(basename "$cert_dir")
            if [[ -f "${cert_dir}fullchain.pem" ]]; then
                cp "${cert_dir}fullchain.pem" "/etc/nginx/ssl/fullchain.pem"
                cp "${cert_dir}privkey.pem" "/etc/nginx/ssl/privkey.pem"
                cp "${cert_dir}chain.pem" "/etc/nginx/ssl/chain.pem"
                
                chmod 644 "/etc/nginx/ssl/fullchain.pem" "/etc/nginx/ssl/chain.pem"
                chmod 600 "/etc/nginx/ssl/privkey.pem"
            fi
        done
        
        notify_slack "✅ SSL certificates renewed successfully for LeanVibe production"
    else
        log "No certificates were renewed (not due for renewal)"
    fi
else
    log "ERROR: Certificate renewal failed"
    notify_slack "❌ SSL certificate renewal FAILED for LeanVibe production" "danger"
    exit 1
fi
EOF

    chmod +x "/usr/local/bin/leanvibe-ssl-renew"
    
    # Create systemd timer for renewal
    cat > "/etc/systemd/system/leanvibe-ssl-renew.service" << EOF
[Unit]
Description=LeanVibe SSL Certificate Renewal
Wants=leanvibe-ssl-renew.timer

[Service]
Type=oneshot
ExecStart=/usr/local/bin/leanvibe-ssl-renew
User=root
Group=root

[Install]
WantedBy=multi-user.target
EOF

    cat > "/etc/systemd/system/leanvibe-ssl-renew.timer" << EOF
[Unit]
Description=Run LeanVibe SSL Certificate Renewal twice daily
Requires=leanvibe-ssl-renew.service

[Timer]
OnCalendar=*-*-* 06,18:00:00
RandomizedDelaySec=3600
Persistent=true

[Install]
WantedBy=timers.target
EOF

    # Enable and start the timer
    systemctl daemon-reload
    systemctl enable leanvibe-ssl-renew.timer
    systemctl start leanvibe-ssl-renew.timer
    
    log "Auto-renewal configured with systemd timer"
}

create_monitoring_script() {
    log "Creating SSL certificate monitoring script"
    
    cat > "/usr/local/bin/leanvibe-ssl-monitor" << 'EOF'
#!/bin/bash

# LeanVibe SSL Certificate Monitoring Script

set -euo pipefail

DOMAIN_NAME="${DOMAIN_NAME:-example.com}"
CERT_EXPIRY_DAYS="${CERT_EXPIRY_DAYS:-30}"
SLACK_WEBHOOK="${SSL_SLACK_WEBHOOK:-}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

notify_slack() {
    if [[ -n "${SLACK_WEBHOOK}" ]]; then
        local message="$1"
        local color="${2:-warning}"
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"attachments\":[{\"color\":\"${color}\",\"text\":\"${message}\"}]}" \
            "${SLACK_WEBHOOK}" || true
    fi
}

check_certificate_expiry() {
    local domain="$1"
    
    # Get certificate expiry date
    local expiry_date
    expiry_date=$(echo | openssl s_client -servername "$domain" -connect "$domain:443" 2>/dev/null | \
                  openssl x509 -noout -dates | grep notAfter | cut -d= -f2)
    
    if [[ -z "$expiry_date" ]]; then
        log "ERROR: Could not retrieve certificate expiry date for $domain"
        notify_slack "❌ Could not check SSL certificate for $domain" "danger"
        return 1
    fi
    
    # Convert to epoch time
    local expiry_epoch
    expiry_epoch=$(date -d "$expiry_date" +%s)
    local current_epoch
    current_epoch=$(date +%s)
    local days_until_expiry
    days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
    
    log "Certificate for $domain expires in $days_until_expiry days"
    
    # Check if certificate expires soon
    if [[ $days_until_expiry -le $CERT_EXPIRY_DAYS ]]; then
        local message="⚠️ SSL certificate for $domain expires in $days_until_expiry days"
        log "WARNING: $message"
        notify_slack "$message" "warning"
        return 1
    else
        log "Certificate for $domain is valid for $days_until_expiry days"
        return 0
    fi
}

# Check main domain and additional domains
domains=(${DOMAIN_NAME} ${ADDITIONAL_DOMAINS:-})
warning_count=0

for domain in "${domains[@]}"; do
    if ! check_certificate_expiry "$domain"; then
        ((warning_count++))
    fi
done

if [[ $warning_count -gt 0 ]]; then
    log "SSL certificate monitoring found $warning_count certificate(s) expiring soon"
    exit 1
else
    log "All SSL certificates are valid and not expiring soon"
    exit 0
fi
EOF

    chmod +x "/usr/local/bin/leanvibe-ssl-monitor"
    
    # Create daily monitoring cron job
    cat > "/etc/cron.d/leanvibe-ssl-monitor" << EOF
# LeanVibe SSL Certificate Monitoring
# Check certificate expiry daily at 9 AM
0 9 * * * root /usr/local/bin/leanvibe-ssl-monitor
EOF
    
    log "SSL certificate monitoring configured"
}

validate_setup() {
    log "Validating SSL setup"
    
    # Test HTTPS connection
    if curl -sSf "https://${DOMAIN_NAME}/health" >/dev/null 2>&1; then
        log "HTTPS connection test passed"
    else
        log "WARNING: HTTPS connection test failed"
    fi
    
    # Test certificate chain
    if echo | openssl s_client -servername "$DOMAIN_NAME" -connect "$DOMAIN_NAME:443" 2>/dev/null | \
       openssl x509 -noout -text | grep -q "Let's Encrypt"; then
        log "Certificate chain validation passed"
    else
        log "WARNING: Certificate chain validation failed"
    fi
    
    # Test auto-renewal
    if systemctl is-active --quiet leanvibe-ssl-renew.timer; then
        log "Auto-renewal timer is active"
    else
        log "WARNING: Auto-renewal timer is not active"
    fi
    
    log "SSL setup validation completed"
}

main() {
    log "Starting SSL certificate setup for ${DOMAIN_NAME}"
    
    check_prerequisites
    create_temporary_nginx_config
    obtain_certificate
    install_certificate
    create_production_nginx_config
    setup_auto_renewal
    create_monitoring_script
    validate_setup
    
    log "SSL certificate setup completed successfully"
    notify_slack "✅ SSL certificates configured successfully for ${DOMAIN_NAME} and additional domains"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi