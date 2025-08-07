#!/bin/bash
# SSL Certificate Generation Script for Production

set -e

DOMAIN=${DOMAIN_NAME:-localhost}
CERT_DIR="/etc/nginx/ssl"
EMAIL=${ADMIN_EMAIL:-admin@leanvibe.com}

echo "Generating SSL certificates for domain: $DOMAIN"

# Create SSL directory if it doesn't exist
mkdir -p $CERT_DIR

# Generate private key
openssl genrsa -out $CERT_DIR/key.pem 4096

# Generate certificate signing request
openssl req -new -key $CERT_DIR/key.pem -out $CERT_DIR/csr.pem -subj "/C=US/ST=CA/L=San Francisco/O=LeanVibe/OU=Engineering/CN=$DOMAIN"

# Generate self-signed certificate for development
openssl x509 -req -days 365 -in $CERT_DIR/csr.pem -signkey $CERT_DIR/key.pem -out $CERT_DIR/cert.pem

# Set proper permissions
chmod 600 $CERT_DIR/key.pem
chmod 644 $CERT_DIR/cert.pem

echo "SSL certificates generated successfully!"

# For production, use Let's Encrypt
if [ "$ENVIRONMENT" = "production" ] && [ "$DOMAIN" != "localhost" ]; then
    echo "Setting up Let's Encrypt for production..."
    
    # Install certbot if not already installed
    if ! command -v certbot &> /dev/null; then
        echo "Installing certbot..."
        apt-get update
        apt-get install -y certbot python3-certbot-nginx
    fi
    
    # Get Let's Encrypt certificate
    certbot certonly --webroot \
        --webroot-path=/var/www/certbot \
        --email $EMAIL \
        --agree-tos \
        --no-eff-email \
        --staging \
        -d $DOMAIN \
        -d www.$DOMAIN \
        -d app.$DOMAIN
    
    # Copy certificates to nginx directory
    cp /etc/letsencrypt/live/$DOMAIN/fullchain.pem $CERT_DIR/cert.pem
    cp /etc/letsencrypt/live/$DOMAIN/privkey.pem $CERT_DIR/key.pem
    
    # Set up auto-renewal
    crontab -l | { cat; echo "0 12 * * * /usr/bin/certbot renew --quiet"; } | crontab -
    
    echo "Let's Encrypt certificates configured successfully!"
fi

# Generate DH parameters for enhanced security
openssl dhparam -out $CERT_DIR/dhparam.pem 2048

echo "SSL setup complete!"