#!/bin/bash
# Production deployment script for Epic 7 Phase 2
# Deploys v2 APIs with production database connectivity

set -e

echo "🚀 Epic 7 Phase 2: Production API Deployment"
echo "============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running from project root
if [ ! -f "docker-compose.production.yml" ]; then
    echo -e "${RED}❌ Please run this script from the project root directory${NC}"
    exit 1
fi

# Create required directories
echo -e "${BLUE}📁 Creating production directories...${NC}"
sudo mkdir -p /var/lib/leanvibe/{postgres,redis,prometheus,grafana,alertmanager}
sudo chown -R $(whoami):$(whoami) /var/lib/leanvibe/

# Copy environment file if it doesn't exist
if [ ! -f ".env.production" ]; then
    echo -e "${YELLOW}⚠️ Creating .env.production from template...${NC}"
    cp .env.example .env.production
    
    # Generate secure secrets
    SECRET_KEY=$(openssl rand -hex 32)
    JWT_SECRET_KEY=$(openssl rand -hex 64)
    POSTGRES_PASSWORD=$(openssl rand -hex 16)
    REDIS_PASSWORD=$(openssl rand -hex 16)
    
    # Update .env.production with production values
    sed -i.bak \
        -e "s/DEBUG=true/DEBUG=false/" \
        -e "s/ENVIRONMENT=development/ENVIRONMENT=production/" \
        -e "s/development-secret-key-change-in-production-minimum-32-chars/$SECRET_KEY/" \
        -e "s/development-jwt-secret-key-change-in-production-minimum-64-chars/$JWT_SECRET_KEY/" \
        -e "s/localhost:5432/postgres:5432/" \
        -e "s/localhost:6379/redis:6379/" \
        .env.production
    
    # Add production-specific variables
    cat >> .env.production << EOF

# Production-specific variables
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
REDIS_PASSWORD=$REDIS_PASSWORD
DOMAIN_NAME=localhost
ADMIN_EMAIL=admin@leanvibe.com
CORS_ORIGINS=https://localhost,https://app.localhost
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=$(openssl rand -hex 12)
EOF

    echo -e "${GREEN}✅ Production environment file created${NC}"
    echo -e "${YELLOW}⚠️ Please update ANTHROPIC_API_KEY in .env.production${NC}"
fi

# Load environment variables
set -a
source .env.production
set +a

# Stop any existing containers
echo -e "${BLUE}🛑 Stopping existing containers...${NC}"
docker-compose -f docker-compose.production.yml down --remove-orphans || true

# Pull latest images
echo -e "${BLUE}📥 Pulling latest Docker images...${NC}"
docker-compose -f docker-compose.production.yml pull

# Build application image
echo -e "${BLUE}🔨 Building application image...${NC}"
docker-compose -f docker-compose.production.yml build api

# Start infrastructure services first
echo -e "${BLUE}🗄️ Starting infrastructure services...${NC}"
docker-compose -f docker-compose.production.yml up -d postgres redis

# Wait for database to be ready
echo -e "${BLUE}⏳ Waiting for database to be ready...${NC}"
max_attempts=30
attempt=1

while ! docker exec leanvibe_postgres_prod pg_isready -U leanvibe_user -d leanvibe_agent_hive >/dev/null 2>&1; do
    if [ $attempt -ge $max_attempts ]; then
        echo -e "${RED}❌ Database failed to start within timeout${NC}"
        exit 1
    fi
    echo "Attempt $attempt/$max_attempts: Waiting for database..."
    sleep 2
    attempt=$((attempt + 1))
done

echo -e "${GREEN}✅ Database is ready${NC}"

# Wait for Redis to be ready  
echo -e "${BLUE}⏳ Waiting for Redis to be ready...${NC}"
max_attempts=15
attempt=1

while ! docker exec leanvibe_redis_prod redis-cli -a "$REDIS_PASSWORD" ping >/dev/null 2>&1; do
    if [ $attempt -ge $max_attempts ]; then
        echo -e "${RED}❌ Redis failed to start within timeout${NC}"
        exit 1
    fi
    echo "Attempt $attempt/$max_attempts: Waiting for Redis..."
    sleep 2
    attempt=$((attempt + 1))
done

echo -e "${GREEN}✅ Redis is ready${NC}"

# Start application services
echo -e "${BLUE}🚀 Starting application services...${NC}"
docker-compose -f docker-compose.production.yml up -d api

# Wait for API to be ready
echo -e "${BLUE}⏳ Waiting for API to be ready...${NC}"
max_attempts=30
attempt=1

while ! curl -s http://localhost:8000/health >/dev/null 2>&1; do
    if [ $attempt -ge $max_attempts ]; then
        echo -e "${RED}❌ API failed to start within timeout${NC}"
        echo -e "${YELLOW}📋 API logs:${NC}"
        docker-compose -f docker-compose.production.yml logs --tail=20 api
        exit 1
    fi
    echo "Attempt $attempt/$max_attempts: Waiting for API..."
    sleep 3
    attempt=$((attempt + 1))
done

echo -e "${GREEN}✅ API is ready${NC}"

# Start monitoring services
echo -e "${BLUE}📊 Starting monitoring services...${NC}"
docker-compose -f docker-compose.production.yml up -d prometheus grafana alertmanager mobile-monitor

# Start reverse proxy
echo -e "${BLUE}🌐 Starting reverse proxy...${NC}"
docker-compose -f docker-compose.production.yml up -d nginx

# Validate deployment
echo -e "${BLUE}🔍 Validating deployment...${NC}"

# Test API health
health_response=$(curl -s http://localhost:8000/health)
if echo "$health_response" | grep -q '"status": "healthy"'; then
    echo -e "${GREEN}✅ API health check passed${NC}"
else
    echo -e "${RED}❌ API health check failed${NC}"
    echo "Response: $health_response"
    exit 1
fi

# Test v2 API root
v2_response=$(curl -s http://localhost:8000/api/v2/)
if echo "$v2_response" | grep -q "LeanVibe Agent Hive 2.0"; then
    echo -e "${GREEN}✅ v2 API root endpoint working${NC}"
else
    echo -e "${RED}❌ v2 API root endpoint failed${NC}"
    echo "Response: $v2_response"
fi

# Test authentication endpoints
auth_health=$(curl -s http://localhost:8000/api/v2/auth/health)
if echo "$auth_health" | grep -q '"status": "healthy"'; then
    echo -e "${GREEN}✅ Authentication service healthy${NC}"
else
    echo -e "${YELLOW}⚠️ Authentication service health check returned: $auth_health${NC}"
fi

# Display status
echo -e "\n${BLUE}📊 Deployment Status:${NC}"
docker-compose -f docker-compose.production.yml ps

# Display endpoints
echo -e "\n${GREEN}🌐 Available Endpoints:${NC}"
echo "API Root: http://localhost:8000/"
echo "API v2: http://localhost:8000/api/v2/"
echo "Health: http://localhost:8000/health"
echo "Docs: http://localhost:8000/docs"
echo "Authentication: http://localhost:8000/api/v2/auth/"
echo "Grafana: http://localhost:3000/"
echo "Prometheus: http://localhost:9090/"

# Display credentials
echo -e "\n${YELLOW}🔐 Default Credentials:${NC}"
echo "Admin User: admin@leanvibe.com"
echo "Admin Password: AdminPassword123!"
echo "Grafana User: admin"
echo "Grafana Password: $(grep GRAFANA_ADMIN_PASSWORD .env.production | cut -d'=' -f2)"

echo -e "\n${GREEN}🎉 Production deployment completed successfully!${NC}"
echo -e "${BLUE}📝 Next steps:${NC}"
echo "1. Test user registration: POST http://localhost:8000/api/v2/auth/register"
echo "2. Test user login: POST http://localhost:8000/api/v2/auth/login"
echo "3. Verify JWT authentication on protected endpoints"
echo "4. Monitor system health via Grafana dashboard"
echo "5. Review logs: docker-compose -f docker-compose.production.yml logs -f"

# Save deployment info
cat > deployment_info.txt << EOF
Epic 7 Phase 2 Deployment - $(date)
====================================

Status: Completed Successfully
API Endpoint: http://localhost:8000/
v2 API: http://localhost:8000/api/v2/
Documentation: http://localhost:8000/docs

Default Admin:
- Email: admin@leanvibe.com
- Password: AdminPassword123!

Services Running:
- PostgreSQL (port 5432)
- Redis (port 6379) 
- API Server (port 8000)
- Nginx Proxy (ports 80, 443)
- Prometheus (port 9090)
- Grafana (port 3000)
- AlertManager (port 9093)

Environment: Production (.env.production)
Containers: $(docker-compose -f docker-compose.production.yml ps --services | wc -l) services
EOF

echo -e "\n${GREEN}📋 Deployment details saved to deployment_info.txt${NC}"