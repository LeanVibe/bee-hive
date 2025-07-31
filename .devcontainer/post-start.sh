#!/bin/bash

# LeanVibe Agent Hive 2.0 - Dev Container Post-Start Script
# Runs every time the container starts

set -euo pipefail

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}🔄 Starting LeanVibe Agent Hive 2.0 services...${NC}"

# Ensure database and Redis are running
docker compose up -d postgres redis

# Wait for services to be ready
echo -e "${BLUE}⏳ Waiting for services to be ready...${NC}"

# Wait for PostgreSQL
timeout=30
counter=0
while ! docker compose exec -T postgres pg_isready -U leanvibe_user -d leanvibe_agent_hive >/dev/null 2>&1; do
    if [[ $counter -ge $timeout ]]; then
        echo -e "${YELLOW}⚠️  PostgreSQL not ready after ${timeout}s${NC}"
        break
    fi
    sleep 1
    counter=$((counter + 1))
done

# Wait for Redis
counter=0
while ! docker compose exec -T redis redis-cli ping >/dev/null 2>&1; do
    if [[ $counter -ge $timeout ]]; then
        echo -e "${YELLOW}⚠️  Redis not ready after ${timeout}s${NC}"
        break
    fi
    sleep 1
    counter=$((counter + 1))
done

# Activate virtual environment for the session
if [[ -f "/workspace/venv/bin/activate" ]]; then
    source /workspace/venv/bin/activate
fi

# Set PYTHONPATH
export PYTHONPATH="/workspace:$PYTHONPATH"

echo -e "${GREEN}✅ Services are ready!${NC}"
echo -e "${BLUE}🚀 Ready for development. Run '${GREEN}health${BLUE}' to check system status.${NC}"