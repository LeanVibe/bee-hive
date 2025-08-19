#!/bin/bash
# Project Index Health Check Script

set -e

PROJECT_NAME="${PROJECT_NAME:-project-index}"
API_URL="${API_URL:-http://localhost:8100}"

echo "🔍 Checking Project Index health..."

# Check API health
echo "📡 Checking API health..."
if curl -sf "$API_URL/health" > /dev/null; then
    echo "✅ API is healthy"
else
    echo "❌ API is not responding"
    exit 1
fi

# Check database connection
echo "🗄️  Checking database connection..."
if docker exec "${PROJECT_NAME}_postgres" pg_isready -U project_index > /dev/null; then
    echo "✅ Database is healthy"
else
    echo "❌ Database is not responding"
    exit 1
fi

# Check Redis connection
echo "📦 Checking Redis connection..."
if docker exec "${PROJECT_NAME}_redis" redis-cli ping > /dev/null; then
    echo "✅ Redis is healthy"
else
    echo "❌ Redis is not responding"
    exit 1
fi

echo "🎉 All services are healthy!"
        