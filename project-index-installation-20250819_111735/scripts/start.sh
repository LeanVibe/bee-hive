#!/bin/bash
# Project Index Start Script

set -e

PROJECT_NAME="bee-hive"
COMPOSE_FILE="docker-compose.yml"

echo "🚀 Starting Project Index infrastructure..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Pull latest images
echo "📥 Pulling latest images..."
docker-compose -f "$COMPOSE_FILE" pull

# Build custom images if needed
echo "🔨 Building custom images..."
docker-compose -f "$COMPOSE_FILE" build

# Start services
echo "🌟 Starting services..."
docker-compose -f "$COMPOSE_FILE" up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Run health check
if [ -f "scripts/health-check.sh" ]; then
    echo "🔍 Running health check..."
    ./scripts/health-check.sh
fi

echo "✅ Project Index is ready!"
echo "📊 API: http://localhost:8100"
if 'project-index-dashboard' in config.services:
    echo "📈 Dashboard: http://localhost:8101"
if config.monitoring_enabled and 'prometheus' in config.services:
    echo "📊 Metrics: http://localhost:9090"
        