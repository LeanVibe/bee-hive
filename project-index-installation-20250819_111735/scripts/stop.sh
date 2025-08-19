#!/bin/bash
# Project Index Stop Script

set -e

COMPOSE_FILE="docker-compose.yml"

echo "🛑 Stopping Project Index infrastructure..."

if [ -f "$COMPOSE_FILE" ]; then
    docker-compose -f "$COMPOSE_FILE" down
    echo "✅ All services stopped"
else
    echo "❌ Docker Compose file not found"
    exit 1
fi
        