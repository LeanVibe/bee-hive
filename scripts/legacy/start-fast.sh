#!/bin/bash
# ⚠️  DEPRECATED: Use 'make start' instead
# 🚀 NEW: make start | make start-minimal | make start-full

echo "⚠️  DEPRECATION WARNING: Using legacy start-fast.sh"
echo "🚀 Please use 'make start' instead."
echo "⏳ Continuing in 2 seconds..."
sleep 2

# LeanVibe Agent Hive 2.0 - Fast Startup (LEGACY)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Starting LeanVibe Agent Hive 2.0 (Fast Mode)..."

# Activate virtual environment
source "${SCRIPT_DIR}/venv/bin/activate"

# Start services using fast compose
docker compose -f docker-compose.fast.yml up -d postgres redis

# Wait briefly for services
sleep 5

# Start API with performance optimizations
echo "🌟 Starting optimized application server..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --loop uvloop --http httptools

echo "✅ Agent Hive running at http://localhost:8000"
echo "📊 Health: http://localhost:8000/health"
echo "📖 API Docs: http://localhost:8000/docs"