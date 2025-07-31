#!/bin/bash
# ⚠️  DEPRECATED: Use 'make stop' instead

echo "⚠️  DEPRECATION WARNING: Using legacy stop-fast.sh"
echo "🚀 Please use 'make stop' instead."
sleep 1

# LeanVibe Agent Hive 2.0 - Fast Stop (LEGACY)

echo "🛑 Stopping LeanVibe Agent Hive 2.0 (Fast Mode)..."
docker compose -f docker-compose.fast.yml down
echo "✅ Agent Hive stopped"