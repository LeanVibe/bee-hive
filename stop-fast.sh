#!/bin/bash
# LeanVibe Agent Hive 2.0 - Fast Stop

echo "🛑 Stopping LeanVibe Agent Hive 2.0 (Fast Mode)..."
docker compose -f docker-compose.fast.yml down
echo "✅ Agent Hive stopped"