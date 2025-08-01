#!/bin/bash

# ⚠️  MIGRATION WRAPPER - DEPRECATED SCRIPT NAME
# 
# 🚀 NEW COMMAND: make start
# 📖 Migration guide: docs/MIGRATION.md

set -euo pipefail

echo "⚠️  MIGRATION NOTICE: 'start-fast.sh' is deprecated"
echo "🚀 NEW: Use 'make start' instead"
echo "⏳ Auto-redirecting in 2 seconds..."
sleep 2

# Log usage
echo "$(date): Legacy start-fast.sh accessed, redirected to 'make start'" >> .migration_usage.log

exec make start