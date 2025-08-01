#!/bin/bash

# ⚠️  MIGRATION WRAPPER - DEPRECATED SCRIPT NAME
# 
# 🚀 NEW COMMAND: make stop
# 📖 Migration guide: docs/MIGRATION.md

set -euo pipefail

echo "⚠️  MIGRATION NOTICE: 'stop-fast.sh' is deprecated"
echo "🚀 NEW: Use 'make stop' instead"
echo "⏳ Auto-redirecting in 2 seconds..."
sleep 2

# Log usage
echo "$(date): Legacy stop-fast.sh accessed, redirected to 'make stop'" >> .migration_usage.log

exec make stop