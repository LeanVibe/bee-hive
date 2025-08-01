#!/bin/bash

# ⚠️  MIGRATION WRAPPER - DEPRECATED SCRIPT NAME
# 
# 🚀 NEW COMMAND: make setup
# 📖 Migration guide: docs/MIGRATION.md

set -euo pipefail

echo "⚠️  MIGRATION NOTICE: 'setup-fast.sh' is deprecated"
echo "🚀 NEW: Use 'make setup' instead"
echo "⏳ Auto-redirecting in 3 seconds..."
sleep 3

# Log usage
echo "$(date): Legacy setup-fast.sh accessed, redirected to 'make setup'" >> .migration_usage.log

exec make setup