#!/bin/bash

# ⚠️  MIGRATION WRAPPER - DEPRECATED SCRIPT NAME
# 
# This wrapper script helps users migrate from old script names to the new
# standardized Makefile approach.
#
# 🚀 NEW RECOMMENDED COMMANDS:
#   make setup           # Standard setup (replaces setup.sh)
#   make setup-minimal   # Minimal setup for CI/CD
#   make setup-full      # Complete development setup
#
# This wrapper will be removed in a future version.

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}   🚨 MIGRATION NOTICE: Script Organization Update${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${RED}⚠️  The script 'setup.sh' has been deprecated and moved to scripts/legacy/${NC}"
echo ""
echo -e "${GREEN}🚀 NEW STANDARDIZED APPROACH:${NC}"
echo -e "   ${BLUE}make setup${NC}           # Standard setup (recommended)"
echo -e "   ${BLUE}make setup-minimal${NC}   # Minimal setup for CI/CD"
echo -e "   ${BLUE}make setup-full${NC}      # Complete development setup"
echo ""
echo -e "${GREEN}🎯 BENEFITS OF NEW APPROACH:${NC}"
echo -e "   ✅ Consistent command interface across all operations"
echo -e "   ✅ Better error handling and progress reporting"
echo -e "   ✅ Integrated with development workflow"
echo -e "   ✅ Cross-platform compatibility"
echo ""
echo -e "${YELLOW}📖 For full migration guide, see: docs/MIGRATION.md${NC}"
echo ""
echo -e "${BLUE}🔄 AUTO-REDIRECTING TO: make setup${NC}"
echo -e "⏳ Starting in 5 seconds... (Ctrl+C to cancel)"
echo ""

# Countdown with visual indicator
for i in 5 4 3 2 1; do
    echo -ne "   ${YELLOW}$i${NC}..."
    sleep 1
done
echo -e "\n"

# Log usage for monitoring migration progress
echo "$(date): Legacy setup.sh accessed, redirected to 'make setup'" >> .migration_usage.log

# Execute the new command
echo -e "${GREEN}🚀 Executing: make setup${NC}"
exec make setup