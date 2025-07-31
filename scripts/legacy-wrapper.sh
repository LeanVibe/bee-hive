#!/bin/bash

# LeanVibe Agent Hive 2.0 - Legacy Script Wrapper
# Maintains backward compatibility while transitioning to new structure
#
# This script provides a migration path for existing scripts

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Color codes
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

show_migration_notice() {
    local old_script="$1"
    local new_command="$2"
    
    echo -e "${YELLOW}📢 MIGRATION NOTICE${NC}"
    echo "================="
    echo
    echo "You're using the legacy script: $old_script"
    echo -e "Please use the new command: ${BLUE}$new_command${NC}"
    echo
    echo "The legacy script will continue to work but is deprecated."
    echo "New command provides better error handling and features."
    echo
    echo "Continuing with legacy script in 3 seconds..."
    sleep 3
    echo
}

# Handle different legacy scripts
case "$(basename "$0")" in
    "setup.sh")
        show_migration_notice "setup.sh" "make setup"
        # Redirect to new setup script
        exec "$SCRIPT_DIR/scripts/setup.sh" fast
        ;;
    "setup-fast.sh")
        show_migration_notice "setup-fast.sh" "make setup"
        exec "$SCRIPT_DIR/scripts/setup.sh" fast
        ;;
    "setup-ultra-fast.sh")
        show_migration_notice "setup-ultra-fast.sh" "make setup-minimal"
        exec "$SCRIPT_DIR/scripts/setup.sh" minimal
        ;;
    "start-fast.sh")
        show_migration_notice "start-fast.sh" "make start"
        exec "$SCRIPT_DIR/scripts/start.sh" fast
        ;;
    "start-sandbox-demo.sh")
        show_migration_notice "start-sandbox-demo.sh" "make sandbox"
        exec "$SCRIPT_DIR/scripts/sandbox.sh" interactive
        ;;
    *)
        echo "Unknown legacy script: $(basename "$0")"
        exit 1
        ;;
esac