#!/bin/bash
#
# DEPRECATED: This script will be removed in a future version.
# Please use 'make setup' instead.
#
echo "⚠️  WARNING: The ./setup-fast.sh entry point is deprecated and will be removed."
echo "   Please use 'make setup' instead for the professional interface."
echo "   Run 'make help' to see all available commands."
echo ""
sleep 2 # Give the user a moment to see the warning

# Pass all arguments to the make command
exec make setup "$@"