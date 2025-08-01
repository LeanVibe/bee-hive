#!/bin/bash
#
# DEPRECATED: This script will be removed in a future version.
# Please use 'make start' instead.
#
echo "⚠️  WARNING: The ./start-fast.sh entry point is deprecated and will be removed."
echo "   Please use 'make start' instead for the professional interface."
echo "   Run 'make help' to see all available commands."
echo ""
sleep 2 # Give the user a moment to see the warning

# Pass all arguments to the make command
exec make start "$@"