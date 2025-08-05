#!/bin/bash

# Quality Gate Hook - Consolidated PreToolUse/PostToolUse/Stop validation
# Part of Streamlined Claude Code Hooks System for LeanVibe Agent Hive 2.0

set -euo pipefail

# Read hook data from stdin
HOOK_DATA=$(cat)

# Extract key information
TOOL_NAME=$(echo "$HOOK_DATA" | jq -r '.tool_name // ""')
EVENT=$(echo "$HOOK_DATA" | jq -r '.hook_event_name // ""')
CONTENT=$(echo "$HOOK_DATA" | jq -r '.content // ""')
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Log hook execution
echo "🎯 Quality Gate Hook: $EVENT:$TOOL_NAME at $TIMESTAMP" >&2

case "$EVENT:$TOOL_NAME" in
  "PreToolUse:Bash")
    echo "🔍 Validating bash command before execution..." >&2
    
    # Extract command from hook data
    COMMAND=$(echo "$HOOK_DATA" | jq -r '.command // ""')
    
    # Basic bash command validation
    if [[ "$COMMAND" == *"rm -rf /"* ]] || [[ "$COMMAND" == *":(){ :|:& };:"* ]]; then
      echo "❌ Dangerous bash command blocked: $COMMAND" >&2
      
      # Send critical notification to mobile dashboard
      curl -s -X POST http://localhost:8000/api/mobile/notifications \
        -H "Content-Type: application/json" \
        -d "{
          \"priority\": \"critical\",
          \"title\": \"Dangerous Command Blocked\",
          \"message\": \"Potentially harmful bash command was blocked by quality gate\",
          \"data\": {
            \"command\": \"$COMMAND\",
            \"type\": \"security_violation\"
          },
          \"hook_event\": \"PreToolUse\"
        }" || echo "⚠️ Failed to send mobile notification" >&2
      
      exit 1
    fi
    
    echo "✅ Bash command validation passed" >&2
    ;;
    
  "PostToolUse:Edit"|"PostToolUse:Write")
    echo "🎨 Running code formatter after file edit..." >&2
    
    # Extract file path if available
    FILE_PATH=$(echo "$HOOK_DATA" | jq -r '.file_path // ""')
    
    if [[ -n "$FILE_PATH" && "$FILE_PATH" == *.py ]]; then
      # Format Python files if black is available
      if command -v black >/dev/null 2>&1; then
        echo "🐍 Formatting Python file: $FILE_PATH" >&2
        black --quiet "$FILE_PATH" 2>/dev/null || echo "⚠️ Black formatting failed" >&2
      fi
    elif [[ -n "$FILE_PATH" && "$FILE_PATH" == *.ts ]]; then
      # Format TypeScript files if prettier is available
      if command -v prettier >/dev/null 2>&1; then
        echo "📝 Formatting TypeScript file: $FILE_PATH" >&2
        prettier --write "$FILE_PATH" >/dev/null 2>&1 || echo "⚠️ Prettier formatting failed" >&2
      fi
    fi
    
    echo "✅ Code formatting completed" >&2
    ;;
    
  "PostToolUse:*")
    echo "🧪 Running comprehensive validation after tool use..." >&2
    
    # Check if we're in a git repository
    if git rev-parse --git-dir >/dev/null 2>&1; then
      # Check for uncommitted changes that might need attention
      if ! git diff --quiet HEAD 2>/dev/null; then
        echo "📋 Uncommitted changes detected - consider committing" >&2
        
        # Send notification about uncommitted changes
        curl -s -X POST http://localhost:8000/api/mobile/notifications \
          -H "Content-Type: application/json" \
          -d "{
            \"priority\": \"medium\",
            \"title\": \"Uncommitted Changes\",
            \"message\": \"Code changes detected - consider committing your work\",
            \"data\": {
              \"tool_name\": \"$TOOL_NAME\",
              \"type\": \"code_changes\"
            },
            \"hook_event\": \"PostToolUse\"
          }" 2>/dev/null || true
      fi
    fi
    
    # Run basic tests if test files exist and we have testing tools
    if [[ -f "pytest.ini" || -f "pyproject.toml" ]] && command -v pytest >/dev/null 2>&1; then
      echo "🧪 Running Python tests..." >&2
      if timeout 30s pytest --quiet --tb=no >/dev/null 2>&1; then
        echo "✅ Tests passed" >&2
      else
        echo "⚠️ Tests failed or timed out" >&2
        
        # Send test failure notification
        curl -s -X POST http://localhost:8000/api/mobile/notifications \
          -H "Content-Type: application/json" \
          -d "{
            \"priority\": \"high\",
            \"title\": \"Tests Failed\",
            \"message\": \"Python tests failed after code changes\",
            \"data\": {
              \"tool_name\": \"$TOOL_NAME\",
              \"type\": \"test_failure\"
            },
            \"hook_event\": \"PostToolUse\"
          }" 2>/dev/null || true
      fi
    fi
    
    echo "✅ Comprehensive validation completed" >&2
    ;;
    
  "Stop:*")
    echo "🛑 Processing stop event for quality gate..." >&2
    
    # Send session stop notification to mobile dashboard
    curl -s -X POST http://localhost:8000/api/mobile/notifications \
      -H "Content-Type: application/json" \
      -d "{
        \"priority\": \"medium\",
        \"title\": \"Session Stopped\",
        \"message\": \"Claude Code session ended - quality gate processing complete\",
        \"data\": {
          \"event\": \"session_stop\",
          \"type\": \"session_lifecycle\"
        },
        \"hook_event\": \"Stop\"
      }" 2>/dev/null || true
    
    echo "✅ Stop event processed" >&2
    ;;
    
  *)
    echo "ℹ️ Quality gate: No specific action for $EVENT:$TOOL_NAME" >&2
    ;;
esac

# Always log successful hook execution
echo "✅ Quality Gate Hook completed successfully: $EVENT:$TOOL_NAME" >&2

# Return success
exit 0