#!/bin/bash

# Session Lifecycle Hook - Consolidated SessionStart/PreCompact/Notification management
# Part of Streamlined Claude Code Hooks System for LeanVibe Agent Hive 2.0

set -euo pipefail

# Read hook data from stdin
HOOK_DATA=$(cat)

# Extract key information
EVENT=$(echo "$HOOK_DATA" | jq -r '.hook_event_name // ""')
SOURCE=$(echo "$HOOK_DATA" | jq -r '.source // ""')
MESSAGE=$(echo "$HOOK_DATA" | jq -r '.message // ""')
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Log hook execution
echo "🔄 Session Lifecycle Hook: $EVENT:$SOURCE at $TIMESTAMP" >&2

case "$EVENT:$SOURCE" in
  "SessionStart:startup")
    echo "🚀 Initializing LeanVibe Agent Hive connection..." >&2
    
    # Initialize session with LeanVibe Agent Hive 2.0
    RESPONSE=$(curl -s -X POST http://localhost:8000/api/claude/session-start \
      -H "Content-Type: application/json" \
      -d "{
        \"hooks_enabled\": true,
        \"mobile_integration\": true,
        \"context\": {
          \"startup_time\": \"$TIMESTAMP\",
          \"hook_version\": \"streamlined-v1.0\"
        }
      }" 2>/dev/null)
    
    if [[ $? -eq 0 ]] && echo "$RESPONSE" | jq -e '.success' >/dev/null 2>&1; then
      SESSION_ID=$(echo "$RESPONSE" | jq -r '.session_id // ""')
      AGENT_COUNT=$(echo "$RESPONSE" | jq -r '.active_agents // 0')
      MOBILE_URL=$(echo "$RESPONSE" | jq -r '.mobile_dashboard_url // ""')
      
      echo "✅ LeanVibe Agent Hive 2.0 active - Enhanced mobile oversight available" >&2
      echo "📱 Mobile Dashboard: $MOBILE_URL" >&2
      echo "🤖 Active Agents: $AGENT_COUNT" >&2
      echo "🆔 Session ID: $SESSION_ID" >&2
      
      # Store session ID for other hooks
      echo "$SESSION_ID" > /tmp/claude_session_id 2>/dev/null || true
    else
      echo "⚠️ Failed to connect to LeanVibe Agent Hive - continuing without integration" >&2
    fi
    ;;
    
  "SessionStart:resume")
    echo "🔄 Restoring agent states and mobile dashboard..." >&2
    
    # Try to get existing session ID
    SESSION_ID=""
    if [[ -f /tmp/claude_session_id ]]; then
      SESSION_ID=$(cat /tmp/claude_session_id 2>/dev/null || echo "")
    fi
    
    # Resume session with LeanVibe Agent Hive 2.0
    RESUME_DATA="{\"hooks_enabled\": true, \"mobile_integration\": true}"
    if [[ -n "$SESSION_ID" ]]; then
      RESUME_DATA=$(echo "$RESUME_DATA" | jq --arg sid "$SESSION_ID" '. + {session_id: $sid}')
    fi
    
    RESPONSE=$(curl -s -X POST http://localhost:8000/api/claude/session-resume \
      -H "Content-Type: application/json" \
      -d "$RESUME_DATA" 2>/dev/null)
    
    if [[ $? -eq 0 ]] && echo "$RESPONSE" | jq -e '.success' >/dev/null 2>&1; then
      NEW_SESSION_ID=$(echo "$RESPONSE" | jq -r '.session_id // ""')
      AGENT_COUNT=$(echo "$RESPONSE" | jq -r '.active_agents // 0')
      RESUME_COUNT=$(echo "$RESPONSE" | jq -r '.integration_status.resume_count // 1')
      
      echo "✅ Session restored - $AGENT_COUNT agents active, resume #$RESUME_COUNT" >&2
      
      # Update stored session ID
      echo "$NEW_SESSION_ID" > /tmp/claude_session_id 2>/dev/null || true
    else
      echo "⚠️ Failed to resume LeanVibe Agent Hive session - starting fresh" >&2
    fi
    ;;
    
  "PreCompact:*")
    echo "🗜️ Preparing for context compaction..." >&2
    
    # Send pre-compaction notification to mobile dashboard
    curl -s -X POST http://localhost:8000/api/mobile/notifications \
      -H "Content-Type: application/json" \
      -d "{
        \"priority\": \"medium\",
        \"title\": \"Context Compaction\",
        \"message\": \"Preparing to compact context - saving current state\",
        \"data\": {
          \"event\": \"pre_compact\",
          \"type\": \"memory_management\",
          \"timestamp\": \"$TIMESTAMP\"
        },
        \"hook_event\": \"PreCompact\"
      }" 2>/dev/null || true
    
    # Save current working state if in git repo
    if git rev-parse --git-dir >/dev/null 2>&1; then
      if ! git diff --quiet HEAD 2>/dev/null; then
        echo "💾 Auto-saving work in progress before compaction..." >&2
        git add -A 2>/dev/null || true
        git commit -m "🤖 Auto-save before context compaction - $(date)" 2>/dev/null || true
      fi
    fi
    
    echo "✅ Pre-compaction preparation completed" >&2
    ;;
    
  "Notification:system"|"Notification:error"|"Notification:warning")
    echo "📢 Processing $SOURCE notification..." >&2
    
    # Determine notification priority based on source
    PRIORITY="medium"
    case "$SOURCE" in
      "error") PRIORITY="high" ;;
      "system") PRIORITY="medium" ;;
      "warning") PRIORITY="medium" ;;
    esac
    
    # Filter and route notifications to mobile
    if [[ -n "$MESSAGE" ]]; then
      # Only send high-priority or critical system notifications to mobile
      if [[ "$PRIORITY" == "high" ]] || [[ "$MESSAGE" == *"critical"* ]] || [[ "$MESSAGE" == *"error"* ]]; then
        curl -s -X POST http://localhost:8000/api/mobile/notifications \
          -H "Content-Type: application/json" \
          -d "{
            \"priority\": \"$PRIORITY\",
            \"title\": \"System Notification\",
            \"message\": \"$MESSAGE\",
            \"data\": {
              \"source\": \"$SOURCE\",
              \"type\": \"system_notification\",
              \"original_event\": \"Notification\"
            },
            \"hook_event\": \"Notification\"
          }" 2>/dev/null || true
        
        echo "📱 Notification routed to mobile dashboard: $MESSAGE" >&2
      else
        echo "ℹ️ Low-priority notification filtered: $MESSAGE" >&2
      fi
    fi
    ;;
    
  *)
    echo "ℹ️ Session manager: No specific action for $EVENT:$SOURCE" >&2
    ;;
esac

# Health check - ensure LeanVibe connection is healthy
if curl -s http://localhost:8000/api/mobile/status >/dev/null 2>&1; then
  echo "✅ LeanVibe Agent Hive connection healthy" >&2
else
  echo "⚠️ LeanVibe Agent Hive connection unavailable" >&2
fi

# Always log successful hook execution
echo "✅ Session Lifecycle Hook completed successfully: $EVENT:$SOURCE" >&2

# Return success
exit 0