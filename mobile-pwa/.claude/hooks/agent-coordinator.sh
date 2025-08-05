#!/bin/bash

# Agent Coordination Hook - LeanVibe Agent Hive 2.0 multi-agent system integration
# Part of Streamlined Claude Code Hooks System for autonomous development coordination

set -euo pipefail

# Read hook data from stdin
HOOK_DATA=$(cat)

# Extract key information
EVENT=$(echo "$HOOK_DATA" | jq -r '.hook_event_name // ""')
AGENT_ID=$(echo "$HOOK_DATA" | jq -r '.agent_id // ""')
TASK_ID=$(echo "$HOOK_DATA" | jq -r '.task_id // ""')
STATUS=$(echo "$HOOK_DATA" | jq -r '.status // ""')
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Log hook execution
echo "🤖 Agent Coordination Hook: $EVENT for agent $AGENT_ID at $TIMESTAMP" >&2

case "$EVENT" in
  "Task")
    echo "📋 Coordinating with LeanVibe Agent Hive for task management..." >&2
    
    # Extract task details
    TASK_DESCRIPTION=$(echo "$HOOK_DATA" | jq -r '.task_description // ""')
    TASK_TYPE=$(echo "$HOOK_DATA" | jq -r '.task_type // "general"')
    PRIORITY=$(echo "$HOOK_DATA" | jq -r '.priority // "medium"')
    
    # Coordinate with LeanVibe Agent Hive multi-agent system
    COORDINATION_RESPONSE=$(curl -s -X POST http://localhost:8000/api/agents/coordinate \
      -H "Content-Type: application/json" \
      -d "{
        \"task_id\": \"$TASK_ID\",
        \"agent_id\": \"$AGENT_ID\",
        \"task_type\": \"$TASK_TYPE\",
        \"description\": \"$TASK_DESCRIPTION\",
        \"priority\": \"$PRIORITY\",
        \"timestamp\": \"$TIMESTAMP\",
        \"source\": \"claude_code_hook\"
      }" 2>/dev/null)
    
    if [[ $? -eq 0 ]] && echo "$COORDINATION_RESPONSE" | jq -e '.success' >/dev/null 2>&1; then
      ASSIGNED_AGENTS=$(echo "$COORDINATION_RESPONSE" | jq -r '.assigned_agents // 0')
      COORDINATION_ID=$(echo "$COORDINATION_RESPONSE" | jq -r '.coordination_id // ""')
      
      echo "✅ Task coordinated with $ASSIGNED_AGENTS agents (ID: $COORDINATION_ID)" >&2
      
      # Send mobile notification about task coordination
      curl -s -X POST http://localhost:8000/api/mobile/notifications \
        -H "Content-Type: application/json" \
        -d "{
          \"priority\": \"$PRIORITY\",
          \"title\": \"Task Coordinated\",
          \"message\": \"Task assigned to $ASSIGNED_AGENTS agents in autonomous development platform\",
          \"data\": {
            \"task_id\": \"$TASK_ID\",
            \"agent_id\": \"$AGENT_ID\",
            \"coordination_id\": \"$COORDINATION_ID\",
            \"assigned_agents\": $ASSIGNED_AGENTS,
            \"type\": \"task_coordination\"
          },
          \"hook_event\": \"Task\"
        }" 2>/dev/null || true
    else
      echo "⚠️ Failed to coordinate with LeanVibe Agent Hive - task proceeding independently" >&2
      
      # Send fallback notification
      curl -s -X POST http://localhost:8000/api/mobile/notifications \
        -H "Content-Type: application/json" \
        -d "{
          \"priority\": \"high\",
          \"title\": \"Coordination Failed\",
          \"message\": \"Unable to coordinate task with agent system - operating in standalone mode\",
          \"data\": {
            \"task_id\": \"$TASK_ID\",
            \"agent_id\": \"$AGENT_ID\",
            \"type\": \"coordination_failure\"
          },
          \"hook_event\": \"Task\"
        }" 2>/dev/null || true
    fi
    ;;
    
  "SubagentStop")
    echo "🛑 Processing subagent stop event..." >&2
    
    # Extract stop details
    STOP_REASON=$(echo "$HOOK_DATA" | jq -r '.stop_reason // "completed"')
    TASK_RESULT=$(echo "$HOOK_DATA" | jq -r '.task_result // ""')
    SUCCESS=$(echo "$HOOK_DATA" | jq -r '.success // false')
    
    # Update mobile dashboard with agent completion
    curl -s -X POST http://localhost:8000/api/mobile/agent-update \
      -H "Content-Type: application/json" \
      -d "{
        \"agent_id\": \"$AGENT_ID\",
        \"status\": \"stopped\",
        \"event\": \"agent_stop\",
        \"data\": {
          \"stop_reason\": \"$STOP_REASON\",
          \"task_result\": \"$TASK_RESULT\",
          \"success\": $SUCCESS,
          \"timestamp\": \"$TIMESTAMP\"
        }
      }" 2>/dev/null
    
    if [[ $? -eq 0 ]]; then
      echo "📱 Mobile dashboard updated with agent stop event" >&2
    else
      echo "⚠️ Failed to update mobile dashboard" >&2
    fi
    
    # Send completion notification based on success status
    if [[ "$SUCCESS" == "true" ]]; then
      NOTIFICATION_TITLE="Agent Task Completed"
      NOTIFICATION_MESSAGE="Agent $AGENT_ID successfully completed task"
      NOTIFICATION_PRIORITY="medium"
    else
      NOTIFICATION_TITLE="Agent Task Failed"
      NOTIFICATION_MESSAGE="Agent $AGENT_ID stopped with failure: $STOP_REASON"
      NOTIFICATION_PRIORITY="high"
    fi
    
    curl -s -X POST http://localhost:8000/api/mobile/notifications \
      -H "Content-Type: application/json" \
      -d "{
        \"priority\": \"$NOTIFICATION_PRIORITY\",
        \"title\": \"$NOTIFICATION_TITLE\",
        \"message\": \"$NOTIFICATION_MESSAGE\",
        \"data\": {
          \"agent_id\": \"$AGENT_ID\",
          \"stop_reason\": \"$STOP_REASON\",
          \"success\": $SUCCESS,
          \"type\": \"agent_completion\"
        },
        \"hook_event\": \"SubagentStop\"
      }" 2>/dev/null || true
    
    echo "✅ Subagent stop processed and mobile dashboard notified" >&2
    ;;
    
  "AgentStart")
    echo "🚀 Processing agent start event..." >&2
    
    # Extract agent details
    AGENT_TYPE=$(echo "$HOOK_DATA" | jq -r '.agent_type // "general"')
    CAPABILITIES=$(echo "$HOOK_DATA" | jq -r '.capabilities // ""')
    
    # Update mobile dashboard with agent start
    curl -s -X POST http://localhost:8000/api/mobile/agent-update \
      -H "Content-Type: application/json" \
      -d "{
        \"agent_id\": \"$AGENT_ID\",
        \"status\": \"active\",
        \"event\": \"agent_start\",
        \"data\": {
          \"agent_type\": \"$AGENT_TYPE\",
          \"capabilities\": \"$CAPABILITIES\",
          \"timestamp\": \"$TIMESTAMP\"
        }
      }" 2>/dev/null
    
    # Send start notification
    curl -s -X POST http://localhost:8000/api/mobile/notifications \
      -H "Content-Type: application/json" \
      -d "{
        \"priority\": \"medium\",
        \"title\": \"Agent Started\",
        \"message\": \"$AGENT_TYPE agent $AGENT_ID is now active in the autonomous development platform\",
        \"data\": {
          \"agent_id\": \"$AGENT_ID\",
          \"agent_type\": \"$AGENT_TYPE\",
          \"capabilities\": \"$CAPABILITIES\",
          \"type\": \"agent_start\"
        },
        \"hook_event\": \"AgentStart\"
      }" 2>/dev/null || true
    
    echo "✅ Agent start processed and mobile dashboard updated" >&2
    ;;
    
  *)
    echo "ℹ️ Agent coordinator: No specific action for event $EVENT" >&2
    
    # Generic agent event notification
    if [[ -n "$AGENT_ID" ]]; then
      curl -s -X POST http://localhost:8000/api/mobile/notifications \
        -H "Content-Type: application/json" \
        -d "{
          \"priority\": \"low\",
          \"title\": \"Agent Event\",
          \"message\": \"Agent $AGENT_ID triggered event: $EVENT\",
          \"data\": {
            \"agent_id\": \"$AGENT_ID\",
            \"event\": \"$EVENT\",
            \"type\": \"generic_agent_event\"
          },
          \"hook_event\": \"$EVENT\"
        }" 2>/dev/null || true
    fi
    ;;
esac

# Health check - verify LeanVibe Agent Hive coordination is available
HEALTH_CHECK=$(curl -s http://localhost:8000/api/agents/status 2>/dev/null || echo '{"error": "unavailable"}')
if echo "$HEALTH_CHECK" | jq -e '.active_agents' >/dev/null 2>&1; then
  ACTIVE_AGENTS=$(echo "$HEALTH_CHECK" | jq -r '.active_agents // 0')
  echo "✅ LeanVibe Agent Hive coordination healthy - $ACTIVE_AGENTS active agents" >&2
else
  echo "⚠️ LeanVibe Agent Hive coordination unavailable - operating in standalone mode" >&2
fi

# Check mobile dashboard connectivity
if curl -s http://localhost:8000/api/mobile/status >/dev/null 2>&1; then
  echo "📱 Mobile dashboard connectivity confirmed" >&2
else
  echo "⚠️ Mobile dashboard unavailable" >&2
fi

# Always log successful hook execution
echo "✅ Agent Coordination Hook completed successfully: $EVENT" >&2

# Return success
exit 0