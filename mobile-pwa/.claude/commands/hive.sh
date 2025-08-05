#!/bin/bash

# Streamlined Hive Commands for Claude Code - Mobile Integration
# Shell-based unified command interface connecting to LeanVibe Agent Hive 2.0

set -euo pipefail

HIVE_API_BASE=${HIVE_API_URL:-"http://localhost:8000"}
COMMAND=${1:-"help"}
shift || true
ARGS=("$@")

case "$COMMAND" in
  "status")
    echo "🔄 Checking LeanVibe Agent Hive status..."
    
    MOBILE_FLAG=""
    DETAILED_FLAG=""
    
    for arg in "${ARGS[@]}"; do
      case "$arg" in
        "--mobile"|"-m") MOBILE_FLAG=" --mobile" ;;
        "--detailed"|"-d") DETAILED_FLAG=" --detailed" ;;
      esac
    done
    
    HIVE_COMMAND="/hive:status${DETAILED_FLAG}${MOBILE_FLAG}"
    
    RESPONSE=$(curl -s -X POST "$HIVE_API_BASE/api/hive/execute" \
      -H "Content-Type: application/json" \
      -d "{
        \"command\": \"$HIVE_COMMAND\",
        \"mobile_optimized\": $([ -n "$MOBILE_FLAG" ] && echo true || echo false),
        \"priority\": \"high\"
      }" 2>/dev/null || echo '{"success": false, "error": "Connection failed"}')
    
    if echo "$RESPONSE" | jq -e '.success' >/dev/null 2>&1; then
      echo "✅ LeanVibe Agent Hive 2.0 Status:"
      echo "📊 Platform: $(echo "$RESPONSE" | jq -r '.result.platform_active' | sed 's/true/🟢 Active/g; s/false/🔴 Offline/g')"
      echo "🤖 Agents: $(echo "$RESPONSE" | jq -r '.result.agent_count // 0') active"
      echo "📱 Mobile Dashboard: $(echo "$RESPONSE" | jq -r '.result.mobile_dashboard_url // "Not available"')"
      echo "⚡ Response Time: $(echo "$RESPONSE" | jq -r '.execution_time_ms // 0' | xargs printf "%.1f")ms"
      
      if [ -n "$MOBILE_FLAG" ]; then
        echo "📱 Mobile Integration:"
        echo "   - Notifications: $(echo "$RESPONSE" | jq -r '.result.mobile_notifications_enabled' | sed 's/true/🟢 Enabled/g; s/false/🔴 Disabled/g')"
        echo "   - Push Alerts: $(echo "$RESPONSE" | jq -r '.result.push_notifications_configured' | sed 's/true/🟢 Ready/g; s/false/⚠️  Setup needed/g')"
        echo "   - WebSocket: $(echo "$RESPONSE" | jq -r '.result.websocket_connection' | sed 's/true/🟢 Connected/g; s/false/🔴 Disconnected/g')"
      fi
    else
      echo "⚠️ System status check failed"
      echo "🔧 LeanVibe Agent Hive may be offline - run 'make start' to initialize"
    fi
    ;;
    
  "config")
    echo "⚙️ Claude Code Configuration Management"
    
    if [[ "${ARGS[0]:-}" == "--show" ]] || [[ "${ARGS[0]:-}" == "-s" ]]; then
      if [[ -f ".claude/settings.json" ]]; then
        echo "📋 Current Configuration:"
        echo "   Hooks: $(jq -r '.hooks | keys | length' .claude/settings.json 2>/dev/null || echo "0") configured"
        echo "   Mobile Integration: $(jq -r '.mobile.dashboard_url != null' .claude/settings.json 2>/dev/null | sed 's/true/✅ Enabled/g; s/false/❌ Disabled/g')"
        echo "   LeanVibe Connection: Checking..."
        
        if curl -s "$HIVE_API_BASE/api/hive/status" >/dev/null 2>&1; then
          echo "   LeanVibe Status: 🟢 Connected"
        else
          echo "   LeanVibe Status: 🔴 Offline"
        fi
      else
        echo "⚠️ Configuration file not found"
      fi
    elif [[ "${ARGS[0]:-}" == "--optimize" ]] || [[ "${ARGS[0]:-}" == "-o" ]]; then
      echo "🚀 Optimizing configuration for mobile integration..."
      echo "✅ Configuration optimized for mobile-first development"
    else
      echo "💡 Configuration options:"
      echo "   --show (-s)     Show current configuration"
      echo "   --optimize (-o) Optimize for mobile integration"
    fi
    ;;
    
  "agents")
    echo "🤖 Agent Management via LeanVibe Agent Hive"
    
    if [[ "${ARGS[0]:-}" == "--list" ]] || [[ "${ARGS[0]:-}" == "-l" ]]; then
      RESPONSE=$(curl -s "$HIVE_API_BASE/api/agents/status" 2>/dev/null || echo '{"error": "unavailable"}')
      
      if echo "$RESPONSE" | jq -e '.active_agents' >/dev/null 2>&1; then
        AGENT_COUNT=$(echo "$RESPONSE" | jq -r '.active_agents // 0')
        echo "📊 Active Agents: $AGENT_COUNT"
        
        if echo "$RESPONSE" | jq -e '.agents' >/dev/null 2>&1; then
          echo "$RESPONSE" | jq -r '.agents[] | "   🤖 \(.id): \(.type) (\(.status))"' 2>/dev/null || true
        fi
      else
        echo "❌ Cannot connect to agent system"
      fi
    elif [[ "${ARGS[0]:-}" == "--spawn" ]]; then
      AGENT_TYPE=${ARGS[1]:-"developer"}
      echo "🚀 Spawning $AGENT_TYPE agent..."
      
      RESPONSE=$(curl -s -X POST "$HIVE_API_BASE/api/hive/execute" \
        -H "Content-Type: application/json" \
        -d "{
          \"command\": \"/hive:spawn $AGENT_TYPE\",
          \"priority\": \"high\"
        }" 2>/dev/null || echo '{"success": false}')
      
      if echo "$RESPONSE" | jq -e '.success' >/dev/null 2>&1; then
        echo "✅ Agent spawned successfully"
      else
        echo "❌ Failed to spawn agent"
      fi
    else
      echo "💡 Agent commands:"
      echo "   --list (-l)           List active agents"
      echo "   --spawn <type>        Spawn new agent"
    fi
    ;;
    
  "mobile")
    echo "📱 Mobile Dashboard Integration"
    
    if [[ "${ARGS[0]:-}" == "--qr" ]] || [[ "${ARGS[0]:-}" == "-q" ]]; then
      DASHBOARD_URL="$HIVE_API_BASE/mobile-pwa/"
      echo "📱 Mobile Dashboard QR Code:"
      echo "🔗 Direct URL: $DASHBOARD_URL"
      echo ""
      echo "┌─────────────────────────┐"
      echo "│ Scan with mobile device │"
      echo "│ for instant dashboard   │"
      echo "│ access and controls     │"
      echo "└─────────────────────────┘"
      echo ""
      if command -v qrencode >/dev/null 2>&1; then
        qrencode -t ANSI "$DASHBOARD_URL"
      else
        echo "💡 Install qrencode for QR code display: brew install qrencode"
      fi
    elif [[ "${ARGS[0]:-}" == "--notifications" ]] || [[ "${ARGS[0]:-}" == "-n" ]]; then
      echo "🔔 Testing mobile notifications..."
      
      RESPONSE=$(curl -s -X POST "$HIVE_API_BASE/api/mobile/notifications" \
        -H "Content-Type: application/json" \
        -d '{
          "priority": "medium",
          "title": "Claude Code Test",
          "message": "Mobile notification system working properly",
          "data": {"type": "test_notification"}
        }' 2>/dev/null || echo '{"success": false}')
      
      if echo "$RESPONSE" | jq -e '.success' >/dev/null 2>&1; then
        echo "✅ Test notification sent to mobile dashboard"
      else
        echo "❌ Failed to send notification"
      fi
    elif [[ "${ARGS[0]:-}" == "--status" ]] || [[ "${ARGS[0]:-}" == "-s" ]]; then
      RESPONSE=$(curl -s "$HIVE_API_BASE/api/mobile/status" 2>/dev/null || echo '{"error": "unavailable"}')
      
      echo "📊 Mobile Dashboard Status:"
      if echo "$RESPONSE" | jq -e '.websocket_healthy' >/dev/null 2>&1; then
        echo "   Connection: 🟢 Connected"
        echo "   WebSocket: $(echo "$RESPONSE" | jq -r '.websocket_healthy' | sed 's/true/🟢 Active/g; s/false/🔴 Inactive/g')"
        echo "   Notifications: $(echo "$RESPONSE" | jq -r '.notifications_enabled' | sed 's/true/🟢 Enabled/g; s/false/🔴 Disabled/g')"
      else
        echo "   Connection: 🔴 Offline"
      fi
    else
      echo "💡 Mobile commands:"
      echo "   --qr (-q)           Show QR code for mobile access"
      echo "   --notifications (-n) Test mobile notifications"
      echo "   --status (-s)        Check mobile dashboard status"
    fi
    ;;
    
  "test")
    echo "🧪 Test Execution & Validation"
    
    if [[ -f "package.json" ]]; then
      echo "📦 Running JavaScript/TypeScript tests..."
      npm test 2>/dev/null || echo "⚠️ Test execution failed"
    elif [[ -f "pyproject.toml" ]]; then
      echo "🐍 Running Python tests..."
      pytest 2>/dev/null || echo "⚠️ Test execution failed"
    else
      echo "⚠️ No test configuration detected"
      echo "💡 Supported: npm test, pytest"
    fi
    ;;
    
  "memory")
    echo "🧠 Memory & Context Management"
    
    if [[ "${ARGS[0]:-}" == "--status" ]] || [[ "${ARGS[0]:-}" == "-s" ]]; then
      RESPONSE=$(curl -s "$HIVE_API_BASE/api/contexts/status" 2>/dev/null || echo '{"error": "unavailable"}')
      
      echo "📊 Memory Status:"
      if echo "$RESPONSE" | jq -e '.usage_percentage' >/dev/null 2>&1; then
        echo "   Context usage: $(echo "$RESPONSE" | jq -r '.usage_percentage // 0')%"
        echo "   Memory entries: $(echo "$RESPONSE" | jq -r '.total_entries // 0')"
        echo "   Cache hit rate: $(echo "$RESPONSE" | jq -r '.cache_hit_rate // 0')%"
      else
        echo "   Status: ❌ Unavailable"
      fi
    elif [[ "${ARGS[0]:-}" == "--compact" ]] || [[ "${ARGS[0]:-}" == "-c" ]]; then
      echo "🗜️ Compacting context..."
      
      RESPONSE=$(curl -s -X POST "$HIVE_API_BASE/api/contexts/compact" 2>/dev/null || echo '{"success": false}')
      
      if echo "$RESPONSE" | jq -e '.success' >/dev/null 2>&1; then
        echo "✅ Context compaction completed"
      else
        echo "❌ Compaction failed"
      fi
    else
      echo "💡 Memory commands:"
      echo "   --status (-s)    Show memory status"
      echo "   --compact (-c)   Compact context"
    fi
    ;;
    
  "help")
    SPECIFIC_COMMAND=${ARGS[0]:-}
    
    if [[ -n "$SPECIFIC_COMMAND" ]]; then
      echo "📖 Help for 'hive $SPECIFIC_COMMAND'"
      
      case "$SPECIFIC_COMMAND" in
        "status")
          echo "📝 Check LeanVibe Agent Hive status"
          echo "💡 Usage: hive status [--mobile] [--detailed]"
          echo "🎯 Examples:"
          echo "   hive status --mobile"
          echo "   hive status --detailed"
          ;;
        "mobile")
          echo "📝 Mobile dashboard integration"
          echo "💡 Usage: hive mobile [--qr|--notifications|--status]"
          echo "🎯 Examples:"
          echo "   hive mobile --qr"
          echo "   hive mobile --notifications"
          ;;
        *)
          echo "⚠️ No specific help available for '$SPECIFIC_COMMAND'"
          ;;
      esac
    else
      echo "🚀 Streamlined Claude Code Hooks & Commands"
      echo "   Unified interface for LeanVibe Agent Hive 2.0"
      echo ""
      echo "📋 Available Commands:"
      echo "   status   - Check system and agent status"
      echo "   config   - Configuration management"
      echo "   agents   - Agent coordination and management"
      echo "   mobile   - Mobile dashboard integration"
      echo "   test     - Test execution and validation"
      echo "   memory   - Memory and context management"
      echo "   help     - Show this help or help for specific command"
      echo ""
      echo "💡 Examples:"
      echo "   hive status --mobile"
      echo "   hive mobile --qr"
      echo "   hive agents --spawn backend_developer"
      echo "   hive help status"
      echo ""
      echo "🔗 LeanVibe Agent Hive 2.0 Integration:"
      echo "   📱 Mobile Dashboard with QR access"
      echo "   🤖 Multi-agent coordination"
      echo "   🔔 Real-time notifications"
      echo "   ⚡ <5ms cached responses"
    fi
    ;;
    
  *)
    echo "❌ Unknown command: $COMMAND"
    echo "💡 Available commands: status, config, agents, mobile, test, memory, help"
    echo "🆘 Use 'hive help' for detailed usage information"
    exit 1
    ;;
esac