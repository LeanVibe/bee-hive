#!/bin/bash
# LeanVibe Agent Hive 2.0 - Session Summary Hook
# Automatically triggered when Claude Code session ends

set -euo pipefail

# Read hook input from stdin  
input_data=$(cat)

echo "📊 LeanVibe Session Summary"
echo "=========================="

# Check if we're in the bee-hive project
if [[ "$PWD" =~ bee-hive ]]; then
    echo "🚀 Project: LeanVibe Agent Hive 2.0"
    
    # Quick system status
    echo ""
    echo "🔍 System Status:"
    
    # Check API status
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        echo "   ✅ API Server: Online"
        
        # Get agent count
        agent_count=$(curl -s http://localhost:8000/api/agents/debug 2>/dev/null | jq -r '.agents | length' 2>/dev/null || echo "0")
        echo "   🤖 Active Agents: $agent_count"
    else
        echo "   ❌ API Server: Offline"
    fi
    
    # Check Docker services
    if docker ps --format "{{.Names}}" 2>/dev/null | grep -q postgres; then
        echo "   ✅ Database: Running"
    else
        echo "   ❌ Database: Not running"
    fi
    
    if docker ps --format "{{.Names}}" 2>/dev/null | grep -q redis; then
        echo "   ✅ Redis: Running"
    else
        echo "   ❌ Redis: Not running"
    fi
    
    # Session recommendations  
    echo ""
    echo "💡 Quick Actions:"
    echo "   • Start system: /hive start"
    echo "   • Check status: /hive status"
    echo "   • Run demo: /hive demo"
    echo "   • Mobile access: /hive mobile"
    echo "   • Fix issues: /hive fix"
    
    # Check for common issues
    if ! curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        echo ""
        echo "🔧 System appears to be down. Consider running:"
        echo "   /hive fix    # Automatic recovery"
        echo "   /hive start  # Full system startup"
    fi
    
else
    echo "ℹ️  Not in LeanVibe project directory"
fi

echo ""
echo "🎯 Ready for autonomous development!"
exit 0