---
allowed-tools: Bash(docker:*), Bash(curl:*), Bash(tmux:*), Bash(lsof:*), Bash(ps:*), Bash(kill:*), Bash(npm:*), Bash(python:*), Bash(uvicorn:*), Read(*), Write(*)
description: LeanVibe Agent Hive 2.0 autonomous development platform control (project)
argument-hint: start | stop | status | demo | fix | mobile | logs
---

# 🚀 LeanVibe Agent Hive 2.0 Control Center

Execute LeanVibe operations with intelligent error recovery and mobile oversight.

## System Context Check
- API Status: !`timeout 5 curl -sf http://localhost:8000/health 2>/dev/null && echo "✅ ONLINE" || echo "❌ OFFLINE (checking...)"`
- Docker Services: !`docker ps --format "table {{.Names}}\t{{.Status}}" 2>/dev/null | grep -E "(postgres|redis)" | wc -l | xargs -I {} echo "{} services running"`
- Agent Count: !`timeout 3 curl -s http://localhost:8000/debug-agents 2>/dev/null | jq -r '.agent_count // 0' 2>/dev/null || echo "0"`
- System Load: !`uptime | awk '{print $10,$11,$12}' | sed 's/,//g'`

## Command: $ARGUMENTS

Based on your command, I'll execute the appropriate LeanVibe operation:

### Available Commands:
- **start**: Launch complete autonomous development platform with error recovery
- **stop**: Clean shutdown of all services and agents  
- **status**: Comprehensive system health check with actionable insights
- **demo**: Execute 60-second autonomous development demonstration
- **fix**: Intelligent system diagnosis and automated repair
- **mobile**: Generate mobile dashboard with QR code for remote oversight
- **logs**: Show real-time system logs and debugging information

### Execution Strategy:
1. **Detect current system state** and identify any issues
2. **Apply intelligent error recovery** for common problems
3. **Execute requested operation** with progress feedback
4. **Validate success** and provide next steps
5. **Generate mobile access** if requested

#### Intelligent Error Recovery System:
```bash
# Check if API is unresponsive
if ! timeout 10 curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "🔧 API unresponsive, attempting recovery..."
    
    # Kill hanging processes
    pkill -f "uvicorn.*app.main:app" || true
    
    # Wait for clean shutdown
    sleep 2
    
    # Restart API server
    python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 > api_server.log 2>&1 &
    
    # Wait for startup
    echo "🔄 Waiting for API startup..."
    for i in {1..30}; do
        if timeout 5 curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            echo "✅ API recovered successfully"
            break
        fi
        sleep 1
    done
fi
```

The system will automatically handle:
- Port conflicts and service issues
- Missing dependencies and configuration
- Database connectivity and migrations
- Agent spawning and coordination
- Real-time monitoring and alerts
- API server recovery and restart

Ready to execute: **$ARGUMENTS**