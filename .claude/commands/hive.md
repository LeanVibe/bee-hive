---
allowed-tools: Bash(docker:*), Bash(curl:*), Bash(tmux:*), Bash(lsof:*), Bash(ps:*), Bash(kill:*), Bash(npm:*), Bash(python:*), Bash(uvicorn:*), Read(*), Write(*)
description: LeanVibe Agent Hive 2.0 autonomous development platform control (project)
argument-hint: start | stop | status | demo | fix | mobile | logs
---

# 🚀 LeanVibe Agent Hive 2.0 Control Center

Execute LeanVibe operations with intelligent error recovery and mobile oversight.

## System Context Check
- API Status: !`curl -sf http://localhost:8000/health 2>/dev/null && echo "✅ ONLINE" || echo "❌ OFFLINE"`
- Docker Services: !`docker ps --format "table {{.Names}}\t{{.Status}}" 2>/dev/null | grep -E "(postgres|redis)" | wc -l | xargs -I {} echo "{} services running"`
- Agent Count: !`curl -s http://localhost:8000/api/agents/debug 2>/dev/null | jq -r '.agents | length' 2>/dev/null || echo "0"`
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

The system will automatically handle:
- Port conflicts and service issues
- Missing dependencies and configuration
- Database connectivity and migrations
- Agent spawning and coordination
- Real-time monitoring and alerts

Ready to execute: **$ARGUMENTS**