---
allowed-tools: Read(*), Bash(python:*), Bash(curl:*), Bash(docker:*)
description: Intelligent wake cycle with LeanVibe Agent Hive state restoration  
argument-hint: [quick|full|validate] [system-check]
---

# 🌅 LeanVibe Agent Hive - Intelligent Wake Cycle

Restore session context, reactivate agent states, and resume autonomous development operations with full system validation.

## Wake Mode: $ARGUMENTS

### Available Wake Types:
- **quick**: Fast context restoration (2-3 min)
- **full**: Complete system restoration with validation (default)
- **validate**: System health check with recommendations

## Memory Restoration
- Previous Session: !`cat /Users/bogdan/.claude/memory/session_context.md | head -3 | tail -1 | sed 's/^## //' || echo "No previous session found"`
- Project State: !`cat /Users/bogdan/.claude/memory/project_state.json | jq -r '.session_metrics.completion_rate // "Unknown"' 2>/dev/null || echo "Unknown"`
- Next Tasks: !`cat /Users/bogdan/.claude/memory/next_tasks.md | grep -c "###" || echo "0"` priority tasks

## System Status Check
- API Health: !`curl -sf http://localhost:8000/health 2>/dev/null && echo "✅ ONLINE" || echo "❌ OFFLINE"`
- Docker Services: !`docker ps --format "{{.Names}}" 2>/dev/null | grep -E "(postgres|redis)" | wc -l | xargs -I {} echo "{}/2 services running"`
- Active Agents: !`curl -s http://localhost:8000/api/agents/debug 2>/dev/null | jq -r '.agents | length' 2>/dev/null || echo "0"`

## Execution Strategy

### Phase 1: Memory Context Restoration
Restore comprehensive session context from memory files:
- Load previous session summary and achievements
- Restore agent coordination patterns and specializations  
- Recover autonomous development task states
- Reestablish mobile oversight configurations

### Phase 2: System State Validation
Verify LeanVibe Agent Hive operational status:
- Health check all critical services (API, Database, Redis)
- Validate agent system availability and responsiveness
- Confirm mobile dashboard accessibility
- Test quality gate hooks and automation

### Phase 3: Intelligent Recovery (if needed)
Auto-execute recovery procedures for any detected issues:
- Port conflict resolution
- Service restart automation  
- Database connectivity restoration
- Agent team reactivation

### Phase 4: Context Integration
Merge previous session insights with current environment:
- Update project priorities based on system changes
- Integrate any external changes since last session
- Refresh mobile access links and QR codes
- Validate autonomous development capabilities

### Phase 5: Session Initialization
Prepare for immediate productivity:
- Display high-priority action items
- Show system status dashboard
- Generate mobile oversight links
- Confirm autonomous development readiness

## Wake Cycle Outputs
1. **System Status**: Complete health check with recommendations
2. **Priority Actions**: Immediate tasks from previous session
3. **Agent Readiness**: Current coordination capabilities
4. **Mobile Access**: Updated QR codes and dashboard links
5. **Development Status**: Autonomous capabilities validation

## Quick Actions Available
- **System Startup**: `/hive start` for full system activation
- **Health Check**: `/hive status` for comprehensive diagnostics  
- **Mobile Access**: `/hive mobile` for QR code generation
- **Demo Ready**: `/hive demo` for 60-second autonomous development proof
- **Auto Recovery**: `/hive fix` for intelligent error resolution

Ready to resume autonomous development operations...