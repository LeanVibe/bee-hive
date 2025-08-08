# 🎯 Streamlined Claude Code Hooks & Commands Implementation Plan
## LeanVibe Agent Hive 2.0 Integration Strategy

**Created**: August 5, 2025  
**Status**: Ready for Implementation  
**Priority**: P1 - Developer Experience Critical  

---

## Executive Summary

**Problem**: Current Claude Code hooks and commands system is overly complex with 8+ hook events, scattered configuration files, and fragmented command interfaces creating cognitive overload.

**Solution**: Streamline to 3 essential hooks, unify commands under enhanced `/hive` system, and integrate with mobile-first developer experience.

**Impact**: 70% reduction in configuration complexity, unified command interface, seamless integration with LeanVibe Agent Hive 2.0.

---

## 🔍 Current State Analysis

### Complexity Issues Identified
- **8 Hook Events**: PreToolUse, PostToolUse, Notification, UserPromptSubmit, Stop, SubagentStop, PreCompact, SessionStart
- **3 Configuration Locations**: ~/.claude/settings.json, .claude/settings.json, .claude/settings.local.json  
- **4 Command Types**: Built-in slash commands, custom slash commands, MCP commands, subagent commands
- **Complex JSON Schemas**: Requiring deep technical knowledge for basic customization
- **Fragmented Documentation**: Spread across multiple reference files

### Integration Gaps with LeanVibe Agent Hive 2.0
- **No Integration** with mobile decision interface
- **No Connection** to agent orchestration system
- **Missing Hooks** for autonomous development workflows
- **Scattered Commands** not aligned with `/hive` system

---

## 🎯 Streamlined Architecture Design

### Core Philosophy: Essential Hooks Only
**Focus on 3 critical hook events that deliver 80% of value:**

#### 1. **Quality Gate Hook** (Replaces: PreToolUse, PostToolUse, Stop)
```json
{
  "hooks": {
    "QualityGate": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/quality-gate.sh"
          }
        ]
      }
    ]
  }
}
```

**Purpose**: Single validation point for all code changes and tool usage  
**Benefits**: Unified quality control, simplified configuration, comprehensive validation

#### 2. **Session Lifecycle Hook** (Replaces: SessionStart, PreCompact, Notification)
```json
{
  "hooks": {
    "SessionLifecycle": [
      {
        "matcher": "startup|resume|compact|notification",
        "hooks": [
          {
            "type": "command", 
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/session-manager.sh"
          }
        ]
      }
    ]
  }
}
```

**Purpose**: Manage session state, context, and developer notifications  
**Benefits**: Centralized session management, intelligent notification filtering

#### 3. **Agent Coordination Hook** (New - LeanVibe Integration)
```json
{
  "hooks": {
    "AgentCoordination": [
      {
        "matcher": "Task|SubagentStop",
        "hooks": [
          {
            "type": "command",
            "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/agent-coordinator.sh"
          }
        ]
      }
    ]
  }
}
```

**Purpose**: Integrate with LeanVibe Agent Hive 2.0 multi-agent system  
**Benefits**: Seamless agent orchestration, mobile notifications, autonomous coordination

### Unified Command Interface: Enhanced `/hive` System

#### Core `/hive` Commands (Streamlined)
```bash
# Essential system control
/hive status           # Intelligent system overview (replaces /status, /cost, /doctor)
/hive config           # Unified configuration management (replaces /config, /permissions)
/hive agents           # Agent management (replaces /agents, integrates subagents)
/hive mobile           # Mobile dashboard control (new - mobile-first approach)

# Developer workflow commands  
/hive review [scope]   # Code review with quality gates (replaces /review)
/hive test [pattern]   # Intelligent testing workflows (new)
/hive deploy [env]     # Deployment coordination (new)
/hive fix [issue]      # Automated issue resolution (new)

# Session management
/hive memory           # Memory management (replaces /memory, /compact, /clear)
/hive help [topic]     # Context-aware help (replaces /help)
```

#### Custom Command Integration
```bash
# Project commands inherit /hive namespace
/hive:optimize        # From .claude/commands/optimize.md
/hive:security-review # From .claude/commands/security-review.md

# User commands available globally
/hive:personal-workflow # From ~/.claude/commands/personal-workflow.md
```

#### MCP Command Integration
```bash
# MCP commands under /hive namespace
/hive:github:pr-review    # From mcp__github__pr_review
/hive:jira:create-issue   # From mcp__jira__create_issue
```

---

## 📱 Mobile-First Integration Strategy

### Mobile Notification System
Enhanced hooks that integrate with iPhone 14+ mobile dashboard:

```bash
# agent-coordinator.sh integration
#!/bin/bash
HOOK_DATA=$(cat)
PRIORITY=$(echo "$HOOK_DATA" | jq -r '.priority // "medium"')

case $PRIORITY in
  "critical"|"high")
    # Send push notification to mobile dashboard
    curl -X POST http://localhost:8000/api/mobile/notifications \
      -H "Content-Type: application/json" \
      -d "{\"priority\": \"$PRIORITY\", \"data\": $(echo \"$HOOK_DATA\" | jq -c .)}"
    ;;
  *)
    # Log for mobile dashboard polling
    echo "$HOOK_DATA" >> ~/.claude/mobile-events.jsonl
    ;;
esac
```

### Mobile Command Interface
```typescript
// Mobile dashboard integration
class MobileCommandInterface {
  async executeHiveCommand(command: string, args?: string[]): Promise<void> {
    const response = await fetch('/api/hive/execute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command, args })
    });
    
    if (!response.ok) {
      this.showError(`Command failed: ${command}`);
      return;
    }
    
    const result = await response.json();
    this.updateDashboard(result);
  }
}
```

---

## 🔧 Implementation Strategy

### Phase 1: Hook Consolidation (2 hours)

#### Step 1: Create Unified Hook Scripts
```bash
# Create consolidated hook directory
mkdir -p .claude/hooks

# Quality gate script (replaces multiple PreToolUse/PostToolUse hooks)
cat > .claude/hooks/quality-gate.sh << 'EOF'
#!/bin/bash
HOOK_DATA=$(cat)
TOOL_NAME=$(echo "$HOOK_DATA" | jq -r '.tool_name // ""')
EVENT=$(echo "$HOOK_DATA" | jq -r '.hook_event_name // ""')

case "$EVENT:$TOOL_NAME" in
  "PreToolUse:Bash")
    # Validate bash commands
    python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/bash-validator.py"
    ;;
  "PostToolUse:Edit"|"PostToolUse:Write")
    # Format code after edits
    python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/code-formatter.py"
    ;;
  "PostToolUse:*")
    # Run tests after significant changes
    python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/test-runner.py"
    ;;
esac
EOF

chmod +x .claude/hooks/quality-gate.sh
```

#### Step 2: Session Lifecycle Management
```bash
# Session manager script
cat > .claude/hooks/session-manager.sh << 'EOF'
#!/bin/bash
HOOK_DATA=$(cat)
EVENT=$(echo "$HOOK_DATA" | jq -r '.hook_event_name // ""')
SOURCE=$(echo "$HOOK_DATA" | jq -r '.source // ""')

case "$EVENT:$SOURCE" in
  "SessionStart:startup")
    # Initialize LeanVibe Agent Hive connection
    curl -s http://localhost:8000/api/claude/session-start
    echo "LeanVibe Agent Hive 2.0 active - Enhanced mobile oversight available"
    ;;
  "SessionStart:resume")
    # Restore agent states and mobile dashboard
    curl -s http://localhost:8000/api/claude/session-resume
    ;;
  "Notification:*")
    # Filter and route notifications to mobile
    python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/notification-filter.py"
    ;;
esac
EOF

chmod +x .claude/hooks/session-manager.sh
```

#### Step 3: Agent Coordination Integration
```bash
# Agent coordinator script  
cat > .claude/hooks/agent-coordinator.sh << 'EOF'
#!/bin/bash
HOOK_DATA=$(cat)
EVENT=$(echo "$HOOK_DATA" | jq -r '.hook_event_name // ""')

case "$EVENT" in
  "Task")
    # Coordinate with LeanVibe Agent Hive
    curl -X POST http://localhost:8000/api/agents/coordinate \
      -H "Content-Type: application/json" \
      -d "$HOOK_DATA"
    ;;
  "SubagentStop")
    # Update mobile dashboard with agent completion
    curl -X POST http://localhost:8000/api/mobile/agent-update \
      -H "Content-Type: application/json" \
      -d "$HOOK_DATA"
    ;;
esac
EOF

chmod +x .claude/hooks/agent-coordinator.sh
```

### Phase 2: Command Unification (3 hours)

#### Step 1: Enhanced `/hive` Command Implementation
```bash
# Create unified hive command processor
mkdir -p .claude/commands/hive

# Status command with intelligence
cat > .claude/commands/hive/status.md << 'EOF'
---
description: Intelligent system overview with mobile integration
allowed-tools: Bash, WebFetch
---

## System Status Analysis

Get comprehensive system status integrating:
- LeanVibe Agent Hive 2.0 health: !`curl -s http://localhost:8000/health`
- Active agents: !`curl -s http://localhost:8000/api/agents/status`
- Mobile dashboard status: !`curl -s http://localhost:8000/api/mobile/status`
- Recent activity: !`tail -n 5 ~/.claude/activity.log`

Provide intelligent summary with:
1. **Critical alerts** requiring immediate attention
2. **Agent coordination** status and performance
3. **Mobile oversight** connectivity and alerts
4. **System recommendations** for optimization

Format for both terminal and mobile consumption.
EOF
```

#### Step 2: Mobile Command Integration
```bash
# Mobile dashboard command
cat > .claude/commands/hive/mobile.md << 'EOF'
---
description: Mobile dashboard control and QR code generation
allowed-tools: Bash, Write
---

## Mobile Dashboard Control

Generate mobile access for iPhone 14+ oversight:

1. **Check mobile dashboard status**: !`curl -s http://localhost:8000/api/mobile/status`
2. **Generate QR code** for quick mobile access
3. **Test push notifications** to mobile device
4. **Display mobile-optimized** system summary

Commands:
- Display current mobile access URL with QR code
- Test mobile notification system
- Show mobile-friendly status summary
- Configure mobile notification preferences

Integration with LeanVibe Agent Hive 2.0 mobile PWA dashboard.
EOF
```

#### Step 3: Intelligent Command Routing
```bash
# Command router for /hive namespace
cat > .claude/commands/hive.md << 'EOF'
---
description: Unified LeanVibe Agent Hive 2.0 command interface
allowed-tools: Bash, Read, WebFetch
argument-hint: status | config | agents | mobile | review [scope] | test [pattern] | deploy [env] | fix [issue] | memory | help [topic]
---

## LeanVibe Agent Hive 2.0 Command Interface

Unified command system integrating Claude Code with autonomous development platform.

**Available Commands**:
- `status` - Intelligent system overview with mobile integration
- `config` - Unified configuration management  
- `agents` - Agent management and coordination
- `mobile` - Mobile dashboard control and QR generation
- `review [scope]` - Code review with quality gates
- `test [pattern]` - Intelligent testing workflows
- `deploy [env]` - Deployment coordination
- `fix [issue]` - Automated issue resolution
- `memory` - Memory and session management
- `help [topic]` - Context-aware help system

**Arguments**: $ARGUMENTS

Execute the specified command with LeanVibe Agent Hive 2.0 integration.
EOF
```

### Phase 3: Configuration Simplification (1 hour)

#### Streamlined Settings Structure
```json
{
  "hooks": {
    "QualityGate": [
      {
        "matcher": "*",
        "hooks": [{"type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/quality-gate.sh"}]
      }
    ],
    "SessionLifecycle": [
      {
        "matcher": "startup|resume|compact|notification",
        "hooks": [{"type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/session-manager.sh"}]
      }
    ],
    "AgentCoordination": [
      {
        "matcher": "Task|SubagentStop", 
        "hooks": [{"type": "command", "command": "$CLAUDE_PROJECT_DIR/.claude/hooks/agent-coordinator.sh"}]
      }
    ]
  },
  "leanvibe": {
    "api_url": "http://localhost:8000",
    "mobile_dashboard": true,
    "agent_coordination": true,
    "quality_gates": true
  }
}
```

---

## 📊 Benefits Analysis

### Complexity Reduction
| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| Hook Events | 8 events | 3 events | 63% reduction |
| Configuration Files | 3 locations | 1 location | 67% reduction |
| Command Types | 4 systems | 1 unified system | 75% reduction |
| Setup Time | 30+ minutes | 5 minutes | 83% reduction |

### Developer Experience Enhancement
- **Unified Interface**: Single `/hive` command for all operations
- **Mobile Integration**: Real-time oversight from iPhone 14+
- **Intelligent Defaults**: Working configuration out-of-the-box
- **Context-Aware Help**: Commands adapt to current system state

### LeanVibe Agent Hive 2.0 Integration
- **Seamless Coordination**: Direct integration with multi-agent system
- **Mobile-First Approach**: Aligns with enhanced mobile decision interface
- **Autonomous Development**: Hooks support autonomous workflows
- **Quality Gates**: Integrated validation with agent orchestration

---

## 🚀 Implementation Timeline

### Day 1: Hook Consolidation (2 hours)
- Create unified hook scripts
- Test quality gate integration
- Validate session lifecycle management

### Day 2: Command Unification (3 hours)  
- Implement enhanced `/hive` commands
- Create mobile dashboard integration
- Test command routing and execution

### Day 3: Integration Testing (1 hour)
- End-to-end workflow validation
- Mobile dashboard connectivity testing
- Documentation and deployment

**Total Implementation**: 6 hours for complete transformation

---

## 🎯 Success Metrics

### Quantitative Targets
- **Configuration Complexity**: 70% reduction in lines of configuration
- **Setup Time**: 83% reduction (30min → 5min)
- **Command Discoverability**: 100% of commands under `/hive` namespace
- **Mobile Integration**: Real-time updates <50ms

### Qualitative Improvements
- **Developer Cognitive Load**: Significantly reduced through unification
- **System Reliability**: Enhanced through consolidated quality gates
- **Mobile Oversight**: Strategic decision-making from anywhere
- **Agent Coordination**: Seamless autonomous development workflows

---

## 🔮 Future Enhancements

### Phase 4: AI-Powered Intelligence (Future)
- **Smart Hook Triggering**: ML-based hook execution optimization
- **Predictive Commands**: Suggest commands based on context and history
- **Adaptive Configuration**: Self-tuning hooks based on usage patterns

### Phase 5: Advanced Mobile Features (Future)
- **Voice Commands**: "Hey Claude, run hive status"  
- **Gesture Controls**: Advanced swipe patterns for complex operations
- **AR Overlay**: Code visualization and debugging on mobile

---

## 🎉 Implementation Ready

**Status**: ✅ **READY FOR IMMEDIATE IMPLEMENTATION**

**Next Steps**:
1. **Execute Phase 1**: Hook consolidation and testing
2. **Deploy Phase 2**: Command unification with mobile integration  
3. **Validate Phase 3**: End-to-end testing and documentation
4. **Production Ready**: Streamlined Claude Code hooks and commands

**The streamlined system transforms Claude Code from complex configuration to intuitive, mobile-integrated developer experience aligned with LeanVibe Agent Hive 2.0! 🚀**

---

*Streamlined Claude Code Hooks & Commands Plan - From Complexity to Simplicity with Mobile-First Integration*