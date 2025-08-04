---
allowed-tools: Write(*), Read(*), Bash(python:*), Bash(git:*)
description: Intelligent sleep cycle with LeanVibe Agent Hive state preservation
argument-hint: [light|deep|emergency] [duration]
---

# 💤 LeanVibe Agent Hive - Intelligent Sleep Cycle

Consolidate current session context, preserve agent states, and prepare for wake cycle with intelligent memory management.

## Sleep Mode: $ARGUMENTS

### Available Sleep Types:
- **light**: Quick context consolidation (5-10 min sessions)
- **deep**: Full session consolidation with agent state preservation (default)
- **emergency**: Rapid state save for critical situations

## Current Session Context
- Active Agents: !`curl -s http://localhost:8000/api/agents/debug 2>/dev/null | jq -r '.agents | length' 2>/dev/null || echo "0"`
- System Status: !`curl -sf http://localhost:8000/health 2>/dev/null && echo "ONLINE" || echo "OFFLINE"`
- Git Status: !`git status --porcelain | wc -l | xargs -I {} echo "{} uncommitted changes"`

## Execution Strategy

### Phase 1: Agent State Preservation
Save current agent coordination state, task assignments, and active workflows to persistent memory.

### Phase 2: Context Consolidation
Based on sleep type:
- **Light**: Preserve active context, compress background information
- **Deep**: Full session summary with insights extraction
- **Emergency**: Critical state only, minimal processing

### Phase 3: System State Management
- Capture current LeanVibe system configuration
- Preserve mobile dashboard state and QR codes
- Save quality gate configurations and hook states
- Document any in-progress autonomous development tasks

### Phase 4: Memory Organization
- Update `.claude/memory/session_context.md` with current state
- Consolidate insights into `.claude/memory/reflection.md`
- Preserve next session priorities in `.claude/memory/next_tasks.md`
- Update project state in `.claude/memory/project_state.json`

### Phase 5: Wake Preparation
- Generate wake script with system restoration steps
- Prepare quick status check commands
- Create mobile access restoration links
- Set up autonomous development resumption pathway

## Sleep Cycle Outputs
1. **Session Summary**: Complete activity log and achievements
2. **Agent State**: Current coordination patterns and specializations
3. **System Configuration**: Infrastructure state and service status
4. **Wake Instructions**: Step-by-step restoration guide
5. **Priority Queue**: Next session immediate action items

Ready to enter sleep cycle with LeanVibe state preservation...