---
description: Intelligent session consolidation with context preservation
allowed-tools: TodoWrite(*), Bash(git status:*), Bash(git add:*), Bash(git commit:*)
---

# Intelligent Session Consolidation

Consolidate current session state with comprehensive context preservation:

## Current Session Assessment
!`git status --porcelain`
!`git diff --name-only HEAD`

## Consolidation Process

1. **Capture Session State**:
   - Document current progress and achievements
   - Preserve active task states and priorities
   - Record any unresolved issues or blockers

2. **Validate Quality Gates**:
   - Ensure all modified files compile successfully
   - Verify no broken imports or syntax errors
   - Check for any uncommitted critical changes

3. **Memory Optimization**:
   - Consolidate key insights and learnings
   - Preserve context for seamless session resume
   - Update session summary with progress made

4. **Context Preparation**:
   - Prepare consolidated memory files
   - Document next session priorities
   - Create handoff notes for continuation

## Auto-Commit Changes (if stable)
If changes are stable and tests pass:
- Add all relevant changes to git staging
- Create descriptive commit message with session summary
- Commit with Claude Code signature

## Session Summary Output
Provide concise summary of:
- ✅ **Completed tasks** and achievements  
- 🔄 **In-progress work** and current state
- ⏳ **Next priorities** for resume session
- ⚠️ **Issues/blockers** requiring attention
- 📊 **Quality metrics** and system health

Ready to transition to sleep mode or continue with optimized context.