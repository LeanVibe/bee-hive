# 🚀 EXECUTE: Claude Code Context Compression Implementation

## 🎯 **IMMEDIATE MISSION**
Implement a production-ready `/compact` command for Claude Code that prevents context rot through intelligent conversation summarization. Use multi-agent coordination to avoid context overflow during implementation.

---

## 📋 **TASK DECOMPOSITION USING PROJECT INDEX SYSTEM**

**Step 1**: Use the Project Index system to analyze the Claude Code codebase and decompose this implementation task:

```
POST /api/project-index/{claude-code-project-id}/decompose-task
{
  "task_description": "implement context compression system with /compact command that summarizes conversations, compacts context, integrates with existing chat flow, handles edge cases, and provides seamless user experience",
  "task_type": "feature-implementation"
}
```

**Step 2**: Get context-optimized file discovery for implementation:

```
POST /api/project-index/{claude-code-project-id}/context  
{
  "task_description": "session management, message storage, command system, streaming, prompt management, CLI integration",
  "max_files": 15,
  "include_dependencies": true,
  "focus_languages": ["typescript", "javascript", "python"]
}
```

---

## 🔧 **CORE IMPLEMENTATION REQUIREMENTS**

### **Essential Components to Build**

1. **Summarization Pipeline** (`summarizeSession` function)
   ```typescript
   async function summarizeSession(sessionID: string, providerID: string, modelID: string): Promise<SummaryResult> {
     // Find last assistant summary in message history
     // Filter to messages since last summary  
     // Build Claude-optimized summary prompt
     // Stream summary with metadata.summary=true
     // Track costs/tokens and persist state
     // Emit UI progress events
   }
   ```

2. **Context Compaction Logic** (integrate into chat flow)
   ```typescript
   // Before sending requests: detect last summary, include only messages since
   // Exactly mirror the Session.summarize pattern you described
   ```

3. **Command Integration** (`/compact` command)
   ```typescript
   // Add to command completion list
   // Wire to summarizeSession function
   // Show progress/toast while streaming
   // Handle all edge cases gracefully
   ```

### **Required Edge Cases**
- **No active session**: Create one or no-op
- **Busy session**: Debounce or reject during generation
- **Multiple compactions**: Always trim to last summary
- **Insufficient context**: Validate minimum message threshold
- **API failures**: Graceful degradation and user feedback

### **Integration Requirements**
- **CLI Mode**: Direct function calls in runtime state
- **Server Mode**: `POST /session/:id/summarize` endpoint
- **Prompt Management**: Create `summarize.txt` for Claude models
- **UI Events**: Progress indication and completion feedback

---

## 🤖 **AGENT DELEGATION STRATEGY**

### **Phase 1: Architecture Analysis (Architecture Agent)**
**Context Size**: ~30K tokens | **Duration**: 1-2 hours

**Tasks**:
1. Map Claude Code's current session management and message storage
2. Identify command registration patterns and completion system
3. Analyze streaming infrastructure and event emission patterns
4. Document API architecture (CLI vs server deployment)
5. Create integration specification document

**Deliverables**: 
- Session management analysis
- Command system integration plan  
- API architecture decision matrix
- Implementation roadmap

### **Phase 2: Core Engine (Backend Agent)**
**Context Size**: ~40K tokens | **Duration**: 2-3 hours

**Tasks**:
1. Implement message filtering and summary detection logic
2. Build summarization engine with Claude model integration
3. Create context compaction and compression tracking
4. Implement persistence and state management
5. Add comprehensive error handling

**Deliverables**:
- `summarizeSession` function implementation
- Context compaction logic for chat flow
- Message processing and filtering utilities
- Error handling and validation

### **Phase 3: Command & UI (Frontend Agent)**  
**Context Size**: ~25K tokens | **Duration**: 1-2 hours

**Tasks**:
1. Implement `/compact` command and registration
2. Add command completion and help text
3. Create progress indication and user feedback
4. Wire UI events and toast notifications
5. Handle user interaction edge cases

**Deliverables**:
- `/compact` command implementation
- UI progress and feedback system
- Command help and completion integration
- User experience optimization

### **Phase 4: Quality & Testing (QA Agent)**
**Context Size**: ~35K tokens | **Duration**: 2-3 hours

**Tasks**:
1. Implement all edge case handlers
2. Add concurrency control and debouncing
3. Create comprehensive test suite
4. Add performance monitoring and metrics
5. Validate production readiness

**Deliverables**:
- Edge case handling implementation
- Concurrency and state consistency
- Test coverage and validation
- Performance monitoring

---

## 📝 **PROMPT ENGINEERING SPECIFICATION**

### **Summarize.txt Prompt Template**
```
You are summarizing a conversation to preserve context while reducing token usage.

<conversation>
{messages_since_last_summary}
</conversation>

Create a concise but comprehensive summary that:
1. Preserves key decisions and outcomes
2. Retains important technical details and context
3. Maintains conversation flow and user intent
4. Enables seamless conversation continuation

Focus on:
- Main topics and themes discussed
- Key decisions made and rationale
- Technical implementations or solutions chosen
- User goals and current progress
- Any unresolved questions or next steps

Format as a natural continuation of the conversation that preserves essential context for future interactions.
```

### **Model Selection Logic**
- **Claude-3.5-Sonnet**: Primary choice for balanced speed/quality
- **Claude-3-Opus**: For complex conversations requiring deeper understanding
- **Fallback handling**: Graceful degradation if preferred model unavailable

---

## ⚡ **IMPLEMENTATION EXECUTION PROTOCOL**

### **Coordination Checkpoints**
1. **Architecture Review** (End of Phase 1): Validate integration approach
2. **Core Implementation Review** (End of Phase 2): Test summarization pipeline  
3. **UI Integration Review** (End of Phase 3): Validate user experience
4. **Final Validation** (End of Phase 4): Production readiness check

### **Context Rot Prevention**
- **Agent Context Limits**: 50K tokens max per agent
- **Refresh Triggers**: Automatic refresh at 40K tokens
- **Handoff Protocol**: Clean interface definitions between agents
- **Integration Testing**: Continuous validation of component interactions

### **Quality Gates**
- **Functional**: All core requirements implemented and tested
- **Performance**: <2s summarization, <200ms context retrieval
- **UX**: Intuitive command with clear feedback
- **Edge Cases**: All scenarios handled gracefully

---

## 🎯 **SUCCESS VALIDATION**

### **Acceptance Criteria**
```bash
# Test the complete implementation
/compact  # Should compress context and show progress
# Continue conversation - should only use context since summary
# Multiple /compact commands should work correctly
# Edge cases should be handled gracefully
```

### **Performance Validation**
- ✅ Summarization completes in <2 seconds
- ✅ Context compression achieves 60-80% reduction
- ✅ No impact on normal chat flow performance
- ✅ Memory usage remains stable

### **Integration Validation**  
- ✅ Works in both CLI and server modes
- ✅ Integrates seamlessly with existing commands
- ✅ Maintains backward compatibility
- ✅ Provides clear user feedback

---

## 🚀 **EXECUTE NOW**

**Start with Project Index Analysis**:
1. Use Project Index to decompose this task intelligently
2. Get context-optimized file discovery for Claude Code codebase  
3. Begin with Architecture Agent for system analysis
4. Deploy specialized agents based on Project Index recommendations
5. Use context monitoring to prevent agent context rot during implementation

**Expected Timeline**: 6-10 hours total with 4 specialized agents working in coordinated phases.

**Outcome**: Production-ready `/compact` command that makes Claude Code conversations infinitely extensible through intelligent context compression.

---

*This prompt leverages the Project Index Agent Delegation System we just built to ensure efficient, context-rot-free implementation of this critical feature.*