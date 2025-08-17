# Claude Code Context Compression Implementation - Master Planning Prompt

## 🎯 **Mission: Implement Production-Ready Context Compression System**

You are Claude Code, tasked with implementing a sophisticated context compression system that prevents context rot through intelligent summarization and compaction. This is a **critical feature** for maintaining agent efficiency in long conversations.

---

## 📋 **Context & Requirements Analysis**

### **Current State Assessment Needed**
Before implementation, perform comprehensive analysis:

1. **Claude Code Architecture Discovery**
   - Map current session management architecture
   - Identify existing message storage and retrieval patterns  
   - Analyze current context handling and token management
   - Document API structure (CLI vs local server vs hybrid)
   - Assess existing streaming and event emission patterns

2. **Integration Points Analysis**
   - Identify where summarization hooks into existing chat flow
   - Map current command system and completion infrastructure
   - Analyze message metadata schema and extension points
   - Document current UI event system and progress indicators

3. **Technical Dependencies Assessment**
   - Catalog existing prompt management system
   - Identify current model routing and provider abstraction
   - Assess existing cost/token tracking infrastructure
   - Map current persistence and state management patterns

### **Target Implementation Specification**

**Core Functionality Requirements:**
```typescript
interface SummarizationSystem {
  // Core summarization pipeline
  summarizeSession(sessionID: string, providerID: string, modelID: string): Promise<SummaryResult>
  
  // Context compaction behavior  
  getCompactedContext(sessionID: string): Promise<Message[]>
  
  // API integration
  handleCompactCommand(): Promise<void>
  
  // Edge case handling
  validateSummarizationPreconditions(sessionID: string): SummarizationStatus
}

interface SummaryResult {
  summaryMessage: Message
  tokensSaved: number
  compressionRatio: number
  persistenceStatus: PersistenceResult
}
```

---

## 🏗️ **Implementation Architecture Plan**

### **Phase 1: Foundation Analysis & Design (Sub-Agent: Architecture Specialist)**

**Task**: Comprehensive codebase analysis and integration design

**Specific Actions**:
1. **Session Management Analysis**
   - Map current session state management patterns
   - Identify message storage schema and indexing
   - Document existing metadata extension points
   - Analyze current streaming message infrastructure

2. **Command System Integration Design** 
   - Document current command registration and completion patterns
   - Design `/compact` command integration with existing CLI
   - Plan progress indication and user feedback systems
   - Design error handling and validation integration

3. **API Architecture Decision**
   - Analyze current Claude Code deployment patterns (CLI vs server)
   - Design summarization API shape for both deployment modes
   - Plan HTTP route integration if server mode exists
   - Design direct function call patterns for CLI mode

**Deliverables**:
- Comprehensive architecture analysis document
- Integration design specification
- API design for both CLI and server modes
- Command system integration plan

### **Phase 2: Core Summarization Engine (Sub-Agent: Backend Engineer)**

**Task**: Implement the core summarization pipeline and logic

**Specific Actions**:
1. **Message Processing Pipeline**
   ```typescript
   class MessageProcessor {
     findLastSummary(messages: Message[]): Message | null
     filterMessagesSinceLastSummary(messages: Message[], lastSummary?: Message): Message[]
     validateSummarizationCandidates(messages: Message[]): ValidationResult
   }
   ```

2. **Summarization Engine**
   ```typescript
   class SummarizationEngine {
     buildSummaryPrompt(messages: Message[]): PromptRequest
     streamSummaryGeneration(prompt: PromptRequest, sessionID: string): AsyncGenerator<SummaryChunk>
     persistSummaryMessage(summary: Message, sessionID: string): Promise<PersistenceResult>
   }
   ```

3. **Context Compaction Logic**
   ```typescript
   class ContextCompactor {
     getCompactedContext(sessionID: string): Promise<Message[]>
     calculateCompressionMetrics(original: Message[], compacted: Message[]): CompressionMetrics
     updateSessionContextState(sessionID: string, newContext: Message[]): Promise<void>
   }
   ```

**Deliverables**:
- Core summarization engine implementation
- Message processing and filtering logic
- Context compaction and compression tracking
- Comprehensive error handling and validation

### **Phase 3: Prompt Engineering & Model Integration (Sub-Agent: AI Specialist)**

**Task**: Design optimal prompts and model integration for summarization

**Specific Actions**:
1. **Summarization Prompt Design**
   ```
   Create `summarize.txt` prompt optimized for Claude models:
   - Conversation context preservation
   - Key decision and outcome capture
   - Technical detail retention
   - Concise but comprehensive format
   - Token-efficient structure
   ```

2. **Model Provider Integration**
   ```typescript
   class SummarizationModelAdapter {
     selectOptimalModel(context: SummarizationContext): ModelSelection
     optimizePromptForProvider(prompt: string, provider: string): OptimizedPrompt
     handleProviderSpecificStreaming(provider: string): StreamingAdapter
   }
   ```

3. **Quality Assurance System**
   ```typescript
   class SummaryQualityAssessment {
     validateSummaryQuality(original: Message[], summary: string): QualityScore
     detectInformationLoss(context: Message[], summary: string): LossAnalysis
     recommendSummaryImprovements(summary: string): Recommendation[]
   }
   ```

**Deliverables**:
- Optimized summarization prompts for different Claude models
- Model-specific integration adapters
- Summary quality assessment and validation
- Prompt versioning and A/B testing framework

### **Phase 4: API & Command Integration (Sub-Agent: Frontend Engineer)**

**Task**: Implement user-facing interfaces and command integration

**Specific Actions**:
1. **Command System Integration**
   ```typescript
   class CompactCommand {
     registerCommandCompletion(): void
     validateSessionState(): CommandValidationResult
     executeWithProgressIndicator(): Promise<CompactionResult>
     handleCompactionErrors(error: Error): UserFeedback
   }
   ```

2. **HTTP API Integration** (if applicable)
   ```typescript
   // POST /session/:id/summarize
   async function handleSummarizeRoute(
     sessionId: string, 
     options: SummarizationOptions
   ): Promise<APIResponse<SummaryResult>>
   ```

3. **UI Event System Integration**
   ```typescript
   class UIEventEmitter {
     emitCompactionStarted(sessionID: string): void
     emitCompactionProgress(progress: CompactionProgress): void
     emitCompactionComplete(result: SummaryResult): void
     emitCompactionError(error: CompactionError): void
   }
   ```

**Deliverables**:
- `/compact` command implementation and registration
- HTTP API routes (if server mode)
- UI progress indication and feedback systems
- Command completion and help integration

### **Phase 5: Edge Cases & Production Hardening (Sub-Agent: QA Specialist)**

**Task**: Comprehensive edge case handling and production readiness

**Specific Actions**:
1. **Edge Case Implementation**
   ```typescript
   class EdgeCaseHandler {
     handleNoActiveSession(): CompactionResult
     handleBusySession(): CompactionResult  
     handleMultipleCompactions(): CompactionResult
     handleCorruptedSummaries(): RecoveryResult
     handleInsufficientContext(): ValidationResult
   }
   ```

2. **Concurrency & State Management**
   ```typescript
   class ConcurrencyManager {
     preventConcurrentSummarization(sessionID: string): boolean
     debounceSummarizationRequests(sessionID: string): Promise<void>
     handleStateConsistency(): ConsistencyResult
   }
   ```

3. **Performance & Monitoring**
   ```typescript
   class PerformanceMonitor {
     trackSummarizationLatency(): LatencyMetrics
     monitorCompressionEfficiency(): CompressionMetrics
     detectSummarizationFailures(): FailureAnalysis
     optimizeContextRetrieval(): OptimizationResult
   }
   ```

**Deliverables**:
- Comprehensive edge case handling
- Concurrency control and state consistency
- Performance monitoring and optimization
- Error recovery and graceful degradation

---

## 🎛️ **Sub-Agent Coordination Strategy**

### **Agent Specialization & Task Assignment**

**Architecture Specialist Agent**:
- **Focus**: System analysis and integration design
- **Context**: Current Claude Code architecture patterns
- **Duration**: 2-3 hours
- **Deliverables**: Architecture analysis, integration specs

**Backend Engineer Agent**:
- **Focus**: Core engine implementation and data flow
- **Context**: Message processing, summarization logic, persistence
- **Duration**: 4-6 hours  
- **Deliverables**: Core pipeline implementation

**AI Specialist Agent**:
- **Focus**: Prompt engineering and model optimization
- **Context**: Claude model capabilities, prompt optimization
- **Duration**: 2-3 hours
- **Deliverables**: Optimized prompts, model integration

**Frontend Engineer Agent**:
- **Focus**: User interface and command integration
- **Context**: CLI patterns, UI events, user experience
- **Duration**: 3-4 hours
- **Deliverables**: Command implementation, UI integration

**QA Specialist Agent**:
- **Focus**: Edge cases, testing, production hardening
- **Context**: Error scenarios, performance, reliability
- **Duration**: 3-4 hours
- **Deliverables**: Edge case handling, monitoring, testing

### **Coordination Protocol**

1. **Sequential Dependencies**:
   - Architecture analysis → Core engine design
   - Core engine → API integration
   - All components → Edge case implementation

2. **Parallel Opportunities**:
   - Prompt engineering can proceed parallel to core engine
   - UI integration can develop parallel to backend engine
   - Performance monitoring can be designed early

3. **Integration Points**:
   - Daily coordination checkpoints
   - Shared interface definitions
   - Integration testing milestones

---

## 📊 **Success Metrics & Validation**

### **Technical Performance Targets**

**Compression Efficiency**:
- ✅ 60-80% context size reduction while preserving key information
- ✅ <2 second summarization latency for typical conversations
- ✅ <5% information loss in critical decision points

**System Integration**:
- ✅ Seamless integration with existing Claude Code command system
- ✅ Zero impact on normal chat flow performance
- ✅ 100% backward compatibility with existing sessions

**User Experience**:
- ✅ Intuitive `/compact` command with clear progress indication
- ✅ Graceful error handling with actionable user feedback
- ✅ Consistent behavior across CLI and server deployment modes

### **Quality Assurance Validation**

**Functional Testing**:
```bash
# Test scenarios to validate
1. Normal conversation compaction (happy path)
2. Multiple sequential compactions
3. Compaction with insufficient context
4. Concurrent summarization attempts  
5. Recovery from corrupted summaries
6. Edge cases (empty sessions, single message, etc.)
```

**Performance Testing**:
```bash
# Performance validation requirements
1. Summarization latency <2s for 50-message conversations
2. Context retrieval <200ms after compaction
3. Memory usage efficiency (no memory leaks)
4. Concurrent session handling without degradation
```

---

## 🔄 **Implementation Execution Plan**

### **Phase Execution Strategy**

**Week 1: Foundation & Analysis**
- Architecture analysis and integration design
- Technical dependency assessment
- API architecture decisions
- Initial prompt engineering research

**Week 2: Core Implementation**  
- Summarization engine development
- Message processing pipeline
- Context compaction logic
- Model integration and testing

**Week 3: Integration & UI**
- Command system integration
- API endpoint implementation
- UI event system integration
- Error handling and validation

**Week 4: Hardening & Deployment**
- Edge case implementation
- Performance optimization
- Comprehensive testing
- Production readiness validation

### **Context Rot Prevention During Implementation**

**Agent Context Management**:
- Each sub-agent works on focused, well-defined components
- Regular context refresh cycles every 24 hours
- Shared documentation to prevent information loss
- Integration testing to ensure component compatibility

**Coordination Checkpoints**:
- Daily progress reviews and integration validation
- Shared interface definitions and API contracts
- Continuous testing of integrated components
- Performance monitoring throughout development

---

## 🎯 **Immediate Next Actions**

1. **Start with Architecture Analysis**: Deploy Architecture Specialist agent to analyze current Claude Code structure and identify integration points

2. **Establish Shared Definitions**: Create shared interface definitions and data structures for cross-agent coordination

3. **Set Up Development Environment**: Ensure proper development environment with testing capabilities

4. **Begin Parallel Development**: Once architecture is clear, deploy multiple specialized agents on independent components

5. **Implement Integration Testing**: Set up continuous integration testing to validate component interactions

---

## 💡 **Success Criteria Summary**

**Implementation Success**:
- ✅ `/compact` command functional in Claude Code
- ✅ Context compression achieves 60-80% size reduction
- ✅ Seamless integration with existing chat flow
- ✅ Production-ready error handling and edge cases

**User Experience Success**:
- ✅ Intuitive command with clear feedback
- ✅ Transparent context management
- ✅ Improved conversation efficiency
- ✅ No disruption to existing workflows

**Technical Success**:
- ✅ <2s summarization latency
- ✅ Robust concurrency handling
- ✅ Comprehensive test coverage
- ✅ Performance monitoring and optimization

This implementation will deliver a production-ready context compression system that significantly improves Claude Code's efficiency in long conversations while maintaining the quality and continuity that users expect.