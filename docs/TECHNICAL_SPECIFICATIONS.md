# Technical Specifications for Multi-CLI Agent Coordination
*Complete implementation specifications for autonomous development*

## 🎯 Document Purpose

This document provides complete technical specifications for implementing the multi-CLI agent coordination system. It serves as the authoritative reference for autonomous implementation by Claude Code.

**Context**: Following Epic 1's 85% technical debt reduction, we're implementing heterogeneous CLI agent coordination to enable Claude Code, Cursor, Gemini CLI, and other tools to work together in isolated git worktrees.

---

## 📋 Architecture Overview

### Current vs Target State

**Current**: Homogeneous Python agent coordination
```
[Redis Orchestrator] → [Python Agent] → [Python Agent] → [Python Agent]
```

**Target**: Heterogeneous multi-CLI coordination with isolation
```
[Universal Orchestrator] → [Claude Code] → [Cursor] → [Gemini CLI]
                         ↓              ↓           ↓
                    [Worktree A]   [Worktree B] [Worktree C]
```

### Core Components to Implement

1. **Universal Agent Interface Layer** (`app/core/agents/`)
2. **CLI Adapters** (`app/core/agents/adapters/`)
3. **Git Worktree Isolation** (`app/core/isolation/`)
4. **Enhanced Orchestration** (`app/core/orchestration/`)
5. **Multi-CLI Communication** (`app/core/communication/`)

---

## 🔧 Implementation Specifications

### 1. Universal Agent Interface (CRITICAL - Foundation)

**File**: `app/core/agents/universal_agent_interface.py` ✅ COMPLETE
**Status**: Fully implemented with comprehensive interface definitions

**Key Classes**:
- `UniversalAgentInterface`: Abstract base for all CLI agents
- `AgentTask`: Standardized task format
- `AgentResult`: Standardized result format
- `ExecutionContext`: Context preservation for handoffs
- `AgentCapability`: Capability discovery and routing

**Implementation Complete**: ✅
- All abstract methods defined with detailed docstrings
- Complete type hints and validation
- Error handling and performance monitoring
- Thread safety and concurrent execution support

### 2. Claude Code Adapter (HIGH PRIORITY)

**File**: `app/core/agents/adapters/claude_code_adapter.py` 
**Status**: TEMPLATE - Requires implementation completion

**Critical Methods to Implement**:

#### `execute_task()` - REQUIRES IMPLEMENTATION
```python
async def execute_task(self, task: AgentTask) -> AgentResult:
    """
    IMPLEMENTATION REQUIRED:
    1. Validate task against capabilities and security constraints
    2. Translate universal task to Claude Code CLI format
    3. Execute subprocess with proper isolation
    4. Monitor performance and resource usage
    5. Parse results and create AgentResult
    
    Current Status: Template with structure, needs completion
    Priority: CRITICAL (blocks all functionality)
    """
```

**Translation Logic Needed**:
```python
def _translate_task_to_command(self, task: AgentTask) -> ClaudeCodeCommand:
    """
    Map CapabilityType to Claude Code commands:
    - CODE_ANALYSIS → "claude analyze [files] --output=json"
    - CODE_REVIEW → "claude review [files] --format=structured"  
    - DOCUMENTATION → "claude document [files] --style=comprehensive"
    - REFACTORING → "claude refactor [files] --pattern=[pattern]"
    - DEBUGGING → "claude debug [files] --context=[context]"
    """
```

**Subprocess Execution Pattern**:
```python
async def _execute_claude_command(self, command: ClaudeCodeCommand, context: ExecutionContext):
    """
    Required implementation:
    1. Set up isolated environment (worktree path, env vars)
    2. Build CLI arguments with security validation
    3. Execute asyncio.create_subprocess_exec() with limits
    4. Handle timeout, capture stdout/stderr
    5. Parse JSON output and handle errors
    6. Track resource usage and performance
    """
```

#### `get_capabilities()` - REQUIRES IMPLEMENTATION
```python
async def get_capabilities(self) -> List[AgentCapability]:
    """
    IMPLEMENTATION REQUIRED:
    Return dynamic capability assessment based on:
    1. CLI tool availability and version
    2. Current system load and performance
    3. Historical success rates for each capability
    4. Resource availability and constraints
    
    Current Status: Static capabilities defined, needs dynamic assessment
    """
```

#### `health_check()` - REQUIRES IMPLEMENTATION
```python
async def health_check(self) -> HealthStatus:
    """
    IMPLEMENTATION REQUIRED:
    1. Check CLI availability: `claude --version`
    2. Measure response time and resource usage
    3. Assess current capacity and load
    4. Evaluate error rates and performance trends
    5. Return comprehensive health status
    
    Current Status: Template structure, needs implementation
    """
```

### 3. Git Worktree Isolation System (CRITICAL - Security)

**File**: `app/core/isolation/worktree_manager.py` - REQUIRES CREATION
**Priority**: CRITICAL (security foundation)

**Core Implementation Required**:
```python
class WorktreeManager:
    """Manages isolated git worktrees for agent execution"""
    
    async def create_worktree(self, agent_id: str, branch_name: str, base_path: str) -> WorktreeContext:
        """
        IMPLEMENTATION REQUIRED:
        1. Create isolated git worktree: `git worktree add [path] [branch]`
        2. Set up directory permissions and access controls
        3. Configure environment isolation
        4. Return WorktreeContext with path and metadata
        
        Security Requirements:
        - Prevent path traversal attacks (../../etc/passwd)
        - Restrict access to parent directories
        - Set resource limits (disk space, file count)
        - Monitor for suspicious file operations
        """
        
    async def cleanup_worktree(self, worktree_id: str) -> bool:
        """
        IMPLEMENTATION REQUIRED:
        1. Complete any active tasks in worktree
        2. Save important changes to git
        3. Remove worktree: `git worktree remove [path]`
        4. Clean up temporary files and processes
        """
        
    async def validate_path_access(self, worktree_id: str, requested_path: str) -> bool:
        """
        IMPLEMENTATION REQUIRED:
        1. Resolve absolute path and check bounds
        2. Validate against worktree boundaries
        3. Block access to system directories
        4. Check file permissions and ownership
        """
```

**Security Implementation**:
```python
class PathValidator:
    """Security layer for path access validation"""
    
    def validate_file_access(self, worktree_path: str, file_path: str) -> bool:
        """
        IMPLEMENTATION REQUIRED:
        1. Resolve paths: os.path.realpath() to handle symlinks
        2. Check bounds: file_path must be within worktree_path
        3. Block system paths: /etc, /usr, /bin, /sys, /proc
        4. Validate file extensions and size limits
        
        Security Patterns:
        - Path traversal: block ../../../
        - Symlink escape: resolve and validate real paths
        - System access: block access to critical directories
        - Resource limits: max file size, disk usage
        """
```

### 4. Enhanced Orchestrator (HIGH PRIORITY)

**File**: `app/core/orchestration/universal_orchestrator.py` - REQUIRES CREATION
**Priority**: HIGH (system coordination)

**Core Functionality**:
```python
class UniversalOrchestrator:
    """Multi-CLI agent orchestration engine"""
    
    async def delegate_task(self, task: Task, preferred_agent_type: Optional[str] = None, 
                          worktree_isolation: bool = True) -> OrchestrationResult:
        """
        IMPLEMENTATION REQUIRED:
        1. Analyze task requirements and capabilities needed
        2. Query agent registry for capable agents
        3. Select optimal agent based on load and performance
        4. Create isolated worktree if required
        5. Execute task with monitoring and error handling
        6. Clean up resources and return results
        
        Key Logic:
        - Capability matching: find agents with required skills
        - Load balancing: prefer less loaded agents
        - Failure recovery: retry with different agents
        - Context preservation: maintain state across handoffs
        """
        
    async def coordinate_workflow(self, workflow: WorkflowDefinition) -> WorkflowResult:
        """
        IMPLEMENTATION REQUIRED:
        1. Parse workflow steps and dependencies
        2. Execute steps in correct order (sequential/parallel)
        3. Handle context handoffs between agents
        4. Monitor progress and handle failures
        5. Provide real-time status updates
        
        Workflow Patterns:
        - Sequential: A → B → C (with context handoff)
        - Parallel: A + B + C → merge results
        - Conditional: if A succeeds then B else C
        - Pipeline: continuous A → B → C → A...
        """
```

### 5. Multi-CLI Communication Enhancement (MEDIUM PRIORITY)

**File**: `app/core/communication/multi_cli_protocol.py` - REQUIRES CREATION
**Priority**: MEDIUM (protocol standardization)

**Protocol Translation**:
```python
class MultiCLIProtocol:
    """Enhanced protocol for multi-CLI coordination"""
    
    async def translate_message(self, message: AgentMessage, target_cli_type: str) -> Dict[str, Any]:
        """
        IMPLEMENTATION REQUIRED:
        1. Parse universal message format
        2. Map to CLI-specific format and terminology
        3. Handle CLI-specific options and parameters
        4. Preserve semantic meaning across translations
        
        Translation Examples:
        Universal → Claude Code:
        {
            "type": "task_request",
            "capability": "code_analysis", 
            "files": ["src/main.py"],
            "context": "Find performance bottlenecks"
        }
        →
        ["claude", "analyze", "src/main.py", "--focus=performance", "--output=json"]
        
        Universal → Cursor:
        Same input →
        {
            "action": "analyze_performance",
            "files": ["src/main.py"],
            "instructions": "Find performance bottlenecks"
        }
        """
```

---

## 🧪 Testing Specifications

### Testing Framework Status
**File**: `tests/multi_cli_agent_testing_framework.py` ✅ COMPLETE
**Status**: Comprehensive testing framework implemented

**Test Coverage Required**:
1. **Unit Tests**: Each adapter method (90% coverage target)
2. **Integration Tests**: Multi-agent workflows
3. **Security Tests**: Path traversal, injection attacks  
4. **Performance Tests**: Load testing, response times
5. **End-to-End Tests**: Complete workflows with multiple agents

**Critical Test Scenarios**:
```python
# Security validation
async def test_path_traversal_prevention():
    """Verify agents cannot access files outside worktree"""
    
# Multi-agent coordination
async def test_sequential_workflow():
    """Claude Code analysis → Cursor implementation → Gemini CLI testing"""
    
# Error recovery
async def test_agent_failure_recovery():
    """Workflow continues when one agent fails"""
    
# Performance benchmarks
async def test_concurrent_execution():
    """10+ agents working simultaneously within performance targets"""
```

---

## 🔧 Implementation Priority Matrix

### CRITICAL (Week 1) - Foundation
1. **Complete Claude Code Adapter** (`claude_code_adapter.py`)
   - `execute_task()` method implementation
   - `health_check()` dynamic assessment  
   - `get_capabilities()` dynamic reporting
   - Subprocess execution with security

2. **Git Worktree Isolation** (`app/core/isolation/`)
   - `WorktreeManager` with create/cleanup/validate
   - `PathValidator` with security enforcement
   - Resource monitoring and limits

3. **Basic Orchestrator** (`app/core/orchestration/`)
   - `UniversalOrchestrator` with task delegation
   - Agent registry integration
   - Error handling and recovery

### HIGH (Week 2) - Integration
1. **Cursor Adapter Implementation**
2. **Multi-Agent Workflow Coordination**
3. **Enhanced Communication Protocol**
4. **Comprehensive Testing**

### MEDIUM (Week 3) - Polish
1. **Gemini CLI Adapter**
2. **Performance Optimization**
3. **Monitoring and Observability**
4. **Documentation and Examples**

---

## 📊 Success Criteria and Validation

### Technical Validation
- [ ] Claude Code adapter executes all task types successfully
- [ ] Git worktree isolation prevents security breaches
- [ ] Multi-agent workflows complete end-to-end
- [ ] Performance targets met: <5s response, <1GB memory
- [ ] 90% test coverage across all components

### Functional Validation
```python
# End-to-end workflow test
async def test_feature_development_workflow():
    """
    Test complete workflow:
    1. Claude Code: Analyze requirements → generate plan
    2. Cursor: Implement feature based on plan  
    3. Gemini CLI: Create tests and validation
    4. Verify: All files created, tests pass, feature works
    """
```

### Security Validation  
```python
# Security penetration test
async def test_security_boundaries():
    """
    Verify security constraints:
    1. Path traversal attacks blocked
    2. System file access prevented
    3. Resource limits enforced
    4. Process isolation maintained
    """
```

---

## 🚀 Implementation Execution Guide

### Phase 1: Foundation (Days 1-7)
1. **Day 1-2**: Complete Claude Code adapter `execute_task()` method
2. **Day 3-4**: Implement git worktree isolation system
3. **Day 5-6**: Create basic universal orchestrator
4. **Day 7**: Integration testing and validation

### Phase 2: Integration (Days 8-14)  
1. **Day 8-10**: Cursor adapter and multi-agent workflows
2. **Day 11-12**: Communication protocol enhancement
3. **Day 13-14**: Comprehensive testing and optimization

### Phase 3: Production (Days 15-21)
1. **Day 15-17**: Additional adapters and performance tuning
2. **Day 18-19**: Monitoring and observability
3. **Day 20-21**: Documentation and deployment preparation

### Validation Gates
**After Phase 1**: Basic functionality working
**After Phase 2**: Multi-agent coordination operational  
**After Phase 3**: Production-ready system

---

## 📁 File Structure Summary

```
app/core/agents/
├── __init__.py ✅
├── universal_agent_interface.py ✅
├── models.py ✅
├── agent_registry.py ✅
└── adapters/
    ├── __init__.py ✅
    ├── claude_code_adapter.py 🔨 (REQUIRES COMPLETION)
    ├── cursor_adapter.py 🔨 (TEMPLATE)
    ├── gemini_cli_adapter.py 🔨 (TEMPLATE)
    ├── opencode_adapter.py 🔨 (TEMPLATE)
    └── github_copilot_adapter.py 🔨 (TEMPLATE)

app/core/isolation/ (REQUIRES CREATION)
├── __init__.py
├── worktree_manager.py 
├── path_validator.py
└── security_enforcer.py

app/core/orchestration/ (REQUIRES CREATION)
├── __init__.py
├── universal_orchestrator.py
├── task_router.py
└── coordination_engine.py

tests/ ✅
├── multi_cli_agent_testing_framework.py ✅
├── git_worktree_isolation_tests.py ✅
└── ... (complete test suite) ✅
```

**Legend**: ✅ Complete | 🔨 Requires Implementation

---

## 💡 Implementation Notes for Claude Code

### Key Implementation Patterns

1. **Async/Await**: All methods use async/await for non-blocking execution
2. **Error Handling**: Comprehensive try/catch with specific error types
3. **Resource Management**: Context managers and proper cleanup
4. **Security First**: Validate all inputs, paths, and commands
5. **Performance Monitoring**: Track execution time and resource usage

### Common Implementation Challenges

1. **Subprocess Security**: Use `asyncio.create_subprocess_exec()` with argument lists (not shell strings)
2. **Path Validation**: Always resolve paths and check boundaries  
3. **JSON Parsing**: Handle malformed output gracefully
4. **Resource Limits**: Monitor and enforce CPU, memory, and time limits
5. **Error Recovery**: Implement retry logic and fallback strategies

### Development Tips

1. **Start Simple**: Implement basic functionality first, add complexity incrementally
2. **Test Early**: Write tests alongside implementation  
3. **Log Everything**: Comprehensive logging for debugging and monitoring
4. **Document Decisions**: Comment complex logic and security measures
5. **Validate Continuously**: Run tests and security checks frequently

This technical specification provides complete implementation guidance for autonomous development. Each section includes specific requirements, code patterns, and validation criteria needed for successful implementation.

**Ready for autonomous implementation by Claude Code.** 🚀