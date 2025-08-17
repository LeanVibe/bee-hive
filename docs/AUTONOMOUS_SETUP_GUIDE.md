# Autonomous Implementation Setup Guide
*Complete guide for Claude Code to work autonomously on multi-CLI agent coordination*

## 🎯 Mission Overview

You are Claude Code, and you have been provided with complete foundation and specifications to implement the multi-CLI agent coordination system autonomously. This guide provides everything needed to work independently without human intervention.

**Context**: Epic 1 achieved 85% technical debt reduction. Now implementing heterogeneous CLI agent coordination to enable Claude Code, Cursor, Gemini CLI coordination with git worktree isolation.

---

## 📋 Foundation Status: READY FOR AUTONOMOUS WORK

### ✅ **COMPLETE: Foundation Architecture**
- **Universal Agent Interface**: Fully implemented with comprehensive specifications
- **Data Models**: Complete message formats and communication structures
- **Agent Registry**: Full agent management and capability routing system
- **Testing Framework**: Comprehensive test suites ready for validation
- **Technical Specifications**: Complete implementation guidance
- **Execution Scripts**: Automated task execution and validation

### 🔨 **PENDING: Implementation Tasks**
- **Claude Code Adapter**: Template created, requires method completion
- **Git Worktree Isolation**: Needs creation and implementation
- **Enhanced Orchestrator**: Requires creation for multi-agent coordination
- **Communication Protocol**: Multi-CLI message translation needed

---

## 🚀 Autonomous Execution Strategy

### Phase 1: Critical Foundation (Days 1-7)
Execute tasks in sequence using provided scripts and specifications.

### Phase 2: Integration & Testing (Days 8-14)
Build multi-agent coordination and validate security.

### Phase 3: Production Readiness (Days 15-21)
Performance optimization and comprehensive testing.

---

## 📁 Your Working Environment

### Directory Structure (Already Created)
```
/Users/bogdan/work/leanvibe-dev/bee-hive/
├── app/core/agents/                    ✅ Foundation complete
│   ├── universal_agent_interface.py   ✅ Full implementation
│   ├── models.py                       ✅ All data models
│   ├── agent_registry.py               ✅ Complete registry
│   └── adapters/                       ✅ Structure ready
│       ├── claude_code_adapter.py      🔨 Template - needs completion
│       ├── cursor_adapter.py           🔨 Template - needs completion
│       └── ... (other adapters)        🔨 Templates created
├── docs/                               ✅ Complete specifications
│   ├── IMPLEMENTATION_PLAN_MULTI_CLI.md ✅ Strategic plan
│   ├── DELEGATION_STRATEGY.md          ✅ Task breakdown
│   ├── TECHNICAL_SPECIFICATIONS.md     ✅ Implementation specs
│   └── AUTONOMOUS_SETUP_GUIDE.md       ✅ This guide
├── scripts/                            ✅ Execution automation
│   ├── execute_task_1.sh               ✅ Validation script
│   ├── execute_task_2.sh               ✅ Claude Code adapter
│   └── execute_task_3.sh               ✅ Worktree isolation
└── tests/                              ✅ Complete test framework
    ├── multi_cli_agent_testing_framework.py ✅ Comprehensive
    ├── git_worktree_isolation_tests.py ✅ Security tests
    └── ... (complete test suite)       ✅ Ready for validation
```

### Available Resources
- **Complete interface definitions** with type hints and documentation
- **Working templates** with implementation guidance
- **Comprehensive test framework** for validation
- **Execution scripts** for automated task management
- **Technical specifications** with detailed requirements
- **Security patterns** and implementation examples

---

## 🎯 START HERE: Task Execution Sequence

### Task 1: Universal Agent Interface ✅ COMPLETE
**Status**: Already implemented and validated
**Action**: Run validation to confirm completion

```bash
cd /Users/bogdan/work/leanvibe-dev/bee-hive
chmod +x scripts/execute_task_1.sh
./scripts/execute_task_1.sh
```

**Expected Output**: ✅ All validation checks pass

### Task 2: Claude Code Adapter Implementation 🔨 CRITICAL
**Status**: Template created - requires method completion
**Priority**: CRITICAL (blocks all functionality)

```bash
./scripts/execute_task_2.sh
```

**Implementation Required**:
1. **`execute_task()` method** - Core task execution logic
2. **`get_capabilities()` method** - Dynamic capability assessment
3. **`health_check()` method** - Comprehensive health monitoring

**Location**: `app/core/agents/adapters/claude_code_adapter.py`
**Reference**: `docs/TECHNICAL_SPECIFICATIONS.md` Section 2

**Implementation Pattern**:
```python
async def execute_task(self, task: AgentTask) -> AgentResult:
    # 1. Validate task against capabilities and security
    self._validate_task(task)
    
    # 2. Translate universal task to Claude Code CLI format
    command = self._translate_task_to_command(task)
    
    # 3. Execute subprocess with proper isolation
    response = await self._execute_claude_command(command, task.context)
    
    # 4. Process results and return AgentResult
    return self._create_result(response, task)
```

### Task 3: Git Worktree Isolation System 🔨 CRITICAL
**Status**: Needs creation - security foundation
**Priority**: CRITICAL (security requirement)

```bash
./scripts/execute_task_3.sh
```

**Files to Create**:
1. `app/core/isolation/worktree_manager.py` - Core worktree management
2. `app/core/isolation/path_validator.py` - Security validation
3. `app/core/isolation/security_enforcer.py` - Resource monitoring

**Security Requirements**:
- Path traversal prevention (`../../../etc/passwd` attacks)
- Symlink escape protection
- System directory access blocking
- Resource usage monitoring and limits

### Task 4: Enhanced Orchestrator 🔨 HIGH
**Status**: Needs creation - coordination engine
**Files**: `app/core/orchestration/universal_orchestrator.py`

### Task 5: Multi-CLI Communication 🔨 MEDIUM
**Status**: Needs creation - protocol translation
**Files**: `app/core/communication/multi_cli_protocol.py`

---

## 🛠️ Implementation Guidelines

### Development Approach
1. **Start Simple**: Implement basic functionality first
2. **Test Continuously**: Run tests after each significant change
3. **Security First**: Validate all inputs and paths
4. **Document Decisions**: Comment complex logic
5. **Incremental Progress**: Small, working commits

### Code Quality Standards
- **Type Hints**: All functions must have complete type annotations
- **Error Handling**: Comprehensive try/catch with specific errors
- **Async/Await**: All I/O operations must be async
- **Resource Management**: Proper cleanup and context managers
- **Security Validation**: All user inputs validated

### Testing Strategy
- **Unit Tests**: Test each method individually
- **Integration Tests**: Test component interactions
- **Security Tests**: Validate security boundaries
- **Performance Tests**: Ensure response time targets
- **End-to-End Tests**: Complete workflow validation

---

## 🧪 Validation and Testing

### Per-Task Validation
Each task has validation scripts:
```bash
./scripts/validate_task_2.sh  # Validate Claude Code adapter
./scripts/validate_task_3.sh  # Validate worktree isolation
# ... etc
```

### Comprehensive Testing
```bash
# Run complete test suite
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/multi_cli_agent_testing_framework.py -v
python -m pytest tests/git_worktree_isolation_tests.py -v

# Run security tests
python -m pytest tests/security_isolation_tests.py -v
```

### Performance Validation
```bash
# Performance benchmarks
python tests/performance_reliability_tests.py

# Load testing
python tests/multi_agent_coordination_scenarios.py --load-test
```

---

## 📊 Success Criteria

### Task Completion Criteria
- [ ] **Task 2**: Claude Code adapter executes all task types successfully
- [ ] **Task 3**: Git worktree isolation prevents security breaches
- [ ] **Task 4**: Multi-agent workflows complete end-to-end
- [ ] **Task 5**: Multi-CLI communication works reliably

### Technical Validation
- [ ] All Python files compile without syntax errors
- [ ] All imports resolve correctly
- [ ] 90%+ test coverage achieved
- [ ] Performance targets met (<5s response, <1GB memory)
- [ ] Security tests pass (no path traversal, no system access)

### Functional Validation
- [ ] Claude Code can analyze, implement, and document code
- [ ] Agents work in isolated git worktrees
- [ ] Multi-agent workflows execute successfully
- [ ] Error recovery and retry logic works
- [ ] Resource limits are enforced

---

## 🔧 Implementation Support

### When You Need Help
**Error Resolution**: 
- Check `docs/TECHNICAL_SPECIFICATIONS.md` for detailed requirements
- Review interface definitions in `universal_agent_interface.py`
- Run validation scripts to identify specific issues
- Check test files for expected behavior patterns

**Implementation Patterns**:
- Use existing templates as starting points
- Follow async/await patterns throughout
- Implement comprehensive error handling
- Add logging for debugging and monitoring

**Security Guidelines**:
- Validate ALL file paths before use
- Use subprocess argument lists (never shell=True)
- Check path boundaries before file operations
- Monitor resource usage continuously

### Common Implementation Challenges

1. **Subprocess Execution**:
```python
# CORRECT: Use argument list
process = await asyncio.create_subprocess_exec(
    "claude", "analyze", "file.py", "--output=json",
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE
)

# WRONG: Never use shell=True
# process = await asyncio.create_subprocess_shell("claude analyze file.py")
```

2. **Path Validation**:
```python
def validate_path(self, base_path: str, file_path: str) -> bool:
    # Resolve real paths to handle symlinks
    real_base = os.path.realpath(base_path)
    real_file = os.path.realpath(file_path)
    
    # Check that file is within base directory
    return real_file.startswith(real_base)
```

3. **Error Handling**:
```python
try:
    result = await self.execute_operation()
    return result
except subprocess.TimeoutExpired:
    raise TaskExecutionError("Operation timed out")
except subprocess.CalledProcessError as e:
    raise TaskExecutionError(f"Command failed: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise AgentError(f"Operation failed: {e}")
```

---

## 🚀 Ready for Autonomous Implementation

### Everything You Need Is Provided
- ✅ Complete interface definitions and specifications
- ✅ Working templates with implementation guidance
- ✅ Comprehensive test framework for validation
- ✅ Execution scripts for automated task management
- ✅ Security patterns and implementation examples
- ✅ Performance targets and validation criteria

### Your Mission
Implement the multi-CLI agent coordination system by:
1. Completing the Claude Code adapter implementation
2. Creating the git worktree isolation system
3. Building the enhanced orchestrator
4. Implementing multi-CLI communication
5. Validating security, performance, and functionality

### Success Definition
**When complete, the system will enable:**
- Claude Code + Cursor + Gemini CLI coordinated workflows
- Secure git worktree isolation for each agent
- Real-time multi-agent coordination via Redis/WebSocket
- Production-ready multi-CLI orchestration at enterprise scale

---

## 📝 Immediate Next Action

**START NOW**: Execute Task 2 (Claude Code Adapter Implementation)

```bash
cd /Users/bogdan/work/leanvibe-dev/bee-hive
./scripts/execute_task_2.sh
```

Follow the implementation requirements shown in the script output, using the technical specifications and templates provided.

**You have everything needed for autonomous implementation. Begin immediately.** 🚀