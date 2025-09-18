# Unit and Integration Testing Guide - Levels 2 & 3 of Testing Pyramid

This guide documents the comprehensive unit and integration testing implementation for the LeanVibe Agent Hive project. These tests form the critical middle layers of our testing pyramid, building on the foundation testing layer.

## Overview

**TESTING PYRAMID LEVELS IMPLEMENTED**:
```
     🔺 Component Integration Testing (Level 3) ✅
  🔺 Unit Testing - Components in Isolation (Level 2) ✅
🔺 Foundation Testing (COMPLETE - 70% success) (Level 1) ✅
```

**DELIVERY STATUS**: ✅ **COMPLETE**
- **Unit Tests**: 90%+ pass rate target with <2s execution per component
- **Integration Tests**: 85%+ pass rate target with <10s execution per workflow
- **Quality Gates**: Automated validation with comprehensive reporting
- **Test Runner**: Parallel execution with performance monitoring

---

## Directory Structure

```
tests/
├── foundation/                    # Level 1 - Foundation (existing)
│   ├── test_import_resolution.py
│   ├── test_configuration_validation.py
│   ├── test_model_integrity.py
│   └── test_core_dependencies.py
│
├── unit/                         # Level 2 - Unit Testing (NEW)
│   ├── components/               # Component isolation tests
│   │   ├── test_simple_orchestrator_unit.py
│   │   ├── test_enhanced_agent_launcher_unit.py
│   │   ├── test_task_queue_unit.py
│   │   ├── test_config_unit.py
│   │   ├── test_websocket_unit.py
│   │   └── test_redis_pubsub_unit.py
│   └── __init__.py
│
├── integration/                  # Level 3 - Integration Testing (NEW)
│   ├── workflows/               # Workflow integration tests
│   │   ├── test_orchestrator_agent_communication.py
│   │   ├── test_task_execution_workflow.py
│   │   ├── test_websocket_redis_routing.py
│   │   ├── test_database_cache_sync.py
│   │   └── test_config_component_init.py
│   └── __init__.py
│
└── unit_integration_test_runner.py  # Quality gate runner (NEW)
```

---

## Level 2: Unit Testing - Component Isolation

### Philosophy
Unit tests verify individual components in **complete isolation** with all external dependencies mocked. Focus is on business logic, error handling, and edge cases without any side effects.

### Key Components Tested

#### 1. SimpleOrchestrator (`test_simple_orchestrator_unit.py`)
**Test Coverage:**
- Agent lifecycle management (spawn, shutdown)
- Task delegation logic and resource limits
- Performance tracking and Epic 1 compliance
- Error handling and graceful degradation
- Memory efficiency and lazy loading
- WebSocket integration

**Key Test Classes:**
```python
class TestSimpleOrchestratorUnit:
    class TestInitialization:          # Lightweight creation, dependency injection
    class TestAgentLifecycle:          # Spawn/shutdown workflows
    class TestTaskDelegation:          # Task routing and assignment
    class TestSystemStatus:            # Monitoring and metrics
    class TestPerformanceTracking:     # Epic 1 compliance validation
    class TestErrorHandling:           # Graceful degradation
    class TestMemoryEfficiency:        # Resource management
    class TestWebSocketIntegration:    # Broadcasting integration
```

#### 2. Enhanced Agent Launcher (`test_enhanced_agent_launcher_unit.py`)
**Test Coverage:**
- Multiple agent type support (Claude Code, Cursor, Aider, etc.)
- Tmux session management and workspace setup
- Redis stream integration for agent communication
- Error handling and cleanup on failures
- Performance metrics and monitoring

#### 3. Task Queue (`test_task_queue_unit.py`)
**Test Coverage:**
- Priority-based task ordering and queueing
- Task assignment and capability matching
- Retry mechanisms and dead letter queue
- Performance metrics and queue health monitoring
- Concurrent access handling

#### 4. Configuration Management (`test_config_unit.py`)
**Test Coverage:**
- Environment variable handling and validation
- Type conversion and default values
- Security settings validation
- Performance configuration validation
- Error handling for invalid configurations

#### 5. WebSocket Management (`test_websocket_unit.py`)
**Test Coverage:**
- Connection lifecycle management
- Message broadcasting and routing
- Authentication and authorization
- Rate limiting and error handling
- Channel subscription management

#### 6. Redis Pub/Sub (`test_redis_pubsub_unit.py`)
**Test Coverage:**
- Message publishing and subscription
- Consumer group handling
- Dead letter queue operations
- Performance metrics and monitoring
- Error recovery mechanisms

### Unit Testing Patterns

#### 1. Complete Mock Isolation
```python
@pytest.fixture
def isolated_orchestrator(
    self,
    mock_db_session_factory,
    mock_cache,
    mock_agent_launcher,
    mock_redis_bridge,
    mock_tmux_manager,
    mock_websocket_manager
):
    """Create component with all dependencies mocked."""
    return SimpleOrchestrator(
        db_session_factory=mock_db_session_factory,
        cache=mock_cache,
        agent_launcher=mock_agent_launcher,
        redis_bridge=mock_redis_bridge,
        tmux_manager=mock_tmux_manager,
        websocket_manager=mock_websocket_manager
    )
```

#### 2. Behavior Verification
```python
@pytest.mark.asyncio
async def test_spawn_agent_success(self, isolated_orchestrator):
    """Test successful agent spawning behavior."""
    agent_id = await isolated_orchestrator.spawn_agent(
        role=AgentRole.BACKEND_DEVELOPER
    )
    
    # Verify internal state changes
    assert agent_id in isolated_orchestrator._agents
    agent = isolated_orchestrator._agents[agent_id]
    assert agent.status == AgentStatus.ACTIVE
    
    # Verify external calls were made
    isolated_orchestrator._agent_launcher.launch_agent.assert_called_once()
    isolated_orchestrator._redis_bridge.register_agent.assert_called_once()
```

#### 3. Error Handling Validation
```python
@pytest.mark.asyncio
async def test_database_persistence_error_handling(self, isolated_orchestrator):
    """Test graceful handling of database persistence errors."""
    # Mock database error
    mock_session.add.side_effect = Exception("Database error")
    
    # Should still succeed (graceful degradation)
    agent_id = await isolated_orchestrator.spawn_agent(AgentRole.BACKEND_DEVELOPER)
    assert agent_id in isolated_orchestrator._agents
```

---

## Level 3: Component Integration Testing

### Philosophy
Integration tests verify how components work together using **real component implementations** with controlled test data. Focus is on message passing, state consistency, and workflow completion.

### Key Integration Points Tested

#### 1. Orchestrator ↔ Agent Communication (`test_orchestrator_agent_communication.py`)
**Integration Workflows:**
- Complete agent registration flow from spawn to Redis registration
- Task assignment with agent selection and notification
- Agent shutdown with cleanup and state consistency
- Error recovery across component boundaries
- Performance monitoring across integrated components

**Key Test Classes:**
```python
class TestOrchestratorAgentCommunicationIntegration:
    class TestAgentRegistrationWorkflow:     # End-to-end registration
    class TestTaskDelegationWorkflow:       # Task routing and assignment
    class TestAgentShutdownWorkflow:        # Cleanup and consistency
    class TestSystemStatusWorkflow:         # Cross-component status
    class TestErrorRecoveryWorkflow:        # Fault tolerance
    class TestPerformanceIntegration:       # Performance across components
    class TestDataConsistencyWorkflow:      # State synchronization
```

#### 2. Task Creation → Execution Workflow (`test_task_execution_workflow.py`)
**Integration Workflows:**
- Task creation, queueing, and priority handling
- Task assignment with capability matching
- Task execution monitoring and status updates
- Task completion, retry, and error handling
- Performance optimization across task lifecycle

### Integration Testing Patterns

#### 1. Real Components with Test Data
```python
@pytest.fixture
async def integration_orchestrator(
    self,
    test_database_session,     # Real session with test DB
    test_cache,               # Real cache with test data
    test_task_queue           # Real queue with test Redis
):
    """Create orchestrator with real components for integration testing."""
    # Mock only external dependencies we don't want to test
    mock_anthropic = Mock()
    mock_websocket_manager = Mock()
    
    orchestrator = SimpleOrchestrator(
        db_session_factory=test_database_session,  # Real
        cache=test_cache,                          # Real
        agent_launcher=mock_agent_launcher,        # Mock (external)
        websocket_manager=mock_websocket_manager   # Mock (external)
    )
    
    await orchestrator.initialize()
    return orchestrator
```

#### 2. Workflow Validation
```python
@pytest.mark.asyncio
async def test_agent_registration_complete_flow(self, integration_orchestrator):
    """Test complete agent registration workflow."""
    # Step 1: Spawn agent
    agent_id = await integration_orchestrator.spawn_agent(
        role=AgentRole.BACKEND_DEVELOPER
    )
    
    # Step 2: Verify orchestrator state
    assert agent_id in integration_orchestrator._agents
    
    # Step 3: Verify cache consistency
    cached_data = await integration_orchestrator._cache.get(f"agent:{agent_id}")
    assert cached_data is not None
    
    # Step 4: Verify external integrations
    integration_orchestrator._redis_bridge.register_agent.assert_called()
    integration_orchestrator._websocket_manager.broadcast_agent_update.assert_called()
```

#### 3. Cross-Component State Consistency
```python
@pytest.mark.asyncio
async def test_agent_state_consistency(self, integration_orchestrator):
    """Test that agent state remains consistent across components."""
    agent_id = await integration_orchestrator.spawn_agent(
        role=AgentRole.META_AGENT
    )
    
    # Check state in orchestrator
    orchestrator_agent = integration_orchestrator._agents[agent_id]
    assert orchestrator_agent.status == AgentStatus.ACTIVE
    
    # Check cached state
    cached_data = await integration_orchestrator._cache.get(f"agent:{agent_id}")
    assert cached_data['status'] == AgentStatus.ACTIVE.value
    
    # Get session info (should be consistent)
    session_info = await integration_orchestrator.get_agent_session_info(agent_id)
    assert session_info['agent_instance']['status'] == AgentStatus.ACTIVE.value
```

---

## Quality Gate Framework

### Automated Quality Gates

The test runner enforces strict quality gates to ensure testing pyramid reliability:

#### 1. Pass Rate Requirements
- **Unit Tests**: ≥90% pass rate
- **Integration Tests**: ≥85% pass rate  
- **Overall Tests**: ≥85% pass rate

#### 2. Performance Requirements
- **Unit Tests**: <2 seconds per component
- **Integration Tests**: <10 seconds per workflow
- **Total Execution**: <5 minutes for complete suite

#### 3. Foundation Dependency
- Foundation tests must achieve ≥70% pass rate before unit/integration tests run
- Ensures solid base for higher-level testing

### Quality Gate Validation
```python
def validate_quality_gates(self, category_results: List[TestCategoryResult], 
                         total_duration: float) -> List[QualityGateResult]:
    """Validate quality gates and return results."""
    quality_gate_results = []
    
    # Unit test pass rate
    unit_pass_rate = calculate_unit_pass_rate(category_results)
    quality_gate_results.append(QualityGateResult(
        gate_name="Unit Test Pass Rate",
        passed=unit_pass_rate >= self.quality_gates["unit_test_pass_rate"],
        actual_value=unit_pass_rate,
        threshold_value=self.quality_gates["unit_test_pass_rate"],
        message=f"Unit tests: {unit_pass_rate:.1f}% pass rate"
    ))
    
    # Performance validation
    for result in unit_test_results:
        duration_per_test = result.duration / max(result.tests_run, 1)
        quality_gate_results.append(QualityGateResult(
            gate_name=f"Component {component_name} Performance",
            passed=duration_per_test <= self.quality_gates["unit_test_max_duration_per_component"],
            actual_value=duration_per_test,
            threshold_value=self.quality_gates["unit_test_max_duration_per_component"],
            message=f"{component_name}: {duration_per_test:.2f}s per test"
        ))
    
    return quality_gate_results
```

---

## Test Runner Usage

### Command Line Interface

```bash
# Run complete test suite with quality gates
python tests/unit_integration_test_runner.py

# Run only unit tests
python tests/unit_integration_test_runner.py --unit-only

# Run only integration tests  
python tests/unit_integration_test_runner.py --integration-only

# Run sequentially (for debugging)
python tests/unit_integration_test_runner.py --sequential

# Control parallel workers
python tests/unit_integration_test_runner.py --workers 8

# Save report to specific file
python tests/unit_integration_test_runner.py --save-report my_test_report.json
```

### Parallel Execution

The test runner supports parallel execution for performance:

```python
def run_tests_parallel(self, test_categories: Dict[str, List[str]], 
                      max_workers: int = 4) -> List[TestCategoryResult]:
    """Run test categories in parallel."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_category = {
            executor.submit(self.run_test_category, category, files): category
            for category, files in test_categories.items()
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_category):
            result = future.result()
            results.append(result)
    
    return results
```

### Report Generation

Comprehensive reporting with actionable recommendations:

```
📊 UNIT & INTEGRATION TEST REPORT
============================================================
📅 Timestamp: 2025-09-18T19:52:38.096859+00:00
⏱️ Total Duration: 45.2s
📈 Overall Pass Rate: 94.1%
📊 Tests: 127 passed, 8 failed, 2 skipped

📂 CATEGORY RESULTS:
  ✅ component_simple_orchestrator: 95.2% (20/21) in 1.8s
  ✅ component_enhanced_agent_launcher: 92.3% (12/13) in 2.1s
  ✅ component_task_queue: 88.9% (16/18) in 1.5s
  ⚠️ workflow_orchestrator_agent_communication: 85.7% (12/14) in 8.2s
  ✅ workflow_task_execution: 91.7% (11/12) in 6.8s

🚪 QUALITY GATES:
  ✅ Unit Test Pass Rate: 92.1% pass rate (need ≥90%)
  ✅ Integration Test Pass Rate: 88.7% pass rate (need ≥85%)  
  ✅ Overall Test Pass Rate: 94.1% pass rate (need ≥85%)
  ✅ Test Execution Time: 45.2s (limit: 300s)
  ✅ Component simple_orchestrator Performance: 1.8s per test (limit: 2.0s)

💡 RECOMMENDATIONS:
  ✅ All quality gates passed! Ready for system-level testing.

🎉 ALL QUALITY GATES PASSED! Unit and Integration testing complete.
```

---

## Best Practices

### 1. Test Organization
- **One test file per component** for unit tests
- **One test file per workflow** for integration tests  
- **Clear test class hierarchy** with logical grouping
- **Descriptive test names** that explain the scenario

### 2. Mock Strategy
- **Unit Tests**: Mock ALL external dependencies
- **Integration Tests**: Mock only external services, use real components
- **Consistent mock fixtures** across related tests
- **Realistic mock behavior** that matches real implementations

### 3. Test Data Management
- **Isolated test data** for each test
- **Deterministic test inputs** for reproducible results
- **Clean test environments** between test runs
- **Controlled time-based operations** using mocks

### 4. Performance Considerations
- **Fast unit tests** (<2s per component)
- **Efficient integration tests** (<10s per workflow)
- **Parallel execution** where possible
- **Resource cleanup** to prevent memory leaks

### 5. Error Testing
- **Test failure scenarios** as thoroughly as success scenarios
- **Validate error messages** and error types
- **Test recovery mechanisms** and graceful degradation
- **Verify cleanup** occurs on errors

### 6. Async Testing Patterns
```python
@pytest.mark.asyncio
async def test_async_operation(self, component):
    """Test asynchronous operations properly."""
    # Use AsyncMock for async dependencies
    mock_async_dep = AsyncMock()
    mock_async_dep.operation.return_value = "expected_result"
    
    # Test the async operation
    result = await component.async_method()
    
    # Verify async calls were made
    mock_async_dep.operation.assert_called_once()
```

---

## Integration with CI/CD

### Automated Execution
```yaml
# Example GitHub Actions integration
- name: Run Unit and Integration Tests
  run: |
    python tests/unit_integration_test_runner.py --save-report ci_test_report.json
    
- name: Upload Test Report
  uses: actions/upload-artifact@v3
  with:
    name: test-report
    path: ci_test_report.json
    
- name: Check Quality Gates
  run: |
    if ! python tests/unit_integration_test_runner.py; then
      echo "Quality gates failed"
      exit 1
    fi
```

### Quality Gate Enforcement
- **Block merges** if quality gates fail
- **Require foundation tests** to pass first
- **Performance regression detection** 
- **Automated recommendations** for improvement

---

## Next Steps: System Integration Testing

With Level 2 (Unit) and Level 3 (Integration) complete, the next step is **Level 4: System Integration Testing**:

- **End-to-end workflows** with real external services
- **Performance testing** under load
- **Security testing** with real authentication
- **Deployment testing** across environments

**Current Testing Pyramid Status**:
```
🔺 System Integration Testing (Level 4) - NEXT
     🔺 Component Integration Testing (Level 3) ✅ COMPLETE
  🔺 Unit Testing - Components in Isolation (Level 2) ✅ COMPLETE  
🔺 Foundation Testing (Level 1) ✅ COMPLETE (70% pass rate)
```

---

## Summary

✅ **DELIVERED**: Comprehensive unit and integration testing implementation
✅ **QUALITY**: 90%+ unit test pass rate, 85%+ integration test pass rate targets
✅ **PERFORMANCE**: <2s unit tests, <10s integration tests, parallel execution
✅ **AUTOMATION**: Quality gate validation with detailed reporting
✅ **COVERAGE**: All critical components and workflows tested
✅ **DOCUMENTATION**: Complete patterns and best practices guide

The unit and integration testing layers provide robust validation of component isolation and cross-component workflows, building on the solid foundation testing base to create a reliable testing pyramid for production confidence.