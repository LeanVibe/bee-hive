# CLAUDE.md - Core System Components

## 🎯 **Context: Core System Architecture**

You are working in the **core system architecture layer** of LeanVibe Agent Hive 2.0. This directory contains the fundamental building blocks that power autonomous multi-agent coordination.

## 🏗️ **System Architecture Principles**

### **Epic 1 Focus: Agent Orchestration Consolidation**
- **Critical Mission**: Consolidate 19+ orchestrator implementations into production-ready core
- **Primary Files**: `orchestrator.py`, `enhanced_orchestrator_integration.py`, `production_orchestrator.py`
- **Key Challenge**: Unified agent lifecycle management with <100ms response times

### **Performance Requirements**
- **Agent Registration**: <100ms per agent
- **Task Delegation**: <500ms for complex routing decisions  
- **Concurrent Agents**: Support 50+ simultaneous agents
- **Memory Efficiency**: <50MB base overhead per orchestrator instance

## 🔧 **Development Guidelines**

### **Code Organization Standards**
```python
# Preferred patterns for core components
from typing import Protocol, Optional, Dict, Any
from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager

class AgentProtocol(Protocol):
    """Clear interface definitions"""
    async def execute_task(self, task: Task) -> TaskResult: ...
    
class ProductionOrchestrator:
    """Consolidated orchestrator - single source of truth"""
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self._agents: Dict[str, AgentProtocol] = {}
        self._task_queue = asyncio.Queue()
        
    async def register_agent(self, agent: AgentProtocol) -> str:
        """<100ms registration requirement"""
```

### **Error Handling Patterns**
```python
# Mandatory error handling for core components
from app.core.circuit_breaker import CircuitBreaker
from app.core.retry_policies import exponential_backoff

@exponential_backoff(max_retries=3)
@CircuitBreaker(failure_threshold=5)
async def critical_operation(self, operation_data):
    try:
        result = await self.perform_operation(operation_data)
        return result
    except Exception as e:
        self.logger.error(f"Critical operation failed: {e}")
        await self.alert_system.notify_failure(e)
        raise
```

### **Testing Requirements**
- **Unit Tests**: 85%+ coverage for all new core components
- **Integration Tests**: Cross-component interaction validation
- **Performance Tests**: Validate all timing requirements
- **Circuit Breaker Tests**: Failure scenario validation

## 🚦 **Quality Gates Specific to Core**

### **Pre-Commit Requirements**
- All core components must pass performance benchmarks
- Memory leak detection via profiling
- Thread safety validation for concurrent operations
- Database connection pooling efficiency verification

### **Production Readiness Checklist**
- [ ] Circuit breaker patterns implemented
- [ ] Comprehensive error recovery mechanisms  
- [ ] Resource leak prevention validated
- [ ] Monitoring and alerting integration complete
- [ ] Graceful shutdown procedures tested

## ⚠️ **Critical Considerations**

### **Epic 1 Orchestration Consolidation Priority**
When working on orchestrator components:
1. **Backward Compatibility**: Maintain existing agent interfaces during transition
2. **Performance First**: Any change must maintain or improve response times
3. **Resource Management**: Careful memory and connection management
4. **State Management**: Ensure clean state transitions and recovery

### **Security Considerations**
- All inter-agent communication must be authenticated
- Resource limits enforced to prevent abuse
- Audit logging for all critical operations
- Input validation and sanitization mandatory

### **Monitoring Integration**
```python
# Required monitoring for core components
from app.observability import metrics, tracing

@metrics.timed("orchestrator.agent_registration")
@tracing.trace("agent_lifecycle")
async def register_agent(self, agent_spec: AgentSpec):
    with metrics.histogram("agent_registration_time"):
        # Implementation
        pass
```

## 🎯 **Success Criteria**

Your work in `/app/core/` is successful when:
- **Performance**: All timing requirements consistently met
- **Reliability**: 99.9% uptime under normal load
- **Scalability**: Handles 50+ concurrent agents without degradation
- **Maintainability**: Clear interfaces and comprehensive documentation
- **Production Ready**: Passes all Epic 1 consolidation requirements

## 🔄 **Integration Points**

### **Dependencies**
- `/app/models/`: Data model definitions
- `/app/schemas/`: API contract definitions  
- `/app/observability/`: Monitoring and alerting
- Database layer for persistence
- Redis for real-time communication

### **Consumers**
- `/app/api/`: REST API endpoints
- `/app/cli/`: Command-line interface
- Agent implementations across the system
- WebSocket real-time updates

## 📋 **Current Epic Status**

**Epic 1: Agent Orchestration Consolidation** - 70% Complete
- ✅ Core interfaces defined
- ✅ Basic orchestration working  
- 🔄 Performance optimization in progress
- ❌ Production consolidation pending
- ❌ Resource limit enforcement needed

Focus your efforts on **consolidation and production readiness** to advance Epic 1 toward completion.