# Epic 1.5 Manager Consolidation: Implementation Roadmap

## Phase-by-Phase File Migration Plan

### Phase 1: Foundation Setup (Weeks 1-2)

#### Core Manager Skeleton Creation
```
/app/core/managers/
├── workflow_manager.py          # New - WorkflowManager core
├── agent_manager.py            # New - AgentManager core  
├── resource_manager.py         # New - ResourceManager core
├── configuration_manager.py    # New - ConfigurationManager core
├── infrastructure_manager.py   # New - InfrastructureManager core
├── __init__.py                 # Manager exports
└── base/
    ├── manager_base.py         # Base manager interface
    ├── dependency_injection.py # DI framework
    └── shared_utilities.py     # Common utilities
```

#### Shared Infrastructure
- Abstract base classes for all managers
- Dependency injection framework
- Common logging and error handling
- Shared configuration patterns
- Test fixtures and mocking framework

---

### Phase 2: ResourceManager & ConfigurationManager (Weeks 3-4)

#### ResourceManager Migration
**Target File**: `/app/core/managers/resource_manager.py`

**Files to Consolidate** (20 files → 1):
```
# Resource Domain (7 files)
capacity_manager.py                 → ResourceManager.capacity_planner
backpressure_manager.py            → ResourceManager.load_balancer  
enterprise_backpressure_manager.py → ResourceManager.load_balancer
enhanced_failure_recovery_manager.py → ResourceManager.health_checker
context_cache_manager.py           → ResourceManager.cache_manager
workspace_manager.py               → ResourceManager.workspace_controller
tmux_session_manager.py           → ResourceManager.session_manager

# Monitoring Domain (13 files)  
intelligent_sleep_manager.py       → ResourceManager.sleep_controller
sleep_wake_manager.py              → ResourceManager.sleep_controller
enterprise_consumer_group_manager.py → ResourceManager.consumer_manager
enhanced_redis_streams_manager.py  → ResourceManager.stream_manager
redis_pubsub_manager.py            → ResourceManager.pubsub_manager
pgvector_manager.py                → ResourceManager.vector_manager
optimized_pgvector_manager.py      → ResourceManager.vector_manager
enhanced_git_checkpoint_manager.py → ResourceManager.checkpoint_manager
consumer_group_coordinator.py      → ResourceManager.consumer_manager
index_management.py                → ResourceManager.index_manager
... (3 additional monitoring files)
```

**New Structure**:
```python
class ResourceManager:
    def __init__(self):
        self.capacity_planner = CapacityPlanner()
        self.load_balancer = LoadBalancer()
        self.health_checker = HealthChecker()
        self.performance_monitor = PerformanceMonitor()
        self.cache_manager = CacheManager()
        
    # Unified resource allocation interface
    async def allocate_resources(self, agent_id, requirements)
    async def monitor_health(self, component_id)
    async def handle_backpressure(self, queue_id, threshold)
```

#### ConfigurationManager Migration  
**Target File**: `/app/core/managers/configuration_manager.py`

**Files to Consolidate** (5 files → 1):
```
feature_flag_manager.py         → ConfigurationManager.feature_engine
secret_manager.py              → ConfigurationManager.vault
api_key_manager.py             → ConfigurationManager.vault  
enhanced_jwt_manager.py        → ConfigurationManager.auth_system
enterprise_secrets_manager.py  → ConfigurationManager.vault
```

**New Structure**:
```python
class ConfigurationManager:
    def __init__(self):
        self.feature_engine = FeatureFlagEngine()
        self.vault = SecurityVault()
        self.auth_system = AuthenticationSystem()
        self.config_engine = ConfigurationEngine()
        
    # Unified configuration interface
    async def get_feature_flag(self, flag_name, context)
    async def get_secret(self, secret_name, agent_id) 
    async def validate_token(self, token)
```

---

### Phase 3: InfrastructureManager (Weeks 5-6)

#### InfrastructureManager Migration
**Target File**: `/app/core/managers/infrastructure_manager.py`

**Files to Consolidate** (9 files → 1):
```
branch_manager.py                   → InfrastructureManager.git_controller
work_tree_manager.py               → InfrastructureManager.workspace_orchestrator
enhanced_git_checkpoint_manager.py → InfrastructureManager.git_controller
self_modification_git_manager.py   → InfrastructureManager.git_controller
version_control_manager.py         → InfrastructureManager.git_controller
advanced_repository_management.py  → InfrastructureManager.git_controller
tmux_session_manager.py           → InfrastructureManager.session_manager
enterprise_tmux_manager.py        → InfrastructureManager.session_manager
international_operations_management.py → InfrastructureManager.deployment_engine
```

**New Structure**:
```python
class InfrastructureManager:
    def __init__(self):
        self.git_controller = GitController()
        self.workspace_orchestrator = WorkspaceOrchestrator()
        self.session_manager = SessionManager()
        self.deployment_engine = DeploymentEngine()
        
    # Unified infrastructure interface
    async def create_branch(self, agent_id, branch_name)
    async def isolate_workspace(self, agent_id, project_path)
    async def create_session(self, agent_id, environment)
```

---

### Phase 4: AgentManager & WorkflowManager (Weeks 7-8)

#### AgentManager Migration
**Target File**: `/app/core/managers/agent_manager.py`

**Files to Consolidate** (7 files → 1):
```
agent_lifecycle_manager.py      → AgentManager.lifecycle_controller
agent_knowledge_manager.py      → AgentManager.knowledge_engine
cross_agent_knowledge_manager.py → AgentManager.knowledge_engine
memory_hierarchy_manager.py     → AgentManager.memory_manager
enhanced_memory_manager.py      → AgentManager.memory_manager
context_manager.py             → AgentManager.memory_manager
context_memory_manager.py      → AgentManager.memory_manager
context_lifecycle_manager.py   → AgentManager.memory_manager
```

**New Structure**:
```python
class AgentManager:
    def __init__(self):
        self.lifecycle_controller = LifecycleController()
        self.knowledge_engine = KnowledgeEngine()
        self.memory_manager = MemoryManager()
        self.persona_system = PersonaSystem()
        
    # Unified agent management interface
    async def register_agent(self, agent_spec)
    async def share_knowledge(self, from_agent, to_agent, knowledge)
    async def compress_context(self, agent_id, context)
```

#### WorkflowManager Migration
**Target File**: `/app/core/managers/workflow_manager.py`

**Files to Consolidate** (8 files → 1):
```
workflow_state_manager.py      → WorkflowManager.state_manager
workflow_context_manager.py    → WorkflowManager.context_manager  
issue_manager.py              → WorkflowManager.task_orchestrator
checkpoint_manager.py         → WorkflowManager.checkpoint_engine
recovery_manager.py           → WorkflowManager.checkpoint_engine
enhanced_state_manager.py     → WorkflowManager.state_manager
chat_transcript_manager.py    → WorkflowManager.communication_bridge
enterprise_pilot_manager.py   → WorkflowManager.task_orchestrator
```

**New Structure**:
```python
class WorkflowManager:
    def __init__(self):
        self.task_orchestrator = TaskOrchestrator()
        self.state_manager = StateManager()
        self.checkpoint_engine = CheckpointEngine()
        self.communication_bridge = CommunicationBridge()
        
    # Unified workflow interface
    async def execute_workflow(self, workflow_spec)
    async def create_checkpoint(self, workflow_id)
    async def recover_workflow(self, workflow_id, checkpoint_id)
```

---

### Phase 5: Integration & Testing (Weeks 9-10)

#### Legacy Manager Deprecation
**Files to Remove** (46 total):
```
# All existing manager files will be deprecated
*_manager.py              → Deprecated (functionality migrated)
*_management.py           → Deprecated (functionality migrated) 
*_coordinator.py          → Deprecated (functionality migrated)
```

#### New Manager Integration
```
# Updated import paths throughout codebase
from app.core.managers import (
    WorkflowManager,
    AgentManager, 
    ResourceManager,
    ConfigurationManager,
    InfrastructureManager
)

# Dependency injection setup
from app.core.managers.base import ManagerRegistry
managers = ManagerRegistry()
workflow_mgr = managers.get(WorkflowManager)
```

#### Backwards Compatibility Layer
```python
# /app/core/legacy_compatibility.py
# Temporary shims for existing code during transition

class LegacyManagerAdapter:
    """Provides backwards compatibility for legacy manager imports"""
    
    def __getattr__(self, name):
        # Route legacy calls to new managers
        if name.endswith('_manager'):
            return self._get_legacy_adapter(name)
        raise AttributeError(f"No legacy adapter for {name}")
```

---

## File Impact Summary

### Files Being Consolidated
| Phase | Files In | Files Out | Reduction |
|-------|----------|-----------|-----------|
| Phase 2 | 25 files | 2 files | 92% |
| Phase 3 | 9 files | 1 file | 89% |
| Phase 4 | 15 files | 2 files | 87% |
| **Total** | **49 files** | **5 files** | **90%** |

### Directory Structure After Consolidation
```
/app/core/
├── managers/
│   ├── workflow_manager.py      # 🆕 Consolidated workflow management
│   ├── agent_manager.py         # 🆕 Consolidated agent management  
│   ├── resource_manager.py      # 🆕 Consolidated resource management
│   ├── configuration_manager.py # 🆕 Consolidated configuration
│   ├── infrastructure_manager.py# 🆕 Consolidated infrastructure
│   └── base/
│       ├── manager_base.py      # 🆕 Base manager interface
│       └── dependency_injection.py # 🆕 DI framework
├── legacy_compatibility.py     # 🆕 Backwards compatibility
├── [46 legacy manager files]   # ❌ To be removed in Phase 5
└── ... (rest of core modules)
```

### Testing Strategy Per Phase
```
Phase 2: Resource & Configuration
├── Unit tests for ResourceManager components
├── Integration tests for ConfigurationManager  
├── Performance benchmarks for resource allocation
└── Security tests for secret management

Phase 3: Infrastructure  
├── Git operation integration tests
├── Workspace isolation verification
├── Session management tests
└── Deployment automation tests

Phase 4: Agent & Workflow
├── Agent lifecycle end-to-end tests
├── Knowledge sharing integration tests
├── Workflow execution performance tests  
└── Context compression validation

Phase 5: Integration
├── Full system integration tests
├── Legacy compatibility verification
├── Performance regression testing
└── Production readiness validation
```

## Success Criteria Per Phase

### Phase 2 Success Metrics
- ✅ ResourceManager handles all capacity planning
- ✅ ConfigurationManager serves all feature flags and secrets
- ✅ Zero performance regression in resource allocation
- ✅ All security tests pass

### Phase 3 Success Metrics  
- ✅ InfrastructureManager handles all Git operations
- ✅ Workspace isolation works for all agents
- ✅ Session management scales to 100+ concurrent agents
- ✅ Deployment automation is fully functional

### Phase 4 Success Metrics
- ✅ AgentManager handles full agent lifecycles
- ✅ WorkflowManager executes all workflow types
- ✅ Knowledge sharing works across all agent types
- ✅ Context compression maintains quality

### Phase 5 Success Metrics
- ✅ Zero legacy manager dependencies remain
- ✅ All existing functionality preserved
- ✅ Performance meets or exceeds baseline
- ✅ Documentation is complete

This roadmap provides a clear, executable path to consolidate 49 manager files into 5 core managers while maintaining system functionality and enabling continuous integration throughout the process.