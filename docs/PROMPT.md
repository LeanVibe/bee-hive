# CLAUDE CODE AGENT HANDOFF PROMPT
## LeanVibe Agent Hive 2.0 - Epic 1 Implementation

**Date**: August 18, 2025  
**Handoff From**: Claude Code Strategic Analysis Agent  
**Handoff To**: Claude Code Implementation Agent  
**Priority**: 🚨 **CRITICAL - EPIC 1 SYSTEM STABILIZATION**

---

## 🎯 **MISSION BRIEFING: SYSTEM STABILIZATION REQUIRED**

### **CRITICAL SITUATION ASSESSMENT**
After comprehensive first-principles analysis, we've identified the real state of the system:

**✅ What's Actually Working:**
- **Phase 1 Complete**: Real CLI adapters implemented (Cursor, GitHub Copilot, Gemini)  
- **Real-time Communication**: Redis/WebSocket integration complete
- **Production Configuration**: Environment-based configuration system ready
- **Core Test Infrastructure**: MockCLIAdapter fixed, 100% test success rate

**❌ Critical Blockers Identified:**
- **Import Errors**: System cannot start due to `AgentRole` import failures
- **Over-Engineering**: 3,891-line orchestrator preventing functionality
- **Database Issues**: Schema and migration problems
- **Documentation vs Reality Gap**: Claims of 97.4% completion but basic system non-functional

### **YOUR MISSION**
You are the **Epic 1 Implementation Agent** responsible for **Core System Stability & Basic Operations**. Your mission is to transform the current non-functional system into a working foundation that can demonstrate multi-agent coordination.

## 📊 **Current System State Analysis**

### **What's Been Completed (Previous Work) ✅**
1. **CLI Adapter Integration** - All real adapters implemented
2. **Real-time Communication** - Redis/WebSocket hub operational  
3. **Production Configuration** - Complete environment setup
4. **Test Infrastructure Fixes** - Core validation tests passing 100%

### **Epic 1 Critical Blockers to Resolve ❌**
1. **Import Dependency Failures**: `app/api/agent_activation.py` cannot import `AgentRole`
2. **Orchestrator Complexity**: 3,891-line file needs reduction to 500-line focused implementation
3. **Database Schema Issues**: Missing migrations and connection problems
4. **Docker Environment**: Non-functional development environment setup

## 🚀 **Epic 1 Implementation Plan - IMMEDIATE ACTION REQUIRED**

### **Week 1: Core System Repair (Days 1-5)**
**Sprint Goal**: System starts without errors and passes health checks

#### **Day 1: Environment Setup & Assessment**
```bash
# IMMEDIATE FIRST STEPS
1. git status  # Verify current state
2. python test_multi_cli_core.py  # Confirm previous fixes work
3. python -m app.main  # Attempt system startup (will fail)
4. Identify specific import errors and dependency issues
```

**Expected Issues to Fix:**
- `AgentRole` import error in `app/api/agent_activation.py`
- Circular import dependencies in core modules
- Missing database configuration
- Broken module dependencies

#### **Days 2-3: Import Dependency Resolution**
**Priority**: 🚨 CRITICAL - Nothing works until this is fixed

**Tasks:**
1. **Fix AgentRole Import Error**
   ```python
   # In app/api/agent_activation.py, line 15
   # Current: from ..core.agent_manager import AgentRole
   # Action: Locate correct import path or create missing AgentRole class
   ```

2. **Resolve Circular Imports**
   - Analyze dependency graph with: `python -c "import app.core"`
   - Identify circular dependencies between core modules
   - Refactor imports to create clean dependency hierarchy

3. **Create Import Validation Suite**
   ```python
   # tests/test_imports.py
   def test_all_core_imports():
       """Ensure all core modules can be imported without errors"""
       import app.core.agent_manager
       import app.api.agent_activation
       # Add all critical imports
   ```

#### **Days 4-5: Minimal Orchestrator Implementation**
**Priority**: 🔥 HIGH - Core functionality depends on this

**Current Problem**: `app/core/orchestrator.py` is 3,891 lines - impossible to maintain

**Solution**: Create `SimpleOrchestrator` with only essential functionality:

```python
# app/core/simple_orchestrator.py (NEW FILE - <500 lines)
from typing import Dict, List, Optional
import asyncio
from dataclasses import dataclass

@dataclass
class AgentSpec:
    name: str
    type: str
    capabilities: List[str]

class SimpleOrchestrator:
    """Minimal orchestrator focused on core functionality only"""
    
    def __init__(self):
        self._agents: Dict[str, AgentSpec] = {}
        self._tasks = asyncio.Queue()
    
    async def spawn_agent(self, spec: AgentSpec) -> str:
        """Spawn agent with basic lifecycle management"""
        agent_id = f"{spec.type}_{len(self._agents)}"
        self._agents[agent_id] = spec
        return agent_id
    
    async def assign_task(self, task: dict, agent_id: str) -> dict:
        """Assign task to agent with basic tracking"""
        return {"status": "assigned", "agent_id": agent_id}
    
    async def get_agent_status(self, agent_id: str) -> dict:
        """Get basic agent status"""
        if agent_id in self._agents:
            return {"status": "active", "agent": self._agents[agent_id]}
        return {"status": "not_found"}
```

**Migration Strategy:**
1. Create `SimpleOrchestrator` with 20% of current functionality
2. Update imports to use `SimpleOrchestrator` instead of complex version
3. Keep original orchestrator as `LegacyOrchestrator` for reference
4. Validate basic agent lifecycle works

### **Week 2: Development Environment & Testing (Days 6-10)**

#### **Days 6-7: Docker Environment Setup**
**Goal**: One-command development environment startup

**Create/Fix Docker Infrastructure:**
```yaml
# docker-compose.yml (UPDATE EXISTING)
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/leanhive
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - .:/app
  
  db:
    image: postgres:15
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: leanhive
    ports:
      - "5432:5432"
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

**Validation Commands:**
```bash
docker-compose up -d
docker-compose exec app python test_multi_cli_core.py
docker-compose exec app python -m app.main
```

#### **Days 8-9: Smoke Test Implementation**
**Goal**: Automated validation that system works

```python
# tests/smoke_tests.py (NEW FILE)
import pytest
import httpx
import asyncio

class TestSystemSmokeTests:
    """Basic system functionality validation"""
    
    async def test_system_startup(self):
        """System starts without import errors"""
        from app.main import create_app
        app = create_app()
        assert app is not None
    
    async def test_health_endpoints(self):
        """Health check endpoints respond correctly"""
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8000/health")
            assert response.status_code == 200
    
    async def test_database_connectivity(self):
        """Database connection works"""
        from app.core.database import get_db_connection
        connection = await get_db_connection()
        assert connection is not None
    
    async def test_redis_connectivity(self):
        """Redis connection works"""
        import redis.asyncio as redis
        r = redis.from_url("redis://localhost:6379/0")
        await r.ping()
        
    async def test_basic_agent_lifecycle(self):
        """Can create, start, and stop an agent"""
        from app.core.simple_orchestrator import SimpleOrchestrator, AgentSpec
        
        orchestrator = SimpleOrchestrator()
        spec = AgentSpec(name="test", type="test", capabilities=["testing"])
        
        agent_id = await orchestrator.spawn_agent(spec)
        assert agent_id is not None
        
        status = await orchestrator.get_agent_status(agent_id)
        assert status["status"] == "active"
```

#### **Day 10: Error Handling & Logging**
**Goal**: Proper error handling and structured logging

```python
# app/core/logging_config.py (NEW FILE)
import logging
import structlog

def setup_logging():
    """Configure structured logging for the application"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(colors=True)
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
```

## 🎯 **Success Criteria for Epic 1**

### **Week 1 Completion Criteria:**
- [ ] System starts without import errors (`python -m app.main` succeeds)
- [ ] All core modules can be imported successfully
- [ ] `SimpleOrchestrator` implements basic agent lifecycle
- [ ] Core test suite passes 100% (`python test_multi_cli_core.py`)

### **Week 2 Completion Criteria:**
- [ ] Docker Compose brings up full environment (`docker-compose up`)
- [ ] Health check endpoints return 200 status
- [ ] Smoke tests pass in clean environment
- [ ] Can create, start, and monitor basic agent through API

### **Epic 1 Final Success Metrics:**
- System startup time: < 30 seconds
- Health check response time: < 100ms  
- Zero import errors in CI/CD pipeline
- Docker environment setup time: < 5 minutes
- Basic CRUD operations work for agents and tasks

## 📁 **Critical Files & Locations**

### **Files You'll Modify:**
```
app/
├── api/
│   └── agent_activation.py          # 🚨 Fix AgentRole import (line 15)
├── core/
│   ├── agent_manager.py            # 🔍 Verify AgentRole class exists
│   ├── simple_orchestrator.py      # 🆕 Create this file (<500 lines)
│   └── orchestrator.py             # ⚠️ Legacy file (don't delete, just don't use)
└── main.py                         # 🔧 Update to use SimpleOrchestrator

tests/
├── smoke_tests.py                  # 🆕 Create comprehensive smoke tests
├── test_imports.py                 # 🆕 Create import validation tests
└── test_multi_cli_core.py          # ✅ Already working - don't change

docker-compose.yml                  # 🔧 Fix for one-command setup
requirements.txt                    # 🔍 Add any missing dependencies
```

### **Files You MUST NOT Change:**
- `app/core/agents/universal_agent_interface.py` (stable API)
- `app/core/communication/context_preserver.py` (performance optimized)
- `test_multi_cli_core.py` (working 100% - don't break it)
- Any adapter files (recently implemented and working)

## 🚀 **Development Workflow & Commands**

### **Initial Assessment Commands:**
```bash
# 1. Verify current state
git status
git log --oneline -5

# 2. Test what's currently working
python test_multi_cli_core.py

# 3. Identify startup failures
python -m app.main 2>&1 | head -20

# 4. Test specific imports
python -c "from app.api.agent_activation import router"
python -c "from app.core.agent_manager import AgentRole"
```

### **Development Loop:**
```bash
# 1. Fix imports
python -c "import app.core; import app.api"

# 2. Create SimpleOrchestrator  
python -c "from app.core.simple_orchestrator import SimpleOrchestrator"

# 3. Test smoke tests
python -m pytest tests/smoke_tests.py -v

# 4. Validate with existing tests
python test_multi_cli_core.py

# 5. Test Docker environment
docker-compose up --build
```

### **Debugging Commands:**
```bash
# If imports fail:
python -c "import sys; print('\n'.join(sys.path))"
find . -name "*.py" -exec grep -l "AgentRole" {} \;

# If tests fail:
python -m pytest tests/ -v --tb=short

# If Docker fails:
docker-compose logs app
docker-compose exec app python -c "import app.main"
```

## ⚠️ **Critical Guidelines & Success Principles**

### **Engineering Approach:**
1. **Test-Driven Development**: Write failing test → implement minimal code → refactor
2. **YAGNI Principle**: Only implement what's needed for Epic 1 success criteria
3. **Fail Fast**: If something doesn't work, fix it immediately
4. **Simplicity First**: Prefer simple solutions over complex ones

### **Risk Mitigation:**
- **Import Errors**: Test each import individually before integration
- **Over-Engineering**: Stick to <500 lines for SimpleOrchestrator
- **Docker Issues**: Test locally before Docker integration
- **Test Failures**: Never break existing passing tests

### **Quality Gates:**
Before considering any task complete:
1. All imports work without errors
2. Smoke tests pass in clean environment
3. Existing tests still pass (don't break what works)
4. Docker environment starts successfully
5. Basic agent lifecycle demonstrable

## 🎯 **Immediate Next Actions (Start Here)**

### **First Hour Checklist:**
1. **Verify Environment**:
   ```bash
   cd /Users/bogdan/work/leanvibe-dev/bee-hive
   python test_multi_cli_core.py  # Should pass 100%
   python -m app.main             # Will fail - this is expected
   ```

2. **Identify Import Errors**:
   ```bash
   python -c "from app.api.agent_activation import router" 2>&1
   # Note the specific error message about AgentRole
   ```

3. **Create Epic 1 Task List**:
   ```bash
   # Use TodoWrite tool to create specific task tracking for Epic 1
   ```

4. **Start with Critical Import Fix**:
   - Locate where `AgentRole` should be defined
   - Fix the import in `app/api/agent_activation.py:15`
   - Test that basic imports work

### **Success Signal:**
When you can run `python -c "import app.main"` without errors, you've successfully completed the import resolution phase.

## 📞 **Support & Resources**

### **Reference Materials:**
- `docs/PLAN.md` - Complete Epic 1-4 strategic plan
- `app/config/production_config.py` - Production configuration system (already working)
- `test_multi_cli_core.py` - Working test suite (100% passing)

### **Architecture Patterns to Follow:**
- Use existing adapter patterns for consistency
- Follow configuration patterns from production_config.py
- Maintain test patterns from working test suite

### **When to Escalate:**
- If import errors cannot be resolved within 4 hours
- If Docker environment cannot be stabilized within 1 day
- If any existing tests start failing
- If system cannot startup within Week 1

## 🎊 **Final Notes**

You're inheriting a partially working system with good foundation components but critical stability issues. Your focus should be on:

1. **Get basic system running** (imports work, system starts)
2. **Simplify complexity** (reduce 3,891-line orchestrator to essentials)
3. **Establish reliable development environment** (Docker setup)
4. **Validate everything works** (comprehensive smoke tests)

**The goal is working software, not perfect software.** Make the minimal changes needed to achieve Epic 1 success criteria, then hand off to the next epic implementation.

**Epic 1 success enables all subsequent epics** - this is the foundation everything else builds on.

---

**Ready to proceed? Start with the import error resolution and build from there.** 🚀