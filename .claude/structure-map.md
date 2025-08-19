# LeanVibe Agent Hive 2.0 - Project Structure Map

## 🚀 Quick Navigation Guide for Claude Code

### 🎯 Most Important Files for Claude to Understand

#### 1. **Core Documentation**
- `docs/CLAUDE.md` - Claude-specific development guidelines
- `docs/ARCHITECTURE.md` - System architecture overview
- `README.md` - Project overview and getting started
- `docs/DEVELOPER_GUIDE.md` - Development workflow and practices

#### 2. **Entry Points & Configuration**
- `app/main.py` - FastAPI application entry point
- `start_hive.py` - Main system startup script
- `bee-hive-config.json` - Primary project configuration
- `project-index-installation-*/project-index-config.json` - Project index configuration

#### 3. **Core Orchestration System** 🤖
```
app/core/
├── orchestrator.py                    # Main orchestration engine
├── unified_orchestrator.py           # Unified orchestration logic
├── simple_orchestrator_enhanced.py   # Enhanced simple orchestrator
├── agent_manager.py                  # Agent lifecycle management
├── communication.py                  # Agent communication protocols
└── coordination.py                   # Multi-agent coordination
```

#### 4. **Project Index System** 📊 (Advanced AI-powered codebase analysis)
```
app/project_index/
├── core.py                           # Main indexing engine
├── analyzer.py                       # Code analysis logic
├── context_optimizer.py             # Context optimization for AI
├── debt_analyzer.py                  # Technical debt analysis
├── intelligent_detector.py          # AI-powered pattern detection
├── websocket_integration.py         # Real-time index updates
└── models.py                        # Project index data models
```

#### 5. **Database Models** 💾
```
app/models/
├── project_index.py                  # Project index database models
├── agent.py                          # Agent models
├── session.py                        # Session management models
├── task.py                          # Task tracking models
└── workflow.py                       # Workflow models
```

#### 6. **API Layer** 🌐
```
app/api/
├── project_index.py                  # Project index API endpoints
├── agent_coordination.py            # Agent coordination endpoints
├── dashboard_websockets.py          # WebSocket handlers
└── routes.py                        # Main API routes
```

### 🔧 Development Workflows

#### Common Claude Tasks:

1. **Adding New Agent Type:**
   - Update `app/agents/` with new agent implementation
   - Add agent configuration to `app/models/agent.py`
   - Update orchestrator in `app/core/orchestrator.py`

2. **Extending Project Index:**
   - Add analyzer to `app/project_index/analyzer.py`
   - Update models in `app/models/project_index.py`
   - Add API endpoints in `app/api/project_index.py`

3. **Creating New API Endpoints:**
   - Add endpoint to appropriate file in `app/api/`
   - Define schemas in `app/schemas/`
   - Add tests in `tests/`

4. **WebSocket Integration:**
   - Update `app/api/dashboard_websockets.py`
   - Add event handlers in `app/project_index/websocket_events.py`

### 🏗️ Architecture Layers

```
┌─────────────────────────────────────────────┐
│                Frontend                      │
│  mobile-pwa/ (TypeScript + Vite + Lit)     │
├─────────────────────────────────────────────┤
│                API Layer                     │
│  app/api/ (FastAPI + WebSockets)           │
├─────────────────────────────────────────────┤
│             Business Logic                   │
│  app/core/ (Orchestration & Services)      │
├─────────────────────────────────────────────┤
│            Agent System                      │
│  app/agents/ (Multi-Agent Coordination)    │
├─────────────────────────────────────────────┤
│          Project Intelligence               │
│  app/project_index/ (AI Analysis)          │
├─────────────────────────────────────────────┤
│             Data Layer                       │
│  app/models/ (SQLAlchemy + PostgreSQL)     │
├─────────────────────────────────────────────┤
│          Infrastructure                      │
│  Redis + Docker + Kubernetes               │
└─────────────────────────────────────────────┘
```

### 📁 Directory Priority Guide

**🔴 Critical (Always Consider):**
- `app/core/` - Core orchestration logic
- `app/project_index/` - AI-powered analysis system
- `app/api/` - API endpoints and WebSocket handlers
- `app/models/` - Database models

**🟡 High Priority:**
- `app/agents/` - Agent implementations
- `docs/` - Documentation and guides
- `tests/` - Test suite
- `mobile-pwa/` - Frontend application

**🟢 Medium Priority:**
- `scripts/` - Automation scripts
- `config/` - Configuration files
- `monitoring/` - Observability tools
- `migrations/` - Database migrations

### 🧠 Key Concepts for Claude

1. **Multi-Agent Orchestration:** The system coordinates multiple AI agents working together
2. **Project Index:** Advanced AI-powered codebase analysis and understanding
3. **Real-time Communication:** WebSocket-based real-time updates and coordination
4. **Context Optimization:** Smart context management for AI agents
5. **Technical Debt Analysis:** Automated code quality and debt tracking

### 🚦 Development Commands

```bash
# Start the system
python start_hive.py

# Run tests
pytest tests/

# Check project index
python project_index_server.py

# Validate system
python scripts/validate_system_integration.py

# Build and deploy
docker-compose up -d
```

### 🔍 Finding Things Quickly

**Looking for Agent Logic?** → `app/agents/` + `app/core/orchestrator*.py`  
**Looking for API Endpoints?** → `app/api/`  
**Looking for Database Stuff?** → `app/models/` + `migrations/`  
**Looking for Frontend?** → `mobile-pwa/`  
**Looking for Configuration?** → `*config*.json` + `docker-compose*.yml`  
**Looking for Documentation?** → `docs/` + `README.md`  
**Looking for Tests?** → `tests/`  

### 🎯 Claude-Specific Notes

- The project already has a sophisticated project index system - use it!
- WebSocket integration is extensive - real-time updates are key
- Multi-agent coordination is the core feature - understand the orchestration patterns
- Technical debt analysis is built-in - leverage it for code quality
- The system is designed for enterprise scale - consider performance implications
- Comprehensive testing framework exists - always add tests for new features

### 📊 Project Stats
- **Languages:** Python (60%), HTML (25%), TypeScript (11%), Shell (2%), JS (1%)
- **Total Files:** 3,225
- **Lines of Code:** 1,472,303  
- **Complexity:** Enterprise-scale
- **Architecture:** Event-driven microservices with multi-agent coordination