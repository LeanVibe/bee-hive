# 🛠️ Developer Fast Track

**Your goal**: Get hands-on with autonomous development quickly, understand the architecture, and start customizing.

**Time commitment**: 10 minutes to working system, 30 minutes to full understanding

---

## 🎯 What You'll Achieve

By the end of this path, you'll have:
- ✅ Working autonomous development system on your machine
- ✅ Understanding of the multi-agent architecture
- ✅ Ability to customize and extend the platform
- ✅ Knowledge of key APIs and integration points

---

## Stage 1: Quick Win (10 minutes)

### Get It Running Locally

**Goal**: See autonomous development working on your machine

```bash
# 1. Clone and setup (one command handles everything)
git clone https://github.com/LeanVibe/bee-hive.git && cd bee-hive
./setup-fast.sh

# 2. Add your API key (required for AI agents)
echo "ANTHROPIC_API_KEY=your_key_here" >> .env.local

# 3. Start the system
./start-fast.sh

# 4. Validate everything is running
./health-check.sh
```

**Success criteria**: 
- Health check shows all services green
- API docs accessible at http://localhost:8000/docs
- No error messages in logs

### See Autonomous Development in Action

```bash
# Watch AI agents coordinate to build complete features
python scripts/demos/autonomous_development_demo.py
```

**What you're seeing**:
- 🧠 **Architect agent** analyzing requirements and creating implementation plans
- 💻 **Developer agent** writing code based on the architecture
- 🧪 **Tester agent** creating and running comprehensive tests
- 👀 **Reviewer agent** checking code quality and suggesting improvements
- 🔄 **Coordination** between agents sharing context and knowledge

**✅ Stage 1 Complete**: You now have working autonomous development!

---

## Stage 2: Architecture Understanding (20 minutes)

### Multi-Agent Coordination Deep Dive

**Goal**: Understand how the agents work together

The system uses specialized AI agents that coordinate through:

```
┌─────────────────────────────────────────────────────────────┐
│                 Agent Coordination Layer                    │
├─────────────────┬─────────────────┬─────────────────────────┤
│   AI Agents     │   Task Router   │    Context Memory       │
│                 │                 │                         │
│ • Architect     │ • Load Balance  │  • Project Knowledge    │
│ • Developer     │ • Priority      │  • Code Context         │
│ • Tester        │ • Capability    │  • Decision History     │
│ • Reviewer      │ • Health        │  • Learning Data        │
└─────────────────┴─────────────────┴─────────────────────────┘
         │                    │                      │
┌─────────▼────────┐ ┌────────▼──────┐ ┌───────────▼─────────┐
│   FastAPI Core   │ │ Redis Streams │ │ PostgreSQL+pgvector │
│                  │ │               │ │                     │
│ • REST API       │ │ • Message Bus │ │ • Persistent State  │
│ • WebSocket      │ │ • Real-time   │ │ • Vector Search     │
│ • Health Checks  │ │ • Pub/Sub     │ │ • Context Storage   │
└──────────────────┘ └───────────────┘ └─────────────────────┘
```

### Key Components to Understand

#### 1. Agent Orchestrator (`app/core/orchestrator.py`)
- **Coordinates** all agent activities
- **Distributes** tasks based on agent capabilities
- **Monitors** agent health and performance
- **Handles** error recovery and retry logic

#### 2. Context Engine (`app/core/context_engine.py`)
- **Stores** project knowledge using pgvector embeddings
- **Provides** relevant context to agents for decision making
- **Learns** from agent interactions and outcomes
- **Maintains** conversation history and decision trails

#### 3. Communication Bus (`app/core/redis.py`)
- **Real-time messaging** between agents via Redis Streams
- **Event-driven** coordination and status updates
- **Pub/sub** for system-wide notifications
- **Message persistence** for reliability

#### 4. Agent Implementations (`app/core/`)
- Each agent type has specialized capabilities and knowledge
- Agents communicate through structured messages
- Context sharing enables intelligent collaboration
- Self-healing through automatic retry and escalation

### Explore the Code

```bash
# Look at the main orchestrator logic
cat app/core/orchestrator.py | head -50

# See how agents communicate
cat app/core/communication.py | head -30

# Understand the context memory system
cat app/core/context_engine.py | head -40
```

**✅ Stage 2 Complete**: You understand the architecture!

---

## Stage 3: Customization & Extension (varies)

### Add Your Own Agent Type

**Goal**: Create a specialized agent for your domain

1. **Define Agent Capabilities**:
```python
# app/schemas/agent.py
class CustomAgentCreate(BaseModel):
    name: str = "domain-expert"
    role: str = "specialist"
    capabilities: List[str] = ["domain-knowledge", "validation"]
    persona: str = "Expert in your specific domain"
```

2. **Implement Agent Logic**:
```python
# app/core/custom_agent.py
class CustomAgent(BaseAgent):
    async def process_task(self, task: Task) -> TaskResult:
        # Your custom logic here
        return await self.coordinate_with_other_agents(task)
```

3. **Register with Orchestrator**:
```python
# Register your agent type with the system
orchestrator.register_agent_type("domain-expert", CustomAgent)
```

### Connect External Tools

**Goal**: Integrate with your existing development workflow

```python
# app/core/external_tools.py
class GitHubIntegration:
    async def create_pull_request(self, code: str, description: str):
        # Automatic PR creation from agent output
        
class JiraIntegration:
    async def update_ticket_status(self, ticket_id: str, status: str):
        # Sync agent progress with project management
        
class SlackIntegration:
    async def notify_team(self, message: str):
        # Team notifications for important events
```

### Custom Workflows

**Goal**: Define your own autonomous development workflows

```python
# app/workflow/custom_workflow.py
class FeatureDevelopmentWorkflow:
    stages = [
        "requirements_analysis",    # Architect agent
        "technical_design",        # Architect + Developer agents
        "implementation",          # Developer agent
        "testing",                # Tester agent
        "code_review",            # Reviewer agent
        "integration",            # All agents coordinate
        "deployment_prep"         # DevOps integration
    ]
```

### API Integration Points

**Key endpoints for custom integrations**:

```bash
# Create custom agent
curl -X POST "http://localhost:8000/api/v1/agents/" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-agent", "role": "specialist", "capabilities": ["custom"]}'

# Assign custom task
curl -X POST "http://localhost:8000/api/v1/tasks/" \
  -H "Content-Type: application/json" \
  -d '{"title": "Custom task", "agent_id": "agent-uuid"}'

# Monitor via WebSocket
wscat -c ws://localhost:8000/ws/observability
```

**✅ Stage 3 Complete**: You can customize and extend the platform!

---

## 🚀 Next Steps Based on Your Goals

### I Want to Build Production Features
→ **[Production Deployment Guide](../PRODUCTION_DEPLOYMENT_RUNBOOK.md)** - Scale to production

### I Want to Contribute to the Project
→ **[Contributing Guide](../../CONTRIBUTING.md)** - Join the development community

### I Want Enterprise Features
→ **[Enterprise Assessment Path](ENTERPRISE_PATH.md)** - Security, compliance, scalability

### I Want to Understand the Business Value
→ **[Executive Brief Path](EXECUTIVE_PATH.md)** - ROI analysis and strategic positioning

### I Need Troubleshooting Help
→ **[Troubleshooting Guide](../TROUBLESHOOTING_GUIDE_COMPREHENSIVE.md)** - Comprehensive problem solving

---

## 📚 Developer Resources

### Essential Documentation
- **[API Reference](../API_REFERENCE_COMPREHENSIVE.md)** - Complete API documentation
- **[Developer Guide](../DEVELOPER_GUIDE.md)** - Comprehensive development guide
- **[Hook Integration](../HOOK_INTEGRATION_GUIDE.md)** - System integration patterns
- **[Multi-Agent Coordination](../MULTI_AGENT_COORDINATION_GUIDE.md)** - Advanced patterns

### Code Examples
- **Autonomous Development Demo**: `scripts/demos/autonomous_development_demo.py`
- **Custom Agent Example**: `app/core/real_agent_implementations.py`
- **Workflow Examples**: `app/workflow/semantic_nodes.py`
- **Integration Tests**: `tests/test_enhanced_orchestrator_comprehensive.py`

### Development Tools
```bash
# Test your changes
pytest -v --cov=app

# Code quality checks
black app/ tests/
ruff check app/ tests/

# Performance validation
./validate-setup-performance.sh

# System health monitoring
./health-check.sh
```

---

## 🎯 Success Validation

**Check your progress**:

- [ ] ✅ Autonomous development demo runs successfully
- [ ] ✅ All health checks pass
- [ ] ✅ You understand the agent coordination model
- [ ] ✅ You can create custom agents
- [ ] ✅ You can integrate external tools
- [ ] ✅ You can define custom workflows

**Congratulations!** You're now ready to build autonomous development solutions with LeanVibe Agent Hive 2.0.

---

**Need help?** → [GitHub Discussions](https://github.com/LeanVibe/bee-hive/discussions) | [Issues](https://github.com/LeanVibe/bee-hive/issues)

---

*Part of the [LeanVibe Agent Hive 2.0](../../WELCOME.md) documentation system*