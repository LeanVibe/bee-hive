# Developer Resources

**Everything you need to build with LeanVibe Agent Hive 2.0**

You've chosen the developer path - here's your hands-on journey to autonomous development mastery.

## 🚀 Quick Start for Developers

### 1. Get Running (2-15 minutes)

**DevContainer (Recommended - <2 minutes):**
```bash
git clone https://github.com/LeanVibe/bee-hive.git
code bee-hive  # Opens in VS Code
# Click "Reopen in Container" when prompted
```

**Fast Setup (5-12 minutes):**
```bash
git clone https://github.com/LeanVibe/bee-hive.git && cd bee-hive
make setup
echo "ANTHROPIC_API_KEY=your_key_here" >> .env.local
make start
```

### 2. Validate Everything Works
```bash
make health                                          # System health
curl http://localhost:8000/health                    # API status
python scripts/demos/autonomous_development_demo.py  # See autonomous development
```

### 3. Explore the Architecture
- **API Docs**: http://localhost:8000/docs
- **System Health**: http://localhost:8000/health
- **Web Dashboard**: http://localhost:3000 (optional)

## 📚 Essential Developer Documentation

### Core Development
- **[Developer Guide](../DEVELOPER_GUIDE.md)** - Complete development guide
- **[API Reference](../API_REFERENCE_COMPREHENSIVE.md)** - Full API documentation
- **[Architecture Overview](../ENTERPRISE_SYSTEM_ARCHITECTURE.md)** - System architecture

### Multi-Agent Development
- **[Multi-Agent Coordination](../MULTI_AGENT_COORDINATION_GUIDE.md)** - Agent orchestration
- **[Agent Specialization](../AGENT_SPECIALIZATION_TEMPLATES.md)** - Custom agent creation
- **[Workflow Optimization](../WORKFLOW_ORCHESTRATION_OPTIMIZATION.md)** - Performance tuning

### Integration & Customization
- **[GitHub Integration](../GITHUB_INTEGRATION_API_COMPREHENSIVE.md)** - GitHub API integration
- **[Custom Commands](../CUSTOM_COMMANDS_USER_GUIDE.md)** - Extending functionality
- **[External Tools](../EXTERNAL_TOOLS_GUIDE.md)** - Tool integrations

## 🛠️ Development Workflow

### Daily Development Commands
```bash
# Start development environment
make start

# Run tests with coverage
make test

# Check code quality
make lint
make format

# Health monitoring
make health
```

### Database Operations
```bash
# Create migration
alembic revision --autogenerate -m "Your changes"

# Apply migrations
alembic upgrade head

# Database shell
psql postgresql://postgres:password@localhost:5432/agent_hive
```

### Multi-Agent Testing
```bash
# Test agent coordination  
python scripts/demos/autonomous_development_demo.py

# Test specific agent interactions
pytest tests/agents/test_coordination.py -v

# Performance benchmarks
python scripts/benchmarks/agent_performance.py
```

## 🔧 Customization Guide

### Creating Custom Agents
```python
from app.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="custom-agent",
            role="specialist",
            capabilities=["custom-task", "analysis"]
        )
    
    async def execute_task(self, task):
        # Your custom agent logic
        pass
```

### Adding New Endpoints
```python
from fastapi import APIRouter
from app.core.dependencies import get_db

router = APIRouter()

@router.post("/api/v1/custom")
async def custom_endpoint(db: Session = Depends(get_db)):
    # Your custom endpoint logic
    pass
```

### Extending Agent Capabilities
See [Agent Specialization Templates](../AGENT_SPECIALIZATION_TEMPLATES.md) for detailed examples.

## 🧪 Testing Strategy

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Multi-component interactions  
- **Agent Tests**: Multi-agent coordination scenarios
- **Performance Tests**: System performance validation

### Running Tests
```bash
# All tests
pytest

# Specific test categories
pytest tests/unit/ -v        # Unit tests
pytest tests/integration/ -v # Integration tests
pytest tests/agents/ -v      # Agent coordination tests

# With coverage
pytest --cov=app --cov-report=html
```

## 🔍 Debugging & Troubleshooting

### Debug Mode Setup
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Start with debug mode
uvicorn app.main:app --reload --log-level debug
```

### Common Development Issues
- **Port conflicts**: Use `lsof -i :8000` to find conflicting processes
- **Database issues**: Run `./troubleshoot.sh` for automated fixes
- **Agent coordination problems**: Check agent logs in the dashboard

### Debugging Tools
- **Interactive API**: http://localhost:8000/docs
- **Database browser**: Use pgAdmin or similar tools
- **Log monitoring**: `tail -f logs/app.log`
- **Agent tracing**: Web dashboard shows real-time agent activity

## 📈 Performance Optimization

### Monitoring
- **System metrics**: Available at http://localhost:8000/metrics
- **Agent performance**: Dashboard shows coordination efficiency
- **Database performance**: Built-in query monitoring

### Optimization Tips
1. **Agent Configuration**: Tune concurrent task limits per agent
2. **Database**: Ensure proper indexing for semantic searches
3. **Redis**: Monitor memory usage for message queues
4. **API**: Use async/await patterns consistently

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Set up development environment using DevContainer
3. Create feature branch: `git checkout -b feature/your-feature`
4. Make changes with tests
5. Submit pull request

### Code Standards
- **Python**: Follow PEP 8, use Black formatting
- **API**: RESTful design, comprehensive OpenAPI docs
- **Tests**: Minimum 90% coverage for new code
- **Documentation**: Update relevant docs with changes

## 🆘 Need Help?

### Development Support
- **Technical Questions**: [GitHub Discussions](https://github.com/LeanVibe/bee-hive/discussions)
- **Bug Reports**: [GitHub Issues](https://github.com/LeanVibe/bee-hive/issues)
- **Troubleshooting**: [Complete Guide](../TROUBLESHOOTING_GUIDE_COMPREHENSIVE.md)

### Quick Fixes
```bash
# Reset everything
./setup-fast.sh

# Fix common issues
./troubleshoot.sh

# Health diagnostics
./health-check.sh
```

---

**Ready to build autonomous development tools? Start with the quick setup above and dive into the documentation!**