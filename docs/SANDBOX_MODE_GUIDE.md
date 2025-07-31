# LeanVibe Sandbox Mode - Complete Guide

## Overview

LeanVibe's Sandbox Mode provides a complete autonomous development demonstration without requiring any API keys or external services. It's designed to eliminate friction and enable immediate evaluation of our autonomous AI development capabilities.

## 🏖️ What is Sandbox Mode?

Sandbox Mode is a **complete implementation** of LeanVibe's autonomous development platform using mock AI services that provide realistic responses and demonstrations. It's not a limited demo—it showcases the full range of autonomous development capabilities.

### Key Features

- **🚀 Zero Configuration**: Works immediately without API keys
- **🤖 Realistic AI Simulation**: Context-aware responses that demonstrate real capabilities  
- **🔧 Multi-Agent Coordination**: Full orchestration of specialized AI agents
- **📊 Professional Quality**: Enterprise-ready demonstrations suitable for evaluation
- **⚡ Instant Response**: No waiting for external API calls
- **🎯 Complete Scenarios**: End-to-end autonomous development workflows

## Quick Start

### Option 1: One-Command Demo Launch
```bash
# Clone and start sandbox demo in under 2 minutes
git clone https://github.com/leanvibe/agent-hive.git
cd agent-hive
./start-sandbox-demo.sh

# Demo will be available at: http://localhost:8080
```

### Option 2: Manual Setup
```bash
# 1. Clone repository
git clone https://github.com/leanvibe/agent-hive.git
cd agent-hive

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install minimal dependencies
pip install fastapi uvicorn pydantic structlog

# 4. Set sandbox environment
export SANDBOX_MODE=true
export SANDBOX_DEMO_MODE=true

# 5. Launch demo
python demo_launcher.py
```

## How Sandbox Mode Works

### Automatic Detection
Sandbox mode automatically activates when:
- `ANTHROPIC_API_KEY` is missing or invalid
- `SANDBOX_MODE=true` is set in environment
- Running via `start-sandbox-demo.sh` script

### Mock AI Services

#### Mock Anthropic Client
- **Context-Aware Responses**: Adapts to task type and development phase
- **Realistic Timing**: 2-6 second delays to simulate real API calls
- **Progressive Conversations**: Multi-turn dialogues that build context
- **Error Simulation**: Realistic error handling and recovery

#### Intelligent Response Generation
```python
# Automatically detects task type and generates appropriate response
Task: "Create a Fibonacci calculator"
→ Mock client provides: Algorithm analysis, implementation plan, working code

Task: "Build user authentication system"  
→ Mock client provides: Security analysis, architecture design, secure implementation
```

### Multi-Agent Simulation

Sandbox mode includes specialized AI agents:

- **🏗️ System Architect**: Requirements analysis, system design
- **👨‍💻 Senior Developer**: Code implementation, best practices
- **🧪 QA Engineer**: Test creation, validation, quality assurance
- **👀 Code Reviewer**: Code quality, security, optimization  
- **📝 Technical Writer**: Documentation, user guides, API docs

## Demo Scenarios

### Simple Scenarios (5-7 minutes)
Perfect for initial evaluation and quick demonstrations.

#### Fibonacci Calculator
- **Description**: Create efficient Fibonacci number calculator
- **Demonstrates**: Function implementation, error handling, testing
- **Artifacts**: Python code, test suite, documentation
- **Key Features**: Input validation, iterative algorithm, comprehensive tests

#### Temperature Converter  
- **Description**: Multi-unit temperature converter (C/F/K)
- **Demonstrates**: Class design, validation, CLI interface
- **Artifacts**: Converter class, CLI tool, test suite, user guide
- **Key Features**: Physical limits validation, precision handling, user-friendly interface

### Moderate Scenarios (10-15 minutes)
Showcase complex feature development and system integration.

#### User Authentication System
- **Description**: Secure authentication with JWT and password hashing
- **Demonstrates**: Security best practices, database integration, API design
- **Artifacts**: Auth system, user models, security tests, API documentation
- **Key Features**: bcrypt hashing, JWT tokens, rate limiting, audit logging

#### Data Processing Pipeline
- **Description**: Scalable CSV/JSON data processing with validation
- **Demonstrates**: Pipeline architecture, error handling, performance optimization
- **Artifacts**: Pipeline engine, validators, transformers, performance tests
- **Key Features**: Stream processing, error recovery, configurable validation

### Complex Scenarios (20-25 minutes)
Full application development with multiple components.

#### REST API with Database
- **Description**: Complete FastAPI application with SQLAlchemy
- **Demonstrates**: Full-stack development, database design, API architecture
- **Artifacts**: FastAPI app, database models, API tests, Docker config
- **Key Features**: CRUD operations, authentication, OpenAPI docs, containerization

### Enterprise Scenarios (30+ minutes)
Large-scale system architecture and deployment.

#### Microservices Architecture
- **Description**: Multi-service system with API gateway and monitoring
- **Demonstrates**: Distributed systems, service orchestration, DevOps practices
- **Artifacts**: Multiple services, Docker Compose, monitoring setup, CI/CD config
- **Key Features**: Service discovery, load balancing, health checks, observability

## API Reference

### Sandbox Status
```bash
GET /api/demo/sandbox/status
```
Returns current sandbox configuration and capabilities.

**Response:**
```json
{
  "sandbox_mode": {
    "enabled": true,
    "auto_detected": true,
    "reason": "Required API keys missing: ANTHROPIC_API_KEY"
  },
  "mock_services": {
    "anthropic": true,
    "openai": false,
    "github": false
  },
  "demo_features": {
    "scenarios_enabled": true,
    "realistic_timing": true,
    "progress_simulation": true
  }
}
```

### Available Scenarios
```bash
GET /api/demo/sandbox/scenarios
```
Returns all available demo scenarios with metadata.

### Start Demo
```bash
POST /api/demo/sandbox/start
Content-Type: application/json

{
  "session_id": "demo-session-123",
  "task": {
    "description": "Create a Fibonacci calculator",
    "complexity": "simple",
    "requirements": [
      "Handle positive integers",
      "Include input validation",
      "Use efficient algorithm"
    ]
  }
}
```

### Session Progress
```bash
GET /api/demo/sandbox/session/{session_id}
```
Returns real-time progress of autonomous development session.

## Integration Examples

### JavaScript/React Frontend
```javascript
// Start autonomous development demo
const startDemo = async (taskDescription) => {
  const response = await fetch('/api/demo/sandbox/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id: `demo-${Date.now()}`,
      task: {
        description: taskDescription,
        complexity: 'simple'
      }
    })
  });
  
  const result = await response.json();
  return result.session_id;
};

// Monitor progress
const monitorProgress = async (sessionId) => {
  const response = await fetch(`/api/demo/sandbox/session/${sessionId}`);
  const status = await response.json();
  
  console.log(`Progress: ${status.progress}%`);
  console.log(`Status: ${status.status}`);
  console.log(`Agents: ${status.agents.length} active`);
  
  return status;
};
```

### Python Client
```python
import requests
import time

def run_sandbox_demo(task_description):
    # Start demo
    response = requests.post('/api/demo/sandbox/start', json={
        'session_id': f'demo-{int(time.time())}',
        'task': {
            'description': task_description,
            'complexity': 'simple'
        }
    })
    
    session_id = response.json()['session_id']
    
    # Monitor progress
    while True:
        status_response = requests.get(f'/api/demo/sandbox/session/{session_id}')
        status = status_response.json()
        
        print(f"Progress: {status['progress']}%")
        print(f"Current phase: {status.get('current_phase', 'Starting...')}")
        
        if status['status'] == 'completed':
            print("Demo completed successfully!")
            return status['artifacts']
        elif status['status'] == 'error':
            print(f"Demo failed: {status.get('error', 'Unknown error')}")
            break
            
        time.sleep(2)
```

## Customization

### Adding Custom Scenarios
```python
# Create custom scenario
from app.core.sandbox.demo_scenarios import DemoScenario, ScenarioComplexity, ScenarioCategory

custom_scenario = DemoScenario(
    id="custom-calculator",
    title="Scientific Calculator",
    description="Create a scientific calculator with advanced functions",
    category=ScenarioCategory.FUNCTION_DEVELOPMENT,
    complexity=ScenarioComplexity.MODERATE,
    estimated_duration_minutes=8,
    requirements=[
        "Basic arithmetic operations",
        "Scientific functions (sin, cos, log)",
        "Memory operations",
        "Expression parsing"
    ],
    expected_artifacts=[
        "calculator.py - Main calculator class",
        "parser.py - Expression parser",
        "test_calculator.py - Test suite"
    ],
    success_criteria=[
        "All mathematical operations work correctly",
        "Expression parsing handles complex expressions",
        "Memory operations function properly"
    ],
    demonstration_script={
        "phases": [
            {
                "name": "Requirements Analysis",
                "duration_seconds": 60,
                "agent": "architect",
                "description": "Analyze calculator requirements and design approach"
            },
            # ... more phases
        ]
    }
)
```

### Custom Mock Responses
```python
# Extend mock client with custom responses
from app.core.sandbox.mock_anthropic_client import MockAnthropicClient

class CustomMockClient(MockAnthropicClient):
    def _generate_custom_response(self, prompt: str) -> str:
        if "calculator" in prompt.lower():
            return self._generate_calculator_response(prompt)
        return super()._generate_response(prompt)
    
    def _generate_calculator_response(self, prompt: str) -> str:
        # Custom calculator-specific responses
        return "I'll create a comprehensive calculator implementation..."
```

## Performance & Limitations

### Sandbox Performance
- **Response Time**: Instant to 6 seconds (simulated)
- **Concurrent Sessions**: Unlimited (memory permitting)
- **Resource Usage**: <100MB RAM, minimal CPU
- **Storage**: Temporary files only, automatic cleanup

### Limitations
- **No Real AI**: Responses are pre-defined or template-based
- **No External Integration**: GitHub, external APIs not functional
- **No Persistent Storage**: Sessions are memory-only
- **Limited Customization**: Scenarios are pre-built

### When to Upgrade to Production
- Need real AI-generated code and responses
- Require external service integrations
- Want persistent data and user management
- Need custom scenarios and workflows

## Troubleshooting

### Common Issues

**Demo doesn't start**
```bash
# Check Python version (3.8+ required)
python --version

# Check dependencies
pip list | grep -E "(fastapi|uvicorn|pydantic)"

# Check environment variables
echo $SANDBOX_MODE
```

**Import errors**
```bash
# Ensure you're in the project directory
pwd
ls -la app/core/sandbox/

# Check Python path
python -c "import sys; print(sys.path)"
```

**Port conflicts**
```bash
# Check if port 8080 is in use
lsof -i :8080

# Use different port
export DEMO_PORT=9000
./start-sandbox-demo.sh
```

### Getting Help

- **GitHub Issues**: https://github.com/leanvibe/agent-hive/issues
- **Documentation**: https://github.com/leanvibe/agent-hive/docs
- **Discord Community**: https://discord.gg/leanvibe

## Architecture Deep Dive

### Sandbox Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Sandbox Mode Architecture                │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  Configuration  │    │  Demo Scenarios │                │  
│  │   Auto-detect   │    │   Pre-defined   │                │
│  │   API keys      │    │   workflows     │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                       │                        │
│           ▼                       ▼                        │
│  ┌─────────────────────────────────────────────────────────┐│
│  │            Sandbox Orchestrator                         ││
│  │  • Multi-agent coordination simulation                  ││
│  │  • Realistic timing and progress tracking               ││
│  │  • Session management and artifact generation           ││
│  └─────────────────────────────────────────────────────────┘│
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Mock AI Services                           ││
│  │  ┌─────────────────┐  ┌─────────────────┐              ││
│  │  │ Mock Anthropic  │  │ Mock OpenAI     │              ││
│  │  │ • Context-aware │  │ • Embeddings    │              ││
│  │  │ • Realistic AI  │  │ • Search        │              ││
│  │  │ • Progressive   │  │ • Semantic      │              ││
│  │  └─────────────────┘  └─────────────────┘              ││
│  └─────────────────────────────────────────────────────────┘│
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                Demo API Layer                           ││
│  │  • RESTful endpoints                                    ││
│  │  • Real-time progress streaming                         ││
│  │  • Session management                                   ││
│  │  • Artifact delivery                                    ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

### Request Flow

1. **Demo Request** → Sandbox detection and configuration
2. **Scenario Selection** → Load appropriate demo workflow  
3. **Mock AI Processing** → Generate realistic responses
4. **Progress Simulation** → Real-time status updates
5. **Artifact Generation** → Create demo deliverables
6. **Session Completion** → Return comprehensive results

## Business Value

### For Developers
- **Zero Friction Evaluation**: Try before committing resources
- **Understanding Capabilities**: See what autonomous development can do
- **Integration Planning**: Understand API structure and workflows
- **Risk Mitigation**: Evaluate fit before production investment

### For Enterprises  
- **Proof of Concept**: Validate autonomous development for your use cases
- **Stakeholder Demos**: Show capabilities to decision makers
- **Procurement Support**: Include in vendor evaluation processes
- **Training**: Familiarize teams with autonomous development concepts

### For Sales & Marketing
- **Live Demonstrations**: Always-ready demos for any audience
- **Conference Presentations**: Reliable demonstrations without connectivity
- **Customer Meetings**: Professional quality evaluation capability
- **Lead Qualification**: Let prospects experience the technology

## Success Metrics

Track these metrics to measure sandbox effectiveness:

- **Time to First Value**: <2 minutes from clone to running demo
- **Completion Rate**: >95% of users complete at least one scenario  
- **Engagement Time**: Average 15-25 minutes per session
- **Conversion Rate**: 30-40% of sandbox users request production access
- **Feedback Quality**: Professional evaluation suitable for enterprise decisions

---

🚀 **Ready to Experience Autonomous Development?**

```bash
git clone https://github.com/leanvibe/agent-hive.git
cd agent-hive  
./start-sandbox-demo.sh
```

Visit http://localhost:8080 and see the future of AI-powered development!