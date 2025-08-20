# LeanVibe Agent Hive 2.0 - Tmux Integration System

## Overview

The LeanVibe Agent Hive 2.0 Tmux Integration System provides a comprehensive solution for spawning, managing, and debugging CLI coding agents in isolated tmux sessions. Each agent runs in its own tmux session with dedicated workspace directories, Redis stream communication, and extensive monitoring capabilities.

## Architecture

### Core Components

1. **Enhanced Agent Launcher** (`enhanced_agent_launcher.py`)
   - Spawns multiple types of CLI agents (Claude Code, Cursor, Open Code, Aider, Continue)
   - Creates isolated tmux sessions with unique workspace directories
   - Manages environment setup and configuration
   - Provides health checking and agent lifecycle management

2. **Tmux Session Manager** (`tmux_session_manager.py`)
   - Low-level tmux session creation and management
   - Git repository setup and branch management
   - Command execution within sessions
   - Session monitoring and cleanup

3. **Agent Redis Bridge** (`agent_redis_bridge.py`)
   - Redis Stream integration for orchestrator communication
   - Message routing and task distribution
   - Agent registration and heartbeat management
   - Load balancing across available agents

4. **Session Health Monitor** (`session_health_monitor.py`)
   - Real-time health monitoring of agent sessions
   - Automatic failure detection and recovery
   - Performance analytics and alerting
   - Predictive failure analysis

5. **Enhanced CLI Integration** (`enhanced_cli_integration.py`)
   - Short ID system for easy agent access
   - Rich CLI output formatting
   - Session bookmarking and quick access
   - Batch operations and pattern matching

6. **Enhanced SimpleOrchestrator** (updated `simple_orchestrator.py`)
   - Integration with all tmux components
   - Enhanced agent spawning with tmux sessions
   - Task delegation via Redis streams
   - Comprehensive system status reporting

## Quick Start

### Prerequisites

- tmux installed and available in PATH
- Redis server running (for agent communication)
- Python 3.8+ with required dependencies
- CLI tools for desired agent types (claude, cursor, etc.)

### Basic Usage

```bash
# Initialize the system (handled automatically)
hive init

# Spawn a Claude Code agent
hive agent spawn --type claude-code --task TSK-123

# List all active agents
hive agent list

# Attach to an agent session for manual inspection
hive agent attach AGT-A7B2

# View agent logs
hive agent logs AGT-A7B2 --follow

# Kill an agent
hive agent kill AGT-A7B2

# Show comprehensive dashboard
hive agent dashboard
```

## Detailed Usage

### Agent Management

#### Spawning Agents

```bash
# Basic agent spawn
hive agent spawn --type claude-code

# Spawn with specific configuration
hive agent spawn \
  --type cursor-agent \
  --task TSK-123 \
  --workspace my-project \
  --branch feature/new-feature \
  --workdir /path/to/project \
  --env "DEBUG=true" \
  --env "LOG_LEVEL=info"

# Spawn multiple agents
hive agent spawn --type claude-code --count 3

# Wait for agents to be ready
hive agent spawn --type claude-code --wait
```

#### Agent Discovery and Access

```bash
# List agents with various formats
hive agent list                    # Basic table
hive agent list --wide            # Extended information
hive agent list --output json     # JSON format
hive agent list --sessions        # Include tmux session details

# Show agents in tree view
hive agent tree

# Filter agents by pattern
hive agent list --pattern "claude"
hive agent tree --pattern "backend"

# Watch for changes
hive agent list --watch
```

#### Session Interaction

```bash
# Attach to agent session
hive agent attach AGT-A7B2
hive agent attach AGT-A7B2 --new-window

# Execute commands in agent session
hive agent exec AGT-A7B2 "ls -la"
hive agent exec AGT-A7B2 "git status" --window main --capture

# View logs with filtering
hive agent logs AGT-A7B2 --lines 100
hive agent logs AGT-A7B2 --follow
hive agent logs AGT-A7B2 --filter "ERROR"
```

#### Agent Information

```bash
# Basic status
hive agent status AGT-A7B2

# Comprehensive information
hive agent info AGT-A7B2 --include-health --include-logs

# Export agent data
hive agent info AGT-A7B2 --format json > agent.json
hive agent info AGT-A7B2 --format yaml > agent.yaml
```

### Session Bookmarking

```bash
# Create bookmark for quick access
hive agent bookmark dev-main AGT-A7B2 --description "Main development agent"

# List bookmarks
hive agent bookmarks

# Remove bookmark
hive agent bookmarks --remove dev-main

# Use bookmark to attach
hive agent attach @dev-main
```

### Batch Operations

```bash
# Kill agents matching pattern
hive agent kill-pattern "claude*" --dry-run
hive agent kill-pattern "claude*" --confirm-each

# Batch operations (future enhancement)
hive agent batch --pattern "stopped" --operation restart
```

### Health Monitoring

```bash
# View health dashboard
hive agent dashboard --health --performance

# Export health data
hive agent dashboard --export dashboard.json
```

## Agent Types

### Claude Code
- **Command**: `claude`
- **Environment**: `CLAUDE_CODE_MODE=agent`
- **Features**: Full Claude Code functionality in agent mode

### Cursor Agent
- **Command**: `cursor --agent-mode`
- **Environment**: `CURSOR_AGENT_MODE=true`
- **Features**: Cursor's AI-powered coding assistance

### Open Code
- **Command**: `opencode --agent`
- **Environment**: `OPENCODE_AGENT_MODE=true`
- **Features**: Open-source coding agent

### Aider
- **Command**: `aider --auto-commits --stream`
- **Environment**: `AIDER_AGENT_MODE=true`
- **Features**: AI pair programming with automatic commits

### Continue
- **Command**: `continue --agent-mode`
- **Environment**: `CONTINUE_AGENT_MODE=true`
- **Features**: VS Code extension in agent mode

## Session Management

### Tmux Sessions

Each agent gets a unique tmux session with:
- **Session naming**: `agent-{SHORT_ID}` (e.g., `agent-A7B2`)
- **Workspace directory**: `workspaces/workspace-{SHORT_ID}/`
- **Git repository**: Initialized with agent-specific branch
- **Environment variables**: Pre-configured for agent communication

### Workspace Structure

```
workspaces/
├── workspace-A7B2/           # Agent AGT-A7B2 workspace
│   ├── .git/                 # Git repository
│   ├── .leanvibe/           # LeanVibe configuration
│   │   ├── redis_config.json
│   │   └── agent.log
│   ├── README.md            # Workspace documentation
│   └── [project files]
└── workspace-B3C4/          # Agent AGT-B3C4 workspace
    └── ...
```

### Environment Variables

Each session includes:
```bash
LEANVIBE_AGENT_ID=AGT-A7B2
LEANVIBE_AGENT_SHORT_ID=A7B2
LEANVIBE_AGENT_TYPE=claude-code
LEANVIBE_TASK_ID=TSK-123
LEANVIBE_REDIS_STREAM=agent_tasks
LEANVIBE_CONSUMER_GROUP=general_agents
WORKSPACE_PATH=/path/to/workspace
PYTHONPATH=/path/to/workspace:$PYTHONPATH
```

## Redis Stream Integration

### Message Flow

```
Orchestrator ──→ Redis Stream ──→ Agent
     ↑                              │
     └──────── Redis Stream ←───────┘
```

### Message Types

#### Orchestrator → Agent
- `TASK_ASSIGNMENT`: New task delegation
- `CONFIG_UPDATE`: Configuration changes
- `SHUTDOWN_REQUEST`: Graceful shutdown
- `HEALTH_CHECK`: Health verification

#### Agent → Orchestrator
- `AGENT_READY`: Agent initialization complete
- `TASK_PROGRESS`: Task progress updates
- `TASK_COMPLETED`: Task completion
- `TASK_FAILED`: Task failure
- `HEARTBEAT`: Keep-alive signal
- `LOG_MESSAGE`: Log events
- `STATUS_UPDATE`: Status changes

### Consumer Groups

Agents are organized into consumer groups:
- `architects`: AI architect agents
- `backend_engineers`: Backend development agents
- `frontend_developers`: Frontend development agents
- `qa_engineers`: Quality assurance agents
- `devops_engineers`: DevOps and infrastructure agents
- `general_agents`: General purpose agents

## Health Monitoring

### Health Checks

The system performs comprehensive health checks:

1. **Tmux Session Health**
   - Session existence and responsiveness
   - Window and pane status
   - Response time monitoring

2. **Agent Process Health**
   - Process existence and status
   - Resource usage (CPU, memory)
   - Performance metrics

3. **Redis Connectivity**
   - Connection status
   - Message flow verification
   - Heartbeat monitoring

4. **Workspace Integrity**
   - Directory existence
   - Disk usage monitoring
   - File permissions

5. **Resource Usage**
   - System CPU and memory
   - Disk space availability
   - Network connectivity

### Recovery Actions

Automatic recovery actions:
- `RESTART_AGENT`: Restart agent process
- `RECREATE_SESSION`: Recreate tmux session
- `CLEANUP_WORKSPACE`: Clean temporary files
- `RESET_CONFIGURATION`: Reset to default config
- `QUARANTINE_SESSION`: Isolate problematic session
- `ESCALATE_TO_HUMAN`: Alert human operators

### Health Status Levels

- `HEALTHY`: All checks passing
- `WARNING`: Minor issues detected
- `CRITICAL`: Major issues requiring attention
- `FAILED`: Agent non-functional
- `RECOVERING`: Recovery in progress
- `UNKNOWN`: Unable to determine status

## Short ID System

### ID Generation

Short IDs are generated for easy reference:
- **Format**: 3-4 character alphanumeric (e.g., `A7B2`, `C3D4`)
- **Uniqueness**: Guaranteed unique within namespace
- **Human-friendly**: Easy to type and remember

### ID Resolution

The system resolves agent references in order:
1. Exact short ID match
2. Exact full ID match
3. Session name match
4. Partial short ID match (minimum 3 characters)
5. Fuzzy matching with suggestions

### Usage Examples

```bash
# All of these can refer to the same agent:
hive agent attach AGT-A7B2        # Full ID
hive agent attach A7B2            # Short ID
hive agent attach agent-A7B2      # Session name
hive agent attach A7              # Partial short ID
```

## Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_STREAM_NAME=agent_tasks

# Workspace Configuration
WORKSPACES_DIR=./workspaces
MAX_CONCURRENT_AGENTS=10

# Health Monitoring
HEALTH_CHECK_INTERVAL=30
RECOVERY_COOLDOWN=300
MAX_RECOVERY_ATTEMPTS=3

# Tmux Configuration
TMUX_SESSION_PREFIX=agent
TMUX_SOCKET_NAME=leanvibe
```

### Agent Configuration

Each agent type can be configured:

```python
# enhanced_agent_launcher.py
self.agent_configs = {
    AgentLauncherType.CLAUDE_CODE: {
        "command": "claude",
        "default_args": ["--session", "{session_name}"],
        "env_vars": {
            "CLAUDE_CODE_MODE": "agent",
            "CLAUDE_CODE_REDIS_STREAM": "{redis_stream}"
        },
        "health_check_command": "claude --version",
        "setup_commands": [
            "echo 'Initializing Claude Code agent...'"
        ]
    }
}
```

## Troubleshooting

### Common Issues

#### Agent Won't Start
```bash
# Check agent availability
hive agent spawn --type claude-code --dry-run

# Check tmux sessions
tmux list-sessions

# Check logs
hive agent logs AGT-A7B2 --lines 100
```

#### Session Lost
```bash
# Check session health
hive agent status AGT-A7B2 --include-health

# Attempt recovery
hive agent kill AGT-A7B2
hive agent spawn --type claude-code
```

#### Redis Connection Issues
```bash
# Check Redis status
redis-cli ping

# Check agent registration
hive agent list --format json | jq '.[] | .bridge_status'
```

### Debugging Commands

```bash
# Show system status
hive status

# Show detailed agent tree
hive agent tree

# Export all agent data
hive agent dashboard --export debug.json

# Check tmux sessions directly
tmux list-sessions | grep agent

# Monitor Redis streams
redis-cli XREAD STREAMS agent_tasks $
```

## API Integration

### REST Endpoints

```bash
# Agent management
POST /api/agents/spawn
GET  /api/agents/list
GET  /api/agents/{agent_id}/status
POST /api/agents/{agent_id}/terminate
POST /api/agents/{agent_id}/execute

# Health monitoring
GET  /api/health/summary
GET  /api/health/agents/{agent_id}
POST /api/health/recover/{agent_id}

# Session management
GET  /api/sessions/list
GET  /api/sessions/{session_id}/info
POST /api/sessions/{session_id}/command
```

### Python API

```python
from app.core import create_enhanced_simple_orchestrator

# Initialize orchestrator
orchestrator = await create_enhanced_simple_orchestrator()

# Spawn agent
agent_id = await orchestrator.spawn_agent(
    role=AgentRole.BACKEND_DEVELOPER,
    agent_type=AgentLauncherType.CLAUDE_CODE,
    task_id="TSK-123"
)

# Get agent status
status = await orchestrator.get_agent_session_info(agent_id)

# Execute command
result = await orchestrator.execute_command_in_agent_session(
    agent_id, "git status"
)
```

## Performance Optimization

### Resource Management

- **Session Limits**: Configure `MAX_CONCURRENT_AGENTS`
- **Workspace Cleanup**: Automatic cleanup of idle sessions
- **Memory Monitoring**: Track agent memory usage
- **Disk Management**: Monitor workspace disk usage

### Scaling Considerations

- **Redis Clustering**: Scale Redis for high throughput
- **Load Balancing**: Distribute agents across nodes
- **Session Persistence**: Handle node failures gracefully
- **Monitoring**: Comprehensive metrics collection

## Security

### Isolation

- **Process Isolation**: Each agent in separate tmux session
- **Workspace Isolation**: Dedicated directories per agent
- **Network Isolation**: Redis-based communication only
- **Resource Limits**: CPU and memory constraints

### Access Control

- **Agent Authentication**: Redis-based agent registration
- **Command Validation**: Sanitize executed commands
- **Workspace Permissions**: Restricted file access
- **Audit Logging**: Track all agent operations

## Contributing

### Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install tmux (Ubuntu/Debian)
sudo apt-get install tmux

# Install tmux (macOS)
brew install tmux

# Start Redis
redis-server

# Run tests
pytest tests/test_tmux_integration.py
```

### Adding New Agent Types

1. Update `AgentLauncherType` enum
2. Add configuration to `agent_configs`
3. Implement health check command
4. Add CLI support
5. Update documentation

### Testing

```bash
# Unit tests
pytest tests/unit/test_enhanced_agent_launcher.py
pytest tests/unit/test_tmux_session_manager.py

# Integration tests
pytest tests/integration/test_tmux_integration.py

# End-to-end tests
pytest tests/e2e/test_agent_lifecycle.py
```

## Roadmap

### Planned Features

- **Multi-node Support**: Distribute agents across multiple nodes
- **Advanced Recovery**: ML-based failure prediction
- **Session Sharing**: Collaborative agent sessions
- **IDE Integration**: VS Code and other editor plugins
- **Container Support**: Docker-based agent isolation

### Version History

- **v2.0.0**: Initial tmux integration release
- **v2.1.0**: Enhanced health monitoring
- **v2.2.0**: Multi-agent type support
- **v2.3.0**: Advanced CLI features

## License

LeanVibe Agent Hive 2.0 Tmux Integration System is licensed under the MIT License.