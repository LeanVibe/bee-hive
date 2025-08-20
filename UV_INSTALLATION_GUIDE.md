# 🚀 LeanVibe Agent Hive with `uv` - Installation Guide

## Quick Installation

### Option 1: Global Installation (Recommended)

```bash
# Install as a global uv tool (adds to PATH automatically)
uv tool install -e .

# Now hive is available globally
hive --help
hive doctor
hive status
```

### Option 2: Development Installation

```bash
# Install in development mode
uv pip install -e .

# Activate virtual environment
source .venv/bin/activate

# Or use direct path
.venv/bin/hive --help
```

### Option 3: Direct Execution (No Installation)

```bash
# Run directly with uv (no installation needed)
uv run -m app.hive_cli --help
uv run -m app.hive_cli status
uv run -m app.hive_cli agent deploy backend-developer
```

## 🎯 Available Commands

Once installed, you get **4 CLI commands**:

| Command | Description | Usage |
|---------|-------------|-------|
| `hive` | **Unified CLI** (Docker/kubectl style) | `hive status`, `hive agent deploy backend-developer` |
| `agent-hive` | Legacy full CLI | `agent-hive start`, `agent-hive develop "Build API"` |
| `ahive` | Short alias for agent-hive | `ahive status` |
| `lv` | Developer experience CLI | `lv intelligent-mode` |

## 🤖 Unified CLI Usage (Recommended)

### System Management
```bash
hive doctor           # System diagnostics
hive start           # Start all services  
hive status          # Show system status
hive status --watch  # Real-time monitoring
hive stop            # Stop all services
```

### Agent Management  
```bash
hive agent list                           # List all agents
hive agent ps                            # Docker ps style
hive agent deploy backend-developer      # Deploy backend agent
hive agent deploy qa-engineer            # Deploy QA agent
hive agent run frontend-developer        # Alias for deploy
```

### Monitoring & Tools
```bash
hive dashboard       # Open web dashboard
hive logs --follow   # Follow logs
hive demo           # Run complete demo
hive version        # Version info
```

### Quick Actions (Docker-compose style)
```bash
hive up             # Quick start services
hive down           # Quick stop services  
```

## 🔧 Development Workflow

### 1. First Time Setup
```bash
# Clone and setup
git clone <repository>
cd bee-hive

# Install globally with uv
uv tool install -e .

# Check system health
hive doctor
```

### 2. Daily Development
```bash
# Start the platform
hive start

# Deploy agents for development
hive agent deploy backend-developer --task "Implement user authentication"
hive agent deploy qa-engineer --task "Create integration tests"

# Monitor in real-time
hive status --watch

# Open monitoring dashboard
hive dashboard
```

### 3. Quick Development Cycle
```bash
hive up                                    # Quick start
hive agent deploy backend-developer        # Deploy agent
hive dashboard                             # Monitor progress
```

## 🌐 Service Integration

When running, the following services are available:

| Service | URL | Description |
|---------|-----|-------------|
| **API Documentation** | http://localhost:8000/docs | FastAPI Swagger UI |
| **System Health** | http://localhost:8000/health | Health check endpoint |
| **PWA Dashboard** | http://localhost:51735 | Real-time monitoring (when PWA is running) |

## 🚨 Troubleshooting

### Command Not Found
```bash
# If 'hive' command not found after uv tool install:
uv tool list                    # Check if installed
uv tool install -e . --force    # Reinstall

# Alternative: use direct path
~/.local/bin/hive --help
```

### Permission Issues
```bash
# Ensure uv tool directory is in PATH
echo $PATH | grep -E "(\.local/bin|uv)"

# Add to shell profile if needed (bash/zsh)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Development Issues
```bash
# Check system health
hive doctor

# Verify all services
hive status

# Reset system
hive down && hive up
```

## 📋 Available Entry Points

The `pyproject.toml` defines these commands:

```toml
[project.scripts]
agent-hive = "app.cli:main"          # Legacy full CLI
ahive = "app.cli:main"               # Short alias  
lv = "app.dx_cli:main"               # Developer experience CLI
hive = "app.hive_cli:main"           # Unified CLI (recommended)
```

## 🔄 Updates and Maintenance

### Updating the Installation
```bash
# Reinstall after code changes
uv tool install -e . --force

# Or for development mode
uv pip install -e . --force-reinstall
```

### Uninstalling
```bash
# Remove global tool installation
uv tool uninstall leanvibe-agent-hive

# Or remove from virtual environment
uv pip uninstall leanvibe-agent-hive
```

## 💡 Pro Tips

### 1. Shell Aliases
Add to your shell profile for even faster access:
```bash
# ~/.bashrc or ~/.zshrc
alias h='hive'
alias hs='hive status' 
alias ha='hive agent'
alias hd='hive dashboard'
```

### 2. JSON Output for Scripting
```bash
# Get system status as JSON
hive status --json

# Use with jq for filtering
hive status --json | jq '.agents.total'
```

### 3. Background Operations
```bash
# Start services in background
hive start --background

# Monitor with watch mode
hive status --watch
```

### 4. Integration with CI/CD
```yaml
# GitHub Actions example
- name: Setup Agent Hive
  run: |
    uv tool install -e .
    hive doctor
    hive start --background
    hive agent deploy backend-developer
```

## 🎯 Migration from Manual Commands

| Old Command | New Command |
|-------------|-------------|
| `python deploy_agent_cli.py deploy` | `hive agent deploy backend-developer` |
| `uvicorn app.main:app --reload` | `hive start` |
| `curl http://localhost:8000/health` | `hive status` |
| Manual browser opening | `hive dashboard` |

---

The unified `hive` CLI provides a professional, Docker/kubectl-style interface that makes Agent Hive easy to use for both development and production scenarios.