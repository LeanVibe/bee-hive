# 🤖 LeanVibe Agent Hive CLI Usage Guide

## Quick Start

The LeanVibe Agent Hive provides a unified CLI following Unix philosophies, similar to `docker`, `kubectl`, and `terraform`.

### Installation & Setup

```bash
# Make CLI executable
chmod +x hive

# Check system health
python hive doctor

# Start the platform
python hive start
```

## 🚀 Core Commands

### System Management

| Command | Description | Example |
|---------|-------------|---------|
| `hive start` | Start all platform services | `python hive start --background` |
| `hive stop` | Stop all services | `python hive stop` |
| `hive status` | Show system status | `python hive status --watch` |
| `hive up` | Quick start (docker-compose style) | `python hive up` |
| `hive down` | Quick stop (docker-compose style) | `python hive down` |

### Agent Management (docker/kubectl style)

| Command | Description | Example |
|---------|-------------|---------|
| `hive agent list` | List all agents | `python hive agent list` |
| `hive agent ls` | List agents (alias) | `python hive agent ls` |
| `hive agent ps` | Show running agents | `python hive agent ps` |
| `hive agent deploy <role>` | Deploy new agent | `python hive agent deploy backend-developer` |
| `hive agent run <role>` | Run agent (alias for deploy) | `python hive agent run qa-engineer` |

### Monitoring & Diagnostics

| Command | Description | Example |
|---------|-------------|---------|
| `hive dashboard` | Open monitoring dashboard | `python hive dashboard` |
| `hive logs` | View system logs | `python hive logs --follow` |
| `hive doctor` | System diagnostics | `python hive doctor` |
| `hive version` | Show version info | `python hive version` |

## 🎯 Common Use Cases

### 1. Starting the System

```bash
# Quick health check
python hive doctor

# Start all services
python hive start

# Or start in background
python hive start --background

# Check status
python hive status
```

### 2. Agent Management

```bash
# Deploy a backend developer agent
python hive agent deploy backend-developer --task "Implement API endpoints"

# List all active agents
python hive agent list

# Deploy multiple agents
python hive agent deploy qa-engineer --task "Create test suite"
python hive agent deploy devops-engineer --task "Setup CI/CD"

# Monitor agents
python hive agent ps
```

### 3. Real-time Monitoring

```bash
# Watch system status (updates every 2s)
python hive status --watch

# Follow logs in real-time
python hive logs --follow

# Open web dashboard
python hive dashboard
```

### 4. Development Workflow

```bash
# Complete development workflow
python hive demo

# Start system and deploy agents
python hive up
python hive agent deploy backend-developer
python hive dashboard
```

## 🔧 Advanced Usage

### Agent Deployment Options

```bash
# Deploy with custom task
python hive agent deploy backend-developer --task "Build authentication API"

# Deploy with custom name
python hive agent deploy frontend-developer --name "ui-specialist"

# Available agent roles:
# - backend-developer
# - frontend-developer  
# - qa-engineer
# - devops-engineer
```

### System Monitoring

```bash
# Watch system status with real-time updates
python hive status --watch

# JSON output for scripting
python hive status --json

# View last 100 log lines
python hive logs --lines 100

# Follow logs continuously
python hive logs --follow
```

### Docker/kubectl Style Commands

```bash
# Docker-compose style
python hive up          # Start services in background
python hive down        # Stop all services

# Docker ps style
python hive agent ps    # Show running agents

# kubectl style
python hive agent list  # List all resources
python hive status      # Cluster status
```

## 🌐 Service URLs

When the system is running, you can access:

| Service | URL | Description |
|---------|-----|-------------|
| **API Documentation** | http://localhost:8000/docs | FastAPI interactive docs |
| **System Health** | http://localhost:8000/health | Health check endpoint |
| **PWA Dashboard** | http://localhost:51735 | Mobile-optimized monitoring |

## 🩺 Troubleshooting

### Common Issues

1. **"System not responding"**
   ```bash
   python hive doctor  # Check system health
   python hive start   # Start services
   ```

2. **Port conflicts**
   ```bash
   python hive doctor  # Check port status
   # Default ports: 8000 (API), 51735 (PWA), 5432 (PostgreSQL), 6379 (Redis)
   ```

3. **Missing dependencies**
   ```bash
   pip install fastapi uvicorn click rich pydantic
   ```

### Health Check

```bash
python hive doctor
```

This command checks:
- ✅ Python environment and packages
- 🔌 Port availability (8000, 5432, 6379, 51735)
- 🏥 API health status
- 💡 Actionable recommendations

## 🚀 Integration with CI/CD

### GitHub Actions Example

```yaml
name: Agent Hive Deployment
on: [push]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Agent Hive
        run: |
          python hive doctor
          python hive start --background
          python hive agent deploy backend-developer
```

### Script Integration

```bash
#!/bin/bash
# deployment-script.sh

set -e

echo "🚀 Starting Agent Hive deployment..."

# Health check
python hive doctor

# Start system
python hive up

# Deploy agents
python hive agent deploy backend-developer --task "$1"
python hive agent deploy qa-engineer --task "Test $1"

# Monitor
echo "✅ Deployment complete! Monitor at: http://localhost:8000/docs"
```

## 📋 Command Reference

### Help System

Every command has built-in help following Unix conventions:

```bash
python hive --help              # Main help
python hive agent --help        # Agent management help
python hive agent deploy --help # Deployment options help
python hive start --help        # Start command options
```

### Output Formats

Most commands support multiple output formats:

```bash
python hive status              # Human-readable table
python hive status --json       # JSON output for scripts
python hive agent list          # Table format
python hive agent list --format json  # JSON format
```

### Watch/Follow Options

Real-time monitoring is built into key commands:

```bash
python hive status --watch      # Live status updates
python hive logs --follow       # Live log streaming
```

## 🔄 Updates and Maintenance

```bash
# Check system version
python hive version

# Run diagnostics
python hive doctor

# Restart services
python hive down && python hive up
```

---

## 💡 Pro Tips

1. **Use aliases**: Create shell aliases for frequent commands
   ```bash
   alias h='python hive'
   alias hs='python hive status'
   alias ha='python hive agent'
   ```

2. **Background processes**: Use `--background` flag for CI/CD
   ```bash
   python hive start --background
   ```

3. **JSON output**: Use JSON format for scripting and automation
   ```bash
   python hive status --json | jq '.agents.total'
   ```

4. **Watch mode**: Monitor system changes in real-time
   ```bash
   python hive status --watch
   ```

This CLI follows Unix philosophy principles: each command does one thing well, commands are composable, and the interface is consistent and predictable.