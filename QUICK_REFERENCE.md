# 🚀 LeanVibe Agent Hive - Quick Reference

## Essential Commands

### 🔧 System Management
```bash
python hive doctor           # System diagnostics & health check
python hive start            # Start all services
python hive status           # Show system status
python hive status --watch   # Monitor system in real-time
python hive stop             # Stop all services
```

### 🤖 Agent Management
```bash
python hive agent list       # List all agents
python hive agent ps         # Show running agents (docker style)
python hive agent deploy backend-developer    # Deploy backend agent
python hive agent deploy qa-engineer          # Deploy QA agent
python hive agent deploy frontend-developer   # Deploy frontend agent
python hive agent deploy devops-engineer      # Deploy DevOps agent
```

### 📊 Monitoring
```bash
python hive dashboard        # Open web dashboard
python hive logs --follow    # Follow logs in real-time
python hive version          # Show version info
```

### ⚡ Quick Actions
```bash
python hive up               # Quick start (docker-compose style)
python hive down             # Quick stop (docker-compose style)
python hive demo             # Run complete demonstration
```

## 🌐 Service URLs

- **API Documentation**: http://localhost:8000/docs
- **System Health**: http://localhost:8000/health  
- **PWA Dashboard**: http://localhost:51735 (when running)

## 🎯 Common Workflows

### Starting the System
```bash
python hive doctor                    # Check system health
python hive start                     # Start services
python hive agent deploy backend     # Deploy your first agent
python hive dashboard                 # Open monitoring
```

### Development Workflow
```bash
python hive up                        # Quick start
python hive agent deploy backend-developer --task "Build API"
python hive agent deploy qa-engineer --task "Create tests"
python hive status --watch            # Monitor progress
```

### System Monitoring
```bash
python hive status --watch            # Live status updates
python hive logs --follow             # Live log streaming  
python hive agent ps                  # Check running agents
```

## 🚨 Troubleshooting

```bash
python hive doctor                    # Comprehensive diagnostics
python hive status                    # Check system health
python hive stop && python hive start  # Restart system
```

## 💡 Pro Tips

- Use `--help` on any command for detailed options
- Commands follow Unix philosophy: composable and predictable
- JSON output available: `python hive status --json`
- Background mode: `python hive start --background`

---

**📋 Full Documentation**: [CLI_USAGE_GUIDE.md](CLI_USAGE_GUIDE.md)