# LeanVibe Agent Hive 2.0

Modern FastAPI backend with a Lit + Vite PWA for real‑time operational dashboards and autonomous multi-agent orchestration.

## 🚀 Quick Start

### Using the Unified CLI (Recommended)

LeanVibe Agent Hive provides a unified CLI following Unix philosophies, similar to `docker`, `kubectl`, and `terraform`:

**🚀 Installation with uv:**
```bash
# Global installation (recommended)
uv tool install -e .

# Now use hive command anywhere
hive doctor           # System diagnostics
hive start           # Start the platform  
hive agent deploy backend-developer  # Deploy agents
hive status --watch   # Monitor system
hive dashboard       # Open dashboard
```

**📋 Documentation:**
- **[uv Installation Guide](UV_INSTALLATION_GUIDE.md)** - Complete setup with uv
- **[CLI Usage Guide](CLI_USAGE_GUIDE.md)** - Docker/kubectl-style commands
- **[Port Configuration](PORT_CONFIGURATION.md)** - Non-standard ports to avoid conflicts

### Manual Setup (Alternative)
```bash
# Infrastructure
docker compose up -d postgres redis

# Backend API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend PWA
cd mobile-pwa && npm ci && npm run dev
```

### Health Checks & Service URLs
- **CLI**: `hive doctor` (comprehensive diagnostics)
- **API Health**: `http://localhost:18080/health` (non-standard port)
- **API Docs**: `http://localhost:18080/docs`
- **PWA Dashboard**: `http://localhost:18443`
- **WebSocket**: `ws://localhost:18080/api/dashboard/ws/dashboard`

> **Note**: Uses non-standard ports (18080, 18443, 15432, 16379) to avoid conflicts with other development tools.

## 🧪 Testing

### Test Suites
- **Full test suite**: `make test`
- **Fast lanes**:
  - `make test-core-fast` (smoke + ws + prompt core)
  - `make test-backend-fast` (contracts + core + smoke)
  - `make test-prompt` (prompt optimization engines)

### Performance Testing
WebSocket load testing with k6:
```bash
cd scripts/performance
make smoke BACKEND_WS_URL=ws://localhost:18080/api/dashboard/ws/dashboard ACCESS_TOKEN=dev-token
```

Environment configuration:
- `WS_RATE_TOKENS_PER_SEC`, `WS_RATE_BURST` - Rate limiting
- `WS_COMPRESSION_ENABLED` - WebSocket compression (reserved)

## 🏗️ Architecture

### WebSocket Contract Guarantees
- All `error` frames include `timestamp` and `correlation_id`
- All `data_error` frames include `timestamp`, `error`, and `correlation_id`
- `data_response` messages include `type`, `data_type`, `data`
- `pong` frames include `timestamp`

### Safety & Observability
- **Rate Limiting**: 20 rps per connection, burst 40
- **Message Size**: 64KB cap with error responses
- **Subscription Limits**: Max subscriptions enforced per connection
- **Tracing**: All frames include `correlation_id`
- **Metrics**: Prometheus metrics at `/api/dashboard/metrics/websockets`

## 📋 Design Policies

- **No server-rendered dashboards** - Use API/WebSocket endpoints
- **Brand**: "HiveOps" (configured in `mobile-pwa/src/components/layout/app-header.ts`)
- **Enterprise templates**: Optional, gated by `ENABLE_ENTERPRISE_TEMPLATES` (default: disabled)
- **Port configuration**: Non-standard ports to avoid conflicts

## 📚 Documentation

**Epic 9 Consolidated Documentation** - Choose your journey:

### 🚀 Quick Start Journey (5 minutes)
- **[Setup Guide](UV_INSTALLATION_GUIDE.md)** - Get running fast with uv installation
- **[CLI Commands](CLI_USAGE_GUIDE.md)** - Essential commands for daily use
- **Ready to go!** 🎉

### 👨‍💻 Developer Journey (30 minutes)
- **[Developer Onboarding](DEVELOPER_ONBOARDING_30MIN.md)** - Complete developer setup
- **[System Architecture](ARCHITECTURE_CONSOLIDATED.md)** - Understand the unified architecture
- **[API Reference](API_REFERENCE_CONSOLIDATED.md)** - Complete API documentation
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute code
- **Ready to develop!** ⚡

### 🏢 Enterprise Journey (60 minutes)
- **[Production Deployment](DEPLOYMENT_CHECKLIST.md)** - Enterprise deployment procedures
- **[Enterprise Features](docs/guides/ENTERPRISE_USER_GUIDE.md)** - Advanced enterprise capabilities
- **[Operations Runbook](docs/OPERATIONAL_RUNBOOK.md)** - Production operations guide
- **[Troubleshooting](docs/runbooks/TROUBLESHOOTING_GUIDE_COMPREHENSIVE.md)** - Problem resolution
- **Ready for production!** 🏭

### 📖 Complete Documentation Map
- **[Navigation Index](docs/NAV_INDEX.md)** - Complete documentation index
- **[Implementation Guides](docs/guides/)** - Detailed implementation guides
- **[API & Reference](docs/reference/)** - Technical reference materials

## 🛠️ Development

### Project Structure
```
bee-hive/
├── app/                    # FastAPI backend
├── mobile-pwa/            # Lit + Vite frontend
├── docs/                  # Documentation
├── scripts/               # Automation and testing
├── k8s/                   # Kubernetes manifests
├── terraform/             # Infrastructure as code
└── helm/                  # Helm charts
```

### Key Commands

**🤖 Unified CLI (Recommended)**
```bash
hive doctor                          # System diagnostics
hive start                           # Start platform services  
hive agent deploy <role>             # Deploy agents
hive status --watch                  # Monitor system
hive dashboard                       # Open monitoring UI
```

**📋 Traditional Commands**
- `./scripts/setup.sh` - Initial project setup
- `./scripts/start.sh` - Start all services
- `make test` - Run complete test suite
- `make help` - Show all available commands

**🎯 Quick Reference**
| Task | CLI Command | Traditional |
|------|-------------|-------------|
| Start system | `hive start` | `uvicorn app.main:app` |
| Deploy agent | `hive agent deploy backend-developer` | `python deploy_agent_cli.py deploy` |
| Check status | `hive status` | `curl http://localhost:8000/health` |
| View logs | `hive logs -f` | `tail -f logs/app.log` |
| Open dashboard | `hive dashboard` | Open browser manually |

## 🔗 Related Projects

- [Mobile PWA Dashboard](mobile-pwa/README.md)
- [API Documentation](docs/reference/API_REFERENCE_COMPREHENSIVE.md)
- [Deployment Guide](docs/guides/deployment-guide.md)