# LeanVibe Agent Hive 2.0

Modern FastAPI backend with a Lit + Vite PWA for real‑time operational dashboards and autonomous multi-agent orchestration.

## 🚀 Quick Start

For detailed setup instructions, see [Getting Started Guide](docs/GETTING_STARTED.md).

### Minimal Setup
```bash
# Infrastructure
docker compose up -d postgres redis

# Backend API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend PWA
cd mobile-pwa && npm ci && npm run dev
```

### Health Checks
- API Health: `GET http://localhost:8000/health`
- WebSocket: `ws://localhost:8000/api/dashboard/ws/dashboard`

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

### Core Documentation
- [Getting Started](docs/GETTING_STARTED.md) - Setup and installation
- [Core Overview](docs/CORE.md) - System architecture and concepts
- [Architecture](docs/ARCHITECTURE.md) - Technical implementation details
- [Navigation Index](docs/NAV_INDEX.md) - Complete documentation map

### Implementation Guides
- [Context Compression](docs/implementation/context-compression.md) - Intelligent conversation compression
- [Validation Framework](docs/reference/validation-framework.md) - Testing and validation
- [Mobile PWA](docs/guides/MOBILE_PWA_IMPLEMENTATION_GUIDE.md) - Frontend development

### Operations
- [Agent Consolidation Prompt](docs/AGENT_PROMPT_CONSOLIDATION.md) - Debt cleanup guidance
- [Troubleshooting](docs/runbooks/TROUBLESHOOTING_GUIDE_COMPREHENSIVE.md) - Problem resolution

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
- `./scripts/setup.sh` - Initial project setup
- `./scripts/start.sh` - Start all services
- `make test` - Run complete test suite
- `make help` - Show all available commands

## 🔗 Related Projects

- [Mobile PWA Dashboard](mobile-pwa/README.md)
- [API Documentation](docs/reference/API_REFERENCE_COMPREHENSIVE.md)
- [Deployment Guide](docs/guides/deployment-guide.md)