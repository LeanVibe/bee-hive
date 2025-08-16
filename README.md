## Performance (WS SLOs)

Scripts under `scripts/performance` provide k6 scenarios for WebSocket load testing. Configure non-standard ports via env.

Example:

```
cd scripts/performance
make smoke BACKEND_WS_URL=ws://localhost:18080/api/dashboard/ws/dashboard ACCESS_TOKEN=dev-token
```

Environment knobs:
- `WS_RATE_TOKENS_PER_SEC`, `WS_RATE_BURST` to tune rate limit
- `WS_COMPRESSION_ENABLED` (reserved)
# HiveOps (Bee Hive)

Modern FastAPI backend with a Lit + Vite PWA for real‑time operational dashboards.

## Quick start

For tested, canonical setup steps, see `docs/GETTING_STARTED.md`.

Short version:
- Infra: `docker compose up -d postgres redis`
- Backend: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
- PWA: `cd mobile-pwa && npm ci && npm run dev` then open the dev URL

Health: `GET http://localhost:8000/health`
WebSocket: `ws://localhost:8000/api/dashboard/ws/dashboard`

## Tests

- Full: `make test`
- Fast lanes:
  - `make test-core-fast` (smoke + ws + prompt core)
  - `make test-backend-fast` (contracts + core + smoke)
  - `make test-prompt` (prompt optimization engines)

### WebSocket contract invariants
- All generic `error` frames include a `timestamp` string and `correlation_id`
- All `data_error` frames include `timestamp`, `error` message, and `correlation_id`
- `data_response` messages include `type`, `data_type`, `data`
- `pong` frames include a `timestamp`

### WebSocket safety & observability
- Per-connection token bucket rate limiting (20 rps, burst 40); over-limit requests receive an `error` periodically
- Inbound message size capped (64KB); oversize messages receive an `error`
- Max subscriptions per connection enforced; unknown subscriptions produce an `error`, responses return sorted unique lists
- Outbound frames include `correlation_id` for tracing
- Prometheus metrics at `/api/dashboard/metrics/websockets` include WS counters (messages sent/received/dropped, errors, connections)

## Policies

- No server-rendered dashboards; use API/WebSocket endpoints.
- Brand as "HiveOps"; default title in `mobile-pwa/src/components/layout/app-header.ts`.
 - Enterprise HTML templates are optional and gated by `ENABLE_ENTERPRISE_TEMPLATES` (default: disabled). Prefer PWA flows.

## Docs

**📚 Documentation Consolidation Complete!** All documentation has been consolidated into comprehensive guides.

### **Quick Start**
- **Master Index**: [DOCUMENTATION_MASTER_INDEX.md](DOCUMENTATION_MASTER_INDEX.md) - Single source of truth for all docs
- **Getting Started**: `docs/GETTING_STARTED.md`
- **Core Overview**: `docs/CORE.md`
- **Architecture**: `docs/ARCHITECTURE.md`

### **Consolidated Documentation**
- **Strategic Planning**: [STRATEGIC_PLAN_CONSOLIDATED.md](STRATEGIC_PLAN_CONSOLIDATED.md)
- **Testing Strategy**: [TESTING_STRATEGY_CONSOLIDATED.md](TESTING_STRATEGY_CONSOLIDATED.md)
- **Project Index**: [PROJECT_INDEX_CONSOLIDATED.md](PROJECT_INDEX_CONSOLIDATED.md)
- **Implementation Progress**: [IMPLEMENTATION_PROGRESS_CONSOLIDATED.md](IMPLEMENTATION_PROGRESS_CONSOLIDATED.md)
- **Product Documentation**: [docs/product/](docs/product/)

### **Legacy Documentation**
- **Archived**: `docs/archive/` - Previous fragmented documentation (archived)
