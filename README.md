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

## Docs

- Getting started: `docs/GETTING_STARTED.md`
- Core overview: `docs/CORE.md`
- Architecture: `docs/ARCHITECTURE.md`
- Index: `docs/DOCS_INDEX.md`
- Agent prompt for consolidation/debt cleanup: `docs/AGENT_PROMPT_CONSOLIDATION.md`
