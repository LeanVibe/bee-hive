# HiveOps (Bee Hive)

Modern FastAPI backend with a Lit + Vite PWA for real‑time operational dashboards.

## Quick start

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
- All generic `error` frames include a `timestamp` string
- All `data_error` frames include `timestamp` and `error` message

## Policies

- No server-rendered dashboards; use API/WebSocket endpoints.
- Brand as "HiveOps"; default title in `mobile-pwa/src/components/layout/app-header.ts`.

## Docs

See `docs/extra/prompts.md` and `docs/core/prompt-optimization-system-prd.md` for implementation notes and design.
