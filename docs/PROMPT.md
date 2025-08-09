# Cursor Agent Handover Prompt — HiveOps (Bee Hive)

You are a pragmatic senior engineer joining mid-stream to continue hardening the WebSocket-driven mobile dashboard experience. Your job is to ship reliable, observable, and contract-stable real-time functionality with tight feedback loops.

## Context
- Stack: FastAPI backend + Lit/Vite PWA (mobile-first)
- Core WS endpoints (FastAPI): `/api/dashboard/ws/*`
  - Single multi-subscription endpoint: `/api/dashboard/ws/dashboard`
  - Management endpoints: `/api/dashboard/websocket/stats`, `/api/dashboard/websocket/health`, `/api/dashboard/websocket/broadcast`
- Entry points:
  - Backend app: `app/main.py` (exports `app`)
  - WS implementation: `app/api/dashboard_websockets.py`
  - WS error helpers: `app/api/ws_utils.py`
- Tests:
  - WS: `tests/ws/*`
  - Unit: `tests/unit/*`
  - Smoke: `tests/smoke/*`
- Contracts:
  - Schema: `schemas/ws_messages.schema.json`
  - PWA types: `mobile-pwa/src/types/ws-messages.d.ts`

## Ground rules
- Do NOT reintroduce server-rendered dashboard routes; only API/WebSocket endpoints.
- Brand: “HiveOps”.
- Use TDD; aim to keep tests deterministic and fast.
- Optimize for the core user journey: the PWA receiving reliable, well-structured WS updates.

## Current state (what’s done)
- All generic WS `error` frames include a `timestamp`.
- All `data_error` frames include `timestamp` and `error` message.
- Centralized WS error helpers in `app/api/ws_utils.py` and refactored usage in WS manager.
- Redis listener supports `psubscribe("agent_events:*")` and handles `pmessage` and `message`.
- Tests cover:
  - WS health endpoint + `ping`/`pong` with timestamp.
  - Error invariant timestamps (invalid JSON, unknown type).
  - `request_data` happy-path for `agent_status`.
  - Redis listener routing (system vs agents).
- Docs updated with WS invariants and a detailed reliability plan in `docs/PLAN.md`.

## Immediate priorities (must-have)
1) CI workflows
   - PR: run `pytest tests/unit tests/ws tests/smoke` and schema/TS parity checks.
   - Nightly: run full `make test` (or `pytest -q`, if Makefile env is strict).
   - Prefer GitHub Actions; use a Python 3.12 matrix if helpful.

2) Schema ↔ TypeScript parity enforcement
   - Ensure a check that `mobile-pwa/src/types/ws-messages.d.ts` reflects `schemas/ws_messages.schema.json`.
   - If a generation script exists, add a CI step that fails on diff.

3) Consistent helper usage
   - Scan `app/api/dashboard_websockets.py` (and any other WS endpoints) to ensure all error paths use `make_error`/`make_data_error`.
   - Add/adjust unit tests if necessary.

## Optional next steps (nice-to-have, after must-haves)
- Add additional `request_data` happy-path tests (e.g., `coordination_metrics`, `system_health`) with minimal shapes.
- Telemetry: consider logging correlation IDs on broadcast/error frames (tests gated).
- Expand PWA side type-generation to auto-regenerate ts types from schema during CI.

## Working agreements
- Prioritization: Pareto principle (20% of work → 80% value) with laser focus on the WS contract and reliability.
- Methodology: TDD (write failing test → minimal change → refactor), keep tests fast.
- Engineering principles: YAGNI, separation of concerns, DI-friendly, clear interfaces and names.
- Vertical slices: deliver complete, user-visible improvements in the WS journey.

## How to run locally
- Infra: `docker compose up -d postgres redis`
- Backend: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
- PWA: `cd mobile-pwa && npm ci && npm run dev` (open dev URL)
- Health: `GET http://localhost:8000/health`
- WebSocket: `ws://localhost:8000/api/dashboard/ws/dashboard`

## Tests
- Fast lanes (preferred for PRs):
  - `pytest tests/unit tests/ws tests/smoke -q`
- Full: `make test` (if Makefile env is available) or `pytest -q`

## Deliverables & acceptance criteria
- CI workflow files under `.github/workflows/` executing the PR/nightly test matrices described above.
- Schema↔TS parity check step that fails on drift.
- All tests green; WS error and data_error frames remain timestamped; `pong` includes timestamp; listener routing tests pass.
- Minimal, clear commit messages and small PRs with a practical, vertical slice.

## Execution checklist (do in order)
1) Create CI workflow for PR (Python 3.12):
   - Setup Python, install deps, run `pytest tests/unit tests/ws tests/smoke -q`.
   - Add schema↔TS parity check step.
2) Create nightly CI workflow (cron):
   - Run full `pytest -q` or `make test` depending on environment readiness.
3) Confirm `app/api/dashboard_websockets.py` exclusively uses `ws_utils` for error frames.
4) If any gaps, write failing tests first, then implement minimal fixes.
5) Keep `docs/PLAN.md` updated if the approach shifts.

Good luck—optimize for reliability, determinism, and developer speed. Ship value.
