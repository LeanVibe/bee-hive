You are taking over as the Cursor Agent on the `auth-foundation` branch of LeanVibe Bee Hive (HiveOps). Your mission: finish CI stabilization and complete must-have scope for Epic 2 (Offline-first) and Epic 3 (WS SLOs), adhering to our governance rails (Epic 4) and engineering protocol.

Context you should know:
- FastAPI backend entry: `app/main.py` (exports `app`); CI startup now skips heavy init when `CI` or `SKIP_STARTUP_INIT` is set.
- Mobile PWA (Lit + Vite): `mobile-pwa/`. Backend adapter/WS wiring: `mobile-pwa/src/services/backend-adapter.ts`. System health: `mobile-pwa/src/services/system-health.ts`.
- WS endpoint: `/api/dashboard/ws/dashboard`. Contract governance: `/api/dashboard/websocket/contract`. PWA emits a version-mismatch banner when client contract unsupported.
- Tests mock Redis/DB in `tests/conftest.py`. Avoid real services in fast lanes. Use provided fixtures (`test_app`, `test_client`, etc.).
- CI Workflows: 
  - `.github/workflows/ci.yml` (backend, pwa) 
  - `.github/workflows/ci-fast.yml` (backend-fast, pwa lanes, schema drift)
  - `.github/workflows/fast-type-test.yml` (ruff + mypy + smoke)
  - `ruff check` is used. Fast lanes export `SKIP_STARTUP_INIT`.

Immediate priorities (must-haves, Pareto):
1) Ensure PR #3 (auth-foundation) goes green end-to-end.
   - If Ruff still flags files, fix import ordering and unused imports in changed files. Prefer code fixes over disables. See `.ruff` config in `pyproject.toml`.
   - Confirm backend-fast no longer hangs; it should run `pytest -q tests/smoke tests/ws` without DB/Redis. If any test assumes real Redis, adapt to fixtures or add light mocks.
2) Epic 2 — Offline-first resiliency:
   - Add durable retry with jitter and max attempts per queued op in `mobile-pwa` sync queue.
   - Surface conflicts: prefer last-write-wins but show a badge/tooltip on reconciled items when server overwrote client.
   - UI: persistent mini-banner or badge when `queuedCount > 0` with “Sync now” action.
   - Tests:
     - Unit tests for queue state machine and conflict surfacing (Vitest).
     - E2E Playwright: offline CRUD while offline, then online reconciliation with visible cues.
3) Epic 3 — WS SLOs enforcement:
   - Implement per-connection send rate limiter (bytes/sec) in server WS manager; drop/backoff and increment counters.
   - Track latency distribution (p50/p95/p99) across sends; expose via Prometheus in `app/api/dashboard_prometheus.py` and/or `app/core/prometheus_exporter.py`.
   - Tests: unit/integration for limiter and metrics; expand k6 WS scenarios to assert bps ceilings and latency SLOs.

Guidelines and guardrails:
- TDD: write failing tests first, minimal code to pass, refactor clean.
- YAGNI: implement only what the tests and user journey require.
- Vertical slices: complete feature behavior across backend, PWA, and tests before moving on.
- No server-rendered dashboard; PWA + API/WS only.
- Keep governance:
  - Schema changes to `schemas/ws_messages.schema.json` require versioning, migration notes, and labels.
  - PWA types must be generated from schema; CI fails on drift.

Helpful entry points:
- WS manager: `app/api/dashboard_websockets.py`. Add rate limiter there and counters (bytes sent/recv totals already exist; add bps limiter and latency histograms).
- Prometheus surfacing: `app/api/dashboard_prometheus.py` and `app/core/prometheus_exporter.py`.
- PWA offline queue: `mobile-pwa/src/services/` (search queue/sync logic) and UI badges/banners under `mobile-pwa/src/components/`.
- k6 scenarios live under performance/devops docs or scripts (search `k6`); extend to include bps/latency constraints.

Definition of done for you:
- All CI lanes green on PR #3.
- Offline-first queue retry+jitter+max-attempts implemented with tests; visible UI hint and “Sync now”.
- WS SLOs limiter + latency metrics implemented with tests and updated k6.
- Commit in vertical slices, with concise messages like:
  - `feat(offline): retry+jitter+max-attempts for sync queue + tests`
  - `feat(ws-slo): per-conn send rate limiter + latency metrics + tests`
  - `test(e2e): offline CRUD reconciliation with UI cues`
  - `chore(ci): minor ruff fixes in changed files`

Local dev quickstart:
- Backend: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
- PWA: `cd mobile-pwa && npm ci && npm run dev`
- E2E: `cd mobile-pwa && npx playwright test`

Remember: prioritize must-have behavior and keep tests fast and deterministic. If blocked by uncertainty, prefer minimal mocks and move forward.
