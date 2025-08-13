You are a senior Cursor agent stepping into the HiveOps repo to continue backend and PWA platform work. Read this fully, then execute without asking for confirmations unless blocked.

High-level context
- Backend: FastAPI (`app/main.py`), WebSockets under `/api/dashboard/ws/*`. Limits, contract, and health/stats endpoints exist. WS manager lives in `app/api/dashboard_websockets.py`.
- PWA: Lit + Vite in `mobile-pwa/`. Types generated from `schemas/ws_messages.schema.json` and enforced in CI. WebSocket client in `mobile-pwa/src/services/websocket.ts`.
- CI: PR lanes run Ruff and a focused pytest subset; PWA type drift gates build. Coverage gate is 45% (backend).

Current branch/status (auth-foundation, 2025-08-13)
- Fixed Ruff issues and an `UnboundLocalError` in `dashboard_websockets.py`; pushed.
- CI is running. Previous failures were lint and backend-fast due to the error; those should now be resolved.
- Smoke auth tests in this branch target `/api/v1/auth/login` and `/api/v1/auth/refresh`; they currently 404 locally/CI. Coverage gate also fails. Epic 4 is complete; Epics 2 and 3 partially complete; Epic 1 mostly complete except for mounting/implementing auth endpoints.

Immediate priorities (do these now)
1) Implement and mount backend auth routes to satisfy smoke tests and coverage gate.
   - Add `app/api/auth.py`:
     - `POST /login` (expects `{email, password}`) -> `{access_token, refresh_token}`
     - `POST /refresh` (expects `{refresh_token}`) -> `{success: true, access_token}`
     - `POST /logout` (optional no-op for now) -> `{success: true}`
     - `GET /me` (requires `Authorization: Bearer <access_token>`) -> user JSON with `email`
   - Use local JWT in `app/core/auth.py` if not present; keep deterministic and keyless in dev (e.g., HMAC with a default secret if env missing). Ensure verify path matches test expectations.
   - Mount router in `app/main.py`: `app.include_router(auth_router, prefix="/api/v1/auth")`.
   - Keep payloads and shapes as smoke tests expect.
   - Add minimal unit tests in `tests/unit/test_auth_*.py` to raise coverage above 45%.

2) WS metrics endpoint for Epic 3 completion
   - Expose Prometheus metrics for WS manager (counters/gauges already tracked). Add endpoint `/api/dashboard/metrics/websockets` that emits standard Prometheus text format.
   - Add a lightweight test asserting metric names (no strict values) to avoid flakes in PR lanes.

3) Rate limit tests (Epic 3)
   - Unit test token-bucket behavior in the manager. Avoid long sleeps; simulate time where possible.
   - WS integration test: send >N messages quickly and assert at least one `error` for rate limit and that not all messages are processed.

4) Offline reconciliation (Epic 2 follow-ups)
   - Define conflict policy (last-write-wins with `correlation_id` tie-breaker).
   - Implement reconciliation logging and backoff in PWA; add unit tests. Keep E2E minimal.

How to work (guardrails)
- Vertical slices only. For each slice: add failing test → implement minimal passing code → keep Ruff/tests green.
- Don’t introduce external service dependencies; keep dev/test deterministic (no external IdP).
- Update `docs/PLAN.md` as you make meaningful progress; keep it the living source of truth.
- Keep changes scoped; when an epic’s remaining items are complete, commit and push with a clear message.

Concrete action plan to start
- Backend auth
  - Create `app/api/auth.py` and (if needed) `app/core/auth.py` helpers (issue/verify tokens, password check stub that accepts the default admin from env).
  - Mount router in `app/main.py` under `/api/v1/auth`.
  - Ensure WS auth (already present in `dashboard_websockets.py`) continues to verify tokens in JWT mode.
  - Tests: add unit tests for token issue/verify and refresh; re-run `tests/smoke/test_auth_login_and_protected.py`.
- Coverage
  - If coverage <45%, add unit tests for `app/api/ws_utils.py` and auth utilities.
- WS metrics endpoint
  - Add endpoint and a test asserting metric keys exist.

References
- Plan: `docs/PLAN.md` (updated with status, missing work, and acceptance).
- WS manager: `app/api/dashboard_websockets.py`.
- WS utils: `app/api/ws_utils.py`.
- PWA websocket: `mobile-pwa/src/services/websocket.ts`.

Deliverable for this handoff
- Implement auth router and mount to pass auth smoke tests and lift coverage past 45%.
- Add WS metrics endpoint and basic test.
- Update docs as you go; keep CI green.
