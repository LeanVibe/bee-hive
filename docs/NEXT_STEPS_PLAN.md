# Next Steps Plan (Autonomous Execution)

## Goals
- Maximize hands-off operation with high-signal gates (CI, Nightly, Canary)
- Tighten contracts (REST/WS) and codegen types to prevent drift
- Incrementally raise coverage and reduce flakiness

## Backlog (P0 → P1)

### P0 (Do now)
1) CI: Generate TS types from JSON schemas in PR workflow and fail on diff
2) Nightly: Add light mutation tests on a small, critical module (e.g., app/api/dashboard_compat.py)
3) Reduce warnings: remove remaining AsyncMock warnings by replacing listen() consumption in tests if still present
4) Raise coverage gate for targeted modules to 45% after adding 1-2 more small tests

### P1 (Next)
5) Expand WS schema to include error/alert messages and validate
6) Add small WS handler error-path unit (invalid message type → error log)
7) Add core verify script for local dev (runs REST checks + WS handshake) and hook into `make verify-core`
8) Extend Playwright smoke to assert one live-data field renders (fast and stable)

## Implementation Outline
- Add schema generation step to CI, commit failure if `git diff` detects changes in `mobile-pwa/src/types/ws-messages.d.ts`
- Add mutation test job in nightly using `mutmut` limited to `app/api/dashboard_compat.py`
- Add 1-2 unit tests (backend) to increase coverage to 45% (e.g., WS stats shape, /metrics fallback path)
- Update Makefile or scripts to include verify-core

## Acceptance Criteria
- CI fails if schema TS types drift from JSON schemas
- Nightly runs mutation tests and reports score; not blocking
- Coverage gate at 45% passes locally
- No remaining AsyncMock warnings in focused suites
