# XP-Inspired Autonomous Development Workflow (HiveOps)

## Principles
- Single-piece flow, trunk-based; small vertical slices; TDD always.
- Contracts as source of truth (REST/WS schemas); generate PWA types from schemas.

## Branching & Commits
- Branch per slice; conventional commits; keep PRs small (<300 LOC).
- Auto-squash on merge; PR template enforces tests/Docs/Contracts updated.

## Red / Green / Refactor (Automatable)
1) Red: write a failing test (unit/contract/WS/PWA)
2) Green: implement minimal code to pass
3) Refactor: remove duplication, improve names, update docs

## Contracts & Schemas
- Maintain JSON Schemas for REST and WS messages.
- CI validates responses against schemas; PWA types generated from schemas.
- Version schemas with semver; alert on breaking diffs.

## Test Hierarchy
- Tier 0 (pre-commit, <1m): unit + contract + PWA vitest
- Tier 1 (PR, ~3–5m): focused contracts/ws/smoke + vitest; coverage ratchet
- Tier 2 (nightly): extended integration, light chaos, minimal Playwright smoke
- Tier 3 (weekly): load, resilience, dependency update validation

## CI/CD (Fast-fail, Self-healing)
- Lint/type/format → Tier 1 tests → build artifacts → (optional) deploy preview → synthetic probes
- Canary gate: synthetic checks; auto-rollback on failure

## Observability-as-Tests
- Synthetic monitors hit /health, /metrics, live-data, and WS; auto-create issues with traces.
- Error budgets: freeze merges to core areas when budgets are burned.

## Tooling & Automation
- Renovate/Dependabot (weekly) for Actions/npm; batch updates.
- Codegen: regenerate PWA types from schemas on merge (commit bot).
- Pre-commit: ruff/black/mypy/pytest -k "unit or contract" (fast Tier 0).

## Quality Ratchets
- Coverage: +5% per sprint on targeted modules (e.g., 30% → 35% → 40%).
- Light mutation tests nightly; ratchet quarterly.
- Performance budgets for critical endpoints.

## Async Ceremonies
- Daily bot digest: gate status, flake list, slowest tests, error budget.
- Weekly planning: max 3 slices; each begins with a test.

## Slice Template (Vertical)
- Write/update schema (REST/WS) + failing contract test
- Implement endpoint/handler + minimal PWA adapter change
- Unit tests + vitest for adapters/helpers
- Observability: add one metric/log and an assertion in tests
- Docs: README/API reference snippet update

## Fast-Start Commands
- Backend: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
- PWA: `cd mobile-pwa && npm ci && npm run dev`
- Focused tests: `pytest -q` and `cd mobile-pwa && npm run -s test`
- Core verify (REST): `scripts/verify_core.sh`

## Immediate Adoption Checklist
- [x] Focused CI (contracts/ws/smoke + vitest) with coverage gate
- [x] PWA-first quickstart in README
- [ ] WS message schemas + PWA type generation
- [ ] Nightly Tier 2 job (Playwright smoke + mutation tests light)
- [ ] Canary gate & auto-rollback
- [ ] Coverage ratchet to 40% after 2–3 more unit tests
