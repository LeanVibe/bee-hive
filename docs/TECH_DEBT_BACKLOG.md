# Technical Debt Backlog (First Pass)

Prioritized, actionable items gathered from static scans and repo policies. Severity: high → low.

## High
- CI robustness for heavy workflows
  - Location: .github/workflows/devops-quality-gates.yml
  - Issue: Fails before jobs start; add job-level validation and fast-fail diagnostics.
  - Fix: Validate script paths, add setup steps, or reduce scope; keep workflow_dispatch only.

- Complexity hotspots (refactor with tests)
  - Location: radon C/D/E classes/functions in `app/**`
  - Fix: Extract functions, reduce nesting, guard clauses, add tests around behavior.

## Medium
- TODO markers in production paths
  - Location: multiple (`app/observability/*`, `app/core/*`, etc.)
  - Fix: Convert TODOs into issues; implement or remove placeholders; add telemetry where promised.

- Dead-code candidates
  - Location: vulture findings (manual verification required)
  - Fix: Remove or move to `archive/` after confirming unreferenced by app/tests.

- Docs duplication
  - Location: many Quick Start / Getting Started sections
  - Fix: Canonicalize to docs/GETTING_STARTED.md; replace duplicates with links.

## Low
- Security hardening checks
  - Location: bandit report follow-ups
  - Fix: Address any B* findings pragmatically; suppress with justification where low-risk.

- Type coverage
  - Location: mypy ignores in some modules
  - Fix: Reduce ignores, add precise types to public interfaces.

## Notes
- Keep focused CI lanes green (tests/unit, tests/ws, tests/smoke).
- No reintroduction of server-rendered dashboard.
