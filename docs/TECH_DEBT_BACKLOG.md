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

---

## Findings — Ruff (style/lint)

- Result: No lint violations reported by `ruff check`.
- Config deprecations detected in `pyproject.toml`:
  - `[tool.ruff]` top-level linter keys are deprecated; migrate to `[tool.ruff.lint]`:
    - `select` -> `lint.select`
    - `ignore` -> `lint.ignore`
    - `unfixable` -> `lint.unfixable`
    - `isort` -> `lint.isort`
    - `per-file-ignores` -> `lint.per-file-ignores`
  - Action: Update `pyproject.toml` to new sections and run Ruff again.

## Findings — Mypy (typing)

- Scale: 11,289 errors across 1,000+ lines reported; top categories:
  - `attr-defined` (~2,185): attributes used that types don’t declare (often ORM columns vs. runtime attrs)
  - `no-untyped-def` (~1,893): missing function type hints
  - `arg-type` / `assignment` / `call-arg` (~3,197 combined): incompatible types passed/assigned
  - `union-attr` (857): Optional handling issues
  - `var-annotated` (506): variables need explicit annotations
- Heaviest files (top 10 by error count):
  - `app/core/recovery_manager.py` (~206)
  - `app/core/config.py` (~146)
  - `app/api/dashboard_task_management.py` (~123)
  - `app/core/consolidation_engine.py` (~122)
  - `app/core/command_templates.py` (~109)
  - `app/core/self_modification/self_modification_service.py` (~102)
  - `app/api/v1/comprehensive_dashboard.py` (~102)
  - `app/core/context_engine_integration.py` (~97)
  - `app/core/orchestrator.py` (~94)
  - `app/api/v1/team_coordination.py` (~84)
- Root causes and plan:
  - Pydantic v2 `Field` kwargs (`min_items`, `max_items`, `env`) used in v1 style → update to v2 equivalents or model_config.
  - SQLAlchemy models mixing `Column[...]` with business-return types → add `Mapped[...]` annotations and accessor methods.
  - Broad `Collection` vs `list`/`dict` misuse; Optional defaults violating `no_implicit_optional`.
  - Action: Introduce an incremental mypy target focusing on top 5 files, enforce `--warn-unused-ignores`, and add `py.typed` where needed.

## Findings — Bandit (security)

- Total findings: 383
  - Severity: HIGH=6, MEDIUM=41, LOW=336
  - Common tests triggered:
    - B311: use of `random` for security (~132) → OK for non-crypto; otherwise use `secrets`/`os.urandom`.
    - B110: try/except pass (~65) → replace with explicit logging or handling.
    - B603/B607: subprocess usage (~63) → ensure safe arguments; prefer `shlex.split`, avoid shell=True.
    - B105/B106: hardcoded password/crypt (~38) → verify these are placeholders/tests; move to secrets management.
    - B404: import subprocess (~29) → acceptable with guardrails.
    - B608: SQL injection potential (~9) → parameterize queries.
- Action:
  - Audit HIGH first; document accepted risks with rationale.
  - Add helper wrappers for subprocess and parameterized DB ops; enforce via code review checks.

## Findings — Radon (complexity & maintainability)

- Cyclomatic complexity (blocks by rank): A≈10,740, B≈2,197, C≈678, D≈54, E≈5
- Files with worst ranks:
  - E: `app/core/communication_analyzer.py`, `app/core/performance_metrics_collector.py`, `app/core/enterprise_roi_tracker.py`, `app/dashboard/coordination_dashboard.py`, `app/api/v1/sleep_wake_vs7_1.py`
  - D: `app/core/intelligent_workflow_automation.py`, `app/core/enhanced_jwt_manager.py`, `app/core/multi_agent_commands.py`, `app/core/memory_hierarchy_manager.py`, `app/core/workflow_engine.py`, `app/core/capacity_manager.py`, `app/core/enhanced_context_consolidator.py`, `app/core/sleep_analytics.py`, `app/core/enhanced_security_audit.py`, `app/core/optimized_embedding_pipeline.py`
- Maintainability Index (buckets): D≈441, C≈16, B≈1, A≈29; lowest MI include:
  - `app/core/intelligent_workflow_automation.py`, `app/core/production_orchestrator.py`, `app/core/advanced_repository_management.py`, `app/core/sleep_wake_system.py`, `app/core/self_optimization_agent.py`, `app/core/orchestrator.py`, `app/core/consolidation_engine.py`
- Action:
  - Prioritize E/D ranks for refactoring with tests; extract pure functions, reduce branching, add guards.
  - Schedule MI D files into an incremental refactor queue; enforce max function length in CI for these paths.
