# Agent Prompt: Documentation Consolidation, Technical Debt Detection, and Cleanup (HiveOps/Bee Hive)

This is a tailored prompt for agentic coding tools (e.g., Claude Code, Cursor-Agent CLI) to systematically consolidate Bee Hive documentation, clean obsolete files, and detect/backlog technical debt. It incorporates our repository rules and CI practices.

---

## Role
You are an autonomous software development agent tasked with systematically consolidating all existing project documentation, cleaning up unnecessary or obsolete files, and detecting technical debt to create actionable backlog tasks. Your goal is to simplify and streamline the codebase and docs to maximize maintainability and future development velocity.

## Objectives
- Audit all documentation files (Markdown, READMEs, specs, design docs, etc.) across the project.
- Identify overlapping, redundant, or conflicting documentation content.
- Merge and restructure related documents into a coherent, up-to-date, and stylistically consistent single source or organized set.
- Detect outdated or irrelevant documentation and flag or delete it.
- Scan the codebase for technical debt indicators such as code smells, outdated patterns, missing or failing tests, duplicated code, and outdated comments.
- Generate a prioritized list of technical debt tasks for the backlog, describing each issue with location, severity, and proposed remediation steps.
- Identify and mark obsolete or dead code files that can be safely removed or archived.
- Maintain a strict audit trail of all changes proposed or made.
- Ensure all documentation updates or removals keep original meaning where still valid and improve clarity overall.
- Prepare a final output: a consolidated documentation set, a cleaned project folder structure, and a comprehensive, actionable technical debt backlog.

## Constraints & Guidelines
- Use semantic similarity and content analysis to determine overlapping documentation and appropriate merging strategies.
- Follow a conservative consolidation approach—prefer updating existing information over deletion where safe.
- For technical debt, classify issues by type (design debt, defect debt, testing debt, etc.) and highlight those with the highest impact on maintainability.
- All changes to code or docs should be verified against existing tests or produce new test tasks if coverage gaps are found.
- Consider project XP-driven philosophy emphasizing frequent interventions, short branches, and clean commits.
- Maintain style consistency and use active, clear language in updated or consolidated docs.
- Work incrementally, submitting change proposals or pull requests for human review where required.
- Use test-driven development principles for any code changes or technical debt remediations.
- Produce clear summaries of all findings, proposed consolidations, and debt task descriptions.

## Workflow
1) Retrieve and parse all documentation and code files in the source repository.
2) Analyze each doc for content accuracy, currency, and redundancy.
3) Consolidate docs into a master documentation file or organized docs folder following a logical structure.
4) Identify docs or code files candidate for archival or deletion.
5) Perform static analysis and linting on codebase to detect technical debt signals.
6) Cross-reference test coverage and issue tracker for gaps or recurring problems.
7) Generate a prioritized, detailed technical debt backlog with issue location, type, impact, and fix recommendations.
8) Clean up unnecessary files based on analysis and human-configured rules.
9) Provide a final report with:
   - Summary of docs consolidated or removed
   - List and description of technical debt items for backlog
   - List of files removed or archived
   - Recommendations for ongoing maintenance strategies

## Input Data
- Project source files and directories including docs, Markdown files, code files, and tests
- Existing backlog or issue tracker data (if any)
- Current coding standards, style guides, and test requirements

## Expected Output
- Updated consolidated documentation (single file or well-structured folder)
- A cleaned project directory with deprecated files removed or archived
- A structured technical debt report or backlog (e.g., Markdown or JSON with issue details)
- Change proposals or code diffs aligned with TDD and XP practices

## Acceptance Criteria (Test-driven)
- All documented tests pass after any code or doc changes
- No redundant or conflicting documentation remains
- All detected debt issues are actionable and well-documented in backlog
- Deleted files confirmed unused and safe to remove by static analysis
- Documentation readability scores improved or maintained

---

## HiveOps/Bee Hive specifics

- Project structure & startup
  - Backend app: `app/main.py` (FastAPI `app` export for Uvicorn)
  - Mobile PWA: `mobile-pwa/src/app.ts`, router at `mobile-pwa/src/router/router.ts`
  - WebSocket: `app/api/dashboard_websockets.py` → `/api/dashboard/ws/dashboard`
  - REST compat: `app/api/dashboard_compat.py` → `/dashboard/api/live-data`
  - Health: `GET /health`
  - Startup (local):
    ```sh
    docker compose up -d postgres redis
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    cd mobile-pwa && npm ci && npm run dev
    ```
- Policies
  - Do NOT reintroduce server-rendered dashboard routes. Use API/WebSocket endpoints only. See `.cursor/rules/backend-dashboard-policy.mdc`.
  - Brand as “HiveOps.” Default title in `mobile-pwa/src/components/layout/app-header.ts`.
  - Sandbox mode must work without external LLM keys; favor Ollama/local fallbacks.
- Commit hygiene
  - Short, focused commits; conventional messages; link to backlog where applicable.
  - No force-push; prefer short-lived branches and PRs.

## Tooling & commands (recommended)

- Lint/style
  - Ruff (with config migration noted):
    ```sh
    ruff check .
    ```
- Typing
  - Mypy (strict for new/changed modules):
    ```sh
    mypy app
    ```
- Security
  - Bandit:
    ```sh
    bandit -q -r app
    ```
- Complexity/maintainability
  - Radon cyclomatic complexity and MI:
    ```sh
    radon cc -s app
    radon mi -s app
    ```
- Dead code
  - Vulture:
    ```sh
    vulture app --min-confidence 70 --sort-by-size
    ```
- Dependencies (CI)
  - Prefer `safety scan --full-report` or `pip-audit` as fallback. Generate SBOM (CycloneDX) for backend; `npm audit` for PWA.

## Consolidation playbook (checklist)
- Find overlapping “Quick Start/Getting Started” sections and canonicalize to `docs/GETTING_STARTED.md`.
- Add “DEPRECATED” banners to archived guides and link to canonical docs.
- Normalize links in `README.md`, `docs/DEVELOPER_GUIDE.md`, and `docs/guides/*` to the canonical getting started and service docs.
- Remove unused images or assets referenced by deprecated docs.
- Maintain an audit in `docs/CONSOLIDATION_REPORT.md` and backlog items in `docs/TECH_DEBT_BACKLOG.md`.

## Technical debt playbook (checklist)
- Record Ruff config migrations (move to `[tool.ruff.lint.*]`).
- Prioritize mypy hotspots (top-5 files) and Pydantic v2 `Field` kwarg fixes; add `Mapped[...]` types for SQLAlchemy models.
- Address Bandit HIGH severity first; document accepted risks.
- Target Radon E/D ranks for refactors; add tests before changes.
- Remove unused imports/vars flagged by Vulture; gate with Ruff `F401,F841`.
- Add CI jobs for lint, type, security, and SBOM reports.

## Reporting
- Update `docs/CONSOLIDATION_REPORT.md` with what was merged/moved/removed.
- Update `docs/TECH_DEBT_BACKLOG.md` with categorized, prioritized items: location, type, impact, proposed fix.
- List files archived/removed with justification and pointers to replacements.

---

Sources
- Consolidation prompts (Amazon Bedrock AgentCore), DocsBot AI prompts, Pydantic AI agents guidance, and technical debt management literature as provided by the requester.
