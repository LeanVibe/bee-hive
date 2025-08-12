## HiveOps Repository Navigation Index (Generated)

- **Last updated**: 2025-08-12
- **Scope**: Source files, configs, scripts, docs (excluding dependency caches and binary artifacts)
- **Core entry points**:
  - Backend app: `app/main.py` (FastAPI `app`)
  - WebSocket: `app/api/dashboard_websockets.py` → `/api/dashboard/ws/dashboard`
  - REST live data (compat): `app/api/dashboard_compat.py` → `/dashboard/api/live-data`
  - Health: `GET /health`
  - Mobile PWA: `mobile-pwa/src/app.ts` (router at `mobile-pwa/src/router/router.ts`)

### Top-level structure

- **`app/`**: FastAPI backend
  - Notable: `app/main.py`, `app/api/dashboard_websockets.py`, `app/api/dashboard_compat.py`, `app/api/ws_utils.py`
  - Models: `app/models/`
  - Core: `app/core/` (optimizers, engines, integrations, config)
  - Observability: `app/observability/`
  - Templates (legacy, deprecated): `app/dashboard/templates/`
- **`mobile-pwa/`**: Lit + Vite PWA
  - Notable: `src/app.ts`, `src/services/backend-adapter.ts`, `src/services/websocket.ts`, `src/components/layout/app-header.ts`
  - Tests: `tests/e2e/*.spec.ts`, `src/services/__tests__/*.spec.ts`, `playwright.config.ts`
- **`frontend/`**: Ancillary frontend assets (tailwind, store)
- **`docs/`**: Core documentation
  - Core: `CORE.md`, `ARCHITECTURE.md`, `PRD.md`, `PLAN.md`, `GETTING_STARTED.md`, `DOCS_INDEX.md`
  - Guides/Runbooks/Reference: `docs/guides/*`, `docs/runbooks/*`, `docs/reference/*`
  - Reports and archive: `docs/reports/*`, `docs/archive/*`
  - Manifest: `docs/docs-manifest.json`
- **`tests/`**: Python test suites
  - Suites: `tests/unit/`, `tests/smoke/`, `tests/ws/`, `tests/contracts/`, `tests/e2e-validation/`
- **`integrations/`**: VS Code extension, Kubernetes, Docker, CI examples
- **`infrastructure/`**: Monitoring, logging, disaster recovery assets
- **`schemas/`** and `resources/`: JSON/YAML schemas and API contracts
- **`scripts/`** and root `*.sh`: Setup/validate/run helpers (fast lanes)
- **`docker*` + `Dockerfile*`**: Compose files and images
- **`docs-site/`**: Static site sources
- **`memory-bank/`**: Contextual product/tech notes

### Backend (app/)
- APIs: `app/api/`
  - `routes.py`, `dashboard_websockets.py`, `dashboard_compat.py`, `dashboard_monitoring.py`
- Core: `app/core/`
  - Config: `config.py`
  - Engines: `workflow_engine.py`, `autonomous_development_engine.py`
  - Optimizers: `prompt_optimizer.py`, `gradient_optimizer.py`
  - Integrations: `github_api_client.py`, `github_quality_integration.py`
- Models: `app/models/` (e.g., `agent.py`, `coordination.py`, `session.py`)
- Observability: `app/observability/` (e.g., `hooks.py`, `real_time_processor.py`)
- CLI: `app/cli.py`

### Mobile PWA (mobile-pwa/)
- App bootstrap: `src/app.ts`, `src/main.ts`
- Services: `src/services/backend-adapter.ts`, `src/services/websocket.ts`, `src/services/system-health.ts`
- Components: `src/components/**`
- Types: `src/types/ws-messages.d.ts`
- Config: `vite.config.ts`, `tsconfig.json`, `tailwind.config.js`
- Tests: `tests/e2e/*`, `src/services/__tests__/*`, `src/tests/*`

### Tests (Python)
- Smoke: `tests/smoke/*` (health, metrics, WS handshake)
- WebSocket contracts: `tests/ws/*`
- Contracts: `tests/contracts/*`
- Unit: `tests/unit/*`

### Infrastructure and Integrations
- Kubernetes: `integrations/kubernetes/*` (Helm chart, manifests)
- Docker: `Dockerfile*`, `docker-compose*.yml`, `integrations/docker/*`
- Monitoring: `infrastructure/monitoring/*` (Prometheus, Grafana dashboards)
- Logging: `infrastructure/logging/*`

### Schemas and API Contracts
- WS schema: `schemas/ws_messages.schema.json`
- Live dashboard schema: `schemas/live_dashboard_data.schema.json`
- Additional: `resources/schemas/*`, `resources/api_contracts/*`

### Scripts and Utilities
- Quick start: `start-fast.sh`, `stop-fast.sh`, `setup-fast.sh`, `scripts/start.sh`, `scripts/test.sh`
- Validation: `scripts/validate_setup_time.sh`, `scripts/sandbox.sh`
- Demo hooks: `demo_consolidated_hooks/*.sh`

### Documentation (docs/)
- Source of truth:
  - `docs/CORE.md` — Product overview and scope
  - `docs/ARCHITECTURE.md` — Implementation-aligned architecture
  - `docs/PRD.md` — Product requirements
  - `docs/PLAN.md` — Current plan and next steps
  - `docs/GETTING_STARTED.md` — Canonical local setup
  - `docs/DOCS_INDEX.md` — Manual curated map
- Index manifest: `docs/docs-manifest.json` (machine-readable)

### Configuration and Project Files
- Python: `pyproject.toml`, `requirements.txt`, `requirements-agent.txt`, `pytest.ini`, `alembic.ini`
- Node/TS: `package.json` (root, PWA, integrations, frontend), `tsconfig.json`
- CI: `.github/workflows/*.yml` (fast lanes, lint)
- Env: `.env`

### Migrations
- Alembic env: `migrations/env.py`
- Versions: `migrations/versions/*`

### Known Policies
- Backend Dashboard Policy: `.cursor/rules/backend-dashboard-policy.mdc`

---

This index is generated to aid navigation by humans and CLI agents. See `docs/docs-manifest.json` for a structured machine-readable map.
