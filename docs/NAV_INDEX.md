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
  - Templates (legacy removed): enterprise HTML optional under `app/templates/` (gated)
- **`mobile-pwa/`**: Lit + Vite PWA
  - Notable: `src/app.ts`, `src/services/backend-adapter.ts`, `src/services/websocket.ts`, `src/components/layout/app-header.ts`
  - Tests: `tests/e2e/*.spec.ts`, `src/services/__tests__/*.spec.ts`, `playwright.config.ts`
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

---

This index is generated to aid navigation by humans and CLI agents. See `docs/docs-manifest.json` for a structured machine-readable map.
