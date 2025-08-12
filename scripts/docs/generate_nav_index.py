#!/usr/bin/env python3
import os
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
NAV_MD = DOCS / "NAV_INDEX.md"
MANIFEST = DOCS / "docs-manifest.json"

SECTIONS = [
    ("app/", ["app/main.py", "app/api/dashboard_websockets.py", "app/api/dashboard_compat.py", "app/api/ws_utils.py"],
     "FastAPI backend"),
    ("mobile-pwa/", ["mobile-pwa/src/app.ts", "mobile-pwa/src/services/backend-adapter.ts", "mobile-pwa/src/services/websocket.ts", "mobile-pwa/src/components/layout/app-header.ts"],
     "Lit + Vite PWA"),
    ("docs/", ["docs/CORE.md", "docs/ARCHITECTURE.md", "docs/PRD.md", "docs/PLAN.md", "docs/GETTING_STARTED.md", "docs/DOCS_INDEX.md", "docs/docs-manifest.json"],
     "Documentation tree"),
]

HEADER = """## HiveOps Repository Navigation Index (Generated)\n\n- **Last updated**: {date}\n- **Scope**: Source files, configs, scripts, docs (excluding dependency caches and binary artifacts)\n- **Core entry points**:\n  - Backend app: `app/main.py` (FastAPI `app`)\n  - WebSocket: `app/api/dashboard_websockets.py` → `/api/dashboard/ws/dashboard`\n  - REST live data (compat): `app/api/dashboard_compat.py` → `/dashboard/api/live-data`\n  - Health: `GET /health`\n  - Mobile PWA: `mobile-pwa/src/app.ts` (router at `mobile-pwa/src/router/router.ts`)\n\n"""

TOP = """### Top-level structure\n\n"""

FOOTER = """\n---\n\nThis index is generated to aid navigation by humans and CLI agents. See `docs/docs-manifest.json` for a structured machine-readable map.\n"""

def load_manifest():
    try:
        return json.loads(MANIFEST.read_text())
    except Exception:
        return {}


def write_nav():
    manifest = load_manifest()
    lines = []
    lines.append(HEADER.format(date=datetime.utcnow().date()))
    lines.append(TOP)
    # app
    lines.append("- **`app/`**: FastAPI backend")
    lines.append("  - Notable: `app/main.py`, `app/api/dashboard_websockets.py`, `app/api/dashboard_compat.py`, `app/api/ws_utils.py`")
    lines.append("  - Models: `app/models/`")
    lines.append("  - Core: `app/core/` (optimizers, engines, integrations, config)")
    lines.append("  - Observability: `app/observability/`")
    lines.append("  - Templates (legacy removed): enterprise HTML optional under `app/templates/` (gated)")
    # pwa
    lines.append("- **`mobile-pwa/`**: Lit + Vite PWA")
    lines.append("  - Notable: `src/app.ts`, `src/services/backend-adapter.ts`, `src/services/websocket.ts`, `src/components/layout/app-header.ts`")
    lines.append("  - Tests: `tests/e2e/*.spec.ts`, `src/services/__tests__/*.spec.ts`, `playwright.config.ts`")
    # docs
    lines.append("- **`docs/`**: Core documentation")
    lines.append("  - Core: `CORE.md`, `ARCHITECTURE.md`, `PRD.md`, `PLAN.md`, `GETTING_STARTED.md`, `DOCS_INDEX.md`")
    lines.append("  - Guides/Runbooks/Reference: `docs/guides/*`, `docs/runbooks/*`, `docs/reference/*`")
    lines.append("  - Reports and archive: `docs/reports/*`, `docs/archive/*`")
    lines.append("  - Manifest: `docs/docs-manifest.json`")
    # other
    lines.append("- **`tests/`**: Python test suites")
    lines.append("  - Suites: `tests/unit/`, `tests/smoke/`, `tests/ws/`, `tests/contracts/`, `tests/e2e-validation/`")
    lines.append("- **`integrations/`**: VS Code extension, Kubernetes, Docker, CI examples")
    lines.append("- **`infrastructure/`**: Monitoring, logging, disaster recovery assets")
    lines.append("- **`schemas/`** and `resources/`: JSON/YAML schemas and API contracts")
    lines.append("- **`scripts/`** and root `*.sh`: Setup/validate/run helpers (fast lanes)")
    lines.append("- **`docker*` + `Dockerfile*`**: Compose files and images")
    lines.append("- **`docs-site/`**: Static site sources")
    lines.append("- **`memory-bank/`**: Contextual product/tech notes")
    lines.append(FOOTER)
    NAV_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    write_nav()
    print(f"Updated {NAV_MD}")
