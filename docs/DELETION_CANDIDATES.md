# Deletion Candidates (for approval)

Only remove after confirming no references in code/tests and that `docs/NAV_INDEX.md` covers the remaining navigation.

## Code (legacy server-rendered dashboard)
- `app/dashboard/coordination_dashboard.py` — Jinja templates, not used by app bindings
- `app/dashboard/simple_agent_dashboard.py` — Jinja templates, not used by app bindings
- `app/dashboard/templates/dashboard.html` — legacy HTML

Rationale: PWA-first policy; tests enforce no legacy `/dashboard` routes. Keep enterprise HTML gated under `/enterprise` (optional), or replace later with PWA.

## Docs (duplicates/outdated)
- `docs/DEVELOPER_GUIDE.md` — archived banner added; keep or delete after a sprint
- `docs/archive/QUICK_START*.md` — duplicates; keep archived only
- `docs/archive/GETTING_STARTED.md` — legacy; keep archived
- Any guide duplicating `docs/GETTING_STARTED.md` quick-start steps beyond a minimal link

## Link fixes applied
- Updated links in `mobile-pwa/README.md` and `integrations/README.md`
- Replaced `VITE_WS_URL` example to `/api/dashboard/ws/dashboard`

Approve deletions? I can remove files and update the nav accordingly.
