# Active Context

- Focus: Decommission server-rendered dashboard; PWA rebrand to HiveOps; consolidate markdown docs
- Decisions:
  - Legacy backend dashboard removed; compatibility API kept
  - PWA branding updated; WS endpoint fixed via adapter
  - Core docs consolidated: `docs/CORE.md`, `docs/ARCHITECTURE.md`, `docs/PRD.md`
  - All duplicate root `.md` moved to `docs/archive/`; index/manifest updated
- Next:
  - Optional: auto-fix formatting in archived files or compress to stubs
  - Run Playwright smoke and update titles/selectors as needed
