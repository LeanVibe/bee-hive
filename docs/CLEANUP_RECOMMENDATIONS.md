# Markdown Cleanup Recommendations

- Merge root-level topical docs into `docs/` under clear categories (guides, runbooks, reference).
- Keep only one PRD: `docs/PRD.md`; mark older PRDs as archived (done for core).
- Prefer `docs-site/` for public-facing guides; use `docs/` for engineering source-of-truth.
- Tag scratchpads and reports under `scratchpad/` or `docs/archive/`; avoid mixing with core.
- Update links to point to `docs/CORE.md`, `docs/ARCHITECTURE.md`, `docs/PRD.md`.
- Enforce markdown lint spacing on edited files; allow legacy archives to retain content except banner.
