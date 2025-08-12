# Documentation Consolidation Report (First Pass)

This report inventories overlapping docs and proposes a conservative consolidation plan.

## Canonical entry points to keep
- README.md: top-level overview + quick links
- docs/GETTING_STARTED.md: one canonical getting started (merge any duplicates into this)
- docs/CORE.md: concise architecture/core overview
- docs/ARCHITECTURE.md: implementation-aligned architecture details
- docs/NAV_INDEX.md: generated repository navigation index (authoritative navigation)
- docs/DOCS_INDEX.md: human landing that links to `docs/NAV_INDEX.md`

## Notable overlaps (Getting Started / Quick Start)
- docs/DEVELOPER_GUIDE.md (Getting Started section)
- docs/archive/GETTING_STARTED.md (legacy, very long)
- docs/archive/QUICK_START.md and docs/archive/QUICK_START_README.md (archived quick starts)
- multiple Quick Start sections across guides/tutorials

Proposal:
- Consolidate “Getting Started” content into docs/GETTING_STARTED.md (short, tested).
- Keep only a short “Quick start” section in README that links to the canonical guide.
- Remove exact duplicates; keep archived versions under docs/archive/ for provenance.

## Deprecated/legacy dashboard notes
- No server-rendered dashboard under `/dashboard` (policy enforced; tests exist).
- Enterprise HTML templates under `/enterprise` are non-core; gate behind feature flag and document as optional. Prefer PWA equivalents.

## Action items (docs)
- Merge README quick start with canonical docs/GETTING_STARTED.md and remove redundant quick start sections from docs that only duplicate the same steps.
- Add cross-links from docs/DEVELOPER_GUIDE.md to canonical sections instead of duplicating content.

## Next steps
- PR 1: Link hygiene + minimal deduplication edits (done: generated nav, manifest merge).
- PR 2: Larger merges (archive superseded files; update tables of contents).
- PR 3: Gate enterprise HTML templates with env flag; add PWA route/docs alternative.
