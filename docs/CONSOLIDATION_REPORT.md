# Documentation Consolidation Report (First Pass)

This report inventories overlapping docs and proposes a conservative consolidation plan.

## Canonical entry points to keep
- README.md: top-level overview + quick links
- docs/GETTING_STARTED.md: one canonical getting started (merge any duplicates into this)
- docs/CORE.md: concise architecture/core overview
- docs/ARCHITECTURE.md: implementation-aligned architecture details
- docs/DOCS_INDEX.md: navigational index to sections

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
- No server-rendered dashboard routes should remain (policy).
- Ensure references focus on API/WebSocket and mobile PWA.

## Action items (docs)
- Merge README quick start with canonical docs/GETTING_STARTED.md and remove redundant quick start sections from docs that only duplicate the same steps.
- Add cross-links from docs/DEVELOPER_GUIDE.md to canonical sections instead of duplicating content.

## Next steps
- PR 1: Link hygiene + minimal deduplication edits.
- PR 2: Larger merges (archive superseded files; update tables of contents).
