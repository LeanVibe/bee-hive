# Product Context

## Why it exists

Senior builders need a lightweight, modern way to observe and guide autonomous dev agents without heavyweight dashboards or external key setup.

## Experience principles

- Mobile-first, fast, minimal friction.
- Clear data hierarchy: health, activity, tasks, events.
- Professional aesthetic suitable for SV-grade tooling.

## How it works (high-level)

- PWA connects to `/api/dashboard/ws/dashboard` for live updates and `/dashboard/api/live-data` for initial state.
