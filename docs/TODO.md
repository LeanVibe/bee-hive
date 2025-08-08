# TODOs (Now → Next → Later)

## Now (must-have)

- [ ] Run Playwright smoke (mobile/desktop/dark) and update selectors/titles for "HiveOps"
- [ ] Verify REST/WS endpoints: `/health`, `/api/dashboard/ws/dashboard`, `/dashboard/api/live-data`
- [ ] Fix any failing tests and commit with descriptive messages
- [ ] Polish PWA header and typography; add initial design tokens
- [ ] Update `docs/DOCS_INDEX.md` and `docs/docs-manifest.json` if new docs are added
- [ ] Stabilize backend for e2e (from mobile dashboard validation): ensure server runs during tests; align component selectors; add test-mode SW config

## Next (high value)

- [ ] Dark mode refinement and visual polish across primary views
- [ ] Expand Playwright coverage for nav, live data render, error states
- [ ] Generate marketing-ready screenshots for README/docs-site
- [ ] Compress archived docs into short redirects (optional)
- [ ] Wire docs-site to surface `docs/CORE.md`, `docs/ARCHITECTURE.md`, `docs/PRD.md`
- [ ] Implement agent lifecycle service and continuous coordination scheduler; refactor PWA services to enhanced APIs

## Later (roadmap)

- [ ] Autonomous APIs: project mgmt, code intelligence, deploy pipeline, learning/adaptation
- [ ] API rate limiting/versioning/auth hardening; service-to-service auth
- [ ] DB: pooling optimization, query profiling, archiving/retention; cache strategy
- [ ] MQ: DLQ with backoff, queue monitoring, consumer lag metrics
- [ ] Scalability: distributed orchestration patterns; Redis Cluster; read replicas
- [ ] Context engine: compression (60–80%), temporal windows, cross-agent knowledge, <50ms retrieval
- [ ] Integrations: VS Code extension, CI templates, Helm/Terraform optional IaC
- [ ] Observability: tracing and intelligent alerting hooks; optional enterprise dashboards
- [ ] DR/backup automation scripts and runbooks
