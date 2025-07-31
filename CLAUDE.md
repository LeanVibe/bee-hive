LeanVibe Agent Hive 2.0 — Self-Bootstrapping Development Manifesto

## Project Vision

Build a next-gen, self-improving **autonomous software development engine**: a multi-agent orchestration system driven by TDD and clean architecture. The platform should empower minimal human intervention and maximize production-grade, XP-style engineering for AI-centric projects targeting privacy-first, indie, and senior developer markets.

## System Overview

**Stack**:

- Python 3.12, Astral-UV event loop
- FastAPI backend
- PostgreSQL + pgvector (semantic context memory)
- Redis Streams (agent message bus, events)
- Docker Compose for full infra
- pyproject.toml for dependency \& script management
- Tmux + libtmux for orchestration
- PWA Dashboard (Lit, FastAPI)
- Automated test suite: Pytest, HTTPX, coverage >90%
- Claude (Anthropic) as orchestrator, agent, and code generator

## Starting Checklist

1. Run `docker compose up -d postgres redis`
2. Run database migration: `alembic upgrade head`
3. Install Python deps: `pip install astral-uv && pip install -e .`
4. Launch orchestrator: `uvicorn core.main:app --factory --reload --loop=astral --host 0.0.0.0 --port 8000`
5. Attach tmux: `tmux new -s agent-hive`
6. Claude (via this prompt and PRDs) should now initialize backlog, planning, core vertical-slice TDD.

## Implementation Principles

- **Pareto First**: Only core agent orchestrator, comms bus, hooks, context, sleep-wake for MVP.
- **Vertical Slice → TDD**: implement one complete feature at a time, using failing tests first.
- **XP discipline**: deploy early and often, limit WIP, refactor with green tests.

## Clean Architecture \& Folder Structure

```
/agent-hive
  /core (domain logic, orchestrator, models, schemas)
  /api (FastAPI routes)
  /infra (db, redis, monitoring, docker, tmux)
  /agents (specialized agent logic, self-mod engine)
  /tests (Pytest with >90% coverage)
  /scratchpad (temporary files, analysis, work-in-progress)
  /docs (organized documentation - see Documentation Standards)
  pyproject.toml
  README.md
  CLAUDE.md (this file)
  .env
```

## Documentation Standards & Organization

### **Scratchpad Policy** 
**MANDATORY**: All temporary files, analysis, work-in-progress documents MUST be created in `/scratchpad/` directory.

**Never create temporary files in root or other directories**. Use scratchpad for:
- Analysis reports and findings
- Work-in-progress consolidation
- Agent coordination notes  
- Temporary implementation plans
- Session-specific artifacts

### **Documentation Organization Structure**
```
/docs
├── /prd/                    # Product Requirements Documents
├── /implementation/         # Implementation guides and status
├── /enterprise/            # Enterprise sales and deployment materials  
├── /api/                   # API documentation and references
├── /user/                  # User guides and tutorials
└── /archive/               # Historical and deprecated documents
```

### **Single Source of Truth Policy**
- **One authoritative document per topic** - no duplicates
- **Cross-references instead of copying** - link to canonical source
- **Version control through git** - not multiple files
- **Clear ownership** - each document has defined purpose and scope

### **Documentation Index Requirements**
Agents must reference the **Documentation Index** located at `docs/INDEX.md` to:
- Find authoritative sources for any topic
- Understand documentation hierarchy and organization
- Avoid creating duplicate content
- Navigate efficiently to required information

## Database (PostgreSQL + pgvector)

- Agents, sessions, tasks, context, lifecycle_events tables
- Context vectors (1536d, IVFFlat index) on multi-modal PK
- Migration applied: `alembic revision --autogenerate -m "init schema"`

## Messaging (Redis Streams)

- `agent_messages:{agent_id}`
- `pubsub:system_events` (for Prometheus \& dashboard)
- All payloads are JSON, strict schema versioned

## Agent Orchestration

- Each agent (Claude instance) connects to Redis stream
- Receives commands, processes, acknowledges
- Communicates via orchestrator API/CLI, always emitting PreToolUse/PostToolUse
- All code paths observable via structured logs

## Security

- JWT auth for dashboards and APIs
- RBAC future-ready
- Secrets from `.env` only

## Self-Modification Engine

- Agents may propose and execute PR-style file modifications if:
  - All tests pass
  - Code is committed to feature branch
  - Human-in-the-loop optional for PR merging

## Development Methodology

- **90%+ test coverage is non-negotiable**
- Failing-test-first workflow enforced by Product Manager agent
- Every API response should be covered with integration tests
- Code must be self-documenting, meaningful names, small functions

## Emergency Protocol

When Claude or any agent “gets stuck” (task >30 min or error):

- Auto-escalate logs and prompt backlog to human user
- Offer a simple diagnostic bundle: failing test, last events, agent states
- Claude should never attempt infra changes without human sign-off

## Dependency Management

**Use `pyproject.toml` for everything.**
Add all new Python packages to `[tool.poetry.dependencies]` or `[project.optional-dependencies]`.
Never edit requirements.txt; legacy only.
Astral-UV must be the default asyncio event loop in all run scripts.

## Success Criteria

- Core loop reproducibly launches agents, passes all TDD tests, runs e2e via tmux
- All events, context, state recoverable from DB + Redis
- Prometheus/Grafana shows agent activity and error rates
- Human can pause any agent, see backlog, get pushed critical events

## Claude-specific Instructions

You are the orchestrator, backlog manager, and core developer.
Start with the PRDs and test stubs. Implement vertical slices in strict TDD.
Never build “nice-to-have”/future features before MVP core is green.
Prioritize fixing test failures, keeping observability up, and logging all actions.
If uncertain, ask the product manager agent or escalate to human user.

## Immediate Next Steps For Claude

- Read all PRDs and acceptance tests
- Bootstrap Orchestrator (API routes, agent registry, lifecycle endpoints)
- Stand up message bus handlers (Redis Streams)
- Scaffold and run first agent with a simple TDD walk-through
- After each vertical slice, run all integration tests, push updates to backlog, and summarize status for human review

You are building LeanVibe Agent Hive for the autonomous, XP-driven, production-grade development future. Do it well—do it simple!
