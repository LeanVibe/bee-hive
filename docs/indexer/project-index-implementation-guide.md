# Project Index Implementation Guide for Claude

## Overview
This document provides comprehensive guidance for implementing the Project Index feature for the bee-hive multi-agent orchestration system. The Project Index will provide intelligent code structure analysis and context optimization for AI agents.

## Architecture Decision: Embedded Module Approach

**Decision**: Implement as an embedded module within bee-hive rather than a separate project.

**Rationale**:
- Leverages existing FastAPI backend infrastructure
- Utilizes existing WebSocket real-time capabilities 
- Integrates with existing multi-agent orchestration
- Maintains single deployment and maintenance surface
- Enables tight integration with bee-hive's agent system

## Implementation Strategy

### Phase 1: Core Infrastructure
1. **Project Index Core Module** (`app/project_index/`)
2. **WebSocket Integration** (extend existing dashboard WebSocket)
3. **API Endpoints** (RESTful API for index management)
4. **Database Schema** (extend existing PostgreSQL)
5. **Background Tasks** (using existing task queue)

### Phase 2: AI Integration
1. **Agent Integration** (connect with existing agent system)
2. **Context Analysis** (AI-powered code understanding)
3. **Smart Recommendations** (leveraging existing AI capabilities)

### Phase 3: Dashboard Integration
1. **PWA Frontend** (extend existing Lit + Vite PWA)
2. **Real-time Updates** (via existing WebSocket infrastructure)
3. **Visualization** (project structure visualization)

## File Structure Integration

```
bee-hive/
├── app/
│   ├── project_index/           # NEW: Core Project Index module
│   │   ├── __init__.py
│   │   ├── core.py             # Main indexing engine
│   │   ├── models.py           # Pydantic models
│   │   ├── analyzer.py         # Code analysis logic
│   │   ├── agent.py            # AI agent integration
│   │   └── schemas.py          # Database schemas
│   ├── api/
│   │   └── routes/
│   │       └── project_index.py # NEW: API endpoints
│   ├── dashboard/
│   │   ├── websocket.py        # EXTEND: Add index events
│   │   └── models.py           # EXTEND: Add index models
│   └── database/
│       ├── models.py           # EXTEND: Add index tables
│       └── migrations/         # NEW: Index table migrations
├── mobile-pwa/
│   └── src/
│       ├── components/
│       │   └── project-index/  # NEW: Index UI components
│       └── pages/
│           └── project-index/  # NEW: Index dashboard pages
└── docs/
    ├── PROJECT_INDEX.md        # NEW: Feature documentation
    └── PROJECT_INDEX_API.md    # NEW: API documentation
```

## Database Schema Extensions

The Project Index will extend the existing PostgreSQL database with new tables:

- `project_indexes` - Main index metadata
- `file_entries` - Individual file analysis results  
- `dependency_relationships` - Code dependency mapping
- `index_snapshots` - Historical index states
- `analysis_sessions` - AI analysis tracking

## Integration Points

### 1. WebSocket Integration
Extend existing `app/dashboard/websocket.py` to include:
- `project_index_updated` events
- `analysis_progress` events  
- `dependency_changes` events

### 2. Agent System Integration
Connect with existing agent orchestration to:
- Provide context-aware code analysis
- Enable intelligent agent task routing
- Support multi-agent collaboration with shared project understanding

### 3. API Integration
Extend existing FastAPI routes to include:
- Index management endpoints
- Real-time analysis triggers
- Project health metrics

## Implementation Requirements

### Technical Prerequisites
- Python 3.8+ (already satisfied by bee-hive)
- PostgreSQL database (already available)
- Redis for caching (already available)
- FastAPI framework (already in use)
- WebSocket support (already implemented)

### New Dependencies
- `tree-sitter` - For advanced code parsing
- `networkx` - For dependency graph analysis
- `gitpython` - For Git integration
- `watchdog` - For file system monitoring

## Success Metrics

### Technical Metrics
- Index update latency < 500ms for typical files
- Memory usage < 100MB for projects up to 10k files
- WebSocket event delivery success rate > 99.9%

### User Experience Metrics  
- Context retrieval time < 200ms
- Agent task accuracy improvement > 30%
- Project complexity visualization load time < 2s

## Next Steps

1. **Review Implementation Plan** with development team
2. **Set up Development Environment** with required dependencies
3. **Create Database Migrations** for new schema
4. **Implement Core Module** following this specification
5. **Add API Endpoints** with comprehensive testing
6. **Integrate WebSocket Events** for real-time updates
7. **Build PWA Frontend** components
8. **Conduct Integration Testing** with existing bee-hive features

## Risk Mitigation

### Performance Risks
- Implement incremental indexing to avoid blocking operations
- Use background task queues for heavy analysis
- Add caching layers for frequently accessed data

### Integration Risks  
- Maintain backward compatibility with existing APIs
- Implement feature flags for gradual rollout
- Create comprehensive test coverage for integration points

### Data Risks
- Implement robust error handling for malformed code
- Add data validation at all input points
- Create backup and recovery procedures for index data