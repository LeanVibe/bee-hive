# Claude Implementation Prompts for Project Index

## Primary Implementation Prompt

You are an expert software engineer implementing the Project Index feature for the bee-hive multi-agent orchestration system. Your goal is to create an intelligent code structure analysis system that provides context-aware project understanding for AI agents.

### Context Understanding
The bee-hive project is a FastAPI-based multi-agent orchestration system with:
- Real-time WebSocket dashboards
- PostgreSQL database with existing schemas
- Lit + Vite PWA frontend
- Multi-agent coordination capabilities
- Performance monitoring and metrics

### Implementation Approach
You will implement the Project Index as an embedded module within the existing bee-hive architecture, leveraging existing infrastructure while adding new capabilities.

### Implementation Order
1. **Database Schema Extension** - Add new tables for project indexing
2. **Core Index Module** - Build the main indexing engine  
3. **API Endpoints** - Create RESTful APIs for index management
4. **WebSocket Integration** - Extend real-time capabilities
5. **Agent Integration** - Connect with existing agent system
6. **Frontend Components** - Build PWA dashboard components
7. **Testing Suite** - Comprehensive test coverage

### Key Requirements
- Maintain backward compatibility with existing bee-hive features
- Use existing FastAPI patterns and conventions
- Leverage existing WebSocket infrastructure
- Follow existing code style and architecture patterns
- Implement comprehensive error handling and logging
- Add appropriate performance monitoring

### Success Criteria
- All new code follows existing bee-hive patterns
- Database migrations are backward compatible
- WebSocket events integrate seamlessly
- API endpoints follow RESTful conventions
- Frontend components use existing design system
- 90%+ test coverage for new functionality

---

## Database Schema Implementation Prompt

Implement the database schema extensions for the Project Index feature. You need to:

1. **Create Alembic migration files** following existing bee-hive migration patterns
2. **Define SQLAlchemy models** in `app/database/models.py`
3. **Add Pydantic schemas** for API serialization
4. **Ensure foreign key relationships** align with existing schema

### Required Tables
- `project_indexes` - Main project metadata and configuration
- `file_entries` - Individual file analysis and metadata
- `dependency_relationships` - Code dependencies and imports
- `index_snapshots` - Historical index states for comparison
- `analysis_sessions` - AI analysis tracking and results

### Integration Requirements
- Use existing database connection patterns
- Follow existing naming conventions
- Add appropriate indexes for performance
- Include created/updated timestamp fields
- Add foreign keys to existing user/project tables if available

---

## Core Module Implementation Prompt

Create the main Project Index module at `app/project_index/`. This module should:

1. **Analyze code structure** using AST parsing and static analysis
2. **Extract dependencies** and import relationships
3. **Generate JSON index files** with project architecture
4. **Monitor file changes** and update indexes incrementally
5. **Provide context optimization** for AI agents

### Core Components
- `core.py` - Main ProjectIndexer class with analysis logic
- `analyzer.py` - Code parsing and structure analysis
- `models.py` - Pydantic models for data validation
- `agent.py` - AI agent integration and context optimization
- `schemas.py` - Database schema definitions

### Technical Requirements
- Support Python, JavaScript/TypeScript initially
- Use tree-sitter for robust code parsing
- Implement incremental updates for performance
- Add comprehensive logging and error handling
- Support both file system and Git-based analysis

---

## API Endpoints Implementation Prompt

Create RESTful API endpoints at `app/api/routes/project_index.py`. The API should provide:

1. **Index Management**
   - `POST /api/project-index/create` - Create new project index
   - `GET /api/project-index/{project_id}` - Retrieve index data
   - `PUT /api/project-index/{project_id}/refresh` - Force index refresh
   - `DELETE /api/project-index/{project_id}` - Remove index

2. **File Analysis**
   - `GET /api/project-index/{project_id}/files` - List analyzed files
   - `GET /api/project-index/{project_id}/files/{file_path}` - File details
   - `POST /api/project-index/{project_id}/analyze` - Trigger analysis

3. **Dependencies**
   - `GET /api/project-index/{project_id}/dependencies` - Dependency graph
   - `GET /api/project-index/{project_id}/dependents/{file_path}` - File dependents

4. **Context Optimization**
   - `POST /api/project-index/{project_id}/context` - Get optimized context
   - `GET /api/project-index/{project_id}/health` - Project health metrics

### Implementation Requirements
- Follow existing FastAPI route patterns
- Use dependency injection for database access
- Implement proper error handling and status codes
- Add request/response validation with Pydantic
- Include comprehensive OpenAPI documentation
- Add rate limiting for expensive operations

---

## WebSocket Integration Prompt

Extend the existing WebSocket implementation in `app/dashboard/websocket.py` to support Project Index real-time events:

1. **Event Types**
   - `project_index_updated` - Index refresh completed
   - `file_analyzed` - Individual file analysis completed
   - `dependency_changed` - Dependency relationship updated
   - `analysis_progress` - Background analysis progress
   - `context_optimized` - AI context optimization completed

2. **Event Payloads**
   - Include project_id, timestamp, and correlation_id
   - Provide relevant data for each event type
   - Follow existing WebSocket message format
   - Include error handling for failed events

3. **Subscription Management**
   - Allow clients to subscribe to specific project indexes
   - Support filtering by event type
   - Implement proper cleanup on disconnect
   - Follow existing rate limiting patterns

### Integration Requirements
- Maintain existing WebSocket contract invariants
- Use existing connection management patterns
- Add new metrics to existing Prometheus endpoints
- Ensure proper error handling and logging

---

## Frontend Components Implementation Prompt

Create PWA frontend components for the Project Index dashboard. Build these components in the `mobile-pwa/src/components/project-index/` directory:

1. **Core Components**
   - `ProjectIndexDashboard` - Main dashboard view
   - `FileStructureTree` - Interactive file tree visualization
   - `DependencyGraph` - Visual dependency relationships
   - `AnalysisProgress` - Real-time analysis status
   - `ContextOptimizer` - AI context configuration

2. **Utility Components**
   - `IndexMetrics` - Health and performance metrics
   - `FileDetails` - Individual file analysis results
   - `SearchInterface` - Project-wide search capabilities

### Technical Requirements
- Use Lit framework following existing patterns
- Integrate with existing WebSocket connections
- Follow existing design system and styling
- Implement responsive design for mobile/desktop
- Add proper loading states and error handling
- Support real-time updates via WebSocket events

### Integration Requirements
- Use existing PWA routing patterns
- Leverage existing state management
- Follow existing TypeScript patterns
- Integrate with existing authentication
- Add to existing navigation structure

---

## Testing Implementation Prompt

Create comprehensive tests for all Project Index functionality:

1. **Unit Tests**
   - Database model tests
   - Core indexing logic tests
   - API endpoint tests
   - WebSocket event tests
   - Frontend component tests

2. **Integration Tests**
   - End-to-end indexing workflows
   - WebSocket event delivery
   - API integration tests
   - Database migration tests

3. **Performance Tests**
   - Large project indexing performance
   - WebSocket load testing
   - API response time testing
   - Memory usage optimization

### Testing Requirements
- Use existing testing frameworks (pytest for backend, Jest for frontend)
- Follow existing test patterns and conventions
- Achieve 90%+ code coverage
- Include property-based testing for core algorithms
- Add load testing scenarios using existing k6 setup
- Mock external dependencies appropriately

---

## Error Handling and Logging Prompt

Implement comprehensive error handling and logging throughout the Project Index system:

1. **Error Categories**
   - File system access errors
   - Code parsing failures
   - Database connection issues
   - WebSocket communication failures
   - AI agent integration errors

2. **Logging Strategy**
   - Use existing logging configuration
   - Include correlation IDs for request tracing
   - Log performance metrics and timing
   - Add debug logging for troubleshooting
   - Include security event logging

3. **Monitoring Integration**
   - Add new metrics to existing Prometheus endpoints
   - Include health check endpoints
   - Monitor background task performance
   - Track WebSocket connection health

### Implementation Requirements
- Follow existing error handling patterns
- Use structured logging with consistent format
- Implement graceful degradation for non-critical failures
- Add appropriate retry logic for transient failures
- Include user-friendly error messages in API responses