# Technical Specifications for Project Index Implementation

## Database Schema Specification

### Table: project_indexes
```sql
CREATE TABLE project_indexes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    root_path VARCHAR(500) NOT NULL,
    git_repository VARCHAR(500),
    git_branch VARCHAR(100) DEFAULT 'main',
    configuration JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    last_analyzed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_project_indexes_status ON project_indexes(status);
CREATE INDEX idx_project_indexes_last_analyzed ON project_indexes(last_analyzed_at);
```

### Table: file_entries
```sql
CREATE TABLE file_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_index_id UUID NOT NULL REFERENCES project_indexes(id) ON DELETE CASCADE,
    file_path VARCHAR(1000) NOT NULL,
    relative_path VARCHAR(1000) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size BIGINT NOT NULL,
    language VARCHAR(50),
    encoding VARCHAR(50) DEFAULT 'utf-8',
    analysis_data JSONB NOT NULL DEFAULT '{}',
    hash_sha256 VARCHAR(64) NOT NULL,
    last_modified TIMESTAMP WITH TIME ZONE,
    analyzed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_file_entries_project_path ON file_entries(project_index_id, relative_path);
CREATE INDEX idx_file_entries_language ON file_entries(language);
CREATE INDEX idx_file_entries_type ON file_entries(file_type);
CREATE INDEX idx_file_entries_hash ON file_entries(hash_sha256);
```

### Table: dependency_relationships
```sql
CREATE TABLE dependency_relationships (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_index_id UUID NOT NULL REFERENCES project_indexes(id) ON DELETE CASCADE,
    source_file_id UUID NOT NULL REFERENCES file_entries(id) ON DELETE CASCADE,
    target_file_id UUID REFERENCES file_entries(id) ON DELETE CASCADE,
    target_external VARCHAR(500), -- For external dependencies
    relationship_type VARCHAR(50) NOT NULL, -- 'import', 'require', 'include', etc.
    relationship_data JSONB DEFAULT '{}',
    line_number INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_dependency_source ON dependency_relationships(source_file_id);
CREATE INDEX idx_dependency_target ON dependency_relationships(target_file_id);
CREATE INDEX idx_dependency_type ON dependency_relationships(relationship_type);
CREATE INDEX idx_dependency_project ON dependency_relationships(project_index_id);
```

### Table: index_snapshots
```sql
CREATE TABLE index_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_index_id UUID NOT NULL REFERENCES project_indexes(id) ON DELETE CASCADE,
    snapshot_type VARCHAR(50) NOT NULL, -- 'manual', 'scheduled', 'pre_analysis'
    metadata JSONB NOT NULL DEFAULT '{}',
    file_count INTEGER NOT NULL,
    dependency_count INTEGER NOT NULL,
    analysis_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_snapshots_project ON index_snapshots(project_index_id);
CREATE INDEX idx_snapshots_type ON index_snapshots(snapshot_type);
CREATE INDEX idx_snapshots_created ON index_snapshots(created_at);
```

### Table: analysis_sessions
```sql
CREATE TABLE analysis_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_index_id UUID NOT NULL REFERENCES project_indexes(id) ON DELETE CASCADE,
    session_type VARCHAR(50) NOT NULL, -- 'full', 'incremental', 'context_optimization'
    status VARCHAR(50) NOT NULL DEFAULT 'running', -- 'running', 'completed', 'failed'
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    progress_percentage INTEGER DEFAULT 0,
    files_processed INTEGER DEFAULT 0,
    total_files INTEGER DEFAULT 0,
    error_message TEXT,
    session_data JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_analysis_sessions_project ON analysis_sessions(project_index_id);
CREATE INDEX idx_analysis_sessions_status ON analysis_sessions(status);
CREATE INDEX idx_analysis_sessions_started ON analysis_sessions(started_at);
```

## API Specification

### Endpoint Schemas

#### POST /api/project-index/create
```json
{
  "request": {
    "name": "string (required, max 255)",
    "description": "string (optional)",
    "root_path": "string (required, valid directory path)",
    "git_repository": "string (optional, valid git URL)",
    "git_branch": "string (optional, default 'main')",
    "configuration": {
      "languages": ["python", "javascript", "typescript"],
      "exclude_patterns": ["node_modules", "__pycache__", ".git"],
      "include_patterns": ["*.py", "*.js", "*.ts"],
      "analysis_depth": "integer (1-5, default 3)",
      "enable_ai_analysis": "boolean (default true)"
    }
  },
  "response": {
    "id": "uuid",
    "name": "string",
    "status": "string",
    "created_at": "iso8601",
    "analysis_session_id": "uuid"
  }
}
```

#### GET /api/project-index/{project_id}
```json
{
  "response": {
    "id": "uuid",
    "name": "string",
    "description": "string",
    "root_path": "string",
    "status": "string",
    "configuration": "object",
    "statistics": {
      "total_files": "integer",
      "analyzed_files": "integer",
      "total_dependencies": "integer",
      "languages_detected": ["string"],
      "last_analysis_duration": "float (seconds)"
    },
    "last_analyzed_at": "iso8601",
    "created_at": "iso8601",
    "updated_at": "iso8601"
  }
}
```

#### POST /api/project-index/{project_id}/context
```json
{
  "request": {
    "task_description": "string (required)",
    "files_mentioned": ["string (file paths)"],
    "context_type": "string (enum: 'refactoring', 'feature', 'debugging', 'analysis')",
    "max_files": "integer (default 10, max 50)",
    "include_dependencies": "boolean (default true)"
  },
  "response": {
    "optimized_context": {
      "relevant_files": [
        {
          "file_path": "string",
          "relevance_score": "float (0-1)",
          "analysis_summary": "string",
          "key_functions": ["string"],
          "dependencies": ["string"]
        }
      ],
      "dependency_graph": {
        "nodes": ["string"],
        "edges": [{"source": "string", "target": "string", "type": "string"}]
      },
      "recommendations": ["string"],
      "estimated_complexity": "string (low/medium/high)"
    },
    "context_id": "uuid",
    "generated_at": "iso8601"
  }
}
```

## WebSocket Event Specifications

### Event: project_index_updated
```json
{
  "type": "project_index_updated",
  "data": {
    "project_id": "uuid",
    "files_updated": "integer",
    "dependencies_updated": "integer",
    "analysis_duration": "float",
    "status": "string"
  },
  "timestamp": "iso8601",
  "correlation_id": "uuid"
}
```

### Event: analysis_progress
```json
{
  "type": "analysis_progress",
  "data": {
    "session_id": "uuid",
    "project_id": "uuid",
    "progress_percentage": "integer (0-100)",
    "files_processed": "integer",
    "total_files": "integer",
    "current_file": "string",
    "estimated_completion": "iso8601"
  },
  "timestamp": "iso8601",
  "correlation_id": "uuid"
}
```

### Event: dependency_changed
```json
{
  "type": "dependency_changed",
  "data": {
    "project_id": "uuid",
    "file_path": "string",
    "change_type": "string (added/removed/modified)",
    "dependency_target": "string",
    "relationship_type": "string"
  },
  "timestamp": "iso8601",
  "correlation_id": "uuid"
}
```

## Core Module Interface Specifications

### ProjectIndexer Class
```python
class ProjectIndexer:
    def __init__(self, project_id: UUID, config: ProjectIndexConfig):
        """Initialize project indexer with configuration."""
        pass
    
    async def analyze_project(self) -> AnalysisSession:
        """Perform full project analysis."""
        pass
    
    async def analyze_file(self, file_path: str) -> FileEntry:
        """Analyze individual file."""
        pass
    
    async def update_index(self, changed_files: List[str]) -> None:
        """Incrementally update index for changed files."""
        pass
    
    async def get_dependencies(self, file_path: str) -> List[DependencyRelationship]:
        """Get dependencies for specific file."""
        pass
    
    async def optimize_context(self, request: ContextRequest) -> ContextResponse:
        """Generate optimized context for AI agents."""
        pass
```

### CodeAnalyzer Class
```python
class CodeAnalyzer:
    def __init__(self, language: str):
        """Initialize analyzer for specific programming language."""
        pass
    
    def parse_file(self, file_path: str, content: str) -> ParseResult:
        """Parse file and extract structure information."""
        pass
    
    def extract_imports(self, ast_tree) -> List[ImportStatement]:
        """Extract import/require statements."""
        pass
    
    def extract_functions(self, ast_tree) -> List[FunctionDefinition]:
        """Extract function definitions."""
        pass
    
    def extract_classes(self, ast_tree) -> List[ClassDefinition]:
        """Extract class definitions."""
        pass
    
    def calculate_complexity(self, ast_tree) -> ComplexityMetrics:
        """Calculate code complexity metrics."""
        pass
```

## Configuration Specifications

### Project Index Configuration Schema
```json
{
  "languages": {
    "type": "array",
    "items": {"type": "string"},
    "default": ["python", "javascript", "typescript"],
    "description": "Programming languages to analyze"
  },
  "exclude_patterns": {
    "type": "array", 
    "items": {"type": "string"},
    "default": ["node_modules", "__pycache__", ".git", "*.pyc"],
    "description": "File/directory patterns to exclude"
  },
  "include_patterns": {
    "type": "array",
    "items": {"type": "string"}, 
    "default": ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx"],
    "description": "File patterns to include"
  },
  "analysis_depth": {
    "type": "integer",
    "minimum": 1,
    "maximum": 5,
    "default": 3,
    "description": "Depth of dependency analysis"
  },
  "max_file_size": {
    "type": "integer",
    "default": 1048576,
    "description": "Maximum file size to analyze (bytes)"
  },
  "enable_ai_analysis": {
    "type": "boolean", 
    "default": true,
    "description": "Enable AI-powered code analysis"
  },
  "context_optimization": {
    "type": "object",
    "properties": {
      "max_context_files": {"type": "integer", "default": 20},
      "relevance_threshold": {"type": "number", "default": 0.3},
      "include_test_files": {"type": "boolean", "default": false}
    }
  }
}
```

## Performance Requirements

### Indexing Performance
- **Small Projects** (< 100 files): Complete analysis < 30 seconds
- **Medium Projects** (100-1000 files): Complete analysis < 5 minutes  
- **Large Projects** (1000-10000 files): Complete analysis < 30 minutes
- **Incremental Updates**: Single file analysis < 2 seconds

### API Performance
- **Index Retrieval**: < 200ms for basic project info
- **Context Optimization**: < 500ms for up to 20 files
- **Dependency Queries**: < 100ms for single file dependencies
- **Real-time Updates**: WebSocket event delivery < 50ms

### Memory Usage
- **Base Memory**: < 50MB for indexer module
- **Per Project**: < 10MB for projects up to 1000 files
- **Analysis Session**: < 100MB peak during large project analysis
- **Cache Size**: Configurable, default 256MB

### Database Performance
- **Write Operations**: Batch inserts for analysis results
- **Read Operations**: Optimized queries with proper indexing
- **Connection Pooling**: Use existing bee-hive database pool
- **Migration Time**: < 1 minute for schema updates