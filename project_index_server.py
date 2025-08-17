#!/usr/bin/env python3
"""
Standalone Project Index API Server

Simple FastAPI server to test Project Index functionality
without the complexity of the full bee-hive application.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4

import asyncpg
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import sys
from pathlib import Path

# Add the app directory to Python path so we can import the project index modules
sys.path.append(str(Path(__file__).parent))

try:
    from app.project_index import ProjectIndexer, AnalysisConfiguration, ProjectIndexConfig
    from app.models.project_index import AnalysisSessionType
    from app.project_index.language_parsers import LanguageParserFactory, Dependency
    from app.project_index.agent_delegation import (
        TaskDecomposer, AgentCoordinator, ContextRotPrevention,
        TaskType, TaskComplexity, AgentSpecialization, AgentTask
    )
    FULL_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Full Project Index integration not available: {e}")
    FULL_INTEGRATION_AVAILABLE = False


# Configuration
DATABASE_URL = "postgresql://leanvibe_user:leanvibe_secure_pass@localhost:5432/leanvibe_agent_hive"

# Database connection pool
pool = None

async def get_db_pool():
    global pool
    if not pool:
        pool = await asyncpg.create_pool(DATABASE_URL)
    return pool

# Pydantic models
class ProjectIndexCreate(BaseModel):
    name: str
    description: Optional[str] = None
    root_path: str
    git_repository_url: Optional[str] = None
    git_branch: Optional[str] = None

class ProjectIndexResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    root_path: str
    git_repository_url: Optional[str]
    git_branch: Optional[str]
    status: str
    file_count: int
    dependency_count: int
    created_at: datetime

class HealthResponse(BaseModel):
    status: str
    database: str
    project_index_tables: bool

class AnalysisRequest(BaseModel):
    force_reanalysis: bool = False
    analysis_type: str = "full_analysis"  # full_analysis, incremental, dependency_mapping, context_optimization

class AnalysisResponse(BaseModel):
    session_id: str
    status: str
    message: str
    started_at: datetime

class AnalysisStatusResponse(BaseModel):
    session_id: str
    status: str
    progress_percentage: float
    files_processed: int
    files_total: int
    errors_count: int
    current_phase: Optional[str]
    estimated_completion: Optional[datetime]

class ContextAssemblyRequest(BaseModel):
    task_description: str
    max_files: int = 10
    include_dependencies: bool = True
    focus_languages: Optional[List[str]] = None

class ContextFileResult(BaseModel):
    id: str
    relative_path: str
    file_name: str
    language: Optional[str]
    relevance_score: float
    content_preview: Optional[str]
    reason: str  # Why this file was included
    related_files: List[str] = []  # Related files via dependencies

class TaskDecompositionRequest(BaseModel):
    task_description: str
    task_type: str = "feature-implementation"  # Maps to TaskType enum
    
class SubtaskInfo(BaseModel):
    id: str
    title: str
    description: str
    complexity: str
    estimated_duration_minutes: int
    preferred_specialization: str
    primary_files: List[str]
    dependencies: List[str] = []
    
class TaskDecompositionResponse(BaseModel):
    original_task: SubtaskInfo
    subtasks: List[SubtaskInfo]
    coordination_plan: Dict[str, Any]
    estimated_total_duration: int
    decomposition_strategy: str
    success: bool
    reason: str

class AgentAssignmentRequest(BaseModel):
    decomposition_result_id: str  # Reference to stored decomposition
    
class AgentAssignmentInfo(BaseModel):
    task_id: str
    agent_id: str
    specialization: str
    estimated_start_time: datetime
    context_requirements: Dict[str, Any]
    
class AgentCoordinationResponse(BaseModel):
    assignments: List[AgentAssignmentInfo]
    coordination_plan: Dict[str, Any]
    total_agents: int
    estimated_completion: datetime

class DependencyResponse(BaseModel):
    target_name: str
    dependency_type: str
    is_external: bool
    line_number: Optional[int]
    source_text: Optional[str]
    confidence_score: float
    metadata: dict

class FileDetailResponse(BaseModel):
    id: str
    relative_path: str
    file_name: str
    file_type: str
    language: Optional[str]
    file_size: int
    line_count: Optional[int]
    content_preview: Optional[str]
    last_modified: Optional[datetime]
    dependencies: List[DependencyResponse]

async def run_project_analysis_background(
    project_id: str,
    session_id: str,
    analysis_type: str,
    force_reanalysis: bool,
    root_path: str
):
    """Background task to run project analysis"""
    pool = await get_db_pool()
    
    try:
        # Simplified analysis for standalone mode
        # This does basic file scanning without the full ProjectIndexer complexity
        
        async with pool.acquire() as conn:
            # Update session to running
            await conn.execute("""
                UPDATE analysis_sessions 
                SET status = 'running', current_phase = 'scanning_files'
                WHERE id = $1
            """, UUID(session_id))
            
            # Scan files in the project directory
            project_path = Path(root_path)
            if not project_path.exists():
                await conn.execute("""
                    UPDATE analysis_sessions 
                    SET status = 'failed', error_log = $2, completed_at = $3
                    WHERE id = $1
                """, UUID(session_id), [{"error": "Project path not found"}], datetime.utcnow())
                return
            
            # Simple file scanning and counting
            supported_extensions = {'.py', '.js', '.ts', '.json', '.yaml', '.yml', '.sql', '.md', '.txt'}
            
            files_found = []
            total_files = 0
            processed_files = 0
            
            # Count files first
            for file_path in project_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    # Skip common ignore patterns
                    if any(part.startswith('.') for part in file_path.parts):
                        continue
                    if any(part in ['node_modules', '__pycache__', 'venv', '.git'] for part in file_path.parts):
                        continue
                    files_found.append(file_path)
                    total_files += 1
            
            # Update total files count
            await conn.execute("""
                UPDATE analysis_sessions 
                SET files_total = $2, current_phase = 'analyzing_files'
                WHERE id = $1
            """, UUID(session_id), total_files)
            
            # Process files and store them in database
            dependencies_found = 0
            
            for i, file_path in enumerate(files_found):
                try:
                    # Get file info
                    file_stat = file_path.stat()
                    relative_path = str(file_path.relative_to(project_path))
                    
                    # Determine file type and language
                    extension = file_path.suffix.lower()
                    if extension in ['.py']:
                        file_type = 'source'
                        language = 'python'
                    elif extension in ['.js', '.ts']:
                        file_type = 'source' 
                        language = 'javascript' if extension == '.js' else 'typescript'
                    elif extension in ['.json', '.yaml', '.yml']:
                        file_type = 'config'
                        language = 'json' if extension == '.json' else 'yaml'
                    elif extension in ['.sql']:
                        file_type = 'source'
                        language = 'sql'
                    elif extension in ['.md', '.txt']:
                        file_type = 'documentation'
                        language = 'markdown' if extension == '.md' else 'text'
                    else:
                        file_type = 'other'
                        language = None
                    
                    # Read file content for analysis (limit to reasonable size)
                    content = ""
                    content_preview = ""
                    line_count = 0
                    
                    try:
                        if file_stat.st_size < 1024 * 1024:  # Only read files smaller than 1MB
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                line_count = len(content.splitlines())
                                # Create preview (first 500 characters)
                                content_preview = content[:500] + ("..." if len(content) > 500 else "")
                    except Exception as read_error:
                        print(f"Could not read {file_path}: {read_error}")
                        content = ""
                    
                    # Extract dependencies if content is available and parser exists
                    file_dependencies = []
                    if content and language and FULL_INTEGRATION_AVAILABLE:
                        try:
                            parser = LanguageParserFactory.get_parser(language)
                            if parser:
                                file_dependencies = parser.extract_dependencies(file_path, content)
                                dependencies_found += len(file_dependencies)
                        except Exception as parse_error:
                            print(f"Error parsing dependencies in {file_path}: {parse_error}")
                    
                    # Check if file already exists for this project
                    existing = await conn.fetchval("""
                        SELECT id FROM file_entries 
                        WHERE project_id = $1 AND relative_path = $2
                    """, UUID(project_id), relative_path)
                    
                    if existing:
                        # Update existing file entry
                        await conn.execute("""
                            UPDATE file_entries SET
                                file_size = $3, last_modified = $4, indexed_at = $5,
                                content_preview = $6, line_count = $7
                            WHERE id = $1 AND project_id = $2
                        """, existing, UUID(project_id), file_stat.st_size,
                        datetime.fromtimestamp(file_stat.st_mtime), datetime.utcnow(),
                        content_preview, line_count)
                        file_entry_id = existing
                    else:
                        # Insert new file entry
                        file_entry_id = await conn.fetchval("""
                            INSERT INTO file_entries (
                                project_id, file_path, relative_path, file_name, file_extension,
                                file_type, language, file_size, last_modified, indexed_at,
                                content_preview, line_count
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                            RETURNING id
                        """, 
                        UUID(project_id), str(file_path), relative_path, file_path.name,
                        extension, file_type, language, file_stat.st_size,
                        datetime.fromtimestamp(file_stat.st_mtime), datetime.utcnow(),
                        content_preview, line_count)
                    
                    # Insert dependencies
                    for dep in file_dependencies:
                        try:
                            await conn.execute("""
                                INSERT INTO dependency_relationships (
                                    project_id, source_file_id, target_name, dependency_type,
                                    line_number, column_number, source_text, is_external,
                                    is_dynamic, confidence_score, metadata
                                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                            """,
                            UUID(project_id), file_entry_id, dep.target_name, dep.dependency_type,
                            dep.line_number, dep.column_number, dep.source_text, dep.is_external,
                            dep.is_dynamic, dep.confidence_score, json.dumps(dep.metadata or {}))
                        except Exception as dep_error:
                            print(f"Error inserting dependency {dep.target_name}: {dep_error}")
                    
                    processed_files += 1
                    progress = (processed_files / total_files) * 100
                    
                    # Update progress every 10 files
                    if processed_files % 10 == 0 or processed_files == total_files:
                        await conn.execute("""
                            UPDATE analysis_sessions 
                            SET files_processed = $2, progress_percentage = $3
                            WHERE id = $1
                        """, UUID(session_id), processed_files, progress)
                        
                except Exception as e:
                    # Log error but continue processing
                    print(f"Error processing file {file_path}: {e}")
                    continue
            
            # Update project file count and dependency count
            await conn.execute("""
                UPDATE project_indexes 
                SET file_count = $2, dependency_count = $3, last_indexed_at = $4, status = 'active'
                WHERE id = $1
            """, UUID(project_id), processed_files, dependencies_found, datetime.utcnow())
            
            # Complete analysis session
            await conn.execute("""
                UPDATE analysis_sessions 
                SET status = 'completed', progress_percentage = 100.0, completed_at = $2,
                    current_phase = 'completed'
                WHERE id = $1
            """, UUID(session_id), datetime.utcnow())
            
    except Exception as e:
        # Handle analysis failure
        async with pool.acquire() as conn:
            await conn.execute("""
                UPDATE analysis_sessions 
                SET status = 'failed', error_log = $2, completed_at = $3
                WHERE id = $1
            """, UUID(session_id), [{"error": str(e)}], datetime.utcnow())
        print(f"Analysis failed for project {project_id}: {e}")

# FastAPI app
app = FastAPI(
    title="Project Index API",
    description="Standalone Project Index System",
    version="1.0.0"
)

@app.on_event("startup")
async def startup():
    await get_db_pool()

@app.on_event("shutdown") 
async def shutdown():
    if pool:
        await pool.close()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            # Test database connection
            await conn.fetchval("SELECT 1")
            
            # Check if Project Index tables exist
            tables_exist = await conn.fetchval("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name IN ('project_indexes', 'file_entries', 'dependency_relationships')
            """)
            
            return HealthResponse(
                status="healthy",
                database="connected",
                project_index_tables=bool(tables_exist >= 3)
            )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")

@app.get("/api/project-index", response_model=List[ProjectIndexResponse])
async def list_project_indexes():
    """List all project indexes"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, name, description, root_path, git_repository_url, git_branch,
                   status, file_count, dependency_count, created_at
            FROM project_indexes
            ORDER BY created_at DESC
        """)
        
        return [
            ProjectIndexResponse(
                id=str(row['id']),
                name=row['name'],
                description=row['description'],
                root_path=row['root_path'],
                git_repository_url=row['git_repository_url'],
                git_branch=row['git_branch'],
                status=row['status'],
                file_count=row['file_count'],
                dependency_count=row['dependency_count'],
                created_at=row['created_at']
            )
            for row in rows
        ]

@app.post("/api/project-index/create", response_model=ProjectIndexResponse, status_code=201)
async def create_project_index(project: ProjectIndexCreate):
    """Create a new project index"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        project_id = uuid4()
        
        row = await conn.fetchrow("""
            INSERT INTO project_indexes (id, name, description, root_path, git_repository_url, git_branch, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id, name, description, root_path, git_repository_url, git_branch, 
                      status, file_count, dependency_count, created_at
        """, project_id, project.name, project.description, project.root_path,
            project.git_repository_url, project.git_branch, 'inactive')
        
        return ProjectIndexResponse(
            id=str(row['id']),
            name=row['name'],
            description=row['description'], 
            root_path=row['root_path'],
            git_repository_url=row['git_repository_url'],
            git_branch=row['git_branch'],
            status=row['status'],
            file_count=row['file_count'],
            dependency_count=row['dependency_count'],
            created_at=row['created_at']
        )

@app.get("/api/project-index/{project_id}", response_model=ProjectIndexResponse)
async def get_project_index(project_id: str):
    """Get a specific project index"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT id, name, description, root_path, git_repository_url, git_branch,
                   status, file_count, dependency_count, created_at
            FROM project_indexes
            WHERE id = $1
        """, UUID(project_id))
        
        if not row:
            raise HTTPException(status_code=404, detail="Project index not found")
        
        return ProjectIndexResponse(
            id=str(row['id']),
            name=row['name'],
            description=row['description'],
            root_path=row['root_path'],
            git_repository_url=row['git_repository_url'],
            git_branch=row['git_branch'],
            status=row['status'],
            file_count=row['file_count'],
            dependency_count=row['dependency_count'],
            created_at=row['created_at']
        )

@app.post("/api/project-index/{project_id}/analyze", response_model=AnalysisResponse)
async def trigger_project_analysis(
    project_id: str, 
    request: AnalysisRequest = AnalysisRequest(),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Trigger project analysis"""
    if not FULL_INTEGRATION_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Full analysis engine not available. Install full bee-hive application for complete functionality."
        )
    
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # Verify project exists
        project_row = await conn.fetchrow("""
            SELECT id, name, root_path FROM project_indexes WHERE id = $1
        """, UUID(project_id))
        
        if not project_row:
            raise HTTPException(status_code=404, detail="Project index not found")
        
        # Create analysis session in the database
        session_id = uuid4()
        session_name = f"{request.analysis_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        await conn.execute("""
            INSERT INTO analysis_sessions (id, project_id, session_name, session_type, status, started_at)
            VALUES ($1, $2, $3, $4, $5, $6)
        """, session_id, UUID(project_id), session_name, request.analysis_type, 'running', datetime.utcnow())
        
        # Schedule background analysis task
        background_tasks.add_task(
            run_project_analysis_background,
            project_id,
            str(session_id),
            request.analysis_type,
            request.force_reanalysis,
            project_row['root_path']
        )
        
        return AnalysisResponse(
            session_id=str(session_id),
            status="started",
            message=f"Analysis started for project {project_row['name']}",
            started_at=datetime.utcnow()
        )

@app.get("/api/project-index/{project_id}/analysis/{session_id}", response_model=AnalysisStatusResponse)
async def get_analysis_status(project_id: str, session_id: str):
    """Get analysis session status"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        session_row = await conn.fetchrow("""
            SELECT session_name, session_type, status, progress_percentage, 
                   files_processed, files_total, errors_count, current_phase,
                   estimated_completion, started_at, completed_at
            FROM analysis_sessions 
            WHERE id = $1 AND project_id = $2
        """, UUID(session_id), UUID(project_id))
        
        if not session_row:
            raise HTTPException(status_code=404, detail="Analysis session not found")
        
        return AnalysisStatusResponse(
            session_id=session_id,
            status=session_row['status'],
            progress_percentage=float(session_row['progress_percentage'] or 0),
            files_processed=session_row['files_processed'] or 0,
            files_total=session_row['files_total'] or 0,
            errors_count=session_row['errors_count'] or 0,
            current_phase=session_row['current_phase'],
            estimated_completion=session_row['estimated_completion']
        )

@app.get("/api/project-index/{project_id}/files", response_model=List[FileDetailResponse])
async def get_project_files(
    project_id: str,
    language: Optional[str] = None,
    file_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """Get files in a project with optional filtering"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # Build query with optional filters
        where_conditions = ["fe.project_id = $1"]
        params = [UUID(project_id)]
        param_count = 1
        
        if language:
            param_count += 1
            where_conditions.append(f"fe.language = ${param_count}")
            params.append(language)
        
        if file_type:
            param_count += 1
            where_conditions.append(f"fe.file_type = ${param_count}")
            params.append(file_type)
        
        where_clause = " AND ".join(where_conditions)
        
        query = f"""
            SELECT fe.id, fe.relative_path, fe.file_name, fe.file_extension as file_type,
                   fe.language, fe.file_size, fe.line_count, fe.content_preview, fe.last_modified,
                   COUNT(dr.id) as dependency_count
            FROM file_entries fe
            LEFT JOIN dependency_relationships dr ON fe.id = dr.source_file_id
            WHERE {where_clause}
            GROUP BY fe.id, fe.relative_path, fe.file_name, fe.file_extension,
                     fe.language, fe.file_size, fe.line_count, fe.content_preview, fe.last_modified
            ORDER BY fe.relative_path
            LIMIT ${param_count + 1} OFFSET ${param_count + 2}
        """
        
        params.extend([limit, offset])
        rows = await conn.fetch(query, *params)
        
        files = []
        for row in rows:
            # Get dependencies for this file
            deps_query = """
                SELECT target_name, dependency_type, is_external, line_number, 
                       source_text, confidence_score, metadata
                FROM dependency_relationships 
                WHERE source_file_id = $1
                ORDER BY line_number, target_name
            """
            dep_rows = await conn.fetch(deps_query, row['id'])
            
            dependencies = [
                DependencyResponse(
                    target_name=dep['target_name'],
                    dependency_type=dep['dependency_type'],
                    is_external=dep['is_external'],
                    line_number=dep['line_number'],
                    source_text=dep['source_text'],
                    confidence_score=float(dep['confidence_score']),
                    metadata=dep['metadata'] if isinstance(dep['metadata'], dict) else {}
                )
                for dep in dep_rows
            ]
            
            file_detail = FileDetailResponse(
                id=str(row['id']),
                relative_path=row['relative_path'],
                file_name=row['file_name'],
                file_type=row['file_type'],
                language=row['language'],
                file_size=row['file_size'],
                line_count=row['line_count'],
                content_preview=row['content_preview'],
                last_modified=row['last_modified'],
                dependencies=dependencies
            )
            files.append(file_detail)
        
        return files

@app.get("/api/project-index/{project_id}/dependencies")
async def get_project_dependencies(
    project_id: str,
    dependency_type: Optional[str] = None,
    is_external: Optional[bool] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get dependencies in a project with optional filtering"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        where_conditions = ["dr.project_id = $1"]
        params = [UUID(project_id)]
        param_count = 1
        
        if dependency_type:
            param_count += 1
            where_conditions.append(f"dr.dependency_type = ${param_count}")
            params.append(dependency_type)
        
        if is_external is not None:
            param_count += 1
            where_conditions.append(f"dr.is_external = ${param_count}")
            params.append(is_external)
        
        where_clause = " AND ".join(where_conditions)
        
        query = f"""
            SELECT dr.target_name, dr.dependency_type, dr.is_external, dr.line_number,
                   dr.source_text, dr.confidence_score, dr.metadata,
                   fe.relative_path, fe.file_name, fe.language,
                   COUNT(*) OVER() as total_count
            FROM dependency_relationships dr
            JOIN file_entries fe ON dr.source_file_id = fe.id
            WHERE {where_clause}
            ORDER BY dr.target_name, fe.relative_path
            LIMIT ${param_count + 1} OFFSET ${param_count + 2}
        """
        
        params.extend([limit, offset])
        rows = await conn.fetch(query, *params)
        
        dependencies = []
        total_count = 0
        
        for row in rows:
            total_count = row['total_count']
            
            dep = {
                "target_name": row['target_name'],
                "dependency_type": row['dependency_type'],
                "is_external": row['is_external'],
                "line_number": row['line_number'],
                "source_text": row['source_text'],
                "confidence_score": float(row['confidence_score']),
                "metadata": row['metadata'] if isinstance(row['metadata'], dict) else {},
                "source_file": {
                    "relative_path": row['relative_path'],
                    "file_name": row['file_name'],
                    "language": row['language']
                }
            }
            dependencies.append(dep)
        
        return {
            "dependencies": dependencies,
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }

@app.get("/api/project-index/{project_id}/search")
async def search_project_files(
    project_id: str,
    query: str,
    file_type: Optional[str] = None,
    language: Optional[str] = None,
    limit: int = 20
):
    """Search through project files by content"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        where_conditions = ["fe.project_id = $1"]
        params = [UUID(project_id)]
        param_count = 1
        
        # Add content search condition
        param_count += 1
        where_conditions.append(f"(fe.content_preview ILIKE ${param_count} OR fe.relative_path ILIKE ${param_count})")
        params.append(f"%{query}%")
        
        if file_type:
            param_count += 1
            where_conditions.append(f"fe.file_type = ${param_count}")
            params.append(file_type)
        
        if language:
            param_count += 1
            where_conditions.append(f"fe.language = ${param_count}")
            params.append(language)
        
        where_clause = " AND ".join(where_conditions)
        
        query_sql = f"""
            SELECT fe.id, fe.relative_path, fe.file_name, fe.file_extension as file_type,
                   fe.language, fe.file_size, fe.line_count, fe.content_preview, fe.last_modified,
                   -- Simple relevance scoring
                   CASE 
                       WHEN fe.relative_path ILIKE ${param_count - len(params) + 2} THEN 3
                       WHEN fe.file_name ILIKE ${param_count - len(params) + 2} THEN 2
                       ELSE 1
                   END as relevance_score
            FROM file_entries fe
            WHERE {where_clause}
            ORDER BY relevance_score DESC, fe.relative_path
            LIMIT ${param_count + 1}
        """
        
        params.append(limit)
        rows = await conn.fetch(query_sql, *params)
        
        results = []
        for row in rows:
            result = {
                "id": str(row['id']),
                "relative_path": row['relative_path'],
                "file_name": row['file_name'],
                "file_type": row['file_type'],
                "language": row['language'],
                "file_size": row['file_size'],
                "line_count": row['line_count'],
                "content_preview": row['content_preview'],
                "last_modified": row['last_modified'],
                "relevance_score": row['relevance_score']
            }
            results.append(result)
        
        return {
            "query": query,
            "results": results,
            "total_results": len(results)
        }

@app.post("/api/project-index/{project_id}/context")
async def assemble_context(
    project_id: str,
    request: ContextAssemblyRequest
):
    """Assemble relevant context for AI agents based on task description"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # Verify project exists
        project_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM project_indexes WHERE id = $1)",
            UUID(project_id)
        )
        if not project_exists:
            raise HTTPException(status_code=404, detail="Project index not found")
        
        # Extract keywords from task description for semantic search
        keywords = request.task_description.lower().split()
        search_terms = [word for word in keywords if len(word) > 3][:5]  # Top 5 meaningful terms
        
        # Build search conditions
        where_conditions = ["fe.project_id = $1"]
        params = [UUID(project_id)]
        param_count = 1
        
        # Language filtering
        if request.focus_languages:
            placeholders = []
            for lang in request.focus_languages:
                param_count += 1
                placeholders.append(f'${param_count}')
                params.append(lang)
            where_conditions.append(f"fe.language = ANY(ARRAY[{','.join(placeholders)}])")
        
        # Build simple search condition 
        if search_terms:
            search_term = f"%{search_terms[0]}%"  # Use first search term only for simplicity
            param_count += 1
            where_conditions.append(f"""
                (fe.relative_path ILIKE ${param_count} OR 
                 fe.file_name ILIKE ${param_count} OR 
                 fe.content_preview ILIKE ${param_count})
            """)
            params.append(search_term)
        
        where_clause = " AND ".join(where_conditions)
        
        # Simplified query
        main_query = f"""
            SELECT fe.id, fe.relative_path, fe.file_name, fe.language, 
                   fe.content_preview, fe.file_size, fe.line_count,
                   1 as relevance_score,
                   COUNT(dr.id) as dependency_count
            FROM file_entries fe
            LEFT JOIN dependency_relationships dr ON fe.id = dr.source_file_id
            WHERE {where_clause}
            GROUP BY fe.id, fe.relative_path, fe.file_name, fe.language, 
                     fe.content_preview, fe.file_size, fe.line_count
            ORDER BY fe.file_size ASC
            LIMIT ${param_count + 1}
        """
        
        params.append(request.max_files)
        rows = await conn.fetch(main_query, *params)
        
        context_files = []
        
        for row in rows:
            file_id = row['id']
            
            # Determine inclusion reason
            if row['relevance_score'] >= 4:
                reason = f"High relevance: matches task keywords in file name/path"
            elif row['relevance_score'] >= 3:
                reason = f"Content match: contains task-related keywords"
            elif row['dependency_count'] > 0:
                reason = f"Dependency hub: {row['dependency_count']} dependencies"
            else:
                reason = "General relevance to task"
            
            # Get related files via dependencies if requested
            related_files = []
            if request.include_dependencies:
                related_query = """
                    SELECT DISTINCT fe2.relative_path
                    FROM dependency_relationships dr
                    JOIN file_entries fe2 ON dr.source_file_id = fe2.id
                    WHERE dr.target_name IN (
                        SELECT dr2.target_name 
                        FROM dependency_relationships dr2 
                        WHERE dr2.source_file_id = $1
                        AND dr2.is_external = false
                        LIMIT 3
                    )
                    AND fe2.id != $1
                    LIMIT 5
                """
                related_rows = await conn.fetch(related_query, file_id)
                related_files = [r['relative_path'] for r in related_rows]
            
            context_file = ContextFileResult(
                id=str(file_id),
                relative_path=row['relative_path'],
                file_name=row['file_name'],
                language=row['language'],
                relevance_score=float(row['relevance_score']),
                content_preview=row['content_preview'],
                reason=reason,
                related_files=related_files
            )
            context_files.append(context_file)
        
        # Calculate context statistics
        total_lines = sum(file.content_preview.count('\n') if file.content_preview else 0 for file in context_files)
        estimated_tokens = total_lines * 15  # Rough estimate: ~15 tokens per line
        
        return {
            "task_description": request.task_description,
            "context_files": context_files,
            "total_files": len(context_files),
            "estimated_tokens": estimated_tokens,
            "search_terms": search_terms,
            "assembly_metadata": {
                "include_dependencies": request.include_dependencies,
                "focus_languages": request.focus_languages,
                "max_files_requested": request.max_files,
                "total_project_files": await conn.fetchval(
                    "SELECT COUNT(*) FROM file_entries WHERE project_id = $1", 
                    UUID(project_id)
                )
            }
        }

@app.post("/api/project-index/{project_id}/decompose-task", response_model=TaskDecompositionResponse)
async def decompose_task(
    project_id: str,
    request: TaskDecompositionRequest
):
    """Decompose a large task into agent-sized subtasks"""
    if not FULL_INTEGRATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Agent delegation requires full Project Index integration"
        )
    
    pool = await get_db_pool()
    
    # Verify project exists
    async with pool.acquire() as conn:
        project_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM project_indexes WHERE id = $1)",
            UUID(project_id)
        )
        if not project_exists:
            raise HTTPException(status_code=404, detail="Project index not found")
    
    # Map string to TaskType enum
    task_type_mapping = {
        "feature-implementation": TaskType.FEATURE_IMPLEMENTATION,
        "bug-fix": TaskType.BUG_FIX,
        "refactoring": TaskType.REFACTORING,
        "testing": TaskType.TESTING,
        "documentation": TaskType.DOCUMENTATION,
        "optimization": TaskType.OPTIMIZATION,
        "security-audit": TaskType.SECURITY_AUDIT,
        "database-migration": TaskType.DATABASE_MIGRATION,
        "api-development": TaskType.API_DEVELOPMENT,
        "ui-implementation": TaskType.UI_IMPLEMENTATION
    }
    
    task_type = task_type_mapping.get(request.task_type, TaskType.FEATURE_IMPLEMENTATION)
    
    # Create task decomposer and perform decomposition
    decomposer = TaskDecomposer(UUID(project_id), pool)
    decomposition_result = await decomposer.decompose_task(request.task_description, task_type)
    
    # Convert result to response format
    def convert_task_to_info(task: AgentTask) -> SubtaskInfo:
        return SubtaskInfo(
            id=task.id,
            title=task.title,
            description=task.description,
            complexity=task.complexity.value,
            estimated_duration_minutes=task.estimated_duration_minutes,
            preferred_specialization=task.preferred_specialization.value,
            primary_files=task.primary_files,
            dependencies=task.dependency_task_ids
        )
    
    return TaskDecompositionResponse(
        original_task=convert_task_to_info(decomposition_result.original_task),
        subtasks=[convert_task_to_info(task) for task in decomposition_result.subtasks],
        coordination_plan=decomposition_result.coordination_plan,
        estimated_total_duration=decomposition_result.estimated_total_duration,
        decomposition_strategy=decomposition_result.decomposition_strategy,
        success=decomposition_result.success,
        reason=decomposition_result.reason
    )

@app.post("/api/project-index/{project_id}/assign-agents")
async def assign_agents_to_tasks(
    project_id: str,
    request: TaskDecompositionRequest  # For simplicity, reusing the same request
):
    """Assign agents to decomposed tasks with coordination"""
    if not FULL_INTEGRATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Agent coordination requires full Project Index integration"
        )
    
    pool = await get_db_pool()
    
    # First decompose the task
    task_type_mapping = {
        "feature-implementation": TaskType.FEATURE_IMPLEMENTATION,
        "bug-fix": TaskType.BUG_FIX,
        "refactoring": TaskType.REFACTORING,
        "testing": TaskType.TESTING,
        "documentation": TaskType.DOCUMENTATION,
        "optimization": TaskType.OPTIMIZATION,
        "security-audit": TaskType.SECURITY_AUDIT,
        "database-migration": TaskType.DATABASE_MIGRATION,
        "api-development": TaskType.API_DEVELOPMENT,
        "ui-implementation": TaskType.UI_IMPLEMENTATION
    }
    
    task_type = task_type_mapping.get(request.task_type, TaskType.FEATURE_IMPLEMENTATION)
    
    decomposer = TaskDecomposer(UUID(project_id), pool)
    decomposition_result = await decomposer.decompose_task(request.task_description, task_type)
    
    if not decomposition_result.success:
        raise HTTPException(
            status_code=400,
            detail=f"Task decomposition failed: {decomposition_result.reason}"
        )
    
    # Create agent coordinator and assign agents
    coordinator = AgentCoordinator(UUID(project_id), pool)
    assignment_result = await coordinator.assign_agents_to_tasks(decomposition_result)
    
    # Convert to response format
    assignments = []
    for assignment in assignment_result["assignments"]:
        assignments.append(AgentAssignmentInfo(
            task_id=assignment["task_id"],
            agent_id=assignment["agent_id"],
            specialization=assignment.get("specialization_match", "general-purpose"),
            estimated_start_time=assignment["estimated_start_time"],
            context_requirements={
                "estimated_tokens": assignment["context_requirements"].estimated_tokens,
                "max_files": assignment["context_requirements"].max_files,
                "primary_languages": assignment["context_requirements"].primary_languages
            }
        ))
    
    return AgentCoordinationResponse(
        assignments=assignments,
        coordination_plan=assignment_result["coordination_plan"],
        total_agents=assignment_result["total_agents"],
        estimated_completion=assignment_result["estimated_completion"]
    )

@app.get("/api/project-index/{project_id}/context-monitoring/{agent_id}")
async def monitor_agent_context(
    project_id: str,
    agent_id: str,
    current_context_size: int = 50000
):
    """Monitor agent context usage and get recommendations"""
    if not FULL_INTEGRATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Context monitoring requires full Project Index integration"
        )
    
    pool = await get_db_pool()
    
    # Create context rot prevention system
    context_monitor = ContextRotPrevention(pool)
    
    # Monitor the agent's context
    monitoring_result = await context_monitor.monitor_agent_context(agent_id, current_context_size)
    
    return monitoring_result

@app.post("/api/project-index/{project_id}/refresh-agent-context/{agent_id}")
async def refresh_agent_context(
    project_id: str,
    agent_id: str,
    refresh_type: str = "full"
):
    """Trigger context refresh for an agent"""
    if not FULL_INTEGRATION_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Context refresh requires full Project Index integration"
        )
    
    pool = await get_db_pool()
    
    # Create context rot prevention system
    context_monitor = ContextRotPrevention(pool)
    
    # Trigger refresh
    refresh_result = await context_monitor.trigger_context_refresh(agent_id, refresh_type)
    
    return refresh_result

@app.delete("/api/project-index/{project_id}")
async def delete_project_index(project_id: str):
    """Delete a project index"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        result = await conn.execute("""
            DELETE FROM project_indexes WHERE id = $1
        """, UUID(project_id))
        
        if result == "DELETE 0":
            raise HTTPException(status_code=404, detail="Project index not found")
        
        return {"message": "Project index deleted successfully"}

if __name__ == "__main__":
    uvicorn.run(
        "project_index_server:app",
        host="0.0.0.0",
        port=8081,
        reload=True,
        log_level="info"
    )