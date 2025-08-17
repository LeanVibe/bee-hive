#!/usr/bin/env python3
"""
Standalone Project Index API Server

Simple FastAPI server to test Project Index functionality
without the complexity of the full bee-hive application.
"""

import asyncio
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
            for i, file_path in enumerate(files_found):
                try:
                    # Get file info
                    file_stat = file_path.stat()
                    relative_path = str(file_path.relative_to(project_path))
                    
                    # Determine file type
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
                    
                    # Check if file already exists for this project
                    existing = await conn.fetchval("""
                        SELECT id FROM file_entries 
                        WHERE project_id = $1 AND relative_path = $2
                    """, UUID(project_id), relative_path)
                    
                    if existing:
                        # Update existing file entry
                        await conn.execute("""
                            UPDATE file_entries SET
                                file_size = $3, last_modified = $4, indexed_at = $5
                            WHERE id = $1 AND project_id = $2
                        """, existing, UUID(project_id), file_stat.st_size,
                        datetime.fromtimestamp(file_stat.st_mtime), datetime.utcnow())
                    else:
                        # Insert new file entry
                        await conn.execute("""
                            INSERT INTO file_entries (
                                project_id, file_path, relative_path, file_name, file_extension,
                                file_type, language, file_size, last_modified, indexed_at
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        """, 
                        UUID(project_id), str(file_path), relative_path, file_path.name,
                        extension, file_type, language, file_stat.st_size,
                        datetime.fromtimestamp(file_stat.st_mtime), datetime.utcnow())
                    
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
            
            # Update project file count
            await conn.execute("""
                UPDATE project_indexes 
                SET file_count = $2, last_indexed_at = $3, status = 'active'
                WHERE id = $1
            """, UUID(project_id), processed_files, datetime.utcnow())
            
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