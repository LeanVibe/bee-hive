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
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import uvicorn


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