"""
Comprehensive test data fixtures and mock services for Project Index testing.

This module provides realistic test data, mock services, and factory functions
for creating consistent test scenarios across all test suites.
"""

import uuid
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from unittest.mock import Mock, AsyncMock

import pytest
from faker import Faker

from app.models.project_index import (
    ProjectIndex, FileEntry, DependencyRelationship, AnalysisSession,
    ProjectStatus, FileType, DependencyType, AnalysisStatus, AnalysisSessionType
)

fake = Faker()


@dataclass
class TestProjectData:
    """Test data container for a complete project scenario."""
    project: ProjectIndex
    files: List[FileEntry] = field(default_factory=list)
    dependencies: List[DependencyRelationship] = field(default_factory=list)
    analysis_sessions: List[AnalysisSession] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API testing."""
        return {
            "project": {
                "id": str(self.project.id),
                "name": self.project.name,
                "description": self.project.description,
                "root_path": self.project.root_path,
                "status": self.project.status.value,
                "file_count": len(self.files),
                "dependency_count": len(self.dependencies)
            },
            "files": [
                {
                    "id": str(f.id),
                    "relative_path": f.relative_path,
                    "file_name": f.file_name,
                    "file_type": f.file_type.value,
                    "language": f.language,
                    "file_size": f.file_size,
                    "line_count": f.line_count
                } for f in self.files
            ],
            "dependencies": [
                {
                    "id": str(d.id),
                    "source_file_id": str(d.source_file_id),
                    "target_name": d.target_name,
                    "dependency_type": d.dependency_type.value,
                    "is_external": d.is_external
                } for d in self.dependencies
            ]
        }


class ProjectIndexTestDataFactory:
    """Factory for creating realistic Project Index test data."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            fake.seed_instance(seed)
            random.seed(seed)
    
    def create_project_data(
        self,
        project_type: str = "python_webapp",
        complexity: str = "medium",
        include_dependencies: bool = True,
        include_analysis: bool = True
    ) -> TestProjectData:
        """Create complete project test data based on specified parameters."""
        
        # Create base project
        project = self._create_project(project_type, complexity)
        
        # Create files based on project type
        files = self._create_files_for_project(project, project_type, complexity)
        
        # Create dependencies if requested
        dependencies = []
        if include_dependencies:
            dependencies = self._create_dependencies(project, files)
        
        # Create analysis sessions if requested
        analysis_sessions = []
        if include_analysis:
            analysis_sessions = self._create_analysis_sessions(project, len(files), len(dependencies))
        
        return TestProjectData(
            project=project,
            files=files,
            dependencies=dependencies,
            analysis_sessions=analysis_sessions
        )
    
    def _create_project(self, project_type: str, complexity: str) -> ProjectIndex:
        """Create a project based on type and complexity."""
        
        project_templates = {
            "python_webapp": {
                "name": f"{fake.word().title()} Web Application",
                "description": "A Python web application with FastAPI and React frontend",
                "languages": ["python", "javascript", "typescript"],
                "framework": "fastapi"
            },
            "data_science": {
                "name": f"{fake.word().title()} Analytics Platform", 
                "description": "Data science project with ML models and analysis notebooks",
                "languages": ["python", "jupyter"],
                "framework": "sklearn"
            },
            "microservice": {
                "name": f"{fake.word().title()} Microservice",
                "description": "Containerized microservice with API endpoints",
                "languages": ["python"],
                "framework": "fastapi"
            },
            "frontend_spa": {
                "name": f"{fake.word().title()} Dashboard",
                "description": "Single-page application with modern frontend stack",
                "languages": ["javascript", "typescript", "css"],
                "framework": "react"
            }
        }
        
        template = project_templates.get(project_type, project_templates["python_webapp"])
        
        complexity_settings = {
            "simple": {"file_count_range": (5, 15), "depth": 2},
            "medium": {"file_count_range": (15, 50), "depth": 3},
            "complex": {"file_count_range": (50, 150), "depth": 5},
            "large": {"file_count_range": (150, 500), "depth": 7}
        }
        
        settings = complexity_settings.get(complexity, complexity_settings["medium"])
        
        return ProjectIndex(
            id=uuid.uuid4(),
            name=template["name"],
            description=template["description"],
            root_path=f"/test/projects/{fake.slug()}",
            git_repository_url=f"https://github.com/{fake.user_name()}/{fake.slug()}.git",
            git_branch="main",
            status=random.choice([ProjectStatus.ACTIVE, ProjectStatus.INACTIVE]),
            configuration={
                "languages": template["languages"],
                "analysis_depth": settings["depth"],
                "framework": template.get("framework"),
                "enable_ai_analysis": True,
                "max_file_size": 1024 * 1024  # 1MB
            },
            analysis_settings={
                "parse_ast": True,
                "extract_dependencies": True,
                "calculate_complexity": True,
                "detect_patterns": True
            },
            file_patterns={
                "include": ["**/*.py", "**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx"],
                "exclude": ["**/node_modules/**", "**/__pycache__/**", "**/venv/**"]
            },
            meta_data={
                "created_by": fake.user_name(),
                "team": fake.word(),
                "environment": random.choice(["development", "staging", "production"]),
                "tags": [fake.word() for _ in range(random.randint(1, 4))]
            },
            created_at=fake.date_time_between(start_date="-30d", end_date="now"),
            updated_at=datetime.utcnow()
        )
    
    def _create_files_for_project(
        self, 
        project: ProjectIndex, 
        project_type: str, 
        complexity: str
    ) -> List[FileEntry]:
        """Create realistic file entries for a project."""
        
        file_structures = {
            "python_webapp": [
                ("app/main.py", "python", FileType.SOURCE, self._python_main_content),
                ("app/models/user.py", "python", FileType.SOURCE, self._python_model_content),
                ("app/api/endpoints.py", "python", FileType.SOURCE, self._python_api_content),
                ("app/core/config.py", "python", FileType.SOURCE, self._python_config_content),
                ("app/utils/helpers.py", "python", FileType.SOURCE, self._python_utils_content),
                ("tests/test_main.py", "python", FileType.TEST, self._python_test_content),
                ("tests/test_api.py", "python", FileType.TEST, self._python_test_content),
                ("requirements.txt", "text", FileType.CONFIGURATION, self._requirements_content),
                ("Dockerfile", "dockerfile", FileType.CONFIGURATION, self._dockerfile_content),
                ("README.md", "markdown", FileType.DOCUMENTATION, self._readme_content),
                ("frontend/src/App.tsx", "typescript", FileType.SOURCE, self._react_component_content),
                ("frontend/src/api/client.ts", "typescript", FileType.SOURCE, self._api_client_content),
                ("frontend/package.json", "json", FileType.CONFIGURATION, self._package_json_content)
            ],
            "data_science": [
                ("src/analysis.py", "python", FileType.SOURCE, self._python_analysis_content),
                ("src/models/classifier.py", "python", FileType.SOURCE, self._ml_model_content),
                ("src/data/loader.py", "python", FileType.SOURCE, self._data_loader_content),
                ("notebooks/exploration.ipynb", "jupyter", FileType.NOTEBOOK, self._jupyter_content),
                ("notebooks/modeling.ipynb", "jupyter", FileType.NOTEBOOK, self._jupyter_content),
                ("tests/test_models.py", "python", FileType.TEST, self._python_test_content),
                ("config/settings.yaml", "yaml", FileType.CONFIGURATION, self._yaml_config_content),
                ("requirements.txt", "text", FileType.CONFIGURATION, self._requirements_content),
                ("README.md", "markdown", FileType.DOCUMENTATION, self._readme_content)
            ],
            "microservice": [
                ("src/main.py", "python", FileType.SOURCE, self._python_main_content),
                ("src/handlers.py", "python", FileType.SOURCE, self._python_api_content),
                ("src/models.py", "python", FileType.SOURCE, self._python_model_content),
                ("src/database.py", "python", FileType.SOURCE, self._python_db_content),
                ("tests/test_handlers.py", "python", FileType.TEST, self._python_test_content),
                ("docker-compose.yml", "yaml", FileType.CONFIGURATION, self._docker_compose_content),
                ("Dockerfile", "dockerfile", FileType.CONFIGURATION, self._dockerfile_content),
                ("requirements.txt", "text", FileType.CONFIGURATION, self._requirements_content),
                ("README.md", "markdown", FileType.DOCUMENTATION, self._readme_content)
            ]
        }
        
        file_structure = file_structures.get(project_type, file_structures["python_webapp"])
        
        complexity_multipliers = {
            "simple": 0.5,
            "medium": 1.0, 
            "complex": 2.0,
            "large": 4.0
        }
        
        multiplier = complexity_multipliers.get(complexity, 1.0)
        
        files = []
        for relative_path, language, file_type, content_generator in file_structure:
            # Adjust file count based on complexity
            if file_type == FileType.SOURCE and multiplier > 1.0:
                # Add additional source files for complex projects
                for i in range(int(multiplier)):
                    if i == 0:
                        path = relative_path
                    else:
                        base_path = Path(relative_path)
                        path = str(base_path.parent / f"{base_path.stem}_{i}{base_path.suffix}")
                    
                    files.append(self._create_file_entry(project, path, language, file_type, content_generator))
            else:
                files.append(self._create_file_entry(project, relative_path, language, file_type, content_generator))
        
        return files
    
    def _create_file_entry(
        self,
        project: ProjectIndex,
        relative_path: str,
        language: str,
        file_type: FileType,
        content_generator
    ) -> FileEntry:
        """Create a single file entry with realistic content."""
        
        content = content_generator()
        file_size = len(content.encode('utf-8'))
        line_count = content.count('\n') + 1
        
        # Calculate complexity based on content
        complexity_score = self._calculate_complexity(content, language)
        
        return FileEntry(
            id=uuid.uuid4(),
            project_id=project.id,
            file_path=str(Path(project.root_path) / relative_path),
            relative_path=relative_path,
            file_name=Path(relative_path).name,
            file_extension=Path(relative_path).suffix or ".txt",
            file_type=file_type,
            language=language,
            file_size=file_size,
            line_count=line_count,
            sha256_hash=fake.sha256(),
            content_preview=content[:500] + "..." if len(content) > 500 else content,
            analysis_data={
                "functions": self._extract_functions(content, language),
                "imports": self._extract_imports(content, language),
                "complexity_metrics": {
                    "cyclomatic_complexity": complexity_score,
                    "cognitive_complexity": max(1, complexity_score - 1),
                    "maintainability_index": random.randint(60, 100)
                },
                "quality_metrics": {
                    "code_duplication": random.uniform(0, 0.1),
                    "test_coverage": random.uniform(0.7, 1.0) if file_type == FileType.TEST else None
                }
            },
            tags=self._generate_file_tags(relative_path, file_type, language),
            last_modified=fake.date_time_between(start_date="-7d", end_date="now"),
            indexed_at=datetime.utcnow()
        )
    
    def _create_dependencies(
        self, 
        project: ProjectIndex, 
        files: List[FileEntry]
    ) -> List[DependencyRelationship]:
        """Create realistic dependency relationships between files and external packages."""
        
        dependencies = []
        source_files = [f for f in files if f.file_type in [FileType.SOURCE, FileType.TEST]]
        
        # Standard library dependencies for Python files
        python_stdlib = [
            "os", "sys", "json", "datetime", "pathlib", "logging", "typing",
            "asyncio", "collections", "itertools", "functools", "dataclasses"
        ]
        
        # External package dependencies
        external_packages = {
            "python": [
                "fastapi", "pydantic", "sqlalchemy", "redis", "httpx", "pytest", 
                "uvicorn", "alembic", "click", "python-multipart", "jinja2"
            ],
            "javascript": [
                "react", "axios", "lodash", "moment", "express", "webpack"
            ],
            "typescript": [
                "@types/node", "@types/react", "typescript", "ts-node"
            ]
        }
        
        for source_file in source_files:
            language = source_file.language
            
            # Add external dependencies
            if language in external_packages:
                for package in random.sample(
                    external_packages[language], 
                    random.randint(2, min(6, len(external_packages[language])))
                ):
                    dependencies.append(DependencyRelationship(
                        id=uuid.uuid4(),
                        project_id=project.id,
                        source_file_id=source_file.id,
                        target_name=package,
                        dependency_type=DependencyType.IMPORT,
                        line_number=random.randint(1, 10),
                        source_text=f"import {package}" if language == "python" else f"import {{ {package} }}",
                        is_external=True,
                        confidence_score=random.uniform(0.8, 1.0),
                        metadata={
                            "package_type": "external",
                            "package_manager": "pip" if language == "python" else "npm"
                        }
                    ))
            
            # Add standard library dependencies for Python
            if language == "python":
                for stdlib_module in random.sample(python_stdlib, random.randint(2, 5)):
                    dependencies.append(DependencyRelationship(
                        id=uuid.uuid4(),
                        project_id=project.id,
                        source_file_id=source_file.id,
                        target_name=stdlib_module,
                        dependency_type=DependencyType.IMPORT,
                        line_number=random.randint(1, 8),
                        source_text=f"import {stdlib_module}",
                        is_external=True,
                        confidence_score=1.0,
                        metadata={
                            "standard_library": True,
                            "python_version": "3.12"
                        }
                    ))
            
            # Add internal dependencies (file-to-file)
            potential_targets = [f for f in source_files if f.id != source_file.id and f.language == language]
            for target_file in random.sample(
                potential_targets, 
                random.randint(0, min(3, len(potential_targets)))
            ):
                # Calculate relative import path
                source_parts = Path(source_file.relative_path).parts[:-1]
                target_parts = Path(target_file.relative_path).parts[:-1]
                
                if source_parts == target_parts:
                    # Same directory
                    import_name = Path(target_file.file_name).stem
                else:
                    # Different directory - create relative path
                    import_name = ".".join(target_parts + (Path(target_file.file_name).stem,))
                
                dependencies.append(DependencyRelationship(
                    id=uuid.uuid4(),
                    project_id=project.id,
                    source_file_id=source_file.id,
                    target_file_id=target_file.id,
                    target_path=target_file.relative_path,
                    target_name=import_name,
                    dependency_type=DependencyType.IMPORT,
                    line_number=random.randint(10, 20),
                    source_text=f"from {import_name} import something",
                    is_external=False,
                    confidence_score=random.uniform(0.85, 1.0),
                    metadata={
                        "internal_dependency": True,
                        "import_type": "relative"
                    }
                ))
        
        return dependencies
    
    def _create_analysis_sessions(
        self, 
        project: ProjectIndex, 
        file_count: int, 
        dependency_count: int
    ) -> List[AnalysisSession]:
        """Create realistic analysis session history."""
        
        sessions = []
        
        # Initial full analysis
        full_analysis = AnalysisSession(
            id=uuid.uuid4(),
            project_id=project.id,
            session_name="Initial Project Analysis",
            session_type=AnalysisSessionType.FULL_ANALYSIS,
            status=AnalysisStatus.COMPLETED,
            started_at=project.created_at + timedelta(minutes=5),
            completed_at=project.created_at + timedelta(minutes=25),
            progress_percentage=100.0,
            current_phase="completed",
            files_total=file_count,
            files_processed=file_count,
            dependencies_found=dependency_count,
            performance_metrics={
                "total_duration": 1200.0,  # 20 minutes
                "avg_file_analysis_time": 1200.0 / file_count,
                "peak_memory_usage": random.randint(100, 500) * 1024 * 1024,
                "cache_hit_rate": random.uniform(0.6, 0.9)
            },
            result_data={
                "summary": {
                    "files_analyzed": file_count,
                    "dependencies_extracted": dependency_count,
                    "languages_detected": project.configuration.get("languages", []),
                    "analysis_quality": "high"
                },
                "statistics": {
                    "total_lines_of_code": random.randint(1000, 10000),
                    "avg_complexity": random.uniform(2.0, 6.0),
                    "test_coverage": random.uniform(0.6, 0.9),
                    "documentation_coverage": random.uniform(0.4, 0.8)
                },
                "insights": [
                    "High code complexity in core modules",
                    "Good test coverage across the project",
                    "Well-structured dependency graph"
                ]
            }
        )
        sessions.append(full_analysis)
        
        # Recent incremental analyses
        for i in range(random.randint(1, 4)):
            incremental = AnalysisSession(
                id=uuid.uuid4(),
                project_id=project.id,
                session_name=f"Incremental Analysis #{i+1}",
                session_type=AnalysisSessionType.INCREMENTAL,
                status=AnalysisStatus.COMPLETED,
                started_at=fake.date_time_between(start_date="-7d", end_date="now"),
                files_total=random.randint(1, max(1, file_count // 4)),
                files_processed=random.randint(1, max(1, file_count // 4)),
                dependencies_found=random.randint(0, max(1, dependency_count // 4)),
                session_data={
                    "trigger": "file_modification",
                    "modified_files": [f"src/file_{i}.py" for i in range(random.randint(1, 3))],
                    "change_type": "content_update"
                }
            )
            incremental.complete_session({
                "files_updated": incremental.files_processed,
                "new_dependencies": random.randint(0, 2),
                "removed_dependencies": random.randint(0, 1)
            })
            sessions.append(incremental)
        
        return sessions
    
    # Content generation methods
    def _python_main_content(self) -> str:
        return f'''#!/usr/bin/env python3
"""
{fake.sentence(nb_words=8)}
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import engine
from app.api import endpoints

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="{fake.company()}",
    description="{fake.text(max_nb_chars=100)}",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(endpoints.router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {{"status": "healthy", "timestamp": "{datetime.utcnow().isoformat()}"}}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
'''
    
    def _python_model_content(self) -> str:
        return f'''"""
Database models for {fake.word()} functionality.
"""

from datetime import datetime
from typing import Optional, List
import uuid

from sqlalchemy import Column, String, DateTime, Boolean, Integer, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class {fake.word().title()}(Base):
    """Model for {fake.word()} entities."""
    
    __tablename__ = "{fake.word()}_table"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    status = Column(String(50), default="active", index=True)
    metadata = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<{fake.word().title()}(id={{self.id}}, name={{self.name}})>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {{
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }}
'''
    
    def _python_api_content(self) -> str:
        return f'''"""
API endpoints for {fake.word()} operations.
"""

from typing import List, Optional
import uuid

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.{fake.word()} import {fake.word().title()}
from app.schemas.{fake.word()} import {fake.word().title()}Create, {fake.word().title()}Update, {fake.word().title()}Response

router = APIRouter()

@router.get("/{fake.word()}s/", response_model=List[{fake.word().title()}Response])
async def list_{fake.word()}s(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """List all {fake.word()}s with pagination."""
    items = db.query({fake.word().title()}).offset(skip).limit(limit).all()
    return items

@router.post("/{fake.word()}s/", response_model={fake.word().title()}Response, status_code=201)
async def create_{fake.word()}(
    item_data: {fake.word().title()}Create,
    db: Session = Depends(get_db)
):
    """Create a new {fake.word()}."""
    item = {fake.word().title()}(**item_data.dict())
    db.add(item)
    db.commit()
    db.refresh(item)
    return item

@router.get("/{fake.word()}s/{{item_id}}", response_model={fake.word().title()}Response)
async def get_{fake.word()}(
    item_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Get a specific {fake.word()} by ID."""
    item = db.query({fake.word().title()}).filter({fake.word().title()}.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="{fake.word().title()} not found")
    return item

@router.put("/{fake.word()}s/{{item_id}}", response_model={fake.word().title()}Response)
async def update_{fake.word()}(
    item_id: uuid.UUID,
    item_data: {fake.word().title()}Update,
    db: Session = Depends(get_db)
):
    """Update a {fake.word()}."""
    item = db.query({fake.word().title()}).filter({fake.word().title()}.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="{fake.word().title()} not found")
    
    for key, value in item_data.dict(exclude_unset=True).items():
        setattr(item, key, value)
    
    db.commit()
    db.refresh(item)
    return item

@router.delete("/{fake.word()}s/{{item_id}}")
async def delete_{fake.word()}(
    item_id: uuid.UUID,
    db: Session = Depends(get_db)
):
    """Delete a {fake.word()}."""
    item = db.query({fake.word().title()}).filter({fake.word().title()}.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="{fake.word().title()} not found")
    
    db.delete(item)
    db.commit()
    return {{"message": "{fake.word().title()} deleted successfully"}}
'''
    
    def _python_config_content(self) -> str:
        return f'''"""
Configuration settings for the application.
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    app_name: str = Field(default="{fake.company()}", env="APP_NAME")
    debug: bool = Field(default=False, env="DEBUG")
    version: str = Field(default="1.0.0", env="VERSION")
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    database_pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_max_connections: int = Field(default=100, env="REDIS_MAX_CONNECTIONS")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
'''
    
    def _python_utils_content(self) -> str:
        return f'''"""
Utility functions for {fake.word()} operations.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

def calculate_hash(content: Union[str, bytes]) -> str:
    """Calculate SHA256 hash of content."""
    if isinstance(content, str):
        content = content.encode('utf-8')
    return hashlib.sha256(content).hexdigest()

def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Format datetime to string."""
    if dt is None:
        return ""
    return dt.strftime(format_str)

def parse_datetime(dt_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> Optional[datetime]:
    """Parse datetime string."""
    try:
        return datetime.strptime(dt_str, format_str)
    except (ValueError, TypeError):
        return None

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string."""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(data: Any, default: str = "{{}}") -> str:
    """Safely serialize data to JSON."""
    try:
        return json.dumps(data, default=str)
    except (TypeError, ValueError):
        return default

def measure_execution_time(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{{func.__name__}} executed in {{execution_time:.4f}} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{{func.__name__}} failed after {{execution_time:.4f}} seconds: {{e}}")
            raise
    return wrapper

def validate_file_path(file_path: Union[str, Path], base_path: Optional[Union[str, Path]] = None) -> bool:
    """Validate if file path is safe and within base path."""
    try:
        path = Path(file_path).resolve()
        
        if base_path:
            base = Path(base_path).resolve()
            return str(path).startswith(str(base))
        
        return path.exists() and path.is_file()
    except (OSError, ValueError):
        return False

class RateLimiter:
    """Simple rate limiter implementation."""
    
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {{}}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for identifier."""
        now = time.time()
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Remove old requests outside time window
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if now - req_time < self.time_window
        ]
        
        # Check if under limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(now)
            return True
        
        return False
'''
    
    def _python_test_content(self) -> str:
        return f'''"""
Tests for {fake.word()} functionality.
"""

import pytest
import uuid
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.models.{fake.word()} import {fake.word().title()}
from app.core.database import get_db

client = TestClient(app)

class Test{fake.word().title()}:
    """Test cases for {fake.word()} operations."""
    
    def test_create_{fake.word()}_success(self, db_session: Session):
        """Test successful {fake.word()} creation."""
        {fake.word()}_data = {{
            "name": "{fake.name()}",
            "description": "{fake.text(max_nb_chars=100)}",
            "status": "active"
        }}
        
        response = client.post("/{fake.word()}s/", json={fake.word()}_data)
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == {fake.word()}_data["name"]
        assert data["description"] == {fake.word()}_data["description"]
        assert "id" in data
        assert "created_at" in data
    
    def test_get_{fake.word()}_success(self, db_session: Session):
        """Test successful {fake.word()} retrieval."""
        # Create test {fake.word()}
        {fake.word()} = {fake.word().title()}(
            name="{fake.name()}",
            description="{fake.text(max_nb_chars=100)}"
        )
        db_session.add({fake.word()})
        db_session.commit()
        db_session.refresh({fake.word()})
        
        response = client.get(f"/{fake.word()}s/{{{fake.word()}.id}}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str({fake.word()}.id)
        assert data["name"] == {fake.word()}.name
    
    def test_get_{fake.word()}_not_found(self):
        """Test {fake.word()} retrieval with non-existent ID."""
        non_existent_id = uuid.uuid4()
        
        response = client.get(f"/{fake.word()}s/{{non_existent_id}}")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
    
    def test_list_{fake.word()}s_pagination(self, db_session: Session):
        """Test {fake.word()} listing with pagination."""
        # Create test {fake.word()}s
        for i in range(15):
            {fake.word()} = {fake.word().title()}(
                name=f"Test {fake.word().title()} {{i}}",
                description=f"Description {{i}}"
            )
            db_session.add({fake.word()})
        db_session.commit()
        
        # Test pagination
        response = client.get("/{fake.word()}s/?skip=0&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 10
        
        response = client.get("/{fake.word()}s/?skip=10&limit=10")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 5
    
    def test_update_{fake.word()}_success(self, db_session: Session):
        """Test successful {fake.word()} update."""
        # Create test {fake.word()}
        {fake.word()} = {fake.word().title()}(
            name="{fake.name()}",
            description="{fake.text(max_nb_chars=100)}"
        )
        db_session.add({fake.word()})
        db_session.commit()
        db_session.refresh({fake.word()})
        
        update_data = {{
            "name": "Updated {fake.name()}",
            "description": "Updated description"
        }}
        
        response = client.put(f"/{fake.word()}s/{{{fake.word()}.id}}", json=update_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == update_data["name"]
        assert data["description"] == update_data["description"]
    
    def test_delete_{fake.word()}_success(self, db_session: Session):
        """Test successful {fake.word()} deletion."""
        # Create test {fake.word()}
        {fake.word()} = {fake.word().title()}(
            name="{fake.name()}",
            description="{fake.text(max_nb_chars=100)}"
        )
        db_session.add({fake.word()})
        db_session.commit()
        db_session.refresh({fake.word()})
        
        response = client.delete(f"/{fake.word()}s/{{{fake.word()}.id}}")
        
        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]
        
        # Verify deletion
        response = client.get(f"/{fake.word()}s/{{{fake.word()}.id}}")
        assert response.status_code == 404

@pytest.mark.asyncio
class Test{fake.word().title()}Async:
    """Async test cases for {fake.word()} operations."""
    
    async def test_async_{fake.word()}_operation(self):
        """Test async {fake.word()} operation."""
        with patch('app.core.database.get_db') as mock_get_db:
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db
            
            # Test async operation
            result = await some_async_function()
            assert result is not None
'''
    
    def _requirements_content(self) -> str:
        packages = [
            "fastapi==0.104.1",
            "uvicorn[standard]==0.24.0",
            "pydantic==2.5.0",
            "sqlalchemy==2.0.23",
            "psycopg2-binary==2.9.9",
            "alembic==1.13.0",
            "redis==5.0.1",
            "httpx==0.25.2",
            "python-multipart==0.0.6",
            "jinja2==3.1.2",
            "pytest==7.4.3",
            "pytest-asyncio==0.21.1",
            "pytest-cov==4.1.0"
        ]
        return "\n".join(packages)
    
    def _dockerfile_content(self) -> str:
        return f'''FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
'''
    
    def _readme_content(self) -> str:
        return f'''# {fake.catch_phrase()}

{fake.text(max_nb_chars=200)}

## Features

- {fake.bs()}
- {fake.bs()}
- {fake.bs()}
- {fake.bs()}

## Installation

```bash
# Clone the repository
git clone https://github.com/{fake.user_name()}/{fake.slug()}.git
cd {fake.slug()}

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env

# Run database migrations
alembic upgrade head

# Start the application
uvicorn app.main:app --reload
```

## Usage

The API will be available at `http://localhost:8000`

### API Documentation

- Interactive API docs: `http://localhost:8000/docs`
- ReDoc documentation: `http://localhost:8000/redoc`

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html
```

## License

MIT License - see LICENSE file for details.
'''
    
    def _react_component_content(self) -> str:
        return f'''import React, {{ useState, useEffect }} from 'react';
import axios from 'axios';

interface {fake.word().title()}Props {{
  id?: string;
  onUpdate?: () => void;
}}

interface {fake.word().title()}Data {{
  id: string;
  name: string;
  description: string;
  status: string;
  created_at: string;
}}

const {fake.word().title()}Component: React.FC<{fake.word().title()}Props> = ({{ id, onUpdate }}) => {{
  const [data, setData] = useState<{fake.word().title()}Data | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {{
    if (id) {{
      fetchData();
    }}
  }}, [id]);

  const fetchData = async () => {{
    setLoading(true);
    setError(null);
    
    try {{
      const response = await axios.get(`/api/v1/{fake.word()}s/${{id}}`);
      setData(response.data);
    }} catch (err) {{
      setError('Failed to fetch data');
      console.error('Error fetching data:', err);
    }} finally {{
      setLoading(false);
    }}
  }};

  const handleAction = async () => {{
    try {{
      await axios.post(`/api/v1/{fake.word()}s/${{id}}/action`);
      onUpdate?.();
    }} catch (err) {{
      setError('Action failed');
      console.error('Error performing action:', err);
    }}
  }};

  if (loading) {{
    return <div className="loading">Loading...</div>;
  }}

  if (error) {{
    return <div className="error">Error: {{error}}</div>;
  }}

  if (!data) {{
    return <div className="no-data">No data available</div>;
  }}

  return (
    <div className="{fake.word()}-component">
      <h2>{{data.name}}</h2>
      <p>{{data.description}}</p>
      <div className="metadata">
        <span className="status">Status: {{data.status}}</span>
        <span className="created">Created: {{new Date(data.created_at).toLocaleDateString()}}</span>
      </div>
      <button onClick={{handleAction}} className="action-button">
        Perform Action
      </button>
    </div>
  );
}};

export default {fake.word().title()}Component;
'''
    
    # Additional content generators
    def _api_client_content(self) -> str:
        return f'''import axios, {{ AxiosInstance, AxiosResponse }} from 'axios';

class ApiClient {{
  private client: AxiosInstance;

  constructor(baseURL: string = '/api/v1') {{
    this.client = axios.create({{
      baseURL,
      timeout: 10000,
      headers: {{
        'Content-Type': 'application/json',
      }},
    }});

    this.setupInterceptors();
  }}

  private setupInterceptors(): void {{
    this.client.interceptors.request.use(
      (config) => {{
        const token = localStorage.getItem('access_token');
        if (token) {{
          config.headers.Authorization = `Bearer ${{token}}`;
        }}
        return config;
      }},
      (error) => Promise.reject(error)
    );

    this.client.interceptors.response.use(
      (response) => response,
      (error) => {{
        if (error.response?.status === 401) {{
          localStorage.removeItem('access_token');
          window.location.href = '/login';
        }}
        return Promise.reject(error);
      }}
    );
  }}

  async get<T>(url: string): Promise<T> {{
    const response: AxiosResponse<T> = await this.client.get(url);
    return response.data;
  }}

  async post<T>(url: string, data?: any): Promise<T> {{
    const response: AxiosResponse<T> = await this.client.post(url, data);
    return response.data;
  }}

  async put<T>(url: string, data?: any): Promise<T> {{
    const response: AxiosResponse<T> = await this.client.put(url, data);
    return response.data;
  }}

  async delete<T>(url: string): Promise<T> {{
    const response: AxiosResponse<T> = await this.client.delete(url);
    return response.data;
  }}
}}

export const apiClient = new ApiClient();
'''
    
    def _package_json_content(self) -> str:
        return json.dumps({
            "name": fake.slug(),
            "version": "1.0.0",
            "description": fake.text(max_nb_chars=100),
            "main": "src/index.tsx",
            "scripts": {
                "start": "react-scripts start",
                "build": "react-scripts build",
                "test": "react-scripts test",
                "eject": "react-scripts eject"
            },
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0",
                "axios": "^1.6.0",
                "typescript": "^4.9.5"
            },
            "devDependencies": {
                "@types/react": "^18.2.0",
                "@types/react-dom": "^18.2.0",
                "react-scripts": "5.0.1"
            }
        }, indent=2)
    
    def _jupyter_content(self) -> str:
        return json.dumps({
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# {fake.sentence(nb_words=6)}\n",
                        f"\n{fake.text(max_nb_chars=200)}\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n",
                        f"\n# {fake.sentence()}\n",
                        f"data = pd.read_csv('{fake.file_name()}.csv')\n",
                        "data.head()\n"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }, indent=2)
    
    def _python_analysis_content(self) -> str:
        return f'''"""
Data analysis module for {fake.word()} processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Main data analysis class."""
    
    def __init__(self, data_source: str):
        self.data_source = data_source
        self.data: Optional[pd.DataFrame] = None
        self.results: Dict[str, Any] = {{}}
    
    def load_data(self) -> pd.DataFrame:
        """Load data from source."""
        try:
            self.data = pd.read_csv(self.data_source)
            logger.info(f"Loaded {{len(self.data)}} rows from {{self.data_source}}")
            return self.data
        except Exception as e:
            logger.error(f"Failed to load data: {{e}}")
            raise
    
    def analyze_distribution(self, column: str) -> Dict[str, Any]:
        """Analyze distribution of a column."""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        if column not in self.data.columns:
            raise ValueError(f"Column {{column}} not found")
        
        series = self.data[column]
        
        analysis = {{
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "quartiles": {{
                "q25": series.quantile(0.25),
                "q50": series.quantile(0.50),
                "q75": series.quantile(0.75)
            }},
            "missing_values": series.isnull().sum(),
            "unique_values": series.nunique()
        }}
        
        self.results[f"{{column}}_distribution"] = analysis
        return analysis
    
    def correlation_analysis(self) -> pd.DataFrame:
        """Perform correlation analysis."""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_columns].corr()
        
        self.results["correlation_matrix"] = correlation_matrix.to_dict()
        return correlation_matrix
    
    def detect_outliers(self, column: str, method: str = "iqr") -> List[int]:
        """Detect outliers in a column."""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        series = self.data[column]
        
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        elif method == "zscore":
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = series[z_scores > 3].index.tolist()
        else:
            raise ValueError(f"Unknown method: {{method}}")
        
        self.results[f"{{column}}_outliers_{{method}}"] = outliers
        return outliers
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        report = {{
            "dataset_info": {{
                "shape": self.data.shape,
                "columns": self.data.columns.tolist(),
                "dtypes": self.data.dtypes.to_dict(),
                "missing_values": self.data.isnull().sum().to_dict()
            }},
            "analysis_results": self.results,
            "summary": {{
                "total_rows": len(self.data),
                "total_columns": len(self.data.columns),
                "numeric_columns": len(self.data.select_dtypes(include=[np.number]).columns),
                "categorical_columns": len(self.data.select_dtypes(include=['object']).columns)
            }}
        }}
        
        return report
'''
    
    def _ml_model_content(self) -> str:
        return f'''"""
Machine learning model for {fake.word()} classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import logging

logger = logging.getLogger(__name__)

class {fake.word().title()}Classifier(BaseEstimator, ClassifierMixin):
    """Custom classifier for {fake.word()} prediction."""
    
    def __init__(
        self, 
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> '{fake.word().title()}Classifier':
        """Train the classifier."""
        logger.info(f"Training classifier with {{len(X)}} samples")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y_encoded)
        self.is_fitted = True
        
        logger.info("Classifier training completed")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        y_pred_encoded = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before accessing feature importance")
        
        importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        evaluation = {{
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "feature_importance": self.feature_importance(),
            "prediction_confidence": {{
                "mean": np.mean(np.max(y_proba, axis=1)),
                "std": np.std(np.max(y_proba, axis=1)),
                "min": np.min(np.max(y_proba, axis=1)),
                "max": np.max(np.max(y_proba, axis=1))
            }}
        }}
        
        return evaluation
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation."""
        logger.info(f"Performing {{cv}}-fold cross-validation")
        
        y_encoded = self.label_encoder.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)
        
        scores = cross_val_score(self.model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        
        cv_results = {{
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std(),
            "scores": scores.tolist()
        }}
        
        logger.info(f"Cross-validation completed: {{cv_results['mean_accuracy']:.4f}} ± {{cv_results['std_accuracy']:.4f}}")
        return cv_results
    
    def save_model(self, filepath: str) -> None:
        """Save trained model to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted classifier")
        
        model_data = {{
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'hyperparameters': {{
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'random_state': self.random_state
            }}
        }}
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {{filepath}}")
    
    @classmethod
    def load_model(cls, filepath: str) -> '{fake.word().title()}Classifier':
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        
        classifier = cls(
            n_estimators=model_data['hyperparameters']['n_estimators'],
            max_depth=model_data['hyperparameters']['max_depth'],
            random_state=model_data['hyperparameters']['random_state']
        )
        
        classifier.model = model_data['model']
        classifier.scaler = model_data['scaler']
        classifier.label_encoder = model_data['label_encoder']
        classifier.feature_names = model_data['feature_names']
        classifier.is_fitted = True
        
        logger.info(f"Model loaded from {{filepath}}")
        return classifier

def train_model(data_path: str, target_column: str) -> {fake.word().title()}Classifier:
    """Train a new {fake.word()} classifier."""
    # Load data
    data = pd.read_csv(data_path)
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train classifier
    classifier = {fake.word().title()}Classifier()
    classifier.fit(X_train, y_train)
    
    # Evaluate
    evaluation = classifier.evaluate(X_test, y_test)
    logger.info(f"Model accuracy: {{evaluation['accuracy']:.4f}}")
    
    return classifier
'''
    
    def _data_loader_content(self) -> str:
        return f'''"""
Data loading utilities for {fake.word()} processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import requests
from sqlalchemy import create_engine
import sqlite3

logger = logging.getLogger(__name__)

class DataLoader:
    """Unified data loading interface."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {{}}
        self.cache_dir = Path(self.config.get('cache_dir', './data_cache'))
        self.cache_dir.mkdir(exist_ok=True)
    
    def load_csv(
        self, 
        filepath: Union[str, Path], 
        **kwargs
    ) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            data = pd.read_csv(filepath, **kwargs)
            logger.info(f"Loaded {{len(data)}} rows from CSV: {{filepath}}")
            return data
        except Exception as e:
            logger.error(f"Failed to load CSV {{filepath}}: {{e}}")
            raise
    
    def load_from_database(
        self, 
        connection_string: str, 
        query: str
    ) -> pd.DataFrame:
        """Load data from database."""
        try:
            engine = create_engine(connection_string)
            data = pd.read_sql(query, engine)
            logger.info(f"Loaded {{len(data)}} rows from database")
            return data
        except Exception as e:
            logger.error(f"Failed to load from database: {{e}}")
            raise
    
    def load_from_api(
        self, 
        url: str, 
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Load data from REST API."""
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = pd.json_normalize(response.json())
            logger.info(f"Loaded {{len(data)}} rows from API: {{url}}")
            return data
        except Exception as e:
            logger.error(f"Failed to load from API {{url}}: {{e}}")
            raise
    
    def load_json(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """Load data from JSON file."""
        try:
            data = pd.read_json(filepath)
            logger.info(f"Loaded {{len(data)}} rows from JSON: {{filepath}}")
            return data
        except Exception as e:
            logger.error(f"Failed to load JSON {{filepath}}: {{e}}")
            raise
    
    def load_excel(
        self, 
        filepath: Union[str, Path], 
        sheet_name: Union[str, int] = 0,
        **kwargs
    ) -> pd.DataFrame:
        """Load data from Excel file."""
        try:
            data = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
            logger.info(f"Loaded {{len(data)}} rows from Excel: {{filepath}}")
            return data
        except Exception as e:
            logger.error(f"Failed to load Excel {{filepath}}: {{e}}")
            raise
    
    def save_to_cache(
        self, 
        data: pd.DataFrame, 
        cache_key: str,
        format: str = 'parquet'
    ) -> Path:
        """Save data to cache."""
        cache_file = self.cache_dir / f"{{cache_key}}.{{format}}"
        
        try:
            if format == 'parquet':
                data.to_parquet(cache_file, index=False)
            elif format == 'csv':
                data.to_csv(cache_file, index=False)
            elif format == 'json':
                data.to_json(cache_file, orient='records')
            else:
                raise ValueError(f"Unsupported cache format: {{format}}")
            
            logger.info(f"Cached data to {{cache_file}}")
            return cache_file
        except Exception as e:
            logger.error(f"Failed to cache data: {{e}}")
            raise
    
    def load_from_cache(
        self, 
        cache_key: str,
        format: str = 'parquet'
    ) -> Optional[pd.DataFrame]:
        """Load data from cache."""
        cache_file = self.cache_dir / f"{{cache_key}}.{{format}}"
        
        if not cache_file.exists():
            return None
        
        try:
            if format == 'parquet':
                data = pd.read_parquet(cache_file)
            elif format == 'csv':
                data = pd.read_csv(cache_file)
            elif format == 'json':
                data = pd.read_json(cache_file)
            else:
                raise ValueError(f"Unsupported cache format: {{format}}")
            
            logger.info(f"Loaded {{len(data)}} rows from cache: {{cache_file}}")
            return data
        except Exception as e:
            logger.error(f"Failed to load from cache: {{e}}")
            return None
    
    def validate_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate loaded data quality."""
        validation_report = {{
            "shape": data.shape,
            "columns": data.columns.tolist(),
            "dtypes": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "duplicate_rows": data.duplicated().sum(),
            "memory_usage": data.memory_usage(deep=True).sum(),
            "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": data.select_dtypes(include=['object']).columns.tolist(),
            "datetime_columns": data.select_dtypes(include=['datetime64']).columns.tolist()
        }}
        
        # Check for data quality issues
        issues = []
        
        if validation_report["duplicate_rows"] > 0:
            issues.append(f"Found {{validation_report['duplicate_rows']}} duplicate rows")
        
        missing_pct = (data.isnull().sum() / len(data)) * 100
        high_missing = missing_pct[missing_pct > 50].index.tolist()
        if high_missing:
            issues.append(f"Columns with >50% missing values: {{high_missing}}")
        
        validation_report["issues"] = issues
        validation_report["quality_score"] = self._calculate_quality_score(data)
        
        return validation_report
    
    def _calculate_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-100)."""
        score = 100.0
        
        # Penalize missing values
        missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
        score -= missing_pct
        
        # Penalize duplicate rows
        duplicate_pct = (data.duplicated().sum() / len(data)) * 100
        score -= duplicate_pct * 2  # Weight duplicates more heavily
        
        # Ensure score is within bounds
        return max(0.0, min(100.0, score))

def load_sample_data(dataset_name: str) -> pd.DataFrame:
    """Load sample datasets for testing."""
    sample_datasets = {{
        "{fake.word()}": {{
            "url": "https://raw.githubusercontent.com/example/data/{fake.word()}.csv",
            "description": "{fake.text(max_nb_chars=100)}"
        }},
        "synthetic": {{
            "generator": "synthetic",
            "rows": 1000,
            "columns": ["feature_1", "feature_2", "feature_3", "target"]
        }}
    }}
    
    if dataset_name == "synthetic":
        # Generate synthetic data
        np.random.seed(42)
        data = pd.DataFrame({{
            "feature_1": np.random.normal(0, 1, 1000),
            "feature_2": np.random.uniform(-1, 1, 1000),
            "feature_3": np.random.exponential(1, 1000),
            "target": np.random.choice(['A', 'B', 'C'], 1000)
        }})
        return data
    
    elif dataset_name in sample_datasets:
        loader = DataLoader()
        return loader.load_from_api(sample_datasets[dataset_name]["url"])
    
    else:
        raise ValueError(f"Unknown sample dataset: {{dataset_name}}")
'''
    
    # Utility methods for content analysis
    def _extract_functions(self, content: str, language: str) -> List[Dict[str, Any]]:
        """Extract function definitions from content."""
        functions = []
        
        if language == "python":
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if line.strip().startswith('def ') or line.strip().startswith('async def '):
                    func_name = line.split('(')[0].replace('def ', '').replace('async ', '').strip()
                    functions.append({
                        "name": func_name,
                        "line": i,
                        "complexity": random.randint(1, 5),
                        "async": "async def" in line
                    })
        
        return functions
    
    def _extract_imports(self, content: str, language: str) -> List[Dict[str, Any]]:
        """Extract import statements from content."""
        imports = []
        
        if language == "python":
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith('import ') or stripped.startswith('from '):
                    imports.append({
                        "line": i,
                        "statement": stripped,
                        "module": self._extract_module_name(stripped)
                    })
        
        return imports
    
    def _extract_module_name(self, import_statement: str) -> str:
        """Extract module name from import statement."""
        if import_statement.startswith('from '):
            return import_statement.split(' ')[1]
        elif import_statement.startswith('import '):
            return import_statement.split(' ')[1].split('.')[0]
        return ""
    
    def _calculate_complexity(self, content: str, language: str) -> int:
        """Calculate cyclomatic complexity."""
        if language == "python":
            # Simple heuristic based on control flow keywords
            complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with']
            complexity = 1  # Base complexity
            
            for keyword in complexity_keywords:
                complexity += content.count(f' {keyword} ') + content.count(f'\n{keyword} ')
            
            return min(complexity, 10)  # Cap at 10
        
        return random.randint(1, 5)
    
    def _generate_file_tags(self, relative_path: str, file_type: FileType, language: str) -> List[str]:
        """Generate relevant tags for a file."""
        tags = [language]
        
        if file_type == FileType.TEST:
            tags.append("test")
        elif file_type == FileType.DOCUMENTATION:
            tags.extend(["documentation", "docs"])
        elif file_type == FileType.CONFIGURATION:
            tags.extend(["config", "configuration"])
        
        # Add tags based on path
        path_parts = Path(relative_path).parts
        for part in path_parts:
            if part in ['api', 'models', 'tests', 'docs', 'config', 'utils', 'core']:
                tags.append(part)
        
        # Add framework-specific tags
        if 'fastapi' in relative_path.lower() or 'api' in relative_path.lower():
            tags.append("api")
        if 'model' in relative_path.lower():
            tags.append("model")
        if 'test' in relative_path.lower():
            tags.append("test")
        
        return list(set(tags))  # Remove duplicates


# Mock service classes for testing
class MockRedisClient:
    """Mock Redis client for testing."""
    
    def __init__(self):
        self.data = {}
        self.connected = True
    
    async def get(self, key: str) -> Optional[str]:
        return self.data.get(key)
    
    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self.data[key] = value
        return True
    
    async def delete(self, key: str) -> int:
        return 1 if self.data.pop(key, None) is not None else 0
    
    async def ping(self) -> bool:
        return self.connected
    
    def disconnect(self):
        self.connected = False

class MockEventPublisher:
    """Mock event publisher for testing."""
    
    def __init__(self):
        self.published_events = []
    
    async def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        self.published_events.append({
            "type": event_type,
            "data": data,
            "timestamp": datetime.utcnow()
        })
    
    def get_published_events(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if event_type:
            return [e for e in self.published_events if e["type"] == event_type]
        return self.published_events.copy()

class MockFileSystem:
    """Mock file system for testing."""
    
    def __init__(self):
        self.files = {}
        self.directories = set()
    
    def create_file(self, path: str, content: str = "") -> None:
        self.files[path] = content
        # Create parent directories
        parent = str(Path(path).parent)
        if parent != ".":
            self.directories.add(parent)
    
    def read_file(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(f"File not found: {path}")
        return self.files[path]
    
    def exists(self, path: str) -> bool:
        return path in self.files or path in self.directories
    
    def list_files(self, directory: str = ".") -> List[str]:
        return [f for f in self.files.keys() if f.startswith(directory)]


# Factory instance for easy use
test_data_factory = ProjectIndexTestDataFactory()


# Predefined test scenarios
def create_small_python_project() -> TestProjectData:
    """Create a small Python project for quick testing."""
    return test_data_factory.create_project_data(
        project_type="python_webapp",
        complexity="simple",
        include_dependencies=True,
        include_analysis=True
    )

def create_large_enterprise_project() -> TestProjectData:
    """Create a large enterprise project for stress testing."""
    return test_data_factory.create_project_data(
        project_type="python_webapp", 
        complexity="large",
        include_dependencies=True,
        include_analysis=True
    )

def create_data_science_project() -> TestProjectData:
    """Create a data science project with notebooks."""
    return test_data_factory.create_project_data(
        project_type="data_science",
        complexity="medium",
        include_dependencies=True,
        include_analysis=True
    )

def create_microservice_project() -> TestProjectData:
    """Create a microservice project."""
    return test_data_factory.create_project_data(
        project_type="microservice",
        complexity="medium", 
        include_dependencies=True,
        include_analysis=True
    )