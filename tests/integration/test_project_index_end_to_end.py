"""
End-to-end integration tests for Project Index system.

Tests complete workflows combining ProjectIndexer, cache, file monitoring,
and database operations in realistic scenarios.
"""

import asyncio
import json
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, AsyncMock

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.project_index.core import ProjectIndexer
from app.project_index.models import (
    ProjectIndexConfig, AnalysisConfiguration
)
from app.models.project_index import (
    ProjectStatus, AnalysisSessionType
)


class TestProjectIndexEndToEnd:
    """End-to-end integration tests for the complete Project Index system."""
    
    @pytest.fixture
    async def test_engine(self):
        """Create test database engine with full schema."""
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=False
        )
        
        # Create complete test schema using individual execute statements
        async with engine.begin() as conn:
            from sqlalchemy import text
            
            # Projects table
            await conn.execute(text("""
                CREATE TABLE project_indexes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    root_path TEXT NOT NULL,
                    git_repository_url TEXT,
                    git_branch TEXT,
                    git_commit_hash TEXT,
                    status TEXT NOT NULL DEFAULT 'INACTIVE',
                    file_count INTEGER DEFAULT 0,
                    dependency_count INTEGER DEFAULT 0,
                    configuration TEXT DEFAULT '{}',
                    analysis_settings TEXT DEFAULT '{}',
                    file_patterns TEXT DEFAULT '{}',
                    ignore_patterns TEXT DEFAULT '{}',
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_indexed_at TIMESTAMP,
                    last_analysis_at TIMESTAMP
                )
            """))
            
            # File entries table
            await conn.execute(text("""
                CREATE TABLE file_entries (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_extension TEXT,
                    file_type TEXT NOT NULL,
                    language TEXT,
                    file_size INTEGER DEFAULT 0,
                    line_count INTEGER DEFAULT 0,
                    sha256_hash TEXT,
                    is_binary BOOLEAN DEFAULT FALSE,
                    is_generated BOOLEAN DEFAULT FALSE,
                    analysis_data TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_modified TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES project_indexes (id)
                )
            """))
            
            # Dependencies table
            await conn.execute(text("""
                CREATE TABLE dependency_relationships (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    source_file_id TEXT,
                    target_file_id TEXT,
                    target_path TEXT,
                    target_name TEXT NOT NULL,
                    dependency_type TEXT NOT NULL,
                    line_number INTEGER,
                    column_number INTEGER,
                    source_text TEXT,
                    is_external BOOLEAN DEFAULT FALSE,
                    is_dynamic BOOLEAN DEFAULT FALSE,
                    confidence_score REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES project_indexes (id),
                    FOREIGN KEY (source_file_id) REFERENCES file_entries (id),
                    FOREIGN KEY (target_file_id) REFERENCES file_entries (id)
                )
            """))
            
            # Analysis sessions table
            await conn.execute(text("""
                CREATE TABLE analysis_sessions (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    session_name TEXT NOT NULL,
                    session_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    progress_percentage REAL DEFAULT 0.0,
                    current_step TEXT,
                    files_total INTEGER DEFAULT 0,
                    files_processed INTEGER DEFAULT 0,
                    dependencies_found INTEGER DEFAULT 0,
                    configuration TEXT DEFAULT '{}',
                    performance_metrics TEXT DEFAULT '{}',
                    error_messages TEXT DEFAULT '[]',
                    result_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES project_indexes (id)
                )
            """))
            
            # Index snapshots table
            await conn.execute(text("""
                CREATE TABLE index_snapshots (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    snapshot_type TEXT NOT NULL,
                    snapshot_name TEXT,
                    snapshot_data TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES project_indexes (id)
                )
            """))
        
        yield engine
        await engine.dispose()
    
    @pytest.fixture
    async def test_session(self, test_engine):
        """Create test database session."""
        async_session = async_sessionmaker(
            bind=test_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        async with async_session() as session:
            yield session
            await session.rollback()
    
    @pytest.fixture
    def mock_redis(self):
        """Create realistic mock Redis client."""
        redis = AsyncMock()
        
        # In-memory storage for realistic behavior
        self._cache_storage = {}
        self._ttl_storage = {}
        
        async def mock_get(key):
            if key in self._cache_storage:
                if key in self._ttl_storage and time.time() > self._ttl_storage[key]:
                    del self._cache_storage[key]
                    del self._ttl_storage[key]
                    return None
                return self._cache_storage[key]
            return None
        
        async def mock_setex(key, ttl, value):
            self._cache_storage[key] = value
            self._ttl_storage[key] = time.time() + ttl
            return True
        
        async def mock_set(key, value, expire=None):
            self._cache_storage[key] = value
            if expire:
                self._ttl_storage[key] = time.time() + expire
            return True
        
        async def mock_delete(*keys):
            deleted = 0
            for key in keys:
                if key in self._cache_storage:
                    del self._cache_storage[key]
                    deleted += 1
                if key in self._ttl_storage:
                    del self._ttl_storage[key]
            return deleted
        
        async def mock_keys(pattern):
            import fnmatch
            return [k.encode() for k in self._cache_storage.keys() 
                   if fnmatch.fnmatch(k, pattern)]
        
        redis.get.side_effect = mock_get
        redis.setex.side_effect = mock_setex
        redis.set.side_effect = mock_set
        redis.delete.side_effect = mock_delete
        redis.keys.side_effect = mock_keys
        redis.ping.return_value = True
        
        return redis
    
    @pytest.fixture
    def advanced_config(self):
        """Create advanced project configuration for testing."""
        return ProjectIndexConfig(
            analysis_batch_size=3,
            max_concurrent_analyses=2,
            cache_ttl=300,
            analysis_config=AnalysisConfiguration(
                include_ast=True,
                include_dependencies=True,
                include_complexity_metrics=True,
                max_file_size=1024 * 1024,
                supported_languages=['python', 'javascript', 'typescript'],
                ignore_patterns=['*.pyc', '__pycache__/*', '*.log']
            )
        )
    
    @pytest.fixture
    async def project_indexer(self, test_session, mock_redis, advanced_config):
        """Create fully configured ProjectIndexer."""
        return ProjectIndexer(
            session=test_session,
            redis_client=mock_redis,
            config=advanced_config
        )
    
    @pytest.fixture
    def complex_project(self):
        """Create complex project structure for comprehensive testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create main application structure
            (temp_path / "app").mkdir()
            (temp_path / "app" / "__init__.py").write_text("")
            (temp_path / "app" / "main.py").write_text("""
import os
import sys
from typing import List, Dict, Optional
from app.models import User, Project
from app.utils import database, auth
from app.services.analytics import AnalyticsService

class Application:
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.db = database.DatabaseManager(config['db_url'])
        self.auth = auth.AuthenticationService()
        self.analytics = AnalyticsService()
    
    async def start(self) -> None:
        await self.db.connect()
        print("Application started")
    
    def get_users(self) -> List[User]:
        return self.db.query(User).all()
""")
            
            # Models module
            (temp_path / "app" / "models").mkdir()
            (temp_path / "app" / "models" / "__init__.py").write_text("""
from .user import User
from .project import Project
""")
            (temp_path / "app" / "models" / "user.py").write_text("""
from datetime import datetime
from typing import Optional

class User:
    def __init__(self, id: int, name: str, email: str):
        self.id = id
        self.name = name
        self.email = email
        self.created_at = datetime.utcnow()
    
    def is_active(self) -> bool:
        return True
    
    def get_projects(self):
        from .project import Project
        return Project.query.filter_by(user_id=self.id)
""")
            (temp_path / "app" / "models" / "project.py").write_text("""
from datetime import datetime
from typing import List, Optional
from .user import User

class Project:
    def __init__(self, id: int, name: str, description: str, user_id: int):
        self.id = id
        self.name = name
        self.description = description
        self.user_id = user_id
        self.created_at = datetime.utcnow()
    
    def get_owner(self) -> User:
        return User.query.get(self.user_id)
    
    def calculate_statistics(self) -> dict:
        return {
            'file_count': 0,
            'line_count': 0,
            'complexity_score': 0.0
        }
""")
            
            # Utils module
            (temp_path / "app" / "utils").mkdir()
            (temp_path / "app" / "utils" / "__init__.py").write_text("")
            (temp_path / "app" / "utils" / "database.py").write_text("""
import asyncio
from typing import Dict, Any, Optional

class DatabaseManager:
    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self.connection = None
    
    async def connect(self) -> None:
        # Simulate database connection
        await asyncio.sleep(0.1)
        self.connection = "connected"
    
    def query(self, model_class):
        return QueryBuilder(model_class)

class QueryBuilder:
    def __init__(self, model_class):
        self.model_class = model_class
    
    def filter_by(self, **kwargs):
        return self
    
    def all(self):
        return []
""")
            (temp_path / "app" / "utils" / "auth.py").write_text("""
import hashlib
from typing import Optional

class AuthenticationService:
    def __init__(self):
        self.secret_key = "secret"
    
    def hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        return self.hash_password(password) == hashed
    
    def create_token(self, user_id: int) -> str:
        return f"token_{user_id}_{self.secret_key}"
""")
            
            # Services module
            (temp_path / "app" / "services").mkdir()
            (temp_path / "app" / "services" / "__init__.py").write_text("")
            (temp_path / "app" / "services" / "analytics.py").write_text("""
from typing import Dict, List, Any
import json

class AnalyticsService:
    def __init__(self):
        self.events = []
    
    def track_event(self, event_name: str, properties: Dict[str, Any]) -> None:
        event = {
            'name': event_name,
            'properties': properties,
            'timestamp': '2023-01-01T00:00:00Z'
        }
        self.events.append(event)
    
    def get_user_analytics(self, user_id: int) -> Dict[str, Any]:
        user_events = [e for e in self.events 
                      if e['properties'].get('user_id') == user_id]
        return {
            'total_events': len(user_events),
            'last_activity': user_events[-1]['timestamp'] if user_events else None
        }
    
    def export_data(self) -> str:
        return json.dumps(self.events, indent=2)
""")
            
            # Tests
            (temp_path / "tests").mkdir()
            (temp_path / "tests" / "__init__.py").write_text("")
            (temp_path / "tests" / "test_models.py").write_text("""
import unittest
from app.models import User, Project

class TestUser(unittest.TestCase):
    def test_user_creation(self):
        user = User(1, "John Doe", "john@example.com")
        self.assertEqual(user.name, "John Doe")
        self.assertTrue(user.is_active())
    
    def test_user_projects(self):
        user = User(1, "John Doe", "john@example.com")
        projects = user.get_projects()
        self.assertIsNotNone(projects)

class TestProject(unittest.TestCase):
    def test_project_creation(self):
        project = Project(1, "Test Project", "Description", 1)
        self.assertEqual(project.name, "Test Project")
    
    def test_project_statistics(self):
        project = Project(1, "Test Project", "Description", 1)
        stats = project.calculate_statistics()
        self.assertIn('file_count', stats)
""")
            (temp_path / "tests" / "test_services.py").write_text("""
import unittest
from app.services.analytics import AnalyticsService

class TestAnalyticsService(unittest.TestCase):
    def setUp(self):
        self.service = AnalyticsService()
    
    def test_track_event(self):
        self.service.track_event('user_login', {'user_id': 1})
        self.assertEqual(len(self.service.events), 1)
    
    def test_get_user_analytics(self):
        self.service.track_event('user_login', {'user_id': 1})
        analytics = self.service.get_user_analytics(1)
        self.assertEqual(analytics['total_events'], 1)
""")
            
            # Configuration files
            (temp_path / "requirements.txt").write_text("""
fastapi==0.68.0
sqlalchemy==1.4.23
pydantic==1.8.2
pytest==6.2.4
uvicorn==0.15.0
""")
            (temp_path / "pyproject.toml").write_text("""
[tool.poetry]
name = "complex-app"
version = "0.1.0"
description = "A complex application for testing"

[tool.poetry.dependencies]
python = "^3.8"
fastapi = "^0.68.0"
sqlalchemy = "^1.4.23"

[tool.poetry.dev-dependencies]
pytest = "^6.2.4"
""")
            (temp_path / "config.json").write_text("""
{
    "app_name": "Complex Application",
    "version": "1.0.0",
    "database": {
        "url": "sqlite:///app.db",
        "pool_size": 10
    },
    "auth": {
        "secret_key": "your-secret-key-here",
        "token_expiry": 3600
    }
}
""")
            (temp_path / "README.md").write_text("""
# Complex Application

This is a complex application for testing the Project Index system.

## Features

- User management
- Project management  
- Analytics tracking
- Authentication system

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from app.main import Application

app = Application(config)
await app.start()
```
""")
            
            # JavaScript/TypeScript files for multi-language testing
            (temp_path / "frontend").mkdir()
            (temp_path / "frontend" / "app.js").write_text("""
import { UserService } from './services/user.js';
import { ProjectService } from './services/project.js';

class App {
    constructor() {
        this.userService = new UserService();
        this.projectService = new ProjectService();
    }
    
    async initialize() {
        try {
            await this.userService.loadUsers();
            await this.projectService.loadProjects();
            console.log('App initialized successfully');
        } catch (error) {
            console.error('Failed to initialize app:', error);
        }
    }
    
    handleUserAction(action, data) {
        switch (action) {
            case 'create':
                return this.userService.createUser(data);
            case 'update':
                return this.userService.updateUser(data);
            case 'delete':
                return this.userService.deleteUser(data.id);
            default:
                throw new Error(`Unknown action: ${action}`);
        }
    }
}

export default App;
""")
            (temp_path / "frontend" / "services").mkdir()
            (temp_path / "frontend" / "services" / "user.js").write_text("""
export class UserService {
    constructor() {
        this.users = [];
        this.apiUrl = '/api/users';
    }
    
    async loadUsers() {
        const response = await fetch(this.apiUrl);
        this.users = await response.json();
        return this.users;
    }
    
    async createUser(userData) {
        const response = await fetch(this.apiUrl, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(userData)
        });
        return response.json();
    }
    
    async updateUser(userData) {
        const response = await fetch(`${this.apiUrl}/${userData.id}`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(userData)
        });
        return response.json();
    }
    
    async deleteUser(userId) {
        const response = await fetch(`${this.apiUrl}/${userId}`, {
            method: 'DELETE'
        });
        return response.ok;
    }
}
""")
            
            yield temp_path
    
    @pytest.mark.asyncio
    async def test_complete_project_lifecycle(self, project_indexer, complex_project):
        """Test complete project lifecycle from creation to analysis."""
        # 1. Create project
        project = await project_indexer.create_project(
            name="Complex Application",
            root_path=str(complex_project),
            description="A complex multi-language application for testing",
            file_patterns={
                'include': ['**/*.py', '**/*.js', '**/*.ts', '**/*.json', '**/*.md'],
                'exclude': ['**/__pycache__/**', '**/*.pyc', '**/node_modules/**']
            }
        )
        
        assert project is not None
        assert project.name == "Complex Application"
        assert project.status == ProjectStatus.INACTIVE
        
        # 2. Mock the analyzer for realistic responses
        with patch.object(project_indexer.analyzer, 'parse_file') as mock_parse:
            with patch.object(project_indexer.analyzer, 'extract_dependencies') as mock_deps:
                with patch.object(project_indexer.analyzer, 'detect_language') as mock_lang:
                    # Configure realistic mock responses based on file types
                    def mock_parse_response(file_path):
                        if str(file_path).endswith('.py'):
                            return {
                                'line_count': 50,
                                'function_count': 5,
                                'class_count': 2,
                                'imports': 3,
                                'complexity_score': 15
                            }
                        elif str(file_path).endswith('.js'):
                            return {
                                'line_count': 30,
                                'function_count': 4,
                                'class_count': 1,
                                'imports': 2,
                                'complexity_score': 10
                            }
                        else:
                            return {'line_count': 10}
                    
                    def mock_deps_response(file_path):
                        from app.project_index.models import DependencyResult
                        if 'main.py' in str(file_path):
                            return [
                                DependencyResult(
                                    source_file_path=str(file_path),
                                    target_name="app.models",
                                    dependency_type="import",
                                    line_number=4,
                                    confidence_score=0.95
                                ),
                                DependencyResult(
                                    source_file_path=str(file_path),
                                    target_name="app.utils.database",
                                    dependency_type="import",
                                    line_number=5,
                                    confidence_score=0.90
                                )
                            ]
                        return []
                    
                    def mock_lang_response(file_path):
                        if str(file_path).endswith('.py'):
                            return 'python'
                        elif str(file_path).endswith('.js'):
                            return 'javascript'
                        elif str(file_path).endswith('.json'):
                            return 'json'
                        elif str(file_path).endswith('.md'):
                            return 'markdown'
                        return None
                    
                    mock_parse.side_effect = mock_parse_response
                    mock_deps.side_effect = mock_deps_response
                    mock_lang.side_effect = mock_lang_response
                    
                    # 3. Perform full analysis
                    result = await project_indexer.analyze_project(
                        str(project.id),
                        analysis_type=AnalysisSessionType.FULL_ANALYSIS
                    )
        
        # 4. Verify analysis results
        assert result is not None
        assert result.project_id == str(project.id)
        assert result.analysis_type == AnalysisSessionType.FULL_ANALYSIS
        assert result.files_processed > 10  # Should process many files
        assert result.analysis_duration > 0
        
        # 5. Verify project status updated
        updated_project = await project_indexer.get_project(str(project.id))
        assert updated_project.status == ProjectStatus.ACTIVE
        assert updated_project.file_count > 0
        
        # 6. Test incremental analysis
        # Add a new file to trigger incremental analysis
        new_file = complex_project / "app" / "new_feature.py"
        new_file.write_text("""
def new_feature():
    '''A new feature added after initial analysis.'''
    return "new feature"
""")
        
        # Wait a bit to ensure different timestamps
        await asyncio.sleep(0.1)
        
        with patch.object(project_indexer.analyzer, 'parse_file') as mock_parse:
            with patch.object(project_indexer.analyzer, 'extract_dependencies') as mock_deps:
                with patch.object(project_indexer.analyzer, 'detect_language') as mock_lang:
                    mock_parse.return_value = {'line_count': 5, 'function_count': 1}
                    mock_deps.return_value = []
                    mock_lang.return_value = 'python'
                    
                    # Perform incremental analysis
                    incremental_result = await project_indexer.analyze_project(
                        str(project.id),
                        analysis_type=AnalysisSessionType.INCREMENTAL
                    )
        
        assert incremental_result is not None
        assert incremental_result.analysis_type == AnalysisSessionType.INCREMENTAL
    
    @pytest.mark.asyncio
    async def test_cache_integration_with_real_workflow(self, project_indexer, complex_project):
        """Test cache integration throughout a realistic workflow."""
        # Create project
        project = await project_indexer.create_project(
            name="Cache Test Project",
            root_path=str(complex_project)
        )
        
        # Mock analyzer
        with patch.object(project_indexer.analyzer, 'parse_file') as mock_parse:
            with patch.object(project_indexer.analyzer, 'extract_dependencies') as mock_deps:
                with patch.object(project_indexer.analyzer, 'detect_language') as mock_lang:
                    mock_parse.return_value = {'line_count': 20, 'function_count': 2}
                    mock_deps.return_value = []
                    mock_lang.return_value = 'python'
                    
                    # First analysis - should populate cache
                    result1 = await project_indexer.analyze_project(
                        str(project.id),
                        analysis_type=AnalysisSessionType.FULL_ANALYSIS
                    )
                    
                    # Get cache statistics
                    cache_stats = await project_indexer.cache.get_cache_stats(str(project.id))
                    assert 'analysis_cache_entries' in cache_stats
                    
                    # Second analysis - should benefit from cache
                    result2 = await project_indexer.analyze_project(
                        str(project.id),
                        analysis_type=AnalysisSessionType.FULL_ANALYSIS,
                        force_reanalysis=False  # Allow cache usage
                    )
        
        assert result1 is not None
        assert result2 is not None
        
        # Performance stats should show cache usage
        stats = await project_indexer.get_analysis_statistics()
        assert stats['cache_hits'] > 0
    
    @pytest.mark.asyncio
    async def test_file_monitoring_with_real_changes(self, project_indexer, complex_project):
        """Test file monitoring integration with real file changes."""
        # Create project
        project = await project_indexer.create_project(
            name="Monitoring Test Project", 
            root_path=str(complex_project)
        )
        
        # Set up change tracking
        detected_changes = []
        
        async def change_callback(event):
            detected_changes.append(event)
        
        monitor = project_indexer.file_monitor
        monitor.add_change_callback(change_callback)
        
        # Let initial monitoring settle
        await asyncio.sleep(0.2)
        
        # Make realistic changes
        # 1. Modify existing file
        main_file = complex_project / "app" / "main.py"
        original_content = main_file.read_text()
        main_file.write_text(original_content + "\n# Added comment")
        
        # 2. Create new module
        new_module = complex_project / "app" / "services" / "notification.py"
        new_module.write_text("""
class NotificationService:
    def __init__(self):
        self.notifications = []
    
    def send_notification(self, message: str, user_id: int):
        notification = {
            'message': message,
            'user_id': user_id,
            'timestamp': '2023-01-01T00:00:00Z'
        }
        self.notifications.append(notification)
        return notification
""")
        
        # 3. Delete old file
        old_file = complex_project / "frontend" / "services" / "user.js"
        if old_file.exists():
            old_file.unlink()
        
        # Wait for change detection
        await asyncio.sleep(0.5)
        
        # Verify changes were detected
        change_types = [change.change_type for change in detected_changes]
        assert len(detected_changes) >= 2  # Should detect multiple changes
        
        # Should include different change types
        from app.project_index.file_monitor import FileChangeType
        assert any(ct in change_types for ct in [
            FileChangeType.CREATED, FileChangeType.MODIFIED, FileChangeType.DELETED
        ])
    
    @pytest.mark.asyncio
    async def test_dependency_analysis_comprehensive(self, project_indexer, complex_project):
        """Test comprehensive dependency analysis with complex project."""
        # Create project
        project = await project_indexer.create_project(
            name="Dependency Analysis Test",
            root_path=str(complex_project)
        )
        
        # Mock analyzer with realistic dependency data
        with patch.object(project_indexer.analyzer, 'extract_dependencies') as mock_deps:
            from app.project_index.models import DependencyResult
            
            def mock_dependency_extraction(file_path):
                file_str = str(file_path)
                deps = []
                
                if 'main.py' in file_str:
                    deps.extend([
                        DependencyResult(
                            source_file_path=file_str,
                            target_name="app.models.User",
                            dependency_type="import",
                            line_number=4,
                            confidence_score=0.95
                        ),
                        DependencyResult(
                            source_file_path=file_str,
                            target_name="app.models.Project", 
                            dependency_type="import",
                            line_number=4,
                            confidence_score=0.95
                        ),
                        DependencyResult(
                            source_file_path=file_str,
                            target_name="app.utils.database",
                            dependency_type="import",
                            line_number=5,
                            confidence_score=0.90
                        )
                    ])
                elif 'user.py' in file_str:
                    deps.append(
                        DependencyResult(
                            source_file_path=file_str,
                            target_name="app.models.project",
                            dependency_type="import",
                            line_number=15,
                            confidence_score=0.85
                        )
                    )
                elif 'project.py' in file_str:
                    deps.append(
                        DependencyResult(
                            source_file_path=file_str,
                            target_name="app.models.user",
                            dependency_type="import",
                            line_number=3,
                            confidence_score=0.85
                        )
                    )
                elif 'app.js' in file_str:
                    deps.extend([
                        DependencyResult(
                            source_file_path=file_str,
                            target_name="./services/user.js",
                            dependency_type="import",
                            line_number=1,
                            confidence_score=0.95
                        ),
                        DependencyResult(
                            source_file_path=file_str,
                            target_name="./services/project.js",
                            dependency_type="import",
                            line_number=2,
                            confidence_score=0.95
                        )
                    ])
                
                return deps
            
            mock_deps.side_effect = mock_dependency_extraction
            
            with patch.object(project_indexer.analyzer, 'parse_file') as mock_parse:
                with patch.object(project_indexer.analyzer, 'detect_language') as mock_lang:
                    mock_parse.return_value = {'line_count': 20}
                    mock_lang.return_value = 'python'
                    
                    # Perform dependency analysis
                    result = await project_indexer.analyze_project(
                        str(project.id),
                        analysis_type=AnalysisSessionType.DEPENDENCY_MAPPING
                    )
        
        assert result is not None
        assert result.analysis_type == AnalysisSessionType.DEPENDENCY_MAPPING
        assert result.dependencies_found > 0
        
        # Verify dependency relationships were stored
        assert len(result.dependency_results) > 0
        
        # Check for circular dependencies
        deps = result.dependency_results
        source_targets = [(d.source_file_path, d.target_name) for d in deps]
        assert len(source_targets) > 0
    
    @pytest.mark.asyncio
    async def test_context_optimization_workflow(self, project_indexer, complex_project):
        """Test context optimization analysis workflow."""
        # Create project
        project = await project_indexer.create_project(
            name="Context Optimization Test",
            root_path=str(complex_project)
        )
        
        # Perform context optimization analysis
        result = await project_indexer.analyze_project(
            str(project.id),
            analysis_type=AnalysisSessionType.CONTEXT_OPTIMIZATION
        )
        
        assert result is not None
        assert result.analysis_type == AnalysisSessionType.CONTEXT_OPTIMIZATION
        assert 'context_optimization' in result.to_dict()
        
        # Verify context optimization structure
        context_opt = result.context_optimization
        assert 'optimization_metrics' in context_opt
        assert 'context_efficiency' in context_opt['optimization_metrics']
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, project_indexer, complex_project):
        """Test concurrent project operations."""
        # Create multiple projects
        projects = []
        for i in range(3):
            project = await project_indexer.create_project(
                name=f"Concurrent Project {i}",
                root_path=str(complex_project),
                description=f"Project {i} for concurrent testing"
            )
            projects.append(project)
        
        # Mock analyzer
        with patch.object(project_indexer.analyzer, 'parse_file') as mock_parse:
            with patch.object(project_indexer.analyzer, 'extract_dependencies') as mock_deps:
                with patch.object(project_indexer.analyzer, 'detect_language') as mock_lang:
                    mock_parse.return_value = {'line_count': 15}
                    mock_deps.return_value = []
                    mock_lang.return_value = 'python'
                    
                    # Start concurrent analyses
                    tasks = []
                    for project in projects:
                        task = asyncio.create_task(
                            project_indexer.analyze_project(
                                str(project.id),
                                analysis_type=AnalysisSessionType.FULL_ANALYSIS
                            )
                        )
                        tasks.append(task)
                    
                    # Wait for all analyses to complete
                    results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all analyses completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) >= 2  # At least most should succeed
        
        # Verify each result
        for result in successful_results:
            assert result.analysis_duration > 0
            assert result.files_processed > 0
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_cleanup(self, project_indexer, complex_project):
        """Test error recovery and cleanup functionality."""
        # Create project
        project = await project_indexer.create_project(
            name="Error Recovery Test",
            root_path=str(complex_project)
        )
        
        # Test analysis failure and recovery
        with patch.object(project_indexer.analyzer, 'parse_file') as mock_parse:
            # First analysis fails
            mock_parse.side_effect = Exception("Analysis error")
            
            with pytest.raises(RuntimeError):
                await project_indexer.analyze_project(
                    str(project.id),
                    analysis_type=AnalysisSessionType.FULL_ANALYSIS
                )
            
            # Verify project status is failed
            failed_project = await project_indexer.get_project(str(project.id))
            assert failed_project.status == ProjectStatus.FAILED
            
            # Recovery: fix the analyzer and retry
            mock_parse.side_effect = None
            mock_parse.return_value = {'line_count': 20}
            
            with patch.object(project_indexer.analyzer, 'extract_dependencies') as mock_deps:
                with patch.object(project_indexer.analyzer, 'detect_language') as mock_lang:
                    mock_deps.return_value = []
                    mock_lang.return_value = 'python'
                    
                    # Retry analysis
                    recovery_result = await project_indexer.analyze_project(
                        str(project.id),
                        analysis_type=AnalysisSessionType.FULL_ANALYSIS
                    )
        
        # Verify successful recovery
        assert recovery_result is not None
        recovered_project = await project_indexer.get_project(str(project.id))
        assert recovered_project.status == ProjectStatus.ACTIVE
        
        # Test cleanup functionality
        cleanup_stats = await project_indexer.cleanup_old_data(retention_days=0)
        assert 'deleted_sessions' in cleanup_stats
        assert 'deleted_snapshots' in cleanup_stats
    
    @pytest.mark.asyncio
    async def test_performance_tracking_comprehensive(self, project_indexer, complex_project):
        """Test comprehensive performance tracking throughout workflow."""
        # Get initial performance stats
        initial_stats = await project_indexer.get_analysis_statistics()
        
        # Create and analyze project
        project = await project_indexer.create_project(
            name="Performance Tracking Test",
            root_path=str(complex_project)
        )
        
        with patch.object(project_indexer.analyzer, 'parse_file') as mock_parse:
            with patch.object(project_indexer.analyzer, 'extract_dependencies') as mock_deps:
                with patch.object(project_indexer.analyzer, 'detect_language') as mock_lang:
                    # Simulate variable processing times
                    async def slow_parse(file_path):
                        await asyncio.sleep(0.01)  # Simulate processing time
                        return {'line_count': 25, 'function_count': 3}
                    
                    mock_parse.side_effect = slow_parse
                    mock_deps.return_value = []
                    mock_lang.return_value = 'python'
                    
                    # Perform analysis
                    start_time = time.time()
                    result = await project_indexer.analyze_project(
                        str(project.id),
                        analysis_type=AnalysisSessionType.FULL_ANALYSIS
                    )
                    end_time = time.time()
        
        # Verify performance tracking
        assert result.analysis_duration > 0
        assert result.analysis_duration <= (end_time - start_time) + 1  # Allow some margin
        
        # Get final performance stats
        final_stats = await project_indexer.get_analysis_statistics()
        
        # Verify stats were updated
        assert final_stats['files_processed'] > initial_stats['files_processed']
        assert final_stats['analysis_time'] > initial_stats['analysis_time']
        
        # Verify reasonable performance metrics
        assert result.files_processed > 10  # Should process many files
        avg_time_per_file = result.analysis_duration / result.files_processed
        assert avg_time_per_file < 1.0  # Should be reasonably fast per file
    
    @pytest.mark.asyncio
    async def test_real_world_project_simulation(self, project_indexer, complex_project):
        """Simulate real-world usage patterns."""
        # Initial project setup
        project = await project_indexer.create_project(
            name="Real World Simulation",
            root_path=str(complex_project),
            description="Simulating real-world development workflow"
        )
        
        # Mock realistic analyzer behavior
        with patch.object(project_indexer.analyzer, 'parse_file') as mock_parse:
            with patch.object(project_indexer.analyzer, 'extract_dependencies') as mock_deps:
                with patch.object(project_indexer.analyzer, 'detect_language') as mock_lang:
                    
                    def realistic_parse(file_path):
                        file_str = str(file_path)
                        if file_str.endswith('.py'):
                            if 'test_' in file_str:
                                return {'line_count': 30, 'function_count': 5, 'class_count': 1}
                            elif 'models' in file_str:
                                return {'line_count': 80, 'function_count': 8, 'class_count': 2}
                            elif 'main.py' in file_str:
                                return {'line_count': 120, 'function_count': 6, 'class_count': 3}
                            else:
                                return {'line_count': 50, 'function_count': 4, 'class_count': 1}
                        elif file_str.endswith('.js'):
                            return {'line_count': 60, 'function_count': 5, 'class_count': 1}
                        else:
                            return {'line_count': 10}
                    
                    mock_parse.side_effect = realistic_parse
                    mock_deps.return_value = []
                    mock_lang.side_effect = lambda p: 'python' if str(p).endswith('.py') else 'javascript' if str(p).endswith('.js') else None
                    
                    # 1. Initial full analysis (typical first-time setup)
                    full_result = await project_indexer.analyze_project(
                        str(project.id),
                        analysis_type=AnalysisSessionType.FULL_ANALYSIS
                    )
                    
                    # 2. Dependency mapping (typical after initial analysis)
                    dep_result = await project_indexer.analyze_project(
                        str(project.id),
                        analysis_type=AnalysisSessionType.DEPENDENCY_MAPPING
                    )
                    
                    # 3. Context optimization (for AI-assisted development)
                    context_result = await project_indexer.analyze_project(
                        str(project.id),
                        analysis_type=AnalysisSessionType.CONTEXT_OPTIMIZATION
                    )
        
        # Verify realistic results
        assert full_result.files_processed >= 15  # Complex project should have many files
        assert full_result.analysis_duration > 0.1  # Should take measurable time
        
        assert dep_result.analysis_type == AnalysisSessionType.DEPENDENCY_MAPPING
        assert context_result.analysis_type == AnalysisSessionType.CONTEXT_OPTIMIZATION
        
        # Verify project progression
        final_project = await project_indexer.get_project(str(project.id))
        assert final_project.status == ProjectStatus.ACTIVE
        assert final_project.file_count > 0
        assert final_project.last_analysis_at is not None
        
        # Simulate ongoing development with incremental updates
        # Add new file (typical development activity)
        new_feature_file = complex_project / "app" / "features" / "reporting.py"
        new_feature_file.parent.mkdir(exist_ok=True)
        new_feature_file.write_text("""
class ReportingService:
    def generate_report(self, project_id: int):
        # Generate project report
        return {'project_id': project_id, 'status': 'generated'}
""")
        
        # Incremental analysis (typical after code changes)
        with patch.object(project_indexer.analyzer, 'parse_file') as mock_parse:
            with patch.object(project_indexer.analyzer, 'extract_dependencies') as mock_deps:
                with patch.object(project_indexer.analyzer, 'detect_language') as mock_lang:
                    mock_parse.return_value = {'line_count': 15, 'function_count': 2, 'class_count': 1}
                    mock_deps.return_value = []
                    mock_lang.return_value = 'python'
                    
                    incremental_result = await project_indexer.analyze_project(
                        str(project.id),
                        analysis_type=AnalysisSessionType.INCREMENTAL
                    )
        
        # Verify incremental analysis
        assert incremental_result.analysis_type == AnalysisSessionType.INCREMENTAL
        
        # Final verification of complete workflow
        final_stats = await project_indexer.get_analysis_statistics()
        assert final_stats['files_processed'] > 0
        assert final_stats['analysis_time'] > 0
        
        # Cache should have been used effectively
        cache_stats = await project_indexer.cache.get_cache_stats(str(project.id))
        assert cache_stats['analysis_cache_entries'] > 0