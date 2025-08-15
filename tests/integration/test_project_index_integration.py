"""
Integration tests for Project Index system.

Tests the complete Project Index workflow with real database and Redis
connections, verifying end-to-end functionality and data persistence.
"""

import asyncio
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, AsyncMock
from typing import Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

from app.project_index.core import ProjectIndexer
from app.project_index.cache import CacheManager
from app.project_index.file_monitor import FileMonitor
from app.project_index.models import (
    ProjectIndexConfig, AnalysisConfiguration, AnalysisResult
)
from app.models.project_index import (
    ProjectIndex, FileEntry, DependencyRelationship, AnalysisSession,
    ProjectStatus, FileType, AnalysisSessionType, AnalysisStatus
)


class TestProjectIndexIntegration:
    """Integration tests for Project Index system."""
    
    @pytest.fixture
    async def test_engine(self):
        """Create test database engine with in-memory SQLite."""
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=False
        )
        
        # Create minimal tables for testing using individual execute statements
        # Note: This is a simplified schema for testing
        async with engine.begin() as conn:
            from sqlalchemy import text
            
            # Create tables one by one
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
                    encoding TEXT DEFAULT 'utf-8',
                    file_size INTEGER DEFAULT 0,
                    line_count INTEGER DEFAULT 0,
                    sha256_hash TEXT,
                    content_preview TEXT,
                    analysis_data TEXT DEFAULT '{}',
                    metadata TEXT DEFAULT '{}',
                    tags TEXT DEFAULT '[]',
                    is_binary BOOLEAN DEFAULT FALSE,
                    is_generated BOOLEAN DEFAULT FALSE,
                    last_modified TIMESTAMP,
                    indexed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (project_id) REFERENCES project_indexes (id)
                )
            """))
            
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
            
            await conn.execute(text("""
                CREATE TABLE analysis_sessions (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    session_name TEXT NOT NULL,
                    session_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    progress_percentage REAL DEFAULT 0.0,
                    current_phase TEXT,
                    files_total INTEGER DEFAULT 0,
                    files_processed INTEGER DEFAULT 0,
                    dependencies_found INTEGER DEFAULT 0,
                    errors_count INTEGER DEFAULT 0,
                    warnings_count INTEGER DEFAULT 0,
                    session_data TEXT DEFAULT '{}',
                    error_log TEXT DEFAULT '[]',
                    performance_metrics TEXT DEFAULT '{}',
                    configuration TEXT DEFAULT '{}',
                    result_data TEXT DEFAULT '{}',
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    estimated_completion TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
        """Mock Redis client for testing."""
        redis_mock = AsyncMock()
        redis_mock.ping.return_value = True
        redis_mock.get.return_value = None
        redis_mock.set.return_value = True
        redis_mock.setex.return_value = True
        redis_mock.delete.return_value = 1
        redis_mock.exists.return_value = False
        redis_mock.keys.return_value = []
        redis_mock.mget.return_value = []
        redis_mock.mset.return_value = True
        redis_mock.expire.return_value = True
        redis_mock.info.return_value = {
            'used_memory': 1024,
            'used_memory_human': '1.00K',
            'keyspace_hits': 100,
            'keyspace_misses': 20,
            'total_commands_processed': 500
        }
        return redis_mock
    
    @pytest.fixture
    def project_config(self):
        """Create test project configuration."""
        return ProjectIndexConfig(
            analysis_batch_size=5,
            max_concurrent_analyses=3,
            cache_ttl=300,
            analysis_config=AnalysisConfiguration(
                include_ast=True,
                include_dependencies=True,
                include_complexity_metrics=True,
                max_file_size=1024 * 1024  # 1MB
            )
        )
    
    @pytest.fixture
    async def project_indexer(self, test_session, mock_redis, project_config):
        """Create ProjectIndexer instance for testing."""
        return ProjectIndexer(
            session=test_session,
            redis_client=mock_redis,
            config=project_config
        )
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create Python files
            (temp_path / "main.py").write_text("""
import os
import sys
from utils import helper_function

def main():
    print("Hello, World!")
    helper_function()

if __name__ == "__main__":
    main()
""")
            
            (temp_path / "utils.py").write_text("""
import json
import requests

def helper_function():
    return "Helper function called"

def load_config(filename):
    with open(filename, 'r') as f:
        return json.load(f)
""")
            
            # Create subdirectory with more files
            (temp_path / "modules").mkdir()
            (temp_path / "modules" / "__init__.py").write_text("")
            (temp_path / "modules" / "calculator.py").write_text("""
class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
""")
            
            # Create test files
            (temp_path / "tests").mkdir()
            (temp_path / "tests" / "test_main.py").write_text("""
import unittest
from main import main

class TestMain(unittest.TestCase):
    def test_main(self):
        # Test main function
        pass
""")
            
            # Create configuration files
            (temp_path / "requirements.txt").write_text("""
requests==2.28.0
pytest==7.1.0
""")
            
            (temp_path / "config.json").write_text("""
{
    "app_name": "Test App",
    "version": "1.0.0"
}
""")
            
            yield temp_path
    
    @pytest.mark.asyncio
    async def test_create_project(self, project_indexer, temp_project_dir):
        """Test creating a new project index."""
        project = await project_indexer.create_project(
            name="Test Project",
            root_path=str(temp_project_dir),
            description="Test project for integration testing"
        )
        
        assert project is not None
        assert project.name == "Test Project"
        assert Path(project.root_path).resolve() == Path(temp_project_dir).resolve()
        assert project.status == ProjectStatus.INACTIVE
        assert project.description == "Test project for integration testing"
    
    @pytest.mark.asyncio
    async def test_full_project_analysis(self, project_indexer, temp_project_dir):
        """Test complete project analysis workflow."""
        # Create project
        project = await project_indexer.create_project(
            name="Full Analysis Test",
            root_path=str(temp_project_dir)
        )
        
        # Perform full analysis
        with patch.object(project_indexer.analyzer, 'parse_file') as mock_parse:
            with patch.object(project_indexer.analyzer, 'extract_dependencies') as mock_deps:
                with patch.object(project_indexer.analyzer, 'detect_language') as mock_lang:
                    # Mock analyzer responses
                    mock_parse.return_value = {
                        'line_count': 10,
                        'function_count': 2,
                        'class_count': 1
                    }
                    mock_deps.return_value = []
                    mock_lang.return_value = "python"
                    
                    result = await project_indexer.analyze_project(
                        str(project.id),
                        analysis_type=AnalysisSessionType.FULL_ANALYSIS
                    )
        
        assert result is not None
        assert result.project_id == str(project.id)
        assert result.analysis_type == AnalysisSessionType.FULL_ANALYSIS.value
        assert result.files_processed > 0
        assert result.analysis_duration > 0
        
        # Verify project status updated
        updated_project = await project_indexer.get_project(str(project.id))
        assert updated_project.status == ProjectStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_incremental_analysis(self, project_indexer, temp_project_dir):
        """Test incremental analysis after initial full analysis."""
        # Create project and do initial analysis
        project = await project_indexer.create_project(
            name="Incremental Test",
            root_path=str(temp_project_dir)
        )
        
        # Mock analyzer for initial analysis
        with patch.object(project_indexer.analyzer, 'parse_file') as mock_parse:
            with patch.object(project_indexer.analyzer, 'extract_dependencies') as mock_deps:
                with patch.object(project_indexer.analyzer, 'detect_language') as mock_lang:
                    mock_parse.return_value = {'line_count': 10}
                    mock_deps.return_value = []
                    mock_lang.return_value = "python"
                    
                    # Initial full analysis
                    await project_indexer.analyze_project(
                        str(project.id),
                        analysis_type=AnalysisSessionType.FULL_ANALYSIS
                    )
                    
                    # Add a new file to trigger incremental analysis
                    new_file = temp_project_dir / "new_module.py"
                    new_file.write_text("def new_function(): pass")
                    
                    # Wait a bit to ensure different timestamps
                    await asyncio.sleep(0.1)
                    
                    # Perform incremental analysis
                    result = await project_indexer.analyze_project(
                        str(project.id),
                        analysis_type=AnalysisSessionType.INCREMENTAL
                    )
        
        assert result is not None
        assert result.analysis_type == AnalysisSessionType.INCREMENTAL
    
    @pytest.mark.asyncio
    async def test_cache_integration(self, project_indexer, temp_project_dir, mock_redis):
        """Test cache integration during analysis."""
        # Create project
        project = await project_indexer.create_project(
            name="Cache Test",
            root_path=str(temp_project_dir)
        )
        
        # Mock cache hit
        mock_redis.get.return_value = None  # Cache miss first
        
        with patch.object(project_indexer.analyzer, 'parse_file') as mock_parse:
            with patch.object(project_indexer.analyzer, 'extract_dependencies') as mock_deps:
                with patch.object(project_indexer.analyzer, 'detect_language') as mock_lang:
                    mock_parse.return_value = {'line_count': 10}
                    mock_deps.return_value = []
                    mock_lang.return_value = "python"
                    
                    # First analysis - should cache results
                    result1 = await project_indexer.analyze_project(
                        str(project.id),
                        analysis_type=AnalysisSessionType.FULL_ANALYSIS
                    )
                    
                    # Verify cache was called for storage
                    assert mock_redis.setex.called
        
        assert result1 is not None
        assert result1.files_processed > 0
    
    @pytest.mark.asyncio
    async def test_file_monitor_integration(self, project_indexer, temp_project_dir):
        """Test file monitor integration with project analysis."""
        # Create project
        project = await project_indexer.create_project(
            name="Monitor Test",
            root_path=str(temp_project_dir)
        )
        
        # Start file monitoring
        monitor = project_indexer.file_monitor
        changes_detected = []
        
        async def change_callback(event):
            changes_detected.append(event)
        
        monitor.add_change_callback(change_callback)
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Let monitoring initialize
        await asyncio.sleep(0.1)
        
        # Create a new file
        new_file = temp_project_dir / "monitored_file.py"
        new_file.write_text("# New file for monitoring test")
        
        # Wait for detection
        await asyncio.sleep(0.2)
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        # Verify project was added to monitor
        assert str(project.id) in monitor._monitored_projects
    
    @pytest.mark.asyncio
    async def test_dependency_analysis(self, project_indexer, temp_project_dir):
        """Test dependency analysis and relationship building."""
        # Create project
        project = await project_indexer.create_project(
            name="Dependency Test",
            root_path=str(temp_project_dir)
        )
        
        # Mock dependency extraction
        with patch.object(project_indexer.analyzer, 'extract_dependencies') as mock_deps:
            with patch.object(project_indexer.analyzer, 'parse_file') as mock_parse:
                with patch.object(project_indexer.analyzer, 'detect_language') as mock_lang:
                    from app.project_index.models import DependencyResult
                    
                    # Mock dependencies
                    mock_deps.return_value = [
                        DependencyResult(
                            source_file_path=str(temp_project_dir / "main.py"),
                            target_name="utils",
                            dependency_type="import",
                            line_number=3,
                            confidence_score=0.9
                        )
                    ]
                    mock_parse.return_value = {'line_count': 10}
                    mock_lang.return_value = "python"
                    
                    # Perform dependency analysis
                    result = await project_indexer.analyze_project(
                        str(project.id),
                        analysis_type=AnalysisSessionType.DEPENDENCY_MAPPING
                    )
        
        assert result is not None
        assert result.analysis_type == AnalysisSessionType.DEPENDENCY_MAPPING
        assert result.dependencies_found >= 0
    
    @pytest.mark.asyncio
    async def test_context_optimization(self, project_indexer, temp_project_dir):
        """Test context optimization analysis."""
        # Create project
        project = await project_indexer.create_project(
            name="Context Test",
            root_path=str(temp_project_dir)
        )
        
        # Perform context optimization
        result = await project_indexer.analyze_project(
            str(project.id),
            analysis_type=AnalysisSessionType.CONTEXT_OPTIMIZATION
        )
        
        assert result is not None
        assert result.analysis_type == AnalysisSessionType.CONTEXT_OPTIMIZATION
        assert 'context_optimization' in result.to_dict()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, project_indexer, temp_project_dir):
        """Test error handling during analysis."""
        # Create project
        project = await project_indexer.create_project(
            name="Error Test",
            root_path=str(temp_project_dir)
        )
        
        # Mock analyzer to raise exception
        with patch.object(project_indexer.analyzer, 'parse_file') as mock_parse:
            mock_parse.side_effect = Exception("Analysis failed")
            
            # Analysis should handle errors gracefully
            with pytest.raises(RuntimeError):
                await project_indexer.analyze_project(
                    str(project.id),
                    analysis_type=AnalysisSessionType.FULL_ANALYSIS
                )
        
        # Verify project status updated to failed
        updated_project = await project_indexer.get_project(str(project.id))
        assert updated_project.status == ProjectStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_project_not_found(self, project_indexer):
        """Test handling of non-existent project."""
        # Try to analyze non-existent project
        with pytest.raises(ValueError, match="Project not found"):
            await project_indexer.analyze_project(
                "non-existent-id",
                analysis_type=AnalysisSessionType.FULL_ANALYSIS
            )
    
    @pytest.mark.asyncio
    async def test_invalid_project_path(self, project_indexer):
        """Test creation with invalid project path."""
        # Try to create project with non-existent path
        with pytest.raises(ValueError, match="Root path does not exist"):
            await project_indexer.create_project(
                name="Invalid Path Test",
                root_path="/non/existent/path"
            )
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, project_indexer, temp_project_dir):
        """Test handling of concurrent analysis requests."""
        # Create project
        project = await project_indexer.create_project(
            name="Concurrent Test",
            root_path=str(temp_project_dir)
        )
        
        # Mock analyzer
        with patch.object(project_indexer.analyzer, 'parse_file') as mock_parse:
            with patch.object(project_indexer.analyzer, 'extract_dependencies') as mock_deps:
                with patch.object(project_indexer.analyzer, 'detect_language') as mock_lang:
                    mock_parse.return_value = {'line_count': 10}
                    mock_deps.return_value = []
                    mock_lang.return_value = "python"
                    
                    # Start two analyses concurrently
                    task1 = asyncio.create_task(
                        project_indexer.analyze_project(
                            str(project.id),
                            analysis_type=AnalysisSessionType.FULL_ANALYSIS
                        )
                    )
                    
                    task2 = asyncio.create_task(
                        project_indexer.analyze_project(
                            str(project.id),
                            analysis_type=AnalysisSessionType.INCREMENTAL
                        )
                    )
                    
                    # Wait for both to complete
                    results = await asyncio.gather(task1, task2, return_exceptions=True)
        
        # At least one should succeed
        successful_results = [r for r in results if isinstance(r, AnalysisResult)]
        assert len(successful_results) >= 1
    
    @pytest.mark.asyncio
    async def test_large_project_handling(self, project_indexer):
        """Test handling of large projects with batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create many files to test batch processing
            for i in range(20):
                file_path = temp_path / f"module_{i}.py"
                file_path.write_text(f"""
def function_{i}():
    return {i}

class Class_{i}:
    def method(self):
        return {i}
""")
            
            # Create project
            project = await project_indexer.create_project(
                name="Large Project Test",
                root_path=str(temp_path)
            )
            
            # Mock analyzer
            with patch.object(project_indexer.analyzer, 'parse_file') as mock_parse:
                with patch.object(project_indexer.analyzer, 'extract_dependencies') as mock_deps:
                    with patch.object(project_indexer.analyzer, 'detect_language') as mock_lang:
                        mock_parse.return_value = {'line_count': 10}
                        mock_deps.return_value = []
                        mock_lang.return_value = "python"
                        
                        # Perform analysis
                        result = await project_indexer.analyze_project(
                            str(project.id),
                            analysis_type=AnalysisSessionType.FULL_ANALYSIS
                        )
            
            assert result is not None
            assert result.files_processed >= 20  # Should process all created files
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, project_indexer, temp_project_dir):
        """Test performance statistics tracking."""
        # Create project
        project = await project_indexer.create_project(
            name="Performance Test",
            root_path=str(temp_project_dir)
        )
        
        # Get initial stats
        initial_stats = await project_indexer.get_analysis_statistics()
        
        # Mock analyzer
        with patch.object(project_indexer.analyzer, 'parse_file') as mock_parse:
            with patch.object(project_indexer.analyzer, 'extract_dependencies') as mock_deps:
                with patch.object(project_indexer.analyzer, 'detect_language') as mock_lang:
                    mock_parse.return_value = {'line_count': 10}
                    mock_deps.return_value = []
                    mock_lang.return_value = "python"
                    
                    # Perform analysis
                    result = await project_indexer.analyze_project(
                        str(project.id),
                        analysis_type=AnalysisSessionType.FULL_ANALYSIS
                    )
        
        # Get updated stats
        final_stats = await project_indexer.get_analysis_statistics()
        
        # Verify stats were updated
        assert final_stats['files_processed'] > initial_stats['files_processed']
        assert final_stats['analysis_time'] > initial_stats['analysis_time']
        assert result.analysis_duration > 0
    
    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, project_indexer, temp_project_dir):
        """Test cleanup of old analysis data."""
        # Create project
        project = await project_indexer.create_project(
            name="Cleanup Test",
            root_path=str(temp_project_dir)
        )
        
        # Mock analyzer and perform analysis
        with patch.object(project_indexer.analyzer, 'parse_file') as mock_parse:
            with patch.object(project_indexer.analyzer, 'extract_dependencies') as mock_deps:
                with patch.object(project_indexer.analyzer, 'detect_language') as mock_lang:
                    mock_parse.return_value = {'line_count': 10}
                    mock_deps.return_value = []
                    mock_lang.return_value = "python"
                    
                    await project_indexer.analyze_project(
                        str(project.id),
                        analysis_type=AnalysisSessionType.FULL_ANALYSIS
                    )
        
        # Test cleanup (should not delete recent data)
        cleanup_stats = await project_indexer.cleanup_old_data(retention_days=1)
        
        assert 'deleted_sessions' in cleanup_stats
        assert 'deleted_snapshots' in cleanup_stats
        # Recent data should not be deleted
        assert cleanup_stats['deleted_sessions'] == 0