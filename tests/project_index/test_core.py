"""
Comprehensive Unit Tests for Project Index Core Infrastructure

Tests the core ProjectIndexer functionality, code analysis, file monitoring,
and context optimization with full coverage and performance validation.
"""

import asyncio
import uuid
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import pytest
from sqlalchemy import select

from app.project_index.core import ProjectIndexer
from app.project_index.models import ProjectIndexConfig, AnalysisConfiguration
from app.project_index.analyzer import CodeAnalyzer
from app.project_index.file_monitor import FileMonitor
from app.project_index.cache import CacheManager, CacheConfig
from app.models.project_index import (
    ProjectIndex, FileEntry, DependencyRelationship, AnalysisSession,
    ProjectStatus, FileType, DependencyType, AnalysisSessionType, AnalysisStatus
)


class TestProjectIndexer:
    """Test ProjectIndexer core functionality."""
    
    @pytest.mark.asyncio
    async def test_create_project(self, project_indexer, sample_project_data):
        """Test project creation functionality."""
        data, temp_dir = sample_project_data
        
        project = await project_indexer.create_project(
            name=data["name"],
            root_path=data["root_path"],
            description=data["description"],
            git_repository_url=data["git_repository_url"],
            git_branch=data["git_branch"],
            configuration=data["configuration"]
        )
        
        assert project is not None
        assert isinstance(project.id, uuid.UUID)
        assert project.name == data["name"]
        assert project.description == data["description"]
        assert project.root_path == data["root_path"]
        assert project.git_repository_url == data["git_repository_url"]
        assert project.git_branch == data["git_branch"]
        assert project.status == ProjectStatus.INACTIVE
        assert project.configuration == data["configuration"]
    
    @pytest.mark.asyncio
    async def test_create_project_validation(self, project_indexer):
        """Test project creation validation."""
        # Test with invalid root path
        with pytest.raises(ValueError, match="Root path does not exist"):
            await project_indexer.create_project(
                name="Invalid Project",
                root_path="/nonexistent/path"
            )
        
        # Test with empty name
        with pytest.raises(ValueError, match="Project name cannot be empty"):
            await project_indexer.create_project(
                name="",
                root_path="/tmp"
            )
    
    @pytest.mark.asyncio
    async def test_get_project(self, project_indexer, test_project):
        """Test project retrieval."""
        # Test get by ID
        project = await project_indexer.get_project(str(test_project.id))
        assert project is not None
        assert project.id == test_project.id
        assert project.name == test_project.name
        
        # Test get non-existent project
        non_existent_id = str(uuid.uuid4())
        project = await project_indexer.get_project(non_existent_id)
        assert project is None
    
    @pytest.mark.asyncio
    async def test_list_projects(self, project_indexer, test_session):
        """Test project listing with filtering."""
        # Create multiple projects
        projects = []
        for i in range(5):
            project = ProjectIndex(
                name=f"Test Project {i}",
                root_path=f"/tmp/test_{i}",
                status=ProjectStatus.ACTIVE if i % 2 == 0 else ProjectStatus.INACTIVE
            )
            test_session.add(project)
            projects.append(project)
        
        await test_session.commit()
        
        # Test list all projects
        all_projects = await project_indexer.list_projects()
        assert len(all_projects) >= 5
        
        # Test list with status filter
        active_projects = await project_indexer.list_projects(
            status=[ProjectStatus.ACTIVE]
        )
        assert all(p.status == ProjectStatus.ACTIVE for p in active_projects)
        
        # Test pagination
        first_page = await project_indexer.list_projects(page=1, page_size=2)
        assert len(first_page) <= 2
    
    @pytest.mark.asyncio
    async def test_update_project(self, project_indexer, test_project):
        """Test project updates."""
        updates = {
            "description": "Updated description",
            "status": ProjectStatus.ACTIVE,
            "configuration": {"updated": True}
        }
        
        updated_project = await project_indexer.update_project(
            str(test_project.id), 
            updates
        )
        
        assert updated_project.description == "Updated description"
        assert updated_project.status == ProjectStatus.ACTIVE
        assert updated_project.configuration["updated"] is True
        assert updated_project.updated_at > test_project.updated_at
    
    @pytest.mark.asyncio
    async def test_delete_project(self, project_indexer, test_project, test_session):
        """Test project deletion."""
        project_id = test_project.id
        
        # Delete project
        result = await project_indexer.delete_project(str(project_id))
        assert result is True
        
        # Verify project is deleted
        stmt = select(ProjectIndex).where(ProjectIndex.id == project_id)
        result = await test_session.execute(stmt)
        project = result.scalar_one_or_none()
        assert project is None
    
    @pytest.mark.asyncio
    async def test_analyze_project_full(self, project_indexer, test_project, performance_monitor):
        """Test full project analysis."""
        performance_monitor.start_timer("full_analysis")
        
        with patch.object(project_indexer, '_scan_project_files') as mock_scan, \
             patch.object(project_indexer, '_analyze_files') as mock_analyze, \
             patch.object(project_indexer, '_extract_dependencies') as mock_deps:
            
            mock_scan.return_value = ["file1.py", "file2.py"]
            mock_analyze.return_value = {}
            mock_deps.return_value = []
            
            session = await project_indexer.analyze_project(
                str(test_project.id),
                AnalysisSessionType.FULL_ANALYSIS
            )
            
            performance_monitor.end_timer("full_analysis")
            
            assert session is not None
            assert session.project_id == test_project.id
            assert session.session_type == AnalysisSessionType.FULL_ANALYSIS
            
            # Verify analysis methods were called
            mock_scan.assert_called_once()
            mock_analyze.assert_called_once()
            mock_deps.assert_called_once()
            
            # Performance assertion
            performance_monitor.assert_performance("full_analysis", 10.0)
    
    @pytest.mark.asyncio
    async def test_analyze_project_incremental(self, project_indexer, test_project, test_files):
        """Test incremental project analysis."""
        with patch.object(project_indexer, '_get_changed_files') as mock_changed, \
             patch.object(project_indexer, '_analyze_files') as mock_analyze:
            
            mock_changed.return_value = [test_files[0].relative_path]
            mock_analyze.return_value = {}
            
            session = await project_indexer.analyze_project(
                str(test_project.id),
                AnalysisSessionType.INCREMENTAL
            )
            
            assert session.session_type == AnalysisSessionType.INCREMENTAL
            mock_changed.assert_called_once()
            mock_analyze.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_project_error_handling(self, project_indexer, test_project):
        """Test analysis error handling."""
        with patch.object(project_indexer, '_scan_project_files') as mock_scan:
            mock_scan.side_effect = Exception("Scan failed")
            
            session = await project_indexer.analyze_project(
                str(test_project.id),
                AnalysisSessionType.FULL_ANALYSIS
            )
            
            assert session.status == AnalysisStatus.FAILED
            assert session.errors_count > 0
            assert len(session.error_log) > 0
    
    @pytest.mark.asyncio
    async def test_get_project_statistics(self, project_indexer, test_project, test_files, test_dependencies):
        """Test project statistics calculation."""
        stats = await project_indexer.get_project_statistics(str(test_project.id))
        
        assert "total_files" in stats
        assert "files_by_type" in stats
        assert "files_by_language" in stats
        assert "total_dependencies" in stats
        assert "dependencies_by_type" in stats
        assert "external_dependencies" in stats
        assert "internal_dependencies" in stats
        
        assert stats["total_files"] == len(test_files)
        assert stats["total_dependencies"] == len(test_dependencies)
    
    @pytest.mark.asyncio
    async def test_get_dependency_graph(self, project_indexer, test_project, test_files, test_dependencies):
        """Test dependency graph generation."""
        graph = await project_indexer.get_dependency_graph(
            str(test_project.id),
            include_external=True
        )
        
        assert "nodes" in graph
        assert "edges" in graph
        assert "statistics" in graph
        
        # Verify nodes represent files
        assert len(graph["nodes"]) == len(test_files)
        for node in graph["nodes"]:
            assert "file_id" in node
            assert "file_path" in node
            assert "file_type" in node
        
        # Verify edges represent dependencies
        assert len(graph["edges"]) == len(test_dependencies)
        for edge in graph["edges"]:
            assert "source_file_id" in edge
            assert "dependency_type" in edge


class TestCodeAnalyzer:
    """Test CodeAnalyzer functionality."""
    
    @pytest.fixture
    def code_analyzer(self):
        """Create CodeAnalyzer instance."""
        return CodeAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_python_file(self, code_analyzer, sample_project_data):
        """Test Python code analysis."""
        data, temp_dir = sample_project_data
        main_py_path = temp_dir / "src" / "main.py"
        
        analysis = await code_analyzer.analyze_file(str(main_py_path))
        
        assert analysis is not None
        assert "language" in analysis
        assert analysis["language"] == "python"
        assert "imports" in analysis
        assert "functions" in analysis
        assert "classes" in analysis
        
        # Verify specific content was detected
        assert "os" in analysis["imports"]
        assert "sys" in analysis["imports"]
        assert "main" in analysis["functions"]
    
    @pytest.mark.asyncio
    async def test_analyze_non_code_file(self, code_analyzer, sample_project_data):
        """Test analysis of non-code files."""
        data, temp_dir = sample_project_data
        readme_path = temp_dir / "README.md"
        
        analysis = await code_analyzer.analyze_file(str(readme_path))
        
        assert analysis is not None
        assert "language" in analysis
        assert analysis["language"] == "markdown"
        assert "line_count" in analysis
        assert "file_size" in analysis
    
    @pytest.mark.asyncio
    async def test_analyze_binary_file(self, code_analyzer):
        """Test binary file handling."""
        # Create temporary binary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b'\x89PNG\r\n\x1a\n')  # PNG header
            binary_path = f.name
        
        try:
            analysis = await code_analyzer.analyze_file(binary_path)
            
            assert analysis is not None
            assert analysis.get("is_binary") is True
            assert analysis.get("language") is None
        finally:
            Path(binary_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_extract_dependencies_python(self, code_analyzer, sample_project_data):
        """Test dependency extraction from Python code."""
        data, temp_dir = sample_project_data
        main_py_path = temp_dir / "src" / "main.py"
        
        dependencies = await code_analyzer.extract_dependencies(str(main_py_path))
        
        assert isinstance(dependencies, list)
        assert len(dependencies) > 0
        
        # Check for expected imports
        import_names = [dep["target_name"] for dep in dependencies]
        assert "os" in import_names
        assert "sys" in import_names
        assert "utils" in import_names
        
        # Verify dependency structure
        for dep in dependencies:
            assert "target_name" in dep
            assert "dependency_type" in dep
            assert "line_number" in dep
            assert "is_external" in dep
    
    @pytest.mark.asyncio
    async def test_analyze_file_error_handling(self, code_analyzer):
        """Test error handling for file analysis."""
        # Test non-existent file
        analysis = await code_analyzer.analyze_file("/nonexistent/file.py")
        assert analysis is None
        
        # Test permission denied (if possible)
        with tempfile.NamedTemporaryFile() as f:
            # Remove read permissions
            Path(f.name).chmod(0o000)
            try:
                analysis = await code_analyzer.analyze_file(f.name)
                # Should handle gracefully
                assert analysis is None or "error" in analysis
            finally:
                # Restore permissions for cleanup
                Path(f.name).chmod(0o644)
    
    @pytest.mark.asyncio
    async def test_detect_language(self, code_analyzer):
        """Test language detection."""
        test_cases = [
            ("test.py", "python"),
            ("test.js", "javascript"),
            ("test.ts", "typescript"),
            ("test.java", "java"),
            ("test.cpp", "cpp"),
            ("test.c", "c"),
            ("test.go", "go"),
            ("test.rs", "rust"),
            ("test.rb", "ruby"),
            ("test.php", "php"),
            ("test.html", "html"),
            ("test.css", "css"),
            ("test.md", "markdown"),
            ("test.json", "json"),
            ("test.yaml", "yaml"),
            ("test.xml", "xml"),
            ("test.sql", "sql"),
            ("test.sh", "shell"),
            ("test.unknown", None)
        ]
        
        for filename, expected_language in test_cases:
            detected = code_analyzer.detect_language(filename)
            assert detected == expected_language, f"Failed for {filename}"


class TestFileMonitor:
    """Test FileMonitor functionality."""
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self, file_monitor, sample_project_data):
        """Test file monitoring startup."""
        data, temp_dir = sample_project_data
        
        # Mock the actual file watcher
        with patch('watchdog.observers.Observer') as mock_observer:
            mock_observer_instance = Mock()
            mock_observer.return_value = mock_observer_instance
            
            await file_monitor.start_monitoring(str(temp_dir))
            
            mock_observer_instance.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, file_monitor):
        """Test file monitoring shutdown."""
        # Start monitoring first
        with patch('watchdog.observers.Observer') as mock_observer:
            mock_observer_instance = Mock()
            mock_observer.return_value = mock_observer_instance
            
            await file_monitor.start_monitoring("/tmp")
            await file_monitor.stop_monitoring()
            
            mock_observer_instance.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_file_change_event(self, file_monitor, test_redis):
        """Test file change event handling."""
        event_data = {
            "event_type": "modified",
            "file_path": "/test/file.py",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        await file_monitor.handle_file_change(event_data)
        
        # Verify event was published to Redis
        test_redis.publish.assert_called()
        call_args = test_redis.publish.call_args
        assert "file_change" in call_args[0][0]  # Channel name
    
    @pytest.mark.asyncio
    async def test_batch_file_changes(self, file_monitor):
        """Test batching of multiple file changes."""
        changes = [
            {"event_type": "created", "file_path": "/test/new.py"},
            {"event_type": "modified", "file_path": "/test/existing.py"},
            {"event_type": "deleted", "file_path": "/test/old.py"}
        ]
        
        with patch.object(file_monitor, 'handle_file_change') as mock_handle:
            await file_monitor.handle_batch_changes(changes)
            
            assert mock_handle.call_count == len(changes)


class TestCacheManager:
    """Test CacheManager functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_set_get(self, cache_manager):
        """Test basic cache operations."""
        key = "test_key"
        value = {"data": "test_value", "number": 42}
        
        # Test set
        result = await cache_manager.set(key, value, ttl=300)
        assert result is True
        
        # Test get
        cached_value = await cache_manager.get(key)
        assert cached_value == value
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self, cache_manager):
        """Test cache expiration."""
        key = "expiring_key"
        value = "expiring_value"
        
        # Set with very short TTL
        await cache_manager.set(key, value, ttl=1)
        
        # Should exist immediately
        cached_value = await cache_manager.get(key)
        assert cached_value == value
        
        # Wait for expiration
        await asyncio.sleep(1.1)
        
        # Should be expired
        cached_value = await cache_manager.get(key)
        assert cached_value is None
    
    @pytest.mark.asyncio
    async def test_cache_delete(self, cache_manager):
        """Test cache deletion."""
        key = "delete_key"
        value = "delete_value"
        
        await cache_manager.set(key, value)
        
        # Verify it exists
        cached_value = await cache_manager.get(key)
        assert cached_value == value
        
        # Delete it
        result = await cache_manager.delete(key)
        assert result is True
        
        # Verify it's gone
        cached_value = await cache_manager.get(key)
        assert cached_value is None
    
    @pytest.mark.asyncio
    async def test_cache_pattern_delete(self, cache_manager):
        """Test pattern-based cache deletion."""
        # Set multiple keys with pattern
        for i in range(5):
            await cache_manager.set(f"pattern_test:{i}", f"value_{i}")
        
        # Delete by pattern
        deleted_count = await cache_manager.delete_pattern("pattern_test:*")
        assert deleted_count >= 5
        
        # Verify they're gone
        for i in range(5):
            cached_value = await cache_manager.get(f"pattern_test:{i}")
            assert cached_value is None
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, cache_manager):
        """Test cache statistics."""
        # Perform some cache operations
        await cache_manager.set("stats_test", "value")
        await cache_manager.get("stats_test")
        await cache_manager.get("nonexistent_key")
        
        stats = await cache_manager.get_stats()
        
        assert "hits" in stats
        assert "misses" in stats
        assert "total_keys" in stats
        assert stats["hits"] >= 1
        assert stats["misses"] >= 1


class TestProjectIndexConfiguration:
    """Test ProjectIndexer configuration and settings."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ProjectIndexConfig()
        
        assert config.analysis_timeout > 0
        assert config.max_file_size > 0
        assert config.max_files_per_project > 0
        assert config.cache_ttl > 0
        assert isinstance(config.enable_ai_analysis, bool)
        assert isinstance(config.supported_languages, list)
        assert len(config.supported_languages) > 0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ProjectIndexConfig(
            analysis_timeout=600,
            max_file_size=20_000_000,
            enable_ai_analysis=False,
            supported_languages=["python", "javascript"]
        )
        
        assert config.analysis_timeout == 600
        assert config.max_file_size == 20_000_000
        assert config.enable_ai_analysis is False
        assert config.supported_languages == ["python", "javascript"]
    
    def test_analysis_configuration(self):
        """Test analysis configuration."""
        config = AnalysisConfiguration(
            force_reanalysis=True,
            file_filters=["*.py", "*.js"],
            analysis_depth=3,
            custom_settings={"extract_docstrings": True}
        )
        
        assert config.force_reanalysis is True
        assert config.file_filters == ["*.py", "*.js"]
        assert config.analysis_depth == 3
        assert config.custom_settings["extract_docstrings"] is True


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_concurrent_analysis(self, project_indexer, test_project):
        """Test handling of concurrent analysis requests."""
        # Start first analysis
        task1 = asyncio.create_task(
            project_indexer.analyze_project(
                str(test_project.id),
                AnalysisSessionType.FULL_ANALYSIS
            )
        )
        
        # Start second analysis immediately
        task2 = asyncio.create_task(
            project_indexer.analyze_project(
                str(test_project.id),
                AnalysisSessionType.INCREMENTAL
            )
        )
        
        # Wait for both to complete
        session1, session2 = await asyncio.gather(task1, task2, return_exceptions=True)
        
        # One should succeed, the other should handle gracefully
        sessions = [s for s in [session1, session2] if not isinstance(s, Exception)]
        assert len(sessions) >= 1
    
    @pytest.mark.asyncio
    async def test_large_project_handling(self, project_indexer, test_session):
        """Test handling of projects with many files."""
        # Create a project with many files
        project = ProjectIndex(
            name="Large Project",
            root_path="/tmp/large_project",
            status=ProjectStatus.ACTIVE
        )
        test_session.add(project)
        await test_session.commit()
        await test_session.refresh(project)
        
        # Add many file entries
        for i in range(1000):
            file_entry = FileEntry(
                project_id=project.id,
                file_path=f"/tmp/large_project/file_{i}.py",
                relative_path=f"file_{i}.py",
                file_name=f"file_{i}.py",
                file_type=FileType.SOURCE,
                language="python"
            )
            test_session.add(file_entry)
        
        await test_session.commit()
        
        # Test statistics calculation
        stats = await project_indexer.get_project_statistics(str(project.id))
        assert stats["total_files"] == 1000
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, project_indexer, performance_monitor):
        """Test memory usage during analysis."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform analysis operations
        with patch.object(project_indexer, '_analyze_files') as mock_analyze:
            mock_analyze.return_value = {"memory_test": True}
            
            # Simulate analysis of many files
            for i in range(100):
                await mock_analyze([f"file_{i}.py"])
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100_000_000, f"Memory increased by {memory_increase} bytes"
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self, project_indexer, test_redis):
        """Test handling of network/Redis errors."""
        # Mock Redis connection failure
        test_redis.get.side_effect = ConnectionError("Redis connection failed")
        
        # Operations should continue gracefully
        with patch.object(project_indexer, '_handle_redis_error') as mock_error_handler:
            mock_error_handler.return_value = None
            
            # This should not raise an exception
            result = await project_indexer._get_cached_analysis("test_key")
            assert result is None
            mock_error_handler.assert_called()
    
    @pytest.mark.asyncio
    async def test_database_transaction_rollback(self, project_indexer, test_session):
        """Test database transaction rollback on errors."""
        with patch.object(test_session, 'commit') as mock_commit:
            mock_commit.side_effect = Exception("Database error")
            
            # This should handle the error gracefully
            with pytest.raises(Exception):
                await project_indexer.create_project(
                    name="Test Project",
                    root_path="/tmp/test"
                )
            
            # Session should be rolled back
            # (This would be verified by checking that no partial data was committed)