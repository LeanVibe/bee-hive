"""
Unit tests for Project Index file monitoring.

Tests for file system change monitoring with polling-based
cross-platform compatibility and change detection.
"""

import pytest
import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

from app.project_index.file_monitor import (
    FileChangeType,
    FileChangeEvent,
    FileMonitor
)


class TestFileChangeType:
    """Test FileChangeType enum."""
    
    def test_file_change_types(self):
        """Test file change type enum values."""
        assert FileChangeType.CREATED.value == "created"
        assert FileChangeType.MODIFIED.value == "modified"
        assert FileChangeType.DELETED.value == "deleted"
        assert FileChangeType.MOVED.value == "moved"


class TestFileChangeEvent:
    """Test FileChangeEvent dataclass."""
    
    def test_basic_event_creation(self):
        """Test basic file change event creation."""
        now = datetime.utcnow()
        event = FileChangeEvent(
            file_path="/path/to/file.py",
            change_type=FileChangeType.MODIFIED,
            timestamp=now,
            old_hash="abc123",
            new_hash="def456"
        )
        
        assert event.file_path == "/path/to/file.py"
        assert event.change_type == FileChangeType.MODIFIED
        assert event.timestamp == now
        assert event.old_hash == "abc123"
        assert event.new_hash == "def456"
        assert event.metadata == {}
    
    def test_event_with_metadata(self):
        """Test file change event with metadata."""
        event = FileChangeEvent(
            file_path="/path/to/file.py",
            change_type=FileChangeType.CREATED,
            timestamp=datetime.utcnow(),
            metadata={"size": 1024, "permissions": "644"}
        )
        
        assert event.metadata == {"size": 1024, "permissions": "644"}
    
    def test_event_equality(self):
        """Test file change event equality."""
        timestamp = datetime.utcnow()
        
        event1 = FileChangeEvent(
            file_path="/path/to/file.py",
            change_type=FileChangeType.MODIFIED,
            timestamp=timestamp
        )
        
        event2 = FileChangeEvent(
            file_path="/path/to/file.py",
            change_type=FileChangeType.MODIFIED,
            timestamp=timestamp
        )
        
        event3 = FileChangeEvent(
            file_path="/path/to/other.py",
            change_type=FileChangeType.MODIFIED,
            timestamp=timestamp
        )
        
        assert event1 == event2
        assert event1 != event3


class TestFileMonitor:
    """Test FileMonitor functionality."""
    
    def test_file_monitor_initialization(self):
        """Test file monitor initialization."""
        monitor = FileMonitor(poll_interval=1.0)
        
        assert monitor.poll_interval == 1.0
        assert monitor.projects == {}
        assert monitor._running is False
        assert monitor._monitor_task is None
        assert monitor._file_states == {}
    
    def test_file_monitor_default_poll_interval(self):
        """Test file monitor with default poll interval."""
        monitor = FileMonitor()
        assert monitor.poll_interval == 2.0
    
    def test_add_project(self):
        """Test adding project to monitor."""
        monitor = FileMonitor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_id = monitor.add_project("test_project", temp_dir)
            
            assert project_id in monitor.projects
            assert monitor.projects[project_id]["name"] == "test_project"
            assert monitor.projects[project_id]["root_path"] == temp_dir
            assert "include_patterns" in monitor.projects[project_id]
            assert "exclude_patterns" in monitor.projects[project_id]
    
    def test_add_project_with_patterns(self):
        """Test adding project with custom patterns."""
        monitor = FileMonitor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_id = monitor.add_project(
                "test_project",
                temp_dir,
                include_patterns=["*.py", "*.js"],
                exclude_patterns=["*.pyc", "node_modules/**"]
            )
            
            project = monitor.projects[project_id]
            assert project["include_patterns"] == ["*.py", "*.js"]
            assert project["exclude_patterns"] == ["*.pyc", "node_modules/**"]
    
    def test_remove_project(self):
        """Test removing project from monitor."""
        monitor = FileMonitor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_id = monitor.add_project("test_project", temp_dir)
            assert project_id in monitor.projects
            
            removed = monitor.remove_project(project_id)
            assert removed is True
            assert project_id not in monitor.projects
            
            # Try to remove non-existent project
            removed_again = monitor.remove_project(project_id)
            assert removed_again is False
    
    def test_get_project_info(self):
        """Test getting project information."""
        monitor = FileMonitor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_id = monitor.add_project("test_project", temp_dir)
            
            info = monitor.get_project_info(project_id)
            assert info is not None
            assert info["name"] == "test_project"
            assert info["root_path"] == temp_dir
            
            # Non-existent project
            info = monitor.get_project_info("nonexistent")
            assert info is None
    
    def test_list_projects(self):
        """Test listing all projects."""
        monitor = FileMonitor()
        
        assert monitor.list_projects() == {}
        
        with tempfile.TemporaryDirectory() as temp_dir1:
            with tempfile.TemporaryDirectory() as temp_dir2:
                project_id1 = monitor.add_project("project1", temp_dir1)
                project_id2 = monitor.add_project("project2", temp_dir2)
                
                projects = monitor.list_projects()
                assert len(projects) == 2
                assert project_id1 in projects
                assert project_id2 in projects
                assert projects[project_id1]["name"] == "project1"
                assert projects[project_id2]["name"] == "project2"
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self):
        """Test starting monitoring."""
        monitor = FileMonitor(poll_interval=0.1)  # Fast polling for test
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor.add_project("test_project", temp_dir)
            
            await monitor.start_monitoring()
            
            assert monitor._running is True
            assert monitor._monitor_task is not None
            
            # Let it run briefly
            await asyncio.sleep(0.2)
            
            # Stop monitoring
            await monitor.stop_monitoring()
            
            assert monitor._running is False
            assert monitor._monitor_task is None
    
    @pytest.mark.asyncio
    async def test_stop_monitoring_not_running(self):
        """Test stopping monitoring when not running."""
        monitor = FileMonitor()
        
        # Should not raise exception
        await monitor.stop_monitoring()
        
        assert monitor._running is False
        assert monitor._monitor_task is None
    
    @pytest.mark.asyncio
    async def test_monitoring_context_manager(self):
        """Test monitoring as context manager."""
        monitor = FileMonitor(poll_interval=0.1)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor.add_project("test_project", temp_dir)
            
            async with monitor:
                assert monitor._running is True
                await asyncio.sleep(0.1)
            
            assert monitor._running is False
    
    def test_scan_directory(self):
        """Test directory scanning functionality."""
        monitor = FileMonitor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "file1.py").write_text("print('hello')")
            (temp_path / "file2.js").write_text("console.log('hello')")
            (temp_path / "file3.txt").write_text("hello")
            
            # Create subdirectory with files
            sub_dir = temp_path / "subdir"
            sub_dir.mkdir()
            (sub_dir / "nested.py").write_text("print('nested')")
            
            # Scan with include patterns
            files = monitor._scan_directory(
                temp_dir,
                include_patterns=["*.py"],
                exclude_patterns=[]
            )
            
            # Should find Python files
            py_files = [f for f in files if f.endswith('.py')]
            assert len(py_files) == 2
            assert any('file1.py' in f for f in py_files)
            assert any('nested.py' in f for f in py_files)
    
    def test_scan_directory_with_exclude_patterns(self):
        """Test directory scanning with exclude patterns."""
        monitor = FileMonitor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files and directories
            (temp_path / "file1.py").write_text("code")
            (temp_path / "file2.py").write_text("code")
            
            # Create __pycache__ directory
            cache_dir = temp_path / "__pycache__"
            cache_dir.mkdir()
            (cache_dir / "file1.pyc").write_text("bytecode")
            
            # Scan excluding cache
            files = monitor._scan_directory(
                temp_dir,
                include_patterns=["**/*"],
                exclude_patterns=["**/__pycache__/**"]
            )
            
            # Should not include cached files
            assert not any('__pycache__' in f for f in files)
            assert len([f for f in files if f.endswith('.py')]) == 2
    
    def test_should_include_file(self):
        """Test file inclusion logic."""
        monitor = FileMonitor()
        
        # Test include patterns
        assert monitor._should_include_file(
            "/path/file.py",
            include_patterns=["*.py"],
            exclude_patterns=[]
        ) is True
        
        assert monitor._should_include_file(
            "/path/file.js",
            include_patterns=["*.py"],
            exclude_patterns=[]
        ) is False
        
        # Test exclude patterns
        assert monitor._should_include_file(
            "/path/__pycache__/file.pyc",
            include_patterns=["**/*"],
            exclude_patterns=["**/__pycache__/**"]
        ) is False
        
        # Test multiple patterns
        assert monitor._should_include_file(
            "/path/test_file.py",
            include_patterns=["*.py", "*.js"],
            exclude_patterns=["test_*"]
        ) is False
    
    def test_get_file_hash(self):
        """Test file hash calculation."""
        monitor = FileMonitor()
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write("test content")
            temp_file.flush()
            
            try:
                hash1 = monitor._get_file_hash(temp_file.name)
                assert hash1 is not None
                assert len(hash1) == 64  # SHA256 hex length
                
                # Same content should produce same hash
                hash2 = monitor._get_file_hash(temp_file.name)
                assert hash1 == hash2
                
            finally:
                Path(temp_file.name).unlink()
    
    def test_get_file_hash_nonexistent(self):
        """Test file hash for non-existent file."""
        monitor = FileMonitor()
        
        hash_result = monitor._get_file_hash("/nonexistent/file.py")
        assert hash_result is None
    
    @pytest.mark.asyncio
    async def test_file_change_detection(self):
        """Test file change detection during monitoring."""
        monitor = FileMonitor(poll_interval=0.1)
        changes = []
        
        # Mock callback to capture changes
        async def change_callback(event):
            changes.append(event)
        
        monitor.add_change_callback(change_callback)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            project_id = monitor.add_project("test_project", temp_dir)
            
            # Start monitoring
            await monitor.start_monitoring()
            
            # Let initial scan complete
            await asyncio.sleep(0.2)
            
            # Create a new file
            test_file = temp_path / "new_file.py"
            test_file.write_text("print('hello')")
            
            # Wait for detection
            await asyncio.sleep(0.3)
            
            # Modify the file
            test_file.write_text("print('hello world')")
            
            # Wait for detection
            await asyncio.sleep(0.3)
            
            # Delete the file
            test_file.unlink()
            
            # Wait for detection
            await asyncio.sleep(0.3)
            
            await monitor.stop_monitoring()
            
            # Check detected changes
            assert len(changes) >= 1  # At least one change should be detected
            
            # Check change types
            change_types = [change.change_type for change in changes]
            assert FileChangeType.CREATED in change_types or \
                   FileChangeType.MODIFIED in change_types or \
                   FileChangeType.DELETED in change_types
    
    def test_add_change_callback(self):
        """Test adding change callback."""
        monitor = FileMonitor()
        
        async def callback1(event):
            pass
        
        async def callback2(event):
            pass
        
        monitor.add_change_callback(callback1)
        assert len(monitor._change_callbacks) == 1
        
        monitor.add_change_callback(callback2)
        assert len(monitor._change_callbacks) == 2
    
    def test_remove_change_callback(self):
        """Test removing change callback."""
        monitor = FileMonitor()
        
        async def callback1(event):
            pass
        
        async def callback2(event):
            pass
        
        monitor.add_change_callback(callback1)
        monitor.add_change_callback(callback2)
        assert len(monitor._change_callbacks) == 2
        
        monitor.remove_change_callback(callback1)
        assert len(monitor._change_callbacks) == 1
        assert callback1 not in monitor._change_callbacks
        assert callback2 in monitor._change_callbacks
        
        # Try to remove non-existent callback
        monitor.remove_change_callback(callback1)
        assert len(monitor._change_callbacks) == 1
    
    def test_get_changes_since(self):
        """Test getting changes since specific timestamp."""
        monitor = FileMonitor()
        
        # Create some mock changes
        now = datetime.utcnow()
        old_change = FileChangeEvent(
            file_path="/path/old.py",
            change_type=FileChangeType.MODIFIED,
            timestamp=now - timedelta(minutes=10)
        )
        new_change = FileChangeEvent(
            file_path="/path/new.py",
            change_type=FileChangeType.CREATED,
            timestamp=now - timedelta(minutes=1)
        )
        
        monitor._change_history.extend([old_change, new_change])
        
        # Get changes since 5 minutes ago
        since = now - timedelta(minutes=5)
        recent_changes = monitor.get_changes_since(since)
        
        assert len(recent_changes) == 1
        assert recent_changes[0] == new_change
    
    def test_get_changes_since_project(self):
        """Test getting changes for specific project since timestamp."""
        monitor = FileMonitor()
        
        with tempfile.TemporaryDirectory() as temp_dir1:
            with tempfile.TemporaryDirectory() as temp_dir2:
                project_id1 = monitor.add_project("project1", temp_dir1)
                project_id2 = monitor.add_project("project2", temp_dir2)
                
                now = datetime.utcnow()
                
                # Create changes for different projects
                change1 = FileChangeEvent(
                    file_path=str(Path(temp_dir1) / "file1.py"),
                    change_type=FileChangeType.MODIFIED,
                    timestamp=now - timedelta(minutes=1)
                )
                change2 = FileChangeEvent(
                    file_path=str(Path(temp_dir2) / "file2.py"),
                    change_type=FileChangeType.CREATED,
                    timestamp=now - timedelta(minutes=1)
                )
                
                monitor._change_history.extend([change1, change2])
                
                # Get changes for project1 only
                since = now - timedelta(minutes=5)
                project1_changes = monitor.get_changes_since(since, project_id=project_id1)
                
                assert len(project1_changes) == 1
                assert project1_changes[0] == change1
    
    def test_clear_change_history(self):
        """Test clearing change history."""
        monitor = FileMonitor()
        
        # Add some changes
        change = FileChangeEvent(
            file_path="/path/file.py",
            change_type=FileChangeType.MODIFIED,
            timestamp=datetime.utcnow()
        )
        monitor._change_history.append(change)
        
        assert len(monitor._change_history) == 1
        
        monitor.clear_change_history()
        
        assert len(monitor._change_history) == 0
    
    def test_get_statistics(self):
        """Test getting monitoring statistics."""
        monitor = FileMonitor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_id = monitor.add_project("test_project", temp_dir)
            
            # Add some mock changes
            changes = [
                FileChangeEvent(
                    file_path="/path/file1.py",
                    change_type=FileChangeType.CREATED,
                    timestamp=datetime.utcnow()
                ),
                FileChangeEvent(
                    file_path="/path/file2.py",
                    change_type=FileChangeType.MODIFIED,
                    timestamp=datetime.utcnow()
                )
            ]
            monitor._change_history.extend(changes)
            
            stats = monitor.get_statistics()
            
            assert stats['total_projects'] == 1
            assert stats['total_changes'] == 2
            assert stats['changes_by_type']['created'] == 1
            assert stats['changes_by_type']['modified'] == 1
            assert stats['is_running'] is False
            assert 'uptime' in stats
    
    @pytest.mark.asyncio
    async def test_monitoring_error_handling(self):
        """Test error handling during monitoring."""
        monitor = FileMonitor(poll_interval=0.1)
        
        # Add project with non-existent directory
        project_id = monitor.add_project("test_project", "/nonexistent/directory")
        
        # Mock _scan_directory to raise exception
        with patch.object(monitor, '_scan_directory', side_effect=Exception("Scan error")):
            await monitor.start_monitoring()
            
            # Let it run briefly - should handle error gracefully
            await asyncio.sleep(0.2)
            
            await monitor.stop_monitoring()
            
            # Monitor should still be functional
            assert monitor._running is False
    
    def test_file_state_tracking(self):
        """Test file state tracking for change detection."""
        monitor = FileMonitor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            project_id = monitor.add_project("test_project", temp_dir)
            
            # Create test file
            test_file = temp_path / "test.py"
            test_file.write_text("original content")
            
            # Initial scan
            files = monitor._scan_directory(
                temp_dir,
                include_patterns=["**/*"],
                exclude_patterns=[]
            )
            
            # Update file states
            for file_path in files:
                file_hash = monitor._get_file_hash(file_path)
                if file_hash:
                    monitor._file_states[file_path] = {
                        'hash': file_hash,
                        'last_modified': Path(file_path).stat().st_mtime
                    }
            
            # Modify file
            test_file.write_text("modified content")
            
            # Check if file is detected as changed
            current_hash = monitor._get_file_hash(str(test_file))
            stored_state = monitor._file_states.get(str(test_file))
            
            assert stored_state is not None
            assert current_hash != stored_state['hash']
    
    @pytest.mark.asyncio
    async def test_concurrent_monitoring(self):
        """Test monitoring multiple projects concurrently."""
        monitor = FileMonitor(poll_interval=0.1)
        
        with tempfile.TemporaryDirectory() as temp_dir1:
            with tempfile.TemporaryDirectory() as temp_dir2:
                project_id1 = monitor.add_project("project1", temp_dir1)
                project_id2 = monitor.add_project("project2", temp_dir2)
                
                await monitor.start_monitoring()
                
                # Let monitoring run
                await asyncio.sleep(0.2)
                
                # Both projects should be monitored
                assert project_id1 in monitor.projects
                assert project_id2 in monitor.projects
                
                await monitor.stop_monitoring()
                
                assert monitor._running is False