"""
Integration tests for Project Index file monitoring system.

Tests real file system monitoring, change detection, and callback
notifications with actual file operations and timing.
"""

import asyncio
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

from app.project_index.file_monitor import (
    FileMonitor, FileChangeType, FileChangeEvent
)


class TestFileMonitorIntegration:
    """Integration tests for file monitoring system."""
    
    @pytest.fixture
    def file_monitor(self):
        """Create FileMonitor instance for testing."""
        return FileMonitor(poll_interval=0.1)  # Fast polling for tests
    
    @pytest.fixture
    def temp_project(self):
        """Create temporary project directory for monitoring."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create initial project structure
            (temp_path / "src").mkdir()
            (temp_path / "src" / "main.py").write_text("print('Hello, World!')")
            (temp_path / "src" / "utils.py").write_text("def helper(): pass")
            
            (temp_path / "tests").mkdir()
            (temp_path / "tests" / "test_main.py").write_text("import unittest")
            
            (temp_path / "README.md").write_text("# Test Project")
            (temp_path / "requirements.txt").write_text("pytest==7.1.0")
            
            yield temp_path
    
    @pytest.mark.asyncio
    async def test_add_and_remove_project(self, file_monitor, temp_project):
        """Test adding and removing projects from monitoring."""
        # Add project
        success = await file_monitor.add_project("test_project", temp_project)
        assert success is True
        
        # Verify project is in monitored projects
        assert "test_project" in file_monitor._monitored_projects
        assert file_monitor._monitored_projects["test_project"] == temp_project
        
        # Verify file states were initialized
        assert "test_project" in file_monitor._file_states
        assert len(file_monitor._file_states["test_project"]) > 0
        
        # Remove project
        success = await file_monitor.remove_project("test_project")
        assert success is True
        
        # Verify project is removed
        assert "test_project" not in file_monitor._monitored_projects
        assert "test_project" not in file_monitor._file_states
    
    @pytest.mark.asyncio
    async def test_add_nonexistent_project(self, file_monitor):
        """Test adding project with non-existent path."""
        success = await file_monitor.add_project("bad_project", Path("/nonexistent/path"))
        assert success is False
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, file_monitor, temp_project):
        """Test starting and stopping monitoring."""
        # Add project
        await file_monitor.add_project("test_project", temp_project)
        
        # Monitoring should start automatically
        assert file_monitor._is_monitoring is True
        assert file_monitor._monitoring_task is not None
        
        # Stop monitoring
        success = await file_monitor.stop_monitoring()
        assert success is True
        assert file_monitor._is_monitoring is False
        assert file_monitor._monitoring_task is None
        
        # Start monitoring again
        success = await file_monitor.start_monitoring()
        assert success is True
        assert file_monitor._is_monitoring is True
    
    @pytest.mark.asyncio
    async def test_file_creation_detection(self, file_monitor, temp_project):
        """Test detection of new file creation."""
        changes_detected = []
        
        async def change_callback(event):
            changes_detected.append(event)
        
        file_monitor.add_change_callback(change_callback)
        
        # Add project (starts monitoring)
        await file_monitor.add_project("test_project", temp_project)
        
        # Wait for initial scan
        await asyncio.sleep(0.2)
        
        # Create new file
        new_file = temp_project / "new_module.py"
        new_file.write_text("def new_function(): pass")
        
        # Wait for detection
        await asyncio.sleep(0.3)
        
        # Stop monitoring
        await file_monitor.stop_monitoring()
        
        # Check if change was detected
        create_events = [e for e in changes_detected if e.change_type == FileChangeType.CREATED]
        assert len(create_events) >= 1
        
        # Verify event details
        found_event = None
        for event in create_events:
            if "new_module.py" in str(event.file_path):
                found_event = event
                break
        
        assert found_event is not None
        assert found_event.project_id == "test_project"
        assert found_event.change_type == FileChangeType.CREATED
    
    @pytest.mark.asyncio
    async def test_file_modification_detection(self, file_monitor, temp_project):
        """Test detection of file modifications."""
        changes_detected = []
        
        async def change_callback(event):
            changes_detected.append(event)
        
        file_monitor.add_change_callback(change_callback)
        
        # Add project
        await file_monitor.add_project("test_project", temp_project)
        
        # Wait for initial scan
        await asyncio.sleep(0.2)
        
        # Modify existing file
        main_file = temp_project / "src" / "main.py"
        original_content = main_file.read_text()
        main_file.write_text(original_content + "\n# Modified content")
        
        # Wait for detection
        await asyncio.sleep(0.3)
        
        # Stop monitoring
        await file_monitor.stop_monitoring()
        
        # Check if modification was detected
        modify_events = [e for e in changes_detected if e.change_type == FileChangeType.MODIFIED]
        assert len(modify_events) >= 1
        
        # Verify event details
        found_event = None
        for event in modify_events:
            if "main.py" in str(event.file_path):
                found_event = event
                break
        
        assert found_event is not None
        assert found_event.project_id == "test_project"
        assert found_event.change_type == FileChangeType.MODIFIED
    
    @pytest.mark.asyncio
    async def test_file_deletion_detection(self, file_monitor, temp_project):
        """Test detection of file deletions."""
        changes_detected = []
        
        async def change_callback(event):
            changes_detected.append(event)
        
        file_monitor.add_change_callback(change_callback)
        
        # Add project
        await file_monitor.add_project("test_project", temp_project)
        
        # Wait for initial scan
        await asyncio.sleep(0.2)
        
        # Delete existing file
        utils_file = temp_project / "src" / "utils.py"
        utils_file.unlink()
        
        # Wait for detection
        await asyncio.sleep(0.3)
        
        # Stop monitoring
        await file_monitor.stop_monitoring()
        
        # Check if deletion was detected
        delete_events = [e for e in changes_detected if e.change_type == FileChangeType.DELETED]
        assert len(delete_events) >= 1
        
        # Verify event details
        found_event = None
        for event in delete_events:
            if "utils.py" in str(event.file_path):
                found_event = event
                break
        
        assert found_event is not None
        assert found_event.project_id == "test_project"
        assert found_event.change_type == FileChangeType.DELETED
    
    @pytest.mark.asyncio
    async def test_multiple_file_changes(self, file_monitor, temp_project):
        """Test detection of multiple simultaneous file changes."""
        changes_detected = []
        
        async def change_callback(event):
            changes_detected.append(event)
        
        file_monitor.add_change_callback(change_callback)
        
        # Add project
        await file_monitor.add_project("test_project", temp_project)
        
        # Wait for initial scan
        await asyncio.sleep(0.2)
        
        # Make multiple changes
        # 1. Create new file
        (temp_project / "new_file.py").write_text("# New file")
        
        # 2. Modify existing file
        main_file = temp_project / "src" / "main.py"
        main_file.write_text(main_file.read_text() + "\n# Modified")
        
        # 3. Delete file
        (temp_project / "requirements.txt").unlink()
        
        # 4. Create file in subdirectory
        (temp_project / "src" / "another_module.py").write_text("# Another module")
        
        # Wait for detection
        await asyncio.sleep(0.5)
        
        # Stop monitoring
        await file_monitor.stop_monitoring()
        
        # Verify all change types were detected
        change_types = [e.change_type for e in changes_detected]
        assert FileChangeType.CREATED in change_types
        assert FileChangeType.MODIFIED in change_types
        assert FileChangeType.DELETED in change_types
        
        # Verify minimum number of changes
        assert len(changes_detected) >= 3
    
    @pytest.mark.asyncio
    async def test_ignore_patterns(self, file_monitor, temp_project):
        """Test file filtering with ignore patterns."""
        changes_detected = []
        
        async def change_callback(event):
            changes_detected.append(event)
        
        file_monitor.add_change_callback(change_callback)
        
        # Add custom ignore patterns
        file_monitor.add_ignore_pattern("*.tmp")
        file_monitor.add_ignore_pattern("*.log")
        file_monitor.add_ignore_pattern("__pycache__")
        
        # Add project
        await file_monitor.add_project("test_project", temp_project)
        
        # Wait for initial scan
        await asyncio.sleep(0.2)
        
        # Create files that should be ignored
        (temp_project / "temp_file.tmp").write_text("Temporary file")
        (temp_project / "debug.log").write_text("Log content")
        
        # Create __pycache__ directory
        cache_dir = temp_project / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "main.pyc").write_text("Bytecode")
        
        # Create file that should NOT be ignored
        (temp_project / "valid_file.py").write_text("Valid Python file")
        
        # Wait for detection
        await asyncio.sleep(0.3)
        
        # Stop monitoring
        await file_monitor.stop_monitoring()
        
        # Verify ignored files were not detected
        detected_files = [str(e.file_path) for e in changes_detected]
        assert not any("temp_file.tmp" in f for f in detected_files)
        assert not any("debug.log" in f for f in detected_files)
        assert not any("__pycache__" in f for f in detected_files)
        
        # Verify valid file was detected
        assert any("valid_file.py" in f for f in detected_files)
    
    @pytest.mark.asyncio
    async def test_get_changes_since(self, file_monitor, temp_project):
        """Test retrieving changes since specific timestamp."""
        # Add project
        await file_monitor.add_project("test_project", temp_project)
        
        # Wait for initial scan
        await asyncio.sleep(0.1)
        
        # Record timestamp
        checkpoint_time = datetime.utcnow()
        
        # Wait a bit
        await asyncio.sleep(0.1)
        
        # Make changes after checkpoint
        new_file = temp_project / "after_checkpoint.py"
        new_file.write_text("# Created after checkpoint")
        
        # Wait for file system to register the change
        await asyncio.sleep(0.1)
        
        # Get changes since checkpoint
        changes = await file_monitor.get_changes_since("test_project", checkpoint_time)
        
        # Stop monitoring
        await file_monitor.stop_monitoring()
        
        # Verify changes were detected
        assert len(changes) >= 1
        
        # Verify all changes are after checkpoint
        for change in changes:
            assert change.timestamp > checkpoint_time
        
        # Verify our specific file is included
        change_files = [str(c.file_path) for c in changes]
        assert any("after_checkpoint.py" in f for f in change_files)
    
    @pytest.mark.asyncio
    async def test_force_scan(self, file_monitor, temp_project):
        """Test manual force scan functionality."""
        # Add project but don't start automatic monitoring
        file_monitor._is_monitoring = False
        await file_monitor.add_project("test_project", temp_project)
        
        # Create new file
        new_file = temp_project / "manual_scan_test.py"
        new_file.write_text("# Manual scan test")
        
        # Force scan
        scan_stats = await file_monitor.force_scan("test_project")
        
        assert scan_stats['scanned_projects'] == 1
        assert scan_stats['changes_detected'] >= 1
        
        # Test scan all projects
        await file_monitor.add_project("test_project_2", temp_project)
        scan_stats = await file_monitor.force_scan()  # No specific project
        
        assert scan_stats['scanned_projects'] == 2
    
    @pytest.mark.asyncio
    async def test_multiple_callbacks(self, file_monitor, temp_project):
        """Test multiple change callbacks."""
        callback1_events = []
        callback2_events = []
        
        async def callback1(event):
            callback1_events.append(event)
        
        async def callback2(event):
            callback2_events.append(event)
        
        # Add both callbacks
        file_monitor.add_change_callback(callback1)
        file_monitor.add_change_callback(callback2)
        
        # Add project
        await file_monitor.add_project("test_project", temp_project)
        
        # Wait for initial scan
        await asyncio.sleep(0.2)
        
        # Create file
        (temp_project / "multi_callback_test.py").write_text("# Test")
        
        # Wait for detection
        await asyncio.sleep(0.3)
        
        # Stop monitoring
        await file_monitor.stop_monitoring()
        
        # Both callbacks should have received events
        assert len(callback1_events) >= 1
        assert len(callback2_events) >= 1
        
        # Events should be the same
        assert len(callback1_events) == len(callback2_events)
        
        # Remove one callback
        file_monitor.remove_change_callback(callback1)
        assert len(file_monitor._change_callbacks) == 1
    
    @pytest.mark.asyncio
    async def test_monitoring_statistics(self, file_monitor, temp_project):
        """Test monitoring statistics tracking."""
        # Get initial stats
        initial_stats = file_monitor.get_monitoring_stats()
        assert initial_stats['is_monitoring'] is False
        assert initial_stats['monitored_projects'] == 0
        
        # Add project (starts monitoring)
        await file_monitor.add_project("test_project", temp_project)
        
        # Check updated stats
        updated_stats = file_monitor.get_monitoring_stats()
        assert updated_stats['is_monitoring'] is True
        assert updated_stats['monitored_projects'] == 1
        assert updated_stats['poll_interval'] == 0.1
        assert 'total_scans' in updated_stats
        assert 'files_checked' in updated_stats
        
        # Let monitoring run and generate some stats
        await asyncio.sleep(0.3)
        
        # Create a file to trigger changes
        (temp_project / "stats_test.py").write_text("# Stats test")
        
        # Wait for detection
        await asyncio.sleep(0.3)
        
        # Stop monitoring
        await file_monitor.stop_monitoring()
        
        # Get final stats
        final_stats = file_monitor.get_monitoring_stats()
        assert final_stats['total_scans'] > initial_stats['total_scans']
        assert final_stats['files_checked'] > initial_stats['files_checked']
    
    @pytest.mark.asyncio
    async def test_context_manager(self, file_monitor, temp_project):
        """Test file monitor as async context manager."""
        changes_detected = []
        
        async def change_callback(event):
            changes_detected.append(event)
        
        file_monitor.add_change_callback(change_callback)
        
        # Use as context manager
        async with file_monitor:
            # Add project
            await file_monitor.add_project("test_project", temp_project)
            
            # Wait for initial scan
            await asyncio.sleep(0.2)
            
            # Create file
            (temp_project / "context_test.py").write_text("# Context test")
            
            # Wait for detection
            await asyncio.sleep(0.3)
        
        # After exiting context, monitoring should be stopped
        assert file_monitor._is_monitoring is False
        
        # But changes should have been detected
        assert len(changes_detected) >= 1
    
    @pytest.mark.asyncio
    async def test_concurrent_monitoring(self, temp_project):
        """Test monitoring multiple projects concurrently."""
        with tempfile.TemporaryDirectory() as temp_dir2:
            temp_path2 = Path(temp_dir2)
            (temp_path2 / "file1.py").write_text("# Project 2 file")
            
            monitor = FileMonitor(poll_interval=0.1)
            changes_detected = []
            
            async def change_callback(event):
                changes_detected.append(event)
            
            monitor.add_change_callback(change_callback)
            
            # Add multiple projects
            await monitor.add_project("project1", temp_project)
            await monitor.add_project("project2", temp_path2)
            
            # Wait for initial scans
            await asyncio.sleep(0.2)
            
            # Make changes in both projects
            (temp_project / "project1_file.py").write_text("# Project 1 change")
            (temp_path2 / "project2_file.py").write_text("# Project 2 change")
            
            # Wait for detection
            await asyncio.sleep(0.3)
            
            # Stop monitoring
            await monitor.stop_monitoring()
            
            # Verify changes from both projects were detected
            project1_changes = [e for e in changes_detected if e.project_id == "project1"]
            project2_changes = [e for e in changes_detected if e.project_id == "project2"]
            
            assert len(project1_changes) >= 1
            assert len(project2_changes) >= 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self, file_monitor):
        """Test error handling in file monitoring."""
        # Test adding project with file instead of directory
        with tempfile.NamedTemporaryFile() as temp_file:
            success = await file_monitor.add_project("bad_project", Path(temp_file.name))
            assert success is False
        
        # Test removing non-existent project
        success = await file_monitor.remove_project("nonexistent_project")
        assert success is True  # Should not fail
        
        # Test getting changes for non-existent project
        changes = await file_monitor.get_changes_since("nonexistent", datetime.utcnow())
        assert changes == []
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self, file_monitor, temp_project):
        """Test error handling in change callbacks."""
        callback_errors = []
        
        async def failing_callback(event):
            raise Exception("Callback error")
        
        async def working_callback(event):
            callback_errors.append("working")
        
        # Add both callbacks
        file_monitor.add_change_callback(failing_callback)
        file_monitor.add_change_callback(working_callback)
        
        # Add project
        await file_monitor.add_project("test_project", temp_project)
        
        # Wait for initial scan
        await asyncio.sleep(0.2)
        
        # Create file to trigger callbacks
        (temp_project / "callback_test.py").write_text("# Test")
        
        # Wait for detection
        await asyncio.sleep(0.3)
        
        # Stop monitoring
        await file_monitor.stop_monitoring()
        
        # Working callback should still have worked despite failing callback
        assert "working" in callback_errors
    
    @pytest.mark.asyncio
    async def test_large_directory_monitoring(self, file_monitor):
        """Test monitoring large directory structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create large directory structure
            for i in range(10):
                subdir = temp_path / f"subdir_{i}"
                subdir.mkdir()
                for j in range(5):
                    file_path = subdir / f"file_{j}.py"
                    file_path.write_text(f"# File {i}-{j}")
            
            changes_detected = []
            
            async def change_callback(event):
                changes_detected.append(event)
            
            file_monitor.add_change_callback(change_callback)
            
            # Add large project
            await file_monitor.add_project("large_project", temp_path)
            
            # Wait for initial scan
            await asyncio.sleep(0.3)
            
            # Make some changes
            (temp_path / "subdir_0" / "new_file.py").write_text("# New file")
            (temp_path / "subdir_5" / "file_2.py").write_text("# Modified content")
            
            # Wait for detection
            await asyncio.sleep(0.5)
            
            # Stop monitoring
            await file_monitor.stop_monitoring()
            
            # Should handle large project efficiently
            stats = file_monitor.get_monitoring_stats()
            assert stats['files_checked'] >= 50  # Should have checked all files