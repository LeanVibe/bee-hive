"""
Comprehensive Unit Tests for Project Index Database Models

Tests all database models with full coverage of functionality,
validation, relationships, and edge cases.
"""

import uuid
import pytest
from datetime import datetime, timezone
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select

from app.models.project_index import (
    ProjectIndex, FileEntry, DependencyRelationship, AnalysisSession, IndexSnapshot,
    ProjectStatus, FileType, DependencyType, AnalysisSessionType, AnalysisStatus, SnapshotType
)


class TestProjectIndex:
    """Test ProjectIndex model functionality."""
    
    @pytest.mark.asyncio
    async def test_create_project_index(self, test_session):
        """Test creating a basic project index."""
        project = ProjectIndex(
            name="Test Project",
            description="A test project",
            root_path="/path/to/project",
            git_repository_url="https://github.com/test/repo.git",
            git_branch="main",
            git_commit_hash="a" * 40
        )
        
        test_session.add(project)
        await test_session.commit()
        await test_session.refresh(project)
        
        assert project.id is not None
        # In SQLite, UUID is stored as string; in PostgreSQL as UUID object
        assert isinstance(project.id, (uuid.UUID, str))
        if isinstance(project.id, str):
            # Verify it's a valid UUID string
            uuid.UUID(project.id)
        assert project.name == "Test Project"
        assert project.description == "A test project"
        assert project.root_path == "/path/to/project"
        assert project.status == ProjectStatus.INACTIVE  # Default status
        assert project.file_count == 0  # Default count
        assert project.dependency_count == 0  # Default count
        assert project.created_at is not None
        assert project.updated_at is not None
    
    @pytest.mark.asyncio
    async def test_project_index_defaults(self, test_session):
        """Test that defaults are properly set."""
        project = ProjectIndex(
            name="Minimal Project",
            root_path="/minimal"
        )
        
        test_session.add(project)
        await test_session.commit()
        await test_session.refresh(project)
        
        assert project.status == ProjectStatus.INACTIVE
        assert project.configuration == {}
        assert project.analysis_settings == {}
        assert project.file_patterns == {}
        assert project.ignore_patterns == {}
        assert project.meta_data == {}
        assert project.file_count == 0
        assert project.dependency_count == 0
    
    @pytest.mark.asyncio
    async def test_project_index_with_configuration(self, test_session):
        """Test project with complex configuration."""
        config = {
            "languages": ["python", "javascript"],
            "analysis_depth": 3,
            "enable_ai_analysis": True
        }
        
        analysis_settings = {
            "extract_functions": True,
            "extract_classes": True,
            "analyze_imports": True
        }
        
        file_patterns = {
            "include": ["**/*.py", "**/*.js"],
            "exclude": ["**/test_*"]
        }
        
        project = ProjectIndex(
            name="Complex Project",
            root_path="/complex",
            configuration=config,
            analysis_settings=analysis_settings,
            file_patterns=file_patterns,
            status=ProjectStatus.ACTIVE
        )
        
        test_session.add(project)
        await test_session.commit()
        await test_session.refresh(project)
        
        assert project.configuration == config
        assert project.analysis_settings == analysis_settings
        assert project.file_patterns == file_patterns
        assert project.status == ProjectStatus.ACTIVE
    
    @pytest.mark.asyncio
    async def test_project_index_validation(self, test_session):
        """Test validation constraints."""
        # Test required fields
        with pytest.raises(IntegrityError):
            project = ProjectIndex()  # Missing required fields
            test_session.add(project)
            await test_session.commit()
    
    def test_project_index_to_dict(self):
        """Test to_dict serialization."""
        project = ProjectIndex(
            name="Serialization Test",
            root_path="/test",
            description="Test serialization",
            status=ProjectStatus.ACTIVE,
            file_count=10,
            dependency_count=25
        )
        
        data = project.to_dict()
        
        assert data["name"] == "Serialization Test"
        assert data["root_path"] == "/test"
        assert data["description"] == "Test serialization"
        assert data["status"] == ProjectStatus.ACTIVE.value
        assert data["file_count"] == 10
        assert data["dependency_count"] == 25
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_project_index_repr(self):
        """Test string representation."""
        project = ProjectIndex(
            name="Repr Test",
            root_path="/test",
            status=ProjectStatus.ANALYZING
        )
        project.id = uuid.uuid4()
        
        repr_str = repr(project)
        assert "ProjectIndex" in repr_str
        assert project.name in repr_str
        assert str(project.id) in repr_str
        # Check for either the enum value or the enum name
        assert (project.status.value in repr_str or project.status.name in repr_str)


class TestFileEntry:
    """Test FileEntry model functionality."""
    
    @pytest.mark.asyncio
    async def test_create_file_entry(self, test_session, test_project):
        """Test creating a basic file entry."""
        file_entry = FileEntry(
            project_id=test_project.id,
            file_path="/path/to/project/src/main.py",
            relative_path="src/main.py",
            file_name="main.py",
            file_extension=".py",
            file_type=FileType.SOURCE,
            language="python",
            file_size=1024,
            line_count=50,
            sha256_hash="a" * 64
        )
        
        test_session.add(file_entry)
        await test_session.commit()
        await test_session.refresh(file_entry)
        
        assert file_entry.id is not None
        assert file_entry.project_id == test_project.id
        assert file_entry.file_name == "main.py"
        assert file_entry.file_type == FileType.SOURCE
        assert file_entry.language == "python"
        assert file_entry.file_size == 1024
        assert file_entry.line_count == 50
        assert file_entry.encoding == "utf-8"  # Default encoding
        assert not file_entry.is_binary  # Default is False
        assert not file_entry.is_generated  # Default is False
    
    @pytest.mark.asyncio
    async def test_file_entry_with_analysis(self, test_session, test_project):
        """Test file entry with analysis data."""
        analysis_data = {
            "functions": ["main", "helper"],
            "classes": ["DataProcessor"],
            "imports": ["os", "sys", "json"],
            "complexity": 5,
            "maintainability": 8.5
        }
        
        metadata = {
            "last_analyzed": datetime.now(timezone.utc).isoformat(),
            "analyzer_version": "1.0.0"
        }
        
        tags = ["core", "entry-point", "python"]
        
        file_entry = FileEntry(
            project_id=test_project.id,
            file_path="/path/to/project/src/complex.py",
            relative_path="src/complex.py",
            file_name="complex.py",
            file_extension=".py",
            file_type=FileType.SOURCE,
            language="python",
            analysis_data=analysis_data,
            meta_data=metadata,
            tags=tags,
            content_preview="import os\nimport sys\n\ndef main():"
        )
        
        test_session.add(file_entry)
        await test_session.commit()
        await test_session.refresh(file_entry)
        
        assert file_entry.analysis_data == analysis_data
        assert file_entry.meta_data == metadata
        assert file_entry.tags == tags
        assert "import os" in file_entry.content_preview
    
    @pytest.mark.asyncio
    async def test_file_entry_binary_file(self, test_session, test_project):
        """Test binary file entry."""
        file_entry = FileEntry(
            project_id=test_project.id,
            file_path="/path/to/project/assets/image.png",
            relative_path="assets/image.png",
            file_name="image.png",
            file_extension=".png",
            file_type=FileType.OTHER,
            file_size=102400,  # 100KB
            is_binary=True,
            encoding=None
        )
        
        test_session.add(file_entry)
        await test_session.commit()
        await test_session.refresh(file_entry)
        
        assert file_entry.is_binary
        assert file_entry.language is None
        assert file_entry.line_count is None
        assert file_entry.content_preview is None
    
    @pytest.mark.asyncio
    async def test_file_entry_defaults(self, test_session, test_project):
        """Test file entry defaults."""
        file_entry = FileEntry(
            project_id=test_project.id,
            file_path="/test.py",
            relative_path="test.py",
            file_name="test.py",
            file_type=FileType.SOURCE
        )
        
        test_session.add(file_entry)
        await test_session.commit()
        await test_session.refresh(file_entry)
        
        assert file_entry.analysis_data == {}
        assert file_entry.meta_data == {}
        assert file_entry.tags == []
        assert not file_entry.is_binary
        assert not file_entry.is_generated
        assert file_entry.encoding == "utf-8"
    
    def test_file_entry_to_dict(self, test_project):
        """Test file entry serialization."""
        file_entry = FileEntry(
            project_id=test_project.id,
            file_path="/test.py",
            relative_path="test.py",
            file_name="test.py",
            file_type=FileType.SOURCE,
            language="python",
            file_size=500,
            line_count=25
        )
        
        data = file_entry.to_dict()
        
        assert data["project_id"] == str(test_project.id)
        assert data["file_name"] == "test.py"
        assert data["file_type"] == FileType.SOURCE.value
        assert data["language"] == "python"
        assert data["file_size"] == 500
        assert data["line_count"] == 25


class TestDependencyRelationship:
    """Test DependencyRelationship model functionality."""
    
    @pytest.mark.asyncio
    async def test_create_internal_dependency(self, test_session, test_files):
        """Test creating internal dependency relationship."""
        main_file = test_files[0]
        utils_file = test_files[1]
        
        dependency = DependencyRelationship(
            project_id=main_file.project_id,
            source_file_id=main_file.id,
            target_file_id=utils_file.id,
            target_name="utils",
            dependency_type=DependencyType.IMPORT,
            line_number=1,
            source_text="from utils import helper",
            is_external=False,
            confidence_score=1.0
        )
        
        test_session.add(dependency)
        await test_session.commit()
        await test_session.refresh(dependency)
        
        assert dependency.id is not None
        assert dependency.source_file_id == main_file.id
        assert dependency.target_file_id == utils_file.id
        assert dependency.target_name == "utils"
        assert dependency.dependency_type == DependencyType.IMPORT
        assert not dependency.is_external
        assert dependency.confidence_score == 1.0
    
    @pytest.mark.asyncio
    async def test_create_external_dependency(self, test_session, test_files):
        """Test creating external dependency relationship."""
        source_file = test_files[0]
        
        dependency = DependencyRelationship(
            project_id=source_file.project_id,
            source_file_id=source_file.id,
            target_file_id=None,  # External dependency
            target_path="numpy",
            target_name="numpy",
            dependency_type=DependencyType.IMPORT,
            line_number=2,
            source_text="import numpy as np",
            is_external=True,
            confidence_score=0.95
        )
        
        test_session.add(dependency)
        await test_session.commit()
        await test_session.refresh(dependency)
        
        assert dependency.target_file_id is None
        assert dependency.target_path == "numpy"
        assert dependency.target_name == "numpy"
        assert dependency.is_external
        assert dependency.confidence_score == 0.95
    
    @pytest.mark.asyncio
    async def test_dependency_types(self, test_session, test_files):
        """Test different dependency types."""
        source_file = test_files[0]
        target_file = test_files[1]
        
        dependency_types = [
            (DependencyType.IMPORT, "import utils"),
            (DependencyType.REQUIRE, "require('./utils')"),
            (DependencyType.INCLUDE, "#include <stdio.h>"),
            (DependencyType.EXTENDS, "class Child extends Parent"),
            (DependencyType.IMPLEMENTS, "class Service implements Interface"),
            (DependencyType.CALLS, "helper_function()"),
            (DependencyType.REFERENCES, "config.DATABASE_URL")
        ]
        
        dependencies = []
        for dep_type, source_text in dependency_types:
            dependency = DependencyRelationship(
                project_id=source_file.project_id,
                source_file_id=source_file.id,
                target_file_id=target_file.id,
                target_name=f"target_{dep_type.value}",
                dependency_type=dep_type,
                source_text=source_text
            )
            
            test_session.add(dependency)
            dependencies.append(dependency)
        
        await test_session.commit()
        
        # Verify all dependency types were created
        for i, dependency in enumerate(dependencies):
            await test_session.refresh(dependency)
            assert dependency.dependency_type == dependency_types[i][0]
    
    @pytest.mark.asyncio
    async def test_dependency_defaults(self, test_session, test_files):
        """Test dependency defaults."""
        source_file = test_files[0]
        
        dependency = DependencyRelationship(
            project_id=source_file.project_id,
            source_file_id=source_file.id,
            target_name="test_target",
            dependency_type=DependencyType.IMPORT
        )
        
        test_session.add(dependency)
        await test_session.commit()
        await test_session.refresh(dependency)
        
        assert dependency.meta_data == {}
        assert not dependency.is_external
        assert not dependency.is_dynamic
        assert dependency.confidence_score == 1.0
    
    def test_dependency_to_dict(self, test_files):
        """Test dependency serialization."""
        source_file = test_files[0]
        target_file = test_files[1]
        
        dependency = DependencyRelationship(
            project_id=source_file.project_id,
            source_file_id=source_file.id,
            target_file_id=target_file.id,
            target_name="utils",
            dependency_type=DependencyType.IMPORT,
            line_number=1,
            is_external=False,
            confidence_score=0.9
        )
        
        data = dependency.to_dict()
        
        assert data["project_id"] == str(source_file.project_id)
        assert data["source_file_id"] == str(source_file.id)
        assert data["target_file_id"] == str(target_file.id)
        assert data["target_name"] == "utils"
        assert data["dependency_type"] == DependencyType.IMPORT.value
        assert data["line_number"] == 1
        assert not data["is_external"]
        assert data["confidence_score"] == 0.9


class TestAnalysisSession:
    """Test AnalysisSession model functionality."""
    
    @pytest.mark.asyncio
    async def test_create_analysis_session(self, test_session, test_project):
        """Test creating analysis session."""
        session = AnalysisSession(
            project_id=test_project.id,
            session_name="Initial Analysis",
            session_type=AnalysisSessionType.FULL_ANALYSIS,
            files_total=100,
            configuration={"depth": 3}
        )
        
        test_session.add(session)
        await test_session.commit()
        await test_session.refresh(session)
        
        assert session.id is not None
        assert session.project_id == test_project.id
        assert session.session_name == "Initial Analysis"
        assert session.session_type == AnalysisSessionType.FULL_ANALYSIS
        assert session.status == AnalysisStatus.PENDING  # Default status
        assert session.progress_percentage == 0.0  # Default progress
        assert session.files_total == 100
        assert session.configuration == {"depth": 3}
    
    @pytest.mark.asyncio
    async def test_analysis_session_defaults(self, test_session, test_project):
        """Test analysis session defaults."""
        session = AnalysisSession(
            project_id=test_project.id,
            session_name="Test Session",
            session_type=AnalysisSessionType.INCREMENTAL
        )
        
        test_session.add(session)
        await test_session.commit()
        await test_session.refresh(session)
        
        assert session.status == AnalysisStatus.PENDING
        assert session.progress_percentage == 0.0
        assert session.files_processed == 0
        assert session.files_total == 0
        assert session.dependencies_found == 0
        assert session.errors_count == 0
        assert session.warnings_count == 0
        assert session.session_data == {}
        assert session.error_log == []
        assert session.performance_metrics == {}
        assert session.configuration == {}
        assert session.result_data == {}
    
    @pytest.mark.asyncio
    async def test_analysis_session_lifecycle(self, test_session, test_project):
        """Test analysis session lifecycle methods."""
        session = AnalysisSession(
            project_id=test_project.id,
            session_name="Lifecycle Test",
            session_type=AnalysisSessionType.CONTEXT_OPTIMIZATION,
            files_total=10
        )
        
        test_session.add(session)
        await test_session.commit()
        await test_session.refresh(session)
        
        # Test start_session
        session.start_session()
        assert session.status == AnalysisStatus.RUNNING
        assert session.started_at is not None
        
        # Test update_progress
        session.update_progress(50.0, "Processing files")
        assert session.progress_percentage == 50.0
        assert session.current_phase == "Processing files"
        
        # Test complete_session
        result_data = {"files_analyzed": 10, "dependencies_found": 25}
        session.complete_session(result_data)
        assert session.status == AnalysisStatus.COMPLETED
        assert session.completed_at is not None
        assert session.progress_percentage == 100.0
        assert session.result_data == result_data
    
    @pytest.mark.asyncio
    async def test_analysis_session_error_handling(self, test_session, test_project):
        """Test analysis session error handling."""
        session = AnalysisSession(
            project_id=test_project.id,
            session_name="Error Test",
            session_type=AnalysisSessionType.DEPENDENCY_MAPPING
        )
        
        test_session.add(session)
        await test_session.commit()
        await test_session.refresh(session)
        
        # Test add_error
        session.add_error("Warning: file too large", "warning")
        assert session.warnings_count == 1
        assert len(session.error_log) == 1
        assert session.error_log[0]["level"] == "warning"
        
        session.add_error("Error: cannot parse file", "error")
        assert session.errors_count == 1
        assert len(session.error_log) == 2
        
        # Test fail_session
        session.fail_session("Critical error occurred")
        assert session.status == AnalysisStatus.FAILED
        assert session.completed_at is not None
        assert session.errors_count == 2  # One more error added
        assert len(session.error_log) == 3
        assert session.error_log[-1]["level"] == "fatal"
    
    def test_analysis_session_to_dict(self, test_project):
        """Test analysis session serialization."""
        session = AnalysisSession(
            project_id=test_project.id,
            session_name="Serialization Test",
            session_type=AnalysisSessionType.FILE_SCANNING,
            status=AnalysisStatus.RUNNING,
            progress_percentage=75.0,
            files_processed=30,
            files_total=40
        )
        
        data = session.to_dict()
        
        assert data["project_id"] == str(test_project.id)
        assert data["session_name"] == "Serialization Test"
        assert data["session_type"] == AnalysisSessionType.FILE_SCANNING.value
        assert data["status"] == AnalysisStatus.RUNNING.value
        assert data["progress_percentage"] == 75.0
        assert data["files_processed"] == 30
        assert data["files_total"] == 40


class TestIndexSnapshot:
    """Test IndexSnapshot model functionality."""
    
    @pytest.mark.asyncio
    async def test_create_index_snapshot(self, test_session, test_project):
        """Test creating index snapshot."""
        snapshot = IndexSnapshot(
            project_id=test_project.id,
            snapshot_name="Initial Snapshot",
            description="Snapshot after initial analysis",
            snapshot_type=SnapshotType.MANUAL,
            git_commit_hash="b" * 40,
            git_branch="main",
            file_count=50,
            dependency_count=120
        )
        
        test_session.add(snapshot)
        await test_session.commit()
        await test_session.refresh(snapshot)
        
        assert snapshot.id is not None
        assert snapshot.project_id == test_project.id
        assert snapshot.snapshot_name == "Initial Snapshot"
        assert snapshot.snapshot_type == SnapshotType.MANUAL
        assert snapshot.git_commit_hash == "b" * 40
        assert snapshot.file_count == 50
        assert snapshot.dependency_count == 120
    
    @pytest.mark.asyncio
    async def test_snapshot_with_changes(self, test_session, test_project):
        """Test snapshot with change tracking."""
        changes = {
            "files_added": ["new_module.py", "helper.py"],
            "files_modified": ["main.py"],
            "files_deleted": ["old_script.py"],
            "dependencies_added": 5,
            "dependencies_removed": 2
        }
        
        metrics = {
            "analysis_duration": 120.5,
            "files_per_second": 2.3,
            "memory_usage_mb": 256
        }
        
        metadata = {
            "analyzer_version": "2.0.0",
            "created_by": "automated_system"
        }
        
        snapshot = IndexSnapshot(
            project_id=test_project.id,
            snapshot_name="Change Tracking",
            snapshot_type=SnapshotType.SCHEDULED,
            file_count=55,
            dependency_count=123,
            changes_since_last=changes,
            analysis_metrics=metrics,
            meta_data=metadata,
            data_checksum="checksum123"
        )
        
        test_session.add(snapshot)
        await test_session.commit()
        await test_session.refresh(snapshot)
        
        assert snapshot.changes_since_last == changes
        assert snapshot.analysis_metrics == metrics
        assert snapshot.meta_data == metadata
        assert snapshot.data_checksum == "checksum123"
    
    @pytest.mark.asyncio
    async def test_snapshot_defaults(self, test_session, test_project):
        """Test snapshot defaults."""
        snapshot = IndexSnapshot(
            project_id=test_project.id,
            snapshot_name="Default Test",
            snapshot_type=SnapshotType.GIT_COMMIT
        )
        
        test_session.add(snapshot)
        await test_session.commit()
        await test_session.refresh(snapshot)
        
        assert snapshot.file_count == 0
        assert snapshot.dependency_count == 0
        assert snapshot.changes_since_last == {}
        assert snapshot.analysis_metrics == {}
        assert snapshot.meta_data == {}
    
    def test_snapshot_to_dict(self, test_project):
        """Test snapshot serialization."""
        snapshot = IndexSnapshot(
            project_id=test_project.id,
            snapshot_name="Serialization Test",
            snapshot_type=SnapshotType.PRE_ANALYSIS,
            file_count=100,
            dependency_count=250,
            git_commit_hash="c" * 40
        )
        
        data = snapshot.to_dict()
        
        assert data["project_id"] == str(test_project.id)
        assert data["snapshot_name"] == "Serialization Test"
        assert data["snapshot_type"] == SnapshotType.PRE_ANALYSIS.value
        assert data["file_count"] == 100
        assert data["dependency_count"] == 250
        assert data["git_commit_hash"] == "c" * 40


class TestModelRelationships:
    """Test relationships between models."""
    
    @pytest.mark.asyncio
    async def test_project_file_relationship(self, test_session, test_project, test_files):
        """Test project to files relationship."""
        # Test relationship by querying files for this project
        files_stmt = select(FileEntry).where(FileEntry.project_id == test_project.id)
        files_result = await test_session.execute(files_stmt)
        files = files_result.scalars().all()
        
        assert len(files) == len(test_files)
        
        for file_entry in files:
            assert file_entry.project_id == test_project.id
    
    @pytest.mark.asyncio
    async def test_file_dependency_relationship(self, test_session, test_dependencies):
        """Test file to dependency relationship."""
        for dependency in test_dependencies:
            await test_session.refresh(dependency)
            
            # Test source file relationship by querying
            source_file_stmt = select(FileEntry).where(FileEntry.id == dependency.source_file_id)
            source_file_result = await test_session.execute(source_file_stmt)
            source_file = source_file_result.scalar_one()
            assert source_file is not None
            
            # Test outgoing dependencies by querying
            outgoing_deps_stmt = select(DependencyRelationship).where(
                DependencyRelationship.source_file_id == source_file.id
            )
            outgoing_deps_result = await test_session.execute(outgoing_deps_stmt)
            outgoing_deps = outgoing_deps_result.scalars().all()
            dependency_ids = [dep.id for dep in outgoing_deps]
            assert dependency.id in dependency_ids
            
            # Test target file relationship (if internal)
            if dependency.target_file_id:
                target_file_stmt = select(FileEntry).where(FileEntry.id == dependency.target_file_id)
                target_file_result = await test_session.execute(target_file_stmt)
                target_file = target_file_result.scalar_one()
                assert target_file is not None
                
                # Test incoming dependencies by querying
                incoming_deps_stmt = select(DependencyRelationship).where(
                    DependencyRelationship.target_file_id == target_file.id
                )
                incoming_deps_result = await test_session.execute(incoming_deps_stmt)
                incoming_deps = incoming_deps_result.scalars().all()
                incoming_dependency_ids = [dep.id for dep in incoming_deps]
                assert dependency.id in incoming_dependency_ids
    
    @pytest.mark.asyncio
    async def test_cascade_delete(self, test_session, test_project):
        """Test cascade deletion of related records."""
        # Create a file entry
        file_entry = FileEntry(
            project_id=test_project.id,
            file_path="/test.py",
            relative_path="test.py",
            file_name="test.py",
            file_type=FileType.SOURCE
        )
        test_session.add(file_entry)
        await test_session.commit()
        await test_session.refresh(file_entry)
        
        # Create a dependency
        dependency = DependencyRelationship(
            project_id=test_project.id,
            source_file_id=file_entry.id,
            target_name="test_module",
            dependency_type=DependencyType.IMPORT
        )
        test_session.add(dependency)
        await test_session.commit()
        
        # Delete the project
        await test_session.delete(test_project)
        await test_session.commit()
        
        # Verify related records are deleted
        file_stmt = select(FileEntry).where(FileEntry.id == file_entry.id)
        file_result = await test_session.execute(file_stmt)
        assert file_result.scalar_one_or_none() is None
        
        dep_stmt = select(DependencyRelationship).where(DependencyRelationship.id == dependency.id)
        dep_result = await test_session.execute(dep_stmt)
        assert dep_result.scalar_one_or_none() is None


class TestModelValidation:
    """Test model validation and constraints."""
    
    @pytest.mark.asyncio
    async def test_enum_validation(self, test_session, test_project):
        """Test enum field validation."""
        # Test valid enum values
        file_entry = FileEntry(
            project_id=test_project.id,
            file_path="/test.py",
            relative_path="test.py",
            file_name="test.py",
            file_type=FileType.SOURCE  # Valid enum
        )
        test_session.add(file_entry)
        await test_session.commit()
        
        # Test invalid enum values would be caught by Pydantic/SQLAlchemy
        # during normal operation, but in tests we can verify enum types
        assert isinstance(file_entry.file_type, FileType)
        assert file_entry.file_type in FileType
    
    @pytest.mark.asyncio
    async def test_uuid_generation(self, test_session):
        """Test UUID field generation."""
        project = ProjectIndex(name="UUID Test", root_path="/test")
        
        test_session.add(project)
        await test_session.commit()
        await test_session.refresh(project)
        
        # ID should be generated when saved to database
        assert project.id is not None
        assert isinstance(project.id, (uuid.UUID, str))  # SQLite returns string
        
        # Should be unique for each instance
        project2 = ProjectIndex(name="UUID Test 2", root_path="/test2")
        test_session.add(project2)
        await test_session.commit()
        await test_session.refresh(project2)
        
        assert project.id != project2.id
    
    @pytest.mark.asyncio
    async def test_timestamp_generation(self, test_session):
        """Test automatic timestamp generation."""
        project = ProjectIndex(name="Timestamp Test", root_path="/test")
        
        test_session.add(project)
        await test_session.commit()
        await test_session.refresh(project)
        
        # Timestamps should be generated when saved to database
        assert project.created_at is not None
        assert project.updated_at is not None
        assert isinstance(project.created_at, datetime)
        assert isinstance(project.updated_at, datetime)