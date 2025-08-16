"""
Unit tests for Project Index database models.

Tests for SQLAlchemy models including validation, relationships,
serialization, and model methods.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project_index import (
    ProjectIndex, FileEntry, DependencyRelationship, AnalysisSession, IndexSnapshot,
    ProjectStatus, FileType, DependencyType, AnalysisSessionType, AnalysisStatus, SnapshotType
)
from tests.project_index_conftest import (
    assert_project_index_valid, assert_file_entry_valid, assert_dependency_valid
)


@pytest.mark.project_index_models
@pytest.mark.unit
class TestProjectIndexModel:
    """Test ProjectIndex database model."""
    
    async def test_create_project_index_minimal(self, project_index_session: AsyncSession):
        """Test creating project index with minimal required fields."""
        project = ProjectIndex(
            name="Test Project",
            root_path="/test/path"
        )
        
        project_index_session.add(project)
        await project_index_session.commit()
        await project_index_session.refresh(project)
        
        assert_project_index_valid(project)
        assert project.name == "Test Project"
        assert project.root_path == "/test/path"
        assert project.status == ProjectStatus.INACTIVE
        assert project.file_count == 0
        assert project.dependency_count == 0
        assert project.configuration == {}
        assert project.analysis_settings == {}
        assert project.file_patterns == {}
        assert project.ignore_patterns == {}
        assert project.meta_data == {}
    
    async def test_create_project_index_full(self, project_index_session: AsyncSession):
        """Test creating project index with all fields."""
        project = ProjectIndex(
            name="Full Test Project",
            description="A comprehensive test project",
            root_path="/full/test/path",
            git_repository_url="https://github.com/test/project.git",
            git_branch="develop",
            git_commit_hash="abc123def456789",
            status=ProjectStatus.ACTIVE,
            configuration={
                "languages": ["python", "javascript"],
                "analysis_depth": 3
            },
            analysis_settings={
                "parse_ast": True,
                "extract_dependencies": True
            },
            file_patterns={
                "include": ["**/*.py", "**/*.js"],
                "exclude": ["__pycache__/**"]
            },
            ignore_patterns={
                "directories": [".git", "node_modules"],
                "files": ["*.pyc", "*.log"]
            },
            meta_data={
                "created_by": "test_user",
                "version": "1.0.0"
            },
            file_count=150,
            dependency_count=250
        )
        
        project_index_session.add(project)
        await project_index_session.commit()
        await project_index_session.refresh(project)
        
        assert_project_index_valid(project)
        assert project.description == "A comprehensive test project"
        assert project.git_repository_url == "https://github.com/test/project.git"
        assert project.git_branch == "develop"
        assert project.status == ProjectStatus.ACTIVE
        assert project.configuration["languages"] == ["python", "javascript"]
        assert project.file_count == 150
        assert project.dependency_count == 250
    
    async def test_project_index_defaults(self, project_index_session: AsyncSession):
        """Test that project index defaults are properly set."""
        project = ProjectIndex(
            name="Default Test",
            root_path="/default/path"
        )
        
        project_index_session.add(project)
        await project_index_session.commit()
        await project_index_session.refresh(project)
        
        # Check defaults from __init__
        assert project.status == ProjectStatus.INACTIVE
        assert project.configuration == {}
        assert project.analysis_settings == {}
        assert project.file_patterns == {}
        assert project.ignore_patterns == {}
        assert project.meta_data == {}
        assert project.file_count == 0
        assert project.dependency_count == 0
    
    async def test_project_index_to_dict(self, project_index_session: AsyncSession):
        """Test project index serialization to dictionary."""
        project = ProjectIndex(
            name="Serialization Test",
            root_path="/serial/path",
            description="Test serialization",
            status=ProjectStatus.ACTIVE,
            file_count=10,
            dependency_count=15
        )
        
        project_index_session.add(project)
        await project_index_session.commit()
        await project_index_session.refresh(project)
        
        project_dict = project.to_dict()
        
        assert isinstance(project_dict, dict)
        assert project_dict["name"] == "Serialization Test"
        assert project_dict["root_path"] == "/serial/path"
        assert project_dict["description"] == "Test serialization"
        assert project_dict["status"] == "active"
        assert project_dict["file_count"] == 10
        assert project_dict["dependency_count"] == 15
        assert "id" in project_dict
        assert "created_at" in project_dict
        assert "updated_at" in project_dict
    
    async def test_project_index_string_representation(self, project_index_session: AsyncSession):
        """Test project index string representation."""
        project = ProjectIndex(
            name="Repr Test",
            root_path="/repr/path"
        )
        
        project_index_session.add(project)
        await project_index_session.commit()
        await project_index_session.refresh(project)
        
        repr_str = repr(project)
        assert "ProjectIndex" in repr_str
        assert str(project.id) in repr_str
        assert "Repr Test" in repr_str
        assert "inactive" in repr_str.lower()
    
    async def test_project_index_unique_name_constraint(self, project_index_session: AsyncSession):
        """Test that project names should be unique (if enforced by application logic)."""
        # Create first project
        project1 = ProjectIndex(
            name="Unique Test",
            root_path="/path1"
        )
        project_index_session.add(project1)
        await project_index_session.commit()
        
        # Create second project with same name (should be allowed at DB level)
        project2 = ProjectIndex(
            name="Unique Test",
            root_path="/path2"
        )
        project_index_session.add(project2)
        await project_index_session.commit()  # This should work at DB level
        
        # Application logic should handle uniqueness if needed
        assert project1.name == project2.name
        assert project1.root_path != project2.root_path


@pytest.mark.project_index_models
@pytest.mark.unit
class TestFileEntryModel:
    """Test FileEntry database model."""
    
    async def test_create_file_entry_minimal(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex
    ):
        """Test creating file entry with minimal required fields."""
        file_entry = FileEntry(
            project_id=sample_project_index.id,
            file_path="/test/file.py",
            relative_path="file.py",
            file_name="file.py",
            file_type=FileType.SOURCE
        )
        
        project_index_session.add(file_entry)
        await project_index_session.commit()
        await project_index_session.refresh(file_entry)
        
        assert_file_entry_valid(file_entry)
        assert file_entry.project_id == sample_project_index.id
        assert file_entry.file_path == "/test/file.py"
        assert file_entry.relative_path == "file.py"
        assert file_entry.file_name == "file.py"
        assert file_entry.file_type == FileType.SOURCE
        assert file_entry.analysis_data == {}
        assert file_entry.meta_data == {}
        assert file_entry.tags == []
        assert file_entry.is_binary is False
        assert file_entry.is_generated is False
        assert file_entry.encoding == "utf-8"
    
    async def test_create_file_entry_full(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex
    ):
        """Test creating file entry with all fields."""
        file_entry = FileEntry(
            project_id=sample_project_index.id,
            file_path="/test/src/module.py",
            relative_path="src/module.py",
            file_name="module.py",
            file_extension=".py",
            file_type=FileType.SOURCE,
            language="python",
            encoding="utf-8",
            file_size=2048,
            line_count=75,
            sha256_hash="abc123def456",
            content_preview="import os\nimport sys\n\ndef main():",
            analysis_data={
                "functions": [{"name": "main", "line": 4}],
                "imports": [{"module": "os"}, {"module": "sys"}],
                "complexity": {"cyclomatic": 3}
            },
            meta_data={
                "analyzed_by": "test_analyzer",
                "analysis_version": "1.0"
            },
            tags=["main", "utility"],
            is_binary=False,
            is_generated=False,
            last_modified=datetime.utcnow() - timedelta(hours=1),
            indexed_at=datetime.utcnow()
        )
        
        project_index_session.add(file_entry)
        await project_index_session.commit()
        await project_index_session.refresh(file_entry)
        
        assert_file_entry_valid(file_entry)
        assert file_entry.file_extension == ".py"
        assert file_entry.language == "python"
        assert file_entry.file_size == 2048
        assert file_entry.line_count == 75
        assert file_entry.sha256_hash == "abc123def456"
        assert "import os" in file_entry.content_preview
        assert len(file_entry.analysis_data["functions"]) == 1
        assert len(file_entry.tags) == 2
        assert "main" in file_entry.tags
    
    async def test_file_entry_defaults(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex
    ):
        """Test file entry default values."""
        file_entry = FileEntry(
            project_id=sample_project_index.id,
            file_path="/test/default.py",
            relative_path="default.py",
            file_name="default.py",
            file_type=FileType.SOURCE
        )
        
        project_index_session.add(file_entry)
        await project_index_session.commit()
        await project_index_session.refresh(file_entry)
        
        # Check defaults from __init__
        assert file_entry.analysis_data == {}
        assert file_entry.meta_data == {}
        assert file_entry.tags == []
        assert file_entry.is_binary is False
        assert file_entry.is_generated is False
        assert file_entry.encoding == "utf-8"
    
    async def test_file_entry_to_dict(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex
    ):
        """Test file entry serialization to dictionary."""
        file_entry = FileEntry(
            project_id=sample_project_index.id,
            file_path="/test/serialize.py",
            relative_path="serialize.py",
            file_name="serialize.py",
            file_type=FileType.SOURCE,
            language="python",
            file_size=1024,
            line_count=50
        )
        
        project_index_session.add(file_entry)
        await project_index_session.commit()
        await project_index_session.refresh(file_entry)
        
        file_dict = file_entry.to_dict()
        
        assert isinstance(file_dict, dict)
        assert file_dict["file_path"] == "/test/serialize.py"
        assert file_dict["relative_path"] == "serialize.py"
        assert file_dict["file_name"] == "serialize.py"
        assert file_dict["file_type"] == "source"
        assert file_dict["language"] == "python"
        assert file_dict["file_size"] == 1024
        assert file_dict["line_count"] == 50
        assert "id" in file_dict
        assert "project_id" in file_dict
    
    async def test_file_entry_project_relationship(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex
    ):
        """Test file entry relationship to project."""
        file_entry = FileEntry(
            project_id=sample_project_index.id,
            file_path="/test/relation.py",
            relative_path="relation.py",
            file_name="relation.py",
            file_type=FileType.SOURCE
        )
        
        project_index_session.add(file_entry)
        await project_index_session.commit()
        await project_index_session.refresh(file_entry)
        
        # Test relationship access
        assert file_entry.project is not None
        assert file_entry.project.id == sample_project_index.id
        assert file_entry.project.name == sample_project_index.name
    
    async def test_file_entry_foreign_key_constraint(
        self, 
        project_index_session: AsyncSession
    ):
        """Test file entry foreign key constraint to project."""
        # Try to create file entry with non-existent project ID
        non_existent_project_id = uuid.uuid4()
        file_entry = FileEntry(
            project_id=non_existent_project_id,
            file_path="/test/orphan.py",
            relative_path="orphan.py",
            file_name="orphan.py",
            file_type=FileType.SOURCE
        )
        
        project_index_session.add(file_entry)
        
        # This should fail due to foreign key constraint
        with pytest.raises(IntegrityError):
            await project_index_session.commit()


@pytest.mark.project_index_models
@pytest.mark.unit
class TestDependencyRelationshipModel:
    """Test DependencyRelationship database model."""
    
    async def test_create_dependency_minimal(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex,
        sample_file_entries: list
    ):
        """Test creating dependency with minimal required fields."""
        source_file = sample_file_entries[0]
        
        dependency = DependencyRelationship(
            project_id=sample_project_index.id,
            source_file_id=source_file.id,
            target_name="os",
            dependency_type=DependencyType.IMPORT
        )
        
        project_index_session.add(dependency)
        await project_index_session.commit()
        await project_index_session.refresh(dependency)
        
        assert_dependency_valid(dependency)
        assert dependency.project_id == sample_project_index.id
        assert dependency.source_file_id == source_file.id
        assert dependency.target_name == "os"
        assert dependency.dependency_type == DependencyType.IMPORT
        assert dependency.is_external is False
        assert dependency.is_dynamic is False
        assert dependency.confidence_score == 1.0
        assert dependency.meta_data == {}
    
    async def test_create_dependency_full(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex,
        sample_file_entries: list
    ):
        """Test creating dependency with all fields."""
        source_file = sample_file_entries[0]
        target_file = sample_file_entries[1]
        
        dependency = DependencyRelationship(
            project_id=sample_project_index.id,
            source_file_id=source_file.id,
            target_file_id=target_file.id,
            target_path="/test/target.py",
            target_name="target_module",
            dependency_type=DependencyType.IMPORT,
            line_number=5,
            column_number=1,
            source_text="from target_module import function",
            is_external=False,
            is_dynamic=True,
            confidence_score=0.85,
            meta_data={
                "import_type": "relative",
                "detected_by": "ast_parser"
            }
        )
        
        project_index_session.add(dependency)
        await project_index_session.commit()
        await project_index_session.refresh(dependency)
        
        assert_dependency_valid(dependency)
        assert dependency.target_file_id == target_file.id
        assert dependency.target_path == "/test/target.py"
        assert dependency.line_number == 5
        assert dependency.column_number == 1
        assert dependency.source_text == "from target_module import function"
        assert dependency.is_external is False
        assert dependency.is_dynamic is True
        assert dependency.confidence_score == 0.85
        assert dependency.meta_data["import_type"] == "relative"
    
    async def test_dependency_defaults(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex,
        sample_file_entries: list
    ):
        """Test dependency default values."""
        source_file = sample_file_entries[0]
        
        dependency = DependencyRelationship(
            project_id=sample_project_index.id,
            source_file_id=source_file.id,
            target_name="json",
            dependency_type=DependencyType.IMPORT
        )
        
        project_index_session.add(dependency)
        await project_index_session.commit()
        await project_index_session.refresh(dependency)
        
        # Check defaults from __init__
        assert dependency.meta_data == {}
        assert dependency.is_external is False
        assert dependency.is_dynamic is False
        assert dependency.confidence_score == 1.0
    
    async def test_dependency_to_dict(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex,
        sample_file_entries: list
    ):
        """Test dependency serialization to dictionary."""
        source_file = sample_file_entries[0]
        
        dependency = DependencyRelationship(
            project_id=sample_project_index.id,
            source_file_id=source_file.id,
            target_name="requests",
            dependency_type=DependencyType.IMPORT,
            is_external=True,
            confidence_score=0.9
        )
        
        project_index_session.add(dependency)
        await project_index_session.commit()
        await project_index_session.refresh(dependency)
        
        dep_dict = dependency.to_dict()
        
        assert isinstance(dep_dict, dict)
        assert dep_dict["target_name"] == "requests"
        assert dep_dict["dependency_type"] == "import"
        assert dep_dict["is_external"] is True
        assert dep_dict["confidence_score"] == 0.9
        assert "id" in dep_dict
        assert "project_id" in dep_dict
        assert "source_file_id" in dep_dict
    
    async def test_dependency_relationships(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex,
        sample_file_entries: list
    ):
        """Test dependency relationships to project and files."""
        source_file = sample_file_entries[0]
        target_file = sample_file_entries[1]
        
        dependency = DependencyRelationship(
            project_id=sample_project_index.id,
            source_file_id=source_file.id,
            target_file_id=target_file.id,
            target_name="target_module",
            dependency_type=DependencyType.IMPORT
        )
        
        project_index_session.add(dependency)
        await project_index_session.commit()
        await project_index_session.refresh(dependency)
        
        # Test relationships
        assert dependency.project is not None
        assert dependency.project.id == sample_project_index.id
        assert dependency.source_file is not None
        assert dependency.source_file.id == source_file.id
        assert dependency.target_file is not None
        assert dependency.target_file.id == target_file.id


@pytest.mark.project_index_models
@pytest.mark.unit
class TestAnalysisSessionModel:
    """Test AnalysisSession database model."""
    
    async def test_create_analysis_session_minimal(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex
    ):
        """Test creating analysis session with minimal required fields."""
        session = AnalysisSession(
            project_id=sample_project_index.id,
            session_name="Test Analysis",
            session_type=AnalysisSessionType.FULL_ANALYSIS
        )
        
        project_index_session.add(session)
        await project_index_session.commit()
        await project_index_session.refresh(session)
        
        assert session.project_id == sample_project_index.id
        assert session.session_name == "Test Analysis"
        assert session.session_type == AnalysisSessionType.FULL_ANALYSIS
        assert session.status == AnalysisStatus.PENDING
        assert session.progress_percentage == 0.0
        assert session.files_processed == 0
        assert session.files_total == 0
        assert session.dependencies_found == 0
        assert session.errors_count == 0
        assert session.warnings_count == 0
    
    async def test_analysis_session_methods(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex
    ):
        """Test analysis session state management methods."""
        session = AnalysisSession(
            project_id=sample_project_index.id,
            session_name="Method Test",
            session_type=AnalysisSessionType.INCREMENTAL
        )
        
        project_index_session.add(session)
        await project_index_session.commit()
        await project_index_session.refresh(session)
        
        # Test start_session
        session.start_session()
        assert session.status == AnalysisStatus.RUNNING
        assert session.started_at is not None
        
        # Test update_progress
        session.update_progress(50.0, "analyzing_files")
        assert session.progress_percentage == 50.0
        assert session.current_phase == "analyzing_files"
        
        # Test complete_session
        result_data = {"files_analyzed": 10, "dependencies_found": 5}
        session.complete_session(result_data)
        assert session.status == AnalysisStatus.COMPLETED
        assert session.completed_at is not None
        assert session.progress_percentage == 100.0
        assert session.result_data == result_data
    
    async def test_analysis_session_error_handling(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex
    ):
        """Test analysis session error handling methods."""
        session = AnalysisSession(
            project_id=sample_project_index.id,
            session_name="Error Test",
            session_type=AnalysisSessionType.FULL_ANALYSIS
        )
        
        project_index_session.add(session)
        await project_index_session.commit()
        await project_index_session.refresh(session)
        
        # Test add_error
        session.add_error("Test warning", "warning")
        assert session.warnings_count == 1
        assert len(session.error_log) == 1
        assert session.error_log[0]["error"] == "Test warning"
        assert session.error_log[0]["level"] == "warning"
        
        session.add_error("Test error", "error")
        assert session.errors_count == 1
        assert session.warnings_count == 1
        assert len(session.error_log) == 2
        
        # Test fail_session
        session.fail_session("Critical failure")
        assert session.status == AnalysisStatus.FAILED
        assert session.completed_at is not None
        assert session.errors_count == 2  # Previous error + failure error
        assert len(session.error_log) == 3
        assert any(log["level"] == "fatal" for log in session.error_log)
    
    async def test_analysis_session_to_dict(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex
    ):
        """Test analysis session serialization to dictionary."""
        session = AnalysisSession(
            project_id=sample_project_index.id,
            session_name="Dict Test",
            session_type=AnalysisSessionType.CONTEXT_OPTIMIZATION,
            status=AnalysisStatus.RUNNING,
            progress_percentage=75.0,
            files_processed=15,
            files_total=20
        )
        
        project_index_session.add(session)
        await project_index_session.commit()
        await project_index_session.refresh(session)
        
        session_dict = session.to_dict()
        
        assert isinstance(session_dict, dict)
        assert session_dict["session_name"] == "Dict Test"
        assert session_dict["session_type"] == "context_optimization"
        assert session_dict["status"] == "running"
        assert session_dict["progress_percentage"] == 75.0
        assert session_dict["files_processed"] == 15
        assert session_dict["files_total"] == 20


@pytest.mark.project_index_models
@pytest.mark.unit
class TestIndexSnapshotModel:
    """Test IndexSnapshot database model."""
    
    async def test_create_index_snapshot_minimal(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex
    ):
        """Test creating index snapshot with minimal required fields."""
        snapshot = IndexSnapshot(
            project_id=sample_project_index.id,
            snapshot_name="Test Snapshot",
            snapshot_type=SnapshotType.MANUAL
        )
        
        project_index_session.add(snapshot)
        await project_index_session.commit()
        await project_index_session.refresh(snapshot)
        
        assert snapshot.project_id == sample_project_index.id
        assert snapshot.snapshot_name == "Test Snapshot"
        assert snapshot.snapshot_type == SnapshotType.MANUAL
        assert snapshot.file_count == 0
        assert snapshot.dependency_count == 0
        assert snapshot.changes_since_last == {}
        assert snapshot.analysis_metrics == {}
        assert snapshot.meta_data == {}
    
    async def test_index_snapshot_git_integration(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex
    ):
        """Test index snapshot with Git integration fields."""
        snapshot = IndexSnapshot(
            project_id=sample_project_index.id,
            snapshot_name="Git Snapshot",
            snapshot_type=SnapshotType.GIT_COMMIT,
            git_commit_hash="abc123def456",
            git_branch="feature/test",
            file_count=25,
            dependency_count=40,
            changes_since_last={
                "files_added": 3,
                "files_modified": 2,
                "files_deleted": 1
            },
            analysis_metrics={
                "avg_complexity": 2.5,
                "total_lines": 1500,
                "test_coverage": 85.0
            }
        )
        
        project_index_session.add(snapshot)
        await project_index_session.commit()
        await project_index_session.refresh(snapshot)
        
        assert snapshot.git_commit_hash == "abc123def456"
        assert snapshot.git_branch == "feature/test"
        assert snapshot.file_count == 25
        assert snapshot.dependency_count == 40
        assert snapshot.changes_since_last["files_added"] == 3
        assert snapshot.analysis_metrics["avg_complexity"] == 2.5
    
    async def test_index_snapshot_to_dict(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex
    ):
        """Test index snapshot serialization to dictionary."""
        snapshot = IndexSnapshot(
            project_id=sample_project_index.id,
            snapshot_name="Serialization Test",
            snapshot_type=SnapshotType.PRE_ANALYSIS,
            file_count=10,
            dependency_count=15
        )
        
        project_index_session.add(snapshot)
        await project_index_session.commit()
        await project_index_session.refresh(snapshot)
        
        snapshot_dict = snapshot.to_dict()
        
        assert isinstance(snapshot_dict, dict)
        assert snapshot_dict["snapshot_name"] == "Serialization Test"
        assert snapshot_dict["snapshot_type"] == "pre_analysis"
        assert snapshot_dict["file_count"] == 10
        assert snapshot_dict["dependency_count"] == 15
        assert "id" in snapshot_dict
        assert "project_id" in snapshot_dict
        assert "created_at" in snapshot_dict


@pytest.mark.project_index_models
@pytest.mark.unit
class TestModelRelationships:
    """Test model relationships and cascading deletes."""
    
    async def test_project_file_relationship(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex,
        sample_file_entries: list
    ):
        """Test project to file entries relationship."""
        # Refresh to load relationships
        await project_index_session.refresh(sample_project_index)
        
        # Access file entries through relationship
        assert len(sample_project_index.file_entries) == len(sample_file_entries)
        
        # Check that all file entries belong to the project
        for file_entry in sample_project_index.file_entries:
            assert file_entry.project_id == sample_project_index.id
    
    async def test_file_dependency_relationship(
        self, 
        project_index_session: AsyncSession, 
        sample_file_entries: list,
        sample_dependencies: list
    ):
        """Test file entry to dependencies relationship."""
        # Find a file that has dependencies
        main_file = next(f for f in sample_file_entries if f.file_name == "main.py")
        await project_index_session.refresh(main_file)
        
        # Check outgoing dependencies
        assert len(main_file.outgoing_dependencies) > 0
        
        for dep in main_file.outgoing_dependencies:
            assert dep.source_file_id == main_file.id
    
    async def test_cascading_delete_project(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex,
        sample_file_entries: list,
        sample_dependencies: list,
        sample_analysis_session: AnalysisSession,
        sample_index_snapshot: IndexSnapshot
    ):
        """Test cascading delete when project is deleted."""
        project_id = sample_project_index.id
        
        # Count related objects before deletion
        from sqlalchemy import select, func
        
        file_count = await project_index_session.scalar(
            select(func.count(FileEntry.id)).where(FileEntry.project_id == project_id)
        )
        dep_count = await project_index_session.scalar(
            select(func.count(DependencyRelationship.id)).where(DependencyRelationship.project_id == project_id)
        )
        session_count = await project_index_session.scalar(
            select(func.count(AnalysisSession.id)).where(AnalysisSession.project_id == project_id)
        )
        snapshot_count = await project_index_session.scalar(
            select(func.count(IndexSnapshot.id)).where(IndexSnapshot.project_id == project_id)
        )
        
        # Verify we have related objects
        assert file_count > 0
        assert dep_count > 0
        assert session_count > 0
        assert snapshot_count > 0
        
        # Delete the project
        await project_index_session.delete(sample_project_index)
        await project_index_session.commit()
        
        # Verify all related objects were deleted (cascade)
        remaining_files = await project_index_session.scalar(
            select(func.count(FileEntry.id)).where(FileEntry.project_id == project_id)
        )
        remaining_deps = await project_index_session.scalar(
            select(func.count(DependencyRelationship.id)).where(DependencyRelationship.project_id == project_id)
        )
        remaining_sessions = await project_index_session.scalar(
            select(func.count(AnalysisSession.id)).where(AnalysisSession.project_id == project_id)
        )
        remaining_snapshots = await project_index_session.scalar(
            select(func.count(IndexSnapshot.id)).where(IndexSnapshot.project_id == project_id)
        )
        
        assert remaining_files == 0
        assert remaining_deps == 0
        assert remaining_sessions == 0
        assert remaining_snapshots == 0


@pytest.mark.project_index_models
@pytest.mark.unit
class TestModelValidation:
    """Test model validation and constraints."""
    
    async def test_project_index_required_fields(self, project_index_session: AsyncSession):
        """Test that required fields are enforced."""
        # Test without name - should fail at database level or application validation
        with pytest.raises((IntegrityError, ValueError)):
            project = ProjectIndex(root_path="/test/path")
            project_index_session.add(project)
            await project_index_session.commit()
    
    async def test_file_entry_project_id_required(self, project_index_session: AsyncSession):
        """Test that project_id is required for file entries."""
        with pytest.raises(IntegrityError):
            file_entry = FileEntry(
                file_path="/test/file.py",
                relative_path="file.py",
                file_name="file.py",
                file_type=FileType.SOURCE
                # Missing project_id
            )
            project_index_session.add(file_entry)
            await project_index_session.commit()
    
    async def test_enum_validation(
        self, 
        project_index_session: AsyncSession, 
        sample_project_index: ProjectIndex
    ):
        """Test that enum values are properly validated."""
        # Valid enum value should work
        file_entry = FileEntry(
            project_id=sample_project_index.id,
            file_path="/test/valid.py",
            relative_path="valid.py",
            file_name="valid.py",
            file_type=FileType.SOURCE
        )
        
        project_index_session.add(file_entry)
        await project_index_session.commit()
        await project_index_session.refresh(file_entry)
        
        assert file_entry.file_type == FileType.SOURCE