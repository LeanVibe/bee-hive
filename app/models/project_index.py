"""
Project Index models for LeanVibe Agent Hive 2.0

Models for intelligent project analysis and context optimization system.
Supports code intelligence, dependency tracking, and analysis sessions.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from sqlalchemy import Column, String, Text, DateTime, JSON, Enum as SQLEnum, Integer, ForeignKey, BigInteger, Float, Boolean
from sqlalchemy.dialects.postgresql import ENUM, JSONB, ARRAY
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from ..core.database import Base
from ..core.database_types import DatabaseAgnosticUUID, StringArray


class ProjectStatus(Enum):
    """Project analysis status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    ANALYZING = "analyzing"
    FAILED = "failed"


class FileType(Enum):
    """Types of files in the project."""
    SOURCE = "source"
    CONFIG = "config"
    DOCUMENTATION = "documentation"
    TEST = "test"
    BUILD = "build"
    OTHER = "other"


class DependencyType(Enum):
    """Types of code dependencies."""
    IMPORT = "import"
    REQUIRE = "require"
    INCLUDE = "include"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    CALLS = "calls"
    REFERENCES = "references"


class SnapshotType(Enum):
    """Types of index snapshots."""
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    PRE_ANALYSIS = "pre_analysis"
    POST_ANALYSIS = "post_analysis"
    GIT_COMMIT = "git_commit"


class AnalysisSessionType(Enum):
    """Types of analysis sessions."""
    FULL_ANALYSIS = "full_analysis"
    INCREMENTAL = "incremental"
    CONTEXT_OPTIMIZATION = "context_optimization"
    DEPENDENCY_MAPPING = "dependency_mapping"
    FILE_SCANNING = "file_scanning"


class AnalysisStatus(Enum):
    """Analysis session status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ProjectIndex(Base):
    """
    Main project metadata and configuration table.
    
    Represents a project that is being analyzed for code intelligence
    and context optimization. Contains project-level settings and status.
    """
    
    __tablename__ = "project_indexes"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Project paths and repository information
    root_path = Column(String(1000), nullable=False, index=True)
    git_repository_url = Column(String(500), nullable=True, index=True)
    git_branch = Column(String(100), nullable=True, index=True)
    git_commit_hash = Column(String(40), nullable=True, index=True)
    
    # Project status and configuration
    status = Column(ENUM(ProjectStatus, name='project_status'), nullable=False, default=ProjectStatus.INACTIVE, index=True)
    configuration = Column(JSONB, nullable=True, default=dict)
    analysis_settings = Column(JSONB, nullable=True, default=dict)
    file_patterns = Column(JSONB, nullable=True, default=dict)
    ignore_patterns = Column(JSONB, nullable=True, default=dict)
    meta_data = Column("metadata", JSONB, nullable=True, default=dict)
    
    # Statistics
    file_count = Column(Integer, nullable=False, default=0)
    dependency_count = Column(Integer, nullable=False, default=0)
    
    # Timestamps
    last_indexed_at = Column(DateTime(timezone=True), nullable=True)
    last_analysis_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    file_entries = relationship("FileEntry", back_populates="project", cascade="all, delete-orphan")
    dependency_relationships = relationship("DependencyRelationship", back_populates="project", cascade="all, delete-orphan")
    snapshots = relationship("IndexSnapshot", back_populates="project", cascade="all, delete-orphan")
    analysis_sessions = relationship("AnalysisSession", back_populates="project", cascade="all, delete-orphan")
    
    def __init__(self, **kwargs):
        """Initialize project index with proper defaults."""
        if 'status' not in kwargs:
            kwargs['status'] = ProjectStatus.INACTIVE
        if 'configuration' not in kwargs:
            kwargs['configuration'] = {}
        if 'analysis_settings' not in kwargs:
            kwargs['analysis_settings'] = {}
        if 'file_patterns' not in kwargs:
            kwargs['file_patterns'] = {}
        if 'ignore_patterns' not in kwargs:
            kwargs['ignore_patterns'] = {}
        if 'meta_data' not in kwargs:
            kwargs['meta_data'] = {}
        if 'file_count' not in kwargs:
            kwargs['file_count'] = 0
        if 'dependency_count' not in kwargs:
            kwargs['dependency_count'] = 0
        
        super().__init__(**kwargs)
    
    def __repr__(self) -> str:
        return f"<ProjectIndex(id={self.id}, name='{self.name}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert project index to dictionary for serialization."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "root_path": self.root_path,
            "git_repository_url": self.git_repository_url,
            "git_branch": self.git_branch,
            "git_commit_hash": self.git_commit_hash,
            "status": self.status.value,
            "configuration": self.configuration,
            "analysis_settings": self.analysis_settings,
            "file_patterns": self.file_patterns,
            "ignore_patterns": self.ignore_patterns,
            "metadata": self.meta_data,
            "file_count": self.file_count,
            "dependency_count": self.dependency_count,
            "last_indexed_at": self.last_indexed_at.isoformat() if self.last_indexed_at else None,
            "last_analysis_at": self.last_analysis_at.isoformat() if self.last_analysis_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class FileEntry(Base):
    """
    Individual file analysis and metadata.
    
    Represents a single file within a project with its analysis data,
    metadata, and relationships to other files through dependencies.
    """
    
    __tablename__ = "file_entries"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(DatabaseAgnosticUUID(), ForeignKey("project_indexes.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # File path information
    file_path = Column(String(1000), nullable=False, index=True)
    relative_path = Column(String(1000), nullable=False, index=True)
    file_name = Column(String(255), nullable=False, index=True)
    file_extension = Column(String(50), nullable=True, index=True)
    
    # File classification
    file_type = Column(ENUM(FileType, name='file_type'), nullable=False, index=True)
    language = Column(String(50), nullable=True, index=True)
    encoding = Column(String(20), nullable=True, default="utf-8")
    
    # File metadata
    file_size = Column(BigInteger, nullable=True)
    line_count = Column(Integer, nullable=True)
    sha256_hash = Column(String(64), nullable=True, index=True)
    content_preview = Column(Text, nullable=True)
    
    # Analysis data and metadata
    analysis_data = Column(JSONB, nullable=True, default=dict)
    meta_data = Column("metadata", JSONB, nullable=True, default=dict)
    tags = Column(ARRAY(String), nullable=True, default=list)
    
    # File flags
    is_binary = Column(Boolean, nullable=False, default=False)
    is_generated = Column(Boolean, nullable=False, default=False)
    
    # Timestamps
    last_modified = Column(DateTime(timezone=True), nullable=True)
    indexed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    project = relationship("ProjectIndex", back_populates="file_entries")
    outgoing_dependencies = relationship("DependencyRelationship", foreign_keys="[DependencyRelationship.source_file_id]", back_populates="source_file", cascade="all, delete-orphan")
    incoming_dependencies = relationship("DependencyRelationship", foreign_keys="[DependencyRelationship.target_file_id]", back_populates="target_file")
    
    def __init__(self, **kwargs):
        """Initialize file entry with proper defaults."""
        if 'analysis_data' not in kwargs:
            kwargs['analysis_data'] = {}
        if 'meta_data' not in kwargs:
            kwargs['meta_data'] = {}
        if 'tags' not in kwargs:
            kwargs['tags'] = []
        if 'is_binary' not in kwargs:
            kwargs['is_binary'] = False
        if 'is_generated' not in kwargs:
            kwargs['is_generated'] = False
        if 'encoding' not in kwargs:
            kwargs['encoding'] = "utf-8"
        
        super().__init__(**kwargs)
    
    def __repr__(self) -> str:
        return f"<FileEntry(id={self.id}, path='{self.relative_path}', type='{self.file_type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert file entry to dictionary for serialization."""
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "file_path": self.file_path,
            "relative_path": self.relative_path,
            "file_name": self.file_name,
            "file_extension": self.file_extension,
            "file_type": self.file_type.value,
            "language": self.language,
            "encoding": self.encoding,
            "file_size": self.file_size,
            "line_count": self.line_count,
            "sha256_hash": self.sha256_hash,
            "content_preview": self.content_preview,
            "analysis_data": self.analysis_data,
            "metadata": self.meta_data,
            "tags": self.tags,
            "is_binary": self.is_binary,
            "is_generated": self.is_generated,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "indexed_at": self.indexed_at.isoformat() if self.indexed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class DependencyRelationship(Base):
    """
    Code dependencies and import mapping.
    
    Represents relationships between files through imports, requires,
    function calls, and other types of code dependencies.
    """
    
    __tablename__ = "dependency_relationships"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(DatabaseAgnosticUUID(), ForeignKey("project_indexes.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Source and target file relationships
    source_file_id = Column(DatabaseAgnosticUUID(), ForeignKey("file_entries.id", ondelete="CASCADE"), nullable=False, index=True)
    target_file_id = Column(DatabaseAgnosticUUID(), ForeignKey("file_entries.id", ondelete="SET NULL"), nullable=True, index=True)
    
    # Target information (for external dependencies or unresolved references)
    target_path = Column(String(1000), nullable=True, index=True)
    target_name = Column(String(255), nullable=False, index=True)
    
    # Dependency classification
    dependency_type = Column(ENUM(DependencyType, name='dependency_type'), nullable=False, index=True)
    
    # Location information
    line_number = Column(Integer, nullable=True)
    column_number = Column(Integer, nullable=True)
    source_text = Column(Text, nullable=True)
    
    # Dependency flags and metadata
    is_external = Column(Boolean, nullable=False, default=False, index=True)
    is_dynamic = Column(Boolean, nullable=False, default=False)
    confidence_score = Column(Float, nullable=True, default=1.0)
    meta_data = Column("metadata", JSONB, nullable=True, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    project = relationship("ProjectIndex", back_populates="dependency_relationships")
    source_file = relationship("FileEntry", foreign_keys=[source_file_id], back_populates="outgoing_dependencies")
    target_file = relationship("FileEntry", foreign_keys=[target_file_id], back_populates="incoming_dependencies")
    
    def __init__(self, **kwargs):
        """Initialize dependency relationship with proper defaults."""
        if 'meta_data' not in kwargs:
            kwargs['meta_data'] = {}
        if 'is_external' not in kwargs:
            kwargs['is_external'] = False
        if 'is_dynamic' not in kwargs:
            kwargs['is_dynamic'] = False
        if 'confidence_score' not in kwargs:
            kwargs['confidence_score'] = 1.0
        
        super().__init__(**kwargs)
    
    def __repr__(self) -> str:
        return f"<DependencyRelationship(id={self.id}, type='{self.dependency_type}', target='{self.target_name}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dependency relationship to dictionary for serialization."""
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "source_file_id": str(self.source_file_id),
            "target_file_id": str(self.target_file_id) if self.target_file_id else None,
            "target_path": self.target_path,
            "target_name": self.target_name,
            "dependency_type": self.dependency_type.value,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "source_text": self.source_text,
            "is_external": self.is_external,
            "is_dynamic": self.is_dynamic,
            "confidence_score": self.confidence_score,
            "metadata": self.meta_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class IndexSnapshot(Base):
    """
    Historical index states for comparison.
    
    Captures the state of a project index at specific points in time
    for version comparison, change tracking, and historical analysis.
    """
    
    __tablename__ = "index_snapshots"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(DatabaseAgnosticUUID(), ForeignKey("project_indexes.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Snapshot information
    snapshot_name = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    snapshot_type = Column(ENUM(SnapshotType, name='snapshot_type'), nullable=False, index=True)
    
    # Git integration
    git_commit_hash = Column(String(40), nullable=True, index=True)
    git_branch = Column(String(100), nullable=True)
    
    # Snapshot statistics
    file_count = Column(Integer, nullable=False, default=0)
    dependency_count = Column(Integer, nullable=False, default=0)
    
    # Change tracking and analysis
    changes_since_last = Column(JSONB, nullable=True, default=dict)
    analysis_metrics = Column(JSONB, nullable=True, default=dict)
    meta_data = Column("metadata", JSONB, nullable=True, default=dict)
    data_checksum = Column(String(64), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    project = relationship("ProjectIndex", back_populates="snapshots")
    
    def __init__(self, **kwargs):
        """Initialize index snapshot with proper defaults."""
        if 'file_count' not in kwargs:
            kwargs['file_count'] = 0
        if 'dependency_count' not in kwargs:
            kwargs['dependency_count'] = 0
        if 'changes_since_last' not in kwargs:
            kwargs['changes_since_last'] = {}
        if 'analysis_metrics' not in kwargs:
            kwargs['analysis_metrics'] = {}
        if 'meta_data' not in kwargs:
            kwargs['meta_data'] = {}
        
        super().__init__(**kwargs)
    
    def __repr__(self) -> str:
        return f"<IndexSnapshot(id={self.id}, name='{self.snapshot_name}', type='{self.snapshot_type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert index snapshot to dictionary for serialization."""
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "snapshot_name": self.snapshot_name,
            "description": self.description,
            "snapshot_type": self.snapshot_type.value,
            "git_commit_hash": self.git_commit_hash,
            "git_branch": self.git_branch,
            "file_count": self.file_count,
            "dependency_count": self.dependency_count,
            "changes_since_last": self.changes_since_last,
            "analysis_metrics": self.analysis_metrics,
            "metadata": self.meta_data,
            "data_checksum": self.data_checksum,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class AnalysisSession(Base):
    """
    AI analysis tracking and progress monitoring.
    
    Tracks the progress and results of analysis sessions that process
    project files and generate intelligence data and context optimization.
    """
    
    __tablename__ = "analysis_sessions"
    
    # Primary identification
    id = Column(DatabaseAgnosticUUID(), primary_key=True, default=uuid.uuid4)
    project_id = Column(DatabaseAgnosticUUID(), ForeignKey("project_indexes.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Session information
    session_name = Column(String(255), nullable=False, index=True)
    session_type = Column(ENUM(AnalysisSessionType, name='session_type'), nullable=False, index=True)
    status = Column(ENUM(AnalysisStatus, name='analysis_status'), nullable=False, default=AnalysisStatus.PENDING, index=True)
    
    # Progress tracking
    progress_percentage = Column(Float, nullable=False, default=0.0)
    current_phase = Column(String(100), nullable=True)
    files_processed = Column(Integer, nullable=False, default=0)
    files_total = Column(Integer, nullable=False, default=0)
    dependencies_found = Column(Integer, nullable=False, default=0)
    
    # Error and warning tracking
    errors_count = Column(Integer, nullable=False, default=0)
    warnings_count = Column(Integer, nullable=False, default=0)
    
    # Session data and results
    session_data = Column(JSONB, nullable=True, default=dict)
    error_log = Column(JSONB, nullable=True, default=list)
    performance_metrics = Column(JSONB, nullable=True, default=dict)
    configuration = Column(JSONB, nullable=True, default=dict)
    result_data = Column(JSONB, nullable=True, default=dict)
    
    # Timing information
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    estimated_completion = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    project = relationship("ProjectIndex", back_populates="analysis_sessions")
    
    def __init__(self, **kwargs):
        """Initialize analysis session with proper defaults."""
        if 'status' not in kwargs:
            kwargs['status'] = AnalysisStatus.PENDING
        if 'progress_percentage' not in kwargs:
            kwargs['progress_percentage'] = 0.0
        if 'files_processed' not in kwargs:
            kwargs['files_processed'] = 0
        if 'files_total' not in kwargs:
            kwargs['files_total'] = 0
        if 'dependencies_found' not in kwargs:
            kwargs['dependencies_found'] = 0
        if 'errors_count' not in kwargs:
            kwargs['errors_count'] = 0
        if 'warnings_count' not in kwargs:
            kwargs['warnings_count'] = 0
        if 'session_data' not in kwargs:
            kwargs['session_data'] = {}
        if 'error_log' not in kwargs:
            kwargs['error_log'] = []
        if 'performance_metrics' not in kwargs:
            kwargs['performance_metrics'] = {}
        if 'configuration' not in kwargs:
            kwargs['configuration'] = {}
        if 'result_data' not in kwargs:
            kwargs['result_data'] = {}
        
        super().__init__(**kwargs)
    
    def __repr__(self) -> str:
        return f"<AnalysisSession(id={self.id}, name='{self.session_name}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis session to dictionary for serialization."""
        return {
            "id": str(self.id),
            "project_id": str(self.project_id),
            "session_name": self.session_name,
            "session_type": self.session_type.value,
            "status": self.status.value,
            "progress_percentage": self.progress_percentage,
            "current_phase": self.current_phase,
            "files_processed": self.files_processed,
            "files_total": self.files_total,
            "dependencies_found": self.dependencies_found,
            "errors_count": self.errors_count,
            "warnings_count": self.warnings_count,
            "session_data": self.session_data,
            "error_log": self.error_log,
            "performance_metrics": self.performance_metrics,
            "configuration": self.configuration,
            "result_data": self.result_data,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    def start_session(self) -> None:
        """Mark the session as started."""
        self.status = AnalysisStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def update_progress(self, percentage: float, current_phase: str = None) -> None:
        """Update session progress."""
        self.progress_percentage = min(100.0, max(0.0, percentage))
        if current_phase:
            self.current_phase = current_phase
    
    def complete_session(self, result_data: Dict[str, Any] = None) -> None:
        """Mark the session as completed."""
        self.status = AnalysisStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.progress_percentage = 100.0
        if result_data:
            self.result_data = result_data
    
    def fail_session(self, error_message: str) -> None:
        """Mark the session as failed."""
        self.status = AnalysisStatus.FAILED
        self.completed_at = datetime.utcnow()
        if not self.error_log:
            self.error_log = []
        self.error_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "error": error_message,
            "level": "fatal"
        })
        self.errors_count += 1
    
    def add_error(self, error_message: str, level: str = "error") -> None:
        """Add an error to the session log."""
        if not self.error_log:
            self.error_log = []
        self.error_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "error": error_message,
            "level": level
        })
        if level in ["error", "fatal"]:
            self.errors_count += 1
        elif level == "warning":
            self.warnings_count += 1