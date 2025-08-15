"""
Pydantic schemas for Project Index API endpoints.

Request and response schemas for the intelligent project analysis
and context optimization system.
"""

import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Union

from pydantic import BaseModel, Field, ConfigDict

from ..models.project_index import (
    ProjectStatus, FileType, DependencyType, SnapshotType, 
    AnalysisSessionType, AnalysisStatus
)


# ================== PROJECT INDEX SCHEMAS ==================

class ProjectIndexCreate(BaseModel):
    """Schema for creating new project indexes."""
    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    root_path: str = Field(..., min_length=1, max_length=1000, description="Project root path")
    git_repository_url: Optional[str] = Field(None, max_length=500, description="Git repository URL")
    git_branch: Optional[str] = Field(None, max_length=100, description="Git branch name")
    git_commit_hash: Optional[str] = Field(None, min_length=40, max_length=40, description="Git commit hash")
    configuration: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Project configuration")
    analysis_settings: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Analysis settings")
    file_patterns: Optional[Dict[str, Any]] = Field(default_factory=dict, description="File pattern filters")
    ignore_patterns: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Ignore pattern filters")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class ProjectIndexUpdate(BaseModel):
    """Schema for updating existing project indexes."""
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    status: Optional[ProjectStatus] = Field(None, description="Project status")
    git_branch: Optional[str] = Field(None, max_length=100, description="Git branch name")
    git_commit_hash: Optional[str] = Field(None, min_length=40, max_length=40, description="Git commit hash")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Project configuration")
    analysis_settings: Optional[Dict[str, Any]] = Field(None, description="Analysis settings")
    file_patterns: Optional[Dict[str, Any]] = Field(None, description="File pattern filters")
    ignore_patterns: Optional[Dict[str, Any]] = Field(None, description="Ignore pattern filters")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ProjectIndexResponse(BaseModel):
    """Schema for project index API responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    name: str
    description: Optional[str]
    root_path: str
    git_repository_url: Optional[str]
    git_branch: Optional[str]
    git_commit_hash: Optional[str]
    status: ProjectStatus
    configuration: Dict[str, Any]
    analysis_settings: Dict[str, Any]
    file_patterns: Dict[str, Any]
    ignore_patterns: Dict[str, Any]
    metadata: Dict[str, Any]
    file_count: int
    dependency_count: int
    last_indexed_at: Optional[datetime]
    last_analysis_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


class ProjectIndexListResponse(BaseModel):
    """Schema for paginated project index list responses."""
    projects: List[ProjectIndexResponse]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


# ================== FILE ENTRY SCHEMAS ==================

class FileEntryCreate(BaseModel):
    """Schema for creating new file entries."""
    project_id: uuid.UUID = Field(..., description="Project ID")
    file_path: str = Field(..., min_length=1, max_length=1000, description="Absolute file path")
    relative_path: str = Field(..., min_length=1, max_length=1000, description="Relative file path")
    file_name: str = Field(..., min_length=1, max_length=255, description="File name")
    file_extension: Optional[str] = Field(None, max_length=50, description="File extension")
    file_type: FileType = Field(..., description="File type classification")
    language: Optional[str] = Field(None, max_length=50, description="Programming language")
    encoding: str = Field(default="utf-8", max_length=20, description="File encoding")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    line_count: Optional[int] = Field(None, ge=0, description="Number of lines")
    sha256_hash: Optional[str] = Field(None, min_length=64, max_length=64, description="SHA256 hash")
    content_preview: Optional[str] = Field(None, description="Content preview")
    analysis_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Analysis results")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    tags: Optional[List[str]] = Field(default_factory=list, description="File tags")
    is_binary: bool = Field(default=False, description="Whether file is binary")
    is_generated: bool = Field(default=False, description="Whether file is generated")
    last_modified: Optional[datetime] = Field(None, description="Last modification time")


class FileEntryUpdate(BaseModel):
    """Schema for updating existing file entries."""
    file_type: Optional[FileType] = Field(None, description="File type classification")
    language: Optional[str] = Field(None, max_length=50, description="Programming language")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    line_count: Optional[int] = Field(None, ge=0, description="Number of lines")
    sha256_hash: Optional[str] = Field(None, min_length=64, max_length=64, description="SHA256 hash")
    content_preview: Optional[str] = Field(None, description="Content preview")
    analysis_data: Optional[Dict[str, Any]] = Field(None, description="Analysis results")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    tags: Optional[List[str]] = Field(None, description="File tags")
    is_binary: Optional[bool] = Field(None, description="Whether file is binary")
    is_generated: Optional[bool] = Field(None, description="Whether file is generated")
    last_modified: Optional[datetime] = Field(None, description="Last modification time")


class FileEntryResponse(BaseModel):
    """Schema for file entry API responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    project_id: uuid.UUID
    file_path: str
    relative_path: str
    file_name: str
    file_extension: Optional[str]
    file_type: FileType
    language: Optional[str]
    encoding: str
    file_size: Optional[int]
    line_count: Optional[int]
    sha256_hash: Optional[str]
    content_preview: Optional[str]
    analysis_data: Dict[str, Any]
    metadata: Dict[str, Any]
    tags: List[str]
    is_binary: bool
    is_generated: bool
    last_modified: Optional[datetime]
    indexed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


class FileEntryListResponse(BaseModel):
    """Schema for paginated file entry list responses."""
    files: List[FileEntryResponse]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


# ================== DEPENDENCY RELATIONSHIP SCHEMAS ==================

class DependencyRelationshipCreate(BaseModel):
    """Schema for creating new dependency relationships."""
    project_id: uuid.UUID = Field(..., description="Project ID")
    source_file_id: uuid.UUID = Field(..., description="Source file ID")
    target_file_id: Optional[uuid.UUID] = Field(None, description="Target file ID (if internal)")
    target_path: Optional[str] = Field(None, max_length=1000, description="Target path (if external)")
    target_name: str = Field(..., min_length=1, max_length=255, description="Target name")
    dependency_type: DependencyType = Field(..., description="Type of dependency")
    line_number: Optional[int] = Field(None, ge=1, description="Line number where dependency occurs")
    column_number: Optional[int] = Field(None, ge=1, description="Column number where dependency occurs")
    source_text: Optional[str] = Field(None, description="Source code text")
    is_external: bool = Field(default=False, description="Whether dependency is external")
    is_dynamic: bool = Field(default=False, description="Whether dependency is dynamic")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class DependencyRelationshipUpdate(BaseModel):
    """Schema for updating existing dependency relationships."""
    target_file_id: Optional[uuid.UUID] = Field(None, description="Target file ID (if internal)")
    target_path: Optional[str] = Field(None, max_length=1000, description="Target path (if external)")
    dependency_type: Optional[DependencyType] = Field(None, description="Type of dependency")
    line_number: Optional[int] = Field(None, ge=1, description="Line number where dependency occurs")
    column_number: Optional[int] = Field(None, ge=1, description="Column number where dependency occurs")
    source_text: Optional[str] = Field(None, description="Source code text")
    is_external: Optional[bool] = Field(None, description="Whether dependency is external")
    is_dynamic: Optional[bool] = Field(None, description="Whether dependency is dynamic")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class DependencyRelationshipResponse(BaseModel):
    """Schema for dependency relationship API responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    project_id: uuid.UUID
    source_file_id: uuid.UUID
    target_file_id: Optional[uuid.UUID]
    target_path: Optional[str]
    target_name: str
    dependency_type: DependencyType
    line_number: Optional[int]
    column_number: Optional[int]
    source_text: Optional[str]
    is_external: bool
    is_dynamic: bool
    confidence_score: float
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class DependencyRelationshipListResponse(BaseModel):
    """Schema for paginated dependency relationship list responses."""
    dependencies: List[DependencyRelationshipResponse]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


# ================== INDEX SNAPSHOT SCHEMAS ==================

class IndexSnapshotCreate(BaseModel):
    """Schema for creating new index snapshots."""
    project_id: uuid.UUID = Field(..., description="Project ID")
    snapshot_name: str = Field(..., min_length=1, max_length=255, description="Snapshot name")
    description: Optional[str] = Field(None, description="Snapshot description")
    snapshot_type: SnapshotType = Field(..., description="Type of snapshot")
    git_commit_hash: Optional[str] = Field(None, min_length=40, max_length=40, description="Git commit hash")
    git_branch: Optional[str] = Field(None, max_length=100, description="Git branch name")
    changes_since_last: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Changes since last snapshot")
    analysis_metrics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Analysis metrics")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class IndexSnapshotResponse(BaseModel):
    """Schema for index snapshot API responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    project_id: uuid.UUID
    snapshot_name: str
    description: Optional[str]
    snapshot_type: SnapshotType
    git_commit_hash: Optional[str]
    git_branch: Optional[str]
    file_count: int
    dependency_count: int
    changes_since_last: Dict[str, Any]
    analysis_metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    data_checksum: Optional[str]
    created_at: datetime


class IndexSnapshotListResponse(BaseModel):
    """Schema for paginated index snapshot list responses."""
    snapshots: List[IndexSnapshotResponse]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


# ================== ANALYSIS SESSION SCHEMAS ==================

class AnalysisSessionCreate(BaseModel):
    """Schema for creating new analysis sessions."""
    project_id: uuid.UUID = Field(..., description="Project ID")
    session_name: str = Field(..., min_length=1, max_length=255, description="Session name")
    session_type: AnalysisSessionType = Field(..., description="Type of analysis session")
    files_total: int = Field(default=0, ge=0, description="Total number of files to process")
    configuration: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Session configuration")


class AnalysisSessionUpdate(BaseModel):
    """Schema for updating existing analysis sessions."""
    status: Optional[AnalysisStatus] = Field(None, description="Analysis status")
    progress_percentage: Optional[float] = Field(None, ge=0.0, le=100.0, description="Progress percentage")
    current_phase: Optional[str] = Field(None, max_length=100, description="Current processing phase")
    files_processed: Optional[int] = Field(None, ge=0, description="Number of files processed")
    files_total: Optional[int] = Field(None, ge=0, description="Total number of files to process")
    dependencies_found: Optional[int] = Field(None, ge=0, description="Number of dependencies found")
    session_data: Optional[Dict[str, Any]] = Field(None, description="Session-specific data")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics")
    result_data: Optional[Dict[str, Any]] = Field(None, description="Analysis results")


class AnalysisSessionResponse(BaseModel):
    """Schema for analysis session API responses."""
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    project_id: uuid.UUID
    session_name: str
    session_type: AnalysisSessionType
    status: AnalysisStatus
    progress_percentage: float
    current_phase: Optional[str]
    files_processed: int
    files_total: int
    dependencies_found: int
    errors_count: int
    warnings_count: int
    session_data: Dict[str, Any]
    error_log: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    configuration: Dict[str, Any]
    result_data: Dict[str, Any]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    estimated_completion: Optional[datetime]
    created_at: datetime
    updated_at: datetime


class AnalysisSessionListResponse(BaseModel):
    """Schema for paginated analysis session list responses."""
    sessions: List[AnalysisSessionResponse]
    total: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


# ================== SEARCH AND FILTERING SCHEMAS ==================

class ProjectIndexFilter(BaseModel):
    """Schema for filtering project indexes."""
    status: Optional[List[ProjectStatus]] = Field(None, description="Filter by project status")
    git_repository_url: Optional[str] = Field(None, description="Filter by git repository URL")
    git_branch: Optional[str] = Field(None, description="Filter by git branch")
    name_contains: Optional[str] = Field(None, description="Filter by name containing text")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date")
    created_before: Optional[datetime] = Field(None, description="Filter by creation date")
    last_indexed_after: Optional[datetime] = Field(None, description="Filter by last indexed date")


class FileEntryFilter(BaseModel):
    """Schema for filtering file entries."""
    project_id: Optional[uuid.UUID] = Field(None, description="Filter by project ID")
    file_type: Optional[List[FileType]] = Field(None, description="Filter by file type")
    language: Optional[List[str]] = Field(None, description="Filter by programming language")
    file_extension: Optional[List[str]] = Field(None, description="Filter by file extension")
    is_binary: Optional[bool] = Field(None, description="Filter by binary files")
    is_generated: Optional[bool] = Field(None, description="Filter by generated files")
    name_contains: Optional[str] = Field(None, description="Filter by file name containing text")
    path_contains: Optional[str] = Field(None, description="Filter by path containing text")
    modified_after: Optional[datetime] = Field(None, description="Filter by modification date")
    modified_before: Optional[datetime] = Field(None, description="Filter by modification date")


class DependencyRelationshipFilter(BaseModel):
    """Schema for filtering dependency relationships."""
    project_id: Optional[uuid.UUID] = Field(None, description="Filter by project ID")
    source_file_id: Optional[uuid.UUID] = Field(None, description="Filter by source file ID")
    target_file_id: Optional[uuid.UUID] = Field(None, description="Filter by target file ID")
    dependency_type: Optional[List[DependencyType]] = Field(None, description="Filter by dependency type")
    is_external: Optional[bool] = Field(None, description="Filter by external dependencies")
    is_dynamic: Optional[bool] = Field(None, description="Filter by dynamic dependencies")
    target_name_contains: Optional[str] = Field(None, description="Filter by target name containing text")
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum confidence score")


class AnalysisSessionFilter(BaseModel):
    """Schema for filtering analysis sessions."""
    project_id: Optional[uuid.UUID] = Field(None, description="Filter by project ID")
    session_type: Optional[List[AnalysisSessionType]] = Field(None, description="Filter by session type")
    status: Optional[List[AnalysisStatus]] = Field(None, description="Filter by status")
    started_after: Optional[datetime] = Field(None, description="Filter by start date")
    started_before: Optional[datetime] = Field(None, description="Filter by start date")
    completed_after: Optional[datetime] = Field(None, description="Filter by completion date")
    completed_before: Optional[datetime] = Field(None, description="Filter by completion date")
    min_progress: Optional[float] = Field(None, ge=0.0, le=100.0, description="Minimum progress percentage")


# ================== BULK OPERATION SCHEMAS ==================

class BulkFileCreate(BaseModel):
    """Schema for bulk creating file entries."""
    files: List[FileEntryCreate] = Field(..., description="List of files to create")


class BulkDependencyCreate(BaseModel):
    """Schema for bulk creating dependency relationships."""
    dependencies: List[DependencyRelationshipCreate] = Field(..., description="List of dependencies to create")


class BulkOperationResponse(BaseModel):
    """Schema for bulk operation responses."""
    created_count: int = Field(..., description="Number of items successfully created")
    failed_count: int = Field(..., description="Number of items that failed to create")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="List of errors encountered")


# ================== ANALYTICS AND STATISTICS SCHEMAS ==================

class ProjectStatistics(BaseModel):
    """Schema for project statistics."""
    total_files: int = Field(..., description="Total number of files")
    files_by_type: Dict[str, int] = Field(..., description="File count by type")
    files_by_language: Dict[str, int] = Field(..., description="File count by language")
    total_dependencies: int = Field(..., description="Total number of dependencies")
    dependencies_by_type: Dict[str, int] = Field(..., description="Dependency count by type")
    external_dependencies: int = Field(..., description="Number of external dependencies")
    internal_dependencies: int = Field(..., description="Number of internal dependencies")
    total_lines_of_code: Optional[int] = Field(None, description="Total lines of code")
    binary_files: int = Field(..., description="Number of binary files")
    generated_files: int = Field(..., description="Number of generated files")


class DependencyGraphNode(BaseModel):
    """Schema for dependency graph nodes."""
    file_id: uuid.UUID = Field(..., description="File ID")
    file_path: str = Field(..., description="File path")
    file_type: FileType = Field(..., description="File type")
    language: Optional[str] = Field(None, description="Programming language")
    in_degree: int = Field(..., description="Number of incoming dependencies")
    out_degree: int = Field(..., description="Number of outgoing dependencies")


class DependencyGraphEdge(BaseModel):
    """Schema for dependency graph edges."""
    source_file_id: uuid.UUID = Field(..., description="Source file ID")
    target_file_id: Optional[uuid.UUID] = Field(None, description="Target file ID")
    target_name: str = Field(..., description="Target name")
    dependency_type: DependencyType = Field(..., description="Dependency type")
    is_external: bool = Field(..., description="Whether dependency is external")


class DependencyGraph(BaseModel):
    """Schema for complete dependency graph."""
    nodes: List[DependencyGraphNode] = Field(..., description="Graph nodes")
    edges: List[DependencyGraphEdge] = Field(..., description="Graph edges")
    statistics: Dict[str, Any] = Field(..., description="Graph statistics")


class AnalysisProgress(BaseModel):
    """Schema for real-time analysis progress."""
    session_id: uuid.UUID = Field(..., description="Analysis session ID")
    project_id: uuid.UUID = Field(..., description="Project ID")
    progress_percentage: float = Field(..., ge=0.0, le=100.0, description="Progress percentage")
    current_phase: Optional[str] = Field(None, description="Current processing phase")
    files_processed: int = Field(..., description="Number of files processed")
    files_total: int = Field(..., description="Total number of files to process")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    last_updated: datetime = Field(..., description="Last update timestamp")