"""
Internal Pydantic Models for LeanVibe Agent Hive 2.0 Project Index

Internal data structures and configuration models for project analysis,
code intelligence, and context optimization. Used within the project
indexer system for type safety and validation.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict, field_validator

from ..models.project_index import FileType, DependencyType


# ================== CONFIGURATION MODELS ==================

class AnalysisConfiguration(BaseModel):
    """Configuration for code analysis operations."""
    model_config = ConfigDict(validate_assignment=True)
    
    # Language support
    enabled_languages: List[str] = Field(
        default=['python', 'javascript', 'typescript', 'json'],
        description="List of programming languages to analyze"
    )
    
    # Analysis depth
    parse_ast: bool = Field(default=True, description="Enable AST parsing")
    extract_dependencies: bool = Field(default=True, description="Extract code dependencies")
    calculate_complexity: bool = Field(default=True, description="Calculate complexity metrics")
    analyze_docstrings: bool = Field(default=True, description="Analyze documentation")
    
    # Performance settings
    max_file_size_mb: int = Field(default=10, description="Maximum file size to analyze in MB")
    max_line_count: int = Field(default=50000, description="Maximum lines per file to analyze")
    timeout_seconds: int = Field(default=30, description="Analysis timeout per file")
    
    # File filtering
    include_patterns: List[str] = Field(
        default=['**/*.py', '**/*.js', '**/*.ts', '**/*.json'],
        description="File patterns to include"
    )
    exclude_patterns: List[str] = Field(
        default=['**/node_modules/**', '**/__pycache__/**', '**/.git/**'],
        description="File patterns to exclude"
    )
    
    # Dependency extraction
    resolve_relative_imports: bool = Field(default=True, description="Resolve relative imports")
    include_external_deps: bool = Field(default=True, description="Include external dependencies")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence for dependencies")
    
    @field_validator('max_file_size_mb')
    @classmethod
    def validate_file_size(cls, v):
        if v <= 0 or v > 100:
            raise ValueError('File size must be between 1 and 100 MB')
        return v
    
    @field_validator('confidence_threshold')
    @classmethod
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence threshold must be between 0.0 and 1.0')
        return v


class ProjectIndexConfig(BaseModel):
    """Configuration for project indexing operations."""
    model_config = ConfigDict(validate_assignment=True)
    
    # Analysis configuration
    analysis_config: AnalysisConfiguration = Field(default_factory=AnalysisConfiguration)
    
    # Performance settings
    max_concurrent_analyses: int = Field(default=4, description="Maximum concurrent file analyses")
    analysis_batch_size: int = Field(default=50, description="Files to analyze per batch")
    cache_enabled: bool = Field(default=True, description="Enable result caching")
    
    # Database settings
    batch_insert_size: int = Field(default=100, description="Database batch insert size")
    transaction_timeout: int = Field(default=300, description="Database transaction timeout")
    
    # Context optimization
    context_optimization_enabled: bool = Field(default=True, description="Enable context optimization")
    max_context_files: int = Field(default=50, description="Maximum files in context recommendations")
    
    # File monitoring
    monitoring_enabled: bool = Field(default=True, description="Enable real-time file monitoring")
    monitoring_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'debounce_seconds': 2.0,
            'include_patterns': ['*'],
            'exclude_patterns': [],
            'max_file_size_mb': 10,
            'watch_subdirectories': True
        },
        description="File monitoring configuration"
    )
    
    # Incremental updates
    incremental_updates: bool = Field(default=True, description="Enable incremental analysis updates")
    incremental_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'full_rebuild_threshold': 50,
            'cascading_threshold': 20,
            'batch_timeout': 5.0,
            'max_batch_size': 100
        },
        description="Incremental update configuration"
    )
    
    # Caching
    cache_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'enable_compression': True,
            'compression_threshold': 1024,
            'max_memory_mb': 500,
            'layer_ttls': {
                'ast': 3600 * 24 * 3,
                'analysis': 3600 * 24,
                'dependency': 3600 * 12,
                'context': 3600 * 2,
                'project': 3600,
                'language': 3600 * 24 * 7,
                'hash': 3600 * 24 * 30
            }
        },
        description="Advanced caching configuration"
    )
    
    # Event system
    events_enabled: bool = Field(default=True, description="Enable real-time event publishing")
    event_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'max_history': 1000,
            'websocket_enabled': True,
            'event_persistence': False
        },
        description="Event system configuration"
    )
    
    @field_validator('max_concurrent_analyses')
    @classmethod
    def validate_concurrency(cls, v):
        if v <= 0 or v > 20:
            raise ValueError('Concurrent analyses must be between 1 and 20')
        return v
    
    @field_validator('monitoring_config')
    @classmethod
    def validate_monitoring_config(cls, v):
        required_keys = {'debounce_seconds', 'max_file_size_mb'}
        if not all(key in v for key in required_keys):
            raise ValueError(f'Monitoring config must contain: {required_keys}')
        
        if v['debounce_seconds'] < 0.1 or v['debounce_seconds'] > 60:
            raise ValueError('Debounce seconds must be between 0.1 and 60')
        
        return v
    
    @field_validator('incremental_config')
    @classmethod
    def validate_incremental_config(cls, v):
        thresholds = ['full_rebuild_threshold', 'cascading_threshold']
        for threshold in thresholds:
            if threshold in v and (v[threshold] < 1 or v[threshold] > 1000):
                raise ValueError(f'{threshold} must be between 1 and 1000')
        
        return v


# ================== ANALYSIS RESULT MODELS ==================

class ComplexityMetrics(BaseModel):
    """Code complexity metrics."""
    model_config = ConfigDict(validate_assignment=True)
    
    cyclomatic_complexity: int = Field(default=1, description="Cyclomatic complexity")
    cognitive_complexity: int = Field(default=1, description="Cognitive complexity")
    nesting_depth: Optional[int] = Field(default=None, description="Maximum nesting depth")
    halstead_volume: Optional[float] = Field(default=None, description="Halstead volume")
    
    @field_validator('cyclomatic_complexity', 'cognitive_complexity')
    @classmethod
    def validate_complexity(cls, v):
        if v < 1:
            raise ValueError('Complexity must be at least 1')
        return v


class CodeStructure(BaseModel):
    """Code structure information."""
    model_config = ConfigDict(validate_assignment=True)
    
    functions: List[Dict[str, Any]] = Field(default_factory=list, description="Function definitions")
    classes: List[Dict[str, Any]] = Field(default_factory=list, description="Class definitions")
    imports: List[Dict[str, Any]] = Field(default_factory=list, description="Import statements")
    exports: List[Dict[str, Any]] = Field(default_factory=list, description="Export statements")
    variables: List[Dict[str, Any]] = Field(default_factory=list, description="Variable declarations")
    constants: List[Dict[str, Any]] = Field(default_factory=list, description="Constant declarations")


class DependencyResult(BaseModel):
    """Result of dependency extraction."""
    model_config = ConfigDict(validate_assignment=True)
    
    # Source information
    source_file_path: str = Field(..., description="Path to source file")
    source_file_id: Optional[str] = Field(default=None, description="Source file database ID")
    
    # Target information
    target_name: str = Field(..., description="Name of the dependency target")
    target_path: Optional[str] = Field(default=None, description="Path to target file")
    target_file_id: Optional[str] = Field(default=None, description="Target file database ID")
    
    # Dependency details
    dependency_type: str = Field(..., description="Type of dependency (import, require, etc.)")
    line_number: Optional[int] = Field(default=None, description="Line number where dependency occurs")
    column_number: Optional[int] = Field(default=None, description="Column number where dependency occurs")
    source_text: Optional[str] = Field(default=None, description="Source code text")
    
    # Classification
    is_external: bool = Field(default=False, description="Whether dependency is external")
    is_dynamic: bool = Field(default=False, description="Whether dependency is dynamic")
    confidence_score: float = Field(default=1.0, description="Confidence in dependency extraction")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('confidence_score')
    @classmethod
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence score must be between 0.0 and 1.0')
        return v
    
    @field_validator('line_number', 'column_number')
    @classmethod
    def validate_position(cls, v):
        if v is not None and v < 1:
            raise ValueError('Line and column numbers must be positive')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DependencyResult':
        """Create from dictionary."""
        return cls(**data)


class FileAnalysisResult(BaseModel):
    """Result of single file analysis."""
    model_config = ConfigDict(validate_assignment=True)
    
    # File information
    file_path: str = Field(..., description="Absolute file path")
    relative_path: Optional[str] = Field(default=None, description="Relative file path")
    file_name: Optional[str] = Field(default=None, description="File name")
    file_extension: Optional[str] = Field(default=None, description="File extension")
    
    # File classification
    file_type: Optional[str] = Field(default=None, description="File type classification")
    language: Optional[str] = Field(default=None, description="Programming language")
    encoding: str = Field(default='utf-8', description="File encoding")
    
    # File metadata
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    line_count: Optional[int] = Field(default=None, description="Number of lines")
    sha256_hash: Optional[str] = Field(default=None, description="SHA256 hash of content")
    
    # File flags
    is_binary: bool = Field(default=False, description="Whether file is binary")
    is_generated: bool = Field(default=False, description="Whether file is generated")
    
    # Analysis results
    analysis_data: Dict[str, Any] = Field(default_factory=dict, description="AST and structure analysis")
    dependencies: List[DependencyResult] = Field(default_factory=list, description="Extracted dependencies")
    complexity_metrics: Optional[ComplexityMetrics] = Field(default=None, description="Complexity metrics")
    code_structure: Optional[CodeStructure] = Field(default=None, description="Code structure")
    
    # Analysis metadata
    analysis_successful: bool = Field(default=True, description="Whether analysis succeeded")
    analysis_duration: Optional[float] = Field(default=None, description="Analysis duration in seconds")
    error_message: Optional[str] = Field(default=None, description="Error message if analysis failed")
    last_modified: Optional[datetime] = Field(default=None, description="File last modification time")
    
    # Tags and metadata
    tags: List[str] = Field(default_factory=list, description="File tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('file_size')
    @classmethod
    def validate_file_size(cls, v):
        if v is not None and v < 0:
            raise ValueError('File size cannot be negative')
        return v
    
    @field_validator('line_count')
    @classmethod
    def validate_line_count(cls, v):
        if v is not None and v < 0:
            raise ValueError('Line count cannot be negative')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = self.model_dump()
        # Convert datetime to ISO string
        if data.get('last_modified'):
            value = data['last_modified']
            if hasattr(value, 'isoformat'):
                data['last_modified'] = value.isoformat()
            elif isinstance(value, str):
                data['last_modified'] = value  # Already a string
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileAnalysisResult':
        """Create from dictionary."""
        # Parse datetime from ISO string
        if 'last_modified' in data and isinstance(data['last_modified'], str):
            data['last_modified'] = datetime.fromisoformat(data['last_modified'])
        
        # Convert dependencies
        if 'dependencies' in data:
            deps = []
            for dep_data in data['dependencies']:
                if isinstance(dep_data, dict):
                    deps.append(DependencyResult(**dep_data))
                else:
                    deps.append(dep_data)
            data['dependencies'] = deps
        
        return cls(**data)


class AnalysisResult(BaseModel):
    """Result of complete project analysis."""
    model_config = ConfigDict(validate_assignment=True)
    
    # Session information
    project_id: str = Field(..., description="Project identifier")
    session_id: str = Field(..., description="Analysis session identifier")
    analysis_type: str = Field(..., description="Type of analysis performed")
    
    # Analysis statistics
    files_processed: int = Field(default=0, description="Number of files processed")
    files_analyzed: int = Field(default=0, description="Number of files successfully analyzed")
    dependencies_found: int = Field(default=0, description="Number of dependencies found")
    
    # Timing information
    analysis_duration: float = Field(default=0.0, description="Total analysis duration in seconds")
    started_at: Optional[datetime] = Field(default=None, description="Analysis start time")
    completed_at: Optional[datetime] = Field(default=None, description="Analysis completion time")
    
    # Results
    file_results: List[FileAnalysisResult] = Field(default_factory=list, description="Individual file results")
    dependency_results: List[DependencyResult] = Field(default_factory=list, description="Dependency results")
    
    # Analysis insights
    context_optimization: Dict[str, Any] = Field(default_factory=dict, description="Context optimization results")
    project_statistics: Dict[str, Any] = Field(default_factory=dict, description="Project statistics")
    quality_metrics: Dict[str, Any] = Field(default_factory=dict, description="Code quality metrics")
    
    # Performance metrics
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Analysis performance metrics")
    
    # Warnings and errors
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")
    errors: List[str] = Field(default_factory=list, description="Analysis errors")
    
    @field_validator('files_processed', 'files_analyzed', 'dependencies_found')
    @classmethod
    def validate_counts(cls, v):
        if v < 0:
            raise ValueError('Counts cannot be negative')
        return v
    
    @field_validator('analysis_duration')
    @classmethod
    def validate_duration(cls, v):
        if v < 0:
            raise ValueError('Duration cannot be negative')
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = self.model_dump()
        
        # Convert datetime fields to ISO strings
        for field in ['started_at', 'completed_at']:
            value = data.get(field)
            if value and hasattr(value, 'isoformat'):
                data[field] = value.isoformat()
            elif value and isinstance(value, str):
                data[field] = value  # Already a string
        
        # Convert nested objects to dicts to avoid datetime serialization issues
        if 'file_results' in data:
            data['file_results'] = [
                result.to_dict() if hasattr(result, 'to_dict') else result 
                for result in data['file_results']
            ]
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisResult':
        """Create from dictionary."""
        # Parse datetime fields
        for field in ['started_at', 'completed_at']:
            if field in data and isinstance(data[field], str):
                data[field] = datetime.fromisoformat(data[field])
        
        # Convert file results
        if 'file_results' in data:
            file_results = []
            for file_data in data['file_results']:
                if isinstance(file_data, dict):
                    file_results.append(FileAnalysisResult.from_dict(file_data))
                else:
                    file_results.append(file_data)
            data['file_results'] = file_results
        
        # Convert dependency results
        if 'dependency_results' in data:
            dep_results = []
            for dep_data in data['dependency_results']:
                if isinstance(dep_data, dict):
                    dep_results.append(DependencyResult(**dep_data))
                else:
                    dep_results.append(dep_data)
            data['dependency_results'] = dep_results
        
        return cls(**data)


# ================== STATISTICS AND METRICS MODELS ==================

class ProjectStatistics(BaseModel):
    """Project-level statistics and metrics."""
    model_config = ConfigDict(validate_assignment=True)
    
    # File statistics
    total_files: int = Field(default=0, description="Total number of files")
    files_by_type: Dict[str, int] = Field(default_factory=dict, description="File count by type")
    files_by_language: Dict[str, int] = Field(default_factory=dict, description="File count by language")
    
    # Code statistics
    total_lines_of_code: int = Field(default=0, description="Total lines of code")
    total_file_size: int = Field(default=0, description="Total file size in bytes")
    average_file_size: float = Field(default=0.0, description="Average file size")
    
    # Dependency statistics
    total_dependencies: int = Field(default=0, description="Total number of dependencies")
    internal_dependencies: int = Field(default=0, description="Internal dependencies")
    external_dependencies: int = Field(default=0, description="External dependencies")
    dependencies_by_type: Dict[str, int] = Field(default_factory=dict, description="Dependencies by type")
    
    # Complexity statistics
    average_complexity: float = Field(default=0.0, description="Average cyclomatic complexity")
    max_complexity: int = Field(default=0, description="Maximum complexity in project")
    complexity_distribution: Dict[str, int] = Field(default_factory=dict, description="Complexity distribution")
    
    # Quality metrics
    documentation_coverage: float = Field(default=0.0, description="Documentation coverage percentage")
    test_coverage: float = Field(default=0.0, description="Test coverage percentage")
    code_duplication: float = Field(default=0.0, description="Code duplication percentage")
    
    # Analysis metadata
    last_updated: Optional[datetime] = Field(default=None, description="Last statistics update")
    analysis_version: str = Field(default="1.0", description="Analysis version")
    
    @field_validator('total_files', 'total_lines_of_code', 'total_file_size')
    @classmethod
    def validate_totals(cls, v):
        if v < 0:
            raise ValueError('Total counts cannot be negative')
        return v
    
    @field_validator('documentation_coverage', 'test_coverage', 'code_duplication')
    @classmethod
    def validate_percentages(cls, v):
        if not 0.0 <= v <= 100.0:
            raise ValueError('Percentages must be between 0.0 and 100.0')
        return v


# ================== CONTEXT OPTIMIZATION MODELS ==================

class ContextRecommendation(BaseModel):
    """Context-based file recommendation."""
    model_config = ConfigDict(validate_assignment=True)
    
    file_path: str = Field(..., description="Recommended file path")
    relevance_score: float = Field(..., description="Relevance score (0-1)")
    confidence: float = Field(..., description="Confidence in recommendation (0-1)")
    reasoning: List[str] = Field(default_factory=list, description="Reasoning for recommendation")
    tags: List[str] = Field(default_factory=list, description="Associated tags")
    
    @field_validator('relevance_score', 'confidence')
    @classmethod
    def validate_scores(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Scores must be between 0.0 and 1.0')
        return v


class ContextClusterInfo(BaseModel):
    """Information about a context cluster."""
    model_config = ConfigDict(validate_assignment=True)
    
    cluster_id: str = Field(..., description="Unique cluster identifier")
    name: str = Field(..., description="Human-readable cluster name")
    description: str = Field(..., description="Cluster description")
    files: List[str] = Field(..., description="Files in the cluster")
    central_files: List[str] = Field(default_factory=list, description="Most important files in cluster")
    cluster_score: float = Field(..., description="Overall cluster relevance score")
    
    @field_validator('cluster_score')
    @classmethod
    def validate_score(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Cluster score must be between 0.0 and 1.0')
        return v


# ================== MONITORING AND CHANGE MODELS ==================

class FileChangeInfo(BaseModel):
    """Information about a file system change."""
    model_config = ConfigDict(validate_assignment=True)
    
    file_path: str = Field(..., description="Path to changed file")
    change_type: str = Field(..., description="Type of change (created, modified, deleted)")
    timestamp: datetime = Field(..., description="When the change occurred")
    old_hash: Optional[str] = Field(default=None, description="Previous file hash")
    new_hash: Optional[str] = Field(default=None, description="New file hash")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional change metadata")


class AnalysisProgress(BaseModel):
    """Progress information for ongoing analysis."""
    model_config = ConfigDict(validate_assignment=True)
    
    session_id: str = Field(..., description="Analysis session ID")
    project_id: str = Field(..., description="Project ID")
    current_phase: str = Field(..., description="Current analysis phase")
    progress_percentage: float = Field(..., description="Progress percentage (0-100)")
    files_processed: int = Field(default=0, description="Files processed so far")
    files_total: int = Field(default=0, description="Total files to process")
    estimated_completion: Optional[datetime] = Field(default=None, description="Estimated completion time")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last progress update")
    
    @field_validator('progress_percentage')
    @classmethod
    def validate_progress(cls, v):
        if not 0.0 <= v <= 100.0:
            raise ValueError('Progress percentage must be between 0.0 and 100.0')
        return v


# ================== ERROR AND VALIDATION MODELS ==================

class AnalysisError(BaseModel):
    """Represents an error during analysis."""
    model_config = ConfigDict(validate_assignment=True)
    
    error_type: str = Field(..., description="Type of error")
    error_message: str = Field(..., description="Error message")
    file_path: Optional[str] = Field(default=None, description="File where error occurred")
    line_number: Optional[int] = Field(default=None, description="Line number where error occurred")
    severity: str = Field(default="error", description="Error severity (warning, error, critical)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When error occurred")
    stack_trace: Optional[str] = Field(default=None, description="Stack trace if available")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional error metadata")


class ValidationResult(BaseModel):
    """Result of data validation."""
    model_config = ConfigDict(validate_assignment=True)
    
    is_valid: bool = Field(..., description="Whether validation passed")
    errors: List[AnalysisError] = Field(default_factory=list, description="Validation errors")
    warnings: List[AnalysisError] = Field(default_factory=list, description="Validation warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Validation metadata")


# ================== UTILITY FUNCTIONS ==================

def create_default_analysis_config() -> AnalysisConfiguration:
    """Create default analysis configuration."""
    return AnalysisConfiguration()


def create_default_project_config() -> ProjectIndexConfig:
    """Create default project index configuration."""
    return ProjectIndexConfig()


def validate_file_path(file_path: Union[str, Path]) -> str:
    """Validate and normalize file path."""
    if isinstance(file_path, Path):
        file_path = str(file_path)
    
    if not file_path:
        raise ValueError("File path cannot be empty")
    
    # Convert to absolute path
    path_obj = Path(file_path).resolve()
    return str(path_obj)


def calculate_file_metrics(file_results: List[FileAnalysisResult]) -> Dict[str, Any]:
    """Calculate aggregate metrics from file results."""
    if not file_results:
        return {}
    
    total_files = len(file_results)
    successful_analyses = len([r for r in file_results if r.analysis_successful])
    
    total_lines = sum(r.line_count or 0 for r in file_results)
    total_size = sum(r.file_size or 0 for r in file_results)
    
    # Language distribution
    languages = [r.language for r in file_results if r.language]
    language_counts = {lang: languages.count(lang) for lang in set(languages)}
    
    # File type distribution
    file_types = [r.file_type for r in file_results if r.file_type]
    type_counts = {ft: file_types.count(ft) for ft in set(file_types)}
    
    return {
        'total_files': total_files,
        'successful_analyses': successful_analyses,
        'success_rate': successful_analyses / total_files if total_files > 0 else 0,
        'total_lines_of_code': total_lines,
        'total_file_size': total_size,
        'average_file_size': total_size / total_files if total_files > 0 else 0,
        'language_distribution': language_counts,
        'file_type_distribution': type_counts
    }