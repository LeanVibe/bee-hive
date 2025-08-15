"""
LeanVibe Agent Hive 2.0 - Project Index System

Intelligent project analysis and context optimization for code intelligence.
Provides AST-based analysis, dependency extraction, and AI-powered context optimization.

Main Components:
- ProjectIndexer: Core orchestration for project analysis
- CodeAnalyzer: Multi-language AST parsing and code analysis
- FileMonitor: File system change monitoring
- CacheManager: Redis-based caching for analysis results
- DependencyGraph: Graph operations for dependency analysis
- ContextOptimizer: AI-powered context optimization

Example Usage:
    from app.project_index import ProjectIndexer, AnalysisConfiguration
    
    config = AnalysisConfiguration(
        enabled_languages=['python', 'javascript'],
        parse_ast=True,
        extract_dependencies=True
    )
    
    async with ProjectIndexer(config=config) as indexer:
        project = await indexer.create_project(
            name="My Project",
            root_path="/path/to/project"
        )
        
        result = await indexer.analyze_project(str(project.id))
        print(f"Analyzed {result.files_processed} files")
"""

from typing import List

# Core components
from .core import ProjectIndexer
from .analyzer import CodeAnalyzer
from .file_monitor import (
    EnhancedFileMonitor, FileMonitor, FileChangeEvent, FileChangeType,
    MonitoringStatus, ProjectMonitorConfig
)
from .cache import AdvancedCacheManager, CacheManager, CacheLayer, CacheStatistics, CacheKey
from .graph import DependencyGraph, GraphNode, GraphEdge, GraphCycle, GraphMetrics
from .context import ContextOptimizer, ContextType, ContextRelevanceScore, ContextCluster
from .incremental import IncrementalUpdateEngine, UpdateStrategy, UpdateResult, FileChangeContext
from .events import (
    EventPublisher, ProjectIndexEvent, EventType, EventFilter,
    get_event_publisher, create_file_event, create_analysis_event, create_cache_event, create_system_event
)

# Models and configurations
from .models import (
    # Configuration models
    AnalysisConfiguration,
    ProjectIndexConfig,
    
    # Analysis result models
    FileAnalysisResult,
    DependencyResult,
    AnalysisResult,
    ComplexityMetrics,
    CodeStructure,
    
    # Statistics and metrics
    ProjectStatistics,
    
    # Context optimization models
    ContextRecommendation,
    ContextClusterInfo,
    
    # Monitoring and progress
    FileChangeInfo,
    AnalysisProgress,
    AnalysisError,
    ValidationResult
)

# Utility functions and classes
from .utils import (
    PathUtils,
    FileUtils,
    HashUtils,
    GitUtils,
    ProjectUtils,
    validate_project_path,
    sanitize_filename,
    get_file_extension_info
)

# Package metadata
__version__ = "1.0.0"
__author__ = "LeanVibe Agent Hive Team"
__email__ = "team@leanvibe.com"
__description__ = "Intelligent project analysis and context optimization for code intelligence"

# Public API exports
__all__ = [
    # Core classes
    "ProjectIndexer",
    "CodeAnalyzer",
    "EnhancedFileMonitor",
    "FileMonitor",
    "AdvancedCacheManager",
    "CacheManager", 
    "DependencyGraph",
    "ContextOptimizer",
    "IncrementalUpdateEngine",
    "EventPublisher",
    
    # Configuration and models
    "AnalysisConfiguration",
    "ProjectIndexConfig",
    "FileAnalysisResult",
    "DependencyResult", 
    "AnalysisResult",
    "ComplexityMetrics",
    "CodeStructure",
    "ProjectStatistics",
    
    # Context optimization
    "ContextType",
    "ContextRelevanceScore",
    "ContextCluster",
    "ContextRecommendation",
    "ContextClusterInfo",
    
    # Graph components
    "GraphNode",
    "GraphEdge", 
    "GraphCycle",
    "GraphMetrics",
    
    # File monitoring
    "FileChangeEvent",
    "FileChangeType",
    "FileChangeInfo",
    "MonitoringStatus",
    "ProjectMonitorConfig",
    
    # Progress and errors
    "AnalysisProgress",
    "AnalysisError",
    "ValidationResult",
    
    # Caching
    "CacheLayer",
    "CacheStatistics", 
    "CacheKey",
    
    # Incremental updates
    "UpdateStrategy",
    "UpdateResult",
    "FileChangeContext",
    
    # Events
    "ProjectIndexEvent",
    "EventType",
    "EventFilter",
    "get_event_publisher",
    "create_file_event",
    "create_analysis_event",
    "create_cache_event",
    "create_system_event",
    
    # Utilities
    "PathUtils",
    "FileUtils",
    "HashUtils",
    "GitUtils",
    "ProjectUtils",
    "validate_project_path",
    "sanitize_filename",
    "get_file_extension_info",
    
    # Package info
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]

# Convenience functions for common operations
def create_default_config() -> AnalysisConfiguration:
    """
    Create default analysis configuration.
    
    Returns:
        Default AnalysisConfiguration instance
    """
    return AnalysisConfiguration()


def create_project_config(
    max_concurrent_analyses: int = 4,
    cache_enabled: bool = True,
    context_optimization_enabled: bool = True,
    monitoring_enabled: bool = True,
    incremental_updates: bool = True
) -> ProjectIndexConfig:
    """
    Create project index configuration with common settings.
    
    Args:
        max_concurrent_analyses: Maximum concurrent file analyses
        cache_enabled: Enable result caching
        context_optimization_enabled: Enable context optimization
        monitoring_enabled: Enable real-time file monitoring
        incremental_updates: Enable incremental analysis updates
        
    Returns:
        ProjectIndexConfig instance
    """
    return ProjectIndexConfig(
        max_concurrent_analyses=max_concurrent_analyses,
        cache_enabled=cache_enabled,
        context_optimization_enabled=context_optimization_enabled,
        monitoring_enabled=monitoring_enabled,
        incremental_updates=incremental_updates
    )


async def quick_analyze_project(
    project_path: str,
    languages: List[str] = None,
    include_dependencies: bool = True,
    enable_caching: bool = True
) -> AnalysisResult:
    """
    Quick project analysis with sensible defaults.
    
    Args:
        project_path: Path to project root
        languages: List of languages to analyze (default: ['python', 'javascript', 'typescript'])
        include_dependencies: Extract code dependencies
        enable_caching: Enable result caching
        
    Returns:
        AnalysisResult with analysis data
        
    Example:
        result = await quick_analyze_project("/path/to/project")
        print(f"Found {result.dependencies_found} dependencies in {result.files_processed} files")
    """
    if languages is None:
        languages = ['python', 'javascript', 'typescript', 'json']
    
    # Create configuration
    analysis_config = AnalysisConfiguration(
        enabled_languages=languages,
        extract_dependencies=include_dependencies,
        parse_ast=True,
        calculate_complexity=True
    )
    
    project_config = ProjectIndexConfig(
        analysis_config=analysis_config,
        cache_enabled=enable_caching,
        context_optimization_enabled=True
    )
    
    # Validate project path
    is_valid, error = validate_project_path(project_path)
    if not is_valid:
        raise ValueError(f"Invalid project path: {error}")
    
    # Perform analysis
    async with ProjectIndexer(config=project_config) as indexer:
        # Create project
        project = await indexer.create_project(
            name=PathUtils.normalize_path(project_path).name,
            root_path=project_path
        )
        
        # Analyze project
        result = await indexer.analyze_project(str(project.id))
        
        return result


def get_supported_languages() -> List[str]:
    """
    Get list of supported programming languages.
    
    Returns:
        List of supported language names
    """
    return [
        'python',
        'javascript', 
        'typescript',
        'json',
        'yaml',
        'toml',
        'markdown',
        'text',
        'sql',
        'shell',
        'go',
        'rust',
        'java',
        'c',
        'cpp'
    ]


def get_supported_file_types() -> List[str]:
    """
    Get list of supported file types for analysis.
    
    Returns:
        List of supported file type names
    """
    from ..models.project_index import FileType
    return [ft.value for ft in FileType]


def get_dependency_types() -> List[str]:
    """
    Get list of supported dependency types.
    
    Returns:
        List of dependency type names
    """
    from ..models.project_index import DependencyType
    return [dt.value for dt in DependencyType]


def get_context_types() -> List[str]:
    """
    Get list of available context optimization types.
    
    Returns:
        List of context type names
    """
    return [ct.value for ct in ContextType]


# Package-level configuration
class ProjectIndexSystemConfig:
    """Global configuration for the project index system."""
    
    # Performance settings
    DEFAULT_MAX_FILE_SIZE_MB = 10
    DEFAULT_MAX_LINE_COUNT = 50000
    DEFAULT_ANALYSIS_TIMEOUT = 30
    
    # Cache settings
    DEFAULT_CACHE_TTL_HOURS = 24
    DEFAULT_CACHE_ENABLED = True
    
    # Analysis settings
    DEFAULT_LANGUAGES = ['python', 'javascript', 'typescript', 'json']
    DEFAULT_PARSE_AST = True
    DEFAULT_EXTRACT_DEPENDENCIES = True
    DEFAULT_CALCULATE_COMPLEXITY = True
    
    # Monitoring settings
    DEFAULT_FILE_MONITORING_ENABLED = True
    DEFAULT_MONITORING_INTERVAL = 2.0
    
    @classmethod
    def create_optimized_config_for_size(cls, project_size: str) -> ProjectIndexConfig:
        """
        Create optimized configuration based on project size.
        
        Args:
            project_size: 'small', 'medium', 'large', or 'enterprise'
            
        Returns:
            Optimized ProjectIndexConfig
        """
        size_configs = {
            'small': {
                'max_concurrent_analyses': 2,
                'analysis_batch_size': 25,
                'max_context_files': 20
            },
            'medium': {
                'max_concurrent_analyses': 4,
                'analysis_batch_size': 50,
                'max_context_files': 50
            },
            'large': {
                'max_concurrent_analyses': 8,
                'analysis_batch_size': 100,
                'max_context_files': 100
            },
            'enterprise': {
                'max_concurrent_analyses': 16,
                'analysis_batch_size': 200,
                'max_context_files': 200
            }
        }
        
        config_params = size_configs.get(project_size, size_configs['medium'])
        
        return ProjectIndexConfig(**config_params)


# Module initialization
def _initialize_module():
    """Initialize the project index module."""
    import logging
    
    # Set up logging for the module
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Add a null handler to prevent logging errors
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())


# Initialize on import
_initialize_module()


# Compatibility aliases for backward compatibility
ProjectAnalyzer = ProjectIndexer  # Alias for backward compatibility
FileAnalyzer = CodeAnalyzer       # Alias for backward compatibility