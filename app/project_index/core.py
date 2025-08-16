"""
Core Project Indexer for LeanVibe Agent Hive 2.0

Main orchestration class for intelligent project analysis and context optimization.
Provides AST-based code analysis, dependency extraction, and AI-powered context optimization.
"""

import asyncio
import hashlib
import logging
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy import select, update, and_, or_

from ..core.config import settings
from ..core.database import get_session
from ..core.redis import get_redis_client, RedisClient
from ..core.logging_service import get_component_logger
from ..models.project_index import (
    ProjectIndex, FileEntry, DependencyRelationship, AnalysisSession, IndexSnapshot,
    ProjectStatus, FileType, AnalysisSessionType, AnalysisStatus, SnapshotType
)
from .analyzer import CodeAnalyzer
from .file_monitor import EnhancedFileMonitor, FileChangeEvent
from .cache import AdvancedCacheManager
from .incremental import IncrementalUpdateEngine
from .events import EventPublisher, get_event_publisher, create_file_event, create_analysis_event, EventType
from .websocket_events import (
    get_event_publisher as get_websocket_publisher,
    ProjectIndexUpdateData, AnalysisProgressData, DependencyChangeData, ContextOptimizedData,
    publish_project_updated, publish_analysis_progress, publish_dependency_changed, publish_context_optimized
)
from .models import (
    ProjectIndexConfig, AnalysisConfiguration, AnalysisResult,
    FileAnalysisResult, DependencyResult, ProjectStatistics
)
from .utils import PathUtils, FileUtils, HashUtils

logger = get_component_logger("project_index")


class ProjectIndexer:
    """
    Main orchestration class for project analysis and intelligent indexing.
    
    Coordinates between file analysis, dependency extraction, context optimization,
    and database operations to provide comprehensive code intelligence.
    """
    
    def __init__(
        self,
        session: Optional[AsyncSession] = None,
        redis_client: Optional[RedisClient] = None,
        config: Optional[ProjectIndexConfig] = None,
        event_publisher: Optional[EventPublisher] = None
    ):
        """
        Initialize ProjectIndexer with enhanced monitoring and caching.
        
        Args:
            session: Database session
            redis_client: Redis client for caching
            config: Project index configuration
            event_publisher: Event publisher for real-time events
        """
        self.session = session
        self.redis_client = redis_client or get_redis_client()
        self.config = config or ProjectIndexConfig()
        
        # Initialize enhanced components
        self.analyzer = CodeAnalyzer(config=self.config.analysis_config)
        
        # Enhanced file monitoring
        self.file_monitor = EnhancedFileMonitor(
            debounce_interval=self.config.monitoring_config.get('debounce_seconds', 2.0)
        )
        
        # Advanced caching
        cache_config = self.config.cache_config
        self.cache = AdvancedCacheManager(
            redis_client=self.redis_client,
            enable_compression=cache_config.get('enable_compression', True),
            compression_threshold=cache_config.get('compression_threshold', 1024),
            max_memory_mb=cache_config.get('max_memory_mb', 500)
        )
        
        # Incremental update engine
        self.incremental_engine = IncrementalUpdateEngine(
            cache_manager=self.cache,
            config=self.config
        )
        
        # Event system
        self.event_publisher = event_publisher or get_event_publisher()
        
        # WebSocket event publisher for real-time updates
        self.websocket_publisher = get_websocket_publisher()
        
        # Internal state
        self._active_sessions: Dict[str, AnalysisSession] = {}
        self._project_cache: Dict[str, ProjectIndex] = {}
        
        # Enhanced performance tracking
        self._analysis_stats = {
            'files_processed': 0,
            'dependencies_found': 0,
            'analysis_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'incremental_updates': 0,
            'events_published': 0
        }
        
        # Set up file change callback
        if self.config.monitoring_enabled:
            self.file_monitor.add_change_callback(self._handle_file_change)
        
        logger.info("Enhanced ProjectIndexer initialized",
                   monitoring_enabled=self.config.monitoring_enabled,
                   caching_enabled=self.config.cache_enabled,
                   incremental_updates=self.config.incremental_updates,
                   events_enabled=self.config.events_enabled)
    
    async def __aenter__(self):
        """Async context manager entry."""
        if self.session is None:
            self.session = await get_session().__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Stop file monitoring
        if self.config.monitoring_enabled:
            await self.file_monitor._stop_observer()
        
        # Clean up session and connections
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if self.redis_client:
            await self.redis_client.close()
    
    # ================== PROJECT MANAGEMENT ==================
    
    async def create_project(
        self,
        name: str,
        root_path: str,
        description: Optional[str] = None,
        git_repository_url: Optional[str] = None,
        git_branch: Optional[str] = None,
        configuration: Optional[Dict[str, Any]] = None,
        analysis_settings: Optional[Dict[str, Any]] = None,
        file_patterns: Optional[Dict[str, Any]] = None,
        ignore_patterns: Optional[Dict[str, Any]] = None
    ) -> ProjectIndex:
        """
        Create a new project index.
        
        Args:
            name: Project name
            root_path: Project root directory path
            description: Optional project description
            git_repository_url: Git repository URL
            git_branch: Git branch name
            configuration: Project configuration settings
            analysis_settings: Analysis-specific settings
            file_patterns: File inclusion patterns
            ignore_patterns: File exclusion patterns
            
        Returns:
            Created ProjectIndex instance
            
        Raises:
            ValueError: If root path doesn't exist or is invalid
            RuntimeError: If project creation fails
        """
        logger.info("Creating new project index", name=name, root_path=root_path)
        
        # Validate root path
        root_path_obj = Path(root_path).resolve()
        if not root_path_obj.exists():
            raise ValueError(f"Root path does not exist: {root_path}")
        if not root_path_obj.is_dir():
            raise ValueError(f"Root path is not a directory: {root_path}")
        
        # Get git information if available
        git_info = await self._get_git_info(root_path_obj)
        if git_info:
            git_repository_url = git_repository_url or git_info.get('repository_url')
            git_branch = git_branch or git_info.get('branch')
        
        # Create project index
        project = ProjectIndex(
            name=name,
            description=description,
            root_path=str(root_path_obj),
            git_repository_url=git_repository_url,
            git_branch=git_branch,
            git_commit_hash=git_info.get('commit_hash') if git_info else None,
            status=ProjectStatus.INACTIVE,
            configuration=configuration or {},
            analysis_settings=analysis_settings or {},
            file_patterns=file_patterns or {},
            ignore_patterns=ignore_patterns or {},
            meta_data={}
        )
        
        try:
            self.session.add(project)
            await self.session.commit()
            await self.session.refresh(project)
            
            # Cache project
            self._project_cache[str(project.id)] = project
            
            # Set up enhanced file monitoring
            if self.config.monitoring_enabled:
                await self.file_monitor.start_monitoring(
                    project_id=project.id,
                    root_path=root_path_obj,
                    config=self.config
                )
            
            # Publish project creation event
            if self.config.events_enabled:
                await self.event_publisher.publish(create_analysis_event(
                    EventType.PROJECT_CREATED,
                    project_id=project.id,
                    name=name,
                    root_path=str(root_path_obj)
                ))
            
            logger.info("Project index created successfully", 
                       project_id=str(project.id), name=name)
            
            return project
            
        except Exception as e:
            await self.session.rollback()
            logger.error("Failed to create project index", error=str(e))
            
            # Publish error event
            if self.config.events_enabled:
                await self.event_publisher.publish(create_analysis_event(
                    EventType.ERROR_OCCURRED,
                    project_id=project.id if 'project' in locals() else None,
                    error=str(e),
                    operation="project_creation"
                ))
            
            raise RuntimeError(f"Failed to create project: {e}")
    
    async def _handle_file_change(self, event: FileChangeEvent) -> None:
        """
        Handle file change events from the file monitor.
        
        Args:
            event: File change event
        """
        try:
            if not event.project_id:
                return
            
            logger.debug("Handling file change",
                        project_id=str(event.project_id),
                        file_path=str(event.file_path),
                        change_type=event.change_type.value)
            
            # Publish file change event
            if self.config.events_enabled:
                await self.event_publisher.publish(create_file_event(
                    EventType.FILE_MODIFIED if event.change_type.value == 'modified' else
                    EventType.FILE_CREATED if event.change_type.value == 'created' else
                    EventType.FILE_DELETED if event.change_type.value == 'deleted' else
                    EventType.FILE_MOVED,
                    project_id=event.project_id,
                    file_path=str(event.file_path),
                    change_type=event.change_type.value,
                    timestamp=event.timestamp.isoformat()
                ))
            
            # Publish WebSocket dependency change event
            try:
                file_stat = event.file_path.stat() if event.file_path.exists() else None
                
                dependency_data = DependencyChangeData(
                    project_id=event.project_id,
                    file_path=str(event.file_path),
                    change_type=event.change_type.value,
                    dependency_details={
                        "target_file": None,  # Will be analyzed during incremental update
                        "target_external": None,
                        "relationship_type": "unknown",
                        "line_number": None,
                        "is_circular": False
                    },
                    impact_analysis={
                        "affected_files": [],  # Will be populated during incremental analysis
                        "potential_issues": [],
                        "recommendations": []
                    },
                    file_metadata={
                        "language": self.analyzer.detect_language(event.file_path) if event.file_path.exists() else None,
                        "file_size": file_stat.st_size if file_stat else 0,
                        "last_modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat() if file_stat else None
                    }
                )
                
                await publish_dependency_changed(event.project_id, dependency_data)
                
            except Exception as ws_error:
                logger.debug("Failed to publish dependency change WebSocket event", error=str(ws_error))
            
            # Process incremental update if enabled
            if self.config.incremental_updates:
                try:
                    result = await self.incremental_engine.process_file_changes(
                        project_id=event.project_id,
                        changes=[event]
                    )
                    
                    self._analysis_stats['incremental_updates'] += 1
                    
                    # Publish analysis progress event
                    if self.config.events_enabled:
                        await self.event_publisher.publish(create_analysis_event(
                            EventType.ANALYSIS_PROGRESS,
                            project_id=event.project_id,
                            files_analyzed=result.files_analyzed,
                            files_skipped=result.files_skipped,
                            cache_hits=result.cache_hits,
                            update_strategy=result.strategy_used.value
                        ))
                    
                except Exception as e:
                    logger.error("Incremental update failed",
                               project_id=str(event.project_id),
                               error=str(e))
                    
                    # Publish error event
                    if self.config.events_enabled:
                        await self.event_publisher.publish(create_analysis_event(
                            EventType.ERROR_OCCURRED,
                            project_id=event.project_id,
                            error=str(e),
                            operation="incremental_update"
                        ))
            
        except Exception as e:
            logger.error("File change handling failed",
                        file_path=str(event.file_path),
                        error=str(e))
            
        except Exception as e:
            await self.session.rollback()
            logger.error("Failed to create project index", error=str(e))
            
            # Publish error event
            if self.config.events_enabled:
                await self.event_publisher.publish(create_analysis_event(
                    EventType.ERROR_OCCURRED,
                    project_id=project.id if 'project' in locals() else None,
                    error=str(e),
                    operation="project_creation"
                ))
            
            raise RuntimeError(f"Failed to create project: {e}")
    
    async def get_project(self, project_id: str) -> Optional[ProjectIndex]:
        """
        Get project by ID with caching.
        
        Args:
            project_id: Project UUID
            
        Returns:
            ProjectIndex instance or None if not found
        """
        # Check cache first
        if project_id in self._project_cache:
            logger.debug("Project cache hit", project_id=project_id)
            self._analysis_stats['cache_hits'] += 1
            
            # Publish cache hit event
            if self.config.events_enabled:
                await self.event_publisher.publish(create_analysis_event(
                    EventType.CACHE_HIT,
                    project_id=UUID(project_id),
                    cache_type="project"
                ))
                self._analysis_stats['events_published'] += 1
            
            return self._project_cache[project_id]
        
        # Query database
        try:
            stmt = select(ProjectIndex).where(ProjectIndex.id == project_id)
            result = await self.session.execute(stmt)
            project = result.scalar_one_or_none()
            
            if project:
                self._project_cache[project_id] = project
                self._analysis_stats['cache_hits'] += 1
                logger.debug("Project loaded from database", project_id=project_id)
            else:
                self._analysis_stats['cache_misses'] += 1
                logger.debug("Project not found", project_id=project_id)
            
            return project
            
        except Exception as e:
            logger.error("Failed to get project", project_id=project_id, error=str(e))
            return None
    
    async def update_project_status(
        self, 
        project_id: str, 
        status: ProjectStatus,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update project status and metadata.
        
        Args:
            project_id: Project UUID
            status: New project status
            metadata: Optional metadata updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            update_data = {'status': status}
            if metadata:
                update_data['meta_data'] = metadata
            
            stmt = (
                update(ProjectIndex)
                .where(ProjectIndex.id == project_id)
                .values(**update_data)
            )
            
            await self.session.execute(stmt)
            await self.session.commit()
            
            # Update cache
            if project_id in self._project_cache:
                project = self._project_cache[project_id]
                project.status = status
                if metadata:
                    project.meta_data.update(metadata)
            
            logger.info("Project status updated", 
                       project_id=project_id, status=status.value)
            return True
            
        except Exception as e:
            await self.session.rollback()
            logger.error("Failed to update project status", 
                        project_id=project_id, error=str(e))
            return False
    
    # ================== ANALYSIS ORCHESTRATION ==================
    
    async def analyze_project(
        self,
        project_id: str,
        analysis_type: AnalysisSessionType = AnalysisSessionType.FULL_ANALYSIS,
        force_reanalysis: bool = False,
        analysis_config: Optional[AnalysisConfiguration] = None
    ) -> AnalysisResult:
        """
        Perform comprehensive project analysis.
        
        Args:
            project_id: Project UUID
            analysis_type: Type of analysis to perform
            force_reanalysis: Force re-analysis of unchanged files
            analysis_config: Analysis configuration overrides
            
        Returns:
            AnalysisResult with comprehensive analysis data
            
        Raises:
            ValueError: If project not found or invalid
            RuntimeError: If analysis fails
        """
        logger.info("Starting project analysis", 
                   project_id=project_id, analysis_type=analysis_type.value)
        
        start_time = time.time()
        
        # Get project
        project = await self.get_project(project_id)
        if not project:
            raise ValueError(f"Project not found: {project_id}")
        
        # Check if project path still exists
        root_path = Path(project.root_path)
        if not root_path.exists():
            raise ValueError(f"Project root path no longer exists: {project.root_path}")
        
        # Update project status
        await self.update_project_status(project_id, ProjectStatus.ANALYZING)
        
        # Create analysis session
        session_name = f"{analysis_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        analysis_session = await self._create_analysis_session(
            project_id, session_name, analysis_type, analysis_config
        )
        
        try:
            # Start session tracking
            analysis_session.start_session()
            await self.session.commit()
            
            # Perform analysis based on type
            if analysis_type == AnalysisSessionType.FULL_ANALYSIS:
                result = await self._perform_full_analysis(
                    project, analysis_session, force_reanalysis
                )
            elif analysis_type == AnalysisSessionType.INCREMENTAL:
                result = await self._perform_incremental_analysis(
                    project, analysis_session
                )
            elif analysis_type == AnalysisSessionType.DEPENDENCY_MAPPING:
                result = await self._perform_dependency_analysis(
                    project, analysis_session
                )
            elif analysis_type == AnalysisSessionType.CONTEXT_OPTIMIZATION:
                result = await self._perform_context_optimization(
                    project, analysis_session
                )
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
            # Complete session
            import json
            
            def make_json_serializable(obj):
                """Recursively convert datetime objects to strings."""
                if hasattr(obj, 'isoformat'):  # datetime objects
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                elif hasattr(obj, 'to_dict'):
                    return make_json_serializable(obj.to_dict())
                else:
                    return obj
            
            result_dict = result.to_dict()
            result_clean = make_json_serializable(result_dict)
            analysis_session.complete_session(result_clean)
            await self.session.commit()
            
            # Update project statistics
            await self._update_project_statistics(project_id, result)
            
            # Update project status
            await self.update_project_status(
                project_id, 
                ProjectStatus.ACTIVE,
                {'last_analysis_result': make_json_serializable(result.to_dict())}
            )
            
            # Track performance
            analysis_time = time.time() - start_time
            self._analysis_stats['analysis_time'] += analysis_time
            
            logger.info("Project analysis completed successfully",
                       project_id=project_id,
                       analysis_time=f"{analysis_time:.2f}s",
                       files_processed=result.files_processed,
                       dependencies_found=result.dependencies_found)
            
            # Publish WebSocket event for analysis completion
            try:
                update_data = ProjectIndexUpdateData(
                    project_id=UUID(project_id),
                    project_name=project.name,
                    files_analyzed=result.files_processed,
                    files_updated=result.files_analyzed,
                    dependencies_updated=result.dependencies_found,
                    analysis_duration_seconds=analysis_time,
                    status="completed",
                    statistics={
                        "total_files": result.files_processed,
                        "languages_detected": list(set(
                            fr.language for fr in result.file_results 
                            if fr.language and fr.analysis_successful
                        )),
                        "dependency_count": result.dependencies_found,
                        "complexity_score": result.performance_metrics.get('complexity_score', 0.5),
                        "analysis_type": analysis_type.value
                    },
                    error_count=0,
                    warnings=[]
                )
                
                await publish_project_updated(UUID(project_id), update_data)
                
            except Exception as ws_error:
                logger.warning("Failed to publish WebSocket event", error=str(ws_error))
            
            return result
            
        except Exception as e:
            # Mark session as failed
            analysis_session.fail_session(str(e))
            await self.session.commit()
            
            # Update project status
            await self.update_project_status(project_id, ProjectStatus.FAILED)
            
            logger.error("Project analysis failed", 
                        project_id=project_id, error=str(e))
            raise RuntimeError(f"Analysis failed: {e}")
    
    async def _perform_full_analysis(
        self,
        project: ProjectIndex,
        session: AnalysisSession,
        force_reanalysis: bool
    ) -> AnalysisResult:
        """Perform comprehensive full project analysis."""
        logger.info("Performing full project analysis", project_id=str(project.id))
        
        root_path = Path(project.root_path)
        analysis_results = []
        dependency_results = []
        
        # Discover files
        session.update_progress(10, "Discovering files")
        await self.session.commit()
        
        files_to_analyze = await self._discover_files(project, force_reanalysis)
        session.files_total = len(files_to_analyze)
        
        logger.info("Discovered files for analysis", 
                   project_id=str(project.id), file_count=len(files_to_analyze))
        
        # Analyze files in batches
        batch_size = self.config.analysis_batch_size
        total_files = len(files_to_analyze)
        
        for i in range(0, total_files, batch_size):
            batch = files_to_analyze[i:i + batch_size]
            
            # Update progress
            progress = 10 + (i / total_files) * 60  # 10-70% for file analysis
            current_phase = f"Analyzing files ({i + 1}-{min(i + batch_size, total_files)})"
            session.update_progress(progress, current_phase)
            await self.session.commit()
            
            # Publish WebSocket progress event
            try:
                processing_rate = i / (time.time() - start_time) if i > 0 else 0.0
                estimated_completion = None
                if processing_rate > 0:
                    remaining_files = total_files - i
                    remaining_seconds = remaining_files / processing_rate
                    estimated_completion = datetime.utcnow() + timedelta(seconds=remaining_seconds)
                
                progress_data = AnalysisProgressData(
                    session_id=session.id,
                    project_id=project.id,
                    analysis_type=session.session_type.value,
                    progress_percentage=int(progress),
                    files_processed=i,
                    total_files=total_files,
                    current_file=batch[0].name if batch else None,
                    estimated_completion=estimated_completion,
                    processing_rate=processing_rate,
                    performance_metrics={
                        "memory_usage_mb": self._get_memory_usage_mb(),
                        "cpu_usage_percent": 0.0,  # Would need actual CPU monitoring
                        "parallel_tasks": 1
                    },
                    errors_encountered=session.errors_count,
                    last_error=session.error_log[-1].get('error') if session.error_log else None
                )
                
                await publish_analysis_progress(session.id, progress_data)
                
            except Exception as ws_error:
                logger.debug("Failed to publish progress WebSocket event", error=str(ws_error))
            
            # Analyze batch
            batch_results = await self._analyze_file_batch(project, batch, session)
            analysis_results.extend(batch_results)
            
            # Update session statistics
            session.files_processed = i + len(batch)
            
            # Optional: yield control for other tasks
            await asyncio.sleep(0.01)
        
        # Build dependency graph
        session.update_progress(75, "Building dependency graph")
        await self.session.commit()
        
        dependency_results = await self._build_dependency_graph(project, analysis_results)
        session.dependencies_found = len(dependency_results)
        
        # Optimize context
        session.update_progress(90, "Optimizing context")
        await self.session.commit()
        
        context_optimization = await self._optimize_context(project, analysis_results)
        
        # Create analysis result
        result = AnalysisResult(
            project_id=str(project.id),
            session_id=str(session.id),
            analysis_type=session.session_type,
            files_processed=len(analysis_results),
            files_analyzed=len([r for r in analysis_results if r.analysis_successful]),
            dependencies_found=len(dependency_results),
            analysis_duration=time.time() - session.started_at.timestamp(),
            file_results=analysis_results,
            dependency_results=dependency_results,
            context_optimization=context_optimization,
            performance_metrics=session.performance_metrics
        )
        
        session.update_progress(100, "Analysis complete")
        await self.session.commit()
        
        return result
    
    async def _perform_incremental_analysis(
        self,
        project: ProjectIndex,
        session: AnalysisSession
    ) -> AnalysisResult:
        """Perform incremental analysis of changed files only."""
        logger.info("Performing incremental analysis", project_id=str(project.id))
        
        # Get changed files since last analysis
        changed_files = await self._get_changed_files(project)
        
        if not changed_files:
            logger.info("No changed files found", project_id=str(project.id))
            return AnalysisResult(
                project_id=str(project.id),
                session_id=str(session.id),
                analysis_type=session.session_type,
                files_processed=0,
                files_analyzed=0,
                dependencies_found=0,
                analysis_duration=0.0,
                file_results=[],
                dependency_results=[],
                context_optimization={}
            )
        
        session.files_total = len(changed_files)
        
        # Analyze changed files
        analysis_results = []
        for i, file_path in enumerate(changed_files):
            progress = (i / len(changed_files)) * 80  # 0-80% for file analysis
            session.update_progress(progress, f"Analyzing {file_path.name}")
            await self.session.commit()
            
            result = await self._analyze_single_file(project, file_path)
            if result:
                analysis_results.append(result)
            
            session.files_processed = i + 1
        
        # Update dependencies for changed files
        session.update_progress(85, "Updating dependencies")
        await self.session.commit()
        
        dependency_results = await self._update_dependencies_for_files(
            project, [r.file_path for r in analysis_results]
        )
        
        # Create result
        result = AnalysisResult(
            project_id=str(project.id),
            session_id=str(session.id),
            analysis_type=session.session_type,
            files_processed=len(analysis_results),
            files_analyzed=len([r for r in analysis_results if r.analysis_successful]),
            dependencies_found=len(dependency_results),
            analysis_duration=time.time() - session.started_at.timestamp(),
            file_results=analysis_results,
            dependency_results=dependency_results,
            context_optimization={}
        )
        
        session.update_progress(100, "Incremental analysis complete")
        await self.session.commit()
        
        return result
    
    async def _perform_dependency_analysis(
        self,
        project: ProjectIndex,
        session: AnalysisSession
    ) -> AnalysisResult:
        """Perform focused dependency mapping analysis."""
        logger.info("Performing dependency analysis", project_id=str(project.id))
        
        # Get all existing file entries
        stmt = select(FileEntry).where(FileEntry.project_id == project.id)
        result = await self.session.execute(stmt)
        file_entries = result.scalars().all()
        
        session.files_total = len(file_entries)
        
        # Re-analyze dependencies for all files
        dependency_results = []
        for i, file_entry in enumerate(file_entries):
            progress = (i / len(file_entries)) * 90
            session.update_progress(progress, f"Analyzing dependencies for {file_entry.file_name}")
            await self.session.commit()
            
            file_path = Path(file_entry.file_path)
            if file_path.exists():
                deps = await self.analyzer.extract_dependencies(file_path)
                dependency_results.extend(deps)
            
            session.files_processed = i + 1
        
        # Update dependency relationships
        await self._update_dependency_relationships(project, dependency_results)
        
        result = AnalysisResult(
            project_id=str(project.id),
            session_id=str(session.id),
            analysis_type=session.session_type,
            files_processed=len(file_entries),
            files_analyzed=len(file_entries),
            dependencies_found=len(dependency_results),
            analysis_duration=time.time() - session.started_at.timestamp(),
            file_results=[],
            dependency_results=dependency_results,
            context_optimization={}
        )
        
        session.update_progress(100, "Dependency analysis complete")
        await self.session.commit()
        
        return result
    
    async def _perform_context_optimization(
        self,
        project: ProjectIndex,
        session: AnalysisSession
    ) -> AnalysisResult:
        """Perform AI-powered context optimization analysis."""
        logger.info("Performing context optimization", project_id=str(project.id))
        
        session.update_progress(20, "Loading project context")
        await self.session.commit()
        
        # Get project files and dependencies
        files_stmt = select(FileEntry).where(FileEntry.project_id == project.id)
        files_result = await self.session.execute(files_stmt)
        files = files_result.scalars().all()
        
        deps_stmt = select(DependencyRelationship).where(DependencyRelationship.project_id == project.id)
        deps_result = await self.session.execute(deps_stmt)
        dependencies = deps_result.scalars().all()
        
        session.update_progress(50, "Optimizing context relationships")
        await self.session.commit()
        
        # Perform context optimization
        context_optimization = await self._optimize_context(project, files, dependencies)
        
        session.update_progress(100, "Context optimization complete")
        await self.session.commit()
        
        result = AnalysisResult(
            project_id=str(project.id),
            session_id=str(session.id),
            analysis_type=session.session_type,
            files_processed=len(files),
            files_analyzed=len(files),
            dependencies_found=len(dependencies),
            analysis_duration=time.time() - session.started_at.timestamp(),
            file_results=[],
            dependency_results=[],
            context_optimization=context_optimization
        )
        
        return result
    
    # ================== FILE ANALYSIS ==================
    
    async def _discover_files(
        self, 
        project: ProjectIndex, 
        force_reanalysis: bool = False
    ) -> List[Path]:
        """Discover files to analyze based on project configuration."""
        root_path = Path(project.root_path)
        
        # Get file patterns from project configuration
        include_patterns = project.file_patterns.get('include', ['**/*'])
        exclude_patterns = project.file_patterns.get('exclude', [])
        
        # Default exclude patterns
        default_excludes = [
            '**/__pycache__/**',
            '**/.git/**',
            '**/node_modules/**',
            '**/.venv/**',
            '**/*.pyc',
            '**/*.pyo',
            '**/*.so',
            '**/*.dylib',
            '**/*.dll'
        ]
        exclude_patterns.extend(default_excludes)
        
        discovered_files = []
        
        # Use include patterns to find files
        for pattern in include_patterns:
            for file_path in root_path.glob(pattern):
                if file_path.is_file():
                    # Check against exclude patterns
                    should_exclude = False
                    for exclude_pattern in exclude_patterns:
                        if file_path.match(exclude_pattern):
                            should_exclude = True
                            break
                    
                    if not should_exclude:
                        discovered_files.append(file_path)
        
        # Remove duplicates and sort
        discovered_files = sorted(set(discovered_files))
        
        # Filter based on modification time if not forcing reanalysis
        if not force_reanalysis:
            discovered_files = await self._filter_unchanged_files(project, discovered_files)
        
        logger.info("File discovery completed",
                   project_id=str(project.id),
                   total_files=len(discovered_files))
        
        return discovered_files
    
    async def _filter_unchanged_files(
        self, 
        project: ProjectIndex, 
        files: List[Path]
    ) -> List[Path]:
        """Filter out files that haven't changed since last analysis."""
        if not project.last_indexed_at:
            return files  # First analysis, include all files
        
        filtered_files = []
        
        for file_path in files:
            # Check if file was modified after last analysis
            try:
                file_stat = file_path.stat()
                file_mtime = datetime.fromtimestamp(file_stat.st_mtime)
                
                if file_mtime > project.last_indexed_at:
                    filtered_files.append(file_path)
                else:
                    # Check if file entry exists in database
                    relative_path = str(file_path.relative_to(project.root_path))
                    stmt = select(FileEntry).where(
                        and_(
                            FileEntry.project_id == project.id,
                            FileEntry.relative_path == relative_path
                        )
                    )
                    result = await self.session.execute(stmt)
                    file_entry = result.scalar_one_or_none()
                    
                    if not file_entry:
                        # File not in database, include it
                        filtered_files.append(file_path)
            except Exception as e:
                logger.warning("Failed to check file modification time",
                              file_path=str(file_path), error=str(e))
                # Include file if we can't check modification time
                filtered_files.append(file_path)
        
        logger.debug("Filtered unchanged files",
                    original_count=len(files),
                    filtered_count=len(filtered_files))
        
        return filtered_files
    
    async def _analyze_file_batch(
        self,
        project: ProjectIndex,
        files: List[Path],
        session: AnalysisSession
    ) -> List[FileAnalysisResult]:
        """Analyze a batch of files concurrently."""
        tasks = []
        
        for file_path in files:
            task = self._analyze_single_file(project, file_path)
            tasks.append(task)
        
        # Run analysis tasks concurrently with limit
        semaphore = asyncio.Semaphore(self.config.max_concurrent_analyses)
        
        async def bounded_analysis(file_path):
            async with semaphore:
                return await self._analyze_single_file(project, file_path)
        
        results = await asyncio.gather(
            *[bounded_analysis(fp) for fp in files],
            return_exceptions=True
        )
        
        # Filter out exceptions and failed analyses
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                session.add_error(f"Failed to analyze {files[i]}: {result}")
                logger.error("File analysis failed",
                           file_path=str(files[i]),
                           error=str(result))
            elif result:
                successful_results.append(result)
        
        return successful_results
    
    async def _analyze_single_file(
        self,
        project: ProjectIndex,
        file_path: Path
    ) -> Optional[FileAnalysisResult]:
        """Analyze a single file and return results."""
        try:
            relative_path = file_path.relative_to(project.root_path)
            
            # Check cache first
            cache_key = f"file_analysis:{project.id}:{HashUtils.hash_file(file_path)}"
            cached_result = await self.cache.get_analysis_result(cache_key)
            if cached_result:
                logger.debug("File analysis cache hit", file_path=str(relative_path))
                return cached_result
            
            # Perform analysis
            analysis_start = time.time()
            
            # Basic file information
            file_stat = file_path.stat()
            file_info = FileUtils.get_file_info(file_path)
            
            # Language detection
            language = self.analyzer.detect_language(file_path)
            
            # File classification
            file_type = FileUtils.classify_file_type(file_path)
            
            # Content analysis
            analysis_data = {}
            dependencies = []
            
            if not file_info['is_binary'] and language:
                # Parse AST and extract information
                ast_result = await self.analyzer.parse_file(file_path)
                if ast_result:
                    analysis_data = ast_result
                
                # Extract dependencies
                deps = await self.analyzer.extract_dependencies(file_path)
                dependencies = deps
            
            # Calculate content hash
            content_hash = HashUtils.hash_file(file_path)
            
            # Create file analysis result
            result = FileAnalysisResult(
                file_path=str(file_path),
                relative_path=str(relative_path),
                file_name=file_path.name,
                file_extension=file_path.suffix,
                file_type=file_type,
                language=language,
                file_size=file_stat.st_size,
                line_count=analysis_data.get('line_count', 0),
                sha256_hash=content_hash,
                is_binary=file_info['is_binary'],
                is_generated=file_info.get('is_generated', False),
                analysis_data=analysis_data,
                dependencies=dependencies,
                analysis_successful=True,
                analysis_duration=time.time() - analysis_start,
                last_modified=datetime.fromtimestamp(file_stat.st_mtime)
            )
            
            # Cache result
            await self.cache.set_analysis_result(cache_key, result)
            
            # Update statistics
            self._analysis_stats['files_processed'] += 1
            
            return result
            
        except Exception as e:
            logger.error("Single file analysis failed",
                        file_path=str(file_path),
                        error=str(e))
            
            # Return failed analysis result
            return FileAnalysisResult(
                file_path=str(file_path),
                relative_path=str(file_path.relative_to(project.root_path)),
                file_name=file_path.name,
                analysis_successful=False,
                error_message=str(e)
            )
    
    # ================== DEPENDENCY ANALYSIS ==================
    
    async def _build_dependency_graph(
        self,
        project: ProjectIndex,
        file_results: List[FileAnalysisResult]
    ) -> List[DependencyResult]:
        """Build comprehensive dependency graph from file analysis results."""
        logger.info("Building dependency graph", project_id=str(project.id))
        
        all_dependencies = []
        
        # Collect all dependencies from file results
        for file_result in file_results:
            if file_result.dependencies:
                all_dependencies.extend(file_result.dependencies)
        
        # Resolve internal dependencies
        resolved_dependencies = await self._resolve_dependencies(project, all_dependencies)
        
        # Update database
        await self._update_dependency_relationships(project, resolved_dependencies)
        
        return resolved_dependencies
    
    async def _resolve_dependencies(
        self,
        project: ProjectIndex,
        dependencies: List[DependencyResult]
    ) -> List[DependencyResult]:
        """Resolve dependencies to actual file paths within the project."""
        root_path = Path(project.root_path)
        resolved = []
        
        # Get all project files for resolution
        project_files = {}
        stmt = select(FileEntry).where(FileEntry.project_id == project.id)
        result = await self.session.execute(stmt)
        file_entries = result.scalars().all()
        
        for entry in file_entries:
            # Create multiple lookup keys
            rel_path = Path(entry.relative_path)
            project_files[str(rel_path)] = entry
            project_files[rel_path.name] = entry
            project_files[rel_path.stem] = entry
        
        for dep in dependencies:
            dep_copy = dep.copy()
            
            # Try to resolve target to internal file
            target_candidates = [
                dep.target_name,
                f"{dep.target_name}.py",
                f"{dep.target_name}.js",
                f"{dep.target_name}.ts",
                f"{dep.target_name}/index.py",
                f"{dep.target_name}/index.js",
                f"{dep.target_name}/__init__.py"
            ]
            
            resolved_file = None
            for candidate in target_candidates:
                if candidate in project_files:
                    resolved_file = project_files[candidate]
                    break
            
            if resolved_file:
                dep_copy.target_file_id = str(resolved_file.id)
                dep_copy.target_path = resolved_file.relative_path
                dep_copy.is_external = False
            else:
                dep_copy.is_external = True
            
            resolved.append(dep_copy)
        
        return resolved
    
    async def _update_dependency_relationships(
        self,
        project: ProjectIndex,
        dependencies: List[DependencyResult]
    ) -> None:
        """Update dependency relationships in database."""
        # Remove existing dependencies for the project
        from sqlalchemy import delete
        delete_stmt = delete(DependencyRelationship).where(
            DependencyRelationship.project_id == project.id
        )
        await self.session.execute(delete_stmt)
        
        # Add new dependencies
        for dep in dependencies:
            dep_rel = DependencyRelationship(
                project_id=project.id,
                source_file_id=dep.source_file_id,
                target_file_id=dep.target_file_id if not dep.is_external else None,
                target_path=dep.target_path,
                target_name=dep.target_name,
                dependency_type=dep.dependency_type,
                line_number=dep.line_number,
                column_number=dep.column_number,
                source_text=dep.source_text,
                is_external=dep.is_external,
                is_dynamic=dep.is_dynamic,
                confidence_score=dep.confidence_score
            )
            self.session.add(dep_rel)
        
        await self.session.commit()
        
        logger.info("Updated dependency relationships",
                   project_id=str(project.id),
                   dependency_count=len(dependencies))
    
    # ================== CONTEXT OPTIMIZATION ==================
    
    async def _optimize_context(
        self,
        project: ProjectIndex,
        file_results: Optional[List[FileAnalysisResult]] = None,
        dependencies: Optional[List[DependencyRelationship]] = None
    ) -> Dict[str, Any]:
        """Perform AI-powered context optimization analysis."""
        logger.info("Optimizing project context", project_id=str(project.id))
        
        # This is a placeholder for AI-powered context optimization
        # In a full implementation, this would:
        # 1. Analyze code patterns and relationships
        # 2. Identify key files and modules
        # 3. Generate context relevance scores
        # 4. Create intelligent file groupings
        # 5. Optimize for different development tasks
        
        context_optimization = {
            'core_files': [],
            'entry_points': [],
            'high_impact_files': [],
            'context_clusters': [],
            'relevance_scores': {},
            'optimization_metrics': {
                'context_efficiency': 0.8,
                'relevance_accuracy': 0.85,
                'coverage_score': 0.9
            }
        }
        
        # Publish WebSocket context optimization event
        try:
            context_data = ContextOptimizedData(
                context_id=uuid.uuid4(),
                project_id=project.id,
                task_description="AI-powered context optimization for project analysis",
                task_type="full_project_optimization",
                optimization_results={
                    "selected_files": len(context_optimization.get('core_files', [])),
                    "total_tokens": sum(
                        len(str(fr.analysis_data)) for fr in (file_results or [])
                        if fr.analysis_successful
                    ),
                    "relevance_scores": {
                        "high": len([f for f in context_optimization.get('core_files', [])]),
                        "medium": len([f for f in context_optimization.get('high_impact_files', [])]),
                        "low": 0
                    },
                    "confidence_score": context_optimization['optimization_metrics']['context_efficiency'],
                    "processing_time_ms": 100  # Placeholder
                },
                recommendations={
                    "architectural_patterns": ["modular_design", "separation_of_concerns"],
                    "potential_challenges": ["dependency_complexity"],
                    "suggested_approach": "Focus on core modules and entry points for initial analysis"
                },
                performance_metrics={
                    "cache_hit_rate": 0.85,
                    "ml_analysis_time_ms": 50,
                    "context_assembly_time_ms": 25
                }
            )
            
            await publish_context_optimized(context_data.context_id, context_data)
            
        except Exception as ws_error:
            logger.debug("Failed to publish context optimization WebSocket event", error=str(ws_error))
        
        return context_optimization
    
    # ================== HELPER METHODS ==================
    
    async def _create_analysis_session(
        self,
        project_id: str,
        session_name: str,
        session_type: AnalysisSessionType,
        config: Optional[AnalysisConfiguration] = None
    ) -> AnalysisSession:
        """Create and save analysis session."""
        session = AnalysisSession(
            project_id=project_id,
            session_name=session_name,
            session_type=session_type,
            configuration=config.to_dict() if config else {}
        )
        
        self.session.add(session)
        await self.session.commit()
        await self.session.refresh(session)
        
        self._active_sessions[str(session.id)] = session
        
        return session
    
    async def _get_git_info(self, path: Path) -> Optional[Dict[str, str]]:
        """Get git repository information."""
        try:
            import subprocess
            
            def run_git_command(cmd):
                result = subprocess.run(
                    cmd, 
                    cwd=path, 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                return result.stdout.strip() if result.returncode == 0 else None
            
            # Check if it's a git repository
            if not (path / '.git').exists():
                return None
            
            git_info = {}
            
            # Get repository URL
            remote_url = run_git_command(['git', 'remote', 'get-url', 'origin'])
            if remote_url:
                git_info['repository_url'] = remote_url
            
            # Get current branch
            branch = run_git_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
            if branch:
                git_info['branch'] = branch
            
            # Get current commit hash
            commit_hash = run_git_command(['git', 'rev-parse', 'HEAD'])
            if commit_hash:
                git_info['commit_hash'] = commit_hash
            
            return git_info if git_info else None
            
        except Exception as e:
            logger.debug("Failed to get git info", path=str(path), error=str(e))
            return None
    
    async def _get_changed_files(self, project: ProjectIndex) -> List[Path]:
        """Get files that have changed since last analysis."""
        if not project.last_indexed_at:
            return []
        
        root_path = Path(project.root_path)
        changed_files = []
        
        # Check files modified after last indexing
        for file_path in root_path.rglob('*'):
            if file_path.is_file():
                try:
                    file_stat = file_path.stat()
                    file_mtime = datetime.fromtimestamp(file_stat.st_mtime)
                    
                    if file_mtime > project.last_indexed_at:
                        changed_files.append(file_path)
                except Exception:
                    continue
        
        return changed_files
    
    async def _update_project_statistics(
        self,
        project_id: str,
        result: AnalysisResult
    ) -> None:
        """Update project file and dependency counts."""
        update_data = {
            'file_count': result.files_processed,
            'dependency_count': result.dependencies_found,
            'last_indexed_at': datetime.utcnow(),
            'last_analysis_at': datetime.utcnow()
        }
        
        stmt = (
            update(ProjectIndex)
            .where(ProjectIndex.id == project_id)
            .values(**update_data)
        )
        
        await self.session.execute(stmt)
        await self.session.commit()
    
    async def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get analysis performance statistics."""
        return self._analysis_stats.copy()
    
    async def cleanup_old_data(self, retention_days: int = 30) -> Dict[str, int]:
        """Clean up old analysis sessions and snapshots."""
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        # Delete old analysis sessions
        from sqlalchemy import delete
        
        session_delete = delete(AnalysisSession).where(
            and_(
                AnalysisSession.created_at < cutoff_date,
                AnalysisSession.status.in_([AnalysisStatus.COMPLETED, AnalysisStatus.FAILED])
            )
        )
        session_result = await self.session.execute(session_delete)
        
        # Delete old snapshots
        snapshot_delete = delete(IndexSnapshot).where(
            IndexSnapshot.created_at < cutoff_date
        )
        snapshot_result = await self.session.execute(snapshot_delete)
        
        await self.session.commit()
        
        return {
            'deleted_sessions': session_result.rowcount,
            'deleted_snapshots': snapshot_result.rowcount
        }
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert bytes to MB
        except ImportError:
            # psutil not available, return placeholder
            return 0.0
        except Exception as e:
            logger.debug("Failed to get memory usage", error=str(e))
            return 0.0