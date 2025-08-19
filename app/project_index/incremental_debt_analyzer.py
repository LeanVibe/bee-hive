"""
Incremental Technical Debt Analyzer for LeanVibe Agent Hive 2.0

Real-time debt analysis with file monitor integration. Provides efficient
incremental updates when files change, minimizing analysis overhead while
maintaining comprehensive debt tracking.
"""

import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path

import structlog
import numpy as np
from watchdog.events import FileSystemEvent

from .file_monitor import FileChangeEvent, FileChangeType, EnhancedFileMonitor
from .debt_analyzer import TechnicalDebtAnalyzer, DebtAnalysisResult, DebtItem, DebtCategory, DebtSeverity
from .advanced_debt_detector import AdvancedDebtDetector, AdvancedDebtPattern
from .websocket_events import publish_project_updated, ProjectIndexEventType
from .incremental import IncrementalUpdateEngine
from ..models.project_index import ProjectIndex, FileEntry
from ..core.database import get_session

logger = structlog.get_logger()


@dataclass
class DebtChangeEvent:
    """Event representing a change in technical debt."""
    project_id: str
    file_path: str
    change_type: FileChangeType
    previous_debt_score: float
    current_debt_score: float
    debt_delta: float
    affected_patterns: List[str]
    remediation_priority: str  # "immediate", "high", "medium", "low"
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebtAnalysisCache:
    """Cache for debt analysis results with invalidation tracking."""
    file_hash: str
    debt_score: float
    debt_items: List[DebtItem]
    advanced_patterns: List[AdvancedDebtPattern]
    analysis_timestamp: datetime
    dependencies_hash: str
    is_stale: bool = False


@dataclass
class IncrementalDebtMetrics:
    """Metrics for incremental debt analysis performance."""
    files_analyzed: int = 0
    files_cached: int = 0
    total_analysis_time: float = 0.0
    cache_hit_rate: float = 0.0
    debt_change_events: int = 0
    significant_changes: int = 0
    average_analysis_time: float = 0.0
    peak_memory_usage: int = 0


class IncrementalDebtAnalyzer:
    """
    Real-time incremental technical debt analyzer.
    
    Integrates with file monitoring system to provide efficient, event-driven
    debt analysis that only processes changed files and their dependencies.
    """
    
    def __init__(
        self,
        debt_analyzer: TechnicalDebtAnalyzer,
        advanced_detector: AdvancedDebtDetector,
        incremental_engine: IncrementalUpdateEngine
    ):
        """Initialize incremental debt analyzer."""
        self.debt_analyzer = debt_analyzer
        self.advanced_detector = advanced_detector
        self.incremental_engine = incremental_engine
        
        # Caching and state management
        self.debt_cache: Dict[str, DebtAnalysisCache] = {}
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.analysis_queue = asyncio.Queue()
        self.metrics = IncrementalDebtMetrics()
        
        # Event handling
        self.change_callbacks: List[Callable[[DebtChangeEvent], None]] = []
        self.event_history = deque(maxlen=1000)  # Keep last 1000 events
        
        # Configuration
        self.config = {
            'debt_change_threshold': 0.1,  # Minimum change to trigger notification
            'batch_analysis_delay': 1.0,   # Seconds to batch rapid changes
            'max_concurrent_analysis': 5,   # Parallel analysis limit
            'cache_ttl_hours': 24,          # Cache time-to-live
            'dependency_depth': 2           # Max dependency traversal depth
        }
        
        # State tracking
        self._analysis_tasks: Set[asyncio.Task] = set()
        self._last_analysis_batch: Dict[str, float] = {}
        self._running = False
    
    async def start_monitoring(self, project: ProjectIndex) -> None:
        """Start incremental debt monitoring for a project."""
        logger.info("Starting incremental debt monitoring", project_id=str(project.id))
        
        self._running = True
        
        # Initialize dependency graph
        await self._build_dependency_graph(project)
        
        # Warm up cache with initial analysis
        await self._warm_cache(project)
        
        # Start analysis queue processor
        analysis_task = asyncio.create_task(self._process_analysis_queue())
        self._analysis_tasks.add(analysis_task)
        
        logger.info(
            "Incremental debt monitoring started",
            project_id=str(project.id),
            cached_files=len(self.debt_cache),
            dependencies_tracked=sum(len(deps) for deps in self.dependency_graph.values())
        )
    
    async def stop_monitoring(self) -> None:
        """Stop incremental debt monitoring."""
        logger.info("Stopping incremental debt monitoring")
        
        self._running = False
        
        # Cancel all analysis tasks
        for task in self._analysis_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._analysis_tasks:
            await asyncio.gather(*self._analysis_tasks, return_exceptions=True)
        
        # Clear state
        self.debt_cache.clear()
        self.dependency_graph.clear()
        self._analysis_tasks.clear()
        
        logger.info("Incremental debt monitoring stopped")
    
    async def handle_file_change(self, event: FileChangeEvent) -> None:
        """Handle file change event for incremental debt analysis."""
        if not self._running:
            return
            
        start_time = time.time()
        
        try:
            logger.debug(
                "Handling file change for debt analysis",
                file_path=event.file_path,
                change_type=event.change_type.value,
                project_id=event.project_id
            )
            
            # Determine analysis scope
            files_to_analyze = await self._determine_analysis_scope(event)
            
            if not files_to_analyze:
                logger.debug("No files require debt analysis", file_path=event.file_path)
                return
            
            # Queue analysis request
            await self.analysis_queue.put({
                'event': event,
                'files': files_to_analyze,
                'timestamp': time.time()
            })
            
            # Update metrics
            self.metrics.files_analyzed += len(files_to_analyze)
            
        except Exception as e:
            logger.error(
                "Error handling file change for debt analysis",
                file_path=event.file_path,
                error=str(e)
            )
        finally:
            processing_time = time.time() - start_time
            self.metrics.total_analysis_time += processing_time
    
    async def get_incremental_debt_status(self, project_id: str) -> Dict[str, Any]:
        """Get current incremental debt analysis status."""
        cache_stats = self._calculate_cache_stats()
        
        return {
            'monitoring_active': self._running,
            'cached_files': len(self.debt_cache),
            'dependency_relationships': sum(len(deps) for deps in self.dependency_graph.values()),
            'pending_analysis': self.analysis_queue.qsize(),
            'active_tasks': len(self._analysis_tasks),
            'metrics': {
                'files_analyzed': self.metrics.files_analyzed,
                'cache_hit_rate': cache_stats['hit_rate'],
                'average_analysis_time': cache_stats['avg_time'],
                'debt_change_events': self.metrics.debt_change_events,
                'significant_changes': self.metrics.significant_changes
            },
            'recent_events': list(self.event_history)[-10:],  # Last 10 events
            'configuration': self.config
        }
    
    def add_change_callback(self, callback: Callable[[DebtChangeEvent], None]) -> None:
        """Add callback for debt change events."""
        self.change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[DebtChangeEvent], None]) -> None:
        """Remove debt change callback."""
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
    
    async def force_analysis(self, project_id: str, file_paths: List[str]) -> DebtAnalysisResult:
        """Force debt analysis for specific files (bypass cache)."""
        logger.info(
            "Forcing debt analysis",
            project_id=project_id,
            files_count=len(file_paths)
        )
        
        # Clear cache for specified files
        for file_path in file_paths:
            if file_path in self.debt_cache:
                del self.debt_cache[file_path]
        
        # Perform fresh analysis
        async with get_session() as session:
            # Get project and file entries
            from sqlalchemy import select
            from sqlalchemy.orm import selectinload
            
            stmt = select(ProjectIndex).where(ProjectIndex.id == project_id).options(
                selectinload(ProjectIndex.file_entries)
            )
            result = await session.execute(stmt)
            project = result.scalar_one_or_none()
            
            if not project:
                raise ValueError(f"Project {project_id} not found")
            
            # Filter to requested files
            target_files = [
                fe for fe in project.file_entries 
                if fe.file_path in file_paths
            ]
            
            # Analyze each file
            debt_items = []
            for file_entry in target_files:
                file_debt = await self._analyze_single_file(file_entry, project, session)
                debt_items.extend(file_debt)
            
            # Create analysis result
            total_debt_score = sum(item.debt_score for item in debt_items) / max(len(target_files), 1)
            
            category_scores = defaultdict(float)
            for item in debt_items:
                category_scores[item.category.value] += item.debt_score
            
            return DebtAnalysisResult(
                project_id=project_id,
                total_debt_score=total_debt_score,
                debt_items=debt_items,
                category_scores=dict(category_scores),
                file_count=len(target_files),
                lines_of_code=sum(fe.line_count or 0 for fe in target_files),
                analysis_duration=0.0,  # Not tracking for forced analysis
                recommendations=[]
            )
    
    # Private methods
    
    async def _process_analysis_queue(self) -> None:
        """Process queued analysis requests with batching."""
        batch_requests = {}
        last_batch_time = 0
        
        while self._running:
            try:
                # Wait for analysis request with timeout
                try:
                    request = await asyncio.wait_for(
                        self.analysis_queue.get(), 
                        timeout=self.config['batch_analysis_delay']
                    )
                    
                    # Add to batch
                    event = request['event']
                    batch_key = event.project_id
                    
                    if batch_key not in batch_requests:
                        batch_requests[batch_key] = {
                            'files': set(),
                            'events': [],
                            'first_timestamp': request['timestamp']
                        }
                    
                    batch_requests[batch_key]['files'].update(request['files'])
                    batch_requests[batch_key]['events'].append(event)
                    
                except asyncio.TimeoutError:
                    # Process accumulated batches
                    pass
                
                # Process batches if delay elapsed or queue is empty
                current_time = time.time()
                if (current_time - last_batch_time >= self.config['batch_analysis_delay'] 
                    or self.analysis_queue.empty()):
                    
                    for batch_key, batch_data in batch_requests.items():
                        if batch_data['files']:
                            await self._process_analysis_batch(batch_key, batch_data)
                    
                    batch_requests.clear()
                    last_batch_time = current_time
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in analysis queue processor", error=str(e))
                await asyncio.sleep(1)  # Prevent tight error loop
    
    async def _process_analysis_batch(self, project_id: str, batch_data: Dict[str, Any]) -> None:
        """Process a batch of analysis requests."""
        files_to_analyze = list(batch_data['files'])
        events = batch_data['events']
        
        logger.debug(
            "Processing debt analysis batch",
            project_id=project_id,
            files_count=len(files_to_analyze),
            events_count=len(events)
        )
        
        start_time = time.time()
        
        try:
            # Get project data
            async with get_session() as session:
                from sqlalchemy import select
                from sqlalchemy.orm import selectinload
                
                stmt = select(ProjectIndex).where(ProjectIndex.id == project_id).options(
                    selectinload(ProjectIndex.file_entries)
                )
                result = await session.execute(stmt)
                project = result.scalar_one_or_none()
                
                if not project:
                    logger.warning("Project not found for debt analysis", project_id=project_id)
                    return
                
                # Process each file
                debt_changes = []
                
                for file_path in files_to_analyze:
                    file_entry = next(
                        (fe for fe in project.file_entries if fe.file_path == file_path),
                        None
                    )
                    
                    if file_entry:
                        change_event = await self._analyze_file_incrementally(
                            file_entry, project, session
                        )
                        if change_event:
                            debt_changes.append(change_event)
                
                # Publish aggregated debt update event
                if debt_changes:
                    await self._publish_debt_update_event(project_id, debt_changes)
                
                # Update metrics
                analysis_time = time.time() - start_time
                self.metrics.total_analysis_time += analysis_time
                if debt_changes:
                    self.metrics.debt_change_events += len(debt_changes)
                    self.metrics.significant_changes += len([
                        c for c in debt_changes 
                        if abs(c.debt_delta) > self.config['debt_change_threshold']
                    ])
                
                logger.debug(
                    "Debt analysis batch completed",
                    project_id=project_id,
                    files_analyzed=len(files_to_analyze),
                    debt_changes=len(debt_changes),
                    analysis_time=analysis_time
                )
                
        except Exception as e:
            logger.error(
                "Error processing debt analysis batch",
                project_id=project_id,
                error=str(e)
            )
    
    async def _analyze_file_incrementally(
        self, 
        file_entry: FileEntry, 
        project: ProjectIndex,
        session
    ) -> Optional[DebtChangeEvent]:
        """Analyze a single file incrementally with caching."""
        file_path = file_entry.file_path
        
        # Check cache validity
        cache_entry = self.debt_cache.get(file_path)
        current_hash = await self._calculate_file_hash(file_path)
        
        if (cache_entry and 
            cache_entry.file_hash == current_hash and 
            not cache_entry.is_stale and
            (datetime.utcnow() - cache_entry.analysis_timestamp).total_seconds() < 
             self.config['cache_ttl_hours'] * 3600):
            
            self.metrics.files_cached += 1
            logger.debug("Using cached debt analysis", file_path=file_path)
            return None
        
        # Perform fresh analysis
        start_time = time.time()
        
        try:
            # Get previous debt score
            previous_score = cache_entry.debt_score if cache_entry else 0.0
            
            # Analyze file debt
            debt_items = await self.debt_analyzer._analyze_file_debt(file_entry, session)
            
            # Run advanced pattern detection
            base_analysis = DebtAnalysisResult(
                project_id=str(project.id),
                total_debt_score=sum(item.debt_score for item in debt_items),
                debt_items=debt_items,
                category_scores={},
                file_count=1,
                lines_of_code=file_entry.line_count or 0,
                analysis_duration=0.0,
                recommendations=[]
            )
            
            advanced_patterns = await self.advanced_detector._detect_architectural_patterns(
                project, base_analysis, np.array([])
            )
            
            # Calculate new debt score
            current_score = sum(item.debt_score for item in debt_items)
            debt_delta = current_score - previous_score
            
            # Update cache
            dependencies_hash = await self._calculate_dependencies_hash(file_path)
            self.debt_cache[file_path] = DebtAnalysisCache(
                file_hash=current_hash,
                debt_score=current_score,
                debt_items=debt_items,
                advanced_patterns=advanced_patterns,
                analysis_timestamp=datetime.utcnow(),
                dependencies_hash=dependencies_hash
            )
            
            # Create change event if significant
            if abs(debt_delta) >= self.config['debt_change_threshold']:
                priority = self._calculate_remediation_priority(debt_delta, debt_items)
                
                change_event = DebtChangeEvent(
                    project_id=str(project.id),
                    file_path=file_path,
                    change_type=FileChangeType.MODIFIED,
                    previous_debt_score=previous_score,
                    current_debt_score=current_score,
                    debt_delta=debt_delta,
                    affected_patterns=[p.pattern_name for p in advanced_patterns],
                    remediation_priority=priority,
                    metadata={
                        'debt_items_count': len(debt_items),
                        'advanced_patterns_count': len(advanced_patterns),
                        'analysis_time': time.time() - start_time
                    }
                )
                
                # Add to event history
                self.event_history.append(change_event)
                
                # Notify callbacks
                for callback in self.change_callbacks:
                    try:
                        await callback(change_event) if asyncio.iscoroutinefunction(callback) else callback(change_event)
                    except Exception as e:
                        logger.warning("Error in debt change callback", error=str(e))
                
                return change_event
            
            return None
            
        except Exception as e:
            logger.error(
                "Error in incremental file debt analysis",
                file_path=file_path,
                error=str(e)
            )
            return None
    
    async def _determine_analysis_scope(self, event: FileChangeEvent) -> List[str]:
        """Determine which files need analysis based on change event."""
        files_to_analyze = []
        
        # Always analyze the changed file
        file_entry = event.metadata.get('file_entry') if hasattr(event, 'metadata') and event.metadata else None
        if not file_entry or (hasattr(file_entry, 'is_binary') and file_entry.is_binary):
            return files_to_analyze
            
        files_to_analyze.append(str(event.file_path))
        
        # Add dependent files if significant change
        if event.change_type in [FileChangeType.MODIFIED, FileChangeType.CREATED]:
            dependents = self._get_dependent_files(str(event.file_path), max_depth=self.config['dependency_depth'])
            files_to_analyze.extend(dependents)
        
        # Remove duplicates
        return list(set(files_to_analyze))
    
    def _get_dependent_files(self, file_path: str, max_depth: int = 2) -> List[str]:
        """Get files that depend on the given file."""
        dependents = set()
        queue = [(file_path, 0)]
        visited = set()
        
        while queue:
            current_file, depth = queue.pop(0)
            
            if current_file in visited or depth >= max_depth:
                continue
                
            visited.add(current_file)
            
            # Find files that depend on current_file
            for dependent, dependencies in self.dependency_graph.items():
                if current_file in dependencies and dependent not in visited:
                    dependents.add(dependent)
                    queue.append((dependent, depth + 1))
        
        return list(dependents)
    
    async def _build_dependency_graph(self, project: ProjectIndex) -> None:
        """Build dependency graph for the project."""
        logger.debug("Building dependency graph for debt analysis")
        
        # Use existing dependency relationships
        for dep_rel in project.dependency_relationships:
            from_file = next(
                (fe.file_path for fe in project.file_entries if str(fe.id) == dep_rel.from_file_id),
                None
            )
            to_file = next(
                (fe.file_path for fe in project.file_entries if str(fe.id) == dep_rel.to_file_id),
                None
            )
            
            if from_file and to_file:
                self.dependency_graph[from_file].add(to_file)
        
        logger.debug(
            "Dependency graph built",
            files_with_dependencies=len(self.dependency_graph),
            total_dependencies=sum(len(deps) for deps in self.dependency_graph.values())
        )
    
    async def _warm_cache(self, project: ProjectIndex) -> None:
        """Warm up debt analysis cache with existing results."""
        logger.debug("Warming debt analysis cache")
        
        # This would typically load from database or run initial analysis
        # For now, we'll mark cache as empty and let lazy loading handle it
        logger.debug("Cache warming completed (lazy loading enabled)")
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of file contents for cache invalidation."""
        try:
            with open(file_path, 'rb') as f:
                import hashlib
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return str(time.time())  # Fallback to timestamp
    
    async def _calculate_dependencies_hash(self, file_path: str) -> str:
        """Calculate hash of file dependencies for cache invalidation."""
        dependencies = self.dependency_graph.get(file_path, set())
        dep_string = ''.join(sorted(dependencies))
        import hashlib
        return hashlib.md5(dep_string.encode()).hexdigest()
    
    def _calculate_remediation_priority(self, debt_delta: float, debt_items: List[DebtItem]) -> str:
        """Calculate remediation priority based on debt change and severity."""
        if debt_delta > 0.5 or any(item.severity == DebtSeverity.CRITICAL for item in debt_items):
            return "immediate"
        elif debt_delta > 0.3 or any(item.severity == DebtSeverity.HIGH for item in debt_items):
            return "high"
        elif debt_delta > 0.1:
            return "medium"
        else:
            return "low"
    
    def _calculate_cache_stats(self) -> Dict[str, float]:
        """Calculate cache performance statistics."""
        total_requests = self.metrics.files_analyzed + self.metrics.files_cached
        hit_rate = (self.metrics.files_cached / total_requests) if total_requests > 0 else 0.0
        avg_time = (self.metrics.total_analysis_time / self.metrics.files_analyzed) if self.metrics.files_analyzed > 0 else 0.0
        
        return {
            'hit_rate': hit_rate,
            'avg_time': avg_time,
            'total_requests': total_requests
        }
    
    async def _publish_debt_update_event(self, project_id: str, debt_changes: List[DebtChangeEvent]) -> None:
        """Publish debt update event via WebSocket."""
        try:
            event_data = {
                'project_id': project_id,
                'changes_count': len(debt_changes),
                'significant_changes': len([c for c in debt_changes if c.remediation_priority in ['immediate', 'high']]),
                'total_debt_delta': sum(c.debt_delta for c in debt_changes),
                'affected_files': [c.file_path for c in debt_changes],
                'timestamp': datetime.utcnow().isoformat()
            }
            
            from uuid import UUID
            # Create proper data structure for project update
            from .websocket_events import ProjectIndexUpdateData
            update_data = ProjectIndexUpdateData(
                project_id=UUID(project_id),
                project_name="Unknown",  # Would need to be passed in
                files_analyzed=event_data['changes_count'],
                files_updated=event_data['changes_count'],
                dependencies_updated=0,
                analysis_duration_seconds=0.0,
                status="completed",
                statistics={
                    "debt_changes": event_data['changes_count'],
                    "total_debt_delta": event_data['total_debt_delta'],
                    "affected_files": event_data['affected_files']
                }
            )
            try:
                project_uuid = UUID(project_id) if len(project_id) == 36 else UUID(int=hash(project_id) & ((1<<128)-1))
            except (ValueError, TypeError):
                import uuid
                project_uuid = uuid.uuid4()  # Fallback for invalid UUIDs
            await publish_project_updated(project_uuid, update_data)
            
            logger.debug(
                "Published debt update event",
                project_id=project_id,
                changes_count=len(debt_changes)
            )
            
        except Exception as e:
            logger.error("Failed to publish debt update event", error=str(e))
    
    async def _analyze_single_file(self, file_entry: FileEntry, project: ProjectIndex, session) -> List[DebtItem]:
        """Analyze a single file and return debt items."""
        return await self.debt_analyzer._analyze_file_debt(file_entry, session)