"""
File System Monitor for LeanVibe Agent Hive 2.0

Monitors file system changes for incremental project analysis updates.
Tracks file modifications, additions, and deletions for efficient indexing.
"""

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from uuid import UUID

import structlog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.events import (
    FileCreatedEvent, FileModifiedEvent, FileDeletedEvent, FileMovedEvent,
    DirCreatedEvent, DirModifiedEvent, DirDeletedEvent, DirMovedEvent
)

# Import for type checking to avoid circular imports
if TYPE_CHECKING:
    from .models import ProjectIndexConfig

logger = structlog.get_logger()


class FileChangeType(Enum):
    """Types of file system changes."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


@dataclass
class FileChangeEvent:
    """Represents a file system change event."""
    file_path: Path
    change_type: FileChangeType
    timestamp: datetime
    project_id: Optional[UUID] = None
    old_path: Optional[Path] = None  # For moved files
    file_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.metadata is None:
            self.metadata = {}
        
        # Add file size and modification time if file exists
        if self.file_path.exists() and self.file_path.is_file():
            try:
                stat = self.file_path.stat()
                self.metadata.update({
                    'size': stat.st_size,
                    'mtime': stat.st_mtime,
                    'mode': stat.st_mode
                })
            except (OSError, PermissionError):
                pass


@dataclass
class MonitoringStatus:
    """Status information for file monitoring."""
    is_active: bool
    monitored_projects: int
    total_files_tracked: int
    events_processed: int
    last_scan_time: Optional[datetime] = None
    performance_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ProjectMonitorConfig:
    """Configuration for monitoring a specific project."""
    project_id: UUID
    root_path: Path
    include_patterns: List[str] = field(default_factory=lambda: ['*'])
    exclude_patterns: List[str] = field(default_factory=list)
    watch_subdirectories: bool = True
    debounce_seconds: float = 2.0
    max_file_size_mb: int = 10
    
    def __post_init__(self):
        """Validate configuration."""
        self.root_path = self.root_path.resolve()
        if not self.root_path.exists():
            raise ValueError(f"Root path does not exist: {self.root_path}")
        if not self.root_path.is_dir():
            raise ValueError(f"Root path is not a directory: {self.root_path}")


class EnhancedFileSystemEventHandler(FileSystemEventHandler):
    """Enhanced file system event handler with debouncing and filtering."""
    
    def __init__(self, monitor: 'EnhancedFileMonitor', config: ProjectMonitorConfig):
        super().__init__()
        self.monitor = monitor
        self.config = config
        self.logger = structlog.get_logger().bind(project_id=str(config.project_id))
    
    def on_any_event(self, event: FileSystemEvent):
        """Handle any file system event."""
        if event.is_directory:
            return  # Skip directory events for now
        
        try:
            # Convert watchdog event to our FileChangeEvent
            file_path = Path(event.src_path)
            
            # Apply filtering
            if not self._should_process_file(file_path):
                return
            
            # Determine change type
            change_type = self._get_change_type(event)
            if change_type is None:
                return
            
            # Create our event
            file_event = FileChangeEvent(
                file_path=file_path,
                change_type=change_type,
                timestamp=datetime.utcnow(),
                project_id=self.config.project_id,
                old_path=Path(event.dest_path) if hasattr(event, 'dest_path') else None
            )
            
            # Add to debounced queue
            self.monitor._add_to_debounce_queue(file_event)
            
        except Exception as e:
            self.logger.error("Error processing file event", 
                            event_path=event.src_path, 
                            error=str(e))
    
    def _get_change_type(self, event: FileSystemEvent) -> Optional[FileChangeType]:
        """Convert watchdog event to our change type."""
        if isinstance(event, (FileCreatedEvent, DirCreatedEvent)):
            return FileChangeType.CREATED
        elif isinstance(event, (FileModifiedEvent, DirModifiedEvent)):
            return FileChangeType.MODIFIED
        elif isinstance(event, (FileDeletedEvent, DirDeletedEvent)):
            return FileChangeType.DELETED
        elif isinstance(event, (FileMovedEvent, DirMovedEvent)):
            return FileChangeType.MOVED
        return None
    
    def _should_process_file(self, file_path: Path) -> bool:
        """Check if file should be processed based on patterns."""
        # Check file size limit
        try:
            if file_path.exists() and file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > self.config.max_file_size_mb:
                    return False
        except (OSError, PermissionError):
            return False
        
        # Check exclude patterns
        for pattern in self.config.exclude_patterns:
            if self._matches_pattern(file_path, pattern):
                return False
        
        # Check include patterns
        if self.config.include_patterns:
            for pattern in self.config.include_patterns:
                if self._matches_pattern(file_path, pattern):
                    return True
            return False  # No include pattern matched
        
        return True  # No include patterns specified, allow all (except excluded)
    
    def _matches_pattern(self, file_path: Path, pattern: str) -> bool:
        """Check if file matches a glob pattern."""
        import fnmatch
        return fnmatch.fnmatch(str(file_path), pattern) or fnmatch.fnmatch(file_path.name, pattern)


class EnhancedFileMonitor:
    """
    Enhanced file system monitor using watchdog for real-time change detection.
    
    Features:
    - Real-time file system monitoring using watchdog
    - Debounced change detection to avoid flooding
    - Intelligent filtering based on project configuration
    - Cross-platform compatibility
    - Memory-efficient monitoring for large directory trees
    - Graceful error handling for permission issues
    """
    
    def __init__(self, debounce_interval: float = 2.0):
        """
        Initialize EnhancedFileMonitor.
        
        Args:
            debounce_interval: Time to wait before processing batched changes
        """
        self.debounce_interval = debounce_interval
        
        # Core components
        self.observer = Observer()
        self._monitored_projects: Dict[UUID, ProjectMonitorConfig] = {}
        self._event_handlers: Dict[UUID, EnhancedFileSystemEventHandler] = {}
        self._change_callbacks: List[Callable[[FileChangeEvent], Union[None, Any]]] = []
        
        # Debouncing system
        self._debounce_queue: Dict[str, FileChangeEvent] = {}  # file_path -> latest_event
        self._debounce_task: Optional[asyncio.Task] = None
        self._processing_queue = deque()
        
        # State management
        self._is_monitoring = False
        self._last_processed_events: Dict[str, datetime] = {}
        
        # Performance tracking
        self._stats = {
            'events_received': 0,
            'events_processed': 0,
            'events_debounced': 0,
            'callbacks_executed': 0,
            'errors_encountered': 0,
            'start_time': None,
            'last_event_time': None
        }
        
        # Default ignore patterns for all projects
        self._global_ignore_patterns = {
            '*.pyc', '*.pyo', '*.pyd', '__pycache__/*',
            '.git/*', '.svn/*', '.hg/*', '.bzr/*',
            'node_modules/*', '.venv/*', 'venv/*', '.env',
            '*.log', '*.tmp', '*.temp', '*.swp', '*.swo',
            '.DS_Store', 'Thumbs.db', '*.orig', '*.bak',
            '*.lock', '.pytest_cache/*', '__pycache__/*',
            '.mypy_cache/*', '.coverage', 'htmlcov/*'
        }
        
        self.logger = structlog.get_logger()
    
    async def start_monitoring(self, project_id: UUID, root_path: Path, config: 'ProjectIndexConfig') -> bool:
        """
        Start monitoring a project directory for file changes.
        
        Args:
            project_id: Unique project identifier
            root_path: Project root directory path
            config: Project index configuration
            
        Returns:
            True if successfully started monitoring, False otherwise
        """
        try:
            # Create project monitoring configuration
            monitor_config = ProjectMonitorConfig(
                project_id=project_id,
                root_path=root_path,
                include_patterns=config.monitoring_config.get('include_patterns', ['*']),
                exclude_patterns=config.monitoring_config.get('exclude_patterns', []) + list(self._global_ignore_patterns),
                debounce_seconds=config.monitoring_config.get('debounce_seconds', self.debounce_interval),
                max_file_size_mb=config.monitoring_config.get('max_file_size_mb', 10)
            )
            
            # Store project configuration
            self._monitored_projects[project_id] = monitor_config
            
            # Create event handler for this project
            event_handler = EnhancedFileSystemEventHandler(self, monitor_config)
            self._event_handlers[project_id] = event_handler
            
            # Start watching the directory
            self.observer.schedule(
                event_handler,
                str(root_path),
                recursive=monitor_config.watch_subdirectories
            )
            
            # Start observer if not already running
            if not self._is_monitoring:
                await self._start_observer()
            
            self.logger.info("Project monitoring started", 
                           project_id=str(project_id), 
                           path=str(root_path),
                           patterns_included=len(monitor_config.include_patterns),
                           patterns_excluded=len(monitor_config.exclude_patterns))
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to start project monitoring", 
                            project_id=str(project_id), 
                            error=str(e))
            return False
    
    async def stop_monitoring(self, project_id: UUID) -> bool:
        """
        Stop monitoring a specific project.
        
        Args:
            project_id: Project identifier to stop monitoring
            
        Returns:
            True if successfully stopped, False otherwise
        """
        try:
            # Remove event handler if exists
            if project_id in self._event_handlers:
                handler = self._event_handlers[project_id]
                config = self._monitored_projects[project_id]
                
                # Unschedule the handler
                self.observer.unschedule_all()
                
                # Re-schedule remaining projects
                del self._monitored_projects[project_id]
                del self._event_handlers[project_id]
                
                # Re-add remaining projects
                for remaining_id, remaining_config in self._monitored_projects.items():
                    if remaining_id in self._event_handlers:
                        self.observer.schedule(
                            self._event_handlers[remaining_id],
                            str(remaining_config.root_path),
                            recursive=remaining_config.watch_subdirectories
                        )
                
                self.logger.info("Project monitoring stopped", project_id=str(project_id))
            
            # Stop observer if no projects left
            if not self._monitored_projects and self._is_monitoring:
                await self._stop_observer()
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to stop project monitoring", 
                            project_id=str(project_id), 
                            error=str(e))
            return False
    
    async def _start_observer(self) -> bool:
        """
        Start the watchdog observer for file system monitoring.
        
        Returns:
            True if successfully started, False otherwise
        """
        if self._is_monitoring:
            self.logger.warning("File monitoring already active")
            return True
        
        try:
            # Start the watchdog observer
            self.observer.start()
            self._is_monitoring = True
            self._stats['start_time'] = datetime.utcnow()
            
            # Start debounce processing task
            self._debounce_task = asyncio.create_task(self._process_debounced_events())
            
            self.logger.info("Enhanced file monitoring started", 
                           monitored_projects=len(self._monitored_projects),
                           debounce_interval=self.debounce_interval)
            
            return True
            
        except Exception as e:
            self.logger.error("Failed to start file monitoring", error=str(e))
            self._is_monitoring = False
            return False
    
    async def _stop_observer(self) -> bool:
        """
        Stop the watchdog observer and cleanup.
        
        Returns:
            True if successfully stopped, False otherwise
        """
        if not self._is_monitoring:
            return True
        
        try:
            self._is_monitoring = False
            
            # Stop debounce task
            if self._debounce_task:
                self._debounce_task.cancel()
                try:
                    await self._debounce_task
                except asyncio.CancelledError:
                    pass
                self._debounce_task = None
            
            # Stop observer
            self.observer.stop()
            self.observer.join(timeout=5.0)
            
            # Process any remaining events
            await self._flush_debounce_queue()
            
            self.logger.info("Enhanced file monitoring stopped")
            return True
            
        except Exception as e:
            self.logger.error("Failed to stop file monitoring", error=str(e))
            return False
    
    def add_change_callback(self, callback: Callable[[FileChangeEvent], Union[None, Any]]) -> None:
        """
        Add a callback function to be called when file changes are detected.
        
        Args:
            callback: Function to call with FileChangeEvent (can be async or sync)
        """
        self._change_callbacks.append(callback)
        self.logger.debug("Change callback added", callback_count=len(self._change_callbacks))
    
    def remove_change_callback(self, callback: Callable[[FileChangeEvent], Union[None, Any]]) -> None:
        """
        Remove a change callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self._change_callbacks:
            self._change_callbacks.remove(callback)
            self.logger.debug("Change callback removed", callback_count=len(self._change_callbacks))
    
    def _add_to_debounce_queue(self, event: FileChangeEvent) -> None:
        """
        Add event to debounce queue for batch processing.
        
        Args:
            event: File change event to queue
        """
        file_key = str(event.file_path)
        self._debounce_queue[file_key] = event
        self._stats['events_received'] += 1
        self._stats['last_event_time'] = datetime.utcnow()
    
    async def _process_debounced_events(self) -> None:
        """
        Process debounced events in batches to avoid flooding.
        """
        self.logger.info("Debounced event processing started")
        
        try:
            while self._is_monitoring:
                await asyncio.sleep(self.debounce_interval)
                
                if not self._debounce_queue:
                    continue
                
                # Extract events to process
                events_to_process = list(self._debounce_queue.values())
                self._debounce_queue.clear()
                
                # Group events by project for efficient processing
                events_by_project = defaultdict(list)
                for event in events_to_process:
                    if event.project_id:
                        events_by_project[event.project_id].append(event)
                
                # Process events
                for project_id, project_events in events_by_project.items():
                    await self._process_project_events(project_id, project_events)
                
                self._stats['events_processed'] += len(events_to_process)
                
                if events_to_process:
                    self.logger.debug("Processed debounced events", 
                                    event_count=len(events_to_process),
                                    project_count=len(events_by_project))
                
        except asyncio.CancelledError:
            self.logger.info("Debounced event processing cancelled")
        except Exception as e:
            self.logger.error("Error in debounced event processing", error=str(e))
            self._stats['errors_encountered'] += 1
    
    async def _process_project_events(self, project_id: UUID, events: List[FileChangeEvent]) -> None:
        """
        Process a batch of events for a specific project.
        
        Args:
            project_id: Project identifier
            events: List of events to process
        """
        try:
            # Deduplicate events for the same file (keep latest)
            latest_events = {}
            for event in events:
                file_key = str(event.file_path)
                if file_key not in latest_events or event.timestamp > latest_events[file_key].timestamp:
                    latest_events[file_key] = event
            
            # Calculate file hashes for existing files
            for event in latest_events.values():
                if event.change_type != FileChangeType.DELETED and event.file_path.exists():
                    try:
                        event.file_hash = await self._calculate_file_hash(event.file_path)
                    except Exception as e:
                        self.logger.warning("Failed to calculate file hash", 
                                          file_path=str(event.file_path), 
                                          error=str(e))
            
            # Notify callbacks
            for event in latest_events.values():
                await self._notify_callbacks(event)
            
            self._stats['events_debounced'] += len(events) - len(latest_events)
            
        except Exception as e:
            self.logger.error("Error processing project events", 
                            project_id=str(project_id), 
                            error=str(e))
            self._stats['errors_encountered'] += 1
    
    async def _flush_debounce_queue(self) -> None:
        """
        Flush any remaining events in the debounce queue.
        """
        if not self._debounce_queue:
            return
        
        self.logger.info("Flushing remaining debounced events", 
                        event_count=len(self._debounce_queue))
        
        events_to_process = list(self._debounce_queue.values())
        self._debounce_queue.clear()
        
        for event in events_to_process:
            if event.project_id:
                await self._process_project_events(event.project_id, [event])
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate SHA-256 hash of file content.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal hash string
        """
        import hashlib
        
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error("Failed to calculate file hash", 
                            file_path=str(file_path), 
                            error=str(e))
            raise
    
    async def get_monitoring_status(self, project_id: Optional[UUID] = None) -> MonitoringStatus:
        """
        Get current monitoring status.
        
        Args:
            project_id: Optional project ID to get specific status
            
        Returns:
            MonitoringStatus object
        """
        total_files = 0
        if project_id and project_id in self._monitored_projects:
            # Count files in specific project
            config = self._monitored_projects[project_id]
            try:
                total_files = sum(1 for _ in config.root_path.rglob('*') if _.is_file())
            except Exception:
                total_files = 0
        else:
            # Count files in all projects
            for config in self._monitored_projects.values():
                try:
                    total_files += sum(1 for _ in config.root_path.rglob('*') if _.is_file())
                except Exception:
                    continue
        
        return MonitoringStatus(
            is_active=self._is_monitoring,
            monitored_projects=len(self._monitored_projects),
            total_files_tracked=total_files,
            events_processed=self._stats['events_processed'],
            last_scan_time=self._stats.get('last_event_time'),
            performance_stats=self._stats.copy()
        )
    
    async def force_scan(self, project_id: Optional[UUID] = None) -> Dict[str, int]:
        """
        Force immediate processing of any pending events.
        
        Args:
            project_id: Specific project to scan, or None for all projects
            
        Returns:
            Dictionary with scan statistics
        """
        stats = {'processed_events': 0, 'projects_affected': 0}
        
        try:
            if not self._debounce_queue:
                self.logger.info("No pending events to process")
                return stats
            
            # Filter events by project if specified
            events_to_process = []
            if project_id:
                events_to_process = [e for e in self._debounce_queue.values() if e.project_id == project_id]
                # Remove processed events from queue
                for event in events_to_process:
                    self._debounce_queue.pop(str(event.file_path), None)
            else:
                events_to_process = list(self._debounce_queue.values())
                self._debounce_queue.clear()
            
            # Group by project and process
            events_by_project = defaultdict(list)
            for event in events_to_process:
                if event.project_id:
                    events_by_project[event.project_id].append(event)
            
            for pid, project_events in events_by_project.items():
                await self._process_project_events(pid, project_events)
            
            stats['processed_events'] = len(events_to_process)
            stats['projects_affected'] = len(events_by_project)
            
            self.logger.info("Manual scan completed", **stats)
            return stats
            
        except Exception as e:
            self.logger.error("Failed to complete manual scan", error=str(e))
            return stats
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        uptime = None
        if self._stats['start_time']:
            uptime = (datetime.utcnow() - self._stats['start_time']).total_seconds()
        
        return {
            'is_monitoring': self._is_monitoring,
            'monitored_projects': len(self._monitored_projects),
            'debounce_interval': self.debounce_interval,
            'callback_count': len(self._change_callbacks),
            'pending_events': len(self._debounce_queue),
            'uptime_seconds': uptime,
            **self._stats
        }
    
    # ================== PRIVATE METHODS ==================
    
    async def _notify_callbacks(self, event: FileChangeEvent) -> None:
        """
        Notify all registered callbacks about a file change.
        
        Args:
            event: File change event to notify about
        """
        if not self._change_callbacks:
            return
        
        for callback in self._change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
                self._stats['callbacks_executed'] += 1
            except Exception as e:
                self.logger.error("Callback notification failed", 
                                callback=getattr(callback, '__name__', str(callback)), 
                                error=str(e))
                self._stats['errors_encountered'] += 1
    
    def add_ignore_pattern(self, pattern: str, project_id: Optional[UUID] = None) -> None:
        """
        Add a file pattern to ignore during monitoring.
        
        Args:
            pattern: Pattern to ignore (supports * wildcards)
            project_id: Optional project ID to add pattern for specific project
        """
        if project_id and project_id in self._monitored_projects:
            config = self._monitored_projects[project_id]
            if pattern not in config.exclude_patterns:
                config.exclude_patterns.append(pattern)
                self.logger.debug("Project-specific ignore pattern added", 
                                project_id=str(project_id), 
                                pattern=pattern)
        else:
            self._global_ignore_patterns.add(pattern)
            self.logger.debug("Global ignore pattern added", pattern=pattern)
    
    def remove_ignore_pattern(self, pattern: str, project_id: Optional[UUID] = None) -> None:
        """
        Remove an ignore pattern.
        
        Args:
            pattern: Pattern to remove
            project_id: Optional project ID to remove pattern from specific project
        """
        if project_id and project_id in self._monitored_projects:
            config = self._monitored_projects[project_id]
            if pattern in config.exclude_patterns:
                config.exclude_patterns.remove(pattern)
                self.logger.debug("Project-specific ignore pattern removed", 
                                project_id=str(project_id), 
                                pattern=pattern)
        else:
            self._global_ignore_patterns.discard(pattern)
            self.logger.debug("Global ignore pattern removed", pattern=pattern)
    
    def get_ignore_patterns(self, project_id: Optional[UUID] = None) -> Set[str]:
        """
        Get current ignore patterns.
        
        Args:
            project_id: Optional project ID to get project-specific patterns
            
        Returns:
            Set of ignore patterns
        """
        if project_id and project_id in self._monitored_projects:
            config = self._monitored_projects[project_id]
            return set(config.exclude_patterns)
        return self._global_ignore_patterns.copy()
    
    async def get_changes_since(
        self, 
        project_id: UUID, 
        since: datetime
    ) -> List[FileChangeEvent]:
        """
        Get file changes for a project since a specific timestamp.
        Note: This is a compatibility method. Real-time monitoring with callbacks is preferred.
        
        Args:
            project_id: Project identifier
            since: Timestamp to check changes since
            
        Returns:
            List of FileChangeEvent objects
        """
        if project_id not in self._monitored_projects:
            return []
        
        # For real-time monitoring, we don't maintain historical states
        # This method could be enhanced to query a persistent event store
        self.logger.warning("get_changes_since called - consider using real-time callbacks instead",
                          project_id=str(project_id))
        
        # Return empty list as real-time events are handled via callbacks
        return []
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._stop_observer()
    
    def __del__(self):
        """Cleanup on deletion."""
        if self._is_monitoring:
            try:
                self.observer.stop()
            except Exception:
                pass
    
    
    
# Backward compatibility alias
FileMonitor = EnhancedFileMonitor