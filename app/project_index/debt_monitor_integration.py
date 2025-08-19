"""
Debt Monitor Integration for LeanVibe Agent Hive 2.0

Integrates incremental debt analysis with the file monitoring system
to provide real-time debt tracking and notifications.
"""

import asyncio
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime

import structlog

from .file_monitor import EnhancedFileMonitor, FileChangeEvent, FileChangeType
from .incremental_debt_analyzer import IncrementalDebtAnalyzer, DebtChangeEvent
from .debt_analyzer import TechnicalDebtAnalyzer
from .advanced_debt_detector import AdvancedDebtDetector
from .incremental import IncrementalUpdateEngine
from .ml_analyzer import MLAnalyzer
from .historical_analyzer import HistoricalAnalyzer
from .websocket_events import publish_project_updated, ProjectIndexEventType, ProjectIndexUpdateData
from ..models.project_index import ProjectIndex
from ..core.database import get_session

logger = structlog.get_logger()


@dataclass
class DebtMonitorConfig:
    """Configuration for debt monitoring integration."""
    enabled: bool = True
    debt_change_threshold: float = 0.1
    batch_analysis_delay: float = 1.0
    max_concurrent_analysis: int = 5
    notification_enabled: bool = True
    dashboard_updates_enabled: bool = True
    alert_critical_debt: bool = True
    historical_tracking: bool = True


class DebtMonitorIntegration:
    """
    Integration layer between file monitoring and debt analysis.
    
    Coordinates real-time debt analysis when files change and provides
    comprehensive debt tracking capabilities.
    """
    
    def __init__(self, config: Optional[DebtMonitorConfig] = None):
        """Initialize debt monitor integration."""
        self.config = config or DebtMonitorConfig()
        
        # Core components (will be initialized when needed)
        self.file_monitor: Optional[EnhancedFileMonitor] = None
        self.debt_analyzer: Optional[TechnicalDebtAnalyzer] = None
        self.advanced_detector: Optional[AdvancedDebtDetector] = None
        self.incremental_analyzer: Optional[IncrementalDebtAnalyzer] = None
        
        # State tracking
        self.monitored_projects: Dict[str, ProjectIndex] = {}
        self.active_monitors: Dict[str, EnhancedFileMonitor] = {}
        self.debt_trends: Dict[str, List[float]] = {}
        
        # Performance metrics
        self.total_files_monitored = 0
        self.total_debt_events = 0
        self.active_since: Optional[datetime] = None
    
    async def initialize_components(self) -> None:
        """Initialize all required components."""
        if not self.config.enabled:
            logger.info("Debt monitoring disabled by configuration")
            return
        
        logger.info("Initializing debt monitor integration components")
        
        try:
            # Initialize core analyzers
            self.debt_analyzer = TechnicalDebtAnalyzer()
            
            # Initialize ML components
            ml_analyzer = MLAnalyzer()
            historical_analyzer = HistoricalAnalyzer()
            
            self.advanced_detector = AdvancedDebtDetector(
                self.debt_analyzer,
                ml_analyzer,
                historical_analyzer
            )
            
            # Initialize incremental analysis
            incremental_engine = IncrementalUpdateEngine()
            self.incremental_analyzer = IncrementalDebtAnalyzer(
                self.debt_analyzer,
                self.advanced_detector,
                incremental_engine
            )
            
            # Configure incremental analyzer
            self.incremental_analyzer.config.update({
                'debt_change_threshold': self.config.debt_change_threshold,
                'batch_analysis_delay': self.config.batch_analysis_delay,
                'max_concurrent_analysis': self.config.max_concurrent_analysis
            })
            
            # Register debt change callback
            self.incremental_analyzer.add_change_callback(self._handle_debt_change)
            
            self.active_since = datetime.utcnow()
            
            logger.info("Debt monitor integration components initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize debt monitor components", error=str(e))
            raise
    
    async def start_monitoring_project(self, project: ProjectIndex) -> None:
        """Start debt monitoring for a specific project."""
        if not self.config.enabled:
            return
            
        project_id = str(project.id)
        
        if project_id in self.monitored_projects:
            logger.warning("Project already being monitored", project_id=project_id)
            return
        
        logger.info("Starting debt monitoring for project", project_id=project_id, project_name=project.name)
        
        try:
            # Initialize components if needed
            if not self.incremental_analyzer:
                await self.initialize_components()
            
            # Create file monitor for this project
            file_monitor = EnhancedFileMonitor(
                project_path=project.root_path,
                project_id=project_id,
                include_patterns=["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.java", "*.cpp", "*.c", "*.h"],
                exclude_patterns=["**/node_modules/**", "**/__pycache__/**", "**/venv/**", "**/env/**", "**/dist/**"]
            )
            
            # Register debt analysis callback
            file_monitor.add_change_callback(self._handle_file_change)
            
            # Start file monitoring
            await file_monitor.start_monitoring()
            
            # Start incremental debt analysis
            await self.incremental_analyzer.start_monitoring(project)
            
            # Store references
            self.monitored_projects[project_id] = project
            self.active_monitors[project_id] = file_monitor
            self.debt_trends[project_id] = []
            
            # Update metrics
            self.total_files_monitored += len(project.file_entries)
            
            logger.info(
                "Debt monitoring started for project",
                project_id=project_id,
                files_monitored=len(project.file_entries)
            )
            
        except Exception as e:
            logger.error(
                "Failed to start debt monitoring for project",
                project_id=project_id,
                error=str(e)
            )
            raise
    
    async def stop_monitoring_project(self, project_id: str) -> None:
        """Stop debt monitoring for a specific project."""
        if project_id not in self.monitored_projects:
            logger.warning("Project not being monitored", project_id=project_id)
            return
        
        logger.info("Stopping debt monitoring for project", project_id=project_id)
        
        try:
            # Stop file monitor
            if project_id in self.active_monitors:
                await self.active_monitors[project_id].stop_monitoring()
                del self.active_monitors[project_id]
            
            # Stop incremental analyzer (if this was the last project)
            if len(self.monitored_projects) == 1 and self.incremental_analyzer:
                await self.incremental_analyzer.stop_monitoring()
            
            # Clean up references
            if project_id in self.monitored_projects:
                project = self.monitored_projects[project_id]
                self.total_files_monitored -= len(project.file_entries)
                del self.monitored_projects[project_id]
            
            if project_id in self.debt_trends:
                del self.debt_trends[project_id]
            
            logger.info("Debt monitoring stopped for project", project_id=project_id)
            
        except Exception as e:
            logger.error(
                "Error stopping debt monitoring for project",
                project_id=project_id,
                error=str(e)
            )
    
    async def stop_all_monitoring(self) -> None:
        """Stop all debt monitoring."""
        logger.info("Stopping all debt monitoring")
        
        # Stop all project monitors
        for project_id in list(self.monitored_projects.keys()):
            await self.stop_monitoring_project(project_id)
        
        # Stop incremental analyzer
        if self.incremental_analyzer:
            await self.incremental_analyzer.stop_monitoring()
        
        # Reset state
        self.monitored_projects.clear()
        self.active_monitors.clear()
        self.debt_trends.clear()
        self.total_files_monitored = 0
        self.active_since = None
        
        logger.info("All debt monitoring stopped")
    
    async def get_monitoring_status(self) -> Dict[str, any]:
        """Get comprehensive monitoring status."""
        status = {
            'enabled': self.config.enabled,
            'active_since': self.active_since.isoformat() if self.active_since else None,
            'monitored_projects_count': len(self.monitored_projects),
            'total_files_monitored': self.total_files_monitored,
            'total_debt_events': self.total_debt_events,
            'configuration': {
                'debt_change_threshold': self.config.debt_change_threshold,
                'batch_analysis_delay': self.config.batch_analysis_delay,
                'max_concurrent_analysis': self.config.max_concurrent_analysis,
                'notifications_enabled': self.config.notification_enabled,
                'dashboard_updates_enabled': self.config.dashboard_updates_enabled,
                'alert_critical_debt': self.config.alert_critical_debt,
                'historical_tracking': self.config.historical_tracking
            },
            'projects': {}
        }
        
        # Add per-project status
        for project_id, project in self.monitored_projects.items():
            file_monitor = self.active_monitors.get(project_id)
            monitor_stats = await file_monitor.get_monitoring_stats() if file_monitor else {}
            
            incremental_status = {}
            if self.incremental_analyzer:
                incremental_status = await self.incremental_analyzer.get_incremental_debt_status(project_id)
            
            status['projects'][project_id] = {
                'name': project.name,
                'root_path': project.root_path,
                'files_count': len(project.file_entries),
                'monitoring_active': file_monitor is not None,
                'file_monitor_stats': monitor_stats,
                'incremental_debt_status': incremental_status,
                'recent_debt_trend': self.debt_trends.get(project_id, [])[-10:]  # Last 10 values
            }
        
        return status
    
    async def force_debt_analysis(self, project_id: str, file_paths: Optional[List[str]] = None) -> Dict[str, any]:
        """Force debt analysis for project or specific files."""
        if project_id not in self.monitored_projects:
            raise ValueError(f"Project {project_id} is not being monitored")
        
        if not self.incremental_analyzer:
            raise RuntimeError("Incremental analyzer not initialized")
        
        project = self.monitored_projects[project_id]
        
        # Determine files to analyze
        target_files = file_paths or [fe.file_path for fe in project.file_entries if not fe.is_binary]
        
        logger.info(
            "Forcing debt analysis",
            project_id=project_id,
            files_count=len(target_files)
        )
        
        try:
            # Force analysis
            result = await self.incremental_analyzer.force_analysis(project_id, target_files)
            
            # Update trend data
            if project_id not in self.debt_trends:
                self.debt_trends[project_id] = []
            self.debt_trends[project_id].append(result.total_debt_score)
            
            # Keep only recent trend data (last 100 points)
            if len(self.debt_trends[project_id]) > 100:
                self.debt_trends[project_id] = self.debt_trends[project_id][-100:]
            
            # Publish update event
            if self.config.dashboard_updates_enabled:
                await self._publish_debt_analysis_result(project_id, result)
            
            return {
                'project_id': project_id,
                'files_analyzed': len(target_files),
                'total_debt_score': result.total_debt_score,
                'debt_items_found': len(result.debt_items),
                'analysis_duration': result.analysis_duration,
                'category_breakdown': result.category_scores
            }
            
        except Exception as e:
            logger.error(
                "Error in forced debt analysis",
                project_id=project_id,
                error=str(e)
            )
            raise
    
    # Private event handlers
    
    async def _handle_file_change(self, event: FileChangeEvent) -> None:
        """Handle file change events for debt analysis."""
        try:
            # Delegate to incremental analyzer
            if self.incremental_analyzer:
                await self.incremental_analyzer.handle_file_change(event)
            
        except Exception as e:
            logger.error(
                "Error handling file change for debt analysis",
                file_path=event.file_path,
                error=str(e)
            )
    
    async def _handle_debt_change(self, event: DebtChangeEvent) -> None:
        """Handle debt change events."""
        try:
            self.total_debt_events += 1
            
            logger.debug(
                "Debt change detected",
                project_id=event.project_id,
                file_path=event.file_path,
                debt_delta=event.debt_delta,
                priority=event.remediation_priority
            )
            
            # Update trend data
            if event.project_id not in self.debt_trends:
                self.debt_trends[event.project_id] = []
            self.debt_trends[event.project_id].append(event.current_debt_score)
            
            # Publish notifications
            if self.config.notification_enabled:
                await self._publish_debt_change_notification(event)
            
            # Send critical alerts
            if self.config.alert_critical_debt and event.remediation_priority == "immediate":
                await self._send_critical_debt_alert(event)
            
            # Update dashboard
            if self.config.dashboard_updates_enabled:
                await self._update_dashboard(event)
            
            # Store historical data
            if self.config.historical_tracking:
                await self._store_historical_debt_data(event)
            
        except Exception as e:
            logger.error(
                "Error handling debt change event",
                project_id=event.project_id,
                file_path=event.file_path,
                error=str(e)
            )
    
    async def _publish_debt_change_notification(self, event: DebtChangeEvent) -> None:
        """Publish debt change notification via WebSocket."""
        try:
            notification_data = {
                'type': 'debt_change',
                'project_id': event.project_id,
                'file_path': event.file_path,
                'debt_delta': event.debt_delta,
                'current_score': event.current_debt_score,
                'priority': event.remediation_priority,
                'affected_patterns': event.affected_patterns,
                'timestamp': event.timestamp.isoformat()
            }
            
            from uuid import UUID
            # Create proper notification data
            update_data = ProjectIndexUpdateData(
                project_id=UUID(event.project_id),
                project_name="Unknown",
                files_analyzed=1,
                files_updated=1,
                dependencies_updated=0,
                analysis_duration_seconds=0.0,
                status="completed",
                statistics=notification_data
            )
            try:
                project_uuid = UUID(event.project_id) if len(event.project_id) == 36 else UUID(int=hash(event.project_id) & ((1<<128)-1))
            except (ValueError, TypeError):
                import uuid
                project_uuid = uuid.uuid4()  # Fallback for invalid UUIDs
            await publish_project_updated(project_uuid, update_data)
            
        except Exception as e:
            logger.error("Failed to publish debt change notification", error=str(e))
    
    async def _send_critical_debt_alert(self, event: DebtChangeEvent) -> None:
        """Send critical debt alert."""
        try:
            alert_data = {
                'type': 'critical_debt_alert',
                'severity': 'critical',
                'project_id': event.project_id,
                'file_path': event.file_path,
                'message': f"Critical debt increase in {event.file_path}: +{event.debt_delta:.2f}",
                'remediation_priority': event.remediation_priority,
                'timestamp': event.timestamp.isoformat()
            }
            
            # This could integrate with external alerting systems
            # For now, we'll use the WebSocket event system
            from uuid import UUID
            # Create alert as project update with critical metadata
            update_data = ProjectIndexUpdateData(
                project_id=UUID(event.project_id),
                project_name="Unknown",
                files_analyzed=1,
                files_updated=1,
                dependencies_updated=0,
                analysis_duration_seconds=0.0,
                status="failed" if alert_data['severity'] == 'critical' else "completed",
                statistics=alert_data,
                error_count=1 if alert_data['severity'] == 'critical' else 0
            )
            try:
                project_uuid = UUID(event.project_id) if len(event.project_id) == 36 else UUID(int=hash(event.project_id) & ((1<<128)-1))
            except (ValueError, TypeError):
                import uuid
                project_uuid = uuid.uuid4()  # Fallback for invalid UUIDs
            await publish_project_updated(project_uuid, update_data)
            
            logger.warning(
                "Critical debt alert sent",
                project_id=event.project_id,
                file_path=event.file_path,
                debt_delta=event.debt_delta
            )
            
        except Exception as e:
            logger.error("Failed to send critical debt alert", error=str(e))
    
    async def _update_dashboard(self, event: DebtChangeEvent) -> None:
        """Update dashboard with debt change information."""
        try:
            dashboard_data = {
                'type': 'debt_dashboard_update',
                'project_id': event.project_id,
                'file_path': event.file_path,
                'current_debt_score': event.current_debt_score,
                'debt_trend': self.debt_trends.get(event.project_id, [])[-10:],
                'priority_distribution': self._calculate_priority_distribution(event.project_id),
                'timestamp': event.timestamp.isoformat()
            }
            
            from uuid import UUID
            # Create dashboard update
            update_data = ProjectIndexUpdateData(
                project_id=UUID(event.project_id),
                project_name="Unknown",
                files_analyzed=1,
                files_updated=1,
                dependencies_updated=0,
                analysis_duration_seconds=0.0,
                status="completed",
                statistics=dashboard_data
            )
            try:
                project_uuid = UUID(event.project_id) if len(event.project_id) == 36 else UUID(int=hash(event.project_id) & ((1<<128)-1))
            except (ValueError, TypeError):
                import uuid
                project_uuid = uuid.uuid4()  # Fallback for invalid UUIDs
            await publish_project_updated(project_uuid, update_data)
            
        except Exception as e:
            logger.error("Failed to update dashboard", error=str(e))
    
    async def _store_historical_debt_data(self, event: DebtChangeEvent) -> None:
        """Store historical debt data for trend analysis."""
        try:
            # This would typically store in database
            # For now, we'll log the event for historical tracking
            logger.info(
                "Historical debt data point",
                project_id=event.project_id,
                file_path=event.file_path,
                debt_score=event.current_debt_score,
                debt_delta=event.debt_delta,
                timestamp=event.timestamp.isoformat(),
                structured=True  # Mark for historical data collection
            )
            
        except Exception as e:
            logger.error("Failed to store historical debt data", error=str(e))
    
    async def _publish_debt_analysis_result(self, project_id: str, result) -> None:
        """Publish comprehensive debt analysis result."""
        try:
            event_data = {
                'type': 'debt_analysis_complete',
                'project_id': project_id,
                'total_debt_score': result.total_debt_score,
                'debt_items_count': len(result.debt_items),
                'category_scores': result.category_scores,
                'files_analyzed': result.file_count,
                'analysis_duration': result.analysis_duration,
                'recommendations': result.recommendations,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            from uuid import UUID
            # Create comprehensive analysis result update
            update_data = ProjectIndexUpdateData(
                project_id=UUID(project_id),
                project_name="Unknown",
                files_analyzed=result.file_count,
                files_updated=result.file_count,
                dependencies_updated=0,
                analysis_duration_seconds=result.analysis_duration,
                status="completed",
                statistics=event_data
            )
            await publish_project_updated(UUID(project_id), update_data)
            
        except Exception as e:
            logger.error("Failed to publish debt analysis result", error=str(e))
    
    def _calculate_priority_distribution(self, project_id: str) -> Dict[str, int]:
        """Calculate distribution of debt priorities for a project."""
        # This would typically query recent debt events
        # For now, return placeholder data
        return {
            'immediate': 0,
            'high': 1,
            'medium': 3,
            'low': 5
        }