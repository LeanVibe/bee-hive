"""
Project Index WebSocket Event Publisher and Subscription Management

This module provides real-time WebSocket event publishing for project index operations,
including analysis progress tracking, dependency changes, and context optimization results.

Supports the following event types:
- project_index_updated: Notify when project index analysis is complete
- analysis_progress: Real-time progress updates during analysis operations
- dependency_changed: Notify when dependency relationships change
- context_optimized: Notify when AI context optimization is complete
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union, Callable
from uuid import UUID
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
from pydantic import BaseModel, Field, ConfigDict

from ..core.redis import get_redis_client, RedisClient
from ..core.config import settings

logger = structlog.get_logger()


class ProjectIndexEventType(Enum):
    """Project Index WebSocket event types."""
    PROJECT_INDEX_UPDATED = "project_index_updated"
    ANALYSIS_PROGRESS = "analysis_progress"
    DEPENDENCY_CHANGED = "dependency_changed"
    CONTEXT_OPTIMIZED = "context_optimized"


@dataclass
class ProjectIndexUpdateData:
    """Data structure for project index update events."""
    project_id: UUID
    project_name: str
    files_analyzed: int
    files_updated: int
    dependencies_updated: int
    analysis_duration_seconds: float
    status: str  # completed, failed, partial
    statistics: Dict[str, Any]
    error_count: int = 0
    warnings: Optional[List[str]] = None


@dataclass
class AnalysisProgressData:
    """Data structure for analysis progress events."""
    session_id: UUID
    project_id: UUID
    analysis_type: str  # full, incremental, context_optimization
    progress_percentage: int  # 0-100
    files_processed: int
    total_files: int
    current_file: Optional[str]
    estimated_completion: Optional[datetime]
    processing_rate: float  # files per second
    performance_metrics: Dict[str, Any]
    errors_encountered: int = 0
    last_error: Optional[str] = None


@dataclass
class DependencyChangeData:
    """Data structure for dependency change events."""
    project_id: UUID
    file_path: str
    change_type: str  # added, removed, modified, file_created, file_deleted
    dependency_details: Dict[str, Any]
    impact_analysis: Dict[str, Any]
    file_metadata: Dict[str, Any]


@dataclass
class ContextOptimizedData:
    """Data structure for context optimization events."""
    context_id: UUID
    project_id: UUID
    task_description: str
    task_type: str
    optimization_results: Dict[str, Any]
    recommendations: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class ProjectIndexWebSocketEvent(BaseModel):
    """Base WebSocket event schema for project index events."""
    model_config = ConfigDict(use_enum_values=True)
    
    type: ProjectIndexEventType
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: UUID = Field(default_factory=uuid.uuid4)


class EventSubscription:
    """Manages event subscriptions for WebSocket connections."""
    
    def __init__(self):
        # Map of connection_id -> set of event types
        self.event_subscriptions: Dict[str, Set[ProjectIndexEventType]] = {}
        
        # Map of project_id -> set of connection_ids
        self.project_subscriptions: Dict[str, Set[str]] = {}
        
        # Map of session_id -> set of connection_ids  
        self.session_subscriptions: Dict[str, Set[str]] = {}
        
        # Map of connection_id -> subscription metadata
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
    
    def subscribe_to_events(self, connection_id: str, event_types: List[ProjectIndexEventType]) -> None:
        """Subscribe connection to specific event types."""
        if connection_id not in self.event_subscriptions:
            self.event_subscriptions[connection_id] = set()
        
        self.event_subscriptions[connection_id].update(event_types)
        logger.debug("Subscribed to events", 
                    connection_id=connection_id, 
                    event_types=[e.value for e in event_types])
    
    def subscribe_to_project(self, connection_id: str, project_id: str) -> None:
        """Subscribe connection to project-specific events."""
        if project_id not in self.project_subscriptions:
            self.project_subscriptions[project_id] = set()
        
        self.project_subscriptions[project_id].add(connection_id)
        logger.debug("Subscribed to project", 
                    connection_id=connection_id, 
                    project_id=project_id)
    
    def subscribe_to_session(self, connection_id: str, session_id: str) -> None:
        """Subscribe connection to analysis session events."""
        if session_id not in self.session_subscriptions:
            self.session_subscriptions[session_id] = set()
        
        self.session_subscriptions[session_id].add(connection_id)
        logger.debug("Subscribed to session", 
                    connection_id=connection_id, 
                    session_id=session_id)
    
    def unsubscribe_connection(self, connection_id: str) -> None:
        """Remove connection from all subscriptions."""
        # Remove from event subscriptions
        self.event_subscriptions.pop(connection_id, None)
        
        # Remove from project subscriptions
        for project_id, connections in self.project_subscriptions.items():
            connections.discard(connection_id)
        
        # Remove from session subscriptions
        for session_id, connections in self.session_subscriptions.items():
            connections.discard(connection_id)
        
        # Remove metadata
        self.connection_metadata.pop(connection_id, None)
        
        logger.debug("Unsubscribed connection", connection_id=connection_id)
    
    def get_subscribers_for_event(
        self, 
        event_type: ProjectIndexEventType,
        project_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Set[str]:
        """Get connection IDs that should receive this event."""
        subscribers = set()
        
        # Find connections subscribed to this event type
        for connection_id, subscribed_events in self.event_subscriptions.items():
            if event_type in subscribed_events:
                subscribers.add(connection_id)
        
        # Filter by project if specified
        if project_id and project_id in self.project_subscriptions:
            project_subscribers = self.project_subscriptions[project_id]
            subscribers = subscribers.intersection(project_subscribers)
        
        # Filter by session if specified
        if session_id and session_id in self.session_subscriptions:
            session_subscribers = self.session_subscriptions[session_id]
            subscribers = subscribers.intersection(session_subscribers)
        
        return subscribers


class EventHistory:
    """Manages event history for client reconnection and replay."""
    
    def __init__(self, max_events_per_project: int = 100, ttl_hours: int = 24):
        self.max_events_per_project = max_events_per_project
        self.ttl_hours = ttl_hours
        
        # Map of project_id -> list of events
        self.project_events: Dict[str, List[Dict[str, Any]]] = {}
        
        # Event timestamps for TTL cleanup
        self.event_timestamps: Dict[str, datetime] = {}
    
    def add_event(self, event: ProjectIndexWebSocketEvent, project_id: Optional[str] = None) -> None:
        """Add event to history."""
        if not project_id:
            # Extract project_id from event data if possible
            project_id = event.data.get('project_id')
            if not project_id:
                return
        
        project_id = str(project_id)
        
        if project_id not in self.project_events:
            self.project_events[project_id] = []
        
        # Add event with timestamp
        event_dict = {
            "type": event.type.value,
            "data": event.data,
            "timestamp": event.timestamp.isoformat(),
            "correlation_id": str(event.correlation_id)
        }
        
        self.project_events[project_id].append(event_dict)
        self.event_timestamps[f"{project_id}:{event.correlation_id}"] = event.timestamp
        
        # Maintain max events limit
        if len(self.project_events[project_id]) > self.max_events_per_project:
            removed_event = self.project_events[project_id].pop(0)
            # Remove timestamp entry
            removed_key = f"{project_id}:{removed_event['correlation_id']}"
            self.event_timestamps.pop(removed_key, None)
    
    def get_recent_events(
        self, 
        project_id: str, 
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get recent events for a project."""
        project_id = str(project_id)
        
        if project_id not in self.project_events:
            return []
        
        events = self.project_events[project_id]
        
        # Filter by timestamp if provided
        if since:
            events = [
                e for e in events 
                if datetime.fromisoformat(e['timestamp']) > since
            ]
        
        # Apply limit
        if limit:
            events = events[-limit:]
        
        return events
    
    def cleanup_expired_events(self) -> int:
        """Remove events older than TTL."""
        cutoff = datetime.utcnow() - timedelta(hours=self.ttl_hours)
        removed_count = 0
        
        for key, timestamp in list(self.event_timestamps.items()):
            if timestamp < cutoff:
                project_id, correlation_id = key.split(':', 1)
                
                # Remove from project events
                if project_id in self.project_events:
                    self.project_events[project_id] = [
                        e for e in self.project_events[project_id]
                        if e['correlation_id'] != correlation_id
                    ]
                
                # Remove timestamp entry
                del self.event_timestamps[key]
                removed_count += 1
        
        return removed_count


class ProjectIndexEventPublisher:
    """Main event publisher for project index WebSocket events."""
    
    def __init__(self, redis_client: Optional[RedisClient] = None):
        self.redis_client = redis_client or get_redis_client()
        self.subscriptions = EventSubscription()
        self.event_history = EventHistory()
        
        # Performance tracking
        self.metrics = {
            "events_published": 0,
            "events_delivered": 0,
            "events_failed": 0,
            "subscribers_notified": 0,
            "events_batched": 0
        }
        
        # Event batching for performance
        self.batch_queue: List[ProjectIndexWebSocketEvent] = []
        self.batch_task: Optional[asyncio.Task] = None
        self.batch_interval = 0.1  # 100ms batching interval
        
        # Rate limiting
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.rate_limit_window = 60  # 1 minute window
        self.rate_limit_max_events = 100  # Max events per connection per minute
    
    async def publish_project_updated(
        self, 
        project_id: UUID, 
        data: ProjectIndexUpdateData
    ) -> int:
        """Publish project index updated event."""
        event = ProjectIndexWebSocketEvent(
            type=ProjectIndexEventType.PROJECT_INDEX_UPDATED,
            data={
                "project_id": str(project_id),
                "project_name": data.project_name,
                "files_analyzed": data.files_analyzed,
                "files_updated": data.files_updated,
                "dependencies_updated": data.dependencies_updated,
                "analysis_duration_seconds": data.analysis_duration_seconds,
                "status": data.status,
                "statistics": data.statistics,
                "error_count": data.error_count,
                "warnings": data.warnings or []
            }
        )
        
        return await self._publish_event(event, str(project_id))
    
    async def publish_analysis_progress(
        self, 
        session_id: UUID, 
        data: AnalysisProgressData
    ) -> int:
        """Publish analysis progress event."""
        event = ProjectIndexWebSocketEvent(
            type=ProjectIndexEventType.ANALYSIS_PROGRESS,
            data={
                "session_id": str(session_id),
                "project_id": str(data.project_id),
                "analysis_type": data.analysis_type,
                "progress_percentage": data.progress_percentage,
                "files_processed": data.files_processed,
                "total_files": data.total_files,
                "current_file": data.current_file,
                "estimated_completion": data.estimated_completion.isoformat() if data.estimated_completion else None,
                "processing_rate": data.processing_rate,
                "performance_metrics": data.performance_metrics,
                "errors_encountered": data.errors_encountered,
                "last_error": data.last_error
            }
        )
        
        return await self._publish_event(event, str(data.project_id), str(session_id))
    
    async def publish_dependency_changed(
        self, 
        project_id: UUID, 
        data: DependencyChangeData
    ) -> int:
        """Publish dependency changed event."""
        event = ProjectIndexWebSocketEvent(
            type=ProjectIndexEventType.DEPENDENCY_CHANGED,
            data={
                "project_id": str(project_id),
                "file_path": data.file_path,
                "change_type": data.change_type,
                "dependency_details": data.dependency_details,
                "impact_analysis": data.impact_analysis,
                "file_metadata": data.file_metadata
            }
        )
        
        return await self._publish_event(event, str(project_id))
    
    async def publish_context_optimized(
        self, 
        context_id: UUID, 
        data: ContextOptimizedData
    ) -> int:
        """Publish context optimized event."""
        event = ProjectIndexWebSocketEvent(
            type=ProjectIndexEventType.CONTEXT_OPTIMIZED,
            data={
                "context_id": str(context_id),
                "project_id": str(data.project_id),
                "task_description": data.task_description,
                "task_type": data.task_type,
                "optimization_results": data.optimization_results,
                "recommendations": data.recommendations,
                "performance_metrics": data.performance_metrics
            }
        )
        
        return await self._publish_event(event, str(data.project_id))
    
    async def subscribe_to_project(self, client_id: str, project_id: UUID) -> None:
        """Subscribe client to project events."""
        self.subscriptions.subscribe_to_project(client_id, str(project_id))
    
    async def subscribe_to_events(
        self, 
        client_id: str, 
        event_types: List[str]
    ) -> None:
        """Subscribe client to specific event types."""
        parsed_types = []
        for event_type in event_types:
            try:
                parsed_types.append(ProjectIndexEventType(event_type))
            except ValueError:
                logger.warning("Invalid event type", event_type=event_type)
        
        if parsed_types:
            self.subscriptions.subscribe_to_events(client_id, parsed_types)
    
    async def unsubscribe_client(self, client_id: str) -> None:
        """Unsubscribe client from all events."""
        self.subscriptions.unsubscribe_connection(client_id)
    
    async def get_recent_events(
        self, 
        project_id: UUID, 
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get recent events for client reconnection."""
        return self.event_history.get_recent_events(str(project_id), since, limit)
    
    async def _publish_event(
        self, 
        event: ProjectIndexWebSocketEvent, 
        project_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> int:
        """Internal method to publish event to subscribers."""
        # Add to history
        self.event_history.add_event(event, project_id)
        
        # Get subscribers
        subscribers = self.subscriptions.get_subscribers_for_event(
            event.type, project_id, session_id
        )
        
        if not subscribers:
            logger.debug("No subscribers for event", 
                        event_type=event.type.value, 
                        project_id=project_id)
            return 0
        
        # Filter subscribers by rate limit
        valid_subscribers = []
        for subscriber in subscribers:
            if await self._check_rate_limit(subscriber):
                valid_subscribers.append(subscriber)
        
        if not valid_subscribers:
            logger.debug("All subscribers rate limited", event_type=event.type.value)
            return 0
        
        # Publish to WebSocket connections via Redis
        await self._publish_to_redis(event, valid_subscribers)
        
        # Update metrics
        self.metrics["events_published"] += 1
        self.metrics["subscribers_notified"] += len(valid_subscribers)
        
        logger.debug("Published event", 
                    event_type=event.type.value,
                    subscribers=len(valid_subscribers))
        
        return len(valid_subscribers)
    
    async def _check_rate_limit(self, connection_id: str) -> bool:
        """Check if connection is within rate limits."""
        now = datetime.utcnow()
        
        if connection_id not in self.rate_limits:
            self.rate_limits[connection_id] = []
        
        # Clean old timestamps
        cutoff = now - timedelta(seconds=self.rate_limit_window)
        self.rate_limits[connection_id] = [
            ts for ts in self.rate_limits[connection_id] if ts > cutoff
        ]
        
        # Check limit
        if len(self.rate_limits[connection_id]) >= self.rate_limit_max_events:
            return False
        
        # Add current timestamp
        self.rate_limits[connection_id].append(now)
        return True
    
    async def _publish_to_redis(
        self, 
        event: ProjectIndexWebSocketEvent, 
        subscribers: List[str]
    ) -> None:
        """Publish event to Redis for WebSocket delivery."""
        message = {
            "event": {
                "type": event.type.value,
                "data": event.data,
                "timestamp": event.timestamp.isoformat(),
                "correlation_id": str(event.correlation_id)
            },
            "subscribers": subscribers
        }
        
        try:
            # Publish to Redis channel for WebSocket delivery
            await self.redis_client.publish(
                "project_index:websocket_events",
                json.dumps(message)
            )
            
            self.metrics["events_delivered"] += 1
            
        except Exception as e:
            logger.error("Failed to publish event to Redis", error=str(e))
            self.metrics["events_failed"] += 1
    
    async def start_batch_processing(self) -> None:
        """Start background task for event batching."""
        if self.batch_task is None or self.batch_task.done():
            self.batch_task = asyncio.create_task(self._batch_processor())
    
    async def stop_batch_processing(self) -> None:
        """Stop background batch processing."""
        if self.batch_task and not self.batch_task.done():
            self.batch_task.cancel()
            try:
                await self.batch_task
            except asyncio.CancelledError:
                pass
    
    async def _batch_processor(self) -> None:
        """Background task to process event batches."""
        while True:
            try:
                await asyncio.sleep(self.batch_interval)
                
                if self.batch_queue:
                    events_to_process = self.batch_queue.copy()
                    self.batch_queue.clear()
                    
                    # Process batched events
                    for event in events_to_process:
                        await self._publish_event(event)
                    
                    self.metrics["events_batched"] += len(events_to_process)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in batch processor", error=str(e))
    
    async def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up expired events and rate limit data."""
        # Clean event history
        expired_events = self.event_history.cleanup_expired_events()
        
        # Clean rate limit data
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.rate_limit_window * 2)
        
        expired_connections = 0
        for connection_id in list(self.rate_limits.keys()):
            self.rate_limits[connection_id] = [
                ts for ts in self.rate_limits[connection_id] if ts > cutoff
            ]
            if not self.rate_limits[connection_id]:
                del self.rate_limits[connection_id]
                expired_connections += 1
        
        return {
            "expired_events": expired_events,
            "expired_connections": expired_connections
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get publisher performance metrics."""
        return {
            **self.metrics,
            "active_subscriptions": len(self.subscriptions.event_subscriptions),
            "project_subscriptions": len(self.subscriptions.project_subscriptions),
            "session_subscriptions": len(self.subscriptions.session_subscriptions),
            "events_in_history": sum(
                len(events) for events in self.event_history.project_events.values()
            ),
            "rate_limited_connections": len(self.rate_limits)
        }


# Global event publisher instance
_event_publisher: Optional[ProjectIndexEventPublisher] = None


def get_event_publisher() -> ProjectIndexEventPublisher:
    """Get or create the global event publisher instance."""
    global _event_publisher
    if _event_publisher is None:
        _event_publisher = ProjectIndexEventPublisher()
    return _event_publisher


# Convenience functions for publishing events
async def publish_project_updated(project_id: UUID, data: ProjectIndexUpdateData) -> int:
    """Convenience function to publish project updated event."""
    publisher = get_event_publisher()
    return await publisher.publish_project_updated(project_id, data)


async def publish_analysis_progress(session_id: UUID, data: AnalysisProgressData) -> int:
    """Convenience function to publish analysis progress event."""
    publisher = get_event_publisher()
    return await publisher.publish_analysis_progress(session_id, data)


async def publish_dependency_changed(project_id: UUID, data: DependencyChangeData) -> int:
    """Convenience function to publish dependency changed event."""
    publisher = get_event_publisher()
    return await publisher.publish_dependency_changed(project_id, data)


async def publish_context_optimized(context_id: UUID, data: ContextOptimizedData) -> int:
    """Convenience function to publish context optimized event."""
    publisher = get_event_publisher()
    return await publisher.publish_context_optimized(context_id, data)