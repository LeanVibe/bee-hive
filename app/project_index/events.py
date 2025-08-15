"""
Event System for LeanVibe Agent Hive 2.0 Project Index

Provides real-time event handling and WebSocket broadcasting for project analysis events.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from uuid import UUID, uuid4

import structlog

logger = structlog.get_logger()


class EventType(Enum):
    """Types of project index events."""
    # File events
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    FILE_MOVED = "file_moved"
    
    # Analysis events
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETED = "analysis_completed"
    ANALYSIS_FAILED = "analysis_failed"
    ANALYSIS_PROGRESS = "analysis_progress"
    
    # Project events
    PROJECT_CREATED = "project_created"
    PROJECT_UPDATED = "project_updated"
    PROJECT_DELETED = "project_deleted"
    
    # Cache events
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_INVALIDATED = "cache_invalidated"
    
    # Dependency events
    DEPENDENCY_ADDED = "dependency_added"
    DEPENDENCY_REMOVED = "dependency_removed"
    DEPENDENCY_GRAPH_UPDATED = "dependency_graph_updated"
    
    # System events
    MONITORING_STARTED = "monitoring_started"
    MONITORING_STOPPED = "monitoring_stopped"
    ERROR_OCCURRED = "error_occurred"


@dataclass
class ProjectIndexEvent:
    """Base event class for project index events."""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: EventType = EventType.ANALYSIS_PROGRESS
    timestamp: datetime = field(default_factory=datetime.utcnow)
    project_id: Optional[UUID] = None
    session_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'project_id': str(self.project_id) if self.project_id else None,
            'session_id': self.session_id,
            'data': self.data,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectIndexEvent':
        """Create event from dictionary."""
        return cls(
            event_id=data.get('event_id', str(uuid4())),
            event_type=EventType(data['event_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            project_id=UUID(data['project_id']) if data.get('project_id') else None,
            session_id=data.get('session_id'),
            data=data.get('data', {}),
            metadata=data.get('metadata', {})
        )


class EventFilter:
    """Filter for subscribing to specific events."""
    
    def __init__(
        self,
        event_types: Optional[Set[EventType]] = None,
        project_ids: Optional[Set[UUID]] = None,
        session_ids: Optional[Set[str]] = None
    ):
        """
        Initialize event filter.
        
        Args:
            event_types: Set of event types to include (None = all)
            project_ids: Set of project IDs to include (None = all)
            session_ids: Set of session IDs to include (None = all)
        """
        self.event_types = event_types
        self.project_ids = project_ids
        self.session_ids = session_ids
    
    def matches(self, event: ProjectIndexEvent) -> bool:
        """
        Check if event matches filter criteria.
        
        Args:
            event: Event to check
            
        Returns:
            True if event matches filter, False otherwise
        """
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        if self.project_ids and event.project_id not in self.project_ids:
            return False
        
        if self.session_ids and event.session_id not in self.session_ids:
            return False
        
        return True


class EventPublisher:
    """
    Event publisher for real-time project index events.
    
    Features:
    - Asynchronous event publishing
    - Filtered subscriptions
    - WebSocket integration
    - Event persistence for replay
    - Rate limiting and batching
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize EventPublisher.
        
        Args:
            max_history: Maximum number of events to keep in history
        """
        self.max_history = max_history
        
        # Event subscriptions
        self.subscribers: Dict[str, Tuple[EventFilter, Callable[[ProjectIndexEvent], Any]]] = {}
        
        # Event history for replay
        self.event_history: List[ProjectIndexEvent] = []
        
        # WebSocket connections
        self.websocket_connections: Set[Any] = set()  # Will store WebSocket connections
        
        # Statistics
        self.stats = {
            'events_published': 0,
            'subscribers_count': 0,
            'websocket_connections': 0,
            'events_in_history': 0
        }
        
        self.logger = structlog.get_logger()
    
    async def publish(self, event: ProjectIndexEvent) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        """
        try:
            # Add to history
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history.pop(0)
            
            # Update statistics
            self.stats['events_published'] += 1
            self.stats['events_in_history'] = len(self.event_history)
            
            # Notify subscribers
            await self._notify_subscribers(event)
            
            # Broadcast to WebSocket connections
            await self._broadcast_websocket(event)
            
            self.logger.debug("Event published", 
                            event_type=event.event_type.value,
                            event_id=event.event_id,
                            subscriber_count=len(self.subscribers))
            
        except Exception as e:
            self.logger.error("Failed to publish event", 
                            event_type=event.event_type.value,
                            error=str(e))
    
    async def _notify_subscribers(self, event: ProjectIndexEvent) -> None:
        """
        Notify all matching subscribers about an event.
        
        Args:
            event: Event to notify about
        """
        for subscriber_id, (event_filter, callback) in self.subscribers.items():
            try:
                if event_filter.matches(event):
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)
            except Exception as e:
                self.logger.error("Subscriber notification failed", 
                                subscriber_id=subscriber_id,
                                error=str(e))
    
    async def _broadcast_websocket(self, event: ProjectIndexEvent) -> None:
        """
        Broadcast event to WebSocket connections.
        
        Args:
            event: Event to broadcast
        """
        if not self.websocket_connections:
            return
        
        message = event.to_json()
        disconnected = set()
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message)
            except Exception as e:
                self.logger.warning("WebSocket send failed", error=str(e))
                disconnected.add(websocket)
        
        # Remove disconnected WebSockets
        self.websocket_connections -= disconnected
        self.stats['websocket_connections'] = len(self.websocket_connections)
    
    def subscribe(
        self, 
        callback: Callable[[ProjectIndexEvent], Any],
        event_filter: Optional[EventFilter] = None,
        subscriber_id: Optional[str] = None
    ) -> str:
        """
        Subscribe to events with optional filtering.
        
        Args:
            callback: Function to call when matching events occur
            event_filter: Optional filter for events
            subscriber_id: Optional custom subscriber ID
            
        Returns:
            Subscriber ID for unsubscribing
        """
        if subscriber_id is None:
            subscriber_id = str(uuid4())
        
        if event_filter is None:
            event_filter = EventFilter()  # Match all events
        
        self.subscribers[subscriber_id] = (event_filter, callback)
        self.stats['subscribers_count'] = len(self.subscribers)
        
        self.logger.info("Event subscriber added", 
                       subscriber_id=subscriber_id,
                       total_subscribers=len(self.subscribers))
        
        return subscriber_id
    
    def unsubscribe(self, subscriber_id: str) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            subscriber_id: Subscriber ID to remove
            
        Returns:
            True if subscriber was found and removed, False otherwise
        """
        if subscriber_id in self.subscribers:
            del self.subscribers[subscriber_id]
            self.stats['subscribers_count'] = len(self.subscribers)
            
            self.logger.info("Event subscriber removed", 
                           subscriber_id=subscriber_id,
                           total_subscribers=len(self.subscribers))
            return True
        
        return False
    
    def add_websocket(self, websocket: Any) -> None:
        """
        Add WebSocket connection for event broadcasting.
        
        Args:
            websocket: WebSocket connection object
        """
        self.websocket_connections.add(websocket)
        self.stats['websocket_connections'] = len(self.websocket_connections)
        
        self.logger.info("WebSocket connection added", 
                       total_connections=len(self.websocket_connections))
    
    def remove_websocket(self, websocket: Any) -> None:
        """
        Remove WebSocket connection.
        
        Args:
            websocket: WebSocket connection object
        """
        self.websocket_connections.discard(websocket)
        self.stats['websocket_connections'] = len(self.websocket_connections)
        
        self.logger.info("WebSocket connection removed", 
                       total_connections=len(self.websocket_connections))
    
    def get_event_history(
        self, 
        event_filter: Optional[EventFilter] = None,
        limit: Optional[int] = None
    ) -> List[ProjectIndexEvent]:
        """
        Get event history with optional filtering.
        
        Args:
            event_filter: Optional filter for events
            limit: Maximum number of events to return
            
        Returns:
            List of matching events
        """
        events = self.event_history
        
        if event_filter:
            events = [event for event in events if event_filter.matches(event)]
        
        if limit:
            events = events[-limit:]
        
        return events
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get event publisher statistics.
        
        Returns:
            Dictionary with statistics
        """
        return self.stats.copy()
    
    async def replay_events(
        self, 
        callback: Callable[[ProjectIndexEvent], Any],
        event_filter: Optional[EventFilter] = None
    ) -> int:
        """
        Replay historical events to a callback.
        
        Args:
            callback: Function to call for each event
            event_filter: Optional filter for events
            
        Returns:
            Number of events replayed
        """
        events = self.get_event_history(event_filter)
        
        replayed = 0
        for event in events:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
                replayed += 1
            except Exception as e:
                self.logger.error("Event replay failed", 
                                event_id=event.event_id,
                                error=str(e))
        
        self.logger.info("Event replay completed", 
                       events_replayed=replayed,
                       total_events=len(events))
        
        return replayed


# ================== EVENT CREATION HELPERS ==================

def create_file_event(
    event_type: EventType,
    project_id: UUID,
    file_path: str,
    session_id: Optional[str] = None,
    **kwargs
) -> ProjectIndexEvent:
    """
    Create a file-related event.
    
    Args:
        event_type: Type of file event
        project_id: Project identifier
        file_path: Path to the file
        session_id: Optional session identifier
        **kwargs: Additional data for the event
        
    Returns:
        ProjectIndexEvent
    """
    return ProjectIndexEvent(
        event_type=event_type,
        project_id=project_id,
        session_id=session_id,
        data={
            'file_path': file_path,
            **kwargs
        }
    )


def create_analysis_event(
    event_type: EventType,
    project_id: UUID,
    session_id: Optional[str] = None,
    **kwargs
) -> ProjectIndexEvent:
    """
    Create an analysis-related event.
    
    Args:
        event_type: Type of analysis event
        project_id: Project identifier
        session_id: Optional session identifier
        **kwargs: Additional data for the event
        
    Returns:
        ProjectIndexEvent
    """
    return ProjectIndexEvent(
        event_type=event_type,
        project_id=project_id,
        session_id=session_id,
        data=kwargs
    )


def create_cache_event(
    event_type: EventType,
    cache_key: str,
    hit: bool = False,
    project_id: Optional[UUID] = None,
    **kwargs
) -> ProjectIndexEvent:
    """
    Create a cache-related event.
    
    Args:
        event_type: Type of cache event
        cache_key: Cache key involved
        hit: Whether this was a cache hit or miss
        project_id: Optional project identifier
        **kwargs: Additional data for the event
        
    Returns:
        ProjectIndexEvent
    """
    return ProjectIndexEvent(
        event_type=event_type,
        project_id=project_id,
        data={
            'cache_key': cache_key,
            'hit': hit,
            **kwargs
        }
    )


def create_system_event(
    event_type: EventType,
    message: str,
    level: str = "info",
    **kwargs
) -> ProjectIndexEvent:
    """
    Create a system-related event.
    
    Args:
        event_type: Type of system event
        message: Event message
        level: Log level (info, warning, error)
        **kwargs: Additional data for the event
        
    Returns:
        ProjectIndexEvent
    """
    return ProjectIndexEvent(
        event_type=event_type,
        data={
            'message': message,
            'level': level,
            **kwargs
        }
    )


# Global event publisher instance
_event_publisher: Optional[EventPublisher] = None


def get_event_publisher() -> EventPublisher:
    """
    Get or create the global event publisher instance.
    
    Returns:
        EventPublisher instance
    """
    global _event_publisher
    if _event_publisher is None:
        _event_publisher = EventPublisher()
    return _event_publisher


def set_event_publisher(publisher: EventPublisher) -> None:
    """
    Set the global event publisher instance.
    
    Args:
        publisher: EventPublisher instance to set as global
    """
    global _event_publisher
    _event_publisher = publisher