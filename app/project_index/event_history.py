"""
Event History and Replay System for Project Index WebSocket Events

This module provides comprehensive event history management, client reconnection 
support, and event replay capabilities for project index WebSocket communications.
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from uuid import UUID
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Column, String, DateTime, JSON, Text, Integer, Boolean
from sqlalchemy.sql import func, select, delete, and_

from ..core.database import Base, get_session
from ..core.redis import get_redis_client, RedisClient
from ..core.database_types import DatabaseAgnosticUUID
from .websocket_events import ProjectIndexWebSocketEvent, ProjectIndexEventType

logger = structlog.get_logger()


class EventPersistenceLevel(Enum):
    """Event persistence levels for different event types."""
    MEMORY_ONLY = "memory_only"        # Redis only, short TTL
    SHORT_TERM = "short_term"          # Redis + DB, 24 hours
    MEDIUM_TERM = "medium_term"        # Redis + DB, 7 days
    LONG_TERM = "long_term"           # Redis + DB, 30 days
    PERMANENT = "permanent"            # DB only, no TTL


class EventStorageModel(Base):
    """Database model for persistent event storage."""
    __tablename__ = "project_index_event_history"
    
    id = Column(DatabaseAgnosticUUID(), primary_key=True)
    project_id = Column(DatabaseAgnosticUUID(), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    event_data = Column(JSON, nullable=False)
    correlation_id = Column(String(36), nullable=False, index=True)
    persistence_level = Column(String(20), nullable=False, index=True)
    checksum = Column(String(64), nullable=False)
    client_delivered = Column(JSON, nullable=True, default=list)  # List of client IDs that received this event
    replay_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)


@dataclass
class EventHistoryEntry:
    """Event history entry with metadata."""
    event_id: str
    project_id: str
    event_type: str
    event_data: Dict[str, Any]
    correlation_id: str
    timestamp: datetime
    persistence_level: EventPersistenceLevel
    checksum: str
    delivered_to: Set[str] = None
    replay_count: int = 0
    
    def __post_init__(self):
        if self.delivered_to is None:
            self.delivered_to = set()


@dataclass
class ReplayRequest:
    """Client request for event replay."""
    client_id: str
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    since_timestamp: Optional[datetime] = None
    event_types: Optional[List[str]] = None
    max_events: int = 100
    include_delivered: bool = False


class EventHistoryManager:
    """Manages event history storage, retrieval, and replay."""
    
    def __init__(
        self, 
        redis_client: Optional[RedisClient] = None,
        db_session: Optional[AsyncSession] = None
    ):
        self.redis_client = redis_client or get_redis_client()
        self.db_session = db_session
        
        # Configuration
        self.memory_ttl_hours = 1
        self.short_term_ttl_hours = 24
        self.medium_term_ttl_days = 7
        self.long_term_ttl_days = 30
        
        # Memory storage for high-frequency access
        self.memory_events: Dict[str, Dict[str, EventHistoryEntry]] = {}  # project_id -> event_id -> entry
        
        # Performance tracking
        self.metrics = {
            "events_stored": 0,
            "events_retrieved": 0,
            "events_replayed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "database_writes": 0,
            "database_reads": 0,
            "cleanup_operations": 0
        }
        
        # Event type persistence mapping
        self.persistence_mapping = {
            ProjectIndexEventType.PROJECT_INDEX_UPDATED: EventPersistenceLevel.LONG_TERM,
            ProjectIndexEventType.ANALYSIS_PROGRESS: EventPersistenceLevel.SHORT_TERM,
            ProjectIndexEventType.DEPENDENCY_CHANGED: EventPersistenceLevel.MEDIUM_TERM,
            ProjectIndexEventType.CONTEXT_OPTIMIZED: EventPersistenceLevel.MEDIUM_TERM
        }
    
    async def store_event(
        self, 
        event: ProjectIndexWebSocketEvent,
        project_id: str,
        delivered_to: Optional[Set[str]] = None
    ) -> str:
        """Store event in appropriate storage layer."""
        try:
            # Determine persistence level
            persistence_level = self.persistence_mapping.get(
                event.type, 
                EventPersistenceLevel.SHORT_TERM
            )
            
            # Create history entry
            event_id = str(event.correlation_id)
            checksum = self._calculate_checksum(event.data)
            
            history_entry = EventHistoryEntry(
                event_id=event_id,
                project_id=project_id,
                event_type=event.type.value,
                event_data=event.data,
                correlation_id=str(event.correlation_id),
                timestamp=event.timestamp,
                persistence_level=persistence_level,
                checksum=checksum,
                delivered_to=delivered_to or set()
            )
            
            # Store in memory for fast access
            if project_id not in self.memory_events:
                self.memory_events[project_id] = {}
            self.memory_events[project_id][event_id] = history_entry
            
            # Store in Redis
            await self._store_in_redis(history_entry)
            
            # Store in database if required
            if persistence_level in [EventPersistenceLevel.SHORT_TERM, 
                                   EventPersistenceLevel.MEDIUM_TERM,
                                   EventPersistenceLevel.LONG_TERM, 
                                   EventPersistenceLevel.PERMANENT]:
                await self._store_in_database(history_entry)
            
            self.metrics["events_stored"] += 1
            
            logger.debug("Event stored in history",
                        event_id=event_id,
                        project_id=project_id,
                        event_type=event.type.value,
                        persistence_level=persistence_level.value)
            
            return event_id
            
        except Exception as e:
            logger.error("Failed to store event in history", error=str(e))
            raise
    
    async def get_events(
        self, 
        project_id: str,
        since: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
        limit: int = 100,
        include_delivered_to: Optional[str] = None
    ) -> List[EventHistoryEntry]:
        """Retrieve events from history with filtering."""
        try:
            events = []
            
            # Try memory first
            memory_events = await self._get_from_memory(
                project_id, since, event_types, limit, include_delivered_to
            )
            events.extend(memory_events)
            
            # If not enough events from memory, try Redis
            if len(events) < limit:
                redis_events = await self._get_from_redis(
                    project_id, since, event_types, limit - len(events), include_delivered_to
                )
                events.extend(redis_events)
            
            # If still not enough, try database
            if len(events) < limit:
                db_events = await self._get_from_database(
                    project_id, since, event_types, limit - len(events), include_delivered_to
                )
                events.extend(db_events)
            
            # Sort by timestamp and apply final limit
            events.sort(key=lambda e: e.timestamp, reverse=True)
            events = events[:limit]
            
            self.metrics["events_retrieved"] += len(events)
            
            logger.debug("Retrieved events from history",
                        project_id=project_id,
                        count=len(events),
                        since=since.isoformat() if since else None)
            
            return events
            
        except Exception as e:
            logger.error("Failed to retrieve events from history", error=str(e))
            return []
    
    async def replay_events(self, replay_request: ReplayRequest) -> List[Dict[str, Any]]:
        """Replay events for client reconnection."""
        try:
            project_id = replay_request.project_id
            if not project_id:
                logger.warning("Replay request without project_id")
                return []
            
            # Get events based on request criteria
            events = await self.get_events(
                project_id=project_id,
                since=replay_request.since_timestamp,
                event_types=replay_request.event_types,
                limit=replay_request.max_events,
                include_delivered_to=replay_request.client_id if replay_request.include_delivered else None
            )
            
            # Filter out events already delivered to this client (unless explicitly requested)
            if not replay_request.include_delivered:
                events = [
                    event for event in events 
                    if replay_request.client_id not in event.delivered_to
                ]
            
            # Convert to WebSocket format
            replay_events = []
            for event in events:
                replay_event = {
                    "type": event.event_type,
                    "data": event.event_data,
                    "timestamp": event.timestamp.isoformat(),
                    "correlation_id": event.correlation_id,
                    "replay": True,
                    "replay_timestamp": datetime.utcnow().isoformat()
                }
                replay_events.append(replay_event)
                
                # Update replay count
                event.replay_count += 1
                await self._update_replay_count(event.event_id, event.replay_count)
            
            # Mark events as delivered to this client
            await self._mark_events_delivered(
                [e.event_id for e in events], 
                replay_request.client_id
            )
            
            self.metrics["events_replayed"] += len(replay_events)
            
            logger.info("Replayed events for client",
                       client_id=replay_request.client_id,
                       project_id=project_id,
                       event_count=len(replay_events))
            
            return replay_events
            
        except Exception as e:
            logger.error("Failed to replay events", error=str(e))
            return []
    
    async def mark_event_delivered(self, event_id: str, client_id: str) -> bool:
        """Mark event as delivered to specific client."""
        try:
            await self._mark_events_delivered([event_id], client_id)
            return True
        except Exception as e:
            logger.error("Failed to mark event as delivered", 
                        event_id=event_id, client_id=client_id, error=str(e))
            return False
    
    async def cleanup_expired_events(self) -> Dict[str, int]:
        """Clean up expired events from all storage layers."""
        try:
            cleanup_stats = {
                "memory_cleaned": 0,
                "redis_cleaned": 0,
                "database_cleaned": 0
            }
            
            # Clean memory
            cleanup_stats["memory_cleaned"] = await self._cleanup_memory()
            
            # Clean Redis
            cleanup_stats["redis_cleaned"] = await self._cleanup_redis()
            
            # Clean database
            cleanup_stats["database_cleaned"] = await self._cleanup_database()
            
            self.metrics["cleanup_operations"] += 1
            
            logger.info("Completed event history cleanup", **cleanup_stats)
            
            return cleanup_stats
            
        except Exception as e:
            logger.error("Failed to cleanup expired events", error=str(e))
            return {"error": str(e)}
    
    async def get_project_event_summary(self, project_id: str) -> Dict[str, Any]:
        """Get summary statistics for project events."""
        try:
            # Count events by type from memory and Redis
            event_counts = {}
            total_events = 0
            
            # Memory events
            if project_id in self.memory_events:
                for event in self.memory_events[project_id].values():
                    event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
                    total_events += 1
            
            # Database summary (for complete picture)
            if self.db_session:
                db_summary = await self._get_database_summary(project_id)
                for event_type, count in db_summary.items():
                    event_counts[event_type] = event_counts.get(event_type, 0) + count
                    total_events += count
            
            # Calculate time ranges
            earliest_event = None
            latest_event = None
            
            if project_id in self.memory_events:
                timestamps = [e.timestamp for e in self.memory_events[project_id].values()]
                if timestamps:
                    earliest_event = min(timestamps)
                    latest_event = max(timestamps)
            
            return {
                "project_id": project_id,
                "total_events": total_events,
                "events_by_type": event_counts,
                "earliest_event": earliest_event.isoformat() if earliest_event else None,
                "latest_event": latest_event.isoformat() if latest_event else None,
                "storage_layers": {
                    "memory": len(self.memory_events.get(project_id, {})),
                    "redis": await self._count_redis_events(project_id),
                    "database": await self._count_database_events(project_id)
                }
            }
            
        except Exception as e:
            logger.error("Failed to get project event summary", error=str(e))
            return {"error": str(e)}
    
    # ================ INTERNAL STORAGE METHODS ================
    
    async def _store_in_redis(self, entry: EventHistoryEntry) -> None:
        """Store event entry in Redis."""
        try:
            key = f"project_index:events:{entry.project_id}:{entry.event_id}"
            value = {
                "event_type": entry.event_type,
                "event_data": entry.event_data,
                "correlation_id": entry.correlation_id,
                "timestamp": entry.timestamp.isoformat(),
                "persistence_level": entry.persistence_level.value,
                "checksum": entry.checksum,
                "delivered_to": list(entry.delivered_to),
                "replay_count": entry.replay_count
            }
            
            # Set TTL based on persistence level
            ttl_seconds = self._get_redis_ttl(entry.persistence_level)
            
            await self.redis_client.setex(
                key, 
                ttl_seconds, 
                json.dumps(value)
            )
            
        except Exception as e:
            logger.error("Failed to store event in Redis", error=str(e))
    
    async def _store_in_database(self, entry: EventHistoryEntry) -> None:
        """Store event entry in database."""
        if not self.db_session:
            return
        
        try:
            # Calculate expiration time
            expires_at = None
            if entry.persistence_level != EventPersistenceLevel.PERMANENT:
                expires_at = entry.timestamp + self._get_database_ttl(entry.persistence_level)
            
            db_entry = EventStorageModel(
                id=entry.event_id,
                project_id=entry.project_id,
                event_type=entry.event_type,
                event_data=entry.event_data,
                correlation_id=entry.correlation_id,
                persistence_level=entry.persistence_level.value,
                checksum=entry.checksum,
                client_delivered=list(entry.delivered_to),
                replay_count=entry.replay_count,
                created_at=entry.timestamp,
                expires_at=expires_at
            )
            
            self.db_session.add(db_entry)
            await self.db_session.commit()
            
            self.metrics["database_writes"] += 1
            
        except Exception as e:
            logger.error("Failed to store event in database", error=str(e))
            if self.db_session:
                await self.db_session.rollback()
    
    async def _get_from_memory(
        self, 
        project_id: str, 
        since: Optional[datetime], 
        event_types: Optional[List[str]], 
        limit: int,
        include_delivered_to: Optional[str]
    ) -> List[EventHistoryEntry]:
        """Get events from memory storage."""
        if project_id not in self.memory_events:
            return []
        
        events = list(self.memory_events[project_id].values())
        
        # Apply filters
        if since:
            events = [e for e in events if e.timestamp > since]
        
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        if include_delivered_to:
            events = [e for e in events if include_delivered_to not in e.delivered_to]
        
        # Sort and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        
        self.metrics["cache_hits"] += min(len(events), limit)
        
        return events[:limit]
    
    async def _get_from_redis(
        self, 
        project_id: str, 
        since: Optional[datetime], 
        event_types: Optional[List[str]], 
        limit: int,
        include_delivered_to: Optional[str]
    ) -> List[EventHistoryEntry]:
        """Get events from Redis storage."""
        try:
            pattern = f"project_index:events:{project_id}:*"
            keys = await self.redis_client.keys(pattern)
            
            if not keys:
                self.metrics["cache_misses"] += 1
                return []
            
            # Get all event data
            events = []
            for key in keys:
                try:
                    data = await self.redis_client.get(key)
                    if data:
                        event_data = json.loads(data)
                        
                        entry = EventHistoryEntry(
                            event_id=key.split(':')[-1],
                            project_id=project_id,
                            event_type=event_data["event_type"],
                            event_data=event_data["event_data"],
                            correlation_id=event_data["correlation_id"],
                            timestamp=datetime.fromisoformat(event_data["timestamp"]),
                            persistence_level=EventPersistenceLevel(event_data["persistence_level"]),
                            checksum=event_data["checksum"],
                            delivered_to=set(event_data.get("delivered_to", [])),
                            replay_count=event_data.get("replay_count", 0)
                        )
                        events.append(entry)
                except Exception as e:
                    logger.warning("Failed to parse Redis event", key=key, error=str(e))
            
            # Apply filters
            if since:
                events = [e for e in events if e.timestamp > since]
            
            if event_types:
                events = [e for e in events if e.event_type in event_types]
            
            if include_delivered_to:
                events = [e for e in events if include_delivered_to not in e.delivered_to]
            
            # Sort and limit
            events.sort(key=lambda e: e.timestamp, reverse=True)
            
            return events[:limit]
            
        except Exception as e:
            logger.error("Failed to get events from Redis", error=str(e))
            return []
    
    async def _get_from_database(
        self, 
        project_id: str, 
        since: Optional[datetime], 
        event_types: Optional[List[str]], 
        limit: int,
        include_delivered_to: Optional[str]
    ) -> List[EventHistoryEntry]:
        """Get events from database storage."""
        if not self.db_session:
            return []
        
        try:
            query = select(EventStorageModel).where(
                EventStorageModel.project_id == project_id
            )
            
            if since:
                query = query.where(EventStorageModel.created_at > since)
            
            if event_types:
                query = query.where(EventStorageModel.event_type.in_(event_types))
            
            query = query.order_by(EventStorageModel.created_at.desc()).limit(limit)
            
            result = await self.db_session.execute(query)
            db_entries = result.scalars().all()
            
            events = []
            for db_entry in db_entries:
                # Check delivered_to filter
                if include_delivered_to and include_delivered_to in (db_entry.client_delivered or []):
                    continue
                
                entry = EventHistoryEntry(
                    event_id=str(db_entry.id),
                    project_id=str(db_entry.project_id),
                    event_type=db_entry.event_type,
                    event_data=db_entry.event_data,
                    correlation_id=db_entry.correlation_id,
                    timestamp=db_entry.created_at,
                    persistence_level=EventPersistenceLevel(db_entry.persistence_level),
                    checksum=db_entry.checksum,
                    delivered_to=set(db_entry.client_delivered or []),
                    replay_count=db_entry.replay_count
                )
                events.append(entry)
            
            self.metrics["database_reads"] += len(events)
            
            return events
            
        except Exception as e:
            logger.error("Failed to get events from database", error=str(e))
            return []
    
    async def _mark_events_delivered(self, event_ids: List[str], client_id: str) -> None:
        """Mark multiple events as delivered to client."""
        try:
            # Update memory
            for project_events in self.memory_events.values():
                for event_id in event_ids:
                    if event_id in project_events:
                        project_events[event_id].delivered_to.add(client_id)
            
            # Update Redis
            for event_id in event_ids:
                await self._update_redis_delivered(event_id, client_id)
            
            # Update database
            if self.db_session:
                await self._update_database_delivered(event_ids, client_id)
                
        except Exception as e:
            logger.error("Failed to mark events as delivered", error=str(e))
    
    async def _update_redis_delivered(self, event_id: str, client_id: str) -> None:
        """Update delivered clients list in Redis."""
        try:
            # Find the Redis key for this event
            pattern = f"project_index:events:*:{event_id}"
            keys = await self.redis_client.keys(pattern)
            
            for key in keys:
                data = await self.redis_client.get(key)
                if data:
                    event_data = json.loads(data)
                    delivered_to = set(event_data.get("delivered_to", []))
                    delivered_to.add(client_id)
                    event_data["delivered_to"] = list(delivered_to)
                    
                    # Update with same TTL
                    ttl = await self.redis_client.ttl(key)
                    if ttl > 0:
                        await self.redis_client.setex(key, ttl, json.dumps(event_data))
        except Exception as e:
            logger.warning("Failed to update Redis delivered status", error=str(e))
    
    async def _update_database_delivered(self, event_ids: List[str], client_id: str) -> None:
        """Update delivered clients list in database."""
        if not self.db_session:
            return
        
        try:
            for event_id in event_ids:
                # Get current delivered list
                query = select(EventStorageModel).where(EventStorageModel.id == event_id)
                result = await self.db_session.execute(query)
                entry = result.scalar_one_or_none()
                
                if entry:
                    delivered_to = set(entry.client_delivered or [])
                    delivered_to.add(client_id)
                    entry.client_delivered = list(delivered_to)
            
            await self.db_session.commit()
            
        except Exception as e:
            logger.error("Failed to update database delivered status", error=str(e))
            if self.db_session:
                await self.db_session.rollback()
    
    # ================ UTILITY METHODS ================
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate checksum for event data."""
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def _get_redis_ttl(self, persistence_level: EventPersistenceLevel) -> int:
        """Get Redis TTL in seconds for persistence level."""
        if persistence_level == EventPersistenceLevel.MEMORY_ONLY:
            return self.memory_ttl_hours * 3600
        elif persistence_level == EventPersistenceLevel.SHORT_TERM:
            return self.short_term_ttl_hours * 3600
        elif persistence_level == EventPersistenceLevel.MEDIUM_TERM:
            return self.medium_term_ttl_days * 24 * 3600
        elif persistence_level == EventPersistenceLevel.LONG_TERM:
            return self.long_term_ttl_days * 24 * 3600
        else:  # PERMANENT
            return 30 * 24 * 3600  # 30 days in Redis, then DB only
    
    def _get_database_ttl(self, persistence_level: EventPersistenceLevel) -> timedelta:
        """Get database TTL for persistence level."""
        if persistence_level == EventPersistenceLevel.SHORT_TERM:
            return timedelta(hours=self.short_term_ttl_hours)
        elif persistence_level == EventPersistenceLevel.MEDIUM_TERM:
            return timedelta(days=self.medium_term_ttl_days)
        elif persistence_level == EventPersistenceLevel.LONG_TERM:
            return timedelta(days=self.long_term_ttl_days)
        else:  # PERMANENT has no TTL
            return timedelta(days=365 * 10)  # 10 years
    
    async def _cleanup_memory(self) -> int:
        """Clean expired events from memory."""
        cleaned_count = 0
        cutoff = datetime.utcnow() - timedelta(hours=self.memory_ttl_hours)
        
        for project_id in list(self.memory_events.keys()):
            project_events = self.memory_events[project_id]
            
            expired_events = [
                event_id for event_id, entry in project_events.items()
                if entry.timestamp < cutoff
            ]
            
            for event_id in expired_events:
                del project_events[event_id]
                cleaned_count += 1
            
            # Remove empty project entries
            if not project_events:
                del self.memory_events[project_id]
        
        return cleaned_count
    
    async def _cleanup_redis(self) -> int:
        """Clean expired events from Redis (Redis handles TTL automatically)."""
        # Redis handles TTL cleanup automatically
        return 0
    
    async def _cleanup_database(self) -> int:
        """Clean expired events from database."""
        if not self.db_session:
            return 0
        
        try:
            now = datetime.utcnow()
            
            delete_query = delete(EventStorageModel).where(
                and_(
                    EventStorageModel.expires_at.isnot(None),
                    EventStorageModel.expires_at < now
                )
            )
            
            result = await self.db_session.execute(delete_query)
            await self.db_session.commit()
            
            return result.rowcount or 0
            
        except Exception as e:
            logger.error("Failed to cleanup database events", error=str(e))
            if self.db_session:
                await self.db_session.rollback()
            return 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event history performance metrics."""
        return {
            **self.metrics,
            "memory_projects": len(self.memory_events),
            "memory_events_total": sum(len(events) for events in self.memory_events.values())
        }


# Global event history manager
_event_history_manager: Optional[EventHistoryManager] = None


async def get_event_history_manager() -> EventHistoryManager:
    """Get or create the global event history manager."""
    global _event_history_manager
    if _event_history_manager is None:
        _event_history_manager = EventHistoryManager()
    return _event_history_manager