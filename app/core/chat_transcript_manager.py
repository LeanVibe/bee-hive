"""
Chat Transcript Manager for LeanVibe Agent Hive 2.0

Provides comprehensive conversation tracking, analysis, and storage system
for multi-agent communications with advanced debugging capabilities.

Features:
- Real-time conversation tracking and persistence
- Advanced pattern detection and analysis
- Semantic search and filtering capabilities
- Conversation replay and debugging tools
- Performance metrics and bottleneck detection
- Integration with Redis Streams and PostgreSQL
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import UUID, uuid4
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
from sqlalchemy import select, func, and_, or_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from .database import get_async_session
from .redis import get_message_broker
from .embedding_service import EmbeddingService
from .context_engine_integration import ContextEngineIntegration
from ..models.message import StreamMessage, MessageAudit, MessageType, MessageStatus
from ..models.conversation import Conversation, MessageType as ConversationType
from ..models.agent import Agent
from ..models.session import Session

logger = structlog.get_logger()


class ConversationEventType(Enum):
    """Types of conversation events for tracking and analysis."""
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    TOOL_INVOCATION = "tool_invocation"
    CONTEXT_SHARING = "context_sharing"
    ERROR_OCCURRED = "error_occurred"
    COORDINATION_REQUEST = "coordination_request"
    STATUS_UPDATE = "status_update"
    TASK_DELEGATION = "task_delegation"
    COLLABORATION_START = "collaboration_start"
    COLLABORATION_END = "collaboration_end"


class ConversationPattern(Enum):
    """Identified conversation patterns for debugging insights."""
    REQUEST_RESPONSE = "request_response"
    BROADCAST_FANOUT = "broadcast_fanout"
    COORDINATION_CHAIN = "coordination_chain"
    ERROR_CASCADE = "error_cascade"
    INFINITE_LOOP = "infinite_loop"
    BOTTLENECK_DETECTED = "bottleneck_detected"
    CONTEXT_EXPLOSION = "context_explosion"
    DEADLOCK_PATTERN = "deadlock_pattern"


@dataclass
class ConversationMetrics:
    """Metrics for conversation analysis and performance monitoring."""
    total_messages: int = 0
    unique_participants: int = 0
    average_response_time: float = 0.0
    message_frequency: float = 0.0
    error_rate: float = 0.0
    context_sharing_count: int = 0
    tool_invocations: int = 0
    longest_chain: int = 0
    circular_references: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationEvent:
    """Individual conversation event for detailed tracking."""
    id: str
    session_id: str
    timestamp: datetime
    event_type: ConversationEventType
    source_agent_id: str
    target_agent_id: Optional[str]
    message_content: str
    metadata: Dict[str, Any]
    response_time_ms: Optional[float] = None
    context_references: List[str] = None
    tool_calls: List[Dict[str, Any]] = None
    embedding_vector: Optional[List[float]] = None
    
    def __post_init__(self):
        if self.context_references is None:
            self.context_references = []
        if self.tool_calls is None:
            self.tool_calls = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'context_references': self.context_references,
            'tool_calls': self.tool_calls
        }


@dataclass
class ConversationThread:
    """Represents a conversation thread between agents."""
    thread_id: str
    session_id: str
    participants: Set[str]
    events: List[ConversationEvent]
    start_time: datetime
    end_time: Optional[datetime]
    thread_pattern: Optional[ConversationPattern]
    metrics: ConversationMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'thread_id': self.thread_id,
            'session_id': self.session_id,
            'participants': list(self.participants),
            'events': [event.to_dict() for event in self.events],
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'thread_pattern': self.thread_pattern.value if self.thread_pattern else None,
            'metrics': self.metrics.to_dict()
        }


class SearchFilter(BaseModel):
    """Filter configuration for conversation search."""
    session_ids: List[str] = Field(default_factory=list)
    agent_ids: List[str] = Field(default_factory=list)
    event_types: List[ConversationEventType] = Field(default_factory=list)
    content_keywords: List[str] = Field(default_factory=list)
    time_range: Optional[Dict[str, datetime]] = None
    min_response_time: Optional[float] = None
    max_response_time: Optional[float] = None
    patterns: List[ConversationPattern] = Field(default_factory=list)
    include_errors: bool = True
    include_system: bool = True
    semantic_query: Optional[str] = None
    limit: int = 1000


class ChatTranscriptManager:
    """
    Comprehensive chat transcript manager for multi-agent conversation tracking.
    
    Provides real-time conversation analysis, pattern detection, and debugging
    capabilities with advanced search and filtering functionality.
    """
    
    def __init__(self, db_session: AsyncSession, embedding_service: EmbeddingService):
        self.db_session = db_session
        self.embedding_service = embedding_service
        self.context_engine = ContextEngineIntegration()
        
        # In-memory tracking for real-time analysis
        self.active_conversations: Dict[str, ConversationThread] = {}
        self.conversation_cache: Dict[str, List[ConversationEvent]] = {}
        
        # Performance and debugging metrics
        self.processing_times: List[float] = []
        self.error_patterns: Dict[str, int] = {}
        
        # Configuration
        self.max_cache_size = 10000
        self.cache_ttl_hours = 24
        self.pattern_detection_enabled = True
        
        logger.info("ChatTranscriptManager initialized")
    
    async def track_conversation_event(
        self,
        session_id: str,
        event_type: ConversationEventType,
        source_agent_id: str,
        target_agent_id: Optional[str],
        message_content: str,
        metadata: Dict[str, Any] = None,
        response_time_ms: Optional[float] = None
    ) -> ConversationEvent:
        """
        Track a conversation event with comprehensive analysis.
        
        Args:
            session_id: Session identifier
            event_type: Type of conversation event
            source_agent_id: ID of the source agent
            target_agent_id: ID of the target agent (None for broadcast)
            message_content: Content of the message
            metadata: Additional event metadata
            response_time_ms: Response time in milliseconds
            
        Returns:
            ConversationEvent: The tracked event
        """
        start_time = time.time()
        
        try:
            # Create conversation event
            event = ConversationEvent(
                id=str(uuid4()),
                session_id=session_id,
                timestamp=datetime.utcnow(),
                event_type=event_type,
                source_agent_id=source_agent_id,
                target_agent_id=target_agent_id,
                message_content=message_content,
                metadata=metadata or {},
                response_time_ms=response_time_ms
            )
            
            # Generate embedding for semantic search
            if len(message_content.strip()) > 0:
                try:
                    embedding = await self.embedding_service.get_embedding(message_content)
                    event.embedding_vector = embedding
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for message: {e}")
            
            # Extract context references and tool calls from metadata
            if metadata:
                event.context_references = metadata.get("context_refs", [])
                event.tool_calls = metadata.get("tool_calls", [])
            
            # Store in database
            await self._persist_conversation_event(event)
            
            # Update in-memory tracking
            await self._update_conversation_thread(event)
            
            # Detect patterns in real-time
            if self.pattern_detection_enabled:
                await self._detect_conversation_patterns(session_id, event)
            
            # Update performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 1000:
                self.processing_times = self.processing_times[-1000:]
            
            logger.info(
                "Conversation event tracked",
                event_id=event.id,
                session_id=session_id,
                event_type=event_type.value,
                processing_time_ms=processing_time
            )
            
            return event
            
        except Exception as e:
            logger.error(
                "Failed to track conversation event",
                session_id=session_id,
                event_type=event_type.value,
                error=str(e)
            )
            raise
    
    async def get_conversation_transcript(
        self,
        session_id: str,
        agent_filter: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[ConversationEvent]:
        """
        Get chronological conversation transcript for a session.
        
        Args:
            session_id: Session identifier
            agent_filter: Optional list of agent IDs to filter by
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of events to return
            
        Returns:
            List of conversation events in chronological order
        """
        try:
            # Check cache first
            cache_key = f"{session_id}:{hash(str(agent_filter))}:{start_time}:{end_time}:{limit}"
            if cache_key in self.conversation_cache:
                return self.conversation_cache[cache_key]
            
            # Build query
            query = select(Conversation).where(Conversation.session_id == UUID(session_id))
            
            if agent_filter:
                agent_uuids = [UUID(agent_id) for agent_id in agent_filter]
                query = query.where(
                    or_(
                        Conversation.from_agent_id.in_(agent_uuids),
                        Conversation.to_agent_id.in_(agent_uuids)
                    )
                )
            
            if start_time:
                query = query.where(Conversation.created_at >= start_time)
            
            if end_time:
                query = query.where(Conversation.created_at <= end_time)
            
            query = query.order_by(asc(Conversation.created_at)).limit(limit)
            
            # Execute query
            result = await self.db_session.execute(query)
            conversations = result.scalars().all()
            
            # Convert to conversation events
            events = []
            for conv in conversations:
                event = ConversationEvent(
                    id=str(conv.id),
                    session_id=str(conv.session_id),
                    timestamp=conv.created_at,
                    event_type=self._map_message_type_to_event_type(conv.message_type),
                    source_agent_id=str(conv.from_agent_id),
                    target_agent_id=str(conv.to_agent_id) if conv.to_agent_id else None,
                    message_content=conv.content,
                    metadata=conv.conversation_metadata or {},
                    context_references=conv.context_refs or []
                )
                events.append(event)
            
            # Cache results
            self.conversation_cache[cache_key] = events
            await self._cleanup_cache()
            
            logger.info(
                "Conversation transcript retrieved",
                session_id=session_id,
                event_count=len(events),
                agent_filter=agent_filter
            )
            
            return events
            
        except Exception as e:
            logger.error(
                "Failed to get conversation transcript",
                session_id=session_id,
                error=str(e)
            )
            raise
    
    async def search_conversations(self, search_filter: SearchFilter) -> List[ConversationEvent]:
        """
        Advanced search across conversations with semantic analysis.
        
        Args:
            search_filter: Search filter configuration
            
        Returns:
            List of matching conversation events
        """
        try:
            events = []
            
            # Semantic search if query provided
            if search_filter.semantic_query:
                events.extend(await self._semantic_search(search_filter))
            
            # Keyword and metadata search
            if search_filter.content_keywords or search_filter.session_ids or search_filter.agent_ids:
                events.extend(await self._keyword_search(search_filter))
            
            # Pattern-based search
            if search_filter.patterns:
                events.extend(await self._pattern_search(search_filter))
            
            # Remove duplicates and sort by relevance/timestamp
            unique_events = {event.id: event for event in events}
            sorted_events = sorted(
                unique_events.values(),
                key=lambda x: x.timestamp,
                reverse=True
            )
            
            return sorted_events[:search_filter.limit]
            
        except Exception as e:
            logger.error(f"Failed to search conversations: {e}")
            raise
    
    async def get_conversation_analytics(
        self,
        session_id: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get comprehensive conversation analytics for a session.
        
        Args:
            session_id: Session identifier
            time_window_hours: Time window for analysis
            
        Returns:
            Analytics dictionary with metrics and insights
        """
        try:
            start_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            events = await self.get_conversation_transcript(
                session_id=session_id,
                start_time=start_time
            )
            
            # Calculate metrics
            metrics = await self._calculate_conversation_metrics(events)
            
            # Detect patterns
            patterns = await self._analyze_conversation_patterns(events)
            
            # Performance analysis
            performance = await self._analyze_performance_metrics(events)
            
            # Agent participation analysis
            participation = await self._analyze_agent_participation(events)
            
            return {
                "session_id": session_id,
                "time_window_hours": time_window_hours,
                "total_events": len(events),
                "metrics": metrics.to_dict(),
                "patterns": [p.value for p in patterns],
                "performance": performance,
                "participation": participation,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(
                f"Failed to get conversation analytics for session {session_id}: {e}"
            )
            raise
    
    async def replay_conversation(
        self,
        session_id: str,
        target_session_id: Optional[str] = None,
        speed_multiplier: float = 1.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Replay conversation events for debugging purposes.
        
        Args:
            session_id: Original session ID to replay
            target_session_id: Target session for replay (creates new if None)
            speed_multiplier: Replay speed (1.0 = real-time)
            start_time: Start time for replay
            end_time: End time for replay
            
        Returns:
            Replay session information
        """
        try:
            if not target_session_id:
                target_session_id = f"replay_{session_id}_{uuid4().hex[:8]}"
            
            # Get events to replay
            events = await self.get_conversation_transcript(
                session_id=session_id,
                start_time=start_time,
                end_time=end_time
            )
            
            if not events:
                return {
                    "status": "no_events",
                    "message": "No events found for replay"
                }
            
            # Calculate timing for replay
            base_time = events[0].timestamp
            replay_start = datetime.utcnow()
            
            replayed_count = 0
            
            for event in events:
                # Calculate delay based on original timing and speed multiplier
                original_delay = (event.timestamp - base_time).total_seconds()
                replay_delay = original_delay / speed_multiplier
                
                # Wait for the appropriate time
                if replay_delay > 0:
                    await asyncio.sleep(min(replay_delay, 60))  # Cap at 60 seconds
                
                # Create replay event
                replay_event = ConversationEvent(
                    id=str(uuid4()),
                    session_id=target_session_id,
                    timestamp=datetime.utcnow(),
                    event_type=event.event_type,
                    source_agent_id=f"replay_{event.source_agent_id}",
                    target_agent_id=f"replay_{event.target_agent_id}" if event.target_agent_id else None,
                    message_content=f"[REPLAY] {event.message_content}",
                    metadata={
                        **event.metadata,
                        "original_event_id": event.id,
                        "original_session_id": session_id,
                        "replay_marker": True
                    }
                )
                
                # Track replay event
                await self._persist_conversation_event(replay_event)
                replayed_count += 1
                
                base_time = event.timestamp
            
            return {
                "status": "completed",
                "target_session_id": target_session_id,
                "events_replayed": replayed_count,
                "original_session_id": session_id,
                "replay_duration_seconds": (datetime.utcnow() - replay_start).total_seconds(),
                "speed_multiplier": speed_multiplier
            }
            
        except Exception as e:
            logger.error(
                f"Failed to replay conversation for session {session_id}: {e}"
            )
            raise
    
    async def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights for the transcript manager."""
        return {
            "processing_times": {
                "average_ms": sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0,
                "max_ms": max(self.processing_times) if self.processing_times else 0,
                "min_ms": min(self.processing_times) if self.processing_times else 0,
                "samples": len(self.processing_times)
            },
            "cache_stats": {
                "active_conversations": len(self.active_conversations),
                "cached_transcripts": len(self.conversation_cache),
                "max_cache_size": self.max_cache_size
            },
            "error_patterns": dict(self.error_patterns),
            "pattern_detection_enabled": self.pattern_detection_enabled
        }
    
    # Private helper methods
    
    async def _persist_conversation_event(self, event: ConversationEvent) -> None:
        """Persist conversation event to database."""
        try:
            conversation = Conversation(
                session_id=UUID(event.session_id),
                from_agent_id=UUID(event.source_agent_id),
                to_agent_id=UUID(event.target_agent_id) if event.target_agent_id else None,
                message_type=self._map_event_type_to_message_type(event.event_type),
                content=event.message_content,
                embedding=event.embedding_vector,
                context_refs=event.context_references,
                conversation_metadata=event.metadata
            )
            
            self.db_session.add(conversation)
            await self.db_session.commit()
            
        except Exception as e:
            await self.db_session.rollback()
            logger.error(f"Failed to persist conversation event: {e}")
            raise
    
    async def _update_conversation_thread(self, event: ConversationEvent) -> None:
        """Update in-memory conversation thread tracking."""
        thread_key = f"{event.session_id}:{event.source_agent_id}"
        
        if thread_key not in self.active_conversations:
            self.active_conversations[thread_key] = ConversationThread(
                thread_id=thread_key,
                session_id=event.session_id,
                participants={event.source_agent_id},
                events=[],
                start_time=event.timestamp,
                end_time=None,
                thread_pattern=None,
                metrics=ConversationMetrics()
            )
        
        thread = self.active_conversations[thread_key]
        thread.events.append(event)
        thread.participants.add(event.source_agent_id)
        if event.target_agent_id:
            thread.participants.add(event.target_agent_id)
        
        # Update metrics
        thread.metrics.total_messages += 1
        thread.metrics.unique_participants = len(thread.participants)
        
        if event.response_time_ms:
            current_avg = thread.metrics.average_response_time
            total_msgs = thread.metrics.total_messages
            thread.metrics.average_response_time = (
                (current_avg * (total_msgs - 1) + event.response_time_ms) / total_msgs
            )
        
        if event.event_type == ConversationEventType.ERROR_OCCURRED:
            thread.metrics.error_rate = (
                thread.metrics.error_rate * (thread.metrics.total_messages - 1) + 1
            ) / thread.metrics.total_messages
        
        if event.context_references:
            thread.metrics.context_sharing_count += 1
        
        if event.tool_calls:
            thread.metrics.tool_invocations += len(event.tool_calls)
    
    async def _detect_conversation_patterns(
        self,
        session_id: str,
        event: ConversationEvent
    ) -> None:
        """Detect conversation patterns for debugging insights."""
        try:
            # Simple pattern detection - can be enhanced with ML models
            thread_key = f"{session_id}:{event.source_agent_id}"
            if thread_key in self.active_conversations:
                thread = self.active_conversations[thread_key]
                
                # Detect infinite loops
                if len(thread.events) > 10:
                    recent_sources = [e.source_agent_id for e in thread.events[-10:]]
                    if len(set(recent_sources)) <= 2:  # Only 2 agents talking back and forth
                        thread.thread_pattern = ConversationPattern.INFINITE_LOOP
                
                # Detect error cascades
                recent_errors = [
                    e for e in thread.events[-5:]
                    if e.event_type == ConversationEventType.ERROR_OCCURRED
                ]
                if len(recent_errors) >= 3:
                    thread.thread_pattern = ConversationPattern.ERROR_CASCADE
                
                # Detect bottlenecks
                if event.response_time_ms and event.response_time_ms > 5000:  # 5+ seconds
                    thread.thread_pattern = ConversationPattern.BOTTLENECK_DETECTED
                
        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}")
    
    def _map_message_type_to_event_type(self, message_type: ConversationType) -> ConversationEventType:
        """Map conversation message type to event type."""
        mapping = {
            ConversationType.TASK_ASSIGNMENT: ConversationEventType.TASK_DELEGATION,
            ConversationType.STATUS_UPDATE: ConversationEventType.STATUS_UPDATE,
            ConversationType.COMPLETION: ConversationEventType.STATUS_UPDATE,
            ConversationType.ERROR: ConversationEventType.ERROR_OCCURRED,
            ConversationType.COLLABORATION: ConversationEventType.COLLABORATION_START,
            ConversationType.COORDINATION: ConversationEventType.COORDINATION_REQUEST
        }
        return mapping.get(message_type, ConversationEventType.MESSAGE_SENT)
    
    def _map_event_type_to_message_type(self, event_type: ConversationEventType) -> ConversationType:
        """Map event type to conversation message type."""
        mapping = {
            ConversationEventType.TASK_DELEGATION: ConversationType.TASK_ASSIGNMENT,
            ConversationEventType.STATUS_UPDATE: ConversationType.STATUS_UPDATE,
            ConversationEventType.ERROR_OCCURRED: ConversationType.ERROR,
            ConversationEventType.COLLABORATION_START: ConversationType.COLLABORATION,
            ConversationEventType.COORDINATION_REQUEST: ConversationType.COORDINATION
        }
        return mapping.get(event_type, ConversationType.COLLABORATION)
    
    async def _semantic_search(self, search_filter: SearchFilter) -> List[ConversationEvent]:
        """Perform semantic search using embeddings."""
        # Implementation would use vector similarity search
        # Placeholder for now
        return []
    
    async def _keyword_search(self, search_filter: SearchFilter) -> List[ConversationEvent]:
        """Perform keyword-based search."""
        # Implementation would use text search in database
        # Placeholder for now
        return []
    
    async def _pattern_search(self, search_filter: SearchFilter) -> List[ConversationEvent]:
        """Search for specific conversation patterns."""
        # Implementation would analyze thread patterns
        # Placeholder for now
        return []
    
    async def _calculate_conversation_metrics(self, events: List[ConversationEvent]) -> ConversationMetrics:
        """Calculate comprehensive metrics from conversation events."""
        if not events:
            return ConversationMetrics()
        
        metrics = ConversationMetrics()
        metrics.total_messages = len(events)
        metrics.unique_participants = len(set(e.source_agent_id for e in events))
        
        response_times = [e.response_time_ms for e in events if e.response_time_ms]
        if response_times:
            metrics.average_response_time = sum(response_times) / len(response_times)
        
        errors = [e for e in events if e.event_type == ConversationEventType.ERROR_OCCURRED]
        metrics.error_rate = len(errors) / len(events) if events else 0
        
        metrics.context_sharing_count = sum(len(e.context_references) for e in events)
        metrics.tool_invocations = sum(len(e.tool_calls) for e in events)
        
        return metrics
    
    async def _analyze_conversation_patterns(self, events: List[ConversationEvent]) -> List[ConversationPattern]:
        """Analyze conversation events for patterns."""
        patterns = []
        
        # Simple pattern detection
        if len(events) > 50:
            # Check for potential infinite loops
            recent_events = events[-20:]
            sources = [e.source_agent_id for e in recent_events]
            if len(set(sources)) <= 2:
                patterns.append(ConversationPattern.INFINITE_LOOP)
        
        # Check for error cascades
        error_events = [e for e in events if e.event_type == ConversationEventType.ERROR_OCCURRED]
        if len(error_events) > len(events) * 0.1:  # More than 10% errors
            patterns.append(ConversationPattern.ERROR_CASCADE)
        
        return patterns
    
    async def _analyze_performance_metrics(self, events: List[ConversationEvent]) -> Dict[str, Any]:
        """Analyze performance metrics from events."""
        response_times = [e.response_time_ms for e in events if e.response_time_ms]
        
        return {
            "response_times": {
                "average": sum(response_times) / len(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
                "min": min(response_times) if response_times else 0,
                "samples": len(response_times)
            },
            "message_frequency": len(events) / 24 if events else 0,  # messages per hour
            "tool_usage": sum(len(e.tool_calls) for e in events),
            "context_sharing": sum(len(e.context_references) for e in events)
        }
    
    async def _analyze_agent_participation(self, events: List[ConversationEvent]) -> Dict[str, Any]:
        """Analyze agent participation patterns."""
        agent_stats = {}
        
        for event in events:
            agent_id = event.source_agent_id
            if agent_id not in agent_stats:
                agent_stats[agent_id] = {
                    "message_count": 0,
                    "tool_calls": 0,
                    "context_shares": 0,
                    "errors": 0,
                    "first_seen": event.timestamp.isoformat(),
                    "last_seen": event.timestamp.isoformat()
                }
            
            stats = agent_stats[agent_id]
            stats["message_count"] += 1
            stats["tool_calls"] += len(event.tool_calls)
            stats["context_shares"] += len(event.context_references)
            
            if event.event_type == ConversationEventType.ERROR_OCCURRED:
                stats["errors"] += 1
            
            if event.timestamp.isoformat() > stats["last_seen"]:
                stats["last_seen"] = event.timestamp.isoformat()
        
        return {
            "total_agents": len(agent_stats),
            "agent_details": agent_stats,
            "most_active": max(
                agent_stats.items(),
                key=lambda x: x[1]["message_count"],
                default=(None, {"message_count": 0})
            )[0] if agent_stats else None
        }
    
    async def _cleanup_cache(self) -> None:
        """Clean up old cache entries."""
        if len(self.conversation_cache) > self.max_cache_size:
            # Remove oldest 25% of cache entries
            to_remove = len(self.conversation_cache) // 4
            keys_to_remove = list(self.conversation_cache.keys())[:to_remove]
            for key in keys_to_remove:
                del self.conversation_cache[key]