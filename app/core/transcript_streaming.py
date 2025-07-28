"""
Real-time Transcript Streaming System for LeanVibe Agent Hive 2.0

Provides WebSocket-based real-time streaming of conversation transcripts
with advanced filtering, session management, and debugging capabilities.

Features:
- WebSocket streaming of live conversations
- Advanced filtering and session management
- Real-time pattern detection and alerts
- Conversation replay streaming
- Performance metrics and monitoring
- Integration with chat transcript manager
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from uuid import uuid4
from dataclasses import dataclass, asdict
from enum import Enum

import structlog
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .chat_transcript_manager import (
    ChatTranscriptManager, ConversationEvent, ConversationEventType,
    SearchFilter, ConversationThread
)
from .communication_analyzer import (
    CommunicationAnalyzer, CommunicationInsight, AnalysisType, AlertSeverity
)

logger = structlog.get_logger()


class StreamEventType(Enum):
    """Types of streaming events."""
    CONVERSATION_EVENT = "conversation_event"
    PATTERN_ALERT = "pattern_alert"
    PERFORMANCE_UPDATE = "performance_update"
    SESSION_UPDATE = "session_update"
    ERROR_ALERT = "error_alert"
    INSIGHT_GENERATED = "insight_generated"
    REPLAY_EVENT = "replay_event"
    SYSTEM_STATUS = "system_status"


class FilterMode(Enum):
    """Filter application modes."""
    INCLUSIVE = "inclusive"  # Show events matching any filter
    EXCLUSIVE = "exclusive"  # Show events matching all filters
    CUSTOM = "custom"  # Custom filter logic


@dataclass
class StreamingFilter:
    """Advanced streaming filter configuration."""
    session_ids: List[str]
    agent_ids: List[str] 
    event_types: List[ConversationEventType]
    stream_events: List[StreamEventType]
    keywords: List[str]
    min_severity: AlertSeverity
    real_time_only: bool
    include_patterns: bool
    include_performance: bool
    filter_mode: FilterMode
    
    def matches_event(self, event: ConversationEvent) -> bool:
        """Check if event matches filter criteria."""
        matches = []
        
        if self.session_ids:
            matches.append(event.session_id in self.session_ids)
        
        if self.agent_ids:
            matches.append(
                event.source_agent_id in self.agent_ids or
                (event.target_agent_id and event.target_agent_id in self.agent_ids)
            )
        
        if self.event_types:
            matches.append(event.event_type in self.event_types)
        
        if self.keywords:
            content = event.message_content.lower()
            matches.append(any(keyword.lower() in content for keyword in self.keywords))
        
        if not matches:
            return True  # No filters applied
        
        if self.filter_mode == FilterMode.INCLUSIVE:
            return any(matches)
        elif self.filter_mode == FilterMode.EXCLUSIVE:
            return all(matches)
        else:  # CUSTOM
            return any(matches)  # Default to inclusive


@dataclass
class StreamingSession:
    """Represents a WebSocket streaming session."""
    session_id: str
    websocket: WebSocket
    filter_config: StreamingFilter
    start_time: datetime
    last_activity: datetime
    events_sent: int
    bytes_sent: int
    connection_id: str
    
    def update_activity(self, bytes_count: int = 0) -> None:
        """Update session activity metrics."""
        self.last_activity = datetime.utcnow()
        self.events_sent += 1
        self.bytes_sent += bytes_count


@dataclass
class StreamMessage:
    """Message sent over WebSocket stream."""
    event_type: StreamEventType
    timestamp: datetime
    session_id: str
    data: Dict[str, Any]
    sequence_id: int
    
    def to_json(self) -> str:
        """Convert to JSON string for WebSocket transmission."""
        return json.dumps({
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'data': self.data,
            'sequence_id': self.sequence_id
        })


class TranscriptStreamingManager:
    """
    Manages real-time transcript streaming with WebSocket connections.
    
    Provides live conversation streaming, filtering, and analysis with
    comprehensive session management and performance monitoring.
    """
    
    def __init__(
        self,
        transcript_manager: ChatTranscriptManager,
        analyzer: CommunicationAnalyzer
    ):
        self.transcript_manager = transcript_manager
        self.analyzer = analyzer
        
        # Active streaming sessions
        self.active_sessions: Dict[str, StreamingSession] = {}
        self.session_filters: Dict[str, StreamingFilter] = {}
        
        # Event broadcasting
        self.event_subscribers: Dict[str, Set[str]] = {}  # event_type -> session_ids
        self.sequence_counter = 0
        
        # Performance monitoring
        self.streaming_metrics: Dict[str, Any] = {
            'total_sessions': 0,
            'active_connections': 0,
            'events_streamed': 0,
            'bytes_transmitted': 0,
            'average_latency_ms': 0,
            'error_count': 0
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        logger.info("TranscriptStreamingManager initialized")
    
    async def register_streaming_session(
        self,
        websocket: WebSocket,
        session_filter: StreamingFilter,
        connection_id: Optional[str] = None
    ) -> str:
        """
        Register a new WebSocket streaming session.
        
        Args:
            websocket: WebSocket connection
            session_filter: Filter configuration for the session
            connection_id: Optional connection identifier
            
        Returns:
            Session ID for the registered connection
        """
        if not connection_id:
            connection_id = str(uuid4())
        
        try:
            await websocket.accept()
            
            # Create streaming session
            session = StreamingSession(
                session_id=connection_id,
                websocket=websocket,
                filter_config=session_filter,
                start_time=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                events_sent=0,
                bytes_sent=0,
                connection_id=connection_id
            )
            
            self.active_sessions[connection_id] = session
            self.session_filters[connection_id] = session_filter
            
            # Subscribe to relevant event types
            for event_type in session_filter.stream_events:
                if event_type.value not in self.event_subscribers:
                    self.event_subscribers[event_type.value] = set()
                self.event_subscribers[event_type.value].add(connection_id)
            
            # Update metrics
            self.streaming_metrics['total_sessions'] += 1
            self.streaming_metrics['active_connections'] = len(self.active_sessions)
            
            # Send welcome message
            await self._send_stream_message(
                connection_id,
                StreamEventType.SYSTEM_STATUS,
                {
                    'status': 'connected',
                    'session_id': connection_id,
                    'filter_config': asdict(session_filter),
                    'server_time': datetime.utcnow().isoformat()
                }
            )
            
            # Start session management task
            task = asyncio.create_task(self._manage_session(connection_id))
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            
            logger.info(
                "Streaming session registered",
                session_id=connection_id,
                filter_config=asdict(session_filter),
                active_sessions=len(self.active_sessions)
            )
            
            return connection_id
            
        except Exception as e:
            logger.error(f"Failed to register streaming session: {e}")
            raise
    
    async def unregister_streaming_session(self, connection_id: str) -> None:
        """Unregister a WebSocket streaming session."""
        if connection_id in self.active_sessions:
            session = self.active_sessions[connection_id]
            
            # Remove from event subscribers
            for event_type_set in self.event_subscribers.values():
                event_type_set.discard(connection_id)
            
            # Clean up session
            del self.active_sessions[connection_id]
            if connection_id in self.session_filters:
                del self.session_filters[connection_id]
            
            # Update metrics
            self.streaming_metrics['active_connections'] = len(self.active_sessions)
            
            logger.info(
                "Streaming session unregistered",
                session_id=connection_id,
                session_duration=f"{(datetime.utcnow() - session.start_time).total_seconds():.1f}s",
                events_sent=session.events_sent,
                remaining_sessions=len(self.active_sessions)
            )
    
    async def stream_conversation_event(
        self,
        event: ConversationEvent,
        real_time: bool = True
    ) -> None:
        """
        Stream a conversation event to subscribed sessions.
        
        Args:
            event: Conversation event to stream
            real_time: Whether this is a real-time event
        """
        try:
            # Find matching sessions
            matching_sessions = []
            for session_id, session in self.active_sessions.items():
                if (StreamEventType.CONVERSATION_EVENT in [se for se in session.filter_config.stream_events] and
                    session.filter_config.matches_event(event)):
                    
                    # Check real-time filter
                    if session.filter_config.real_time_only and not real_time:
                        continue
                    
                    matching_sessions.append(session_id)
            
            # Stream to matching sessions
            if matching_sessions:
                stream_data = {
                    'event': event.to_dict(),
                    'real_time': real_time,
                    'analysis_metadata': {
                        'processing_time_ms': time.time() * 1000,
                        'stream_latency_ms': 0  # Will be calculated
                    }
                }
                
                # Add pattern analysis if enabled
                if any(self.session_filters[sid].include_patterns for sid in matching_sessions):
                    patterns = await self._quick_pattern_analysis(event)
                    stream_data['patterns'] = patterns
                
                await self._broadcast_to_sessions(
                    matching_sessions,
                    StreamEventType.CONVERSATION_EVENT,
                    stream_data
                )
            
            # Update metrics
            self.streaming_metrics['events_streamed'] += 1
            
        except Exception as e:
            self.streaming_metrics['error_count'] += 1
            logger.error(f"Failed to stream conversation event: {e}")
    
    async def stream_insight(
        self,
        insight: CommunicationInsight,
        affected_sessions: Optional[List[str]] = None
    ) -> None:
        """
        Stream a communication insight/alert to relevant sessions.
        
        Args:
            insight: Communication insight to stream
            affected_sessions: Specific sessions to send to (None for all matching)
        """
        try:
            # Determine target sessions
            if affected_sessions:
                target_sessions = [
                    sid for sid in affected_sessions 
                    if sid in self.active_sessions
                ]
            else:
                # Find sessions interested in insights and matching severity
                target_sessions = []
                for session_id, session in self.active_sessions.items():
                    if (StreamEventType.INSIGHT_GENERATED in session.filter_config.stream_events and
                        insight.severity.value >= session.filter_config.min_severity.value):
                        
                        # Check if insight affects agents in filter
                        if session.filter_config.agent_ids:
                            if any(agent in insight.affected_agents for agent in session.filter_config.agent_ids):
                                target_sessions.append(session_id)
                        else:
                            target_sessions.append(session_id)
            
            if target_sessions:
                stream_data = {
                    'insight': insight.to_dict(),
                    'urgency': insight.severity.value,
                    'action_required': insight.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]
                }
                
                await self._broadcast_to_sessions(
                    target_sessions,
                    StreamEventType.INSIGHT_GENERATED,
                    stream_data
                )
            
        except Exception as e:
            logger.error(f"Failed to stream insight: {e}")
    
    async def stream_performance_update(
        self,
        session_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Stream performance metrics update."""
        try:
            # Find sessions interested in performance updates
            target_sessions = [
                sid for sid, session in self.active_sessions.items()
                if (StreamEventType.PERFORMANCE_UPDATE in session.filter_config.stream_events and
                    (not session.filter_config.session_ids or session_id in session.filter_config.session_ids))
            ]
            
            if target_sessions:
                stream_data = {
                    'session_id': session_id,
                    'metrics': metrics,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                await self._broadcast_to_sessions(
                    target_sessions,
                    StreamEventType.PERFORMANCE_UPDATE,
                    stream_data
                )
            
        except Exception as e:
            logger.error(f"Failed to stream performance update: {e}")
    
    async def start_conversation_replay_stream(
        self,
        connection_id: str,
        original_session_id: str,
        speed_multiplier: float = 1.0,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Start streaming a conversation replay to a specific connection.
        
        Args:
            connection_id: Target WebSocket connection
            original_session_id: Session to replay
            speed_multiplier: Replay speed (1.0 = real-time)
            start_time: Start time for replay
            end_time: End time for replay
            
        Returns:
            Replay session information
        """
        if connection_id not in self.active_sessions:
            raise ValueError(f"Connection {connection_id} not found")
        
        try:
            session = self.active_sessions[connection_id]
            
            # Get conversation events to replay
            events = await self.transcript_manager.get_conversation_transcript(
                session_id=original_session_id,
                start_time=start_time,
                end_time=end_time
            )
            
            if not events:
                await self._send_stream_message(
                    connection_id,
                    StreamEventType.REPLAY_EVENT,
                    {
                        'status': 'no_events',
                        'message': 'No events found for replay',
                        'original_session_id': original_session_id
                    }
                )
                return {'status': 'no_events', 'event_count': 0}
            
            # Send replay start notification
            await self._send_stream_message(
                connection_id,
                StreamEventType.REPLAY_EVENT,
                {
                    'status': 'starting',
                    'original_session_id': original_session_id,
                    'event_count': len(events),
                    'speed_multiplier': speed_multiplier,
                    'estimated_duration_seconds': (
                        (events[-1].timestamp - events[0].timestamp).total_seconds() / speed_multiplier
                        if len(events) > 1 else 0
                    )
                }
            )
            
            # Start replay task
            replay_task = asyncio.create_task(
                self._execute_replay_stream(
                    connection_id, events, speed_multiplier, original_session_id
                )
            )
            self.background_tasks.add(replay_task)
            replay_task.add_done_callback(self.background_tasks.discard)
            
            return {
                'status': 'started',
                'original_session_id': original_session_id,
                'event_count': len(events),
                'speed_multiplier': speed_multiplier
            }
            
        except Exception as e:
            logger.error(f"Failed to start replay stream: {e}")
            raise
    
    async def get_streaming_statistics(self) -> Dict[str, Any]:
        """Get comprehensive streaming statistics."""
        try:
            # Calculate session statistics
            session_stats = []
            total_duration = 0
            
            for session in self.active_sessions.values():
                duration = (datetime.utcnow() - session.start_time).total_seconds()
                total_duration += duration
                
                session_stats.append({
                    'session_id': session.session_id,
                    'duration_seconds': duration,
                    'events_sent': session.events_sent,
                    'bytes_sent': session.bytes_sent,
                    'last_activity': session.last_activity.isoformat(),
                    'filter_active': len(session.filter_config.session_ids) > 0
                })
            
            # Event subscription statistics
            subscription_stats = {
                event_type: len(sessions)
                for event_type, sessions in self.event_subscribers.items()
            }
            
            return {
                'overview': {
                    **self.streaming_metrics,
                    'active_connections': len(self.active_sessions),
                    'average_session_duration': total_duration / len(self.active_sessions) if self.active_sessions else 0
                },
                'sessions': session_stats,
                'subscriptions': subscription_stats,
                'background_tasks': len(self.background_tasks),
                'generated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get streaming statistics: {e}")
            return {'error': str(e)}
    
    async def update_session_filter(
        self,
        connection_id: str,
        new_filter: StreamingFilter
    ) -> bool:
        """Update filter configuration for an active session."""
        if connection_id not in self.active_sessions:
            return False
        
        try:
            # Update session filter
            old_filter = self.session_filters[connection_id]
            self.session_filters[connection_id] = new_filter
            self.active_sessions[connection_id].filter_config = new_filter
            
            # Update event subscriptions
            # Remove old subscriptions
            for event_type in old_filter.stream_events:
                if event_type.value in self.event_subscribers:
                    self.event_subscribers[event_type.value].discard(connection_id)
            
            # Add new subscriptions
            for event_type in new_filter.stream_events:
                if event_type.value not in self.event_subscribers:
                    self.event_subscribers[event_type.value] = set()
                self.event_subscribers[event_type.value].add(connection_id)
            
            # Notify session of filter update
            await self._send_stream_message(
                connection_id,
                StreamEventType.SYSTEM_STATUS,
                {
                    'status': 'filter_updated',
                    'new_filter': asdict(new_filter),
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            logger.info(
                "Session filter updated",
                session_id=connection_id,
                old_events=len(old_filter.stream_events),
                new_events=len(new_filter.stream_events)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session filter: {e}")
            return False
    
    # Private helper methods
    
    async def _manage_session(self, connection_id: str) -> None:
        """Manage a WebSocket session lifecycle."""
        try:
            session = self.active_sessions.get(connection_id)
            if not session:
                return
            
            # Session management loop
            while connection_id in self.active_sessions:
                try:
                    # Wait for client messages or timeout
                    message = await asyncio.wait_for(
                        session.websocket.receive_text(),
                        timeout=30.0
                    )
                    
                    # Handle client message
                    await self._handle_client_message(connection_id, message)
                    
                except asyncio.TimeoutError:
                    # Send keepalive
                    await self._send_stream_message(
                        connection_id,
                        StreamEventType.SYSTEM_STATUS,
                        {'status': 'keepalive', 'timestamp': datetime.utcnow().isoformat()}
                    )
                    continue
                    
                except WebSocketDisconnect:
                    break
                    
                except Exception as e:
                    logger.error(f"Session management error for {connection_id}: {e}")
                    break
            
        except Exception as e:
            logger.error(f"Session management failed for {connection_id}: {e}")
        finally:
            await self.unregister_streaming_session(connection_id)
    
    async def _handle_client_message(self, connection_id: str, message: str) -> None:
        """Handle incoming client message."""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type == 'filter_update':
                # Update session filter
                filter_data = data.get('filter', {})
                new_filter = StreamingFilter(**filter_data)
                await self.update_session_filter(connection_id, new_filter)
                
            elif message_type == 'request_replay':
                # Start conversation replay
                session_id = data.get('session_id')
                speed = data.get('speed_multiplier', 1.0)
                
                if session_id:
                    await self.start_conversation_replay_stream(
                        connection_id, session_id, speed
                    )
                    
            elif message_type == 'request_statistics':
                # Send streaming statistics
                stats = await self.get_streaming_statistics()
                await self._send_stream_message(
                    connection_id,
                    StreamEventType.SYSTEM_STATUS,
                    {'status': 'statistics', 'data': stats}
                )
                
            elif message_type == 'ping':
                # Respond to ping
                await self._send_stream_message(
                    connection_id,
                    StreamEventType.SYSTEM_STATUS,
                    {'status': 'pong', 'timestamp': datetime.utcnow().isoformat()}
                )
            
        except json.JSONDecodeError:
            await self._send_stream_message(
                connection_id,
                StreamEventType.SYSTEM_STATUS,
                {'status': 'error', 'message': 'Invalid JSON'}
            )
        except Exception as e:
            logger.error(f"Client message handling failed: {e}")
            await self._send_stream_message(
                connection_id,
                StreamEventType.SYSTEM_STATUS,
                {'status': 'error', 'message': str(e)}
            )
    
    async def _send_stream_message(
        self,
        connection_id: str,
        event_type: StreamEventType,
        data: Dict[str, Any]
    ) -> None:
        """Send a message to a specific streaming session."""
        if connection_id not in self.active_sessions:
            return
        
        session = self.active_sessions[connection_id]
        
        try:
            message = StreamMessage(
                event_type=event_type,
                timestamp=datetime.utcnow(),
                session_id=connection_id,
                data=data,
                sequence_id=self._next_sequence_id()
            )
            
            json_message = message.to_json()
            await session.websocket.send_text(json_message)
            
            # Update session metrics
            session.update_activity(len(json_message))
            self.streaming_metrics['bytes_transmitted'] += len(json_message)
            
        except WebSocketDisconnect:
            await self.unregister_streaming_session(connection_id)
        except Exception as e:
            logger.error(f"Failed to send stream message to {connection_id}: {e}")
            self.streaming_metrics['error_count'] += 1
    
    async def _broadcast_to_sessions(
        self,
        session_ids: List[str],
        event_type: StreamEventType,
        data: Dict[str, Any]
    ) -> None:
        """Broadcast a message to multiple sessions."""
        tasks = []
        for session_id in session_ids:
            if session_id in self.active_sessions:
                task = self._send_stream_message(session_id, event_type, data)
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_replay_stream(
        self,
        connection_id: str,
        events: List[ConversationEvent],
        speed_multiplier: float,
        original_session_id: str
    ) -> None:
        """Execute conversation replay streaming."""
        try:
            if not events:
                return
            
            base_time = events[0].timestamp
            replay_start = time.time()
            
            for i, event in enumerate(events):
                if connection_id not in self.active_sessions:
                    break  # Session disconnected
                
                # Calculate delay based on original timing
                if i > 0:
                    original_delay = (event.timestamp - events[i-1].timestamp).total_seconds()
                    replay_delay = original_delay / speed_multiplier
                    
                    if replay_delay > 0:
                        await asyncio.sleep(min(replay_delay, 60))  # Cap at 60 seconds
                
                # Send replay event
                replay_data = {
                    'status': 'event',
                    'original_event': event.to_dict(),
                    'replay_progress': (i + 1) / len(events),
                    'original_session_id': original_session_id,
                    'replay_timestamp': datetime.utcnow().isoformat()
                }
                
                await self._send_stream_message(
                    connection_id,
                    StreamEventType.REPLAY_EVENT,
                    replay_data
                )
            
            # Send completion notification
            replay_duration = time.time() - replay_start
            await self._send_stream_message(
                connection_id,
                StreamEventType.REPLAY_EVENT,
                {
                    'status': 'completed',
                    'original_session_id': original_session_id,
                    'events_replayed': len(events),
                    'replay_duration_seconds': replay_duration,
                    'speed_multiplier': speed_multiplier
                }
            )
            
        except Exception as e:
            logger.error(f"Replay stream execution failed: {e}")
            if connection_id in self.active_sessions:
                await self._send_stream_message(
                    connection_id,
                    StreamEventType.REPLAY_EVENT,
                    {
                        'status': 'error',
                        'message': str(e),
                        'original_session_id': original_session_id
                    }
                )
    
    async def _quick_pattern_analysis(self, event: ConversationEvent) -> List[str]:
        """Perform quick pattern analysis for streaming."""
        patterns = []
        
        # Simple pattern detection for real-time streaming
        if event.response_time_ms and event.response_time_ms > 5000:
            patterns.append("slow_response")
        
        if event.event_type == ConversationEventType.ERROR_OCCURRED:
            patterns.append("error_event")
        
        if len(event.tool_calls) > 3:
            patterns.append("tool_heavy")
        
        if len(event.context_references) > 5:
            patterns.append("context_heavy")
        
        return patterns
    
    def _next_sequence_id(self) -> int:
        """Get next sequence ID for stream messages."""
        self.sequence_counter += 1
        return self.sequence_counter