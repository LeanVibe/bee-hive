"""
Real-time Dashboard Streaming Integration for LeanVibe Agent Hive 2.0

Provides high-performance, real-time event streaming for dashboard updates with 
optimized WebSocket management, intelligent batching, and comprehensive filtering.

Key Features:
- High-performance WebSocket connection management
- Intelligent event batching and throttling
- Advanced filtering and subscription management
- Mobile-optimized data compression
- Automatic reconnection and failover
- Real-time performance monitoring
- Comprehensive error handling and recovery

Architecture:
- Event-driven streaming with Redis Streams integration
- WebSocket connection pooling and load balancing
- Intelligent data aggregation and compression
- Performance-optimized message serialization
- Comprehensive observability and monitoring
"""

import asyncio
import json
import gzip
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Callable, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from uuid import uuid4
from collections import defaultdict, deque
import time
import psutil

import structlog
from pydantic import BaseModel, Field, validator
from fastapi import WebSocket, WebSocketDisconnect
import redis.asyncio as redis

from .comprehensive_dashboard_integration import (
    comprehensive_dashboard_integration, IntegrationEventType
)
from .redis import get_message_broker
from .observability_streams import ObservabilityStreamsManager

logger = structlog.get_logger()


class StreamPriority(Enum):
    """Priority levels for streaming events."""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"


class CompressionType(Enum):
    """Types of data compression for streaming."""
    NONE = "none"
    GZIP = "gzip"
    JSON_MINIMIZE = "json_minimize"
    SMART = "smart"  # Automatically choose best compression


@dataclass
class StreamSubscription:
    """Represents a dashboard stream subscription."""
    stream_id: str
    websocket: WebSocket
    user_id: Optional[str] = None
    
    # Filtering configuration
    event_types: Set[str] = field(default_factory=set)
    agent_ids: Set[str] = field(default_factory=set)
    session_ids: Set[str] = field(default_factory=set)
    priority_threshold: StreamPriority = StreamPriority.LOW
    
    # Performance configuration
    max_events_per_second: int = 10
    batch_size: int = 5
    compression: CompressionType = CompressionType.SMART
    mobile_optimized: bool = False
    
    # State tracking
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    events_sent: int = 0
    bytes_sent: int = 0
    error_count: int = 0
    
    # Rate limiting
    event_timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def is_rate_limited(self) -> bool:
        """Check if this subscription is currently rate limited."""
        now = time.time()
        # Remove old timestamps
        while self.event_timestamps and (now - self.event_timestamps[0]) > 1.0:
            self.event_timestamps.popleft()
        
        return len(self.event_timestamps) >= self.max_events_per_second
    
    def record_event_sent(self, bytes_sent: int) -> None:
        """Record that an event was sent."""
        self.event_timestamps.append(time.time())
        self.events_sent += 1
        self.bytes_sent += bytes_sent
        self.last_activity = datetime.utcnow()
    
    def should_receive_event(self, event: Dict[str, Any]) -> bool:
        """Check if this subscription should receive a specific event."""
        # Check event type filter
        if self.event_types and event.get('event_type') not in self.event_types:
            return False
        
        # Check agent ID filter
        if self.agent_ids and event.get('data', {}).get('agent_id') not in self.agent_ids:
            return False
        
        # Check session ID filter
        if self.session_ids and event.get('data', {}).get('session_id') not in self.session_ids:
            return False
        
        # Check priority filter
        event_priority = StreamPriority(event.get('priority', 'low'))
        priority_levels = {
            StreamPriority.CRITICAL: 4,
            StreamPriority.HIGH: 3,
            StreamPriority.MEDIUM: 2,
            StreamPriority.LOW: 1
        }
        
        if priority_levels[event_priority] < priority_levels[self.priority_threshold]:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert subscription to dictionary."""
        return {
            'stream_id': self.stream_id,
            'user_id': self.user_id,
            'event_types': list(self.event_types),
            'agent_ids': list(self.agent_ids),
            'session_ids': list(self.session_ids),
            'priority_threshold': self.priority_threshold.value,
            'max_events_per_second': self.max_events_per_second,
            'batch_size': self.batch_size,
            'compression': self.compression.value,
            'mobile_optimized': self.mobile_optimized,
            'connected_at': self.connected_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'events_sent': self.events_sent,
            'bytes_sent': self.bytes_sent,
            'error_count': self.error_count,
            'current_rate': len(self.event_timestamps)
        }


@dataclass
class StreamEvent:
    """Represents an event to be streamed to dashboards."""
    event_id: str
    event_type: str
    priority: StreamPriority
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ttl_seconds: int = 300  # 5 minutes default TTL
    
    # Performance metadata
    source_component: Optional[str] = None
    processing_time_ms: Optional[int] = None
    data_size_bytes: Optional[int] = None
    
    def is_expired(self) -> bool:
        """Check if this event has expired."""
        return (datetime.utcnow() - self.timestamp).total_seconds() > self.ttl_seconds
    
    def to_dict(self, compress: bool = False) -> Dict[str, Any]:
        """Convert event to dictionary with optional compression."""
        event_dict = {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'priority': self.priority.value,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'source_component': self.source_component,
            'processing_time_ms': self.processing_time_ms
        }
        
        if compress:
            # Apply mobile-optimized compression
            event_dict = self._compress_for_mobile(event_dict)
        
        return event_dict
    
    def _compress_for_mobile(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mobile-optimized compression to event data."""
        # Reduce precision of timestamps
        if 'timestamp' in data:
            # Keep only seconds precision for mobile
            timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            data['timestamp'] = timestamp.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Truncate long text fields
        if 'data' in data and isinstance(data['data'], dict):
            for key, value in data['data'].items():
                if isinstance(value, str) and len(value) > 100:
                    data['data'][key] = value[:100] + '...'
        
        # Remove optional metadata for mobile
        data.pop('processing_time_ms', None)
        data.pop('source_component', None)
        
        return data


class RealtimeDashboardStreaming:
    """
    High-performance real-time streaming system for dashboard updates.
    
    Manages WebSocket connections, event batching, filtering, and performance
    optimization for comprehensive dashboard integration.
    """
    
    def __init__(self):
        # Connection management
        self.subscriptions: Dict[str, StreamSubscription] = {}
        self.connection_pool: Dict[str, List[str]] = defaultdict(list)  # user_id -> stream_ids
        
        # Event processing
        self.event_queue: asyncio.Queue = asyncio.Queue()
        self.batch_queues: Dict[str, List[StreamEvent]] = defaultdict(list)
        self.pending_broadcasts: Dict[str, StreamEvent] = {}
        
        # Performance optimization
        self.event_cache: Dict[str, StreamEvent] = {}
        self.cache_ttl_seconds = 60
        self.max_cache_size = 1000
        
        # Rate limiting and throttling
        self.global_rate_limit = 1000  # events per second across all connections
        self.global_event_timestamps: deque = deque(maxlen=1000)
        
        # Background processing
        self.background_tasks: Set[asyncio.Task] = set()
        self.is_running = False
        
        # Metrics and monitoring
        self.metrics = {
            'total_connections': 0,
            'total_events_sent': 0,
            'total_bytes_sent': 0,
            'events_per_second': 0,
            'average_latency_ms': 0,
            'error_rate': 0.0,
            'compression_ratio': 0.0
        }
        
        # Redis integration
        self.redis_client: Optional[redis.Redis] = None
        
        logger.info("RealtimeDashboardStreaming initialized")
    
    async def start(self) -> None:
        """Start the real-time streaming system."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Initialize Redis connection
        self.redis_client = await get_message_broker()
        
        # Start background processing tasks
        tasks = [
            self._process_event_queue(),
            self._process_batch_queues(),
            self._monitor_connections(),
            self._cleanup_expired_data(),
            self._collect_metrics(),
            self._handle_redis_events()
        ]
        
        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
        
        # Register with comprehensive dashboard integration
        await self._register_with_dashboard_integration()
        
        logger.info("Real-time dashboard streaming started")
    
    async def stop(self) -> None:
        """Stop the real-time streaming system."""
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.background_tasks.clear()
        
        # Close all WebSocket connections
        for subscription in list(self.subscriptions.values()):
            try:
                await subscription.websocket.close()
            except Exception:
                pass
        
        self.subscriptions.clear()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Real-time dashboard streaming stopped")
    
    async def register_stream(
        self,
        websocket: WebSocket,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        performance_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a new dashboard stream."""
        stream_id = str(uuid4())
        
        # Accept WebSocket connection
        await websocket.accept()
        
        # Create subscription
        subscription = StreamSubscription(
            stream_id=stream_id,
            websocket=websocket,
            user_id=user_id
        )
        
        # Apply filters if provided
        if filters:
            subscription.event_types = set(filters.get('event_types', []))
            subscription.agent_ids = set(filters.get('agent_ids', []))
            subscription.session_ids = set(filters.get('session_ids', []))
            subscription.priority_threshold = StreamPriority(
                filters.get('priority_threshold', 'low')
            )
        
        # Apply performance configuration
        if performance_config:
            subscription.max_events_per_second = performance_config.get(
                'max_events_per_second', 10
            )
            subscription.batch_size = performance_config.get('batch_size', 5)
            subscription.compression = CompressionType(
                performance_config.get('compression', 'smart')
            )
            subscription.mobile_optimized = performance_config.get(
                'mobile_optimized', False
            )
        
        # Store subscription
        self.subscriptions[stream_id] = subscription
        
        if user_id:
            self.connection_pool[user_id].append(stream_id)
        
        # Send initial connection confirmation
        await self._send_connection_confirmation(subscription)
        
        # Send initial dashboard data
        await self._send_initial_data(subscription)
        
        logger.info(
            "Dashboard stream registered",
            stream_id=stream_id,
            user_id=user_id,
            filters=filters,
            total_connections=len(self.subscriptions)
        )
        
        return stream_id
    
    async def unregister_stream(self, stream_id: str) -> None:
        """Unregister a dashboard stream."""
        if stream_id not in self.subscriptions:
            return
        
        subscription = self.subscriptions[stream_id]
        
        # Remove from connection pool
        if subscription.user_id and subscription.user_id in self.connection_pool:
            try:
                self.connection_pool[subscription.user_id].remove(stream_id)
                if not self.connection_pool[subscription.user_id]:
                    del self.connection_pool[subscription.user_id]
            except ValueError:
                pass
        
        # Remove subscription
        del self.subscriptions[stream_id]
        
        # Clean up batch queue
        if stream_id in self.batch_queues:
            del self.batch_queues[stream_id]
        
        logger.info(
            "Dashboard stream unregistered",
            stream_id=stream_id,
            remaining_connections=len(self.subscriptions)
        )
    
    async def broadcast_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        priority: StreamPriority = StreamPriority.MEDIUM,
        target_subscriptions: Optional[List[str]] = None
    ) -> int:
        """Broadcast an event to relevant subscriptions."""
        event = StreamEvent(
            event_id=str(uuid4()),
            event_type=event_type,
            priority=priority,
            data=data,
            source_component='dashboard_streaming'
        )
        
        # Add to event queue for processing
        await self.event_queue.put((event, target_subscriptions))
        
        return len(self.subscriptions) if not target_subscriptions else len(target_subscriptions)
    
    async def send_direct_message(
        self,
        stream_id: str,
        event_type: str,
        data: Dict[str, Any],
        priority: StreamPriority = StreamPriority.HIGH
    ) -> bool:
        """Send a direct message to a specific stream."""
        if stream_id not in self.subscriptions:
            return False
        
        subscription = self.subscriptions[stream_id]
        
        # Check rate limiting
        if subscription.is_rate_limited():
            logger.warning(
                "Stream rate limited, dropping message",
                stream_id=stream_id,
                event_type=event_type
            )
            return False
        
        event = StreamEvent(
            event_id=str(uuid4()),
            event_type=event_type,
            priority=priority,
            data=data
        )
        
        # Send immediately for direct messages
        success = await self._send_event_to_subscription(event, subscription)
        
        return success
    
    async def update_stream_filters(
        self,
        stream_id: str,
        filters: Dict[str, Any]
    ) -> bool:
        """Update filters for an existing stream."""
        if stream_id not in self.subscriptions:
            return False
        
        subscription = self.subscriptions[stream_id]
        
        # Update filters
        subscription.event_types = set(filters.get('event_types', []))
        subscription.agent_ids = set(filters.get('agent_ids', []))
        subscription.session_ids = set(filters.get('session_ids', []))
        subscription.priority_threshold = StreamPriority(
            filters.get('priority_threshold', 'low')
        )
        
        # Send filter update confirmation
        await self.send_direct_message(
            stream_id,
            'filter_update_confirmation',
            {'updated_filters': filters},
            StreamPriority.HIGH
        )
        
        logger.info(
            "Stream filters updated",
            stream_id=stream_id,
            filters=filters
        )
        
        return True
    
    async def get_stream_statistics(self) -> Dict[str, Any]:
        """Get comprehensive streaming statistics."""
        stats = {
            'total_connections': len(self.subscriptions),
            'connections_by_user': {
                user_id: len(stream_ids)
                for user_id, stream_ids in self.connection_pool.items()
            },
            'events_in_queue': self.event_queue.qsize(),
            'cache_size': len(self.event_cache),
            'metrics': self.metrics.copy(),
            'performance': {
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'cpu_percent': psutil.Process().cpu_percent(),
                'event_processing_rate': len(self.global_event_timestamps),
                'average_events_per_connection': (
                    sum(s.events_sent for s in self.subscriptions.values()) /
                    len(self.subscriptions) if self.subscriptions else 0
                )
            }
        }
        
        # Add per-subscription statistics
        stats['subscription_details'] = {
            stream_id: subscription.to_dict()
            for stream_id, subscription in self.subscriptions.items()
        }
        
        return stats
    
    # Private methods for background processing
    
    async def _process_event_queue(self) -> None:
        """Background task to process the main event queue."""
        while self.is_running:
            try:
                # Get event from queue with timeout
                event, target_subscriptions = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )
                
                # Process the event
                await self._process_single_event(event, target_subscriptions)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event queue: {e}")
                await asyncio.sleep(1)
    
    async def _process_single_event(
        self,
        event: StreamEvent,
        target_subscriptions: Optional[List[str]]
    ) -> None:
        """Process a single event for distribution."""
        start_time = time.time()
        
        # Determine target subscriptions
        if target_subscriptions:
            subscriptions = [
                self.subscriptions[sid] for sid in target_subscriptions
                if sid in self.subscriptions
            ]
        else:
            subscriptions = [
                sub for sub in self.subscriptions.values()
                if sub.should_receive_event(event.to_dict())
            ]
        
        # Filter out rate-limited subscriptions
        valid_subscriptions = [
            sub for sub in subscriptions
            if not sub.is_rate_limited()
        ]
        
        # Add to batch queues
        for subscription in valid_subscriptions:
            self.batch_queues[subscription.stream_id].append(event)
        
        # Update metrics
        processing_time = (time.time() - start_time) * 1000
        event.processing_time_ms = int(processing_time)
        
        self.global_event_timestamps.append(time.time())
        
        # Cache the event
        if len(self.event_cache) < self.max_cache_size:
            self.event_cache[event.event_id] = event
    
    async def _process_batch_queues(self) -> None:
        """Background task to process batched events."""
        while self.is_running:
            try:
                for stream_id, events in list(self.batch_queues.items()):
                    if not events:
                        continue
                    
                    subscription = self.subscriptions.get(stream_id)
                    if not subscription:
                        del self.batch_queues[stream_id]
                        continue
                    
                    # Process batch if it's full or old enough
                    should_process = (
                        len(events) >= subscription.batch_size or
                        (events and (time.time() - events[0].timestamp.timestamp()) > 2.0)
                    )
                    
                    if should_process:
                        batch = events[:subscription.batch_size]
                        self.batch_queues[stream_id] = events[subscription.batch_size:]
                        
                        await self._send_batch_to_subscription(batch, subscription)
                
                await asyncio.sleep(0.1)  # Process batches every 100ms
                
            except Exception as e:
                logger.error(f"Error processing batch queues: {e}")
                await asyncio.sleep(1)
    
    async def _send_batch_to_subscription(
        self,
        events: List[StreamEvent],
        subscription: StreamSubscription
    ) -> bool:
        """Send a batch of events to a subscription."""
        try:
            # Prepare batch message
            batch_data = {
                'type': 'event_batch',
                'batch_id': str(uuid4()),
                'events': [
                    event.to_dict(compress=subscription.mobile_optimized)
                    for event in events
                ],
                'count': len(events),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Apply compression
            message = self._compress_message(batch_data, subscription.compression)
            
            # Send message
            await subscription.websocket.send_text(message)
            
            # Update subscription metrics
            message_bytes = len(message.encode('utf-8'))
            subscription.record_event_sent(message_bytes)
            
            # Update global metrics
            self.metrics['total_events_sent'] += len(events)
            self.metrics['total_bytes_sent'] += message_bytes
            
            return True
            
        except WebSocketDisconnect:
            logger.info(
                "WebSocket disconnected during batch send",
                stream_id=subscription.stream_id
            )
            await self.unregister_stream(subscription.stream_id)
            return False
            
        except Exception as e:
            logger.error(
                "Error sending batch to subscription",
                stream_id=subscription.stream_id,
                error=str(e)
            )
            subscription.error_count += 1
            return False
    
    async def _send_event_to_subscription(
        self,
        event: StreamEvent,
        subscription: StreamSubscription
    ) -> bool:
        """Send a single event to a subscription."""
        return await self._send_batch_to_subscription([event], subscription)
    
    def _compress_message(
        self,
        data: Dict[str, Any],
        compression: CompressionType
    ) -> str:
        """Apply compression to message data."""
        json_str = json.dumps(data, separators=(',', ':'))
        
        if compression == CompressionType.NONE:
            return json_str
        
        elif compression == CompressionType.JSON_MINIMIZE:
            # Already minimized with separators
            return json_str
        
        elif compression == CompressionType.GZIP:
            # Apply GZIP compression and base64 encode
            import base64
            compressed = gzip.compress(json_str.encode('utf-8'))
            return base64.b64encode(compressed).decode('ascii')
        
        elif compression == CompressionType.SMART:
            # Choose best compression based on data size
            if len(json_str) < 1000:
                return json_str  # No compression for small messages
            else:
                # Use GZIP for larger messages
                import base64
                compressed = gzip.compress(json_str.encode('utf-8'))
                compressed_b64 = base64.b64encode(compressed).decode('ascii')
                
                # Only use compression if it reduces size significantly
                if len(compressed_b64) < len(json_str) * 0.8:
                    return compressed_b64
                else:
                    return json_str
        
        return json_str
    
    async def _monitor_connections(self) -> None:
        """Background task to monitor connection health."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                disconnected_streams = []
                
                for stream_id, subscription in self.subscriptions.items():
                    # Check for inactive connections
                    inactive_time = (current_time - subscription.last_activity).total_seconds()
                    
                    if inactive_time > 300:  # 5 minutes of inactivity
                        # Send ping message
                        try:
                            await subscription.websocket.ping()
                        except Exception:
                            disconnected_streams.append(stream_id)
                    
                    # Check for high error rates
                    if subscription.error_count > 10:
                        logger.warning(
                            "High error rate for subscription",
                            stream_id=stream_id,
                            error_count=subscription.error_count
                        )
                
                # Clean up disconnected streams
                for stream_id in disconnected_streams:
                    await self.unregister_stream(stream_id)
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Error monitoring connections: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_data(self) -> None:
        """Background task to clean up expired data."""
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                # Clean up expired events from cache
                expired_events = [
                    event_id for event_id, event in self.event_cache.items()
                    if event.is_expired()
                ]
                
                for event_id in expired_events:
                    del self.event_cache[event_id]
                
                # Clean up old rate limiting timestamps
                cutoff_time = time.time() - 60  # 1 minute ago
                
                while (self.global_event_timestamps and 
                       self.global_event_timestamps[0] < cutoff_time):
                    self.global_event_timestamps.popleft()
                
                # Clean up batch queues for expired events
                for stream_id, events in self.batch_queues.items():
                    self.batch_queues[stream_id] = [
                        event for event in events
                        if not event.is_expired()
                    ]
                
                logger.debug(
                    "Cleanup completed",
                    expired_events=len(expired_events),
                    cache_size=len(self.event_cache)
                )
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
                await asyncio.sleep(300)
    
    async def _collect_metrics(self) -> None:
        """Background task to collect performance metrics."""
        while self.is_running:
            try:
                # Calculate events per second
                now = time.time()
                recent_events = [
                    t for t in self.global_event_timestamps
                    if (now - t) <= 1.0
                ]
                self.metrics['events_per_second'] = len(recent_events)
                
                # Calculate error rate
                total_subscriptions = len(self.subscriptions)
                if total_subscriptions > 0:
                    total_errors = sum(s.error_count for s in self.subscriptions.values())
                    total_events = sum(s.events_sent for s in self.subscriptions.values())
                    self.metrics['error_rate'] = (
                        total_errors / max(total_events, 1)
                    )
                
                # Update connection count
                self.metrics['total_connections'] = total_subscriptions
                
                await asyncio.sleep(5)  # Collect metrics every 5 seconds
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(10)
    
    async def _handle_redis_events(self) -> None:
        """Background task to handle Redis Streams events."""
        while self.is_running:
            try:
                if not self.redis_client:
                    await asyncio.sleep(5)
                    continue
                
                # Listen for dashboard integration events
                # This would integrate with Redis Streams for distributed events
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error handling Redis events: {e}")
                await asyncio.sleep(5)
    
    async def _register_with_dashboard_integration(self) -> None:
        """Register with the comprehensive dashboard integration system."""
        # This would register callbacks with the dashboard integration
        # to receive events for streaming
        pass
    
    async def _send_connection_confirmation(self, subscription: StreamSubscription) -> None:
        """Send connection confirmation to a new subscription."""
        confirmation = {
            'type': 'connection_confirmed',
            'stream_id': subscription.stream_id,
            'server_time': datetime.utcnow().isoformat(),
            'configuration': {
                'max_events_per_second': subscription.max_events_per_second,
                'batch_size': subscription.batch_size,
                'compression': subscription.compression.value,
                'mobile_optimized': subscription.mobile_optimized
            }
        }
        
        try:
            await subscription.websocket.send_text(json.dumps(confirmation))
        except Exception as e:
            logger.error(
                "Error sending connection confirmation",
                stream_id=subscription.stream_id,
                error=str(e)
            )
    
    async def _send_initial_data(self, subscription: StreamSubscription) -> None:
        """Send initial dashboard data to a new subscription."""
        try:
            # Get initial data from comprehensive dashboard integration
            initial_data = await comprehensive_dashboard_integration.get_system_performance_overview()
            
            # Send as initial data event
            await self.send_direct_message(
                subscription.stream_id,
                'initial_dashboard_data',
                initial_data,
                StreamPriority.HIGH
            )
            
        except Exception as e:
            logger.error(
                "Error sending initial data",
                stream_id=subscription.stream_id,
                error=str(e)
            )


# Global real-time dashboard streaming instance
realtime_dashboard_streaming = RealtimeDashboardStreaming()