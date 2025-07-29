"""
WebSocket API for Real-Time Dashboard Event Streaming - VS 6.2
LeanVibe Agent Hive 2.0

Provides real-time event streaming for Live Dashboard Integration with:
- <1s event latency for dashboard updates
- Event filtering and routing for semantic intelligence
- Connection management with 1000+ concurrent connections
- Integration with hook system and event collector
- Performance optimizations for dashboard requirements
"""

import asyncio
import json
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import redis.asyncio as redis

import structlog
from fastapi import WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from ...core.database import get_async_session
from ...core.redis import get_redis
from ...core.hook_lifecycle_system import get_hook_lifecycle_system, HookEvent
from ...services.event_collector_service import get_event_collector
from ...models.observability import AgentEvent, EventType
from ...schemas.observability import BaseObservabilityEvent

logger = structlog.get_logger()

# Create router
router = APIRouter(prefix="/observability", tags=["observability-websocket"])


class DashboardEventType(str, Enum):
    """Types of dashboard events for real-time streaming."""
    HOOK_EVENT = "hook_event"
    WORKFLOW_UPDATE = "workflow_update" 
    SEMANTIC_INTELLIGENCE = "semantic_intelligence"
    PERFORMANCE_METRIC = "performance_metric"
    AGENT_STATUS = "agent_status"
    SYSTEM_ALERT = "system_alert"
    CONTEXT_FLOW = "context_flow"
    INTELLIGENCE_KPI = "intelligence_kpi"


class SubscriptionFilter(BaseModel):
    """WebSocket subscription filters for targeted event streaming."""
    agent_ids: Optional[List[str]] = Field(default_factory=list, description="Filter by agent IDs")
    session_ids: Optional[List[str]] = Field(default_factory=list, description="Filter by session IDs")
    event_types: Optional[List[DashboardEventType]] = Field(default_factory=list, description="Filter by event types")
    semantic_concepts: Optional[List[str]] = Field(default_factory=list, description="Filter by semantic concepts")
    intelligence_metrics: Optional[List[str]] = Field(default_factory=list, description="Filter by intelligence KPIs")
    min_priority: Optional[int] = Field(default=1, description="Minimum event priority (1-10)")
    max_latency_ms: Optional[int] = Field(default=1000, description="Maximum acceptable event latency in ms")
    buffer_size: Optional[int] = Field(default=100, description="Client-side event buffer size")


class DashboardEvent(BaseModel):
    """Standardized dashboard event for real-time streaming."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: DashboardEventType
    source: str = Field(description="Event source identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    priority: int = Field(default=5, ge=1, le=10, description="Event priority (1=highest, 10=lowest)")
    
    # Event data
    data: Dict[str, Any] = Field(default_factory=dict)
    
    # Semantic intelligence metadata
    semantic_embedding: Optional[List[float]] = Field(default=None, description="Semantic vector embedding")
    semantic_concepts: Optional[List[str]] = Field(default_factory=list, description="Extracted semantic concepts")
    context_references: Optional[List[str]] = Field(default_factory=list, description="Referenced context IDs")
    
    # Performance metadata
    latency_ms: Optional[float] = Field(default=None, description="Event processing latency")
    correlation_id: Optional[str] = Field(default=None, description="Cross-system correlation ID")
    
    # Dashboard-specific metadata
    visualization_hint: Optional[str] = Field(default=None, description="Hint for dashboard visualization")
    requires_acknowledgment: bool = Field(default=False, description="Whether client should acknowledge receipt")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: str
        }


@dataclass
class WebSocketConnection:
    """WebSocket connection metadata and state management."""
    id: str
    websocket: WebSocket
    filters: SubscriptionFilter
    connected_at: datetime
    last_activity: datetime
    messages_sent: int = 0
    messages_received: int = 0
    buffer: List[DashboardEvent] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def add_to_buffer(self, event: DashboardEvent):
        """Add event to client buffer with size management."""
        self.buffer.append(event)
        if len(self.buffer) > self.filters.buffer_size:
            self.buffer.pop(0)  # Remove oldest event
    
    def should_receive_event(self, event: DashboardEvent) -> bool:
        """Check if this connection should receive the event based on filters."""
        # Agent ID filter
        if self.filters.agent_ids and event.data.get('agent_id'):
            if event.data['agent_id'] not in self.filters.agent_ids:
                return False
        
        # Session ID filter
        if self.filters.session_ids and event.data.get('session_id'):
            if event.data['session_id'] not in self.filters.session_ids:
                return False
        
        # Event type filter
        if self.filters.event_types and event.type not in self.filters.event_types:
            return False
        
        # Priority filter
        if event.priority < self.filters.min_priority:
            return False
        
        # Latency filter
        if self.filters.max_latency_ms and event.latency_ms:
            if event.latency_ms > self.filters.max_latency_ms:
                return False
        
        # Semantic concepts filter
        if self.filters.semantic_concepts and event.semantic_concepts:
            if not any(concept in self.filters.semantic_concepts for concept in event.semantic_concepts):
                return False
        
        return True


class DashboardWebSocketManager:
    """
    High-performance WebSocket manager for dashboard real-time streaming.
    
    Features:
    - <1s event latency guarantee
    - 1000+ concurrent connections support
    - Event filtering and routing
    - Performance optimization and monitoring
    - Integration with hook system and event collector
    """
    
    def __init__(self):
        self.connections: Dict[str, WebSocketConnection] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.hook_system = None
        self.event_collector = None
        
        # Performance tracking
        self.metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "events_streamed": 0,
            "average_latency_ms": 0.0,
            "events_filtered": 0,
            "reconnections": 0,
            "errors": 0
        }
        
        # Configuration
        self.config = {
            "max_connections": 1000,
            "heartbeat_interval": 30,
            "max_buffer_size": 1000,
            "event_batch_size": 50,
            "performance_threshold_ms": 1000,
            "cleanup_interval": 300  # 5 minutes
        }
        
        # Event processing state
        self._running = False
        self._background_tasks = set()
        
        logger.info("Dashboard WebSocket Manager initialized")
    
    async def initialize(self):
        """Initialize WebSocket manager with dependencies."""
        try:
            # Initialize Redis connection
            self.redis_client = get_redis()
            
            # Get hook system and event collector
            self.hook_system = await get_hook_lifecycle_system()
            self.event_collector = get_event_collector()
            
            # Start background tasks
            self._running = True
            self._start_background_tasks()
            
            logger.info("Dashboard WebSocket Manager ready for connections")
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket manager: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown WebSocket manager and cleanup resources."""
        self._running = False
        
        # Disconnect all clients
        for connection_id in list(self.connections.keys()):
            await self.disconnect_client(connection_id)
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        logger.info("Dashboard WebSocket Manager shutdown complete")
    
    async def connect_client(
        self, 
        websocket: WebSocket, 
        filters: SubscriptionFilter = None
    ) -> str:
        """Connect a new WebSocket client with optional filters."""
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        connection = WebSocketConnection(
            id=connection_id,
            websocket=websocket,
            filters=filters or SubscriptionFilter(),
            connected_at=now,
            last_activity=now
        )
        
        self.connections[connection_id] = connection
        self.metrics["total_connections"] += 1
        self.metrics["active_connections"] = len(self.connections)
        
        logger.info(
            "Dashboard client connected",
            connection_id=connection_id,
            active_connections=self.metrics["active_connections"],
            filters=filters.dict() if filters else None
        )
        
        # Send connection confirmation
        await self._send_to_connection(connection, DashboardEvent(
            type=DashboardEventType.SYSTEM_ALERT,
            source="websocket_manager",
            data={
                "message": "Connected to dashboard stream",
                "connection_id": connection_id,
                "server_time": datetime.utcnow().isoformat()
            },
            priority=1
        ))
        
        return connection_id
    
    async def disconnect_client(self, connection_id: str):
        """Disconnect a WebSocket client."""
        connection = self.connections.get(connection_id)
        if not connection:
            return
        
        try:
            await connection.websocket.close()
        except Exception:
            pass  # Connection may already be closed
        
        del self.connections[connection_id]
        self.metrics["active_connections"] = len(self.connections)
        
        logger.info(
            "Dashboard client disconnected",
            connection_id=connection_id,
            active_connections=self.metrics["active_connections"],
            session_duration_seconds=(datetime.utcnow() - connection.connected_at).total_seconds(),
            messages_sent=connection.messages_sent
        )
    
    async def update_client_filters(self, connection_id: str, filters: SubscriptionFilter):
        """Update filters for an existing client connection."""
        connection = self.connections.get(connection_id)
        if not connection:
            raise ValueError(f"Connection not found: {connection_id}")
        
        connection.filters = filters
        connection.update_activity()
        
        logger.info(
            "Client filters updated",
            connection_id=connection_id,
            new_filters=filters.dict()
        )
    
    async def broadcast_event(self, event: DashboardEvent):
        """Broadcast an event to all connected clients with filtering."""
        if not self.connections:
            return
        
        start_time = time.time()
        sent_count = 0
        filtered_count = 0
        
        # Process connections in batches for performance
        connection_items = list(self.connections.items())
        batch_size = self.config["event_batch_size"]
        
        for i in range(0, len(connection_items), batch_size):
            batch = connection_items[i:i + batch_size]
            
            # Create tasks for parallel processing
            tasks = []
            for connection_id, connection in batch:
                if connection.should_receive_event(event):
                    tasks.append(self._send_to_connection(connection, event))
                    sent_count += 1
                else:
                    filtered_count += 1
            
            # Execute batch in parallel
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle any errors
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        connection_id, _ = batch[i]
                        logger.error(
                            "Failed to send event to client",
                            connection_id=connection_id,
                            error=str(result)
                        )
                        # Schedule for disconnection
                        asyncio.create_task(self.disconnect_client(connection_id))
        
        # Update metrics
        latency_ms = (time.time() - start_time) * 1000
        self.metrics["events_streamed"] += sent_count
        self.metrics["events_filtered"] += filtered_count
        
        # Update average latency
        current_avg = self.metrics["average_latency_ms"]
        total_events = self.metrics["events_streamed"]
        if total_events > 0:
            self.metrics["average_latency_ms"] = (
                (current_avg * (total_events - sent_count) + latency_ms * sent_count) / total_events
            )
        
        # Performance monitoring
        if latency_ms > self.config["performance_threshold_ms"]:
            logger.warning(
                "High broadcast latency detected",
                latency_ms=latency_ms,
                sent_count=sent_count,
                filtered_count=filtered_count,
                active_connections=len(self.connections)
            )
        
        logger.debug(
            "Event broadcast completed",
            event_type=event.type,
            sent_count=sent_count,
            filtered_count=filtered_count,
            latency_ms=round(latency_ms, 2)
        )
    
    async def _send_to_connection(self, connection: WebSocketConnection, event: DashboardEvent):
        """Send an event to a specific connection."""
        try:
            # Add event to buffer
            connection.add_to_buffer(event)
            
            # Serialize and send
            event_data = event.dict()
            await connection.websocket.send_text(json.dumps(event_data))
            
            connection.messages_sent += 1
            connection.update_activity()
            
        except WebSocketDisconnect:
            # Handle graceful disconnect
            asyncio.create_task(self.disconnect_client(connection.id))
        except Exception as e:
            logger.error(
                "Failed to send event to connection",
                connection_id=connection.id,
                error=str(e)
            )
            self.metrics["errors"] += 1
            raise
    
    def _start_background_tasks(self):
        """Start background tasks for maintenance and monitoring."""
        # Heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_task())
        self._background_tasks.add(heartbeat_task)
        
        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_task())
        self._background_tasks.add(cleanup_task)
        
        # Redis event listener task
        redis_task = asyncio.create_task(self._redis_event_listener())
        self._background_tasks.add(redis_task)
    
    async def _heartbeat_task(self):
        """Background task for client heartbeat monitoring."""
        while self._running:
            try:
                now = datetime.utcnow()
                stale_connections = []
                
                for connection_id, connection in self.connections.items():
                    # Check for stale connections
                    if (now - connection.last_activity).total_seconds() > self.config["heartbeat_interval"] * 2:
                        stale_connections.append(connection_id)
                    else:
                        # Send heartbeat
                        heartbeat_event = DashboardEvent(
                            type=DashboardEventType.SYSTEM_ALERT,
                            source="heartbeat",
                            data={"ping": True, "server_time": now.isoformat()},
                            priority=10  # Lowest priority
                        )
                        await self._send_to_connection(connection, heartbeat_event)
                
                # Disconnect stale connections
                for connection_id in stale_connections:
                    logger.info(f"Disconnecting stale connection: {connection_id}")
                    await self.disconnect_client(connection_id)
                
                await asyncio.sleep(self.config["heartbeat_interval"])
                
            except Exception as e:
                logger.error(f"Heartbeat task error: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_task(self):
        """Background task for periodic cleanup and maintenance."""
        while self._running:
            try:
                # Clean up connection buffers
                for connection in self.connections.values():
                    if len(connection.buffer) > self.config["max_buffer_size"]:
                        # Keep only recent events
                        connection.buffer = connection.buffer[-self.config["max_buffer_size"]:]
                
                # Log performance metrics
                logger.info(
                    "Dashboard WebSocket metrics",
                    **self.metrics
                )
                
                await asyncio.sleep(self.config["cleanup_interval"])
                
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(30)
    
    async def _redis_event_listener(self):
        """Background task to listen for events from Redis streams."""
        while self._running:
            try:
                if not self.redis_client:
                    await asyncio.sleep(5)
                    continue
                
                # Listen for hook lifecycle events
                streams = {
                    "hook_lifecycle_events": "$",
                    "hook_security_events": "$",
                    "hook_performance_events": "$"
                }
                
                messages = await self.redis_client.xread(
                    streams=streams,
                    count=10,
                    block=1000  # 1 second timeout
                )
                
                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        try:
                            # Convert Redis event to dashboard event
                            dashboard_event = await self._convert_redis_event_to_dashboard_event(
                                stream.decode(), fields
                            )
                            
                            if dashboard_event:
                                await self.broadcast_event(dashboard_event)
                                
                        except Exception as e:
                            logger.error(f"Failed to process Redis event: {e}")
                
            except Exception as e:
                logger.error(f"Redis event listener error: {e}")
                await asyncio.sleep(5)
    
    async def _convert_redis_event_to_dashboard_event(
        self, 
        stream_name: str, 
        fields: Dict[bytes, bytes]
    ) -> Optional[DashboardEvent]:
        """Convert Redis stream event to dashboard event."""
        try:
            # Decode fields
            decoded_fields = {
                k.decode() if isinstance(k, bytes) else k: 
                v.decode() if isinstance(v, bytes) else v 
                for k, v in fields.items()
            }
            
            # Determine event type based on stream
            if "security" in stream_name:
                event_type = DashboardEventType.SYSTEM_ALERT
            elif "performance" in stream_name:
                event_type = DashboardEventType.PERFORMANCE_METRIC
            else:
                event_type = DashboardEventType.HOOK_EVENT
            
            # Create dashboard event
            dashboard_event = DashboardEvent(
                type=event_type,
                source=f"redis_{stream_name}",
                data=decoded_fields,
                priority=3,  # High priority for real-time events
                correlation_id=decoded_fields.get("correlation_id")
            )
            
            return dashboard_event
            
        except Exception as e:
            logger.error(f"Failed to convert Redis event: {e}")
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current WebSocket manager metrics."""
        return {
            **self.metrics,
            "config": self.config,
            "connections": {
                connection_id: {
                    "connected_at": connection.connected_at.isoformat(),
                    "last_activity": connection.last_activity.isoformat(),
                    "messages_sent": connection.messages_sent,
                    "messages_received": connection.messages_received,
                    "buffer_size": len(connection.buffer),
                    "filters": connection.filters.dict()
                }
                for connection_id, connection in self.connections.items()
            }
        }


# Global manager instance
_websocket_manager: Optional[DashboardWebSocketManager] = None


async def get_websocket_manager() -> DashboardWebSocketManager:
    """Get global WebSocket manager instance."""
    global _websocket_manager
    
    if _websocket_manager is None:
        _websocket_manager = DashboardWebSocketManager()
        await _websocket_manager.initialize()
    
    return _websocket_manager


# WebSocket endpoints

@router.websocket("/dashboard/stream")
async def dashboard_websocket(
    websocket: WebSocket,
    agent_ids: Optional[str] = Query(default=None, description="Comma-separated agent IDs to filter"),
    session_ids: Optional[str] = Query(default=None, description="Comma-separated session IDs to filter"),
    event_types: Optional[str] = Query(default=None, description="Comma-separated event types to filter"),
    min_priority: Optional[int] = Query(default=1, description="Minimum event priority"),
    max_latency_ms: Optional[int] = Query(default=1000, description="Maximum acceptable latency")
):
    """
    Main WebSocket endpoint for dashboard real-time event streaming.
    
    Supports filtering by:
    - Agent IDs
    - Session IDs  
    - Event types
    - Priority levels
    - Latency requirements
    """
    manager = await get_websocket_manager()
    
    # Parse filters from query parameters
    filters = SubscriptionFilter(
        agent_ids=agent_ids.split(",") if agent_ids else [],
        session_ids=session_ids.split(",") if session_ids else [],
        event_types=[DashboardEventType(t.strip()) for t in event_types.split(",") if t.strip()] if event_types else [],
        min_priority=min_priority,
        max_latency_ms=max_latency_ms
    )
    
    connection_id = await manager.connect_client(websocket, filters)
    
    try:
        while True:
            # Listen for client messages (filter updates, acknowledgments)
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "update_filters":
                    new_filters = SubscriptionFilter(**data.get("filters", {}))
                    await manager.update_client_filters(connection_id, new_filters)
                    
                elif data.get("type") == "ping":
                    # Respond to client ping
                    pong_event = DashboardEvent(
                        type=DashboardEventType.SYSTEM_ALERT,
                        source="websocket_manager",
                        data={"pong": True, "client_id": connection_id},
                        priority=10
                    )
                    connection = manager.connections.get(connection_id)
                    if connection:
                        await manager._send_to_connection(connection, pong_event)
                        connection.messages_received += 1
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from client {connection_id}")
            except Exception as e:
                logger.error(f"Error processing client message: {e}")
    
    finally:
        await manager.disconnect_client(connection_id)


@router.get("/dashboard/connections")
async def get_dashboard_connections():
    """Get information about active dashboard WebSocket connections."""
    try:
        manager = await get_websocket_manager()
        return {
            "active_connections": len(manager.connections),
            "metrics": manager.get_metrics()
        }
    except Exception as e:
        logger.error(f"Failed to get connection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dashboard/broadcast")
async def broadcast_test_event(event: DashboardEvent):
    """Broadcast a test event to all connected dashboard clients."""
    try:
        manager = await get_websocket_manager()
        await manager.broadcast_event(event)
        
        return {
            "success": True,
            "message": "Event broadcasted successfully",
            "active_connections": len(manager.connections)
        }
    except Exception as e:
        logger.error(f"Failed to broadcast event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Event creation utilities

async def create_workflow_update_event(
    workflow_id: str,
    agent_updates: List[Dict[str, Any]],
    semantic_flow: Optional[Dict[str, Any]] = None
) -> DashboardEvent:
    """Create a workflow update event for live constellation view."""
    return DashboardEvent(
        type=DashboardEventType.WORKFLOW_UPDATE,
        source=f"workflow_{workflow_id}",
        data={
            "workflow_id": workflow_id,
            "agent_updates": agent_updates,
            "semantic_flow": semantic_flow,
            "update_type": "agent_interaction"
        },
        priority=2,
        visualization_hint="update_graph_nodes"
    )


async def create_semantic_intelligence_event(
    concept: str,
    embedding: List[float],
    context_references: List[str],
    intelligence_metrics: Dict[str, float]
) -> DashboardEvent:
    """Create a semantic intelligence event for context trajectory view."""
    return DashboardEvent(
        type=DashboardEventType.SEMANTIC_INTELLIGENCE,
        source="semantic_memory",
        data={
            "concept": concept,
            "intelligence_metrics": intelligence_metrics,
            "context_lineage": context_references
        },
        semantic_embedding=embedding,
        semantic_concepts=[concept],
        context_references=context_references,
        priority=3,
        visualization_hint="update_context_trajectory"
    )


async def create_intelligence_kpi_event(
    kpi_name: str,
    current_value: float,
    trend_data: List[Dict[str, Any]],
    threshold_status: str
) -> DashboardEvent:
    """Create an intelligence KPI event for real-time metrics dashboard."""
    return DashboardEvent(
        type=DashboardEventType.INTELLIGENCE_KPI,
        source="kpi_monitor",
        data={
            "kpi_name": kpi_name,
            "current_value": current_value,
            "trend_data": trend_data,
            "threshold_status": threshold_status,
            "update_timestamp": datetime.utcnow().isoformat()
        },
        priority=4,
        visualization_hint="update_kpi_chart"
    )


# Convenience functions for integration

async def broadcast_hook_event(hook_event: HookEvent):
    """Broadcast a hook lifecycle event to dashboard clients."""
    try:
        manager = await get_websocket_manager()
        
        # Convert hook event to dashboard event
        dashboard_event = DashboardEvent(
            type=DashboardEventType.HOOK_EVENT,
            source=f"hook_{hook_event.hook_type.value}",
            data={
                "hook_type": hook_event.hook_type.value,
                "agent_id": str(hook_event.agent_id),
                "session_id": str(hook_event.session_id) if hook_event.session_id else None,
                "payload": hook_event.payload,
                "correlation_id": hook_event.correlation_id
            },
            priority=hook_event.priority,
            correlation_id=hook_event.correlation_id,
            latency_ms=0.5  # Estimate for hook events
        )
        
        await manager.broadcast_event(dashboard_event)
        
    except Exception as e:
        logger.error(f"Failed to broadcast hook event: {e}")


async def broadcast_performance_metric(
    metric_name: str,
    value: float,
    source: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Broadcast a performance metric to dashboard clients."""
    try:
        manager = await get_websocket_manager()
        
        dashboard_event = DashboardEvent(
            type=DashboardEventType.PERFORMANCE_METRIC,
            source=source,
            data={
                "metric_name": metric_name,
                "value": value,
                "metadata": metadata or {},
                "timestamp": datetime.utcnow().isoformat()
            },
            priority=5,
            visualization_hint="update_performance_chart"
        )
        
        await manager.broadcast_event(dashboard_event)
        
    except Exception as e:
        logger.error(f"Failed to broadcast performance metric: {e}")