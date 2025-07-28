"""
WebSocket endpoints for real-time observability and communication.

Provides real-time streaming of events, metrics, and system status
for the observability dashboard and monitoring systems.

Enhanced for Vertical Slice 1.2: Real-Time Monitoring Dashboard
- Agent lifecycle event streaming
- Performance metrics broadcasting
- System health monitoring
- Hook event forwarding
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Set, Optional, Any, List
from weakref import WeakSet

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel, ValidationError

from app.core.event_processor import get_event_processor
from app.core.redis import get_redis
from app.models.observability import EventType

logger = structlog.get_logger()

router = APIRouter()

# Connection management
class ConnectionManager:
    """Manages WebSocket connections for real-time streaming."""
    
    def __init__(self):
        # Use WeakSet to automatically handle garbage collection
        self.observability_connections: WeakSet[WebSocket] = WeakSet()
        self.agent_connections: Dict[str, WeakSet[WebSocket]] = {}
        self.session_connections: Dict[str, WeakSet[WebSocket]] = {}
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        
    async def connect_observability(self, websocket: WebSocket, metadata: Dict[str, Any] = None):
        """Add connection to observability stream."""
        await websocket.accept()
        self.observability_connections.add(websocket)
        self.connection_metadata[websocket] = {
            "type": "observability",
            "connected_at": datetime.utcnow(),
            "last_ping": datetime.utcnow(),
            **(metadata or {})
        }
        
        # Send welcome message
        await self._send_message(websocket, {
            "type": "connection",
            "status": "connected",
            "message": "Connected to observability stream",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info("🔌 WebSocket connected to observability stream", 
                   total_connections=len(self.observability_connections))
    
    async def connect_agent(self, websocket: WebSocket, agent_id: str):
        """Add connection to specific agent stream."""
        await websocket.accept()
        
        if agent_id not in self.agent_connections:
            self.agent_connections[agent_id] = WeakSet()
        
        self.agent_connections[agent_id].add(websocket)
        self.connection_metadata[websocket] = {
            "type": "agent",
            "agent_id": agent_id,
            "connected_at": datetime.utcnow(),
            "last_ping": datetime.utcnow()
        }
        
        await self._send_message(websocket, {
            "type": "connection",
            "status": "connected",
            "agent_id": agent_id,
            "message": f"Connected to agent {agent_id} stream",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info("🔌 WebSocket connected to agent stream", 
                   agent_id=agent_id, agent_connections=len(self.agent_connections[agent_id]))
    
    def disconnect(self, websocket: WebSocket):
        """Remove connection and cleanup."""
        metadata = self.connection_metadata.pop(websocket, {})
        connection_type = metadata.get("type")
        
        if connection_type == "observability":
            self.observability_connections.discard(websocket)
            logger.info("🔌 WebSocket disconnected from observability stream",
                       remaining_connections=len(self.observability_connections))
        
        elif connection_type == "agent":
            agent_id = metadata.get("agent_id")
            if agent_id and agent_id in self.agent_connections:
                self.agent_connections[agent_id].discard(websocket)
                if not self.agent_connections[agent_id]:
                    del self.agent_connections[agent_id]
                logger.info("🔌 WebSocket disconnected from agent stream",
                           agent_id=agent_id)
    
    async def broadcast_event(self, event_data: Dict[str, Any]):
        """Broadcast event to all observability connections."""
        if not self.observability_connections:
            return
        
        message = {
            "type": "event",
            "data": event_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to all connections (dead connections will be removed automatically)
        await self._broadcast_to_connections(self.observability_connections, message)
    
    async def broadcast_metric(self, metric_data: Dict[str, Any]):
        """Broadcast metric update to all observability connections."""
        if not self.observability_connections:
            return
        
        message = {
            "type": "metric",
            "data": metric_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._broadcast_to_connections(self.observability_connections, message)
    
    async def broadcast_alert(self, alert_data: Dict[str, Any]):
        """Broadcast alert to all observability connections."""
        message = {
            "type": "alert",
            "data": alert_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._broadcast_to_connections(self.observability_connections, message)
    
    async def broadcast_agent_lifecycle_event(self, event_data: Dict[str, Any]):
        """Broadcast agent lifecycle event to all connections."""
        message = {
            "type": "agent_lifecycle",
            "data": event_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Broadcast to all observability connections
        await self._broadcast_to_connections(self.observability_connections, message)
        
        # Also send to specific agent subscribers if applicable
        agent_id = event_data.get("agent_id")
        if agent_id and agent_id in self.agent_connections:
            await self._broadcast_to_connections(self.agent_connections[agent_id], message)
    
    async def broadcast_performance_update(self, performance_data: Dict[str, Any]):
        """Broadcast performance metrics update."""
        message = {
            "type": "performance_update",
            "data": performance_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._broadcast_to_connections(self.observability_connections, message)
    
    async def broadcast_hook_event(self, hook_data: Dict[str, Any]):
        """Broadcast hook execution event."""
        message = {
            "type": "hook_event",
            "data": hook_data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._broadcast_to_connections(self.observability_connections, message)
    
    async def send_to_agent_subscribers(self, agent_id: str, data: Dict[str, Any]):
        """Send data to all connections subscribed to specific agent."""
        if agent_id not in self.agent_connections:
            return
        
        message = {
            "type": "agent_event",
            "agent_id": agent_id,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self._broadcast_to_connections(self.agent_connections[agent_id], message)
    
    async def _broadcast_to_connections(self, connections, message: Dict[str, Any]):
        """Broadcast message to a set of connections."""
        if not connections:
            return
        
        disconnected = []
        
        for websocket in list(connections):  # Convert to list to avoid modification during iteration
            try:
                await self._send_message(websocket, message)
            except Exception as e:
                logger.warning("📡 Failed to send WebSocket message", error=str(e))
                disconnected.append(websocket)
        
        # Clean up disconnected connections
        for websocket in disconnected:
            self.disconnect(websocket)
    
    async def _send_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send JSON message to WebSocket connection."""
        try:
            await websocket.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.error("📡 WebSocket send failed", error=str(e))
            raise
    
    async def handle_ping(self, websocket: WebSocket, ping_data: Dict[str, Any]):
        """Handle ping message and respond with pong."""
        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]["last_ping"] = datetime.utcnow()
        
        pong_message = {
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat(),
            "original_timestamp": ping_data.get("timestamp")
        }
        
        await self._send_message(websocket, pong_message)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "observability_connections": len(self.observability_connections),
            "agent_connections": {
                agent_id: len(connections) 
                for agent_id, connections in self.agent_connections.items()
            },
            "total_connections": (
                len(self.observability_connections) + 
                sum(len(conns) for conns in self.agent_connections.values())
            )
        }

# Global connection manager
connection_manager = ConnectionManager()

class WebSocketMessage(BaseModel):
    """WebSocket message schema."""
    type: str
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None

@router.websocket("/observability")
async def observability_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time observability streaming.
    
    Streams events, metrics, alerts, and system status updates
    to connected dashboard clients.
    """
    await connection_manager.connect_observability(websocket)
    
    try:
        # Start Redis stream listener for this connection
        redis_task = None
        try:
            redis_client = get_redis()
            redis_task = asyncio.create_task(
                _redis_stream_listener(connection_manager)
            )
            
            # Handle incoming messages
            while True:
                try:
                    # Set a timeout to periodically check connection health
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    
                    try:
                        message = WebSocketMessage.model_validate_json(data)
                        await _handle_websocket_message(websocket, message)
                    except ValidationError as e:
                        logger.warning("📡 Invalid WebSocket message", error=str(e))
                        await connection_manager._send_message(websocket, {
                            "type": "error",
                            "message": "Invalid message format",
                            "error": str(e)
                        })
                        
                except asyncio.TimeoutError:
                    # Send keepalive ping
                    await connection_manager._send_message(websocket, {
                        "type": "keepalive",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
        except Exception as e:
            logger.error("📡 Observability WebSocket error", error=str(e))
        finally:
            if redis_task:
                redis_task.cancel()
                try:
                    await redis_task
                except asyncio.CancelledError:
                    pass
                    
    except WebSocketDisconnect:
        logger.info("📡 Observability WebSocket disconnected")
    finally:
        connection_manager.disconnect(websocket)

@router.websocket("/agents/{agent_id}")
async def agent_websocket(websocket: WebSocket, agent_id: str):
    """WebSocket endpoint for agent-specific real-time updates."""
    # Validate agent_id format
    try:
        uuid.UUID(agent_id)
    except ValueError:
        await websocket.close(code=1003, reason="Invalid agent ID format")
        return
    
    await connection_manager.connect_agent(websocket, agent_id)
    
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                try:
                    message = WebSocketMessage.model_validate_json(data)
                    await _handle_agent_message(websocket, agent_id, message)
                except ValidationError as e:
                    logger.warning("📡 Invalid agent WebSocket message", 
                                 agent_id=agent_id, error=str(e))
                    
            except asyncio.TimeoutError:
                # Send keepalive
                await connection_manager._send_message(websocket, {
                    "type": "keepalive",
                    "agent_id": agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info("📡 Agent WebSocket disconnected", agent_id=agent_id)
    finally:
        connection_manager.disconnect(websocket)

async def _handle_websocket_message(websocket: WebSocket, message: WebSocketMessage):
    """Handle incoming WebSocket messages from observability clients."""
    if message.type == "ping":
        await connection_manager.handle_ping(websocket, message.data or {})
    
    elif message.type == "subscribe":
        # Handle subscription requests (future enhancement)
        await connection_manager._send_message(websocket, {
            "type": "subscription",
            "status": "acknowledged",
            "subscription": message.data
        })
    
    elif message.type == "get_stats":
        # Send connection statistics
        stats = connection_manager.get_connection_stats()
        await connection_manager._send_message(websocket, {
            "type": "stats",
            "data": stats
        })
    
    else:
        logger.warning("📡 Unknown message type", message_type=message.type)

async def _handle_agent_message(websocket: WebSocket, agent_id: str, message: WebSocketMessage):
    """Handle incoming WebSocket messages from agent clients."""
    if message.type == "ping":
        await connection_manager.handle_ping(websocket, message.data or {})
    
    elif message.type == "status_update":
        # Broadcast agent status to observability connections
        await connection_manager.broadcast_event({
            "event_type": "agent_status_update",
            "agent_id": agent_id,
            "status": message.data
        })
    
    else:
        logger.info("📡 Agent message received", 
                   agent_id=agent_id, message_type=message.type)

async def _redis_stream_listener(conn_manager: ConnectionManager):
    """
    Listen to Redis streams and forward events to WebSocket connections.
    
    Enhanced for Vertical Slice 1.2 to handle:
    - Agent lifecycle events
    - Hook execution events  
    - Performance metrics
    - System health updates
    """
    try:
        redis_client = get_redis()
        
        # Multiple stream subscriptions for comprehensive monitoring
        streams = {
            "observability_events": "websocket_observability",
            "system_events:agent_lifecycle": "websocket_lifecycle", 
            "hook_events": "websocket_hooks",
            "performance_metrics": "websocket_performance"
        }
        
        # Consumer name for this WebSocket listener instance
        consumer_name = f"websocket_{uuid.uuid4().hex[:8]}"
        
        # Create consumer groups for all streams
        for stream_name, consumer_group in streams.items():
            try:
                await redis_client.xgroup_create(stream_name, consumer_group, id='0', mkstream=True)
            except Exception:
                # Group might already exist
                pass
        
        logger.info("📡 Starting enhanced Redis stream listener for WebSocket broadcasting",
                   streams=list(streams.keys()))
        
        # Also listen to pub/sub for real-time events
        pubsub_task = asyncio.create_task(
            _redis_pubsub_listener(conn_manager, redis_client)
        )
        
        while True:
            try:
                # Read from each stream individually since xreadgroup doesn't accept None for group
                all_messages = []
                for stream_name, consumer_group in streams.items():
                    try:
                        messages = await redis_client.xreadgroup(
                            consumer_group,
                            consumer_name,
                            {stream_name: '>'},
                            count=10,
                            block=100  # Short timeout for each stream
                        )
                        all_messages.extend(messages)
                    except Exception as e:
                        logger.debug(f"📡 No messages from {stream_name}: {e}")
                
                messages = all_messages
                
                for stream, stream_messages in messages:
                    consumer_group = streams.get(stream.decode() if isinstance(stream, bytes) else stream)
                    
                    for message_id, fields in stream_messages:
                        try:
                            # Decode message
                            event_data = {}
                            for key, value in fields.items():
                                key_str = key.decode() if isinstance(key, bytes) else key
                                value_str = value.decode() if isinstance(value, bytes) else value
                                event_data[key_str] = value_str
                            
                            # Parse JSON payload if present
                            if 'payload' in event_data:
                                try:
                                    event_data['payload'] = json.loads(event_data['payload'])
                                except json.JSONDecodeError:
                                    pass
                            
                            # Route message based on stream type
                            await _route_stream_message(conn_manager, stream, event_data)
                            
                            # Acknowledge message
                            if consumer_group:
                                await redis_client.xack(stream, consumer_group, message_id)
                            
                        except Exception as e:
                            logger.error("📡 Failed to process Redis stream message", 
                                       stream=stream, message_id=message_id, error=str(e))
                            
            except Exception as e:
                logger.error("📡 Redis stream listener error", error=str(e))
                await asyncio.sleep(5)  # Wait before retrying
                
    except Exception as e:
        logger.error("📡 Redis stream listener failed to start", error=str(e))
    finally:
        if 'pubsub_task' in locals():
            pubsub_task.cancel()
            try:
                await pubsub_task
            except asyncio.CancelledError:
                pass

async def _redis_pubsub_listener(conn_manager: ConnectionManager, redis_client):
    """Listen to Redis pub/sub channels for real-time events."""
    try:
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(
            "realtime:agent_lifecycle",
            "realtime:performance_metrics", 
            "realtime:system_health",
            "realtime:hook_events"
        )
        
        logger.info("📡 Started Redis pub/sub listener for real-time events")
        
        async for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    channel = message['channel'].decode()
                    data = json.loads(message['data'].decode())
                    
                    # Route pub/sub message to appropriate broadcast method
                    if channel == "realtime:agent_lifecycle":
                        await conn_manager.broadcast_agent_lifecycle_event(data)
                    elif channel == "realtime:performance_metrics":
                        await conn_manager.broadcast_performance_update(data)
                    elif channel == "realtime:hook_events":
                        await conn_manager.broadcast_hook_event(data)
                    elif channel == "realtime:system_health":
                        await conn_manager.broadcast_event(data)
                    
                except Exception as e:
                    logger.error("📡 Failed to process pub/sub message", 
                               channel=message.get('channel'), error=str(e))
    
    except Exception as e:
        logger.error("📡 Redis pub/sub listener error", error=str(e))

async def _route_stream_message(conn_manager: ConnectionManager, stream, event_data: Dict[str, Any]):
    """Route stream message to appropriate WebSocket broadcast method."""
    stream_name = stream.decode() if isinstance(stream, bytes) else stream
    
    if stream_name == "system_events:agent_lifecycle":
        await conn_manager.broadcast_agent_lifecycle_event(event_data)
    elif stream_name == "hook_events":
        await conn_manager.broadcast_hook_event(event_data)
    elif stream_name == "performance_metrics":
        await conn_manager.broadcast_performance_update(event_data)
    else:
        # Default to general event broadcasting
        await conn_manager.broadcast_event(event_data)

@router.websocket("/monitoring/agents")
async def agent_monitoring_websocket(websocket: WebSocket):
    """
    WebSocket endpoint specifically for agent lifecycle monitoring.
    
    Streams:
    - Agent registration/deregistration events
    - Task assignment and completion events
    - Agent status changes and heartbeats
    - Performance metrics
    """
    await connection_manager.connect_observability(websocket, {
        "type": "agent_monitoring",
        "subscriptions": ["agent_lifecycle", "performance_metrics", "hook_events"]
    })
    
    try:
        # Start specialized Redis listener for agent events
        agent_listener_task = asyncio.create_task(
            _agent_lifecycle_stream_listener(connection_manager, websocket)
        )
        
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                try:
                    message = WebSocketMessage.model_validate_json(data) 
                    await _handle_agent_monitoring_message(websocket, message)
                except ValidationError as e:
                    logger.warning("📡 Invalid agent monitoring message", error=str(e))
                    
            except asyncio.TimeoutError:
                # Send keepalive with agent statistics
                stats = await _get_real_time_agent_stats()
                await connection_manager._send_message(websocket, {
                    "type": "agent_stats",
                    "data": stats,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info("📡 Agent monitoring WebSocket disconnected")
    finally:
        if 'agent_listener_task' in locals():
            agent_listener_task.cancel()
            try:
                await agent_listener_task
            except asyncio.CancelledError:
                pass
        connection_manager.disconnect(websocket)

@router.websocket("/monitoring/performance")
async def performance_monitoring_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for system performance monitoring.
    
    Streams:
    - Real-time performance metrics
    - Resource utilization
    - Response time analytics
    - System health indicators
    """
    await connection_manager.connect_observability(websocket, {
        "type": "performance_monitoring",
        "subscriptions": ["performance_metrics", "system_health"]
    })
    
    try:
        # Start performance metrics collection
        perf_collector_task = asyncio.create_task(
            _performance_metrics_collector(connection_manager, websocket)
        )
        
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=15.0)
                
                try:
                    message = WebSocketMessage.model_validate_json(data)
                    await _handle_performance_monitoring_message(websocket, message)
                except ValidationError as e:
                    logger.warning("📡 Invalid performance monitoring message", error=str(e))
                    
            except asyncio.TimeoutError:
                # Send real-time performance snapshot
                metrics = await _get_current_performance_metrics()
                await connection_manager._send_message(websocket, {
                    "type": "performance_snapshot",
                    "data": metrics,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info("📡 Performance monitoring WebSocket disconnected")
    finally:
        if 'perf_collector_task' in locals():
            perf_collector_task.cancel()
            try:
                await perf_collector_task
            except asyncio.CancelledError:
                pass
        connection_manager.disconnect(websocket)

async def _agent_lifecycle_stream_listener(conn_manager: ConnectionManager, websocket: WebSocket):
    """Dedicated listener for agent lifecycle events."""
    try:
        redis_client = get_redis()
        stream_name = "system_events:agent_lifecycle"
        consumer_group = "agent_monitoring"
        consumer_name = f"agent_monitor_{uuid.uuid4().hex[:8]}"
        
        # Create consumer group
        try:
            await redis_client.xgroup_create(stream_name, consumer_group, id='0', mkstream=True)
        except Exception:
            pass
        
        logger.info("📡 Starting dedicated agent lifecycle stream listener")
        
        while True:
            try:
                messages = await redis_client.xreadgroup(
                    consumer_group,
                    consumer_name,
                    {stream_name: '>'},
                    count=20,
                    block=2000
                )
                
                for stream, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        try:
                            event_data = {
                                key.decode() if isinstance(key, bytes) else key:
                                value.decode() if isinstance(value, bytes) else value
                                for key, value in fields.items()
                            }
                            
                            if 'payload' in event_data:
                                try:
                                    event_data['payload'] = json.loads(event_data['payload'])
                                except json.JSONDecodeError:
                                    pass
                            
                            # Send directly to this WebSocket connection
                            await conn_manager._send_message(websocket, {
                                "type": "agent_lifecycle_event",
                                "data": event_data,
                                "timestamp": datetime.utcnow().isoformat()
                            })
                            
                            await redis_client.xack(stream_name, consumer_group, message_id)
                            
                        except Exception as e:
                            logger.error("📡 Failed to process agent lifecycle message", 
                                       message_id=message_id, error=str(e))
                            
            except Exception as e:
                logger.error("📡 Agent lifecycle stream listener error", error=str(e))
                await asyncio.sleep(3)
                
    except Exception as e:
        logger.error("📡 Agent lifecycle stream listener failed", error=str(e))

async def _performance_metrics_collector(conn_manager: ConnectionManager, websocket: WebSocket):
    """Collect and stream performance metrics."""
    try:
        while True:
            try:
                # Collect current performance metrics
                metrics = await _get_current_performance_metrics()
                
                # Send to WebSocket
                await conn_manager._send_message(websocket, {
                    "type": "performance_metrics",
                    "data": metrics,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Wait 5 seconds before next collection
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error("📡 Performance metrics collection error", error=str(e))
                await asyncio.sleep(10)
                
    except Exception as e:
        logger.error("📡 Performance metrics collector failed", error=str(e))

async def _handle_agent_monitoring_message(websocket: WebSocket, message: WebSocketMessage):
    """Handle agent monitoring specific messages."""
    if message.type == "get_agent_details":
        agent_id = message.data.get("agent_id") if message.data else None
        if agent_id:
            details = await _get_agent_details(agent_id)
            await connection_manager._send_message(websocket, {
                "type": "agent_details",
                "data": details,
                "timestamp": datetime.utcnow().isoformat()
            })
    
    elif message.type == "ping":
        await connection_manager.handle_ping(websocket, message.data or {})

async def _handle_performance_monitoring_message(websocket: WebSocket, message: WebSocketMessage):
    """Handle performance monitoring specific messages."""
    if message.type == "get_performance_history":
        duration = message.data.get("duration", "1h") if message.data else "1h"
        history = await _get_performance_history(duration)
        await connection_manager._send_message(websocket, {
            "type": "performance_history",
            "data": history,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    elif message.type == "ping":
        await connection_manager.handle_ping(websocket, message.data or {})

async def _get_real_time_agent_stats() -> Dict[str, Any]:
    """Get real-time agent statistics."""
    try:
        from app.core.agent_lifecycle_manager import AgentLifecycleManager
        
        # This would normally be injected, but for now we'll create an instance
        lifecycle_manager = AgentLifecycleManager()
        stats = await lifecycle_manager.get_system_metrics()
        
        return {
            "active_agents": len(lifecycle_manager.active_agents),
            "total_agents_registered": stats.get("total_agents_registered", 0),
            "tasks_in_progress": stats.get("tasks_in_progress", 0),
            "tasks_completed_today": stats.get("tasks_completed_today", 0),
            "average_task_completion_time": stats.get("average_task_completion_time", 0),
            "system_load": stats.get("system_load", 0.0)
        }
    except Exception as e:
        logger.error("Failed to get real-time agent stats", error=str(e))
        return {"error": "Failed to collect agent statistics"}

async def _get_current_performance_metrics() -> Dict[str, Any]:
    """Get current system performance metrics."""
    try:
        # Collect various performance metrics
        import psutil
        
        return {
            "cpu_usage_percent": psutil.cpu_percent(interval=1),
            "memory_usage_mb": psutil.virtual_memory().used / (1024 * 1024),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "active_connections": len(connection_manager.observability_connections),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error("Failed to get performance metrics", error=str(e))
        return {"error": "Failed to collect performance metrics"}

async def _get_agent_details(agent_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific agent."""
    try:
        from app.core.database import get_async_session
        from app.models.agent import Agent
        from sqlalchemy import select
        
        async with get_async_session() as db:
            result = await db.execute(select(Agent).where(Agent.id == agent_id))
            agent = result.scalar_one_or_none()
            
            if not agent:
                return {"error": "Agent not found"}
            
            return {
                "id": str(agent.id),
                "name": agent.name,
                "status": agent.status.value,
                "agent_type": agent.agent_type.value if agent.agent_type else "unknown",
                "created_at": agent.created_at.isoformat() if agent.created_at else None,
                "last_activity": agent.last_activity.isoformat() if agent.last_activity else None,
                "total_tasks_completed": agent.total_tasks_completed or 0,
                "current_memory_usage": getattr(agent, 'current_memory_usage', 0),
                "performance_score": getattr(agent, 'performance_score', 0.0)
            }
    except Exception as e:
        logger.error("Failed to get agent details", agent_id=agent_id, error=str(e))
        return {"error": "Failed to get agent details"}

async def _get_performance_history(duration: str) -> Dict[str, Any]:
    """Get performance metrics history for specified duration."""
    try:
        # This would normally query a time-series database
        # For now, return mock historical data
        return {
            "duration": duration,
            "metrics": {
                "cpu_usage": [
                    {"timestamp": datetime.utcnow().isoformat(), "value": 45.2},
                    {"timestamp": (datetime.utcnow()).isoformat(), "value": 42.1}
                ],
                "memory_usage": [
                    {"timestamp": datetime.utcnow().isoformat(), "value": 68.5},
                    {"timestamp": (datetime.utcnow()).isoformat(), "value": 71.2}
                ]
            }
        }
    except Exception as e:
        logger.error("Failed to get performance history", duration=duration, error=str(e))
        return {"error": "Failed to get performance history"}

# API endpoint to get WebSocket connection stats
@router.get("/websocket/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics."""
    return {
        "status": "success",
        "stats": connection_manager.get_connection_stats(),
        "timestamp": datetime.utcnow().isoformat()
    }