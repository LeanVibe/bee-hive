"""
WebSocket endpoints for real-time observability and communication.

Provides real-time streaming of events, metrics, and system status
for the observability dashboard and monitoring systems.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Set, Optional, Any
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
    
    This function runs in the background and forwards events from Redis
    to all connected WebSocket clients.
    """
    try:
        redis_client = get_redis()
        event_processor = get_event_processor()
        
        if not event_processor:
            logger.warning("📡 Event processor not available for WebSocket streaming")
            return
        
        # Subscribe to Redis stream
        stream_name = "observability_events"
        consumer_group = "websocket_broadcaster"
        consumer_name = f"websocket_{uuid.uuid4().hex[:8]}"
        
        # Create consumer group if it doesn't exist
        try:
            await redis_client.xgroup_create(stream_name, consumer_group, id='0', mkstream=True)
        except Exception:
            # Group might already exist
            pass
        
        logger.info("📡 Starting Redis stream listener for WebSocket broadcasting")
        
        while True:
            try:
                # Read from stream
                messages = await redis_client.xreadgroup(
                    consumer_group,
                    consumer_name,
                    {stream_name: '>'},
                    count=10,
                    block=1000  # 1 second timeout
                )
                
                for stream, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        try:
                            # Decode message
                            event_data = {
                                key.decode(): value.decode() if isinstance(value, bytes) else value
                                for key, value in fields.items()
                            }
                            
                            # Parse JSON payload if present
                            if 'payload' in event_data:
                                try:
                                    event_data['payload'] = json.loads(event_data['payload'])
                                except json.JSONDecodeError:
                                    pass
                            
                            # Broadcast to WebSocket connections
                            await conn_manager.broadcast_event(event_data)
                            
                            # Acknowledge message
                            await redis_client.xack(stream_name, consumer_group, message_id)
                            
                        except Exception as e:
                            logger.error("📡 Failed to process Redis stream message", 
                                       message_id=message_id, error=str(e))
                            
            except Exception as e:
                logger.error("📡 Redis stream listener error", error=str(e))
                await asyncio.sleep(5)  # Wait before retrying
                
    except Exception as e:
        logger.error("📡 Redis stream listener failed to start", error=str(e))

# API endpoint to get WebSocket connection stats
@router.get("/websocket/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics."""
    return {
        "status": "success",
        "stats": connection_manager.get_connection_stats(),
        "timestamp": datetime.utcnow().isoformat()
    }